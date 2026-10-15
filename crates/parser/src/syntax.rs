//! A typed view of what a Java file declares.
//!
//! Tree-sitter hands back a tree in which every construct is a `Node` carrying a
//! string kind and string field names. Without this module every consumer that
//! wants "the methods of this class" has to re-learn the grammar, and the
//! grammar then leaks out of this crate. So the node and field names are learned
//! once, here, and the tree is projected onto declarations that have a name, two
//! ranges and source-order nesting.
//!
//! It is deliberately *syntax only*. A type is the text the author wrote — never
//! a resolved type, never an inherited member, never a followed import.
//! Resolution belongs to `javelin-semantic`, which does not exist yet.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use javelin_document::{Range, TextDocument};
use tree_sitter::Node;

use crate::SyntaxTree;
use crate::expr::{Expr, project};

/// How deep the walk will follow nesting before giving up on a subtree.
///
/// ponytail: a fixed cap rather than an explicit work list. Declarations nest a
/// handful of levels in real code and expressions rarely more, but generated
/// sources do contain thousand-term concatenations, and a stack overflow is not
/// a failure mode this crate is allowed to have. Raise it or go iterative if a
/// real file is ever truncated by it.
pub(crate) const MAX_DEPTH: u32 = 128;

/// Where a piece of syntax is, in both of the units the editor path needs:
/// bytes, for slicing the source, and an LSP range, for the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start_byte: u32,
    pub end_byte: u32,
    pub range: Range,
}

/// What a [`Decl`] declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeclKind {
    Class,
    Interface,
    Enum,
    Record,
    /// `@interface Foo {}`, not a use of an annotation.
    AnnotationType,
    /// A class body attached to a `new` expression.
    AnonymousClass,
    Field,
    Method,
    Constructor,
    StaticInitializer,
    InstanceInitializer,
    EnumConstant,
    /// A component of a record header, e.g. `x` in `record Point(int x) {}`.
    RecordComponent,
    /// A member of an annotation type, e.g. `value()` in `@interface Foo`.
    AnnotationElement,
    /// A formal parameter of a method or constructor.
    Parameter,
    LocalVariable,
}

/// One declaration, and everything it declares in turn.
///
/// Every string here is source text exactly as the author wrote it. Nothing is
/// normalised, expanded or resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl {
    pub kind: DeclKind,
    /// Empty for the constructs that genuinely have no name: initialiser blocks.
    pub name: String,
    /// The whole declaration, from its first modifier to its closing brace.
    pub span: Span,
    /// Just the name, which is where an editor puts the selection.
    pub name_span: Span,
    /// The `/** … */` immediately preceding, verbatim and unparsed.
    pub doc: Option<String>,
    /// Its byte offsets, for a reader that stores where it is rather than
    /// what it says.
    pub doc_span: Option<(u32, u32)>,
    /// Modifier keywords and annotations, in source order, as written.
    pub modifiers: Vec<String>,
    /// As written and including the angle brackets, e.g. `<T extends Number>`.
    pub type_parameters: Option<String>,
    /// The superclass as written, e.g. `Base<T>`. `None` for an interface,
    /// which has no single superclass, and for anything that is not a type.
    pub extends: Option<String>,
    /// Interfaces as written: what a class or enum implements, and everything
    /// an interface extends. Both are supertypes reached the same way, so
    /// splitting them by keyword would buy nothing.
    pub implements: Vec<String>,
    /// The declared type as written: a field's or local's type, a method's
    /// return type, a parameter's type. `None` where the construct has none.
    pub declared_type: Option<String>,
    /// Formal parameters of a method or constructor. Empty for everything else;
    /// record components are children, because they are also fields.
    pub parameters: Vec<Decl>,
    /// The thrown types as written, one per entry.
    pub throws: Vec<String>,
    /// Only ever true for a [`DeclKind::Parameter`]. The `...` is not part of
    /// [`declared_type`](Self::declared_type).
    pub varargs: bool,
    /// Members, nested types and — inside a body — locals and anonymous classes.
    pub children: Vec<Decl>,
    /// The projected initialiser of a `var` local, and only of a `var` local:
    /// for a for-each's, an [`Expr::Element`] of what it iterates.
    ///
    /// `None` everywhere else, including for a local with a written type: that
    /// local's type is its `declared_type` and projecting an expression per
    /// declaration would be file-sized work for an answer nothing asks for.
    ///
    /// Boxed, and it has to stay boxed: an inline `Expr` is 64 bytes on every
    /// `Decl` to carry what is `None` on all but a handful, which at
    /// `Tier::Huge` is megabytes per `FileSyntax` — rebuilt whole per document
    /// version, with the previous one still alive. The box is 8 bytes and
    /// costs one allocation per `var` local.
    pub initializer: Option<Box<Expr>>,
    /// Whether a field or a local is written with `= …`, which is what
    /// decides whether Lombok's `@Builder` sets a `final` field.
    pub has_initializer: bool,
}

/// Lombok annotations that may write a constructor into the type they sit on.
const LOMBOK_CONSTRUCTORS: [&str; 7] = [
    "AllArgsConstructor",
    "RequiredArgsConstructor",
    "NoArgsConstructor",
    "Data",
    "Value",
    "Builder",
    "SuperBuilder",
];

impl Decl {
    /// Whether this type's written constructors are all it has: a class or
    /// an enum that writes one at least and carries no annotation by which
    /// Lombok may write another, read by simple name whatever the file
    /// imports. A record and an interface are never: a record's canonical
    /// constructor may be implicit beside a written one.
    pub fn writes_its_constructors(&self) -> bool {
        matches!(self.kind, DeclKind::Class | DeclKind::Enum)
            && self
                .children
                .iter()
                .any(|child| child.kind == DeclKind::Constructor)
            && !self.modifiers.iter().any(|modifier| {
                modifier.strip_prefix('@').is_some_and(|annotation| {
                    let name = annotation.split('(').next().unwrap_or(annotation).trim();
                    LOMBOK_CONSTRUCTORS.contains(&name.rsplit('.').next().unwrap_or(name))
                })
            })
    }
}

/// A lambda's parameter: one the scope tree binds and no [`Decl`] records,
/// with what typing it needs — its written type if it has one, and otherwise
/// the call whose argument the lambda is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LambdaParameter {
    pub name_span: Span,
    /// `(String s) -> …`'s `String`; `None` for an inferred parameter.
    pub declared: Option<String>,
    /// The method invocation the lambda is an argument of, projected once
    /// and shared by every lambda parameter among its arguments. `None` for a
    /// lambda anywhere else — an initialiser, a constructor's argument, a
    /// return.
    pub call: Option<Arc<Expr>>,
    /// Which of the call's arguments the lambda is.
    pub argument: usize,
    /// Which of the lambda's parameters this is.
    pub position: usize,
    /// How many parameters the lambda takes.
    pub arity: usize,
}

/// A type a file declares, and the fully-qualified name it is known by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedType<'a> {
    pub decl: &'a Decl,
    pub fqn: String,
}

/// `package com.example;`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageDecl {
    pub name: String,
    pub span: Span,
    pub name_span: Span,
}

/// `import java.util.List;`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportDecl {
    /// The name as written, without the trailing `.*` of an on-demand import.
    pub name: String,
    pub span: Span,
    pub name_span: Span,
    pub is_static: bool,
    /// `import java.util.*;` — an on-demand import.
    pub is_on_demand: bool,
}

/// Everything one file declares, tied to the document version it describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSyntax {
    pub version: i32,
    pub package: Option<PackageDecl>,
    pub imports: Vec<ImportDecl>,
    /// Top-level declarations, in source order.
    pub declarations: Vec<Decl>,
    /// Sorted, deduplicated hashes of every simple name this file declares a
    /// type under — see [`declares_type_named`](Self::declares_type_named).
    ///
    /// A `Vec<u64>` rather than a `HashSet<String>` because a `FileSyntax` is
    /// rebuilt whole for every document version with the previous one still
    /// alive: at 40 001 declarations a `HashSet<String>` would repeat names
    /// the `Decl`s already own, roughly 1.5 MB against this field's 320 KB.
    declared_type_names: Vec<u64>,
    /// Every lambda parameter, by name offset.
    lambda_parameters: Vec<LambdaParameter>,
    /// Whether every list of declarations here is in source order and nests
    /// inside the declaration that owns it — checked once, when the view is
    /// built, and what licenses [`declaration_at`](Self::declaration_at)'s
    /// descent. `false` falls back to searching every declaration.
    ordered: bool,
}

impl FileSyntax {
    /// Project a parsed tree onto declarations.
    ///
    /// Never fails. A file mid-keystroke is full of `ERROR` and `MISSING` nodes,
    /// and the answer to one is to keep whatever else parsed rather than to
    /// refuse the file: the user is looking at that buffer right now.
    pub fn build(document: &TextDocument, tree: &SyntaxTree) -> Self {
        let builder = Builder {
            document,
            source: document.text(),
            type_names: RefCell::new(Vec::new()),
            lambdas: RefCell::new(Vec::new()),
            lambda_sites: RefCell::new(HashMap::new()),
        };
        let mut view = FileSyntax {
            version: tree.version(),
            package: None,
            imports: Vec::new(),
            declarations: Vec::new(),
            declared_type_names: Vec::new(),
            lambda_parameters: Vec::new(),
            ordered: false,
        };
        builder.file(tree.root(), &mut view, 0);
        let mut lambdas = builder.lambdas.into_inner();
        lambdas.sort_by_key(|parameter| parameter.name_span.start_byte);
        view.lambda_parameters = lambdas;
        let mut names = builder.type_names.into_inner();
        names.sort_unstable();
        names.dedup();
        view.declared_type_names = names;
        view.ordered = in_order(&view.declarations, None);
        view
    }

    /// Whether this file declares a type of this simple name, at any nesting
    /// depth — including a nested type of a type it declares, which
    /// [`ScopeTree`](crate::ScopeTree) binds only inside its own enclosing
    /// type's scope.
    ///
    /// Answers about *this file* only: a supertype outside it, an import, and
    /// the classpath are all someone else's question.
    ///
    /// ponytail: a hash collision answers `true` for a name this file does not
    /// declare. Every caller of this method uses it to *veto* an answer built
    /// from elsewhere (a classpath lookup), so a collision costs a missing
    /// answer, never a wrong one — the safe direction. The exact upgrade is a
    /// `HashSet<String>`, traded back for the memory it costs; see
    /// `declared_type_names`.
    pub fn declares_type_named(&self, simple: &str) -> bool {
        self.declared_type_names
            .binary_search(&hash_name(simple))
            .is_ok()
    }

    /// The lambda parameter whose name starts at `start_byte`.
    pub fn lambda_parameter(&self, start_byte: u32) -> Option<&LambdaParameter> {
        let at = self
            .lambda_parameters
            .binary_search_by_key(&start_byte, |parameter| parameter.name_span.start_byte)
            .ok()?;
        self.lambda_parameters.get(at)
    }

    /// Every lambda parameter whose name starts inside `within`, in source
    /// order.
    pub fn lambda_parameters_in(&self, within: std::ops::Range<u32>) -> &[LambdaParameter] {
        let start = self
            .lambda_parameters
            .partition_point(|parameter| parameter.name_span.start_byte < within.start);
        let end = self
            .lambda_parameters
            .partition_point(|parameter| parameter.name_span.start_byte < within.end);
        &self.lambda_parameters[start..end.max(start)]
    }

    /// Every type this file declares that has a name anything outside the file
    /// can reach it by, each with its fully-qualified name.
    ///
    /// Nested types join on `.` — `com.example.Outer.Inner`, never `$`, the
    /// convention `javelin-classpath` records for the JDK's own nested types —
    /// and the walk descends into a type's children only, so a local class in
    /// a method body and an anonymous class are not here: neither has a name
    /// another file could write.
    ///
    /// One place so the workspace index and a query at the cursor cannot
    /// compute two different names for one declaration.
    pub fn named_types(&self) -> Vec<NamedType<'_>> {
        fn collect<'a>(decls: &'a [Decl], prefix: &str, out: &mut Vec<NamedType<'a>>) {
            for decl in decls.iter().filter(|decl| declares_a_type(decl.kind)) {
                let fqn = if prefix.is_empty() {
                    decl.name.clone()
                } else {
                    format!("{prefix}.{}", decl.name)
                };
                out.push(NamedType {
                    decl,
                    fqn: fqn.clone(),
                });
                collect(&decl.children, &fqn, out);
            }
        }
        let mut out = Vec::new();
        let package = self
            .package
            .as_ref()
            .map_or("", |package| package.name.as_str());
        collect(&self.declarations, package, &mut out);
        out
    }

    /// Whether `offset` is inside the body of an anonymous class, or of a
    /// class declared inside a method, a constructor, an initialiser or a
    /// field — where the member types its supertypes declare are in scope,
    /// which no closure the server walks reaches.
    pub fn in_local_or_anonymous_class(&self, offset: u32) -> bool {
        fn visit(decls: &[Decl], inside_code: bool, offset: u32) -> bool {
            decls.iter().any(|decl| {
                if offset < decl.span.start_byte || decl.span.end_byte <= offset {
                    return false;
                }
                let is_type = matches!(
                    decl.kind,
                    DeclKind::Class
                        | DeclKind::Interface
                        | DeclKind::Enum
                        | DeclKind::Record
                        | DeclKind::AnnotationType
                );
                decl.kind == DeclKind::AnonymousClass
                    || (is_type && inside_code)
                    || visit(&decl.children, !is_type, offset)
            })
        }
        visit(&self.declarations, false, offset)
    }

    /// Whether every list of declarations is in source order and nests in
    /// its owner, so that a search by offset may bisect it.
    pub fn in_source_order(&self) -> bool {
        self.ordered
    }

    /// The innermost class, interface, enum, record or annotation type whose
    /// declaration holds `offset`, bisecting each list where it is in order.
    pub fn innermost_type_at(&self, offset: u32) -> Option<&Decl> {
        let mut decls = &self.declarations[..];
        let mut found = None;
        loop {
            let holds =
                |decl: &&Decl| decl.span.start_byte <= offset && offset <= decl.span.end_byte;
            let decl = if self.ordered {
                decls[..decls.partition_point(|decl| decl.span.start_byte <= offset)]
                    .last()
                    .filter(holds)
            } else {
                decls.iter().rfind(holds)
            };
            let Some(decl) = decl else {
                return found;
            };
            if matches!(
                decl.kind,
                DeclKind::Class
                    | DeclKind::Interface
                    | DeclKind::Enum
                    | DeclKind::Record
                    | DeclKind::AnnotationType
            ) {
                found = Some(decl);
            }
            decls = &decl.children;
        }
    }

    /// The declaration whose own name occupies `name_span`.
    ///
    /// Name spans are unique within a file, so this is how a binding, a scope
    /// entry or a resolved type gets back to the declaration it came from.
    pub fn declaration_at(&self, name_span: Span) -> Option<&Decl> {
        fn linear(decls: &[Decl], name_span: Span) -> Option<&Decl> {
            decls.iter().find_map(|decl| {
                if names(decl, name_span) {
                    return Some(decl);
                }
                linear(&decl.parameters, name_span).or_else(|| linear(&decl.children, name_span))
            })
        }
        // Only the declarations whose span holds the name can be the one that
        // owns it, and in a list in source order those are a run ending at
        // the last one starting at or before it: an earlier sibling that ends
        // before the name has every sibling before it ending earlier still.
        // `int a, b;` gives two declarations one span, so the run can be
        // longer than one.
        fn descend(decls: &[Decl], name_span: Span) -> Option<&Decl> {
            let at = name_span.start_byte;
            let after = decls.partition_point(|decl| decl.span.start_byte <= at);
            for decl in decls[..after].iter().rev() {
                if decl.span.end_byte < at {
                    break;
                }
                if names(decl, name_span) {
                    return Some(decl);
                }
                if let Some(found) = descend(&decl.parameters, name_span)
                    .or_else(|| descend(&decl.children, name_span))
                {
                    return Some(found);
                }
            }
            None
        }
        if self.ordered {
            descend(&self.declarations, name_span)
        } else {
            linear(&self.declarations, name_span)
        }
    }
}

/// Whether `decl` is the declaration whose own name is `name_span`.
fn names(decl: &Decl, name_span: Span) -> bool {
    !decl.name.is_empty()
        && decl.name_span.start_byte == name_span.start_byte
        && decl.name_span.end_byte == name_span.end_byte
}

/// Whether every list under `decls` is in source order, with siblings apart or
/// sharing one span, and every declaration's name, parameters and children
/// inside its own span — what [`FileSyntax::declaration_at`]'s descent needs
/// to be exact. Checked rather than assumed: error recovery hoists
/// declarations about, and a descent over a list that broke the rule would
/// miss a declaration rather than fail loudly.
fn in_order(decls: &[Decl], outer: Option<Span>) -> bool {
    let inside = |span: Span| {
        outer.is_none_or(|outer| {
            outer.start_byte <= span.start_byte && span.end_byte <= outer.end_byte
        })
    };
    decls.windows(2).all(|pair| {
        let (a, b) = (pair[0].span, pair[1].span);
        a.end_byte <= b.start_byte || (a.start_byte == b.start_byte && a.end_byte == b.end_byte)
    }) && decls.iter().all(|decl| {
        inside(decl.span)
            && (decl.name.is_empty()
                || (decl.span.start_byte <= decl.name_span.start_byte
                    && decl.name_span.end_byte <= decl.span.end_byte))
            && in_order(&decl.parameters, Some(decl.span))
            && in_order(&decl.children, Some(decl.span))
    })
}

/// A simple name's hash, for [`FileSyntax::declared_type_names`].
///
/// `DefaultHasher::new()` starts from fixed keys, so this is deterministic
/// within one run of the process — the only guarantee a build-then-look-up
/// pair needs, since a `FileSyntax` never crosses a process boundary.
fn hash_name(name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

/// Whether a declaration kind names a type this file *declares*.
///
/// Not [`DeclKind::AnonymousClass`]: `anonymous_class` records its `name` as
/// the type it instantiates, never as an `extends`, so `new Runnable() {}`
/// declares nothing named `Runnable` — the file only refers to that name, and
/// treating the reference as a declaration would make a file that instantiates
/// an anonymous JDK interface lose that interface's name to the classpath
/// everywhere else in the file too.
pub(crate) fn declares_a_type(kind: DeclKind) -> bool {
    matches!(
        kind,
        DeclKind::Class
            | DeclKind::Interface
            | DeclKind::Enum
            | DeclKind::Record
            | DeclKind::AnnotationType
    )
}

pub(crate) fn named_child<'t>(node: Node<'t>, kind: &str) -> Option<Node<'t>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .find(|child| child.kind() == kind)
}

#[derive(Debug)]
struct Builder<'a> {
    document: &'a TextDocument,
    source: &'a str,
    /// Collected alongside `declarations` rather than by a second walk over
    /// it afterwards — every `Decl` passes through `base`, so that is the one
    /// place this needs to hook into. `RefCell` because every `Builder`
    /// method takes `&self`: the tree is walked recursively by ordinary calls,
    /// not by threading a mutable accumulator through every signature.
    type_names: RefCell<Vec<u64>>,
    /// Collected beside the declarations for the same reason, from the one
    /// walk that already visits every statement.
    lambdas: RefCell<Vec<LambdaParameter>>,
    /// Each lambda a method invocation takes, by node id: the call, projected
    /// once, and which argument the lambda is. Filled when the walk reaches
    /// the call, before it descends to the lambda.
    lambda_sites: RefCell<HashMap<usize, (Arc<Expr>, usize)>>,
}

impl Builder<'_> {
    fn file(&self, node: Node<'_>, view: &mut FileSyntax, depth: u32) {
        if depth > MAX_DEPTH {
            return;
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "package_declaration" if view.package.is_none() => {
                    view.package = self.package(child);
                }
                "import_declaration" => {
                    if let Some(import) = self.import(child) {
                        view.imports.push(import);
                    }
                }
                // Recovery can wrap a whole file's worth of declarations in one
                // ERROR node; the declarations inside it still parsed.
                "ERROR" => self.file(child, view, depth + 1),
                // A block at file level is a statement, not an initialiser.
                "block" => {}
                _ => self.member(child, &mut view.declarations, depth),
            }
        }
    }

    fn package(&self, node: Node<'_>) -> Option<PackageDecl> {
        let name = self.qualified_name(node)?;
        Some(PackageDecl {
            name: self.text(name).to_string(),
            span: self.span(node),
            name_span: self.span(name),
        })
    }

    fn import(&self, node: Node<'_>) -> Option<ImportDecl> {
        let name = self.qualified_name(node)?;
        let mut is_static = false;
        let mut is_on_demand = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "static" => is_static = true,
                "asterisk" => is_on_demand = true,
                _ => {}
            }
        }
        Some(ImportDecl {
            name: self.text(name).to_string(),
            span: self.span(node),
            name_span: self.span(name),
            is_static,
            is_on_demand,
        })
    }

    /// The `identifier` or `scoped_identifier` of a package or import, ignoring
    /// any annotations that precede it.
    fn qualified_name<'t>(&self, node: Node<'t>) -> Option<Node<'t>> {
        let mut cursor = node.walk();
        node.named_children(&mut cursor)
            .find(|child| matches!(child.kind(), "identifier" | "scoped_identifier"))
    }

    /// Every named child of a class, interface, enum or annotation body.
    fn members(&self, body: Node<'_>, out: &mut Vec<Decl>, depth: u32) {
        if depth > MAX_DEPTH {
            return;
        }
        let mut cursor = body.walk();
        for child in body.named_children(&mut cursor) {
            self.member(child, out, depth + 1);
        }
    }

    fn member(&self, node: Node<'_>, out: &mut Vec<Decl>, depth: u32) {
        match node.kind() {
            "class_declaration" => out.push(self.type_decl(DeclKind::Class, node, depth)),
            "interface_declaration" => out.push(self.type_decl(DeclKind::Interface, node, depth)),
            "enum_declaration" => out.push(self.type_decl(DeclKind::Enum, node, depth)),
            "record_declaration" => out.push(self.type_decl(DeclKind::Record, node, depth)),
            "annotation_type_declaration" => {
                out.push(self.type_decl(DeclKind::AnnotationType, node, depth));
            }
            // `constant_declaration` is how the grammar spells a field of an
            // interface or annotation type.
            "field_declaration" | "constant_declaration" => {
                self.variables(DeclKind::Field, node, out, depth);
            }
            "method_declaration" => out.push(self.callable(DeclKind::Method, node, depth)),
            "constructor_declaration" | "compact_constructor_declaration" => {
                out.push(self.callable(DeclKind::Constructor, node, depth));
            }
            "annotation_type_element_declaration" => out.push(self.annotation_element(node)),
            "enum_constant" => out.push(self.enum_constant(node, depth)),
            "static_initializer" => {
                out.push(self.initializer(DeclKind::StaticInitializer, node, depth));
            }
            "block" => out.push(self.initializer(DeclKind::InstanceInitializer, node, depth)),
            // Neither is a declaration; both hold declarations one level down.
            "enum_body_declarations" | "ERROR" => self.members(node, out, depth),
            _ => {}
        }
    }

    fn type_decl(&self, kind: DeclKind, node: Node<'_>, depth: u32) -> Decl {
        let mut decl = self.base(kind, node, node.child_by_field_name("name"));
        decl.type_parameters = self.field_text(node, "type_parameters");
        decl.extends = node
            .child_by_field_name("superclass")
            .and_then(|clause| clause.named_child(0))
            .map(|ty| self.text(ty).to_string());
        decl.implements = self.implemented(node);
        // Only a record has these, and they are members as much as parameters.
        if let Some(header) = node.child_by_field_name("parameters") {
            decl.children = self.parameters(header, DeclKind::RecordComponent);
        }
        if let Some(body) = node.child_by_field_name("body") {
            self.members(body, &mut decl.children, depth + 1);
        }
        decl
    }

    fn callable(&self, kind: DeclKind, node: Node<'_>, depth: u32) -> Decl {
        let mut decl = self.base(kind, node, node.child_by_field_name("name"));
        decl.type_parameters = self.field_text(node, "type_parameters");
        // `int f()[]` returns an `int[]`.
        decl.declared_type = self.with_dimensions(self.field_text(node, "type"), node);
        decl.throws = self.throws(node);
        if let Some(parameters) = node.child_by_field_name("parameters") {
            decl.parameters = self.parameters(parameters, DeclKind::Parameter);
        }
        if let Some(body) = node.child_by_field_name("body") {
            self.statements(body, &mut decl.children, depth + 1);
        }
        decl
    }

    fn parameters(&self, list: Node<'_>, kind: DeclKind) -> Vec<Decl> {
        let mut out = Vec::new();
        let mut cursor = list.walk();
        for child in list.named_children(&mut cursor) {
            match child.kind() {
                "formal_parameter" => {
                    let mut decl = self.base(kind, child, child.child_by_field_name("name"));
                    decl.declared_type =
                        self.with_dimensions(self.field_text(child, "type"), child);
                    out.push(decl);
                }
                // A varargs parameter has no `name` field: the grammar wraps the
                // name in a declarator and leaves the type an unlabelled child.
                "spread_parameter" => {
                    let mut inner = child.walk();
                    let children: Vec<_> = child.named_children(&mut inner).collect();
                    let name = children
                        .iter()
                        .find(|c| c.kind() == "variable_declarator")
                        .and_then(|c| c.child_by_field_name("name"));
                    let mut decl = self.base(kind, child, name);
                    decl.declared_type = children
                        .iter()
                        .find(|c| {
                            !matches!(
                                c.kind(),
                                "variable_declarator"
                                    | "modifiers"
                                    | "annotation"
                                    | "marker_annotation"
                            )
                        })
                        .map(|c| self.text(*c).to_string());
                    decl.varargs = true;
                    out.push(decl);
                }
                // `receiver_parameter` (`Foo this`) binds no new name.
                _ => {}
            }
        }
        out
    }

    fn annotation_element(&self, node: Node<'_>) -> Decl {
        let mut decl = self.base(
            DeclKind::AnnotationElement,
            node,
            node.child_by_field_name("name"),
        );
        decl.declared_type = self.field_text(node, "type");
        decl
    }

    fn enum_constant(&self, node: Node<'_>, depth: u32) -> Decl {
        let mut decl = self.base(
            DeclKind::EnumConstant,
            node,
            node.child_by_field_name("name"),
        );
        if let Some(body) = node.child_by_field_name("body") {
            self.members(body, &mut decl.children, depth + 1);
        }
        decl
    }

    fn initializer(&self, kind: DeclKind, node: Node<'_>, depth: u32) -> Decl {
        let mut decl = self.base(kind, node, None);
        self.statements(node, &mut decl.children, depth + 1);
        decl
    }

    /// `ty` with the C-style `[]`s written after a name — `String a[]` is a
    /// `String[]` — so a reader of `declared_type` never sees the element type.
    fn with_dimensions(&self, ty: Option<String>, node: Node<'_>) -> Option<String> {
        let Some(dimensions) = node.child_by_field_name("dimensions") else {
            return ty;
        };
        let rank = self.text(dimensions).matches('[').count();
        ty.map(|ty| format!("{ty}{}", "[]".repeat(rank)))
    }

    /// One `Decl` per declarator, so `int a, b;` yields two.
    fn variables(&self, kind: DeclKind, node: Node<'_>, out: &mut Vec<Decl>, depth: u32) {
        let declared_type = self.field_text(node, "type");
        let doc = self.doc(node);
        let doc_span = self.doc_span(node);
        let modifiers = self.modifiers(node);
        let mut cursor = node.walk();
        let declarators: Vec<_> = node
            .named_children(&mut cursor)
            .filter(|child| child.kind() == "variable_declarator")
            .collect();

        for declarator in &declarators {
            // Sibling ranges must not overlap, so only a lone declarator gets to
            // claim the modifiers, the type and the semicolon around it.
            let whole = if declarators.len() == 1 {
                node
            } else {
                *declarator
            };
            let mut decl = self.base(kind, whole, declarator.child_by_field_name("name"));
            decl.doc = doc.clone();
            decl.doc_span = doc_span;
            decl.modifiers = modifiers.clone();
            decl.declared_type = self.with_dimensions(declared_type.clone(), *declarator);
            // An initialiser can hold an anonymous class, which nests inside the
            // variable rather than beside it.
            if let Some(value) = declarator.child_by_field_name("value") {
                decl.has_initializer = true;
                self.statement(value, &mut decl.children, depth + 1);
                // Only a `var` local: the type of every other declaration is
                // already written down, and `FileSyntax` is rebuilt whole for
                // every document version, so projecting an expression per
                // declarator would add file-sized work to the path that
                // already costs completion its p95 on a generated god-class.
                if decl.declared_type.as_deref() == Some("var") {
                    decl.initializer = Some(Box::new(project(self.document, value, 0)));
                }
            }
            out.push(decl);
        }
    }

    /// Walk a body for the declarations a statement can introduce: locals, local
    /// types and anonymous classes.
    ///
    /// ponytail: everything found lands in one flat list per enclosing member,
    /// so a local declared inside a lambda or an `if` is reported at method
    /// level. The outline wants exactly that; `ScopeTree`, which does need block
    /// structure, has its own walk in `scopes.rs`. Nest them by block only if
    /// a reader other than the outline must know which block a local is in —
    /// until then the scope tree is the one to ask.
    fn statements(&self, node: Node<'_>, out: &mut Vec<Decl>, depth: u32) {
        if depth > MAX_DEPTH {
            return;
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            self.statement(child, out, depth + 1);
        }
    }

    fn statement(&self, node: Node<'_>, out: &mut Vec<Decl>, depth: u32) {
        match node.kind() {
            "local_variable_declaration" => {
                self.variables(DeclKind::LocalVariable, node, out, depth);
            }
            "class_declaration"
            | "interface_declaration"
            | "enum_declaration"
            | "record_declaration"
            | "annotation_type_declaration" => {
                self.member(node, out, depth);
            }
            // A for-each variable, a catch parameter, a try resource and a
            // pattern binding are locals too; the grammar simply spells each of
            // them its own way. Where no single node covers the declaration, the
            // name stands in for it — `range == selectionRange` is allowed.
            "enhanced_for_statement" => {
                if let Some(name) = node.child_by_field_name("name") {
                    let declared = self.field_text(node, "type");
                    let mut decl = self.local(name, name, declared);
                    if decl.declared_type.as_deref() == Some("var")
                        && let Some(value) = node.child_by_field_name("value")
                    {
                        decl.initializer = Some(Box::new(Expr::Element {
                            iterable: Box::new(project(self.document, value, 0)),
                            span: decl.name_span,
                        }));
                    }
                    out.push(decl);
                }
                self.statements(node, out, depth);
            }
            "catch_formal_parameter" => {
                if let Some(name) = node.child_by_field_name("name") {
                    let declared =
                        named_child(node, "catch_type").map(|ty| self.text(ty).to_string());
                    out.push(self.local(node, name, declared));
                }
            }
            "resource" => match node.child_by_field_name("name") {
                Some(name) => {
                    let declared = self.field_text(node, "type");
                    out.push(self.local(node, name, declared));
                }
                // `try (existing)` names a variable rather than declaring one.
                None => self.statements(node, out, depth),
            },
            "instanceof_expression" => {
                if let Some(name) = node.child_by_field_name("name") {
                    let declared = self.field_text(node, "right");
                    out.push(self.local(name, name, declared));
                }
                self.statements(node, out, depth);
            }
            "type_pattern" => {
                if let Some(name) = named_child(node, "identifier") {
                    let mut cursor = node.walk();
                    let declared = node
                        .named_children(&mut cursor)
                        .find(|child| child.kind() != "identifier")
                        .map(|ty| self.text(ty).to_string());
                    out.push(self.local(name, name, declared));
                }
            }
            "lambda_expression" => {
                self.lambda(node);
                self.statements(node, out, depth);
            }
            "method_invocation" => {
                self.lambda_arguments(node);
                self.statements(node, out, depth);
            }
            // The grammar leaves the body of an anonymous class an unlabelled
            // child rather than giving it a field.
            "object_creation_expression" => match named_child(node, "class_body") {
                Some(body) => out.push(self.anonymous_class(node, body, depth)),
                // No body: not anonymous, but the arguments may still hold one.
                None => self.statements(node, out, depth),
            },
            _ => self.statements(node, out, depth),
        }
    }

    /// Note which of a call's arguments are lambdas with parameters, and
    /// project the call once for all of them — only if there is one. A lambda
    /// among the arguments projects its body's results, at most
    /// `MAX_RETURNS` of a block's, so this is bounded by the call and the
    /// statements of its lambdas' blocks.
    fn lambda_arguments(&self, call: Node<'_>) {
        let Some(arguments) = call.child_by_field_name("arguments") else {
            return;
        };
        let mut projected: Option<Arc<Expr>> = None;
        let mut cursor = arguments.walk();
        for (index, argument) in arguments
            .named_children(&mut cursor)
            .filter(|argument| !argument.is_extra())
            .enumerate()
        {
            let takes_parameters = argument.kind() == "lambda_expression"
                && argument
                    .child_by_field_name("parameters")
                    .is_some_and(|parameters| {
                        parameters.kind() == "identifier" || parameters.named_child_count() > 0
                    });
            if !takes_parameters {
                continue;
            }
            let call = projected
                .get_or_insert_with(|| Arc::new(project(self.document, call, 0)))
                .clone();
            self.lambda_sites
                .borrow_mut()
                .insert(argument.id(), (call, index));
        }
    }

    /// Record a lambda's parameters, and the call it is an argument of — as
    /// [`Self::lambda_arguments`] found it.
    fn lambda(&self, node: Node<'_>) {
        let Some(parameters) = node.child_by_field_name("parameters") else {
            return;
        };
        let mut cursor = parameters.walk();
        let names: Vec<(Node<'_>, Option<String>)> = match parameters.kind() {
            "identifier" => vec![(parameters, None)],
            "inferred_parameters" => parameters
                .named_children(&mut cursor)
                .filter(|child| child.kind() == "identifier")
                .map(|name| (name, None))
                .collect(),
            "formal_parameters" => {
                let mut names = Vec::new();
                for child in parameters.named_children(&mut cursor) {
                    match (child.kind(), child.child_by_field_name("name")) {
                        ("formal_parameter", Some(name)) => {
                            names.push((name, self.field_text(child, "type")));
                        }
                        // A `...` parameter or recovery: the positions after
                        // it are not what they look like.
                        _ => return,
                    }
                }
                names
            }
            _ => return,
        };
        if names.is_empty() {
            return;
        }
        let site = self.lambda_sites.borrow().get(&node.id()).cloned();
        let argument = site.as_ref().map_or(0, |(_, argument)| *argument);
        let call = site.map(|(call, _)| call);
        let arity = names.len();
        let mut lambdas = self.lambdas.borrow_mut();
        for (position, (name, declared)) in names.into_iter().enumerate() {
            lambdas.push(LambdaParameter {
                name_span: self.span(name),
                declared,
                call: call.clone(),
                argument,
                position,
                arity,
            });
        }
    }

    fn anonymous_class(&self, node: Node<'_>, body: Node<'_>, depth: u32) -> Decl {
        // The name is the type being instantiated: `new Runnable() {}` is the
        // closest thing to a name such a class has.
        let mut decl = self.base(
            DeclKind::AnonymousClass,
            node,
            node.child_by_field_name("type"),
        );
        self.members(body, &mut decl.children, depth + 1);
        // Arguments to the constructor can hold further anonymous classes, but
        // they belong beside the class body, not inside it.
        if let Some(arguments) = node.child_by_field_name("arguments") {
            self.statements(arguments, &mut decl.children, depth + 1);
        }
        decl
    }

    /// A local the grammar spells its own way: a for-each variable, a catch
    /// parameter, a try resource, a pattern binding.
    ///
    /// None of them carries an `initializer` here, so a `var` is unknown — but
    /// the for-each arm gives its `var` one, an [`Expr::Element`]. A resource's
    /// `var` would need `AutoCloseable`'s, and a pattern's has no `var`.
    fn local(&self, anchor: Node<'_>, name: Node<'_>, declared: Option<String>) -> Decl {
        let mut decl = self.base(DeclKind::LocalVariable, anchor, Some(name));
        decl.declared_type = declared;
        decl
    }

    fn base(&self, kind: DeclKind, node: Node<'_>, name: Option<Node<'_>>) -> Decl {
        let span = self.span(node);
        let text = name.map(|n| self.text(n).to_string()).unwrap_or_default();
        if declares_a_type(kind) && !text.is_empty() {
            self.type_names.borrow_mut().push(hash_name(&text));
        }
        Decl {
            kind,
            name: text,
            span,
            name_span: name.map_or(span, |n| self.span(n)),
            doc: self.doc(node),
            doc_span: self.doc_span(node),
            modifiers: self.modifiers(node),
            type_parameters: None,
            extends: None,
            implements: Vec::new(),
            declared_type: None,
            parameters: Vec::new(),
            throws: Vec::new(),
            varargs: false,
            children: Vec::new(),
            initializer: None,
            has_initializer: false,
        }
    }

    /// Modifier keywords and annotations, as written and in source order.
    fn modifiers(&self, node: Node<'_>) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "modifiers" => {
                    let mut inner = child.walk();
                    out.extend(child.children(&mut inner).map(|m| self.text(m).to_string()));
                }
                "annotation" | "marker_annotation" => out.push(self.text(child).to_string()),
                _ => {}
            }
        }
        out
    }

    /// Every interface a type declaration names, whichever keyword introduced
    /// it. A class and an enum hang theirs off the `interfaces` field; an
    /// interface has an `extends_interfaces` child with no field name at all.
    fn implemented(&self, node: Node<'_>) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        let clauses = node.child_by_field_name("interfaces").into_iter().chain(
            node.named_children(&mut cursor)
                .filter(|child| child.kind() == "extends_interfaces"),
        );
        for clause in clauses {
            let Some(list) = named_child(clause, "type_list") else {
                continue;
            };
            let mut inner = list.walk();
            out.extend(
                list.named_children(&mut inner)
                    .map(|ty| self.text(ty).to_string()),
            );
        }
        out
    }

    fn throws(&self, node: Node<'_>) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            // The keyword shares its kind with the clause; only one is named.
            if child.is_named() && child.kind() == "throws" {
                let mut inner = child.walk();
                out.extend(
                    child
                        .named_children(&mut inner)
                        .map(|thrown| self.text(thrown).to_string()),
                );
            }
        }
        out
    }

    /// The doc comment immediately preceding a declaration, verbatim.
    ///
    /// Javadoc tags are not parsed: hover renders the comment, and nothing else
    /// needs to know what is inside it.
    fn doc(&self, node: Node<'_>) -> Option<String> {
        let previous = node.prev_sibling()?;
        if previous.kind() != "block_comment" {
            return None;
        }
        let text = self.text(previous);
        // `/**/` is an empty block comment, not a doc comment.
        (text.starts_with("/**") && text.len() > 4).then(|| text.to_string())
    }

    /// Where [`Self::doc`]'s comment is.
    fn doc_span(&self, node: Node<'_>) -> Option<(u32, u32)> {
        let previous = node.prev_sibling()?;
        (previous.kind() == "block_comment"
            && self.text(previous).starts_with("/**")
            && self.text(previous).len() > 4)
            .then(|| (previous.start_byte() as u32, previous.end_byte() as u32))
    }

    fn field_text(&self, node: Node<'_>, field: &str) -> Option<String> {
        node.child_by_field_name(field)
            .map(|child| self.text(child).to_string())
    }

    fn span(&self, node: Node<'_>) -> Span {
        let start_byte = node.start_byte() as u32;
        let end_byte = node.end_byte() as u32;
        Span {
            start_byte,
            end_byte,
            range: Range::new(
                self.document.position_at(start_byte),
                self.document.position_at(end_byte),
            ),
        }
    }

    /// Source text of a node. Slicing rather than `utf8_text` so a node whose
    /// range the grammar reported oddly yields nothing instead of an error.
    fn text(&self, node: Node<'_>) -> &str {
        self.source
            .get(node.start_byte()..node.end_byte())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JavaParser;
    use javelin_document::{DocumentId, Position};

    fn view(source: &str) -> FileSyntax {
        let document = TextDocument::new(DocumentId::new("file:///A.java"), 1, source);
        let tree = JavaParser::new()
            .expect("grammar")
            .parse(&document)
            .expect("parse");
        FileSyntax::build(&document, &tree)
    }

    #[test]
    fn a_class_or_an_enum_writes_its_constructors_unless_lombok_may_add_one() {
        let writes = |source: &str| view(source).declarations[0].writes_its_constructors();
        assert!(writes("class A { A(int a) {} void b() {} }"));
        assert!(writes("@Deprecated class A { A() {} }"));
        assert!(!writes("class A { void a() {} }"));
        assert!(!writes("class A { }"));
        assert!(!writes("class A { int a; class B {} }"));
        assert!(!writes("@lombok.Data class A { A() {} }"));
        assert!(!writes(
            "@AllArgsConstructor(staticName = \"of\") class A { A() {} }"
        ));
        assert!(!writes("@Builder class A { A() {} }"));
        assert!(!writes("record A(int a) { A {} }"));
        assert!(writes("enum A { X; A() {} }"));
        assert!(!writes("enum A { X }"));
        assert!(!writes("interface A { }"));
    }

    #[test]
    fn the_innermost_type_holding_an_offset_is_found_through_members_and_bodies() {
        let source = "class A { void m() { class L { int x; } } enum E { X(1); E(int n) {} } }";
        let syntax = view(source);
        let at = |needle: &str| {
            let offset = u32::try_from(source.find(needle).expect("needle")).expect("offset");
            syntax
                .innermost_type_at(offset)
                .map(|decl| decl.name.as_str())
        };
        assert_eq!(at("void"), Some("A"));
        assert_eq!(at("int x"), Some("L"));
        assert_eq!(at("X(1)"), Some("E"));
        assert_eq!(at("n) {}"), Some("E"));
        assert_eq!(at("class A"), Some("A"));
        assert_eq!(view("int x;").innermost_type_at(0), None);
    }

    /// The first declaration reachable by walking the given names downward.
    fn find<'a>(decls: &'a [Decl], path: &[&str]) -> &'a Decl {
        let (head, rest) = path.split_first().expect("a non-empty path");
        let found = decls
            .iter()
            .find(|decl| decl.name == *head)
            .unwrap_or_else(|| panic!("no declaration named {head} in {:?}", names(decls)));
        if rest.is_empty() {
            found
        } else {
            find(&found.children, rest)
        }
    }

    /// `declaration_at` as it was, a linear search of every declaration, kept
    /// as the reference the guided descent is checked against.
    fn linear_declaration_at(decls: &[Decl], name_span: Span) -> Option<&Decl> {
        decls.iter().find_map(|decl| {
            if !decl.name.is_empty()
                && decl.name_span.start_byte == name_span.start_byte
                && decl.name_span.end_byte == name_span.end_byte
            {
                return Some(decl);
            }
            linear_declaration_at(&decl.parameters, name_span)
                .or_else(|| linear_declaration_at(&decl.children, name_span))
        })
    }

    const DECLARATION_SOURCES: &[&str] = &[
        "package p; import a.B; class A { int x, y; void m(int p, String... rest) { int l = 1, k; \
         for (int i = 0;;) {} Runnable r = new Runnable() { public void run() { int in; } }; \
         class Local { int f; } } A() {} static {} {} }",
        "enum E { ONE, TWO { void f() {} }; int code; E() {} }",
        "record R(int a, String b) { R { } static int s; }",
        "@interface Ann { int value() default 1; String[] names(); }",
        "interface I { int C = 1; default void m() { int l = 1; } }",
        "class A { void m( } class B { int x; }",
        "class A { class B { class C { void deep(int p) { int q; } } } }",
        // Recovery drops the class and hoists its members: whichever side of
        // `ordered` this lands on, the answer must not change.
        "class A { Box b; void m() { b. } }",
    ];

    #[test]
    fn the_guided_descent_agrees_with_a_linear_search_at_every_name() {
        for source in DECLARATION_SOURCES {
            let document = TextDocument::new(DocumentId::new("file:///A.java"), 1, *source);
            let tree = JavaParser::new()
                .expect("grammar")
                .parse(&document)
                .expect("parse");
            let syntax = FileSyntax::build(&document, &tree);
            // Every token span in the file, which covers every declaration's
            // name and every position that is not one.
            let mut spans = Vec::new();
            let mut stack = vec![tree.root()];
            while let Some(node) = stack.pop() {
                spans.push(crate::queries::span(&document, node));
                let mut cursor = node.walk();
                stack.extend(node.children(&mut cursor));
            }
            for span in spans {
                assert_eq!(
                    syntax.declaration_at(span).map(|decl| decl.name_span),
                    linear_declaration_at(&syntax.declarations, span).map(|decl| decl.name_span),
                    "{}..{} in {source:?}",
                    span.start_byte,
                    span.end_byte
                );
            }
        }
    }

    #[test]
    fn a_file_in_declaration_order_is_searched_by_descent() {
        // The premise of the equivalence test above: its fixtures exercise the
        // descent, not the fallback.
        for source in &DECLARATION_SOURCES[..DECLARATION_SOURCES.len() - 1] {
            assert!(view(source).ordered, "{source:?}");
        }
    }

    fn fqns(source: &str) -> Vec<String> {
        view(source)
            .named_types()
            .into_iter()
            .map(|named| named.fqn)
            .collect()
    }

    #[test]
    fn named_types_qualify_nested_types_and_skip_members_and_local_classes() {
        assert_eq!(
            fqns(
                "package com.example;\n\
                 class Outer { int f; void m() { class Local {} new Runnable() {}; } \
                 interface Inner { enum Deep {} } }\n\
                 record R(int x) {}"
            ),
            vec![
                "com.example.Outer",
                "com.example.Outer.Inner",
                "com.example.Outer.Inner.Deep",
                "com.example.R",
            ]
        );
    }

    #[test]
    fn named_types_in_the_default_package_have_no_prefix() {
        assert_eq!(fqns("class A { class B {} }"), vec!["A", "A.B"]);
    }

    #[test]
    fn a_named_types_decl_is_the_declaration_itself() {
        let syntax = view("package p; @interface Marker {}");
        let named = syntax.named_types();
        assert_eq!(named[0].decl.kind, DeclKind::AnnotationType);
        assert_eq!(named[0].decl.name, "Marker");
    }

    fn names(decls: &[Decl]) -> Vec<&str> {
        decls.iter().map(|decl| decl.name.as_str()).collect()
    }

    fn kinds(decls: &[Decl]) -> Vec<DeclKind> {
        decls.iter().map(|decl| decl.kind).collect()
    }

    #[test]
    fn a_package_declaration_is_captured_as_written() {
        let view = view("package com.example.deep;\nclass A {}\n");
        let package = view.package.expect("a package");
        assert_eq!(package.name, "com.example.deep");
        assert_eq!(package.name_span.start_byte, 8);
        assert_eq!(package.span.start_byte, 0);
        assert_eq!(package.span.end_byte, 25);
    }

    #[test]
    fn a_file_without_a_package_has_none() {
        assert!(view("class A {}").package.is_none());
    }

    #[test]
    fn imports_record_static_and_on_demand_separately() {
        let view = view(
            "import java.util.List;\n\
             import static java.util.Arrays.asList;\n\
             import java.io.*;\n\
             import static java.util.Arrays.*;\n",
        );
        let flags: Vec<_> = view
            .imports
            .iter()
            .map(|i| (i.name.as_str(), i.is_static, i.is_on_demand))
            .collect();
        assert_eq!(
            flags,
            vec![
                ("java.util.List", false, false),
                ("java.util.Arrays.asList", true, false),
                ("java.io", false, true),
                ("java.util.Arrays", true, true),
            ]
        );
    }

    #[test]
    fn a_class_carries_its_modifiers_annotations_and_type_parameters() {
        let view = view("@Deprecated\npublic abstract class Box<T extends Number> {}");
        let class = &view.declarations[0];
        assert_eq!(class.kind, DeclKind::Class);
        assert_eq!(class.name, "Box");
        assert_eq!(class.modifiers, vec!["@Deprecated", "public", "abstract"]);
        assert_eq!(class.type_parameters.as_deref(), Some("<T extends Number>"));
    }

    #[test]
    fn the_name_span_covers_the_name_and_the_declaration_span_covers_everything() {
        let view = view("public class Box {}");
        let class = &view.declarations[0];
        assert_eq!((class.span.start_byte, class.span.end_byte), (0, 19));
        assert_eq!(
            (class.name_span.start_byte, class.name_span.end_byte),
            (13, 16)
        );
        assert_eq!(class.name_span.range.start, Position::new(0, 13));
    }

    #[test]
    fn spans_count_characters_the_way_the_editor_does() {
        // The name has to sit *after* the emoji on the same line: the emoji is
        // four bytes but two UTF-16 code units, so a byte column and an LSP
        // character column disagree from there to the end of that line. On any
        // later line the two agree again and the test would prove nothing.
        let source = "class A {\n  String s = \"🚀\"; int after;\n}\n";
        let view = view(source);
        let field = find(&view.declarations, &["A", "after"]);
        let bytes = field.name_span.start_byte as usize..field.name_span.end_byte as usize;
        assert_eq!(&source[bytes], "after");

        let line_start = source.find("  String").expect("the second line") as u32;
        let byte_column = field.name_span.start_byte - line_start;
        let reported = field.name_span.range.start;
        assert_eq!(reported.line, 1);
        assert_ne!(
            reported.character, byte_column,
            "a byte column reported as an LSP character is the exact bug this guards"
        );
        assert_eq!(
            reported.character,
            byte_column - 2,
            "the emoji costs four bytes but only two UTF-16 code units"
        );
    }

    #[test]
    fn every_type_declaration_kind_is_recognised() {
        let view = view("class C {}\ninterface I {}\nenum E {}\nrecord R() {}\n@interface A {}\n");
        assert_eq!(
            kinds(&view.declarations),
            vec![
                DeclKind::Class,
                DeclKind::Interface,
                DeclKind::Enum,
                DeclKind::Record,
                DeclKind::AnnotationType,
            ]
        );
    }

    #[test]
    fn a_field_carries_its_declared_type_as_written() {
        let view = view("class A { private final Map<String, List<Integer>> m = null; }");
        let field = find(&view.declarations, &["A", "m"]);
        assert_eq!(field.kind, DeclKind::Field);
        assert_eq!(
            field.declared_type.as_deref(),
            Some("Map<String, List<Integer>>")
        );
        assert_eq!(field.modifiers, vec!["private", "final"]);
    }

    #[test]
    fn one_field_declaration_with_several_declarators_yields_one_decl_each() {
        let view = view("class A { int a, b, c; }");
        let fields = &view.declarations[0].children;
        assert_eq!(names(fields), vec!["a", "b", "c"]);
        assert!(
            fields
                .iter()
                .all(|f| f.declared_type.as_deref() == Some("int"))
        );
        // Sibling spans must not overlap, so none of them may claim the type.
        assert!(fields[0].span.end_byte <= fields[1].span.start_byte);
    }

    #[test]
    fn a_method_captures_its_signature_as_written() {
        let view = view(
            "class A {\n\
             @Override\n\
             public <T> List<T> pick(int index, T... rest)\n\
             throws IOException, SQLException { return null; }\n\
             }\n",
        );
        let method = find(&view.declarations, &["A", "pick"]);
        assert_eq!(method.kind, DeclKind::Method);
        assert_eq!(method.declared_type.as_deref(), Some("List<T>"));
        assert_eq!(method.type_parameters.as_deref(), Some("<T>"));
        assert_eq!(method.modifiers, vec!["@Override", "public"]);
        assert_eq!(method.throws, vec!["IOException", "SQLException"]);
        assert_eq!(names(&method.parameters), vec!["index", "rest"]);
        assert_eq!(method.parameters[0].declared_type.as_deref(), Some("int"));
        assert!(!method.parameters[0].varargs);
        assert_eq!(method.parameters[1].declared_type.as_deref(), Some("T"));
        assert!(method.parameters[1].varargs, "T... is varargs");
    }

    #[test]
    fn brackets_after_a_name_are_part_of_its_written_type() {
        let view = view("class A { int a[], b; void f(String s[][]) { long c[] = null; } }");
        let field = |name| {
            find(&view.declarations, &["A", name])
                .declared_type
                .as_deref()
        };
        assert_eq!(field("a"), Some("int[]"));
        assert_eq!(field("b"), Some("int"));
        let method = find(&view.declarations, &["A", "f"]);
        assert_eq!(
            method.parameters[0].declared_type.as_deref(),
            Some("String[][]")
        );
        assert_eq!(method.children[0].declared_type.as_deref(), Some("long[]"));
    }

    #[test]
    fn an_abstract_method_without_a_body_is_still_a_method() {
        let view = view("interface I { void go(); }");
        let method = find(&view.declarations, &["I", "go"]);
        assert_eq!(method.declared_type.as_deref(), Some("void"));
        assert!(method.children.is_empty());
    }

    #[test]
    fn a_parameter_keeps_its_own_modifiers() {
        let view = view("class A { void go(final @NotNull String s) {} }");
        let parameter = &find(&view.declarations, &["A", "go"]).parameters[0];
        assert_eq!(parameter.kind, DeclKind::Parameter);
        assert_eq!(parameter.modifiers, vec!["final", "@NotNull"]);
    }

    #[test]
    fn a_constructor_is_distinguished_from_a_method_of_the_same_name() {
        let view = view("class A { A(int x) {} void A() {} }");
        assert_eq!(
            kinds(&view.declarations[0].children),
            vec![DeclKind::Constructor, DeclKind::Method]
        );
    }

    #[test]
    fn initialiser_blocks_are_captured_and_have_no_name() {
        let view = view("class A { static { int s = 1; } { int i = 2; } }");
        let members = &view.declarations[0].children;
        assert_eq!(
            kinds(members),
            vec![DeclKind::StaticInitializer, DeclKind::InstanceInitializer]
        );
        assert!(members.iter().all(|m| m.name.is_empty()));
        assert_eq!(names(&members[0].children), vec!["s"]);
        assert_eq!(names(&members[1].children), vec!["i"]);
    }

    #[test]
    fn enum_constants_and_their_bodies_are_members_of_the_enum() {
        let view =
            view("enum E {\n  ONE,\n  TWO { int extra() { return 2; } };\n  int shared;\n}\n");
        let members = &view.declarations[0].children;
        assert_eq!(names(members), vec!["ONE", "TWO", "shared"]);
        assert_eq!(members[0].kind, DeclKind::EnumConstant);
        assert_eq!(names(&members[1].children), vec!["extra"]);
        assert_eq!(members[2].kind, DeclKind::Field);
    }

    #[test]
    fn record_components_and_a_compact_constructor_are_both_members() {
        let view = view("record Point(int x, int y) { Point { } int sum() { return x + y; } }");
        let record = &view.declarations[0];
        assert_eq!(names(&record.children), vec!["x", "y", "Point", "sum"]);
        assert_eq!(record.children[0].kind, DeclKind::RecordComponent);
        assert_eq!(record.children[0].declared_type.as_deref(), Some("int"));
        assert_eq!(record.children[2].kind, DeclKind::Constructor);
    }

    #[test]
    fn annotation_elements_carry_their_type() {
        let view = view("@interface Config { String name(); int count() default 1; }");
        let members = &view.declarations[0].children;
        assert_eq!(names(members), vec!["name", "count"]);
        assert_eq!(members[0].kind, DeclKind::AnnotationElement);
        assert_eq!(members[0].declared_type.as_deref(), Some("String"));
    }

    #[test]
    fn a_field_of_an_interface_is_a_field() {
        let view = view("interface I { int MAX = 1; }");
        let field = find(&view.declarations, &["I", "MAX"]);
        assert_eq!(field.kind, DeclKind::Field);
        assert_eq!(field.declared_type.as_deref(), Some("int"));
    }

    #[test]
    fn nested_inner_and_local_types_nest_where_they_were_written() {
        let view = view(
            "class Outer {\n\
             static class Nested {}\n\
             class Inner {}\n\
             void go() { class Local {} }\n\
             }\n",
        );
        let outer = &view.declarations[0];
        assert_eq!(names(&outer.children), vec!["Nested", "Inner", "go"]);
        assert_eq!(outer.children[0].modifiers, vec!["static"]);
        assert_eq!(
            names(&find(&view.declarations, &["Outer", "go"]).children),
            vec!["Local"]
        );
    }

    #[test]
    fn an_anonymous_class_nests_under_the_variable_it_initialises() {
        let view = view(
            "class A {\n\
             void go() {\n\
             Runnable r = new Runnable() {\n\
             public void run() {}\n\
             };\n\
             }\n\
             }\n",
        );
        let local = find(&view.declarations, &["A", "go", "r"]);
        assert_eq!(local.kind, DeclKind::LocalVariable);
        assert_eq!(local.declared_type.as_deref(), Some("Runnable"));
        let anonymous = &local.children[0];
        assert_eq!(anonymous.kind, DeclKind::AnonymousClass);
        assert_eq!(anonymous.name, "Runnable");
        assert_eq!(names(&anonymous.children), vec!["run"]);
    }

    #[test]
    fn an_anonymous_class_in_an_argument_is_found_too() {
        let view = view("class A { void go() { register(new Listener() { int seen; }); } }");
        let anonymous = &find(&view.declarations, &["A", "go"]).children[0];
        assert_eq!(anonymous.kind, DeclKind::AnonymousClass);
        assert_eq!(names(&anonymous.children), vec!["seen"]);
    }

    #[test]
    fn a_lambda_parameter_records_the_call_it_is_an_argument_of() {
        let source = "class A { void go(java.util.List<String> l) { \
                      l.forEach(x -> x.length()); m(1, (a, b) -> a); \
                      Runnable r = () -> {}; f((String s) -> s); new T(y -> y); } }";
        let view = view(source);
        let at = |needle: &str| {
            view.lambda_parameter(source.find(needle).expect(needle) as u32)
                .unwrap_or_else(|| panic!("no lambda parameter at {needle}"))
        };
        let called = |parameter: &LambdaParameter| match parameter.call.as_deref() {
            Some(Expr::Call { name, .. }) => name.clone(),
            other => panic!("not a call: {other:?}"),
        };
        let x = at("x ->");
        assert_eq!(
            (x.declared.as_deref(), x.argument, x.position, x.arity),
            (None, 0, 0, 1)
        );
        assert_eq!(called(x), "forEach");
        let b = at("b) ->");
        assert_eq!((b.argument, b.position, b.arity), (1, 1, 2));
        assert_eq!(called(b), "m");
        let s = at("s) ->");
        assert_eq!(s.declared.as_deref(), Some("String"));
        assert_eq!(called(s), "f");
        // A constructor's argument is no call this records.
        assert!(at("y ->").call.is_none());
        assert!(
            view.lambda_parameter(source.find("l) {").expect("l") as u32)
                .is_none()
        );
    }

    /// A call is projected once however many lambdas it takes, and a lambda
    /// with no parameters records nothing: `FileSyntax` is rebuilt on every
    /// keystroke, and projecting the call per lambda made a generated
    /// `List.of(x0 -> x0, …)` quadratic.
    #[test]
    fn a_call_taking_many_lambdas_is_projected_once() {
        let arguments: Vec<String> = (0..4000).map(|n| format!("x{n} -> x{n}")).collect();
        let source = format!(
            "class A {{ void go() {{ of(\n{},\n() -> 0); }} }}",
            arguments.join(",\n")
        );
        let started = std::time::Instant::now();
        let view = view(&source);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "{:?} for 4 000 lambdas",
            started.elapsed()
        );
        let first = view
            .lambda_parameter(source.find("x0 ->").expect("x0") as u32)
            .expect("x0");
        let last = view
            .lambda_parameter(source.find("x3999 ->").expect("x3999") as u32)
            .expect("x3999");
        assert_eq!(last.argument, 3999);
        assert!(
            std::sync::Arc::ptr_eq(
                first.call.as_ref().expect("a call"),
                last.call.as_ref().expect("a call")
            ),
            "one projection, shared"
        );
        assert_eq!(
            view.lambda_parameters.len(),
            4000,
            "`() -> 0` records nothing"
        );
    }

    #[test]
    fn locals_declared_inside_a_lambda_are_reported_against_the_enclosing_method() {
        let view =
            view("class A { void go() { run(() -> { int inside = 1; }); Runnable r = A::go; } }");
        assert_eq!(
            names(&find(&view.declarations, &["A", "go"]).children),
            vec!["inside", "r"]
        );
    }

    #[test]
    fn only_a_var_local_carries_a_projected_initialiser() {
        // The pin on a performance guard, not on a nicety: `FileSyntax` is
        // rebuilt whole for every document version, so projecting an
        // expression per declarator would put file-sized work back on a path
        // that already misses completion's p95 on a generated god-class.
        // Without this test the guard rots quietly into "project everything".
        //
        // `var d;` carries a second pin. It is not legal Java — the JLS
        // requires an initialiser — but it is what a buffer looks like
        // mid-keystroke, and `infer`'s
        // `a_var_local_with_no_initialiser_is_unknown` rests on recovery still
        // reading it as a declarator with a name and no value. If it ever
        // stops, that test would keep passing for a different reason; this one
        // fails instead, which is where the assumption belongs.
        let view = view("class A { void go() { Box b = new Box(); var c = new Box(); var d; } }");
        let locals = &find(&view.declarations, &["A", "go"]).children;
        assert_eq!(names(locals), vec!["b", "c", "d"]);
        assert_eq!(locals[0].initializer, None);
        assert!(matches!(
            locals[1].initializer.as_deref(),
            Some(Expr::New { type_text, .. }) if type_text == "Box"
        ));
        assert_eq!(locals[2].declared_type.as_deref(), Some("var"));
        assert_eq!(locals[2].initializer, None);
    }

    #[test]
    fn a_for_each_variable_is_a_local_with_its_declared_type() {
        let view = view("class A { void go(List<String> all) { for (String s : all) {} } }");
        let local = &find(&view.declarations, &["A", "go"]).children[0];
        assert_eq!(local.name, "s");
        assert_eq!(local.kind, DeclKind::LocalVariable);
        assert_eq!(local.declared_type.as_deref(), Some("String"));
        assert_eq!(local.initializer, None, "a written type needs no element");
    }

    #[test]
    fn a_for_each_var_carries_the_element_of_what_it_iterates() {
        let view = view("class A { void go(List<String> all) { for (var s : all) {} } }");
        let local = &find(&view.declarations, &["A", "go"]).children[0];
        assert!(matches!(
            local.initializer.as_deref(),
            Some(Expr::Element { iterable, .. })
                if matches!(iterable.as_ref(), Expr::Name { name, .. } if name == "all")
        ));
    }

    #[test]
    fn a_catch_parameter_and_a_try_resource_are_locals() {
        let view = view(
            "class A { void go() { try (Reader r = open()) { } \
             catch (IOException | SQLException e) { } } }",
        );
        let locals = &find(&view.declarations, &["A", "go"]).children;
        assert_eq!(names(locals), vec!["r", "e"]);
        assert_eq!(locals[0].declared_type.as_deref(), Some("Reader"));
        assert_eq!(
            locals[1].declared_type.as_deref(),
            Some("IOException | SQLException"),
            "a multi-catch type is text like any other"
        );
    }

    #[test]
    fn a_pattern_binding_is_a_local() {
        let view = view(
            "class A { void go(Object o) { if (o instanceof String s) {} \
             switch (o) { case Integer i -> {} default -> {} } } }",
        );
        let locals = &find(&view.declarations, &["A", "go"]).children;
        assert_eq!(names(locals), vec!["s", "i"]);
        assert_eq!(locals[0].declared_type.as_deref(), Some("String"));
        assert_eq!(locals[1].declared_type.as_deref(), Some("Integer"));
    }

    #[test]
    fn a_doc_comment_is_attached_verbatim_to_what_it_precedes() {
        let view = view(
            "/** The type. */\n\
             class A {\n\
             /** The field. */\n\
             int x;\n\
             /** The method. */\n\
             void go() {}\n\
             // not a doc comment\n\
             void other() {}\n\
             }\n",
        );
        let class = &view.declarations[0];
        assert_eq!(class.doc.as_deref(), Some("/** The type. */"));
        assert_eq!(
            find(&view.declarations, &["A", "x"]).doc.as_deref(),
            Some("/** The field. */")
        );
        assert_eq!(
            find(&view.declarations, &["A", "go"]).doc.as_deref(),
            Some("/** The method. */")
        );
        assert!(find(&view.declarations, &["A", "other"]).doc.is_none());
    }

    #[test]
    fn a_plain_block_comment_is_not_a_doc_comment() {
        let view = view("/* just a comment */\nclass A {}");
        assert!(view.declarations[0].doc.is_none());
    }

    #[test]
    fn a_half_typed_file_yields_the_declarations_that_did_parse() {
        let view = view(
            "package com.example;\n\
             public class Broken {\n\
             int good;\n\
             public void go() {\n\
             int x = ;\n",
        );
        assert_eq!(view.package.map(|p| p.name).as_deref(), Some("com.example"));
        let class = &view.declarations[0];
        assert_eq!(class.name, "Broken");
        assert!(
            names(&class.children).contains(&"good"),
            "the members before the damage survive: {:?}",
            names(&class.children)
        );
    }

    #[test]
    fn a_file_of_pure_noise_yields_an_empty_view_rather_than_a_panic() {
        let view = view("}}} ??? <<< @@@ ;;;");
        assert!(view.declarations.is_empty());
        assert!(view.package.is_none());
    }

    #[test]
    fn an_empty_file_yields_an_empty_view() {
        let view = view("");
        assert!(view.declarations.is_empty());
        assert_eq!(view.version, 1);
    }

    #[test]
    fn a_class_missing_its_closing_brace_still_reports_its_members() {
        let view = view("class A {\n  int x;\n  void go() {}\n");
        assert_eq!(names(&view.declarations[0].children), vec!["x", "go"]);
    }

    #[test]
    fn the_view_remembers_the_version_of_the_tree_it_was_built_from() {
        let document = TextDocument::new(DocumentId::new("file:///A.java"), 7, "class A {}");
        let tree = JavaParser::new()
            .expect("grammar")
            .parse(&document)
            .expect("parse");
        assert_eq!(FileSyntax::build(&document, &tree).version, 7);
    }

    #[test]
    fn deeply_nested_source_is_truncated_rather_than_overflowing_the_stack() {
        let depth = MAX_DEPTH as usize + 50;
        let source = format!(
            "class A {{ {} int deep; {} }}",
            "class N {".repeat(depth),
            "}".repeat(depth)
        );
        assert!(!view(&source).declarations.is_empty());
    }

    #[test]
    fn a_class_records_its_superclass_and_interfaces_as_written() {
        let view = view("class Box<T> extends Base<T> implements Holder, Sized {}");
        let class = &view.declarations[0];
        assert_eq!(class.extends.as_deref(), Some("Base<T>"));
        assert_eq!(class.implements, vec!["Holder", "Sized"]);
    }

    #[test]
    fn an_interface_records_everything_it_extends_as_an_implement() {
        let view = view("interface Shape extends Drawable, Sized {}");
        let interface = &view.declarations[0];
        assert_eq!(interface.extends, None);
        assert_eq!(interface.implements, vec!["Drawable", "Sized"]);
    }

    #[test]
    fn an_enum_records_its_interfaces() {
        let view = view("enum Colour implements Named { RED }");
        assert_eq!(view.declarations[0].implements, vec!["Named"]);
    }

    #[test]
    fn a_type_with_no_supertypes_records_none() {
        let view = view("class Plain {}");
        assert_eq!(view.declarations[0].extends, None);
        assert!(view.declarations[0].implements.is_empty());
    }

    #[test]
    fn a_declaration_is_found_by_its_name_span() {
        let view = view("class Outer { int field; }");
        let field = &view.declarations[0].children[0];
        let found = view.declaration_at(field.name_span).expect("the field");
        assert_eq!(found.name, "field");
    }

    #[test]
    fn a_span_that_names_no_declaration_finds_nothing() {
        let view = view("class Outer {}");
        let nowhere = Span {
            start_byte: 999,
            end_byte: 1000,
            range: view.declarations[0].name_span.range,
        };
        assert!(view.declaration_at(nowhere).is_none());
    }

    #[test]
    fn a_top_level_type_is_reported_as_declared() {
        let view = view("class Box {}");
        assert!(view.declares_type_named("Box"));
    }

    #[test]
    fn a_nested_type_is_reported_as_declared() {
        let view = view("class Outer { class Inner {} }");
        assert!(view.declares_type_named("Inner"));
    }

    #[test]
    fn a_type_declared_inside_a_method_body_is_reported_as_declared() {
        let view = view("class Outer { void go() { class Local {} } }");
        assert!(view.declares_type_named("Local"));
    }

    #[test]
    fn a_name_the_file_does_not_declare_is_not_reported() {
        let view = view("class Box {}");
        assert!(!view.declares_type_named("String"));
    }

    #[test]
    fn a_field_or_method_name_is_not_a_declared_type() {
        let view = view("class Box { int Field; void Method() {} }");
        assert!(!view.declares_type_named("Field"));
        assert!(!view.declares_type_named("Method"));
    }

    #[test]
    fn an_anonymous_class_contributes_no_name() {
        let view = view(
            "class A {\n\
             void go() {\n\
             Runnable r = new Runnable() {\n\
             public void run() {}\n\
             };\n\
             }\n\
             }\n",
        );
        // The premise: `FileSyntax` really does record the anonymous class,
        // under the name of the type it instantiates.
        let local = find(&view.declarations, &["A", "go", "r"]);
        assert_eq!(local.children[0].kind, DeclKind::AnonymousClass);
        assert_eq!(local.children[0].name, "Runnable");
        // But that is not a declaration of `Runnable`: the file never says
        // `class Runnable`, only instantiates one anonymously, and treating
        // the two the same would veto the classpath for `Runnable` anywhere
        // else in the file.
        assert!(!view.declares_type_named("Runnable"));
    }
}
