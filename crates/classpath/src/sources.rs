//! Reading Java type facts out of a JDK's `src.zip`.
//!
//! Two passes over one parse, because parsing all of `src.zip` is the expensive
//! part and must happen exactly once. Pass one reads every candidate entry,
//! parses it, and keeps its declarations, its supertypes and its members'
//! types exactly as written, and the package/imports needed to make sense of
//! those names later. Pass two resolves every written name against a
//! [`TypeTable`] built from every fully-qualified name pass one found — this
//! is [`crate::resolve`]'s first caller.
//!
//! A member is read in that same first pass and not on demand later: the walk
//! over a type's children is already there, and a second visit would mean
//! either keeping every parse tree alive or opening the archive again.

use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::rc::Rc;

use javelin_core::{CancellationToken, Error, Result};
use javelin_document::{DocumentId, TextDocument};
use javelin_parser::{Decl, DeclKind, FileSyntax, ImportDecl, JavaParser};

use crate::resolve::{Resolution, TypeTable, simple_name};

/// One type the sources declare.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypeFacts {
    /// Source-form fully-qualified name, `.` between nesting levels:
    /// `java.util.Map.Entry`.
    pub fqn: String,
    pub simple: String,
    pub kind: DeclKind,
    /// Written `public`, or a member type of an interface or an annotation type,
    /// which is implicitly so. A class file says `public` for a nested type
    /// written `protected` too. A type is nameable from another package only
    /// when this holds for it and for every type it is nested in, which a
    /// caller checks against the outer types' facts.
    pub is_public: bool,
    /// The declaration's type parameters by name, in order: `["K", "V"]`.
    /// What a [`MemberType::Variable`] names, and what a use's type arguments
    /// bind to.
    pub type_parameters: Vec<String>,
    /// Direct supertypes, fully qualified, and only those that resolved to
    /// exactly one type. A supertype that resolved ambiguously or not at all
    /// is *dropped*, and `supertypes_complete` goes false.
    pub supertypes: Vec<String>,
    /// The type arguments a direct supertype was written with, for each one
    /// that wrote any: `ArrayList<E>` has `("java.util.List", [Variable("E")])`.
    /// A supertype absent here is raw — or synthesised, like `Object`.
    pub supertype_arguments: Vec<(String, Vec<MemberType>)>,
    /// False when any direct supertype was dropped. A closure that contains an
    /// incomplete node cannot answer "is not a subtype of".
    pub supertypes_complete: bool,
    /// False when `members` may be short of what the type declares: its
    /// source parsed with errors, which recovery can drop a member over; or
    /// a multi-release JAR overlays its class with a newer one this reader
    /// does not read. An enum's `values()` and `valueOf` and a record's
    /// accessors are among `members`, from [`implicit_members`]. A caller that claims a member is
    /// *absent* needs this; one that lists members does not.
    pub members_complete: bool,
    /// Every member this type declares that another file could select — its
    /// own, never an inherited one. A walk that wants the inherited ones
    /// follows `supertypes`, which is how the file-local walk reaches them too.
    pub members: Vec<MemberFacts>,
    /// The constructors the type writes, apart from `members` so that no
    /// member list can hold one — and only where they are all it has
    /// (`Decl::writes_its_constructors`): empty otherwise, and empty for a
    /// class file's facts, whose constructors are not read.
    pub constructors: Vec<MemberFacts>,
    /// Byte offsets of the type's `/** … */` within `source`, where it has
    /// one.
    pub doc_span: Option<(u32, u32)>,
    /// The `src.zip` entry this type was read out of, as the archive names it:
    /// `java.base/java/lang/String.java`. A nested type carries its *outer*
    /// type's entry, because that is the file both are written in.
    pub source: String,
    /// Byte offsets of the type's own name within `source`, which is what a
    /// jump to the type lands on. Both zero where nothing recorded one — a
    /// class file has no source to put a name in.
    pub name_start: u32,
    pub name_end: u32,
}

/// One member a type declares.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemberFacts {
    pub name: String,
    pub kind: DeclKind,
    pub is_static: bool,
    /// A method with no body to call: an interface's that is not `default`,
    /// `static` or `private`, or one declared `abstract`. A functional
    /// interface has exactly one, which is what types a lambda's parameters.
    pub is_abstract: bool,
    /// Callable from any package: written `public`, or implicitly so as an
    /// interface's or an annotation type's non-`private` member or an enum
    /// constant. What most-specific selection needs of a winner, since a
    /// member the caller cannot see is no candidate of Java's.
    pub is_public: bool,
    /// A field's type or a method's return type.
    pub declared: MemberType,
    /// A method's formal parameters, in order. Empty for a field.
    pub parameters: Vec<MemberType>,
    /// Their names, in the same order — or empty where the reader could not
    /// say: a class file keeps none unless compiled with `-parameters`, and
    /// is only given them by its sources JAR.
    pub parameter_names: Vec<String>,
    /// Byte offsets of the member's `/** … */` within [`TypeFacts::source`],
    /// where it has one — what hover reads its Javadoc out of.
    pub doc_span: Option<(u32, u32)>,
    /// A generic method's own type parameters, `<R>` in `<R> R map(…)`: the
    /// names a [`MemberType::Unresolved`] in its types can be bound to at a
    /// call, by what the call passes.
    pub type_parameters: Vec<String>,
    /// Whether the last parameter is `...`.
    pub variadic: bool,
    /// Byte offsets of the member's own name within [`TypeFacts::source`].
    /// Where go-to-definition lands, converted against the entry's text at the
    /// moment it is asked for — see `crates/server/src/jdk_sources.rs`.
    pub name_start: u32,
    pub name_end: u32,
}

/// A member's type, resolved as far as pass two could take it.
///
/// There is no `complete` flag beside this on [`MemberFacts`], and there is no
/// need for one: [`MemberType::Unresolved`] says per member — and per
/// parameter — that nothing is claimed about that one name, where
/// [`TypeFacts::supertypes_complete`] has to speak for a whole closure at once.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemberType {
    Void,
    /// The keyword, `"int"` or `"boolean"`, rather than a
    /// `javelin_semantic::Primitive`: this crate does not depend on
    /// `javelin-semantic`, and a keyword is also what survives a round trip
    /// through the fact cache. Mapping it to a primitive is the server's job,
    /// where the two crates already meet.
    Primitive(String),
    /// Resolved to exactly one classpath type, written with no type
    /// arguments.
    Class(String),
    /// Resolved to exactly one classpath type and written with arguments:
    /// `List<E>`, `Map<String, V>`.
    Parameterized(String, Vec<MemberType>),
    /// A type parameter of the declaring type — never a method's own, which
    /// is [`MemberType::Unresolved`] — so a use's type argument can replace
    /// it.
    Variable(String),
    /// `? extends bound`, as a type argument.
    Extends(Box<MemberType>),
    /// `? super bound`, as a type argument.
    Super(Box<MemberType>),
    /// A method's own type variable, an array, an unbounded `?`, an ambiguous
    /// name, or one nothing reachable declares. The text as written, and
    /// nothing claimed about it.
    Unresolved(String),
}

/// A declaration pass one found, still carrying its supertypes as written and
/// the file context [`simple_name`] needs to resolve them in pass two.
struct RawType {
    fqn: String,
    simple: String,
    kind: DeclKind,
    is_public: bool,
    type_parameters: Vec<String>,
    extends: Option<String>,
    implements: Vec<String>,
    members: Vec<WrittenMember>,
    constructors: Vec<WrittenMember>,
    package: Option<String>,
    imports: Rc<Vec<ImportDecl>>,
    /// The entry parsed without an error node.
    clean: bool,
    /// The type's `/** … */` within `source`.
    doc_span: Option<(u32, u32)>,
    /// The zip entry this declaration was read from — a nested type's is its
    /// outer type's, since `collect_types` passes the same one down.
    source: String,
    /// The type's own name within `source`.
    name_start: u32,
    name_end: u32,
}

/// A member a `.` can select, as written: what [`written_members`] reads off a
/// declaration, with its types still text.
///
/// Public because the JDK's sources are not the only Java this server reads
/// members out of. The workspace walk records the same shape for the
/// workspace's own types, and one reader of what a member *is* — which kinds
/// count, which are static, what `private` drops — is what keeps the two from
/// disagreeing across the seam. Types stay text for the reason a `RawType`
/// carries its supertypes that way: the table they resolve against is not
/// built, or not current, when the member is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenMember {
    pub name: String,
    pub kind: DeclKind,
    pub is_static: bool,
    /// See [`MemberFacts::is_abstract`].
    pub is_abstract: bool,
    /// See [`MemberFacts::is_public`].
    pub is_public: bool,
    /// `None` where the declaration writes no type: an enum constant, whose
    /// type is the enum itself.
    pub declared: Option<String>,
    pub parameters: Vec<String>,
    /// The parameters' names, as written beside their types.
    pub parameter_names: Vec<String>,
    /// The member's `/** … */`, by byte offsets in its file.
    pub doc_span: Option<(u32, u32)>,
    pub variadic: bool,
    /// The member's own name, which is where a jump to it lands.
    pub name_span: javelin_parser::Span,
    /// The whole declaration, which is what a use inside it is inside — the
    /// call hierarchy's question.
    pub span: javelin_parser::Span,
    /// Declared `private`. Kept rather than dropped, because the workspace
    /// walk lists a private member in a symbol search even though no other
    /// file can select it; a reader answering a `.` filters on it.
    pub is_private: bool,
    /// The type parameters a written type of this member can name: its
    /// owner's and, for a generic method, its own. A written type whose name
    /// is one of these is a type variable and must never be resolved through
    /// the package and import tiers — `class Result<Error>`'s `Error error()`
    /// is not `java.lang.Error`.
    ///
    /// ponytail: an enclosing type's parameters are not here, so in the JDK's
    /// sources an inner class's member written as its outer class's `T`
    /// resolves through the tiers — wrongly only where a type called `T` is
    /// also in the package or imported. The server refuses such a name for a
    /// workspace type. The upgrade is to pass the enclosing chain's names down.
    pub type_parameters: Vec<String>,
}

/// The names `<K, V extends Map<K, V>>` introduces: `K` and `V`.
///
/// The comma inside a bound is not a separator, which is why this counts
/// brackets rather than splitting on `,` — the same care
/// `count_type_parameters` takes.
pub fn type_parameter_names(written: Option<&str>) -> Vec<String> {
    let Some(written) = written else {
        return Vec::new();
    };
    let inner = written.trim();
    let inner = inner.strip_prefix('<').unwrap_or(inner);
    let inner = inner.strip_suffix('>').unwrap_or(inner);
    let mut names = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    let mut take = true;
    let mut chars = inner.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                if !current.is_empty() {
                    names.push(std::mem::take(&mut current));
                }
                take = true;
                continue;
            }
            // `<@A T>` declares `T`: the annotation's name, and its
            // arguments if it has any, are not the parameter's.
            '@' if depth == 0 && take && current.is_empty() => {
                while chars
                    .next_if(|c| c.is_alphanumeric() || matches!(c, '_' | '$' | '.'))
                    .is_some()
                {}
                while chars.next_if(|c| c.is_whitespace()).is_some() {}
                if chars.next_if_eq(&'(').is_some() {
                    let mut parens = 1usize;
                    for c in chars.by_ref() {
                        match c {
                            '(' => parens += 1,
                            ')' => parens -= 1,
                            _ => {}
                        }
                        if parens == 0 {
                            break;
                        }
                    }
                }
                continue;
            }
            _ => {}
        }
        if depth == 0 && take {
            if character.is_alphanumeric() || character == '_' || character == '$' {
                current.push(character);
            } else if !current.is_empty() {
                take = false;
            }
        }
    }
    if !current.is_empty() {
        names.push(current);
    }
    names
}

/// The name a written member type needs resolved, if it needs one: the base
/// of `Map.Entry<K, V>` is `Map.Entry`. `None` for `void`, a primitive and an
/// array — the three [`resolve_member_type`] answers without any table.
pub(crate) fn named_type_base(written: &str) -> Option<&str> {
    let written = written.trim();
    if written.is_empty() || written == "void" || PRIMITIVES.contains(&written) || is_array(written)
    {
        return None;
    }
    Some(strip_arguments(written))
}

/// Whether a written type names one of `type_parameters` — a type variable,
/// which no table may resolve.
pub fn names_a_type_parameter(written: &str, type_parameters: &[String]) -> bool {
    named_type_base(written)
        .and_then(|base| base.split('.').next())
        .is_some_and(|first| type_parameters.iter().any(|name| name == first))
}

/// A bare list of the fully-qualified names pass one collected — [`TypeTable`]'s
/// doc comment names this as the src.zip reader's implementation, used only
/// until an on-disk fact store exists.
struct FqnTable(HashSet<String>);

impl TypeTable for FqnTable {
    fn contains(&self, fqn: &str) -> bool {
        self.0.contains(fqn)
    }
}

/// Read every type the sources declare.
///
/// `max_entry_bytes` bounds a single decompressed entry; an entry over the
/// limit — declared or actual, a zip's declared size is not trusted — is
/// skipped rather than read into memory. `cancel` is checked once per zip
/// entry, so a shutdown arriving mid-read stops promptly rather than after the
/// whole of `src.zip` has been parsed.
pub fn read_type_facts(
    src_zip: &Path,
    max_entry_bytes: u64,
    cancel: &CancellationToken,
) -> Result<Vec<TypeFacts>> {
    let file = std::fs::File::open(src_zip).map_err(|source| Error::io(src_zip, source))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| Error::Parse {
        path: src_zip.to_path_buf(),
        message: err.to_string(),
    })?;

    let mut parser = JavaParser::new()?;
    let mut raw_types = Vec::new();

    for index in 0..archive.len() {
        cancel.check()?;

        let entry = match archive.by_index(index) {
            Ok(entry) => entry,
            Err(err) => {
                tracing::debug!(src_zip = %src_zip.display(), index, %err, "unreadable zip entry");
                continue;
            }
        };
        let name = entry.name().to_owned();
        if entry.is_dir() || !is_candidate_source(&name) {
            continue;
        }
        let declared_size = entry.size();
        let text = match read_capped(entry, max_entry_bytes) {
            Ok(text) => text,
            Err(CappedReadError::TooLarge { declared: true }) => {
                tracing::debug!(
                    entry = %name,
                    size = declared_size,
                    max_entry_bytes,
                    "entry too large, skipped"
                );
                continue;
            }
            // Its own line because `size` would be the declared number here,
            // visibly *under* the cap beside it, which reads as a bug in the
            // log rather than as an entry whose metadata lied.
            Err(CappedReadError::TooLarge { declared: false }) => {
                tracing::debug!(
                    entry = %name,
                    declared_size,
                    max_entry_bytes,
                    "entry decompresses to more than it declares, skipped"
                );
                continue;
            }
            Err(CappedReadError::Io(err)) => {
                tracing::debug!(entry = %name, %err, "failed to decompress entry, skipped");
                continue;
            }
            Err(CappedReadError::NotUtf8) => {
                tracing::debug!(entry = %name, "not valid UTF-8, skipped");
                continue;
            }
        };

        let document = TextDocument::new(DocumentId::new(format!("jdk-src:{name}")), 0, text);
        let tree = match parser.parse(&document) {
            Ok(tree) => tree,
            Err(err) => {
                tracing::debug!(entry = %name, %err, "failed to parse, skipped");
                continue;
            }
        };
        let syntax = FileSyntax::build(&document, &tree);
        if syntax.declarations.is_empty() {
            continue;
        }

        let package = syntax.package.map(|package| package.name);
        let imports = Rc::new(syntax.imports);
        let prefix = package.clone().unwrap_or_default();
        collect_types(
            &syntax.declarations,
            &prefix,
            &package,
            &imports,
            (&name, !tree.has_errors()),
            &mut raw_types,
            None,
        );
    }

    let table = FqnTable(raw_types.iter().map(|raw| raw.fqn.clone()).collect());
    Ok(raw_types
        .into_iter()
        .map(|raw| resolve_facts(raw, &table))
        .collect())
}

/// Why [`read_capped`] refused an entry. `read_type_facts`'s loop and
/// [`SourceArchive::read`] each turn this into their own kind of failure — a
/// logged skip for the bulk read, a `javelin_core::Error` for an on-demand
/// one — so the cap and the UTF-8 check live here exactly once.
enum CappedReadError {
    /// Over `max_bytes`. `declared` is whether the zip's own metadata said so;
    /// `false` means the declared size fitted and the capped read caught it
    /// anyway, which is a different thing to report.
    TooLarge {
        declared: bool,
    },
    Io(std::io::Error),
    NotUtf8,
}

/// The size-cap-and-UTF-8 body shared by `read_type_facts`'s loop and
/// [`SourceArchive::read`], so the two limits cannot drift apart. The size a
/// zip entry declares is metadata, not a guarantee; the actual read is capped
/// too, one byte over `max_bytes` is enough to detect it.
fn read_capped(
    mut entry: zip::read::ZipFile<'_, std::fs::File>,
    max_bytes: u64,
) -> std::result::Result<String, CappedReadError> {
    if entry.size() > max_bytes {
        return Err(CappedReadError::TooLarge { declared: true });
    }
    let mut buf = Vec::new();
    let mut limited = (&mut entry).take(max_bytes.saturating_add(1));
    limited.read_to_end(&mut buf).map_err(CappedReadError::Io)?;
    if buf.len() as u64 > max_bytes {
        return Err(CappedReadError::TooLarge { declared: false });
    }
    String::from_utf8(buf).map_err(|_| CappedReadError::NotUtf8)
}

/// One `src.zip` open for as long as a session wants to read entries out of
/// it. Opening the archive reads the whole central directory — tens of
/// thousands of entries for a JDK — so a caller that will ask more than once
/// holds one of these rather than re-opening.
#[derive(Debug)]
pub struct SourceArchive {
    archive: zip::ZipArchive<std::fs::File>,
    path: std::path::PathBuf,
}

impl SourceArchive {
    pub fn open(src_zip: &Path) -> Result<Self> {
        let file = std::fs::File::open(src_zip).map_err(|source| Error::io(src_zip, source))?;
        let archive = zip::ZipArchive::new(file).map_err(|err| Error::Parse {
            path: src_zip.to_path_buf(),
            message: err.to_string(),
        })?;
        Ok(Self {
            archive,
            path: src_zip.to_path_buf(),
        })
    }

    /// The entry's decompressed text, or an error. `max_bytes` bounds the
    /// read exactly as `read_type_facts` bounds one: the declared size is
    /// metadata, so the actual read is capped too.
    pub fn read(&mut self, entry: &str, max_bytes: u64) -> Result<String> {
        let zip_entry = self.archive.by_name(entry).map_err(|err| Error::Parse {
            path: self.path.clone(),
            message: format!("{entry}: {err}"),
        })?;
        read_capped(zip_entry, max_bytes).map_err(|err| Error::Parse {
            path: self.path.clone(),
            message: match err {
                CappedReadError::TooLarge { declared } => {
                    let how = if declared {
                        "declares"
                    } else {
                        "decompresses to"
                    };
                    format!("{entry}: {how} more than the {max_bytes}-byte cap")
                }
                CappedReadError::Io(source) => format!("{entry}: {source}"),
                CappedReadError::NotUtf8 => format!("{entry}: not valid UTF-8"),
            },
        })
    }
}

/// `<module>/<pkg…>/<Name>.java`, minus `module-info.java` and
/// `package-info.java` — neither declares a type.
fn is_candidate_source(entry_name: &str) -> bool {
    let Some(basename) = entry_name.rsplit('/').next() else {
        return false;
    };
    basename.ends_with(".java") && basename != "module-info.java" && basename != "package-info.java"
}

/// Walk declarations, descending only into a type's own children: a nested
/// type is reachable through it, but a method or constructor's children are a
/// local class, which the classpath does not offer, and neither does an
/// anonymous class body, which has no name to reach it by.
///
/// A nested type's fqn joins on `.`, as source writes it: `java.util.Map.Entry`
/// and never `java.util.Map$Entry`. That is the choice `simple_name`'s
/// dotted-name path resolves `Map.Entry` against, so a `.class` or JAR reader
/// in a later phase has to normalise the binary `$` to a `.` rather than
/// storing what the classfile holds.
fn collect_types(
    decls: &[Decl],
    prefix: &str,
    package: &Option<String>,
    imports: &Rc<Vec<ImportDecl>>,
    (entry, clean): (&str, bool),
    out: &mut Vec<RawType>,
    outer: Option<DeclKind>,
) {
    for decl in decls {
        if !is_named_type(decl.kind) {
            continue;
        }
        let fqn = if prefix.is_empty() {
            decl.name.clone()
        } else {
            format!("{prefix}.{}", decl.name)
        };
        out.push(RawType {
            fqn: fqn.clone(),
            simple: decl.name.clone(),
            kind: decl.kind,
            is_public: has_modifier(decl, "public")
                || matches!(outer, Some(DeclKind::Interface | DeclKind::AnnotationType)),
            type_parameters: type_parameter_names(decl.type_parameters.as_deref()),
            extends: decl.extends.clone(),
            implements: decl.implements.clone(),
            // A private member is no `.`'s to select from another type, which
            // is every question these facts answer.
            members: written_members(decl)
                .into_iter()
                .filter(|member| !member.is_private)
                .chain(implicit_members(decl, &fqn))
                .collect(),
            constructors: written_constructors(decl)
                .into_iter()
                .filter(|constructor| !constructor.is_private)
                .collect(),
            package: package.clone(),
            imports: Rc::clone(imports),
            clean,
            doc_span: decl.doc_span,
            source: entry.to_owned(),
            name_start: decl.name_span.start_byte,
            name_end: decl.name_span.end_byte,
        });
        collect_types(
            &decl.children,
            &fqn,
            package,
            imports,
            (entry, clean),
            out,
            Some(decl.kind),
        );
    }
}

/// A declaration with a name the classpath can offer — excludes
/// [`DeclKind::AnonymousClass`], which is recorded as its instantiated type
/// rather than a declaration and is never emitted.
fn is_named_type(kind: DeclKind) -> bool {
    matches!(
        kind,
        DeclKind::Class
            | DeclKind::Interface
            | DeclKind::Enum
            | DeclKind::Record
            | DeclKind::AnnotationType
    )
}

/// The members of one type, read in the same pass that reads the type itself —
/// the parse is the expensive part and it happens once.
///
/// ponytail: `private` is the only accessibility recorded — `is_private`, which
/// readers filter on — so the ceiling is that a `protected` or package-private
/// member is recorded and offered to a
/// type that could not name it. That matches the file-local walk, which does
/// not check accessibility either, so the two answer alike rather than
/// disagreeing across the seam; the upgrade is one accessibility check both
/// walks share, which needs the *asking* file's package and, for `protected`,
/// whether it is a subtype of the owner.
pub fn written_members(owner: &Decl) -> Vec<WrittenMember> {
    let owned = type_parameter_names(owner.type_parameters.as_deref());
    owner
        .children
        .iter()
        .filter(|child| is_member_kind(child.kind))
        .map(|child| written(child, owner, &owned))
        .collect()
}

/// The constructors `owner` writes, where they are all it has
/// (`Decl::writes_its_constructors`), and none otherwise — so a reader
/// holding any holds them all.
pub fn written_constructors(owner: &Decl) -> Vec<WrittenMember> {
    if !owner.writes_its_constructors() {
        return Vec::new();
    }
    let owned = type_parameter_names(owner.type_parameters.as_deref());
    owner
        .children
        .iter()
        .filter(|child| child.kind == DeclKind::Constructor)
        .map(|child| written(child, owner, &owned))
        .collect()
}

/// One declaration `owner` writes, as a member's facts with its types still
/// text.
fn written(child: &Decl, owner: &Decl, owned: &[String]) -> WrittenMember {
    WrittenMember {
        doc_span: child.doc_span,
        name: child.name.clone(),
        kind: child.kind,
        is_static: is_static_member(child, owner.kind),
        is_abstract: is_abstract_member(child, owner.kind),
        is_public: is_public_member(child, owner.kind),
        declared: component_type(child),
        parameters: child
            .parameters
            .iter()
            .map(|parameter| parameter.declared_type.clone().unwrap_or_default())
            .collect(),
        // All or none: error recovery can leave a parameter nameless, and
        // a list with a hole in it would name the wrong positions.
        parameter_names: if child.parameters.iter().all(|it| !it.name.is_empty()) {
            child
                .parameters
                .iter()
                .map(|parameter| parameter.name.clone())
                .collect()
        } else {
            Vec::new()
        },
        variadic: child.parameters.last().is_some_and(|last| last.varargs),
        name_span: child.name_span,
        span: child.span,
        is_private: has_modifier(child, "private"),
        type_parameters: owned
            .iter()
            .cloned()
            .chain(type_parameter_names(child.type_parameters.as_deref()))
            .collect(),
    }
}

/// A field's or a record component's declared type, where `String... names`
/// is a `String[]`.
fn component_type(decl: &Decl) -> Option<String> {
    let written = decl.declared_type.clone()?;
    Some(if decl.varargs {
        format!("{written}[]")
    } else {
        written
    })
}

/// What an enum or a record declares that no one writes (JLS 8.9.3,
/// 8.10.3): an enum's `values()` and `valueOf(String)`, sited as empty spans
/// at the start and the end of its name — where Lombok sites `builder()` and
/// `build()`, which it refuses on an enum — and a record's accessor for each
/// component it writes no zero-argument method of the name for, sited at the
/// component as a Lombok getter is at its field. Types are written fully
/// qualified, so no import or nested type of the file can take them over.
pub fn implicit_members(owner: &Decl, fqn: &str) -> Vec<WrittenMember> {
    let implicit = |name: &str, span: javelin_parser::Span| WrittenMember {
        name: name.to_owned(),
        kind: DeclKind::Method,
        is_static: false,
        is_abstract: false,
        is_public: true,
        declared: None,
        parameters: Vec::new(),
        parameter_names: Vec::new(),
        doc_span: None,
        variadic: false,
        name_span: span,
        span,
        is_private: false,
        type_parameters: Vec::new(),
    };
    let at = |byte: u32, position: javelin_document::Position| javelin_parser::Span {
        start_byte: byte,
        end_byte: byte,
        range: javelin_document::Range::new(position, position),
    };
    let name = owner.name_span;
    match owner.kind {
        DeclKind::Enum => vec![
            WrittenMember {
                is_static: true,
                declared: Some(format!("{fqn}[]")),
                ..implicit("values", at(name.start_byte, name.range.start))
            },
            WrittenMember {
                is_static: true,
                declared: Some(fqn.to_owned()),
                parameters: vec!["java.lang.String".to_owned()],
                parameter_names: vec!["name".to_owned()],
                ..implicit("valueOf", at(name.end_byte, name.range.end))
            },
        ],
        DeclKind::Record => owner
            .children
            .iter()
            .filter(|component| component.kind == DeclKind::RecordComponent)
            .filter(|component| {
                !owner.children.iter().any(|written| {
                    written.kind == DeclKind::Method
                        && written.name == component.name
                        && written.parameters.is_empty()
                })
            })
            .map(|component| WrittenMember {
                declared: component_type(component),
                type_parameters: type_parameter_names(owner.type_parameters.as_deref()),
                span: component.name_span,
                ..implicit(&component.name, component.name_span)
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Which of a type's children another file could select through a `.`.
///
/// Deliberately the same list as `javelin_semantic::members::is_member`, minus
/// the type kinds: a nested type is a [`TypeFacts`] row of its own, reached by
/// its own name, so recording it as a member here as well would offer it
/// twice. A constructor, an initialiser block, a local and a parameter are
/// absent from both lists.
fn is_member_kind(kind: DeclKind) -> bool {
    matches!(
        kind,
        DeclKind::Field
            | DeclKind::Method
            | DeclKind::EnumConstant
            | DeclKind::RecordComponent
            | DeclKind::AnnotationElement
    )
}

/// The same rules `javelin_semantic::members::is_static` applies, minus the
/// nested-type one, which cannot arise because a nested type is not recorded
/// as a member: the keyword, an enum constant, and a field or annotation
/// element of an interface or annotation type being implicitly static.
fn is_static_member(decl: &Decl, owner_kind: DeclKind) -> bool {
    if decl.kind == DeclKind::EnumConstant {
        return true;
    }
    if matches!(owner_kind, DeclKind::Interface | DeclKind::AnnotationType)
        && matches!(decl.kind, DeclKind::Field | DeclKind::AnnotationElement)
    {
        return true;
    }
    has_modifier(decl, "static")
}

/// See [`MemberFacts::is_abstract`]. An interface method with a body says so
/// with one of three keywords, because without one it could not have one.
fn is_abstract_member(decl: &Decl, owner_kind: DeclKind) -> bool {
    if decl.kind != DeclKind::Method {
        return false;
    }
    has_modifier(decl, "abstract")
        || (owner_kind == DeclKind::Interface
            && !["default", "static", "private"]
                .iter()
                .any(|keyword| has_modifier(decl, keyword)))
}

/// See [`MemberFacts::is_public`].
fn is_public_member(decl: &Decl, owner_kind: DeclKind) -> bool {
    has_modifier(decl, "public")
        || decl.kind == DeclKind::EnumConstant
        || (matches!(owner_kind, DeclKind::Interface | DeclKind::AnnotationType)
            && !has_modifier(decl, "private"))
}

/// `modifiers` holds annotations as written alongside the keywords, so this is
/// an exact match rather than a substring one.
fn has_modifier(decl: &Decl, keyword: &str) -> bool {
    decl.modifiers.iter().any(|modifier| modifier == keyword)
}

/// The type variables a written type can name where it is written: the
/// declaring type's, which become [`MemberType::Variable`], and a generic
/// method's own, which shadow them and stay unresolved.
pub(crate) struct InScope<'a> {
    pub(crate) owner: &'a [String],
    pub(crate) own: &'a [String],
}

/// How deep type arguments are followed before the rest is given up on — a
/// real JDK nests a handful, and the text is not trusted to stop.
const MAX_ARGUMENT_DEPTH: u32 = 16;

/// A written type with its arguments, resolved against the same table and the
/// same [`simple_name`] rule [`resolve_member_type`] uses.
fn resolve_generic(
    written: &str,
    scope: &InScope<'_>,
    package: Option<&str>,
    imports: &[ImportDecl],
    table: &dyn TypeTable,
    depth: u32,
) -> MemberType {
    let base = |name: &str, _: bool| match simple_name(name, package, imports, table) {
        Resolution::One(fqn) => Some(fqn),
        Resolution::Ambiguous | Resolution::Unknown => None,
    };
    generic(written, scope, &base, depth)
}

/// A written type with its arguments, where `owner` names the declaring
/// type's variables and `own` a generic method's, and `base` resolves every
/// other name it spells, at any depth, told whether the name is inside a type
/// argument — the reader the JDK's sources go through, for a caller with
/// resolution rules of its own: a workspace file's own declarations ahead of
/// its package and imports.
pub fn resolve_generic_with(
    written: &str,
    owner: &[String],
    own: &[String],
    base: &dyn Fn(&str, bool) -> Option<String>,
) -> MemberType {
    generic(written, &InScope { owner, own }, base, 0)
}

fn generic(
    written: &str,
    scope: &InScope<'_>,
    resolve: &dyn Fn(&str, bool) -> Option<String>,
    depth: u32,
) -> MemberType {
    let written = strip_leading_annotations(written.trim());
    let unresolved = || MemberType::Unresolved(written.to_owned());
    if let Some(rest) = written.strip_prefix('?') {
        let rest = rest.trim_start();
        let bound = |keyword: &str| {
            rest.strip_prefix(keyword)
                .filter(|bound| bound.starts_with(char::is_whitespace))
        };
        let next = depth + 1;
        return match (bound("extends"), bound("super")) {
            (Some(bound), _) if next < MAX_ARGUMENT_DEPTH => {
                MemberType::Extends(Box::new(generic(bound, scope, resolve, next)))
            }
            (_, Some(bound)) if next < MAX_ARGUMENT_DEPTH => {
                MemberType::Super(Box::new(generic(bound, scope, resolve, next)))
            }
            _ => unresolved(),
        };
    }
    if written.is_empty() {
        return MemberType::Unresolved(String::new());
    }
    if written == "void" {
        return MemberType::Void;
    }
    if PRIMITIVES.contains(&written) {
        return MemberType::Primitive(written.to_owned());
    }
    if is_array(written) {
        return unresolved();
    }
    let (base, arguments) = match trailing_arguments(written) {
        Some((base, arguments)) => (base.trim(), Some(arguments)),
        None => (written, None),
    };
    // `Outer<A>.Inner` is left with its arguments dropped, as before.
    let base_name = written_type_name(base);
    let first = base_name.split('.').next().unwrap_or(&base_name);
    if scope.own.iter().any(|name| name == first) {
        return unresolved();
    }
    if scope.owner.iter().any(|name| name == first) {
        return if base_name == first && arguments.is_none() {
            MemberType::Variable(base_name)
        } else {
            unresolved()
        };
    }
    let Some(fqn) = resolve(&base_name, depth > 0) else {
        return unresolved();
    };
    match arguments {
        Some(arguments) if depth + 1 < MAX_ARGUMENT_DEPTH => MemberType::Parameterized(
            fqn,
            split_arguments(arguments)
                .into_iter()
                .map(|argument| generic(argument, scope, resolve, depth + 1))
                .collect(),
        ),
        _ => MemberType::Class(fqn),
    }
}

/// `List` and `String, V` out of `List<String, V>` — only when the first `<`
/// is closed by the last character, so `Outer<A>.Inner` has none.
pub(crate) fn trailing_arguments(written: &str) -> Option<(&str, &str)> {
    let open = written.find('<')?;
    let mut depth = 0i32;
    for (at, character) in written.char_indices().skip_while(|(at, _)| *at < open) {
        match character {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return (at + 1 == written.len())
                        .then(|| (&written[..open], &written[open + 1..at]));
                }
            }
            _ => {}
        }
    }
    None
}

/// Split `String, Map<K, V>` on the commas outside brackets.
pub(crate) fn split_arguments(inner: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (at, character) in inner.char_indices() {
        match character {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&inner[start..at]);
                start = at + 1;
            }
            _ => {}
        }
    }
    out.push(&inner[start..]);
    out.into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect()
}

/// `@A @B(x = 1) Foo` is a `Foo`.
fn strip_leading_annotations(mut written: &str) -> &str {
    while let Some(after) = written.strip_prefix('@') {
        let name_end = after
            .find(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '$' | '.')))
            .unwrap_or(after.len());
        let mut rest = after[name_end..].trim_start();
        if rest.starts_with('(') {
            let mut parens = 0usize;
            let mut end = rest.len();
            for (at, c) in rest.char_indices() {
                match c {
                    '(' => parens += 1,
                    ')' => {
                        parens -= 1;
                        if parens == 0 {
                            end = at + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            rest = rest[end..].trim_start();
        }
        written = rest;
    }
    written
}

/// Strip a written supertype's type arguments before resolving it: 3a compares
/// declarations, not instantiations, so `AbstractList<E>` resolves as
/// `AbstractList`.
/// A written type as the name it resolves by: type arguments removed from
/// every segment — `Outer<String>.Inner` is `Outer.Inner`, where cutting at
/// the first `<` would make it `Outer` — and type annotations dropped.
pub fn written_type_name(written: &str) -> String {
    let mut name = String::with_capacity(written.len());
    let mut chars = written.chars().peekable();
    let mut depth = 0usize;
    while let Some(c) = chars.next() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth > 0 => {}
            '@' => {
                while chars
                    .next_if(|c| c.is_alphanumeric() || matches!(c, '_' | '$' | '.'))
                    .is_some()
                {}
                while chars.next_if(|c| c.is_whitespace()).is_some() {}
                if chars.next_if_eq(&'(').is_some() {
                    let mut parens = 1usize;
                    for c in chars.by_ref() {
                        match c {
                            '(' => parens += 1,
                            ')' => parens -= 1,
                            _ => {}
                        }
                        if parens == 0 {
                            break;
                        }
                    }
                }
            }
            c if c.is_whitespace() => {}
            c => name.push(c),
        }
    }
    name
}

fn strip_arguments(written: &str) -> &str {
    written.split('<').next().unwrap_or(written).trim()
}

/// The keywords that are a type on their own. `void` is not among them: it is
/// the absence of one, and [`MemberType`] keeps them apart.
const PRIMITIVES: [&str; 8] = [
    "byte", "short", "int", "long", "char", "float", "double", "boolean",
];

/// Whether a written type is an array of something.
///
/// The `[]` has to be looked for *outside* the type arguments, in the text
/// after the last `>`: `Map<String, int[]>` is a map and `List<String>[]` is
/// an array, and stripping the arguments first would confuse the two.
fn is_array(written: &str) -> bool {
    let tail = match written.rfind('>') {
        Some(at) => &written[at + 1..],
        None => written,
    };
    tail.contains('[')
}

fn resolve_member(member: WrittenMember, raw: &RawType, table: &FqnTable) -> MemberFacts {
    // An enum constant's type is the enum, which nobody writes down;
    // `javelin-semantic` answers a local one the same way.
    // `WrittenMember::type_parameters` is the owner's followed by the
    // method's own.
    let scope = InScope {
        owner: &raw.type_parameters,
        own: member
            .type_parameters
            .get(raw.type_parameters.len()..)
            .unwrap_or_default(),
    };
    let resolve = |written: Option<&str>| match written {
        Some(written) => resolve_generic(
            written,
            &scope,
            raw.package.as_deref(),
            &raw.imports,
            table,
            0,
        ),
        None => resolve_member_type(None, raw.package.as_deref(), &raw.imports, table),
    };
    let declared = if member.kind == DeclKind::EnumConstant {
        MemberType::Class(raw.fqn.clone())
    } else {
        resolve(member.declared.as_deref())
    };
    MemberFacts {
        type_parameters: scope.own.to_vec(),
        doc_span: member.doc_span,
        declared,
        parameters: member
            .parameters
            .iter()
            .map(|written| resolve(Some(written)))
            .collect(),
        parameter_names: member.parameter_names,
        name: member.name,
        kind: member.kind,
        is_static: member.is_static,
        is_abstract: member.is_abstract,
        is_public: member.is_public,
        variadic: member.variadic,
        name_start: member.name_span.start_byte,
        name_end: member.name_span.end_byte,
    }
}

/// Resolve one written member type against the same table and the same
/// [`simple_name`] rule a supertype goes through, so a member type and a
/// supertype can never disagree about what a name means in this file.
///
/// `package` and `imports` are the declaring file's.
pub(crate) fn resolve_member_type(
    written: Option<&str>,
    package: Option<&str>,
    imports: &[ImportDecl],
    table: &dyn TypeTable,
) -> MemberType {
    // Nothing written is nothing to resolve, and `Void` would claim the member
    // has no type rather than that this reader could not find one.
    let Some(written) = written.map(str::trim) else {
        return MemberType::Unresolved(String::new());
    };
    if written == "void" {
        return MemberType::Void;
    }
    if PRIMITIVES.contains(&written) {
        return MemberType::Primitive(written.to_owned());
    }
    // ponytail: the ceiling is that an array member is unresolved rather than
    // an array of its element type, so nothing can be selected on it. Claiming
    // the element type would be worse than refusing — it would answer
    // `String`'s members on a `String[]`. `javelin-semantic` already has a
    // `Type::Array`, so the upgrade is an array depth beside the element type
    // here, which is enough to rebuild one.
    if is_array(written) {
        return MemberType::Unresolved(written.to_owned());
    }
    match simple_name(strip_arguments(written), package, imports, table) {
        Resolution::One(fqn) => with_written_arguments(fqn, written),
        Resolution::Ambiguous | Resolution::Unknown => MemberType::Unresolved(written.to_owned()),
    }
}

/// `fqn`, which `written` resolved to, keeping that it was written with
/// arguments: a `List<String>` parameter read as a raw `List` would let a
/// `List<Integer>` argument prove it the most specific. The arguments
/// themselves stay unresolved, which proves nothing about them, and
/// `Outer<A>.Inner` — arguments not at the end — is no `Outer` at all.
pub(crate) fn with_written_arguments(fqn: String, written: &str) -> MemberType {
    if !written.contains('<') {
        return MemberType::Class(fqn);
    }
    match trailing_arguments(written.trim()) {
        Some((_, arguments)) => MemberType::Parameterized(
            fqn,
            split_arguments(arguments)
                .into_iter()
                .map(|argument| MemberType::Unresolved(argument.trim().to_owned()))
                .collect(),
        ),
        None => MemberType::Unresolved(written.to_owned()),
    }
}

fn resolve_facts(mut raw: RawType, table: &FqnTable) -> TypeFacts {
    // Taken out first because resolving a member borrows the rest of `raw` for
    // its package and imports.
    let members = std::mem::take(&mut raw.members)
        .into_iter()
        .map(|member| resolve_member(member, &raw, table))
        .collect();
    let constructors = std::mem::take(&mut raw.constructors)
        .into_iter()
        .map(|constructor| resolve_member(constructor, &raw, table))
        .collect();

    let written_supertypes = raw
        .extends
        .iter()
        .cloned()
        .chain(raw.implements.iter().cloned());

    let mut supertypes = Vec::new();
    let mut supertype_arguments = Vec::new();
    let mut supertypes_complete = true;
    let scope = InScope {
        owner: &raw.type_parameters,
        own: &[],
    };
    for written in written_supertypes {
        let stripped = written_type_name(&written);
        let stripped = stripped.as_str();
        match simple_name(stripped, raw.package.as_deref(), &raw.imports, table) {
            Resolution::One(fqn) => {
                if let MemberType::Parameterized(written_fqn, arguments) = resolve_generic(
                    &written,
                    &scope,
                    raw.package.as_deref(),
                    &raw.imports,
                    table,
                    0,
                ) && written_fqn == fqn
                {
                    supertype_arguments.push((fqn.clone(), arguments));
                }
                supertypes.push(fqn);
            }
            Resolution::Ambiguous | Resolution::Unknown => supertypes_complete = false,
        }
    }

    // JLS 8.9, 8.10 and 9.6 give an enum `java.lang.Enum`, a record
    // `java.lang.Record` and an annotation type
    // `java.lang.annotation.Annotation`, none of it written down. That the
    // *named* supertype is itself missing from the closure is already the
    // whole reason: draining it to `Some(false)` would separate a
    // `@interface` from `java.lang.annotation.Annotation`. For an enum it is
    // more than the one name — `java.lang.Enum` implements `Comparable` and
    // `Serializable`, so an enum is both without saying so — where
    // `java.lang.Record` implements nothing and
    // `java.lang.annotation.Annotation` extends nothing, and one missing name
    // is enough on its own. A class's and an interface's unwritten supertype
    // is `java.lang.Object` alone, which `TypeWorld::is_subtype` answers
    // without a fact.
    //
    // ponytail: the ceiling is that one of these three, *as the argument*,
    // excludes no external parameter type — the same one `apply.rs`'s
    // `may_be_subtype` records for a local declaration. As a parameter type it
    // is still separable, because the closure a walk drains belongs to the
    // argument. The upgrade is to synthesise the named supertype — it is a
    // fact this world holds — and leave the closure complete.
    if matches!(
        raw.kind,
        DeclKind::Enum | DeclKind::Record | DeclKind::AnnotationType
    ) {
        supertypes_complete = false;
    }

    // A class with no written `extends` implicitly extends Object; making that
    // explicit lets a closure walk reach it. Object itself is the base of that
    // rule, not an instance of it: its own source is `public class Object {`
    // with no `extends`, so without this exception it would be synthesised as
    // its own supertype.
    if raw.kind == DeclKind::Class && raw.extends.is_none() && raw.fqn != "java.lang.Object" {
        supertypes.push("java.lang.Object".to_owned());
    }

    TypeFacts {
        doc_span: raw.doc_span,
        name_start: raw.name_start,
        name_end: raw.name_end,
        fqn: raw.fqn,
        simple: raw.simple,
        kind: raw.kind,
        is_public: raw.is_public,
        type_parameters: raw.type_parameters,
        source: raw.source,
        supertypes,
        supertype_arguments,
        supertypes_complete,
        members_complete: raw.clean,
        members,
        constructors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join("javelin-classpath-tests")
            .join("sources")
            .join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Build a `src.zip` at `dir/src.zip` from `(entry_name, contents)` pairs,
    /// stored uncompressed — the reader never cares which method wrote a
    /// deflate-only crate's archive, only that it can read one back.
    fn write_zip(dir: &std::path::Path, entries: &[(&str, &str)]) -> std::path::PathBuf {
        let path = dir.join("src.zip");
        let file = fs::File::create(&path).expect("zip file");
        let mut writer = zip::ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, contents) in entries {
            writer.start_file(*name, options).expect("start entry");
            writer.write_all(contents.as_bytes()).expect("write entry");
        }
        writer.finish().expect("finish zip");
        path
    }

    const NO_LIMIT: u64 = u64::MAX;

    fn read(zip_path: &std::path::Path) -> Result<Vec<TypeFacts>> {
        read_type_facts(zip_path, NO_LIMIT, &CancellationToken::never())
    }

    fn syntax_of(source: &str) -> FileSyntax {
        let document = javelin_document::TextDocument::new(
            javelin_document::DocumentId::new("file:///A.java"),
            1,
            source,
        );
        let tree = JavaParser::new()
            .expect("grammar")
            .parse(&document)
            .expect("parses");
        FileSyntax::build(&document, &tree)
    }

    #[test]
    fn a_written_type_name_loses_its_arguments_per_segment_and_its_annotations() {
        assert_eq!(written_type_name("Outer<String>.Inner"), "Outer.Inner");
        assert_eq!(written_type_name("Map.Entry<K, V>"), "Map.Entry");
        assert_eq!(written_type_name("@A Foo"), "Foo");
        assert_eq!(
            written_type_name("@A(x = 1) com.x.Foo<List<T>>"),
            "com.x.Foo"
        );
        assert_eq!(written_type_name(" Plain "), "Plain");
    }

    /// JLS 8.9.3 and 8.10.3: what an enum and a record declare that no one
    /// writes, each at a site no other member has.
    #[test]
    fn an_enum_and_a_record_declare_members_nobody_writes() {
        let line = |member: &WrittenMember| {
            format!(
                "{}{}({}): {} @{}..{}",
                if member.is_static { "static " } else { "" },
                member.name,
                member.parameters.join(", "),
                member.declared.as_deref().unwrap_or(""),
                member.name_span.start_byte,
                member.name_span.end_byte,
            )
        };
        let syntax = syntax_of("enum Day { MON }");
        let implicit: Vec<String> = implicit_members(&syntax.declarations[0], "p.Day")
            .iter()
            .map(line)
            .collect();
        assert_eq!(
            implicit,
            vec![
                "static values(): p.Day[] @5..5",
                "static valueOf(java.lang.String): p.Day @8..8",
            ]
        );
        let syntax = syntax_of(
            "record Entry(String date, int n) { public int n() { return 1; } \
             int date(int x) { return x; } }",
        );
        let implicit: Vec<String> = implicit_members(&syntax.declarations[0], "p.Entry")
            .iter()
            .map(line)
            .collect();
        assert_eq!(
            implicit,
            vec!["date(): String @20..24"],
            "`n()` is written, and `date(int)` is no accessor"
        );
        let syntax = syntax_of("record Args(String... names) {}");
        let record = &syntax.declarations[0];
        assert_eq!(
            implicit_members(record, "p.Args")[0].declared.as_deref(),
            Some("String[]"),
            "a variadic component is an array"
        );
        assert_eq!(
            written_members(record)[0].declared.as_deref(),
            Some("String[]")
        );
        let syntax = syntax_of("class A { int a; }");
        assert!(implicit_members(&syntax.declarations[0], "p.A").is_empty());
    }

    /// A member's and a type's Javadoc are read for where they are.
    #[test]
    fn a_doc_comments_span_is_read_beside_the_name() {
        let syntax = syntax_of("class A { /** Counts. */ int count() { return 0; } int bare; }");
        let members = written_members(&syntax.declarations[0]);
        let (start, end) = members[0].doc_span.expect("a doc");
        assert_eq!(end - start, "/** Counts. */".len() as u32);
        assert_eq!(members[1].doc_span, None);
    }

    /// Error recovery can leave a parameter with a type and no name; then no
    /// parameter is named, rather than the rest named in the wrong places.
    #[test]
    fn a_nameless_parameter_names_none_of_them() {
        let syntax = syntax_of("class A { void f(int a, String) {} void g(int b) {} }");
        let members = written_members(&syntax.declarations[0]);
        assert!(members[0].parameter_names.is_empty());
        assert_eq!(members[1].parameter_names, vec!["b"]);
    }

    #[test]
    fn written_members_flag_private_and_read_types_as_written() {
        let syntax = syntax_of(
            "class A<T> { private int hidden; public List<String> names; \
             static <U> void go(U n, String... rest) {} A() {} class Nested {} }",
        );
        let members = written_members(&syntax.declarations[0]);
        assert_eq!(
            members.last().map(|go| go.parameter_names.clone()),
            Some(vec!["n".to_owned(), "rest".to_owned()]),
            "a parameter's name is read beside its type"
        );
        // Name, private, static, declared, parameters, variadic, type parameters.
        type Row<'a> = (
            &'a str,
            bool,
            bool,
            Option<&'a str>,
            Vec<&'a str>,
            bool,
            Vec<&'a str>,
        );
        let summary: Vec<Row<'_>> = members
            .iter()
            .map(|member| {
                (
                    member.name.as_str(),
                    member.is_private,
                    member.is_static,
                    member.declared.as_deref(),
                    member.parameters.iter().map(String::as_str).collect(),
                    member.variadic,
                    member.type_parameters.iter().map(String::as_str).collect(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                ("hidden", true, false, Some("int"), vec![], false, vec!["T"]),
                (
                    "names",
                    false,
                    false,
                    Some("List<String>"),
                    vec![],
                    false,
                    vec!["T"]
                ),
                (
                    "go",
                    false,
                    true,
                    Some("void"),
                    vec!["U", "String"],
                    true,
                    vec!["T", "U"]
                ),
            ]
        );
        let names = &members[1];
        assert_eq!(
            names.name_span.start_byte,
            "class A<T> { private int hidden; public List<String> ".len() as u32
        );
    }

    #[test]
    fn type_parameter_names_skip_bounds_and_their_commas() {
        assert_eq!(
            type_parameter_names(Some("<K, V extends Map<K, V>, E extends Enum<E>>")),
            vec!["K", "V", "E"]
        );
        assert!(type_parameter_names(None).is_empty());
        assert!(!names_a_type_parameter("List<V>[]", &["List".to_owned()]));
        assert!(names_a_type_parameter("V", &["V".to_owned()]));
        assert!(names_a_type_parameter("V.Inner<String>", &["V".to_owned()]));
        assert!(!names_a_type_parameter("int", &["int".to_owned()]));
    }

    #[test]
    fn a_jdk_members_type_parameter_is_never_resolved_through_the_tiers() {
        // `Error` is the class's own type parameter; `java.lang.Error` is in
        // the same read, and a tier lookup would find it.
        let dir = scratch("type_parameter_member");
        let zip = write_zip(
            &dir,
            &[
                (
                    "java.base/java/lang/Error.java",
                    "package java.lang;\npublic class Error {}\n",
                ),
                (
                    "java.base/java/util/Result.java",
                    "package java.util;\npublic class Result<Error> { public Error error() { return null; } }\n",
                ),
            ],
        );
        let facts = read(&zip).expect("read");
        let result = fact(&facts, "java.util.Result");
        assert_eq!(
            result.members[0].declared,
            MemberType::Variable("Error".to_owned())
        );
    }

    #[test]
    fn resolve_member_type_resolves_through_the_given_table() {
        let table = FqnTable(
            ["java.util.List", "p.Local"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
        let imports = syntax_of("import java.util.List; class A {}").imports;
        assert_eq!(
            resolve_member_type(Some("List<String>"), Some("p"), &imports, &table),
            MemberType::Parameterized(
                "java.util.List".to_owned(),
                vec![MemberType::Unresolved("String".to_owned())]
            )
        );
        assert_eq!(
            resolve_member_type(Some("List<String>.Inner"), Some("p"), &imports, &table),
            MemberType::Unresolved("List<String>.Inner".to_owned())
        );
        assert_eq!(
            resolve_member_type(Some("Local"), Some("p"), &imports, &table),
            MemberType::Class("p.Local".to_owned())
        );
        assert_eq!(
            resolve_member_type(Some("int"), Some("p"), &imports, &table),
            MemberType::Primitive("int".to_owned())
        );
        assert_eq!(
            resolve_member_type(Some("Nowhere"), Some("p"), &imports, &table),
            MemberType::Unresolved("Nowhere".to_owned())
        );
    }

    fn fact<'a>(facts: &'a [TypeFacts], fqn: &str) -> &'a TypeFacts {
        facts
            .iter()
            .find(|f| f.fqn == fqn)
            .unwrap_or_else(|| panic!("no fact for {fqn}, got {facts:#?}"))
    }

    #[test]
    fn a_type_in_the_zip_is_read_with_its_package_and_kind() {
        let dir = scratch("basic");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/java/util/Widget.java",
                "package java.util;\npublic class Widget {}\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "java.util.Widget");
        assert_eq!(widget.simple, "Widget");
        assert_eq!(widget.kind, DeclKind::Class);
        assert!(widget.type_parameters.is_empty());
    }

    #[test]
    fn a_nested_type_is_read_with_a_dotted_fully_qualified_name() {
        let dir = scratch("nested");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/java/util/Map.java",
                "package java.util;\npublic interface Map<K, V> { interface Entry<K, V> {} }",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let entry = fact(&facts, "java.util.Map.Entry");
        assert_eq!(entry.simple, "Entry");
        assert_eq!(entry.type_parameters, vec!["K", "V"]);
    }

    /// Error recovery can drop a member the entry declares: a caller that
    /// claims a member is absent must know the list is short. An enum's
    /// `values()` and a record's accessors are read as its members.
    #[test]
    fn members_are_whole_only_for_a_clean_entry() {
        let dir = scratch("members_complete");
        let zip_path = write_zip(
            &dir,
            &[
                (
                    "java.base/java/lang/Clean.java",
                    "package java.lang;\npublic class Clean { public void a() {} }\n",
                ),
                (
                    "java.base/java/lang/Broken.java",
                    "package java.lang;\npublic class Broken { public void a( {} \
                     public void b() {} }\n",
                ),
                (
                    "java.base/java/lang/Level.java",
                    "package java.lang;\npublic enum Level { LOW }\n",
                ),
                (
                    "java.base/java/lang/Point.java",
                    "package java.lang;\npublic record Point(int x) {}\n",
                ),
            ],
        );
        let facts = read(&zip_path).expect("read");
        for fqn in ["java.lang.Clean", "java.lang.Level", "java.lang.Point"] {
            assert!(fact(&facts, fqn).members_complete, "{fqn}");
        }
        assert!(!fact(&facts, "java.lang.Broken").members_complete);
        let names = |fqn| {
            let mut names: Vec<String> = fact(&facts, fqn)
                .members
                .iter()
                .map(|member| format!("{}: {:?}", member.name, member.declared))
                .collect();
            names.sort();
            names
        };
        assert_eq!(
            names("java.lang.Level"),
            vec![
                "LOW: Class(\"java.lang.Level\")",
                "valueOf: Class(\"java.lang.Level\")",
                "values: Unresolved(\"java.lang.Level[]\")",
            ]
        );
        assert_eq!(
            names("java.lang.Point"),
            vec!["x: Primitive(\"int\")", "x: Primitive(\"int\")"],
            "the component and its accessor"
        );
    }

    #[test]
    fn a_supertype_named_by_an_import_resolves_to_its_fully_qualified_name() {
        let dir = scratch("import_resolves");
        let zip_path = write_zip(
            &dir,
            &[
                (
                    "java.base/java/util/AbstractList.java",
                    "package java.util;\npublic class AbstractList<E> {}\n",
                ),
                (
                    "java.base/java/util/ArrayList.java",
                    "package java.util;\n\
                     public class ArrayList<E> extends AbstractList<E> {}\n",
                ),
            ],
        );

        let facts = read(&zip_path).expect("read");
        let array_list = fact(&facts, "java.util.ArrayList");
        assert_eq!(array_list.supertypes, vec!["java.util.AbstractList"]);
        assert!(array_list.supertypes_complete);
    }

    #[test]
    fn a_supertype_that_does_not_resolve_is_dropped_and_marks_the_closure_incomplete() {
        let dir = scratch("unresolved_supertype");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\npublic class Widget extends Nowhere {}\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert!(widget.supertypes.is_empty());
        assert!(!widget.supertypes_complete);
    }

    #[test]
    fn a_class_with_no_extends_gets_object_as_a_supertype_but_an_interface_does_not() {
        let dir = scratch("implicit_object");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Shapes.java",
                "package com.example;\n\
                 public class Widget {}\n\
                 interface Gadget {}\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert_eq!(widget.supertypes, vec!["java.lang.Object"]);
        assert!(widget.supertypes_complete);

        let gadget = fact(&facts, "com.example.Gadget");
        assert!(gadget.supertypes.is_empty());
        assert!(gadget.supertypes_complete);
    }

    #[test]
    fn an_enums_closure_is_never_complete_because_its_supertype_is_unwritten() {
        // JLS 8.9, 8.10 and 9.6 give these three a *named* supertype nobody
        // writes down, and that name's own absence from the closure is enough:
        // `java.lang.Record` implements nothing and
        // `java.lang.annotation.Annotation` extends nothing, so for those two
        // the one missing name is the whole gap, while `java.lang.Enum` brings
        // `Comparable` and `Serializable` with it as well. A closure missing
        // part of itself must not be walked to `Some(false)`, so the fact says
        // up front that it is not whole.
        let dir = scratch("implicit_named_supertypes");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Kinds.java",
                "package com.example;\n\
                 public enum Colour { RED }\n\
                 record Point(int x) {}\n\
                 @interface Marker {}\n\
                 interface Plain {}\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        for name in ["Colour", "Point", "Marker"] {
            let found = fact(&facts, &format!("com.example.{name}"));
            assert!(
                found.supertypes.is_empty() && !found.supertypes_complete,
                "{name}: nothing to walk and nothing claimed, got {found:?}"
            );
        }
        // The interface is not in that group: it has no implicit *named*
        // supertype at all, and `Object` is answered by the world rather than
        // written into the fact.
        let plain = fact(&facts, "com.example.Plain");
        assert!(plain.supertypes.is_empty());
        assert!(plain.supertypes_complete);
    }

    #[test]
    fn java_lang_object_is_not_its_own_supertype() {
        let dir = scratch("object_is_not_its_own_supertype");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/java/lang/Object.java",
                "package java.lang;\npublic class Object {}\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let object = fact(&facts, "java.lang.Object");
        assert!(object.supertypes.is_empty());
        assert!(object.supertypes_complete);
    }

    #[test]
    fn an_unparseable_entry_is_skipped_rather_than_failing_the_read() {
        let dir = scratch("unparseable");
        let mut garbage = vec![0xffu8, 0xfe, 0x00, 0x01, 0x02, 0xd8, 0x00];
        garbage.extend_from_slice(b"\xC0\xAF not valid utf-8 at all");
        let path = dir.join("src.zip");
        let file = fs::File::create(&path).expect("zip file");
        let mut writer = zip::ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        writer
            .start_file("java.base/com/example/Garbage.java", options)
            .expect("start entry");
        writer.write_all(&garbage).expect("write entry");
        writer
            .start_file(
                "java.base/com/example/Fine.java",
                SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
            )
            .expect("start entry");
        writer
            .write_all(b"package com.example;\npublic class Fine {}\n")
            .expect("write entry");
        writer.finish().expect("finish zip");

        let facts = read(&path).expect("read");
        assert!(facts.iter().any(|f| f.fqn == "com.example.Fine"));
        assert!(!facts.iter().any(|f| f.fqn.contains("Garbage")));
    }

    #[test]
    fn module_info_and_package_info_contribute_nothing() {
        let dir = scratch("module_and_package_info");
        let zip_path = write_zip(
            &dir,
            &[
                ("java.base/module-info.java", "module java.base {}\n"),
                (
                    "java.base/com/example/package-info.java",
                    "/** docs */\npackage com.example;\n",
                ),
                (
                    "java.base/com/example/Real.java",
                    "package com.example;\npublic class Real {}\n",
                ),
            ],
        );

        let facts = read(&zip_path).expect("read");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].fqn, "com.example.Real");
    }

    #[test]
    fn a_local_class_inside_a_method_is_not_a_type_fact() {
        let dir = scratch("local_class");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Outer.java",
                "package com.example;\n\
                 public class Outer {\n\
                 void m() { class Local {} }\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].fqn, "com.example.Outer");
    }

    #[test]
    fn cancellation_stops_the_read() {
        let dir = scratch("cancellation");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\npublic class Widget {}\n",
            )],
        );

        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = read_type_facts(&zip_path, NO_LIMIT, &cancel);
        assert!(matches!(result, Err(Error::Cancelled)));
    }

    // Only the declared-size check (`entry.size() > max_entry_bytes`) is
    // exercised here: hitting the second, actual-read cap would need an entry
    // whose central-directory size lies about its content, which the `zip`
    // crate's writer will not produce — it always records the real length of
    // what was written. Faking that would mean hand-assembling zip bytes
    // rather than writing a real archive, which is the kind of contortion
    // this test is better off not attempting.
    #[test]
    fn an_entry_larger_than_the_limit_is_skipped_rather_than_read() {
        let dir = scratch("entry_too_large");
        let huge_content = format!("/* {} */\nclass Huge {{}}\n", "x".repeat(100));
        let small_content = "class Small{}\n";
        assert!(huge_content.len() > 50 && small_content.len() < 50);
        let zip_path = write_zip(
            &dir,
            &[
                ("java.base/com/example/Huge.java", &huge_content),
                ("java.base/com/example/Small.java", small_content),
            ],
        );

        let facts = read_type_facts(&zip_path, 50, &CancellationToken::never()).expect("read");

        assert!(!facts.iter().any(|f| f.fqn.contains("Huge")));
        assert!(facts.iter().any(|f| f.fqn == "Small"));
    }

    #[test]
    fn a_corrupt_archive_answers_an_error_rather_than_panicking() {
        let dir = scratch("corrupt_archive");
        let path = dir.join("src.zip");
        fs::write(&path, b"this is not a zip file, just garbage bytes").expect("write garbage");

        let result = read_type_facts(&path, NO_LIMIT, &CancellationToken::never());
        assert!(result.is_err());
    }

    fn member<'a>(facts: &'a TypeFacts, name: &str) -> &'a MemberFacts {
        facts
            .members
            .iter()
            .find(|m| m.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "no member {name} on {}, got {:#?}",
                    facts.fqn, facts.members
                )
            })
    }

    #[test]
    fn a_types_fields_and_methods_are_recorded_as_members() {
        let dir = scratch("members_recorded");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 int count;\n\
                 void reset() {}\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert_eq!(member(widget, "count").kind, DeclKind::Field);
        assert_eq!(member(widget, "reset").kind, DeclKind::Method);
        assert_eq!(widget.members.len(), 2);
    }

    #[test]
    fn a_private_member_is_not_recorded() {
        let dir = scratch("private_member");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 private int hidden;\n\
                 private void secret() {}\n\
                 protected int visible;\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert_eq!(
            widget
                .members
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
            vec!["visible"]
        );
    }

    #[test]
    fn a_constructor_is_not_recorded_as_a_member() {
        let dir = scratch("constructor_member");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 public Widget() {}\n\
                 static { }\n\
                 void reset() {}\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert_eq!(
            widget
                .members
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
            vec!["reset"]
        );
    }

    /// Beside the members and never among them, resolved like a method's
    /// parameters; a private one is no other type's to call.
    #[test]
    fn a_classs_constructors_are_recorded_apart_from_its_members() {
        let dir = scratch("constructors");
        let zip_path = write_zip(
            &dir,
            &[
                (
                    "java.base/com/example/Widget.java",
                    "package com.example;\n\
                     public class Widget {\n\
                     public Widget(Plain name, int... sizes) {}\n\
                     Widget() {}\n\
                     private Widget(long id) {}\n\
                     }\n",
                ),
                (
                    "java.base/com/example/Plain.java",
                    "package com.example;\npublic class Plain { void p() {} }\n",
                ),
            ],
        );
        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert!(widget.members.is_empty());
        let [named, bare] = widget.constructors.as_slice() else {
            panic!("two, not {:?}", widget.constructors);
        };
        assert_eq!(named.kind, DeclKind::Constructor);
        assert_eq!(
            named.parameters[0],
            MemberType::Class("com.example.Plain".to_owned())
        );
        assert!(named.variadic && named.is_public);
        assert_eq!(named.parameter_names, ["name", "sizes"]);
        assert!(bare.parameters.is_empty() && !bare.is_public);
        assert!(fact(&facts, "com.example.Plain").constructors.is_empty());
    }

    #[test]
    fn a_nested_type_is_not_recorded_as_a_member_of_its_outer_type() {
        let dir = scratch("nested_not_a_member");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Outer.java",
                "package com.example;\n\
                 public class Outer {\n\
                 static class Inner {}\n\
                 int count;\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let outer = fact(&facts, "com.example.Outer");
        assert_eq!(
            outer
                .members
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
            vec!["count"]
        );
        // It is reachable, just by its own name rather than as a member.
        assert_eq!(fact(&facts, "com.example.Outer.Inner").simple, "Inner");
    }

    #[test]
    fn a_static_method_is_recorded_as_static_and_an_instance_method_is_not() {
        let dir = scratch("static_method");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 public static void of() {}\n\
                 public void reset() {}\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert!(member(widget, "of").is_static);
        assert!(!member(widget, "reset").is_static);
    }

    #[test]
    fn a_field_of_an_interface_is_static_without_the_keyword() {
        let dir = scratch("interface_field");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Limits.java",
                "package com.example;\n\
                 public interface Limits {\n\
                 int MAX = 10;\n\
                 void check();\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let limits = fact(&facts, "com.example.Limits");
        assert!(member(limits, "MAX").is_static);
        // The implicit rule is about fields, not about every interface member:
        // an abstract instance method is still an instance method.
        assert!(!member(limits, "check").is_static);
    }

    #[test]
    fn an_enum_constant_is_recorded_as_a_static_member() {
        let dir = scratch("enum_constant");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Colour.java",
                "package com.example;\n\
                 public enum Colour { RED, GREEN }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let colour = fact(&facts, "com.example.Colour");
        let red = member(colour, "RED");
        assert_eq!(red.kind, DeclKind::EnumConstant);
        assert!(red.is_static);
        // Its type is the enum, which the declaration does not write down.
        assert_eq!(red.declared, MemberType::Class("com.example.Colour".into()));
    }

    #[test]
    fn a_member_type_the_file_imports_resolves_to_its_fully_qualified_name() {
        let dir = scratch("member_type_imported");
        let zip_path = write_zip(&dir, &[LIST_ENTRY, HOLDER_ENTRY]);

        let facts = read(&zip_path).expect("read");
        let holder = fact(&facts, "com.example.Holder");
        assert_eq!(
            member(holder, "first").declared,
            MemberType::Class("java.util.List".into())
        );
        assert_eq!(
            member(holder, "accept").parameters,
            vec![MemberType::Class("java.util.List".into())]
        );
    }

    #[test]
    fn a_member_type_nothing_resolves_stays_unresolved_rather_than_guessing() {
        let dir = scratch("member_type_unresolved");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 Nowhere thing;\n\
                 <T> T pick(T candidate) { return candidate; }\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert_eq!(
            member(widget, "thing").declared,
            MemberType::Unresolved("Nowhere".into())
        );
        // A type variable is a name this table cannot hold, so it goes the
        // same way rather than being read as a class called `T`.
        assert_eq!(
            member(widget, "pick").declared,
            MemberType::Unresolved("T".into())
        );
    }

    #[test]
    fn a_generic_member_type_records_its_arguments() {
        let dir = scratch("member_type_generic");
        let zip_path = write_zip(&dir, &[LIST_ENTRY, HOLDER_ENTRY]);

        let facts = read(&zip_path).expect("read");
        let holder = fact(&facts, "com.example.Holder");
        // `String` is not in this read, so the argument is as unresolved as
        // a member typed `String` would be — the structure around it is not.
        assert_eq!(
            member(holder, "items").declared,
            MemberType::Parameterized(
                "java.util.List".into(),
                vec![MemberType::Unresolved("String".into())]
            )
        );
        assert_eq!(
            member(holder, "first").declared,
            MemberType::Class("java.util.List".into()),
            "a raw type stays raw"
        );
    }

    fn unresolved(text: &str) -> MemberType {
        MemberType::Unresolved(text.into())
    }

    fn variable(name: &str) -> MemberType {
        MemberType::Variable(name.into())
    }

    const FUNCTION_ENTRY: (&str, &str) = (
        "java.base/java/util/function/Function.java",
        "package java.util.function;\npublic interface Function<T, R> { R apply(T t); }\n",
    );

    /// The owner's type variable is a `Variable`; a method's own is never
    /// one, and shadows an owner variable of the same name — `of`'s `E` is
    /// not `List`'s.
    #[test]
    fn an_owners_type_variable_is_a_variable_and_a_methods_own_is_unresolved() {
        let dir = scratch("member_type_variables");
        let zip_path = write_zip(
            &dir,
            &[
                FUNCTION_ENTRY,
                (
                    "java.base/java/util/List.java",
                    "package java.util;\n\
                     import java.util.function.Function;\n\
                     public interface List<E> {\n\
                     E get(int index);\n\
                     <T> T[] toArray(T[] a);\n\
                     <R> R map(Function<? super E, ? extends R> f);\n\
                     static <E> List<E> of(E e) { return null; }\n\
                     List<?> any();\n\
                     E[] raw();\n\
                     }\n",
                ),
            ],
        );
        let facts = read(&zip_path).expect("read");
        let list = fact(&facts, "java.util.List");
        assert_eq!(list.type_parameters, vec!["E"]);
        assert_eq!(member(list, "get").declared, variable("E"));
        assert_eq!(member(list, "toArray").declared, unresolved("T[]"));
        let map = member(list, "map");
        assert_eq!(map.declared, unresolved("R"));
        assert_eq!(
            map.parameters,
            vec![MemberType::Parameterized(
                "java.util.function.Function".into(),
                vec![
                    MemberType::Super(Box::new(variable("E"))),
                    MemberType::Extends(Box::new(unresolved("R"))),
                ]
            )]
        );
        let of = member(list, "of");
        assert_eq!(
            of.declared,
            MemberType::Parameterized("java.util.List".into(), vec![unresolved("E")])
        );
        assert_eq!(of.parameters, vec![unresolved("E")]);
        assert_eq!(
            member(list, "any").declared,
            MemberType::Parameterized("java.util.List".into(), vec![unresolved("?")])
        );
        assert_eq!(member(list, "raw").declared, unresolved("E[]"));
    }

    #[test]
    fn a_supertype_records_the_arguments_it_writes() {
        let dir = scratch("supertype_arguments");
        let zip_path = write_zip(
            &dir,
            &[
                LIST_ENTRY,
                (
                    "java.base/java/util/RandomAccess.java",
                    "package java.util;\npublic interface RandomAccess {}\n",
                ),
                (
                    "java.base/java/util/AbstractList.java",
                    "package java.util;\npublic abstract class AbstractList<E> implements List<E> {}\n",
                ),
                (
                    "java.base/java/util/ArrayList.java",
                    "package java.util;\n\
                     public class ArrayList<T> extends AbstractList<T> \
                     implements List<T>, RandomAccess {}\n",
                ),
                (
                    "java.base/java/util/Names.java",
                    "package java.util;\npublic class Names extends ArrayList<Names> {}\n",
                ),
            ],
        );
        let facts = read(&zip_path).expect("read");
        let array_list = fact(&facts, "java.util.ArrayList");
        assert_eq!(array_list.type_parameters, vec!["T"]);
        assert_eq!(
            array_list.supertype_arguments,
            vec![
                ("java.util.AbstractList".to_owned(), vec![variable("T")]),
                ("java.util.List".to_owned(), vec![variable("T")]),
            ],
            "RandomAccess writes no arguments and is not listed"
        );
        assert_eq!(
            fact(&facts, "java.util.Names").supertype_arguments,
            vec![(
                "java.util.ArrayList".to_owned(),
                vec![MemberType::Class("java.util.Names".into())]
            )]
        );
    }

    /// What a lambda's target type has exactly one of.
    #[test]
    fn an_interface_method_is_abstract_unless_it_has_a_body_and_a_class_says_so() {
        let dir = scratch("abstract_members");
        let zip_path = write_zip(
            &dir,
            &[
                (
                    "java.base/java/util/function/Consumer.java",
                    "package java.util.function;\n\
                     public interface Consumer<T> {\n\
                     void accept(T t);\n\
                     default Consumer<T> andThen(Consumer<? super T> after) { return null; }\n\
                     static <T> Consumer<T> noop() { return null; }\n\
                     boolean equals(Object o);\n\
                     }\n",
                ),
                (
                    "java.base/java/util/Base.java",
                    "package java.util;\npublic abstract class Base { \
                     public abstract int size(); public boolean isEmpty() { return true; } }\n",
                ),
            ],
        );
        let facts = read(&zip_path).expect("read");
        let consumer = fact(&facts, "java.util.function.Consumer");
        let abstract_of = |facts: &TypeFacts, name: &str| member(facts, name).is_abstract;
        assert!(abstract_of(consumer, "accept"));
        assert!(abstract_of(consumer, "equals"), "abstract, though Object's");
        assert!(!abstract_of(consumer, "andThen"));
        assert!(!abstract_of(consumer, "noop"));
        let base = fact(&facts, "java.util.Base");
        assert!(abstract_of(base, "size"));
        assert!(!abstract_of(base, "isEmpty"));
    }

    #[test]
    fn an_annotated_type_parameter_is_named_by_its_name_not_its_annotation() {
        assert_eq!(
            type_parameter_names(Some("<@A T, @B(x = 1) U>")),
            vec!["T", "U"]
        );
    }

    #[test]
    fn an_array_member_type_stays_unresolved_rather_than_claiming_its_element_type() {
        let dir = scratch("member_type_array");
        let zip_path = write_zip(
            &dir,
            &[
                LIST_ENTRY,
                (
                    "java.base/com/example/Holder.java",
                    "package com.example;\n\
                     import java.util.List;\n\
                     public class Holder {\n\
                     List<String>[] buckets;\n\
                     int[] counts;\n\
                     List<int[]> boxed;\n\
                     }\n",
                ),
            ],
        );

        let facts = read(&zip_path).expect("read");
        let holder = fact(&facts, "com.example.Holder");
        assert_eq!(
            member(holder, "buckets").declared,
            MemberType::Unresolved("List<String>[]".into())
        );
        assert_eq!(
            member(holder, "counts").declared,
            MemberType::Unresolved("int[]".into())
        );
        // The `[]` is inside the arguments here, so this one is a list and not
        // an array of anything.
        assert_eq!(
            member(holder, "boxed").declared,
            MemberType::Parameterized(
                "java.util.List".into(),
                vec![MemberType::Unresolved("int[]".into())]
            )
        );
    }

    #[test]
    fn a_primitive_return_type_is_recorded_as_a_primitive_and_void_as_void() {
        let dir = scratch("member_type_primitive");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 int count() { return 0; }\n\
                 void reset() {}\n\
                 boolean empty;\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        assert_eq!(
            member(widget, "count").declared,
            MemberType::Primitive("int".into())
        );
        assert_eq!(member(widget, "reset").declared, MemberType::Void);
        assert_eq!(
            member(widget, "empty").declared,
            MemberType::Primitive("boolean".into())
        );
    }

    #[test]
    fn a_variadic_method_records_its_last_parameter_as_variadic() {
        let dir = scratch("member_variadic");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 void log(int level, boolean... flags) {}\n\
                 void fixed(int level) {}\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        let log = member(widget, "log");
        assert!(log.variadic);
        assert_eq!(
            log.parameters,
            vec![
                MemberType::Primitive("int".into()),
                MemberType::Primitive("boolean".into())
            ]
        );
        assert!(!member(widget, "fixed").variadic);
    }

    #[test]
    fn a_types_facts_name_the_entry_they_were_read_from() {
        let dir = scratch("provenance_entry");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/java/util/Widget.java",
                "package java.util;\npublic class Widget {}\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "java.util.Widget");
        assert_eq!(widget.source, "java.base/java/util/Widget.java");
    }

    #[test]
    fn a_nested_type_carries_its_outer_types_entry() {
        let dir = scratch("provenance_nested");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/java/util/Map.java",
                "package java.util;\npublic interface Map<K, V> { interface Entry<K, V> {} }",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let map = fact(&facts, "java.util.Map");
        let entry = fact(&facts, "java.util.Map.Entry");
        assert_eq!(entry.source, map.source);
        assert_eq!(entry.source, "java.base/java/util/Map.java");
    }

    #[test]
    fn a_members_name_span_slices_its_own_name_out_of_the_entry() {
        let dir = scratch("provenance_member_span");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 int count;\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let widget = fact(&facts, "com.example.Widget");
        let count = member(widget, "count");

        // Pinning both halves together: the offsets are only useful if they
        // slice the same name back out of the entry they claim to be in.
        let mut archive = SourceArchive::open(&zip_path).expect("open");
        let text = archive.read(&widget.source, NO_LIMIT).expect("read entry");
        assert_eq!(
            &text[count.name_start as usize..count.name_end as usize],
            "count"
        );
    }

    /// What a jump to the type itself lands on: its own name, for a nested
    /// type inside its outer type's entry as much as for the outer one.
    #[test]
    fn a_types_name_span_slices_its_own_name_out_of_the_entry() {
        let dir = scratch("provenance_type_span");
        let zip_path = write_zip(
            &dir,
            &[(
                "java.base/com/example/Widget.java",
                "package com.example;\n\
                 public class Widget {\n\
                 static class Part {}\n\
                 }\n",
            )],
        );

        let facts = read(&zip_path).expect("read");
        let mut archive = SourceArchive::open(&zip_path).expect("open");
        for (fqn, name) in [
            ("com.example.Widget", "Widget"),
            ("com.example.Widget.Part", "Part"),
        ] {
            let ty = fact(&facts, fqn);
            let text = archive.read(&ty.source, NO_LIMIT).expect("read entry");
            assert_eq!(&text[ty.name_start as usize..ty.name_end as usize], name);
        }
    }

    #[test]
    fn an_entry_read_twice_gives_the_same_text() {
        let dir = scratch("provenance_read_twice");
        let contents = "package com.example;\npublic class Widget {}\n";
        let zip_path = write_zip(&dir, &[("java.base/com/example/Widget.java", contents)]);

        let mut archive = SourceArchive::open(&zip_path).expect("open");
        let first = archive
            .read("java.base/com/example/Widget.java", NO_LIMIT)
            .expect("first read");
        let second = archive
            .read("java.base/com/example/Widget.java", NO_LIMIT)
            .expect("second read");
        assert_eq!(first, contents);
        assert_eq!(first, second);
    }

    #[test]
    fn reading_an_entry_that_is_not_in_the_archive_is_an_error() {
        let dir = scratch("provenance_missing_entry");
        let zip_path = write_zip(
            &dir,
            &[("java.base/com/example/Widget.java", "class Widget {}\n")],
        );

        let mut archive = SourceArchive::open(&zip_path).expect("open");
        let result = archive.read("java.base/com/example/Nowhere.java", NO_LIMIT);
        assert!(result.is_err());
    }

    #[test]
    fn an_entry_over_the_size_cap_is_refused_rather_than_read() {
        let dir = scratch("provenance_size_cap");
        let contents = "x".repeat(100);
        let zip_path = write_zip(&dir, &[("java.base/com/example/Huge.java", &contents)]);

        let mut archive = SourceArchive::open(&zip_path).expect("open");
        let result = archive.read("java.base/com/example/Huge.java", 50);
        assert!(result.is_err());
    }

    const LIST_ENTRY: (&str, &str) = (
        "java.base/java/util/List.java",
        "package java.util;\npublic interface List<E> {}\n",
    );

    const HOLDER_ENTRY: (&str, &str) = (
        "java.base/com/example/Holder.java",
        "package com.example;\n\
         import java.util.List;\n\
         public class Holder {\n\
         List<String> items;\n\
         List first;\n\
         void accept(List other) {}\n\
         }\n",
    );

    #[test]
    #[ignore = "reads the real JDK found on this machine; run by hand"]
    fn reading_a_real_jdk_reports_its_cost() {
        let jdk = crate::discover(None, &[]).expect("a JDK on this machine");
        let src_zip = jdk.sources.expect("this JDK ships src.zip");

        let start = std::time::Instant::now();
        let facts = read_type_facts(&src_zip, 64 * 1024 * 1024, &CancellationToken::never())
            .expect("read the real JDK");
        let elapsed = start.elapsed();

        let members: usize = facts.iter().map(|fact| fact.members.len()).sum();
        // The two figures Task 2 needs to size the cache growth against: the
        // entry names now stored once per type, and the two `u32` offsets now
        // stored once per member.
        let source_bytes: usize = facts.iter().map(|fact| fact.source.len()).sum();
        let avg_entry_name_len = source_bytes as f64 / facts.len().max(1) as f64;
        let span_bytes = members * (std::mem::size_of::<u32>() * 2);
        // Types a member diagnostic cannot judge because their entry parsed
        // with errors.
        let unparsed: Vec<&str> = facts
            .iter()
            .filter(|fact| !fact.members_complete)
            .map(|fact| fact.fqn.as_str())
            .collect();
        eprintln!(
            "jdk version={:?} facts={} members={} elapsed={:.2?} \
             avg_entry_name_len={avg_entry_name_len:.1} source_bytes={source_bytes} \
             span_bytes={span_bytes} unparsed={} first_unparsed={:?}",
            jdk.version,
            facts.len(),
            members,
            elapsed,
            unparsed.len(),
            &unparsed[..unparsed.len().min(40)]
        );
        // What generic substitution reads, for a human to eyeball against
        // the JDK's own declarations.
        for (fqn, member) in [
            ("java.util.List", "get"),
            ("java.util.Map", "entrySet"),
            ("java.lang.Iterable", "forEach"),
            ("java.util.Optional", "get"),
            ("java.util.stream.Stream", "map"),
        ] {
            let Some(fact) = facts.iter().find(|fact| fact.fqn == fqn) else {
                continue;
            };
            let found = fact.members.iter().find(|m| m.name == member);
            eprintln!(
                "{fqn}<{:?}> supertypes={:?} {member}: {:?}",
                fact.type_parameters,
                fact.supertype_arguments,
                found.map(|m| (&m.declared, &m.parameters))
            );
        }
    }

    /// The facts a prebuilt index ships as must come back exactly as they went in.
    #[cfg(feature = "serde")]
    #[test]
    fn type_facts_survive_a_serde_round_trip() {
        let dir = scratch("serde-round-trip");
        let source = "package p;\npublic class Box<T> extends Object {\n    public T get() { return null; }\n    public static int count(String... xs) { return 0; }\n}\n";
        let zip_path = write_zip(&dir, &[("java.base/p/Box.java", source)]);

        let facts = read(&zip_path).expect("read");
        assert!(!facts.is_empty());
        let json = serde_json::to_string(&facts).expect("serialises");
        let back: Vec<TypeFacts> = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(facts, back);
    }
}
