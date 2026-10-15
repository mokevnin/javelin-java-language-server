//! The workspace's own sources: what each file declares, and the store that
//! remembers it between sessions.
//!
//! The JDK's facts are read once and shared by every project on the machine.
//! These are the opposite — they belong to one workspace and change while the
//! user works — so the store is per-workspace and the read is *incremental*:
//! a file whose modification time and size are the ones already recorded is
//! not opened at all. On a warm start most of a workspace is skipped, which is
//! the whole reason the `source_files` table exists.
//!
//! Like [`crate::types`], this is a bulk store rather than a query engine.
//! [`index_sources`](CacheDb::index_sources) hands back an immutable
//! [`WorkspaceIndex`] and the request path asks *that*, never SQLite.
//!
//! What is recorded is types, not members: a fully-qualified name, a simple
//! name, a kind, the spans of the name and of the whole declaration, and the
//! supertypes **as written**. Resolving a written supertype needs the
//! declaring file's package and imports, which is a question the slice with a
//! reader for it gets to answer; storing the text keeps that slice's options
//! open without guessing now.
//!
//! Beside the types, every plain name a file writes where a type could be
//! meant — [`javelin_parser::type_occurrences`] — and the file's package and
//! imports, which is what `textDocument/references` reads. An occurrence
//! stores its spelling and, where the file itself could say, the type it
//! means; everything else is resolved when a query asks, because what a
//! simple name means depends on every other type the workspace and the JDK
//! hold, and both change after the walk.
//!
//! ponytail: occurrences are held in memory for the whole workspace, one row
//! per type-like name per file — tens of megabytes for a large repository.
//! The upgrade, if a monorepo makes it measurable, is to keep only the
//! spelling-to-file map in memory and read the rows for one query out of
//! SQLite on a worker.

#[cfg(not(target_family = "wasm"))]
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use javelin_classpath::{
    DeclKind, LombokUse, WrittenMember, implicit_members, lombok_members, type_parameter_names,
    written_constructors, written_members,
};
use javelin_core::{CancellationToken, Result};
use javelin_document::{DocumentId, Position, Range, TextDocument};
use javelin_parser::{
    FileSyntax, ImportDecl, JavaParser, ScopeTree, TypeOccurrence, type_occurrences,
};
use rusqlite::Connection;

use crate::types::{from_kind_code, kind_code};
use crate::walk::Found;
use crate::{CacheDb, to_index_error};

/// Files written per transaction.
///
/// A whole-walk transaction would make a session killed halfway through leave
/// nothing behind; one transaction per file would fsync tens of thousands of
/// times. Batching leaves a consistent prefix the next walk continues from.
const BATCH: usize = 256;

pub(crate) fn initialize(connection: &Connection) -> Result<()> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS source_files (
                 path     TEXT PRIMARY KEY,
                 mtime_ns INTEGER NOT NULL,
                 size     INTEGER NOT NULL,
                 package  TEXT,
                 member_refs INTEGER NOT NULL DEFAULT 0,
                 parsed_cleanly INTEGER NOT NULL
             ) STRICT;
             CREATE TABLE IF NOT EXISTS source_types (
                 path       TEXT NOT NULL,
                 fqn        TEXT NOT NULL,
                 simple     TEXT NOT NULL,
                 kind       INTEGER NOT NULL,
                 name_start INTEGER NOT NULL,
                 name_end   INTEGER NOT NULL,
                 decl_start INTEGER NOT NULL,
                 decl_end   INTEGER NOT NULL,
                 name_line_start INTEGER NOT NULL,
                 name_char_start INTEGER NOT NULL,
                 name_line_end   INTEGER NOT NULL,
                 name_char_end   INTEGER NOT NULL,
                 synthetic       INTEGER NOT NULL,
                 doc_start       INTEGER,
                 doc_end         INTEGER,
                 lombok          INTEGER NOT NULL,
                 type_params     TEXT NOT NULL,
                 is_public       INTEGER NOT NULL
             ) STRICT;
             CREATE INDEX IF NOT EXISTS source_types_path ON source_types (path);
             CREATE TABLE IF NOT EXISTS source_supertypes (
                 path    TEXT NOT NULL,
                 fqn     TEXT NOT NULL,
                 written TEXT NOT NULL
             ) STRICT;
             CREATE INDEX IF NOT EXISTS source_supertypes_path ON source_supertypes (path);
             CREATE TABLE IF NOT EXISTS source_imports (
                 path         TEXT NOT NULL,
                 position     INTEGER NOT NULL,
                 name         TEXT NOT NULL,
                 is_static    INTEGER NOT NULL,
                 is_on_demand INTEGER NOT NULL
             ) STRICT;
             CREATE INDEX IF NOT EXISTS source_imports_path ON source_imports (path);
             CREATE TABLE IF NOT EXISTS source_occurrences (
                 path       TEXT NOT NULL,
                 name       TEXT NOT NULL,
                 start_byte INTEGER NOT NULL,
                 end_byte   INTEGER NOT NULL,
                 line_start INTEGER NOT NULL,
                 char_start INTEGER NOT NULL,
                 line_end   INTEGER NOT NULL,
                 char_end   INTEGER NOT NULL,
                 fqn        TEXT
             ) STRICT;
             CREATE INDEX IF NOT EXISTS source_occurrences_path ON source_occurrences (path);
             CREATE TABLE IF NOT EXISTS source_members (
                 path       TEXT NOT NULL,
                 owner      TEXT NOT NULL,
                 position   INTEGER NOT NULL,
                 name       TEXT NOT NULL,
                 kind       INTEGER NOT NULL,
                 is_static  INTEGER NOT NULL,
                 declared   TEXT,
                 arity      INTEGER NOT NULL,
                 params     TEXT NOT NULL,
                 variadic   INTEGER NOT NULL,
                 name_start INTEGER NOT NULL,
                 name_end   INTEGER NOT NULL,
                 line_start INTEGER NOT NULL,
                 char_start INTEGER NOT NULL,
                 line_end   INTEGER NOT NULL,
                 char_end   INTEGER NOT NULL,
                 is_private INTEGER NOT NULL,
                 type_params TEXT NOT NULL,
                 decl_start INTEGER NOT NULL,
                 decl_end   INTEGER NOT NULL,
                 synthetic  INTEGER NOT NULL,
                 param_names TEXT NOT NULL,
                 doc_start  INTEGER,
                 doc_end    INTEGER,
                 is_public  INTEGER NOT NULL
             ) STRICT;
             CREATE INDEX IF NOT EXISTS source_members_path ON source_members (path);
             CREATE TABLE IF NOT EXISTS source_member_refs (
                 path          TEXT NOT NULL,
                 name          TEXT NOT NULL,
                 start_byte    INTEGER NOT NULL,
                 end_byte      INTEGER NOT NULL,
                 line_start    INTEGER NOT NULL,
                 char_start    INTEGER NOT NULL,
                 line_end      INTEGER NOT NULL,
                 char_end      INTEGER NOT NULL,
                 target_source TEXT NOT NULL,
                 target_start  INTEGER NOT NULL
             ) STRICT;
             CREATE INDEX IF NOT EXISTS source_member_refs_path ON source_member_refs (path);
             CREATE TABLE IF NOT EXISTS source_member_deps (
                 path TEXT NOT NULL,
                 fqn  TEXT NOT NULL
             ) STRICT;
             CREATE INDEX IF NOT EXISTS source_member_deps_path ON source_member_deps (path);
             CREATE INDEX IF NOT EXISTS source_member_deps_fqn ON source_member_deps (fqn);",
        )
        .map_err(to_index_error)
}

/// One type a workspace source file declares.
///
/// The spans are byte offsets into that file as it was on disk when it was
/// read. Converting them to an editor's line-and-character range is the
/// caller's job, against a document over the file's current text — the same
/// rule a jump into the JDK's sources already follows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceType {
    pub fqn: String,
    pub simple: String,
    pub kind: DeclKind,
    pub file: PathBuf,
    pub name_start: u32,
    pub name_end: u32,
    pub decl_start: u32,
    pub decl_end: u32,
    /// The name's span again, as a line and a UTF-16 character offset.
    ///
    /// Stored rather than derived, which is a deliberate exception to the
    /// rule the byte spans follow. A reader on the request path has no
    /// document over a file it has never opened, and building one to convert
    /// four integers is reading and scanning a whole file — file-sized work
    /// on the request loop, for one row of a symbol list. The walk already
    /// holds the document, so the conversion is free there.
    ///
    /// ponytail: as stale as everything else here, and stale in the one way
    /// that shows. A byte span that has moved renders as a range over the
    /// wrong text; a line that has moved sends the editor to the wrong line
    /// and the user sees it. The upgrade is a reader that prefers the open
    /// document, which is the same upgrade the index's staleness wants
    /// anyway.
    pub name_range: javelin_document::Range,
    /// Supertypes as the author wrote them: `Base<T>`, `java.io.Serializable`,
    /// `Map.Entry`. Unresolved on purpose — see the module doc.
    pub supertypes: Vec<String>,
    /// Written `public`, or a member type of an interface or an annotation type;
    /// see `javelin_classpath::TypeFacts::is_public`, which this agrees with.
    pub is_public: bool,
    /// Generated by Lombok rather than written — `@Builder`'s nested class —
    /// and sited at the type it was generated for.
    pub synthetic: bool,
    /// The type's `/** … */`, by byte offsets in its file.
    pub doc_span: Option<(u32, u32)>,
    /// Whether Lombok writes into the type, and whether this index holds all
    /// it writes — not below a `lombok.config` that renames it.
    pub lombok: LombokUse,
    /// The type's own type parameter names, in order: what a `T` its members
    /// and its supertypes write means.
    pub type_parameters: Vec<String>,
}

/// A source file as a name in it is resolved against: where it is, and the
/// package and imports that decide what a simple name written in it means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceFile {
    pub path: PathBuf,
    pub package: Option<String>,
    /// As `javelin_classpath::simple_name` reads them. The spans are zero: an
    /// import is stored for the name and the two flags the resolver reads,
    /// and nothing that holds one of these renders where it was written.
    pub imports: Vec<ImportDecl>,
    /// The file parsed without an error node. Recovery can drop a member a
    /// broken file declares, so a type in one has no whole member list.
    pub parsed_cleanly: bool,
}

/// One plain name a workspace file writes where a type could be meant — a
/// [`javelin_parser::TypeOccurrence`] as the walk stored it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceOccurrence {
    /// Which [`WorkspaceIndex::file`] it is in.
    pub file: usize,
    pub name: String,
    pub start_byte: u32,
    pub end_byte: u32,
    /// The name's line and UTF-16 characters, stored for the reason
    /// [`WorkspaceType::name_range`] is: the reader has never opened the file.
    pub range: Range,
    /// Set when the file itself said which type the name means; `None` when
    /// it is resolved at query time against the file's package and imports.
    pub fqn: Option<String>,
}

/// One member a workspace type declares, with its types as written — the
/// shape `javelin_classpath::written_members` reads, stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceMember {
    /// The declaring type's fully-qualified name.
    pub owner: String,
    /// Which [`WorkspaceIndex::file`] declares it.
    pub file: usize,
    pub name: String,
    pub kind: DeclKind,
    pub is_static: bool,
    pub declared: Option<String>,
    pub parameters: Vec<String>,
    /// The parameters' names, as written.
    pub parameter_names: Vec<String>,
    /// The member's `/** … */`, by byte offsets in its file.
    pub doc_span: Option<(u32, u32)>,
    pub variadic: bool,
    pub name_start: u32,
    pub name_end: u32,
    /// The name's line and UTF-16 characters, for the reason
    /// [`WorkspaceType::name_range`] stores them.
    pub name_range: Range,
    /// Declared `private`: listed by a symbol search, selectable from no
    /// other file.
    pub is_private: bool,
    /// The type parameters a written type of this member can name — see
    /// `javelin_classpath::WrittenMember::type_parameters`.
    pub type_parameters: Vec<String>,
    /// Byte offsets of the whole declaration, for which member a use is
    /// written inside.
    pub decl_start: u32,
    pub decl_end: u32,
    /// Generated by Lombok rather than written, and sited at what generated
    /// it: a getter at its field's name, which is the same site — so a
    /// getter's uses are its field's — and a declaration nothing is inside.
    pub synthetic: bool,
    /// See `javelin_classpath::MemberFacts::is_public`.
    pub is_public: bool,
}

/// One member selection a file makes, resolved by the server to the member it
/// lands on: that member's declaring source and the byte offset of its name
/// there — the key `javelin_semantic::ExternalSite` carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberRef {
    pub name: String,
    pub start_byte: u32,
    pub end_byte: u32,
    pub range: Range,
    pub target_source: String,
    pub target_start: u32,
}

/// A [`MemberRef`] as the published index holds it, in the file at `file`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceMemberRef {
    pub file: usize,
    pub name: String,
    /// Where the use starts in its file, in bytes — which member it is inside.
    pub start_byte: u32,
    pub range: Range,
    pub target_source: String,
    pub target_start: u32,
}

/// A walk's index, and which files it re-read or found gone — what the
/// server re-resolves member references for.
#[derive(Debug)]
pub struct Walked {
    pub index: WorkspaceIndex,
    pub changed: Vec<PathBuf>,
    pub gone: Vec<PathBuf>,
}

/// One file an event says has changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceChange {
    /// Read the file from disk: re-index it if it is there, forget it if not.
    Disk(PathBuf),
    /// Index this text as the file's content — an unsaved buffer.
    Buffer { path: PathBuf, text: String },
}

/// What one file declares, ready to be written down.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceFacts {
    path: PathBuf,
    mtime_ns: i64,
    size: i64,
    package: Option<String>,
    parsed_cleanly: bool,
    types: Vec<WorkspaceType>,
    imports: Vec<ImportDecl>,
    occurrences: Vec<TypeOccurrence>,
    /// Each named type's members, by the type's fully-qualified name, and
    /// whether Lombok generated it.
    members: Vec<(String, WrittenMember, bool)>,
}

/// `source_files.member_refs`, the world a file's member references were
/// resolved against: `0`, the column's default that every re-read file gets,
/// is not resolved at all; [`REFS_WITHOUT_WORLD`] before the JDK's facts
/// arrived; [`REFS_WITH_JDK`] with them, before the dependencies settled; and
/// anything above, [`refs_settled`] — the key of a settled world, so a session
/// whose JDK or JARs differ, or that turned the build import on, owes every
/// file again rather than trusting answers read through another classpath.
pub const REFS_WITHOUT_WORLD: i64 = 1;
pub const REFS_WITH_JDK: i64 = 2;

/// The `source_files.member_refs` state of references resolved against the
/// settled world `key` names: never one of the two states below it.
pub fn refs_settled(key: u64) -> i64 {
    (key >> 2) as i64 + REFS_WITH_JDK + 1
}

/// Every type the workspace's own sources declare, as of the last walk.
///
/// Immutable once built and shared as an `Arc`, exactly as
/// [`javelin_classpath::TypeWorld`] is: the request path holds a refcount, not
/// a lock and not a database handle.
///
/// A fully-qualified name maps to a *list*, not to one type. Two files
/// declaring `com.example.Foo` is a broken workspace rather than an impossible
/// one — a stray copy, a half-finished move — and the engine's rule is that
/// ambiguity answers nothing. Collapsing them here would pick one silently.
#[derive(Debug, Default)]
pub struct WorkspaceIndex {
    types: Vec<WorkspaceType>,
    by_fqn: HashMap<String, Vec<usize>>,
    by_simple: HashMap<String, Vec<usize>>,
    /// Types by the file declaring them, for the worker's per-file questions.
    by_file: HashMap<PathBuf, Vec<usize>>,
    /// Types by the base simple name of each supertype they write — the
    /// inverse edge, before resolution says which of them mean the type.
    by_super: HashMap<String, Vec<usize>>,
    /// Members by their file and the offset of their name: a site.
    by_site: HashMap<(usize, u32), usize>,
    /// Generated members by the site they share with what generated them.
    generated_at: HashMap<(usize, u32), Vec<usize>>,
    /// Members by their file, in the order `members` holds them.
    members_by_file: HashMap<usize, Vec<usize>>,
    /// Member references by the file making them.
    refs_by_file: HashMap<usize, Vec<usize>>,
    files: usize,
    sources: Vec<WorkspaceFile>,
    occurrences: Vec<WorkspaceOccurrence>,
    /// Occurrences by spelling, which is the whole of what a references query
    /// looks up: the rows spelled like the target, each resolved on its own.
    by_name: HashMap<String, Vec<usize>>,
    members: Vec<WorkspaceMember>,
    /// Members by their owner's fully-qualified name, each a contiguous run of
    /// `members` in source order.
    by_owner: HashMap<String, std::ops::Range<usize>>,
    /// Constructors, apart from `members` and by owner the same way.
    constructors: Vec<WorkspaceMember>,
    constructors_by_owner: HashMap<String, std::ops::Range<usize>>,
    /// Sources by path, for the file a type is declared in.
    by_path: HashMap<PathBuf, usize>,
    member_refs: Vec<WorkspaceMemberRef>,
    /// Member references by the member they land on: its source and the
    /// offset of its name there.
    by_target: HashMap<(String, u32), Vec<usize>>,
}

/// Each owner's contiguous run of `members`, sorted by owner.
fn runs(members: &[WorkspaceMember]) -> HashMap<String, std::ops::Range<usize>> {
    let mut by_owner = HashMap::new();
    let mut start = 0;
    while start < members.len() {
        let owner = &members[start].owner;
        let end = start
            + members[start..]
                .iter()
                .take_while(|member| &member.owner == owner)
                .count();
        by_owner.insert(owner.clone(), start..end);
        start = end;
    }
    by_owner
}

impl WorkspaceIndex {
    /// Index these types. `pub` for the same reason
    /// [`javelin_classpath::TypeWorld::new`] is: a caller that has facts from
    /// somewhere other than SQLite — a test, a benchmark — needs a way in.
    pub fn new(types: Vec<WorkspaceType>) -> Self {
        Self::with_occurrences(types, Vec::new(), Vec::new())
    }

    /// Index these types and these occurrences, each occurrence's `file`
    /// being a position in `sources`.
    pub fn with_occurrences(
        types: Vec<WorkspaceType>,
        sources: Vec<WorkspaceFile>,
        occurrences: Vec<WorkspaceOccurrence>,
    ) -> Self {
        let mut by_fqn: HashMap<String, Vec<usize>> = HashMap::new();
        let mut by_simple: HashMap<String, Vec<usize>> = HashMap::new();
        let mut by_file: HashMap<PathBuf, Vec<usize>> = HashMap::new();
        for (position, ty) in types.iter().enumerate() {
            by_fqn.entry(ty.fqn.clone()).or_default().push(position);
            by_simple
                .entry(ty.simple.clone())
                .or_default()
                .push(position);
            by_file.entry(ty.file.clone()).or_default().push(position);
        }
        let mut by_super: HashMap<String, Vec<usize>> = HashMap::new();
        for (position, ty) in types.iter().enumerate() {
            for written in &ty.supertypes {
                let base = javelin_classpath::written_type_name(written);
                let simple = base.rsplit('.').next().unwrap_or(&base);
                let listed = by_super.entry(simple.to_owned()).or_default();
                if listed.last() != Some(&position) {
                    listed.push(position);
                }
            }
        }
        let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
        // By the last segment: a qualified occurrence's name is its dotted
        // text, and a query looks a type up by its simple name.
        for (position, occurrence) in occurrences.iter().enumerate() {
            let simple = occurrence
                .name
                .rsplit('.')
                .next()
                .unwrap_or(&occurrence.name);
            by_name.entry(simple.to_owned()).or_default().push(position);
        }
        let by_path = sources
            .iter()
            .enumerate()
            .map(|(position, file)| (file.path.clone(), position))
            .collect();
        Self {
            files: by_file.len(),
            types,
            by_fqn,
            by_simple,
            by_file,
            by_super,
            by_site: HashMap::new(),
            generated_at: HashMap::new(),
            members_by_file: HashMap::new(),
            refs_by_file: HashMap::new(),
            sources,
            occurrences,
            by_name,
            members: Vec::new(),
            by_owner: HashMap::new(),
            constructors: Vec::new(),
            constructors_by_owner: HashMap::new(),
            by_path,
            member_refs: Vec::new(),
            by_target: HashMap::new(),
        }
    }

    /// Every type declared under this fully-qualified name. More than one
    /// means the workspace declares it twice.
    pub fn with_fqn(&self, fqn: &str) -> impl Iterator<Item = &WorkspaceType> {
        self.at(self.by_fqn.get(fqn))
    }

    /// Every type the file at `path` declares.
    pub fn types_in(&self, path: &Path) -> impl Iterator<Item = &WorkspaceType> {
        self.at(self.by_file.get(path))
    }

    /// Whether some workspace type is named `prefix` or lies under it — see
    /// `javelin_classpath::TypeWorld::holds_prefix`, whose scan this shares.
    pub fn holds_prefix(&self, prefix: &str) -> bool {
        self.types.iter().any(|ty| {
            ty.fqn
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
        })
    }

    /// Every type the workspace declares.
    pub fn types(&self) -> &[WorkspaceType] {
        &self.types
    }

    /// Every type declared under this simple name, in any package.
    pub fn with_simple_name(&self, simple: &str) -> impl Iterator<Item = &WorkspaceType> {
        self.at(self.by_simple.get(simple))
    }

    /// Every occurrence of a name spelled `name`, in any file.
    pub fn occurrences_named(&self, name: &str) -> impl Iterator<Item = &WorkspaceOccurrence> {
        self.by_name
            .get(name)
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .map(|&position| &self.occurrences[position])
    }

    /// Add members to an index, each `file` a position in its sources.
    ///
    /// Grouped by owner here, stably, so a caller may hand them over in any
    /// order and each owner's members keep the order they were written in.
    pub fn with_members(mut self, members: Vec<WorkspaceMember>) -> Self {
        let (mut constructors, mut members): (Vec<_>, Vec<_>) = members
            .into_iter()
            .partition(|member| member.kind == DeclKind::Constructor);
        constructors.sort_by(|a, b| a.owner.cmp(&b.owner));
        self.constructors_by_owner = runs(&constructors);
        self.constructors = constructors;
        members.sort_by(|a, b| a.owner.cmp(&b.owner));
        let by_owner = runs(&members);
        let mut by_site = HashMap::with_capacity(members.len());
        let mut generated_at: HashMap<(usize, u32), Vec<usize>> = HashMap::new();
        let mut members_by_file: HashMap<usize, Vec<usize>> = HashMap::new();
        for (position, member) in members.iter().enumerate() {
            // A generated member shares its field's site and is inside
            // nothing: the field is what the site names, and what a use is
            // inside.
            if member.synthetic {
                by_site
                    .entry((member.file, member.name_start))
                    .or_insert(position);
                generated_at
                    .entry((member.file, member.name_start))
                    .or_default()
                    .push(position);
                continue;
            }
            by_site.insert((member.file, member.name_start), position);
            members_by_file
                .entry(member.file)
                .or_default()
                .push(position);
        }
        // By declaration start, which is what `member_enclosing` searches.
        for positions in members_by_file.values_mut() {
            positions.sort_by_key(|&position| members[position].decl_start);
        }
        self.members = members;
        self.by_owner = by_owner;
        self.by_site = by_site;
        self.generated_at = generated_at;
        self.members_by_file = members_by_file;
        self
    }

    /// Every type that writes a supertype whose simple name is `simple` —
    /// each a candidate subtype, until resolution says what it wrote means.
    pub fn types_writing(&self, simple: &str) -> impl Iterator<Item = &WorkspaceType> {
        self.at(self.by_super.get(simple))
    }

    /// The member whose name starts at `name_start` in the file at `path`.
    pub fn member_at(&self, path: &Path, name_start: u32) -> Option<&WorkspaceMember> {
        let file = *self.by_path.get(path)?;
        self.by_site
            .get(&(file, name_start))
            .map(|&position| &self.members[position])
    }

    /// The members Lombok generated at the site `name_start` in the file at
    /// `path` — a field's getter and setter, a type's `builder()`.
    pub fn generated_at(
        &self,
        path: &Path,
        name_start: u32,
    ) -> impl Iterator<Item = &WorkspaceMember> {
        self.by_path
            .get(path)
            .and_then(|file| self.generated_at.get(&(*file, name_start)))
            .into_iter()
            .flatten()
            .map(|&position| &self.members[position])
    }

    /// The member of the file at `path` whose declaration holds `offset`.
    ///
    /// A binary search, because a file's members do not nest — a nested
    /// type's members are inside the type, not inside another member, and
    /// neither an anonymous nor a local class is indexed.
    pub fn member_enclosing(&self, path: &Path, offset: u32) -> Option<&WorkspaceMember> {
        let positions = self
            .by_path
            .get(path)
            .and_then(|file| self.members_by_file.get(file))?;
        let after =
            positions.partition_point(|&position| self.members[position].decl_start <= offset);
        let member = &self.members[*positions.get(after.checked_sub(1)?)?];
        (offset < member.decl_end).then_some(member)
    }

    /// The recorded member selections the file at `path` makes between
    /// `start` and `end`, in bytes, in order.
    pub fn member_refs_between(
        &self,
        path: &Path,
        start: u32,
        end: u32,
    ) -> impl Iterator<Item = &WorkspaceMemberRef> {
        let positions = self
            .by_path
            .get(path)
            .and_then(|file| self.refs_by_file.get(file))
            .map(Vec::as_slice)
            .unwrap_or_default();
        let from =
            positions.partition_point(|&position| self.member_refs[position].start_byte < start);
        positions[from..]
            .iter()
            .map(|&position| &self.member_refs[position])
            .take_while(move |found| found.start_byte < end)
    }

    /// Every member `owner` itself declares, in source order — not the ones it
    /// inherits, which are the member closure's to walk.
    pub fn members_of(&self, owner: &str) -> &[WorkspaceMember] {
        self.by_owner
            .get(owner)
            .map_or(&[], |range| &self.members[range.clone()])
    }

    /// The constructors `owner` writes, in source order — none where they
    /// may not be all it has (`javelin_classpath::written_constructors`).
    pub fn constructors_of(&self, owner: &str) -> &[WorkspaceMember] {
        self.constructors_by_owner
            .get(owner)
            .map_or(&[], |range| &self.constructors[range.clone()])
    }

    /// Every member the workspace declares.
    pub fn members(&self) -> &[WorkspaceMember] {
        &self.members
    }

    /// The file at `path`, with the package and imports a name written in it
    /// resolves against.
    pub fn file_of(&self, path: &Path) -> Option<&WorkspaceFile> {
        self.by_path
            .get(path)
            .map(|&position| &self.sources[position])
    }

    /// Every recorded use of the member whose name starts at `target_start`
    /// in `target_source`.
    pub fn member_refs_to(
        &self,
        target_source: &str,
        target_start: u32,
    ) -> impl Iterator<Item = &WorkspaceMemberRef> {
        self.by_target
            .get(&(target_source.to_owned(), target_start))
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .map(|&position| &self.member_refs[position])
    }

    /// Add member references to an index, each `file` a position in its
    /// sources.
    pub fn with_member_refs(mut self, refs: Vec<WorkspaceMemberRef>) -> Self {
        let mut by_target: HashMap<(String, u32), Vec<usize>> = HashMap::new();
        let mut refs_by_file: HashMap<usize, Vec<usize>> = HashMap::new();
        for (position, found) in refs.iter().enumerate() {
            by_target
                .entry((found.target_source.clone(), found.target_start))
                .or_default()
                .push(position);
            refs_by_file.entry(found.file).or_default().push(position);
        }
        // By offset, which is what `member_refs_between` searches.
        for positions in refs_by_file.values_mut() {
            positions.sort_by_key(|&position| refs[position].start_byte);
        }
        self.member_refs = refs;
        self.by_target = by_target;
        self.refs_by_file = refs_by_file;
        self
    }

    /// The file an occurrence is in.
    ///
    /// Panics on an index no occurrence of this `WorkspaceIndex` carries, the
    /// way a slice does: an occurrence and its file are built together.
    pub fn file(&self, index: usize) -> &WorkspaceFile {
        &self.sources[index]
    }

    /// Every source file's package and imports.
    pub fn sources(&self) -> &[WorkspaceFile] {
        &self.sources
    }

    fn at<'a>(
        &'a self,
        positions: Option<&'a Vec<usize>>,
    ) -> impl Iterator<Item = &'a WorkspaceType> {
        positions
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .map(|&position| &self.types[position])
    }

    /// Up to `limit` types whose simple name contains `query`, ignoring ASCII
    /// case, with the ones that *start* with it first.
    ///
    /// An empty query matches everything, which is what a symbol picker sends
    /// before the user has typed. Two tiers rather than a ranking: every
    /// client that has a symbol picker re-orders the list against its own
    /// fuzzy matcher, so ranking here would be work thrown away. What the
    /// tiers buy is the *cap* landing on the right side — `Order` before
    /// `WorkOrder` matters only because the list is truncated.
    ///
    /// Case folding is ASCII-only. A Java identifier may hold any letter
    /// Unicode calls one, so `Straße` matches `STRA` and not `STRASSE`;
    /// folding properly means a dependency or a table, for a case no
    /// workspace in this suite has.
    ///
    /// ponytail: a linear scan of every indexed name, which is the answer's
    /// cost rather than the file's — so it belongs on the request loop — but
    /// it is proportional to the *workspace* rather than to `limit`. A
    /// million-type monorepo scans a million names per keystroke. The upgrade
    /// is a prefix index over the simple names, which only pays for itself
    /// once the scan is measurably too slow; `benchmarks/benches/workspace_index.rs`
    /// is where that becomes visible.
    pub fn matching(&self, query: &str, limit: usize) -> Vec<&WorkspaceType> {
        tiered(&self.types, |ty| ty.simple.as_str(), query, limit)
    }

    /// [`matching`](Self::matching) over members' names: a method, a field,
    /// an enum constant, in the same two tiers and under the same cap.
    pub fn matching_members(&self, query: &str, limit: usize) -> Vec<&WorkspaceMember> {
        tiered(&self.members, |member| member.name.as_str(), query, limit)
    }

    /// How many types are indexed.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// How many distinct files those types came from.
    pub fn files(&self) -> usize {
        self.files
    }
}

impl CacheDb {
    /// Bring the store level with `found`, and hand back what it now holds.
    ///
    /// Every file in `found` whose modification time and size match what was
    /// already stored is kept without being opened; every other one is parsed
    /// and replaces its rows; every stored path `found` does not mention is
    /// deleted, because the file is gone, renamed, or now behind a prune rule.
    ///
    /// Cancellation stops the walk between batches and returns what the store
    /// holds at that point rather than an error: a session ending mid-index is
    /// normal, and the next one continues from the prefix this one committed.
    pub fn index_sources(
        &mut self,
        found: &[Found],
        cancel: &CancellationToken,
    ) -> Result<WorkspaceIndex> {
        self.walk_into_store(found, cancel)?;
        self.published()
    }

    /// The walk's writes, and which files it re-read and found gone.
    fn walk_into_store(
        &mut self,
        found: &[Found],
        cancel: &CancellationToken,
    ) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
        let mut changed = Vec::new();
        let stored = self.stored_files()?;
        let seen: HashMap<&str, &Found> = found
            .iter()
            .filter_map(|file| file.path.to_str().map(|path| (path, file)))
            .collect();

        let gone: Vec<String> = stored
            .keys()
            .filter(|path| !seen.contains_key(path.as_str()))
            .cloned()
            .collect();
        let stale: Vec<&Found> = found
            .iter()
            .filter(
                |file| match file.path.to_str().and_then(|path| stored.get(path)) {
                    Some(&(mtime_ns, size)) => mtime_ns != file.mtime_ns || size != file.size,
                    None => true,
                },
            )
            .collect();

        if !gone.is_empty() {
            let transaction = self.connection.transaction().map_err(to_index_error)?;
            for path in &gone {
                forget(&transaction, path)?;
            }
            transaction.commit().map_err(to_index_error)?;
        }

        // Parsed on every core but one — a cold walk of a real workspace is
        // seconds on one — and stored here, where the connection is, a batch
        // per transaction. Parsers are built only where something needs
        // parsing: a warm start that skips every file must not pay for one.
        #[cfg(not(target_family = "wasm"))]
        let threads = std::thread::available_parallelism()
            .map_or(1, |cores| cores.get().saturating_sub(1).max(1))
            .min(stale.len());
        #[cfg(not(target_family = "wasm"))]
        let parsers = (0..threads)
            .map(|_| JavaParser::new())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        #[cfg(not(target_family = "wasm"))]
        let next = std::sync::atomic::AtomicUsize::new(0);
        #[cfg(not(target_family = "wasm"))]
        std::thread::scope(|scope| {
            // Inside the scope, so an early return drops the receiver and
            // every thread stops at its next send.
            let (sender, parsed) = std::sync::mpsc::sync_channel(BATCH * 2);
            for mut parser in parsers {
                let (sender, stale, next) = (sender.clone(), &stale, &next);
                scope.spawn(move || {
                    while !cancel.is_cancelled() {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(file) = stale.get(at) else {
                            return;
                        };
                        if sender
                            .send((at, read_facts(file, &mut parser, cancel)))
                            .is_err()
                        {
                            return;
                        }
                    }
                });
            }
            drop(sender);
            // Stored in walk order whatever order the threads finish in, as
            // the sequential walk stored them: readers list rows in the
            // order they were written.
            let mut ahead: BTreeMap<usize, Option<SourceFacts>> = BTreeMap::new();
            let mut due = 0;
            let mut batch: Vec<SourceFacts> = Vec::with_capacity(BATCH);
            for (at, facts) in parsed.iter() {
                ahead.insert(at, facts);
                while let Some(facts) = ahead.remove(&due) {
                    due += 1;
                    batch.extend(facts);
                    if batch.len() == BATCH {
                        self.store_facts(&batch, &mut changed)?;
                        batch.clear();
                    }
                }
            }
            self.store_facts(&batch, &mut changed)
        })?;
        // A target with no threads, `wasm32-wasip1`, parses the same stale files
        // one after another and stores them in the same order.
        #[cfg(target_family = "wasm")]
        {
            let mut parser = JavaParser::new()?;
            let mut batch: Vec<SourceFacts> = Vec::with_capacity(BATCH);
            for file in &stale {
                if cancel.is_cancelled() {
                    break;
                }
                batch.extend(read_facts(file, &mut parser, cancel));
                if batch.len() == BATCH {
                    self.store_facts(&batch, &mut changed)?;
                    batch.clear();
                }
            }
            self.store_facts(&batch, &mut changed)?;
        }

        Ok((changed, gone.into_iter().map(PathBuf::from).collect()))
    }

    /// One transaction of the walk's writes, each file's old rows replaced.
    fn store_facts(&mut self, facts: &[SourceFacts], changed: &mut Vec<PathBuf>) -> Result<()> {
        if facts.is_empty() {
            return Ok(());
        }
        let transaction = self.connection.transaction().map_err(to_index_error)?;
        for file in facts {
            let path = file.path.to_string_lossy();
            forget(&transaction, &path)?;
            remember(&transaction, &path, file)?;
        }
        transaction.commit().map_err(to_index_error)?;
        changed.extend(facts.iter().map(|file| file.path.clone()));
        Ok(())
    }

    /// Every file the store holds with its resolution context, and every
    /// occurrence in them, each pointing at its file's position in the first.
    #[allow(clippy::type_complexity)]
    fn read_source_files(
        &self,
    ) -> Result<(
        Vec<WorkspaceFile>,
        Vec<WorkspaceOccurrence>,
        Vec<WorkspaceMember>,
    )> {
        let mut sources = Vec::new();
        let mut positions: HashMap<String, usize> = HashMap::new();
        let mut statement = self
            .connection
            .prepare("SELECT path, package, parsed_cleanly FROM source_files ORDER BY path")
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        while let Some(row) = rows.next().map_err(to_index_error)? {
            let path: String = row.get(0).map_err(to_index_error)?;
            positions.insert(path.clone(), sources.len());
            sources.push(WorkspaceFile {
                path: PathBuf::from(path),
                package: row.get(1).map_err(to_index_error)?,
                imports: Vec::new(),
                parsed_cleanly: row.get::<_, i64>(2).map_err(to_index_error)? != 0,
            });
        }

        let mut statement = self
            .connection
            .prepare(
                "SELECT path, name, is_static, is_on_demand FROM source_imports
                 ORDER BY path, position",
            )
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        while let Some(row) = rows.next().map_err(to_index_error)? {
            let path: String = row.get(0).map_err(to_index_error)?;
            let Some(&file) = positions.get(&path) else {
                continue;
            };
            sources[file].imports.push(ImportDecl {
                name: row.get(1).map_err(to_index_error)?,
                span: unplaced(),
                name_span: unplaced(),
                is_static: row.get(2).map_err(to_index_error)?,
                is_on_demand: row.get(3).map_err(to_index_error)?,
            });
        }

        let mut statement = self
            .connection
            .prepare(
                "SELECT path, name, start_byte, end_byte, line_start, char_start,
                        line_end, char_end, fqn
                 FROM source_occurrences",
            )
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        let mut occurrences = Vec::new();
        while let Some(row) = rows.next().map_err(to_index_error)? {
            let path: String = row.get(0).map_err(to_index_error)?;
            let Some(&file) = positions.get(&path) else {
                continue;
            };
            occurrences.push(WorkspaceOccurrence {
                file,
                name: row.get(1).map_err(to_index_error)?,
                start_byte: row.get(2).map_err(to_index_error)?,
                end_byte: row.get(3).map_err(to_index_error)?,
                range: Range::new(
                    Position::new(
                        row.get(4).map_err(to_index_error)?,
                        row.get(5).map_err(to_index_error)?,
                    ),
                    Position::new(
                        row.get(6).map_err(to_index_error)?,
                        row.get(7).map_err(to_index_error)?,
                    ),
                ),
                fqn: row.get(8).map_err(to_index_error)?,
            });
        }
        let mut statement = self
            .connection
            .prepare(
                "SELECT path, owner, name, kind, is_static, declared, arity, params, variadic,
                        name_start, name_end, line_start, char_start, line_end, char_end,
                        is_private, type_params, decl_start, decl_end, synthetic, param_names,
                        doc_start, doc_end, is_public
                 FROM source_members ORDER BY path, owner, position",
            )
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        let mut members = Vec::new();
        while let Some(row) = rows.next().map_err(to_index_error)? {
            let path: String = row.get(0).map_err(to_index_error)?;
            let Some(&file) = positions.get(&path) else {
                continue;
            };
            let stored_kind: i64 = row.get(3).map_err(to_index_error)?;
            let Some(kind) = from_kind_code(stored_kind) else {
                continue;
            };
            let arity: i64 = row.get(6).map_err(to_index_error)?;
            let params: String = row.get(7).map_err(to_index_error)?;
            let names: String = row.get(20).map_err(to_index_error)?;
            let doc_start: Option<u32> = row.get(21).map_err(to_index_error)?;
            let doc_end: Option<u32> = row.get(22).map_err(to_index_error)?;
            members.push(WorkspaceMember {
                doc_span: doc_start.zip(doc_end),
                parameter_names: decode_names(&names),
                owner: row.get(1).map_err(to_index_error)?,
                file,
                name: row.get(2).map_err(to_index_error)?,
                kind,
                is_static: row.get(4).map_err(to_index_error)?,
                declared: row.get(5).map_err(to_index_error)?,
                parameters: decode_parameters(arity, &params),
                variadic: row.get(8).map_err(to_index_error)?,
                name_start: row.get(9).map_err(to_index_error)?,
                name_end: row.get(10).map_err(to_index_error)?,
                name_range: Range::new(
                    Position::new(
                        row.get(11).map_err(to_index_error)?,
                        row.get(12).map_err(to_index_error)?,
                    ),
                    Position::new(
                        row.get(13).map_err(to_index_error)?,
                        row.get(14).map_err(to_index_error)?,
                    ),
                ),
                is_private: row.get(15).map_err(to_index_error)?,
                type_parameters: split_names(&row.get::<_, String>(16).map_err(to_index_error)?),
                decl_start: row.get(17).map_err(to_index_error)?,
                decl_end: row.get(18).map_err(to_index_error)?,
                synthetic: row.get(19).map_err(to_index_error)?,
                is_public: row.get(23).map_err(to_index_error)?,
            });
        }
        Ok((sources, occurrences, members))
    }

    /// Replace each file's member references and dependencies with these,
    /// resolved against the world `state` names — see [`REFS_WITH_JDK`].
    ///
    /// Marks each file resolved in the same transaction, which is what
    /// [`owed_member_refs`](Self::owed_member_refs) reads.
    pub fn store_member_refs(
        &mut self,
        files: &[(PathBuf, Vec<MemberRef>, Vec<String>)],
        state: i64,
    ) -> Result<()> {
        let transaction = self.connection.transaction().map_err(to_index_error)?;
        for (path, refs, deps) in files {
            let path = path.to_string_lossy();
            transaction
                .execute(
                    "UPDATE source_files SET member_refs = ?2 WHERE path = ?1",
                    (path.as_ref(), state),
                )
                .map_err(to_index_error)?;
            for statement in [
                "DELETE FROM source_member_refs WHERE path = ?1",
                "DELETE FROM source_member_deps WHERE path = ?1",
            ] {
                transaction
                    .execute(statement, [path.as_ref()])
                    .map_err(to_index_error)?;
            }
            let mut insert = transaction
                .prepare_cached(
                    "INSERT INTO source_member_refs
                         (path, name, start_byte, end_byte, line_start, char_start, line_end,
                          char_end, target_source, target_start)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                )
                .map_err(to_index_error)?;
            for found in refs {
                insert
                    .execute(rusqlite::params![
                        path.as_ref(),
                        &found.name,
                        found.start_byte,
                        found.end_byte,
                        found.range.start.line,
                        found.range.start.character,
                        found.range.end.line,
                        found.range.end.character,
                        &found.target_source,
                        found.target_start,
                    ])
                    .map_err(to_index_error)?;
            }
            let mut insert = transaction
                .prepare_cached("INSERT INTO source_member_deps (path, fqn) VALUES (?1, ?2)")
                .map_err(to_index_error)?;
            for fqn in deps {
                insert
                    .execute((path.as_ref(), fqn))
                    .map_err(to_index_error)?;
            }
        }
        transaction.commit().map_err(to_index_error)
    }

    /// Mark `paths` as owing their member references.
    ///
    /// Called before a pass re-resolves files whose own rows are unchanged —
    /// dependents of a file that changed — so that a session killed partway
    /// leaves the rest for the next session rather than marked resolved with
    /// answers keyed to offsets that have moved.
    pub fn mark_member_refs_pending(&mut self, paths: &[PathBuf]) -> Result<()> {
        let transaction = self.connection.transaction().map_err(to_index_error)?;
        for path in paths {
            transaction
                .execute(
                    "UPDATE source_files SET member_refs = 0 WHERE path = ?1",
                    [path.to_string_lossy().as_ref()],
                )
                .map_err(to_index_error)?;
        }
        transaction.commit().map_err(to_index_error)
    }

    /// The stored files whose member references a pass resolving against the
    /// world `state` names still owes: every file resolved against a lesser
    /// world, or for a settled world, against any other. What a session
    /// interrupted before its resolution pass finished leaves behind, since
    /// the next walk finds those files unchanged and re-reads none of them.
    pub fn owed_member_refs(&self, state: i64) -> Result<Vec<PathBuf>> {
        let query = if state > REFS_WITH_JDK {
            "SELECT path FROM source_files WHERE member_refs != ?1 ORDER BY path"
        } else {
            "SELECT path FROM source_files WHERE member_refs < ?1 ORDER BY path"
        };
        let mut statement = self.connection.prepare(query).map_err(to_index_error)?;
        let rows = statement
            .query_map([state], |row| row.get::<_, String>(0))
            .map_err(to_index_error)?;
        rows.map(|row| row.map(PathBuf::from).map_err(to_index_error))
            .collect()
    }

    /// Every file whose member references depended on one of `fqns`.
    pub fn dependents(&self, fqns: &[String]) -> Result<Vec<PathBuf>> {
        let mut found = std::collections::BTreeSet::new();
        let mut statement = self
            .connection
            .prepare_cached("SELECT DISTINCT path FROM source_member_deps WHERE fqn = ?1")
            .map_err(to_index_error)?;
        for fqn in fqns {
            let mut rows = statement.query([fqn]).map_err(to_index_error)?;
            while let Some(row) = rows.next().map_err(to_index_error)? {
                found.insert(row.get::<_, String>(0).map_err(to_index_error)?);
            }
        }
        Ok(found.into_iter().map(PathBuf::from).collect())
    }

    /// [`index_sources`](Self::index_sources), also saying which files it
    /// re-read and which it found gone.
    pub fn index_sources_reporting(
        &mut self,
        found: &[Found],
        cancel: &CancellationToken,
    ) -> Result<Walked> {
        let (changed, gone) = self.walk_into_store(found, cancel)?;
        Ok(Walked {
            index: self.published()?,
            changed,
            gone,
        })
    }

    /// The index the store now describes, members and member references
    /// included.
    pub fn workspace_index(&self) -> Result<WorkspaceIndex> {
        self.published()
    }

    fn published(&self) -> Result<WorkspaceIndex> {
        let (sources, mut occurrences, mut members) = self.read_source_files()?;
        let mut refs = self.read_member_refs(&sources)?;
        let mut types = self.read_source_types()?;
        // What Lombok generates below a `lombok.config` that renames it is
        // not what this walk would name it.
        types.retain(|ty| !ty.synthetic || !self.lombok_renamed_at(&ty.file));
        for ty in &mut types {
            ty.lombok.unmodelled |= ty.lombok.annotated && self.lombok_renamed_at(&ty.file);
        }
        // What Java declares implicitly is no Lombok's: an enum's static
        // `values` and `valueOf`, which Lombok writes nothing static of the
        // name of, and a record's accessors, since it writes nothing into a
        // record.
        // Keyed by file too: a class elsewhere declaring the record's fqn
        // keeps its Lombok rows renameable.
        let implicit_in: HashMap<(&Path, &str), DeclKind> = types
            .iter()
            .filter(|ty| matches!(ty.kind, DeclKind::Enum | DeclKind::Record))
            .map(|ty| ((ty.file.as_path(), ty.fqn.as_str()), ty.kind))
            .collect();
        let implicit = |member: &WorkspaceMember| {
            member.synthetic
                && match implicit_in
                    .get(&(sources[member.file].path.as_path(), member.owner.as_str()))
                {
                    Some(DeclKind::Record) => true,
                    Some(_) => {
                        member.is_static && matches!(member.name.as_str(), "values" | "valueOf")
                    }
                    None => false,
                }
        };
        members.retain(|member| {
            !member.synthetic
                || implicit(member)
                || !self.lombok_renamed_at(&sources[member.file].path)
        });
        // The one field Lombok generates is a log annotation's, named where
        // the configs say.
        for member in &mut members {
            if member.synthetic && member.kind == DeclKind::Field {
                member.name =
                    crate::LombokLog::field_name(&self.lombok_log, &sources[member.file].path)
                        .to_owned();
            }
        }
        // And a use of one resolved before the config came: its name is not
        // the written member's at its site.
        if !self.lombok_renamed.is_empty() {
            let written: HashSet<(String, u32, &str)> = members
                .iter()
                .filter(|member| !member.synthetic || implicit(member))
                .map(|member| {
                    (
                        sources[member.file].path.to_string_lossy().into_owned(),
                        member.name_start,
                        member.name.as_str(),
                    )
                })
                .collect();
            refs.retain(|found| {
                !self.lombok_renamed_at(Path::new(&found.target_source))
                    || written.contains(&(
                        found.target_source.clone(),
                        found.target_start,
                        found.name.as_str(),
                    ))
            });
        }
        // A declaration the sources make wins over a generated copy of it: a
        // processor's output is a build's, possibly stale, and without this
        // a hand-written type with a copy left in `build/` would be two
        // declarations — an ambiguity that answers nothing. The copy's file
        // goes whole, uses and all: what it names is the stale copy's view,
        // and its own `class Car` would otherwise be listed as a use of the
        // hand-written one.
        let output: Vec<bool> = sources
            .iter()
            .map(|file| crate::is_processor_output(&file.path))
            .collect();
        let position: HashMap<&Path, usize> = sources
            .iter()
            .enumerate()
            .map(|(at, file)| (file.path.as_path(), at))
            .collect();
        let is_output = |path: &Path| position.get(path).is_some_and(|&at| output[at]);
        let written: HashSet<&str> = types
            .iter()
            .filter(|ty| !is_output(&ty.file))
            .map(|ty| ty.fqn.as_str())
            .collect();
        let stale: HashSet<usize> = types
            .iter()
            .filter(|ty| is_output(&ty.file) && written.contains(ty.fqn.as_str()))
            .filter_map(|ty| position.get(ty.file.as_path()).copied())
            .collect();
        if !stale.is_empty() {
            let stale_paths: HashSet<String> = stale
                .iter()
                .map(|&at| sources[at].path.to_string_lossy().into_owned())
                .collect();
            types.retain(|ty| {
                position
                    .get(ty.file.as_path())
                    .is_none_or(|at| !stale.contains(at))
            });
            members.retain(|member| !stale.contains(&member.file));
            occurrences.retain(|occurrence| !stale.contains(&occurrence.file));
            refs.retain(|found| {
                !stale.contains(&found.file) && !stale_paths.contains(&found.target_source)
            });
        }
        Ok(
            WorkspaceIndex::with_occurrences(types, sources, occurrences)
                .with_members(members)
                .with_member_refs(refs),
        )
    }

    fn read_member_refs(&self, sources: &[WorkspaceFile]) -> Result<Vec<WorkspaceMemberRef>> {
        let positions: HashMap<String, usize> = sources
            .iter()
            .enumerate()
            .map(|(position, file)| (file.path.to_string_lossy().into_owned(), position))
            .collect();
        let mut statement = self
            .connection
            .prepare(
                "SELECT path, name, line_start, char_start, line_end, char_end, target_source,
                        target_start, start_byte
                 FROM source_member_refs",
            )
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        let mut refs = Vec::new();
        while let Some(row) = rows.next().map_err(to_index_error)? {
            let path: String = row.get(0).map_err(to_index_error)?;
            let Some(&file) = positions.get(&path) else {
                continue;
            };
            refs.push(WorkspaceMemberRef {
                file,
                name: row.get(1).map_err(to_index_error)?,
                range: Range::new(
                    Position::new(
                        row.get(2).map_err(to_index_error)?,
                        row.get(3).map_err(to_index_error)?,
                    ),
                    Position::new(
                        row.get(4).map_err(to_index_error)?,
                        row.get(5).map_err(to_index_error)?,
                    ),
                ),
                target_source: row.get(6).map_err(to_index_error)?,
                target_start: row.get(7).map_err(to_index_error)?,
                start_byte: row.get(8).map_err(to_index_error)?,
            });
        }
        Ok(refs)
    }

    /// Apply `changes` and hand back what the store now holds.
    ///
    /// Unlike [`index_sources`](Self::index_sources), nothing is skipped on
    /// its time and size: an event about a file is the file changing, and a
    /// save within the clock's granularity must not be lost to the check that
    /// exists to spare an unchanged file. A buffer's rows are stored under a
    /// time and size of `-1`, which no file has, so the next walk reads the
    /// disk again.
    pub fn refresh(
        &mut self,
        changes: &[SourceChange],
        max_file_size_bytes: u64,
        cancel: &CancellationToken,
    ) -> Result<WorkspaceIndex> {
        let mut parser = JavaParser::new()?;
        let transaction = self.connection.transaction().map_err(to_index_error)?;
        for change in changes {
            if cancel.is_cancelled() {
                break;
            }
            let (path, facts) = match change {
                SourceChange::Disk(path) => (
                    path,
                    crate::walk::measure(path, max_file_size_bytes)
                        .and_then(|found| read_facts(&found, &mut parser, cancel)),
                ),
                SourceChange::Buffer { path, text } => {
                    // Never what the disk holds: `-1` matches no file's
                    // time or size, so the next walk reads the disk again.
                    let found = Found {
                        path: path.clone(),
                        mtime_ns: -1,
                        size: -1,
                    };
                    // The limit a walk applies to a file applies to a buffer:
                    // a generated file too large to index on disk is not
                    // indexed while it is open either.
                    let admitted = text.len() as u64 <= max_file_size_bytes;
                    (
                        path,
                        admitted
                            .then(|| facts_of(&found, text.clone(), &mut parser, cancel))
                            .flatten(),
                    )
                }
            };
            let stored = path.to_string_lossy();
            forget(&transaction, &stored)?;
            // Gone, too large, unreadable or not UTF-8: forgotten, which is
            // what a walk that no longer found it would have done.
            if let Some(facts) = facts {
                remember(&transaction, &stored, &facts)?;
            }
        }
        transaction.commit().map_err(to_index_error)?;
        self.published()
    }

    /// Every type the store holds, for the caller that is building an index
    /// out of it. One ordered pass, grouped by file and name, so a type's
    /// supertypes arrive with it.
    pub fn read_source_types(&self) -> Result<Vec<WorkspaceType>> {
        let mut supertypes: HashMap<(String, String), Vec<String>> = HashMap::new();
        let mut statement = self
            .connection
            .prepare("SELECT path, fqn, written FROM source_supertypes")
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        while let Some(row) = rows.next().map_err(to_index_error)? {
            let path: String = row.get(0).map_err(to_index_error)?;
            let fqn: String = row.get(1).map_err(to_index_error)?;
            let written: String = row.get(2).map_err(to_index_error)?;
            supertypes.entry((path, fqn)).or_default().push(written);
        }

        let mut statement = self
            .connection
            .prepare(
                "SELECT path, fqn, simple, kind, name_start, name_end, decl_start, decl_end,
                        name_line_start, name_char_start, name_line_end, name_char_end,
                        synthetic, doc_start, doc_end, lombok, type_params, is_public
                 FROM source_types",
            )
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        let mut types = Vec::new();
        while let Some(row) = rows.next().map_err(to_index_error)? {
            let path: String = row.get(0).map_err(to_index_error)?;
            let fqn: String = row.get(1).map_err(to_index_error)?;
            let stored_kind: i64 = row.get(3).map_err(to_index_error)?;
            let Some(kind) = from_kind_code(stored_kind) else {
                tracing::debug!(fqn = %fqn, stored_kind, "unknown DeclKind code, row skipped");
                continue;
            };
            let doc_start: Option<u32> = row.get(13).map_err(to_index_error)?;
            let doc_end: Option<u32> = row.get(14).map_err(to_index_error)?;
            types.push(WorkspaceType {
                doc_span: doc_start.zip(doc_end),
                supertypes: supertypes
                    .remove(&(path.clone(), fqn.clone()))
                    .unwrap_or_default(),
                fqn,
                simple: row.get(2).map_err(to_index_error)?,
                kind,
                file: PathBuf::from(path),
                name_start: row.get(4).map_err(to_index_error)?,
                name_end: row.get(5).map_err(to_index_error)?,
                decl_start: row.get(6).map_err(to_index_error)?,
                decl_end: row.get(7).map_err(to_index_error)?,
                name_range: Range::new(
                    Position::new(
                        row.get(8).map_err(to_index_error)?,
                        row.get(9).map_err(to_index_error)?,
                    ),
                    Position::new(
                        row.get(10).map_err(to_index_error)?,
                        row.get(11).map_err(to_index_error)?,
                    ),
                ),
                synthetic: row.get(12).map_err(to_index_error)?,
                lombok: LombokUse::from_bits(row.get(15).map_err(to_index_error)?),
                type_parameters: split_names(&row.get::<_, String>(16).map_err(to_index_error)?),
                is_public: row.get(17).map_err(to_index_error)?,
            });
        }
        Ok(types)
    }

    /// How many files and how many types this store holds, for `doctor`.
    ///
    /// A count rather than the rows: the question is whether a previous
    /// session left anything behind, and reading tens of thousands of rows to
    /// answer it would make asking cost more than it is worth.
    pub fn source_counts(&self) -> Result<(u64, u64)> {
        let count = |table: &str| -> Result<u64> {
            self.connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get::<_, i64>(0)
                })
                .map(|count| u64::try_from(count).unwrap_or(0))
                .map_err(to_index_error)
        };
        Ok((count("source_files")?, count("source_types")?))
    }

    fn stored_files(&self) -> Result<HashMap<String, (i64, i64)>> {
        let mut statement = self
            .connection
            .prepare("SELECT path, mtime_ns, size FROM source_files")
            .map_err(to_index_error)?;
        let mut rows = statement.query([]).map_err(to_index_error)?;
        let mut stored = HashMap::new();
        while let Some(row) = rows.next().map_err(to_index_error)? {
            stored.insert(
                row.get(0).map_err(to_index_error)?,
                (
                    row.get(1).map_err(to_index_error)?,
                    row.get(2).map_err(to_index_error)?,
                ),
            );
        }
        Ok(stored)
    }
}

/// Up to `limit` of `items` whose name contains `query`, ignoring ASCII case,
/// the ones that *start* with it first — [`WorkspaceIndex::matching`]'s rule,
/// for any list with a name.
fn tiered<'a, T>(
    items: &'a [T],
    name: impl Fn(&T) -> &str,
    query: &str,
    limit: usize,
) -> Vec<&'a T> {
    let needle = query.as_bytes();
    let mut prefixed: Vec<&T> = Vec::new();
    let mut contained: Vec<&T> = Vec::new();
    for item in items {
        if prefixed.len() >= limit {
            break;
        }
        let name = name(item).as_bytes();
        if name.len() < needle.len() {
            continue;
        }
        if name[..needle.len()].eq_ignore_ascii_case(needle) {
            prefixed.push(item);
        } else if name
            .windows(needle.len())
            .any(|window| window.eq_ignore_ascii_case(needle))
        {
            contained.push(item);
        }
    }
    prefixed.extend(contained);
    prefixed.truncate(limit);
    prefixed
}

/// Drop everything recorded about one path, in every table at once. A file
/// whose rows outlive its `source_files` entry would be indexed forever.
fn forget(connection: &Connection, path: &str) -> Result<()> {
    for statement in [
        "DELETE FROM source_files WHERE path = ?1",
        "DELETE FROM source_types WHERE path = ?1",
        "DELETE FROM source_supertypes WHERE path = ?1",
        "DELETE FROM source_imports WHERE path = ?1",
        "DELETE FROM source_occurrences WHERE path = ?1",
        "DELETE FROM source_members WHERE path = ?1",
        "DELETE FROM source_member_refs WHERE path = ?1",
        "DELETE FROM source_member_deps WHERE path = ?1",
    ] {
        connection
            .execute(statement, [path])
            .map_err(to_index_error)?;
    }
    Ok(())
}

fn remember(connection: &Connection, path: &str, file: &SourceFacts) -> Result<()> {
    connection
        .execute(
            "INSERT INTO source_files (path, mtime_ns, size, package, parsed_cleanly)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            (
                path,
                file.mtime_ns,
                file.size,
                file.package.as_deref(),
                i64::from(file.parsed_cleanly),
            ),
        )
        .map_err(to_index_error)?;
    for ty in &file.types {
        connection
            .execute(
                "INSERT INTO source_types
                     (path, fqn, simple, kind, name_start, name_end, decl_start, decl_end,
                      name_line_start, name_char_start, name_line_end, name_char_end,
                      synthetic, doc_start, doc_end, lombok, type_params, is_public)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                         ?17, ?18)",
                rusqlite::params![
                    path,
                    &ty.fqn,
                    &ty.simple,
                    kind_code(ty.kind),
                    ty.name_start,
                    ty.name_end,
                    ty.decl_start,
                    ty.decl_end,
                    ty.name_range.start.line,
                    ty.name_range.start.character,
                    ty.name_range.end.line,
                    ty.name_range.end.character,
                    ty.synthetic,
                    ty.doc_span.map(|(start, _)| start),
                    ty.doc_span.map(|(_, end)| end),
                    ty.lombok.bits(),
                    ty.type_parameters.join(";"),
                    ty.is_public,
                ],
            )
            .map_err(to_index_error)?;
        for written in &ty.supertypes {
            connection
                .execute(
                    "INSERT INTO source_supertypes (path, fqn, written) VALUES (?1, ?2, ?3)",
                    (path, &ty.fqn, written),
                )
                .map_err(to_index_error)?;
        }
    }
    for (position, import) in file.imports.iter().enumerate() {
        connection
            .execute(
                "INSERT INTO source_imports (path, position, name, is_static, is_on_demand)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                (
                    path,
                    position as i64,
                    &import.name,
                    import.is_static,
                    import.is_on_demand,
                ),
            )
            .map_err(to_index_error)?;
    }
    let mut insert = connection
        .prepare_cached(
            "INSERT INTO source_occurrences
                 (path, name, start_byte, end_byte, line_start, char_start, line_end, char_end,
                  fqn)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )
        .map_err(to_index_error)?;
    for occurrence in &file.occurrences {
        let span = occurrence.span;
        insert
            .execute((
                path,
                &occurrence.name,
                span.start_byte,
                span.end_byte,
                span.range.start.line,
                span.range.start.character,
                span.range.end.line,
                span.range.end.character,
                occurrence.fqn.as_deref(),
            ))
            .map_err(to_index_error)?;
    }
    let mut insert = connection
        .prepare_cached(
            "INSERT INTO source_members
                 (path, owner, position, name, kind, is_static, declared, arity, params,
                  variadic, name_start, name_end, line_start, char_start, line_end, char_end,
                  is_private, type_params, decl_start, decl_end, synthetic, param_names,
                  doc_start, doc_end, is_public)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                     ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25)",
        )
        .map_err(to_index_error)?;
    for (position, (owner, member, synthetic)) in file.members.iter().enumerate() {
        let (arity, params) = encode_parameters(&member.parameters);
        let span = member.name_span;
        insert
            .execute(rusqlite::params![
                path,
                owner,
                position as i64,
                &member.name,
                kind_code(member.kind),
                member.is_static,
                member.declared.as_deref(),
                arity,
                params,
                member.variadic,
                span.start_byte,
                span.end_byte,
                span.range.start.line,
                span.range.start.character,
                span.range.end.line,
                span.range.end.character,
                member.is_private,
                member.type_parameters.join(";"),
                member.span.start_byte,
                member.span.end_byte,
                synthetic,
                member.parameter_names.join(","),
                member.doc_span.map(|(start, _)| start),
                member.doc_span.map(|(_, end)| end),
                member.is_public,
            ])
            .map_err(to_index_error)?;
    }
    Ok(())
}

/// Parameter types joined on `;`, which no Java type in source form contains —
/// the same encoding the JDK's member store uses — with the count stored
/// beside them, so an empty list and one parameter of no written type differ.
fn encode_parameters(parameters: &[String]) -> (i64, String) {
    (parameters.len() as i64, parameters.join(";"))
}

/// Parameter names joined on `,`, which no Java identifier holds — empty for
/// none.
fn decode_names(joined: &str) -> Vec<String> {
    joined
        .split(',')
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Type parameter names stored joined by `;`, none for an empty string.
fn split_names(joined: &str) -> Vec<String> {
    joined
        .split(';')
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

fn decode_parameters(arity: i64, joined: &str) -> Vec<String> {
    if arity == 0 {
        return Vec::new();
    }
    joined.split(';').map(str::to_owned).collect()
}

/// The span a stored import is rebuilt with: nothing reads where it was.
fn unplaced() -> javelin_parser::Span {
    javelin_parser::Span {
        start_byte: 0,
        end_byte: 0,
        range: Range::new(Position::new(0, 0), Position::new(0, 0)),
    }
}

/// Read and parse one file. `None` for a file that cannot be read or is not
/// UTF-8 — it costs itself and nothing else, the way an unreadable `src.zip`
/// entry does.
///
/// A walk cancelled partway through a file drops that file rather than storing
/// half its occurrences: it is not recorded as seen, so the next walk reads it
/// again.
fn read_facts(
    file: &Found,
    parser: &mut JavaParser,
    cancel: &CancellationToken,
) -> Option<SourceFacts> {
    let bytes = std::fs::read(&file.path)
        .inspect_err(|err| tracing::debug!(path = %file.path.display(), %err, "workspace: unread"))
        .ok()?;
    facts_of(file, decode_source(bytes), parser, cancel)
}

/// A source file's text: UTF-8, or else Latin-1, one character per byte.
///
/// A Latin-1 file is common in older code — `class Größe` included — and
/// dropping it would make every type it declares unknown. A line's UTF-16
/// columns are the same either way; a byte offset after a non-ASCII
/// character differs from the editor's, if the editor decoded it otherwise.
pub fn decode_source(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).unwrap_or_else(|err| {
        err.into_bytes()
            .iter()
            .map(|&byte| char::from(byte))
            .collect()
    })
}

/// What `text` declares, as the content of `file` — read off disk by
/// [`read_facts`], or an unsaved buffer's.
fn facts_of(
    file: &Found,
    text: String,
    parser: &mut JavaParser,
    cancel: &CancellationToken,
) -> Option<SourceFacts> {
    let document = TextDocument::new(DocumentId::new(file.path.to_string_lossy()), 0, text);
    let tree = parser
        .parse(&document)
        .inspect_err(
            |err| tracing::debug!(path = %file.path.display(), %err, "workspace: unparsed"),
        )
        .ok()?;
    let syntax = FileSyntax::build(&document, &tree);
    let scopes = ScopeTree::build(&document, &tree);
    let occurrences = type_occurrences(&document, &tree, &scopes, &syntax, cancel).ok()?;
    let package = syntax.package.as_ref().map(|package| package.name.clone());
    let named = syntax.named_types();
    let mut members = Vec::new();
    let mut builders = Vec::new();
    let mut lombok = Vec::with_capacity(named.len());
    for ty in &named {
        let written = written_members(ty.decl);
        members.extend(
            written
                .into_iter()
                .map(|member| (ty.fqn.clone(), member, false)),
        );
        // Rows of the constructor kind, which the published index keeps out
        // of every member list.
        members.extend(
            written_constructors(ty.decl)
                .into_iter()
                .map(|constructor| (ty.fqn.clone(), constructor, false)),
        );
        // Stored as generated, so the declaring file's walk asks for them
        // as it asks for Lombok's.
        members.extend(
            implicit_members(ty.decl, &ty.fqn)
                .into_iter()
                .map(|member| (ty.fqn.clone(), member, true)),
        );
        // The types around this one, outermost first: a nested type's fqn
        // continues each of theirs.
        let mut enclosing: Vec<&javelin_parser::NamedType<'_>> = named
            .iter()
            .filter(|outer| {
                ty.fqn
                    .strip_prefix(outer.fqn.as_str())
                    .is_some_and(|rest| rest.starts_with('.'))
            })
            .collect();
        enclosing.sort_by_key(|outer| outer.fqn.len());
        let enclosing: Vec<&javelin_parser::Decl> =
            enclosing.iter().map(|outer| outer.decl).collect();
        let generated = lombok_members(ty.decl, &enclosing, &syntax.imports);
        lombok.push(generated.lombok);
        members.extend(
            generated
                .members
                .into_iter()
                .map(|member| (ty.fqn.clone(), member, true)),
        );
        for builder in generated.builders {
            let fqn = format!("{}.{}", ty.fqn, builder.simple);
            members.extend(
                builder
                    .members
                    .into_iter()
                    .map(|member| (fqn.clone(), member, true)),
            );
            let decl = ty.decl;
            builders.push(WorkspaceType {
                doc_span: None,
                fqn,
                simple: builder.simple,
                kind: DeclKind::Class,
                file: file.path.clone(),
                name_start: decl.name_span.start_byte,
                name_end: decl.name_span.end_byte,
                decl_start: decl.name_span.start_byte,
                decl_end: decl.name_span.start_byte,
                name_range: decl.name_span.range,
                supertypes: builder.supertypes,
                synthetic: true,
                lombok: LombokUse::default(),
                // Lombok's builder is `public static` unless `@Builder(access)`
                // says not, which is not read, so it is never offered for import.
                is_public: false,
                type_parameters: builder.type_parameters,
            });
        }
    }
    // A member type of an interface or an annotation type is public unwritten.
    let implicitly_public: HashSet<String> = named
        .iter()
        .filter(|named| {
            matches!(
                named.decl.kind,
                DeclKind::Interface | DeclKind::AnnotationType
            )
        })
        .map(|named| named.fqn.clone())
        .collect();
    let mut types: Vec<WorkspaceType> = named
        .into_iter()
        .zip(lombok)
        .map(|(named, lombok)| {
            let decl = named.decl;
            let is_public = decl.modifiers.iter().any(|modifier| modifier == "public")
                || named
                    .fqn
                    .rsplit_once('.')
                    .is_some_and(|(outer, _)| implicitly_public.contains(outer));
            let mut supertypes = Vec::new();
            supertypes.extend(decl.extends.clone());
            supertypes.extend(decl.implements.iter().cloned());
            WorkspaceType {
                doc_span: decl.doc_span,
                fqn: named.fqn,
                simple: decl.name.clone(),
                kind: decl.kind,
                file: file.path.clone(),
                name_start: decl.name_span.start_byte,
                name_end: decl.name_span.end_byte,
                decl_start: decl.span.start_byte,
                decl_end: decl.span.end_byte,
                name_range: decl.name_span.range,
                supertypes,
                synthetic: false,
                lombok,
                is_public,
                type_parameters: type_parameter_names(decl.type_parameters.as_deref()),
            }
        })
        .collect();
    types.extend(builders);
    Some(SourceFacts {
        path: file.path.clone(),
        mtime_ns: file.mtime_ns,
        size: file.size,
        parsed_cleanly: !tree.has_errors(),
        types,
        imports: syntax.imports.clone(),
        occurrences,
        members,
        package,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walk::java_sources;

    struct Tree(PathBuf);

    impl Tree {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "javelin-workspace-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("temp tree");
            Self(path)
        }

        fn file(&self, relative: &str, text: &str) -> PathBuf {
            // Spelled as the walk spells it: rows are keyed by the path's text,
            // and on Windows `join("src/A.java")` keeps the `/`.
            let path: PathBuf = self.0.join(relative).components().collect();
            std::fs::create_dir_all(path.parent().expect("a parent")).expect("dirs");
            std::fs::write(&path, text).expect("write");
            path
        }

        fn found(&self) -> Vec<Found> {
            java_sources(
                std::slice::from_ref(&self.0),
                1 << 20,
                &CancellationToken::new(),
            )
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn index(db: &mut CacheDb, tree: &Tree) -> WorkspaceIndex {
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing")
    }

    fn one<'a>(index: &'a WorkspaceIndex, fqn: &str) -> &'a WorkspaceType {
        let mut found = index.with_fqn(fqn);
        let ty = found
            .next()
            .unwrap_or_else(|| panic!("no type named {fqn}"));
        assert!(found.next().is_none(), "{fqn} is declared more than once");
        ty
    }

    /// A copy of a hand-written type left in a processor's output — a
    /// processor that once generated it, a source moved out of `build/` — is
    /// the build's: the source's declaration and members win, rather than the
    /// two being an ambiguity that answers nothing.
    #[test]
    fn a_source_wins_over_a_processors_copy_of_the_same_type() {
        let tree = Tree::new("processor-copy");
        let source = tree.file(
            "src/p/Car.java",
            "package p;\npublic class Car { int speed; }",
        );
        tree.file(
            "build/generated/sources/annotationProcessor/java/main/p/Car.java",
            "package p;\npublic class Car { int stale; Car next; }",
        );
        let generated = tree.file(
            "build/generated/sources/annotationProcessor/java/main/p/CarImpl.java",
            "package p;\npublic class CarImpl { int wheels; }",
        );
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(one(&index, "p.Car").file, source);
        let names: Vec<&str> = index
            .members_of("p.Car")
            .iter()
            .map(|member| member.name.as_str())
            .collect();
        assert_eq!(names, vec!["speed"]);
        let uses: Vec<&Path> = index
            .occurrences_named("Car")
            .map(|occurrence| index.file(occurrence.file).path.as_path())
            .collect();
        assert!(
            uses.iter().all(|path| *path == source),
            "the copy's names are the copy's: {uses:?}"
        );
        assert_eq!(
            one(&index, "p.CarImpl").file,
            generated,
            "output alone is kept"
        );
        assert_eq!(index.members_of("p.CarImpl").len(), 1);
    }

    const LOMBOK_CAR: &str = "package p;\nimport lombok.Builder;\nimport lombok.Getter;\n\
                              @Builder @Getter\npublic class Car { private String name; }\n";

    /// Lombok's getter and builder are members like any other, flagged as
    /// generated and sited at the field and the type they came from — and
    /// the field, not its getter, is the member its site names.
    #[test]
    fn what_lombok_generates_is_indexed_at_what_generated_it() {
        let tree = Tree::new("lombok");
        let path = tree.file("src/p/Car.java", LOMBOK_CAR);
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        let members = |owner: &str| -> Vec<(String, bool)> {
            index
                .members_of(owner)
                .iter()
                .map(|member| (member.name.clone(), member.synthetic))
                .collect()
        };
        assert_eq!(
            members("p.Car"),
            vec![
                ("name".to_owned(), false),
                ("getName".to_owned(), true),
                ("builder".to_owned(), true),
            ]
        );
        assert_eq!(
            members("p.Car.CarBuilder"),
            vec![("name".to_owned(), true), ("build".to_owned(), true)]
        );
        assert_eq!(
            index.members_of("p.Car.CarBuilder")[0].parameter_names,
            vec!["name"],
            "a parameter's name survives the store"
        );
        let builder = one(&index, "p.Car.CarBuilder");
        assert!(builder.synthetic && builder.name_range == one(&index, "p.Car").name_range);
        let field = &index.members_of("p.Car")[0];
        let at = index
            .member_at(&path, field.name_start)
            .expect("the field's site");
        assert!(!at.synthetic, "the site names the field");
    }

    /// A `lombok.config` that renames what Lombok generates: what was read
    /// is not what Lombok writes, so none of it is published below it.
    #[test]
    fn a_lombok_config_that_renames_hides_what_lombok_generates_below_it() {
        let tree = Tree::new("lombok-config");
        tree.file("a/src/p/Car.java", LOMBOK_CAR);
        tree.file(
            "b/src/q/Car.java",
            &LOMBOK_CAR.replace("package p", "package q"),
        );
        tree.file(
            "a/lombok.config",
            "# fluent\nlombok.accessors.fluent = true\n",
        );
        tree.file(
            "b/lombok.config",
            "lombok.getter.flagUsage = warning\nconfig.stopBubbling = true\n",
        );
        let (found, survey) = crate::java_sources_surveyed(
            std::slice::from_ref(&tree.0),
            1 << 20,
            &CancellationToken::new(),
        );
        assert_eq!(
            survey.lombok_renamed.iter().collect::<Vec<_>>(),
            vec![&tree.0.join("a")]
        );
        let mut db = CacheDb::in_memory().expect("db");
        db.set_lombok_renamed(survey.lombok_renamed.iter().cloned());
        let index = db
            .index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        assert_eq!(index.members_of("p.Car").len(), 1, "the field alone");
        assert!(index.with_fqn("p.Car.CarBuilder").next().is_none());
        assert_eq!(index.members_of("q.Car").len(), 3, "a flag only warns");
        let lombok = |fqn: &str| one(&index, fqn).lombok;
        assert!(
            lombok("p.Car").unmodelled,
            "what Lombok writes there is not what was read"
        );
        assert!(lombok("q.Car").annotated && !lombok("q.Car").unmodelled);
    }

    /// A log annotation's field is called what the closest `lombok.config`
    /// naming it says, short of one that stops the search, and `log`
    /// where one clears it.
    #[test]
    fn a_log_field_is_named_by_the_closest_lombok_config() {
        let tree = Tree::new("lombok-log");
        for (directory, package) in [("a", "pa"), ("b", "pb"), ("c", "pc"), ("d", "pd")] {
            tree.file(
                &format!("{directory}/src/{package}/A.java"),
                &format!(
                    "package {package};\nimport lombok.extern.slf4j.Slf4j;\n@Slf4j class A {{}}\n"
                ),
            );
        }
        tree.file("lombok.config", "lombok.log.fieldName = logger\n");
        tree.file("a/lombok.config", "# loud\nLombok.Log.FieldName = LOG\n");
        tree.file("b/lombok.config", "config.stopBubbling = true\n");
        tree.file("d/lombok.config", "clear lombok.log.fieldName\n");
        let (found, survey) = crate::java_sources_surveyed(
            std::slice::from_ref(&tree.0),
            1 << 20,
            &CancellationToken::new(),
        );
        assert!(survey.lombok_renamed.is_empty(), "nothing else is renamed");
        let mut db = CacheDb::in_memory().expect("db");
        db.set_lombok_log(survey.lombok_log.clone());
        let index = db
            .index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        let field = |fqn: &str| {
            let members = index.members_of(fqn);
            assert_eq!(members.len(), 1, "{fqn}");
            (
                members[0].name.clone(),
                members[0].is_private,
                members[0].is_static,
            )
        };
        assert_eq!(field("pa.A"), ("LOG".to_owned(), true, true));
        assert_eq!(field("pb.A").0, "log", "a config that stops the search");
        assert_eq!(field("pc.A").0, "logger");
        assert_eq!(field("pd.A").0, "log", "a config that clears it");
    }

    /// An enum's `values()` and `valueOf` and a record's accessors are stored
    /// as generated, and a `lombok.config` that renames what Lombok writes
    /// leaves them, since Java writes them.
    #[test]
    fn what_java_declares_implicitly_is_stored_and_no_config_renames_it() {
        let tree = Tree::new("implicit");
        tree.file(
            "src/p/Day.java",
            "package p;\nimport lombok.Getter;\n@Getter enum Day { MON; int n; }\n",
        );
        tree.file(
            "src/p/Entry.java",
            "package p;\nrecord Entry(String date) {}\n",
        );
        // Another file declaring the record's fqn: its getter is Lombok's.
        tree.file(
            "src/q/Entry.java",
            "package p;\nimport lombok.Getter;\n@Getter class Entry { int k; }\n",
        );
        tree.file("lombok.config", "lombok.accessors.fluent = true\n");
        let (found, survey) = crate::java_sources_surveyed(
            std::slice::from_ref(&tree.0),
            1 << 20,
            &CancellationToken::new(),
        );
        let members = |renamed: bool| {
            let mut db = CacheDb::in_memory().expect("db");
            if renamed {
                db.set_lombok_renamed(survey.lombok_renamed.iter().cloned());
            }
            let index = db
                .index_sources(&found, &CancellationToken::new())
                .expect("indexing");
            ["p.Day", "p.Entry"]
                .iter()
                .flat_map(|owner| index.members_of(owner).to_vec())
                .map(|member| format!("{}{}", member.name, if member.synthetic { "*" } else { "" }))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            members(false),
            vec![
                "MON", "n", "values*", "valueOf*", "getN*", "date", "date*", "k", "getK*"
            ]
        );
        assert_eq!(
            members(true),
            vec!["MON", "n", "values*", "valueOf*", "date", "date*", "k"],
            "Lombok's getter goes, Java's members stay"
        );
    }

    /// A getter call resolved before a renaming `lombok.config` arrived is
    /// a use of a member that is no longer believed; a use of the field
    /// itself still is.
    #[test]
    fn a_use_of_a_generated_member_goes_with_it_below_a_renaming_config() {
        let tree = Tree::new("lombok-config-refs");
        let path = tree.file("src/p/Car.java", LOMBOK_CAR);
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        let field = index.members_of("p.Car")[0].name_start;
        let source = path.to_string_lossy().into_owned();
        let use_of = |name: &str| MemberRef {
            name: name.to_owned(),
            start_byte: 0,
            end_byte: 0,
            range: Range::new(Position::new(0, 0), Position::new(0, 0)),
            target_source: source.clone(),
            target_start: field,
        };
        db.store_member_refs(
            &[(
                path.clone(),
                vec![use_of("getName"), use_of("name")],
                Vec::new(),
            )],
            REFS_WITH_JDK,
        )
        .expect("stored");
        let names = |db: &CacheDb| -> Vec<String> {
            let index = db.workspace_index().expect("readable");
            index
                .member_refs_to(&source, field)
                .map(|found| found.name.clone())
                .collect()
        };
        assert_eq!(names(&db), vec!["getName", "name"]);
        db.set_lombok_renamed([tree.0.clone()]);
        assert_eq!(names(&db), vec!["name"]);
    }

    #[test]
    fn a_types_fully_qualified_name_carries_its_package() {
        let tree = Tree::new("package");
        tree.file("src/A.java", "package com.example;\nclass A {}");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(one(&index, "com.example.A").simple, "A");
        assert_eq!(index.files(), 1);
    }

    /// A Latin-1 source — an umlaut in a comment — still declares its types:
    /// dropping it would make every use of them elsewhere unknown.
    #[test]
    fn a_file_that_is_not_utf8_is_still_indexed() {
        let tree = Tree::new("latin1");
        let path = tree.file("src/p/Latin.java", "");
        let mut bytes = b"package p;\n// Gr".to_vec();
        bytes.push(0xFC);
        bytes.extend_from_slice(b"n\nclass Latin {}\nclass Gr");
        bytes.push(0xF6);
        bytes.extend_from_slice(b"sse {}\n");
        std::fs::write(&path, bytes).expect("write");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(one(&index, "p.Latin").simple, "Latin");
        assert_eq!(
            one(&index, "p.Grösse").simple,
            "Grösse",
            "a Latin-1 identifier"
        );
    }

    #[test]
    fn a_file_with_no_package_names_its_types_by_their_simple_names() {
        let tree = Tree::new("default-package");
        tree.file("src/A.java", "class A {}");
        let mut db = CacheDb::in_memory().expect("db");
        assert_eq!(one(&index(&mut db, &tree), "A").simple, "A");
    }

    #[test]
    fn a_nested_type_joins_on_a_dot_and_a_local_class_is_not_indexed() {
        let tree = Tree::new("nested");
        tree.file(
            "src/Outer.java",
            "package p;\nclass Outer { class Inner {} void m() { class Local {} } }",
        );
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(one(&index, "p.Outer.Inner").simple, "Inner");
        assert_eq!(index.with_simple_name("Local").count(), 0);
    }

    #[test]
    fn a_supertype_is_stored_as_it_was_written() {
        let tree = Tree::new("supertypes");
        tree.file(
            "src/A.java",
            "package p;\nclass A extends Base<T> implements java.io.Serializable {}",
        );
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(
            one(&index, "p.A").supertypes,
            vec!["Base<T>".to_owned(), "java.io.Serializable".to_owned()]
        );
    }

    #[test]
    fn a_types_name_span_is_where_its_name_starts() {
        let tree = Tree::new("spans");
        let text = "package p;\nclass Alpha {}";
        tree.file("src/Alpha.java", text);
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        let ty = one(&index, "p.Alpha");
        assert_eq!(&text[ty.name_start as usize..ty.name_end as usize], "Alpha");
        assert!(ty.decl_start < ty.name_start && ty.decl_end > ty.name_end);
    }

    #[test]
    fn a_file_whose_time_and_size_have_not_moved_is_not_opened_again() {
        let tree = Tree::new("unchanged");
        tree.file("src/A.java", "package p;\nclass A {}");
        let found = tree.found();
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        // Same length, different content — and the walk's answer is reused, so
        // the store is asked about a file whose bytes have moved underneath it.
        // Reading it again would find `B`; skipping it keeps `A`.
        tree.file("src/A.java", "package p;\nclass B {}");
        let index = db
            .index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        assert_eq!(one(&index, "p.A").simple, "A");
        assert_eq!(index.with_fqn("p.B").count(), 0);
    }

    #[test]
    fn a_changed_file_replaces_what_it_declared() {
        let tree = Tree::new("changed");
        tree.file("src/A.java", "package p;\nclass A {}");
        let mut db = CacheDb::in_memory().expect("db");
        index(&mut db, &tree);
        tree.file("src/A.java", "package p;\nclass Renamed {}");
        let index = index(&mut db, &tree);
        assert_eq!(index.with_fqn("p.A").count(), 0, "the old name is gone");
        assert_eq!(one(&index, "p.Renamed").simple, "Renamed");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn a_deleted_file_loses_its_types() {
        let tree = Tree::new("deleted");
        let path = tree.file("src/A.java", "package p;\nclass A {}");
        tree.file("src/B.java", "package p;\nclass B {}");
        let mut db = CacheDb::in_memory().expect("db");
        assert_eq!(index(&mut db, &tree).len(), 2);
        std::fs::remove_file(&path).expect("remove");
        let index = index(&mut db, &tree);
        assert_eq!(index.with_fqn("p.A").count(), 0);
        assert_eq!(index.len(), 1);
        assert_eq!(index.files(), 1);
    }

    #[test]
    fn two_files_declaring_one_name_both_stay_and_neither_is_picked() {
        let tree = Tree::new("duplicate");
        tree.file("one/A.java", "package p;\nclass A {}");
        tree.file("two/A.java", "package p;\nclass A {}");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(index.with_fqn("p.A").count(), 2);
        assert_eq!(index.files(), 2);
    }

    #[test]
    fn a_file_that_is_not_utf8_costs_itself_and_nothing_else() {
        let tree = Tree::new("not-utf8");
        tree.file("src/A.java", "package p;\nclass A {}");
        std::fs::write(tree.0.join("src/Bad.java"), [0xff, 0xfe, 0x00]).expect("write");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(one(&index, "p.A").simple, "A");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn a_file_mid_edit_keeps_whatever_parsed() {
        let tree = Tree::new("broken");
        tree.file("src/A.java", "package p;\nclass A { void m( }\nclass B {}");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(one(&index, "p.A").simple, "A");
        assert_eq!(one(&index, "p.B").simple, "B");
    }

    #[test]
    fn what_one_session_wrote_the_next_one_reads() {
        let tree = Tree::new("persisted");
        tree.file("src/A.java", "package p;\nclass A {}");
        let cache = tree.0.join("cache.sqlite");
        {
            let mut db = CacheDb::open(&cache).expect("db");
            assert_eq!(index(&mut db, &tree).len(), 1);
        }
        let db = CacheDb::open(&cache).expect("reopened");
        let types = db.read_source_types().expect("read");
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].fqn, "p.A");
    }

    #[test]
    fn a_cancelled_index_stops_rather_than_erroring() {
        let tree = Tree::new("cancelled");
        tree.file("src/A.java", "package p;\nclass A {}");
        let mut db = CacheDb::in_memory().expect("db");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let index = db.index_sources(&tree.found(), &cancel).expect("indexing");
        assert!(index.is_empty());
    }

    #[test]
    fn a_types_name_range_survives_the_store_as_a_line_and_a_character() {
        let tree = Tree::new("ranges");
        tree.file("src/Alpha.java", "package p;\nclass Alpha {}");
        let mut db = CacheDb::in_memory().expect("db");
        // `index_sources` reads its answer back out of SQLite, so asserting on
        // what it returns is asserting on the round trip.
        let index = index(&mut db, &tree);
        let ty = one(&index, "p.Alpha");
        assert_eq!(ty.name_range.start, javelin_document::Position::new(1, 6));
        assert_eq!(ty.name_range.end, javelin_document::Position::new(1, 11));
    }

    #[test]
    fn matching_ignores_ascii_case_in_both_the_query_and_the_name() {
        let tree = Tree::new("case");
        tree.file("src/Widget.java", "package p;\nclass Widget {}");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        for query in ["widget", "WIDGET", "Wid", "dge"] {
            assert_eq!(
                index.matching(query, 10).len(),
                1,
                "{query} must find p.Widget"
            );
        }
        assert!(index.matching("gadget", 10).is_empty());
    }

    #[test]
    fn matching_puts_a_prefix_match_ahead_of_a_mere_substring() {
        let tree = Tree::new("tiers");
        tree.file("src/Order.java", "package p;\nclass Order {}");
        tree.file("src/WorkOrder.java", "package p;\nclass WorkOrder {}");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        let found: Vec<&str> = index
            .matching("order", 10)
            .iter()
            .map(|ty| ty.simple.as_str())
            .collect();
        assert_eq!(found, vec!["Order", "WorkOrder"]);
    }

    #[test]
    fn matching_on_an_empty_query_answers_every_type() {
        let tree = Tree::new("everything");
        tree.file("src/A.java", "package p;\nclass A { class Inner {} }");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(index.matching("", 10).len(), 2);
    }

    #[test]
    fn matching_stops_at_the_limit_it_was_given() {
        let tree = Tree::new("capped");
        for n in 0..5 {
            tree.file(
                &format!("src/A{n}.java"),
                &format!("package p;\nclass A{n} {{}}"),
            );
        }
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(index.matching("A", 3).len(), 3);
        assert_eq!(index.matching("", 0).len(), 0);
    }

    fn occurrences<'a>(index: &'a WorkspaceIndex, name: &str) -> Vec<&'a WorkspaceOccurrence> {
        index.occurrences_named(name).collect()
    }

    #[test]
    fn occurrences_round_trip_through_the_store() {
        let tree = Tree::new("occurrences");
        let text = "package p;\nclass A {}\nclass B { A a; Greeter g; }";
        tree.file("src/A.java", text);
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);

        let a = occurrences(&index, "A");
        assert_eq!(a.len(), 2, "the declaration and the field's type");
        assert!(
            a.iter()
                .all(|occurrence| occurrence.fqn.as_deref() == Some("p.A"))
        );
        let field = a[1];
        assert_eq!(
            &text[field.start_byte as usize..field.end_byte as usize],
            "A"
        );
        assert_eq!(field.range.start, javelin_document::Position::new(2, 10));
        assert_eq!(field.range.end, javelin_document::Position::new(2, 11));

        let greeter = occurrences(&index, "Greeter");
        assert_eq!(greeter.len(), 1);
        assert_eq!(greeter[0].fqn, None, "resolved at query time, not here");
        assert!(
            occurrences(&index, "g").is_empty(),
            "a field's name is a value"
        );
    }

    #[test]
    fn imports_and_the_package_round_trip_as_the_resolution_context() {
        let tree = Tree::new("context");
        tree.file(
            "src/A.java",
            "package p;\nimport java.util.List;\nimport java.io.*;\nimport static q.R.s;\n\
             class A { List l; }",
        );
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        let list = occurrences(&index, "List");
        let names: Vec<&str> = list.iter().map(|found| found.name.as_str()).collect();
        assert_eq!(
            names,
            ["java.util.List", "List"],
            "the import's segment, then the use"
        );
        let file = index.file(list[1].file);
        assert_eq!(file.package.as_deref(), Some("p"));
        let imports: Vec<(&str, bool, bool)> = file
            .imports
            .iter()
            .map(|import| (import.name.as_str(), import.is_static, import.is_on_demand))
            .collect();
        assert_eq!(
            imports,
            vec![
                ("java.util.List", false, false),
                ("java.io", false, true),
                ("q.R.s", true, false),
            ]
        );
        assert!(file.path.ends_with("src/A.java"));
    }

    #[test]
    fn a_changed_file_replaces_its_occurrences() {
        let tree = Tree::new("occurrences-changed");
        tree.file("src/A.java", "class A { Old o; }");
        let mut db = CacheDb::in_memory().expect("db");
        index(&mut db, &tree);
        tree.file("src/A.java", "class A { Renamed r; }");
        let index = index(&mut db, &tree);
        assert!(occurrences(&index, "Old").is_empty());
        assert_eq!(occurrences(&index, "Renamed").len(), 1);
    }

    #[test]
    fn a_deleted_file_loses_its_occurrences() {
        let tree = Tree::new("occurrences-deleted");
        let path = tree.file("src/A.java", "class A { Gone g; }");
        tree.file("src/B.java", "class B { Kept k; }");
        let mut db = CacheDb::in_memory().expect("db");
        index(&mut db, &tree);
        std::fs::remove_file(&path).expect("remove");
        let index = index(&mut db, &tree);
        assert!(occurrences(&index, "Gone").is_empty());
        assert_eq!(occurrences(&index, "Kept").len(), 1);
    }

    #[test]
    fn a_warm_walk_keeps_occurrences_without_reading_the_file() {
        let tree = Tree::new("occurrences-warm");
        tree.file("src/A.java", "class A { Aaa x; }");
        let found = tree.found();
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        // Same length, different content, same recorded time and size: the
        // walk must keep the stored rows rather than re-read.
        tree.file("src/A.java", "class A { Bbb x; }");
        let index = db
            .index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        assert_eq!(occurrences(&index, "Aaa").len(), 1);
        assert!(occurrences(&index, "Bbb").is_empty());
    }

    fn member_summary(
        index: &WorkspaceIndex,
        owner: &str,
    ) -> Vec<(String, Option<String>, Vec<String>, bool, bool)> {
        index
            .members_of(owner)
            .iter()
            .map(|member| {
                (
                    member.name.clone(),
                    member.declared.clone(),
                    member.parameters.clone(),
                    member.is_static,
                    member.variadic,
                )
            })
            .collect()
    }

    #[test]
    fn members_round_trip_with_their_written_types() {
        let tree = Tree::new("members");
        let text = "package p;\nclass Greeter {\n  private int hidden;\n  String name;\n  \
                    static void greet(String who, Object... rest) {}\n  void none() {}\n}";
        tree.file("src/Greeter.java", text);
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(
            member_summary(&index, "p.Greeter"),
            vec![
                (
                    "hidden".to_owned(),
                    Some("int".to_owned()),
                    vec![],
                    false,
                    false
                ),
                (
                    "name".to_owned(),
                    Some("String".to_owned()),
                    vec![],
                    false,
                    false
                ),
                (
                    "greet".to_owned(),
                    Some("void".to_owned()),
                    vec!["String".to_owned(), "Object".to_owned()],
                    true,
                    true
                ),
                (
                    "none".to_owned(),
                    Some("void".to_owned()),
                    vec![],
                    false,
                    false
                ),
            ]
        );
        let members = index.members_of("p.Greeter");
        assert!(members[0].is_private && !members[1].is_private);
        let greet = &members[2];
        assert_eq!(
            &text[greet.name_start as usize..greet.name_end as usize],
            "greet"
        );
        assert_eq!(
            greet.name_range.start,
            javelin_document::Position::new(4, 14)
        );
        assert_eq!(greet.kind, DeclKind::Method);
        assert_eq!(greet.owner, "p.Greeter");
        assert!(index.file(greet.file).path.ends_with("src/Greeter.java"));
    }

    #[test]
    fn a_members_type_parameters_round_trip() {
        let tree = Tree::new("members-type-parameters");
        tree.file(
            "src/Box.java",
            "package p;\nclass Box<T> { T get() { return null; } <U> U map(U u) { return u; } }\n\
             class Pair<K extends Comparable<K>, V> {}",
        );
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(one(&index, "p.Box").type_parameters, vec!["T".to_owned()]);
        assert_eq!(
            one(&index, "p.Pair").type_parameters,
            vec!["K".to_owned(), "V".to_owned()]
        );
        let members = index.members_of("p.Box");
        assert_eq!(members[0].type_parameters, vec!["T".to_owned()]);
        assert_eq!(
            members[1].type_parameters,
            vec!["T".to_owned(), "U".to_owned()]
        );
    }

    #[test]
    fn a_nested_types_members_are_its_own() {
        let tree = Tree::new("members-nested");
        tree.file(
            "src/A.java",
            "package p;\nclass A { int outer; class B { int inner; } }",
        );
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert_eq!(member_summary(&index, "p.A").len(), 1);
        assert_eq!(member_summary(&index, "p.A.B")[0].0, "inner");
    }

    #[test]
    fn a_changed_file_replaces_its_members() {
        let tree = Tree::new("members-changed");
        tree.file("src/A.java", "class A { int old; }");
        let mut db = CacheDb::in_memory().expect("db");
        index(&mut db, &tree);
        tree.file("src/A.java", "class A { int renamed; }");
        let index = index(&mut db, &tree);
        assert_eq!(member_summary(&index, "A")[0].0, "renamed");
        assert_eq!(index.members().len(), 1);
    }

    #[test]
    fn a_deleted_file_loses_its_members() {
        let tree = Tree::new("members-deleted");
        let path = tree.file("src/A.java", "class A { int gone; }");
        let mut db = CacheDb::in_memory().expect("db");
        index(&mut db, &tree);
        std::fs::remove_file(&path).expect("remove");
        assert!(index(&mut db, &tree).members().is_empty());
    }

    #[test]
    fn a_warm_walk_keeps_members() {
        let tree = Tree::new("members-warm");
        tree.file("src/A.java", "class A { int aaa; }");
        let found = tree.found();
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        tree.file("src/A.java", "class A { int bbb; }");
        let index = db
            .index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        assert_eq!(member_summary(&index, "A")[0].0, "aaa");
    }

    #[test]
    fn file_of_finds_the_resolution_context_for_a_type() {
        let tree = Tree::new("file-of");
        tree.file(
            "src/A.java",
            "package p;\nimport java.util.List;\nclass A {}",
        );
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        let ty = one(&index, "p.A");
        let file = index.file_of(&ty.file).expect("the declaring file");
        assert_eq!(file.package.as_deref(), Some("p"));
        assert_eq!(file.imports[0].name, "java.util.List");
    }

    #[test]
    fn a_refreshed_file_is_read_again_even_with_the_same_time_and_size() {
        let tree = Tree::new("refresh-disk");
        let path = tree.file("src/A.java", "package p;\nclass A {}");
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        // Same length: a walk would skip it; an explicit event must not.
        tree.file("src/A.java", "package p;\nclass B {}");
        let index = db
            .refresh(
                &[SourceChange::Disk(path)],
                1 << 20,
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert_eq!(index.with_fqn("p.A").count(), 0);
        assert_eq!(one(&index, "p.B").simple, "B");
    }

    #[test]
    fn a_refreshed_file_that_is_gone_is_forgotten() {
        let tree = Tree::new("refresh-gone");
        let path = tree.file("src/A.java", "package p;\nclass A { Other o; }");
        tree.file("src/B.java", "package p;\nclass B {}");
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        std::fs::remove_file(&path).expect("remove");
        let index = db
            .refresh(
                &[SourceChange::Disk(path)],
                1 << 20,
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert_eq!(index.with_fqn("p.A").count(), 0);
        assert_eq!(index.occurrences_named("Other").count(), 0);
        assert_eq!(one(&index, "p.B").simple, "B");
    }

    /// Error recovery can drop a member a broken file declares, so the file
    /// says whether it parsed — from disk and from a buffer alike.
    #[test]
    fn a_file_that_parses_with_errors_says_so() {
        let tree = Tree::new("parsed-cleanly");
        let clean = tree.file(
            "src/p/Clean.java",
            "package p;\nclass Clean { void a() {} }",
        );
        let broken = tree.file("src/p/Broken.java", "package p;\nclass Broken { void a( }");
        let mut db = CacheDb::in_memory().expect("db");
        let index = index(&mut db, &tree);
        assert!(index.file_of(&clean).expect("clean").parsed_cleanly);
        assert!(!index.file_of(&broken).expect("broken").parsed_cleanly);
        let index = db
            .refresh(
                &[SourceChange::Buffer {
                    path: clean.clone(),
                    text: "package p;\nclass Clean { void a( }".to_owned(),
                }],
                1 << 20,
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert!(!index.file_of(&clean).expect("clean").parsed_cleanly);
    }

    #[test]
    fn a_buffer_over_the_size_limit_is_not_indexed() {
        let tree = Tree::new("refresh-buffer-too-big");
        let path = tree.file("src/A.java", "class A {}");
        let mut db = CacheDb::in_memory().expect("db");
        let index = db
            .refresh(
                &[SourceChange::Buffer {
                    path,
                    text: format!("class Big {{}}\n{}", "// padding\n".repeat(100)),
                }],
                64,
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert_eq!(index.with_fqn("Big").count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_file_is_not_indexed_under_a_second_path() {
        let tree = Tree::new("refresh-symlink");
        tree.file("real/A.java", "package p;\nclass A {}");
        std::os::unix::fs::symlink(tree.0.join("real/A.java"), tree.0.join("Link.java"))
            .expect("symlink");
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        let index = db
            .refresh(
                &[SourceChange::Disk(tree.0.join("Link.java"))],
                1 << 20,
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert_eq!(
            index.with_fqn("p.A").count(),
            1,
            "the walk's copy, and only it"
        );
    }

    #[test]
    fn a_buffer_is_indexed_from_its_text_and_read_from_disk_by_the_next_walk() {
        let tree = Tree::new("refresh-buffer");
        let path = tree.file("src/A.java", "package p;\nclass A {}");
        let found = tree.found();
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        let index = db
            .refresh(
                &[SourceChange::Buffer {
                    path: path.clone(),
                    text: "package p;\nclass A {}\nclass Unsaved {}".to_owned(),
                }],
                1 << 20,
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert_eq!(one(&index, "p.Unsaved").simple, "Unsaved");
        // The disk still says only `A`: the next walk must not trust the
        // buffer's rows as if they were the file.
        let index = db
            .index_sources(&found, &CancellationToken::new())
            .expect("indexing");
        assert_eq!(index.with_fqn("p.Unsaved").count(), 0);
        assert_eq!(one(&index, "p.A").simple, "A");
    }

    fn member_ref(name: &str, line: u32, target: &str, start: u32) -> MemberRef {
        MemberRef {
            name: name.to_owned(),
            start_byte: 0,
            end_byte: 0,
            range: Range::new(
                Position::new(line, 4),
                Position::new(line, 4 + name.len() as u32),
            ),
            target_source: target.to_owned(),
            target_start: start,
        }
    }

    #[test]
    fn member_references_and_dependencies_round_trip_and_answer_by_target() {
        let tree = Tree::new("member-refs");
        let a = tree.file("src/A.java", "package p;\nclass A {}");
        let b = tree.file("src/B.java", "package p;\nclass B {}");
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        db.store_member_refs(
            &[
                (
                    a.clone(),
                    vec![member_ref("greet", 3, "/w/Greeter.java", 40)],
                    vec!["p.Greeter".to_owned()],
                ),
                (
                    b.clone(),
                    vec![
                        member_ref("greet", 7, "/w/Greeter.java", 40),
                        member_ref("other", 8, "/w/Greeter.java", 90),
                    ],
                    vec!["p.Greeter".to_owned(), "p.Other".to_owned()],
                ),
            ],
            REFS_WITH_JDK,
        )
        .expect("store");
        let index = db
            .index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        let mut lines: Vec<(String, u32)> = index
            .member_refs_to("/w/Greeter.java", 40)
            .map(|found| {
                (
                    index
                        .file(found.file)
                        .path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    found.range.start.line,
                )
            })
            .collect();
        lines.sort();
        assert_eq!(
            lines,
            vec![("A.java".to_owned(), 3), ("B.java".to_owned(), 7)]
        );

        let mut dependents = db.dependents(&["p.Other".to_owned()]).expect("dependents");
        dependents.sort();
        assert_eq!(dependents, vec![b.clone()]);
        let mut both = db
            .dependents(&["p.Greeter".to_owned()])
            .expect("dependents");
        both.sort();
        assert_eq!(both, vec![a, b]);
    }

    #[test]
    fn a_changed_or_deleted_file_loses_its_member_references() {
        let tree = Tree::new("member-refs-forgotten");
        let a = tree.file("src/A.java", "class A {}");
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        db.store_member_refs(
            &[(
                a.clone(),
                vec![member_ref("m", 1, "/w/X.java", 5)],
                vec!["X".to_owned()],
            )],
            REFS_WITH_JDK,
        )
        .expect("store");
        tree.file("src/A.java", "class A { int changed; }");
        let walked = db
            .index_sources_reporting(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        assert_eq!(walked.changed, vec![a.clone()]);
        assert_eq!(walked.index.member_refs_to("/w/X.java", 5).count(), 0);
        assert!(
            db.dependents(&["X".to_owned()])
                .expect("dependents")
                .is_empty()
        );
    }

    /// What lets a session killed halfway through the resolution pass be
    /// finished by the next one, whose walk finds every file unchanged: a file
    /// is owed until its references are stored against a world at least as
    /// whole as the pass's, and a settled world owes every file resolved
    /// against any other — another JDK, other JARs, the import turned on.
    #[test]
    fn a_file_is_owed_until_its_member_references_are_stored_against_the_same_world() {
        let tree = Tree::new("member-refs-pending");
        let a = tree.file("src/A.java", "class A {}");
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        let owed = |db: &CacheDb, state| db.owed_member_refs(state).expect("owed");
        assert_eq!(owed(&db, REFS_WITHOUT_WORLD), vec![a.clone()]);

        db.store_member_refs(&[(a.clone(), Vec::new(), Vec::new())], REFS_WITHOUT_WORLD)
            .expect("store");
        assert!(owed(&db, REFS_WITHOUT_WORLD).is_empty());
        assert_eq!(owed(&db, REFS_WITH_JDK), vec![a.clone()]);

        db.store_member_refs(&[(a.clone(), Vec::new(), Vec::new())], REFS_WITH_JDK)
            .expect("store");
        assert!(owed(&db, REFS_WITH_JDK).is_empty());
        assert_eq!(owed(&db, refs_settled(7)), vec![a.clone()]);

        db.store_member_refs(&[(a.clone(), Vec::new(), Vec::new())], refs_settled(7))
            .expect("store");
        assert!(owed(&db, refs_settled(7)).is_empty());
        assert!(
            owed(&db, REFS_WITH_JDK).is_empty(),
            "a settled world is never owed to a JDK-only pass"
        );
        assert_eq!(
            owed(&db, refs_settled(1 << 40)),
            vec![a.clone()],
            "another settled world owes it again"
        );

        tree.file("src/A.java", "class A { int changed; }");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        assert_eq!(
            owed(&db, REFS_WITHOUT_WORLD),
            vec![a],
            "a re-read file's references are owed again"
        );
    }

    #[test]
    fn a_settled_key_is_never_one_of_the_lesser_states() {
        for key in [0, 1, 2, 3, 4, u64::MAX, u64::MAX - 3] {
            assert!(refs_settled(key) > REFS_WITH_JDK, "{key}");
        }
    }

    /// A dependent about to be re-resolved is owed first, so a session
    /// killed partway through leaves the rest owed rather than stale.
    #[test]
    fn a_file_marked_pending_is_unresolved_again() {
        let tree = Tree::new("member-refs-marked");
        let a = tree.file("src/A.java", "class A {}");
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        db.store_member_refs(&[(a.clone(), Vec::new(), Vec::new())], REFS_WITH_JDK)
            .expect("store");
        db.mark_member_refs_pending(std::slice::from_ref(&a))
            .expect("marked");
        assert_eq!(
            db.owed_member_refs(REFS_WITHOUT_WORLD).expect("owed"),
            vec![a]
        );
    }

    /// What the hierarchies read: who writes a supertype by name, a member by
    /// its site, the members of a file with their declaration spans, and the
    /// recorded uses a file makes with their offsets.
    #[test]
    fn the_inverse_edges_are_looked_up_by_name_site_and_file() {
        let tree = Tree::new("inverse-edges");
        tree.file(
            "src/Shape.java",
            "package p;\npublic interface Shape { double area(); }\n",
        );
        let circle_text = "package p;\nclass Circle implements Shape, java.io.Serializable {\n    \
                           public double area() { return 0; }\n    void m() { area(); }\n}\n";
        let circle = tree.file("src/Circle.java", circle_text);
        let mut db = CacheDb::in_memory().expect("db");
        db.index_sources(&tree.found(), &CancellationToken::new())
            .expect("indexing");
        let call = circle_text.rfind("area").expect("the call") as u32;
        let mut used = member_ref("area", 3, &circle.to_string_lossy(), 0);
        used.start_byte = call;
        db.store_member_refs(&[(circle.clone(), vec![used], Vec::new())], REFS_WITH_JDK)
            .expect("store");
        let index = db.workspace_index().expect("readable");

        let writers: Vec<&str> = index
            .types_writing("Shape")
            .map(|ty| ty.fqn.as_str())
            .collect();
        assert_eq!(writers, vec!["p.Circle"]);
        assert_eq!(
            index.types_writing("Serializable").count(),
            1,
            "by the last segment"
        );

        let area = circle_text.find("area").expect("the declaration") as u32;
        let member = index.member_at(&circle, area).expect("Circle.area");
        assert_eq!(member.name, "area");
        assert!(
            member.decl_start < area && member.decl_end > area,
            "{member:?}"
        );
        let m = index
            .member_enclosing(&circle, call)
            .expect("the call is in m");
        assert_eq!(m.name, "m");
        assert!(
            index.member_enclosing(&circle, 0).is_none(),
            "the package line is in no member"
        );
        let uses: Vec<u32> = index
            .member_refs_between(&circle, m.decl_start, m.decl_end)
            .map(|found| found.start_byte)
            .collect();
        assert_eq!(uses, vec![call]);
        assert_eq!(
            index
                .member_refs_between(&circle, member.decl_start, member.decl_end)
                .count(),
            0,
            "area's own body calls nothing"
        );
    }

    #[test]
    fn an_empty_workspace_indexes_to_nothing() {
        let mut db = CacheDb::in_memory().expect("db");
        let index = db
            .index_sources(&[], &CancellationToken::new())
            .expect("indexing");
        assert_eq!(index.len(), 0);
        assert_eq!(index.files(), 0);
    }
}
