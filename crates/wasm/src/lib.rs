//! Member completion for a browser worker: a WASI *reactor* (no `_start`), loaded once with the
//! facts of a JDK and then asked about one buffer at a time.
//!
//! No server, no threads, no files: `javelin-server`'s loop needs threads and `wasm32-wasip1` has
//! none, so this calls the engine the way the completion handler does, with `TypeWorld` built from
//! facts shipped as JSON instead of read out of `src.zip`.
//!
//! ABI, all pointers into this module's linear memory:
//!
//! - `hx_alloc(len) -> ptr`, `hx_free(ptr, len)` — buffers the host fills and reads;
//! - `hx_load(ptr, len) -> i32` — the facts, a JSON array of `TypeFacts`; `0` is success;
//! - `hx_complete(ptr, len) -> ptr` — a JSON request `{"text", "line", "character"}`; the answer
//!   is `[u32 LE length][JSON]` at the returned pointer, to be freed with `hx_free(ptr, 4 + length)`.
//!
//! A panic aborts the instance on this target, so the host re-instantiates after a trap.
#![allow(unsafe_code)]

use std::cell::RefCell;
use std::sync::Arc;

use javelin_classpath::{TypeFacts, TypeWorld};
use javelin_document::{DocumentId, Position, TextDocument};
use javelin_index::WorkspaceIndex;
use javelin_parser::{DefinitionKind, FileSyntax, JavaParser, ScopeTree};
use javelin_semantic::Files;
use javelin_server::classpath::ClasspathExternals;
use serde::{Deserialize, Serialize};

struct Engine {
    externals: ClasspathExternals,
    parser: JavaParser,
}

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
}

#[derive(Deserialize)]
struct Request {
    text: String,
    line: u32,
    character: u32,
}

#[derive(Serialize)]
struct Item {
    label: String,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    documentation: Option<String>,
}

#[derive(Serialize)]
struct Answer {
    items: Vec<Item>,
    /// Whether the list is everything the receiver's type exposes under the typed prefix.
    complete: bool,
}

fn kind(kind: DefinitionKind) -> &'static str {
    match kind {
        DefinitionKind::LocalVariable
        | DefinitionKind::Parameter
        | DefinitionKind::PatternBinding => "variable",
        DefinitionKind::Field => "field",
        DefinitionKind::TypeParameter => "typeParameter",
        DefinitionKind::Type => "class",
        DefinitionKind::Method => "method",
        DefinitionKind::Constructor => "constructor",
    }
}

fn load(facts: &[u8]) -> Result<(), String> {
    let facts: Vec<TypeFacts> = serde_json::from_slice(facts).map_err(|err| err.to_string())?;
    let world = Arc::new(TypeWorld::new(facts));
    let externals = ClasspathExternals::new(world, Arc::new(WorkspaceIndex::new(Vec::new())));
    let parser = JavaParser::new().map_err(|err| err.to_string())?;

    ENGINE.with(|engine| *engine.borrow_mut() = Some(Engine { externals, parser }));

    Ok(())
}

fn complete(request: &[u8]) -> Result<Answer, String> {
    let request: Request = serde_json::from_slice(request).map_err(|err| err.to_string())?;

    ENGINE.with(|engine| {
        let mut engine = engine.borrow_mut();
        let engine = engine.as_mut().ok_or("facts are not loaded")?;

        let id = DocumentId::new("file:///Main.java");
        let document = TextDocument::new(id.clone(), 1, request.text);
        let offset = document.offset_at(Position::new(request.line, request.character));
        let tree = engine
            .parser
            .parse(&document)
            .map_err(|err| err.to_string())?;
        let syntax = FileSyntax::build(&document, &tree);
        let scopes = ScopeTree::build(&document, &tree);
        let files = Files::single(id, &syntax, &scopes);

        let found = javelin_semantic::member_completions_at(
            &document,
            &tree,
            &files,
            &engine.externals,
            offset,
        );

        Ok(match found {
            Some(found) => Answer {
                items: found
                    .candidates
                    .into_iter()
                    .map(|candidate| Item {
                        label: candidate.name,
                        kind: kind(candidate.kind),
                        detail: candidate.detail,
                        documentation: candidate.doc,
                    })
                    .collect(),
                complete: found.complete,
            },
            None => Answer {
                items: Vec::new(),
                complete: false,
            },
        })
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hx_alloc(len: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; len].into_boxed_slice()).cast()
}

/// # Safety
/// `ptr` and `len` come from `hx_alloc(len)` (or the pointer `hx_complete` returned, with its
/// full `4 + length`) and are freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hx_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
}

/// # Safety
/// `ptr` points at `len` initialised bytes from `hx_alloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hx_load(ptr: *const u8, len: usize) -> i32 {
    match load(unsafe { std::slice::from_raw_parts(ptr, len) }) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// # Safety
/// `ptr` points at `len` initialised bytes from `hx_alloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hx_complete(ptr: *const u8, len: usize) -> *mut u8 {
    let json = match complete(unsafe { std::slice::from_raw_parts(ptr, len) }) {
        Ok(answer) => serde_json::to_vec(&answer).unwrap_or_default(),
        Err(_) => b"{\"items\":[],\"complete\":false}".to_vec(),
    };

    let mut out = Vec::with_capacity(4 + json.len());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    Box::into_raw(out.into_boxed_slice()).cast()
}
