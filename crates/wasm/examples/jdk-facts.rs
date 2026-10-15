//! Writes the facts of a JDK's modules as the JSON `hx_load` takes.
//!
//!   cargo run --release -p javelin-wasm --example jdk-facts -- <JDK>/lib/src.zip out.json java.base
//!
//! The facts are what the JDK *you point at* declares: ship the ones for the JDK the students'
//! code is built with, or the editor will offer methods the build rejects.
use std::path::Path;

use javelin_classpath::{TypeFacts, read_type_facts};
use javelin_core::CancellationToken;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let modules = &args[3..];
    let facts: Vec<TypeFacts> =
        read_type_facts(Path::new(&args[1]), 16 << 20, &CancellationToken::never())
            .expect("read src.zip")
            .into_iter()
            .filter(|fact| {
                modules
                    .iter()
                    .any(|module| fact.source.starts_with(&format!("{module}/")))
            })
            .collect();
    std::fs::write(&args[2], serde_json::to_vec(&facts).expect("json")).expect("write");
    println!("{} types", facts.len());
}
