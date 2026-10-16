#!/bin/sh
# Builds the two files a browser needs for Java member completion:
#   javelin.wasm       — the `javelin-wasm` reactor (crates/wasm), WASI, no threads
#   jdk-base.facts     — the facts of java.base, from the JDK you point JDK_HOME at, gzipped JSON.
#                        No `.gz` in the name on purpose: dev servers that see it add
#                        `Content-Encoding: gzip`, the browser unpacks the body behind the
#                        page's back, and a worker that unpacks it again fails.
#
# Needs: Rust with the wasm32-wasip1 target, wasi-sdk (WASI_SDK=/path/to/wasi-sdk-34.0-...),
# a JDK with lib/src.zip. Ship the facts of the JDK the students' code is built with.
#
#   WASI_SDK=... JDK_HOME=... scripts/build-wasm.sh out/
set -eu
out=${1:?output directory}
: "${WASI_SDK:?path to wasi-sdk}" "${JDK_HOME:?path to a JDK with lib/src.zip}"

export CC_wasm32_wasip1="$WASI_SDK/bin/clang" AR_wasm32_wasip1="$WASI_SDK/bin/llvm-ar"
export CFLAGS_wasm32_wasip1="--sysroot=$WASI_SDK/share/wasi-sysroot"

mkdir -p "$out"
cargo build --release -p javelin-wasm --target wasm32-wasip1
"$WASI_SDK/bin/llvm-strip" -o "$out/javelin.wasm" target/wasm32-wasip1/release/javelin_wasm.wasm

cargo run --release -p javelin-wasm --example jdk-facts -- "$JDK_HOME/lib/src.zip" "$out/jdk-base.json" java.base
gzip -9 -n -f "$out/jdk-base.json"
mv "$out/jdk-base.json.gz" "$out/jdk-base.facts"
sha256sum "$out"/javelin.wasm "$out"/jdk-base.facts 2>/dev/null || shasum -a 256 "$out"/javelin.wasm "$out"/jdk-base.facts
