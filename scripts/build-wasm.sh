#!/bin/sh
# Builds the two files a browser needs for Java member completion:
#   javelin.wasm       — the `javelin-wasm` reactor (crates/wasm), WASI, no threads
#   jdk-base.json.gz   — the facts of java.base, from the JDK you point JDK_HOME at
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
sha256sum "$out"/javelin.wasm "$out"/jdk-base.json.gz 2>/dev/null || shasum -a 256 "$out"/javelin.wasm "$out"/jdk-base.json.gz
