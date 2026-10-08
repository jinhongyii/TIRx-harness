//! `validate-program FILE.json...` — deserialize each lowered `Module` with
//! `numsim_core::program::Module::from_json`, run `Program::validate()` on
//! every kernel, and check the postcard round trip. One line per file:
//! `OK <path> <kernels>` or `ERR <path> <message>`. Exit status 1 if any
//! file fails. `-` reads newline-separated paths from stdin.

use numsim_core::program::Module;
use std::io::BufRead;

fn check(path: &str) -> Result<usize, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let module = Module::from_json(&text).map_err(|e| e.to_string())?;
    for kernel in &module.kernels {
        kernel.validate().map_err(|e| format!("{}: {e}", kernel.name))?;
    }
    let bytes = module.to_bytes();
    let back = Module::from_bytes(&bytes).map_err(|e| format!("postcard: {e}"))?;
    if back != module {
        return Err("postcard round trip changed the module".into());
    }
    Ok(module.kernels.len())
}

fn main() {
    let mut paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.iter().any(|p| p == "-") {
        paths.retain(|p| p != "-");
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if !line.trim().is_empty() {
                paths.push(line.trim().to_string());
            }
        }
    }
    let mut failed = false;
    for path in &paths {
        match check(path) {
            Ok(n) => println!("OK {path} {n}"),
            Err(message) => {
                failed = true;
                println!("ERR {path} {message}");
            }
        }
    }
    std::process::exit(if failed { 1 } else { 0 });
}
