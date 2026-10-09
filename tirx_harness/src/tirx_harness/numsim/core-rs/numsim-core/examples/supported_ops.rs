//! Regenerate `numsim-oplib/SUPPORTED_OPS.md` (`cargo run -p numsim-core
//! --example supported_ops`). See `numsim_core::oplib::supported_ops`.

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../numsim-oplib/SUPPORTED_OPS.md");
    let text = numsim_core::oplib::supported_ops::render();
    std::fs::write(&path, &text).expect("write SUPPORTED_OPS.md");
    let gaps = text.split("## Gaps vs legacy").nth(1).map_or(0, |g| g.lines().filter(|l| l.starts_with("| `")).count());
    println!("wrote {} ({} gap rows)", path.display(), gaps);
}
