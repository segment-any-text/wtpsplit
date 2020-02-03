use fs_extra::dir::*;
use std::path::Path;

fn main() -> failure::Fallible<()> {
    let options = CopyOptions::new();

    if !Path::new("./data").exists() {
        copy(Path::new("../data"), Path::new("."), &options)?;
    }

    Ok(())
}