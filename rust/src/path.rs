//! # path
//!
//! File path traversal utilities.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::path::PathBuf;

/// Walk a directory and filter invalid paths.
pub fn walk_dir(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let files: Vec<_> = fs::read_dir(dir)?
        .filter_map(|x| x.ok())
        .map(|x| x.path())
        .collect();
    Ok(files)
}

/// Extract the file stem from a path.
pub fn extract_file_stem(dir: &Path) -> Result<String> {
    let file_stem = dir
        .file_stem()
        .context("Cannot parse file stem.")?
        .to_str()
        .context("Cannot convert file stem to string.")?
        .to_string();
    Ok(file_stem)
}
