//! Name / energy table: one `name energy` pair per line.
//!
//! ```text
//! # name   energy (Hartree)
//! 0000p   -76.026759123456
//! 0000m   -76.026761234567
//! ```
//!
//! Blank lines and lines starting with `#` are skipped. Columns after the
//! second are ignored.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnergiesError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("line {line}: expected `name energy`, got {found:?}")]
    InvalidLine { line: usize, found: String },

    #[error("line {line}: invalid energy {found:?}")]
    InvalidEnergy { line: usize, found: String },

    #[error("line {line}: duplicate name {name:?} (first on line {first})")]
    Duplicate {
        line: usize,
        name: String,
        first: usize,
    },
}

/// Reads the table into `name -> energy`.
pub fn read<R: BufRead>(r: R) -> Result<BTreeMap<String, f64>, EnergiesError> {
    // name -> (energy, line it was found on)
    let mut table: BTreeMap<String, (f64, usize)> = BTreeMap::new();

    for (i, text) in r.lines().enumerate() {
        let text = text?;
        let line = i + 1;
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut cols = trimmed.split_whitespace();
        let (Some(name), Some(value)) = (cols.next(), cols.next()) else {
            return Err(EnergiesError::InvalidLine {
                line,
                found: text.clone(),
            });
        };
        let energy = value
            .parse::<f64>()
            .ok()
            .filter(|e| e.is_finite())
            .ok_or_else(|| EnergiesError::InvalidEnergy {
                line,
                found: value.to_owned(),
            })?;

        if let Some(&(_, first)) = table.get(name) {
            return Err(EnergiesError::Duplicate {
                line,
                name: name.to_owned(),
                first,
            });
        }
        table.insert(name.to_owned(), (energy, line));
    }

    Ok(table.into_iter().map(|(k, (e, _))| (k, e)).collect())
}

pub fn read_file(path: impl AsRef<Path>) -> Result<BTreeMap<String, f64>, EnergiesError> {
    read(BufReader::new(File::open(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_table_skipping_comments_and_blank_lines() {
        let input = "# name energy\n\n0000p  -76.5\n0000m -76.25 extra\n  # indented comment\n";
        let t = read(input.as_bytes()).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t["0000p"], -76.5);
        assert_eq!(t["0000m"], -76.25);
    }

    #[test]
    fn missing_column_reports_line() {
        let err = read("0000p -1.0\n0000m\n".as_bytes()).unwrap_err();
        assert!(matches!(err, EnergiesError::InvalidLine { line: 2, .. }));
    }

    #[test]
    fn bad_energy_reports_line() {
        for bad in ["abc", "nan", "inf"] {
            let err = read(format!("0000p {bad}\n").as_bytes()).unwrap_err();
            assert!(
                matches!(err, EnergiesError::InvalidEnergy { line: 1, .. }),
                "{bad}"
            );
        }
    }

    #[test]
    fn duplicate_name_is_error() {
        let err = read("0000p -1\n0000m -2\n0000p -3\n".as_bytes()).unwrap_err();
        assert!(matches!(
            err,
            EnergiesError::Duplicate {
                line: 3,
                first: 1,
                ..
            }
        ));
    }
}
