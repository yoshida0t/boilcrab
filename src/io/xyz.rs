use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use nalgebra::Vector3;
use thiserror::Error;

use crate::core::{Molecule, MoleculeError};
use crate::numdiff::Displaced;

#[derive(Debug, Error)]
pub enum XyzError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("no frame found in input")]
    Empty,

    #[error("line {line}: invalid atom count {found:?}")]
    InvalidCount { line: usize, found: String },

    #[error("line {line}: unexpected end of file (expected {expected} atoms, read {read})")]
    UnexpectedEof {
        line: usize,
        expected: usize,
        read: usize,
    },

    #[error("line {line}: expected `symbol x y z`, got {found:?}")]
    InvalidAtomLine { line: usize, found: String },

    #[error("line {line}: invalid coordinate {found:?}")]
    InvalidCoordinate { line: usize, found: String },

    #[error(transparent)]
    Molecule(#[from] MoleculeError),

    #[error("in {}", path.display())]
    InFile {
        path: PathBuf,
        #[source]
        source: Box<XyzError>,
    },

    #[error("no displaced structures (NNNNp.xyz / NNNNm.xyz) in {}", dir.display())]
    NoDisplaced { dir: PathBuf },

    #[error("{}: no `step_bohr=` in the comment line", path.display())]
    MissingStep { path: PathBuf },

    #[error("{}: step_bohr={found}, but {} has {expected}", path.display(), first.display())]
    StepMismatch {
        path: PathBuf,
        first: PathBuf,
        expected: f64,
        found: f64,
    },

    #[error("{}: atoms differ from {}", path.display(), first.display())]
    AtomsMismatch { path: PathBuf, first: PathBuf },

    #[error("{}: missing {}", dir.display(), missing.join(", "))]
    Incomplete { dir: PathBuf, missing: Vec<String> },
}


#[derive(Debug, Clone, PartialEq)]
pub struct XyzFrame {
    pub molecule: Molecule,
    pub comment: String,
}

pub struct Reader<R> {
    inner: R,
    buf: String,
    line: usize,
    done: bool,
}

impl<R: BufRead> Reader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            buf: String::new(),
            line: 0,
            done: false,
        }
    }

    fn next_line(&mut self) -> io::Result<bool> {
        self.buf.clear();
        if self.inner.read_line(&mut self.buf)? == 0 {
            return Ok(false);
        }
        self.line += 1;
        let trimmed_len = self.buf.trim_end_matches(['\n', '\r']).len();
        self.buf.truncate(trimmed_len);
        Ok(true)
    }

    fn read_frame(&mut self) -> Result<Option<XyzFrame>, XyzError> {
        // Atom count; blank lines between frames / at EOF are tolerated.
        loop {
            if !self.next_line()? {
                return Ok(None);
            }
            if !self.buf.trim().is_empty() {
                break;
            }
        }
        let n: usize = self
            .buf
            .trim()
            .parse()
            .map_err(|_| XyzError::InvalidCount {
                line: self.line,
                found: self.buf.clone(),
            })?;

        // Comment line.
        if !self.next_line()? {
            return Err(self.eof(n, 0));
        }
        let comment = self.buf.clone();

        let mut atoms = Vec::with_capacity(n);
        let mut coord = Vec::with_capacity(n);
        for i in 0..n {
            if !self.next_line()? {
                return Err(self.eof(n, i));
            }
            let (sym, r) = parse_atom_line(&self.buf, self.line)?;
            atoms.push(sym);
            coord.push(r);
        }

        Ok(Some(XyzFrame {
            molecule: Molecule::new(atoms, coord)?,
            comment,
        }))
    }

    fn eof(&self, expected: usize, read: usize) -> XyzError {
        XyzError::UnexpectedEof {
            line: self.line + 1,
            expected,
            read,
        }
    }
}



impl<R: BufRead> Iterator for Reader<R> {
    type Item = Result<XyzFrame, XyzError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let res = self.read_frame().transpose();
        if !matches!(res, Some(Ok(_))) {
            self.done = true;
        }
        res
    }
}

fn parse_atom_line(s: &str, line: usize) -> Result<(String, Vector3<f64>), XyzError> {
    let mut it = s.split_whitespace();
    let (Some(sym), Some(x), Some(y), Some(z)) = (it.next(), it.next(), it.next(), it.next())
    else {
        return Err(XyzError::InvalidAtomLine {
            line,
            found: s.to_owned(),
        });
    };
    let f = |t: &str| {
        t.parse::<f64>().map_err(|_| XyzError::InvalidCoordinate {
            line,
            found: t.to_owned(),
        })
    };
    Ok((sym.to_owned(), Vector3::new(f(x)?, f(y)?, f(z)?)))
}

pub fn read<R: BufRead>(r: R) -> Result<Molecule, XyzError> {
    Reader::new(r)
        .next()
        .unwrap_or(Err(XyzError::Empty))
        .map(|f| f.molecule)
}

pub fn read_file(path: impl AsRef<Path>) -> Result<Molecule, XyzError> {
    read(BufReader::new(File::open(path)?))
}

pub fn read_frames_file(path: impl AsRef<Path>) -> Result<Vec<XyzFrame>, XyzError> {
    Reader::new(BufReader::new(File::open(path)?)).collect()
}




pub fn write<W: Write>(mut w: W, mol: &Molecule, comment: &str) -> io::Result<()> {
    writeln!(w, "{}", mol.len())?;
    writeln!(w, "{}", comment.replace(['\n', '\r'], " "))?;
    for (sym, r) in mol.atoms().iter().zip(mol.coord()) {
        writeln!(w, "{:<3}{:>16.10}{:>16.10}{:>16.10}", sym, r.x, r.y, r.z)?;
    }
    Ok(())
}

pub fn write_file(path: impl AsRef<Path>, mol: &Molecule, comment: &str) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write(&mut w, mol, comment)?;
    // Flush explicitly: BufWriter's Drop swallows write errors.
    w.flush()
}

/// Writes displaced structures into `dir`, one file per structure.
///
/// File names are `{index:04}p.xyz` for `+step` and `{index:04}m.xyz` for
/// `-step`, where `index = 3 * atom + axis` (atom 0: x = 0000, y = 0001,
/// z = 0002; atom 1: x = 0003, ...). The comment line holds the label and
/// the step, e.g. `a0+x step_bohr=0.005`.
///
/// Creates `dir` if needed. Existing files with the same name are overwritten.
/// Returns the written paths in input order.
pub fn write_displaced(
    dir: impl AsRef<Path>,
    displaced: &[Displaced],
) -> io::Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    displaced
        .iter()
        .map(|d| {
            let path = dir.join(format!("{}.xyz", d.name()));
            let comment = format!("{} step_bohr={}", d.label(), d.step_bohr);
            write_file(&path, &d.molecule, &comment)?;
            Ok(path)
        })
        .collect()
}

/// Reads structures written by [`write_displaced`] back from `dir`.
///
/// Only files named like `0000p.xyz` / `0000m.xyz` are read; anything else in
/// the directory (inputs or outputs of other programs) is ignored. Atom, axis
/// and sign come from the file name, the step from `step_bohr=` in the
/// comment line.
///
/// All files must have the same atoms and step, and every coordinate of every
/// atom must have both `p` and `m`. The result is in the same order as
/// [`crate::numdiff::displace`].
pub fn read_displaced(dir: impl AsRef<Path>) -> Result<Vec<Displaced>, XyzError> {
    let dir = dir.as_ref();

    let mut found = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if let Some((index, sign)) = parse_displaced_name(&path) {
            found.push((index, sign, path));
        }
    }
    found.sort_by_key(|(index, sign, _)| (*index, -sign)); // p before m

    let mut out: Vec<Displaced> = Vec::with_capacity(found.len());
    let mut first_path: Option<PathBuf> = None;
    for (index, sign, path) in found {
        let in_file = |e: XyzError| XyzError::InFile {
            path: path.clone(),
            source: Box::new(e),
        };
        let frame = File::open(&path)
            .map_err(XyzError::from)
            .and_then(|f| {
                Reader::new(BufReader::new(f))
                    .next()
                    .unwrap_or(Err(XyzError::Empty))
            })
            .map_err(in_file)?;

        let step_bohr = parse_step(&frame.comment)
            .ok_or_else(|| XyzError::MissingStep { path: path.clone() })?;

        if let (Some(first), Some(first_path)) = (out.first(), &first_path) {
            if frame.molecule.atoms() != first.molecule.atoms() {
                return Err(XyzError::AtomsMismatch {
                    path,
                    first: first_path.clone(),
                });
            }
            if step_bohr != first.step_bohr {
                return Err(XyzError::StepMismatch {
                    path,
                    first: first_path.clone(),
                    expected: first.step_bohr,
                    found: step_bohr,
                });
            }
        } else {
            first_path = Some(path.clone());
        }

        out.push(Displaced {
            atom: index / 3,
            axis: index % 3,
            sign,
            step_bohr,
            molecule: frame.molecule,
        });
    }

    let Some(first) = out.first() else {
        return Err(XyzError::NoDisplaced {
            dir: dir.to_owned(),
        });
    };

    // Every coordinate needs both signs, and nothing beyond 3N.
    let ncoord = 3 * first.molecule.len();
    let have: std::collections::BTreeSet<(usize, i8)> =
        out.iter().map(|d| (d.index(), d.sign)).collect();
    let mut missing = Vec::new();
    for index in 0..ncoord {
        for (sign, pm) in [(1, 'p'), (-1, 'm')] {
            if !have.contains(&(index, sign)) {
                missing.push(format!("{index:04}{pm}.xyz"));
            }
        }
    }
    if !missing.is_empty() || out.len() != 2 * ncoord {
        return Err(XyzError::Incomplete {
            dir: dir.to_owned(),
            missing,
        });
    }
    Ok(out)
}

/// `0012p.xyz` -> `Some((12, 1))`, `0003m.xyz` -> `Some((3, -1))`.
fn parse_displaced_name(path: &Path) -> Option<(usize, i8)> {
    let stem = path.file_name()?.to_str()?.strip_suffix(".xyz")?;
    let (digits, sign) = if let Some(d) = stem.strip_suffix('p') {
        (d, 1)
    } else {
        (stem.strip_suffix('m')?, -1)
    };
    if digits.len() < 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((digits.parse().ok()?, sign))
}

/// Finds `step_bohr=<value>` among the whitespace-separated comment tokens.
fn parse_step(comment: &str) -> Option<f64> {
    comment
        .split_whitespace()
        .find_map(|t| t.strip_prefix("step_bohr="))
        .and_then(|v| v.parse().ok())
}









#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    const WATER: &str = "3
water molecule
O   0.000  0.000  0.000
H   0.757  0.586  0.000
H  -0.757  0.586  0.000
";

    #[test]
    fn reads_single_frame() {
        let m = read(WATER.as_bytes()).unwrap();
        assert_eq!(m.atoms(), ["O", "H", "H"]);
        assert_eq!(m.coord()[1], Vector3::new(0.757, 0.586, 0.0));
    }

    #[test]
    fn keeps_comment_and_reads_multiple_frames() {
        let input = format!("{WATER}\n{WATER}");
        let frames: Vec<_> = Reader::new(input.as_bytes())
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].comment, "water molecule");
    }

    #[test]
    fn handles_crlf_and_extra_columns() {
        let input = "1\r\nextxyz\r\nC 1.0 2.0 3.0 0.1 0.2 0.3\r\n";
        let m = read(input.as_bytes()).unwrap();
        assert_eq!(m.coord()[0], Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn empty_input_is_error() {
        assert!(matches!(read("\n\n".as_bytes()), Err(XyzError::Empty)));
    }

    #[test]
    fn truncated_file_reports_eof() {
        let input = "3\nc\nO 0 0 0\n";
        let err = read(input.as_bytes()).unwrap_err();
        assert!(matches!(
            err,
            XyzError::UnexpectedEof {
                line: 4,
                expected: 3,
                read: 1
            }
        ));
    }

    #[test]
    fn bad_coordinate_reports_line() {
        let input = "1\nc\nO 0 abc 0\n";
        let err = read(input.as_bytes()).unwrap_err();
        assert!(matches!(err, XyzError::InvalidCoordinate { line: 3, .. }));
    }

    #[test]
    fn bad_count_reports_line() {
        let err = read("three\nc\n".as_bytes()).unwrap_err();
        assert!(matches!(err, XyzError::InvalidCount { line: 1, .. }));
    }

    #[test]
    fn reader_stops_after_error() {
        let input = format!("x\n{WATER}");
        let mut r = Reader::new(input.as_bytes());
        assert!(r.next().unwrap().is_err());
        assert!(r.next().is_none());
    }

    #[test]
    fn comment_newlines_are_sanitized() {
        let m = read(WATER.as_bytes()).unwrap();
        let mut out = Vec::new();
        write(&mut out, &m, "a\nb").unwrap();
        let frame = Reader::new(out.as_slice()).next().unwrap().unwrap();
        assert_eq!(frame.comment, "a b");
    }

    #[test]
    fn file_roundtrip() {
        let m = read(WATER.as_bytes()).unwrap();
        let path = std::env::temp_dir().join(format!("boilcrab_xyz_{}.xyz", std::process::id()));
        write_file(&path, &m, "rt").unwrap();
        let back = read_file(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back, m);
    }

    #[test]
    fn writes_displaced_with_index_and_sign_names() {
        let m = read(WATER.as_bytes()).unwrap();
        let disp = crate::numdiff::displace(&m, 0.005);
        let dir = std::env::temp_dir().join(format!("boilcrab_disp_{}", std::process::id()));

        let paths = write_displaced(&dir, &disp).unwrap();
        let names: Vec<_> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(names.len(), 18);
        assert_eq!(names[..4], ["0000p.xyz", "0000m.xyz", "0001p.xyz", "0001m.xyz"]);
        assert_eq!(names[17], "0008m.xyz");

        // 0001m = atom 0, y, minus
        let frame = Reader::new(BufReader::new(File::open(dir.join("0001m.xyz")).unwrap()))
            .next()
            .unwrap()
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(frame.comment, "a0-y step_bohr=0.005");
        for (a, b) in frame.molecule.coord().iter().zip(disp[3].molecule.coord()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    fn fresh_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("boilcrab_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn read_displaced_roundtrip_ignores_other_files() {
        let m = read(WATER.as_bytes()).unwrap();
        let disp = crate::numdiff::displace(&m, 0.005);
        let dir = fresh_dir("read_disp");
        write_displaced(&dir, &disp).unwrap();
        fs::write(dir.join("0000p.out"), "not xyz").unwrap();
        fs::write(dir.join("notes.xyz"), "junk").unwrap();

        let back = read_displaced(&dir).unwrap();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(back.len(), disp.len());
        for (a, b) in back.iter().zip(&disp) {
            assert_eq!(
                (a.atom, a.axis, a.sign, a.step_bohr),
                (b.atom, b.axis, b.sign, b.step_bohr)
            );
            assert_eq!(a.molecule.atoms(), b.molecule.atoms());
            for (ra, rb) in a.molecule.coord().iter().zip(b.molecule.coord()) {
                assert_relative_eq!(ra, rb, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn read_displaced_reports_missing_files() {
        let m = read(WATER.as_bytes()).unwrap();
        let dir = fresh_dir("read_disp_missing");
        write_displaced(&dir, &crate::numdiff::displace(&m, 0.005)).unwrap();
        fs::remove_file(dir.join("0003m.xyz")).unwrap();

        let err = read_displaced(&dir).unwrap_err();
        fs::remove_dir_all(&dir).ok();
        match err {
            XyzError::Incomplete { missing, .. } => assert_eq!(missing, ["0003m.xyz"]),
            e => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn read_displaced_rejects_mixed_steps_and_missing_step() {
        let m = read(WATER.as_bytes()).unwrap();
        let dir = fresh_dir("read_disp_step");
        write_displaced(&dir, &crate::numdiff::displace(&m, 0.005)).unwrap();
        write_file(dir.join("0004p.xyz"), &m, "a1+y step_bohr=0.01").unwrap();
        assert!(matches!(
            read_displaced(&dir).unwrap_err(),
            XyzError::StepMismatch { found: 0.01, .. }
        ));

        write_file(dir.join("0004p.xyz"), &m, "a1+y").unwrap();
        assert!(matches!(
            read_displaced(&dir).unwrap_err(),
            XyzError::MissingStep { .. }
        ));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_displaced_rejects_empty_dir() {
        let dir = fresh_dir("read_disp_empty");
        fs::create_dir_all(&dir).unwrap();
        let err = read_displaced(&dir).unwrap_err();
        fs::remove_dir_all(&dir).ok();
        assert!(matches!(err, XyzError::NoDisplaced { .. }));
    }

    proptest! {
        #[test]
        fn roundtrip(
            atoms in prop::collection::vec(
                ("[A-Z][a-z]?", -1e3f64..1e3, -1e3f64..1e3, -1e3f64..1e3),
                0..20,
            ),
            comment in "[^\r\n]*",
        ) {
            let (syms, coord): (Vec<_>, Vec<_>) = atoms
                .into_iter()
                .map(|(s, x, y, z)| (s, Vector3::new(x, y, z)))
                .unzip();
            let m = Molecule::new(syms, coord).unwrap();

            let mut out = Vec::new();
            write(&mut out, &m, &comment).unwrap();
            let frame = Reader::new(out.as_slice()).next().unwrap().unwrap();

            prop_assert_eq!(&frame.comment, &comment);
            prop_assert_eq!(frame.molecule.atoms(), m.atoms());
            for (a, b) in frame.molecule.coord().iter().zip(m.coord()) {
                assert_relative_eq!(a, b, epsilon = 1e-9);
            }
        }
    }
}
