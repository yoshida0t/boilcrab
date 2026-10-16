//! Runs the `boilcrab` binary end to end.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use boilcrab::Molecule;
use boilcrab::io::xyz;

const WATER: &str = "3
water
O   0.000  0.000  0.117
H   0.000  0.757 -0.469
H   0.000 -0.757 -0.469
";

/// Fresh empty directory per test.
fn workdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("boilcrab_cli_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn boilcrab(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boilcrab"))
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}

fn file_count(dir: &Path) -> usize {
    fs::read_dir(dir).unwrap().count()
}

fn comment_line(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .nth(1)
        .unwrap()
        .to_owned()
}

#[test]
fn disp_writes_6n_files_to_default_dir() {
    let dir = workdir("default");
    fs::write(dir.join("water.xyz"), WATER).unwrap();

    let out = boilcrab(&["disp", "water.xyz"], &dir);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("wrote 18 structures"));

    let disp = dir.join("water_disp");
    assert_eq!(file_count(&disp), 18);
    assert_eq!(
        comment_line(&disp.join("0000p.xyz")),
        "a0+x step_bohr=0.005"
    );
    assert_eq!(
        comment_line(&disp.join("0008m.xyz")),
        "a2-z step_bohr=0.005"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn disp_respects_step_and_out() {
    let dir = workdir("options");
    fs::write(dir.join("water.xyz"), WATER).unwrap();

    let out = boilcrab(
        &["disp", "water.xyz", "--step", "0.01", "--out", "run1"],
        &dir,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        comment_line(&dir.join("run1/0001m.xyz")),
        "a0-y step_bohr=0.01"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn disp_rejects_bad_step() {
    let dir = workdir("badstep");
    fs::write(dir.join("water.xyz"), WATER).unwrap();

    for step in ["0", "-0.01", "nan"] {
        let out = boilcrab(&["disp", "water.xyz", &format!("--step={step}")], &dir);
        assert!(!out.status.success(), "step {step} should fail");
        assert!(String::from_utf8_lossy(&out.stderr).contains("--step"));
    }
    assert!(!dir.join("water_disp").exists());
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn disp_refuses_non_empty_out_without_force() {
    let dir = workdir("force");
    fs::write(dir.join("water.xyz"), WATER).unwrap();

    assert!(boilcrab(&["disp", "water.xyz"], &dir).status.success());

    let again = boilcrab(&["disp", "water.xyz", "--step", "0.01"], &dir);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("--force"));
    assert_eq!(
        comment_line(&dir.join("water_disp/0000p.xyz")),
        "a0+x step_bohr=0.005"
    );

    let forced = boilcrab(&["disp", "water.xyz", "--step", "0.01", "--force"], &dir);
    assert!(forced.status.success());
    assert_eq!(
        comment_line(&dir.join("water_disp/0000p.xyz")),
        "a0+x step_bohr=0.01"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn disp_rejects_multi_frame_input() {
    let dir = workdir("multiframe");
    fs::write(dir.join("traj.xyz"), format!("{WATER}{WATER}")).unwrap();

    let out = boilcrab(&["disp", "traj.xyz"], &dir);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("2 frames"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn disp_reports_missing_input() {
    let dir = workdir("missing");
    let out = boilcrab(&["disp", "nope.xyz"], &dir);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("nope.xyz"));
    fs::remove_dir_all(&dir).ok();
}

// --- grad ------------------------------------------------------------------

/// E = sum_i (c_i x_i + a_i x_i^2), x in bohr. Central differences are exact.
fn coeffs(i: usize) -> (f64, f64) {
    (0.1 * i as f64 - 0.3, 0.5 + 0.2 * i as f64)
}

fn flat_bohr(m: &Molecule) -> Vec<f64> {
    m.coord_bohr()
        .iter()
        .flat_map(|r| [r.x, r.y, r.z])
        .collect()
}

fn model_energy(m: &Molecule) -> f64 {
    flat_bohr(m)
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let (c, a) = coeffs(i);
            c * x + a * x * x
        })
        .sum()
}

fn model_gradient(m: &Molecule) -> Vec<f64> {
    flat_bohr(m)
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let (c, a) = coeffs(i);
            c + 2.0 * a * x
        })
        .collect()
}

/// Runs `disp` in a fresh directory and writes `energies.txt` from the model,
/// skipping names for which `keep` returns false.
fn setup_grad(name: &str, keep: impl Fn(&str) -> bool) -> PathBuf {
    let dir = workdir(name);
    fs::write(dir.join("water.xyz"), WATER).unwrap();
    assert!(boilcrab(&["disp", "water.xyz"], &dir).status.success());

    let mut table = String::from("# name energy\n");
    for d in xyz::read_displaced(dir.join("water_disp")).unwrap() {
        if keep(&d.name()) {
            table += &format!("{} {}\n", d.name(), model_energy(&d.molecule));
        }
    }
    fs::write(dir.join("energies.txt"), table).unwrap();
    dir
}

const GRAD_ARGS: [&str; 5] = ["grad", "--disp", "water_disp", "--energies", "energies.txt"];

#[test]
fn grad_matches_analytic_gradient() {
    let dir = setup_grad("grad_ok", |_| true);
    let out = boilcrab(&GRAD_ARGS, &dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let rows: Vec<Vec<&str>> = stdout
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split_whitespace().collect())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter().map(|r| r[0]).collect::<Vec<_>>(),
        ["O", "H", "H"]
    );

    let reference = xyz::read_file(dir.join("water.xyz")).unwrap();
    let expected = model_gradient(&reference);
    let got: Vec<f64> = rows
        .iter()
        .flat_map(|r| r[1..4].iter().map(|v| v.parse::<f64>().unwrap()))
        .collect();
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-6, "coord {i}: got {g}, expected {e}");
    }
    assert!(stdout.contains("step = 0.005 bohr"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn grad_reports_missing_energies() {
    let dir = setup_grad("grad_missing", |n| n != "0003m" && n != "0007p");
    let out = boilcrab(&GRAD_ARGS, &dir);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("2 structure(s): 0003m, 0007p"), "{stderr}");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn grad_warns_about_extra_names() {
    let dir = setup_grad("grad_extra", |_| true);
    let mut table = fs::read_to_string(dir.join("energies.txt")).unwrap();
    table += "ref -76.0\n";
    fs::write(dir.join("energies.txt"), table).unwrap();

    let out = boilcrab(&GRAD_ARGS, &dir);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("warning: ignoring 1 name(s)"), "{stderr}");
    assert!(stderr.contains("ref"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn grad_rejects_incomplete_disp_dir() {
    let dir = setup_grad("grad_incomplete", |_| true);
    fs::remove_file(dir.join("water_disp/0002p.xyz")).unwrap();

    let out = boilcrab(&GRAD_ARGS, &dir);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("missing 0002p.xyz"), "{stderr}");
    fs::remove_dir_all(&dir).ok();
}
