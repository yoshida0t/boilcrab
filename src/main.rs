use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, bail, ensure};
use clap::{Args, Parser, Subcommand};

use boilcrab::io::{energies, xyz};
use boilcrab::numdiff;

#[derive(Parser)]
#[command(version, about = "Quantum chemistry utilities")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write structures with each Cartesian coordinate shifted by ±step
    Disp(DispArgs),

    /// Central-difference gradient from energies of the `disp` structures
    Grad(GradArgs),
}

#[derive(Args)]
struct DispArgs {
    /// Reference structure (XYZ, Angstrom, single frame)
    input: PathBuf,

    /// Displacement step in bohr
    #[arg(long, default_value_t = 0.005)]
    step: f64,

    /// Output directory [default: <INPUT>_disp next to the input file]
    #[arg(long)]
    out: Option<PathBuf>,

    /// Write even if the output directory is not empty
    #[arg(long)]
    force: bool,
}

#[derive(Args)]
struct GradArgs {
    /// Directory written by `boilcrab disp`
    #[arg(long)]
    disp: PathBuf,

    /// Table of `name energy` lines, e.g. `0000p -76.0267591234` (Hartree)
    #[arg(long)]
    energies: PathBuf,
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Disp(args) => run_disp(args),
        Command::Grad(args) => run_grad(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // `{:#}` prints the whole cause chain on one line.
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_disp(args: DispArgs) -> anyhow::Result<()> {
    ensure!(
        args.step.is_finite() && args.step > 0.0,
        "--step must be positive and finite, got {}",
        args.step
    );

    let input = &args.input;
    let mut frames = xyz::read_frames_file(input)
        .with_context(|| format!("failed to read {}", input.display()))?;
    ensure!(
        frames.len() == 1,
        "{} contains {} frames; expected exactly one",
        input.display(),
        frames.len()
    );
    let mol = frames.remove(0).molecule;

    let out = args.out.unwrap_or_else(|| default_out_dir(input));
    if !args.force && is_non_empty_dir(&out)? {
        bail!(
            "{} is not empty (use --force to write into it anyway)",
            out.display()
        );
    }

    let displaced = numdiff::displace(&mol, args.step);
    let written = xyz::write_displaced(&out, &displaced)
        .with_context(|| format!("failed to write into {}", out.display()))?;

    println!(
        "wrote {} structures to {} (step = {} bohr)",
        written.len(),
        out.display(),
        args.step
    );
    Ok(())
}

fn run_grad(args: GradArgs) -> anyhow::Result<()> {
    let displaced = xyz::read_displaced(&args.disp).with_context(|| {
        format!(
            "failed to read displaced structures from {}",
            args.disp.display()
        )
    })?;
    let mut table = energies::read_file(&args.energies)
        .with_context(|| format!("failed to read {}", args.energies.display()))?;

    // read_displaced guarantees a complete, consistent set.
    let reference = &displaced[0];
    let step = reference.step_bohr;
    let atoms = reference.molecule.atoms();

    // Match energies to structures by name: pairs[i] = (E(+h), E(-h)).
    let mut pairs = vec![(f64::NAN, f64::NAN); 3 * atoms.len()];
    let mut missing = Vec::new();
    for d in &displaced {
        let name = d.name();
        match table.remove(&name) {
            Some(e) if d.sign > 0 => pairs[d.index()].0 = e,
            Some(e) => pairs[d.index()].1 = e,
            None => missing.push(name),
        }
    }
    ensure!(
        missing.is_empty(),
        "{} has no energy for {} structure(s): {}",
        args.energies.display(),
        missing.len(),
        abbreviate(&missing)
    );
    if !table.is_empty() {
        let extra: Vec<_> = table.into_keys().collect();
        eprintln!(
            "warning: ignoring {} name(s) not in {}: {}",
            extra.len(),
            args.disp.display(),
            abbreviate(&extra)
        );
    }

    let grad = numdiff::gradient(step, &pairs);

    println!(
        "# gradient (Hartree/bohr), central difference, step = {step} bohr, {} atoms",
        atoms.len()
    );
    for (sym, g) in atoms.iter().zip(&grad) {
        println!("{:<3}{:>18.10}{:>18.10}{:>18.10}", sym, g.x, g.y, g.z);
    }
    Ok(())
}

/// `a, b, c` — or the first 10 followed by `... (N more)`.
fn abbreviate(names: &[String]) -> String {
    const MAX: usize = 10;
    let shown = names[..names.len().min(MAX)].join(", ");
    if names.len() > MAX {
        format!("{shown}, ... ({} more)", names.len() - MAX)
    } else {
        shown
    }
}

/// `path/to/water.xyz` -> `path/to/water_disp`
fn default_out_dir(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map_or("out".into(), |s| s.to_string_lossy());
    input.with_file_name(format!("{stem}_disp"))
}

fn is_non_empty_dir(path: &Path) -> anyhow::Result<bool> {
    match fs::read_dir(path) {
        Ok(mut entries) => Ok(entries.next().is_some()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e).with_context(|| format!("cannot use {} as output", path.display())),
    }
}
