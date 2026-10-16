//! Displaced structures for finite-difference gradients / Hessians.

use nalgebra::Vector3;

use crate::core::Molecule;
use crate::core::units::BOHR_TO_ANG;

/// A copy of the reference structure with one Cartesian coordinate
/// shifted by `sign * step`.
#[derive(Debug, Clone, PartialEq)]
pub struct Displaced {
    pub atom: usize,
    pub axis: usize, // 0 = x, 1 = y, 2 = z
    pub sign: i8,    // +1 or -1
    pub step_bohr: f64,
    pub molecule: Molecule,
}

impl Displaced {
    /// Cartesian coordinate index `3 * atom + axis`
    /// (x0 = 0, y0 = 1, z0 = 2, x1 = 3, ...).
    pub fn index(&self) -> usize {
        3 * self.atom + self.axis
    }

    /// File stem on disk: `{index:04}p` for +step, `{index:04}m` for -step
    /// (e.g. `0000p`, `0001m`).
    pub fn name(&self) -> String {
        let pm = if self.sign > 0 { 'p' } else { 'm' };
        format!("{:04}{pm}", self.index())
    }

    /// e.g. `a0+x`, `a2-z` (atom index is 0-based).
    pub fn label(&self) -> String {
        let sign = if self.sign > 0 { '+' } else { '-' };
        let axis = ['x', 'y', 'z'][self.axis];
        format!("a{}{}{}", self.atom, sign, axis)
    }
}

/// Shifts every Cartesian coordinate by `+step_bohr` and `-step_bohr`.
///
/// Returns `6N` structures in the order `a0+x, a0-x, a0+y, a0-y, ..., a(N-1)-z`.
///
/// # Panics
/// If `step_bohr` is not positive and finite.
pub fn displace(mol: &Molecule, step_bohr: f64) -> Vec<Displaced> {
    assert_step(step_bohr);
    let h = step_bohr * BOHR_TO_ANG; // Molecule stores Angstrom

    let mut out = Vec::with_capacity(6 * mol.len());
    for atom in 0..mol.len() {
        for axis in 0..3 {
            for sign in [1i8, -1] {
                let mut coord = mol.coord().to_vec();
                coord[atom][axis] += f64::from(sign) * h;
                let molecule = Molecule::new(mol.atoms().to_vec(), coord)
                    .expect("atom and coord counts are unchanged");
                out.push(Displaced {
                    atom,
                    axis,
                    sign,
                    step_bohr,
                    molecule,
                });
            }
        }
    }
    out
}

/// Central-difference gradient from energies.
///
/// `pairs[i] = (E(+h), E(-h))` for Cartesian coordinate `i = 3 * atom + axis`,
/// so `pairs.len()` is `3N`. Returns one vector per atom, in energy units per
/// bohr (Hartree/bohr when energies are in Hartree).
///
/// # Panics
/// If `pairs.len()` is not a multiple of 3, or `step_bohr` is not positive
/// and finite.
pub fn gradient(step_bohr: f64, pairs: &[(f64, f64)]) -> Vec<Vector3<f64>> {
    assert_step(step_bohr);
    assert!(
        pairs.len().is_multiple_of(3),
        "need 3 coordinates per atom, got {} pairs",
        pairs.len()
    );
    pairs
        .chunks_exact(3)
        .map(|c| Vector3::from_fn(|k, _| (c[k].0 - c[k].1) / (2.0 * step_bohr)))
        .collect()
}

fn assert_step(step_bohr: f64) {
    assert!(
        step_bohr.is_finite() && step_bohr > 0.0,
        "step must be positive and finite, got {step_bohr}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn water() -> Molecule {
        Molecule::new(
            vec!["O".into(), "H".into(), "H".into()],
            vec![
                Vector3::new(0.0, 0.0, 0.117),
                Vector3::new(0.0, 0.757, -0.469),
                Vector3::new(0.0, -0.757, -0.469),
            ],
        )
        .unwrap()
    }

    #[test]
    fn makes_6n_structures_in_order() {
        let d = displace(&water(), 0.005);
        assert_eq!(d.len(), 18);
        let labels: Vec<_> = d.iter().take(4).map(Displaced::label).collect();
        assert_eq!(labels, ["a0+x", "a0-x", "a0+y", "a0-y"]);
        assert_eq!(d[17].label(), "a2-z");
        let idx: Vec<_> = d.iter().take(6).map(Displaced::index).collect();
        assert_eq!(idx, [0, 0, 1, 1, 2, 2]);
        assert_eq!(d[17].index(), 8);
    }

    #[test]
    fn names_are_index_and_sign() {
        let d = displace(&water(), 0.005);
        assert_eq!(d[0].name(), "0000p");
        assert_eq!(d[1].name(), "0000m");
        assert_eq!(d[3].name(), "0001m");
        assert_eq!(d[17].name(), "0008m");
    }

    /// E = sum_i (c_i x_i + a_i x_i^2), x in bohr. Central differences are exact.
    fn quadratic(m: &Molecule) -> f64 {
        let x: Vec<f64> = m
            .coord_bohr()
            .iter()
            .flat_map(|r| [r.x, r.y, r.z])
            .collect();
        x.iter()
            .enumerate()
            .map(|(i, &xi)| (0.1 * i as f64 - 0.3) * xi + (0.5 + 0.2 * i as f64) * xi * xi)
            .sum()
    }

    fn quadratic_gradient(m: &Molecule) -> Vec<Vector3<f64>> {
        let x: Vec<f64> = m
            .coord_bohr()
            .iter()
            .flat_map(|r| [r.x, r.y, r.z])
            .collect();
        let g: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (0.1 * i as f64 - 0.3) + 2.0 * (0.5 + 0.2 * i as f64) * xi)
            .collect();
        g.chunks_exact(3).map(Vector3::from_column_slice).collect()
    }

    #[test]
    fn gradient_matches_analytic_for_quadratic() {
        let m = water();
        let step = 0.005;
        let d = displace(&m, step);
        let e: Vec<f64> = d.iter().map(|d| quadratic(&d.molecule)).collect();
        let pairs: Vec<(f64, f64)> = e.chunks_exact(2).map(|e| (e[0], e[1])).collect();

        let g = gradient(step, &pairs);
        assert_eq!(g.len(), 3);
        for (a, b) in g.iter().zip(quadratic_gradient(&m)) {
            assert_relative_eq!(*a, b, epsilon = 1e-9);
        }
    }

    #[test]
    #[should_panic(expected = "3 coordinates per atom")]
    fn gradient_rejects_incomplete_atoms() {
        gradient(0.005, &[(0.0, 0.0); 4]);
    }

    #[test]
    fn shifts_only_one_coordinate_by_step() {
        let m = water();
        let step = 0.01; // bohr
        let d = displace(&m, step);

        // d[9] = a1-y
        assert_eq!(d[9].label(), "a1-y");
        let diff = d[9].molecule.coord()[1] - m.coord()[1];
        assert_relative_eq!(diff, Vector3::new(0.0, -step * BOHR_TO_ANG, 0.0));
        assert_eq!(d[9].molecule.coord()[0], m.coord()[0]);
        assert_eq!(d[9].molecule.coord()[2], m.coord()[2]);
        assert_eq!(d[9].molecule.atoms(), m.atoms());
    }

    #[test]
    fn displacement_is_in_bohr() {
        let m = water();
        let d = displace(&m, 1.0);
        let shift_bohr = d[0].molecule.coord_bohr()[0].x - m.coord_bohr()[0].x;
        assert_relative_eq!(shift_bohr, 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn rejects_non_positive_step() {
        displace(&water(), 0.0);
    }
}
