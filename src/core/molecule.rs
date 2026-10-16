use nalgebra::Vector3;
use thiserror::Error;

use super::units::ANG_TO_BOHR;

#[derive(Debug, Error, PartialEq)]
pub enum MoleculeError {
    #[error("atom/coord mismatch: {atoms} atoms vs {coords} coords")]
    LengthMismatch { atoms: usize, coords: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Molecule {
    atoms: Vec<String>,
    coord: Vec<Vector3<f64>>, // Angstrom 
}

impl Molecule {
    pub fn new(atoms: Vec<String>, coord: Vec<Vector3<f64>>) -> Result<Self, MoleculeError> {
        if atoms.len() != coord.len() {
            return Err(MoleculeError::LengthMismatch {
                atoms: atoms.len(),
                coords: coord.len(),
            });
        }
        Ok(Self { atoms, coord })
    }

    pub fn atoms(&self) -> &[String] {
        &self.atoms
    }

    pub fn coord(&self) -> &[Vector3<f64>] {
        &self.coord
    }

    pub fn coord_bohr(&self) -> Vec<Vector3<f64>> {
        self.coord.iter().map(|r| r * ANG_TO_BOHR).collect()
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    pub fn centroid(&self) -> Vector3<f64> {
        self.coord.iter().sum::<Vector3<f64>>() / self.len() as f64
    }

    pub fn map_coord(&self, f: impl Fn(&Vector3<f64>) -> Vector3<f64>) -> Self {
        Self {
            atoms: self.atoms.clone(),
            coord: self.coord.iter().map(f).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Molecule {
        Molecule::new(
            vec!["O".into(), "H".into(), "H".into()],
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.757, 0.586, 0.0),
                Vector3::new(-0.757, 0.586, 0.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn rejects_length_mismatch() {
        let err = Molecule::new(vec!["H".into()], vec![]).unwrap_err();
        assert_eq!(err, MoleculeError::LengthMismatch { atoms: 1, coords: 0 });
    }

    #[test]
    fn map_coord_keeps_atoms() {
        let m = water();
        let shifted = m.map_coord(|r| r + Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(shifted.atoms(), m.atoms());
        assert_eq!(shifted.coord()[0], Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(m.coord()[0], Vector3::zeros());
    }

    #[test]
    fn centroid_of_water() {
        let c = water().centroid();
        approx::assert_relative_eq!(c, Vector3::new(0.0, 2.0 * 0.586 / 3.0, 0.0));
    }
}
