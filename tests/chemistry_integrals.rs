//! Tests for the STO-3G integral engine.
//!
//! Expected values come from an independent source, not from the formulas the
//! implementation uses: the H2 overlap at 1.4 bohr is the value printed by the
//! teaching notebook referenced in docs/quantum-chemistry-spec.md, and the
//! remaining checks are physical identities an integral engine must satisfy.

use quantumforge::chemistry::{Atom, Element, sto3g_integrals};

fn h2() -> Vec<Atom> {
    vec![
        Atom {
            element: Element::H,
            position: [0.0, 0.0, 0.0],
        },
        Atom {
            element: Element::H,
            position: [0.0, 0.0, 1.4],
        },
    ]
}

#[test]
fn h2_sto3g_overlap_matches_an_independent_reference() {
    let integrals = sto3g_integrals(&h2()).expect("H2 integrals");
    // Published notebook value for the raw (unrenormalised) contraction:
    // 0.65931916, with diagonal 1.00000134. Rescaling to exactly normalised
    // functions gives 0.65931916 / 1.00000134.
    let expected = 0.65931916 / 1.00000134;
    assert!(
        (integrals.overlap[0][1] - expected).abs() < 2e-6,
        "overlap {} vs {expected}",
        integrals.overlap[0][1]
    );
    assert!((integrals.overlap[0][0] - 1.0).abs() < 1e-12);
    assert!((integrals.overlap[1][1] - 1.0).abs() < 1e-12);
}

#[test]
fn integral_tensors_have_the_required_symmetries() {
    let integrals = sto3g_integrals(&h2()).expect("H2 integrals");
    for p in 0..2 {
        for q in 0..2 {
            assert!((integrals.overlap[p][q] - integrals.overlap[q][p]).abs() < 1e-14);
            assert!((integrals.kinetic[p][q] - integrals.kinetic[q][p]).abs() < 1e-14);
            for r in 0..2 {
                for s in 0..2 {
                    let v = integrals.repulsion[p][q][r][s];
                    assert!((v - integrals.repulsion[q][p][r][s]).abs() < 1e-14);
                    assert!((v - integrals.repulsion[p][q][s][r]).abs() < 1e-14);
                    assert!((v - integrals.repulsion[r][s][p][q]).abs() < 1e-14);
                }
            }
        }
    }
}

#[test]
fn one_center_helium_repulsion_matches_the_analytic_gaussian_result() {
    let atoms = vec![Atom {
        element: Element::He,
        position: [0.0, 0.0, 0.0],
    }];
    let integrals = sto3g_integrals(&atoms).expect("He integrals");
    let coulomb = integrals.repulsion[0][0][0][0];
    assert!(coulomb > 0.0);
    assert!(coulomb < 2.0);

    // Independent check: quadrature of the spherical average 1/max(r1,r2)
    // over the contracted radial density gives 1.05571294 hartree.
    assert!(
        (coulomb - 1.05571294).abs() < 1e-6,
        "contracted Coulomb {coulomb} vs independent quadrature"
    );
}

#[test]
fn nuclear_repulsion_is_the_geometric_distance() {
    let integrals = sto3g_integrals(&h2()).expect("H2 integrals");
    // E_nuc = Z_a Z_b / R = 1 / 1.4 hartree for two protons.
    assert!((integrals.nuclear_repulsion - 1.0 / 1.4).abs() < 1e-12);
}
