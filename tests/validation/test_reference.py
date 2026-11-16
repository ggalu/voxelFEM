"""
Unit tests for the analytical reference solutions (layered composite).

Pure NumPy — no FEM solver, no GPU required.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_solutions import layered_reference, make_layered_voxels   # noqa: E402


# ---------------------------------------------------------------------------
# layered_reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45])
def test_layered_homogeneous_limit(nu):
    """E1 == E2 => both effective moduli equal that common modulus."""
    ref = layered_reference(E1=7.0, E2=7.0, f1=0.5, nu=nu)
    assert ref['E_eff_normal'] == pytest.approx(7.0, rel=1e-12)
    assert ref['E_eff_parallel'] == pytest.approx(7.0, rel=1e-12)


def test_layered_series_softer_than_parallel():
    """Series (harmonic) arrangement is always softer than parallel (arithmetic)."""
    ref = layered_reference(E1=1.0, E2=10.0, f1=0.5, nu=0.3)
    assert ref['E_eff_normal'] < ref['E_eff_parallel']
    # Sanity: harmonic mean of 1 and 10 at f=0.5 is 2*1*10/11 = 1.818...
    assert ref['E_eff_normal'] == pytest.approx(2 * 1.0 * 10.0 / 11.0, rel=1e-12)
    assert ref['E_eff_parallel'] == pytest.approx(5.5, rel=1e-12)


def test_layered_stress_strain_partitioning():
    """Equal stress in both phases (normal); equal strain in both phases (parallel)."""
    ref = layered_reference(E1=2.0, E2=20.0, f1=0.4, nu=0.3)
    assert ref['sigma_1_normal'] == pytest.approx(ref['sigma_2_normal'])
    assert ref['sigma_1_normal'] == pytest.approx(ref['E_eff_normal'])
    assert ref['eps_1_parallel'] == pytest.approx(ref['eps_2_parallel'])


# ---------------------------------------------------------------------------
# make_layered_voxels
# ---------------------------------------------------------------------------

def test_make_layered_voxels_fractions_and_values():
    voxels, info = make_layered_voxels(
        Nx=40, Ny=40, Nz=40, layer_axis=2, E1=1.0, E2=10.0, f1=0.5, E_max=10.0
    )
    assert voxels.shape == (40, 40, 40)
    assert voxels.dtype == np.uint8
    assert info['f1_actual'] == pytest.approx(0.5)

    # Phase counts along the layer axis match the reported fraction.
    n1 = int(round(info['f1_actual'] * 40))
    assert (voxels == info['val1']).sum() == n1 * 40 * 40
    assert (voxels == info['val2']).sum() == (40 - n1) * 40 * 40

    # Each slab is constant across the in-plane directions.
    assert np.all(voxels[:, :, 0] == voxels[0, 0, 0])
    assert np.all(voxels[:, :, -1] == voxels[0, 0, -1])


def test_make_layered_voxels_actual_moduli_ordered():
    voxels, info = make_layered_voxels(
        Nx=8, Ny=8, Nz=8, layer_axis=0, E1=1.0, E2=100.0, f1=0.5, E_max=100.0
    )
    assert info['E1_actual'] < info['E2_actual']
    assert info['val1'] >= 1 and info['val2'] <= 255
