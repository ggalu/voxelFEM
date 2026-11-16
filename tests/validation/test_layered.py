"""
Layered-composite FEM validation tests.
========================================

A two-phase layered composite has *exact* closed-form effective moduli on a
voxel grid (harmonic mean normal to the layers, arithmetic mean parallel to
them).  These tests drive ``VoxelFEMSolverNonlinear`` in its linear mode
(``is_linear=True``, ``preconditioner="jacobi"``) and compare the FEM effective modulus / per-phase fields against ``layered_reference``.

To remove uint8 quantisation from the comparison, the reference is evaluated
with the *actual* discretised moduli the solver sees (``info['E*_actual']``),
so the only error sources are the free-lateral-face Poisson coupling and
float32 round-off.

Boundary conditions use a minimal 3-2-1 anchoring so the lateral faces stay
traction-free (required for a clean uniaxial effective-modulus measurement);
fully constraining the lateral faces would inject a spurious Poisson stiffness.

Poisson note
------------
The series (harmonic-mean) result is the *uniaxial-stress* effective modulus.
With ``nu > 0`` and free lateral faces, bonded layers are forced to share a
common in-plane strain, so the apparent normal modulus sits measurably above
the harmonic mean (~3.8% at E2/E1=10, nu=0.3 — a real laminate-coupling effect,
not solver error).  The harmonic mean is *exact* only when there is no Poisson
coupling, so the series/normal tests below run at ``nu=0`` (verified exact to
<0.1% up to contrast 1000).  The parallel (arithmetic-mean) and iso-strain
results are exact at any ``nu`` and are tested at the physical ``nu=0.3``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cupy as cp           # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")

from geometry import build_node_index                                    # noqa: E402
from reference_solutions import make_layered_voxels, layered_reference   # noqa: E402
from report import (                                                     # noqa: E402
    build_conn_dof, E_elem_from_voxels, effective_modulus_from_reaction,
    volume_average_stress, volume_average_strain,
)

E_VOID = 1e-6
DELTA_FRAC = 1e-3       # prescribed face displacement = DELTA_FRAC * L


# ---------------------------------------------------------------------------
# BC helper — minimal 3-2-1 anchoring, lateral faces free
# ---------------------------------------------------------------------------

def _face_nodes(node_idx, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    return node_idx[tuple(sl)].ravel()


def _apply_uniaxial_bc(solver, node_idx, load_axis, delta):
    """Prescribe axial displacement on the high face; anchor rigid-body modes."""
    a = load_axis
    b = (a + 1) % 3
    c = (a + 2) % 3

    low = _face_nodes(node_idx, a, 0)
    high = _face_nodes(node_idx, a, -1)
    solver.apply_bc_displacement(low, np.full(len(low), a), np.zeros(len(low)))
    solver.apply_bc_displacement(high, np.full(len(high), a),
                                 np.full(len(high), delta))

    # Anchor the 3 transverse rigid-body modes without touching the lateral
    # faces' axial freedom (so they remain traction-free).
    origin = int(node_idx[0, 0, 0])
    solver.apply_bc_displacement(np.array([origin, origin]),
                                 np.array([b, c]), np.array([0.0, 0.0]))
    sel = [0, 0, 0]
    sel[b] = -1
    bnode = int(node_idx[tuple(sel)])               # +b extreme, kills rot about a
    solver.apply_bc_displacement(np.array([bnode]), np.array([c]),
                                 np.array([0.0]))


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------

def run_layered_case(N, layer_axis, load_axis, E1, E2, f1, E_max, nu,
                     max_pcg_iter=2000, tol_pcg=1e-9):
    """
    Build, solve and post-process one layered case.

    Returns a dict with the FEM effective modulus, the analytic reference
    (evaluated at the *actual* discretised moduli), per-element data, and the
    solution vector.
    """
    from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear

    voxels, info = make_layered_voxels(N, N, N, layer_axis, E1, E2, f1,
                                       E_max, E_void=E_VOID)
    L = N
    delta = DELTA_FRAC * L

    solver = VoxelFEMSolverNonlinear(
        voxels, nu=nu, E_max=E_max, E_void=E_VOID,
        is_linear=True, preconditioner="jacobi", verbose=False,
        tol_pcg=tol_pcg, max_pcg_iter=max_pcg_iter,
    )
    node_idx = build_node_index(N, N, N)
    _apply_uniaxial_bc(solver, node_idx, load_axis, delta)
    u = solver.solve()

    conn_dof = build_conn_dof(N, N, N)
    E_elem = E_elem_from_voxels(voxels, E_max, E_VOID)
    Ndof = solver.Ndof

    high = _face_nodes(node_idx, load_axis, -1)
    A = N * N
    E_eff_fem = effective_modulus_from_reaction(
        u, E_elem, conn_dof, nu, Ndof, high, load_axis, delta, A, L)

    ref = layered_reference(info['E1_actual'], info['E2_actual'],
                            info['f1_actual'], nu)

    return dict(u=u, info=info, voxels=voxels, E_elem=E_elem, conn_dof=conn_dof,
                Ndof=Ndof, nu=nu, node_idx=node_idx, E_eff_fem=E_eff_fem,
                ref=ref, delta=delta, L=L, N=N)


# ---------------------------------------------------------------------------
# Test 3a — effective modulus, normal loading (harmonic mean)
# ---------------------------------------------------------------------------

def test_layered_normal_modulus():
    """Loading normal to the layers => harmonic mean (nu=0, exact), within 0.5%."""
    r = run_layered_case(N=40, layer_axis=2, load_axis=2,
                         E1=1.0, E2=10.0, f1=0.5, E_max=10.0, nu=0.0)
    fem = r['E_eff_fem']
    ref = r['ref']['E_eff_normal']
    err = abs(fem - ref) / ref
    print(f"Normal modulus: FEM={fem:.4f}  ref={ref:.4f}  err={err*100:.2f}%")
    assert err < 0.005, f"Normal modulus error {err*100:.2f}% (FEM={fem}, ref={ref})"


# ---------------------------------------------------------------------------
# Test 3b — effective modulus, parallel loading (arithmetic mean)
# ---------------------------------------------------------------------------

def test_layered_parallel_modulus():
    """Loading parallel to the layers => arithmetic mean, within 1%."""
    r = run_layered_case(N=40, layer_axis=2, load_axis=0,
                         E1=1.0, E2=10.0, f1=0.5, E_max=10.0, nu=0.3)
    fem = r['E_eff_fem']
    ref = r['ref']['E_eff_parallel']
    err = abs(fem - ref) / ref
    print(f"Parallel modulus: FEM={fem:.4f}  ref={ref:.4f}  err={err*100:.2f}%")
    assert err < 0.01, f"Parallel modulus error {err*100:.2f}% (FEM={fem}, ref={ref})"


# ---------------------------------------------------------------------------
# Test 3c — contrast-ratio sweep (harmonic mean, normal loading)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("contrast", [2, 10, 100, 1000])
def test_layered_contrast_sweep(contrast):
    """
    Harmonic-mean formula across contrast ratios, normal loading (nu=0), <1%.

    At nu=0 the series result is exact, so this isolates the solver / PCG
    behaviour at increasing stiffness contrast — the contrast=1000 case is the
    hardest for CG conditioning.  The reference is evaluated at the actual
    discretised moduli, so uint8 quantisation of the soft phase is not counted
    as error; the stiff phase is pinned to value 255.
    """
    E1, E2, E_max = 1.0, float(contrast), float(contrast)
    r = run_layered_case(N=40, layer_axis=2, load_axis=2,
                         E1=E1, E2=E2, f1=0.5, E_max=E_max, nu=0.0,
                         max_pcg_iter=8000)
    fem = r['E_eff_fem']
    ref = r['ref']['E_eff_normal']
    err = abs(fem - ref) / ref
    print(f"Contrast {contrast}: FEM={fem:.4f}  ref={ref:.4f}  err={err*100:.2f}% "
          f"(actual E1={r['info']['E1_actual']:.4g}, E2={r['info']['E2_actual']:.4g})")
    assert err < 0.01, f"Contrast {contrast} error {err*100:.2f}%"


# ---------------------------------------------------------------------------
# Test 3d — per-phase stress, normal loading (stress continuity)
# ---------------------------------------------------------------------------

def test_layered_per_phase_stress():
    """
    Normal loading: both phases carry the same sigma_zz (equilibrium), and the
    volume-averaged sigma_zz equals E_eff * applied_strain (global equilibrium).

    Run at nu=0 so the harmonic-mean reference for the global-equilibrium check
    is exact (stress continuity itself holds at any nu).
    """
    r = run_layered_case(N=40, layer_axis=2, load_axis=2,
                         E1=1.0, E2=10.0, f1=0.5, E_max=10.0, nu=0.0)
    voxels = r['voxels']
    val1, val2 = r['info']['val1'], r['info']['val2']
    mask1 = voxels.ravel() == val1
    mask2 = voxels.ravel() == val2

    s1 = volume_average_stress(r['u'], r['E_elem'], r['conn_dof'], r['nu'], mask1)
    s2 = volume_average_stress(r['u'], r['E_elem'], r['conn_dof'], r['nu'], mask2)
    s1_zz, s2_zz = s1[2], s2[2]
    s_avg = 0.5 * (s1_zz + s2_zz)

    rel_diff = abs(s1_zz - s2_zz) / abs(s_avg)
    print(f"Per-phase sigma_zz: phase1={s1_zz:.5e}  phase2={s2_zz:.5e}  "
          f"rel.diff={rel_diff*100:.2f}%")
    assert rel_diff < 0.01, f"Stress discontinuity {rel_diff*100:.2f}% across phases"

    # Global equilibrium: <sigma_zz> = E_eff_normal * applied strain.
    eps_applied = r['delta'] / r['L']
    sigma_expected = r['ref']['E_eff_normal'] * eps_applied
    all_zz = volume_average_stress(r['u'], r['E_elem'], r['conn_dof'], r['nu'],
                                   np.ones(voxels.size, dtype=bool))[2]
    err = abs(all_zz - sigma_expected) / abs(sigma_expected)
    print(f"<sigma_zz>={all_zz:.5e}  expected={sigma_expected:.5e}  err={err*100:.2f}%")
    assert err < 0.005, f"Global sigma_zz off by {err*100:.2f}%"


# ---------------------------------------------------------------------------
# Test 3e — per-phase strain, parallel loading (iso-strain)
# ---------------------------------------------------------------------------

def test_layered_per_phase_strain():
    """Parallel loading: both phases carry the same eps_xx (compatibility)."""
    r = run_layered_case(N=40, layer_axis=2, load_axis=0,
                         E1=1.0, E2=10.0, f1=0.5, E_max=10.0, nu=0.3)
    voxels = r['voxels']
    val1, val2 = r['info']['val1'], r['info']['val2']
    mask1 = voxels.ravel() == val1
    mask2 = voxels.ravel() == val2

    e1 = volume_average_strain(r['u'], r['conn_dof'], mask1)
    e2 = volume_average_strain(r['u'], r['conn_dof'], mask2)
    e1_xx, e2_xx = e1[0], e2[0]
    e_avg = 0.5 * (e1_xx + e2_xx)

    rel_diff = abs(e1_xx - e2_xx) / abs(e_avg)
    print(f"Per-phase eps_xx: phase1={e1_xx:.5e}  phase2={e2_xx:.5e}  "
          f"rel.diff={rel_diff*100:.2f}%")
    assert rel_diff < 0.01, f"Strain mismatch {rel_diff*100:.2f}% across phases"

    # eps_xx should equal the applied axial strain.
    eps_applied = r['delta'] / r['L']
    err = abs(e_avg - eps_applied) / eps_applied
    print(f"<eps_xx>={e_avg:.5e}  applied={eps_applied:.5e}  err={err*100:.2f}%")
    assert err < 0.01, f"Average eps_xx off by {err*100:.2f}%"


# ---------------------------------------------------------------------------
# Report-generator hook
# ---------------------------------------------------------------------------

def collect_report_rows():
    """
    Run a representative subset and yield (label, fem, ref, rel_err, passed)
    rows for ``report.run_validation_report``.  Uses a modest grid for speed.
    """
    rows = []

    r = run_layered_case(32, 2, 2, 1.0, 10.0, 0.5, 10.0, 0.0)
    fem, ref = r['E_eff_fem'], r['ref']['E_eff_normal']
    err = abs(fem - ref) / ref
    rows.append(("Normal modulus (E2/E1=10)", fem, ref, err, err < 0.01))

    r = run_layered_case(32, 2, 0, 1.0, 10.0, 0.5, 10.0, 0.3)
    fem, ref = r['E_eff_fem'], r['ref']['E_eff_parallel']
    err = abs(fem - ref) / ref
    rows.append(("Parallel modulus (E2/E1=10)", fem, ref, err, err < 0.01))

    r = run_layered_case(32, 2, 2, 1.0, 100.0, 0.5, 100.0, 0.0, max_pcg_iter=8000)
    fem, ref = r['E_eff_fem'], r['ref']['E_eff_normal']
    err = abs(fem - ref) / ref
    rows.append(("Normal modulus (E2/E1=100)", fem, ref, err, err < 0.01))

    return rows
