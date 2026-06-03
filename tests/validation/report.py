"""
Post-processing helpers + validation report generator (layered composite).
===========================================================================

The solver (``VoxelFEMSolverNonlinear``) exposes only ``solve()`` and the BC /
force setters — it has no public stress-recovery or ``K·u`` method, and
``validation.md`` forbids modifying existing files.  So every post-processing
quantity here is rebuilt independently from the pure-NumPy primitives in
``geometry.py``:

  * reaction force / effective modulus  ← ``matvec_Ku`` (f_int = K·u)
  * element-centroid stress & strain    ← ``_B_matrix(0,0,0)``

These use the *same* per-element modulus encoding as the solver, so they
reproduce the solver's internal force operator on the CPU exactly (up to
float32 round-off).

Run as a script for a formatted report::

    python tests/validation/report.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from geometry import (                       # noqa: E402
    build_node_index, element_node_ids, build_K_ref, matvec_Ku, _B_matrix,
)


# ---------------------------------------------------------------------------
# Shared FEM bookkeeping
# ---------------------------------------------------------------------------

def build_conn_dof(Nx, Ny, Nz):
    """(Ne, 24) element DOF table — element order matches voxels.ravel()."""
    node_idx = build_node_index(Nx, Ny, Nz)
    conn_nodes = element_node_ids(node_idx, Nx, Ny, Nz)
    return (conn_nodes[:, :, np.newaxis] * 3
            + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)


def E_elem_from_voxels(voxels, E_max, E_void=1e-6):
    """Per-element base modulus under the solver's default voxel encoding."""
    E_scale = (E_max - E_void) / 255.0
    return voxels.ravel().astype(np.float32) * E_scale + E_void


def _D_ref(nu):
    """Unit-modulus isotropic elasticity matrix (matches geometry.build_K_ref)."""
    c1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c3 = 1.0 / (2.0 * (1.0 + nu))
    return np.array([
        [c1, c2, c2,  0,  0,  0],
        [c2, c1, c2,  0,  0,  0],
        [c2, c2, c1,  0,  0,  0],
        [ 0,  0,  0, c3,  0,  0],
        [ 0,  0,  0,  0, c3,  0],
        [ 0,  0,  0,  0,  0, c3],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Reaction force / effective modulus
# ---------------------------------------------------------------------------

def internal_force(u, E_elem, conn_dof, nu, Ndof):
    """f_int = K·u via the matrix-free NumPy reference operator."""
    K_ref = build_K_ref(nu)
    return matvec_Ku(u.astype(np.float32), E_elem, conn_dof, K_ref, Ndof)


def reaction_force(u, E_elem, conn_dof, nu, Ndof, face_node_ids, dof):
    """
    Total reaction force on a constrained face.

    Sums the ``dof`` component of f_int = K·u over the given face nodes.
    """
    f_int = internal_force(u, E_elem, conn_dof, nu, Ndof)
    return float(f_int[np.asarray(face_node_ids) * 3 + dof].sum())


def effective_modulus_from_reaction(u, E_elem, conn_dof, nu, Ndof,
                                    face_node_ids, dof, delta, A, L):
    """
    E_eff = (F / A) / (delta / L).

    F   : total reaction force on the prescribed face (``dof`` component)
    A   : loaded cross-section area (voxel units)
    delta : prescribed face displacement
    L   : domain length along the load direction (voxel units)
    """
    F = reaction_force(u, E_elem, conn_dof, nu, Ndof, face_node_ids, dof)
    return (F / A) / (delta / L)


# ---------------------------------------------------------------------------
# Element-centroid stress / strain recovery
# ---------------------------------------------------------------------------

def _centroid_eps(u, conn_dof, chunk_size=500_000):
    """Strain (Ne, 6) at every element centroid (xi=eta=zeta=0)."""
    B_c, _ = _B_matrix(0.0, 0.0, 0.0)        # centroid of [-1,1]^3 -> [0,1]^3
    B_c = B_c.astype(np.float64)
    Ne = conn_dof.shape[0]
    eps = np.zeros((Ne, 6), dtype=np.float64)
    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        u_loc = u[conn_dof[start:end]]       # (chunk, 24)
        eps[start:end] = u_loc @ B_c.T       # (chunk, 6)
    return eps


def element_stress(u, E_elem, conn_dof, nu, chunk_size=500_000):
    """
    Average stress at each element centroid: sigma = E_elem * D_ref · (B·u_loc).

    Returns (Ne, 6) [xx, yy, zz, xy, yz, xz] (Voigt order per geometry._B_matrix).
    """
    eps = _centroid_eps(u, conn_dof, chunk_size)
    D = _D_ref(nu)
    return (eps @ D.T) * E_elem.astype(np.float64)[:, None]


def element_strain(u, conn_dof, chunk_size=500_000):
    """Average strain at each element centroid. Returns (Ne, 6)."""
    return _centroid_eps(u, conn_dof, chunk_size)


def volume_average_stress(u, E_elem, conn_dof, nu, mask):
    """Volume-averaged stress (6,) over elements where ``mask`` is True."""
    sigma = element_stress(u, E_elem, conn_dof, nu)
    return sigma[mask].mean(axis=0)


def volume_average_strain(u, conn_dof, mask):
    """Volume-averaged strain (6,) over elements where ``mask`` is True."""
    eps = element_strain(u, conn_dof)
    return eps[mask].mean(axis=0)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def _fmt_row(name, fem, ref, err, status):
    return f"{name:<30s}{fem:<11}{ref:<12}{err:<9}{status}"


def run_validation_report():
    """Run the layered-composite validation cases and print a formatted table."""
    from tests.validation import test_layered as TL

    print("=" * 60)
    print("  VOXEL FEM SOLVER — ANALYTICAL VALIDATION REPORT")
    print("=" * 60)
    print("\nLAYERED COMPOSITE TESTS")
    print("-" * 23)
    print(_fmt_row("Test", "FEM", "Reference", "Error", "Status"))
    print(_fmt_row("----", "---", "---------", "-----", "------"))

    n_pass = 0
    n_total = 0
    for label, fem, ref, err, ok in TL.collect_report_rows():
        n_total += 1
        n_pass += int(ok)
        if ref is None:
            print(_fmt_row(label, f"{fem}", "--", "--",
                           "PASS" if ok else "FAIL"))
        else:
            print(_fmt_row(label, f"{fem:.4g}", f"{ref:.4g}",
                           f"{err*100:.2f}%", "PASS" if ok else "FAIL"))

    print("\n" + "=" * 60)
    print(f"OVERALL: {n_pass}/{n_total} checks PASSED")
    print("=" * 60)
    return n_pass == n_total


if __name__ == "__main__":
    ok = run_validation_report()
    sys.exit(0 if ok else 1)
