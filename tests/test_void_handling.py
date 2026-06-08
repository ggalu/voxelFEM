"""
Stage 1 void-handling tests: binary solid/void masking.

Geometry
--------
40×40×40 voxel domain.  A central fibre column oriented along Z occupies a
28×28 cross-section (voxels x ∈ [6,34), y ∈ [6,34), all z).
Column width = 28, so 28²/40² ≈ 0.49 volume fraction.

The surrounding region is AIR/VOID (solid_mask=False).  The fibre column is the
only load-bearing structure.

Under uniaxial tension along Z at 1 % engineering strain with nu=0 the recovered
axial Young's modulus must equal E_fibre to within 5 % (homogeneous prism in the
continuum limit → exact; FEM discretization introduces only a small error).

Test coverage
-------------
  1. Solver converges without stalling (void DOFs properly decoupled).
  2. E_eff = reaction_force / (solid_area × strain) ≈ E_fibre.
  3. All inactive (void-only) DOFs remain at zero in the solution.
  4. Connected-component check finds no floating islands.
"""

import sys
import os
import io
import contextlib
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not available"
)

# ---------------------------------------------------------------------------
# Geometry / problem constants
# ---------------------------------------------------------------------------
NX = NY = NZ = 40          # domain edge length (voxels)
COL_LO = 6                 # column starts at index 6 in x and y (inclusive)
COL_HI = 34                # column ends at index 34 (exclusive) → 28-voxel width
COL_W = COL_HI - COL_LO   # = 28
SOLID_AREA = float(COL_W ** 2)   # 784 unit cells (= voxels²)

E_FIBRE = 1.0    # Young's modulus of fibre material
NU      = 0.0    # zero Poisson → exact uniaxial decoupling (no lateral forces)
STRAIN  = 0.01   # 1 % engineering strain
DISP_TOP = STRAIN * NZ   # = 0.4 voxel units


# ---------------------------------------------------------------------------
# Helper: build solver for the column-in-void geometry
# ---------------------------------------------------------------------------

def _make_column_problem(preconditioner="jacobi", verbose=False):
    """
    Build and return a configured VoxelFEMSolverNonlinear for the column test.

    All voxels are set to 255 so solid elements have E = E_FIBRE.
    The solid_mask restricts load-bearing to the 28×28 column only.
    BCs: z = 0 face fixed, z = NZ face prescribed; 3-2-1 anchoring on solid
    column nodes to suppress transverse rigid-body modes.
    """
    from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
    from geometry import build_node_index

    voxels = np.full((NX, NY, NZ), 255, dtype=np.uint8)

    solid_mask = np.zeros((NX, NY, NZ), dtype=bool)
    solid_mask[COL_LO:COL_HI, COL_LO:COL_HI, :] = True

    solver = VoxelFEMSolverNonlinear(
        voxels,
        nu=NU,
        E_max=E_FIBRE,
        E_void=1e-6,
        solid_mask=solid_mask,
        is_linear=True,
        tol_pcg=1e-8,
        max_pcg_iter=500,
        preconditioner=preconditioner,
        verbose=verbose,
    )

    node_idx = build_node_index(NX, NY, NZ)
    bottom_nodes = node_idx[:, :, 0].ravel()    # all nodes on z = 0
    top_nodes    = node_idx[:, :, -1].ravel()   # all nodes on z = NZ

    # Uniaxial tension along Z
    solver.apply_bc_displacement(
        bottom_nodes,
        np.full(len(bottom_nodes), 2, dtype=int),
        np.zeros(len(bottom_nodes)),
    )
    solver.apply_bc_displacement(
        top_nodes,
        np.full(len(top_nodes), 2, dtype=int),
        np.full(len(top_nodes), DISP_TOP),
    )

    # 3-2-1 anchoring on solid-column nodes to fix transverse rigid-body modes.
    # Nodes at corners of the column base (all active, touching solid elements):
    #   (COL_LO, COL_LO, 0): fix u_x, u_y
    #   (COL_HI, COL_LO, 0): fix u_y
    anchor_a = int(node_idx[COL_LO, COL_LO, 0])
    anchor_b = int(node_idx[COL_HI, COL_LO, 0])
    solver.apply_bc_displacement([anchor_a], [0], [0.0])   # u_x
    solver.apply_bc_displacement([anchor_a], [1], [0.0])   # u_y
    solver.apply_bc_displacement([anchor_b], [1], [0.0])   # u_y

    return solver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_void_column_convergence():
    """
    Solver must converge without stalling on the column-in-void geometry.

    Confirms that void DOFs are properly decoupled (identity rows) so the PCG
    operator remains SPD on the full vector space.
    """
    solver = _make_column_problem(preconditioner="jacobi", verbose=True)
    u = solver.solve()

    assert not np.any(np.isnan(u)), "NaN in displacement"
    assert not np.any(np.isinf(u)), "Inf in displacement"
    assert np.max(np.abs(u)) > 1e-6, "All-zero displacement (solver failed silently)"
    print(f"✓ Convergence: max|u| = {np.max(np.abs(u)):.6f}")


def test_void_column_effective_modulus():
    """
    Effective Young's modulus of the fibre column must equal E_fibre within 5 %.

    E_eff = reaction_force / (solid_area × strain)

    The column is a homogeneous prism under uniaxial tension with nu=0, so
    E_eff → E_fibre as the discretisation is refined.  At 28×28 the error from
    FEM approximation should be well below 5 %.

    Reaction force is computed via compute_f_int_cupy with the solid mask so
    that void-element contributions are excluded.
    """
    from kernels_cupy_nl import compute_f_int_cupy
    from geometry import build_node_index

    solver = _make_column_problem(preconditioner="jacobi", verbose=False)
    u = solver.solve()

    u_gpu = cp.asarray(u.astype(np.float32))
    f_int_gpu = compute_f_int_cupy(
        u_gpu,
        solver.E0_elem_gpu,
        solver.B_gauss_gpu,
        solver.w_gauss_gpu,
        NU,
        solver.constitutive,
        solver.Ny, solver.Nz,
        solver.Ndof,
        solid_mask_gpu=solver.solid_elem_gpu,
    )
    f_int = cp.asnumpy(f_int_gpu)

    # Sum z-DOF reaction forces at the top face
    node_idx = build_node_index(NX, NY, NZ)
    top_nodes  = node_idx[:, :, -1].ravel()
    top_z_dofs = top_nodes * 3 + 2
    reaction   = float(np.sum(f_int[top_z_dofs]))

    E_eff    = reaction / (SOLID_AREA * STRAIN)
    rel_err  = abs(E_eff - E_FIBRE) / E_FIBRE

    print(f"Reaction = {reaction:.6f}, solid_area = {SOLID_AREA:.0f}, "
          f"strain = {STRAIN}")
    print(f"E_eff = {E_eff:.6f}  (E_fibre = {E_FIBRE:.6f},  rel_err = {rel_err:.3%})")

    assert rel_err < 0.05, (
        f"Effective modulus error {rel_err:.3%} exceeds 5 % tolerance "
        f"(E_eff = {E_eff:.6f}, E_fibre = {E_FIBRE:.6f})"
    )
    print(f"✓ E_eff = {E_eff:.4f}  (error {rel_err:.2%} < 5 %)")


def test_void_inactive_dofs_are_zero():
    """
    All DOFs belonging exclusively to void-region nodes must stay at zero.

    Inactive DOFs receive identity rows (y[i] = v[i]) in the operator and are
    zeroed in the preconditioner, so PCG keeps them exactly at the initial
    value of zero throughout the solve.
    """
    solver = _make_column_problem(preconditioner="jacobi", verbose=False)
    u = solver.solve()

    assert solver.inactive_dofs_gpu is not None, \
        "solver.inactive_dofs_gpu not built — solid_mask not applied?"

    inactive_np = cp.asnumpy(solver.inactive_dofs_gpu).astype(np.int64)
    n_inactive = len(inactive_np)
    assert n_inactive > 0, "Expected non-zero inactive DOF count for column-in-void"

    # Inactive DOFs that also have explicit BCs (e.g. void top-face nodes with
    # prescribed z-displacement) can legitimately be non-zero — exclude them.
    # Only DOFs that are inactive AND not explicitly BC-constrained must stay 0.
    unconstrained_inactive = inactive_np[~solver.bc_mask[inactive_np]]
    n_free = len(unconstrained_inactive)

    max_inactive = (float(np.max(np.abs(u[unconstrained_inactive])))
                    if n_free > 0 else 0.0)
    assert max_inactive < 1e-6, (
        f"Free inactive DOFs are non-zero: max|u[inactive_free]| = {max_inactive:.3e}"
    )
    print(f"✓ {n_inactive:,} inactive DOFs total; "
          f"{n_free:,} unconstrained all ≤ {max_inactive:.1e}")


def test_void_no_floating_islands():
    """
    The fibre column is a single connected component anchored by bottom-face BCs.
    _check_floating_islands must produce no WARNING output.
    """
    from geometry import build_node_index

    solver = _make_column_problem(preconditioner="jacobi", verbose=False)

    # Build the same bottom-face BC DOFs that solve() would use
    node_idx    = build_node_index(NX, NY, NZ)
    bottom_nodes = node_idx[:, :, 0].ravel()
    bc_z_dofs   = (bottom_nodes * 3 + 2).astype(np.int32)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._check_floating_islands(bc_z_dofs)
    output = buf.getvalue()

    assert "WARNING" not in output, (
        f"Unexpected floating-island warning:\n{output}"
    )
    print("✓ No floating islands detected")
