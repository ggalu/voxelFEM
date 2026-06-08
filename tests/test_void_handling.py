"""
Stage 1 void-handling tests: binary solid/void masking.

Part A — clean column in void
------------------------------
40×40×40 voxel domain.  A central fibre column oriented along Z occupies a
28×28 cross-section (voxels x ∈ [6,34), y ∈ [6,34), all z).
Column width = 28, so 28²/40² ≈ 0.49 volume fraction.

Under uniaxial tension along Z at 1 % engineering strain with nu=0 the recovered
axial Young's modulus must equal E_fibre to within 5 %.

Tests:
  1. Solver converges without stalling.
  2. E_eff ≈ E_fibre.
  3. Free inactive DOFs remain zero.
  4. No floating islands detected.

Part B — column + floating island clusters in void
---------------------------------------------------
Three fibre clusters are embedded in the void region around the column.
None touch the column, each other, or the z = 0 / z = NZ boundary faces:

  Cluster A: 1×1×1  at voxel (ix=2,  iy=2,  iz=20)  →  1 element
  Cluster B: 1×2×3  at voxel (ix=36, iy∈[2,4), iz∈[10,13))  →  6 elements
  Cluster C: 2×2×4  at voxel (ix∈[2,4), iy∈[36,38), iz∈[5,9))  → 16 elements

Because the islands have no load path to any Dirichlet BC:
  - _check_floating_islands warns about all three.
  - Their DOFs stay exactly zero throughout the PCG solve (zero RHS, zero
    initial guess, completely isolated from the column subspace).
  - E_eff is unchanged — islands do not contribute to the reaction force.
  - PCG converges in the same number of iterations as the island-free case,
    confirming that the null-space modes of the floating islands do not
    pollute the column's Krylov subspace.
"""

import sys
import os
import io
import re
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
# Part B — island geometry
# ---------------------------------------------------------------------------
# Island positions are chosen so that none of them are 26-adjacent (corner-
# connected) to the column or to each other, and none touch z=0 / z=NZ faces.
#
#   A:  1×1×1  at (2, 2, 20)  → 1 element,  8 nodes
#   B:  1×2×3  at (36, 2, 10) → ix=36, iy=[2,4), iz=[10,13) = 6 elements
#   C:  2×2×4  at (2, 36, 5)  → ix=[2,4), iy=[36,38), iz=[5,9) = 16 elements
#
# Column 26-expanded box: x∈[5,35), y∈[5,35) — all islands are outside.
# Islands are > 1 voxel apart from each other and from z-face nodes.

_ISLAND_SIZES = {
    "A": 1,    # 1×1×1
    "B": 6,    # 1×2×3
    "C": 16,   # 2×2×4
}
_ISLAND_TOTAL = sum(_ISLAND_SIZES.values())   # 23 elements


def _island_mask():
    """Return a (NX,NY,NZ) bool array with only the three island clusters set."""
    m = np.zeros((NX, NY, NZ), dtype=bool)
    m[2, 2, 20] = True                           # A: 1×1×1
    m[36, 2:4, 10:13] = True                     # B: 1×2×3
    m[2:4, 36:38, 5:9] = True                    # C: 2×2×4
    return m


# ---------------------------------------------------------------------------
# Helper: build solver for the column-in-void geometry
# ---------------------------------------------------------------------------

def _make_column_problem(preconditioner="jacobi", verbose=False,
                         extra_solid=None):
    """
    Build and return a configured VoxelFEMSolverNonlinear for the column test.

    All voxels are set to 255 so solid elements have E = E_FIBRE.
    The solid_mask restricts load-bearing to the 28×28 column only.
    If *extra_solid* (bool array same shape as voxels) is given, those voxels
    are added to the solid_mask (used to inject floating island clusters).

    BCs: z = 0 face fixed, z = NZ face prescribed; 3-2-1 anchoring on solid
    column nodes to suppress transverse rigid-body modes.
    """
    from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
    from geometry import build_node_index

    voxels = np.full((NX, NY, NZ), 255, dtype=np.uint8)

    solid_mask = np.zeros((NX, NY, NZ), dtype=bool)
    solid_mask[COL_LO:COL_HI, COL_LO:COL_HI, :] = True
    if extra_solid is not None:
        solid_mask |= extra_solid

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


# ===========================================================================
# Part B — floating island cluster tests
# ===========================================================================

def _pcg_iter_from_output(output):
    """Parse 'PCG converged at iter N' from captured stdout; return N or None."""
    m = re.search(r"PCG converged at iter (\d+)", output)
    return int(m.group(1)) if m else None


def test_islands_detected():
    """
    _check_floating_islands warns about all three clusters.

    Each cluster is isolated in the void: none touch the column, each other,
    or the z = 0 / z = NZ boundary faces that carry Dirichlet BCs.
    The warning must list exactly 3 islands with a total of 23 elements
    (sizes 16 + 6 + 1).
    """
    from geometry import build_node_index

    solver = _make_column_problem(
        preconditioner="jacobi", verbose=False,
        extra_solid=_island_mask(),
    )

    node_idx     = build_node_index(NX, NY, NZ)
    bottom_nodes = node_idx[:, :, 0].ravel()
    bc_z_dofs    = (bottom_nodes * 3 + 2).astype(np.int32)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver._check_floating_islands(bc_z_dofs)
    output = buf.getvalue()

    print("Island check output:", output.strip())

    assert "WARNING" in output, "Expected a floating-island WARNING but got none"

    # Extract island count from "N floating solid island(s)"
    m_count = re.search(r"(\d+) floating solid island", output)
    assert m_count is not None, "Could not parse island count from warning"
    n_islands = int(m_count.group(1))
    assert n_islands == 3, f"Expected 3 floating islands, got {n_islands}"

    # Extract total element count from "total N elements"
    m_total = re.search(r"total (\d+) elements", output)
    assert m_total is not None, "Could not parse total element count from warning"
    n_total = int(m_total.group(1))
    assert n_total == _ISLAND_TOTAL, (
        f"Expected {_ISLAND_TOTAL} total island elements, got {n_total}"
    )

    print(f"✓ {n_islands} floating islands detected, {n_total} total elements")


def test_islands_no_modulus_effect():
    """
    Adding floating islands must not change E_eff.

    Islands have no load path to any BC.  Their DOFs carry zero internal force
    and zero residual throughout the solve, so they cannot contribute to the
    reaction force on the top face.
    """
    from kernels_cupy_nl import compute_f_int_cupy
    from geometry import build_node_index

    def _E_eff(solver):
        u = solver.solve()
        u_gpu = cp.asarray(u.astype(np.float32))
        f_int = cp.asnumpy(
            compute_f_int_cupy(
                u_gpu, solver.E0_elem_gpu, solver.B_gauss_gpu,
                solver.w_gauss_gpu, NU, solver.constitutive,
                solver.Ny, solver.Nz, solver.Ndof,
                solid_mask_gpu=solver.solid_elem_gpu,
            )
        )
        node_idx   = build_node_index(NX, NY, NZ)
        top_z_dofs = node_idx[:, :, -1].ravel() * 3 + 2
        return float(np.sum(f_int[top_z_dofs])) / (SOLID_AREA * STRAIN)

    E_column_only  = _E_eff(_make_column_problem(verbose=False))
    E_with_islands = _E_eff(_make_column_problem(verbose=False,
                                                  extra_solid=_island_mask()))

    rel_diff = abs(E_with_islands - E_column_only) / E_FIBRE
    print(f"E_eff without islands: {E_column_only:.6f}")
    print(f"E_eff with    islands: {E_with_islands:.6f}")
    print(f"Relative difference:   {rel_diff:.2e}")

    assert rel_diff < 1e-4, (
        f"Floating islands changed E_eff by {rel_diff:.2e} "
        f"(expected < 1e-4, got E_column={E_column_only:.6f}, "
        f"E_islands={E_with_islands:.6f})"
    )
    assert abs(E_with_islands - E_FIBRE) / E_FIBRE < 0.05, (
        f"E_eff with islands = {E_with_islands:.6f} deviates from E_fibre by > 5 %"
    )
    print(f"✓ E_eff unchanged: {E_with_islands:.6f}  (Δ = {rel_diff:.1e})")


def test_island_dofs_stay_zero():
    """
    DOFs of floating island nodes must be exactly zero after the solve.

    Islands are completely isolated from the column subspace (no shared DOFs,
    no shared elements).  Starting from u = 0 with zero external force and zero
    internal force (u_island = 0 → ε = 0 → σ = 0), the residual for island
    DOFs is zero throughout PCG, so the solution stays zero.

    Verified for all three clusters (A, B, C).
    """
    from geometry import build_node_index

    solver = _make_column_problem(
        preconditioner="jacobi", verbose=False,
        extra_solid=_island_mask(),
    )
    u = solver.solve()

    node_idx = build_node_index(NX, NY, NZ)

    def _island_dof_max(x_slice, y_slice, z_slice, label):
        """Max |u| over all DOFs of nodes in the given index range."""
        nodes = node_idx[x_slice, y_slice, z_slice].ravel()
        dofs  = np.concatenate([nodes * 3 + d for d in range(3)])
        val   = float(np.max(np.abs(u[dofs])))
        print(f"  Cluster {label}: max|u| = {val:.2e}  ({len(nodes)} nodes)")
        return val

    # Element (2,2,20) → nodes span x∈[2,4], y∈[2,4], z∈[20,22]
    max_A = _island_dof_max(slice(2, 4), slice(2, 4), slice(20, 22), "A (1×1×1)")
    # Elements ix=36, iy∈[2,4), iz∈[10,13) → nodes: x∈[36,38], y∈[2,5], z∈[10,14]
    max_B = _island_dof_max(slice(36, 38), slice(2, 5), slice(10, 14), "B (1×2×3)")
    # Elements ix∈[2,4), iy∈[36,38), iz∈[5,9) → nodes: x∈[2,5], y∈[36,39], z∈[5,10]
    max_C = _island_dof_max(slice(2, 5), slice(36, 39), slice(5, 10), "C (2×2×4)")

    for name, val in [("A", max_A), ("B", max_B), ("C", max_C)]:
        assert val < 1e-6, (
            f"Cluster {name} DOFs are non-zero: max|u| = {val:.3e}; "
            "islands should carry zero displacement"
        )
    print("✓ All island DOFs are zero")


def test_islands_convergence_monitoring():
    """
    Floating islands must not stall or slow PCG convergence.

    Because island DOFs start at zero, have zero RHS, and are completely
    isolated from the column subspace, the Krylov basis built by PCG is
    identical to the island-free case.  The PCG iteration count should be
    exactly the same (or within 1 due to floating-point rounding order).

    The test prints both iteration counts for visibility.
    """
    def _solve_and_capture(extra_solid=None):
        solver = _make_column_problem(
            preconditioner="jacobi", verbose=False,
            extra_solid=extra_solid,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            u = solver.solve()
        return u, buf.getvalue()

    u_clean,   out_clean   = _solve_and_capture(extra_solid=None)
    u_islands, out_islands = _solve_and_capture(extra_solid=_island_mask())

    iters_clean   = _pcg_iter_from_output(out_clean)
    iters_islands = _pcg_iter_from_output(out_islands)

    print(f"PCG iters without islands: {iters_clean}")
    print(f"PCG iters with    islands: {iters_islands}")

    assert iters_clean   is not None, "PCG did not converge for column-only case"
    assert iters_islands is not None, "PCG did not converge with floating islands"

    # Iteration count must be identical (islands are fully decoupled)
    assert abs(iters_islands - iters_clean) <= 1, (
        f"Floating islands changed PCG iter count: "
        f"{iters_clean} → {iters_islands} (expected ≤ 1 difference)"
    )

    # Both solutions give the same displacement field on the column
    rel_u_diff = (np.max(np.abs(u_islands - u_clean))
                  / (np.max(np.abs(u_clean)) + 1e-30))
    print(f"Max |u_islands − u_clean| / max|u_clean| = {rel_u_diff:.2e}")
    assert rel_u_diff < 1e-5, (
        f"Displacement fields differ: rel diff = {rel_u_diff:.2e}"
    )

    print(f"✓ PCG iters: {iters_islands} (same as clean, max diff 1)")
    print(f"✓ Displacement field unchanged: rel diff = {rel_u_diff:.2e}")
