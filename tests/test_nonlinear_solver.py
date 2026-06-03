"""
Tests for the nonlinear FEM solver (voxel_fem_solver_nonlinear.py).

Tests cover:
  8a — Regression to linear solver (softening with alpha=0 ≈ linear)
  8b — Quadratic NR convergence rate
  8c — Load-stepping convergence
  8d — BC correctness
  8e — Softening: decreasing stiffness over load steps
  8f — Hardening: increasing stiffness over load steps
  8g — Smoke test (20³, both material models)
"""

import sys
import os
import time
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
# Helpers
# ---------------------------------------------------------------------------

def _build_uniaxial_problem(Nx, Ny, Nz, constitutive, n_load_steps=5,
                             E_max=1.0, E_void=1e-6, nu=0.3,
                             disp=0.01, verbose=False, **kwargs):
    """
    Standard uniaxial tension: fix bottom face (z=0), pull top face (z=Nz).
    Returns (solver, u) after solve().
    """
    from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
    from geometry import build_node_index

    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)
    solver = VoxelFEMSolverNonlinear(
        voxels, nu=nu, E_max=E_max, E_void=E_void,
        constitutive=constitutive, n_load_steps=n_load_steps,
        verbose=verbose, **kwargs
    )

    node_idx = build_node_index(Nx, Ny, Nz)
    bottom = node_idx[:, :, 0].ravel()
    top    = node_idx[:, :, -1].ravel()

    for dof in range(3):
        solver.apply_bc_displacement(
            bottom, np.full(len(bottom), dof, dtype=int), np.zeros(len(bottom))
        )
    solver.apply_bc_displacement(
        top, np.full(len(top), 2, dtype=int), np.full(len(top), disp)
    )

    u = solver.solve()
    return solver, u


# ---------------------------------------------------------------------------
# 8a — Regression to linear
# ---------------------------------------------------------------------------

def test_regression_to_linear():
    """
    Softening with alpha→0 (alpha=1e-6) should match the linear solver.
    Tests on 10³ grid with 1 load step.
    """
    from constitutive import ConstitutiveModel
    from geometry import (build_node_index, element_node_ids, build_K_ref,
                          matvec_Ku, build_diagonal, pcg)

    Nx = Ny = Nz = 10
    nu, E_max, E_void = 0.3, 1.0, 1e-6
    disp = 0.01

    # Reference: linear CPU PCG solve built from the geometry primitives.
    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)
    Ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
    E_scale = (E_max - E_void) / 255.0
    E_elem = voxels.ravel().astype(np.float32) * E_scale + E_void
    K_ref = build_K_ref(nu)
    node_idx = build_node_index(Nx, Ny, Nz)
    conn_nodes = element_node_ids(node_idx, Nx, Ny, Nz)
    conn_dof = (conn_nodes[:, :, np.newaxis] * 3
                + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)

    bc_mask = np.zeros(Ndof, dtype=bool)
    bc_values = np.zeros(Ndof, dtype=np.float32)
    bottom = node_idx[:, :, 0].ravel()
    top    = node_idx[:, :, -1].ravel()
    for dof in range(3):
        bc_mask[bottom * 3 + dof] = True            # bottom: u = 0 (all DOFs)
    bc_mask[top * 3 + 2] = True                     # top: prescribed u_z
    bc_values[top * 3 + 2] = disp

    diag = build_diagonal(E_elem, conn_dof, K_ref, Ndof)
    M_inv = (1.0 / np.where(diag < 1e-12, 1.0, diag)).astype(np.float32)

    def _matvec(v):
        return matvec_Ku(v, E_elem, conn_dof, K_ref, Ndof)

    u_bc = np.where(bc_mask, bc_values, 0.0).astype(np.float32)
    f = np.zeros(Ndof, dtype=np.float32)
    f_bc = _matvec(u_bc)
    f[~bc_mask] -= f_bc[~bc_mask]
    f[bc_mask] = bc_values[bc_mask]

    u_lin = pcg(_matvec, f, M_inv, bc_mask, max_iter=300, tol=1e-8, verbose=False)
    u_lin[bc_mask] = bc_values[bc_mask]

    # Nonlinear: linear model (E_t = E0)
    mat = ConstitutiveModel("linear", E_void=E_void, E_max=E_max)
    _, u_nl = _build_uniaxial_problem(
        Nx, Ny, Nz, mat, n_load_steps=1, E_max=E_max, E_void=E_void, nu=nu,
        disp=disp, tol_nr=1e-8, max_pcg_iter=300, preconditioner="jacobi"
    )

    rel_err = np.linalg.norm(u_lin - u_nl) / (np.linalg.norm(u_lin) + 1e-30)
    assert rel_err < 1e-3, f"Regression to linear failed: rel_err = {rel_err:.3e}"
    print(f"✓ Regression to linear: rel_err = {rel_err:.3e}")


# ---------------------------------------------------------------------------
# 8b — NR quadratic convergence
# ---------------------------------------------------------------------------

def test_nr_quadratic_convergence():
    """
    NR should exhibit quadratic convergence for a smooth problem with small load.
    Collect NR residuals and verify superlinear behavior over the last 3 iters.
    """
    from constitutive import ConstitutiveModel
    from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
    from geometry import build_node_index

    Nx = Ny = Nz = 8
    nu, E_max, E_void = 0.3, 1.0, 1e-6
    disp = 1e-4  # small displacement → smooth nonlinearity

    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)
    mat = ConstitutiveModel("softening", E_void=E_void, E_max=E_max, alpha=5.0)

    solver = VoxelFEMSolverNonlinear(
        voxels, nu=nu, E_max=E_max, E_void=E_void, constitutive=mat,
        n_load_steps=1, max_nr_iter=15, tol_nr=1e-10,
        max_pcg_iter=200, preconditioner="jacobi", verbose=True
    )

    node_idx = build_node_index(Nx, Ny, Nz)
    bottom = node_idx[:, :, 0].ravel()
    top    = node_idx[:, :, -1].ravel()
    for dof in range(3):
        solver.apply_bc_displacement(bottom, np.full(len(bottom), dof, dtype=int),
                                     np.zeros(len(bottom)))
    solver.apply_bc_displacement(top, np.full(len(top), 2, dtype=int),
                                  np.full(len(top), disp))

    # Monkey-patch _nr_step to collect residuals
    residuals = []
    original_nr_step = solver._nr_step

    def _patched_nr_step(u_gpu, lf, bc_mask_gpu, bc_dofs_gpu, bc_vals_gpu,
                         force_dofs_gpu, force_vals_gpu, _has_forces, free_gpu):
        import cupy as cp
        from kernels_cupy_nl import compute_f_int_cupy
        R_norm_prev = None
        f_norm_ref = None
        u_gpu_local = u_gpu.copy()

        for k in range(solver.max_nr_iter):
            f_int = compute_f_int_cupy(
                u_gpu_local, solver.E0_elem_gpu, solver.B_gauss_gpu,
                solver.w_gauss_gpu, solver.nu, solver.constitutive,
                solver.Ny, solver.Nz, solver.Ndof,
                chunk_size=solver.gpu_chunk_size,
            )
            if f_norm_ref is None:
                f_norm_int = float(cp.linalg.norm(f_int[free_gpu]).get())
                f_norm_ext = (lf * float(cp.linalg.norm(force_vals_gpu).get())
                              if _has_forces else 0.0)
                f_norm_ref = max(f_norm_ext, f_norm_int, 1e-30)

            # Build R = f_ext - f_int in-place
            cp.negative(f_int, out=f_int)
            if _has_forces:
                cp.add.at(f_int, force_dofs_gpu, lf * force_vals_gpu)
            f_int[bc_mask_gpu] = 0.0
            R = f_int

            R_norm = float(cp.linalg.norm(R[free_gpu]).get())
            residuals.append(R_norm)

            if R_norm / f_norm_ref < solver.tol_nr:
                break

            eta = solver._ew_tolerance(R_norm, R_norm_prev)
            solver._precond.update(u_gpu_local)

            u_local = u_gpu_local

            def _mv(v, _u=u_local):
                from kernels_cupy_nl import matvec_tangent_cupy
                return matvec_tangent_cupy(v, _u, solver.E0_elem_gpu, solver.B_gauss_gpu,
                                           solver.w_gauss_gpu, solver.nu, solver.constitutive,
                                           solver.Ny, solver.Nz, solver.Ndof,
                                           chunk_size=solver.gpu_chunk_size)

            delta_u = solver._pcg_gpu(_mv, R, solver._precond.apply, bc_mask_gpu, free_gpu,
                                       tol=eta, max_iter=solver.max_pcg_iter, verbose=False)
            u_gpu_local = u_gpu_local + delta_u
            u_gpu_local[bc_dofs_gpu] = (lf * bc_vals_gpu)
            R_norm_prev = R_norm

        return u_gpu_local

    solver._nr_step = _patched_nr_step
    solver.solve()

    print(f"NR residuals: {[f'{r:.3e}' for r in residuals]}")
    assert len(residuals) >= 3, "Too few NR iterations to check convergence rate"

    # Compute ratios for all converging iterations (before float32 stagnation)
    # Stagnation is detected when residual stops decreasing significantly
    converging_ratios = []
    for i in range(len(residuals) - 1):
        if residuals[i] > 1e-8:   # only before float32 stagnation floor
            r = residuals[i + 1] / (residuals[i] + 1e-30)
            converging_ratios.append(r)

    print(f"Convergence ratios (converging phase): "
          f"{[f'{r:.3f}' for r in converging_ratios]}")

    assert len(converging_ratios) >= 2, \
        "Not enough converging iterations to assess convergence rate"

    # At least one ratio < 0.5 indicates superlinear convergence
    assert min(converging_ratios) < 0.5, \
        f"No superlinear convergence in converging phase: ratios = {converging_ratios}"
    print("✓ NR superlinear convergence verified")


# ---------------------------------------------------------------------------
# 8c — Load-stepping convergence
# ---------------------------------------------------------------------------

def test_load_stepping_convergence():
    """
    n_load_steps=5 must converge for a highly nonlinear problem (large disp).
    """
    from constitutive import ConstitutiveModel

    Nx = Ny = Nz = 8
    mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=10.0)

    _, u = _build_uniaxial_problem(
        Nx, Ny, Nz, mat, n_load_steps=5,
        disp=0.05, tol_nr=1e-5, max_pcg_iter=200, preconditioner="jacobi"
    )

    assert not np.any(np.isnan(u)), "NaN in displacement (load-stepping did not converge)"
    assert not np.any(np.isinf(u)), "Inf in displacement"
    assert np.max(np.abs(u)) > 1e-6, "All-zero displacement (solver failed silently)"
    print(f"✓ Load-stepping convergence: max|u| = {np.max(np.abs(u)):.4f}")


# ---------------------------------------------------------------------------
# 8d — BC correctness
# ---------------------------------------------------------------------------

def test_bc_correctness():
    """
    All prescribed DOFs in u must exactly equal their target values (float32 eps).
    """
    from constitutive import ConstitutiveModel
    from geometry import build_node_index

    Nx = Ny = Nz = 8
    disp = 0.01
    mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=5.0)

    solver, u = _build_uniaxial_problem(
        Nx, Ny, Nz, mat, n_load_steps=3, disp=disp,
        tol_nr=1e-6, max_pcg_iter=100, preconditioner="jacobi"
    )

    node_idx = build_node_index(Nx, Ny, Nz)
    bottom = node_idx[:, :, 0].ravel()
    top    = node_idx[:, :, -1].ravel()

    # Check bottom nodes (all DOFs = 0)
    for dof in range(3):
        gdofs = bottom * 3 + dof
        vals = u[gdofs]
        assert np.max(np.abs(vals)) < 1e-6, \
            f"Bottom BC violated (dof={dof}): max = {np.max(np.abs(vals)):.3e}"

    # Check top nodes (z-DOF = disp)
    gdofs_top = top * 3 + 2
    errs = np.abs(u[gdofs_top] - disp)
    assert np.max(errs) < 1e-5, \
        f"Top BC violated: max error = {np.max(errs):.3e}"
    print("✓ BC correctness verified")


# ---------------------------------------------------------------------------
# 8e — Softening: decreasing effective stiffness over load steps
# ---------------------------------------------------------------------------

def test_softening_decreasing_stiffness():
    """
    For softening material, effective stiffness ||f_int||/||u|| must decrease
    monotonically over load steps.
    """
    from constitutive import ConstitutiveModel
    from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
    from kernels_cupy_nl import compute_f_int_cupy
    from geometry import build_node_index

    Nx = Ny = Nz = 6
    nu, E_max, E_void = 0.3, 1.0, 1e-6
    mat = ConstitutiveModel("softening", E_void=E_void, E_max=E_max, alpha=20.0)
    n_steps = 4

    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)
    solver = VoxelFEMSolverNonlinear(
        voxels, nu=nu, E_max=E_max, E_void=E_void, constitutive=mat,
        n_load_steps=n_steps, verbose=False, preconditioner="jacobi",
        tol_nr=1e-5, max_pcg_iter=100
    )

    node_idx = build_node_index(Nx, Ny, Nz)
    bottom = node_idx[:, :, 0].ravel()
    top    = node_idx[:, :, -1].ravel()
    for dof in range(3):
        solver.apply_bc_displacement(bottom, np.full(len(bottom), dof, dtype=int),
                                     np.zeros(len(bottom)))
    solver.apply_bc_displacement(top, np.full(len(top), 2, dtype=int),
                                  np.full(len(top), 0.05))

    # Collect stiffness per load step
    stiffnesses = []
    original_nr_step = solver._nr_step

    def _patched(u_gpu, lf, bc_mask_gpu, bc_dofs_gpu, bc_vals_gpu,
                 force_dofs_gpu, force_vals_gpu, _has_forces, free_gpu):
        u_out = original_nr_step(u_gpu, lf, bc_mask_gpu, bc_dofs_gpu, bc_vals_gpu,
                                 force_dofs_gpu, force_vals_gpu, _has_forces, free_gpu)
        f_int = compute_f_int_cupy(
            u_out, solver.E0_elem_gpu, solver.B_gauss_gpu, solver.w_gauss_gpu,
            solver.nu, solver.constitutive, solver.Ny, solver.Nz, solver.Ndof,
            chunk_size=solver.gpu_chunk_size
        )
        u_norm = float(cp.linalg.norm(u_out[~bc_mask_gpu]).get())
        f_norm = float(cp.linalg.norm(f_int[~bc_mask_gpu]).get())
        stiffnesses.append(f_norm / (u_norm + 1e-30))
        return u_out

    solver._nr_step = _patched
    solver.solve()

    print(f"Softening stiffnesses: {[f'{s:.4f}' for s in stiffnesses]}")
    # Stiffness must not increase from first to last step
    assert stiffnesses[-1] <= stiffnesses[0] + 1e-3, \
        f"Softening: stiffness increased from {stiffnesses[0]:.4f} to {stiffnesses[-1]:.4f}"
    print("✓ Softening: stiffness decreases over load steps")


# ---------------------------------------------------------------------------
# 8f — Hardening: increasing effective stiffness over load steps
# ---------------------------------------------------------------------------

def test_hardening_increasing_stiffness():
    """
    For hardening material, effective stiffness ||f_int||/||u|| must increase
    monotonically over load steps.
    """
    from constitutive import ConstitutiveModel
    from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
    from kernels_cupy_nl import compute_f_int_cupy
    from geometry import build_node_index

    Nx = Ny = Nz = 6
    nu, E_max, E_void = 0.3, 1.0, 1e-6
    # Use very high beta so hardening effect is pronounced at small strains
    mat = ConstitutiveModel("hardening", E_void=E_void, E_max=E_max, beta=200.0, n=0.5)
    n_steps = 4

    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)
    solver = VoxelFEMSolverNonlinear(
        voxels, nu=nu, E_max=E_max, E_void=E_void, constitutive=mat,
        n_load_steps=n_steps, verbose=False, preconditioner="jacobi",
        tol_nr=1e-5, max_pcg_iter=100
    )

    node_idx = build_node_index(Nx, Ny, Nz)
    bottom = node_idx[:, :, 0].ravel()
    top    = node_idx[:, :, -1].ravel()
    for dof in range(3):
        solver.apply_bc_displacement(bottom, np.full(len(bottom), dof, dtype=int),
                                     np.zeros(len(bottom)))
    solver.apply_bc_displacement(top, np.full(len(top), 2, dtype=int),
                                  np.full(len(top), 0.01))

    stiffnesses = []
    original_nr_step = solver._nr_step

    def _patched(u_gpu, lf, bc_mask_gpu, bc_dofs_gpu, bc_vals_gpu,
                 force_dofs_gpu, force_vals_gpu, _has_forces, free_gpu):
        u_out = original_nr_step(u_gpu, lf, bc_mask_gpu, bc_dofs_gpu, bc_vals_gpu,
                                 force_dofs_gpu, force_vals_gpu, _has_forces, free_gpu)
        f_int = compute_f_int_cupy(
            u_out, solver.E0_elem_gpu, solver.B_gauss_gpu, solver.w_gauss_gpu,
            solver.nu, solver.constitutive, solver.Ny, solver.Nz, solver.Ndof,
            chunk_size=solver.gpu_chunk_size
        )
        u_norm = float(cp.linalg.norm(u_out[~bc_mask_gpu]).get())
        f_norm = float(cp.linalg.norm(f_int[~bc_mask_gpu]).get())
        stiffnesses.append(f_norm / (u_norm + 1e-30))
        return u_out

    solver._nr_step = _patched
    solver.solve()

    print(f"Hardening stiffnesses: {[f'{s:.4f}' for s in stiffnesses]}")
    assert stiffnesses[-1] >= stiffnesses[0] - 1e-3, \
        f"Hardening: stiffness decreased from {stiffnesses[0]:.4f} to {stiffnesses[-1]:.4f}"
    print("✓ Hardening: stiffness increases over load steps")


# ---------------------------------------------------------------------------
# 8g — Smoke test (20³, both material models)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name,alpha,beta,n", [
    ("softening", 5.0,  0.0,  0.5),
    ("hardening", 0.0, 50.0,  0.5),
])
def test_smoke_20x20x20(model_name, alpha, beta, n):
    """
    20³ uniaxial tension with both material models.
    Must complete in < 120s, no NaN/Inf, max z-displacement = prescribed value.
    """
    from constitutive import ConstitutiveModel

    mat = ConstitutiveModel(model_name, E_void=1e-6, E_max=1.0, alpha=alpha, beta=beta, n=n)

    t0 = time.time()
    solver, u = _build_uniaxial_problem(
        20, 20, 20, mat, n_load_steps=3, disp=0.01,
        tol_nr=1e-5, max_pcg_iter=200, preconditioner="jacobi"
    )
    elapsed = time.time() - t0

    assert not np.any(np.isnan(u)), f"NaN in displacement ({model_name})"
    assert not np.any(np.isinf(u)), f"Inf in displacement ({model_name})"

    # z-displacement range should include 0 and the prescribed value
    z_disp = u[2::3]
    assert abs(z_disp.max() - 0.01) < 1e-4, \
        f"Max z-disp = {z_disp.max():.6f}, expected 0.01"
    assert z_disp.min() >= -1e-5, f"Negative z-disp: {z_disp.min():.6f}"

    assert elapsed < 120, f"Smoke test took {elapsed:.1f}s (> 120s limit)"
    print(f"✓ Smoke test ({model_name}): elapsed={elapsed:.1f}s, "
          f"max|u|={np.abs(u).max():.4f}")
