"""
Tests for the GeometricMultigrid preconditioner (multigrid.py).

Pure NumPy — multigrid runs entirely on the CPU, so these always run (no GPU
guard). They lock in the three correctness properties the V-cycle must satisfy:

  1. Restriction is the transpose of prolongation, R = P^T (adjoint property).
  2. The V-cycle is a convergent stationary iteration for A e = r.
  3. As a PCG preconditioner it converges and agrees with a Jacobi-PCG solve,
     in far fewer iterations.

See CLAUDE.md "Known Issues" for the history of the bugs these guard against.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry import (
    build_node_index, element_node_ids, build_K_ref, matvec_Ku,
    build_diagonal, pcg,
)
from multigrid import GeometricMultigrid

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _build_problem(N=10, nu=0.3, E_max=1.0, E_void=1e-6):
    """Uniform-voxel cube under uniaxial displacement, with rigid modes anchored."""
    voxels = np.full((N, N, N), 255, dtype=np.uint8)
    Ndof = 3 * (N + 1) ** 3
    K_ref = build_K_ref(nu)
    node_idx = build_node_index(N, N, N)
    conn_nodes = element_node_ids(node_idx, N, N, N)
    conn_dof = (conn_nodes[:, :, None] * 3
                + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)
    E_scale = (E_max - E_void) / 255.0
    E_elem = voxels.ravel().astype(np.float32) * E_scale + E_void

    bc_mask = np.zeros(Ndof, dtype=bool)
    bc_val = np.zeros(Ndof, dtype=np.float32)
    delta = 1e-3 * N
    low = node_idx[:, :, 0].ravel()
    high = node_idx[:, :, -1].ravel()
    bc_mask[low * 3 + 2] = True
    bc_mask[high * 3 + 2] = True
    bc_val[high * 3 + 2] = delta
    o = int(node_idx[0, 0, 0])
    bc_mask[o * 3 + 0] = True
    bc_mask[o * 3 + 1] = True
    bc_mask[int(node_idx[-1, 0, 0]) * 3 + 1] = True

    def matvec(u):
        return matvec_Ku(u, E_elem, conn_dof, K_ref, Ndof)

    u_bc = np.zeros(Ndof, dtype=np.float32)
    u_bc[bc_mask] = bc_val[bc_mask]
    b = -matvec(u_bc)
    b[bc_mask] = bc_val[bc_mask]

    diag = build_diagonal(E_elem, conn_dof, K_ref, Ndof)
    diag = np.where(diag < 1e-12, 1.0, diag)
    jac_inv = (1.0 / diag).astype(np.float32)

    return dict(N=N, nu=nu, E_max=E_max, E_void=E_void, voxels=voxels,
                Ndof=Ndof, matvec=matvec, b=b, bc_mask=bc_mask, jac_inv=jac_inv)


def _gmg(prob):
    return GeometricMultigrid(prob["voxels"], prob["nu"], prob["E_max"],
                              prob["E_void"], n_pre=2, n_post=2,
                              bc_mask=prob["bc_mask"])


def test_restriction_is_prolongation_transpose():
    """Adjoint property ⟨r, P e⟩ = ⟨R r, e⟩ for every fine/coarse level pair."""
    prob = _build_problem(N=10)
    gmg = _gmg(prob)
    assert len(gmg.levels) >= 2
    rng = np.random.default_rng(0)
    for lf in range(len(gmg.levels) - 1):
        fine, coarse = gmg.levels[lf], gmg.levels[lf + 1]
        r_fine = rng.standard_normal(fine.Ndof).astype(np.float32)
        e_coarse = rng.standard_normal(coarse.Ndof).astype(np.float32)
        lhs = float(np.dot(r_fine, gmg._prolongate(e_coarse, coarse, fine)))
        rhs = float(np.dot(gmg._restrict(r_fine, fine, coarse), e_coarse))
        assert abs(lhs - rhs) <= 1e-5 * (abs(lhs) + abs(rhs) + 1e-30)


def test_vcycle_is_convergent_iteration():
    """Iterative refinement with the V-cycle must reduce the residual of A e = r."""
    prob = _build_problem(N=10)
    gmg = _gmg(prob)
    matvec, bc_mask = prob["matvec"], prob["bc_mask"]
    free = ~bc_mask

    def Afree(v):
        w = matvec(v)
        w[bc_mask] = v[bc_mask]
        return w

    rng = np.random.default_rng(1)
    r = rng.standard_normal(prob["Ndof"]).astype(np.float32)
    r[bc_mask] = 0.0
    r0 = np.linalg.norm(r[free])

    e = np.zeros_like(r)
    res_hist = []
    for _ in range(8):
        e = e + gmg.vcycle(r - Afree(e))
        res_hist.append(np.linalg.norm((r - Afree(e))[free]) / r0)

    # Monotone decrease (the over-correction bug made this grow) and a clear
    # net reduction after several cycles.
    assert all(res_hist[i + 1] < res_hist[i] for i in range(len(res_hist) - 1))
    assert res_hist[-1] < 0.5 * res_hist[0]


def test_pcg_multigrid_converges_and_matches_jacobi():
    """Multigrid-PCG converges, agrees with Jacobi-PCG, and uses fewer iters."""
    prob = _build_problem(N=10)
    gmg = _gmg(prob)

    x_j, it_j = _pcg(prob, prob["jac_inv"])
    x_m, it_m = _pcg(prob, gmg.apply)

    # Both reach the same solution (cross-check against geometry.pcg too).
    x_ref = pcg(prob["matvec"], prob["b"], prob["jac_inv"], prob["bc_mask"],
                max_iter=500, tol=1e-8, verbose=False)
    assert np.linalg.norm(x_j - x_ref) / (np.linalg.norm(x_ref) + 1e-30) < 1e-3
    assert np.linalg.norm(x_j - x_m) / (np.linalg.norm(x_j) + 1e-30) < 1e-3
    assert it_m < it_j        # multigrid should need fewer iterations


def _pcg(prob, precond, tol=1e-8, max_iter=500):
    """BC-aware PCG returning (solution, iterations to reach tol)."""
    matvec, bc_mask, b = prob["matvec"], prob["bc_mask"], prob["b"]
    free = ~bc_mask

    def Afree(v):
        w = matvec(v)
        w[bc_mask] = v[bc_mask]
        return w

    def M(r):
        z = precond(r) if callable(precond) else precond * r
        z[bc_mask] = 0.0
        return z

    x = np.zeros_like(b)
    r = b - Afree(x)
    r[bc_mask] = 0.0
    z = M(r)
    p = z.copy()
    rz = float(np.dot(r, z))
    r0 = float(np.dot(b[free], b[free])) + 1e-30
    for it in range(1, max_iter + 1):
        Ap = Afree(p)
        alpha = rz / (float(np.dot(p, Ap)) + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        r[bc_mask] = 0.0
        if np.sqrt(float(np.dot(r[free], r[free])) / r0) < tol:
            return x, it
        z = M(r)
        rz_new = float(np.dot(r, z))
        p = z + (rz_new / (rz + 1e-30)) * p
        rz = rz_new
    return x, max_iter


# ---------------------------------------------------------------------------
# GPU backend: the CuPy-resident V-cycle must reproduce the NumPy reference.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy/GPU not available")
def test_gpu_matvec_Ku_matches_numpy_reference():
    """matvec_Ku_cupy (implicit connectivity) == geometry.matvec_Ku (conn_dof)."""
    from kernels_cupy_nl import matvec_Ku_cupy

    N, nu = 12, 0.3
    Ndof = 3 * (N + 1) ** 3
    K_ref = build_K_ref(nu)
    node_idx = build_node_index(N, N, N)
    conn_nodes = element_node_ids(node_idx, N, N, N)
    conn_dof = (conn_nodes[:, :, None] * 3
                + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)
    rng = np.random.default_rng(0)
    E_elem = rng.uniform(0.1, 1.0, N ** 3).astype(np.float32)
    u = rng.standard_normal(Ndof).astype(np.float32)

    f_cpu = matvec_Ku(u, E_elem, conn_dof, K_ref, Ndof)
    f_gpu = matvec_Ku_cupy(
        cp.asarray(u), cp.asarray(E_elem),
        cp.asarray(K_ref.astype(np.float32)), N, N, Ndof,
    ).get()

    assert np.linalg.norm(f_gpu - f_cpu) <= 1e-4 * (np.linalg.norm(f_cpu) + 1e-30)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy/GPU not available")
def test_gpu_vcycle_matches_cpu():
    """A full GPU V-cycle apply() agrees with the NumPy reference apply()."""
    prob = _build_problem(N=10)
    gmg_cpu = _gmg(prob)
    gmg_gpu = GeometricMultigrid(
        prob["voxels"], prob["nu"], prob["E_max"], prob["E_void"],
        n_pre=2, n_post=2, bc_mask=prob["bc_mask"], use_gpu=True,
    )

    rng = np.random.default_rng(2)
    r = rng.standard_normal(prob["Ndof"]).astype(np.float32)
    r[prob["bc_mask"]] = 0.0

    z_cpu = gmg_cpu.apply(r)
    z_gpu = gmg_gpu.apply(cp.asarray(r)).get()

    # GPU path stores E_elem as float16 (eps_rel ≈ 1e-3) while CPU uses float32,
    # so the V-cycle outputs differ by O(eps_float16). Tolerance reflects this.
    assert np.linalg.norm(z_gpu - z_cpu) <= 2e-3 * (np.linalg.norm(z_cpu) + 1e-30)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy/GPU not available")
def test_gpu_pcg_matches_jacobi():
    """GPU-multigrid-PCG converges to the same solution as Jacobi-PCG."""
    prob = _build_problem(N=10)
    gmg_gpu = GeometricMultigrid(
        prob["voxels"], prob["nu"], prob["E_max"], prob["E_void"],
        n_pre=2, n_post=2, bc_mask=prob["bc_mask"], use_gpu=True,
    )

    # Wrap the GPU apply so the NumPy reference PCG can call it transparently.
    def gpu_apply(r):
        return gmg_gpu.apply(cp.asarray(r)).get()

    x_ref = pcg(prob["matvec"], prob["b"], prob["jac_inv"], prob["bc_mask"],
                max_iter=500, tol=1e-8, verbose=False)
    x_m, it_m = _pcg(prob, gpu_apply)
    x_j, it_j = _pcg(prob, prob["jac_inv"])

    assert np.linalg.norm(x_m - x_ref) / (np.linalg.norm(x_ref) + 1e-30) < 1e-3
    assert it_m < it_j
