"""
Tests and benchmarks for nonlinear CuPy kernels (kernels_cupy_nl.py).
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

def _make_linear_reference(Nx, Ny, Nz, nu=0.3, E_max=1.0, E_void=1e-6):
    """
    Build the linear-FEM reference arrays (E_elem, conn_dof, K_ref) from the
    geometry primitives, used only for matvec_Ku comparisons.
    """
    from types import SimpleNamespace
    from geometry import build_node_index, element_node_ids, build_K_ref

    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)
    E_scale = (E_max - E_void) / 255.0
    E_elem = voxels.ravel().astype(np.float32) * E_scale + E_void

    node_idx = build_node_index(Nx, Ny, Nz)
    conn_nodes = element_node_ids(node_idx, Nx, Ny, Nz)        # (Ne, 8)
    conn_dof = (conn_nodes[:, :, np.newaxis] * 3
                + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)

    return SimpleNamespace(E_elem=E_elem, conn_dof=conn_dof, K_ref=build_K_ref(nu))


def _make_nl_kernels(Nx, Ny, Nz, nu=0.3, E_max=1.0, E_void=1e-6):
    """Return (E0_elem_gpu, B_gauss_gpu, w_gauss_gpu, Ne, Ndof) for nonlinear tests."""
    from kernels_cupy_nl import build_B_gauss
    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)
    E_scale = (E_max - E_void) / 255.0
    E0_cpu = voxels.ravel().astype(np.float32) * E_scale + E_void
    E0_gpu = cp.asarray(E0_cpu)

    B_cpu, w_cpu = build_B_gauss()
    B_gpu = cp.asarray(B_cpu.astype(np.float32))
    w_gpu = cp.asarray(w_cpu.astype(np.float32))

    Ne = Nx * Ny * Nz
    Nn = (Nx + 1) * (Ny + 1) * (Nz + 1)
    Ndof = 3 * Nn
    return E0_gpu, B_gpu, w_gpu, Ne, Ndof


# ---------------------------------------------------------------------------
# Test 1: B_gauss consistency with K_ref
# ---------------------------------------------------------------------------

def test_b_gauss_consistency():
    """
    Verify: Σ_gp w[gp] * B[gp].T @ D @ B[gp]  ==  K_ref  (within 1e-10).

    This checks that the nonlinear kernel precomputation reproduces the
    reference element stiffness from the linear solver.
    """
    from kernels_cupy_nl import build_B_gauss
    from geometry import build_K_ref

    nu = 0.3
    K_ref = build_K_ref(nu)  # (24, 24) float64

    # Material matrix D for E=1
    c1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c3 = 1.0 / (2.0 * (1.0 + nu))
    D = np.array([
        [c1, c2, c2,  0,  0,  0],
        [c2, c1, c2,  0,  0,  0],
        [c2, c2, c1,  0,  0,  0],
        [ 0,  0,  0, c3,  0,  0],
        [ 0,  0,  0,  0, c3,  0],
        [ 0,  0,  0,  0,  0, c3],
    ])

    B_gauss, w_gauss = build_B_gauss()

    K_assembled = np.zeros((24, 24))
    for gp in range(8):
        K_assembled += w_gauss[gp] * (B_gauss[gp].T @ D @ B_gauss[gp])

    err = np.max(np.abs(K_assembled - K_ref))
    assert err < 1e-10, f"B_gauss consistency failed: max error = {err:.3e}"
    print(f"✓ B_gauss consistency: max error = {err:.3e}")


# ---------------------------------------------------------------------------
# Test 2: f_int = 0 at zero displacement
# ---------------------------------------------------------------------------

def test_f_int_zero_displacement():
    """f_int must be exactly zero when u = 0 (no pre-stress)."""
    from kernels_cupy_nl import compute_f_int_cupy
    from constitutive import ConstitutiveModel

    Nx = Ny = Nz = 8
    E0_gpu, B_gpu, w_gpu, Ne, Ndof = _make_nl_kernels(Nx, Ny, Nz)
    mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=10.0)

    u_gpu = cp.zeros(Ndof, dtype=cp.float32)
    f_int = compute_f_int_cupy(u_gpu, E0_gpu, B_gpu, w_gpu,
                                0.3, mat, Ny, Nz, Ndof, chunk_size=50_000)
    max_val = float(cp.max(cp.abs(f_int)).get())
    assert max_val < 1e-10, f"f_int non-zero at u=0: max = {max_val:.3e}"
    print(f"✓ f_int zero displacement: max = {max_val:.3e}")


# ---------------------------------------------------------------------------
# Test 3: f_int linearity for linear constitutive model
# ---------------------------------------------------------------------------

def test_f_int_linear_consistency():
    """
    With a linear constitutive model, f_int(u) must match the linear solver's
    K·u within 1e-4 relative error.
    """
    from kernels_cupy_nl import compute_f_int_cupy
    from constitutive import ConstitutiveModel
    from geometry import matvec_Ku

    Nx = Ny = Nz = 8
    nu, E_max, E_void = 0.3, 1.0, 1e-6

    solver = _make_linear_reference(Nx, Ny, Nz, nu, E_max, E_void)
    E0_gpu, B_gpu, w_gpu, Ne, Ndof = _make_nl_kernels(Nx, Ny, Nz, nu, E_max, E_void)
    mat = ConstitutiveModel("linear", E_void=E_void, E_max=E_max)

    np.random.seed(7)
    u_test = np.random.randn(Ndof).astype(np.float32) * 0.001

    # Linear solver reference
    f_ref = matvec_Ku(u_test, solver.E_elem, solver.conn_dof, solver.K_ref, Ndof)

    # Nonlinear kernel (linear model)
    u_gpu = cp.asarray(u_test)
    f_nl = cp.asnumpy(compute_f_int_cupy(
        u_gpu, E0_gpu, B_gpu, w_gpu, nu, mat, Ny, Nz, Ndof, chunk_size=50_000
    ))

    rel_err = np.linalg.norm(f_ref - f_nl) / (np.linalg.norm(f_ref) + 1e-30)
    assert rel_err < 1e-4, f"f_int linear consistency: rel_err = {rel_err:.3e}"
    print(f"✓ f_int linear consistency: rel_err = {rel_err:.3e}")


# ---------------------------------------------------------------------------
# Test 4: matvec_tangent symmetry
# ---------------------------------------------------------------------------

def test_matvec_tangent_symmetry():
    """
    K_tangent(u) must be symmetric:  dot(K_t·x, y) ≈ dot(x, K_t·y)  (1e-4).
    """
    from kernels_cupy_nl import matvec_tangent_cupy
    from constitutive import ConstitutiveModel

    Nx = Ny = Nz = 6
    E0_gpu, B_gpu, w_gpu, Ne, Ndof = _make_nl_kernels(Nx, Ny, Nz)
    mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=5.0)

    np.random.seed(13)
    u = cp.asarray(np.random.randn(Ndof).astype(np.float32) * 0.001)
    x = cp.asarray(np.random.randn(Ndof).astype(np.float32))
    y = cp.asarray(np.random.randn(Ndof).astype(np.float32))

    Ktx = matvec_tangent_cupy(x, u, E0_gpu, B_gpu, w_gpu,
                               0.3, mat, Ny, Nz, Ndof, chunk_size=50_000)
    Kty = matvec_tangent_cupy(y, u, E0_gpu, B_gpu, w_gpu,
                               0.3, mat, Ny, Nz, Ndof, chunk_size=50_000)

    lhs = float(cp.dot(Ktx, y).get())
    rhs = float(cp.dot(x, Kty).get())

    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    assert rel_err < 1e-4, f"matvec_tangent symmetry failed: rel_err = {rel_err:.3e}"
    print(f"✓ matvec_tangent symmetry: rel_err = {rel_err:.3e}")


# ---------------------------------------------------------------------------
# Test 5: finite-difference consistency  (most important)
# ---------------------------------------------------------------------------

def test_matvec_tangent_fd_consistency():
    """
    Finite-difference check:
    (f_int(u + h·v) - f_int(u)) / h  ≈  K_tangent(u) · v  (within 5%).
    """
    from kernels_cupy_nl import compute_f_int_cupy, matvec_tangent_cupy
    from constitutive import ConstitutiveModel

    Nx = Ny = Nz = 6
    E0_gpu, B_gpu, w_gpu, Ne, Ndof = _make_nl_kernels(Nx, Ny, Nz)
    mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=5.0)

    np.random.seed(17)
    u_np = np.random.randn(Ndof).astype(np.float32) * 0.001
    v_np = np.random.randn(Ndof).astype(np.float32)
    u_norm = float(np.linalg.norm(u_np))
    v_norm = float(np.linalg.norm(v_np))
    h = 1e-5 * (u_norm + 1e-10) / (v_norm + 1e-10)

    u_gpu = cp.asarray(u_np)
    v_gpu = cp.asarray(v_np)
    uph_gpu = cp.asarray(u_np + h * v_np)

    f0 = compute_f_int_cupy(u_gpu, E0_gpu, B_gpu, w_gpu,
                             0.3, mat, Ny, Nz, Ndof, chunk_size=50_000)
    fh = compute_f_int_cupy(uph_gpu, E0_gpu, B_gpu, w_gpu,
                             0.3, mat, Ny, Nz, Ndof, chunk_size=50_000)
    fd = (fh - f0) / h  # finite-difference approximation

    Ktv = matvec_tangent_cupy(v_gpu, u_gpu, E0_gpu, B_gpu, w_gpu,
                               0.3, mat, Ny, Nz, Ndof, chunk_size=50_000)

    rel_err = float(
        cp.linalg.norm(fd - Ktv) / (cp.linalg.norm(Ktv) + 1e-30)
    )
    assert rel_err < 0.05, f"FD consistency failed: rel_err = {rel_err:.3e}"
    print(f"✓ FD consistency: rel_err = {rel_err:.3e}")


# ---------------------------------------------------------------------------
# Test 6: softening model reduces effective stiffness
# ---------------------------------------------------------------------------

def test_softening_stiffness_decrease():
    """
    With softening, ||K_tangent(u)|| decreases as ||u|| increases.
    """
    from kernels_cupy_nl import matvec_tangent_cupy
    from constitutive import ConstitutiveModel

    Nx = Ny = Nz = 4
    E0_gpu, B_gpu, w_gpu, Ne, Ndof = _make_nl_kernels(Nx, Ny, Nz)
    mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=20.0)

    np.random.seed(3)
    v = cp.asarray(np.random.randn(Ndof).astype(np.float32))

    norms = []
    for scale in [0.0, 0.01, 0.05, 0.1]:
        u = cp.asarray(np.random.randn(Ndof).astype(np.float32) * scale)
        Ktv = matvec_tangent_cupy(v, u, E0_gpu, B_gpu, w_gpu,
                                   0.3, mat, Ny, Nz, Ndof, chunk_size=10_000)
        norms.append(float(cp.linalg.norm(Ktv).get()))

    # Stiffness should not increase as we go from scale=0 to scale=0.1
    assert norms[0] >= norms[-1], \
        f"Softening: stiffness increased with strain: {norms}"
    print(f"✓ Softening stiffness decrease: norms = {[f'{n:.4f}' for n in norms]}")


# ---------------------------------------------------------------------------
# Test 7: hardening model increases effective stiffness
# ---------------------------------------------------------------------------

def test_hardening_stiffness_increase():
    """
    With hardening, ||K_tangent(u)|| increases as ||u|| increases.
    """
    from kernels_cupy_nl import matvec_tangent_cupy
    from constitutive import ConstitutiveModel

    Nx = Ny = Nz = 4
    E0_gpu, B_gpu, w_gpu, Ne, Ndof = _make_nl_kernels(Nx, Ny, Nz)
    mat = ConstitutiveModel("hardening", E_void=1e-6, E_max=1.0, beta=200.0, n=0.5)

    np.random.seed(5)
    v = cp.asarray(np.random.randn(Ndof).astype(np.float32))

    norms = []
    for scale in [0.0, 0.001, 0.005, 0.01]:
        u = cp.asarray(np.random.randn(Ndof).astype(np.float32) * scale)
        Ktv = matvec_tangent_cupy(v, u, E0_gpu, B_gpu, w_gpu,
                                   0.3, mat, Ny, Nz, Ndof, chunk_size=10_000)
        norms.append(float(cp.linalg.norm(Ktv).get()))

    # Stiffness should not decrease as we go from scale=0 to scale=0.01
    assert norms[-1] >= norms[0], \
        f"Hardening: stiffness decreased with strain: {norms}"
    print(f"✓ Hardening stiffness increase: norms = {[f'{n:.4f}' for n in norms]}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [10, 20])
def test_benchmark_kernels(N):
    """
    Benchmark f_int and matvec_tangent. Prints timing table.
    """
    from kernels_cupy_nl import compute_f_int_cupy, matvec_tangent_cupy
    from constitutive import ConstitutiveModel

    Nx = Ny = Nz = N
    E0_gpu, B_gpu, w_gpu, Ne, Ndof = _make_nl_kernels(Nx, Ny, Nz)
    mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=10.0)

    u_gpu = cp.asarray(np.random.randn(Ndof).astype(np.float32) * 0.001)
    v_gpu = cp.asarray(np.random.randn(Ndof).astype(np.float32))

    # Warm up
    compute_f_int_cupy(u_gpu, E0_gpu, B_gpu, w_gpu, 0.3, mat, Ny, Nz, Ndof)
    matvec_tangent_cupy(v_gpu, u_gpu, E0_gpu, B_gpu, w_gpu, 0.3, mat, Ny, Nz, Ndof)
    cp.cuda.stream.get_current_stream().synchronize()

    n_runs = 3
    t0 = time.perf_counter()
    for _ in range(n_runs):
        compute_f_int_cupy(u_gpu, E0_gpu, B_gpu, w_gpu, 0.3, mat, Ny, Nz, Ndof)
    cp.cuda.stream.get_current_stream().synchronize()
    t_fint = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        matvec_tangent_cupy(v_gpu, u_gpu, E0_gpu, B_gpu, w_gpu, 0.3, mat, Ny, Nz, Ndof)
    cp.cuda.stream.get_current_stream().synchronize()
    t_mv = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\nBenchmark: {N}³ grid ({Ne:,} elements)")
    print(f"  f_int:         {t_fint:.1f} ms")
    print(f"  matvec_tangent:{t_mv:.1f} ms")

    assert True  # benchmark — always passes
