"""
Unit tests for constitutive models (constitutive.py).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not available"
)


@pytest.fixture
def models():
    from constitutive import ConstitutiveModel
    return {
        "linear":    ConstitutiveModel("linear",    E_void=1e-6, E_max=1.0),
        "softening": ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=10.0),
        "hardening": ConstitutiveModel("hardening", E_void=1e-6, E_max=1.0, beta=50.0, n=0.5),
    }


def _make_inputs(N=100):
    """Return (E0, eps_vm) CuPy arrays of size N."""
    E0 = cp.full(N, 0.5, dtype=cp.float32)  # base modulus 0.5
    eps_vm = cp.linspace(0.0, 0.1, N, dtype=cp.float32)
    return E0, eps_vm


def test_linear_model(models):
    """Linear model: E_t == E0 for all inputs."""
    E0, eps_vm = _make_inputs()
    E_t = models["linear"].tangent_modulus(E0, eps_vm)
    diff = float(cp.max(cp.abs(E_t - E0)).get())
    assert diff < 1e-6, f"Linear model: E_t != E0, max diff = {diff:.3e}"


def test_softening_zero_strain(models):
    """Softening at eps_vm=0: E_t == E0."""
    E0 = cp.full(10, 0.5, dtype=cp.float32)
    eps_vm = cp.zeros(10, dtype=cp.float32)
    E_t = models["softening"].tangent_modulus(E0, eps_vm)
    diff = float(cp.max(cp.abs(E_t - E0)).get())
    assert diff < 1e-5, f"Softening at zero strain: E_t != E0, diff={diff:.3e}"


def test_softening_decreases(models):
    """Softening: E_t < E0 for eps_vm > 0."""
    E0, eps_vm = _make_inputs()
    # Only check non-zero strains
    E_t = models["softening"].tangent_modulus(E0, eps_vm)
    mask = cp.asnumpy(eps_vm) > 1e-6
    E_t_np = cp.asnumpy(E_t)
    E0_np = cp.asnumpy(E0)
    assert all(E_t_np[mask] < E0_np[mask]), \
        "Softening model: E_t not strictly less than E0 for eps_vm > 0"


def test_softening_large_strain_clamped(models):
    """Softening: E_t clamped to E_void for large eps_vm."""
    E0 = cp.full(5, 0.5, dtype=cp.float32)
    eps_vm = cp.full(5, 1000.0, dtype=cp.float32)  # extreme strain
    E_t = models["softening"].tangent_modulus(E0, eps_vm)
    E_t_np = cp.asnumpy(E_t)
    assert all(abs(v - 1e-6) < 1e-8 for v in E_t_np), \
        f"Softening at large strain not clamped to E_void: {E_t_np}"


def test_hardening_nonnegative(models):
    """Hardening: E_t >= E0 for all eps_vm >= 0."""
    E0, eps_vm = _make_inputs()
    E_t = models["hardening"].tangent_modulus(E0, eps_vm)
    E_t_np = cp.asnumpy(E_t)
    E0_np = cp.asnumpy(E0)
    assert all(E_t_np >= E0_np - 1e-6), \
        "Hardening model: E_t < E0 for some element"


def test_hardening_monotone(models):
    """Hardening: E_t increases monotonically with eps_vm."""
    E0 = cp.full(50, 0.5, dtype=cp.float32)
    eps_vm = cp.linspace(0.0, 0.05, 50, dtype=cp.float32)
    E_t = models["hardening"].tangent_modulus(E0, eps_vm)
    E_t_np = cp.asnumpy(E_t)
    diffs = E_t_np[1:] - E_t_np[:-1]
    assert all(d >= -1e-6 for d in diffs), \
        "Hardening model: E_t not monotonically increasing"


def test_hardening_large_strain_clamped(models):
    """Hardening: E_t clamped to E_max for large eps_vm."""
    E0 = cp.full(5, 0.5, dtype=cp.float32)
    eps_vm = cp.full(5, 1000.0, dtype=cp.float32)
    E_t = models["hardening"].tangent_modulus(E0, eps_vm)
    E_t_np = cp.asnumpy(E_t)
    assert all(abs(v - 1.0) < 1e-6 for v in E_t_np), \
        f"Hardening at large strain not clamped to E_max: {E_t_np}"


def test_clamp_bounds(models):
    """No model returns values outside [E_void, E_max] for any input."""
    E0 = cp.array([1e-8, 0.1, 0.5, 1.0, 1.5], dtype=cp.float32)
    eps_vm = cp.array([0.0, 0.01, 0.1, 1.0, 100.0], dtype=cp.float32)
    for name, model in models.items():
        E_t = model.tangent_modulus(E0, eps_vm)
        E_t_np = cp.asnumpy(E_t)
        assert all(v >= model.E_void - 1e-10 for v in E_t_np), \
            f"{name}: output below E_void"
        assert all(v <= model.E_max + 1e-10 for v in E_t_np), \
            f"{name}: output above E_max"


def test_cupy_dtype(models):
    """All outputs are float32 CuPy arrays."""
    E0, eps_vm = _make_inputs()
    for name, model in models.items():
        E_t = model.tangent_modulus(E0, eps_vm)
        assert isinstance(E_t, cp.ndarray), f"{name}: output is not a CuPy array"
        assert E_t.dtype == cp.float32, f"{name}: dtype is {E_t.dtype}, expected float32"
