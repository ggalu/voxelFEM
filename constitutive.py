"""
Constitutive models for the nonlinear voxel FEM solver.

Defines E(ε) relationships where the tangent modulus depends on the local
strain state. All models operate on CuPy arrays and return float32.

Models
------
linear    : E_t = E0  (reference; reproduces linear elasticity)
softening : E_t = E0 · exp(-alpha · eps_vm)   (foams, soft tissue)
hardening : E_t = E0 · (1 + beta · eps_vm^n)  (rubber, bone)

All outputs are clamped to [E_void, E_max].
"""

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    raise RuntimeError(
        "CuPy is required for constitutive models. "
        "Install with: pip install cupy-cuda12x"
    )


class ConstitutiveModel:
    """
    Nonlinear elastic material model: E_t = E(E0, eps_vm).

    Parameters
    ----------
    model : str
        "linear", "softening", or "hardening"
    E_void : float
        Minimum allowed modulus (void regions). Output is clipped to this.
    E_max : float
        Maximum allowed modulus. Output is clipped to this.
    alpha : float
        Softening rate (model="softening"). Typical: 10–100.
        Higher alpha → faster softening with strain.
    beta : float
        Hardening coefficient (model="hardening"). Typical: 50.
    n : float
        Hardening exponent (model="hardening"). Typical: 0.5.
    """

    def __init__(self, model="linear", E_void=1e-6, E_max=1.0,
                 alpha=10.0, beta=50.0, n=0.5):
        valid = ("linear", "softening", "hardening")
        if model not in valid:
            raise ValueError(f"Unknown model '{model}'. Choose from: {valid}")
        self.model = model
        self.E_void = float(E_void)
        self.E_max = float(E_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n = float(n)

    def tangent_modulus(self, E0_elem, eps_vm):
        """
        Compute tangent modulus for each element/Gauss point.

        Parameters
        ----------
        E0_elem : (Ne,) float32 CuPy array — base modulus per element
        eps_vm  : (Ne,) float32 CuPy array — von Mises equivalent strain

        Returns
        -------
        E_t : (Ne,) float32 CuPy array — tangent modulus in [E_void, E_max]
        """
        if self.model == "linear":
            # Identity: tangent modulus equals the base modulus everywhere
            E_t = E0_elem.copy()

        elif self.model == "softening":
            # Exponential softening: material weakens as strain increases.
            # E_t → E0 at zero strain, E_t → 0 as eps_vm → ∞ (clamped to E_void).
            E_t = E0_elem * cp.exp(-self.alpha * eps_vm)

        elif self.model == "hardening":
            # Power-law hardening: material stiffens with deformation.
            # Small eps_vm offset (1e-30) avoids 0^n issues for n < 1.
            E_t = E0_elem * (1.0 + self.beta * cp.power(eps_vm + 1e-30, self.n))

        # Clamp to [E_void, E_max] — prevents blow-up in near-void regions
        return cp.clip(E_t, self.E_void, self.E_max).astype(cp.float32)

    def __repr__(self):
        if self.model == "linear":
            return f"ConstitutiveModel(linear, E_void={self.E_void}, E_max={self.E_max})"
        elif self.model == "softening":
            return (f"ConstitutiveModel(softening, alpha={self.alpha}, "
                    f"E_void={self.E_void}, E_max={self.E_max})")
        else:
            return (f"ConstitutiveModel(hardening, beta={self.beta}, n={self.n}, "
                    f"E_void={self.E_void}, E_max={self.E_max})")
