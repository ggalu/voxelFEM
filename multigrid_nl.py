"""
Geometric Multigrid preconditioner for the nonlinear voxel FEM solver.

Implements a "frozen-tangent" multigrid strategy:
  1. At the start of each Newton-Raphson iteration, compute per-element
     tangent moduli from the current displacement u.
  2. Map tangent moduli to a uint8 voxel image for coarsening.
  3. Rebuild the multigrid hierarchy using these effective moduli.
  4. The hierarchy is treated as fixed during the inner PCG solve.

This reuses the linear GeometricMultigrid (multigrid.py) with updated
moduli — no new coarsening or smoothing algorithms are needed.
"""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from multigrid import GeometricMultigrid
from kernels_cupy_nl import compute_avg_strain_per_element


class NonlinearMultigrid:
    """
    GMG preconditioner that updates its hierarchy at each NR iteration.

    Parameters
    ----------
    voxels : np.ndarray (Nx, Ny, Nz) uint8 — original voxel stiffness map
    nu : float
    E_max : float
    E_void : float
    constitutive : ConstitutiveModel
    B_gauss_gpu : (8, 6, 24) float32 CuPy array
    E0_elem_gpu : (Ne,) float32 CuPy array — base moduli
    n_levels : int, optional
    n_pre : int — pre-smoothing iterations (default 2)
    n_post : int — post-smoothing iterations (default 2)
    chunk_size : int — elements per GPU chunk for strain computation
    """

    def __init__(self, voxels, nu, E_max, E_void, constitutive,
                 B_gauss_gpu, E0_elem_gpu,
                 n_levels=None, n_pre=2, n_post=2, chunk_size=2_000_000,
                 bc_mask=None, use_gpu=None,
                 smoother="jacobi", chebyshev_c_ratio=20.0):
        self.base_voxels = voxels.copy()
        self.nu = nu
        self.E_max = E_max
        self.E_void = E_void
        self.constitutive = constitutive
        self.B_gauss_gpu = B_gauss_gpu
        self.E0_elem_gpu = E0_elem_gpu
        self.n_levels = n_levels
        self.n_pre = n_pre
        self.n_post = n_post
        self.chunk_size = chunk_size
        # Run the whole V-cycle on the GPU when CuPy is available, so each PCG
        # iteration applies the preconditioner without a host round-trip.
        self.use_gpu = CUPY_AVAILABLE if use_gpu is None else use_gpu
        self.bc_mask = bc_mask
        self.smoother = smoother
        self.chebyshev_c_ratio = chebyshev_c_ratio

        self.Nx, self.Ny, self.Nz = voxels.shape

        # Build initial hierarchy from base voxels (before any deformation).
        # E_void=0.0 so that void elements (uint8=0) contribute zero stiffness
        # at every level, consistent with the fine-level solid_mask treatment.
        self._gmg = GeometricMultigrid(
            voxels, nu, E_max, 0.0,
            n_levels=n_levels, n_pre=n_pre, n_post=n_post, bc_mask=self.bc_mask,
            use_gpu=self.use_gpu, chunk_size=self.chunk_size,
            smoother=self.smoother, chebyshev_c_ratio=self.chebyshev_c_ratio,
        )

    def set_bc_mask(self, bc_mask):
        """Attach the fine-grid constrained-DOF mask (call before solving)."""
        self.bc_mask = bc_mask
        self._gmg._set_bcs(bc_mask)

    def update(self, u_gpu, force_rebuild=True):
        """
        Recompute tangent moduli from current u and rebuild the GMG hierarchy.

        Call once at the start of each NR iteration, before the inner PCG.

        Algorithm:
        1. Compute average von Mises strain per element.
        2. Compute tangent modulus E_t = constitutive(E0, eps_vm_avg).
        3. Map E_t → uint8 image scale for use by GeometricMultigrid.
        4. Rebuild multigrid hierarchy.

        Parameters
        ----------
        u_gpu : (Ndof,) float32 CuPy
        force_rebuild : bool — rebuild even if change is small (default True)
        """
        Ne = int(self.E0_elem_gpu.shape[0])

        # Upcast reduced-precision inputs to float32 for computation
        u_f32 = u_gpu.astype(cp.float32) if u_gpu.dtype != cp.float32 else u_gpu
        E0_f32 = (self.E0_elem_gpu.astype(cp.float32)
                  if self.E0_elem_gpu.dtype != cp.float32
                  else self.E0_elem_gpu)

        # Average von Mises strain per element (over 8 Gauss points)
        eps_vm_avg = compute_avg_strain_per_element(
            u_f32, self.B_gauss_gpu, self.Ny, self.Nz, Ne,
            chunk_size=self.chunk_size
        )

        # Tangent modulus at current state
        E_t = self.constitutive.tangent_modulus(E0_f32, eps_vm_avg)

        # Map to uint8 relative to E_max for coarsening
        E_t_cpu = cp.asnumpy(E_t)
        E_t_uint8 = np.clip((E_t_cpu / self.E_max) * 255.0, 0, 255).astype(np.uint8)
        E_t_voxels = E_t_uint8.reshape(self.Nx, self.Ny, self.Nz)

        # Fallback to base voxels if all tangent moduli are negligible
        if int(E_t_voxels.max()) == 0:
            E_t_voxels = self.base_voxels.copy()

        # Free old hierarchy before building the new one so they don't coexist.
        del self._gmg
        cp.get_default_memory_pool().free_all_blocks()

        self._gmg = GeometricMultigrid(
            E_t_voxels, self.nu, self.E_max, 0.0,
            n_levels=self.n_levels, n_pre=self.n_pre, n_post=self.n_post,
            bc_mask=self.bc_mask,
            use_gpu=self.use_gpu, chunk_size=self.chunk_size,
            smoother=self.smoother, chebyshev_c_ratio=self.chebyshev_c_ratio,
        )

    def apply(self, r):
        """
        Apply GMG preconditioner: M^{-1} · r via one V-cycle.

        Parameters
        ----------
        r : (Ndof,) float32 — residual (CuPy or NumPy)

        Returns
        -------
        z : (Ndof,) float32 — preconditioned residual

        With the GPU backend the V-cycle is fully GPU-resident: a CuPy residual
        goes straight through with no host transfer. The CPU fallback (and a
        NumPy residual handed to a GPU hierarchy, or vice-versa) is converted as
        needed so the call is type-robust.
        """
        if self.use_gpu:
            if isinstance(r, cp.ndarray):
                return self._gmg.apply(r)                       # GPU-resident
            # NumPy residual into a GPU hierarchy: move it across.
            return cp.asnumpy(self._gmg.apply(cp.asarray(r, dtype=cp.float32)))

        # CPU hierarchy.
        if CUPY_AVAILABLE and isinstance(r, cp.ndarray):
            return cp.asarray(self._gmg.apply(cp.asnumpy(r)), dtype=cp.float32)
        return self._gmg.apply(r).astype(np.float32)
