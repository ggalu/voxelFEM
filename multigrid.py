"""
Geometric Multigrid preconditioner for voxel FEM solver.

Implements a V-cycle preconditioner that reduces CG iterations from O(N^(1/3))
to O(1) by recursively solving on coarser grids.
"""

import numpy as np
from geometry import (
    build_node_index, element_node_ids, build_K_ref, matvec_Ku, build_diagonal
)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class GridLevel:
    """
    Holds data for one multigrid level.

    Backend-parametric: with ``xp is np`` everything runs on the CPU (the
    reference path the pure-NumPy tests exercise); with ``xp is cupy`` the level
    operator and Jacobi diagonal are GPU-resident and use implicit connectivity
    (``conn_dof`` is never stored — required for the 300M-voxel budget).
    """
    def __init__(
        self,
        Nx: int, Ny: int, Nz: int,
        voxels: np.ndarray,
        E_max: float,
        E_void: float,
        nu: float,
        stiffness_scale: float = 1.0,
        xp=np,
        chunk_size: int = 2_000_000,
    ):
        self.xp = xp
        self.chunk_size = chunk_size
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.voxels = voxels.copy()           # CPU image (used for coarsening)

        # Build DOF mapping
        Nn = (Nx + 1) * (Ny + 1) * (Nz + 1)
        self.Ndof = 3 * Nn

        # Build element properties. The geometric hierarchy builds every level
        # from unit cubes, but a coarse element physically spans 2× the side per
        # coarsening and 3-D elasticity stiffness scales as K ∝ h, so the coarse
        # operator must be scaled (stiffness_scale = 2^level) to approximate the
        # Galerkin operator P^T A P. Without it the V-cycle over-corrects.
        E_scale = (E_max - E_void) / 255.0
        E_elem_cpu = ((voxels.ravel().astype(np.float32) * E_scale + E_void)
                      * np.float32(stiffness_scale))
        K_ref_cpu = build_K_ref(nu)           # (24, 24) float64

        if xp is np:
            # ----- CPU reference path (unchanged numerics) -----
            self.E_elem = E_elem_cpu
            self.K_ref = K_ref_cpu
            node_idx = build_node_index(Nx, Ny, Nz)
            conn_nodes = element_node_ids(node_idx, Nx, Ny, Nz)
            self.conn_dof = (conn_nodes[:, :, np.newaxis] * 3
                            + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)
            diag = build_diagonal(self.E_elem, self.conn_dof,
                                  self.K_ref, self.Ndof)
        else:
            # ----- GPU path: implicit connectivity, no conn_dof stored -----
            from kernels_cupy_nl import build_diagonal_cupy
            self.E_elem = xp.asarray(E_elem_cpu, dtype=xp.float32)
            self.K_ref = xp.asarray(K_ref_cpu.astype(np.float32))
            self.conn_dof = None
            diag_ref = xp.asarray(np.diag(K_ref_cpu).astype(np.float32))
            diag = build_diagonal_cupy(self.E_elem, diag_ref,
                                       self.Ny, self.Nz, self.Ndof,
                                       chunk_size=self.chunk_size)

        # Identify zero-diagonal DOFs before the floor replacement.  These are
        # nodes adjacent only to void/zero-E elements; they need identity rows
        # in matvec (same as Dirichlet BCs) so the coarse operator is
        # non-singular when void regions are present.
        zero_diag_bool = diag < 1e-12
        zero_diag_cpu = (cp.asnumpy(zero_diag_bool)
                         if xp is not np else zero_diag_bool)
        self._void_dofs_np = np.where(zero_diag_cpu)[0].astype(np.int32)

        diag = xp.where(zero_diag_bool, xp.asarray(1.0, dtype=diag.dtype), diag)
        self.diag_inv = (1.0 / diag).astype(np.float32)

        # bc_mask holds the union of Dirichlet + void-only DOF indices (identity
        # rows in matvec); initialised here, extended by set_bc().
        self.bc_mask = None

    def set_bc(self, bc_mask):
        """
        Attach the constrained-DOF mask for this level.

        Constrained DOFs become identity rows in ``matvec`` and are frozen by
        the smoother (``diag_inv = 1`` there), so a homogeneous correction keeps
        them at zero throughout the V-cycle. ``bc_mask`` arrives as a NumPy
        bool array; it is converted to int32 indices and stored on ``xp`` to
        avoid holding a full-Ndof bool per level.

        Void-only DOFs (``_void_dofs_np``, pre-computed in ``__init__``) are
        always merged into the combined mask so zero-E nodes at coarse levels
        also receive identity rows.  This keeps the coarse operator consistent
        with the fine-level solid_mask treatment of void elements.
        """
        bc_idx = (np.where(bc_mask)[0].astype(np.int32)
                  if bc_mask is not None else np.empty(0, dtype=np.int32))
        combined = np.union1d(bc_idx, self._void_dofs_np).astype(np.int32)
        if len(combined) == 0:
            self.bc_mask = None
            return
        self.bc_mask = self.xp.asarray(combined)
        self.diag_inv = self.diag_inv.copy()
        self.diag_inv[self.bc_mask] = 1.0

    def matvec(self, u):
        """Apply K·u for this level (identity rows on constrained DOFs)."""
        if self.xp is np:
            w = matvec_Ku(u, self.E_elem, self.conn_dof, self.K_ref, self.Ndof)
        else:
            from kernels_cupy_nl import matvec_Ku_cupy
            w = matvec_Ku_cupy(u, self.E_elem, self.K_ref,
                               self.Ny, self.Nz, self.Ndof,
                               chunk_size=self.chunk_size)
        if self.bc_mask is not None:
            w[self.bc_mask] = u[self.bc_mask]
        return w


class GeometricMultigrid:
    """
    Geometric multigrid preconditioner for voxel FEM.

    Parameters
    ----------
    voxels : np.ndarray (Nx, Ny, Nz) uint8
        Voxel stiffness map
    nu : float
        Poisson's ratio
    E_max : float
        Maximum Young's modulus
    E_void : float
        Young's modulus for void (voxel=0)
    n_levels : int, optional
        Number of grid levels. If None, coarsen until min dimension < 4.
    n_pre : int
        Number of pre-smoothing iterations (default 2)
    n_post : int
        Number of post-smoothing iterations (default 2)
    coarse_solver : str
        "cg" or "direct" (default "cg")
    use_gpu : bool
        If True, levels are GPU-resident (CuPy, implicit connectivity) and the
        whole V-cycle runs on the GPU — no per-apply host transfer. If False
        (default) everything runs on NumPy (the pure-CPU reference path).
    chunk_size : int
        Elements per GPU chunk for the level operator (GPU path only).
    """

    def __init__(
        self,
        voxels: np.ndarray,
        nu: float = 0.3,
        E_max: float = 1.0,
        E_void: float = 1e-6,
        n_levels: int = None,
        n_pre: int = 2,
        n_post: int = 2,
        coarse_solver: str = "cg",
        bc_mask: np.ndarray = None,
        use_gpu: bool = False,
        chunk_size: int = 2_000_000,
        smoother: str = "jacobi",
        chebyshev_c_ratio: float = 20.0,
    ):
        if use_gpu and not CUPY_AVAILABLE:
            raise RuntimeError("use_gpu=True requires CuPy, which is not available")
        self.xp = cp if use_gpu else np
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        self.voxels_fine = voxels.copy()
        self.Nx, self.Ny, self.Nz = voxels.shape
        self.nu = nu
        self.E_max = E_max
        self.E_void = E_void
        self.n_pre = n_pre
        self.n_post = n_post
        self.coarse_solver = coarse_solver
        self.verbose = False
        self.smoother = smoother
        self.chebyshev_c_ratio = chebyshev_c_ratio
        self._lambda_max_cache = {}  # {id(level): estimated lam_max}

        # Build multigrid hierarchy
        self.levels = []
        img = voxels.copy()
        level_idx = 0
        while True:
            level = GridLevel(img.shape[0], img.shape[1], img.shape[2],
                             img, E_max, E_void, nu,
                             stiffness_scale=2.0 ** level_idx,
                             xp=self.xp, chunk_size=chunk_size)
            self.levels.append(level)

            if n_levels and len(self.levels) >= n_levels:
                break
            if max(img.shape) < 4:
                break

            img = self._coarsen_image_anisotropic(img)
            level_idx += 1

        # Propagate the fine constrained-DOF mask down the hierarchy so every
        # coarse operator is non-singular (BCs anchor the rigid-body modes).
        self._set_bcs(bc_mask)

        if self.verbose:
            print(f"Built multigrid hierarchy with {len(self.levels)} levels")
            for i, level in enumerate(self.levels):
                print(f"  Level {i}: {level.Nx}×{level.Ny}×{level.Nz}, "
                      f"{level.Ndof} DOFs")

    def _set_bcs(self, bc_mask_fine):
        """
        Set the constrained-DOF mask on every level.

        The fine mask is injected coarse-ward: a coarse DOF (ic,jc,kc,d) is
        constrained iff the coincident fine DOF (2ic,2jc,2kc,d) is constrained.
        """
        if bc_mask_fine is None:
            for lvl in self.levels:
                lvl.set_bc(None)
            return
        self.levels[0].set_bc(bc_mask_fine)
        mask = bc_mask_fine
        for lf in range(len(self.levels) - 1):
            mask = self._inject_bc(mask, self.levels[lf], self.levels[lf + 1])
            self.levels[lf + 1].set_bc(mask)

    @staticmethod
    def _inject_bc(bc_fine, fine: "GridLevel", coarse: "GridLevel"):
        """Coarse DOF mask by injection from the coincident fine even nodes."""
        bf = bc_fine.reshape(fine.Nx + 1, fine.Ny + 1, fine.Nz + 1, 3)
        ci = np.minimum(2 * np.arange(coarse.Nx + 1), fine.Nx)
        cj = np.minimum(2 * np.arange(coarse.Ny + 1), fine.Ny)
        ck = np.minimum(2 * np.arange(coarse.Nz + 1), fine.Nz)
        bc = bf[np.ix_(ci, cj, ck)]            # (nc+1, nc+1, nc+1, 3)
        return bc.reshape(-1).copy()

    @staticmethod
    def _coarsen_image(img: np.ndarray) -> np.ndarray:
        """
        Coarsen voxel image by 2× in each dimension using 2×2×2 averaging.

        Vectorised: ``np.add.reduceat`` sums each length-2 group along every
        axis (the trailing group covers a length-1 remainder for odd sizes);
        dividing by the per-group counts gives the block mean, equivalent to a
        single 2×2×2 mean. The number of coarse cells per axis is ``(N+1)//2``.
        """
        a = img.astype(np.float64)
        for axis in range(3):
            n = a.shape[axis]
            starts = np.arange(0, n, 2)
            block_sum = np.add.reduceat(a, starts, axis=axis)
            counts = np.add.reduceat(np.ones(n), starts)   # 2 per group, 1 if odd tail
            shape = [1, 1, 1]
            shape[axis] = len(starts)
            a = block_sum / counts.reshape(shape)
        return np.clip(a + 0.5, 0, 255).astype(np.uint8)

    @staticmethod
    def _coarsen_image_anisotropic(img: np.ndarray) -> np.ndarray:
        """
        Coarsen only dimensions whose current size is ≥ 4; freeze smaller ones.

        For strongly anisotropic grids (e.g. 450×450×20) the thin dimension
        exhausts first under uniform 2×2×2 coarsening, capping the hierarchy at
        4 levels.  This method keeps coarsening the large dimensions even after
        the thin one becomes ≤ 3, growing the hierarchy until max(shape) < 4.
        Thin dimensions (size < 4) are kept at their current size so the element
        geometry remains non-degenerate.
        """
        a = img.astype(np.float64)
        for axis in range(3):
            n = a.shape[axis]
            if n < 4:
                continue            # freeze this dimension
            starts = np.arange(0, n, 2)
            block_sum = np.add.reduceat(a, starts, axis=axis)
            counts = np.add.reduceat(np.ones(n), starts)
            shape = [1, 1, 1]
            shape[axis] = len(starts)
            a = block_sum / counts.reshape(shape)
        return np.clip(a + 0.5, 0, 255).astype(np.uint8)

    @staticmethod
    def _prolong_matrix_1d(n_fine: int, n_coarse: int) -> np.ndarray:
        """
        1-D trilinear prolongation matrix P, shape (n_fine+1, n_coarse+1).

        Even fine nodes inject the coincident coarse value; odd fine nodes are
        the average of their two coarse neighbours. The full 3-D prolongation is
        the tensor product Px ⊗ Py ⊗ Pz; restriction is the exact transpose
        (full weighting), so the adjoint property ⟨r, P e⟩ = ⟨P^T r, e⟩ holds.
        """
        P = np.zeros((n_fine + 1, n_coarse + 1), dtype=np.float64)
        for i in range(n_fine + 1):
            ic = i // 2
            if i % 2 == 0:
                P[i, ic] = 1.0
            else:
                P[i, ic] = 0.5
                if ic + 1 <= n_coarse:
                    P[i, ic + 1] = 0.5
        return P

    def _axis_matrices(self, fine_level: GridLevel,
                       coarse_level: GridLevel):
        """Per-axis 1-D prolongation matrices for a fine/coarse level pair.

        Built on the CPU (small, pure structure) and moved to ``xp`` so the
        einsums stay on the active backend.
        """
        return (
            self.xp.asarray(self._prolong_matrix_1d(fine_level.Nx, coarse_level.Nx)),
            self.xp.asarray(self._prolong_matrix_1d(fine_level.Ny, coarse_level.Ny)),
            self.xp.asarray(self._prolong_matrix_1d(fine_level.Nz, coarse_level.Nz)),
        )

    def _restrict(self, r_fine, fine_level: GridLevel,
                  coarse_level: GridLevel):
        """
        Restrict residual fine → coarse by full weighting, R = P^T.

        Applies the transpose of each 1-D prolongation matrix along its axis.
        """
        xp = self.xp
        Px, Py, Pz = self._axis_matrices(fine_level, coarse_level)
        r = r_fine.reshape(fine_level.Nx + 1, fine_level.Ny + 1,
                           fine_level.Nz + 1, 3)
        r = xp.einsum('ai,abcd->ibcd', Px, r)   # contract fine axis a → coarse i
        r = xp.einsum('bj,ibcd->ijcd', Py, r)
        r = xp.einsum('ck,ijcd->ijkd', Pz, r)
        return r.reshape(-1).astype(np.float32)

    def _prolongate(self, e_coarse, coarse_level: GridLevel,
                    fine_level: GridLevel):
        """
        Prolongate correction coarse → fine by trilinear interpolation, P.
        """
        xp = self.xp
        Px, Py, Pz = self._axis_matrices(fine_level, coarse_level)
        e = e_coarse.reshape(coarse_level.Nx + 1, coarse_level.Ny + 1,
                            coarse_level.Nz + 1, 3)
        e = xp.einsum('ai,ijkd->ajkd', Px, e)   # coarse i → fine a
        e = xp.einsum('bj,ajkd->abkd', Py, e)
        e = xp.einsum('ck,abkd->abcd', Pz, e)
        return e.reshape(-1).astype(np.float32)

    def _smooth(self, level: GridLevel, u: np.ndarray, f: np.ndarray,
                n_iter: int, copy_input: bool = True) -> np.ndarray:
        """
        Damped Jacobi smoother: u ← u + ω · D⁻¹ · (f − K·u)

        Uses in-place ops to keep transient allocation to one Ndof array
        (the matvec output), rather than the ~4× Ndof of the naive version.
        """
        if copy_input:
            u = u.copy()

        for _ in range(n_iter):
            delta = level.matvec(u)  # allocs one Ndof array (unavoidable)
            delta -= f               # delta = K·u − f
            delta *= level.diag_inv  # in-place: diag_inv·(K·u − f)
            delta *= 0.67            # in-place scalar
            u -= delta               # u += 0.67·diag_inv·(f − K·u)

        return u

    def _estimate_lambda_max(self, level):
        """
        Estimate the largest eigenvalue of D⁻¹A on this level via 10 power
        iterations.  Used to set the Chebyshev smoothing interval.
        """
        xp = level.xp
        rng = np.random.default_rng(42)
        v = xp.asarray(rng.standard_normal(level.Ndof).astype(np.float32))
        if level.bc_mask is not None:
            v[level.bc_mask] = 0.0
        nrm = float(xp.linalg.norm(v))
        if nrm < 1e-30:
            return 1.0
        v = v / nrm

        lam = 1.0
        for _ in range(10):
            w = level.matvec(v)
            if level.bc_mask is not None:
                w[level.bc_mask] = 0.0
            w *= level.diag_inv
            if level.bc_mask is not None:
                w[level.bc_mask] = 0.0
            lam = float(xp.dot(v, w))
            nrm = float(xp.linalg.norm(w))
            if nrm < 1e-30:
                break
            v = w / nrm

        return max(lam * 1.1, 1e-10)

    def _get_lambda_max(self, level):
        """Return cached lambda_max estimate for D⁻¹A on this level."""
        key = id(level)
        if key not in self._lambda_max_cache:
            self._lambda_max_cache[key] = self._estimate_lambda_max(level)
        return self._lambda_max_cache[key]

    def _smooth_chebyshev(self, level, u, f, n_sweeps):
        """
        Jacobi-preconditioned Chebyshev polynomial smoother.

        Applies n_sweeps Richardson steps with step sizes 1/omega_j, where
        omega_j are the Chebyshev nodes of T_n on [lam_max/c_ratio, lam_max].
        This gives the optimal degree-n polynomial smoother on the high-
        frequency interval [lam_max/c_ratio, lam_max] of D⁻¹A, while leaving
        smooth modes (eigenvalues near 0) undisturbed.

        Inactive / constrained DOFs have diag_inv = 1 and are zeroed in the
        residual, so they remain at zero throughout (mask-aware by design).
        """
        xp = level.xp
        lam_max = self._get_lambda_max(level)
        lam_min = lam_max / self.chebyshev_c_ratio

        center = (lam_max + lam_min) / 2.0
        half_w = (lam_max - lam_min) / 2.0
        j = np.arange(1, n_sweeps + 1)
        omegas = center + half_w * np.cos(np.pi * (2 * j - 1) / (2 * n_sweeps))

        for omega in omegas:
            r = f - level.matvec(u)
            if level.bc_mask is not None:
                r[level.bc_mask] = 0.0
            z = level.diag_inv * r
            if level.bc_mask is not None:
                z[level.bc_mask] = 0.0
            z *= 1.0 / float(omega)
            u += z
            del r, z

        return u

    def _smooth_dispatch(self, level, u, f, n_iter):
        """Route to Chebyshev or Jacobi smoother based on self.smoother."""
        if self.smoother == "chebyshev":
            return self._smooth_chebyshev(level, u, f, n_iter)
        return self._smooth(level, u, f, n_iter, copy_input=False)

    def vcycle(self, r: np.ndarray, level_idx: int = 0) -> np.ndarray:
        """
        Perform one V-cycle. Returns approximate solution e to K·e = r.
        """
        level = self.levels[level_idx]

        if level_idx == len(self.levels) - 1:
            # Coarsest level: solve with a few CG iterations
            if self.coarse_solver == "cg":
                return self._cg_solve(level, r, max_iter=50)
            else:
                # Direct solve (not practical for large problems)
                return r / (level.diag_inv + 1e-12)

        coarse = self.levels[level_idx + 1]

        # Pre-smooth (zeros_like is freshly allocated — no copy needed)
        e = self._smooth_dispatch(level, self.xp.zeros_like(r), r, self.n_pre)

        # Restrict residual; zero constrained coarse DOFs (full weighting mixes
        # in free neighbours, so the coarse residual must be re-masked).
        r_res = r - level.matvec(e)
        r_coarse = self._restrict(r_res, level, coarse)
        del r_res   # free fine-level residual before recursion and post-smooth
        if coarse.bc_mask is not None:
            r_coarse[coarse.bc_mask] = 0.0

        # Recurse
        e_coarse = self.vcycle(r_coarse, level_idx + 1)

        # Prolongate and correct (constrained fine DOFs stay at zero)
        e_correction = self._prolongate(e_coarse, coarse, level)
        if level.bc_mask is not None:
            e_correction[level.bc_mask] = 0.0
        e = e + e_correction
        del e_correction

        # Post-smooth (e is freshly allocated above — no copy needed)
        e = self._smooth_dispatch(level, e, r, self.n_post)

        return e

    def _cg_solve(self, level: GridLevel, b, max_iter: int = 50,
                  tol: float = 1e-6):
        """
        Simple CG solver on the coarsest level.
        """
        xp = self.xp
        x = xp.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rsold = float(xp.dot(r, r))

        for it in range(max_iter):
            Ap = level.matvec(p)
            pAp = float(xp.dot(p, Ap))
            alpha = rsold / (pAp + 1e-30)

            x += alpha * p
            r -= alpha * Ap

            rsnew = float(xp.dot(r, r))
            if rsnew ** 0.5 < tol * (rsold + 1e-30) ** 0.5:
                break

            beta = rsnew / (rsold + 1e-30)
            p = r + beta * p
            rsold = rsnew

        return x

    def apply(self, r: np.ndarray) -> np.ndarray:
        """
        Preconditioner application: M⁻¹·r via one V-cycle.

        The result is a symmetric SPD operator (R = P^T, symmetric smoother,
        n_pre == n_post), so it is a valid PCG preconditioner.
        """
        bc = self.levels[0].bc_mask
        if bc is not None:
            r = r.copy()
            r[bc] = 0.0
        z = self.vcycle(r, level_idx=0)
        if bc is not None:
            z[bc] = 0.0
        return z
