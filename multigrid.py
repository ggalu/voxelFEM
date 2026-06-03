"""
Geometric Multigrid preconditioner for voxel FEM solver.

Implements a V-cycle preconditioner that reduces CG iterations from O(N^(1/3))
to O(1) by recursively solving on coarser grids.
"""

import numpy as np
from geometry import (
    build_node_index, element_node_ids, build_K_ref, matvec_Ku, build_diagonal
)


class GridLevel:
    """
    Holds data for one multigrid level.
    """
    def __init__(
        self,
        Nx: int, Ny: int, Nz: int,
        voxels: np.ndarray,
        E_max: float,
        E_void: float,
        nu: float,
        stiffness_scale: float = 1.0,
    ):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.voxels = voxels.copy()

        # Build DOF mapping
        Nn = (Nx + 1) * (Ny + 1) * (Nz + 1)
        self.Ndof = 3 * Nn

        # Build element properties. The geometric hierarchy builds every level
        # from unit cubes, but a coarse element physically spans 2× the side per
        # coarsening and 3-D elasticity stiffness scales as K ∝ h, so the coarse
        # operator must be scaled (stiffness_scale = 2^level) to approximate the
        # Galerkin operator P^T A P. Without it the V-cycle over-corrects.
        E_scale = (E_max - E_void) / 255.0
        self.E_elem = ((voxels.ravel().astype(np.float32) * E_scale + E_void)
                       * np.float32(stiffness_scale))

        # Build reference element stiffness (same for all levels)
        self.K_ref = build_K_ref(nu)

        # Build connectivity
        node_idx = build_node_index(Nx, Ny, Nz)
        conn_nodes = element_node_ids(node_idx, Nx, Ny, Nz)
        self.conn_dof = (conn_nodes[:, :, np.newaxis] * 3
                        + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)

        # Precompute diagonal for Jacobi smoother
        diag = build_diagonal(self.E_elem, self.conn_dof, self.K_ref, self.Ndof)
        diag = np.where(diag < 1e-12, 1.0, diag)
        self.diag_inv = (1.0 / diag).astype(np.float32)

        # Essential (Dirichlet) DOFs for this level; set via set_bc(). Without
        # BCs the operator is singular (rigid-body modes) and the V-cycle
        # diverges, so the constrained DOFs are treated as identity rows.
        self.bc_mask = None

    def set_bc(self, bc_mask):
        """
        Attach the constrained-DOF mask for this level.

        Constrained DOFs become identity rows in ``matvec`` and are frozen by
        the smoother (``diag_inv = 1`` there), so a homogeneous correction keeps
        them at zero throughout the V-cycle.
        """
        self.bc_mask = bc_mask
        if bc_mask is not None:
            self.diag_inv = self.diag_inv.copy()
            self.diag_inv[bc_mask] = 1.0

    def matvec(self, u: np.ndarray) -> np.ndarray:
        """Apply K·u for this level (identity rows on constrained DOFs)."""
        w = matvec_Ku(u, self.E_elem, self.conn_dof, self.K_ref, self.Ndof)
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
    ):
        self.voxels_fine = voxels.copy()
        self.Nx, self.Ny, self.Nz = voxels.shape
        self.nu = nu
        self.E_max = E_max
        self.E_void = E_void
        self.n_pre = n_pre
        self.n_post = n_post
        self.coarse_solver = coarse_solver
        self.verbose = False

        # Build multigrid hierarchy
        self.levels = []
        img = voxels.copy()
        level_idx = 0
        while True:
            level = GridLevel(img.shape[0], img.shape[1], img.shape[2],
                             img, E_max, E_void, nu,
                             stiffness_scale=2.0 ** level_idx)
            self.levels.append(level)

            if n_levels and len(self.levels) >= n_levels:
                break
            if min(img.shape) < 4:
                break

            img = self._coarsen_image(img)
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
        """Per-axis 1-D prolongation matrices for a fine/coarse level pair."""
        return (
            self._prolong_matrix_1d(fine_level.Nx, coarse_level.Nx),
            self._prolong_matrix_1d(fine_level.Ny, coarse_level.Ny),
            self._prolong_matrix_1d(fine_level.Nz, coarse_level.Nz),
        )

    def _restrict(self, r_fine: np.ndarray, fine_level: GridLevel,
                  coarse_level: GridLevel) -> np.ndarray:
        """
        Restrict residual fine → coarse by full weighting, R = P^T.

        Applies the transpose of each 1-D prolongation matrix along its axis.
        """
        Px, Py, Pz = self._axis_matrices(fine_level, coarse_level)
        r = r_fine.reshape(fine_level.Nx + 1, fine_level.Ny + 1,
                           fine_level.Nz + 1, 3)
        r = np.einsum('ai,abcd->ibcd', Px, r)   # contract fine axis a → coarse i
        r = np.einsum('bj,ibcd->ijcd', Py, r)
        r = np.einsum('ck,ijcd->ijkd', Pz, r)
        return r.reshape(-1).astype(np.float32)

    def _prolongate(self, e_coarse: np.ndarray, coarse_level: GridLevel,
                    fine_level: GridLevel) -> np.ndarray:
        """
        Prolongate correction coarse → fine by trilinear interpolation, P.
        """
        Px, Py, Pz = self._axis_matrices(fine_level, coarse_level)
        e = e_coarse.reshape(coarse_level.Nx + 1, coarse_level.Ny + 1,
                            coarse_level.Nz + 1, 3)
        e = np.einsum('ai,ijkd->ajkd', Px, e)   # coarse i → fine a
        e = np.einsum('bj,ajkd->abkd', Py, e)
        e = np.einsum('ck,abkd->abcd', Pz, e)
        return e.reshape(-1).astype(np.float32)

    def _smooth(self, level: GridLevel, u: np.ndarray, f: np.ndarray,
                n_iter: int) -> np.ndarray:
        """
        Damped Jacobi smoother: u ← u + ω · D⁻¹ · (f − K·u)
        """
        omega = 0.67
        u = u.copy()

        for _ in range(n_iter):
            r = f - level.matvec(u)
            u = u + omega * level.diag_inv * r

        return u

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

        # Pre-smooth
        e = self._smooth(level, np.zeros_like(r), r, self.n_pre)

        # Restrict residual; zero constrained coarse DOFs (full weighting mixes
        # in free neighbours, so the coarse residual must be re-masked).
        r_res = r - level.matvec(e)
        r_coarse = self._restrict(r_res, level, coarse)
        if coarse.bc_mask is not None:
            r_coarse[coarse.bc_mask] = 0.0

        # Recurse
        e_coarse = self.vcycle(r_coarse, level_idx + 1)

        # Prolongate and correct (constrained fine DOFs stay at zero)
        e_correction = self._prolongate(e_coarse, coarse, level)
        if level.bc_mask is not None:
            e_correction[level.bc_mask] = 0.0
        e = e + e_correction

        # Post-smooth
        e = self._smooth(level, e, r, self.n_post)

        return e

    def _cg_solve(self, level: GridLevel, b: np.ndarray, max_iter: int = 50,
                  tol: float = 1e-6) -> np.ndarray:
        """
        Simple CG solver on the coarsest level.
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rsold = float(np.dot(r, r))

        for it in range(max_iter):
            Ap = level.matvec(p)
            pAp = float(np.dot(p, Ap))
            alpha = rsold / (pAp + 1e-30)

            x += alpha * p
            r -= alpha * Ap

            rsnew = float(np.dot(r, r))
            if np.sqrt(rsnew) < tol * np.sqrt(rsold + 1e-30):
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
