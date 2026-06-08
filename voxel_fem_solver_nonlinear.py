"""
Voxel FEM Solver — Nonlinear Elasticity
=========================================
Extends the linear solver to support nonlinear elasticity where E = E(ε)
via a ConstitutiveModel.

Algorithm
---------
Newton-Raphson outer loop with incremental load stepping:

  u ← 0  (or u_bc at load step 1)
  for load_step s = 1 … n_steps:
      f_target = (s / n_steps) · f_total
      u_bc_step = (s / n_steps) · bc_values_total
      for NR iter k = 0 … max_nr:
          f_int = Σ_e Σ_gp  w · B^T · D_t(ε) · B · u
          R     = f_target - f_int;   R[bc_dofs] = 0
          if ||R||/f_norm < tol_nr: break
          η_k   = Eisenstat-Walker inner tolerance
          Δu    = PCG(K_tangent(u), R, GMG_precond)
          u    += Δu;   u[bc_dofs] = u_bc_step

Constitutive models (via ConstitutiveModel):
  linear    : E_t = E0
  softening : E_t = E0 · exp(-alpha · eps_vm)
  hardening : E_t = E0 · (1 + beta · eps_vm^n)

GPU memory (300M elements, 48 GB VRAM):
  E0_elem (float16):                  0.6 GB
  u, x/Δu, p, r, z (float32):       21.6 GB   (6 arrays × 3.6 GB)
  f_int/R, Ap (float32):             7.2 GB   (2 arrays × 3.6 GB)
  MG fine diagonal (float32):        3.6 GB
  MG coarse diagonals:               0.5 GB
  Chunk working buffers (2M chunk): ~1.0 GB
  CuPy pool cache:                  ~1.0 GB
  Total:                            ~32 GB ✓ (16 GB headroom at 48 GB)

Example
-------
  import numpy as np
  from geometry import build_node_index
  from constitutive import ConstitutiveModel
  from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear

  voxels = np.full((20, 20, 20), 200, dtype=np.uint8)
  mat = ConstitutiveModel("softening", E_void=1e-6, E_max=1.0, alpha=10.0)
  solver = VoxelFEMSolverNonlinear(voxels, nu=0.3, E_max=1.0, E_void=1e-6,
                                    constitutive=mat, n_load_steps=5)
  node_idx = build_node_index(20, 20, 20)
  bottom = node_idx[:, :, 0].ravel()
  top    = node_idx[:, :, -1].ravel()
  for dof in range(3):
      solver.apply_bc_displacement(bottom, np.full(len(bottom), dof), np.zeros(len(bottom)))
  solver.apply_bc_displacement(top, np.full(len(top), 2), np.full(len(top), 0.01))
  u = solver.solve()
"""

import math
import numpy as np
from typing import Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    raise RuntimeError(
        "CuPy is required for VoxelFEMSolverNonlinear. "
        "Install with: pip install cupy-cuda12x"
    )

from geometry import build_node_index
from kernels_cupy_nl import (build_B_gauss, compute_f_int_cupy, matvec_tangent_cupy,
                              build_active_dof_gpu)
from constitutive import ConstitutiveModel
from multigrid_nl import NonlinearMultigrid


# ---------------------------------------------------------------------------
# PCG convergence monitoring helpers (CPU-only, same as Taichi solver)
# ---------------------------------------------------------------------------

def _lanczos_kappa(alphas, betas):
    """Estimate κ(M⁻¹K) from Lanczos tridiagonal built from PCG coefficients."""
    k = len(alphas)
    if k < 2 or len(betas) < k - 1:
        return None, None, None
    try:
        diag = np.empty(k)
        diag[0] = 1.0 / alphas[0]
        for j in range(1, k):
            diag[j] = 1.0 / alphas[j] + betas[j - 1] / alphas[j - 1]
        off = np.array([math.sqrt(abs(betas[j])) / abs(alphas[j])
                        for j in range(k - 1)])
        T   = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        ev  = np.linalg.eigvalsh(T)
        lam_min = float(max(ev[0],  1e-30))
        lam_max = float(ev[-1])
        return lam_max / lam_min, lam_min, lam_max
    except Exception:
        return None, None, None


def _iter_bound(kappa, res0, tol):
    """CG iteration bound: 2ρ^k = tol/res₀, ρ = (√κ−1)/(√κ+1)."""
    if kappa is None or kappa <= 1.0 or res0 <= tol:
        return None
    rho   = (kappa ** 0.5 - 1.0) / (kappa ** 0.5 + 1.0)
    ratio = tol / (2.0 * max(res0, 1e-300))
    if ratio >= 1.0 or rho <= 0.0:
        return None
    return max(1, int(math.ceil(math.log(ratio) / math.log(rho))))


def _rate_and_eta(res_history, res, tol, window=10):
    """Geometric-mean rate over last `window` iters and ETA from it."""
    n = min(len(res_history), window)
    if n < 2:
        return None, None
    r_new, r_old = res_history[-1], res_history[-n]
    if r_old <= 0 or r_new <= 0 or r_new >= r_old:
        return None, None
    rate = (r_new / r_old) ** (1.0 / n)
    if rate >= 1.0:
        return rate, None
    eta = max(0, int(math.ceil(
        math.log(tol / max(res, 1e-300)) / math.log(rate)
    )))
    return rate, eta


class VoxelFEMSolverNonlinear:
    """
    Nonlinear elastic FEM solver for voxel images.

    E = E(eps) via a ConstitutiveModel. Requires CuPy (GPU). Uses implicit
    connectivity — conn_dof is never stored.

    Parameters
    ----------
    voxels : np.ndarray (Nx, Ny, Nz) uint8
        Voxel array. Interpretation depends on `material_table`:
        - material_table=None  → voxel values are stiffness encoded as
          E = value/255 * E_max + E_void  (original behaviour)
        - material_table given → voxel values are material IDs (0–255);
          E for each element = material_table[voxel_value]
    nu : float — Poisson ratio (uniform)
    E_max : float — reference maximum modulus (used for multigrid scaling)
    E_void : float — modulus for void when material_table=None
    constitutive : ConstitutiveModel, optional
        Default: linear model (recovers linear elasticity)
    material_table : array-like of length 256, optional
        Young's modulus for each of the 256 possible material IDs.
        When provided, voxel values are treated as material IDs and E_void /
        E_max are ignored for E0 computation.  E_max is auto-set to the
        maximum value in the table unless explicitly passed.
    solid_mask : np.ndarray (Nx, Ny, Nz) bool, optional
        Binary solid/void map.  True → solid element carries load; False → void
        element contributes exactly zero to f_int and K·v.  Active nodes are
        those touched by ≥1 solid element; inactive DOFs receive identity rows
        so they stay decoupled without compacting the DOF numbering.
        None (default) preserves the original behaviour (all elements solid).
    is_linear : bool — if True, skip NR loop and run a single tight PCG
        solve.  Forces n_load_steps=1.  Use this for linear-elastic problems
        to avoid the overhead of the NR outer loop.
    tol_pcg : float — PCG tolerance used when is_linear=True (default 1e-8)
    n_load_steps : int — load increments (default 10; ignored when is_linear)
    max_nr_iter : int — max NR iterations per step (default 20)
    tol_nr : float — NR convergence: ||R||/f_norm < tol_nr (default 1e-6)
    max_pcg_iter : int — max PCG iterations per NR step (default 200)
    preconditioner : str — "multigrid" or "jacobi" (default "multigrid")
    gpu_chunk_size : int — elements per GPU chunk (default 2_000_000)
    verbose : bool — print NR/PCG progress (default True)
    """

    def __init__(
        self,
        voxels: np.ndarray,
        nu: float = 0.3,
        E_max: float = 1.0,
        E_void: float = 1e-6,
        constitutive: Optional[ConstitutiveModel] = None,
        material_table: Optional[np.ndarray] = None,
        solid_mask: Optional[np.ndarray] = None,
        is_linear: bool = False,
        tol_pcg: float = 1e-8,
        n_load_steps: int = 10,
        max_nr_iter: int = 20,
        tol_nr: float = 1e-6,
        max_pcg_iter: int = 200,
        preconditioner: str = "multigrid",
        gpu_chunk_size: int = 2_000_000,
        verbose: bool = True,
    ):
        assert voxels.ndim == 3 and voxels.dtype == np.uint8, \
            "voxels must be a 3D uint8 array"
        self.Nx, self.Ny, self.Nz = voxels.shape
        self.nu = nu
        self.E_void = E_void
        self.is_linear = is_linear
        self.tol_pcg = tol_pcg
        self.constitutive = constitutive or ConstitutiveModel(
            "linear", E_void=E_void, E_max=E_max
        )
        self.n_load_steps = 1 if is_linear else n_load_steps
        self.max_nr_iter = max_nr_iter
        self.tol_nr = tol_nr
        self.max_pcg_iter = max_pcg_iter
        self.preconditioner = preconditioner
        self.gpu_chunk_size = gpu_chunk_size
        self.verbose = verbose
        self.pcg_print_interval = 10
        self._kappa_probe = 20

        Ne = self.Nx * self.Ny * self.Nz
        Nn = (self.Nx + 1) * (self.Ny + 1) * (self.Nz + 1)
        self.Ndof = 3 * Nn
        self.Ne = Ne

        if verbose:
            print(f"Grid: {self.Nx}×{self.Ny}×{self.Nz} = {Ne:,} elements")
            print(f"Nodes: {Nn:,}   DOFs: {self.Ndof:,}")

        # Per-element base Young's modulus
        if material_table is not None:
            mat = np.asarray(material_table, dtype=np.float32)
            assert len(mat) == 256, "material_table must have exactly 256 entries"
            E0_cpu = mat[voxels.ravel()]          # lookup: material ID → E
            E_max  = float(mat.max()) if E_max == 1.0 else E_max
            if verbose:
                unique_ids = np.unique(voxels)
                print(f"Material table: {len(unique_ids)} distinct IDs in use, "
                      f"E range [{E0_cpu.min():.3g}, {E0_cpu.max():.3g}]")
        else:
            E_scale = (E_max - E_void) / 255.0
            E0_cpu = voxels.ravel().astype(np.float32) * E_scale + E_void

        self.E_max = E_max
        self.E0_elem_gpu = cp.asarray(E0_cpu, dtype=cp.float16)

        # Solid/void mask (Stage 1: binary void elimination).
        # solid_elem_gpu : (Ne,) bool CuPy — True for load-bearing elements.
        # inactive_dofs_gpu : (n_inactive,) int32 CuPy — DOFs not touched by
        #   any solid element; these receive identity rows throughout the solve.
        if solid_mask is not None:
            assert (solid_mask.shape == voxels.shape
                    and solid_mask.dtype == bool), \
                "solid_mask must be a bool array with the same shape as voxels"
            self.solid_elem_gpu = cp.asarray(solid_mask.ravel())
            active_dof_gpu = build_active_dof_gpu(
                self.solid_elem_gpu, self.Ny, self.Nz, self.Ndof,
                chunk_size=gpu_chunk_size,
            )
            inactive_np = np.where(~cp.asnumpy(active_dof_gpu))[0].astype(np.int32)
            self.inactive_dofs_gpu = cp.asarray(inactive_np)
            del active_dof_gpu, inactive_np
            if verbose:
                n_solid = int(cp.count_nonzero(self.solid_elem_gpu))
                n_inact = int(self.inactive_dofs_gpu.shape[0])
                print(f"Solid elements: {n_solid:,} / {Ne:,} "
                      f"({100*n_solid/Ne:.1f}%)")
                print(f"Inactive DOFs: {n_inact:,} / {self.Ndof:,} "
                      f"({100*n_inact/self.Ndof:.1f}%)")
        else:
            self.solid_elem_gpu = None
            self.inactive_dofs_gpu = None
        self._solid_mask_np = solid_mask   # kept for floating-island check

        # Precompute B_gauss matrices (CPU, then GPU)
        if verbose:
            print("Precomputing B_gauss ...", flush=True)
        B_cpu, w_cpu = build_B_gauss()
        self.B_gauss_gpu = cp.asarray(B_cpu.astype(np.float32))
        self.w_gauss_gpu = cp.asarray(w_cpu.astype(np.float32))

        # BC and external force storage (CPU)
        self.bc_mask = np.zeros(self.Ndof, dtype=bool)
        self.bc_values = np.zeros(self.Ndof, dtype=np.float32)
        self._f_ext = np.zeros(self.Ndof, dtype=np.float32)

        # Build preconditioner
        if preconditioner == "multigrid":
            if verbose:
                print("Building multigrid preconditioner ...", flush=True)
            self._precond = NonlinearMultigrid(
                voxels, nu, E_max, E_void, self.constitutive,
                self.B_gauss_gpu, self.E0_elem_gpu,
                chunk_size=gpu_chunk_size,
            )
        else:
            if verbose:
                print("Building Jacobi preconditioner ...", flush=True)
            self._precond = self._build_jacobi(voxels, E0_cpu)

    def _build_jacobi(self, voxels, E0_cpu):
        """Build diagonal Jacobi preconditioner from base moduli."""
        from geometry import (
            build_node_index, element_node_ids, build_K_ref, build_diagonal
        )
        K_ref = build_K_ref(self.nu)
        node_idx = build_node_index(self.Nx, self.Ny, self.Nz)
        conn_nodes = element_node_ids(node_idx, self.Nx, self.Ny, self.Nz)
        conn_dof = (conn_nodes[:, :, np.newaxis] * 3
                    + np.array([0, 1, 2], dtype=np.int32)).reshape(-1, 24)
        diag = build_diagonal(E0_cpu, conn_dof, K_ref, self.Ndof)
        diag = np.where(diag < 1e-12, 1.0, diag)
        diag_inv_gpu = cp.asarray((1.0 / diag).astype(np.float32))
        # Inactive DOFs (void-only nodes) must not be corrected by the
        # preconditioner — set their inverse diagonal to 1 (identity).
        if (self.inactive_dofs_gpu is not None
                and self.inactive_dofs_gpu.shape[0] > 0):
            diag_inv_gpu[self.inactive_dofs_gpu] = 1.0

        class _JacobiPrecond:
            def __init__(self, d):
                self._d = d
            def apply(self, r):
                return self._d * r
            def update(self, u_gpu, force_rebuild=True):
                pass  # Jacobi does not update

        return _JacobiPrecond(diag_inv_gpu)

    def apply_bc_displacement(self, node_ids, dof_ids, values):
        """Prescribe displacement on selected node DOFs (total values)."""
        global_dofs = np.asarray(node_ids) * 3 + np.asarray(dof_ids)
        self.bc_mask[global_dofs] = True
        self.bc_values[global_dofs] = np.asarray(values, dtype=np.float32)

    def apply_force(self, node_ids, dof_ids, values):
        """Accumulate nodal forces into the external force vector."""
        global_dofs = np.asarray(node_ids) * 3 + np.asarray(dof_ids)
        np.add.at(self._f_ext, global_dofs, np.asarray(values, dtype=np.float32))

    def solve(self):
        """
        Run Newton-Raphson with load stepping.

        Returns
        -------
        u : (Ndof,) float32 NumPy array — final displacement field
        """
        # Sparse BC storage: indices and values for constrained DOFs only.
        # Avoids uploading dense bc_values (Ndof) and bc_step (Ndof) arrays.
        # The full-Ndof bool mask (bc_mask_gpu) and its complement (free_gpu)
        # are never created — bc_dofs_gpu is used for all BC operations.
        bc_dofs_np = np.where(self.bc_mask)[0].astype(np.int32)
        bc_dofs_gpu = cp.asarray(bc_dofs_np)
        bc_vals_gpu = cp.asarray(self.bc_values[self.bc_mask].astype(np.float32))

        # Sparse external force storage (typically zero for displacement control).
        _has_forces = bool(np.any(self._f_ext != 0))
        if _has_forces:
            f_nz = np.where(self._f_ext != 0)[0].astype(np.int32)
            force_dofs_gpu = cp.asarray(f_nz)
            force_vals_gpu = cp.asarray(self._f_ext[f_nz].astype(np.float32))
        else:
            force_dofs_gpu = force_vals_gpu = None

        # Build combined constrained-DOF array: BC DOFs ∪ inactive (void) DOFs.
        # All these DOFs receive identity rows in the operator and are zeroed
        # in the preconditioner output, so PCG keeps them exactly at zero.
        if (self.inactive_dofs_gpu is not None
                and self.inactive_dofs_gpu.shape[0] > 0):
            all_constrained_gpu = cp.concatenate([bc_dofs_gpu,
                                                  self.inactive_dofs_gpu])
            all_constrained_np = np.concatenate(
                [bc_dofs_np,
                 cp.asnumpy(self.inactive_dofs_gpu).astype(np.int32)]
            )
        else:
            all_constrained_gpu = bc_dofs_gpu
            all_constrained_np = bc_dofs_np

        # Multigrid coarse operators are singular without the BCs that anchor
        # the rigid-body modes; hand the preconditioner the combined mask
        # (BC + inactive) so inactive DOFs are also treated as identity rows
        # at every multigrid level.
        if hasattr(self._precond, "set_bc_mask"):
            combined_mask_np = np.zeros(self.Ndof, dtype=bool)
            combined_mask_np[all_constrained_np] = True
            self._precond.set_bc_mask(combined_mask_np)

        # Store on self so _nr_step/_linear_step can access without a signature
        # change — this avoids breaking tests that monkey-patch _nr_step with
        # the original 7-argument signature.
        self._all_constrained_gpu = all_constrained_gpu

        # Warn about floating solid islands not anchored by any BC.
        if self._solid_mask_np is not None:
            self._check_floating_islands(bc_dofs_np)

        u_gpu = cp.zeros(self.Ndof, dtype=cp.float32)

        for step in range(1, self.n_load_steps + 1):
            lf = step / self.n_load_steps    # load fraction

            # Enforce displacement BCs at this load fraction (sparse scatter)
            u_gpu[bc_dofs_gpu] = lf * bc_vals_gpu

            if self.verbose:
                print(f"\n=== Load step {step}/{self.n_load_steps} "
                      f"(fraction={lf:.3f}) ===")

            u_gpu = self._nr_step(
                u_gpu, lf, bc_dofs_gpu, bc_vals_gpu,
                force_dofs_gpu, force_vals_gpu, _has_forces,
            )

        return cp.asnumpy(u_gpu)

    def _nr_step(self, u_gpu, lf, bc_dofs_gpu, bc_vals_gpu,
                 force_dofs_gpu, force_vals_gpu, _has_forces):
        """One load step: NR loop, or single PCG when is_linear=True."""
        all_constrained_gpu = self._all_constrained_gpu
        if self.is_linear:
            return self._linear_step(u_gpu, lf, bc_dofs_gpu, bc_vals_gpu,
                                     force_dofs_gpu, force_vals_gpu, _has_forces)


        R_norm_prev = None
        f_norm_ref = None

        for k in range(self.max_nr_iter):
            f_int = compute_f_int_cupy(
                u_gpu, self.E0_elem_gpu, self.B_gauss_gpu, self.w_gauss_gpu,
                self.nu, self.constitutive, self.Ny, self.Nz, self.Ndof,
                chunk_size=self.gpu_chunk_size,
                solid_mask_gpu=self.solid_elem_gpu,
            )

            # Zero constrained DOFs (BC + inactive) so their norms are excluded.
            f_int[all_constrained_gpu] = 0.0

            if f_norm_ref is None:
                f_norm_int = float(cp.linalg.norm(f_int).get())
                f_norm_ext = (lf * float(cp.linalg.norm(force_vals_gpu).get())
                              if _has_forces else 0.0)
                f_norm_ref = max(f_norm_ext, f_norm_int, 1e-30)

            # Build R = f_ext - f_int in-place in f_int's buffer (saves 1 Ndof alloc).
            cp.negative(f_int, out=f_int)           # f_int ← -f_int
            if _has_forces:
                cp.add.at(f_int, force_dofs_gpu, lf * force_vals_gpu)
            f_int[all_constrained_gpu] = 0.0       # re-zero after force scatter
            R = f_int
            del f_int                               # drop alias; buffer lives via R

            R_norm = float(cp.linalg.norm(R).get())
            rel_res = R_norm / f_norm_ref

            if self.verbose:
                print(f"  NR iter {k:3d}  ||R||/f_norm = {rel_res:.3e}")

            if rel_res < self.tol_nr:
                if self.verbose:
                    print(f"  NR converged at iter {k}")
                break

            # Eisenstat-Walker: loose inner tolerance early, tight near convergence
            eta = self._ew_tolerance(R_norm, R_norm_prev)

            # Update frozen-tangent multigrid for current u
            self._precond.update(u_gpu)

            def _matvec(v_gpu, _u=u_gpu):
                return matvec_tangent_cupy(
                    v_gpu, _u, self.E0_elem_gpu, self.B_gauss_gpu,
                    self.w_gauss_gpu, self.nu, self.constitutive,
                    self.Ny, self.Nz, self.Ndof,
                    chunk_size=self.gpu_chunk_size,
                    solid_mask_gpu=self.solid_elem_gpu,
                )

            def _apply_precond(r, _p=self._precond):
                return _p.apply(r)

            # Inner PCG solve: K_t(u) · Δu = R
            delta_u = self._pcg_gpu(
                _matvec, R, _apply_precond, all_constrained_gpu,
                tol=eta, max_iter=self.max_pcg_iter,
                verbose=self.verbose,
            )
            del R

            u_gpu += delta_u
            del delta_u
            # Enforce BC values (safeguard against floating-point drift)
            u_gpu[bc_dofs_gpu] = lf * bc_vals_gpu

            R_norm_prev = R_norm

        else:
            if self.verbose:
                print(f"  NR did not converge after {self.max_nr_iter} iterations")

        return u_gpu

    def _linear_step(self, u_gpu, lf, bc_dofs_gpu, bc_vals_gpu,
                     force_dofs_gpu, force_vals_gpu, _has_forces):
        """
        Single-pass linear solve: one PCG at tight tolerance, no NR loop.

        For a linear constitutive model (E_t = E0 everywhere), K_tangent is
        constant and the NR loop converges in exactly one step if the inner
        PCG is solved to tolerance.  This path skips the NR overhead entirely
        and uses tol_pcg directly instead of the Eisenstat-Walker criterion.
        """
        all_constrained_gpu = self._all_constrained_gpu
        f_int = compute_f_int_cupy(
            u_gpu, self.E0_elem_gpu, self.B_gauss_gpu, self.w_gauss_gpu,
            self.nu, self.constitutive, self.Ny, self.Nz, self.Ndof,
            chunk_size=self.gpu_chunk_size,
            solid_mask_gpu=self.solid_elem_gpu,
        )

        # Zero constrained DOFs (BC + inactive) so their norms are excluded.
        f_int[all_constrained_gpu] = 0.0
        f_norm_int = float(cp.linalg.norm(f_int).get())
        f_norm_ext = (lf * float(cp.linalg.norm(force_vals_gpu).get())
                      if _has_forces else 0.0)
        f_norm_ref = max(f_norm_int, f_norm_ext, 1e-30)

        # Build R = f_ext - f_int in-place in f_int's buffer.
        cp.negative(f_int, out=f_int)
        if _has_forces:
            cp.add.at(f_int, force_dofs_gpu, lf * force_vals_gpu)
        f_int[all_constrained_gpu] = 0.0       # re-zero after force scatter
        R = f_int
        del f_int

        R_norm = float(cp.linalg.norm(R).get())

        if self.verbose:
            print(f"  Linear solve: ||R|| = {R_norm:.3e}  "
                  f"(rel = {R_norm/f_norm_ref:.3e})  "
                  f"PCG tol = {self.tol_pcg:.1e}")

        # Build preconditioner once (no update needed — K is constant for linear)
        self._precond.update(u_gpu)

        def _matvec(v, _u=u_gpu):
            return matvec_tangent_cupy(
                v, _u, self.E0_elem_gpu, self.B_gauss_gpu,
                self.w_gauss_gpu, self.nu, self.constitutive,
                self.Ny, self.Nz, self.Ndof,
                chunk_size=self.gpu_chunk_size,
                solid_mask_gpu=self.solid_elem_gpu,
            )

        delta_u = self._pcg_gpu(
            _matvec, R, self._precond.apply, all_constrained_gpu,
            tol=self.tol_pcg, max_iter=self.max_pcg_iter,
            verbose=self.verbose,
        )
        del R

        u_gpu += delta_u
        del delta_u
        u_gpu[bc_dofs_gpu] = lf * bc_vals_gpu
        return u_gpu

    def _check_floating_islands(self, bc_dofs_np):
        """
        Detect solid islands not connected to any Dirichlet BC node.

        Floating islands carry rigid-body null-space modes that stall CG.
        Prints a warning listing island sizes; does not abort the solve.
        Requires scipy (skipped with a warning if unavailable).
        """
        try:
            from scipy import ndimage
        except ImportError:
            if self.verbose:
                print("WARNING: scipy not available; floating-island check skipped.")
            return

        solid_3d = self._solid_mask_np.reshape(self.Nx, self.Ny, self.Nz)
        # Use 26-connectivity: elements sharing even a corner share DOFs.
        labeled, n_comps = ndimage.label(solid_3d, structure=np.ones((3, 3, 3)))
        if n_comps == 0:
            return

        # Which nodes have at least one constrained DOF?
        Nn = (self.Nx + 1) * (self.Ny + 1) * (self.Nz + 1)
        constrained_node = np.zeros(Nn, dtype=bool)
        constrained_node[bc_dofs_np // 3] = True
        cn3d = constrained_node.reshape(self.Nx + 1, self.Ny + 1, self.Nz + 1)

        # An element is "anchored" iff any of its 8 corner nodes is constrained.
        ix = np.arange(self.Nx)[:, None, None]
        iy = np.arange(self.Ny)[None, :, None]
        iz = np.arange(self.Nz)[None, None, :]
        anchored = (
            cn3d[ix,   iy,   iz  ] | cn3d[ix+1, iy,   iz  ] |
            cn3d[ix,   iy+1, iz  ] | cn3d[ix+1, iy+1, iz  ] |
            cn3d[ix,   iy,   iz+1] | cn3d[ix+1, iy,   iz+1] |
            cn3d[ix,   iy+1, iz+1] | cn3d[ix+1, iy+1, iz+1]
        )  # (Nx, Ny, Nz) bool

        # Mark components that contain at least one anchored solid element.
        solid_and_anchored = solid_3d & anchored
        anchored_comps = set(np.unique(labeled[solid_and_anchored]))
        anchored_comps.discard(0)

        floating_sizes = []
        for cid in range(1, n_comps + 1):
            if cid not in anchored_comps:
                floating_sizes.append(int((labeled == cid).sum()))

        if floating_sizes:
            total = sum(floating_sizes)
            sizes_str = str(sorted(floating_sizes, reverse=True)[:5])
            if len(floating_sizes) > 5:
                sizes_str = sizes_str[:-1] + ", ...]"
            print(
                f"\nWARNING: {len(floating_sizes)} floating solid island(s) detected "
                f"(total {total} elements, sizes: {sizes_str}).\n"
                f"  Floating islands have rigid-body null-space modes and will stall CG.\n"
                f"  Consider removing specks below a size threshold or pinning their DOFs."
            )

    def _ew_tolerance(self, R_norm, R_norm_prev):
        """
        Eisenstat-Walker criterion: η_k = min(0.5, ||R_k|| / ||R_{k-1}||).

        Loose inner tolerance early → tight when NR nears convergence.
        Returns η in [tol_nr/10, 0.5].
        """
        if R_norm_prev is None:
            return 0.5
        eta = min(0.5, R_norm / (R_norm_prev + 1e-30))
        return max(eta, self.tol_nr / 10.0)

    def _pcg_gpu(self, matvec, b_gpu, precond, bc_dofs_gpu,
                 tol=1e-6, max_iter=200, verbose=False):
        """
        Preconditioned Conjugate Gradient with GPU-resident float32 vectors.

        r is explicitly zeroed at constrained DOFs after every update, so
        cp.dot(r, r) gives ||r_free||² without extracting a sub-vector.

        Parameters
        ----------
        matvec : callable (Ndof,)→(Ndof,) on GPU — applies K_tangent
        b_gpu  : (Ndof,) float32 CuPy — right-hand side (zeroed at BC DOFs).
                 Aliased as the initial residual — modified in-place during PCG.
                 Callers must not read b_gpu after this call returns.
        precond : callable (Ndof,)→(Ndof,) — preconditioner M^{-1}
        bc_dofs_gpu : (n_bc,) int32 CuPy — indices of constrained DOFs
        """
        x = cp.zeros(b_gpu.shape[0], dtype=cp.float32)

        def _Kfree(v):
            """K·v with identity rows for constrained DOFs (v must be float32)."""
            w = matvec(v)
            w[bc_dofs_gpu] = v[bc_dofs_gpu]
            return w

        def _Mfree(r):
            """M^{-1}·r, zeroed at constrained DOFs (returns float32)."""
            z = precond(r)
            if isinstance(z, np.ndarray):
                z = cp.asarray(z, dtype=cp.float32)
            z[bc_dofs_gpu] = 0.0
            return z

        # Alias b_gpu as the initial residual (x=0 → r=b). b_gpu already has
        # BC DOFs zeroed by the caller, so no copy or re-zeroing is needed.
        # r0 must be computed before the loop modifies r in-place.
        r = b_gpu
        r0 = float(cp.dot(r, r).get()) + 1e-30

        z = _Mfree(r)          # float32
        p = z                  # p is always float32 (z already is)
        rz = float(cp.dot(r, z).get())

        alphas:      list = []
        betas:       list = []
        res_history: list = []
        kappa_est        = None

        if verbose:
            print(f"    {'iter':>5}  {'residual':>11}  {'rate(10)':>9}  {'ETA':>6}")

        res = 1.0
        for it in range(max_iter):
            # p is always float32 — no upcast needed
            Ap = _Kfree(p)                              # float32
            pAp = float(cp.dot(p, Ap).get())
            alpha = rz / (pAp + 1e-30)
            alphas.append(alpha)

            x += alpha * p

            r -= alpha * Ap                              # float32 in-place
            del Ap
            r[bc_dofs_gpu] = 0.0

            z = _Mfree(r)                               # float32
            rz_new = float(cp.dot(r, z).get())

            # r is zeroed at BC DOFs → dot(r,r) = ||r_free||² without sub-vector
            res = float(cp.sqrt(cp.dot(r, r) / r0).get())
            res_history.append(res)

            # Condition number estimate after kappa_probe iterations
            if (kappa_est is None
                    and len(alphas) >= self._kappa_probe
                    and len(betas)  >= self._kappa_probe - 1
                    and verbose):
                kappa_est, lam_min, lam_max = _lanczos_kappa(
                    alphas[: self._kappa_probe],
                    betas [: self._kappa_probe - 1],
                )
                if kappa_est is not None:
                    k_bd = _iter_bound(kappa_est, res_history[0], tol)
                    bd_str = f"~{k_bd}" if k_bd is not None else "?"
                    print(f"    [κ ≈ {kappa_est:.2e}  "
                          f"√κ = {kappa_est**0.5:.1f}  "
                          f"λ ∈ [{lam_min:.2e}, {lam_max:.2e}]  "
                          f"CG bound: {bd_str} total iters]")

            if verbose and (it % self.pcg_print_interval == 0 or it < 3):
                rate, eta = _rate_and_eta(res_history, res, tol)
                r_str   = f"{rate:.4f}" if rate is not None else "       —"
                eta_str = str(eta)       if eta  is not None else "     —"
                print(f"    {it:>5}  {res:>11.3e}  {r_str:>9}  {eta_str:>6}")

            if res < tol:
                print(f"    PCG converged at iter {it},  residual = {res:.3e}")
                del z
                break

            beta = rz_new / (rz + 1e-30)
            betas.append(beta)
            # p stays float32: accumulate in z's buffer (no copy needed)
            z += beta * p
            p = z
            rz = rz_new
        else:
            print(f"    ⚠ PCG did NOT converge: {max_iter} iters, "
                  f"rel.residual = {res:.3e} (tol {tol:.1e})")

        return x
