"""
CuPy GPU kernels for the nonlinear voxel FEM solver.

All hot-path computation uses CuPy. numpy is only used for one-time setup
(build_B_gauss precomputation), then arrays are cast to float32 on GPU.

Key design: implicit connectivity — DOF indices are computed from element
indices on-the-fly; conn_dof is never stored.

For element e in (Nx, Ny, Nz) grid:
    sj = Nz+1,  si = (Ny+1)*sj
    i = e//(Ny*Nz), j = (e//Nz)%Ny, k = e%Nz
    n0 = i*si + j*sj + k
    node offsets: [0, si, si+sj, sj, 1, si+1, si+sj+1, sj+1]
"""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    raise RuntimeError(
        "CuPy is required for the nonlinear voxel FEM solver. "
        "Install with: pip install cupy-cuda12x"
    )

from geometry import _B_matrix

# 2×2×2 Gauss quadrature points and weights in [-1,1]
# ---------------------------------------------------------------------------
# NOTE: solid_mask_gpu (bool, shape (Ne,)) gates element contributions so that
# void elements contribute exactly zero to f_int / K·v. Pass None (default)
# to preserve the original behaviour (all elements active).
# ---------------------------------------------------------------------------
_GP = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
_GW = np.array([1.0, 1.0])


def build_B_gauss():
    """
    Precompute the 8 strain-displacement matrices for a unit cube element.

    These are pure geometry — identical for every element on a regular voxel
    grid. Computed once on CPU, transferred to GPU at startup.

    Returns
    -------
    B_gauss : np.ndarray (8, 6, 24) float64
        B matrix at each of the 8 Gauss points.
    w_gauss : np.ndarray (8,) float64
        Integration weight × detJ.  For 2×2×2 Gauss in [-1,1] on unit cube:
        detJ = 0.125,  w_i = 1.0  →  w_gauss[gp] = 0.125 for all gp.
    """
    B_gauss = np.zeros((8, 6, 24), dtype=np.float64)
    w_gauss = np.zeros(8, dtype=np.float64)

    gp_idx = 0
    for i, xi in enumerate(_GP):
        for j, eta in enumerate(_GP):
            for k, zeta in enumerate(_GP):
                B, detJ = _B_matrix(xi, eta, zeta)
                B_gauss[gp_idx] = B
                w_gauss[gp_idx] = _GW[i] * _GW[j] * _GW[k] * detJ
                gp_idx += 1

    return B_gauss, w_gauss


def build_active_dof_gpu(solid_mask_gpu, Ny, Nz, Ndof, chunk_size=2_000_000):
    """
    Build active-DOF mask: True for every DOF whose node is touched by ≥1 solid element.

    A node is active iff at least one of its 8 adjacent elements is solid.  DOFs
    belonging only to void-element nodes are *inactive* — they carry no load and
    should receive identity rows in the system operator.

    Parameters
    ----------
    solid_mask_gpu : (Ne,) bool CuPy
    Ny, Nz : int
    Ndof : int
    chunk_size : int

    Returns
    -------
    active_dof : (Ndof,) bool CuPy
    """
    Ne = int(solid_mask_gpu.shape[0])
    active_count = cp.zeros(Ndof, dtype=cp.int32)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        elem_ids = cp.arange(start, end, dtype=cp.int32)
        dofs = _get_elem_dofs(elem_ids, Ny, Nz)    # (chunk, 24)
        del elem_ids
        solid_chunk = solid_mask_gpu[start:end]     # (chunk,) bool
        solid_dofs = dofs[solid_chunk]              # (n_solid, 24) — empty if all void
        cp.add.at(active_count, solid_dofs.ravel(), 1)
        del dofs, solid_chunk, solid_dofs

    return active_count > 0


def _get_elem_dofs(elem_ids_gpu, Ny, Nz):
    """
    Compute DOF indices for a chunk of elements via implicit connectivity.

    Parameters
    ----------
    elem_ids_gpu : (chunk,) int32 CuPy array
    Ny, Nz : int

    Returns
    -------
    dofs : (chunk, 24) int32 CuPy array
    """
    sj = Nz + 1
    si = (Ny + 1) * sj

    i_e = elem_ids_gpu // (Ny * Nz)
    j_e = (elem_ids_gpu // Nz) % Ny
    k_e = elem_ids_gpu % Nz
    n0 = i_e * si + j_e * sj + k_e  # (chunk,)
    del i_e, j_e, k_e

    # Local node ordering matches geometry.element_node_ids:
    # node 0=(i,j,k) 1=(i+1,j,k) 2=(i+1,j+1,k) 3=(i,j+1,k)
    # node 4=(i,j,k+1) 5=(i+1,j,k+1) 6=(i+1,j+1,k+1) 7=(i,j+1,k+1)
    node_offsets = cp.array(
        [0, si, si + sj, sj, 1, si + 1, si + sj + 1, sj + 1],
        dtype=cp.int32
    )
    nodes = n0[:, None] + node_offsets[None, :]  # (chunk, 8)
    del n0

    # 3 DOFs per node, interleaved: [3n, 3n+1, 3n+2]
    dofs = nodes[:, :, None] * 3 + cp.arange(3, dtype=cp.int32)[None, None, :]
    del nodes
    return dofs.reshape(-1, 24)  # (chunk, 24)


def _build_D_ref(nu):
    """
    Build the unit-modulus isotropic D matrix, (6, 6) float32 CuPy array.

    D_t for a chunk = E_t[:, None, None] * D_ref[None, :, :], but callers
    in the hot path fuse E_t scaling into a matmul instead of materialising
    the (chunk, 6, 6) tensor.
    """
    c1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c3 = 1.0 / (2.0 * (1.0 + nu))
    return cp.array([
        [c1, c2, c2,  0,  0,  0],
        [c2, c1, c2,  0,  0,  0],
        [c2, c2, c1,  0,  0,  0],
        [ 0,  0,  0, c3,  0,  0],
        [ 0,  0,  0,  0, c3,  0],
        [ 0,  0,  0,  0,  0, c3],
    ], dtype=cp.float32)


def build_D_cupy(E_t, nu):
    """
    Build per-element (chunk, 6, 6) isotropic elasticity matrices.

    D[c] = E_t[c] * D_ref(nu),  where D_ref is the unit-modulus matrix.

    Parameters
    ----------
    E_t : (chunk,) float32 CuPy array
    nu : float

    Returns
    -------
    D : (chunk, 6, 6) float32 CuPy array
    """
    return E_t[:, None, None] * _build_D_ref(nu)[None, :, :]  # (chunk, 6, 6)


def von_mises_strain(eps):
    """
    Von Mises equivalent strain for Voigt strain vectors.

    eps_vm = sqrt(2/3 * dev(eps):dev(eps))

    Parameters
    ----------
    eps : (..., 6) CuPy array — Voigt [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_xz, gamma_yz]

    Returns
    -------
    eps_vm : (...,) CuPy array — equivalent strain
    """
    exx = eps[..., 0]
    eyy = eps[..., 1]
    ezz = eps[..., 2]
    gxy = eps[..., 3]
    gxz = eps[..., 4]
    gyz = eps[..., 5]

    emean = (exx + eyy + ezz) / 3.0
    # Deviatoric normal strains
    dxx = exx - emean
    dyy = eyy - emean
    dzz = ezz - emean
    # Engineering shear → tensor shear (divide by 2)
    exy = gxy / 2.0
    exz = gxz / 2.0
    eyz = gyz / 2.0

    return cp.sqrt(
        2.0 / 3.0 * (dxx**2 + dyy**2 + dzz**2
                     + 2.0 * (exy**2 + exz**2 + eyz**2))
    ).astype(cp.float32)


def compute_f_int_cupy(
    u_gpu,
    E0_elem_gpu,
    B_gauss_gpu,
    w_gauss_gpu,
    nu,
    constitutive,
    Ny, Nz,
    Ndof,
    chunk_size=2_000_000,
    solid_mask_gpu=None,
):
    """
    Compute internal force vector  f_int = Σ_e Σ_gp  w · B^T · σ(ε).

    Uses implicit connectivity and processes elements in chunks to fit VRAM.

    Memory per chunk (chunk_size=2M):
      u_local + f_loc: 2 × 2M × 24 × 4 = 384 MB
      eps + sigma: 2 × 2M × 6 × 4 = 96 MB per Gauss point (reused)
      D_t: 2M × 36 × 4 = 288 MB per Gauss point (reused)
      Peak: ~768 MB — well within VRAM budget.

    Parameters
    ----------
    u_gpu : (Ndof,) float32 CuPy
    E0_elem_gpu : (Ne,) float32 CuPy — base moduli from voxel image
    B_gauss_gpu : (8, 6, 24) float32 CuPy — precomputed B matrices
    w_gauss_gpu : (8,) float32 CuPy — Gauss weights × detJ
    nu : float — Poisson ratio
    constitutive : ConstitutiveModel
    Ny, Nz : int — grid dimensions
    Ndof : int — total DOFs
    chunk_size : int — elements per GPU chunk

    Returns
    -------
    f_int : (Ndof,) float32 CuPy
    """
    Ne = int(E0_elem_gpu.shape[0])
    f_int = cp.zeros(Ndof, dtype=cp.float32)

    # Build unit-modulus D_ref once (6×6) — avoids allocating (chunk,6,6) D_t
    D_ref = _build_D_ref(nu)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        chunk = end - start

        elem_ids = cp.arange(start, end, dtype=cp.int32)
        dofs = _get_elem_dofs(elem_ids, Ny, Nz)  # (chunk, 24)
        del elem_ids
        u_local = u_gpu[dofs]                     # (chunk, 24)
        E0_chunk = E0_elem_gpu[start:end].astype(cp.float32)  # (chunk,) — E0 stored as fp16

        f_loc = cp.zeros((chunk, 24), dtype=cp.float32)

        for gp in range(8):
            B = B_gauss_gpu[gp]          # (6, 24)
            w = float(w_gauss_gpu[gp])

            # Strain: eps[c,i] = Σ_j B[i,j] * u_local[c,j]
            eps = cp.einsum('ij,cj->ci', B, u_local)  # (chunk, 6)

            # Compute D_ref @ eps before eps_vm to reuse memory
            D_ref_eps = cp.einsum('ij,cj->ci', D_ref, eps)  # (chunk, 6)
            eps_vm = von_mises_strain(eps)                   # (chunk,)
            del eps
            E_t = constitutive.tangent_modulus(E0_chunk, eps_vm)  # (chunk,)
            del eps_vm

            # Cauchy stress: sigma = E_t * (D_ref @ eps)  — no (chunk,6,6) D_t
            sigma = E_t[:, None] * D_ref_eps  # (chunk, 6)
            del E_t, D_ref_eps

            # Internal force: f[c,j] += w * Σ_i B[i,j] * sigma[c,i]
            f_loc += w * (sigma @ B)  # (chunk, 24)
            del sigma

        # Zero contributions from void elements before scatter-adding
        if solid_mask_gpu is not None:
            solid_chunk = solid_mask_gpu[start:end]
            f_loc *= solid_chunk[:, None].astype(cp.float32)
            del solid_chunk

        # Atomic scatter-add to global f_int (handles repeated DOF indices)
        cp.add.at(f_int, dofs.ravel(), f_loc.ravel())
        del dofs, u_local, E0_chunk, f_loc

    return f_int


def matvec_tangent_cupy(
    v_gpu,
    u_gpu,
    E0_elem_gpu,
    B_gauss_gpu,
    w_gauss_gpu,
    nu,
    constitutive,
    Ny, Nz,
    Ndof,
    chunk_size=2_000_000,
    solid_mask_gpu=None,
):
    """
    Compute K_tangent(u) · v without assembling K_tangent.

    K_t(u) · v = Σ_e Σ_gp  w · B^T · D_t(ε_u) · B · v_local

    Both u_gpu and v_gpu are gathered for every chunk:
      u defines the tangent moduli (ε_u → D_t)
      v is the vector being multiplied

    Parameters
    ----------
    v_gpu : (Ndof,) float32 CuPy — vector to multiply
    u_gpu : (Ndof,) float32 CuPy — current displacement (defines K_tangent)
    ... (same as compute_f_int_cupy)

    Returns
    -------
    Kv : (Ndof,) float32 CuPy
    """
    Ne = int(E0_elem_gpu.shape[0])
    Kv = cp.zeros(Ndof, dtype=cp.float32)

    # Build unit-modulus D_ref once (6×6) — avoids allocating (chunk,6,6) D_t
    D_ref = _build_D_ref(nu)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        chunk = end - start

        elem_ids = cp.arange(start, end, dtype=cp.int32)
        dofs = _get_elem_dofs(elem_ids, Ny, Nz)
        del elem_ids
        u_local = u_gpu[dofs]   # (chunk, 24) — defines tangent moduli
        v_local = v_gpu[dofs]   # (chunk, 24) — vector to multiply
        E0_chunk = E0_elem_gpu[start:end].astype(cp.float32)  # E0 stored as fp16

        Kv_loc = cp.zeros((chunk, 24), dtype=cp.float32)

        for gp in range(8):
            B = B_gauss_gpu[gp]
            w = float(w_gauss_gpu[gp])

            # Tangent moduli from u at this Gauss point
            eps_u = cp.einsum('ij,cj->ci', B, u_local)    # (chunk, 6)
            eps_vm = von_mises_strain(eps_u)
            del eps_u
            E_t = constitutive.tangent_modulus(E0_chunk, eps_vm)
            del eps_vm

            # Strain from v (test vector)
            eps_v = cp.einsum('ij,cj->ci', B, v_local)    # (chunk, 6)

            # D_t @ eps_v = E_t * (D_ref @ eps_v)  — no (chunk,6,6) D_t
            D_ref_eps_v = cp.einsum('ij,cj->ci', D_ref, eps_v)  # (chunk, 6)
            del eps_v
            scaled = E_t[:, None] * D_ref_eps_v  # (chunk, 6)
            del E_t, D_ref_eps_v

            # B^T @ D_t @ B @ v
            Kv_loc += w * (scaled @ B)  # (chunk, 24)
            del scaled

        # Zero contributions from void elements before scatter-adding
        if solid_mask_gpu is not None:
            solid_chunk = solid_mask_gpu[start:end]
            Kv_loc *= solid_chunk[:, None].astype(cp.float32)
            del solid_chunk

        cp.add.at(Kv, dofs.ravel(), Kv_loc.ravel())
        del dofs, u_local, v_local, E0_chunk, Kv_loc

    return Kv


def matvec_Ku_cupy(u_gpu, E_elem_gpu, K_ref_gpu, Ny, Nz, Ndof,
                   chunk_size=2_000_000):
    """
    Linear matrix-free product  f = K·u  with a constant reference stiffness.

    K_e = E_elem[e] · K_ref (same K_ref for every element). This is the operator
    used by the geometric-multigrid levels — unlike ``matvec_tangent_cupy`` it
    does not evaluate the constitutive model per Gauss point. GPU-resident,
    implicit connectivity (conn_dof is never stored).

    Parameters
    ----------
    u_gpu : (Ndof,) float32 CuPy
    E_elem_gpu : (Ne,) float32 CuPy — per-element modulus (already scaled)
    K_ref_gpu : (24, 24) float32 CuPy — reference element stiffness
    Ny, Nz : int — grid dimensions of this level
    Ndof : int
    chunk_size : int

    Returns
    -------
    f : (Ndof,) float32 CuPy
    """
    Ne = int(E_elem_gpu.shape[0])
    f = cp.zeros(Ndof, dtype=cp.float32)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)

        elem_ids = cp.arange(start, end, dtype=cp.int32)
        dofs = _get_elem_dofs(elem_ids, Ny, Nz)            # (chunk, 24)
        del elem_ids
        u_loc = u_gpu[dofs]                                # (chunk, 24)

        # K_ref · u_loc per element, then scale by E_elem
        Ku_loc = cp.einsum('ij,cj->ci', K_ref_gpu, u_loc)  # (chunk, 24)
        del u_loc
        Ku_loc *= E_elem_gpu[start:end, None]

        cp.add.at(f, dofs.ravel(), Ku_loc.ravel())
        del dofs, Ku_loc

    return f


def build_diagonal_cupy(E_elem_gpu, diag_ref_gpu, Ny, Nz, Ndof,
                        chunk_size=2_000_000):
    """
    Accumulate diag(K) on the GPU for the Jacobi smoother.

    diag(K_e) = E_elem[e] · diag(K_ref). Implicit connectivity, GPU-resident.

    Parameters
    ----------
    E_elem_gpu : (Ne,) float32 CuPy
    diag_ref_gpu : (24,) float32 CuPy — diag(K_ref)
    Ny, Nz : int
    Ndof : int
    chunk_size : int

    Returns
    -------
    diag : (Ndof,) float32 CuPy
    """
    Ne = int(E_elem_gpu.shape[0])
    diag = cp.zeros(Ndof, dtype=cp.float32)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)

        elem_ids = cp.arange(start, end, dtype=cp.int32)
        dofs = _get_elem_dofs(elem_ids, Ny, Nz)               # (chunk, 24)
        del elem_ids
        contrib = E_elem_gpu[start:end, None] * diag_ref_gpu[None, :]
        cp.add.at(diag, dofs.ravel(), contrib.ravel())
        del dofs, contrib

    return diag


def compute_avg_strain_per_element(u_gpu, B_gauss_gpu, Ny, Nz, Ne,
                                   chunk_size=2_000_000):
    """
    Compute average von Mises strain per element (average over 8 Gauss points).

    Used to build the frozen-tangent multigrid hierarchy.

    Parameters
    ----------
    u_gpu : (Ndof,) float32 CuPy
    B_gauss_gpu : (8, 6, 24) float32 CuPy
    Ny, Nz : int
    Ne : int
    chunk_size : int

    Returns
    -------
    eps_vm_avg : (Ne,) float32 CuPy
    """
    eps_vm_avg = cp.zeros(Ne, dtype=cp.float32)

    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        chunk = end - start

        elem_ids = cp.arange(start, end, dtype=cp.int32)
        dofs = _get_elem_dofs(elem_ids, Ny, Nz)
        del elem_ids
        u_local = u_gpu[dofs]  # (chunk, 24)

        eps_vm_sum = cp.zeros(chunk, dtype=cp.float32)
        for gp in range(8):
            B = B_gauss_gpu[gp]
            eps = cp.einsum('ij,cj->ci', B, u_local)
            eps_vm_sum += von_mises_strain(eps)  # in-place: avoids new allocation
            del eps

        eps_vm_avg[start:end] = eps_vm_sum / 8.0
        del dofs, u_local, eps_vm_sum

    return eps_vm_avg
