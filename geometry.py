# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
"""
Voxel FEM Geometry & Matrix-Free Primitives
============================================
Pure, backend-agnostic building blocks shared by the nonlinear voxel FEM
solver and its preconditioners/tests:

  - Reference element stiffness for a unit-cube Q1 hex (`build_K_ref`)
  - Q1 shape-function gradients and strain-displacement matrix
    (`_shape_grads`, `_B_matrix`)
  - DOF/connectivity helpers (`build_node_index`, `element_node_ids`)
  - Matrix-free K·u product and diagonal assembly (`matvec_Ku`,
    `build_diagonal`) — NumPy reference implementations
  - Preconditioned Conjugate Gradient (`pcg`)

These functions contain no solver state and no GPU dependency. The nonlinear
solver imports the geometry routines (`_B_matrix`, `build_node_index`) from
here; the linear multigrid preconditioner and the test suite reuse the rest as
a CPU reference.

Element geometry: every voxel is a unit cube. Gauss points come in on
[-1,1]³; `_B_matrix` maps them to [0,1]³ internally, giving detJ = 0.125.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  Reference element stiffness for a unit cube (Q1 trilinear hex)
# ---------------------------------------------------------------------------

def build_K_ref(nu: float) -> np.ndarray:
    """
    Compute the 24×24 reference element stiffness matrix for a unit cube
    with E=1, using full 2×2×2 Gauss quadrature.

    DOF ordering: node 0 (u0,v0,w0), node 1 (u1,v1,w1), ..., node 7 (u7,v7,w7)
    Node numbering (local):
        0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
        4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
    """
    # Gauss points and weights for [-1,1]
    gp = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    gw = np.array([1.0, 1.0])

    # Material matrix D (plane-independent isotropic linear elasticity)
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
    ])  # 6×6

    K = np.zeros((24, 24))

    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):
            for k, zeta in enumerate(gp):
                w = gw[i] * gw[j] * gw[k]
                B, detJ = _B_matrix(xi, eta, zeta)   # 6×24, scalar
                K += w * detJ * (B.T @ D @ B)

    return K  # (24, 24)


def _shape_grads(xi: float, eta: float, zeta: float) -> np.ndarray:
    """
    Gradients of Q1 shape functions w.r.t. (xi, eta, zeta) — 3×8 matrix.
    Node ordering matches build_K_ref docstring.
    """
    # Shape functions in [-1,1]^3 mapped to [0,1]^3
    # We use (xi,eta,zeta) in [0,1] directly (unit cube, J = I/2... actually J=0.5*I)
    # Using natural coords ξ,η,ζ ∈ [0,1]:
    xi1, eta1, zeta1 = 1 - xi, 1 - eta, 1 - zeta

    # dN/dxi  (row 0), dN/deta (row 1), dN/dzeta (row 2)
    dN = np.array([
        [-eta1*zeta1,  eta1*zeta1,  eta*zeta1, -eta*zeta1,
         -eta1*zeta,   eta1*zeta,   eta*zeta,  -eta*zeta],
        [-xi1*zeta1,  -xi*zeta1,    xi*zeta1,   xi1*zeta1,
         -xi1*zeta,   -xi*zeta,     xi*zeta,    xi1*zeta],
        [-xi1*eta1,   -xi*eta1,    -xi*eta,    -xi1*eta,
          xi1*eta1,    xi*eta1,     xi*eta,     xi1*eta],
    ])   # (3, 8)
    return dN


def _B_matrix(xi: float, eta: float, zeta: float):
    """
    Strain-displacement matrix B (6×24) and detJ for unit cube.

    Gauss points xi,eta,zeta come in on [-1,1].  We map to physical
    coords xi_ ∈ [0,1] via xi_ = (xi+1)/2.  _shape_grads returns
    ∂N/∂xi_ = ∂N/∂x directly (unit cube, J_phys = I).  detJ = 0.125
    is the Jacobian of the [-1,1]→[0,1] quadrature mapping.
    """
    # Map Gauss coords [-1,1] → physical coords [0,1]
    xi_    = 0.5 * (xi + 1.0)
    eta_   = 0.5 * (eta + 1.0)
    zeta_  = 0.5 * (zeta + 1.0)
    detJ   = 0.125   # (0.5)^3 — volume ratio of the mapping above

    # _shape_grads already returns ∂N/∂x (physical coords); no extra factor needed
    dN = _shape_grads(xi_, eta_, zeta_)   # (3, 8)

    B = np.zeros((6, 24))
    for a in range(8):
        col = 3 * a
        B[0, col]     = dN[0, a]
        B[1, col+1]   = dN[1, a]
        B[2, col+2]   = dN[2, a]
        B[3, col]     = dN[1, a];  B[3, col+1] = dN[0, a]
        B[4, col+1]   = dN[2, a];  B[4, col+2] = dN[1, a]
        B[5, col]     = dN[2, a];  B[5, col+2] = dN[0, a]

    return B, detJ


# ---------------------------------------------------------------------------
# 2.  DOF mapping helpers
# ---------------------------------------------------------------------------

def build_node_index(Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """
    Global node index for node (i,j,k):  idx = i*(Ny+1)*(Nz+1) + j*(Nz+1) + k
    Returns shape (Nx+1, Ny+1, Nz+1) int32 array.
    """
    Nn = (Nx + 1) * (Ny + 1) * (Nz + 1)
    return np.arange(Nn, dtype=np.int32).reshape(Nx + 1, Ny + 1, Nz + 1)


def element_node_ids(node_idx: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """
    For each element (i,j,k), return its 8 node global IDs.
    Returns shape (Nx*Ny*Nz, 8) int32.

    Local node order matches build_K_ref:
        0=(i,j,k)   1=(i+1,j,k)   2=(i+1,j+1,k)   3=(i,j+1,k)
        4=(i,j,k+1) 5=(i+1,j,k+1) 6=(i+1,j+1,k+1) 7=(i,j+1,k+1)
    """
    i = node_idx[:Nx, :Ny, :Nz]
    conn = np.stack([
        i,
        node_idx[1:Nx+1, :Ny,   :Nz  ],
        node_idx[1:Nx+1, 1:Ny+1, :Nz  ],
        node_idx[:Nx,    1:Ny+1, :Nz  ],
        node_idx[:Nx,    :Ny,    1:Nz+1],
        node_idx[1:Nx+1, :Ny,    1:Nz+1],
        node_idx[1:Nx+1, 1:Ny+1, 1:Nz+1],
        node_idx[:Nx,    1:Ny+1, 1:Nz+1],
    ], axis=-1)   # (Nx, Ny, Nz, 8)
    return conn.reshape(-1, 8)   # (Ne, 8)


# ---------------------------------------------------------------------------
# 3.  Matrix-free K·u product (NumPy reference kernel)
# ---------------------------------------------------------------------------

def matvec_Ku(
    u: np.ndarray,           # (Ndof,)  float32
    E_elem: np.ndarray,      # (Ne,)    float32  — per-element Young's modulus
    conn_dof: np.ndarray,    # (Ne, 24) int32    — element DOF indices
    K_ref: np.ndarray,       # (24, 24) float64
    Ndof: int,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """
    Compute f = K·u without assembling K.

    Vectorized over elements in chunks to balance memory vs speed.

    Memory per chunk (chunk_size=500k):
      u_local: 500k × 24 × 4 bytes = 48 MB
      f_local: same
    """
    f = np.zeros(Ndof, dtype=np.float32)
    K_ref_f = K_ref.astype(np.float32)   # keep on stack once

    Ne = E_elem.shape[0]
    for start in range(0, Ne, chunk_size):
        end = min(start + chunk_size, Ne)
        sl = slice(start, end)

        dofs = conn_dof[sl]           # (chunk, 24)
        u_loc = u[dofs]               # (chunk, 24)  — gather

        # (chunk, 24) = (chunk, 1) * (chunk, 24) where K_ref·u_loc uses einsum
        # K_ref @ u_loc[e]  for each e:  result shape (chunk, 24)
        Ku_loc = np.einsum('ij,kj->ki', K_ref_f, u_loc)   # (chunk, 24)
        Ku_loc *= E_elem[sl, np.newaxis]                    # scale by E

        # Scatter-add  (np.add.at is unbuffered accumulation)
        np.add.at(f, dofs, Ku_loc)

    return f


# ---------------------------------------------------------------------------
# 4.  Jacobi preconditioner (diagonal of K)
# ---------------------------------------------------------------------------

def build_diagonal(
    E_elem: np.ndarray,
    conn_dof: np.ndarray,
    K_ref: np.ndarray,
    Ndof: int,
) -> np.ndarray:
    """Accumulate diagonal of K (needed for Jacobi preconditioner)."""
    diag_ref = np.diag(K_ref).astype(np.float32)   # (24,)
    diag = np.zeros(Ndof, dtype=np.float32)
    for start in range(0, len(E_elem), 500_000):
        end = min(start + 500_000, len(E_elem))
        sl  = slice(start, end)
        dofs = conn_dof[sl]                                   # (chunk, 24)
        contributions = E_elem[sl, None] * diag_ref[None, :]  # (chunk, 24)
        np.add.at(diag, dofs, contributions)
    return diag


# ---------------------------------------------------------------------------
# 5.  Preconditioned Conjugate Gradient
# ---------------------------------------------------------------------------

def pcg(
    matvec,          # callable: u -> K·u
    b: np.ndarray,   # RHS, (Ndof,)
    M_inv,           # preconditioner: array or callable (Ndof,) or callable
    bc_mask: np.ndarray,  # bool (Ndof,) — True = constrained DOF
    x0: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
    verbose: bool = True,
) -> np.ndarray:
    """
    PCG with Jacobi or GMG preconditioner and essential BC enforcement via
    zeroing out constrained DOF rows/columns (penalty-free approach).

    M_inv can be either:
      - numpy array (Ndof,): diagonal preconditioner inverse
      - callable: function taking r -> z where z = M^{-1} r
    """
    Ndof = b.shape[0]
    x = np.zeros(Ndof, dtype=np.float32) if x0 is None else x0.copy()

    # Enforce BCs: constrained DOFs stay fixed
    free = ~bc_mask

    def _matvec_free(v):
        """Apply K only on free DOFs; constrained DOFs treated as identity."""
        w = matvec(v)
        w[bc_mask] = v[bc_mask]   # identity rows for constrained DOFs
        return w

    def _apply_precond(r):
        """Apply preconditioner: either array or callable."""
        if callable(M_inv):
            z = M_inv(r)
        else:
            z = M_inv * r
        z[bc_mask] = 0.0
        return z

    r = b - _matvec_free(x)
    r[bc_mask] = 0.0

    z = _apply_precond(r)
    p = z.copy()
    rz = float(np.dot(r, z))
    r0 = float(np.dot(b[free], b[free])) + 1e-30

    for it in range(max_iter):
        Ap = _matvec_free(p)
        pAp = float(np.dot(p, Ap))
        alpha = rz / (pAp + 1e-30)

        x += alpha * p
        r -= alpha * Ap
        r[bc_mask] = 0.0

        z = _apply_precond(r)
        rz_new = float(np.dot(r, z))
        res = float(np.sqrt(np.dot(r[free], r[free]) / r0))

        if verbose and (it % 10 == 0 or it < 5):
            print(f"  CG iter {it:4d}  rel.residual = {res:.3e}")

        if res < tol:
            print(f"  Converged at iter {it}, residual = {res:.3e}")
            break

        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new

    return x
