"""
Analytical reference solutions — layered composite.
====================================================

Pure Python / NumPy, no FEM-solver dependency.  This module can be imported and
verified on its own (``test_reference.py``).

Only the **layered composite** benchmark is implemented here.

Voxel-encoding note
-------------------
``VoxelFEMSolverNonlinear`` maps a ``uint8`` voxel value ``v`` to a per-element
Young's modulus by

    E = v / 255 * (E_max - E_void) + E_void           (material_table=None)

``make_layered_voxels`` returns the chosen voxel values so that tests can
reconstruct the *actual* moduli the solver will use and feed them to
``layered_reference``.  This removes uint8 quantisation from the FEM-vs-analytic
comparison, leaving only discretisation / boundary error.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1a — Layered composite reference
# ---------------------------------------------------------------------------

def layered_reference(E1, E2, f1, nu):
    """
    Exact effective moduli for a two-phase layered composite.

    Parameters
    ----------
    E1, E2 : float — Young's moduli of the two phases
    f1     : float — volume fraction of phase 1  (f2 = 1 - f1)
    nu     : float — Poisson ratio (same for both phases)

    Returns
    -------
    dict with keys:
      'E_eff_normal'   : effective modulus, loading normal to the layers
                         (series / harmonic-mean arrangement)
      'E_eff_parallel' : effective modulus, loading parallel to the layers
                         (parallel / arithmetic-mean arrangement)
      'sigma_1_normal' : stress in phase 1 under unit far-field strain,
                         normal loading (equal in both phases by equilibrium)
      'sigma_2_normal' : stress in phase 2, normal loading
      'eps_1_parallel' : strain in phase 1 under unit far-field strain,
                         parallel loading (equal in both phases by compatibility)
      'eps_2_parallel' : strain in phase 2, parallel loading
    """
    f2 = 1.0 - f1

    # Series (normal to layers): harmonic mean.
    E_eff_normal = 1.0 / (f1 / E1 + f2 / E2)

    # Parallel (in-plane): arithmetic mean.
    E_eff_parallel = f1 * E1 + f2 * E2

    # Normal loading under unit far-field STRAIN (eps_inf = 1):
    #   equilibrium => sigma is continuous across the interface, so both phases
    #   carry the same stress.  The volume-averaged stress equals E_eff_normal.
    sigma_normal = E_eff_normal          # identical in both phases

    # Parallel loading under unit far-field STRAIN (eps_inf = 1):
    #   compatibility => both phases share the same strain.
    return {
        'E_eff_normal':   E_eff_normal,
        'E_eff_parallel': E_eff_parallel,
        'sigma_1_normal': sigma_normal,
        'sigma_2_normal': sigma_normal,
        'eps_1_parallel': 1.0,
        'eps_2_parallel': 1.0,
    }


# ---------------------------------------------------------------------------
# 2a — Layered composite voxel image
# ---------------------------------------------------------------------------

def voxel_to_E(value, E_max, E_void):
    """Modulus a voxel value maps to under the solver's default encoding."""
    return value / 255.0 * (E_max - E_void) + E_void


def make_layered_voxels(Nx, Ny, Nz, layer_axis, E1, E2, f1, E_max, E_void=1e-6):
    """
    Build a (Nx,Ny,Nz) uint8 voxel array with two alternating slabs of E1, E2.

    Parameters
    ----------
    Nx, Ny, Nz : int — grid dimensions
    layer_axis : int — 0=x, 1=y, 2=z — axis NORMAL to the layers
    E1, E2     : float — Young's moduli of the two phases
    f1         : float — volume fraction of phase 1 (rounded to a whole slab)
    E_max      : float — modulus corresponding to voxel value 255
    E_void     : float — modulus offset at voxel value 0 (solver default 1e-6)

    Returns
    -------
    voxels    : np.ndarray (Nx,Ny,Nz) uint8
    info      : dict with the *actual* discretised quantities the solver sees:
                'f1_actual', 'val1', 'val2', 'E1_actual', 'E2_actual'
    """
    N = [Nx, Ny, Nz][layer_axis]
    n1 = int(np.clip(round(f1 * N), 1, N - 1))   # at least one slab of each phase
    n2 = N - n1
    f1_actual = n1 / N

    val1 = int(np.clip(round(E1 / E_max * 255), 1, 255))
    val2 = int(np.clip(round(E2 / E_max * 255), 1, 255))

    stripe = np.array([val1] * n1 + [val2] * n2, dtype=np.uint8)   # (N,)

    voxels = np.empty((Nx, Ny, Nz), dtype=np.uint8)
    if layer_axis == 0:
        voxels[:, :, :] = stripe[:, np.newaxis, np.newaxis]
    elif layer_axis == 1:
        voxels[:, :, :] = stripe[np.newaxis, :, np.newaxis]
    else:
        voxels[:, :, :] = stripe[np.newaxis, np.newaxis, :]

    info = {
        'f1_actual': f1_actual,
        'val1': val1,
        'val2': val2,
        'E1_actual': voxel_to_E(val1, E_max, E_void),
        'E2_actual': voxel_to_E(val2, E_max, E_void),
    }
    return voxels, info
