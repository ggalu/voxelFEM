# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-04 21:15:10
# @Modified by: Lokesh Metta, Albert-Ludwigs Universitaet Freiburg, Germany
# @Modified:     2026-09-25 (comments: use warp_cupy.warp_lagrangian for ground-truth images)
#!/usr/bin/env python
"""
Voxel FEM simulation of a CT-segmented sample
==============================================
Loads a preprocessed ``.npz`` file (keys: ``labels``, ``Young_modulus``; shape N×M×K),
where Young's modulus has been precomputed for each voxel, and runs a single
**linear-elastic** solve under 1% uniaxial tensile strain. Default loading axis
is **z**.

Affine initialisation
---------------------
The displacement field is seeded with the homogeneous affine solution
``u_a = ε·x_a``, ``u_b = u_c = −ν·ε·x_{b,c}`` before the PCG solve starts.
This is the exact solution for a homogeneous material and a good first guess
for heterogeneous media, typically halving the required PCG iterations.

Void/air elimination
--------------------
Labels 0 (void/background) and 255 (air) are treated as empty space and
**removed from the system** via a binary solid mask. Only labels 1 (matrix)
and 2 (fibre) carry load. Void nodes receive identity rows in the operator
so the DOF vector stays at full-grid length (implicit connectivity preserved)
and inactive DOFs stay exactly zero throughout the solve.

Boundary conditions
-------------------
Tensile strain ``ε`` is applied along the chosen axis by prescribing the
displacement of both end-faces in that direction. The remaining four faces are
left free so the body contracts according to Poisson's ratio (ν=0.3); three
minimal "3-2-1" anchor constraints remove residual rigid-body translations/
rotation without restraining the lateral faces. Because every voxel is a unit
cube the domain length equals N_axis, so Δ = ε·N_axis.

Usage
-----
Edit the parameters in ``simulateCT.json``, then run:

    python simulateCT.py

Example ``simulateCT.json``:

    {
      "data": "assets/PA66GF30.npz",
      "axis": "x",
      "strain": 0.01,
      "nu": 0.3,
      "maxiter": 100,
      "preconditioner": "multigrid",
      "smoother": "chebyshev",
      "vtk": "ct_result"
    }
"""

import json
import os
import sys
import time
import numpy as np
import cupy as cp

from geometry import build_node_index
from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
from kernels_cupy_nl import compute_f_int_cupy, build_B_gauss

# Reuse the validation package's post-processing helpers. Imported via a
# sys.path insert (not `from tests.validation.report import ...`) because a
# `tests` package may already be installed in site-packages, which shadows
# this repo's `tests/` (a namespace package, since it has no __init__.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests", "validation"))
from report import element_stress, element_strain, build_conn_dof  # noqa: E402


# Labels treated as void/air: removed from the FEM system via solid_mask.
VOID_LABELS = {0, 255}

_AXIS_NAMES = ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Boundary conditions: uniaxial tension along `axis`, lateral faces free
# ---------------------------------------------------------------------------

def apply_uniaxial_bc(solver, node_idx, axis, delta):
    """
    Prescribe displacement on the two end-faces along `axis` (0=x,1=y,2=z),
    leave the remaining four faces free, and apply a minimal 3-2-1 anchor.

    The 3-2-1 anchor fixes:
      - transverse DOF b at the origin node            (kills b-translation)
      - transverse DOF c at the origin node            (kills c-translation)
      - transverse DOF c at the max-b corner node      (kills rotation about axis)
    """
    tr = [d for d in range(3) if d != axis]   # two transverse axes
    b, c = tr[0], tr[1]

    sl_left  = [slice(None), slice(None), slice(None)]
    sl_right = [slice(None), slice(None), slice(None)]
    sl_left[axis]  = 0
    sl_right[axis] = -1
    left  = node_idx[tuple(sl_left)].ravel()
    right = node_idx[tuple(sl_right)].ravel()

    solver.apply_bc_displacement(left,  np.full(len(left),  axis), np.zeros(len(left)))
    solver.apply_bc_displacement(right, np.full(len(right), axis), np.full(len(right), delta))

    origin = int(node_idx[0, 0, 0])
    solver.apply_bc_displacement(np.array([origin, origin]),
                                 np.array([b, c]), np.array([0.0, 0.0]))

    # Fix c at the max-b corner to kill rotation about `axis`
    idx_bmax = [0, 0, 0]
    idx_bmax[b] = -1
    bmax_node = int(node_idx[tuple(idx_bmax)])
    solver.apply_bc_displacement(np.array([bmax_node]), np.array([c]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Affine homogeneous initial displacement field
# ---------------------------------------------------------------------------

def build_affine_u0(Nx, Ny, Nz, strain, nu, axis):
    """
    Build the exact displacement field for a homogeneous uniaxial strain ε:

        u_axis(node)        =  ε · coord_axis
        u_transverse(node)  = −ν · ε · coord_transverse

    This is the exact solution for a uniform material and a good first guess
    for heterogeneous media, reducing the initial residual substantially.

    Built directly on the GPU (CuPy) — for large grids the host-side
    `np.mgrid` + assembly is a multi-second, single-threaded CPU step that
    runs while the GPU sits idle.
    """
    Ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
    u0 = cp.zeros(Ndof, dtype=cp.float32)

    # C-order node indexing: flat node i has coordinates given by mgrid
    ii, jj, kk = cp.meshgrid(cp.arange(Nx + 1, dtype=cp.float32),
                             cp.arange(Ny + 1, dtype=cp.float32),
                             cp.arange(Nz + 1, dtype=cp.float32),
                             indexing='ij')
    coords = [ii.ravel(), jj.ravel(), kk.ravel()]

    u0[axis::3] = strain * coords[axis]
    for d in range(3):
        if d != axis:
            u0[d::3] = -nu * strain * coords[d]

    return u0


# ---------------------------------------------------------------------------
# Effective modulus from the reaction force on the loaded end-face
# ---------------------------------------------------------------------------

def effective_modulus(u, E_voxels, Nx, Ny, Nz, nu, axis, strain,
                      solid_mask=None):
    """
    E_eff = (F_axis / A_total) / strain, where F_axis is the summed reaction
    on the far face and A_total is the cross-sectional area perpendicular to
    `axis` (unit-cube voxels, full face).
    """
    from constitutive import ConstitutiveModel
    mat = ConstitutiveModel("linear", E_void=float(E_voxels.min()),
                            E_max=float(E_voxels.max()))
    E0_gpu = cp.asarray(E_voxels.ravel().astype(np.float32))
    B_cpu, w_cpu = build_B_gauss()
    B_gpu = cp.asarray(B_cpu.astype(np.float32))
    w_gpu = cp.asarray(w_cpu.astype(np.float32))
    Ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)

    solid_mask_gpu = (cp.asarray(solid_mask.ravel())
                      if solid_mask is not None else None)

    f_int = compute_f_int_cupy(cp.asarray(u), E0_gpu, B_gpu, w_gpu,
                               nu, mat, Ny, Nz, Ndof,
                               solid_mask_gpu=solid_mask_gpu)
    f_int_np = cp.asnumpy(f_int)

    node_idx = build_node_index(Nx, Ny, Nz)
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = -1
    far_face = node_idx[tuple(sl)].ravel()
    F_reaction = f_int_np[3 * far_face + axis].sum()

    dims = [Nx, Ny, Nz]
    area = dims[(axis + 1) % 3] * dims[(axis + 2) % 3]
    return F_reaction / area / strain, F_reaction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load parameters from JSON file
    with open("simulateCT.json", "r") as f:
        config = json.load(f)

    # Extract and remove help documentation (if present)
    help_doc = config.pop("_help", {})

    # Validate and set defaults
    config.setdefault("data", "exp_data.npz")
    config.setdefault("axis", "z")
    config.setdefault("strain", 0.01)
    config.setdefault("nu", 0.3)
    config.setdefault("maxiter", 100)
    config.setdefault("preconditioner", "multigrid")
    config.setdefault("smoother", "chebyshev")
    config.setdefault("vtk", "ct_result")

    args = config

    # Validate axis choice
    if args["axis"] not in ["x", "y", "z"]:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{args['axis']}'")

    # Validate preconditioner choice
    if args["preconditioner"] not in ["jacobi", "multigrid"]:
        raise ValueError(f"preconditioner must be 'jacobi' or 'multigrid', got '{args['preconditioner']}'")

    # Validate smoother choice
    if args["smoother"] not in ["jacobi", "chebyshev"]:
        raise ValueError(f"smoother must be 'jacobi' or 'chebyshev', got '{args['smoother']}'")

    axis = {"x": 0, "y": 1, "z": 2}[args["axis"]]
    aname = args["axis"]

    print("=" * 70)
    print(f"  VOXEL FEM — CT data, linear elasticity, {aname}-tension")
    print("=" * 70)

    # --- Load data -----------------------------------------------------------
    data = np.load(args["data"])
    voxels_labels = data["labels"].astype(np.uint8)
    E_voxels = data["Young_modulus"].astype(np.float32)
    Nx, Ny, Nz = E_voxels.shape
    dims = [Nx, Ny, Nz]
    labels = np.unique(voxels_labels)
    print(f"  Data  : {args['data']}")
    print(f"  Grid  : {Nx}×{Ny}×{Nz} = {Nx*Ny*Nz:,} voxels")
    print(f"  Labels present : {labels.tolist()}")
    print(f"  Young's modulus range: [{E_voxels.min():.4f}, {E_voxels.max():.4f}] GPa")

    # --- Solid/void mask: labels 0 (void) and 255 (air) are excluded ---------
    solid_mask = ~np.isin(voxels_labels, sorted(VOID_LABELS))
    n_solid = int(solid_mask.sum())
    n_void  = Nx * Ny * Nz - n_solid
    print(f"\n  Solid voxels   : {n_solid:,} / {Nx*Ny*Nz:,} "
          f"({100*n_solid/(Nx*Ny*Nz):.1f}%)")
    print(f"  Void/air voxels: {n_void:,} "
          f"(labels {sorted(VOID_LABELS)} — excluded from system)")

    # --- E_voxels loaded directly from file -----------------------------------
    E_min = E_voxels.min()
    E_max = E_voxels.max()
    print(f"\n  E range from file : [{E_min:.4f}, {E_max:.4f}] GPa")

    # --- Convert to voxel format: linear interpolation mode ---
    # Map E_voxels [E_min, E_max] → uint8 [0, 255]
    voxels = np.uint8(255.0 * (E_voxels - E_min) / (E_max - E_min))

    # --- Solver (linear elasticity) -----------------------------------------
    # Use linear interpolation mode: voxel values [0,255] map to [E_void, E_max]
    solver = VoxelFEMSolverNonlinear(
        voxels,
        nu=args["nu"],
        E_void=E_min,
        E_max=E_max,
        solid_mask=solid_mask,
        is_linear=True,
        tol_pcg=1e-8,
        max_pcg_iter=args["maxiter"],
        preconditioner=args["preconditioner"],
        mg_smoother=args["smoother"],
        verbose=True,
    )

    # --- Boundary conditions: uniaxial tension, lateral faces free ----------
    # Domain length along loading axis = dims[axis] (unit cubes).
    delta = args["strain"] * dims[axis]
    node_idx = build_node_index(Nx, Ny, Nz)
    apply_uniaxial_bc(solver, node_idx, axis, delta)

    print(f"\n  Applied {aname}-strain  : {args['strain']*100:.2f}%  "
          f"(Δ = {delta:.4f}, L{aname} = {dims[axis]})")
    print(f"  Poisson ratio    : {args['nu']}")
    print(f"  Max PCG iters    : {args['maxiter']}")
    print(f"  Preconditioner   : {args['preconditioner']}")
    if args["preconditioner"] == "multigrid":
        print(f"  MG smoother      : {args['smoother']}")

    # --- Floating-island diagnostic -----------------------------------------
    print()
    bc_dofs_np = np.where(solver.bc_mask)[0].astype(np.int32)
    n_comps, n_floating, island_sizes = solver._check_floating_islands(bc_dofs_np)

    if n_comps == 0:
        print("  Isolated clusters  : solid mask is empty — nothing to check")
    elif n_floating == 0:
        print(f"  Isolated clusters  : 0  ({n_comps} solid component(s), "
              f"all anchored to Dirichlet BCs)")
    else:
        total_elem = sum(island_sizes)
        top5 = sorted(island_sizes, reverse=True)[:5]
        tail = ", ...]" if len(island_sizes) > 5 else "]"
        sizes_str = str(top5)[:-1] + tail
        print(f"  Isolated clusters  : {n_floating} of {n_comps} solid component(s) "
              f"have no BC contact  ({total_elem} elements, sizes {sizes_str})")
        print(f"  → Their DOFs are decoupled via the void mask and stay at zero.")

    # --- Affine initial displacement field -----------------------------------
    u0 = build_affine_u0(Nx, Ny, Nz, args["strain"], args["nu"], axis)
    print(f"\n  Affine u0 seeded  : {aname}-displ range "
          f"[{float(u0[axis::3].min()):.4f}, {float(u0[axis::3].max()):.4f}]")

    # --- Solve ---------------------------------------------------------------
    t0 = time.perf_counter()
    u = solver.solve(u0=u0)
    cp.cuda.Device(0).synchronize()
    elapsed = time.perf_counter() - t0

    # --- Report --------------------------------------------------------------
    disp = [u[0::3], u[1::3], u[2::3]]
    print(f"\n  {'─'*60}")
    print(f"  RESULTS")
    print(f"  {'─'*60}")
    print(f"  Solve time        : {elapsed:.2f} s")
    for d, dname in enumerate(_AXIS_NAMES):
        tag = f"(prescribed Δ = {delta:.6f})" if d == axis else "(Poisson contraction)"
        print(f"  {dname}-disp range      : [{disp[d].min():.6f}, {disp[d].max():.6f}]  {tag}")

    E_eff, F_reaction = effective_modulus(u, E_voxels, Nx, Ny, Nz, args["nu"],
                                          axis, args["strain"], solid_mask=solid_mask)
    # Compute E range from solid voxels only
    E_solid = E_voxels[solid_mask]
    E_lo = E_solid.min() if len(E_solid) > 0 else 0.0
    E_hi = E_solid.max() if len(E_solid) > 0 else 0.0
    print(f"\n  Far-face {aname}-reaction   : {F_reaction:.4f}")
    print(f"  Effective E_{aname}          : {E_eff:.4f}")
    print(f"  (solid phase moduli   : [{E_lo:.3f}, {E_hi:.3f}] — "
          f"E_eff should lie within or below)")

    # --- Compute stress and strain fields (used for VTI and NPZ output) ------
    conn = build_conn_dof(Nx, Ny, Nz)
    E0 = E_voxels.ravel().astype(np.float32)  # per-voxel moduli from file
    strain_field = element_strain(u, conn)  # (Ne, 6) with Voigt order
    stress_field = element_stress(u, E0, conn, args["nu"])  # (Ne, 6)

    # Compute von Mises stress and strain
    # stress_field and strain_field are (Ne, 6) with Voigt order [xx, yy, zz, xy, yz, xz]
    Sxx, Syy, Szz = stress_field[:, 0], stress_field[:, 1], stress_field[:, 2]
    Sxy, Syz, Sxz = stress_field[:, 3], stress_field[:, 4], stress_field[:, 5]
    Exx, Eyy, Ezz = strain_field[:, 0], strain_field[:, 1], strain_field[:, 2]
    Exy, Eyz, Exz = strain_field[:, 3], strain_field[:, 4], strain_field[:, 5]

    # von Mises stress: σ_vm = sqrt(((Sxx-Syy)^2 + (Syy-Szz)^2 + (Szz-Sxx)^2 + 6*(Sxy^2 + Syz^2 + Sxz^2)) / 2)
    stress_vm = np.sqrt(((Sxx - Syy)**2 + (Syy - Szz)**2 + (Szz - Sxx)**2 +
                         6 * (Sxy**2 + Syz**2 + Sxz**2)) / 2.0).astype(np.float32)
    stress_vm = stress_vm.reshape(Nx, Ny, Nz)

    # von Mises strain: ε_vm = sqrt((1/3) * ((Exx-Eyy)^2 + (Eyy-Ezz)^2 + (Ezz-Exx)^2 + 2*(Exy^2 + Eyz^2 + Exz^2)))
    strain_vm = np.sqrt(((Exx - Eyy)**2 + (Eyy - Ezz)**2 + (Ezz - Exx)**2 +
                         2 * (Exy**2 + Eyz**2 + Exz**2)) / 3.0).astype(np.float32)
    strain_vm = strain_vm.reshape(Nx, Ny, Nz)

    # --- VTI output (default on) --------------------------------------------
    if args["vtk"]:
        from vtk_export import save_stress_strain_vti
        vti_path = args["vtk"] if args["vtk"].endswith(".vti") else args["vtk"] + ".vti"
        save_stress_strain_vti(vti_path, (Nx, Ny, Nz), strain_field, stress_field,
                               extra={"youngs_modulus": E0,
                                      "solid_mask": solid_mask.ravel()})
        print(f"\n  Wrote VTI: {vti_path}")

    # --- Save effective stress and strain as numpy array ----------------------
    # Stack along last axis: shape (Nx, Ny, Nz, 2) where last index is [stress, strain]
    results = np.stack([stress_vm, strain_vm], axis=-1).astype(np.float32)
    npy_path = args["vtk"] if args["vtk"].endswith(".npy") else args["vtk"] + "_results.npy"
    np.save(npy_path, results)
    print(f"  Wrote effective stress and strain: {npy_path}")

    # --- Save displacement field as 3D numpy array (nodes × components) --------
    # Reshape from (Ndof,) to (Nx+1, Ny+1, Nz+1, 3) where last axis is [ux, uy, uz]
    #
    # NOTE: The displacement field is defined at the FEM nodes, which form a
    # grid of size (Nx+1, Ny+1, Nz+1) for a voxel domain of size (Nx, Ny, Nz).
    # Thus the displacement array is **one voxel larger in each direction** than
    # the input intensity/voxel data.
    #
    # If displacements at voxel centers (i.e., corresponding to the gray-level
    # intensities in the input .npz file) are required, they must be obtained by
    # **linear interpolation** of the nodal displacements. This is consistent with
    # the trilinear shape functions used in the voxel FEM scheme (each voxel is a
    # Q1 hex element).
    #
    # To make the deformed image of a ground truth from this nodal field, use
    # warp_cupy.py (since 2026-09-25 it solves X + u(X) = y exactly and samples
    # the image with a cubic B-spline; node index = image coordinate):
    #
    #     python warp_cupy.py input.npz ct_result_displacement.npy \
    #            --image-key intensity --output deformed.npy
    #
    # (--legacy-pull gives the old first-order pull warp with node-to-voxel-
    # centre averaging; do not use it for ground-truth images.)
    #
    # warp_intensity.py is a standalone script for the same first-order pull
    # warp, I_def(x) = I_ref(x - u(x)), and also prints the mean normal strains:
    #
    #     python warp_intensity.py --intensity ct_result_intensity.npy \
    #            --displacement ct_result_displacement.npy
    #
    u_reshaped = u.reshape((Nx + 1, Ny + 1, Nz + 1, 3)).astype(np.float32)
    disp_path = args["vtk"] if args["vtk"].endswith(".npy") else args["vtk"] + "_displacement.npy"
    np.save(disp_path, u_reshaped)
    print(f"  Wrote displacement field: {disp_path}")

    # --- Save input gray-scale intensities ------------------------------------
    # Write out the per-voxel gray levels from the input .npz (shape Nx×Ny×Nz),
    # so the intensities that correspond to the displacement field are available
    # alongside the FEM results (e.g. as the reference image for warp_intensity.py).
    if "intensity" in data:
        intensity = data["intensity"]
        intensity_path = (args["vtk"] if args["vtk"].endswith(".npy")
                          else args["vtk"] + "_intensity.npy")
        np.save(intensity_path, intensity)
        print(f"  Wrote gray-scale intensities: {intensity_path}")
    else:
        print("  No 'intensity' key in input .npz — skipping intensity output.")

    print(f"\n{'='*70}")
    print("  Simulation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
