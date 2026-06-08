# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-04 21:15:10
#!/usr/bin/env python
"""
Voxel FEM simulation of a CT-segmented sample
==============================================
Loads ``exp_data.npz`` (keys: ``labels``, ``intensity``; shape 100×600×600),
uses the ``labels`` array directly as Young's moduli (label value = E in GPa),
and runs a single **linear-elastic** solve under 1% uniaxial tensile strain.
Default loading axis is **z** (length 100).

Bounding-box crop
-----------------
Before constructing the solver the voxel array is cropped to the tight
bounding box of solid voxels plus ``--pad`` voxels of air padding on every
face (default 2).  This has two effects:

1. *Multigrid effectiveness*: the geometric multigrid coarsens by 2×2×2
   averaging.  When a solid specimen is embedded in a large air region the
   coarse levels blend solid and air stiffness into ghost intermediate-
   stiffness elements that represent neither material; coarse corrections
   are inaccurate and the V-cycle stalls.  After cropping the coarse levels
   see mostly solid elements and the transition zone is at most one cell
   thick at the coarsest level.

2. *Domain reduction*: the system to be solved is proportionally smaller.

Pass ``--no-crop`` to skip the crop and simulate the full CT volume.

Solid-only boundary conditions
-------------------------------
After cropping, displacement BCs are applied only to the end-face nodes that
are corners of ≥1 solid element.  Air nodes in the padding region of the
end-face receive no BC and remain free (inactive nodes are frozen at zero by
the void-mask mechanism; active air end-face nodes are left traction-free).
The 3-2-1 anchor also uses solid-adjacent nodes chosen from the loaded face.

Affine initialisation
---------------------
The displacement field is seeded with the homogeneous affine solution
``u_a = ε·x_a``, ``u_b = u_c = −ν·ε·x_{b,c}`` before the PCG solve starts.

Void/air elimination
--------------------
Labels 0 (void/background) and 255 (air) are treated as empty space and
removed from the system via a binary solid mask.

Usage
-----
    python simulateCT.py
    python simulateCT.py --axis z --strain 0.01 --maxiter 200 --pad 2
    python simulateCT.py --no-crop --preconditioner jacobi
    python simulateCT.py --vtk ct_result --axis x
"""

import argparse
import time
import numpy as np
import cupy as cp

from geometry import build_node_index
from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
from kernels_cupy_nl import compute_f_int_cupy, build_B_gauss


# ---------------------------------------------------------------------------
# Label -> Young's modulus mapping  (edit here to change material properties)
# ---------------------------------------------------------------------------
#   label 0   : void / background  → removed from system (solid_mask = False)
#   label 1   : matrix             [GPa]
#   label 2   : fibres             [GPa]
#   label 255 : air                → removed from system (solid_mask = False)
LABEL_MODULI = {
    0:   1.0e-3,    # void  (near-zero; excluded via solid_mask)
    1:   1.0,       # matrix  [GPa]
    2:   10.0,      # fibres  [GPa]
    255: 1.0e-3,    # air   (near-zero; excluded via solid_mask)
}

# Labels treated as void/air: removed from the FEM system via solid_mask.
VOID_LABELS = {0, 255}

_AXIS_NAMES = ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Material table: label value -> Young's modulus
# ---------------------------------------------------------------------------

def build_label_material_table(label_moduli=LABEL_MODULI):
    """
    Build a length-256 material table from ``label_moduli`` (dict label->E).
    Labels not listed default to E_void (the value for label 0, or 1e-6).
    """
    E_void = label_moduli.get(0, 1e-6)
    table = np.full(256, E_void, dtype=np.float32)
    for lbl, E in label_moduli.items():
        table[int(lbl)] = float(E)
    return table


# ---------------------------------------------------------------------------
# Bounding-box crop to solid region
# ---------------------------------------------------------------------------

def crop_to_solid(voxels, solid_mask, pad=2):
    """
    Crop ``voxels`` and ``solid_mask`` to the tight bounding box of solid
    voxels, expanded by ``pad`` voxels on each face (clamped to domain edges).

    Returns
    -------
    voxels_crop    : (Nx', Ny', Nz') uint8
    solid_mask_crop: (Nx', Ny', Nz') bool
    crop_slices    : tuple of 3 slice objects (for VTI origin recovery)
    crop_origin    : (x0, y0, z0) int — offset of the cropped corner in the
                     original CT coordinate system
    """
    pts = np.argwhere(solid_mask)
    if pts.shape[0] == 0:
        slices = (slice(0, voxels.shape[0]),
                  slice(0, voxels.shape[1]),
                  slice(0, voxels.shape[2]))
        return voxels, solid_mask, slices, (0, 0, 0)

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    shape = np.array(voxels.shape)

    lo_pad = np.maximum(lo - pad, 0)
    hi_pad = np.minimum(hi + pad + 1, shape)   # +1: slice upper bound is exclusive

    slices = tuple(slice(int(lo_pad[a]), int(hi_pad[a])) for a in range(3))
    origin = tuple(int(lo_pad[a]) for a in range(3))

    voxels_crop    = voxels[slices]
    solid_mask_crop = solid_mask[slices]

    void_before = 1.0 - solid_mask.mean()
    void_after  = 1.0 - solid_mask_crop.mean()
    print(f"  Crop  : {voxels.shape} → {voxels_crop.shape}  "
          f"(pad={pad}, origin={origin})")
    print(f"  Void fraction  : {void_before:.3f} → {void_after:.3f}")

    return voxels_crop, solid_mask_crop, slices, origin


# ---------------------------------------------------------------------------
# Solid end-face node selection
# ---------------------------------------------------------------------------

def solid_end_face_nodes(node_idx, solid_mask_3d, axis, side):
    """
    Return the unique node IDs on the ``side`` end-face (0=near, -1=far)
    along ``axis`` that are corners of ≥1 solid element on that face.

    Parameters
    ----------
    node_idx      : (Nx+1, Ny+1, Nz+1) int32 from build_node_index
    solid_mask_3d : (Nx, Ny, Nz) bool
    axis          : 0, 1, or 2
    side          : 0 (near face, coord=0) or -1 (far face, coord=N_axis)

    Returns
    -------
    node_ids : (n,) int32 — sorted unique node IDs on the face
    """
    tr = [d for d in range(3) if d != axis]
    b, c = tr[0], tr[1]

    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = 0 if side == 0 else -1
    face_mask = solid_mask_3d[tuple(sl)]          # (N_b, N_c) bool

    ib_solid, ic_solid = np.where(face_mask)
    if len(ib_solid) == 0:
        return np.array([], dtype=np.int32)

    ax_coord = 0 if side == 0 else (node_idx.shape[axis] - 1)

    node_parts = []
    for db in (0, 1):
        for dc in (0, 1):
            idx = [None, None, None]
            idx[axis] = ax_coord
            idx[b]    = ib_solid + db
            idx[c]    = ic_solid + dc
            node_parts.append(node_idx[tuple(idx)])

    return np.unique(np.concatenate(node_parts)).astype(np.int32)


def _node_ijk(node_ids, Ny, Nz):
    """Recover (i, j, k) coordinate arrays from flat node IDs."""
    sj = Nz + 1
    si = (Ny + 1) * sj
    i = node_ids // si
    j = (node_ids // sj) % (Ny + 1)
    k = node_ids % sj
    return i, j, k


# ---------------------------------------------------------------------------
# Boundary conditions: uniaxial tension along `axis`, lateral faces free
# ---------------------------------------------------------------------------

def apply_uniaxial_bc(solver, node_idx, axis, delta, solid_mask_3d=None):
    """
    Prescribe displacement on the two end-faces along ``axis`` (0=x,1=y,2=z),
    leave the remaining four faces free, and apply a minimal 3-2-1 anchor.

    When ``solid_mask_3d`` is provided the BCs are applied only to nodes that
    are corners of ≥1 solid element on each end-face; the 3-2-1 anchor nodes
    are also chosen from the set of solid end-face nodes.

    The 3-2-1 anchor fixes:
      - transverse DOF b and c at the min-b solid node  (kills b/c-translation)
      - transverse DOF c at the max-b solid node         (kills rotation about axis)
    """
    tr = [d for d in range(3) if d != axis]
    b, c = tr[0], tr[1]

    if solid_mask_3d is not None:
        left  = solid_end_face_nodes(node_idx, solid_mask_3d, axis, side=0)
        right = solid_end_face_nodes(node_idx, solid_mask_3d, axis, side=-1)
    else:
        sl_left  = [slice(None), slice(None), slice(None)]
        sl_right = [slice(None), slice(None), slice(None)]
        sl_left[axis]  = 0
        sl_right[axis] = -1
        left  = node_idx[tuple(sl_left)].ravel()
        right = node_idx[tuple(sl_right)].ravel()

    solver.apply_bc_displacement(left,  np.full(len(left),  axis), np.zeros(len(left)))
    solver.apply_bc_displacement(right, np.full(len(right), axis), np.full(len(right), delta))

    # 3-2-1 anchor: pick solid-adjacent nodes from the loaded face when possible
    if solid_mask_3d is not None and len(left) > 0:
        Ny_n = node_idx.shape[1] - 1
        Nz_n = node_idx.shape[2] - 1
        ci, cj, ck = _node_ijk(left, Ny_n, Nz_n)
        bc_coords = [ci, cj, ck]
        origin_node = int(left[np.argmin(bc_coords[b])])
        bmax_node   = int(left[np.argmax(bc_coords[b])])
    else:
        origin_node = int(node_idx[0, 0, 0])
        idx_bmax = [0, 0, 0]
        idx_bmax[b] = -1
        bmax_node = int(node_idx[tuple(idx_bmax)])

    solver.apply_bc_displacement(np.array([origin_node, origin_node]),
                                 np.array([b, c]), np.array([0.0, 0.0]))
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
    """
    Ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
    u0 = np.zeros(Ndof, dtype=np.float32)

    ii, jj, kk = np.mgrid[0:Nx+1, 0:Ny+1, 0:Nz+1]
    coords = [ii.ravel().astype(np.float32),
              jj.ravel().astype(np.float32),
              kk.ravel().astype(np.float32)]

    u0[axis::3] = strain * coords[axis]
    for d in range(3):
        if d != axis:
            u0[d::3] = -nu * strain * coords[d]

    return u0


# ---------------------------------------------------------------------------
# Effective modulus from the reaction force on the loaded end-face
# ---------------------------------------------------------------------------

def effective_modulus(u, material_table, voxels, nu, axis, strain,
                      solid_mask=None):
    """
    E_eff = (F_axis / A_total) / strain, where F_axis is the summed reaction
    on the far face and A_total is the cross-sectional area perpendicular to
    ``axis`` (unit-cube voxels, full face).
    """
    from constitutive import ConstitutiveModel
    Nx, Ny, Nz = voxels.shape
    mat = ConstitutiveModel("linear", E_void=float(material_table.min()),
                            E_max=float(material_table.max()))
    E0_gpu = cp.asarray(material_table[voxels.ravel()].astype(np.float32))
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
    p = argparse.ArgumentParser(
        description="Linear-elastic voxel FEM simulation of CT data, uniaxial tension")
    p.add_argument("--data", default="exp_data.npz",
                   help="path to the .npz file with a 'labels' array (default exp_data.npz)")
    p.add_argument("--axis", choices=["x", "y", "z"], default="z",
                   help="loading axis (default z)")
    p.add_argument("--strain", type=float, default=0.01,
                   help="applied uniaxial strain (default 0.01 = 1%%)")
    p.add_argument("--nu", type=float, default=0.3,
                   help="Poisson ratio (default 0.3)")
    p.add_argument("--maxiter", type=int, default=100,
                   help="maximum PCG iterations (default 100)")
    p.add_argument("--pad", type=int, default=2,
                   help="air padding (voxels) kept around the solid bounding "
                        "box after crop (default 2)")
    p.add_argument("--no-crop", action="store_true", default=False,
                   help="skip the bounding-box crop and simulate the full "
                        "CT volume (disables solid-only BCs; for debugging)")
    p.add_argument("--preconditioner", choices=["jacobi", "multigrid"],
                   default="multigrid",
                   help="PCG preconditioner (default multigrid)")
    p.add_argument("--smoother", choices=["jacobi", "chebyshev"],
                   default="chebyshev",
                   help="multigrid smoother (default chebyshev; "
                        "ignored for --preconditioner jacobi)")
    p.add_argument("--vtk", default="ct_result",
                   help="write per-voxel stress/strain to <prefix>.vti "
                        "(default 'ct_result'; set to '' to disable)")
    args = p.parse_args()

    axis = {"x": 0, "y": 1, "z": 2}[args.axis]
    aname = args.axis

    print("=" * 70)
    print(f"  VOXEL FEM — CT data, linear elasticity, {aname}-tension")
    print("=" * 70)

    # --- Load data -----------------------------------------------------------
    data = np.load(args.data)
    voxels = data["labels"].astype(np.uint8)
    Nx, Ny, Nz = voxels.shape
    labels = np.unique(voxels)
    print(f"  Data  : {args.data}")
    print(f"  Grid  : {Nx}×{Ny}×{Nz} = {Nx*Ny*Nz:,} voxels")
    print(f"  Labels present : {labels.tolist()}")

    # --- Solid/void mask: labels 0 (void) and 255 (air) are excluded ---------
    solid_mask = ~np.isin(voxels, sorted(VOID_LABELS))
    n_solid = int(solid_mask.sum())
    n_void  = Nx * Ny * Nz - n_solid
    print(f"  Solid voxels   : {n_solid:,} / {Nx*Ny*Nz:,} "
          f"({100*n_solid/(Nx*Ny*Nz):.1f}%)")
    print(f"  Void/air voxels: {n_void:,} "
          f"(labels {sorted(VOID_LABELS)} — excluded from system)")

    # --- Bounding-box crop (focus on specimen, improve multigrid convergence) -
    crop_origin = (0, 0, 0)
    if not args.no_crop:
        voxels, solid_mask, _crop_slices, crop_origin = \
            crop_to_solid(voxels, solid_mask, pad=args.pad)
        Nx, Ny, Nz = voxels.shape
        labels = np.unique(voxels)   # recompute: crop may remove some labels
        n_solid_c = int(solid_mask.sum())
        print(f"  Solid (cropped): {n_solid_c:,} / {Nx*Ny*Nz:,} "
              f"({100*n_solid_c/(Nx*Ny*Nz):.1f}%)")
    else:
        print("  Crop disabled  : simulating full volume (--no-crop)")

    dims = [Nx, Ny, Nz]

    # --- Material table: label i -> E ----------------------------------------
    material_table = build_label_material_table(LABEL_MODULI)
    print("  Material table (label -> E):")
    for lbl in labels:
        tag = "  ← void (masked out)" if lbl in VOID_LABELS else ""
        print(f"      {int(lbl):3d} -> {material_table[lbl]:8.4f}{tag}")

    # --- Solver (linear elasticity) -----------------------------------------
    solver = VoxelFEMSolverNonlinear(
        voxels,
        nu=args.nu,
        material_table=material_table,
        solid_mask=solid_mask,
        is_linear=True,
        tol_pcg=1e-8,
        max_pcg_iter=args.maxiter,
        preconditioner=args.preconditioner,
        mg_smoother=args.smoother,
        verbose=True,
    )

    # --- Boundary conditions: uniaxial tension, lateral faces free ----------
    # Domain length along loading axis = dims[axis] (unit cubes).
    delta = args.strain * dims[axis]
    node_idx = build_node_index(Nx, Ny, Nz)
    solid_mask_for_bc = solid_mask if not args.no_crop else None
    apply_uniaxial_bc(solver, node_idx, axis, delta,
                      solid_mask_3d=solid_mask_for_bc)

    print(f"\n  Applied {aname}-strain  : {args.strain*100:.2f}%  "
          f"(Δ = {delta:.4f}, L{aname} = {dims[axis]})")
    print(f"  BC nodes       : solid end-face only"
          if not args.no_crop else
          f"  BC nodes       : all end-face nodes (--no-crop)")
    print(f"  Poisson ratio  : {args.nu}")
    print(f"  Max PCG iters  : {args.maxiter}")
    print(f"  Preconditioner : {args.preconditioner}")
    if args.preconditioner == "multigrid":
        print(f"  MG smoother    : {args.smoother}")

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
    u0 = build_affine_u0(Nx, Ny, Nz, args.strain, args.nu, axis)
    print(f"\n  Affine u0 seeded  : {aname}-displ range "
          f"[{u0[axis::3].min():.4f}, {u0[axis::3].max():.4f}]")

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

    E_eff, F_reaction = effective_modulus(u, material_table, voxels, args.nu,
                                          axis, args.strain, solid_mask=solid_mask)
    non_void_labels = labels[~np.isin(labels, sorted(VOID_LABELS))]
    E_lo = material_table[non_void_labels].min() if len(non_void_labels) else 0.0
    E_hi = material_table[non_void_labels].max() if len(non_void_labels) else 0.0
    print(f"\n  Far-face {aname}-reaction   : {F_reaction:.4f}")
    print(f"  Effective E_{aname}          : {E_eff:.4f}")
    print(f"  (solid phase moduli   : [{E_lo:.3f}, {E_hi:.3f}] — "
          f"E_eff should lie within or below)")

    # --- VTI output (default on, origin-aligned to original CT coords) -------
    if args.vtk:
        from tests.validation.report import (element_stress, element_strain,
                                             build_conn_dof)
        from vtk_export import save_stress_strain_vti
        conn = build_conn_dof(Nx, Ny, Nz)
        E0 = material_table[voxels.ravel()].astype(np.float32)
        strain_field = element_strain(u, conn)
        stress_field = element_stress(u, E0, conn, args.nu)
        vti_path = args.vtk if args.vtk.endswith(".vti") else args.vtk + ".vti"
        save_stress_strain_vti(vti_path, (Nx, Ny, Nz), strain_field, stress_field,
                               origin=crop_origin,
                               extra={"label": voxels.ravel().astype(np.float64),
                                      "youngs_modulus": E0.astype(np.float64),
                                      "solid_mask": solid_mask.ravel().astype(np.float64)})
        print(f"\n  Wrote VTI: {vti_path}  (origin={crop_origin})")

    print(f"\n{'='*70}")
    print("  Simulation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
