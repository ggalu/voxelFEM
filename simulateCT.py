# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-04 21:15:10
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-08 13:57:53
#!/usr/bin/env python
"""
Voxel FEM simulation of a CT-segmented sample
==============================================
Loads ``exp_data.npz`` (keys: ``labels``, ``intensity``; shape 100×600×600),
uses the ``labels`` array directly as Young's moduli (label value = E in GPa),
and runs a single **linear-elastic** solve under 1% uniaxial tensile strain
along the first dimension (x, length 100).

Void/air elimination
--------------------
Labels 0 (void/background) and 255 (air) are treated as empty space and
**removed from the system** via a binary solid mask. Only labels 1 (matrix)
and 2 (fibre) carry load. Void nodes receive identity rows in the operator
so the DOF vector stays at full-grid length (implicit connectivity preserved)
and inactive DOFs stay exactly zero throughout the solve.

Boundary conditions
-------------------
Tensile strain ``ε`` is applied along x by prescribing the x-displacement of
the left face (x=0, u_x=0) and the right face (x=Nx, u_x=Δ).  The y and z
faces are left free so the body contracts according to Poisson's ratio (ν=0.3);
three minimal "3-2-1" anchor constraints remove residual rigid-body translations/
rotation without restraining the lateral faces.  Because every voxel is a unit
cube, the x-domain length equals Nx, so Δ = ε · Nx.

Usage
-----
    python simulateCT.py
    python simulateCT.py --strain 0.01 --preconditioner multigrid --smoother chebyshev
    python simulateCT.py --vtk ct_result
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
# Boundary conditions: x-tension, lateral faces free (3-2-1 anchoring)
# ---------------------------------------------------------------------------

def _face_nodes(node_idx, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    return node_idx[tuple(sl)].ravel()


def apply_uniaxial_x_bc(solver, node_idx, delta):
    """
    Prescribe x-displacement on the left (x=0) and right (x=Nx) faces,
    leave y/z free, and anchor the transverse rigid-body modes minimally.
    """
    a = 0              # load axis = x
    b, c = 1, 2        # transverse axes = y, z

    left  = _face_nodes(node_idx, a, 0)
    right = _face_nodes(node_idx, a, -1)
    solver.apply_bc_displacement(left,  np.full(len(left),  a), np.zeros(len(left)))
    solver.apply_bc_displacement(right, np.full(len(right), a), np.full(len(right), delta))

    # Anchor the 3 transverse rigid-body modes (y-translation, z-translation,
    # rotation about x) without restraining the lateral faces' axial freedom.
    origin = int(node_idx[0, 0, 0])
    solver.apply_bc_displacement(np.array([origin, origin]),
                                 np.array([b, c]), np.array([0.0, 0.0]))
    ynode = int(node_idx[0, -1, 0])          # +y extreme: kills rotation about x
    solver.apply_bc_displacement(np.array([ynode]), np.array([c]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Effective x-modulus from the right-face reaction
# ---------------------------------------------------------------------------

def effective_modulus_x(u, material_table, voxels, nu, Nx, Ny, Nz, strain,
                        solid_mask=None):
    """
    E_eff = (F_x / A_total) / strain, where F_x is the summed x-reaction on
    the right face (x=Nx) and A_total = Ny·Nz (unit-cube voxels, full face).

    solid_mask : (Nx,Ny,Nz) bool or None — if given, void elements contribute
    zero to f_int (consistent with the solver's masking).
    """
    from constitutive import ConstitutiveModel
    mat = ConstitutiveModel("linear", E_void=float(material_table.min()),
                            E_max=float(material_table.max()))
    E0_cpu = material_table[voxels.ravel()].astype(np.float32)
    E0_gpu = cp.asarray(E0_cpu)

    B_cpu, w_cpu = build_B_gauss()
    B_gpu = cp.asarray(B_cpu.astype(np.float32))
    w_gpu = cp.asarray(w_cpu.astype(np.float32))
    Ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)

    solid_mask_gpu = (cp.asarray(solid_mask.ravel())
                      if solid_mask is not None else None)

    f_int = compute_f_int_cupy(cp.asarray(u), E0_gpu, B_gpu, w_gpu,
                               nu, mat, Ny, Nz, Ndof,
                               solid_mask_gpu=solid_mask_gpu)
    f_int = cp.asnumpy(f_int)

    node_idx = build_node_index(Nx, Ny, Nz)
    right = node_idx[-1, :, :].ravel()          # x = Nx face
    Fx = f_int[3 * right + 0].sum()             # DOF 0 = x-component

    area = Ny * Nz                               # total cross-section
    return Fx / area / strain, Fx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Linear-elastic voxel FEM simulation of CT data, x-tension")
    p.add_argument("--data", default="exp_data.npz",
                   help="path to the .npz file with a 'labels' array (default exp_data.npz)")
    p.add_argument("--strain", type=float, default=0.01,
                   help="applied uniaxial x-strain (default 0.01 = 1%%)")
    p.add_argument("--nu", type=float, default=0.3,
                   help="Poisson ratio (default 0.3)")
    p.add_argument("--preconditioner", choices=["jacobi", "multigrid"],
                   default="multigrid",
                   help="PCG preconditioner (default multigrid)")
    p.add_argument("--smoother", choices=["jacobi", "chebyshev"],
                   default="chebyshev",
                   help="multigrid smoother (default chebyshev; ignored for --preconditioner jacobi)")
    p.add_argument("--vtk", default="",
                   help="if set, write per-voxel stress/strain VTK with this prefix")
    args = p.parse_args()

    print("=" * 70)
    print("  VOXEL FEM — CT data, linear elasticity, x-tension")
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

    # --- Material table: label i -> E = float(i), label 0 -> void -----------
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
        max_pcg_iter=2000,
        preconditioner=args.preconditioner,
        mg_smoother=args.smoother,
        verbose=True,
    )

    # --- Boundary conditions: 1% x-tension, lateral faces free --------------
    # Each voxel is a unit cube; x-domain length = Nx, so Δ = ε · Nx.
    delta = args.strain * Nx
    node_idx = build_node_index(Nx, Ny, Nz)
    apply_uniaxial_x_bc(solver, node_idx, delta)

    print(f"\n  Applied x-strain : {args.strain*100:.2f}%  "
          f"(right displacement Δ = {delta:.4f}, domain length Lx = {Nx})")
    print(f"  Poisson ratio    : {args.nu}")
    print(f"  Preconditioner   : {args.preconditioner}")
    if args.preconditioner == "multigrid":
        print(f"  MG smoother      : {args.smoother}")

    # --- Floating-island diagnostic -----------------------------------------
    # Run connected-component labeling on the solid mask now that BCs are
    # known.  Any solid cluster not touching a Dirichlet node is a floating
    # island: it has rigid-body null-space modes that would stall PCG.
    # Calling the method here (before solve()) sets the _islands_checked flag
    # so solve() skips the duplicate check.
    print()
    bc_dofs_np = np.where(solver.bc_mask)[0].astype(np.int32)
    n_comps, n_floating, island_sizes = solver._check_floating_islands(bc_dofs_np)

    if n_comps == 0:
        print("  Isolated clusters  : solid mask is empty — nothing to check")
    elif n_floating == 0:
        print(f"  Isolated clusters  : 0  ({n_comps} solid component(s), "
              f"all anchored to Dirichlet BCs)")
    else:
        # WARNING was already printed by _check_floating_islands above.
        total_elem = sum(island_sizes)
        top5 = sorted(island_sizes, reverse=True)[:5]
        tail = ", ...]" if len(island_sizes) > 5 else "]"
        sizes_str = str(top5)[:-1] + tail
        print(f"  Isolated clusters  : {n_floating} of {n_comps} solid component(s) "
              f"have no BC contact  ({total_elem} elements, sizes {sizes_str})")
        print(f"  → Their DOFs are decoupled via the void mask and stay at zero.")

    # --- Solve --------------------------------------------------------------
    t0 = time.perf_counter()
    u = solver.solve()
    cp.cuda.Device(0).synchronize()
    elapsed = time.perf_counter() - t0

    # --- Report -------------------------------------------------------------
    ux, uy, uz = u[0::3], u[1::3], u[2::3]
    print(f"\n  {'─'*60}")
    print(f"  RESULTS")
    print(f"  {'─'*60}")
    print(f"  Solve time        : {elapsed:.2f} s")
    print(f"  x-disp range      : [{ux.min():.6f}, {ux.max():.6f}]  "
          f"(prescribed right Δ = {delta:.6f})")
    print(f"  y-disp range      : [{uy.min():.6f}, {uy.max():.6f}]  (Poisson contraction)")
    print(f"  z-disp range      : [{uz.min():.6f}, {uz.max():.6f}]  (Poisson contraction)")

    E_eff, Fx = effective_modulus_x(u, material_table, voxels, args.nu,
                                    Nx, Ny, Nz, args.strain,
                                    solid_mask=solid_mask)
    non_void_labels = labels[~np.isin(labels, sorted(VOID_LABELS))]
    E_lo = material_table[non_void_labels].min() if len(non_void_labels) else 0.0
    E_hi = material_table[non_void_labels].max() if len(non_void_labels) else 0.0
    print(f"\n  Right-face x-reaction : {Fx:.4f}")
    print(f"  Effective E_x         : {E_eff:.4f}")
    print(f"  (solid phase moduli   : [{E_lo:.3f}, {E_hi:.3f}] — "
          f"E_eff should lie within or below)")

    # --- Optional VTK -------------------------------------------------------
    if args.vtk:
        from tests.validation.report import (element_stress, element_strain,
                                             build_conn_dof)
        from vtk_export import save_stress_strain_vtk
        conn = build_conn_dof(Nx, Ny, Nz)
        E0 = material_table[voxels.ravel()].astype(np.float32)
        strain_field = element_strain(u, conn)
        stress_field = element_stress(u, E0, conn, args.nu)
        vtk_path = args.vtk if args.vtk.endswith(".vtk") else args.vtk + ".vtk"
        save_stress_strain_vtk(vtk_path, (Nx, Ny, Nz), strain_field, stress_field,
                               extra={"label": voxels.ravel().astype(np.float64),
                                      "youngs_modulus": E0.astype(np.float64),
                                      "solid_mask": solid_mask.ravel().astype(np.float64)})
        print(f"\n  Wrote VTK: {vtk_path}")

    print(f"\n{'='*70}")
    print("  Simulation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
