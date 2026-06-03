#!/usr/bin/env python
"""
Voxel FEM simulation of a kanapy RVE — Example
================================================
Loads a voxelized representative volume element (RVE) produced by
``kanapy/generate_rve.py`` (``rve_grains.npy``: an (Nx,Ny,Nz) uint8 array of
grain *labels*), assigns each grain label a Young's modulus through a
``material_table``, and runs a single **linear-elastic** solve under 1%
uniaxial tensile strain in the z-direction.

Boundary conditions
-------------------
A tensile strain ``ε`` is applied along z by prescribing the z-displacement of
the bottom layer (z = 0) and the top layer (z = +Δ). The two lateral
directions (x, y) are left free so the body contracts according to Poisson's
ratio (ν = 0.3); three minimal "3-2-1" anchor constraints remove the residual
rigid-body translations/rotation without restraining the lateral faces (they
stay traction-free). Because every voxel is a unit cube, the domain length
along z equals ``Nz``, so a strain ε corresponds to a top displacement
``Δ = ε · Nz``.

Usage
-----
    python simulateRVE.py
    python simulateRVE.py --rve kanapy/rve_grains.npy --strain 0.01
    python simulateRVE.py --preconditioner multigrid --vtk rve
"""

import argparse
import time
import numpy as np
import cupy as cp

from geometry import build_node_index
from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
from kernels_cupy_nl import compute_f_int_cupy, build_B_gauss


# ---------------------------------------------------------------------------
# Material table: grain label -> Young's modulus
# ---------------------------------------------------------------------------

def build_material_table(labels, E_grains=None, E_void=1e-6):
    """
    Build a length-256 material table mapping each grain label to a modulus.

    Parameters
    ----------
    labels : 1-D int array
        The grain labels present in the RVE (e.g. [1, 2, 3, 4]). Label 0 is
        treated as void / unassigned voxels.
    E_grains : dict {label: E} or 1-D array, optional
        Explicit modulus per grain. If None, distinct moduli are spread
        linearly over [100, 200] across the present labels (in whatever unit
        the table is interpreted — here treated as GPa-like reference values).
    E_void : float
        Modulus assigned to label 0 (void) and to any unused table entry.

    Returns
    -------
    table : (256,) float32
    """
    table = np.full(256, E_void, dtype=np.float32)

    present = np.array(sorted(int(l) for l in labels if int(l) != 0))
    if E_grains is None:
        if len(present) == 1:
            vals = np.array([150.0])
        else:
            vals = np.linspace(100.0, 200.0, len(present))
        E_grains = {int(l): float(v) for l, v in zip(present, vals)}
    elif not isinstance(E_grains, dict):
        E_grains = {int(l): float(v) for l, v in zip(present, np.asarray(E_grains))}

    for lbl, E in E_grains.items():
        table[int(lbl)] = float(E)
    return table


# ---------------------------------------------------------------------------
# Boundary conditions: z-tension, lateral faces free (3-2-1 anchoring)
# ---------------------------------------------------------------------------

def _face_nodes(node_idx, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    return node_idx[tuple(sl)].ravel()


def apply_uniaxial_z_bc(solver, node_idx, delta):
    """
    Prescribe z-displacement on the bottom (z=0) and top (z=+delta) layers,
    leave x/y free, and anchor the transverse rigid-body modes minimally.
    """
    a = 2              # load axis = z
    b, c = 0, 1        # transverse axes = x, y

    bottom = _face_nodes(node_idx, a, 0)
    top    = _face_nodes(node_idx, a, -1)
    solver.apply_bc_displacement(bottom, np.full(len(bottom), a), np.zeros(len(bottom)))
    solver.apply_bc_displacement(top, np.full(len(top), a), np.full(len(top), delta))

    # Anchor the 3 transverse rigid-body modes (x-translation, y-translation,
    # rotation about z) without restraining the lateral faces' axial freedom.
    origin = int(node_idx[0, 0, 0])
    solver.apply_bc_displacement(np.array([origin, origin]),
                                 np.array([b, c]), np.array([0.0, 0.0]))
    xnode = int(node_idx[-1, 0, 0])          # +x extreme: kills rotation about z
    solver.apply_bc_displacement(np.array([xnode]), np.array([c]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Effective z-modulus from the top-face reaction
# ---------------------------------------------------------------------------

def effective_modulus_z(u, material_table, voxels, nu, Nx, Ny, Nz, strain):
    """
    E_eff = (F_z / A) / strain, where F_z is the summed z-reaction on the top
    face and A = Nx·Ny is the cross-sectional area (unit-cube voxels).
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

    f_int = compute_f_int_cupy(cp.asarray(u), E0_gpu, B_gpu, w_gpu,
                               nu, mat, Ny, Nz, Ndof)
    f_int = cp.asnumpy(f_int)

    node_idx = build_node_index(Nx, Ny, Nz)
    top = node_idx[:, :, -1].ravel()
    Fz = f_int[3 * top + 2].sum()

    area = Nx * Ny
    return Fz / area / strain, Fz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Linear-elastic voxel FEM simulation of a kanapy RVE")
    p.add_argument("--rve", default="kanapy/rve_grains.npy",
                   help="path to the RVE grain-label .npy (default kanapy/rve_grains.npy)")
    p.add_argument("--strain", type=float, default=0.01,
                   help="applied uniaxial z-strain (default 0.01 = 1%%)")
    p.add_argument("--nu", type=float, default=0.3,
                   help="Poisson ratio (default 0.3)")
    p.add_argument("--preconditioner", choices=["jacobi", "multigrid"],
                   default="jacobi",
                   help="PCG preconditioner (default jacobi)")
    p.add_argument("--vtk", default="",
                   help="if set, write per-voxel stress/strain VTK with this prefix")
    args = p.parse_args()

    print("=" * 70)
    print("  VOXEL FEM — kanapy RVE, linear elasticity, z-tension")
    print("=" * 70)

    # --- Load RVE -----------------------------------------------------------
    voxels = np.load(args.rve)
    assert voxels.ndim == 3, "RVE must be a 3-D array"
    voxels = voxels.astype(np.uint8)
    Nx, Ny, Nz = voxels.shape
    labels = np.unique(voxels)
    print(f"  RVE: {args.rve}")
    print(f"  Grid: {Nx}×{Ny}×{Nz} = {Nx*Ny*Nz:,} voxels")
    print(f"  Grain labels present: {labels.tolist()}")

    # --- Material table -----------------------------------------------------
    material_table = build_material_table(labels)
    print("  Material table (label -> E):")
    for lbl in labels:
        tag = "  (void)" if lbl == 0 else ""
        print(f"      {int(lbl):3d} -> {material_table[lbl]:8.3f}{tag}")

    # --- Solver (linear elasticity) -----------------------------------------
    solver = VoxelFEMSolverNonlinear(
        voxels,
        nu=args.nu,
        material_table=material_table,
        is_linear=True,                 # skip Newton-Raphson: single tight PCG
        tol_pcg=1e-8,
        max_pcg_iter=2000,
        preconditioner=args.preconditioner,
        verbose=True,
    )

    # --- Boundary conditions: 1% z-tension, lateral faces free --------------
    # Each voxel is a unit cube, so the z-domain length is Nz; a strain ε maps
    # to a top displacement Δ = ε · Nz.
    delta = args.strain * Nz
    node_idx = build_node_index(Nx, Ny, Nz)
    apply_uniaxial_z_bc(solver, node_idx, delta)

    print(f"\n  Applied z-strain : {args.strain*100:.2f}%  "
          f"(top displacement Δ = {delta:.4f}, domain length Lz = {Nz})")
    print(f"  Poisson ratio    : {args.nu}")
    print(f"  Preconditioner   : {args.preconditioner}\n")

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
    print(f"  z-disp range      : [{uz.min():.6f}, {uz.max():.6f}]  "
          f"(prescribed top Δ = {delta:.6f})")
    print(f"  x-disp range      : [{ux.min():.6f}, {ux.max():.6f}]  (Poisson contraction)")
    print(f"  y-disp range      : [{uy.min():.6f}, {uy.max():.6f}]  (Poisson contraction)")

    E_eff, Fz = effective_modulus_z(u, material_table, voxels, args.nu,
                                    Nx, Ny, Nz, args.strain)
    E_lo, E_hi = material_table[labels[labels != 0]].min(), material_table[labels].max()
    print(f"\n  Top-face z-reaction : {Fz:.4f}")
    print(f"  Effective E_z       : {E_eff:.4f}")
    print(f"  (grain moduli span  : [{E_lo:.3f}, {E_hi:.3f}] — E_eff should lie within)")

    # --- Optional VTK -------------------------------------------------------
    if args.vtk:
        from tests.validation.report import (element_stress, element_strain,
                                             build_conn_dof)
        from vtk_export import save_stress_strain_vtk
        conn = build_conn_dof(Nx, Ny, Nz)
        E0 = material_table[voxels.ravel()].astype(np.float32)
        strain = element_strain(u, conn)
        stress = element_stress(u, E0, conn, args.nu)
        vtk_path = args.vtk if args.vtk.endswith(".vtk") else args.vtk + ".vtk"
        save_stress_strain_vtk(vtk_path, (Nx, Ny, Nz), strain, stress,
                               extra={"grain_label": voxels.ravel().astype(np.float64),
                                      "youngs_modulus": E0.astype(np.float64)})
        print(f"\n  Wrote VTK: {vtk_path}")

    print(f"\n{'='*70}")
    print("  Simulation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
