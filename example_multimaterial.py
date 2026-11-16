#!/usr/bin/env python
"""
Multi-material Nonlinear Voxel FEM — Example
=============================================
Demonstrates per-voxel material assignment using a 256-entry material lookup
table.  Each voxel carries an 8-bit material ID (0–255); the solver maps IDs
to Young's moduli at initialisation — no approximation through linear
interpolation.

Problem
-------
  80³ homogeneous block, 1% uniaxial tensile strain in z.
  Three materials assigned randomly with equal probability:

    ID 0  →  E =   1   (soft)
    ID 1  →  E =  10   (medium)
    ID 2  →  E = 100   (stiff)
    IDs 3–255  →  E = 1e-6  (void / unused)

  ν = 0.3 everywhere.

Because the problem is linear (ConstitutiveModel("linear")), the is_linear=True
flag is set.  This replaces the Newton-Raphson outer loop with a single PCG
solve at tight tolerance (tol_pcg=1e-8), which is both faster and more
accurate for linear problems.

The PCG solve runs with either the Jacobi or the geometric-multigrid
preconditioner (``--preconditioner``).  Jacobi is the default and the robust
choice here: the random/checkerboard patterns oscillate at the voxel scale and
average out under 2x coarsening, so geometric multigrid loses its coarse
correction and stalls.  (Multigrid shines instead on smoothly/structurally
varying stiffness — see ``example_layered.py``.)

Output
------
  - Solver initialisation info (material table stats, VRAM)
  - PCG convergence history (iteration + relative residual)
  - Timing breakdown
  - Displacement statistics and effective stiffness
  - VRAM usage at key points
  - Per-voxel stress/strain written to a ParaView .vti (or legacy .vtk) file
"""

import argparse
import os
import sys
import time
import numpy as np
import cupy as cp

# Reuse the validation package's centroid stress/strain recovery helpers.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests", "validation"))

from geometry import build_node_index
from constitutive import ConstitutiveModel
from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear
from report import build_conn_dof, element_strain, element_stress  # noqa: E402
from vtk_export import save_stress_strain_vtk, save_stress_strain_vti  # noqa: E402


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def vram_report(label=""):
    free_b, total_b = cp.cuda.Device(0).mem_info
    used = (total_b - free_b) / 1e9
    free = free_b / 1e9
    total = total_b / 1e9
    tag = f"  [{label}]" if label else ""
    print(f"  VRAM{tag}: {used:.3f} GB used / {total:.2f} GB total  "
          f"({free:.3f} GB free)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_material_grid(N, pattern, seed=42):
    """
    Return (N, N, N) uint8 array of material IDs 0/1/2.

    Patterns
    --------
    random      : each voxel independently drawn from {0,1,2} with equal prob.
    checkerboard: ID = (i+j+k) % 3  — periodic, perfectly equal thirds.
    """
    if pattern == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, 3, size=(N, N, N), dtype=np.uint8)
    elif pattern == "checkerboard":
        i, j, k = np.mgrid[0:N, 0:N, 0:N]
        return ((i + j + k) % 3).astype(np.uint8)
    else:
        raise ValueError(f"Unknown pattern '{pattern}'. Use 'random' or 'checkerboard'.")


def main(pattern="random", preconditioner="jacobi", N=80,
         vtk_path="multimaterial", vtk_format="vti"):
    # ------------------------------------------------------------------
    # Problem parameters
    # ------------------------------------------------------------------
    nu         = 0.3
    disp_frac  = 0.01          # 1% tensile strain
    seed       = 42
    # Jacobi PCG tolerance.  With 100:1 stiffness contrast and float32
    # arithmetic, the condition number κ(K) ≈ E_max/E_min × N² ≈ 640 000,
    # which limits achievable residual.  1e-5 is a practical target.
    tol_pcg    = 1e-5
    max_pcg    = 600           # upper bound on PCG iterations

    print("=" * 72)
    print(f"  MULTI-MATERIAL VOXEL FEM — LINEAR ELASTIC, {N}³ GRID  [{pattern}]")
    print("=" * 72)
    dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    free0_b, total_vram_b = cp.cuda.Device(0).mem_info
    print(f"  GPU   : {dev_name}")
    print(f"  VRAM  : {total_vram_b/1e9:.2f} GB total,  {free0_b/1e9:.2f} GB free")
    print(f"  Grid  : {N}³ = {N**3:,} elements,  {3*(N+1)**3:,} DOFs")
    print(f"  Strain: {disp_frac*100:.1f}% uniaxial tension (z)")
    print()

    # ------------------------------------------------------------------
    # Material table: 256 entries, IDs 0/1/2 active, rest = void
    # ------------------------------------------------------------------
    E_soft   = 1.0
    E_medium = 10.0
    E_stiff  = 100.0
    E_void   = 1e-6

    material_table = np.full(256, E_void, dtype=np.float32)
    material_table[0] = E_soft
    material_table[1] = E_medium
    material_table[2] = E_stiff

    E_max = float(material_table.max())   # = 100.0

    print("  Material table (active IDs):")
    print(f"    ID 0  →  E = {E_soft:>6.1f}  (soft)")
    print(f"    ID 1  →  E = {E_medium:>6.1f}  (medium)")
    print(f"    ID 2  →  E = {E_stiff:>6.1f}  (stiff)")
    print(f"    ID 3–255 →  E = {E_void:.1e}  (void / unused)")
    print()

    # ------------------------------------------------------------------
    # Voxel grid
    # ------------------------------------------------------------------
    material_ids = build_material_grid(N, pattern, seed=seed)

    counts = np.bincount(material_ids.ravel(), minlength=3)
    fracs  = counts / counts.sum()
    print(f"  Material distribution in voxel grid  (pattern: {pattern}):")
    for i, (c, f) in enumerate(zip(counts, fracs)):
        print(f"    ID {i}  (E={material_table[i]:>6.1f}): "
              f"{c:>8,} voxels  ({f*100:.1f}%)")
    print(f"    Total                : {counts.sum():>8,} voxels")
    print()

    # Volume-average Young's modulus (Voigt upper bound)
    E_voigt = sum(f * material_table[i] for i, f in enumerate(fracs))
    print(f"  Voigt (rule-of-mixtures) E_avg = {E_voigt:.2f}")
    # Voigt and Reuss bounds
    E_reuss = 1.0 / sum(f / material_table[i] for i, f in enumerate(fracs))
    print(f"  Reuss (harmonic mean)   E_avg = {E_reuss:.2f}")
    print(f"  Note: effective E will fall between Reuss ({E_reuss:.2f}) "
          f"and Voigt ({E_voigt:.2f}) bounds.")
    print()

    # ------------------------------------------------------------------
    # Solver setup
    # ------------------------------------------------------------------
    print(f"  Preconditioner: {preconditioner}")
    if preconditioner == "multigrid":
        print("    NOTE: geometric multigrid excels on smoothly/structurally")
        print("    varying stiffness (see example_layered.py), but the 'random'")
        print("    and 'checkerboard' patterns here oscillate at the voxel scale")
        print("    and average out under 2× coarsening, so multigrid can stall.")
        print("    Jacobi is the more robust choice for these patterns.")
    print()
    vram_report("before solver init")

    mat = ConstitutiveModel("linear", E_void=E_void, E_max=E_max)

    t_init = time.perf_counter()
    solver = VoxelFEMSolverNonlinear(
        material_ids,
        nu=nu,
        E_max=E_max,
        E_void=E_void,
        constitutive=mat,
        material_table=material_table,
        is_linear=True,           # skip NR loop — single tight PCG
        tol_pcg=tol_pcg,
        max_pcg_iter=max_pcg,
        preconditioner=preconditioner,
        verbose=True,
    )
    solver.pcg_print_interval = 50   # reduce output: print every 50 iters
    t_init_done = time.perf_counter()
    print(f"  Solver init time: {t_init_done - t_init:.2f} s")
    vram_report("after solver init")
    print()

    # ------------------------------------------------------------------
    # Boundary conditions: roller at bottom (u_z=0), prescribed u_z at top.
    #
    # Rigid-body modes eliminated (6 total):
    #   z-translation, x-rotation, y-rotation  → u_z=0 at all bottom nodes
    #   x-translation                           → u_x=0 at corner (0,0,0)
    #   y-translation                           → u_y=0 at corner (0,0,0)
    #   z-rotation                              → u_y=0 at corner (N,0,0)
    #                    (needs nonzero x; (0,0,0) alone leaves z-rotation free)
    #
    # The top corner (0,0,N) is additionally pinned in x,y so that the top
    # plane cannot drift laterally as a rigid block.
    # ------------------------------------------------------------------
    node_idx     = build_node_index(N, N, N)
    bottom       = node_idx[:, :, 0].ravel()
    top          = node_idx[:, :, -1].ravel()
    corner_bot   = np.array([node_idx[0, 0,  0]], dtype=np.int32)  # (0,0,0)
    corner_bot2  = np.array([node_idx[N, 0,  0]], dtype=np.int32)  # (N,0,0)
    corner_top   = np.array([node_idx[0, 0, -1]], dtype=np.int32)  # (0,0,N)

    # Roller: fix only u_z on the bottom face
    solver.apply_bc_displacement(
        bottom, np.full(len(bottom), 2, dtype=int), np.zeros(len(bottom))
    )
    # Pull top face in z; u_x, u_y remain free everywhere on the top face
    solver.apply_bc_displacement(
        top, np.full(len(top), 2, dtype=int), np.full(len(top), disp_frac)
    )
    # Corner (0,0,0): pin u_x and u_y  → removes x- and y-translation
    solver.apply_bc_displacement(corner_bot, np.array([0]), np.zeros(1))
    solver.apply_bc_displacement(corner_bot, np.array([1]), np.zeros(1))
    # Corner (N,0,0): pin u_y only     → removes z-rotation
    solver.apply_bc_displacement(corner_bot2, np.array([1]), np.zeros(1))
    # Corner (0,0,N): pin u_x and u_y  → top plane cannot translate in xy
    solver.apply_bc_displacement(corner_top, np.array([0]), np.zeros(1))
    solver.apply_bc_displacement(corner_top, np.array([1]), np.zeros(1))

    n_bc = solver.bc_mask.sum()
    print(f"  BCs applied: {n_bc:,} constrained DOFs  "
          f"({n_bc/solver.Ndof*100:.1f}% of total)")
    print()

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    print("  Starting linear PCG solve ...")
    print(f"  {'─'*60}")
    t_solve = time.perf_counter()
    u = solver.solve()
    t_solve_done = time.perf_counter()
    elapsed = t_solve_done - t_solve

    cp.cuda.Device(0).synchronize()
    vram_report("after solve")
    print()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    z_disp = u[2::3]
    x_disp = u[0::3]
    y_disp = u[1::3]

    print(f"  {'─'*60}")
    print(f"  RESULTS")
    print(f"  {'─'*60}")
    print(f"  Solve time          : {elapsed:.2f} s")
    print(f"  Solver init time    : {t_init_done - t_init:.2f} s")
    print(f"  Total wall time     : {t_solve_done - t_init:.2f} s")
    print()
    print(f"  Displacement field:")
    print(f"    z-disp range  : [{z_disp.min():.7f},  {z_disp.max():.7f}]"
          f"  (target max = {disp_frac})")
    print(f"    x-disp range  : [{x_disp.min():.7f},  {x_disp.max():.7f}]")
    print(f"    y-disp range  : [{y_disp.min():.7f},  {y_disp.max():.7f}]")
    print(f"    max |u|       : {np.abs(u).max():.7f}")
    print()

    # BC verification
    bottom_z = node_idx[:, :, 0].ravel() * 3 + 2
    top_z    = node_idx[:, :, -1].ravel() * 3 + 2
    bc_err_bottom = np.max(np.abs(u[bottom_z]))
    bc_err_top    = np.max(np.abs(u[top_z] - disp_frac))
    print(f"  BC verification:")
    print(f"    Max error at bottom (u_z = 0)    : {bc_err_bottom:.2e}")
    print(f"    Max error at top    (u_z = 0.01) : {bc_err_top:.2e}")
    print()

    # Effective stiffness from reaction forces
    from kernels_cupy_nl import compute_f_int_cupy, build_B_gauss
    B_cpu, w_cpu = build_B_gauss()
    B_gpu = cp.asarray(B_cpu.astype(np.float32))
    w_gpu = cp.asarray(w_cpu.astype(np.float32))
    u_gpu  = cp.asarray(u)

    f_int = compute_f_int_cupy(u_gpu, solver.E0_elem_gpu, B_gpu, w_gpu,
                                nu, mat, N, N, solver.Ndof)

    # Reaction on top face (sum of nodal forces in z)
    top_z_dofs  = cp.asarray(top * 3 + 2)
    F_reaction  = float(cp.sum(f_int[top_z_dofs]).get())
    top_area    = N * N           # cross-section in element units (N² unit faces)
    sigma_avg   = F_reaction / top_area
    strain_appl = disp_frac / N  # engineering strain ε = δ / L  (L = N voxel lengths)
    E_eff_comp  = sigma_avg / strain_appl

    print(f"  Effective moduli (from FEM reaction forces):")
    print(f"    Applied displacement δ  : {disp_frac:.4f}  (over height L = {N})")
    print(f"    Engineering strain ε_z  : {strain_appl:.6f}")
    print(f"    Reaction force Σf_z     : {F_reaction:.4f}")
    print(f"    Avg stress σ_z = F/A    : {sigma_avg:.6f}")
    print(f"    Effective E = σ/ε       : {E_eff_comp:.4f}")
    print(f"    Voigt upper bound       : {E_voigt:.4f}")
    print(f"    Reuss lower bound       : {E_reuss:.4f}")
    print(f"    Ratio E_eff / E_Voigt   : {E_eff_comp/E_voigt:.4f}")
    print(f"    E_eff / E_Reuss         : {E_eff_comp/E_reuss:.4f}")
    in_bounds = E_reuss <= E_eff_comp <= E_voigt
    print(f"  → E_eff {'lies' if in_bounds else 'does NOT lie'} between "
          f"Reuss ({E_reuss:.2f}) and Voigt ({E_voigt:.2f}) bounds.")
    print()

    # ------------------------------------------------------------------
    # Per-voxel stress/strain export for ParaView
    # ------------------------------------------------------------------
    if vtk_path:
        # Per-element base modulus exactly as the solver sees it (material-table
        # lookup, not the linear-interpolation encoding).
        E_elem = cp.asnumpy(solver.E0_elem_gpu)
        conn_dof = build_conn_dof(N, N, N)
        strain = element_strain(u, conn_dof)                 # (Ne, 6) centroid
        stress = element_stress(u, E_elem, conn_dof, nu)     # (Ne, 6) centroid
        extra = {
            "material_id": material_ids.ravel().astype(np.float32),
            "youngs_modulus": E_elem,
        }
        if vtk_format == "vti":
            # Binary, zlib-compressed XML ImageData with tensor-notation fields.
            save_stress_strain_vti(vtk_path, (N, N, N), strain, stress,
                                   extra=extra)
        else:
            # Legacy ASCII STRUCTURED_POINTS with per-component scalars.
            save_stress_strain_vtk(
                vtk_path, (N, N, N), strain, stress, extra=extra,
                title=f"multimaterial {pattern}, N={N}, nu={nu:g}")
        print(f"  Wrote per-voxel stress/strain : {vtk_path}")
        print()

    print("=" * 72)
    print("  Example completed.")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-material voxel FEM example")
    parser.add_argument(
        "--pattern", choices=["random", "checkerboard"], default="random",
        help="Material assignment pattern (default: random)",
    )
    parser.add_argument(
        "--preconditioner", choices=["jacobi", "multigrid"], default="jacobi",
        help="PCG preconditioner; multigrid needs far fewer iterations "
             "(default: jacobi)",
    )
    parser.add_argument(
        "--size", type=int, default=80,
        help="Grid side length N (grid is N³, default: 80)",
    )
    parser.add_argument(
        "--vtk", type=str, default="multimaterial",
        help="output filename prefix for per-voxel stress/strain; the format "
             "extension is appended (use '' to disable output)",
    )
    parser.add_argument(
        "--format", choices=["vti", "vtk"], default="vti",
        help="vti = binary zlib-compressed XML ImageData with tensor-notation "
             "strain/stress (needs pyvista); vtk = legacy ASCII with "
             "per-component scalars (default: vti)",
    )
    args = parser.parse_args()
    vtk_path = f"{args.vtk}.{args.format}" if args.vtk else None
    main(pattern=args.pattern, preconditioner=args.preconditioner, N=args.size,
         vtk_path=vtk_path, vtk_format=args.format)
