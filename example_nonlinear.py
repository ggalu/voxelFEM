#!/usr/bin/env python
"""
Nonlinear Voxel FEM Solver — Example
======================================
Demonstrates VoxelFEMSolverNonlinear on a uniform cube (default 40³) under 1%
uniaxial tensile strain in the z-direction, comparing two material models:

  1. Linear elasticity     — baseline, no stiffness change with strain
  2. Power-law hardening   — stiffness increases with deformation

Reports:
  - NR and PCG convergence history at each load step
  - Per-load-step timing
  - VRAM usage (peak and breakdown)
  - Displacement field statistics and derived quantities

Usage:
    python example_nonlinear.py
    python example_nonlinear.py --size 80 --preconditioner jacobi
"""

import argparse
import time
import numpy as np
import cupy as cp

from geometry import build_node_index
from constitutive import ConstitutiveModel
from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def vram_used_gb():
    """VRAM currently used (CuPy pool + untracked allocations)."""
    pool = cp.get_default_memory_pool()
    return pool.used_bytes() / 1e9


def vram_free_total_gb():
    """(free_GB, total_GB) from the CUDA device query."""
    free, total = cp.cuda.Device(0).mem_info
    return free / 1e9, total / 1e9


def vram_report(label=""):
    free, total = vram_free_total_gb()
    used = total - free
    tag = f"  [{label}]" if label else ""
    print(f"  VRAM{tag}: {used:.2f} GB used / {total:.2f} GB total  "
          f"({free:.2f} GB free)")


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------

def run_model(name, constitutive, Nx, Ny, Nz, nu, E_max, E_void,
              n_load_steps, tol_nr, max_pcg_iter, disp_frac,
              preconditioner="multigrid"):
    """
    Solve uniaxial tension problem with the given constitutive model.

    Returns (u, elapsed_s, per_step_times).
    """
    print(f"\n{'='*70}")
    print(f"  MODEL: {name}")
    print(f"  {constitutive}")
    print(f"{'='*70}")

    vram_report("before init")

    voxels = np.full((Nx, Ny, Nz), 200, dtype=np.uint8)   # homogeneous block
    solver = VoxelFEMSolverNonlinear(
        voxels,
        nu=nu,
        E_max=E_max,
        E_void=E_void,
        constitutive=constitutive,
        n_load_steps=n_load_steps,
        max_nr_iter=15,
        tol_nr=tol_nr,
        max_pcg_iter=max_pcg_iter,
        preconditioner=preconditioner,
        gpu_chunk_size=2_000_000,
        verbose=True,                 # the solver prints NR/PCG progress
    )

    vram_report("after init")

    # Boundary conditions: fix bottom face, pull top face 1%
    node_idx = build_node_index(Nx, Ny, Nz)
    bottom = node_idx[:, :, 0].ravel()
    top    = node_idx[:, :, -1].ravel()

    for dof in range(3):
        solver.apply_bc_displacement(
            bottom, np.full(len(bottom), dof, dtype=int), np.zeros(len(bottom))
        )
    # disp_frac is absolute displacement (domain spans [0,1] in each direction
    # since we use unit-cube elements, so 0.01 = 1% strain)
    solver.apply_bc_displacement(
        top, np.full(len(top), 2, dtype=int), np.full(len(top), disp_frac)
    )

    # Solve with per-step timing
    print(f"\n  Grid: {Nx}³ = {Nx*Ny*Nz:,} elements, {solver.Ndof:,} DOFs")
    print(f"  Applied strain: {disp_frac*100:.1f}%  ({n_load_steps} load steps)")
    print()

    per_step_times = []
    original_nr_step = solver._nr_step

    # Wrap the solver's per-load-step routine to time each step. The arguments
    # are passed through unchanged, so the wrapper does not depend on the
    # solver's internal signature.
    def _timed_nr_step(*step_args, **step_kwargs):
        step_num = len(per_step_times) + 1
        lf = step_num / n_load_steps
        print(f"  --- Load step {step_num}/{n_load_steps}  (fraction={lf:.2f}) ---")
        t0 = time.perf_counter()
        u_out = original_nr_step(*step_args, **step_kwargs)
        dt = time.perf_counter() - t0
        per_step_times.append(dt)
        print(f"  Step {step_num} wall time: {dt:.2f} s")
        vram_report(f"step {step_num}")
        return u_out

    solver._nr_step = _timed_nr_step

    t_total_start = time.perf_counter()
    u = solver.solve()
    elapsed = time.perf_counter() - t_total_start

    cp.cuda.Device(0).synchronize()
    vram_report("after solve")

    return u, elapsed, per_step_times


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def print_results(name, u, elapsed, per_step_times, Nx, Ny, Nz, disp_frac):
    print(f"\n  {'─'*60}")
    print(f"  RESULTS — {name}")
    print(f"  {'─'*60}")

    z_disp = u[2::3]
    x_disp = u[0::3]
    y_disp = u[1::3]

    applied_disp = disp_frac
    max_z = z_disp.max()
    strain_z = max_z                           # = Δz / L, L=1 (unit domain)
    strain_x = x_disp.std()                    # lateral contraction spread

    print(f"  Displacement field:")
    print(f"    z-disp range : [{z_disp.min():.6f},  {z_disp.max():.6f}]  "
          f"(target max = {applied_disp:.6f})")
    print(f"    x-disp range : [{x_disp.min():.6f},  {x_disp.max():.6f}]")
    print(f"    y-disp range : [{y_disp.min():.6f},  {y_disp.max():.6f}]")
    print(f"    max |u|      : {np.abs(u).max():.6f}")

    print(f"\n  Timing:")
    print(f"    Total solve time : {elapsed:.2f} s")
    for i, dt in enumerate(per_step_times, 1):
        print(f"    Load step {i:2d}      : {dt:.2f} s")

    print(f"\n  Solver summary:")
    n_steps = len(per_step_times)
    print(f"    Load steps       : {n_steps}")
    print(f"    Avg step time    : {elapsed/n_steps:.2f} s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Nonlinear voxel FEM uniaxial-tension example")
    p.add_argument("--size", type=int, default=40,
                   help="grid side length N (N x N x N voxels), default 40")
    p.add_argument("--preconditioner", choices=["multigrid", "jacobi"],
                   default="multigrid",
                   help="PCG preconditioner (default multigrid)")
    args = p.parse_args()

    # Problem parameters
    N          = args.size   # grid size
    precond    = args.preconditioner
    nu         = 0.3         # Poisson ratio
    E_max      = 1.0         # Young's modulus at voxel=255
    E_void     = 1e-6        # void modulus
    disp_frac  = 0.01        # 1% tensile strain (displacement / domain length)
    n_steps    = 5           # load steps
    tol_nr     = 1e-5        # NR convergence tolerance (float32 precision ~1e-7)
    max_pcg    = 200         # max PCG iterations per NR step

    print("=" * 70)
    print(f"  NONLINEAR VOXEL FEM — {N}³ UNIAXIAL TENSION EXAMPLE")
    print("=" * 70)
    free, total = vram_free_total_gb()
    print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  VRAM: {total:.2f} GB total,  {free:.2f} GB free at start")
    print(f"  Grid: {N}³ = {N**3:,} elements")
    print(f"  DOFs: {3*(N+1)**3:,}")
    print(f"  Applied strain: {disp_frac*100:.1f}%  ({n_steps} load steps)")
    print(f"  Preconditioner: {precond}")

    results = {}

    # ------------------------------------------------------------------
    # Model 1: Linear elasticity
    # ------------------------------------------------------------------
    mat_linear = ConstitutiveModel(
        "linear",
        E_void=E_void,
        E_max=E_max,
    )
    u_lin, t_lin, steps_lin = run_model(
        "Linear elasticity",
        mat_linear, N, N, N, nu, E_max, E_void,
        n_steps, tol_nr, max_pcg, disp_frac,
        preconditioner=precond,
    )
    results["Linear"] = (u_lin, t_lin, steps_lin)
    print_results("Linear elasticity", u_lin, t_lin, steps_lin, N, N, N, disp_frac)

    # ------------------------------------------------------------------
    # Model 2: Power-law hardening
    # ------------------------------------------------------------------
    mat_hard = ConstitutiveModel(
        "hardening",
        E_void=E_void,
        E_max=E_max,
        beta=50.0,    # stiffening coefficient
        n=0.5,        # exponent (√ε_vm hardening)
    )
    u_hard, t_hard, steps_hard = run_model(
        "Power-law hardening  (β=50, n=0.5)",
        mat_hard, N, N, N, nu, E_max, E_void,
        n_steps, tol_nr, max_pcg, disp_frac,
        preconditioner=precond,
    )
    results["Hardening"] = (u_hard, t_hard, steps_hard)
    print_results("Power-law hardening", u_hard, t_hard, steps_hard, N, N, N, disp_frac)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<40}  {'Linear':>12}  {'Hardening':>12}")
    print(f"  {'─'*66}")
    print(f"  {'Total solve time (s)':<40}  {t_lin:>12.2f}  {t_hard:>12.2f}")
    print(f"  {'Max z-displacement':<40}  {u_lin[2::3].max():>12.6f}  {u_hard[2::3].max():>12.6f}")
    print(f"  {'Max lateral displacement':<40}  {np.abs(u_lin[0::3]).max():>12.6f}  {np.abs(u_hard[0::3]).max():>12.6f}")

    # Effective stiffness: max reaction force proxy = ||K*u|| / ||u||
    # Approximate via ||f_int|| using a quick call
    from kernels_cupy_nl import compute_f_int_cupy, build_B_gauss
    B_cpu, w_cpu = build_B_gauss()
    B_gpu = cp.asarray(B_cpu.astype(np.float32))
    w_gpu = cp.asarray(w_cpu.astype(np.float32))
    voxels_ref = np.full((N, N, N), 200, dtype=np.uint8)
    E_scale = (E_max - E_void) / 255.0
    E0_gpu = cp.asarray(voxels_ref.ravel().astype(np.float32) * E_scale + E_void)
    Ndof = 3 * (N + 1) ** 3

    eff_k = {}
    for label, (u_arr, _, _) in results.items():
        mat = mat_linear if label == "Linear" else mat_hard
        u_gpu = cp.asarray(u_arr)
        f_int = compute_f_int_cupy(u_gpu, E0_gpu, B_gpu, w_gpu, nu, mat, N, N, Ndof)
        f_norm = float(cp.linalg.norm(f_int).get())
        u_norm = float(cp.linalg.norm(u_gpu).get())
        eff_k[label] = f_norm / (u_norm + 1e-30)
        print(f"  {'Effective stiffness ||f_int||/||u||  ' + label:<40}  {eff_k[label]:>12.4f}")

    ratio = eff_k["Hardening"] / (eff_k["Linear"] + 1e-30)
    print(f"\n  Power-law hardening model is {ratio:.2f}× stiffer than linear "
          f"at 1% strain (β=50, n=0.5 → stiffness grows as √ε_vm).")

    print(f"\n{'='*70}")
    print("  Example completed.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
