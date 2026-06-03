#!/usr/bin/env python
"""
Layered Composite — Effective Modulus Example
=============================================
Builds a two-phase layered voxel composite, solves it with
``VoxelFEMSolverNonlinear`` in linear mode, and compares the FEM effective
Young's modulus against the exact analytic result:

  * loading NORMAL to the layers   -> harmonic (series) mean   1/(f1/E1 + f2/E2)
  * loading PARALLEL to the layers -> arithmetic (parallel) mean  f1*E1 + f2*E2

The FEM modulus is measured from the reaction force on the loaded face:

      E_eff = (F / A) / (delta / L)

The analytic reference is evaluated at the *actual* uint8-discretised moduli
the solver sees, so the comparison reflects discretisation/BC error only, not
voxel quantisation.

Poisson note
------------
The harmonic mean is the *uniaxial-stress* result and is exact only without
Poisson coupling.  At nu>0 with free lateral faces, bonded layers share a
common in-plane strain and the apparent normal modulus runs above the harmonic
mean (a real laminate effect).  This example therefore runs the NORMAL case at
nu=0 (exact) and the PARALLEL case at the physical nu=0.3 (exact there too).
Pass --nu to override and see the coupling error grow for the normal case.

Usage:
    python example_layered.py
    python example_layered.py --size 48 --contrast 100
    python example_layered.py --nu 0.3        # show Poisson coupling on normal case
"""

import argparse
import os
import sys

import numpy as np

# Reuse the validation package (analytic reference + post-processing helpers).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests", "validation"))

from geometry import build_node_index                              # noqa: E402
from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear     # noqa: E402
from reference_solutions import make_layered_voxels, layered_reference  # noqa: E402
from report import (                                               # noqa: E402
    build_conn_dof, E_elem_from_voxels, effective_modulus_from_reaction,
    element_strain, element_stress,
)
from vtk_export import save_stress_strain_vtk, save_stress_strain_vti  # noqa: E402

E_VOID = 1e-6
DELTA_FRAC = 1e-3        # prescribed face displacement = DELTA_FRAC * L


# ---------------------------------------------------------------------------
# Boundary conditions: minimal 3-2-1 anchoring (lateral faces stay free)
# ---------------------------------------------------------------------------

def _face_nodes(node_idx, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    return node_idx[tuple(sl)].ravel()


def apply_uniaxial_bc(solver, node_idx, load_axis, delta):
    """Prescribe axial displacement on the high face; anchor rigid-body modes."""
    a = load_axis
    b = (a + 1) % 3
    c = (a + 2) % 3

    low = _face_nodes(node_idx, a, 0)
    high = _face_nodes(node_idx, a, -1)
    solver.apply_bc_displacement(low, np.full(len(low), a), np.zeros(len(low)))
    solver.apply_bc_displacement(high, np.full(len(high), a),
                                 np.full(len(high), delta))

    origin = int(node_idx[0, 0, 0])
    solver.apply_bc_displacement(np.array([origin, origin]),
                                 np.array([b, c]), np.array([0.0, 0.0]))
    sel = [0, 0, 0]
    sel[b] = -1
    bnode = int(node_idx[tuple(sel)])
    solver.apply_bc_displacement(np.array([bnode]), np.array([c]),
                                 np.array([0.0]))


# ---------------------------------------------------------------------------
# Solve one loading orientation and measure the effective modulus
# ---------------------------------------------------------------------------

def measure_effective_modulus(N, layer_axis, load_axis, E1, E2, f1, E_max, nu,
                              max_pcg_iter, tol_pcg, verbose, vtk_path=None,
                              vtk_format="vti", preconditioner="multigrid"):
    voxels, info = make_layered_voxels(N, N, N, layer_axis, E1, E2, f1,
                                       E_max, E_void=E_VOID)
    L = N
    delta = DELTA_FRAC * L

    solver = VoxelFEMSolverNonlinear(
        voxels, nu=nu, E_max=E_max, E_void=E_VOID,
        is_linear=True, preconditioner=preconditioner, verbose=verbose,
        tol_pcg=tol_pcg, max_pcg_iter=max_pcg_iter,
    )
    node_idx = build_node_index(N, N, N)
    apply_uniaxial_bc(solver, node_idx, load_axis, delta)
    u = solver.solve()

    conn_dof = build_conn_dof(N, N, N)
    E_elem = E_elem_from_voxels(voxels, E_max, E_VOID)
    high = _face_nodes(node_idx, load_axis, -1)
    A = N * N
    E_eff_fem = effective_modulus_from_reaction(
        u, E_elem, conn_dof, nu, solver.Ndof, high, load_axis, delta, A, L)

    if vtk_path is not None:
        # Per-voxel centroid strain/stress -> structured-grid VTK.
        strain = element_strain(u, conn_dof)
        stress = element_stress(u, E_elem, conn_dof, nu)
        if vtk_format == "vti":
            # Binary, zlib-compressed XML ImageData with tensor-notation fields.
            save_stress_strain_vti(
                vtk_path, (N, N, N), strain, stress,
                extra={"youngs_modulus": E_elem})
        else:
            # Legacy ASCII STRUCTURED_POINTS with per-component scalar fields.
            save_stress_strain_vtk(
                vtk_path, (N, N, N), strain, stress,
                extra={"youngs_modulus": E_elem},
                title=f"layered composite, load_axis={load_axis}, nu={nu:g}")
        print(f"    wrote VTK       : {vtk_path}")

    ref = layered_reference(info['E1_actual'], info['E2_actual'],
                            info['f1_actual'], nu)
    return E_eff_fem, ref, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Layered composite effective modulus")
    p.add_argument("--size", type=int, default=40,
                   help="grid side length N (N x N x N voxels), default 40")
    p.add_argument("--contrast", type=float, default=10.0,
                   help="stiffness contrast E2/E1, default 10")
    p.add_argument("--f1", type=float, default=0.5,
                   help="volume fraction of phase 1, default 0.5")
    p.add_argument("--nu", type=float, default=None,
                   help="Poisson ratio override (default: 0 for normal, 0.3 for "
                        "parallel — keeps both analytic means exact)")
    p.add_argument("--max-pcg-iter", type=int, default=8000)
    p.add_argument("--tol-pcg", type=float, default=1e-9)
    p.add_argument("--preconditioner", choices=["multigrid", "jacobi"],
                   default="multigrid",
                   help="PCG preconditioner (default multigrid)")
    p.add_argument("--vtk", type=str, default="layered",
                   help="output filename prefix; writes <prefix>_normal.<ext> "
                        "and <prefix>_parallel.<ext> (use '' to disable)")
    p.add_argument("--format", choices=["vti", "vtk"], default="vti",
                   help="vti = binary zlib-compressed XML ImageData with "
                        "tensor-notation strain/stress (needs pyvista); "
                        "vtk = legacy ASCII with per-component scalars "
                        "(default vti)")
    p.add_argument("--verbose", action="store_true",
                   help="print NR/PCG convergence history")
    args = p.parse_args()

    N = args.size
    E1 = 1.0
    E2 = args.contrast
    E_max = max(E1, E2)
    f1 = args.f1

    # Layers stacked along z (layer_axis=2): normal loading = z, parallel = x.
    nu_normal = 0.0 if args.nu is None else args.nu
    nu_parallel = 0.3 if args.nu is None else args.nu

    print("=" * 66)
    print("  LAYERED COMPOSITE — EFFECTIVE MODULUS")
    print("=" * 66)
    print(f"  Grid           : {N} x {N} x {N}  ({N**3:,} voxels)")
    print(f"  Phases         : E1 = {E1:g},  E2 = {E2:g}  (contrast {E2/E1:g})")
    print(f"  Volume fraction: f1 = {f1:g}")
    print(f"  Layers stacked along z; loading normal=z, parallel=x")
    print("=" * 66)

    # --- Normal to layers: harmonic (series) mean ---
    ext = args.format
    vtk_normal = f"{args.vtk}_normal.{ext}" if args.vtk else None
    vtk_parallel = f"{args.vtk}_parallel.{ext}" if args.vtk else None

    print(f"\n[1/2] NORMAL to layers (series / harmonic mean)   nu = {nu_normal:g}")
    fem_n, ref_n, info = measure_effective_modulus(
        N, layer_axis=2, load_axis=2, E1=E1, E2=E2, f1=f1, E_max=E_max,
        nu=nu_normal, max_pcg_iter=args.max_pcg_iter, tol_pcg=args.tol_pcg,
        verbose=args.verbose, vtk_path=vtk_normal, vtk_format=args.format,
        preconditioner=args.preconditioner)
    ana_n = ref_n['E_eff_normal']
    err_n = abs(fem_n - ana_n) / ana_n
    print(f"    actual moduli  : E1 = {info['E1_actual']:.5g},  "
          f"E2 = {info['E2_actual']:.5g},  f1 = {info['f1_actual']:.4f}")
    print(f"    FEM   E_eff     = {fem_n:.6g}")
    print(f"    analytic E_eff  = {ana_n:.6g}   (harmonic mean)")
    print(f"    relative error  = {err_n * 100:.3f} %")

    # --- Parallel to layers: arithmetic (parallel) mean ---
    print(f"\n[2/2] PARALLEL to layers (parallel / arithmetic mean)   nu = {nu_parallel:g}")
    fem_p, ref_p, info = measure_effective_modulus(
        N, layer_axis=2, load_axis=0, E1=E1, E2=E2, f1=f1, E_max=E_max,
        nu=nu_parallel, max_pcg_iter=args.max_pcg_iter, tol_pcg=args.tol_pcg,
        verbose=args.verbose, vtk_path=vtk_parallel, vtk_format=args.format,
        preconditioner=args.preconditioner)
    ana_p = ref_p['E_eff_parallel']
    err_p = abs(fem_p - ana_p) / ana_p
    print(f"    actual moduli  : E1 = {info['E1_actual']:.5g},  "
          f"E2 = {info['E2_actual']:.5g},  f1 = {info['f1_actual']:.4f}")
    print(f"    FEM   E_eff     = {fem_p:.6g}")
    print(f"    analytic E_eff  = {ana_p:.6g}   (arithmetic mean)")
    print(f"    relative error  = {err_p * 100:.3f} %")

    # --- Summary table ---
    print("\n" + "=" * 66)
    print(f"  {'Loading':<14}{'FEM E_eff':>14}{'Analytic E_eff':>18}{'Error':>10}")
    print(f"  {'-'*12:<14}{'-'*11:>14}{'-'*14:>18}{'-'*7:>10}")
    print(f"  {'normal':<14}{fem_n:>14.6g}{ana_n:>18.6g}{err_n*100:>9.3f}%")
    print(f"  {'parallel':<14}{fem_p:>14.6g}{ana_p:>18.6g}{err_p*100:>9.3f}%")
    print("=" * 66)


if __name__ == "__main__":
    main()
