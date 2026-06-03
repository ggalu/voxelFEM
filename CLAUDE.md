# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **matrix-free, GPU-resident voxel FEM solver for nonlinear elasticity**. Each
voxel is one trilinear hex (Q1) element. The Young's modulus is a function of
the local strain state, `E = E(ε)`, evaluated per Gauss point. The solver uses
**CuPy exclusively** for the hot path, **implicit connectivity** (DOF indices
are computed on-the-fly, never stored), a **Newton-Raphson outer loop** with
incremental load stepping, and **inexact-Newton inner PCG** solves.

**Target problem size**: ~300 million voxels, ~1 billion DOFs.
**Hardware**: 1 TB RAM, GPU with 48 GB VRAM.

> The original linear solver (`voxel_fem_solver.py`) and its CPU/Numba/CuPy
> backends have been removed. The reusable, backend-agnostic FEM primitives it
> contained now live in `geometry.py`, which the nonlinear solver imports.

## Commands

```bash
# Run all tests (47 tests). The core kernel/solver tests require CuPy/GPU —
# they skip if CuPy is absent; the layered-validation reference and multigrid
# tests are pure NumPy and always run.
python -m pytest tests/ -v --tb=short

# Run a single test file
python -m pytest tests/test_nonlinear_solver.py -v
python -m pytest tests/test_kernels_nl.py -v

# Run a single test by name (most critical kernel-correctness test)
python -m pytest tests/test_kernels_nl.py::test_matvec_tangent_fd_consistency -v

# Kernel benchmarks run as part of test_kernels_nl.py (test_benchmark_kernels[*])

# Analytical validation (layered composite) + formatted report
python -m pytest tests/validation/ -v
python tests/validation/report.py

# Example scripts (CuPy/GPU required)
# Multi-material: --preconditioner {jacobi,multigrid}, writes per-voxel .vti
# (use jacobi here — the random/checkerboard patterns oscillate at the voxel
# scale and stall geometric multigrid; see "Multigrid preconditioner" notes).
python example_multimaterial.py --pattern checkerboard --size 80 --preconditioner jacobi
python example_multimaterial.py --pattern random --vtk mm --format vti
python example_nonlinear.py

# Layered composite: FEM vs analytic effective modulus; writes per-voxel VTK
python example_layered.py                       # 40³, contrast 10, writes .vti
python example_layered.py --size 48 --contrast 100 --format vtk
```

**After every change to a `.py` file, run the test suite before proceeding.**
All 47 tests must stay green. **Do not move on if any test fails** — fix the
regression first. The single most important test is
`test_matvec_tangent_fd_consistency`: it verifies
`(f_int(u+hv) − f_int(u))/h ≈ K_t(u)·v`. If it fails, the NR solver will not
converge.

## Git workflow

**Always commit and push directly to the default `main` branch.** Do not create
feature branches for this project — commit straight onto `main` and `git push
origin main`. (This overrides the usual "branch first off the default branch"
convention.)

## Architecture

### The solver: `VoxelFEMSolverNonlinear` (`voxel_fem_solver_nonlinear.py`)

GPU-resident PCG vectors. Newton-Raphson outer loop with incremental load
stepping; inner PCG tolerance controlled by the Eisenstat-Walker criterion. Set
`is_linear=True` to skip the NR loop and run a single tight PCG — correct and
faster for linear materials (this forces `n_load_steps=1`).

Key constructor parameters (`__init__`):

| Param | Meaning |
|---|---|
| `voxels` | `(Nx,Ny,Nz)` uint8 — stiffness map or material IDs (see Material encoding) |
| `nu` | Poisson ratio (uniform) |
| `E_max`, `E_void` | reference max / void modulus |
| `constitutive` | `ConstitutiveModel` (default: linear) |
| `material_table` | optional length-256 table; if given, voxel values are material IDs |
| `is_linear` | skip NR, single tight PCG (`tol_pcg`, default 1e-8) |
| `n_load_steps` | load increments (default 10; ignored when `is_linear`) |
| `max_nr_iter`, `tol_nr` | NR loop limit / convergence `‖R‖/‖f‖ < tol_nr` |
| `max_pcg_iter` | inner PCG iteration cap |
| `preconditioner` | `"multigrid"` or `"jacobi"` (both work; multigrid is far fewer PCG iters) |
| `gpu_chunk_size` | elements per GPU chunk (default 2,000,000) |

Public interface: `apply_bc_displacement(node_ids, dof_ids, values)`,
`apply_force(node_ids, dof_ids, values)`, `solve()` → returns `(Ndof,)`
float32 NumPy array.

### The Newton-Raphson loop

```
u ← 0
for load_step s = 1 ... n_steps:
    f_target = (s / n_steps) · f_total
    for NR iteration k = 0 ... max_nr:
        f_int  = compute_f_int(u)          # internal force from current u
        R      = f_target - f_int          # residual
        R[bc]  = 0
        if ‖R‖ / ‖f_target‖ < tol_nr:  break
        η_k    = min(0.5, ‖R_k‖/‖R_{k-1}‖) # Eisenstat-Walker inner tol
        Δu     = pcg(K_tangent(u), R, precond(u), tol=η_k)  # inexact solve
        Δu[bc] = 0
        u ← u + Δu
```

BCs are **total values** (not increments); prescribed values scale with the
load fraction at each step. `R[bc]=0` after every `f_int`; `Δu[bc]=0` on every
update.

### Matrix-free K·u — the hot loop

Neither force nor tangent assembles K. For the linear solver `K_e = E_e · K_ref`
factored cleanly because `E` was constant per element. In the nonlinear case
`E` varies across the 8 Gauss points, so `K_ref` is **replaced by 8 precomputed
`B_gp` matrices** (each (6,24), pure geometry, identical for every element):

```
K_e = Σ_{gp=0}^{7}  w_gp · B_gp^T · D_t(ε_gp) · B_gp
```

`build_B_gauss()` in `kernels_cupy_nl.py` precomputes the 8 B matrices and
weights once at startup (it wraps `_B_matrix` from `geometry.py`).

**Consistency invariant**: `Σ_gp w_gp · B_gp^T · D(E=1,ν) · B_gp` must equal
`build_K_ref(ν)` — verified by `test_b_gauss_consistency`.

### Two hot-path kernels (`kernels_cupy_nl.py`)

- `compute_f_int_cupy` — internal force `f_int = Σ_e Σ_gp w · B^T · σ(ε)`.
  Called once per NR iteration.
- `matvec_tangent_cupy` — `K_tangent(u) · v` without assembling K. Called every
  PCG iteration. Both `u` (defines the tangent) and `v` are gathered per chunk,
  so gather bandwidth is ~2× the linear case — unavoidable.
- `build_D_cupy` — builds the `(chunk, 6, 6)` isotropic material matrix
  `D_t = E_t · D_ref(ν)`.
- `compute_avg_strain_per_element` — per-element average von Mises strain, used
  by the multigrid preconditioner to refresh tangent moduli.

### Implicit connectivity (mandatory, never store `conn_dof`)

Every CuPy kernel computes DOF indices on-the-fly inside each GPU chunk:

```python
sj = Nz + 1;  si = (Ny + 1) * sj
n0 = (e // (Ny*Nz)) * si + ((e // Nz) % Ny) * sj + (e % Nz)
node_offsets = [0, si, si+sj, sj, 1, si+1, si+sj+1, sj+1]
# dofs = 3*node + {0,1,2}  → 24 DOFs
```

This local node ordering matches `geometry.element_node_ids`. `conn_dof` is
**never stored** in the nonlinear path. (`geometry.matvec_Ku` / `build_diagonal`
do take an explicit `conn_dof` — those are the CPU reference / Jacobi-build
helpers, used for small grids and the Jacobi preconditioner only.)

### Element geometry

Every element is a unit cube (voxel side = 1). Gauss points come in on `[-1,1]³`;
`_B_matrix` in `geometry.py` maps them to `[0,1]³` internally, so `detJ = 0.125`.
`build_B_gauss()` wraps this to precompute the 8 B matrices once.

### Material encoding

Voxels are `uint8` (0–255). Two modes:

| Mode | How `E0_elem` is computed |
|---|---|
| Default (`material_table=None`) | `E = value/255 · (E_max − E_void) + E_void` — linear interpolation |
| `material_table` given | `E = material_table[value]` — voxel values are material IDs; table has 256 entries |

When a `material_table` is supplied, `E_max` is auto-set to `table.max()` unless
explicitly passed.

### Constitutive models (`constitutive.py`)

All models use the **von Mises equivalent strain** `ε_vm = sqrt(2/3 · dev(ε):dev(ε))`
as the scalar measure and clamp the output to `[E_void, E_max]`:

| Model | Formula | Physical meaning |
|---|---|---|
| `linear` | `E_t = E0` | reference |
| `softening` | `E_t = E0 · exp(−α · ε_vm)` | foams, soft tissue (α ≈ 10–100) |
| `hardening` | `E_t = E0 · (1 + β · ε_vm^n)` | rubber, bone (β ≈ 50, n ≈ 0.5) |

`ConstitutiveModel.tangent_modulus(E0_elem, eps_vm)` returns the tangent modulus
per element/Gauss point. All operate on CuPy float32 arrays. `constitutive.py`
raises `RuntimeError` at import if CuPy is absent.

### Multigrid preconditioner

`multigrid.py` — `GeometricMultigrid`: linear, **backend-parametric** (`xp` is
NumPy or CuPy, chosen by `use_gpu`). With `use_gpu=False` (default) it is the
pure-NumPy CPU reference the tests exercise; with `use_gpu=True` every level
operator, the Jacobi diagonal, and the whole V-cycle are GPU-resident (CuPy).
The GPU level operator (`matvec_Ku_cupy`/`build_diagonal_cupy` in
`kernels_cupy_nl.py`) uses **implicit connectivity** — `conn_dof` is never
stored on the GPU path (mandatory for the 300M-voxel budget; the NumPy path
still stores `conn_dof` as the CPU reference). Hierarchy built by repeated
2×2×2 averaging of the voxel image. Prolongation is trilinear interpolation
(tensor product of 1-D operators); restriction is full weighting, `R = P^T`
(the exact transpose, so the V-cycle is SPD); smoother is damped Jacobi
(ω = 0.67); coarsest level solved by CG. Coarse operators are scaled by
`2^level` (3-D elasticity stiffness K ∝ h) so the V-cycle approximates the
Galerkin operator and does not over-correct.

**BC-aware**: the fine constrained-DOF mask is injected coarse-ward
(`_set_bcs`/`_inject_bc`); constrained DOFs are identity rows in `matvec` and are
frozen by the smoother. Without this the coarse operators are singular (rigid-body
modes) and the V-cycle diverges. The solver hands the mask to the preconditioner
in `solve()` (via `NonlinearMultigrid.set_bc_mask`) once BCs are known.

`multigrid_nl.py` — `NonlinearMultigrid`: wraps `GeometricMultigrid` with an
`update(u_gpu)` method (**frozen-tangent strategy**). At each NR iteration it
computes the per-element average tangent modulus from current `u`, maps it to a
uint8 image, and rebuilds the `GeometricMultigrid` hierarchy (passing the stored
`bc_mask`). It builds the hierarchy with `use_gpu=True`, so the V-cycle runs
**entirely on the GPU** and `apply()` consumes/returns a CuPy residual with no
host round-trip per PCG iteration.

`preconditioner="multigrid"` is now working. On the layered example it converges
in a near grid-independent ~20–30 PCG iterations vs Jacobi's O(N) (e.g. 40³
parallel case: 31 vs 1083). Correctness is locked in by `tests/test_multigrid.py`
(adjoint property, convergent V-cycle, PCG agreement with Jacobi).

**Suitability caveat**: this is *geometric* multigrid — it coarsens by 2×2×2
averaging of the stiffness image. It shines when the stiffness varies smoothly or
structurally (layered, graded), but **stalls on coefficients that oscillate at
the voxel scale** (the `example_multimaterial.py` random/checkerboard patterns):
those average out under coarsening, so coarse grids cannot represent the fine
jumps and the V-cycle loses its coarse correction. Use Jacobi there. A
non-converged inner PCG (max_iter reached without hitting tol) now prints a
visible `⚠ PCG did NOT converge` warning rather than silently returning a wrong
field.

### Key cross-file dependencies

```
geometry.py                   — _B_matrix, build_K_ref, build_node_index,
                                 element_node_ids, matvec_Ku, build_diagonal, pcg
    ↑                              (pure NumPy, no GPU; imported by the rest)
constitutive.py               — ConstitutiveModel (CuPy)
kernels_cupy_nl.py            ← geometry (_B_matrix)
multigrid.py                  ← geometry
multigrid_nl.py               ← multigrid, kernels_cupy_nl
voxel_fem_solver_nonlinear.py ← geometry, kernels_cupy_nl, constitutive, multigrid_nl
```

### Float precision convention

- **float32** everywhere for large arrays (E0_elem, u, f, PCG vectors).
- **float64** only for the one-time `build_B_gauss()` / `build_K_ref()`
  accumulation (negligible cost), then cast to float32.
- The solver keeps all PCG vectors on GPU as float32 throughout. `.get()` is
  called only for convergence checks and printing.

### GPU memory budget (300M elements, 48 GB VRAM)

| Item | GB |
|---|---|
| `conn_dof` (implicit — not stored) | 0.0 |
| `E0_elem` (float32) | 1.1 |
| `B_gauss` (8 × 6×24 float32) | 0.0 |
| u, Δu, f_int, R, p, z (6 × ~3.4 GB) | 20.4 |
| Tangent matvec working buffers | 4.0 |
| Multigrid coarse levels | 1.5 |
| **Total** | **~27 GB** ✓ |

### Test layout

All tests live in `tests/` (47 total). Each file inserts `sys.path` to find the
parent directory — no `conftest.py`. The GPU tests are guarded with
`pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, ...)`, so the suite skips
cleanly with no GPU. Files:

- `test_constitutive.py` — material-model unit tests.
- `test_kernels_nl.py` — kernel correctness (B_gauss consistency, f_int zero/
  linearity, tangent symmetry, **finite-difference consistency**) + benchmarks.
  Builds its linear reference from `geometry` primitives via
  `_make_linear_reference()`.
- `test_nonlinear_solver.py` — NR convergence + physics correctness.
  `test_regression_to_linear` builds a CPU linear PCG reference from `geometry`
  (`matvec_Ku` + `build_diagonal` + `pcg`) and compares against the nonlinear
  solver with a linear constitutive model.
- `test_multigrid.py` — `GeometricMultigrid` correctness. The 3 CPU tests
  (`R = Pᵀ` adjoint property, convergent V-cycle iteration, PCG-with-multigrid
  agreement with Jacobi-PCG in fewer iterations) are pure NumPy and always run.
  3 further GPU-guarded tests cross-validate the CuPy backend against the NumPy
  reference: `matvec_Ku_cupy` vs `matvec_Ku`, GPU V-cycle `apply()` vs CPU
  `apply()`, and GPU-multigrid-PCG vs Jacobi-PCG.

### Analytical validation: layered composite (`tests/validation/`)

A self-contained validation package that checks the solver against the exact
closed-form effective moduli of a two-phase **layered composite** (harmonic
mean normal to the layers, arithmetic mean parallel to them). It runs the
solver in linear mode — `VoxelFEMSolverNonlinear(is_linear=True,
preconditioner="jacobi")` — since the original linear `VoxelFEMSolver` was
removed. All post-processing is rebuilt from `geometry.py` primitives (the
solver exposes no `_matvec`/stress API and existing files are not modified):
`f_int = K·u` via `matvec_Ku` for reaction-force / effective modulus, and
centroid stress/strain via `_B_matrix(0,0,0)`.

- `reference_solutions.py` — pure-NumPy `layered_reference()` (harmonic/
  arithmetic means + per-phase stress/strain) and `make_layered_voxels()`. The
  helper reports the **actual discretised moduli** (`E*_actual`) the uint8
  encoding produces, so quantisation is excluded from the FEM-vs-analytic
  comparison.
- `report.py` — post-processing helpers (`reaction_force`,
  `effective_modulus_from_reaction`, `element_stress`, `volume_average_*`) and a
  `run_validation_report()` script entry point.
- `test_reference.py` — pure-NumPy unit tests for the reference formulas (run
  even without a GPU).
- `test_layered.py` — FEM tests: normal/parallel effective modulus, contrast
  sweep `[2,10,100,1000]`, per-phase stress continuity (normal), per-phase
  iso-strain (parallel). Uses a minimal **3-2-1 anchoring** so lateral faces
  stay traction-free.

> **Poisson caveat:** the harmonic-mean-of-E (series) result is the
> *uniaxial-stress* modulus and is exact only with **no** Poisson coupling. At
> `nu=0.3` with free lateral faces, bonded layers share a common in-plane strain
> and the apparent normal modulus runs ~3.8% above the harmonic mean (real
> laminate coupling, not solver error — per-phase stress continuity and the
> parallel iso-strain case are exact to <0.01%). The series/normal tests
> therefore use `nu=0` (exact to <0.1% up to contrast 1000); the parallel and
> iso-strain checks use the physical `nu=0.3`. The Eshelby spherical-inclusion
> benchmark described in `validation.md` is **not** implemented.

### Layered example (`example_layered.py`)

Standalone script: builds a two-phase layered composite, solves it in linear
mode, and prints the FEM effective modulus next to the analytic harmonic
(normal) / arithmetic (parallel) means. Reuses the `tests/validation` helpers
(via a `sys.path` insert) and writes per-voxel strain/stress to VTK. Defaults
to `nu=0` for the normal case and `nu=0.3` for the parallel case (both means
exact); `--nu` overrides. Flags: `--size`, `--contrast`, `--f1`, `--nu`,
`--format {vti,vtk}` (default `vti`), `--vtk <prefix>` (`''` disables output),
`--preconditioner {multigrid,jacobi}` (default `multigrid`).

### VTK export (`vtk_export.py`)

Writes per-voxel (per-element) fields to a regular-voxel structured grid for
ParaView/VisIt. Pure NumPy except the optional `.vti` path. Fields are
**CELL_DATA** (one value per voxel) on an `Nx×Ny×Nz` cell grid.

Two writers:

- `save_stress_strain_vtk` / `write_structured_points_vtk` — **legacy ASCII
  STRUCTURED_POINTS** (`vtkImageData`). No external dependency. Writes
  `effective_strain`, `effective_stress`, all 12 strain/stress component
  scalars, and any `extra` fields.
- `save_stress_strain_vti` — **binary, zlib-compressed XML ImageData (.vti)**
  via `pyvista` (optional dep; lazy import, clear `RuntimeError` if missing).
  Writes the effective scalars plus strain/stress in **tensor notation** —
  `tensor_components=6` (compact VTK symmetric, default) or `9` (full 3×3, for
  ParaView's Tensor Glyph / eigenvalue filters) — and omits the redundant
  component scalars. Roundtrip is lossless (float64).

Two invariants the writers enforce:

1. **Cell ordering.** Per-element arrays are C-order over `(Nx,Ny,Nz)`
   (z-fastest); VTK cell data is x-fastest. Every field is reordered via
   `reshape(Nx,Ny,Nz).ravel(order='F')` (a row permutation for tensors) —
   pass `grid_shape=(Nx,Ny,Nz)` so this is correct.
2. **Strain shear.** Strain arrays use *engineering* shear (γ); the strain
   *tensor* off-diagonals are γ/2, so `voigt_to_sym6`/`voigt_to_full9` halve
   them for `is_strain=True`. Stress shears are unchanged. Voigt order is
   `[xx,yy,zz,xy,yz,xz]` (matches `geometry._B_matrix`), which equals VTK's
   symmetric-tensor order, so the 6-component form needs no remap.

> The per-voxel stress/strain comes from centroid recovery (`_B_matrix(0,0,0)`)
> in `tests/validation/report.py` (`element_stress`/`element_strain`). Under
> normal loading the effective **stress** is physically uniform (stress
> continuity) — it looks like noise in ParaView only because auto-scaling
> stretches the float-level variation; the effective **strain** carries the
> layer structure. Parallel loading is the mirror image.

## Constraints and style rules

- **CuPy only** for all hot-path computation. No NumPy in any kernel. `np` is
  for one-time setup (B_gauss precomputation, host arrays) only.
- **Implicit connectivity everywhere** in the GPU path — no `conn_dof` array.
- **float32** for all large GPU arrays; float64 only for one-time setup accumulation.
- **No sparse matrix assembly** — matrix-free end-to-end.
- **Graceful GPU check**: kernels/constitutive raise a clear `RuntimeError` at
  import if CuPy is unavailable.
- `chunk_size` is tunable on every kernel; default 2M is conservative for 48 GB.

## Quick reference: key formulas

- **Strain at Gauss point**: `ε_gp = B_gp · u_local`  — (6,) from (6,24)·(24,)
- **Von Mises strain**: `ε_vm = sqrt(2/3 · dev(ε):dev(ε))`
- **Tangent modulus**: `E_t = E_model(E0, ε_vm)`
- **Material matrix**: `D_t = E_t · D_ref(ν)` — (6,6), isotropic
- **Internal force**: `f_int = Σ_e Σ_gp w_gp · B_gp^T · D_t · B_gp · u_local`
- **Tangent matvec**: `K_t · v = Σ_e Σ_gp w_gp · B_gp^T · D_t(u) · B_gp · v_local`
- **NR update**: `K_t(u) · Δu = R = f_ext − f_int(u)`,  `u ← u + Δu`
- **Eisenstat-Walker**: `η_k = min(0.5, ‖R_k‖ / ‖R_{k−1}‖)`
- **Implicit node index**: `sj = Nz+1, si = (Ny+1)·sj`,
  `n0 = (e//(Ny·Nz))·si + ((e//Nz)%Ny)·sj + (e%Nz)`

---

## Known Issues

### GeometricMultigrid — GPU V-cycle (resolved)

The correctness bugs are **fixed** (see "Multigrid preconditioner" above and
`tests/test_multigrid.py`). The earlier CPU-V-cycle performance limitation is
also **resolved**: the V-cycle is now GPU-resident (`use_gpu=True`,
backend-parametric `GeometricMultigrid`), so `apply()` no longer round-trips to
the host per PCG iteration.

Profiling note: the bottleneck was *not* the GPU↔CPU transfer (~0.2% of
`solve()`) as previously assumed, but the NumPy V-cycle arithmetic itself —
chiefly `matvec_Ku`'s `np.add.at` scatter. The GPU port (`matvec_Ku_cupy` /
`build_diagonal_cupy`, implicit connectivity + `cp.add.at`) cut `example_layered`
`solve()` by ~2.6× at 40³ and ~15× at 80³ (per-V-cycle ~22× faster at 80³); the
gap widens with problem size.

**`example_layered.py` total run time** (full process: import + CUDA init + two
solves, normal & parallel; VTK disabled). BEFORE = CPU V-cycle (commit
`8a0c2e9`), AFTER = GPU V-cycle:

| Size | Voxels  | BEFORE   | AFTER  | Speedup       |
|------|---------|----------|--------|---------------|
| 20³  | 8,000   | 2.68 s   | 4.90 s | 0.55× (slower)|
| 40³  | 64,000  | 17.58 s  | 5.01 s | 3.5×          |
| 60³  | 216,000 | 41.98 s  | 6.20 s | 6.8×          |
| 80³  | 512,000 | 129.11 s | 9.79 s | 13.2×         |

BEFORE scales with element count; AFTER stays nearly flat (4.9 → 9.8 s over a
64× voxel increase) because it is dominated by fixed per-process overhead, not
the V-cycle. The crossover is ~40³: below it (e.g. 20³) the GPU version is
slower — at that scale the V-cycle is trivial either way and the run is
dominated by CUDA-context init, GPU hierarchy setup, and one-time CuPy kernel
JIT compilation. This per-process / kernel-launch overhead amortises away at
the target scale; the totals converge toward the solve-only speedup as N grows.

### Multigrid fix history (for reference)

`preconditioner="multigrid"` previously stalled (relative residual plateaued near
0.65–0.77) and produced a physically wrong displacement field. Three independent
defects had to be fixed:

1. **Prolongation condition** — `_prolongate` interpolated any node with a zero
   value (`np.any(e_fine_3d == 0)`), clobbering legitimately-zero injected values
   (e.g. at BCs). Replaced with a structural tensor-product trilinear operator
   (`_prolong_matrix_1d`): even fine nodes inject, odd nodes average two coarse
   neighbours.

2. **Restriction ≠ Pᵀ** — restriction used injection while prolongation averaged,
   so `R ≠ Pᵀ`, the V-cycle was non-symmetric, and PCG (which needs an SPD
   preconditioner) broke down. `_restrict` is now exactly `Pᵀ` (full weighting),
   built from the transpose of the same 1-D matrices. Verified by the **adjoint
   property** `⟨r, P·e⟩ = ⟨Rᵀ·r, e⟩` (a `‖e − R·P·e‖` roundtrip is *wrong* with
   full weighting since R·P scales interior nodes).

3. **Singular coarse operators / over-correction** — two further problems beyond
   the original write-up:
   - *No BC handling*: coarse operators were singular (unconstrained rigid-body
     modes), so the coarsest CG solve produced garbage and the V-cycle diverged.
     Fixed by injecting the constrained-DOF mask coarse-ward and treating those
     DOFs as identity rows (BC-aware, see above).
   - *Wrong coarse scaling*: every level used unit-cube `K_ref`, but 3-D
     elasticity stiffness scales as K ∝ h, so coarse corrections overshot (the
     V-cycle was a good PCG preconditioner but a *divergent* stationary
     iteration). Fixed by scaling coarse operators by `2^level`.

The Python triple-for-loops in `_coarsen_image`, `_restrict`, `_prolongate` were
also replaced with vectorised NumPy (`reduceat` / `einsum`).
