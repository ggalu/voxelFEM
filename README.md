# voxelFEM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24288861.svg)](https://doi.org/10.5281/zenodo.24288861)

A matrix-free finite-element solver for elasticity on voxel grids, running on the
GPU. Every voxel is one trilinear hexahedral (Q1) element, so a segmented CT scan
or any other 3-D image can be simulated directly, without meshing.

The solver is written in Python with [CuPy](https://cupy.dev). It never assembles
a stiffness matrix and never stores element connectivity: the degree-of-freedom
indices of an element are computed from its position in the grid when they are
needed. This keeps the memory per voxel small, which is the point of the design.
The intended problem size is a few hundred million voxels (about one billion
degrees of freedom) on a single GPU with 48 GB of memory.

## What it does

- **Linear and nonlinear elasticity.** The Young's modulus can depend on the local
  strain, `E = E(ε)`, evaluated at each of the eight Gauss points of an element.
  The available material models are linear, exponential softening and power-law
  hardening, all driven by the von Mises equivalent strain. Poisson's ratio is
  uniform.
- **Newton-Raphson with load stepping.** The load is applied in increments. Each
  Newton step solves the tangent system with a preconditioned conjugate gradient
  (PCG) method, whose tolerance follows the Eisenstat-Walker rule. For a linear
  material, `is_linear=True` skips the Newton loop and runs one PCG solve.
- **Two preconditioners.** Jacobi (diagonal) scaling, and a geometric multigrid
  V-cycle that runs entirely on the GPU. See
  [Choosing a preconditioner](#choosing-a-preconditioner) for when each one is the
  right choice.
- **Void handling.** Voxels marked as void or air are removed from the system, so
  porous samples and samples with a free surface can be simulated.
- **Output for ParaView.** Per-voxel strain and stress are written as compressed
  `.vti` files (tensor notation) or as legacy ASCII `.vtk` files.
- **A pipeline for CT data.** Scripts map CT grey levels to a Young's modulus
  field, run a uniaxial tension or compression test on the scan, and report the
  effective modulus.

## Requirements

- Python 3 (developed and tested with Python 3.12)
- An NVIDIA GPU with CUDA. The solver has no CPU fallback; the kernels raise a
  clear error on import if CuPy is missing.
- The packages in `requirements.txt`. Only NumPy and CuPy are needed for the
  solver itself. The other packages are needed by individual scripts, as noted in
  that file.

```bash
git clone https://github.com/ggalu/voxelFEM.git
cd voxelFEM
pip install -r requirements.txt
```

`requirements.txt` lists `cupy-cuda12x`. If your CUDA version differs, install
the matching CuPy package instead (see the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)).

There is no installable package. The modules sit in the repository root, and the
scripts are run from there.

## Quick start

The layered-composite example solves a two-phase laminate, once loaded normal to
the layers and once parallel to them, and compares the effective modulus with the
exact closed-form values (the harmonic and the arithmetic mean of the two moduli):

```bash
python example_layered.py --size 40 --contrast 10
```

```
  Loading            FEM E_eff    Analytic E_eff     Error
  ------------     -----------    --------------   -------
  normal               1.85036           1.85054    0.010%
  parallel             5.50981            5.5098    0.000%
```

Using the solver from your own code:

```python
import numpy as np
from geometry import build_node_index
from voxel_fem_solver_nonlinear import VoxelFEMSolverNonlinear

Nx = Ny = Nz = 40
voxels = np.full((Nx, Ny, Nz), 255, dtype=np.uint8)   # 255 -> E_max, 0 -> E_void
voxels[:, :, Nz // 2:] = 25                           # a softer upper half

solver = VoxelFEMSolverNonlinear(
    voxels, nu=0.3, E_max=10.0, E_void=1e-3,
    is_linear=True, preconditioner="multigrid",
)

node = build_node_index(Nx, Ny, Nz)                   # (Nx+1, Ny+1, Nz+1) node ids
bottom, top = node[:, :, 0].ravel(), node[:, :, -1].ravel()
for dof in range(3):                                  # clamp the bottom face
    solver.apply_bc_displacement(bottom, np.full(bottom.size, dof), np.zeros(bottom.size))
solver.apply_bc_displacement(top, np.full(top.size, 2), np.full(top.size, 0.01 * Nz))

u = solver.solve()                 # (Ndof,) float32; DOF index = 3 * node id + component
u = u.reshape(Nx + 1, Ny + 1, Nz + 1, 3)
```

Conventions that are worth knowing before you start:

- Every voxel is a unit cube. Lengths and displacements are in voxel units, so a
  strain `ε` along an axis with `N` voxels is a displacement of `ε · N`.
- By default, a `uint8` voxel value `v` is mapped to a modulus by linear
  interpolation, `E = E_void + (v / 255) · (E_max − E_void)`. If you pass a
  `material_table` with 256 entries, the voxel values are material IDs instead
  and `E = material_table[v]`.
- Boundary values are total values, not increments. With several load steps they
  are scaled by the load fraction of each step.
- To use a nonlinear material, pass a `ConstitutiveModel` from `constitutive.py`
  and leave `is_linear=False`. `example_nonlinear.py` shows this.

## Examples

| Script | What it shows |
|---|---|
| `example_layered.py` | Layered composite, compared with the analytic effective moduli. Options: `--size`, `--contrast`, `--f1`, `--nu`, `--preconditioner`, `--format {vti,vtk}`. |
| `example_nonlinear.py` | Uniform cube in tension with a linear and a power-law hardening material; prints the Newton and PCG history, the time per load step and the GPU memory in use. |
| `example_multimaterial.py` | Four materials assigned per voxel, either at random or as a checkerboard, through a `material_table`. Use `--preconditioner jacobi` here (see below). |
| `measure_vram.py` | Runs `example_layered.py` at several sizes and records the peak GPU memory. |

## Simulating CT data

The CT pipeline has two stages, each configured by a JSON file. The repository
includes a small sample (`assets/PA66GF30.npz`, 50³ voxels of a glass-fibre
reinforced polyamide), so the pipeline can be run as it is.

The input is an `.npz` file with two arrays of the same shape: `labels` (`uint8`
segmentation; the values 0 and 255 mean void and air) and `intensity` (the grey
levels).

```bash
# 1. Grey level -> Young's modulus. Adds the array `Young_modulus` to the .npz.
python preprocess.py                 # reads preprocess.json

# 2. Uniaxial test on the scan. Prints the effective modulus, writes fields.
python simulateCT.py                 # reads simulateCT.json
```

**Stage 1** reads the grey-level histogram, takes its peak as the matrix grey
level, and sets the fibre grey level so that the mean fibre fraction of the scan
equals the known fibre volume fraction. Each voxel then receives a modulus from
its fibre fraction through a mixing rule (geometric, Voigt or Reuss). The method
and its reasoning are written up in `modulus_mapping_method.md`.

**Stage 2** prescribes the displacement of the two end faces along the loading
axis and leaves the four side faces free of traction. Three further point
constraints remove the rigid-body motion. The effective modulus is the reaction
force on the loaded face, divided by the cross-section and the applied strain.
The script writes `<prefix>.vti` (modulus, strain and stress per voxel),
`<prefix>_displacement.npy` (nodal displacements, shape `(Nx+1, Ny+1, Nz+1, 3)`)
and `<prefix>_intensity.npy`.

Further tools around the pipeline:

| Script | Purpose |
|---|---|
| `preprocess_smooth.py` | Optional step after stage 1. Lowers the modulus gradually towards solid/void interfaces, which reduces the artificial stress peaks at the staircase corners of a voxel surface. |
| `warp_cupy.py` | Deforms the grey-level image by the computed displacement field, giving a synthetic "deformed scan", for example as a ground truth for digital volume correlation. It solves `X + u(X) = y` for every voxel `y` of the deformed image by a fixed-point iteration and samples the image with a cubic B-spline. Use this script when the deformed image must match the displacement field exactly. |
| `warp_intensity.py` | A simpler first-order version of the same operation, `I_def(x) = I_ref(x − u(x))`, with trilinear sampling. It writes `ref.npy` and `def.npy` and prints the mean normal strains. The error against the exact warp is about `ε² · x` for a uniform strain `ε` at distance `x` from the fixed end (0.25 voxel at 5% strain and 100 voxels). Run with `--test` for a self-check. |
| `recover_affine.py` | Recovers the best-fitting affine deformation from a reference and a deformed image by image registration; its diagonal gives the mean normal strains. Run with `--test` for a self-check. |
| `crop_numpy_array.py` | Crops every array in an `.npz` file to a given size and shows the central slices. |
| `view.py` | Shows the effective stress and strain of a result in [napari](https://napari.org). |

## Choosing a preconditioner

The multigrid preconditioner builds its coarse grids by averaging the stiffness
image over blocks of 2×2×2 voxels. It works well whenever this average still
describes the material: smooth or layered stiffness fields, and microstructures
whose features span several voxels, such as the fibres in a CT scan. In these
cases the number of PCG iterations is small and nearly independent of the grid
size. On the 40³ layered example (parallel loading), multigrid needs 31
iterations where Jacobi needs 1083. On the fibre sample, multigrid converges in
about 30 iterations, while Jacobi does not converge within 500.

Multigrid does not help when the modulus alternates from every voxel to the
next, as in the random and checkerboard patterns of `example_multimaterial.py`.
The averaging then removes all stiffness variation from the coarse grids. Use
`preconditioner="jacobi"` for such fields.

If the inner PCG reaches its iteration limit without meeting the tolerance, the
solver prints a visible warning. Do not use the result in that case.

## Tests and validation

```bash
python -m pytest tests/ -v            # 59 tests
python tests/validation/report.py     # layered-composite report
```

The GPU tests are skipped when CuPy is not installed; the reference-formula and
CPU multigrid tests are pure NumPy and always run. The most important single
test is `test_matvec_tangent_fd_consistency`, which checks that the tangent
operator is the derivative of the internal force.

The validation package in `tests/validation/` compares the solver with the exact
effective moduli of a layered composite for stiffness contrasts from 2 to 1000.
One point needs care when reading these results: the harmonic mean is the exact
modulus normal to the layers only without Poisson coupling, so those tests use
`nu = 0`. With `nu = 0.3` and free side faces, the layers constrain each other
in the plane and the apparent modulus lies about 3.8% above the harmonic mean.
This is a property of the laminate, not a solver error.

## Code layout

| File | Content |
|---|---|
| `voxel_fem_solver_nonlinear.py` | `VoxelFEMSolverNonlinear`: boundary conditions, load stepping, Newton loop, PCG. |
| `kernels_cupy_nl.py` | The GPU kernels: internal force and the tangent matrix-vector product, both matrix-free. |
| `constitutive.py` | `ConstitutiveModel`: the strain-dependent modulus. |
| `multigrid.py`, `multigrid_nl.py` | Geometric multigrid (NumPy reference and CuPy backend) and its wrapper that refreshes the tangent moduli at each Newton step. |
| `geometry.py` | Element matrices, node numbering and CPU reference routines, in pure NumPy. |
| `vtk_export.py` | Writers for `.vti` and legacy `.vtk` files. |

Large GPU arrays are stored in single precision. The per-element modulus is
stored in half precision to save memory. Displacements are never stored in half
precision, because the strain is a difference of nearly equal nodal values and
would lose all accuracy.

## Limitations

- An NVIDIA GPU is required.
- The elements are unit cubes; anisotropic voxel sizes are not supported.
- The material is isotropic, with one Poisson's ratio for the whole model. The
  nonlinearity is a strain-dependent modulus under small strains; there is no
  plasticity and no finite-strain kinematics.
- The multigrid preconditioner is geometric and has the restriction described
  above.

## License

Released under the MIT License; see [LICENSE](LICENSE).

## Citing

Each release is archived on Zenodo. To cite the software, use the DOI
[10.5281/zenodo.24288861](https://doi.org/10.5281/zenodo.24288861), which always
resolves to the latest version; the Zenodo page also lists a separate DOI for
each version. `CITATION.cff` holds the citation metadata.

## Authors

Georg C. Ganzenmüller, Albert-Ludwigs-Universität Freiburg, Germany.

Lokesh Metta, Albert-Ludwigs-Universität Freiburg, Germany, contributed the
Lagrangian warp in `warp_cupy.py` and its tests.
