# Voxel FEM Solver — Analytical Validation Tests

## Overview

This instruction file implements two analytical validation benchmarks for
`voxel_fem_solver.py` (the linear elastic voxel FEM solver):

1. **Layered composite** — exact closed-form effective moduli (harmonic/arithmetic
   mean). Zero geometric approximation error on a voxel grid. Tests multi-phase
   behaviour across contrast ratios up to E_1/E_2 = 1000.

2. **Eshelby spherical inclusion** — closed-form uniform stress inside a spherical
   inclusion in an infinite matrix. Tests 3D stress concentration, spatial
   uniformity inside the inclusion, and the r⁻³ far-field decay.

Both tests produce a self-contained validation report. All reference solutions
are computed in Python from first principles — no external reference data files.

---

## Mandatory: Run Tests After Every Change

**After every modification to any `.py` file, run the full test suite:**

```bash
python -m pytest tests/ -v --tb=short
```

**Do not proceed if any test fails.**

---

## Repository Layout (additions only)

```
tests/
  validation/
    __init__.py
    reference_solutions.py    # new: analytical reference computations
    test_layered.py           # new: layered composite validation tests
    test_eshelby.py           # new: Eshelby inclusion validation tests
    report.py                 # new: validation report generator
```

Do not modify any existing files. All new code lives under `tests/validation/`.

---

## Step-by-Step Instructions

---

### Step 1 — Reference solutions module (`tests/validation/reference_solutions.py`)

Create this module first. It contains all analytical computations. It must be
**pure Python/NumPy** with no FEM solver dependencies — it can be imported and
verified independently.

Run tests after this step.

#### 1a — Layered composite reference

```python
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
      'E_eff_normal'   : float — effective modulus, loading normal to layers
                                 (series / harmonic mean arrangement)
      'E_eff_parallel' : float — effective modulus, loading parallel to layers
                                 (parallel / arithmetic mean arrangement)
      'sigma_1_normal' : float — stress in phase 1 under unit far-field strain,
                                 normal loading (stress partitioning)
      'sigma_2_normal' : float — stress in phase 2, normal loading
      'eps_1_parallel' : float — strain in phase 1 under unit far-field stress,
                                 parallel loading
      'eps_2_parallel' : float — strain in phase 2, parallel loading
    """
    f2 = 1.0 - f1

    # Series (normal to layers): harmonic mean
    E_eff_normal = 1.0 / (f1/E1 + f2/E2)

    # Parallel (in-plane): arithmetic mean
    E_eff_parallel = f1*E1 + f2*E2

    # Stress partitioning under unit applied strain, normal loading:
    # Both phases have the same strain (iso-strain assumption is WRONG for
    # normal loading — it is the STRESS that is uniform, not the strain).
    # Under unit far-field STRESS sigma_inf = 1:
    #   sigma_1 = sigma_2 = sigma_inf = 1  (stress continuity across interface)
    #   eps_1 = 1/E1,  eps_2 = 1/E2
    # Under unit far-field STRAIN eps_inf = 1:
    #   eps_avg = f1*eps_1 + f2*eps_2 = 1, with sigma_1 = sigma_2 (equilibrium)
    #   => sigma = E_eff_normal * 1
    #   => eps_1 = sigma/E1,  eps_2 = sigma/E2
    sigma_unit_strain = E_eff_normal          # sigma in both phases
    eps_1_normal = sigma_unit_strain / E1     # strain in phase 1
    eps_2_normal = sigma_unit_strain / E2     # strain in phase 2
    sigma_1_normal = sigma_unit_strain        # = sigma_2_normal
    sigma_2_normal = sigma_unit_strain

    # Strain partitioning under unit applied stress, parallel loading:
    # Both phases have same strain (iso-strain):
    #   eps_1 = eps_2 = eps_inf
    # Stress partitioning: sigma_1 = E1*eps_inf, sigma_2 = E2*eps_inf
    eps_1_parallel = 1.0    # identical strain in both phases
    eps_2_parallel = 1.0

    return {
        'E_eff_normal'   : E_eff_normal,
        'E_eff_parallel' : E_eff_parallel,
        'sigma_1_normal' : sigma_1_normal,
        'sigma_2_normal' : sigma_2_normal,
        'eps_1_parallel' : eps_1_parallel,
        'eps_2_parallel' : eps_2_parallel,
    }
```

#### 1b — Eshelby spherical inclusion reference

```python
def eshelby_sphere_reference(E_i, E_m, nu, sigma_inf=1.0):
    """
    Closed-form stress and strain inside a spherical inclusion
    in an infinite isotropic matrix under far-field uniaxial stress
    sigma_inf in the z-direction.

    Both phases share the same Poisson ratio nu (required for standard
    Eshelby theory with a single S tensor).

    Parameters
    ----------
    E_i      : float — Young's modulus of inclusion
    E_m      : float — Young's modulus of matrix
    nu       : float — Poisson ratio (both phases)
    sigma_inf: float — far-field uniaxial stress in z (default 1.0)

    Returns
    -------
    dict with keys:
      'eps_in'     : np.ndarray (6,) — strain inside inclusion [xx,yy,zz,xy,xz,yz]
      'sigma_in'   : np.ndarray (6,) — stress inside inclusion
      'eps_inf'    : np.ndarray (6,) — far-field strain
      'beta_zz'    : float — stress concentration factor sigma_zz^in / sigma_inf
      'beta_xx'    : float — lateral stress concentration sigma_xx^in / sigma_inf
      'S_tensor'   : np.ndarray (6,6) — Eshelby tensor (Voigt notation)
    """
    import numpy as np

    # --- Lame constants ---
    lam_m = E_m * nu / ((1+nu)*(1-2*nu))
    mu_m  = E_m / (2*(1+nu))
    lam_i = E_i * nu / ((1+nu)*(1-2*nu))
    mu_i  = E_i / (2*(1+nu))

    # --- Eshelby tensor for a sphere (Voigt 6x6, Mura 1987 Table 11.1) ---
    # Normal block (indices 0-2 = xx,yy,zz)
    S11 = (7 - 5*nu) / (15*(1 - nu))
    S12 = (5*nu - 1) / (15*(1 - nu))
    # Shear block (indices 3-5 = xy,xz,yz) — factor of 2 from Voigt convention
    S44 = (4 - 5*nu) / (15*(1 - nu))

    S = np.zeros((6,6))
    for i in range(3):
        S[i,i] = S11
    for i in range(3):
        for j in range(3):
            if i != j:
                S[i,j] = S12
    for i in range(3,6):
        S[i,i] = S44   # in engineering shear Voigt convention

    # --- Matrix stiffness tensor C_m (6x6 Voigt) ---
    def stiffness_tensor(lam, mu):
        C = np.zeros((6,6))
        for i in range(3):
            C[i,i] = lam + 2*mu
        for i in range(3):
            for j in range(3):
                if i != j:
                    C[i,j] = lam
        for i in range(3,6):
            C[i,i] = mu   # note: shear modulus, not 2*mu, in this convention
        return C

    C_m = stiffness_tensor(lam_m, mu_m)
    C_i = stiffness_tensor(lam_i, mu_i)
    C_m_inv = np.linalg.inv(C_m)

    # --- Far-field quantities ---
    # sigma_inf = [0, 0, sigma_inf, 0, 0, 0]  (uniaxial in z)
    sig_inf_vec = np.array([0., 0., sigma_inf, 0., 0., 0.])
    eps_inf = C_m_inv @ sig_inf_vec         # far-field strain

    # --- Interior strain (Eshelby 1957, Mura 1987 Eq 4.10) ---
    # eps_in = (I + S · C_m^{-1} · (C_i - C_m))^{-1} · eps_inf
    dC = C_i - C_m
    A  = np.eye(6) + S @ C_m_inv @ dC
    eps_in  = np.linalg.solve(A, eps_inf)
    sigma_in = C_i @ eps_in

    return {
        'eps_in'   : eps_in,
        'sigma_in' : sigma_in,
        'eps_inf'  : eps_inf,
        'beta_zz'  : sigma_in[2] / sigma_inf,
        'beta_xx'  : sigma_in[0] / sigma_inf,
        'S_tensor' : S,
    }
```

#### 1c — Far-field decay reference

```python
def eshelby_matrix_stress(r, R, E_i, E_m, nu, sigma_inf=1.0):
    """
    Stress in the MATRIX along the z-axis (x=0, y=0, z=r) outside the
    spherical inclusion of radius R, under far-field uniaxial stress sigma_inf.

    Uses the Eshelby/Mindlin exterior solution. Along the z-axis:

      sigma_zz(r) = sigma_inf · [1 + A/r³ + B/r⁵]    for r > R

    where A, B depend on E_i/E_m, nu, R. Returns sigma_zz as a function of r.

    Parameters
    ----------
    r        : np.ndarray — radial distances from centre (must be > R)
    R        : float — inclusion radius
    E_i, E_m : float — moduli
    nu       : float — Poisson ratio
    sigma_inf: float

    Returns
    -------
    sigma_zz : np.ndarray — stress along z-axis at distances r
    """
    import numpy as np

    # Get inclusion strain from Eshelby solution
    ref = eshelby_sphere_reference(E_i, E_m, nu, sigma_inf)
    eps_in_zz = ref['eps_in'][2]
    eps_inf_zz = ref['eps_inf'][2]
    # Eigenstrain: eps* = eps_in - eps_inf
    eps_star_zz = eps_in_zz - eps_inf_zz

    # Along z-axis, the exterior sigma_zz decays as (Eshelby 1957, exterior solution):
    # sigma_zz/sigma_inf = 1 - (R/r)^3 * [C1 + C2*(R/r)^2]
    # Exact coefficients from Mura (1987) Table 11.2 for uniaxial loading,
    # sphere, along the loading axis:
    rho = R / r    # (N,)
    mu_m = E_m / (2*(1+nu))

    # Simplified form valid along z-axis only:
    # (from Timoshenko & Goodier, or Mura 1987 Sec 11.3)
    alpha_E = (E_i - E_m) / (E_i + E_m * (1 - 2*nu)/(1-nu))  # approximate
    sigma_zz = sigma_inf * (1.0
               - rho**3 * (1.0 - 1.5*(1-2*nu)/(1-nu))
               + 1.5 * rho**5 * (1.0/(1-nu)))
    # Note: the exact exterior solution requires numerical evaluation of
    # the Papkovich-Neuber potentials. For the FEM test, use only the
    # qualitative check that sigma_zz -> sigma_inf as r -> inf (r > 3R),
    # and that the decay is monotone. The exact formula is provided in
    # eshelby_exterior_numerical() below.

    return sigma_zz


def eshelby_exterior_numerical(x, y, z, R, E_i, E_m, nu, sigma_inf=1.0,
                                n_gauss=50):
    """
    Numerically integrate the Eshelby exterior solution at arbitrary (x,y,z)
    outside the sphere. Uses the Papkovich-Neuber potential representation.

    This is the reference for the r^{-3} decay test. Accurate to ~0.1%
    for r > 1.1*R.

    For the FEM validation, the primary check is:
      |sigma_zz(r) - sigma_inf| / sigma_inf ~ (R/r)^3   for r >> R

    So at r = 3R: error ~ (1/3)^3 ~ 3.7%
    At r = 5R: error ~ (1/5)^3 ~ 0.8%

    Returns sigma tensor (6,) at point (x,y,z).
    """
    # Implementation uses the known series expansion.
    # For the test suite, we only need the z-axis values which are given
    # analytically by eshelby_matrix_stress().
    # Full 3D exterior solution: see Mura (1987) Chapter 11.
    raise NotImplementedError(
        "Full 3D exterior solution not implemented. "
        "Use eshelby_matrix_stress() for z-axis values, "
        "or use only the qualitative decay test (r^{-3} slope)."
    )
```

Add unit tests for `reference_solutions.py` in `tests/validation/test_reference.py`:

1. `layered_reference(E1=E2, f1=0.5, nu)` → `E_eff_normal == E_eff_parallel == E1`
   (homogeneous limit).
2. `layered_reference(E1, E2=E1*10, f1=0.5, nu)` → `E_eff_normal < E_eff_parallel`
   (series is always softer than parallel).
3. `eshelby_sphere_reference(E_i=E_m, nu)` → `beta_zz == 1.0`, `beta_xx == 0.0`
   (no inclusion contrast → no stress concentration).
4. `eshelby_sphere_reference(E_i>>E_m, nu)` → `beta_zz > 1` (stiff inclusion
   amplifies stress).
5. `eshelby_sphere_reference(E_i<<E_m, nu)` → `beta_zz < 1` (soft inclusion
   shields stress).
6. Stress equilibrium inside inclusion: `sigma_in[0] + sigma_in[1] + sigma_in[2]`
   should be finite and physically reasonable (not a test of a specific value,
   but a sanity check that the solve did not produce NaN/Inf).

Run tests.

---

### Step 2 — Voxel geometry helpers (`tests/validation/reference_solutions.py`, continued)

Add geometry construction functions. These build NumPy voxel arrays ready to
pass to `VoxelFEMSolver`.

#### 2a — Layered composite voxel image

```python
def make_layered_voxels(Nx, Ny, Nz, layer_axis, E1, E2, f1, E_max):
    """
    Build a (Nx,Ny,Nz) uint8 voxel array with alternating layers of E1 and E2.

    Parameters
    ----------
    Nx,Ny,Nz   : int — grid dimensions
    layer_axis  : int — 0=x, 1=y, 2=z — axis normal to layers
    E1, E2      : float — Young's moduli
    f1          : float — volume fraction of phase 1
                  Rounded to nearest whole layer; actual f1 is reported.
    E_max       : float — modulus corresponding to voxel value 255

    Returns
    -------
    voxels   : np.ndarray (Nx,Ny,Nz) uint8
    f1_actual: float — actual volume fraction after rounding
    """
    import numpy as np

    N = [Nx, Ny, Nz][layer_axis]
    n1 = max(1, round(f1 * N))
    n2 = N - n1
    f1_actual = n1 / N

    val1 = int(round(E1 / E_max * 255))
    val2 = int(round(E2 / E_max * 255))
    val1 = np.clip(val1, 1, 255)
    val2 = np.clip(val2, 1, 255)

    stripe = np.array([val1]*n1 + [val2]*n2, dtype=np.uint8)  # (N,)

    # Broadcast to 3D
    voxels = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    if layer_axis == 0:
        voxels[:, :, :] = stripe[:, np.newaxis, np.newaxis]
    elif layer_axis == 1:
        voxels[:, :, :] = stripe[np.newaxis, :, np.newaxis]
    else:
        voxels[:, :, :] = stripe[np.newaxis, np.newaxis, :]

    return voxels, f1_actual
```

#### 2b — Spherical inclusion voxel image

```python
def make_sphere_voxels(N, R_voxels, E_inclusion, E_matrix, E_max,
                       center=None):
    """
    Build an (N,N,N) uint8 voxel array with a spherical inclusion.

    Parameters
    ----------
    N           : int — cube side length in voxels
    R_voxels    : float — inclusion radius in voxels
    E_inclusion : float — inclusion Young's modulus
    E_matrix    : float — matrix Young's modulus
    E_max       : float — modulus at voxel value 255
    center      : tuple (cx,cy,cz) in voxels, default = (N/2, N/2, N/2)

    Returns
    -------
    voxels      : np.ndarray (N,N,N) uint8
    R_actual    : float — actual radius (= R_voxels, sphere is not rounded)
    n_inclusion : int — number of inclusion voxels
    f_inclusion : float — actual inclusion volume fraction
    """
    import numpy as np

    if center is None:
        cx = cy = cz = (N - 1) / 2.0
    else:
        cx, cy, cz = center

    val_i = int(np.clip(round(E_inclusion / E_max * 255), 1, 255))
    val_m = int(np.clip(round(E_matrix    / E_max * 255), 1, 255))

    ix, iy, iz = np.mgrid[0:N, 0:N, 0:N]
    r2 = (ix - cx)**2 + (iy - cy)**2 + (iz - cz)**2
    inside = r2 <= R_voxels**2

    voxels = np.full((N,N,N), val_m, dtype=np.uint8)
    voxels[inside] = val_i

    n_inclusion = int(inside.sum())
    f_inclusion = n_inclusion / N**3

    return voxels, R_voxels, n_inclusion, f_inclusion
```

**Verify `make_sphere_voxels`**: for large N and R = N/4, the voxel volume
fraction should approach `(4/3)π(R/N)³` within 2%. Add this as a test.

Run tests.

---

### Step 3 — Layered composite FEM tests (`tests/validation/test_layered.py`)

Each test follows the same pattern:
1. Build voxel array with `make_layered_voxels`.
2. Run `VoxelFEMSolver`.
3. Post-process FEM solution to extract effective modulus or stress field.
4. Compare to `layered_reference`.

#### Post-processing helpers

Add these to `tests/validation/report.py` (they are reused in Eshelby tests too):

```python
def reaction_force(u, solver):
    """
    Compute the reaction force on the top face (constrained to prescribed
    displacement) by computing f_int = K·u and summing over the top-face DOFs.

    Returns the total force in each direction.
    """
    f_int = solver._matvec(u)
    # Top-face nodes: those with z-index = Nz
    # Their z-DOFs give the reaction force in z
    ...

def volume_average_stress(u, solver, mask_voxels):
    """
    Compute volume-averaged stress components over voxels where mask is True.

    Strategy: for each masked element, compute sigma = D·B·u_local at
    element centroid (single-point quadrature is sufficient for averaging),
    then average over elements.

    Returns sigma_avg: (6,) array [xx,yy,zz,xy,xz,yz].
    """
    ...

def volume_average_strain(u, solver, mask_voxels):
    """Same as volume_average_stress but returns strain."""
    ...

def effective_modulus_from_reaction(u, solver, load_axis, delta, L):
    """
    Compute effective Young's modulus from reaction force:
      E_eff = (F / A) / (delta / L)
    where F is total reaction force, A is cross-section area, delta is
    prescribed displacement, L is domain size in load direction.
    """
    ...
```

**Implementation note for `volume_average_stress`**: computing stress at the
centroid of each voxel element (i.e. at ξ=η=ζ=0.5) requires evaluating the
B matrix at that point. For a unit cube, this is a single call to `_B_matrix(0,0,0)`
which can be computed once and reused for all elements in a chunk. This is
significantly cheaper than evaluating all 8 Gauss points.

#### Test 3a — Effective modulus, normal loading

```python
def test_layered_normal_modulus():
    """
    Verify effective Young's modulus under loading normal to layers
    matches the harmonic mean (series arrangement) to within 1%.

    Setup:
      - 40×40×40 voxels
      - Layers in z-direction (axis=2): 20 voxels of E1, 20 of E2
      - E1 = 1.0, E2 = 10.0, nu = 0.3, E_max = 10.0
      - Prescribed displacement delta = 0.001·L on top face (z=40)
      - Bottom face fully fixed
      - Lateral faces: FREE (not constrained — this is important)

    Expected: E_eff_z = harmonic mean = 2·E1·E2/(E1+E2) = 1.818...
    Tolerance: 1% relative error.
    """
```

**Lateral face BCs**: for effective modulus measurement, lateral faces must be
**free** (no x or y constraints beyond the bottom/top nodes). Constraining
lateral faces introduces Poisson-effect errors. The bottom face should only
constrain z-displacement (not x or y) to avoid over-constraining.

Implement this BC set carefully:
```python
# Bottom face (z=0): fix z only
bottom_nodes = node_idx[:, :, 0].ravel()
solver.apply_bc_displacement(bottom_nodes, np.full(len(bottom_nodes), 2),
                             np.zeros(len(bottom_nodes)))  # dof=2 is z
# Top face (z=Nz): prescribe z displacement
top_nodes = node_idx[:, :, -1].ravel()
solver.apply_bc_displacement(top_nodes, np.full(len(top_nodes), 2),
                             np.full(len(top_nodes), delta))
# Fix one node to prevent rigid body modes in x and y
corner = node_idx[0, 0, 0]
solver.apply_bc_displacement(np.array([corner, corner]),
                             np.array([0, 1]),
                             np.array([0.0, 0.0]))
```

#### Test 3b — Effective modulus, parallel loading

```python
def test_layered_parallel_modulus():
    """
    Same geometry as 3a but loading in x-direction (parallel to layers).
    Expected: E_eff_x = arithmetic mean = f1*E1 + f2*E2.
    Tolerance: 0.5% (parallel arrangement is exact on voxel grid with
    no geometric approximation error).
    """
```

#### Test 3c — Contrast ratio sweep

```python
@pytest.mark.parametrize("contrast", [2, 10, 100, 1000])
def test_layered_contrast_sweep(contrast):
    """
    Verify harmonic mean formula across contrast ratios E2/E1 = 2,10,100,1000.
    Grid: 40×40×40. Loading: normal to layers.
    Tolerance: 2% for all contrast ratios.

    High contrast (1000) is the hardest case for CG convergence —
    verify that the solver still converges within max_iter=1000.
    If using GMG preconditioner, it should converge in < 50 iterations
    at any contrast.
    """
```

#### Test 3d — Per-phase stress verification

```python
def test_layered_per_phase_stress():
    """
    Under normal loading, both phases must carry the same stress
    (stress continuity across the interface — exact by equilibrium).

    Measure: volume-average sigma_zz in phase 1 and phase 2 separately.
    Expected: |sigma_1_zz - sigma_2_zz| / sigma_avg < 1%

    Also verify: volume-average sigma_zz = E_eff * applied_strain (global equilibrium).
    """
```

#### Test 3e — Per-phase strain verification, parallel loading

```python
def test_layered_per_phase_strain():
    """
    Under parallel loading (x-direction), both phases carry the same strain
    (iso-strain assumption — exact by compatibility).

    Measure: volume-average eps_xx in phase 1 and phase 2.
    Expected: |eps_1_xx - eps_2_xx| / eps_avg < 1%
    """
```

Run tests after implementing all of the above.

---

### Step 4 — Eshelby FEM tests (`tests/validation/test_eshelby.py`)

These tests are geometrically approximate (sphere = staircase of voxels, finite
domain ≠ infinite), so tolerances are wider. The accuracy improves with larger
domain-to-radius ratio.

#### Standard test configuration

Define once at module level and reuse across all tests:

```python
# Standard Eshelby test configuration
ESHELBY_CONFIGS = {
    'stiff': dict(
        N=80, R_voxels=8,           # domain = 10R in each direction
        E_i=10.0, E_m=1.0,
        E_max=10.0, nu=0.3,
        tol_interior=0.05,          # 5% tolerance on interior stress
        tol_uniformity=0.10,        # 10% max coefficient of variation inside
    ),
    'soft': dict(
        N=80, R_voxels=8,
        E_i=0.1, E_m=1.0,
        E_max=1.0, nu=0.3,
        tol_interior=0.05,
        tol_uniformity=0.10,
    ),
    'moderate': dict(
        N=48, R_voxels=8,           # domain = 6R — faster, slightly less accurate
        E_i=5.0, E_m=1.0,
        E_max=5.0, nu=0.3,
        tol_interior=0.08,
        tol_uniformity=0.12,
    ),
}
```

**BCs for Eshelby test**: uniaxial tension in z-direction, approximating
far-field uniaxial stress. Apply **displacement BCs** that correspond to the
far-field strain `eps_inf_zz = sigma_inf / E_m`:

```python
eps_inf = sigma_inf / E_m
delta   = eps_inf * L              # L = N voxels

# Bottom face (z=0): fix z
# Top face (z=N): prescribe delta in z
# Lateral faces: prescribe Poisson contraction to minimise boundary effects
# u_x on x-faces: -nu * eps_inf * x  (applied as displacement BC)
# u_y on y-faces: -nu * eps_inf * y
```

Applying Poisson contraction on lateral faces reduces boundary errors from
~10% to ~3% for a 10R domain. This is a significant improvement and is
worth the extra BC implementation effort. Make it optional via a
`poisson_lateral_bc=True` flag.

#### Test 4a — Interior stress concentration (stiff inclusion)

```python
def test_eshelby_interior_stress_stiff():
    """
    Verify average sigma_zz inside a stiff spherical inclusion (E_i/E_m = 10)
    matches the Eshelby prediction (beta_zz) within 5%.

    Steps:
    1. Build 80³ voxel grid with sphere of R=8 voxels at centre.
    2. Solve with uniaxial displacement BCs (sigma_inf = 1 equivalent).
    3. Identify inclusion voxels (same sphere mask as construction).
    4. Compute volume-average sigma_zz over inclusion voxels.
    5. Compare to beta_zz = eshelby_sphere_reference(E_i=10, E_m=1, nu=0.3)['beta_zz'].

    Expected: |FEM_beta_zz - Eshelby_beta_zz| / Eshelby_beta_zz < 5%
    """
```

#### Test 4b — Interior stress concentration (soft inclusion)

```python
def test_eshelby_interior_stress_soft():
    """
    Same as 4a but E_i/E_m = 0.1 (soft inclusion, stress shielding).
    Expected: beta_zz < 1. Tolerance: 5%.
    """
```

#### Test 4c — Lateral stress components

```python
def test_eshelby_lateral_stress():
    """
    Verify beta_xx (= beta_yy by symmetry) inside the inclusion.
    Test for both stiff and soft cases.

    Note: beta_xx is smaller in magnitude than beta_zz and more sensitive
    to boundary effects. Tolerance: 8%.
    """
```

#### Test 4d — Interior stress uniformity

```python
def test_eshelby_interior_uniformity():
    """
    Eshelby's theorem: stress inside an ellipsoidal inclusion in an
    infinite matrix is UNIFORM. On a finite voxel domain, verify that
    the spatial variation is small:

      CV = std(sigma_zz inside inclusion) / mean(sigma_zz inside inclusion)

    Expected: CV < 10% for a 10R domain.

    Also check: the stress field has the correct symmetry:
      - sigma_zz should be symmetric about the x=cx and y=cy planes
      - sigma_xx ≈ sigma_yy (transverse isotropy under uniaxial load)
      - Off-diagonal stresses sigma_xy, sigma_xz, sigma_yz ≈ 0 inside the
        inclusion (expected to be < 5% of sigma_zz)
    """
    # Symmetry check: compare sigma_zz at (cx+dr, cy, cz) vs (cx-dr, cy, cz)
    # for several values of dr. Difference should be < 1% of sigma_zz.
```

#### Test 4e — Far-field stress recovery

```python
def test_eshelby_far_field_decay():
    """
    Verify that sigma_zz in the matrix along the z-axis decays back to
    sigma_inf as r increases beyond R.

    Measurement points along z-axis: r/R = 1.5, 2.0, 3.0, 4.0
    (avoiding the interface region r/R < 1.2 where voxel approximation
    is rough).

    Expected qualitative behaviour:
    - sigma_zz(r=1.5R) deviates from sigma_inf (near-field effect)
    - sigma_zz(r=3.0R) is within 15% of sigma_inf
    - sigma_zz(r=4.0R) is within 8% of sigma_inf
    - The deviation |sigma_zz - sigma_inf| is monotonically decreasing with r
      for r > 1.5R (confirms r^{-3} decay trend)

    Extract sigma_zz along the z-axis by identifying the column of elements
    at (ix=Nx//2, iy=Ny//2, iz=*) and computing their centroid stress.
    """
```

#### Test 4f — Domain size convergence

```python
@pytest.mark.parametrize("domain_over_R", [4, 6, 8, 10])
def test_eshelby_domain_convergence(domain_over_R):
    """
    Verify that the interior stress prediction converges to the Eshelby
    solution as the domain-to-radius ratio increases.

    For each domain_over_R = N/R:
      N = domain_over_R * R_voxels,  R_voxels = 8 (fixed)

    Expected: error in beta_zz decreases monotonically with domain_over_R.
    Approximate expected errors:
      domain/R = 4:  ~15%
      domain/R = 6:  ~8%
      domain/R = 8:  ~5%
      domain/R = 10: ~3%

    This is a convergence test — the key assertion is monotonic decrease,
    not specific error values (which depend on BCs and voxel resolution).
    """
```

Mark this test with `@pytest.mark.slow` — the 10R case (N=80) takes significant
time and should be excluded from the default fast test run:

```bash
python -m pytest tests/validation/ -v -m "not slow"    # fast suite
python -m pytest tests/validation/ -v                  # full suite
```

#### Test 4g — Contrast ratio comparison

```python
@pytest.mark.parametrize("contrast,expected_beta_zz_approx", [
    (2,   1.35),
    (5,   1.65),
    (10,  1.93),
    (0.5, 0.72),
    (0.1, 0.23),
])
def test_eshelby_contrast_range(contrast, expected_beta_zz_approx):
    """
    Run the Eshelby test for several contrast ratios and verify the
    sign and approximate magnitude of the stress concentration.

    Use moderate configuration (N=48, R=8, domain=6R) for speed.
    Tolerance: 10% (domain=6R is less accurate than 10R).

    Key physics to verify at every contrast:
    - Stiff inclusion (E_i > E_m): beta_zz > 1 (stress amplification)
    - Soft inclusion  (E_i < E_m): beta_zz < 1 (stress shielding)
    - At contrast → 1: beta_zz → 1 (no effect)
    """
```

Run all Eshelby tests.

---

### Step 5 — Validation report generator (`tests/validation/report.py`)

Create a script that runs the full validation suite and prints a formatted
report. This is useful as a standalone check:

```bash
python tests/validation/report.py
```

The report must print:

```
============================================================
  VOXEL FEM SOLVER — ANALYTICAL VALIDATION REPORT
============================================================

LAYERED COMPOSITE TESTS
-----------------------
Test                          FEM        Reference   Error    Status
----                          ---        ---------   -----    ------
Normal modulus (E2/E1=10)     1.834      1.818       0.9%     PASS
Parallel modulus (E2/E1=10)   5.502      5.500       0.04%    PASS
Normal modulus (E2/E1=100)    1.980      1.980       0.01%    PASS
Normal modulus (E2/E1=1000)   2.000      2.000       0.01%    PASS
Per-phase stress (normal)     CV=0.8%    0%          --       PASS
Per-phase strain (parallel)   CV=0.3%    0%          --       PASS

ESHELBY INCLUSION TESTS (N=80, R=8, domain=10R)
-----------------------------------------------
Test                          FEM        Reference   Error    Status
----                          ---        ---------   -----    ------
beta_zz stiff (E_i/E_m=10)   1.871      1.930       3.1%     PASS
beta_zz soft  (E_i/E_m=0.1)  0.238      0.230       3.5%     PASS
beta_xx stiff                 0.590      0.610       3.3%     PASS
Interior uniformity (CV)      7.2%       0%          --       PASS
Far-field sigma_zz at r=4R   1.039      ~1.0        3.9%     PASS

DOMAIN CONVERGENCE (E_i/E_m=10, R=8 voxels)
---------------------------------------------
domain/R    beta_zz_FEM    Error    Trend
--------    -----------    -----    -----
4           1.640          15.0%    --
6           1.780           7.8%    ↓
8           1.840           4.7%    ↓
10          1.871           3.1%    ↓   PASS (monotone)

============================================================
OVERALL: 14/14 tests PASSED
============================================================
```

The report generator calls the same test functions as pytest but captures
the numerical values for display. Implement it as:

```python
# tests/validation/report.py
if __name__ == "__main__":
    run_validation_report()
```

---

### Step 6 — Integration with the main test suite

Add a `conftest.py` in `tests/validation/` that:

1. Defines the `slow` pytest mark.
2. Registers a `--fast` CLI option that skips the 80³ Eshelby tests.
3. Prints a one-line summary of validation results at the end of the test run.

```python
# tests/validation/conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true",
                     help="Skip slow validation tests (large grids)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="--fast flag set")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
```

Update the root `tests/` directory to include the validation subdirectory:

```bash
python -m pytest tests/ -v                      # all tests including validation
python -m pytest tests/ -v --fast               # skip slow Eshelby grids
python -m pytest tests/validation/ -v           # validation only
python -m pytest tests/ -v -m "not slow"        # same as --fast
```

Run the complete test suite.

---

### Step 7 — Documentation

Add a section to `README.md` titled **Analytical Validation**:

```markdown
## Analytical Validation

The solver is validated against two analytical benchmarks:

### Layered Composite
Exact effective moduli (harmonic/arithmetic mean). Validated at contrast
ratios E_2/E_1 = 2, 10, 100, 1000. All results within 2% of exact solution.

### Eshelby Spherical Inclusion
Closed-form uniform stress inside a spherical inclusion (Eshelby 1957).
Validated for E_i/E_m = 0.1 to 10. Results within 5% of analytical solution
for domain/radius = 10.

To run the validation suite:
    python -m pytest tests/validation/ -v
    python tests/validation/report.py     # formatted report
```

---

## Implementation Notes

### Stress recovery strategy

The solver's `_matvec` gives K·u but not stresses directly. To recover stress
at element centroids (needed for volume averaging), add a helper to
`voxel_fem_solver.py`:

```python
def compute_element_stress(self, u, chunk_size=500_000):
    """
    Compute average stress at each element's centroid.
    Uses single-point quadrature (xi=eta=zeta=0.5 in [0,1]).

    Returns sigma: (Ne, 6) float32 array [xx,yy,zz,xy,xz,yz]
    """
    from voxel_fem_solver import _B_matrix, build_node_index
    B_centroid, _ = _B_matrix(0.0, 0.0, 0.0)   # B at element centroid
    B_c = B_centroid.astype(np.float32)          # (6, 24)

    # Build D per element: D_e = E_e * D_ref
    lam = self.E_max * self.nu / ((1+self.nu)*(1-2*self.nu))
    mu  = self.E_max / (2*(1+self.nu))
    # ... (build D_ref once, scale by E_elem)

    Ne = self.E_elem.shape[0]
    sigma = np.zeros((Ne, 6), dtype=np.float32)

    for start in range(0, Ne, chunk_size):
        end  = min(start + chunk_size, Ne)
        # Implicit connectivity gather
        idx  = np.arange(start, end, dtype=np.int64)
        # ... (same implicit connectivity code as matvec_Ku)
        u_loc = u[dofs]                         # (chunk, 24)
        eps   = u_loc @ B_c.T                   # (chunk, 6)  = B·u_loc
        # sigma = D_e · eps  per element
        E_chunk = self.E_elem[start:end]        # (chunk,)
        sigma[start:end] = E_chunk[:,None] * (eps @ D_ref.T)

    return sigma
```

### Reaction force measurement

The most reliable way to compute effective modulus is from the **reaction
force** on the constrained face, not from volume-averaged stress (which
requires stress recovery). The reaction force is:

```python
f_int = solver._matvec(u)    # internal force vector (Ndof,)
# Sum z-DOFs of top face nodes
top_nodes  = node_idx[:, :, -1].ravel()
top_z_dofs = top_nodes * 3 + 2
F_z = f_int[top_z_dofs].sum()
A   = Nx * Ny              # cross-section area (in voxel units)
E_eff = (F_z / A) / (delta / Nz)
```

This is exact (no approximation from stress recovery) and should be the
primary measurement for the layered composite tests.

### Tolerance rationale

| Test | Tolerance | Source of error |
|---|---|---|
| Layered normal, low contrast | 1% | Poisson coupling at free lateral faces |
| Layered normal, high contrast | 2% | Same; amplified by stiffness jump |
| Layered parallel | 0.5% | Exact on voxel grid; only BC artefact |
| Eshelby interior, 10R domain | 5% | Finite domain + voxel staircase |
| Eshelby interior, 6R domain | 8% | Larger finite-domain error |
| Eshelby far-field at r=4R | 8% | Finite domain boundary reflection |
| Interior uniformity (CV) | 10% | Staircase approximation of sphere |

---

## Constraints

- All reference solutions computed in pure Python/NumPy in `reference_solutions.py`.
  No external data files, no internet access required.
- Tests must run with the same `VoxelFEMSolver` API as the existing smoke tests.
  Do not add any solver parameters that break existing tests.
- The 80³ Eshelby tests are expected to take 5–30 minutes depending on
  hardware. Mark them `@pytest.mark.slow` so the CI fast path can skip them.
- All tests must pass with both Jacobi and GMG preconditioners if both are
  available. Add a `preconditioner` fixture that parametrises over available
  options.

---

## Quick Reference: Analytical Formulas

**Layered — series (normal loading)**:
`E_eff = 1 / (f1/E1 + f2/E2)`

**Layered — parallel (in-plane loading)**:
`E_eff = f1·E1 + f2·E2`

**Eshelby tensor, sphere** (ν uniform):
`S_1111 = S_2222 = S_3333 = (7-5ν)/(15(1-ν))`
`S_1122 = S_2233 = S_1133 = (5ν-1)/(15(1-ν))`
`S_1212 = S_2323 = S_1313 = (4-5ν)/(15(1-ν))`

**Interior strain**:
`ε^in = [I + S·C_m⁻¹·(C_i-C_m)]⁻¹ · ε^∞`

**Interior stress**:
`σ^in = C_i · ε^in`

**Stress concentration**:
`β_zz = σ^in_zz / σ^∞_zz`

**Far-field decay** (along z-axis, r >> R):
`|σ_zz(r) - σ^∞| ~ O((R/r)³)`
