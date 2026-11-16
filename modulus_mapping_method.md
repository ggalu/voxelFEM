# Mapping CT Grey Values to a Per-Voxel Young's Modulus Field

## Overview

A Young's modulus is assigned to every voxel of a micro-computed-tomography
(µCT) volume of a short-fibre reinforced polymer from four inputs: the
grey-level (intensity) histogram of the reconstructed volume, the constituent
moduli of fibre and matrix, the global fibre volume fraction, and the measured
macroscopic Young's modulus of the composite. The procedure is constructed so
that, once the quantities that are fixed directly by the data have been
determined, a single scalar — the exponent $p$ of the modulus interpolation —
remains adjustable. For the present material this exponent takes its natural
value $p = 1$ (the unweighted geometric mean), at which finite-element
homogenisation of the assigned field already reproduces the measured macroscopic
modulus. The model is therefore effectively parameter-free, and the macroscopic
modulus serves as an independent validation rather than as a fitted target. The
four steps are summarised below; symbol definitions are collected in the final
table.

## 1. Phase endpoints in grey-value space

X-ray attenuation within a voxel is, to first order, linear in local
composition. A voxel containing a local fibre volume fraction
$\varphi \in [0,1]$ therefore has an expected grey value

$$
g \;=\; \varphi\, g_\mathrm{f} \;+\; (1-\varphi)\, g_\mathrm{m},
$$

where $g_\mathrm{m}$ and $g_\mathrm{f}$ are the grey values of pure matrix and
pure fibre, respectively. The matrix endpoint $g_\mathrm{m}$ is taken as the
mode of the histogram, i.e. the dominant matrix peak.

Because the fibres are only marginally resolved (of the order of one to two
voxels in diameter), the histogram exhibits no distinct pure-fibre peak and
$g_\mathrm{f}$ cannot be read directly from the distribution. It is instead
calibrated against the known fibre volume fraction $V_\mathrm{f}$. The local
fraction implied by the linear attenuation model,

$$
\varphi(g) \;=\; \operatorname{clip}\!\left(
\frac{g - g_\mathrm{m}}{\,g_\mathrm{f} - g_\mathrm{m}\,},\; 0,\; 1 \right),
$$

must, when averaged over all included voxels, reproduce $V_\mathrm{f}$. With
$n_k$ the voxel count of histogram bin $k$ centred at grey value $g_k$, the
fibre endpoint is the unique root of

$$
\frac{\sum_k n_k\, \varphi(g_k)}{\sum_k n_k} \;=\; V_\mathrm{f}.
$$

The left-hand side decreases monotonically with $g_\mathrm{f}$, so the root is
unique and is obtained by bisection. The fibre endpoint is thereby fixed by the
volume fraction alone.

## 2. Local fibre-fraction field

The clipped linear relation $\varphi(g)$ is evaluated for every voxel grey
value, yielding a continuous fibre-fraction field $\varphi \in [0,1]$. Voxels at
or below $g_\mathrm{m}$ are pure matrix ($\varphi = 0$); the small population at
or above $g_\mathrm{f}$ is fibre-dominated ($\varphi = 1$); the majority are
graded partial-volume voxels lying between the two endpoints.

## 3. Sub-voxel modulus interpolation

The local modulus is obtained from the fibre fraction through a geometric
(log-linear) interpolation between the constituent moduli,

$$
E(\varphi) \;=\; E_\mathrm{m}
\left( \frac{E_\mathrm{f}}{E_\mathrm{m}} \right)^{\,p\,\varphi}.
$$

For $p = 1$ this reduces to the geometric-mean rule, which lies between the
Voigt (iso-strain) bound $E = E_\mathrm{m} + \varphi\,(E_\mathrm{f}-E_\mathrm{m})$
and the Reuss (iso-stress) bound
$E^{-1} = \varphi\,E_\mathrm{f}^{-1} + (1-\varphi)\,E_\mathrm{m}^{-1}$. A linear
(Voigt) assignment is not used: for the present stiffness contrast
$E_\mathrm{f}/E_\mathrm{m} \approx 40$ and the partial-volume-dominated fibre
signal, it overestimates the effective stiffness substantially, whereas the
measured composite modulus lies close to the lower Hill bound, indicating a
compliance-controlled response. The exponent $p$ controls the weight assigned to
the stiff phase within partial-volume voxels and is the sole adjustable quantity
of the model; $p = 1$ recovers the unbiased geometric mean and is retained as the
natural default (see Section 4).

## 4. Validation against the macroscopic modulus

The exponent is the only quantity not fixed directly by the histogram or the
constituent data. Its admissible value is examined by requiring that
finite-element homogenisation of the assigned modulus field reproduce the
experimentally measured macroscopic modulus $E_\mathrm{c}$,

$$
E_\mathrm{c}^{\,\mathrm{FE}}(p) \;=\; E_\mathrm{c}.
$$

The homogenised modulus increases monotonically with $p$, so any required value
would be uniquely determined by bisection, with the homogenisation boundary
conditions chosen to match the loading direction of the experimental
measurement. For the present material the condition is satisfied at $p = 1$,
i.e. by the unweighted geometric-mean rule, with no adjustment away from its
natural default. Consequently the macroscopic modulus is reproduced without
expending the single free parameter: it constitutes an independent validation of
the assignment rather than a calibration target, and the modulus field is a fixed
closed-form function of grey value,
$E(g) = E_\mathrm{m}\,(E_\mathrm{f}/E_\mathrm{m})^{\varphi(g)}$.

That the unbiased rule succeeds also indicates that the partial-volume voxels
require no artificial softening or stiffening to compensate for fibre geometry
unresolved by the voxel grid; the geometric mean alone is sufficient, which
limits the resolution dependence of the result. The validation reported here
rests on a single specimen and loading direction; transferability of $p = 1$
should be confirmed against an independent specimen, loading direction, or
volume fraction before it is treated as a general material rule.

## Parameter accounting

| Quantity | Determined by | Role in the model |
|---|---|---|
| $g_\mathrm{m}$ | histogram mode | matrix grey endpoint |
| $g_\mathrm{f}$ | fibre volume fraction $V_\mathrm{f}$ | fibre grey endpoint (geometry) |
| $\varphi(g)$ | $g_\mathrm{m},\, g_\mathrm{f}$ | local fibre volume fraction |
| $E_\mathrm{m},\, E_\mathrm{f}$ | constituent material data | modulus endpoints |
| $p$ | validated against macroscopic modulus $E_\mathrm{c}$ | constitutive interpolation — sole free parameter; takes natural value $p = 1$ |

The two experimental observables, the fibre volume fraction $V_\mathrm{f}$ and
the macroscopic modulus $E_\mathrm{c}$, are independent and constrain independent
aspects of the model: $V_\mathrm{f}$ fixes the geometry endpoint $g_\mathrm{f}$,
while $E_\mathrm{c}$ tests the constitutive assignment. The volume fraction is
therefore never used to fit stiffness, nor the stiffness to fit the geometry,
which avoids parameter degeneracy. Because the single free parameter is found to
take its natural value $p = 1$, the macroscopic modulus is reproduced without
adjustment and the assignment is, in effect, parameter-free.

*In the present study,* $E_\mathrm{f} = 72\ \mathrm{GPa}$,
$E_\mathrm{m} = 1.8\ \mathrm{GPa}$, $V_\mathrm{f} = 0.30$ and
$E_\mathrm{c} = 5.5\ \mathrm{GPa}$, giving a matrix endpoint
$g_\mathrm{m} = 26857$ and a calibrated fibre endpoint $g_\mathrm{f} = 47222$
(16-bit grey levels). Finite-element homogenisation of the resulting field
reproduces $E_\mathrm{c}$ at $p = 1$.
