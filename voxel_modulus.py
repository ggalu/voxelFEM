"""
voxel_modulus.py

Estimate a per-voxel Young's modulus field for a short-fibre reinforced polymer
from the grey-level histogram of a CT scan.

Method (derived in-session for a glass/thermoplastic SFRP,
E_fibre = 72 GPa, E_matrix = 1.8 GPa, fibre volume fraction = 30 %,
measured composite modulus = 5.5 GPa):

  1. Matrix endpoint  g_m : the histogram mode (the matrix peak).

  2. Fibre endpoint   g_f : CALIBRATED from the KNOWN fibre volume fraction.
     CT attenuation is ~linear in composition, so a partial-volume voxel reads
         g = phi*g_f + (1-phi)*g_m   ->   phi(g) = (g - g_m)/(g_f - g_m).
     g_f is chosen so the clamped, count-weighted mean of phi over the
     histogram equals the known fibre volume fraction. (For this data the
     fibres are only ~1-2 voxels across, so there is no pure-fibre peak to read
     g_f off directly -- it must be calibrated.)

  3. Sub-voxel modulus  E(phi) : a mixing rule applied per voxel. The pure
     linear (Voigt) map overestimates stiffness ~4x for this material (its
     volume average is ~23 GPa vs a measured composite of 5.5 GPa), so the
     default is the geometric / power-law map
         E = E_m * (E_f / E_m) ** (p * phi).

  4. The exponent p is the ONE remaining free parameter. It is calibrated
     EXTERNALLY against the measured composite modulus (5.5 GPa) by running the
     FE homogenisation and bisecting p until the effective modulus matches.
     The histogram alone cannot do this step -- see calibrate_exponent().

Volume fraction and composite modulus are independent observables that pin
independent quantities: the 30 % fixes g_f (geometry); the 5.5 GPa fixes p
(constitutive law). Do not use one to fit the other.

The resulting per-bin E(g) is a lookup table; apply it to the voxel field with
ModulusMap.modulus_from_grey(grey_values).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModulusMap:
    """Result of mapping histogram bins -> fibre fraction -> Young's modulus."""

    bin_centers: np.ndarray   # grey level at each bin
    counts: np.ndarray        # voxel count per bin
    phi: np.ndarray           # estimated fibre volume fraction per bin
    E: np.ndarray             # estimated Young's modulus per bin (same units as inputs)
    g_matrix: float           # matrix grey endpoint
    g_fibre: float            # fibre grey endpoint
    mixing: str               # mixing rule used
    exponent: float           # geometric-map exponent p

    def modulus_from_grey(self, grey):
        """Apply the per-bin E(g) map to arbitrary voxel grey values via
        piecewise-linear interpolation, clamped at the endpoints. This is how
        you assign moduli to the actual voxel field."""
        grey = np.asarray(grey, dtype=float)
        return np.interp(grey, self.bin_centers, self.E,
                         left=self.E[0], right=self.E[-1])

    def realised_fibre_fraction(self):
        """Count-weighted mean fibre fraction actually produced (sanity check
        against the target volume fraction)."""
        return float(np.average(self.phi, weights=self.counts))


def load_histogram(path):
    """Parse the '# comment' header + 'bin_center<whitespace>count' format."""
    centers, counts = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("bin_center"):
                continue
            c, n = line.split()
            centers.append(float(c))
            counts.append(float(n))
    return np.asarray(centers), np.asarray(counts)


def _calibrate_g_fibre(centers, counts, g_matrix, fibre_volume_fraction):
    """Solve for g_f so the clamped, count-weighted mean fibre fraction equals
    the target. mean_phi(g_f) decreases monotonically in g_f, so we bisect.
    Calibrating on the *clamped* field (not the closed-form mean inversion)
    keeps the field you actually use consistent with the known fraction."""
    total = counts.sum()
    target = float(fibre_volume_fraction)

    def mean_phi(g_f):
        phi = np.clip((centers - g_matrix) / (g_f - g_matrix), 0.0, 1.0)
        return float((phi * counts).sum() / total)

    lo = g_matrix + 1e-6
    hi = g_matrix + 1.0
    while mean_phi(hi) > target:           # expand until fraction drops below target
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Could not bracket g_fibre; check inputs.")
    for _ in range(200):                   # bisection
        mid = 0.5 * (lo + hi)
        if mean_phi(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_modulus_per_bin(
    bin_centers,
    counts,
    E_matrix=1.8,
    E_fibre=72.0,
    fibre_volume_fraction=0.30,
    g_matrix=None,
    g_fibre=None,
    mixing="geometric",
    exponent=1.0,
    clamp_modulus=True,
):
    """Estimate Young's modulus for every histogram bin.

    Parameters
    ----------
    bin_centers, counts : 1-D array-like
        Grey level and voxel count per histogram bin.
    E_matrix, E_fibre : float
        Constituent moduli (same units; GPa here).
    fibre_volume_fraction : float
        Known global fibre volume fraction; used to calibrate g_fibre.
    g_matrix : float, optional
        Matrix grey endpoint. Default: histogram mode (matrix peak).
    g_fibre : float, optional
        Fibre grey endpoint. Default: calibrated from fibre_volume_fraction.
    mixing : {'geometric', 'voigt', 'reuss'}
        Sub-voxel rule mapping fibre fraction -> modulus.
          'geometric' : E = E_m * (E_f/E_m) ** (exponent * phi)   (recommended)
          'voigt'      : E = E_m + phi*(E_f - E_m)                 (stiff bound; overestimates)
          'reuss'      : 1/E = phi/E_f + (1 - phi)/E_m             (compliant bound)
    exponent : float
        Calibration knob 'p' for the geometric map. Determined externally by
        matching the composite modulus (see calibrate_exponent()); 1.0 is the
        uncalibrated geometric map. Ignored for 'voigt' / 'reuss'.
    clamp_modulus : bool
        Clamp E into [E_matrix, E_fibre] (guards against exponent > 1 overshoot).

    Returns
    -------
    ModulusMap
    """
    centers = np.asarray(bin_centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if centers.shape != counts.shape:
        raise ValueError("bin_centers and counts must have the same shape.")
    if not np.all(np.diff(centers) > 0):
        raise ValueError("bin_centers must be strictly increasing.")

    if g_matrix is None:
        g_matrix = float(centers[np.argmax(counts)])          # histogram mode
    if g_fibre is None:
        g_fibre = _calibrate_g_fibre(centers, counts, g_matrix,
                                     fibre_volume_fraction)
    if g_fibre <= g_matrix:
        raise ValueError("g_fibre must exceed g_matrix.")

    phi = np.clip((centers - g_matrix) / (g_fibre - g_matrix), 0.0, 1.0)

    if mixing == "geometric":
        E = E_matrix * (E_fibre / E_matrix) ** (exponent * phi)
    elif mixing == "voigt":
        E = E_matrix + phi * (E_fibre - E_matrix)
    elif mixing == "reuss":
        E = 1.0 / (phi / E_fibre + (1.0 - phi) / E_matrix)
    else:
        raise ValueError(f"Unknown mixing rule: {mixing!r}")

    if clamp_modulus:
        E = np.clip(E, E_matrix, E_fibre)

    return ModulusMap(centers, counts, phi, E,
                      float(g_matrix), float(g_fibre), mixing, float(exponent))


def calibrate_exponent(
    bin_centers,
    counts,
    run_homogenisation,
    target_modulus,
    E_matrix=1.8,
    E_fibre=72.0,
    fibre_volume_fraction=0.30,
    p_bracket=(0.1, 3.0),
    tol=1e-3,
    max_iter=40,
    **kwargs,
):
    """Calibrate the geometric exponent p so the FE-homogenised composite
    modulus matches `target_modulus` (e.g. 5.5 GPa).

    `run_homogenisation` is a user-supplied callable: E_composite = f(ModulusMap).
    It should assign per-voxel moduli (ModulusMap.modulus_from_grey), run the
    solve, and return the effective modulus along the direction the target was
    measured in. Assumes E_composite increases monotonically with p, so bisects.

    Returns (p_calibrated, ModulusMap_at_p).
    """
    lo, hi = p_bracket

    def comp(p):
        mmap = estimate_modulus_per_bin(
            bin_centers, counts, E_matrix=E_matrix, E_fibre=E_fibre,
            fibre_volume_fraction=fibre_volume_fraction,
            mixing="geometric", exponent=p, **kwargs)
        return float(run_homogenisation(mmap)), mmap

    c_lo, _ = comp(lo)
    c_hi, _ = comp(hi)
    if not (min(c_lo, c_hi) <= target_modulus <= max(c_lo, c_hi)):
        raise ValueError(
            f"target {target_modulus} not bracketed by p in {p_bracket} "
            f"(E_composite spans {c_lo:.3g}..{c_hi:.3g}); widen p_bracket.")

    mmap = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c_mid, mmap = comp(mid)
        if abs(c_mid - target_modulus) < tol:
            return mid, mmap
        if (c_mid < target_modulus) == (c_lo < target_modulus):
            lo, c_lo = mid, c_mid
        else:
            hi = mid
    return mid, mmap


if __name__ == "__main__":
    centers, counts = load_histogram("ct_histogram.txt")

    mmap = estimate_modulus_per_bin(
        centers, counts,
        E_matrix=1.8, E_fibre=72.0,
        fibre_volume_fraction=0.30,
        mixing="geometric", exponent=1.0,   # exponent calibrated separately to 5.5 GPa
    )

    print(f"g_matrix (mode)       = {mmap.g_matrix:8.1f}")
    print(f"g_fibre  (calibrated) = {mmap.g_fibre:8.1f}")
    print(f"realised mean phi     = {mmap.realised_fibre_fraction() * 100:5.2f} %  (target 30 %)")
    print("\n   grey      phi    E (GPa)")
    for g in (20000.0, 26857.0, 32000.0, 40000.0, 47222.0, 60000.0):
        phi = float(np.clip((g - mmap.g_matrix) / (mmap.g_fibre - mmap.g_matrix), 0, 1))
        print(f"  {g:7.0f}  {phi:6.3f}  {float(mmap.modulus_from_grey(g)):7.2f}")

    print("\nNOTE: exponent=1.0 is the UNCALIBRATED geometric map.")
    print("Calibrate it to the 5.5 GPa composite modulus via calibrate_exponent(),")
    print("passing your FE homogenisation as the run_homogenisation callback.")
