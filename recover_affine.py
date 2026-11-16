"""
Recover the affine transformation that maps ref.npy onto def.npy.

Model (must match how def.npy was produced by warp_intensity.py)
----------------------------------------------------------------
warp_intensity.py builds the deformed image by a BACKWARD (pull) warp:

    def(x) = ref( x - u(x) ),

where u is the displacement field. Here we model u as *affine*,

    u(x) = M (x - c) + t,

with M the 3x3 displacement-gradient ("stretch") tensor, c the volume centre,
and t a translation. A trial transformation is therefore always applied by
DEFORMING ref (sampling from ref) and comparing the result to def — never the
other way round — so the interpolation error is identical in kind to the one
baked into def.npy.

Because u(x) = M (x-c) + t is linear, the volume-average normal strain along
axis i is exactly M[i, i]. So the diagonal of the recovered M equals the mean
normal strains reported by warp_intensity.py (eps_xx, eps_yy, eps_zz), and the
symmetric part sym(M) is the full small-strain tensor.

Method
------
Forward-additive Gauss-Newton (Lucas-Kanade) over the 12 affine parameters.
For the current guess, W(x) = ref(x - u(x)); with e(x) = def(x) - W(x),

    dW/dM_ij = -grad_i(ref)|_s * (x - c)_j
    dW/dt_i  = -grad_i(ref)|_s

(s = x - u(x) the sample location). We solve the 12x12 normal equations
H dp = b, H = sum J J^T, b = sum J e, in float64, and update the parameters.
Voxels whose sample falls outside the volume, or that are fill in def, are
masked out.

Coarse-to-fine (avoids local minima)
------------------------------------
Gauss-Newton on the raw images has a narrow capture range and can settle into a
local minimum. We therefore run a Gaussian SCALE-SPACE pyramid: the images are
heavily smoothed first (wide, convex-ish basin), then the smoothing is reduced
level by level down to sigma = 0 (full resolution) for the final precision. The
recovered (M, t) is carried from each level to the next as the initial guess.
Smoothing does not change the coordinate grid, so no rescaling of M or t is
needed between levels. `def` is smoothed by *normalized convolution*
(smooth(def*valid)/smooth(valid)) so the background fill never bleeds across the
void boundary; the validity mask always comes from the original, unsmoothed def.
"""

import argparse
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates, gaussian_filter


def _warp_ref(ref, grid, Xc, M, t, cval):
    """Pull-warp ref by the affine displacement u(x) = M (x-c) + t.

    Returns (warped, coords, inbounds_mask). `grid` is cp.indices(ref.shape);
    `Xc` = grid - c (centred coordinates). Everything float32 on the GPU.
    """
    # u_i = sum_j M_ij Xc_j + t_i
    u = cp.tensordot(M, Xc, axes=([1], [0])) + t[:, None, None, None]
    coords = grid - u                                     # sample from ref here
    warped = map_coordinates(ref, coords, order=1, mode="constant",
                             cval=cval, prefilter=False)
    N = cp.asarray(ref.shape, dtype=cp.float32)[:, None, None, None]
    inb = cp.all((coords >= 0) & (coords <= (N - 1)), axis=0)
    return warped, coords, inb


def _smooth_ref(img, sigma):
    """Gaussian-smooth a fully-valid image (no masking needed)."""
    if sigma <= 0:
        return img
    return gaussian_filter(img, sigma, mode="nearest")


def _smooth_masked(img, valid_f, sigma):
    """Normalized-convolution Gaussian smooth: keeps background fill from bleeding
    across the mask boundary. `valid_f` is the float32 validity mask."""
    if sigma <= 0:
        return img
    num = gaussian_filter(img * valid_f, sigma, mode="constant", cval=0.0)
    den = gaussian_filter(valid_f, sigma, mode="constant", cval=0.0)
    out = img.copy()
    good = den > 1e-6
    out[good] = num[good] / den[good]
    return out


def _gauss_newton(ref, dfm, valid_def, grid, Xc, M, t, cval, max_iter, tol,
                  verbose):
    """Forward-additive Gauss-Newton on the given (possibly smoothed) images.

    `M` (3,3) and `t` (3,) are float64 cupy arrays used as the initial guess;
    updated copies are returned. `valid_def` is the boolean mask from the
    ORIGINAL (unsmoothed) def.
    """
    gref = cp.gradient(ref)                               # grad of (smoothed) ref
    rms = np.inf
    for it in range(max_iter):
        warped, coords, inb = _warp_ref(ref, grid, Xc,
                                        M.astype(cp.float32),
                                        t.astype(cp.float32), cval)
        mask = (inb & valid_def).astype(cp.float32)
        e = (dfm - warped) * mask                         # residual (masked)

        n = float(mask.sum())
        rms = float(cp.sqrt((e * e).sum() / max(n, 1.0)))

        # grad_i(ref) evaluated at the sample locations s = coords
        gs = [map_coordinates(gref[i], coords, order=1, mode="nearest",
                              prefilter=False) for i in range(3)]

        # Assemble the 12 per-voxel Jacobian channels.
        #   params: M00 M01 M02 M10 M11 M12 M20 M21 M22 t0 t1 t2
        J = []
        for i in range(3):
            for j in range(3):
                J.append(-gs[i] * Xc[j])                  # dW/dM_ij
        for i in range(3):
            J.append(-gs[i])                              # dW/dt_i
        for k in range(12):
            J[k] = J[k] * mask

        # Normal equations in float64.
        H = cp.empty((12, 12), dtype=cp.float64)
        b = cp.empty(12, dtype=cp.float64)
        for k in range(12):
            b[k] = (J[k] * e).sum(dtype=cp.float64)
            for l in range(k, 12):
                H[k, l] = (J[k] * J[l]).sum(dtype=cp.float64)
                H[l, k] = H[k, l]

        # Levenberg damping keeps H well-conditioned if a channel is degenerate.
        H += cp.eye(12, dtype=cp.float64) * (1e-9 * float(cp.trace(H)) / 12.0 + 1e-30)
        dp = cp.linalg.solve(H, b)

        M = M + dp[:9].reshape(3, 3)
        t = t + dp[9:]

        step = float(cp.linalg.norm(dp))
        if verbose:
            print(f"    iter {it:2d}:  RMS={rms:.4e}   ||dp||={step:.3e}")
        if step < tol:
            break

    return M, t, rms


def recover_affine(ref, dfm, cval=0.0, sigmas=(4.0, 2.0, 1.0, 0.0),
                   max_iter=50, tol=1e-6, verbose=True):
    """Coarse-to-fine Gauss-Newton affine registration: ref(x - u(x)) ~= def(x).

    Runs a Gaussian scale-space pyramid (large sigma -> 0) to widen the capture
    basin and avoid local minima, carrying (M, t) between levels.

    Parameters
    ----------
    ref, dfm : cupy.ndarray, shape (Nx, Ny, Nz), float32
        Reference and deformed volumes on the GPU.
    cval : float
        Fill value used for out-of-volume samples (def's background); masked out.
    sigmas : sequence of float
        Gaussian smoothing levels, applied in order (coarse first). End with 0.0
        for a final full-resolution refinement.
    max_iter, tol : int, float
        Per-level iteration cap and convergence tolerance on ||dp||.

    Returns
    -------
    M : (3,3) ndarray   displacement-gradient tensor (diag = normal strains)
    t : (3,)  ndarray   translation (about the volume centre)
    rms : float         final masked RMS intensity residual (finest level)
    """
    shape = ref.shape
    grid = cp.indices(shape, dtype=cp.float32)            # (3, Nx, Ny, Nz)
    centre = (cp.asarray(shape, dtype=cp.float32) - 1.0) / 2.0
    Xc = grid - centre[:, None, None, None]               # centred coords

    valid_def = dfm != cval                               # from ORIGINAL def
    valid_f = valid_def.astype(cp.float32)

    M = cp.zeros((3, 3), dtype=cp.float64)
    t = cp.zeros(3, dtype=cp.float64)

    rms = np.inf
    for sigma in sigmas:
        if verbose:
            print(f"--- smoothing level sigma = {sigma:g} ---")
        ref_s = _smooth_ref(ref, sigma)
        dfm_s = _smooth_masked(dfm, valid_f, sigma)
        M, t, rms = _gauss_newton(ref_s, dfm_s, valid_def, grid, Xc,
                                  M, t, cval, max_iter, tol, verbose)

    return cp.asnumpy(M), cp.asnumpy(t), rms


def _report(M, t, rms):
    strains = np.diag(M)
    sym = 0.5 * (M + M.T)
    print("\nRecovered affine displacement u(x) = M (x - c) + t")
    print("  displacement gradient M (dU):")
    for row in M:
        print("    [{:+.6e} {:+.6e} {:+.6e}]".format(*row))
    print("  translation t (about volume centre):")
    print("    [{:+.6e} {:+.6e} {:+.6e}]".format(*t))
    print("\n  Normal strains (diag M):")
    print(f"    eps_xx = {strains[0]:+.6e}")
    print(f"    eps_yy = {strains[1]:+.6e}")
    print(f"    eps_zz = {strains[2]:+.6e}")
    print("  Symmetric strain tensor sym(M):")
    for row in sym:
        print("    [{:+.6e} {:+.6e} {:+.6e}]".format(*row))
    print(f"\n  Final masked RMS intensity residual: {rms:.4e}")


def _self_test():
    """Fabricate a known affine, deform ref with it, then recover and compare."""
    print("=== self-test: recover a known affine from ref.npy ===")
    ref_np = np.load("ref.npy").astype(np.float32)
    ref = cp.asarray(ref_np)
    shape = ref.shape
    grid = cp.indices(shape, dtype=cp.float32)
    centre = (cp.asarray(shape, dtype=cp.float32) - 1.0) / 2.0
    Xc = grid - centre[:, None, None, None]

    M_true = cp.asarray([[0.012, 0.004, 0.000],
                         [0.000, -0.006, 0.003],
                         [0.002, 0.000, 0.020]], dtype=cp.float32)
    # A large translation: single-scale GN would lose this, the pyramid finds it.
    t_true = cp.asarray([5.0, -4.0, 6.0], dtype=cp.float32)

    # Deform ref with the known affine (same pull warp used everywhere).
    dfm, _, _ = _warp_ref(ref, grid, Xc, M_true, t_true, cval=0.0)

    M, t, rms = recover_affine(ref, dfm, cval=0.0, verbose=True)
    dM = np.abs(M - cp.asnumpy(M_true)).max()
    dt = np.abs(t - cp.asnumpy(t_true)).max()
    _report(M, t, rms)
    print(f"\n  max|M - M_true| = {dM:.3e}   max|t - t_true| = {dt:.3e}")
    ok = dM < 5e-4 and dt < 5e-2
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(
        description="Recover the affine transformation mapping ref.npy -> def.npy "
                    "by Gauss-Newton registration (deforms ref, compares to def).")
    p.add_argument("--ref", default="ref.npy", help="Reference volume (.npy).")
    p.add_argument("--def", dest="dfm", default="def.npy",
                   help="Deformed volume (.npy).")
    p.add_argument("--cval", type=float, default=0.0,
                   help="Fill value of def's background (masked out). Default 0.")
    p.add_argument("--iters", type=int, default=50,
                   help="Max GN iterations per smoothing level.")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Convergence tolerance on ||dp||.")
    p.add_argument("--sigmas", default="4,2,1,0",
                   help="Comma-separated Gaussian smoothing levels, coarse "
                        "first; end with 0 for full resolution. Default 4,2,1,0.")
    p.add_argument("--output", "-o", default=None,
                   help="Optional .npz to save M and t.")
    p.add_argument("--test", action="store_true",
                   help="Run the known-affine self-test and exit.")
    args = p.parse_args()

    if args.test:
        raise SystemExit(0 if _self_test() else 1)

    sigmas = tuple(float(s) for s in args.sigmas.split(","))

    ref = cp.asarray(np.load(args.ref).astype(np.float32))
    dfm = cp.asarray(np.load(args.dfm).astype(np.float32))
    if ref.shape != dfm.shape:
        raise ValueError(f"ref {ref.shape} and def {dfm.shape} must match.")
    print(f"Loaded ref {args.ref} and def {args.dfm}: {tuple(ref.shape)}")

    M, t, rms = recover_affine(ref, dfm, cval=args.cval, sigmas=sigmas,
                               max_iter=args.iters, tol=args.tol)
    _report(M, t, rms)

    if args.output:
        np.savez(args.output, M=M, t=t, rms=rms)
        print(f"\nSaved affine to {args.output}")


if __name__ == "__main__":
    main()
