# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# warp_lagrangian (Lagrangian fixed-point warp, cubic B-spline sampling) added by
# Lokesh Metta, Albert-Ludwigs Universitaet Freiburg, Germany, 2026-09-25.
"""
GPU warping of a 3D image by a displacement field.

Two operations
--------------
1. ``warp_lagrangian`` (the default of the command line, since 2026-09-25):
   make the *deformed* image for a Lagrangian displacement field u(X), which
   moves each reference point X to y = X + u(X):

       deformed(y) = image(X),   X solved from  X + u(X) = y

   at every deformed-image voxel y, by the fixed-point iteration
   X <- y - u(X) (converges when |grad u| < 1). The image is sampled with a
   cubic B-spline by default. This is the correct forward model for a DVC
   ground truth: the DVC then recovers exactly u(X). It is the same method
   as voxelDVC 0.2.0's ``prepare-ground-truth`` (``make_deformed_image``),
   run on the GPU.

2. ``warp_image`` (unchanged): the plain pull warp

       warped[x] = image(x + u[x])

   with trilinear interpolation. With ``negate_u`` this gives
   image(x - u(x)), i.e. the displacement evaluated at the *deformed*
   position instead of the reference position. For a Lagrangian ground truth
   that is only a first-order approximation: the image it makes encodes
   about u + (grad u) . u instead of u. For a uniform strain e at distance
   x from the fixed end the difference is about e^2 x, e.g. 0.25 voxel at
   5 % strain and x = 100 voxels. Keep it for other uses; do not use it to
   make ground-truth images. (Before 2026-09-25 the command line used this.)

Displacement convention (``warp_lagrangian``)
---------------------------------------------
``u`` is voxelFEM's nodal field, shape (Nx+1, Ny+1, Nz+1, 3) (or with the
component axis first), with node index k at image coordinate k (nodes at
0..N, image voxels at 0..N-1), the same convention as voxelDVC 0.2.0's
ground-truth generator. The field is sampled trilinearly between nodes. This
differs by half a voxel from the --legacy-pull path, which averages the nodes
to voxel centres (voxel i centred at node coordinate i + 0.5). A field already
on the image grid, shape (Nx, Ny, Nz, 3), is also accepted.
"""

import cupy as cp
from cupyx.scipy.ndimage import map_coordinates


def warp_image(image, u, cval=0.0, negate_u=False, output_dtype=None):
    """
    Trilinear backward (pull) warp of a 3D volume by a displacement field, on the GPU:
    warped[x] = image(x + u[x]) (or image(x - u[x]) with negate_u).

    Not a correct ground-truth generator for a Lagrangian field; use
    ``warp_lagrangian`` for that (see the module docstring).

    Parameters
    ----------
    image : cupy.ndarray, shape (Nx, Ny, Nz)
        Input gray-value volume on the GPU. Any real dtype (e.g. uint16, float32).
    u : cupy.ndarray, shape (3, Nx, Ny, Nz)
        Displacement field. u[i] is the displacement along axis i, in voxels.
    cval : float, optional
        Fill value for sample points outside the input volume. Default 0.0
        (set this to your background/air value if that is not zero).
    negate_u : bool, optional
        If True, use -u (flip the sign convention). Default False.
    output_dtype : cupy/numpy dtype or None, optional
        dtype of the result. If None, matches `image.dtype`. Use cp.float32
        if you need fractional values preserved (an integer image dtype rounds).

    Returns
    -------
    warped : cupy.ndarray, shape (Nx, Ny, Nz)
    """
    if image.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {image.shape}")
    if u.shape != (3,) + image.shape:
        raise ValueError(
            f"u must have shape (3, {image.shape[0]}, {image.shape[1]}, "
            f"{image.shape[2]}), got {u.shape}"
        )

    # Build sample coordinates: coords[i] = index_i + u[i], in float32.
    # cp.indices gives the identity grid; adding u in place keeps one fewer
    # large (3x-volume) array alive than coords = grid + u would.
    coords = cp.indices(image.shape, dtype=cp.float32)
    if negate_u:
        coords -= u
    else:
        coords += u

    # order=1  -> trilinear (no spline prefilter needed; prefilter=False is a no-op safeguard)
    # mode='constant', cval -> constant fill outside the volume
    warped = map_coordinates(
        image, coords, order=1, mode="constant", cval=cval, prefilter=False
    )

    if output_dtype is not None and warped.dtype != cp.dtype(output_dtype):
        warped = warped.astype(output_dtype)
    return warped


def _components(u, image_shape):
    """Return the three displacement components as float64 cupy arrays."""
    u = cp.asarray(u)
    nodal = tuple(n + 1 for n in image_shape)
    if u.ndim == 4 and u.shape[-1] == 3 and u.shape[:3] in (nodal, tuple(image_shape)):
        return [cp.ascontiguousarray(u[..., c], dtype=cp.float64) for c in range(3)]
    if u.ndim == 4 and u.shape[0] == 3 and u.shape[1:] in (nodal, tuple(image_shape)):
        return [cp.ascontiguousarray(u[c], dtype=cp.float64) for c in range(3)]
    raise ValueError(f"displacement shape {u.shape} does not fit image {image_shape}: expected "
                     f"{nodal + (3,)} (nodal) or {tuple(image_shape) + (3,)}, component axis first or last")


def warp_lagrangian(image, u, order=3, n_iter=50, tol=1e-6, slab=32):
    """
    Deformed image for a Lagrangian displacement field, on the GPU:
    deformed(y) = image(X) with X + u(X) = y at every voxel y.

    Parameters
    ----------
    image : array, shape (Nx, Ny, Nz)
        Reference volume (numpy or cupy, any real dtype; computed in float64).
    u : array
        Displacement in voxels, shape (Nx+1, Ny+1, Nz+1, 3) nodal (node index =
        image coordinate) or (Nx, Ny, Nz, 3); the component axis may also come
        first. Sampled trilinearly, clamped at the border.
    order : int
        Spline order used to sample the image (3 = cubic B-spline, 1 = trilinear).
    n_iter, tol : int, float
        Fixed-point iterations X <- y - u(X): at most n_iter, stop when the
        largest change is below tol (voxels).
    slab : int
        Number of axis-0 planes solved at a time (bounds GPU memory).

    Returns
    -------
    deformed : cupy.ndarray, float64, shape (Nx, Ny, Nz)
    mismatch : float, max |X + u(X) - y| over all voxels, in voxels
    """
    image = cp.asarray(image, dtype=cp.float64)
    if image.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {image.shape}")
    nx, ny, nz = image.shape
    uc = _components(u, image.shape)

    def u_at(X):
        return cp.stack([map_coordinates(c, X, order=1, mode="nearest") for c in uc])

    X_all = cp.empty((3, nx * ny * nz), dtype=cp.float64)
    j = cp.arange(ny, dtype=cp.float64)
    k = cp.arange(nz, dtype=cp.float64)
    mismatch = 0.0
    for i0 in range(0, nx, slab):
        i = cp.arange(i0, min(i0 + slab, nx), dtype=cp.float64)
        y = cp.stack(cp.meshgrid(i, j, k, indexing="ij")).reshape(3, -1)
        X = y - u_at(y)
        for _ in range(n_iter):
            X_new = y - u_at(X)
            step = float(cp.abs(X_new - X).max())
            X = X_new
            if step < tol:
                break
        mismatch = max(mismatch, float(cp.abs(X + u_at(X) - y).max()))
        X_all[:, i0 * ny * nz:(i0 + len(i)) * ny * nz] = X
        del y, X, X_new
    deformed = map_coordinates(image, X_all, order=order, mode="nearest")
    return deformed.reshape(nx, ny, nz), mismatch


if __name__ == "__main__":
    import argparse
    import json
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Make the deformed image for a displacement field on the GPU "
                    "(default: correct Lagrangian warp, cubic B-spline; see the module docstring)."
    )
    parser.add_argument(
        "image",
        help="Input image file (.npz with array, or .npy). For .npz, specify array key with --image-key."
    )
    parser.add_argument(
        "displacement",
        nargs='?',
        default=None,
        help="Displacement field file (.npy, voxelFEM's nodal field (Nx+1, Ny+1, Nz+1, 3), in voxels; "
             "component axis first is also accepted). Ignored if --test is passed."
    )
    parser.add_argument(
        "--image-key",
        default="intensity",
        help="Key to extract from .npz file (default: intensity)."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file (.npy). If not specified, only prints stats. With the default warp, "
             "<output>.json records the method and the fixed-point mismatch."
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Spline order for sampling the image with the default warp (3 = cubic, 1 = trilinear). Default 3."
    )
    parser.add_argument(
        "--legacy-pull",
        action="store_true",
        help="Use the old pull warp image(x + u(x)) (or image(x - u(x)) with --negate), trilinear, with the "
             "nodal field averaged to voxel centres. First-order only: not for ground-truth images."
    )
    parser.add_argument(
        "--cval",
        type=float,
        default=0.0,
        help="--legacy-pull only: fill value for samples outside the volume (default: 0.0)."
    )
    parser.add_argument(
        "--negate",
        action="store_true",
        help="--legacy-pull only: negate the displacement field (flip sign convention)."
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Output dtype (e.g., float32, uint16). Default: float64 for the default warp, "
             "the input dtype for --legacy-pull."
    )
    parser.add_argument(
        "--disp-axis-last",
        action="store_true",
        help="--legacy-pull only: displacement shape is (Nx, Ny, Nz, 3), not (3, Nx, Ny, Nz) "
             "(detected automatically when the last axis has length 3)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate a synthetic displacement field (1%% uniaxial strain in x direction). "
             "Displacement argument is ignored if --test is passed."
    )

    args = parser.parse_args()

    # Load image
    if args.image.endswith(".npz"):
        data = np.load(args.image)
        print(f"Available keys in {args.image}: {list(data.keys())}")
        image_np = data[args.image_key]
        print(f"Loaded image '{args.image_key}' from {args.image}, shape {image_np.shape}, dtype {image_np.dtype}")
    else:
        image_np = np.load(args.image)
        print(f"Loaded image from {args.image}, shape {image_np.shape}, dtype {image_np.dtype}")

    # Load or generate displacement field
    if args.test:
        # Generate synthetic displacement field: 1% uniaxial strain in x direction
        Nx, Ny, Nz = image_np.shape
        xx = np.arange(Nx + 1, dtype=np.float32)
        yy = np.arange(Ny + 1, dtype=np.float32)
        zz = np.arange(Nz + 1, dtype=np.float32)
        xx_grid, yy_grid, zz_grid = np.meshgrid(xx, yy, zz, indexing='ij')

        # Affine strain: u_x = 0.01 * x, u_y = 0, u_z = 0
        u_np = np.zeros((3, Nx + 1, Ny + 1, Nz + 1), dtype=np.float32)
        u_np[0] = 0.01 * xx_grid
        print(f"Generated synthetic displacement (1%% x-strain), shape {u_np.shape}, dtype {u_np.dtype}")
    else:
        if args.displacement is None:
            raise ValueError("displacement argument required unless --test is passed")
        u_np = np.load(args.displacement)
        print(f"Loaded displacement from {args.displacement}, shape {u_np.shape}, dtype {u_np.dtype}")

    if not args.legacy_pull:
        deformed_gpu, mismatch = warp_lagrangian(image_np, u_np, order=args.order)
        cp.cuda.Stream.null.synchronize()
        warped_np = cp.asnumpy(deformed_gpu)
        if args.dtype is not None:
            dt = np.dtype(args.dtype)
            if np.issubdtype(dt, np.integer):
                info = np.iinfo(dt)
                warped_np = np.clip(np.rint(warped_np), info.min, info.max)
            warped_np = warped_np.astype(dt)
        print(f"Lagrangian warp (spline order {args.order}): max |X + u(X) - y| = {mismatch:.2e} voxel")
        print(f"Deformed shape: {warped_np.shape}, dtype: {warped_np.dtype}, "
              f"range {warped_np.min()} to {warped_np.max()}")
        if args.output is not None:
            np.save(args.output, warped_np)
            meta = dict(method="lagrangian_fixed_point", spline_order=args.order,
                        max_mismatch_vox=mismatch, image=args.image,
                        displacement="synthetic 1% x-strain" if args.test else args.displacement,
                        node_convention="node index = image coordinate")
            with open(args.output + ".json", "w") as fh:
                json.dump(meta, fh, indent=1)
            print(f"Saved deformed image to {args.output} (+ .json)")
        raise SystemExit(0)

    print("WARNING: --legacy-pull is a first-order warp; do not use it for ground-truth images.")

    # Handle axis ordering: convert (Nx, Ny, Nz, 3) to (3, Nx, Ny, Nz) if needed
    if args.disp_axis_last or (u_np.ndim == 4 and u_np.shape[-1] == 3):
        u_np = np.moveaxis(u_np, -1, 0)
        print(f"Transposed displacement to (3, Nx, Ny, Nz), new shape: {u_np.shape}")

    # Handle shape mismatch: interpolate displacement to voxel centers if it has one extra node in each direction
    if u_np.shape != (3,) + image_np.shape:
        if u_np.shape == (3, image_np.shape[0] + 1, image_np.shape[1] + 1, image_np.shape[2] + 1):
            # Interpolate displacement from nodes to voxel centers by averaging adjacent nodes
            # along each spatial dimension
            print(f"Interpolating displacement from nodes to voxel centers...")
            u_interp = np.zeros((3,) + image_np.shape, dtype=u_np.dtype)
            for c in range(3):
                # Interpolate along first spatial dimension (axis 1)
                u_x = 0.5 * (u_np[c, :-1, :, :] + u_np[c, 1:, :, :])
                # Interpolate along second spatial dimension (axis 2)
                u_xy = 0.5 * (u_x[:, :-1, :] + u_x[:, 1:, :])
                # Interpolate along third spatial dimension (axis 3)
                u_interp[c] = 0.5 * (u_xy[:, :, :-1] + u_xy[:, :, 1:])
            u_np = u_interp
            print(f"Interpolated displacement to voxel centers, new shape: {u_np.shape}")
        else:
            raise ValueError(
                f"Displacement shape {u_np.shape} does not match (3, {image_np.shape[0]}, "
                f"{image_np.shape[1]}, {image_np.shape[2]}). "
                f"Expected either exact match or one extra node per dimension."
            )

    # Transfer to GPU
    image_gpu = cp.asarray(image_np, dtype=cp.float32)
    u_gpu = cp.asarray(u_np, dtype=cp.float32)

    # Warp
    output_dtype = None
    if args.dtype is not None:
        output_dtype = cp.dtype(args.dtype)

    warped_gpu = warp_image(image_gpu, u_gpu, cval=args.cval, negate_u=args.negate, output_dtype=output_dtype)
    cp.cuda.Stream.null.synchronize()

    # Transfer back to CPU
    warped_np = cp.asnumpy(warped_gpu)

    print(f"Warped shape: {warped_np.shape}, dtype: {warped_np.dtype}")
    print(f"Warped range: {warped_np.min()} to {warped_np.max()}")

    # Save original intensity at original coordinates
    np.save("original_intensity.npy", image_np)
    print(f"Saved original intensity to original_intensity.npy")

    if args.output is not None:
        np.save(args.output, warped_np)
        print(f"Saved warped image to {args.output}")
