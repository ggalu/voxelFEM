"""
GPU trilinear backward (pull) warping of a 3D image by a displacement field.

Operation
---------
    warped[x] = image(x + u[x])

For each output voxel x, the sample location in the input is x + u[x];
the gray value there is obtained by trilinear interpolation. Sample points
falling outside the input volume take a constant fill value (mode='constant').

Sign convention
---------------
If your displacement field is defined in the opposite sense (where each
reference point moves *to*, rather than where each output voxel reads *from*),
pass `-u` or set `negate_u=True`.
"""

import cupy as cp
from cupyx.scipy.ndimage import map_coordinates


def warp_image(image, u, cval=0.0, negate_u=False, output_dtype=None):
    """
    Trilinear backward warp of a 3D volume by a displacement field, on the GPU.

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


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Warp a 3D image by a displacement field using GPU trilinear interpolation."
    )
    parser.add_argument(
        "image",
        help="Input image file (.npz with array, or .npy). For .npz, specify array key with --image-key."
    )
    parser.add_argument(
        "displacement",
        help="Displacement field file (.npy, shape (3, Nx, Ny, Nz), in voxels)."
    )
    parser.add_argument(
        "--image-key",
        default="intensity",
        help="Key to extract from .npz file (default: intensity)."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file (.npy). If not specified, only prints stats."
    )
    parser.add_argument(
        "--cval",
        type=float,
        default=0.0,
        help="Fill value for samples outside the volume (default: 0.0)."
    )
    parser.add_argument(
        "--negate",
        action="store_true",
        help="Negate the displacement field (flip sign convention)."
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Output dtype (e.g., float32, uint8). If not specified, matches input image."
    )
    parser.add_argument(
        "--disp-axis-last",
        action="store_true",
        help="If set, assume displacement shape is (Nx, Ny, Nz, 3), not (3, Nx, Ny, Nz)."
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

    # Load displacement field
    u_np = np.load(args.displacement)
    print(f"Loaded displacement from {args.displacement}, shape {u_np.shape}, dtype {u_np.dtype}")

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
