"""
Standalone: apply a FEM displacement field to a gray-scale intensity volume,
producing a deformed intensity field.

This script is fully self-contained — it depends only on cupy (and numpy for
I/O).

Convention (this is the important part)
---------------------------------------
The solver produces a displacement field where a reference material point X
moves to its deformed position

    x = X + u(X).

For an isotropic tensile (expanding) strain eps, u(X) = eps * X, so a feature
spanning [0, L] in the reference spans [0, L*(1+eps)] after deformation: the
image content must GROW.

To render the deformed image we use a BACKWARD (pull) warp — for each output
voxel x we read the reference gray value from where that material came from:

    I_def(x) = I_ref( x - u(x) ).

Sanity check with u(x) = eps*x:

    I_def(x) = I_ref( x*(1 - eps) )

so a feature at reference position X shows up at x = X / (1 - eps) ~= X*(1+eps).
Its edge length L therefore becomes ~ L*(1+eps) — an expansion, as required.
(A forward/push convention I_ref(x + u) would instead SHRINK a tensile field;
that is the opposite of what we want here.)

The pull warp is exact to first order in eps for a general (non-affine) field
and is the standard way to resample a deformed image without leaving holes.

Inputs (defaults match simulateCT.py's outputs)
-----------------------------------------------
    ct_result_intensity.npy      (Nx, Ny, Nz)          reference gray levels
    ct_result_displacement.npy   (Nx+1, Ny+1, Nz+1, 3) nodal displacement (voxels)

Outputs
-------
    ref.npy   (Nx, Ny, Nz)   reference (original) gray levels, copied through
    def.npy   (Nx, Ny, Nz)   deformed (warped) gray levels
"""

import argparse
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates


def nodal_to_voxel_centre(u_nodal):
    """Average each voxel's 8 corner nodes to get the voxel-centre displacement.

    Parameters
    ----------
    u_nodal : cupy.ndarray, shape (3, Nx+1, Ny+1, Nz+1)

    Returns
    -------
    u_centre : cupy.ndarray, shape (3, Nx, Ny, Nz)
    """
    u = u_nodal
    u = 0.5 * (u[:, :-1, :, :] + u[:, 1:, :, :])
    u = 0.5 * (u[:, :, :-1, :] + u[:, :, 1:, :])
    u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
    return u


def deform_image(image, u_centre, cval=0.0, output_dtype=None):
    """Backward (pull) trilinear warp: deformed[x] = image(x - u_centre[x]).

    Parameters
    ----------
    image : cupy.ndarray, shape (Nx, Ny, Nz)
        Reference gray-value volume on the GPU.
    u_centre : cupy.ndarray, shape (3, Nx, Ny, Nz)
        Displacement at voxel centres, in voxel units, component-first.
    cval : float
        Fill value for sample points outside the reference volume.
    output_dtype : dtype or None
        Result dtype; defaults to image.dtype.

    Returns
    -------
    deformed : cupy.ndarray, shape (Nx, Ny, Nz)
    """
    if image.ndim != 3:
        raise ValueError(f"image must be 3D, got {image.shape}")
    if u_centre.shape != (3,) + image.shape:
        raise ValueError(
            f"u_centre must be (3, {image.shape[0]}, {image.shape[1]}, "
            f"{image.shape[2]}), got {u_centre.shape}")

    # coords[i] = index_i - u_i   ->  pull from the reference configuration.
    coords = cp.indices(image.shape, dtype=cp.float32)
    coords -= u_centre.astype(cp.float32, copy=False)

    deformed = map_coordinates(
        image.astype(cp.float32, copy=False), coords,
        order=1, mode="constant", cval=cval, prefilter=False)

    if output_dtype is None:
        output_dtype = image.dtype
    if deformed.dtype != cp.dtype(output_dtype):
        # Round before casting to an integer dtype so gray levels are nearest,
        # not truncated toward zero.
        if cp.issubdtype(cp.dtype(output_dtype), cp.integer):
            deformed = cp.rint(deformed)
        deformed = deformed.astype(output_dtype)
    return deformed


def _normalize_disp(u_np):
    """Return the displacement field as component-first (3, X, Y, Z)."""
    if u_np.ndim == 4 and u_np.shape[-1] == 3:
        u_np = np.moveaxis(u_np, -1, 0)          # (...,3) -> (3,...)
    if u_np.ndim != 4 or u_np.shape[0] != 3:
        raise ValueError(f"displacement must be (3,X,Y,Z) or (X,Y,Z,3), got "
                         f"{u_np.shape}")
    return u_np


def to_voxel_centre(u_np, image_shape):
    """Bring a (3, X, Y, Z) displacement onto the voxel-centre grid, on the GPU.

    Accepts nodal ((3, Nx+1, Ny+1, Nz+1)) or already voxel-centred
    ((3, Nx, Ny, Nz)) layouts and returns a (3, Nx, Ny, Nz) cupy array.
    """
    u = cp.asarray(u_np, dtype=cp.float32)
    Nx, Ny, Nz = image_shape
    if u.shape == (3, Nx + 1, Ny + 1, Nz + 1):
        u = nodal_to_voxel_centre(u)
        print(f"Interpolated nodal displacement -> voxel centres: {u.shape}")
    elif u.shape == (3, Nx, Ny, Nz):
        print("Displacement already on the voxel-centre grid.")
    else:
        raise ValueError(
            f"displacement grid {tuple(u.shape[1:])} incompatible with "
            f"intensity {image_shape} (want {(Nx,Ny,Nz)} or {(Nx+1,Ny+1,Nz+1)}).")
    return u


def mean_normal_strains(u_np):
    """Volume-average normal strains (eps_xx, eps_yy, eps_zz) from displacement.

    By the divergence theorem, the volume average of the normal strain along
    axis i is

        <du_i/dx_i> = ( mean(u_i on the +i face) - mean(u_i on the -i face) ) / L_i

    which depends only on the two extreme planes and is exact for the volume
    average of a nodal FEM field. `u_np` is component-first (3, X, Y, Z), in
    voxel units (voxel side = 1), so L_i is the plane-to-plane span in voxels.

    Returns
    -------
    (eps_xx, eps_yy, eps_zz) : tuple of float
    """
    strains = []
    for ax in range(3):
        comp = u_np[ax]                                  # (X, Y, Z)
        n = comp.shape[ax]
        L = n - 1                                        # span between extreme planes
        near = np.take(comp, 0, axis=ax).mean()
        far = np.take(comp, n - 1, axis=ax).mean()
        strains.append(float((far - near) / L))
    return tuple(strains)


def _self_test():
    """Verify the edge-length law: isotropic tensile eps -> edge L*(1+eps)."""
    print("=== self-test: isotropic tensile expansion ===")
    N = 80
    eps = 0.1
    # A bright cube of edge L, offset from the origin (u = eps*X expands about
    # the origin, so a feature at [a,b] maps to [a/(1-eps), b/(1-eps)]).
    a, b = 10, 42            # reference span -> edge L = 32
    L = b - a
    img = np.zeros((N, N, N), dtype=np.float32)
    img[a:b, a:b, a:b] = 1000.0

    # Nodal affine displacement u(X) = eps * X on the (N+1)^3 node grid.
    idx = np.indices((N + 1, N + 1, N + 1), dtype=np.float32)
    u_nodal = (eps * idx).astype(np.float32)      # (3, N+1, N+1, N+1)

    image_gpu = cp.asarray(img)
    u_gpu = load_displacement_from_array(u_nodal, img.shape)
    out = cp.asnumpy(deform_image(image_gpu, u_gpu, cval=0.0,
                                  output_dtype=np.float32))

    # Measure the bright-region bounding box along axis 0 (threshold at half-max).
    mask = out > 500.0
    coords0 = np.where(mask.any(axis=(1, 2)))[0]
    lo, hi = coords0.min(), coords0.max() + 1     # +1: bounding box is inclusive
    measured_edge = hi - lo
    expected_edge = L / (1.0 - eps)               # exact for this affine field

    print(f"  reference edge L        : {L}")
    print(f"  measured deformed edge  : {measured_edge}  (span [{lo},{hi}])")
    print(f"  expected L/(1-eps)      : {expected_edge:.2f}")
    print(f"  first-order L*(1+eps)   : {L*(1+eps):.2f}")
    grew = measured_edge > L
    ok = grew and abs(measured_edge - expected_edge) <= 1.0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(image {'expanded' if grew else 'did NOT expand'})")
    return ok


def load_displacement_from_array(u_np, image_shape):
    """Bring an in-memory nodal displacement to voxel centres (used by the test)."""
    u = cp.asarray(u_np, dtype=cp.float32)
    Nx, Ny, Nz = image_shape
    if u.shape == (3, Nx + 1, Ny + 1, Nz + 1):
        u = nodal_to_voxel_centre(u)
    return u


def main():
    p = argparse.ArgumentParser(
        description="Apply a FEM displacement field to an intensity volume, "
                    "producing a deformed (tensile => expanded) intensity field.")
    p.add_argument("--intensity", default="ct_result_intensity.npy",
                   help="Reference gray-scale volume (.npy, Nx*Ny*Nz).")
    p.add_argument("--displacement", default="ct_result_displacement.npy",
                   help="Nodal displacement field (.npy).")
    p.add_argument("--output", "-o", default="def.npy",
                   help="Deformed-intensity output .npy (default: def.npy).")
    p.add_argument("--ref-output", default="ref.npy",
                   help="Reference (original) intensity output .npy (default: ref.npy).")
    p.add_argument("--cval", type=float, default=0.0,
                   help="Fill value for samples outside the reference volume.")
    p.add_argument("--dtype", default=None,
                   help="Output dtype (e.g. uint16, float32). Default: match input.")
    p.add_argument("--test", action="store_true",
                   help="Run the isotropic-expansion self-test and exit.")
    args = p.parse_args()

    if args.test:
        ok = _self_test()
        raise SystemExit(0 if ok else 1)

    image_np = np.load(args.intensity)
    if image_np.ndim != 3:
        raise ValueError(f"intensity must be 3D, got {image_np.shape}")
    print(f"Loaded intensity {args.intensity}: {image_np.shape}, {image_np.dtype}")

    # Write the reference (original) intensities alongside the deformed result.
    np.save(args.ref_output, image_np)
    print(f"Wrote reference intensity: {args.ref_output}")

    u_np = _normalize_disp(np.load(args.displacement))
    print(f"Loaded displacement {args.displacement}: {u_np.shape}, {u_np.dtype}")

    eps_xx, eps_yy, eps_zz = mean_normal_strains(u_np)
    print("Mean normal strains (volume-average):")
    print(f"  eps_xx = {eps_xx:+.6e}")
    print(f"  eps_yy = {eps_yy:+.6e}")
    print(f"  eps_zz = {eps_zz:+.6e}")

    image_gpu = cp.asarray(image_np)
    u_gpu = to_voxel_centre(u_np, image_np.shape)

    out_dtype = np.dtype(args.dtype) if args.dtype else image_np.dtype
    deformed = deform_image(image_gpu, u_gpu, cval=args.cval, output_dtype=out_dtype)
    cp.cuda.Stream.null.synchronize()
    deformed_np = cp.asnumpy(deformed)

    print(f"Deformed intensity: {deformed_np.shape}, {deformed_np.dtype}, "
          f"range [{deformed_np.min()}, {deformed_np.max()}]")
    np.save(args.output, deformed_np)
    print(f"Wrote deformed intensity: {args.output}")


if __name__ == "__main__":
    main()
