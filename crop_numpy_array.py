# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-04 18:56:00
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-09 11:08:40
"""Crop a numpy .npz array to a specified Nx×Ny×Nz region (centered)."""
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt


def crop_centered(arr, nx, ny, nz, offset_x=0, offset_y=0, offset_z=0):
    sx, sy, sz = arr.shape
    if nx > sx or ny > sy or nz > sz:
        print(f"Error: crop ({nx},{ny},{nz}) exceeds array shape ({sx},{sy},{sz})", file=sys.stderr)
        sys.exit(1)
    x0 = (sx - nx) // 2 + offset_x
    y0 = (sy - ny) // 2 + offset_y
    z0 = (sz - nz) // 2 + offset_z

    # Validate offsets don't push crop outside bounds
    if x0 < 0 or x0 + nx > sx:
        print(f"Error: X offset {offset_x} pushes crop outside bounds (X: {x0}..{x0+nx}, array: 0..{sx})", file=sys.stderr)
        sys.exit(1)
    if y0 < 0 or y0 + ny > sy:
        print(f"Error: Y offset {offset_y} pushes crop outside bounds (Y: {y0}..{y0+ny}, array: 0..{sy})", file=sys.stderr)
        sys.exit(1)
    if z0 < 0 or z0 + nz > sz:
        print(f"Error: Z offset {offset_z} pushes crop outside bounds (Z: {z0}..{z0+nz}, array: 0..{sz})", file=sys.stderr)
        sys.exit(1)

    return arr[x0:x0+nx, y0:y0+ny, z0:z0+nz]


def main():
    parser = argparse.ArgumentParser(description="Crop a .npz array to Nx×Ny×Nz (centered with optional offset).")
    parser.add_argument("input",  help="Input .npz file")
    parser.add_argument("output", help="Output .npz file")
    parser.add_argument("Nx", type=int)
    parser.add_argument("Ny", type=int)
    parser.add_argument("Nz", type=int)
    parser.add_argument("--offset", type=int, nargs=3, default=[0, 0, 0], metavar=('OX', 'OY', 'OZ'),
                        help="Shift crop in X, Y, Z directions (default: 0 0 0, centered)")
    args = parser.parse_args()

    data = np.load(args.input)
    keys = list(data.files)
    if not keys:
        print(f"Error: no arrays found in {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate all arrays are 3-D with the same shape
    shapes = {}
    for key in keys:
        print("key:", key)
        arr = data[key]

        if arr.ndim != 3:
            print(f"Error: array '{key}' has shape {arr.shape}; expected 3-D", file=sys.stderr)
            sys.exit(1)
        shapes[key] = arr.shape

    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        print(f"Error: arrays have different shapes: {shapes}", file=sys.stderr)
        sys.exit(1)

    src_shape = next(iter(unique_shapes))
    cropped = {key: crop_centered(data[key], args.Nx, args.Ny, args.Nz, args.offset[0], args.offset[1], args.offset[2]) for key in keys}
    np.savez_compressed(args.output, **cropped)
    crop_shape = next(iter(cropped.values())).shape
    print(f"Cropped {src_shape} → {crop_shape}, saved to '{args.output}' (keys: {keys})")

    # Visualize central slices from all 3 directions
    if 'labels' in cropped:
        labels = cropped['labels']
        nx, ny, nz = labels.shape
        x_mid = nx // 2
        y_mid = ny // 2
        z_mid = nz // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # YZ plane (X-direction)
        slice_yz = labels[x_mid, :, :]
        im0 = axes[0].imshow(slice_yz, cmap='gray', origin='lower')
        axes[0].set_title(f'YZ plane (X={x_mid})')
        axes[0].set_xlabel('Z')
        axes[0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0], label='Label')

        # XZ plane (Y-direction)
        slice_xz = labels[:, y_mid, :]
        im1 = axes[1].imshow(slice_xz, cmap='gray', origin='lower')
        axes[1].set_title(f'XZ plane (Y={y_mid})')
        axes[1].set_xlabel('Z')
        axes[1].set_ylabel('X')
        plt.colorbar(im1, ax=axes[1], label='Label')

        # XY plane (Z-direction)
        slice_xy = labels[:, :, z_mid]
        im2 = axes[2].imshow(slice_xy, cmap='gray', origin='lower')
        axes[2].set_title(f'XY plane (Z={z_mid})')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('X')
        plt.colorbar(im2, ax=axes[2], label='Label')

        fig.suptitle(f'Central slices of cropped labels\nShape: {crop_shape}', fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
