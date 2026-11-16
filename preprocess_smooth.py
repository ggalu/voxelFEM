# -*- coding: utf-8 -*-
"""
Smooth the per-voxel Young's modulus field near solid/void interfaces
=======================================================================
Post-processing step applied **after** ``preprocess.py``. Voxel FEM meshes
represent curved/rough solid-void interfaces as a "staircase" of axis-aligned
faces with many 90-degree re-entrant corners. Combined with the sharp
material jump ``E_solid -> 0`` at the void boundary, these corners produce a
near-singular local stress concentration that does not converge under mesh
refinement (a numerical artefact, on top of any genuine curvature-driven
concentration).

This script reduces that artefact by ramping ``Young_modulus`` down from its
bulk value to ``floor_fraction * E_bulk`` over ``ramp_width`` voxel layers as
solid voxels approach the interface, using a cosine (smoothstep) profile.
Only the **solid side** of the interface is touched: void/air voxels (labels
in ``--void-labels``, default ``[0, 255]``) are left at ``E = 0`` and remain
excluded from the FEM system via ``solid_mask`` in ``simulateCT.py`` -
unchanged.

The floor is kept well above zero (default 3% of each voxel's own bulk
modulus) so the added contrast stays comparable to existing fibre/matrix
contrast (~40:1) rather than reintroducing the high-contrast regime where
Jacobi/multigrid preconditioning degrades.

Usage
-----
    python preprocess_smooth.py input.npz output.npz
    python preprocess_smooth.py input.npz output.npz --ramp-width 3 --floor 0.03

Input: ``.npz`` with ``labels`` and ``Young_modulus`` (output of
``preprocess.py``). Output: a copy of the input with ``Young_modulus``
replaced by the smoothed field; the original is kept under
``Young_modulus_unsmoothed`` for comparison.
"""

import argparse
import sys
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    print("Error: scipy is required for preprocess_smooth.py "
          "(scipy.ndimage.distance_transform_edt)", file=sys.stderr)
    sys.exit(1)


DEFAULT_VOID_LABELS = (0, 255)


def smooth_young_modulus(Young_modulus, labels, void_labels=DEFAULT_VOID_LABELS,
                          ramp_width=3, floor_fraction=0.03):
    """
    Ramp ``Young_modulus`` down to ``floor_fraction * E_bulk`` over
    ``ramp_width`` voxel layers approaching a solid/void interface.

    Only solid voxels (``labels`` not in ``void_labels``) are modified; void
    voxels stay at their original value (0). Returns the smoothed array and
    the per-voxel scale factor applied (both same shape as ``Young_modulus``).
    """
    solid_mask = ~np.isin(labels, list(void_labels))

    if not np.any(~solid_mask):
        # No void voxels present - nothing to smooth.
        return Young_modulus.astype(np.float32), np.ones_like(Young_modulus, dtype=np.float32)

    # Distance (in voxels) from each solid voxel to the nearest void voxel;
    # 0 for void voxels themselves.
    dist = distance_transform_edt(solid_mask)

    if ramp_width > 1:
        t = np.clip((dist - 1.0) / (ramp_width - 1.0), 0.0, 1.0)
    else:
        t = np.where(dist <= 1.0, 0.0, 1.0)

    # Cosine smoothstep: 0 at the interface (factor -> floor_fraction),
    # 1 at >= ramp_width voxels away (factor -> 1, unchanged).
    s = 0.5 * (1.0 - np.cos(np.pi * t))
    factor = floor_fraction + (1.0 - floor_fraction) * s

    # Only solid voxels within ramp_width of the interface are affected.
    affected = solid_mask & (dist <= ramp_width)
    factor = np.where(affected, factor, 1.0).astype(np.float32)

    return (Young_modulus * factor).astype(np.float32), factor


def main():
    parser = argparse.ArgumentParser(
        description="Smooth Young's modulus near solid/void interfaces to "
                     "reduce staircase stress-concentration artefacts."
    )
    parser.add_argument("input", help="Input .npz (must have 'labels' and 'Young_modulus')")
    parser.add_argument("output", help="Output .npz (copy of input with smoothed 'Young_modulus')")
    parser.add_argument("--ramp-width", type=int, default=3,
                         help="Transition width in voxel layers (default: 3)")
    parser.add_argument("--floor", type=float, default=0.03,
                         help="Floor fraction of each voxel's bulk modulus at the "
                              "interface, E_floor = floor * E_bulk (default: 0.03)")
    parser.add_argument("--void-labels", type=int, nargs="+", default=list(DEFAULT_VOID_LABELS),
                         help="Label values treated as void/air (default: 0 255)")
    args = parser.parse_args()

    data = np.load(args.input)
    if "labels" not in data.files:
        print(f"Error: 'labels' not found in '{args.input}'. "
              f"Available keys: {list(data.files)}", file=sys.stderr)
        sys.exit(1)
    if "Young_modulus" not in data.files:
        print(f"Error: 'Young_modulus' not found in '{args.input}'. "
              f"Run preprocess.py first. Available keys: {list(data.files)}", file=sys.stderr)
        sys.exit(1)

    labels = data["labels"]
    E = data["Young_modulus"].astype(np.float32)

    print(f"Loaded '{args.input}':")
    print(f"  shape: {labels.shape}")
    print(f"  Young_modulus range: [{E.min():.4f}, {E.max():.4f}]")
    print(f"  void labels        : {args.void_labels}")
    print(f"  ramp width         : {args.ramp_width} voxel layer(s)")
    print(f"  floor fraction     : {args.floor} (E_floor = {args.floor} * E_bulk per voxel)")

    E_smooth, factor = smooth_young_modulus(
        E, labels, void_labels=args.void_labels,
        ramp_width=args.ramp_width, floor_fraction=args.floor,
    )

    n_affected = int(np.sum(factor < 1.0))
    n_total = labels.size
    print(f"\n  Voxels in transition layer: {n_affected:,} / {n_total:,} "
          f"({100 * n_affected / n_total:.2f}%)")
    print(f"  Young_modulus range after smoothing: [{E_smooth.min():.4f}, {E_smooth.max():.4f}]")

    data_dict = {name: data[name] for name in data.files}
    if "Young_modulus_unsmoothed" in data.files:
        print(f"\n⚠ WARNING: Overwriting existing 'Young_modulus_unsmoothed' array", file=sys.stderr)
    data_dict["Young_modulus_unsmoothed"] = E
    data_dict["Young_modulus"] = E_smooth

    np.savez(args.output, **data_dict)
    print(f"\nWrote '{args.output}' with smoothed 'Young_modulus' "
          f"(original kept as 'Young_modulus_unsmoothed').")


if __name__ == "__main__":
    main()
