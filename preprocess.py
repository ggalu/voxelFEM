# -*- coding: utf-8 -*-
"""Compute grey level histogram from CT scan data, excluding void and air labels."""
import argparse
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from voxel_modulus import estimate_modulus_per_bin


def load_npz(filepath):
    """Load .npz file and return the data dictionary."""
    try:
        data = np.load(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: file not found '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading .npz file: {e}", file=sys.stderr)
        sys.exit(1)


def load_parameters(config_file):
    """Load parameters from JSON configuration file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: configuration file '{config_file}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON configuration: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compute grey level histogram from CT scan, excluding voids (label 0, 255). "
                    "Parameters are read from preprocess.json."
    )
    parser.add_argument("input", help="Input .npz file")
    parser.add_argument("--config", default="preprocess.json", help="Parameter configuration file (default: preprocess.json)")
    args = parser.parse_args()

    # Load parameters from JSON file
    params = load_parameters(args.config)

    # Extract parameters with type conversions
    label_key = params.get('label_key', 'labels')
    intensity_key = params.get('intensity_key', 'intensity')
    bins = int(params.get('bins', 256))
    exclude_labels = params.get('exclude_labels', [0, 255])
    output = params.get('output')
    text_output = params.get('text_output')
    E_matrix = float(params.get('E_matrix', 1.8))
    E_fibre = float(params.get('E_fibre', 72.0))
    fibre_fraction = float(params.get('fibre_fraction', 0.30))
    exponent = float(params.get('exponent', 1.0))
    mixing = params.get('mixing', 'geometric')

    # Load data
    data = load_npz(args.input)
    print(f"Available arrays: {list(data.files)}")

    # Get label and intensity arrays
    if label_key not in data.files:
        print(f"Error: key '{label_key}' not found in {args.input}", file=sys.stderr)
        print(f"Available keys: {list(data.files)}", file=sys.stderr)
        sys.exit(1)
    if intensity_key not in data.files:
        print(f"Error: key '{intensity_key}' not found in {args.input}", file=sys.stderr)
        print(f"Available keys: {list(data.files)}", file=sys.stderr)
        sys.exit(1)

    labels = data[label_key]
    intensity = data[intensity_key]

    # Validate shapes match
    if labels.shape != intensity.shape:
        print(f"Error: shape mismatch: labels {labels.shape} != intensity {intensity.shape}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded arrays:")
    print(f"  labels: shape {labels.shape}, dtype {labels.dtype}, min {labels.min()}, max {labels.max()}")
    print(f"  intensity: shape {intensity.shape}, dtype {intensity.dtype}, min {intensity.min()}, max {intensity.max()}")

    # Create mask for voxels to include (exclude specified labels)
    mask = np.ones(labels.shape, dtype=bool)
    for exclude_label in exclude_labels:
        mask &= (labels != exclude_label)

    n_included = np.sum(mask)
    n_total = labels.size
    print(f"\nIncluding {n_included} / {n_total} voxels ({100*n_included/n_total:.1f}%)")
    print(f"Excluding {n_total - n_included} voxels with labels: {exclude_labels}")

    # Extract intensities for included voxels
    intensities_filtered = intensity[mask]

    # Compute histogram
    hist, bin_edges = np.histogram(intensities_filtered, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Print statistics
    print(f"\nHistogram statistics:")
    print(f"  Mean: {intensities_filtered.mean():.2f}")
    print(f"  Std:  {intensities_filtered.std():.2f}")
    print(f"  Min:  {intensities_filtered.min()}")
    print(f"  Max:  {intensities_filtered.max()}")
    print(f"  Median: {np.median(intensities_filtered):.2f}")

    # Estimate modulus per bin
    mmap = estimate_modulus_per_bin(
        bin_centers, hist,
        E_matrix=E_matrix,
        E_fibre=E_fibre,
        fibre_volume_fraction=fibre_fraction,
        mixing=mixing,
        exponent=exponent,
    )

    print(f"\nModulus mapping:")
    print(f"  Matrix grey endpoint: {mmap.g_matrix:.1f}")
    print(f"  Fibre grey endpoint:  {mmap.g_fibre:.1f}")
    print(f"  Realised fibre fraction: {mmap.realised_fibre_fraction()*100:.2f}% (target: {fibre_fraction*100:.1f}%)")
    print(f"  Mixing rule: {mmap.mixing} (exponent={mmap.exponent})")
    print(f"  Young's modulus range: {mmap.E.min():.2f} - {mmap.E.max():.2f} GPa")

    # Plot histogram and Young's modulus
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: histogram
    ax1.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, color='steelblue', edgecolor='navy')
    ax1.set_xlabel('Grey Level (Intensity)')
    ax1.set_ylabel('Voxel Count')
    ax1.set_title(f'Grey Level Histogram\n({n_included:,} voxels, excluding labels {exclude_labels})')
    ax1.grid(True, alpha=0.3)

    # Add statistics to left plot
    stats_text = f'Mean: {intensities_filtered.mean():.1f}\nStd: {intensities_filtered.std():.1f}\nMedian: {np.median(intensities_filtered):.1f}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: Young's modulus vs grey level
    ax2.plot(mmap.bin_centers, mmap.E, linewidth=2.5, color='darkred', label='Young\'s Modulus')
    ax2.scatter(mmap.bin_centers, mmap.E, s=30, color='darkred', alpha=0.6, zorder=3)
    ax2.axhline(E_matrix, color='green', linestyle='--', linewidth=1.5, label=f'Matrix ({E_matrix:.1f} GPa)')
    ax2.axhline(E_fibre, color='blue', linestyle='--', linewidth=1.5, label=f'Fibre ({E_fibre:.1f} GPa)')
    ax2.axvline(mmap.g_matrix, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'Matrix grey ({mmap.g_matrix:.0f})')
    ax2.axvline(mmap.g_fibre, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'Fibre grey ({mmap.g_fibre:.0f})')
    ax2.set_xlabel('Grey Level (Intensity)')
    ax2.set_ylabel('Young\'s Modulus (GPa)')
    ax2.set_title(f'Young\'s Modulus Mapping\n({mmap.mixing}, exponent={mmap.exponent})')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')

    # Add material parameters to right plot
    mat_text = f'E_matrix={E_matrix:.1f}, E_fibre={E_fibre:.1f}\nφ_target={fibre_fraction:.1%}, φ_realised={mmap.realised_fibre_fraction():.1%}'
    ax2.text(0.02, 0.98, mat_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"\nHistogram plot saved to '{output}'")
    else:
        plt.show()

    # Save histogram data to text file
    if text_output:
        with open(text_output, 'w') as f:
            f.write("# Grey Level Histogram and Young's Modulus Mapping\n")
            f.write(f"# Total voxels: {n_total:,}\n")
            f.write(f"# Included voxels: {n_included:,}\n")
            f.write(f"# Excluded labels: {exclude_labels}\n")
            f.write(f"# Statistics: mean={intensities_filtered.mean():.2f}, std={intensities_filtered.std():.2f}, median={np.median(intensities_filtered):.2f}\n")
            f.write(f"# \n")
            f.write(f"# Modulus Mapping Parameters:\n")
            f.write(f"# E_matrix={E_matrix} GPa, E_fibre={E_fibre} GPa\n")
            f.write(f"# Fibre volume fraction (target)={fibre_fraction:.1%}, (realised)={mmap.realised_fibre_fraction():.1%}\n")
            f.write(f"# Mixing rule={mmap.mixing}, exponent={mmap.exponent}\n")
            f.write(f"# Matrix grey={mmap.g_matrix:.1f}, Fibre grey={mmap.g_fibre:.1f}\n")
            f.write("# \n")
            f.write("bin_center\tcount\tfibre_fraction\tYoungs_modulus_GPa\n")
            for center, count, phi, E in zip(mmap.bin_centers, mmap.counts, mmap.phi, mmap.E):
                f.write(f"{center:.1f}\t{count}\t{phi:.4f}\t{E:.4f}\n")
        print(f"Histogram data saved to '{text_output}'")

    # Create per-voxel Young's modulus array and save to .npz
    E_per_voxel = np.zeros_like(intensity, dtype=np.float32)

    # Map each intensity value to corresponding modulus via histogram binning
    bin_indices = np.digitize(intensity.ravel(), bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(mmap.E) - 1)
    E_per_voxel.ravel()[:] = mmap.E[bin_indices]

    # Set excluded label voxels to 0.0 (void)
    for exclude_label in exclude_labels:
        E_per_voxel[labels == exclude_label] = 0.0

    # Reload the input file and save with new Young_modulus array
    data_dict = {name: data[name] for name in data.files}

    # Warn if overwriting existing Young_modulus
    if 'Young_modulus' in data.files:
        print(f"\n⚠ WARNING: Overwriting existing 'Young_modulus' array in '{args.input}'", file=sys.stderr)

    data_dict['Young_modulus'] = E_per_voxel
    np.savez(args.input, **data_dict)
    print(f"Young's modulus array appended to '{args.input}' with key 'Young_modulus'")

    # Print analytic mapping formula and parameters
    print(f"\n{'='*70}")
    print(f"ANALYTIC MAPPING FORMULA: Intensity (grey level) → Young's Modulus")
    print(f"{'='*70}")

    # Print the formula
    print(f"\nStep 1: Fibre volume fraction from grey level")
    print(f"  φ(g) = clip((g − g_matrix) / (g_fibre − g_matrix), 0, 1)")

    if mmap.mixing == "geometric":
        print(f"\nStep 2: Young's modulus via geometric (power-law) mixing")
        print(f"  E(g) = E_matrix · (E_fibre / E_matrix)^(p · φ(g))")
    elif mmap.mixing == "voigt":
        print(f"\nStep 2: Young's modulus via Voigt (stiff bound) mixing")
        print(f"  E(g) = E_matrix + φ(g) · (E_fibre − E_matrix)")
    elif mmap.mixing == "reuss":
        print(f"\nStep 2: Young's modulus via Reuss (compliant bound) mixing")
        print(f"  E(g) = 1 / (φ(g) / E_fibre + (1 − φ(g)) / E_matrix)")

    print(f"\n{'-'*70}")
    print(f"PARAMETERS:")
    print(f"{'-'*70}")
    print(f"  g_matrix (matrix grey endpoint)      = {mmap.g_matrix:.4f}")
    print(f"  g_fibre  (fibre grey endpoint)       = {mmap.g_fibre:.4f}")
    print(f"  E_matrix (matrix Young's modulus)    = {E_matrix:.4f} GPa")
    print(f"  E_fibre  (fibre Young's modulus)     = {E_fibre:.4f} GPa")
    print(f"  Mixing rule                          = {mmap.mixing}")
    if mmap.mixing == "geometric":
        print(f"  p (geometric exponent)               = {mmap.exponent:.4f}")
    print(f"  Target fibre volume fraction         = {fibre_fraction:.1%}")
    print(f"  Realised fibre volume fraction       = {mmap.realised_fibre_fraction():.1%}")


if __name__ == "__main__":
    main()
