#!/usr/bin/env python
"""
VTK Export — per-voxel FEM fields
=================================
Write per-voxel (per-element) FEM fields to a regular-voxel structured grid,
readable directly in ParaView / VisIt.  Two writers:

  * ``save_stress_strain_vtk`` / ``write_structured_points_vtk`` — legacy ASCII
    **VTK STRUCTURED_POINTS** (``vtkImageData``).  Pure NumPy, no GPU and no
    external dependency.  Writes effective scalars + all 12 component scalars.

  * ``save_stress_strain_vti`` — binary, zlib-compressed **XML ImageData
    (.vti)** via ``pyvista`` (optional dependency).  Writes effective scalars
    plus strain/stress in proper **tensor notation** (compact 6-component
    symmetric, or full 9-component); omits the redundant component scalars.

Each voxel is one Q1 hex element, so the fields are written as CELL_DATA on a
grid of ``Nx x Ny x Nz`` cells (``(Nx+1) x (Ny+1) x (Nz+1)`` points).

Element/voxel ordering convention
---------------------------------
The solver stores per-element arrays in C-order over ``(Nx, Ny, Nz)`` — i.e.
element ``e = i*Ny*Nz + j*Nz + k`` with the z-index ``k`` fastest.  VTK
STRUCTURED_POINTS expects cell data with the **x-index fastest**, so this
module reorders every field with ``reshape(Nx,Ny,Nz).ravel(order='F')`` before
writing.  Pass ``grid_shape=(Nx,Ny,Nz)`` so the reorder is correct.

Voigt convention (matches ``geometry._B_matrix``)
-------------------------------------------------
Strain/stress tensors are 6-component Voigt vectors ordered
``[xx, yy, zz, xy, yz, xz]`` (engineering shear strains for ``strain``).

Typical use
-----------
    from vtk_export import save_stress_strain_vtk
    save_stress_strain_vtk("out.vtk", (Nx, Ny, Nz), strain, stress,
                           extra={"youngs_modulus": E_elem})
"""

import numpy as np

# Output float datatype for VTK export — change here to float64 if needed
OUTPUT_DTYPE = np.float32

# ---------------------------------------------------------------------------
# Equivalent (effective) scalar measures
# ---------------------------------------------------------------------------

def von_mises_strain(eps):
    """
    Von Mises equivalent strain  eps_vm = sqrt(2/3 * dev(eps):dev(eps)).

    Parameters
    ----------
    eps : (..., 6) array — Voigt [xx, yy, zz, xy, yz, xz], engineering shears.
    """
    eps = np.asarray(eps, dtype=OUTPUT_DTYPE)
    exx, eyy, ezz = eps[..., 0], eps[..., 1], eps[..., 2]
    gxy, gyz, gxz = eps[..., 3], eps[..., 4], eps[..., 5]
    emean = (exx + eyy + ezz) / 3.0
    dxx, dyy, dzz = exx - emean, eyy - emean, ezz - emean
    # engineering -> tensor shear (divide by 2)
    exy, eyz, exz = gxy / 2.0, gyz / 2.0, gxz / 2.0
    return np.sqrt(2.0 / 3.0 * (dxx**2 + dyy**2 + dzz**2
                                + 2.0 * (exy**2 + eyz**2 + exz**2)))


def von_mises_stress(sigma):
    """
    Von Mises equivalent stress
        sigma_vm = sqrt(0.5*[(sxx-syy)^2+(syy-szz)^2+(szz-sxx)^2]
                        + 3*(sxy^2 + syz^2 + sxz^2)).

    Parameters
    ----------
    sigma : (..., 6) array — Voigt [xx, yy, zz, xy, yz, xz].
    """
    sigma = np.asarray(sigma, dtype=OUTPUT_DTYPE)
    sxx, syy, szz = sigma[..., 0], sigma[..., 1], sigma[..., 2]
    sxy, syz, sxz = sigma[..., 3], sigma[..., 4], sigma[..., 5]
    return np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
                   + 3.0 * (sxy**2 + syz**2 + sxz**2))


# ---------------------------------------------------------------------------
# Low-level writer
# ---------------------------------------------------------------------------

def _reorder_to_vtk(field, grid_shape):
    """Element-order (z fastest) -> VTK cell-order (x fastest)."""
    Nx, Ny, Nz = grid_shape
    field = np.asarray(field).reshape(Nx, Ny, Nz)
    return field.ravel(order="F")


def write_structured_points_vtk(filename, grid_shape, cell_data,
                                origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0),
                                title="voxel FEM fields", fmt="%.7g"):
    """
    Write a legacy ASCII VTK STRUCTURED_POINTS file with per-voxel CELL_DATA.

    Parameters
    ----------
    filename   : str — output path (.vtk)
    grid_shape : (Nx, Ny, Nz) — number of voxels (cells) along each axis
    cell_data  : dict {name: array}.  Each array has Nx*Ny*Nz entries and is
                 either (Ne,) -> SCALARS or (Ne, 3) -> VECTORS, in element order.
    origin     : grid origin (default (0,0,0))
    spacing    : voxel size along each axis (default unit cubes)
    title      : free-text header line
    fmt        : value format string
    """
    Nx, Ny, Nz = grid_shape
    n_cell = Nx * Ny * Nz

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx + 1} {Ny + 1} {Nz + 1}\n")
        f.write(f"ORIGIN {origin[0]:g} {origin[1]:g} {origin[2]:g}\n")
        f.write(f"SPACING {spacing[0]:g} {spacing[1]:g} {spacing[2]:g}\n")
        f.write(f"CELL_DATA {n_cell}\n")

        for name, arr in cell_data.items():
            arr = np.asarray(arr)
            if arr.shape[0] != n_cell:
                raise ValueError(
                    f"field '{name}' has {arr.shape[0]} entries, "
                    f"expected {n_cell} (= Nx*Ny*Nz)")

            if arr.ndim == 1:
                vals = _reorder_to_vtk(arr, grid_shape)
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                np.savetxt(f, vals, fmt=fmt)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                # reorder each component, then interleave x y z per cell
                cols = [_reorder_to_vtk(arr[:, c], grid_shape) for c in range(3)]
                vals = np.column_stack(cols)
                f.write(f"VECTORS {name} float\n")
                np.savetxt(f, vals, fmt=fmt)
            else:
                raise ValueError(
                    f"field '{name}' must be (Ne,) scalar or (Ne,3) vector, "
                    f"got shape {arr.shape}")

    return filename


# ---------------------------------------------------------------------------
# High-level convenience: stress + strain (effective + components)
# ---------------------------------------------------------------------------

def save_stress_strain_vtk(filename, grid_shape, strain, stress,
                           extra=None, write_components=True,
                           origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0),
                           title="voxel FEM stress/strain"):
    """
    Save per-voxel effective strain & stress (and optionally all 6 components)
    to a structured-grid VTK file.

    Parameters
    ----------
    filename     : output .vtk path
    grid_shape   : (Nx, Ny, Nz) voxel counts
    strain       : (Ne, 6) Voigt strain per voxel  [xx,yy,zz,xy,yz,xz]
    stress       : (Ne, 6) Voigt stress per voxel
    extra        : optional dict of additional per-voxel scalar/vector fields
                   (e.g. {"youngs_modulus": E_elem})
    write_components : also write the 6 strain_* and stress_* component fields
    origin, spacing, title : passed through to the writer

    Returns
    -------
    fields : dict of every field written (name -> array)
    """
    strain = np.asarray(strain, dtype=OUTPUT_DTYPE)
    stress = np.asarray(stress, dtype=OUTPUT_DTYPE)

    fields = {
        "effective_strain": von_mises_strain(strain),
        "effective_stress": von_mises_stress(stress),
    }
    if write_components:
        comp = ["xx", "yy", "zz", "xy", "yz", "xz"]
        for c, name in enumerate(comp):
            fields[f"strain_{name}"] = strain[:, c]
            fields[f"stress_{name}"] = stress[:, c]
    if extra:
        fields.update(extra)

    write_structured_points_vtk(filename, grid_shape, fields,
                                origin=origin, spacing=spacing, title=title)
    return fields


# ---------------------------------------------------------------------------
# Voigt -> tensor conversions
# ---------------------------------------------------------------------------
# Our Voigt order is [xx, yy, zz, xy, yz, xz].  VTK's symmetric-tensor order is
# (00, 11, 22, 01, 12, 02) = (xx, yy, zz, xy, yz, xz) — identical, so the
# symmetric form needs no reordering.  STRAIN is stored with *engineering*
# shears (gamma); the strain *tensor* off-diagonals are gamma/2, so they are
# halved here.  STRESS off-diagonals are unchanged.

def voigt_to_sym6(t, is_strain):
    """(Ne,6) Voigt -> (Ne,6) VTK symmetric tensor [xx,yy,zz,xy,yz,xz]."""
    out = np.asarray(t, dtype=OUTPUT_DTYPE).copy()
    if is_strain:
        out[:, 3:] *= 0.5            # engineering -> tensor shear
    return out


def voigt_to_full9(t, is_strain):
    """(Ne,6) Voigt -> (Ne,9) row-major full 3x3 tensor (00,01,02,10,11,...)."""
    t = np.asarray(t, dtype=OUTPUT_DTYPE)
    xx, yy, zz, xy, yz, xz = (t[:, i] for i in range(6))
    if is_strain:
        xy, yz, xz = xy * 0.5, yz * 0.5, xz * 0.5
    T = np.empty((t.shape[0], 9), dtype=OUTPUT_DTYPE)
    T[:, 0], T[:, 1], T[:, 2] = xx, xy, xz
    T[:, 3], T[:, 4], T[:, 5] = xy, yy, yz
    T[:, 6], T[:, 7], T[:, 8] = xz, yz, zz
    return T


# ---------------------------------------------------------------------------
# XML ImageData (.vti) export via pyvista — binary + zlib-compressed, tensors
# ---------------------------------------------------------------------------

def save_stress_strain_vti(filename, grid_shape, strain, stress, extra=None,
                           tensor_components=6, compress=True,
                           origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    """
    Save per-voxel effective scalars + full strain/stress tensors to a binary,
    zlib-compressed XML VTK ImageData (.vti) file via pyvista.

    Unlike the legacy writer this emits the tensors in proper tensor notation
    (a single multi-component cell array each) and omits the 12 redundant
    per-component scalars.

    Parameters
    ----------
    filename     : output .vti path
    grid_shape   : (Nx, Ny, Nz) voxel counts
    strain, stress : (Ne, 6) Voigt arrays [xx,yy,zz,xy,yz,xz] in element order
    extra        : optional dict of extra per-voxel scalar fields
    tensor_components : 6 -> compact VTK symmetric tensor (default; uses tensor
                   shear gamma/2 for strain); 9 -> full 3x3 (needed by ParaView's
                   Tensor Glyph / eigenvalue filters)
    compress     : write zlib-compressed (default True)
    origin, spacing : ImageData geometry

    Returns
    -------
    grid : the pyvista.ImageData written
    """
    try:
        import pyvista as pv
    except ImportError as e:
        raise RuntimeError(
            "save_stress_strain_vti requires pyvista (pip install pyvista). "
            "Use save_stress_strain_vtk for dependency-free legacy output."
        ) from e

    Nx, Ny, Nz = grid_shape
    Ne = Nx * Ny * Nz
    strain = np.asarray(strain, dtype=OUTPUT_DTYPE)
    stress = np.asarray(stress, dtype=OUTPUT_DTYPE)

    # Element order is z-fastest (C-order over (Nx,Ny,Nz)); VTK ImageData cells
    # are x-fastest.  This permutation maps one to the other.
    perm = np.arange(Ne).reshape(Nx, Ny, Nz).ravel(order="F")

    grid = pv.ImageData(dimensions=(Nx + 1, Ny + 1, Nz + 1),
                        spacing=tuple(spacing), origin=tuple(origin))

    grid.cell_data["effective_strain"] = von_mises_strain(strain)[perm]
    grid.cell_data["effective_stress"] = von_mises_stress(stress)[perm]

    if tensor_components == 6:
        grid.cell_data["strain"] = voigt_to_sym6(strain, True)[perm]
        grid.cell_data["stress"] = voigt_to_sym6(stress, False)[perm]
    elif tensor_components == 9:
        grid.cell_data["strain"] = voigt_to_full9(strain, True)[perm]
        grid.cell_data["stress"] = voigt_to_full9(stress, False)[perm]
    else:
        raise ValueError("tensor_components must be 6 or 9")

    if extra:
        for name, arr in extra.items():
            arr = np.asarray(arr)
            if arr.dtype == np.float16:      # VTK has no half-precision type
                arr = arr.astype(np.float32)
            grid.cell_data[name] = arr[perm]

    # vtkXMLImageDataWriter: binary + zlib compression.
    grid.save(filename, binary=True,
              compression="zlib" if compress else None)
    return grid
