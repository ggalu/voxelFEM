#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example for visualization of microstructures with pyvista
Requires a pyvista installation in the knpy envirnment, e.g. with
(knpy) $ conda install -c conda-forge pyvista



@author: Alexander Hartmaier, ICAMS/Ruhr University Germany
August 2022
"""
import numpy as np
import kanapy as knpy
import pyvista as pv
from vtk import VTK_HEXAHEDRON
import sys

Nv = 20
size = 20
periodic = False
if periodic:
    name = 'pbc_px'
else:
    name = 'px'
matname = 'Simulanium fcc'
ms_elong = {'Grain type': 'Elongated', 
          'Equivalent diameter': {'sig': 1.0, 'scale': 2.0, 'loc': 6.0, 'cutoff_min': 5.0, 'cutoff_max': 10.0},
          'Aspect ratio': {'sig': 1.5, 'scale': 1.7, 'loc': 0.8, 'cutoff_min': 1.0, 'cutoff_max': 1.41},
          'Tilt angle': {'kappa': 15., 'loc': 90., "cutoff_min": 0.0, "cutoff_max": 180.0},
          'RVE': {'sideX': size, 'sideY': size, 'sideZ': size, 'Nx': Nv, 'Ny': Nv, 'Nz': Nv}, 
          'Simulation': {'periodicity': str(periodic), 'output_units': 'um'},
          'Phase': {'Name': 'Simulanium fcc', 'Number': 1, 'Volume fraction': 1.0}}

ms = knpy.Microstructure(ms_elong)
ms.init_stats()



ms.init_RVE()
ms.pack()
ms.plot_ellipsoids()
sys.exit()
ms.voxelize()
ms.plot_voxels()

# Save the voxelized RVE as a plain numpy array of grain labels.
# ms.mesh.grains is an (Nx, Ny, Nz) int array of grain IDs (1-based;
# 0 marks unassigned/void voxels). We only need the grain label as the
# identifier, stored as uint8.
grains = ms.mesh.grains
gmax = int(grains.max())
if gmax > 255:
    raise ValueError(
        f"RVE has {gmax} grain labels, which exceeds the uint8 range (0-255). "
        "Reduce the number of grains or use a wider dtype."
    )
grains_u8 = grains.astype(np.uint8)
np.save('rve_grains.npy', grains_u8)
print(f"Saved RVE grain labels to rve_grains.npy "
      f"(shape {grains_u8.shape}, dtype {grains_u8.dtype}, "
      f"{gmax} grains).")
