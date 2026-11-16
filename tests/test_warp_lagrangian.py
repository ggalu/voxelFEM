# -*- coding: utf-8 -*-
# @Author: Lokesh Metta, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-09-25
"""Checks of warp_cupy.warp_lagrangian against exact answers (GPU required).

    python tests/test_warp_lagrangian.py      (or: pytest tests/test_warp_lagrangian.py)

1. A rigid shift t: the deformed image must equal image(y - t).
2. A uniform 5 % stretch along axis 0 (u_x = 0.05 x): the inverse map is
   X = y / 1.05 exactly.
Both on a random 64^3 volume, cubic B-spline sampling. On 2026-09-25 the
function also matched voxelDVC 0.2.0's make_deformed_image on a 201^3 FEM
field at 5 % strain to 3.7e-8 of the grey-value range.
"""
import os
import sys

import cupy as cp
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from warp_cupy import warp_lagrangian  # noqa: E402

N = 64
IMG = np.random.default_rng(0).normal(size=(N, N, N))
Y = np.stack(np.meshgrid(*[np.arange(N, dtype=float)] * 3, indexing="ij")).reshape(3, -1)


def _exact(X):
    return map_coordinates(IMG, X, order=3, mode="nearest").reshape(IMG.shape)


def test_rigid_shift():
    t = np.array([0.3, -1.7, 2.25])
    u = np.broadcast_to(t, (N + 1, N + 1, N + 1, 3)).copy()
    g, mismatch = warp_lagrangian(IMG, u, order=3)
    assert mismatch < 1e-10
    assert np.abs(cp.asnumpy(g) - _exact(Y - t[:, None])).max() < 1e-10


def test_uniform_stretch():
    e = 0.05
    u = np.zeros((N + 1, N + 1, N + 1, 3))
    u[..., 0] = e * np.arange(N + 1)[:, None, None]
    g, mismatch = warp_lagrangian(IMG, u, order=3)
    X = Y.copy()
    X[0] = Y[0] / (1 + e)
    assert mismatch < 1e-6
    assert np.abs(cp.asnumpy(g) - _exact(X)).max() < 1e-5


if __name__ == "__main__":
    test_rigid_shift()
    test_uniform_stretch()
    print("warp_lagrangian: both checks passed")
