"""Point-cloud tools: voxel grids, scalar fields and marching cubes."""
import numpy as np

import threepp as tp


def test_scalar_field_from_numpy_round_trip():
    z, y, x = np.mgrid[0:20, 0:24, 0:28]
    data = (x + 100 * y + 10000 * z).astype(np.float32)
    f = tp.ScalarField.from_numpy(data, tp.Vector3(1, 2, 3), 0.5)
    assert (f.nx, f.ny, f.nz) == (28, 24, 20)
    assert f.cell_size == 0.5
    assert f.at(3, 2, 1) == data[1, 2, 3]                   # (x, y, z) indexes [z, y, x]
    assert np.array_equal(f.data_numpy(), data)


def test_marching_cubes_on_a_field_from_numpy():
    # a signed distance field of a sphere (inside positive), meshed at level 0
    n, cell, r = 24, 0.01, 0.07
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    c = (n - 1) / 2
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) * cell - r
    f = tp.ScalarField.from_numpy(-d.astype(np.float32), tp.Vector3(0, 0, 0), cell)
    iso = tp.marching_cubes(f, 0.0)
    V = np.asarray(iso.positions).reshape(-1, 3)
    radius = np.linalg.norm(V - c * cell, axis=1)
    assert len(V) > 100
    assert abs(radius.mean() - r) < 0.002
    N = np.asarray(iso.normals).reshape(-1, 3)
    assert np.mean(np.sum(N * (V - c * cell), axis=1) > 0) > 0.99   # normals point out


def test_from_numpy_rejects_bad_input():
    import pytest
    with pytest.raises(RuntimeError):
        tp.ScalarField.from_numpy(np.zeros((4, 4), np.float32), tp.Vector3(), 1.0)
    with pytest.raises(RuntimeError):
        tp.ScalarField.from_numpy(np.zeros((2, 2, 2), np.float32), tp.Vector3(), 0.0)
