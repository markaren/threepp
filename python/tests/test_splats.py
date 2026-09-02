"""SplatLoader + SplatCloud — 3D Gaussian Splatting from Python.

No asset files: the tests struct.pack a minimal degree-0 binary PLY the way
SplatLoader_test hand-builds buffers in C++, then render it headless on GL.
Conventions pinned by the loader (and exercised here): opacity is stored
pre-sigmoid, scale_* pre-exp, rot_* w-first.
"""
import math
import struct

import pytest

import threepp as tp


def _splat_ply(path, splats):
    """Write a minimal 3DGS binary PLY: position, f_dc rgb, opacity, scale, rot."""
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {len(splats)}"]
        + [f"property float {p}" for p in props] + ["end_header", ""])
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for s in splats:
            f.write(struct.pack("<14f", *s))


def _white_splat(x=0.0, y=0.0, z=0.0, scale=0.0):
    # f_dc ~1.77 -> SH degree-0 colour ~1.0; opacity 9 (pre-sigmoid) -> ~1.0;
    # scale is pre-exp, so 0.0 -> 1 m; rot is w-first identity.
    dc = (1.0 - 0.5) / 0.28209479177  # inverse of c = 0.5 + SH_C0 * f_dc
    return (x, y, z, dc, dc, dc, 9.0, scale, scale, scale, 1.0, 0.0, 0.0, 0.0)


def test_is_splat_ply_discriminates(tmp_path):
    ply = tmp_path / "cloud.ply"
    _splat_ply(ply, [_white_splat()])
    assert tp.SplatLoader.is_splat_ply(ply)
    assert not tp.SplatLoader.is_splat_ply(tmp_path / "missing.ply")  # never raises


def test_load_ply_counts_and_consuming_create(tmp_path):
    ply = tmp_path / "cloud.ply"
    _splat_ply(ply, [_white_splat(-1), _white_splat(+1)])
    data = tp.SplatLoader.load_ply(ply)
    assert data.count == 2 and len(data) == 2
    cloud = tp.SplatCloud(data)
    assert cloud.splat_count == 2
    assert data.count == 0, "create() documents that it CONSUMES the SplatData"
    assert cloud.cpu_bytes > 0


def test_splat_cloud_renders_on_gl(renderer, tmp_path):
    ply = tmp_path / "cloud.ply"
    _splat_ply(ply, [_white_splat()])
    cloud = tp.SplatCloud(tp.SplatLoader.load_ply(ply))

    scene = tp.Scene()  # splats carry their own colour; no lights needed
    scene.add(cloud)
    camera = tp.PerspectiveCamera(60, 200 / 150, 0.1, 100)
    camera.position.set(0, 0, 4)
    camera.look_at(0, 0, 0)

    renderer.set_clear_color(0x000000)
    cloud.update(camera)  # explicit: the pre-render hook sorts one frame late
    renderer.render(scene, camera)
    img = renderer.read_pixels()
    renderer.set_clear_color(0x202830)  # restore session default

    assert int(img.max()) > 32, "a unit white splat at the origin left no pixels"


def test_submit_ranges_roundtrip(tmp_path):
    ply = tmp_path / "cloud.ply"
    _splat_ply(ply, [_white_splat(float(i)) for i in range(4)])
    cloud = tp.SplatCloud(tp.SplatLoader.load_ply(ply))
    assert cloud.submit_ranges == []
    cloud.submit_ranges = [(0, 2), (3, 1)]
    assert cloud.submit_ranges == [(0, 2), (3, 1)]


def _point_cloud_ply(path, points):
    """Write a colour-only binary PLY: x y z as float, red green blue as uchar."""
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}",
         "property float x", "property float y", "property float z",
         "property uchar red", "property uchar green", "property uchar blue",
         "end_header", ""])
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for x, y, z, r, g, b in points:
            f.write(struct.pack("<3f3B", x, y, z, r, g, b))


def test_point_cloud_ply_loads_as_splats(tmp_path):
    ply = tmp_path / "scan.ply"
    grid = [(x * 0.5, y * 0.5, 0.0, 255, 0, 0) for x in range(4) for y in range(4)]
    _point_cloud_ply(ply, grid)
    assert tp.SplatLoader.is_point_cloud_ply(ply)
    assert not tp.SplatLoader.is_splat_ply(ply)
    data = tp.SplatLoader.load_point_cloud_ply(ply)
    assert data.count == 16
    cloud = tp.SplatCloud(data)
    assert cloud.splat_count == 16


def test_point_mode_roundtrip(tmp_path):
    ply = tmp_path / "cloud.ply"
    _splat_ply(ply, [_white_splat()])
    cloud = tp.SplatCloud(tp.SplatLoader.load_ply(ply))
    assert cloud.point_mix == 0.0 and cloud.point_size == 2.0
    cloud.point_mix = 1.0
    cloud.point_size = 5.0
    assert cloud.point_mix == 1.0 and cloud.point_size == 5.0
    cloud.point_mix = 7.0  # clamped
    cloud.point_size = 0.0  # floored
    assert cloud.point_mix == 1.0 and cloud.point_size == 1.0


def test_write_ply_round_trips(tmp_path):
    ply = tmp_path / "in.ply"
    _splat_ply(ply, [_white_splat(-1), _white_splat(+1, scale=-2.0)])
    data = tp.SplatLoader.load_ply(ply)
    out = tmp_path / "out.ply"
    tp.SplatLoader.write_ply(data, out)
    assert tp.SplatLoader.is_splat_ply(out)
    assert tp.SplatLoader.load_ply(out).count == 2
    with pytest.raises(RuntimeError):
        tp.SplatLoader.write_ply(data, tmp_path / "missing_dir" / "x.ply")
