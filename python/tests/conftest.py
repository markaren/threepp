"""Shared pytest fixtures for the threepp binding tests.

Run from the python/ directory:

    pip install pytest numpy pillow
    pytest

Requires the built `threepp` module in python/ (build the threepp_py target).
"""
import gc
import os
import struct
import sys

import pytest


@pytest.fixture(autouse=True)
def _release_physx_between_tests():
    """PhysX allows only one foundation per process, so a lingering PhysxWorld from
    one test makes the next test's world fail to construct. Force a collection after
    every test so the previous world is released regardless of GC timing."""
    yield
    gc.collect()

# Make the built `threepp` module (in python/, the parent of tests/) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp  # noqa: E402


@pytest.fixture(scope="session")
def renderer():
    """One shared headless GL context for the whole session.

    There is a single GL context per process, so all rendering tests share this
    200x150 off-screen renderer rather than creating their own.
    """
    canvas = tp.Canvas("pytest", width=200, height=150, headless=True)
    r = tp.GLRenderer(canvas)
    r.set_clear_color(0x202830)
    yield r


@pytest.fixture
def lit_scene():
    """A scene with hemisphere + directional lighting, ready for content."""
    scene = tp.Scene()
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    key = tp.DirectionalLight(0xffffff, 2.0)
    key.position.set(3, 5, 2)
    scene.add(key)
    return scene


# Unit cube as 12 triangles, shared by the loader fixtures.
_CUBE_V = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
           (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
_CUBE_TRIS = [((0, 1, 3), (-1, 0, 0)), ((0, 3, 2), (-1, 0, 0)),
              ((4, 6, 7), (1, 0, 0)), ((4, 7, 5), (1, 0, 0)),
              ((0, 4, 5), (0, -1, 0)), ((0, 5, 1), (0, -1, 0)),
              ((2, 3, 7), (0, 1, 0)), ((2, 7, 6), (0, 1, 0)),
              ((0, 2, 6), (0, 0, -1)), ((0, 6, 4), (0, 0, -1)),
              ((1, 5, 7), (0, 0, 1)), ((1, 7, 3), (0, 0, 1))]


@pytest.fixture
def stl_cube(tmp_path):
    """Path to a small binary-STL cube (threepp's STLLoader reads binary)."""
    path = tmp_path / "cube.stl"
    with open(path, "wb") as f:
        f.write(b"\x00" * 80 + struct.pack("<I", len(_CUBE_TRIS)))
        for (a, b, c), n in _CUBE_TRIS:
            f.write(struct.pack("<3f", *n))
            for i in (a, b, c):
                f.write(struct.pack("<3f", *_CUBE_V[i]))
            f.write(struct.pack("<H", 0))
    return str(path)


@pytest.fixture
def obj_cube(tmp_path):
    """Path to a cube OBJ in v//vn form (threepp's OBJLoader needs the normal slot)."""
    path = tmp_path / "cube.obj"
    path.write_text(
        "v -1 -1 -1\nv -1 -1 1\nv -1 1 -1\nv -1 1 1\n"
        "v 1 -1 -1\nv 1 -1 1\nv 1 1 -1\nv 1 1 1\n"
        "vn -1 0 0\nvn 1 0 0\nvn 0 -1 0\nvn 0 1 0\nvn 0 0 -1\nvn 0 0 1\n"
        "f 1//1 2//1 4//1\nf 1//1 4//1 3//1\n"
        "f 5//2 7//2 8//2\nf 5//2 8//2 6//2\n"
        "f 1//3 5//3 6//3\nf 1//3 6//3 2//3\n"
        "f 3//4 4//4 8//4\nf 3//4 8//4 7//4\n"
        "f 1//5 3//5 7//5\nf 1//5 7//5 5//5\n"
        "f 2//6 6//6 8//6\nf 2//6 8//6 4//6\n"
    )
    return str(path)


@pytest.fixture
def hdr_env(tmp_path):
    """Path to a tiny valid Radiance .hdr (flat RGBE, 4x2, ~0.5 grey).

    stb_image (used by RGBELoader) reads scanlines with width<8 uncompressed,
    so no RLE encoding is needed — RGBE byte (128,128,128,128) decodes to ~0.5
    linear. 2:1 to look like an equirect map.
    """
    path = tmp_path / "env.hdr"
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n"
    body = bytes([128, 128, 128, 128]) * (4 * 2)
    path.write_bytes(header + body)
    return str(path)


@pytest.fixture
def checker_texture(tmp_path):
    """Path to a 256x256 (power-of-two) checkerboard PNG. Skips if Pillow is absent."""
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "checker.png"
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(256):
        for x in range(256):
            px[x, y] = (235, 90, 40) if ((x // 32) + (y // 32)) % 2 else (40, 110, 235)
    img.save(path)
    return str(path)
