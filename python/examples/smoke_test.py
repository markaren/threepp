"""Regression smoke test for the threepp Python bindings.

Exercises the high-impact API end-to-end and asserts behaviour, including the
load-bearing 'mutate in place like three.js' contract and a headless render.

    python smoke_test.py    ->    prints 'ALL OK' and exits 0 on success
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

# ---- math value types --------------------------------------------------------
assert tp.Vector3(0, 3, 4).length() == 5
assert (tp.Vector3(1, 0, 0) + tp.Vector3(0, 1, 0)) == tp.Vector3(1, 1, 0)
assert tp.Color(0xFF8000).r == 1.0 and tp.Color(0xFF8000).b == 0.0
assert tp.Color("#00ff00").get_hex() == 0x00FF00

# ---- mutate-in-place (the core three.js promise) -----------------------------
mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
mesh.position.x = 0.5
assert mesh.position.x == 0.5, "position must be a live reference, not a copy"
mesh.rotation.y = 1.0
assert abs(mesh.rotation.y - 1.0) < 1e-6

# ---- hex-int implicitly becomes a Color on assignment ------------------------
mesh.material.color = 0x00FF00
assert mesh.material.color == tp.Color(0x00FF00)
mesh.material.roughness = 0.4
assert abs(mesh.material.roughness - 0.4) < 1e-6
mesh.material.name = "green"  # std::string on the (virtual) Material base
assert mesh.material.name == "green"

# ---- scene graph: add / children / traverse / lookup -------------------------
scene = tp.Scene()
group = tp.Group()
sphere = tp.Mesh(tp.SphereGeometry(0.5), tp.MeshStandardMaterial())
sphere.name = "ball"
group.add(sphere)
scene.add(mesh, group)
assert len(scene.children) == 2
seen = []
scene.traverse(lambda o: seen.append(type(o).__name__))
assert seen.count("Mesh") == 2 and "Group" in seen
assert scene.get_object_by_name("ball") is not None

# ---- model loading (dependency-free binary STL) ------------------------------
stl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_cube.stl")
_v = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
      (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
_tris = [((0, 1, 3), (-1, 0, 0)), ((0, 3, 2), (-1, 0, 0)), ((4, 6, 7), (1, 0, 0)),
         ((4, 7, 5), (1, 0, 0)), ((0, 4, 5), (0, -1, 0)), ((0, 5, 1), (0, -1, 0)),
         ((2, 3, 7), (0, 1, 0)), ((2, 7, 6), (0, 1, 0)), ((0, 2, 6), (0, 0, -1)),
         ((0, 6, 4), (0, 0, -1)), ((1, 5, 7), (0, 0, 1)), ((1, 7, 3), (0, 0, 1))]
with open(stl_path, "wb") as _f:
    _f.write(b"\x00" * 80 + struct.pack("<I", len(_tris)))
    for (a, b, c), nrm in _tris:
        _f.write(struct.pack("<3f", *nrm))
        for _i in (a, b, c):
            _f.write(struct.pack("<3f", *_v[_i]))
        _f.write(struct.pack("<H", 0))
loaded = tp.ModelLoader().load(stl_path)
assert sum(1 for _ in [c for c in loaded.children]) >= 1, "model loaded no children"
bbox = tp.Box3().set_from_object(loaded)
assert not bbox.is_empty()
os.remove(stl_path)

# ---- headless render to numpy ------------------------------------------------
canvas = tp.Canvas("smoke", width=320, height=240, headless=True)
renderer = tp.GLRenderer(canvas)
renderer.set_clear_color(0x202830)

camera = tp.PerspectiveCamera(60, canvas.aspect(), 0.1, 100)
camera.position.set(0, 2, 4)
camera.look_at(0, 0, 0)

scene.background = 0x202830
scene.add(tp.HemisphereLight(0xffffff, 0x444444, 1.0))
key = tp.DirectionalLight(0xffffff, 2.0)
key.position.set(3, 5, 2)
scene.add(key)

renderer.render(scene, camera)
img = renderer.read_pixels()
assert img.shape == (240, 320, 3) and str(img.dtype) == "uint8"
assert int(img.max()) > int(img.min()), "image is flat — nothing drew"

print("ALL OK", img.shape)
