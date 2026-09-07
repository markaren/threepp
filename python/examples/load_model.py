"""Load a 3D model and render it headless.

    python load_model.py path/to/model.glb     # .obj / .gltf / .glb / .stl / .dae

ModelLoader dispatches by file extension to threepp's first-party loaders (no
Assimp needed). The model is auto-framed from its bounding box. With no argument
a small built-in cube is generated and loaded, to show the path end-to-end.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

here = os.path.dirname(os.path.abspath(__file__))


def make_demo_stl():
    """Write a small binary-STL cube and return its path."""
    v = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
         (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    tris = [((0, 1, 3), (-1, 0, 0)), ((0, 3, 2), (-1, 0, 0)),
            ((4, 6, 7), (1, 0, 0)), ((4, 7, 5), (1, 0, 0)),
            ((0, 4, 5), (0, -1, 0)), ((0, 5, 1), (0, -1, 0)),
            ((2, 3, 7), (0, 1, 0)), ((2, 7, 6), (0, 1, 0)),
            ((0, 2, 6), (0, 0, -1)), ((0, 6, 4), (0, 0, -1)),
            ((1, 5, 7), (0, 0, 1)), ((1, 7, 3), (0, 0, 1))]
    path = os.path.join(here, "demo_cube.stl")
    with open(path, "wb") as f:
        f.write(b"\x00" * 80 + struct.pack("<I", len(tris)))
        for (a, b, c), n in tris:
            f.write(struct.pack("<3f", *n))
            for i in (a, b, c):
                f.write(struct.pack("<3f", *v[i]))
            f.write(struct.pack("<H", 0))
    return path


model_path = sys.argv[1] if len(sys.argv) > 1 else make_demo_stl()
print(f"loading {model_path}")

model = tp.ModelLoader().load(model_path)

# Frame the model from its world-space bounding box.
bbox = tp.Box3().set_from_object(model)
center = bbox.get_center()
size = bbox.get_size()
radius = max(size.x, size.y, size.z) or 1.0

scene = tp.Scene()
scene.background = 0x202830
scene.add(model)
scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.2))
sun = tp.DirectionalLight(0xffffff, 1.8)
sun.position.set(1, 2, 1.5)
scene.add(sun)

canvas = tp.Canvas("model", width=720, height=540, headless=True)
renderer = tp.GLRenderer(canvas)

camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.01, 100 * radius)
camera.position.set(center.x + radius * 2, center.y + radius * 1.5, center.z + radius * 2.5)
camera.look_at(center.x, center.y, center.z)

renderer.render(scene, camera)
out = os.path.join(here, "model_render.png")
renderer.save_frame(out)
print(f"wrote {out}")
