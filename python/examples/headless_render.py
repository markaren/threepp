"""Render a scene off-screen, straight into a numpy array — no window.

This is the path for ML data generation, robotics cameras and machine vision:
build a scene in Python, render headless, get pixels as a (H, W, 3) uint8 array.

    python headless_render.py

Writes render.png (via threepp) and, if Pillow is installed, render_numpy.png
(via the numpy array) so you can confirm they match.
"""
import os
import sys

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

WIDTH, HEIGHT = 640, 480

canvas = tp.Canvas("offscreen", width=WIDTH, height=HEIGHT, headless=True)
renderer = tp.GLRenderer(canvas)
renderer.set_clear_color(0x101820)

scene = tp.Scene()

camera = tp.PerspectiveCamera(55, WIDTH / HEIGHT, 0.1, 100)
camera.position.set(3.0, 2.0, 4.0)
camera.look_at(0, 0, 0)

scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.0))
sun = tp.DirectionalLight(0xffffff, 2.5)
sun.position.set(5, 8, 5)
scene.add(sun)

# A shiny torus knot on a matte ground plane.
knot_mat = tp.MeshStandardMaterial()
knot_mat.color = 0xff8800
knot_mat.metalness = 0.35
knot_mat.roughness = 0.35
scene.add(tp.Mesh(tp.TorusKnotGeometry(0.7, 0.24), knot_mat))

ground_mat = tp.MeshStandardMaterial()
ground_mat.color = 0x555560
ground = tp.Mesh(tp.PlaneGeometry(20, 20), ground_mat)
ground.position.y = -1.5
ground.rotate_x(-3.14159 / 2)
scene.add(ground)

renderer.render(scene, camera)

img = renderer.read_pixels()  # (HEIGHT, WIDTH, 3) uint8, top-down
print(f"rendered numpy image: shape={img.shape} dtype={img.dtype} "
      f"min={int(img.min())} max={int(img.max())}")

renderer.save_frame("render.png")
print("wrote render.png")

try:
    from PIL import Image

    Image.fromarray(img).save("render_numpy.png")
    print("wrote render_numpy.png (from the numpy array)")
except ImportError:
    print("(install Pillow to also save the numpy array as an image)")
