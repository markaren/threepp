"""Load an image as a texture and map it onto a mesh — headless.

Generates a checkerboard PNG (needs Pillow), loads it with TextureLoader, and
renders a textured box off-screen to textured_box.png.

    python textured_box.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))

# A 256x256 (power-of-two) checkerboard. GL needs power-of-two textures.
tex_path = os.path.join(here, "checker.png")
img = Image.new("RGB", (256, 256))
px = img.load()
for y in range(256):
    for x in range(256):
        px[x, y] = (235, 90, 40) if ((x // 32) + (y // 32)) % 2 else (40, 110, 235)
img.save(tex_path)

canvas = tp.Canvas("textured", width=600, height=450, headless=True)
renderer = tp.GLRenderer(canvas)
renderer.set_clear_color(0x202830)

camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 100)
camera.position.set(2.2, 1.8, 3.0)
camera.look_at(0, 0, 0)

scene = tp.Scene()
scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.2))
sun = tp.DirectionalLight(0xffffff, 1.5)
sun.position.set(3, 6, 4)
scene.add(sun)

# Diffuse/albedo maps want SRGB so the sampler decodes to linear.
texture = tp.TextureLoader().load(tex_path, tp.ColorSpace.SRGB)

material = tp.MeshStandardMaterial()
material.map = texture
material.roughness = 0.6
scene.add(tp.Mesh(tp.BoxGeometry(1.6, 1.6, 1.6), material))

renderer.render(scene, camera)
renderer.save_frame(os.path.join(here, "textured_box.png"))
print("wrote textured_box.png")
