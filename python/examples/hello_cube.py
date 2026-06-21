"""The three.js 'hello world' — spinning, lit cubes you can orbit — in Python.

    python hello_cube.py

Opens a window. Drag to orbit, scroll to zoom, Esc to quit.
"""
import os
import sys

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

canvas = tp.Canvas("threepp — hello cube", width=900, height=600, antialiasing=4)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True

scene = tp.Scene()
scene.background = 0x202830

camera = tp.PerspectiveCamera(70, canvas.aspect(), 0.1, 100)
camera.position.set(0, 2, 6)

controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True

scene.add(tp.HemisphereLight(0xffffff, 0x333344, 1.0))
key_light = tp.DirectionalLight(0xffffff, 2.5)
key_light.position.set(5, 10, 7)
key_light.cast_shadow = True
scene.add(key_light)

cubes = tp.Group()
for i, color in enumerate((0xff5555, 0x55ff66, 0x5599ff)):
    material = tp.MeshStandardMaterial()
    material.color = color          # hex int converts straight to a Color
    material.roughness = 0.35
    material.metalness = 0.1
    cube = tp.Mesh(tp.BoxGeometry(), material)
    cube.position.x = (i - 1) * 1.7
    cube.cast_shadow = True
    cubes.add(cube)
scene.add(cubes)

clock = tp.Clock()


def animate():
    t = clock.get_elapsed_time()
    cubes.rotation.y = t * 0.5
    for i, cube in enumerate(cubes.children):
        cube.rotation.x = t * (1.0 + 0.3 * i)
    controls.update()
    renderer.render(scene, camera)


canvas.animate(animate)
