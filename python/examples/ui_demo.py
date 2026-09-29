"""In-window UI with Dear ImGui — sliders/buttons that drive the scene live.

    python ui_demo.py            # GLRenderer
    python ui_demo.py --vulkan   # VulkanRenderer (deferred RasterFirst)

A torus knot you can orbit, with an ImGui control panel: tweak the material,
toggle spin/wireframe, reset the view. Drag the 3D view to orbit; the panel
captures the mouse while you're over it. Needs a display.

The ImGui code is the same on both backends. On Vulkan the overlay is recorded
into the deferred frame after the scene, and a material edit is pushed to the
GPU with mat.needs_update().
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp
from demo_common import resize_handler

VULKAN = "--vulkan" in sys.argv

if VULKAN and not tp.HAS_VULKAN:
    print("This build has no Vulkan backend (configure -DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)

if VULKAN:
    canvas = tp.Canvas("threepp — Vulkan + ImGui", width=1000, height=700, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
else:
    canvas = tp.Canvas("threepp - ImGui UI", antialiasing=4)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
ui = tp.ImguiContext(canvas, renderer)  # create AFTER the renderer

scene = tp.Scene()
scene.background = 0x202830

camera = tp.PerspectiveCamera(60, canvas.aspect(), 0.1, 100)
camera.position.set(0, 2, 6)
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True

scene.add(tp.HemisphereLight(0xffffff, 0x333344, 1.0))
key = tp.DirectionalLight(0xffffff, 3.0 if VULKAN else 2.5)
key.position.set(5, 10, 7)
key.cast_shadow = not VULKAN
scene.add(key)

mat = tp.MeshStandardMaterial()
mat.color = 0xff8800
mat.roughness = 0.4
mat.metalness = 0.1
knot = tp.Mesh(tp.TorusKnotGeometry(0.7, 0.25, 128, 64), mat)
knot.cast_shadow = not VULKAN
scene.add(knot)

ground = tp.Mesh(tp.PlaneGeometry(40, 40), tp.MeshStandardMaterial())
ground.position.y = -1.6
ground.rotate_x(-math.pi / 2)
ground.receive_shadow = not VULKAN
scene.add(ground)

canvas.on_window_resize(resize_handler(camera, renderer))

state = {
    "roughness": 0.4, "metalness": 0.1, "color": (1.0, 0.53, 0.0),
    "wireframe": False, "spin": True, "speed": 0.6,
}
clock = tp.Clock()


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(290, 0)
    tp.imgui.begin("Material & Scene (Vulkan)" if VULKAN else "Material & Scene")

    changed = False
    ch, state["roughness"] = tp.imgui.slider_float("roughness", state["roughness"], 0.0, 1.0)
    changed |= ch; mat.roughness = state["roughness"]
    ch, state["metalness"] = tp.imgui.slider_float("metalness", state["metalness"], 0.0, 1.0)
    changed |= ch; mat.metalness = state["metalness"]
    ch, state["color"] = tp.imgui.color_edit3("color", state["color"])
    changed |= ch; mat.color = tp.Color(*state["color"])
    if not VULKAN:
        _, state["wireframe"] = tp.imgui.checkbox("wireframe", state["wireframe"])
        mat.wireframe = state["wireframe"]
    if VULKAN and changed:
        mat.needs_update()   # the deferred renderer keeps materials in a GPU buffer

    tp.imgui.separator()
    _, state["spin"] = tp.imgui.checkbox("spin", state["spin"])
    _, state["speed"] = tp.imgui.slider_float("speed", state["speed"], 0.0, 3.0)
    if tp.imgui.button("reset view"):
        camera.position.set(0, 2, 6)
        controls.target = tp.Vector3(0, 0, 0)

    tp.imgui.separator()
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
    tp.imgui.end()


def animate():
    dt = clock.get_delta()
    if state["spin"]:
        knot.rotation.y += dt * state["speed"]
    controls.enabled = not ui.want_capture_mouse   # don't orbit while using the panel
    controls.update()
    renderer.render(scene, camera)
    ui.render(draw_ui)                              # draw the UI on top


canvas.animate(animate)
