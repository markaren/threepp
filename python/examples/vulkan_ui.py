"""In-window Dear ImGui UI over the Vulkan deferred renderer.

    python vulkan_ui.py

Same as ui_demo.py but on the Vulkan (RasterFirst) backend — the ImGui overlay
is recorded into the deferred frame after the scene. Needs a Vulkan build and a
display.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

if not tp.HAS_VULKAN:
    print("This build has no Vulkan backend (configure -DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)

canvas = tp.Canvas("threepp — Vulkan + ImGui", width=1000, height=700, vsync=False)
renderer = tp.VulkanRenderer(canvas)
ui = tp.ImguiContext(canvas, renderer)   # Vulkan backend; create after the renderer

scene = tp.Scene()
scene.background = 0x202830

camera = tp.PerspectiveCamera(60, canvas.aspect(), 0.1, 100)
camera.position.set(0, 2, 6)
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True

scene.add(tp.HemisphereLight(0xffffff, 0x333344, 1.0))
key = tp.DirectionalLight(0xffffff, 3.0)
key.position.set(5, 10, 7)
scene.add(key)

mat = tp.MeshStandardMaterial()
mat.color = 0xff8800
mat.roughness = 0.4
mat.metalness = 0.2
knot = tp.Mesh(tp.TorusKnotGeometry(0.7, 0.25, 128, 64), mat)
scene.add(knot)

ground = tp.Mesh(tp.PlaneGeometry(40, 40), tp.MeshStandardMaterial())
ground.position.y = -1.6
ground.rotate_x(-math.pi / 2)
scene.add(ground)

def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

state = {"roughness": 0.4, "metalness": 0.2, "color": (1.0, 0.53, 0.0), "spin": True, "speed": 0.6}
clock = tp.Clock()


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(290, 0)
    tp.imgui.begin("Material & Scene (Vulkan)")
    changed = False
    ch, state["roughness"] = tp.imgui.slider_float("roughness", state["roughness"], 0.0, 1.0)
    changed |= ch; mat.roughness = state["roughness"]
    ch, state["metalness"] = tp.imgui.slider_float("metalness", state["metalness"], 0.0, 1.0)
    changed |= ch; mat.metalness = state["metalness"]
    ch, state["color"] = tp.imgui.color_edit3("color", state["color"])
    changed |= ch; mat.color = tp.Color(*state["color"])
    if changed:
        mat.needs_update()
    tp.imgui.separator()
    _, state["spin"] = tp.imgui.checkbox("spin", state["spin"])
    _, state["speed"] = tp.imgui.slider_float("speed", state["speed"], 0.0, 3.0)
    if tp.imgui.button("reset view"):
        camera.position.set(0, 2, 6)
        controls.target = tp.Vector3(0, 0, 0)
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
    tp.imgui.end()


def animate():
    dt = clock.get_delta()
    if state["spin"]:
        knot.rotation.y += dt * state["speed"]
    controls.enabled = not ui.want_capture_mouse
    controls.update()
    renderer.render(scene, camera)
    ui.render(draw_ui)


canvas.animate(animate)
