"""In-window UI with Dear ImGui — sliders/buttons that drive the scene live.

    python ui_demo.py

A torus knot you can orbit, with an ImGui control panel: tweak the material,
toggle spin/wireframe, reset the view. Drag the 3D view to orbit; the panel
captures the mouse while you're over it. Needs a display.

ImGui uses the GL backend, so this pairs with GLRenderer (not VulkanRenderer).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

canvas = tp.Canvas("threepp - ImGui UI", antialiasing=4)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True
ui = tp.ImguiContext(canvas)        # create AFTER the GLRenderer

scene = tp.Scene()
scene.background = 0x202830

camera = tp.PerspectiveCamera(60, canvas.aspect(), 0.1, 100)
camera.position.set(0, 2, 6)
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True

scene.add(tp.HemisphereLight(0xffffff, 0x333344, 1.0))
key = tp.DirectionalLight(0xffffff, 2.5)
key.position.set(5, 10, 7)
key.cast_shadow = True
scene.add(key)

mat = tp.MeshStandardMaterial()
mat.color = 0xff8800
mat.roughness = 0.4
mat.metalness = 0.1
knot = tp.Mesh(tp.TorusKnotGeometry(0.7, 0.25), mat)
knot.cast_shadow = True
scene.add(knot)

ground = tp.Mesh(tp.PlaneGeometry(40, 40), tp.MeshStandardMaterial())
ground.position.y = -1.6
ground.rotate_x(-math.pi / 2)
ground.receive_shadow = True
scene.add(ground)

def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)

state = {
    "roughness": 0.4, "metalness": 0.1, "color": (1.0, 0.53, 0.0),
    "wireframe": False, "spin": True, "speed": 0.6, "demo": False,
}
clock = tp.Clock()


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(290, 0)
    tp.imgui.begin("Material & Scene")

    _, state["roughness"] = tp.imgui.slider_float("roughness", state["roughness"], 0.0, 1.0)
    mat.roughness = state["roughness"]
    _, state["metalness"] = tp.imgui.slider_float("metalness", state["metalness"], 0.0, 1.0)
    mat.metalness = state["metalness"]
    _, state["color"] = tp.imgui.color_edit3("color", state["color"])
    mat.color = tp.Color(*state["color"])
    _, state["wireframe"] = tp.imgui.checkbox("wireframe", state["wireframe"])
    mat.wireframe = state["wireframe"]

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
