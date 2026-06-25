"""DepthSensor showcase — a sweeping GPU depth camera that produces a live, colored point cloud.

A `tp.DepthSensor` sits in the middle of a ring of colored props and slowly sweeps in yaw (and gently
in pitch). Every frame it renders the scene from its own viewpoint, reads the depth (and color) back,
and reprojects to a world-space point cloud which is drawn with per-point colors. Orbit with the mouse.

    python depth_sensor.py                 # interactive window
    python depth_sensor.py --shot out.png  # headless: one scan, saved to a PNG

Controls: drag to orbit, scroll to zoom. ImGui panel (if built): range-noise, point size, RGB-D vs
distance coloring, and sweep on/off.

Showcases the Python bindings added for DepthSensor.scan / scan_rgbd (numpy point cloud + colors) and
BufferGeometry.set_attribute / update_attribute / set_draw_range (a fixed-capacity dynamic point cloud).
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threepp as tp

SENSOR_W, SENSOR_H = 160, 120        # depth-image resolution -> up to W*H points per scan
MAXPTS = SENSOR_W * SENSOR_H
FAR = 18.0


def build_scene():
    scene = tp.Scene()
    scene.background = 0x141a22
    scene.add(tp.HemisphereLight(0xdfeaff, 0x404a55, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.6)
    sun.position.set(6, 12, 8)
    sun.cast_shadow = True
    scene.add(sun)

    # dark ground so the colored props (and their colored points) pop
    ground = tp.Mesh(tp.PlaneGeometry(60, 60), tp.MeshStandardMaterial())
    ground.material.color = 0x2a2f37
    ground.material.roughness = 0.95
    ground.rotation.x = -math.pi / 2          # plane is XY; lay it flat (Y-up world)
    ground.receive_shadow = True
    scene.add(ground)

    # a ring of vividly colored props at varied radius/height -> varied color AND depth to sense
    props = tp.Group()
    palette = [0xff5252, 0xffb142, 0xffe14d, 0x6ddf6d, 0x4dd0e1, 0x5c8dff, 0xb96dff, 0xff6db4]
    n = 14
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 5.0 + 1.6 * math.sin(i * 1.7)
        col = palette[i % len(palette)]
        tall = (i % 3 == 0)
        h = 2.4 if tall else 1.0
        if i % 2 == 0:
            m = tp.Mesh(tp.BoxGeometry(0.9, h, 0.9), tp.MeshStandardMaterial())
        else:
            m = tp.Mesh(tp.SphereGeometry(0.55, 24, 16), tp.MeshStandardMaterial())
        m.material.color = col
        m.material.roughness = 0.6
        m.position.set(r * math.cos(a), h / 2.0, r * math.sin(a))
        m.cast_shadow = True
        m.receive_shadow = True
        props.add(m)
    scene.add(props)
    return scene, props


def distance_colors(pts, origin):
    """Map per-point distance from the sensor to a blue->cyan->green->red ramp (for the non-RGB-D mode)."""
    d = np.linalg.norm(pts - np.asarray(origin, np.float32), axis=1)
    t = np.clip(d / FAR, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t - 1.0) * 2.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t - 0.5) * 2.5, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t - 0.0) * 2.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", metavar="PNG", help="headless: scan once from a 3/4 angle and save a PNG")
    args = ap.parse_args()
    headless = bool(args.shot)

    canvas = tp.Canvas("threepp - DepthSensor", width=1000, height=680, antialiasing=4, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 1.1

    scene, props = build_scene()

    # the sensor: a GPU depth camera at human height in the middle of the ring. Default orientation
    # looks down -z (camera convention); we aim it by setting rotation (NOT look_at, which orients a
    # plain Object3D the opposite way). It is NOT added to the scene — scan() refreshes its own matrix.
    sensor = tp.DepthSensor(fov_y=58, width=SENSOR_W, height=SENSOR_H, near=0.1, far=FAR)
    sensor.position.set(0, 1.1, 0)
    sensor.range_noise = 0.015

    # the visualized point cloud: a fixed-capacity dynamic Points object (allocate MAXPTS once, then
    # update in place + draw range each frame -> no per-frame GPU buffer churn).
    geom = tp.BufferGeometry()
    geom.set_attribute("position", np.zeros((MAXPTS, 3), np.float32))
    geom.set_attribute("color", np.zeros((MAXPTS, 3), np.float32))
    geom.set_draw_range(0, 0)
    pmat = tp.PointsMaterial()
    pmat.size = 0.07
    pmat.size_attenuation = True
    pmat.vertex_colors = True
    cloud = tp.Points(geom, pmat)
    cloud.frustum_culled = False
    scene.add(cloud)

    camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
    camera.position.set(9, 7, 9)

    state = {"rgbd": True, "sweep": True, "props": False, "size": 0.07, "noise": 0.015, "yaw": 0.0, "npts": 0}

    def do_scan():
        sensor.rotation.y = state["yaw"]
        sensor.rotation.x = -0.18 + 0.12 * math.sin(state["yaw"] * 0.7)   # gentle pitch bob
        sensor.range_noise = state["noise"]
        props.visible = True                           # the sensor must see the props (scan needs them)
        cloud.visible = False                          # but hide the cloud so it doesn't scan itself
        if state["rgbd"]:
            pts, cols = sensor.scan_rgbd(renderer, scene)
        else:
            pts = sensor.scan(renderer, scene)
            cols = distance_colors(pts, (sensor.position.x, sensor.position.y, sensor.position.z))
        cloud.visible = True
        n = int(pts.shape[0])
        state["npts"] = n
        if n:
            geom.update_attribute("position", pts)
            geom.update_attribute("color", cols)
        geom.set_draw_range(0, n)
        props.visible = state["props"]                 # hide the solids for a clean "sensor data only" view

    # ---- headless still: render the reconstructed point cloud alone (props hidden) ----
    if headless:
        state["yaw"] = 0.9
        state["props"] = False
        pmat.size = 0.09
        do_scan()
        camera.position.set(10, 8, 10)
        camera.look_at(0, 1.0, 0)
        renderer.render(scene, camera)
        renderer.save_frame(args.shot)
        print(f"saved {args.shot}  ({state['npts']} points, props hidden = sensor data only)")
        return

    # ---- interactive ----
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target = tp.Vector3(0, 1.0, 0)

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)
    canvas.on_window_resize(on_resize)

    clock = tp.Clock()
    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("DepthSensor")
        tp.imgui.text(f"{SENSOR_W}x{SENSOR_H} depth image -> {state['npts']} points")
        _, state["rgbd"] = tp.imgui.checkbox("RGB-D colors (off = by distance)", state["rgbd"])
        _, state["props"] = tp.imgui.checkbox("show props (off = sensor data only)", state["props"])
        _, state["sweep"] = tp.imgui.checkbox("sweep", state["sweep"])
        _, state["noise"] = tp.imgui.slider_float("range noise (m)", state["noise"], 0.0, 0.15)
        changed, state["size"] = tp.imgui.slider_float("point size", state["size"], 0.01, 0.15)
        if changed:
            pmat.size = state["size"]
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   drag to orbit")
        tp.imgui.end()

    def frame():
        if state["sweep"]:
            state["yaw"] = clock.get_elapsed_time() * 0.6
        do_scan()
        controls.update()
        renderer.render(scene, camera)
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
            ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()
