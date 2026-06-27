"""DepthSensor showcase — live colored point cloud + incremental VoxelGrid map + marching-cubes surface.

A `tp.DepthSensor` sits in the middle of a ring of colored props and sweeps in yaw. Every frame it
produces an (N,3) numpy point cloud. Toggle "accumulate map" in the panel to feed each scan into a
VoxelGrid map; hit "rebuild surface" (or let it auto-rebuild every 90 frames) to run marching cubes
and show the reconstructed surface alongside the raw cloud.

    python depth_sensor.py                 # interactive window
    python depth_sensor.py --shot out.png  # headless: one scan, saved to a PNG

Controls: drag to orbit, scroll to zoom.
"""
import argparse
import math
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threepp as tp

SENSOR_W, SENSOR_H = 160, 120
MAXPTS = SENSOR_W * SENSOR_H
FAR = 18.0
MAP_VOXEL  = 0.12   # map accumulation voxel size
SURF_CELL  = 0.18   # marching-cubes grid cell
SURF_RAD   = 0.38   # union-of-balls radius (should be > SURF_CELL)
SURF_ISO   = 0.45   # isosurface level


def build_scene():
    scene = tp.Scene()
    scene.background = 0x141a22
    scene.add(tp.HemisphereLight(0xdfeaff, 0x404a55, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.6)
    sun.position.set(6, 12, 8)
    sun.cast_shadow = True
    scene.add(sun)

    ground = tp.Mesh(tp.PlaneGeometry(60, 60), tp.MeshStandardMaterial())
    ground.material.color = 0x2a2f37
    ground.material.roughness = 0.95
    ground.rotation.x = -math.pi / 2
    ground.receive_shadow = True
    scene.add(ground)

    props = tp.Group()
    palette = [0xff5252, 0xffb142, 0xffe14d, 0x6ddf6d, 0x4dd0e1, 0x5c8dff, 0xb96dff, 0xff6db4]
    n = 14
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 5.0 + 1.6 * math.sin(i * 1.7)
        col = palette[i % len(palette)]
        h = 2.4 if (i % 3 == 0) else 1.0
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
    d = np.linalg.norm(pts - np.asarray(origin, np.float32), axis=1)
    t = np.clip(d / FAR, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t - 1.0) * 2.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t - 0.5) * 2.5, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t - 0.0) * 2.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", metavar="PNG", help="headless: scan once and save PNG")
    args = ap.parse_args()
    headless = bool(args.shot)

    canvas = tp.Canvas("threepp - DepthSensor + SLAM", width=1100, height=700,
                       antialiasing=4, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 1.1

    scene, props = build_scene()

    sensor = tp.DepthSensor(fov_y=58, width=SENSOR_W, height=SENSOR_H, near=0.1, far=FAR)
    sensor.position.set(0, 1.1, 0)
    sensor.range_noise = 0.015

    # ---- live point cloud (fixed capacity, updated in place) ----
    cloud_geom = tp.BufferGeometry()
    cloud_geom.set_attribute("position", np.zeros((MAXPTS, 3), np.float32))
    cloud_geom.set_attribute("color",    np.zeros((MAXPTS, 3), np.float32))
    cloud_geom.set_draw_range(0, 0)
    pmat = tp.PointsMaterial()
    pmat.size = 0.07
    pmat.size_attenuation = True
    pmat.vertex_colors = True
    cloud = tp.Points(cloud_geom, pmat)
    cloud.frustum_culled = False
    scene.add(cloud)

    # ---- incremental VoxelGrid map + surface mesh ----
    map_grid = tp.VoxelGrid(MAP_VOXEL, max_points_per_voxel=3, min_spacing=0.07)
    surface_ref  = [None]   # current surface Mesh
    pending_iso  = [None]   # IsoMesh produced by background thread
    rebuild_lock = threading.Lock()
    rebuild_busy = [False]  # True while a background build is running

    def _rebuild_worker(pts_snapshot):
        field = tp.splat_points_to_field(pts_snapshot, SURF_CELL, SURF_RAD)
        iso   = tp.marching_cubes(field, SURF_ISO)
        with rebuild_lock:
            pending_iso[0] = iso if not iso.empty else None
            rebuild_busy[0] = False

    def trigger_rebuild():
        """Snapshot the map and kick off a background rebuild (no-op if one is running)."""
        with rebuild_lock:
            if rebuild_busy[0]:
                return
            pts = map_grid.collect()
            if len(pts) < 20:
                return
            rebuild_busy[0] = True
        t = threading.Thread(target=_rebuild_worker, args=(pts,), daemon=True)
        t.start()

    def apply_pending_surface():
        """Call from main thread each frame: swap in a finished IsoMesh if ready."""
        with rebuild_lock:
            iso = pending_iso[0]
            pending_iso[0] = None
        if iso is None:
            return
        geom = tp.iso_mesh_to_geometry(iso)
        mat  = tp.MeshStandardMaterial()
        mat.color     = 0x88bbff
        mat.roughness = 0.45
        mat.metalness = 0.05
        mat.side      = tp.Side.Double
        new_mesh = tp.Mesh(geom, mat)
        if surface_ref[0] is not None:
            scene.remove(surface_ref[0])
        surface_ref[0] = new_mesh
        if state["show_surface"]:
            scene.add(new_mesh)
        state["surf_verts"] = len(iso.positions) // 3

    camera = tp.PerspectiveCamera(55, canvas.aspect(), 0.1, 200)
    camera.position.set(9, 7, 9)

    state = {
        "rgbd": True, "sweep": True, "props": False,
        "size": 0.07, "noise": 0.015, "yaw": 0.0, "npts": 0,
        "accumulate": False, "show_surface": False,
        "surf_verts": 0, "frame_count": 0,
    }

    def do_scan():
        sensor.rotation.y = state["yaw"]
        sensor.rotation.x = -0.18 + 0.12 * math.sin(state["yaw"] * 0.7)
        sensor.range_noise = state["noise"]
        props.visible = True
        cloud.visible = False
        if surface_ref[0] is not None:
            surface_ref[0].visible = False   # don't scan the reconstructed surface
        if state["rgbd"]:
            pts, cols = sensor.scan_rgbd(renderer, scene)
        else:
            pts = sensor.scan(renderer, scene)
            cols = distance_colors(pts, (sensor.position.x, sensor.position.y, sensor.position.z))
        cloud.visible = True
        if surface_ref[0] is not None:
            surface_ref[0].visible = state["show_surface"]
        n = int(pts.shape[0])
        state["npts"] = n
        if n:
            cloud_geom.update_attribute("position", pts)
            cloud_geom.update_attribute("color", cols)
            if state["accumulate"]:
                map_grid.insert_array(pts)
        cloud_geom.set_draw_range(0, n)
        props.visible = state["props"]
        return pts

    # ---- headless ----
    if headless:
        state["yaw"] = 0.9
        state["props"] = False
        pmat.size = 0.09
        do_scan()
        camera.position.set(10, 8, 10)
        camera.look_at(0, 1.0, 0)
        renderer.render(scene, camera)
        renderer.save_frame(args.shot)
        print(f"saved {args.shot}  ({state['npts']} points)")
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
        tp.imgui.set_next_window_size(310, 0)
        tp.imgui.begin("DepthSensor + map")

        tp.imgui.text(f"{SENSOR_W}x{SENSOR_H}  ->  {state['npts']} pts/frame")
        _, state["rgbd"]  = tp.imgui.checkbox("RGB-D colors", state["rgbd"])
        _, state["props"] = tp.imgui.checkbox("show props",   state["props"])
        _, state["sweep"] = tp.imgui.checkbox("sweep",        state["sweep"])
        _, state["noise"] = tp.imgui.slider_float("range noise (m)", state["noise"], 0.0, 0.15)
        changed, state["size"] = tp.imgui.slider_float("point size", state["size"], 0.01, 0.15)
        if changed:
            pmat.size = state["size"]

        tp.imgui.separator()
        tp.imgui.text("— Map & surface —")
        _, state["accumulate"] = tp.imgui.checkbox("accumulate map", state["accumulate"])
        tp.imgui.same_line()
        if tp.imgui.button("clear"):
            map_grid.clear()
            state["surf_verts"] = 0
            if surface_ref[0] is not None:
                scene.remove(surface_ref[0])
                surface_ref[0] = None
        tp.imgui.text(f"map: {map_grid.voxel_count} voxels  ({map_grid.size} pts)")

        busy = rebuild_busy[0]
        if tp.imgui.button("rebuild surface" if not busy else "rebuilding..."):
            if not busy:
                trigger_rebuild()
        tp.imgui.same_line()
        ch, state["show_surface"] = tp.imgui.checkbox("show", state["show_surface"])
        if ch and surface_ref[0] is not None:
            if state["show_surface"]:
                scene.add(surface_ref[0])
            else:
                scene.remove(surface_ref[0])
        if state["surf_verts"]:
            tp.imgui.text(f"surface: {state['surf_verts']} tris")

        tp.imgui.separator()
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   drag to orbit")
        tp.imgui.end()

    AUTO_REBUILD_FRAMES = 90  # auto-rebuild surface every N frames while accumulating

    def frame():
        if state["sweep"]:
            state["yaw"] = clock.get_elapsed_time() * 0.6
        do_scan()

        # apply any finished background surface build
        apply_pending_surface()

        # kick off a new background build periodically while accumulating
        state["frame_count"] += 1
        if (state["accumulate"] and state["show_surface"] and
                state["frame_count"] % AUTO_REBUILD_FRAMES == 0 and
                map_grid.voxel_count > 50):
            trigger_rebuild()

        controls.update()
        renderer.render(scene, camera)
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
            ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()
