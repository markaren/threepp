"""
Procedural terrain generator — Python demo.

Generates a heightfield via multifractal noise (fBm / Ridged / Hybrid)
with optional hydraulic + thermal erosion, and bakes a slope/altitude
splat texture (grass / scree / rock / snow). An ImGui panel exposes
four species presets and the key generation knobs.

    python terrain_demo.py
    python terrain_demo.py --shot out.png
"""

import argparse
import math
import threading
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import threepp as tp

PRESET_NAMES = ["Alpine", "Rolling Hills", "Desert Mesa", "Volcanic"]
NOISE_NAMES  = ["fBm", "Ridged", "Hybrid"]
EROSION_NAMES = ["None", "Hydraulic", "Thermal", "Both"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", metavar="PNG")
    args = ap.parse_args()
    headless = bool(args.shot)

    canvas = tp.Canvas("threepp - Procedural Terrain", width=1200, height=750,
                       antialiasing=4, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 1.1

    scene = tp.Scene()
    scene.background = 0x8ab4d4

    scene.add(tp.HemisphereLight(0xd0e8ff, 0x304020, 0.7))
    sun = tp.DirectionalLight(0xfff8e0, 3.0)
    sun.position.set(600, 800, 400)
    sun.cast_shadow = True
    sun.set_shadow_frustum(-900, 900, 900, -900)
    sun.set_shadow_bias(-0.0005)
    scene.add(sun)

    w, h = canvas.size()
    camera = tp.PerspectiveCamera(55, w / h, 1, 8000)
    camera.position.set(0, 600, 1400)
    camera.look_at(tp.Vector3(0, 0, 0))

    controls = tp.OrbitControls(camera, canvas)
    controls.target = tp.Vector3(0, 50, 0)
    controls.min_distance = 50
    controls.max_distance = 5000

    ui = tp.ImguiContext(canvas, renderer) if (tp.HAS_IMGUI and not headless) else None

    # ── terrain state ────────────────────────────────────────────────────────
    gen = tp.TerrainGenerator(1337)
    params = tp.TerrainParams()
    tp.apply_terrain_preset(0, params)      # start as Alpine
    params.resolution = 512                  # higher res → smoother splat
    params.ao_strength = 10.0               # GL has no IBL fill; baked AO must be subtle

    pending   = [None]   # result from worker: (geometry, texture) or None
    rebuilding = [False]
    status     = [""]

    # Current terrain mesh (None until first build)
    terrain_mesh = [None]

    def _rebuild_worker(p):
        nonlocal pending
        g = tp.TerrainGenerator(p.seed)
        g.build_field(p)
        if p.erosion != tp.ErosionType.Off:
            g.erode(p)
        geo = g.make_geometry(p)
        tex = g.bake_splat_texture(p)
        pending[0] = (geo, tex)
        rebuilding[0] = False
        status[0] = ""

    def trigger_rebuild():
        if rebuilding[0]:
            return
        rebuilding[0] = True
        status[0] = "Building..."
        p = _copy_params(params)
        threading.Thread(target=_rebuild_worker, args=(p,), daemon=True).start()

    def _copy_params(src):
        dst = tp.TerrainParams()
        dst.seed              = src.seed
        dst.world_size        = src.world_size
        dst.resolution        = src.resolution
        dst.noise_type        = src.noise_type
        dst.feature_scale     = src.feature_scale
        dst.octaves           = src.octaves
        dst.lacunarity        = src.lacunarity
        dst.gain              = src.gain
        dst.amplitude         = src.amplitude
        dst.warp              = src.warp
        dst.ridge_sharpness   = src.ridge_sharpness
        dst.height_exponent   = src.height_exponent
        dst.terraces          = src.terraces
        dst.falloff           = src.falloff
        dst.falloff_start     = src.falloff_start
        dst.erosion           = src.erosion
        dst.droplets          = src.droplets
        dst.droplet_lifetime  = src.droplet_lifetime
        dst.inertia           = src.inertia
        dst.sediment_capacity = src.sediment_capacity
        dst.min_slope         = src.min_slope
        dst.erode_speed       = src.erode_speed
        dst.deposit_speed     = src.deposit_speed
        dst.evaporation       = src.evaporation
        dst.gravity           = src.gravity
        dst.erosion_radius    = src.erosion_radius
        dst.talus_angle       = src.talus_angle
        dst.thermal_iterations= src.thermal_iterations
        dst.thermal_rate      = src.thermal_rate
        dst.snow_line         = src.snow_line
        dst.snow_noise_amp    = src.snow_noise_amp
        dst.snow_slope_max    = src.snow_slope_max
        dst.slope_grass_max   = src.slope_grass_max
        dst.slope_rock_min    = src.slope_rock_min
        dst.band_edge         = src.band_edge
        dst.rock_color        = list(src.rock_color)
        dst.grass_color       = list(src.grass_color)
        dst.scree_color       = list(src.scree_color)
        dst.snow_color        = list(src.snow_color)
        return dst

    def apply_pending():
        if pending[0] is None:
            return
        geo, tex = pending[0]
        pending[0] = None

        mat = tp.MeshStandardMaterial()
        mat.roughness = 0.92
        mat.metalness = 0.0
        mat.map = tex

        mesh = tp.Mesh(geo, mat)
        mesh.cast_shadow    = True
        mesh.receive_shadow = True

        if terrain_mesh[0] is not None:
            scene.remove(terrain_mesh[0])
        scene.add(mesh)
        terrain_mesh[0] = mesh

    # ── ImGui ────────────────────────────────────────────────────────────────
    imgui = tp.imgui  # widget functions must be called inside ui.render(callback)

    def draw_ui():
        nonlocal params

        imgui.set_next_window_size(310, 620)
        imgui.set_next_window_pos(10, 10)
        imgui.begin("Terrain")

        # Preset buttons
        imgui.text("Preset")
        imgui.separator()
        for i, name in enumerate(PRESET_NAMES):
            if imgui.button(name):
                tp.apply_terrain_preset(i, params)
                params.resolution  = 512
                params.ao_strength = 10.0
                trigger_rebuild()
            if i < len(PRESET_NAMES) - 1:
                imgui.same_line()
        imgui.spacing()

        # Noise
        imgui.text("Noise")
        imgui.separator()
        noise_idx = int(params.noise_type)
        ch, noise_idx = imgui.combo("Type", noise_idx, NOISE_NAMES)
        if ch:
            params.noise_type = [tp.NoiseType.fBm, tp.NoiseType.Ridged, tp.NoiseType.Hybrid][noise_idx]
        ch, v = imgui.slider_float("Amplitude (m)", params.amplitude, 50, 800)
        if ch: params.amplitude = v
        ch, v = imgui.slider_float("Feature scale", params.feature_scale, 100, 1000)
        if ch: params.feature_scale = v
        ch, v = imgui.slider_int("Octaves", params.octaves, 1, 10)
        if ch: params.octaves = v
        ch, v = imgui.slider_float("Warp", params.warp, 0, 1)
        if ch: params.warp = v
        ch, v = imgui.slider_float("Ridge sharp.", params.ridge_sharpness, 0, 1)
        if ch: params.ridge_sharpness = v
        ch, v = imgui.slider_float("Height exp.", params.height_exponent, 0.5, 2.5)
        if ch: params.height_exponent = v
        ch, v = imgui.slider_int("Terraces", params.terraces, 0, 20)
        if ch: params.terraces = v
        imgui.spacing()

        # Erosion
        imgui.text("Erosion")
        imgui.separator()
        er_idx = int(params.erosion)
        ch, er_idx = imgui.combo("Mode", er_idx, EROSION_NAMES)
        if ch:
            params.erosion = [tp.ErosionType.Off, tp.ErosionType.Hydraulic,
                              tp.ErosionType.Thermal, tp.ErosionType.Both][er_idx]
        ch, v = imgui.slider_int("Droplets", params.droplets, 10000, 200000)
        if ch: params.droplets = v
        ch, v = imgui.slider_float("Erode speed", params.erode_speed, 0.05, 0.8)
        if ch: params.erode_speed = v
        ch, v = imgui.slider_int("Talus iters", params.thermal_iterations, 0, 80)
        if ch: params.thermal_iterations = v
        imgui.spacing()

        # Texturing
        imgui.text("Texturing")
        imgui.separator()
        ch, v = imgui.slider_float("Snow line", params.snow_line, 0, 1.2)
        if ch: params.snow_line = v
        ch, v = imgui.slider_float("Grass max slope", params.slope_grass_max, 0.05, 0.6)
        if ch: params.slope_grass_max = v
        imgui.spacing()

        # Generate button
        lbl = "Generating..." if rebuilding[0] else "Generate"
        if imgui.button(lbl) and not rebuilding[0]:
            trigger_rebuild()
        if status[0]:
            imgui.text(status[0])

        imgui.end()

    # ── render loop ──────────────────────────────────────────────────────────
    trigger_rebuild()   # first build

    def frame():
        apply_pending()
        controls.update()
        renderer.render(scene, camera)
        if ui:
            controls.enabled = not ui.want_capture_mouse
            ui.render(draw_ui)

        if args.shot and terrain_mesh[0] is not None and not rebuilding[0]:
            renderer.read_rgb_pixels(0, 0, canvas.width, canvas.height, args.shot)
            canvas.close()

    canvas.animate(frame)


if __name__ == "__main__":
    main()
