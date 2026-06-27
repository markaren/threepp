"""Procedural tree generator — Python demo.

Generates trunk + leaf geometry via space colonisation (TreeGenerator) with
procedural bark and leaf textures (no external files). An ImGui panel exposes
all TreeParams knobs and four species presets.

    python tree_demo.py
    python tree_demo.py --shot out.png
"""
import argparse
import math
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threepp as tp

PRESETS = ["Oak", "Pine", "Birch", "Willow", "Custom"]


def make_tree(params, bark_seed=None):
    """Build trunk + leaf geometry + materials from params. Thread-safe (GIL released in C++)."""
    gen = tp.TreeGenerator(params.seed)
    gen.build_skeleton(params)
    trunk_geo = gen.make_trunk_geometry(params)
    leaf_geo  = gen.make_leaf_geometry(params)

    bseed = bark_seed if bark_seed is not None else params.seed
    bark_alb, bark_nrm = tp.make_bark_textures(256, bseed,
                                                base_color=params.bark_color)
    bark_alb.repeat.set(2, 4)
    bark_nrm.repeat.set(2, 4)

    trunk_mat = tp.MeshStandardMaterial()
    trunk_mat.map          = bark_alb
    trunk_mat.normal_map   = bark_nrm
    trunk_mat.normal_scale = tp.Vector2(0.8, 0.8)
    trunk_mat.roughness    = 0.9
    trunk_mat.metalness    = 0.0

    leaf_tex = tp.make_leaf_texture(256, params.seed ^ 0xABCD,
                                    base_color=params.leaf_color)
    leaf_mat = tp.MeshStandardMaterial()
    leaf_mat.map        = leaf_tex
    leaf_mat.alpha_test = 0.45
    leaf_mat.side       = tp.Side.Double
    leaf_mat.roughness  = 0.85

    return trunk_geo, trunk_mat, leaf_geo, leaf_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", metavar="PNG")
    args = ap.parse_args()
    headless = bool(args.shot)

    canvas = tp.Canvas("threepp - Procedural Tree", width=1100, height=750,
                       antialiasing=4, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = False
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 1.15

    scene = tp.Scene()
    scene.background = 0x8ab4d4

    scene.add(tp.HemisphereLight(0xcfe8ff, 0x304020, 0.9))
    sun = tp.DirectionalLight(0xfff5e0, 3.5)
    sun.position.set(8, 14, 6)
    scene.add(sun)

    ground = tp.Mesh(tp.PlaneGeometry(40, 40), tp.MeshStandardMaterial())
    ground.material.color    = 0x3a4a2a
    ground.material.roughness = 0.95
    ground.rotation.x = -math.pi / 2
    scene.add(ground)

    # ---- current tree meshes (swapped each rebuild) ----
    current_meshes = [None, None]  # [trunk, leaf]

    params = tp.TreeParams()
    tp.apply_tree_preset(0, params)   # start as Oak

    state = {
        "preset":      0,
        "dirty":       True,   # rebuild on first frame
        "rebuilding":  False,
        # mirror a few TreeParams for the ImGui sliders
        "trunk_height": params.trunk_height,
        "trunk_radius": params.trunk_radius,
        "crown_shape":  int(params.crown_shape),
        "crown_rx":     params.crown_radius_x,
        "crown_rz":     params.crown_radius_z,
        "crown_h":      params.crown_height,
        "attractors":   params.attractor_count,
        "seg_len":      params.segment_length,
        "randomness":   params.randomness,
        "tropism":      params.tropism,
        "leaf_style":   int(params.leaf_style),
        "leaf_size":    params.leaf_size,
        "leaf_density": params.leaf_density,
        "leaf_clump":   params.leaf_clumping,
        "seed":         params.seed,
        "node_count":   0,
    }

    pending = [None]   # (trunk_geo, trunk_mat, leaf_geo, leaf_mat) when ready

    def _rebuild_worker(p):
        result = make_tree(p)
        pending[0] = result
        state["node_count"] = tp.TreeGenerator(p.seed).node_count  # rough count from last build
        state["rebuilding"] = False

    def trigger_rebuild():
        if state["rebuilding"]:
            return
        state["rebuilding"] = True
        p = tp.TreeParams()
        p.seed             = state["seed"]
        p.trunk_height     = state["trunk_height"]
        p.trunk_radius     = state["trunk_radius"]
        p.crown_shape      = tp.CrownShape(state["crown_shape"])
        p.crown_radius_x   = state["crown_rx"]
        p.crown_radius_z   = state["crown_rz"]
        p.crown_height     = state["crown_h"]
        p.attractor_count  = state["attractors"]
        p.segment_length   = state["seg_len"]
        p.randomness       = state["randomness"]
        p.tropism          = state["tropism"]
        p.leaf_style       = tp.LeafStyle(state["leaf_style"])
        p.leaf_size        = state["leaf_size"]
        p.leaf_density     = state["leaf_density"]
        p.leaf_clumping    = state["leaf_clump"]
        threading.Thread(target=_rebuild_worker, args=(p,), daemon=True).start()

    def apply_pending():
        if pending[0] is None:
            return
        trunk_geo, trunk_mat, leaf_geo, leaf_mat = pending[0]
        pending[0] = None
        # Remove old meshes by reference (avoids iterating children as T&)
        if current_meshes[0] is not None:
            scene.remove(current_meshes[0])
        if current_meshes[1] is not None:
            scene.remove(current_meshes[1])
        trunk = tp.Mesh(trunk_geo, trunk_mat)
        leaf  = tp.Mesh(leaf_geo, leaf_mat)
        scene.add(trunk)
        scene.add(leaf)
        current_meshes[0] = trunk
        current_meshes[1] = leaf

    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.1, 200)
    camera.position.set(0, 6, 14)

    if headless:
        trigger_rebuild()
        import time
        while state["rebuilding"]:
            time.sleep(0.01)
        apply_pending()
        camera.look_at(0, 5, 0)
        renderer.render(scene, camera)
        renderer.save_frame(args.shot)
        print(f"saved {args.shot}")
        return

    controls = tp.OrbitControls(camera, canvas)
    controls.target = tp.Vector3(0, 4, 0)
    controls.enable_damping = True

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)
    canvas.on_window_resize(on_resize)

    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None
    CROWN_NAMES  = ["Sphere", "Ellipsoid", "Cone", "Hemisphere", "Cylinder"]
    LEAF_NAMES   = ["Quad", "Cluster", "CrossQuad", "Blob"]

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("Tree Generator")

        # ── Presets ────────────────────────────────────────────────────────
        tp.imgui.text("Species preset")
        for i, name in enumerate(PRESETS[:-1]):
            if tp.imgui.button(name):
                state["preset"] = i
                p = tp.TreeParams()
                p.seed = state["seed"]
                tp.apply_tree_preset(i, p)
                state["trunk_height"] = p.trunk_height
                state["trunk_radius"] = p.trunk_radius
                state["crown_shape"]  = int(p.crown_shape)
                state["crown_rx"]     = p.crown_radius_x
                state["crown_rz"]     = p.crown_radius_z
                state["crown_h"]      = p.crown_height
                state["attractors"]   = p.attractor_count
                state["seg_len"]      = p.segment_length
                state["randomness"]   = p.randomness
                state["tropism"]      = p.tropism
                state["leaf_style"]   = int(p.leaf_style)
                state["leaf_size"]    = p.leaf_size
                state["leaf_density"] = p.leaf_density
                state["leaf_clump"]   = p.leaf_clumping
                state["dirty"]        = True
            if i < 3:
                tp.imgui.same_line()

        tp.imgui.separator()

        ch, v = tp.imgui.slider_int("seed", state["seed"], 1, 9999)
        if ch: state["seed"] = v; state["dirty"] = True

        tp.imgui.text("Trunk")
        ch, v = tp.imgui.slider_float("height",  state["trunk_height"], 1.0, 12.0)
        if ch: state["trunk_height"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("radius",  state["trunk_radius"], 0.02, 0.5)
        if ch: state["trunk_radius"] = v; state["dirty"] = True

        tp.imgui.text("Crown")
        ch, v = tp.imgui.combo("shape", state["crown_shape"], CROWN_NAMES)
        if ch: state["crown_shape"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("radius X", state["crown_rx"], 0.5, 8.0)
        if ch: state["crown_rx"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("radius Z", state["crown_rz"], 0.5, 8.0)
        if ch: state["crown_rz"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("height##c", state["crown_h"], 1.0, 12.0)
        if ch: state["crown_h"] = v; state["dirty"] = True

        tp.imgui.text("Growth")
        ch, v = tp.imgui.slider_int("attractors", state["attractors"], 100, 2000)
        if ch: state["attractors"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("seg length", state["seg_len"], 0.1, 0.8)
        if ch: state["seg_len"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("randomness", state["randomness"], 0.0, 0.3)
        if ch: state["randomness"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("tropism", state["tropism"], -0.2, 0.2)
        if ch: state["tropism"] = v; state["dirty"] = True

        tp.imgui.text("Leaves")
        ch, v = tp.imgui.combo("style", state["leaf_style"], LEAF_NAMES)
        if ch: state["leaf_style"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("size",    state["leaf_size"],    0.1, 1.5)
        if ch: state["leaf_size"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("density", state["leaf_density"], 0.1, 1.0)
        if ch: state["leaf_density"] = v; state["dirty"] = True
        ch, v = tp.imgui.slider_float("clumping",state["leaf_clump"],   0.0, 0.9)
        if ch: state["leaf_clump"] = v; state["dirty"] = True

        tp.imgui.separator()
        busy = state["rebuilding"]
        if tp.imgui.button("Rebuild" if not busy else "Building..."):
            if not busy:
                state["dirty"] = False
                trigger_rebuild()
        if state["dirty"] and not busy:
            tp.imgui.same_line()
            tp.imgui.text("(params changed)")

        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
        tp.imgui.end()

    def frame():
        apply_pending()
        if state["dirty"] and not state["rebuilding"]:
            state["dirty"] = False
            trigger_rebuild()
        controls.update()
        renderer.render(scene, camera)
        if ui:
            controls.enabled = not ui.want_capture_mouse
            ui.render(draw_ui)

    print(__doc__)
    canvas.animate(frame)


if __name__ == "__main__":
    main()
