"""threepp PBR showcase — interactive, HDR image-based lighting, no assets.

    python pbr_showcase.py            # interactive (drag to orbit, scroll to zoom)
    python pbr_showcase.py --shot out.png   # render one frame headless and save

A gold torus-knot ringed by metal spheres and orbiting emissive gems on a
glossy floor, lit entirely by a *procedurally generated* HDR sky (so the demo
needs no downloaded environment map). Everything reflects the sky via threepp's
image-based lighting; an ImGui panel drives it live.

Shows off, all from Python:
  * RGBELoader + scene.environment  -> real HDR image-based lighting
  * ACESFilmic tone mapping         -> filmic highlights instead of clipping
  * MeshStandardMaterial metalness/roughness across a curated set of metals
  * Vector3.project(camera)         -> the hero's live screen-space position
  * OrbitControls + ImGui           -> interactive camera and controls
"""
import argparse
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

# --------------------------------------------------------------------------- #
#  Procedural HDR sky  (numpy -> Radiance .hdr -> RGBELoader)
# --------------------------------------------------------------------------- #
SUN_DIR = np.array([0.45, 0.40, 0.80])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)


def _encode_rgbe(rgb):
    """Vectorised linear-RGB float -> Radiance RGBE bytes, shape (H, W, 4)."""
    rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
    m = rgb.max(axis=2)
    mask = m >= 1e-32
    safe = np.where(mask, m, 1.0)
    mant, exp = np.frexp(safe)             # m = mant * 2**exp,  mant in [0.5, 1)
    scale = np.where(mask, mant * 256.0 / safe, 0.0)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., 0] = np.clip(rgb[..., 0] * scale, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(rgb[..., 1] * scale, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(rgb[..., 2] * scale, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    return out


def make_sky_hdr(path, W=2048, H=1024):
    """Write an equirectangular HDR sky: warm horizon, blue zenith, a bright sun."""
    j = np.arange(H).reshape(H, 1)
    i = np.arange(W).reshape(1, W)
    theta = (j / H) * math.pi               # polar angle, 0 = straight up (row 0 = zenith)
    phi = (i / W) * 2 * math.pi - math.pi   # azimuth
    y = np.broadcast_to(np.cos(theta), (H, W))  # +1 zenith .. -1 nadir, full grid
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)                      # (H,1)*(1,W) -> (H,W)
    z = sin_t * np.sin(phi)

    up = np.clip(y, 0.0, 1.0)[..., None]
    zenith = np.array([0.05, 0.16, 0.42])
    horizon = np.array([0.55, 0.45, 0.38])
    t = up ** 0.35                                   # fast roll into blue above the horizon
    sky = horizon * (1.0 - t) + zenith * t

    # a thin, bright warm band right at the horizon line (a line, not a wash)
    glow = np.exp(-(y * y) / (2 * 0.0045))[..., None]
    sky = sky + glow * np.array([1.0, 0.62, 0.32]) * 0.7

    # dark ground below the horizon
    below = (y < 0)[..., None]
    sky = np.where(below, np.array([0.05, 0.045, 0.04]), sky)

    # sun: smooth gaussian core + wide halo. A soft angular falloff (no hard disc
    # cutoff) means no stair-stepped edge in the sky or in glossy reflections.
    d = np.stack([x, y, z], axis=-1)
    cosang = np.clip((d * SUN_DIR).sum(-1), -1.0, 1.0)
    ang = np.arccos(cosang)                                   # angle from sun centre
    core = np.exp(-(ang / math.radians(1.5)) ** 2)
    halo = np.exp(-(ang / math.radians(11.0)) ** 2)
    sky = sky + (core * 60.0 + halo * 4.5)[..., None] * np.array([1.0, 0.93, 0.82])

    rgbe = _encode_rgbe(sky)
    # ensure stb reads the scanlines uncompressed: the first pixel must not look
    # like an RLE run marker (2, 2, <128).
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (H, W))
        f.write(rgbe.tobytes())
    return path


# --------------------------------------------------------------------------- #
#  Materials
# --------------------------------------------------------------------------- #
# (name, hex color, metalness, roughness)
METALS = [
    ("Gold",       0xffc24d, 1.0, 0.16),
    ("Chrome",     0xf3f5f8, 1.0, 0.05),
    ("Copper",     0xd9885a, 1.0, 0.22),
    ("Rose Gold",  0xe7a596, 1.0, 0.18),
    ("Steel",      0xaab2bd, 1.0, 0.34),
    ("Emerald",    0x29b06a, 0.9, 0.12),
    ("Sapphire",   0x3f74e0, 0.9, 0.12),
    ("Obsidian",   0x15171f, 0.35, 0.07),
]


def standard(color, metalness, roughness):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.metalness = metalness
    m.roughness = roughness
    m.env_map_intensity = 1.0
    return m


# --------------------------------------------------------------------------- #
#  Scene
# --------------------------------------------------------------------------- #
FLOOR_Y = -1.25


def build_scene(env_tex):
    scene = tp.Scene()
    scene.environment = env_tex            # <-- image-based lighting
    scene.background = env_tex             # show the HDR sky behind the objects

    # glossy dark floor that catches the sky + shadows; large enough to recede
    # to the horizon and blend into the sky (no hard far edge)
    floor = tp.Mesh(tp.PlaneGeometry(400, 400),
                    standard(0x0b0c11, 0.5, 0.32))
    floor.rotate_x(-math.pi / 2)
    floor.position.y = FLOOR_Y
    floor.receive_shadow = True
    scene.add(floor)

    # hero: a big gold torus knot, lifted off the floor
    hero = tp.Mesh(tp.TorusKnotGeometry(0.78, 0.27, 220, 32),
                   standard(*METALS[0][1:]))
    hero.position.y = 0.9
    hero.cast_shadow = True
    scene.add(hero)

    # ring of metal spheres resting on the floor
    ring = tp.Group()
    n = len(METALS)
    radius = 3.1
    spheres = []
    for k, (name, color, metal, rough) in enumerate(METALS):
        a = (k / n) * 2 * math.pi
        s = tp.Mesh(tp.SphereGeometry(0.62, 48, 32), standard(color, metal, rough))
        s.position.set(radius * math.cos(a), FLOOR_Y + 0.62, radius * math.sin(a))
        s.cast_shadow = True
        ring.add(s)
        spheres.append(s)
    scene.add(ring)

    # orbiting emissive gems (flat-shaded -> faceted, and they glow)
    gems = []
    gem_colors = [0xff3d8b, 0x21e0c8, 0xffb020]
    for k, c in enumerate(gem_colors):
        mat = tp.MeshStandardMaterial()
        mat.color = c
        mat.emissive = c
        mat.emissive_intensity = 2.4
        mat.metalness = 0.1
        mat.roughness = 0.35
        mat.flat_shading = True
        g = tp.Mesh(tp.IcosahedronGeometry(0.32), mat)
        g.cast_shadow = True
        scene.add(g)
        gems.append(g)

    # key light aligned with the painted sun + a touch of sky fill
    key = tp.DirectionalLight(0xfff1dd, 2.3)
    key.position.set(*(SUN_DIR * 12))
    key.cast_shadow = True
    scene.add(key)
    scene.add(tp.HemisphereLight(0x9fb4d6, 0x14110e, 0.28))

    return scene, hero, ring, spheres, gems


# --------------------------------------------------------------------------- #
#  App
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", metavar="PNG", help="render one frame headless and save")
    args = ap.parse_args()
    headless = bool(args.shot)

    hdr_path = os.path.join(tempfile.gettempdir(), "threepp_pbr_sky.hdr")
    make_sky_hdr(hdr_path)

    w, h = (1280, 720)
    canvas = tp.Canvas("threepp - PBR showcase", width=w, height=h,
                       antialiasing=4, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 1.0

    env = tp.RGBELoader().load(hdr_path)
    scene, hero, ring, spheres, gems = build_scene(env)

    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.1, 200)
    camera.position.set(0.0, 3.0, 9.0)
    camera.look_at(0, 0.8, 0)

    def step(t, dt):
        hero.rotation.y += dt * 0.5
        hero.rotation.z += dt * 0.18
        ring.rotation.y += dt * 0.12
        for k, g in enumerate(gems):
            a = t * 0.6 + k * (2 * math.pi / len(gems))
            g.position.set(2.0 * math.cos(a), 1.4 + 0.5 * math.sin(t * 1.3 + k), 2.0 * math.sin(a))
            g.rotation.x += dt * 1.1
            g.rotation.y += dt * 0.7

    # ---- headless: render a few frames so motion/IBL settle, then save ----
    if headless:
        for f in range(4):
            step(f * 0.1, 0.1)
            renderer.render(scene, camera)
        renderer.save_frame(args.shot)
        print(f"saved {args.shot}")
        return

    # ---- interactive ----
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target = tp.Vector3(0, 0.8, 0)
    controls.auto_rotate = True
    controls.auto_rotate_speed = 0.4

    ui = tp.ImguiContext(canvas, renderer)
    clock = tp.Clock()
    state = {
        "exposure": 1.0, "env_intensity": 1.0,
        "hero_metal": 1.0, "hero_rough": 0.16, "preset": 0,
        "auto_rotate": True, "spin": True, "wireframe": False, "sky": True,
    }

    def on_resize(width, height):
        camera.aspect = width / max(height, 1)
        camera.update_projection_matrix()
        renderer.set_size(width, height)

    canvas.on_window_resize(on_resize)

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(310, 0)
        tp.imgui.begin("threepp - PBR showcase")

        tp.imgui.text("Lighting")
        _, state["exposure"] = tp.imgui.slider_float("exposure", state["exposure"], 0.2, 3.0)
        renderer.tone_mapping_exposure = state["exposure"]
        _, state["env_intensity"] = tp.imgui.slider_float("IBL intensity", state["env_intensity"], 0.0, 3.0)
        _, state["sky"] = tp.imgui.checkbox("show HDR sky", state["sky"])
        scene.background = env if state["sky"] else tp.Background(0x0a0c12)

        tp.imgui.separator()
        tp.imgui.text("Hero material")
        # Edit inside the block; material_edits() flushes once on exit so the
        # changes reach the GPU under Vulkan (no-op on GL).
        with tp.material_edits(renderer, hero.material):
            changed, state["preset"] = tp.imgui.combo("metal", state["preset"], [m[0] for m in METALS])
            if changed:
                _, color, metal, rough = METALS[state["preset"]]
                hero.material.color = color
                state["hero_metal"], state["hero_rough"] = metal, rough
            _, state["hero_metal"] = tp.imgui.slider_float("metalness", state["hero_metal"], 0.0, 1.0)
            _, state["hero_rough"] = tp.imgui.slider_float("roughness", state["hero_rough"], 0.0, 1.0)
            hero.material.metalness = state["hero_metal"]
            hero.material.roughness = state["hero_rough"]
            _, state["wireframe"] = tp.imgui.checkbox("wireframe", state["wireframe"])
            hero.material.wireframe = state["wireframe"]

        tp.imgui.separator()
        tp.imgui.text("Camera")
        _, state["auto_rotate"] = tp.imgui.checkbox("auto-orbit", state["auto_rotate"])
        _, state["spin"] = tp.imgui.checkbox("animate objects", state["spin"])
        if tp.imgui.button("reset view"):
            camera.position.set(0.0, 3.0, 9.0)
            controls.target = tp.Vector3(0, 0.8, 0)

        # showcase Vector3.project: where the hero lands on screen
        p = hero.get_world_position().project(camera)
        sw, sh = canvas.size()
        px, py = (p.x * 0.5 + 0.5) * sw, (1 - (p.y * 0.5 + 0.5)) * sh
        tp.imgui.separator()
        tp.imgui.text(f"hero -> screen ({px:.0f}, {py:.0f})")
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
        tp.imgui.end()

    def animate():
        dt = clock.get_delta()
        t = clock.get_elapsed_time()
        if state["spin"]:
            step(t, dt)
        for s in spheres:
            s.material.env_map_intensity = state["env_intensity"]
        hero.material.env_map_intensity = state["env_intensity"]
        controls.auto_rotate = state["auto_rotate"]
        controls.enabled = not ui.want_capture_mouse
        controls.update()
        renderer.render(scene, camera)
        ui.render(draw_ui)

    canvas.animate(animate)


if __name__ == "__main__":
    main()
