"""threepp text & SVG overlay — world-anchored labels + a 2D HUD, no assets.

    python text_overlay.py            # interactive (drag to orbit)
    python text_overlay.py --shot out.png

Three lit objects, each tagged with a billboard TextSprite that floats above it
and always faces the camera. On top, a pixel-space HUD (an OrthographicCamera
rendered with auto_clear=False) draws a Text2D title and a small SVG badge.

Shows, all from Python:
  * FontLoader.default_font()  -> text with no font file
  * TextSprite                 -> world-anchored labels that billboard
  * Text2D + OrthographicCamera overlay (auto_clear=False) -> a 2D HUD
  * SVGLoader.parse(...)        -> an SVG turned into scene meshes
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

FONT = None  # the embedded default font (loaded in main)

# a tiny inline SVG "badge": a teal rounded chevron made of two filled paths
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48">'
    '<path d="M6,30 L18,8 L30,30 L24,30 L18,18 L12,30 Z" fill="#36e0c8"/>'
    '<path d="M20,40 L32,18 L44,40 L38,40 L32,28 L26,40 Z" fill="#ffb030"/>'
    '</svg>'
)

# (label, hex color, factory)
ITEMS = [
    ("Torus Knot", 0xffb030, lambda: tp.TorusKnotGeometry(0.55, 0.18, 160, 24)),
    ("Gem",        0x36e0c8, lambda: tp.IcosahedronGeometry(0.7)),
    ("Cube",       0xff5d8b, lambda: tp.BoxGeometry(1.1, 1.1, 1.1)),
]


def labeled(name, color, geom):
    """A lit object with a TextSprite label floating above it."""
    group = tp.Group()
    mat = tp.MeshStandardMaterial()
    mat.color = color
    mat.metalness = 0.25
    mat.roughness = 0.4
    mesh = tp.Mesh(geom, mat)
    mesh.cast_shadow = True
    group.add(mesh)

    label = tp.TextSprite(FONT, world_scale=0.34)
    label.set_text(name)
    label.set_color(0xffffff)
    label.set_vertical_alignment(tp.VerticalAlignment.Above)
    label.position.set(0, 1.0, 0)
    group.add(label)
    return group


def build_hud(w, h):
    """A pixel-space HUD scene + ortho camera (origin bottom-left, y up)."""
    hud = tp.Scene()
    cam = tp.OrthographicCamera(0, w, h, 0, -10, 10)

    title = tp.Text2D(FONT, "threepp  -  text & SVG overlay", size=26)
    title.set_color(0xeaf2ff)
    title.position.set(24, h - 48, 0)
    hud.add(title)

    sub = tp.Text2D(FONT, "TextSprite labels - Text2D HUD - SVGLoader badge", size=15)
    sub.set_color(0x8aa0c0)
    sub.position.set(24, h - 74, 0)
    hud.add(sub)

    badge = tp.SVGLoader().parse(LOGO_SVG)   # Group of filled meshes (y already flipped)
    badge.scale.set(1.6, -1.6, 1.6)          # keep the loader's y-flip, scale up
    badge.position.set(w - 110, 92, 0)
    hud.add(badge)
    return hud, cam


def main():
    global FONT
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", metavar="PNG", help="render one frame headless and save")
    args = ap.parse_args()
    headless = bool(args.shot)

    w, h = 1100, 620
    canvas = tp.Canvas("threepp - text & SVG overlay", width=w, height=h,
                       antialiasing=4, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic

    FONT = tp.FontLoader().default_font()

    scene = tp.Scene()
    scene.background = tp.Background(0x0d0f15)
    scene.add(tp.HemisphereLight(0xbcd0ff, 0x202028, 0.7))
    key = tp.DirectionalLight(0xffffff, 2.4)
    key.position.set(4, 8, 6)
    key.cast_shadow = True
    scene.add(key)
    rim = tp.DirectionalLight(0x4060ff, 1.0)
    rim.position.set(-6, 2, -4)
    scene.add(rim)

    floor = tp.Mesh(tp.PlaneGeometry(60, 60), tp.MeshStandardMaterial())
    floor.material.color = 0x10131b
    floor.material.roughness = 0.6
    floor.rotate_x(-math.pi / 2)
    floor.position.y = -1.15
    floor.receive_shadow = True
    scene.add(floor)

    objects = tp.Group()
    for i, (name, color, factory) in enumerate(ITEMS):
        node = labeled(name, color, factory())
        node.position.x = (i - 1) * 3.4
        objects.add(node)
    scene.add(objects)

    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.1, 100)
    camera.position.set(0, 1.7, 9.0)
    camera.look_at(0, 0.4, 0)

    hud, hud_cam = build_hud(w, h)

    def render():
        renderer.auto_clear = True
        renderer.render(scene, camera)       # main 3D pass (clears)
        renderer.auto_clear = False          # overlay on top, don't clear
        renderer.render(hud, hud_cam)        # 2D HUD pass

    if headless:
        for _ in range(2):
            render()
        renderer.save_frame(args.shot)
        print(f"saved {args.shot}")
        return

    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target = tp.Vector3(0, 0.4, 0)
    controls.auto_rotate = True
    controls.auto_rotate_speed = 0.5

    def on_resize(width, height):
        camera.aspect = width / max(height, 1)
        camera.update_projection_matrix()
        renderer.set_size(width, height)

    canvas.on_window_resize(on_resize)
    clock = tp.Clock()

    def animate():
        dt = clock.get_delta()
        for k, node in enumerate(objects.children):
            node.children[0].rotation.y += dt * (0.5 + 0.15 * k)
        controls.update()
        render()

    canvas.animate(animate)


if __name__ == "__main__":
    main()
