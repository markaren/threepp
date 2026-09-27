"""threepp GL studio: studio lighting, soft shadows, ambient occlusion and outlines.

    python gl_studio.py                      # interactive (drag to orbit, scroll to zoom)
    python gl_studio.py --headless out.png   # render one frame without a window and save it

A row of spheres, from rough plastic to polished chrome, and a torus knot on a
pale floor. No HDR file: the light comes from a RoomEnvironment turned into an
environment map with PMREMGenerator, plus one key light for the shadows.

Shows, all on the OpenGL renderer:
  * RoomEnvironment + PMREMGenerator.from_scene -> image-based light without assets
  * light.shadow.radius / .intensity            -> soft, lighter shadows
  * EffectComposer + RenderPass                 -> a post-processing chain
  * GTAOPass                                    -> ambient occlusion in the contacts
  * OutlinePass                                 -> an editor-style selection outline

Keys: 1-6 select an object (0 clears the selection), A toggles ambient
occlusion, O toggles the outline.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

# (color, metalness, roughness) for the row of spheres, rough to mirror-like.
SPHERES = [
    (0xd8573c, 0.0, 0.85),
    (0xe0b040, 0.0, 0.35),
    (0x4f8fd6, 0.3, 0.25),
    (0xc9a36b, 1.0, 0.30),
    (0xf0f2f5, 1.0, 0.04),
]


def standard(color, metalness, roughness):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.metalness = metalness
    m.roughness = roughness
    return m


def build_scene(renderer):
    scene = tp.Scene()
    scene.background = tp.Background(0x2b2f36)

    # Image-based light from a procedural studio room: no HDR file needed.
    room = tp.RoomEnvironment()
    scene.environment = tp.PMREMGenerator(renderer).from_scene(room, 0.04)
    room.dispose()
    scene.environment_intensity = 0.5  # leave room for the key light's shadows

    floor = tp.Mesh(tp.PlaneGeometry(200, 200), standard(0x5c636c, 0.0, 0.8))
    floor.rotate_x(-math.pi / 2)
    floor.receive_shadow = True
    scene.add(floor)

    objects = []
    r = 0.55
    for k, (color, metal, rough) in enumerate(SPHERES):
        s = tp.Mesh(tp.SphereGeometry(r, 64, 32), standard(color, metal, rough))
        s.position.set((k - 2) * 1.35, r, 1.3)
        s.cast_shadow = True
        s.receive_shadow = True
        scene.add(s)
        objects.append(s)

    knot = tp.Mesh(tp.TorusKnotGeometry(0.7, 0.24, 220, 32), standard(0x2a9d8f, 0.6, 0.22))
    knot.position.set(0, 1.25, -1.0)
    knot.cast_shadow = True
    knot.receive_shadow = True
    scene.add(knot)
    objects.append(knot)

    # A box the knot hangs over, so AO has a crease and a contact to find.
    block = tp.Mesh(tp.BoxGeometry(2.4, 0.3, 1.2), standard(0xe8e4dc, 0.0, 0.7))
    block.position.set(0, 0.15, -1.0)
    block.cast_shadow = True
    block.receive_shadow = True
    scene.add(block)

    key = tp.DirectionalLight(0xffffff, 3.5)
    key.position.set(4, 7, 5)
    key.cast_shadow = True
    key.set_shadow_frustum(-6, 6, 6, -6)
    key.shadow.map_size.set(2048, 2048)
    key.shadow.radius = 6.0       # a wide, soft penumbra
    key.shadow.intensity = 0.75   # shadows lose 75 % of the key light, not all of it
    key.shadow.bias = -0.0005
    scene.add(key)

    return scene, objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", metavar="PNG", help="render one frame without a window and save it")
    ap.add_argument("--size", default="1280x720", help="window size WxH (default 1280x720)")
    args = ap.parse_args()
    headless = bool(args.headless)
    w, h = (int(v) for v in args.size.lower().split("x"))

    canvas = tp.Canvas("threepp - GL studio", width=w, height=h, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic

    scene, objects = build_scene(renderer)

    camera = tp.PerspectiveCamera(40, canvas.aspect(), 0.1, 100)
    camera.position.set(0.5, 3.2, 7.5)
    camera.look_at(0, 0.6, 0)

    # The chain: draw the scene, darken its contacts, outline the selection.
    # The composer tone-maps once at the end (an implicit OutputPass).
    fw, fh = renderer.size()
    composer = tp.EffectComposer(renderer, samples=4)
    composer.add_pass(tp.RenderPass(scene, camera))
    gtao = tp.GTAOPass(scene, camera, fw, fh)
    gtao.update_gtao_material(radius=0.5, thickness=1.0)
    composer.add_pass(gtao)
    outline = tp.OutlinePass(tp.Vector2(fw, fh), scene, camera, [objects[-1]])
    outline.visible_edge_color = tp.Color(0xff5000)
    outline.hidden_edge_color = tp.Color(0x301000)
    outline.edge_strength = 6.0  # the blurred edge peaks near 0.2, so scale it up
    outline.edge_thickness = 1.0
    composer.add_pass(outline)

    if headless:
        for _ in range(3):  # a couple of frames so the environment's PMREM is built
            composer.render()
        os.makedirs(os.path.dirname(os.path.abspath(args.headless)), exist_ok=True)
        renderer.save_frame(args.headless)
        print(f"saved {args.headless}")
        return

    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target = tp.Vector3(0, 0.6, 0)

    def on_resize(width, height):
        camera.aspect = width / max(height, 1)
        camera.update_projection_matrix()
        renderer.set_size(width, height)
        composer.set_size(width, height)

    canvas.on_window_resize(on_resize)

    clock = tp.Clock()
    held = set()  # keys down last frame, for press edges

    def pressed(key):
        down = canvas.is_key_down(key)
        was = key in held
        (held.add if down else held.discard)(key)
        return down and not was

    def animate():
        dt = clock.get_delta()
        objects[-1].rotation.y += dt * 0.4

        for i in range(len(objects)):
            if pressed(str(i + 1)):
                outline.selected_objects = [objects[i]]  # assign a whole list
        if pressed("0"):
            outline.selected_objects = []
        if pressed("A"):
            gtao.enabled = not gtao.enabled
        if pressed("O"):
            outline.enabled = not outline.enabled

        controls.update()
        composer.render(dt)

    canvas.animate(animate)


if __name__ == "__main__":
    main()
