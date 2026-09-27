"""The GL studio kit: shadow/scene/texture/material knobs, RoomEnvironment +
PMREMGenerator, and the EffectComposer passes (GTAOPass, OutlinePass)."""
import numpy as np

import threepp as tp


def _camera():
    cam = tp.PerspectiveCamera(50, 200 / 150, 0.1, 100)
    cam.position.set(0, 0, 3.2)
    cam.look_at(0, 0, 0)
    return cam


def test_new_properties_round_trip():
    light = tp.DirectionalLight()
    light.shadow.intensity = 0.25
    light.shadow.radius = 4.0
    scene = tp.Scene()
    scene.background_blurriness = 0.5
    scene.background_intensity = 2.0
    scene.environment_intensity = 0.5
    scene.environment_rotation.y = 0.75  # in-place, like Object3D.rotation
    scene.background_rotation.x = -0.5
    tex = tp.Texture()
    tex.channel = 1
    mat = tp.MeshStandardMaterial()
    mat.alpha_hash = True
    mat.force_single_pass = True
    assert (light.shadow.intensity, light.shadow.radius, scene.background_blurriness,
            scene.background_intensity, scene.environment_intensity,
            round(scene.environment_rotation.y, 5), round(scene.background_rotation.x, 5),
            tex.channel, mat.alpha_hash, mat.force_single_pass) == \
        (0.25, 4.0, 0.5, 2.0, 0.5, 0.75, -0.5, 1, True, True)


def test_room_environment_lights_a_metal_sphere(renderer):
    env = tp.PMREMGenerator(renderer).from_scene(tp.RoomEnvironment(), 0.04)
    assert isinstance(env, tp.Texture)

    def centre_mean(environment):
        scene = tp.Scene()
        scene.background = tp.Background(0x000000)
        if environment is not None:
            scene.environment = environment
        mat = tp.MeshStandardMaterial()
        mat.metalness, mat.roughness = 1.0, 0.3
        scene.add(tp.Mesh(tp.SphereGeometry(1.0, 48, 32), mat))
        for _ in range(2):
            renderer.render(scene, _camera())
        return float(renderer.read_pixels()[55:95, 80:120].mean())

    assert centre_mean(env) > centre_mean(None) + 30


def test_composer_with_gtao_renders(renderer, lit_scene):
    floor = tp.Mesh(tp.PlaneGeometry(10, 10), tp.MeshStandardMaterial())
    floor.rotate_x(-1.5707963)
    floor.position.y = -1
    lit_scene.add(floor, tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
    cam = _camera()
    composer = tp.EffectComposer(renderer)
    composer.add_pass(tp.RenderPass(lit_scene, cam))
    gtao = tp.GTAOPass(lit_scene, cam, 200, 150)
    gtao.output = tp.GTAOPass.Output.Default
    gtao.update_gtao_material(radius=0.5, samples=8)
    composer.add_pass(gtao)
    composer.render()
    img = renderer.read_pixels()
    assert img.shape == (150, 200, 3) and int(img.max()) > int(img.min()) + 20


def test_outline_pass_draws_the_edge_colour(renderer):
    scene = tp.Scene()
    scene.background = tp.Background(0x000000)
    black = tp.MeshBasicMaterial()
    black.color = 0x000000
    ball = tp.Mesh(tp.SphereGeometry(0.8, 32, 16), black)
    scene.add(ball)
    cam = _camera()
    composer = tp.EffectComposer(renderer)
    composer.add_pass(tp.RenderPass(scene, cam))
    outline = tp.OutlinePass(tp.Vector2(200, 150), scene, cam)
    outline.selected_objects = [ball]  # set from a Python list
    outline.visible_edge_color = tp.Color(0x00ff00)
    outline.edge_strength = 5.0
    composer.add_pass(outline)
    composer.render()
    img = renderer.read_pixels().astype(int)
    green = (img[..., 1] > 120) & (img[..., 0] < 60) & (img[..., 2] < 60)
    assert outline.selected_objects[0] is ball and int(green.sum()) > 50
