"""Shared world for the spider demos — a lit ground arena both spider_demo.py and
play.py build, so they look the same and the robot has things to walk around and
shove. Returns the list of dynamic prop meshes (already added to scene + world).
"""
import math
import random

import threepp as tp

_EARTH = [0x6b5b43, 0x7a6a4f, 0x5c4f3a, 0x837154, 0x4f4636]
_PROPS = [0xE5C04B, 0x3CA0E5, 0xE5814B, 0x49C66A, 0xA64BE5, 0xE0556E]
_SKY = 0x1a2530


def _mat(color, rough=0.85, metal=0.0):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.roughness = rough
    m.metalness = metal
    return m


def _scatter(rng, extent, clear):
    while True:
        x, z = rng.uniform(-extent, extent), rng.uniform(-extent, extent)
        if math.hypot(x, z) > clear:
            return x, z


def build_arena(scene, world, extent=15.0, n_rocks=14, n_props=14, seed=3, shadows=True):
    rng = random.Random(seed)

    scene.background = _SKY
    scene.set_fog(_SKY, 11.0, 48.0)  # distance haze toward the background colour

    # Lighting: sky/ground hemisphere + a warm key (the shadow caster) + cool fill.
    scene.add(tp.HemisphereLight(0xbcd4f0, 0x2c2a22, 0.85))
    sun = tp.DirectionalLight(0xfff0d8, 2.6)
    sun.position.set(9, 18, 7)
    sun.cast_shadow = shadows
    scene.add(sun)
    fill = tp.DirectionalLight(0x7090c0, 0.45)
    fill.position.set(-7, 9, -5)
    scene.add(fill)

    # Ground.
    ground = tp.Mesh(tp.BoxGeometry(2 * extent + 12, 1, 2 * extent + 12), _mat(0x39463c, 0.97))
    ground.position.y = -0.5
    ground.receive_shadow = True
    scene.add(ground)
    world.add_static(ground)

    # Subtle grid (visual only — thin, flush with the ground, no collider).
    line_mat = tp.MeshBasicMaterial()
    line_mat.color = 0x45554a
    step, n = 3.0, int(extent // 3.0)
    for k in range(-n, n + 1):
        c = k * step
        for w, d, px, pz in ((0.035, 2 * extent, c, 0.0), (2 * extent, 0.035, 0.0, c)):
            ln = tp.Mesh(tp.BoxGeometry(w, 0.02, d), line_mat)
            ln.position.set(px, 0.011, pz)
            scene.add(ln)

    # Perimeter wall (static) so it reads as an arena.
    for w, d, px, pz in ((2 * extent + 1, 0.6, 0, -extent), (2 * extent + 1, 0.6, 0, extent),
                         (0.6, 2 * extent + 1, -extent, 0), (0.6, 2 * extent + 1, extent, 0)):
        wall = tp.Mesh(tp.BoxGeometry(w, 1.3, d), _mat(0x2c3530, 0.9))
        wall.position.set(px, 0.65, pz)
        wall.cast_shadow = wall.receive_shadow = True
        scene.add(wall)
        world.add_static(wall)

    # Scattered rock obstacles (static colliders), kept clear of the start area.
    for _ in range(n_rocks):
        x, z = _scatter(rng, extent - 2.5, 3.5)
        if rng.random() < 0.55:
            h = rng.uniform(0.45, 1.3)
            rock = tp.Mesh(tp.BoxGeometry(rng.uniform(0.5, 1.4), h, rng.uniform(0.5, 1.4)),
                           _mat(rng.choice(_EARTH), 0.95))
            rock.rotate_y(rng.uniform(0, math.pi))
            y = h * 0.5
        else:
            r = rng.uniform(0.45, 0.95)
            rock = tp.Mesh(tp.SphereGeometry(r), _mat(rng.choice(_EARTH), 0.92))
            y = r
        rock.position.set(x, y, z)
        rock.cast_shadow = rock.receive_shadow = True
        scene.add(rock)
        world.add_static(rock)

    # Knock-around dynamic props. Returns (body, home_position) so the demos can
    # snap them back on reset.
    props = []
    for _ in range(n_props):
        x, z = _scatter(rng, extent - 2.0, 2.8)
        s = rng.uniform(0.22, 0.4)
        prop = tp.Mesh(tp.BoxGeometry(s, s, s), _mat(rng.choice(_PROPS), 0.55))
        home = (x, s * 0.5 + 0.02, z)
        prop.position.set(*home)
        prop.cast_shadow = True
        scene.add(prop)
        body = world.add(prop, density=55)
        props.append((body, home))
    return props


def reset_props(props):
    """Snap every dynamic prop back to its home pose with zero velocity."""
    zero = tp.Vector3(0, 0, 0)
    for body, home in props:
        body.set_pose(tp.Vector3(*home))
        body.set_linear_velocity(zero)
        body.set_angular_velocity(zero)
