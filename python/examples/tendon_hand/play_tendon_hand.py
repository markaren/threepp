"""Run a trained hold policy on the CPU hand, with its REAL cables, through the roll-over.

This is the deploy path and the sim-to-sim check in one. Training runs on the batched torch
cable because PhysX rejects addForce under the direct-GPU API; the film runs on the C++
`TendonCable`, which is the reference implementation of the same law. Nothing about the policy
changes between the two -- the same 25 tensions go in -- so if the port is right the posture
here is the posture the policy was trained into, and if it is not, this is where it shows.

The roll is world.set_gravity every control step, following the same schedule function the env
trained on (tendon_hand_env.roll_theta), with the film schedule from the checkpoint's meta by
default. Training applied gravity as a per-link force in a zero-gravity scene; here it is
PhysX's own gravity, which is the same force through the same centres of mass.

The observation is rebuilt from the CPU state in the same order `TendonHandEnv.observe` builds
it, and every constant it needs (T_MAX, the filter time constant, the palm target, the pull and
roll schedules) is read from the checkpoint's meta rather than re-declared, so the two cannot
drift.

Run:  python play_tendon_hand.py tendon_hand_hold.pt --view
      python play_tendon_hand.py tendon_hand_hold.pt --object sphere --density 1000
      python play_tendon_hand.py tendon_hand_hold.pt --shots out/ --shot-times 1.0,3.5,7.5
      python play_tendon_hand.py tendon_hand_hold.pt --film out/film_sphere/

--shots renders headless stills out of the SAME run that prints the hold numbers, on the same
GL headless path tendon_hand.py's visual() uses. It is deliberately not a separate replay:
a still taken from a second, differently-seeded run would be a picture of a different grasp
than the one the numbers describe.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))    # python/
sys.path.insert(0, _HERE)

import threepp as tp
from tendon_hand import DT, Forearm, Hand, RopeView, VIEWS, _aim, _scene, _unit_cyl
from tendon_hand_env import (OBJ_TYPES, gravity_at, object_desc, object_mesh, roll_theta,
                             sample_object)
from threepp.rl import load_policy


# ---- the still/film camera and scene -------------------------------------------------------
#
# SCREEN UP IS WORLD -Y, and that is not a style choice. Gravity in this task is +Y: the hand
# is fixed-base with its pad along -Y, so "palm up" is achieved by turning gravity round rather
# than by rotating the hand. World +Y is therefore DOWN. Leave the camera's default up of +Y
# and the render comes out upside down -- the object falls upward out of frame and the floor is
# in the sky. Every framing constant below is written against that.
#
# THE ROLL IS RENDERED BY TURNING THE CAMERA. The hand never moves; gravity turns about an axis
# through the palm. Turning the camera, the lights and the HUD about that same axis by the same
# angle gives the picture of a hand that turned under a fixed camera in a fixed world, which is
# what the film has to read as. The pull arrow is attached to the object in world coordinates
# and needs nothing done to it.
SHOT_UP = (0.0, -1.0, 0.0)
SHOT_TARGET = (0.062, -0.022, 0.002)    # the palm, with the object sitting on it
SHOT_DIR = (0.45, -0.78, -0.46)         # target -> camera: distal, above the palm, thumb side
SHOT_DIST = 0.29
SHOT_FOV = 38.0
# NO FLOOR AND NO SHADOW, and that is a decision with numbers behind it, not an omission.
# Three attempts, each diagnosed by printing the shadow camera and measuring the hand in the
# LIGHT'S frame rather than by staring at the render:
#   1. light at its natural 0.47 m -- a directional light's shadow camera has a 0.5 near plane,
#      so the whole hand sat in front of it and the map was empty.
#   2. light at 1.9 m, ortho +-0.40, but aimed at the world ORIGIN while the hand lives 60 mm
#      from it: the map was mostly empty space and what reached the floor were two small
#      detached blobs, which is what the v2 stills show.
#   3. aimed at the palm, ortho +-0.12 -- measured, the hand spans x [-0.035, +0.122] and
#      y [-0.108, +0.031] in the light's frame, so the fingertips fell outside and only the
#      part that fitted cast anything. At +-0.17 all 21 links are inside and the map is right,
#      and the shadow is then invisible anyway: the light is nearly straight above the palm and
#      so is the camera, so it lands squarely behind the hand, which occludes it.
# Getting it right needs a light direction chosen for the shadow rather than for the modelling,
# and a floor close enough to catch it without intersecting the fingers. That is a lighting
# design problem, and a plain light background beats a wrong shadow, so the floor is out.
# With the roll, a floor would also have to turn with the camera, which is one more reason.
KEY_OFFSET = (0.12, -1.10, 0.18)        # the key light, as an offset from the subject
FILL_POS = (-0.10, -0.26, -0.30)        # a volar fill from the camera side
RIM_POS = (-0.30, 0.10, 0.26)           # a low rim from behind, for the silhouette
HUD_POS = (0.02, -0.118, -0.01)         # above a 90 mm object standing on the palm (the old spot)
HUD_DEPTH = 0.06                        # in front of the palm, as HUD_POS was
HUD_UP = 0.96                           # of the half frame height at that depth
CABLE_XRAY = 0.85                       # the cables read as being inside the fingers


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def rot_about(v, axis, theta):
    """v turned by theta about the unit axis (Rodrigues), numpy."""
    c, s = math.cos(theta), math.sin(theta)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


class ShotRig:
    """The camera, the three lights and the HUD as offsets from the palm, turned together
    about the roll axis by the roll angle. set(0) is the framing every still before the roll
    existed was taken with."""

    def __init__(self, cam, lights, hud, pivot, axis, legend=None):
        self.cam, self.lights, self.hud, self.legend = cam, lights, hud, legend
        self.pivot, self.axis = np.asarray(pivot, dtype=float), _unit(axis)
        self.cam_rel = _unit(SHOT_DIR) * SHOT_DIST
        self.cam_up = np.array(SHOT_UP, dtype=float)
        self.light_rel = [np.asarray(p, dtype=float) - self.pivot for p in
                          (self.pivot + np.array(KEY_OFFSET), FILL_POS, RIM_POS)]
        # The HUD on the view axis, centred, 6 cm in front of the palm and near the top edge of
        # the frame at that depth -- the height HUD_POS had, above a 90 mm object on the palm.
        right, up = screen_axes()
        depth = SHOT_DIST - HUD_DEPTH
        self.hud_rel = (_unit(SHOT_DIR) * HUD_DEPTH
                        + up * (HUD_UP * depth * math.tan(math.radians(0.5 * SHOT_FOV))))
        self.set(0.0)

    def set(self, theta):
        R = lambda v: rot_about(v, self.axis, theta)
        self.cam.position.set(*(self.pivot + R(self.cam_rel)))
        self.cam.up.set(*R(self.cam_up))
        self.cam.look_at(*self.pivot)
        for light, rel in zip(self.lights, self.light_rel):
            light.position.set(*(self.pivot + R(rel)))
        if self.hud is not None:
            self.hud.position.set(*(self.pivot + R(self.hud_rel)))
        if self.legend is not None:
            self.legend.set(theta)


def shot_scene(width, height, headless=True):
    """A lit scene for a picture someone who has not seen this hand can read.

    tendon_hand.py's own `_scene` is a dark studio void with a hard key: right for a cutaway
    poster of the mechanism, wrong here, where the subject is a grasp and the viewer needs a
    light background and enough fill to see finger volume. Every light is aimed at the palm so
    the rig can turn it about the palm without changing what it lights.
    """
    canvas = tp.Canvas("tendon hand", width=width, height=height, antialiasing=4,
                       headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    scene = tp.Scene()
    scene.background = tp.Background(0xDFE3E8)

    cam = tp.PerspectiveCamera(SHOT_FOV, width / height, 0.01, 10)

    scene.add(tp.HemisphereLight(0xFFFFFF, 0x9AA4B0, 0.75))
    key = tp.DirectionalLight(0xFFF6EC, 2.1)
    fill = tp.DirectionalLight(0xE6EEFF, 1.0)
    rim = tp.DirectionalLight(0xFFE4CC, 0.8)
    for light in (key, fill, rim):
        light.get_target().position.set(*SHOT_TARGET)
        scene.add(light)
    return canvas, renderer, scene, cam, (key, fill, rim)


class PullArrow:
    """The disturbance force, as an arrow that is actually attached to the object.

    tp.ArrowHelper's shaft is a LINE -- one pixel wide whatever the scene scale -- so in a
    still all that survived was the cone: a red triangle floating near the hand with nothing
    connecting it to anything. At t = 5.5 s in the first contact sheet it sat ABOVE the hand
    and read as a stray mark. A 3 mm cylinder plus a proportionate cone is the whole fix.
    """
    SHAFT_R = 0.003
    HEAD_R = 0.0075            # 2.5x the shaft radius
    HEAD_L = 0.016
    MIN_N = 0.3                # below this the arrow is hidden rather than drawn as a stub
    N_PER_M = 0.01             # 1 cm per newton
    MAX_M = 0.10               # and no longer than this

    def __init__(self, scene, colour=0xD81028):
        mat = tp.MeshStandardMaterial()
        mat.color = tp.Color(colour)
        mat.roughness = 0.45
        mat.metalness = 0.05
        self.shaft = tp.Mesh(tp.CylinderGeometry(1.0, 1.0, 1.0, 16), mat)   # unit, scaled below
        self.head = tp.Mesh(tp.ConeGeometry(self.HEAD_R, self.HEAD_L, 20), mat)
        for m in (self.shaft, self.head):
            m.visible = False
            scene.add(m)

    @staticmethod
    def _aim(mesh, u):
        """Rotate the mesh's own +Y (the axis both a cylinder and a cone run along) onto u."""
        ax = np.cross((0.0, 1.0, 0.0), u)
        s = float(np.linalg.norm(ax))
        if s < 1e-9:
            mesh.quaternion.set(0.0, 0.0, 0.0, 1.0) if u[1] > 0 else \
                mesh.quaternion.set(1.0, 0.0, 0.0, 0.0)
            return
        ax = ax / s
        ang = math.acos(max(-1.0, min(1.0, float(u[1]))))
        mesh.quaternion.set_from_axis_angle(
            tp.Vector3(float(ax[0]), float(ax[1]), float(ax[2])), ang)

    def update(self, origin, force):
        n = float(np.linalg.norm(force))
        if n < self.MIN_N:
            self.shaft.visible = self.head.visible = False
            return
        u = np.asarray(force, dtype=float) / n
        # 1 cm per newton sized the arrow for the trained 8 N. A pull-to-failure take goes
        # past 25 N, where the arrow would cross the frame and the HUD; the HUD carries the
        # number, so the arrow stops growing at MAX_M.
        total = min(self.N_PER_M * n, self.MAX_M)
        shaft_len = max(1e-4, total - self.HEAD_L)
        # The shaft starts at the object's CENTRE, so the arrow is attached in every frame
        # whatever the object is doing.
        mid = np.asarray(origin, dtype=float) + u * (0.5 * shaft_len)
        self.shaft.position.set(*mid)
        self.shaft.scale.set(self.SHAFT_R, shaft_len, self.SHAFT_R)
        self._aim(self.shaft, u)
        self.head.position.set(*(np.asarray(origin, dtype=float)
                                 + u * (shaft_len + 0.5 * self.HEAD_L)))
        self._aim(self.head, u)
        self.shaft.visible = self.head.visible = True


def dress(hand, obj_mesh):
    """Turn the cutaway hand into a readable one, without touching how it is built.

    Everything here is a material property on meshes that already exist, set after Hand has
    constructed them -- Hand, the cables and the env are untouched. Three changes:
      the SKIN goes to 0.85 with depth write ON. At 0.28 with depth write off, every capsule
        blends with every capsule behind it and a finger reads as a brown cloud rather than as
        three segments.
      the CABLES become an x-ray overlay: depth test off, drawn after the skin. Without this
        they are occluded in the places that matter and look like rods lying on top of the
        hand; with it they read as running inside the fingers, which is what they do.
      the OBJECT goes opaque and saturated, so it is obviously a held thing and not more skin.
    """
    for m in hand.meshes:
        # OPAQUE, not 0.85. Two reasons, both found by rendering: a transparent skin does not
        # land in the shadow map, so the hand floated with no contact shadow at all; and at any
        # alpha the capsules still tint each other where they overlap, which is exactly the
        # blob the cutaway look produces. The cables stay visible because they are an x-ray
        # overlay drawn with depth test off, so nothing is lost by closing the skin.
        m.material.transparent = False
        m.material.opacity = 1.0
        m.material.depth_write = True
        m.material.roughness = 0.65
    obj_mesh.material.transparent = False
    obj_mesh.material.opacity = 1.0
    obj_mesh.material.depth_write = True
    obj_mesh.material.roughness = 0.45
    obj_mesh.material.metalness = 0.05


def xray(view):
    """The cable materials, after RopeView has made them. Shared per cable, so setting them
    here covers every segment the pool will ever create."""
    for m in view.mats.values():
        m.transparent = True
        m.opacity = CABLE_XRAY
        m.depth_test = False
        m.roughness = 0.4
        m.metalness = 0.0


# ---- the tendons, lit by tension -----------------------------------------------------------
#
# Coloured by finger, the ropes show the routing. The film is about the policy, and what the
# policy does is tension, so each rope takes the colour of its commanded tension every frame:
# slack ropes recede to slate, loaded ones go amber, then red, then white-hot at T_MAX. The
# colour goes on the cable's material, which every segment of that rope shares, and the
# emissive term rises with it so a loaded rope reads as lit rather than as painted. The
# legend is a bar of the same rope under the hand, in the same material, turned by the
# ShotRig with the camera so it stays put on screen while the hand turns over.
TENSION_RAMP = ((0.00, 0x46577A), (0.30, 0xE8B23C), (0.65, 0xF04A1E), (1.00, 0xFFE6B4))
# The legend sits on the view axis LEGEND_DEPTH in front of the palm, nearer the camera than
# any part of the hand can reach, so it is never behind a finger; LEGEND_DROP of the frame's
# half height below centre at that depth. Sizes are given at the palm's depth and scaled.
LEGEND_DEPTH = 0.16                     # an extended fingertip reaches ~0.10 toward the camera
LEGEND_DROP = 0.84
LEGEND_LEFT = 0.55                      # of the half frame width: the corner no finger reaches
LEGEND_LEN = 0.070
LEGEND_RADIUS = 0.0022
LEGEND_TEXT = 0x1A1F28


def tension_colour(u):
    """The ramp colour at u = T / T_MAX, piecewise linear between the stations."""
    u = min(1.0, max(0.0, float(u)))
    for (u0, c0), (u1, c1) in zip(TENSION_RAMP, TENSION_RAMP[1:]):
        if u <= u1:
            return tp.Color(c0).lerp(tp.Color(c1), (u - u0) / (u1 - u0))
    return tp.Color(TENSION_RAMP[-1][1])


def light_by_tension(mat, u):
    c = tension_colour(u)
    mat.color = c
    mat.emissive = tp.Color(c.r, c.g, c.b)
    mat.emissive_intensity = 0.15 + 0.85 * u ** 1.5


def tint(view, names, tension, t_max):
    """Recolour every rope from this step's filtered tensions, the ones the cables carry."""
    for n, T in zip(names, tension):
        light_by_tension(view.mats[n], float(T) / t_max)


def screen_axes():
    """World directions that are screen right and screen up in the still framing."""
    f = -_unit(SHOT_DIR)
    up = np.array(SHOT_UP, dtype=float)
    up = _unit(up - np.dot(up, f) * f)
    return _unit(np.cross(f, up)), up


class Legend:
    """The tension ramp as a bar of rope, 'slack' to T_MAX, with the rope's own material, so
    the legend is the look and not a chart of it. A Group at the palm pivot, so one rotation
    about the roll axis moves the whole thing with the camera."""

    def __init__(self, scene, font, t_max, pivot, axis, aspect, caption=None, scale=1.0, n=24):
        self.axis = tp.Vector3(*(float(v) for v in _unit(axis)))
        self.group = tp.Group()
        self.group.position.set(*(float(v) for v in pivot))
        right, up = screen_axes()
        depth = SHOT_DIST - LEGEND_DEPTH                  # legend to camera
        k = depth / SHOT_DIST                             # same size on screen as at the palm
        half_h = depth * math.tan(math.radians(0.5 * SHOT_FOV))
        rel = (_unit(SHOT_DIR) * LEGEND_DEPTH - up * (LEGEND_DROP * half_h)
               - right * (LEGEND_LEFT * half_h * aspect))
        length, radius = LEGEND_LEN * k, LEGEND_RADIUS * k
        for i in range(n):
            m = tp.MeshStandardMaterial()
            m.roughness = 0.4
            m.metalness = 0.0
            light_by_tension(m, i / (n - 1))
            seg = tp.Mesh(_unit_cyl(), m)
            a = rel + right * (length * (i / n - 0.5))
            b = rel + right * (length * ((i + 1) / n - 0.5))
            _aim(seg, a, b, radius)
            self.group.add(seg)
        H = tp.HorizontalAlignment
        labels = [("slack", rel - right * (0.5 * length + 0.006 * k), H.Right),
                  (f"{t_max:.0f} N", rel + right * (0.5 * length + 0.006 * k), H.Left),
                  ("cable tension", rel + up * (0.011 * k), H.Center)]
        if caption:
            # What is in the hand: the object, its mass, its friction. Top-left, level with
            # the HUD: text sprites are not depth-tested against the hand, and the top-left
            # is the one corner no finger reaches in any frame.
            # A bigger HUD is a taller HUD: the caption drops a row so the two do not touch.
            row = 0.86 - 0.16 * (scale - 1.0)
            labels.append((caption, rel + up * ((LEGEND_DROP + row) * half_h)
                           - right * (0.5 * length + 0.02 * k), H.Left))
        for text, at, align in labels:
            s = tp.TextSprite(font)
            s.set_text(text)
            s.set_horizontal_alignment(align)
            s.set_color(tp.Color(LEGEND_TEXT))
            s.set_world_scale(0.0055 * k * scale)
            s.position.set(*(float(v) for v in at))
            self.group.add(s)
        scene.add(self.group)

    def set(self, theta):
        self.group.quaternion.set_from_axis_angle(self.axis, float(theta))


def contact_times(roll, seconds, t_drop=None, pull_full=None):
    """Six moments that show the episode: settled, mid-turn, hanging, hanging at full pull,
    mid-turn back, the end. For a take that ends at the drop: settled, mid-turn, at the
    trained pull, just before the slip, the slip, the end."""
    t0, dur, hang = roll["t0"], roll["dur"], roll["hang"]
    if t_drop is not None:
        return (0.3, t0 + 0.5 * dur, pull_full or t0 + dur + 0.3,
                max(t0 + dur + 0.3, t_drop - 0.6), max(0.3, t_drop - 0.05), seconds)
    return (0.3, t0 + 0.5 * dur, t0 + dur + 0.3, t0 + dur + hang - 0.2,
            t0 + 1.5 * dur + hang, seconds)


def contact_sheet(film_dir, times, hz, n_steps, cols=3, cell=(640, 360)):
    """Six frames of the episode in one image, so the MOTION is visible in a still.

    A 3x2 grid rather than one row of six: at six across, each cell is 426 px wide and the
    cables stop being separable, which is the whole thing the sheet is meant to show.
    """
    from PIL import Image, ImageDraw
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1]), (198, 204, 212))
    draw = ImageDraw.Draw(sheet)
    for k, t in enumerate(times):
        fr = min(n_steps, max(1, int(round(t * hz))))
        fn = os.path.join(film_dir, f"frame_{fr:04d}.png")
        if not os.path.exists(fn):
            continue
        im = Image.open(fn).convert("RGB").resize(cell, Image.LANCZOS)
        x, y = (k % cols) * cell[0], (k // cols) * cell[1]
        sheet.paste(im, (x, y))
        draw.text((x + 12, y + 10), f"t = {t:.1f} s", fill=(26, 31, 40))
    out = os.path.join(film_dir, "contact.png")
    sheet.save(out)
    print(f"    wrote {out}  ({cols}x{rows} of {cell[0]}x{cell[1]})")
    return out


def build(meta, kind, dims, density, friction, seed):
    g = np.array(meta["gravity"], dtype=float)
    world = tp.PhysxWorld(gravity=tp.Vector3(*g), fixed_timestep=DT, max_substeps=1,
                          tgs_pcm=True)
    pad = world.create_material(1.2, 1.1, 0.0, friction_combine="min")
    hand = Hand(world, material=pad).finalize()
    hand.route()                                    # the REAL cables, this time
    rng = np.random.default_rng(seed)

    target = np.array(meta["palm_target"], dtype=float)
    g_hat = g / np.linalg.norm(g)
    mesh = object_mesh(kind, dims)
    jitter = np.array([(rng.random() * 2 - 1) * 0.015, 0.0, (rng.random() * 2 - 1) * 0.015])
    mesh.position.set(*(target + jitter - g_hat * (0.03 + 0.03 * rng.random())))
    # A random orientation, as the env drops them: a bar or cylinder dropped upright is an
    # easier catch than the training distribution, and the hold rate here has to be
    # comparable with the GPU's.
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    mesh.quaternion.set(*(float(v) for v in q))
    art = world.create_articulation(fixed_base=False)
    mat = world.create_material(friction, 0.9 * friction, 0.0, friction_combine="min")
    art.add_link(mesh, density=density, material=mat)
    art.finalize()
    return world, hand, art, mesh, target, g_hat


def observe(meta, hand, art, mesh, target, tension, pull, g):
    """The same numbers, in the same order, that TendonHandEnv.observe stacks."""
    q = hand.pose()
    qd = np.array([hand.links[n].joint_velocity for n in hand.dof_names])
    rv = art.root_velocity()
    pos = np.array([mesh.position.x, mesh.position.y, mesh.position.z])
    desc = meta["_desc"]
    return np.concatenate([q, qd * 0.1, tension / meta["t_max"], pos - target,
                           rv[0:3], rv[3:6] * 0.1, desc, pull / 10.0,
                           g / 9.81]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--object", default="sphere", choices=list(OBJ_TYPES))
    ap.add_argument("--density", type=float, default=None,
                    help="kg/m^3; default draws from the training range with --seed")
    ap.add_argument("--friction", type=float, default=None,
                    help="object material friction; default draws from the training range")
    ap.add_argument("--seconds", type=float, default=None, help="default: the episode length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roll-angle", type=float, default=None,
                    help="degrees the hand turns over; default the checkpoint's film schedule")
    ap.add_argument("--roll-axis", default=None,
                    help="x,y,z axis gravity turns about; default the finger axis (pronation)")
    ap.add_argument("--no-roll", action="store_true", help="gravity stays palm-up")
    ap.add_argument("--roll-hang", type=float, default=None,
                    help="seconds held turned over (default: the film schedule's)")
    ap.add_argument("--pull-max", type=float, default=None, metavar="N",
                    help="keep ramping at the training slope past the trained pull, to this")
    ap.add_argument("--until-drop", type=float, default=None, metavar="S",
                    help="end the run S seconds after the object drops: a pull-to-failure take")
    ap.add_argument("--hud-scale", type=float, default=1.0,
                    help="HUD, legend and caption text size; 1.4 survives a phone screen")
    ap.add_argument("--no-hud", action="store_true",
                    help="no HUD, legend or caption: a clean plate for compositing")
    ap.add_argument("--view", action="store_true", help="window, with every cable drawn")
    ap.add_argument("--shots", default=None, metavar="DIR",
                    help="render headless PNG stills into DIR, no window")
    ap.add_argument("--shot-times", default="1.0,3.5,7.5",
                    help="seconds at which --shots writes a still")
    ap.add_argument("--film", default=None, metavar="DIR",
                    help="60 fps PNG sequence of the whole episode at 1280x720, plus a "
                         "contact sheet; encode it with the ffmpeg line in README.md")
    ap.add_argument("--size", default="1280x800")
    a = ap.parse_args()

    ac, norm, meta = load_policy(a.checkpoint, "cpu")
    hz, dt = meta["control_hz"], 1.0 / meta["control_hz"]
    sub = meta["substeps"]
    alpha = dt / (meta["tau_filter"] + dt)
    seconds = a.seconds if a.seconds is not None else float(meta["episode_s"])
    pull_start = meta["pull_start_s"]
    pull_full = meta.get("pull_full_s", meta["episode_s"])
    rng = np.random.default_rng(a.seed)
    dims, density, friction = sample_object(a.object, rng, a.density, a.friction)
    meta["_desc"] = np.array(object_desc(a.object, dims))

    # The roll schedule: the checkpoint's film schedule, with the angle and axis overridable.
    roll = dict(meta["roll_film"])
    if a.roll_angle is not None:
        roll["ang_deg"] = a.roll_angle
    if a.roll_axis is not None:
        roll["axis"] = tuple(float(v) for v in a.roll_axis.split(","))
    if a.no_roll:
        roll["ang_deg"] = 0.0
    if a.roll_hang is not None:
        roll["hang"] = a.roll_hang
    axis = _unit(roll["axis"])
    # The pull past the trained limit is the same ramp carried on: the policy never saw more
    # than pull_max in training, so where it lets go is a measurement of margin, not of skill.
    pull_cap = a.pull_max if a.pull_max is not None else float(meta["pull_max"])
    sched = tuple(torch.tensor(float(v)) for v in
                  (roll["t0"], roll["dur"], roll["hang"], math.radians(roll["ang_deg"])))
    axis_t = torch.tensor(axis, dtype=torch.float32)[None]
    g0 = torch.tensor(meta["gravity"], dtype=torch.float32)

    def gravity(t):
        th = roll_theta(torch.tensor(float(t)), *sched)
        return th.item(), gravity_at(th[None], axis_t, g0)[0].numpy().astype(float)

    world, hand, art, mesh, target, g_hat = build(meta, a.object, dims, density, friction,
                                                  a.seed)
    # The cable order the 25 actions are in. The checkpoint carries it; the hand's own routing
    # is the same list by construction, so it is both the fallback for an older checkpoint and
    # a check that the two have not drifted.
    names = list(hand.routing)
    if "cable_names" in meta and list(meta["cable_names"]) != names:
        raise RuntimeError("the checkpoint's cable order is not this hand's:\n  "
                           f"checkpoint {list(meta['cable_names'])}\n  hand {names}")
    tension = np.zeros(len(names))
    pull = np.zeros(3)
    pull_dir = np.zeros(3)
    t_drop = pull_at_drop = None      # set once the object leaves the hand; read in control()
    g_now = np.array(meta["gravity"], dtype=float)
    theta_now = 0.0
    body = art.link(0)
    shot_t = [float(s) for s in a.shot_times.split(",") if s.strip()] if a.shots else []
    # Run at least long enough to reach the last still asked for, so --shots 8.0 does not
    # silently write nothing.
    n_steps = int(round(max(seconds, max(shot_t, default=0.0)) * hz))
    shot_frame = {int(round(t * hz)): t for t in shot_t}
    every = max(1, int(round(meta["pull_resample_s"] * hz)))

    canvas = renderer = scene = cam = view = arm = rig = None
    arrow = hud = None
    render_w, render_h = (int(v) for v in a.size.split("x"))
    if a.film:
        render_w, render_h = 1280, 720
    if a.view or a.shots or a.film:
        if a.view:
            canvas, renderer, scene, cam = _scene(render_w, render_h, False)
        else:
            canvas, renderer, scene, cam, lights = shot_scene(render_w, render_h)
        for m in hand.meshes:
            scene.add(m)
        scene.add(mesh)
        # The Forearm belongs in the window and NOT in a still. It extends every cable
        # backwards to a motor bank 130 mm proximal, and at this framing those 25 straight rods
        # cross the entire frame and sit on top of the palm -- measured on the first render,
        # they were the most prominent thing in the picture and the grasp was behind them. The
        # extension is decoration either way: the physics cable begins at a via point on the
        # palm, which is the fixed root.
        arm = Forearm(scene, hand) if a.view else None
        view = RopeView(scene, hand)
        if a.view:
            pos, tgt = VIEWS["3q"]
            cam.position.set(*pos)
            cam.look_at(*tgt)
            controls = tp.OrbitControls(cam, canvas)
            controls.target = tp.Vector3(*tgt)
            controls.enable_damping = True
        else:
            dress(hand, mesh)
            xray(view)
            # The pull is the whole point of the second half of the episode and it is invisible
            # -- an 8 N force on a 30 g object is a picture of nothing. 1 cm per newton makes it
            # the same order of size as the hand, so a still shows what is being resisted.
            arrow = PullArrow(scene)
            legend = None
            if not a.no_hud:
                font = tp.FontLoader().default_font()
                hud = tp.TextSprite(font)
                hud.set_horizontal_alignment(tp.HorizontalAlignment.Center)
                hud.set_color(tp.Color(0x1A1F28))
                hud.set_world_scale(0.0075 * a.hud_scale)
                scene.add(hud)
                legend = Legend(scene, font, meta["t_max"], SHOT_TARGET, axis,
                                render_w / render_h, scale=a.hud_scale,
                                caption=f"{a.object}  {body.mass*1000:.0f} g  friction {friction:.2f}")
            rig = ShotRig(cam, lights, hud, SHOT_TARGET, axis, legend)
    for d in (a.shots, a.film):
        if d:
            os.makedirs(d, exist_ok=True)

    state = {"i": 0}

    def control():
        i = state["i"]
        nonlocal pull_dir, pull, tension, g_now, theta_now
        if i % every == 0:
            d = rng.normal(size=3)
            pull_dir = d / max(np.linalg.norm(d), 1e-6)
        t = i * dt
        k = max(0.0, (t - pull_start) / (pull_full - pull_start))
        pull = pull_dir * min(pull_cap, meta["pull_max"] * k)
        if t_drop is not None:
            # The rig stops pulling once the object is out of the hand. The object still leaves
            # fast: 26 N on 35 g is 750 m/s^2, and by the time the offset crosses the drop
            # limit it is doing ~25 m/s, gone in a frame. The tail of the take is the empty
            # hand with the pull it let go at on the HUD.
            pull = np.zeros(3)
        theta_now, g_now = gravity(t)
        world.set_gravity(tp.Vector3(*g_now))

        obs = torch.from_numpy(observe(meta, hand, art, mesh, target, tension, pull, g_now))[None]
        if norm is not None:
            obs = norm.norm(obs)
        act = ac.act_mean(obs)[0].clamp(-1.0, 1.0).numpy()
        cmd = meta["t_max"] * 0.5 * (act + 1.0)
        tension = (1.0 - alpha) * tension + alpha * cmd
        for n, T in zip(names, tension):
            hand.cables[n].set_tension(float(T))
        if view is not None:
            tint(view, names, tension, meta["t_max"])
        for _ in range(sub):
            body.add_force(tp.Vector3(*pull))        # re-applied: PhysX clears it every step
            world.step(dt / sub)
        state["i"] = i + 1
        return np.linalg.norm(np.array([mesh.position.x, mesh.position.y,
                                        mesh.position.z]) - target)

    def pull_now():
        """The pull on the HUD and in the log: frozen at the value it let go at."""
        return pull_at_drop if t_drop is not None else float(np.linalg.norm(pull))

    def status(t, d):
        return (f"t {t:4.2f} s   |d| {d*1000:6.1f} mm   turned {math.degrees(theta_now):4.0f} deg"
                f"   pull {pull_now():4.2f} N   mean tension {tension.mean():5.2f} N")

    if a.view:
        def loop():
            d = control()
            view.update(arm.update())
            controls.update()
            renderer.render(scene, cam)
            if state["i"] % 30 == 0:
                print("  " + status(state["i"] * dt, d))
        canvas.animate(loop)
        return

    print(f"policy {a.checkpoint} on the CPU hand with its real cables: {a.object} "
          f"dims {tuple(round(v*1000, 1) for v in dims if v > 0)} mm, "
          f"{body.mass*1000:.1f} g ({density:.0f} kg/m^3), friction {friction:.2f}; "
          f"roll {roll['ang_deg']:.0f} deg about ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}) "
          f"from {roll['t0']:.1f} s over {roll['dur']:.1f} s, held {roll['hang']:.1f} s; "
          f"{seconds:.1f} s at {hz} Hz")
    worst = 0.0
    for i in range(n_steps):
        d = control()
        worst = max(worst, d)
        t = (i + 1) * dt
        if t_drop is None and d >= meta["drop_dist"]:
            t_drop, pull_at_drop = t, float(np.linalg.norm(pull))
        if (i + 1) % 30 == 0:
            print("  " + status(t, d))
        if scene is not None:
            # Rebuilt from the LIVE link poses every step, exactly as the CPU demo's rope view
            # does, so what is in the still is what is pulling -- not a cable drawn from the
            # routing and hoping the two agree.
            view.update(arm.update() if arm is not None else None)
            if arrow is not None:
                p = np.array([mesh.position.x, mesh.position.y, mesh.position.z])
                f = pull_now()
                arrow.update(p, pull)
                rig.set(theta_now)
                if hud is not None:
                    hud.set_text(f"t {t:4.2f} s    turned {math.degrees(theta_now):3.0f} deg    "
                                 f"pull {f:4.1f} N    "
                                 f"{'held' if d < meta['drop_dist'] else 'DROPPED'}")
            if (i + 1) in shot_frame:
                renderer.render(scene, cam)
                fn = os.path.join(a.shots, f"{a.object}_t{shot_frame[i+1]:.1f}.png")
                renderer.save_frame(fn)
                print(f"    wrote {fn}   |d| {d*1000:.1f} mm   turned "
                      f"{math.degrees(theta_now):.0f} deg   pull {np.linalg.norm(pull):.2f} N")
            if a.film:
                renderer.render(scene, cam)
                renderer.save_frame(os.path.join(a.film, f"frame_{i+1:04d}.png"))
        if a.until_drop is not None and t_drop is not None and t >= t_drop + a.until_drop:
            break
    n_done = state["i"]
    if a.film:
        contact_sheet(a.film, contact_times(roll, n_done * dt, t_drop, pull_full), hz, n_done)
    held = worst < meta["drop_dist"]
    print(f"  worst offset {worst*1000:.1f} mm against a {meta['drop_dist']*1000:.0f} mm "
          f"limit -> {'HELD' if held else 'DROPPED'}")
    if pull_cap > meta["pull_max"]:
        if t_drop is None:
            print(f"  pull to failure: held to the {pull_cap:.1f} N cap over {n_done * dt:.2f} s "
                  f"(trained to {meta['pull_max']:.0f} N)")
        else:
            print(f"  pull to failure: dropped at {pull_at_drop:.2f} N, t {t_drop:.2f} s "
                  f"(trained to {meta['pull_max']:.0f} N)")


if __name__ == "__main__":
    main()
