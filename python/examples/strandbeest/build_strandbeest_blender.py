"""
build_strandbeest_blender -- the Strandbeest's parts as one .glb of named objects (headless Blender).

    blender --background --factory-startup --python build_strandbeest_blender.py [-- --out strandbeest.glb]

Every length comes from ../rigs/strandbeest_rig.py (Jansen's numbers, the scale, the stations), so
the model and the physics cannot disagree. The parts, each with its origin where the rig places it:

    bar_b .. bar_k    one cane per bar of the linkage, along +X from its first joint, with a
                      lashing at each end and a wrap near it
    pin, foot         a hinge pin through a leg's three layers; a foot
    crank             the crankshaft with its throws at the stations' phases, about the axle
    frame             the rails the legs hang from, the ribs with the axle's bearings, the masts,
                      the top spar and the yard the sails hang from
    sail_0 ..         a billowed sail each (the rig draws its own cloth and takes their material)

Blender's frame here is X forward, Z up, Y along the axle; the glTF export turns it into the
rig's machine frame (x forward, y up, z along the axle). The cane's bark is a generated, tileable
image packed into the .glb.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))
import strandbeest_rig as LKG  # noqa: E402
from strandbeest_rig import HOLY, SCALE  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(HERE, "strandbeest.glb")
R = 0.016                 # cane radius
STATIONS, PITCH, PHASES = LKG.STATIONS, LKG.PITCH, LKG.PHASES_DEG
WEB = 0.16                # crank web offset from the station centre
A, Lh, M = HOLY["a"] * SCALE, HOLY["l"] * SCALE, HOLY["m"] * SCALE
HALF, MAST = LKG.HALF, LKG.MAST
SX, SY = LKG.SHOULDER

bpy.ops.wm.read_factory_settings(use_empty=True)


def mat(name, rgb, rough, metal=0.0, double=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    m.use_backface_culling = not double
    return m


def cane_image(name, tint, seed):
    """Tileable cane bark: fibres along the pole, blotches, a dark node ring with a pale collar."""
    import numpy as np
    rng = np.random.default_rng(seed)
    W, H = 256, 1024
    u = np.linspace(0, 2 * np.pi, W, endpoint=False)[None, :]
    v = np.linspace(0, 2 * np.pi, H, endpoint=False)[:, None]
    fib = sum(rng.uniform(0.3, 1.0) * np.sin(k * u + rng.uniform(0, 6.28) + 0.6 * np.sin(v * rng.integers(1, 4) + rng.uniform(0, 6.28)))
              for k in rng.integers(6, 60, 26)) / 10.0
    blot = sum(rng.uniform(0.4, 1.0) * np.sin(rng.integers(1, 5) * u + rng.integers(1, 9) * v + rng.uniform(0, 6.28)) for _ in range(14)) / 6.0
    speck = rng.normal(0, 1, (H, W))
    val = 1.0 + 0.10 * fib + 0.13 * blot + 0.035 * speck
    vv = (np.arange(H) / H)[:, None]
    for c in (0.0, 0.5):                                         # two nodes per tile
        d = np.minimum(np.abs(vv - c), 1 - np.abs(vv - c)) * H
        val = val * (1 - 0.55 * np.exp(-(d / 3.0) ** 2)) + 0.22 * np.exp(-((d - 9) / 5.0) ** 2)
    grey = np.clip(blot * 0.5 + 0.5, 0, 1)[..., None]            # sun-bleached patches lose colour
    rgb = np.array(tint)[None, None, :] * val[..., None]
    rgb = rgb * (1 - 0.25 * grey) + rgb.mean(-1, keepdims=True) * 1.15 * 0.25 * grey
    px = np.concatenate([np.clip(rgb, 0, 1) ** (1 / 2.2), np.ones((H, W, 1))], -1).astype(np.float32)
    img = bpy.data.images.new(name, W, H, alpha=False)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img


def cane(name, tint, rough, seed):
    m = mat(name, tint, rough)
    nt = m.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = cane_image(name + "_tex", tint, seed)
    nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


CANE = cane("cane", (0.62, 0.47, 0.22), 0.55, 1)
CANE_OLD = cane("cane_old", (0.42, 0.33, 0.20), 0.65, 2)
FIT = mat("fitting", (0.13, 0.09, 0.06), 0.85)
TAPE = mat("tape", (0.50, 0.40, 0.26), 0.95)
STEEL = mat("steel", (0.45, 0.45, 0.46), 0.35, 1.0)
RUBBER = mat("rubber", (0.03, 0.03, 0.03), 0.85)
SAIL = mat("sail", (0.88, 0.84, 0.74), 0.9, double=True)
ROPE = mat("rope", (0.55, 0.45, 0.30), 0.9)


def tube(p0, p1, r, material, verts=16):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d.length, location=(p0 + p1) / 2)
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    uv = o.data.uv_layers.active.data          # sides: v along the pole in tiles of 0.9 m; caps: one texel
    off = (hash((round(p0.x, 3), round(p0.y, 3), round(p0.z, 3))) % 97) / 97.0
    for poly in o.data.polygons:
        cap = len(poly.loop_indices) > 4
        for li in poly.loop_indices:
            if cap:
                uv[li].uv = (0.5, 0.01)
            else:
                uv[li].uv = (uv[li].uv[0] + off, (uv[li].uv[1] - 0.5) * 2.0 * d.length / 0.9 + off)
    return o


def ball(p, r, material, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=r, location=p)
    o = bpy.context.active_object
    o.scale = scale
    o.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return o


def join(name, objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if len(objs) > 1:
        bpy.ops.object.join()
    o = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    o.name = name
    o.data.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    return o


def bar(name, L):
    """Conduit along +X with a cross sleeve at each end (the hinge eye) and two tape wraps."""
    parts = [tube((0.02, 0, 0), (L - 0.02, 0, 0), R, CANE)]
    for x in (0.0, L):
        parts.append(tube((x, -0.024, 0), (x, 0.024, 0), R * 1.45, FIT))
    for x in (0.075, L - 0.075):
        parts.append(tube((x - 0.02, 0, 0), (x + 0.02, 0, 0), R * 1.12, TAPE))
    return join(name, parts)


for k in "bcdefghijk":
    bar(f"bar_{k}", HOLY[k] * SCALE)

# hinge pin through the three layers of one leg, and the foot
join("pin", [tube((0, -0.085, 0), (0, 0.085, 0), 0.0075, STEEL, 10),
             tube((0, -0.092, 0), (0, -0.085, 0), 0.013, STEEL, 10), tube((0, 0.085, 0), (0, 0.092, 0), 0.013, STEEL, 10)])
join("foot", [ball((0, 0, 0.012), 0.034, RUBBER, (1.25, 1.0, 0.62)), tube((0, 0, 0.02), (0, 0, 0.07), R * 1.2, FIT)])

# crankshaft: rotates about the axle (Blender Y) through the origin
parts = []
ys = [(i - (STATIONS - 1) / 2) * PITCH for i in range(STATIONS)]
prev = -HALF
for y0, ph in zip(ys, PHASES):
    a = math.radians(ph)
    c = Vector((M * math.cos(a), 0.0, M * math.sin(a)))
    parts.append(tube((0, prev, 0), (0, y0 - WEB, 0), R * 1.15, CANE_OLD))
    for s in (-1, 1):
        y = y0 + s * WEB
        parts.append(tube((0, y, 0), (c.x, y, c.z), R * 1.15, CANE_OLD))
        parts.append(ball((0, y, 0), R * 1.7, FIT))
        parts.append(ball((c.x, y, c.z), R * 1.7, FIT))
    parts.append(tube((c.x, y0 - WEB, c.z), (c.x, y0 + WEB, c.z), R * 1.0, CANE_OLD))
    prev = y0 + WEB
parts.append(tube((0, prev, 0), (0, HALF, 0), R * 1.15, CANE_OLD))
join("crank", parts)

# frame: the two rails the legs hang from, ribs with the axle bearings, masts and spars for the sails
parts = []
for s in (-1, 1):
    parts.append(tube((s * A, -HALF - 0.1, -Lh), (s * A, HALF + 0.1, -Lh), R * 1.1, CANE))
    parts.append(tube((s * SX, -HALF - 0.1, SY), (s * SX, HALF + 0.1, SY), R, CANE))      # shoulder rails
rib_y = [-HALF] + [(ys[i] + ys[i + 1]) / 2 for i in range(STATIONS - 1)] + [HALF]
for n, y in enumerate(rib_y):
    for s in (-1, 1):
        parts.append(tube((0, y, 0), (s * A, y, -Lh), R, CANE))
        parts.append(tube((s * A, y, -Lh), (s * SX, y, SY), R, CANE))
        parts.append(tube((s * SX, y, SY), (0, y, 0), R, CANE))
        parts.append(ball((s * A, y, -Lh), R * 1.6, FIT))
        parts.append(ball((s * SX, y, SY), R * 1.6, FIT))
    parts.append(tube((-SX, y, SY), (SX, y, SY), R, CANE))
    parts.append(tube((0, y - 0.03, 0), (0, y + 0.03, 0), R * 2.0, FIT))                         # bearing
    if n % 2 == 0:                                                                               # mast
        parts.append(tube((0, y, SY), (0, y, MAST), R, CANE))
        parts.append(tube((-SX, y, SY), (0, y, MAST * 0.67), R * 0.8, CANE))
        parts.append(tube((SX, y, SY), (0, y, MAST * 0.67), R * 0.8, CANE))
parts.append(tube((0, -HALF - 0.25, MAST), (0, HALF + 0.25, MAST), R, CANE))                      # top spar
YARD_X = LKG.YARD_X
parts.append(tube((YARD_X, -HALF - 0.2, MAST - 0.04), (YARD_X, HALF + 0.2, MAST - 0.04), R * 0.9, CANE))
for n, y in enumerate(rib_y):
    if n % 2 == 0:
        parts.append(tube((0, y, MAST - 0.04), (YARD_X, y, MAST - 0.04), R * 0.8, CANE))
        parts.append(tube((0, y, MAST * 0.8), (YARD_X, y, MAST - 0.04), R * 0.7, CANE))
# guy ropes from the spar ends down to the rails
for sy in (-1, 1):
    for sx in (-1, 1):
        parts.append(tube((0, sy * (HALF + 0.25), MAST), (sx * A, sy * (HALF + 0.1), -Lh), 0.004, ROPE, 6))
join("frame", parts)


def sail(name, w, h):
    """A billowed panel hanging from its origin (the top spar): local X is downwind, -Z down."""
    nx, nz = 10, 14
    verts, faces = [], []
    for j in range(nz + 1):
        v = j / nz
        for i in range(nx + 1):
            u = i / nx
            belly = 0.11 * math.sin(math.pi * u) * math.sin(math.pi * min(1.0, v * 1.15)) ** 0.8
            taper = 1.0 - 0.18 * v
            verts.append((belly + 0.05 * v, (u - 0.5) * w * taper, -0.03 - v * h))
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.materials.append(SAIL)
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    # a batten along the foot of the sail
    b = tube((0.05, -w * 0.41, -0.03 - h), (0.05, w * 0.41, -0.03 - h), 0.008, CANE, 8)
    return join(name, [o, b])


SAIL_W = LKG.SAIL_W
for n in range(LKG.NSAIL):
    sail(f"sail_{n}", SAIL_W * 0.9, LKG.SAIL_H)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", export_yup=True, export_apply=True)
tris = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
print(f"strandbeest: wrote {OUT}: {[o.name for o in bpy.data.objects]} faces {tris}")
