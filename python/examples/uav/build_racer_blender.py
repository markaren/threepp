"""
build_racer_blender -- the 3D model of the racing quad (racer_spec.json), built in Blender and
exported as racer.glb.

    blender --background --factory-startup --python build_racer_blender.py -- --spec racer_spec.json --out racer.glb

A 500 mm class X frame (motor axes 0.5 m apart on the diagonal), 16 mm round carbon tube arms clamped
between a 2 mm bottom plate and a 2.5 mm top plate, a white cover with soft edges on the top plate and
four blue lights (paint, not emitters) at its base corners, M3 standoffs and button-head screws,
anodised aluminium motor bells with a lathe finish, black prop hubs and silver spinner nuts, a LiPo
under the bottom plate. Ten inch two-blade props are modelled (material 'prop_blade'); a scene hides
them and draws overlay discs instead (rigs/racer_rig.py, Visual).

Modelled at that size, +Y forward, +X right, +Z up, the origin on the arm tubes' axis plane at the
frame's centre; the root node holds the spec's model.drawn_scale and a quarter turn, so the .glb is
the examples' X forward, Y up, Z right, in metres.

Unlike the other generators here it works in Blender's own bmesh and curves, for the bevels and the
plates' holes, so there is no --check without Blender: it fails instead if a rotor leaves the spec's.
The textures are made with numpy (Blender's Python has no Pillow): a 2x2 twill carbon weave (colour,
normal, packed occlusion/roughness/metal) and a radial brushed finish for the motor bells.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
TEX = Path()                   # the textures' folder, for the build (main)

CARBON_TILE_M = 0.036          # one carbon texture tile on the tubes (16 tow cells of 2.25 mm)
PLATE_TILE_M = 0.064           # on the plates (4 mm cells): finer moires on a plate seen edge-on
DIAG = 0.50                    # motor to motor across the diagonal
ARM_R = DIAG / 2
TUBE_R = 0.008
PROP_R = 0.127                 # 10 inch
BOT_Z = (-0.012, -0.010)       # bottom plate, under the tubes
TOP_Z = (0.010, 0.0125)        # top plate, over the tubes
MOTOR_SPIN = (("fr", 45.0, 1.0), ("fl", 135.0, -1.0), ("rl", 225.0, 1.0), ("rr", 315.0, -1.0))  # +1 = CCW from above


# ---------------------------------------------------------------- textures
SIZE = 1024
CELLS = 16                     # tow cells across a tile: CARBON_TILE_M / CELLS = 2.25 mm, a 6K tow


def write_png(path: Path, rgb: np.ndarray) -> None:
    """An (h, w, 3) uint8 array as a PNG."""
    h, w = rgb.shape[:2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, -1)], 1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _noise1d(n: int, rng: np.random.Generator, smooth: int) -> np.ndarray:
    v = rng.standard_normal(n + smooth)
    k = np.hanning(smooth + 2)[1:-1]
    k /= k.sum()
    return np.convolve(v, k, mode="valid")[:n]


def carbon() -> None:
    rng = np.random.default_rng(7)
    px = SIZE // CELLS
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    i, j = x // px, y // px
    u, v = (x % px + 0.5) / px, (y % px + 0.5) / px
    phase = (i - j) % 4
    warp = phase < 2                                   # fibres along v (image y) on top
    # Position along the two-cell float of the tow on top (0 at one end, 1 at the other).
    w = np.where(warp, (1 - phase + v) / 2.0, (phase - 2 + u) / 2.0)
    across = np.where(warp, u, v)
    cross = np.sqrt(np.clip(1.0 - (2.0 * across - 1.0) ** 2, 0.0, 1.0))
    along = np.sin(np.pi * np.clip(w, 0.0, 1.0))
    h = cross ** 0.55 * along ** 0.35                  # bulge of the tow; grooves where tows dive under

    # Fibre striation: fine lines along the fibre direction, a different random set per tow.
    stri_x = _noise1d(SIZE, rng, 3)
    stri_y = _noise1d(SIZE, rng, 3)
    stri = np.where(warp, stri_x[x], stri_y[y])
    stri = (stri - stri.mean()) / (stri.std() + 1e-6)
    tow_jitter = rng.normal(0.0, 1.0, (CELLS, CELLS))[j, i]

    # Albedo (sRGB): the tow whose fibres face the camera along the twill diagonal reads lighter;
    # isotropic shading cannot do the anisotropic sheen, so the stripe contrast lives in the colour.
    lo = np.array([13, 14, 16], np.float32)
    hi = np.array([78, 82, 90], np.float32)
    k = np.where(warp, 0.80, 0.16) + 0.05 * stri + 0.04 * tow_jitter
    k = np.clip(k * (0.55 + 0.45 * h), 0.0, 1.0)
    groove = 1.0 - np.clip(h * 3.0, 0.0, 1.0)
    rgb = lo + (hi - lo) * k[..., None]
    rgb = rgb * (1.0 - 0.45 * groove[..., None])
    write_png(TEX / "carbon_col.png", np.clip(rgb, 0, 255).astype(np.uint8))

    # Normal (OpenGL, +Y up in the image = -row) from the height, wrapped at the tile edge.
    hh = h + 0.03 * stri * cross
    dx = (np.roll(hh, -1, 1) - np.roll(hh, 1, 1)) * 0.5
    dy = (np.roll(hh, -1, 0) - np.roll(hh, 1, 0)) * 0.5
    s = 2.2
    n = np.stack([-dx * s, dy * s, np.ones_like(hh)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    write_png(TEX / "carbon_nrm.png", ((n * 0.5 + 0.5) * 255).round().astype(np.uint8))

    # Packed R occlusion, G roughness, B metal: glossy resin on the tows, a little rougher in the grooves.
    # The tows facing along the twill catch the sky (glossy), the crossing ones scatter it: real carbon's
    # anisotropic sheen, faked per tow because the engine shades isotropically.
    rough = np.clip(np.where(warp, 0.15, 0.42) + 0.06 * groove + 0.015 * stri, 0.10, 0.6)
    occ = np.clip(0.55 + 0.45 * h, 0.0, 1.0)
    orm = np.stack([occ, rough, np.zeros_like(rough)], -1)
    write_png(TEX / "carbon_orm.png", (orm * 255).round().astype(np.uint8))


def brushed() -> None:
    """Concentric turning marks for the bell tops (a lathe finish): roughness and a faint normal."""
    rng = np.random.default_rng(11)
    n = 512
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n / 2)
    rings = _noise1d(2048, rng, 2)
    idx = np.clip((r * 2047).astype(int), 0, 2047)
    g = rings[idx]
    g = (g - g.mean()) / (g.std() + 1e-6)
    rough = np.clip(0.30 + 0.05 * g, 0.15, 0.5)
    orm = np.stack([np.ones_like(rough), rough, np.ones_like(rough)], -1)
    write_png(TEX / "bell_orm.png", (orm * 255).round().astype(np.uint8))


# ---------------------------------------------------------------- materials
def _img(name: str, non_color: bool):
    im = bpy.data.images.load(str((TEX / name).resolve()))
    if non_color:
        im.colorspace_settings.name = "Non-Color"
    return im


def _orm_group():
    g = bpy.data.node_groups.get("glTF Material Output")
    if g is None:
        g = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
        g.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    return g


def material(name: str, color: tuple, rough: float, metal: float = 0.0, col_tex: str | None = None,
             nrm_tex: str | None = None, orm_tex: str | None = None, nrm_strength: float = 1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if col_tex:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(col_tex, False)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if nrm_tex:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(nrm_tex, True)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = nrm_strength
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if orm_tex:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(orm_tex, True)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(t.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
        grp = nt.nodes.new("ShaderNodeGroup")
        grp.node_tree = _orm_group()
        nt.links.new(sep.outputs["Red"], grp.inputs["Occlusion"])
    return m


def srgb(hexv: int) -> tuple:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return tuple(lin(((hexv >> s) & 255) / 255.0) for s in (16, 8, 0))


# ---------------------------------------------------------------- mesh helpers
def link(obj, parent=None):
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def mesh_obj(name: str, bm: bmesh.types.BMesh, mat, parent=None, smooth_angle: float | None = 35.0):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    if smooth_angle is not None:
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(smooth_angle))
    return link(bpy.data.objects.new(name, me), parent)


def bake_modifiers(obj) -> None:
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    bpy.data.meshes.remove(old)


def rounded_box(name: str, size: tuple, radius: float, mat, loc: tuple, parent=None, segments: int = 4):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=radius, segments=segments, profile=0.5, affect="EDGES",
                    clamp_overlap=True)
    return mesh_obj(name, bm, mat, parent, smooth_angle=50.0)


def cylinder(name: str, r: float, z0: float, z1: float, mat, loc=(0.0, 0.0), seg: int = 40, parent=None,
             bevel: float = 0.0, r_top: float | None = None):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r,
                          radius2=r if r_top is None else r_top, depth=z1 - z0)
    bmesh.ops.translate(bm, vec=Vector((loc[0], loc[1], 0.5 * (z0 + z1))), verts=bm.verts)
    if bevel > 0.0:
        rims = [e for e in bm.edges if len(e.link_faces) == 2 and abs(e.link_faces[0].normal.z - e.link_faces[1].normal.z) > 0.5]
        bmesh.ops.bevel(bm, geom=rims, offset=bevel, segments=3, profile=0.5, affect="EDGES", clamp_overlap=True)
    return mesh_obj(name, bm, mat, parent, smooth_angle=40.0)


def dome(name: str, r: float, z0: float, mat, loc=(0.0, 0.0), parent=None, squash: float = 1.0,
         down: bool = False):
    """A half sphere standing on z0 (hanging under it with down=True), open at its base."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=r)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -1e-6], context="VERTS")
    for v in bm.verts:
        v.co.z *= -squash if down else squash
    if down:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bmesh.ops.translate(bm, vec=Vector((loc[0], loc[1], z0)), verts=bm.verts)
    return mesh_obj(name, bm, mat, parent, smooth_angle=70.0)


def planar_uv(obj, tile: float, angle_deg: float = 0.0) -> None:
    """Top-down UVs in metres / tile (plates): the weave lies flat on both faces."""
    me = obj.data
    while me.uv_layers:                     # a converted curve brings its own UVs along the outline
        me.uv_layers.remove(me.uv_layers[0])
    uv = me.uv_layers.new(name="UVMap")
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for loop in me.loops:
        co = me.vertices[loop.vertex_index].co
        x, y = co.x * ca - co.y * sa, co.x * sa + co.y * ca
        uv.data[loop.index].uv = (x / tile + 0.37, y / tile + 0.11)


def tube(name: str, r: float, length: float, mat, parent=None, seg: int = 32, rings: int = 24):
    """An open tube along +Y from 0 to length, with UVs that wrap the weave round it (a seam column)."""
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    circ = 2 * math.pi * r
    grid = []
    for k in range(rings + 1):
        y = length * k / rings
        row = []
        for s in range(seg + 1):
            a = 2 * math.pi * s / seg
            row.append(bm.verts.new((r * math.cos(a), y, r * math.sin(a))))
        grid.append(row)
    for k in range(rings):
        for s in range(seg):
            f = bm.faces.new((grid[k][s], grid[k][s + 1], grid[k + 1][s + 1], grid[k + 1][s]))
            for loop in f.loops:
                kk = 0 if loop.vert in (grid[k][s], grid[k][s + 1]) else 1
                ss = s if loop.vert in (grid[k][s], grid[k + 1][s]) else s + 1
                loop[uvl].uv = ((ss / seg) * circ / CARBON_TILE_M, (length * (k + kk) / rings) / CARBON_TILE_M)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)   # welds the seam column; UVs stay per loop
    bm.normal_update()
    for f in bm.faces:                                      # outward
        c = f.calc_center_median()
        if f.normal.dot(Vector((c.x, 0.0, c.z))) < 0:
            f.normal_flip()
    return mesh_obj(name, bm, mat, parent, smooth_angle=60.0)


# ---------------------------------------------------------------- plates (2D curves with holes)
def rounded_poly(pts: list[tuple], radius: float, arc_seg: int = 6) -> list[tuple]:
    """A closed polygon with every corner filleted."""
    out = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = Vector(pts[i - 1]), Vector(pts[i]), Vector(pts[(i + 1) % n])
        d0, d1 = (p0 - p1).normalized(), (p2 - p1).normalized()
        half = math.acos(max(-1.0, min(1.0, d0.dot(d1)))) / 2
        t = radius / math.tan(half)
        a, b = p1 + d0 * t, p1 + d1 * t
        centre = p1 + (d0 + d1).normalized() * (radius / math.sin(half))
        a0 = math.atan2(a.y - centre.y, a.x - centre.x)
        a1 = math.atan2(b.y - centre.y, b.x - centre.x)
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        for k in range(arc_seg + 1):
            ang = a0 + da * k / arc_seg
            out.append((centre.x + radius * math.cos(ang), centre.y + radius * math.sin(ang)))
    return out


def slot(cx: float, cy: float, length: float, width: float, angle_deg: float = 90.0, seg: int = 8) -> list[tuple]:
    """A stadium-shaped slot (a round hole when length == width)."""
    r = width / 2
    half = max(length / 2 - r, 0.0)
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    pts = []
    for end, a0 in ((half, -math.pi / 2), (-half, math.pi / 2)):
        for k in range(seg + 1):
            a = a0 + math.pi * k / seg
            x, y = end + r * math.cos(a), r * math.sin(a)
            pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def plate(name: str, outline: list[tuple], holes: list[list[tuple]], z0: float, z1: float, mat, parent=None):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    edge = 0.0004
    cu.extrude = max((z1 - z0) / 2 - edge, 1e-5)
    cu.bevel_depth = edge
    cu.bevel_resolution = 2
    for poly in [outline] + holes:
        sp = cu.splines.new("POLY")
        sp.points.add(len(poly) - 1)
        for p, (x, y) in zip(sp.points, poly):
            p.co = (x, y, 0.0, 1.0)
        sp.use_cyclic_u = True
    tmp = link(bpy.data.objects.new(name + "_curve", cu))
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    me.transform(Matrix.Translation((0.0, 0.0, 0.5 * (z0 + z1))))
    me.materials.clear()
    me.materials.append(mat)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(40.0))
    obj = link(bpy.data.objects.new(name, me), parent)
    planar_uv(obj, PLATE_TILE_M)
    return obj


def mirror4(x: float, y: float) -> list[tuple]:
    return [(x, y), (-x, y), (-x, -y), (x, -y)]


# ---------------------------------------------------------------- the build
def build(scale: float) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    M = {
        "carbon": material("carbon", srgb(0x26282c), 0.24, 0.0, "carbon_col.png", "carbon_nrm.png", "carbon_orm.png", 1.0),
        "cover": material("cover_white", srgb(0xe9e9e5), 0.42),
        "led": material("led_blue", srgb(0x2f66ff), 0.28),
        "alu_black": material("alu_black", srgb(0x1f2023), 0.38, 0.75),
        "bell": material("bell_silver", srgb(0xc6cad0), 0.30, 1.0, None, None, "bell_orm.png"),
        "steel": material("steel", srgb(0xb4b7bb), 0.26, 1.0),
        "spinner": material("spinner", srgb(0xc9d5e8), 0.24, 0.55),
        "windings": material("windings", srgb(0x5a3420), 0.45, 0.6),
        "hub": material("prop_hub", srgb(0x18191b), 0.45),
        "blade": material("prop_blade", srgb(0x1b1c1e), 0.40),
        "battery": material("battery", srgb(0x2b2d31), 0.55),
        "strap": material("strap", srgb(0x141416), 0.85),
        "pcb": material("pcb", srgb(0x1d3527), 0.5),
        "label": material("battery_label", srgb(0xb8452a), 0.5),
    }
    root = link(bpy.data.objects.new("racer", None))
    root.scale = (scale, scale, scale)
    root.rotation_euler = (0.0, 0.0, -math.pi / 2)  # forward +Y to +X: the exporter makes that glTF's X forward, Z right

    # Top plate: wide, chamfered where the arms leave it, lightened by slots, two strap slots behind the cover.
    hx, yf, yr = 0.072, 0.078, -0.086
    top_outline = rounded_poly([(hx - 0.032, yf), (-hx + 0.032, yf), (-hx, yf - 0.032), (-hx, yr + 0.030),
                                (-hx + 0.030, yr), (hx - 0.030, yr), (hx, yr + 0.030), (hx, yf - 0.032)], 0.006)
    holes = []
    for sx in (-1, 1):
        for cy in (-0.040, -0.006, 0.028):
            holes.append(slot(sx * 0.054, cy, 0.022, 0.0048, 90.0 - sx * 8.0))
        for cy in (-0.058, 0.050):
            holes.append(slot(sx * 0.036, cy, 0.0065, 0.0065))
    holes.append(slot(0.0, -0.030, 0.032, 0.0055, 0.0))
    holes.append(slot(0.0, -0.062, 0.040, 0.0055, 0.0))
    screw_pts = mirror4(0.062, 0.038) + [(0.0, 0.066), (0.0, -0.076)]
    arm_screws = []
    for _key, ang, _s in MOTOR_SPIN:
        a = math.radians(ang)
        for rr in (0.054, 0.072):
            for side in (-1, 1):
                px, py = rr * math.cos(a), rr * math.sin(a)
                arm_screws.append((px - side * 0.0085 * math.sin(a), py + side * 0.0085 * math.cos(a)))
    for p in screw_pts + arm_screws:
        holes.append(slot(p[0], p[1], 0.0032, 0.0032, seg=6))
    plate("top_plate", top_outline, holes, TOP_Z[0], TOP_Z[1], M["carbon"], root)

    bhx, byf, byr = 0.074, 0.084, -0.104
    bot_outline = rounded_poly([(bhx - 0.034, byf), (-bhx + 0.034, byf), (-bhx, byf - 0.034), (-bhx, byr + 0.026),
                                (-bhx + 0.026, byr), (bhx - 0.026, byr), (bhx, byr + 0.026), (bhx, byf - 0.034)], 0.006)
    bholes = [slot(sx * 0.030, cy, 0.030, 0.006, 0.0) for sx in (-1, 1) for cy in (-0.070, 0.060)]
    plate("bottom_plate", bot_outline, bholes, BOT_Z[0], BOT_Z[1], M["carbon"], root)

    # Standoffs and button-head screws (top and bottom) at the plate corners.
    for (x, y) in screw_pts:
        cylinder("standoff", 0.0026, BOT_Z[1], TOP_Z[0], M["alu_black"], (x, y), seg=12, parent=root)
        for z0, sgn in ((TOP_Z[1], 1.0), (BOT_Z[0], -1.0)):
            dome("screw", 0.0028, z0, M["steel"], (x, y), parent=root, squash=0.55, down=sgn < 0)
    for (x, y) in arm_screws:
        dome("screw_arm", 0.0026, TOP_Z[1], M["alu_black"], (x, y), parent=root, squash=0.55)

    # The flight-controller stack between the plates (seen through the slots).
    rounded_box("fc_stack", (0.038, 0.038, 0.012), 0.0015, M["alu_black"], (0.0, -0.004, 0.0), root, 2)
    rounded_box("fc_board", (0.036, 0.036, 0.0016), 0.0004, M["pcb"], (0.0, -0.004, 0.0072), root, 1)

    # The white cover with soft edges, the four blue lights at its base corners.
    cw, cl, ch, cy0 = 0.064, 0.074, 0.020, 0.020
    rounded_box("cover", (cw, cl, ch), 0.0055, M["cover"], (0.0, cy0, TOP_Z[1] + ch / 2), root, 5)
    for sx in (-1, 1):
        for sy in (-1, 1):
            rounded_box("led", (0.0095, 0.0095, 0.0085), 0.0032, M["led"],
                        (sx * (cw / 2 + 0.0015), cy0 + sy * (cl / 2 + 0.0015), TOP_Z[1] + 0.0043), root, 3)

    # The LiPo and its strap under the bottom plate.
    rounded_box("battery", (0.044, 0.140, 0.032), 0.004, M["battery"], (0.0, -0.010, BOT_Z[0] - 0.016 - 0.0005), root, 3)
    rounded_box("battery_label", (0.0448, 0.050, 0.020), 0.003, M["label"], (0.0, -0.030, BOT_Z[0] - 0.0165), root, 2)
    for y in (-0.050, 0.035):
        rounded_box("strap", (0.050, 0.020, 0.037), 0.002, M["strap"], (0.0, y, BOT_Z[0] - 0.0175), root, 2)

    rotors = {}
    for key, ang, spin in MOTOR_SPIN:
        a = math.radians(ang)
        mx, my = ARM_R * math.cos(a), ARM_R * math.sin(a)
        # Arm tube from under the plate to the motor clamp.
        t0, t1 = 0.030, ARM_R + 0.012
        arm = tube(f"arm_{key}", TUBE_R, t1 - t0, M["carbon"], root)
        arm.rotation_euler = (0.0, 0.0, a - math.pi / 2)
        arm.location = (t0 * math.cos(a), t0 * math.sin(a), 0.0)
        # Centre clamp block between the plates.
        blk = rounded_box(f"clamp_c_{key}", (0.024, 0.030, 0.019), 0.002, M["alu_black"], (0.0, 0.0, 0.0), root, 2)
        blk.rotation_euler = (0.0, 0.0, a - math.pi / 2)
        blk.location = (0.064 * math.cos(a), 0.064 * math.sin(a), 0.0)

        motor = link(bpy.data.objects.new(f"motor_{key}", None), root)
        motor.location = (mx, my, 0.0)
        mclamp = rounded_box(f"clamp_m_{key}", (0.026, 0.030, 0.022), 0.0025, M["alu_black"], (0.0, 0.0, 0.0), motor, 2)
        mclamp.rotation_euler = (0.0, 0.0, a - math.pi / 2)
        z = 0.011
        mount = cylinder(f"mount_{key}", 0.0255, z, z + 0.0025, M["carbon"], seg=40, parent=motor, bevel=0.0006)
        planar_uv(mount, CARBON_TILE_M)
        z += 0.0025
        cylinder(f"stator_base_{key}", 0.0225, z, z + 0.0035, M["alu_black"], seg=48, parent=motor, bevel=0.0007)
        z += 0.0035
        cylinder(f"windings_{key}", 0.0205, z, z + 0.0015, M["windings"], seg=40, parent=motor)
        z += 0.0015
        bell_z0, bell_z1 = z, z + 0.0125
        bell = cylinder(f"bell_{key}", 0.0240, bell_z0, bell_z1, M["bell"], seg=48, parent=motor, bevel=0.0018)
        planar_uv(bell, 0.049)
        # Dark vents in the bell top: six sectors, a hair above the face.
        bm = bmesh.new()
        for k in range(6):
            a0 = 2 * math.pi * k / 6 + 0.18
            a1 = a0 + 0.62
            ring = []
            for s in range(7):
                aa = a0 + (a1 - a0) * s / 6
                ring.append((math.cos(aa), math.sin(aa)))
            outer = [bm.verts.new((0.0180 * c, 0.0180 * s_, bell_z1 + 0.0002)) for c, s_ in ring]
            inner = [bm.verts.new((0.0110 * c, 0.0110 * s_, bell_z1 + 0.0002)) for c, s_ in ring]
            for s in range(6):
                bm.faces.new((inner[s], inner[s + 1], outer[s + 1], outer[s]))
        mesh_obj(f"vents_{key}", bm, M["windings"], motor, smooth_angle=None)
        z = bell_z1
        cylinder(f"prop_hub_{key}", 0.0125, z, z + 0.0075, M["hub"], seg=32, parent=motor, bevel=0.0012)
        blade_z = z + 0.0037
        z += 0.0075
        cylinder(f"spinner_base_{key}", 0.0090, z, z + 0.003, M["spinner"], seg=6, parent=motor, bevel=0.0004)
        z += 0.003
        dome(f"spinner_{key}", 0.0092, z, M["spinner"], parent=motor, squash=1.05)
        # Two blades (hidden in the scene: the overlay discs stand in for them).
        bm = bmesh.new()
        n = 12
        for b in (0, 1):
            verts = []
            for k in range(n + 1):
                f = k / n
                r = 0.012 + (PROP_R - 0.012) * f
                chord = 0.022 * (1 - f) + 0.011 * f + 0.008 * math.sin(math.pi * min(f * 1.6, 1.0))
                twist = math.radians(24.0 * (1 - f) + 9.0)
                le = Vector((r, -0.5 * chord * math.cos(twist), 0.5 * chord * math.sin(twist) * spin))
                te = Vector((r, 0.5 * chord * math.cos(twist), -0.5 * chord * math.sin(twist) * spin))
                rot = Matrix.Rotation(math.pi * b, 3, "Z")
                verts.append((bm.verts.new(rot @ le + Vector((0, 0, blade_z))), bm.verts.new(rot @ te + Vector((0, 0, blade_z)))))
            for k in range(n):
                bm.faces.new((verts[k][0], verts[k + 1][0], verts[k + 1][1], verts[k][1]))
        mesh_obj(f"prop_blades_{key}", bm, M["blade"], motor, smooth_angle=None)
        rotors[key] = {"pos": [my * scale, blade_z * scale, mx * scale], "spin": spin}   # X forward, Y up, Z right
    return rotors


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = {"--spec": str(HERE / "racer_spec.json"), "--out": str(HERE / "racer.glb")}
    for k, v in zip(args[::2], args[1::2]):
        opt[k] = v
    spec = json.loads(Path(opt["--spec"]).read_text(encoding="utf-8"))
    global TEX
    with tempfile.TemporaryDirectory() as tmp:
        TEX = Path(tmp)
        carbon()
        brushed()
        rotors = build(spec["model"]["drawn_scale"])
        for key, want in spec["layout"]["rotors"].items():
            got = rotors[key]
            off = max(abs(a - b) for a, b in zip(got["pos"], want["pos"]))
            if off > 5e-5 or got["spin"] != want["spin"]:
                raise SystemExit(f"racer: rotor {key} at {got}, the spec has {want}")
        path = os.path.abspath(opt["--out"])
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_tangents=False, export_yup=True,
                                  export_apply=True, export_cameras=False, export_lights=False,
                                  export_image_format="AUTO", export_vertex_color="MATERIAL")
    tris = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
    print(f"racer: wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB, {tris} faces)")


main()
