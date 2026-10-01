"""Ålesund inner harbour: the Mariner leaves the basin past moored sjarks, out through the marks.

Phases A and B of plans/harbour-scene.md. Five sjarks lie along the north quay, MS Trollfjord
alongside the Cruise Pier mole's outer face, the marks at the basin mouth; the Mariner casts off
from the east end of the quay and runs out on her DP along a smooth route. Headless only.

    python harbour_scene.py --shot all [--out DIR] [--size 1600x900] [--terrain DIR] [--light bright|overcast]
    python harbour_scene.py --shot drone,sealevel,close,spar,bob
    python harbour_scene.py --film-test [--out DIR]     # 3 stills per cut + a contact sheet
    python harbour_scene.py --film [--out DIR]          # the film: frames at 30 fps, then an mp4
    python harbour_scene.py --labels DIR [--out DIR]    # phase C: the POV cuts as a LaRS split in DIR
                                                        # (harbour_labels.py), POV | mask film, gates

--terrain defaults to <repo>/geodata/aalesund; --out defaults to out/ beside this script.
A file that already exists in --out is never overwritten (use a new folder per iteration).

Frame: the pack's (x east, y up, z south, metres, origin the pack centre, sea level 0).
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXAMPLES))
sys.path.insert(0, os.path.dirname(EXAMPLES))                  # python/ (threepp)
sys.path.insert(0, EXAMPLES)                                   # examples/ (demo_common)
sys.path.insert(0, os.path.join(EXAMPLES, "turbine"))          # fleet_buoyancy, turbine_site (sky, tube)
sys.path.insert(0, HERE)

import threepp as tp
from demo_common import cli_arg, parse_size

from fleet_buoyancy import StripHull
from harbour_float import BuoyFloat, load_marks, sag_line
import turbine_site as ts

# --------------------------------------------------------------------------- #
#  The site, picked from the pack's data (geodata/aalesund: 2 m DTM heights.f32, OSM
#  buildings.json and landuse.json pier / breakwater polygons). The DTM's sea cells
#  (h <= 0.05) mapped against the pier polygons and building footprints show the town's
#  south harbour: a basin south of the centre, closed on the south by the Cruise Pier mole
#  (a 220 m land strip, x 165..385, z 345..405) and on the east by Buholmen (the OSM pier at
#  x 648, z 288..297), open to Nørvasundet between the two. The basin's NORTH quay (below
#  Keiser Wilhelms gate / Rådstugata) is dead straight: scanning each column south from the
#  town, the first sea cell sits at z 204 for x 320..380 and z 206 for x 390..470, with the
#  quay top 2.2..2.4 m above the sea 4 m inland; east of x 480 a notch and a small pier
#  jut out. That straight is the berth: 150 m, room for 6 sjarks of 11 m (and it is where the
#  town's sjarks do lie, below Keiser Wilhelms gate). The DTM has no soundings: GeoScene's
#  distance-to-shore bathymetry with shore_slope 3 / max_depth 40 (build()) puts the seabed at
#  -3.6 m under the berthed sjark's CoG (1.15 m draft) and 34..40 m in the basin and the mouth.
# --------------------------------------------------------------------------- #
QUAY_A = (325.0, 205.6)            # the quay face (x, z) at its west end ...
QUAY_B = (468.0, 207.6)            # ... and east end: where GeoScene's ground crosses sea level (v1 profile)
QUAY_TOP = 2.35                    # quay deck height above sea level (DTM, 4 m inland)
QUAY_HDG = -math.atan2(QUAY_B[1] - QUAY_A[1], QUAY_B[0] - QUAY_A[0])   # +X along the quay, east: -0.80 deg
FENDER_R = 0.18                    # rubber fenders on the quay face
BERTH_X = 400.0                    # the phase-A sjark's centre along the quay
# the mouth of the basin, and the route out (phase B drives it): off the berth, south through
# the mouth between the mole's east end (~385, 390) and Buholmen (~650, 345), out into the sound
MOUTH = (500.0, 372.0)
ROUTE = [(BERTH_X, 211.0), (445.0, 232.0), (490.0, 290.0), MOUTH, (520.0, 520.0), (600.0, 720.0)]
# the marks (IALA A, the red mark on the port hand coming IN, i.e. heading north into the basin
# the red is to the west): the lateral pair across the mouth, the special mark where the basin's
# yellow buoy lies, a south cardinal, a mooring buoy in the basin and two gillnet floats (blåser)
# off the sjarks, clear of the Mariner's turn. Phase B (a scene choice, not charted marks): the
# laterals close to a 30 m gate across her route at the mouth (15 m either side of it), and the
# cardinal sits 55 m past the gate, 18 m off her port hand as she clears it (phase A's spot off the
# mole tip is where Trollfjord's bow lies now), so each reads in her camera_main view.
MARKS = {
    "lateral_port": (486.9, 379.3),
    "lateral_stbd": (513.1, 364.7),
    "special": (470.0, 285.0),
    "cardinal_s": (512.0, 425.0),
}
EXTRA_FLOATS = [("mooring_buoy", "mooring_buoy", (352.0, 262.0)),
                ("blaase_a5", "blaase_1", (366.0, 227.0)),
                ("blaase_a3", "blaase_2", (389.0, 224.5))]
FOCUS = (450.0, 290.0)             # ocean vertex focus: the berth at 85 m, the mouth at 95 m

# ---- phase B: the boats along the quay straight. (x of the CoG along the quay, glb, side to the
# quay, a small yaw off the quay line in deg). Gaps of 8..12 m between hulls of 11 m, two lying
# starboard side to (bow west), the others port side to (bow east); five liveries.
SJARKS = [(336.0, "out/sjark_s2.glb", "port", 0.6),
          (355.5, "out/sjark_s4.glb", "stbd", -0.8),
          (377.0, "out/sjark_s3.glb", "port", 0.3),
          (BERTH_X, "out/sjark_s1.glb", "port", 0.0),      # phase A's boat
          (423.0, "sjark.glb", "stbd", 0.9)]
MARINER_X = 449.0                  # her berth near the east end: starboard side to, bow west
# her route out: ahead along the quay, a turn to port away from it, the special mark close on her
# port hand, the gate between the laterals, then west along the mole's outer face past Trollfjord
# (so her camera sees the ship). Waypoints (x, z); the berth is prepended in build().
MARINER_ROUTE = [(438.0, 212.8), (425.0, 223.0), (411.0, 240.0), (428.0, 262.0), (460.0, 300.0),
                 (500.0, 372.0), (494.0, 420.0), (458.0, 457.0), (392.0, 478.0), (300.0, 490.0),
                 (210.0, 492.0)]
V_BASIN, V_OUT = 2.5, 4.0          # 5 kn in the basin, 8 kn once past the mouth
T_CAST = 7.5                       # sim s (from the film's start) she casts off
T_RAMP = 6.0                       # s to reach basin speed
S_EASE = 35.0                      # m over which she eases up to V_OUT after the mouth

# --------------------------------------------------------------------------- #
#  Look
# --------------------------------------------------------------------------- #
SUN_DIR_A = np.array([-0.52, 0.46, 0.72])     # phase A: afternoon sun from the south-west, 27 deg up (overcast)
SUN_DIR_A /= np.linalg.norm(SUN_DIR_A)
_hz = SUN_DIR_A[[0, 2]] / np.linalg.norm(SUN_DIR_A[[0, 2]])
SUN_ELEV_B = math.radians(38.0)               # phase B: the same bearing (it lights the town front), 38 deg up
SUN_DIR_B = np.array([_hz[0] * math.cos(SUN_ELEV_B), math.sin(SUN_ELEV_B), _hz[1] * math.cos(SUN_ELEV_B)])
SUN_DIR = SUN_DIR_B
HAZE_B = (0.47, 0.59, 0.78)                   # a clear day's horizon: the sky's horizon and the air fog agree
FOG = 0.00012                                 # air medium sigma_t (/m): 4 % at 300 m, the town crisp; the far ring
                                              # beyond the sheet edge (4.5 km, ~40 % hazed) carries the horizon
MURK = 0.20                                   # the bed is not seen from the air
MURK_COLOUR = (0.10, 0.14, 0.125)             # harbour water from the air: dark green-grey
SHEET = 9000.0                                # one sheet over the whole pack, packed toward FOCUS
OCEAN_RES = 1024
T_SHOT = 30.0                                 # the still time (s of sea state)
BOB_S = 12.0                                  # the bob check's length, ending at T_SHOT


def z_face(x):
    t = (x - QUAY_A[0]) / (QUAY_B[0] - QUAY_A[0])
    return QUAY_A[1] + t * (QUAY_B[1] - QUAY_A[1])


def std_mat(rgb, rough=0.85, metal=0.0):
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(*rgb)
    m.roughness = rough
    m.metalness = metal
    return m


def shadows(o):
    if isinstance(o, tp.Mesh):
        o.cast_shadow = True
        o.receive_shadow = True


def bright_sky(sun_dir, w=2048, h=1024, haze=None, zenith=(0.03, 0.11, 0.40), cover=(0.64, 0.74)):
    """A clear-to-broken day over Ålesund: turbine_site.sky_env's cloud deck (the same fbm, lit
    flanks toward the sun) thinned to fair-weather cumulus in a blue sky, a clear horizon, the sun
    in a gap. Background and environment, so the water mirrors what the camera sees."""
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    d = np.empty((h, w, 3), np.float32)
    d[..., 0] = np.cos(elev)[:, None] * np.cos(az)[None, :]
    d[..., 1] = np.sin(elev)[:, None]
    d[..., 2] = np.cos(elev)[:, None] * np.sin(az)[None, :]
    y = d[..., 1]
    haze = np.float32(HAZE_B if haze is None else haze)
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.45
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.4
    blue = haze * (1.0 - up) + np.float32(zenith) * up
    col = np.where(y[..., None] >= 0.0, blue, haze * (1.0 - down) + np.float32([0.05, 0.07, 0.07]) * down)
    ang = np.arccos(np.clip(d @ np.asarray(sun_dir, np.float32), -1.0, 1.0))
    yy = np.maximum(y, 0.03)
    pu, pv = d[..., 0] / yy * 0.9, d[..., 2] / yy * 0.9
    n = ts._fbm(pu, pv, 5.0)
    sh = np.float32(sun_dir)[[0, 2]] / np.linalg.norm(np.float32(sun_dir)[[0, 2]])
    n_sun = ts._fbm(pu + 0.08 * sh[0], pv + 0.08 * sh[1], 5.0)
    cover = ts._smooth(cover[0], cover[1], n) * ts._smooth(0.02, 0.12, y) * (1.0 - np.exp(-(ang / math.radians(9.0)) ** 2))
    lit = np.clip(0.9 + 4.0 * (n - n_sun), 0.55, 1.35)
    thick = ts._smooth(0.66, 0.86, n)
    cloud = (np.float32([1.45, 1.43, 1.38]) * (1.0 - thick[..., None]) + np.float32([0.62, 0.66, 0.72]) * thick[..., None])
    cloud = cloud * lit[..., None] * (1.0 + 1.2 * np.exp(-(ang / math.radians(18.0)) ** 2))[..., None]
    col = col * (1.0 - cover[..., None]) + cloud * cover[..., None]
    gap = (1.0 - cover)[..., None]
    col += gap * ((np.exp(-(ang / math.radians(1.6)) ** 2) * 60.0
                   + np.exp(-(ang / math.radians(12.0)) ** 2) * 1.0)[..., None] * np.float32([1.0, 0.95, 0.86]))
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


def build(renderer, terrain_dir, light="bright"):
    import types
    S = types.SimpleNamespace()
    S.registry = []                    # phase C's labels: (node, lars_class, instance_name)
    scene = tp.Scene()
    if light == "overcast":            # phase A's look, kept for the A/B pair
        sun_dir, haze, sun_i = SUN_DIR_A, ts.HAZE, 2.4
        sky = ts.sky_env(sun_dir)
    else:
        sun_dir, haze, sun_i = SUN_DIR_B, HAZE_B, 5.6
        sky = bright_sky(sun_dir)
    scene.background = sky
    scene.environment = sky
    sun = tp.DirectionalLight(0xfff2e0, sun_i)
    sun.position.set(*(sun_dir * 1000.0))
    sun.cast_shadow = True
    scene.add(sun)
    scene.set_fog_exp2(tp.Color(*haze), FOG)

    # ---- the town
    t0 = time.perf_counter()
    # boats off: the procedural harbour boats would clash with the sjarks. scatter off: its
    # stones/tufts ring follows the camera, and a camera over water puts them on the seabed.
    # A harbour is quay walls, not beaches: shore_slope 3 with a 40 m floor puts ~4 m of water
    # under a keel 2.6 m off the quay face (the default 0.35 leaves 0.3 m: a sjark on a beach).
    geo = tp.GeoScene(terrain_dir, forest_focus=(FOCUS[0], 0.0, FOCUS[1]), scatter=False, boats=False,
                      shore_slope=3.0, max_depth=40.0)
    scene.add(geo)
    print(f"[harbour] terrain {os.path.basename(os.path.normpath(terrain_dir))}: {geo.pack_world_size:.0f} m, "
          f"loaded in {time.perf_counter() - t0:.1f} s; {geo.stats}")

    # ---- the sea: calm, sheltered by the mole (a light breeze over a few km of fetch)
    ocean = tp.Ocean(size=SHEET, resolution=OCEAN_RES, wind_speed=5.0, wind_theta=math.radians(160.0),
                     choppiness=0.7, fft_size=512, fetch=4000.0)
    ocean.params.foam_amount = 0.0              # before the first frame: the default (size/300) seeds a foam buffer
    ocean.params.tile_size_0 = 80.0
    ocean.params.tile_size_1 = 17.3
    ocean.params.tile_size_2 = 2.98
    h = 0.5 * SHEET
    a = 0.16
    ocean.warp.center_x, ocean.warp.center_z = FOCUS
    ocean.warp.half_range = h
    ocean.warp.coef_a = a
    tt = np.linspace(0.0, 1.0, 200001)
    xs = h * (a * tt + (1.0 - a) * tt ** 3)
    sp = h * (a + 3.0 * (1.0 - a) * tt ** 2) * 2.0 / (OCEAN_RES - 1)
    S.spacing = lambda d: float(np.interp(d, xs, sp))
    ocean.material.attenuation_color = tp.Color(0.11, 0.17, 0.15)
    ocean.material.specular_intensity = 0.7
    scene.add(ocean)
    renderer.set_underwater_murk(MURK, tp.Color(*MURK_COLOUR))
    # beyond the sheet: a flat far sea in the haze (turbine_site), so the horizon is water, not sky
    far_mat = std_mat((0.20, 0.23, 0.25), 0.35)
    far_mat.side = tp.Side.Double
    far = tp.Mesh(tp.RingGeometry(0.49 * SHEET, 40000.0, 256), far_mat)
    far.rotate_x(-math.pi / 2)
    far.position.set(FOCUS[0], -0.05, FOCUS[1])
    scene.add(far)
    S.far = far                        # phase C: water, like the sheet

    # ---- the quay face: the DTM's 2 m cells make the quay edge a ramp; a real quay is a wall
    conc = std_mat((0.42, 0.41, 0.39), 0.92)
    L = math.hypot(QUAY_B[0] - QUAY_A[0], QUAY_B[1] - QUAY_A[1]) + 4.0
    T, bottom = 3.0, -6.0
    wall = tp.Mesh(tp.BoxGeometry(L, QUAY_TOP - bottom, T), conc)
    cx, cz = 0.5 * (QUAY_A[0] + QUAY_B[0]), 0.5 * (QUAY_A[1] + QUAY_B[1])
    n_sea = np.array([math.sin(QUAY_HDG), math.cos(QUAY_HDG)])          # local +Z in world xz: the sea side
    wall.position.set(cx - n_sea[0] * T / 2, 0.5 * (QUAY_TOP + bottom), cz - n_sea[1] * T / 2)
    wall.rotation.y = QUAY_HDG
    shadows(wall)
    scene.add(wall)
    S.registry.append((wall, "static_obstacle", "quay"))
    rubber = std_mat((0.035, 0.035, 0.035), 0.8)
    iron = std_mat((0.06, 0.065, 0.07), 0.55, 0.3)
    for x in np.arange(QUAY_A[0] + 3.0, QUAY_B[0] - 2.0, 5.0):
        f = tp.Mesh(tp.CylinderGeometry(FENDER_R, FENDER_R, 2.6, 12), rubber)
        f.position.set(float(x) + n_sea[0] * FENDER_R, QUAY_TOP - 0.25 - 1.3, z_face(x) + n_sea[1] * FENDER_R)
        shadows(f)
        scene.add(f)
        S.registry.append((f, "static_obstacle", f"fender_{len(S.registry)}"))
    S.bollard_tops = []
    for x in np.arange(QUAY_A[0] + 5.0, QUAY_B[0] - 4.0, 9.0):
        b = tp.Mesh(tp.CylinderGeometry(0.16, 0.2, 0.5, 14), iron)
        bx, bz = float(x) - n_sea[0] * 0.9, z_face(x) - n_sea[1] * 0.9
        b.position.set(bx, QUAY_TOP + 0.25, bz)
        shadows(b)
        scene.add(b)
        S.bollard_tops.append(np.array([bx, QUAY_TOP + 0.42, bz]))

    # ---- the Cruise Pier mole: a concrete pier, which the 2 m DTM + land-use paint leave a grassy
    # mound. Fit a box to the mole's land cells (PCA of the cells above the sea) and stand it there.
    MOLE_BOX = (182.0, 398.0, 330.0, 420.0)            # x0, x1, z0, z1: the mole, clear of the west shore
    P = np.array([(x, z) for x in np.arange(MOLE_BOX[0], MOLE_BOX[1], 2.0) for z in np.arange(MOLE_BOX[2], MOLE_BOX[3], 2.0)
                  if geo.height_at(x, z) > 0.3])
    c = P.mean(0)
    d, nrm = np.linalg.svd(P - c)[2]
    if d[0] < 0.0:
        d = -d
    u, w = (P - c) @ d, (P - c) @ nrm
    top = float(np.percentile([geo.height_at(*q) for q in P], 99.5)) + 0.06     # over the grass everywhere
    mole_len = u.max() - u.min() + 12.0            # its root runs 10 m into the shore
    mole_w = w.max() - w.min() + 2.0               # the heightfield's sloping edges stay inside the walls
    mc = c + d * (0.5 * (u.max() + u.min()) - 5.0) + nrm * 0.5 * (w.max() + w.min())
    mole = tp.Mesh(tp.BoxGeometry(float(mole_len), top - bottom, float(mole_w)), conc)
    mole.position.set(float(mc[0]), 0.5 * (top + bottom), float(mc[1]))
    mole.rotation.y = -math.atan2(d[1], d[0])
    shadows(mole)
    scene.add(mole)
    S.registry.append((mole, "static_obstacle", "cruise_pier_mole"))
    S.mole = (mc, d, mole_len, mole_w, top)
    print(f"[harbour] mole: {len(P)} land cells -> concrete pier {mole_len:.0f} x {mole_w:.0f} m, deck {top:.2f} m, "
          f"centre ({mc[0]:.0f}, {mc[1]:.0f}), heading {math.degrees(-math.atan2(d[1], d[0])):+.1f} deg, "
          f"tip ({(mc + d * mole_len / 2)[0]:.0f}, {(mc + d * mole_len / 2)[1]:.0f})")

    # ---- the sjarks, alongside on their strips (each its own StripHull: cheap, no hull exclusion)
    rope = std_mat((0.85, 0.42, 0.08), 0.75)
    S.rope = rope
    S.boats = []                       # dicts: name, obj, hull, berth, lines [vessel pt, world end, mesh]
    tops = np.array(S.bollard_tops)
    for i, (x, rel, side, dyaw) in enumerate(SJARKS):
        glb = os.path.join(HERE, rel)
        if not os.path.exists(glb):
            raise FileNotFoundError(f"{rel} is generated: see harbour/README.md (Build)")
        obj = tp.GLTFLoader().load(glb).scene
        obj.traverse(shadows)
        scene.add(obj)
        hull = StripHull("sjark", ocean, spec_dir=HERE)
        hull.wn_xy, hull.wn_yaw = 0.12, 0.2          # moored: surge / sway / yaw held softly by the lines
        # the loading condition as the builder solved it (sjark_hydro.json: cog z 0, upright). The spec's
        # budget puts the net hauler 0.55 m to starboard (cog z +0.04 m, a 1.6 deg list); the skipper
        # trims that out with fuel and gear, and the hydrostatics file is the condition of record.
        hull.cog = np.array(hull.design["cog"], float)
        half_beam = 0.5 * hull.spec["principal"]["beam"]
        off = 2.0 * FENDER_R + half_beam + 0.05 + 0.04 * (i % 3)
        hdg = QUAY_HDG + (0.0 if side == "port" else math.pi) + math.radians(dyaw)
        bx, bz = x + n_sea[0] * off, z_face(x) + n_sea[1] * off
        names = (f"bollard_fwd_{side}", f"bollard_aft_{side}")
        pts = {n: obj.get_object_by_name(n).get_world_position() for n in names}
        fwd = np.array([math.cos(hdg), -math.sin(hdg)])      # her bow direction in world xz
        lines = []
        for n in names:
            pv = np.array([pts[n].x, pts[n].y, pts[n].z])     # vessel frame: the root sits at identity here
            ahead = 1.0 if "fwd" in n else -1.0
            # the line leads to the quay bollard nearest 7 m beyond her bollard, along her axis
            wx = bx + fwd[0] * (pv[0] + ahead * 7.0)
            end = tops[np.argmin(np.abs(tops[:, 0] - wx))]
            lines.append([pv, end, None])
        name = f"sjark_{i + 1}"
        S.boats.append({"name": name, "obj": obj, "hull": hull, "berth": (bx, bz, hdg), "lines": lines})
        S.registry.append((obj, "boat", name))
    S.hull, S.boat = S.boats[3]["hull"], S.boats[3]["obj"]          # phase A's boat (the stills, the bob plot)
    S.berth = S.boats[3]["berth"]
    bx, bz = S.berth[0], S.berth[1]
    hull = S.hull

    # ---- MS Trollfjord alongside the mole's OUTER (south) face, port side to, bow toward the tip
    mc, dm, mlen, mw, _ = S.mole
    nm = np.array([-dm[1], dm[0]])                       # across the mole ...
    if nm[1] < 0.0:
        nm = -nm                                         # ... toward its south (outer) face
    tf_glb = os.path.join(HERE, "trollfjord.glb")
    if not os.path.exists(tf_glb):
        raise FileNotFoundError("trollfjord.glb is generated: see harbour/README.md (Build)")
    tf = tp.GLTFLoader().load(tf_glb).scene
    tf.traverse(shadows)
    scene.add(tf)
    tfh = StripHull("trollfjord", ocean, spec_dir=HERE)
    tfh.wn_xy, tfh.wn_yaw = 0.25, 0.3                     # a stiff hold: lines tight on the fenders
    if "cog" in tfh.design:
        tfh.cog = np.array(tfh.design["cog"], float)
    tf_half = 0.5 * tfh.spec["principal"]["beam"]
    tf_len = tfh.spec["principal"]["loa"]
    tip = mc + dm * mlen / 2.0
    along = tip - dm * (0.5 * tf_len + 12.0)             # her bow 12 m short of the tip
    tf_off = 0.5 * mw + 0.6 + tf_half                    # 0.6 m of pneumatic fenders
    tfx, tfz = along + nm * tf_off
    tf_hdg = -math.atan2(dm[1], dm[0])
    S.tf = {"name": "trollfjord", "obj": tf, "hull": tfh, "berth": (float(tfx), float(tfz), tf_hdg), "lines": []}
    S.registry.append((tf, "boat", "trollfjord"))
    dep = [geo.height_at(*(np.array([tfx, tfz]) + dm * u + nm * v)) for u in (-60, -30, 0, 30, 60) for v in (-9, 0, 9)]
    bow = np.array([tfx, tfz]) + dm * tf_len / 2
    print(f"[harbour] Trollfjord berth ({tfx:.0f}, {tfz:.0f}), heading {math.degrees(tf_hdg):+.1f} deg, {tf_off:.1f} m "
          f"off the mole axis; seabed under her {max(dep):+.1f}..{min(dep):+.1f} m (she draws "
          f"{tfh.design['draft_max']:.2f} m): {'OK' if max(dep) < -6.0 else 'TOO SHALLOW'}; bow at ({bow[0]:.0f}, {bow[1]:.0f}), "
          f"{math.hypot(*(bow - np.array(MARKS['cardinal_s']))):.0f} m from cardinal_s")

    # ---- the Mariner at her berth near the east end, starboard side to (bow west), on her DP
    m_glb = os.path.join(EXAMPLES, "usv", "mariner.glb")
    if not os.path.exists(m_glb):
        raise FileNotFoundError("usv/mariner.glb is generated: see usv/README.md")
    mar = tp.GLTFLoader().load(m_glb).scene
    mar.traverse(shadows)
    scene.add(mar)
    mh = StripHull("mariner", ocean)
    mh.wn_xy, mh.wn_yaw = 0.5, 0.8                       # tracking a moving target: stiffer than a hold
    m_off = 2.0 * FENDER_R + 0.5 * mh.spec["principal"]["beam"] + 0.1
    m_hdg = QUAY_HDG + math.pi
    mbx, mbz = MARINER_X + n_sea[0] * m_off, z_face(MARINER_X) + n_sea[1] * m_off
    S.mariner = {"name": "mariner", "obj": mar, "hull": mh, "berth": (mbx, mbz, m_hdg), "lines": []}
    S.mariner_cam = mar.get_object_by_name("camera_main")
    S.mariner_steer = mar.get_object_by_name("jet_steering")
    S.u_top = mh.spec["propulsion"].get("top_speed_mps", 12.0)
    S.registry.append((mar, "boat", "mariner"))
    S.path = MarinerPath([(mbx, mbz)] + MARINER_ROUTE, m_hdg)
    S.wake = {"accum": 0.0, "last": None}

    # ---- the marks and floats, each on its own BuoyFloat (depth-decayed wave excitation)
    marks = load_marks()
    glb_b = os.path.join(HERE, "buoys.glb")
    if not os.path.exists(glb_b):
        raise FileNotFoundError("buoys.glb is generated: see harbour/README.md (Build)")
    lib = tp.GLTFLoader().load(glb_b).scene
    S.buoys = []
    items = [(k, k, v) for k, v in MARKS.items()] + EXTRA_FLOATS
    for mark, inst, xz in items:
        obj = lib.get_object_by_name(mark).clone()
        obj.traverse(shadows)
        scene.add(obj)
        # the lateral numbers face +-X: yaw the mark so they face the fairway (the route runs N-S here)
        fl = BuoyFloat(mark, ocean, xz, yaw=math.radians(90.0), hydro=marks)
        S.buoys.append((fl, obj))
        cls = "float" if mark.startswith("blaase") or mark == "garnblaase" else "buoy"
        S.registry.append((obj, cls, inst))

    # ---- report the site
    print(f"[harbour] quay heading {math.degrees(QUAY_HDG):+.2f} deg, face z {z_face(QUAY_A[0]):.1f}..{z_face(QUAY_B[0]):.1f} "
          f"over x {QUAY_A[0]:.0f}..{QUAY_B[0]:.0f}; berth CoG ({bx:.1f}, {bz:.1f}), {off:.2f} m off the face")
    prof = " ".join(f"{d:+.0f}:{geo.height_at(BERTH_X + n_sea[0] * d, z_face(BERTH_X) + n_sea[1] * d):+.2f}"
                    for d in (-4, -2, -1, 0, 1, 2, 3, 4, 6, 8))
    print(f"[harbour] ground across the quay at x {BERTH_X:.0f} (m off the face: height): {prof}")
    depth_keel = geo.height_at(bx, bz)
    print(f"[harbour] SEABED AT THE BERTH {depth_keel:+.2f} m (the sjark draws {hull.design['draft_max']:.2f} m): "
          f"{'OK' if depth_keel < -hull.design['draft_max'] - 0.5 else 'ON THE BOTTOM'}")
    for k, v in MARKS.items():
        print(f"[harbour] mark {k:13s} at ({v[0]:.0f}, {v[1]:.0f}): seabed {geo.height_at(*v):+.1f} m, "
              f"sea mesh spacing {S.spacing(math.hypot(v[0] - FOCUS[0], v[1] - FOCUS[1])):.2f} m")
    rl = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ROUTE, ROUTE[1:]))
    rm = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ROUTE[:4], ROUTE[1:4]))
    print(f"[harbour] route out {rl:.0f} m ({rm:.0f} m berth to mouth); depths along it "
          + " ".join(f"{geo.height_at(*p):+.0f}" for p in ROUTE[1:]))
    print(f"[harbour] sea mesh spacing at the berth {S.spacing(math.hypot(bx - FOCUS[0], bz - FOCUS[1])):.2f} m")

    S.__dict__.update(scene=scene, sky=sky, sun=sun, geo=geo, ocean=ocean, n_sea=n_sea)
    S.floaters = S.boats + [S.tf, S.mariner]
    print(f"[harbour] registry: {len(S.registry)} objects: " + ", ".join(
        f"{c} {sum(1 for r in S.registry if r[1] == c)}" for c in ("boat", "buoy", "float", "static_obstacle")))
    return S


# --------------------------------------------------------------------------- #
#  Cameras: name -> fn(S) -> (eye, target, vfov)
# --------------------------------------------------------------------------- #
def _cam_drone(S):
    return np.array([520.0, 120.0, 560.0]), np.array([420.0, 0.0, 250.0]), 45.0


def _cam_sealevel(S):
    p = S.hull.p
    d = np.array([0.42, 0.0, 0.91])
    eye = p + 40.0 * d
    eye[1] = S.ocean.sample_height(float(eye[0]), float(eye[2])) + 2.2
    return eye, np.array([p[0], 1.6, p[2]]), 40.0


def _cam_close(S):
    p = S.hull.p
    eye = np.array([p[0] + 11.0, 0.0, p[2] + 7.5])
    eye[1] = S.ocean.sample_height(float(eye[0]), float(eye[2])) + 2.6
    return eye, np.array([p[0] + 1.5, 1.2, p[2] - 1.0]), 50.0


def _cam_spar(S):
    fl = S.buoys[0][0]
    p = fl.p
    d = np.array([0.30, 0.0, 0.954])
    eye = p + 30.0 * d
    eye[1] = S.ocean.sample_height(float(eye[0]), float(eye[2])) + 2.2
    return eye, np.array([p[0], 2.2, p[2]]), 35.0


CAMS = {"drone": _cam_drone, "sealevel": _cam_sealevel, "close": _cam_close, "spar": _cam_spar}


def apply_camera(renderer, S, camera, name):
    eye, tgt, fov = CAMS[name](S)
    camera.fov = fov
    camera.near, camera.far = 0.2, 20000.0
    camera.update_projection_matrix()
    camera.position.set(*map(float, eye))
    camera.look_at(tp.Vector3(*map(float, tgt)))
    return eye


def frame(renderer, S, camera, t):
    """One frame at sim time t: the pinned clock, terrain LOD at the camera, the over/under split."""
    renderer.sim_time = t
    cp = camera.position
    S.geo.update(cp)
    renderer.set_fog_water_surface_y(float(S.ocean.sample_height(cp.x, cp.z)))
    renderer.render(S.scene, camera)


class MarinerPath:
    """The Mariner's reference: a C2 cubic spline through the waypoints (chord-length knots, the
    berth's heading as the start tangent), a speed schedule in time (at rest until T_CAST, eased to
    V_BASIN over T_RAMP, eased up to V_OUT over S_EASE metres past the mouth), tabulated at 100 Hz:
    position, velocity, acceleration, heading and its rate come from the SAME table, so the DP's
    target, its velocity and its feed-forward agree and the cameras that ride the path are smooth."""

    def __init__(self, pts, hdg0, t_end=400.0):
        from scipy.interpolate import CubicSpline
        P = np.asarray(pts, float)
        u = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        d0 = np.array([math.cos(hdg0), -math.sin(hdg0)])
        cs = CubicSpline(u, P, bc_type=((1, d0), (2, np.zeros(2))))
        uu = np.linspace(0.0, u[-1], 40001)
        q = cs(uu)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        self.length = float(s[-1])
        self.s_tab, self.q_tab = s, q
        # the mouth: the arc length nearest MOUTH
        self.s_mouth = float(s[np.argmin(np.hypot(q[:, 0] - MOUTH[0], q[:, 1] - MOUTH[1]))])
        dt = 0.01
        self.t = np.arange(0.0, t_end, dt)
        sv = np.zeros_like(self.t)
        v = np.zeros_like(self.t)
        for i in range(1, len(self.t)):
            ti = self.t[i]
            r = np.clip((ti - T_CAST) / T_RAMP, 0.0, 1.0)
            e = np.clip((sv[i - 1] - self.s_mouth) / S_EASE, 0.0, 1.0)
            v[i] = V_BASIN * r * r * (3 - 2 * r) + (V_OUT - V_BASIN) * e * e * (3 - 2 * e)
            sv[i] = min(sv[i - 1] + v[i] * dt, self.length - 1e-3)
        self.sv = sv
        self.xz = self.pos_s(sv)
        vel = np.gradient(self.xz, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        tan = self.tangent_s(sv)
        self.hdg = np.unwrap(np.arctan2(-tan[:, 1], tan[:, 0]))
        self.rate = np.gradient(self.hdg, dt)
        self.vel, self.acc = vel, acc

    def pos_s(self, s):
        s = np.asarray(s, float)
        x = np.interp(s, self.s_tab, self.q_tab[:, 0])
        z = np.interp(s, self.s_tab, self.q_tab[:, 1])
        out = np.stack([x, z], axis=-1)
        # before the start: straight back along the berth's heading (a chase camera's eye)
        t0 = self.tangent_s(0.0)
        neg = np.minimum(s, 0.0)[..., None]
        return out + neg * t0

    def tangent_s(self, s, h=0.5):
        s = np.asarray(s, float)
        a = np.clip(s - h, 0.0, self.length - 2 * h)
        b = a + 2 * h
        d = np.stack([np.interp(b, self.s_tab, self.q_tab[:, k]) - np.interp(a, self.s_tab, self.q_tab[:, k])
                      for k in range(2)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def at(self, t):
        """(x, z, heading, vx, vz, rate, ax, az, s, v) at sim time t (from the film's start)."""
        f = lambda a: float(np.interp(t, self.t, a))
        return (f(self.xz[:, 0]), f(self.xz[:, 1]), f(self.hdg), f(self.vel[:, 0]), f(self.vel[:, 1]),
                f(self.rate), f(self.acc[:, 0]), f(self.acc[:, 1]), f(self.sv),
                math.hypot(f(self.vel[:, 0]), f(self.vel[:, 1])))


def rope_tube(P, r=0.02, sides=8):
    """Positions and normals of a round tube along the polyline P (ts.tube's layout)."""
    Tg = np.gradient(P, axis=0)
    Tg /= np.linalg.norm(Tg, axis=1, keepdims=True)
    nrm = np.cross(Tg[0], [0.0, 1.0, 0.0])
    nrm /= np.linalg.norm(nrm)
    Ns = []
    for t in Tg:
        nrm = nrm - t * np.dot(nrm, t)
        nrm /= np.linalg.norm(nrm)
        Ns.append(nrm.copy())
    Ns = np.array(Ns)
    Bs = np.cross(Tg, Ns)
    a = 2 * np.pi * np.arange(sides + 1) / sides
    N = (np.cos(a)[None, :, None] * Ns[:, None, :] + np.sin(a)[None, :, None] * Bs[:, None, :])
    pos = (P[:, None, :] + r * N).reshape(-1, 3)
    return pos.astype(np.float32), N.reshape(-1, 3).astype(np.float32)


def update_lines(S):
    """Every boat's mooring lines, from her bollards to the quay's, following her each frame: the
    tube is built once, then its positions and normals are rewritten in place."""
    for b in S.boats + [S.tf]:
        for ln in b["lines"]:
            pv, p1, mesh = ln
            p0 = b["hull"].to_world(pv)
            span = float(np.linalg.norm(p1 - p0))
            pts = sag_line(p0, p1, 0.025 * span + 0.1, 28)
            if mesh is None:
                mesh = tp.Mesh(ts.tube(pts, lambda s: np.full(len(s), 0.02), 8), S.rope)
                mesh.cast_shadow = True
                S.scene.add(mesh)
                ln[2] = mesh
            else:
                pos, nrm = rope_tube(pts)
                mesh.geometry.update_attribute("position", pos)
                mesh.geometry.update_attribute("normal", nrm)


def seat_all(S, t):
    for b in S.floaters:
        b["hull"].seat(*b["berth"])
    for fl, _ in S.buoys:
        fl.seat()
    S.ocean.clear_wake()
    S.wake = {"accum": 0.0, "last": None}


WAKE_MAX_AGE, WAKE_MAX_SAMPLES = 8.0, 64          # usv_ocean
# NO FOAM from the Mariner. The ocean's foam texture spans ONE swell tile (tile_size_0 = 80 m,
# foam_world.comp) and repeats over the whole sheet, so every splat and the wake's foam trail came
# back as a lattice of white dashes 80 m apart over the basin (pB v4, the wide cut). The Kelvin
# wake's HEIGHT is world-anchored (her trail samples) and stays; its foam trail gates on
# smoothstep(0.5, 1.5, forward_speed), so the speed handed to it is held just over the wake's
# 0.5 m/s floor: a faint V behind her and next to no foam. usv_ocean's 0.6 + 0.10 u (cap 1.2)
# returns once the foam is anchored beyond one tile.
WAKE_FLOOR, WAKE_GAIN, WAKE_CAP = 0.56, 0.02, 0.62
FOAM_SPLATS = False


def mariner_sea(S, t, dt):
    """Her hull footprint, the Kelvin wake trail and her foam (usv_ocean.sea_update's pattern)."""
    h = S.mariner["hull"]
    h.footprint(S.ocean)
    R = h.R
    fwd = R[:, 0]
    u = float(fwd @ h.v)
    ws = math.copysign(min(WAKE_FLOOR + WAKE_GAIN * abs(u), WAKE_CAP), u) if abs(u) > 0.2 else 0.0
    S.ocean.wake.forward_speed = ws
    c = h.to_world([h.excl[2], 0.0, 0.0])
    yaw_ex = math.atan2(fwd[0], fwd[2])
    if dt > 0.0:
        S.ocean.age_wake(dt, WAKE_MAX_AGE, WAKE_MAX_SAMPLES)
        S.wake["accum"] += dt
        last = S.wake["last"]
        moved = 1e9 if last is None else math.hypot(c[0] - last[0], c[2] - last[1])
        if abs(u) > 0.4 and (S.wake["accum"] >= 0.1 or moved >= 1.0):
            S.wake["accum"] = 0.0
            S.wake["last"] = (c[0], c[2])
            S.ocean.add_wake_sample(float(c[0]), float(c[2]), math.sin(yaw_ex), math.cos(yaw_ex), float(ws),
                                    WAKE_MAX_SAMPLES)
    S.ocean.clear_foam_disturbances()
    spd = min(abs(u) / 6.0, 1.0)

    def foam(xb, zb, radius, k):
        q = h.to_world([xb, 0.4, zb])
        if k > 0.02:
            S.ocean.add_foam_disturbance(float(q[0]), float(q[2]), radius, min(k, 1.0))
    if FOAM_SPLATS and abs(u) > 0.3:
        for zs in (-0.75, 0.75):
            foam(2.2, zs, 0.9, 0.10 + 0.45 * spd)            # bow wave, off the shoulders
        foam(-3.4, 0.0, 1.0 + 0.6 * spd, 0.25 + 0.4 * spd)    # jet wash
    return u


def step_floats(S, dt, t=None, substeps=None):
    """AFTER a render: every float samples that frame's sea, steps, and is placed. t (sim s from the
    film's start) drives the Mariner's moving target; None leaves her holding her berth."""
    m = S.mariner["hull"]
    if t is not None:
        x, z, hd, vx, vz, rate, ax, az, s, v = S.path.at(t + dt)
        m.set_target(x, z, hd, vx, vz, rate, ax, az)
        S.mariner_rate = rate
    n = substeps or max(2, int(math.ceil(dt * 120.0)))
    if dt > 0.0:
        for b in S.boats + [S.tf]:
            b["hull"].advance(dt, max(1, n // 2))
        m.advance(dt, n)
        for fl, _ in S.buoys:
            fl.advance(dt, max(1, n // 2))
    for b in S.floaters:
        b["hull"].place(b["obj"])
    for fl, obj in S.buoys:
        fl.place(obj)
    u = mariner_sea(S, t, dt)
    if S.mariner_steer is not None:
        # the jet nozzle from the path's turn rate (usv_ocean: steer < 0 turns her to port)
        S.mariner_steer.rotation.y = float(np.clip(-getattr(S, "mariner_rate", 0.0) / 0.25, -1.0, 1.0) * math.radians(25.0))
    update_lines(S)
    return u


def film_frame(renderer, S, camera, t, path=None):
    """One frame at sim time t (the renderer's clock is T_BASE + t): the camera is already placed."""
    renderer.sim_time = T_BASE + t
    cp = camera.position
    S.geo.update(cp)
    renderer.set_fog_water_surface_y(float(S.ocean.sample_height(cp.x, cp.z)))
    if path is None:
        renderer.render(S.scene, camera)
    else:
        tmp = os.path.join(os.path.dirname(path), "_tmp_" + os.path.basename(path))
        renderer.save_frame(S.scene, camera, tmp)
        os.replace(tmp, path)


# --------------------------------------------------------------------------- #
#  The film: cuts in sim time (s from the film's start). The sim runs on through the gaps between
#  cuts (an ellipsis: the route to the mouth is ~210 m at 5 kn, longer than a film should dwell).
# --------------------------------------------------------------------------- #
T_BASE = 30.0                      # the renderer's clock at the film's start (s of sea state)
T_PRE = 6.0                        # settle before the first frame


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _film_drone(S, t, t0, t1):
    k = ease((t - t0) / (t1 - t0))
    e0, e1 = np.array([455.0, 150.0, 760.0]), np.array([452.0, 62.0, 420.0])
    g0, g1 = np.array([420.0, 0.0, 290.0]), np.array([418.0, 2.0, 222.0])
    return e0 + (e1 - e0) * k, g0 + (g1 - g0) * k, 45.0


def _film_chase(S, t, t0, t1):
    """Behind and above her, on her port quarter (the sea side), riding the reference path."""
    s = float(np.interp(t, S.path.t, S.path.sv))
    back = S.path.pos_s(s - 15.0)
    tan = S.path.tangent_s(max(s - 6.0, 0.0))
    port = np.array([tan[1], -tan[0]])                 # left of the heading, in world xz
    eye = np.array([back[0] + 5.0 * port[0], 7.0, back[1] + 5.0 * port[1]])
    ahead = S.path.pos_s(s + 8.0)
    return eye, np.array([ahead[0], 0.8, ahead[1]]), 50.0


def _film_pov(S, t, t0, t1):
    return None                        # camera_main: set from the node


def _film_wide(S, t, t0, t1):
    k = ease((t - t0) / (t1 - t0))
    x, z = S.path.at(t)[:2]
    e0, e1 = np.array([180.0, 95.0, 640.0]), np.array([215.0, 80.0, 610.0])
    g = np.array([0.6 * x + 0.4 * 360.0, 3.0, 0.6 * z + 0.4 * 400.0])
    return e0 + (e1 - e0) * k, g, 42.0


# (name, sim t0, sim t1, camera fn)
CUTS = [("a_drone", 0.0, 7.0, _film_drone),
        ("b_chase", 7.0, 17.0, _film_chase),
        ("c1_pov_quay", 18.0, 26.0, _film_pov),
        ("c2_pov_special", 40.0, 48.0, _film_pov),
        ("c3_pov_gate", 79.0, 89.0, _film_pov),
        ("d_wide", 100.0, 109.0, _film_wide)]


def place_film_camera(S, camera, fn, t, t0, t1, W, H):
    r = fn(S, t, t0, t1)
    camera.near, camera.far = 0.2, 20000.0
    if r is None:
        hfov = math.radians(S.mariner["hull"].spec["sensors"]["camera_main"]["hfov_deg"])
        camera.fov = math.degrees(2.0 * math.atan(math.tan(hfov / 2.0) * H / W))
        camera.update_projection_matrix()
        S.mariner["obj"].update_matrix_world(True)
        wp, wq = S.mariner_cam.get_world_position(), S.mariner_cam.get_world_quaternion()
        camera.position.set(wp.x, wp.y, wp.z)
        camera.quaternion.set(wq.x, wq.y, wq.z, wq.w)
        return
    eye, tgt, fov = r
    camera.fov = fov
    camera.update_projection_matrix()
    camera.position.set(*map(float, eye))
    camera.look_at(tp.Vector3(*map(float, tgt)))


def run_film(renderer, S, camera, outdir, W, H, test=False, labels=None):
    """Simulate from -T_PRE to the last cut. In a cut: 30 fps (film) or 10 fps (test); between cuts
    and in the pre-roll: 5 fps, unsaved. The test saves 3 stills per cut; the film every frame.
    labels (phase C, harbour_labels.Exporter): the same sim, run to the last POV cut; each POV frame
    goes to labels.pov_frame instead of a png, every other frame is rendered and dropped."""
    fps = 30.0
    t_end = CUTS[-1][2] if labels is None else labels.t_end
    stills = {}
    for name, t0, t1, _ in CUTS:
        for j, f in enumerate((0.1, 0.5, 0.9)):
            stills[round(t0 + f * (t1 - t0), 2)] = f"{name}_{j}"
    frames_dir = os.path.join(outdir, "frames")
    if not test and labels is None:
        os.makedirs(frames_dir, exist_ok=False)
    rec = []
    t = -T_PRE
    k_frame = 0
    seated = False
    t_wall = time.perf_counter()
    cut_i = 0
    while t <= t_end + 1e-6:
        cut = next((c for c in CUTS if c[1] - 1e-6 <= t < c[2] - 1e-6), None)
        # labels: a cut named in labels.pov_over is seen from camera_main instead (c0: the chase
        # window, same times and frame rate, so the sim is the film's)
        pov = cut is not None and (cut[3] is _film_pov or (labels is not None and cut[0] in labels.pov_over))
        if cut is not None:
            place_film_camera(S, camera, _film_pov if pov else cut[3], t, cut[1], cut[2], W, H)
            dt = 1.0 / (10.0 if test else fps)
        else:
            _film_drone(S, 0.0, 0.0, 1.0)
            place_film_camera(S, camera, _film_drone, 0.0, 0.0, 1.0, W, H) if t < 0 else \
                place_film_camera(S, camera, _film_chase, t, 0.0, 1.0, W, H)
            dt = 0.2
        path = None
        if labels is not None:
            if pov:
                labels.pov_frame(t, cut)
            else:
                film_frame(renderer, S, camera, t)
        elif cut is not None and not test:
            path = os.path.join(frames_dir, f"f_{k_frame:05d}.png")
            k_frame += 1
        elif cut is not None and test:
            key = next((k for k in stills if abs(k - t) < 0.5 * dt), None)
            if key is not None:
                # a still: let the terrain stream in at this view, then save
                for _ in range(6):
                    film_frame(renderer, S, camera, t)
                path = new_path(outdir, stills.pop(key) + ".png")
        if labels is None:
            film_frame(renderer, S, camera, t, path)
        if path is not None and test:
            print(f"[harbour] still {os.path.basename(path)} at sim {t:.2f} s")
        if not seated:
            seat_all(S, t)
            seated = True
        # next cut boundary: land exactly on it
        nxt = min([c[1] for c in CUTS if c[1] > t + 1e-6] + [t + dt])
        dt = min(dt, nxt - t)
        u = step_floats(S, dt, t)
        if labels is not None and not labels.assigned:
            labels.assign()                        # the mooring lines exist after the first step
        m = S.mariner["hull"]
        x, z = S.path.at(t + dt)[:2]
        rec.append((t + dt, u, math.hypot(m.p[0] - x, m.p[2] - z)))
        t = round(t + dt, 6)
        if cut is not None and cut[0] != CUTS[cut_i][0]:
            cut_i = CUTS.index(cut)
        if int(t) != int(t - dt):
            el = time.perf_counter() - t_wall
            print(f"[harbour] sim {t:6.1f} s  wall {el:6.0f} s  mariner u {u:+.2f} m/s, off the path "
                  f"{rec[-1][2]:.2f} m", flush=True)
    rec = np.array(rec)
    mv = rec[rec[:, 0] > T_CAST + 2.0]
    print(f"[harbour] Mariner tracking error: mean {mv[:, 2].mean():.2f} m, max {mv[:, 2].max():.2f} m; top speed "
          f"{rec[:, 1].max():.2f} m/s")
    for b in S.floaters:
        print("[harbour] " + b["hull"].summary(len(b["hull"].log) // 2))
    for fl, _ in S.buoys:
        print("[harbour] " + fl.summary(len(fl.log) // 2))
    return k_frame, frames_dir


def encode(frames_dir, outdir, fps=30):
    import subprocess
    import imageio_ffmpeg
    k = 1
    while os.path.exists(os.path.join(outdir, f"harbour_film_v{k}.mp4")):
        k += 1
    out = os.path.join(outdir, f"harbour_film_v{k}.mp4")
    tmp = os.path.join(outdir, f"_tmp_harbour_film_v{k}.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frames_dir, "f_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "slow", "-threads", "1", tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)
    return out


def contact_sheet(outdir, names, path, cols=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    rows = (len(names) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 5.3, rows * 3.2))
    for a in np.ravel(ax):
        a.axis("off")
    for a, n in zip(np.ravel(ax), names):
        a.imshow(mpimg.imread(os.path.join(outdir, n + ".png")))
        a.set_title(n, fontsize=9)
    fig.tight_layout()
    tmp = os.path.join(outdir, "_tmp_" + os.path.basename(path))
    fig.savefig(tmp, dpi=90)
    os.replace(tmp, path)


def new_path(outdir, name):
    p = os.path.join(outdir, name)
    if os.path.exists(p):
        raise FileExistsError(f"{p} exists: pass a new --out folder (never overwrite a delivered file)")
    return p


def bob_plot(S, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = rec["t"] - rec["t"][0]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, rec["sjark_heave"] - rec["sjark_heave"].mean(), label="sjark CoG heave about its mean (m)", color="#1f5f99")
    ax[0].plot(t, rec["buoy_y"] - rec["buoy_y"].mean(), label="spar heave about its mean (m)", color="#c0392b")
    ax[0].plot(t, rec["sea"], label="sea at the sjark (m)", color="#1f5f99", lw=0.7, ls=":")
    ax[0].plot(t, rec["sea_buoy"], label="sea at the spar (m)", color="#c0392b", lw=0.7, ls=":")
    ax[1].plot(t, rec["sjark_roll"], label="sjark roll", color="#1f5f99")
    ax[1].plot(t, rec["buoy_roll"], label="spar roll", color="#c0392b")
    ax[2].plot(t, rec["sjark_pitch"], label="sjark trim", color="#1f5f99")
    ax[2].plot(t, rec["buoy_pitch"], label="spar pitch", color="#c0392b")
    ax[1].set_ylabel("deg")
    ax[2].set_ylabel("deg")
    ax[2].set_xlabel(f"s (sim time {rec['t'][0]:.0f}..{rec['t'][-1]:.0f})")
    for a in ax:
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("harbour bob check: the moored sjark and the lateral_port spar, 60 Hz")
    fig.tight_layout()
    tmp = os.path.join(os.path.dirname(path), "_tmp_" + os.path.basename(path))
    fig.savefig(tmp, dpi=100)
    os.replace(tmp, path)


def run_labels(renderer, S, camera, outdir, W, H, split_dir):
    """Phase C: the POV cuts of the film's sim as a LaRS split (harbour_labels), the POV | mask
    film, overlay stills, a contact sheet, the stats and the three gates, all into outdir."""
    import json
    import harbour_labels as hl
    from PIL import Image
    if os.path.exists(split_dir):
        raise FileExistsError(f"{split_dir} exists: pass a new --labels folder")
    exp = hl.Exporter(sys.modules[__name__], renderer, S, camera, W, H, split_dir, outdir)
    t1 = time.perf_counter()
    run_film(renderer, S, camera, outdir, W, H, labels=exp)
    st = exp.close()
    print(f"[labels] {len(exp.keys)} keyframes, {exp.film_n} POV frames in {time.perf_counter() - t1:.0f} s -> {split_dir}")
    print(f"[labels] film {exp.film_path} ({exp.film_n / 30.0:.1f} s)")
    gates = {"gate1_every_visible_instance_labelled": {}, "gate2_own_hull_never_boat": {}, "gate3_no_frame_lag": {}}
    g1 = {k: v for k, v in exp.gate1.items()}
    fail1 = [k for k, v in g1.items() if v["expected"] > 0 and v["seen"] == 0]
    gates["gate1_every_visible_instance_labelled"] = {
        "pass": not fail1, "failed": fail1,
        "never_expected_visible": [k for k, v in g1.items() if v["expected"] == 0],
        "per_instance": g1}
    thing_px = sum(g["thing_px_on_own_hull"] for g in exp.gate2)
    gates["gate2_own_hull_never_boat"] = {
        "pass": thing_px == 0 and all(set(g["labels_on_own_hull"]) <= {"Ignore (void)"} for g in exp.gate2),
        "own_hull_px_total": sum(g["own_hull_px"] for g in exp.gate2), "thing_px_on_own_hull": thing_px,
        "per_keyframe": exp.gate2}
    g3 = hl.gate3(exp.track)
    gates["gate3_no_frame_lag"] = {"pass": bool(g3) and all(v["pass"] for v in g3.values()), "marks": g3}
    for k in ("gate1_every_visible_instance_labelled", "gate2_own_hull_never_boat", "gate3_no_frame_lag"):
        g = gates[k]
        print(f"[labels] {k}: {'PASS' if g['pass'] else 'FAIL'}  " + json.dumps(
            {kk: vv for kk, vv in g.items() if kk not in ("per_instance", "per_keyframe", "pass")})[:600])
    for name, path, doc in (("stats", "labels_stats.json", st), ("gates", "labels_gates.json", gates)):
        p = new_path(outdir, path)
        with open(p + ".tmp", "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(p + ".tmp", p)
        print(f"[labels] {name}: {p}")
    print("[labels] classes: " + json.dumps(st["classes"]))
    # overlay stills: 2 keyframes spread over c0, 6 over c1..c3; a contact sheet of every keyframe overlay
    k0 = [j for j, k in enumerate(exp.keys) if k[0].startswith("harbour_c0")]
    rest = [j for j, k in enumerate(exp.keys) if not k[0].startswith("harbour_c0")]
    pick = sorted(set([k0[int(round(x))] for x in np.linspace(0, len(k0) - 1, 2)] if k0 else []) |
                  set(rest[int(round(x))] for x in np.linspace(0, len(rest) - 1, 6)))
    for j in pick:
        name, t, cut, ov = exp.keys[j]
        p = new_path(outdir, f"overlay_{name}.png")
        ov.save(p + ".tmp.png")
        os.replace(p + ".tmp.png", p)
        print(f"[labels] overlay still {p}")
    tw, th, cols = 427, 240, 6
    rows = (len(exp.keys) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th))
    for j, (name, t, cut, ov) in enumerate(exp.keys):
        o = ov.crop((2 * W, 34, 3 * W, 34 + H)).resize((tw, th), Image.LANCZOS)
        sheet.paste(o, ((j % cols) * tw, (j // cols) * th))
    p = new_path(outdir, "contact_sheet_overlays.png")
    sheet.save(p + ".tmp.png")
    os.replace(p + ".tmp.png", p)
    print(f"[labels] contact sheet {p}")


def main():
    shot = cli_arg("--shot", "all", str)
    W, H = parse_size(cli_arg("--size", "1280x720" if "--labels" in sys.argv else "1600x900", str))
    t_shot = cli_arg("--time", T_SHOT, float)
    terrain = cli_arg("--terrain", os.path.join(REPO, "geodata", "aalesund"), str)
    outdir = cli_arg("--out", os.path.join(HERE, "out"), str)
    os.makedirs(outdir, exist_ok=True)
    names = list(CAMS) + ["bob"] if shot == "all" else shot.split(",")

    canvas = tp.Canvas("threepp - harbour", width=W, height=H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.AgX
    renderer.tone_mapping_exposure = 0.9 if cli_arg("--light", "bright", str) == "bright" else 0.8
    if cli_arg("--light", "bright", str) == "bright":
        renderer.set_color_grade(saturation=1.12, contrast=1.06)      # a clear day's colour, after the tone map
    renderer.sun_angular_radius = 0.6
    renderer.bloom_intensity = 0.06
    renderer.auto_exposure = False
    t0 = time.perf_counter()
    light = cli_arg("--light", "bright", str)
    S = build(renderer, terrain, light)
    camera = tp.PerspectiveCamera(45.0, W / H, 0.2, 20000.0)
    if "--labels" in sys.argv:
        run_labels(renderer, S, camera, outdir, W, H, cli_arg("--labels", "", str))
        return
    if "--film" in sys.argv or "--film-test" in sys.argv:
        test = "--film-test" in sys.argv
        t1 = time.perf_counter()
        n, frames_dir = run_film(renderer, S, camera, outdir, W, H, test=test)
        print(f"[harbour] film {'test ' if test else ''}simulated + rendered in {time.perf_counter() - t1:.0f} s")
        if test:
            names = [f"{c[0]}_{j}" for c in CUTS for j in range(3)]
            names = [n_ for n_ in names if os.path.exists(os.path.join(outdir, n_ + ".png"))]
            sheet = new_path(outdir, "contact_sheet.png")
            contact_sheet(outdir, names, sheet)
            print(f"[harbour] contact sheet {sheet}")
        else:
            out = encode(frames_dir, outdir)
            print(f"[harbour] film: {n} frames ({n / 30.0:.1f} s) -> {out}")
        return

    # settle, then the bob check: 60 Hz from t_shot - settle, the last BOB_S seconds logged
    settle = cli_arg("--settle", 20.0, float)
    t_start = t_shot - settle
    apply_camera(renderer, S, camera, "drone")
    n = int(round(settle * 60.0))
    rec = {k: [] for k in ("t", "sea", "sjark_heave", "sjark_roll", "sjark_pitch", "buoy_heave", "buoy_roll", "buoy_pitch",
                           "buoy_y", "sea_buoy")}
    for k in range(n + 1):
        t = t_start + k / 60.0
        frame(renderer, S, camera, t)
        if k == 0:
            seat_all(S, t)
        step_floats(S, 1.0 / 60.0 if k < n else 0.0)
        if t >= t_shot - BOB_S - 1e-9:
            r = S.hull.readout()
            fl = S.buoys[0][0]
            rb = fl.readout()
            rec["t"].append(t)
            rec["sea"].append(float(np.mean(S.hull.water)))
            rec["sjark_heave"].append(float(S.hull.p[1]))
            rec["sjark_roll"].append(r["roll"])
            rec["sjark_pitch"].append(r["trim"])
            rec["buoy_heave"].append(rb["heave"])
            rec["buoy_roll"].append(rb["roll"])
            rec["buoy_pitch"].append(rb["pitch"])
            rec["buoy_y"].append(float(fl.p[1]))
            rec["sea_buoy"].append(float(fl.eta))
    rec = {k: np.array(v) for k, v in rec.items()}
    print(f"[harbour] settled {settle:.0f} s of sea in {time.perf_counter() - t0:.1f} s (incl. build)")
    r = S.hull.readout()
    print(f"[harbour] sjark at t {t_shot:.1f}: CoG ({S.hull.p[0]:.2f}, {S.hull.p[1]:+.3f}, {S.hull.p[2]:.2f}), "
          f"draft {r['draft']:.3f} m, trim {r['trim']:+.2f}, roll {r['roll']:+.2f}, hdg {r['heading']:+.2f} deg; "
          f"drift from the berth {math.hypot(S.hull.p[0] - S.berth[0], S.hull.p[2] - S.berth[1]):.2f} m")
    skip = int((settle - BOB_S) * 60.0)
    print("[harbour] " + S.hull.summary(skip))
    for fl, _ in S.buoys:
        print("[harbour] " + fl.summary(skip))
    for k in ("sjark_heave", "sjark_roll", "sjark_pitch", "buoy_y", "buoy_heave", "buoy_roll", "buoy_pitch", "sea", "sea_buoy"):
        v = rec[k]
        print(f"[harbour] bob {k:12s} {v.min():+.3f} .. {v.max():+.3f} (range {v.max() - v.min():.3f})")
    for name in names:
        if name == "bob":
            path = new_path(outdir, "e_bob_check.png")
            bob_plot(S, rec, path)
            print(f"[harbour] bob: wrote {path}")
            continue
        t1 = time.perf_counter()
        label = {"drone": "a_drone", "sealevel": "b_sealevel", "close": "c_close", "spar": "d_spar"}[name]
        path = new_path(outdir, f"{label}.png")
        apply_camera(renderer, S, camera, name)
        # a new view streams its terrain tiles in over a few updates; then TAA needs a few frames
        for i in range(40):
            frame(renderer, S, camera, t_shot)
            if i >= 3 and S.geo.stats["baking"] == 0:
                break
        for i in range(4):
            frame(renderer, S, camera, t_shot - (3 - i) / 60.0)
        renderer.sim_time = t_shot
        tmp = os.path.join(outdir, "_tmp_" + os.path.basename(path))
        renderer.save_frame(S.scene, camera, tmp)
        os.replace(tmp, path)
        print(f"[harbour] {name}: wrote {path} in {time.perf_counter() - t1:.1f} s")


if __name__ == "__main__":
    main()
