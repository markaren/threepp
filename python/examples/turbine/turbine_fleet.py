"""Five robots at the offshore turbine: the site from turbine_site.py plus the fleet at its work
stations. The Mariner holds station off the boat landing, the Otter X runs a slow survey circle,
the drone inspects the down-pointing blade, the ROV lights the anode cage on its tether, the
snake reads the cable protection system where it meets the rock.

    python turbine_fleet.py --shot all --tag v01 [--outdir <dir>]
    python turbine_fleet.py --shot mariner_station --out x.png [--settle 10] [--sheet 6000]

Importable: build_fleet(renderer, S) adds the vehicles to a built site and returns a namespace
whose update(t, dt) advances every vehicle one frame (the film drives that). Per frame:
    renderer.sim_time = t; S.update(t, camera, under, dt); renderer.render(S.scene, camera)
    F.update(t, dt)          # boats sample THAT frame's sea, step 2 x dt/2, place; drone, tether follow
F.mariner / F.otter are the StripHull bodies (set_target moves a DP station), F.vehicles holds the
drone, ROV, snake links and lamps. Headless only.
Frame: X upwind, Y up, origin on the pile axis at MSL (turbine_site.py).
"""
import math
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(_HERE))                     # examples/ (demo_common, drone_rig, usv_rig)
sys.path.insert(0, _HERE)

import threepp as tp
from demo_common import cli_arg, parse_size

import turbine_site as ts
from usv_rig import StripHull, USV_DIR
import fleet_vehicles as fv

KN = 0.5144
# ---- stations (assumed, see plans/offshore-turbine.md)
LANDING_AZ = ts.spec["transition_piece"]["boat_landing_azimuth_deg"]["value"]   # 90: the +Z side
MARINER_R = 18.5                   # CoG off the pile axis: ~13 m off the TP wall (r 5.3), fenders at ~7 m
MARINER_HDG = 0.0                  # bow into the wind (X upwind)
OTTER_R, OTTER_KN = 40.0, 2.0      # survey circle radius, speed
OTTER_AZ0 = -20.0                  # where on the circle she is at the still time (deg, atan2(z, x))
T_SHOT = ts.T_SHOT


def otter_on_circle(t):
    """Target pose on the survey circle at time t (counter-clockwise in atan2(z, x))."""
    om = OTTER_KN * KN / OTTER_R
    th = math.radians(OTTER_AZ0) + om * (t - T_SHOT)
    x, z = OTTER_R * math.cos(th), OTTER_R * math.sin(th)
    v = OTTER_KN * KN
    vx, vz = -v * math.sin(th), v * math.cos(th)
    ax, az = -om * om * x, -om * om * z                  # centripetal feed-forward
    hd = -(th + math.pi / 2)                             # vessel +X along the tangent
    return x, z, hd, vx, vz, -om, ax, az


FOAM_LACE = 0.0                    # wash along each hull's waterline (0 = off): 0.1 and 0.4 showed no visible foam at sea level


def wash(F):
    """Re-seed the foam sources each frame: the site's ring at the TP (turbine_site.build_site, copied)
    plus a thin lace along both boats' sides, so a hull that moves takes its wash with it."""
    oc = F.ocean
    oc.clear_foam_disturbances()
    for k in range(28):
        a = 2 * math.pi * k / 28
        oc.add_foam_disturbance(5.75 * math.cos(a), 5.75 * math.sin(a), 0.55, 0.12)
    for boat, xs, zs in ((F.mariner, np.arange(-2.6, 2.61, 0.65), (-1.05, 1.05)),
                         (F.otter, np.arange(-2.0, 2.01, 0.5), (-1.12, -0.66, 0.66, 1.12))):
        for x in xs:
            for z in zs:
                q = boat.to_world([x, 0.4, z])
                oc.add_foam_disturbance(float(q[0]), float(q[2]), 0.35, F.foam_lace)


def build_fleet(renderer, S):
    F = types.SimpleNamespace()
    scene, ocean = S.scene, S.ocean
    F.ocean = ocean

    def load_boat(name):
        g = tp.GLTFLoader().load(os.path.join(USV_DIR, f"{name}.glb")).scene
        g.traverse(lambda o: (setattr(o, "cast_shadow", True), setattr(o, "receive_shadow", True))
                   if isinstance(o, tp.Mesh) else None)
        scene.add(g)
        return g

    def livery(o):
        # AgX (this scene) pulls the spec's (0.62, 0.165, 0.068) to salmon; deeper chroma reads as safety orange
        if isinstance(o, tp.Mesh) and "orange" in o.material.name:
            o.material.color = tp.Color(0.78, 0.085, 0.004)
            o.material.roughness = 0.55
    F.mariner_obj = load_boat("mariner")
    F.mariner_obj.traverse(livery)
    F.otter_obj = load_boat("otterx")
    F.otter_obj.traverse(livery)
    F.mariner = StripHull("mariner", ocean)
    F.otter = StripHull("otterx", ocean)
    a = math.radians(LANDING_AZ)
    F.mariner_xz = (MARINER_R * math.cos(a), MARINER_R * math.sin(a))
    F.seated = False
    F.vehicles = fv.build_vehicles(scene, S, F)

    def seat():
        x, z = F.mariner_xz
        F.mariner.seat(x, z, MARINER_HDG)
        ox, oz, ohd, *_ = otter_on_circle(F.t)
        F.otter.seat(ox, oz, ohd)
        F.seated = True

    def update(t, dt, substeps=2):
        """Per frame, AFTER a render at sim time t (sample_height reads that frame's sea)."""
        F.t = t
        if not F.seated:
            seat()
        ox, oz, ohd, vx, vz, rate, ax, az = otter_on_circle(t)
        F.otter.set_target(ox, oz, ohd, vx, vz, rate, ax, az)
        if not F.otter.log:                             # start her moving with the target
            F.otter.v = np.array([vx, 0.0, vz])
            F.otter.w = np.array([0.0, rate, 0.0])
        if dt > 0.0:
            F.mariner.advance(dt, substeps)
            F.otter.advance(dt, substeps)
        F.mariner.footprint(ocean)
        F.mariner.place(F.mariner_obj)
        F.otter.place(F.otter_obj)
        if F.foam_lace:
            wash(F)
        fv.update_vehicles(F.vehicles, S, F, t, dt)

    F.foam_lace = FOAM_LACE

    F.t = T_SHOT
    F.update = update
    return F


# --------------------------------------------------------------------------- #
#  Cameras (names are the contract): name -> (fn(F) -> (eye, target, vfov), underwater, exposure)
# --------------------------------------------------------------------------- #
def _cam_mariner(F):
    p = F.mariner.p
    return ts.pol(38.0, 62.0, 2.6), np.array([0.0, 3.0, 0.45 * p[2]]), 40.0


def _cam_otter(F):
    p = F.otter.p
    d = p[[0, 2]] / np.linalg.norm(p[[0, 2]])
    eye = np.array([p[0] + 14.0 * d[0] - 4.0 * d[1], 1.6, p[2] + 14.0 * d[1] + 4.0 * d[0]])
    return eye, np.array([0.55 * p[0], 3.0, 0.55 * p[2]]), 40.0


def _cam_aerial(F):
    """35 m up, 80 m out between the boats: both boats, the transition piece, the Mariner's pile side."""
    return ts.pol(80.0, 35.0, 35.0), np.array([12.0, 1.0, 3.0]), 40.0


CAMS = {
    "mariner_station": (_cam_mariner, False, 0.8),
    "otter_survey": (_cam_otter, False, 0.8),
    "fleet_aerial": (_cam_aerial, False, 0.8),
    "drone_blade": (lambda F: fv.drone_chase(F.vehicles), False, 0.6),
    "rov_pile": (lambda F: fv.rov_cam(F.vehicles), True, 1.4),
    "snake_cps": (lambda F: fv.snake_cam(F.vehicles), True, 1.4),
}


def apply_camera(renderer, S, F, camera, name):
    fn, under, expo = CAMS[name]
    eye, tgt, fov = fn(F)
    camera.fov = fov
    camera.near, camera.far = (0.03, 20000.0) if under else (0.3, 20000.0)
    camera.update_projection_matrix()
    camera.position.set(*map(float, eye))
    camera.look_at(tp.Vector3(*map(float, tgt)))
    renderer.tone_mapping_exposure = expo
    S.set_lamps(eye, tgt, False)                        # the vehicles bring their own light
    return under, eye


def main():
    shot = cli_arg("--shot", "all", str)
    W, H = parse_size(cli_arg("--size", "1920x1080", str))
    t_shot = cli_arg("--time", T_SHOT, float)
    settle = cli_arg("--settle", 10.0, float)
    canvas = tp.Canvas("threepp - offshore fleet", width=W, height=H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.AgX
    renderer.tone_mapping_exposure = 0.8
    renderer.sun_angular_radius = 0.6
    renderer.bloom_intensity = 0.06
    renderer.auto_exposure = False
    t0 = time.perf_counter()
    S = ts.build_site(renderer, sheet=cli_arg("--sheet", 12000.0, float))
    camera = tp.PerspectiveCamera(45.0, W / H, 0.1, 6000.0)
    t_start = t_shot - settle
    global FOAM_LACE
    FOAM_LACE = cli_arg("--foam-lace", FOAM_LACE, float)
    fv.ROV_SPOT_I = cli_arg("--rov-lamp", fv.ROV_SPOT_I, float)
    fv.SNAKE_LAMP_I = cli_arg("--snake-lamp", fv.SNAKE_LAMP_I, float)
    F = build_fleet(renderer, S)
    F.t = t_start
    # the sea mesh spacing at the boats (cubic warp, turbine_site.build_site)
    h = 0.5 * cli_arg("--sheet", 12000.0, float)
    a = S.ocean.warp.coef_a
    for nm, r in (("mariner", MARINER_R), ("otterx", OTTER_R)):
        tt = np.linspace(0.0, 1.0, 200001)
        xs = h * (a * tt + (1.0 - a) * tt ** 3)
        sp = h * (a + 3.0 * (1.0 - a) * tt ** 2) * 2.0 / (ts.OCEAN_RES - 1)
        print(f"[fleet] sea mesh spacing at the {nm} (r {r:.0f} m): {float(np.interp(r, xs, sp)):.2f} m")

    first = shot.split(",")[0] if shot != "all" else "mariner_station"
    apply_camera(renderer, S, F, camera, first)
    # settle the boats: the sea advances frame by frame, the hulls ride it (dt 1/120, 2 per frame)
    n = int(round(settle * 60.0))
    for k in range(n + 1):
        t = t_start + k / 60.0
        renderer.sim_time = t
        S.update(t, camera, False, 0.0)
        renderer.render(S.scene, camera)
        F.update(t, 1.0 / 60.0 if k < n else 0.0)
        if k in (n - 120, n - 60, n):
            for bt in (F.mariner, F.otter):
                r = bt.readout()
                print(f"[fleet] t {t:5.2f} {bt.boat:8s} heave {bt.p[1]:+.3f} draft {r['draft']:.3f} trim {r['trim']:+.2f} "
                      f"roll {r['roll']:+.2f} hdg {r['heading']:+.1f} xz ({bt.p[0]:.2f}, {bt.p[2]:.2f})")
    skip = int(2.0 * 60)
    print("[fleet] settled in %.1f s" % (time.perf_counter() - t0))
    print("[fleet] " + F.mariner.summary(skip))
    print("[fleet] " + F.otter.summary(skip))

    def shoot(name, out):
        t1 = time.perf_counter()
        under, eye = apply_camera(renderer, S, F, camera, name)
        S.update(t_shot, camera, under, 0.0)
        if under:
            S.settle_snow(camera, t_shot)
        for i in range(4):                              # a headless render needs a few frames to settle
            apply_camera(renderer, S, F, camera, name)
            S.update(t_shot - (3 - i) / 60.0, camera, under)
            F.update(t_shot, 0.0)
            renderer.render(S.scene, camera)
        renderer.sim_time = t_shot
        d = os.path.dirname(out) or "."
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "_tmp_" + os.path.basename(out))
        renderer.save_frame(S.scene, camera, tmp)
        os.replace(tmp, out)
        print(f"[fleet] {name}: wrote {out} in {time.perf_counter() - t1:.1f} s")

    if shot == "all" or "," in shot:
        outdir = cli_arg("--outdir", ts.out_dir("fleet"), str)
        tag = cli_arg("--tag", "auto", str)            # auto: the next free _vNN per camera
        names = list(CAMS) if shot == "all" else shot.split(",")
        for name in names:
            k = 1
            while tag == "auto" and os.path.exists(os.path.join(outdir, f"{name}_v{k:02d}.png")):
                k += 1
            out = os.path.join(outdir, f"{name}_{tag if tag != 'auto' else 'v%02d' % k}.png")
            if os.path.exists(out):
                print(f"{out} exists; renders are never overwritten, skipped")
                continue
            shoot(name, out)
    else:
        if shot not in CAMS:
            sys.exit(f"unknown shot {shot!r}; one of {', '.join(CAMS)} or all")
        out = cli_arg("--out", os.path.join(ts.out_dir("fleet"), f"{shot}.png"), str)
        if os.path.exists(out):
            sys.exit(f"{out} exists; renders are never overwritten")
        shoot(shot, out)


if __name__ == "__main__":
    main()
