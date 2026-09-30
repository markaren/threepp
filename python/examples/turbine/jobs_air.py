"""Phase 3A, the air and surface jobs at the offshore turbine.

  drone: an autonomous blade inspection flight, rotor locked. Lift off the Mariner's pad, climb
         clear, transit to blade_1's tip, fly up its leading edge at a 9 m stand-off with the
         gimbal on the blade, hold at the erosion patch. Its gimbal camera is a PiP.
  otter: the Otter X runs her 40 m survey circle with a multibeam fan (see survey()).

    python jobs_air.py --job drone [--tag v01]      the drone_job clip
    python jobs_air.py --job survey                  the soundings, the map, the numbers
    python jobs_air.py --job otter                   the otter_survey clip (needs the survey npz)

For the film: build_drone_job(S, F) -> D; D.pose(tau) sets the drone for route time tau (call
before F.update), D.chase(tau) -> (eye, aim, fov), D.gimbal_cam(tau) -> (eye, aim, fov).
Headless only. Frame: X upwind, Y up, origin on the pile axis at MSL (turbine_site.py).
"""
import math
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import threepp as tp
from demo_common import Encoder, cli_arg
from drone_rig import Leg, LegRoute, Route, trap, join_cubic, blur, smoothstep_f

import turbine_site as ts
import turbine_fleet as tf
import fleet_vehicles as fv

OUT = ts.out_dir("jobs")
W, H, FPS = 1920, 1080, 30
UP = np.array([0.0, 1.0, 0.0])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
#  Files: versioned names, written as _tmp_* and renamed
# --------------------------------------------------------------------------- #
def versioned(stem, ext, outdir=OUT):
    os.makedirs(outdir, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(outdir, f"{stem}_v{k:02d}.{ext}")):
        k += 1
    return os.path.join(outdir, f"{stem}_v{k:02d}.{ext}")


def save_png(rgb, path):
    from PIL import Image
    tmp = os.path.join(os.path.dirname(path), "_tmp_" + os.path.basename(path))
    Image.fromarray(np.ascontiguousarray(rgb)).save(tmp)
    os.replace(tmp, path)
    print(f"[jobs] wrote {path}")


def contact_sheet(frames, path, cols=4, w=480):
    from PIL import Image
    ims = [Image.fromarray(f).resize((w, int(w * f.shape[0] / f.shape[1]))) for f in frames]
    h = ims[0].size[1]
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4), (20, 20, 20))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    save_png(np.asarray(sheet), path)


def pip_frame(rgb, x, y, w, h, label):
    """A thin frame and a small label round the composited PiP (drawn on the read-back frame)."""
    from PIL import Image, ImageDraw, ImageFont
    rgb[y - 2:y, x - 2:x + w + 2] = 235
    rgb[y + h:y + h + 2, x - 2:x + w + 2] = 235
    rgb[y - 2:y + h + 2, x - 2:x] = 235
    rgb[y - 2:y + h + 2, x + w:x + w + 2] = 235
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([x, y, x + 132, y + 28], fill=(12, 14, 16))
    d.text((x + 9, y + 3), label, fill=(235, 235, 235), font=font)
    return np.asarray(im).copy()


# --------------------------------------------------------------------------- #
#  The site + fleet, headless
# --------------------------------------------------------------------------- #
def make_renderer(w, h):
    canvas = tp.Canvas("threepp - turbine jobs", width=w, height=h, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.AgX
    renderer.tone_mapping_exposure = 0.8
    renderer.sun_angular_radius = 0.6
    renderer.bloom_intensity = 0.06
    renderer.auto_exposure = False
    return canvas, renderer


def settle(renderer, S, F, camera, t0, t1, rate=30.0):
    """Ride the boats from t0 to t1 on the rendered sea (one render per step)."""
    n = int(round((t1 - t0) * rate))
    for k in range(n):
        t = t0 + k / rate
        renderer.sim_time = t
        S.update(t, camera, False, 0.0)
        renderer.render(S.scene, camera)
        F.update(t, 1.0 / rate)
    return t0 + n / rate


# --------------------------------------------------------------------------- #
#  The drone job
# --------------------------------------------------------------------------- #
STANDOFF = 9.0                   # m off the leading edge (industry: 8-10 m)
DEFECT_SPAN = float(ts.spec.get("blade_detail", {}).get("erosion", {}).get("span_m", 82.0))
L_BLADE = 117.0
T_HOLD_PAD, T_CLIMB, T_TRANSIT, T_TIP, T_RUN, T_DEF = 1.0, 4.0, 12.0, 2.0, 14.0, 14.0
SKID = 0.37                      # drone root above the pad (industrial_kit skids reach -0.36)


def le_point(span):
    """Leading edge of blade_1 at `span` m from the root, and its outward normal (world)."""
    L = L_BLADE
    s = span / L
    tab = np.asarray(ts.tb.val(ts.spec["blade"]["planform"]), float)
    c, tw, t, pa = (float(np.interp(s, tab[:, 0], tab[:, k])) for k in (1, 2, 3, 4))
    sec = ts.tb.airfoil_section(c, tw, t, pa, 180)
    i = int(np.argmax(sec[:, 1]))
    xoff = ts.tb.val(ts.spec["blade"]["tip_prebend"]) * max((s - 0.2) / 0.8, 0.0) ** 2
    M = ts._TP @ ts.tb.pitch_node(ts.spec, 1)
    y = -1.0 + s * (L + 1.0)
    p = (M @ np.array([sec[i, 0] + xoff, y, sec[i, 1], 1.0]))[:3]
    ax = ts.blade_point(1, span)
    n = p - ax
    up = _unit(ts.blade_point(1, span - 1.0) - ts.blade_point(1, span + 1.0))
    n = _unit(n - up * np.dot(n, up))
    return p, n


def build_drone_job(S, F):
    """The authored route in route time tau (0 = on the pad, spooling). pad0: the pad at tau 0."""
    D = types.SimpleNamespace()
    V = F.vehicles
    drone = V["drone"]

    def pad_world(ext=0.0):
        b = F.mariner
        p = b.p + b.v * ext
        return p + b.R @ (np.array([fv.PAD_X, V["deck_y"] + 0.012 + SKID, 0.0]) - b.cog)
    D.pad_world = pad_world
    P0 = pad_world()
    D.pad0 = P0.copy()
    top = P0 + np.array([0.0, 9.0, 0.0])
    tip, n_tip = le_point(L_BLADE - 0.5)
    A = tip + STANDOFF * n_tip
    dfx, n_d = le_point(DEFECT_SPAN)
    B = dfx + STANDOFF * n_d
    D.defect, D.defect_n = dfx, n_d

    def run_pos(u):                 # up the leading edge at the stand-off
        sp = (L_BLADE - 0.5) + u * (DEFECT_SPAN - (L_BLADE - 0.5))
        p, n = le_point(sp)
        return p + STANDOFF * n

    def run_aim(u, p):
        sp = (L_BLADE - 0.5) + trap(0.2, 0.3)(u) * (DEFECT_SPAN - (L_BLADE - 0.5))
        return le_point(sp)[0]
    look0 = tip
    deck = P0 + np.array([7.0, -4.0, -2.0])          # gimbal down at the sea off the bow while it lifts
    legs = LegRoute([
        Leg(T_HOLD_PAD, pos=P0, aim=deck),
        Leg(T_CLIMB, pos=(P0, top), aim=(deck, look0), warp=trap(0.35, 0.35)),
        Leg(T_TRANSIT, pos=join_cubic(top, A, _unit(A - top) * np.array([1, 0.3, 1]), _unit(A - top)), aim=look0,
            warp=trap(0.3, 0.35)),
        Leg(T_TIP, pos=A, aim=tip),
        Leg(T_RUN, pos=run_pos, aim=run_aim, warp=trap(0.2, 0.3)),
        Leg(T_DEF, pos=B, aim=dfx),
    ])
    D.route = Route(legs, sea_y=-100.0, smooth=1.0)
    D.T_RUN0 = T_HOLD_PAD + T_CLIMB + T_TRANSIT + T_TIP
    D.T_DEF0 = D.T_RUN0 + T_RUN
    D.duration = legs.duration
    path = D.route.path
    v = np.diff(path[:, :3], axis=0) / D.route.dt
    a = np.diff(v, axis=0) / D.route.dt
    a = blur(np.linalg.norm(a, axis=1), 3)
    D.peak_v, D.peak_a = float(np.linalg.norm(v, axis=1).max()), float(a.max())
    print(f"[drone] route {D.duration:.0f} s, {np.sum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)):.0f} m; "
          f"peak speed {D.peak_v:.2f} m/s, peak acceleration {D.peak_a:.2f} m/s^2; tip stand-off {np.round(A, 1)}, "
          f"defect {np.round(dfx, 1)} (span {DEFECT_SPAN:.0f} m)")
    # the chase: beside and below the drone, sun behind the camera (+Z), the blade off to one side
    off_run = np.array([5.0, -3.2, 10.0])
    D.off_run = off_run

    def pose(tau):
        """Set the drone for route time tau (before F.update); lifts off the moving pad."""
        p, aim = D.route.pose(tau)
        w = smoothstep_f(T_HOLD_PAD - 0.2, T_HOLD_PAD + 2.0, tau)
        p = p + (1.0 - w) * (pad_world(1.0 / FPS) - D.pad0)
        V["drone_pos"], V["drone_look"] = p, aim
        drone.face_w = smoothstep_f(D.T_RUN0 - 3.0, D.T_RUN0, tau)
        return p, aim
    D.pose = pose

    def chase(tau):
        p, aim = D.route.pose(tau)
        if tau < T_HOLD_PAD + T_CLIMB:              # lift-off: from the Mariner's quarter, low
            eye = D.pad0 + np.array([7.5, 1.8, 9.0]) + np.array([0.0, 0.25, 0.0]) * tau
            tgt = D.pad0 + np.array([0.0, 1.2 + 0.45 * tau, 0.0])
            return eye, tgt, 34.0
        # the run: above, beside and a little out from the drone, looking down the blade; the LE
        # runs down the left of the frame, the drone sits right against the sea far below
        blade = le_point(max(DEFECT_SPAN, min(L_BLADE, L_BLADE - np.dot(p - A, _unit(B - A)))))[0]
        upb = _unit(ts.blade_point(1, 80.0) - ts.blade_point(1, 90.0))
        eye = p + 0.5 * n_d + 2.0 * upb + np.array([0.0, 0.0, 5.0])
        return eye, p + 0.4 * (blade - p) - 2.0 * upb, 48.0
    D.chase = chase

    def gimbal_cam(tau):
        drone.root.update_matrix_world(True)
        g = drone.gimbal.get_world_position()          # the pod on the gimbal, under the nose
        fov = 50.0 + (15.0 - 50.0) * smoothstep_f(T_HOLD_PAD + 2.0, T_HOLD_PAD + T_CLIMB + 4.0, tau)   # wide, then the zoom
        return np.array([g.x, g.y - 0.05, g.z]), V["drone_look"], fov
    D.gimbal_cam = gimbal_cam
    return D


def film_drone(tag=None):
    t_all = time.perf_counter()
    canvas, renderer = make_renderer(W, H)
    S = ts.build_site(renderer)
    camera = tp.PerspectiveCamera(40.0, W / H, 0.3, 20000.0)
    camera.far = 20000.0
    F = tf.build_fleet(renderer, S)
    t0 = ts.T_SHOT - 8.0
    F.t = t0
    S.apply_camera(camera, "site_far")
    tf.apply_camera(renderer, S, F, camera, "mariner_station")
    renderer.sim_time = t0
    renderer.render(S.scene, camera)
    T0 = settle(renderer, S, F, camera, t0, t0 + 8.0)          # boats on the sea; T0: route time 0
    D = build_drone_job(S, F)
    drone = F.vehicles["drone"]
    drone.have = False
    renderer.tone_mapping_exposure = 0.7
    # the gimbal camera, composited on the device
    PW, PH, M = 640, 360, 36
    pip = tp.PerspectiveCamera(13.0, PW / PH, 0.3, 20000.0)
    view = renderer.add_view(pip, PW, PH)
    ok = view and renderer.set_view_display_rect(view, W - M - PW, H - M - PH, PW, PH)
    print(f"[drone] PiP view {view}, composite {'on' if ok else 'OFF'}; setup {time.perf_counter() - t_all:.0f} s")

    # film: 0-3 s lift-off (tau 0.3..3.3), jump cut, 3-14 s the run (tau T_RUN0 + 7 - 11 .. )
    CUT = 3.0
    TAU_B = D.T_DEF0 - 7.0
    n = int(round(14.0 * FPS))
    out = versioned("drone_job", "mp4") if tag is None else os.path.join(OUT, f"drone_job_{tag}.mp4")
    stem = os.path.basename(out)[:-4]
    tmp = os.path.join(OUT, "_tmp_" + os.path.basename(out))
    enc = Encoder(tmp, W, H, FPS, crf=18, preset="medium", faststart=True)
    picks = set(np.linspace(0, n - 1, 8).round().astype(int))
    keys = {int(1.5 * FPS): "liftoff", int(8.0 * FPS): "run", n - 15: "hold"}
    sheet, stills = [], {}
    dt = 1.0 / FPS
    tau_prev = None
    for k in range(n):
        tfilm = k / FPS
        tau = 0.3 + tfilm if tfilm < CUT else TAU_B + (tfilm - CUT)
        if tau_prev is not None and tau - tau_prev > 2 * dt:           # the jump cut: ride the skipped time
            t_a = T0 + tau_prev + dt
            for j in range(int(round((tau - tau_prev - dt) * FPS))):
                tt = t_a + j * dt
                D.pose(tt - T0 + dt)
                renderer.sim_time = tt
                S.update(tt, camera, False, 0.0)
                renderer.render(S.scene, camera)
                F.update(tt, dt)
            print(f"[drone] cut: rode {tau - tau_prev:.1f} s of sim time")
        tau_prev = tau
        t = T0 + tau
        eye, tgt, fov = D.chase(tau)
        camera.fov = fov
        camera.update_projection_matrix()
        camera.position.set(*map(float, eye))
        camera.look_at(tp.Vector3(*map(float, tgt)))
        ge, gt, gf = D.gimbal_cam(tau)
        pip.fov = gf
        pip.update_projection_matrix()
        pip.position.set(*map(float, ge))
        pip.look_at(tp.Vector3(*map(float, gt)))
        renderer.sim_time = t
        S.update(t, camera, False, dt)
        renderer.render(S.scene, camera)
        rgb = renderer.read_pixels()
        if ok:
            rgb = pip_frame(rgb, W - M - PW, H - M - PH, PW, PH, "DRONE CAM")
        enc.send(rgb)
        if k in picks:
            sheet.append(rgb.copy())
        if k in keys:
            stills[keys[k]] = rgb.copy()
        D.pose(tau + dt)
        F.update(t, dt)
        if k % 60 == 0:
            print(f"[drone] frame {k}/{n} tau {tau:.1f} drone {np.round(F.vehicles['drone_pos'], 1)}")
    enc.close()
    os.replace(tmp, out)
    print(f"[drone] wrote {out} in {time.perf_counter() - t_all:.0f} s")
    v = stem.split("_")[-1]
    contact_sheet(sheet, os.path.join(OUT, f"drone_job_sheet_{v}.png"))
    for name, img in stills.items():
        save_png(img, os.path.join(OUT, f"drone_job_{name}_{v}.png"))
    renderer.remove_view(view)


# --------------------------------------------------------------------------- #
#  The Otter X survey: a multibeam fan traced through the renderer's TLAS (scan_lidar beam table)
# --------------------------------------------------------------------------- #
MB_BEAMS, MB_SWATH = 256, 120.0          # across-track fan (deg), equiangular
MB_RATE = 15.0                           # pings per second (every other 30 Hz step)
MB_SIGMA = 0.04                          # range noise, m (1 sigma, Gaussian, seed 3)
MB_HEAD = (1.62, -0.25)                  # transducer on the mbes_mount pole: vessel x, and y below the keel
CELL, R_MAP = 0.5, 60.0
SURVEY_T = 2 * math.pi * tf.OTTER_R / (tf.OTTER_KN * tf.KN)     # one full circle, s
NPZ = os.path.join(OUT, "survey_soundings.npz")


def mb_fan():
    th = np.radians(np.linspace(-0.5 * MB_SWATH, 0.5 * MB_SWATH, MB_BEAMS))
    return np.stack([np.zeros_like(th), -np.cos(th), np.sin(th)], 1)         # vessel frame: down, across (+Z)


def transducer(boat):
    return np.array([MB_HEAD[0], boat.lowest + MB_HEAD[1], 0.0])


def survey_pass(size=(320, 180)):
    """Ride the Otter once round her circle on the rendered sea; ping at MB_RATE. Saves the raw
    ranges with each ping's true pose to NPZ."""
    t_all = time.perf_counter()
    canvas, renderer = make_renderer(*size)
    S = ts.build_site(renderer)
    camera = tp.PerspectiveCamera(40.0, size[0] / size[1], 0.3, 20000.0)
    F = tf.build_fleet(renderer, S)
    V = F.vehicles
    # the ROV, its tether and the snake are recovered for the survey (their echoes are not the bed)
    for o in [V["rov"]] + list(V["snake_links"]):
        o.visible = False
    t0 = ts.T_SHOT - 8.0
    F.t = t0
    tf.apply_camera(renderer, S, F, camera, "fleet_aerial")
    renderer.sim_time = t0
    renderer.render(S.scene, camera)
    if V["tether"] is not None:
        V["tether"].visible = False
    T0 = settle(renderer, S, F, camera, t0, t0 + 8.0)
    fan = mb_fan()
    prm = tp.LidarParams()
    prm.max_range = 200.0
    prm.detector_threshold = 0.0
    prm.medium_extinction = 0.0                          # acoustic: no optical murk in the water column
    # instance ids, for the hit table (the sonar keys on them)
    ids = {"seabed": (S.seabed, 1), "seabed_close": (S.seabed_close, 2), "seabed_far": (S.seabed_far, 3), "far_ring": (S.far, 5),
           "turbine": (S.turbine, 10), "cps": (S.cps, 11), "ocean": (S.ocean, 20), "otter": (F.otter_obj, 30),
           "mariner": (F.mariner_obj, 31), "motes": (S.motes, 40)}
    for nm, (o, i) in ids.items():
        o.traverse(lambda m, i=i: renderer.set_instance_id(m, i) if isinstance(m, tp.Mesh) else None)
        try:
            renderer.set_instance_id(o, i)
        except TypeError:
            pass
    names = {i: nm for nm, (o, i) in ids.items()}

    def hit_table(ret, o, R):
        iid, kind, dist = ret["instance_id"], ret["return_kind"], ret["distance"]
        ok = ret["return_no"] > 0
        y = o[1] + dist * (fan @ R.T)[:, 1]
        print("[survey] one ping, hits by object: " + ", ".join(
            f"{names.get(int(i), 'id %d' % i)}/kind {int(k)}: {int(((iid == i) & (kind == k) & ok).sum())} beams, y {np.nanmin(np.where((iid == i) & (kind == k) & ok, y, np.nan)):.2f}"
            f"..{np.nanmax(np.where((iid == i) & (kind == k) & ok, y, np.nan)):.2f}"
            for i, k in sorted(set(zip(iid[ok].tolist(), kind[ok].tolist())))) + f"; no return {int((~ok).sum())}")
    dt = 1.0 / 30.0
    n = int(math.ceil(SURVEY_T / dt))
    O, R, T, RNG, NVOL = [], [], [], [], [0]
    batch_o, batch_d = [], []

    def flush():
        if not batch_o:
            return
        o = np.concatenate(batch_o).astype(np.float32)
        d = np.concatenate(batch_d).astype(np.float32)
        ret = renderer.scan_lidar(o, d, prm)
        ok = (ret["return_no"] > 0) & (ret["return_kind"] == 0)      # kind != 0: water-column (murk) scatter
        NVOL[0] += int(((ret["return_no"] > 0) & (ret["return_kind"] != 0)).sum())
        RNG.append(np.where(ok, ret["distance"], np.nan).reshape(-1, MB_BEAMS))
        batch_o.clear()
        batch_d.clear()
    t_scan = 0.0
    t_loop = time.perf_counter()
    for k in range(n):
        t = T0 + k * dt
        renderer.sim_time = t
        S.update(t, camera, False, 0.0)
        tf.apply_camera(renderer, S, F, camera, "otter_survey")
        renderer.render(S.scene, camera)
        F.update(t, dt)
        if V["tether"] is not None:
            V["tether"].visible = False
        if k % 2 == 0:                                   # a ping at t + dt (the pose F.update just set)
            b = F.otter
            o = b.to_world(transducer(b))
            if k in (0, 2400, 4800):
                hit_table(renderer.scan_lidar(np.repeat(o[None], MB_BEAMS, 0).astype(np.float32),
                                              (fan @ b.R.T).astype(np.float32), prm), o, b.R)
            O.append(o)
            R.append(b.R.copy())
            T.append(t + dt)
            batch_o.append(np.repeat(o[None], MB_BEAMS, 0))
            batch_d.append(fan @ b.R.T)
            if len(batch_o) >= 64:
                s0 = time.perf_counter()
                flush()
                t_scan += time.perf_counter() - s0
        if k % 900 == 0:
            r = F.otter.readout()
            print(f"[survey] {k}/{n} t {t - T0:6.1f} s roll {r['roll']:+.2f} trim {r['trim']:+.2f} heave {F.otter.p[1]:+.3f} "
                  f"({time.perf_counter() - t_loop:.0f} s)")
    flush()
    wall = time.perf_counter() - t_all
    print(f"[survey] {len(O)} pings x {MB_BEAMS} beams in {wall:.0f} s wall ({time.perf_counter() - t_loop:.0f} s loop, "
          f"{t_scan:.1f} s tracing); {NVOL[0]} water-column returns dropped")
    os.makedirs(OUT, exist_ok=True)
    tmp = NPZ[:-4] + "_tmp_.npz"
    np.savez_compressed(tmp, origin=np.array(O), R=np.array(R), t=np.array(T) - T0, T0=T0, rng=np.concatenate(RNG),
                        fan=fan, wall=wall, lowest=F.otter.lowest)
    os.replace(tmp, NPZ)
    print(f"[survey] wrote {NPZ}")


def soundings(D, comp=True, seed=3):
    """World soundings (N, 3) from the raw ranges: with the true attitude (comp) or with the boat
    assumed level at her mean transducer depth (heading and GNSS position kept)."""
    rng = np.random.default_rng(seed)
    r = D["rng"] + rng.normal(0.0, MB_SIGMA, D["rng"].shape)
    fan, R, O = D["fan"], D["R"], D["origin"]
    if comp:
        dirs = np.einsum("pij,bj->pbi", R, fan)
        org = O
    else:
        hd = np.arctan2(-R[:, 2, 0], R[:, 0, 0])                    # heading of the vessel +X (rot_y)
        c, s = np.cos(hd), np.sin(hd)
        Rl = np.zeros_like(R)
        Rl[:, 0, 0], Rl[:, 0, 2], Rl[:, 1, 1], Rl[:, 2, 0], Rl[:, 2, 2] = c, s, 1.0, -s, c
        dirs = np.einsum("pij,bj->pbi", Rl, fan)
        org = O.copy()
        org[:, 1] = O[:, 1].mean()
    P = org[:, None, :] + r[..., None] * dirs
    return P.reshape(-1, 3), np.repeat(D["t"], fan.shape[0])


def grid_map(P, upto=None, tt=None):
    n = int(2 * R_MAP / CELL)
    ok = np.isfinite(P[:, 1])
    if upto is not None:
        ok &= tt <= upto
    i = np.floor((P[ok, 0] + R_MAP) / CELL).astype(int)
    j = np.floor((P[ok, 2] + R_MAP) / CELL).astype(int)
    k = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    cell = j[k] * n + i[k]
    y = P[ok, 1][k]
    o = np.lexsort((y, cell))                            # per-cell median (what a multibeam processor grids)
    cell, y = cell[o], y[o]
    u, st, cnt = np.unique(cell, return_index=True, return_counts=True)
    med = 0.5 * (y[st + (cnt - 1) // 2] + y[st + cnt // 2])
    h = np.full(n * n, np.nan)
    c = np.zeros(n * n)
    h[u], c[u] = med, cnt
    return h.reshape(n, n), c.reshape(n, n)


def truth_grid():
    n = int(2 * R_MAP / CELL)
    sub = (np.arange(5) + 0.5) / 5 * CELL
    xs = -R_MAP + np.arange(n)[:, None] * CELL + sub[None, :]
    X, Z = np.meshgrid(xs.reshape(-1), xs.reshape(-1))
    h = ts.seabed_height(X, Z).reshape(n, 5, n, 5).mean(axis=(1, 3))
    return h


def map_mask():
    """True where the map is scored: inside R_MAP, off the pile (and its growth) and the cable sleeve."""
    n = int(2 * R_MAP / CELL)
    c = -R_MAP + (np.arange(n) + 0.5) * CELL
    X, Z = np.meshgrid(c, c)
    r = np.hypot(X, Z)
    cab = ts.cable_path()[0]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(cab[:, [0, 2]]).query(np.stack([X.ravel(), Z.ravel()], 1))
    keep = (r < R_MAP) & (r > ts.PILE_R + 0.8) & (d.reshape(n, n) > 1.2)
    return keep, X, Z, r


def berm_numbers(h, X, Z, r, keep):
    phi = np.arctan2(Z, X)

    def away(c, half):
        return np.abs((phi - c + np.pi) % (2 * np.pi) - np.pi) > math.radians(half)
    sect = keep & away(ts.CURRENT_AZ, 60.0) & away(ts.DAMAGE_AZ, 25.0) & away(ts.CABLE_AZ, 15.0) & np.isfinite(h)
    rb = np.arange(8.0, 34.0, 0.5)
    prof = np.array([np.nanmedian(h[sect & (r >= a) & (r < a + 0.5)]) for a in rb]) if sect.any() else rb * np.nan
    top = np.nanmedian(prof[(rb >= 10.0) & (rb < 18.0)])
    sand = np.nanmedian(prof[(rb >= 28.0) & (rb < 33.0)])
    half = 0.5 * (top + sand)
    i = np.where((prof[:-1] >= half) & (prof[1:] < half))[0]
    rad = float(rb[i[-1]] + 0.25 + 0.5 * (prof[i[-1]] - half) / (prof[i[-1]] - prof[i[-1] + 1])) if len(i) else float("nan")
    # the edge-scour pit: the lowest cell in its sector against the sand beyond it on the same sector
    pit_s = keep & ~away(ts.CURRENT_AZ, 15.0) & np.isfinite(h)
    low = np.nanmin(np.where(pit_s & (r > ts.R_ARMOUR) & (r < ts.PIT_R + 6.0), h, np.nan))
    ref = np.nanmedian(np.where(pit_s & (r > ts.PIT_R + 11.0) & (r < ts.PIT_R + 16.0), h, np.nan))
    return dict(radius=rad, thick=float(top - sand), pit=float(ref - low), top=float(top), sand=float(sand))


def survey_numbers():
    D = dict(np.load(NPZ))
    keep, X, Z, r = map_mask()
    truth = truth_grid()
    fine = keep & (np.abs(X) < ts.FINE_HALF - 1.0) & (np.abs(Z) < ts.FINE_HALF - 1.0)
    rows = {}
    maps = {}
    for name, comp in (("compensated", True), ("no attitude", False)):
        P, tt = soundings(D, comp)
        h, cnt = grid_map(P)
        maps[name] = (h, cnt)
        e = np.abs(h - truth)
        rows[name] = {}
        zones = (("all r<60", keep), ("fine mesh", fine), ("sand r>27", keep & (r > 27.0)), ("rock berm r<22", keep & (r < 22.0)))
        for zone, m in zones:
            ee = e[m & np.isfinite(e)]
            rows[name][zone] = (float(ee.mean()), float(np.percentile(ee, 95)), float((m & np.isfinite(h)).sum() / m.sum()))
        rows[name]["berm"] = berm_numbers(h, X, Z, r, keep)
    tb_ = berm_numbers(truth, X, Z, r, keep)
    lines = [f"Otter X multibeam survey, one 40 m circle at 2 kn ({SURVEY_T:.0f} s), {len(D['t'])} pings x {MB_BEAMS} beams, "
             f"{MB_SWATH:.0f} deg swath, {MB_RATE:.0f} Hz, range noise sigma {MB_SIGMA * 100:.0f} cm (seed 3)",
             "rays: renderer.scan_lidar beam table through the render TLAS (the SonarSensor's tracer), surface returns only "
             "(return_kind 0), medium_extinction 0; ROV, tether and snake recovered for the pass",
             "gridding: per-cell MEDIAN of the soundings, 0.5 m cells to r 60 m, pile and CPS masked; truth = seabed_height "
             "averaged over each cell (5x5)",
             "no attitude = the boat assumed level at her mean transducer depth; heading and GNSS position kept",
             f"survey pass wall time {float(D['wall']):.0f} s at 320x180", "",
             f"{'':26s}{'compensated':>16s}{'no attitude':>16s}{'truth':>12s}"]
    for zone, _ in zones:
        lines.append(f"{'|err| mean ' + zone:26s}" + "".join(f"{rows[n][zone][0]:14.3f} m" for n in rows))
    for zone, _ in zones:
        lines.append(f"{'|err| p95 ' + zone:26s}" + "".join(f"{rows[n][zone][1]:14.3f} m" for n in rows))
    lines.append(f"{'coverage (scored)':26s}" + "".join(f"{100 * rows[n]['all r<60'][2]:15.1f}%" for n in rows))
    for key, lab, const in (("pit", "scour pit depth", ts.PIT_DEPTH), ("radius", "berm half-height r", ts.R_ARMOUR - 0.5 * ts.TOE_W),
                            ("thick", "berm thickness", ts.T_ARMOUR)):
        lines.append(f"{lab:26s}" + "".join(f"{rows[n]['berm'][key]:14.2f} m" for n in rows) + f"{tb_[key]:10.2f} m"
                     + f"   (constant {const:.2f} m)")
    lines += ["", "constants (turbine_site.py): PIT_DEPTH 2.0, R_ARMOUR 22.5 with TOE_W 3.0 (toe slope 19.5..22.5, half-height 21.0), "
              "T_ARMOUR 1.2 (stone relief rides on it); 'truth' = the same estimator run on the seabed_height grid",
              "fine mesh = |x|,|z| < 41 m, where the rendered bed is seabed_height at 9 cm; beyond it the bed mesh is a 1-1.5 m band-limited grid"]
    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(OUT, "survey_numbers.txt")
    import shutil
    if os.path.exists(out):                             # keep the earlier table under a version
        shutil.copy(out, versioned("survey_numbers_prev", "txt"))
    tmp = os.path.join(OUT, "_tmp_survey_numbers.txt")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    os.replace(tmp, out)
    print(f"[jobs] wrote {out}")
    survey_figures(D, maps, truth, keep, X, Z)
    return D, maps, truth, keep


def bathy_rgb(h, keep, lo=-32.3, hi=-28.4):
    """Depth ramp: deep navy -> teal -> sand yellow; masked/no data dark grey. Row 0 = -Z (north up is +X? no: image
    x = world X, image y = world Z)."""
    import matplotlib
    cm = matplotlib.colormaps["viridis"]
    u = np.clip((h - lo) / (hi - lo), 0.0, 1.0)
    rgb = (cm(np.nan_to_num(u))[..., :3] * 255).astype(np.uint8)
    # hillshade for the stones and the berm edge
    gy, gx = np.gradient(np.nan_to_num(h, nan=float(np.nanmedian(h)) if np.isfinite(h).any() else -30.0), CELL)
    sh = np.clip(0.75 + 0.9 * (-0.6 * gx - 0.5 * gy) / np.sqrt(1 + gx * gx + gy * gy), 0.45, 1.25)
    rgb = np.clip(rgb * sh[..., None], 0, 255).astype(np.uint8)
    rgb[~keep | ~np.isfinite(h)] = (38, 40, 44)
    return rgb


def survey_figures(D, maps, truth, keep, X, Z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [-R_MAP, R_MAP, R_MAP, -R_MAP]
    h, _ = maps["compensated"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2), dpi=120)
    lo, hi = 28.4, 32.3                                   # depth below MSL, same limits on both panels
    trk = D["origin"]
    for a, grid, title in ((ax[0], h, "Otter X multibeam, motion compensated (0.5 m cells, median)"),
                           (ax[1], truth, "ground truth seabed_height (cell means)")):
        im = a.imshow(np.where(keep, -grid, np.nan), extent=ext, cmap="viridis_r", vmin=lo, vmax=hi, interpolation="nearest")
        a.plot(trk[:, 0], trk[:, 2], color="k", lw=0.7)
        a.set_title(title)
        a.set_xlabel("x (m, upwind)")
        a.set_ylabel("z (m)")
        a.add_patch(plt.Circle((0, 0), ts.R_ARMOUR, fill=False, ls="--", color="w", lw=0.6, alpha=0.6))
        fig.colorbar(im, ax=a, label="depth below MSL (m)", shrink=0.8)
    fig.tight_layout()
    p = versioned("survey_map", "png")
    tmp = os.path.join(OUT, "_tmp_" + os.path.basename(p))
    fig.savefig(tmp)
    os.replace(tmp, p)
    plt.close(fig)
    print(f"[jobs] wrote {p}")
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2), dpi=120)
    for a, name in zip(ax, ("compensated", "no attitude")):
        e = np.where(keep, maps[name][0] - truth, np.nan)
        im = a.imshow(e, extent=ext, cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
        a.plot(D["origin"][:, 0], D["origin"][:, 2], color="k", lw=0.7)
        a.set_title(f"map - truth, {name}")
        a.set_xlabel("x (m, upwind)")
        a.set_ylabel("z (m)")
        fig.colorbar(im, ax=a, label="map shallower (+) / deeper (-) than truth (m)", shrink=0.8)
    fig.tight_layout()
    p = versioned("survey_error", "png")
    tmp = os.path.join(OUT, "_tmp_" + os.path.basename(p))
    fig.savefig(tmp)
    os.replace(tmp, p)
    plt.close(fig)
    print(f"[jobs] wrote {p}")


SEG_FRACS, SEG_T = (0.0, 0.25, 0.5, 0.75), 3.0      # four stations round her circle, 3 s each in real time


def otter_cam(tau_mid):
    """Fixed for a segment (relative to the turbine): 10 m up, 22 m outboard of her mid-segment spot,
    looking in past her at the transition piece, 16 deg down (the horizon just out of frame)."""
    x, z, *_ = tf.otter_on_circle(tf.T_SHOT + tau_mid)
    rn = np.array([x, 0.0, z]) / math.hypot(x, z)
    eye = np.array([x, 0.0, z]) + 22.0 * rn + np.array([0.0, 10.0, 0.0])
    tgt = 0.55 * np.array([x, 0.0, z])
    tgt[1] = eye[1] - math.tan(math.radians(16.0)) * float(np.linalg.norm((tgt - eye)[[0, 2]]))
    return eye, tgt, 28.0


class MapInset:
    """The bathymetry map building along the track (per-cell median, the survey's gridding)."""

    def __init__(self, D, px=2):
        self.P, self.tt = soundings(D, True)
        self.keep = map_mask()[0]
        self.O, self.T = D["origin"], D["t"]
        self.px = px
        n = int(2 * R_MAP / CELL)
        self.size = n * px

    def image(self, upto):
        h, _ = grid_map(self.P, upto, self.tt)
        rgb = bathy_rgb(h, self.keep)
        rgb = np.repeat(np.repeat(rgb, self.px, 0), self.px, 1)
        k = self.T <= upto
        for (x, _, z) in self.O[k][::3]:                  # the track, thin
            i, j = int((x + R_MAP) / CELL * self.px), int((z + R_MAP) / CELL * self.px)
            if 0 <= i < self.size and 0 <= j < self.size:
                rgb[j, i] = (250, 250, 250)
        if k.any():                                        # the boat now
            x, _, z = self.O[k][-1]
            i, j = int((x + R_MAP) / CELL * self.px), int((z + R_MAP) / CELL * self.px)
            rgb[max(j - 6, 0):j + 7, max(i - 6, 0):i + 7] = (255, 120, 20)
        return rgb


def film_otter():
    """Four 3 s segments in real time at stations round the circle, hard cuts between; the map
    shows every sounding up to that moment (north/upwind up, as the figures)."""
    t_all = time.perf_counter()
    D = dict(np.load(NPZ))
    inset = MapInset(D)
    canvas, renderer = make_renderer(W, H)
    S = ts.build_site(renderer)
    camera = tp.PerspectiveCamera(26.0, W / H, 0.3, 20000.0)
    F = tf.build_fleet(renderer, S)
    t0 = ts.T_SHOT - 8.0
    F.t = t0
    tf.apply_camera(renderer, S, F, camera, "fleet_aerial")
    renderer.sim_time = t0
    renderer.render(S.scene, camera)
    T0 = settle(renderer, S, F, camera, t0, t0 + 8.0)
    print(f"[otter] setup {time.perf_counter() - t_all:.0f} s")
    out = versioned("otter_survey", "mp4")
    v = os.path.basename(out)[:-4].split("_")[-1]
    tmp = os.path.join(OUT, "_tmp_" + os.path.basename(out))
    enc = Encoder(tmp, W, H, FPS, crf=18, preset="medium", faststart=True)
    m = int(round(SEG_T * FPS))
    n = m * len(SEG_FRACS)
    picks = set(np.linspace(0, n - 1, 8).round().astype(int))
    sheet, stills = [], {}
    dt = 1.0 / FPS
    S_ = inset.size
    M = 36
    k = 0
    for si, fr in enumerate(SEG_FRACS):
        tau0 = fr * SURVEY_T
        if tau0 > 0.0:                                   # the cut: re-seat her at the station, ride 2 s
            x, z, hd, vx, vz, rate, *_ = tf.otter_on_circle(T0 + tau0 - 2.0)
            F.otter.seat(x, z, hd)
            F.otter.v = np.array([vx, 0.0, vz])
            F.otter.w = np.array([0.0, rate, 0.0])
            settle(renderer, S, F, camera, T0 + tau0 - 2.0, T0 + tau0)
        eye, tgt, fov = otter_cam(tau0 + 0.5 * SEG_T)
        camera.fov = fov
        camera.update_projection_matrix()
        camera.position.set(*map(float, eye))
        camera.look_at(tp.Vector3(*map(float, tgt)))
        renderer.tone_mapping_exposure = 0.8
        img = None
        for j in range(m):
            t = T0 + tau0 + j * dt
            renderer.sim_time = t
            S.update(t, camera, False, dt)
            renderer.render(S.scene, camera)
            rgb = renderer.read_pixels()
            if img is None or j % 3 == 0:
                img = inset.image(tau0 + j * dt)
            x0, y0 = W - M - S_, M
            rgb[y0:y0 + S_, x0:x0 + S_] = img
            rgb = pip_frame(rgb, x0, y0, S_, S_, "SONAR MAP")
            enc.send(rgb)
            if k in picks:
                sheet.append(rgb.copy())
            if j == m // 2:
                stills[k] = rgb.copy()
            F.update(t, dt)
            k += 1
        print(f"[otter] segment {si} at tau {tau0:.0f} s done")
    enc.close()
    os.replace(tmp, out)
    print(f"[otter] wrote {out} in {time.perf_counter() - t_all:.0f} s")
    contact_sheet(sheet, os.path.join(OUT, f"otter_survey_sheet_{v}.png"))
    for i, (k, im) in enumerate(sorted(stills.items())):
        save_png(im, os.path.join(OUT, f"otter_survey_{'abcd'[i]}_{v}.png"))


def main():
    job = cli_arg("--job", "drone", str)
    os.makedirs(OUT, exist_ok=True)
    if job == "drone":
        film_drone()
    elif job == "survey":
        if not os.path.exists(NPZ) or "--rescan" in sys.argv:
            survey_pass()
        survey_numbers()
    elif job == "otter":
        film_otter()
    else:
        sys.exit(f"unknown job {job!r}")


if __name__ == "__main__":
    main()
