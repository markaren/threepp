"""
stewart_balancer.py – Stewart platform ball balancer
LQR control · inverse kinematics · stepper-motor dynamics

A 6-leg hexapod platform balances a ball along a rose-curve
reference path.  Periodic disturbances kick the ball off track;
the LQR controller drives it back.  Two HUD plots show the
phase-portrait (error spiral) and stepper torque vs pull-out.
"""

import math
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import threepp as tp


# ═══════════════════════════════════════════════════════════════
#  Physical constants
# ═══════════════════════════════════════════════════════════════
G           = 9.81                      # m/s²
ALPHA       = (5.0 / 7.0) * G          # rolling-ball accel factor
BALL_R      = 0.012                     # ball radius          [m]
PLATE_HALF  = 0.10                      # plate half-width     [m]
PLAT_H      = 0.16                      # neutral platform height
BASE_R      = 0.12                      # base attachment circle
PLAT_R      = 0.065                     # platform attachment circle
MAX_TILT    = math.radians(10)          # actuator limit

# stepper motor / leadscrew
LEAD        = 0.002                     # screw lead  [m/rev]
USTEPS      = 200 * 16                  # steps × microsteps
STEP_MM     = LEAD / USTEPS             # linear resolution
MAX_VEL     = 600 * LEAD               # max linear speed
PULLOUT     = 0.020                     # pull-out torque at speed [N·m]
SCREW_EFF   = 0.40                      # screw efficiency

# reference – Lissajous 3:4 (never crosses the centre)
REF_A       = 0.050                     # amplitude   [m]
REF_T       = 10.0                      # period      [s]

# LQR weights  (Q_V kept low → fast + slightly oscillatory)
QP, QV, RU  = 120.0, 6.0, 1.0
ACT_TAU     = 0.030                     # actuator lag [s]


# ═══════════════════════════════════════════════════════════════
#  Stewart platform geometry
# ═══════════════════════════════════════════════════════════════
def _hex_pts(r, half_angle):
    """Three close-spaced pairs on a circle of radius *r*."""
    pts = np.zeros((6, 3))
    for i in range(3):
        c = math.radians(i * 120)
        for j, s in enumerate((-1, 1)):
            a = c + s * half_angle
            pts[2 * i + j] = [r * math.cos(a), 0.0, r * math.sin(a)]
    return pts


BASE_PTS = _hex_pts(BASE_R, math.radians(10))
BASE_PTS[:, 1] = 0.016                     # top of base mount
PLAT_PTS = _hex_pts(PLAT_R, math.radians(22))

# rotate platform points 60° for crossed-leg topology
_c60, _s60 = math.cos(math.pi / 3), math.sin(math.pi / 3)
for _i in range(6):
    _x, _z = PLAT_PTS[_i, 0], PLAT_PTS[_i, 2]
    PLAT_PTS[_i, 0] = _c60 * _x - _s60 * _z
    PLAT_PTS[_i, 2] = _s60 * _x + _c60 * _z

# neutral leg lengths
_c0 = np.array([0.0, PLAT_H, 0.0])
NEUTRAL_L = np.array(
    [np.linalg.norm(_c0 + PLAT_PTS[i] - BASE_PTS[i]) for i in range(6)]
)


# ═══════════════════════════════════════════════════════════════
#  Rotation matrix  (Rz · Rx — matches Euler XYZ with ry = 0)
# ═══════════════════════════════════════════════════════════════
def _rot(pitch, roll):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [[cr, -sr * cp,  sr * sp],
         [sr,  cr * cp, -cr * sp],
         [0.0, sp,       cp     ]]
    )


# ═══════════════════════════════════════════════════════════════
#  Inverse kinematics
# ═══════════════════════════════════════════════════════════════
def _ik(pitch, roll, heave=0.0):
    R = _rot(pitch, roll)
    c = np.array([0.0, PLAT_H + heave, 0.0])
    wp = np.zeros((6, 3))
    lens = np.zeros(6)
    for i in range(6):
        wp[i] = c + R @ PLAT_PTS[i]
        lens[i] = np.linalg.norm(wp[i] - BASE_PTS[i])
    return lens, wp


# ═══════════════════════════════════════════════════════════════
#  LQR gain matrix (analytical CARE for decoupled channels)
# ═══════════════════════════════════════════════════════════════
def _lqr():
    """
    Two decoupled double-integrator channels:
        ẍ = −α·roll     (B_x = −α)
        z̈ = +α·pitch    (B_z = +α)
    CARE for each yields |k_p|, |k_v|; signs differ per channel.
    """
    a = ALPHA
    p12 = math.sqrt(QP * RU) / a
    p22 = math.sqrt((2 * p12 + QV) * RU) / a
    kp = a * p12 / RU
    kv = a * p22 / RU
    # X channel (B = −α): K signs [−kp, −kv]
    # Z channel (B = +α): K signs [+kp, +kv]
    return np.array(
        [[-kp, 0.0, -kv, 0.0],          # → roll command
         [0.0, kp,  0.0, kv ]]           # → pitch command
    )


K_LQR = _lqr()


# ═══════════════════════════════════════════════════════════════
#  Stepper-motor actuator
# ═══════════════════════════════════════════════════════════════
class Stepper:
    __slots__ = ("pos", "neutral", "load_tq")

    def __init__(self, neutral_len: float):
        self.pos = neutral_len
        self.neutral = neutral_len
        self.load_tq = 0.0

    def step(self, target: float, load_force: float, dt: float) -> float:
        self.load_tq = abs(load_force) * LEAD / (2 * math.pi * SCREW_EFF)
        ratio = min(1.0, self.load_tq / PULLOUT) if PULLOUT > 0 else 0.0
        max_v = MAX_VEL * (1.0 - 0.6 * ratio)
        err = target - self.pos
        v = float(np.clip(err / max(dt, 1e-6), -max_v, max_v))
        self.pos += v * dt
        self.pos = round(self.pos / STEP_MM) * STEP_MM
        return self.pos


# ═══════════════════════════════════════════════════════════════
#  Reference path  (rose curve)
# ═══════════════════════════════════════════════════════════════
def _ref(t):
    w = 2 * math.pi / REF_T
    return (REF_A * math.sin(3 * w * t),
            REF_A * math.cos(4 * w * t))


def _ref_v(t):
    w = 2 * math.pi / REF_T
    return ( REF_A * 3 * w * math.cos(3 * w * t),
            -REF_A * 4 * w * math.sin(4 * w * t))


# ═══════════════════════════════════════════════════════════════
#  HUD helper
# ═══════════════════════════════════════════════════════════════
def _plot_frame(scene, font, x, y, w, h, label):
    """Border rectangle + label for a HUD plot."""
    pts = [tp.Vector3(x, y, 0), tp.Vector3(x + w, y, 0),
           tp.Vector3(x + w, y + h, 0), tp.Vector3(x, y + h, 0),
           tp.Vector3(x, y, 0)]
    g = tp.BufferGeometry(); g.set_from_points(pts)
    m = tp.LineBasicMaterial(); m.color = tp.Color(0x2a3540)
    scene.add(tp.Line(g, m))
    lb = tp.Text2D(font, label, size=11)
    lb.set_color(0x778899)
    lb.position.set(x + 4, y + h + 6, 0)
    scene.add(lb)


# ═══════════════════════════════════════════════════════════════
#  Dynamic-line helper (pre-allocated GPU buffer)
# ═══════════════════════════════════════════════════════════════
def _make_trail(scene, max_pts, color_hex):
    buf = np.zeros((max_pts, 3), np.float32)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", buf.copy())
    geo.set_draw_range(0, 0)
    mat = tp.LineBasicMaterial(); mat.color = tp.Color(color_hex)
    line = tp.Line(geo, mat)
    line.frustum_culled = False
    scene.add(line)
    return geo, buf


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    # ── canvas / renderer ──────────────────────────────────
    canvas = tp.Canvas(
        "Stewart Platform Ball Balancer", 1280, 720,
        antialiasing=4, resizable=True,
    )
    renderer = tp.GLRenderer(canvas)
    renderer.set_clear_color(tp.Color(0x080810), 1.0)
    renderer.shadow_map_enabled = True

    scene = tp.Scene()
    cam   = tp.PerspectiveCamera(42, canvas.aspect(), 0.01, 10.0)
    cam.position.set(0.30, 0.28, 0.30)
    ctrl  = tp.OrbitControls(cam, canvas)
    ctrl.target.set(0, PLAT_H * 0.45, 0)
    ctrl.enable_damping = True
    ctrl.damping_factor = 0.08

    # ── lights ─────────────────────────────────────────────
    scene.add(tp.AmbientLight(0x303050, 0.5))
    sun = tp.DirectionalLight(0xfff0e0, 1.0)
    sun.position.set(0.4, 0.8, 0.3)
    sun.cast_shadow = True
    scene.add(sun)

    # ── ground ─────────────────────────────────────────────
    gnd = tp.Mesh(
        tp.PlaneGeometry(0.8, 0.8),
        tp.MeshStandardMaterial(),
    )
    gnd.material.color = tp.Color(0x141418)
    gnd.material.roughness = 0.95
    gnd.rotation.x = -math.pi / 2
    gnd.position.y = -0.002
    gnd.receive_shadow = True
    scene.add(gnd)

    # ── base mounts ────────────────────────────────────────
    bx_geo = tp.BoxGeometry(0.024, 0.016, 0.024)
    bx_mat = tp.MeshStandardMaterial()
    bx_mat.color = tp.Color(0x28282f)
    bx_mat.roughness = 0.7
    bx_mat.metalness = 0.2
    for i in range(6):
        m = tp.Mesh(bx_geo, bx_mat)
        m.position.set(float(BASE_PTS[i, 0]), 0.008, float(BASE_PTS[i, 2]))
        m.cast_shadow = True
        scene.add(m)

    # ── joint spheres (base + platform) ────────────────────
    j_geo = tp.SphereGeometry(0.005, 10, 8)
    j_mat = tp.MeshStandardMaterial()
    j_mat.color = tp.Color(0x556666)
    j_mat.metalness = 0.6
    j_mat.roughness = 0.3
    base_j = []
    for i in range(6):
        j = tp.Mesh(j_geo, j_mat)
        j.position.set(float(BASE_PTS[i, 0]),
                       float(BASE_PTS[i, 1]),
                       float(BASE_PTS[i, 2]))
        scene.add(j); base_j.append(j)
    plat_j = []
    for _ in range(6):
        j = tp.Mesh(j_geo, j_mat)
        scene.add(j); plat_j.append(j)

    # ── legs (unit-height cylinders, scaled per frame) ─────
    l_mat = tp.MeshStandardMaterial()
    l_mat.color = tp.Color(0xbb44ff)
    l_mat.roughness = 0.30
    l_mat.metalness = 0.50
    l_mat.emissive = tp.Color(0x4400cc)
    l_mat.emissive_intensity = 0.15
    legs = []
    for _ in range(6):
        leg = tp.Mesh(tp.CylinderGeometry(0.004, 0.004, 1.0, 8), l_mat)
        leg.cast_shadow = True
        scene.add(leg); legs.append(leg)

    # ── top plate ──────────────────────────────────────────
    plate = tp.Group()
    face = tp.Mesh(
        tp.BoxGeometry(PLATE_HALF * 2, 0.003, PLATE_HALF * 2),
        tp.MeshStandardMaterial(),
    )
    face.material.color = tp.Color(0x0e1822)
    face.material.roughness = 0.20
    face.material.metalness = 0.50
    face.material.transparent = True
    face.material.opacity = 0.82
    face.receive_shadow = True
    plate.add(face)

    # cyan edges
    hs = PLATE_HALF
    e_geo = tp.BufferGeometry()
    e_geo.set_from_points([
        tp.Vector3(-hs, 0.002, -hs), tp.Vector3( hs, 0.002, -hs),
        tp.Vector3( hs, 0.002,  hs), tp.Vector3(-hs, 0.002,  hs),
        tp.Vector3(-hs, 0.002, -hs),
    ])
    e_mat = tp.LineBasicMaterial(); e_mat.color = tp.Color(0x00e5ff)
    plate.add(tp.Line(e_geo, e_mat))

    # dashed reference path (LineSegments pairs = visual dashes)
    N_REF = 400
    dash = []
    for j in range(0, N_REF - 2, 3):
        t0 = j       / N_REF * REF_T
        t1 = (j + 1) / N_REF * REF_T
        x0, z0 = _ref(t0); x1, z1 = _ref(t1)
        dash.append(tp.Vector3(x0, 0.003, z0))
        dash.append(tp.Vector3(x1, 0.003, z1))
    rg = tp.BufferGeometry(); rg.set_from_points(dash)
    rm = tp.LineBasicMaterial(); rm.color = tp.Color(0x006688)
    plate.add(tp.LineSegments(rg, rm))

    # reference-position marker (small green sphere)
    ref_dot = tp.Mesh(
        tp.SphereGeometry(0.003, 8, 6),
        tp.MeshBasicMaterial(),
    )
    ref_dot.material.color = tp.Color(0x00ff88)
    plate.add(ref_dot)

    plate.position.set(0, PLAT_H, 0)
    scene.add(plate)

    # ── ball ───────────────────────────────────────────────
    ball = tp.Mesh(
        tp.SphereGeometry(BALL_R, 24, 16),
        tp.MeshStandardMaterial(),
    )
    ball.material.color = tp.Color(0xff8800)
    ball.material.roughness = 0.25
    ball.material.metalness = 0.35
    ball.material.emissive = tp.Color(0xff4400)
    ball.material.emissive_intensity = 0.20
    ball.cast_shadow = True
    scene.add(ball)

    # ── ball trail (world space) ───────────────────────────
    TRAIL = 120
    tr_geo, tr_buf = _make_trail(scene, TRAIL, 0xcc5500)

    # ══════════════════════════════════════════════════════
    #  HUD (orthographic overlay)
    # ══════════════════════════════════════════════════════
    hud = tp.Scene()
    w0, h0 = canvas.size()
    hud_cam = tp.OrthographicCamera(0, w0, h0, 0, -10, 10)
    fnt = tp.FontLoader().default_font()

    stat = tp.Text2D(fnt, "", size=16)
    stat.set_color(0xccddee)
    stat.position.set(24, 28, 0)
    hud.add(stat)

    title = tp.Text2D(fnt, "LQR + IK + stepper dynamics", size=12)
    title.set_color(0x556677)
    title.position.set(24, h0 - 20, 0)
    hud.add(title)

    # ── phase portrait ─────────────────────────────────────
    PX, PY, PW, PH = 30, 70, 200, 150
    _plot_frame(hud, fnt, PX, PY, PW, PH,
                "phase portrait (stable spiral)")

    # crosshair axes
    ax_g = tp.BufferGeometry()
    ax_g.set_from_points([
        tp.Vector3(PX, PY + PH // 2, 0),
        tp.Vector3(PX + PW, PY + PH // 2, 0),
        tp.Vector3(PX + PW // 2, PY, 0),
        tp.Vector3(PX + PW // 2, PY + PH, 0),
    ])
    ax_m = tp.LineBasicMaterial(); ax_m.color = tp.Color(0x1a2530)
    hud.add(tp.LineSegments(ax_g, ax_m))

    # axis labels
    pp_x_lbl = tp.Text2D(fnt, "pos err", size=9)
    pp_x_lbl.set_color(0x556666)
    pp_x_lbl.position.set(PX + PW - 40, PY + PH // 2 + 4, 0)
    hud.add(pp_x_lbl)
    pp_y_lbl = tp.Text2D(fnt, "vel", size=9)
    pp_y_lbl.set_color(0x556666)
    pp_y_lbl.position.set(PX + PW // 2 + 4, PY + PH - 12, 0)
    hud.add(pp_y_lbl)

    PP = 200
    pp_geo_x, pp_buf_x = _make_trail(hud, PP, 0x22aadd)
    pp_geo_z, pp_buf_z = _make_trail(hud, PP, 0xddaa22)

    # ── stepper torque plot (in a movable group) ──────────
    SY, SW, SH = PY, PW, PH
    st_grp = tp.Group()
    st_grp.position.set(w0 - 240, 0, 0)
    hud.add(st_grp)

    _plot_frame(st_grp, fnt, 0, SY, SW, SH,
                "stepper lead vs pull-out")

    # pull-out limit line (red)
    po_y = SY + SH * 0.85
    po_g = tp.BufferGeometry()
    po_g.set_from_points([tp.Vector3(0, po_y, 0),
                          tp.Vector3(SW, po_y, 0)])
    po_m = tp.LineBasicMaterial(); po_m.color = tp.Color(0xdd3333)
    st_grp.add(tp.Line(po_g, po_m))
    po_lbl = tp.Text2D(fnt, "pull-out", size=9)
    po_lbl.set_color(0xdd3333)
    po_lbl.position.set(SW - 48, po_y + 5, 0)
    st_grp.add(po_lbl)

    ST = 200
    st_geo, st_buf = _make_trail(st_grp, ST, 0xddcc22)

    # ══════════════════════════════════════════════════════
    #  Simulation state
    # ══════════════════════════════════════════════════════
    bx = bz = 0.015                       # initial ball offset [m]
    bvx = bvz = 0.0                       # ball velocity
    pit = rol = 0.0                       # current platform tilt
    steppers = [Stepper(float(NEUTRAL_L[i])) for i in range(6)]

    trail_d = deque(maxlen=TRAIL)
    pp_dx   = deque(maxlen=PP)
    pp_dz   = deque(maxlen=PP)
    st_d    = deque(maxlen=ST)
    clock   = tp.Clock()
    sim_t   = 0.0
    dist_t  = 0.0
    ramp    = 0.0

    # ── resize handler ─────────────────────────────────────
    def on_resize(w2, h2):
        nonlocal hud_cam
        cam.aspect = w2 / max(h2, 1)
        cam.update_projection_matrix()
        renderer.set_size(w2, h2)
        hud_cam = tp.OrthographicCamera(0, w2, h2, 0, -10, 10)
        title.position.set(24, h2 - 20, 0)
        st_grp.position.set(w2 - 240, 0, 0)

    canvas.on_window_resize(on_resize)

    # ══════════════════════════════════════════════════════
    #  Animation loop
    # ══════════════════════════════════════════════════════
    UP = np.array([0.0, 1.0, 0.0])

    def animate():
        nonlocal bx, bz, bvx, bvz, pit, rol, sim_t, dist_t, ramp

        dt_frame = min(clock.get_delta(), 0.04)
        SUBSTEPS = 5
        ds = dt_frame / SUBSTEPS

        # ── physics substeps ───────────────────────────────
        for _ in range(SUBSTEPS):
            sim_t += ds
            ramp = min(1.0, sim_t / 3.0)

            # reference (ramped in over 3 s)
            rx, rz = _ref(sim_t)
            rx *= ramp; rz *= ramp
            rvx, rvz = _ref_v(sim_t)
            rvx *= ramp; rvz *= ramp

            # periodic disturbance (~every 3.5 s)
            dist_t += ds
            if dist_t > 3.5 and sim_t > 2.0:
                dist_t -= 3.5
                bvx += 0.025 * math.sin(sim_t * 1.7)
                bvz += 0.020 * math.cos(sim_t * 2.3)

            # LQR  (u = −K·e)
            err = np.array([bx - rx, bz - rz, bvx - rvx, bvz - rvz])
            u = -(K_LQR @ err)
            r_cmd = float(np.clip(u[0], -MAX_TILT, MAX_TILT))
            p_cmd = float(np.clip(u[1], -MAX_TILT, MAX_TILT))

            # first-order actuator lag
            f = min(1.0, ds / ACT_TAU)
            rol += (r_cmd - rol) * f
            pit += (p_cmd - pit) * f

            # ball-on-plate dynamics (local frame)
            # g_local = R^T·(0,-g,0) → x: -g·sin(roll),
            #                           z:  g·cos(roll)·sin(pitch)
            ax = -ALPHA * math.sin(rol)
            az =  ALPHA * math.cos(rol) * math.sin(pit)
            bvx += ax * ds
            bvz += az * ds
            bvx *= 1.0 - 0.3 * ds          # rolling friction
            bvz *= 1.0 - 0.3 * ds
            bx  += bvx * ds
            bz  += bvz * ds

            # elastic edge bounce
            lim = PLATE_HALF - BALL_R
            if abs(bx) > lim:
                bx = math.copysign(lim, bx); bvx *= -0.4
            if abs(bz) > lim:
                bz = math.copysign(lim, bz); bvz *= -0.4

            # IK  +  stepper update (dynamic load from leg velocity)
            lens, _ = _ik(pit, rol)
            for i in range(6):
                leg_v = abs(lens[i] - steppers[i].pos) / max(ds, 1e-6)
                leg_f = 2.0 + leg_v * 80.0          # static + dynamic [N]
                steppers[i].step(lens[i], leg_f, ds)

        # ── update visuals ─────────────────────────────────
        R = _rot(pit, rol)

        # platform
        plate.rotation.x = pit
        plate.rotation.z = rol

        # legs + platform joints
        _, wp = _ik(pit, rol)
        plate_down = R @ np.array([0.0, -0.007, 0.0])  # below plate
        for i in range(6):
            b = BASE_PTS[i]
            jp = wp[i] + plate_down             # joint sits under the plate
            mid = (b + jp) * 0.5
            v = jp - b
            ln = np.linalg.norm(v)
            d = v / ln

            legs[i].position.set(float(mid[0]), float(mid[1]), float(mid[2]))
            legs[i].scale.set(1.0, ln, 1.0)
            dot = float(np.dot(UP, d))
            if abs(dot) < 0.9999:
                ax_v = np.cross(UP, d)
                ax_v /= np.linalg.norm(ax_v)
                ang = math.acos(np.clip(dot, -1.0, 1.0))
                legs[i].quaternion.set_from_axis_angle(
                    tp.Vector3(float(ax_v[0]), float(ax_v[1]), float(ax_v[2])),
                    ang,
                )

            plat_j[i].position.set(float(jp[0]), float(jp[1]), float(jp[2]))

        # ball (plate-local → world)
        bl = np.array([bx, BALL_R + 0.002, bz])
        bw = np.array([0.0, PLAT_H, 0.0]) + R @ bl
        ball.position.set(float(bw[0]), float(bw[1]), float(bw[2]))

        # reference marker (plate-local)
        rx_now = _ref(sim_t)[0] * ramp
        rz_now = _ref(sim_t)[1] * ramp
        ref_dot.position.set(float(rx_now), 0.004, float(rz_now))

        # ball trail
        trail_d.append((float(bw[0]), float(bw[1]), float(bw[2])))
        n = len(trail_d)
        b = np.zeros((TRAIL, 3), np.float32)
        b[:n] = np.array(trail_d, np.float32)
        tr_geo.update_attribute("position", b)
        tr_geo.set_draw_range(0, n)

        # ── HUD ────────────────────────────────────────────
        track = math.sqrt((bx - rx_now) ** 2 + (bz - rz_now) ** 2) * 1000
        max_tq = max(s.load_tq for s in steppers)
        tq_pct = max_tq / PULLOUT * 100
        stat.set_text(
            f"pitch {math.degrees(pit):+.1f}  "
            f"roll {math.degrees(rol):+.1f}  "
            f"track {track:.1f} mm  "
            f"holding torque {tq_pct:.0f}%"
        )

        # phase portrait (error-x in cyan, error-z in gold)
        cx = PX + PW / 2;   cy = PY + PH / 2
        sp_p = PW / 2 / 15  # ±15 mm
        sp_v = PH / 2 / 12  # ±12 cm/s

        ex = (bx - rx_now) * 1000;  evx = bvx * 100
        ez = (bz - rz_now) * 1000;  evz = bvz * 100
        pp_dx.append((float(np.clip(cx + ex * sp_p, PX, PX + PW)),
                      float(np.clip(cy + evx * sp_v, PY, PY + PH)), 0.0))
        pp_dz.append((float(np.clip(cx + ez * sp_p, PX, PX + PW)),
                      float(np.clip(cy + evz * sp_v, PY, PY + PH)), 0.0))

        n_pp = len(pp_dx)
        b1 = np.zeros((PP, 3), np.float32)
        b1[:n_pp] = np.array(pp_dx, np.float32)
        pp_geo_x.update_attribute("position", b1)
        pp_geo_x.set_draw_range(0, n_pp)

        n_pz = len(pp_dz)
        b2 = np.zeros((PP, 3), np.float32)
        b2[:n_pz] = np.array(pp_dz, np.float32)
        pp_geo_z.update_attribute("position", b2)
        pp_geo_z.set_draw_range(0, n_pz)

        # stepper torque time series (local to st_grp)
        st_d.append(tq_pct / 100.0)
        n_st = len(st_d)
        b3 = np.zeros((ST, 3), np.float32)
        for j_idx, v_val in enumerate(st_d):
            b3[j_idx] = (j_idx / ST * SW,
                         SY + v_val * SH * 0.85,
                         0.0)
        st_geo.update_attribute("position", b3)
        st_geo.set_draw_range(0, n_st)

        # ── render ─────────────────────────────────────────
        ctrl.update()
        renderer.auto_clear = True
        renderer.render(scene, cam)
        renderer.auto_clear = False
        renderer.render(hud, hud_cam)

    canvas.animate(animate)


if __name__ == "__main__":
    main()
