"""A sea drone on the FFT ocean, floating on its own sections: the Mariner or the Otter X.

The boat is <boat>.glb, the model build_<boat>_blender.py makes from
<boat>_spec.json, and it floats on <boat>_hydro.json: the same generator's
half-section buoyancy tables (Bonjean curves). Every frame the ocean's own
height field is sampled under every station on both sides, each half-section
looks up its immersed area and centroid at that local water level, and the sum
of those strip forces and their moments heaves, pitches and rolls a 6-DOF rigid
body. Nothing pulls the hull toward the surface: a crest under the bow lifts the
bow because the bow's sections got deeper. In flat water it settles where the
generator's hydrostatic solver put it; --calm prints that comparison.

--boat mariner (the default): the 6 m jet boat, 0.465 m draft and 1.3 deg by
the stern at the departure load. Propulsion is the Hamilton-type jet: thrust
along the steerable nozzle, a reverse bucket that turns it round, and the bow
thruster for slow work. The hull's resistance, the planing hump and the planing
lift are a few lumped terms tuned so full throttle makes the brochure's 24 kn.

--boat otterx: the 4.6 m electric catamaran on two rim-drive azimuth pods, which
steer together; differential thrust turns her on the spot and the pods run
astern to stop her. A displacement catamaran's resistance, with a mild
wave-making hump, is tuned so full throttle makes 8 kn, an assumption until the
vendor publishes a figure.

Both: the hydrostatics are the tables, the manoeuvring is a sketch.

    python usv_ocean.py                          # drive the Mariner
    python usv_ocean.py --boat otterx            # drive the Otter X
    python usv_ocean.py --calm                   # flat water: settle and compare
    python usv_ocean.py --shot 20 --out usv.png  # headless still after the scripted run
    python usv_ocean.py --record 36 --out usv.mp4

Options: --boat mariner|otterx, --wind 7 (m/s), --fetch 30000 (m, 0 = open
ocean), --load <condition> (the spec's: departure|lightship|full_load for the
Mariner, survey|lightship|full_load for the Otter X), --size 1280x720, --cam
chase|bow|abeam (headless framing), --telemetry, --flat (no waves, scripted
run), --drop (start displaced; with --calm she must come back to the solver's
pose), --diag (roll-moment balance every 0.1 s).

Keys: W/S throttle, A/D steer, Space all stop, C camera (chase / mast camera),
X hull displacement on/off. The Mariner: Q/E bow thruster, R reverse bucket
(hold). The Otter X: Q/E turn on the spot, S past zero runs the pods astern.
Needs a Vulkan build and the boat's .glb (build it once, see build_<boat>_blender.py).
"""
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(_HERE))                     # examples/ (demo_common)

import threepp as tp
from demo_common import Encoder, cli_arg, parse_size, resize_handler, write_radiance_hdr

if not tp.HAS_VULKAN:
    print("The FFT ocean needs a Vulkan build of threepp (-DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)

BOAT = cli_arg("--boat", "mariner", str)
if BOAT not in ("mariner", "otterx"):
    print("--boat is mariner or otterx")
    sys.exit(1)
GLB = os.path.join(_HERE, f"{BOAT}.glb")
HYDRO = os.path.join(_HERE, f"{BOAT}_hydro.json")
SPEC = os.path.join(_HERE, f"{BOAT}_spec.json")
if not (os.path.exists(GLB) and os.path.exists(HYDRO)):
    print(f"{BOAT}.glb / {BOAT}_hydro.json are generated. Build them once, from this folder:\n"
          "  blender --background "
          f"--factory-startup --python build_{BOAT}_blender.py -- --spec {BOAT}_spec.json --out {BOAT}.glb")
    sys.exit(1)
with open(SPEC, encoding="utf-8") as fh:
    spec = json.load(fh)
with open(HYDRO, encoding="utf-8") as fh:
    hydro = json.load(fh)

SHOT_T = cli_arg("--shot", 0.0, float) if "--shot" in sys.argv else 0.0
RECORD_T = cli_arg("--record", 0.0, float) if "--record" in sys.argv else 0.0
HEADLESS = SHOT_T > 0.0 or RECORD_T > 0.0
CALM = "--calm" in sys.argv                  # flat water and no helm: the static check
FLAT = CALM or "--flat" in sys.argv           # flat water, scripted run as usual
DROP = "--drop" in sys.argv
DIAG = "--diag" in sys.argv
TELEMETRY = "--telemetry" in sys.argv or HEADLESS or CALM
WIND = cli_arg("--wind", 7.0, float)
FETCH = cli_arg("--fetch", 30000.0, float)
LOAD = cli_arg("--load", spec["mass"]["design_condition"], str)
W, H = parse_size(cli_arg("--size", "1280x720", str))
FILM_CAM = cli_arg("--cam", "chase", str)       # headless framing: chase | bow | abeam


# --------------------------------------------------------------------------- #
#  The boat as the spec and the generator describe it.
#
#  Vessel frame (the spec's): X forward, Y up, Z starboard, origin on the
#  baseline amidships. The rigid body tracks the CoG in world space and a
#  rotation R (vessel -> world); positive yaw turns to port, positive pitch
#  lifts the bow, positive roll puts the starboard side down.
# --------------------------------------------------------------------------- #
RHO, G = hydro["rho"], hydro["g"]


def mass_condition(cond):
    items = spec["mass"]["budget"] + spec["mass"]["conditions"][cond]["extra"]
    m = sum(b["mass"] for b in items)
    return m, np.array([sum(b["mass"] * b["centroid"][i] for b in items) / m for i in range(3)])


MASS, COG = mass_condition(LOAD)
DESIGN = hydro["conditions"][LOAD]
GYR = spec["mass"]["gyradius"]
# Added mass and inertia as fractions (surge, heave, sway) and (roll, yaw, pitch):
# the usual rough figures for a hull this shape, not a strip-theory result.
M_EFF = MASS * (1.0 + np.array([0.05, 0.80, 0.60]))
I_EFF = MASS * np.array([GYR["roll"], GYR["yaw"], GYR["pitch"]]) ** 2 * (1.0 + np.array([0.25, 0.30, 0.60]))

# Bonjean strips: every station, both sides. The generator tabulated the
# starboard half; port is its mirror.
BJ = hydro["bonjean"]
XS = np.array(BJ["station_x"])
NS = len(XS)
DX = BJ["station_dx"]
HS = np.array(BJ["heights"])
H0, DH, NH = HS[0], HS[1] - HS[0], len(HS)
TAB_A = np.vstack([BJ["area_half"], BJ["area_half"]])
TAB_Z = np.vstack([BJ["zc_half"], BJ["zc_half"]])
TAB_Y = np.vstack([BJ["yc_half"], BJ["yc_half"]])
SX = np.concatenate([XS, XS])
SIDE = np.concatenate([np.ones(NS), -np.ones(NS)])
ROW = np.arange(2 * NS)
BUOY_MASK = 0b011                  # swell + mid band; cascade 2 is chop the hull ignores
SUBSTEPS_HZ = 240.0

# Damping from the design hydrostatics: a fraction of critical on each axis.
K_HEAVE = RHO * G * DESIGN["waterplane_area"]
K_ROLL = RHO * G * DESIGN["displacement_m3"] * DESIGN["gm_t"]
K_PITCH = RHO * G * DESIGN["displacement_m3"] * DESIGN["gm_l"]
C_HEAVE = 2.0 * 0.35 * math.sqrt(K_HEAVE * M_EFF[1])
C_ROLL = 2.0 * 0.15 * math.sqrt(K_ROLL * I_EFF[0])
C_PITCH = 2.0 * 0.45 * math.sqrt(K_PITCH * I_EFF[2])


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def quat_of(R):
    """(x, y, z, w) of a rotation matrix."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def attitude(R):
    """Heading, pitch (bow up +) and roll (starboard down +), radians."""
    return (math.atan2(-R[2, 0], R[0, 0]), math.asin(max(-1.0, min(1.0, R[1, 0]))),
            math.atan2(-R[1, 2], R[1, 1]))


# --------------------------------------------------------------------------- #
#  The boat's own half: propulsion and the manoeuvring terms, the helm, the
#  foam she throws, the scripted run. Both boats define the same names; the
#  buoyancy, the rigid body and the sea are shared.
#
#  boat_loads() returns the vessel-frame force and the moment about the CoG
#  from everything but buoyancy and gravity, damping included.
# --------------------------------------------------------------------------- #
def chase(key, target, rate, dt):
    """Slew an actuator toward its command at a realistic rate."""
    st[key] += max(-rate * dt, min(rate * dt, target - st[key]))


if BOAT == "mariner":
    TITLE = "Mariner USV"
    PROBE_Z = 0.55                     # where each half's water level is read
    THROTTLE_MIN = 0.0
    HELM = {"throttle": 0.0, "throttle_cmd": 0.0, "steer": 0.0, "steer_cmd": 0.0,
            "bucket": 0.0, "bucket_cmd": 0.0, "thruster": 0.0}
    HELP = ("W/S throttle  A/D steer  Q/E bow thruster", "R reverse bucket  Space stop  C camera  X hull")

    # Propulsion (spec) and the lumped hull terms, tuned to 24 kn at full throttle.
    JET = spec["propulsion"]["waterjet"]
    T_BOLLARD = JET["bollard_thrust_n"]
    U_TOP = spec["propulsion"]["top_speed_mps"]
    JET_AT = np.array(JET["thrust_point"])
    STEER_MAX = math.radians(JET["steering"]["limit_deg"])
    BUCKET_MAX = math.radians(JET["reverse_bucket"]["limit_deg"][1])
    BT = spec["propulsion"]["bow_thruster"]
    BT_AT = np.array(BT["position"])

    def jet_thrust_max(u):
        """A jet loses thrust with speed: bollard at rest, ~55 % of it at the top."""
        return T_BOLLARD * (1.0 - 0.45 * min(max(u / U_TOP, 0.0), 1.3))

    HUMP_U, HUMP_W, HUMP_R = 4.2, 1.8, 1600.0           # the planing hump
    TRIM_PLANE = math.radians(3.5)                      # running trim on the plane
    R_LIN = 80.0
    # Full throttle balances resistance plus the weight's pull along a hull trimmed
    # TRIM_PLANE at U_TOP, so she makes the brochure's 24 kn.
    R_QUAD = (jet_thrust_max(U_TOP) - MASS * G * math.sin(TRIM_PLANE) - R_LIN * U_TOP) / U_TOP ** 2

    def resistance(u):
        au = abs(u)
        return math.copysign(R_LIN * au + R_QUAD * au * au
                             + HUMP_R * math.exp(-((au - HUMP_U) / HUMP_W) ** 2), u)

    # Planing lift grows with speed AND with trim, and its centre of pressure slides
    # aft as the bow comes up; that is what makes a planing hull stable in pitch.
    # A lift fixed ahead of the CoG with a constant hump moment has no such brake:
    # the first version stood her on her transom at 30 deg and rolled her over.
    LIFT_MAX = 0.45                    # x weight, on the plane at LIFT_TRIM
    LIFT_TRIM = 5.0                    # deg of (trim + 1.5) that gives LIFT_MAX
    HUMP_M = 15000.0                   # bow-up moment at the hump (N m), gone by 8 deg of trim
    K_BANK = 1200.0                    # deep-V banks into a turn
    # Up on the plane she rides the narrow bottom of her V, where the waterline is
    # half as wide and the hydrostatic righting moment (~ breadth cubed) a quarter
    # of what it is at rest: measured in the first run, 2 kN m at 20 deg against
    # 8.8. A planing hull's roll stiffness comes from the planing surface instead:
    # the lift's centre moves toward the side that goes down. LIFT_ROLL_ARM (m per
    # rad of roll) is sized so that dynamic stiffness matches the static one.
    LIFT_ROLL_ARM = K_ROLL / (MASS * G * LIFT_MAX)
    KV1, KV2 = 1500.0, 3000.0          # sway damping
    KR1, KR2 = 575.0, 32000.0          # yaw damping
    SWAY_Y = 0.35                      # height the lateral force acts at

    EXCL_HALF_LENGTH, EXCL_HALF_BEAM = 2.95, 0.85
    X_CENTRE = -0.1                    # waterplane centre, vessel x
    LOWEST = spec["principal"]["lowest_point_y"]
    _ap = spec["appendages"]
    # the ends of the two flats that reach the lowest point: keel shoe and gondola
    DEEP_PTS = np.array([[x, LOWEST, 0.0] for x in (_ap["keel_shoe"]["x_aft"], 1.66, _ap["gondola"]["x_fwd"],
                                                   _ap["gondola"]["x_flat_aft"])])

    def boat_loads(u, heave, sway, w, wet, upright, trim_deg, roll, area):
        """Hull resistance, planing lift, the jet, the bow thruster, damping."""
        Fb, Mb = np.zeros(3), np.zeros(3)

        def load(force, at):
            nonlocal Fb, Mb
            Fb = Fb + force
            Mb = Mb + np.cross(np.asarray(at) - COG, force)

        aft = (SX > -2.2) & (SX < -1.4)
        jet_wet = min(max(float(area[aft].sum()) / 0.25, 0.0), 1.0) * upright
        load(np.array([-resistance(u) * wet, -C_HEAVE * heave, 0.0]), COG)
        load(np.array([0.0, 0.0, -(KV1 * (abs(u) + 1.0) * sway + KV2 * sway * abs(sway)) * wet]),
             (COG[0], SWAY_Y, COG[2]))
        lift = (MASS * G * LIFT_MAX * smoothstep(2.5, 9.0, u) * wet * upright
                * min(max((trim_deg + 1.5) / LIFT_TRIM, 0.0), 1.4))
        x_cp = COG[0] + min(max(0.8 - 0.22 * (trim_deg - 2.0), -1.2), 1.0)
        z_cp = min(max(LIFT_ROLL_ARM * roll, -0.8), 0.8)
        load(np.array([0.0, lift, 0.0]), (x_cp, COG[1], z_cp))
        t_ax = st["throttle"] * jet_thrust_max(u) * (1.0 - 1.6 * st["bucket"]) * jet_wet
        d = st["steer"]
        load(t_ax * np.array([math.cos(d), 0.0, -math.sin(d)]), JET_AT)
        load(np.array([0.0, 0.0, BT["thrust_n"] * st["thruster"] * max(0.0, 1.0 - abs(u) / 3.0) * wet]), BT_AT)
        p_, r_, q_ = w                                  # roll, yaw, pitch rates (vessel X, Y, Z)
        Mb = Mb + np.array([-C_ROLL * p_ - K_BANK * u * r_ * wet,
                            -(KR1 * (abs(u) + 1.0) * r_ + KR2 * r_ * abs(r_)) * max(wet, 0.2),
                            -C_PITCH * q_ + HUMP_M * math.exp(-((u - HUMP_U) / 2.0) ** 2) * wet
                            * min(max(1.0 - trim_deg / 8.0, 0.0), 1.0)])
        return Fb, Mb

    def helm_tick(dt):
        chase("throttle", st["throttle_cmd"], 0.8, dt)
        chase("steer", st["steer_cmd"], STEER_MAX / 0.7, dt)
        chase("bucket", st["bucket_cmd"], 1.0 / 0.8, dt)

    def helm_keys():
        st["thruster"] = (1.0 if canvas.is_key_down("E") else 0.0) - (1.0 if canvas.is_key_down("Q") else 0.0)
        st["bucket_cmd"] = 1.0 if canvas.is_key_down("R") else 0.0

    def helm_status():
        return f"thr {st['throttle']:.2f}  steer {math.degrees(st['steer']):+5.1f}  bucket {st['bucket']:.2f}"

    def bind_nodes(root):
        return {"steer": root.get_object_by_name("jet_steering"), "bucket": root.get_object_by_name("jet_reverse_bucket")}

    def pose_nodes():
        NODES["steer"].rotation.y = st["steer"]
        NODES["bucket"].rotation.z = st["bucket"] * BUCKET_MAX

    def boat_foam(foam, u):
        # Splats are gaussians whose visible halo reaches ~1.9 radii, so each sits far
        # enough aft that none of it lands ahead of the stem, and at intensities that
        # leave lace at the edges instead of a solid sheet.
        spd = min(abs(u) / U_TOP, 1.0)
        for zs in (-0.75, 0.75):
            foam(2.2, zs, 0.9, 0.10 + 0.45 * spd)                            # bow wave, off the shoulders
        for zs in (-1.15, 1.15):
            foam(0.2, zs, 1.1, 0.45 * spd)                                   # spray off the collar
        foam(-3.4, 0.0, 1.0 + 0.6 * spd, 0.55 * st["throttle"] + 0.3 * st["bucket"] * st["throttle"])  # jet wash
        if abs(st["thruster"]) > 0.05:
            for zs in (-0.6, 0.6):
                foam(BT_AT[0], zs, 0.7, 0.6 * abs(st["thruster"]))

    def autopilot(t):
        """Idle on the swell, run up onto the plane, a hard turn to port, ease off, brake on the
        reverse bucket, then walk the bow round on the thruster."""
        thr = 0.0 if t < 3.0 else (1.0 if t < 22.0 else (0.35 if t < 27.0 else (0.6 if t < 31.5 else 0.0)))
        steer = 0.0
        if 13.0 <= t < 19.0:
            steer = -STEER_MAX
        elif 23.0 <= t < 27.0:
            steer = 0.6 * STEER_MAX
        st["throttle_cmd"] = thr
        st["steer_cmd"] = steer
        st["bucket_cmd"] = 1.0 if 27.0 <= t < 31.5 else 0.0   # jet turned round: braking
        st["thruster"] = 1.0 if t >= 31.5 else 0.0
        if CALM:
            st["throttle_cmd"] = st["steer_cmd"] = st["bucket_cmd"] = st["thruster"] = 0.0

else:
    TITLE = "Otter X USV"
    PONTOON_Z = spec["pontoons"]["keel_z"]
    PROBE_Z = 0.87                     # where each half's water level is read: mid-pontoon
    THROTTLE_MIN = -0.5                # S past zero runs the pods astern
    HELM = {"throttle": 0.0, "throttle_cmd": 0.0, "steer": 0.0, "steer_cmd": 0.0,
            "diff": 0.0, "diff_cmd": 0.0, "spin_port": 0.0, "spin_stbd": 0.0}
    HELP = ("W/S throttle (S past zero: astern)  A/D steer", "Q/E turn on the spot  Space stop  C camera  X hull")

    # Two azimuth pods under the keel lines; both steer to the helm's angle. Q/E
    # adds DIFF_GAIN of thrust to one pod and takes it off the other.
    TH = spec["propulsion"]["thrusters"]
    T_BOLLARD = TH["bollard_thrust_n"]                 # per pod
    ASTERN = TH["astern_fraction"]
    U_TOP = spec["propulsion"]["demo_top_speed_kn"] * 0.5144
    POD_AT = {side: np.array([TH["duct"]["centre_xy"][0], TH["duct"]["centre_xy"][1], s * PONTOON_Z])
              for side, s in (("port", -1.0), ("stbd", 1.0))}
    STEER_MAX = math.radians(TH["helm_deg"])
    DIFF_GAIN = 0.6
    ROTOR_RATE = 30.0                  # rad/s of the drawn rotors at full thrust

    def pod_thrust_max(u):
        """A rim drive loses thrust with speed: bollard at rest, 60 % of it at the top."""
        return T_BOLLARD * (1.0 - 0.4 * min(max(u / U_TOP, 0.0), 1.3))

    def pod_command(side):
        """-1 (full astern) .. 1 (full ahead) for one pod."""
        return min(max(st["throttle"] + (DIFF_GAIN if side == "port" else -DIFF_GAIN) * st["diff"], -1.0), 1.0)

    # A displacement catamaran: friction and form drag, and a mild wave-making hump
    # at Froude number 0.45 on her waterline length. Full throttle on both pods
    # balances it at U_TOP.
    HUMP_U, HUMP_W, HUMP_R = 0.45 * math.sqrt(G * DESIGN["lwl"]), 1.0, 200.0
    R_LIN = 40.0
    R_QUAD = (2.0 * pod_thrust_max(U_TOP) - R_LIN * U_TOP
              - HUMP_R * math.exp(-((U_TOP - HUMP_U) / HUMP_W) ** 2)) / U_TOP ** 2

    def resistance(u):
        au = abs(u)
        return math.copysign(R_LIN * au + R_QUAD * au * au
                             + HUMP_R * math.exp(-((au - HUMP_U) / HUMP_W) ** 2), u)

    SQUAT_M = 1400.0                   # bow-up moment at the hump (N m), about 1.5 deg of trim
    K_BANK = 0.0                       # a catamaran heels out of a turn on the sway force alone
    KV1, KV2 = 600.0, 1800.0           # sway damping: two hulls and their skegs broadside
    KR1, KR2 = 250.0, 7000.0           # yaw damping
    SWAY_Y = 0.2                       # height the lateral force acts at: mid-draft
    # A pod draws air when the aft sections on its side come out of the water.
    AFT = {"port": (SX < -1.5) & (SIDE < 0), "stbd": (SX < -1.5) & (SIDE > 0)}
    POD_WET_AREA = 0.3 * float(TAB_A[AFT["stbd"], int(round((DESIGN["waterline_y_at_x0"] - H0) / DH))].sum())

    # The ocean's footprint is one monohull plan form (elliptic bow, 75 % beam at the
    # stern) that holds the water on her waterline plane and fades out over 0.6 x
    # its half-beam. Anything wide enough to cover the pontoons covers the tunnel
    # too, so the water between them lies flat; this one keeps its flat core inside
    # her length so the fade, not a flat apron, is what trails past the transoms.
    EXCL_HALF_LENGTH, EXCL_HALF_BEAM = 2.0, 1.0
    X_CENTRE = 0.0
    _heel = min(spec["appendages"]["skeg"]["profile_xy"], key=lambda p: p[1])
    _duct = TH["duct"]
    # skeg heels and duct bottoms, both sides
    DEEP_PTS = np.array([[x, y, s * PONTOON_Z] for s in (-1.0, 1.0)
                         for (x, y) in (_heel, (_duct["centre_xy"][0], _duct["centre_xy"][1] - _duct["outer_r"]))])

    def boat_loads(u, heave, sway, w, wet, upright, trim_deg, roll, area):
        """Hull resistance, the two pods, the squat moment, damping."""
        Fb, Mb = np.zeros(3), np.zeros(3)

        def load(force, at):
            nonlocal Fb, Mb
            Fb = Fb + force
            Mb = Mb + np.cross(np.asarray(at) - COG, force)

        load(np.array([-resistance(u) * wet, -C_HEAVE * heave, 0.0]), COG)
        load(np.array([0.0, 0.0, -(KV1 * (abs(u) + 1.0) * sway + KV2 * sway * abs(sway)) * wet]),
             (COG[0], SWAY_Y, COG[2]))
        d = st["steer"]
        line = np.array([math.cos(d), 0.0, -math.sin(d)])        # thrust line of both pods
        for side in ("port", "stbd"):
            c = pod_command(side)
            pod_wet = min(max(float(area[AFT[side]].sum()) / POD_WET_AREA, 0.0), 1.0) * upright
            load(c * (pod_thrust_max(u) if c >= 0.0 else ASTERN * T_BOLLARD) * pod_wet * line, POD_AT[side])
        p_, r_, q_ = w                                  # roll, yaw, pitch rates (vessel X, Y, Z)
        Mb = Mb + np.array([-C_ROLL * p_,
                            -(KR1 * (abs(u) + 1.0) * r_ + KR2 * r_ * abs(r_)) * max(wet, 0.2),
                            -C_PITCH * q_ + SQUAT_M * math.exp(-((abs(u) - HUMP_U) / HUMP_W) ** 2) * wet])
        return Fb, Mb

    def helm_tick(dt):
        chase("throttle", st["throttle_cmd"], 0.8, dt)
        chase("steer", st["steer_cmd"], math.radians(60.0), dt)      # the pods slew at 60 deg/s
        chase("diff", st["diff_cmd"], 2.0, dt)
        for side in ("port", "stbd"):
            st["spin_" + side] = (st["spin_" + side] + ROTOR_RATE * pod_command(side) * dt) % (2.0 * math.pi)

    def helm_keys():
        st["diff_cmd"] = (1.0 if canvas.is_key_down("E") else 0.0) - (1.0 if canvas.is_key_down("Q") else 0.0)

    def helm_status():
        return f"thr {st['throttle']:+.2f}  steer {math.degrees(st['steer']):+5.1f}  diff {st['diff']:+.2f}"

    def bind_nodes(root):
        return {k + "_" + s: root.get_object_by_name(n.format(s)) for s in ("port", "stbd")
                for k, n in (("pod", "thruster_{}"), ("rotor", "thruster_{}_rotor"))}

    def pose_nodes():
        for s in ("port", "stbd"):
            NODES["pod_" + s].rotation.y = st["steer"]
            NODES["rotor_" + s].rotation.x = st["spin_" + s]

    def boat_foam(foam, u):
        spd = min(abs(u) / U_TOP, 1.0)
        for zs in (-PONTOON_Z, PONTOON_Z):
            foam(1.3, zs, 0.5, 0.06 + 0.35 * spd)                            # bow wave off each pontoon
        d = st["steer"]
        for side in ("port", "stbd"):
            c = pod_command(side)
            wash = POD_AT[side] - math.copysign(0.9, c) * np.array([math.cos(d), 0.0, -math.sin(d)])
            foam(wash[0], wash[2], 0.5 + 0.3 * spd, 0.5 * abs(c))            # pod wash, down the thrust line

    def autopilot(t):
        """Idle on the swell, both pods full ahead, a hard turn to port, ease off, stop her on
        the pods run astern, then turn her on the spot on differential thrust."""
        thr = 0.0 if t < 3.0 else (1.0 if t < 20.0 else (0.35 if t < 25.0 else (-0.5 if t < 29.0 else 0.0)))
        steer = 0.0
        if 11.0 <= t < 16.0:
            steer = -STEER_MAX
        elif 20.0 <= t < 25.0:
            steer = 0.6 * STEER_MAX
        st["throttle_cmd"] = thr
        st["steer_cmd"] = steer
        st["diff_cmd"] = 1.0 if t >= 29.5 else 0.0
        if CALM:
            st["throttle_cmd"] = st["steer_cmd"] = st["diff_cmd"] = 0.0


# --------------------------------------------------------------------------- #
#  Scene
# --------------------------------------------------------------------------- #
canvas = tp.Canvas(f"threepp - the {TITLE}", width=W, height=H, vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.72
ui = tp.ImguiContext(canvas, renderer) if (tp.HAS_IMGUI and not HEADLESS) else None

SUN_DIR = np.array([0.55, 0.42, -0.72])
SUN_DIR /= np.linalg.norm(SUN_DIR)


def make_sky_hdr(path, w=2048, h=1024):
    """A procedural sky with a sun, as in vulkan_ocean.py: no downloaded map needed."""
    j = np.arange(h).reshape(h, 1)
    i = np.arange(w).reshape(1, w)
    theta = (j / h) * math.pi
    phi = (i / w) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (h, w))
    x, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    horizon, zenith, haze = np.array([0.58, 0.64, 0.76]), np.array([0.07, 0.18, 0.44]), np.array([0.30, 0.34, 0.40])
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.35
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.6
    sky = np.where(y[..., None] >= 0.0, horizon * (1.0 - up) + zenith * up, horizon * (1.0 - down) + haze * down)
    sky = sky + np.exp(-(y * y) / (2 * 0.0045))[..., None] * np.array([1.0, 0.8, 0.6]) * 0.35
    ang = np.arccos(np.clip(np.stack([x, y, z], axis=-1) @ SUN_DIR, -1.0, 1.0))
    sky = sky + (np.exp(-(ang / math.radians(1.5)) ** 2) * 60.0
                 + np.exp(-(ang / math.radians(11.0)) ** 2) * 4.0)[..., None] * np.array([1.0, 0.95, 0.85])
    return write_radiance_hdr(path, sky)


env = tp.RGBELoader().load(make_sky_hdr(os.path.join(tempfile.gettempdir(), "threepp_usv_sky.hdr")))
scene = tp.Scene()
scene.background = env
scene.environment = env

SEA = 1000.0
ocean = tp.Ocean(size=SEA, resolution=512, wind_speed=max(WIND, 0.5), wind_theta=0.4,
                 choppiness=0.5, fetch=FETCH)
if FLAT:
    ocean.params.wave_scale = 0.0
scene.add(ocean)
floor_mat = tp.MeshStandardMaterial()
floor_mat.color = 0x04070a
floor_mat.roughness = 1.0
floor = tp.Mesh(tp.PlaneGeometry(SEA, SEA), floor_mat)
floor.rotate_x(-math.pi / 2)
floor.position.y = -30.0
scene.add(floor)

sun = tp.DirectionalLight(0xfff0dc, 2.6)
sun.position.set(*(SUN_DIR * 500.0))
sun.cast_shadow = True
scene.add(sun)

boat = tp.GLTFLoader().load(GLB).scene
boat.traverse(lambda o: (setattr(o, "cast_shadow", True), setattr(o, "receive_shadow", True))
              if isinstance(o, tp.Mesh) else None)
scene.add(boat)
NODES = bind_nodes(boat)
node_cam = boat.get_object_by_name("camera_main")

camera = tp.PerspectiveCamera(45.0, W / H, 0.1, 3000.0)
camera.position.set(-14.0, 5.0, -9.0)
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True
controls.min_distance = 4.0
controls.max_distance = 150.0
controls.target = tp.Vector3(0.0, 1.0, 0.0)
canvas.on_window_resize(resize_handler(camera, renderer))
cam_mode = "chase"
CHASE_FOV = 45.0
MAST_VFOV = math.degrees(2.0 * math.atan(math.tan(math.radians(spec["sensors"]["camera_main"]["hfov_deg"]) / 2.0)
                                          * H / W))


# --------------------------------------------------------------------------- #
#  State and one physics step
# --------------------------------------------------------------------------- #
st = {
    "p": np.array([0.0, 0.0, 0.0]),          # CoG, world
    "v": np.zeros(3),                        # CoG velocity, world
    "R": rot_y(0.0) @ rot_z(math.radians(DESIGN["trim_deg"])),
    "w": np.zeros(3),                        # angular velocity, vessel frame
    "floating": False, "water": np.zeros(2 * NS), "buoy": 0.0, "u": 0.0,
    **HELM,                                  # the boat's actuators and their commands
}
hull_excl_on = True
first_render_done = False


def probe_points():
    """World XZ under each half-section's probe, at the CoG's height."""
    rel = np.stack([SX - COG[0], np.zeros(2 * NS), SIDE * PROBE_Z - COG[2]], axis=1)
    return st["p"] + rel @ st["R"].T


def sample_water():
    pts = probe_points()
    st["water"] = np.array([ocean.sample_height(float(x), float(z), BUOY_MASK) for x, _, z in pts])
    return pts


def seat_on_water():
    """Drop her on her design waterline and trim where the sea is, at rest."""
    pts = sample_water()
    h0 = float(st["water"].mean())
    R = st["R"]
    wl = DESIGN["waterline_y_at_x0"]
    rel = np.array([0.0, wl, 0.0]) - COG
    st["p"][1] = h0 - float(R[1] @ rel)
    st["v"][:] = 0.0
    st["w"][:] = 0.0
    st["floating"] = True
    if DROP:
        # --drop: start 0.25 m high, 4 deg bow up and 6 deg to starboard, and let the
        # tables bring her back (--calm then shows she returns to the solver's pose)
        st["p"][1] += 0.25
        st["R"] = rot_y(0.0) @ rot_z(math.radians(DESIGN["trim_deg"] + 4.0)) @ np.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(0.105), -math.sin(0.105)], [0.0, math.sin(0.105), math.cos(0.105)]])


def step(dt):
    """One substep: strip buoyancy + the boat's loads, semi-implicit Euler."""
    R, v, w = st["R"], st["v"], st["w"]
    # --- buoyancy: each half-section at its own water level, in vessel y
    rx, rz = SX - COG[0], SIDE * PROBE_Z - COG[2]
    yb = COG[1] + (st["water"] - st["p"][1] - R[1, 0] * rx - R[1, 2] * rz) / max(R[1, 1], 0.3)
    t = np.clip((yb - H0) / DH, 0.0, NH - 1.001)
    i = t.astype(np.intp)
    f = t - i
    area = TAB_A[ROW, i] + (TAB_A[ROW, i + 1] - TAB_A[ROW, i]) * f
    zc = TAB_Z[ROW, i] + (TAB_Z[ROW, i + 1] - TAB_Z[ROW, i]) * f
    yc = TAB_Y[ROW, i] + (TAB_Y[ROW, i + 1] - TAB_Y[ROW, i]) * f
    fb = RHO * G * DX * area
    arm = np.stack([SX - COG[0], yc - COG[1], SIDE * zc - COG[2]], axis=1) @ R.T
    F = np.array([0.0, float(fb.sum()) - MASS * G, 0.0])
    tau = np.array([-float(arm[:, 2] @ fb), 0.0, float(arm[:, 0] @ fb)])     # sum r x (0, fb, 0)
    st["buoy"] = float(fb.sum())
    wet = min(max(st["buoy"] / (0.35 * MASS * G), 0.0), 1.0)
    upright = min(max(R[1, 1] * 2.0, 0.0), 1.0)
    trim_deg = math.degrees(math.asin(max(-1.0, min(1.0, R[1, 0]))))
    roll = math.atan2(-R[1, 2], R[1, 1])

    # --- vessel-frame loads: hull, propulsion, damping
    vb = R.T @ v
    u, heave, sway = vb
    st["u"] = u
    Fb, Mb = boat_loads(u, heave, sway, w, wet, upright, trim_deg, roll, area)

    # --- integrate: forces in the vessel frame against the per-axis added mass
    Fv = R.T @ F + Fb
    tv = R.T @ tau + Mb
    if DIAG:
        p_, r_ = w[0], w[1]
        st["diag"] = {"buoy_roll": float((R.T @ tau)[0]), "loads_roll": float(Mb[0] + C_ROLL * p_),
                      "damp_roll": float(-C_ROLL * p_), "bank": float(-K_BANK * u * r_ * wet),
                      "sway": float(sway), "r": float(r_), "u": float(u)}
    st["v"] = v + (R @ (Fv / M_EFF)) * dt
    st["p"] = st["p"] + st["v"] * dt
    Iw = I_EFF * w
    st["w"] = w + (tv - np.cross(w, Iw)) / I_EFF * dt
    st["R"] = orthonormal(R @ rodrigues(st["w"] * dt))


# --------------------------------------------------------------------------- #
#  The sea around her: displacement footprint, Kelvin wake, foam
# --------------------------------------------------------------------------- #
WAKE_MAX_AGE, WAKE_MAX_SAMPLES = 8.0, 64
# The ocean's analytic wake gates its foam trail on smoothstep(0.5, 1.5, speed)
# and spreads it over 1.5 x the footprint's half-beam (foam_world.comp), so fed
# her true 12 m/s it is a white carpet three beams wide from 3 kn up. Hand it a
# compressed speed and a footprint narrower than the collar; the churn right
# behind the transom is the jet-wash foam placed below, which follows throttle.
WAKE_FLOOR, WAKE_GAIN, WAKE_CAP = 0.6, 0.10, 1.2


def wake_speed(u):
    return math.copysign(min(WAKE_FLOOR + WAKE_GAIN * abs(u), WAKE_CAP), u) if abs(u) > 0.2 else 0.0


_wake = {"accum": 0.0, "last": None}


def sea_update(dt, pts):
    R, p = st["R"], st["p"]
    fwd = R[:, 0]
    yaw_ex = math.atan2(fwd[0], fwd[2])          # the ocean's heading: forward = (sin, cos)
    s_ex = np.array([math.cos(yaw_ex), -math.sin(yaw_ex)])
    c = p + R @ (np.array([X_CENTRE, 0.0, 0.0]) - COG)
    # the sea's own plane under her, least squares over the probes
    A = np.stack([np.ones(2 * NS), pts[:, 0] - c[0], pts[:, 2] - c[2]], axis=1)
    h0, gx, gz = np.linalg.lstsq(A, st["water"], rcond=None)[0]
    pitch = math.atan(gx * fwd[0] + gz * fwd[2])
    roll = math.atan(gx * s_ex[0] + gz * s_ex[1])
    ocean.hull_exclusion.set_pose(float(c[0]), float(c[2]), float(h0), yaw=yaw_ex, pitch=pitch, roll=roll,
                                  half_length=EXCL_HALF_LENGTH if hull_excl_on else 0.0, half_beam=EXCL_HALF_BEAM)
    u = st["u"]
    ws = wake_speed(u)
    ocean.wake.forward_speed = ws if hull_excl_on else 0.0
    if hull_excl_on:
        ocean.age_wake(dt, WAKE_MAX_AGE, WAKE_MAX_SAMPLES)
        _wake["accum"] += dt
        moved = 1e9 if _wake["last"] is None else math.hypot(c[0] - _wake["last"][0], c[2] - _wake["last"][1])
        if abs(u) > 0.4 and (_wake["accum"] >= 0.1 or moved >= 1.0):
            _wake["accum"] = 0.0
            _wake["last"] = (c[0], c[2])
            ocean.add_wake_sample(float(c[0]), float(c[2]), math.sin(yaw_ex), math.cos(yaw_ex), float(ws),
                                  WAKE_MAX_SAMPLES)

    ocean.clear_foam_disturbances()

    def foam(xb, zb, radius, k):
        q = p + R @ (np.array([xb, 0.4, zb]) - COG)
        if k > 0.02:
            ocean.add_foam_disturbance(float(q[0]), float(q[2]), radius, min(k, 1.0))
    boat_foam(foam, u)
    ocean.warp_toward(float(c[0]), float(c[2]), 0.3)


def place_boat():
    R, p = st["R"], st["p"]
    o = p - R @ COG                              # where the vessel origin is
    boat.position.set(float(o[0]), float(o[1]), float(o[2]))
    boat.quaternion.set(*quat_of(R))
    pose_nodes()


def frame(dt):
    """Advance the boat one rendered frame (needs the sea already rendered once)."""
    helm_tick(dt)
    if first_render_done:
        pts = sample_water()
        if not st["floating"]:
            seat_on_water()
        n = max(1, int(math.ceil(dt * SUBSTEPS_HZ)))
        for _ in range(n):
            step(dt / n)
        sea_update(dt, pts)
    place_boat()


# --------------------------------------------------------------------------- #
#  Readouts
# --------------------------------------------------------------------------- #
def readout():
    R, p = st["R"], st["p"]
    hd, pitch, roll = attitude(R)
    wl = float(np.mean(st["water"])) if st["floating"] else 0.0
    low = float(((DEEP_PTS - COG) @ R.T + p)[:, 1].min())
    return {"kn": st["u"] / 0.5144, "heading": math.degrees(hd) % 360.0, "trim": math.degrees(pitch),
            "roll": math.degrees(roll), "draft": wl - low, "buoy_kg": st["buoy"] / G}


_tele = {"next": 1.0}


def telemetry(t):
    if TELEMETRY and t >= _tele["next"]:
        _tele["next"] += 1.0
        r = readout()
        print(f"[usv] t {t:5.1f}  {r['kn']:5.1f} kn  hdg {r['heading']:5.1f}  trim {r['trim']:+5.2f}  "
              f"roll {r['roll']:+5.1f}  draft {r['draft']:.3f}  {helm_status()}")


# --------------------------------------------------------------------------- #
#  Input, the panel, the cameras
# --------------------------------------------------------------------------- #
_prev = {}


def pressed(key):
    now = canvas.is_key_down(key)
    fired = now and not _prev.get(key, False)
    _prev[key] = now
    return fired


def handle_keys(dt):
    global cam_mode, hull_excl_on
    if ui is not None and ui.want_capture_keyboard:
        return
    if canvas.is_key_down("W"):
        st["throttle_cmd"] = min(1.0, st["throttle_cmd"] + 0.5 * dt)
    if canvas.is_key_down("S"):
        st["throttle_cmd"] = max(THROTTLE_MIN, st["throttle_cmd"] - 0.5 * dt)
    steer = (1.0 if canvas.is_key_down("D") else 0.0) - (1.0 if canvas.is_key_down("A") else 0.0)
    st["steer_cmd"] = steer * STEER_MAX
    helm_keys()
    if pressed("Space"):
        st["throttle_cmd"] = 0.0
    if pressed("C"):
        cam_mode = "mast" if cam_mode == "chase" else "chase"
        camera.fov = MAST_VFOV if cam_mode == "mast" else CHASE_FOV
        camera.update_projection_matrix()
    if pressed("X"):
        hull_excl_on = not hull_excl_on
        if not hull_excl_on:
            ocean.clear_wake()


_anchor = np.zeros(3)


def camera_tick():
    """Chase: keep the orbit target on her without stealing the user's orbit. Mast: be the sensor."""
    global _anchor
    c = st["p"]
    if cam_mode == "mast":
        boat.update_matrix_world(True)
        wp, wq = node_cam.get_world_position(), node_cam.get_world_quaternion()
        camera.position.set(wp.x, wp.y, wp.z)
        camera.quaternion.set(wq.x, wq.y, wq.z, wq.w)
        _anchor = c.copy()
        return
    d = c - _anchor
    _anchor = c.copy()
    camera.position.set(camera.position.x + d[0], camera.position.y + 0.3 * d[1], camera.position.z + d[2])
    controls.target = tp.Vector3(float(c[0]), float(c[1]) + 0.6, float(c[2]))
    controls.update()


def draw_ui():
    r = readout()
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(330, 0)
    tp.imgui.begin(TITLE)
    tp.imgui.text(f"{r['kn']:5.1f} kn   heading {r['heading']:5.1f} deg")
    tp.imgui.text(f"trim {r['trim']:+5.2f}  roll {r['roll']:+5.1f}  draft {r['draft']:.3f} m")
    tp.imgui.text(helm_status())
    tp.imgui.separator()
    ch, v = tp.imgui.slider_float("throttle", st["throttle_cmd"], THROTTLE_MIN, 1.0)
    if ch:
        st["throttle_cmd"] = v
    ch, v = tp.imgui.slider_float("wave scale", ocean.params.wave_scale, 0.0, 2.5)
    if ch:
        ocean.params.wave_scale = v
    tp.imgui.separator()
    for line in HELP:
        tp.imgui.text(line)
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   load: {LOAD} {MASS:.0f} kg")
    tp.imgui.end()


# --------------------------------------------------------------------------- #
#  The scripted run's camera (the run itself is the boat's autopilot())
# --------------------------------------------------------------------------- #
_film = {"yaw": None}


def film_camera(t, dt):
    """A chase from her port quarter that swings slowly round to abeam, heading low-passed;
    --cam bow stands ahead of her starboard bow looking back at the stem instead."""
    fwd = st["R"][:, 0]
    if FILM_CAM == "abeam":
        # low, square off her starboard side: where the sea meets the hull
        c = st["p"]
        stb = st["R"][:, 2]
        eye = c + 9.0 * np.array([stb[0], 0.0, stb[2]])
        camera.position.set(float(eye[0]), 0.9, float(eye[2]))
        camera.look_at(tp.Vector3(float(c[0]), 0.4, float(c[2])))
        return
    if FILM_CAM == "bow":
        c = st["p"]
        stb = st["R"][:, 2]
        eye = c + 16.0 * np.array([fwd[0], 0.0, fwd[2]]) + 6.0 * np.array([stb[0], 0.0, stb[2]])
        camera.position.set(float(eye[0]), 2.6, float(eye[2]))
        tgt = c + 2.0 * np.array([fwd[0], 0.0, fwd[2]])
        camera.look_at(tp.Vector3(float(tgt[0]), 0.3, float(tgt[2])))
        return
    yaw = math.atan2(fwd[0], fwd[2])
    if _film["yaw"] is None:
        _film["yaw"] = yaw
    dy = (yaw - _film["yaw"] + math.pi) % (2 * math.pi) - math.pi
    _film["yaw"] += dy * (1.0 - math.exp(-dt / 1.2))
    a = _film["yaw"] + math.radians(200.0 - 70.0 * smoothstep(0.0, 34.0, t))
    dist, height = 15.0 - 3.0 * smoothstep(20.0, 34.0, t), 3.6
    c = st["p"]
    camera.position.set(float(c[0] + dist * math.sin(a)), float(c[1]) * 0.3 + height,
                        float(c[2] + dist * math.cos(a)))
    camera.look_at(tp.Vector3(float(c[0]), float(c[1]) + 0.4, float(c[2])))


sim_clock = 0.0


def sim_tick(dt):
    """Offline, the sea runs on the film's clock, not the wall clock (see warp_sailboat.py)."""
    global sim_clock
    sim_clock += dt
    renderer.sim_time = sim_clock


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
if HEADLESS:
    T_END = max(SHOT_T, RECORD_T)
    out = cli_arg("--out", "usv_ocean.mp4" if RECORD_T else "usv_ocean.png", str)
    fps = 60
    enc = None
    part = None
    if RECORD_T:
        part = out + ".part.mp4"
        enc = Encoder(part, W, H, fps, crf=18, preset="medium", faststart=True)
    t0 = time.perf_counter()
    for k in range(int(round(T_END * fps))):
        t = k / fps
        autopilot(t)
        frame(1.0 / fps)
        film_camera(t, 1.0 / fps)
        sim_tick(1.0 / fps)
        renderer.render(scene, camera)
        first_render_done = True
        telemetry(t)
        if DIAG and k % 6 == 0 and "diag" in st:          # roll balance every 0.1 s
            d_ = st["diag"]
            print(f"[diag] t {t:5.2f} roll {readout()['roll']:+6.1f}  buoyM {d_['buoy_roll']:+8.0f}  "
                  f"loadsM {d_['loads_roll']:+8.0f} (bank {d_['bank']:+7.0f})  dampM {d_['damp_roll']:+7.0f}  "
                  f"u {d_['u']:5.2f} sway {d_['sway']:+5.2f} r {d_['r']:+5.2f}")
        if enc is not None:
            enc.send(renderer.read_pixels())
    if enc is not None:
        rc = enc.close()
        if rc == 0:
            os.replace(part, out)
        print(f"[usv] wrote {out} ({T_END:.0f} s, {(time.perf_counter() - t0):.0f} s wall, encoder rc {rc})")
    else:
        renderer.save_frame(scene, camera, out)
        print(f"[usv] wrote {out}")
    if CALM:
        r = readout()
        print(f"[usv] calm water, load {LOAD}: draft {r['draft']:.3f} m (hydrostatic solver "
              f"{DESIGN['draft_max']:.3f}), trim {r['trim']:+.2f} deg (solver {DESIGN['trim_deg']:+.2f}), "
              f"buoyancy {r['buoy_kg']:.0f} kg of {MASS:.0f}")
else:
    clock = tp.Clock()
    wall = {"t": 0.0}

    def animate():
        global first_render_done
        dt = min(clock.get_delta(), 0.05)
        wall["t"] += dt
        handle_keys(dt)
        if CALM:
            autopilot(wall["t"])
        if ui is not None:
            controls.enabled = (not ui.want_capture_mouse) and cam_mode == "chase"
        frame(dt)
        camera_tick()
        renderer.render(scene, camera)
        first_render_done = True
        telemetry(wall["t"])
        if ui is not None:
            ui.render(draw_ui)

    canvas.animate(animate)
