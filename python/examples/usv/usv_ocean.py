"""A sea drone on the FFT ocean, floating on its own sections: the Mariner, the Otter X or the Otter.

The boat is <boat>.glb, the model build_<boat>_blender.py makes from
<boat>_spec.json, and it floats on <boat>_hydro.json: the same generator's
half-section buoyancy tables (Bonjean curves). Every frame the ocean's own
height field is sampled under every station on both sides, each half-section
looks up its immersed area and centroid at that local water level, and the sum
of those strip forces and their moments heaves, pitches and rolls a 6-DOF rigid
body. Nothing pulls the hull toward the surface: a crest under the bow lifts the
bow because the bow's sections got deeper. In flat water it settles where the
generator's hydrostatic solver put it; --calm prints that comparison.

The boat herself is ../usv_rig.py, which any scene can import: StripHull (the
strips and the rigid body), her drive (thrust, resistance, the helm, the
actuator nodes, the foam) and Wake (her footprint and wake on the ocean). This
file is the scene around her, the keys, the scripted run and the cameras.

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

--boat otter: the 2 m electric catamaran on two fixed pods. The helm is the
thrust difference between them, so with the throttle at zero she turns on the
spot. The Otter X's resistance terms at her size, tuned so full throttle makes
the vendor's 4.5 kn. The cameras stand closer to her.

All: the hydrostatics are the tables, the manoeuvring is a sketch.

    python usv_ocean.py                          # drive the Mariner
    python usv_ocean.py --boat otterx            # drive the Otter X
    python usv_ocean.py --boat otter             # drive the Otter
    python usv_ocean.py --calm                   # flat water: settle and compare
    python usv_ocean.py --shot 20 --out usv.png  # headless still after the scripted run
    python usv_ocean.py --record 36 --out usv.mp4

Options: --boat mariner|otterx|otter, --wind 7 (m/s), --fetch 30000 (m, 0 = open
ocean), --load <condition> (the spec's: departure|lightship|full_load for the
Mariner, survey|lightship|full_load for the Otters), --size 1280x720, --cam
chase|bow|abeam (headless framing), --telemetry, --flat (no waves, scripted
run), --drop (start displaced; with --calm she must come back to the solver's
pose), --diag (roll-moment balance every 0.1 s).

Keys: W/S throttle, A/D steer, Space all stop, C camera (chase / mast camera),
X hull displacement on/off. The Mariner: Q/E bow thruster, R reverse bucket
(hold). The Otter X: Q/E turn on the spot, S past zero runs the pods astern.
The Otter: A/D is the thrust difference, S past zero runs the pods astern.
Needs a Vulkan build and the boat's .glb (build it once, see build_<boat>_blender.py).
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(_HERE))                     # examples/ (demo_common, usv_rig)

import threepp as tp
from demo_common import Encoder, cli_arg, parse_size, resize_handler, write_radiance_hdr
from usv_rig import StripHull, Wake, Wash, attitude, drive_for, rot_y, rot_z, smoothstep

if not tp.HAS_VULKAN:
    print("The FFT ocean needs a Vulkan build of threepp (-DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)

BOAT = cli_arg("--boat", "mariner", str)
if BOAT not in ("mariner", "otterx", "otter"):
    print("--boat is mariner, otterx or otter")
    sys.exit(1)
GLB = os.path.join(_HERE, f"{BOAT}.glb")
HYDRO = os.path.join(_HERE, f"{BOAT}_hydro.json")
if not (os.path.exists(GLB) and os.path.exists(HYDRO)):
    print(f"{BOAT}.glb / {BOAT}_hydro.json are generated. Build them once, from this folder:\n"
          "  blender --background "
          f"--factory-startup --python build_{BOAT}_blender.py -- --spec {BOAT}_spec.json --out {BOAT}.glb")
    sys.exit(1)

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
LOAD = cli_arg("--load", None, str)          # None: the spec's design condition
W, H = parse_size(cli_arg("--size", "1280x720", str))
FILM_CAM = cli_arg("--cam", "chase", str)       # headless framing: chase | bow | abeam
SUBSTEPS_HZ = 240.0


# --------------------------------------------------------------------------- #
#  The demo's half of each boat: its title, how close the cameras stand (VIEW:
#  the distances are the Mariner's; a smaller boat scales them), the keys beyond
#  W/S/A/D and the scripted run. Each takes the boat's drive (usv_rig) and
#  writes its commands.
# --------------------------------------------------------------------------- #
if BOAT == "mariner":
    TITLE = "Mariner USV"
    VIEW = 1.0
    HELP = ("W/S throttle  A/D steer  Q/E bow thruster", "R reverse bucket  Space stop  C camera  X hull")

    def helm_keys(d):
        d.thruster = (1.0 if canvas.is_key_down("E") else 0.0) - (1.0 if canvas.is_key_down("Q") else 0.0)
        d.bucket_cmd = 1.0 if canvas.is_key_down("R") else 0.0

    def autopilot(d, t):
        """Idle on the swell, run up onto the plane, a hard turn to port, ease off, brake on the
        reverse bucket, then walk the bow round on the thruster."""
        thr = 0.0 if t < 3.0 else (1.0 if t < 22.0 else (0.35 if t < 27.0 else (0.6 if t < 31.5 else 0.0)))
        steer = 0.0
        if 13.0 <= t < 19.0:
            steer = -d.steer_max
        elif 23.0 <= t < 27.0:
            steer = 0.6 * d.steer_max
        d.throttle_cmd = thr
        d.steer_cmd = steer
        d.bucket_cmd = 1.0 if 27.0 <= t < 31.5 else 0.0   # jet turned round: braking
        d.thruster = 1.0 if t >= 31.5 else 0.0
        if CALM:
            d.throttle_cmd = d.steer_cmd = d.bucket_cmd = d.thruster = 0.0

elif BOAT == "otterx":
    TITLE = "Otter X USV"
    VIEW = 1.0
    HELP = ("W/S throttle (S past zero: astern)  A/D steer", "Q/E turn on the spot  Space stop  C camera  X hull")

    def helm_keys(d):
        d.diff_cmd = (1.0 if canvas.is_key_down("E") else 0.0) - (1.0 if canvas.is_key_down("Q") else 0.0)

    def autopilot(d, t):
        """Idle on the swell, both pods full ahead, a hard turn to port, ease off, stop her on
        the pods run astern, then turn her on the spot on differential thrust."""
        thr = 0.0 if t < 3.0 else (1.0 if t < 20.0 else (0.35 if t < 25.0 else (-0.5 if t < 29.0 else 0.0)))
        steer = 0.0
        if 11.0 <= t < 16.0:
            steer = -d.steer_max
        elif 20.0 <= t < 25.0:
            steer = 0.6 * d.steer_max
        d.throttle_cmd = thr
        d.steer_cmd = steer
        d.diff_cmd = 1.0 if t >= 29.5 else 0.0
        if CALM:
            d.throttle_cmd = d.steer_cmd = d.diff_cmd = 0.0

else:
    TITLE = "Otter USV"
    VIEW = 0.45                        # a 2 m boat: the cameras stand this much closer
    HELP = ("W/S throttle (S past zero: astern)", "A/D steer on differential thrust  Space stop  C camera  X hull")

    def helm_keys(d):
        pass

    def autopilot(d, t):
        """Idle on the swell, both pods full ahead, a turn to port on the thrust difference, ease
        off, stop her on the pods run astern, then turn her on the spot."""
        thr = 0.0 if t < 3.0 else (1.0 if t < 20.0 else (0.35 if t < 25.0 else (-0.5 if t < 27.5 else 0.0)))
        steer = 0.0
        if 11.0 <= t < 16.0:
            steer = -1.0
        elif 20.0 <= t < 25.0:
            steer = 0.6
        d.throttle_cmd = thr
        d.steer_cmd = 1.0 if t >= 28.5 else steer
        if CALM:
            d.throttle_cmd = d.steer_cmd = 0.0


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
ocean.wake_field.resolution, ocean.wake_field.patches = 1024, 1      # what she leaves on the water (usv_rig.Wash)
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
node_cam = boat.get_object_by_name("camera_main")

# The boat: her strips, her drive at the helm, her mark on the sea (usv_rig).
hull = StripHull(BOAT, ocean, LOAD)
hull.drive = drive = drive_for(hull).bind(boat)
hull.R = rot_y(0.0) @ rot_z(math.radians(hull.design["trim_deg"]))
wake = Wake(hull, ocean)
wash = Wash(hull, drive, ocean, 0)

camera = tp.PerspectiveCamera(45.0, W / H, 0.1, 3000.0)
camera.position.set(-14.0 * VIEW, 5.0 * VIEW, -9.0 * VIEW)
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True
controls.min_distance = 4.0 * VIEW
controls.max_distance = 150.0
controls.target = tp.Vector3(0.0, 1.0, 0.0)
canvas.on_window_resize(resize_handler(camera, renderer))
cam_mode = "chase"
CHASE_FOV = 45.0
MAST_VFOV = math.degrees(2.0 * math.atan(math.tan(math.radians(hull.spec["sensors"]["camera_main"]["hfov_deg"]) / 2.0)
                                          * H / W))


# --------------------------------------------------------------------------- #
#  One frame of the boat
# --------------------------------------------------------------------------- #
afloat = False
hull_excl_on = True
first_render_done = False


def seat_on_water():
    """Drop her on her design waterline and trim where the sea is, at rest."""
    global afloat
    hull.seat(0.0, 0.0, 0.0)
    afloat = True
    if DROP:
        # --drop: start 0.25 m high, 4 deg bow up and 6 deg to starboard, and let the
        # tables bring her back (--calm then shows she returns to the solver's pose)
        hull.p[1] += 0.25
        hull.R = rot_y(0.0) @ rot_z(math.radians(hull.design["trim_deg"] + 4.0)) @ np.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(0.105), -math.sin(0.105)], [0.0, math.sin(0.105), math.cos(0.105)]])


def sea_update(dt):
    """The sea around her: displacement footprint, Kelvin wake, foam."""
    u = wake.update(dt, hull_excl_on)
    ocean.clear_foam_disturbances()
    drive.foam(ocean, u)
    ocean.clear_wake_sources()
    wash.update(dt, hull_excl_on)             # her propulsors' races and her hull's lane, in the wake field
    c = wake.centre
    ocean.warp_toward(float(c[0]), float(c[2]), 0.3)


def frame(dt):
    """Advance the boat one rendered frame (needs the sea already rendered once)."""
    drive.tick(dt)
    if first_render_done:
        if not afloat:
            seat_on_water()
        hull.advance(dt, max(1, int(math.ceil(dt * SUBSTEPS_HZ))))
        sea_update(dt)
    hull.place(boat)
    drive.pose()


# --------------------------------------------------------------------------- #
#  Readouts
# --------------------------------------------------------------------------- #
def readout():
    R, p = hull.R, hull.p
    hd, pitch, roll = attitude(R)
    wl = float(np.mean(hull.water)) if afloat else 0.0
    low = float(((drive.deep_pts - hull.cog) @ R.T + p)[:, 1].min())
    return {"kn": hull.u / 0.5144, "heading": math.degrees(hd) % 360.0, "trim": math.degrees(pitch),
            "roll": math.degrees(roll), "draft": wl - low, "buoy_kg": hull.buoy / hull.g}


_tele = {"next": 1.0}


def telemetry(t):
    if TELEMETRY and t >= _tele["next"]:
        _tele["next"] += 1.0
        r = readout()
        print(f"[usv] t {t:5.1f}  {r['kn']:5.1f} kn  hdg {r['heading']:5.1f}  trim {r['trim']:+5.2f}  "
              f"roll {r['roll']:+5.1f}  draft {r['draft']:.3f}  {drive.status()}")


def roll_balance(t):
    """--diag: the last substep's roll moments about the CoG, vessel frame."""
    vb, w, tb, Mb = hull.balance
    p_, r_ = w[0], w[1]
    damp = -hull.c_roll * p_
    print(f"[diag] t {t:5.2f} roll {readout()['roll']:+6.1f}  buoyM {float(tb[0]):+8.0f}  "
          f"loadsM {float(Mb[0] - damp):+8.0f} (bank {float(-drive.K_BANK * vb[0] * r_ * drive.wet):+7.0f})  "
          f"dampM {float(damp):+7.0f}  u {float(vb[0]):5.2f} sway {float(vb[2]):+5.2f} r {float(r_):+5.2f}")


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
        drive.throttle_cmd = min(1.0, drive.throttle_cmd + 0.5 * dt)
    if canvas.is_key_down("S"):
        drive.throttle_cmd = max(drive.throttle_min, drive.throttle_cmd - 0.5 * dt)
    steer = (1.0 if canvas.is_key_down("D") else 0.0) - (1.0 if canvas.is_key_down("A") else 0.0)
    drive.steer_cmd = steer * drive.steer_max
    helm_keys(drive)
    if pressed("Space"):
        drive.throttle_cmd = 0.0
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
    c = hull.p
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
    tp.imgui.text(drive.status())
    tp.imgui.separator()
    ch, v = tp.imgui.slider_float("throttle", drive.throttle_cmd, drive.throttle_min, 1.0)
    if ch:
        drive.throttle_cmd = v
    ch, v = tp.imgui.slider_float("wave scale", ocean.params.wave_scale, 0.0, 2.5)
    if ch:
        ocean.params.wave_scale = v
    tp.imgui.separator()
    for line in HELP:
        tp.imgui.text(line)
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   load: {hull.load} {hull.mass:.0f} kg")
    tp.imgui.end()


# --------------------------------------------------------------------------- #
#  The scripted run's camera (the run itself is the boat's autopilot())
# --------------------------------------------------------------------------- #
_film = {"yaw": None}


def film_camera(t, dt):
    """A chase from her port quarter that swings slowly round to abeam, heading low-passed;
    --cam bow stands ahead of her starboard bow looking back at the stem instead."""
    fwd = hull.R[:, 0]
    if FILM_CAM == "abeam":
        # low, square off her starboard side: where the sea meets the hull
        c = hull.p
        stb = hull.R[:, 2]
        eye = c + 9.0 * VIEW * np.array([stb[0], 0.0, stb[2]])
        camera.position.set(float(eye[0]), 0.9 * VIEW, float(eye[2]))
        camera.look_at(tp.Vector3(float(c[0]), 0.4 * VIEW, float(c[2])))
        return
    if FILM_CAM == "bow":
        c = hull.p
        stb = hull.R[:, 2]
        eye = c + 16.0 * VIEW * np.array([fwd[0], 0.0, fwd[2]]) + 6.0 * VIEW * np.array([stb[0], 0.0, stb[2]])
        camera.position.set(float(eye[0]), 2.6 * VIEW, float(eye[2]))
        tgt = c + 2.0 * VIEW * np.array([fwd[0], 0.0, fwd[2]])
        camera.look_at(tp.Vector3(float(tgt[0]), 0.3, float(tgt[2])))
        return
    yaw = math.atan2(fwd[0], fwd[2])
    if _film["yaw"] is None:
        _film["yaw"] = yaw
    dy = (yaw - _film["yaw"] + math.pi) % (2 * math.pi) - math.pi
    _film["yaw"] += dy * (1.0 - math.exp(-dt / 1.2))
    a = _film["yaw"] + math.radians(200.0 - 70.0 * smoothstep(0.0, 34.0, t))
    dist, height = (15.0 - 3.0 * smoothstep(20.0, 34.0, t)) * VIEW, 3.6 * VIEW
    c = hull.p
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
        autopilot(drive, t)
        frame(1.0 / fps)
        film_camera(t, 1.0 / fps)
        sim_tick(1.0 / fps)
        renderer.render(scene, camera)
        first_render_done = True
        telemetry(t)
        if DIAG and k % 6 == 0 and hull.balance is not None:          # roll balance every 0.1 s
            roll_balance(t)
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
        print(f"[usv] calm water, load {hull.load}: draft {r['draft']:.3f} m (hydrostatic solver "
              f"{drive.solver_draft:.3f}), trim {r['trim']:+.2f} deg (solver {hull.design['trim_deg']:+.2f}), "
              f"buoyancy {r['buoy_kg']:.0f} kg of {hull.mass:.0f}")
else:
    clock = tp.Clock()
    wall = {"t": 0.0}

    def animate():
        global first_render_done
        dt = min(clock.get_delta(), 0.05)
        wall["t"] += dt
        handle_keys(dt)
        if CALM:
            autopilot(drive, wall["t"])
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
