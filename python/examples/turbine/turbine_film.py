"""Phase 4, the film: five robots inspect an offshore wind turbine, from blade tip to seabed.

    python turbine_film.py --preview                 640x360, every 2nd frame, the whole film
    python turbine_film.py                           1920x1080, 30 fps
    python turbine_film.py --shots air_climb,rov     only these shots (each renderable alone)
    python turbine_film.py --from 20 --to 40         only this film-time window (s)
    python turbine_film.py --cards                   title and end card stills over existing renders (no GPU)
    [--sheet] a contact sheet, 3 tiles per shot (always written with the film)

One process: the site, the fleet and the jobs are built once; each shot says which slice of
which job clock it shows; within a shot film time and job time run together at real time. The
sea clock only moves forward. Text (title, labels, inset frames, end card) is drawn with PIL in
1920x1080 design pixels and composited on the read-back frame (scaled for the preview).
Headless only. Frame: X upwind, Y up, origin on the pile axis at MSL (turbine_site.py).
"""
import math
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)
for _p in (os.path.join(_EX, "netpen"), os.path.dirname(_EX), _EX, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from turbine_out import latest, out_dir

# ----------------------------------------------------------------------------- #
#  The words (one place)
# ----------------------------------------------------------------------------- #
TITLE = "Blade Tip to Seabed"
SUBTITLE = "Five robots inspect an offshore wind turbine"
CREDIT = "Simulated and rendered in threepp"
LABELS = {
    "drone": ("DRONE", "blade inspection"),
    "mariner": ("MARINER", "mothership"),
    "otter": ("OTTER X", "multibeam survey"),
    "rov": ("ROV", "pile and anodes"),
    "snake": ("SNAKE ROBOT", "cable protection"),
}
OUT = out_dir("film")
NUMBERS = os.path.join(out_dir("jobs"), "survey_numbers.txt")     # written by jobs_air.py
FONTS = {"regular": "segoeui.ttf", "light": "segoeuisl.ttf", "bold": "segoeuib.ttf"}
DW, DH = 1920, 1080                    # design pixels
FPS = 30


def findings():
    """The end card's three lines, numbers from the survey table."""
    txt = open(NUMBERS, encoding="utf-8").read()
    err = float(re.search(r"\|err\| mean all r<60\s+([\d.]+) m", txt).group(1))
    m = re.search(r"scour pit depth\s+([\d.]+) m\s+([\d.]+) m\s+([\d.]+) m", txt)
    pit, true = m.group(1), m.group(3)
    return [("Drone", "leading-edge erosion found on blade 1"),
            ("Otter X sonar", f"seabed mapped to {round(err * 100):.0f} cm; scour pit {pit} m deep (true {true} m)"),
            ("ROV", "anodes and pile inspected"),
            ("Snake robot", "cable sleeve worn through where it meets the rock")]


# ----------------------------------------------------------------------------- #
#  Text: RGBA overlays in design pixels, scaled once, composited with an alpha
# ----------------------------------------------------------------------------- #
def _font(kind, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONTS[kind], int(round(size)))      # PIL searches the system font folders
    except OSError:
        return ImageFont.load_default(int(round(size)))


def _spaced(draw, xy, s, font, fill, tracking):
    x, y = xy
    for ch in s:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking
    return x


def _gradient(img, x0, y0, x1, y1, alpha, fade_right=True, fade_top=True):
    """A soft dark wash behind text: full at the lower left, fading right and up."""
    a = np.zeros((DH, DW), np.float32)
    ys, xs = np.mgrid[0:DH, 0:DW].astype(np.float32)
    u = np.clip((xs - x0) / max(x1 - x0, 1), 0, 1) if fade_right else np.zeros_like(xs)
    v = np.clip((y1 - ys) / max(y1 - y0, 1), 0, 1) if fade_top else np.zeros_like(ys)
    w = (1 - u * u * (3 - 2 * u)) * (1 - v * v * (3 - 2 * v))
    a = alpha * w * ((xs >= 0) & (ys >= y0 - 1))
    arr = np.asarray(img).copy()
    arr[..., 3] = np.maximum(arr[..., 3], (a * 255).astype(np.uint8))
    return arr


class Overlay:
    """Premultiplied colour + alpha at the output size, only inside its bounding box."""

    def __init__(self, rgba, W, H):
        from PIL import Image
        im = Image.fromarray(rgba, "RGBA")
        if (W, H) != (DW, DH):
            # premultiply before the resize so edges do not pick up the transparent black
            a = rgba[..., 3:4].astype(np.float32) / 255.0
            pm = np.concatenate([rgba[..., :3] * a, a * 255.0], -1)
            chans = [np.asarray(Image.fromarray(pm[..., k].astype(np.float32), "F").resize((W, H), Image.LANCZOS))
                     for k in range(4)]
            A = np.clip(chans[3] / 255.0, 0, 1)
            C = np.stack([np.clip(c, 0, 255) for c in chans[:3]], -1) / 255.0
        else:
            A = rgba[..., 3].astype(np.float32) / 255.0
            C = rgba[..., :3].astype(np.float32) / 255.0 * A[..., None]
        ys, xs = np.nonzero(A > 1e-3)
        if len(ys) == 0:
            self.box = None
            return
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        self.box = (y0, y1, x0, x1)
        self.A = A[y0:y1, x0:x1, None]
        self.C = C[y0:y1, x0:x1]

    def apply(self, rgb, alpha=1.0):
        if self.box is None or alpha <= 0.002:
            return rgb
        y0, y1, x0, x1 = self.box
        f = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
        f = f * (1.0 - alpha * self.A) + alpha * self.C
        rgb[y0:y1, x0:x1] = np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return rgb


def _canvas():
    from PIL import Image
    return Image.new("RGBA", (DW, DH), (0, 0, 0, 0))


def title_rgba():
    from PIL import ImageDraw
    arr = _gradient(_canvas(), 0, 640, 1180, 1080, 0.36)
    from PIL import Image
    im = Image.fromarray(arr, "RGBA")
    d = ImageDraw.Draw(im)
    d.text((118, 812), TITLE, font=_font("light", 112), fill=(246, 247, 245, 255), anchor="ls")
    d.text((124, 884), SUBTITLE, font=_font("regular", 46), fill=(228, 231, 232, 255), anchor="ls")
    return np.asarray(im)


def label_rgba(key):
    from PIL import Image, ImageDraw
    name, job = LABELS[key]
    arr = _gradient(_canvas(), 0, 860, 900, 1080, 0.45)
    im = Image.fromarray(arr, "RGBA")
    d = ImageDraw.Draw(im)
    d.rectangle([96, 952, 99, 1000], fill=(246, 247, 245, 255))                  # a thin bar
    fb, fr = _font("bold", 36), _font("regular", 36)
    x = _spaced(d, (118, 950), name, fb, (246, 247, 245, 255), 2.0)
    d.text((x + 22, 950), job, font=fr, fill=(206, 213, 216, 255))
    return np.asarray(im)


def inset_rgba(x, y, w, h, label):
    """A thin frame round the inset and its name tag (design pixels)."""
    from PIL import Image, ImageDraw
    im = _canvas()
    d = ImageDraw.Draw(im)
    d.rectangle([x - 2, y - 2, x + w + 1, y + h + 1], outline=(236, 238, 236, 235), width=2)
    f = _font("bold", 22)
    tw = _spaced(ImageDraw.Draw(_canvas()), (0, 0), label, f, (0, 0, 0, 0), 1.5)
    d.rectangle([x, y, x + tw + 22, y + 34], fill=(10, 12, 14, 190))
    _spaced(d, (x + 11, y + 3), label, f, (240, 242, 240, 255), 1.5)
    return np.asarray(im)


def endcard_rgba(lines):
    from PIL import Image, ImageDraw
    im = _canvas()
    d = ImageDraw.Draw(im)
    fb, fr = _font("bold", 42), _font("regular", 42)
    y = 360
    out = []
    for who, what in lines:
        one = _canvas()
        dd = ImageDraw.Draw(one)
        x = 160
        dd.text((x, y), who + ":", font=fb, fill=(246, 247, 245, 255))
        x += fb.getlength(who + ":") + 16
        dd.text((x, y), what, font=fr, fill=(222, 228, 230, 255))
        out.append(np.asarray(one))
        y += 84
    cr = _canvas()
    ImageDraw.Draw(cr).text((160, y + 90), CREDIT, font=_font("regular", 28), fill=(196, 204, 208, 255))
    out.append(np.asarray(cr))
    return out


# ----------------------------------------------------------------------------- #
#  Files
# ----------------------------------------------------------------------------- #
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
    print(f"[film] wrote {path}", flush=True)


def cards_from_stills():
    """M1: the title and end cards at full size over existing renders (no GPU)."""
    from PIL import Image
    title_bg, end_bg = latest("site", "site_far"), latest("jobs", "subsea_pair_still")
    if not (title_bg and end_bg):
        raise FileNotFoundError("the cards need stills: run turbine_site.py --shot site_far and jobs_subsea.py --jobs pair first")
    bg = np.asarray(Image.open(title_bg).convert("RGB")).copy()
    Overlay(title_rgba(), DW, DH).apply(bg)
    save_png(bg, versioned("title_card", "png"))
    bg = np.asarray(Image.open(end_bg).convert("RGB")).astype(np.float32)
    bg = (bg * 0.38).astype(np.uint8)
    for o in endcard_rgba(findings()):
        Overlay(o, DW, DH).apply(bg)
    save_png(bg, versioned("end_card", "png"))


# ----------------------------------------------------------------------------- #
#  Motion helpers
# ----------------------------------------------------------------------------- #
def min_jerk(u):
    u = min(max(float(u), 0.0), 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def smooth(e0, e1, x):
    u = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def bez(P, u):
    P = [np.asarray(p, float) for p in P]
    while len(P) > 1:
        P = [(1 - u) * a + u * b for a, b in zip(P[:-1], P[1:])]
    return P[0]


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fade(s, dur, t_in=0.35, t_out=0.5, start=0.0, length=None):
    """0..1 envelope for a text block on [start, start + length]."""
    length = dur - start if length is None else length
    a = smooth(start, start + t_in, s)
    b = 1.0 - smooth(start + length - t_out, start + length, s)
    return min(a, b)


# ----------------------------------------------------------------------------- #
#  The film
# ----------------------------------------------------------------------------- #
ROV_LAMP_I = 5.0             # ROV job lamps (jobs_subsea 38 took the anode toward white)
PAIR_LAMP_I = 22.0 
CLOSE_ROV_I = 7.0            # the ROV over the close hold: the patch keeps its scuffed grey          # the ROV over the work site (jobs_subsea 48)
SNAKE_LAMP_I = 0.30          # the snake's lamps: one constant value, no feedback
TETHER_SLACK = 1.14
SURVEY_SEGS = (58.0, 150.0, None)      # survey tau at each segment start (None: ends at the full circle)
SEG_DUR = 3.7


def build(W, H, preview):
    import threepp as tp
    import turbine_site as ts
    import turbine_fleet as tf
    import fleet_vehicles as fv
    import jobs_air as ja
    import jobs_subsea as js
    from demo_common import standard_material
    ctx = type("Ctx", (), {})()
    ctx.tp, ctx.ts, ctx.tf, ctx.fv, ctx.ja, ctx.js = tp, ts, tf, fv, ja, js
    ctx.W, ctx.H, ctx.k = W, H, W / DW
    t0 = time.perf_counter()
    canvas, renderer = ja.make_renderer(W, H)
    ctx.canvas, ctx.renderer = canvas, renderer
    S = ts.build_site(renderer)
    # the Otter's circle on her own clock: F.update reads tf.otter_on_circle at call time
    ctx.circle0 = tf.otter_on_circle
    ctx.otter_off = [0.0]
    tf.otter_on_circle = lambda t: ctx.circle0(t + ctx.otter_off[0])
    F = tf.build_fleet(renderer, S)
    J = js.build_subsea(renderer, S, F)
    V = F.vehicles
    S.set_lamps((0, 0, 0), (1, 0, 0), False)
    camera = tp.PerspectiveCamera(45.0, W / H, 0.3, 20000.0)
    ctx.S, ctx.F, ctx.J, ctx.V, ctx.camera = S, F, J, V, camera
    # ROV: film lamps, glowing lenses seen from the side, spinning props
    for sp in V["rov_lights"]:
        sp.intensity = ROV_LAMP_I
    J.flood.intensity = 2.0
    lens = standard_material(0xfff4e0, roughness=0.3, emissive=0xfff0d0, emissive_intensity=30.0)
    for z in (-0.12, 0.12):
        m = tp.Mesh(tp.SphereGeometry(0.018, 18, 12), lens)
        m.scale.set(0.5, 1.0, 1.0)
        m.position.set(0.212, -0.075, z)
        V["rov"].add(m)
    ctx.rov_props = []
    for c in V["rov"].children:
        try:
            ch = c.children
            if len(ch) == 6 and len(ch[5].children) == 3:
                ctx.rov_props.append(ch[5])
        except Exception:
            pass
    ctx.rov_prop_ang = 0.0
    # the snake: the physics model if it imports, else the keyframed stand-in
    ctx.snake_dyn = None
    try:
        import jobs_snake_dyn as sd
        ctx.sd = sd
        ctx.snake_props = sd.add_props(V["snake_links"])
        ctx.snake_dyn = sd.SnakeDyn(S, V["snake_out"], look=J.look)
        me = S.motes.emitter                      # the snow drifts with the snake model's current
        me.wind = tp.Vector3(float(sd.CURRENT[0]), 0.0, float(sd.CURRENT[2]))
        S.motes.set_emitter(me)
        print("[film] snake: physics model (jobs_snake_dyn.SnakeDyn)", flush=True)
    except Exception as e:
        print(f"[film] snake: dynamics module unavailable ({e!r}); keyframed stand-in", flush=True)
    # the survey map
    J.head_cam.far = 10.0                          # no unfogged background along a level line of sight
    J.head_cam.update_projection_matrix()
    ctx.map = ja.MapInset(dict(np.load(ja.NPZ)))
    print(f"[film] built in {time.perf_counter() - t0:.0f} s; ROV props {len(ctx.rov_props)}", flush=True)
    # the ocean's first frame, then the boats settle on it
    ctx.sea_t = ts.T_SHOT - 6.0
    F.t = ctx.sea_t
    aim(ctx, np.array([300.0, 20.0, 190.0]), np.array([0.0, 88.0, 0.0]), 45.0, False)
    renderer.sim_time = ctx.sea_t
    renderer.render(S.scene, camera)
    rate = 15.0 if preview else 30.0
    for _ in range(int(6.0 * rate)):
        ride(ctx, 1.0 / rate, False)
    ctx.D = ja.build_drone_job(S, F)
    V["drone"].have = False
    J.rov_reset(-30.0)
    ctx.views = {}
    return ctx


def aim(ctx, eye, tgt, fov, under):
    c = ctx.camera
    c.fov = fov
    c.near, c.far = (0.03, 20000.0) if under else (0.3, 20000.0)
    c.update_projection_matrix()
    c.position.set(*map(float, eye))
    c.look_at(ctx.tp.Vector3(*map(float, tgt)))


def ride(ctx, dt, under):
    """One unrecorded frame: render the sea at sea_t, then the boats step on it."""
    r, S, F = ctx.renderer, ctx.S, ctx.F
    r.sim_time = ctx.sea_t
    S.update(ctx.sea_t, ctx.camera, under, 0.0)
    r.sim_time = ctx.sea_t
    r.render(S.scene, ctx.camera)
    F.update(ctx.sea_t, dt)
    hide_phase2_tether(ctx)
    ctx.sea_t += dt


def hide_phase2_tether(ctx):
    if ctx.V["tether"] is not None:
        ctx.V["tether"].visible = False


# ---- the ROV's tether: slack in a hanging chain, a bight off the ROV that curves up
def rov_tether(ctx):
    js, J, st = ctx.js, ctx.J, ctx.J.rov_state
    R = J.rov_R
    tail = st["p"] + R @ js.TETHER_LOCAL
    f = _unit(R[:, 0] * np.array([1.0, 0.0, 1.0]))
    top = J.moonpool()
    guide = top - 1.2 * js.UP + 0.03 * _unit((tail - top) * np.array([1.0, 0.0, 1.0]))
    drift = _unit(np.array([-0.08, 0.0, 0.015]))
    q = tail - 0.3 * f + 1.4 * js.UP + 0.2 * drift              # where the bight meets the main span
    cat = js.catenary(guide, q, slack=TETHER_SLACK, n=132)
    chord = guide + np.linspace(0.0, 1.0, 132)[:, None] * (q - guide)
    sag = np.linalg.norm(cat - chord, axis=1)
    cat = chord + sag[:, None] * drift[None, :]                  # neutrally buoyant: the slack streams with the current
    u = np.linspace(0.0, 1.0, 17)[1:]
    ctl = tail - 0.25 * f + 0.35 * js.UP                             # leaves aft, then curves up
    tan_q = _unit(cat[-1] - cat[-3])                                 # join the chain smoothly
    bight = np.array([bez([q, q + 0.45 * tan_q, ctl, tail], w) for w in u])
    P = np.vstack([top + np.linspace(0.0, 1.0, 12, endpoint=False)[:, None] * (guide - top), cat, bight])
    return _tether_mesh(ctx, P)


def _tether_mesh(ctx, P):
    J, ts, js = ctx.J, ctx.ts, ctx.js
    geo = ts.tube(P, lambda s: np.full_like(s, js.TETHER_R), sides=8)
    if J.tether_mesh is None:                       # 160 points, as jobs_subsea's tether (updated in place)
        J.tether_mesh = ctx.tp.Mesh(geo, ctx.V["tether_mat"])
        ctx.S.scene.add(J.tether_mesh)
    else:
        g = J.tether_mesh.geometry
        g.update_attribute("position", np.asarray(geo.get_attribute("position"), np.float32))
        g.update_attribute("normal", np.asarray(geo.get_attribute("normal"), np.float32))
    return P


def spin_rov_props(ctx, dt):
    v = float(np.linalg.norm(ctx.J.rov_state.get("v", np.zeros(3))))
    rev = min(2.5 + 3.0 * v / 0.45, 4.5)
    ctx.rov_prop_ang += 2 * math.pi * rev * dt
    for k, p in enumerate(ctx.rov_props):
        p.rotation.y = float(ctx.rov_prop_ang * (1 if k % 2 else -1))


def rov_close_cam(ctx, tau):
    """jobs_subsea.rov_cam brought in: the ROV about a fifth of the frame width."""
    st, R = ctx.J.rov_state, ctx.J.rov_R
    p = st["p"]
    f = _unit(R[:, 0] * np.array([1.0, 0.0, 1.0]))
    sd = np.cross(ctx.js.UP, f)
    tang = -sd if np.dot(-sd, np.array([-p[2], 0.0, p[0]])) < 0 else sd
    eye = p + 1.75 * tang - 0.3 * f + np.array([0.0, 0.6, 0.0])
    lit = p + 1.8 * f + np.array([0.0, -0.4, 0.0])
    tgt = 0.7 * p + 0.3 * lit + np.array([0.0, 0.05, 0.0])
    return eye, tgt, 40.0


# ---- the snake (physics model)
def snake_pose(ctx):
    sd, SD, V = ctx.sd, ctx.snake_dyn, ctx.V
    j = SD.joints()
    ctx.fv.pose_snake(V["snake_links"], j)
    for sp in V["snake_lamps"]:
        sp.intensity = SNAKE_LAMP_I
    for g, s_, kind, idx in ctx.snake_props:
        a = SD.prop_ang[idx]
        if kind == "side":
            g.rotation.x = float(a)
        else:
            g.rotation.z = float(a)
    hc = ctx.J.head_cam
    f = _unit(j[0] - j[1])
    r = _unit(np.cross(f, sd.UP))
    u = np.cross(r, f)
    hc.position.set(*map(float, j[0] + 0.004 * f))
    hc.quaternion.set(*map(float, ctx.js.quat_of(np.stack([r, u, -f], axis=1))))
    ctx.snake_nose = j[0]
    return j


def snake_film_cam(ctx, j, state):
    """jobs_snake_dyn.render's camera: low, three-quarter front, low-passed; never locked to the head."""
    sd, D = ctx.sd, ctx.snake_dyn
    fwd = _unit(sd.ydir(D.psi_f))
    eye_hold = D.look + 1.55 * _unit(0.55 * _unit(D.SL.at(D.SL.s_td)[1] * [1, 0, 1]) + 1.0 * D.out)
    eye_tr = j[4] + 1.9 * _unit(1.0 * fwd + 0.9 * D.out)
    w = smooth(-1.0, 6.0, D.t - (D.t_app if D.t_app is not None else 1e9))
    eye = (1 - w) * eye_tr + w * eye_hold
    eye[1] = float(sd.bed_max(eye[0], eye[2], 0.2)) + 0.5
    tgt = (1 - w) * (0.55 * j[1] + 0.45 * j[5]) + w * (0.45 * D.look + 0.55 * (D.p_f + [0, D.y, 0]))
    tgt[1] = min(tgt[1], eye[1] - 0.12)
    a_e = 1.0 - math.exp(-(1.0 / FPS) / 1.2)
    a_t = 1.0 - math.exp(-(1.0 / FPS) / 0.8)
    if "e" not in state:
        state["e"], state["t"] = eye, tgt
    state["e"] = state["e"] + a_e * (eye - state["e"])
    state["t"] = state["t"] + a_t * (tgt - state["t"])
    return state["e"].copy(), state["t"].copy(), 44.0


# ----------------------------------------------------------------------------- #
#  Shots: name, duration, the job clock, the camera, insets, label
# ----------------------------------------------------------------------------- #
def mariner_cam(ctx, a_deg):
    """Close on the Mariner at her station: she about a quarter of the width, the pad in view."""
    mx, mz = ctx.F.mariner_xz
    M = np.array([mx, 0.0, mz])
    a = math.radians(a_deg)
    return M + 17.0 * np.array([math.cos(a), 0.0, math.sin(a)]) + np.array([0.0, 7.0, 0.0]), M + np.array([0.0, 1.0, 0.0])


def descent_cam(ctx, u):
    """Blade to the boats: an eased Bezier; the pitch never lets the far sea (beyond ~300 m) in."""
    e3, t3 = mariner_cam(ctx, 50.0)
    E = [np.array([50.0, 92.0, 30.0]), np.array([66.0, 62.0, 54.0]), np.array([40.0, 26.0, 58.0]), e3]
    T = [ctx.D.defect, np.array([5.0, 45.0, 0.0]), np.array([0.0, 8.0, 8.0]), t3]
    eye, t = bez(E, u), bez(T, u)
    dh = t - eye
    hd = float(np.linalg.norm(dh[[0, 2]]))
    dep = math.degrees(math.atan2(-dh[1], hd))
    rule = 19.0 + math.degrees(math.atan(max(eye[1], 0.0) / 300.0)) + 1.0
    p = math.radians(dep + max(rule - dep, 0.0) * (1.0 - smooth(0.8, 1.0, u)))
    dirh = _unit(dh * np.array([1.0, 0.0, 1.0]))
    return eye, eye + hd * (math.cos(p) * dirh - math.sin(p) * np.array([0.0, 1.0, 0.0])) / max(math.cos(p), 0.2), 38.0


def shot_table(ctx):
    """[(name, dur, enter(ctx), frame(ctx, s, dt) -> (eye, tgt, fov, under, exposure), inset, label)]."""
    shots = []
    D, J, V, ts = ctx.D, ctx.J, ctx.V, ctx.ts

    def drone_pose(tau):
        D.pose(tau)

    # 1 title: from 3 m above the sea, looking a little up; the turbine on the right third
    def f_title(c, s, dt):
        drone_pose(0.0)
        E = ts.pol(205.0 - 8.0 * s / 4.5, 40.0, 3.0)                 # a slow push in
        d = _unit(-E * np.array([1.0, 0.0, 1.0]))
        r = _unit(np.cross(d, np.array([0.0, 1.0, 0.0])))
        a = math.radians(13.0)
        d2 = math.cos(a) * d - math.sin(a) * r
        return E, E + 100.0 * d2 + np.array([0.0, 100.0 * math.tan(math.radians(19.0)), 0.0]), 44.0, False, 0.6
    shots.append(dict(name="title", dur=4.5, frame=f_title, text="title"))

    # 2 air: lift-off from the Mariner, then the climb up the leading edge to the patch
    def e_liftoff(c):
        V["drone"].have = False
        c.views_on("drone")

    def f_liftoff(c, s, dt):
        tau = 0.3 + s
        drone_pose(tau)
        e, t, fov = D.chase(tau)
        return e, t, fov, False, 0.7
    shots.append(dict(name="air_liftoff", dur=3.5, enter=e_liftoff, frame=f_liftoff, inset="drone", label="drone"))

    TAU_CLIMB = 22.0
    upb = _unit(ts.blade_point(1, 80.0) - ts.blade_point(1, 90.0))

    def e_climb(c):
        V["drone"].have = False
        for _ in range(30):
            D.pose(TAU_CLIMB)
            V["drone"].set_pose(V["drone_pos"], V["drone_look"], 1.0 / 60.0)

    def f_climb(c, s, dt):
        tau = TAU_CLIMB + s
        drone_pose(tau)
        p, _ = D.route.pose(tau)
        blade = p - ctx.ja.STANDOFF * D.defect_n
        eye = p + 0.5 * D.defect_n + 1.3 * upb + np.array([0.0, 0.0, 4.2])     # close: the drone ~ a fifth of the width
        return eye, p + 0.2 * (blade - p) - 0.6 * upb, 48.0, False, 0.7
    shots.append(dict(name="air_climb", dur=15.0, enter=e_climb, frame=f_climb, inset="drone"))

    # 3 descent 1: blade to the boats, one eased move; then the Mariner holding station
    DES = 8.0

    def f_descent(c, s, dt):
        drone_pose(TAU_CLIMB + 15.0 + s)
        e, t, fov = descent_cam(c, min_jerk(s / DES))
        return e, t, fov, False, 0.75
    shots.append(dict(name="descent_air", dur=DES, enter=lambda c: c.views_off(), frame=f_descent))

    def f_mariner(c, s, dt):
        drone_pose(TAU_CLIMB + 15.0 + DES + s)
        u = s / 4.0
        w = u - math.sin(math.pi * u) / math.pi                          # starts at rest, as the descent ended
        e, t = mariner_cam(c, 50.0 + 14.0 * w)
        e = e + w * np.array([0.0, -1.2, 0.0])
        return e, t, 38.0, False, 0.75
    shots.append(dict(name="surface_mariner", dur=4.0, frame=f_mariner, label="mariner"))

    # 4 the Otter X survey: three real-time segments, camera fixed to the turbine
    for i, tau0 in enumerate(SURVEY_SEGS):
        tau0 = (ctx.ja.SURVEY_T - SEG_DUR) if tau0 is None else tau0

        def e_otter(c, tau0=tau0):
            preroll = 1.0
            c.otter_off[0] = ctx.tf.T_SHOT + tau0 - (c.sea_t + preroll)
            x, z, hd, vx, vz, rate, *_ = c.circle0(ctx.tf.T_SHOT + tau0 - preroll)
            c.F.otter.seat(x, z, hd)
            c.F.otter.v = np.array([vx, 0.0, vz])
            c.F.otter.w = np.array([0.0, rate, 0.0])
            e, t, fov = otter_cam(c, tau0 + 0.5 * SEG_DUR)
            aim(c, e, t, fov, False)
            for _ in range(int(round(preroll / c.dt))):
                ride(c, c.dt, False)
            c.map_img = None

        def f_otter(c, s, dt, tau0=tau0):
            drone_pose(47.0)
            c.survey_tau = tau0 + s
            e, t, fov = otter_cam(c, tau0 + 0.5 * SEG_DUR)
            return e, t, fov, False, 0.8
        shots.append(dict(name=f"surface_otter_{'abc'[i]}", dur=SEG_DUR, enter=e_otter, frame=f_otter, inset="map",
                          label="otter" if i == 0 else None))

    # 5 descent 2: under water by the pile, down through the murk
    def e_uw(c):
        J.rov_reset(-6.0)
        c.snow_settle = True

    def f_uw(c, s, dt):
        J.rov_job(s - 6.0, dt)
        rov_tether(c)
        u = s / 6.0
        y = -1.6 - 7.2 * (0.35 * u + 0.65 * min_jerk(u))
        eye = ts.pol(11.5, 116.0 - 6.0 * u, y)
        tgt = ts.pol(5.2, 98.0, y - 2.6)
        return eye, tgt, 60.0, True, 1.4
    shots.append(dict(name="descent_water", dur=6.0, enter=e_uw, frame=f_uw))

    # 6 the ROV at the anode cage
    def f_rov(c, s, dt):
        J.rov_job(s, dt)
        rov_tether(c)
        e, t, fov = rov_close_cam(c, s)
        return e, t, fov, True, 1.4
    shots.append(dict(name="rov", dur=12.0, enter=lambda c: J.rov_reset(0.0) if "p" not in J.rov_state else None,
                      frame=f_rov, label="rov"))

    # 7 seabed: ONE snake clock across both shots. Wide first (the ROV lights the site, the snake
    # under way along the sleeve), then close for the arrival, the neck onto the patch and the hold.
    SN0, WIDE, CLOSE = 1.0, 8.0, 12.0
    cam_state = {}

    def snake_at(c, t_dyn):
        if c.snake_dyn is not None:
            c.snake_dyn.advance(t_dyn)
            return snake_pose(c)
        j = J.snake_job(min(t_dyn * ctx.js.SNAKE_T / 29.0, ctx.js.SNAKE_T))
        c.snake_nose = j[0]
        return j

    def e_wide(c):
        c.snow_settle = True
        c.views_off()

    def f_wide(c, s, dt):
        J.pair(s)                                   # the ROV parked over the site, its tether; the snake is posed below
        for sp in V["rov_lights"]:
            sp.intensity = PAIR_LAMP_I
        snake_at(c, SN0 + s)
        look = J.look
        c_, t_ = J.sleeve.at(J.sleeve.s_td)
        fwdc = _unit(t_ * np.array([1.0, 0.0, 1.0]))
        e0 = look + 5.5 * _unit(0.55 * fwdc + J.out) + np.array([0.0, 1.6, 0.0])
        t0 = look + np.array([0.0, 0.9, 0.0])
        w = min_jerk(s / WIDE)
        e = e0 + 0.6 * w * (look - e0) * np.array([1.0, 0.0, 1.0]) * 0.25       # a slow push in
        return e, t0, 60.0, True, 1.4
    shots.append(dict(name="seabed_wide", dur=WIDE, enter=e_wide, frame=f_wide))

    def e_close(c):
        cam_state.clear()
        c.views_on("snake")

    def f_close(c, s, dt, t0=SN0 + WIDE):
        J.pair(WIDE + s)                            # the ROV stays over the site
        for sp in V["rov_lights"]:
            sp.intensity = CLOSE_ROV_I
        J.flood.intensity = 0.5
        j = snake_at(c, t0 + s)
        if c.snake_dyn is not None:
            e, t, fov = snake_film_cam(c, j, cam_state)
        else:
            e, t, fov = J.snake_cam(j, 14.0)
        return e, t, fov, True, 1.4
    shots.append(dict(name="snake", dur=CLOSE, enter=e_close, frame=f_close, inset="snake", label="snake"))

    def f_end(c, s, dt):
        e, t, fov, u, x = f_close(c, CLOSE + s, dt)
        return e, t, fov, u, x
    shots.append(dict(name="end", dur=8.0, enter=lambda c: c.views_off(), frame=f_end, text="end"))
    t = 0.0
    for sh in shots:
        sh["start"] = t
        t += sh["dur"]
    return shots


def otter_cam(ctx, tau_mid):
    """jobs_air.otter_cam on the survey clock (fixed for a segment, relative to the turbine)."""
    x, z, *_ = ctx.circle0(ctx.tf.T_SHOT + tau_mid)
    rn = np.array([x, 0.0, z]) / math.hypot(x, z)
    tn = np.array([-rn[2], 0.0, rn[0]])
    eye = np.array([x, 0.0, z]) + 20.0 * rn - 6.0 * tn + np.array([0.0, 9.0, 0.0])
    tgt = 0.55 * np.array([x, 0.0, z])
    tgt[1] = eye[1] - math.tan(math.radians(16.0)) * float(np.linalg.norm((tgt - eye)[[0, 2]]))
    return eye, tgt, 30.0


# ----------------------------------------------------------------------------- #
#  The master loop
# ----------------------------------------------------------------------------- #
def run():
    from demo_common import Encoder, cli_arg
    preview = "--preview" in sys.argv
    W, H = (640, 360) if preview else (1920, 1080)
    step = 2 if preview else 1
    only = cli_arg("--shots", "", str)
    only = set(only.split(",")) if only else None
    t_from, t_to = cli_arg("--from", 0.0, float), cli_arg("--to", 1e9, float)
    t_all = time.perf_counter()
    ctx = None
    for attempt in range(2):
        try:
            ctx = build(W, H, preview)
            break
        except RuntimeError as e:
            if attempt:
                raise
            print("[film] build failed, retry in 30 s:", e, flush=True)
            time.sleep(30.0)
    ctx.dt = step / FPS
    r, S, F, J, V, tp = ctx.renderer, ctx.S, ctx.F, ctx.J, ctx.V, ctx.tp
    k = ctx.k
    M = int(round(36 * k))
    # insets: the drone's gimbal and the snake's head camera as device views; the map pasted
    PW, PH = int(round(560 * k)), int(round(315 * k))
    pip_cam = tp.PerspectiveCamera(15.0, PW / PH, 0.3, 20000.0)
    rect = (W - M - PW, M, PW, PH)                     # snake cam: top right
    rect_d = (W - M - PW, H - M - PH, PW, PH)          # drone cam: bottom right (the drone flies top right)
    MS = int(round(360 * k))
    rect_map = (W - M - MS, M, MS, MS)
    dz = lambda x, y, w, h: (int(round(x / k)), int(round(y / k)), int(round(w / k)), int(round(h / k)))
    ov = {"drone": Overlay(inset_rgba(*dz(*rect_d), "DRONE CAM"), W, H),
          "snake": Overlay(inset_rgba(*dz(*rect), "SNAKE CAM"), W, H),
          "map": Overlay(inset_rgba(*dz(*rect_map), "SONAR MAP"), W, H),
          "title": Overlay(title_rgba(), W, H)}
    for key in LABELS:
        ov["label_" + key] = Overlay(label_rgba(key), W, H)
    ov["end"] = [Overlay(a, W, H) for a in endcard_rgba(findings())]

    def views_on(which):
        views_off()
        cam = pip_cam if which == "drone" else J.head_cam
        if which == "snake":
            J.head_cam.aspect = PW / PH
            J.head_cam.update_projection_matrix()
        v = r.add_view(cam, PW, PH)
        if v:
            r.set_view_display_rect(v, *(rect_d if which == "drone" else rect))
        ctx.views[which] = v

    def views_off():
        for w_, v in list(ctx.views.items()):
            if v:
                r.remove_view(v)
        ctx.views.clear()
    ctx.views_on, ctx.views_off = views_on, views_off

    shots = shot_table(ctx)
    total = sum(s["dur"] for s in shots)
    print("[film] shots: " + ", ".join(f"{s['name']} {s['start']:.1f}+{s['dur']:.1f}" for s in shots) + f"; total {total:.1f} s", flush=True)
    stem = "film_preview" if preview else "blade_tip_to_seabed"
    out = versioned(stem, "mp4")
    ver = os.path.basename(out)[:-4].split("_")[-1]
    tmp = os.path.join(OUT, "_tmp_" + os.path.basename(out))
    enc = Encoder(tmp, W, H, FPS // step, crf=18, preset="medium", faststart=True)
    tiles, stats = [], []
    ctx.snow_ever = False
    t_loop = time.perf_counter()
    nfr = 0
    for sh in shots:
        if only and sh["name"] not in only:
            continue
        n = int(round(sh["dur"] * FPS / step))
        ks = [i for i in range(n) if t_from <= sh["start"] + i * ctx.dt < t_to]
        if not ks:
            continue
        t_sh = time.perf_counter()
        ctx.snow_settle = False
        if sh.get("enter"):
            sh["enter"](ctx)
        pick = {ks[int(round(f * (len(ks) - 1)))] for f in (0.1, 0.5, 0.9)}
        rec = dict(name=sh["name"], cam=[], veh=[], clip=[])
        for i in ks:
            s = i * ctx.dt
            dt = ctx.dt if i != ks[0] else 0.0
            eye, tgt, fov, under, expo = sh["frame"](ctx, s, ctx.dt)
            aim(ctx, eye, tgt, fov, under)
            r.tone_mapping_exposure = expo
            if under and (ctx.snow_settle or not ctx.snow_ever):
                S.update(ctx.sea_t, ctx.camera, True, 0.0)
                S.settle_snow(ctx.camera, ctx.sea_t)
                ctx.snow_settle, ctx.snow_ever = False, True
            spin_rov_props(ctx, ctx.dt)
            if "drone" in ctx.views:
                ge, gt, gf = ctx.D.gimbal_cam(sh_tau(sh, s))
                pip_cam.fov = gf
                pip_cam.update_projection_matrix()
                pip_cam.position.set(*map(float, ge))
                pip_cam.look_at(tp.Vector3(*map(float, gt)))
            r.sim_time = ctx.sea_t
            S.update(ctx.sea_t, ctx.camera, under, ctx.dt)
            r.sim_time = ctx.sea_t
            r.render(S.scene, ctx.camera)
            rgb = r.read_pixels().copy()
            rec["clip"].append(float((rgb.max(axis=2) >= 250).mean()))
            # insets
            ins = sh.get("inset")
            if ins in ("drone", "snake") and ctx.views.get(ins):
                x, y, w, h = rect_d if ins == "drone" else rect
                img = r.read_view_rgb_pixels(ctx.views[ins])
                if img.size:
                    rgb[y:y + h, x:x + w] = img[:h, :w]
                ov[ins].apply(rgb)
            elif ins == "map":
                if ctx.map_img is None or i % max(1, 3 // step) == 0:
                    from PIL import Image
                    m = ctx.map.image(ctx.survey_tau)
                    ctx.map_img = np.asarray(Image.fromarray(m).resize((MS, MS), Image.LANCZOS))
                x, y, w, h = rect_map
                rgb[y:y + h, x:x + w] = ctx.map_img
                ov["map"].apply(rgb)
            # words
            if sh.get("label"):
                ov["label_" + sh["label"]].apply(rgb, fade(s, sh["dur"], 0.35, 0.5, 0.25, min(3.2, sh["dur"] - 0.4)))
            if sh.get("text") == "title":
                ov["title"].apply(rgb, fade(s, sh["dur"], 0.9, 0.9, 0.5, 3.6))
            if sh.get("text") == "end":
                dim = 1.0 - 0.62 * smooth(0.0, 1.5, s)
                rgb[:] = (rgb.astype(np.float32) * dim).astype(np.uint8)
                for j_, o in enumerate(ov["end"]):
                    o.apply(rgb, fade(s, sh["dur"], 0.6, 0.8, 1.0 + 0.8 * j_, 7.0 - 0.8 * j_))
            enc.send(rgb)
            nfr += 1
            if i in pick:
                tiles.append((sh["name"], rgb.copy()))
            if not preview and i == ks[len(ks) // 2]:
                save_png(rgb, versioned(f"shot_{sh['name']}", "png"))
            if not preview and i == ks[-1] and sh["name"] in ("air_climb", "snake"):
                nm = {"air_climb": "drone_hold", "snake": "snake_hold"}[sh["name"]]
                save_png(rgb, versioned(f"blade_tip_to_seabed_still_{nm}", "png"))
            if sh.get("text") == "title" and not preview and i == ks[int(len(ks) * 0.55)]:
                save_png(rgb, versioned("title_card", "png"))
                save_png(rgb, versioned("blade_tip_to_seabed_still_title", "png"))
            if sh.get("text") == "end" and not preview and i == ks[-1]:
                save_png(rgb, versioned("end_card", "png"))
            rec["cam"].append(np.array(eye, float))
            subj = subject_points(ctx, sh["name"])
            if subj is not None:
                rec.setdefault("margin", []).append(frame_margin(subj, eye, tgt, fov, W / H,
                                                                 rect_d if sh.get("inset") == "drone" else (rect if sh.get("inset") else None), W, H))
            rec["veh"].append(vehicle_positions(ctx))
            # the boats (and the drone rig) step on this frame's sea
            if sh["name"].startswith("air") or sh["name"] in ("title", "descent_air", "surface_mariner") or \
                    sh["name"].startswith("surface_otter"):
                ctx.D.pose(sh_tau(sh, s + ctx.dt))
            F.update(ctx.sea_t, ctx.dt)
            hide_phase2_tether(ctx)
            ctx.sea_t += ctx.dt
        stats.append(rec)
        print(f"[film] {sh['name']}: {len(ks)} frames in {time.perf_counter() - t_sh:.0f} s", flush=True)
    views_off()
    rc = enc.close()
    if rc == 0:
        os.replace(tmp, out)
        print(f"[film] wrote {out}: {nfr} frames in {time.perf_counter() - t_loop:.0f} s loop, "
              f"{time.perf_counter() - t_all:.0f} s total", flush=True)
    else:
        print(f"[film] encoder rc {rc}", flush=True)
    contact(tiles, versioned("film_sheet", "png"))
    report(stats, ctx.dt)


def subject_points(ctx, name):
    """World points bounding the shot's subject (for the in-frame margin check)."""
    V = ctx.V
    if name == "air_climb":
        c = np.array(V["drone_pos"], float)
        o = np.array([[x, y, z] for x in (-0.6, 0.6) for y in (-0.35, 0.25) for z in (-0.6, 0.6)])
        return c + o
    if name == "rov":
        c = np.array(ctx.J.rov_state["p"], float)
        o = np.array([[x, y, z] for x in (-0.3, 0.3) for y in (-0.15, 0.2) for z in (-0.25, 0.25)])
        return c + o
    if name == "snake" and ctx.snake_dyn is not None:
        return ctx.snake_dyn.joints()
    return None


def frame_margin(P, eye, tgt, fov, aspect, inset, W, H):
    """Smallest distance of the subject's projected points to a frame edge (fraction of the width),
    and whether any point falls inside the inset."""
    f = _unit(np.asarray(tgt, float) - eye)
    r = _unit(np.cross(f, [0.0, 1.0, 0.0]))
    u = np.cross(r, f)
    d = P - eye
    z = d @ f
    ty = math.tan(math.radians(fov) / 2)
    x = (d @ r) / (z * ty * aspect)
    y = (d @ u) / (z * ty)
    mx = np.minimum(1 - x, 1 + x) / 2
    my = np.minimum(1 - y, 1 + y) / 2 * (H / W)
    hit = False
    if inset is not None:
        px, py = (x + 1) / 2 * W, (1 - y) / 2 * H
        ix, iy, iw, ih = inset
        hit = bool(np.any((px > ix) & (px < ix + iw) & (py > iy) & (py < iy + ih)))
    return float(min(mx.min(), my.min())), hit


def sh_tau(sh, s):
    """The drone's route time for a shot at shot time s (the drone clock)."""
    base = {"title": None, "air_liftoff": 0.3, "air_climb": 22.0, "descent_air": 37.0, "surface_mariner": 45.0}
    b = base.get(sh["name"])
    if b is None:
        return 0.0 if sh["name"] == "title" else 47.0
    return b + s


def vehicle_positions(ctx):
    V, F = ctx.V, ctx.F
    return dict(drone=np.array(V["drone_pos"], float), mariner=F.mariner.p.copy(), otter=F.otter.p.copy(),
                rov=np.array(ctx.J.rov_state.get("p", np.zeros(3)), float),
                snake=np.array(getattr(ctx, "snake_nose", np.zeros(3)), float))


def report(stats, dt):
    print("[film] per shot: camera peak speed (m/s) / accel (m/s^2); clipped px (>=250) max %; jump flags "
          "(per-frame step > 3x the shot median and > 2 cm)")
    for rec in stats:
        c = np.array(rec["cam"])
        line = f"  {rec['name']:18s}"
        if len(c) > 2:
            v = np.linalg.norm(np.diff(c, axis=0), axis=1) / dt
            a = np.abs(np.diff(v)) / dt
            line += f" cam v {v.max():6.2f} a {a.max():6.2f}"
            flags = []
            steps = {"camera": np.linalg.norm(np.diff(c, axis=0), axis=1)}
            for key in rec["veh"][0]:
                P = np.array([vv[key] for vv in rec["veh"]])
                steps[key] = np.linalg.norm(np.diff(P, axis=0), axis=1)
            for key, st in steps.items():
                med = float(np.median(st))
                bad = np.nonzero((st > 3.0 * med) & (st > 0.02))[0]
                if len(bad):
                    flags.append(f"{key}@{list(bad[:4])} max {st.max():.3f} m (median {med:.3f})")
            line += f" clip {100 * max(rec['clip']):5.2f}%"
            if rec.get("margin"):
                mm = np.array([m for m, _ in rec["margin"]])
                line += f" subject margin min {100 * mm.min():.1f}% of width, in inset {sum(h for _, h in rec['margin'])} frames;"
            line += " jumps: " + ("; ".join(flags) if flags else "none")
        print(line, flush=True)


def contact(tiles, path, w=480):
    from PIL import Image, ImageDraw
    if not tiles:
        return
    names = []
    for nm, _ in tiles:
        if nm not in names:
            names.append(nm)
    h = int(w * tiles[0][1].shape[0] / tiles[0][1].shape[1])
    sheet = Image.new("RGB", (3 * w + 8, len(names) * (h + 4) - 4), (20, 20, 20))
    col = {}
    d = ImageDraw.Draw(sheet)
    for nm, img in tiles:
        row = names.index(nm)
        c = col.get(nm, 0)
        col[nm] = c + 1
        sheet.paste(Image.fromarray(img).resize((w, h), Image.LANCZOS), (c * (w + 4), row * (h + 4)))
        if c == 0:
            d.text((6, row * (h + 4) + 4), nm, fill=(255, 230, 60))
    save_png(np.asarray(sheet), path)


if __name__ == "__main__":
    if "--cards" in sys.argv:
        cards_from_stills()
    else:
        run()
