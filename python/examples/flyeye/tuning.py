"""Phase 2 figure (b): T4/T5 tuning to full-field drifting sinusoidal gratings rendered by threepp Vulkan.

    py -3.14 python/examples/flyeye/tuning.py                 # run, figure, renders (about 5 min)
    py -3.14 python/examples/flyeye/tuning.py --figure-only   # figure from <out>/gratings.npz
    # more wavelengths (3, 16 columns) into <out>/extra, merged into the figure by --extra; run before --figure-only:
    py -3.14 python/examples/flyeye/tuning.py --lambdas 3,16 --contrasts "" --extra --out C:/dev/_flyeye/phase2/extra
        --renders C:/dev/_flyeye/renders/phase2/extra --figure C:/dev/_flyeye/phase2/extra/figure.png

Stage: render_edges.Stage (403 px primary canvas, 90 deg camera at the origin looking down -Z,
everything adaptive off, flush 1). One unlit MeshBasicMaterial plane per (wavelength, contrast,
axis) at depth 1, parked far outside the view when unused. Its sRGB texture is one sinusoid period
per wavelength in display-encoded 8-bit values, 128 * (1 + c sin), one texel per screen pixel, so c is
the Michelson contrast of the luma flyvis sees. A condition slides the plane at TF * wavelength px/s
(geometry motion, so TAA reprojects with exact motion vectors). The unlit colour passes an 8-bit
LINEAR store, so the rendered sinusoid is quantised (1 level near grey, up to 13 in the darks at
c = 0.8); contrast is measured from the rendered frames, per condition.

Eye: an EyeView (add_view secondary view, 403 px, the primary's twin camera), the sensor's own
path. Not the primary through FrameTensors view 0: the primary gets an RCAS sharpen that raises the
contrast of the 26 px grating. The one secondary view sees only 4 of the 8 TAA jitter phases (shared
Halton counter), a static +0.06 px mean shift that does not change a drifting grating's amplitude.

Per condition, 100 Hz, sim time pinned: 8 static renders, fade_in on the first frame (1 s of model
steps), then the drift for 0.5 s + max(2 s, 2 periods); the first 0.5 s is dropped. Wavelengths 2, 4,
8 columns (13 px per column), TF 0.25..16 Hz in octaves, drift right/left/up/down (image space), c
0.5; then c 0.05..0.8 at the best TF of 4 columns. Inner columns: hex radius <= 11 (397 of 721).
Column centres sit on a 13 px grid horizontally, so a 2-column (26 px) grating is at the lattice's
Nyquist period: right and left drift are the same counterphase flicker per column, and the peak
spread test uses wavelengths of 3 columns and up.

- amplitude: mean over inner columns of the time-averaged relu rate minus rest (MotionField.raw),
  per type and direction; preferred T?a left, b right, c up, d down; null = the opposite.
- direction accuracy: fraction of inner columns whose time-averaged MotionField vector is within
  45 deg of the drift, and the same per frame (averaged over frames).
- measured contrast: over the central whole periods within 208 px, the luma profile across the stripes; its mean,
  fundamental amplitude / mean, and (max - min) / (max + min), averaged over the analysis frames.

Writes <out>/gratings.npz, <out>/report.json, figures/phase2_tuning.png, and under --renders a
contact sheet, its tiles, an mp4 of the first 0.6 s of every condition, and a short mp4 of a few.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

import threepp as tp  # noqa: E402

from flyeye.eye import EyeView, bgra_luma_u8  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402
from flyeye.readouts import MOTION_TYPES, MotionField  # noqa: E402
from flyeye.render_edges import Stage  # noqa: E402
from flyeye.sanity_rig import VideoOut  # noqa: E402

SIZE, FOV, DT, COL_PX = 403, 90.0, 0.01, 13
LAMBDAS = (2, 4, 8)  # columns
TFS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DIRS = ("right", "left", "up", "down")
DIR_VEC = dict(right=(1.0, 0.0), left=(-1.0, 0.0), up=(0.0, 1.0), down=(0.0, -1.0))  # x right, y up
OPPOSITE = dict(right="left", left="right", up="down", down="up")
PREF = ("left", "right", "up", "down")  # a, b, c, d for T4 and T5
CONTRASTS = (0.05, 0.1, 0.2, 0.4, 0.8)
MAIN_C, MID_LAMBDA = 0.5, 4
DROP, WARM, MARGIN_PX, INNER_RADIUS, BAND = 0.5, 8, 8, 11, 208
TRACE = 250  # frames of the inner-mean trace kept per condition
FIGURE = Path(__file__).resolve().parent / "figures" / "phase2_tuning.png"


def duration(tf):
    return DROP + max(2.0, 2.0 / tf)


class Grating:
    """A 1D sinusoid texture on one plane per axis ('x': stripes vertical, moving along columns)."""

    def __init__(self, stage, tex_dir, lam, contrast, travel_px):
        from PIL import Image

        T = lam * COL_PX
        n = math.ceil((SIZE + 2 * MARGIN_PX + travel_px) / T) + 1
        self.base = 128 * (1 + contrast * np.sin(2 * np.pi * (np.arange(n * T) + 0.5) / T))
        self.delta = 0.0  # DC offset set by calibrate_mean so the rendered mean is 128
        path = tex_dir / f"grating_l{lam}_c{contrast:g}.png"
        Image.fromarray(self.pixels()).save(path)
        self.tex = tp.TextureLoader().load(str(path), tp.ColorSpace.SRGB)
        self.tex.anisotropy = 8
        mat = tp.MeshBasicMaterial()
        mat.map = self.tex
        self.wpp = stage.world_per_px
        self.width = n * T * self.wpp
        self.meshes = {}
        for axis in ("x", "y"):
            m = tp.Mesh(tp.PlaneGeometry(self.width, 4.0), mat)
            if axis == "y":
                m.rotation.z = math.pi / 2  # texture u along world +Y
            stage.scene.add(m)
            self.meshes[axis] = m
        self.park()

    def pixels(self):
        row = np.round(self.base + self.delta).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(np.repeat(row[None], 4, 0)[..., None], 3, 2))

    def park(self):
        for m in self.meshes.values():
            m.position.set(1000.0, 1000.0, -1.0)

    def place(self, direction, s_px):
        """Plane edge on the frame border (plus margin) at s = 0, then s_px along the drift."""
        sign = 1.0 if direction in ("right", "up") else -1.0
        c = sign * ((SIZE / 2 + MARGIN_PX) * self.wpp - self.width / 2 + s_px * self.wpp)
        if direction in ("right", "left"):
            self.meshes["x"].position.set(c, 0.0, -1.0)
        else:
            self.meshes["y"].position.set(0.0, c, -1.0)


def contrast_probe(color, direction, T):
    """(mean, fundamental contrast, Michelson) of the luma profile across the stripes, central band."""
    n = max(1, BAND // T) * T  # whole periods, so the mean and the fundamental do not leak
    a = (SIZE - n) // 2
    band = bgra_luma_u8(color[a:a + n, a:a + n]).float()
    prof = band.mean(0) if direction in ("right", "left") else band.mean(1)
    x = torch.arange(n, device=prof.device, dtype=torch.float32) + 0.5
    z = torch.complex(prof * torch.cos(2 * math.pi * x / T), -prof * torch.sin(2 * math.pi * x / T)).mean()
    m = prof.mean()
    return torch.stack([m, 2 * z.abs() / m, (prof.max() - prof.min()) / (prof.max() + prof.min())])


class Video(VideoOut):
    """Horizontal: the eye render | the 721 receptor columns | the T4/T5 field (x2 brightness)."""

    def __init__(self, path, lattice, fps=100):
        import subprocess

        import imageio_ffmpeg
        from PIL import Image, ImageDraw

        self.Image, self.ImageDraw = Image, ImageDraw
        rc = lattice.pixel_rc(SIZE).float().cuda()
        yy, xx = torch.meshgrid(torch.arange(SIZE, device="cuda"), torch.arange(SIZE, device="cuda"), indexing="ij")
        pix = torch.stack([yy.flatten(), xx.flatten()], 1).float()
        d, i = zip(*(torch.cdist(c, rc).min(1) for c in pix.split(16384)))
        self.index = torch.cat(i).view(SIZE, SIZE)
        self.inside = (torch.cat(d) <= 8.5).view(SIZE, SIZE)
        self.W, self.H = 3 * SIZE + 2 * self.GAP + 1, self.TEXT + SIZE + 1
        self.W, self.H = self.W + self.W % 2, self.H + self.H % 2
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.W}x{self.H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def compose(self, color, receptors, field, line1, line2):
        img = torch.zeros(self.H, self.W, 3, dtype=torch.uint8, device="cuda")
        y = self.TEXT
        img[y:y + SIZE, :SIZE] = color[..., [2, 1, 0]]
        g = (receptors.clamp(0, 1) * 255).to(torch.uint8)[self.index] * self.inside
        img[y:y + SIZE, SIZE + self.GAP:2 * SIZE + self.GAP] = g[..., None]
        img[y:y + SIZE, 2 * (SIZE + self.GAP):3 * SIZE + 2 * self.GAP] = \
            self.field_rgb(2 * field.float())[self.index] * self.inside[..., None]
        im = self.Image.fromarray(img.cpu().numpy())
        dr = self.ImageDraw.Draw(im)
        dr.text((6, 4), line1, fill=(255, 255, 255))
        dr.text((6, 22), line2, fill=(255, 255, 255))
        return im

    def write_image(self, im):
        self.proc.stdin.write(im.tobytes())


def calibrate_mean(ctx, g, lam):
    """Shift the texture's DC so the rendered luma mean over the central band is 128. The 8-bit linear
    store quantises the darks coarsely, which pulls the mean down (126.8 at c 0.5 uncorrected)."""
    means = []
    for _ in range(4):
        g.place("right", 0.0)
        ctx["r"].reset_temporal_history()
        for _ in range(WARM):
            ctx["render"]()
        means.append(float(contrast_probe(ctx["eye"].color, "right", lam * COL_PX)[0]))
        if abs(128 - means[-1]) < 0.15:
            break
        g.delta += 128 - means[-1]
        g.tex.update_data(g.pixels())
    g.park()
    return dict(delta=g.delta, means=means)


def inner_mask(lat):
    u, v = lat.u, lat.v
    return (torch.maximum(torch.maximum(u.abs(), v.abs()), (u + v).abs()) <= INNER_RADIUS).cuda()


def run_condition(ctx, grating, lam, tf, direction, contrast, record=None):
    """One drifting grating. Returns dict of per-condition numbers (host) and the mid-window frame."""
    r, eye, lobe, motion, inner = ctx["r"], ctx["eye"], ctx["lobe"], ctx["motion"], ctx["inner"]
    T = lam * COL_PX
    speed = tf * T  # px/s
    n = int(round(duration(tf) / DT))
    k0 = int(round(DROP / DT))
    grating.place(direction, 0.0)
    r.reset_temporal_history()
    for _ in range(WARM):
        ctx["render"]()
    x0 = eye.receptors()
    lobe.fade_in(x0, DT)
    d = torch.tensor(DIR_VEC[direction], device="cuda")
    raw_sum = torch.zeros(8, 721, device="cuda")
    field_sum = torch.zeros(2, 721, device="cuda")
    acc_frames, trace, probe = [], [], []
    still = None
    for k in range(n):
        grating.place(direction, speed * (k + 1) * DT)
        ctx["render"]()
        x = eye.receptors()
        v = lobe.step(x, DT)
        raw = motion.raw(v)
        f = motion(v)
        if k < TRACE:
            trace.append(raw[:, inner].mean(1))
        if k >= k0:
            raw_sum += raw
            field_sum += f
            fi = f[:, inner]
            acc_frames.append(((d[:, None] * fi).sum(0) > math.cos(math.pi / 4) * fi.norm(dim=0)).float().mean())
            probe.append(contrast_probe(eye.color, direction, T))
        if k == (k0 + n) // 2:
            still = eye.color[..., [2, 1, 0]].cpu().numpy()
        if record is not None:
            record(k, eye.color, x, f)
    m = n - k0
    raw_mean, field_mean = raw_sum / m, field_sum / m
    fm = field_mean[:, inner]
    acc_avg = ((d[:, None] * fm).sum(0) > math.cos(math.pi / 4) * fm.norm(dim=0)).float().mean()
    grating.park()
    tr = torch.zeros(TRACE, 8, device="cuda")
    tr[:len(trace)] = torch.stack(trace)
    return dict(amp=raw_mean[:, inner].mean(1).cpu().numpy(), acc_avg=float(acc_avg),
                acc_frame=float(torch.stack(acc_frames).mean()), probe=torch.stack(probe).mean(0).cpu().numpy(),
                field=field_mean.half().cpu().numpy(), trace=tr.cpu().numpy(), frames=n), still


def pref_amp(amp):
    """amp (..., 4 dirs, 8 types) -> preferred (..., 8) and null (..., 8)."""
    pref = np.stack([amp[..., DIRS.index(PREF[i % 4]), i] for i in range(8)], -1)
    null = np.stack([amp[..., DIRS.index(OPPOSITE[PREF[i % 4]]), i] for i in range(8)], -1)
    return pref, null


def run(args):
    from PIL import Image, ImageDraw

    t_start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    args.renders.mkdir(parents=True, exist_ok=True)
    lambdas = [int(x) for x in args.lambdas.split(",")]
    tfs = [float(x) for x in args.tfs.split(",")]
    dirs = args.dirs.split(",")
    contrasts = [float(x) for x in args.contrasts.split(",")] if args.contrasts else []

    stage = Stage(SimpleNamespace(size=SIZE, fov=FOV, flush=1, fps=1 / DT))
    r = stage.r
    stage.hide_region()
    stage.set_bg(55 / 255)  # linear code 55 renders 128

    def render():
        stage.t += DT
        r.sim_time = stage.t
        r.render(stage.scene, stage.cam)

    travel = max(tf * duration(tf) for tf in TFS)  # periods
    gratings = {}
    for lam in lambdas:
        gratings[(lam, MAIN_C)] = Grating(stage, args.out, lam, MAIN_C, travel * lam * COL_PX)
    for c in contrasts:
        if (MID_LAMBDA, c) not in gratings:
            gratings[(MID_LAMBDA, c)] = Grating(stage, args.out, MID_LAMBDA, c, travel * MID_LAMBDA * COL_PX)
    for _ in range(5):
        render()
    lat = HexLattice()
    eye = EyeView(r, tp.PerspectiveCamera(FOV, 1.0, 0.1, 10.0), SIZE, lattice=lat)
    render()
    eye.arm()
    for _ in range(10):
        render()
    lobe = OpticLobe(device="cuda")
    ctx = dict(r=r, stage=stage, eye=eye, lobe=lobe, motion=MotionField(lobe), inner=inner_mask(lat), render=render)
    calib = {f"l{lam}_c{c:g}": calibrate_mean(ctx, g, lam) for (lam, c), g in gratings.items()}
    print("mean calibration", calib)
    print(f"inner columns {int(ctx['inner'].sum())}, setup {time.perf_counter() - t_start:.1f} s")

    run_video = Video(args.renders / "phase2_gratings_all_conditions.mp4", lat) if args.video else None
    selected = {(2, 1.0, "right", MAIN_C), (4, 4.0, "up", MAIN_C), (8, 16.0, "left", MAIN_C), (8, 0.25, "down", MAIN_C)}
    any_selected = bool(contrasts) or any(s[0] in lambdas for s in selected)
    sel_video = Video(args.renders / "phase2_gratings_selected.mp4", lat) if args.video and any_selected else None

    def recorder(lam, tf, direction, c):
        if run_video is None:
            return None
        sel = (lam, tf, direction, c) in selected

        def rec(k, color, x, f):
            if k >= 60 and not (sel and k < 200):
                return
            speed = tf * lam
            im = run_video.compose(color, x, f, f"grating {lam} columns ({lam * COL_PX} px), TF {tf:g} Hz, "
                                   f"{speed:g} columns/s, drift {direction}, contrast {c:g}, t {k * DT:4.2f} s",
                                   "left: eye view (secondary, flush 1)   middle: 721 receptor columns   "
                                   "right: T4/T5 field, hue = direction (red right), brightness = 2 x size")
            if k < 60:
                run_video.write_image(im)
            if sel:
                sel_video.write_image(im)
        return rec

    L, F, D = len(lambdas), len(tfs), len(dirs)
    res = dict(amp=np.zeros((L, F, D, 8)), acc_avg=np.zeros((L, F, D)), acc_frame=np.zeros((L, F, D)),
               probe=np.zeros((L, F, D, 3)), field=np.zeros((L, F, D, 2, 721), np.float16),
               trace=np.zeros((L, F, D, TRACE, 8), np.float32))
    stills = {}
    for li, lam in enumerate(lambdas):
        for fi, tf in enumerate(tfs):
            for di, dname in enumerate(dirs):
                out, still = run_condition(ctx, gratings[(lam, MAIN_C)], lam, tf, dname, MAIN_C,
                                           recorder(lam, tf, dname, MAIN_C))
                for key in res:
                    res[key][li, fi, di] = out[key]
                if dname == "right" and tf == 1.0 or (lam, tf, dname) == (4, 1.0, "up"):
                    stills[f"l{lam}_tf1_c0.5_{dname}"] = still
                print(f"lambda {lam} TF {tf:5.2f} {dname:5s} acc {out['acc_avg']:.2f}/{out['acc_frame']:.2f} "
                      f"contrast {out['probe'][1]:.3f} mean {out['probe'][0]:.1f} "
                      f"amp {np.round(out['amp'], 3).tolist()}", flush=True)

    best_tf = None
    if contrasts and MID_LAMBDA in lambdas and len(dirs) == 4:
        pref, _ = pref_amp(res["amp"][lambdas.index(MID_LAMBDA)])  # (F, 8)
        best_tf = tfs[int(np.argmax(pref.mean(1)))]
    elif contrasts:
        best_tf = tfs[0]
    selected.add((MID_LAMBDA, best_tf, "right", 0.05))
    C = len(contrasts)
    resc = dict(amp=np.zeros((C, D, 8)), acc_avg=np.zeros((C, D)), acc_frame=np.zeros((C, D)),
                probe=np.zeros((C, D, 3)),
                field=np.zeros((C, D, 2, 721), np.float16), trace=np.zeros((C, D, TRACE, 8), np.float32))
    for ci, c in enumerate(contrasts):
        for di, dname in enumerate(dirs):
            out, still = run_condition(ctx, gratings[(MID_LAMBDA, c)], MID_LAMBDA, best_tf, dname, c,
                                       recorder(MID_LAMBDA, best_tf, dname, c))
            for key in resc:
                resc[key][ci, di] = out[key]
            if dname == "right":
                stills[f"l{MID_LAMBDA}_tf{best_tf:g}_c{c:g}_right"] = still
            print(f"contrast {c:4.2f} TF {best_tf:g} {dname:5s} acc {out['acc_avg']:.2f}/{out['acc_frame']:.2f} "
                  f"measured {out['probe'][1]:.3f} michelson {out['probe'][2]:.3f} mean {out['probe'][0]:.1f}",
                  flush=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    for v in (run_video, sel_video):
        if v is not None:
            v.close()
    eye.close()

    for name, img in stills.items():
        Image.fromarray(img).save(args.renders / f"still_{name}.png")
    tiles = [(f"l{lam}_tf1_c0.5_right", f"{lam} columns, TF 1 Hz, c 0.5, right") for lam in lambdas]
    if best_tf is not None:
        tiles += [(f"l{MID_LAMBDA}_tf{best_tf:g}_c{c:g}_right", f"4 columns, TF {best_tf:g} Hz, c {c:g}, right")
                  for c in contrasts if c in (0.05, 0.2, 0.8)]
    tiles = [t for t in tiles if t[0] in stills]
    if tiles:
        cols = 3
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * (SIZE + 4), rows * (SIZE + 22)), (0, 0, 0))
        dr = ImageDraw.Draw(sheet)
        for i, (key, label) in enumerate(tiles):
            x, y = (i % cols) * (SIZE + 4), (i // cols) * (SIZE + 22)
            sheet.paste(Image.fromarray(stills[key]), (x, y + 18))
            dr.text((x + 4, y + 3), label, fill=(255, 255, 255))
        sheet.save(args.renders / "phase2_gratings_contact_sheet.png")

    np.savez(args.out / "gratings.npz", lambdas=np.array(lambdas), tfs=np.array(tfs), dirs=np.array(dirs),
             types=np.array(MOTION_TYPES), contrasts=np.array(contrasts), best_tf=np.array(best_tf or np.nan),
             main_contrast=MAIN_C, mid_lambda=MID_LAMBDA, dt=DT, drop_s=DROP, inner_radius=INNER_RADIUS,
             probe_fields=np.array(["mean_luma", "fundamental_contrast", "michelson"]),
             **{k: v for k, v in res.items()}, **{"c_" + k: v for k, v in resc.items()})
    json.dump(dict(wall_s=wall, settings={k: str(v) for k, v in stage.applied.items()}, view="EyeView secondary 403 px",
                   flush=1, best_tf=best_tf, mean_calibration=calib), open(args.out / "run.json", "w"), indent=1)
    print(f"done in {wall:.1f} s")


def peak(xs, ys):
    """Discrete argmax and a log2-parabola peak through the max and its neighbours."""
    i = int(np.argmax(ys))
    lx = np.log2(xs)
    if 0 < i < len(xs) - 1:
        a, b, _ = np.polyfit(lx[i - 1:i + 2], ys[i - 1:i + 2], 2)
        return float(xs[i]), float(2 ** (-b / (2 * a))) if a < 0 else float(xs[i])
    return float(xs[i]), float(xs[i])


def load(args):
    """gratings.npz plus any --extra runs (more wavelengths, same TFs), sorted by wavelength."""
    z = dict(np.load(args.out / "gratings.npz"))
    keys = ("amp", "acc_avg", "acc_frame", "probe")
    for path in args.extra:
        if not Path(path).exists():
            continue
        e = np.load(path)
        assert np.allclose(e["tfs"], z["tfs"])
        z["lambdas"] = np.concatenate([z["lambdas"], e["lambdas"]])
        for k in keys:
            z[k] = np.concatenate([z[k], e[k]])
    order = np.argsort(z["lambdas"])
    for k in ("lambdas",) + keys:
        z[k] = z[k][order]
    return z


def figure(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = load(args)
    lambdas, tfs, contrasts = z["lambdas"], z["tfs"], z["contrasts"]
    best_tf = float(z["best_tf"])
    pref, null = pref_amp(z["amp"])  # (L, F, 8)
    t4, t5 = pref[..., :4].mean(-1), pref[..., 4:].mean(-1)
    n4, n5 = null[..., :4].mean(-1), null[..., 4:].mean(-1)
    acc_avg, acc_frame = z["acc_avg"].mean(-1), z["acc_frame"].mean(-1)  # mean over the 4 directions
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

    report = dict(best_tf_mid_lambda=best_tf, per_lambda={}, contrast={})
    for li, lam in enumerate(lambdas):
        both, ds = t4[li] + t5[li], t4[li] + t5[li] - n4[li] - n5[li]
        report["per_lambda"][int(lam)] = dict(
            T4_pref=np.round(t4[li], 4).tolist(), T5_pref=np.round(t5[li], 4).tolist(),
            T4_null=np.round(n4[li], 4).tolist(), T5_null=np.round(n5[li], 4).tolist(),
            peak_tf_T4=peak(tfs, t4[li]), peak_tf_T5=peak(tfs, t5[li]), peak_tf_T4_plus_T5=peak(tfs, both),
            peak_speed_T4_plus_T5_cols_s=[p * float(lam) for p in peak(tfs, both)],
            peak_tf_pref_minus_null=peak(tfs, ds), peak_speed_pref_minus_null=[p * float(lam) for p in peak(tfs, ds)],
            acc_avg=np.round(acc_avg[li], 3).tolist(), acc_frame=np.round(acc_frame[li], 3).tolist(),
            acc_avg_by_dir=np.round(z["acc_avg"][li], 3).tolist(),
            measured_contrast=np.round(z["probe"][li, ..., 1].mean(-1), 4).tolist(),
            measured_michelson=np.round(z["probe"][li, ..., 2].mean(-1), 4).tolist(),
            mean_luma=np.round(z["probe"][li, ..., 0].mean(-1), 2).tolist(),
            per_type_pref=np.round(pref[li], 4).tolist(), per_type_null=np.round(null[li], 4).tolist())
    use = [int(x) for x in lambdas if x >= 3]  # 2 columns is the lattice Nyquist period: direction is ambiguous
    for key in ("peak_tf_T4_plus_T5", "peak_tf_pref_minus_null"):
        tf_pk = np.log2([report["per_lambda"][x][key][1] for x in use])
        report[f"octave_spread_{key}"] = dict(lambdas=use, tf=float(np.ptp(tf_pk)),
                                              speed=float(np.ptp(tf_pk + np.log2(use))))
    if len(contrasts):
        cp, cn = pref_amp(z["c_amp"])  # (C, 8)
        report["contrast"] = dict(
            contrasts=contrasts.tolist(), tf=best_tf, measured=np.round(z["c_probe"][..., 1].mean(-1), 4).tolist(),
            michelson=np.round(z["c_probe"][..., 2].mean(-1), 4).tolist(),
            T4_pref=np.round(cp[:, :4].mean(1), 4).tolist(), T5_pref=np.round(cp[:, 4:].mean(1), 4).tolist(),
            T4_null=np.round(cn[:, :4].mean(1), 4).tolist(), T5_null=np.round(cn[:, 4:].mean(1), 4).tolist(),
            acc_avg=np.round(z["c_acc_avg"].mean(-1), 3).tolist(),
            acc_frame=np.round(z["c_acc_frame"].mean(-1), 3).tolist(),
            acc_avg_by_dir=np.round(z["c_acc_avg"], 3).tolist(), per_type_pref=np.round(cp, 4).tolist())
    json.dump(report, open(args.out / "report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_lambda"}, indent=1))

    fig = plt.figure(figsize=(20, 12))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.4, 1), hspace=0.42)
    top = outer[0].subgridspec(1, 3, wspace=0.25)
    bottom = outer[1].subgridspec(1, 4, wspace=0.3)

    def lam_lines(ax, xs_of, y4, y5, ylabel):
        for li, lam in enumerate(lambdas):
            xs = xs_of(lam)
            ax.plot(xs, y4[li], color=colors[li], lw=2, marker="o", ms=5, label=f"T4, {lam} columns")
            ax.plot(xs, y5[li], color=colors[li], lw=2, ls="--", marker="s", ms=5, label=f"T5, {lam} columns")
        ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.grid(alpha=0.25)

    for col, (xlab, xs_of) in enumerate((("temporal frequency (Hz)", lambda lam: tfs),
                                         ("speed (columns/s)", lambda lam: tfs * lam))):
        ax = fig.add_subplot(top[0, col])
        lam_lines(ax, xs_of, t4, t5, "preferred-direction amplitude (relu rate - rest, a.u.)")
        ax.set_xlabel(xlab)
        ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), columnspacing=0.8)
        ax.set_title("vs " + xlab.split(" (")[0] + ", contrast 0.5, mean of the 4 subtypes", fontsize=10)
        lines = ["peak of T4 + T5 (log-parabola)"]
        for lam in lambdas:
            p = report["per_lambda"][int(lam)]["peak_tf_T4_plus_T5"][1]
            lines.append(f"{lam} columns: {p:.2f} Hz = {p * lam:.1f} columns/s")
        ax.text(0.02, 0.98, chr(10).join(lines), transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax = fig.add_subplot(top[0, 2])
    if len(contrasts):
        meas = z["c_probe"][..., 1].mean(-1)
        ax.plot(meas, cp[:, :4].mean(1), color=colors[0], lw=2, marker="o", label="T4 preferred")
        ax.plot(meas, cp[:, 4:].mean(1), color=colors[0], lw=2, ls="--", marker="s", label="T5 preferred")
        ax.plot(meas, cn[:, :4].mean(1), color="#888888", lw=1.2, marker="o", ms=4, label="T4 null")
        ax.plot(meas, cn[:, 4:].mean(1), color="#888888", lw=1.2, ls="--", marker="s", ms=4, label="T5 null")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("measured contrast (fundamental / mean of the rendered luma)")
        ax.set_ylabel("amplitude (relu rate - rest, a.u.)")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"vs contrast, {MID_LAMBDA} columns at TF {best_tf:g} Hz (nominal "
                     + ", ".join(f"{c:g}" for c in contrasts) + ")", fontsize=10)

    ax = fig.add_subplot(bottom[0, 0])
    for li, lam in enumerate(lambdas):
        ax.plot(tfs, acc_avg[li], color=colors[li], lw=2, marker="o", label=f"{lam} columns")
        ax.plot(tfs, acc_frame[li], color=colors[li], lw=1.2, ls=":", marker=".")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("temporal frequency (Hz)")
    ax.set_ylabel("fraction of inner columns within 45 deg")
    ax.set_title("MotionField direction accuracy, mean of 4 directions\nsolid: time-averaged vector, dotted: per frame",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3, loc="lower center")

    ax = fig.add_subplot(bottom[0, 1])
    lam_lines(ax, lambda lam: tfs, t4 - n4, t5 - n5, "preferred - null amplitude")
    ax.set_xlabel("temporal frequency (Hz)")
    ax.set_title("direction-selective part: preferred - null\n(the static-pattern offset cancels)", fontsize=10)
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(bottom[0, 2])
    li = list(lambdas).index(MID_LAMBDA) if MID_LAMBDA in lambdas else 0
    for i, name in enumerate(MOTION_TYPES):
        ax.plot(tfs, pref[li, :, i], color=colors[i % 4], lw=1.6, ls="-" if i < 4 else "--",
                marker="o" if i < 4 else "s", ms=4, label=f"{name} ({PREF[i % 4]})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("temporal frequency (Hz)")
    ax.set_ylabel("preferred amplitude")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_title(f"per subtype, {lambdas[li]} columns, contrast 0.5", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(bottom[0, 3])
    for li, lam in enumerate(lambdas):
        ax.plot(tfs, z["probe"][li, ..., 1].mean(-1), color=colors[li], lw=2, marker="o", label=f"{lam} columns")
    ax.axhline(MAIN_C, color="#999999", lw=0.8, ls="--", label="nominal 0.5")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("temporal frequency (Hz)")
    ax.set_ylabel("measured contrast of the rendered frames")
    ax.set_title("render check: fundamental contrast\nper condition (TAA at flush 1)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.suptitle("flyeye Phase 2 (b): T4/T5 tuning to threepp-rendered drifting sinusoidal gratings; 403 px 90 deg "
                 "secondary view, 100 Hz, flush 1, inner columns (hex radius <= 11)", fontsize=12)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=100)
    print("wrote", args.figure)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(r"C:\dev\_flyeye\phase2"))
    ap.add_argument("--renders", type=Path, default=Path(r"C:\dev\_flyeye\renders\phase2"))
    ap.add_argument("--figure", type=Path, default=FIGURE)
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDAS))
    ap.add_argument("--tfs", default=",".join(f"{x:g}" for x in TFS))
    ap.add_argument("--dirs", default=",".join(DIRS))
    ap.add_argument("--contrasts", default=",".join(f"{x:g}" for x in CONTRASTS))
    ap.add_argument("--extra", nargs="*", default=[r"C:\dev\_flyeye\phase2\extra\gratings.npz"],
                    help="more gratings.npz runs (other wavelengths, same TFs) merged into the figure")
    ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.figure_only:
        run(args)
    figure(args)


if __name__ == "__main__":
    main()
