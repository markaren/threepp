"""Train Gaussian splats on a make_dataset.py dataset with gsplat, write a .ply threepp loads.

gsplat (https://github.com/nerfstudio-project/gsplat, Apache-2.0) does the
differentiable rasterization and the densify/prune schedule (DefaultStrategy).
The rest is the textbook 3DGS loop: L1 + SSIM against the camera output, SH degree
raised one step per 1000 iterations, start points from the dataset's depth maps.
Depth supervision is on by default: the dataset's depth is exact, so it costs
nothing to use, and it pulls splats onto the surfaces that bake_surface and the
sensors will later read.

The .ply is in the INRIA 3DGS layout (opacity as a logit, log scales, quaternion w
first, f_rest channel-major) and in the frame every other 3DGS .ply comes in, the
COLMAP one with +Y down: training runs in the scene's own frame (Y up) and the
export turns the splats half a turn about X. The editor and the gaussian_splats
example give an imported .ply the opposite half turn, so it stands upright there,
in the scene's own place.

Runs in its own venv that sees the system torch, because gsplat compiles CUDA
kernels on first use:

    py -3.14 -m venv --system-site-packages <venv>
    <venv>/Scripts/python -m pip install gsplat

The first run compiles for a few minutes and must see MSVC and a CUDA toolkit
(a VS developer shell, TORCH_CUDA_ARCH_LIST=12.0 for an RTX 50). Later runs load
the built module from torch's cache and need neither. README.md has the Windows
snags (torch's CUDA major vs the toolkit's, a GCC flag gsplat 1.5.3 hands MSVC).

    <venv>/Scripts/python python/examples/scene_splats/train.py --steps 7000

It is also the trainer of the mapping harness (../splat_mapping/): a dataset may name any number of splits
("train" is fitted, every other split, "test", "on_path", "off_path:<name>", is scored and never trained on),
and may carry a mask per frame (mask_path, 255 = use the pixel). With masks every loss term is taken over the
masked pixels only, the depth term only where the depth is valid too, and the background colour is a constant
(nothing unmasked shows it); the scores are over the masked pixels, with the masked share beside them.
--seed drives torch, numpy and the order of the views; gsplat's kernels are not bit-deterministic, so two runs
with one seed still differ a little. Every run writes the harness's manifest.json beside its splats.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as _cpp
from PIL import Image

# torch may be built against a newer CUDA major than the installed toolkit (cu130
# vs 12.8 here). The kernels need only the driver, so let the JIT build proceed,
# and let the built module find the toolkit's runtime DLL (cudart64_12.dll), which
# Windows does not look for on PATH.
if hasattr(_cpp, "_check_cuda_version"):
    _cpp._check_cuda_version = lambda *a, **k: None
if os.name == "nt":
    # The CUDA installer sets CUDA_PATH_V<major>_<minor>, and not always CUDA_PATH.
    _vers = sorted((k for k in os.environ if k.startswith("CUDA_PATH_V")),
                   key=lambda k: [int(x) for x in k[len("CUDA_PATH_V"):].split("_")], reverse=True)
    for _var in ["CUDA_HOME", "CUDA_PATH"] + _vers:
        _bin = os.path.join(os.environ.get(_var, ""), "bin")
        if os.environ.get(_var) and os.path.isdir(_bin):
            os.environ.setdefault("CUDA_HOME", os.environ[_var])
            os.add_dll_directory(_bin)
            break
# torch runs ninja on every JIT load, cached or not; the venv has it next to python.
os.environ["PATH"] = os.path.dirname(os.path.abspath(__import__("sys").executable)) + os.pathsep + os.environ["PATH"]


def _preload_built_gsplat():
    """Import gsplat's compiled module straight from torch's extension cache.

    torch's JIT loader asks for MSVC (`where cl`) and ninja on every load, even
    when nothing needs compiling, so without this each run would have to start in
    a Visual Studio developer shell. gsplat imports `gsplat.csrc` first and only
    JIT-builds when that fails. Delete the cache folder after upgrading gsplat.
    """
    import importlib.util
    import sys
    path = os.path.join(_cpp._get_build_directory("gsplat_cuda", verbose=False),
                        "gsplat_cuda" + (".pyd" if os.name == "nt" else ".so"))
    if not os.path.isfile(path):
        return False
    spec = importlib.util.spec_from_file_location("gsplat_cuda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["gsplat.csrc"] = mod
    return True


_preload_built_gsplat()

from gsplat import rasterization  # noqa: E402
from gsplat.strategy import DefaultStrategy  # noqa: E402
from gsplat.strategy import ops as gs_ops  # noqa: E402

DEFAULT_DATA = os.path.join(os.path.expanduser("~"), ".cache", "threepp", "scene_splats", "rover")
SH_C0 = 0.28209479177387814

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DEFAULT_DATA)
ap.add_argument("--out", default=None, help="default <data>/splats_<steps>")
ap.add_argument("--steps", type=int, default=7000)
ap.add_argument("--sh-degree", type=int, default=3)
ap.add_argument("--ssim-weight", type=float, default=0.2)
ap.add_argument("--depth-weight", type=float, default=0.2,
                help="L1 on expected depth, in units of the scene scale; 0 = off")
ap.add_argument("--refine-stop", type=int, default=None, help="default steps // 2")
ap.add_argument("--max-splats", type=int, default=0,
                help="hold the cloud at this many splats: at each refinement past it, the lowest-opacity splats "
                     "are removed (0 = no cap). Densification and pruning run on to the end of the refine window")
ap.add_argument("--cap-after-reset", type=int, default=300,
                help="steps after an opacity reset during which the cap is not enforced (every opacity was just "
                     "set alike, so the lowest ones say nothing yet)")
ap.add_argument("--train-every", type=int, default=1, help="train on every Nth training frame only (a control)")
ap.add_argument("--eval-data", default=None,
                help="score on this dataset's non-train splits instead of --data's (same intrinsics; a control "
                     "trained on other data, scored on the same held-out views)")
ap.add_argument("--sheet-views", type=int, default=4, help="views per split on the result sheets")
ap.add_argument("--seed", type=int, default=0, help="torch, numpy and the view order")
args = ap.parse_args()
out_dir = args.out or os.path.join(args.data, f"splats_{args.steps}")
os.makedirs(out_dir, exist_ok=True)

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splat_mapping"))
from manifest import write_manifest  # noqa: E402

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = torch.device("cuda")


def read_points_ply(path):
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                             ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return (np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32),
            np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.float32) / 255.0)


# --- the dataset -------------------------------------------------------------
with open(os.path.join(args.data, "transforms.json"), encoding="utf-8") as f:
    tf = json.load(f)
W, H = tf["w"], tf["h"]
def intrinsics_of(fr, base):
    """A frame's pinhole matrix: its own fl_x, fl_y, cx, cy when it carries them (a dataset that mixes cameras),
    else the dataset's. Every frame shares the dataset's w and h."""
    return [[fr.get("fl_x", base["fl_x"]), 0.0, fr.get("cx", base["cx"])],
            [0.0, fr.get("fl_y", base["fl_y"]), fr.get("cy", base["cy"])], [0.0, 0.0, 1.0]]
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

splits = {}
for fr in tf["frames"]:
    splits.setdefault(fr["split"], []).append(fr)
splits["train"] = splits["train"][::args.train_every]
if args.eval_data:
    with open(os.path.join(args.eval_data, "transforms.json"), encoding="utf-8") as f:
        tf_eval = json.load(f)
    assert (tf_eval["w"], tf_eval["h"], tf_eval["fl_x"], tf_eval["fl_y"]) == (W, H, tf["fl_x"], tf["fl_y"])
    splits = {"train": splits["train"]}
    for fr in tf_eval["frames"]:
        if fr["split"] != "train":
            splits.setdefault(fr["split"], []).append(dict(fr, _root=args.eval_data))
masked = any("mask_path" in fr for fr in tf["frames"])
depth_unit = tf.get("depth_unit_scale_factor", 0.001)


def load_split(frames):
    """GPU tensors of a split: images uint8 [N,H,W,3], depth fp16 metres (0 = none), masks bool [N,H,W]
    (all true without masks), world -> OpenCV-camera view matrices; and the camera centres."""
    imgs, depths, masks, viewmats, centres, Ks = [], [], [], [], [], []
    for fr in frames:
        Ks.append(intrinsics_of(fr, tf))
        root = fr.get("_root", args.data)
        imgs.append(np.asarray(Image.open(os.path.join(root, fr["file_path"])).convert("RGB")))
        d = np.asarray(Image.open(os.path.join(root, fr["depth_file_path"]))).astype(np.float32)
        depths.append(d * depth_unit)
        masks.append(np.asarray(Image.open(os.path.join(root, fr["mask_path"]))) > 127 if "mask_path" in fr
                     else np.ones(d.shape, bool))
        c2w = np.array(fr["transform_matrix"], np.float64) @ GL_TO_CV
        viewmats.append(np.linalg.inv(c2w))
        centres.append(c2w[:3, 3])
    return (torch.from_numpy(np.stack(imgs)).to(dev),
            torch.from_numpy(np.stack(depths)).to(dev, torch.float16),
            torch.from_numpy(np.stack(masks)).to(dev),
            torch.from_numpy(np.stack(viewmats)).to(dev, torch.float32),
            np.stack(centres), torch.tensor(Ks, dtype=torch.float32, device=dev))


tr_img, tr_depth, tr_mask, tr_view, tr_centres, tr_K = load_split(splits["train"])
eval_splits = [s for s in splits if s != "train"]
scene_scale = float(np.linalg.norm(tr_centres - tr_centres.mean(0), axis=1).max() * 1.1)

if masked:
    # Nothing the loss sees shows the background: any constant does.
    background = torch.zeros(3, device=dev)
else:
    # The background the camera actually delivered (after the tone map), from the
    # pixels no surface covers.
    bg_mask = tr_depth == 0
    background = (tr_img.float()[bg_mask].median(0).values / 255.0) if bg_mask.any() \
        else torch.tensor(tf["threepp"]["background"], device=dev)
print(f"{len(tr_img)} train views; scored: " + ", ".join(f"{s} {len(splits[s])}" for s in eval_splits)
      + f"; {W}x{H}, scene scale {scene_scale:.2f}, masked {masked}, "
      f"masked share of the training pixels {1.0 - tr_mask.float().mean().item():.3f}, "
      f"background {[round(v, 3) for v in background.tolist()]}")

# --- the start splats --------------------------------------------------------
pts, rgb = read_points_ply(os.path.join(args.data, tf["ply_file_path"]))
from scipy.spatial import cKDTree  # noqa: E402

d, _ = cKDTree(pts).query(pts, k=4)
dist = np.sqrt((d[:, 1:] ** 2).mean(1)).clip(1e-4)
n0 = len(pts)
n_sh = (args.sh_degree + 1) ** 2
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(torch.from_numpy(pts).to(dev)),
    "scales": torch.nn.Parameter(torch.log(torch.from_numpy(dist).to(dev, torch.float32))[:, None].repeat(1, 3)),
    "quats": torch.nn.Parameter(F.normalize(torch.rand(n0, 4, device=dev), dim=-1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((n0,), 0.1, device=dev))),
    "sh0": torch.nn.Parameter(((torch.from_numpy(rgb).to(dev) - 0.5) / SH_C0)[:, None, :]),
    "shN": torch.nn.Parameter(torch.zeros(n0, n_sh - 1, 3, device=dev)),
})
lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
       "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
              for k, lr in lrs.items()}
means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.steps))

strategy = DefaultStrategy(refine_stop_iter=args.refine_stop or args.steps // 2,
                           reset_every=3000, verbose=False)
strategy.check_sanity(params, optimizers)
state = strategy.initialize_state(scene_scale=scene_scale)


def ssim(a, b, mask=None):
    """Mean SSIM of two [1,3,H,W] images in [0,1], 11x11 Gaussian window; over the pixels of mask [H,W] if
    one is given."""
    g = torch.exp(-((torch.arange(11, device=dev) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    mu_a = F.conv2d(a, win, padding=5, groups=3)
    mu_b = F.conv2d(b, win, padding=5, groups=3)
    s_aa = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    if mask is None:
        return m.mean()
    return m.mean(1)[0][mask].mean()


def render(viewmat, sh_degree, with_depth, K):
    out, alpha, info = rasterization(
        params["means"], params["quats"], torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), torch.cat([params["sh0"], params["shN"]], 1),
        viewmat[None], K[None], W, H, sh_degree=sh_degree, packed=False,
        backgrounds=background[None], render_mode="RGB+ED" if with_depth else "RGB")
    return out[0], alpha[0], info


# --- training ----------------------------------------------------------------
log = []
capped_at = None
cap_events = []
t0 = time.perf_counter()
order = np.random.permutation(len(tr_img))
for step in range(args.steps):
    i = int(order[step % len(order)])
    if step % len(order) == len(order) - 1:
        order = np.random.permutation(len(tr_img))
    sh_now = min(step // 1000, args.sh_degree)
    use_depth = args.depth_weight > 0
    out, alpha, info = render(tr_view[i], sh_now, use_depth, tr_K[i])
    strategy.step_pre_backward(params, optimizers, state, step, info)

    gt = tr_img[i].float() / 255.0
    pred = out[..., :3]
    mk = tr_mask[i] if masked else None
    l1 = (pred - gt).abs()[mk].mean() if masked else (pred - gt).abs().mean()
    s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], mk)
    loss = (1 - args.ssim_weight) * l1 + args.ssim_weight * (1 - s)
    if use_depth:
        gd = tr_depth[i].float()
        valid = (gd > 0) & mk if masked else gd > 0
        loss = loss + args.depth_weight * (out[..., 3] - gd)[valid].abs().mean() / scene_scale
    loss.backward()

    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    means_sched.step()
    strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
    if (args.max_splats and len(params["means"]) > args.max_splats
            and (step < strategy.reset_every or step % strategy.reset_every >= args.cap_after_reset)):
        # the cap: the lowest-opacity splats go, through gsplat's own remove() (parameters, Adam state and the
        # strategy's running state together); DefaultStrategy goes on densifying and pruning to refine_stop
        with torch.no_grad():
            excess = len(params["means"]) - args.max_splats
            drop = torch.zeros(len(params["means"]), dtype=torch.bool, device=dev)
            drop[torch.topk(params["opacities"].flatten(), excess, largest=False).indices] = True
        gs_ops.remove(params, optimizers, state, drop)
        cap_events.append((step, excess))
        if capped_at is None:
            capped_at = step
            print(f"step {step}: the cloud passed the cap {args.max_splats}; from here the lowest-opacity splats are removed")

    if step % 250 == 0 or step == args.steps - 1:
        with torch.no_grad():
            psnr = -10 * math.log10((((pred - gt) ** 2)[mk] if masked else (pred - gt) ** 2).mean().item() + 1e-12)
        el = time.perf_counter() - t0
        n = len(params["means"])
        print(f"step {step:6d}  loss {loss.item():.4f}  train psnr {psnr:5.2f}  splats {n:8d}  "
              f"{(step + 1) / el:6.1f} it/s  {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        log.append({"step": step, "loss": loss.item(), "psnr": psnr, "splats": n, "seconds": el})
train_s = time.perf_counter() - t0

# --- evaluation, every split but "train" -------------------------------------
# PSNR and SSIM over the unmasked pixels of each view (every pixel without masks); the depth error is the median
# |expected depth - true depth| over the unmasked pixels with depth. Per split: the mean, the worst view, the
# masked share. A sheet per split: truth | gsplat render | |difference| x4, masked pixels darkened.
del tr_img, tr_depth, tr_mask
torch.cuda.empty_cache()
results = {}
for split in eval_splits:
    ev_img, ev_depth, ev_mask, ev_view, _, ev_K = load_split(splits[split])
    psnrs, ssims, depth_err, shares = [], [], [], []
    sheet_rows = []
    show = set(np.linspace(0, len(ev_img) - 1, min(args.sheet_views, len(ev_img))).round().astype(int).tolist())
    with torch.no_grad():
        for j in range(len(ev_img)):
            out, _, _ = render(ev_view[j], args.sh_degree, True, ev_K[j])
            gt = ev_img[j].float() / 255.0
            pred = out[..., :3].clamp(0, 1)
            mk = ev_mask[j]
            shares.append(1.0 - mk.float().mean().item())
            psnrs.append(-10 * math.log10(((pred - gt) ** 2)[mk].mean().item() + 1e-12) if mk.any() else float("nan"))
            ssims.append(ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], mk if masked else None).item())
            gd = ev_depth[j].float()
            v = (gd > 0) & mk
            depth_err.append((out[..., 3] - gd)[v].abs().median().item() if v.any() else float("nan"))
            Image.fromarray((pred.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(
                os.path.join(out_dir, f"{split.replace(':', '_')}_{j:03d}_gsplat.png"))
            if j in show:
                diff = ((pred - gt).abs() * 4).clamp(0, 1)
                dim = torch.where(mk[..., None], 1.0, 0.25)
                sheet_rows.append(torch.cat([gt * dim, pred * dim, diff * dim], 1).cpu().numpy())
    sheet = (np.concatenate(sheet_rows, 0) * 255 + 0.5).astype(np.uint8)
    name = "eval_sheet.jpg" if split == "test" else f"eval_sheet_{split.replace(':', '_')}.jpg"
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(os.path.join(out_dir, name), quality=90)
    p = np.array(psnrs)
    results[split] = {"views": len(psnrs), "psnr_mean": float(np.nanmean(p)), "psnr_min": float(np.nanmin(p)),
                      "worst_view": int(np.nanargmin(p)), "ssim_mean": float(np.nanmean(ssims)),
                      "masked_share_mean": float(np.mean(shares)),
                      "depth_median_abs_err_m": float(np.nanmedian(depth_err)),
                      "psnr": [round(x, 3) for x in psnrs], "ssim": [round(x, 4) for x in ssims]}
    print(f"  {split:18s} {len(psnrs):3d} views: PSNR {results[split]['psnr_mean']:6.2f} dB (worst "
          f"{results[split]['psnr_min']:.2f}), SSIM {results[split]['ssim_mean']:.4f}, masked "
          f"{results[split]['masked_share_mean'] * 100:.1f} %, depth median |err| "
          f"{results[split]['depth_median_abs_err_m'] * 1000:.1f} mm")
    del ev_img, ev_depth, ev_mask, ev_view
    torch.cuda.empty_cache()

# --- export: INRIA 3DGS .ply, in the COLMAP frame (+Y down) ------------------
# Half a turn about X, R = diag(1, -1, -1), exactly: the means take R; each
# Gaussian's rotation takes R in front, which for a w-first quaternion is
# (w, x, y, z) -> (-x, w, -z, y); and the colour seen along d must be the colour
# the turned splat shows along R d, so every SH basis function that is odd in
# (y, z) changes sign (INRIA order, l = 1..3).
SH_SIGN_HALF_TURN_X = torch.tensor([-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1],
                                   dtype=torch.float32, device=dev)
with torch.no_grad():
    means = params["means"].cpu().numpy() * np.array([1.0, -1.0, -1.0], np.float32)
    n = len(means)
    q = params["quats"]
    quats = torch.stack([-q[:, 1], q[:, 0], -q[:, 3], q[:, 2]], 1).cpu().numpy()
    shN = params["shN"] * SH_SIGN_HALF_TURN_X[: n_sh - 1, None]
    f_dc = params["sh0"][:, 0, :].cpu().numpy()
    f_rest = shN.transpose(1, 2).reshape(n, -1).cpu().numpy()   # channel-major
    cols = [means, np.zeros_like(means), f_dc, f_rest,
            params["opacities"][:, None].cpu().numpy(),
            params["scales"].cpu().numpy(), quats]
names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
         + [f"f_rest_{k}" for k in range(f_rest.shape[1])]
         + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
data = np.concatenate(cols, 1).astype("<f4")
assert data.shape[1] == len(names)
ply_path = os.path.join(out_dir, "splats.ply")
with open(ply_path, "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\n"
             f"element vertex {n}\n"
             + "".join(f"property float {nm}\n" for nm in names)
             + "end_header\n").encode("ascii"))
    f.write(data.tobytes())

summary = {
    "steps": args.steps, "train_seconds": round(train_s, 1), "splats": n, "seed": args.seed,
    "start_points": n0, "sh_degree": args.sh_degree, "depth_weight": args.depth_weight,
    "ssim_weight": args.ssim_weight, "refine_stop": args.refine_stop or args.steps // 2,
    "max_splats": args.max_splats, "capped_at_step": capped_at, "cap_events": len(cap_events),
    "cap_removed": int(sum(e for _, e in cap_events)), "train_every": args.train_every, "eval_data": args.eval_data, "scene_scale": scene_scale,
    "masked": masked, "train_views": len(splits["train"]),
    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    "splits": results, "background": background.tolist(), "log": log,
}
if "test" in results:                                     # the pilot's keys
    summary.update(test_psnr_mean=results["test"]["psnr_mean"], test_psnr_min=results["test"]["psnr_min"],
                   test_ssim_mean=results["test"]["ssim_mean"],
                   test_depth_median_abs_err_m=results["test"]["depth_median_abs_err_m"])
with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1)
write_manifest(out_dir, seed=args.seed, name=f"train_seed{args.seed}",
               extra={"data": os.path.basename(os.path.abspath(args.data)), "out": os.path.basename(out_dir),
                      "train_seconds": round(train_s, 1), "splats": n})
print(f"trained {args.steps} steps in {train_s:.0f}s -> {n} splats, peak {summary['peak_gpu_gib']:.1f} GiB")
print(f"wrote {ply_path} ({os.path.getsize(ply_path) / 2**20:.1f} MiB)")
