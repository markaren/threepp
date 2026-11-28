"""PyTorch RF-DETR baseline for vulkan_rfdetr_inference, tuned for speed and timed the same way.

The Vulkan port's --bench times infer() on an image decoded once: the RGBA copy into the frame
staging, the upload, the resize + normalize on the GPU, the forward pass, the readback and the
postprocess. This script times PyTorch along the same path: the RGBA image copied into pinned
staging, uploaded, resized + normalized on the GPU, the forward pass, rfdetr's own PostProcess on
the GPU, and the kept detections copied back.

Configs, all on rfdetr's own optimized path (RFDETR.inference(), rebuilt step by step so the
module can be timed directly: deepcopy -> export() -> cast -> TorchScript trace -> freeze ->
rfdetr's CUDA graph wrapper):
    fp32-eager   export-mode eager forward, TF32 off (the package's default numerics)
    fp32-graph   trace + freeze + one CUDA graph, TF32 off (the Vulkan port's precision)
    tf32-graph   the same with TF32 matmuls and convolutions
    fp16-graph   half-precision weights and activations + CUDA graph
    bf16-graph   bfloat16 + CUDA graph

Per config: the forward pass's GPU time (CUDA events) and the end-to-end wall time (median, p10,
p90 after a 2 s warm-up, as --bench does), and the detections at 0.5 against fp32-eager's: the
confidence and, per detection matched by box IoU, the box and keypoint error in image pixels.

Usage:
  python bench_rfdetr_pytorch.py --variant nano|small|medium [--image bus.jpg] [--runs 100]
  python bench_rfdetr_pytorch.py --variant keypoint --checkpoint rf-detr-keypoint-preview-xlarge.pth
  [--configs fp32-graph,fp16-graph] [--accuracy-only]

Written against rfdetr 1.12.1 and torch 2.12.1+cu130 (it uses rfdetr's private trace and CUDA
graph helpers). Without Triton (torch.compile) or TensorRT, fp16-graph is the fastest PyTorch
config available.
"""
import argparse
import gc
import os
import statistics
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CONFIGS = {
    "fp32-eager": (torch.float32, False, False),
    "fp32-graph": (torch.float32, True, False),
    "tf32-graph": (torch.float32, True, True),
    "fp16-graph": (torch.float16, True, True),
    "bf16-graph": (torch.bfloat16, True, True),
}


def find_default_image() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("../cmake-build-relwithdebinfo/_deps/threepp_data-src/images/bus.jpg",
                "../data/images/bus.jpg", "bus.jpg"):
        p = os.path.normpath(os.path.join(here, rel))
        if os.path.exists(p):
            return p
    return "bus.jpg"


def load(variant: str, checkpoint: str | None):
    import rfdetr
    if variant == "keypoint":
        return rfdetr.RFDETRKeypointPreview(pretrain_weights=checkpoint, num_classes=1, device="cuda")
    classes = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}
    return classes[variant](device="cuda")  # downloads pretrained weights on first use


def build(det, dtype: torch.dtype, graph: bool, R: int, dev: torch.device):
    """RFDETR.inference(compile=graph, dtype=dtype, compile_backend="cudagraph"), keeping the module."""
    from rfdetr.detr import _CUDAGraphInferenceModel, _trace_for_prediction
    m = deepcopy(det.model.model).to(dev).eval()
    m.export()
    m = m.to(dtype=dtype)
    if not graph:
        return m
    dummy = torch.randn(1, 3, R, R, device=dev, dtype=dtype)
    traced = torch.jit.freeze(_trace_for_prediction(m, dummy))
    return _CUDAGraphInferenceModel(traced, dummy, dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("nano", "small", "medium", "keypoint"), default="nano")
    ap.add_argument("--checkpoint", help="rf-detr-keypoint-preview-xlarge.pth, for --variant keypoint")
    ap.add_argument("--image", default=find_default_image())
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--accuracy-only", action="store_true", help="skip the timing")
    args = ap.parse_args()
    if args.variant == "keypoint" and not args.checkpoint:
        ap.error("--variant keypoint needs --checkpoint")

    dev = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = True
    det = load(args.variant, args.checkpoint)
    R = int(det.model.resolution)
    post = det.model.postprocess
    mean = torch.tensor(det.means, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(det.stds, device=dev).view(1, 3, 1, 1)
    print(f"RF-DETR {args.variant}: {R}x{R}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}", flush=True)

    rgba = np.ascontiguousarray(np.asarray(Image.open(args.image).convert("RGBA")))
    H, W = rgba.shape[:2]
    pinned = torch.empty((H, W, 4), dtype=torch.uint8).pin_memory()
    pinned_np = pinned.numpy()
    gpu_img = torch.empty((H, W, 4), dtype=torch.uint8, device=dev)
    target = torch.tensor([[H, W]], device=dev)

    def preprocess(dtype):
        pinned_np[...] = rgba  # the host copy into the staging, as the Vulkan port's recordUpload does
        gpu_img.copy_(pinned, non_blocking=True)
        x = gpu_img[..., :3].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False)
        return ((x - mean) / std).to(dtype)

    def heads(out):
        d = {"pred_boxes": out[0].float(), "pred_logits": out[1].float()}
        if len(out) > 2:
            d["pred_keypoints"] = out[2].float()
        return d

    def postprocess(out):
        r = post(heads(out), target)[0]
        keep = r["scores"] > 0.5
        return {k: v[keep].cpu() for k, v in r.items() if isinstance(v, torch.Tensor) and v.shape[:1] == keep.shape}

    results, ref = [], None
    for name in args.configs.split(","):
        dtype, graph, tf32 = CONFIGS[name]
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        fwd = None
        try:
            fwd = build(det, dtype, graph, R, dev)  # outside inference_mode, as RFDETR.inference() does
            with torch.inference_mode():
                x = preprocess(dtype)
                dets = postprocess(fwd(x))
                if ref is None:
                    ref = dets
                # Against the first config: confidences, and per detection (matched by IoU) box + keypoints.
                n, same = len(dets["scores"]), len(dets["scores"]) == len(ref["scores"])
                dconf = ((dets["scores"].sort(descending=True).values - ref["scores"].sort(descending=True).values)
                         .abs().max().item()) if same and n else float("nan")
                dbox = dkp = 0.0
                for i in range(len(ref["scores"])):
                    if not n:
                        dbox = float("inf")
                        break
                    b, rb = dets["boxes"], ref["boxes"][i]
                    inter = ((torch.minimum(b[:, 2], rb[2]) - torch.maximum(b[:, 0], rb[0])).clamp(min=0) *
                             (torch.minimum(b[:, 3], rb[3]) - torch.maximum(b[:, 1], rb[1])).clamp(min=0))
                    union = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (rb[2] - rb[0]) * (rb[3] - rb[1]) - inter
                    j = int((inter / union).argmax())
                    dbox = max(dbox, (b[j] - rb).abs().max().item())
                    if "keypoints" in dets:
                        dkp = max(dkp, (dets["keypoints"][j][..., :2] - ref["keypoints"][i][..., :2]).abs().max().item())
                acc = (n, same, dconf, dbox, dkp)
                if args.accuracy_only:
                    results.append((name, None, None, None, None, acc))
                    continue

                for _ in range(20):
                    fwd(x)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                fw = []
                for _ in range(args.runs):
                    s.record()
                    fwd(x)
                    e.record()
                    e.synchronize()
                    fw.append(s.elapsed_time(e))

                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 2.0:  # hold the boost clock, as --bench does
                    postprocess(fwd(preprocess(dtype)))
                e2e = []
                for _ in range(args.runs):
                    t = time.perf_counter()
                    postprocess(fwd(preprocess(dtype)))
                    e2e.append((time.perf_counter() - t) * 1000.0)
            e2e.sort()
            q = lambda f: e2e[int(f * (len(e2e) - 1) + 0.5)]  # noqa: E731
            results.append((name, statistics.median(fw), statistics.median(e2e), q(0.1), q(0.9), acc))
            print(f"  {name:<11} forward {statistics.median(fw):7.2f} ms   end-to-end median "
                  f"{statistics.median(e2e):7.2f}  p10 {q(0.1):6.2f}  p90 {q(0.9):6.2f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<11} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        finally:
            fwd = None
            gc.collect()
            torch.cuda.empty_cache()

    first = args.configs.split(",")[0]
    print(f"\ndetections > 0.5 against {first}:")
    for name, _, _, _, _, (n, same, dconf, dbox, dkp) in results:
        print(f"  {name:<11} {n} found{'' if same else ' (count differs)'}"
              f"{'' if dconf != dconf else f', confidence within {dconf:.4f}'}, boxes within {dbox:.2f} px"
              f"{f', keypoints within {dkp:.2f} px' if args.variant == 'keypoint' else ''}")


if __name__ == "__main__":
    main()
