"""PyTorch RF-DETR-Nano baseline for benchmarking against the Vulkan-compute port.

Times the *same* model (Roboflow `rfdetr` pretrained Nano, 384x384, fp32) on the
same GPU so the numbers are directly comparable to vulkan_rfdetr_inference.exe.

Reports two figures:
  * model.forward  — raw nn.Module forward only (apples-to-apples vs the Vulkan
                     GPU compute; both fp32, both 384x384).
  * predict()      — end-to-end incl. pre/post-processing (the Vulkan exe's
                     wall-clock 24.8 ms / 40.2 FPS also includes pre/post).

Usage:
  python bench_rfdetr_pytorch.py [--image bus.jpg] [--device cuda] \
                                 [--runs 50] [--warmup 10] [--half] [--vulkan-ms 24.8]
"""
import argparse
import time

import numpy as np
import torch


def find_default_image() -> str:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("../cmake-build-relwithdebinfo/_deps/threepp_data-src/images/bus.jpg",
                "../data/images/bus.jpg", "bus.jpg"):
        p = os.path.normpath(os.path.join(here, rel))
        if os.path.exists(p):
            return p
    return "bus.jpg"


def timed(fn, runs: int, warmup: int, cuda: bool) -> float:
    """Return mean ms/iter over `runs` (after `warmup`), syncing CUDA properly."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=find_default_image())
    ap.add_argument("--variant", choices=("nano", "small", "medium"), default="nano")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--half", action="store_true", help="also time an fp16 forward")
    ap.add_argument("--vulkan-ms", type=float, default=24.8,
                    help="measured Vulkan-port wall-clock ms, for the comparison line")
    args = ap.parse_args()

    dev = args.device
    cuda = dev.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        dev, cuda = "cpu", False
    if cuda:
        torch.backends.cudnn.benchmark = True  # autotune conv algos (fair-baseline protocol)

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    variants = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}
    print(f"Loading RF-DETR-{args.variant} (downloads pretrained weights on first run)...")
    det = variants[args.variant]()
    R = det.model_config.resolution
    module = det.model.model.eval().to(dev)

    if cuda:
        name = torch.cuda.get_device_name(0)
        print(f"Device: {dev}  ({name})   resolution: {R}x{R}")
    else:
        print(f"Device: cpu   resolution: {R}x{R}")

    # Full-res RGB (for the lean end-to-end path, which resizes — like the Vulkan
    # exe's infer()), plus a pre-normalized [1,3,R,R] tensor for forward-only timing.
    from PIL import Image
    rgb_full = np.asarray(Image.open(args.image).convert("RGB"))  # (H,W,3) uint8
    img_h, img_w = rgb_full.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    arr = (np.asarray(Image.fromarray(rgb_full).resize((R, R), Image.BILINEAR), np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(dev)
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)

    # Lean end-to-end: minimal pre/post (resize -> normalize -> forward -> sigmoid +
    # threshold + cxcywh->xyxy + scale -> results to CPU), matching what the Vulkan
    # infer() does. This is the FAIR baseline — supervision's predict() adds a large
    # convenience-API tax that no deployment would pay.
    def lean_e2e(fwd, dtype):
        pil = Image.fromarray(rgb_full).resize((R, R), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(pil, np.float32)).to(dev).permute(2, 0, 1).unsqueeze(0).div_(255.0)
        t = ((t - mean_t) / std_t).to(dtype)
        out = fwd(t)
        logits = (out["pred_logits"] if isinstance(out, dict) else out[0])[0].float()
        boxes = (out["pred_boxes"] if isinstance(out, dict) else out[1])[0].float()
        prob = logits.sigmoid()
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                            (cx + w / 2) * img_w, (cy + h / 2) * img_h], -1)
        idx = (prob > 0.5).nonzero(as_tuple=False)
        _ = (xyxy[idx[:, 0]].cpu(), prob[idx[:, 0], idx[:, 1]].cpu(), idx[:, 1].cpu())

    # Collected per config: (label, forward_ms, predict_e2e_ms, lean_e2e_ms).
    results: list[tuple[str, float | None, float | None, float | None]] = []

    print("\n=== timing ===")
    e2e_runs, e2e_warm = max(args.runs // 5, 3), max(args.warmup // 2, 2)

    def bench_predict() -> float | None:
        try:
            with torch.inference_mode():
                return timed(lambda: det.predict(args.image, threshold=0.5), e2e_runs, e2e_warm, cuda)
        except Exception as e:  # noqa: BLE001
            print(f"    predict() failed ({type(e).__name__}: {e})")
            return None

    # Pre/post overhead (resize+normalize+sigmoid+topk+decode) is model-agnostic, so
    # measure it once on the eager model (which returns the dict lean_e2e expects) and
    # compose lean-e2e = forward + overhead for the traced configs (whose exported
    # forward returns a different structure).
    lean_overhead = [None]  # boxed for closure assignment

    def bench_config(label: str, fwd, dtype: torch.dtype, measure_lean: bool) -> None:
        xo = x.to(dtype)
        with torch.inference_mode():
            mf = timed(lambda: fwd(xo), args.runs, args.warmup, cuda)
            if measure_lean:
                ml = timed(lambda: lean_e2e(fwd, dtype), e2e_runs, e2e_warm, cuda)
                lean_overhead[0] = ml - mf  # the pre+post cost
            else:
                ml = mf + lean_overhead[0] if lean_overhead[0] is not None else None
        mp = bench_predict()  # supervision predict() (convenience API)
        print(f"  {label:13s} forward {mf:6.2f} ({1000.0/mf:5.1f} FPS)   "
              f"lean-e2e {ml:6.2f} ({1000.0/ml:5.1f} FPS)" if ml else
              f"  {label:13s} forward {mf:6.2f} ({1000.0/mf:5.1f} FPS)")
        results.append((f"PyTorch {label.strip()}", mf, mp, ml))

    # (1) Eager fp32 — the package default; also fixes the model-agnostic pre/post cost.
    bench_config("eager fp32", module, torch.float32, measure_lean=True)

    # (2,3) optimize_for_inference: TorchScript trace, fp32 then fp16 (the documented
    # deployment path). Reset between configs with remove_optimized_model().
    def bench_optimized(dtype: torch.dtype, label: str) -> None:
        try:
            det.remove_optimized_model()
            det.optimize_for_inference(compile=True, batch_size=1, dtype=dtype)
            bench_config(label, det.model.inference_model, dtype, measure_lean=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:13s} failed ({type(e).__name__}: {e})")
        finally:
            try:
                det.remove_optimized_model()
            except Exception:  # noqa: BLE001
                pass

    bench_optimized(torch.float32, "TS fp32")
    if cuda:
        bench_optimized(torch.float16, "TS fp16")

    # (4) TensorRT — gold-standard NVIDIA baseline. Only if installed.
    try:
        import tensorrt  # noqa: F401
        print("  TensorRT: tensorrt is installed but no engine-build path is wired here yet.")
    except ImportError:
        print("  TensorRT: skipped (tensorrt not installed). For a reviewer-proof upper bound,\n"
              "           `pip install tensorrt`, export via rfdetr's ONNX path, build an engine,\n"
              "           and add it as config (5).")

    # Detections (sanity: should be bus + 4 people, matching the Vulkan port).
    try:
        det_out = det.predict(args.image, threshold=0.5)
        names = det_out.data.get("class_name") if hasattr(det_out, "data") else None
        print(f"\n=== detections (threshold 0.5) ===  {len(det_out)} found")
        order = np.argsort(-det_out.confidence)
        for i in order:
            nm = names[i] if names is not None else str(int(det_out.class_id[i]))
            x1, y1, x2, y2 = det_out.xyxy[i]
            print(f"  {str(nm):16s}  conf {det_out.confidence[i]:.3f}  "
                  f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    except Exception as e:  # noqa: BLE001
        print(f"detections failed ({type(e).__name__}: {e})")

    print(f"\n=== summary (RTX 4070, {R}x{R}, fp32 unless noted) ===")
    cell = lambda v: f"{v:6.1f}" if v else f"{'-':>6s}"  # noqa: E731
    fps = lambda v: f"{1000.0 / v:5.1f}" if v else f"{'-':>5s}"  # noqa: E731
    hdr = f"  {'config':22s} {'fwd ms':>7s} {'lean-e2e ms':>12s} {'lean FPS':>9s} {'predict() ms':>13s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'Vulkan (this work)':22s} {'-':>7s} {args.vulkan_ms:12.1f} {fps(args.vulkan_ms):>9s} {'-':>13s}")
    for label, mf, mp, ml in results:
        print(f"  {label:22s} {cell(mf):>7s} {cell(ml):>12s} {fps(ml):>9s} {cell(mp):>13s}")
    print("\n  lean-e2e = minimal resize+normalize / forward / sigmoid+topk+box-decode (the FAIR")
    print("  comparison; matches the Vulkan exe's infer()). predict() = supervision convenience")
    print("  API (large host tax; not a deployment path). fwd = nn.Module compute only.")


if __name__ == "__main__":
    main()
