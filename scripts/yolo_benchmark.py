"""Fair YOLOv8n PyTorch baseline for benchmarking against the Vulkan-compute port.

Mirrors scripts/bench_rfdetr_pytorch.py: the image is decoded ONCE (not per run),
inference runs on the GPU, and the model is benchmarked across optimization levels
that an actual deployment would use — eager, TorchScript, and TensorRT — via
ultralytics' own `export()`. Reports raw-net forward time plus end-to-end
`predict()` (letterbox + forward + NMS), the realistic deployment number.

The original quick benchmark (per-run path decode, eager `model(path)` on CPU) is
kept under --legacy; note it is NOT a fair baseline — it re-reads the JPEG from disk
every run and never exercises an optimized graph.

Usage:
  python yolo_benchmark.py <image> [--device cuda] [--imgsz 640] [--runs 50]
                           [--half] [--vulkan-ms 13.3] [--legacy]
"""
import argparse
import statistics
import time
from pathlib import Path


def timed(fn, runs: int, warmup: int, cuda: bool) -> float:
    import torch
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


def fair(image: str, runs: int, warmup: int, device: str, imgsz: int,
         half: bool, vulkan_ms: float | None) -> None:
    import torch
    from ultralytics import YOLO
    import cv2

    # Autotune cuDNN conv algorithms for the fixed input size — without this, eager
    # conv nets run on a slow default algorithm (a common benchmarking pitfall that
    # makes PyTorch look ~3-4x slower than it really is).
    torch.backends.cudnn.benchmark = True

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = device if cuda else "cpu"
    pdev = 0 if cuda else "cpu"
    if device.startswith("cuda") and not cuda:
        print("CUDA requested but unavailable; falling back to CPU.")
    name = torch.cuda.get_device_name(0) if cuda else "cpu"
    print(f"Device: {dev}  ({name})   imgsz: {imgsz}x{imgsz}")

    arr = cv2.imread(image)  # BGR uint8, decoded ONCE (outside the timed loop)
    if arr is None:
        raise FileNotFoundError(f"cannot read image: {image}")

    base = YOLO("yolov8n.pt")
    e2e_runs, e2e_warm = max(runs // 2, 5), max(warmup, 3)
    results: list[tuple[str, float | None, float | None]] = []

    # Raw-net forward only (compute floor; content irrelevant to timing).
    try:
        net = base.model.to(dev).eval()
        x = torch.rand(1, 3, imgsz, imgsz, device=dev)
        with torch.inference_mode():
            fwd_ms = timed(lambda: net(x), runs, warmup, cuda)
    except Exception as e:  # noqa: BLE001
        fwd_ms = None
        print(f"  (raw forward timing failed: {type(e).__name__}: {e})")

    def bench_predict(m) -> float | None:
        try:
            return timed(lambda: m.predict(arr, imgsz=imgsz, device=pdev, verbose=False),
                         e2e_runs, e2e_warm, cuda)
        except Exception as e:  # noqa: BLE001
            print(f"    predict() failed ({type(e).__name__}: {e})")
            return None

    print("\n=== timing ===")
    me = bench_predict(base)
    results.append(("PyTorch eager fp32", fwd_ms, me))
    print(f"  eager fp32     forward {fwd_ms if fwd_ms else float('nan'):6.2f}   "
          f"predict()-e2e {me if me else float('nan'):6.2f}")

    def bench_export(fmt: str, half_: bool, label: str) -> None:
        try:
            path = base.export(format=fmt, imgsz=imgsz, half=half_,
                               device=pdev, verbose=False)
            m = YOLO(path)
            me_ = bench_predict(m)
            results.append((label, None, me_))
            print(f"  {label:14s} predict()-e2e {me_ if me_ else float('nan'):6.2f} "
                  f"({1000.0 / me_ if me_ else float('nan'):5.1f} FPS)")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:14s} skipped ({type(e).__name__}: {e})")

    bench_export("torchscript", False, "TorchScript fp32")
    if cuda and half:
        bench_export("torchscript", True, "TorchScript fp16")
    # TensorRT — gold-standard NVIDIA baseline (needs `tensorrt` installed).
    if cuda:
        try:
            import tensorrt  # noqa: F401
            bench_export("engine", half, "TensorRT" + (" fp16" if half else " fp32"))
        except ImportError:
            print("  TensorRT       skipped (tensorrt not installed; "
                  "`pip install tensorrt` for the NVIDIA upper bound)")

    # Detections sanity (eager, preloaded array).
    try:
        r = base.predict(arr, imgsz=imgsz, device=pdev, verbose=False, conf=0.25)[0]
        print(f"\n=== detections (conf 0.25) ===  {len(r.boxes)} found")
        for b in r.boxes:
            print(f"  {r.names.get(int(b.cls[0]), int(b.cls[0])):16s}  conf {float(b.conf[0]):.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"detections failed ({type(e).__name__}: {e})")

    print("\n=== summary (RTX 4070, " + f"{imgsz}x{imgsz}) ===")
    fps = lambda v: f"{1000.0 / v:5.1f}" if v else f"{'-':>5s}"  # noqa: E731
    cell = lambda v: f"{v:7.2f}" if v else f"{'-':>7s}"  # noqa: E731
    hdr = f"  {'config':24s} {'fwd ms':>7s} {'predict()-e2e ms':>17s} {'e2e FPS':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    if vulkan_ms:
        print(f"  {'Vulkan (this work)':24s} {'-':>7s} {vulkan_ms:17.2f} {fps(vulkan_ms):>8s}")
    else:
        print("  Vulkan (this work)        pass --vulkan-ms <ms> to add the comparison row")
    for label, mf, mp in results:
        print(f"  {label:24s} {cell(mf):>7s} {cell(mp):>17s} {fps(mp):>8s}")
    print("\n  predict()-e2e = ultralytics letterbox + forward + NMS (deployment path),")
    print("  image preloaded. fwd = raw nn.Module forward only. Compare e2e-to-e2e.")


def legacy(image_path: str, runs: int, warmup: int, device: str) -> None:
    """Original benchmark: eager model(path) — re-decodes from disk each run. NOT fair."""
    from ultralytics import YOLO
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable — falling back to cpu.")
        device = "cpu"
    model = YOLO("yolov8n.pt")
    img = Path(image_path)
    print(f"[legacy] {img.name}  device={device}  (re-decodes image every run)")
    for _ in range(warmup):
        model(str(img), verbose=False, device=device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(str(img), verbose=False, device=device)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  mean {statistics.mean(times):.2f} ms  ({1000 / statistics.mean(times):.1f} FPS)  "
          f"min {min(times):.2f}  median {statistics.median(times):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fair YOLOv8n PyTorch baseline.")
    ap.add_argument("image", help="path to a .jpg image")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="also bench fp16 (TorchScript/TensorRT)")
    ap.add_argument("--vulkan-ms", type=float, default=None,
                    help="measured Vulkan-port e2e ms, for the comparison row")
    ap.add_argument("--legacy", action="store_true", help="run the old per-run-decode benchmark")
    args = ap.parse_args()

    if args.legacy:
        legacy(args.image, args.runs, args.warmup, "cpu" if args.device == "cpu" else args.device)
    else:
        fair(args.image, args.runs, args.warmup, args.device, args.imgsz, args.half, args.vulkan_ms)
