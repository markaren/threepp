"""
Benchmark YOLOv8n inference time over 20 runs on a single image.
"""

import time
import statistics
import argparse
from pathlib import Path


def benchmark(image_path: str, runs: int = 20, warmup: int = 3, device: str = "cpu"):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but torch not compiled with CUDA — falling back to cpu.")
        device = "cpu"

    model = YOLO("yolov8n.pt")

    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n{'='*52}")
    print(f"  YOLOv8n Inference Benchmark")
    print(f"{'='*52}")
    print(f"  Image  : {img.name} ({img.stat().st_size / 1024:.1f} KB)")
    print(f"  Device : {device}")
    print(f"  Warmup : {warmup} runs")
    print(f"  Runs   : {runs}")
    print(f"{'='*52}\n")

    # Warmup
    print(f"Warming up ({warmup} runs)...", end=" ", flush=True)
    for _ in range(warmup):
        model(str(img), verbose=False, device=device)
    print("done.\n")

    # Benchmark
    times_ms = []
    for i in range(runs):
        t0 = time.perf_counter()
        results = model(str(img), verbose=False, device=device)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        times_ms.append(elapsed)

        boxes = results[0].boxes
        names = results[0].names
        n_det = len(boxes) if boxes is not None else 0
        print(f"  Run {i+1:>2}/{runs}  {elapsed:7.2f} ms   detections: {n_det}")
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = names.get(cls_id, str(cls_id))
                print(f"             ↳ {label:20s}  conf: {conf:.3f}")

    # Stats
    mean   = statistics.mean(times_ms)
    median = statistics.median(times_ms)
    stdev  = statistics.stdev(times_ms)
    mn     = min(times_ms)
    mx     = max(times_ms)

    print(f"\n{'='*52}")
    print(f"  Results ({runs} runs, ms)")
    print(f"{'─'*52}")
    print(f"  Mean   : {mean:8.2f} ms   ({1000/mean:6.1f} FPS)")
    print(f"  Median : {median:8.2f} ms   ({1000/median:6.1f} FPS)")
    print(f"  Std    : {stdev:8.2f} ms")
    print(f"  Min    : {mn:8.2f} ms")
    print(f"  Max    : {mx:8.2f} ms")
    print(f"{'='*52}\n")

    return times_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8n inference on a single image.")
    parser.add_argument("image", help="Path to the .jpg image")
    parser.add_argument("--runs",   type=int, default=50,    help="Number of timed runs (default: 20)")
    parser.add_argument("--warmup", type=int, default=3,     help="Warmup runs before timing (default: 3)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu | cuda | mps (default: cpu)")
    args = parser.parse_args()

    benchmark(args.image, runs=args.runs, warmup=args.warmup, device=args.device)
