"""
Benchmark RT-DETR-L inference speed using ultralytics (PyTorch).

Runs on the same bus.jpg image used by the C++ benchmark. Measures
warm-up + 5 timed runs, prints per-run and average latency.

Usage:
    python scripts/bench_rtdetr_pytorch.py [--device cpu|cuda] [--image data/bus.jpg]
"""

import argparse
import time
from pathlib import Path

import torch
from ultralytics import RTDETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image', default='data/bus.jpg')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--conf', type=float, default=0.3)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Image:  {args.image}")
    print(f"PyTorch version: {torch.__version__}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = RTDETR('rtdetr-l.pt')

    # Warm-up runs
    print(f"\nWarm-up ({args.warmup} runs)...")
    for i in range(args.warmup):
        results = model.predict(args.image, conf=args.conf, verbose=False)

    # Timed runs
    print(f"\nTimed runs ({args.runs}):")
    times = []
    for i in range(args.runs):
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        results = model.predict(args.image, conf=args.conf, verbose=False)

        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)

        # Print detections on first run
        if i == 0:
            r = results[0]
            for j, (cls, conf, box) in enumerate(zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)):
                name = r.names[int(cls)]
                x1, y1, x2, y2 = box.tolist()
                print(f"    [{j}] {name:<20s} conf={conf:.3f}  box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        print(f"  Run {i+1}: {elapsed:.1f} ms")

    avg = sum(times) / len(times)
    mn = min(times)
    print(f"\nResults ({args.device}):")
    print(f"  Average: {avg:.1f} ms")
    print(f"  Best:    {mn:.1f} ms")
    print(f"  FPS:     {1000.0/avg:.1f}")


if __name__ == '__main__':
    main()
