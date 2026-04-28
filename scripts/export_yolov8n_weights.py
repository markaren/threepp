#!/usr/bin/env python3
"""Export YOLOv8n pre-trained weights to a simple binary format for C++ inference.

Usage:
    pip install ultralytics
    python scripts/export_yolov8n_weights.py [output_path]

Output:
    yolov8n.bin  (~12 MB) — place next to the yolov8_inference executable,
    or pass its path as the second CLI argument.

Binary format:
    Header:  magic[4]="YOLO"  version:u32=1  num_tensors:u32
    Per tensor:
        name_len:u32  name:bytes  ndim:u32  shape[ndim]:u32[]
        data_bytes:u32  data:float32[]
"""
import struct
import sys
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed.  Run:  pip install ultralytics")
    sys.exit(1)

output_path = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.bin"

print("Loading YOLOv8n (downloads ~6 MB on first run)…")
model = YOLO("yolov8n.pt")
sd = model.model.state_dict()

# Filter to float tensors only (skip integer buffers like num_batches_tracked)
tensors = {k: v for k, v in sd.items() if v.dtype in (
    __import__('torch').float32, __import__('torch').float16, __import__('torch').bfloat16)}

print(f"Exporting {len(tensors)} tensors → {output_path}")

with open(output_path, "wb") as f:
    # Header
    f.write(b"YOLO")
    f.write(struct.pack("<II", 1, len(tensors)))

    for name, tensor in tensors.items():
        arr = tensor.cpu().float().numpy()
        name_bytes = name.encode("utf-8")
        raw = arr.tobytes()

        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", arr.ndim))
        if arr.ndim > 0:
            f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
        f.write(struct.pack("<I", len(raw)))
        f.write(raw)

print(f"Done.  File size: {__import__('os').path.getsize(output_path) / 1e6:.1f} MB")
print(f"\nTensor summary (first 10):")
for i, (name, tensor) in enumerate(list(tensors.items())[:10]):
    print(f"  {name:50s}  {list(tensor.shape)}")
if len(tensors) > 10:
    print(f"  … and {len(tensors)-10} more")

print(f"""
Next steps:
  1. Build:   cmake --build <build_dir> --target yolov8_inference
  2. Copy:    cp {output_path} <build_dir>/examples/misc/
  3. Run:     ./yolov8_inference /path/to/image.jpg {output_path}
""")
