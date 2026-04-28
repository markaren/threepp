"""
Export RT-DETR-L weights to the same binary format used by YoloV8n.

Format (little-endian):
  4 bytes  magic "YOLO"
  u32      version = 1
  u32      num_tensors
  per tensor:
    u32    name_len
    char[] name  (UTF-8, no null)
    u32    ndim
    u32[ndim] shape
    u32    data_byte_count
    float32[] raw data  (PyTorch NCHW layout)

Usage:
    pip install ultralytics
    python scripts/export_rtdetr_weights.py
    # → rtdetr_l.weights next to this script
"""

import struct
from pathlib import Path

from ultralytics import RTDETR


def main():
    out_path = Path(__file__).with_name('rtdetr_l.weights')
    model = RTDETR('rtdetr-l.pt')
    sd = model.model.state_dict()

    n = 0
    with open(out_path, 'wb') as f:
        f.write(b'YOLO')
        f.write(struct.pack('<II', 1, 0))  # placeholder count

        for name, tensor in sd.items():
            arr = tensor.detach().cpu().float().numpy()
            raw = arr.tobytes()
            nb = name.encode('utf-8')
            f.write(struct.pack('<I', len(nb)))
            f.write(nb)
            f.write(struct.pack('<I', arr.ndim))
            f.write(struct.pack(f'<{arr.ndim}I', *arr.shape))
            f.write(struct.pack('<I', len(raw)))
            f.write(raw)
            n += 1

        # Back-patch the tensor count
        f.seek(8)
        f.write(struct.pack('<I', n))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {n} tensors → {out_path.name}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
