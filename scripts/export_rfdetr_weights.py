"""Export RF-DETR-Nano weights to the 'YOLO' binary format used by the threepp
Vulkan inference examples (same format as export_rtdetr_weights.py).

    python scripts/export_rfdetr_weights.py   # -> rfdetr-nano.weights

LWDETR state_dict is dumped verbatim (fp32). The C++ side folds nothing here —
RF-DETR's norms are LayerNorm (folded at runtime in the shaders), not Conv+BN,
so weights are uploaded as-is.
"""
import struct
from pathlib import Path

import torch
from rfdetr import RFDETRNano


def main():
    out_path = Path(__file__).with_name('rfdetr-nano.weights')
    det = RFDETRNano()
    net = det.model.model  # LWDETR nn.Module
    net.eval()
    sd = net.state_dict()

    n = 0
    with open(out_path, 'wb') as f:
        f.write(b'YOLO')
        f.write(struct.pack('<II', 1, 0))  # version, placeholder count
        for name, tensor in sd.items():
            arr = tensor.detach().cpu().float().contiguous().numpy()
            raw = arr.tobytes()
            nb = name.encode('utf-8')
            f.write(struct.pack('<I', len(nb)))
            f.write(nb)
            f.write(struct.pack('<I', arr.ndim))
            f.write(struct.pack(f'<{arr.ndim}I', *arr.shape))
            f.write(struct.pack('<I', len(raw)))
            f.write(raw)
            n += 1
        f.seek(8)
        f.write(struct.pack('<I', n))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {n} tensors -> {out_path.name}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
