"""
Capture RT-DETR-L intermediate activations for strict C++ test validation.

Runs the ultralytics RT-DETR-L model on a seeded deterministic input and
saves (input + every model[i] output) to a binary file in the same "YOLO"
format used for weights. The C++ side loads this file and compares the
corresponding GPU activations element-wise.

Usage:
    pip install ultralytics
    python scripts/capture_rtdetr_activations.py
    # -> rtdetr_l_ref.bin next to this script  (~ 60 MB fp32)

The saved names mirror the module index:
    "input"      shape [3, 640, 640]  (batch stripped)
    "model.0"    HGStem output
    "model.1"    HGBlock (48->128)
    ...
    "model.10"   P5 projection  (256 ch)

Note: ultralytics RTDETR forwards through an nn.Sequential named `model`.
We hook every top-level submodule in that Sequential.
"""

import struct
from pathlib import Path

import torch
from ultralytics import RTDETR


def main():
    out_path = Path(__file__).with_name('rtdetr_l_ref.bin')

    det = RTDETR('rtdetr-l.pt')
    pt = det.model
    pt.eval()

    # Deterministic input: seeded normal noise in [−3, 3]-ish.
    # Same as a "normalized image" in range. We feed raw floats (no
    # /255 normalization) so the C++ side just uploads the same bytes.
    torch.manual_seed(42)
    x = torch.randn(1, 3, 640, 640)

    activations = {}

    # Find the inner Sequential that holds model.0 .. model.N
    inner = pt.model if hasattr(pt, 'model') else pt
    # Ultralytics typically exposes it as .model (DetectionModel.model is nn.Sequential)
    seq = inner

    hooks = []
    for i, m in enumerate(seq):
        def make_hook(idx):
            name = f"model.{idx}"
            def h(_mod, _inp, out):
                # Some modules return tuples (e.g., transformer); take first.
                t = out[0] if isinstance(out, (list, tuple)) else out
                activations[name] = t.detach().cpu().float().contiguous()
            return h
        hooks.append(m.register_forward_hook(make_hook(i)))

    # Also hook the decoder's input_proj submodules so we can validate
    # the C++ inputProj_() in isolation. input_proj is an nn.ModuleList
    # of nn.Sequential(Conv2d, BatchNorm2d) on the RTDETRDecoder.
    decoder = seq[-1]
    if hasattr(decoder, 'input_proj'):
        for i, sub in enumerate(decoder.input_proj):
            def make_ip_hook(idx):
                name = f"decoder.input_proj.{idx}"
                def h(_m, _i, out):
                    t = out[0] if isinstance(out, (list, tuple)) else out
                    activations[name] = t.detach().cpu().float().contiguous()
                return h
            hooks.append(sub.register_forward_hook(make_ip_hook(i)))

    # enc_output is the first Linear on the concatenated memory tokens.
    # Its INPUT is the memory tensor [B, sum_HW, C], so a pre-hook captures it.
    if hasattr(decoder, 'enc_output'):
        def mem_pre_hook(_m, inp):
            t = inp[0] if isinstance(inp, (list, tuple)) else inp
            activations['decoder.memory'] = t.detach().cpu().float().contiguous()
        hooks.append(decoder.enc_output.register_forward_pre_hook(mem_pre_hook))
        # enc_output output (Linear -> LayerNorm).
        def enc_out_hook(_m, _i, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            activations['decoder.enc_output'] = t.detach().cpu().float().contiguous()
        hooks.append(decoder.enc_output.register_forward_hook(enc_out_hook))

    with torch.no_grad():
        _ = pt(x)

    for h in hooks:
        h.remove()

    # Strip batch dim for the saved tensors (B=1 everywhere).
    def strip_batch(t):
        return t[0] if t.dim() >= 3 and t.shape[0] == 1 else t

    activations['input'] = strip_batch(x)
    for k in list(activations.keys()):
        if k != 'input':
            activations[k] = strip_batch(activations[k])

    n = 0
    with open(out_path, 'wb') as f:
        f.write(b'YOLO')
        f.write(struct.pack('<II', 1, 0))  # placeholder count

        for name, tensor in activations.items():
            arr = tensor.numpy()
            raw = arr.tobytes()
            nb = name.encode('utf-8')
            f.write(struct.pack('<I', len(nb)))
            f.write(nb)
            f.write(struct.pack('<I', arr.ndim))
            f.write(struct.pack(f'<{arr.ndim}I', *arr.shape))
            f.write(struct.pack('<I', len(raw)))
            f.write(raw)
            n += 1
            print(f"  {name:<12s} shape {tuple(arr.shape)}"
                  f"  mean {arr.mean():+.4f}  std {arr.std():.4f}"
                  f"  min {arr.min():+.4f}  max {arr.max():+.4f}")

        f.seek(8)
        f.write(struct.pack('<I', n))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {n} activations -> {out_path.name}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
