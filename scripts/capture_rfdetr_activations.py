"""Capture RF-DETR-Nano intermediate activations for strict C++ validation.
Seeded deterministic input (raw floats, no normalization — the C++ side uploads
the same bytes). Saves to rfdetr_nano_ref.bin in the 'YOLO' format.

Validation checkpoints (normal NCHW / token layout, i.e. not windowed):
    input          [3, 384, 384]
    proj.P4        projector output  [256, 24, 24]   (validates patch-embed + 12
                                                       windowed ViT layers + taps)
    pred_logits    [300, 90]         final class logits
    pred_boxes     [300, 4]          final boxes (cx,cy,w,h, normalized)
"""
import struct
from pathlib import Path

import torch
from rfdetr import RFDETRNano


def main():
    out_path = Path(__file__).with_name('rfdetr_nano_ref.bin')
    det = RFDETRNano()
    net = det.model.model
    net.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 3, 384, 384)

    acts = {}
    hooks = []

    backbone0 = net.backbone[0]
    enc = backbone0.encoder.encoder  # WindowedDinov2WithRegistersBackbone
    # patch-embed conv output [1,384,24,24]
    pe = enc.embeddings.patch_embeddings.projection
    hooks.append(pe.register_forward_hook(
        lambda _m, _i, out: acts.__setitem__('patch', out.detach().cpu().float().contiguous())))
    # embeddings output (windowed token layout [W*W, 1+T, C])
    hooks.append(enc.embeddings.register_forward_hook(
        lambda _m, _i, out: acts.__setitem__('embed',
            (out[0] if isinstance(out, (list, tuple)) else out).detach().cpu().float().contiguous())))
    # a couple of ViT layer outputs (windowed layout) for finer debugging
    for li in (0, 3):
        def mk(idx):
            return lambda _m, _i, out: acts.__setitem__(
                f'vit{idx}', (out[0] if isinstance(out, (list, tuple)) else out).detach().cpu().float().contiguous())
        hooks.append(enc.encoder.layer[li].register_forward_hook(mk(li)))

    if hasattr(backbone0, 'projector'):
        # projector INPUT = the de-windowed tap features (list of [1,384,24,24]).
        def proj_pre(_m, inp):
            feats = inp[0] if isinstance(inp, tuple) else inp
            feats = feats if isinstance(feats, (list, tuple)) else [feats]
            for k, t in enumerate(feats):
                acts[f'tap.{k}'] = t.detach().cpu().float().contiguous()
        hooks.append(backbone0.projector.register_forward_pre_hook(proj_pre))
        def proj_hook(_m, _i, out):
            feats = out if isinstance(out, (list, tuple)) else [out]
            for k, t in enumerate(feats):
                acts[f'proj.{k}'] = t.detach().cpu().float().contiguous()
        hooks.append(backbone0.projector.register_forward_hook(proj_hook))

    with torch.no_grad():
        out = net(x)

    for h in hooks:
        h.remove()

    if isinstance(out, dict):
        if 'pred_logits' in out:
            acts['pred_logits'] = out['pred_logits'].detach().cpu().float().contiguous()
        if 'pred_boxes' in out:
            acts['pred_boxes'] = out['pred_boxes'].detach().cpu().float().contiguous()

    def strip_batch(t):
        return t[0] if t.dim() >= 2 and t.shape[0] == 1 else t

    acts['input'] = x
    for k in list(acts.keys()):
        acts[k] = strip_batch(acts[k]).contiguous()

    n = 0
    with open(out_path, 'wb') as f:
        f.write(b'YOLO')
        f.write(struct.pack('<II', 1, 0))
        for name, tensor in acts.items():
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
            print(f"  {name:<14s} {tuple(arr.shape)}  mean {arr.mean():+.4f}  std {arr.std():.4f}")
        f.seek(8)
        f.write(struct.pack('<I', n))
    print(f"\nWrote {n} activations -> {out_path.name}")


if __name__ == '__main__':
    main()
