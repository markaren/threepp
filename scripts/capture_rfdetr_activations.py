"""Capture RF-DETR intermediate activations for strict C++ validation.
Seeded deterministic input (raw floats, no normalization — the C++ side uploads
the same bytes). Saves to rfdetr_<variant>_ref.bin in the 'YOLO' format.

    python scripts/capture_rfdetr_activations.py                   # -> rfdetr_nano_ref.bin
    python scripts/capture_rfdetr_activations.py --variant medium  # -> rfdetr_medium_ref.bin
    python scripts/capture_rfdetr_activations.py --variant keypoint --checkpoint rf-detr-keypoint-preview-xlarge.pth
                                                                   # -> rfdetr_keypoint_ref.bin
    ... --image bus.jpg                                            # -> rfdetr_<variant>_bus_ref.bin

Input resolution is the variant's native size (nano 384 / small 512 / medium 576,
keypoint 576 at patch 12); the backbone (dinov2_windowed_small, no register tokens)
and hook points are the same across variants — only the token count and decoder
depth change.

--image captures on a real image instead of noise: square bilinear resize (pixel
centres at +0.5, no antialiasing, the same sampling as the C++ preprocess), /255,
ImageNet normalization. The decoder's two-stage top-K is only stable on a real
image, so that is where the decoder and keypoint outputs can be compared.

Validation checkpoints (normal NCHW / token layout, i.e. not windowed), e.g. nano:
    input          [3, 384, 384]
    proj.P4        projector output  [256, 24, 24]   (validates patch-embed + 12
                                                       windowed ViT layers + taps)
    pred_logits    [300, 90]         final class logits
    pred_boxes     [300, 4]          final boxes (cx,cy,w,h, normalized)
Keypoint variant, in addition:
    projkp.0       second (keypoint cross-attention) projector output [256, 48, 48]
    dec.hs         decoder output after the final norm [100, 256]
    dec.ref        decoder reference boxes (cx,cy,w,h) [100, 4]
    dec.kp         last layer's keypoint hidden states [100, 17, 256]
    pred_keypoints the person slot of the keypoint head [100, 17, 8]: x, y
                   (normalized image coords), findable, visible, log_l11, l21,
                   log_l22, class-logit contribution
"""
import argparse
import struct
from pathlib import Path

import torch
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRKeypointPreview

# class + native input resolution (from rfdetr/config.py); must match the C++ RfDetrConfig.
VARIANTS = {'nano': (RFDETRNano, 384), 'small': (RFDETRSmall, 512), 'medium': (RFDETRMedium, 576),
            'keypoint': (RFDETRKeypointPreview, 576)}


def load_image(path, res):
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    t = torch.from_numpy(img).permute(2, 0, 1)[None]  # [1,3,H,W], 0..255
    t = torch.nn.functional.interpolate(t, size=(res, res), mode='bilinear', align_corners=False, antialias=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t / 255.0 - mean) / std


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', choices=list(VARIANTS), default='nano')
    ap.add_argument('--checkpoint', help='weights file for --variant keypoint')
    ap.add_argument('--image', help='capture on this image instead of seeded noise')
    args = ap.parse_args()

    cls, res = VARIANTS[args.variant]
    suffix = f'_{Path(args.image).stem}' if args.image else ''
    out_path = Path(__file__).with_name(f'rfdetr_{args.variant}{suffix}_ref.bin')
    if args.variant == 'keypoint':
        if not args.checkpoint:
            ap.error('--variant keypoint needs --checkpoint')
        det = cls(pretrain_weights=args.checkpoint, num_classes=1, device='cpu')  # plain fp32 reference
    else:
        det = cls()
    net = det.model.model
    net.eval()

    if args.image:
        x = load_image(args.image, res)
    else:
        torch.manual_seed(42)
        x = torch.randn(1, 3, res, res)

    acts = {}
    hooks = []

    def keep(t):
        return t.detach().cpu().float().contiguous()

    backbone0 = net.backbone[0]
    enc = backbone0.encoder.encoder  # WindowedDinov2WithRegistersBackbone
    # patch-embed conv output [1,384,G,G]
    pe = enc.embeddings.patch_embeddings.projection
    hooks.append(pe.register_forward_hook(lambda _m, _i, out: acts.__setitem__('patch', keep(out))))
    # embeddings output (windowed token layout [W*W, 1+T, C])
    hooks.append(enc.embeddings.register_forward_hook(
        lambda _m, _i, out: acts.__setitem__('embed', keep(out[0] if isinstance(out, (list, tuple)) else out))))
    # a couple of ViT layer outputs (windowed layout) for finer debugging
    for li in (0, 3):
        def mk(idx):
            return lambda _m, _i, out: acts.__setitem__(
                f'vit{idx}', keep(out[0] if isinstance(out, (list, tuple)) else out))
        hooks.append(enc.encoder.layer[li].register_forward_hook(mk(li)))

    def projector_hooks(module, prefix, taps):
        if taps:
            # projector INPUT = the de-windowed tap features (list of [1,384,G,G]).
            def pre(_m, inp):
                feats = inp[0] if isinstance(inp, tuple) else inp
                feats = feats if isinstance(feats, (list, tuple)) else [feats]
                for k, t in enumerate(feats):
                    acts[f'tap.{k}'] = keep(t)
            hooks.append(module.register_forward_pre_hook(pre))

        def post(_m, _i, out):
            feats = out if isinstance(out, (list, tuple)) else [out]
            for k, t in enumerate(feats):
                acts[f'{prefix}.{k}'] = keep(t)
        hooks.append(module.register_forward_hook(post))

    if hasattr(backbone0, 'projector'):
        projector_hooks(backbone0.projector, 'proj', taps=True)
    if getattr(backbone0, 'cross_attn_projector', None) is not None:
        projector_hooks(backbone0.cross_attn_projector, 'projkp', taps=False)

    keypoints = getattr(net, 'keypoint_embed', None) is not None
    if keypoints:
        def dec_hook(_m, _i, out):
            # (stack of normed outputs [L,B,N,C], references [1,B,N,4], keypoint states [L,B,N,K,C])
            acts['dec.hs'] = keep(out[0][-1])
            acts['dec.ref'] = keep(out[1][-1])
            acts['dec.kp'] = keep(out[2][-1])
        hooks.append(net.transformer.decoder.register_forward_hook(dec_hook))

    with torch.no_grad():
        out = net(x)

    for h in hooks:
        h.remove()

    if isinstance(out, dict):
        if 'pred_logits' in out:
            acts['pred_logits'] = keep(out['pred_logits'])
        if 'pred_boxes' in out:
            acts['pred_boxes'] = keep(out['pred_boxes'])
        if keypoints and 'pred_keypoints' in out:
            # padded [B, N, classes*maxK, 8] -> the slot of the class that has keypoints
            schema = list(net.num_keypoints_per_class)
            max_k = max(schema)
            cls_idx = next(i for i, k in enumerate(schema) if k > 0)
            acts['pred_keypoints'] = keep(out['pred_keypoints'][:, :, cls_idx * max_k:cls_idx * max_k + schema[cls_idx]])

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
    print(f"\n[{args.variant}] Wrote {n} activations -> {out_path.name}")


if __name__ == '__main__':
    main()
