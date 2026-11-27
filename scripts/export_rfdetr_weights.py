"""Export RF-DETR weights to the 'YOLO' binary format used by the threepp
Vulkan inference examples.

    python scripts/export_rfdetr_weights.py                   # -> rfdetr-nano.weights
    python scripts/export_rfdetr_weights.py --variant medium  # -> rfdetr-medium.weights
    python scripts/export_rfdetr_weights.py --variant keypoint --checkpoint rf-detr-keypoint-preview-xlarge.pth
                                                              # -> rfdetr-keypoint.weights

LWDETR state_dict is dumped verbatim (fp32). The C++ side folds nothing here —
RF-DETR's norms are LayerNorm (folded at runtime in the shaders), not Conv+BN,
so weights are uploaded as-is. Variant must match the RfDetrVariant passed C++-side
(the loader asserts dec_layers agree).

The keypoint variant is RF-DETR Keypoint (preview: COCO person, 17 keypoints).
It also gets "derived.keypoint_queries" [num_queries, 17, 256]: the decoder's
initial keypoint queries. They depend only on the learned query_feat, so they
are computed here once, with the model's own ConditionalQueryInitializer.
"""
import argparse
import struct
from pathlib import Path

import torch
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRKeypointPreview

VARIANTS = {'nano': RFDETRNano, 'small': RFDETRSmall, 'medium': RFDETRMedium,
            'keypoint': RFDETRKeypointPreview}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', choices=list(VARIANTS), default='nano')
    ap.add_argument('--checkpoint', help='weights file for --variant keypoint')
    args = ap.parse_args()

    out_path = Path(__file__).with_name(f'rfdetr-{args.variant}.weights')
    if args.variant == 'keypoint':
        if not args.checkpoint:
            ap.error('--variant keypoint needs --checkpoint')
        det = VARIANTS[args.variant](pretrain_weights=args.checkpoint, num_classes=1, device='cpu')
    else:
        det = VARIANTS[args.variant]()
    net = det.model.model  # LWDETR nn.Module
    net.eval()
    sd = dict(net.state_dict())
    if args.variant == 'keypoint':
        nq = det.model_config.num_queries
        with torch.no_grad():
            queries = net.query_feat.weight[:nq].unsqueeze(0)
            sd['derived.keypoint_queries'] = net.transformer.keypoint_query_initializer(queries)[0]

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
    print(f"[{args.variant}] Wrote {n} tensors -> {out_path.name}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
