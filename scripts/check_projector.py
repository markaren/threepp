"""Localize the RF-DETR projector mismatch: rebuild C2f from weights + captured
taps (ref.bin) in torch, compare each stage to the captured proj.0."""
import struct, sys
import numpy as np
import torch
import torch.nn.functional as F

def parse(path):
    f = open(path, 'rb'); assert f.read(4) == b'YOLO'
    struct.unpack('<I', f.read(4)); n = struct.unpack('<I', f.read(4))[0]
    data, shapes = {}, {}
    for _ in range(n):
        ln = struct.unpack('<I', f.read(4))[0]; nm = f.read(ln).decode()
        nd = struct.unpack('<I', f.read(4))[0]; sh = struct.unpack('<%dI' % nd, f.read(4*nd))
        db = struct.unpack('<I', f.read(4))[0]
        arr = np.frombuffer(f.read(db), dtype=np.float32).copy()
        data[nm] = arr; shapes[nm] = list(sh)
    return data, shapes

W, WS = parse('rfdetr-nano.weights')
R, RS = parse('rfdetr_nano_ref.bin')

def t(name, src=W, srcs=WS):
    return torch.from_numpy(src[name]).reshape(srcs[name])

# taps captured as projector input [1,384,24,24] (strip leading 1 if present)
taps = []
for k in range(4):
    a = R[f'tap.{k}']; sh = RS[f'tap.{k}']
    taps.append(torch.from_numpy(a).reshape(sh).reshape(1, 384, 24, 24))
x = torch.cat(taps, dim=1)  # [1,1536,24,24]
print('cat in', x.shape)

def ln_chw(z, w, b, eps=1e-6):
    z = z.permute(0, 2, 3, 1)
    z = F.layer_norm(z, (z.shape[-1],), w, b, eps)
    return z.permute(0, 3, 1, 2)

def convx(z, pfx, k):
    wt = t(pfx + '.conv.weight')
    z = F.conv2d(z, wt, bias=None, stride=1, padding=k // 2)
    z = ln_chw(z, t(pfx + '.bn.weight'), t(pfx + '.bn.bias'))
    return F.silu(z)

def bottleneck(z, pfx):
    return convx(convx(z, pfx + '.cv1', 3), pfx + '.cv2', 3)

P = 'backbone.0.projector.stages.0.0'
cv1 = convx(x, P + '.cv1', 1)            # [1,256,24,24]
y0, y1 = cv1[:, :128], cv1[:, 128:]
y2 = bottleneck(y1, P + '.m.0')
y3 = bottleneck(y2, P + '.m.1')
y4 = bottleneck(y3, P + '.m.2')
cat = torch.cat([y0, y1, y2, y3, y4], dim=1)  # [1,640,24,24]
c2f = convx(cat, P + '.cv2', 1)          # [1,256,24,24]
proj0 = ln_chw(c2f, t('backbone.0.projector.stages.0.1.weight'),
               t('backbone.0.projector.stages.0.1.bias'))

ref = torch.from_numpy(R['proj.0']).reshape(RS['proj.0'])
got = proj0.reshape(ref.shape)
err = (got - ref).abs().max().item()
rel = (got - ref).abs().sum().item() / ref.abs().sum().item()
print('proj0 vs ref  maxAbsErr %.3e  relL1 %.3e' % (err, rel))
print('ref  proj0 stats: mean %.4f min %.4f max %.4f' % (ref.mean(), ref.min(), ref.max()))
print('mine proj0 stats: mean %.4f min %.4f max %.4f' % (got.mean(), got.min(), got.max()))
