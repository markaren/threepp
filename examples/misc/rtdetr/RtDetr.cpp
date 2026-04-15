/// RtDetr.cpp — RT-DETR-L object detector via WGSL compute shaders.
///
/// Milestone 1: weight loading, BatchNorm folding, fp16 weight packing.
/// Milestone 2: DWConv (depthwise convolution) kernel.

#include "RtDetr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <webgpu/webgpu.h>

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#else
#  include <webgpu/wgpu.h>
#endif

using namespace threepp;

namespace rtdetr {

// ============================================================
//  GPU param structs (16-byte aligned for uniform buffers)
// ============================================================
namespace {

struct ConvParams {
    uint32_t in_c, out_c, in_h, in_w;
    uint32_t out_h, out_w, k_h, k_w;
    uint32_t stride_h, stride_w, pad_h, pad_w;
    uint32_t has_bias, activation, _p0, _p1;   // 64 bytes
};

inline uint32_t divCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ============================================================
//  WGSL: depthwise conv
// ============================================================
//
// Weight layout: [C, 1, kH, kW] packed as f16 pairs → u32.
// Each output element out[c, y, x] reads only from input channel c.
// Workgroup 8×8×4 covers 8×8 output × 4 channels.
//
static const char* kDwConvWgsl = R"WGSL(
struct ConvParams {
    in_c: u32, out_c: u32, in_h: u32, in_w: u32,
    out_h: u32, out_w: u32, k_h: u32, k_w: u32,
    stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
    has_bias: u32, activation: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform>             p:    ConvParams;
@group(0) @binding(1) var<storage, read>       inp:  array<f32>;
@group(0) @binding(2) var<storage, read>       wt:   array<u32>;   // packed f16 pairs
@group(0) @binding(3) var<storage, read>       bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

fn load_weight(i: u32) -> f32 {
    let pair = unpack2x16float(wt[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let oc = gid.z;
    if ox >= p.out_w || oy >= p.out_h || oc >= p.out_c { return; }

    let kstride = p.k_h * p.k_w;            // weights per output channel
    let in_pitch  = p.in_h * p.in_w;
    let out_pitch = p.out_h * p.out_w;

    var sum = 0.0;
    for (var ky = 0u; ky < p.k_h; ky++) {
        for (var kx = 0u; kx < p.k_w; kx++) {
            let iy = i32(oy * p.stride_h + ky) - i32(p.pad_h);
            let ix = i32(ox * p.stride_w + kx) - i32(p.pad_w);
            if iy >= 0 && iy < i32(p.in_h) && ix >= 0 && ix < i32(p.in_w) {
                let ii = oc * in_pitch + u32(iy) * p.in_w + u32(ix);
                let wi = oc * kstride + ky * p.k_w + kx;
                sum += inp[ii] * load_weight(wi);
            }
        }
    }

    if p.has_bias != 0u { sum += bias[oc]; }

    // activation: 0=None, 1=ReLU, 2=SiLU
    if p.activation == 1u {
        sum = max(0.0, sum);
    } else if p.activation == 2u {
        sum = sum / (1.0 + exp(-sum));
    }

    out[oc * out_pitch + oy * p.out_w + ox] = sum;
}
)WGSL";

// ============================================================
//  WGSL: standard (non-grouped) 2D conv
// ============================================================
//
// Weight layout: [out_c, in_c, kH, kW] packed as f16 pairs → u32.
// One thread per output element. Workgroup 8×8×4 covers
// 8×8 output pixels × 4 output channels.
//
static const char* kConv2dWgsl = R"WGSL(
struct ConvParams {
    in_c: u32, out_c: u32, in_h: u32, in_w: u32,
    out_h: u32, out_w: u32, k_h: u32, k_w: u32,
    stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
    has_bias: u32, activation: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var<uniform>             p:    ConvParams;
@group(0) @binding(1) var<storage, read>       inp:  array<f32>;
@group(0) @binding(2) var<storage, read>       wt:   array<u32>;   // packed f16 pairs
@group(0) @binding(3) var<storage, read>       bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

fn load_weight(i: u32) -> f32 {
    let pair = unpack2x16float(wt[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let oc = gid.z;
    if ox >= p.out_w || oy >= p.out_h || oc >= p.out_c { return; }

    let in_pitch  = p.in_h * p.in_w;
    let out_pitch = p.out_h * p.out_w;
    let k_area    = p.k_h * p.k_w;
    let w_per_oc  = p.in_c * k_area;

    var sum = 0.0;
    for (var ic = 0u; ic < p.in_c; ic++) {
        for (var ky = 0u; ky < p.k_h; ky++) {
            for (var kx = 0u; kx < p.k_w; kx++) {
                let iy = i32(oy * p.stride_h + ky) - i32(p.pad_h);
                let ix = i32(ox * p.stride_w + kx) - i32(p.pad_w);
                if iy >= 0 && iy < i32(p.in_h) && ix >= 0 && ix < i32(p.in_w) {
                    let ii = ic * in_pitch + u32(iy) * p.in_w + u32(ix);
                    let wi = oc * w_per_oc + ic * k_area + ky * p.k_w + kx;
                    sum += inp[ii] * load_weight(wi);
                }
            }
        }
    }

    if p.has_bias != 0u { sum += bias[oc]; }

    if p.activation == 1u {
        sum = max(0.0, sum);
    } else if p.activation == 2u {
        sum = sum / (1.0 + exp(-sum));
    }

    out[oc * out_pitch + oy * p.out_w + ox] = sum;
}
)WGSL";

// ============================================================
//  WGSL: 2D max-pool with asymmetric padding
// ============================================================
struct PoolParams {
    uint32_t c, in_h, in_w, out_h;
    uint32_t out_w, k_h, k_w, stride_h;
    uint32_t stride_w, pad_t, pad_l, _p0;   // 48 bytes → pad to 64
    uint32_t _p1, _p2, _p3, _p4;
};

static const char* kMaxPoolWgsl = R"WGSL(
struct PoolParams {
    c: u32, in_h: u32, in_w: u32, out_h: u32,
    out_w: u32, k_h: u32, k_w: u32, stride_h: u32,
    stride_w: u32, pad_t: u32, pad_l: u32, _p0: u32,
    _p1: u32, _p2: u32, _p3: u32, _p4: u32,
}
@group(0) @binding(0) var<uniform>             p:   PoolParams;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let oc = gid.z;
    if ox >= p.out_w || oy >= p.out_h || oc >= p.c { return; }

    let in_pitch  = p.in_h * p.in_w;
    let out_pitch = p.out_h * p.out_w;

    var m: f32 = -3.4e38;
    for (var ky = 0u; ky < p.k_h; ky++) {
        for (var kx = 0u; kx < p.k_w; kx++) {
            let iy = i32(oy * p.stride_h + ky) - i32(p.pad_t);
            let ix = i32(ox * p.stride_w + kx) - i32(p.pad_l);
            if iy >= 0 && iy < i32(p.in_h) && ix >= 0 && ix < i32(p.in_w) {
                let v = inp[oc * in_pitch + u32(iy) * p.in_w + u32(ix)];
                m = max(m, v);
            }
        }
    }
    out[oc * out_pitch + oy * p.out_w + ox] = m;
}
)WGSL";

// ============================================================
//  WGSL: channel-axis concat of two tensors with matching H, W
// ============================================================
struct ConcatParams {
    uint32_t c_a, c_b, h, w;   // 16 bytes
};

static const char* kConcatCWgsl = R"WGSL(
struct ConcatParams { c_a: u32, c_b: u32, h: u32, w: u32, }
@group(0) @binding(0) var<uniform>             p:   ConcatParams;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x  = gid.x; let y  = gid.y; let oc = gid.z;
    let c_total = p.c_a + p.c_b;
    if x >= p.w || y >= p.h || oc >= c_total { return; }

    let pitch = p.h * p.w;
    let off   = y * p.w + x;
    var v: f32;
    if oc < p.c_a {
        v = a[oc * pitch + off];
    } else {
        v = b[(oc - p.c_a) * pitch + off];
    }
    out[oc * pitch + off] = v;
}
)WGSL";

// ============================================================
//  WGSL: elementwise add
// ============================================================
struct AddParams { uint32_t n, _p0, _p1, _p2; };

static const char* kAddWgsl = R"WGSL(
struct AddParams { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform>             p:   AddParams;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.n { return; }
    out[i] = a[i] + b[i];
}
)WGSL";

// ============================================================
//  WGSL: fused elementwise (a + b) -> SiLU
// ============================================================
static const char* kAddSiluWgsl = R"WGSL(
struct AddParams { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform>             p:   AddParams;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.n { return; }
    let v = a[i] + b[i];
    out[i] = v / (1.0 + exp(-v));
}
)WGSL";

// ============================================================
//  WGSL: 2x nearest-neighbor upsample along H and W
// ============================================================
struct UpsampleParams { uint32_t C, inH, inW, _p; };

static const char* kUpsample2xWgsl = R"WGSL(
struct UpsampleParams { C: u32, inH: u32, inW: u32, _p: u32, }
@group(0) @binding(0) var<uniform>             p:   UpsampleParams;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outH = p.inH * 2u;
    let outW = p.inW * 2u;
    let ox = gid.x; let oy = gid.y; let oc = gid.z;
    if ox >= outW || oy >= outH || oc >= p.C { return; }
    let ix = ox >> 1u; let iy = oy >> 1u;
    let v = inp[oc * p.inH * p.inW + iy * p.inW + ix];
    out[oc * outH * outW + oy * outW + ox] = v;
}
)WGSL";

// ============================================================
//  WGSL: linear / GEMM  Y[M,N] = X[M,K] @ W[N,K]^T + b[N]
// ============================================================
struct LinearParams { uint32_t M, N, K, hasBias; };

static const char* kLinearWgsl = R"WGSL(
struct LinearParams { M: u32, N: u32, K: u32, hasBias: u32, }
@group(0) @binding(0) var<uniform>             p: LinearParams;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;   // [N, K] row-major
@group(0) @binding(3) var<storage, read>       b: array<f32>;   // [N] (unused if hasBias==0)
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.y;
    let n = gid.x;
    if (m >= p.M || n >= p.N) { return; }
    var acc: f32 = 0.0;
    let xBase = m * p.K;
    let wBase = n * p.K;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        acc = acc + x[xBase + k] * w[wBase + k];
    }
    if (p.hasBias != 0u) { acc = acc + b[n]; }
    y[m * p.N + n] = acc;
}
)WGSL";

// ============================================================
//  WGSL: LayerNorm along last axis
// ============================================================
struct LNParams { uint32_t M, D; float eps, _pad; };
static const char* kLayerNormWgsl = R"WGSL(
struct LNParams { M: u32, D: u32, eps: f32, _pad: f32, }
@group(0) @binding(0) var<uniform>             p:     LNParams;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       beta:  array<f32>;
@group(0) @binding(4) var<storage, read_write> y:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= p.M) { return; }
    let base = m * p.D;
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < p.D; k = k + 1u) { sum = sum + x[base + k]; }
    let mean = sum / f32(p.D);
    var sqsum: f32 = 0.0;
    for (var k: u32 = 0u; k < p.D; k = k + 1u) {
        let d = x[base + k] - mean;
        sqsum = sqsum + d * d;
    }
    let inv = inverseSqrt(sqsum / f32(p.D) + p.eps);
    for (var k: u32 = 0u; k < p.D; k = k + 1u) {
        y[base + k] = (x[base + k] - mean) * inv * gamma[k] + beta[k];
    }
}
)WGSL";

// ============================================================
//  WGSL: row-wise softmax along last axis (numerically stable)
// ============================================================
struct SoftmaxParams { uint32_t M, N, _p0, _p1; };
static const char* kSoftmaxWgsl = R"WGSL(
struct SoftmaxParams { M: u32, N: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform>             p: SoftmaxParams;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= p.M) { return; }
    let base = m * p.N;
    var mx: f32 = x[base];
    for (var k: u32 = 1u; k < p.N; k = k + 1u) {
        let v = x[base + k]; if (v > mx) { mx = v; }
    }
    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < p.N; k = k + 1u) {
        let e = exp(x[base + k] - mx);
        y[base + k] = e;
        s = s + e;
    }
    let inv = 1.0 / s;
    for (var k: u32 = 0u; k < p.N; k = k + 1u) { y[base + k] = y[base + k] * inv; }
}
)WGSL";

// ============================================================
//  WGSL: exact GELU (erf form)
// ============================================================
struct GeluParams { uint32_t n, _p0, _p1, _p2; };
static const char* kGeluWgsl = R"WGSL(
struct GeluParams { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform>             p: GeluParams;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// Abramowitz-Stegun 7.1.26 approximation of erf (max abs err ~1.5e-7).
fn erf_approx(z: f32) -> f32 {
    let sign = select(1.0, -1.0, z < 0.0);
    let a = abs(z);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = (((( 1.061405429 * t
                   - 1.453152027) * t
                   + 1.421413741) * t
                   - 0.284496736) * t
                   + 0.254829592) * t;
    return sign * (1.0 - poly * exp(-a * a));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let v = x[i];
    y[i] = 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865475));
}
)WGSL";

// ============================================================
//  WGSL: elementwise ReLU
// ============================================================
static const char* kReluWgsl = R"WGSL(
struct ReluParams { n: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<uniform>             p: ReluParams;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = max(0.0, x[i]);
}
)WGSL";

// ============================================================
//  WGSL: Multi-Scale Deformable Attention (MSDeformAttn)
// ============================================================
//
// RT-DETR-L parameters: H=8 heads, L=3 levels, P=4 points, d=32 head dim.
// Inputs:
//   value:   [total_tokens, H*d]  (projected values, flattened across levels)
//   shapes:  [L, 2]  (H_l, W_l per level)
//   offsets: [N_q, H*L*P*2]  (predicted sampling offsets)
//   attnW:   [N_q, H*L*P]   (attention weights, already softmaxed)
//   refPts:  [N_q, 2]       (reference points in [0,1] normalized coords)
//   lvlStarts: [L]           (cumulative token offset per level)
// Output:  [N_q, H*d]
//
// Each thread computes one (query, head_dim_k) output element, iterating
// over L*P sampling points and accumulating the bilinearly-interpolated
// value weighted by attention.
//
struct MsDeformParams {
    uint32_t Nq;       ///< number of queries
    uint32_t H;        ///< number of attention heads
    uint32_t d;        ///< head dimension (D / H)
    uint32_t L;        ///< number of levels
    uint32_t P;        ///< number of sampling points per head per level
    uint32_t _p0, _p1, _p2;   // pad to 32 bytes
};

static const char* kMsDeformAttnWgsl = R"WGSL(
struct MsDeformParams {
    Nq: u32, H: u32, d: u32, L: u32, P: u32,
    _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform>             p:         MsDeformParams;
@group(0) @binding(1) var<storage, read>       value:     array<f32>;   // [total_tokens, H*d]
@group(0) @binding(2) var<storage, read>       shapes:    array<u32>;   // [L*2]: H0,W0,H1,W1,...
@group(0) @binding(3) var<storage, read>       lvlStart:  array<u32>;   // [L]: cumulative token offset
@group(0) @binding(4) var<storage, read>       refPts:    array<f32>;   // [Nq, 2]: cx, cy in [0,1]
@group(0) @binding(5) var<storage, read>       offsets:   array<f32>;   // [Nq, H*L*P*2]
@group(0) @binding(6) var<storage, read>       attnW:     array<f32>;   // [Nq, H*L*P]
@group(0) @binding(7) var<storage, read_write> out:       array<f32>;   // [Nq, H*d]

// Bilinear sampling helper.  Samples value[token_offset + gy*W + gx, h*d + k]
// at fractional position (sx, sy) within the level's spatial grid.
// Clamps to grid boundaries.
fn bilinear(h: u32, k: u32, base: u32, Hl: u32, Wl: u32, sx: f32, sy: f32) -> f32 {
    let Hd = p.H * p.d;
    // Pixel centers at 0.5 .. Wl-0.5; convert to integer grid
    let fx = sx - 0.5;
    let fy = sy - 0.5;

    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let wx = fx - f32(x0);
    let wy = fy - f32(y0);

    let iW = i32(Wl);
    let iH = i32(Hl);
    let col = h * p.d + k;

    var acc: f32 = 0.0;

    // top-left
    if (x0 >= 0 && x0 < iW && y0 >= 0 && y0 < iH) {
        acc += (1.0 - wx) * (1.0 - wy) * value[(base + u32(y0) * Wl + u32(x0)) * Hd + col];
    }
    // top-right
    if (x1 >= 0 && x1 < iW && y0 >= 0 && y0 < iH) {
        acc += wx * (1.0 - wy) * value[(base + u32(y0) * Wl + u32(x1)) * Hd + col];
    }
    // bottom-left
    if (x0 >= 0 && x0 < iW && y1 >= 0 && y1 < iH) {
        acc += (1.0 - wx) * wy * value[(base + u32(y1) * Wl + u32(x0)) * Hd + col];
    }
    // bottom-right
    if (x1 >= 0 && x1 < iW && y1 >= 0 && y1 < iH) {
        acc += wx * wy * value[(base + u32(y1) * Wl + u32(x1)) * Hd + col];
    }
    return acc;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;    // head-dim index  [0..d)
    let q = gid.y;    // query index     [0..Nq)
    let h = gid.z;    // head index      [0..H)
    if (k >= p.d || q >= p.Nq || h >= p.H) { return; }

    // Reference point for this query (normalized 0..1).
    let ref_x = refPts[q * 2u + 0u];
    let ref_y = refPts[q * 2u + 1u];

    let LP = p.L * p.P;
    let offStride = p.H * LP * 2u;   // row stride of offsets: H*L*P*2
    let awStride  = p.H * LP;        // row stride of attn weights: H*L*P

    var acc: f32 = 0.0;
    for (var l: u32 = 0u; l < p.L; l = l + 1u) {
        let Hl = shapes[l * 2u + 0u];
        let Wl = shapes[l * 2u + 1u];
        let base = lvlStart[l];

        for (var pt: u32 = 0u; pt < p.P; pt = pt + 1u) {
            // Offset index: offsets[q, h, l, pt, 0/1]
            let offIdx = q * offStride + (h * LP + l * p.P + pt) * 2u;
            let ox = offsets[offIdx + 0u];
            let oy = offsets[offIdx + 1u];

            // Attention weight index: attnW[q, h, l*P + pt]
            let aw = attnW[q * awStride + h * LP + l * p.P + pt];

            // Sample position in pixel space (pixel centers at 0.5).
            let sx = ref_x * f32(Wl) + ox;
            let sy = ref_y * f32(Hl) + oy;

            acc += aw * bilinear(h, k, base, Hl, Wl, sx, sy);
        }
    }

    out[q * (p.H * p.d) + h * p.d + k] = acc;
}
)WGSL";

// ============================================================
//  WGSL: multi-head attention scores (pre-softmax) from combined QKV
// ============================================================
struct AttnParams {
    uint32_t M;      ///< tokens
    uint32_t H;      ///< num heads
    uint32_t d;      ///< head dim (= D / H)
    uint32_t stride; ///< == 3 * D  (row stride of qkv)
    float    scale;  ///< 1 / sqrt(d)
    uint32_t _p0, _p1, _p2;
};

static const char* kAttnScoresWgsl = R"WGSL(
struct AttnParams {
    M: u32, H: u32, d: u32, stride: u32,
    scale: f32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>             p:   AttnParams;
@group(0) @binding(1) var<storage, read>       qkv: array<f32>;   // [M, 3*D]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;   // [H, M, M]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;   // key index
    let i = gid.y;   // query index
    let h = gid.z;   // head
    if (i >= p.M || j >= p.M || h >= p.H) { return; }
    let D = p.H * p.d;
    let qBase = i * p.stride + 0u * D + h * p.d;
    let kBase = j * p.stride + 1u * D + h * p.d;
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < p.d; k = k + 1u) {
        acc = acc + qkv[qBase + k] * qkv[kBase + k];
    }
    out[h * p.M * p.M + i * p.M + j] = acc * p.scale;
}
)WGSL";

// ============================================================
//  WGSL: apply attention: out[i, h*d + k] = Σ_j attn[h,i,j] * V[j, h, k]
// ============================================================
static const char* kAttnApplyWgsl = R"WGSL(
struct AttnParams {
    M: u32, H: u32, d: u32, stride: u32,
    scale: f32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>             p:    AttnParams;
@group(0) @binding(1) var<storage, read>       qkv:  array<f32>;   // [M, 3*D]
@group(0) @binding(2) var<storage, read>       attn: array<f32>;   // [H, M, M]
@group(0) @binding(3) var<storage, read_write> out:  array<f32>;   // [M, D]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;   // head-dim index
    let i = gid.y;   // query/token
    let h = gid.z;   // head
    if (i >= p.M || k >= p.d || h >= p.H) { return; }
    let D = p.H * p.d;
    let aBase = h * p.M * p.M + i * p.M;
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < p.M; j = j + 1u) {
        let vIdx = j * p.stride + 2u * D + h * p.d + k;
        acc = acc + attn[aBase + j] * qkv[vIdx];
    }
    out[i * D + h * p.d + k] = acc;
}
)WGSL";

// ============================================================
//  fp32 → fp16 converter (flush subnormals, round-toward-zero)
// ============================================================
inline float f16_to_f32(uint16_t h) {
    uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) { f = s; }
        else {
            // subnormal
            int ex = 1;
            while ((m & 0x400u) == 0) { m <<= 1; ex -= 1; }
            m &= 0x3FFu;
            f = s | (uint32_t(ex + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = int32_t((x >> 23) & 0xFFu) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 128) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >  15)  return uint16_t(sign | 0x7C00u);
    if (exp < -14)  return uint16_t(sign);

    uint32_t he = uint32_t(exp + 15) & 0x1Fu;
    uint32_t hm = mant >> 13;
    return uint16_t(sign | (he << 10) | hm);
}

bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

}// namespace

// ============================================================
//  Construction
// ============================================================
RtDetr::RtDetr(WgpuRenderer& renderer)
    : renderer_(renderer),
      dwConvPipe_(renderer, kDwConvWgsl, "main"),
      convPipe_(renderer, kConv2dWgsl, "main"),
      maxPoolPipe_(renderer, kMaxPoolWgsl, "main"),
      concatCPipe_(renderer, kConcatCWgsl, "main"),
      addPipe_(renderer, kAddWgsl, "main"),
      linearPipe_(renderer, kLinearWgsl, "main"),
      layerNormPipe_(renderer, kLayerNormWgsl, "main"),
      softmaxPipe_(renderer, kSoftmaxWgsl, "main"),
      geluPipe_(renderer, kGeluWgsl, "main"),
      attnScoresPipe_(renderer, kAttnScoresWgsl, "main"),
      attnApplyPipe_(renderer, kAttnApplyWgsl, "main"),
      upsample2xPipe_(renderer, kUpsample2xWgsl, "main"),
      addSiluPipe_(renderer, kAddSiluWgsl, "main"),
      msDeformAttnPipe_(renderer, kMsDeformAttnWgsl, "main"),
      reluPipe_(renderer, kReluWgsl, "main"),
      convParamBuf_(renderer, sizeof(ConvParams), WgpuBuffer::Usage::Uniform),
      poolParamBuf_(renderer, sizeof(PoolParams), WgpuBuffer::Usage::Uniform),
      concatParamBuf_(renderer, sizeof(ConcatParams), WgpuBuffer::Usage::Uniform),
      addParamBuf_(renderer, sizeof(AddParams), WgpuBuffer::Usage::Uniform),
      linearParamBuf_(renderer, sizeof(LinearParams), WgpuBuffer::Usage::Uniform),
      lnParamBuf_(renderer, sizeof(LNParams), WgpuBuffer::Usage::Uniform),
      softmaxParamBuf_(renderer, sizeof(SoftmaxParams), WgpuBuffer::Usage::Uniform),
      geluParamBuf_(renderer, sizeof(GeluParams), WgpuBuffer::Usage::Uniform),
      attnParamBuf_(renderer, sizeof(AttnParams), WgpuBuffer::Usage::Uniform),
      upsampleParamBuf_(renderer, sizeof(UpsampleParams), WgpuBuffer::Usage::Uniform),
      msDeformParamBuf_(renderer, sizeof(MsDeformParams), WgpuBuffer::Usage::Uniform)
{
}

RtDetr::~RtDetr() = default;

// ============================================================
//  Weight loading (Milestone 1)
// ============================================================
void RtDetr::loadWeights(const std::string& path) {
    auto w = parseWeightBinary(path);

    // Fold BN into preceding Conv weights. Ultralytics RT-DETR wraps
    //   Conv2d + BN in a `Conv` module: <prefix>.conv.weight + <prefix>.bn.*.
    // DWConv uses the same naming.  The decoder's input_proj uses an
    // nn.Sequential(Conv2d, BatchNorm2d) with names <prefix>.0.weight and
    // <prefix>.1.*, so synthesize aliases before folding.
    for (auto& kv : std::vector<std::pair<std::string, std::vector<float>>>(w.data.begin(), w.data.end())) {
        const auto& name = kv.first;
        if (endsWith(name, ".0.weight")) {
            const std::string prefix = name.substr(0, name.size() - std::string(".0.weight").size());
            const std::string bn1 = prefix + ".1.weight";
            if (w.data.count(bn1) == 0) continue;
            // Make aliases so the existing fold loop picks these up.
            const std::string convAlias = prefix + ".conv.weight";
            if (w.data.count(convAlias)) continue;  // already present — skip
            w.data[convAlias]   = w.data.at(name);
            w.shapes[convAlias] = w.shapes.at(name);
            auto aliasBN = [&](const char* suffix) {
                std::string from = prefix + ".1" + suffix;
                std::string to   = prefix + ".bn" + suffix;
                if (w.data.count(from) && !w.data.count(to)) {
                    w.data[to]   = w.data.at(from);
                    w.shapes[to] = w.shapes.at(from);
                }
            };
            aliasBN(".weight");
            aliasBN(".bias");
            aliasBN(".running_mean");
            aliasBN(".running_var");
        }
    }

    std::vector<std::string> convKeys;
    for (auto& kv : w.data) {
        if (endsWith(kv.first, ".conv.weight")) convKeys.push_back(kv.first);
    }

    size_t foldedCount = 0;
    for (auto& convKey : convKeys) {
        const std::string prefix = convKey.substr(0, convKey.size() - std::string(".conv.weight").size());
        const std::string bnW = prefix + ".bn.weight";
        const std::string bnB = prefix + ".bn.bias";
        const std::string bnM = prefix + ".bn.running_mean";
        const std::string bnV = prefix + ".bn.running_var";
        if (!w.data.count(bnW) || !w.data.count(bnB) ||
            !w.data.count(bnM) || !w.data.count(bnV)) continue;

        const auto& wt    = w.data.at(convKey);
        const auto& gamma = w.data.at(bnW);
        const auto& beta  = w.data.at(bnB);
        const auto& mean  = w.data.at(bnM);
        const auto& var_  = w.data.at(bnV);
        const auto& sh    = w.shapes.at(convKey);
        uint32_t oc  = sh[0];
        uint32_t per = uint32_t(wt.size()) / oc;

        std::vector<float> fw(wt.size());
        std::vector<float> fb(oc);
        for (uint32_t c = 0; c < oc; ++c) {
            float s = gamma[c] / std::sqrt(var_[c] + BN_EPS);
            fb[c]   = beta[c] - mean[c] * s;
            for (uint32_t i = 0; i < per; ++i)
                fw[c * per + i] = wt[c * per + i] * s;
        }

        w.data[prefix + ".fused.weight"]   = std::move(fw);
        w.shapes[prefix + ".fused.weight"] = sh;
        w.data[prefix + ".fused.bias"]     = std::move(fb);
        w.shapes[prefix + ".fused.bias"]   = {oc};
        ++foldedCount;
    }

    weights_.clear();
    cpuWeights_.clear();
    size_t packedCount = 0;
    for (auto& [name, data] : w.data) {
        const auto& sh = w.shapes.at(name);
        bool isConvWeight = (sh.size() == 4 && endsWith(name, ".weight"));
        bool isFusedBias  = endsWith(name, ".fused.bias");

        // Biases upload as fp32 and are also used as-is in CPU references.
        if (isFusedBias) cpuWeights_[name] = data;

        if (isConvWeight) {
            size_t n = data.size();
            if (n == 0 || (n % 2) != 0) {
                auto t = makeTensorV(renderer_, sh);
                t.upload(data.data());
                weights_.emplace(name, std::move(t));
                // No fp16 pack for odd-sized weights; CPU mirror = raw data
                if (endsWith(name, ".fused.weight")) cpuWeights_[name] = data;
                continue;
            }
            std::vector<uint32_t> packed(n / 2);
            std::vector<float>    rt;
            bool keepMirror = endsWith(name, ".fused.weight");
            if (keepMirror) rt.resize(n);
            for (size_t i = 0; i < n; i += 2) {
                uint16_t lo = f32_to_f16(data[i]);
                uint16_t hi = f32_to_f16(data[i + 1]);
                packed[i / 2] = uint32_t(lo) | (uint32_t(hi) << 16);
                if (keepMirror) {
                    rt[i]     = f16_to_f32(lo);
                    rt[i + 1] = f16_to_f32(hi);
                }
            }
            auto t = makeF16WeightTensor(renderer_, sh);
            t.buf->write(packed.data(), packed.size() * sizeof(uint32_t));
            weights_.emplace(name, std::move(t));
            if (keepMirror) cpuWeights_[name] = std::move(rt);
            ++packedCount;
        } else {
            auto t = makeTensorV(renderer_, sh);
            t.upload(data.data());
            weights_.emplace(name, std::move(t));
            // Mirror 2D linear weights and any bias on CPU for analytical tests
            // and reference comparisons.
            if (sh.size() == 2 || (sh.size() == 1 && endsWith(name, ".bias"))
                || (sh.size() == 1 && endsWith(name, ".weight"))) {
                cpuWeights_[name] = data;
            }
        }
    }

    std::cout << "RtDetr::loadWeights: "
              << weights_.size() << " GPU tensors"
              << " (folded " << foldedCount << " BN blocks,"
              << " fp16-packed " << packedCount << " conv weights)\n";
}

// ============================================================
//  Milestone 2: DWConv + test helpers
// ============================================================
GPUTensor RtDetr::allocFilled(std::initializer_list<uint32_t> shape, float value) {
    auto t = makeTensor(renderer_, shape);
    std::vector<float> buf(t.numel(), value);
    t.upload(buf.data());
    return t;
}

GPUTensor RtDetr::dwConv_(const GPUTensor& x,
                          const std::string& weightKey,
                          const std::string& biasKey,
                          int strideH, int strideW,
                          int padH, int padW,
                          Activation act) {
    auto& wt = weights_.at(weightKey);
    // Weight shape: [C, 1, kH, kW]. C must equal input channel count.
    if (wt.shape.size() != 4 || wt.shape[1] != 1u)
        throw std::runtime_error("RtDetr::dwConv_: weight '" + weightKey + "' is not depthwise");
    uint32_t C   = wt.shape[0];
    uint32_t k_h = wt.shape[2];
    uint32_t k_w = wt.shape[3];
    if (x.C() != C)
        throw std::runtime_error("RtDetr::dwConv_: input C mismatch for '" + weightKey + "'");

    uint32_t in_h  = x.H();
    uint32_t in_w  = x.W();
    uint32_t out_h = (in_h + 2 * uint32_t(padH) - k_h) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + 2 * uint32_t(padW) - k_w) / uint32_t(strideW) + 1;

    bool hasBias = !biasKey.empty() && weights_.count(biasKey);

    ConvParams cp{};
    cp.in_c = C; cp.out_c = C;
    cp.in_h = in_h; cp.in_w = in_w;
    cp.out_h = out_h; cp.out_w = out_w;
    cp.k_h = k_h; cp.k_w = k_w;
    cp.stride_h = uint32_t(strideH); cp.stride_w = uint32_t(strideW);
    cp.pad_h = uint32_t(padH);       cp.pad_w = uint32_t(padW);
    cp.has_bias   = hasBias ? 1u : 0u;
    cp.activation = uint32_t(act);
    convParamBuf_.write(&cp, sizeof(cp));

    // Readback-capable output so milestone-2 tests can verify values on CPU.
    // Later milestones will switch this back to plain Storage for intermediate
    // layers (and use readback tensors only at explicit probe points).
    auto out = makeReadbackTensor(renderer_, {C, out_h, out_w});

    WgpuBuffer& biasBuf = hasBias ? weights_.at(biasKey).buffer()
                                  : wt.buffer();   // dummy bind; shader ignores when has_bias==0

    dwConvPipe_.setUniformBuffer(0,     convParamBuf_);
    dwConvPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    dwConvPipe_.setStorageBufferRead(2, wt.buffer());
    dwConvPipe_.setStorageBufferRead(3, biasBuf);
    dwConvPipe_.setStorageBuffer(4,     out.buffer());
    dwConvPipe_.dispatch(divCeil(out_w, 8), divCeil(out_h, 8), divCeil(C, 4));

    return out;
}

GPUTensor RtDetr::conv_(const GPUTensor& x,
                        const std::string& weightKey,
                        const std::string& biasKey,
                        int strideH, int strideW,
                        int padH, int padW,
                        Activation act) {
    return conv_(x, weightKey, biasKey, strideH, strideW,
                 padH, padW, padH, padW, act);
}

GPUTensor RtDetr::conv_(const GPUTensor& x,
                        const std::string& weightKey,
                        const std::string& biasKey,
                        int strideH, int strideW,
                        int padTop, int padLeft, int padBottom, int padRight,
                        Activation act) {
    auto& wt = weights_.at(weightKey);
    // Weight shape: [out_c, in_c, kH, kW].
    if (wt.shape.size() != 4)
        throw std::runtime_error("RtDetr::conv_: weight '" + weightKey + "' is not 4D");
    uint32_t out_c = wt.shape[0];
    uint32_t in_c  = wt.shape[1];
    uint32_t k_h   = wt.shape[2];
    uint32_t k_w   = wt.shape[3];
    if (x.C() != in_c)
        throw std::runtime_error("RtDetr::conv_: input C mismatch for '" + weightKey + "'");

    uint32_t in_h  = x.H();
    uint32_t in_w  = x.W();
    uint32_t out_h = (in_h + uint32_t(padTop)  + uint32_t(padBottom) - k_h) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + uint32_t(padLeft) + uint32_t(padRight)  - k_w) / uint32_t(strideW) + 1;

    bool hasBias = !biasKey.empty() && weights_.count(biasKey);

    ConvParams cp{};
    cp.in_c = in_c; cp.out_c = out_c;
    cp.in_h = in_h; cp.in_w = in_w;
    cp.out_h = out_h; cp.out_w = out_w;
    cp.k_h = k_h;   cp.k_w = k_w;
    cp.stride_h = uint32_t(strideH); cp.stride_w = uint32_t(strideW);
    cp.pad_h = uint32_t(padTop);     cp.pad_w = uint32_t(padLeft);
    cp.has_bias   = hasBias ? 1u : 0u;
    cp.activation = uint32_t(act);
    convParamBuf_.write(&cp, sizeof(cp));

    auto out = makeReadbackTensor(renderer_, {out_c, out_h, out_w});

    WgpuBuffer& biasBuf = hasBias ? weights_.at(biasKey).buffer()
                                  : wt.buffer();

    convPipe_.setUniformBuffer(0,     convParamBuf_);
    convPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    convPipe_.setStorageBufferRead(2, wt.buffer());
    convPipe_.setStorageBufferRead(3, biasBuf);
    convPipe_.setStorageBuffer(4,     out.buffer());
    convPipe_.dispatch(divCeil(out_w, 8), divCeil(out_h, 8), divCeil(out_c, 4));

    return out;
}

GPUTensor RtDetr::uploadTensor(std::initializer_list<uint32_t> shape, const float* data) {
    auto t = makeReadbackTensor(renderer_, shape);
    t.upload(data);
    return t;
}

GPUTensor RtDetr::maxPool_(const GPUTensor& x,
                           int kH, int kW,
                           int strideH, int strideW,
                           int padTop, int padLeft, int padBottom, int padRight) {
    uint32_t C    = x.C();
    uint32_t in_h = x.H();
    uint32_t in_w = x.W();
    uint32_t out_h = (in_h + uint32_t(padTop) + uint32_t(padBottom) - uint32_t(kH)) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + uint32_t(padLeft) + uint32_t(padRight) - uint32_t(kW)) / uint32_t(strideW) + 1;

    PoolParams pp{};
    pp.c = C; pp.in_h = in_h; pp.in_w = in_w;
    pp.out_h = out_h; pp.out_w = out_w;
    pp.k_h = uint32_t(kH); pp.k_w = uint32_t(kW);
    pp.stride_h = uint32_t(strideH); pp.stride_w = uint32_t(strideW);
    pp.pad_t = uint32_t(padTop); pp.pad_l = uint32_t(padLeft);
    poolParamBuf_.write(&pp, sizeof(pp));

    auto out = makeReadbackTensor(renderer_, {C, out_h, out_w});

    maxPoolPipe_.setUniformBuffer(0,     poolParamBuf_);
    maxPoolPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    maxPoolPipe_.setStorageBuffer(2,     out.buffer());
    maxPoolPipe_.dispatch(divCeil(out_w, 8), divCeil(out_h, 8), divCeil(C, 4));

    return out;
}

GPUTensor RtDetr::hgStem_(const GPUTensor& x) {
    // Ultralytics HGStem (block.py):
    //   y  = ReLU(Conv3x3 s2 p1)      (stem1)
    //   y  = F.pad(y,  [0,1,0,1])
    //   x2 = ReLU(Conv2x2 s1 p0)      (stem2a, on padded y)
    //   x2 = F.pad(x2, [0,1,0,1])
    //   x2 = ReLU(Conv2x2 s1 p0)      (stem2b)
    //   x1 = MaxPool2x2 s1 p0 on padded y
    //   z  = concat([x1, x2], dim=1)
    //   z  = ReLU(Conv3x3 s2 p1)      (stem3)
    //   z  = ReLU(Conv1x1 s1 p0)      (stem4)
    using A = Activation;
    auto y   = conv_(x, "model.0.stem1.fused.weight", "model.0.stem1.fused.bias",
                     2, 2, 1, 1, A::ReLU);
    // F.pad [0,1,0,1] + Conv2x2 p=0 ≡ Conv2x2 with pad [top=0, left=0, bottom=1, right=1]
    auto x2a = conv_(y, "model.0.stem2a.fused.weight", "model.0.stem2a.fused.bias",
                     1, 1, 0, 0, 1, 1, A::ReLU);
    auto x2b = conv_(x2a, "model.0.stem2b.fused.weight", "model.0.stem2b.fused.bias",
                     1, 1, 0, 0, 1, 1, A::ReLU);
    auto x1  = maxPool_(y, 2, 2, 1, 1, 0, 0, 1, 1);
    auto cat = concatC_(x1, x2b);
    auto s3  = conv_(cat, "model.0.stem3.fused.weight", "model.0.stem3.fused.bias",
                     2, 2, 1, 1, A::ReLU);
    auto s4  = conv_(s3,  "model.0.stem4.fused.weight", "model.0.stem4.fused.bias",
                     1, 1, 0, 0, A::ReLU);
    return s4;
}

GPUTensor RtDetr::lightConv_(const GPUTensor& x, const std::string& prefix, int k) {
    // conv1: 1×1, BN-folded, NO activation
    auto m = conv_(x, prefix + ".conv1.fused.weight",
                      prefix + ".conv1.fused.bias",
                   1, 1, 0, 0, Activation::None);
    // conv2: DW k×k s=1 p=(k-1)/2, ReLU
    int pad = (k - 1) / 2;
    auto y = dwConv_(m, prefix + ".conv2.fused.weight",
                        prefix + ".conv2.fused.bias",
                     1, 1, pad, pad, Activation::ReLU);
    return y;
}

GPUTensor RtDetr::hgBlock_(const GPUTensor& x, const std::string& prefix,
                           int n, int k, bool shortcut, bool lightconv) {
    using A = Activation;
    int pad = (k - 1) / 2;

    const GPUTensor* prev = &x;
    std::vector<GPUTensor> owned;
    owned.reserve(n);
    for (int i = 0; i < n; ++i) {
        std::string mPrefix = prefix + ".m." + std::to_string(i);
        if (lightconv) {
            owned.push_back(lightConv_(*prev, mPrefix, k));
        } else {
            owned.push_back(conv_(*prev,
                mPrefix + ".fused.weight",
                mPrefix + ".fused.bias",
                1, 1, pad, pad, A::ReLU));
        }
        prev = &owned.back();
    }

    GPUTensor acc = concatC_(x, owned[0]);
    for (int i = 1; i < n; ++i) {
        acc = concatC_(acc, owned[i]);
    }

    auto s = conv_(acc, prefix + ".sc.fused.weight", prefix + ".sc.fused.bias",
                   1, 1, 0, 0, A::ReLU);
    auto e = conv_(s,   prefix + ".ec.fused.weight", prefix + ".ec.fused.bias",
                   1, 1, 0, 0, A::ReLU);

    if (shortcut) {
        if (e.C() != x.C() || e.H() != x.H() || e.W() != x.W())
            throw std::runtime_error("hgBlock_: shortcut shape mismatch for '" + prefix + "'");
        return addTensor_(e, x);
    }
    return e;
}

RtDetr::BackboneFeatures RtDetr::backbone_(const GPUTensor& x) {
    using A = Activation;
    // model.0  HGStem                                       3   → 48   stride 4
    auto f0 = hgStem_(x);
    // model.1  HGBlock plain k=3 n=6                        48  → 128  stride 4
    auto f1 = hgBlock_(f0, "model.1", 6, 3, false, false);
    // model.2  DWConv k=3 s=2 (NO activation — conv+BN only)  128 → 128  stride 8
    auto f2 = dwConv_(f1, "model.2.fused.weight", "model.2.fused.bias",
                      2, 2, 1, 1, A::None);
    // model.3  HGBlock plain k=3 n=6                        128 → 512  stride 8   <- P3
    auto p3 = hgBlock_(f2, "model.3", 6, 3, false, false);
    // model.4  DWConv k=3 s=2 (NO activation)               512 → 512  stride 16
    auto f4 = dwConv_(p3, "model.4.fused.weight", "model.4.fused.bias",
                      2, 2, 1, 1, A::None);
    // model.5  HGBlockLight k=5 n=6                         512 → 1024 stride 16
    auto f5 = hgBlock_(f4, "model.5", 6, 5, false, true);
    // model.6  HGBlockLight k=5 n=6 shortcut                1024→ 1024 stride 16
    auto f6 = hgBlock_(f5, "model.6", 6, 5, true,  true);
    // model.7  HGBlockLight k=5 n=6 shortcut                1024→ 1024 stride 16 <- P4
    auto p4 = hgBlock_(f6, "model.7", 6, 5, true,  true);
    // model.8  DWConv k=3 s=2 (NO activation — conv+BN only) 1024→ 1024 stride 32
    auto f8 = dwConv_(p4, "model.8.fused.weight", "model.8.fused.bias",
                      2, 2, 1, 1, A::None);
    // model.9  HGBlockLight k=5 n=6                         1024→ 2048 stride 32
    auto f9 = hgBlock_(f8, "model.9", 6, 5, false, true);
    // model.10 Conv 1×1 (BN-folded, NO activation)          2048→ 256  stride 32 <- P5
    auto p5 = conv_(f9, "model.10.fused.weight", "model.10.fused.bias",
                    1, 1, 0, 0, A::None);
    return { std::move(p3), std::move(p4), std::move(p5) };
}

GPUTensor RtDetr::linear_(const GPUTensor& x,
                          const std::string& wKey,
                          const std::string& bKey) {
    auto wIt = weights_.find(wKey);
    if (wIt == weights_.end())
        throw std::runtime_error("linear_: missing weight " + wKey);
    const auto& wShape = wIt->second.shape;
    if (wShape.size() != 2)
        throw std::runtime_error("linear_: weight must be 2D");
    uint32_t N = wShape[0];
    uint32_t K = wShape[1];

    // Flatten x's leading dims to M = numel()/K.
    if (x.numel() % K != 0)
        throw std::runtime_error("linear_: x numel not divisible by K");
    uint32_t M = x.numel() / K;

    // Output shape is always [M, N] — we canonicalize to 2D so downstream
    // shape-preserving ops (layerNorm_, addTensor_) don't accumulate stray dims.
    std::vector<uint32_t> outShape = {M, N};

    uint32_t hasBias = 0;
    WgpuBuffer* biasBuf = nullptr;
    if (!bKey.empty()) {
        auto bIt = weights_.find(bKey);
        if (bIt == weights_.end())
            throw std::runtime_error("linear_: missing bias " + bKey);
        hasBias = 1;
        biasBuf = &bIt->second.buffer();
    }

    LinearParams lp{M, N, K, hasBias};
    linearParamBuf_.write(&lp, sizeof(lp));

    auto out = makeReadbackTensor(renderer_, std::initializer_list<uint32_t>{M, N});
    out.shape = outShape;

    linearPipe_.setUniformBuffer(0,     linearParamBuf_);
    linearPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    linearPipe_.setStorageBufferRead(2, wIt->second.buffer());
    // A bias buffer binding is still required by the shader; reuse w when none.
    linearPipe_.setStorageBufferRead(3, biasBuf ? *biasBuf : wIt->second.buffer());
    linearPipe_.setStorageBuffer(4,     out.buffer());
    linearPipe_.dispatch(divCeil(N, 8), divCeil(M, 8), 1);
    return out;
}

GPUTensor RtDetr::layerNorm_(const GPUTensor& x,
                             const std::string& wKey,
                             const std::string& bKey,
                             float eps) {
    auto wIt = weights_.find(wKey);
    auto bIt = weights_.find(bKey);
    if (wIt == weights_.end() || bIt == weights_.end())
        throw std::runtime_error("layerNorm_: missing weight/bias");
    uint32_t D = wIt->second.numel();
    if (x.numel() % D != 0)
        throw std::runtime_error("layerNorm_: last dim mismatch");
    uint32_t M = x.numel() / D;

    LNParams lp{M, D, eps, 0.f};
    lnParamBuf_.write(&lp, sizeof(lp));

    auto out = makeReadbackTensor(renderer_, std::initializer_list<uint32_t>{M, D});
    // Keep canonical [M, D] shape regardless of input rank.

    layerNormPipe_.setUniformBuffer(0,     lnParamBuf_);
    layerNormPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    layerNormPipe_.setStorageBufferRead(2, wIt->second.buffer());
    layerNormPipe_.setStorageBufferRead(3, bIt->second.buffer());
    layerNormPipe_.setStorageBuffer(4,     out.buffer());
    layerNormPipe_.dispatch(divCeil(M, 64), 1, 1);
    return out;
}

GPUTensor RtDetr::softmaxLast_(const GPUTensor& x) {
    if (x.shape.empty())
        throw std::runtime_error("softmaxLast_: need at least 1D shape");
    uint32_t N = x.shape.back();
    uint32_t M = x.numel() / N;
    SoftmaxParams sp{M, N, 0, 0};
    softmaxParamBuf_.write(&sp, sizeof(sp));

    auto out = makeReadbackTensor(renderer_, std::initializer_list<uint32_t>{M, N});
    out.shape = x.shape;

    softmaxPipe_.setUniformBuffer(0,     softmaxParamBuf_);
    softmaxPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    softmaxPipe_.setStorageBuffer(2,     out.buffer());
    softmaxPipe_.dispatch(divCeil(M, 64), 1, 1);
    return out;
}

GPUTensor RtDetr::gelu_(const GPUTensor& x) {
    uint32_t N = x.numel();
    GeluParams gp{N, 0, 0, 0};
    geluParamBuf_.write(&gp, sizeof(gp));

    auto out = makeReadbackTensor(renderer_, std::initializer_list<uint32_t>{N});
    out.shape = x.shape;

    geluPipe_.setUniformBuffer(0,     geluParamBuf_);
    geluPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    geluPipe_.setStorageBuffer(2,     out.buffer());
    geluPipe_.dispatch(divCeil(N, 64), 1, 1);
    return out;
}

GPUTensor RtDetr::relu_(const GPUTensor& x) {
    uint32_t N = x.numel();
    GeluParams gp{N, 0, 0, 0};   // reuse GeluParams (same layout)
    geluParamBuf_.write(&gp, sizeof(gp));

    auto out = makeReadbackTensor(renderer_, std::initializer_list<uint32_t>{N});
    out.shape = x.shape;

    reluPipe_.setUniformBuffer(0,     geluParamBuf_);
    reluPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    reluPipe_.setStorageBuffer(2,     out.buffer());
    reluPipe_.dispatch(divCeil(N, 64), 1, 1);
    return out;
}

GPUTensor RtDetr::attnScores_(const GPUTensor& qkv, uint32_t H) {
    if (qkv.shape.size() != 2)
        throw std::runtime_error("attnScores_: qkv must be 2D [M, 3*D]");
    uint32_t M = qkv.shape[0];
    uint32_t threeD = qkv.shape[1];
    if (threeD % 3 != 0)
        throw std::runtime_error("attnScores_: second dim not divisible by 3");
    uint32_t D = threeD / 3;
    if (D % H != 0)
        throw std::runtime_error("attnScores_: D not divisible by H");
    uint32_t d = D / H;

    AttnParams ap{M, H, d, threeD, 1.0f / std::sqrt(float(d)), 0, 0, 0};
    attnParamBuf_.write(&ap, sizeof(ap));

    auto out = makeReadbackTensor(renderer_, {H, M, M});
    attnScoresPipe_.setUniformBuffer(0,     attnParamBuf_);
    attnScoresPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(qkv).buffer());
    attnScoresPipe_.setStorageBuffer(2,     out.buffer());
    attnScoresPipe_.dispatch(divCeil(M, 8), divCeil(M, 8), H);
    return out;
}

GPUTensor RtDetr::attnApply_(const GPUTensor& qkv, const GPUTensor& attn, uint32_t H) {
    uint32_t M = qkv.shape[0];
    uint32_t D = qkv.shape[1] / 3;
    uint32_t d = D / H;
    AttnParams ap{M, H, d, qkv.shape[1], 1.0f / std::sqrt(float(d)), 0, 0, 0};
    attnParamBuf_.write(&ap, sizeof(ap));

    auto out = makeReadbackTensor(renderer_, {M, D});
    attnApplyPipe_.setUniformBuffer(0,     attnParamBuf_);
    attnApplyPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(qkv).buffer());
    attnApplyPipe_.setStorageBufferRead(2, const_cast<GPUTensor&>(attn).buffer());
    attnApplyPipe_.setStorageBuffer(3,     out.buffer());
    attnApplyPipe_.dispatch(divCeil(d, 8), divCeil(M, 8), H);
    return out;
}

GPUTensor RtDetr::msDeformAttn_(const GPUTensor& value,
                                const std::vector<std::pair<uint32_t,uint32_t>>& spatialShapes,
                                const GPUTensor& refPts,
                                const GPUTensor& samplingOffsets,
                                const GPUTensor& attnWeights,
                                uint32_t numHeads) {
    // value: [total_tokens, D]  (D = numHeads * headDim)
    // refPts: [Nq, 2]
    // samplingOffsets: [Nq, H*L*P*2]
    // attnWeights: [Nq, H*L*P]  (must already be softmaxed)
    uint32_t D  = value.shape.back();
    uint32_t d  = D / numHeads;
    uint32_t L  = uint32_t(spatialShapes.size());
    uint32_t Nq = refPts.shape[0];

    // Infer P (points per head per level) from attnWeights shape.
    uint32_t HLP = attnWeights.shape.back();
    uint32_t P   = HLP / (numHeads * L);

    // Upload spatial shapes [L*2] and level start offsets [L].
    std::vector<uint32_t> shapesBuf(L * 2);
    std::vector<uint32_t> startsBuf(L);
    uint32_t cumulative = 0;
    for (uint32_t i = 0; i < L; ++i) {
        shapesBuf[i * 2 + 0] = spatialShapes[i].first;   // H
        shapesBuf[i * 2 + 1] = spatialShapes[i].second;  // W
        startsBuf[i] = cumulative;
        cumulative += spatialShapes[i].first * spatialShapes[i].second;
    }

    // Create small GPU buffers for shapes and level starts.
    WgpuBuffer shapeGpu(renderer_, L * 2 * sizeof(uint32_t), WgpuBuffer::Usage::Storage);
    shapeGpu.write(shapesBuf.data(), shapesBuf.size() * sizeof(uint32_t));
    WgpuBuffer startsGpu(renderer_, L * sizeof(uint32_t), WgpuBuffer::Usage::Storage);
    startsGpu.write(startsBuf.data(), startsBuf.size() * sizeof(uint32_t));

    // Uniform params.
    MsDeformParams mp{Nq, numHeads, d, L, P, 0, 0, 0};
    msDeformParamBuf_.write(&mp, sizeof(mp));

    auto out = makeReadbackTensor(renderer_, {Nq, D});

    msDeformAttnPipe_.setUniformBuffer(0,     msDeformParamBuf_);
    msDeformAttnPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(value).buffer());
    msDeformAttnPipe_.setStorageBufferRead(2, shapeGpu);
    msDeformAttnPipe_.setStorageBufferRead(3, startsGpu);
    msDeformAttnPipe_.setStorageBufferRead(4, const_cast<GPUTensor&>(refPts).buffer());
    msDeformAttnPipe_.setStorageBufferRead(5, const_cast<GPUTensor&>(samplingOffsets).buffer());
    msDeformAttnPipe_.setStorageBufferRead(6, const_cast<GPUTensor&>(attnWeights).buffer());
    msDeformAttnPipe_.setStorageBuffer(7,     out.buffer());
    msDeformAttnPipe_.dispatch(divCeil(d, 8), divCeil(Nq, 8), numHeads);
    return out;
}

namespace {
/// Build the 2D sinusoidal positional embedding used by ultralytics AIFI.
/// Output layout matches ultralytics' build_2d_sincos_position_embedding:
/// flat [H*W, embed_dim] where embed_dim = 4 * pos_dim. For each spatial
/// position (gy, gx) and temperature T=10000:
///     posX,posY = gx, gy  (1-indexed grid)
/// omega[i] = 1 / T^(i / pos_dim)  for i in [0, pos_dim)
/// Entry layout per token: [sin(posX*omega), cos(posX*omega),
///                          sin(posY*omega), cos(posY*omega)]
std::vector<float> make2dSinCosPosEmbed(uint32_t H, uint32_t W, uint32_t embedDim,
                                        float temperature = 10000.0f) {
    if (embedDim % 4 != 0)
        throw std::runtime_error("sinCosPosEmbed: embed_dim must be multiple of 4");
    uint32_t posDim = embedDim / 4;
    std::vector<float> omega(posDim);
    for (uint32_t i = 0; i < posDim; ++i)
        omega[i] = 1.0f / std::pow(temperature, float(i) / float(posDim));

    std::vector<float> out(size_t(H) * W * embedDim, 0.0f);
    for (uint32_t gy = 0; gy < H; ++gy) {
        for (uint32_t gx = 0; gx < W; ++gx) {
            // Ultralytics' build_2d_sincos uses meshgrid(arange(w), arange(h), "ij")
            // which makes grid_w = outer loop = y, grid_h = inner loop = x in image
            // coords. Mirror that here so the pos-embed layout matches the trained model.
            float posX = float(gy);
            float posY = float(gx);
            size_t base = (size_t(gy) * W + gx) * embedDim;
            for (uint32_t i = 0; i < posDim; ++i) {
                float ax = posX * omega[i];
                float ay = posY * omega[i];
                out[base + 0 * posDim + i] = std::sin(ax);
                out[base + 1 * posDim + i] = std::cos(ax);
                out[base + 2 * posDim + i] = std::sin(ay);
                out[base + 3 * posDim + i] = std::cos(ay);
            }
        }
    }
    return out;
}

/// Transpose [D, H, W] → [H*W, D] on CPU (reads whole tensor back). Used in
/// AIFI where we need token-major layout for linear/attention kernels.
/// This avoids a dedicated transpose kernel; AIFI's input is only 20x20x256.
}

GPUTensor RtDetr::aifi_(const GPUTensor& x) {
    const uint32_t D = x.C();
    const uint32_t Hs = x.H();
    const uint32_t Ws = x.W();
    const uint32_t M = Hs * Ws;
    const uint32_t numHeads = 8;   // ultralytics RT-DETR-L AIFI default

    // --- 1. Readback + transpose [D, H, W] → [M, D] on CPU ---
    std::vector<float> cpuNchw = readback(const_cast<GPUTensor&>(x).buffer(), x.numel());
    std::vector<float> tokens(size_t(M) * D);
    for (uint32_t c = 0; c < D; ++c) {
        for (uint32_t y = 0; y < Hs; ++y) {
            for (uint32_t w = 0; w < Ws; ++w) {
                tokens[(size_t(y) * Ws + w) * D + c] =
                    cpuNchw[(size_t(c) * Hs + y) * Ws + w];
            }
        }
    }

    // --- 2. Upload tokens and pos embed, add (q/k path = tokens+pos, v = tokens) ---
    auto tok  = uploadTensor({M, D}, tokens.data());
    auto pos  = make2dSinCosPosEmbed(Hs, Ws, D);
    // ultralytics AIFI: q = k = src + pos, v = src (pos added to q,k only)
    std::vector<float> tokPlusPos(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) tokPlusPos[i] = tokens[i] + pos[i];
    auto qk   = uploadTensor({M, D}, tokPlusPos.data());

    // --- 3. Combined QKV projection: we need two linear calls because q=k=qk, v=tok.
    //        Allocate a [M, 3*D] buffer manually and run three linears writing into slices.
    //        Simpler: compute q, k from qk with in_proj (q rows 0..D-1, k rows D..2D-1),
    //        and v from tok (rows 2D..3D-1). We run linear three times and concat on CPU.
    // The in_proj_weight is packed as [3D, D] row-major: rows [0..D) for Q, [D..2D) for K,
    // [2D..3D) for V. We can't easily slice the weight on the fly, so run one linear each
    // by reading back + splitting the weight at load time. For now, readback qk and tok and
    // do full [M,3D] = linear(concat_rows), but that still conflates q=k with v.
    //
    // Cleanest path: compute y_qk = linear(qk, in_proj_weight, in_proj_bias) → [M, 3D]
    //              and y_t  = linear(tok, in_proj_weight, 0)                → [M, 3D]
    // Then build QKV = [y_qk.Q || y_qk.K || y_t.V] on CPU and re-upload.
    // (Doing this on CPU is cheap: M=400, 3D=768 ⇒ 921K floats.)
    auto yqk = linear_(qk,  "model.11.ma.in_proj_weight", "model.11.ma.in_proj_bias");
    auto yt  = linear_(tok, "model.11.ma.in_proj_weight", "model.11.ma.in_proj_bias");
    std::vector<float> yqkCpu = readback(yqk.buffer(), yqk.numel());
    std::vector<float> ytCpu  = readback(yt.buffer(),  yt.numel());
    std::vector<float> qkvCpu(size_t(M) * 3 * D);
    for (uint32_t i = 0; i < M; ++i) {
        size_t srcQ = size_t(i) * 3 * D + 0 * D;
        size_t srcK = size_t(i) * 3 * D + 1 * D;
        size_t srcV = size_t(i) * 3 * D + 2 * D;
        size_t dst  = size_t(i) * 3 * D;
        std::copy(yqkCpu.begin() + srcQ, yqkCpu.begin() + srcQ + D, qkvCpu.begin() + dst + 0*D);
        std::copy(yqkCpu.begin() + srcK, yqkCpu.begin() + srcK + D, qkvCpu.begin() + dst + 1*D);
        std::copy(ytCpu.begin()  + srcV, ytCpu.begin()  + srcV + D, qkvCpu.begin() + dst + 2*D);
    }
    auto qkv = uploadTensor({M, 3 * D}, qkvCpu.data());

    // --- 4. Scores, softmax, apply ---
    auto scores  = attnScores_(qkv, numHeads);
    auto attnSm  = softmaxLast_(scores);
    auto heads   = attnApply_(qkv, attnSm, numHeads);       // [M, D]

    // --- 5. Output projection + residual + norm1 ---
    auto attnOut = linear_(heads, "model.11.ma.out_proj.weight", "model.11.ma.out_proj.bias");
    auto r1      = addTensor_(tok, attnOut);
    auto n1      = layerNorm_(r1, "model.11.norm1.weight", "model.11.norm1.bias");

    // --- 6. FFN: fc1 → GELU → fc2 → residual → norm2 ---
    auto h1 = linear_(n1, "model.11.fc1.weight", "model.11.fc1.bias");
    auto hg = gelu_(h1);
    auto h2 = linear_(hg, "model.11.fc2.weight", "model.11.fc2.bias");
    auto r2 = addTensor_(n1, h2);
    auto n2 = layerNorm_(r2, "model.11.norm2.weight", "model.11.norm2.bias");

    // --- 7. Transpose [M, D] → [D, H, W] on CPU, re-upload ---
    std::vector<float> n2Cpu = readback(n2.buffer(), n2.numel());
    std::vector<float> outNchw(size_t(D) * Hs * Ws);
    for (uint32_t c = 0; c < D; ++c) {
        for (uint32_t y = 0; y < Hs; ++y) {
            for (uint32_t w = 0; w < Ws; ++w) {
                outNchw[(size_t(c) * Hs + y) * Ws + w] =
                    n2Cpu[(size_t(y) * Ws + w) * D + c];
            }
        }
    }
    auto result = makeReadbackTensor(renderer_, {D, Hs, Ws});
    result.buf->write(outNchw.data(), outNchw.size() * sizeof(float));
    return result;
}

GPUTensor RtDetr::addTensor_(const GPUTensor& a, const GPUTensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("addTensor_: size mismatch");
    uint32_t N = a.numel();
    AddParams ap{N, 0, 0, 0};
    addParamBuf_.write(&ap, sizeof(ap));

    auto out = makeReadbackTensorV(renderer_, a.shape);
    addPipe_.setUniformBuffer(0,     addParamBuf_);
    addPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(a).buffer());
    addPipe_.setStorageBufferRead(2, const_cast<GPUTensor&>(b).buffer());
    addPipe_.setStorageBuffer(3,     out.buffer());
    addPipe_.dispatch(divCeil(N, 64), 1, 1);
    return out;
}

GPUTensor RtDetr::concatC_(const GPUTensor& a, const GPUTensor& b) {
    if (a.H() != b.H() || a.W() != b.W())
        throw std::runtime_error("RtDetr::concatC_: H/W mismatch");
    uint32_t cA = a.C(), cB = b.C(), H = a.H(), W = a.W();

    ConcatParams cp{cA, cB, H, W};
    concatParamBuf_.write(&cp, sizeof(cp));

    auto out = makeReadbackTensor(renderer_, {cA + cB, H, W});

    concatCPipe_.setUniformBuffer(0,     concatParamBuf_);
    concatCPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(a).buffer());
    concatCPipe_.setStorageBufferRead(2, const_cast<GPUTensor&>(b).buffer());
    concatCPipe_.setStorageBuffer(3,     out.buffer());
    concatCPipe_.dispatch(divCeil(W, 8), divCeil(H, 8), divCeil(cA + cB, 4));

    return out;
}

GPUTensor RtDetr::addSilu_(const GPUTensor& a, const GPUTensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("addSilu_: size mismatch");
    uint32_t N = a.numel();
    AddParams ap{N, 0, 0, 0};
    addParamBuf_.write(&ap, sizeof(ap));

    auto out = makeReadbackTensorV(renderer_, a.shape);
    addSiluPipe_.setUniformBuffer(0,     addParamBuf_);
    addSiluPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(a).buffer());
    addSiluPipe_.setStorageBufferRead(2, const_cast<GPUTensor&>(b).buffer());
    addSiluPipe_.setStorageBuffer(3,     out.buffer());
    addSiluPipe_.dispatch(divCeil(N, 64), 1, 1);
    return out;
}

GPUTensor RtDetr::buildMemory_(const GPUTensor& p0, const GPUTensor& p1, const GPUTensor& p2) {
    // ultralytics applies valid_mask * feats before enc_output.
    // Anchor at (gy, gx) on a (H, W) level: cx=(gx+0.5)/W, cy=(gy+0.5)/H,
    // w=h=0.05*2^level. Valid iff all four components lie strictly inside
    // (eps, 1-eps) with eps = 1e-2. wh is always valid for levels 0..2
    // with grid_size=0.05, so only the cx/cy edge check matters here.
    constexpr float kEps = 1e-2f;
    auto transposeChw2tokens = [&](const GPUTensor& p, std::vector<float>& out, int level) {
        uint32_t C = p.C(), H = p.H(), W = p.W();
        auto cpu = readback(const_cast<GPUTensor&>(p).buffer(), p.numel());
        size_t base = out.size();
        out.resize(base + size_t(H) * W * C);
        float wh = 0.05f * float(1u << level);
        bool whValid = (wh > kEps) && (wh < 1.f - kEps);
        // CHW -> (HW, C), row-major over (gy, gx).
        for (uint32_t c = 0; c < C; ++c) {
            for (uint32_t gy = 0; gy < H; ++gy) {
                float cy = (float(gy) + 0.5f) / float(H);
                for (uint32_t gx = 0; gx < W; ++gx) {
                    float cx = (float(gx) + 0.5f) / float(W);
                    bool valid = whValid
                               && cx > kEps && cx < 1.f - kEps
                               && cy > kEps && cy < 1.f - kEps;
                    size_t yx = size_t(gy) * W + gx;
                    float v = valid ? cpu[size_t(c) * H * W + yx] : 0.f;
                    out[base + yx * C + c] = v;
                }
            }
        }
    };
    std::vector<float> tokens;
    transposeChw2tokens(p0, tokens, 0);
    transposeChw2tokens(p1, tokens, 1);
    transposeChw2tokens(p2, tokens, 2);

    uint32_t M = uint32_t(p0.H() * p0.W() + p1.H() * p1.W() + p2.H() * p2.W());
    uint32_t D = p0.C();
    GPUTensor out;
    out.shape = {M, D};
    out.buf = std::make_unique<WgpuBuffer>(
        renderer_, size_t(M) * D * sizeof(float),
        WgpuBuffer::Usage::StorageReadback);
    out.buf->write(tokens.data(), tokens.size() * sizeof(float));
    return out;
}

GPUTensor RtDetr::inputProj_(const GPUTensor& x, int scaleIdx) {
    const std::string p = "model.28.input_proj." + std::to_string(scaleIdx);
    return conv_(x, p + ".fused.weight", p + ".fused.bias",
                 1, 1, 0, 0, Activation::None);
}

// ============================================================
//  RTDETRDecoder full forward pass (model.28)
// ============================================================
RtDetr::DecoderOutput RtDetr::decoder_(
        const GPUTensor& memory,
        const GPUTensor& encOutput,
        const std::vector<std::pair<uint32_t,uint32_t>>& spatialShapes) {

    const uint32_t totalTokens = memory.shape[0];
    const uint32_t D = 256;
    const uint32_t numHeads = 8;
    const uint32_t headDim  = D / numHeads;    // 32
    const uint32_t numLevels = uint32_t(spatialShapes.size());
    const uint32_t numPoints = 4;
    const uint32_t numQueries = 300;
    const uint32_t numClasses = 80;
    const uint32_t numLayers  = 6;

    // CPU helpers.
    auto sigmoidF = [](float x) -> float { return 1.0f / (1.0f + std::exp(-x)); };
    auto invSigmoid = [](float x) -> float {
        x = std::clamp(x, 1e-5f, 1.0f - 1e-5f);
        return std::log(x / (1.0f - x));
    };
    auto cpuSoftmax = [](float* row, size_t n) {
        float mx = *std::max_element(row, row + n);
        float s = 0.f;
        for (size_t i = 0; i < n; ++i) { row[i] = std::exp(row[i] - mx); s += row[i]; }
        float inv = 1.f / s;
        for (size_t i = 0; i < n; ++i) row[i] *= inv;
    };

    // ----------------------------------------------------------------
    //  Step 1: enc_score_head → top-K selection
    // ----------------------------------------------------------------
    auto encScores = linear_(encOutput, "model.28.enc_score_head.weight",
                                        "model.28.enc_score_head.bias");
    auto scoresCpu = readback(encScores.buffer(), encScores.numel());
    // [totalTokens, 80]: find max class score per token, pick top-300.
    std::vector<std::pair<float, uint32_t>> maxScores(totalTokens);
    for (uint32_t t = 0; t < totalTokens; ++t) {
        float mx = scoresCpu[size_t(t) * numClasses];
        for (uint32_t c = 1; c < numClasses; ++c)
            mx = std::max(mx, scoresCpu[size_t(t) * numClasses + c]);
        maxScores[t] = {mx, t};
    }
    std::partial_sort(maxScores.begin(), maxScores.begin() + numQueries,
                      maxScores.end(), [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<uint32_t> topkIdx(numQueries);
    for (uint32_t i = 0; i < numQueries; ++i) topkIdx[i] = maxScores[i].second;

    // Gather target [300, 256] from enc_output.
    auto encOutCpu = readback(const_cast<GPUTensor&>(encOutput).buffer(), encOutput.numel());
    std::vector<float> targetCpu(size_t(numQueries) * D);
    for (uint32_t i = 0; i < numQueries; ++i)
        std::copy_n(encOutCpu.data() + size_t(topkIdx[i]) * D, D,
                    targetCpu.data() + size_t(i) * D);

    auto target = uploadTensor({numQueries, D}, targetCpu.data());

    // ----------------------------------------------------------------
    //  Step 2: enc_bbox_head MLP on selected tokens → reference_points
    // ----------------------------------------------------------------
    auto bboxMLP = [&](const GPUTensor& x, const std::string& prefix) -> GPUTensor {
        auto h1 = linear_(x, prefix + ".layers.0.weight", prefix + ".layers.0.bias");
        h1 = relu_(h1);
        auto h2 = linear_(h1, prefix + ".layers.1.weight", prefix + ".layers.1.bias");
        h2 = relu_(h2);
        return linear_(h2, prefix + ".layers.2.weight", prefix + ".layers.2.bias");
    };

    auto encBboxRaw = bboxMLP(target, "model.28.enc_bbox_head");
    auto bboxRawCpu = readback(encBboxRaw.buffer(), encBboxRaw.numel());
    // sigmoid → reference_points [300, 4].
    std::vector<float> refPts(size_t(numQueries) * 4);
    for (size_t i = 0; i < refPts.size(); ++i)
        refPts[i] = sigmoidF(bboxRawCpu[i]);

    std::cout << "  Decoder: top-300 selected, enc_bbox ref_pts ready\n";

    // ----------------------------------------------------------------
    //  Step 3: Decoder layers (×6)
    // ----------------------------------------------------------------
    for (uint32_t layer = 0; layer < numLayers; ++layer) {
        std::string p = "model.28.decoder.layers." + std::to_string(layer);

        // --- 3a: query_pos_head MLP (4 → 512 → 256) ---
        // Shared across layers; uses detached ref_points.
        std::vector<float> refXy(size_t(numQueries) * 4);
        std::copy(refPts.begin(), refPts.end(), refXy.begin());
        auto refTensor = uploadTensor({numQueries, 4u}, refXy.data());

        auto qp1 = linear_(refTensor, "model.28.query_pos_head.layers.0.weight",
                                       "model.28.query_pos_head.layers.0.bias");
        qp1 = relu_(qp1);
        auto queryPos = linear_(qp1, "model.28.query_pos_head.layers.1.weight",
                                      "model.28.query_pos_head.layers.1.bias");

        // --- 3b: Self-attention ---
        // q = k = target + query_pos, v = target
        auto qkInput = addTensor_(target, queryPos);

        // Project through in_proj [768, 256].
        auto yqk = linear_(qkInput, p + ".self_attn.in_proj_weight",
                                     p + ".self_attn.in_proj_bias");
        auto yt  = linear_(target,  p + ".self_attn.in_proj_weight",
                                     p + ".self_attn.in_proj_bias");

        // CPU splice: Q from yqk[:,:D], K from yqk[:,D:2D], V from yt[:,2D:3D].
        auto yqkCpu = readback(yqk.buffer(), yqk.numel());
        auto ytCpu  = readback(yt.buffer(),  yt.numel());
        std::vector<float> qkvCpu(size_t(numQueries) * 3 * D);
        for (uint32_t i = 0; i < numQueries; ++i) {
            size_t s = size_t(i) * 3 * D;
            std::copy_n(yqkCpu.data() + s,           D, qkvCpu.data() + s);
            std::copy_n(yqkCpu.data() + s + D,       D, qkvCpu.data() + s + D);
            std::copy_n(ytCpu.data()  + s + 2 * D,   D, qkvCpu.data() + s + 2 * D);
        }
        auto qkv = uploadTensor({numQueries, 3 * D}, qkvCpu.data());

        auto scores  = attnScores_(qkv, numHeads);
        auto attnSm  = softmaxLast_(scores);
        auto heads   = attnApply_(qkv, attnSm, numHeads);

        auto attnOut = linear_(heads, p + ".self_attn.out_proj.weight",
                                      p + ".self_attn.out_proj.bias");
        target = addTensor_(target, attnOut);
        target = layerNorm_(target, p + ".norm1.weight", p + ".norm1.bias");

        // --- 3c: Cross-attention (MSDeformAttn) ---
        auto crossQuery = addTensor_(target, queryPos);

        // Value projection (per-layer).
        auto value = linear_(memory, p + ".cross_attn.value_proj.weight",
                                     p + ".cross_attn.value_proj.bias");

        // Sampling offsets + attention weights.
        auto rawOffsets = linear_(crossQuery, p + ".cross_attn.sampling_offsets.weight",
                                              p + ".cross_attn.sampling_offsets.bias");
        auto rawAttnW   = linear_(crossQuery, p + ".cross_attn.attention_weights.weight",
                                              p + ".cross_attn.attention_weights.bias");

        // Softmax: [300, 96] → treat as [300*8, 12], softmax over 12.
        auto rawAttnCpu = readback(rawAttnW.buffer(), rawAttnW.numel());
        uint32_t LP = numLevels * numPoints;   // 12
        for (uint32_t r = 0; r < numQueries * numHeads; ++r)
            cpuSoftmax(rawAttnCpu.data() + size_t(r) * LP, LP);
        auto attnWSm = uploadTensor({numQueries, numHeads * LP}, rawAttnCpu.data());

        // Preprocess offsets: raw [Nq, H*L*P*2] → pixel-space.
        // PyTorch formula (4-element ref): loc = ref[:2] + raw / P * ref[2:] * 0.5
        // Pixel: pixel = loc * [W, H] - 0.5. My kernel adds ref*W to ox, so:
        // ox = (raw_x / P * ref_w * 0.5) * W_l
        auto rawOffCpu = readback(rawOffsets.buffer(), rawOffsets.numel());
        const uint32_t offStride = numHeads * numLevels * numPoints * 2;
        for (uint32_t q = 0; q < numQueries; ++q) {
            float ref_w = refPts[size_t(q) * 4 + 2];
            float ref_h = refPts[size_t(q) * 4 + 3];
            for (uint32_t h = 0; h < numHeads; ++h) {
                for (uint32_t l = 0; l < numLevels; ++l) {
                    float Wl = float(spatialShapes[l].second);
                    float Hl = float(spatialShapes[l].first);
                    for (uint32_t pt = 0; pt < numPoints; ++pt) {
                        size_t idx = size_t(q) * offStride
                                   + (size_t(h) * numLevels * numPoints + size_t(l) * numPoints + pt) * 2;
                        rawOffCpu[idx + 0] = rawOffCpu[idx + 0] / float(numPoints) * ref_w * 0.5f * Wl;
                        rawOffCpu[idx + 1] = rawOffCpu[idx + 1] / float(numPoints) * ref_h * 0.5f * Hl;
                    }
                }
            }
        }
        auto offsScaled = uploadTensor({numQueries, offStride}, rawOffCpu.data());

        // Reference points [300, 2] (cx, cy only) for the kernel.
        std::vector<float> refXy2(size_t(numQueries) * 2);
        for (uint32_t q = 0; q < numQueries; ++q) {
            refXy2[size_t(q) * 2 + 0] = refPts[size_t(q) * 4 + 0];
            refXy2[size_t(q) * 2 + 1] = refPts[size_t(q) * 4 + 1];
        }
        auto refPtsTensor = uploadTensor({numQueries, 2u}, refXy2.data());

        auto crossOut  = msDeformAttn_(value, spatialShapes, refPtsTensor, offsScaled, attnWSm, numHeads);
        auto crossProj = linear_(crossOut, p + ".cross_attn.output_proj.weight",
                                           p + ".cross_attn.output_proj.bias");
        target = addTensor_(target, crossProj);
        target = layerNorm_(target, p + ".norm2.weight", p + ".norm2.bias");

        // --- 3d: FFN ---
        auto ff1 = linear_(target, p + ".linear1.weight", p + ".linear1.bias");
        auto ffg = gelu_(ff1);
        auto ff2 = linear_(ffg, p + ".linear2.weight", p + ".linear2.bias");
        target = addTensor_(target, ff2);
        target = layerNorm_(target, p + ".norm3.weight", p + ".norm3.bias");

        // --- 3e: Iterative bbox refinement ---
        auto bboxDelta = bboxMLP(target, "model.28.dec_bbox_head." + std::to_string(layer));
        auto deltaCpu = readback(bboxDelta.buffer(), bboxDelta.numel());
        for (uint32_t q = 0; q < numQueries; ++q) {
            for (uint32_t c = 0; c < 4; ++c) {
                size_t i = size_t(q) * 4 + c;
                refPts[i] = sigmoidF(deltaCpu[i] + invSigmoid(refPts[i]));
            }
        }

        std::cout << "  Decoder layer " << layer << " done\n";
    }

    // ----------------------------------------------------------------
    //  Step 4: Final output — last layer's score head + bboxes
    // ----------------------------------------------------------------
    auto finalScores = linear_(target, "model.28.dec_score_head.5.weight",
                                       "model.28.dec_score_head.5.bias");
    auto finalScoresCpu = readback(finalScores.buffer(), finalScores.numel());

    return DecoderOutput{std::move(finalScoresCpu), refPts};
}

GPUTensor RtDetr::upsample2x_(const GPUTensor& x) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    UpsampleParams up{C, H, W, 0};
    upsampleParamBuf_.write(&up, sizeof(up));

    auto out = makeReadbackTensor(renderer_, {C, H * 2, W * 2});
    upsample2xPipe_.setUniformBuffer(0,     upsampleParamBuf_);
    upsample2xPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    upsample2xPipe_.setStorageBuffer(2,     out.buffer());
    upsample2xPipe_.dispatch(divCeil(W * 2, 8), divCeil(H * 2, 8), divCeil(C, 4));
    return out;
}

// RepC3 (ultralytics):
//   a = cv1(x) [Conv1x1 + SiLU]
//   b = cv2(x) [Conv1x1 + SiLU]
//   for i in 0..n-1:
//       a = silu(RepConv(a))    where RepConv = conv3x3 + conv1x1 (both BN-folded, no act)
//   return cv3(a + b)           NOTE: in RT-DETR, cv3 is Identity (c_=c_out), so just a+b.
// ultralytics RepC3 uses Identity cv3 when c_ == c2 (RT-DETR always).
GPUTensor RtDetr::repC3_(const GPUTensor& x, const std::string& prefix, int n) {
    auto a = conv_(x, prefix + ".cv1.fused.weight", prefix + ".cv1.fused.bias",
                   1, 1, 0, 0, Activation::SiLU);
    auto b = conv_(x, prefix + ".cv2.fused.weight", prefix + ".cv2.fused.bias",
                   1, 1, 0, 0, Activation::SiLU);
    for (int i = 0; i < n; ++i) {
        std::string mp = prefix + ".m." + std::to_string(i);
        // RepConv: conv1 is 3x3 BN-folded; conv2 is 1x1 BN-folded; sum + SiLU.
        auto r1 = conv_(a, mp + ".conv1.fused.weight", mp + ".conv1.fused.bias",
                        1, 1, 1, 1, Activation::None);
        auto r2 = conv_(a, mp + ".conv2.fused.weight", mp + ".conv2.fused.bias",
                        1, 1, 0, 0, Activation::None);
        a = addSilu_(r1, r2);
    }
    return addTensor_(a, b);
}

RtDetr::NeckFeatures RtDetr::ccfm_(const GPUTensor& p3, const GPUTensor& p4, const GPUTensor& f5) {
    // model.12: Conv 1x1 256->256 on f5 (AIFI output).
    auto x12 = conv_(f5, "model.12.fused.weight", "model.12.fused.bias",
                     1, 1, 0, 0, Activation::SiLU);
    // model.13: Upsample 2x
    auto u13 = upsample2x_(x12);
    // model.14: Conv 1x1 1024->256 on p4 (lateral).
    // model.14: act=False (reference min ≈ -6.7, well below SiLU floor)
    auto l14 = conv_(p4, "model.14.fused.weight", "model.14.fused.bias",
                     1, 1, 0, 0, Activation::None);
    // model.15: Concat(u13, l14)  -> [512, H/16, W/16]
    auto c15 = concatC_(u13, l14);
    // model.16: RepC3 n=3 512->256
    auto x16 = repC3_(c15, "model.16", 3);
    // model.17: Conv 1x1 256->256
    auto x17 = conv_(x16, "model.17.fused.weight", "model.17.fused.bias",
                     1, 1, 0, 0, Activation::SiLU);
    // model.18: Upsample 2x
    auto u18 = upsample2x_(x17);
    // model.19: Conv 1x1 512->256 on p3 (lateral).
    // model.19: act=False (reference min ≈ -9.3, well below SiLU floor)
    auto l19 = conv_(p3, "model.19.fused.weight", "model.19.fused.bias",
                     1, 1, 0, 0, Activation::None);
    // model.20: Concat(u18, l19) -> [512, H/8, W/8]
    auto c20 = concatC_(u18, l19);
    // model.21: RepC3 n=3 -> S3
    auto s3 = repC3_(c20, "model.21", 3);
    // model.22: Conv 3x3 s=2 256->256 (downsample)
    auto d22 = conv_(s3, "model.22.fused.weight", "model.22.fused.bias",
                     2, 2, 1, 1, Activation::SiLU);
    // model.23: Concat(d22, x17) -> [512, H/16, W/16]
    auto c23 = concatC_(d22, x17);
    // model.24: RepC3 n=3 -> S4
    auto s4 = repC3_(c23, "model.24", 3);
    // model.25: Conv 3x3 s=2 256->256 (downsample)
    auto d25 = conv_(s4, "model.25.fused.weight", "model.25.fused.bias",
                     2, 2, 1, 1, Activation::SiLU);
    // model.26: Concat(d25, x12) -> [512, H/32, W/32]
    auto c26 = concatC_(d25, x12);
    // model.27: RepC3 n=3 -> S5
    auto s5 = repC3_(c26, "model.27", 3);
    return NeckFeatures{std::move(s3), std::move(s4), std::move(s5)};
}

// ============================================================
//  End-to-end inference
// ============================================================
std::vector<RtDetr::Detection> RtDetr::infer(
        const unsigned char* rgba, int srcW, int srcH, float confThresh) {

    const int dstW = INPUT_SIZE, dstH = INPUT_SIZE;

    // --- 1. CPU preprocessing: bilinear resize + /255 → [3, 640, 640] ---
    std::vector<float> chw(size_t(3) * dstH * dstW);
    for (int dy = 0; dy < dstH; ++dy) {
        float sy = (float(dy) + 0.5f) * float(srcH) / float(dstH) - 0.5f;
        int y0 = std::max(0, int(std::floor(sy)));
        int y1 = std::min(srcH - 1, y0 + 1);
        float fy = sy - float(y0);
        for (int dx = 0; dx < dstW; ++dx) {
            float sx = (float(dx) + 0.5f) * float(srcW) / float(dstW) - 0.5f;
            int x0 = std::max(0, int(std::floor(sx)));
            int x1 = std::min(srcW - 1, x0 + 1);
            float fx = sx - float(x0);
            for (int c = 0; c < 3; ++c) {
                float tl = float(rgba[(y0 * srcW + x0) * 4 + c]);
                float tr = float(rgba[(y0 * srcW + x1) * 4 + c]);
                float bl = float(rgba[(y1 * srcW + x0) * 4 + c]);
                float br = float(rgba[(y1 * srcW + x1) * 4 + c]);
                float val = (1 - fy) * ((1 - fx) * tl + fx * tr)
                          + fy       * ((1 - fx) * bl + fx * br);
                chw[size_t(c) * dstH * dstW + size_t(dy) * dstW + dx] = val / 255.0f;
            }
        }
    }
    auto input = uploadTensor({3u, uint32_t(dstH), uint32_t(dstW)}, chw.data());

    // --- 2. Forward pass ---
    auto bp   = backbone_(input);
    auto f5   = aifi_(bp.p5);
    auto neck = ccfm_(bp.p3, bp.p4, f5);
    auto ip0  = inputProj_(neck.s3, 0);
    auto ip1  = inputProj_(neck.s4, 1);
    auto ip2  = inputProj_(neck.s5, 2);
    auto mem  = buildMemory_(ip0, ip1, ip2);
    auto lin  = linear_(mem, "model.28.enc_output.0.weight",
                              "model.28.enc_output.0.bias");
    auto eo   = layerNorm_(lin, "model.28.enc_output.1.weight",
                                 "model.28.enc_output.1.bias");

    std::vector<std::pair<uint32_t,uint32_t>> shapes = {
        {ip0.H(), ip0.W()}, {ip1.H(), ip1.W()}, {ip2.H(), ip2.W()}};

    auto dec = decoder_(mem, eo, shapes);

    // --- 3. Post-processing: sigmoid, threshold, NMS ---
    auto sigmoidF = [](float x) -> float { return 1.0f / (1.0f + std::exp(-x)); };
    auto iou = [](const Detection& a, const Detection& b) -> float {
        float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
        float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
        float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (areaA + areaB - inter + 1e-6f);
    };

    // Convert logits → probabilities, find best class per query.
    std::vector<Detection> candidates;
    for (int q = 0; q < 300; ++q) {
        int bestC = 0;
        float bestP = sigmoidF(dec.scores[size_t(q) * NUM_CLASSES]);
        for (int c = 1; c < NUM_CLASSES; ++c) {
            float p = sigmoidF(dec.scores[size_t(q) * NUM_CLASSES + c]);
            if (p > bestP) { bestP = p; bestC = c; }
        }
        if (bestP < confThresh) continue;

        float cx = dec.bboxes[size_t(q) * 4 + 0];
        float cy = dec.bboxes[size_t(q) * 4 + 1];
        float w  = dec.bboxes[size_t(q) * 4 + 2];
        float h  = dec.bboxes[size_t(q) * 4 + 3];

        // Normalized (cx,cy,w,h) → image pixels (x1,y1,x2,y2).
        float x1 = (cx - w * 0.5f) * float(srcW);
        float y1 = (cy - h * 0.5f) * float(srcH);
        float x2 = (cx + w * 0.5f) * float(srcW);
        float y2 = (cy + h * 0.5f) * float(srcH);
        // Clamp to image bounds.
        x1 = std::clamp(x1, 0.f, float(srcW));
        y1 = std::clamp(y1, 0.f, float(srcH));
        x2 = std::clamp(x2, 0.f, float(srcW));
        y2 = std::clamp(y2, 0.f, float(srcH));

        candidates.push_back({bestC, bestP, x1, y1, x2, y2});
    }

    // NMS: sort by confidence descending, greedily suppress overlaps.
    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.confidence > b.confidence; });
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Detection> results;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        results.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && candidates[j].classId == candidates[i].classId
                && iou(candidates[i], candidates[j]) > 0.7f) {
                suppressed[j] = true;
            }
        }
    }

    return results;
}

// ============================================================
//  GPU readback
// ============================================================
std::vector<float> RtDetr::readback(WgpuBuffer& srcBuf, size_t floatCount) {
    auto* device = static_cast<WGPUDevice>(renderer_.nativeDevice());
    auto* queue  = static_cast<WGPUQueue>(renderer_.nativeQueue());

    size_t byteCount = floatCount * sizeof(float);

    WGPUBufferDescriptor bd{};
    bd.label = WGPUStringView{"rtdetr_staging", WGPU_STRLEN};
    bd.size  = byteCount;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &bd);

    // srcBuf must have CopySrc usage; our dwConv output tensors use Storage,
    // which in this codebase's WgpuBuffer wrapper already includes CopySrc.
    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = WGPUStringView{"rtdetr_readback_enc", WGPU_STRLEN};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, srcBuf.buffer(), 0, staging, 0, byteCount);
    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = WGPUStringView{"rtdetr_readback_cmd", WGPU_STRLEN};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(queue, 1, &cmd);

    struct MapData { bool done = false; WGPUMapAsyncStatus status{}; } md;
    WGPUBufferMapCallbackInfo mapCb{};
    mapCb.mode = WGPUCallbackMode_AllowSpontaneous;
    mapCb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
        auto* d = static_cast<MapData*>(ud1);
        d->status = status;
        d->done   = true;
    };
    mapCb.userdata1 = &md;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, byteCount, mapCb);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!md.done) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("RtDetr::readback: map timed out");
#ifdef __EMSCRIPTEN__
        emscripten_sleep(0);
#else
        wgpuDevicePoll(device, true, nullptr);
#endif
    }

    std::vector<float> result(floatCount, 0.f);
    if (md.status == WGPUMapAsyncStatus_Success) {
        const auto* mapped = static_cast<const float*>(
            wgpuBufferGetConstMappedRange(staging, 0, byteCount));
        std::memcpy(result.data(), mapped, byteCount);
        wgpuBufferUnmap(staging);
    }

    wgpuBufferRelease(staging);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    return result;
}

}// namespace rtdetr
