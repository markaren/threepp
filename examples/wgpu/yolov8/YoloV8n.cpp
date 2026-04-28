/// YoloV8n.cpp — YOLOv8n GPU inference via WGSL compute shaders.
///
/// Architecture: standard YOLOv8n (3.2 M params, 640×640 input, 80 COCO classes).
/// Each operation type has one WgpuComputePipeline instance that is reused for
/// every layer of that type by rebinding resources before each dispatch.

#include "YoloV8n.hpp"
#include "WeightLoader.hpp"

#include <webgpu/webgpu.h>

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#else
#  include <webgpu/wgpu.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace threepp;

// ============================================================
//  GPU param structs (must be 16-byte aligned for uniform buffers)
// ============================================================
namespace {

struct ConvParams {
    uint32_t in_c, out_c, in_h, in_w;
    uint32_t out_h, out_w, k_h, k_w;
    uint32_t stride_h, stride_w, pad_h, pad_w;
    uint32_t has_bias, has_silu, has_residual, _p2;  // 64 bytes
};

struct BnParams {
    uint32_t channels, height, width;
    float    eps;  // 16 bytes
};

struct AddParams {
    uint32_t count, _p0, _p1, _p2;  // 16 bytes
};

struct PoolParams {
    uint32_t channels, in_h, in_w, out_h;
    uint32_t out_w, k, stride, pad;  // 32 bytes
};

struct UpParams {
    uint32_t channels, in_h, in_w, _p;  // 16 bytes
};

struct CatParams {
    uint32_t c_a, c_b, height, width;  // 16 bytes
};

struct SliceParams {
    uint32_t c_in, c_out, offset_c, height;
    uint32_t width, dst_offset_c, _p1, _p2;  // 32 bytes
};

struct DetectParams {
    uint32_t grid_h, grid_w, reg_max, num_classes;
    float    stride, conf_thresh, _p1, _p2;
    uint32_t in_c_box, in_c_cls, max_dets, _p4;   // 48 bytes (also used by fused kernel)
};

// ============================================================
//  WGSL shader sources
// ============================================================

// ---- Conv2D ------------------------------------------------
// Bindings: (0) ConvParams uniform, (1) input r, (2) weight r, (3) bias r, (4) output rw
static const char* kConv2dWgsl = R"WGSL(
struct ConvParams {
    in_c: u32, out_c: u32, in_h: u32, in_w: u32,
    out_h: u32, out_w: u32, k_h: u32, k_w: u32,
    stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
    has_bias: u32, has_silu: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>          p:      ConvParams;
@group(0) @binding(1) var<storage, read>    inp:    array<f32>;
@group(0) @binding(2) var<storage, read>    wt:     array<u32>;   // packed f16 pairs
@group(0) @binding(3) var<storage, read>    bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn load_weight(i: u32) -> f32 {
    let pair = unpack2x16float(wt[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let oc = gid.z;
    if ox >= p.out_w || oy >= p.out_h || oc >= p.out_c { return; }
    var sum = 0.0;
    for (var ic = 0u; ic < p.in_c; ic++) {
        for (var ky = 0u; ky < p.k_h; ky++) {
            for (var kx = 0u; kx < p.k_w; kx++) {
                let iy = i32(oy * p.stride_h + ky) - i32(p.pad_h);
                let ix = i32(ox * p.stride_w + kx) - i32(p.pad_w);
                if iy >= 0 && iy < i32(p.in_h) && ix >= 0 && ix < i32(p.in_w) {
                    let ii = ic * p.in_h * p.in_w + u32(iy) * p.in_w + u32(ix);
                    let wi = oc * p.in_c * p.k_h * p.k_w
                           + ic * p.k_h * p.k_w
                           + ky * p.k_w + kx;
                    sum += inp[ii] * load_weight(wi);
                }
            }
        }
    }
    if p.has_bias != 0u { sum += bias[oc]; }
    if p.has_silu != 0u { sum = sum / (1.0 + exp(-sum)); }
    out[oc * p.out_h * p.out_w + oy * p.out_w + ox] = sum;
}
)WGSL";

// ---- Conv2D specialised: k=3, stride=1, pad=1 --------------
// Same bindings as kConv2dWgsl. Uses workgroup shared memory to cache a
// 10×10 input tile and a 4×9 weight slab per input-channel iteration so
// every multiply-add reads from LDS rather than global memory.
static const char* kConv3x3s1Wgsl = R"WGSL(
struct ConvParams {
    in_c: u32, out_c: u32, in_h: u32, in_w: u32,
    out_h: u32, out_w: u32, k_h: u32, k_w: u32,
    stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
    has_bias: u32, has_silu: u32, has_residual: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>          p:      ConvParams;
@group(0) @binding(1) var<storage, read>    inp:    array<f32>;
@group(0) @binding(2) var<storage, read>    wt:     array<u32>;   // packed f16 pairs
@group(0) @binding(3) var<storage, read>    bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read>    residual: array<f32>; // optional

// Per-thread 2×2 output tiling: 8×8×4 threads cover a 16×16 output × 4 oc block.
// Input tile is 18×18 (= 324 f32) per ic. Each thread loads ~1.3 tile elements
// cooperatively and computes 4 output pixels, amortising the 9 weight loads
// 4× across outputs.
var<workgroup> in_tile: array<f32, 324>;  // 18 × 18
var<workgroup> wt_slab: array<f32, 36>;   // 4 out_channels × 9 kernel weights

fn load_weight(i: u32) -> f32 {
    let pair = unpack2x16float(wt[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let tid = (lid.z * 8u + lid.y) * 8u + lid.x;   // 0..255

    let wg_ox = wid.x * 16u;
    let wg_oy = wid.y * 16u;
    let oc    = wid.z * 4u + lid.z;

    // This thread's 2×2 output block (top-left corner in output space)
    let ox0 = wg_ox + lid.x * 2u;
    let oy0 = wg_oy + lid.y * 2u;

    var s00 = 0.0; var s01 = 0.0;
    var s10 = 0.0; var s11 = 0.0;

    let kstride = p.in_c * 9u;

    for (var ic = 0u; ic < p.in_c; ic++) {
        // Cooperative load: 18×18 = 324 floats with 256 threads → 2 loads per thread max.
        var i = tid;
        loop {
            if i >= 324u { break; }
            let ty = i / 18u;
            let tx = i % 18u;
            let iy = i32(wg_oy + ty) - 1;
            let ix = i32(wg_ox + tx) - 1;
            var v = 0.0;
            if iy >= 0 && iy < i32(p.in_h) && ix >= 0 && ix < i32(p.in_w) {
                v = inp[ic * p.in_h * p.in_w + u32(iy) * p.in_w + u32(ix)];
            }
            in_tile[i] = v;
            i += 256u;
        }
        // Weight slab (4 oc × 9 kernel)
        if tid < 36u {
            let slab_oc = tid / 9u;
            let k_idx   = tid % 9u;
            let wt_oc   = wid.z * 4u + slab_oc;
            if wt_oc < p.out_c {
                wt_slab[tid] = load_weight(wt_oc * kstride + ic * 9u + k_idx);
            } else {
                wt_slab[tid] = 0.0;
            }
        }
        workgroupBarrier();

        // Each thread's 2×2 output block reads a 4×4 input window from the tile.
        // Tile coords: local_x_in_tile = lid.x * 2, local_y_in_tile = lid.y * 2
        // (padding absorbed into the tile's [-1, +1] offset in the loader above).
        let wo = lid.z * 9u;
        let y0 = lid.y * 2u;
        let x0 = lid.x * 2u;

        // Fetch the 4×4 window once into registers.
        let v00 = in_tile[(y0+0u)*18u + x0+0u];
        let v01 = in_tile[(y0+0u)*18u + x0+1u];
        let v02 = in_tile[(y0+0u)*18u + x0+2u];
        let v03 = in_tile[(y0+0u)*18u + x0+3u];
        let v10 = in_tile[(y0+1u)*18u + x0+0u];
        let v11 = in_tile[(y0+1u)*18u + x0+1u];
        let v12 = in_tile[(y0+1u)*18u + x0+2u];
        let v13 = in_tile[(y0+1u)*18u + x0+3u];
        let v20 = in_tile[(y0+2u)*18u + x0+0u];
        let v21 = in_tile[(y0+2u)*18u + x0+1u];
        let v22 = in_tile[(y0+2u)*18u + x0+2u];
        let v23 = in_tile[(y0+2u)*18u + x0+3u];
        let v30 = in_tile[(y0+3u)*18u + x0+0u];
        let v31 = in_tile[(y0+3u)*18u + x0+1u];
        let v32 = in_tile[(y0+3u)*18u + x0+2u];
        let v33 = in_tile[(y0+3u)*18u + x0+3u];

        let w0 = wt_slab[wo+0u]; let w1 = wt_slab[wo+1u]; let w2 = wt_slab[wo+2u];
        let w3 = wt_slab[wo+3u]; let w4 = wt_slab[wo+4u]; let w5 = wt_slab[wo+5u];
        let w6 = wt_slab[wo+6u]; let w7 = wt_slab[wo+7u]; let w8 = wt_slab[wo+8u];

        // out(0,0): uses (0..3, 0..3)
        s00 += v00*w0 + v01*w1 + v02*w2
             + v10*w3 + v11*w4 + v12*w5
             + v20*w6 + v21*w7 + v22*w8;
        // out(0,1): uses (0..3, 1..4)
        s01 += v01*w0 + v02*w1 + v03*w2
             + v11*w3 + v12*w4 + v13*w5
             + v21*w6 + v22*w7 + v23*w8;
        // out(1,0): uses (1..4, 0..3)
        s10 += v10*w0 + v11*w1 + v12*w2
             + v20*w3 + v21*w4 + v22*w5
             + v30*w6 + v31*w7 + v32*w8;
        // out(1,1): uses (1..4, 1..4)
        s11 += v11*w0 + v12*w1 + v13*w2
             + v21*w3 + v22*w4 + v23*w5
             + v31*w6 + v32*w7 + v33*w8;

        workgroupBarrier();
    }

    if oc >= p.out_c { return; }

    let hw = p.out_h * p.out_w;
    let b  = select(0.0, bias[oc], p.has_bias != 0u);

    // Write each of the 2×2 outputs with bias + optional SiLU + residual.
    // Helper (inlined per output below): y = sum + bias; silu; + residual; store.
    if ox0 + 0u < p.out_w && oy0 + 0u < p.out_h {
        var v = s00 + b;
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        let idx = oc * hw + (oy0 + 0u) * p.out_w + (ox0 + 0u);
        if p.has_residual != 0u { v += residual[idx]; }
        out[idx] = v;
    }
    if ox0 + 1u < p.out_w && oy0 + 0u < p.out_h {
        var v = s01 + b;
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        let idx = oc * hw + (oy0 + 0u) * p.out_w + (ox0 + 1u);
        if p.has_residual != 0u { v += residual[idx]; }
        out[idx] = v;
    }
    if ox0 + 0u < p.out_w && oy0 + 1u < p.out_h {
        var v = s10 + b;
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        let idx = oc * hw + (oy0 + 1u) * p.out_w + (ox0 + 0u);
        if p.has_residual != 0u { v += residual[idx]; }
        out[idx] = v;
    }
    if ox0 + 1u < p.out_w && oy0 + 1u < p.out_h {
        var v = s11 + b;
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        let idx = oc * hw + (oy0 + 1u) * p.out_w + (ox0 + 1u);
        if p.has_residual != 0u { v += residual[idx]; }
        out[idx] = v;
    }
}
)WGSL";

// ---- Conv2D specialised: k=1, stride=1, pad=0 --------------
// Each thread computes K=4 output channels for ONE (ox,oy), reusing each
// input-channel load across 4 accumulators. Weight bandwidth is naturally
// amortised over a cache line because adjacent oc weights are consecutive.
static const char* kConv1x1Wgsl = R"WGSL(
struct ConvParams {
    in_c: u32, out_c: u32, in_h: u32, in_w: u32,
    out_h: u32, out_w: u32, k_h: u32, k_w: u32,
    stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
    has_bias: u32, has_silu: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>          p:      ConvParams;
@group(0) @binding(1) var<storage, read>    inp:    array<f32>;
@group(0) @binding(2) var<storage, read>    wt:     array<u32>;
@group(0) @binding(3) var<storage, read>    bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn load_weight(i: u32) -> f32 {
    let pair = unpack2x16float(wt[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y;
    let oc_base = wid.z * 4u;
    if ox >= p.out_w || oy >= p.out_h { return; }

    let in_pitch = p.in_h * p.in_w;
    let in_off   = oy * p.in_w + ox;
    let kstride  = p.in_c;   // k_h*k_w == 1

    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;

    for (var ic = 0u; ic < p.in_c; ic++) {
        let v = inp[ic * in_pitch + in_off];
        s0 += v * load_weight((oc_base + 0u) * kstride + ic);
        s1 += v * load_weight((oc_base + 1u) * kstride + ic);
        s2 += v * load_weight((oc_base + 2u) * kstride + ic);
        s3 += v * load_weight((oc_base + 3u) * kstride + ic);
    }

    let out_pitch = p.out_h * p.out_w;
    let out_off   = oy * p.out_w + ox;

    if oc_base + 0u < p.out_c {
        var v = s0;
        if p.has_bias != 0u { v += bias[oc_base + 0u]; }
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        out[(oc_base + 0u) * out_pitch + out_off] = v;
    }
    if oc_base + 1u < p.out_c {
        var v = s1;
        if p.has_bias != 0u { v += bias[oc_base + 1u]; }
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        out[(oc_base + 1u) * out_pitch + out_off] = v;
    }
    if oc_base + 2u < p.out_c {
        var v = s2;
        if p.has_bias != 0u { v += bias[oc_base + 2u]; }
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        out[(oc_base + 2u) * out_pitch + out_off] = v;
    }
    if oc_base + 3u < p.out_c {
        var v = s3;
        if p.has_bias != 0u { v += bias[oc_base + 3u]; }
        if p.has_silu != 0u { v = v / (1.0 + exp(-v)); }
        out[(oc_base + 3u) * out_pitch + out_off] = v;
    }
}
)WGSL";

// ---- Conv2D specialised: k=3, stride=2, pad=1 --------------
// Workgroup covers 8×8 output × 4 oc. Input tile is 17×17 per ic (= 289 f32,
// 1156 bytes per ic). Cooperative load in 256-thread workgroup: ~2 loads per
// thread on early iterations.
static const char* kConv3x3s2Wgsl = R"WGSL(
struct ConvParams {
    in_c: u32, out_c: u32, in_h: u32, in_w: u32,
    out_h: u32, out_w: u32, k_h: u32, k_w: u32,
    stride_h: u32, stride_w: u32, pad_h: u32, pad_w: u32,
    has_bias: u32, has_silu: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>          p:      ConvParams;
@group(0) @binding(1) var<storage, read>    inp:    array<f32>;
@group(0) @binding(2) var<storage, read>    wt:     array<u32>;
@group(0) @binding(3) var<storage, read>    bias:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn load_weight(i: u32) -> f32 {
    let pair = unpack2x16float(wt[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

var<workgroup> in_tile: array<f32, 289>;  // 17 × 17
var<workgroup> wt_slab: array<f32, 36>;   // 4 oc × 9 kernel

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let oc = gid.z;
    let tid = (lid.z * 8u + lid.y) * 8u + lid.x;

    // Top-left input coord for this workgroup's 8×8 output region.
    let wg_ox = wid.x * 8u;
    let wg_oy = wid.y * 8u;
    let in_base_x = i32(wg_ox * 2u) - 1;   // stride=2, pad=1
    let in_base_y = i32(wg_oy * 2u) - 1;

    var sum = 0.0;
    let kstride = p.in_c * 9u;

    for (var ic = 0u; ic < p.in_c; ic++) {
        // Cooperative load: 17×17 = 289 floats, with 256 threads → 2 loads per thread max.
        var i = tid;
        loop {
            if i >= 289u { break; }
            let ty = i / 17u;
            let tx = i % 17u;
            let iy = in_base_y + i32(ty);
            let ix = in_base_x + i32(tx);
            var v = 0.0;
            if iy >= 0 && iy < i32(p.in_h) && ix >= 0 && ix < i32(p.in_w) {
                v = inp[ic * p.in_h * p.in_w + u32(iy) * p.in_w + u32(ix)];
            }
            in_tile[i] = v;
            i += 256u;
        }
        if tid < 36u {
            let slab_oc = tid / 9u;
            let k_idx   = tid % 9u;
            let wt_oc   = wid.z * 4u + slab_oc;
            if wt_oc < p.out_c {
                wt_slab[tid] = load_weight(wt_oc * kstride + ic * 9u + k_idx);
            } else {
                wt_slab[tid] = 0.0;
            }
        }
        workgroupBarrier();

        if ox < p.out_w && oy < p.out_h && oc < p.out_c {
            // For output (lid.x, lid.y), input base is (lid.x*2, lid.y*2).
            let y0 = lid.y * 2u;
            let x0 = lid.x * 2u;
            let wo = lid.z * 9u;
            sum += in_tile[(y0+0u)*17u + x0+0u] * wt_slab[wo+0u];
            sum += in_tile[(y0+0u)*17u + x0+1u] * wt_slab[wo+1u];
            sum += in_tile[(y0+0u)*17u + x0+2u] * wt_slab[wo+2u];
            sum += in_tile[(y0+1u)*17u + x0+0u] * wt_slab[wo+3u];
            sum += in_tile[(y0+1u)*17u + x0+1u] * wt_slab[wo+4u];
            sum += in_tile[(y0+1u)*17u + x0+2u] * wt_slab[wo+5u];
            sum += in_tile[(y0+2u)*17u + x0+0u] * wt_slab[wo+6u];
            sum += in_tile[(y0+2u)*17u + x0+1u] * wt_slab[wo+7u];
            sum += in_tile[(y0+2u)*17u + x0+2u] * wt_slab[wo+8u];
        }
        workgroupBarrier();
    }

    if ox < p.out_w && oy < p.out_h && oc < p.out_c {
        if p.has_bias != 0u { sum += bias[oc]; }
        if p.has_silu != 0u { sum = sum / (1.0 + exp(-sum)); }
        out[oc * p.out_h * p.out_w + oy * p.out_w + ox] = sum;
    }
}
)WGSL";

// ---- GPU preprocess: bilinear resize + normalise RGBA u8 → f32 NCHW
// Bindings: (0) PrepParams, (1) packed RGBA storage (u32 per pixel), (2) output [3,H,W] f32
static const char* kPreprocessWgsl = R"WGSL(
struct PrepParams {
    src_w: u32, src_h: u32, dst_w: u32, dst_h: u32,
    inv_scale: f32, pad_x: f32, pad_y: f32, _p2: u32,
}
@group(0) @binding(0) var<uniform>          p:   PrepParams;
@group(0) @binding(1) var<storage, read>    src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn load_rgba(x: u32, y: u32) -> vec3<f32> {
    let packed = src[y * p.src_w + x];
    let r = f32(packed & 0xFFu);
    let g = f32((packed >> 8u) & 0xFFu);
    let b = f32((packed >> 16u) & 0xFFu);
    return vec3<f32>(r, g, b) * (1.0 / 255.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dx = gid.x; let dy = gid.y;
    if dx >= p.dst_w || dy >= p.dst_h { return; }

    // Letterbox: map dst pixel → src pixel, padding with gray (114/255 ~ 0.4471).
    let fx = ((f32(dx) + 0.5) - p.pad_x) * p.inv_scale - 0.5;
    let fy = ((f32(dy) + 0.5) - p.pad_y) * p.inv_scale - 0.5;

    let hw = p.dst_h * p.dst_w;
    let idx = dy * p.dst_w + dx;

    // If sampling position falls outside the source image, write gray padding.
    // Tight clamp: reject as soon as we'd sample past the last full pixel, so
    // the edge row/column of real content is not blended with the gray pad.
    if fx < 0.0 || fy < 0.0 || fx > f32(p.src_w) - 1.0 || fy > f32(p.src_h) - 1.0 {
        dst[0u * hw + idx] = 0.4470588;
        dst[1u * hw + idx] = 0.4470588;
        dst[2u * hw + idx] = 0.4470588;
        return;
    }

    let x0 = u32(clamp(i32(floor(fx)), 0, i32(p.src_w) - 1));
    let y0 = u32(clamp(i32(floor(fy)), 0, i32(p.src_h) - 1));
    let x1 = min(x0 + 1u, p.src_w - 1u);
    let y1 = min(y0 + 1u, p.src_h - 1u);
    let tx = fx - floor(fx);
    let ty = fy - floor(fy);

    let v00 = load_rgba(x0, y0);
    let v01 = load_rgba(x1, y0);
    let v10 = load_rgba(x0, y1);
    let v11 = load_rgba(x1, y1);
    let v = v00 * (1.0 - tx) * (1.0 - ty)
          + v01 * tx         * (1.0 - ty)
          + v10 * (1.0 - tx) * ty
          + v11 * tx         * ty;

    dst[0u * hw + idx] = v.x;
    dst[1u * hw + idx] = v.y;
    dst[2u * hw + idx] = v.z;
}
)WGSL";

// ---- BatchNorm + SiLU (fused) ------------------------------
// Bindings: (0) BnParams, (1) input r, (2) mean r, (3) var r, (4) gamma r, (5) beta r, (6) output rw
static const char* kBnSiluWgsl = R"WGSL(
struct BnParams { channels: u32, height: u32, width: u32, eps: f32 }
@group(0) @binding(0) var<uniform>          p:      BnParams;
@group(0) @binding(1) var<storage, read>    inp:    array<f32>;
@group(0) @binding(2) var<storage, read>    mean:   array<f32>;
@group(0) @binding(3) var<storage, read>    var_:   array<f32>;
@group(0) @binding(4) var<storage, read>    gamma:  array<f32>;
@group(0) @binding(5) var<storage, read>    beta:   array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if x >= p.width || y >= p.height || c >= p.channels { return; }
    let idx = c * p.height * p.width + y * p.width + x;
    let v = inp[idx];
    let norm = (v - mean[c]) / sqrt(var_[c] + p.eps) * gamma[c] + beta[c];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    out[idx] = norm / (1.0 + exp(-norm));
}
)WGSL";

// ---- Element-wise Add --------------------------------------
// Bindings: (0) AddParams {count}, (1) a r, (2) b r, (3) out rw
static const char* kAddWgsl = R"WGSL(
struct AddParams { count: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform>          p:   AddParams;
@group(0) @binding(1) var<storage, read>    a:   array<f32>;
@group(0) @binding(2) var<storage, read>    b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.count { return; }
    out[i] = a[i] + b[i];
}
)WGSL";

// ---- MaxPool2D ---------------------------------------------
// Bindings: (0) PoolParams, (1) input r, (2) output rw
static const char* kMaxpoolWgsl = R"WGSL(
struct PoolParams {
    channels: u32, in_h: u32, in_w: u32, out_h: u32,
    out_w: u32, k: u32, stride: u32, pad: u32,
}
@group(0) @binding(0) var<uniform>          p:   PoolParams;
@group(0) @binding(1) var<storage, read>    inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let c = gid.z;
    if ox >= p.out_w || oy >= p.out_h || c >= p.channels { return; }
    var mx = -1.0e38;
    for (var ky = 0u; ky < p.k; ky++) {
        for (var kx = 0u; kx < p.k; kx++) {
            let iy = i32(oy * p.stride + ky) - i32(p.pad);
            let ix = i32(ox * p.stride + kx) - i32(p.pad);
            if iy >= 0 && iy < i32(p.in_h) && ix >= 0 && ix < i32(p.in_w) {
                mx = max(mx, inp[c * p.in_h * p.in_w + u32(iy) * p.in_w + u32(ix)]);
            }
        }
    }
    out[c * p.out_h * p.out_w + oy * p.out_w + ox] = mx;
}
)WGSL";

// ---- Upsample 2× nearest -----------------------------------
// Bindings: (0) UpParams {channels, in_h, in_w}, (1) input r, (2) output rw
static const char* kUpsampleWgsl = R"WGSL(
struct UpParams { channels: u32, in_h: u32, in_w: u32, _p: u32 }
@group(0) @binding(0) var<uniform>          p:   UpParams;
@group(0) @binding(1) var<storage, read>    inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let c = gid.z;
    let out_h = p.in_h * 2u;
    let out_w = p.in_w * 2u;
    if ox >= out_w || oy >= out_h || c >= p.channels { return; }
    let ix = ox / 2u;
    let iy = oy / 2u;
    out[c * out_h * out_w + oy * out_w + ox] =
        inp[c * p.in_h * p.in_w + iy * p.in_w + ix];
}
)WGSL";

// ---- Channel Concatenation (2 tensors, same H/W) -----------
// Bindings: (0) CatParams {c_a, c_b, height, width}, (1) A r, (2) B r, (3) out rw
static const char* kConcatWgsl = R"WGSL(
struct CatParams { c_a: u32, c_b: u32, height: u32, width: u32 }
@group(0) @binding(0) var<uniform>          p:   CatParams;
@group(0) @binding(1) var<storage, read>    a:   array<f32>;
@group(0) @binding(2) var<storage, read>    b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    let total_c = p.c_a + p.c_b;
    if x >= p.width || y >= p.height || c >= total_c { return; }
    let hw  = p.height * p.width;
    let ofs = y * p.width + x;
    if c < p.c_a {
        out[c * hw + ofs] = a[c * hw + ofs];
    } else {
        let bc = c - p.c_a;
        out[c * hw + ofs] = b[bc * hw + ofs];
    }
}
)WGSL";

// ---- Channel Slice (extract `c_out` channels starting at `offset_c`) ------
// Bindings: (0) SliceParams, (1) input r, (2) output rw
static const char* kSliceWgsl = R"WGSL(
struct SliceParams {
    c_in: u32, c_out: u32, offset_c: u32, height: u32,
    width: u32, dst_offset_c: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<uniform>          p:   SliceParams;
@group(0) @binding(1) var<storage, read>    inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let c = gid.z;
    if x >= p.width || y >= p.height || c >= p.c_out { return; }
    let hw  = p.height * p.width;
    let ofs = y * p.width + x;
    out[(c + p.dst_offset_c) * hw + ofs] = inp[(c + p.offset_c) * hw + ofs];
}
)WGSL";

// ---- Detect Decode (DFL + anchor-free box decode) ----------
// box_raw shape: [4*reg_max, grid_h, grid_w]   (ltrb order, 16 bins each)
// cls_raw shape: [num_classes, grid_h, grid_w]
// dets   shape: [grid_h * grid_w * 6]          (x1,y1,x2,y2,conf,cls_id)
static const char* kDetectWgsl = R"WGSL(
struct DetectParams {
    grid_h: u32, grid_w: u32, reg_max: u32, num_classes: u32,
    stride: f32, conf_thresh: f32, _p1: f32, _p2: f32,
    in_c_box: u32, in_c_cls: u32, max_dets: u32, _p4: u32,
}
@group(0) @binding(0) var<uniform>          p:       DetectParams;
@group(0) @binding(1) var<storage, read>    box_raw: array<f32>;
@group(0) @binding(2) var<storage, read>    cls_raw: array<f32>;
@group(0) @binding(3) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> dets:    array<f32>;

fn dfl_side(anchor: u32, hw: u32, side: u32) -> f32 {
    var max_v = -1.0e38;
    for (var i = 0u; i < p.reg_max; i++) {
        max_v = max(max_v, box_raw[(side * p.reg_max + i) * hw + anchor]);
    }
    var sum_e = 0.0;
    var wsum  = 0.0;
    for (var i = 0u; i < p.reg_max; i++) {
        let e = exp(box_raw[(side * p.reg_max + i) * hw + anchor] - max_v);
        sum_e += e;
        wsum  += e * f32(i);
    }
    return wsum / sum_e;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x; let gy = gid.y;
    if gx >= p.grid_w || gy >= p.grid_h { return; }

    let hw         = p.grid_h * p.grid_w;
    let anchor_idx = gy * p.grid_w + gx;

    // DFL decode: 4 sides (l, t, r, b)
    let dl = dfl_side(anchor_idx, hw, 0u);
    let dt = dfl_side(anchor_idx, hw, 1u);
    let dr = dfl_side(anchor_idx, hw, 2u);
    let db = dfl_side(anchor_idx, hw, 3u);

    // Anchor-free: centre at (gx+0.5, gy+0.5) × stride
    let cx = (f32(gx) + 0.5) * p.stride;
    let cy = (f32(gy) + 0.5) * p.stride;

    // Best class
    var max_cls = -1.0e38;
    var cls_id  = 0u;
    for (var i = 0u; i < p.num_classes; i++) {
        let logit = cls_raw[i * hw + anchor_idx];
        if logit > max_cls { max_cls = logit; cls_id = i; }
    }

    let conf = 1.0 / (1.0 + exp(-max_cls));
    if conf < p.conf_thresh { return; }

    let slot = atomicAdd(&counter, 1u);
    if slot >= p.max_dets { return; }   // overflow guard

    let base = slot * 6u;
    dets[base + 0u] = cx - dl * p.stride;
    dets[base + 1u] = cy - dt * p.stride;
    dets[base + 2u] = cx + dr * p.stride;
    dets[base + 3u] = cy + db * p.stride;
    dets[base + 4u] = conf;
    dets[base + 5u] = f32(cls_id);
}
)WGSL";

// ---- Detect head fused: inlines cv2.i.2 (k=1 → 4*reg_max) and
// cv3.i.2 (k=1 → num_classes) into the decode kernel. Each thread handles
// one anchor and computes box_raw/cls_raw via on-the-fly k=1 matvecs
// against the cv2.i.1 / cv3.i.1 feature maps.
static const char* kDetectFusedWgsl = R"WGSL(
struct DetectParams {
    grid_h: u32, grid_w: u32, reg_max: u32, num_classes: u32,
    stride: f32, _p0: f32, _p1: f32, _p2: f32,
    in_c_box: u32, in_c_cls: u32, _p3: u32, _p4: u32,
}
@group(0) @binding(0) var<uniform>          p:       DetectParams;
@group(0) @binding(1) var<storage, read>    feat_box: array<f32>;   // cv2.i.1 output
@group(0) @binding(2) var<storage, read>    feat_cls: array<f32>;   // cv3.i.1 output
@group(0) @binding(3) var<storage, read>    w_box:    array<u32>;   // packed f16
@group(0) @binding(4) var<storage, read>    b_box:    array<f32>;
@group(0) @binding(5) var<storage, read>    w_cls:    array<u32>;   // packed f16
@group(0) @binding(6) var<storage, read>    b_cls:    array<f32>;
@group(0) @binding(7) var<storage, read_write> dets:  array<f32>;

fn load_w_box(i: u32) -> f32 {
    let pair = unpack2x16float(w_box[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}
fn load_w_cls(i: u32) -> f32 {
    let pair = unpack2x16float(w_cls[i >> 1u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x; let gy = gid.y;
    if gx >= p.grid_w || gy >= p.grid_h { return; }

    let hw         = p.grid_h * p.grid_w;
    let anchor_idx = gy * p.grid_w + gx;

    // --- DFL: 4 sides (l, t, r, b), each with reg_max bins ---
    var dist: array<f32, 4>;
    let box_oc = 4u * p.reg_max;   // = 64
    for (var side = 0u; side < 4u; side = side + 1u) {
        // Compute reg_max raw logits via k=1 conv: out[oc] = Σ_ic feat[ic]*W[oc,ic] + b[oc]
        var logits: array<f32, 16>;                // reg_max == 16
        var max_v = -1.0e38;
        for (var i = 0u; i < p.reg_max; i = i + 1u) {
            let oc = side * p.reg_max + i;
            var s = b_box[oc];
            let w_base = oc * p.in_c_box;
            for (var ic = 0u; ic < p.in_c_box; ic = ic + 1u) {
                s = s + feat_box[ic * hw + anchor_idx] * load_w_box(w_base + ic);
            }
            logits[i] = s;
            if s > max_v { max_v = s; }
        }
        var sum_e = 0.0;
        var wsum  = 0.0;
        for (var i = 0u; i < p.reg_max; i = i + 1u) {
            let e = exp(logits[i] - max_v);
            sum_e = sum_e + e;
            wsum  = wsum + e * f32(i);
        }
        dist[side] = wsum / sum_e;
    }

    let cx = (f32(gx) + 0.5) * p.stride;
    let cy = (f32(gy) + 0.5) * p.stride;

    // --- Class: argmax over inlined k=1 conv (no need to materialise all logits) ---
    var max_cls = -1.0e38;
    var cls_id  = 0u;
    for (var c = 0u; c < p.num_classes; c = c + 1u) {
        var s = b_cls[c];
        let w_base = c * p.in_c_cls;
        for (var ic = 0u; ic < p.in_c_cls; ic = ic + 1u) {
            s = s + feat_cls[ic * hw + anchor_idx] * load_w_cls(w_base + ic);
        }
        if s > max_cls { max_cls = s; cls_id = c; }
    }

    let base = anchor_idx * 6u;
    dets[base + 0u] = cx - dist[0] * p.stride;
    dets[base + 1u] = cy - dist[1] * p.stride;
    dets[base + 2u] = cx + dist[2] * p.stride;
    dets[base + 3u] = cy + dist[3] * p.stride;
    dets[base + 4u] = 1.0 / (1.0 + exp(-max_cls));
    dets[base + 5u] = f32(cls_id);
}
)WGSL";

// ============================================================
//  Inline dispatch helpers
// ============================================================
static uint32_t divCeil(uint32_t n, uint32_t d) { return (n + d - 1) / d; }

}// anonymous namespace

// ============================================================
//  Constructor / Destructor
// ============================================================
namespace yolo {

YoloV8n::YoloV8n(WgpuRenderer& r)
    : renderer_(r),
      convPipe_(r, kConv2dWgsl,   "main"),
      conv3x3s1Pipe_(r, kConv3x3s1Wgsl, "main"),
      conv3x3s2Pipe_(r, kConv3x3s2Wgsl, "main"),
      conv1x1Pipe_(r, kConv1x1Wgsl, "main"),
      bnSiluPipe_(r, kBnSiluWgsl, "main"),
      addPipe_(r, kAddWgsl,       "main"),
      maxpoolPipe_(r, kMaxpoolWgsl,"main"),
      upsamplePipe_(r, kUpsampleWgsl,"main"),
      concatPipe_(r, kConcatWgsl, "main"),
      slicePipe_(r, kSliceWgsl,   "main"),
      detectPipe_(r, kDetectWgsl, "main"),
      detectFusedPipe_(r, kDetectFusedWgsl, "main"),
      preprocessPipe_(r, kPreprocessWgsl, "main"),
      convParamBuf_(r, sizeof(ConvParams)),
      bnParamBuf_(r,   sizeof(BnParams)),
      addParamBuf_(r,  sizeof(AddParams)),
      poolParamBuf_(r, sizeof(PoolParams)),
      upParamBuf_(r,   sizeof(UpParams)),
      catParamBuf_(r,  sizeof(CatParams)),
      sliceParamBuf_(r,sizeof(SliceParams)),
      detectParamBuf_(r,sizeof(DetectParams)),
      prepParamBuf_(r, 32),
      dummyBias_(r, 4, WgpuBuffer::Usage::Storage)
{
    // Zero-initialise the dummy bias buffer
    float zero = 0.f;
    dummyBias_.write(&zero, 4);

    // Shared detection counter + det-list buffer for the GPU pre-filter path.
    // Sized for MAX_DETS surviving anchors across all 3 scales.
    detCounterBuf_ = std::make_unique<WgpuBuffer>(
        r, 4, WgpuBuffer::Usage::StorageReadback);
    detDetsBuf_ = std::make_unique<WgpuBuffer>(
        r, MAX_DETS * 6 * sizeof(float), WgpuBuffer::Usage::StorageReadback);
}

YoloV8n::~YoloV8n() = default;

// ============================================================
//  f32 → IEEE 754 binary16 (half) conversion. Handles normal range,
//  flushes sub-normals to zero (below ~6e-5). Adequate for YOLOv8n weights
//  which sit comfortably in the normal f16 range.
// ============================================================
namespace {
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = int32_t((x >> 23) & 0xFFu) - 127;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp == 128) {                                 // inf / nan
        return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp > 15) return uint16_t(sign | 0x7C00u);    // overflow → inf
    if (exp < -14) return uint16_t(sign);             // underflow → 0 (flush subnorm)

    uint32_t he = uint32_t(exp + 15) & 0x1Fu;
    uint32_t hm = mant >> 13;
    return uint16_t(sign | (he << 10) | hm);
}
}  // namespace

// ============================================================
//  Weight loading
// ============================================================
void YoloV8n::loadWeights(const std::string& path) {
    auto w = parseWeightBinary(path);

    // Fold BatchNorm into preceding Conv weights (offline).
    //   w'[oc,ic,kh,kw] = w[oc,ic,kh,kw] * gamma[oc] / sqrt(var[oc] + eps)
    //   b'[oc]          = beta[oc] - mean[oc] * gamma[oc] / sqrt(var[oc] + eps)
    // Stored under keys "<prefix>.fused.weight" / "<prefix>.fused.bias".
    const float eps = 1e-5f;
    std::vector<std::string> convKeys;
    for (auto& kv : w.data)
        if (kv.first.size() > 12 &&
            kv.first.compare(kv.first.size() - 12, 12, ".conv.weight") == 0)
            convKeys.push_back(kv.first);

    for (auto& convKey : convKeys) {
        std::string prefix = convKey.substr(0, convKey.size() - 12);   // drop ".conv.weight"
        std::string bnW = prefix + ".bn.weight";
        std::string bnB = prefix + ".bn.bias";
        std::string bnM = prefix + ".bn.running_mean";
        std::string bnV = prefix + ".bn.running_var";
        if (!w.data.count(bnW) || !w.data.count(bnB) ||
            !w.data.count(bnM) || !w.data.count(bnV)) continue;

        const auto& wt    = w.data.at(convKey);
        const auto& gamma = w.data.at(bnW);
        const auto& beta  = w.data.at(bnB);
        const auto& mean  = w.data.at(bnM);
        const auto& var_  = w.data.at(bnV);
        const auto& sh    = w.shapes.at(convKey);   // [oc, ic, kh, kw]
        uint32_t oc = sh[0];
        uint32_t per = wt.size() / oc;              // ic*kh*kw

        std::vector<float> fw(wt.size());
        std::vector<float> fb(oc);
        for (uint32_t c = 0; c < oc; ++c) {
            float s = gamma[c] / std::sqrt(var_[c] + eps);
            fb[c] = beta[c] - mean[c] * s;
            for (uint32_t i = 0; i < per; ++i)
                fw[c * per + i] = wt[c * per + i] * s;
        }

        w.data[prefix + ".fused.weight"]   = std::move(fw);
        w.shapes[prefix + ".fused.weight"] = sh;
        w.data[prefix + ".fused.bias"]     = std::move(fb);
        w.shapes[prefix + ".fused.bias"]   = {oc};
    }

    weights_.clear();
    for (auto& [name, data] : w.data) {
        auto& sh = w.shapes.at(name);

        // 4D "*.weight" tensors are conv kernels (including fused ones).
        // Pack as f16 pairs into u32 for halved bandwidth in the conv shaders.
        bool isConvWeight = (sh.size() == 4 &&
                             name.size() >= 7 &&
                             name.compare(name.size() - 7, 7, ".weight") == 0);

        if (isConvWeight) {
            size_t n = data.size();
            if (n == 0 || (n % 2) != 0) {
                // Fallback to f32 if we can't pack cleanly
                auto t = makeTensorV(renderer_, sh);
                t.upload(data.data());
                weights_.emplace(name, std::move(t));
                continue;
            }
            std::vector<uint32_t> packed(n / 2);
            for (size_t i = 0; i < n; i += 2) {
                uint16_t lo = f32_to_f16(data[i]);
                uint16_t hi = f32_to_f16(data[i + 1]);
                packed[i / 2] = uint32_t(lo) | (uint32_t(hi) << 16);
            }
            auto t = makeF16WeightTensor(renderer_, sh);
            t.buf->write(packed.data(), packed.size() * sizeof(uint32_t));
            weights_.emplace(name, std::move(t));
        } else {
            auto t = makeTensorV(renderer_, sh);
            t.upload(data.data());
            weights_.emplace(name, std::move(t));
        }
    }
}

// ============================================================
//  Low-level GPU operations
// ============================================================

GPUTensor YoloV8n::conv_(const GPUTensor& x,
                          const std::string& weightKey,
                          const std::string& biasKey,
                          int strideH, int strideW,
                          bool silu,
                          const GPUTensor* residual) {
    auto& wt = weights_.at(weightKey);
    // Weight shape: [out_c, in_c, k_h, k_w]
    uint32_t out_c  = wt.shape[0];
    uint32_t in_c   = wt.shape[1];
    uint32_t k_h    = wt.shape[2];
    uint32_t k_w    = wt.shape[3];
    uint32_t in_h   = x.H();
    uint32_t in_w   = x.W();
    uint32_t pad_h  = (k_h == 1) ? 0u : k_h / 2;
    uint32_t pad_w  = (k_w == 1) ? 0u : k_w / 2;
    uint32_t out_h  = (in_h + 2*pad_h - k_h) / uint32_t(strideH) + 1;
    uint32_t out_w  = (in_w + 2*pad_w - k_w) / uint32_t(strideW) + 1;

    bool hasBias = !biasKey.empty() && weights_.count(biasKey);

    ConvParams cp{};
    cp.in_c=in_c; cp.out_c=out_c; cp.in_h=in_h; cp.in_w=in_w;
    cp.out_h=out_h; cp.out_w=out_w; cp.k_h=k_h; cp.k_w=k_w;
    cp.stride_h=uint32_t(strideH); cp.stride_w=uint32_t(strideW);
    cp.pad_h=pad_h; cp.pad_w=pad_w; cp.has_bias=hasBias?1u:0u;
    cp.has_silu=silu?1u:0u;
    cp.has_residual = residual ? 1u : 0u;
    convParamBuf_.write(&cp, sizeof(cp));

    auto out = makeTensor(renderer_, {out_c, out_h, out_w});

    WgpuBuffer& biasBuf = hasBias ? weights_.at(biasKey).buffer() : dummyBias_;

    // Pick the most specialised shader available for this conv shape.
    WgpuComputePipeline* pipe;
    uint32_t dispatchX, dispatchY, dispatchZ;
    if (k_h == 3 && k_w == 3 && strideH == 1 && strideW == 1) {
        pipe = &conv3x3s1Pipe_;
        // Each thread in this pipeline computes a 2×2 output block → 16×16 per workgroup.
        dispatchX = divCeil(out_w,16); dispatchY = divCeil(out_h,16); dispatchZ = divCeil(out_c,4);
    } else if (k_h == 3 && k_w == 3 && strideH == 2 && strideW == 2) {
        pipe = &conv3x3s2Pipe_;
        dispatchX = divCeil(out_w,8); dispatchY = divCeil(out_h,8); dispatchZ = divCeil(out_c,4);
    } else if (k_h == 1 && k_w == 1 && strideH == 1 && strideW == 1) {
        pipe = &conv1x1Pipe_;
        dispatchX = divCeil(out_w,8); dispatchY = divCeil(out_h,8); dispatchZ = divCeil(out_c,4);
    } else {
        pipe = &convPipe_;
        dispatchX = divCeil(out_w,8); dispatchY = divCeil(out_h,8); dispatchZ = divCeil(out_c,4);
    }

    pipe->setUniformBuffer(0,     convParamBuf_);
    pipe->setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    pipe->setStorageBufferRead(2, wt.buffer());
    pipe->setStorageBufferRead(3, biasBuf);
    pipe->setStorageBuffer(4,     out.buffer());
    // Only conv3x3s1Pipe_ declares binding(5); always bind something on it so
    // the bind group is complete (dummy when no residual).
    if (pipe == &conv3x3s1Pipe_) {
        WgpuBuffer& resBuf = residual ? const_cast<GPUTensor*>(residual)->buffer() : dummyBias_;
        pipe->setStorageBufferRead(5, resBuf);
    }
    pipe->dispatch(dispatchX, dispatchY, dispatchZ);

    return out;
}

GPUTensor YoloV8n::bnSilu_(const GPUTensor& x, const std::string& bnPrefix) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    BnParams bp{C, H, W, 1e-5f};
    bnParamBuf_.write(&bp, sizeof(bp));

    auto out = makeTensor(renderer_, {C, H, W});

    bnSiluPipe_.setUniformBuffer(0,     bnParamBuf_);
    bnSiluPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    bnSiluPipe_.setStorageBufferRead(2, weights_.at(bnPrefix + ".running_mean").buffer());
    bnSiluPipe_.setStorageBufferRead(3, weights_.at(bnPrefix + ".running_var").buffer());
    bnSiluPipe_.setStorageBufferRead(4, weights_.at(bnPrefix + ".weight").buffer());
    bnSiluPipe_.setStorageBufferRead(5, weights_.at(bnPrefix + ".bias").buffer());
    bnSiluPipe_.setStorageBuffer(6,     out.buffer());
    bnSiluPipe_.dispatch(divCeil(W,8), divCeil(H,8), divCeil(C,4));

    return out;
}

GPUTensor YoloV8n::add_(const GPUTensor& a, const GPUTensor& b) {
    uint32_t n = a.numel();
    AddParams ap{n, 0, 0, 0};
    addParamBuf_.write(&ap, sizeof(ap));

    auto out = makeTensorV(renderer_, a.shape);

    addPipe_.setUniformBuffer(0,     addParamBuf_);
    addPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(a).buffer());
    addPipe_.setStorageBufferRead(2, const_cast<GPUTensor&>(b).buffer());
    addPipe_.setStorageBuffer(3,     out.buffer());
    addPipe_.dispatch(divCeil(n, 256));

    return out;
}

GPUTensor YoloV8n::maxpool_(const GPUTensor& x, int k, int stride, int pad) {
    uint32_t C   = x.C(), H = x.H(), W = x.W();
    uint32_t outH = (H + 2*uint32_t(pad) - uint32_t(k)) / uint32_t(stride) + 1;
    uint32_t outW = (W + 2*uint32_t(pad) - uint32_t(k)) / uint32_t(stride) + 1;

    PoolParams pp{C, H, W, outH, outW, uint32_t(k), uint32_t(stride), uint32_t(pad)};
    poolParamBuf_.write(&pp, sizeof(pp));

    auto out = makeTensor(renderer_, {C, outH, outW});

    maxpoolPipe_.setUniformBuffer(0,     poolParamBuf_);
    maxpoolPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    maxpoolPipe_.setStorageBuffer(2,     out.buffer());
    maxpoolPipe_.dispatch(divCeil(outW,8), divCeil(outH,8), divCeil(C,4));

    return out;
}

GPUTensor YoloV8n::upsample2x_(const GPUTensor& x) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    UpParams up{C, H, W, 0};
    upParamBuf_.write(&up, sizeof(up));

    auto out = makeTensor(renderer_, {C, H*2, W*2});

    upsamplePipe_.setUniformBuffer(0,     upParamBuf_);
    upsamplePipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    upsamplePipe_.setStorageBuffer(2,     out.buffer());
    upsamplePipe_.dispatch(divCeil(W*2,8), divCeil(H*2,8), divCeil(C,4));

    return out;
}

GPUTensor YoloV8n::concat_(const GPUTensor& a, const GPUTensor& b) {
    assert(a.H() == b.H() && a.W() == b.W());
    uint32_t ca = a.C(), cb = b.C(), H = a.H(), W = a.W();
    CatParams cp{ca, cb, H, W};
    catParamBuf_.write(&cp, sizeof(cp));

    auto out = makeTensor(renderer_, {ca+cb, H, W});

    concatPipe_.setUniformBuffer(0,     catParamBuf_);
    concatPipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(a).buffer());
    concatPipe_.setStorageBufferRead(2, const_cast<GPUTensor&>(b).buffer());
    concatPipe_.setStorageBuffer(3,     out.buffer());
    concatPipe_.dispatch(divCeil(W,8), divCeil(H,8), divCeil(ca+cb,4));

    return out;
}

GPUTensor YoloV8n::sliceChannels_(const GPUTensor& x, uint32_t offsetC, uint32_t countC) {
    uint32_t H = x.H(), W = x.W();
    SliceParams sp{x.C(), countC, offsetC, H, W, /*dst_offset_c=*/0u, 0u, 0u};
    sliceParamBuf_.write(&sp, sizeof(sp));

    auto out = makeTensor(renderer_, {countC, H, W});

    slicePipe_.setUniformBuffer(0,     sliceParamBuf_);
    slicePipe_.setStorageBufferRead(1, const_cast<GPUTensor&>(x).buffer());
    slicePipe_.setStorageBuffer(2,     out.buffer());
    slicePipe_.dispatch(divCeil(W,8), divCeil(H,8), divCeil(countC,4));

    return out;
}

// ============================================================
//  Compound block helpers
// ============================================================

GPUTensor YoloV8n::convBnSilu_(const GPUTensor& x, const std::string& prefix, int stride,
                                const GPUTensor* residual) {
    // BN is folded into weights at load time → single fused conv+bias+SiLU dispatch.
    return conv_(x, prefix + ".fused.weight", prefix + ".fused.bias",
                 stride, stride, /*silu=*/true, residual);
}

GPUTensor YoloV8n::plainConv_(const GPUTensor& x, const std::string& prefix) {
    // Plain Conv2d: weight at prefix+".weight", bias at prefix+".bias"
    return conv_(x, prefix + ".weight", prefix + ".bias", 1, 1);
}

GPUTensor YoloV8n::concatMany_(std::vector<GPUTensor>& parts) {
    assert(!parts.empty());
    uint32_t H = parts[0].H(), W = parts[0].W();
    uint32_t totalC = 0;
    for (auto& p : parts) { assert(p.H() == H && p.W() == W); totalC += p.C(); }
    auto out = makeTensor(renderer_, {totalC, H, W});

    // One slice-copy dispatch per part, writing directly to its slot in `out`.
    // This replaces the old chain of concat_(concat_(...)) which re-copied all
    // previously concatenated channels at every step.
    uint32_t dstOffset = 0;
    for (auto& p : parts) {
        SliceParams sp{p.C(), p.C(), /*src_offset_c=*/0u, H, W,
                       /*dst_offset_c=*/dstOffset, 0u, 0u};
        sliceParamBuf_.write(&sp, sizeof(sp));
        slicePipe_.setUniformBuffer(0,     sliceParamBuf_);
        slicePipe_.setStorageBufferRead(1, p.buffer());
        slicePipe_.setStorageBuffer(2,     out.buffer());
        slicePipe_.dispatch(divCeil(W,8), divCeil(H,8), divCeil(p.C(),4));
        dstOffset += p.C();
    }
    return out;
}

// View-based concat: each view is a sub-channel-range of some source buffer.
// Avoids materialising standalone slice tensors before the concat.
GPUTensor YoloV8n::concatViews_(const std::vector<SourceView>& views,
                                 uint32_t H, uint32_t W) {
    uint32_t totalC = 0;
    for (auto& v : views) totalC += v.countC;
    auto out = makeTensor(renderer_, {totalC, H, W});

    uint32_t dstOffset = 0;
    for (auto& v : views) {
        SliceParams sp{v.srcTotalC, v.countC, v.srcOffsetC, H, W,
                       /*dst_offset_c=*/dstOffset, 0u, 0u};
        sliceParamBuf_.write(&sp, sizeof(sp));
        slicePipe_.setUniformBuffer(0,     sliceParamBuf_);
        slicePipe_.setStorageBufferRead(1, *v.buf);
        slicePipe_.setStorageBuffer(2,     out.buffer());
        slicePipe_.dispatch(divCeil(W,8), divCeil(H,8), divCeil(v.countC,4));
        dstOffset += v.countC;
    }
    return out;
}

GPUTensor YoloV8n::c2f_(const GPUTensor& x, const std::string& prefix, int n, bool shortcut) {
    // cv1: Conv(cin, 2*cmid, k=1) + BN + SiLU
    auto cv1out = convBnSilu_(x, prefix + ".cv1");
    uint32_t cmid = cv1out.C() / 2;
    uint32_t H = cv1out.H(), W = cv1out.W();

    // Bottleneck sub-chain: starts from "second half" of cv1out. Materialise
    // that half once (one slice dispatch) and chain the bottleneck convs
    // through it. The first half is NOT sliced — it's kept as a view into
    // cv1out and copied into the final cv2 input directly at concat time.
    std::vector<GPUTensor> bottleneckOuts;
    bottleneckOuts.reserve(size_t(n));   // stable storage for `cur` below
    GPUTensor half1 = sliceChannels_(cv1out, cmid, cmid);

    GPUTensor* cur = &half1;
    for (int i = 0; i < n; ++i) {
        std::string mi = prefix + ".m." + std::to_string(i);
        auto y = convBnSilu_(*cur, mi + ".cv1");
        // Fused residual add into cv2's k=3-s=1 write-back when shortcut=true.
        const GPUTensor* res = shortcut ? cur : nullptr;
        y = convBnSilu_(y, mi + ".cv2", 1, res);
        bottleneckOuts.push_back(std::move(y));
        cur = &bottleneckOuts.back();
    }

    // Build cv2 input via direct views: half0 of cv1out, half1 of cv1out,
    // then every bottleneck output. One output allocation, (2+n) dispatches.
    std::vector<SourceView> views;
    views.reserve(2 + bottleneckOuts.size());
    views.push_back({&cv1out.buffer(), cv1out.C(), 0u,   cmid});  // half0
    views.push_back({&cv1out.buffer(), cv1out.C(), cmid, cmid});  // half1
    for (auto& bo : bottleneckOuts) views.push_back({&bo.buffer(), bo.C(), 0u, bo.C()});

    auto combined = concatViews_(views, H, W);
    return convBnSilu_(combined, prefix + ".cv2");
}

GPUTensor YoloV8n::sppf_(const GPUTensor& x, const std::string& prefix) {
    // cv1: halve channels
    auto y  = convBnSilu_(x, prefix + ".cv1");
    // Three consecutive max-pools with k=5, s=1, p=2
    auto y1 = maxpool_(y,  5, 1, 2);
    auto y2 = maxpool_(y1, 5, 1, 2);
    auto y3 = maxpool_(y2, 5, 1, 2);
    // Concat all four
    std::vector<GPUTensor> parts;
    parts.push_back(std::move(y));
    parts.push_back(std::move(y1));
    parts.push_back(std::move(y2));
    parts.push_back(std::move(y3));
    auto combined = concatMany_(parts);
    return convBnSilu_(combined, prefix + ".cv2");
}

// ============================================================
//  Detect head
// ============================================================

void YoloV8n::decodeScaleGpu_(const GPUTensor& feat, int scaleIdx, float stride,
                               float confThresh) {
    std::string idx = std::to_string(scaleIdx);
    std::string cv2pfx = "model.22.cv2." + idx;
    std::string cv3pfx = "model.22.cv3." + idx;

    auto bx = convBnSilu_(feat, cv2pfx + ".0");
    bx = convBnSilu_(bx, cv2pfx + ".1");
    bx = plainConv_(bx, cv2pfx + ".2");

    auto cx = convBnSilu_(feat, cv3pfx + ".0");
    cx = convBnSilu_(cx, cv3pfx + ".1");
    cx = plainConv_(cx, cv3pfx + ".2");

    uint32_t gH = feat.H(), gW = feat.W();

    DetectParams dp{gH, gW, uint32_t(REG_MAX), uint32_t(NUM_CLASSES),
                    stride, confThresh, 0.f, 0.f,
                    0u, 0u, MAX_DETS, 0u};
    detectParamBuf_.write(&dp, sizeof(dp));

    detectPipe_.setUniformBuffer(0,     detectParamBuf_);
    detectPipe_.setStorageBufferRead(1, bx.buffer());
    detectPipe_.setStorageBufferRead(2, cx.buffer());
    detectPipe_.setStorageBuffer(3,     *detCounterBuf_);
    detectPipe_.setStorageBuffer(4,     *detDetsBuf_);
    detectPipe_.dispatch(divCeil(gW,8), divCeil(gH,8), 1);
}

std::vector<Detection> YoloV8n::nms_(std::vector<Detection>& dets, float iouThresh) {
    // Sort by confidence descending
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){ return a.conf > b.conf; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> out;

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        out.push_back(dets[i]);
        const auto& di = dets[i];
        float area_i = (di.x2 - di.x1) * (di.y2 - di.y1);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            const auto& dj = dets[j];
            float ix1 = std::max(di.x1, dj.x1);
            float iy1 = std::max(di.y1, dj.y1);
            float ix2 = std::min(di.x2, dj.x2);
            float iy2 = std::min(di.y2, dj.y2);
            float iw  = std::max(0.f, ix2 - ix1);
            float ih  = std::max(0.f, iy2 - iy1);
            float inter = iw * ih;
            float area_j = (dj.x2 - dj.x1) * (dj.y2 - dj.y1);
            float iou = inter / (area_i + area_j - inter + 1e-6f);
            if (iou > iouThresh) suppressed[j] = true;
        }
    }
    return out;
}

// ============================================================
//  GPU Readback
// ============================================================

std::vector<float> YoloV8n::readback_(WgpuBuffer& srcBuf, size_t floatCount) {
    auto* device = static_cast<WGPUDevice>(renderer_.nativeDevice());
    auto* queue  = static_cast<WGPUQueue>(renderer_.nativeQueue());

    size_t byteCount = floatCount * sizeof(float);

    WGPUBufferDescriptor bd{};
    bd.label = WGPUStringView{"yolo_staging", WGPU_STRLEN};
    bd.size  = byteCount;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer stagingBuf = wgpuDeviceCreateBuffer(device, &bd);

    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = WGPUStringView{"yolo_readback_enc", WGPU_STRLEN};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

    wgpuCommandEncoderCopyBufferToBuffer(encoder,
        srcBuf.buffer(), 0,
        stagingBuf, 0,
        byteCount);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = WGPUStringView{"yolo_readback_cmd", WGPU_STRLEN};
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
    wgpuBufferMapAsync(stagingBuf, WGPUMapMode_Read, 0, byteCount, mapCb);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!md.done) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("YoloV8n::readback_: GPU map timed out");
#ifdef __EMSCRIPTEN__
        emscripten_sleep(0);
#else
        wgpuDevicePoll(device, true, nullptr);
#endif
    }

    std::vector<float> result(floatCount, 0.f);
    if (md.status == WGPUMapAsyncStatus_Success) {
        const auto* mapped = static_cast<const float*>(
            wgpuBufferGetConstMappedRange(stagingBuf, 0, byteCount));
        std::memcpy(result.data(), mapped, byteCount);
        wgpuBufferUnmap(stagingBuf);
    }

    wgpuBufferRelease(stagingBuf);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    return result;
}

std::vector<std::vector<float>> YoloV8n::readbackMany_(
        std::vector<std::pair<WgpuBuffer*, size_t>> sources) {
    auto* device = static_cast<WGPUDevice>(renderer_.nativeDevice());
    auto* queue  = static_cast<WGPUQueue>(renderer_.nativeQueue());

    size_t n = sources.size();
    std::vector<WGPUBuffer> staging(n);
    std::vector<size_t> bytes(n);

    // Create staging buffers + issue all copies in one encoder
    WGPUCommandEncoderDescriptor encDesc{};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
    for (size_t i = 0; i < n; ++i) {
        bytes[i] = sources[i].second * sizeof(float);
        WGPUBufferDescriptor bd{};
        bd.size  = bytes[i];
        bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        staging[i] = wgpuDeviceCreateBuffer(device, &bd);
        wgpuCommandEncoderCopyBufferToBuffer(encoder,
            sources[i].first->buffer(), 0,
            staging[i], 0, bytes[i]);
    }
    WGPUCommandBufferDescriptor cmdDesc{};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(queue, 1, &cmd);

    // Kick off async maps for all staging buffers, then poll once
    struct MapData { bool done = false; WGPUMapAsyncStatus status{}; };
    std::vector<MapData> mds(n);
    for (size_t i = 0; i < n; ++i) {
        WGPUBufferMapCallbackInfo mapCb{};
        mapCb.mode = WGPUCallbackMode_AllowSpontaneous;
        mapCb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
            auto* d = static_cast<MapData*>(ud1);
            d->status = status;
            d->done   = true;
        };
        mapCb.userdata1 = &mds[i];
        wgpuBufferMapAsync(staging[i], WGPUMapMode_Read, 0, bytes[i], mapCb);
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool allDone = false;
    while (!allDone) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("YoloV8n::readbackMany_: GPU map timed out");
#ifdef __EMSCRIPTEN__
        emscripten_sleep(0);
#else
        wgpuDevicePoll(device, true, nullptr);
#endif
        allDone = true;
        for (auto& md : mds) if (!md.done) { allDone = false; break; }
    }

    std::vector<std::vector<float>> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i].assign(sources[i].second, 0.f);
        if (mds[i].status == WGPUMapAsyncStatus_Success) {
            const auto* mapped = static_cast<const float*>(
                wgpuBufferGetConstMappedRange(staging[i], 0, bytes[i]));
            std::memcpy(results[i].data(), mapped, bytes[i]);
            wgpuBufferUnmap(staging[i]);
        }
        wgpuBufferRelease(staging[i]);
    }
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    return results;
}

// ============================================================
//  Preprocessing
// ============================================================

GPUTensor YoloV8n::preprocess_(const unsigned char* rgba, int srcW, int srcH) {
    const int dstW = INPUT_SIZE, dstH = INPUT_SIZE;

    // Upload the raw RGBA bytes as u32-per-pixel to a storage buffer.
    // The source image data is already laid out as [srcH*srcW*4] u8 which
    // is byte-identical to u32-per-pixel on little-endian hosts.
    size_t srcBytes = size_t(srcW) * size_t(srcH) * 4;
    WgpuBuffer srcBuf(renderer_, srcBytes, WgpuBuffer::Usage::Storage);
    srcBuf.write(rgba, srcBytes);

    // Letterbox: scale preserving aspect, center, pad with gray.
    float scale = std::min(float(dstW) / float(srcW), float(dstH) / float(srcH));
    float newW  = float(srcW) * scale;
    float newH  = float(srcH) * scale;
    float padX  = (float(dstW) - newW) * 0.5f;
    float padY  = (float(dstH) - newH) * 0.5f;

    // Remember for box un-letterbox in infer().
    lbScale_ = scale;
    lbPadX_  = padX;
    lbPadY_  = padY;

    struct {
        uint32_t src_w, src_h, dst_w, dst_h;
        float    inv_scale, pad_x, pad_y;
        uint32_t _p2;
    } pp{uint32_t(srcW), uint32_t(srcH), uint32_t(dstW), uint32_t(dstH),
         1.0f / scale, padX, padY, 0u};
    prepParamBuf_.write(&pp, sizeof(pp));

    auto out = makeTensor(renderer_, {3u, uint32_t(dstH), uint32_t(dstW)});

    preprocessPipe_.setUniformBuffer(0,     prepParamBuf_);
    preprocessPipe_.setStorageBufferRead(1, srcBuf);
    preprocessPipe_.setStorageBuffer(2,     out.buffer());
    preprocessPipe_.dispatch(divCeil(uint32_t(dstW), 16), divCeil(uint32_t(dstH), 16), 1);

    return out;
}

// ============================================================
//  Main inference
// ============================================================

std::vector<Detection> YoloV8n::infer(const unsigned char* rgba,
                                       int width, int height,
                                       float confThresh, float iouThresh) {
    static int inferCallCount = 0;
    bool traceStages = (++inferCallCount == 2);   // trace only the first warm call
    auto* device = static_cast<WGPUDevice>(renderer_.nativeDevice());
    auto* queue  = static_cast<WGPUQueue>(renderer_.nativeQueue());
    auto syncGpu = [&]() {
        // Submit empty command buffer + poll to force completion of pending work.
        WGPUCommandEncoderDescriptor ed{};
        auto enc = wgpuDeviceCreateCommandEncoder(device, &ed);
        WGPUCommandBufferDescriptor cd{};
        auto cb = wgpuCommandEncoderFinish(enc, &cd);
        wgpuQueueSubmit(queue, 1, &cb);
#ifndef __EMSCRIPTEN__
        wgpuDevicePoll(device, true, nullptr);
#endif
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
    };
    using clk = std::chrono::steady_clock;
    auto stage = clk::now();
    auto lap = [&](const char* label) {
        if (!traceStages) return;
        syncGpu();
        auto now = clk::now();
        std::cout << "    [" << label << "] "
                  << std::chrono::duration<double, std::milli>(now - stage).count()
                  << " ms\n";
        stage = now;
    };

    // ---- Backbone ----
    auto x  = preprocess_(rgba, width, height);          // [3,640,640]
    lap("preprocess");
    x  = convBnSilu_(x,  "model.0", 2);                  // [16,320,320]
    x  = convBnSilu_(x,  "model.1", 2);                  // [32,160,160]
    x  = c2f_(x,         "model.2", 1, true);            // backbone: shortcut=True
    x  = convBnSilu_(x,  "model.3", 2);                  // [64, 80, 80]
    auto feat3 = c2f_(x, "model.4", 2, true);
    x  = convBnSilu_(feat3, "model.5", 2);               // [128,40, 40]
    auto feat4 = c2f_(x, "model.6", 2, true);
    x  = convBnSilu_(feat4, "model.7", 2);               // [256,20, 20]
    x  = c2f_(x,         "model.8", 1, true);
    auto feat5 = sppf_(x, "model.9");                    // [256,20, 20]  P5
    lap("backbone");

    // ---- Neck (PAN-FPN) ---- shortcut=False for all head C2f
    x       = upsample2x_(feat5);                        // [256,40, 40]
    x       = concat_(x, feat4);                         // [384,40, 40]
    auto n4 = c2f_(x, "model.12", 1, false);

    x       = upsample2x_(n4);                           // [128,80, 80]
    x       = concat_(x, feat3);                         // [192,80, 80]
    auto h3 = c2f_(x, "model.15", 1, false);             // P3 head

    x       = convBnSilu_(h3,  "model.16", 2);           // [64, 40, 40]
    x       = concat_(x, n4);                            // [192,40, 40]
    auto h4 = c2f_(x, "model.18", 1, false);             // P4 head

    x       = convBnSilu_(h4,  "model.19", 2);           // [128,20, 20]
    x       = concat_(x, feat5);                         // [384,20, 20]
    auto h5 = c2f_(x, "model.21", 1, false);             // P5 head
    lap("neck");

    // ---- Detect head (model.22): GPU pre-filter; one tiny readback ----
    uint32_t zero = 0;
    detCounterBuf_->write(&zero, sizeof(zero));

    struct Scale { GPUTensor* feat; int idx; float stride; };
    Scale scales[3] = {{&h3, 0, 8.f}, {&h4, 1, 16.f}, {&h5, 2, 32.f}};
    for (int i = 0; i < 3; ++i)
        decodeScaleGpu_(*scales[i].feat, scales[i].idx, scales[i].stride, confThresh);
    lap("detect head (GPU)");

    std::vector<std::pair<WgpuBuffer*, size_t>> sources = {
        {detCounterBuf_.get(), 1u},
        {detDetsBuf_.get(),    size_t(MAX_DETS) * 6u},
    };
    auto raws = readbackMany_(sources);
    lap("readback");

    uint32_t count;
    std::memcpy(&count, raws[0].data(), sizeof(uint32_t));
    if (count > MAX_DETS) count = MAX_DETS;

    std::vector<Detection> all;
    all.reserve(count);
    const auto& raw = raws[1];
    for (uint32_t i = 0; i < count; ++i) {
        Detection d;
        d.x1 = raw[i*6 + 0]; d.y1 = raw[i*6 + 1];
        d.x2 = raw[i*6 + 2]; d.y2 = raw[i*6 + 3];
        d.conf = raw[i*6 + 4];
        d.cls_id = static_cast<int>(raw[i*6 + 5]);
        if (d.x2 > d.x1 && d.y2 > d.y1) all.push_back(d);
    }
    auto out = nms_(all, iouThresh);

    // Un-letterbox: boxes are in 640×640 padded space; map back to original-image pixels.
    const float invS = 1.0f / lbScale_;
    for (auto& d : out) {
        d.x1 = (d.x1 - lbPadX_) * invS;
        d.y1 = (d.y1 - lbPadY_) * invS;
        d.x2 = (d.x2 - lbPadX_) * invS;
        d.y2 = (d.y2 - lbPadY_) * invS;
    }
    return out;
}

}// namespace yolo
