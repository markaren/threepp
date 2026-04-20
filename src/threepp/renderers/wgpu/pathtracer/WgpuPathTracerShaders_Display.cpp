#include "WgpuPathTracerShaders.hpp"

namespace threepp::wgpu_pt {

// ---------------------------------------------------------------------------
// WGSL depth-fill shader — reconstruct rasterizer depth from path-tracer gBuffer.
// Reads primary-ray hit distance (t) from the gBuffer's w channel, reconstructs
// the world-space hit point, then projects it to WebGPU NDC [0,1] depth.
// ---------------------------------------------------------------------------
const char* const depthFillWGSL = R"(
struct DepthFillUniforms {
    projView:   mat4x4<f32>,
    camOri:     vec4<f32>,
    camFwd:     vec4<f32>,
    camRgt:     vec4<f32>,
    camUp:      vec4<f32>,
    iRes:       vec4<f32>,
    tanHalfFov: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: DepthFillUniforms;
@group(0) @binding(1) var gBuf: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(vid & 1u) * 4.0 - 1.0;
    let y = f32((vid >> 1u) & 1u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) fpos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let px  = vec2<i32>(i32(fpos.x), i32(fpos.y));
    let t   = textureLoad(gBuf, px, 0).w;
    if (t <= 0.0) { return 1.0; }
    let ndc    = vec2<f32>((fpos.x / u.iRes.x) * 2.0 - 1.0,
                            1.0 - (fpos.y / u.iRes.y) * 2.0);
    let aspect = u.iRes.x / u.iRes.y;
    let rayDir = normalize(u.camFwd.xyz
                         + u.camRgt.xyz * (ndc.x * u.tanHalfFov.x * aspect)
                         + u.camUp.xyz  * (ndc.y * u.tanHalfFov.x));
    let worldPos = u.camOri.xyz + t * rayDir;
    let clip     = u.projView * vec4<f32>(worldPos, 1.0);
    if (clip.w <= 0.0) { return 1.0; }
    return clamp(clip.z / clip.w, 0.0, 1.0);
}
)";

// ---------------------------------------------------------------------------
// WGSL display shader — blit accumulated texture as linear HDR radiance.
// Tonemap + sRGB encoding are applied by WgpuRenderer's post-process pass.
// ---------------------------------------------------------------------------
const char* const displayWGSL = R"(
struct TransformUniforms {
    model:      mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    normalCol0: vec4<f32>,
    normalCol1: vec4<f32>,
    normalCol2: vec4<f32>,
    cameraPos:  vec3<f32>,
    _pad:       f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniforms;
@group(0) @binding(2) var accumTex: texture_2d<f32>;
// binding 3 = accumTex sampler (unused)
@group(0) @binding(4) var diffTex:  texture_2d<f32>;   // diffuse radiance
// binding 5 = diffTex sampler (unused)
@group(0) @binding(6) var gBufTex:  texture_2d<f32>;  // gBuf: .w = primary hit t, 0 = background
// binding 7 = gBufTex sampler (unused)
@group(0) @binding(8) var specTex:     texture_2d<f32>;   // specular radiance
@group(0) @binding(10) var upscaleTex: texture_2d<f32>;   // TAAU full-res output (1×1 dummy when inactive)

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return transform.proj * transform.view * transform.model * vec4<f32>(position, 1.0);
}

// This shader outputs raw linear HDR radiance. Exposure, tone mapping, and
// sRGB encoding are all applied by the WgpuRenderer post-process pass based
// on the renderer's `toneMappingExposure`, `toneMapping`, and `outputEncoding`
// settings — applying any of them here would double up with that pass.
fn upscaleAt(p: vec2<i32>, sz: vec2<i32>) -> vec3<f32> {
    let pc = clamp(p, vec2<i32>(0), sz - vec2<i32>(1, 1));
    let s  = textureLoad(upscaleTex, pc, 0);
    return max(s.xyz, vec3<f32>(0.0));
}

@fragment
fn fs_main(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
    // _pad encodes both pixelScale and AOV mode:
    //   _pad = pixelScale + aovMode * 10.0
    // Normal rendering: aovMode=0, _pad = pixelScale (0.1-2.0)
    // AOV mode N: _pad = pixelScale + N*10  (N in 1..5, so _pad > 10)
    let rawPad    = transform._pad;
    let aovMode   = i32(rawPad / 10.0);
    let pixScale  = max(rawPad - f32(aovMode) * 10.0, 0.01);
    let accumSize = vec2<i32>(textureDimensions(accumTex, 0));

    // AOV mode: direct passthrough of diffTex as linear radiance.
    // Renderer post-process handles sRGB encoding.
    if (aovMode > 0) {
        let px = clamp(vec2<i32>(fragPos.xy * pixScale), vec2<i32>(0), accumSize - 1);
        let col = textureLoad(diffTex, px, 0).xyz;
        return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
    }

    // Temporal upscale path: if upscaleTex is full-res (> 1×1), TAAU is active.
    let upscaleSize = vec2<i32>(textureDimensions(upscaleTex, 0));
    if (upscaleSize.x > 1) {
        let px = vec2<i32>(i32(fragPos.x), i32(fragPos.y));
        return vec4<f32>(upscaleAt(px, upscaleSize), 1.0);
    }

    // When rendering at reduced resolution (pixelScale < 1), use joint bilateral
    // upsampling guided by the G-buffer (normal + depth). This reconstructs sharp
    // edges at mesh/material boundaries while smoothly interpolating interior
    // regions — effectively lossless for geometric edges, unlike nearest-neighbor
    // which produces blocky silhouettes or bilinear which blurs across edges.
    //
    // For each full-res pixel, compute its floating-point position in the low-res
    // buffer, then sample a 2×2 footprint of low-res texels. Weight each texel by
    // spatial proximity (bilinear) × normal similarity × depth similarity. Texels
    // on a different surface (normal mismatch or depth discontinuity) get near-zero
    // weight, so the filter never blends across edges.
    if (pixScale < 0.85) {
        // G-buffer is always at the same resolution as the accum buffer
        let pNearest = clamp(vec2<i32>(fragPos.xy * pixScale), vec2<i32>(0), accumSize - 1);
        let gRef = textureLoad(gBufTex, pNearest, 0);
        let nRef = gRef.xyz;
        let dRef = gRef.w;

        // Background: skip bilateral, just nearest-neighbor.
        if (dRef <= 0.0) {
            let col = textureLoad(diffTex, pNearest, 0).xyz + textureLoad(specTex, pNearest, 0).xyz;
            return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
        }

        // Low-res 2×2 footprint for bilateral interpolation
        let fp  = fragPos.xy * pixScale - 0.5;
        let p0  = vec2<i32>(i32(floor(fp.x)), i32(floor(fp.y)));
        let f   = fp - floor(fp);

        let bw = vec4<f32>((1.0 - f.x) * (1.0 - f.y),
                                  f.x  * (1.0 - f.y),
                           (1.0 - f.x) *        f.y,
                                  f.x  *        f.y);

        let offsets = array<vec2<i32>, 4>(
            vec2<i32>(0, 0), vec2<i32>(1, 0),
            vec2<i32>(0, 1), vec2<i32>(1, 1));

        var colSum  = vec3<f32>(0.0);
        var wSum    = 0.0;
        let bwa = array<f32, 4>(bw.x, bw.y, bw.z, bw.w);

        for (var i = 0u; i < 4u; i++) {
            let sp = clamp(p0 + offsets[i], vec2<i32>(0), accumSize - 1);
            let g  = textureLoad(gBufTex, sp, 0);
            let sn = g.xyz;
            let sd = g.w;

            if (sd <= 0.0) { continue; }

            let wn = pow(max(0.0, dot(nRef, sn)), 64.0);
            let wd = exp(-abs(dRef - sd) * 8.0 / max(dRef, 0.01));

            let w = bwa[i] * wn * wd;
            colSum += (textureLoad(diffTex, sp, 0).xyz + textureLoad(specTex, sp, 0).xyz) * w;
            wSum   += w;
        }

        var col: vec3<f32>;
        if (wSum > 1e-6) {
            col = colSum / wSum;
        } else {
            col = textureLoad(diffTex, pNearest, 0).xyz + textureLoad(specTex, pNearest, 0).xyz;
        }

        return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
    }

    // Native resolution — output linear HDR radiance; renderer post-process
    // applies exposure, tonemap, and sRGB encoding.
    let px  = clamp(vec2<i32>(fragPos.xy * pixScale), vec2<i32>(0), accumSize - 1);
    let col = textureLoad(diffTex, px, 0).xyz + textureLoad(specTex, px, 0).xyz;
    return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
}
)";

// ---------------------------------------------------------------------------
// WGSL temporal upscale shader — runs at full resolution, accumulates low-res
// denoised frames into a full-res history buffer via reprojection + EMA.
// ---------------------------------------------------------------------------
const char* const upscaleWGSL = R"(
struct UpscaleUniforms {
    prevCamOri: vec4<f32>,
    prevCamFwd: vec4<f32>,
    prevCamRgt: vec4<f32>,
    prevCamUp:  vec4<f32>,
    curCamOri:  vec4<f32>,
    curCamFwd:  vec4<f32>,
    curCamRgt:  vec4<f32>,
    curCamUp:   vec4<f32>,
    iRes:       vec4<f32>,   // [0]=fullW [1]=fullH [2]=pixelScale [3]=0
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,
};
@group(0) @binding(0) var<uniform> up: UpscaleUniforms;
@group(0) @binding(1) var denoisedDiff: texture_2d<f32>;
@group(0) @binding(2) var denoisedSpec: texture_2d<f32>;
@group(0) @binding(3) var gBufCurLow: texture_2d<f32>;  // normal.xyz + rayDist.w
@group(0) @binding(4) var historyIn:  texture_2d<f32>;  // previous full-res output
@group(0) @binding(5) var historyOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn upscale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fullPx  = vec2<i32>(i32(gid.x), i32(gid.y));
    let fullW   = up.iRes.x;
    let fullH   = up.iRes.y;
    if (f32(fullPx.x) >= fullW || f32(fullPx.y) >= fullH) { return; }

    let pixScale   = up.iRes.z;
    let lowResSize = vec2<i32>(textureDimensions(denoisedDiff));

    // Map full-res pixel to low-res accum pixel
    let lowResPx = clamp(
        vec2<i32>(vec2<f32>(fullPx) * pixScale),
        vec2<i32>(0),
        lowResSize - vec2<i32>(1)
    );

    // Current denoised color at low-res pixel
    let curColor = textureLoad(denoisedDiff, lowResPx, 0).xyz
                 + textureLoad(denoisedSpec, lowResPx, 0).xyz;

    // G-buffer: normal.xyz + ray distance.w
    let gBuf     = textureLoad(gBufCurLow, lowResPx, 0);
    let curDepth = gBuf.w;

    // Sky pixel — no temporal history possible
    if (curDepth < 1e-6) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, -1.0));
        return;
    }

    // Reconstruct world position from low-res pixel-center ray + ray distance
    let aspect  = fullW / fullH;
    let tanHfov = up.tanHalfFov.x;
    let lowNdc  = vec2<f32>(
        (f32(lowResPx.x) + 0.5) / f32(lowResSize.x) * 2.0 - 1.0,
        1.0 - (f32(lowResPx.y) + 0.5) / f32(lowResSize.y) * 2.0
    );
    let rayDir  = normalize(up.curCamFwd.xyz
                          + up.curCamRgt.xyz * (lowNdc.x * tanHfov * aspect)
                          + up.curCamUp.xyz  * (lowNdc.y * tanHfov));
    let worldPos = up.curCamOri.xyz + rayDir * curDepth;

    // Reproject world position to previous frame's full-res screen
    let relP  = worldPos - up.prevCamOri.xyz;
    let prevZ = dot(relP, up.prevCamFwd.xyz);
    if (prevZ <= 0.001) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, 1.0));
        return;
    }

    let prevNdcX = dot(relP, up.prevCamRgt.xyz) / (prevZ * tanHfov * aspect);
    let prevNdcY = dot(relP, up.prevCamUp.xyz)  / (prevZ * tanHfov);
    let prevU    = (prevNdcX + 1.0) * 0.5 * fullW - 0.5;
    let prevV    = (1.0 - prevNdcY) * 0.5 * fullH - 0.5;
    let prevFullPx = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));

    if (prevFullPx.x < 0 || prevFullPx.y < 0 ||
        prevFullPx.x >= i32(fullW) || prevFullPx.y >= i32(fullH)) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, 1.0));
        return;
    }

    // Fetch full-res history (.w = histLen; -1 = sky sentinel; 0 = fresh/reset)
    let histSamp  = textureLoad(historyIn, prevFullPx, 0);
    let histColor = histSamp.xyz;
    let histLen   = histSamp.w;

    // Disocclusion: only reject sky-sentinel history (depth cross-frame comparison
    // is unreliable for moving cameras — let short max-history handle ghosting).
    var result     = curColor;
    var newHistLen = 1.0;
    if (histLen >= 0.0) {
        newHistLen = min(histLen + 1.0, 32.0);
        let alpha  = max(1.0 / 16.0, 1.0 / newHistLen);
        result = mix(histColor, curColor, alpha);
    }

    textureStore(historyOut, fullPx, vec4<f32>(result, newHistLen));
}
)";

}// namespace threepp::wgpu_pt
