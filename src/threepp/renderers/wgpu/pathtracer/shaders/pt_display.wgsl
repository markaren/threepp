
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
// 2026-04-23: accumTex removed. Bindings renumbered to stay dense since the
// WgpuMaterial pipeline-layout builder assigns texture+sampler slots
// sequentially from sorted customTextures (diffTex, gBufTex, specTex,
// upscaleTex → 2/4, 4/5, 6/7, 8/9 with unused samplers at odd slots).
@group(0) @binding(2) var diffTex:     texture_2d<f32>;   // diffuse radiance
// binding 3 = diffTex sampler (unused)
@group(0) @binding(4) var gBufTex:     texture_2d<f32>;  // gBuf: .w = primary hit t, 0 = background
// binding 5 = gBufTex sampler (unused)
@group(0) @binding(6) var specTex:     texture_2d<f32>;   // specular radiance
// binding 7 = specTex sampler (unused)
@group(0) @binding(8) var upscaleTex:  texture_2d<f32>;   // TAAU full-res output (1×1 dummy when inactive)

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
    let accumSize = vec2<i32>(textureDimensions(diffTex, 0));

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
