#include "WgpuPathTracerShaders.hpp"

namespace threepp::wgpu_pt {

// ---------------------------------------------------------------------------
// WGSL ReLAX-inspired temporal accumulation (Phase 1)
// Per-channel temporal filter with per-pixel history length, moment tracking,
// luminance-based anti-lag clamping, and mode-aware blending (diffuse vs specular).
// Runs separately on diffuse and specular split buffers.
// ---------------------------------------------------------------------------
const char* const taaWGSL = R"(
struct TaaUniforms {
    prevCamOri: vec4<f32>, prevCamFwd: vec4<f32>, prevCamRgt: vec4<f32>, prevCamUp: vec4<f32>,
    curCamOri:  vec4<f32>, curCamFwd:  vec4<f32>, curCamRgt:  vec4<f32>, curCamUp:  vec4<f32>,
    iRes:       vec4<f32>,   // .xy = resolution, .zw = prevJx/prevJy
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,   // .x = global FC, .y = mode (0=diff, 1=spec), .z = curJx, .w = curJy
    movedMeshBits: vec4<u32>,
}

@group(0) @binding(0) var<uniform> taa:      TaaUniforms;
@group(0) @binding(1) var accumIn:   texture_2d<f32>;
@group(0) @binding(2) var gBufCur:   texture_2d<f32>;
@group(0) @binding(3) var taaHistIn: texture_2d<f32>;  // .w = history length
@group(0) @binding(4) var taaOut:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var hitMeshTex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> motionMats: array<mat4x4<f32>>;
@group(0) @binding(7) var albedoTex:  texture_2d<f32>;  // .w = primary linear roughness
@group(0) @binding(8) var momentsIn:  texture_2d<f32>;  // (E[L], E[L²]) from RT pass
@group(0) @binding(9) var gBufPrev:   texture_2d<f32>;  // previous frame g-buffer (depth in .w)

fn pcg_t(v: u32) -> u32 {
    var s = v * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (s >> 22u) ^ s;
}
const R2_A1_T: f32 = 0.7548776662466927;
const R2_A2_T: f32 = 0.5698402909980532;
fn r2Seq_t(n: u32) -> vec2<f32> {
    return fract(vec2<f32>(f32(n) * R2_A1_T, f32(n) * R2_A2_T));
}


@compute @workgroup_size(8, 8)
fn taa_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let res   = taa.iRes.xy;
    let iRes  = vec2<i32>(i32(res.x), i32(res.y));
    if (pixel.x >= iRes.x || pixel.y >= iRes.y) { return; }

    let isSpec  = taa.frameCount.y > 0.5;
    let curSamp  = textureLoad(accumIn, pixel, 0);
    let curColor = curSamp.xyz;
    let curFC    = curSamp.w;
    let curGB    = textureLoad(gBufCur, pixel, 0);
    let curNorm  = curGB.xyz;
    let curDepth = curGB.w;

    // Sky pixels: pass through, history length = 0
    if (curDepth < 1e-6) {
        textureStore(taaOut, pixel, vec4<f32>(curColor, 0.0));
        return;
    }

    // Foveated/checkerboard-skipped pixels: .w = -1 sentinel — pass through history.
    if (curFC < -0.5) {
        let prevHist = textureLoad(taaHistIn, pixel, 0);
        textureStore(taaOut, pixel, vec4<f32>(prevHist.xyz, prevHist.w));
        return;
    }

    let hitInfo  = textureLoad(hitMeshTex, pixel, 0);
    let curMeshId = u32(hitInfo.r);
    let touchedMoved = hitInfo.b > 0.5;

    // No variance box. SVGF design: temporal pass = pure EMA + disocclusion reset.
    // The moments-driven à-trous spatial filter handles all noise removal.
    // Box clamping in temporal passes is a rasterizer TAA technique — incorrect for
    // path tracing where every sample is independently noisy regardless of history.

    // Reproject current pixel into previous frame's screen space.
    // Rays fire through pixel centers (no jitter with denoiser active), so
    // depths are consistent across frames and reprojection is exact for static scenes.
    let aspect = res.x / res.y;
    let ndc = vec2<f32>((f32(pixel.x) + 0.5) / res.x * 2.0 - 1.0,
                         1.0 - (f32(pixel.y) + 0.5) / res.y * 2.0);
    let rayDir   = normalize(taa.curCamFwd.xyz
                            + taa.curCamRgt.xyz * (ndc.x * taa.tanHalfFov.x * aspect)
                            + taa.curCamUp.xyz  * (ndc.y * taa.tanHalfFov.x));
    let worldPos = taa.curCamOri.xyz + rayDir * curDepth;

    // For moved meshes: transform world pos by motion matrix
    let meshIdx = u32(textureLoad(hitMeshTex, pixel, 0).r);
    var prevWorldPos = worldPos;
    if (meshIdx < 128u) {
        let bit = meshIdx & 31u;
        let wi  = meshIdx >> 5u;
        var mbits: u32 = 0u;
        if      (wi == 0u) { mbits = taa.movedMeshBits.x; }
        else if (wi == 1u) { mbits = taa.movedMeshBits.y; }
        else if (wi == 2u) { mbits = taa.movedMeshBits.z; }
        else               { mbits = taa.movedMeshBits.w; }
        if (((mbits >> bit) & 1u) != 0u) {
            prevWorldPos = (motionMats[meshIdx] * vec4<f32>(worldPos, 1.0)).xyz;
        }
    }

    let toPoint  = prevWorldPos - taa.prevCamOri.xyz;
    let prevZ    = dot(toPoint, taa.prevCamFwd.xyz);

    var useHist = prevZ > 0.001;
    let prevNdcX = dot(toPoint, taa.prevCamRgt.xyz) / (prevZ * taa.tanHalfFov.x * aspect);
    let prevNdcY = dot(toPoint, taa.prevCamUp.xyz)  / (prevZ * taa.tanHalfFov.x);
    let prevPx   = vec2<f32>((prevNdcX + 1.0) * 0.5 * res.x - 0.5,
                              (1.0 - prevNdcY) * 0.5 * res.y - 0.5);
    let prevPixel = vec2<i32>(i32(floor(prevPx.x)), i32(floor(prevPx.y)));

    // Bounds check (need 2×2 footprint for bilinear)
    useHist = useHist && prevPixel.x >= 0 && prevPixel.x + 1 < iRes.x
                      && prevPixel.y >= 0 && prevPixel.y + 1 < iRes.y;

    // Depth-based disocclusion (two conditions):
    // 1. Standard: prevZ vs curZDepth — detects when current surface wasn't visible before.
    // 2. Revealed: current surface is much deeper than previous surface at prevPixel
    //    — detects background pixels revealed when a foreground object moved away.
    //    With no jitter the G-buffer depths are stable frame-to-frame, so this
    //    check is reliable in all motion states (no false positives at silhouettes).
    let curZDepth = dot(worldPos - taa.curCamOri.xyz, taa.curCamFwd.xyz);
    let depthRatio = abs(prevZ - curZDepth) / max(curZDepth, 0.01);
    let prevFrameDepth = textureLoad(gBufPrev, prevPixel, 0).w;
    let revealedRatio = abs(curDepth - prevFrameDepth) / max(curDepth, 0.01);
    let disoccluded = useHist && (depthRatio > 0.1 || revealedRatio > 0.15);

    var result = curColor;
    var newHistLen = 1.0;

    if (useHist && !disoccluded) {
        // Per-tap validated bilinear history fetch.
        // Blending across geometry edges mixes foreground/background history.
        // Each tap is validated against depth and normal agreement with the current pixel.
        let fx = fract(prevPx.x);
        let fy = fract(prevPx.y);
        let px00 = prevPixel;
        let px10 = prevPixel + vec2<i32>(1, 0);
        let px01 = prevPixel + vec2<i32>(0, 1);
        let px11 = prevPixel + vec2<i32>(1, 1);

        let h00 = textureLoad(taaHistIn, px00, 0);
        let h10 = textureLoad(taaHistIn, px10, 0);
        let h01 = textureLoad(taaHistIn, px01, 0);
        let h11 = textureLoad(taaHistIn, px11, 0);

        // Use g-buffer on BOTH sides for per-tap validation (no jitter with denoiser active).
        let sg00 = textureLoad(gBufPrev, px00, 0);
        let sg10 = textureLoad(gBufPrev, px10, 0);
        let sg01 = textureLoad(gBufPrev, px01, 0);
        let sg11 = textureLoad(gBufPrev, px11, 0);

        // Without jitter, depths are perfectly consistent between frames for static surfaces,
        // so we can use a tighter depth tolerance for sharper edge stopping.
        let depthTol = 0.12;
        let normTol  = 0.8;  // dot(curNorm, tapNorm) must exceed this

        let v00 = select(0.0, (1.0-fx)*(1.0-fy), abs(sg00.w - curDepth)/max(curDepth,0.01) < depthTol && dot(curNorm, sg00.xyz) > normTol);
        let v10 = select(0.0, fx      *(1.0-fy), abs(sg10.w - curDepth)/max(curDepth,0.01) < depthTol && dot(curNorm, sg10.xyz) > normTol);
        let v01 = select(0.0, (1.0-fx)*fy,       abs(sg01.w - curDepth)/max(curDepth,0.01) < depthTol && dot(curNorm, sg01.xyz) > normTol);
        let v11 = select(0.0, fx      *fy,        abs(sg11.w - curDepth)/max(curDepth,0.01) < depthTol && dot(curNorm, sg11.xyz) > normTol);
        let vSum = v00 + v10 + v01 + v11;

        // No valid taps — treat as disoccluded
        if (vSum < 1e-6) {
            textureStore(taaOut, pixel, vec4<f32>(curColor, 1.0));
            return;
        }
        let inv = 1.0 / vSum;
        let histColor   = (h00.xyz*v00 + h10.xyz*v10 + h01.xyz*v01 + h11.xyz*v11) * inv;
        let prevHistLen = (h00.w  *v00 + h10.w  *v10 + h01.w  *v01 + h11.w  *v11) * inv;

        // SVGF-style: pure EMA accumulation, no color-box clamping.
        // The moments-driven à-trous spatial filter handles all noise removal.
        // Ghosting is controlled by depth disocclusion (above) + touchedMoved cap.

        let primaryRoughForCap = textureLoad(albedoTex, pixel, 0).w;
        let roughT = smoothstep(0.05, 0.3, primaryRoughForCap);
        let maxHist = select(
            256.0,                    // diffuse: large history
            mix(8.0, 32.0, roughT),  // specular: 8 smooth → 32 rough
            isSpec
        );
        var effHistLen = min(prevHistLen, maxHist);
        if (touchedMoved) { effHistLen = min(effHistLen, 4.0); }

        // Minimum alpha: keep filter responsive to lighting changes even at convergence.
        let alpha = max(1.0 / 64.0, 1.0 / (effHistLen + 1.0));
        var finalAlpha = alpha;
        if (isSpec) {
            let smoothness = 1.0 - clamp(primaryRoughForCap, 0.0, 1.0);
            let specBoost = mix(1.0, 2.5, smoothness * smoothness);
            finalAlpha = min(1.0, alpha * specBoost);
        }

        result = mix(histColor, curColor, finalAlpha);
        newHistLen = effHistLen + 1.0;
    }

    // .w = per-pixel history length (used by spatial filter for variance-adaptive sigma)
    textureStore(taaOut, pixel, vec4<f32>(result, newHistLen));
}
)";

// ---------------------------------------------------------------------------
// WGSL variance-guided à-trous spatial filter
// Uses the frame count (w channel) to adapt filter strength:
//   low frame count → more aggressive blur to suppress MC noise
//   high frame count → gentle filter to preserve converged detail
// ---------------------------------------------------------------------------
const char* const svgfAtrousWGSL = R"(
struct AtrousUni { stepSize: u32, mode: u32, frameCount: f32, _p1: f32, }
// mode: 0 = combined/diffuse (wide kernel, relaxed), 1 = specular (tight, aggressive firefly clamp)

@group(0) @binding(0) var<uniform> uni:      AtrousUni;
@group(0) @binding(1) var colorIn:  texture_2d<f32>;
@group(0) @binding(2) var colorOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var gBuf:     texture_2d<f32>;
@group(0) @binding(4) var albedoBuf: texture_2d<f32>;
@group(0) @binding(5) var hitMeshBuf: texture_2d<f32>;
@group(0) @binding(6) var momentsBuf: texture_2d<f32>;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Demodulate: divide out albedo to isolate irradiance (smooth signal).
fn demod(color: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    return color / max(albedo, vec3<f32>(0.1));
}

@compute @workgroup_size(8, 8)
fn svgf_atrous_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let res   = vec2<i32>(textureDimensions(colorIn, 0));
    if (pixel.x >= res.x || pixel.y >= res.y) { return; }

    let step = i32(uni.stepSize);

    let cSamp   = textureLoad(colorIn, pixel, 0);
    let cColor  = cSamp.xyz;
    let cGB     = textureLoad(gBuf, pixel, 0);
    let cNorm   = cGB.xyz;
    let cDepth  = cGB.w;
    let cAlbedoFull = textureLoad(albedoBuf, pixel, 0);
    let cAlbedo = cAlbedoFull.xyz;
    let cRough  = cAlbedoFull.w;  // linear roughness (0 = mirror, 1 = diffuse)
    let cHitId  = textureLoad(hitMeshBuf, pixel, 0);
    let cMeshId = u32(cHitId.r);
    let cMatId  = i32(cHitId.g);

    // mode: 0 = diffuse, 1 = specular, 2 = diffuse temporal, 3 = specular temporal
    let isSpec = (uni.mode & 1u) != 0u;
    let isTemporal = (uni.mode & 2u) != 0u;
    let cFC = cSamp.w;

    // Sky pixels: pass through
    if (cDepth < 1e-6) {
        textureStore(colorOut, pixel, vec4<f32>(cColor, cFC));
        return;
    }

    // Demodulate center pixel for accumulation path (diffuse only)
    let cIrr = select(demod(cColor, cAlbedo), cColor, isSpec);
    // Blend luminance source: demod for bright surfaces (texture-aware), raw for dark (stable)
    let albedoLum = luminance(cAlbedo);
    let demodBlend = select(smoothstep(0.05, 0.2, albedoLum), 0.0, isSpec);
    let cLum = mix(luminance(cColor), luminance(cIrr), demodBlend);

    // Variance-guided luminance sigma.
    // Temporal variance from moments (reprojected, stable after ~10 frames).
    // Floor is generous (0.15 diffuse / 0.4 spec) so that at convergence — where
    // temporal variance shrinks toward zero — the bilateral retains enough
    // headroom to absorb scattered indirect-light noise. Edge protection is
    // delegated to the geometric checks (mesh/material ID, normal, depth);
    // luminance edge-stopping should not be the primary edge guard.
    let moments = textureLoad(momentsBuf, pixel, 0).xy;
    let temporalVar = max(moments.y - moments.x * moments.x, 0.0);
    let baseSigma = select(0.7, mix(0.5, 2.0, cRough), isSpec);
    let lumSigma  = max(select(0.08, 0.2, isSpec), sqrt(temporalVar) * baseSigma);

    // 5×5 bilateral filter — tracks both demodulated irradiance and raw color
    let kw = array<f32, 5>(1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0);

    var irrSum    = vec3<f32>(0.0);
    var rawSum    = vec3<f32>(0.0);
    var weightSum = 0.0;
    // Spatial luminance moments — accumulated during the filter loop below.
    // Used to bootstrap variance in early frames before temporal moments converge.
    var spatLumM1 = 0.0;
    var spatLumM2 = 0.0;
    var spatWsum  = 0.0;

    // Center pixel (dx=0, dy=0) — contribute using pre-loaded registers to avoid
    // 5 redundant textureLoad calls that the loop body would otherwise issue.
    // All edge-stopping weights are 1.0 for the center (same normal, depth, luminance).
    {
        let w_s0      = kw[2] * kw[2];
        let cIrrDemod = demod(cColor, cAlbedo);  // consistent with loop: always demodulate
        let specCap0  = mix(6.0, 16.0, cRough);
        let filterCap0 = select(20.0, specCap0, isSpec);
        let cIrrLum   = luminance(cIrrDemod);
        let cColLum   = luminance(cColor);
        var cIrrC  = cIrrDemod;
        var cColorC = cColor;
        if (cIrrLum > filterCap0) { cIrrC  = cIrrDemod * (filterCap0 / cIrrLum); }
        if (cColLum > filterCap0) { cColorC = cColor   * (filterCap0 / cColLum); }
        irrSum    += cIrrC  * w_s0;
        rawSum    += cColorC * w_s0;
        weightSum += w_s0;
        spatLumM1 += cLum * w_s0;
        spatLumM2 += cLum * cLum * w_s0;
        spatWsum  += w_s0;
    }

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            if (dx == 0 && dy == 0) { continue; }  // handled above with pre-loaded values
            let sp      = clamp(pixel + vec2<i32>(dx, dy) * step, vec2<i32>(0), res - 1);
            let sHitId  = textureLoad(hitMeshBuf, sp, 0);
            let sMeshId = u32(sHitId.r);
            let sMatId  = i32(sHitId.g);
            // Material ID must match. Mesh ID is allowed to differ — that
            // preserves the "share samples across distinct meshes with the
            // same material" benefit (instanced chairs, repeated walls,
            // foliage), but blocks the cross-material bleed that the previous
            // (sMeshId != cMeshId && sMatId != cMatId) form let through at
            // multi-material seams (a single mesh with several submaterials,
            // or two adjacent meshes coincidentally sharing one material on
            // one side and not the other).
            if (sMatId != cMatId) { continue; }

            let sColor  = textureLoad(colorIn, sp, 0).xyz;
            let sGB     = textureLoad(gBuf, sp, 0);
            let sAlbedo = textureLoad(albedoBuf, sp, 0).xyz;
            let sIrr    = demod(sColor, sAlbedo);
            let sNorm   = sGB.xyz;

            // Spatial weight (separable Gaussian)
            let w_s = kw[dy + 2] * kw[dx + 2];
            // Normal edge-stopping: FC-adaptive for BOTH modes.
            // Low cFC (noisy, moving): relaxed (pow=2) so the filter can smooth
            // curved surfaces (torus, sphere) where normals vary rapidly.
            // High cFC (converged): aggressive to preserve sharp edges.
            // The material ID check already prevents cross-object bleed.
            // Faster ramp (24 frames vs 48) helps silhouettes and subtle
            // geometric detail snap into focus sooner after camera settles.
            let specNormPow = mix(2.0, 16.0, smoothstep(0.0, 32.0, cFC));
            let diffNormPow = mix(2.0, 128.0, smoothstep(0.0, 24.0, cFC));
            let normPow = select(diffNormPow, specNormPow, isSpec);
            let w_n = pow(max(0.0, dot(cNorm, sNorm)), normPow);
            // Depth edge-stopping: rays fire through pixel centers (no jitter),
            // so depths are stable frame-to-frame — no patch artifacts.
            let depthBase = select(4.0, mix(1.0, 3.0, cRough), isSpec);
            let depthScale = mix(1.0, depthBase, smoothstep(0.0, 32.0, cFC));
            let w_d = exp(-abs(cDepth - sGB.w) * depthScale / (cDepth + 0.01));
            // Luminance edge-stopping: wider sigma while noisy, tight at convergence.
            // This is the primary mechanism preserving SHADOWS — shadow pixels have
            // same material/normal/depth as lit neighbors, so only the luminance stop
            // can keep them from being smoothed away. Ramp to 1.0× in 24 frames
            // (was 64) so shadow edges sharpen within ~0.8s at 30fps.
            let lumBoost = mix(4.0, 1.0, smoothstep(0.0, 24.0, cFC));
            let effectiveLumSigma = lumSigma * lumBoost;
            let sAlbLum = luminance(sAlbedo);
            let sDemod = smoothstep(0.05, 0.2, sAlbLum);
            let sLum = mix(luminance(sColor), luminance(sIrr), sDemod);
            // Temporal cleanup: widen luminance sigma to tolerate TAA noise residuals
            // without creating grid patches, but keep relative scaling so strong
            // edges (glass/transmission) are still preserved.
            let finalLumSigma = select(effectiveLumSigma, effectiveLumSigma * 2.0, isTemporal);
            let w_l = exp(-(sLum - cLum) * (sLum - cLum) / (finalLumSigma * finalLumSigma + 1e-6));

            // Per-sample outlier clamp: suppress extreme bright samples in filter window.
            // Specular mode: tighter cap to kill fireflies without blurring.
            var sIrrClamped  = sIrr;
            var sColorClamped = sColor;
            let sIrrLum = luminance(sIrr);
            let sColLum = luminance(sColor);
            // Specular firefly cap: roughness-adaptive — rough metals tolerate
            // higher values (wider lobe, more variance), glossy needs tight cap.
            // Aggressive clamping is critical since we don't have TAA temporal smoothing
            // on the specular split buffer.
            let specCap = mix(6.0, 16.0, cRough);
            let filterCap = select(20.0, specCap, isSpec);
            if (sIrrLum > filterCap) { sIrrClamped  = sIrr  * (filterCap / sIrrLum); }
            if (sColLum > filterCap) { sColorClamped = sColor * (filterCap / sColLum); }

            // Albedo-similarity edge-stop (diffuse only).
            // Within a single material, texture variation creates albedo variation
            // without triggering material-ID/normal/depth edge-stops. Without this
            // check, the filter averages directly across texture boundaries and
            // washes out details like marble veins, wood grain, or fabric weave.
            //
            // Factor 40 is aggressive: any noticeable albedo delta (~0.05 per
            // channel) already drops contribution below 70%, and 5 à-trous passes
            // compound any mixing across 32 pixels.  Same-color neighbors within
            // a uniform region (e.g. marble base at ~0.01 delta) still contribute
            // ~96% so noise reduction still works — but marble veins (~0.1 delta)
            // drop to ~30%, bark-to-leaf (~0.3 delta) to <1%.
            //
            // Also protects against color contamination from the per-channel
            // demod floor: neighbors with very different albedos (where the floor
            // would distort hue) contribute little.
            let albDiff = cAlbedo - sAlbedo;
            let w_a = select(exp(-dot(albDiff, albDiff) * 40.0), 1.0, isSpec);

            let w = w_s * w_n * w_d * w_l * w_a;
            irrSum    += sIrrClamped  * w;
            rawSum    += sColorClamped * w;
            weightSum += w;
            // Spatial luminance moments (unweighted by edge-stopping — spatial-only weight)
            // Used for variance bootstrapping in early frames before temporal moments converge.
            spatLumM1 += sLum * w_s;
            spatLumM2 += sLum * sLum * w_s;
            spatWsum  += w_s;
        }
    }

    // Blend temporal and spatial variance for diagnostics / future passes:
    // Early frames (cFC < 8): temporal moments haven't converged → rely on spatial estimate.
    // Converged (cFC > 32): temporal moments are stable → use them (more accurate, follows surfaces).
    let spatVar  = select(temporalVar, max(spatLumM2/spatWsum - (spatLumM1/spatWsum)*(spatLumM1/spatWsum), 0.0), spatWsum > 1e-6);
    let varBlend = smoothstep(4.0, 32.0, cFC);
    let variance = mix(spatVar, temporalVar, varBlend);

    // Blend: demod/remod path for bright surfaces, raw filter for dark surfaces
    let filteredIrr = select(cIrr, irrSum / weightSum, weightSum > 1e-6);
    let filteredRaw = select(cColor, rawSum / weightSum, weightSum > 1e-6);
    let demodResult = filteredIrr * cAlbedo;
    var spatialResult = mix(filteredRaw, demodResult, demodBlend);

    // Output firefly clamp.
    // The per-sample clamp in the kernel bounds neighbor contributions, but a
    // bright center pixel can still dominate the output at small step sizes
    // (step=1 has high center weight).  Clamp the result against the mean of
    // neighbors (rawSum / weightSum, already per-sample-clamped) at 3× so a
    // legitimate bright sample survives but a 10-100× firefly is pulled back.
    let resLum = luminance(spatialResult);
    let neighborMeanLum = luminance(filteredRaw);
    let fireflyCeil = max(neighborMeanLum * 3.0, 0.5);
    if (resLum > fireflyCeil) {
        spatialResult = spatialResult * (fireflyCeil / resLum);
    }
    // Roughness-driven blend for specular: smooth metals (low roughness)
    // have clean reflections — don't blur them. Rough surfaces benefit
    // from spatial filtering. Diffuse always gets full filtering.
    let specRoughBlend = smoothstep(0.05, 0.3, cRough);
    let filterBlend = select(1.0, specRoughBlend, isSpec);
    let result = mix(cColor, spatialResult, filterBlend);
    textureStore(colorOut, pixel, vec4<f32>(result, cFC));
}
)";

}// namespace threepp::wgpu_pt
