#include "WgpuPathTracerShaders.hpp"

namespace threepp::wgpu_pt {

// ---------------------------------------------------------------------------
// WGSL variance-guided à-trous spatial filter
// Uses the frame count (w channel) to adapt filter strength:
//   low frame count → more aggressive blur to suppress MC noise
//   high frame count → gentle filter to preserve converged detail
// ---------------------------------------------------------------------------
const char* const svgfAtrousWGSL = R"(
struct AtrousUni { stepSize: u32, mode: u32, frameCount: f32, _p1: f32, }
// mode: 0 = diffuse (wide kernel, relaxed), 1 = specular (tight, aggressive firefly clamp)

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

    // mode: 0 = diffuse, 1 = specular
    let isSpec = uni.mode != 0u;
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
        // Narkowicz ACES in the display used to hard-clamp anything >~5 to white,
        // hiding firefly leakage. Full three.js ACES (now in renderer post-process)
        // has a rolloff that preserves bright variation, so caps must be tighter.
        let specCap0  = mix(4.0, 10.0, cRough);
        let filterCap0 = select(10.0, specCap0, isSpec);
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
            // Depth edge-stopping: sub-pixel blue-noise jitter contributes
            // <1% depth variance on typical surfaces, well inside the falloff.
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
            let w_l = exp(-(sLum - cLum) * (sLum - cLum) / (effectiveLumSigma * effectiveLumSigma + 1e-6));

            // Per-sample outlier clamp: suppress extreme bright samples in filter window.
            // Specular mode: tighter cap to kill fireflies without blurring.
            var sIrrClamped  = sIrr;
            var sColorClamped = sColor;
            let sIrrLum = luminance(sIrr);
            let sColLum = luminance(sColor);
            // Specular firefly cap: roughness-adaptive — rough metals tolerate
            // higher values (wider lobe, more variance), glossy needs tight cap.
            // The spatial filter is the only smoothing the specular channel gets,
            // so aggressive per-sample clamping is critical to kill fireflies before
            // the kernel averages them in.
            let specCap = mix(4.0, 10.0, cRough);
            let filterCap = select(10.0, specCap, isSpec);
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
    let fireflyCeil = max(neighborMeanLum * 2.0, 0.3);
    if (resLum > fireflyCeil) {
        spatialResult = spatialResult * (fireflyCeil / resLum);
    }
    // Fade filter strength with accumulated frame count. At low FC the
    // accumulation is noisy and spatial filtering is essential. At high FC
    // the accumulation has converged; the filter's G-buffer-driven edge weights
    // vary each frame due to subpixel jitter and cause shimmer. Fading to the
    // raw accumulation at convergence eliminates that shimmer.
    let fcFade = 1.0 - smoothstep(16.0, 64.0, cFC);
    // Roughness-driven blend for specular: smooth metals (low roughness)
    // have clean reflections — don't blur them. Rough surfaces benefit
    // from spatial filtering. Diffuse always gets full filtering.
    let specRoughBlend = smoothstep(0.05, 0.3, cRough);
    let filterBlend = select(1.0, specRoughBlend, isSpec) * fcFade;
    let result = mix(cColor, spatialResult, filterBlend);
    textureStore(colorOut, pixel, vec4<f32>(result, cFC));
}
)";

}// namespace threepp::wgpu_pt
