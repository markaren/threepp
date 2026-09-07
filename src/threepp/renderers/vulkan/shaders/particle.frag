#version 460

// Particle billboard fragment shader. Mirrors the GL ParticleSystem fragment
// shader (modulate per-particle vertex color × particle texture) and the HUD
// overlay_sprite.frag color convention.
//
// The vertex color (HSL→RGB) and the texture are LINEAR (an sRGB-tagged texture
// is hardware-decoded to linear on sample). The swapchain is B8G8R8A8_UNORM with
// no hardware sRGB write-out and the rendered image is sRGB-encoded by
// post_composite.comp, so we apply the same linear→sRGB OETF here. Without it particles
// composite darker/more-saturated than the rest of the frame. Blending (alpha vs
// additive) is set by the pipeline variant, not here. Untextured particle systems
// bind a 1×1 white default so the sampler is always valid.
//
// ── Lit particles (particle_light.comp) ─────────────────────────────────────
// Normal-blend (smoke/dust) particles are LIT: particle_light.comp evaluates
// the deferred light field at each particle center — sun × RT shadow × cloud
// shadow with a forward HG phase, the clustered point/spot list, probe-GI/env
// ambient — plus the camera→particle fog leg's three surface-path terms
// (analytic extinction+in-scatter, froxel-LUT glow, short sun march). The
// vertex shader hands the result over as vLight (radiance ×, T) + vFogAdd.
// This fragment then works in LINEAR HDR: albedo×tex×light×T + fogAdd, and
// converts to the display-referred overlay domain with the SAME exposure +
// tone-map curve post_composite applies to the scene (white balance and the
// grading LUT are skipped — accepted approximation for soft translucents).
// ADDITIVE particles (embers, glows) are emissive by design and keep the
// legacy unlit path below, as does any draw the compute pass didn't cover
// (vLight.a < 0) or a frame with lighting off (ofog.litActive == 0).
//
// ── Phase 2b: unified fog on world-space particles (legacy/unlit path) ──────
// The overlay pass runs POST-TAA / POST-composite and never saw the fog, so a
// chimney smoke puff (or any ParticleSystem billboard) punched through dense
// mist unattenuated. We bind the unified air-fog + underwater-murk medium
// (set 1) and, per fragment, compute the closed-form optical depth over the
// camera→particle leg and let the mist attenuate the puff. Closed-form only —
// the froxel LUT is not bound in the overlay pass (the air medium is a smooth
// exponential-height profile, so the closed form IS its exact integral).
// KEEP-IN-SYNC: airOpticalDepth mirrors heightFogOpticalDepth in
// deferred_shade_60_fog_volumetrics.glsl / deferred_filter_common.glsl.
//
// Value domain of the UNLIT path: display-referred (this shader OETF-encodes
// and the hardware alpha-blends in sRGB over the ACES-tonemapped background),
// NO exposure/ACES. So it deliberately does NOT inject a fog COLOUR: a linear
// mist radiance would be brighter than the tonemapped fog and read as an
// un-attenuated white plume. Instead the ALPHA path reduces the puff's COVERAGE
// by the transmittance, revealing the already-fog-shaded background —
// value-domain-agnostic and exact in near-uniform mist (see main()). The LIT
// path has no such constraint: it carries real exposure + tone mapping, so the
// fog colour (fogAdd) is composited exactly like the surface path does it.

// constant_id 0: 1 in the ADDITIVE pipeline variant (glowing embers). Additive
// blend just ADDs the source, so the mist ATTENUATES the added glow; alpha
// (Normal) smoke instead loses COVERAGE to the fog (melts into the background).
layout(constant_id = 0) const int kAdditive = 0;

layout(set = 0, binding = 0) uniform sampler2D tex;

// Unified fog medium (VulkanCore overlay-fog UBO — a per-frame snapshot of the
// resolved air medium + underwater murk + camera basis). std140.
layout(set = 1, binding = 0, std140) uniform OverlayFog {
    float fogActive;     // >0.5 = a medium is present this frame
    float hfDensity;     // air-medium σ_t at baseY (0 = no air fog)
    float hfBaseY;       // air-medium base world Y
    float hfFalloff;     // air-medium exponential height scale (m); huge ≈ uniform
    float murkDensity;   // underwater-murk σ_t (0 = off)
    float waterSurfaceY; // world Y of the water surface — the murk clip
    float camWorldY;     // camera world Y
    vec3  viewToWorldY;  // world-Y row of the inverse-view: worldY = dot(.,viewPos)+camWorldY
    vec3  fogInscatter;  // vestigial: LINEAR air-fog tint (unused — the LIT path
                         // gets its fog colour from particle_light.comp instead)
    vec3  murkInscatter; // vestigial: LINEAR murk tint (unused — see above)
    float _padA;         // std140: keep litActive OUT of murkInscatter's tail
                         // slot (a float after a vec3 packs at offset 76; the
                         // CPU struct's _pad3 sits there — GpuOverlayFogUbo)
    float litActive;     // >0.5 = particle_light.comp ran this render call
    float exposure;      // FULL tone-map exposure (currentExposure())
    float toneMapMode;   // threepp::ToneMapping (post_composite pc.toneMapping)
} ofog;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUv;
layout(location = 2) in vec3 vViewPos;
layout(location = 3) flat in vec4 vLight; // .rgb radiance factor, .a leg T (<0 = unlit draw)
layout(location = 4) flat in vec3 vFogAdd;// leg in-scatter added after T

layout(location = 0) out vec4 outColor;

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

// ── Tone mapping (lit path) — KEEP IN SYNC with post_composite.comp ─────────
// The lit path produces LINEAR HDR radiance and must land in the same display
// domain the tonemapped background occupies, so the curves are mirrored
// verbatim (white balance + grading LUT intentionally omitted).
vec3 reinhard(vec3 c) { return clamp(c / (vec3(1.0) + c), 0.0, 1.0); }
vec3 cineon(vec3 c) {
    c = max(vec3(0.0), c - 0.004);
    return pow((c * (6.2 * c + 0.5)) / (c * (6.2 * c + 1.7) + 0.06), vec3(2.2));
}
vec3 acesFilmic(vec3 c) {
    const mat3 inMat = mat3(
            vec3(0.59719, 0.07600, 0.02840),
            vec3(0.35458, 0.90834, 0.13383),
            vec3(0.04823, 0.01566, 0.83777));
    const mat3 outMat = mat3(
            vec3( 1.60475, -0.10208, -0.00327),
            vec3(-0.53108,  1.10813, -0.07276),
            vec3(-0.07367, -0.00605,  1.07602));
    c = inMat * c;
    const vec3 a = c * (c + 0.0245786) - 0.000090537;
    const vec3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
    return clamp(outMat * (a / b), 0.0, 1.0);
}
vec3 neutral(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;
    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
}
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}
vec3 agx(vec3 color) {
    const mat3 AgXInsetMatrix = mat3(
            vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
            vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
            vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
    const mat3 AgXOutsetMatrix = mat3(
            vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
            vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
            vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
    const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
            vec3(0.6274, 0.0691, 0.0164),
            vec3(0.3293, 0.9195, 0.0880),
            vec3(0.0433, 0.0113, 0.8956));
    const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
            vec3(1.6605, -0.1246, -0.0182),
            vec3(-0.5876, 1.1329, -0.1006),
            vec3(-0.0728, -0.0083, 1.1187));
    const float AgxMinEv = -12.47393;
    const float AgxMaxEv = 4.026069;

    color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
    color = AgXInsetMatrix * color;
    color = max(color, vec3(1e-10));
    color = log2(color);
    color = (color - AgxMinEv) / (AgxMaxEv - AgxMinEv);
    color = clamp(color, 0.0, 1.0);
    color = agxDefaultContrastApprox(color);
    color = AgXOutsetMatrix * color;
    color = pow(max(vec3(0.0), color), vec3(2.2));
    color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
    return clamp(color, 0.0, 1.0);
}
vec3 toneMap(vec3 c, uint mode, float exposure) {
    if (mode == 1u) return c * exposure;                    // Linear
    if (mode == 2u) return reinhard(c * exposure);          // Reinhard
    if (mode == 3u) return cineon(c * exposure);            // OptimizedCineon
    if (mode == 4u) return acesFilmic(c * (exposure / 0.6));// ACESFilmic
    if (mode == 6u) return neutral(c * exposure);           // Khronos PBR Neutral
    if (mode == 7u) return agx(c * exposure);               // AgX
    return c;// None / Custom — pass-through
}

// Closed-form exponential-height-fog optical depth along [camY, partY] over the
// leg of length `len`. Numerically-stable (1−e^{−x})/x form (avoids the fp32
// cancellation of e^{-ya}−e^{-yb} when the falloff is huge / near-uniform).
// KEEP IN SYNC with heightFogOpticalDepth in the deferred shaders.
float airOpticalDepth(float camY, float partY, float len) {
    if (ofog.hfDensity <= 0.0) return 0.0;
    const float H  = max(ofog.hfFalloff, 1e-3);
    const float ya = max(camY  - ofog.hfBaseY, 0.0);
    const float yb = max(partY - ofog.hfBaseY, 0.0);
    // Overflow-safe difference form (ea−eb)/x + Taylor near x→0, with a finite leg
    // clamp + saturated optical depth so no exp(-od) ever sees Inf/NaN (particle
    // legs are finite, but this KEEPS IN SYNC with heightFogOpticalDepth in
    // deferred_shade_60_fog_volumetrics.glsl / deferred_filter_common.glsl).
    const float clampedLen = min(len, 1.0e7);
    const float ea = exp(-ya / H);
    const float eb = exp(-yb / H);
    const float x  = (yb - ya) / H;
    const float f  = (abs(x) < 1e-3) ? (ea * (1.0 - 0.5 * x + x * x * (1.0 / 6.0)))
                                     : ((ea - eb) / x);
    return min(ofog.hfDensity * clampedLen * f, 80.0);
}

// Homogeneous underwater-murk optical depth over the BELOW-waterSurfaceY portion
// of the leg (0 above the waterline). Mirrors fogPathLength/applyMurk.
float murkOpticalDepth(float camY, float partY, float len) {
    if (ofog.murkDensity <= 0.0) return 0.0;
    float d = len;
    if (ofog.waterSurfaceY < 1e29) {
        const float ya = camY  - ofog.waterSurfaceY;
        const float yb = partY - ofog.waterSurfaceY;
        if (ya >= 0.0 && yb >= 0.0) d = 0.0;                      // wholly above water
        else if (!(ya < 0.0 && yb < 0.0)) {                       // straddles the surface
            const float t = ya / (ya - yb);
            d *= (ya < 0.0) ? t : (1.0 - t);
        }
    }
    return ofog.murkDensity * d;
}

void main() {
    vec4 t = texture(tex, vUv);
    vec3 lin = vColor.rgb * t.rgb;
    float alpha = vColor.a * t.a;

    // ── LIT path (Normal blend only; see header). LINEAR HDR end to end:
    // albedo × light, fogged exactly like a surface (lit×T + in-scatter), then
    // the scene's own exposure + tone-map curve and the shared sRGB encode.
    // Coverage stays the material's alpha — the fog colour now composites in
    // the radiance domain, so the display-referred coverage hack below is not
    // needed (and would double-count the extinction).
    if (kAdditive == 0 && ofog.litActive > 0.5 && vLight.a >= 0.0) {
        vec3 hdr = lin * vLight.rgb;          // albedo × local light field
        hdr = hdr * vLight.a + vFogAdd;       // camera→particle leg fog
        const vec3 mapped = toneMap(hdr, uint(ofog.toneMapMode + 0.5), ofog.exposure);
        outColor = vec4(linearToSRGB(mapped), alpha);
        return;
    }

    if (ofog.fogActive > 0.5) {
        const float len    = length(vViewPos);
        const float partY  = dot(ofog.viewToWorldY, vViewPos) + ofog.camWorldY;
        const float odAir  = airOpticalDepth(ofog.camWorldY, partY, len);
        const float odMurk = murkOpticalDepth(ofog.camWorldY, partY, len);
        const float odTot  = odAir + odMurk;
        if (odTot > 0.0) {
            const float T = exp(-odTot);
            if (kAdditive != 0) {
                // Additive glow (embers): the mist just ATTENUATES the added
                // radiance — additive blend does not composite toward the
                // background, so there is no in-scatter to inject.
                lin *= T;
            } else {
                // Alpha smoke: reduce COVERAGE by the transmittance so the
                // already-fog-shaded background shows through as the mist
                // thickens (T→0 ⇒ the puff melts INTO the fog). This is exact
                // when the in-front in-scatter over the short camera→particle
                // leg ≈ the fogged background behind it (near-uniform mist):
                //   alpha·(rad·T + bg·(1−T)) + (1−alpha)·bg
                //     = (alpha·T)·rad + (1 − alpha·T)·bg,
                // i.e. rgb unchanged, alpha·=T. Crucially it needs NO fog
                // COLOUR — the particle pass is display-referred and never saw
                // exposure/ACES, so injecting a linear mist colour (the previous
                // fogInscatter path) faded the smoke toward a value BRIGHTER than
                // the tonemapped fog and read as an un-attenuated white plume.
                // The unlit puff's own bright albedo over the visible (alpha·T)
                // fraction is by design (lit particles are a follow-up phase).
                alpha *= T;
            }
        }
    }

    outColor = vec4(linearToSRGB(lin), alpha);
}
