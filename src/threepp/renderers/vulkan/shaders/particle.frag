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
// ── Phase 2b: unified fog on world-space particles ──────────────────────────
// The overlay pass runs POST-TAA / POST-composite and never saw the fog, so a
// chimney smoke puff (or any ParticleSystem billboard) punched through dense
// mist unattenuated. We now bind the unified air-fog + underwater-murk medium
// (set 1) and, per fragment, compute the closed-form optical depth over the
// camera→particle leg and let the mist attenuate the puff. Closed-form only —
// the froxel LUT is not bound in the overlay pass (the air medium is a smooth
// exponential-height profile, so the closed form IS its exact integral).
// KEEP-IN-SYNC: airOpticalDepth mirrors heightFogOpticalDepth in
// deferred_shade_60_fog_volumetrics.glsl / deferred_filter_common.glsl.
//
// Value domain: the pass is DISPLAY-referred (this shader OETF-encodes and the
// hardware alpha-blends in sRGB over the ACES-tonemapped background). It has NO
// exposure/ACES and NO lighting inputs — particles are UNLIT by design (lit
// particles via the froxel LUT are a planned follow-up). So we deliberately do
// NOT inject a fog COLOUR here: a linear mist radiance would be brighter than the
// tonemapped fog and read as an un-attenuated white plume. Instead the ALPHA
// path reduces the puff's COVERAGE by the transmittance, revealing the already-
// fog-shaded background — value-domain-agnostic and exact in near-uniform mist
// (see main()). optical-depth==0 → the block is skipped → byte-identical to the
// pre-Phase-2b output when no medium is present or the leg is unfogged.

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
    vec3  fogInscatter;  // vestigial: LINEAR air-fog tint (unused — coverage
                         // reduction needs no colour; reserved for lit-fog follow-up)
    vec3  murkInscatter; // vestigial: LINEAR murk tint (unused — see above)
} ofog;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUv;
layout(location = 2) in vec3 vViewPos;

layout(location = 0) out vec4 outColor;

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
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
