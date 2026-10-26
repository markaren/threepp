#version 460
#extension GL_GOOGLE_include_directive : enable

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

// linearToSRGB (both paths) and the tone-map curves (lit path). The lit path
// produces LINEAR HDR radiance and must land in the same display domain the
// tonemapped background occupies, so it runs post_composite.comp's own curves
// from the shared header (white balance + grading LUT intentionally omitted).
#include "tonemap.glsl"

// Closed-form exponential-height-fog optical depth along [camY, partY] over the
// leg of length `len`, clipped at the waterline: murkOpticalDepth below covers
// the submerged part of the leg.
#include "height_fog.glsl"
float airOpticalDepth(float camY, float partY, float len) {
    return heightFogLegOpticalDepth(camY, partY, len, ofog.hfDensity, ofog.hfFalloff,
                                    ofog.hfBaseY, ofog.waterSurfaceY);
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
