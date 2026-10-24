// main() prologue, part 4 of 4 (inside main(), where the specular terms
// start): the mirror direction, the split-sum terms, the traced band's
// weight and gate, and the emitter-specular split by lobe width that the
// emitter NEE and the reflection trace both key on.
    // Sharp surfaces trace the exact mirror ray; glossy surfaces GGX-importance-
    // sample the reflection lobe so the reflection BLURS with roughness and
    // shows real SCENE geometry — not just the prefiltered environment (the old
    // behaviour, which cut RT reflections off at roughness 0.18 and fell back to
    // env-only). Missed rays read the env at the roughness mip (pre-blurred). A
    // few frame-jittered GGX samples + TAA converge it cleanly on static
    // surfaces; the firefly clamp bounds glossy variance on moving ones. Above
    // ~0.7 roughness the lobe is so wide that prefiltered-env IBL is both
    // cheaper and indistinguishable, so RT bows out there.
    // Specular env strategy: glossy/mirror surfaces (roughness ≲ 0.7) ray-trace
    // real scene geometry (env at the roughness mip on miss); ROUGH surfaces take
    // split-sum env-IBL specular off the PMREM below. The IBL fallback was once
    // dropped because envTex was a plain mip chain (the HDRI sun mirrored as a
    // blocky glare) — but EnvPrefilter has since GGX-prefiltered it (Karis
    // PMREM), so the fallback is restored: without it every matte material had a
    // Fresnel energy hole (rough metals read soot-black under bright HDRIs).
    const vec3  R         = reflect(-V, N);
    const vec2  ab        = envBRDFApprox(NdotV, roughness);
    const float reflBlend = 1.0 - smoothstep(0.55, 0.70, roughness);
    // ONE SHARP MIRROR RAY → the reflection buffer (binding 25). A single ray has
    // NO discrete-GGX ghost copies (the doubled/tripled-object artifact the
    // 8-sample spiral produced) and no per-sample noise. The reflection filter
    // (deferred_refl_filter.comp) then BLURS this by roughness — the spatial blur
    // IS the gloss — and recombines. Pre-weighted by the spec BRDF × reflBlend so
    // the filter just blurs + adds. Glass/sheen reflect inline below (excluded).
    const bool opaqueRefl = (pm.transmission <= 0.0) && (dot(pm.sheenColor, pm.sheenColor) <= 0.0);
    // ── Emitter-specular ESTIMATOR SPLIT by lobe width ──────────────────────
    // specNEET: 0 = near-mirror (the BRDF ray owns emitter spec — tight lobe hits
    // the emitter every frame, no variance, crisp lamp image), 1 = rough (the
    // coherent emissiveSpecNEE owns it — deterministic, settles instantly; the
    // ray's emitter self-emission is scaled out via gReflEmitterScale so nothing
    // double-counts). The NEE add is faded by reflBlend so emitter spec follows
    // the same ≥0.7-roughness specular cutoff as the rest of the spec model.
    // PER LIGHT on top of that: a light whose strata are too far apart for this
    // lobe goes back to the ray, which draws its mirrored image where the strata
    // drew separate highlights (emissive_lights.glsl, kEmRaySpacing0). The NEE
    // keeps 1 - share of each light and the hit scales the emitter it hit by
    // mix(1 - specNEET, 1, share), so the two still add to one. Only where
    // the ray's image is one it can hold: the DETERMINISTIC band (roughness <
    // kReflStochastic, the trace below), where it is one mirror ray, the same
    // image every frame — in the stochastic band a 1-spp GGX sample misses a
    // light 1 time in 4 at y = 12 and drew it as grain on a gold sphere at
    // rest; and a mesh that is NOT MOVING (kInstFlagMoving, its own ids texel)
    // — a turning metal knot reprojects a history of other normals, and the
    // panel over it came out as a jagged patch where the strata draw a smooth
    // band. A floor under that knot still hands off: it is the knot that
    // moves, not the floor. And FLAT (specAaKernelSq, the 4-neighbour normal
    // variance, exactly 0 on a plane): on a thin curved tube the mirror
    // direction turns from pixel to pixel, and one ray through the gloss blur
    // drew the panel on the same knot at rest as scattered patches where the
    // strata draw a smooth band; on a plane the image is coherent and the
    // blur is its gloss. Elsewhere the strata keep every light, the split
    // before the hand-off.
    const float specNEET = smoothstep(0.10, 0.25, roughness);
    // Where the reflection trace GGX-samples the lobe instead of the one mirror
    // ray; the rationale is at the trace. Aligned with specNEET's upper edge.
    const float kReflStochastic = 0.25;
