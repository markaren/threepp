// main() prologue, part 3 of 4 (inside main(), after the ocean early-out):
// the specular AA kernel and the roughness it widens, metalness, emission,
// the material AO, F0 with iridescence, the shadow-ray origin and the
// demodulation decision.
    // GEOMETRIC SPECULAR AA (Kaplanyan-style): widen the GGX roughness by the
    // 4-neighbour G-buffer normal variance, so sub-pixel normal detail (dense
    // normal maps, curvature, distant meshes) shades with the roughness its
    // pixel footprint actually spans. At 1 spp under the Halton jitter a
    // sub-pixel glint otherwise flips bright/dark every frame — an alternating
    // input TAA can never converge (the "distant edges shimmer" bug; the
    // original fix measured 4.5× lower consecutive-frame diff on the ocean
    // harness). kernel² capped at 0.04 so macro geometry edges don't go matte.
    // Sits BELOW the water return: an ocean pixel consumes none of this (shadeWater
    // derives its own roughness from fineKernelSq and reads no emissive map), so
    // for it the four neighbour fetches and the emissive tap were pure dead work.
    // The variance is measured against Nraster, the pre-flip pre-tilt normal — see
    // its capture above for why differencing against the mutated N goes matte.
    float specAaKernelSq = 0.0;
    {
        const ivec2 saMax = ivec2(int(pc.width), int(pc.height)) - 1;
        const ivec2 saOff[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
        float varN = 0.0;
        for (int i = 0; i < 4; ++i) {
            const vec4 nn = texelFetch(gbufNormalTex, clamp(px + saOff[i], ivec2(0), saMax), 0);
            if (nn.w <= 0.0) continue;// sky (0) / unlit (<0) neighbours carry no surface normal
            const vec3 dn = normalize(nn.xyz * 2.0 - 1.0) - Nraster;
            varN += dot(dn, dn) * 0.25;
        }
        specAaKernelSq = min(2.0 * varN, 0.04);
    }
    const float roughMat  = clamp(nTap.w, 0.04, 1.0);
    const float roughness = min(sqrt(roughMat * roughMat + specAaKernelSq), 1.0);
    const float metalness = clamp(am.a, 0.0, 1.0);

    // Emissive: factor × emissive MAP. glTF sets emissiveFactor=(1,1,1) whenever
    // an emissive texture is present (the MAP, not the factor, carries the actual
    // emission), so applying the bare factor makes textured-emissive submeshes
    // (e.g. the SWAT exoskeleton) glow full white. The primary UV comes from the
    // G-buffer (gbuffer.frag attachment 3). Mirrors the reflection-hit path
    // (hitTex at the hit UV); texIndex < 0 → factor unchanged (no regression).
    const vec4 primUvTap  = (msSampleIdx >= 0) ? texelFetch(gbufUvMS, px, msSampleIdx) : texelFetch(gbufUvTex, px, 0);
    const vec2 primUv     = primUvTap.rg;
    const vec3 emissive   = hitTex(pm.emissiveTexIndex, pm.uvTransformEmissive, primUv,
                                   pm.emissive * pm.emissiveIntensity);
    // The material's ambient occlusion (aoMap), sampled by gbuffer.frag into the
    // same attachment's .a as 1 - ao. It scales the INDIRECT light only: the
    // diffuse indirect with the scene ambient (diffIndW below, on the denoised
    // and the inline path alike) and the env specular (specularOcclusion, as
    // three.js's computeSpecularOcclusion). 1 exactly without a map.
    const float matAo     = 1.0 - primUvTap.a;

    const float NdotV = max(dot(N, V), 1e-4);
    const vec3  dielF0 = dielectricF0(pm.ior, pm.specularIntensity, pm.specularColor);
    vec3        F0    = mix(dielF0, albedo, metalness);
    // Thin-film iridescence (KHR_materials_iridescence). Modulates F0 with the
    // wavelength-dependent interference colour; lobe shape (GGX) is unchanged.
    // Skipped when factor == 0 → non-iridescent surfaces pay only the branch.
    if (pm.iridescence > 0.0) {
        const vec3 irid = evalIridescence(1.0, pm.iridescenceIOR, NdotV,
                                          pm.iridescenceThicknessNm, F0);
        F0 = mix(F0, irid, pm.iridescence);
    }
    const vec3  shadowOrig = worldPos + N * SHADOW_EPS;

    // Demodulated (binding 16, accumulated + denoised) vs INLINE diffuse indirect.
    // demodIndirect (opaque, sheen included) → the GI is demodulated, temporally
    // accumulated + à-trous denoised, and the recombine multiplies it by this
    // pixel's demodColor (base diffuse, sheen lobe, env-specular share). INLINE
    // (transparent/glass) → the GI is used directly. Computed BEFORE gatherEnv so
    // the gather picks the right path: stochastic real GI ONLY for the demod path
    // (it gets accumulated/denoised); the INLINE path uses the DETERMINISTIC GI
    // (1-spp would just be noisy there — that was the transparent-quad wobble).
    const bool denoiseOn = flagOn(4u);
    // Dispatch B forces the INLINE path (demodIndirect=false) — same
    // precedent as glass: a self-contained pixel color with no
    // dependency on the demod GI accumulator's cross-frame state, which
    // dispatch B never writes (see the GI/SVGF gate near the end of main()).
    // Sheen used to be excluded here too, because the recombine rebuilt its
    // factor from the G-buffer albedo and a scalar and the sheen response is
    // a vec3; the one-sample ReSTIR emissive light then reached a velvet
    // pixel raw and sparkled every frame. demodColor carries the vec3 now.
    const bool demodIndirect = pc.shadeMode != 1u && denoiseOn && (pm.transmission <= 0.0);
