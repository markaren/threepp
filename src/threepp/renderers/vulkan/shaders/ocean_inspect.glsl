// Inspector-only copies of water_displace and foam_world evaluators.
// Keep arithmetic in sync with those sources; never refactor their default path.
struct WakeSample {
    float worldX;
    float worldZ;
    float sinYaw;
    float cosYaw;
    float speed;
    float age;
    float _p0;
    float _p1;
};
layout(buffer_reference, scalar) buffer WakeTrailBuf { WakeSample s[]; };
layout(buffer_reference, scalar) buffer OceanHullBuf { OceanHull h[]; };

// hullLocal is shared with displacement through ocean_cascade.glsl.

// Interpolating Catmull-Rom reconstruction of FFT SAMPLES (not spline
// coefficients). An unprefiltered B-spline attenuates a mode even exactly
// at its sample locations: (2+cos(k*dx))/3. At 14 m on the 2.93 m
// cascade this was 0.751 before the derivative stencil/mesh interpolation.
// Catmull-Rom is a C1 cubic approximation to band-limited reconstruction.
// Combine its two positive central weights into one hardware-linear tap;
// keep the negative outer weights separate: 3x3, not 4x4 texture taps.
vec4 sampleInterpolatingCubic(sampler2D tex, vec2 uv) {
    const vec2 res=vec2(textureSize(tex,0));
    const vec2 st=uv*res-0.5, i=floor(st), f=st-i;
    const vec2 f2=f*f, f3=f2*f;
    const vec2 w0=-0.5*f+f2-0.5*f3;
    const vec2 w1=1.0-2.5*f2+1.5*f3;
    const vec2 w2=0.5*f+2.0*f2-1.5*f3;
    const vec2 w3=-0.5*f2+0.5*f3;
    const vec2 wm=w1+w2;
    const vec2 c0=(i-0.5)/res;
    const vec2 cm=(i+0.5+w2/wm)/res;
    const vec2 c3=(i+2.5)/res;
    vec4 value=vec4(0.0);
    value+=w0.y*(w0.x*textureLod(tex,vec2(c0.x,c0.y),0.0)+wm.x*textureLod(tex,vec2(cm.x,c0.y),0.0)+w3.x*textureLod(tex,vec2(c3.x,c0.y),0.0));
    value+=wm.y*(w0.x*textureLod(tex,vec2(c0.x,cm.y),0.0)+wm.x*textureLod(tex,vec2(cm.x,cm.y),0.0)+w3.x*textureLod(tex,vec2(c3.x,cm.y),0.0));
    value+=w3.y*(w0.x*textureLod(tex,vec2(c0.x,c3.y),0.0)+wm.x*textureLod(tex,vec2(cm.x,c3.y),0.0)+w3.x*textureLod(tex,vec2(c3.x,c3.y),0.0));
    return value;
}
float sampleBicubicR(sampler2D tex, vec2 uv) { return sampleInterpolatingCubic(tex,uv).r; }
vec2 inspectSampleBicubicRG(sampler2D tex, vec2 uv) { return sampleInterpolatingCubic(tex,uv).rg; }

vec3 sampleDisplacement(vec2 worldXZ, uint mask) {
    // Per-cascade amplitude normalization: divide by the cascade's tile size.
    // Tessendorf's Phillips spectrum produces unnormalized IFFT output whose
    // amplitude grows with the spatial scale; dividing by tileSize gives a
    // physically-comparable wave height regardless of the cascade's k-range.
    // Matches WebTide's vertex shader convention.
    //
    // Vertical (height) gets `waveScale`, horizontal gets `choppiness`. Choppy
    // waves come from large horizontal displacement and produce sharp Gerstner-
    // like crests when choppiness is too high — keep it ≤ ~0.6 for natural look.
    //
    // Cascades 0+1 are BICUBIC-reconstructed: a bilinear height field is only
    // C0 — its slope kinks along every texel boundary. The adaptive warp packs
    // vertices near the vessel down to ~0.15 m, FINER than the cascade texels
    // (0.98 m / 0.20 m), so the mesh faithfully tessellated those kinks into
    // real geometry: under a mirror-sky reflection every cascade-0 texel read
    // as a hard-edged ~1 m world-aligned tile ("the texels are visible").
    // The interpolating Catmull-Rom kernel is C1, avoiding creases without the unprefiltered B-spline attenuation.
    // Cascade 2 stays bilinear: its 1.6 cm texels are far below the finest
    // vertex spacing, so the mesh never resolves them.
    vec3 d = vec3(0.0);
    if ((inspectData.cascadeMask & mask & 1u) != 0u) {
        const float invTile = 1.0 / inspectData.tile[0];
        const vec2 uv = worldXZ * invTile;
        d.y += sampleBicubicR(inspectHeight0, uv) * invTile;
        const vec2 dxz = inspectSampleBicubicRG(inspectDisplace0, uv) * invTile;
        d.x += dxz.x;
        d.z += dxz.y;
    }
    if ((inspectData.cascadeMask & mask & 2u) != 0u) {
        // Cascade 1 samples a ROTATED domain (repeat-lattice break — see
        // ocean_cascade.glsl); its displacement is a domain-space vector, so
        // rotate it back into world before accumulating.
        const float invTile = 1.0 / inspectData.tile[1];
        const vec2 uv = oceanC1Domain(inspectHeight1, worldXZ, inspectData.tile[1]) * invTile;
        d.y += sampleBicubicR(inspectHeight1, uv) * invTile;
        const vec2 dxz = oceanC1World(inspectHeight1, inspectSampleBicubicRG(inspectDisplace1, uv)) * invTile;
        d.x += dxz.x;
        d.z += dxz.y;
    }
    if ((inspectData.cascadeMask & mask & 4u) != 0u) {
        const float invTile = 1.0 / inspectData.tile[2];
        const vec2 uv = worldXZ * invTile;
        d.y += texture(inspectHeight2,   uv).r * invTile;
        const vec2 dxz = texture(inspectDisplace2, uv).rg * invTile;
        d.x += dxz.x;
        d.z += dxz.y;
    }
    // Linear diffraction is an Eulerian heightfield. All horizontal particle
    // displacement must vanish next to the rigid wall, including the fine
    // cascade: otherwise a short-wave fold can still form a polygon sheet.
    if ((inspectData.cascadeMask & 2u) != 0u) d.xz *= 1.0-oceanContactEulerian(inspectHeight1,worldXZ,inspectData.tile[1]);
    return vec3(d.x * inspectData.choppiness,
                d.y * inspectData.waveScale,
                d.z * inspectData.choppiness);
}

vec3 sampleDisplacement(vec2 worldXZ) { return sampleDisplacement(worldXZ, 7u); }

// Height-only counterpart of sampleDisplacement, for the reflected incident
// field. Same cascade domains, reconstruction and normalization; no horizontal
// texture fetches, hull plane, wake or recursive radiation at the mirror point.
float radiationIncidentHeight(vec2 p, uint mask) {
    float h = 0.0;
    if ((inspectData.cascadeMask & mask & 1u) != 0u) {
        const float invTile = 1.0 / inspectData.tile[0];
        h += sampleBicubicR(inspectHeight0, p * invTile) * invTile;
    }
    if ((inspectData.cascadeMask & mask & 2u) != 0u) {
        const float invTile = 1.0 / inspectData.tile[1];
        h += sampleBicubicR(inspectHeight1, oceanC1Domain(inspectHeight1, p, inspectData.tile[1]) * invTile) * invTile;
    }
    if ((inspectData.cascadeMask & mask & 4u) != 0u) {
        const float invTile = 1.0 / inspectData.tile[2];
        h += texture(inspectHeight2, p * invTile).r * invTile;
    }
    return h * inspectData.waveScale;
}

// Called only behind radEnabled. Deliberately duplicate the footprint math:
// hullLocal and every pre-existing arithmetic statement stay textually intact.
float hullRadiationHeight(OceanHull hu, vec2 p, uint mask) {
    if (hu.halfBeam <= 0.0 || hu.radBand <= 0.0) return 0.0;
    const vec3 hl = hullLocal(hu, p);
    if (hl.z <= 0.0) return 0.0;
    const float u = clamp(-hl.y / hu.halfLength, -1.0, 1.0);
    float halfBeamAtLZ;
    if (u <= 0.0) {
        halfBeamAtLZ = hu.halfBeam * pow(max(1.0 - u * u, 0.0), 0.6);
    } else {
        halfBeamAtLZ = hu.halfBeam * (1.0 - 0.25 * u * u);
    }
    const float edgeX = abs(hl.x) - halfBeamAtLZ;
    const float edgeZ = abs(hl.y) - hu.halfLength;
    const float d = max(edgeX, edgeZ); // exactly hullLocal's edge distance
    if (d >= 3.0 * hu.radBand) return 0.0;

    // Nearest side in the exclusion's max-edge metric, with coordinates
    // clamped onto its tapered outline (not a Euclidean closest-point solve).
    const vec2 q = vec2(clamp(hl.x, -halfBeamAtLZ, halfBeamAtLZ),
                        clamp(hl.y, -hu.halfLength, hu.halfLength));
    const float side = hl.x < 0.0 ? -1.0 : 1.0;
    vec2 normalLocal;
    if (edgeZ >= edgeX || (q.y > 0.0 && halfBeamAtLZ < 1e-4)) {
        normalLocal = vec2(0.0, q.y < 0.0 ? -1.0 : 1.0);
    } else {
        // Outward normal of x=+/-width(z), including the tapered bow/stern.
        const float widthSlope = u <= 0.0
            ? 1.2 * hu.halfBeam * u / hu.halfLength * pow(max(1.0 - u * u, 1e-6), -0.4)
            : 0.5 * hu.halfBeam * u / hu.halfLength;
        normalLocal = normalize(vec2(side, -widthSlope));
    }
    const vec2 zSide = vec2(hu.radHeaveRe, hu.radHeaveIm)
                    + q.x * vec2(hu.radRollRe, hu.radRollIm)
                    + q.y * vec2(hu.radPitchRe, hu.radPitchIm);
    const float phase = hu.radPhase - hu.radK * d;
    const float envelope = exp(-d / hu.radBand) / sqrt(1.0 + d / hu.halfBeam);
    float eta = hu.radAbar * (zSide.x * cos(phase) - zSide.y * sin(phase));

    // refDir is propagation, not a meteorological FROM bearing: the outward
    // weather-side normal opposes it. Smooth to exactly zero at grazing/lee.
    const vec2 normalWorld = vec2(hu.cosYaw * normalLocal.x + hu.sinYaw * normalLocal.y,
                                 -hu.sinYaw * normalLocal.x + hu.cosYaw * normalLocal.y);
    const float weather = smoothstep(0.0, 0.35, -dot(normalWorld, vec2(hu.refDirX, hu.refDirZ)));
    if (weather > 0.0 && hu.refR > 0.0) {
        // Local tangent-line reflection at the clamped waterline point.
        const vec2 mirrorLocal = hl.xy - 2.0 * dot(hl.xy - q, normalLocal) * normalLocal;
        const vec2 mirrorWorld = vec2(hu.centerX + hu.cosYaw * mirrorLocal.x + hu.sinYaw * mirrorLocal.y,
                                      hu.centerZ - hu.sinYaw * mirrorLocal.x + hu.cosYaw * mirrorLocal.y);
        eta += hu.refR * weather * radiationIncidentHeight(mirrorWorld, mask);
    }
    return hl.z * envelope * eta;
}

float sampleHullRadiation(vec2 p, uint mask) {
    OceanHullBuf hulls = OceanHullBuf(inspectData.hullAddress);
    float height = 0.0;
    for (uint k = 0u; k < inspectData.hullCount; ++k) {
        const OceanHull hu = hulls.h[k];
        if (hu.radEnabled > 0.0) {
            float otherFade = 1.0;
            for (uint j = 0u; j < inspectData.hullCount; ++j) {
                if (j != k) otherFade *= hullLocal(hulls.h[j], p).z;
            }
            if (otherFade > 0.0) height += otherFade * hullRadiationHeight(hu, p, mask);
        }
    }
    return height;
}


vec3 inspectSurface(vec2 p) {
    vec3 rest = vec3(p.x,0.0,p.y);
    vec3 disp = sampleDisplacement(rest.xz);

    OceanHullBuf hulls = OceanHullBuf(inspectData.hullAddress);

    // === Hull exclusion: every vessel's footprint ===
    // Suppress displacement inside a vessel's footprint so the ocean mesh
    // doesn't clip through the deck. Smooth fade over the margin outside the
    // hull edge keeps the transition invisible.
    //
    // The target inside the footprint is the vessel's OWN WATERLINE PLANE,
    // not the ocean rest plane: a hull that heaves and pitches with the sea
    // has to carry its waterline with it, or a boat sitting on a 2 m crest
    // gets a 2 m hole punched in the water around her. `centerY` is the
    // world height of the hull's design waterline at the exclusion centre and
    // pitch/roll tilt it in the vessel-local frame (hullLocal). With all
    // three at their defaults this is bit-for-bit the old `disp *= hullFade`.
    // All footprints go first, so the wakes below all see the finished
    // surface; with one vessel the order is exactly the old one.
    for (uint k = 0u; k < inspectData.hullCount; ++k) {
        const OceanHull hu = hulls.h[k];
        const vec3  hl     = hullLocal(hu, rest.xz);
        const float planeY = hu.centerY
                           + tan(clamp(hu.pitch, -1.0, 1.0)) * hl.y
                           + tan(clamp(hu.roll,  -1.0, 1.0)) * hl.x;
        disp.y  = mix(planeY, disp.y, hl.z);
        disp.x *= hl.z;
        disp.z *= hl.z;
    }

    // === Vessel wakes ===
    // Analytical wake from each vessel's pose + forward speed:
    //   B. Bow bump (positive height bulge centred at the prow)
    //   C/D. Bow V-wedge (two ridges of pushed-aside water along the hull
    //      sides, forming a V whose vertex is at the bow), traced along the
    //      vessel's own trail
    // (The foam trail and ridge foam live in foam_world.comp.)
    //
    // All effects are gated by a smoothstep on |forwardSpeed| so the wake
    // fades to nothing below ~0.5 m/s — avoids the v²-wavelength blow-up
    // at v=0 and makes anchored vessels (forwardSpeed=0) wake-free without
    // a separate branch.
    //
    // The bow bump and the current-pose V-wedge extend INTO the hull
    // rectangle near the prow and along the sides; they get multiplied by
    // the vessel's own `hullFade` so they fall to zero inside the hull and
    // don't clip through the deck. Every OTHER vessel's footprint fades the
    // whole wake (`otherFade`), so a ship's V-wedge does not lift the water
    // through a smaller boat's deck. With one vessel otherFade is exactly 1.
    //
    // Normal finite-diff taps below do NOT re-sample the wake — its
    // contribution to shading normals is small at long wavelengths and
    // re-sampling would 4× the wake math per vertex.
    for (uint k = 0u; k < inspectData.hullCount; ++k) {
        const OceanHull hu = hulls.h[k];
        const vec3  hl         = hullLocal(hu, rest.xz);
        const float wakeLocalX = hl.x;
        const float wakeLocalZ = hl.y;
        float otherFade = 1.0;
        for (uint j = 0u; j < inspectData.hullCount; ++j) {
            if (j != k) otherFade *= hullLocal(hulls.h[j], rest.xz).z;
        }
        const float hullFade = hl.z * otherFade;

        const float wakeSpeed = abs(hu.forwardSpeed);
        const float wakeGate  = smoothstep(0.5, 1.5, wakeSpeed);
        if (wakeGate > 0.0) {
            const float lX = wakeLocalX;
            const float lZ = wakeLocalZ;

            // --- B. Bow bump: gaussian centred at the bow (lZ = +halfLength) ---
            // Amplitude scales with v² (Bernoulli-flavoured), clamped to keep
            // tall-wave visuals stable. Multiplied by hullFade so the bump
            // disappears inside the prow and doesn't poke through the deck.
            const float distFromBow = lZ - hu.halfLength;            // 0 at bow, +ve ahead
            const float bowR        = 1.5;
            const float bowG        = exp(-(distFromBow * distFromBow) / (bowR * bowR));
            const float bowL        = exp(-(lX * lX) / (hu.halfBeam * hu.halfBeam));
            const float wakeSpeedSq = hu.forwardSpeed * hu.forwardSpeed;
            const float bowAmp      = clamp(0.012 * wakeSpeedSq, 0.0, 0.4);
            disp.y += hullFade * wakeGate * bowAmp * bowG * bowL;
        }

        // --- Historical V-wedge — two sloped ridges diverging from the
        // bow at ~20° half-angle, traced along the boat's past path.
        //
        // For each snapshot of the boat's pose in its trail slice,
        // compute that sample's V-wedge ridge value at this vertex and
        // take MAX across all samples. MAX (not sum) is the right
        // combiner here because the ridges are PHYSICAL CRESTS — the
        // boundary lines of the V-wake — not interfering wave trains.
        // Summing per-sample cosines was what produced the fountain-
        // shaped pillars: 60 samples × cos(...) of different phases
        // stack constructively into ~√60·amp peaks.
        //
        // With MAX along the curved path, each ridge crest is the
        // envelope of the V-wedges contributed by samples whose bow
        // pose places this vertex on their boundary line. The result
        // is two SLOPED RIDGES bending with the boat's turn, fading
        // backward as the trail ages out.
        float vWedgeTrail = 0.0;
        if (hu.trailCount > 0u) {
            WakeTrailBuf wt = WakeTrailBuf(inspectData.wakeTrailAddress);
            for (uint si = 0u; si < hu.trailCount; ++si) {
                const WakeSample s = wt.s[hu.trailFirst + si];
                const float sSpeed = abs(s.speed);
                const float sGate  = smoothstep(0.5, 1.5, sSpeed);
                if (sGate <= 0.0) continue;
                const float dx = rest.x - s.worldX;
                const float dz = rest.z - s.worldZ;
                const float sLX = s.cosYaw * dx - s.sinYaw * dz;
                const float sLZ = s.sinYaw * dx + s.cosYaw * dz;
                const float sDistAft = hu.halfLength - sLZ;
                if (sDistAft <= 0.0 || sDistAft > hu.halfLength * 6.0) continue;
                // Per-sample hullFade — same silhouette taper as the
                // current-pose hull exclusion above, but anchored to THIS
                // sample's pose. Suppresses V-wedge ridges inside the
                // sample's hull footprint so the wake doesn't lift water
                // under the boat as the boat (or any historical past
                // position of it) coincides with this vertex.
                const float uHull = clamp(-sLZ / hu.halfLength, -1.0, 1.0);
                float sHalfBeamAtLZ;
                if (uHull <= 0.0) {
                    sHalfBeamAtLZ = hu.halfBeam *
                                    pow(max(1.0 - uHull * uHull, 0.0), 0.6);
                } else {
                    sHalfBeamAtLZ = hu.halfBeam * (1.0 - 0.25 * uHull * uHull);
                }
                const float sHullEdgeX = abs(sLX) - sHalfBeamAtLZ;
                const float sHullEdgeZ = abs(sLZ) - hu.halfLength;
                const float sHullFade  = smoothstep(0.0, oceanHullMargin(hu.halfBeam),
                                                    max(sHullEdgeX, sHullEdgeZ));
                if (sHullFade <= 0.0) continue;
                const float tanAV     = 0.36;
                const float expectedX = tanAV * sDistAft;
                const float dRidge    = abs(abs(sLX) - expectedX);
                const float sigmaR    = 0.6 + 0.05 * sDistAft;
                const float ridge     = exp(-(dRidge * dRidge) / (sigmaR * sigmaR));
                const float alongDec  = exp(-sDistAft / (hu.halfLength * 4.0));
                const float ageFade   = exp(-s.age / 5.0);
                const float vAmp      = clamp(0.016 * sSpeed * sSpeed +
                                              0.06 * sSpeed, 0.0, 1.0);
                vWedgeTrail = max(vWedgeTrail,
                                  sGate * vAmp * ridge * alongDec * ageFade * sHullFade);
            }
            disp.y += vWedgeTrail * otherFade;
        } else {
            // Fallback: no historical trail — emit a single V-wedge from
            // the current pose. Bow-anchored ridges, same shape as the
            // trail-summed historical path.
            const float distAft = hu.halfLength - wakeLocalZ;
            if (wakeGate > 0.0 && distAft > 0.0 && distAft < hu.halfLength * 6.0) {
                const float tanAV     = 0.36;
                const float expectedX = tanAV * distAft;
                const float dRidge    = abs(abs(wakeLocalX) - expectedX);
                const float sigmaR    = 0.6 + 0.05 * distAft;
                const float ridge     = exp(-(dRidge * dRidge) / (sigmaR * sigmaR));
                const float alongDec  = exp(-distAft / (hu.halfLength * 4.0));
                const float vAmp      = clamp(0.016 * wakeSpeed * wakeSpeed +
                                              0.06 * wakeSpeed, 0.0, 1.0);
                // hullFade is this vessel's current-pose silhouette (times
                // the other footprints) — same suppression-inside-hull
                // behaviour as the trail path.
                disp.y += hullFade * wakeGate * vAmp * ridge * alongDec;
            }
        }
    }

    // Uniform per-hull switch, independent of forward speed. In particular,
    // do not even add a floating-point zero to the historical default path.
    bool radiationEnabled = false;
    for (uint k = 0u; k < inspectData.hullCount; ++k) {
        if (hulls.h[k].radEnabled > 0.0) radiationEnabled = true;
    }
    if (radiationEnabled) {
        disp.y += sampleHullRadiation(rest.xz, 7u);
    }

    return disp;
}
vec2 inspectFoamCubic(sampler2D tex, vec2 uv) {
    const vec2 res = vec2(textureSize(tex, 0));
    const vec2 st  = uv * res - 0.5;
    const vec2 i   = floor(st);
    const vec2 f   = st - i;
    const vec2 f2  = f * f;
    const vec2 f3  = f2 * f;
    const vec2 w0  = (-f3 + 3.0 * f2 - 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w1  = (3.0 * f3 - 6.0 * f2 + 4.0)       * (1.0 / 6.0);
    const vec2 w2  = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w3  = f3 * (1.0 / 6.0);
    const vec2 g0  = w0 + w1;
    const vec2 g1  = w2 + w3;
    const vec2 c0  = (i - 0.5 + w1 / g0) / res;
    const vec2 c1  = (i + 1.5 + w3 / g1) / res;
    const vec2 t00 = texture(tex, vec2(c0.x, c0.y)).rg;
    const vec2 t10 = texture(tex, vec2(c1.x, c0.y)).rg;
    const vec2 t01 = texture(tex, vec2(c0.x, c1.y)).rg;
    const vec2 t11 = texture(tex, vec2(c1.x, c1.y)).rg;
    return g0.y * (g0.x * t00 + g1.x * t10) + g1.y * (g0.x * t01 + g1.x * t11);
}

// Bicubic variant of sampleDispXZ — used ONLY inside the Jacobian stencil
// (the inverse-map keeps the cheap bilinear path: kinks in the VALUE field
// are second-order there and invisible).
vec2 sampleDispXZSmooth(vec2 worldXZ, uint mask) {
    vec2 d = vec2(0.0);
    if ((mask & inspectData.cascadeMask & 1u) != 0u) {
        const float invTile = 1.0 / inspectData.tile[0];
        d += inspectFoamCubic(inspectDisplace0, worldXZ * invTile) * invTile;
    }
    if ((mask & inspectData.cascadeMask & 2u) != 0u) {
        // Rotated domain + world back-rotation — must match water_displace.
        const float invTile = 1.0 / inspectData.tile[1];
        if (!oceanC1Local(inspectHeight1)) d += oceanC1World(inspectFoamCubic(inspectDisplace1, oceanC1Domain(worldXZ) * invTile)) * invTile;
    }
    if ((mask & inspectData.cascadeMask & 4u) != 0u) {
        const float invTile = 1.0 / inspectData.tile[2];
        d += inspectFoamCubic(inspectDisplace2, worldXZ * invTile) * invTile;
    }
    return d * inspectData.choppiness;
}

// 1 − J of (x,z) → (x,z) + Dxz over the masked cascades, central-differenced
// at stencil half-width `eps`. Positive = the surface is compressing/folding
// (a breaking crest); the stencil width sets which wavelengths register.
float jacobianFold(vec2 p, uint mask, float eps) {
    const vec2 dL = sampleDispXZSmooth(p + vec2(-eps, 0.0), mask);
    const vec2 dR = sampleDispXZSmooth(p + vec2( eps, 0.0), mask);
    const vec2 dD = sampleDispXZSmooth(p + vec2(0.0, -eps), mask);
    const vec2 dU = sampleDispXZSmooth(p + vec2(0.0,  eps), mask);
    const float inv2e = 0.5 / eps;
    const float dDxDx = (dR.x - dL.x) * inv2e;
    const float dDxDz = (dU.x - dD.x) * inv2e;
    const float dDzDx = (dR.y - dL.y) * inv2e;
    const float dDzDz = (dU.y - dD.y) * inv2e;
    return 1.0 - ((1.0 + dDxDx) * (1.0 + dDzDz) - dDxDz * dDzDx);
}


vec2 inspectHorizontal(vec2 q) {
    vec2 d=sampleDisplacement(q).xz;
    OceanHullBuf hulls=OceanHullBuf(inspectData.hullAddress);
    for(uint k=0u;k<inspectData.hullCount;++k) d*=hullLocal(hulls.h[k],q).z;
    return d;
}
// Encoded sRGB palette, deliberately bypassing exposure/tone mapping later.
vec3 inspectPalette(float v, float lo, float hi) {
    float t=clamp((v-lo)/(hi-lo),0.0,1.0);
    if (lo<0.0) return t<0.5 ? mix(vec3(0.05,0.2,0.9),vec3(1),2.0*t) : mix(vec3(1),vec3(0.9,0.08,0.03),2.0*t-1.0);
    return t<0.5 ? mix(vec3(0.04,0.06,0.18),vec3(0.0,0.65,0.65),2.0*t) : mix(vec3(0.0,0.65,0.65),vec3(1.0,0.9,0.1),2.0*t-1.0);
}
void inspectNonWater(ivec2 px) {
    if ((pc.inspectMode < 16u || pc.inspectMode == 17u) && pc.shadeMode == 0u) imageStore(outImage,px,vec4(vec3(0.12),1));
}
void inspectWater(ivec2 px, vec3 P, float seaY, float banked, float kernel) {
    if (pc.inspectMode == 17u) {
        imageStore(outImage,px,vec4(inspectPalette(gInspectFreshAlpha,0.0,1.0),1));
        if (px == ivec2(inspectData.pixelX,inspectData.pixelY)) {
            inspectData.valid=1u; inspectData.worldX=P.x; inspectData.worldZ=P.z;
            inspectData.height=P.y-seaY; inspectData.value=gInspectFreshAlpha;
        }
        return;
    }
    if (pc.inspectMode == 16u && px != ivec2(inspectData.pixelX,inspectData.pixelY)) return;
    // Shading-only fields need no displacement inversion except at the probe.
    if (px != ivec2(inspectData.pixelX,inspectData.pixelY) && pc.inspectMode >= 8u && pc.inspectMode <= 14u) {
        float v=0.0, hi=1.0;
        if(pc.inspectMode==8u) v=pc.oceanFoamTileSize>0.0 ? clamp(sampleFoamBicubic(gFoamXZ/pc.oceanFoamTileSize),0.0,1.0) : 0.0;
        if(pc.inspectMode==9u) {v=banked;hi=0.25;}
        if(pc.inspectMode==10u) {v=dot(gInspectGlitter,vec3(0.2126,0.7152,0.0722));hi=10.0;}
        if(pc.inspectMode==11u) v=gInspectReflShare;
        if(pc.inspectMode==12u) v=gInspectRayShare;
        if(pc.inspectMode==13u) v=gInspectTransShare;
        if(pc.inspectMode==14u) v=max(gWake.r,gWake.g);
        imageStore(outImage,px,vec4(inspectPalette(v,0.0,hi),1));
        return;
    }
    // Inverse of the same horizontal map, including hull exclusion and contact.
    // Fixed-point inversion is approximate in overturning/folded regions.
    vec2 q=P.xz;
    for(int i=0;i<12;++i) q=P.xz-inspectHorizontal(q);
    vec3 d=inspectSurface(q);
    float e=1e20;
    if((inspectData.cascadeMask&1u)!=0u && inspectData.tile[0]>0.0) e=min(e,inspectData.tile[0]/float(textureSize(inspectHeight0,0).x));
    if((inspectData.cascadeMask&2u)!=0u && inspectData.tile[1]>0.0) e=min(e,inspectData.tile[1]/float(textureSize(inspectHeight1,0).x));
    if((inspectData.cascadeMask&4u)!=0u && inspectData.tile[2]>0.0) e=min(e,inspectData.tile[2]/float(textureSize(inspectHeight2,0).x));
    e=clamp(e,0.01,1.0);
    vec3 dx=(inspectSurface(q+vec2(e,0))-inspectSurface(q-vec2(e,0)))/(2.0*e);
    vec3 dz=(inspectSurface(q+vec2(0,e))-inspectSurface(q-vec2(0,e)))/(2.0*e);
    mat2 m=mat2(vec2(1.0+dx.x,dx.z),vec2(dz.x,1.0+dz.z));
    float J=determinant(m);
    float slope=abs(J)>1e-6 ? length(transpose(inverse(m))*vec2(dx.y,dz.y)) : 1e6;
    float rad=sampleHullRadiation(q,7u);
    float rawFoam=pc.oceanFoamTileSize>0.0 ? clamp(sampleFoamBicubic(gFoamXZ/pc.oceanFoamTileSize),0.0,1.0) : 0.0;
    float cj=1.0-jacobianFold(q,1u,max(inspectData.tile[0]/256.0,0.01));
    float fj=1.0-jacobianFold(q,4u,max(1.5,inspectData.foamTile/max(float(inspectData.foamRes),1.0)));
    // foam_world's feedback thresholds, before its moving group gate.
    float ct=1.0-inspectData.coarseFold+0.2*rawFoam;
    float ft=1.0-inspectData.fineFold+(0.2/3.0)*rawFoam;
    float value=0.0, lo=0.0, hi=1.0;
    bool inactive=false;
    if(pc.inspectMode<=3u) {
        uint c=pc.inspectMode-1u;
        inactive=(inspectData.cascadeMask&(1u<<c))==0u || inspectData.tile[c]<=0.0;
        if(!inactive) {
            if(c==0u) value=sampleBicubicR(inspectHeight0,q/inspectData.tile[0])/inspectData.tile[0]*inspectData.waveScale;
            if(c==1u) value=sampleBicubicR(inspectHeight1,oceanC1Domain(inspectHeight1,q,inspectData.tile[1])/inspectData.tile[1])/inspectData.tile[1]*inspectData.waveScale;
            if(c==2u) value=textureLod(inspectHeight2,q/inspectData.tile[2],0).r/inspectData.tile[2]*inspectData.waveScale;
        }
        hi=c==0u?5.0:(c==1u?2.0:0.5);lo=-hi;
    }
    else if(pc.inspectMode==4u) {value=P.y-seaY;lo=-5.0;hi=5.0;}
    else if(pc.inspectMode==5u) {value=length(d.xz);hi=5.0;}
    else if(pc.inspectMode==6u) {value=slope;hi=2.0;}
    else if(pc.inspectMode==7u) {value=J-1.0;lo=-1.0;hi=1.0;}// J-1: undisturbed water (J=1) is white, compression blue, a fold (J<=0) the blue end
    else if(pc.inspectMode==8u) value=rawFoam;
    else if(pc.inspectMode==9u) {value=banked;hi=0.25;}
    else if(pc.inspectMode==10u) {value=dot(gInspectGlitter,vec3(0.2126,0.7152,0.0722));hi=10.0;}
    else if(pc.inspectMode==11u) value=gInspectReflShare;
    else if(pc.inspectMode==12u) value=gInspectRayShare;
    else if(pc.inspectMode==13u) value=gInspectTransShare;
    else if(pc.inspectMode==14u) value=max(gWake.r,gWake.g);
    else if(pc.inspectMode==15u) {value=rad;lo=-1.0;hi=1.0;}
    if(px==ivec2(inspectData.pixelX,inspectData.pixelY)) {
        inspectData.valid=1u; inspectData.worldX=P.x; inspectData.worldZ=P.z; inspectData.height=P.y-seaY;
        inspectData.slope=slope; inspectData.jacobian=J;inspectData.restX=q.x;inspectData.restZ=q.y;
        inspectData.foam=rawFoam;inspectData.banked=banked;inspectData.fineKernel=kernel;
        for(int i=0;i<4;++i) inspectData.wake[i]=gWake[i];
        inspectData.radiation=rad;inspectData.value=value;inspectData.coarseJ=cj;inspectData.fineJ=fj;
        inspectData.coarseThreshold=ct;inspectData.fineThreshold=ft;
    }
    if(pc.inspectMode<16u) {
        vec3 color=inactive?vec3(0.12):inspectPalette(value,lo,hi);
        // Magenta stipple marks either actual foam deposition test, not a fictional single J threshold.
        if(pc.inspectMode==7u && (cj<ct || (inspectData.fineFold>0.0 && fj<ft)) && ((px.x+px.y)&7)==0) color=vec3(1,0,1);
        imageStore(outImage,px,vec4(color,1));
    }
}
