// FJORD — a procedural Norwegian fjord through a full day (Vulkan deferred showcase).
//
// Everything is generated in-engine — no asset downloads:
//   • terrain::TerrainGenerator — ridged multifractal + hydraulic/thermal erosion,
//     then a fjord channel is carved along an S-curve (deep water, a green shore
//     bench, steep walls) and a slope/altitude splat albedo is baked from the
//     carved heights (wet shore rock → heath → scree → rock → snow).
//   • Ocean — the FFT water (three Phillips cascades, foam) fills the channel
//     at sea level y=0 as a static channel-fitting rectangle.
//   • vegetation::TreeGenerator — pine + birch prototypes instanced along the
//     shore bench (detailed cross-quad leaf cards near the cabin, cheap blob
//     canopies for the far bank), plus boulders and a GrassMesh meadow
//     (GPU wind compute + BLAS refit).
//   • A red cabin ("hytte") with white trim, glowing windows after sundown,
//     a chimney plume that is a PARTICIPATING MEDIUM (a smoke-only FireEffect —
//     one ParticleField with a DensityRepr, so the sun scatters through it and
//     it composites with the fjord's fog instead of over it; --legacy-smoke
//     brings back the old sprite plume for an A/B), a wooden dock with a
//     lantern, and a moored rowing boat.
//   • A CPU port of the three.js Preetham sky is baked into an equirect float
//     env map (ping-pong pair, throttled re-bakes) so the sky, IBL ambient and
//     water reflections all track the sun continuously — dawn mist and god
//     rays (setVolumetricFog), golden hour, night with a moon, procedural
//     stars (setDeferredStarfield) and warm window light on the water.
//
// Deferred features exercised: RT AO/GI + probe GI, RT soft sun shadows
// (setSunAngularRadius), directional volumetric fog, starfield, auto-exposure,
// bloom, G-buffer MSAA, TAA render-scale, FFT ocean, GrassMesh wind, particle
// overlay billboards, emissive mesh lights.
//
// Controls:  drag = orbit   scroll = zoom   C = cinematic camera   SPACE = play/pause time
//
// ONE WIND: the waves, the cloud deck, the grass and the chimney plume all read
// the same heading and speed (the ocean spectrum and the cloud drift used to
// disagree with the ground weather by 90°), and the panel's wind slider moves
// all of them together. The surface wind is a documented fraction of the free
// stream — see groundWindAt.
//
// The ParticleField weather this demo briefly carried (a shoreline campfire and
// a valley snowfall, plans/particle-atmosphere.md F4) has been REMOVED: the
// fjord is a terrain/ocean/vegetation demo and the atmosphere work has demos of
// its own — `vulkan_fire` for the campfire and `vulkan_snow` for the snowfall,
// both of which exercise it harder than a corner of this scene did. The CHIMNEY
// is the exception, and deliberately so: it is not weather bolted onto the
// valley, it is the cabin's own smoke, it has been in this demo since before the
// atmosphere work existed, and it survived that removal. It uses exactly one of
// the four density-volume slots.
//
// Headless:  vulkan_fjord --shot out.png [--frames N] [--time H] [--view 0..6] [--cycle H_per_s]

#include "capture_util.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"

#include "threepp/extras/architecture/LogCabin.hpp"
#include "threepp/extras/effects/FireEffect.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/RockGeometry.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/extras/vegetation/GrassTiles.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/FogExp2.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"
#include "threepp/utils/Parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kPi = 3.14159265358979f;
    constexpr float kTau = 6.28318530718f;

    // ═════════════════════════════ world layout ═════════════════════════════

    constexpr float kWorldSize = 3000.f;// terrain extent (m)
    constexpr int kRes = 512;           // terrain grid segments per side
    constexpr float kAmplitude = 470.f; // peak elevation (m)

    // Fjord channel: an S-curve running along Z. All carve/placement queries
    // measure the distance from a sample to the channel centreline.
    float channelCenterX(float z) {
        return 150.f * std::sin(z * 0.0019f) + 80.f * std::sin(z * 0.00087f + 1.6f);
    }

    constexpr float kChannelHalf = 130.f;// water half-width at full depth
    constexpr float kBenchEnd = 250.f;   // shore bench outer edge (distance from centre)
    constexpr float kWallEnd = 580.f;    // walls reach natural terrain height here
    constexpr float kFloorDepth = -30.f; // channel floor (m)

    // Shore bench profile: -3 m at the channel edge rising to +7 m — puts a
    // walkable green strip (and the waterline) on both banks everywhere.
    float benchHeight(float d) {
        const float t = std::clamp((d - kChannelHalf) / (kBenchEnd - kChannelHalf), 0.f, 1.f);
        return -3.f + 10.f * t;
    }


    // Cheap deterministic 2D value noise (independent of the terrain seed).
    float hash01(int x, int y) {
        uint32_t n = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u + 0x9E3779B9u;
        n = (n ^ (n >> 13)) * 1274126177u;
        return static_cast<float>((n ^ (n >> 16)) & 0xffffffu) / static_cast<float>(0xffffff);
    }
    float vnoise(float x, float y) {
        const int xi = static_cast<int>(std::floor(x)), yi = static_cast<int>(std::floor(y));
        auto sm = [](float t) { return t * t * (3.f - 2.f * t); };
        const float fx = sm(x - std::floor(x)), fy = sm(y - std::floor(y));
        const float a = hash01(xi, yi), b = hash01(xi + 1, yi);
        const float c = hash01(xi, yi + 1), d = hash01(xi + 1, yi + 1);
        return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
    }
    float fbm2(float x, float y) {
        return 0.55f * vnoise(x, y) + 0.30f * vnoise(x * 2.13f + 7.3f, y * 2.13f) +
               0.15f * vnoise(x * 4.7f, y * 4.7f + 3.1f);
    }

    // Cabin pad (flattened build site on the east bench).
    constexpr float kPadZ = 260.f;
    constexpr float kPadDist = 205.f;// distance from channel centre
    constexpr float kPadHeight = 3.2f;

    // ═════════════════════ terrain: carve + splat bake ══════════════════════

    terrain::HeightGrid carveFjord(const terrain::TerrainGenerator& gen, const terrain::TerrainParams& tp,
                                   float padX, float padZ) {
        const int dim = gen.dim();
        const float step = tp.worldSize / static_cast<float>(dim - 1);
        const float half = tp.worldSize * 0.5f;
        std::vector<float> h(gen.getField().size());

        const auto& base = gen.getField();
        std::vector<int> rows(static_cast<size_t>(dim));
        std::iota(rows.begin(), rows.end(), 0);
        parallelForEach(rows.begin(), rows.end(), [&](int iz) {
            const float z = -half + static_cast<float>(iz) * step;
            const float cx = channelCenterX(z);
            for (int ix = 0; ix < dim; ++ix) {
                const float x = -half + static_cast<float>(ix) * step;
                const float d = std::abs(x - cx);
                const float natural = base[static_cast<size_t>(iz) * dim + ix] * tp.amplitude;

                // Channel floor → shore bench → natural walls.
                const float floorH = kFloorDepth + 4.f * fbm2(x * 0.004f, z * 0.004f);
                const float bench = benchHeight(d) + 2.2f * (fbm2(x * 0.012f, z * 0.012f) - 0.5f);
                const float sFloor = math::smoothstep(kChannelHalf - 90.f, kChannelHalf, d);// floor→bench edge
                const float sWall = math::smoothstep(kBenchEnd, kWallEnd, d);               // bench→natural
                float hm = floorH + (bench - floorH) * sFloor;
                hm = hm + (std::max(natural, hm) - hm) * sWall;

                // Flatten the cabin pad.
                const float dp = std::sqrt((x - padX) * (x - padX) + (z - padZ) * (z - padZ));
                const float sPad = math::smoothstep(24.f, 52.f, dp);
                hm = kPadHeight + (hm - kPadHeight) * sPad;

                h[static_cast<size_t>(iz) * dim + ix] = hm;
            }
        });
        return {std::move(h), dim, tp.worldSize};
    }

    // Sub-grid relief for near tiles: noise octaves below the 512² base grid's
    // Nyquist, slope-modulated (meadows stay smooth, rock gets craggy), damped
    // at the waterline/beach and on the cabin pad so placement stays exact.
    float detailRelief(float x, float z, float baseH, float slopeMask, float padX, float padZ) {
        const float dp = std::sqrt((x - padX) * (x - padX) + (z - padZ) * (z - padZ));
        const float padMask = math::smoothstep(22.f, 56.f, dp);
        const float waterMask = math::smoothstep(0.6f, 2.8f, baseH);
        if (padMask * waterMask < 1e-3f) return 0.f;
        const float amp = 0.22f + 1.35f * math::smoothstep(0.10f, 0.45f, slopeMask);
        // fbm2 wavelengths at this frequency: ~20 m / 9.4 m / 4.3 m.
        return (fbm2(x * 0.05f, z * 0.05f) - 0.5f) * 2.f * amp * padMask * waterMask;
    }

    // Per-sample splat colour on the DETAILED surface (h/slope come from the
    // tile baker's finite differences of the provider). Bands: underwater
    // sediment → wet shore rock → gravel beach → heath grass → scree → rock →
    // snow, with multi-scale albedo variation. Noise runs in world metres.
    void fjordAlbedo(float x, float z, float hC, float slope, float* rgb) {
        const Vector3 cSediment(0.13f, 0.13f, 0.11f);
        const Vector3 cWetRock(0.16f, 0.15f, 0.14f);
        const Vector3 cGrass(0.20f, 0.245f, 0.125f);
        const Vector3 cHeath(0.28f, 0.25f, 0.15f);
        const Vector3 cScree(0.42f, 0.40f, 0.37f);
        const Vector3 cRock(0.34f, 0.33f, 0.32f);
        const Vector3 cSnow(0.86f, 0.88f, 0.92f);

        const float wig = fbm2(x * 0.019f, z * 0.019f) - 0.5f;

        Vector3 col;
        if (hC < 0.35f) {
            // Underwater / waterline: sediment darkening with depth, a
            // wet-rock band just above the waterline.
            const float deep = std::clamp(-hC / 18.f, 0.f, 1.f);
            col.copy(cSediment).multiplyScalar(1.f - 0.55f * deep);
            const float wet = math::smoothstep(-0.6f, 0.35f, hC);
            col.lerp(cWetRock, wet);
        } else if (hC < 1.2f && slope < 0.38f) {
            // Gravel beach strip, noise-broken so it doesn't read as a stripe.
            const Vector3 cGravel(0.30f, 0.28f, 0.24f);
            col.copy(cGravel).multiplyScalar(0.5f + 0.5f * math::smoothstep(0.35f, 1.1f, hC));
            const float breakup = math::smoothstep(0.45f, 0.75f, fbm2(x * 0.038f, z * 0.038f));
            col.lerp(cGrass, std::max(breakup, math::smoothstep(0.8f, 1.2f, hC) * 0.6f));
        } else {
            // Altitude bands (with a noisy snowline), then slope overrides.
            const float snowLine = 300.f + 70.f * wig;
            const float heathT = 0.65f * math::smoothstep(110.f, 230.f, hC);// grass → heath (subtle)
            const float fellT = math::smoothstep(200.f, 280.f, hC);         // heath → grey fell
            const float snowT = math::smoothstep(snowLine - 28.f, snowLine + 28.f, hC);
            Vector3 ground = cGrass;
            ground.lerp(cHeath, heathT);
            ground.lerp(Vector3(0.30f, 0.31f, 0.26f), fellT);

            const float screeT = math::smoothstep(0.24f, 0.42f, slope);
            const float rockT = math::smoothstep(0.45f, 0.62f, slope);
            col.copy(ground).lerp(cScree, screeT).lerp(cRock, rockT);

            const float snowShed = 1.f - math::smoothstep(0.50f, 0.68f, slope);
            col.lerp(cSnow, snowT * snowShed);

            // Dry rock band above the waterline on steep banks.
            const float shoreRock = (1.f - math::smoothstep(0.8f, 6.f, hC)) * math::smoothstep(0.18f, 0.35f, slope);
            col.lerp(cWetRock, shoreRock * 0.7f);
        }

        // Multi-scale albedo variation (de-plastic); occlusion in the folds now
        // comes from the real RT AO over the detail relief.
        const float n1 = fbm2(x * 0.027f, z * 0.027f) - 0.5f;
        const float n2 = vnoise(x * 0.137f, z * 0.137f) - 0.5f;
        const float varia = std::clamp(1.f + 0.30f * n1 + 0.16f * n2, 0.68f, 1.25f);
        col.multiplyScalar(varia);

        rgb[0] = col.x;
        rgb[1] = col.y;
        rgb[2] = col.z;
    }

    // STRUCTURE-band coverage mirroring fjordAlbedo's band decisions (same
    // thresholds and noise, so colour and structure can never disagree):
    // 0 grass (grass/heath/fell ground), 1 rock, 2 scree (+gravel beach),
    // 3 snow. Underwater and the wet shore band stay structure-free — the
    // macro colour carries them (wet rock is smooth; sediment is invisible
    // relief anyway).
    void fjordWeights(float x, float z, float hC, float slope, float* w4) {
        w4[0] = w4[1] = w4[2] = w4[3] = 0.f;
        if (hC < 0.35f) {
            // Wet shore / shallow bottom: modest ROCK structure so the band
            // isn't a featureless smooth strip; fades out with depth (deep
            // bottom stays macro-only — invisible relief anyway).
            w4[1] = 0.35f * math::smoothstep(-4.f, 0.35f, hC);
            return;
        }
        if (hC < 1.2f && slope < 0.38f) {
            // Gravel beach: scree-family structure, grass poking through where
            // the albedo's breakup lerps toward grass.
            const float breakup = math::smoothstep(0.45f, 0.75f, fbm2(x * 0.038f, z * 0.038f));
            const float grassT = std::max(breakup, math::smoothstep(0.8f, 1.2f, hC) * 0.6f);
            w4[2] = 0.85f * (1.f - grassT);
            w4[0] = grassT;
            return;
        }
        const float wig = fbm2(x * 0.019f, z * 0.019f) - 0.5f;
        const float screeT = math::smoothstep(0.24f, 0.42f, slope);
        const float rockT = math::smoothstep(0.45f, 0.62f, slope);
        const float snowLine = 300.f + 70.f * wig;
        const float snowT = math::smoothstep(snowLine - 28.f, snowLine + 28.f, hC) *
                            (1.f - math::smoothstep(0.50f, 0.68f, slope));// slope-shed, like the colour
        // STRUCTURE hands over to the snow band only where snow DOMINATES
        // (squared): at partial snowT the colour is still mostly ground, and
        // binning structure proportionally washed the whole snowline
        // transition zone toward the snow band's soft field — the mid-flank
        // "flat texture" report.
        const float snowW = snowT * snowT;
        w4[0] = (1.f - screeT) * (1.f - snowW);
        w4[2] = screeT * (1.f - rockT) * (1.f - snowW);
        w4[1] = rockT * (1.f - snowW);
        w4[3] = snowW;
    }

    // ═════════════════════════ sun / moon / sky ═════════════════════════════

    struct CelestialState {
        Vector3 sunDir;  // unit, toward the sun
        Vector3 moonDir; // unit, toward the moon
        float sunElev;   // sin(elevation) == sunDir.y
        float daylight;  // 0 night → 1 day
        float moonUp;    // 0..1
    };

    Vector3 dirFromAzElev(float azDeg, float elevDeg) {
        const float az = azDeg * kPi / 180.f, el = elevDeg * kPi / 180.f;
        return {std::sin(az) * std::cos(el), std::sin(el), -std::cos(az) * std::cos(el)};
    }

    CelestialState celestialAt(float tHours) {
        CelestialState s;
        // Sun: up ~05:45–18:45, peak 47° — a Nordic summer arc. Azimuth sweeps
        // so the evening sun rakes across (and partly down) the fjord.
        const float sunElevDeg = 47.f * std::sin(kPi * (tHours - 5.75f) / 13.0f);
        const float sunAzDeg = 60.f + (tHours - 5.75f) * 12.5f;// sets ~SW: evening light RAKES the walls
        s.sunDir = dirFromAzElev(sunAzDeg, sunElevDeg);
        s.sunElev = s.sunDir.y;
        s.daylight = math::smoothstep(-0.10f, 0.05f, s.sunElev);

        // Moon: up ~19:30–05:00, peak ~40° around midnight, opposite side.
        float tm = tHours - 19.5f;
        if (tm < -12.f) tm += 24.f;
        if (tm < 0.f && tm > -12.f) tm += 24.f;// wrap early-morning hours
        const float moonElevDeg = 40.f * std::sin(kPi * tm / 9.5f);
        const float moonAzDeg = 290.f - tm * 14.f;
        s.moonDir = dirFromAzElev(moonAzDeg, moonElevDeg);
        s.moonUp = std::clamp(s.moonDir.y / 0.30f, 0.f, 1.f);
        return s;
    }

    // CPU port of the three.js/threepp Preetham "Sky" shader (objects/Sky.cpp),
    // with a night-sky extension (deep-blue gradient, moon disc + halo) blended
    // in as daylight fades. Returns linear radiance for the env map.
    struct SkyModel {
        float turbidity = 4.5f;
        float rayleigh = 3.0f;
        float mieCoefficient = 0.006f;
        float mieG = 0.8f;
        float gain = 1.0f;

        static float sunIntensityEE(float zenithCos) {
            constexpr float cutoff = 1.6110731556870734f;// pi/1.95
            zenithCos = std::clamp(zenithCos, -1.f, 1.f);
            return 1000.f * std::max(0.f, 1.f - std::exp(-((cutoff - std::acos(zenithCos)) / 1.5f)));
        }
        static float hgPhase(float cosTheta, float g) {
            const float g2 = g * g;
            return 0.07957747154594767f * ((1.f - g2) / std::pow(1.f - 2.f * g * cosTheta + g2, 1.5f));
        }

        // Rayleigh/Mie scattering coefficients for the current sun height (the
        // Rayleigh term collapses as the sun sets, via sunfade). ONE source for
        // both the sky radiance and the sun-light transmittance tint below —
        // these were computed twice with independently-truncated constants
        // (5.8e-6 vs 5.804...e-6) and a hard-coded 2.2 in place of
        // rayleigh - (1 - sunfade), so the sun's colour came from a slightly
        // different atmosphere than the sky it hung in.
        void scatterCoeffs(const CelestialState& cs, Vector3& betaR, Vector3& betaM) const {
            const Vector3 totalRayleigh(5.804542996261093e-6f, 1.3562911419845635e-5f, 3.0265902468824876e-5f);
            const Vector3 mieConst(1.8399918514433978e14f, 2.7798023919660528e14f, 4.0790479543861094e14f);
            const float sunfade = 1.f - std::clamp(1.f - std::exp(cs.sunDir.y), 0.f, 1.f);
            const float rayleighCoeff = rayleigh - (1.f - sunfade);
            betaR.set(totalRayleigh.x * rayleighCoeff, totalRayleigh.y * rayleighCoeff,
                      totalRayleigh.z * rayleighCoeff);
            const float mieC = 0.434f * (0.2f * turbidity) * 1e-17f * mieCoefficient;
            betaM.set(mieConst.x * mieC, mieConst.y * mieC, mieConst.z * mieC);
        }

        // Beer-Lambert extinction along the view path toward elevation dirY:
        // the Preetham optical depth for a slant path, applied to the combined
        // scattering coefficients.
        static Vector3 extinction(const Vector3& betaR, const Vector3& betaM, float dirY) {
            const float zenith = std::acos(std::max(0.f, dirY));
            const float inv = 1.f / (std::cos(zenith) +
                                     0.15f * std::pow(93.885f - zenith * 180.f / kPi, -1.253f));
            const float sR = 8400.f * inv, sM = 1250.f * inv;
            return {std::exp(-(betaR.x * sR + betaM.x * sM)),
                    std::exp(-(betaR.y * sR + betaM.y * sM)),
                    std::exp(-(betaR.z * sR + betaM.z * sM))};
        }

        [[nodiscard]] Vector3 radiance(const Vector3& dir, const CelestialState& cs) const {
            const float sunE = sunIntensityEE(cs.sunDir.y);
            const float sunfade = 1.f - std::clamp(1.f - std::exp(cs.sunDir.y), 0.f, 1.f);
            Vector3 betaR, betaM;
            scatterCoeffs(cs, betaR, betaM);

            const Vector3 fex = extinction(betaR, betaM, dir.y);

            const float cosTheta = std::clamp(dir.dot(cs.sunDir), -1.f, 1.f);
            const float ct = cosTheta * 0.5f + 0.5f;
            const float rPhase = 0.05968310365946075f * (1.f + ct * ct);
            const float mPhase = hgPhase(cosTheta, mieG);

            Vector3 lin;
            {
                const float mixT = std::clamp(std::pow(1.f - cs.sunDir.y, 5.f), 0.f, 1.f);
                auto comp = [&](float bR, float bM, float fx) {
                    const float ratio = (bR * rPhase + bM * mPhase) / (bR + bM);
                    float v = std::pow(std::max(sunE * ratio * (1.f - fx), 0.f), 1.5f);
                    const float vFex = std::pow(std::max(sunE * ratio * fx, 0.f), 0.5f);
                    return v * (1.f + (vFex - 1.f) * mixT);
                };
                lin.set(comp(betaR.x, betaM.x, fex.x), comp(betaR.y, betaM.y, fex.y),
                        comp(betaR.z, betaM.z, fex.z));
            }

            // Base sky glow + solar disc (66 arcsec, radiance clamped for env NEE sanity).
            Vector3 l0(0.1f * fex.x, 0.1f * fex.y, 0.1f * fex.z);
            constexpr float sunDiscCos = 0.9999566769464484f;
            const float sundisk = math::smoothstep(sunDiscCos, sunDiscCos + 0.00002f, cosTheta);
            if (sundisk > 0.f) {
                l0.x += sunE * 19000.f * fex.x * sundisk;
                l0.y += sunE * 19000.f * fex.y * sundisk;
                l0.z += sunE * 19000.f * fex.z * sundisk;
            }

            Vector3 tex((lin.x + l0.x) * 0.04f, (lin.y + l0.y) * 0.04f + 0.0003f,
                        (lin.z + l0.z) * 0.04f + 0.00075f);
            const float invPow = 1.f / (1.2f + 1.2f * sunfade);
            Vector3 col(std::pow(std::max(tex.x, 0.f), invPow), std::pow(std::max(tex.y, 0.f), invPow),
                        std::pow(std::max(tex.z, 0.f), invPow));

            // The Preetham pow-curve lifts the deep-night residual to a grey
            // haze; fade the daylight model out once the sun is well below the
            // horizon so night is owned by the (dark) night model.
            const float preFade = 0.05f + 0.95f * math::smoothstep(-0.25f, -0.02f, cs.sunDir.y);
            col.multiplyScalar(preFade);

            // ── night extension ──
            const float night = 1.f - cs.daylight;
            if (night > 0.f) {
                const float elev = std::asin(std::clamp(dir.y, -1.f, 1.f));
                float r = 0.f, g = 0.f, b = 0.f;
                if (elev >= 0.f) {
                    const float horizon = std::exp(-elev * 4.5f);
                    r = 0.0035f + 0.016f * horizon;
                    g = 0.005f + 0.022f * horizon;
                    b = 0.010f + 0.042f * horizon;
                }
                // Moon disc (~1°) + halo.
                const float cosToMoon = std::clamp(dir.dot(cs.moonDir), -1.f, 1.f);
                const float angTo = std::acos(cosToMoon);
                if (angTo < 0.018f) {
                    r += 16.f * cs.moonUp;
                    g += 18.f * cs.moonUp;
                    b += 23.f * cs.moonUp;
                } else {
                    const float glow = 0.28f * std::exp(-angTo * angTo * 110.f) * cs.moonUp;
                    r += glow * 0.65f;
                    g += glow * 0.75f;
                    b += glow;
                }
                col.x += r * night;
                col.y += g * night;
                col.z += b * night;
            }

            // Below the horizon: fade to a dark sea/ground tone so reflections
            // and the probe grid aren't fed void.
            if (dir.y < 0.f) {
                const float fade = std::exp(dir.y * 3.5f);
                col.multiplyScalar(fade);
                col.x += 0.002f;
                col.y += 0.0025f;
                col.z += 0.003f;
            }

            // Global dim: the analytic DirectionalLight carries the direct sun;
            // pulling the sky down raises the sun:sky contrast (crisper shadows,
            // golden punch) — auto-exposure re-normalises the absolute level.
            col.multiplyScalar(gain * 0.62f);
            col.x = std::min(col.x, 300.f);
            col.y = std::min(col.y, 300.f);
            col.z = std::min(col.z, 300.f);
            return col;
        }
    };

    // Bake the sky into an equirect float image. Mapping matches the renderer's
    // sampleEnvLod: v=1 is the zenith (dir.y=+1), u wraps azimuth.
    void bakeSkyInto(std::vector<float>& data, int W, int H, const SkyModel& sky, const CelestialState& cs) {
        std::vector<int> rows(static_cast<size_t>(H));
        std::iota(rows.begin(), rows.end(), 0);
        parallelForEach(rows.begin(), rows.end(), [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
            const float elev = (v - 0.5f) * kPi;
            const float ce = std::cos(elev), se = std::sin(elev);
            for (int x = 0; x < W; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
                const float az = (u - 0.5f) * kTau;
                const Vector3 dir(ce * std::cos(az), se, ce * std::sin(az));
                const Vector3 c = sky.radiance(dir, cs);
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                data[i + 0] = c.x;
                data[i + 1] = c.y;
                data[i + 2] = c.z;
                data[i + 3] = 1.f;
            }
        });
    }

    // Average horizon radiance → fog / haze colour that always matches the sky.
    Color horizonColor(const SkyModel& sky, const CelestialState& cs) {
        Vector3 acc;
        constexpr int N = 12;
        for (int i = 0; i < N; ++i) {
            const float az = (static_cast<float>(i) + 0.5f) / N * kTau;
            const Vector3 dir(std::cos(0.026f) * std::cos(az), std::sin(0.026f), std::cos(0.026f) * std::sin(az));
            acc.add(sky.radiance(dir, cs));
        }
        acc.multiplyScalar(1.f / N);
        return {acc.x, acc.y, acc.z};
    }

    // ═════════════════════════ vegetation prototypes ════════════════════════

    struct TreeVariant {
        std::shared_ptr<BufferGeometry> trunkGeo;
        std::shared_ptr<BufferGeometry> leafGeo;
        std::shared_ptr<MeshStandardMaterial> barkMat;
        std::shared_ptr<MeshStandardMaterial> leafMat;
    };

    TreeVariant makeTreeVariant(int preset, unsigned int seed, bool cheapBlob) {
        vegetation::TreeParams tp;
        vegetation::applyPreset(preset, tp);
        tp.seed = seed;
        if (preset == 1 && !cheapBlob) {// Norway spruce: whorled monopodial conifer
            // preset 1 already selects BranchingMode::Whorl + LeafStyle::Frond;
            // tune the silhouette for the fjord banks (tall, narrow, near-ground
            // skirt of drooping frond branches).
            // Slim serrated silhouette: height ≈ 12.8 vs width ≈ 3.8 (≥3:1),
            // whorl shelves separated enough to read as distinct layers with
            // sky gaps between them, and a pointed apex (profile → 0 at the
            // top whorl + bare leader above it).
            tp.trunkHeight = 1.8f;
            // Scaled with the height: this spruce is 12.8 m against the preset's
            // 9.6, and carrying the preset radius up made a 20 m tree stand on a
            // pole. (The scatter then instances these at 1.3-2.1×.)
            tp.trunkRadius = 0.22f;
            tp.crownRadiusX = tp.crownRadiusZ = 1.9f;
            tp.crownHeight = 11.f;
            // Close enough that successive shelves overlap over the bole. At
            // 0.85 the layers read as separate pancakes on a pole, and the
            // thicker trunk this preset now carries made that reading worse.
            tp.whorlSpacing = 0.72f;
            tp.branchesPerWhorl = 5;
            tp.branchDroop = 0.44f;
            tp.branchTipUpturn = 0.42f;
            tp.crownProfileExponent = 1.25f;
            tp.sideTwigDensity = 0.6f;
            tp.leafSize = 0.75f;
            tp.leafDensity = 0.92f;
            tp.leafClumping = 0.0f;
            tp.leafColor = {0.13f, 0.34f, 0.10f};
        }
        if (preset == 2) {// birch — mute the pure-white preset bark
            tp.barkColor = {0.72f, 0.71f, 0.67f};
            tp.leafDensity = 0.95f;
            tp.leafClumping = 0.35f;
        }
        if (cheapBlob) {// far-bank silhouette trees: low-poly canopy puffs
            // Keep the CHEAP space-colonisation path (whorl+frond would balloon
            // node/card counts on the far bank); a simple cone of blobs is all a
            // distant silhouette needs.
            tp.branchingMode = vegetation::BranchingMode::Colonise;
            tp.crownShape = vegetation::CrownShape::Cone;
            tp.trunkHeight = 3.5f;
            tp.crownRadiusX = tp.crownRadiusZ = 2.0f;
            tp.crownHeight = 7.0f;
            tp.influenceDistance = 3.5f;
            tp.killDistance = 0.7f;
            tp.segmentLength = 0.45f;
            tp.maxIterations = 200;
            tp.tropism = -0.04f;
            tp.leafStyle = vegetation::LeafStyle::Blob;
            // Several SMALL puffs per node rather than two big ones: two
            // 1.15m spheres per node merge into one smooth dome, so the tree
            // reads as a broccoli floret standing next to the near band's thin
            // dark conifer silhouettes. Smaller puffs let the cone profile show
            // through and give the outline a ragged edge.
            tp.leavesPerCluster = 3;
            tp.leafSize = 0.80f;
            tp.attractorCount = 320;
            tp.radialSegments = 5;
            // Match the near spruce foliage colour ({0.13,0.34,0.10}) — the far
            // bank reading a different green from the near band is most of what
            // made the blob trees look pasted on.
            tp.leafColor = {0.13f, 0.34f, 0.10f};
        }

        vegetation::TreeGenerator gen(seed);
        gen.buildSkeleton(tp);

        TreeVariant v;
        v.trunkGeo = gen.makeTrunkGeometry(tp);
        v.leafGeo = gen.makeLeafGeometry(tp);

        auto bark = vegetation::makeBarkTextures(cheapBlob ? 128 : 256, seed, tp.barkColor, tp.barkStyle);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);
        v.barkMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        v.barkMat->map = bark.first;
        v.barkMat->normalMap = bark.second;
        v.barkMat->vertexColors = true;// twig darkening, baked per-vertex

        v.leafMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        // Foliage translucency: backlit canopies glow through instead of going
        // flat-dark (Vulkan deferred). Tint slightly yellow-green — the colour of
        // light that survives a leaf/needle. Cheap far blobs get a touch less.
        v.leafMat->translucencyColor = Color(0.50f, 0.80f, 0.28f);
        v.leafMat->translucency = cheapBlob ? 0.35f : 0.5f;
        if (cheapBlob) {
            // The baked per-vertex canopy tint now spans 0.14–1.0 and only ever
            // DARKENS, so the old 1.15 pre-brighten (which compensated a tint
            // that could exceed 1) now just makes the far bank glow.
            // leafColor is an sRGB hint (TreeParams doc) — the card path bakes it into an
            // sRGB-tagged texture (decoded on sample), but material->color is LINEAR
            // working space. Without the conversion the blobs render ~4x too bright and
            // read "always lit" (glowing green even at night).
            v.leafMat->color = Color(tp.leafColor[0], tp.leafColor[1], tp.leafColor[2])
                                       .convertSRGBToLinear();
            v.leafMat->vertexColors = true;// canopy tint gradient baked per-vertex
        } else {
            // Conifer fronds use the elongated needle cutout; broadleaf the round
            // leaf-cluster atlas.
            // The atlas grid must be the one the cards were UV'd for (see
            // TreeParams::leafAtlasCells) — the cards pick a cell each, so a
            // mismatch has every card sampling a fraction of the wrong variant.
            v.leafMat->map = (tp.leafStyle == vegetation::LeafStyle::Frond)
                    ? vegetation::makeNeedleFrondTexture(256, seed, tp.leafColor, tp.leafAtlasCells)
                    : vegetation::makeLeafClusterTexture(256, seed, tp.leafColor, tp.leafShape,
                                                         8, tp.leafAtlasCells);
            // Below the antialiased margin of the thin leaflets/needles the
            // atlases are drawn from — 0.5 eats whole leaves off mipped cards.
            v.leafMat->alphaTest = vegetation::kLeafAlphaTest;
            v.leafMat->side = Side::Double;
            v.leafMat->vertexColors = true;
        }
        return v;
    }

    // Grass blades are merged into GrassMesh tiles by the reusable helper
    // threepp::vegetation::buildGrassTiles (extras/vegetation/GrassTiles.hpp) —
    // see the "valley meadow grass" block in main().


    // ═══════════════════════════ cabin / dock / boat ════════════════════════

    std::shared_ptr<BufferGeometry> makeBoatHull() {
        constexpr int N = 9;   // stations along the length
        constexpr float L = 4.2f;
        const int S = 5;// cross-section points
        std::vector<float> pos;
        std::vector<unsigned int> idx;

        auto section = [&](float t, int s) -> Vector3 {
            const float endT = 2.f * std::abs(t - 0.5f);// 0 midship → 1 ends
            const float w = std::max(0.62f * std::pow(std::max(std::sin(kPi * t), 0.f), 0.65f), 0.02f);
            const float rail = 0.46f + 0.15f * std::pow(endT, 1.7f);
            const float keel = 0.05f + 0.28f * endT * endT;
            const float sx[5] = {-1.f, -0.72f, 0.f, 0.72f, 1.f};
            const float sy[5] = {1.f, 0.35f, 0.f, 0.35f, 1.f};
            return {w * sx[s], keel + (rail - keel) * sy[s], (t - 0.5f) * L};
        };

        for (int i = 0; i <= N; ++i) {
            const float t = static_cast<float>(i) / N;
            for (int s = 0; s < S; ++s) {
                const Vector3 p = section(t, s);
                pos.insert(pos.end(), {p.x, p.y, p.z});
            }
        }
        for (int i = 0; i < N; ++i)
            for (int s = 0; s < S - 1; ++s) {
                const auto a = static_cast<unsigned int>(i * S + s);
                const auto b = a + 1;
                const auto c = a + S;
                const auto d = c + 1;
                idx.insert(idx.end(), {a, c, b, b, c, d});
            }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->computeVertexNormals();
        return geo;
    }

    struct Cabin {
        std::shared_ptr<Group> group;
        std::shared_ptr<MeshStandardMaterial> windowMat;// shared warm panes (night glow)
        std::shared_ptr<MeshStandardMaterial> sconceMat;// porch lanterns, same schedule
        Vector3 chimneyTipLocal;
    };

    // The hytte on the east bench, from the procedural log-cabin generator
    // (extras/architecture). Shorter and shallower than the generator's own
    // defaults so it sits inside the flattened pad without crowding the dock
    // path, and stained dark the way decades of weather off the fjord leave it.
    architecture::CabinParams fjordCabinParams() {
        architecture::CabinParams cp;
        cp.seed = 11u;
        // Against a fjord wall and 20 m spruce, a 12 m cabin read as a model.
        // Still inside the pad's 24 m flat radius: half-diagonal of the 16 x
        // 12 m envelope (body + porch) is ~10 m.
        cp.length = 16.0f;
        cp.depth = 9.0f;
        cp.wallHeight = 3.6f;
        cp.floorHeight = 0.80f;
        cp.roofPitchDeg = 40.f;
        cp.porchStartX = -8.0f;
        cp.porchEndX = 8.0f;
        cp.porchDepth = 2.8f;
        cp.stepsCenterX = 0.f;// defaultOpenings puts the door on centre
        cp.stepsWidth = 1.8f;
        cp.dormerCenterX = -4.4f;
        cp.dormerWidth = 4.8f;
        cp.flueX = 4.8f;
        cp.flueZ = -1.1f;
        cp.flueRise = 1.6f;
        cp.logColor = {0.355f, 0.205f, 0.098f};   // dark creosote stain
        cp.logEndColor = {0.440f, 0.320f, 0.180f};
        cp.shingleColor = {0.130f, 0.135f, 0.145f};// slate roof, as the old cabin had
        cp.timberColor = {0.365f, 0.248f, 0.142f};
        return cp;
    }

    // Built once and shared by every instance: the generator bakes five
    // procedural textures, and the far cabin has no reason to bake its own.
    architecture::CabinMaterials& fjordCabinMaterials() {
        static architecture::CabinMaterials mats = [] {
            auto m = architecture::makeCabinMaterials(fjordCabinParams());
            // The panes double as the night light — the frame loop drives
            // emissiveIntensity from the sun elevation.
            m.glass->emissive = Color(1.0f, 0.62f, 0.28f);
            m.glass->emissiveIntensity = 0.f;
            m.lamp->emissiveIntensity = 0.f;// porch sconces: dark by day
            return m;
        }();
        return mats;
    }

    Cabin makeCabin() {
        Cabin cabin;
        const auto cp = fjordCabinParams();
        auto& mats = fjordCabinMaterials();
        cabin.group = architecture::createLogCabin(cp, mats);
        cabin.windowMat = mats.glass;
        cabin.sconceMat = mats.lamp;
        // Reported in cabin-LOCAL space; the smoke emitter transforms it by the
        // group's world matrix (the cabin is rotated to face the water).
        cabin.chimneyTipLocal = architecture::cabinMetrics(cp).flueTip;
        return cabin;
    }

}// namespace

int main(int argc, char** argv) {

    // ── args ────────────────────────────────────────────────────────────────
    std::string shotPath;
    int shotFrames = 220, shotFrame = 0, shotCam = 0;
    float startTime = 17.6f;// golden hour (sunset ~18:45)
    float startCycle = 0.f; // hours per second (0 = paused)
    bool startFly = false;// begin on the cinematic flight path
    // Grass A/B toggle for perf measurement: --single-grass builds the whole
    // valley meadow as ONE merged GrassMesh (its single valley-spanning AABB is
    // ~always on screen and near, so it is never frustum/occlusion-culled and its
    // wind never freezes — the "no culling" baseline). The default builds the
    // same blades as a grid of GrassMesh tiles (cullable + distance-frozen).
    bool singleGrass = false;
    int optProbe = -1;// --probe 0|1: probe-GI override (indirect-lighting A/B harness)
    // --mist [density]: shape the unified AIR medium as a tall ground-mist
    // PROFILE (setHeightFog). With the "haze" slider at 0 (no scene.fog) this
    // density CREATES the medium directly (back-compat); the froxels own the near
    // field [0, 512 m] and the per-pixel march the far tail — one continuous fog.
    bool mistOn = false;
    float mistDensity = 0.0012f;
    float mistFalloff = 300.f;  // --mistfall F: --mist profile falloff (m). Small (e.g.
                                // 40-60) = a shallow ground layer the aerial cam sees
                                // from ABOVE (fog-scout: camera-above-the-layer repro).
    bool  noClouds = false;     // --noclouds: force the cloud deck off (scout: isolate
                                // real clouds from fog-path artefacts under fog).
    float climbRate = 0.f;      // --climb R: in shot mode, raise the camera Y by R m per
                                // frame across the capture — reproduces "flying up/out of
                                // the mist layer" in motion (regression: black on exit).
    float startFogScale = 0.f;  // --fogscale S: AIR-fog (scene.fog) density scale.
                                // 0 (default) = clear air + underwater murk only;
                                // raise it (or the "haze" slider) to fill the fjord
                                // with haze + god rays (Phase 2: a plain density
                                // scale — no clip-lifting hack any more).
    bool profNoAo = false, profNoGi = false, profHardShadow = false;// perf A/B: isolate the RT cost in the deferred shade
    bool noAutoExposure = false;// --noae: fixed exposure — the AE histogram meters the
                                // fog's in-scatter and normalises it away, hiding
                                // exactly the shafts the scout is trying to compare
    float startWind = 4.5f;     // --wind W: initial wind (m/s); the UI slider still works
    float foamAmount = -1.f;    // --foamamount F: override natural whitecap foam (0..1; <0 = keep default)
    int   waterResX = 0, waterResZ = 0;// --waterres X Z: override the water grid (scout: geometric-aliasing tests)
    int   winW = 0, winH = 0;   // --size W H: window/render size (default 960x600)
    int   debugView = 0;        // --debugview N: blit a G-buffer channel (1=normal 2=motion 3=id 4=albedo)
    int   optLod = -1;          // --lod 0|1: auto mesh LOD (the vegetation A/B — the shot
                                // line already prints lod=[...]; this is the OFF leg, so a
                                // foliage artefact can be pinned on the simplifier or cleared)
    float tile2Override = -1.f; // --tile2 F: cascade-2 tile (0 disables the fine cascade)
    // --cam x,y,z / --look x,y,z: free camera override for shot mode —
    // (--cam is the capture_util convention every Vulkan demo shares; the
    // camera PRESET index moved to --view. --campos stays as a legacy alias.) —
    // scouting artefacts at arbitrary viewpoints without adding a preset each time.
    bool  hasCamPos = false, hasCamLook = false;
    float camPos[3]{}, camLook[3]{};
    // --seq DIR: the motion harness. A held pose cannot see a view-anchored
    // defect — that is the whole lesson of the F1 smoke regression, where every
    // still passed and the plume boiled in froxel-sized blocks the moment the
    // camera moved. The path is closed-form in the FRAME INDEX, so two runs of
    // the same command produce the same poses.
    std::string seqDir;
    int   seqFrames = 6, seqWarm = 160;
    float seqOrbit  = 10.f;// degrees per second about the look target
    // --dolly M: metres per second STRAIGHT AHEAD along the ground, eye and
    // target together, instead of orbiting — a walk goes somewhere an orbit
    // cannot, which is what a distance-dependent artefact needs. Non-zero takes
    // precedence over --orbit.
    float seqDolly  = 0.f;
    // Day/night: --day / --night set the demo's own hour (the D key jumps
    // between the same two), rather than adding a second switch beside it.
    constexpr float kDayHour = 13.2f, kNightHour = 22.6f;
    bool  dayFlag = false, nightFlag = false;
    // --legacy-smoke: the pre-2026-08-11 chimney — ~156 alpha-blended sprites on
    // the legacy ParticleSystem path, CPU-integrated with their own RNG. The A/B
    // leg for the density plume that replaced it, in ONE binary (the precedent
    // is FireEffect's --legacy-embers). It brings back the old properties with
    // it: a plume that is drawn OVER the fog rather than composited into it, no
    // sunlight scattering through it, and a scene that is not reproducible
    // run-to-run — the sprite RNG was one of the reasons two fjord captures of
    // the same pose never matched.
    bool  legacySmoke = false;
    auto parseVec3 = [](const char* s, float out[3]) {
        const auto v = capture::parseVec3(s);// shared: accepts "x,y,z" and "x y z"
        if (v) { out[0] = v->x; out[1] = v->y; out[2] = v->z; }
        return v.has_value();
    };
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--probe") == 0 && i + 1 < argc) optProbe = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--time") == 0 && i + 1 < argc) startTime = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--view") == 0 && i + 1 < argc) shotCam = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cycle") == 0 && i + 1 < argc) startCycle = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--fly") == 0) startFly = true;
        else if (std::strcmp(argv[i], "--single-grass") == 0) singleGrass = true;
        else if (std::strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seqDir = argv[++i];
        else if (std::strcmp(argv[i], "--seqframes") == 0 && i + 1 < argc) seqFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warm") == 0 && i + 1 < argc) seqWarm = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--orbit") == 0 && i + 1 < argc) seqOrbit = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--dolly") == 0 && i + 1 < argc) seqDolly = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--day") == 0) dayFlag = true;
        else if (std::strcmp(argv[i], "--night") == 0) nightFlag = true;
        else if (std::strcmp(argv[i], "--legacy-smoke") == 0) legacySmoke = true;
        else if (std::strcmp(argv[i], "--mist") == 0) {
            mistOn = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') mistDensity = static_cast<float>(std::atof(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--mistfall") == 0 && i + 1 < argc) mistFalloff = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--noclouds") == 0) noClouds = true;
        else if (std::strcmp(argv[i], "--climb") == 0 && i + 1 < argc) climbRate = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--fogscale") == 0 && i + 1 < argc) startFogScale = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--noae") == 0) noAutoExposure = true;
        else if (std::strcmp(argv[i], "--noao") == 0) profNoAo = true;
        else if (std::strcmp(argv[i], "--nogi") == 0) profNoGi = true;
        else if (std::strcmp(argv[i], "--hardshadow") == 0) profHardShadow = true;
        else if (std::strcmp(argv[i], "--wind") == 0 && i + 1 < argc) startWind = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--foamamount") == 0 && i + 1 < argc) foamAmount = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--waterres") == 0 && i + 2 < argc) {
            waterResX = std::atoi(argv[++i]);
            waterResZ = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--size") == 0 && i + 2 < argc) {
            winW = std::atoi(argv[++i]);
            winH = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--debugview") == 0 && i + 1 < argc) debugView = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--lod") == 0 && i + 1 < argc) optLod = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tile2") == 0 && i + 1 < argc) tile2Override = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--cam") == 0 && i + 1 < argc) hasCamPos = parseVec3(argv[++i], camPos);
        else if (std::strcmp(argv[i], "--campos") == 0 && i + 1 < argc) hasCamPos = parseVec3(argv[++i], camPos);// legacy alias
        else if (std::strcmp(argv[i], "--look") == 0 && i + 1 < argc) hasCamLook = parseVec3(argv[++i], camLook);
    }
    // After the loop so an explicit --time still wins when both are given.
    if (dayFlag) startTime = kDayHour;
    if (nightFlag) startTime = kNightHour;

    // Window size matters for artefact scouting, not just framing: the water's
    // spec-AA fades the fine chop out of N by the PIXEL FOOTPRINT, so a
    // 960×600 capture banks ~4× more slope variance (blurrier, more forgiving)
    // than a 1920×1129 session at the same view. Repro at the reported size.
    Canvas canvas("threepp - FJORD (Vulkan deferred)",
                  {{"vsync", false},
                   {"size", WindowSize{winW > 0 ? winW : 960, winH > 0 ? winH : 600}}});
    VulkanRenderer renderer(canvas);
    if (optProbe >= 0) renderer.setProbeGI(optProbe != 0);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.setAutoExposure(!noAutoExposure);
    if (debugView > 0) renderer.setHybridDebugView(debugView);
    if (optLod >= 0) renderer.setAutoLod(optLod != 0);
    // Cap the upper end well below the default +3 EV: the eye should NOT
    // re-expose midnight back to daylight — night must stay dark.
    renderer.setAutoExposureRange(-2.5f, 1.5f);
    // --seq behaves like --shot for everything except what it writes: no UI,
    // no orbit controls, a scripted camera and a fast exposure settle.
    const bool headlessish = !shotPath.empty() || !seqDir.empty();
    renderer.setAutoExposureSpeed(headlessish ? 12.0f : 2.0f);
    // renderer.setGbufferMsaa(2);// leaf canopies + grass edges
    renderer.setRenderScale(0.85f);
    renderer.setSunAngularRadius(profHardShadow ? 0.f : 0.6f);// soft RT sun shadows
    if (profNoAo) renderer.setDeferredAO(false);
    if (profNoGi) renderer.setProbeGI(false);
    // Phase 2 unified fog: scene.fog (set per-frame below) is the AIR medium —
    // the haze + god rays through the fjord walls, driven live by the "haze"
    // slider. The volumetrics follow automatically (no setVolumetricFog opt-in).
    if (mistOn) {// --mist: shape the air medium as a tall ground-mist PROFILE
        VulkanRenderer::HeightFogSettings hf;
        hf.density = mistDensity;// explicit density > 0 → OVERRIDES scene.fog's
                                 // density (the advanced override); the mist holds
                                 // its shape even with the haze slider up
        hf.baseY = 0.f;
        hf.falloff = mistFalloff;// tall (default 300) ≈ homogeneous over the shaft zone;
                                 // small (--mistfall) = a shallow ground layer
        hf.noiseAmount = 0.f; // smooth analytic
        renderer.setHeightFog(hf);
    }
    renderer.setFogAnisotropy(0.55f);
    // Underwater MURK: the sky-tinted absorption below the waterline, a SEPARATE
    // medium from the air fog now (Phase 2 decoupling). The water surface at y=0
    // clips it to the water column; the air fog (scene.fog) is NOT clipped by it.
    // Density/colour are refreshed per-frame (dawn-peaked, sky-matched) below.
    renderer.setFogWaterSurfaceY(0.f);
    renderer.setBloomIntensity(0.05f);

    Scene scene;

    // ── terrain ─────────────────────────────────────────────────────────────
    const auto t0 = std::chrono::high_resolution_clock::now();
    terrain::TerrainParams tp;
    tp.seed = 20260706u;
    tp.worldSize = kWorldSize;
    tp.resolution = kRes;
    tp.noiseType = terrain::NoiseType::Ridged;
    tp.featureScale = 760.f;
    tp.octaves = 8;
    tp.amplitude = kAmplitude;
    tp.warp = 0.5f;
    tp.ridgeSharpness = 0.72f;
    tp.heightExponent = 1.22f;
    tp.erosion = terrain::ErosionType::Both;
    tp.droplets = 110000;
    tp.thermalIterations = 26;

    terrain::TerrainGenerator terrainGen(tp.seed);
    terrainGen.buildField(tp);
    terrainGen.erode(tp);

    const float padX = channelCenterX(kPadZ) + kPadDist;
    const terrain::HeightGrid field = carveFjord(terrainGen, tp, padX, kPadZ);

    // Quadtree LOD tiles over the carved grid: bicubic base + slope-modulated
    // sub-grid relief near the camera, per-tile splat albedo. The provider is
    // also the single source of truth for every placement query below.
    terrain::TerrainProvider prov;
    prov.height = [&field, padX](float x, float z) {
        const float base = field.sampleBicubic(x, z);
        const float slopeMask = 1.f - field.slopeNy(x, z);
        return base + detailRelief(x, z, base, slopeMask, padX, kPadZ);
    };
    prov.albedo = &fjordAlbedo;
    prov.weights = &fjordWeights;

    // Shallow tree, fat tiles: a 97²-vert tile at depth 3 already gives ~1 m
    // near vertex spacing, at a third of the live-mesh count of a depth-4 tree.
    terrain::TileTerrainOptions tileOpts;
    tileOpts.worldSize = kWorldSize;
    tileOpts.rootGrid = 4;
    tileOpts.maxDepth = 3;// 93.75 m leaves → ~0.98 m verts, ~0.49 m splat texels
    tileOpts.tileRes = 96;
    tileOpts.splitFactor = 1.15f;
    tileOpts.mergeFactor = 1.6f;
    tileOpts.splatTexelsPerQuad = 2;
    // Tiled cm-scale ground detail (MaterialWithDetailMap): the splat bakes at
    // ~0.49 m/texel — mush underfoot. A 256² TILEABLE linear noise field
    // (0.5-neutral) breaks that up at ~1.25 m repeat, and a matching detail
    // NORMAL + ROUGHNESS map (derived from the same heightfield) gives near
    // ground relief lighting + roughness breakup. The shader distance-fades
    // every term so the far fjord walls stay pattern-free and shimmer-free.
    {
        // Per-band STRUCTURE sets (grass/rock/scree/snow), selected by the
        // baked fjordWeights map: material structure resolves at screen
        // density (stochastic-tiled, triplanar, height-blended) over the
        // fjordAlbedo macro colour. FJ_NO_BANDS=1 falls back to the single
        // detail layer below for A/B.
        const bool noBands = [] {
            const char* v = std::getenv("FJ_NO_BANDS");
            return v && v[0] == '1';
        }();
        if (!noBands) {
            const terrain::TerrainBandSet bands = terrain::makeTerrainBandSet();
            for (size_t i = 0; i < 4; ++i) {
                tileOpts.bandAlbedo[i] = bands.band[i].albedo;
                tileOpts.bandNormalRough[i] = bands.band[i].normalRough;
            }
            tileOpts.bandRepeat = bands.repeat;
            tileOpts.bandRoughness = bands.roughness;
            tileOpts.bandStrength = 0.85f;
            tileOpts.bandNormalScale = 1.4f;
            tileOpts.bandRoughStrength = 0.5f;
        }
        // Legacy cm-scale detail layer — the band-less fallback (env override
        // above; also what the tiles use if bands are ever cleared).
        terrain::DetailMapOptions dopt;
        dopt.dim = 256;
        dopt.seed = 4242u;
        dopt.chroma = 0.06f;        // subtle — the field also lands on snow/rock
        dopt.albedoContrast = 0.42f;
        dopt.normalStrength = 2.0f;
        dopt.roughContrast = 0.5f;
        const terrain::DetailMaps dm = terrain::makeDetailMaps(dopt);
        tileOpts.detailMap = dm.albedo;
        tileOpts.detailNormalMap = dm.normalRough;
        tileOpts.detailRepeat = 0.8f;// one repeat per 1.25 m
        tileOpts.detailStrength = 0.85f;
        tileOpts.detailNormalScale = 0.9f;
        tileOpts.detailRoughStrength = 0.5f;
    }
    auto tiles = terrain::TileTerrain::create(prov, tileOpts);
    tiles->name = "fjord_terrain";
    scene.add(tiles);
    const auto terrainH = [&tiles](float x, float z) { return tiles->heightAt(x, z); };
    {
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "[fjord] terrain (erode+carve+" << tiles->activeTiles() << " root tiles): "
                  << std::chrono::duration<float>(t1 - t0).count() << " s" << std::endl;
    }

    // ── ONE WIND, ONE WORLD ─────────────────────────────────────────────────
    // Every moving air-driven thing in this scene reads the TWO numbers below,
    // and that is a defect fix, not tidiness. Before this the fjord had two
    // winds that had never been introduced: the ocean spectrum and the cloud
    // deck drifted on heading 2.1 rad, while the meadow and the chimney plume
    // each carried a privately authored vector around heading 0.5 rad — 90
    // degrees apart. The waves came from one quarter and the smoke leaned
    // toward another, which is exactly the kind of disagreement that makes a
    // picture read as a set of separate effects sharing a frame instead of one
    // place.
    //
    // WHY THE ATMOSPHERE MOVED TO THE OCEAN'S HEADING and not the reverse: a
    // fjord's swell is the most legible wind indicator in the frame (it covers
    // a third of it and its direction is readable at any distance), and the
    // whole cascade is calibrated against that heading. Rotating the plumes
    // costs nothing but the direction they lean.
    constexpr float kWindHeading = 2.1f;// radians in XZ — the ONE heading
    const Vector3   windDir(std::cos(kWindHeading), 0.f, std::sin(kWindHeading));
    // Surface wind is a FRACTION of the free stream the waves and the cloud
    // deck see: a metre off the ground, inside a valley, behind a treeline, the
    // air is doing a fraction of what it does at 400 m. 0.14 is not a
    // measurement, it is the number that reproduces the plume speeds this scene
    // was already authored with (0.63 m/s at the default 4.5 m/s wind) so the
    // ONLY thing this unification changes is the DIRECTION they agree on.
    constexpr float kGroundWindFactor = 0.14f;
    const auto groundWindAt = [&](float speed) {
        Vector3 w(windDir);
        return w.multiplyScalar(speed * kGroundWindFactor);
    };
    const Vector3 groundWind0 = groundWindAt(startWind);

    // ── water ───────────────────────────────────────────────────────────────
    // A STATIC rectangle over the channel, not a camera-following 3200² sheet:
    // the S-curve centreline wanders ±~230 m and the waterline reaches ~166 m
    // beyond it, so 900 m of X covers every bank with margin while the old
    // square spent ~70% of its 262k vertices under the mountains. 192 × 512
    // verts = 2.7× fewer, with FINER across-channel spacing (4.7 m vs 6.3 m)
    // where the camera actually looks along the water. The wave field is
    // unaffected — FFT tiles key on the larger extent (3200 m, as before) and
    // are world-anchored, so the surface pattern is identical.
    Ocean::Options oo;
    oo.size = 900.f;
    oo.sizeZ = 3200.f;
    oo.resolution = waterResX > 0 ? static_cast<uint32_t>(waterResX) : 192;
    oo.resolutionZ = waterResZ > 0 ? static_cast<uint32_t>(waterResZ) : 512;
    oo.windSpeed = startWind;// --wind, and the SAME number the plumes scale from
    oo.windTheta = kWindHeading;
    oo.choppiness = 0.5f;
    oo.tileSize1 = 90.f;
    oo.tileSize2 = tile2Override >= 0.f ? tile2Override : 7.f;
    auto ocean = Ocean::create(oo);
    if (foamAmount >= 0.f) ocean->params.foamAmount = foamAmount;
    ocean->name = "fjord_water";
    // Nordic water: darker, greener absorption than the default tropical teal.
    if (auto* waterMat = ocean->material()->as<MeshPhysicalMaterial>()) {
        waterMat->attenuationColor = Color(0.045f, 0.14f, 0.16f);
        waterMat->attenuationDistance = 1.9f;
    }
    scene.add(ocean);

    // ── forest ──────────────────────────────────────────────────────────────
    {
        const auto tf0 = std::chrono::high_resolution_clock::now();
        // More seeds per species → no two neighbouring trunks share a curve.
        std::vector<TreeVariant> nearVars{
                makeTreeVariant(1, 301u, false),// spruce
                makeTreeVariant(1, 502u, false),// spruce
                makeTreeVariant(1, 877u, false),// spruce
                makeTreeVariant(1, 913u, false),// spruce
                makeTreeVariant(2, 404u, false),// birch accent
                makeTreeVariant(2, 656u, false),// birch accent
        };
        std::vector<TreeVariant> farVars{
                makeTreeVariant(1, 611u, true),
                makeTreeVariant(1, 733u, true),
                makeTreeVariant(1, 858u, true),
        };

        // Collect every valid site first, shuffle, then trim to the cap — a
        // scan-order cap would spend the whole budget in the first rows.
        auto scatter = [&](const std::vector<TreeVariant>& vars, float spacing, float fill,
                           float minD, float maxD, float minPadDist, float maxPadDist,
                           float minScale, float maxScale, int cap, unsigned int seed, float minNy) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            Quaternion q;
            const Vector3 up{0.f, 1.f, 0.f};
            const float half = kWorldSize * 0.5f * 0.95f;
            const int cells = static_cast<int>((2.f * half) / spacing);
            std::vector<Vector3> sites;
            for (int cz = 0; cz < cells; ++cz) {
                for (int cx = 0; cx < cells; ++cx) {
                    if (u01(rng) > fill) continue;
                    const float x = -half + (static_cast<float>(cx) + u01(rng)) * (2.f * half / cells);
                    const float z = -half + (static_cast<float>(cz) + u01(rng)) * (2.f * half / cells);
                    const float d = std::abs(x - channelCenterX(z));
                    if (d < minD || d > maxD) continue;
                    const float dPad = std::sqrt((x - padX) * (x - padX) + (z - kPadZ) * (z - kPadZ));
                    if (dPad > maxPadDist || dPad < minPadDist) continue;
                    if (dPad < 34.f) continue;// keep the meadow open
                    const float h = field.sampleBilinear(x, z);
                    const float treeline = 185.f - 55.f * fbm2(x * 0.006f, z * 0.006f);
                    if (h < 1.2f || h > treeline) continue;
                    if (field.slopeNy(x, z) < minNy) continue;
                    sites.emplace_back(x, terrainH(x, z) - 0.25f, z);
                }
            }
            std::shuffle(sites.begin(), sites.end(), rng);
            if (static_cast<int>(sites.size()) > cap) sites.resize(static_cast<size_t>(cap));

            std::vector<std::vector<Matrix4>> xf(vars.size());
            for (const auto& p : sites) {
                const size_t vi = static_cast<size_t>(u01(rng) * static_cast<float>(vars.size())) % vars.size();
                const float s = minScale + u01(rng) * (maxScale - minScale);
                q.setFromAxisAngle(up, u01(rng) * kTau);
                Matrix4 m;
                m.compose(p, q, Vector3(s, s, s));
                xf[vi].push_back(m);
            }
            for (size_t vi = 0; vi < vars.size(); ++vi) {
                if (xf[vi].empty()) continue;
                auto trunks = InstancedMesh::create(vars[vi].trunkGeo, vars[vi].barkMat, xf[vi].size());
                auto leaves = InstancedMesh::create(vars[vi].leafGeo, vars[vi].leafMat, xf[vi].size());
                for (size_t i = 0; i < xf[vi].size(); ++i) {
                    trunks->setMatrixAt(i, xf[vi][i]);
                    leaves->setMatrixAt(i, xf[vi][i]);
                }
                trunks->instanceMatrix()->needsUpdate();
                leaves->instanceMatrix()->needsUpdate();
                scene.add(trunks);
                scene.add(leaves);
            }
            return static_cast<int>(sites.size());
        };

        // Detailed band around the cabin (where the camera lives), cheap blob
        // canopies across the wider valley and up both banks — but never close
        // to the camera area (a blob canopy up close reads as a green balloon).
        const int nNear = scatter(nearVars, 13.f, 0.9f, kChannelHalf + 22.f, 560.f, 0.f, 750.f, 1.3f, 2.1f, 950, 11u, 0.66f);
        const int nFar = scatter(farVars, 26.f, 0.85f, kChannelHalf + 22.f, 950.f, 480.f, 1e9f, 1.6f, 2.6f, 1100, 23u, 0.58f);
        const auto tf1 = std::chrono::high_resolution_clock::now();
        std::cout << "[fjord] forest: " << nNear << " near + " << nFar << " far trees, "
                  << std::chrono::duration<float>(tf1 - tf0).count() << " s" << std::endl;
    }

    // ── boulders along the shore ────────────────────────────────────────────
    {
        auto rockMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.30f, 0.29f, 0.27f)).roughness(1.f).metalness(0.f));
        rockMat->flatShading = true;
        rockMat->envMapIntensity = 0.35f;
        std::vector<std::shared_ptr<BufferGeometry>> rgeos{terrain::makeRockGeometry(1u),
                                                           terrain::makeRockGeometry(2u),
                                                           terrain::makeRockGeometry(3u)};
        std::vector<std::vector<Matrix4>> xf(rgeos.size());
        std::mt19937 rng(99u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        Quaternion q;
        Vector3 axis;
        Matrix4 m;
        for (int i = 0; i < 90; ++i) {
            const float z = (u01(rng) - 0.5f) * 2200.f;
            const float side = u01(rng) < 0.5f ? -1.f : 1.f;
            const float d = kChannelHalf + 10.f + u01(rng) * 120.f;
            const float x = channelCenterX(z) + side * d;
            const float h = terrainH(x, z);
            if (h < -1.5f || h > 12.f) continue;
            const float sc = 0.5f + u01(rng) * 1.6f;
            axis.set(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f).normalize();
            q.setFromAxisAngle(axis, u01(rng) * kTau);
            m.compose(Vector3(x, h - sc * 0.3f, z), q, Vector3(sc, sc * (0.7f + u01(rng) * 0.4f), sc));
            xf[static_cast<size_t>(u01(rng) * static_cast<float>(rgeos.size())) % rgeos.size()].push_back(m);
        }
        for (size_t gi = 0; gi < rgeos.size(); ++gi) {
            if (xf[gi].empty()) continue;
            auto rocks = InstancedMesh::create(rgeos[gi], rockMat, xf[gi].size());
            for (size_t i = 0; i < xf[gi].size(); ++i) rocks->setMatrixAt(i, xf[gi][i]);
            rocks->instanceMatrix()->needsUpdate();
            scene.add(rocks);
        }
    }

    // ── valley meadow grass (GPU wind, tiled) ───────────────────────────────
    // Widened from the original 46 m disk (15 k blades in one merged mesh) to
    // cover the whole walkable valley floor (~150 m, ~105 k blades ≈ 7x). Built
    // as a grid of GrassMesh tiles so the renderer can frustum/occlusion-cull the
    // off-screen tiles and distance-freeze the wind on the far ones. Same look
    // (material, colours, wind) as before — sway stays seamless across tile
    // borders because the phase derives from each blade's world XZ. The
    // --single-grass flag rebuilds the identical blades as ONE merged mesh: its
    // single valley-spanning AABB is ~always on screen and near, so it is never
    // culled and never freezes — the "no culling" baseline for A/B measurement.
    // ── waterline + footpath spine ──────────────────────────────────────────
    //
    // Hoisted ABOVE the meadow because the grass scatter has to reject blades
    // inside the path corridor: a gravel ribbon laid under a full-density
    // meadow is simply invisible from anywhere but straight overhead, which is
    // exactly how it behaved before this.
    float xWater = padX;
    for (float x = padX; x > padX - 300.f; x -= 1.f) {
        if (field.sampleBilinear(x, kPadZ) < 0.f) {
            xWater = x;
            break;
        }
    }

    const auto fjordCabinMetrics = architecture::cabinMetrics(fjordCabinParams());
    std::vector<Vector3> pathPts;// sampled centreline, world space
    float pathMinX = 0.f, pathMaxX = 0.f, pathMinZ = 0.f, pathMaxZ = 0.f;
    {
        // Foot of the porch steps. The cabin is turned so its local +Z (the
        // porch side) points down -X, and the steps sit on the door
        // centreline, so the landing is straight out along -X at kPadZ.
        const float xStart = padX - (fjordCabinMetrics.porchOuterZ + 0.16f + 0.29f * 3.f + 0.6f);
        const float xEnd = xWater + 6.5f;// meets the dock's landward end
        auto ground = [&](float x, float z) { return Vector3(x, terrainH(x, z), z); };
        auto lerpX = [&](float t) { return xStart + (xEnd - xStart) * t; };

        // A lazy S, not a straight run: a path worn by feet never takes the
        // shortest line, and a ruler-straight strip reads as CG immediately.
        CatmullRomCurve3 spine({
                ground(xStart + 1.2f, kPadZ + 0.1f),
                ground(lerpX(0.00f), kPadZ + 0.5f),
                ground(lerpX(0.20f), kPadZ + 4.2f),
                ground(lerpX(0.45f), kPadZ + 2.4f),
                ground(lerpX(0.70f), kPadZ - 3.0f),
                ground(lerpX(0.90f), kPadZ - 1.0f),
                ground(xEnd, kPadZ + 0.3f),
        });
        constexpr int kSteps = 220;
        pathPts.reserve(kSteps + 1);
        for (int i = 0; i <= kSteps; ++i) {
            Vector3 c;
            spine.getPoint(static_cast<float>(i) / static_cast<float>(kSteps), c);
            pathPts.push_back(c);
        }
        pathMinX = pathMaxX = pathPts[0].x;
        pathMinZ = pathMaxZ = pathPts[0].z;
        for (const auto& p : pathPts) {
            pathMinX = std::min(pathMinX, p.x);
            pathMaxX = std::max(pathMaxX, p.x);
            pathMinZ = std::min(pathMinZ, p.z);
            pathMaxZ = std::max(pathMaxZ, p.z);
        }
    }

    // Distance from a world XZ to the path centreline. The bounding-box reject
    // comes first so the 2 M-attempt grass loop pays only a few compares for
    // the ~99% of samples nowhere near the corridor.
    constexpr float kPathClear = 1.25f;// bare gravel out to here
    constexpr float kPathFeather = 2.30f;// grass returns to full density here
    auto pathDistance = [&](float x, float z) {
        if (x < pathMinX - kPathFeather || x > pathMaxX + kPathFeather ||
            z < pathMinZ - kPathFeather || z > pathMaxZ + kPathFeather)
            return 1e9f;
        float best = 1e9f;
        for (size_t i = 1; i < pathPts.size(); ++i) {
            const Vector3& a = pathPts[i - 1];
            const Vector3& b = pathPts[i];
            const float ex = b.x - a.x, ez = b.z - a.z;
            const float len2 = ex * ex + ez * ez;
            float t = 0.f;
            if (len2 > 1e-8f) t = std::clamp(((x - a.x) * ex + (z - a.z) * ez) / len2, 0.f, 1.f);
            const float dx = x - (a.x + ex * t), dz = z - (a.z + ez * t);
            best = std::min(best, dx * dx + dz * dz);
        }
        return std::sqrt(best);
    };

    std::vector<std::shared_ptr<GrassMesh>> grassTiles;
    {
        constexpr float kMeadowRadius = 190.f;   // valley-floor coverage (was 46)
        constexpr size_t kBladeTarget = 300000;  // ~17x the AREA at the original density
        std::vector<vegetation::GrassBlade> blades;
        blades.reserve(kBladeTarget);
        std::mt19937 rng(7u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const Vector3 up{0.f, 1.f, 0.f};
        // Bounded attempt loop — rejection sampling (water / rock walls) can't
        // guarantee the target count, so cap the attempts instead of spinning.
        for (int a = 0; a < 2000000 && blades.size() < kBladeTarget; ++a) {
            const float ang = u01(rng) * kTau;
            const float rr = std::sqrt(u01(rng)) * kMeadowRadius;
            const float x = padX + std::cos(ang) * rr;
            const float z = kPadZ + std::sin(ang) * rr;
            const float h = terrainH(x, z);
            if (h < 0.6f) continue;   // not into the water
            if (h > 52.f) continue;   // stay off the rock walls / scree
            // Thin out as the ground climbs so grass fades into the slope rather
            // than ending in a hard altitude line (reuses the terrain's smoothstep).
            if (h > 34.f && u01(rng) < math::smoothstep(34.f, 52.f, h)) continue;
            // Clear the footpath. FEATHERED, not a hard cut: a corridor mown to
            // a crisp edge reads as a stencil, while thinning over a metre
            // reads as ground worn bare by use.
            const float dPath = pathDistance(x, z);
            if (dPath < kPathFeather) {
                if (dPath < kPathClear) continue;
                if (u01(rng) > math::smoothstep(kPathClear, kPathFeather, dPath)) continue;
            }
            vegetation::GrassBlade bl;
            bl.position.set(x, h - 0.04f, z);
            const float s = 0.5f + u01(rng) * 0.5f;
            bl.scale.set(s, 0.28f + u01(rng) * 0.38f, s);
            bl.yaw.setFromAxisAngle(up, u01(rng) * kTau);
            blades.push_back(bl);
        }
        auto grassMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.42f, 0.50f, 0.26f)).roughness(1.f).metalness(0.f));
        grassMat->vertexColors = true;
        grassMat->side = Side::Double;
        grassMat->envMapIntensity = 0.35f;

        GrassMesh::Params mp;
        // The meadow leans the way the waves run and the smoke goes (one wind).
        mp.windDir = Vector2(windDir.x, windDir.z);
        mp.windStrength = 0.16f;
        // Freeze the wind + BLAS refit on tiles whose nearest edge is farther
        // than this from the camera; ~4 tile widths keeps every visibly-swaying
        // blade animated while dropping the far majority. (Ignored below for the
        // single-mesh baseline so it always animates the whole field.)
        mp.maxAnimDistance = 95.f;

        if (singleGrass) {
            auto one = GrassMesh::create(vegetation::buildGrassGeometry(blades), grassMat);
            one->name = "meadow";
            one->params = mp;
            one->params.maxAnimDistance = 0.f;// baseline never freezes
            scene.add(one);
            grassTiles.push_back(one);
        } else {
            // ~40 m tiles over a 190 m disk ⇒ ~70 non-empty tiles: coarse enough
            // to keep the per-tile CPU/record overhead (one BLAS refit + one TLAS
            // instance + one draw each) modest, fine enough to cull/freeze most of
            // the field when the camera isn't standing in the middle of it.
            grassTiles = vegetation::buildGrassTiles(blades, 40.f, grassMat, mp);
            for (auto& t : grassTiles) scene.add(t);
        }
        std::cout << "meadow: " << blades.size() << " blades in "
                  << grassTiles.size() << (singleGrass ? " merged mesh" : " tiles")
                  << std::endl;
    }

    // ── cabin, dock, lantern, boat ──────────────────────────────────────────
    Cabin cabin = makeCabin();
    constexpr float kCabinScale = 1.0f;
    cabin.group->position.set(padX, kPadHeight - 0.1f, kPadZ);
    cabin.group->scale.set(kCabinScale, kCabinScale, kCabinScale);
    // The generator builds with the porch on +Z; turn it to face the water
    // (-X), so the veranda looks down the dock and out over the fjord.
    cabin.group->rotation.y = -kPi * 0.5f;
    scene.add(cabin.group);

    // A second, distant cabin across the fjord — its lit windows carry across
    // the water at night.
    std::shared_ptr<MeshStandardMaterial> farWindowMat;
    {
        auto other = makeCabin();
        const float z2 = -430.f;
        const float x2 = channelCenterX(z2) - 205.f;
        other.group->position.set(x2, terrainH(x2, z2) - 0.25f, z2);
        // West bench, so it faces the channel the other way (+X).
        other.group->rotation.y = kPi * 0.5f;
        other.group->name = "cabin_far";
        scene.add(other.group);
        // Both cabins now share one material set, so the near cabin's glow
        // already drives this one; a second write would only cancel its
        // flicker. Left null on purpose — the frame loop guards on it.
        farWindowMat = nullptr;
    }

    auto woodMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color(0.24f, 0.18f, 0.13f)).roughness(0.9f).metalness(0.f));
    {
        auto dock = Group::create();
        const float dockLen = 16.f, dockW = 2.2f;
        const float x0 = xWater + 5.f;// start on land
        // planks
        for (float dx = 0.f; dx < dockLen; dx += 0.66f) {
            auto plank = Mesh::create(BoxGeometry::create(0.6f, 0.07f, dockW), woodMat);
            plank->position.set(x0 - dx, 0.78f, kPadZ);
            dock->add(plank);
        }
        // rails/beams
        for (int sz = -1; sz <= 1; sz += 2) {
            auto beam = Mesh::create(BoxGeometry::create(dockLen, 0.12f, 0.14f), woodMat);
            beam->position.set(x0 - dockLen * 0.5f + 0.3f, 0.70f, kPadZ + static_cast<float>(sz) * (dockW * 0.5f - 0.1f));
            dock->add(beam);
        }
        // posts
        for (float dx = 0.6f; dx < dockLen + 1.f; dx += 3.6f) {
            for (int sz = -1; sz <= 1; sz += 2) {
                auto post = Mesh::create(CylinderGeometry::create(0.09f, 0.10f, 4.6f, 8), woodMat);
                post->position.set(x0 - dx, 0.78f - 2.3f, kPadZ + static_cast<float>(sz) * (dockW * 0.5f - 0.12f));
                dock->add(post);
            }
        }
        scene.add(dock);
    }

    // ── footpath: cabin steps down to the pier ──────────────────────────────
    //
    // A gravel ribbon laid on the terrain. Two things keep it from reading as
    // a decal painted on a hillside:
    //
    //   * It WANDERS. A path worn by feet never takes the shortest line, and a
    //     ruler-straight strip across a meadow is the first thing the eye
    //     rejects. The centreline is a Catmull-Rom through offset waypoints and
    //     the half-width breathes along it.
    //   * It has a SKIRT. The ribbon samples the same tile heightfield the
    //     terrain renders from, but sub-metre relief still slips between
    //     samples; edges dropped a third of a metre bury those crossings
    //     instead of leaving the path hovering over its own shadow.
    {
        auto pathTex = architecture::makeStoneTextures(256, 5u, {0.315f, 0.278f, 0.226f});
        auto pathMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.98f).metalness(0.f));
        pathMat->map = pathTex.first;
        pathMat->normalMap = pathTex.second;
        pathMat->normalScale.set(0.8f, 0.8f);

        // Ribbon built from the SAME sampled centreline the grass scatter
        // rejected against, so the bare corridor and the gravel cannot drift
        // apart if either is retuned.
        const int kSteps = static_cast<int>(pathPts.size()) - 1;
        constexpr float kSkirt = 0.34f;
        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        float arc = 0.f;
        Vector3 prev;
        for (int i = 0; i <= kSteps; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(kSteps);
            const Vector3& c = pathPts[static_cast<size_t>(i)];
            // Tangent from the neighbouring samples in the XZ plane.
            const Vector3& a = pathPts[static_cast<size_t>(std::max(0, i - 1))];
            const Vector3& b = pathPts[static_cast<size_t>(std::min(kSteps, i + 1))];
            float tx = b.x - a.x, tz = b.z - a.z;
            const float tl = std::sqrt(tx * tx + tz * tz);
            if (tl > 1e-5f) { tx /= tl; tz /= tl; }
            else { tx = 1.f; tz = 0.f; }
            const float px = -tz, pz = tx;// left normal in XZ

            if (i > 0) arc += c.distanceTo(prev);
            prev = c;

            // Worn wider where it is walked most, narrower on the pinch points.
            const float half = 0.62f + 0.30f * noise::valueNoise(t * 9.f, 0.5f, 9, 17u);
            for (int e = 0; e < 4; ++e) {
                // 0,3 = skirt edges (dropped); 1,2 = the walked surface.
                const float s = (e == 0 || e == 1) ? 1.f : -1.f;
                const float w = (e == 0 || e == 3) ? half + 0.22f : half;
                const float drop = (e == 0 || e == 3) ? kSkirt : 0.f;
                const float x = c.x + px * w * s;
                const float z = c.z + pz * w * s;
                pos.insert(pos.end(), {x, terrainH(x, z) + 0.035f - drop, z});
                nrm.insert(nrm.end(), {0.f, 1.f, 0.f});
                uv.insert(uv.end(), {(0.5f + 0.5f * s * (w / 0.9f)), arc / 0.9f});
            }
            if (i > 0) {
                const auto b0 = static_cast<unsigned int>((i - 1) * 4);
                const auto b1 = static_cast<unsigned int>(i * 4);
                for (int e = 0; e < 3; ++e) {
                    const auto q0 = b0 + static_cast<unsigned int>(e);
                    const auto q1 = b1 + static_cast<unsigned int>(e);
                    idx.insert(idx.end(), {q0, q1, q0 + 1, q0 + 1, q1, q1 + 1});
                }
            }
        }
        auto pathGeo = BufferGeometry::create();
        pathGeo->setIndex(idx);
        pathGeo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        pathGeo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        pathGeo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        pathGeo->computeVertexNormals();
        pathGeo->computeBoundingSphere();

        auto path = Mesh::create(pathGeo, pathMat);
        path->name = "footpath";
        path->receiveShadow = true;
        path->castShadow = false;
        scene.add(path);
    }

    // Lantern on the dock end.
    auto lanternMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}
                    .color(Color(0.9f, 0.8f, 0.6f))
                    .roughness(0.4f)
                    .metalness(0.f)
                    .emissive(Color(1.0f, 0.72f, 0.38f))
                    .emissiveIntensity(0.f));
    {
        auto post = Mesh::create(CylinderGeometry::create(0.05f, 0.06f, 1.25f, 8), woodMat);
        post->position.set(xWater + 5.f - 15.4f, 0.78f + 0.62f, kPadZ + 0.9f);
        scene.add(post);
        auto bulb = Mesh::create(SphereGeometry::create(0.10f, 16, 12), lanternMat);
        bulb->position.set(xWater + 5.f - 15.4f, 0.78f + 1.32f, kPadZ + 0.9f);
        scene.add(bulb);
    }

    // Rowing boat moored off the dock end.
    auto boat = Group::create();
    {
        auto hullMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.72f, 0.70f, 0.64f)).roughness(0.55f).metalness(0.f));
        hullMat->side = Side::Double;
        auto hull = Mesh::create(makeBoatHull(), hullMat);
        boat->add(hull);
        for (float tz : {-0.7f, 0.9f}) {
            auto thwart = Mesh::create(BoxGeometry::create(1.0f, 0.05f, 0.28f), woodMat);
            thwart->position.set(0.f, 0.34f, tz);
            boat->add(thwart);
        }
        boat->position.set(xWater + 5.f - 18.5f, -0.14f, kPadZ - 2.6f);
        boat->rotation.y = 0.5f;
        scene.add(boat);
    }

    // ── chimney smoke ───────────────────────────────────────────────────────
    // A PARTICIPATING MEDIUM, not a stack of sprites. The plume is one
    // ParticleField with a DensityRepr — the fjord's only density volume, 1 of
    // the 4 slots — so it is marched by the same `applyParticleFog` the rest of
    // the atmosphere is: Beer-Lambert extinction, sun in-scatter through an HG
    // phase function WITH a shadow ray (which is why the plume goes bright where
    // the sun rakes through it and grey where the roof shades it), ambient
    // in-scatter, and correct compositing against the height fog and the sky
    // rather than an alpha-blended quad drawn on top of them.
    //
    // WHAT IT REPLACES, and why the swap is worth a comment: ~156 live textured
    // puffs on the legacy ParticleSystem path, integrated on the CPU, drawn as
    // camera-facing sprites in the overlay pass. Those sprites are OPAQUE to the
    // idea of fog — they occlude what is behind them and are lit by nothing —
    // so the plume read as crisper than the trees 30 m behind it, and its own
    // RNG made two captures of the same pose differ. --legacy-smoke keeps that
    // path in this binary for the A/B; the library's ParticleSystem is untouched
    // either way.
    //
    // THE PLUME MODEL IS NOT WRITTEN HERE. It is FireEffect's, in smoke-only
    // mode: the three clocks (buoyancy s^0.62, advection LINEAR in age,
    // turbulent dispersion s^0.55 — see plans/particle-atmosphere.md, F4 defect
    // 3.1 for why one clock makes a collimated jet), the per-parcel taper that
    // keeps the downwind end from stopping on a plane, and recomputeSmokeBox(),
    // which derives the volume's DOMAIN from the wind because the splat DROPS a
    // particle outside it. A chimney is a fire whose flame you cannot see, so
    // smokeOnly gives us the plume with no flame volume, no embers and no
    // PointLight — one field, one slot.
    //
    // Local -> world through the group's matrix, NOT position + scale: the cabin
    // is rotated to face the water, so a naive offset would put the smoke column
    // out in the meadow beside the roof. The effect is then parented to the
    // SCENE and not to the cabin: FireEffect composes its world box as
    // boxLocal + worldPosition, which is only the truth under an unrotated
    // parent chain.
    cabin.group->updateMatrixWorld(true);
    Vector3 chimneyTip = cabin.chimneyTipLocal;
    chimneyTip.applyMatrix4(*cabin.group->matrixWorld);

    std::shared_ptr<FireEffect> plume;
    if (!legacySmoke) {
        FireEffect::Params fp;
        fp.smokeOnly = true;// one field: no flame, no embers, no light
        // Source geometry, not a flame: the flue's mouth. `height` only sets
        // where the column starts above the effect's origin (0.55 × it), which
        // for a chimney should be ~nothing — the smoke leaves AT the tip.
        fp.height = 0.30f;
        fp.radius = 0.22f;// the flue bore, so the column emerges thin
        // 7 m of climb before the buoyancy gives out. The old sprite plume rose
        // ~7.5 m over its 6.5 s life, so this is the same column height the
        // scene was authored around — what changes is that it now BENDS.
        fp.smokeHeight = 7.0f;
        fp.smokeSpread = 1.90f;
        // The plume STREAMS: a parcel is followed 3× its own 3.2-5.6 s period,
        // so at the fjord's 0.63 m/s surface wind it travels ~9-11 m downwind
        // while it climbs. The taper frays the far end (F1 note 2's per-parcel
        // ceiling, applied downwind) so it disperses instead of ending on a
        // plane, and it is the reason 30% of the slots are dead at any instant.
        fp.smokeLifeScale = 3.0f;
        fp.smokeTaper     = 0.50f;
        // ── sigma is per-PARTICLE; the quantity to reason with is N × sigma ──
        // A few thousand slots is right for a chimney (the shoreline campfire
        // this demo used to carry ran 18k over a box twice as long). At 5k over
        // a ~10 × 7 × 14 m box at 32³ the mean occupancy is ~1 particle per
        // voxel, which the trilinear splat spreads to ~8 taps — above the grain
        // threshold, and the reason the resolution is 32 and not 48: the finer
        // grid would put 0.4 particles in a voxel and the plume would speckle.
        fp.smokeParticles  = 5'000;
        fp.smokeResolution = 32;
        fp.smokeSigma      = 0.20f;
        // Wood smoke off a stove is pale — condensed water and light ash, not
        // the soot of an open flame — so it scatters far more of what it
        // extinguishes than FireEffect's default 0.25 campfire smoke. This is
        // the number that decides whether the sun makes the plume GLOW or just
        // makes it a grey silhouette.
        fp.smokeAlbedo     = Color(0.50f, 0.50f, 0.51f);
        fp.smokeAnisotropy = 0.35f;// forward-scattering: bright with the sun behind it
        // ── ONE WIND, LATCHED AT INIT ────────────────────────────────────────
        // The same vector the waves, the clouds, the grass and (before it) the
        // sprite plume derive from. LATCHED, exactly as the legacy path latched
        // it at initialize(), and the reason is stronger here than it was there:
        // this emitter is a CLOSED FORM in age, so setWind() does not steer the
        // parcels that are already flying, it re-evaluates where they have
        // ALWAYS been — the whole plume re-aims retroactively, in one frame, as
        // a visible snap. Wind-at-init is the honest v1. Driving the slider live
        // needs the wind to enter the closed form as a history (an integral of
        // past wind over the parcel's age), which is a real feature and not a
        // parameter change; it is written up in the plan as an open issue.
        fp.wind.copy(groundWind0);
        fp.seed = 4711u;
        plume = FireEffect::create(fp);
        plume->name = "chimney_plume";
        plume->position.copy(chimneyTip);
        scene.add(plume);
        plume->ignite();
    }

    // ── the legacy sprite plume (--legacy-smoke), verbatim ──────────────────
    ParticleSystem smoke;
    if (legacySmoke) {
        auto& s = smoke.settings();
        s.makeDefault();
        s.positionStyle = ParticleSystem::Type::BOX;
        s.positionBase = chimneyTip;
        s.positionSpread = {0.15f, 0.05f, 0.15f};
        s.velocityStyle = ParticleSystem::Type::BOX;
        // ONE WIND: the plume's horizontal velocity IS the scene's surface wind.
        // Latched here, at initialize() time, rather than driven by the slider —
        // this is the legacy ParticleSystem path, whose spawn parameters are
        // read when a particle is born and whose live puffs cannot be re-aimed.
        // A wind change therefore reaches the chimney over one plume lifetime,
        // which is also what a real plume does.
        s.velocityBase = {groundWind0.x, 1.15f, groundWind0.z};
        s.velocitySpread = {0.35f, 0.35f, 0.35f};
        s.accelerationBase = {groundWind0.x * 0.16f, 0.10f, groundWind0.z * 0.16f};
        s.particlesPerSecond = 24;
        s.particleDeathAge = 6.5f;
        s.emitterDeathAge = 1e9f;
        s.colorBase = {0.f, 0.f, 0.78f};// HSL: grey
        s.colorSpread = {0.f, 0.f, 0.10f};
        s.setOpacityTween({0.f, 1.2f, 6.5f}, {0.f, 0.30f, 0.f})
                .setSizeTween({0.f, 6.5f}, {0.7f, 3.6f});
        TextureLoader tl;
        s.texture = tl.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png", ColorSpace::sRGB);
        smoke.initialize();
        scene.addRef(smoke);
    }

    // ── sun / moon lights ───────────────────────────────────────────────────
    auto sun = DirectionalLight::create(Color(1.f, 0.95f, 0.88f), 3.f);
    sun->name = "sun";
    scene.add(sun);
    auto moon = DirectionalLight::create(Color(0.62f, 0.72f, 1.0f), 0.f);
    moon->name = "moon";
    scene.add(moon);

    // ── sky env (ping-pong bake) ────────────────────────────────────────────
    constexpr int SKY_W = 1024, SKY_H = 512;
    SkyModel sky;
    std::array<std::shared_ptr<Texture>, 2> skyTex;
    for (auto& t : skyTex) {
        std::vector<float> data(static_cast<size_t>(SKY_W) * SKY_H * 4, 0.f);
        Image img{std::move(data), static_cast<unsigned>(SKY_W), static_cast<unsigned>(SKY_H)};
        t = Texture::create(img);
        t->format = Format::RGBA;
        t->type = Type::Float;
        t->colorSpace = ColorSpace::Linear;
        t->mapping = Mapping::EquirectangularReflection;
        t->needsUpdate();
    }
    int skyFront = 0;

    float timeOfDay = startTime;
    float lastBakedElev = -99.f, lastBakedAz = -99.f;
    float sinceBake = 1e9f;

    auto applySky = [&](const CelestialState& cs, bool force) {
        // Re-bake only when the sun has moved enough (throttled).
        const float elevDeg = std::asin(std::clamp(cs.sunDir.y, -1.f, 1.f)) * 180.f / kPi;
        const float azDeg = std::atan2(cs.sunDir.x, -cs.sunDir.z) * 180.f / kPi;
        if (!force && sinceBake < 0.30f) return;
        if (!force && std::abs(elevDeg - lastBakedElev) < 0.25f && std::abs(azDeg - lastBakedAz) < 0.6f) return;
        lastBakedElev = elevDeg;
        lastBakedAz = azDeg;
        sinceBake = 0.f;

        const int back = 1 - skyFront;
        bakeSkyInto(skyTex[back]->image().data<float>(), SKY_W, SKY_H, sky, cs);
        skyTex[back]->needsUpdate();
        skyFront = back;
        scene.environment = skyTex[skyFront];
        scene.background = skyTex[skyFront];
    };

    // ── camera ──────────────────────────────────────────────────────────────
    PerspectiveCamera camera(55.f, canvas.aspect(), 0.5f, 9000.f);
    camera.position.set(padX - 120.f, 11.f, kPadZ + 130.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(padX - 20.f, 6.f, kPadZ - 60.f);
    controls.update();

    // Cinematic flight path (closed loop over water / meadow / treetops).
    CatmullRomCurve3 flightPath({
            {padX - 150.f, 7.f, kPadZ + 190.f},
            {padX - 210.f, 10.f, kPadZ - 60.f},
            {channelCenterX(kPadZ - 330.f), 16.f, kPadZ - 330.f},
            {channelCenterX(-500.f) + 40.f, 34.f, -500.f},
            {channelCenterX(-250.f) - 150.f, 60.f, -250.f},
            {padX - 60.f, 95.f, kPadZ + 330.f},
            {padX + 100.f, 30.f, kPadZ + 250.f},
            {padX + 10.f, 9.f, kPadZ + 80.f},
    }, true, CatmullRomCurve3::centripetal);
    bool cinematic = startFly;
    float pathU = 0.f;
    constexpr float kPathPeriod = 95.f;

    // Shot camera presets.
    auto applyShotCam = [&](int idx) {
        switch (idx) {
            default:
            case 0:// hero: from the water toward the cabin, fjord receding
                camera.position.set(xWater - 46.f, 5.5f, kPadZ + 62.f);
                controls.target.set(padX - 4.f, 5.f, kPadZ - 18.f);
                break;
            case 1:// down-fjord vista
                camera.position.set(padX - 60.f, 22.f, kPadZ + 60.f);
                controls.target.set(channelCenterX(-900.f), 40.f, -900.f);
                break;
            case 2:// aerial
                camera.position.set(padX + 260.f, 260.f, kPadZ + 380.f);
                controls.target.set(channelCenterX(-200.f), 0.f, -200.f);
                break;
            case 3:// dock close-up
                camera.position.set(xWater - 16.f, 2.4f, kPadZ + 9.f);
                controls.target.set(padX, 4.f, kPadZ);
                break;
            case 4:// INSIDE the valley meadow — grass fills the frame (worst case
                   // for grass cost; most tiles on screen and near ⇒ animated)
                camera.position.set(padX + 18.f, 5.5f, kPadZ + 18.f);
                controls.target.set(padX - 90.f, 3.0f, kPadZ - 90.f);
                break;
            case 5:// standing in the meadow but looking AWAY up-fjord — most grass
                   // is behind the camera (best case: frustum-culled) and the rest
                   // is far (distance-frozen wind)
                camera.position.set(padX, 7.f, kPadZ - 120.f);
                controls.target.set(channelCenterX(-700.f), 90.f, -700.f);
                break;
            case 6:// HIGH overlook, steep pitch DOWN — terrain+water fill most of
                   // the frame while the horizon sits high, leaving a broad SKY band
                   // (with the cloud deck) across the top. This is the black-sky
                   // repro: sky pixels feed compositeClouds a 1e30 sceneDist, whose
                   // camera→cloud fog leg overflowed the closed-form optical depth.
                camera.position.set(padX + 80.f, 380.f, kPadZ + 300.f);
                controls.target.set(channelCenterX(-350.f), 0.f, -350.f);
                break;
        }
        controls.update();
    };
    if (headlessish) applyShotCam(shotCam);
    if (hasCamPos) camera.position.set(camPos[0], camPos[1], camPos[2]);
    if (hasCamLook) controls.target.set(camLook[0], camLook[1], camLook[2]);
    if (hasCamPos || hasCamLook) controls.update();
    // The seq orbit / dolly anchor: the pose the presets / overrides left behind.
    const Vector3 seqEye0    = camera.position;
    const Vector3 seqTarget0 = controls.target;
    if (!seqDir.empty()) std::filesystem::create_directories(seqDir);

    // ── state + UI ──────────────────────────────────────────────────────────
    float cycleSpeed = startCycle;// hours per second (0 = paused)
    float windSpeed = startWind;
    float fogScale = startFogScale;
    bool cloudsOn = !noClouds;
    float cloudCover = 0.42f;
    // Cloud application is ON-CHANGE, not every-frame: the fjord used to call
    // setClouds() unconditionally each frame, which STOMPED the shared
    // RendererSettings panel's cloud checkbox/sliders (uncheck → the demo
    // re-enabled it next frame → the getter-driven box flipped back). We now
    // re-apply only when one of the demo's own cloud inputs actually changes, so
    // external panel edits stick. The per-frame wind DRIFT itself lives in the
    // shader timeSec, so on-change writes do not freeze it.
    bool  cloudsApplied      = false;   // has the demo pushed its state at least once?
    bool  appliedCloudsOn    = false;
    float appliedCloudCover  = -1.f;
    float appliedWindSpeed   = -1.f;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    // Generic renderer settings (render scale, auto LOD, frame timings, ...)
    // come from the shared panel; the demo-specific weather/time widgets live
    // in the extra lambda. Interactive runs only — the headless capture path
    // must not draw UI into the measured frames.
    std::unique_ptr<RendererSettingsUi> ui;
    if (!headlessish) {
        ui = std::make_unique<RendererSettingsUi>(canvas, renderer, [&] {
            ImGui::TextDisabled("terrain tiles %d (baking %d)", tiles->activeTiles(), tiles->pendingBakes());
            ImGui::SeparatorText("Time of day");
            ImGui::SliderFloat("hour", &timeOfDay, 0.f, 24.f, "%.2f");
            ImGui::SliderFloat("speed (h/s)", &cycleSpeed, 0.f, 1.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
            if (ImGui::Button("dawn")) timeOfDay = 6.1f;
            ImGui::SameLine();
            if (ImGui::Button("noon")) timeOfDay = 12.5f;
            ImGui::SameLine();
            if (ImGui::Button("golden")) timeOfDay = 17.6f;
            ImGui::SameLine();
            if (ImGui::Button("night")) timeOfDay = 23.6f;
            ImGui::SeparatorText("Weather");
            ImGui::SliderFloat("wind (m/s)", &windSpeed, 0.5f, 12.f, "%.1f");
            ImGui::SliderFloat("haze", &fogScale, 0.f, 4.f, "%.2f");
            ImGui::Checkbox("clouds", &cloudsOn);
            if (cloudsOn) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(-90.f);
                ImGui::SliderFloat("coverage", &cloudCover, 0.f, 1.f, "%.2f");
            }
            ImGui::SeparatorText("Camera");
            ImGui::Checkbox("cinematic flight (C)", &cinematic);
            ImGui::TextDisabled("drag=orbit scroll=zoom SPACE=pause");
        }, "FJORD");
    }

    // F9 = repro dump: when an artefact shows up in a LIVE session, one press
    // saves the frame plus a sidecar with the exact camera/time/weather state
    // AND a ready-to-run --shot command line that replays the view headlessly.
    // Turns "I saw it while flying around" into a deterministic capture.
    bool reproDumpRequest = false;
    int  reproDumpIndex = 0;
    KeyAdapter keyAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ui && ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::C) cinematic = !cinematic;
        if (evt.key == Key::SPACE) cycleSpeed = cycleSpeed > 0.f ? 0.f : 0.08f;
        if (evt.key == Key::F9) reproDumpRequest = true;
        // Day / night: a jump between the two authored hours rather than a
        // cross-fade, on purpose — the point of the key is to compare them, and
        // a 9-hour sweep is what the `speed` slider is for.
        if (evt.key == Key::D) {
            const bool isNight = timeOfDay > 20.f || timeOfDay < 5.f;
            timeOfDay = isNight ? kDayHour : kNightHour;
        }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    // Initial sky.
    applySky(celestialAt(timeOfDay), true);

    Clock clock;
    float tElapsed = 0.f;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        tElapsed += dt;
        sinceBake += dt;
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum > 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        // ── advance the day ──
        timeOfDay += cycleSpeed * dt;
        if (timeOfDay >= 24.f) timeOfDay -= 24.f;
        const CelestialState cs = celestialAt(timeOfDay);
        applySky(cs, false);

        // Sun light: direction + Preetham transmittance tint.
        {
            sun->position.copy(cs.sunDir).multiplyScalar(1500.f);
            // Same atmosphere as the sky: SkyModel's own coefficients and slant
            // extinction, evaluated toward the sun. (This block used to keep a
            // private copy with truncated constants and a hard-coded Rayleigh
            // scale — the sun could drift out of colour agreement with its sky.)
            Vector3 betaR, betaM;
            sky.scatterCoeffs(cs, betaR, betaM);
            const Vector3 fex = SkyModel::extinction(betaR, betaM, cs.sunDir.y);
            Color tint(fex.x, fex.y, fex.z);
            // Normalise the transmittance tint so a low sun turns ORANGE without
            // losing its punch (raw Beer-Lambert kills the energy with the hue),
            // then soften toward warm-white — full spectral tint paints whole
            // mountain faces flamingo pink.
            const float maxC = std::max(tint.r, std::max(tint.g, tint.b));
            if (maxC > 1e-4f) {
                tint.r /= maxC;
                tint.g /= maxC;
                tint.b /= maxC;
            }
            tint.lerp(Color(1.f, 0.80f, 0.62f), 0.35f);
            sun->color.copy(tint);
            sun->intensity = 3.6f * math::smoothstep(-0.03f, 0.12f, cs.sunElev) * (0.35f + 0.65f * maxC);
        }
        // Moon light at night.
        {
            moon->position.copy(cs.moonDir).multiplyScalar(1500.f);
            moon->intensity = 0.38f * cs.moonUp * (1.f - cs.daylight);
        }

        // Stars fade in only once the sun is well below the horizon (-3°→-10°).
        renderer.setDeferredStarfield(1.1f * (1.f - math::smoothstep(-0.18f, -0.05f, cs.sunElev)));

        // Fog (colour follows the sky horizon; dawn mist peaks ~06:20):
        //   • Underwater MURK — the below-waterline absorption, ALWAYS on, its
        //     dawn-peaked density matching the pre-Phase-2 look. Clipped to y<0 by
        //     the water surface (setFogWaterSurfaceY above).
        //   • AIR fog (scene.fog) — the unified volumetric medium (haze, god rays,
        //     aerial perspective), driven by the "haze" slider (fogScale; default
        //     0 = clear air). Unclipped — it fills the fjord above the waterline.
        {
            const float mist = std::exp(-std::pow((timeOfDay - 6.3f) / 1.5f, 2.f));
            const Color hz = horizonColor(sky, cs);
            renderer.setUnderwaterMurk(0.00035f + 0.0009f * mist, hz);
            const float airDensity = (0.00035f + 0.0009f * mist) * fogScale;
            if (airDensity > 1e-6f) {
                scene.fog = FogExp2(hz, airDensity);
            } else {
                scene.fog.reset();
            }
        }

        // Volumetric cloud deck hugging the ridge line (peaks ~470 m) — the
        // summits pierce the base. The wind slider drives the drift — clouds
        // at altitude run ~3× the surface wind, on the same heading the ocean
        // waves use (setWind dir 2.1 rad). ON-CHANGE only (see the trackers
        // above): re-applying every frame would overwrite the shared panel's
        // cloud controls. --noclouds keeps cloudsOn permanently false.
        if (!cloudsApplied || cloudsOn != appliedCloudsOn ||
            cloudCover != appliedCloudCover || windSpeed != appliedWindSpeed) {
            if (cloudsOn) {
                VulkanRenderer::CloudSettings cl;
                cl.coverage = cloudCover;
                cl.bottomY = 250.f;
                cl.topY = 750.f;
                cl.wind.copy(windDir);
                cl.wind.multiplyScalar(windSpeed * 3.f);
                renderer.setClouds(cl);
            } else {
                renderer.setClouds(std::nullopt);
            }
            cloudsApplied     = true;
            appliedCloudsOn   = cloudsOn;
            appliedCloudCover = cloudCover;
            appliedWindSpeed  = windSpeed;
        }

        // Window / lantern glow after sundown. emissiveIntensity is a plain
        // field — bump the material version so the renderer re-uploads it.
        {
            const float glow = 1.f - math::smoothstep(-0.07f, 0.02f, cs.sunElev);
            const float flicker = 1.f + 0.05f * std::sin(tElapsed * 9.f) * std::sin(tElapsed * 3.7f);
            const float target = 26.f * glow * flicker;
            if (std::abs(target - cabin.windowMat->emissiveIntensity) > 0.2f) {
                cabin.windowMat->emissiveIntensity = target;
                cabin.windowMat->needsUpdate();
                if (farWindowMat) {
                    farWindowMat->emissiveIntensity = 26.f * glow;
                    farWindowMat->needsUpdate();
                }
                lanternMat->emissiveIntensity = 42.f * glow;
                lanternMat->needsUpdate();
                if (cabin.sconceMat) {
                    cabin.sconceMat->emissiveIntensity = 18.f * glow * flicker;
                    cabin.sconceMat->needsUpdate();
                }
            }
        }

        // Terrain LOD follows the camera (async tile bakes, budgeted swaps).
        tiles->update(camera.position);

        // Water, grass, smoke, boat.
        ocean->setWind(windSpeed, kWindHeading);
        // The water grid is STATIC (a channel-fitting rectangle, no camera
        // warp): the mesh never re-tessellates under the viewer, so per-vertex
        // motion vectors stay valid and the surface doesn't subtly reflow as
        // the camera moves.
        // Advance the wind clock on ALL tiles (frozen far tiles ignore it until
        // they re-enter range). Only time + windStrength change; windDir and
        // maxAnimDistance persist from setup.
        for (auto& t : grassTiles) {
            t->params.time = tElapsed;
            t->params.windStrength = 0.10f + 0.02f * windSpeed;
        }
        // The plume takes ABSOLUTE time (its emitter is a closed form in t, so a
        // headless capture of frame 600 does not depend on frames 1..599); the
        // legacy sprite path integrates and takes the delta. The wind slider
        // does NOT reach either of them — see the latch note at construction.
        if (plume) plume->update(tElapsed);
        else smoke.update(dt);
        boat->position.y = -0.14f + 0.05f * std::sin(tElapsed * 0.9f) + 0.02f * std::sin(tElapsed * 1.7f + 1.f);
        boat->rotation.z = 0.030f * std::sin(tElapsed * 0.7f);
        boat->rotation.x = 0.020f * std::sin(tElapsed * 1.1f + 1.f);

        // Camera.
        if (cinematic) {
            pathU += dt / kPathPeriod;
            if (pathU >= 1.f) pathU -= 1.f;
            Vector3 eye, ahead;
            flightPath.getPoint(pathU, eye);
            flightPath.getPoint(std::fmod(pathU + 0.035f, 1.f), ahead);
            // Look mostly along the path, biased toward the cabin.
            Vector3 look;
            look.copy(ahead).multiplyScalar(0.6f).addScaledVector(
                    Vector3(padX, 6.f, kPadZ), 0.4f);
            camera.position.copy(eye);
            camera.lookAt(look);
        } else if (seqDir.empty()) {
            controls.update();
        }
        // (--seq drives the camera itself, at the END of the previous frame.
        //  OrbitControls::update() recomputes the position from its own
        //  spherical state, so letting it run here would undo the orbit.)
        // --climb R (shot mode): rise R m/frame, keeping the shot cam's look —
        // reproduces "flying up/out of the mist layer" in MOTION (the black-on-
        // exit regression). The layer is anchored at baseY; climbing lifts the
        // camera through and above it while the temporal history (froxel EMA,
        // TAA) tracks the transition.
        if (!shotPath.empty() && climbRate > 0.f) {
            camera.position.y += climbRate;
            // Lift the look target in lock-step so the PITCH stays constant while
            // the camera climbs — the horizon/sky band above the cloud layer stays
            // framed (the up-ray clear-sky pixels where the 1e30 cloud-fog leg
            // NaN'd). Without this the fixed target makes the pitch steepen into a
            // pure top-down view (all down-rays), hiding the sky regression.
            controls.target.y += climbRate;
            controls.update();
        }

        renderer.render(scene, camera);

        if (reproDumpRequest) {
            reproDumpRequest = false;
            const auto dir = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps";
            std::filesystem::create_directories(dir);
            char base[64];
            std::snprintf(base, sizeof(base), "fjord_repro_%02d", reproDumpIndex++);
            renderer.writeFramebuffer(dir / (std::string(base) + ".png"));
            char cmd[512];
            std::snprintf(cmd, sizeof(cmd),
                          "vulkan_fjord --shot %s_replay.png --frames 220 --time %.2f --wind %.1f "
                          "--fogscale %.2f%s --cam %.1f,%.1f,%.1f --look %.1f,%.1f,%.1f",
                          base, timeOfDay, windSpeed, fogScale,
                          cloudsOn ? "" : " --noclouds",
                          camera.position.x, camera.position.y, camera.position.z,
                          controls.target.x, controls.target.y, controls.target.z);
            std::ofstream side(dir / (std::string(base) + ".txt"));
            side << cmd << "\n";
            std::cout << "[repro] wrote " << base << ".png  replay: " << cmd << std::endl;
        }

        if (!seqDir.empty()) {
            // Written AFTER the render, so frame N's image is the pose set at
            // the top of frame N (below); the orbit for the NEXT frame is
            // applied here, closed-form in the frame index.
            if (shotFrame >= seqWarm) {
                char name[64];
                std::snprintf(name, sizeof(name), "f%02d.png", shotFrame - seqWarm);
                renderer.writeFramebuffer((std::filesystem::path(seqDir) / name).string());
            }
            ++shotFrame;
            if (shotFrame >= seqWarm + seqFrames) {
                if (seqDolly != 0.f)
                    std::printf("[seq] %d frames -> %s (dolly %.2f m/s, warm %d)\n",
                                seqFrames, seqDir.c_str(), double(seqDolly), seqWarm);
                else
                    std::printf("[seq] %d frames -> %s (orbit %.1f deg/s, warm %d)\n",
                                seqFrames, seqDir.c_str(), double(seqOrbit), seqWarm);
                std::exit(0);
            }
            const float tSeq = float(shotFrame) * (1.f / 60.f);
            if (seqDolly != 0.f) {
                // WALK. Eye and target translate together along the ground
                // heading the shot pose was already looking down, so the framing
                // is constant and the only thing changing is WHERE IN THE WORLD
                // the camera stands — which is precisely the axis a
                // camera-following weather field has to survive. Closed form in
                // the frame index, like the orbit.
                float hx = seqTarget0.x - seqEye0.x, hz = seqTarget0.z - seqEye0.z;
                const float hl = std::sqrt(hx * hx + hz * hz);
                if (hl > 1e-4f) { hx /= hl; hz /= hl; }
                const float s = seqDolly * tSeq;
                camera.position.set(seqEye0.x + hx * s, seqEye0.y, seqEye0.z + hz * s);
                controls.target.set(seqTarget0.x + hx * s, seqTarget0.y, seqTarget0.z + hz * s);
            } else {
                const float ang = seqOrbit * (3.14159265f / 180.f) * tSeq;
                const float dx = seqEye0.x - controls.target.x, dz = seqEye0.z - controls.target.z;
                const float ca = std::cos(ang), sa = std::sin(ang);
                camera.position.set(controls.target.x + dx * ca - dz * sa, seqEye0.y,
                                    controls.target.z + dx * sa + dz * ca);
            }
            camera.lookAt(controls.target);
        } else if (shotPath.empty()) {
            ui->render();
        } else if (++shotFrame >= shotFrames) {
            const auto path = capture::shotOutputPath(shotPath);
            renderer.writeFramebuffer(path);
            const auto t = renderer.lastFrameTimings();
            const auto ls = renderer.autoLodStats();
            std::cout << "wrote " << path.string() << " (" << fps << " fps)\n"
                      << "  gbuf " << t.rasterGbufMs << "  shade " << t.pathTraceMs
                      << "  denoise " << t.denoiseMs << "  taa " << t.taaMs
                      << "  froxel " << t.froxelMs << "  dof " << t.dofMs
                      << "  overlay " << t.overlayMs
                      << "  oceanFft " << t.oceanFftMs << "  oceanDisp " << t.oceanDisplaceMs
                      << "  oceanFoam " << t.oceanFoamMs << "  oceanBlas " << t.oceanBlasMs
                      << "  tlasRefit " << t.tlasRefitMs << "  dynGeom " << t.dynGeomRefitMs
                      << "  instExpand " << t.instanceExpandMs
                      << "  rtao " << t.rtaoMs
                      << "\n  gpuTotal " << t.gpuTotalMs << "  gpuPassSum " << t.gpuPassSumMs
                      << "  unbracketed " << (t.gpuTotalMs - t.gpuPassSumMs)
                      << "  cpuEnsure " << t.cpuEnsureSceneMs << "  cpuRecord " << t.cpuRecordMs
                      << "  cpuFrame " << t.cpuFrameMs
                      << "  tiles " << tiles->activeTiles()
                      << "  lod=[" << ls.entriesPerLevel[0] << "," << ls.entriesPerLevel[1] << ","
                      << ls.entriesPerLevel[2] << "," << ls.entriesPerLevel[3] << ","
                      << ls.entriesPerLevel[4] << "," << ls.entriesPerLevel[5] << "]"
                      << " chains=" << ls.chainsReady << std::endl;
            std::exit(0);
        }
    });

    return 0;
}
