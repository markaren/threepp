// FJORD — a procedural Norwegian fjord through a full day (Vulkan deferred showcase).
//
// Everything is generated in-engine — no asset downloads:
//   • terrain::TerrainGenerator — ridged multifractal + hydraulic/thermal erosion,
//     then a fjord channel is carved along an S-curve (deep water, a green shore
//     bench, steep walls) and a slope/altitude splat albedo is baked from the
//     carved heights (wet shore rock → heath → scree → rock → snow).
//   • Ocean — the FFT water (three Phillips cascades, foam, adaptive density
//     warp toward the camera) fills the channel at sea level y=0.
//   • vegetation::TreeGenerator — pine + birch prototypes instanced along the
//     shore bench (detailed cross-quad leaf cards near the cabin, cheap blob
//     canopies for the far bank), plus boulders and a GrassMesh meadow
//     (GPU wind compute + BLAS refit).
//   • A red cabin ("hytte") with white trim, glowing windows after sundown,
//     chimney smoke (ParticleSystem → Vulkan billboard overlay), a wooden dock
//     with a lantern, and a moored rowing boat.
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
// Headless:  vulkan_fjord --shot out.png [--frames N] [--time H] [--cam 0..3] [--cycle H_per_s]

#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/TextureLoader.hpp"
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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
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

    float smoothstepf(float e0, float e1, float x) {
        const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
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
                const float sFloor = smoothstepf(kChannelHalf - 90.f, kChannelHalf, d);// floor→bench edge
                const float sWall = smoothstepf(kBenchEnd, kWallEnd, d);               // bench→natural
                float hm = floorH + (bench - floorH) * sFloor;
                hm = hm + (std::max(natural, hm) - hm) * sWall;

                // Flatten the cabin pad.
                const float dp = std::sqrt((x - padX) * (x - padX) + (z - padZ) * (z - padZ));
                const float sPad = smoothstepf(24.f, 52.f, dp);
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
        const float padMask = smoothstepf(22.f, 56.f, dp);
        const float waterMask = smoothstepf(0.6f, 2.8f, baseH);
        if (padMask * waterMask < 1e-3f) return 0.f;
        const float amp = 0.22f + 1.35f * smoothstepf(0.10f, 0.45f, slopeMask);
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
            const float wet = smoothstepf(-0.6f, 0.35f, hC);
            col.lerp(cWetRock, wet);
        } else if (hC < 1.2f && slope < 0.38f) {
            // Gravel beach strip, noise-broken so it doesn't read as a stripe.
            const Vector3 cGravel(0.30f, 0.28f, 0.24f);
            col.copy(cGravel).multiplyScalar(0.5f + 0.5f * smoothstepf(0.35f, 1.1f, hC));
            const float breakup = smoothstepf(0.45f, 0.75f, fbm2(x * 0.038f, z * 0.038f));
            col.lerp(cGrass, std::max(breakup, smoothstepf(0.8f, 1.2f, hC) * 0.6f));
        } else {
            // Altitude bands (with a noisy snowline), then slope overrides.
            const float snowLine = 300.f + 70.f * wig;
            const float heathT = 0.65f * smoothstepf(110.f, 230.f, hC);// grass → heath (subtle)
            const float fellT = smoothstepf(200.f, 280.f, hC);         // heath → grey fell
            const float snowT = smoothstepf(snowLine - 28.f, snowLine + 28.f, hC);
            Vector3 ground = cGrass;
            ground.lerp(cHeath, heathT);
            ground.lerp(Vector3(0.30f, 0.31f, 0.26f), fellT);

            const float screeT = smoothstepf(0.24f, 0.42f, slope);
            const float rockT = smoothstepf(0.45f, 0.62f, slope);
            col.copy(ground).lerp(cScree, screeT).lerp(cRock, rockT);

            const float snowShed = 1.f - smoothstepf(0.50f, 0.68f, slope);
            col.lerp(cSnow, snowT * snowShed);

            // Dry rock band above the waterline on steep banks.
            const float shoreRock = (1.f - smoothstepf(0.8f, 6.f, hC)) * smoothstepf(0.18f, 0.35f, slope);
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
        s.daylight = smoothstepf(-0.10f, 0.05f, s.sunElev);

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

        [[nodiscard]] Vector3 radiance(const Vector3& dir, const CelestialState& cs) const {
            const Vector3 totalRayleigh(5.804542996261093e-6f, 1.3562911419845635e-5f, 3.0265902468824876e-5f);
            const Vector3 mieConst(1.8399918514433978e14f, 2.7798023919660528e14f, 4.0790479543861094e14f);

            const float sunE = sunIntensityEE(cs.sunDir.y);
            const float sunfade = 1.f - std::clamp(1.f - std::exp(cs.sunDir.y), 0.f, 1.f);
            const float rayleighCoeff = rayleigh - (1.f - sunfade);
            const Vector3 betaR(totalRayleigh.x * rayleighCoeff, totalRayleigh.y * rayleighCoeff,
                                totalRayleigh.z * rayleighCoeff);
            const float mieC = 0.434f * (0.2f * turbidity) * 1e-17f * mieCoefficient;
            const Vector3 betaM(mieConst.x * mieC, mieConst.y * mieC, mieConst.z * mieC);

            const float zenith = std::acos(std::max(0.f, dir.y));
            const float inv = 1.f / (std::cos(zenith) +
                                     0.15f * std::pow(93.885f - zenith * 180.f / kPi, -1.253f));
            const float sR = 8400.f * inv, sM = 1250.f * inv;
            const Vector3 fex(std::exp(-(betaR.x * sR + betaM.x * sM)),
                              std::exp(-(betaR.y * sR + betaM.y * sM)),
                              std::exp(-(betaR.z * sR + betaM.z * sM)));

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
            const float sundisk = smoothstepf(sunDiscCos, sunDiscCos + 0.00002f, cosTheta);
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
            const float preFade = 0.05f + 0.95f * smoothstepf(-0.25f, -0.02f, cs.sunDir.y);
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
        if (preset == 1) {// Norway spruce: short trunk, narrow crown almost to the ground
            tp.trunkHeight = 2.0f;
            tp.trunkRadius = 0.17f;
            tp.crownShape = vegetation::CrownShape::Cone;
            tp.crownRadiusX = tp.crownRadiusZ = 2.6f;
            tp.crownHeight = 10.f;
            tp.attractorCount = 700;
            tp.influenceDistance = 3.5f;
            tp.segmentLength = 0.45f;
            tp.maxIterations = 220;
            tp.tropism = -0.05f;
            tp.leafSize = 0.85f;
            tp.leavesPerCluster = 6;
            tp.leafDensity = 0.97f;
            tp.leafClumping = 0.25f;
            tp.leafColor = {0.15f, 0.33f, 0.11f};
        }
        if (preset == 2) {// birch — mute the pure-white preset bark
            tp.barkColor = {0.72f, 0.71f, 0.67f};
            tp.leafDensity = 0.95f;
            tp.leafClumping = 0.35f;
        }
        if (cheapBlob) {// far-bank silhouette trees: low-poly canopy puffs
            tp.leafStyle = vegetation::LeafStyle::Blob;
            tp.leavesPerCluster = 2;
            tp.leafSize = 1.15f;
            tp.attractorCount = 320;
            tp.radialSegments = 5;
            tp.leafColor = {0.15f, 0.33f, 0.11f};
        }

        vegetation::TreeGenerator gen(seed);
        gen.buildSkeleton(tp);

        TreeVariant v;
        v.trunkGeo = gen.makeTrunkGeometry(tp);
        v.leafGeo = gen.makeLeafGeometry(tp);

        auto bark = vegetation::makeBarkTextures(cheapBlob ? 128 : 256, seed, tp.barkColor);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);
        v.barkMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        v.barkMat->map = bark.first;
        v.barkMat->normalMap = bark.second;

        v.leafMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        if (cheapBlob) {
            // Slight brighten: the baked per-vertex canopy tint (0.6–1.1) multiplies this.
            // leafColor is an sRGB hint (TreeParams doc) — the card path bakes it into an
            // sRGB-tagged texture (decoded on sample), but material->color is LINEAR
            // working space. Without the conversion the blobs render ~4x too bright and
            // read "always lit" (glowing green even at night).
            v.leafMat->color = Color(tp.leafColor[0] * 1.15f, tp.leafColor[1] * 1.15f, tp.leafColor[2] * 1.15f)
                                       .convertSRGBToLinear();
            v.leafMat->vertexColors = true;// canopy tint gradient baked per-vertex
        } else {
            v.leafMat->map = vegetation::makeLeafClusterTexture(256, seed, tp.leafColor);
            v.leafMat->alphaTest = 0.5f;
            v.leafMat->side = Side::Double;
            v.leafMat->vertexColors = true;
        }
        return v;
    }

    // Merged grass blades → one GrassMesh (GPU wind + single BLAS refit).
    struct Blade {
        Vector3 pos;
        Vector3 scale;
        Quaternion yaw;
    };

    std::shared_ptr<BufferGeometry> makeGrassGeometry(const std::vector<Blade>& blades) {
        constexpr int seg = 4;
        constexpr float wBase = 0.05f;
        const Vector3 bottom{0.055f, 0.11f, 0.035f};
        const Vector3 top{0.19f, 0.30f, 0.10f};
        struct V {
            Vector3 p, n;
            float u, vy;
            Vector3 c;
        };
        std::vector<V> tmpl;
        std::vector<unsigned int> tidx;
        for (int i = 0; i <= seg; ++i) {
            const float t = static_cast<float>(i) / seg;
            const float w = wBase * (1.f - t);
            Vector3 c{bottom.x + (top.x - bottom.x) * t, bottom.y + (top.y - bottom.y) * t,
                      bottom.z + (top.z - bottom.z) * t};
            for (int s = 0; s < 2; ++s)
                tmpl.push_back({Vector3{(s == 0 ? -w : w), t, 0.f}, Vector3{0.f, 0.85f, 0.53f},
                                (s == 0 ? 0.f : 1.f), t, c});
        }
        for (int i = 0; i < seg; ++i) {
            const auto a = static_cast<unsigned int>(i * 2);
            tidx.insert(tidx.end(), {a, a + 1u, a + 2u, a + 1u, a + 3u, a + 2u});
        }

        std::vector<float> pos, nrm, uv, col, hfrac;
        std::vector<unsigned int> idx;
        pos.reserve(blades.size() * tmpl.size() * 3);
        Matrix4 m;
        for (const auto& bl : blades) {
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            m.compose(bl.pos, bl.yaw, bl.scale);
            for (const auto& tv : tmpl) {
                Vector3 p = tv.p;
                p.applyMatrix4(m);
                Vector3 n = tv.n;
                n.applyQuaternion(bl.yaw);
                n.normalize();
                pos.insert(pos.end(), {p.x, p.y, p.z});
                nrm.insert(nrm.end(), {n.x, n.y, n.z});
                uv.insert(uv.end(), {tv.u, tv.vy});
                col.insert(col.end(), {tv.c.x, tv.c.y, tv.c.z});
                hfrac.push_back(tv.vy);
            }
            for (unsigned int t : tidx) idx.push_back(base + t);
        }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        geo->setAttribute("heightFrac", FloatBufferAttribute::create(hfrac, 1));
        return geo;
    }

    // Low-poly faceted boulder (from the forest demo).
    std::shared_ptr<BufferGeometry> makeRock(unsigned int seed) {
        constexpr int latSegs = 5, lonSegs = 7;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-kPi, kPi);
        const float p1 = u(rng), p2 = u(rng), p3 = u(rng);
        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        for (int lat = 0; lat <= latSegs; ++lat) {
            const float theta = static_cast<float>(lat) / latSegs * kPi;
            const float sinT = std::sin(theta), cosT = std::cos(theta);
            for (int lon = 0; lon <= lonSegs; ++lon) {
                const float phi = static_cast<float>(lon) / lonSegs * kTau;
                const float nx = sinT * std::cos(phi), ny = cosT, nz = sinT * std::sin(phi);
                // Every phi-dependent term is gated by sinT so it vanishes at the
                // poles (theta = 0, PI). Ungated cos(3phi)/cos(5phi) give each pole
                // vertex a DIFFERENT radius, smearing the single pole point into a
                // fan of thin sliver triangles whose sub-pixel coverage toggles with
                // the TAA jitter and flickers uniformly every frame. Gated, the pole
                // vertices coincide → zero-area triangles that rasterize to nothing.
                float disp = 1.f + sinT * (0.30f * std::sin(2.f * phi + p1) + 0.24f * std::cos(3.f * phi + p2) +
                                           0.14f * std::cos(5.f * phi + 4.f * theta + p1)) +
                             0.22f * std::sin(3.f * theta + p3);
                disp = std::clamp(disp, 0.6f, 1.5f);
                pos.insert(pos.end(), {nx * disp, ny * disp, nz * disp});
                nrm.insert(nrm.end(), {nx, ny, nz});
                uv.insert(uv.end(), {static_cast<float>(lon) / lonSegs, static_cast<float>(lat) / latSegs});
            }
        }
        const int rowVerts = lonSegs + 1;
        for (int lat = 0; lat < latSegs; ++lat)
            for (int lon = 0; lon < lonSegs; ++lon) {
                const auto a = static_cast<unsigned int>(lat * rowVerts + lon);
                const auto b = static_cast<unsigned int>(a + rowVerts);
                idx.insert(idx.end(), {a, a + 1, b, a + 1, b + 1, b});
            }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        return geo;
    }

    // ═══════════════════════════ cabin / dock / boat ════════════════════════

    // Triangular gable prism: cross-section in XY (base spanX wide, apex `rise`
    // above the base) extruded along Z. Flat-shaded via non-indexed triangles.
    std::shared_ptr<BufferGeometry> makeGablePrism(float spanX, float rise, float lenZ) {
        const float hx = spanX * 0.5f, hz = lenZ * 0.5f;
        const Vector3 A(-hx, 0.f, -hz), B(hx, 0.f, -hz), C(0.f, rise, -hz);
        const Vector3 D(-hx, 0.f, hz), E(hx, 0.f, hz), F(0.f, rise, hz);
        std::vector<Vector3> tris = {
                A, C, B,      // back gable
                D, E, F,      // front gable
                A, F, C, A, D, F,// left slope (as seen from -x)
                B, C, F, B, F, E // right slope
        };
        std::vector<float> pos, nrm;
        for (size_t i = 0; i < tris.size(); i += 3) {
            Vector3 e1, e2, n;
            e1.subVectors(tris[i + 1], tris[i]);
            e2.subVectors(tris[i + 2], tris[i]);
            n.crossVectors(e1, e2).normalize();
            for (int k = 0; k < 3; ++k) {
                pos.insert(pos.end(), {tris[i + k].x, tris[i + k].y, tris[i + k].z});
                nrm.insert(nrm.end(), {n.x, n.y, n.z});
            }
        }
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        return geo;
    }

    // Lofted clinker-style rowing boat hull, origin at keel bottom, bow +Z.
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
        Vector3 chimneyTipLocal;
    };

    Cabin makeCabin() {
        Cabin cabin;
        cabin.group = Group::create();
        auto& g = *cabin.group;

        auto red = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.45f, 0.085f, 0.06f)).roughness(0.78f).metalness(0.f));
        auto trim = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.92f, 0.90f, 0.85f)).roughness(0.55f).metalness(0.f));
        auto roof = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.085f, 0.09f, 0.10f)).roughness(0.85f).metalness(0.f));
        auto stone = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.42f, 0.41f, 0.39f)).roughness(0.95f).metalness(0.f));
        cabin.windowMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}
                        .color(Color(0.05f, 0.06f, 0.07f))
                        .roughness(0.15f)
                        .metalness(0.f)
                        .emissive(Color(1.0f, 0.62f, 0.28f))
                        .emissiveIntensity(0.f));

        // Body: 4.6 m wide (X) × 7.2 m long (Z), walls 2.5 m, ridge along Z.
        constexpr float W = 4.6f, D = 7.2f, H = 2.5f, RISE = 1.5f;

        auto plinth = Mesh::create(BoxGeometry::create(W + 0.3f, 0.6f, D + 0.3f), stone);
        plinth->position.y = 0.3f;
        g.add(plinth);

        auto body = Mesh::create(BoxGeometry::create(W, H, D), red);
        body->position.y = 0.6f + H * 0.5f;
        g.add(body);

        auto attic = Mesh::create(makeGablePrism(W, RISE, D), red);
        attic->position.y = 0.6f + H;
        g.add(attic);

        // Roof slabs with overhang, sloping down toward ±X.
        const float slope = std::sqrt(RISE * RISE + (W * 0.5f) * (W * 0.5f));
        const float ang = std::atan2(RISE, W * 0.5f);
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            auto slab = Mesh::create(BoxGeometry::create(slope + 0.45f, 0.10f, D + 0.6f), roof);
            slab->rotation.z = static_cast<float>(sgn) * ang;
            slab->position.set(static_cast<float>(-sgn) * (W * 0.25f + 0.08f),
                               0.6f + H + RISE * 0.5f + 0.10f, 0.f);
            g.add(slab);
        }

        // Chimney near the ridge.
        auto chimney = Mesh::create(BoxGeometry::create(0.55f, 1.6f, 0.55f), stone);
        chimney->position.set(0.f, 0.6f + H + RISE + 0.35f, 1.4f);
        g.add(chimney);
        cabin.chimneyTipLocal.set(0.f, 0.6f + H + RISE + 1.2f, 1.4f);

        // Window helper: white frame + crossbars + warm pane.
        auto addWindow = [&](float w, float h, const Vector3& p, float rotY) {
            auto win = Group::create();
            auto frame = Mesh::create(BoxGeometry::create(w + 0.18f, h + 0.18f, 0.10f), trim);
            win->add(frame);
            // Pane sits PROUD of the solid frame box so the glow is visible.
            auto pane = Mesh::create(BoxGeometry::create(w, h, 0.06f), cabin.windowMat);
            pane->position.z = 0.055f;
            win->add(pane);
            auto barV = Mesh::create(BoxGeometry::create(0.05f, h, 0.045f), trim);
            barV->position.z = 0.10f;
            win->add(barV);
            auto barH = Mesh::create(BoxGeometry::create(w, 0.05f, 0.045f), trim);
            barH->position.z = 0.10f;
            win->add(barH);
            win->position.copy(p);
            win->rotation.y = rotY;
            g.add(win);
        };

        // Front (water-facing, -X): two windows.
        addWindow(1.0f, 1.15f, Vector3(-W * 0.5f - 0.02f, 1.85f, -1.8f), -kPi / 2.f);
        addWindow(1.0f, 1.15f, Vector3(-W * 0.5f - 0.02f, 1.85f, 1.4f), -kPi / 2.f);
        // Gable ends: one each.
        addWindow(0.9f, 1.0f, Vector3(0.9f, 1.85f, D * 0.5f + 0.02f), 0.f);
        addWindow(0.8f, 0.8f, Vector3(0.f, 3.4f, -D * 0.5f - 0.02f), kPi);

        // Door (south gable, toward the dock path) + white frame + step.
        auto doorFrame = Mesh::create(BoxGeometry::create(1.14f, 2.14f, 0.10f), trim);
        doorFrame->position.set(-1.1f, 0.6f + 1.05f, D * 0.5f + 0.02f);
        g.add(doorFrame);
        auto door = Mesh::create(BoxGeometry::create(0.98f, 2.0f, 0.08f), MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.10f, 0.16f, 0.18f)).roughness(0.6f).metalness(0.f)));
        door->position.set(-1.1f, 0.6f + 1.0f, D * 0.5f + 0.055f);
        g.add(door);
        auto step = Mesh::create(BoxGeometry::create(1.4f, 0.25f, 0.8f), stone);
        step->position.set(-1.1f, 0.12f, D * 0.5f + 0.5f);
        g.add(step);

        // Corner boards (white trim verticals).
        for (int cx = -1; cx <= 1; cx += 2)
            for (int cz = -1; cz <= 1; cz += 2) {
                auto corner = Mesh::create(BoxGeometry::create(0.14f, H, 0.14f), trim);
                corner->position.set(static_cast<float>(cx) * (W * 0.5f - 0.02f), 0.6f + H * 0.5f,
                                     static_cast<float>(cz) * (D * 0.5f - 0.02f));
                g.add(corner);
            }

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
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--time") == 0 && i + 1 < argc) startTime = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--cam") == 0 && i + 1 < argc) shotCam = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cycle") == 0 && i + 1 < argc) startCycle = static_cast<float>(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--fly") == 0) startFly = true;
    }

    Canvas canvas("threepp - FJORD (Vulkan deferred)", {{"vsync", false}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.setAutoExposure(true);
    // Cap the upper end well below the default +3 EV: the eye should NOT
    // re-expose midnight back to daylight — night must stay dark.
    renderer.setAutoExposureRange(-2.5f, 1.5f);
    renderer.setAutoExposureSpeed(shotPath.empty() ? 2.0f : 12.0f);
    renderer.setGbufferMsaa(2);// leaf canopies + grass edges
    renderer.setRenderScale(0.85f);
    renderer.setSunAngularRadius(0.6f);// soft RT sun shadows
    renderer.setVolumetricFog(true);   // god rays through the fjord walls
    renderer.setFogAnisotropy(0.55f);
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
    auto tiles = terrain::TileTerrain::create(prov, tileOpts);
    tiles->name = "fjord_terrain";
    scene.add(tiles);
    const auto terrainH = [&tiles](float x, float z) { return tiles->heightAt(x, z); };
    {
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "[fjord] terrain (erode+carve+" << tiles->activeTiles() << " root tiles): "
                  << std::chrono::duration<float>(t1 - t0).count() << " s" << std::endl;
    }

    // ── water ───────────────────────────────────────────────────────────────
    Ocean::Options oo;
    oo.size = 3200.f;
    oo.resolution = 512;
    oo.windSpeed = 4.5f;
    oo.windTheta = 2.1f;
    oo.choppiness = 0.5f;
    oo.tileSize1 = 90.f;
    oo.tileSize2 = 7.f;
    auto ocean = Ocean::create(oo);
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
        std::vector<TreeVariant> nearVars{
                makeTreeVariant(1, 301u, false),// pine
                makeTreeVariant(1, 502u, false),// pine (alt)
                makeTreeVariant(2, 404u, false),// birch accent
        };
        std::vector<TreeVariant> farVars{
                makeTreeVariant(1, 611u, true),
                makeTreeVariant(1, 733u, true),
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
        std::vector<std::shared_ptr<BufferGeometry>> rgeos{makeRock(1u), makeRock(2u), makeRock(3u)};
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

    // ── meadow grass (GPU wind GrassMesh) ───────────────────────────────────
    std::shared_ptr<GrassMesh> grass;
    {
        std::vector<Blade> blades;
        blades.reserve(15000);
        std::mt19937 rng(7u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const Vector3 up{0.f, 1.f, 0.f};
        while (blades.size() < 15000) {
            const float ang = u01(rng) * kTau;
            const float rr = std::sqrt(u01(rng)) * 46.f;
            const float x = padX + std::cos(ang) * rr;
            const float z = kPadZ + std::sin(ang) * rr;
            const float h = terrainH(x, z);
            if (h < 0.6f) continue;// not into the water
            Blade bl;
            bl.pos.set(x, h - 0.04f, z);
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
        grass = GrassMesh::create(makeGrassGeometry(blades), grassMat);
        grass->name = "meadow";
        grass->params.windDir = Vector2(0.8f, 0.6f);
        grass->params.windStrength = 0.16f;
        scene.add(grass);
    }

    // ── cabin, dock, lantern, boat ──────────────────────────────────────────
    Cabin cabin = makeCabin();
    constexpr float kCabinScale = 1.35f;
    cabin.group->position.set(padX, kPadHeight - 0.1f, kPadZ);
    cabin.group->scale.set(kCabinScale, kCabinScale, kCabinScale);
    scene.add(cabin.group);

    // A second, distant cabin across the fjord — its lit windows carry across
    // the water at night.
    std::shared_ptr<MeshStandardMaterial> farWindowMat;
    {
        auto other = makeCabin();
        const float z2 = -430.f;
        const float x2 = channelCenterX(z2) - 205.f;
        other.group->position.set(x2, terrainH(x2, z2) - 0.25f, z2);
        other.group->rotation.y = kPi;
        other.group->name = "cabin_far";
        scene.add(other.group);
        farWindowMat = other.windowMat;
    }

    // Waterline along -X from the pad (dock goes there).
    float xWater = padX;
    {
        for (float x = padX; x > padX - 300.f; x -= 1.f) {
            if (field.sampleBilinear(x, kPadZ) < 0.f) {
                xWater = x;
                break;
            }
        }
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
    ParticleSystem smoke;
    {
        auto& s = smoke.settings();
        s.makeDefault();
        s.positionStyle = ParticleSystem::Type::BOX;
        Vector3 tip = cabin.chimneyTipLocal;
        tip.multiplyScalar(kCabinScale).add(cabin.group->position);
        s.positionBase = tip;
        s.positionSpread = {0.15f, 0.05f, 0.15f};
        s.velocityStyle = ParticleSystem::Type::BOX;
        s.velocityBase = {0.55f, 1.15f, 0.30f};
        s.velocitySpread = {0.35f, 0.35f, 0.35f};
        s.accelerationBase = {0.10f, 0.10f, 0.05f};
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
        }
        controls.update();
    };
    if (!shotPath.empty()) applyShotCam(shotCam);

    // ── state + UI ──────────────────────────────────────────────────────────
    float cycleSpeed = startCycle;// hours per second (0 = paused)
    float windSpeed = 4.5f;
    float fogScale = 1.f;
    float uiRenderScale = 0.85f;
    bool perfDirty = false;
    bool volumetrics = true;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({340, 0}, ImGuiCond_FirstUseEver);
        ImGui::Begin("FJORD");
        ImGui::Text("FPS %.1f", fps);
        {
            const auto t = renderer.lastFrameTimings();
            ImGui::TextDisabled("gbuf %.2f shade %.2f denoise %.2f taa %.2f",
                                t.rasterGbufMs, t.shadeBMs, t.denoiseMs, t.taaMs);
            ImGui::TextDisabled("terrain tiles %d (baking %d)", tiles->activeTiles(), tiles->pendingBakes());
            // Auto-LOD engagement: entries drawing a simplified level vs full
            // detail. The win is geometry-bound phases (flight through the
            // fjord — press C); near-field orbits keep everything at L0 by
            // design (sub-pixel error threshold), so gbuf ms + this line are
            // where the effect is visible, not necessarily headline FPS.
            const auto al = renderer.autoLodStats();
            ImGui::TextDisabled("lod L0 %u | simplified %u (chains %u, %.0f MB)",
                                al.entriesPerLevel[0],
                                al.entriesPerLevel[1] + al.entriesPerLevel[2] + al.entriesPerLevel[3] +
                                        al.entriesPerLevel[4] + al.entriesPerLevel[5],
                                al.chainsReady,
                                static_cast<double>(al.indexBytes + al.blasBytes) / (1024.0 * 1024.0));
        }
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
        ImGui::Checkbox("volumetric god rays", &volumetrics);
        ImGui::SeparatorText("Camera / perf");
        ImGui::Checkbox("cinematic flight (C)", &cinematic);
        if (ImGui::SliderFloat("render scale", &uiRenderScale, 0.4f, 1.f, "%.2f")) perfDirty = true;
        {
            bool lodOn = renderer.autoLod();
            if (ImGui::Checkbox("auto LOD", &lodOn)) renderer.setAutoLod(lodOn);
        }
        ImGui::TextDisabled("drag=orbit scroll=zoom SPACE=pause");
        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    KeyAdapter keyAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::C) cinematic = !cinematic;
        if (evt.key == Key::SPACE) cycleSpeed = cycleSpeed > 0.f ? 0.f : 0.08f;
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

        if (perfDirty) {
            renderer.setRenderScale(uiRenderScale);
            perfDirty = false;
        }

        // ── advance the day ──
        timeOfDay += cycleSpeed * dt;
        if (timeOfDay >= 24.f) timeOfDay -= 24.f;
        const CelestialState cs = celestialAt(timeOfDay);
        applySky(cs, false);

        // Sun light: direction + Preetham transmittance tint.
        {
            sun->position.copy(cs.sunDir).multiplyScalar(1500.f);
            const float zen = std::acos(std::max(0.f, cs.sunDir.y));
            const float inv = 1.f / (std::cos(zen) + 0.15f * std::pow(93.885f - zen * 180.f / kPi, -1.253f));
            const float sR = 8400.f * inv, sM = 1250.f * inv;
            const Vector3 betaR(5.8e-6f * 2.2f, 1.356e-5f * 2.2f, 3.026e-5f * 2.2f);
            const float mieC = 0.434f * (0.2f * sky.turbidity) * 1e-17f * sky.mieCoefficient;
            Color tint(std::exp(-(betaR.x * sR + 1.84e14f * mieC * sM)),
                       std::exp(-(betaR.y * sR + 2.78e14f * mieC * sM)),
                       std::exp(-(betaR.z * sR + 4.08e14f * mieC * sM)));
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
            sun->intensity = 3.6f * smoothstepf(-0.03f, 0.12f, cs.sunElev) * (0.35f + 0.65f * maxC);
        }
        // Moon light at night.
        {
            moon->position.copy(cs.moonDir).multiplyScalar(1500.f);
            moon->intensity = 0.38f * cs.moonUp * (1.f - cs.daylight);
        }

        // Stars fade in only once the sun is well below the horizon (-3°→-10°).
        renderer.setDeferredStarfield(1.1f * (1.f - smoothstepf(-0.18f, -0.05f, cs.sunElev)));

        // Fog: colour follows the sky horizon; dawn mist peaks ~06:20.
        {
            const float mist = std::exp(-std::pow((timeOfDay - 6.3f) / 1.5f, 2.f));
            const float density = (0.00035f + 0.0009f * mist) * fogScale;
            if (density > 1e-6f) {
                scene.fog = FogExp2(horizonColor(sky, cs), density);
            } else {
                scene.fog.reset();
            }
            renderer.setVolumetricFog(volumetrics);
        }

        // Window / lantern glow after sundown. emissiveIntensity is a plain
        // field — bump the material version so the renderer re-uploads it.
        {
            const float glow = 1.f - smoothstepf(-0.07f, 0.02f, cs.sunElev);
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
            }
        }

        // Terrain LOD follows the camera (async tile bakes, budgeted swaps).
        tiles->update(camera.position);

        // Water, grass, smoke, boat.
        ocean->setWind(windSpeed, 2.1f);
        // Pack water vertices toward the camera at eye level; from high up the
        // density rings show, so relax toward uniform with altitude.
        const float warpCoef = camera.position.y < 60.f
                                       ? 0.12f
                                       : std::min(1.f, 0.12f + (camera.position.y - 60.f) / 200.f);
        ocean->warpToward(camera.position.x, camera.position.z, warpCoef);
        grass->params.time = tElapsed;
        grass->params.windStrength = 0.10f + 0.02f * windSpeed;
        smoke.update(dt);
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
        } else {
            controls.update();
        }

        renderer.render(scene, camera);

        if (shotPath.empty()) {
            ui.render();
        } else if (++shotFrame >= shotFrames) {
            const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            renderer.writeFramebuffer(path);
            const auto t = renderer.lastFrameTimings();
            std::cout << "wrote " << path.string() << " (" << fps << " fps)\n"
                      << "  gbuf " << t.rasterGbufMs << "  shade " << t.shadeBMs
                      << "  denoise " << t.denoiseMs << "  taa " << t.taaMs
                      << "  cpuEnsure " << t.cpuEnsureSceneMs << "  cpuRecord " << t.cpuRecordMs
                      << "  cpuFrame " << t.cpuFrameMs
                      << "  tiles " << tiles->activeTiles() << std::endl;
            std::exit(0);
        }
    });

    return 0;
}
