// FJORD FIRE — the chimney's thread of smoke and the camp fire in the ring.
//
// A C++ port of what read right in the cabin-in-the-woods film (its
// cabin/life/fire.py, judged "looks very good"): sprites, not a density
// volume. A volume's voxels set a minimum width at the flue, so the plume
// left the chimney a metre wide ("way too broad at the beginning"), and one
// volume costs about a millisecond. Here:
//
//   chimney  two ParticleFields sharing one emitter (same seed, same puffs):
//            a grey-blue alpha-over BODY that looks the same in sun and
//            shade, and a lit additive layer the sun's phase lobe flares
//            when the lens looks toward it. Small puffs leave the pot, rise,
//            meander more the higher they get, and lean into the breeze.
//   fire     an ember bed (emissive, NOT a light: setEmissiveCastsLight off),
//            charred billets over it, tongues, a hot heart, sparks now and
//            then, one PointLight with a band-limited flicker, thin blue
//            wood smoke.
//
// Every field is a closed form in scene time (setEmitterTime), so a still at
// any t is exact. Sprites composite after the tone map: their brightness is
// absolute; the *Intensity constants below were set by eye at the fjord's
// golden-hour exposure.

#ifndef THREEPP_FJORD_FIRE_HPP
#define THREEPP_FJORD_FIRE_HPP

#include "threepp/geometries/CircleGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace fjord {

    class FjordFire {
    public:
        // Sprite intensities (absolute, outside the tone map).
        static constexpr float kFlame = 0.24f, kHeart = 0.15f, kSpark = 5.0f;
        static constexpr float kWoodSmoke = 0.060f, kChimneyBody = 0.42f, kChimneySun = 0.0009f;

        FjordFire(threepp::Scene& scene, threepp::VulkanRenderer& renderer, const threepp::Vector3& chimneyTop,
                  const threepp::Vector3& pit, const threepp::Vector3& wind)
            : pit_(pit), wind_(wind) {
            buildChimney(scene, chimneyTop);
            buildBed(scene, renderer);
            buildFlames(scene);
            buildWoodSmoke(scene);
            light_ = threepp::PointLight::create(threepp::Color(1.0f, 0.47f, 0.15f), 0.f, 10.f, 2.f);
            light_->castShadow = true;
            light_->radius = 0.11f;// soft shadows from the stones
            light_->name = "fire_light";
            scene.add(light_);
            update(0.f, 0.f);
        }

        // level 0 = embers, 1 = a good blaze.
        float level = 0.45f;
        // 0 by day .. 1 at night. A camp fire lights the stones round it hard
        // only in the dark: at full strength in daylight its light, a metre
        // away, outshone the sun twice over and the ring stones went white.
        float darkness = 0.f;

        void update(float t, float dt) {
            using namespace threepp;
            const float lv = level;
            const float fl = flicker(t);
            const float gu = gust(t);
            const float floorY = pit_.y - 0.05f;

            auto e = flames_->emitter();
            const float spread = 0.09f + 0.09f * lv;
            e.spawnCenter.set(pit_.x, floorY + 0.10f + 0.04f * lv, pit_.z);
            e.spawnHalfExtent.set(spread, 0.04f, spread);
            e.velocity.set(wind_.x * 0.25f * gu, 0.45f + 1.0f * lv, wind_.z * 0.25f * gu);
            e.speedSpread = 0.10f + 0.16f * lv;
            e.accel.set(wind_.x * 0.8f * gu, 0.6f, wind_.z * 0.8f * gu);
            e.driftAmplitude = 0.05f + 0.03f * lv;
            e.driftFrequency = 1.6f;
            e.driftScale = 0.5f;
            e.lifetime = 0.62f;
            e.lifetimeJitter = 0.5f;
            e.size = 0.045f + 0.04f * lv;
            e.sizeJitter = 0.45f;
            e.seed = 4u;
            flames_->setEmitter(e);
            flames_->billboardRepr().intensity = kFlame * (0.25f + 0.75f * lv) * (0.9f + 0.1f * fl);

            e = heart_->emitter();
            e.spawnCenter.set(pit_.x, floorY + 0.09f, pit_.z);
            e.spawnHalfExtent.set(0.05f + 0.06f * lv, 0.03f, 0.05f + 0.06f * lv);
            e.velocity.set(0.f, 0.35f + 0.55f * lv, 0.f);
            e.speedSpread = 0.10f;
            e.accel.set(0.f, 0.4f, 0.f);
            e.driftAmplitude = 0.035f;
            e.driftFrequency = 2.1f;
            e.driftScale = 0.4f;
            e.lifetime = 0.38f;
            e.lifetimeJitter = 0.4f;
            e.size = 0.030f + 0.028f * lv;
            e.sizeJitter = 0.4f;
            e.seed = 9u;
            heart_->setEmitter(e);
            heart_->billboardRepr().intensity = kHeart * (0.20f + 0.80f * lv);
            sparks_->billboardRepr().intensity = kSpark * (0.35f + 0.65f * lv);

            bedMat_->emissiveIntensity = (3.0f + 5.0f * lv) * (0.94f + 0.06f * std::sin(6.2832f * 0.37f * t));
            bedMat_->needsUpdate();

            light_->intensity = (2.5f + 6.f * lv) * fl * (0.06f + 0.94f * darkness);
            light_->position.set(pit_.x + 0.012f * std::sin(6.2832f * 0.9f * t),
                                 pit_.y + 0.30f + 0.25f * (0.12f + 0.88f * lv),
                                 pit_.z + 0.012f * std::sin(6.2832f * 1.2f * t + 2.1f));

            e = woodSmoke_->emitter();
            e.spawnCenter.set(pit_.x, floorY + 0.30f + 0.25f * lv, pit_.z);
            e.spawnHalfExtent.set(0.07f, 0.04f, 0.07f);
            e.velocity.set(0.f, 0.50f + 0.22f * lv, 0.f);
            e.wind.set(wind_.x * 0.75f * gu, 0.f, wind_.z * 0.75f * gu);
            e.speedSpread = 0.02f;
            e.accel.set(wind_.x * 0.30f * gu, -0.03f, wind_.z * 0.30f * gu);
            e.driftAmplitude = 0.42f;
            e.driftFrequency = 0.16f;
            e.driftGrowth = 1.f;
            e.driftScale = 1.3f;
            e.lifetime = 8.5f;
            e.lifetimeJitter = 0.15f;
            e.size = 0.045f;
            e.sizeJitter = 0.4f;
            e.seed = 57u;
            woodSmoke_->setEmitter(e);
            woodSmoke_->billboardRepr().opacity = (0.026f + 0.018f * lv) * woodK_;

            // The breeze breathes: both chimney layers get the same wind (the same puffs).
            for (auto& f : chimney_) {
                auto ce = f->emitter();
                ce.wind.set(wind_.x * 0.8f * gu, 0.f, wind_.z * 0.8f * gu);
                ce.accel.set(wind_.x * 0.30f * gu, -0.030f, wind_.z * 0.30f * gu);
                f->setEmitter(ce);
            }
            for (auto* f : {flames_.get(), heart_.get(), sparks_.get(), woodSmoke_.get()}) f->setEmitterTime(t, dt);
            for (auto& f : chimney_) f->setEmitterTime(t, dt);
        }

    private:
        // Band-limited: every term under 7 Hz. About +-14 %.
        static float flicker(float t) {
            return 1.f + 0.075f * std::sin(6.2832f * 1.3f * t) + 0.045f * std::sin(6.2832f * 2.9f * t + 1.7f) +
                   0.022f * std::sin(6.2832f * 6.1f * t + 0.4f);
        }
        // The breeze breathing, 0.75..1.25, slow: the plume swings like a reed.
        static float gust(float t) {
            return 1.f + 0.17f * std::sin(6.2832f * 0.083f * t + 0.5f) + 0.08f * std::sin(6.2832f * 0.21f * t + 1.9f);
        }

        static std::shared_ptr<threepp::ParticleField> field(std::uint32_t capacity, float radius) {
            threepp::ParticleField::Config c;
            c.capacity = capacity;
            c.ownership = threepp::ParticleField::Ownership::Renderer;
            c.wSemantic = threepp::ParticleField::WSemantic::Radius;
            c.uniformRadius = radius;
            auto f = threepp::ParticleField::create(c);
            f->frustumCulled = false;
            return f;
        }

        void buildChimney(threepp::Scene& scene, const threepp::Vector3& top) {
            using namespace threepp;
            constexpr std::uint32_t n = 400;
            const float k = 800.f / static_cast<float>(n);// fewer puffs, each a little denser
            const float wl = std::max(1e-3f, std::sqrt(wind_.x * wind_.x + wind_.z * wind_.z));
            const float wx = wind_.x / wl, wz = wind_.z / wl;
            for (int layer = 0; layer < 2; ++layer) {
                auto f = field(n, 0.045f);
                auto e = f->emitter();
                e.spawnCenter.set(top.x + 0.10f * wx, top.y - 0.02f, top.z + 0.10f * wz);
                e.spawnHalfExtent.set(0.07f, 0.025f, 0.07f);
                e.velocity.set(0.10f * wx, 0.55f, 0.10f * wz);
                e.wind.set(wind_.x * 0.8f, 0.f, wind_.z * 0.8f);
                e.speedSpread = 0.028f;
                e.accel.set(wind_.x * 0.30f, -0.030f, wind_.z * 0.30f);
                e.driftAmplitude = 0.55f;
                e.driftFrequency = 0.13f;
                e.driftGrowth = 1.f;// tight at the pot, loosening with height
                e.driftScale = 2.2f;// a travelling meander, not per-puff jitter
                e.lifetime = 11.f;
                e.lifetimeJitter = 0.15f;
                e.size = 0.045f;
                e.sizeJitter = 0.25f;
                e.seed = 101u;// both layers are the same puffs
                f->setEmitter(e);
                if (layer == 0) {
                    f->setBillboardRepr(Color(0.50f, 0.56f, 0.66f), Color(0.42f, 0.47f, 0.56f), kChimneyBody, 3.f);
                    auto& b = f->billboardRepr();
                    b.alphaOver = true;
                    b.opacity = 0.040f * k;
                    b.fadePower = 1.2f;
                } else {
                    f->setBillboardRepr(Color(0.22f, 0.22f, 0.22f), Color(0.90f, 0.90f, 0.90f), kChimneySun * k, 3.f);
                    auto& b = f->billboardRepr();
                    b.lit = true;
                    b.litAmbient = 0.f;
                    b.litPhaseG = 0.40f;
                    b.fadePower = 1.7f;
                }
                auto& b = f->billboardRepr();
                b.softness = 1.f;
                b.coreWeight = 0.f;
                b.sizeTaper = 0.f;
                b.brightJitter = 0.f;
                f->name = layer == 0 ? "chimney_smoke_body" : "chimney_smoke_sun";
                scene.add(f);
                chimney_.push_back(f);
            }
        }

        void buildBed(threepp::Scene& scene, threepp::VulkanRenderer& renderer) {
            using namespace threepp;
            // A bed of coals: charcoal lumps, some live, glow in the gaps.
            constexpr int S = 128;
            std::vector<unsigned char> alb(S * S * 4), emi(S * S * 4);
            std::mt19937 rng(3u);
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            std::vector<std::array<float, 3>> pts(120);
            for (auto& p : pts) p = {u01(rng), u01(rng), u01(rng) < 0.42f ? 0.3f + 0.7f * u01(rng) : 0.04f * u01(rng)};
            for (int y = 0; y < S; ++y)
                for (int x = 0; x < S; ++x) {
                    const float fx = (x + 0.5f) / S, fy = (y + 0.5f) / S;
                    float d1 = 9.f, d2 = 9.f;
                    float heat = 0.f;
                    for (const auto& p : pts) {
                        float dx = std::abs(fx - p[0]), dy = std::abs(fy - p[1]);
                        dx = std::min(dx, 1.f - dx);
                        dy = std::min(dy, 1.f - dy);
                        const float d = std::sqrt(dx * dx + dy * dy);
                        if (d < d1) {
                            d2 = d1;
                            d1 = d;
                            heat = p[2];
                        } else
                            d2 = std::min(d2, d);
                    }
                    const float gap = std::pow(std::clamp(1.f - (d2 - d1) / 0.022f, 0.f, 1.f), 1.5f);
                    const float centre = std::clamp(1.2f - 2.1f * std::sqrt((fx - .5f) * (fx - .5f) + (fy - .5f) * (fy - .5f)), 0.f, 1.f);
                    const float glow = std::clamp(std::max(heat, 0.85f * gap) * (0.25f + 0.75f * centre), 0.f, 1.f);
                    const size_t o = (static_cast<size_t>(y) * S + x) * 4;
                    const auto b = static_cast<unsigned char>(std::clamp(0.03f + 0.03f * u01(rng), 0.f, 1.f) * 255.f);
                    alb[o] = alb[o + 1] = alb[o + 2] = b;
                    alb[o + 3] = 255;
                    emi[o] = static_cast<unsigned char>(std::pow(glow, 0.8f) * 255.f);
                    emi[o + 1] = static_cast<unsigned char>(std::pow(glow, 1.5f) * 0.55f * 255.f);
                    emi[o + 2] = static_cast<unsigned char>(std::pow(glow, 2.8f) * 0.10f * 255.f);
                    emi[o + 3] = 255;
                }
            auto mk = [&](std::vector<unsigned char> px) {
                auto t = DataTexture::create(ImageData{std::move(px)}, S, S);
                t->colorSpace = ColorSpace::sRGB;
                t->generateMipmaps = true;
                t->minFilter = Filter::LinearMipmapLinear;
                t->needsUpdate();
                return t;
            };
            bedMat_ = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::white).roughness(1.f));
            bedMat_->map = mk(std::move(alb));
            bedMat_->emissiveMap = mk(std::move(emi));
            bedMat_->emissive = Color(1.0f, 0.44f, 0.10f);
            bedMat_->emissiveIntensity = 4.f;
            auto bed = Mesh::create(CircleGeometry::create(0.30f, 32), bedMat_);
            bed->rotation.x = -1.5708f;
            bed->position.set(pit_.x, pit_.y + 0.035f, pit_.z);// a centimetre clear of the ring's ash
            bed->name = "fire_ember_bed";
            scene.add(bed);
            renderer.setEmissiveCastsLight(*bed, false);// glow only: an emitter would be a lamp

            auto charMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0.0012f, 0.0010f, 0.0009f)).roughness(0.95f));
            charMat->emissive = Color(1.0f, 0.30f, 0.05f);
            charMat->emissiveIntensity = 0.03f;
            // azimuth, lean from the vertical (deg), length, thickness, distance of the middle from the centre
            const float billets[5][5] = {{0.4f, 50.f, 0.46f, 0.030f, 0.10f}, {2.3f, 61.f, 0.40f, 0.026f, 0.11f},
                                         {4.1f, 55.f, 0.50f, 0.033f, 0.12f}, {5.3f, 72.f, 0.34f, 0.022f, 0.13f},
                                         {3.2f, 80.f, 0.30f, 0.020f, 0.15f}};
            for (const auto& bl : billets) {
                auto log = Mesh::create(CylinderGeometry::create(bl[3] * 0.85f, bl[3], bl[2], 7), charMat);
                const float lean = bl[1] * 0.0174533f;
                log->position.set(pit_.x + bl[4] * std::cos(bl[0]), pit_.y + 0.035f + 0.5f * bl[2] * std::cos(lean) + bl[3],
                                  pit_.z + bl[4] * std::sin(bl[0]));
                log->rotation.set(0.f, -bl[0], lean);
                log->name = "fire_billet";
                scene.add(log);
                renderer.setEmissiveCastsLight(*log, false);
            }
        }

        void buildFlames(threepp::Scene& scene) {
            using namespace threepp;
            flames_ = field(72, 0.075f);
            flames_->setBillboardRepr(Color(1.0f, 0.56f, 0.16f), Color(0.80f, 0.10f, 0.01f), kFlame, 1.f);
            {
                auto& b = flames_->billboardRepr();
                b.softness = 0.85f;
                b.coreWeight = 0.f;
                b.fadePower = 1.5f;
                b.sizeTaper = 0.80f;
                b.brightJitter = 0.4f;
                b.stretchSeconds = 0.11f;
                b.stretchMax = 3.2f;
                b.glow = 0.40f;
                b.glowThreshold = 0.f;
            }
            flames_->name = "fire_flames";
            scene.add(flames_);

            heart_ = field(36, 0.05f);
            heart_->setBillboardRepr(Color(1.0f, 0.80f, 0.42f), Color(1.0f, 0.42f, 0.08f), kHeart, 1.f);
            {
                auto& b = heart_->billboardRepr();
                b.softness = 0.8f;
                b.coreWeight = 0.f;
                b.fadePower = 1.f;
                b.sizeTaper = 0.6f;
                b.brightJitter = 0.3f;
                b.stretchSeconds = 0.08f;
                b.stretchMax = 2.5f;
            }
            heart_->name = "fire_heart";
            scene.add(heart_);

            sparks_ = field(36, 0.005f);
            auto e = sparks_->emitter();
            e.spawnCenter.set(pit_.x, pit_.y + 0.17f, pit_.z);
            e.spawnHalfExtent.set(0.13f, 0.06f, 0.13f);
            e.velocity.set(0.f, 1.25f, 0.f);
            e.speedSpread = 0.5f;
            e.accel.set(wind_.x * 0.5f, -0.22f, wind_.z * 0.5f);
            e.driftAmplitude = 0.24f;
            e.driftFrequency = 0.9f;
            e.driftGrowth = 1.f;
            e.lifetime = 3.2f;
            e.lifetimeJitter = 0.5f;
            e.dutyCycle = 0.42f;// dark for more than half of every period
            e.size = 0.005f;
            e.sizeJitter = 0.6f;
            e.seed = 31u;
            sparks_->setEmitter(e);
            sparks_->setBillboardRepr(Color(1.0f, 0.58f, 0.16f), Color(0.75f, 0.10f, 0.01f), kSpark, 1.f);
            {
                auto& b = sparks_->billboardRepr();
                b.softness = 0.25f;
                b.fadePower = 1.7f;
                b.sizeTaper = 0.5f;
                b.brightJitter = 0.6f;
                b.stretchSeconds = 0.045f;
                b.stretchMax = 10.f;
                b.stretchMaxScreen = 0.03f;
            }
            sparks_->name = "fire_sparks";
            scene.add(sparks_);
        }

        void buildWoodSmoke(threepp::Scene& scene) {
            using namespace threepp;
            constexpr std::uint32_t n = 160;
            woodSmoke_ = field(n, 0.045f);
            woodSmoke_->setBillboardRepr(Color(0.25f, 0.32f, 0.45f), Color(0.55f, 0.72f, 1.0f), kWoodSmoke, 3.f);
            auto& b = woodSmoke_->billboardRepr();
            b.alphaOver = true;
            b.opacity = 0.05f;
            b.softness = 1.f;
            b.coreWeight = 0.f;
            b.sizeTaper = 0.f;
            b.fadePower = 1.25f;
            b.brightJitter = 0.15f;
            b.lit = true;
            b.litAmbient = 3.4f;// the blue floor (the sun term is warm)
            b.litPhaseG = 0.40f;
            woodSmoke_->name = "fire_wood_smoke";
            scene.add(woodSmoke_);
            woodK_ = 352.f / static_cast<float>(n);
        }

        threepp::Vector3 pit_, wind_;
        std::vector<std::shared_ptr<threepp::ParticleField>> chimney_;
        std::shared_ptr<threepp::ParticleField> flames_, heart_, sparks_, woodSmoke_;
        std::shared_ptr<threepp::MeshStandardMaterial> bedMat_;
        std::shared_ptr<threepp::PointLight> light_;
        float woodK_ = 1.f;
    };

}// namespace fjord

#endif//THREEPP_FJORD_FIRE_HPP
