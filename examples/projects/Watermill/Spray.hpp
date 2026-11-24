// Spray: the drops that pour off the wheel's buckets and splash off where the jet lands, and the mist they and the
// stream's white water raise, which the breeze carries away. Two pools of ParticleSprites (lit soft billboards, drawn
// by both renderers): drops fall ballistically with a little drag until they meet the water, where some of them
// become mist; mist puffs swell, drift with the wind (gusting, with a little turbulence and a rise off the falling
// water), and fade.

#ifndef THREEPP_EXAMPLE_WATERMILL_SPRAY_HPP
#define THREEPP_EXAMPLE_WATERMILL_SPRAY_HPP

#include "threepp/objects/ParticleSprites.hpp"

#include "Noise.hpp"

#include <functional>
#include <vector>

namespace mill {

    class Spray {
    public:
        std::shared_ptr<threepp::ParticleSprites> drops, mist, veil;
        threepp::Vector3 wind{-0.3f, 0.f, 0.8f};// m/s, the breeze down the valley, away from the mill

        explicit Spray(size_t dropCapacity = 9000, size_t mistCapacity = 1600, size_t veilCapacity = 500) {
            drops = threepp::ParticleSprites::create(dropCapacity);
            mist = threepp::ParticleSprites::create(mistCapacity);
            veil = threepp::ParticleSprites::create(veilCapacity);
            veil->setAnisotropy(0.6f);
            veil->hideAll();
            v_.resize(veilCapacity);
            drops->setAnisotropy(0.5f);
            mist->setAnisotropy(0.45f);
            drops->hideAll();
            mist->hideAll();
            d_.resize(dropCapacity);
            m_.resize(mistCapacity);
        }

        // `count` drops at a point moving with `vel`, scattered by `spread` m/s
        void emitDrops(const threepp::Vector3& at, const threepp::Vector3& vel, int count, float spread, float jitter = 0.03f) {
            for (int k = 0; k < count; ++k) {
                Particle& p = d_[nextDrop_];
                nextDrop_ = (nextDrop_ + 1) % d_.size();
                p.alive = true;
                p.p = at + threepp::Vector3(rng_.normal(), rng_.normal(), rng_.normal()) * jitter;
                p.v = vel + threepp::Vector3(rng_.normal(), rng_.normal(), rng_.normal()) * spread;
                p.age = 0;
                p.life = rng_.range(1.5f, 3.f);
                p.size = rng_.range(0.012f, 0.03f);
            }
        }

        void emitMist(const threepp::Vector3& at, const threepp::Vector3& vel, float size, float strength = 1.f) {
            Particle& p = m_[nextMist_];
            nextMist_ = (nextMist_ + 1) % m_.size();
            p.alive = true;
            p.p = at;
            p.v = vel;
            p.age = 0;
            p.life = rng_.range(3.f, 6.5f);
            p.size = size * rng_.range(0.7f, 1.3f);
            p.strength = strength;
        }

        // the veil: broad, faint puffs of damp air lying over the water, drifting slowly downstream
        void emitVeil(const threepp::Vector3& at, float size) {
            Particle& p = v_[nextVeil_];
            nextVeil_ = (nextVeil_ + 1) % v_.size();
            p.alive = true;
            p.p = at;
            p.v = threepp::Vector3(0, 0.02f, 0);
            p.age = 0;
            p.life = rng_.range(10.f, 18.f);
            p.size = size * rng_.range(0.7f, 1.3f);
            p.strength = 1.f;
        }

        // waterLevel(x, z): where a drop meets the water (or ground)
        void step(float dt, float time, const std::function<float(float, float)>& surface) {
            using threepp::Vector3;
            // gusts
            const float gust = 0.75f + 0.35f * std::sin(time * 0.37f) + 0.2f * std::sin(time * 1.13f + 1.f);
            const Vector3 w = wind * gust;
            for (auto& p : d_) {
                if (!p.alive) continue;
                p.age += dt;
                // gravity and air drag toward the wind's speed (small drops feel the wind)
                p.v.y -= 9.81f * dt;
                p.v += (w - p.v) * std::min(1.f, dt * 0.6f);
                p.p += p.v * dt;
                const float s = surface(p.p.x, p.p.z);
                if (p.p.y < s || p.age > p.life) {
                    if (p.p.y < s + 0.05f && rng_() < 0.025f) emitMist(Vector3(p.p.x, s + 0.08f, p.p.z), Vector3(p.v.x * 0.2f, 0.25f, p.v.z * 0.2f), 0.25f, 0.8f);
                    p.alive = false;
                }
            }
            for (auto& p : m_) {
                if (!p.alive) continue;
                p.age += dt;
                // the puff relaxes to the wind, rises a little, and wanders
                const float n1 = perlin3(p.p.x * 0.6f, p.p.y * 0.6f, time * 0.3f, 91u), n2 = perlin3(p.p.x * 0.6f + 17.f, p.p.z * 0.6f, time * 0.3f, 92u);
                const Vector3 target = w + Vector3(0.6f * n1, 0.12f + 0.25f * n2, 0.6f * n2);
                p.v += (target - p.v) * std::min(1.f, dt * 0.9f);
                p.p += p.v * dt;
                if (p.age > p.life) p.alive = false;
            }
            for (auto& p : v_) {
                if (!p.alive) continue;
                p.age += dt;
                const float n1 = perlin3(p.p.x * 0.2f, p.p.z * 0.2f, time * 0.1f, 93u);
                const Vector3 target = w * 0.45f + Vector3(0.25f * n1, 0.015f, 0.25f * perlin3(p.p.z * 0.2f, p.p.x * 0.2f, time * 0.1f, 94u));
                p.v += (target - p.v) * std::min(1.f, dt * 0.4f);
                p.p += p.v * dt;
                if (p.age > p.life) p.alive = false;
            }
        }

        // write the pools for this frame
        void commit() {
            using threepp::Color;
            const Color dropColour(0.82f, 0.86f, 0.9f), mistColour(0.92f, 0.94f, 0.96f);
            for (size_t i = 0; i < d_.size(); ++i) {
                const auto& p = d_[i];
                if (!p.alive) {
                    drops->hide(i);
                    continue;
                }
                const float fade = 1.f - smooth(0.7f * p.life, p.life, p.age);
                drops->set(i, p.p, p.size, 0.55f * fade, dropColour);
            }
            for (size_t i = 0; i < m_.size(); ++i) {
                const auto& p = m_[i];
                if (!p.alive) {
                    mist->hide(i);
                    continue;
                }
                const float t = p.age / p.life;
                const float size = p.size * (1.f + 3.5f * std::sqrt(t));
                const float opacity = 0.022f * p.strength * smooth(0.f, 0.2f, t) * (1.f - smooth(0.4f, 1.f, t));
                mist->set(i, p.p, size, opacity, mistColour, 6.2831853f * unit(static_cast<uint32_t>(i) * 2654435761u));
            }
            for (size_t i = 0; i < v_.size(); ++i) {
                const auto& p = v_[i];
                if (!p.alive) {
                    veil->hide(i);
                    continue;
                }
                const float t = p.age / p.life;
                const float opacity = 0.014f * smooth(0.f, 0.25f, t) * (1.f - smooth(0.6f, 1.f, t));
                veil->set(i, p.p, p.size * (1.f + 0.6f * t), opacity, Color(0.9f, 0.93f, 0.95f), 6.2831853f * unit(static_cast<uint32_t>(i) * 2246822519u));
            }
            drops->commit();
            mist->commit();
            veil->commit();
        }

        [[nodiscard]] size_t aliveDrops() const {
            size_t n = 0;
            for (const auto& p : d_) n += p.alive;
            return n;
        }

    private:
        struct Particle {
            threepp::Vector3 p, v;
            float age = 0, life = 1, size = 0.02f, strength = 1;
            bool alive = false;
        };
        std::vector<Particle> d_, m_, v_;
        size_t nextDrop_ = 0, nextMist_ = 0, nextVeil_ = 0;
        Rng rng_{4242};
    };

}// namespace mill

#endif
