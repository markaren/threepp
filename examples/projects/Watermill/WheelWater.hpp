// What turns the wheel: the water. The flume's jet falls on the wheel just past its top and fills the bucket it
// lands in; each bucket's water stands level in it (its cross-section, clipped by a horizontal line, holds the
// bucket's volume) and pours out over the lip once the wheel has tipped the bucket far enough that its mouth is lower
// than its water. The weight of the water on the downstream side turns the wheel (torque = sum of rho g V z about
// the axle); its bearings and the gearing it drives hold it back. Nothing is scripted: start it dry and it fills,
// creeps, and runs up to its speed.
//
// Also drawn here: the jet (a sheet along its ballistic path) and the water standing in each bucket.

#ifndef THREEPP_EXAMPLE_WATERMILL_WHEELWATER_HPP
#define THREEPP_EXAMPLE_WATERMILL_WHEELWATER_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include "Site.hpp"
#include "Wheel.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace mill {

    class WheelSim {
    public:
        static constexpr int N = WheelShape::buckets;
        static constexpr float rho = 1000.f, g = 9.81f;

        float angle = 0.f;      // rad about +x
        float omega = 0.f;      // rad/s
        float inertia = 22000.f;// kg m2, the iron wheel and its gearing
        float coulomb = 650.f;  // N m: the bearings' friction
        float viscous = 13000.f;// N m s: the load the gearing puts on it
        float inflow = 0.11f;   // m3/s from the flume
        float jetSpeed = 1.9f;  // m/s along +z at the flume's lip
        float lipHeight = 0.09f;// the water's depth at the lip

        std::array<float, N> volume{};
        float torque = 0.f;
        float waterMass = 0.f;
        float spillRate = 0.f;// m3/s leaving the wheel (smoothed)

        struct Spill {
            float y, z, vy, vz, volume;// relative to the axle
        };
        std::vector<Spill> spills;// this step's

        struct Surface {
            bool wet = false;
            float y = 0, za = 0, zb = 0;// the level and its extent, relative to the axle
            float fill = 0;              // of the bucket's capacity
        };
        std::array<Surface, N> surfaces{};

        // where the jet comes from and lands, relative to the axle
        float lipY = 0, lipZ = 0, hitY = 0, hitZ = 0, hitVy = 0, hitVz = 0, hitT = 0;

        WheelSim() {
            using namespace site;
            lipZ = flumeEndZ - wheelZ;
            lipY = flumeFloorAtEnd + lipHeight - wheelY;
            // the jet's fall to the rim
            float t = 0;
            for (; t < 1.f; t += 0.001f) {
                const float z = lipZ + jetSpeed * t, y = lipY - 0.5f * g * t * t;
                if (z * z + y * y <= WheelShape::rRim * WheelShape::rRim) break;
            }
            hitT = t;
            hitZ = lipZ + jetSpeed * t;
            hitY = lipY - 0.5f * g * t * t;
            hitVz = jetSpeed;
            hitVy = -g * t;
        }

        // the bucket's cross-section (y, z relative to the axle) at the wheel's angle; its mouth is the last edge
        [[nodiscard]] std::vector<std::array<float, 2>> cell(int k) const {
            const float a0 = WheelShape::pitch * static_cast<float>(k) + angle;
            const float p = WheelShape::pitch;
            const auto at = [](float r, float phi) { return std::array<float, 2>{r * std::cos(phi), r * std::sin(phi)}; };
            return {at(WheelShape::rSole, a0), at(WheelShape::rSole, a0 + 0.5f * p), at(WheelShape::rSole, a0 + p), at(WheelShape::rKnee, a0 + p),
                    at(WheelShape::rRim, a0 + p - WheelShape::lipBack), at(WheelShape::rKnee, a0)};
        }

        // the part of a polygon below the level y = L
        static std::vector<std::array<float, 2>> below(const std::vector<std::array<float, 2>>& poly, float L) {
            std::vector<std::array<float, 2>> out;
            for (size_t i = 0; i < poly.size(); ++i) {
                const auto& a = poly[i];
                const auto& b = poly[(i + 1) % poly.size()];
                const bool ain = a[0] <= L, bin = b[0] <= L;
                if (ain) out.push_back(a);
                if (ain != bin) {
                    const float t = (L - a[0]) / (b[0] - a[0]);
                    out.push_back({L, a[1] + t * (b[1] - a[1])});
                }
            }
            return out;
        }

        // area and centroid z of a polygon (y, z)
        static void areaCentroid(const std::vector<std::array<float, 2>>& p, float& area, float& cz) {
            float a = 0, z = 0;
            for (size_t i = 0; i < p.size(); ++i) {
                const auto& u = p[i];
                const auto& v = p[(i + 1) % p.size()];
                const float cr = u[1] * v[0] - v[1] * u[0];
                a += cr;
                z += (u[1] + v[1]) * cr;
            }
            area = 0.5f * std::abs(a);
            cz = std::abs(a) > 1e-9f ? z / (3.f * a) : 0.f;
        }

        void step(float dt) {
            const float width = 2.f * (WheelShape::halfWidth - WheelShape::shroud);
            spills.clear();
            // the jet fills the bucket whose mouth it falls into
            {
                const float phi = std::atan2(hitZ, hitY);
                float rel = phi - angle;
                rel -= 6.2831853f * std::floor(rel / 6.2831853f);
                const int k = static_cast<int>(rel / WheelShape::pitch) % N;
                volume[k] += inflow * dt;
            }
            torque = 0;
            waterMass = 0;
            float spilled = 0;
            for (int k = 0; k < N; ++k) {
                auto poly = cell(k);
                float lo = 1e9f;
                for (const auto& q : poly) lo = std::min(lo, q[0]);
                const auto& D = poly[4];
                const auto& F = poly[5];
                const float spillLevel = std::min(D[0], F[0]);
                float capArea = 0, cz = 0;
                if (spillLevel > lo) areaCentroid(below(poly, spillLevel), capArea, cz);
                const float cap = capArea * width;
                Surface& s = surfaces[k];
                s.wet = false;
                if (volume[k] > cap) {
                    const float extra = volume[k] - cap;
                    volume[k] = cap;
                    spilled += extra;
                    // it pours from the lowest point of the mouth, moving with the wheel there
                    const auto& m = D[0] < F[0] ? D : F;
                    spills.push_back({m[0], m[1], -omega * m[1], omega * m[0], extra});
                }
                if (volume[k] < 1e-6f) {
                    volume[k] = 0;
                    continue;
                }
                // the level that holds this volume
                const float target = volume[k] / width;
                float a = lo, b = spillLevel;
                for (int it = 0; it < 24; ++it) {
                    const float L = 0.5f * (a + b);
                    float ar, c;
                    areaCentroid(below(poly, L), ar, c);
                    if (ar < target) a = L;
                    else b = L;
                }
                const float L = 0.5f * (a + b);
                float ar, c;
                const auto wet = below(poly, L);
                areaCentroid(wet, ar, c);
                torque += rho * g * volume[k] * c;
                waterMass += rho * volume[k];
                // the free surface: where the level line crosses the bucket
                float za = 1e9f, zb = -1e9f;
                for (const auto& q : wet) {
                    if (std::abs(q[0] - L) < 1e-4f) {
                        za = std::min(za, q[1]);
                        zb = std::max(zb, q[1]);
                    }
                }
                if (zb > za) {
                    s.wet = true;
                    s.y = L;
                    s.za = za;
                    s.zb = zb;
                    s.fill = cap > 0 ? volume[k] / cap : 1.f;
                }
            }
            // the jet's push on the buckets (its speed along the rim beyond the rim's own)
            {
                const float r = std::sqrt(hitY * hitY + hitZ * hitZ);
                const float ty = -hitZ / r, tz = hitY / r;// the rim's direction of travel at the hit
                const float vJet = hitVy * ty + hitVz * tz;
                torque += rho * inflow * (vJet - omega * r) * r;
            }
            const float I = inertia + waterMass * WheelShape::rKnee * WheelShape::rKnee;
            float drive = torque;
            if (std::abs(omega) < 1e-3f && std::abs(drive) < coulomb) {
                omega = 0;
            } else {
                drive -= coulomb * (omega != 0 ? (omega > 0 ? 1.f : -1.f) : (drive > 0 ? 1.f : -1.f));
                omega = (omega + dt * drive / I) / (1.f + dt * viscous / I);
            }
            angle += omega * dt;
            const float rate = spilled / std::max(dt, 1e-4f);
            spillRate += std::min(1.f, dt / 0.5f) * (rate - spillRate);
        }
    };

    // the water standing in the buckets: one level quad a bucket, rebuilt each frame (in the wheel's fixed frame)
    class BucketWater {
    public:
        std::shared_ptr<threepp::Mesh> mesh;

        explicit BucketWater(const std::shared_ptr<threepp::Material>& material) {
            const int n = WheelSim::N * 4;
            pos_.assign(3 * n, 0.f);
            nor_.assign(3 * n, 0.f);
            vel_.assign(2 * n, 0.f);
            info_.assign(2 * n, 0.f);
            uv_.assign(2 * n, 0.f);
            std::vector<unsigned int> index;
            for (unsigned k = 0; k < WheelSim::N; ++k) {
                const unsigned b = 4 * k;
                index.insert(index.end(), {b, b + 2, b + 1, b + 1, b + 2, b + 3});
            }
            auto g = threepp::BufferGeometry::create();
            g->setAttribute("position", threepp::FloatBufferAttribute::create(pos_, 3));
            g->setAttribute("normal", threepp::FloatBufferAttribute::create(nor_, 3));
            g->setAttribute("flowVel", threepp::FloatBufferAttribute::create(vel_, 2));
            g->setAttribute("waterInfo", threepp::FloatBufferAttribute::create(info_, 2));
            g->setAttribute("uv", threepp::FloatBufferAttribute::create(uv_, 2));
            g->setAttribute("uv2", threepp::FloatBufferAttribute::create(vel_, 2));
            g->setIndex(index);
            mesh = threepp::Mesh::create(g, material);
            mesh->frustumCulled = false;
        }

        void update(const WheelSim& sim) {
            const float hx = WheelShape::halfWidth - WheelShape::shroud - 0.002f;
            for (int k = 0; k < WheelSim::N; ++k) {
                const auto& s = sim.surfaces[k];
                for (int v = 0; v < 4; ++v) {
                    const size_t i = 4 * k + v;
                    const float x = (v & 1) ? hx : -hx;
                    const float z = (v & 2) ? s.zb : s.za;
                    pos_[3 * i] = s.wet ? x : 0.f;
                    pos_[3 * i + 1] = s.wet ? s.y + 0.004f : 0.f;
                    pos_[3 * i + 2] = s.wet ? z : 0.f;
                    nor_[3 * i] = 0;
                    nor_[3 * i + 1] = 1;
                    nor_[3 * i + 2] = 0;
                    // churned near the top where the jet lands, calmer lower down
                    const float foam = s.wet ? clamp01(0.25f + 0.6f * smooth(0.6f, 1.f, s.y / WheelShape::rRim)) : 0.f;
                    info_[2 * i] = foam;
                    info_[2 * i + 1] = s.wet ? 0.25f : 0.f;
                    uv_[2 * i] = foam;
                    vel_[2 * i] = 0;
                    vel_[2 * i + 1] = sim.omega * s.y;// the bucket's own motion
                }
            }
            auto& g = *mesh->geometry();
            g.getAttribute<float>("position")->array() = pos_;
            g.getAttribute<float>("normal")->array() = nor_;
            g.getAttribute<float>("flowVel")->array() = vel_;
            g.getAttribute<float>("waterInfo")->array() = info_;
            g.getAttribute<float>("uv")->array() = uv_;
            g.getAttribute<float>("uv2")->array() = vel_;
            for (const auto& name : {"position", "normal", "flowVel", "waterInfo", "uv", "uv2"}) g.getAttribute(name)->needsUpdate();
        }

    private:
        std::vector<float> pos_, nor_, vel_, info_, uv_;
    };

    // what pours out of the tipping buckets: for each bucket spilling now, a falling sheet from its mouth's lowest
    // point down to the tail water, along the path its water takes (it leaves with the bucket's speed). Rebuilt each
    // frame in the wheel's fixed frame; the sheets of buckets that are not spilling collapse to nothing.
    class PourSheets {
    public:
        static constexpr int maxSheets = 8, along = 14, across = 4;
        std::shared_ptr<threepp::Mesh> mesh;

        PourSheets(const std::shared_ptr<threepp::Material>& material, float tailY): tailY_(tailY) {
            const int n = maxSheets * (along + 1) * (across + 1);
            pos_.assign(3 * n, 0.f);
            nor_.assign(3 * n, 0.f);
            uv_.assign(2 * n, 0.f);
            col_.assign(4 * n, 1.f);
            std::vector<unsigned int> index;
            const int w = across + 1;
            for (int s = 0; s < maxSheets; ++s) {
                const unsigned base = s * (along + 1) * w;
                for (int k = 0; k < along; ++k)
                    for (int a = 0; a < across; ++a) {
                        const unsigned i0 = base + k * w + a, i1 = i0 + 1, i2 = i0 + w, i3 = i2 + 1;
                        index.insert(index.end(), {i0, i2, i1, i1, i2, i3});
                    }
            }
            auto g = threepp::BufferGeometry::create();
            g->setAttribute("position", threepp::FloatBufferAttribute::create(pos_, 3));
            g->setAttribute("normal", threepp::FloatBufferAttribute::create(nor_, 3));
            g->setAttribute("uv", threepp::FloatBufferAttribute::create(uv_, 2));
            g->setAttribute("color", threepp::FloatBufferAttribute::create(col_, 4));
            g->setIndex(index);
            mesh = threepp::Mesh::create(g, material);
            mesh->frustumCulled = false;
            mesh->castShadow = false;
        }

        void update(const WheelSim& sim, float dt) {
            // the strongest pours this step
            std::vector<WheelSim::Spill> sp = sim.spills;
            std::sort(sp.begin(), sp.end(), [](const auto& a, const auto& b) { return a.volume > b.volume; });
            const float hw = WheelShape::halfWidth - WheelShape::shroud - 0.01f;
            const int w = across + 1;
            for (int s = 0; s < maxSheets; ++s) {
                const unsigned base = s * (along + 1) * w;
                const bool on = s < static_cast<int>(sp.size()) && sp[s].volume / std::max(dt, 1e-4f) > 0.004f;
                float y0 = 0, z0 = 0, vy = 0, vz = 0, T = 0, width = 0, strength = 0;
                if (on) {
                    y0 = sp[s].y;
                    z0 = sp[s].z;
                    // the bucket's speed, and a little outward as it pours over the lip
                    const float r = std::sqrt(y0 * y0 + z0 * z0);
                    vy = sp[s].vy + 0.35f * y0 / r;
                    vz = sp[s].vz + 0.35f * z0 / r;
                    // time to fall to the tail water: y0 + vy t - g t^2 / 2 = tailY
                    const float a = 4.905f, b = -vy, c = tailY_ - y0;
                    const float disc = b * b - 4.f * a * c;
                    T = disc > 0 ? (-b + std::sqrt(disc)) / (2.f * a) : 0.3f;
                    // a strong pour fills the bucket's width; a weak one is a narrower tongue
                    width = hw * clamp01(0.35f + 8.f * sp[s].volume / std::max(dt, 1e-4f));
                    strength = clamp01(0.4f + 15.f * sp[s].volume / std::max(dt, 1e-4f));
                }
                float dist = 0;
                float py = y0, pz = z0;
                for (int k = 0; k <= along; ++k) {
                    const float t = T * static_cast<float>(k) / static_cast<float>(along);
                    const float y = y0 + vy * t - 4.905f * t * t, z = z0 + vz * t;
                    if (k > 0) dist += std::hypot(y - py, z - pz);
                    py = y;
                    pz = z;
                    float dy = vy - 9.81f * t, dz = vz;
                    const float len = std::max(std::hypot(dy, dz), 1e-4f);
                    dy /= len;
                    dz /= len;
                    // the sheet spreads a little and frays as it falls
                    const float wk = width * (1.f + 0.25f * static_cast<float>(k) / static_cast<float>(along));
                    for (int a = 0; a <= across; ++a) {
                        const size_t i = base + k * w + a;
                        const float x = on ? -wk + 2.f * wk * static_cast<float>(a) / static_cast<float>(across) : 0.f;
                        pos_[3 * i] = x;
                        pos_[3 * i + 1] = on ? y : 0.f;
                        pos_[3 * i + 2] = on ? z : 0.f;
                        nor_[3 * i] = 0;
                        nor_[3 * i + 1] = dz;
                        nor_[3 * i + 2] = -dy;
                        uv_[2 * i] = static_cast<float>(a) / static_cast<float>(across);
                        uv_[2 * i + 1] = dist;
                        // it breaks up into spray before it reaches the water
                        col_[4 * i + 3] = strength * (1.f - smooth(0.6f, 1.f, static_cast<float>(k) / static_cast<float>(along)));
                    }
                }
            }
            auto& g = *mesh->geometry();
            g.getAttribute<float>("position")->array() = pos_;
            g.getAttribute<float>("normal")->array() = nor_;
            g.getAttribute<float>("uv")->array() = uv_;
            g.getAttribute<float>("color")->array() = col_;
            for (const auto& name : {"position", "normal", "uv", "color"}) g.getAttribute(name)->needsUpdate();
        }

    private:
        float tailY_;
        std::vector<float> pos_, nor_, uv_, col_;
    };

    // the jet: a sheet from the flume's lip along its fall to the wheel, its streaks scrolling down it. Built in the
    // wheel's fixed frame (relative to the axle).
    inline std::shared_ptr<threepp::Mesh> makeJet(const WheelSim& sim, const std::shared_ptr<threepp::Material>& material, float extra = 0.25f) {
        using threepp::Vector3;
        std::vector<float> pos, nor, uv, col;
        std::vector<unsigned int> index;
        const int along = 24, across = 6;
        const float hw = 0.5f * site::flumeInner - 0.03f;
        const float tEnd = sim.hitT + extra * sim.hitT;// on a little into the bucket
        float s = 0;
        Vector3 prev;
        for (int k = 0; k <= along; ++k) {
            const float t = tEnd * static_cast<float>(k) / static_cast<float>(along);
            const float z = sim.lipZ + sim.jetSpeed * t, y = sim.lipY - 0.5f * 9.81f * t * t;
            const Vector3 p(0, y, z);
            if (k > 0) s += p.distanceTo(prev);
            prev = p;
            // the sheet's direction, and its normal (in the y-z plane, facing up and back)
            Vector3 dir(0, -9.81f * t, sim.jetSpeed);
            dir.normalize();
            const Vector3 n(0, dir.z, -dir.y);
            // the sheet narrows a little and frays as it falls
            const float w = hw * (1.f - 0.06f * static_cast<float>(k) / static_cast<float>(along));
            for (int a = 0; a <= across; ++a) {
                const float x = -w + 2.f * w * static_cast<float>(a) / static_cast<float>(across);
                // the sheet is thicker in the middle: bow it a few cm
                const float bow = 0.035f * (1.f - (x / w) * (x / w));
                pos.insert(pos.end(), {x, y + n.y * bow, z + n.z * bow});
                nor.insert(nor.end(), {0.f, n.y, n.z});
                uv.insert(uv.end(), {static_cast<float>(a) / static_cast<float>(across), s / 1.f});
                col.insert(col.end(), {1.f, 1.f, 1.f, 1.f - smooth(0.8f, 1.f, static_cast<float>(k) / static_cast<float>(along))});
            }
        }
        const int w = across + 1;
        for (int k = 0; k < along; ++k)
            for (int a = 0; a < across; ++a) {
                const unsigned i0 = k * w + a, i1 = i0 + 1, i2 = i0 + w, i3 = i2 + 1;
                index.insert(index.end(), {i0, i2, i1, i1, i2, i3});
            }
        auto g = threepp::BufferGeometry::create();
        g->setAttribute("position", threepp::FloatBufferAttribute::create(pos, 3));
        g->setAttribute("normal", threepp::FloatBufferAttribute::create(nor, 3));
        g->setAttribute("uv", threepp::FloatBufferAttribute::create(uv, 2));
        g->setAttribute("color", threepp::FloatBufferAttribute::create(col, 4));
        g->setIndex(index);
        auto mesh = threepp::Mesh::create(g, material);
        mesh->castShadow = false;
        mesh->receiveShadow = true;
        return mesh;
    }

}// namespace mill

#endif
