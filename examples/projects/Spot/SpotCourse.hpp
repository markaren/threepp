// The landscape spot_play walks: a meadow ringed by low hills, a dry-stone
// terrace with a granite stair cut into it, a rubble patch on the terrace, and
// mossy boulders. Every solid piece is a PhysX static, and heightAt() raycasts
// those statics, so the policy's height scan reads exactly what the feet touch:
// the stair treads, the stones, the slope down the far side.
//
// World is Z-up, as in SpotScene.hpp. Spot starts at the origin facing +x; the
// gravel path leads to the stair, across the terrace and down its gentle east
// slope, and loops back round to the start.
//
// The stair is the trained geometry: 8 risers of 0.15 m on 0.35 m treads
// (spot_steps_env trains risers up to 0.20 m on 0.30 m treads). The east slope
// stays under 11 degrees, well inside what the policy walks.
//
// Assets (threepp_data, all CC0 from Poly Haven): textures/pbr/{forrest_ground_01,
// forest_ground_04, mossy_rock, rock_wall_10} and models/gltf/rock_moss_set_02.

#ifndef THREEPP_EXAMPLES_SPOT_COURSE_HPP
#define THREEPP_EXAMPLES_SPOT_COURSE_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/threepp.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace spot::course {

    // ── Layout (metres) ─────────────────────────────────────────────────────
    constexpr float SIZE = 96.f;// terrain side
    constexpr float CELL = 0.25f;
    constexpr int N = 385;// samples per side = SIZE / CELL + 1
    constexpr float X0 = -40.f, Y0 = -48.f;// terrain corner; the course runs along +x from the origin

    constexpr int STEPS = 8;
    constexpr float RISE = 0.15f, TREAD = 0.35f;
    constexpr float TERRACE_H = STEPS * RISE;// 1.2
    constexpr float STAIR_X0 = 7.5f;         // first riser, flush with the terrace wall's face
    constexpr float STAIR_TOP = STAIR_X0 + STEPS * TREAD;
    constexpr float STAIR_BACK = 10.5f;// the steps run back to here, under the terrace's first grid row
    constexpr float STAIR_HALF_W = 1.0f;
    constexpr float WALL_T = 0.5f;          // the front wall spans x in [STAIR_X0, STAIR_X0 + WALL_T]
    constexpr float TERRACE_X = 7.75f;      // terrain step to the terrace, hidden inside the front wall
    constexpr float NOTCH_HALF = 1.1f;      // terrain stays at meadow level under the stair...
    constexpr float NOTCH_END = 10.4f;      // ...up to here, hidden inside the steps and the cheek walls
    constexpr float TERRACE_END = 21.f;     // the east slope starts here...
    constexpr float SLOPE_END = 31.f;       // ...and is back at meadow level here
    constexpr float TERRACE_HALF_Y = 6.f;   // full height for |y| below this,
    constexpr float TERRACE_SIDE = 4.f;     // falling to nothing over this much more
    constexpr float RUBBLE_X0 = 14.5f, RUBBLE_X1 = 18.5f, RUBBLE_HALF_Y = 1.8f;

    static_assert(N == static_cast<int>(SIZE / CELL) + 1);

    // ── Deterministic noise and randomness ──────────────────────────────────
    // Hand-rolled rather than <random>: std:: distributions are implementation-
    // defined, and the web build (libc++) must lay out the same course as MSVC.
    inline float hash2(int x, int y, std::uint32_t seed) {
        std::uint32_t h = static_cast<std::uint32_t>(x) * 0x8da6b343u ^ static_cast<std::uint32_t>(y) * 0xd8163841u ^ seed * 0xcb1ab31fu;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        return static_cast<float>(h & 0xffffffu) / static_cast<float>(0xffffffu) * 2.f - 1.f;
    }

    inline float valueNoise(float x, float y, std::uint32_t seed) {
        const float fx = std::floor(x), fy = std::floor(y);
        const int xi = static_cast<int>(fx), yi = static_cast<int>(fy);
        const float tx = x - fx, ty = y - fy;
        const float u = tx * tx * tx * (tx * (tx * 6.f - 15.f) + 10.f);
        const float v = ty * ty * ty * (ty * (ty * 6.f - 15.f) + 10.f);
        const float a = hash2(xi, yi, seed), b = hash2(xi + 1, yi, seed);
        const float c = hash2(xi, yi + 1, seed), d = hash2(xi + 1, yi + 1, seed);
        return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
    }

    inline float fbm(float x, float y, int octaves, std::uint32_t seed) {
        float sum = 0.f, amp = 0.5f, norm = 0.f;
        for (int i = 0; i < octaves; ++i) {
            sum += amp * valueNoise(x, y, seed + static_cast<std::uint32_t>(i) * 101u);
            norm += amp;
            x = x * 2.03f + 17.1f;
            y = y * 2.03f - 9.7f;
            amp *= 0.5f;
        }
        return sum / norm;
    }

    inline float smoothstep(float e0, float e1, float x) {
        const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

    struct Rand {
        std::uint64_t s;
        explicit Rand(std::uint64_t seed): s(seed * 0x9e3779b97f4a7c15ull + 1) {}
        float next() {// [0, 1)
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            return static_cast<float>((s >> 40) & 0xffffffu) / 16777216.f;
        }
        float range(float a, float b) { return a + (b - a) * next(); }
    };

    // ── The gravel path: a closed loop through the course ───────────────────
    inline std::vector<threepp::Vector2> pathPoints() {
        // Control points, Catmull-Rom'd into a dense polyline. The stair sits on the
        // straight between (6.6, 0) and (11.2, 0); the path itself skips the notch.
        const std::vector<threepp::Vector2> ctrl{
                {-4.f, -1.5f}, {0.f, 0.f}, {4.f, 0.2f}, {6.8f, 0.f}, {11.2f, 0.f}, {14.f, 0.4f}, {17.f, -0.5f},
                {20.5f, 0.f}, {25.f, 1.2f}, {30.f, 3.5f}, {33.f, 9.f}, {29.f, 14.f}, {21.f, 15.5f}, {12.f, 14.f},
                {4.f, 11.5f}, {-3.f, 8.f}, {-6.5f, 3.f}};
        std::vector<threepp::Vector2> out;
        const int n = static_cast<int>(ctrl.size());
        for (int i = 0; i < n; ++i) {
            const auto& p0 = ctrl[(i + n - 1) % n];
            const auto& p1 = ctrl[i];
            const auto& p2 = ctrl[(i + 1) % n];
            const auto& p3 = ctrl[(i + 2) % n];
            for (int k = 0; k < 12; ++k) {
                const float t = static_cast<float>(k) / 12.f, t2 = t * t, t3 = t2 * t;
                const auto f = [&](float a, float b, float c, float d) {
                    return 0.5f * (2.f * b + (-a + c) * t + (2.f * a - 5.f * b + 4.f * c - d) * t2 + (-a + 3.f * b - 3.f * c + d) * t3);
                };
                out.emplace_back(f(p0.x, p1.x, p2.x, p3.x), f(p0.y, p1.y, p2.y, p3.y));
            }
        }
        return out;
    }

    inline float distanceToPolyline(const std::vector<threepp::Vector2>& pts, float x, float y) {
        float best = 1e9f;
        const std::size_t n = pts.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& a = pts[i];
            const auto& b = pts[(i + 1) % n];
            const float abx = b.x - a.x, aby = b.y - a.y;
            const float t = std::clamp(((x - a.x) * abx + (y - a.y) * aby) / std::max(abx * abx + aby * aby, 1e-9f), 0.f, 1.f);
            const float dx = x - (a.x + abx * t), dy = y - (a.y + aby * t);
            best = std::min(best, dx * dx + dy * dy);
        }
        return std::sqrt(best);
    }

    // ── The terrain height function ─────────────────────────────────────────
    inline float terrace(float x, float y) {
        const float ay = std::abs(y);
        const float side = 1.f - smoothstep(TERRACE_HALF_Y, TERRACE_HALF_Y + TERRACE_SIDE, ay);
        float along;
        if (x < TERRACE_X || (ay < NOTCH_HALF && x < NOTCH_END)) along = 0.f;
        else if (x < TERRACE_END) along = 1.f;
        else along = 1.f - smoothstep(TERRACE_END, SLOPE_END, x);
        return TERRACE_H * along * side;
    }

    // The meadow's own relief, faded out round the stair so its foot sits exactly
    // on z = 0 and the terrace meets its top tread exactly at TERRACE_H.
    inline float meadow(float x, float y) {
        const float sx = std::clamp(x, STAIR_X0 - 0.5f, STAIR_BACK + 1.f);
        const float d = std::hypot(x - sx, y);
        const float fade = smoothstep(1.5f, 4.5f, d);
        return fade * (0.22f * fbm(x / 9.f, y / 9.f, 3, 11u) + 0.035f * valueNoise(x / 1.7f, y / 1.7f, 23u));
    }

    inline float rim(float x, float y) {
        const float r = std::hypot(x - 12.f, y * 1.1f);
        return smoothstep(27.f, 46.f, r) * (5.5f + 2.5f * fbm(x / 15.f, y / 15.f, 3, 37u));
    }

    inline float terrainHeight(float x, float y) { return meadow(x, y) + terrace(x, y) + rim(x, y); }

    // ── Geometry helpers ────────────────────────────────────────────────────
    // A box whose UVs are world-anchored (metres / tile), so neighbouring boxes of
    // one material line up and every face has the same texel density. A uv offset
    // cuts one stone out of a different part of the texture than its neighbour.
    inline std::shared_ptr<threepp::BufferGeometry> worldUvBox(float sx, float sy, float sz,
                                                               const threepp::Vector3& center, float tile,
                                                               const threepp::Vector2& uvOffset = {}) {
        auto g = threepp::BoxGeometry::create(sx, sy, sz);
        const auto* pos = g->getAttribute<float>("position");
        const auto* nrm = g->getAttribute<float>("normal");
        auto* uv = g->getAttribute<float>("uv");
        for (int i = 0; i < pos->count(); ++i) {
            const float px = pos->getX(i) + center.x, py = pos->getY(i) + center.y, pz = pos->getZ(i) + center.z;
            const float nx = std::abs(nrm->getX(i)), ny = std::abs(nrm->getY(i)), nz = std::abs(nrm->getZ(i));
            float u, v;
            if (nx >= ny && nx >= nz) u = py, v = pz;
            else if (ny >= nz) u = px, v = pz;
            else u = px, v = py;
            uv->setXY(i, u / tile + uvOffset.x, v / tile + uvOffset.y);
        }
        uv->needsUpdate();
        return g;
    }

    struct PbrSet {
        std::shared_ptr<threepp::Texture> map, normal, arm;
    };

    inline PbrSet loadPbr(const std::string& data, const std::string& name, float repeat) {
        threepp::TextureLoader loader;
        const std::string dir = data + "/textures/pbr/" + name + "/" + name;
        PbrSet s{loader.load(dir + "_diff_1k.jpg"), loader.load(dir + "_nor_gl_1k.jpg"), loader.load(dir + "_arm_512.jpg")};
        if (!s.map || !s.normal || !s.arm) throw std::runtime_error("SpotCourse: missing texture set " + dir);
        s.map->colorSpace = threepp::ColorSpace::sRGB;
        for (auto* t : {s.map.get(), s.normal.get(), s.arm.get()}) {
            t->wrapS = t->wrapT = threepp::TextureWrapping::Repeat;
            t->repeat.set(repeat, repeat);
            t->anisotropy = 8;
            t->needsUpdate();
        }
        return s;
    }

    inline std::shared_ptr<threepp::MeshStandardMaterial> pbrMaterial(const PbrSet& s, float normalScale = 1.f) {
        auto m = threepp::MeshStandardMaterial::create();
        m->map = s.map;
        m->normalMap = s.normal;
        m->normalScale.set(normalScale, normalScale);
        m->roughnessMap = s.arm;// G
        m->aoMap = s.arm;       // R
        m->roughness = 1.f;
        m->metalness = 0.f;
        return m;
    }

    // ── The course ──────────────────────────────────────────────────────────
    class Course {
    public:
        // Builds the physics (always) and, with `visuals`, the meshes under `root`.
        Course(threepp::PhysxWorld& world, threepp::Object3D* root, const std::string& data, bool visuals = true)
            : world_(world), scene_(&world.scene()), root_(root), data_(data), visuals_(visuals && root) {
            path_ = pathPoints();
            buildTerrain();
            buildStairAndWalls();
            buildRocks();
            if (visuals_) buildGround();// last: it is shaded by what stands on it
        }

        // Ground height under (x, y): the top of whatever static is there, terrain,
        // step, wall or stone. Off the map it is the analytic terrain.
        [[nodiscard]] float heightAt(float x, float y) const {
            using namespace ::physx;
            PxRaycastBuffer hit;
            const PxQueryFilterData filter(PxQueryFlag::eSTATIC);
            if (scene_->raycast(PxVec3(x, y, 60.f), PxVec3(0, 0, -1), 120.f, hit, PxHitFlag::ePOSITION, filter) && hit.hasBlock) {
                return hit.block.position.z;
            }
            return terrainHeight(x, y);
        }

        [[nodiscard]] static threepp::Vector2 spawn() { return {0.f, 0.f}; }

        // The bare ground under (x, y) as the terrain mesh draws it, between its samples:
        // no steps, walls or stones (heightAt has those).
        [[nodiscard]] float groundAt(float x, float y) const {
            const float fx = std::clamp((x - X0) / CELL, 0.f, N - 1.001f), fy = std::clamp((y - Y0) / CELL, 0.f, N - 1.001f);
            const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
            const float tx = fx - ix, ty = fy - iy;
            const auto h = [&](int i, int j) { return heights_[static_cast<std::size_t>(j) * N + i]; };
            const float a = h(ix, iy) + (h(ix + 1, iy) - h(ix, iy)) * tx;
            const float b = h(ix, iy + 1) + (h(ix + 1, iy + 1) - h(ix, iy + 1)) * tx;
            return a + (b - a) * ty;
        }

        [[nodiscard]] float distanceToPath(float x, float y) const { return distanceToPolyline(path_, x, y); }

        // Every stone on the course: x, y and the radius kept clear round it.
        [[nodiscard]] const std::vector<threepp::Vector3>& stones() const { return placed_; }

        // Darkens the ground by `factor`, one value per height-field sample (N x N, row-major
        // in y): shade cast by what is planted on it after the course was built.
        void shadeGround(const std::vector<float>& factor) {
            if (factor.size() != heights_.size()) return;
            for (const auto& g : ground_) {
                if (!g) continue;
                auto* col = g->getAttribute<float>("color");
                auto& c = col->array();
                for (std::size_t i = 0; i < factor.size(); ++i) {
                    for (int k = 0; k < 3; ++k) c[i * 3 + k] *= factor[i];
                }
                col->needsUpdate();
            }
        }

    private:
        threepp::PhysxWorld& world_;
        const ::physx::PxScene* scene_;
        threepp::Object3D* root_;
        std::string data_;
        bool visuals_;
        std::vector<threepp::Vector2> path_;
        std::vector<float> heights_;          // the height field, N x N, row-major in y
        std::vector<threepp::Vector3> placed_;// every stone: x, y, footprint radius
        std::array<std::shared_ptr<threepp::BufferGeometry>, 2> ground_;// the meadow grid, the overlays' grid

        void addVisual(const std::shared_ptr<threepp::Object3D>& o) {
            if (visuals_) root_->add(o);
        }

        void staticBox(float sx, float sy, float sz, const threepp::Vector3& c,
                       const std::shared_ptr<threepp::Material>& mat, float tile,
                       const threepp::Vector2& uvOffset = {}) {
            using namespace ::physx;
            world_.addStatic(PxBoxGeometry(sx / 2, sy / 2, sz / 2), PxTransform(PxVec3(c.x, c.y, c.z)));
            if (!visuals_) return;
            auto mesh = threepp::Mesh::create(worldUvBox(sx, sy, sz, c, tile, uvOffset), mat);
            mesh->position.copy(c);
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            root_->add(mesh);
        }

        // ── Terrain ─────────────────────────────────────────────────────────
        void buildTerrain() {
            heights_.resize(static_cast<std::size_t>(N) * N);
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    heights_[static_cast<std::size_t>(iy) * N + ix] = terrainHeight(X0 + ix * CELL, Y0 + iy * CELL);
                }
            }
            if (!world_.addStaticHeightField(heights_, N, N, CELL, threepp::Vector3(X0, Y0, 0.f))) {
                throw std::runtime_error("SpotCourse: the height field did not cook");
            }
        }

        // How much of the sky a ground point sees past the stones, steps and walls
        // around it (1 = all of it): cosine-weighted rays against the course's statics,
        // the height field left out, since the open slopes' own shading is the
        // lighting's job. Only points near something standing on the ground are cast.
        [[nodiscard]] std::vector<float> skyVisibility() const {
            using namespace ::physx;
            struct NotTerrain: PxQueryFilterCallback {
                PxQueryHitType::Enum preFilter(const PxFilterData&, const PxShape* shape, const PxRigidActor*, PxHitFlags&) override {
                    return shape->getGeometry().getType() == PxGeometryType::eHEIGHTFIELD ? PxQueryHitType::eNONE : PxQueryHitType::eBLOCK;
                }
                PxQueryHitType::Enum postFilter(const PxFilterData&, const PxQueryHit&, const PxShape*, const PxRigidActor*) override {
                    return PxQueryHitType::eBLOCK;
                }
            } notTerrain;
            const PxQueryFilterData filter(PxQueryFlag::eSTATIC | PxQueryFlag::ePREFILTER);

            constexpr int RAYS = 16;
            constexpr float REACH = 2.5f;// metres: contact shade, not valley shade
            std::array<PxVec3, RAYS> dirs{};
            for (int i = 0; i < RAYS; ++i) {// a cosine-weighted spiral over the upper hemisphere
                const float u = (i + 0.5f) / RAYS, r = std::sqrt(u), a = i * 2.39996f;
                dirs[i] = PxVec3(r * std::cos(a), r * std::sin(a), std::sqrt(1.f - u));
            }
            const auto near = [&](float x, float y) {
                if (x > STAIR_X0 - REACH && x < STAIR_BACK + REACH && std::abs(y) < TERRACE_HALF_Y + TERRACE_SIDE + REACH) return true;
                for (const auto& p : placed_) {
                    if (std::hypot(x - p.x, y - p.y) < p.z + REACH) return true;
                }
                return false;
            };

            std::vector<float> vis(heights_.size(), 1.f);
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    const float x = X0 + ix * CELL, y = Y0 + iy * CELL;
                    if (!near(x, y)) continue;
                    const PxVec3 o(x, y, heights_[static_cast<std::size_t>(iy) * N + ix] + 0.03f);
                    float blocked = 0.f;
                    for (const auto& d : dirs) {
                        PxRaycastBuffer hit;
                        if (scene_->raycast(o, d, REACH, hit, PxHitFlag::ePOSITION, filter, &notTerrain) && hit.hasBlock) {
                            const float t = hit.block.distance / REACH;
                            blocked += 1.f - t * t;// fades out toward the reach instead of stopping there
                        }
                    }
                    vis[static_cast<std::size_t>(iy) * N + ix] = 1.f - blocked / RAYS;
                }
            }
            return vis;
        }

        void buildGround() {
            const auto& h = heights_;
            const std::vector<float> vis = skyVisibility();
            // Darkening at most this much where the sky is fully hidden: the ground at a
            // stone's foot, the meadow along the wall's base, the gravel round the rubble.
            constexpr float CONTACT = 0.6f;

            // One grid, three layers: the meadow, then the gravel path and the granite
            // on steep ground blended over it through their own masks. Each overlay is
            // the same grid drawn again with alpha, so the blend is between two fully
            // lit, shadowed surfaces, which is what a splat shader would compute.
            auto geometry = threepp::BufferGeometry::create();
            std::vector<float> pos, nrm, uv, col, shade;
            pos.reserve(static_cast<std::size_t>(N) * N * 3);
            nrm.reserve(pos.capacity());
            col.reserve(pos.capacity());
            shade.reserve(pos.capacity());
            uv.reserve(static_cast<std::size_t>(N) * N * 2);
            const auto at = [&](int ix, int iy) {
                ix = std::clamp(ix, 0, N - 1);
                iy = std::clamp(iy, 0, N - 1);
                return h[static_cast<std::size_t>(iy) * N + ix];
            };
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    const float x = X0 + ix * CELL, y = Y0 + iy * CELL;
                    pos.insert(pos.end(), {x, y, at(ix, iy)});
                    threepp::Vector3 n(-(at(ix + 1, iy) - at(ix - 1, iy)) / (2 * CELL),
                                       -(at(ix, iy + 1) - at(ix, iy - 1)) / (2 * CELL), 1.f);
                    n.normalize();
                    nrm.insert(nrm.end(), {n.x, n.y, n.z});
                    uv.insert(uv.end(), {(x - X0) / SIZE, (y - Y0) / SIZE});
                    const float s = 1.f - CONTACT * (1.f - vis[static_cast<std::size_t>(iy) * N + ix]);
                    shade.insert(shade.end(), {s, s, s});
                    // Macro tint: drier and lusher drifts and a darker grain, so the
                    // 2.4 m meadow tile does not read as a grid from a distance.
                    const float lush = 0.5f + 0.5f * fbm(x / 13.f, y / 13.f, 3, 51u);
                    const float grain = 0.93f + 0.1f * fbm(x / 3.2f, y / 3.2f, 2, 67u);
                    const float r = (1.06f + (0.86f - 1.06f) * lush) * grain;
                    const float g = (1.00f + (1.00f - 1.00f) * lush) * grain;
                    const float b = (0.80f + (0.84f - 0.80f) * lush) * grain;
                    col.insert(col.end(), {r * s, g * s, b * s});
                }
            }
            std::vector<unsigned int> idx;
            idx.reserve(static_cast<std::size_t>(N - 1) * (N - 1) * 6);
            for (int iy = 0; iy < N - 1; ++iy) {
                for (int ix = 0; ix < N - 1; ++ix) {
                    const unsigned a = iy * N + ix, b = a + 1, c = a + N, d = c + 1;
                    idx.insert(idx.end(), {a, b, d, a, d, c});
                }
            }
            geometry->setAttribute("position", threepp::FloatBufferAttribute::create(std::move(pos), 3));
            geometry->setAttribute("normal", threepp::FloatBufferAttribute::create(std::move(nrm), 3));
            geometry->setAttribute("uv", threepp::FloatBufferAttribute::create(std::move(uv), 2));
            geometry->setAttribute("color", threepp::FloatBufferAttribute::create(std::move(col), 3));
            geometry->setIndex(std::move(idx));
            // The overlays carry the contact shade without the meadow's tint.
            auto overlayGeometry = geometry->clone();
            ground_ = {geometry, overlayGeometry};
            overlayGeometry->setAttribute("color", threepp::FloatBufferAttribute::create(std::move(shade), 3));

            const auto meadowSet = loadPbr(data_, "forrest_ground_01", SIZE / 2.4f);
            const auto gravelSet = loadPbr(data_, "forest_ground_04", SIZE / 2.0f);
            const auto graniteSet = loadPbr(data_, "mossy_rock", SIZE / 3.5f);

            auto meadowMat = pbrMaterial(meadowSet);
            meadowMat->vertexColors = true;
            auto ground = threepp::Mesh::create(geometry, meadowMat);
            ground->receiveShadow = true;
            root_->add(ground);

            auto [gravelMask, graniteMask] = bakeMasks();
            int order = 1;
            for (const auto& [set, mask] : {std::pair{gravelSet, gravelMask}, std::pair{graniteSet, graniteMask}}) {
                auto mat = pbrMaterial(set);
                mat->vertexColors = true;
                mat->alphaMap = mask;
                mat->transparent = true;
                mat->depthWrite = false;
                auto layer = threepp::Mesh::create(overlayGeometry, mat);
                layer->receiveShadow = true;
                layer->renderOrder = order++;
                root_->add(layer);
            }
        }

        // Coverage masks over the whole terrain (alphaMap reads G): the gravel path,
        // and granite where the ground is steep.
        std::pair<std::shared_ptr<threepp::Texture>, std::shared_ptr<threepp::Texture>> bakeMasks() const {
            constexpr int M = 512;
            std::vector<unsigned char> gravel(M * M * 4, 255), granite(M * M * 4, 255);
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < M; ++i) {
                    const float x = X0 + (i + 0.5f) / M * SIZE;
                    const float y = Y0 + (j + 0.5f) / M * SIZE;
                    // The stair, its walls and the terrace edge carry their own stone.
                    const bool built = x > STAIR_X0 - 0.3f && x < STAIR_BACK + 0.3f && std::abs(y) < TERRACE_HALF_Y + TERRACE_SIDE + 0.5f;
                    float g = 0.f;
                    if (!(x > STAIR_X0 - 0.3f && x < STAIR_BACK + 0.2f && std::abs(y) < STAIR_HALF_W + 0.6f)) {
                        const float d = distanceToPolyline(path_, x, y) + 0.22f * fbm(x / 1.3f, y / 1.3f, 2, 71u);
                        g = 1.f - smoothstep(0.55f, 0.95f, d);
                        // the rubble patch lies in loose gravel
                        const float rx = std::max({RUBBLE_X0 - x, x - RUBBLE_X1, 0.f});
                        const float ry = std::max(std::abs(y) - RUBBLE_HALF_Y, 0.f);
                        g = std::max(g, 1.f - smoothstep(0.2f, 0.9f, std::hypot(rx, ry) + 0.2f * fbm(x / 0.9f, y / 0.9f, 2, 73u)));
                    }
                    // Slope of the smooth terrain (no terrace step: that is under the wall).
                    const float e = 0.3f;
                    const auto hs = [](float px, float py) { return meadow(px, py) + rim(px, py) + terrace(px, py); };
                    const float sx = (hs(x + e, y) - hs(x - e, y)) / (2 * e);
                    const float sy = (hs(x, y + e) - hs(x, y - e)) / (2 * e);
                    const float r = built ? 0.f : smoothstep(0.38f, 0.62f, std::hypot(sx, sy) + 0.12f * fbm(x / 4.f, y / 4.f, 3, 79u));
                    gravel[(j * M + i) * 4 + 1] = static_cast<unsigned char>(std::lround(255.f * g));
                    granite[(j * M + i) * 4 + 1] = static_cast<unsigned char>(std::lround(255.f * r));
                }
            }
            const auto make = [](std::vector<unsigned char>&& px) {
                auto t = threepp::DataTexture::create(threepp::ImageData{std::move(px)}, M, M);
                t->magFilter = threepp::Filter::Linear;
                t->minFilter = threepp::Filter::Linear;
                t->needsUpdate();
                return t;
            };
            return {make(std::move(gravel)), make(std::move(granite))};
        }

        // ── The stair, its cheek walls and the terrace's front wall ─────────
        void buildStairAndWalls() {
            std::shared_ptr<threepp::MeshStandardMaterial> granite, wall;
            if (visuals_) {
                granite = pbrMaterial(loadPbr(data_, "mossy_rock", 1.f));
                granite->color = threepp::Color(0.92f, 0.92f, 0.9f);
                wall = pbrMaterial(loadPbr(data_, "rock_wall_10", 1.f), 1.3f);
            }
            // Each step and each coping slab is its own stone: cut from its own part of
            // the texture and a shade lighter or darker than its neighbours, as quarried
            // slabs are. One continuous texture made the stair read as a single block.
            Rand rnd(20261117u);
            const auto stone = [&] {
                const threepp::Vector2 offset(rnd.next(), rnd.next());
                if (!visuals_) return std::pair{std::shared_ptr<threepp::Material>(), offset};
                auto m = granite->clone<threepp::MeshStandardMaterial>();
                const float shade = rnd.range(0.9f, 1.07f), warm = rnd.range(-0.02f, 0.02f);
                m->color.setRGB(granite->color.r * shade * (1.f + warm), granite->color.g * shade, granite->color.b * shade * (1.f - warm));
                return std::pair{std::shared_ptr<threepp::Material>(m), offset};
            };

            constexpr float bottom = -0.4f;
            const float width = 2 * STAIR_HALF_W;
            for (int k = 1; k <= STEPS; ++k) {
                const float x0 = STAIR_X0 + (k - 1) * TREAD;
                const float top = k * RISE;
                const auto [mat, offset] = stone();
                staticBox(STAIR_BACK - x0, width, top - bottom,
                          {(x0 + STAIR_BACK) / 2, 0.f, (top + bottom) / 2}, mat, 1.4f, offset);
            }

            constexpr float coping = 0.07f, overhang = 0.04f;
            const auto wallSegment = [&](float x0, float x1, float y0, float y1, float top) {
                const float sx = x1 - x0, sy = y1 - y0;
                staticBox(sx, sy, top - bottom, {(x0 + x1) / 2, (y0 + y1) / 2, (top + bottom) / 2}, wall, 1.6f);
                const auto [mat, offset] = stone();
                staticBox(sx + 2 * overhang, sy + 2 * overhang, coping,
                          {(x0 + x1) / 2, (y0 + y1) / 2, top + coping / 2}, mat, 1.4f, offset);
            };

            // Cheek walls either side of the stair, flush with the terrace top.
            for (const float s : {-1.f, 1.f}) {
                const float y0 = s > 0 ? STAIR_HALF_W : -STAIR_HALF_W - WALL_T;
                wallSegment(STAIR_X0, STAIR_BACK, y0, y0 + WALL_T, TERRACE_H);
            }
            // The front wall, in 0.75 m courses whose tops follow the terrace as it
            // falls away to either side.
            constexpr float course = 0.75f;
            for (const float s : {-1.f, 1.f}) {
                for (float a = STAIR_HALF_W + WALL_T; a < TERRACE_HALF_Y + TERRACE_SIDE; a += course) {
                    const float b = a + course;
                    float top = 0.f;
                    for (int q = 0; q <= 4; ++q) {
                        const float y = s * (a + (b - a) * q / 4.f);
                        top = std::max(top, terrainHeight(STAIR_X0 + WALL_T + 0.25f, y));
                    }
                    if (top < 0.12f) break;
                    const float y0 = s > 0 ? a : -b, y1 = s > 0 ? b : -a;
                    wallSegment(STAIR_X0, STAIR_X0 + WALL_T, y0, y1, top - coping * 0.5f);
                }
            }
        }

        // ── Rocks ───────────────────────────────────────────────────────────
        struct RockProto {
            std::shared_ptr<threepp::BufferGeometry> geometry;// Z-up, centred, standing on z = 0
            std::shared_ptr<threepp::Material> material;
            ::physx::PxConvexMesh* hull = nullptr;
            float height = 1.f;
            float radius = 1.f;// footprint half-extent
        };

        void buildRocks() {
            threepp::GLTFLoader loader;
            auto gltf = loader.load(data_ + "/models/gltf/rock_moss_set_02/rock_moss_set_02_1k.gltf");
            if (!gltf) throw std::runtime_error("SpotCourse: could not load rock_moss_set_02");

            std::vector<RockProto> protos;
            gltf->scene->updateMatrixWorld(true);
            gltf->scene->traverseType<threepp::Mesh>([&](threepp::Mesh& m) {
                auto g = m.geometry()->clone();
                // The node's own rotation and scale stand each rock up as the set lays it
                // out; drop its translation, then recentre on its footprint, Z-up.
                threepp::Vector3 t, s;
                threepp::Quaternion q;
                m.matrixWorld->decompose(t, q, s);
                threepp::Matrix4 rs;
                rs.compose(threepp::Vector3(), q, s);
                g->applyMatrix4(rs);
                g->rotateX(threepp::math::PI / 2);
                g->computeBoundingBox();
                const auto& bb = *g->boundingBox;
                g->translate(-(bb.min().x + bb.max().x) / 2, -(bb.min().y + bb.max().y) / 2, -bb.min().z);
                g->computeBoundingBox();
                RockProto p;
                p.height = g->boundingBox->max().z;
                p.radius = std::max(g->boundingBox->max().x, g->boundingBox->max().y);
                const auto* posAttr = g->getAttribute<float>("position");
                p.hull = world_.cookConvexHull(posAttr->array().data(), posAttr->count());
                p.geometry = g;
                p.material = m.material();
                if (p.hull) protos.push_back(std::move(p));
            });
            if (protos.empty()) throw std::runtime_error("SpotCourse: no rock meshes in rock_moss_set_02");

            Rand rnd(20261116u);
            auto& placed = placed_;// x, y, radius; the ground's contact shade reads it too
            const auto clear = [&](float x, float y, float r) {
                for (const auto& p : placed) {
                    if (std::hypot(p.x - x, p.y - y) < p.z + r) return false;
                }
                return true;
            };
            const auto onPath = [&](float x, float y, float margin) {
                return distanceToPolyline(path_, x, y) < 0.9f + margin;
            };
            const auto nearBuilt = [](float x, float y, float margin) {
                return x > STAIR_X0 - 1.f - margin && x < STAIR_BACK + 1.f + margin && std::abs(y) < TERRACE_HALF_Y + TERRACE_SIDE + margin;
            };

            // Boulders over the meadow and the lower hills.
            for (int tries = 0, n = 0; n < 22 && tries < 4000; ++tries) {
                const float a = rnd.range(0.f, 2 * threepp::math::PI), r = 4.f + 30.f * std::sqrt(rnd.next());
                const float x = 12.f + std::cos(a) * r * 1.15f, y = std::sin(a) * r;
                const float s = rnd.range(0.55f, 1.5f);
                if (std::hypot(x, y) < 4.f || onPath(x, y, 0.8f * s) || nearBuilt(x, y, 1.f) || !clear(x, y, 1.2f * s)) continue;
                placeRock(protos[rnd.next() * protos.size()], x, y, s, rnd.range(0.12f, 0.3f), 0.15f, rnd);
                placed.emplace_back(x, y, 1.2f * s);
                ++n;
            }
            // Large rocks out on the rim, for the skyline.
            for (int tries = 0, n = 0; n < 9 && tries < 2000; ++tries) {
                const float a = rnd.range(0.f, 2 * threepp::math::PI), r = rnd.range(30.f, 40.f);
                const float x = 12.f + std::cos(a) * r * 1.15f, y = std::sin(a) * r;
                if (x < X0 + 4 || x > X0 + SIZE - 4 || std::abs(y) > SIZE / 2 - 4) continue;
                const float s = rnd.range(1.8f, 3.2f);
                if (!clear(x, y, 1.3f * s)) continue;
                placeRock(protos[rnd.next() * protos.size()], x, y, s, rnd.range(0.15f, 0.3f), 0.2f, rnd);
                placed.emplace_back(x, y, 1.3f * s);
                ++n;
            }
            // Stones heaped along the foot of the terrace wall.
            for (int tries = 0, n = 0; n < 10 && tries < 1000; ++tries) {
                const float y = rnd.range(STAIR_HALF_W + WALL_T + 0.6f, TERRACE_HALF_Y + 2.f) * (rnd.next() < 0.5f ? -1.f : 1.f);
                const float x = STAIR_X0 - rnd.range(0.1f, 0.6f);
                const float s = rnd.range(0.25f, 0.5f);
                if (!clear(x, y, 0.6f * s)) continue;
                placeRock(protos[rnd.next() * protos.size()], x, y, s, rnd.range(0.2f, 0.35f), 0.2f, rnd);
                placed.emplace_back(x, y, 0.6f * s);
                ++n;
            }
            // A few on the terrace, off the path.
            for (int tries = 0, n = 0; n < 6 && tries < 1000; ++tries) {
                const float x = rnd.range(STAIR_BACK + 1.f, TERRACE_END), y = rnd.range(2.8f, TERRACE_HALF_Y - 0.5f) * (rnd.next() < 0.5f ? -1.f : 1.f);
                const float s = rnd.range(0.45f, 0.9f);
                if (onPath(x, y, 0.6f * s) || !clear(x, y, 1.1f * s)) continue;
                placeRock(protos[rnd.next() * protos.size()], x, y, s, rnd.range(0.15f, 0.3f), 0.15f, rnd);
                placed.emplace_back(x, y, 1.1f * s);
                ++n;
            }
            // The rubble patch: stones a hand to a shin high, across the path. The
            // policy walks it; nothing here is taller than its trained risers.
            for (int tries = 0, n = 0; n < 34 && tries < 3000; ++tries) {
                const float x = rnd.range(RUBBLE_X0, RUBBLE_X1), y = rnd.range(-RUBBLE_HALF_Y, RUBBLE_HALF_Y);
                const float s = rnd.range(0.12f, 0.24f);
                if (!clear(x, y, 0.45f * s)) continue;
                placeRock(protos[rnd.next() * protos.size()], x, y, s, rnd.range(0.3f, 0.45f), 0.3f, rnd);
                placed.emplace_back(x, y, 0.45f * s);
                ++n;
            }
            for (auto& p : protos) p.hull->release();// each shape holds its own reference
        }

        void placeRock(const RockProto& p, float x, float y, float scale, float sink, float tilt, Rand& rnd) {
            using namespace ::physx;
            threepp::Quaternion qYaw, qTilt;
            qYaw.setFromAxisAngle(threepp::Vector3(0, 0, 1), rnd.range(0.f, 2 * threepp::math::PI));
            const float ta = rnd.range(0.f, 2 * threepp::math::PI);
            qTilt.setFromAxisAngle(threepp::Vector3(std::cos(ta), std::sin(ta), 0), rnd.range(-tilt, tilt));
            threepp::Quaternion q;
            q.multiplyQuaternions(qTilt, qYaw);
            const float sz = rnd.range(0.75f, 1.05f);// flatter or rounder, for variety
            const threepp::Vector3 scl(scale, scale * rnd.range(0.85f, 1.15f), scale * sz);
            // Seat it on the lowest ground under its footprint, so on a slope the
            // downhill side is buried rather than hanging in the air.
            float ground = terrainHeight(x, y);
            const float reach = 0.7f * p.radius * scale;
            for (int k = 0; k < 8; ++k) {
                const float a = k * threepp::math::PI / 4;
                ground = std::min(ground, terrainHeight(x + reach * std::cos(a), y + reach * std::sin(a)));
            }
            const float z = ground - sink * p.height * scl.z;

            const PxConvexMeshGeometry geom(p.hull, PxMeshScale(PxVec3(scl.x, scl.y, scl.z)));
            world_.addStatic(geom, PxTransform(PxVec3(x, y, z), PxQuat(q.x, q.y, q.z, q.w)));
            if (!visuals_) return;
            auto mesh = threepp::Mesh::create(p.geometry, p.material);
            mesh->position.set(x, y, z);
            mesh->quaternion.copy(q);
            mesh->scale.copy(scl);
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            root_->add(mesh);
        }
    };

}// namespace spot::course

#endif// THREEPP_EXAMPLES_SPOT_COURSE_HPP
