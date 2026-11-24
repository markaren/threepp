// The watermill's ground and stone: the valley's terrain (dense along the stream, coarser up the valley sides),
// the boulders in and along the stream (each one unique, built from a displaced sphere: a rounded block with fracture
// faces, bedding and a flat foot, coloured where it stands: dark and wet under the water and just above it, moss
// on the tops of the bank stones), and the gravel between them.

#ifndef THREEPP_EXAMPLE_WATERMILL_TERRAIN_HPP
#define THREEPP_EXAMPLE_WATERMILL_TERRAIN_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/IcosahedronGeometry.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"

#include "Site.hpp"
#include "TextureLab.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>

namespace mill {

    using threepp::BufferGeometry;
    using threepp::FloatBufferAttribute;

    // ---- the terrain ------------------------------------------------------------------------------------------------
    // grid lines: 0.25 m apart inside [a, b], spreading out to `step` m toward the ends
    inline std::vector<float> gridLines(float lo, float a, float b, float hi, float fine, float coarse) {
        std::vector<float> out;
        float x = a;
        // outward from a toward lo
        std::vector<float> left;
        float s = fine;
        while (x > lo) {
            x -= s;
            left.push_back(std::max(x, lo));
            s = std::min(coarse, s * 1.06f);
        }
        std::reverse(left.begin(), left.end());
        out = left;
        for (x = a; x < b; x += fine) out.push_back(x);
        x = b;
        s = fine;
        while (x < hi) {
            out.push_back(x);
            x += s;
            s = std::min(coarse, s * 1.06f);
        }
        out.push_back(hi);
        out.erase(std::unique(out.begin(), out.end(), [](float p, float q) { return std::abs(p - q) < 1e-4f; }), out.end());
        return out;
    }

    // colour of the ground at a point, by what it is
    inline V3 groundColour(float x, float z, float h, float ny) {
        const float water = site::bedAt(x, z) + 0.3f;
        const float above = h - water;
        const float mottle = fbm2(x * 0.35f, z * 0.35f, 4, 21u);
        const float fine = fbm2(x * 2.1f, z * 2.1f, 2, 22u);
        // the litter is the texture's own colour: the vertex colour tints it
        // the litter is darker back under the trees than by the open creek
        float lcx, lhw;
        site::streamAt(z, lcx, lhw);
        V3 litter = V3(1.f, 1.f, 1.f) * (0.85f + 0.25f * fine) * (0.62f + 0.38f * smooth(9.f, 2.f, std::abs(x - lcx) - lhw));
        const V3 moss = srgb(150, 178, 92);
        const V3 grass = srgb(160, 196, 104);
        V3 c = litter;
        // moss and grass in patches, more of it where it is flat and where the sun comes through over the stream
        float cx, hw;
        site::streamAt(z, cx, hw);
        const float open = smooth(14.f, 4.f, std::abs(x - cx) - hw);
        const float greenness = clamp01(0.25f + 0.6f * mottle + 0.35f * open) * smooth(0.55f, 0.85f, ny);
        c = mix(c, mix(moss, grass, clamp01(0.5f + fine)), 0.75f * greenness);
        // steep banks: moss over dark earth, the earth showing where the moss breaks
        const float steep = smooth(0.8f, 0.5f, ny);
        c = mix(c, mix(srgb(120, 100, 84), srgb(120, 150, 70), smooth(-0.2f, 0.3f, mottle + 0.5f * fine)), 0.7f * steep);
        // the damp margin, then the wet bank
        c = mix(c, srgb(110, 96, 80), 0.6f * smooth(0.6f, 0.1f, above));
        // a steep face in the channel is the sandstone the creek has cut down to: dark and wet
        float ccx, chw;
        site::streamAt(z, ccx, chw);
        if (std::abs(x - ccx) < chw + 1.f && ny < 0.6f && above < 0.4f) return srgb(70, 62, 52) * (0.8f + 0.3f * fine);
        // under the water: sand and fine gravel, tea-stained
        if (above < 0.02f) {
            V3 bed = mix(srgb(190, 168, 132), srgb(150, 130, 100), clamp01(0.5f + mottle));
            c = mix(c, bed, smooth(0.02f, -0.06f, above));
        }
        return c;
    }

    inline std::shared_ptr<threepp::Mesh> makeTerrain(const std::shared_ptr<threepp::Material>& material) {
        const auto xs = gridLines(-95.f, -26.f, 22.f, 95.f, 0.25f, 3.f);
        const auto zs = gridLines(-150.f, -58.f, 60.f, 100.f, 0.25f, 3.f);
        const int nx = static_cast<int>(xs.size()), nz = static_cast<int>(zs.size());
        const size_t n = static_cast<size_t>(nx) * nz;
        std::vector<float> pos(n * 3), nor(n * 3), col(n * 3), uv(n * 2);
        forRows(nz, [&](int j) {
            for (int i = 0; i < nx; ++i) {
                const float x = xs[i], z = zs[j];
                const float h = site::groundHeight(x, z);
                const float e = 0.12f;
                const float gx = (site::groundHeight(x + e, z) - site::groundHeight(x - e, z)) / (2 * e);
                const float gz = (site::groundHeight(x, z + e) - site::groundHeight(x, z - e)) / (2 * e);
                const float len = std::sqrt(gx * gx + gz * gz + 1.f);
                const size_t c = static_cast<size_t>(j) * nx + i;
                pos[3 * c] = x;
                pos[3 * c + 1] = h;
                pos[3 * c + 2] = z;
                nor[3 * c] = -gx / len;
                nor[3 * c + 1] = 1.f / len;
                nor[3 * c + 2] = -gz / len;
                const V3 k = groundColour(x, z, h, 1.f / len);
                col[3 * c] = k.r;
                col[3 * c + 1] = k.g;
                col[3 * c + 2] = k.b;
                // the texture laid along the slope, not squeezed on it: shift by the height along the downhill direction
                uv[2 * c] = (x + h * gx / len) / 3.f;
                uv[2 * c + 1] = (z + h * gz / len) / 3.f;
            }
        });
        std::vector<unsigned int> index;
        index.reserve(static_cast<size_t>(nx - 1) * (nz - 1) * 6);
        for (int j = 0; j + 1 < nz; ++j) {
            for (int i = 0; i + 1 < nx; ++i) {
                const unsigned a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
                // under the mill's basement floor nothing shows
                const float mx = 0.5f * (xs[i] + xs[i + 1]), mz = 0.5f * (zs[j] + zs[j + 1]);
                if (site::inMill(mx, mz, -0.3f)) continue;
                index.insert(index.end(), {a, c, b, b, c, d});
            }
        }
        auto g = BufferGeometry::create();
        g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        g->setAttribute("normal", FloatBufferAttribute::create(nor, 3));
        g->setAttribute("color", FloatBufferAttribute::create(col, 3));
        g->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        g->setIndex(index);
        auto mesh = threepp::Mesh::create(g, material);
        mesh->receiveShadow = true;
        mesh->castShadow = true;
        return mesh;
    }

    // ---- boulders ---------------------------------------------------------------------------------------------------
    struct Rock {
        float x, y, z;   // the centre
        float r;         // horizontal radius
        float squash;    // height / r
        float elong;     // along / across
        float yaw, tilt;
        uint32_t seed;
        V3 tint;
        int detail;
    };

    inline bool rockExcluded(float x, float z, float r) {
        // the bridge's piers, the wheel's pit and pier, the mill
        for (float px : site::bridgePiers) {
            if (std::abs(x - px) < site::pierHalfX + r + 0.15f && std::abs(z - site::bridgeZ) < site::pierHalfZ + r + 0.25f) return true;
        }
        if (x > site::pierX0 - r - 0.6f && x < site::millWest + 0.2f && z > site::wheelZ - site::wheelRadius - 0.3f - r && z < site::wheelZ + site::wheelRadius + 0.3f + r) return true;
        if (site::inMill(x, z, r + 0.3f)) return true;
        return false;
    }

    inline std::vector<Rock> placeRocks() {
        std::vector<Rock> rocks;
        Rng rng(2024);
        const std::array<V3, 6> tints{srgb(178, 160, 136), srgb(160, 150, 136), srgb(186, 166, 140), srgb(150, 138, 124), srgb(170, 150, 120), srgb(140, 136, 130)};
        const auto fits = [&](float x, float z, float r) {
            if (rockExcluded(x, z, r)) return false;
            for (const auto& o : rocks) {
                const float d = std::hypot(x - o.x, z - o.z);
                if (d < 0.82f * (r + o.r)) return false;
            }
            return true;
        };
        const auto add = [&](float x, float z, float r, float bury, int detail) {
            Rock k{};
            k.x = x;
            k.z = z;
            k.r = r;
            k.squash = rng.range(0.42f, 0.75f);
            k.elong = rng.range(1.0f, 1.5f);
            k.yaw = rng() * 6.2832f;
            k.tilt = rng.range(-0.12f, 0.12f);
            k.seed = rng.next();
            k.tint = tints[rng.next() % tints.size()] * rng.range(1.0f, 1.18f);
            k.detail = detail;
            k.y = site::groundHeight(x, z) + k.squash * r * (0.45f - bury);
            rocks.push_back(k);
        };
        // the hero slabs in the foreground and the ledges' big blocks first
        const std::array<std::array<float, 3>, 9> hero{{{-3.6f, 24.5f, 1.5f}, {-0.6f, 21.f, 1.1f}, {-6.2f, 21.5f, 1.3f}, {2.2f, 25.5f, 1.2f}, {-2.8f, 17.6f, 0.9f}, {-5.5f, 26.8f, 1.4f}, {6.0f, 21.5f, 1.0f}, {-7.8f, 17.f, 1.1f}, {0.8f, 13.0f, 0.8f}}};
        for (const auto& h : hero) add(h[0], h[1], h[2], 0.2f, 6);
        // the sandstone outcrop on the right bank below the mill: thick slabs in ledges, each stepped back from the
        // one below, the bank's soil behind them
        for (float z = 8.5f; z < 36.f; z += rng.range(2.4f, 3.4f)) {
            float cx, hw;
            site::streamAt(z, cx, hw);
            const float edge = cx + hw;
            const float water = site::bedAt(cx + hw, z) + 0.3f;
            const int layers = 3 + static_cast<int>(rng() * 2.f);
            for (int k = 0; k < layers; ++k) {
                Rock s{};
                s.r = rng.range(1.5f, 2.3f);
                s.squash = rng.range(0.3f, 0.42f);
                s.elong = rng.range(1.3f, 1.8f);
                s.yaw = 1.5708f + rng.range(-0.3f, 0.3f);
                s.tilt = rng.range(-0.06f, 0.06f);
                s.x = edge + 1.0f + 1.05f * static_cast<float>(k) + rng.range(-0.35f, 0.35f);
                s.z = z + rng.range(-0.7f, 0.7f);
                s.y = water - 0.25f + 0.92f * static_cast<float>(k) + rng.range(-0.1f, 0.1f);
                s.seed = rng.next();
                s.tint = tints[rng.next() % tints.size()] * rng.range(0.85f, 1.0f);
                s.detail = 6;
                rocks.push_back(s);
            }
        }
        const int edges = 2 + static_cast<int>(site::creekSteps().size());
        for (int which = 0; which < edges; ++which) {
            const float z0 = which == 0 ? site::ledgeUpper : which == 1 ? site::ledgeLower : site::creekSteps()[which - 2].z;
            float cx, hw;
            site::streamAt(z0, cx, hw);
            // the boulders the step is jammed with, and its sandstone bed's broken edge
            for (int k = 0; k < (which < 2 ? 400 : 160) && rocks.size() < 2000; ++k) {
                const float x = cx + rng.range(-hw - 0.5f, hw + 0.5f), r = rng.range(0.35f, 1.0f);
                const float edgeZ = which == 0 ? site::upperLedgeZ(x) : which == 1 ? site::lowerLedgeZ(x) : site::stepEdgeZ(site::creekSteps()[which - 2], x);
                const float z = edgeZ + rng.range(-1.2f, 0.6f);
                if (fits(x, z, r)) add(x, z, r, 0.3f, r > 0.7f ? 5 : 4);
            }
        }
        // the channel: a power-law of sizes, most of them small
        // the falls' lip: a hard sandstone layer in slabs, with gaps where the creek goes over in chutes
        {
            float cx, hw;
            site::streamAt(site::fallsZ, cx, hw);
            // the chutes the creek has cut through the lip layer: the main one in the bay, two smaller ones
            const std::array<std::array<float, 2>, 3> chutes{{{cx + 1.6f, 1.9f}, {cx - 3.6f, 1.1f}, {cx + 5.4f, 0.9f}}};
            const auto inChute = [&](float a, float b) {
                for (const auto& c : chutes)
                    if (b > c[0] - 0.5f * c[1] && a < c[0] + 0.5f * c[1]) return true;
                return false;
            };
            float x = cx - hw - 2.2f;
            while (x < cx + hw + 2.2f) {
                Rock s{};
                const bool bank = x < cx - hw + 0.5f || x > cx + hw - 0.5f;
                s.r = rng.range(0.9f, 1.5f);
                s.squash = rng.range(0.22f, 0.3f);
                s.elong = rng.range(1.2f, 1.7f);
                s.yaw = rng.range(-0.25f, 0.25f);
                s.tilt = rng.range(-0.04f, 0.04f);
                s.x = x + s.r * s.elong;
                if (!bank && inChute(x, x + 2.f * s.r * s.elong)) {
                    // leave the chute open: step past it
                    x += 0.35f;
                    continue;
                }
                s.z = site::fallsLipZ(s.x) - 0.62f * s.r + rng.range(-0.15f, 0.1f);
                s.y = site::bedAt(s.x, site::fallsLipZ(s.x) - 1.f) + (bank ? 0.25f : 0.02f) - 0.1f * rng();
                s.seed = rng.next();
                s.tint = tints[rng.next() % tints.size()] * rng.range(0.85f, 1.0f);
                s.detail = 6;
                rocks.push_back(s);
                x += 2.f * s.r * s.elong + (bank ? rng.range(-0.3f, 0.1f) : rng.range(-0.25f, 0.15f));
            }
            // the face of the drop: sandstone beds, each a little behind the one above, the sheet falling in front
            for (float fxp = cx - hw - 1.5f; fxp < cx + hw + 1.5f;) {
                for (int layer = 0; layer < 2; ++layer) {
                    Rock s{};
                    s.r = rng.range(0.9f, 1.3f);
                    s.squash = rng.range(0.24f, 0.3f);
                    s.elong = rng.range(1.3f, 1.7f);
                    s.yaw = rng.range(-0.12f, 0.12f);
                    s.tilt = rng.range(-0.03f, 0.03f);
                    s.x = fxp + s.r * s.elong + rng.range(-0.3f, 0.3f);
                    s.z = site::fallsLipZ(s.x) - 0.62f * s.r - 0.08f * static_cast<float>(layer);
                    const float lipBed = site::bedAt(s.x, site::fallsLipZ(s.x) - 1.f);
                    s.y = lipBed - 0.62f - 0.6f * static_cast<float>(layer);
                    s.seed = rng.next();
                    s.tint = tints[rng.next() % tints.size()] * rng.range(0.7f, 0.85f);// wet and dark
                    s.detail = 5;
                    rocks.push_back(s);
                }
                fxp += rng.range(1.6f, 2.4f);
            }
            // broad flat ledges on both banks below the lip, where people stand to look at the falls
            for (int k = 0; k < 16; ++k) {
                const float side = k % 2 ? 1.f : -1.f;
                Rock s{};
                s.r = rng.range(1.4f, 2.4f);
                s.squash = rng.range(0.2f, 0.3f);
                s.elong = rng.range(1.2f, 1.8f);
                s.yaw = 1.5708f + rng.range(-0.4f, 0.4f);
                s.tilt = rng.range(-0.05f, 0.05f);
                s.z = site::fallsZ + rng.range(0.8f, 14.f);
                float ccx, chw;
                site::streamAt(s.z, ccx, chw);
                s.x = ccx + side * (chw + rng.range(-1.f, 1.5f));
                s.y = site::groundHeight(s.x, s.z) + 0.05f;
                s.seed = rng.next();
                s.tint = tints[rng.next() % tints.size()] * rng.range(0.85f, 1.0f);
                s.detail = 6;
                rocks.push_back(s);
            }
        }
        for (int k = 0; k < 60000 && rocks.size() < 900; ++k) {
            const float z = rng.range(-62.f, 54.f);
            if (std::abs(z - site::fallsZ) < 2.f) continue;
            float cx, hw;
            site::streamAt(z, cx, hw);
            const float side = rng.range(-1.15f, 1.15f);
            const float x = cx + side * hw;
            const float r = std::min(1.3f, 0.14f * std::pow(1.f - rng() * 0.98f, -1.f / 1.5f));
            // fewer in the middle of the pool, more near the banks
            if (site::poolDepth(x, z) > 0.3f && rng() < 0.8f) continue;
            if (std::abs(side) < 0.5f && rng() < 0.35f) continue;
            if (!fits(x, z, r)) continue;
            add(x, z, r, rng.range(0.f, 0.35f), r > 0.8f ? 5 : (r > 0.35f ? 4 : 3));
        }
        // the banks: larger, deeper in the ground
        for (int k = 0; k < 20000 && rocks.size() < 1250; ++k) {
            const float z = rng.range(-62.f, 54.f);
            float cx, hw;
            site::streamAt(z, cx, hw);
            const float side = rng() < 0.5f ? -1.f : 1.f;
            const float x = cx + side * (hw + rng.range(0.5f, 6.f));
            const float r = rng.range(0.3f, 1.4f);
            if (!fits(x, z, r)) continue;
            add(x, z, r, rng.range(0.2f, 0.45f), r > 0.8f ? 5 : 4);
        }
        return rocks;
    }

    // the unit rock's radius toward a direction (local frame): a rounded block, cut by a few fracture planes, with
    // bedding grooves and a flat foot
    struct RockShape {
        std::array<threepp::Vector3, 5> cutNormal;
        std::array<float, 5> cutOffset;
        uint32_t seed;

        explicit RockShape(uint32_t s): seed(s) {
            Rng rng(s);
            for (int k = 0; k < 5; ++k) {
                threepp::Vector3 n(rng.normal(), rng.normal() * 0.6f + (k == 0 ? 1.2f : 0.f), rng.normal());
                n.normalize();
                cutNormal[k] = n;
                cutOffset[k] = rng.range(0.72f, 0.92f);
            }
        }

        [[nodiscard]] float radius(const threepp::Vector3& d) const {
            const float e = 3.0f;
            float r = 1.f / std::pow(std::pow(std::abs(d.x), e) + std::pow(std::abs(d.y), e) + std::pow(std::abs(d.z), e), 1.f / e);
            r = 0.55f * r + 0.45f;// round the box off
            for (int k = 0; k < 5; ++k) {
                const float c = d.dot(cutNormal[k]);
                if (c > 0.05f) {
                    const float plane = cutOffset[k] / c;
                    // a soft minimum, so the cut's edge is rounded by the water
                    const float kk = 0.08f;
                    const float hmin = std::clamp(0.5f + 0.5f * (plane - r) / kk, 0.f, 1.f);
                    r = plane + (r - plane) * hmin - kk * hmin * (1 - hmin);
                }
            }
            r *= 1.f + 0.14f * fbm3(d.x * 1.2f, d.y * 1.2f, d.z * 1.2f, 4, seed) + 0.035f * fbm3(d.x * 5.f, d.y * 5.f, d.z * 5.f, 3, seed + 5);
            // bedding: shallow grooves along the local y
            const float bed = std::sin(d.y * r * 22.f + 3.f * fbm3(d.x * 2.f, d.y * 2.f, d.z * 2.f, 2, seed + 9));
            r *= 1.f - 0.012f * smooth(0.6f, 1.f, bed);
            return r;
        }
    };

    struct RockBatch {
        std::vector<float> pos, nor, col, uv;
        std::vector<unsigned int> index;
    };

    inline void appendRock(RockBatch& out, const Rock& k) {
        using threepp::Vector3;
        // detail is a level: 2 gravel .. 6 a hero slab (the icosahedron's own detail is linear in the edge splits)
        static constexpr std::array<unsigned, 7> splits{1, 2, 3, 5, 10, 17, 26};
        auto ico = threepp::IcosahedronGeometry::create(1, splits[std::clamp(k.detail, 0, 6)]);
        // IcosahedronGeometry is non-indexed: weld its vertices first
        const auto& p = ico->getAttribute<float>("position")->array();
        std::vector<Vector3> verts;
        std::vector<unsigned int> tri;
        {
            std::unordered_map<long long, unsigned int> seen;
            for (size_t i = 0; i < p.size(); i += 3) {
                const auto q = [](float f) { return static_cast<long long>(std::lround(f * 100000.f) + (1 << 20)) & 0x1FFFFF; };
                const long long key = q(p[i]) | (q(p[i + 1]) << 21) | (q(p[i + 2]) << 42);
                auto it = seen.find(key);
                if (it == seen.end()) {
                    it = seen.emplace(key, static_cast<unsigned int>(verts.size())).first;
                    verts.emplace_back(p[i], p[i + 1], p[i + 2]);
                }
                tri.push_back(it->second);
            }
        }
        const RockShape shape(k.seed);
        threepp::Matrix4 m, rot;
        m.makeRotationY(k.yaw);
        rot.makeRotationZ(k.tilt);
        m.multiply(rot);
        std::vector<Vector3> world(verts.size());
        const float foot = -0.55f;
        for (size_t i = 0; i < verts.size(); ++i) {
            Vector3 d = verts[i];
            d.normalize();
            Vector3 v = d * shape.radius(d);
            if (v.y < foot) v.y = foot + 0.25f * (v.y - foot);// the flat foot
            v.x *= k.r * k.elong;
            v.z *= k.r;
            v.y *= k.r * k.squash;
            v.applyMatrix4(m);
            world[i] = v + Vector3(k.x, k.y, k.z);
        }
        // smooth normals
        std::vector<Vector3> normals(verts.size(), Vector3(0, 0, 0));
        for (size_t t = 0; t < tri.size(); t += 3) {
            const auto &a = world[tri[t]], &b = world[tri[t + 1]], &c = world[tri[t + 2]];
            Vector3 n = (b - a).cross(c - a);
            normals[tri[t]] += n;
            normals[tri[t + 1]] += n;
            normals[tri[t + 2]] += n;
        }
        const unsigned base = static_cast<unsigned>(out.pos.size() / 3);
        const float water = site::bedAt(k.x, k.z) + 0.3f;
        for (size_t i = 0; i < verts.size(); ++i) {
            Vector3 n = normals[i];
            n.normalize();
            const auto& w = world[i];
            out.pos.insert(out.pos.end(), {w.x, w.y, w.z});
            out.nor.insert(out.nor.end(), {n.x, n.y, n.z});
            // box projection by the normal's main axis, 1.6 m a tile, offset per rock
            const float ox = unit(k.seed) * 7.f, oy = unit(hash32(k.seed)) * 7.f;
            float u, v;
            if (std::abs(n.y) > std::abs(n.x) && std::abs(n.y) > std::abs(n.z)) {
                u = w.x;
                v = w.z;
            } else if (std::abs(n.x) > std::abs(n.z)) {
                u = w.z;
                v = w.y;
            } else {
                u = w.x;
                v = w.y;
            }
            out.uv.insert(out.uv.end(), {u / 1.6f + ox, v / 1.6f + oy});
            // colour: the rock's own, darker and browner wet, algae under the water, moss on dry tops
            const float above = w.y - water;
            V3 c = k.tint * (0.85f + 0.3f * (0.5f + 0.5f * fbm3(w.x * 0.8f, w.y * 0.8f, w.z * 0.8f, 3, k.seed + 3)));
            const float wet = smooth(0.12f, -0.02f, above);
            c = mix(c, c * srgb(150, 130, 100) * 0.62f, wet);
            c = mix(c, srgb(84, 80, 52), 0.45f * smooth(-0.05f, -0.25f, above));
            // moss only on the tops of the stones the floods leave alone, dark olive and patchy, never lime
            const float mossy = smooth(0.55f, 0.95f, n.y) * smooth(0.45f, 1.1f, above) * smooth(0.1f, 0.6f, fbm3(w.x * 1.5f, w.y * 1.5f, w.z * 1.5f, 3, k.seed + 4));
            c = mix(c, srgb(52, 62, 32) * (0.85f + 0.3f * clamp01(0.5f + fbm3(w.x * 6.f, w.y * 6.f, w.z * 6.f, 2, k.seed + 5))), 0.6f * mossy);
            // the dark line where it meets the ground
            const float g = site::groundHeight(w.x, w.z);
            c = c * (0.55f + 0.45f * smooth(0.f, 0.12f, w.y - g));
            out.col.insert(out.col.end(), {c.r, c.g, c.b});
        }
        for (auto t : tri) out.index.push_back(base + t);
    }

    // the rocks as a few merged meshes, split by area so each can be culled
    // points: if given, every boulder's world vertices are appended to it (x, y, z)
    inline std::shared_ptr<threepp::Group> makeRocks(const std::vector<Rock>& rocks, const std::shared_ptr<threepp::Material>& material,
                                                     std::vector<float>* points = nullptr) {
        auto group = threepp::Group::create();
        std::map<int, RockBatch> batches;
        for (const auto& k : rocks) {
            const int key = static_cast<int>(std::floor(k.z / 12.f)) * 8 + static_cast<int>(std::floor(k.x / 12.f)) + 4;
            appendRock(batches[key], k);
        }
        for (auto& [key, b] : batches) {
            if (points) points->insert(points->end(), b.pos.begin(), b.pos.end());
            auto g = BufferGeometry::create();
            g->setAttribute("position", FloatBufferAttribute::create(b.pos, 3));
            g->setAttribute("normal", FloatBufferAttribute::create(b.nor, 3));
            g->setAttribute("color", FloatBufferAttribute::create(b.col, 3));
            g->setAttribute("uv", FloatBufferAttribute::create(b.uv, 2));
            g->setIndex(b.index);
            g->computeBoundingSphere();
            auto mesh = threepp::Mesh::create(g, material);
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            group->add(mesh);
        }
        return group;
    }

    // the gravel and cobbles between the boulders, instanced
    inline std::shared_ptr<threepp::Group> makeGravel(const std::vector<Rock>& rocks, const std::shared_ptr<threepp::Material>& material) {
        using threepp::Vector3;
        auto group = threepp::Group::create();
        std::vector<std::shared_ptr<BufferGeometry>> shapes;
        for (uint32_t s = 0; s < 4; ++s) {
            RockBatch b;
            Rock k{};
            k.r = 1.f;
            k.squash = 0.6f;
            k.elong = 1.2f;
            k.seed = 900 + s;
            k.tint = V3(1, 1, 1);
            k.detail = 2;
            k.y = 1000.f;// far above any water: plain colour
            appendRock(b, k);
            for (size_t i = 1; i < b.pos.size(); i += 3) b.pos[i] -= 1000.f;
            auto g = BufferGeometry::create();
            g->setAttribute("position", FloatBufferAttribute::create(b.pos, 3));
            g->setAttribute("normal", FloatBufferAttribute::create(b.nor, 3));
            g->setAttribute("color", FloatBufferAttribute::create(b.col, 3));
            g->setAttribute("uv", FloatBufferAttribute::create(b.uv, 2));
            g->setIndex(b.index);
            shapes.push_back(g);
        }
        Rng rng(77);
        const int perShape = 1100;
        for (size_t s = 0; s < shapes.size(); ++s) {
            auto inst = threepp::InstancedMesh::create(shapes[s], material, perShape);
            threepp::Matrix4 m, r;
            int placed = 0;
            for (int tries = 0; placed < perShape && tries < perShape * 20; ++tries) {
                const float z = rng.range(-60.f, 54.f);
                float cx, hw;
                site::streamAt(z, cx, hw);
                const float x = cx + rng.range(-1.1f, 1.1f) * hw;
                const float size = 0.04f + 0.12f * std::pow(rng(), 2.2f);
                if (rockExcluded(x, z, size)) continue;
                bool inside = false;
                for (const auto& k : rocks) {
                    if (std::abs(k.x - x) < k.r * k.elong && std::abs(k.z - z) < k.r * k.elong && std::hypot(k.x - x, k.z - z) < 0.9f * k.r) {
                        inside = true;
                        break;
                    }
                }
                if (inside) continue;
                const float y = site::groundHeight(x, z) + 0.15f * size;
                m.makeRotationY(rng() * 6.283f);
                r.makeRotationX(rng.range(-0.3f, 0.3f));
                m.multiply(r);
                m.scale(Vector3(size, size, size));
                m.setPosition(x, y, z);
                inst->setMatrixAt(placed, m);
                const bool wet = y < site::nominalWater(z) + 0.03f;
                V3 c = mix(srgb(176, 160, 136), srgb(140, 138, 132), rng()) * rng.range(0.75f, 1.1f);
                if (wet) c = c * V3(0.66f, 0.6f, 0.5f);
                inst->setColorAt(placed, threepp::Color(c.r, c.g, c.b));
                ++placed;
            }
            inst->setCount(placed);
            inst->castShadow = true;
            inst->receiveShadow = true;
            group->add(inst);
        }
        return group;
    }

}// namespace mill

#endif
