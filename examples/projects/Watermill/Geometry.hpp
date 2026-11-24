// Mesh building for the watermill's made things: an indexed mesh being assembled from pieces (MeshBuild), local
// frames, and the carpenter's stock: boxes, beams and boards with their grain along their length, round timbers.
// Wood is textured with the board tile of Textures.hpp: 0.32 m across the grain (u) by 1.28 m along it (v).

#ifndef THREEPP_EXAMPLE_WATERMILL_GEOMETRY_HPP
#define THREEPP_EXAMPLE_WATERMILL_GEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"

#include "Noise.hpp"
#include "TextureLab.hpp"

#include <array>
#include <functional>
#include <unordered_map>
#include <vector>

namespace mill {

    // an indexed mesh being built: positions, normals, uvs, colours
    struct MeshBuild {
        std::vector<float> pos, nor, uv, col;
        std::vector<unsigned int> index;

        [[nodiscard]] unsigned vertexCount() const { return static_cast<unsigned>(pos.size() / 3); }

        void vertex(const threepp::Vector3& p, const threepp::Vector3& n, float u, float v, const V3& c) {
            pos.insert(pos.end(), {p.x, p.y, p.z});
            nor.insert(nor.end(), {n.x, n.y, n.z});
            uv.insert(uv.end(), {u, v});
            col.insert(col.end(), {c.r, c.g, c.b});
        }

        void append(const MeshBuild& o) {
            const unsigned base = vertexCount();
            pos.insert(pos.end(), o.pos.begin(), o.pos.end());
            nor.insert(nor.end(), o.nor.begin(), o.nor.end());
            uv.insert(uv.end(), o.uv.begin(), o.uv.end());
            col.insert(col.end(), o.col.begin(), o.col.end());
            for (auto i : o.index) index.push_back(base + i);
        }

        // normals from the faces, area weighted
        void smoothNormals(unsigned from = 0) {
            using threepp::Vector3;
            const unsigned n = vertexCount();
            std::vector<Vector3> acc(n - from, Vector3(0, 0, 0));
            for (size_t t = 0; t < index.size(); t += 3) {
                const unsigned a = index[t], b = index[t + 1], c = index[t + 2];
                if (a < from || b < from || c < from) continue;
                const Vector3 pa(pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]);
                const Vector3 pb(pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]);
                const Vector3 pc(pos[3 * c], pos[3 * c + 1], pos[3 * c + 2]);
                const Vector3 fn = (pb - pa).cross(pc - pa);
                acc[a - from] += fn;
                acc[b - from] += fn;
                acc[c - from] += fn;
            }
            for (unsigned i = from; i < n; ++i) {
                Vector3 v = acc[i - from];
                if (v.lengthSq() < 1e-20f) v.set(0, 1, 0);
                v.normalize();
                nor[3 * i] = v.x;
                nor[3 * i + 1] = v.y;
                nor[3 * i + 2] = v.z;
            }
        }

        [[nodiscard]] std::shared_ptr<threepp::Mesh> mesh(const std::shared_ptr<threepp::Material>& material, bool withColour = true) const {
            using namespace threepp;
            auto g = BufferGeometry::create();
            g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            g->setAttribute("normal", FloatBufferAttribute::create(nor, 3));
            g->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            if (withColour) g->setAttribute("color", FloatBufferAttribute::create(col, 3));
            g->setIndex(index);
            g->computeBoundingSphere();
            auto m = Mesh::create(g, material);
            m->castShadow = true;
            m->receiveShadow = true;
            return m;
        }
    };

    // a frame: an origin and three unit axes (u along, v up, w out of the face)
    struct Frame {
        threepp::Vector3 o, u, v, w;
        [[nodiscard]] threepp::Vector3 at(float a, float b, float c) const { return o + u * a + v * b + w * c; }
    };

    // a box of mortar (the wall's core behind the stones)
    inline void appendBox(MeshBuild& out, const Frame& f, float hx, float hy, float hz, float tile, const V3& c = V3(1, 1, 1)) {
        using threepp::Vector3;
        const std::array<std::array<float, 3>, 6> normals{{{0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}}};
        for (const auto& nn : normals) {
            const Vector3 n(nn[0], nn[1], nn[2]);
            // two in-face axes
            Vector3 a = std::abs(n.y) > 0.5f ? Vector3(1, 0, 0) : Vector3(0, 1, 0);
            Vector3 b = n.clone().cross(a);
            const unsigned base = out.vertexCount();
            const float ha = std::abs(a.x) * hx + std::abs(a.y) * hy + std::abs(a.z) * hz;
            const float hb = std::abs(b.x) * hx + std::abs(b.y) * hy + std::abs(b.z) * hz;
            const float hn = std::abs(n.x) * hx + std::abs(n.y) * hy + std::abs(n.z) * hz;
            for (int k = 0; k < 4; ++k) {
                const float sa = (k & 1) ? 1.f : -1.f, sb = (k & 2) ? 1.f : -1.f;
                const Vector3 local = n * hn + a * (sa * ha) + b * (sb * hb);
                const Vector3 world = f.at(local.x, local.y, local.z);
                const Vector3 wn = f.u * n.x + f.v * n.y + f.w * n.z;
                // texture by world position on the face's plane
                const float u = (world.dot(f.u * a.x + f.v * a.y + f.w * a.z)) / tile;
                const float v = (world.dot(f.u * b.x + f.v * b.y + f.w * b.z)) / tile;
                out.vertex(world, wn, u, v, c);
            }
            out.index.insert(out.index.end(), {base, base + 1, base + 2, base + 1, base + 3, base + 2});
        }
    }

    // ---- carpentry -------------------------------------------------------------------------------------------------
    constexpr float woodTileU = 0.32f, woodTileV = 1.28f;

    struct WoodLook {
        V3 tint{1, 1, 1};
        uint32_t seed = 1;
        float sag = 0.f;   // m the middle drops (along -up)
        float twist = 0.f; // radians end to end
        float bow = 0.f;   // m the middle bows sideways
        // per-vertex weathering at a world point with a world normal
        std::function<V3(const threepp::Vector3&, const threepp::Vector3&, const V3&)> weather;
    };

    // a squared timber from a to b: `width` across (perpendicular to up), `height` along up; grain along its length
    inline void appendBeam(MeshBuild& out, const threepp::Vector3& a, const threepp::Vector3& b, const threepp::Vector3& upHint,
                           float width, float height, const WoodLook& look = {}) {
        using threepp::Vector3;
        Vector3 axis = b - a;
        const float len = axis.length();
        if (len < 1e-5f) return;
        axis.normalize();
        Vector3 side = upHint.clone().cross(axis);
        if (side.lengthSq() < 1e-8f) side = Vector3(1, 0, 0).cross(axis);
        if (side.lengthSq() < 1e-8f) side = Vector3(0, 0, 1).cross(axis);
        side.normalize();
        Vector3 up = axis.clone().cross(side).normalize();
        const int segs = std::clamp(static_cast<int>(len / 0.4f), 1, 24);
        const float hw = 0.5f * width, hh = 0.5f * height;
        const float ou = unit(look.seed) * 9.f, ov = unit(hash32(look.seed + 1)) * 9.f;
        const auto bend = [&](float t, float s, float u) {
            const float mid = 4.f * t * (1.f - t);
            const float tw = look.twist * (t - 0.5f);
            const float cs = std::cos(tw), sn = std::sin(tw);
            const float s2 = s * cs - u * sn, u2 = s * sn + u * cs;
            return a + axis * (t * len) + side * (s2 + look.bow * mid) + up * (u2 - look.sag * mid);
        };
        // the four long faces, corner to corner round the section (side, up): +side, +up, -side, -up
        const std::array<std::array<float, 2>, 4> corner{{{hw, -hh}, {hw, hh}, {-hw, hh}, {-hw, -hh}}};
        for (int f = 0; f < 4; ++f) {
            const auto& A = corner[f];
            const auto& B = corner[(f + 1) % 4];
            const Vector3 n = f == 0 ? side : f == 1 ? up : f == 2 ? side * -1.f : up * -1.f;
            const float across = (f % 2 == 0) ? height : width;
            const unsigned base = out.vertexCount();
            for (int k = 0; k <= segs; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(segs);
                for (int e = 0; e < 2; ++e) {
                    const auto& C = e ? B : A;
                    const Vector3 p = bend(t, C[0], C[1]);
                    const float u = (e ? across : 0.f) / woodTileU + ou + 0.37f * static_cast<float>(f);
                    const float v = t * len / woodTileV + ov;
                    out.vertex(p, n, u, v, look.weather ? look.weather(p, n, look.tint) : look.tint);
                }
            }
            for (int k = 0; k < segs; ++k) {
                const unsigned i0 = base + 2 * k, i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3;
                out.index.insert(out.index.end(), {i0, i1, i2, i1, i3, i2});
            }
        }
        // the end grain
        for (int e = 0; e < 2; ++e) {
            const float t = static_cast<float>(e);
            const Vector3 n = axis * (e ? 1.f : -1.f);
            const unsigned base = out.vertexCount();
            const std::array<std::array<float, 2>, 4> c{{{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}};
            for (const auto& q : c) {
                const Vector3 p = bend(t, q[0], q[1]);
                const V3 endGrain = look.tint * 0.7f;
                out.vertex(p, n, q[0] / woodTileU + ou, q[1] / woodTileU + ov, look.weather ? look.weather(p, n, endGrain) : endGrain);
            }
            if (e) out.index.insert(out.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
            else out.index.insert(out.index.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
        }
    }

    // a round timber or rod from a to b
    inline void appendCylinder(MeshBuild& out, const threepp::Vector3& a, const threepp::Vector3& b, float r0, float r1, int sides,
                               const WoodLook& look = {}, bool caps = true) {
        using threepp::Vector3;
        Vector3 axis = b - a;
        const float len = axis.length();
        if (len < 1e-5f) return;
        axis.normalize();
        Vector3 side = std::abs(axis.y) < 0.9f ? Vector3(0, 1, 0).cross(axis) : Vector3(1, 0, 0).cross(axis);
        side.normalize();
        const Vector3 up = axis.clone().cross(side).normalize();
        const int segs = std::clamp(static_cast<int>(len / 0.4f), 1, 32);
        const float ou = unit(look.seed) * 9.f, ov = unit(hash32(look.seed + 1)) * 9.f;
        const unsigned base = out.vertexCount();
        const float slope = (r0 - r1) / len;
        for (int k = 0; k <= segs; ++k) {
            const float t = static_cast<float>(k) / static_cast<float>(segs);
            const float r = r0 + (r1 - r0) * t;
            const float mid = 4.f * t * (1.f - t);
            for (int s = 0; s <= sides; ++s) {
                const float ang = 6.2831853f * static_cast<float>(s) / static_cast<float>(sides);
                Vector3 radial = side * std::cos(ang) + up * std::sin(ang);
                const Vector3 p = a + axis * (t * len) + radial * r - up * (look.sag * mid);
                Vector3 n = radial + axis * slope;
                n.normalize();
                const float u = (static_cast<float>(s) / static_cast<float>(sides)) * (6.2831853f * std::max(r0, r1)) / woodTileU + ou;
                out.vertex(p, n, u, t * len / woodTileV + ov, look.weather ? look.weather(p, n, look.tint) : look.tint);
            }
        }
        const unsigned ring = sides + 1;
        for (int k = 0; k < segs; ++k) {
            for (int s = 0; s < sides; ++s) {
                const unsigned i0 = base + k * ring + s, i1 = i0 + 1, i2 = i0 + ring, i3 = i2 + 1;
                out.index.insert(out.index.end(), {i0, i1, i2, i1, i3, i2});
            }
        }
        if (!caps) return;
        for (int e = 0; e < 2; ++e) {
            const float r = e ? r1 : r0;
            const Vector3 c = e ? b : a;
            const Vector3 n = axis * (e ? 1.f : -1.f);
            const unsigned cb = out.vertexCount();
            const V3 tint = look.tint * 0.75f;
            out.vertex(c, n, ou, ov, tint);
            for (int s = 0; s <= sides; ++s) {
                const float ang = 6.2831853f * static_cast<float>(s) / static_cast<float>(sides);
                const Vector3 p = c + (side * std::cos(ang) + up * std::sin(ang)) * r;
                out.vertex(p, n, std::cos(ang) * r / woodTileU + ou, std::sin(ang) * r / woodTileU + ov, tint);
            }
            for (int s = 0; s < sides; ++s) {
                if (e) out.index.insert(out.index.end(), {cb, cb + 1 + s, cb + 2 + s});
                else out.index.insert(out.index.end(), {cb, cb + 2 + s, cb + 1 + s});
            }
        }
    }

    // a flat quad (two triangles), uv by the given scale along its two edges
    inline void appendQuad(MeshBuild& out, const threepp::Vector3& p0, const threepp::Vector3& p1, const threepp::Vector3& p2, const threepp::Vector3& p3,
                           float uScale, float vScale, const V3& c = V3(1, 1, 1)) {
        using threepp::Vector3;
        Vector3 n = (p1 - p0).cross(p3 - p0);
        n.normalize();
        const float lu = (p1 - p0).length() / uScale, lv = (p3 - p0).length() / vScale;
        const unsigned base = out.vertexCount();
        out.vertex(p0, n, 0, 0, c);
        out.vertex(p1, n, lu, 0, c);
        out.vertex(p2, n, lu, lv, c);
        out.vertex(p3, n, 0, lv, c);
        out.index.insert(out.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }

}// namespace mill

#endif
