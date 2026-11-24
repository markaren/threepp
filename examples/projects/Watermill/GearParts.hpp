// The millwright's parts: wheels and pinions built round an axis, so each can be posed by one rotation. A gear is
// laid out in a rotation frame (Rot): an origin, the axis it turns about, and two unit vectors e1, e2 across it with
// e1 x e2 = axis, so that a rotation about the axis by an angle adds that angle to every part's angle g in
// dir(g) = e1 cos g + e2 sin g. Tooth k of a wheel of N teeth sits at g = 2 pi k / N (+ the wheel's phase).
//
// Wood carries its grain along the felloes and the cogs; iron is the shared rusty TexSet; every piece writes a vertex
// colour through the WoodLook hooks (its tint, the dust on what faces up).

#ifndef THREEPP_EXAMPLE_WATERMILL_GEARPARTS_HPP
#define THREEPP_EXAMPLE_WATERMILL_GEARPARTS_HPP

#include "Geometry.hpp"

namespace mill::gear {

    using threepp::Vector3;
    constexpr float twoPi = 6.2831853f;

    struct Rot {
        Vector3 o, a, e1, e2;
        [[nodiscard]] Vector3 dir(float g) const { return e1 * std::cos(g) + e2 * std::sin(g); }
        [[nodiscard]] Vector3 at(float r, float g, float h) const { return o + dir(g) * r + a * h; }
    };

    // about +y (the upright shaft, the stone spindles): dir(g) = (cos g, 0, -sin g)
    inline Rot aboutY(const Vector3& o) { return {o, Vector3(0, 1, 0), Vector3(1, 0, 0), Vector3(0, 0, -1)}; }
    // about +x (the wheel's shaft, the lay shaft): dir(g) = (0, cos g, sin g)
    inline Rot aboutX(const Vector3& o) { return {o, Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)}; }

    inline V3 shade(const WoodLook& look, const Vector3& p, const Vector3& n, const V3& c) { return look.weather ? look.weather(p, n, c) : c; }

    // a solid arc of an annulus: radii r0..r1, axial h0..h1, from angle g0 to g1, with its end faces; the grain runs
    // round the arc
    inline void appendArc(MeshBuild& out, const Rot& R, float r0, float r1, float h0, float h1, float g0, float g1, int segs, const WoodLook& look,
                          bool ends = true) {
        const float ou = unit(look.seed) * 9.f, ov = unit(hash32(look.seed + 1)) * 9.f;
        const float rm = 0.5f * (r0 + r1);
        // four faces round the section: outer (r1), top (h1), inner (r0), bottom (h0)
        struct Edge {
            float r, h;
        };
        const std::array<Edge, 5> ring{{{r1, h0}, {r1, h1}, {r0, h1}, {r0, h0}, {r1, h0}}};
        for (int f = 0; f < 4; ++f) {
            const Edge A = ring[f], B = ring[f + 1];
            const unsigned base = out.vertexCount();
            const float across = std::abs(A.r - B.r) + std::abs(A.h - B.h);
            for (int k = 0; k <= segs; ++k) {
                const float g = g0 + (g1 - g0) * static_cast<float>(k) / static_cast<float>(segs);
                const Vector3 radial = R.dir(g);
                Vector3 n = f == 0 ? radial : f == 1 ? R.a : f == 2 ? radial * -1.f : R.a * -1.f;
                for (int e = 0; e < 2; ++e) {
                    const Edge E = e ? B : A;
                    const Vector3 p = R.o + radial * E.r + R.a * E.h;
                    const float u = (e ? across : 0.f) / woodTileU + ou + 0.37f * static_cast<float>(f);
                    const float v = (g - g0) * rm / woodTileV + ov;
                    out.vertex(p, n, u, v, shade(look, p, n, look.tint));
                }
            }
            for (int k = 0; k < segs; ++k) {
                const unsigned i0 = base + 2 * k, i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3;
                // the winding faces outward whichever way the arc runs
                if (g1 > g0) out.index.insert(out.index.end(), {i0, i2, i1, i1, i2, i3});
                else out.index.insert(out.index.end(), {i0, i1, i2, i1, i3, i2});
            }
        }
        if (!ends) return;
        for (int e = 0; e < 2; ++e) {
            const float g = e ? g1 : g0;
            const Vector3 radial = R.dir(g);
            const Vector3 tangent = R.a.clone().cross(radial);
            const Vector3 n = tangent * ((e ? 1.f : -1.f) * (g1 > g0 ? 1.f : -1.f));
            const unsigned base = out.vertexCount();
            const V3 endGrain = look.tint * 0.72f;
            for (const auto& q : {Edge{r0, h0}, Edge{r1, h0}, Edge{r1, h1}, Edge{r0, h1}}) {
                const Vector3 p = R.o + radial * q.r + R.a * q.h;
                out.vertex(p, n, q.r / woodTileU + ou, q.h / woodTileU + ov, shade(look, p, n, endGrain));
            }
            // orient by the normal
            const Vector3 p0(out.pos[3 * base], out.pos[3 * base + 1], out.pos[3 * base + 2]);
            const Vector3 p1(out.pos[3 * base + 3], out.pos[3 * base + 4], out.pos[3 * base + 5]);
            const Vector3 p2(out.pos[3 * base + 6], out.pos[3 * base + 7], out.pos[3 * base + 8]);
            if ((p1 - p0).cross(p2 - p0).dot(n) >= 0) out.index.insert(out.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
            else out.index.insert(out.index.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
        }
    }

    // a ring of `pieces` felloes with hairline joints between them
    inline void appendFelloes(MeshBuild& out, const Rot& R, float r0, float r1, float h0, float h1, int pieces, float phase, const WoodLook& look) {
        const float joint = 0.004f / r1;
        const int segs = std::max(4, 96 / pieces);
        for (int k = 0; k < pieces; ++k) {
            const float g0 = phase + twoPi * static_cast<float>(k) / static_cast<float>(pieces) + joint;
            const float g1 = phase + twoPi * static_cast<float>(k + 1) / static_cast<float>(pieces) - joint;
            WoodLook l = look;
            l.seed = look.seed * 7919u + static_cast<uint32_t>(k);
            l.tint = look.tint * (0.92f + 0.12f * unit(hash32(l.seed)));
            appendArc(out, R, r0, r1, h0, h1, g0, g1, segs, l);
        }
    }

    // a flat disc (both faces and its edge) of radius r, axial h0..h1, with a hole of radius rHole (0: none)
    inline void appendDisc(MeshBuild& out, const Rot& R, float rHole, float r, float h0, float h1, int segs, const WoodLook& look) {
        if (rHole > 0.f) {
            appendArc(out, R, rHole, r, h0, h1, 0.f, twoPi, segs, look, false);
            return;
        }
        // a solid disc: the edge, then the two faces as fans
        appendArc(out, R, r * 0.999f, r, h0, h1, 0.f, twoPi, segs, look, false);
        const float ou = unit(look.seed) * 9.f, ov = unit(hash32(look.seed + 1)) * 9.f;
        for (int e = 0; e < 2; ++e) {
            const float h = e ? h1 : h0;
            const Vector3 n = R.a * (e ? 1.f : -1.f);
            const unsigned c = out.vertexCount();
            const Vector3 pc = R.o + R.a * h;
            out.vertex(pc, n, ou, ov, shade(look, pc, n, look.tint));
            for (int s = 0; s <= segs; ++s) {
                const float g = twoPi * static_cast<float>(s) / static_cast<float>(segs);
                const Vector3 p = R.at(r, g, h);
                out.vertex(p, n, std::cos(g) * r / woodTileU + ou, std::sin(g) * r / woodTileV + ov, shade(look, p, n, look.tint));
            }
            for (int s = 0; s < segs; ++s) {
                if (e) out.index.insert(out.index.end(), {c, c + 1 + s, c + 2 + s});
                else out.index.insert(out.index.end(), {c, c + 2 + s, c + 1 + s});
            }
        }
    }

    // a squared shaft with `sides` flat faces (an octagonal oak upright), axial h0..h1, `r` to the flats
    inline void appendPrism(MeshBuild& out, const Rot& R, float r, float h0, float h1, int sides, const WoodLook& look) {
        const float rc = r / std::cos(3.14159265f / static_cast<float>(sides));
        for (int s = 0; s < sides; ++s) {
            const float ga = twoPi * (static_cast<float>(s) - 0.5f) / static_cast<float>(sides);
            const float gb = twoPi * (static_cast<float>(s) + 0.5f) / static_cast<float>(sides);
            const Vector3 n = R.dir(twoPi * static_cast<float>(s) / static_cast<float>(sides));
            WoodLook l = look;
            l.seed = look.seed + 101u * static_cast<uint32_t>(s);
            const float ou = unit(l.seed) * 9.f, ov = unit(hash32(l.seed + 1)) * 9.f;
            const int segs = std::clamp(static_cast<int>((h1 - h0) / 0.4f), 1, 40);
            const unsigned base = out.vertexCount();
            const float w = 2.f * r * std::tan(3.14159265f / static_cast<float>(sides));
            for (int k = 0; k <= segs; ++k) {
                const float h = h0 + (h1 - h0) * static_cast<float>(k) / static_cast<float>(segs);
                for (int e = 0; e < 2; ++e) {
                    const Vector3 p = R.at(rc, e ? gb : ga, h);
                    out.vertex(p, n, (e ? w : 0.f) / woodTileU + ou, (h - h0) / woodTileV + ov, shade(l, p, n, l.tint));
                }
            }
            for (int k = 0; k < segs; ++k) {
                const unsigned i0 = base + 2 * k, i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3;
                out.index.insert(out.index.end(), {i0, i1, i2, i1, i3, i2});
            }
        }
        // end grain caps
        for (int e = 0; e < 2; ++e) {
            const float h = e ? h1 : h0;
            const Vector3 n = R.a * (e ? 1.f : -1.f);
            const unsigned c = out.vertexCount();
            const Vector3 pc = R.o + R.a * h;
            out.vertex(pc, n, 0, 0, shade(look, pc, n, look.tint * 0.7f));
            for (int s = 0; s <= sides; ++s) {
                const float g = twoPi * (static_cast<float>(s) - 0.5f) / static_cast<float>(sides);
                const Vector3 p = R.at(rc, g, h);
                out.vertex(p, n, std::cos(g) * rc / woodTileU, std::sin(g) * rc / woodTileU, shade(look, p, n, look.tint * 0.7f));
            }
            for (int s = 0; s < sides; ++s) {
                if (e) out.index.insert(out.index.end(), {c, c + 1 + s, c + 2 + s});
                else out.index.insert(out.index.end(), {c, c + 2 + s, c + 1 + s});
            }
        }
    }

    // a cylinder along the frame's axis, axial h0..h1 (bands, hubs, rods)
    inline void appendRound(MeshBuild& out, const Rot& R, float r, float h0, float h1, int sides, const WoodLook& look, bool caps = true) {
        appendCylinder(out, R.o + R.a * h0, R.o + R.a * h1, r, r, sides, look, caps);
    }

    // a cog: a block from radius r0 to r1 at angle g, `w` wide across the radius (tangential), axial h0..h1. Its
    // grain runs radially, as a cog's does.
    inline void appendCog(MeshBuild& out, const Rot& R, float g, float r0, float r1, float w, float h0, float h1, const WoodLook& look) {
        const float hm = 0.5f * (h0 + h1);
        appendBeam(out, R.at(r0, g, hm), R.at(r1, g, hm), R.a, w, h1 - h0, look);
    }

    // a wooden cog tapered toward its tip (spur wheels): grain radial
    inline void appendTaperedCog(MeshBuild& out, const Rot& R, float g, float r0, float r1, float wRoot, float wTip, float h0, float h1, const WoodLook& look) {
        const Vector3 rad = R.dir(g);
        const Vector3 tan = R.a.clone().cross(rad);
        std::array<Vector3, 8> c;
        for (int k = 0; k < 8; ++k) {
            const bool tip = k & 1, side = (k >> 1) & 1, top = (k >> 2) & 1;
            const float w = 0.5f * (tip ? wTip : wRoot) * (side ? 1.f : -1.f);
            c[k] = R.o + rad * (tip ? r1 : r0) + tan * w + R.a * (top ? h1 : h0);
        }
        const float ou = unit(look.seed) * 9.f, ov = unit(hash32(look.seed + 1)) * 9.f;
        // uv: across the grain from the tangential and axial offsets, along it from the radius
        const auto uvOf = [&](const Vector3& p, int axis) {
            const Vector3 d = p - R.o;
            const float r = d.dot(rad), t = d.dot(tan), h = d.dot(R.a);
            return std::array<float, 2>{(axis == 0 ? t : h) / woodTileU + ou, r / woodTileV + ov};
        };
        const auto quad = [&](int a, int b, int cc, int d, int axis) {
            Vector3 n = (c[b] - c[a]).cross(c[d] - c[a]);
            n.normalize();
            const unsigned base = out.vertexCount();
            for (int k : {a, b, cc, d}) {
                const auto uv = uvOf(c[k], axis);
                out.vertex(c[k], n, uv[0], uv[1], shade(look, c[k], n, look.tint));
            }
            out.index.insert(out.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        };
        quad(0, 1, 5, 4, 1);// flank at -w
        quad(3, 2, 6, 7, 1);// flank at +w
        quad(1, 3, 7, 5, 0);// tip
        quad(4, 5, 7, 6, 0);// top
        quad(0, 2, 3, 1, 0);// bottom
    }

    // a tapered tooth (cast iron pinions): wider at the root than at the tip
    inline void appendTooth(MeshBuild& out, const Rot& R, float g, float r0, float r1, float wRoot, float wTip, float h0, float h1, const WoodLook& look = {}) {
        const Vector3 rad = R.dir(g);
        const Vector3 tan = R.a.clone().cross(rad);
        std::array<Vector3, 8> c;
        for (int k = 0; k < 8; ++k) {
            const bool tip = k & 1, side = (k >> 1) & 1, top = (k >> 2) & 1;
            const float r = tip ? r1 : r0;
            const float w = 0.5f * (tip ? wTip : wRoot) * (side ? 1.f : -1.f);
            c[k] = R.o + rad * r + tan * w + R.a * (top ? h1 : h0);
        }
        const auto quad = [&](int a, int b, int cc, int d) {
            Vector3 n = (c[b] - c[a]).cross(c[d] - c[a]);
            n.normalize();
            const unsigned base = out.vertexCount();
            for (int k : {a, b, cc, d}) out.vertex(c[k], n, c[k].x * 1.7f + c[k].z * 1.3f, c[k].y * 1.7f, shade(look, c[k], n, look.tint));
            out.index.insert(out.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        };
        // faces: the two flanks, the tip, the two axial faces
        quad(0, 1, 5, 4);// flank at -w
        quad(3, 2, 6, 7);// flank at +w
        quad(1, 3, 7, 5);// tip
        quad(4, 5, 7, 6);// top
        quad(0, 2, 3, 1);// bottom
    }

}// namespace mill::gear

#endif
