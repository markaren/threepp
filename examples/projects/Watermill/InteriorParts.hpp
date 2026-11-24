// Things that stand in the mill: millstones and their tuns, hoppers on their horses, and what the miller left behind
// (sacks, a chair, a barrel, a grain shovel, leaves, cobwebs). Each is built into a MeshBuild in its own place;
// the Interior decides where.

#ifndef THREEPP_EXAMPLE_WATERMILL_INTERIORPARTS_HPP
#define THREEPP_EXAMPLE_WATERMILL_INTERIORPARTS_HPP

#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"

#include "GearParts.hpp"

namespace mill::parts {

    using threepp::Vector3;
    using gear::Rot;
    using gear::twoPi;

    // copy `src` into `out` moved by `m` (a rigid motion)
    inline void appendTransformed(MeshBuild& out, const MeshBuild& src, const threepp::Matrix4& m) {
        threepp::Matrix3 nm;
        nm.getNormalMatrix(m);
        const unsigned base = out.vertexCount();
        for (unsigned i = 0; i < src.vertexCount(); ++i) {
            Vector3 p(src.pos[3 * i], src.pos[3 * i + 1], src.pos[3 * i + 2]);
            Vector3 n(src.nor[3 * i], src.nor[3 * i + 1], src.nor[3 * i + 2]);
            p.applyMatrix4(m);
            n.applyMatrix3(nm).normalize();
            out.pos.insert(out.pos.end(), {p.x, p.y, p.z});
            out.nor.insert(out.nor.end(), {n.x, n.y, n.z});
        }
        out.uv.insert(out.uv.end(), src.uv.begin(), src.uv.end());
        out.col.insert(out.col.end(), src.col.begin(), src.col.end());
        for (auto i : src.index) out.index.push_back(base + i);
    }

    // recolour a build's vertices by a weathering function of world position and normal
    inline void reweather(MeshBuild& b, unsigned from, const std::function<V3(const Vector3&, const Vector3&, const V3&)>& w) {
        for (unsigned i = from; i < b.vertexCount(); ++i) {
            const Vector3 p(b.pos[3 * i], b.pos[3 * i + 1], b.pos[3 * i + 2]);
            const Vector3 n(b.nor[3 * i], b.nor[3 * i + 1], b.nor[3 * i + 2]);
            const V3 c = w(p, n, V3(b.col[3 * i], b.col[3 * i + 1], b.col[3 * i + 2]));
            b.col[3 * i] = c.r;
            b.col[3 * i + 1] = c.g;
            b.col[3 * i + 2] = c.b;
        }
    }

    // a board given by four corners (c0..c3 round its face) and a thickness behind it; the grain runs c0 -> c3
    inline void appendSlab(MeshBuild& out, const Vector3& c0, const Vector3& c1, const Vector3& c2, const Vector3& c3, float thick, const WoodLook& look) {
        Vector3 n = (c1 - c0).cross(c3 - c0);
        n.normalize();
        const Vector3 back = n * -thick;
        const std::array<Vector3, 8> p{c0, c1, c2, c3, c0 + back, c1 + back, c2 + back, c3 + back};
        const float ou = unit(look.seed) * 9.f, ov = unit(hash32(look.seed + 1)) * 9.f;
        const auto quad = [&](int a, int b, int c, int d, bool grainOnAB) {
            Vector3 fn = (p[b] - p[a]).cross(p[d] - p[a]);
            fn.normalize();
            const float lu = (p[b] - p[a]).length(), lv = (p[d] - p[a]).length();
            const unsigned base = out.vertexCount();
            const std::array<std::array<float, 2>, 4> uv{{{0, 0}, {lu, 0}, {lu, lv}, {0, lv}}};
            const std::array<int, 4> idx{a, b, c, d};
            for (int k = 0; k < 4; ++k) {
                const float u = grainOnAB ? uv[k][1] : uv[k][0], v = grainOnAB ? uv[k][0] : uv[k][1];
                out.vertex(p[idx[k]], fn, u / woodTileU + ou, v / woodTileV + ov, gear::shade(look, p[idx[k]], fn, look.tint));
            }
            out.index.insert(out.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        };
        quad(0, 1, 2, 3, false);// front
        quad(5, 4, 7, 6, false);// back
        quad(4, 5, 1, 0, true); // edge c0-c1
        quad(3, 2, 6, 7, true); // edge c3-c2
        quad(4, 0, 3, 7, false);// edge c0-c3
        quad(1, 5, 6, 2, false);// edge c1-c2
    }

    // ---- millstones ------------------------------------------------------------------------------------------------
    // A French burr stone in the frame R: the grinding face at axial `hFace`, the back at `hBack` (either order), radius
    // `radius`, the eye `eye`. The dressed face goes to `face` (planar uv over the disc, as the face texture is
    // painted), the skirt, back and eye to `plain` (the skirt maps onto the outer pieces of the plain texture, so its
    // joints line up with the face's). Iron hoops round the skirt go to `iron`.
    inline void appendMillstone(MeshBuild& face, MeshBuild& plain, MeshBuild& iron, const Rot& R, float radius, float eye, float hFace, float hBack,
                                const V3& tint, bool dressed, uint32_t seed) {
        const int segs = 72, rings = 8;
        const float S = 2.f * radius;
        const auto planarUV = [&](float r, float g, float& u, float& v) {
            u = r * std::cos(g) / S + 0.5f;
            v = r * std::sin(g) / S + 0.5f;
        };
        const auto tinted = [&](float r, float g) {
            // a little variation and the dust of the stone floor
            return tint * (0.93f + 0.1f * fbm2(r * 4.f * std::cos(g), r * 4.f * std::sin(g), 2, seed, 0, 0));
        };
        // the two flat faces
        for (int side = 0; side < 2; ++side) {
            const bool isFace = side == 0;
            MeshBuild& out = isFace && dressed ? face : plain;
            const float h = isFace ? hFace : hBack;
            const float sgn = (isFace ? (hFace > hBack ? 1.f : -1.f) : (hBack > hFace ? 1.f : -1.f));
            const Vector3 n = R.a * sgn;
            const unsigned base = out.vertexCount();
            for (int i = 0; i <= rings; ++i) {
                const float r = eye + (radius - eye) * static_cast<float>(i) / static_cast<float>(rings);
                for (int s = 0; s <= segs; ++s) {
                    const float g = twoPi * static_cast<float>(s) / static_cast<float>(segs);
                    float u, v;
                    planarUV(r, g, u, v);
                    // the back of a runner is plaster, smoothed and a little domed
                    const float dome = isFace ? 0.f : 0.012f * (1.f - (r - eye) / (radius - eye));
                    out.vertex(R.at(r, g, h + sgn * dome), n, u, v, isFace ? tinted(r, g) : tinted(r, g) * 1.08f);
                }
            }
            const unsigned w = segs + 1;
            for (int i = 0; i < rings; ++i) {
                for (int s = 0; s < segs; ++s) {
                    const unsigned a = base + i * w + s, b = a + 1, c = a + w, d = c + 1;
                    if (sgn > 0) out.index.insert(out.index.end(), {a, c, b, b, c, d});
                    else out.index.insert(out.index.end(), {a, b, c, b, d, c});
                }
            }
        }
        // the skirt and the eye's wall
        const float hLo = std::min(hFace, hBack), hHi = std::max(hFace, hBack);
        for (int wall = 0; wall < 2; ++wall) {
            const float r = wall == 0 ? radius : eye;
            const unsigned base = plain.vertexCount();
            const int hs = 4;
            for (int k = 0; k <= hs; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(hs);
                const float h = hLo + (hHi - hLo) * t;
                for (int s = 0; s <= segs; ++s) {
                    const float g = twoPi * static_cast<float>(s) / static_cast<float>(segs);
                    const Vector3 n = R.dir(g) * (wall == 0 ? 1.f : -1.f);
                    float u, v;
                    planarUV(wall == 0 ? radius - 0.02f - 0.26f * t : eye + 0.02f + 0.1f * t, g, u, v);
                    plain.vertex(R.at(r, g, h), n, u, v, tinted(r, g) * (wall == 0 ? 0.95f : 0.7f));
                }
            }
            const unsigned w = segs + 1;
            for (int k = 0; k < hs; ++k) {
                for (int s = 0; s < segs; ++s) {
                    const unsigned a = base + k * w + s, b = a + 1, c = a + w, d = c + 1;
                    if (wall == 0) plain.index.insert(plain.index.end(), {a, b, c, b, d, c});
                    else plain.index.insert(plain.index.end(), {a, c, b, b, c, d});
                }
            }
        }
        // iron hoops shrunk round the skirt
        {
            const float h = hLo + (hHi - hLo) * 0.5f;
            gear::appendArc(iron, R, radius - 0.002f, radius + 0.009f, h - 0.03f, h + 0.03f, 0.f, twoPi, segs, WoodLook{V3(0.34f, 0.36f, 0.38f), seed + 3}, false);
        }
    }

    // ---- sacks -------------------------------------------------------------------------------------------------------
    // a filled sack standing or slumped: a soft box round its own up axis, gathered and tied at the neck.
    // `fill` 0..1 (how full), `slump` leans its top over toward +x of its frame.
    inline void appendSack(MeshBuild& out, const threepp::Matrix4& place, float width, float depth, float height, float fill, float slump, uint32_t seed) {
        MeshBuild s;
        const int around = 28, up = 14;
        const float perim = 2.f * (width + depth);
        for (int j = 0; j <= up; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(up);
            // the profile: settled wide at the foot, full in the belly, round shoulders, gathered and tied at a short
            // neck, the cloth above the tie flaring a little
            float scale = 0.94f + 0.1f * std::sin(3.14159f * std::min(t / 0.7f, 1.f));
            if (t > 0.66f) scale *= 1.f - 0.78f * smooth(0.66f, 0.86f, t);
            if (t > 0.9f) scale += 0.16f * smooth(0.9f, 1.f, t);
            const float bodyH = 0.86f * (0.8f + 0.2f * fill);
            const float y = height * (t <= 0.86f ? t / 0.86f * bodyH : bodyH + (t - 0.86f) / 0.14f * 0.14f);
            for (int i = 0; i <= around; ++i) {
                const float a = twoPi * static_cast<float>(i) / static_cast<float>(around);
                // a rounded rectangle (superellipse) cross section
                const float ca = std::cos(a), sa = std::sin(a);
                const float ex = std::pow(std::abs(ca), 0.6f) * (ca < 0 ? -1.f : 1.f), ez = std::pow(std::abs(sa), 0.6f) * (sa < 0 ? -1.f : 1.f);
                const float lump = 1.f + 0.06f * fbm2(a * 2.f, t * 3.f, 3, seed, 0, 0);
                float x = 0.5f * width * ex * scale * lump, z = 0.5f * depth * ez * scale * lump;
                const float sl = slump * t * t;
                x += sl * height * 0.6f;
                const float yy = y - std::abs(sl) * height * 0.15f * t;
                // the foot spreads on the floor
                if (t < 0.08f) {
                    x *= 1.05f;
                    z *= 1.05f;
                }
                s.vertex(Vector3(x, std::max(yy, 0.f), z), Vector3(ca, 0, sa), static_cast<float>(i) / static_cast<float>(around) * perim / 0.2f, y / 0.2f, V3(1, 1, 1));
            }
        }
        const unsigned w = around + 1;
        for (int j = 0; j < up; ++j) {
            for (int i = 0; i < around; ++i) {
                const unsigned a = j * w + i, b = a + 1, c = a + w, d = c + 1;
                s.index.insert(s.index.end(), {a, c, b, b, c, d});
            }
        }
        // the floor of the sack and the tied neck's tuft
        {
            const unsigned cb = s.vertexCount();
            s.vertex(Vector3(0, 0.002f, 0), Vector3(0, -1, 0), 0.5f, 0.5f, V3(1, 1, 1));
            for (int i = 0; i < around; ++i) s.index.insert(s.index.end(), {cb, static_cast<unsigned>(i), static_cast<unsigned>(i + 1)});
            const unsigned top = (up)*w;
            const Vector3 tuft(slump * height * 0.6f, height * (0.86f * (0.8f + 0.2f * fill) + 0.1f), 0.f);
            const unsigned ct = s.vertexCount();
            s.vertex(tuft, Vector3(0, 1, 0), 0.5f, 0.5f, V3(1, 1, 1));
            for (int i = 0; i < around; ++i) s.index.insert(s.index.end(), {ct, top + i + 1, top + i});
        }
        s.smoothNormals();
        appendTransformed(out, s, place);
    }

    // an empty sack lying flat and rumpled on the floor (in its own frame: x along, z across, y up)
    inline void appendEmptySack(MeshBuild& out, const threepp::Matrix4& place, float length, float width, uint32_t seed) {
        MeshBuild s;
        const int nx = 28, nz = 18;
        for (int j = 0; j <= nz; ++j) {
            for (int i = 0; i <= nx; ++i) {
                const float u = static_cast<float>(i) / nx, v = static_cast<float>(j) / nz;
                // a ragged outline, folds across it, a ridge where it was folded double
                const float edge = 1.f + 0.12f * fbm2(u * 3.f + 7.f, v * 3.f, 2, seed + 1, 0, 0);
                const float x = (u - 0.5f) * length * edge, z = (v - 0.5f) * width * (1.f + 0.1f * fbm2(u * 2.f, v * 4.f + 3.f, 2, seed + 2, 0, 0));
                const float folds = 0.03f * std::max(0.f, fbm2(u * 5.f, v * 2.5f, 3, seed, 0, 0)) + 0.025f * std::exp(-std::pow((v - 0.62f) / 0.08f, 2.f));
                const float wr = folds * std::pow(std::sin(3.14159f * u) * std::sin(3.14159f * v), 0.4f);
                s.vertex(Vector3(x, 0.004f + wr, z), Vector3(0, 1, 0), x / 0.2f, z / 0.2f, V3(1, 1, 1));
            }
        }
        for (int j = 0; j < nz; ++j)
            for (int i = 0; i < nx; ++i) {
                const unsigned a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
                s.index.insert(s.index.end(), {a, c, b, b, c, d});
            }
        s.smoothNormals();
        appendTransformed(out, s, place);
    }

    // ---- a ladder-back chair, broken: on its side, a back slat and a leg gone ------------------------------------------
    inline void appendBrokenChair(MeshBuild& wood, const threepp::Matrix4& place, uint32_t seed) {
        MeshBuild c;
        WoodLook look{V3(0.62f, 0.5f, 0.4f), seed};
        const float seatY = 0.44f, hw = 0.21f, hd = 0.19f;
        // legs (one broken off short), back posts rising from the rear legs
        appendCylinder(c, Vector3(-hw, 0, -hd), Vector3(-hw, seatY, -hd), 0.017f, 0.019f, 8, look);
        appendCylinder(c, Vector3(hw, 0.27f, -hd), Vector3(hw, seatY, -hd), 0.017f, 0.019f, 8, look);// the broken one
        appendCylinder(c, Vector3(-hw, 0, hd), Vector3(-hw, 0.98f, hd), 0.018f, 0.016f, 8, look);
        appendCylinder(c, Vector3(hw, 0, hd), Vector3(hw, 0.98f, hd), 0.018f, 0.016f, 8, look);
        // rungs
        for (float y : {0.14f, 0.26f}) {
            appendCylinder(c, Vector3(-hw, y, -hd), Vector3(-hw, y, hd), 0.009f, 0.009f, 6, look);
            appendCylinder(c, Vector3(-hw, y, hd), Vector3(hw, y, hd), 0.009f, 0.009f, 6, look);
        }
        // the seat: a split-oak board seat, worn
        appendSlab(c, Vector3(-hw - 0.02f, seatY + 0.02f, -hd - 0.03f), Vector3(hw + 0.02f, seatY + 0.02f, -hd - 0.03f), Vector3(hw + 0.02f, seatY + 0.02f, hd + 0.01f),
                   Vector3(-hw - 0.02f, seatY + 0.02f, hd + 0.01f), 0.022f, WoodLook{V3(0.66f, 0.54f, 0.43f), seed + 1});
        // the ladder back: two slats left of three
        for (float y : {0.66f, 0.86f}) {
            appendSlab(c, Vector3(-hw, y, hd + 0.008f), Vector3(hw, y, hd + 0.008f), Vector3(hw, y + 0.07f, hd + 0.008f), Vector3(-hw, y + 0.07f, hd + 0.008f), 0.012f,
                       WoodLook{V3(0.6f, 0.49f, 0.39f), seed + 2});
        }
        appendTransformed(wood, c, place);
    }

    // a stave barrel with iron hoops, standing; `height`, belly radius `rb`, head radius `rh`
    inline void appendBarrel(MeshBuild& wood, MeshBuild& iron, const Vector3& at, float height, float rb, float rh, uint32_t seed) {
        const int staves = 20;
        Rng rng(seed);
        for (int k = 0; k < staves; ++k) {
            const float g = twoPi * (static_cast<float>(k) + 0.5f) / static_cast<float>(staves);
            const Vector3 dir(std::cos(g), 0, std::sin(g));
            WoodLook look;
            look.tint = V3(0.55f, 0.45f, 0.36f) * rng.range(0.85f, 1.1f);
            look.seed = rng.next();
            look.sag = -(rb - rh);// bows out to the belly
            const float w = twoPi * rh / static_cast<float>(staves) * 1.02f;
            appendBeam(wood, at + dir * (rh - 0.01f), at + dir * (rh - 0.01f) + Vector3(0, height, 0), dir, w, 0.02f, look);
        }
        // the head, set in a little below the chime
        gear::appendDisc(wood, gear::aboutY(at), 0.f, rh - 0.02f, height - 0.05f, height - 0.03f, 24, WoodLook{V3(0.5f, 0.42f, 0.34f), seed + 5});
        for (float t : {0.06f, 0.24f, 0.76f, 0.94f}) {
            const float mid = 4.f * t * (1.f - t);
            const float r = rh + (rb - rh) * mid + 0.004f;
            gear::appendArc(iron, gear::aboutY(at), r - 0.004f, r + 0.003f, height * t - 0.02f, height * t + 0.02f, 0.f, twoPi, 32, WoodLook{V3(0.34f, 0.35f, 0.37f), seed + 7}, false);
        }
    }

    // a corner cobweb: a quarter fan from `corner` along `d1` and `d2` (unit), `size` m
    inline void appendCobweb(MeshBuild& out, const Vector3& corner, const Vector3& d1, const Vector3& d2, float size, float sagAmount) {
        // a few rows of vertices so the web can sag in its middle
        const int n = 4;
        Vector3 nrm = d1.clone().cross(d2);
        nrm.normalize();
        const unsigned base = out.vertexCount();
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                const float u = static_cast<float>(i) / n, v = static_cast<float>(j) / n;
                const float sag = sagAmount * std::sin(3.14159f * std::min(u + v, 1.f)) * u * v * 4.f;
                const Vector3 p = corner + d1 * (u * size) + d2 * (v * size) - Vector3(0, sag, 0);
                out.vertex(p, nrm, u, v, V3(1, 1, 1));
            }
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const unsigned a = base + j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
                out.index.insert(out.index.end(), {a, b, c, b, d, c});
            }
    }

}// namespace mill::parts

#endif
