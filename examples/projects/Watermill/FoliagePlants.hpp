// The woods' smaller plants: rhododendron thickets (sprawling stems that fork and fork again, each tip a rosette of
// long leathery leaves over the drooping ones of earlier years), ferns (a shuttlecock of arching fronds, each a
// curved strip wearing a frond from the atlas), grass and sedge tufts (every blade its own strip), and the low leafy
// herbs of the banks (a few stems with their leaves held out flat).

#ifndef THREEPP_EXAMPLE_WATERMILL_FOLIAGE_PLANTS_HPP
#define THREEPP_EXAMPLE_WATERMILL_FOLIAGE_PLANTS_HPP

#include "TreesGrow.hpp"

#include <functional>

namespace mill::woods {

    struct PlantMeshes {
        MeshBuf wood;  // bark-textured stems
        MeshBuf leaves;// atlas leaves and fronds
        MeshBuf green; // untextured: grass blades, herb stems
        MeshBuf leavesFar;
    };

    // ---- rhododendron ------------------------------------------------------------------------------------------------
    struct ShrubSpec {
        uint32_t seed = 1;
        float height = 2.4f;
        int stems = 5;
        int depth = 3;
        float splay = 0.6f;// how far the stems lean out
        float leafLength = 0.2f;
        int rosette = 9;
        V3 leafTint{1, 1, 1};
        V3 barkTint{0.62f, 0.56f, 0.5f};
    };

    inline PlantMeshes growShrub(const ShrubSpec& s) {
        PlantMeshes out;
        Rng rng(s.seed);
        const P3 up(0, 1, 0);
        std::vector<LeafCard> cards;
        const float L0 = s.height / 2.3f;
        const auto rosette = [&](const P3& at, const P3& d, float spread, int n, float age) {
            float phase = rng() * 6.2831853f;
            const P3 perp = anyPerp(d);
            for (int j = 0; j < n; ++j) {
                LeafCard l;
                const P3 radial = rotated(perp, d, phase + 6.2831853f * static_cast<float>(j) / static_cast<float>(n) + rng.range(-0.25f, 0.25f));
                // leaves splay out from the tip and droop, last year's more than this year's
                const float e = rng.range(-0.15f, 0.45f) - 0.55f * age;
                l.p = at - d * (0.012f * static_cast<float>(j)) + radial * 0.008f;
                l.a = normalized(radial * std::cos(spread) + d * std::sin(spread) + up * e);
                P3 nrm = up + radial * 0.2f + randomUnit(rng) * 0.25f;
                nrm = normalized(nrm - l.a * dot(nrm, l.a), anyPerp(l.a));
                l.n = nrm;
                l.kind = Rhodo;
                l.len = s.leafLength * rng.range(0.8f, 1.12f) * (1.f - 0.12f * age);
                // the margins roll under: a negative fold, the tip curls down
                l.fold = rng.range(-0.35f, -0.05f);
                l.curl = rng.range(0.05f, 0.2f);
                l.mirror = rng() < 0.5f;
                const float shade = (0.62f + 0.38f * smooth(0.2f * s.height, s.height, at.y)) * (1.f - 0.15f * age);
                l.colour = s.leafTint * (shade * rng.range(0.9f, 1.08f));
                l.out = normalized(P3(at.x, 0.4f * (at.y - 0.5f * s.height), at.z), up);
                l.outerness = 1.f;
                cards.push_back(l);
            }
        };
        std::function<void(P3, P3, float, float, int)> branch = [&](P3 p, P3 d, float len, float r, int depth) {
            std::vector<P3> pts{p};
            std::vector<float> rad{r};
            const int segs = 3;
            P3 q = p;
            P3 outward = normalized(P3(p.x, 0, p.z), P3(1, 0, 0));
            for (int k = 1; k <= segs; ++k) {
                d = normalized(d + randomUnit(rng) * 0.18f + up * 0.05f + outward * 0.04f);
                q = q + d * (len / static_cast<float>(segs));
                pts.push_back(q);
                rad.push_back(r * (1.f - 0.25f * static_cast<float>(k) / static_cast<float>(segs)));
            }
            TubeStyle ts;
            ts.sides = r > 0.018f ? 5 : 3;
            ts.colour = s.barkTint;
            ts.groundY = 0.f;
            ts.uRepeat = 1;
            emitTube(out.wood, pts, rad, ts);
            if (depth == 0 || len < 0.22f) {
                rosette(q, d, rng.range(0.2f, 0.5f), s.rosette + static_cast<int>(rng.range(-2.f, 2.f)), 0.f);
                // last year's whorl, lower down the shoot
                rosette(pts[segs - 1] + (q - pts[segs - 1]) * 0.3f, d, rng.range(-0.1f, 0.25f), std::max(3, s.rosette - 3), 1.f);
                return;
            }
            const int forks = rng() < 0.45f ? 3 : 2;
            const P3 axis0 = anyPerp(d);
            const float a0 = rng() * 6.2831853f;
            for (int k = 0; k < forks; ++k) {
                const P3 axis = rotated(axis0, d, a0 + 6.2831853f * static_cast<float>(k) / static_cast<float>(forks));
                const P3 nd = normalized(rotated(d, axis, rng.range(0.3f, 0.7f)));
                branch(q, nd, len * rng.range(0.62f, 0.8f), r * 0.68f, depth - 1);
            }
            // a side rosette where the stem forks: the thicket is leafy all through, not only at the top
            if (depth <= 2 && rng() < 0.5f) rosette(q, d, 0.2f, s.rosette - 2, 0.5f);
        };
        for (int k = 0; k < s.stems; ++k) {
            const float a = rng() * 6.2831853f, rr = rng.range(0.05f, 0.35f);
            const P3 base(rr * std::cos(a), -0.15f, rr * std::sin(a));
            const float tilt = rng.range(0.25f, s.splay + 0.25f);
            const P3 d = normalized(P3(std::cos(a) * std::sin(tilt), std::cos(tilt), std::sin(a) * std::sin(tilt)));
            branch(base, d, L0 * rng.range(0.8f, 1.15f), rng.range(0.025f, 0.04f), s.depth);
        }
        for (size_t i = 0; i < cards.size(); ++i) {
            emitLeaf(out.leaves, cards[i]);
            if (hash2f(static_cast<int>(i), 3, s.seed) < 0.35f) emitLeaf(out.leavesFar, cards[i], 1.5f);
        }
        return out;
    }

    // ---- ferns --------------------------------------------------------------------------------------------------------
    struct FernSpec {
        uint32_t seed = 1;
        int kind = FernLady;
        int fronds = 9;
        float length = 0.75f;
        float rise = 1.05f;// the fronds' rise from the horizontal at the foot, radians
        float arch = 1.2f; // how much they bend over toward the tip
        V3 tint{1, 1, 1};
    };

    inline PlantMeshes growFern(const FernSpec& s) {
        PlantMeshes out;
        Rng rng(s.seed);
        const P3 up(0, 1, 0);
        const AtlasCell cell = atlasCell(s.kind);
        const float aspect = leafAspect(s.kind);
        const int segs = 7;
        for (int f = 0; f < s.fronds; ++f) {
            const float az = 6.2831853f * (static_cast<float>(f) + rng.range(-0.3f, 0.3f)) / static_cast<float>(s.fronds);
            const P3 h(std::cos(az), 0, std::sin(az));
            const float L = s.length * rng.range(0.7f, 1.1f);
            const float W = L * aspect;
            const float rise = s.rise * rng.range(0.85f, 1.15f);
            const float arch = s.arch * rng.range(0.8f, 1.2f);
            P3 p = h * 0.04f + P3(0, -0.02f, 0);
            const P3 side = cross(up, h);// across the frond, horizontal
            const float twist = rng.range(-0.25f, 0.25f);
            const float young = rng();
            const V3 tint = s.tint * mix(V3(1, 1, 1), V3(1.08f, 1.1f, 0.85f), 0.5f * young) * rng.range(0.88f, 1.06f);
            unsigned int prev[3] = {0, 0, 0};
            for (int k = 0; k <= segs; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(segs);
                const float elev = rise - arch * t;
                const P3 d = normalized(h * std::cos(elev) + up * std::sin(elev));
                if (k > 0) p = p + d * (L / static_cast<float>(segs));
                // the blade faces up off the rachis, rolled a little about it
                const P3 sd = normalized(rotated(side, d, twist * t));
                P3 nrm = normalized(cross(sd, d));
                if (nrm.y < 0) nrm = -nrm;
                const float lift = 0.12f * W;// the pinnae rise in a shallow V from the rachis
                const float shade = 0.55f + 0.45f * t;
                const V3 c = tint * shade;
                const float v = cell.v0 + cell.dv * t;
                const unsigned int il = out.leaves.vert(p - sd * (0.5f * W) + nrm * lift, normalized(nrm * 0.8f + up * 0.4f), cell.u0, v, c);
                const unsigned int im = out.leaves.vert(p, normalized(nrm * 0.8f + up * 0.4f), cell.u0 + 0.5f * cell.du, v, c);
                const unsigned int ir = out.leaves.vert(p + sd * (0.5f * W) + nrm * lift, normalized(nrm * 0.8f + up * 0.4f), cell.u0 + cell.du, v, c);
                if (k > 0) {
                    out.leaves.tri(prev[0], prev[1], im);
                    out.leaves.tri(prev[0], im, il);
                    out.leaves.tri(prev[1], prev[2], ir);
                    out.leaves.tri(prev[1], ir, im);
                }
                prev[0] = il;
                prev[1] = im;
                prev[2] = ir;
            }
        }
        out.leavesFar = out.leaves;
        return out;
    }

    // ---- grass and sedge -----------------------------------------------------------------------------------------------
    struct GrassSpec {
        uint32_t seed = 1;
        int blades = 28;
        float length = 0.42f;
        float width = 0.009f;
        float dry = 0.15f;// share of straw-coloured blades
    };

    inline PlantMeshes growGrass(const GrassSpec& s) {
        PlantMeshes out;
        Rng rng(s.seed);
        const P3 up(0, 1, 0);
        const V3 foot = srgb(46, 64, 26), mid = srgb(92, 124, 48), tipC = srgb(132, 150, 70), straw = srgb(150, 140, 92);
        for (int b = 0; b < s.blades; ++b) {
            const float az = rng() * 6.2831853f, rr = rng.range(0.f, 0.07f);
            const P3 h(std::cos(az), 0, std::sin(az));
            P3 p = h * rr + P3(0, -0.02f, 0);
            const float L = s.length * rng.range(0.5f, 1.25f);
            const float rise = rng.range(1.0f, 1.45f), arch = rng.range(0.3f, 1.3f);
            const float W = s.width * rng.range(0.7f, 1.3f);
            const P3 side = normalized(cross(up, h));
            const bool isDry = rng() < s.dry;
            const int segs = 4;
            unsigned int pl = 0, pr = 0;
            for (int k = 0; k <= segs; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(segs);
                const float elev = rise - arch * t * t;
                const P3 d = normalized(h * std::cos(elev) + up * std::sin(elev));
                if (k > 0) p = p + d * (L / static_cast<float>(segs));
                const float w = W * (1.f - 0.9f * t);
                P3 nrm = normalized(cross(side, d));
                if (nrm.y < 0) nrm = -nrm;
                nrm = normalized(nrm + up * 0.6f);
                V3 c = t < 0.5f ? mix(foot, mid, t * 2.f) : mix(mid, tipC, (t - 0.5f) * 2.f);
                if (isDry) c = mix(c, straw, 0.4f + 0.5f * t);
                const unsigned int il = out.green.vert(p - side * w, nrm, 0, t, c);
                const unsigned int ir = out.green.vert(p + side * w, nrm, 1, t, c);
                if (k > 0) {
                    out.green.tri(pl, pr, ir);
                    out.green.tri(pl, ir, il);
                }
                pl = il;
                pr = ir;
            }
        }
        return out;
    }

    // ---- herbs ---------------------------------------------------------------------------------------------------------
    struct HerbSpec {
        uint32_t seed = 1;
        int stems = 4;
        float height = 0.6f;
        int leafKind = Dogwood;
        float leafLength = 0.11f;
        V3 tint{1, 1, 1};
    };

    inline PlantMeshes growHerb(const HerbSpec& s) {
        PlantMeshes out;
        Rng rng(s.seed);
        const P3 up(0, 1, 0);
        for (int k = 0; k < s.stems; ++k) {
            const float az = rng() * 6.2831853f;
            const P3 h(std::cos(az), 0, std::sin(az));
            const float H = s.height * rng.range(0.6f, 1.15f);
            const float tilt = rng.range(0.05f, 0.45f);
            P3 d = normalized(h * std::sin(tilt) + up * std::cos(tilt));
            P3 p = h * rng.range(0.f, 0.1f) + P3(0, -0.03f, 0);
            std::vector<P3> pts{p};
            std::vector<float> rad{0.004f};
            const int nodes = 6;
            float phyl = rng() * 6.2831853f;
            for (int j = 1; j <= nodes; ++j) {
                d = normalized(d + randomUnit(rng) * 0.08f + h * 0.04f);
                p = p + d * (H / static_cast<float>(nodes));
                pts.push_back(p);
                rad.push_back(0.004f * (1.f - 0.6f * static_cast<float>(j) / static_cast<float>(nodes)));
                if (j < 2) continue;
                // a leaf at each node, alternate, held out level; the top ones smaller
                const int count = j == nodes ? 3 : 1;
                for (int c = 0; c < count; ++c) {
                    LeafCard l;
                    phyl += 2.4f;
                    const P3 radial = normalized(P3(std::cos(phyl), 0, std::sin(phyl)));
                    l.p = p;
                    l.a = normalized(radial + up * rng.range(-0.2f, 0.35f));
                    P3 nrm = up + randomUnit(rng) * 0.2f;
                    nrm = normalized(nrm - l.a * dot(nrm, l.a), anyPerp(l.a));
                    l.n = nrm;
                    l.kind = s.leafKind;
                    l.len = s.leafLength * rng.range(0.8f, 1.15f) * (j == nodes ? 0.75f : 1.f);
                    l.fold = rng.range(0.f, 0.3f);
                    l.curl = rng.range(0.05f, 0.25f);
                    l.mirror = rng() < 0.5f;
                    l.colour = s.tint * ((0.7f + 0.3f * static_cast<float>(j) / static_cast<float>(nodes)) * rng.range(0.9f, 1.08f));
                    l.out = up;
                    emitLeaf(out.leaves, l, 1.f, 0.3f);
                }
            }
            TubeStyle ts;
            ts.sides = 3;
            ts.colour = srgb(70, 96, 40);
            ts.groundY = -100.f;
            emitTube(out.green, pts, rad, ts);
        }
        out.leavesFar = out.leaves;
        return out;
    }

}// namespace mill::woods

#endif
