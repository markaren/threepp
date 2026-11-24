// A hardwood grown from scratch: the trunk is laid up first (leaning, crooked, perhaps forked), then the crown's
// branches are grown into it by space colonisation: a cloud of points fills an irregular crown (a cluster of lobes
// with holes in it, thinner inside), and every branch tip within reach of a point grows one step toward the points
// it is nearest to, bent by a little phototropism, until the points are used up. Branch radii follow the pipe model
// (a parent's cross-section carries its children's), the thin ends sag under their weight, and the outermost metre
// or so of every branch carries short shoots with their leaves: each leaf its own card, a diamond folded along its
// midrib, held out from the shoot on its stalk and turned toward the light.
//
// One skeleton gives three levels of detail: every twig and leaf near the camera; fewer, larger leaves and no fine
// twigs in the middle distance; a few big leaves on the limbs for the far hillsides. Kept leaves favour the crown's
// outside, which is what a distant tree shows.

#ifndef THREEPP_EXAMPLE_WATERMILL_TREES_GROW_HPP
#define THREEPP_EXAMPLE_WATERMILL_TREES_GROW_HPP

#include "threepp/core/BufferGeometry.hpp"

#include "FoliageTextures.hpp"
#include "Noise.hpp"
#include "TextureLab.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace mill::woods {

    // ---- small vector maths --------------------------------------------------------------------------------------------
    struct P3 {
        float x = 0, y = 0, z = 0;
        P3() = default;
        P3(float x, float y, float z): x(x), y(y), z(z) {}
        P3 operator+(const P3& o) const { return {x + o.x, y + o.y, z + o.z}; }
        P3 operator-(const P3& o) const { return {x - o.x, y - o.y, z - o.z}; }
        P3 operator*(float s) const { return {x * s, y * s, z * s}; }
        P3 operator-() const { return {-x, -y, -z}; }
        P3& operator+=(const P3& o) {
            x += o.x;
            y += o.y;
            z += o.z;
            return *this;
        }
    };

    inline float dot(const P3& a, const P3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    inline P3 cross(const P3& a, const P3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    inline float length(const P3& a) { return std::sqrt(dot(a, a)); }
    inline P3 normalized(const P3& a, const P3& fallback = P3(0, 1, 0)) {
        const float l = length(a);
        return l > 1e-8f ? a * (1.f / l) : fallback;
    }
    inline P3 anyPerp(const P3& a) {
        return normalized(std::abs(a.y) < 0.9f ? cross(a, P3(0, 1, 0)) : cross(a, P3(1, 0, 0)));
    }
    // v turned by `angle` about the unit axis k
    inline P3 rotated(const P3& v, const P3& k, float angle) {
        const float c = std::cos(angle), s = std::sin(angle);
        return v * c + cross(k, v) * s + k * (dot(k, v) * (1.f - c));
    }
    inline P3 randomUnit(Rng& rng) {
        const float z = rng.range(-1.f, 1.f), a = rng() * 6.2831853f, r = std::sqrt(std::max(0.f, 1.f - z * z));
        return {r * std::cos(a), z, r * std::sin(a)};
    }

    // ---- a mesh being built ----------------------------------------------------------------------------------------------
    struct MeshBuf {
        std::vector<float> pos, nor, uv, col;
        std::vector<unsigned int> idx;

        [[nodiscard]] unsigned int vertexCount() const { return static_cast<unsigned int>(pos.size() / 3); }
        [[nodiscard]] size_t triangles() const { return idx.size() / 3; }
        [[nodiscard]] bool empty() const { return idx.empty(); }

        unsigned int vert(const P3& p, const P3& n, float u, float v, const V3& c) {
            pos.insert(pos.end(), {p.x, p.y, p.z});
            nor.insert(nor.end(), {n.x, n.y, n.z});
            uv.insert(uv.end(), {u, v});
            col.insert(col.end(), {c.r, c.g, c.b});
            return vertexCount() - 1;
        }
        void tri(unsigned int a, unsigned int b, unsigned int c) { idx.insert(idx.end(), {a, b, c}); }

        void append(const MeshBuf& o) {
            const unsigned int base = vertexCount();
            pos.insert(pos.end(), o.pos.begin(), o.pos.end());
            nor.insert(nor.end(), o.nor.begin(), o.nor.end());
            uv.insert(uv.end(), o.uv.begin(), o.uv.end());
            col.insert(col.end(), o.col.begin(), o.col.end());
            for (unsigned int i : o.idx) idx.push_back(base + i);
        }

        [[nodiscard]] std::shared_ptr<threepp::BufferGeometry> geometry() const {
            using namespace threepp;
            auto g = BufferGeometry::create();
            g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            g->setAttribute("normal", FloatBufferAttribute::create(nor, 3));
            g->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            g->setAttribute("color", FloatBufferAttribute::create(col, 3));
            g->setIndex(idx);
            g->computeBoundingSphere();
            g->computeBoundingBox();
            return g;
        }
    };

    // ---- a leaf card ---------------------------------------------------------------------------------------------------
    // stem at `p`, blade along `a` (unit), upper side toward `n` (unit, perpendicular to a); `len` the card's length;
    // `fold` lifts the edges toward the upper side (a V along the midrib), `curl` drops the tip; `shade` the normal's
    // pull toward `out` (the crown's outward direction at the leaf), which makes a crown shade as one volume.
    struct LeafCard {
        P3 p, a, n;
        float len = 0.1f;
        float fold = 0.2f, curl = 0.1f;
        int kind = Beech;
        bool mirror = false;
        V3 colour{1, 1, 1};
        P3 out{0, 1, 0};
        float outerness = 1;// 0 deep inside the crown, 1 on its skin
    };

    inline void emitLeaf(MeshBuf& mb, const LeafCard& l, float scale = 1.f, float shade = 0.45f) {
        const float L = l.len * scale;
        const float W = L * leafAspect(l.kind);
        const P3 s = cross(l.a, l.n);// across the blade
        const P3 stem = l.p;
        const P3 tip = l.p + l.a * L - l.n * (l.curl * L);
        const P3 mid = l.p + l.a * (0.5f * L) + l.n * (l.fold * 0.5f * W);
        const P3 left = mid - s * (0.5f * W);
        const P3 right = mid + s * (0.5f * W);
        // the normal: the blade's own, pulled toward the crown's outside on the side the blade faces
        const P3 outFacing = dot(l.out, l.n) >= 0 ? l.out : -l.out;
        const P3 nl = normalized(normalized(l.n + s * (-l.fold)) * (1.f - shade) + outFacing * shade);
        const P3 nr = normalized(normalized(l.n + s * l.fold) * (1.f - shade) + outFacing * shade);
        const P3 nm = normalized(l.n * (1.f - shade) + outFacing * shade);
        const AtlasCell c = atlasCell(l.kind);
        const float uL = l.mirror ? c.u0 + c.du : c.u0, uR = l.mirror ? c.u0 : c.u0 + c.du;
        const unsigned int i0 = mb.vert(stem, nm, c.u0 + 0.5f * c.du, c.v0, l.colour);
        const unsigned int i1 = mb.vert(right, nr, uR, c.v0 + 0.5f * c.dv, l.colour);
        const unsigned int i2 = mb.vert(tip, nm, c.u0 + 0.5f * c.du, c.v0 + c.dv, l.colour);
        const unsigned int i3 = mb.vert(left, nl, uL, c.v0 + 0.5f * c.dv, l.colour);
        // wind both triangles so their front faces the blade's upper side
        const P3 g = cross(right - stem, tip - stem);
        if (dot(g, l.n) >= 0) {
            mb.tri(i0, i1, i2);
            mb.tri(i0, i2, i3);
        } else {
            mb.tri(i0, i2, i1);
            mb.tri(i0, i3, i2);
        }
    }

    // ---- a tube along a polyline -------------------------------------------------------------------------------------
    // rings at the given points with the given radii; u wraps `uRepeat` times round, v runs with the length
    struct TubeStyle {
        int sides = 6;
        float uRepeat = 1;
        float vScale = 1.f / 0.8f;// tiles a metre
        float flare = 0;          // root flare at the foot (the first point), 0..1
        uint32_t seed = 0;
        V3 colour{1, 1, 1};
        float groundY = -1e9f;// colour darkens and greens toward the ground
    };

    inline void emitTube(MeshBuf& mb, const std::vector<P3>& pts, const std::vector<float>& radii, const TubeStyle& st) {
        const size_t n = pts.size();
        if (n < 2) return;
        // tangents and parallel-transported frames
        std::vector<P3> tan(n);
        for (size_t i = 0; i < n; ++i) {
            const P3 a = pts[i > 0 ? i - 1 : 0], b = pts[i + 1 < n ? i + 1 : n - 1];
            tan[i] = normalized(b - a);
        }
        P3 nrm = anyPerp(tan[0]);
        float v = 0;
        const int sides = std::max(3, st.sides);
        unsigned int prevBase = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) {
                // transport the frame: remove the tangent's component, renormalise
                nrm = normalized(nrm - tan[i] * dot(nrm, tan[i]), anyPerp(tan[i]));
                v += length(pts[i] - pts[i - 1]) * st.vScale;
            }
            const P3 bin = cross(tan[i], nrm);
            const float r = radii[i];
            const float hFoot = st.flare > 0 ? std::max(0.f, pts[i].y - pts[0].y) : 0.f;
            const unsigned int base = mb.vertexCount();
            for (int k = 0; k <= sides; ++k) {
                const float th = 6.2831853f * static_cast<float>(k) / static_cast<float>(sides);
                const float c = std::cos(th), s = std::sin(th);
                float rr = r;
                if (st.flare > 0) {
                    // buttresses spreading into the ground
                    const float f = st.flare * std::exp(-hFoot / 0.45f);
                    const float lobes = std::pow(std::max(0.f, std::cos(5.f * th + static_cast<float>(st.seed % 7))), 2.f);
                    rr *= 1.f + f * (0.55f + 0.6f * lobes);
                }
                // bark relief at the silhouette: a little roughness in the radius
                const P3 dir = nrm * c + bin * s;
                const P3 p = pts[i] + dir * rr;
                V3 colour = st.colour;
                const float above = p.y - st.groundY;
                if (above < 1.5f) colour = colour * (0.62f + 0.38f * smooth(0.f, 1.5f, above));
                // moss on the upper side of leaning stems and limbs
                const float up = std::max(0.f, dir.y);
                colour = mix(colour, colour * V3(0.75f, 0.95f, 0.6f), 0.5f * up * smooth(6.f, 0.f, above));
                mb.vert(p, dir, static_cast<float>(k) / static_cast<float>(sides) * st.uRepeat, v, colour);
            }
            if (i > 0) {
                for (int k = 0; k < sides; ++k) {
                    const unsigned int a = prevBase + k, b = prevBase + k + 1, c = base + k, d = base + k + 1;
                    mb.tri(a, b, d);
                    mb.tri(a, d, c);
                }
            }
            prevBase = base;
        }
    }

    // ---- what kind of tree ---------------------------------------------------------------------------------------------
    struct TreeSpec {
        uint32_t seed = 1;
        float height = 20;       // to the crown's top, m
        float crownBase = 0.5f;  // the lowest limbs, a fraction of the height
        float crownRadius = 5;   // horizontal, m
        float crownFlat = 0.85f; // the lobes' vertical / horizontal radius
        float crownShift = 0;    // the crown's centre moved toward local +x, m (a bank tree reaching over the water)
        int lobes = 6;
        float trunkRadius = 0.25f;// above the flare
        float flare = 0.5f;
        float lean = 0;          // radians toward local +x
        float leanRecover = 0.2f;// how the trunk turns back up toward the light
        float wobble = 0.15f;    // crookedness
        float leader = 0.75f;    // the leader runs up to this fraction of the height
        float forkAt = 0;        // > 0: the trunk forks into two leaders at this fraction of the height
        float forkAngle = 0.32f;
        float pointDensity = 7.f;// attraction points a cubic metre
        float innerDensity = 0.35f;
        float gaps = 0.3f;       // share of the crown's volume left as holes
        float step = 0.4f, influence = 2.4f, kill = 0.6f;
        float upTropism = 0.18f;
        float sag = 0.f;
        float twigRadius = 0.005f;
        int leafKind = Beech;
        int leafKind2 = -1;
        float leafMix = 0.f;
        float leafLength = 0.14f;// the card's length, m
        int leaves = 26000;
        int leavesPerShoot = 6;
        float shootLength = 0.2f;
        int shootsPerSpray = 5;   // a spray: a twig off a branch's outer end, its shoots spread flat to the light
        float sprayLength = 0.8f;
        float leafZone = 3.2f;// leaves on the outermost metres of each branch
        float leafDroop = 0.25f;
        V3 leafTint{1, 1, 1};
        int bark = Furrowed;
        V3 barkTint{1, 1, 1};
    };

    struct TreeMeshes {
        std::array<MeshBuf, 3> wood, leaves;
        float height = 0, crownRadius = 0;
    };

    // what each level of detail keeps
    enum class LeafMode { Leaves, ShootCards, SprayCards };
    struct LodRule {
        float minRadius; // thinner branches are left out
        float sidesScale;// of the tubes' sides
        LeafMode mode;   // every leaf; a leafy-shoot card for each shoot; one for each spray
        float keep;      // share of the cards kept (favouring the crown's outside)
        float cardScale; // the cards' size over the shoot or spray they stand for
        bool twigs;      // the sprays' twigs drawn
    };

    inline const std::array<LodRule, 3>& lodRules() {
        static const std::array<LodRule, 3> r{{{0.f, 1.f, LeafMode::Leaves, 1.f, 1.f, true},
                                               {0.014f, 0.6f, LeafMode::ShootCards, 0.8f, 1.35f, false},
                                               {0.04f, 0.45f, LeafMode::SprayCards, 0.85f, 1.5f, false}}};
        return r;
    }

    // ---- growing it ------------------------------------------------------------------------------------------------------
    struct SNode {
        P3 p;
        int parent = -1;
        bool trunk = false;
        float r = 0;
        int tips = 0;
        float toTip = 0;
        P3 grow;
        int growN = 0;
        std::vector<int> kids;
    };

    struct Lobe {
        P3 c;
        float rh, rv;
    };

    inline TreeMeshes growTree(const TreeSpec& s) {
        Rng rng(s.seed);
        const P3 up(0, 1, 0);
        const float H = s.height;
        std::vector<SNode> nodes;
        nodes.reserve(20000);
        const auto add = [&](const P3& p, int parent, bool trunk) {
            SNode nd;
            nd.p = p;
            nd.parent = parent;
            nd.trunk = trunk;
            nodes.push_back(nd);
            const int i = static_cast<int>(nodes.size()) - 1;
            if (parent >= 0) nodes[parent].kids.push_back(i);
            return i;
        };

        // the trunk: from below the ground up to the crown, leaning and crooked; then one leader or a fork
        std::vector<int> eligible;
        P3 dir = normalized(P3(std::sin(s.lean), std::cos(s.lean), 0));
        P3 p(0, -0.6f, 0);
        int cur = add(p, -1, true);
        const float crownY = s.crownBase * H;
        const float trunkStep = 0.5f;
        const auto growLeader = [&](int from, P3 d, float untilY, uint32_t salt) {
            P3 q = nodes[from].p;
            int at = from;
            int guard = 0;
            while (q.y < untilY && guard++ < 200) {
                const P3 wob(fbm2(q.y * 0.18f, 3.1f, 2, s.seed + salt), 0, fbm2(5.7f, q.y * 0.18f, 2, s.seed + salt + 7));
                d = normalized(d + wob * (s.wobble * trunkStep) + up * (s.leanRecover * trunkStep * 0.08f));
                q = q + d * trunkStep;
                at = add(q, at, true);
                if (q.y > crownY * 0.95f) eligible.push_back(at);
            }
            return at;
        };
        const float forkY = s.forkAt > 0 ? s.forkAt * H : s.leader * H;
        cur = growLeader(cur, dir, forkY, 11u);
        P3 topAxis = nodes[cur].p;
        if (s.forkAt > 0) {
            dir = normalized(nodes[cur].p - nodes[nodes[cur].parent].p);
            const P3 axis = normalized(rotated(anyPerp(dir), dir, rng() * 6.2831853f));
            for (int k = 0; k < 2; ++k) {
                const float ang = (k == 0 ? 1.f : -1.f) * s.forkAngle * rng.range(0.75f, 1.2f);
                const P3 d = rotated(dir, axis, ang);
                const int end = growLeader(cur, d, std::max(s.leader * H, forkY + 2.f) - rng.range(0.f, 2.f), 21u + k * 13u);
                topAxis = k == 0 ? nodes[end].p : (topAxis + nodes[end].p) * 0.5f;
            }
        }

        // the crown: lobes round the axis, the crown's centre pulled toward +x by the shift
        const float yLo = crownY, yHi = H;
        const float halfH = 0.5f * (yHi - yLo);
        const P3 centre(topAxis.x + s.crownShift, yLo + halfH, topAxis.z);
        const float R = s.crownRadius;
        std::vector<Lobe> lobes;
        lobes.push_back({centre, 0.72f * R, 0.92f * halfH});
        for (int k = 0; k < s.lobes; ++k) {
            const float rh = R * rng.range(0.4f, 0.6f);
            const float rv = std::min(rh * s.crownFlat * rng.range(0.8f, 1.15f), halfH * 0.95f);
            const float a = rng() * 6.2831853f, d = R - rh * rng.range(0.85f, 1.05f);
            const float y = rng.range(yLo + rv * 0.8f, yHi - rv * 0.9f);
            lobes.push_back({P3(centre.x + d * std::cos(a), y, centre.z + d * std::sin(a)), rh, rv});
        }
        // a top lobe so the crown rounds off over the leader
        lobes.push_back({P3(centre.x * 0.6f + topAxis.x * 0.4f, yHi - 0.35f * halfH, centre.z), 0.45f * R, 0.38f * halfH});
        const auto depthIn = [&](const P3& q) {
            float best = 1e9f;
            for (const auto& l : lobes) {
                const P3 d = q - l.c;
                const float e = std::sqrt((d.x * d.x + d.z * d.z) / (l.rh * l.rh) + d.y * d.y / (l.rv * l.rv));
                best = std::min(best, e);
            }
            return best;// < 1 inside; 0 at a lobe's centre
        };
        P3 lo(1e9f, 1e9f, 1e9f), hi(-1e9f, -1e9f, -1e9f);
        for (const auto& l : lobes) {
            lo = P3(std::min(lo.x, l.c.x - l.rh), std::min(lo.y, l.c.y - l.rv), std::min(lo.z, l.c.z - l.rh));
            hi = P3(std::max(hi.x, l.c.x + l.rh), std::max(hi.y, l.c.y + l.rv), std::max(hi.z, l.c.z + l.rh));
        }
        const float volume = 4.18879f * R * R * halfH;
        const int target = static_cast<int>(s.pointDensity * volume);
        std::vector<P3> pts;
        pts.reserve(target);
        const float gapThreshold = (s.gaps - 0.5f) * 0.9f;
        for (int k = 0; k < target * 12 && static_cast<int>(pts.size()) < target; ++k) {
            const P3 q(rng.range(lo.x, hi.x), rng.range(lo.y, hi.y), rng.range(lo.z, hi.z));
            const float e = depthIn(q);
            if (e > 1.f) continue;
            if (rng() > s.innerDensity + (1.f - s.innerDensity) * smooth(0.45f, 0.95f, e)) continue;
            if (fbm3(q.x * 0.32f, q.y * 0.32f, q.z * 0.32f, 3, s.seed + 5) < gapThreshold) continue;
            pts.push_back(q);
        }

        // space colonisation
        const float di = s.influence, dk = s.kill;
        P3 glo(std::min(lo.x, -1.f) - di, std::min(lo.y, -1.f) - di, std::min(lo.z, -1.f) - di);
        P3 ghi(std::max(hi.x, 1.f) + di, std::max(hi.y, H) + di, std::max(hi.z, 1.f) + di);
        const int gx = static_cast<int>((ghi.x - glo.x) / di) + 1, gy = static_cast<int>((ghi.y - glo.y) / di) + 1, gz = static_cast<int>((ghi.z - glo.z) / di) + 1;
        std::vector<std::vector<int>> grid(static_cast<size_t>(gx) * gy * gz);
        const auto cellOf = [&](const P3& q, int& ix, int& iy, int& iz) {
            ix = std::clamp(static_cast<int>((q.x - glo.x) / di), 0, gx - 1);
            iy = std::clamp(static_cast<int>((q.y - glo.y) / di), 0, gy - 1);
            iz = std::clamp(static_cast<int>((q.z - glo.z) / di), 0, gz - 1);
        };
        for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
            int ix, iy, iz;
            cellOf(pts[i], ix, iy, iz);
            grid[(static_cast<size_t>(iz) * gy + iy) * gx + ix].push_back(i);
        }
        std::vector<int> nearest(pts.size(), -1);
        std::vector<float> nearD2(pts.size(), 1e30f);
        std::vector<char> alive(pts.size(), 1);
        std::vector<int> fresh = eligible;
        std::vector<int> touched;
        const float di2 = di * di, dk2 = dk * dk;
        for (int iter = 0; iter < 600 && !fresh.empty() && nodes.size() < 60000; ++iter) {
            for (int n : fresh) {
                int ix, iy, iz;
                cellOf(nodes[n].p, ix, iy, iz);
                for (int z = std::max(0, iz - 1); z <= std::min(gz - 1, iz + 1); ++z)
                    for (int y = std::max(0, iy - 1); y <= std::min(gy - 1, iy + 1); ++y)
                        for (int x = std::max(0, ix - 1); x <= std::min(gx - 1, ix + 1); ++x)
                            for (int pi : grid[(static_cast<size_t>(z) * gy + y) * gx + x]) {
                                if (!alive[pi]) continue;
                                const P3 d = pts[pi] - nodes[n].p;
                                const float d2 = dot(d, d);
                                if (d2 < nearD2[pi]) {
                                    nearD2[pi] = d2;
                                    nearest[pi] = n;
                                }
                            }
            }
            fresh.clear();
            touched.clear();
            for (size_t pi = 0; pi < pts.size(); ++pi) {
                if (!alive[pi] || nearest[pi] < 0 || nearD2[pi] > di2) continue;
                if (nearD2[pi] < dk2) {
                    alive[pi] = 0;
                    continue;
                }
                SNode& nd = nodes[nearest[pi]];
                if (nd.growN == 0) touched.push_back(nearest[pi]);
                nd.grow += normalized(pts[pi] - nd.p);
                nd.growN++;
            }
            for (int n : touched) {
                const P3 g = nodes[n].grow;
                nodes[n].grow = P3();
                nodes[n].growN = 0;
                if (length(g) < 1e-3f || nodes[n].kids.size() >= 4) continue;
                P3 d = normalized(normalized(g) + up * s.upTropism + randomUnit(rng) * 0.06f);
                const P3 q = nodes[n].p + d * s.step;
                // a node that keeps reaching for the same points would sprout the same twig again
                bool dup = false;
                for (int k : nodes[n].kids) {
                    const P3 e = nodes[k].p - q;
                    if (dot(e, e) < 0.09f * s.step * s.step) dup = true;
                }
                if (dup) continue;
                fresh.push_back(add(q, n, false));
            }
        }

        const int count = static_cast<int>(nodes.size());
        // smooth the branches' kinks (chains only; the trunk keeps its line)
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<P3> np(count);
            for (int i = 0; i < count; ++i) {
                np[i] = nodes[i].p;
                const auto& nd = nodes[i];
                if (nd.trunk || nd.parent < 0 || nd.kids.size() != 1) continue;
                np[i] = nd.p * 0.5f + (nodes[nd.parent].p + nodes[nd.kids[0]].p) * 0.25f;
            }
            for (int i = 0; i < count; ++i) nodes[i].p = np[i];
        }
        // tips and radii (children always come after their parents)
        for (int i = count - 1; i >= 0; --i) {
            auto& nd = nodes[i];
            if (nd.kids.empty()) {
                nd.tips = 1;
                nd.toTip = 0;
            } else {
                nd.tips = 0;
                nd.toTip = 1e9f;
                for (int k : nd.kids) {
                    nd.tips += nodes[k].tips;
                    nd.toTip = std::min(nd.toTip, nodes[k].toTip + length(nodes[k].p - nd.p));
                }
            }
        }
        const float exponent = std::clamp(std::log(static_cast<float>(std::max(nodes[0].tips, 2))) / std::log(s.trunkRadius / s.twigRadius), 1.6f, 3.f);
        for (int i = 0; i < count; ++i) {
            auto& nd = nodes[i];
            nd.r = s.twigRadius * std::pow(static_cast<float>(nd.tips), 1.f / exponent);
            if (nd.trunk && nd.p.y < crownY) nd.r *= 1.f + 0.22f * (1.f - std::max(0.f, nd.p.y) / std::max(crownY, 1.f));
        }
        // the thin ends sag under their own weight (the offset accumulates outward)
        if (s.sag > 0) {
            std::vector<float> drop(count, 0.f);
            for (int i = 1; i < count; ++i) {
                const auto& nd = nodes[i];
                const float seg = length(nd.p - nodes[nd.parent].p);
                drop[i] = drop[nd.parent] + (nd.trunk ? 0.f : s.sag * seg * std::min(3.f, 0.012f / (nd.r + 0.004f)));
            }
            for (int i = 1; i < count; ++i) nodes[i].p.y -= drop[i];
        }

        // ---- the wood: one tube per branch, following its thickest child ----------------------------------------------
        TreeMeshes out;
        out.height = H;
        out.crownRadius = R;
        const V3 barkCol = s.barkTint;
        const auto rules = lodRules();
        std::vector<int> mainKid(count, -1);
        for (int i = 0; i < count; ++i) {
            int best = -1;
            for (int k : nodes[i].kids)
                if (best < 0 || nodes[k].tips > nodes[best].tips) best = k;
            mainKid[i] = best;
        }
        // branch starts: the root and every child that is not its parent's main one
        std::vector<int> starts{0};
        for (int i = 0; i < count; ++i)
            for (int k : nodes[i].kids)
                if (k != mainKid[i]) starts.push_back(k);
        for (int lod = 0; lod < 3; ++lod) {
            const LodRule& rule = rules[lod];
            for (int st : starts) {
                std::vector<P3> pts;
                std::vector<float> rad;
                if (nodes[st].parent >= 0) {
                    // begin inside the parent
                    pts.push_back(nodes[nodes[st].parent].p);
                    rad.push_back(nodes[st].r);
                }
                const float r0 = nodes[st].r;
                if (r0 < rule.minRadius) continue;
                // follow the main child; keep a ring where the line turns or the length runs long
                int i = st;
                P3 lastDir;
                float run = 0;
                while (i >= 0) {
                    const auto& nd = nodes[i];
                    if (nd.r < rule.minRadius * 0.7f) break;
                    const bool last = mainKid[i] < 0;
                    bool keep = pts.size() < 2 || last;
                    if (!keep) {
                        const P3 d = normalized(nd.p - pts.back());
                        run += length(nd.p - pts.back());
                        const float maxRun = nd.r > 0.1f ? 0.9f : (nd.r > 0.03f ? 1.2f : 1.6f) * (lod == 0 ? 1.f : 1.8f);
                        keep = dot(d, lastDir) < (lod == 0 ? (nd.r > 0.03f ? 0.985f : 0.96f) : 0.95f) || run > maxRun || (nd.trunk && nd.p.y < 1.5f);
                    }
                    if (keep || mainKid[i] < 0) {
                        if (!pts.empty()) lastDir = normalized(nd.p - pts.back());
                        pts.push_back(nd.p);
                        rad.push_back(nd.r);
                        run = 0;
                    }
                    i = mainKid[i];
                }
                if (pts.size() < 2) continue;
                // a cap: the tip narrows to a point
                const P3 tipDir = normalized(pts.back() - pts[pts.size() - 2]);
                pts.push_back(pts.back() + tipDir * (rad.back() * 2.f));
                rad.push_back(rad.back() * 0.2f);
                TubeStyle ts;
                const int baseSides = r0 > 0.18f ? 13 : (r0 > 0.08f ? 8 : (r0 > 0.035f ? 6 : (r0 > 0.015f ? 4 : 3)));
                ts.sides = std::max(3, static_cast<int>(std::round(static_cast<float>(baseSides) * rule.sidesScale)));
                ts.uRepeat = std::max(1.f, std::round(6.2831853f * r0 / 0.4f));
                ts.flare = st == 0 ? s.flare : 0.f;
                ts.seed = s.seed;
                ts.colour = barkCol;
                ts.groundY = 0.f;
                emitTube(out.wood[lod], pts, rad, ts);
            }
        }

        // ---- the leaves: sprays along the outer end of every branch -------------------------------------------------
        // the leafy segments and their share of the sprays: more toward the crown's outside, where the light is
        std::vector<int> leafy;
        std::vector<float> weight;
        float wsum = 0;
        for (int i = 1; i < count; ++i) {
            const auto& nd = nodes[i];
            if (nd.trunk || nd.toTip > s.leafZone || nd.r > 0.04f) continue;
            const float seg = length(nd.p - nodes[nd.parent].p);
            const float e = depthIn(nd.p);
            const float w = seg * (0.3f + 0.7f * smooth(0.3f, 0.95f, e));
            leafy.push_back(i);
            weight.push_back(w);
            wsum += w;
        }
        struct Card {
            P3 p, a, n;
            float len;
            V3 colour;
            float outerness;
            P3 out;
        };
        std::vector<LeafCard> cards;
        std::vector<Card> shootCards, sprayCards;
        std::vector<std::array<P3, 3>> twigs;
        cards.reserve(static_cast<size_t>(s.leaves) + 64);
        const int leavesPerSpray = s.shootsPerSpray * (s.leavesPerShoot + 1) + 2;
        const float spraysWanted = static_cast<float>(s.leaves) / static_cast<float>(leavesPerSpray);
        const float perWeight = wsum > 0 ? spraysWanted / wsum : 0.f;
        const int leafKindMain = s.leafKind;
        float carry = 0;
        const auto shadeAt = [&](const P3& q, float& outerness) {
            outerness = smooth(0.35f, 1.0f, depthIn(q));
            return (0.55f + 0.45f * outerness) * (0.8f + 0.2f * smooth(yLo, yHi, q.y));
        };
        const auto outwardAt = [&](const P3& q) {
            P3 ow = q - centre;
            ow.y *= 0.6f;
            return normalized(ow, up);
        };
        for (size_t li = 0; li < leafy.size(); ++li) {
            const auto& nd = nodes[leafy[li]];
            const P3 a = nodes[nd.parent].p, b = nd.p;
            const P3 segDir = normalized(b - a);
            carry += weight[li] * perWeight;
            int ns = static_cast<int>(carry);
            carry -= static_cast<float>(ns);
            for (int k = 0; k < ns; ++k) {
                const P3 origin = a + (b - a) * rng.range(0.1f, 1.f);
                P3 outward = origin - centre;
                outward.y *= 0.3f;
                outward = normalized(outward, segDir);
                // the twig springs off at 30..65 degrees, out and level, its end dipping a little
                const P3 side = normalized(rotated(anyPerp(segDir), segDir, rng() * 6.2831853f));
                const float ang = rng.range(0.5f, 1.1f);
                const P3 td = normalized(segDir * std::cos(ang) + side * std::sin(ang) + outward * 0.45f - up * 0.08f);
                const float tl = s.sprayLength * rng.range(0.6f, 1.3f);
                // the spray's plane faces the sky, tipped outward
                P3 sn0 = up * 0.75f + outward * 0.65f + randomUnit(rng) * 0.35f;
                const P3 sn = normalized(sn0 - td * dot(sn0, td), up);
                const P3 sside = normalized(cross(td, sn));
                const P3 tmid = origin + td * (0.5f * tl) + sn * (0.03f * tl);
                const P3 tend = origin + td * tl - up * (s.leafDroop * 0.15f * tl);
                twigs.push_back({origin, tmid, tend});
                const float young = rng();
                const V3 sprayTint = mix(V3(1.f, 1.f, 1.f), V3(1.1f, 1.12f, 0.8f), 0.55f * young * young) * rng.range(0.9f, 1.08f);
                const int kind = (s.leafKind2 >= 0 && rng() < s.leafMix) ? s.leafKind2 : leafKindMain;
                float sprayOuter = 0;
                const float sprayShade = shadeAt(tmid, sprayOuter);
                sprayCards.push_back({origin, normalized(tend - origin), sn, tl + s.leafLength, s.leafTint * sprayTint * sprayShade, sprayOuter, outwardAt(tmid)});
                // the shoots along the twig, alternating, spread in the spray's plane
                for (int j = 0; j < s.shootsPerSpray; ++j) {
                    const float t = 0.2f + 0.8f * static_cast<float>(j) / static_cast<float>(std::max(1, s.shootsPerSpray - 1));
                    const P3 sp = t < 0.5f ? origin + (tmid - origin) * (2.f * t) : tmid + (tend - tmid) * (2.f * t - 1.f);
                    const float sideSign = (j % 2 == 0) ? 1.f : -1.f;
                    const bool tipShoot = j == s.shootsPerSpray - 1;
                    const P3 sd = tipShoot ? normalized(td + randomUnit(rng) * 0.2f)
                                           : normalized(td * 0.55f + sside * (sideSign * 0.85f) + sn * rng.range(-0.1f, 0.2f) + randomUnit(rng) * 0.15f);
                    const float sl = s.shootLength * rng.range(0.7f, 1.3f);
                    float shootOuter = 0;
                    const float shootShade = shadeAt(sp, shootOuter);
                    const V3 shootTint = s.leafTint * sprayTint * (shootShade * rng.range(0.92f, 1.06f));
                    shootCards.push_back({sp, sd, sn, sl + s.leafLength * 0.9f, shootTint, shootOuter, outwardAt(sp)});
                    const int nl = std::max(2, s.leavesPerShoot + static_cast<int>(rng.range(-1.5f, 1.5f)));
                    for (int q = 0; q <= nl; ++q) {
                        LeafCard l;
                        const bool terminal = q == nl;
                        const float lt = terminal ? 1.f : 0.15f + 0.85f * static_cast<float>(q) / static_cast<float>(nl);
                        l.p = sp + sd * (sl * lt);
                        // leaves alternate left and right in the spray's plane, a little up or down out of it
                        const float lr = (q % 2 == 0) ? 1.f : -1.f;
                        const P3 across = normalized(cross(sd, sn));
                        P3 la = terminal ? sd : normalized(sd * 0.35f + across * (lr * 0.95f) + sn * rng.range(-0.25f, 0.3f));
                        la = normalized(la - up * (s.leafDroop * rng.range(0.3f, 1.2f)) + randomUnit(rng) * 0.12f);
                        P3 nrm = sn * 0.7f + outward * 0.25f + randomUnit(rng) * 0.55f;
                        nrm = normalized(nrm - la * dot(nrm, la), anyPerp(la));
                        l.a = la;
                        l.n = nrm;
                        l.kind = kind;
                        l.len = s.leafLength * rng.range(0.78f, 1.15f) * (terminal ? 1.08f : 1.f);
                        l.fold = rng.range(-0.05f, 0.45f);
                        l.curl = rng.range(0.f, 0.22f);
                        l.mirror = rng() < 0.5f;
                        float lo = 0;
                        const float ao = shadeAt(l.p, lo);
                        l.outerness = lo;
                        l.colour = s.leafTint * sprayTint * (ao * rng.range(0.9f, 1.08f));
                        l.out = outwardAt(l.p);
                        cards.push_back(l);
                    }
                }
            }
        }
        // per level: twigs, then leaves or the leafy cards that stand in for them
        for (int lod = 0; lod < 3; ++lod) {
            const LodRule& rule = rules[lod];
            if (rule.twigs) {
                TubeStyle ts;
                ts.sides = 3;
                ts.uRepeat = 1;
                ts.colour = barkCol * 0.8f;
                ts.groundY = -100.f;
                for (const auto& tw : twigs) {
                    emitTube(out.wood[lod], {tw[0], tw[1], tw[2]}, {s.twigRadius * 0.8f, s.twigRadius * 0.55f, s.twigRadius * 0.3f}, ts);
                }
            }
            if (rule.mode == LeafMode::Leaves) {
                for (const auto& l : cards) emitLeaf(out.leaves[lod], l);
                continue;
            }
            const auto& list = rule.mode == LeafMode::ShootCards ? shootCards : sprayCards;
            const int shootKind = shootKindFor(leafKindMain);
            for (size_t i = 0; i < list.size(); ++i) {
                const Card& c = list[i];
                // keep the outside of the crown: that is what a distant tree shows
                const float pKeep = rule.keep * (0.45f + 1.1f * c.outerness);
                if (hash2f(static_cast<int>(i), lod, s.seed) > pKeep) continue;
                LeafCard l;
                l.p = c.p;
                l.a = c.a;
                P3 nrm = c.n;
                nrm = normalized(nrm - c.a * dot(nrm, c.a), anyPerp(c.a));
                l.n = nrm;
                l.kind = shootKind;
                l.len = c.len * rule.cardScale;
                l.fold = 0.15f;
                l.curl = 0.08f;
                l.mirror = (i & 1u) != 0;
                l.colour = c.colour;
                l.out = c.out;
                l.outerness = c.outerness;
                emitLeaf(out.leaves[lod], l);
            }
        }
        return out;
    }

}// namespace mill::woods

#endif
