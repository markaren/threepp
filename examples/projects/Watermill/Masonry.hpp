// Stone walls built stone by stone. A wall is laid in courses of random height; each course is a row of stones of
// random length, so the joints never line up. Every stone is its own mesh piece: a box with rounded arrises, a face
// that swells (the mason dressed it with a hammer, not a saw), chipped edges and a colour of its own. Behind them a
// mortar core is set back from the faces, so the joints are recessed and shadowed. Corners of a wall loop bond:
// on alternate courses one wall or the other runs through the corner with a long quoin.

#ifndef THREEPP_EXAMPLE_WATERMILL_MASONRY_HPP
#define THREEPP_EXAMPLE_WATERMILL_MASONRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include "Geometry.hpp"

#include <array>
#include <functional>
#include <unordered_map>
#include <vector>

namespace mill {

    // grid lines over [-h, h] that crowd toward the ends, so a rounding of radius r has vertices to bend
    inline std::vector<float> bevelLines(float h, float r, int inner) {
        std::vector<float> t{-h, -h + 0.3f * r, -h + r};
        for (int k = 1; k < inner; ++k) t.push_back(-h + r + (2 * (h - r)) * static_cast<float>(k) / static_cast<float>(inner));
        t.insert(t.end(), {h - r, h - 0.3f * r, h});
        return t;
    }

    struct StoneLook {
        V3 tint{1, 1, 1};
        float round = 0.025f;    // arris radius, m
        float pillow = 0.025f;   // how far the face's middle swells, m
        float chips = 0.012f;    // depth of the chipping near the edges, m
        float tile = 0.9f;       // the stone texture's tile, m
        float shear = 0.f;       // m the top edge stands to one side of the bottom
        float taper = 0.f;       // one end this much (share) taller than the other
        float wobble = 0.f;      // m the outline wanders: nothing was cut to a line
        float moss = 0.f;        // 0..1 how much moss and grime gathers on its edges and top
        uint32_t seed = 1;
        // colour of a vertex at a world point with a world normal (wet feet, moss, streaks)
        std::function<V3(const threepp::Vector3&, const threepp::Vector3&, const V3&)> weather;
    };

    // one stone: a rounded box of half extents (hx along u, hy along v, hz along w) centred on f.o
    inline void appendStone(MeshBuild& out, const Frame& f, float hx, float hy, float hz, const StoneLook& look) {
        using threepp::Vector3;
        const float r = std::min({look.round, 0.45f * hx, 0.45f * hy, 0.45f * hz});
        const int nx = std::clamp(static_cast<int>(hx / 0.05f), 2, 14), ny = std::clamp(static_cast<int>(hy / 0.05f), 2, 10);
        const int nz = std::clamp(static_cast<int>(hz / 0.12f), 1, 4);
        const auto lx = bevelLines(hx, r, nx), ly = bevelLines(hy, r, ny), lz = bevelLines(hz, r, nz);
        const unsigned start = out.vertexCount();
        const float ox = unit(look.seed) * 5.f, oy = unit(hash32(look.seed)) * 5.f;
        const Vector3 ix(hx - r, hy - r, hz - r);
        std::vector<float> edgeOf;// per emitted vertex: how near the stone's outline it is (0 middle, 1 edge)
        // a vertex of the box surface in the stone's own space (x along, y up, z out), rounded and dressed
        const auto shape = [&](Vector3 p) {
            const Vector3 c(std::clamp(p.x, -ix.x, ix.x), std::clamp(p.y, -ix.y, ix.y), std::clamp(p.z, -ix.z, ix.z));
            Vector3 d = p - c;
            const float len = d.length();
            if (len > 1e-6f) p = c + d * (r / len);
            // not square: the outline wanders, one end stands taller, the top is off to one side of the bottom
            {
                const float ex = smooth(0.55f, 1.f, std::abs(p.x) / hx), ey = smooth(0.55f, 1.f, std::abs(p.y) / hy);
                const float wx = look.wobble * fbm3(p.y * 5.f + ox, p.z * 3.f, oy, 2, look.seed + 11);
                const float wy = look.wobble * fbm3(p.x * 5.f + oy, p.z * 3.f, ox, 2, look.seed + 12);
                p.x += (p.x > 0 ? 1.f : -1.f) * ex * wx;
                p.y += (p.y > 0 ? 1.f : -1.f) * ey * wy;
                p.y *= 1.f + look.taper * (p.x / hx);
                p.x += 0.5f * look.shear * (p.y / hy);
            }
            // the face swells: most on the front, a little on the sides
            const float sx = 1.f - (p.x / hx) * (p.x / hx), sy = 1.f - (p.y / hy) * (p.y / hy);
            const float bulge = look.pillow * std::pow(std::max(sx, 0.f), 0.6f) * std::pow(std::max(sy, 0.f), 0.6f);
            const float n1 = fbm3(p.x * 6.f + ox, p.y * 6.f + oy, p.z * 6.f, 3, look.seed);
            if (p.z > -hz * 0.5f) p.z += bulge * (0.75f + 0.5f * n1) * smooth(-hz * 0.5f, hz, p.z);
            // chips: bites out of the arrises
            const float edge = std::max({std::abs(p.x) / hx, std::abs(p.y) / hy});
            const float chip = std::max(0.f, fbm3(p.x * 14.f + oy, p.y * 14.f, p.z * 14.f + ox, 3, look.seed + 7) - 0.1f);
            const float bite = look.chips * chip * smooth(0.7f, 1.f, edge) * smooth(-hz * 0.3f, hz, p.z);
            Vector3 inward(p.x / hx, p.y / hy, 0.4f);
            inward.normalize();
            p -= inward * bite;
            // a little roughness everywhere
            p.z += 0.003f * fbm3(p.x * 30.f, p.y * 30.f, p.z * 30.f + ox, 2, look.seed + 3);
            return p;
        };
        // six faces as grids, each emitted with its own vertices (the rounding makes their shared edges meet)
        // each face takes the texture over its own two axes (mode 0: x and y, 1: x and z, 2: z and y), or the faces
        // that turn the corners would wear one texel stretched across them
        const auto face = [&](const std::vector<float>& la, const std::vector<float>& lb, const std::function<Vector3(float, float)>& place, bool flip, int mode) {
            const unsigned base = out.vertexCount();
            for (size_t j = 0; j < lb.size(); ++j) {
                for (size_t i = 0; i < la.size(); ++i) {
                    const Vector3 placed = place(la[i], lb[j]);
                    const Vector3 local = shape(placed);
                    const Vector3 world = f.at(local.x, local.y, local.z);
                    edgeOf.push_back(std::max(std::abs(placed.x) / hx, std::abs(placed.y) / hy));
                    // texture over the wall's face, offset per stone
                    const float u = (mode == 2 ? local.z + hz + 0.37f : local.x + hx) / look.tile + ox;
                    const float v = (mode == 1 ? local.z + hz + 0.61f : local.y + hy) / look.tile + oy;
                    out.vertex(world, f.w, u, v, look.tint);
                }
            }
            const unsigned w = static_cast<unsigned>(la.size());
            for (unsigned j = 0; j + 1 < lb.size(); ++j) {
                for (unsigned i = 0; i + 1 < w; ++i) {
                    const unsigned a = base + j * w + i, b = a + 1, c = a + w, d = c + 1;
                    if (flip) out.index.insert(out.index.end(), {a, c, b, b, c, d});
                    else out.index.insert(out.index.end(), {a, b, c, b, d, c});
                }
            }
        };
        face(lx, ly, [&](float a, float b) { return Vector3(a, b, hz); }, false, 0);  // front
        face(lx, ly, [&](float a, float b) { return Vector3(a, b, -hz); }, true, 0);  // back
        face(lx, lz, [&](float a, float b) { return Vector3(a, hy, b); }, true, 1);   // top
        face(lx, lz, [&](float a, float b) { return Vector3(a, -hy, b); }, false, 1); // bottom
        face(lz, ly, [&](float a, float b) { return Vector3(hx, b, a); }, true, 2);   // right
        face(lz, ly, [&](float a, float b) { return Vector3(-hx, b, a); }, false, 2); // left
        // weld the faces' shared border vertices by position, so normals are smooth across the rounded arrises
        {
            std::unordered_map<long long, unsigned> seen;
            std::vector<unsigned> remap(out.vertexCount() - start);
            const auto q = [](float v) { return static_cast<long long>(std::lround(v * 20000.f) + (1 << 20)) & 0x1FFFFF; };
            for (unsigned i = start; i < out.vertexCount(); ++i) {
                const long long key = q(out.pos[3 * i]) | (q(out.pos[3 * i + 1]) << 21) | (q(out.pos[3 * i + 2]) << 42);
                auto it = seen.find(key);
                remap[i - start] = it == seen.end() ? (seen[key] = i) : it->second;
            }
            for (size_t t = 0; t < out.index.size(); ++t) {
                if (out.index[t] >= start) out.index[t] = remap[out.index[t] - start];
            }
        }
        out.smoothNormals(start);
        for (unsigned i = start; i < out.vertexCount(); ++i) {
            const Vector3 p(out.pos[3 * i], out.pos[3 * i + 1], out.pos[3 * i + 2]);
            const Vector3 n(out.nor[3 * i], out.nor[3 * i + 1], out.nor[3 * i + 2]);
            V3 c = look.weather ? look.weather(p, n, look.tint) : look.tint;
            const float edge = edgeOf[i - start];
            // grime in the arrises, moss in clumps along the edges and on the tops, where water stands
            c = c * (1.f - 0.22f * smooth(0.82f, 1.f, edge));
            if (look.moss > 0.f) {
                const float clump = smooth(-0.15f, 0.35f, fbm3(p.x * 2.6f, p.y * 2.6f, p.z * 2.6f, 3, 4711u));
                const float where = 0.75f * smooth(0.8f, 1.f, edge) + smooth(0.25f, 0.8f, n.y);
                const float m = clamp01(look.moss * clump * where);
                const V3 mossCol = mix(srgb(66, 92, 34), srgb(112, 130, 56), clamp01(0.5f + fbm3(p.x * 7.f, p.y * 7.f, p.z * 7.f, 2, 4712u)));
                c = mix(c, mossCol, 0.85f * m);
            }
            out.col[3 * i] = c.r;
            out.col[3 * i + 1] = c.g;
            out.col[3 * i + 2] = c.b;
        }
    }

    // ---- walls -------------------------------------------------------------------------------------------------------
    struct Opening {
        float u0, u1, v0, v1;// along the wall from its start, up from its foot
    };

    struct WallSpec {
        Frame frame;          // o: the foot of the wall's outer face at its start; u along, v up, w out
        float length = 1, height = 1;
        float thickness = 0.6f;
        float batter = 0.f;   // the face leans back this much per metre of height
        float courseMin = 0.22f, courseMax = 0.4f;
        float stoneMin = 0.35f, stoneMax = 0.9f;
        float joint = 0.018f; // gap between stones
        float recess = 0.025f;// how far the mortar sits back from the stones' faces
        std::vector<Opening> openings;
        int parity = 0;       // corner bonding: which courses run through the corners
        bool bondStart = false, bondEnd = false;
        std::vector<float> courses;// shared course heights (so walls of one loop line up); filled if empty
        uint32_t seed = 1;
        V3 baseTint{1, 1, 1};
        float moss = 0.5f;    // how mossy its stones' edges and its joints are
        std::function<V3(const threepp::Vector3&, const threepp::Vector3&, const V3&)> weather;
    };

    inline std::vector<float> layCourses(float height, float lo, float hi, uint32_t seed) {
        Rng rng(seed);
        std::vector<float> c;
        float y = 0;
        while (y < height - 0.05f) {
            float h = rng.range(lo, hi);
            if (height - (y + h) < lo * 0.8f) h = height - y;
            c.push_back(h);
            y += h;
        }
        return c;
    }

    inline void buildWall(MeshBuild& stones, MeshBuild& mortar, WallSpec spec) {
        using threepp::Vector3;
        Rng rng(spec.seed);
        if (spec.courses.empty()) spec.courses = layCourses(spec.height, spec.courseMin, spec.courseMax, spec.seed + 1);
        const Frame& F = spec.frame;
        float y = 0;
        for (size_t ci = 0; ci < spec.courses.size(); ++ci) {
            const float ch = spec.courses[ci];
            // corner bonding: this course runs through the corner (longer) or stops short of it
            const bool through = ((static_cast<int>(ci) + spec.parity) % 2) == 0;
            const float uStart = spec.bondStart && !through ? spec.thickness : 0.f;
            const float uEnd = spec.bondEnd && !through ? spec.length - spec.thickness : spec.length;
            // the stones of the course, cut round the openings
            std::vector<std::pair<float, float>> runs{{uStart, uEnd}};
            for (const auto& o : spec.openings) {
                if (y + ch <= o.v0 + 0.01f || y >= o.v1 - 0.01f) continue;
                std::vector<std::pair<float, float>> next;
                for (auto [a, b] : runs) {
                    if (o.u1 <= a || o.u0 >= b) {
                        next.emplace_back(a, b);
                        continue;
                    }
                    if (o.u0 > a) next.emplace_back(a, o.u0);
                    if (o.u1 < b) next.emplace_back(o.u1, b);
                }
                runs = next;
            }
            const float yc = y + 0.5f * ch;
            const float lean = spec.batter * yc;// the face sits back by this at this height
            for (auto [a, b] : runs) {
                float u = a;
                bool first = true;
                while (u < b - 0.05f) {
                    float len = rng.range(spec.stoneMin, spec.stoneMax);
                    if (first && spec.bondStart && a <= 0.f && through) len = std::max(len, spec.thickness + rng.range(0.15f, 0.4f));// a quoin
                    if (b - (u + len) < spec.stoneMin * 0.7f) len = b - u;
                    first = false;
                    const float uc = u + 0.5f * len;
                    // the stone's depth: most go half into the wall; a stone that turns a corner goes the wall's whole
                    // thickness, or the corner shows the core behind it
                    const bool corner = (spec.bondStart && u < spec.thickness + 0.01f) || (spec.bondEnd && u + len > spec.length - spec.thickness - 0.01f);
                    const float depth = corner ? spec.thickness : std::min(spec.thickness, rng.range(0.25f, 0.45f));
                    StoneLook look;
                    look.seed = rng.next();
                    {
                        static const std::array<V3, 5> hues{V3(0.78f, 0.77f, 0.75f), V3(0.84f, 0.76f, 0.64f), V3(0.70f, 0.64f, 0.56f), V3(0.80f, 0.68f, 0.56f), V3(0.62f, 0.62f, 0.62f)};
                        look.tint = spec.baseTint * hues[rng.next() % hues.size()] * rng.range(0.72f, 1.02f);
                    }
                    look.round = rng.range(0.025f, 0.055f);
                    look.pillow = rng.range(0.015f, 0.045f);
                    look.chips = rng.range(0.01f, 0.028f);
                    look.shear = rng.range(-0.035f, 0.035f);
                    look.taper = rng.range(-0.09f, 0.09f);
                    look.wobble = rng.range(0.008f, 0.022f);
                    look.moss = spec.moss * rng.range(0.6f, 1.3f);
                    look.weather = spec.weather;
                    // each a little short of its place: the joints are not all one width
                    const float slackX = rng.range(0.f, 0.014f), slackY = rng.range(0.f, 0.018f);
                    Frame sf = F;
                    const float proud = rng.range(-0.004f, 0.012f);// not every face is in one plane
                    // laid by hand: each stone rolled a little and bedded a little high or low in its joint
                    {
                        const float roll = 0.012f * rng.normal() * std::min(1.f, 0.6f / std::max(len, 0.3f));
                        const float c = std::cos(roll), sn = std::sin(roll);
                        sf.u = F.u * c + F.v * sn;
                        sf.v = F.v * c - F.u * sn;
                    }
                    const float bed = rng.range(-0.004f, 0.004f);
                    if (ch > 0.3f && len < 0.8f && rng() < 0.18f) {
                        // two thin stones stacked in this course's height
                        const float split = rng.range(0.4f, 0.6f) * ch;
                        sf.o = F.at(uc, y + 0.5f * split + bed, -lean - depth * 0.5f + proud);
                        appendStone(stones, sf, 0.5f * len - 0.5f * spec.joint - slackX, 0.5f * split - 0.5f * spec.joint - slackY, 0.5f * depth, look);
                        look.seed = rng.next();
                        look.tint = look.tint * rng.range(0.85f, 1.1f);
                        sf.o = F.at(uc, y + split + 0.5f * (ch - split), -lean - depth * 0.5f + rng.range(-0.004f, 0.012f));
                        appendStone(stones, sf, 0.5f * len - 0.5f * spec.joint - slackX, 0.5f * (ch - split) - 0.5f * spec.joint - slackY, 0.5f * depth, look);
                    } else {
                        sf.o = F.at(uc, yc + bed, -lean - depth * 0.5f + proud);
                        appendStone(stones, sf, 0.5f * len - 0.5f * spec.joint - slackX, 0.5f * ch - 0.5f * spec.joint - slackY, 0.5f * depth, look);
                    }
                    u += len;
                }
            }
            // the course's mortar core, set back from the faces: its own, and at the corners the face of the wall it
            // meets (flush with that face, it showed wherever a stone sat a few millimetres back)
            for (auto [a, b] : runs) {
                if (spec.bondStart && a <= 0.f) a += spec.recess;
                if (spec.bondEnd && b >= spec.length) b -= spec.recess;
                if (b - a < 0.01f) continue;
                // the core, well back, closes the wall; the face of the mortar in front of it is old lime: washed out
                // deeper in places, grimy, green with moss where it stays damp
                const float back = spec.recess + 0.045f;
                Frame mf = F;
                mf.o = F.at(0.5f * (a + b), yc, -lean - back - 0.5f * (spec.thickness - back));
                appendBox(mortar, mf, 0.5f * (b - a), 0.5f * ch, 0.5f * (spec.thickness - back), 0.9f, V3(0.5f, 0.48f, 0.44f));
                const int cols = std::max(1, static_cast<int>((b - a) / 0.14f)), rowsN = 3;
                const unsigned base = mortar.vertexCount();
                for (int r = 0; r <= rowsN; ++r) {
                    for (int k = 0; k <= cols; ++k) {
                        const float uu = a + (b - a) * static_cast<float>(k) / static_cast<float>(cols);
                        const float vv = y + ch * static_cast<float>(r) / static_cast<float>(rowsN);
                        const Vector3 probe = F.at(uu, vv, 0.f);
                        const float erosion = clamp01(0.5f + 0.9f * fbm3(probe.x * 1.7f, probe.y * 1.7f, probe.z * 1.7f, 3, spec.seed + 5));
                        const float recess = spec.recess * (0.55f + 1.4f * erosion);
                        const Vector3 p = F.at(uu, vv, -lean - recess);
                        V3 c = V3(0.95f, 0.92f, 0.86f) * (0.75f + 0.35f * clamp01(0.5f + fbm3(p.x * 4.f, p.y * 4.f, p.z * 4.f, 2, 77u))) * (1.f - 0.35f * erosion);
                        if (spec.weather) c = spec.weather(p, F.w, c);
                        const float clump = smooth(-0.2f, 0.3f, fbm3(p.x * 2.2f, p.y * 2.2f, p.z * 2.2f, 3, 4713u));
                        c = mix(c, srgb(70, 96, 38), clamp01(0.8f * spec.moss * clump * (0.4f + 0.6f * erosion)));
                        mortar.vertex(p, F.w, uu / 1.2f, vv / 1.2f, c);
                    }
                }
                const unsigned w = cols + 1;
                for (int r = 0; r < rowsN; ++r)
                    for (int k = 0; k < cols; ++k) {
                        const unsigned i0 = base + r * w + k, i1 = i0 + 1, i2 = i0 + w, i3 = i2 + 1;
                        mortar.index.insert(mortar.index.end(), {i0, i1, i2, i1, i3, i2});
                    }
            }
            y += ch;
        }
    }

    // a rectangular loop of four walls (outer faces on the rectangle), bonded at the corners
    struct Loop {
        float x0, x1, z0, z1;// the outer faces
        float base, top;     // feet and tops
        float thickness = 0.6f;
        float batter = 0.f;
        float courseMin = 0.22f, courseMax = 0.4f, stoneMin = 0.35f, stoneMax = 0.9f;
        std::array<std::vector<Opening>, 4> openings;// west (-x), south (+z), east (+x), north (-z), along each wall
        uint32_t seed = 1;
        V3 tint{1, 1, 1};
        float moss = 0.5f;
        std::function<V3(const threepp::Vector3&, const threepp::Vector3&, const V3&)> weather;
    };

    inline void buildLoop(MeshBuild& stones, MeshBuild& mortar, const Loop& L) {
        using threepp::Vector3;
        const auto courses = layCourses(L.top - L.base, L.courseMin, L.courseMax, L.seed);
        const Vector3 up(0, 1, 0);
        // walk round counter-clockwise seen from above: west wall runs +z, south +x, east -z, north -x
        const std::array<Frame, 4> frames{{
                {Vector3(L.x0, L.base, L.z0), Vector3(0, 0, 1), up, Vector3(-1, 0, 0)},
                {Vector3(L.x0, L.base, L.z1), Vector3(1, 0, 0), up, Vector3(0, 0, 1)},
                {Vector3(L.x1, L.base, L.z1), Vector3(0, 0, -1), up, Vector3(1, 0, 0)},
                {Vector3(L.x1, L.base, L.z0), Vector3(-1, 0, 0), up, Vector3(0, 0, -1)},
        }};
        const std::array<float, 4> lengths{L.z1 - L.z0, L.x1 - L.x0, L.z1 - L.z0, L.x1 - L.x0};
        for (int k = 0; k < 4; ++k) {
            WallSpec w;
            w.frame = frames[k];
            w.length = lengths[k];
            w.height = L.top - L.base;
            w.thickness = L.thickness;
            w.batter = L.batter;
            w.courseMin = L.courseMin;
            w.courseMax = L.courseMax;
            w.stoneMin = L.stoneMin;
            w.stoneMax = L.stoneMax;
            w.courses = courses;
            w.parity = k % 2;
            w.bondStart = w.bondEnd = true;
            w.openings = L.openings[k];
            w.seed = L.seed * 31u + static_cast<uint32_t>(k) * 7919u;
            w.baseTint = L.tint;
            w.moss = L.moss;
            w.weather = L.weather;
            buildWall(stones, mortar, w);
        }
    }

}// namespace mill

#endif
