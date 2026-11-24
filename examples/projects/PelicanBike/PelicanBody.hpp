// The pelican's body: torso, S-curved neck and head lofted from sections along one spine; the big bill with its hooked
// nail, the two thin lower mandibles and the pouch hanging between them; the eyes. Built in the bird's own frame:
// +x forward along the body, +y up, +z to its right, centred on the middle of the torso, in metres, a bird of about
// a metre and a half from bill tip to tail tip. The skin carries the colours of the bare parts; the plumage lies
// on top of it (Plumage.hpp).

#ifndef THREEPP_EXAMPLE_PELICANBIKE_PELICANBODY_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_PELICANBODY_HPP

#include "Parts.hpp"

namespace pb {

    struct PelicanBody {
        // ---- the spine and its sections ----------------------------------------------------------------------------
        Loft body;
        std::vector<Vector3> ctrl;
        float tNeckBase = 0, tHead = 0, tEye = 0;
        Vector3 headEnd, headDir;    // the head's last point and its heading
        float billDrop = 0.62f;      // the bill points this many radians below the head's heading

        // bill
        Loft upper;
        float billLength = 0.34f;
        Vector3 billOrigin;
        Quaternion billRot;          // bill frame -> bird frame
        Vector3 billAxis;            // unit, along the bill

        std::vector<Vector3> eyeAt;  // both eyes

        // pouch: samples of the surface, in the bird frame, (u along, v across)
        int pouchU = 56, pouchV = 18;

        PelicanBody() {
            ctrl = {
                    {-0.47f, -0.035f, 0.f},// tail tip
                    {-0.33f, -0.010f, 0.f},// tail base
                    {-0.17f, 0.010f, 0.f}, // back, hips
                    {0.03f, 0.020f, 0.f},  // middle
                    {0.20f, 0.045f, 0.f},  // chest
                    {0.305f, 0.105f, 0.f}, // neck base
                    {0.352f, 0.195f, 0.f}, // lower neck, forward
                    {0.322f, 0.285f, 0.f}, // mid neck, the S's back curve
                    {0.335f, 0.355f, 0.f}, // upper neck
                    {0.395f, 0.405f, 0.f}, // head back
                    {0.478f, 0.418f, 0.f}, // head front
            };
            body.spine.build(ctrl, 600);
            const auto sOf = [&](const Vector3& q) {
                float best = 1e9f, bs = 0;
                for (size_t i = 0; i < body.spine.p.size(); ++i) {
                    const float d = (body.spine.p[i] - q).lengthSq();
                    if (d < best) {
                        best = d;
                        bs = body.spine.s[i];
                    }
                }
                return bs;
            };
            struct K {
                int i;
                float w, up, down, n, shift;
            };
            const std::vector<K> k = {
                    {0, 0.010f, 0.004f, 0.004f, 2.0f, 0.f},
                    {1, 0.055f, 0.032f, 0.040f, 2.2f, 0.f},
                    {2, 0.115f, 0.090f, 0.095f, 2.3f, 0.f},
                    {3, 0.146f, 0.118f, 0.124f, 2.3f, 0.f},
                    {4, 0.138f, 0.108f, 0.134f, 2.3f, 0.f},
                    {5, 0.076f, 0.062f, 0.094f, 2.1f, -0.008f},
                    {6, 0.047f, 0.045f, 0.045f, 2.0f, 0.f},
                    {7, 0.037f, 0.036f, 0.036f, 2.0f, 0.f},
                    {8, 0.032f, 0.032f, 0.032f, 2.0f, 0.f},
                    {9, 0.031f, 0.033f, 0.030f, 2.1f, 0.003f},
                    {10, 0.026f, 0.027f, 0.022f, 2.2f, 0.002f},
            };
            for (const auto& q : k) body.stations.push_back({sOf(ctrl[static_cast<size_t>(q.i)]), q.w, q.up, q.down, q.n, q.shift});
            std::sort(body.stations.begin(), body.stations.end(), [](const Station& a, const Station& b) { return a.s < b.s; });
            body.stations.front().s = 0.f;
            body.stations.back().s = 1.f;
            tNeckBase = sOf(ctrl[5]);
            tHead = sOf(ctrl[9]);
            tEye = 0.975f;
            Vector3 pos, tan, nor;
            body.spine.at(1.f, pos, tan, nor);
            headEnd = pos;
            headDir = tan;
            buildBill();
        }

        // the head's heading below the horizontal, then the bill's drop
        void buildBill() {
            const float headAngle = std::atan2(headDir.y, headDir.x);
            const float a = headAngle - billDrop;
            billAxis = Vector3(std::cos(a), std::sin(a), 0.f);
            billRot.setFromAxisAngle(Vector3(0, 0, 1), a);
            billOrigin = headEnd - billAxis * 0.022f;
            // the upper mandible: straight, flat below and ridged above, the tip hooked down
            const float L = billLength;
            const std::vector<Vector3> c = {{0.f, 0.f, 0.f}, {0.10f * L, 0.f, 0.f}, {0.4f * L, -0.003f, 0.f}, {0.72f * L, -0.010f, 0.f}, {0.92f * L, -0.020f, 0.f}, {L, -0.040f, 0.f}};
            upper.spine.build(c, 300);
            upper.rings = 140;
            upper.around = 40;
            const auto sOf = [&](const Vector3& q) {
                float best = 1e9f, bs = 0;
                for (size_t i = 0; i < upper.spine.p.size(); ++i) {
                    const float d = (upper.spine.p[i] - q).lengthSq();
                    if (d < best) {
                        best = d;
                        bs = upper.spine.s[i];
                    }
                }
                return bs;
            };
            upper.stations = {
                    {sOf(c[0]), 0.034f, 0.026f, 0.012f, 2.6f, 0.f},
                    {sOf(c[1]), 0.036f, 0.020f, 0.006f, 3.0f, -0.001f},
                    {sOf(c[2]), 0.032f, 0.014f, 0.004f, 3.2f, -0.002f},
                    {sOf(c[3]), 0.024f, 0.010f, 0.004f, 3.0f, -0.003f},
                    {sOf(c[4]), 0.015f, 0.009f, 0.005f, 2.6f, -0.003f},
                    {1.f, 0.0045f, 0.0075f, 0.0050f, 2.2f, 0.f}};
            upper.stations.front().s = 0.f;
            eyeAt = {};
        }

        [[nodiscard]] Vector3 toBird(const Vector3& billLocal) const { return billOrigin + Vector3(billLocal).applyQuaternion(billRot); }

        // the bill's half width and the height of the lower edge at a distance x along it
        [[nodiscard]] float billHalfWidth(float x) const {
            const float t = clamp01(x / billLength);
            const Station st = stationAt(upper.stations, t);
            return st.w;
        }

        // the skin colours of the body: white feather down with bare orange skin at the face
        static V3 skinColour(float t, float a, float tHead) {
            const V3 white = srgb(247, 243, 234), cream = srgb(246, 232, 190), orange = srgb(236, 150, 52);
            V3 c = white;
            // a creamy-yellow wash across the breast and the nape
            c = mill::mix(c, cream, 0.35f * smooth(0.4f, 0.62f, t) * (1.f - smooth(0.62f, 0.7f, t)));
            if (t > tHead - 0.03f) {
                const float lateral = std::pow(std::abs(std::cos(a)), 1.5f);
                const float face = smooth(tHead + 0.0f, tHead + 0.04f, t);
                c = mill::mix(c, orange, face * smooth(0.25f, 0.8f, lateral) * 0.95f);
            }
            return c;
        }

        void buildBodyMesh(MeshBuild& out) const {
            body.build(out, [&](float t, float a) { return skinColour(t, a, tHead); }, 3.f);
        }

        // ---- the bill, as meshes ----------------------------------------------------------------------------------------
        void buildUpperMesh(MeshBuild& out) const {
            MeshBuild mb;
            upper.build(mb, [](float t, float a) {
                const V3 orange = srgb(238, 144, 40), yellow = srgb(246, 188, 70), dark = srgb(205, 110, 40);
                const float ridge = smooth(0.2f, 0.9f, std::sin(a));
                V3 c = mill::mix(orange, yellow, 0.4f * ridge);
                c = mill::mix(c, dark, 0.35f * smooth(0.6f, 1.f, t) * (std::sin(a) < 0 ? 1.f : 0.f));
                // the nail at the tip: paler horn
                c = mill::mix(c, srgb(230, 205, 150), smooth(0.90f, 0.985f, t));
                return c;
            }, 6.f);
            // move into the bird frame
            for (size_t i = 0; i < mb.pos.size(); i += 3) {
                Vector3 p(mb.pos[i], mb.pos[i + 1], mb.pos[i + 2]);
                p = toBird(p);
                mb.pos[i] = p.x;
                mb.pos[i + 1] = p.y;
                mb.pos[i + 2] = p.z;
                Vector3 n(mb.nor[i], mb.nor[i + 1], mb.nor[i + 2]);
                n.applyQuaternion(billRot);
                mb.nor[i] = n.x;
                mb.nor[i + 1] = n.y;
                mb.nor[i + 2] = n.z;
            }
            out.append(mb);
        }

        // the point at distance x along the bill's underside, in the bill frame (the line the rami follow)
        [[nodiscard]] Vector3 ramusPoint(float x, float side) const {
            const float t = clamp01(x / billLength);
            Vector3 pos, tan, nor;
            upper.spine.at(t, pos, tan, nor);
            const Station st = stationAt(upper.stations, t);
            const float hw = st.w * 0.92f;
            return pos + nor * (st.shift - st.down - 0.0035f) + Vector3(0, 0, hw * side);
        }

        void buildLowerMesh(MeshBuild& out) const {
            for (int s : {1, -1}) {
                std::vector<Vector3> pts;
                const int n = 60;
                for (int i = 0; i <= n; ++i) {
                    const float x = billLength * 0.985f * static_cast<float>(i) / static_cast<float>(n);
                    // the ramus converges a little near the tip
                    Vector3 q = ramusPoint(x, static_cast<float>(s));
                    pts.push_back(toBird(q));
                }
                MeshBuild mb;
                SweepOptions o;
                o.seg = 12;
                o.aspect = 1.6f;
                o.up = Vector3(0, 0, 1);
                std::vector<float> r = radiiAlong(n + 1, {{0.f, 0.0048f}, {0.5f, 0.0036f}, {1.f, 0.0022f}});
                appendSweep(mb, pts, r, o);
                // colour: the lower bill is paler, pinker
                for (size_t i = 0; i < mb.col.size(); i += 3) {
                    mb.col[i] = 0.82f;
                    mb.col[i + 1] = 0.46f;
                    mb.col[i + 2] = 0.20f;
                }
                out.append(mb);
            }
        }

        // ---- the pouch ----------------------------------------------------------------------------------------------------
        // A skin hung between the two rami, back into the throat, where it joins the neck's front. (u: 0 at the throat,
        // 1 at the tip of the bill; v: -1 .. 1 across.) The pouch sags under its own weight and is a little slack.
        [[nodiscard]] Vector3 pouchPoint(float u, float v, Vector3* nrm = nullptr, float slack = 1.f) const {
            const float throat = 0.07f;   // how far back from the bill's base the pouch reaches, in bill lengths
            const float ub = u * (1.f + throat) - throat;   // along the bill, in bill lengths (negative = the throat)
            const float x = billLength * ub;
            // the two edges
            Vector3 edge;
            float depth;
            if (ub >= 0.f) {
                const Vector3 a = ramusPoint(std::max(x, 0.f), 1.f), b = ramusPoint(std::max(x, 0.f), -1.f);
                edge = lerp3(b, a, 0.5f + 0.5f * v);
                const float uu = clamp01(ub);
                // deepest a third of the way along the bill, ending at nothing at the tip
                depth = 0.050f * std::pow(std::max(std::sin(PI * std::pow(uu, 0.7f)), 0.f), 0.8f) * (1.f - 0.15f * uu);
                depth *= 1.f + 0.0f * slack;
            } else {
                // behind the bill's base: the ramus lines run on straight back, closing into the neck
                const Vector3 a = ramusPoint(0.f, 1.f), b = ramusPoint(0.f, -1.f);
                const float f = -ub / throat;   // 0 at the base .. 1 at the throat's end
                Vector3 ea = a + Vector3(-1, 0.16f, 0) * (0.5f * f * billLength * throat * 0.7f), eb = b + Vector3(-1, 0.16f, 0) * (0.5f * f * billLength * throat * 0.7f);
                ea.z *= 1.f - 0.55f * f;
                eb.z *= 1.f - 0.55f * f;
                edge = lerp3(eb, ea, 0.5f + 0.5f * v);
                depth = 0.050f * (1.f - f) * (1.f - f) * 0.8f;
            }
            const float sag = depth * std::pow(std::max(1.f - v * v, 0.f), 0.75f);
            Vector3 p = edge + Vector3(0, -sag, 0);
            if (nrm) *nrm = Vector3(0, -1, 0);
            return p;
        }

        void buildPouchMesh(MeshBuild& out, bool outside) const {
            const unsigned base = out.vertexCount();
            for (int i = 0; i <= pouchU; ++i)
                for (int j = 0; j <= pouchV; ++j) {
                    const float u = static_cast<float>(i) / static_cast<float>(pouchU), v = static_cast<float>(j) / static_cast<float>(pouchV) * 2.f - 1.f;
                    Vector3 q = pouchPoint(u, v);
                    // in the bill's frame to the bird's
                    q = toBird(q);
                    // pale pink-orange skin, veins toward the middle, a flush at the edge
                    const V3 skin = srgb(238, 156, 108), pale = srgb(246, 198, 150);
                    out.vertex(q, Vector3(0, -1, 0), u * 2.f, (v * 0.5f + 0.5f) * 2.f, mill::mix(pale, skin, 0.5f + 0.5f * std::abs(v)));
                }
            const unsigned row = static_cast<unsigned>(pouchV + 1);
            for (int i = 0; i < pouchU; ++i)
                for (int j = 0; j < pouchV; ++j) {
                    const unsigned a = base + i * row + j, b = a + 1, c = a + row, d = c + 1;
                    if (outside) out.index.insert(out.index.end(), {a, c, b, b, c, d});
                    else out.index.insert(out.index.end(), {a, b, c, b, d, c});
                }
            out.smoothNormals(base);
        }

        // ---- the eyes --------------------------------------------------------------------------------------------------------
        [[nodiscard]] Vector3 eyePosition(int side) const {
            Vector3 pos, tan, nor;
            body.spine.at(tEye - 0.01f, pos, tan, nor);
            const Station st = stationAt(body.stations, tEye - 0.01f);
            return pos + nor * (st.shift + 0.008f) + Vector3(0, 0, (st.w * 0.9f) * static_cast<float>(side));
        }
    };

}// namespace pb

#endif
