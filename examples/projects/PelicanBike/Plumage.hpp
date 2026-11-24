// The pelican's plumage: contour feathers laid over the skin in rows from the breast back to the tail, smaller on the
// neck and smaller again on the head, bare round the eye; the tail's feathers; the nape's yellow crest.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_PLUMAGE_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_PLUMAGE_HPP

#include "Feathers.hpp"
#include "PelicanBody.hpp"

namespace pb {

    struct Plumage {
        FeatherSet body, neck, head, tail, crest, down;

        // does this spot of the skin go bare (the face)?
        static bool bare(const PelicanBody& a, float t, float angle) {
            if (t < a.tHead - 0.03f) return false;
            const float lateral = std::pow(std::abs(std::cos(angle)), 1.5f);
            const float face = smooth(a.tHead + 0.0f, a.tHead + 0.04f, t);
            return face * smooth(0.25f, 0.8f, lateral) > 0.35f;
        }

        // ring-by-ring over the loft: `densityAt(t)` feathers a square metre
        void layContour(const PelicanBody& a, uint32_t seed) {
            Rng rng(seed);
            const Loft& L = a.body;
            const float length = L.spine.length;
            const float ds = 0.0035f;
            const int rings = static_cast<int>(length / ds);
            for (int r = 0; r < rings; ++r) {
                const float t0 = static_cast<float>(r) / static_cast<float>(rings);
                Vector3 pos, tan, nor;
                L.spine.at(t0, pos, tan, nor);
                const Station st = stationAt(L.stations, t0);
                // the ring's circumference, Ramanujan-ish
                const float hy = 0.5f * (st.up + st.down);
                const float circ = PI * (3.f * (st.w + hy) - std::sqrt((3.f * st.w + hy) * (st.w + 3.f * hy)));
                // region: torso, neck, head
                float density, len, wid, lift;
                if (t0 < a.tNeckBase) {
                    density = 9500.f;
                    len = 0.095f;
                    wid = 0.058f;
                    lift = 0.30f;
                } else if (t0 < a.tHead) {
                    const float f = smooth(a.tNeckBase, a.tNeckBase + 0.05f, t0);
                    density = 9500.f + f * 8000.f;
                    len = 0.095f - f * 0.055f;
                    wid = 0.058f - f * 0.03f;
                    lift = 0.3f - 0.1f * f;
                } else {
                    density = 22000.f;
                    len = 0.028f;
                    wid = 0.017f;
                    lift = 0.2f;
                }
                // toward the rump the feathers are shorter and lie closer, so the back ends tidily
                {
                    const float rear = 1.f - smooth(0.04f, 0.30f, t0);
                    len *= 1.f - 0.32f * rear;
                    lift *= 1.f - 0.55f * rear;
                }
                if (t0 < 0.05f) {
                    // the tail's root is the tail feathers' job
                    density *= smooth(0.0f, 0.05f, t0);
                }
                const float expect = density * circ * ds;
                int count = static_cast<int>(expect);
                if (rng() < expect - static_cast<float>(count)) ++count;
                for (int k = 0; k < count; ++k) {
                    const float t = std::clamp(t0 + rng.range(-0.5f, 0.5f) * ds / length, 0.f, 1.f);
                    const float ang = rng.range(0.f, TAU);
                    if (bare(a, t, ang)) continue;
                    Vector3 n;
                    const Vector3 p = L.surface(t, ang, &n);
                    Vector3 pp, tt, nn;
                    L.spine.at(t, pp, tt, nn);
                    // the flow: from the head toward the tail along the skin, with a little sideways drift on the flanks
                    Vector3 flow = tt * -1.f;
                    // on the neck feathers lie down it; on the belly they sweep back and a little up
                    flow = flow - n * flow.dot(n);
                    if (flow.lengthSq() < 1e-8f) continue;
                    flow.normalize();
                    const float side = std::cos(ang);
                    flow = flow + Vector3(0, 0, 1) * (0.12f * side * (1.f - std::abs(n.dot(Vector3(0, 0, 1)))));
                    // random size, a bit of roll and lift
                    const float sz = rng.range(0.8f, 1.25f);
                    const float tint = 0.96f + 0.06f * rng();
                    V3 tc(tint, tint, tint * (0.98f + 0.03f * rng()));
                    FeatherSet& dst = t < a.tNeckBase ? body : (t < a.tHead ? neck : head);
                    dst.add(p - n * 0.006f, flow, n, len * sz, wid * sz * rng.range(0.85f, 1.15f), lift * rng.range(0.55f, 1.5f), rng.range(-0.25f, 0.25f), tc);
                }
            }
        }

        // the tail: a fan of stiff feathers from the tail base, pointing back and a little down
        void layTail(const PelicanBody& a, uint32_t seed) {
            Rng rng(seed);
            Vector3 pos, tan, nor;
            a.body.spine.at(0.10f, pos, tan, nor);
            const int n = 18;
            for (int layer = 0; layer < 2; ++layer)
                for (int i = 0; i < n; ++i) {
                    const float f = (static_cast<float>(i) + 0.5f) / static_cast<float>(n) * 2.f - 1.f;   // -1 .. 1 across
                    const float fan = f * 0.34f;
                    const Vector3 dir = Vector3(-std::cos(fan), -0.10f - 0.07f * layer, std::sin(fan)).normalize();
                    const Vector3 root = pos + Vector3(0.0f, 0.0f - 0.012f * layer, f * 0.05f) + Vector3(0.f, 0.f, 0.f);
                    tail.add(root, dir, Vector3(0, 1, 0), 0.20f * rng.range(0.9f, 1.05f) * (1.f - 0.18f * std::abs(f)), 0.056f, 0.06f + 0.05f * layer, rng.range(-0.06f, 0.06f), V3(0.97f, 0.97f, 0.96f));
                }
        }

        // the nape's crest: a few pale-yellow feathers lying back from the back of the head
        void layCrest(const PelicanBody& a, uint32_t seed) {
            Rng rng(seed);
            const Loft& L = a.body;
            for (int i = 0; i < 18; ++i) {
                const float t = a.tHead + 0.004f + rng.range(0.f, 0.03f);
                const float ang = PI / 2.f + rng.range(-0.4f, 0.4f);
                Vector3 n;
                const Vector3 p = L.surface(t, ang, &n);
                Vector3 pp, tt, nn;
                L.spine.at(t, pp, tt, nn);
                Vector3 flow = (tt * -1.f + n * 0.18f).normalize();
                crest.add(p - n * 0.004f, flow, n, rng.range(0.04f, 0.075f), rng.range(0.008f, 0.012f), 0.22f + rng.range(0.f, 0.3f), rng.range(-0.3f, 0.3f));
            }
        }
    };

}// namespace pb

#endif
