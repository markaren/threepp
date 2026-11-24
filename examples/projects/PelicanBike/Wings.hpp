// The wings, folded with the wrist on the handlebar grip: an arm in a Z (the humerus back and down from the shoulder,
// the forearm forward to the wrist on the grip, the hand folded back along it), found by two-bone inverse
// kinematics from the shoulder and the grip; along the bones, the secondaries and primaries trail back over the
// flank, the coverts shingle over their bases, the tertials lie along the upper arm.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_WINGS_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_WINGS_HPP

#include "Feathers.hpp"
#include "PelicanBody.hpp"

namespace pb {

    struct WingPose {
        Vector3 S, E, W, H;   // shoulder, elbow, wrist, the hand's tip
        float side = 1.f;
    };

    // the torso's skin at (x, y): where a ray along z first meets it, and the outward normal there; false if the spot
    // is off the body
    inline bool bodySurfaceAt(const PelicanBody& a, float x, float y, float side, Vector3& p, Vector3& n) {
        const Loft& L = a.body;
        // the spine sample nearest in x among the torso
        size_t best = 0;
        float bd = 1e9f;
        for (size_t i = 0; i < L.spine.p.size(); ++i) {
            if (L.spine.s[i] > a.tNeckBase) break;
            const float d = std::abs(L.spine.p[i].x - x);
            if (d < bd) {
                bd = d;
                best = i;
            }
        }
        const float t = L.spine.s[best];
        const Station st = stationAt(L.stations, t);
        const float dy = (y - L.spine.p[best].y - st.shift) / std::max(L.spine.N[best].y, 0.2f);
        const float h = dy >= 0 ? st.up : st.down;
        if (std::abs(dy) >= h * 0.999f) return false;
        const float zz = st.w * std::pow(std::max(1.f - std::pow(std::abs(dy) / h, st.n), 0.f), 1.f / st.n);
        p = Vector3(x, y, side * zz);
        // the normal from the implicit form
        const float gx = std::pow(zz / st.w, st.n - 1.f) / st.w;
        const float gy = std::pow(std::abs(dy) / h, st.n - 1.f) / h * (dy >= 0 ? 1.f : -1.f);
        n = Vector3(0.f, gy, side * gx).normalize();
        return true;
    }

    struct Wings {
        FeatherSet primaries, secondaries, tertials, covertsGreater, covertsMedian, covertsLesser, wristDown;
        WingPose pose[2];   // 0: the right wing (+z), 1: the left
        MeshBuild arms;

        static Vector3 inPlane(const Vector3& dir, const Vector3& normal) { return (dir - normal * dir.dot(normal)).normalize(); }

        void build(const PelicanBody& a, const Vector3& gripRight, const Vector3& gripLeft, uint32_t seed) {
            for (int w = 0; w < 2; ++w) {
                const float s = w == 0 ? 1.f : -1.f;
                WingPose& P = pose[w];
                P.side = s;
                Rng rng(seed + static_cast<uint32_t>(w) * 977u);
                // shoulder: on the flank, high and forward of the middle
                Vector3 sp, sn;
                bodySurfaceAt(a, 0.115f, 0.085f, s, sp, sn);
                P.S = sp + sn * 0.012f;
                P.W = (w == 0 ? gripRight : gripLeft);
                // two bones
                const float la = 0.255f, lb = 0.30f;
                const Vector3 sw = P.W - P.S;
                const float d = std::min(sw.length(), la + lb - 0.01f);
                const Vector3 u = Vector3(sw).normalize();
                const float xx = (la * la - lb * lb + d * d) / (2.f * d);
                const float hh = std::sqrt(std::max(la * la - xx * xx, 0.f));
                Vector3 pole(-0.55f, -1.0f, 0.75f * s);
                pole = (pole - u * pole.dot(u)).normalize();
                P.E = P.S + u * xx + pole * hh;
                // the elbow must stand clear of the flank
                {
                    Vector3 ep, en;
                    if (bodySurfaceAt(a, P.E.x, P.E.y, s, ep, en) && std::abs(P.E.z) < std::abs(ep.z) + 0.02f) P.E.z = ep.z + s * 0.025f;
                }
                // the hand folds back along the forearm, a little outside it
                const Vector3 fore = (P.E - P.W).normalize();
                P.H = P.W + fore * 0.115f + Vector3(0.f, 0.f, s * 0.012f);

                // bones as thick skin sleeves
                {
                    std::vector<Vector3> pts;
                    for (int i = 0; i <= 8; ++i) pts.push_back(lerp3(P.S, P.E, static_cast<float>(i) / 8.f));
                    for (int i = 1; i <= 8; ++i) pts.push_back(lerp3(P.E, P.W, static_cast<float>(i) / 8.f));
                    for (int i = 1; i <= 4; ++i) pts.push_back(lerp3(P.W, P.H, static_cast<float>(i) / 4.f));
                    const auto sm = splinePoints(pts, 3);
                    SweepOptions o;
                    o.seg = 12;
                    o.colour = srgb(244, 238, 228);
                    appendSweep(arms, sm, radiiAlong(static_cast<int>(sm.size()), {{0.f, 0.03f}, {0.35f, 0.032f}, {0.45f, 0.028f}, {0.82f, 0.02f}, {1.f, 0.012f}}), o);
                }

                const Vector3 back = Vector3(-0.97f, -0.14f, 0.16f * s).normalize();
                const auto planeNormal = [&](const Vector3& at) {
                    Vector3 pp, nn;
                    if (bodySurfaceAt(a, at.x, at.y, s, pp, nn)) return nn;
                    return Vector3(0.f, 0.35f, 0.94f * s).normalize();
                };
                // lay one feather: its root on the skin (or on the arm where it leaves the body), pointing `back` turned by `fan`
                const auto lay = [&](FeatherSet& set, const Vector3& at, float push, float fan, float len, float wid, float lift, float jitter, const V3& tint = V3(1, 1, 1)) {
                    Vector3 root = at, n;
                    Vector3 sp, sn;
                    if (bodySurfaceAt(a, at.x, at.y, s, sp, sn)) {
                        // the bone's height above the skin: close to it the feathers lie on the skin
                        const float gap = std::abs(at.z) - std::abs(sp.z);
                        const float k = smooth(0.03f, 0.10f, gap);
                        root = lerp3(sp, at, k);
                        n = lerp3(sn, planeNormal(at), k).normalize();
                    } else {
                        n = planeNormal(at);
                    }
                    const Vector3 bx = inPlane(back, n);
                    Vector3 up = Vector3(0, 1, 0);
                    up = (up - n * up.dot(n) - bx * up.dot(bx));
                    if (up.lengthSq() < 1e-6f) up = Vector3(0, 1, 0);
                    up.normalize();
                    const Vector3 dir = (bx * std::cos(fan) + up * std::sin(fan)).normalize();
                    set.add(root + n * push, dir, n, len, wid, lift, jitter, tint);
                };

                // secondaries along the forearm, elbow to wrist
                const int nSec = 34;
                for (int i = 0; i < nSec; ++i) {
                    const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(nSec);
                    const Vector3 at = lerp3(P.E, P.W, 0.04f + 0.9f * u);
                    const float len = (0.20f + 0.075f * std::sin(PI * std::pow(u, 0.8f))) * rng.range(0.96f, 1.04f);
                    lay(secondaries, at, 0.016f + 0.0003f * static_cast<float>(i), -0.16f + 0.26f * u, len, 0.046f, 0.05f + 0.03f * rng(), rng.range(-0.05f, 0.05f), V3(0.85f + 0.3f * rng(), 0.85f + 0.3f * rng(), 0.85f + 0.3f * rng()));
                }
                // tertials along the upper arm
                for (int i = 0; i < 6; ++i) {
                    const float u = (static_cast<float>(i) + 0.5f) / 6.f;
                    const Vector3 at = lerp3(P.E, P.S, 0.1f + 0.7f * u);
                    lay(tertials, at, 0.026f + 0.001f * static_cast<float>(i), -0.2f + 0.2f * u, 0.24f - 0.03f * u, 0.07f, 0.05f, rng.range(-0.05f, 0.05f));
                }
                // primaries on the hand, ten, the outer ones longest
                const int nPri = 11;
                for (int i = 0; i < nPri; ++i) {
                    const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(nPri);
                    const Vector3 at = lerp3(P.W, P.H, 0.1f + 0.9f * u);
                    const float len = (0.30f + 0.13f * std::sin(PI * 0.5f * std::min(u * 1.25f, 1.f))) * (i == nPri - 1 ? 0.93f : 1.f);
                    lay(primaries, at, 0.030f + 0.0004f * static_cast<float>(i), 0.06f - 0.16f * u, len, 0.040f, 0.04f + 0.03f * rng(), rng.range(-0.04f, 0.04f), V3(0.85f + 0.3f * rng(), 0.85f + 0.3f * rng(), 0.85f + 0.3f * rng()));
                }

                // coverts: three rows along the forearm and the hand, shingled toward the leading edge
                const auto rows = [&](FeatherSet& set, const Vector3& a0, const Vector3& a1, int n, float push, float len, float wid, float lead, float lift) {
                    const Vector3 n0 = planeNormal(lerp3(a0, a1, 0.5f));
                    const Vector3 bx = inPlane(back, n0);
                    for (int i = 0; i < n; ++i) {
                        const float u = (static_cast<float>(i) + rng()) / static_cast<float>(n);
                        // toward the leading edge: forward of the bone, against the feathers' direction
                        const Vector3 at = lerp3(a0, a1, u) - bx * lead;
                        lay(set, at, push + 0.0006f * static_cast<float>(i % 5), -0.35f + 0.5f * u + rng.range(-0.05f, 0.05f), len * rng.range(0.9f, 1.1f), wid, lift + 0.06f * rng(), rng.range(-0.1f, 0.1f), V3(0.98f, 0.98f, 0.98f));
                    }
                };
                rows(covertsGreater, P.E, P.W, 30, 0.036f, 0.21f, 0.064f, 0.0f, 0.06f);
                rows(covertsMedian, P.E, P.W, 36, 0.044f, 0.15f, 0.054f, 0.02f, 0.09f);
                rows(covertsLesser, P.E, P.W, 44, 0.052f, 0.10f, 0.046f, 0.045f, 0.12f);
                rows(covertsGreater, P.W, P.H, 10, 0.050f, 0.15f, 0.055f, 0.0f, 0.06f);
                rows(covertsMedian, P.W, P.H, 12, 0.062f, 0.105f, 0.048f, 0.02f, 0.09f);
                rows(covertsLesser, P.S, P.E, 22, 0.050f, 0.1f, 0.052f, 0.03f, 0.1f);
                rows(covertsMedian, P.S, P.E, 20, 0.04f, 0.13f, 0.06f, 0.0f, 0.07f);
            }
        }
    };

}// namespace pb

#endif
