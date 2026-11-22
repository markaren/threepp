// A heavy survey walker crossing a glacier.
//
// An original four-legged machine, modelled, textured and animated entirely in code: no
// asset files, nothing read from disk or the network. Three of them walk in loose column
// across the ice under an overcast sky with a warm break low over the sun; an expedition
// crew walks with the lead machine and two ducted-fan hoppers bank over the glacier.
//
// What a web version of this scene would use, and what it is here:
//   one standalone page, offline         one .cpp: every mesh and texture is generated at start-up
//   WebGL 2 renderer                     GLRenderer (OpenGL core profile)
//   full-window canvas                   Canvas, resizable; camera and renderer follow the window
//   requestAnimationFrame + delta time   Canvas::animate + Clock::getDelta (clamped)
//   GLSL snow and tread shaders          ShaderMaterial built from the engine's physical shader chunks
//   mouse/touch orbit and zoom           OrbitControls (a touch screen arrives as mouse input)
//   click-to-focus                       Raycaster on a click (press and release without a drag)
//   no text, logos or overlays           no ImGui, no HUD
//
// The walker: a long hull with the cab built into its nose, four legs hung from hip drums
// on its flanks, knees carried above the deck, telescoping hydraulic rams across the hip
// and knee, and broad three-toed feet. Each leg is solved every frame (hip yaw + two-bone
// IK) against a foot that stays fixed in the world while it bears weight. The body shifts
// its weight over the three standing feet before each foot lifts.
//
// The snow: a polar height map around the walk ring records every print, pressed with the
// shape of the sole that made it (pad, toes and cleats); a ShaderMaterial displaces the
// snow by it and shades the packed snow apart from fresh powder.
//
// Mouse:  drag = orbit, wheel = zoom, right drag = pan
//         click any part (walker, crew, hopper, snow) to fly in and follow it;
//         click the sky to return to the tracking shot
// Keys:   Space = pause, C = tracking shot
//
// Headless (stills for review):
//   glacier_walker --shot out.png [--time T] [--size WxH]
//                  [--view opening|wide|hatch|knee|ram|foot|prints|crew|hopper|below]
//                  [--click x,y]   the live click path at that pixel, then 1.5 s of follow
//                  [--bench N]     N live frames timed before the still is written
//   glacier_walker --film <dir> [--seconds 20] [--size 1920x1080]   PNG frames at 30 fps
// A Windows desktop clamps even a hidden window: keep portrait stills within it (608x1080).

#include "threepp/extras/terrain/MountainBackdrop.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/input/MouseListener.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"
#include "threepp/renderers/shaders/UniformsLib.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace threepp;

namespace {

    // ================================================================== small maths

    constexpr float kPi = 3.14159265358979f;
    constexpr float kTau = 6.28318530717959f;

    float deg(float d) { return d * kPi / 180.f; }
    float clamp01(float x) { return std::clamp(x, 0.f, 1.f); }
    float mixf(float a, float b, float t) { return a + (b - a) * t; }
    float smoothstepf(float e0, float e1, float x) {
        const float t = clamp01((x - e0) / (e1 - e0));
        return t * t * (3.f - 2.f * t);
    }
    float smoother(float x) {
        x = clamp01(x);
        return x * x * x * (x * (x * 6.f - 15.f) + 10.f);
    }
    float fractf(float x) { return x - std::floor(x); }

    const Vector3 kX{1, 0, 0}, kY{0, 1, 0}, kZ{0, 0, 1};

    Vector3 lerpV(const Vector3& a, const Vector3& b, float t) { return a + (b - a) * t; }
    Vector2 lerp2(const Vector2& a, const Vector2& b, float t) { return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t}; }
    Vector3 cross(const Vector3& a, const Vector3& b) { return Vector3().crossVectors(a, b); }
    Vector3 unit(Vector3 v) { return v.normalize(); }
    Vector3 neg(const Vector3& v) { return v * -1.f; }

    Matrix4 basisAt(const Vector3& p, const Vector3& x, const Vector3& y, const Vector3& z) {
        Matrix4 m;
        m.makeBasis(x, y, z);
        m.setPosition(p);
        return m;
    }
    Matrix4 translation(const Vector3& p) {
        Matrix4 m;
        m.setPosition(p);
        return m;
    }
    Matrix4 rotation(const Vector3& axis, float angle, const Vector3& p = Vector3()) {
        Quaternion q;
        q.setFromAxisAngle(axis, angle);
        Matrix4 m;
        m.compose(p, q, Vector3(1, 1, 1));
        return m;
    }
    Matrix4 scaling(float x, float y, float z) {
        Matrix4 m;
        m.makeScale(x, y, z);
        return m;
    }
    Matrix4 mul(Matrix4 a, const Matrix4& b) {
        a.multiply(b);
        return a;
    }
    // A lathe or cylinder built about +Y, turned to run along +X or +Z.
    Matrix4 alongX(const Vector3& p = Vector3()) { return rotation(kZ, -kPi / 2, p); }
    Matrix4 alongZ(const Vector3& p = Vector3()) { return rotation(kX, kPi / 2, p); }

    void place(Object3D& o, const Matrix4& m) {
        Vector3 s;
        m.decompose(o.position, o.quaternion, s);
        o.scale.copy(s);
    }

    Color srgb(float r, float g, float b) {
        return {std::pow(r, 2.2f), std::pow(g, 2.2f), std::pow(b, 2.2f)};
    }

    // ================================================================== noise (CPU side)

    std::uint32_t hashU(std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }
    float hash3(int x, int y, int seed) {
        const std::uint32_t h = hashU(static_cast<std::uint32_t>(x) * 0x9E3779B1U ^
                                      hashU(static_cast<std::uint32_t>(y) + 0x7F4A7C15U * static_cast<std::uint32_t>(seed + 1)));
        return static_cast<float>(h & 0xffffffU) / 16777215.f;
    }

    // Value noise whose lattice repeats every `period` cells (period <= 0: no repeat).
    float vnoise(float x, float y, int period, int seed) {
        const float fx = std::floor(x), fy = std::floor(y);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
        const float tx = x - fx, ty = y - fy;
        const float ux = tx * tx * (3 - 2 * tx), uy = ty * ty * (3 - 2 * ty);
        auto h = [&](int a, int b) {
            if (period > 0) {
                a = ((a % period) + period) % period;
                b = ((b % period) + period) % period;
            }
            return hash3(a, b, seed);
        };
        return mixf(mixf(h(ix, iy), h(ix + 1, iy), ux), mixf(h(ix, iy + 1), h(ix + 1, iy + 1), ux), uy);
    }

    float fbm(float x, float y, int octaves, int period, int seed) {
        float sum = 0, amp = 0.5f, norm = 0;
        for (int o = 0; o < octaves; ++o) {
            sum += amp * vnoise(x, y, period, seed + o * 17);
            norm += amp;
            amp *= 0.5f;
            x *= 2;
            y *= 2;
            if (period > 0) period *= 2;
        }
        return sum / norm;
    }

    // Gradient noise, roughly [-1, 1].
    float gnoise(float x, float y, int seed) {
        const float fx = std::floor(x), fy = std::floor(y);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
        const float tx = x - fx, ty = y - fy;
        auto grad = [&](int a, int b, float dx, float dy) {
            const float ang = hash3(a, b, seed) * kTau;
            return std::cos(ang) * dx + std::sin(ang) * dy;
        };
        const float ux = tx * tx * tx * (tx * (tx * 6 - 15) + 10);
        const float uy = ty * ty * ty * (ty * (ty * 6 - 15) + 10);
        const float n00 = grad(ix, iy, tx, ty), n10 = grad(ix + 1, iy, tx - 1, ty);
        const float n01 = grad(ix, iy + 1, tx, ty - 1), n11 = grad(ix + 1, iy + 1, tx - 1, ty - 1);
        return 1.4142f * mixf(mixf(n00, n10, ux), mixf(n01, n11, ux), uy);
    }

    // ================================================================== the glacier

    constexpr float kRingR = 120.f;// radius of the lead walker's circuit

    // Broad swells, gentle along the walk ring so the gait reads as walking, not climbing;
    // away from it the wind has shaped long drifts, and the ice rises slowly toward the ranges.
    float groundHeight(float x, float z) {
        const float r = std::sqrt(x * x + z * z);
        float h = 0.9f * gnoise(x / 95.f, z / 95.f, 3) + 0.35f * gnoise(x / 37.f, z / 37.f, 4);
        const float away = smoothstepf(45.f, 320.f, std::abs(r - kRingR));
        if (away > 0.f) {
            const float u = x * 0.8f + z * 0.6f, v = -x * 0.6f + z * 0.8f;// wind frame
            h += away * (5.f * gnoise(u / 420.f, v / 240.f, 8) + 1.6f * gnoise(u / 140.f, v / 60.f, 9));
        }
        h += 0.01f * std::max(0.f, r - 500.f);
        return h;
    }

    Vector3 groundNormal(float x, float z) {
        const float e = 0.5f;
        const float hx = groundHeight(x + e, z) - groundHeight(x - e, z);
        const float hz = groundHeight(x, z + e) - groundHeight(x, z - e);
        return unit(Vector3(-hx, 2 * e, -hz));
    }

    // ================================================================== geometry builder

    int segsFor(float len, float res) { return std::max(1, static_cast<int>(std::ceil(len / res))); }

    // Collects one material's triangles of a rigid part, in that part's local frame.
    // Everything added goes through the current transform; UVs are in metres / tile.
    struct Geo {
        std::vector<float> pos, nrm, uv, col;
        std::vector<unsigned int> idx;
        Matrix4 xf;
        Matrix3 nxf;
        bool mirrored = false;
        float tile = 4.f;

        Geo() { setTransform(Matrix4()); }

        void setTransform(const Matrix4& m) {
            xf = m;
            nxf.getNormalMatrix(m);
            mirrored = m.determinant() < 0;
        }

        [[nodiscard]] unsigned int count() const { return static_cast<unsigned int>(pos.size() / 3); }

        unsigned int vertex(Vector3 p, Vector3 n, float u, float v) {
            p.applyMatrix4(xf);
            n.applyMatrix3(nxf).normalize();
            pos.insert(pos.end(), {p.x, p.y, p.z});
            nrm.insert(nrm.end(), {n.x, n.y, n.z});
            uv.insert(uv.end(), {u / tile, v / tile});
            return count() - 1;
        }

        void tri(unsigned int a, unsigned int b, unsigned int c) {
            if (mirrored) std::swap(b, c);
            idx.insert(idx.end(), {a, b, c});
        }

        // An (ns x nt) grid over P(s, t), s and t in [0, 1]. Normals come from the surface's
        // own derivatives, turned to face away from `inside`; UVs run in metres along the grid.
        void patch(int ns, int nt, const std::function<Vector3(float, float)>& P, const Vector3& inside,
                   const std::function<Vector2(const Vector3&)>& uvOf = nullptr) {
            const int rs = ns + 1, rt = nt + 1;
            std::vector<Vector3> g(static_cast<size_t>(rs) * rt);
            for (int j = 0; j < rt; ++j)
                for (int i = 0; i < rs; ++i) g[j * rs + i] = P(static_cast<float>(i) / ns, static_cast<float>(j) / nt);

            const float e = 1e-3f;
            auto normalAt = [&](float s, float t) {
                return cross(P(s + e, t) - P(s - e, t), P(s, t + e) - P(s, t - e));
            };
            const Vector3 nc = normalAt(0.5f, 0.5f);
            const bool flip = nc.dot(P(0.5f, 0.5f) - inside) < 0;

            std::vector<float> U(g.size()), V(g.size());
            for (int j = 0; j < rt; ++j) {
                float acc = 0;
                U[j * rs] = 0;
                for (int i = 1; i < rs; ++i) {
                    acc += g[j * rs + i].distanceTo(g[j * rs + i - 1]);
                    U[j * rs + i] = acc;
                }
            }
            for (int i = 0; i < rs; ++i) {
                float acc = 0;
                V[i] = 0;
                for (int j = 1; j < rt; ++j) {
                    acc += g[j * rs + i].distanceTo(g[(j - 1) * rs + i]);
                    V[j * rs + i] = acc;
                }
            }

            const unsigned int base = count();
            for (int j = 0; j < rt; ++j)
                for (int i = 0; i < rs; ++i) {
                    Vector3 n = normalAt(static_cast<float>(i) / ns, static_cast<float>(j) / nt);
                    if (n.lengthSq() < 1e-22f) n = nc;
                    if (flip) n.negate();
                    if (uvOf) {
                        const Vector2 t = uvOf(g[j * rs + i]);
                        vertex(g[j * rs + i], n.normalize(), t.x, t.y);
                    } else {
                        vertex(g[j * rs + i], n.normalize(), U[j * rs + i], V[j * rs + i]);
                    }
                }
            for (int j = 0; j < nt; ++j)
                for (int i = 0; i < ns; ++i) {
                    const unsigned int a = base + j * rs + i, b = a + 1, c = a + rs, d = c + 1;
                    if (!flip) {
                        tri(a, b, d);
                        tri(a, d, c);
                    } else {
                        tri(a, d, b);
                        tri(a, c, d);
                    }
                }
        }

        // Surface of revolution about local +Y. prof holds (radius, y) points ordered so the
        // solid lies to the LEFT of the direction of travel; each segment is its own patch, so
        // profile corners stay sharp.
        void lathe(const std::vector<Vector2>& prof, int segs, float res = 0.3f, float a0 = 0.f, float a1 = kTau) {
            for (size_t k = 0; k + 1 < prof.size(); ++k) {
                const Vector2 p0 = prof[k], p1 = prof[k + 1];
                const float len = p0.distanceTo(p1);
                if (len < 1e-5f) continue;
                const float dr = (p1.x - p0.x) / len, dy = (p1.y - p0.y) / len;
                const float nr = dy, ny = -dr;// outward: right of travel
                const float am = 0.5f * (a0 + a1);
                const float rm = 0.5f * (p0.x + p1.x) - nr * 0.01f, ym = 0.5f * (p0.y + p1.y) - ny * 0.01f;
                const Vector3 inside(rm * std::cos(am), ym, -rm * std::sin(am));
                patch(segs, segsFor(len, res), [=](float s, float t) {
                    const float a = a0 + (a1 - a0) * s;
                    const float r = mixf(p0.x, p1.x, t), y = mixf(p0.y, p1.y, t);
                    return Vector3(r * std::cos(a), y, -r * std::sin(a)); }, inside,
                      // a flat ring (disc face, cap) takes a planar projection: arc-length UVs swirl there
                      std::abs(dy) < 0.35f ? std::function<Vector2(const Vector3&)>([](const Vector3& q) { return Vector2(q.x, -q.z); })
                                           : std::function<Vector2(const Vector3&)>());
            }
        }

        void box(const Vector3& c, const Vector3& h, float res = 0.3f) {
            auto face = [&](const Vector3& o, const Vector3& du, const Vector3& dv) {
                patch(segsFor(du.length(), res), segsFor(dv.length(), res),
                      [=](float s, float t) { return o + du * s + dv * t; }, c);
            };
            const Vector3 ex(2 * h.x, 0, 0), ey(0, 2 * h.y, 0), ez(0, 0, 2 * h.z);
            face(c + Vector3(-h.x, -h.y, h.z), ex, ey);
            face(c + Vector3(-h.x, -h.y, -h.z), ex, ey);
            face(c + Vector3(h.x, -h.y, -h.z), ez, ey);
            face(c + Vector3(-h.x, -h.y, -h.z), ez, ey);
            face(c + Vector3(-h.x, h.y, -h.z), ex, ez);
            face(c + Vector3(-h.x, -h.y, -h.z), ex, ez);
        }

        // A box running from a to b, `w` wide and `h` tall; its height follows `up`.
        void beam(const Vector3& a, const Vector3& b, float w, float h, Vector3 up = Vector3(0, 1, 0)) {
            Vector3 x = b - a;
            const float L = x.length();
            if (L < 1e-5f) return;
            x.multiplyScalar(1.f / L);
            if (std::abs(x.dot(unit(up))) > 0.98f) up = std::abs(x.y) < 0.9f ? kY : kX;
            const Vector3 z = unit(cross(x, up));
            const Vector3 y = cross(z, x);
            const Matrix4 saved = xf;
            setTransform(mul(saved, basisAt((a + b) * 0.5f, x, y, z)));
            box(Vector3(), Vector3(L * 0.5f, h * 0.5f, w * 0.5f));
            setTransform(saved);
        }

        // A round tube through the given points (parallel-transported frames).
        void tube(const std::vector<Vector3>& path, float r, int segs = 10) {
            const size_t n = path.size();
            if (n < 2) return;
            std::vector<Vector3> T(n), N(n), B(n);
            for (size_t i = 0; i < n; ++i) {
                const Vector3 a = path[i > 0 ? i - 1 : 0], b = path[std::min(i + 1, n - 1)];
                T[i] = unit(b - a);
            }
            N[0] = unit(cross(T[0], std::abs(T[0].y) < 0.9f ? kY : kX));
            for (size_t i = 1; i < n; ++i) {
                Vector3 b = cross(T[i - 1], T[i]);
                N[i] = N[i - 1];
                if (b.length() > 1e-6f) {
                    const float ang = std::acos(std::clamp(T[i - 1].dot(T[i]), -1.f, 1.f));
                    Quaternion q;
                    q.setFromAxisAngle(unit(b), ang);
                    N[i].applyQuaternion(q);
                }
            }
            for (size_t i = 0; i < n; ++i) B[i] = cross(T[i], N[i]);
            const float span = static_cast<float>(n - 1);
            auto P = [&](float s, float t) {
                const float f = t * span;
                const size_t i = std::min(static_cast<size_t>(f), n - 2);
                const float u = f - static_cast<float>(i);
                const float a = s * kTau;
                const Vector3 c0 = path[i] + (N[i] * std::cos(a) + B[i] * std::sin(a)) * r;
                const Vector3 c1 = path[i + 1] + (N[i + 1] * std::cos(a) + B[i + 1] * std::sin(a)) * r;
                return lerpV(c0, c1, u);
            };
            // One patch per path segment keeps "inside" meaningful on a curving tube.
            for (size_t i = 0; i + 1 < n; ++i) {
                const float t0 = static_cast<float>(i) / span, t1 = static_cast<float>(i + 1) / span;
                const Vector3 mid = (path[i] + path[i + 1]) * 0.5f;
                patch(segs, 1, [&, t0, t1](float s, float t) { return P(s, mixf(t0, t1, t)); }, mid);
            }
        }

        // A flat convex polygon, filled in rings toward its centroid (so baked AO has vertices).
        void capPoly(const std::vector<Vector3>& poly, const Vector3& inside, int rings = 3) {
            const size_t m = poly.size();
            Vector3 c;
            for (const auto& p : poly) c += p;
            c.multiplyScalar(1.f / static_cast<float>(m));
            Vector3 n;
            for (size_t i = 0; i < m; ++i) {
                const Vector3& a = poly[i];
                const Vector3& b = poly[(i + 1) % m];
                n.x += (a.y - b.y) * (a.z + b.z);
                n.y += (a.z - b.z) * (a.x + b.x);
                n.z += (a.x - b.x) * (a.y + b.y);
            }
            n.normalize();
            if (n.dot(c - inside) < 0) n.negate();
            const Vector3 tu = unit(poly[1] - poly[0]);
            const Vector3 tv = cross(n, tu);
            std::vector<Vector3> loc;
            const unsigned int base = count();
            for (int r = 0; r < rings; ++r)
                for (const auto& p : poly) {
                    const Vector3 q = c + (p - c) * (1.f - static_cast<float>(r) / rings);
                    loc.push_back(q);
                    vertex(q, n, (q - c).dot(tu), (q - c).dot(tv));
                }
            loc.push_back(c);
            vertex(c, n, 0, 0);
            const unsigned int center = base + static_cast<unsigned int>(loc.size() - 1);
            auto oriented = [&](unsigned int a, unsigned int b, unsigned int d) {
                const Vector3 pa = loc[a - base], pb = loc[b - base], pd = loc[d - base];
                if (cross(pb - pa, pd - pa).dot(n) < 0) std::swap(b, d);
                tri(a, b, d);
            };
            for (int r = 0; r + 1 < rings; ++r)
                for (size_t i = 0; i < m; ++i) {
                    const unsigned int a = base + r * m + i, b = base + r * m + (i + 1) % m;
                    const unsigned int c2 = base + (r + 1) * m + i, d = base + (r + 1) * m + (i + 1) % m;
                    oriented(a, b, d);
                    oriented(a, d, c2);
                }
            for (size_t i = 0; i < m; ++i) {
                oriented(base + (rings - 1) * m + i, base + (rings - 1) * m + (i + 1) % m, center);
            }
        }

        // A flat rectangle whose UVs span 0..1 (hatches, decals). Its face points along u x v.
        void panel(const Vector3& c, const Vector3& u, const Vector3& v, float W, float H) {
            const Vector3 n = unit(cross(u, v));
            const Vector3 o = c - u * (W / 2) - v * (H / 2);
            const int nu = segsFor(W, 0.2f), nv = segsFor(H, 0.2f);
            const unsigned int base = count();
            for (int j = 0; j <= nv; ++j)
                for (int i = 0; i <= nu; ++i) {
                    const float s = static_cast<float>(i) / nu, t = static_cast<float>(j) / nv;
                    vertex(o + u * (W * s) + v * (H * t), n, s * tile, t * tile);
                }
            for (int j = 0; j < nv; ++j)
                for (int i = 0; i < nu; ++i) {
                    const unsigned int a = base + j * (nu + 1) + i, b = a + 1, c2 = a + nu + 1, d = c2 + 1;
                    tri(a, b, d);
                    tri(a, d, c2);
                }
        }

        [[nodiscard]] std::shared_ptr<BufferGeometry> build() const {
            auto g = BufferGeometry::create();
            g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            g->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            g->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            if (!col.empty()) g->setAttribute("color", FloatBufferAttribute::create(col, 3));
            g->setIndex(idx);
            g->computeBoundingSphere();
            return g;
        }
    };

    // ================================================================== baked ambient occlusion

    // Occupancy of a part's own surfaces on a voxel grid; rays marched through it give each
    // vertex its share of open sky. Done once at start-up, stored as vertex colour.
    struct Voxels {
        Vector3 lo;
        float cell = 0.1f;
        int nx = 0, ny = 0, nz = 0;
        std::vector<std::uint8_t> v;

        [[nodiscard]] bool solid(const Vector3& p) const {
            const int i = static_cast<int>(std::floor((p.x - lo.x) / cell));
            const int j = static_cast<int>(std::floor((p.y - lo.y) / cell));
            const int k = static_cast<int>(std::floor((p.z - lo.z) / cell));
            if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
            return v[(static_cast<size_t>(k) * ny + j) * nx + i] != 0;
        }
    };

    Voxels voxelize(const std::vector<const Geo*>& geos, float cell) {
        Vector3 lo(1e9f, 1e9f, 1e9f), hi(-1e9f, -1e9f, -1e9f);
        for (const auto* g : geos)
            for (size_t i = 0; i < g->pos.size(); i += 3) {
                lo.x = std::min(lo.x, g->pos[i]);
                lo.y = std::min(lo.y, g->pos[i + 1]);
                lo.z = std::min(lo.z, g->pos[i + 2]);
                hi.x = std::max(hi.x, g->pos[i]);
                hi.y = std::max(hi.y, g->pos[i + 1]);
                hi.z = std::max(hi.z, g->pos[i + 2]);
            }
        Voxels vox;
        vox.cell = cell;
        vox.lo = lo - Vector3(3 * cell, 3 * cell, 3 * cell);
        vox.nx = static_cast<int>((hi.x - lo.x) / cell) + 7;
        vox.ny = static_cast<int>((hi.y - lo.y) / cell) + 7;
        vox.nz = static_cast<int>((hi.z - lo.z) / cell) + 7;
        vox.v.assign(static_cast<size_t>(vox.nx) * vox.ny * vox.nz, 0);
        auto mark = [&](const Vector3& p) {
            const int i = static_cast<int>(std::floor((p.x - vox.lo.x) / cell));
            const int j = static_cast<int>(std::floor((p.y - vox.lo.y) / cell));
            const int k = static_cast<int>(std::floor((p.z - vox.lo.z) / cell));
            if (i < 0 || j < 0 || k < 0 || i >= vox.nx || j >= vox.ny || k >= vox.nz) return;
            vox.v[(static_cast<size_t>(k) * vox.ny + j) * vox.nx + i] = 1;
        };
        for (const auto* g : geos)
            for (size_t t = 0; t + 2 < g->idx.size(); t += 3) {
                auto P = [&](unsigned int k) { return Vector3(g->pos[3 * k], g->pos[3 * k + 1], g->pos[3 * k + 2]); };
                const Vector3 a = P(g->idx[t]), b = P(g->idx[t + 1]), c = P(g->idx[t + 2]);
                const float L = std::max({a.distanceTo(b), b.distanceTo(c), c.distanceTo(a)});
                const int n = std::max(1, static_cast<int>(std::ceil(L / (cell * 0.5f))));
                for (int i = 0; i <= n; ++i)
                    for (int j = 0; j <= n - i; ++j)
                        mark(a + (b - a) * (static_cast<float>(i) / n) + (c - a) * (static_cast<float>(j) / n));
            }
        return vox;
    }

    void bakeAO(const std::vector<Geo*>& geos, float cell, float maxDist, float strength) {
        std::vector<const Geo*> all(geos.begin(), geos.end());
        const Voxels vox = voxelize(all, cell);
        // cosine-weighted directions about +Z
        std::vector<Vector3> dirs;
        const float ga = kPi * (3.f - std::sqrt(5.f));
        const int nd = 30;
        for (int i = 0; i < nd; ++i) {
            const float u = (i + 0.5f) / nd, r = std::sqrt(u), a = i * ga;
            dirs.emplace_back(r * std::cos(a), r * std::sin(a), std::sqrt(1.f - u));
        }
        for (Geo* g : geos) {
            g->col.assign(g->pos.size(), 1.f);
            for (unsigned int v = 0; v < g->count(); ++v) {
                const Vector3 p(g->pos[3 * v], g->pos[3 * v + 1], g->pos[3 * v + 2]);
                const Vector3 n(g->nrm[3 * v], g->nrm[3 * v + 1], g->nrm[3 * v + 2]);
                const Vector3 t = unit(cross(std::abs(n.y) < 0.9f ? kY : kX, n));
                const Vector3 b = cross(n, t);
                const Vector3 o = p + n * (1.3f * cell);
                float occ = 0;
                for (const auto& d : dirs) {
                    const Vector3 w = t * d.x + b * d.y + n * d.z;
                    for (float s = 0.5f * cell; s < maxDist; s += 0.6f * cell) {
                        if (vox.solid(o + w * s)) {
                            const float f = s / maxDist;
                            occ += 1.f - f * f;
                            break;
                        }
                    }
                }
                const float ao = std::clamp(1.f - strength * occ / nd, 0.16f, 1.f);
                g->col[3 * v] = g->col[3 * v + 1] = g->col[3 * v + 2] = ao;
            }
        }
    }

    // ================================================================== rigid parts

    struct Part {
        std::vector<std::pair<std::shared_ptr<BufferGeometry>, std::shared_ptr<Material>>> layers;
    };

    struct PartBuilder {
        std::vector<std::pair<std::shared_ptr<Material>, std::unique_ptr<Geo>>> layers;
        std::vector<Matrix4> stack{Matrix4()};

        Geo& on(const std::shared_ptr<Material>& m) {
            Geo* g = nullptr;
            for (auto& [mm, gg] : layers)
                if (mm == m) g = gg.get();
            if (!g) {
                layers.emplace_back(m, std::make_unique<Geo>());
                g = layers.back().second.get();
            }
            g->setTransform(stack.back());
            return *g;
        }
        void push(const Matrix4& m) { stack.push_back(mul(stack.back(), m)); }
        void pop() { stack.pop_back(); }

        Part build(float cell, float maxDist, float strength) {
            std::vector<Geo*> geos;
            for (auto& [m, g] : layers) geos.push_back(g.get());
            bakeAO(geos, cell, maxDist, strength);
            Part p;
            for (auto& [m, g] : layers) p.layers.emplace_back(g->build(), m);
            return p;
        }
    };

    std::shared_ptr<Group> instantiate(const Part& part, bool castShadow) {
        auto g = Group::create();
        for (const auto& [geo, mat] : part.layers) {
            auto mesh = Mesh::create(geo, mat);
            mesh->castShadow = castShadow && !mat->transparent;
            mesh->receiveShadow = true;
            g->add(mesh);
        }
        return g;
    }

    // ================================================================== procedural textures

    struct Rgb {
        float r, g, b;
    };
    Rgb mixc(const Rgb& a, const Rgb& b, float t) { return {mixf(a.r, b.r, t), mixf(a.g, b.g, t), mixf(a.b, b.b, t)}; }
    Rgb scalec(const Rgb& a, float s) { return {a.r * s, a.g * s, a.b * s}; }
    unsigned char toByte(float v) { return static_cast<unsigned char>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f); }

    std::shared_ptr<DataTexture> rgbaTexture(std::vector<unsigned char> px, int w, int h, bool srgbData, bool repeat = true) {
        auto tex = DataTexture::create(ImageData(std::move(px)), w, h);
        tex->format = Format::RGBA;
        tex->type = Type::UnsignedByte;
        tex->wrapS = tex->wrapT = repeat ? TextureWrapping::Repeat : TextureWrapping::ClampToEdge;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::LinearMipmapLinear;
        tex->generateMipmaps = true;
        tex->anisotropy = 8;
        if (srgbData) tex->colorSpace = ColorSpace::sRGB;
        tex->needsUpdate();
        return tex;
    }

    std::vector<unsigned char> normalsFromHeight(const std::vector<float>& h, int w, int ht, float strength) {
        std::vector<unsigned char> out(static_cast<size_t>(w) * ht * 4);
        for (int y = 0; y < ht; ++y)
            for (int x = 0; x < w; ++x) {
                auto H = [&](int xx, int yy) { return h[((yy + ht) % ht) * w + ((xx + w) % w)]; };
                const float dx = 0.5f * (H(x + 1, y) - H(x - 1, y)) * strength;
                const float dy = 0.5f * (H(x, y + 1) - H(x, y - 1)) * strength;
                Vector3 n(-dx, -dy, 1.f);
                n.normalize();
                const size_t i = (static_cast<size_t>(y) * w + x) * 4;
                out[i] = toByte(n.x * 0.5f + 0.5f);
                out[i + 1] = toByte(n.y * 0.5f + 0.5f);
                out[i + 2] = toByte(n.z * 0.5f + 0.5f);
                out[i + 3] = 255;
            }
        return out;
    }

    // Per-texel wear layers shared by the paint variants of one texture set.
    struct Wear {
        int w = 0, h = 0;
        std::vector<float> height, primer, bare, seam, mottle, scratch, rivet, polish;
    };

    struct SurfaceSet {
        std::vector<std::shared_ptr<Texture>> albedo;// one per paint
        std::shared_ptr<Texture> orm, normal;
    };

    const Rgb kPrimer{0.50f, 0.27f, 0.20f};// red-oxide primer (sRGB)
    const Rgb kSteel{0.60f, 0.60f, 0.62f};// bare steel (sRGB)

    SurfaceSet finishSet(const Wear& wr, const std::vector<Rgb>& paints, float normalStrength, bool repeat) {
        SurfaceSet set;
        const size_t n = static_cast<size_t>(wr.w) * wr.h;
        for (const auto& paint : paints) {
            std::vector<unsigned char> px(n * 4);
            for (size_t i = 0; i < n; ++i) {
                Rgb c = scalec(paint, 0.88f + 0.24f * (wr.mottle[i] - 0.5f));
                c = scalec(c, 1.f - 0.5f * wr.seam[i]);
                c = scalec(c, 1.f + 0.12f * wr.rivet[i]);
                c = mixc(c, scalec(kPrimer, 0.92f + 0.16f * wr.mottle[i]), wr.primer[i]);
                c = mixc(c, kSteel, wr.bare[i]);
                c = mixc(c, mixc(kPrimer, kSteel, 0.55f), 0.7f * wr.scratch[i]);
                c = mixc(c, Rgb{0.30f, 0.28f, 0.26f}, 0.25f * wr.seam[i]);// dirt held in the seams
                px[i * 4] = toByte(c.r);
                px[i * 4 + 1] = toByte(c.g);
                px[i * 4 + 2] = toByte(c.b);
                px[i * 4 + 3] = 255;
            }
            set.albedo.push_back(rgbaTexture(std::move(px), wr.w, wr.h, true, repeat));
        }
        std::vector<unsigned char> orm(n * 4);
        for (size_t i = 0; i < n; ++i) {
            float rough = 0.40f + 0.22f * (wr.mottle[i] - 0.5f) + 0.25f * wr.seam[i];
            rough = mixf(rough, 0.80f, wr.primer[i]);
            rough = mixf(rough, 0.30f, wr.bare[i]);
            rough = mixf(rough, 0.26f, 0.8f * wr.scratch[i]);
            rough = mixf(rough, 0.22f, wr.polish[i]);
            float metal = std::max(0.95f * wr.bare[i], 0.75f * wr.scratch[i] * (1.f - wr.primer[i]));
            metal = std::max(metal, 0.6f * wr.polish[i]);
            orm[i * 4] = 255;
            orm[i * 4 + 1] = toByte(rough);
            orm[i * 4 + 2] = toByte(metal);
            orm[i * 4 + 3] = 255;
        }
        set.orm = rgbaTexture(std::move(orm), wr.w, wr.h, false, repeat);
        set.normal = rgbaTexture(normalsFromHeight(wr.height, wr.w, wr.h, normalStrength), wr.w, wr.h, false, repeat);
        return set;
    }

    void addScratches(Wear& wr, int count, int seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        for (int s = 0; s < count; ++s) {
            float x = u01(rng) * wr.w, y = u01(rng) * wr.h;
            float a = u01(rng) * kTau;
            const float len = 6 + u01(rng) * 46, bend = (u01(rng) - 0.5f) * 0.012f, depth = 0.35f + 0.65f * u01(rng);
            for (float l = 0; l < len; l += 0.5f) {
                const int ix = ((static_cast<int>(x) % wr.w) + wr.w) % wr.w;
                const int iy = ((static_cast<int>(y) % wr.h) + wr.h) % wr.h;
                float& v = wr.scratch[iy * wr.w + ix];
                v = std::max(v, depth * (1.f - 0.6f * l / len));
                x += std::cos(a) * 0.5f;
                y += std::sin(a) * 0.5f;
                a += bend;
            }
        }
    }

    // A 4 m tile of plated enamel: plates on a staggered 1 x 2 m grid, rivet rows, chips that
    // gather at the plate seams (primer first, bare steel at the heart of the deep ones) and
    // fine scratches.
    SurfaceSet makePlateTile(const std::vector<Rgb>& paints, int seed) {
        constexpr int N = 512;
        constexpr float mpt = 4.f / N;// metres per texel
        Wear wr;
        wr.w = wr.h = N;
        for (auto* v : {&wr.height, &wr.primer, &wr.bare, &wr.seam, &wr.mottle, &wr.scratch, &wr.rivet, &wr.polish}) v->assign(N * N, 0.f);
        addScratches(wr, 220, seed);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int i = y * N + x;
                const float u = (x + 0.5f) / N, v = (y + 0.5f) / N;
                const float mx = u * 4.f, my = v * 4.f;
                const float column = std::floor(mx);
                const float fx = mx - column;
                const float ry = my + ((static_cast<int>(column) & 1) ? 1.f : 0.f);
                const float fy = ry - 2.f * std::floor(ry / 2.f);
                const float dx = std::min(fx, 1.f - fx), dy = std::min(fy, 2.f - fy);
                const float dSeam = std::min(dx, dy) / mpt;
                const float seam = 1.f - smoothstepf(0.4f, 1.5f, dSeam);
                const float cyy = std::round(my / 0.125f) * 0.125f, cxx = std::round(mx / 0.125f) * 0.125f;
                const float dr = std::min(std::hypot(dx - 0.05f, my - cyy), std::hypot(dy - 0.05f, mx - cxx));
                const float rivet = 1.f - smoothstepf(0.008f, 0.014f, dr);
                const float mottle = fbm(u * 6, v * 6, 5, 6, seed + 1);
                // chips live along the plate edges; the open plate is mostly sound paint
                const float edge = 1.f - smoothstepf(1.f, 9.f, dSeam);
                float chip = fbm(u * 36, v * 36, 4, 36, seed + 2) + 0.26f * edge +
                             0.14f * (fbm(u * 5, v * 5, 3, 5, seed + 3) - 0.5f) + 0.12f * rivet;
                const float primer = smoothstepf(0.80f, 0.81f, chip);
                const float bare = smoothstepf(0.86f, 0.87f, chip);
                wr.seam[i] = seam;
                wr.rivet[i] = rivet;
                wr.mottle[i] = mottle;
                wr.primer[i] = primer;
                wr.bare[i] = bare;
                wr.height[i] = 0.25f * (fbm(u * 96, v * 96, 2, 96, seed + 4) - 0.5f) - 1.0f * seam + 1.1f * rivet -
                               0.5f * primer - 0.3f * bare - 0.3f * wr.scratch[i];
            }
        return finishSet(wr, paints, 2.2f, true);
    }

    // One hatch (UV 0..1 over its face): a groove inset from the edge, bolts, a recessed pull
    // handle, and heavy chipping along the outer edges and round the handle where boots,
    // tools and hands wear it.
    SurfaceSet makeHatchTile(const std::vector<Rgb>& paints, int seed) {
        constexpr int N = 256;
        Wear wr;
        wr.w = wr.h = N;
        for (auto* v : {&wr.height, &wr.primer, &wr.bare, &wr.seam, &wr.mottle, &wr.scratch, &wr.rivet, &wr.polish}) v->assign(N * N, 0.f);
        addScratches(wr, 70, seed);
        auto sdRect = [](float px, float py, float hx, float hy, float r) {
            const float qx = std::abs(px) - hx + r, qy = std::abs(py) - hy + r;
            return std::hypot(std::max(qx, 0.f), std::max(qy, 0.f)) + std::min(std::max(qx, qy), 0.f) - r;
        };
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const int i = y * N + x;
                const float u = (x + 0.5f) / N, v = (y + 0.5f) / N;
                const float dEdge = std::min({u, 1.f - u, v, 1.f - v}) * N;
                const float groove = 1.f - smoothstepf(0.8f, 2.2f, std::abs(dEdge - 15.f));
                const float dRec = sdRect(u - 0.84f, v - 0.5f, 0.05f, 0.12f, 0.03f);
                const float recess = 1.f - smoothstepf(-0.004f, 0.004f, dRec);
                const float bar = 1.f - smoothstepf(-0.003f, 0.003f, sdRect(u - 0.84f, v - 0.5f, 0.013f, 0.09f, 0.01f));
                // bolts on a ring 7 texels in from the edge, every 28 texels
                float bolt = 0;
                if (std::abs(dEdge - 7.f) < 4.f) {
                    const float along = (u < 0.5f ? u : 1.f - u) < (v < 0.5f ? v : 1.f - v) ? v * N : u * N;
                    const float k = std::abs(fractf(along / 28.f) - 0.5f) * 28.f;
                    bolt = 1.f - smoothstepf(2.0f, 3.2f, std::hypot(dEdge - 7.f, 14.f - k));
                }
                const float mottle = fbm(u * 5, v * 5, 4, 0, seed + 1);
                float chip = fbm(u * 18, v * 18, 4, 0, seed + 2) + 0.42f * (1.f - smoothstepf(1.f, 13.f, dEdge)) +
                             0.36f * (1.f - smoothstepf(0.f, 0.05f, dRec)) + 0.12f * (fbm(u * 50, v * 50, 2, 0, seed + 3) - 0.5f) +
                             0.15f * bolt;
                const float primer = smoothstepf(0.78f, 0.79f, chip);
                const float bare = smoothstepf(0.84f, 0.85f, chip);
                const float polish = (1.f - smoothstepf(0.f, 0.05f, dRec)) * smoothstepf(0.45f, 0.7f, fbm(u * 30, v * 30, 3, 0, seed + 5)) * (1.f - recess) +
                                     bar * 0.8f;
                wr.seam[i] = std::max(groove, recess * 0.7f);
                wr.rivet[i] = bolt;
                wr.mottle[i] = mottle;
                wr.primer[i] = primer;
                wr.bare[i] = std::max(bare, bar * 0.3f);
                wr.polish[i] = std::clamp(polish, 0.f, 1.f);
                wr.height[i] = 0.2f * (fbm(u * 60, v * 60, 2, 0, seed + 4) - 0.5f) - 1.3f * groove - 1.0f * recess + 1.1f * bar +
                               1.0f * bolt - 0.55f * primer - 0.3f * bare - 0.3f * wr.scratch[i];
            }
        return finishSet(wr, paints, 2.6f, false);
    }

    // Grime washed down from a vent: dark streaks of uneven length, densest just below the
    // louvres (v = 0 at the top of the decal). Blending happens in linear light, where a
    // middling alpha barely darkens bright paint, so the mask is pushed hard.
    std::shared_ptr<Texture> makeGrimeDecal(int seed) {
        constexpr int W = 128, H = 256;
        std::vector<unsigned char> px(W * H * 4);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float u = (x + 0.5f) / W, v = (y + 0.5f) / H;
                const float strength = smoothstepf(0.38f, 0.62f, fbm(u * 11.f, 0.5f, 3, 0, seed));
                const float length = 0.35f + 0.65f * fbm(u * 4.f, 4.5f, 2, 0, seed + 1);
                float a = strength * std::pow(1.f - smoothstepf(0.f, length, v), 1.2f);
                a += 0.7f * std::pow(1.f - v, 5.f);// soot right under the louvres
                a = std::min(a, 1.f) * (0.6f + 0.4f * fbm(u * 40.f, v * 6.f, 2, 0, seed + 2));
                a *= smoothstepf(0.f, 0.1f, std::min(u, 1.f - u));
                a = 1.f - std::pow(1.f - clamp01(a), 2.2f);
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                px[i] = toByte(0.11f);
                px[i + 1] = toByte(0.095f);
                px[i + 2] = toByte(0.08f);
                px[i + 3] = toByte(std::min(a, 0.94f));
            }
        return rgbaTexture(std::move(px), W, H, true, false);
    }

    // ================================================================== materials

    struct Mats {
        std::shared_ptr<MeshStandardMaterial> red, white, graphite, steel, chrome, rubber, glass, lamp, hatchRed,
                hatchWhite, grime, recess, orange, parka, pants, fur, lens, mitt, boot, pack, ice;
    };

    Mats makeMaterials() {
        const Rgb red{0.60f, 0.11f, 0.075f}, white{0.80f, 0.79f, 0.75f}, graphite{0.20f, 0.21f, 0.22f}, steel{0.30f, 0.30f, 0.31f};
        const auto plates = makePlateTile({red, white, graphite, steel}, 11);
        const auto hatches = makeHatchTile({red, white}, 23);

        auto painted = [](const std::shared_ptr<Texture>& albedo, const SurfaceSet& set) {
            auto m = MeshStandardMaterial::create();
            m->map = albedo;
            m->roughnessMap = set.orm;
            m->metalnessMap = set.orm;
            m->normalMap = set.normal;
            m->normalScale = Vector2(1, 1);
            m->roughness = 1.f;
            m->metalness = 1.f;
            m->vertexColors = true;
            return m;
        };
        auto plain = [](const Color& c, float rough, float metal) {
            auto m = MeshStandardMaterial::create();
            m->color = c;
            m->roughness = rough;
            m->metalness = metal;
            m->vertexColors = true;
            return m;
        };

        Mats m;
        m.red = painted(plates.albedo[0], plates);
        m.white = painted(plates.albedo[1], plates);
        m.graphite = painted(plates.albedo[2], plates);
        m.steel = painted(plates.albedo[3], plates);
        m.steel->metalnessMap = nullptr;
        m.steel->metalness = 0.65f;
        m.hatchRed = painted(hatches.albedo[0], hatches);
        m.hatchWhite = painted(hatches.albedo[1], hatches);
        m.chrome = plain(Color(0.93f, 0.94f, 0.96f), 0.09f, 1.f);
        m.rubber = plain(srgb(0.07f, 0.07f, 0.075f), 0.85f, 0.f);
        m.glass = plain(srgb(0.10f, 0.13f, 0.15f), 0.04f, 0.35f);
        m.glass->emissive = Color(0.9f, 0.55f, 0.25f);
        m.glass->emissiveIntensity = 0.025f;
        m.lamp = plain(Color(1, 1, 1), 0.3f, 0.f);
        m.lamp->emissive = Color(1.f, 0.92f, 0.80f);
        m.lamp->emissiveIntensity = 6.f;
        m.lamp->vertexColors = false;
        m.recess = plain(srgb(0.04f, 0.04f, 0.045f), 0.9f, 0.f);
        m.grime = MeshStandardMaterial::create();
        m.grime->map = makeGrimeDecal(5);
        m.grime->transparent = true;
        m.grime->depthWrite = false;
        m.grime->polygonOffset = true;
        m.grime->polygonOffsetFactor = -2;
        m.grime->polygonOffsetUnits = -2;
        m.grime->roughness = 0.7f;
        m.grime->metalness = 0.f;
        m.orange = plain(srgb(0.92f, 0.36f, 0.05f), 0.45f, 0.f);
        m.parka = plain(srgb(0.93f, 0.40f, 0.06f), 0.75f, 0.f);
        m.pants = plain(srgb(0.16f, 0.17f, 0.19f), 0.85f, 0.f);
        m.fur = plain(srgb(0.58f, 0.53f, 0.47f), 1.f, 0.f);
        m.lens = plain(srgb(0.05f, 0.05f, 0.06f), 0.08f, 0.6f);
        m.mitt = plain(srgb(0.09f, 0.09f, 0.10f), 0.9f, 0.f);
        m.boot = plain(srgb(0.12f, 0.10f, 0.09f), 0.8f, 0.f);
        m.pack = plain(srgb(0.22f, 0.26f, 0.30f), 0.8f, 0.f);
        m.ice = plain(srgb(0.86f, 0.90f, 0.96f), 0.6f, 0.f);
        return m;
    }

    // ================================================================== walker geometry

    // Octagonal cross-section in (z, y), counter-clockwise seen from +x. Facets run from the
    // upper-right chamfer (0) round through the right side (1), lower chamfers, belly (3),
    // the left side (5) to the top (7).
    struct Octa {
        float x, w, wt, wb, top, bot, ct, cb;
    };

    std::vector<Vector2> octagon(const Octa& s) {
        return {{s.wt, s.top}, {s.w, s.top - s.ct}, {s.w, s.bot + s.cb}, {s.wb, s.bot},
                {-s.wb, s.bot}, {-s.w, s.bot + s.cb}, {-s.w, s.top - s.ct}, {-s.wt, s.top}};
    }

    void loftOcta(PartBuilder& pb, const std::vector<Octa>& st,
                  const std::function<std::shared_ptr<Material>(int, int)>& pick,
                  const std::shared_ptr<Material>& capRear, const std::shared_ptr<Material>& capFront, float res = 0.32f) {
        for (size_t i = 0; i + 1 < st.size(); ++i) {
            const auto A = octagon(st[i]), B = octagon(st[i + 1]);
            const float x0 = st[i].x, x1 = st[i + 1].x;
            const Vector3 inside(0.5f * (x0 + x1), 0.25f * (st[i].top + st[i].bot + st[i + 1].top + st[i + 1].bot), 0.f);
            for (int f = 0; f < 8; ++f) {
                auto mat = pick(f, static_cast<int>(i));
                if (!mat) continue;
                const Vector2 a0 = A[f], a1 = A[(f + 1) % 8], b0 = B[f], b1 = B[(f + 1) % 8];
                const float len = std::max(a0.distanceTo(a1), b0.distanceTo(b1));
                if (len < 1e-4f) continue;
                pb.on(mat).patch(segsFor(x1 - x0, res), segsFor(len, res), [=](float s, float t) {
                    const Vector2 p = lerp2(lerp2(a0, a1, t), lerp2(b0, b1, t), s);
                    return Vector3(mixf(x0, x1, s), p.y, p.x); }, inside);
            }
        }
        for (int e = 0; e < 2; ++e) {
            const auto& mat = e == 0 ? capRear : capFront;
            if (!mat) continue;
            const Octa& s = e == 0 ? st.front() : st.back();
            std::vector<Vector3> poly;
            for (const auto& p : octagon(s)) poly.emplace_back(s.x, p.y, p.x);
            pb.on(mat).capPoly(poly, Vector3(s.x + (e == 0 ? 0.3f : -0.3f), 0.5f * (s.top + s.bot), 0.f));
        }
    }

    // A disc-shaped profile for joint housings, about +Y: radius R, half width hw, chamfer c.
    std::vector<Vector2> discProfile(float R, float hw, float c) {
        return {{0.f, -hw}, {R - c, -hw}, {R, -hw + c}, {R, hw - c}, {R - c, hw}, {0.f, hw}};
    }
    std::vector<Vector2> rodProfile(float r, float y0, float y1) {
        return {{0.f, y0}, {r, y0}, {r, y1}, {0.f, y1}};
    }
    std::vector<Vector2> sphereProfile(float r, int n = 10) {
        std::vector<Vector2> p;
        for (int i = 0; i <= n; ++i) {
            const float a = -kPi / 2 + kPi * i / n;
            p.emplace_back(std::max(1e-3f, r * std::cos(a)), r * std::sin(a));
        }
        return p;
    }

    // The three-toed sole: what the snow sees. Foot frame: x forward, z right, metres.
    struct Sole {
        struct Toe {
            float dx, dz, len, r0, r1;
        };
        float pad = 1.35f;
        std::array<Toe, 3> toes{};

        Sole() {
            const float c = std::cos(deg(40)), s = std::sin(deg(40));
            toes = {Toe{c, s, 2.3f, 0.48f, 0.2f}, Toe{c, -s, 2.3f, 0.48f, 0.2f}, Toe{-1.f, 0.f, 2.05f, 0.46f, 0.2f}};
        }

        static float unevenCapsule(float px, float py, float r1, float r2, float h) {
            px = std::abs(px);
            const float b = (r1 - r2) / h, a = std::sqrt(1.f - b * b);
            const float k = -px * b + py * a;
            if (k < 0) return std::hypot(px, py) - r1;
            if (k > a * h) return std::hypot(px, py - h) - r2;
            return px * a + py * b - r1;
        }

        [[nodiscard]] float toeRadius(const Toe& t, float along) const {
            return mixf(t.r0, t.r1, clamp01(along / t.len));
        }

        [[nodiscard]] float sdf(float x, float z) const {
            float d = std::hypot(x, z) - pad;
            for (const auto& t : toes) {
                const float along = x * t.dx + z * t.dz, across = -x * t.dz + z * t.dx;
                d = std::min(d, unevenCapsule(across, along, t.r0, t.r1, t.len));
            }
            return d;
        }

        // 1 where a cleat stands proud of the sole (and so presses 6 cm deeper).
        [[nodiscard]] float cleat(float x, float z) const {
            auto band = [](float v, float centre, float half) {
                return 1.f - smoothstepf(half - 0.035f, half + 0.035f, std::abs(v - centre));
            };
            const float r = std::hypot(x, z);
            float c = 0;
            if (r < pad) {
                c = std::max(band(r, 0.55f, 0.08f), band(r, 1.05f, 0.08f));
                if (r > 0.25f && r < 1.25f) {
                    float off = std::fmod(std::atan2(z, x) - kPi / 6 + 10 * kTau, kPi / 3);
                    if (off > kPi / 6) off -= kPi / 3;
                    c = std::max(c, band(std::abs(std::sin(off)) * r, 0.f, 0.07f));
                }
            }
            for (const auto& t : toes) {
                const float along = x * t.dx + z * t.dz, across = std::abs(-x * t.dz + z * t.dx);
                if (along < pad - 0.05f || across > toeRadius(t, along)) continue;
                for (float b : {1.5f, 1.85f, 2.2f}) c = std::max(c, band(along, b, 0.07f));
            }
            return c;
        }
    };

    struct WalkerKit {
        Part hull, sensor, coxa, femur, tibia, foot, barrelLift, rodLift, barrelKnee, rodKnee;
        // leg geometry (metres)
        std::array<Vector3, 4> hip{};   // drum axis at pivot height, body frame (legs: LF, RF, LH, RH)
        std::array<Vector2, 4> stance{};// nominal foot (x fwd, z right), body frame
        float coxa_ = 1.45f, femur_ = 6.0f, tibia_ = 10.2f, ankle = 1.05f;
        float bodyHeight = 8.6f, sink = 0.32f;
        // Ram mounts, laid out like an excavator's: the lift ram hangs from an outrigger below
        // and beyond the hip pin and pushes on a bracket under the femur; the knee ram runs
        // along the top of the femur to a horn on the tibia that reaches past the knee pin.
        // Chosen by a clearance search over the joint ranges the gait reaches (femur 13-58 deg
        // up, knee -137..-100 deg) with margin: the lift ram keeps 0.2 m off the hip boss,
        // the knee ram 0.18 m off the knee housing. Barrel and rod lengths keep the rod inside
        // the barrel at full stroke and short of the barrel's eye when closed.
        Vector3 coxaLug{2.0f, -1.0f, 0.f}, femurLiftLug{3.4f, -0.85f, 0.f}, femurKneeLug{2.0f, 1.4f, 0.f}, tibiaKneeLug{-1.5f, -0.4f, 0.f};
        float liftBarrel = 1.9f, liftRod = 2.45f, kneeBarrel = 2.3f, kneeRod = 3.0f;
        Vector3 doorLeft{-1.6f, 0.2f, -2.42f};
        Sole sole;
    };

    // Hatch with rim, its face on the u x v side of c.
    void hatch(PartBuilder& pb, const std::shared_ptr<Material>& face, const std::shared_ptr<Material>& rim,
               const Vector3& c, const Vector3& u, const Vector3& v, float W, float H, float thick) {
        const Vector3 n = unit(cross(u, v));
        pb.on(face).panel(c + n * thick, u, v, W, H);
        Geo& r = pb.on(rim);
        const Vector3 inside = c - n * 0.2f;
        const Vector3 hu = u * (W / 2), hv = v * (H / 2), dn = n * thick;
        const std::array<Vector3, 4> corner{c - hu - hv, c + hu - hv, c + hu + hv, c - hu + hv};
        for (int k = 0; k < 4; ++k) {
            const Vector3 a = corner[k], b = corner[(k + 1) % 4];
            r.patch(segsFor(a.distanceTo(b), 0.25f), 1, [=](float s, float t) { return lerpV(a, b, s) + dn * t; }, inside);
        }
    }

    // Louvred vent on a flank (normal n) with a grime decal running down from it.
    void vent(PartBuilder& pb, const Mats& m, const Vector3& c, const Vector3& n, float W, float H, float decalH) {
        const Vector3 v = kY;
        const Vector3 u = unit(cross(v, n));// u x v = n
        pb.on(m.recess).panel(c + n * 0.004f, u, v, W, H);
        Geo& g = pb.on(m.graphite);
        const float fw = 0.05f, depth = 0.09f;
        const Vector3 hu = u * (W / 2 + fw / 2), hv = v * (H / 2 + fw / 2), out = n * (depth / 2);
        g.beam(c - hu - hv + out, c + hu - hv + out, depth, fw, n);
        g.beam(c - hu + hv + out, c + hu + hv + out, depth, fw, n);
        g.beam(c - hu - hv + out, c - hu + hv + out, depth, fw, n);
        g.beam(c + hu - hv + out, c + hu + hv + out, depth, fw, n);
        const int slats = 5;
        for (int k = 0; k < slats; ++k) {
            const float t = (k + 0.5f) / slats - 0.5f;
            const Vector3 sc = c + v * (t * H) + n * 0.05f;
            const Vector3 tilt = unit(v * std::cos(deg(35)) + n * std::sin(deg(35)));
            g.beam(sc - u * (W / 2), sc + u * (W / 2), 0.018f, H / slats * 1.05f, tilt);
        }
        // decal panel: v runs DOWN so the texture's top row sits under the louvres
        Geo& d = pb.on(m.grime);
        const Vector3 dv = neg(kY);
        const Vector3 du = unit(cross(dv, n));// du x dv = n
        d.panel(c - kY * (H / 2 + decalH / 2 - 0.02f) + n * 0.012f, du, dv, W * 1.12f, decalH);
    }

    void buildHull(PartBuilder& pb, const Mats& m, const WalkerKit& kit) {
        const std::vector<Octa> st{
                {-6.2f, 1.9f, 1.4f, 1.0f, 1.25f, -1.05f, 0.5f, 0.6f},
                {-5.6f, 2.4f, 1.8f, 1.3f, 1.7f, -1.6f, 0.5f, 0.8f},
                {2.8f, 2.4f, 1.8f, 1.3f, 1.7f, -1.6f, 0.5f, 0.8f},
                {4.2f, 2.3f, 1.45f, 1.3f, 1.25f, -1.5f, 0.65f, 0.8f},
                {5.6f, 1.95f, 0.95f, 1.1f, 0.55f, -1.3f, 0.6f, 0.7f},
                {6.4f, 1.25f, 0.55f, 0.7f, -0.1f, -1.0f, 0.3f, 0.4f}};
        auto pick = [&](int f, int span) -> std::shared_ptr<Material> {
            if (span == 3 && (f == 0 || f == 6 || f == 7)) return m.glass;
            if (f == 7 || f == 0 || f == 6) return m.white;
            if (f == 3) return m.graphite;
            return m.red;
        };
        loftOcta(pb, st, pick, m.red, m.red);

        // window frames round the wrap-around cab glazing
        {
            Geo& g = pb.on(m.graphite);
            const auto A = octagon(st[3]), B = octagon(st[4]);
            const std::array<int, 4> chain{6, 7, 0, 1};
            auto P = [](float x, const Vector2& p) { return Vector3(x, p.y, p.x); };
            for (int k = 0; k + 1 < 4; ++k) {
                g.beam(P(st[3].x, A[chain[k]]), P(st[3].x, A[chain[k + 1]]), 0.1f, 0.09f);
                g.beam(P(st[4].x, B[chain[k]]), P(st[4].x, B[chain[k + 1]]), 0.1f, 0.09f);
            }
            for (int k : chain) g.beam(P(st[3].x, A[k]), P(st[4].x, B[k]), 0.08f, 0.09f);
            g.beam(P(st[3].x, Vector2(0.f, A[7].y)), P(st[4].x, Vector2(0.f, B[7].y)), 0.07f, 0.06f);
        }

        // deck: radiator spine with fins, rails, mast, roof hatch, exhaust stacks
        pb.on(m.graphite).box(Vector3(-3.3f, 1.92f, 0.f), Vector3(2.0f, 0.22f, 0.95f));
        for (int i = 0; i < 17; ++i) pb.on(m.graphite).box(Vector3(-5.1f + i * 0.225f, 2.42f, 0.f), Vector3(0.025f, 0.28f, 0.85f), 0.25f);
        for (float side : {-1.f, 1.f}) {
            Geo& g = pb.on(m.white);
            for (float x = -5.2f; x <= 2.5f; x += 0.96f) g.beam(Vector3(x, 1.7f, side * 1.62f), Vector3(x, 2.55f, side * 1.62f), 0.05f, 0.05f, kX);
            g.beam(Vector3(-5.2f, 2.55f, side * 1.62f), Vector3(2.48f, 2.55f, side * 1.62f), 0.05f, 0.05f);
            g.beam(Vector3(-5.2f, 2.12f, side * 1.62f), Vector3(2.48f, 2.12f, side * 1.62f), 0.04f, 0.04f);
        }
        pb.push(translation(Vector3(1.0f, 1.7f, -0.6f)));
        pb.on(m.graphite).lathe({{0.f, 0.f}, {0.32f, 0.f}, {0.32f, 0.12f}, {0.16f, 0.22f}, {0.13f, 2.2f}, {0.30f, 2.25f}, {0.30f, 2.4f}, {0.f, 2.4f}}, 20);
        pb.pop();
        hatch(pb, m.hatchWhite, m.white, Vector3(-0.4f, 1.7f, 0.65f), kX, neg(kZ), 1.1f, 0.9f, 0.06f);
        for (float side : {-1.f, 1.f}) {
            pb.push(translation(Vector3(-5.25f, 1.7f, side * 1.05f)));
            pb.on(m.graphite).lathe({{0.f, 0.f}, {0.2f, 0.f}, {0.2f, 0.85f}, {0.17f, 0.85f}, {0.17f, 0.7f}, {0.f, 0.7f}}, 16);
            pb.on(m.rubber).lathe({{0.205f, 0.62f}, {0.205f, 0.86f}, {0.17f, 0.86f}}, 16);
            pb.pop();
        }

        // flanks: doors, grab rails, vents with grime
        for (float side : {-1.f, 1.f}) {
            const Vector3 n(0, 0, side);
            const Vector3 u = side > 0 ? kX : neg(kX);
            hatch(pb, m.hatchRed, m.red, Vector3(-1.6f, 0.2f, side * 2.4f), u, kY, 1.25f, 1.65f, 0.06f);
            Geo& g = pb.on(m.steel);
            for (float x : {-0.78f, -2.42f}) {
                g.beam(Vector3(x, -0.35f, side * 2.4f), Vector3(x, -0.35f, side * 2.56f), 0.04f, 0.04f);
                g.beam(Vector3(x, 0.85f, side * 2.4f), Vector3(x, 0.85f, side * 2.56f), 0.04f, 0.04f);
                g.beam(Vector3(x, -0.37f, side * 2.56f), Vector3(x, 0.87f, side * 2.56f), 0.045f, 0.045f, kX);
            }
            for (float x : {-2.7f, 0.5f}) vent(pb, m, Vector3(x, 0.78f, side * 2.4f), n, 0.95f, 0.42f, 1.3f);
        }

        // nose: lamps and bumper; tail: exhaust grille
        for (float z : {-0.55f, 0.55f}) {
            pb.push(alongX(Vector3(6.36f, -0.55f, z)));
            pb.on(m.graphite).lathe({{0.f, 0.f}, {0.22f, 0.f}, {0.22f, 0.26f}, {0.19f, 0.28f}}, 16);
            pb.on(m.lamp).lathe({{0.19f, 0.28f}, {0.f, 0.3f}}, 16);
            pb.pop();
        }
        pb.on(m.graphite).box(Vector3(6.42f, -0.88f, 0.f), Vector3(0.12f, 0.1f, 0.78f));
        for (float z : {-1.05f, 1.05f}) {
            pb.push(alongX(Vector3(4.95f, 1.0f, z)));
            pb.on(m.graphite).lathe({{0.f, 0.f}, {0.13f, 0.f}, {0.13f, 0.2f}, {0.11f, 0.22f}}, 12);
            pb.on(m.lamp).lathe({{0.11f, 0.22f}, {0.f, 0.235f}}, 12);
            pb.pop();
        }
        pb.on(m.recess).panel(Vector3(-6.205f, 0.15f, 0.f), kZ, kY, 1.6f, 0.9f);
        for (int k = 0; k < 6; ++k) pb.on(m.graphite).box(Vector3(-6.24f, -0.22f + k * 0.15f, 0.f), Vector3(0.04f, 0.025f, 0.8f), 0.25f);

        // belly: sensor pod on a pylon, winch drum
        pb.on(m.graphite).box(Vector3(0.6f, -1.75f, 0.f), Vector3(0.5f, 0.2f, 0.18f));
        pb.push(mul(translation(Vector3(0.6f, -2.2f, 0.f)), alongX()));
        {
            std::vector<Vector2> p;
            for (int i = 0; i <= 12; ++i) {
                const float a = -kPi / 2 + kPi * i / 12;
                p.emplace_back(std::max(1e-3f, 0.5f * std::cos(a)), 1.3f * std::sin(a));
            }
            pb.on(m.white).lathe(p, 20);
        }
        pb.pop();
        pb.push(alongX(Vector3(1.82f, -2.2f, 0.f)));
        pb.on(m.glass).lathe({{0.12f, -0.02f}, {0.f, 0.02f}}, 12);
        pb.pop();
        pb.push(alongZ(Vector3(-4.6f, -1.75f, 0.f)));
        pb.on(m.steel).lathe({{0.f, -0.7f}, {0.3f, -0.7f}, {0.3f, 0.7f}, {0.f, 0.7f}}, 18);
        pb.pop();

        // hip drums (axisymmetric, so they stay put while the coxa yaws inside them)
        for (const auto& h : kit.hip) {
            pb.push(translation(h));
            pb.on(m.white).lathe({{0.f, -1.25f}, {0.9f, -1.25f}, {0.9f, 0.35f}, {0.75f, 0.5f}, {0.f, 0.5f}}, 28);
            pb.on(m.graphite).lathe({{0.9f, -0.62f}, {0.95f, -0.62f}, {0.95f, 0.62f}, {0.9f, 0.62f}}, 28);
            pb.on(m.steel).lathe({{0.f, 0.5f}, {0.4f, 0.5f}, {0.36f, 0.6f}, {0.f, 0.6f}}, 20);
            pb.pop();
        }
    }

    void buildCoxa(PartBuilder& pb, const Mats& m) {
        pb.on(m.graphite).box(Vector3(1.05f, 0.f, 0.f), Vector3(0.6f, 0.38f, 0.42f));
        for (float z : {-0.5f, 0.5f}) pb.on(m.graphite).box(Vector3(1.45f, 0.f, z), Vector3(0.42f, 0.6f, 0.07f));
        pb.push(alongZ(Vector3(1.45f, 0.f, 0.f)));
        pb.on(m.steel).lathe(rodProfile(0.18f, -0.62f, 0.62f), 16);
        pb.pop();
        for (float z : {-0.2f, 0.2f}) {
            pb.on(m.graphite).beam(Vector3(0.75f, -0.25f, z), Vector3(2.1f, -1.05f, z), 0.34f, 0.06f, kZ);
            pb.on(m.graphite).beam(Vector3(0.6f, -0.3f, z), Vector3(1.6f, -0.3f, z), 0.2f, 0.06f, kZ);
        }
        pb.push(alongZ(Vector3(2.0f, -1.0f, 0.f)));
        pb.on(m.steel).lathe(rodProfile(0.09f, -0.26f, 0.26f), 12);
        pb.pop();
        pb.on(m.white).box(Vector3(1.0f, 0.42f, 0.f), Vector3(0.5f, 0.04f, 0.4f));
    }

    void buildFemur(PartBuilder& pb, const Mats& m, const WalkerKit& k) {
        const float L = k.femur_;
        pb.push(alongZ());
        pb.on(m.graphite).lathe(discProfile(0.46f, 0.42f, 0.06f), 22);
        pb.pop();
        loftOcta(pb, {{0.3f, 0.36f, 0.26f, 0.26f, 0.46f, -0.46f, 0.1f, 0.1f}, {L - 0.6f, 0.3f, 0.22f, 0.22f, 0.36f, -0.36f, 0.08f, 0.08f}},
                 [&](int, int) { return m.graphite; }, m.graphite, nullptr);
        for (float z : {-0.37f, 0.37f}) pb.on(m.white).box(Vector3(2.7f, 0.f, z), Vector3(1.6f, 0.27f, 0.025f));
        // knee housing
        pb.push(alongZ(Vector3(L, 0.f, 0.f)));
        pb.on(m.white).lathe(discProfile(0.95f, 0.62f, 0.17f), 36);
        pb.on(m.steel).lathe({{0.f, 0.62f}, {0.45f, 0.62f}, {0.45f, 0.72f}, {0.32f, 0.8f}, {0.f, 0.8f}}, 24);
        pb.pop();
        pb.push(mul(translation(Vector3(L, 0.f, 0.f)), rotation(kX, -kPi / 2)));
        pb.on(m.steel).lathe({{0.f, 0.62f}, {0.45f, 0.62f}, {0.45f, 0.72f}, {0.32f, 0.8f}, {0.f, 0.8f}}, 24);
        pb.pop();
        // ram lugs
        for (float z : {-0.18f, 0.18f}) {
            pb.on(m.graphite).beam(Vector3(k.femurLiftLug.x - 0.45f, -0.3f, z), Vector3(k.femurLiftLug.x + 0.05f, k.femurLiftLug.y - 0.12f, z), 0.3f, 0.06f, kZ);
            pb.on(m.graphite).beam(Vector3(k.femurLiftLug.x + 0.5f, -0.3f, z), Vector3(k.femurLiftLug.x + 0.05f, k.femurLiftLug.y - 0.12f, z), 0.3f, 0.06f, kZ);
            pb.on(m.graphite).beam(Vector3(k.femurKneeLug.x - 0.6f, 0.36f, z), Vector3(k.femurKneeLug.x, k.femurKneeLug.y + 0.12f, z), 0.32f, 0.06f, kZ);
            pb.on(m.graphite).beam(Vector3(k.femurKneeLug.x + 0.6f, 0.36f, z), Vector3(k.femurKneeLug.x, k.femurKneeLug.y + 0.12f, z), 0.32f, 0.06f, kZ);
        }
        for (const Vector3& lug : {k.femurLiftLug, k.femurKneeLug}) {
            pb.push(alongZ(lug));
            pb.on(m.steel).lathe(rodProfile(0.09f, -0.24f, 0.24f), 12);
            pb.pop();
        }
        // hoses along the top
        for (float z : {-0.07f, 0.07f}) {
            std::vector<Vector3> path;
            for (int i = 0; i <= 10; ++i) {
                const float t = i / 10.f;
                path.emplace_back(mixf(0.5f, L - 1.05f, t), 0.42f + 0.12f * std::sin(kPi * t), z);
            }
            pb.on(m.rubber).tube(path, 0.045f, 8);
        }
    }

    void buildTibia(PartBuilder& pb, const Mats& m, const WalkerKit& k) {
        const float L = k.tibia_;
        loftOcta(pb, {{0.5f, 0.34f, 0.24f, 0.24f, 0.44f, -0.44f, 0.1f, 0.1f}, {L - 2.3f, 0.26f, 0.18f, 0.18f, 0.3f, -0.3f, 0.08f, 0.08f}},
                 [&](int, int) { return m.graphite; }, nullptr, m.graphite);
        pb.on(m.white).box(Vector3(3.3f, 0.43f, 0.f), Vector3(2.4f, 0.035f, 0.28f));
        pb.push(alongX(Vector3(L - 2.45f, 0.f, 0.f)));
        pb.on(m.graphite).lathe({{0.f, 0.f}, {0.36f, 0.f}, {0.36f, 1.3f}, {0.3f, 1.4f}, {0.f, 1.4f}}, 22);
        pb.pop();
        pb.push(alongX(Vector3(L - 1.1f, 0.f, 0.f)));
        pb.on(m.chrome).lathe(rodProfile(0.23f, 0.f, 0.75f), 22);
        pb.pop();
        pb.push(alongX(Vector3(L - 1.06f, 0.f, 0.f)));
        pb.on(m.rubber).lathe({{0.23f, 0.f}, {0.31f, 0.f}, {0.31f, 0.07f}, {0.23f, 0.07f}}, 22);
        pb.pop();
        pb.push(translation(Vector3(L, 0.f, 0.f)));
        pb.on(m.steel).lathe(sphereProfile(0.4f, 12), 22);
        pb.pop();
        for (float z : {-0.2f, 0.2f}) {
            pb.on(m.white).beam(Vector3(-0.2f, -0.05f, z), Vector3(k.tibiaKneeLug.x - 0.12f, k.tibiaKneeLug.y - 0.03f, z), 0.4f, 0.07f, kZ);
        }
        pb.push(alongZ(k.tibiaKneeLug));
        pb.on(m.steel).lathe(rodProfile(0.09f, -0.27f, 0.27f), 12);
        pb.pop();
    }

    void buildFoot(PartBuilder& pb, const Mats& m, const WalkerKit& k) {
        const Sole& s = k.sole;
        pb.on(m.steel).lathe({{0.f, 0.f}, {s.pad, 0.f}, {s.pad, 0.2f}, {1.12f, 0.4f}, {0.6f, 0.48f}, {0.f, 0.48f}}, 40);
        pb.on(m.graphite).lathe({{0.f, 0.48f}, {0.5f, 0.48f}, {0.45f, 0.58f}, {0.f, 0.58f}}, 24);
        for (const auto& t : s.toes) {
            const Vector3 dir(t.dx, 0.f, t.dz);
            pb.push(basisAt(Vector3(), dir, kY, cross(dir, kY)));
            const float rA = s.toeRadius(t, 0.8f), rB = s.toeRadius(t, t.len - 0.1f);
            loftOcta(pb, {{0.8f, rA, rA * 0.7f, rA, 0.36f, 0.f, 0.12f, 0.f}, {t.len - 0.1f, rB + 0.03f, 0.1f, rB + 0.03f, 0.16f, 0.f, 0.06f, 0.f}, {t.len + 0.25f, 0.06f, 0.04f, 0.06f, 0.05f, 0.f, 0.02f, 0.f}},
                     [&](int, int) { return m.graphite; }, nullptr, m.graphite, 0.25f);
            pb.pop();
        }
        // cleats, matching Sole::cleat
        for (float rc : {0.55f, 1.05f}) pb.on(m.rubber).lathe({{rc - 0.08f, 0.f}, {rc - 0.08f, -0.06f}, {rc + 0.08f, -0.06f}, {rc + 0.08f, 0.f}}, 40, 0.3f);
        for (int i = 0; i < 6; ++i) {
            const float a = kPi / 6 + i * kPi / 3;
            const Vector3 d(std::cos(a), 0.f, std::sin(a));
            pb.on(m.rubber).beam(d * 0.25f + Vector3(0, -0.03f, 0), d * 1.25f + Vector3(0, -0.03f, 0), 0.14f, 0.06f);
        }
        for (const auto& t : s.toes) {
            const Vector3 dir(t.dx, 0.f, t.dz), acr(-t.dz, 0.f, t.dx);
            for (float b : {1.5f, 1.85f, 2.2f}) {
                const float r = s.toeRadius(t, b) * 0.85f;
                const Vector3 c = dir * b + Vector3(0, -0.03f, 0);
                pb.on(m.rubber).beam(c - acr * r, c + acr * r, 0.14f, 0.06f);
            }
        }
        // the curved ankle fork and its pin
        for (float side : {-1.f, 1.f}) {
            std::vector<Vector3> path;
            for (int i = 0; i <= 8; ++i) {
                const float t = i / 8.f;
                const float x = 0.35f * (1 - t) * (1 - t) + 0.24f * 2 * t * (1 - t) + 0.0f * t * t;
                const float y = 0.45f * (1 - t) * (1 - t) + 0.62f * 2 * t * (1 - t) + k.ankle * t * t;
                const float z = side * (0.64f * (1 - t) * (1 - t) + 0.72f * 2 * t * (1 - t) + 0.53f * t * t);
                path.emplace_back(x, y, z);
            }
            pb.on(m.white).tube(path, 0.11f, 10);
        }
        pb.push(alongZ(Vector3(0.f, k.ankle, 0.f)));
        pb.on(m.steel).lathe(rodProfile(0.13f, -0.66f, 0.66f), 14);
        pb.pop();
    }

    void buildRam(PartBuilder& barrel, PartBuilder& rod, const Mats& m, float Lb, float Lr) {
        barrel.push(alongZ());
        barrel.on(m.steel).lathe(rodProfile(0.17f, -0.12f, 0.12f), 14);
        barrel.pop();
        barrel.on(m.graphite).box(Vector3(0.18f, 0.f, 0.f), Vector3(0.12f, 0.1f, 0.09f), 0.2f);
        barrel.push(alongX());
        barrel.on(m.graphite).lathe({{0.f, 0.28f}, {0.22f, 0.28f}, {0.22f, Lb - 0.16f}, {0.27f, Lb - 0.16f}, {0.27f, Lb}, {0.f, Lb}}, 18, 0.35f);
        barrel.pop();
        barrel.on(m.rubber).tube({Vector3(0.45f, 0.2f, 0.f), Vector3(Lb * 0.5f, 0.32f, 0.f), Vector3(Lb - 0.3f, 0.24f, 0.f)}, 0.035f, 6);
        rod.push(alongZ());
        rod.on(m.steel).lathe(rodProfile(0.15f, -0.1f, 0.1f), 14);
        rod.pop();
        rod.push(alongX());
        rod.on(m.chrome).lathe(rodProfile(0.12f, 0.12f, Lr), 18, 0.4f);
        rod.pop();
    }

    WalkerKit buildWalkerKit(const Mats& m) {
        WalkerKit k;
        k.hip = {Vector3(2.6f, -0.3f, -2.95f), Vector3(2.6f, -0.3f, 2.95f), Vector3(-4.3f, -0.3f, -2.95f), Vector3(-4.3f, -0.3f, 2.95f)};
        k.stance = {Vector2(5.4f, -8.4f), Vector2(5.4f, 8.4f), Vector2(-7.2f, -8.4f), Vector2(-7.2f, 8.4f)};
        {
            PartBuilder pb;
            buildHull(pb, m, k);
            k.hull = pb.build(0.08f, 2.2f, 1.15f);
        }
        {
            PartBuilder pb;
            pb.on(m.white).box(Vector3(0.f, 0.17f, 0.f), Vector3(0.32f, 0.15f, 0.5f), 0.2f);
            pb.push(alongX(Vector3(0.32f, 0.17f, 0.f)));
            pb.on(m.glass).lathe({{0.12f, 0.f}, {0.12f, 0.04f}, {0.f, 0.06f}}, 16);
            pb.pop();
            pb.push(translation(Vector3(-0.1f, 0.32f, 0.f)));
            pb.on(m.graphite).lathe({{0.f, 0.f}, {0.06f, 0.f}, {0.06f, 0.3f}, {0.f, 0.3f}}, 10);
            pb.pop();
            k.sensor = pb.build(0.04f, 0.5f, 0.6f);
        }
        {
            PartBuilder pb;
            buildCoxa(pb, m);
            k.coxa = pb.build(0.06f, 1.2f, 1.0f);
        }
        {
            PartBuilder pb;
            buildFemur(pb, m, k);
            k.femur = pb.build(0.06f, 1.3f, 1.0f);
        }
        {
            PartBuilder pb;
            buildTibia(pb, m, k);
            k.tibia = pb.build(0.06f, 1.3f, 1.0f);
        }
        {
            PartBuilder pb;
            buildFoot(pb, m, k);
            k.foot = pb.build(0.05f, 1.0f, 1.0f);
        }
        {
            PartBuilder b, r;
            buildRam(b, r, m, k.liftBarrel, k.liftRod);
            k.barrelLift = b.build(0.05f, 0.5f, 0.6f);
            k.rodLift = r.build(0.05f, 0.5f, 0.6f);
        }
        {
            PartBuilder b, r;
            buildRam(b, r, m, k.kneeBarrel, k.kneeRod);
            k.barrelKnee = b.build(0.05f, 0.5f, 0.6f);
            k.rodKnee = r.build(0.05f, 0.5f, 0.6f);
        }
        return k;
    }

    // ================================================================== footprint map

    // Every print pressed into the snow round the walk ring, in polar coordinates (u = angle,
    // v = radius): R = depth pressed in, G = snow pushed up round the rim. The snow shader
    // displaces and shades by it; prints stay where they were made.
    struct TrackMap {
        static constexpr int W = 16384, H = 512;
        static constexpr float kDepth = 0.6f, kRim = 0.15f;
        float rIn, rOut;
        std::shared_ptr<DataTexture> tex;
        bool dirty = false;

        TrackMap(float rin, float rout): rIn(rin), rOut(rout) {
            tex = DataTexture::create(ImageData(std::vector<unsigned char>(static_cast<size_t>(W) * H * 2, 0)), W, H);
            tex->format = Format::RG;
            tex->type = Type::UnsignedByte;
            tex->wrapS = TextureWrapping::Repeat;
            tex->wrapT = TextureWrapping::ClampToEdge;
            tex->magFilter = Filter::Linear;
            tex->minFilter = Filter::Linear;
            tex->generateMipmaps = false;
            tex->needsUpdate();
        }

        // shape(lx, lz, depth&, rim&) in the stamp's own frame (x along fwd, z to its right).
        template<class Shape>
        void stamp(float cx, float cz, float fx, float fz, float reach, Shape&& shape) {
            const float rc = std::hypot(cx, cz);
            if (rc + reach < rIn || rc - reach > rOut) return;
            const float ac = std::atan2(cz, cx);
            const float da = reach / std::max(rc - reach, 1.f);
            const int i0 = static_cast<int>(std::floor(((ac - da) / kTau + 0.5f) * W));
            const int i1 = static_cast<int>(std::ceil(((ac + da) / kTau + 0.5f) * W));
            const int j0 = std::max(0, static_cast<int>(std::floor((rc - reach - rIn) / (rOut - rIn) * H)));
            const int j1 = std::min(H - 1, static_cast<int>(std::ceil((rc + reach - rIn) / (rOut - rIn) * H)));
            auto& d = tex->image().data<unsigned char>();
            for (int j = j0; j <= j1; ++j) {
                const float r = rIn + (j + 0.5f) / H * (rOut - rIn);
                for (int ii = i0; ii <= i1; ++ii) {
                    const int i = ((ii % W) + W) % W;
                    const float a = ((ii + 0.5f) / W - 0.5f) * kTau;
                    const float x = r * std::cos(a) - cx, z = r * std::sin(a) - cz;
                    const float lx = x * fx + z * fz, lz = -x * fz + z * fx;
                    float depth = 0, rim = 0;
                    shape(lx, lz, depth, rim);
                    const size_t k = (static_cast<size_t>(j) * W + i) * 2;
                    const float nr = depth / kDepth, ng = rim / kRim;
                    if (nr > 0.003f) {
                        d[k] = std::max(d[k], toByte(nr));
                        if (nr > 0.03f) d[k + 1] = std::min(d[k + 1], toByte(ng));
                    } else if (ng > 0.f) {
                        d[k + 1] = std::max(d[k + 1], toByte(ng));
                    }
                }
            }
            dirty = true;
        }

        void stampFoot(const Sole& sole, const Vector3& p, const Vector3& fwd, float sink, int seed) {
            stamp(p.x, p.z, fwd.x, fwd.z, 3.4f, [&](float lx, float lz, float& depth, float& rim) {
                const float sd = sole.sdf(lx, lz);
                if (sd < 0.1f) depth = smoothstepf(0.1f, -0.1f, sd) * (sink + 0.025f * sole.cleat(lx, lz));
                if (sd > -0.05f && sd < 0.5f) {
                    const float t = std::max(sd, 0.f) / 0.5f;
                    rim = 0.09f * (1 - t) * (1 - t) * smoothstepf(-0.05f, 0.1f, sd) *
                          (0.75f + 0.5f * vnoise(lx * 2.3f + seed * 7.3f, lz * 2.3f, 0, 77));
                }
            });
        }

        void stampBoot(const Vector3& p, const Vector3& fwd) {
            stamp(p.x, p.z, fwd.x, fwd.z, 0.35f, [](float lx, float lz, float& depth, float& rim) {
                const float sd = (std::hypot(lx / 0.15f, lz / 0.06f) - 1.f) * 0.06f;
                if (sd < 0.02f) depth = 0.07f * smoothstepf(0.02f, -0.02f, sd);
                if (sd > 0.f && sd < 0.1f) rim = 0.025f * (1.f - sd / 0.1f);
            });
        }
    };

    // ================================================================== snow, sky, mountains

    std::string hook(std::string s, const std::string& what, const std::string& with) {
        const auto p = s.find(what);
        if (p == std::string::npos) {
            std::cerr << "glacier_walker: shader hook not found: " << what << std::endl;
            return s;
        }
        s.replace(p, what.size(), with);
        return s;
    }

    const char* kSnowCommon = R"(
uniform sampler2D trackMap;
uniform vec4 trackParams;// r inner, r outer, depth scale, rim scale
varying vec3 vSnowW;
vec2 snowTrackUV( vec2 xz ) {
	float r = length( xz );
	return vec2( atan( xz.y, xz.x ) * 0.15915494 + 0.5, ( r - trackParams.x ) / ( trackParams.y - trackParams.x ) );
}
vec2 snowTrackSample( vec2 xz ) {
	vec2 uv = snowTrackUV( xz );
	if ( uv.y <= 0.0 || uv.y >= 1.0 ) return vec2( 0.0 );
	return textureLod( trackMap, uv, 0.0 ).rg;
}
float snowTrackHeight( vec2 xz ) {
	vec2 s = snowTrackSample( xz );
	return - s.r * trackParams.z + s.g * trackParams.w;
}
// the same, averaged over the mesh spacing (vertex displacement only)
float snowTrackHeightSoft( vec2 xz ) {
	float h = 0.0;
	for ( int j = -1; j <= 1; ++ j )
		for ( int i = -1; i <= 1; ++ i ) {
			float w = ( i == 0 ? 2.0 : 1.0 ) * ( j == 0 ? 2.0 : 1.0 );
			h += w * snowTrackHeight( xz + vec2( float( i ), float( j ) ) * 0.08 );
		}
	return h / 16.0;
}
)";

    const char* kSnowFragPars = R"(
float snowHash( vec2 p ) {
	vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
	p3 += dot( p3, p3.yzx + 33.33 );
	return fract( ( p3.x + p3.y ) * p3.z );
}
float snowNoise( vec2 p ) {
	vec2 i = floor( p ), f = fract( p );
	vec2 u = f * f * ( 3.0 - 2.0 * f );
	return mix( mix( snowHash( i ), snowHash( i + vec2( 1.0, 0.0 ) ), u.x ),
	            mix( snowHash( i + vec2( 0.0, 1.0 ) ), snowHash( i + vec2( 1.0, 1.0 ) ), u.x ), u.y );
}
// wind-carved ridges along the prevailing wind, plus fine grain close up (metres)
float snowRelief( vec2 p, float detail ) {
	vec2 q = vec2( dot( p, vec2( 0.8, 0.6 ) ), dot( p, vec2( -0.6, 0.8 ) ) );
	float h = 0.035 * snowNoise( q * vec2( 0.3, 0.9 ) ) + 0.02 * snowNoise( q * vec2( 0.9, 2.6 ) + 7.0 );
	h += detail * ( 0.006 * snowNoise( p * 9.0 ) + 0.0025 * snowNoise( p * 31.0 ) );
	return h;
}
)";

    const char* kSnowSurface = R"(
	// --- snow: prints, packed snow, wind relief
	vec2 snowXZ = vSnowW.xz;
	float snowPx = length( fwidth( snowXZ ) );
	float snowDetail = 1.0 - smoothstep( 0.02, 0.15, snowPx );
	const float snowE = 0.05;
	vec2 snowTrack = snowTrackSample( snowXZ );
	float snowDepth = snowTrack.r * trackParams.z;
	float snowRim = snowTrack.g * trackParams.w;
	float snowPack = smoothstep( 0.012, 0.08, snowDepth );
	vec2 snowGrad = vec2(
		snowTrackHeight( snowXZ + vec2( snowE, 0.0 ) ) - snowTrackHeight( snowXZ - vec2( snowE, 0.0 ) ),
		snowTrackHeight( snowXZ + vec2( 0.0, snowE ) ) - snowTrackHeight( snowXZ - vec2( 0.0, snowE ) ) ) / ( 2.0 * snowE );
	float snowR0 = snowRelief( snowXZ, snowDetail );
	vec2 snowRG = vec2( snowRelief( snowXZ + vec2( snowE, 0.0 ), snowDetail ) - snowR0,
	                    snowRelief( snowXZ + vec2( 0.0, snowE ), snowDetail ) - snowR0 ) / snowE;
	snowGrad += snowRG * ( 1.0 - 0.85 * snowPack );
	vec3 snowN = inverseTransformDirection( normal, viewMatrix );
	snowN = normalize( snowN + vec3( - snowGrad.x, 0.0, - snowGrad.y ) );
	normal = normalize( ( viewMatrix * vec4( snowN, 0.0 ) ).xyz );

	vec3 snowCol = mix( vec3( 0.93, 0.95, 0.98 ), vec3( 0.80, 0.85, 0.92 ), snowPack );
	snowCol *= 1.0 - 0.16 * smoothstep( 0.1, 0.4, snowDepth );
	snowCol = mix( snowCol, snowCol * vec3( 0.86, 0.93, 1.0 ), smoothstep( 0.18, 0.4, snowDepth ) );
	snowCol *= 1.0 + 0.04 * smoothstep( 0.0, 0.08, snowRim );
	snowCol *= 0.97 + 0.03 * snowNoise( snowXZ * 0.7 );
	diffuseColor.rgb *= snowCol;
	roughnessFactor = mix( 0.86, 0.5, snowPack );
)";

    const char* kSnowGlints = R"(
	#if NUM_DIR_LIGHTS > 0
	{
		// sun glints off single grains of fresh powder
		float sunVis = 1.0;
		#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
			DirectionalLightShadow sdl = directionalLightShadows[ 0 ];
			sunVis = getShadow( directionalShadowMap[ 0 ], sdl.shadowMapSize, sdl.shadowIntensity, sdl.shadowBias, sdl.shadowRadius, vDirectionalShadowCoord[ 0 ] );
		#endif
		vec3 L = directionalLights[ 0 ].direction;
		vec3 H = normalize( L + geometry.viewDir );
		vec2 gp = vSnowW.xz * 30.0;
		vec2 cell = floor( gp );
		vec2 spot = vec2( snowHash( cell + 5.3 ), snowHash( cell + 1.9 ) ) * 0.6 + 0.2;
		float grain = 1.0 - smoothstep( 0.08, 0.16, length( fract( gp ) - spot ) );
		float pick = snowHash( cell );
		vec3 jitter = vec3( snowHash( cell + 3.1 ) - 0.5, 0.0, snowHash( cell + 9.7 ) - 0.5 ) * 0.9;
		vec3 facet = normalize( ( viewMatrix * vec4( normalize( snowN + jitter ), 0.0 ) ).xyz );
		float g = pow( max( dot( facet, H ), 0.0 ), 600.0 ) * step( 0.5, pick ) * grain;
		g *= ( 1.0 - smoothstep( 0.012, 0.045, snowPx ) ) * ( 1.0 - snowPack ) * sunVis;
		totalEmissiveRadiance += directionalLights[ 0 ].color * g * 12.0;
	}
	#endif
)";

    std::shared_ptr<ShaderMaterial> makeSnowMaterial(TrackMap& track) {
        const auto& lib = shaders::ShaderLib::instance().get("physical");
        auto m = ShaderMaterial::create();
        m->uniforms = lib.uniforms;
        m->uniforms["diffuse"].setValue(Color(1, 1, 1));
        m->uniforms["roughness"].setValue(0.85f);
        m->uniforms["metalness"].setValue(0.f);
        m->uniforms["emissive"].setValue(Color(0, 0, 0));
        m->uniforms["opacity"].setValue(1.f);
        m->uniforms["trackMap"] = Uniform(track.tex.get());
        m->uniforms["trackParams"] = Uniform(Vector4(track.rIn, track.rOut, TrackMap::kDepth, TrackMap::kRim));

        std::string vs = shaders::ShaderChunk::instance().meshphysical_vert();
        vs = hook(vs, "void main() {", std::string(kSnowCommon) + "\nvoid main() {");
        vs = hook(vs, "#include <begin_vertex>", R"(#include <begin_vertex>
	vec4 snowW0 = modelMatrix * vec4( transformed, 1.0 );
	float snowH = snowTrackHeightSoft( snowW0.xz );
	transformed.y += snowH;
	vSnowW = snowW0.xyz + vec3( 0.0, snowH, 0.0 );)");

        std::string fs = shaders::ShaderChunk::instance().meshphysical_frag();
        fs = hook(fs, "void main() {", std::string(kSnowCommon) + kSnowFragPars + "\nvoid main() {");
        fs = hook(fs, "#include <normal_fragment_maps>", std::string("#include <normal_fragment_maps>\n") + kSnowSurface);
        fs = hook(fs, "#include <lights_fragment_end>", std::string("#include <lights_fragment_end>\n") + kSnowGlints);

        m->vertexShader = vs;
        m->fragmentShader = fs;
        m->lights = true;
        m->fog = true;
        return m;
    }

    // The glacier as one polar mesh: dense (0.2 m) across the walk ring, where the prints
    // are, and stretching out to the foot of the ranges.
    std::shared_ptr<BufferGeometry> buildSnowGeometry(float rIn, float rOut) {
        std::vector<float> rings;
        {
            std::vector<float> inner;
            float r = rIn, step = 0.2f;
            while (r > step) {
                r -= step;
                inner.push_back(r);
                step *= 1.07f;
            }
            inner.push_back(0.f);
            rings.assign(inner.rbegin(), inner.rend());
        }
        for (float r = rIn; r < rOut; r += 0.13f) rings.push_back(r);
        {
            float r = rOut, step = 0.2f;
            while (r < 2800.f) {
                rings.push_back(r);
                r += step;
                step *= 1.03f;
            }
            rings.push_back(2800.f);
        }
        const int S = 6144;
        std::vector<float> pos, nrm;
        pos.reserve(rings.size() * S * 3);
        nrm.reserve(rings.size() * S * 3);
        for (float r : rings)
            for (int i = 0; i < S; ++i) {
                const float a = (i + 0.5f) / S * kTau - kPi;
                const float x = r * std::cos(a), z = r * std::sin(a);
                const Vector3 n = groundNormal(x, z);
                pos.insert(pos.end(), {x, groundHeight(x, z), z});
                nrm.insert(nrm.end(), {n.x, n.y, n.z});
            }
        std::vector<unsigned int> idx;
        idx.reserve((rings.size() - 1) * S * 6);
        for (size_t j = 0; j + 1 < rings.size(); ++j)
            for (int i = 0; i < S; ++i) {
                const unsigned int a = static_cast<unsigned int>(j * S + i), b = static_cast<unsigned int>(j * S + (i + 1) % S);
                const unsigned int c = static_cast<unsigned int>((j + 1) * S + i), d = static_cast<unsigned int>((j + 1) * S + (i + 1) % S);
                idx.insert(idx.end(), {a, b, c, b, d, c});
            }
        auto g = BufferGeometry::create();
        g->setAttribute("position", FloatBufferAttribute::create(std::move(pos), 3));
        g->setAttribute("normal", FloatBufferAttribute::create(std::move(nrm), 3));
        g->setIndex(std::move(idx));
        g->computeBoundingSphere();
        return g;
    }


    const char* kSkyVert = R"(
varying vec3 vDir;
void main() {
	vDir = position;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
)";

    const char* kSkyFrag = R"(
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uHorizon;
uniform vec3 uZenith;
uniform float uTime;
varying vec3 vDir;
float skyHash( vec2 p ) {
	vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
	p3 += dot( p3, p3.yzx + 33.33 );
	return fract( ( p3.x + p3.y ) * p3.z );
}
float skyNoise( vec2 p ) {
	vec2 i = floor( p ), f = fract( p );
	vec2 u = f * f * ( 3.0 - 2.0 * f );
	return mix( mix( skyHash( i ), skyHash( i + vec2( 1.0, 0.0 ) ), u.x ),
	            mix( skyHash( i + vec2( 0.0, 1.0 ) ), skyHash( i + vec2( 1.0, 1.0 ) ), u.x ), u.y );
}
float skyFbm( vec2 p ) {
	float s = 0.0, a = 0.5;
	for ( int i = 0; i < 6; ++ i ) {
		s += a * skyNoise( p );
		p = p * 2.03 + vec2( 1.7, 9.2 );
		a *= 0.5;
	}
	return s / 0.984;
}
void main() {
	vec3 d = normalize( vDir );
	float h = d.y;
	float mu = max( dot( d, uSunDir ), 0.0 );
	vec3 sunFlat = normalize( vec3( uSunDir.x, 0.0, uSunDir.z ) );
	float az = dot( normalize( vec3( d.x, 0.0, d.z ) + vec3( 1e-5 ) ), sunFlat );

	// the break: the deck opens low over the sun's bearing
	float gap = smoothstep( -0.25, 0.85, az ) * ( 1.0 - smoothstep( 0.05, 0.45, h ) );
	float glow = pow( mu, 4.0 );

	// clear sky: cold overhead; inside the break, warm light pours across the low sky
	vec3 clear = mix( uHorizon * 1.05, uZenith, pow( clamp( h, 0.0, 1.0 ), 0.5 ) );
	clear = mix( clear, uSunColor * ( 0.42 + 0.9 * glow ), gap * ( 1.0 - smoothstep( 0.0, 0.32, h ) ) );
	clear += uSunColor * ( 0.35 * pow( mu, 16.0 ) + 1.6 * pow( mu, 160.0 ) );
	vec3 col = clear;

	if ( h > 0.0 ) {
		vec2 p = d.xz / ( h + 0.1 ) * 0.9 + uTime * vec2( 0.004, 0.0015 );
		float n = skyFbm( p * 1.4 );
		float cover = mix( 0.86, 0.28, gap );
		float c = smoothstep( 1.0 - cover - 0.1, 1.0 - cover + 0.16, n );
		float thick = smoothstep( 0.42, 0.9, n );
		vec3 base = mix( vec3( 0.40, 0.44, 0.52 ), vec3( 0.17, 0.20, 0.26 ), thick );
		float edge = ( 1.0 - thick ) * c;
		vec3 lit = uSunColor * ( ( 0.3 + 1.2 * glow ) * edge + 0.25 * glow ) * ( 0.15 + 0.85 * gap );
		vec3 cloud = base + lit;
		col = mix( clear, cloud, c * smoothstep( 0.0, 0.06, h ) );
		float disc = smoothstep( 0.99955, 0.99975, mu );
		col += uSunColor * 20.0 * disc * ( 1.0 - c );
	} else {
		col = mix( uHorizon, vec3( 0.74, 0.78, 0.85 ), smoothstep( 0.0, -0.08, h ) );
	}
	// low haze over the ice, warmed where the break lets the sun through
	vec3 haze = mix( uHorizon, uSunColor * 0.75, 0.5 * gap * ( 0.4 + 0.6 * glow ) );
	col = mix( col, haze, exp( - max( h, 0.0 ) * 34.0 ) * 0.8 );
	gl_FragColor = vec4( col, 1.0 );
	#include <tonemapping_fragment>
	#include <encodings_fragment>
}
)";

    std::shared_ptr<ShaderMaterial> makeSkyMaterial(const Vector3& sunDir, const Color& sun, const Color& horizon) {
        auto m = ShaderMaterial::create();
        m->uniforms["uSunDir"] = Uniform(sunDir);
        m->uniforms["uSunColor"] = Uniform(sun);
        m->uniforms["uHorizon"] = Uniform(horizon);
        m->uniforms["uZenith"] = Uniform(Color(0.22f, 0.31f, 0.46f));
        m->uniforms["uTime"] = Uniform(0.f);
        m->vertexShader = kSkyVert;
        m->fragmentShader = kSkyFrag;
        m->side = Side::Back;
        m->depthWrite = false;
        m->fog = false;
        return m;
    }

    // ================================================================== particles

    const char* kPuffVert = R"(
attribute float aSize;
attribute float aAlpha;
uniform float uScale;
uniform vec3 uSunDir;
varying float vAlpha;
varying float vFwd;
#include <common>
#include <fog_pars_vertex>
void main() {
	vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
	gl_Position = projectionMatrix * mvPosition;
	gl_PointSize = clamp( aSize * uScale / - mvPosition.z, 0.0, 512.0 );
	vAlpha = aAlpha;
	vec3 sunV = normalize( ( viewMatrix * vec4( uSunDir, 0.0 ) ).xyz );
	vFwd = max( dot( normalize( mvPosition.xyz ), sunV ), 0.0 );
	#include <fog_vertex>
}
)";

    const char* kDriftVert = R"(
attribute float aSize;
attribute float aAlpha;
attribute float aSeed;
uniform float uScale;
uniform vec3 uSunDir;
uniform vec3 uOrigin;
uniform vec3 uWind;
uniform vec3 uBox;
uniform float uTime;
varying float vAlpha;
varying float vFwd;
#include <common>
#include <fog_pars_vertex>
void main() {
	vec3 p = position + uWind * uTime * ( 0.7 + 0.6 * aSeed );
	p.xz = uOrigin.xz + mod( p.xz - uOrigin.xz + 0.5 * uBox.xz, uBox.xz ) - 0.5 * uBox.xz;
	p.y = uOrigin.y + position.y + 0.25 * sin( uTime * ( 0.8 + aSeed ) + aSeed * 40.0 );
	vec4 mvPosition = modelViewMatrix * vec4( p, 1.0 );
	gl_Position = projectionMatrix * mvPosition;
	gl_PointSize = clamp( aSize * uScale / - mvPosition.z, 0.0, 512.0 );
	vec2 rel = abs( p.xz - uOrigin.xz ) / uBox.xz;
	vAlpha = aAlpha * ( 1.0 - smoothstep( 0.36, 0.5, max( rel.x, rel.y ) ) );
	vec3 sunV = normalize( ( viewMatrix * vec4( uSunDir, 0.0 ) ).xyz );
	vFwd = max( dot( normalize( mvPosition.xyz ), sunV ), 0.0 );
	#include <fog_vertex>
}
)";

    const char* kPuffFrag = R"(
uniform vec3 uSunColor;
uniform vec3 uSkyColor;
varying float vAlpha;
varying float vFwd;
#include <common>
#include <fog_pars_fragment>
void main() {
	vec2 c = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot( c, c );
	if ( r2 > 1.0 ) discard;
	float a = vAlpha * ( 1.0 - r2 ) * ( 1.0 - r2 );
	vec3 col = uSkyColor + uSunColor * ( 0.35 + 1.6 * pow( vFwd, 6.0 ) );
	gl_FragColor = vec4( col, a );
	#include <tonemapping_fragment>
	#include <encodings_fragment>
	#include <fog_fragment>
}
)";

    std::shared_ptr<ShaderMaterial> makePuffMaterial(const char* vert, const Vector3& sunDir, const Color& sun, const Color& sky) {
        auto m = ShaderMaterial::create();
        m->uniforms = shaders::UniformsLib::instance().fog;
        m->uniforms["uScale"] = Uniform(500.f);
        m->uniforms["uSunDir"] = Uniform(sunDir);
        m->uniforms["uSunColor"] = Uniform(sun);
        m->uniforms["uSkyColor"] = Uniform(sky);
        m->uniforms["uOrigin"] = Uniform(Vector3());
        m->uniforms["uWind"] = Uniform(Vector3(3.2f, 0.f, 1.4f));
        m->uniforms["uBox"] = Uniform(Vector3(90.f, 8.f, 90.f));
        m->uniforms["uTime"] = Uniform(0.f);
        m->vertexShader = vert;
        m->fragmentShader = kPuffFrag;
        m->transparent = true;
        m->depthWrite = false;
        m->fog = true;
        return m;
    }

    // Powder thrown up where a foot comes down: soft billows that spread and settle, and
    // clods that fly out and drop.
    struct Powder {
        static constexpr int N = 9000;
        struct P {
            Vector3 x, v;
            float age = 0, life = 0, s0 = 0, s1 = 0, a0 = 0, drag = 0, grav = 0;
        };
        std::vector<P> ps;
        int cursor = 0;
        std::shared_ptr<BufferGeometry> geo;
        std::shared_ptr<Points> points;
        std::mt19937 rng{99};

        explicit Powder(const std::shared_ptr<Material>& mat) {
            ps.resize(N);
            geo = BufferGeometry::create();
            geo->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(N * 3, 0.f), 3));
            geo->setAttribute("aSize", FloatBufferAttribute::create(std::vector<float>(N, 0.f), 1));
            geo->setAttribute("aAlpha", FloatBufferAttribute::create(std::vector<float>(N, 0.f), 1));
            points = Points::create(geo, mat);
            points->frustumCulled = false;
            points->renderOrder = 2;
        }

        float u() { return std::uniform_real_distribution<float>(0.f, 1.f)(rng); }

        void spawn(const P& p) {
            ps[cursor] = p;
            cursor = (cursor + 1) % N;
        }

        void burst(const Sole& sole, const Vector3& c, const Vector3& fwd, float scale = 1.f) {
            const Vector3 right(-fwd.z, 0.f, fwd.x);
            int made = 0;
            for (int tries = 0; tries < 4000 && made < static_cast<int>(420 * scale); ++tries) {
                const float lx = (u() * 2 - 1) * 2.7f, lz = (u() * 2 - 1) * 2.7f;
                const float sd = sole.sdf(lx, lz);
                if (sd < -0.12f || sd > 0.3f) continue;
                const Vector3 at = c + fwd * lx + right * lz;
                Vector3 out = unit(Vector3(at.x - c.x, 0.f, at.z - c.z));
                P p;
                p.x = at + Vector3(0.f, 0.08f, 0.f);
                const bool clod = made % 3 == 0;
                if (clod) {
                    p.v = out * (1.6f + 3.4f * u()) + Vector3(0.f, 1.8f + 2.6f * u(), 0.f);
                    p.life = 0.8f + 0.8f * u();
                    p.s0 = p.s1 = 0.06f + 0.08f * u();
                    p.a0 = 0.9f;
                    p.drag = 0.5f;
                    p.grav = 9.81f;
                } else {
                    p.v = out * (1.2f + 3.6f * u()) + Vector3(0.f, 0.5f + 2.2f * u(), 0.f);
                    p.life = 1.8f + 2.2f * u();
                    p.s0 = 0.3f + 0.3f * u();
                    p.s1 = 1.3f + 1.5f * u();
                    p.a0 = 0.3f + 0.3f * u();
                    p.drag = 1.7f;
                    p.grav = 0.5f;
                }
                spawn(p);
                ++made;
            }
        }

        // snow shaken off a foot as it lifts clear
        void shed(const Vector3& c) {
            for (int i = 0; i < 60; ++i) {
                P p;
                const float a = u() * kTau, r = 1.3f * std::sqrt(u());
                p.x = c + Vector3(r * std::cos(a), 0.1f + 0.3f * u(), r * std::sin(a));
                p.v = Vector3(0.3f * (u() - 0.5f), -0.2f * u(), 0.3f * (u() - 0.5f));
                p.life = 1.0f + u();
                p.s0 = p.s1 = 0.05f + 0.1f * u();
                p.a0 = 0.8f;
                p.drag = 0.6f;
                p.grav = 6.f;
                spawn(p);
            }
        }

        void update(float dt, const Vector3& wind) {
            auto& pa = geo->getAttribute<float>("position")->array();
            auto& sa = geo->getAttribute<float>("aSize")->array();
            auto& aa = geo->getAttribute<float>("aAlpha")->array();
            for (int i = 0; i < N; ++i) {
                P& p = ps[i];
                if (p.age >= p.life) {
                    sa[i] = 0;
                    aa[i] = 0;
                    continue;
                }
                p.age += dt;
                const float k = std::exp(-p.drag * dt);
                p.v = wind + (p.v - wind) * k;
                p.v.y -= p.grav * dt;
                p.x += p.v * dt;
                const float floor = groundHeight(p.x.x, p.x.z) - 0.25f;
                if (p.x.y < floor) {
                    p.x.y = floor;
                    p.v.multiplyScalar(0.2f);
                }
                const float f = clamp01(p.age / p.life);
                pa[3 * i] = p.x.x;
                pa[3 * i + 1] = p.x.y;
                pa[3 * i + 2] = p.x.z;
                sa[i] = mixf(p.s0, p.s1, std::sqrt(f));
                aa[i] = p.a0 * (1.f - f) * smoothstepf(0.f, 0.06f, f);
            }
            geo->getAttribute<float>("position")->needsUpdate();
            geo->getAttribute<float>("aSize")->needsUpdate();
            geo->getAttribute<float>("aAlpha")->needsUpdate();
        }
    };

    std::shared_ptr<Points> makeDrift(const std::shared_ptr<Material>& mat) {
        const int N = 5000;
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> u(0.f, 1.f);
        std::vector<float> pos, size, alpha, seed;
        for (int i = 0; i < N; ++i) {
            const bool wisp = i % 10 == 0;
            pos.insert(pos.end(), {(u(rng) - 0.5f) * 90.f, wisp ? 0.3f + 1.2f * u(rng) : 0.05f + 2.5f * std::pow(u(rng), 3.f), (u(rng) - 0.5f) * 90.f});
            size.push_back(wisp ? 2.5f + 4.f * u(rng) : 0.03f + 0.05f * u(rng));
            alpha.push_back(wisp ? 0.05f + 0.05f * u(rng) : 0.35f + 0.4f * u(rng));
            seed.push_back(u(rng));
        }
        auto g = BufferGeometry::create();
        g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        g->setAttribute("aSize", FloatBufferAttribute::create(size, 1));
        g->setAttribute("aAlpha", FloatBufferAttribute::create(alpha, 1));
        g->setAttribute("aSeed", FloatBufferAttribute::create(seed, 1));
        auto p = Points::create(g, mat);
        p->frustumCulled = false;
        p->renderOrder = 1;
        return p;
    }

    // ================================================================== the walker

    struct CirclePath {
        float R = 120.f, theta0 = 0.f, speed = 0.8f;// turns left (clockwise seen from above)

        [[nodiscard]] float theta(float t) const { return theta0 - speed * t / R; }
        void frame(float t, Vector3& pos, Vector3& fwd, Vector3& right) const {
            const float th = theta(t);
            pos.set(R * std::cos(th), 0.f, R * std::sin(th));
            fwd.set(std::sin(th), 0.f, -std::cos(th));
            right.set(std::cos(th), 0.f, std::sin(th));
        }
    };

    struct ContactSink {
        std::function<void(int leg, const Vector3& pos, const Vector3& fwd)> landed;
        std::function<void(int leg, const Vector3& pos)> lifted;
    };

    class Walker {
    public:
        struct Leg {
            std::shared_ptr<Group> coxa, femur, tibia, foot, barrelLift, rodLift, barrelKnee, rodKnee;
        };

        static constexpr float kPeriod = 9.f;// s per gait cycle
        static constexpr float kSwing = 0.17f;// fraction of the cycle a foot is in the air
        static constexpr float kLift = 1.6f;
        // leg order LF, RF, LH, RH; they lift in the order LH, LF, RH, RF
        static constexpr std::array<float, 4> kOffset{0.25f, 0.75f, 0.f, 0.5f};
        static constexpr std::array<int, 4> kOrder{2, 0, 3, 1};

        const WalkerKit& kit;
        CirclePath path;
        float phaseTime;
        std::shared_ptr<Group> root, body, sensor;
        std::array<Leg, 4> legs;
        Matrix4 bodyMatrix;
        Vector3 pathPos, fwd, right;
        float groundAvg = 0.f;
        std::array<Vector3, 4> footPos{};

        Walker(const WalkerKit& k, CirclePath p, float phase, bool shadows): kit(k), path(p), phaseTime(phase) {
            root = Group::create();
            body = instantiate(kit.hull, shadows);
            sensor = instantiate(kit.sensor, shadows);
            sensor->position.set(1.0f, 4.12f, -0.6f);
            body->add(sensor);
            root->add(body);
            for (auto& leg : legs) {
                leg.coxa = instantiate(kit.coxa, shadows);
                leg.femur = instantiate(kit.femur, shadows);
                leg.tibia = instantiate(kit.tibia, shadows);
                leg.foot = instantiate(kit.foot, shadows);
                leg.barrelLift = instantiate(kit.barrelLift, shadows);
                leg.rodLift = instantiate(kit.rodLift, shadows);
                leg.barrelKnee = instantiate(kit.barrelKnee, shadows);
                leg.rodKnee = instantiate(kit.rodKnee, shadows);
                for (auto* o : {&leg.coxa, &leg.femur, &leg.tibia, &leg.foot, &leg.barrelLift, &leg.rodLift, &leg.barrelKnee, &leg.rodKnee}) root->add(*o);
            }
        }

        // Where foot k is put down in cycle n: under its hip at the middle of the stance that follows.
        void target(int k, int n, Vector3& pos, Vector3& f) const {
            const float tau = (static_cast<float>(n) + kOffset[k] + kSwing + 0.5f * (1.f - kSwing)) * kPeriod;
            Vector3 p, fw, rt;
            path.frame(tau - phaseTime, p, fw, rt);
            pos = p + fw * kit.stance[k].x + rt * kit.stance[k].y;
            pos.y = groundHeight(pos.x, pos.z) - kit.sink;
            f = fw;
        }

        // Pose of foot k at time t: sole centre, its forward direction and toe pitch.
        void foot(int k, float t, Vector3& pos, Vector3& f, float& pitch, float& swingS) const {
            const float q = (t + phaseTime) / kPeriod - kOffset[k];
            const int n = static_cast<int>(std::floor(q));
            const float ph = q - static_cast<float>(n);
            Vector3 p1, f1;
            target(k, n, p1, f1);
            pitch = 0;
            swingS = -1;
            if (ph >= kSwing) {
                pos = p1;
                f = f1;
                return;
            }
            Vector3 p0, f0;
            target(k, n - 1, p0, f0);
            const float s = ph / kSwing;
            swingS = s;
            const float h = smoothstepf(0.12f, 0.88f, s);
            pos = lerpV(p0, p1, h);
            pos.y = mixf(p0.y, p1.y, h) + kLift * std::pow(std::sin(kPi * s), 0.85f);
            f = unit(lerpV(f0, f1, h));
            pitch = s < 0.5f ? deg(-11) * std::sin(kPi * s / 0.5f) : deg(7) * std::sin(kPi * (s - 0.5f) / 0.5f);
        }

        [[nodiscard]] Vector2 shiftFor(int k) const {
            Vector2 sum;
            for (const auto& s : kit.stance) {
                sum.x += s.x;
                sum.y += s.y;
            }
            const Vector2 c3((sum.x - kit.stance[k].x) / 3.f, (sum.y - kit.stance[k].y) / 3.f);
            return {(c3.x - sum.x / 4.f) * 0.24f, (c3.y - sum.y / 4.f) * 0.24f};
        }

        void update(float t) {
            path.frame(t, pathPos, fwd, right);

            // feet first: they are anchored in the world and the body hangs from them
            std::array<float, 4> pitch{}, swing{};
            std::array<Vector3, 4> ffwd{};
            groundAvg = 0;
            for (int k = 0; k < 4; ++k) {
                foot(k, t, footPos[k], ffwd[k], pitch[k], swing[k]);
                groundAvg += groundHeight(footPos[k].x, footPos[k].z) * 0.25f;
            }

            // weight shift: over the three standing feet before each lift, held through the
            // first part of the swing, then on toward the next
            const float cyc = fractf((t + phaseTime) / kPeriod);
            const int slot = std::min(3, static_cast<int>(cyc / 0.25f));
            const float u = (cyc - 0.25f * slot) / 0.25f;
            const int cur = kOrder[slot], next = kOrder[(slot + 1) % 4];
            const Vector2 s0 = shiftFor(cur), s1 = shiftFor(next);
            const float w = smoother((u - 0.42f) / 0.58f);
            const Vector2 shift(mixf(s0.x, s1.x, w), mixf(s0.y, s1.y, w));

            const float swingU = kSwing / 0.25f;
            float dy = 0, pitchB = 0, rollB = 0;
            if (u < swingU) {
                const float a = std::sin(kPi * u / swingU);
                dy -= 0.10f * a;
                pitchB += (cur < 2 ? -0.012f : 0.010f) * a;
                rollB += (cur % 2 == 0 ? 0.010f : -0.010f) * a;
            } else {
                const float since = (u - swingU) * 0.25f * kPeriod;
                dy -= 0.07f * std::exp(-since * 4.f) * std::sin(since * 13.f);
            }
            rollB -= 0.018f * shift.y;
            pitchB += 0.012f * shift.x;

            Vector3 P = pathPos + fwd * shift.x + right * shift.y;
            P.y = groundAvg + kit.bodyHeight + dy;
            bodyMatrix = mul(mul(basisAt(P, fwd, kY, right), rotation(kZ, pitchB)), rotation(kX, rollB));
            place(*body, bodyMatrix);
            sensor->rotation.y = t * 0.9f;

            Matrix4 inv = bodyMatrix;
            inv.invert();
            for (int k = 0; k < 4; ++k) solveLeg(k, inv, footPos[k], ffwd[k], pitch[k]);
        }

        // Contacts between t0 and t1 (foot meets the snow / leaves it).
        void contacts(float t0, float t1, const ContactSink& sink) const {
            for (int k = 0; k < 4; ++k) {
                const float a = (t0 + phaseTime) / kPeriod - kOffset[k], b = (t1 + phaseTime) / kPeriod - kOffset[k];
                for (int n = static_cast<int>(std::floor(a - 0.95f * kSwing)) + 1; static_cast<float>(n) + 0.95f * kSwing <= b; ++n) {
                    if (static_cast<float>(n) + 0.95f * kSwing <= a) continue;
                    Vector3 p, f;
                    target(k, n, p, f);
                    if (sink.landed) sink.landed(k, p, f);
                }
                for (int n = static_cast<int>(std::floor(a - 0.06f * kSwing)) + 1; static_cast<float>(n) + 0.06f * kSwing <= b; ++n) {
                    if (static_cast<float>(n) + 0.06f * kSwing <= a) continue;
                    Vector3 p, f;
                    target(k, n - 1, p, f);
                    if (sink.lifted) sink.lifted(k, p);
                }
            }
        }

        [[nodiscard]] Vector3 bodyCentre() const {
            return Vector3().setFromMatrixPosition(bodyMatrix);
        }

    private:
        void solveLeg(int k, const Matrix4& inv, const Vector3& footW, const Vector3& footFwd, float pitch) {
            Leg& leg = legs[k];
            const Vector3 fRight(-footFwd.z, 0.f, footFwd.x);
            const Matrix4 footM = mul(basisAt(footW, footFwd, kY, fRight), rotation(kZ, pitch));
            place(*leg.foot, footM);

            const Vector3 ankleW = Vector3(0.f, kit.ankle, 0.f).applyMatrix4(footM);
            const Vector3 A = Vector3(ankleW).applyMatrix4(inv);
            const Vector3 hip = kit.hip[k];
            Vector3 d(A.x - hip.x, 0.f, A.z - hip.z);
            d.normalize();
            const Vector3 P = hip + d * kit.coxa_;
            const Vector3 rel = A - P;
            const float rho = rel.x * d.x + rel.z * d.z, eta = rel.y;
            const float L1 = kit.femur_, L2 = kit.tibia_;
            const float D = std::clamp(std::sqrt(rho * rho + eta * eta), std::abs(L1 - L2) + 0.01f, L1 + L2 - 0.01f);
            const float th1 = std::atan2(eta, rho) + std::acos(std::clamp((L1 * L1 + D * D - L2 * L2) / (2.f * L1 * D), -1.f, 1.f));
            const Vector3 xf = d * std::cos(th1) + kY * std::sin(th1);
            const Vector3 yf = d * -std::sin(th1) + kY * std::cos(th1);
            const Vector3 z = cross(d, kY);
            const Vector3 K = P + xf * L1;
            const Vector3 xt = unit(A - K);
            const Vector3 yt = cross(z, xt);

            const Matrix4 coxaB = basisAt(hip, d, kY, z);
            const Matrix4 femurB = basisAt(P, xf, yf, z);
            const Matrix4 tibiaB = basisAt(K, xt, yt, z);
            place(*leg.coxa, mul(bodyMatrix, coxaB));
            place(*leg.femur, mul(bodyMatrix, femurB));
            place(*leg.tibia, mul(bodyMatrix, tibiaB));

            auto ram = [&](Group& barrel, Group& rod, const Vector3& a, const Vector3& b, float Lb, float Lr) {
                const float span = a.distanceTo(b);
                static bool warned = false;
                if (!warned && (span < Lr + 0.25f || span > Lb + Lr - 0.1f)) {
                    warned = true;
                    std::cerr << "glacier_walker: a hydraulic ram is out of its stroke (span " << span << " m)" << std::endl;
                }
                const Vector3 x = unit(b - a);
                const Vector3 y = unit(cross(z, x));
                const Vector3 zz = cross(x, y);
                place(barrel, mul(bodyMatrix, basisAt(a, x, y, zz)));
                place(rod, mul(bodyMatrix, basisAt(b, neg(x), neg(y), zz)));
            };
            ram(*leg.barrelLift, *leg.rodLift, Vector3(kit.coxaLug).applyMatrix4(coxaB), Vector3(kit.femurLiftLug).applyMatrix4(femurB), kit.liftBarrel, kit.liftRod);
            ram(*leg.barrelKnee, *leg.rodKnee, Vector3(kit.femurKneeLug).applyMatrix4(femurB), Vector3(kit.tibiaKneeLug).applyMatrix4(tibiaB), kit.kneeBarrel, kit.kneeRod);
        }
    };

    // ================================================================== crew

    struct CrewKit {
        Part body, leg, arm, pole;
    };

    CrewKit buildCrewKit(const Mats& m) {
        CrewKit k;
        {
            PartBuilder pb;
            pb.push(scaling(1.f, 1.f, 0.78f));
            pb.on(m.parka).lathe({{0.f, 0.78f}, {0.25f, 0.78f}, {0.27f, 0.95f}, {0.25f, 1.25f}, {0.22f, 1.42f}, {0.12f, 1.5f}, {0.f, 1.5f}}, 18, 0.12f);
            pb.pop();
            pb.push(translation(Vector3(-0.01f, 1.63f, 0.f)));
            pb.on(m.parka).lathe(sphereProfile(0.155f, 10), 18, 0.1f);
            pb.on(m.lens).lathe({{0.158f, -0.01f}, {0.158f, 0.05f}}, 12, 0.1f, deg(-55), deg(55));
            pb.pop();
            {
                std::vector<Vector3> ring;
                for (int i = 0; i <= 16; ++i) {
                    const float a = kTau * i / 16;
                    ring.emplace_back(0.12f, 1.63f + 0.13f * std::sin(a), 0.11f * std::cos(a));
                }
                pb.on(m.fur).tube(ring, 0.035f, 8);
            }
            pb.on(m.pack).box(Vector3(-0.27f, 1.18f, 0.f), Vector3(0.09f, 0.21f, 0.17f), 0.1f);
            pb.on(m.pack).box(Vector3(-0.2f, 0.88f, 0.f), Vector3(0.12f, 0.03f, 0.2f), 0.1f);
            k.body = pb.build(0.03f, 0.4f, 0.8f);
        }
        {
            PartBuilder pb;// hip at the origin, hanging down -Y
            pb.on(m.pants).lathe({{0.f, -0.8f}, {0.075f, -0.8f}, {0.11f, -0.05f}, {0.f, 0.f}}, 12, 0.12f);
            pb.on(m.boot).box(Vector3(0.05f, -0.86f, 0.f), Vector3(0.14f, 0.07f, 0.065f), 0.1f);
            k.leg = pb.build(0.03f, 0.3f, 0.6f);
        }
        {
            PartBuilder pb;// shoulder at the origin
            pb.on(m.parka).lathe({{0.f, -0.58f}, {0.06f, -0.58f}, {0.08f, -0.05f}, {0.f, 0.03f}}, 12, 0.12f);
            pb.push(translation(Vector3(0.f, -0.64f, 0.f)));
            pb.on(m.mitt).lathe(sphereProfile(0.065f, 6), 10, 0.1f);
            pb.pop();
            k.arm = pb.build(0.03f, 0.3f, 0.6f);
        }
        {
            PartBuilder pb;
            for (int i = 0; i < 8; ++i)
                pb.on(i % 2 ? m.white : m.orange).lathe(rodProfile(0.018f, -1.1f + i * 0.28f, -0.82f + i * 0.28f), 8, 0.3f);
            k.pole = pb.build(0.02f, 0.1f, 0.2f);
        }
        return k;
    }

    struct CrewMember {
        std::shared_ptr<Group> root, legL, legR, armL, armR;
        float ahead = 0, lateral = 0, stride = 0.68f, phase = 0;
        bool pole = false;
        int lastStep = 0;
    };

    // ================================================================== hoppers

    Part buildHopper(const Mats& m) {
        PartBuilder pb;
        // fuselage pod along +X
        {
            std::vector<Vector2> p;
            for (int i = 0; i <= 16; ++i) {
                const float t = static_cast<float>(i) / 16;
                const float x = -2.9f + 5.6f * t;
                const float r = 0.82f * std::pow(std::max(0.f, std::sin(kPi * std::pow(t, 0.85f))), 0.6f);
                p.emplace_back(std::max(r, 1e-3f), x);
            }
            pb.push(alongX());
            pb.on(m.white).lathe(p, 22, 0.25f);
            pb.on(m.orange).lathe({{0.835f, 0.25f}, {0.835f, 0.55f}}, 22, 0.25f);
            pb.pop();
        }
        pb.push(mul(translation(Vector3(1.0f, 0.42f, 0.f)), scaling(1.5f, 0.8f, 1.f)));
        pb.on(m.glass).lathe(sphereProfile(0.58f, 10), 20, 0.2f);
        pb.pop();
        // stub wings and ducts
        pb.on(m.white).box(Vector3(-0.2f, 0.05f, 0.f), Vector3(0.55f, 0.07f, 2.5f), 0.3f);
        for (float side : {-1.f, 1.f}) {
            pb.push(mul(translation(Vector3(-0.2f, 0.05f, side * 3.1f)), rotation(kZ, -deg(70))));
            pb.on(m.graphite).lathe({{1.0f, -0.35f}, {1.12f, -0.2f}, {1.12f, 0.3f}, {1.0f, 0.38f}, {0.95f, 0.3f}, {0.95f, -0.3f}, {1.0f, -0.35f}}, 32, 0.2f);
            pb.on(m.orange).lathe({{1.125f, -0.05f}, {1.125f, 0.12f}}, 32, 0.2f);
            pb.on(m.steel).lathe({{0.f, -0.2f}, {0.2f, -0.2f}, {0.2f, 0.15f}, {0.f, 0.25f}}, 14, 0.2f);
            pb.pop();
        }
        for (float side : {-1.f, 1.f})
            pb.on(m.white).beam(Vector3(-2.3f, 0.2f, 0.f), Vector3(-2.75f, 1.2f, side * 0.9f), 0.06f, 0.7f, kX);
        for (float side : {-1.f, 1.f}) {
            pb.on(m.graphite).beam(Vector3(-1.3f, -1.15f, side * 0.7f), Vector3(1.4f, -1.15f, side * 0.7f), 0.08f, 0.08f);
            pb.on(m.graphite).beam(Vector3(-0.8f, -0.5f, side * 0.45f), Vector3(-0.8f, -1.15f, side * 0.7f), 0.06f, 0.06f);
            pb.on(m.graphite).beam(Vector3(0.9f, -0.55f, side * 0.45f), Vector3(0.9f, -1.15f, side * 0.7f), 0.06f, 0.06f);
        }
        return pb.build(0.06f, 0.8f, 0.8f);
    }

    Part buildRotor(const Mats& m) {
        PartBuilder pb;// spins about local +Y
        for (int b = 0; b < 5; ++b) {
            pb.push(rotation(kY, kTau * b / 5));
            pb.push(rotation(kX, deg(22)));
            pb.on(m.rubber).box(Vector3(0.f, 0.f, 0.55f), Vector3(0.09f, 0.012f, 0.4f), 0.2f);
            pb.pop();
            pb.pop();
        }
        return pb.build(0.03f, 0.1f, 0.2f);
    }

    struct Hopper {
        std::shared_ptr<Group> root;
        std::array<std::shared_ptr<Group>, 2> rotors;
        float lag = 0, offset = 0, lift = 0;
    };

    // ================================================================== camera director

    struct Shot {
        Vector3 pos, target;
    };

    class Director {
    public:
        enum class Mode { Tracking,
                          Orbit };
        Mode mode = Mode::Tracking;

        Director(PerspectiveCamera& cam, OrbitControls& oc): camera(cam), controls(oc) {}

        static Shot tracking(float t, const Walker& w, float aspect, float vfovDeg) {
            const float az = deg(38) + deg(12) * std::sin(t * 0.045f);
            const float halfV = deg(vfovDeg * 0.5f);
            const float halfH = std::atan(std::tan(halfV) * aspect);
            const float dist = std::max(31.f, 12.5f / std::sin(std::min(halfV, halfH)) * 0.92f);
            Vector3 c = w.pathPos + w.fwd * 1.0f;
            c.y = w.groundAvg + 5.6f;
            const Vector3 dir = w.fwd * std::cos(az) - w.right * std::sin(az);
            Shot s;
            s.pos = c + dir * dist;
            s.pos.y = groundHeight(s.pos.x, s.pos.z) + 1.7f + 0.035f * dist;
            s.target = c + Vector3(0.f, 0.6f, 0.f);
            return s;
        }

        void toTracking() {
            if (mode == Mode::Tracking && !returning) return;
            returning = true;
            retT = 0;
            retPos = camera.position;
            retTarget = controls.target;
        }

        void orbitFrom(Object3D* anchor, const Vector3& worldPoint) {
            mode = Mode::Orbit;
            returning = false;
            flying = false;
            setAnchor(anchor, worldPoint);
            controls.target.copy(worldPoint);
        }

        void focus(Object3D* anchor, const Vector3& worldPoint, float dist) {
            if (mode == Mode::Tracking) {
                mode = Mode::Orbit;
                returning = false;
            }
            setAnchor(anchor, worldPoint);
            flying = true;
            flyT = 0;
            flyTarget0 = controls.target;
            flyDist0 = camera.position.distanceTo(controls.target);
            flyDist1 = dist;
        }

        // Place an inspection camera directly (headless views).
        void inspect(Object3D* anchor, const Vector3& worldPoint, float dist, float azDeg, float elDeg, const Vector3& fwd, const Vector3& right) {
            mode = Mode::Orbit;
            setAnchor(anchor, worldPoint);
            controls.target.copy(worldPoint);
            const float az = deg(azDeg), el = deg(elDeg);
            const Vector3 dir = (fwd * std::cos(az) - right * std::sin(az)) * std::cos(el) + kY * std::sin(el);
            camera.position.copy(worldPoint + dir * dist);
            camera.lookAt(worldPoint);
            controls.update();
        }

        void update(float t, float dt, const Walker& lead, float aspect) {
            if (mode == Mode::Tracking || returning) {
                const Shot s = tracking(t, lead, aspect, camera.fov);
                if (returning) {
                    retT += dt / 2.0f;
                    const float k = smoother(retT);
                    camera.position.copy(lerpV(retPos, s.pos, k));
                    controls.target.copy(lerpV(retTarget, s.target, k));
                    if (retT >= 1.f) {
                        returning = false;
                        mode = Mode::Tracking;
                    }
                } else {
                    camera.position.copy(s.pos);
                    controls.target.copy(s.target);
                }
                camera.lookAt(controls.target);
                hasLast = false;
                return;
            }
            const Vector3 a = anchorWorld();
            if (hasLast) {
                const Vector3 d = a - lastAnchor;
                camera.position += d;
                controls.target += d;
            }
            lastAnchor = a;
            hasLast = true;
            if (flying) {
                flyT = std::min(1.f, flyT + dt / 1.4f);
                const float k = smoother(flyT);
                const Vector3 dir = unit(camera.position - controls.target);
                controls.target.copy(lerpV(flyTarget0, a, k));
                camera.position.copy(controls.target + dir * mixf(flyDist0, flyDist1, k));
                if (flyT >= 1.f) flying = false;
            }
            controls.update();
            const float floor = groundHeight(camera.position.x, camera.position.z) + 0.35f;
            if (camera.position.y < floor) camera.position.y = floor;
        }

    private:
        PerspectiveCamera& camera;
        OrbitControls& controls;
        Object3D* anchorObj = nullptr;
        Vector3 anchorLocal, anchorFixed, lastAnchor, flyTarget0, retPos, retTarget;
        bool hasLast = false, flying = false, returning = false;
        float flyT = 0, flyDist0 = 0, flyDist1 = 0, retT = 0;

        void setAnchor(Object3D* o, const Vector3& p) {
            anchorObj = o;
            anchorFixed = p;
            if (o) {
                o->updateMatrixWorld(true);
                anchorLocal = p;
                o->worldToLocal(anchorLocal);
            }
            hasLast = false;
        }

        [[nodiscard]] Vector3 anchorWorld() const {
            if (!anchorObj) return anchorFixed;
            Vector3 p = anchorLocal;
            anchorObj->localToWorld(p);
            return p;
        }
    };

    struct InputListener: MouseListener {
        std::function<void(int, const Vector2&)> down, up;
        std::function<void(const Vector2&)> wheel;
        void onMouseDown(int b, const Vector2& p) override {
            if (down) down(b, p);
        }
        void onMouseUp(int b, const Vector2& p) override {
            if (up) up(b, p);
        }
        void onMouseWheel(const Vector2& d) override {
            if (wheel) wheel(d);
        }
    };

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, view = "opening";
    float startTime = 14.f;
    int width = 1600, height = 900;
    std::optional<Vector2> clickAt;
    int benchFrames = 0;
    std::string filmDir;
    float filmSeconds = 20.f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--time" && i + 1 < argc) startTime = std::stof(argv[++i]);
        else if (a == "--view" && i + 1 < argc) view = argv[++i];
        else if (a == "--bench" && i + 1 < argc) benchFrames = std::max(1, std::atoi(argv[++i]));
        else if (a == "--film" && i + 1 < argc) filmDir = argv[++i];
        else if (a == "--seconds" && i + 1 < argc) filmSeconds = std::stof(argv[++i]);
        else if (a == "--click" && i + 1 < argc) {
            float cx = 0, cy = 0;
            if (std::sscanf(argv[++i], "%f,%f", &cx, &cy) == 2) clickAt = Vector2(cx, cy);
        }
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
    }
    const bool headless = !shotPath.empty() || !filmDir.empty();

    Canvas canvas(Canvas::Parameters()
                          .title("Glacier walker")
                          .size(width, height)
                          .antialiasing(8)
                          .vsync(!headless)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.1f;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    // light: a cold overcast day with the sun low in a break to the south-west
    const Vector3 sunDir = unit(Vector3(0.93f, 0.26f, -0.25f));
    const Color sunColor(1.0f, 0.74f, 0.50f);
    const Color horizon(0.62f, 0.68f, 0.76f);
    const Color skyFill(0.42f, 0.52f, 0.66f);

    auto scene = Scene::create();
    scene->background = horizon;
    scene->fog = Fog(horizon, 80.f, 5200.f);

    auto camera = PerspectiveCamera::create(38, canvas.aspect(), 0.1f, 20000.f);
    OrbitControls controls{*camera, canvas};
    controls.enableDamping = true;
    controls.dampingFactor = 0.08f;
    controls.minDistance = 1.5f;
    controls.maxDistance = 900.f;
    controls.maxPolarAngle = deg(92);

    // sky dome (and the same sky, small, baked into the reflection environment)
    const Color skySun(1.0f, 0.64f, 0.28f);
    auto skyMat = makeSkyMaterial(sunDir, skySun, horizon);
    auto sky = Mesh::create(SphereGeometry::create(15000, 64, 32), skyMat);
    sky->frustumCulled = false;
    sky->renderOrder = -10;
    scene->add(sky);

    auto sun = DirectionalLight::create(sunColor, 7.5f);
    sun->castShadow = true;
    sun->shadow->mapSize.set(4096, 4096);
    sun->shadow->bias = -0.0004f;
    sun->shadow->normalBias = 0.04f;
    sun->shadow->radius = 2.f;
    const float shadowHalf = 32.f;
    if (auto* sc = sun->shadow->camera->as<OrthographicCamera>()) {
        sc->left = -shadowHalf;
        sc->right = shadowHalf;
        sc->top = shadowHalf;
        sc->bottom = -shadowHalf;
        sc->nearPlane = 1.f;
        sc->farPlane = 500.f;
        sc->updateProjectionMatrix();
    }
    auto sunTarget = Object3D::create();
    scene->add(sunTarget);
    sun->setTarget(*sunTarget);
    scene->add(sun);
    scene->add(HemisphereLight::create(skyFill, Color(0.80f, 0.83f, 0.88f), 1.3f));

    // the glacier, its footprint record, and the ranges
    const CirclePath leadPath{kRingR, 0.f, 0.8f};
    TrackMap track(leadPath.R - 16.f, leadPath.R + 16.f);
    auto snow = Mesh::create(buildSnowGeometry(track.rIn, track.rOut), makeSnowMaterial(track));
    snow->receiveShadow = true;
    scene->add(snow);

    // the big massif stands behind the opening shot, a little right of the break
    Vector3 hero;
    {
        Vector3 p, f, r;
        leadPath.frame(10.f, p, f, r);
        const float az = deg(38) + deg(12) * std::sin(0.45f);
        const Vector3 look = unit(neg(f * std::cos(az) - r * std::sin(az)));
        const Vector3 rightOfLook = unit(cross(look, kY));
        hero = p + unit(look * std::cos(deg(14)) + rightOfLook * std::sin(deg(14))) * 5600.f;
    }
    terrain::MountainBackdropOptions ranges;
    ranges.ground = groundHeight;
    ranges.massifs.push_back({Vector2(hero.x, hero.z), 1600.f, 1100.f});
    ranges.hazeColor = horizon;
    ranges.hazeSunColor = skySun;
    ranges.sunDirection = sunDir;
    auto mountains = terrain::MountainBackdrop::create(ranges);
    scene->add(mountains);

    // reflections: the same sky, the ranges and the glacier, lit by the same lights
    {
        auto envScene = Scene::create();
        envScene->add(Mesh::create(SphereGeometry::create(11000, 64, 32), makeSkyMaterial(sunDir, skySun, horizon)));
        envScene->add(Mesh::create(mountains->mesh()->geometry(), mountains->mesh()->material()));
        auto iceMat = MeshStandardMaterial::create();
        iceMat->color = Color(0.86f, 0.89f, 0.94f);
        iceMat->roughness = 0.9f;
        auto ice = Mesh::create(CircleGeometry::create(2600, 96), iceMat);
        ice->rotation.x = -kPi / 2;
        envScene->add(ice);
        auto envSun = DirectionalLight::create(sunColor, 7.5f);
        envSun->position.copy(sunDir * 1000.f);
        envScene->add(envSun);
        envScene->add(HemisphereLight::create(skyFill, Color(0.80f, 0.83f, 0.88f), 1.3f));
        PMREMGenerator pmrem(renderer);
        PMREMGenerator::Options opt;
        opt.position = Vector3(0.f, 14.f, 0.f);
        scene->environment = pmrem.fromScene(*envScene, 0.f, 1.f, 14000.f, opt);
    }
    scene->environmentIntensity = 0.6f;

    // the machines
    const Mats mats = makeMaterials();
    const WalkerKit kit = buildWalkerKit(mats);
    Walker lead(kit, leadPath, 0.f, true);
    scene->add(lead.root);

    // the other two walk in the haze beyond the lead, on wider circles, out of step with it
    std::vector<std::unique_ptr<Walker>> others;
    {
        lead.update(0.f);
        const Shot s = Director::tracking(0.f, lead, 16.f / 9.f, 38.f);
        const Vector3 look = unit(Vector3(s.target.x - s.pos.x, 0.f, s.target.z - s.pos.z));
        const Vector3 side(-look.z, 0.f, look.x);
        const std::array<Vector3, 2> spots{lead.pathPos + look * 640.f - side * 170.f, lead.pathPos + look * 960.f + side * 150.f};
        const std::array<float, 2> phases{3.1f, 6.4f};
        for (int i = 0; i < 2; ++i) {
            CirclePath p;
            p.R = std::hypot(spots[i].x, spots[i].z);
            p.theta0 = std::atan2(spots[i].z, spots[i].x);
            p.speed = 0.8f;
            others.push_back(std::make_unique<Walker>(kit, p, phases[i], true));
            scene->add(others.back()->root);
        }
    }

    // crew walking with the lead machine: two under its belly, the rest off its left flank
    const CrewKit crewKit = buildCrewKit(mats);
    std::vector<CrewMember> crew;
    {
        const std::array<std::array<float, 3>, 6> spots{{{2.5f, -1.4f, 0.1f}, {-1.0f, 1.6f, 0.6f}, {7.f, -13.5f, 0.3f},
                                                        {3.f, -15.2f, 0.8f}, {-3.5f, -12.6f, 0.45f}, {1.f, 13.8f, 0.15f}}};
        for (size_t i = 0; i < spots.size(); ++i) {
            CrewMember c;
            c.ahead = spots[i][0];
            c.lateral = spots[i][1];
            c.phase = spots[i][2] * kTau;
            c.stride = 0.64f + 0.06f * static_cast<float>(i % 3);
            c.pole = i == 3;
            c.root = instantiate(crewKit.body, true);
            c.legL = instantiate(crewKit.leg, true);
            c.legR = instantiate(crewKit.leg, true);
            c.armL = instantiate(crewKit.arm, true);
            c.armR = instantiate(crewKit.arm, true);
            c.legL->position.set(0.f, 0.92f, -0.1f);
            c.legR->position.set(0.f, 0.92f, 0.1f);
            c.armL->position.set(0.f, 1.42f, -0.26f);
            c.armR->position.set(0.f, 1.42f, 0.26f);
            for (auto* o : {&c.legL, &c.legR, &c.armL, &c.armR}) c.root->add(*o);
            if (c.pole) {
                auto pole = instantiate(crewKit.pole, true);
                pole->position.set(0.05f, -0.62f, 0.f);
                c.armR->add(pole);
            }
            scene->add(c.root);
            crew.push_back(c);
        }
    }

    // two hoppers banking round the column
    const Part hopperPart = buildHopper(mats);
    const Part rotorPart = buildRotor(mats);
    std::array<Hopper, 2> hoppers;
    for (int i = 0; i < 2; ++i) {
        hoppers[i].root = instantiate(hopperPart, true);
        for (int s = 0; s < 2; ++s) {
            auto r = instantiate(rotorPart, false);
            auto mount = Group::create();
            place(*mount, mul(translation(Vector3(-0.2f, 0.05f, (s ? 1.f : -1.f) * 3.1f)), rotation(kZ, -deg(70))));
            mount->add(r);
            hoppers[i].root->add(mount);
            hoppers[i].rotors[s] = r;
        }
        hoppers[i].lag = i * 1.4f;
        hoppers[i].offset = i * 14.f;
        hoppers[i].lift = i * 4.f;
        scene->add(hoppers[i].root);
    }
    auto hopperFrame = [&](const Hopper& h, float t, Vector3& p, Vector3& vel) {
        auto P = [&](float tt) {
            Vector3 lp, lf, lr;
            leadPath.frame(tt, lp, lf, lr);
            const float w = kTau / 40.f;
            const float a = w * tt + 2.2f;
            Vector3 q = lp + lf * (30.f + 260.f * std::cos(a)) - lr * (140.f * std::sin(a));
            q.y = groundHeight(q.x, q.z) + 20.f + 7.f * std::sin(0.5f * a) + h.lift;
            return q;
        };
        const float tt = t - h.lag, e = 0.05f;
        p = P(tt);
        vel = (P(tt + e) - P(tt - e)) * (0.5f / e);
        const Vector3 acc = (P(tt + e) - p * 2.f + P(tt - e)) * (1.f / (e * e));
        const Vector3 side = unit(cross(vel, kY));
        p += side * h.offset;
        return acc;
    };

    // powder and drifting snow
    auto puffMat = makePuffMaterial(kPuffVert, sunDir, sunColor, skyFill);
    auto driftMat = makePuffMaterial(kDriftVert, sunDir, sunColor, skyFill);
    Powder powder(puffMat);
    scene->add(powder.points);
    auto drift = makeDrift(driftMat);
    scene->add(drift);

    // ---- simulation -----------------------------------------------------------------------

    int stampSeed = 0;
    bool withParticles = true;
    ContactSink sink;
    sink.landed = [&](int, const Vector3& p, const Vector3& f) {
        track.stampFoot(kit.sole, p, f, kit.sink, ++stampSeed);
        if (withParticles) powder.burst(kit.sole, Vector3(p.x, p.y + kit.sink, p.z), f);
    };
    sink.lifted = [&](int, const Vector3& p) {
        if (withParticles) powder.shed(Vector3(p.x, p.y + kit.sink + 0.3f, p.z));
    };

    auto updateCrew = [&](float t, bool prints) {
        for (auto& c : crew) {
            const float R = leadPath.R + c.lateral;
            const float th = leadPath.theta(t) - c.ahead / leadPath.R;
            const Vector3 f(std::sin(th), 0.f, -std::cos(th)), r(std::cos(th), 0.f, std::sin(th));
            const float dist = leadPath.speed * t * R / leadPath.R;
            const float ph = dist / c.stride * kPi + c.phase;
            Vector3 p(R * std::cos(th), 0.f, R * std::sin(th));
            p.y = groundHeight(p.x, p.z) - 0.04f + 0.03f * std::abs(std::cos(ph));
            place(*c.root, basisAt(p, f, kY, r));
            const float sw = std::sin(ph);
            c.legL->rotation.z = 0.42f * sw;
            c.legR->rotation.z = -0.42f * sw;
            c.armL->rotation.z = -0.3f * sw;
            c.armR->rotation.z = c.pole ? 0.25f - 0.1f * sw : 0.3f * sw;
            const int step = static_cast<int>(std::floor((ph - kPi / 2) / kPi));
            if (prints && step != c.lastStep) {
                const float side = (step & 1) ? 0.13f : -0.13f;
                track.stampBoot(p + f * 0.3f + r * side, f);
            }
            c.lastStep = step;
        }
    };

    auto updateHoppers = [&](float t) {
        for (auto& h : hoppers) {
            Vector3 p, v;
            const Vector3 acc = hopperFrame(h, t, p, v);
            const Vector3 x = unit(v);
            const Vector3 lat = acc - x * acc.dot(x);
            const Vector3 up = unit(kY * 9.81f + Vector3(lat.x, 0.f, lat.z));
            const Vector3 z = unit(cross(x, up));
            const Vector3 y = cross(z, x);
            place(*h.root, basisAt(p, x, y, z));
            for (auto& r : h.rotors) r->rotation.y = t * 55.f;
        }
    };

    float simTime = -60.f;
    {
        // the column has been walking a while: lay down the trail behind it
        withParticles = false;
        for (float t = -60.f; t < 0.f; t += 1.f / 15.f) {
            lead.contacts(t, t + 1.f / 15.f, sink);
            updateCrew(t + 1.f / 15.f, true);
        }
        withParticles = true;
        simTime = 0.f;
    }

    float lastUpload = -1.f;
    auto step = [&](float t, float dt) {
        lead.update(t);
        for (auto& w : others) w->update(t);
        lead.contacts(simTime, t, sink);
        updateCrew(t, true);
        updateHoppers(t);
        powder.update(dt, Vector3(2.2f, 0.f, 1.0f));
        simTime = t;
    };

    auto frameUpdate = [&](float t) {
        if (track.dirty && (t - lastUpload > 0.25f || headless)) {
            track.tex->needsUpdate();
            track.dirty = false;
            lastUpload = t;
        }
        // keep the sun's shadow box on the lead walker, snapped to whole shadow texels
        const Vector3 focus = lead.pathPos + Vector3(0.f, lead.groundAvg, 0.f);
        const Vector3 lz = neg(sunDir);
        const Vector3 lx = unit(cross(kY, lz));
        const Vector3 ly = cross(lz, lx);
        const float texel = 2.f * shadowHalf / 4096.f;
        const Vector3 snapped = lx * (std::round(focus.dot(lx) / texel) * texel) + ly * (std::round(focus.dot(ly) / texel) * texel) + lz * focus.dot(lz);
        sunTarget->position.copy(snapped);
        sun->position.copy(snapped + sunDir * 250.f);

        skyMat->uniforms["uTime"].setValue(t);
        sky->position.copy(camera->position);
        const float pxScale = static_cast<float>(canvas.size().height()) / (2.f * std::tan(deg(camera->fov * 0.5f)));
        puffMat->uniforms["uScale"].setValue(pxScale);
        driftMat->uniforms["uScale"].setValue(pxScale);
        driftMat->uniforms["uTime"].setValue(t);
        Vector3 origin = controls.target;
        origin.y = groundHeight(origin.x, origin.z);
        driftMat->uniforms["uOrigin"].setValue(origin);
    };

    Director director(*camera, controls);

    // ---- input ----------------------------------------------------------------------------

    std::vector<Object3D*> pickables{lead.root.get()};
    for (auto& w : others) pickables.push_back(w->root.get());
    for (auto& c : crew) pickables.push_back(c.root.get());
    for (auto& h : hoppers) pickables.push_back(h.root.get());

    Raycaster raycaster;
    Vector2 downPos;
    Clock pressClock;
    InputListener input;
    input.down = [&](int, const Vector2& p) {
        downPos = p;
        pressClock.start();
        if (director.mode == Director::Mode::Tracking) director.orbitFrom(lead.body.get(), controls.target);
    };
    input.wheel = [&](const Vector2&) {
        if (director.mode == Director::Mode::Tracking) director.orbitFrom(lead.body.get(), controls.target);
    };
    // A click: fly to whatever is under the cursor and follow it; the sky returns to the tracking shot.
    auto handleClick = [&](const Vector2& p) {
        const auto size = canvas.size();
        const Vector2 ndc(p.x / static_cast<float>(size.width()) * 2.f - 1.f, -(p.y / static_cast<float>(size.height())) * 2.f + 1.f);
        raycaster.setFromCamera(ndc, *camera);
        const auto hits = raycaster.intersectObjects(pickables, true);
        if (!hits.empty()) {
            Object3D* o = hits.front().object;
            float r = 2.f;
            if (auto* mesh = o->as<Mesh>()) {
                if (auto geo = mesh->geometry()) {
                    if (!geo->boundingSphere) geo->computeBoundingSphere();
                    r = geo->boundingSphere->radius;
                }
            }
            director.focus(o, hits.front().point, std::clamp(r * 1.6f, 3.f, 12.f));
            return;
        }
        // the snow: march the ray down to the surface
        const Ray& ray = raycaster.ray;
        if (ray.direction.y < -0.002f) {
            float tHit = (groundHeight(ray.origin.x, ray.origin.z) - ray.origin.y) / ray.direction.y;
            for (int it = 0; it < 4; ++it) {
                const Vector3 q = ray.origin + ray.direction * tHit;
                tHit = (groundHeight(q.x, q.z) - ray.origin.y) / ray.direction.y;
            }
            if (tHit > 0 && tHit < 600.f) {
                director.focus(nullptr, ray.origin + ray.direction * tHit, 9.f);
                return;
            }
        }
        director.toTracking();
    };
    input.up = [&](int button, const Vector2& p) {
        if (button != 0 || p.distanceTo(downPos) > 5.f || pressClock.getElapsedTime() > 0.35f) return;
        handleClick(p);
    };
    canvas.addMouseListener(input);

    bool paused = false;
    KeyAdapter keys(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent e) {
        if (e.key == Key::SPACE) paused = !paused;
        if (e.key == Key::C) director.toTracking();
    });
    canvas.addKeyListener(keys);

    canvas.onWindowResize([&](WindowSize s) {
        camera->aspect = s.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(s);
    });

    // ---- film: three shots, one PNG per frame at 30 fps ---------------------------------------
    //   0-7 s    the low three-quarter tracking shot
    //   7-13 s   a low dolly beside the left legs: a hind foot lifts, a fore foot comes down
    //   13-20 s  a crane back and up from behind: the trail of prints, the crew, the column

    if (!filmDir.empty()) {
        const float fps = 30.f, dt = 1.f / fps;
        const int total = static_cast<int>(std::lround(filmSeconds * fps));
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        int frame = 0;
        canvas.animate([&] {
            const float t = static_cast<float>(frame) * dt;
            step(t, dt);
            scene->updateMatrixWorld(true);
            Vector3 eye, target;
            const float shotLen = filmSeconds / 20.f;// the shot list scales with --seconds
            if (t < 7.f * shotLen) {
                const Shot s = Director::tracking(t, lead, aspect, camera->fov);
                eye = s.pos;
                target = s.target;
            } else if (t < 13.f * shotLen) {
                const float u = (t - 7.f * shotLen) / (6.f * shotLen);
                eye = lead.pathPos + lead.fwd * mixf(-5.f, 1.f, u) - lead.right * 30.f;
                eye.y = groundHeight(eye.x, eye.z) + 2.6f;
                target = lead.pathPos + lead.fwd * mixf(-1.f, 1.5f, u) - lead.right * 6.f;
                target.y = lead.groundAvg + 5.5f;
            } else {
                const float u = smoother((t - 13.f * shotLen) / (7.f * shotLen));
                eye = lead.pathPos + lead.fwd * mixf(-24.f, -80.f, u) - lead.right * mixf(14.f, 46.f, u);
                eye.y = groundHeight(eye.x, eye.z) + mixf(4.f, 28.f, u);
                target = lead.pathPos + lead.fwd * mixf(0.f, 60.f, u);
                target.y = lead.groundAvg + mixf(6.f, 12.f, u);
            }
            camera->position.copy(eye);
            camera->lookAt(target);
            controls.target.copy(target);
            camera->updateMatrixWorld();
            frameUpdate(t);
            renderer.render(*scene, *camera);
            char name[64];
            std::snprintf(name, sizeof(name), "/frame_%04d.png", frame);
            renderer.writeFramebuffer(filmDir + name);
            if (++frame >= total) {
                std::cout << "wrote " << total << " frames to " << filmDir << std::endl;
                std::exit(0);
            }
        });
        return 0;
    }

    // ---- headless stills --------------------------------------------------------------------

    if (headless) {
        const float dt = 1.f / 30.f;
        for (float t = dt; t <= startTime + 1e-4f; t += dt) step(t, dt);
        scene->updateMatrixWorld(true);
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        Walker& w = lead;
        if (view == "opening") {
            director.update(startTime, 0.f, w, aspect);
        } else if (view == "wide") {
            director.inspect(nullptr, w.pathPos + Vector3(0, w.groundAvg + 6.f, 0), 140.f, 25.f, 14.f, w.fwd, w.right);
        } else if (view == "hatch") {
            Vector3 p = kit.doorLeft;
            w.body->localToWorld(p);
            director.inspect(w.body.get(), p, 4.6f, 70.f, 6.f, w.fwd, w.right);
        } else if (view == "knee") {
            Vector3 p(kit.femur_, 0.f, 0.f);
            w.legs[0].femur->localToWorld(p);
            director.inspect(w.legs[0].femur.get(), p, 6.5f, 55.f, 8.f, w.fwd, w.right);
        } else if (view == "legside") {
            // square-on to the front-left leg's plane, both rams in profile
            Vector3 p(2.6f, 1.5f, 0.f);
            w.legs[0].femur->localToWorld(p);
            Vector3 axis(0.f, 0.f, 1.f);
            axis.transformDirection(*w.legs[0].femur->matrixWorld);
            if (axis.dot(w.right) > 0) axis.negate();// look from the outside of the left leg
            camera->fov = 38;
            director.inspect(w.legs[0].femur.get(), p, 15.f, 0.f, 0.f, w.fwd, w.right);
            camera->position.copy(p + axis * 15.f);
            camera->lookAt(p);
        } else if (view == "ram") {
            Vector3 p(1.f, 0.f, 0.f);
            w.legs[0].rodKnee->localToWorld(p);
            director.inspect(w.legs[0].rodKnee.get(), p, 3.6f, 80.f, 15.f, w.fwd, w.right);
        } else if (view == "foot") {
            int k = 0;
            for (int i = 0; i < 4; ++i)
                if (i % 2 == 0) k = i;
            Vector3 p(0.f, 0.6f, 0.f);
            w.legs[k].foot->localToWorld(p);
            director.inspect(w.legs[k].foot.get(), p, 6.f, 60.f, 16.f, w.fwd, w.right);
        } else if (view == "prints") {
            Vector3 lp, lf, lr;
            leadPath.frame(startTime - 18.f, lp, lf, lr);
            Vector3 p = lp - lr * 8.4f;
            p.y = groundHeight(p.x, p.z);
            director.inspect(nullptr, p, 9.f, 160.f, 22.f, lf, lr);
        } else if (view == "crew") {
            Vector3 p(0.f, 1.2f, 0.f);
            crew[2].root->localToWorld(p);
            director.inspect(crew[2].root.get(), p, 5.f, 30.f, 6.f, w.fwd, w.right);
        } else if (view == "hopper") {
            Vector3 p;
            hoppers[0].root->getWorldPosition(p);
            Vector3 hf;
            hoppers[0].root->localToWorld(hf);
            director.inspect(hoppers[0].root.get(), p, 16.f, 140.f, 10.f, w.fwd, w.right);
        } else if (view == "below") {
            // from the snow under the belly, looking up past the hull
            const Vector3 target = w.bodyCentre() - w.fwd * 1.5f;
            Vector3 eye = w.pathPos + w.fwd * 6.5f - w.right * 2.5f;
            eye.y = groundHeight(eye.x, eye.z) + 1.6f;
            director.inspect(w.body.get(), target, eye.distanceTo(target), 0.f, 0.f, w.fwd, w.right);
            camera->position.copy(eye);
            camera->lookAt(target);
        }
        // --click x,y: the live click path (pick, fly in, follow), then 1.5 s of animation
        if (clickAt) {
            frameUpdate(startTime);
            camera->updateMatrixWorld();
            handleClick(*clickAt);
            float t = startTime;
            for (int i = 0; i < 45; ++i) {
                t += dt;
                step(t, dt);
                scene->updateMatrixWorld(true);
                director.update(t, dt, w, aspect);
            }
            startTime = t;
        }
        frameUpdate(startTime);
        camera->updateMatrixWorld();
        // --bench N: N live frames (simulate, follow, render) timed before the still is written
        int frames = 0;
        float benchT = startTime;
        Clock benchClock;
        canvas.animate([&] {
            if (frames < benchFrames) {
                if (frames == 0) benchClock.start();
                benchT += dt;
                step(benchT, dt);
                scene->updateMatrixWorld(true);
                director.update(benchT, dt, w, aspect);
                frameUpdate(benchT);
            }
            renderer.render(*scene, *camera);
            if (++frames == benchFrames) {
                renderer.writeFramebuffer(shotPath);
                std::printf("bench: %d frames, %.2f ms/frame\n", benchFrames, 1000.f * benchClock.getElapsedTime() / static_cast<float>(benchFrames));
                std::exit(0);
            }
            if (benchFrames == 0 && frames >= 3) {
                renderer.writeFramebuffer(shotPath);
                std::cout << "wrote " << shotPath << std::endl;
                std::exit(0);
            }
        });
        return 0;
    }

    // ---- live -------------------------------------------------------------------------------

    float t = 0.f;
    Clock clock;
    canvas.animate([&] {
        const float dt = std::min(clock.getDelta(), 1.f / 20.f);
        if (!paused) {
            t += dt;
            step(t, dt);
        }
        scene->updateMatrixWorld(true);
        director.update(t, dt, lead, camera->aspect);
        frameUpdate(t);
        renderer.render(*scene, *camera);
    });
}
