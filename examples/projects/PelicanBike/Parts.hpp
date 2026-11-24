// Small tools for building the bicycle and the pelican from code: materials from sRGB values, rods between two
// points, a tube swept along a path, a surface of revolution, a toothed gear ring, and a lofted body whose
// cross-section is a superellipse that varies along a spine.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_PARTS_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_PARTS_HPP

#include "threepp/threepp.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"

#include "Geometry.hpp"
#include "Noise.hpp"
#include "TextureLab.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

namespace pb {

    using namespace threepp;
    using mill::clamp01;
    using mill::MeshBuild;
    using mill::Rng;
    using mill::smooth;
    using mill::srgb;
    using mill::V3;

    constexpr float PI = 3.14159265358979f;
    constexpr float TAU = 6.28318530717959f;

    inline Color lin(int r, int g, int b) {
        const V3 c = srgb(r, g, b);
        return Color(c.r, c.g, c.b);
    }

    inline std::shared_ptr<MeshStandardMaterial> standard(const Color& c, float roughness, float metalness = 0.f) {
        auto m = MeshStandardMaterial::create();
        m->color = c;
        m->roughness = roughness;
        m->metalness = metalness;
        return m;
    }

    inline std::shared_ptr<MeshPhysicalMaterial> physical(const Color& c, float roughness, float metalness = 0.f) {
        auto m = MeshPhysicalMaterial::create();
        m->color = c;
        m->roughness = roughness;
        m->metalness = metalness;
        return m;
    }

    inline Vector3 lerp3(const Vector3& a, const Vector3& b, float t) { return a + (b - a) * t; }

    // orient a mesh so its +Y runs along `dir`
    inline void alignY(Object3D& o, const Vector3& dir) {
        Vector3 d = dir;
        if (d.lengthSq() < 1e-12f) return;
        d.normalize();
        o.quaternion.setFromUnitVectors(Vector3(0, 1, 0), d);
    }

    inline std::shared_ptr<Mesh> finish(const std::shared_ptr<Mesh>& m, bool cast = true, bool receive = true) {
        m->castShadow = cast;
        m->receiveShadow = receive;
        return m;
    }

    // a (tapered) cylinder from a to b
    inline std::shared_ptr<Mesh> rod(const Vector3& a, const Vector3& b, float ra, float rb, const std::shared_ptr<Material>& m, unsigned seg = 20, bool openEnded = false) {
        const Vector3 d = b - a;
        auto mesh = Mesh::create(CylinderGeometry::create(rb, ra, std::max(d.length(), 1e-5f), seg, 1, openEnded), m);
        mesh->position.copy((a + b) * 0.5f);
        alignY(*mesh, d);
        return finish(mesh);
    }

    inline std::shared_ptr<Mesh> ball(const Vector3& p, float r, const std::shared_ptr<Material>& m, unsigned seg = 20) {
        auto mesh = Mesh::create(SphereGeometry::create(r, seg, seg / 2 + 1), m);
        mesh->position.copy(p);
        return finish(mesh);
    }

    // a box of size (sx, sy, sz) centred on p
    inline std::shared_ptr<Mesh> box(const Vector3& p, float sx, float sy, float sz, const std::shared_ptr<Material>& m) {
        auto mesh = Mesh::create(BoxGeometry::create(sx, sy, sz), m);
        mesh->position.copy(p);
        return finish(mesh);
    }

    // ---- a tube swept along a path ----------------------------------------------------------------------------------
    // Rings of `seg` vertices round each point, radius r[i] (an elliptic section if ry > 0: r across, ry along the
    // frame's second axis), the frame carried along without twist. Rounded caps are flat discs.
    struct SweepOptions {
        int seg = 16;
        bool capStart = true, capEnd = true;
        V3 colour = V3(1, 1, 1);
        float uvAlong = 1.f;    // v per metre
        float aspect = 1.f;     // section height / width
        Vector3 up = Vector3(0, 1, 0);// the frame's start normal is the component of this across the first tangent
    };

    inline void appendSweep(MeshBuild& out, const std::vector<Vector3>& p, const std::vector<float>& r, const SweepOptions& o = {}) {
        const int n = static_cast<int>(p.size());
        if (n < 2) return;
        std::vector<Vector3> T(n), N(n), B(n);
        for (int i = 0; i < n; ++i) {
            const Vector3 a = p[std::max(i - 1, 0)], b = p[std::min(i + 1, n - 1)];
            T[i] = (b - a).normalize();
        }
        Vector3 up = o.up;
        if (std::abs(up.dot(T[0])) > 0.95f) up = Vector3(1, 0, 0);
        N[0] = (up - T[0] * up.dot(T[0])).normalize();
        for (int i = 1; i < n; ++i) {
            Vector3 v = N[i - 1] - T[i] * N[i - 1].dot(T[i]);
            if (v.lengthSq() < 1e-12f) v = Vector3(T[i]).cross(Vector3(0, 0, 1));
            N[i] = v.normalize();
        }
        for (int i = 0; i < n; ++i) B[i] = Vector3(T[i]).cross(N[i]);
        const unsigned base = out.vertexCount();
        float along = 0;
        for (int i = 0; i < n; ++i) {
            if (i > 0) along += (p[i] - p[i - 1]).length();
            for (int k = 0; k <= o.seg; ++k) {
                const float a = TAU * static_cast<float>(k) / static_cast<float>(o.seg);
                const Vector3 dir = N[i] * std::cos(a) + B[i] * std::sin(a) * o.aspect;
                Vector3 nrm = N[i] * (std::cos(a) * o.aspect) + B[i] * std::sin(a);
                nrm.normalize();
                out.vertex(p[i] + dir * r[i], nrm, static_cast<float>(k) / static_cast<float>(o.seg), along * o.uvAlong, o.colour);
            }
        }
        const unsigned row = static_cast<unsigned>(o.seg + 1);
        for (int i = 0; i + 1 < n; ++i)
            for (int k = 0; k < o.seg; ++k) {
                const unsigned a = base + i * row + k, b = a + 1, c = a + row, d = c + 1;
                out.index.insert(out.index.end(), {a, b, c, b, d, c});
            }
        const auto cap = [&](int i, bool start) {
            const unsigned c0 = out.vertexCount();
            const Vector3 nrm = start ? Vector3(T[i]) * -1.f : Vector3(T[i]);
            out.vertex(p[i], nrm, 0.5f, 0.5f, o.colour);
            for (int k = 0; k <= o.seg; ++k) {
                const float a = TAU * static_cast<float>(k) / static_cast<float>(o.seg);
                const Vector3 dir = N[i] * std::cos(a) + B[i] * std::sin(a) * o.aspect;
                out.vertex(p[i] + dir * r[i], nrm, 0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a), o.colour);
            }
            for (int k = 0; k < o.seg; ++k) {
                const unsigned a = c0 + 1 + k, b = a + 1;
                if (start) out.index.insert(out.index.end(), {c0, b, a});
                else out.index.insert(out.index.end(), {c0, a, b});
            }
        };
        if (o.capStart) cap(0, true);
        if (o.capEnd) cap(n - 1, false);
    }

    // the points of a Catmull-Rom spline through `ctrl`, `perSegment` samples a span
    inline std::vector<Vector3> splinePoints(const std::vector<Vector3>& ctrl, int perSegment) {
        CatmullRomCurve3 curve(ctrl, false, CatmullRomCurve3::centripetal);
        std::vector<Vector3> out;
        const int count = static_cast<int>(ctrl.size() - 1) * perSegment + 1;
        for (int i = 0; i < count; ++i) {
            Vector3 q;
            curve.getPoint(static_cast<float>(i) / static_cast<float>(count - 1), q);
            out.push_back(q);
        }
        return out;
    }

    // radii interpolated along a path of `count` points from a list of (fraction, radius) knots
    inline std::vector<float> radiiAlong(int count, std::initializer_list<std::pair<float, float>> knots) {
        std::vector<float> r(static_cast<size_t>(count));
        const std::vector<std::pair<float, float>> k(knots);
        for (int i = 0; i < count; ++i) {
            const float t = count > 1 ? static_cast<float>(i) / static_cast<float>(count - 1) : 0.f;
            size_t j = 0;
            while (j + 2 < k.size() && t > k[j + 1].first) ++j;
            const float u = clamp01((t - k[j].first) / std::max(k[j + 1].first - k[j].first, 1e-6f));
            const float s = u * u * (3.f - 2.f * u);
            r[static_cast<size_t>(i)] = k[j].second + (k[j + 1].second - k[j].second) * s;
        }
        return r;
    }

    // ---- a surface of revolution ------------------------------------------------------------------------------------
    // profile: (radius, height) points along the axis; the surface turns about `axis` through `origin`.
    inline void appendLathe(MeshBuild& out, const std::vector<std::pair<float, float>>& profile, int seg, const Vector3& origin, const Vector3& axis, const V3& colour = V3(1, 1, 1)) {
        Vector3 ax = axis;
        ax.normalize();
        Vector3 e1 = std::abs(ax.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
        e1 = (e1 - ax * e1.dot(ax)).normalize();
        const Vector3 e2 = Vector3(ax).cross(e1);
        const int n = static_cast<int>(profile.size());
        const unsigned base = out.vertexCount();
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k <= seg; ++k) {
                const float a = TAU * static_cast<float>(k) / static_cast<float>(seg);
                const Vector3 radial = e1 * std::cos(a) + e2 * std::sin(a);
                out.vertex(origin + ax * profile[static_cast<size_t>(i)].second + radial * profile[static_cast<size_t>(i)].first, radial,
                           static_cast<float>(k) / static_cast<float>(seg), static_cast<float>(i) / static_cast<float>(std::max(n - 1, 1)), colour);
            }
        }
        const unsigned row = static_cast<unsigned>(seg + 1);
        for (int i = 0; i + 1 < n; ++i)
            for (int k = 0; k < seg; ++k) {
                const unsigned a = base + i * row + k, b = a + 1, c = a + row, d = c + 1;
                out.index.insert(out.index.end(), {a, b, c, b, d, c});
            }
        out.smoothNormals(base);
    }

    // a closed cylinder about the z axis through (cx, cy), from z0 to z1, with hard edges (the side and each cap their own
    // vertices): a chain's roller or plate end
    inline void appendCylinderZ(MeshBuild& out, float cx, float cy, float radius, float z0, float z1, int seg, const V3& colour = V3(1, 1, 1)) {
        const unsigned side = out.vertexCount();
        for (int k = 0; k <= seg; ++k) {
            const float a = TAU * static_cast<float>(k) / static_cast<float>(seg);
            const Vector3 n(std::cos(a), std::sin(a), 0.f);
            out.vertex(Vector3(cx + radius * n.x, cy + radius * n.y, z0), n, static_cast<float>(k) / static_cast<float>(seg), 0.f, colour);
            out.vertex(Vector3(cx + radius * n.x, cy + radius * n.y, z1), n, static_cast<float>(k) / static_cast<float>(seg), 1.f, colour);
        }
        for (int k = 0; k < seg; ++k) {
            const unsigned a = side + 2 * k;
            out.index.insert(out.index.end(), {a, a + 2, a + 1, a + 1, a + 2, a + 3});
        }
        for (int cap = 0; cap < 2; ++cap) {
            const float z = cap == 0 ? z0 : z1;
            const Vector3 n(0, 0, cap == 0 ? -1.f : 1.f);
            const unsigned c0 = out.vertexCount();
            out.vertex(Vector3(cx, cy, z), n, 0.5f, 0.5f, colour);
            for (int k = 0; k <= seg; ++k) {
                const float a = TAU * static_cast<float>(k) / static_cast<float>(seg);
                out.vertex(Vector3(cx + radius * std::cos(a), cy + radius * std::sin(a), z), n, 0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a), colour);
            }
            for (int k = 0; k < seg; ++k) {
                const unsigned a = c0 + 1 + k;
                if (cap == 1) out.index.insert(out.index.end(), {c0, a, a + 1});
                else out.index.insert(out.index.end(), {c0, a + 1, a});
            }
        }
    }

    // ---- a toothed ring: a gear or chainring ----------------------------------------------------------------------------
    // teeth round the pitch radius, a flat plate of the given thickness out to `inner`, in the XY plane about z.
    inline void appendGearRing(MeshBuild& out, int teeth, float pitchRadius, float rootDepth, float tipHeight, float inner, float thickness, float z0, const V3& colour = V3(1, 1, 1)) {
        // outline, tooth by tooth: root, rise, tip, tip, fall, root
        std::vector<Vector3> outline;
        for (int t = 0; t < teeth; ++t) {
            const float a0 = TAU * static_cast<float>(t) / static_cast<float>(teeth), da = TAU / static_cast<float>(teeth);
            const float rr = pitchRadius - rootDepth, rt = pitchRadius + tipHeight;
            const float prof[6][2] = {{0.00f, rr}, {0.14f, rr}, {0.34f, rt}, {0.58f, rt}, {0.80f, rr}, {0.94f, rr}};
            for (const auto& q : prof) {
                const float a = a0 + q[0] * da;
                outline.emplace_back(std::cos(a) * q[1], std::sin(a) * q[1], 0.f);
            }
        }
        const int m = static_cast<int>(outline.size());
        const unsigned base = out.vertexCount();
        // front and back faces: a strip between the outline and the inner circle, both with m points
        for (int side = 0; side < 2; ++side) {
            const float z = z0 + (side == 0 ? thickness : 0.f);
            const Vector3 nrm(0, 0, side == 0 ? 1.f : -1.f);
            for (int i = 0; i < m; ++i) {
                const Vector3 o = outline[static_cast<size_t>(i)];
                out.vertex(Vector3(o.x, o.y, z), nrm, 0.5f + o.x * 3.f, 0.5f + o.y * 3.f, colour);
                const float a = std::atan2(o.y, o.x);
                out.vertex(Vector3(std::cos(a) * inner, std::sin(a) * inner, z), nrm, 0.5f + std::cos(a) * inner * 3.f, 0.5f + std::sin(a) * inner * 3.f, colour);
            }
            for (int i = 0; i < m; ++i) {
                const int j = (i + 1) % m;
                const unsigned a = base + side * 2 * m + 2 * i, b = base + side * 2 * m + 2 * j;
                if (side == 0) out.index.insert(out.index.end(), {a, b, a + 1, b, b + 1, a + 1});
                else out.index.insert(out.index.end(), {a, a + 1, b, b, a + 1, b + 1});
            }
        }
        // the rim wall
        const unsigned wall = out.vertexCount();
        for (int i = 0; i < m; ++i) {
            const Vector3 o = outline[static_cast<size_t>(i)], o2 = outline[static_cast<size_t>((i + 1) % m)];
            Vector3 nrm = (o2 - o);
            nrm = Vector3(nrm.y, -nrm.x, 0.f).normalize();
            out.vertex(Vector3(o.x, o.y, z0), nrm, 0, 0, colour);
            out.vertex(Vector3(o.x, o.y, z0 + thickness), nrm, 0, 1, colour);
            out.vertex(Vector3(o2.x, o2.y, z0), nrm, 1, 0, colour);
            out.vertex(Vector3(o2.x, o2.y, z0 + thickness), nrm, 1, 1, colour);
            const unsigned a = wall + 4 * i;
            out.index.insert(out.index.end(), {a, a + 2, a + 1, a + 1, a + 2, a + 3});
        }
    }

    // ---- a lofted body ------------------------------------------------------------------------------------------------
    // A spine in the XY plane (a Catmull-Rom through its points) carrying a frame: T along it, N its normal in the
    // plane, B = +Z across. A station gives the half-width, the half-heights above and below the spine, the
    // superellipse exponent (2 an ellipse, higher squarer) and a sideways shift of the section's centre along N.
    struct Station {
        float s;            // 0..1 along the spine, by arc length
        float w;            // half-width (along Z)
        float up, down;     // half-heights along +N and -N
        float n = 2.4f;     // superellipse exponent
        float shift = 0.f;  // the section's centre along N
    };

    struct Spine {
        std::vector<Vector3> p;      // dense points
        std::vector<Vector3> T, N;   // tangent and in-plane normal at each
        std::vector<float> s;        // arc length 0..1
        float length = 0;

        void build(const std::vector<Vector3>& ctrl, int samples = 400) {
            p = splinePoints(ctrl, samples / std::max<int>(1, static_cast<int>(ctrl.size()) - 1));
            const size_t n = p.size();
            T.resize(n);
            N.resize(n);
            s.resize(n);
            float acc = 0;
            for (size_t i = 0; i < n; ++i) {
                if (i > 0) acc += (p[i] - p[i - 1]).length();
                s[i] = acc;
            }
            length = acc;
            for (size_t i = 0; i < n; ++i) {
                s[i] /= std::max(acc, 1e-6f);
                const Vector3 a = p[i > 0 ? i - 1 : 0], b = p[std::min(i + 1, n - 1)];
                T[i] = (b - a).normalize();
                N[i] = Vector3(-T[i].y, T[i].x, 0.f);
            }
        }

        // the frame at arc-length fraction t
        void at(float t, Vector3& pos, Vector3& tan, Vector3& nor) const {
            t = std::clamp(t, 0.f, 1.f);
            const auto it = std::lower_bound(s.begin(), s.end(), t);
            size_t j = static_cast<size_t>(std::distance(s.begin(), it));
            j = std::clamp<size_t>(j, 1, s.size() - 1);
            const float u = clamp01((t - s[j - 1]) / std::max(s[j] - s[j - 1], 1e-9f));
            pos = lerp3(p[j - 1], p[j], u);
            tan = lerp3(T[j - 1], T[j], u).normalize();
            nor = lerp3(N[j - 1], N[j], u).normalize();
        }
    };

    inline float sgnpow(float v, float e) { return v < 0 ? -std::pow(-v, e) : std::pow(v, e); }

    // smooth interpolation of a station's fields at t (cubic Hermite through the table, monotone enough for shapes)
    inline Station stationAt(const std::vector<Station>& st, float t) {
        size_t j = 0;
        while (j + 2 < st.size() && t > st[j + 1].s) ++j;
        const Station &a = st[j], &b = st[j + 1];
        const float u = clamp01((t - a.s) / std::max(b.s - a.s, 1e-6f));
        // Catmull-Rom with the neighbours when they exist
        const Station &p0 = st[j > 0 ? j - 1 : 0], &p3 = st[std::min(j + 2, st.size() - 1)];
        const auto cr = [&](float y0, float y1, float y2, float y3) {
            const float u2 = u * u, u3 = u2 * u;
            return 0.5f * ((2 * y1) + (-y0 + y2) * u + (2 * y0 - 5 * y1 + 4 * y2 - y3) * u2 + (-y0 + 3 * y1 - 3 * y2 + y3) * u3);
        };
        Station r;
        r.s = t;
        r.w = std::max(cr(p0.w, a.w, b.w, p3.w), 0.f);
        r.up = std::max(cr(p0.up, a.up, b.up, p3.up), 0.f);
        r.down = std::max(cr(p0.down, a.down, b.down, p3.down), 0.f);
        r.n = cr(p0.n, a.n, b.n, p3.n);
        r.shift = cr(p0.shift, a.shift, b.shift, p3.shift);
        return r;
    }

    struct Loft {
        Spine spine;
        std::vector<Station> stations;
        int rings = 120, around = 56;

        // a point of the surface at spine fraction t and angle a (0 = +B side, going up through +N)
        [[nodiscard]] Vector3 surface(float t, float a, Vector3* normal = nullptr) const {
            Vector3 pos, tan, nor;
            spine.at(t, pos, tan, nor);
            const Station st = stationAt(stations, t);
            const float e = 2.f / std::max(st.n, 1.f);
            const float c = std::cos(a), sn = std::sin(a);
            const float x = sgnpow(c, e), y = sgnpow(sn, e);
            const Vector3 B(0, 0, 1);
            const Vector3 q = pos + nor * (st.shift + y * (y >= 0 ? st.up : st.down)) + B * (x * st.w);
            if (normal) {
                // the gradient of the implicit form |x/w|^n + |y/h|^n = 1 at that point
                const float hy = y >= 0 ? st.up : st.down;
                const float gx = sgnpow(std::abs(x), st.n - 1.f) / std::max(st.w, 1e-5f);
                const float gy = sgnpow(std::abs(y), st.n - 1.f) / std::max(hy, 1e-5f);
                *normal = (B * (x >= 0 ? gx : -gx) + nor * (y >= 0 ? gy : -gy)).normalize();
            }
            return q;
        }

        // build the mesh; `colourAt(t, a)` paints it; `u` wraps the angle, `v` runs with the length in metres
        void build(MeshBuild& out, const std::function<V3(float, float)>& colourAt, float vPerMetre = 3.f, bool capEnds = true) const {
            const unsigned base = out.vertexCount();
            for (int i = 0; i <= rings; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(rings);
                for (int k = 0; k <= around; ++k) {
                    const float a = TAU * static_cast<float>(k) / static_cast<float>(around);
                    Vector3 nrm;
                    const Vector3 q = surface(t, a, &nrm);
                    out.vertex(q, nrm, static_cast<float>(k) / static_cast<float>(around), t * spine.length * vPerMetre, colourAt ? colourAt(t, a) : V3(1, 1, 1));
                }
            }
            const unsigned row = static_cast<unsigned>(around + 1);
            for (int i = 0; i < rings; ++i)
                for (int k = 0; k < around; ++k) {
                    const unsigned a = base + i * row + k, b = a + 1, c = a + row, d = c + 1;
                    out.index.insert(out.index.end(), {a, c, b, b, c, d});
                }
            // an end that is not a point is closed with a fan
            const auto cap = [&](float t, bool start) {
                Vector3 pos, tan, nor;
                spine.at(t, pos, tan, nor);
                const Station st = stationAt(stations, t);
                if (st.w < 1e-4f && st.up + st.down < 1e-4f) return;
                const unsigned c0 = out.vertexCount();
                const Vector3 nrm = start ? Vector3(tan) * -1.f : Vector3(tan);
                out.vertex(pos + nor * st.shift, nrm, 0.5f, 0.5f, colourAt ? colourAt(t, 0.f) : V3(1, 1, 1));
                for (int k = 0; k <= around; ++k) {
                    const float a = TAU * static_cast<float>(k) / static_cast<float>(around);
                    out.vertex(surface(t, a), nrm, 0.5f, 0.5f, colourAt ? colourAt(t, a) : V3(1, 1, 1));
                }
                for (int k = 0; k < around; ++k) {
                    const unsigned a = c0 + 1 + k, b = a + 1;
                    if (start) out.index.insert(out.index.end(), {c0, a, b});
                    else out.index.insert(out.index.end(), {c0, b, a});
                }
            };
            if (capEnds) {
                cap(0.f, true);
                cap(1.f, false);
            }
        }
    };

}// namespace pb

#endif
