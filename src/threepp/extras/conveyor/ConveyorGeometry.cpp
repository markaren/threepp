
#include "threepp/extras/conveyor/ConveyorGeometry.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>

namespace threepp::conveyor {

    namespace {

        // The shared per-point frame every strip builder uses: tangent (centered
        // where possible), horizontal lateral and belt normal. Matches
        // segmentOrientation so ribbons line up with the per-segment belt frame.
        void frameAt(const std::vector<Vector3>& centerline, std::size_t i,
                     Vector3& t, Vector3& lat, Vector3& nv) {

            const std::size_t n = centerline.size();
            if (i == 0) {
                t.subVectors(centerline[1], centerline[0]);
            } else if (i + 1 == n) {
                t.subVectors(centerline[n - 1], centerline[n - 2]);
            } else {
                t.subVectors(centerline[i + 1], centerline[i - 1]);
            }
            if (t.length() < 1e-6f) t.set(1, 0, 0);
            t.normalize();

            const Vector3 up(0, 1, 0);
            if (std::abs(t.dot(up)) > 0.999f) {
                lat.set(0, 0, 1);
            } else {
                lat.crossVectors(t, up).normalize();
            }
            nv.crossVectors(lat, t).normalize();
        }

        float pathLength(const std::vector<Vector3>& pts) {

            float total = 0.f;
            for (std::size_t i = 0; i + 1 < pts.size(); ++i) total += pts[i].distanceTo(pts[i + 1]);
            return total;
        }

    }// namespace

    Quaternion segmentOrientation(const Vector3& a, const Vector3& b) {

        Vector3 x(b.x - a.x, b.y - a.y, b.z - a.z);
        if (x.length() < 1e-6f) return Quaternion();
        x.normalize();
        const Vector3 up(0, 1, 0);
        Vector3 z;
        if (std::abs(x.dot(up)) > 0.999f) {
            z.set(0, 0, 1);// (near) vertical travel — arbitrary horizontal width axis
        } else {
            z.crossVectors(x, up).normalize();
        }
        Vector3 y;
        y.crossVectors(z, x).normalize();
        Matrix4 m;
        m.makeBasis(x, y, z);
        Quaternion q;
        q.setFromRotationMatrix(m);
        return q;
    }

    std::shared_ptr<BufferGeometry> ribbonGeometry(const std::vector<Vector3>& centerline,
                                                   float width) {

        auto geom = BufferGeometry::create();
        const std::size_t n = centerline.size();
        if (n < 2) return geom;

        const float hw = width * 0.5f;

        std::vector<float> pos, nrm, uv;
        pos.reserve(n * 6);
        nrm.reserve(n * 6);
        uv.reserve(n * 4);

        float runLen = 0.f;
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0) runLen += centerline[i].distanceTo(centerline[i - 1]);

            Vector3 t, lat, nv;
            frameAt(centerline, i, t, lat, nv);

            const Vector3& c = centerline[i];
            pos.insert(pos.end(), {c.x - lat.x * hw, c.y - lat.y * hw, c.z - lat.z * hw});
            pos.insert(pos.end(), {c.x + lat.x * hw, c.y + lat.y * hw, c.z + lat.z * hw});
            nrm.insert(nrm.end(), {nv.x, nv.y, nv.z, nv.x, nv.y, nv.z});
            uv.insert(uv.end(), {0.f, runLen, 1.f, runLen});
        }

        std::vector<unsigned int> idx;
        idx.reserve((n - 1) * 6);
        for (std::size_t i = 0; i + 1 < n; ++i) {
            const auto l0 = static_cast<unsigned int>(i * 2);
            const unsigned int r0 = l0 + 1, l1 = l0 + 2, r1 = l0 + 3;
            idx.insert(idx.end(), {l0, r0, r1, l0, r1, l1});
        }

        geom->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geom->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geom->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geom->setIndex(idx);
        geom->computeBoundingSphere();
        geom->computeBoundingBox();
        return geom;
    }

    std::shared_ptr<BufferGeometry> wallGeometry(const std::vector<Vector3>& centerline,
                                                 float height) {

        auto geom = BufferGeometry::create();
        const std::size_t n = centerline.size();
        if (n < 2) return geom;

        const Vector3 up(0, 1, 0);
        std::vector<float> pos, nrm, uv;
        pos.reserve(n * 6);
        nrm.reserve(n * 6);
        uv.reserve(n * 4);

        float runLen = 0.f;
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0) runLen += centerline[i].distanceTo(centerline[i - 1]);

            Vector3 t, lat, nv;
            frameAt(centerline, i, t, lat, nv);

            const Vector3& c = centerline[i];
            pos.insert(pos.end(), {c.x, c.y, c.z});                                              // base
            pos.insert(pos.end(), {c.x + up.x * height, c.y + up.y * height, c.z + up.z * height});// top
            nrm.insert(nrm.end(), {lat.x, lat.y, lat.z, lat.x, lat.y, lat.z});
            uv.insert(uv.end(), {runLen, 0.f, runLen, 1.f});
        }

        std::vector<unsigned int> idx;
        idx.reserve((n - 1) * 6);
        for (std::size_t i = 0; i + 1 < n; ++i) {
            const auto b0 = static_cast<unsigned int>(i * 2);
            const unsigned int t0 = b0 + 1, b1 = b0 + 2, t1 = b0 + 3;
            idx.insert(idx.end(), {b0, b1, t1, b0, t1, t0});
        }

        geom->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geom->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geom->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geom->setIndex(idx);
        geom->computeBoundingSphere();
        geom->computeBoundingBox();
        return geom;
    }

    float rollerSpacing(float radius) {

        return radius * 2.15f;
    }

    std::vector<Roller> rollerTransforms(const std::vector<Vector3>& centerline,
                                         float radius, float spacing) {

        std::vector<Roller> out;
        const std::size_t n = centerline.size();
        if (n < 2 || radius <= 1e-4f || spacing <= 1e-4f) return out;
        const Vector3 up(0, 1, 0);

        const float total = pathLength(centerline);
        if (total < 1e-4f) return out;

        float next = spacing * 0.5f;// first roller centre, half-gap margin at the ends
        float acc = 0.f;
        for (std::size_t i = 0; i + 1 < n && next < total; ++i) {
            const float seg = centerline[i].distanceTo(centerline[i + 1]);
            if (seg < 1e-6f) continue;
            Vector3 t(centerline[i + 1].x - centerline[i].x,
                      centerline[i + 1].y - centerline[i].y,
                      centerline[i + 1].z - centerline[i].z);
            t.multiplyScalar(1.f / seg);
            Vector3 lat;
            if (std::abs(t.dot(up)) > 0.999f) lat.set(0, 0, 1);
            else lat.crossVectors(t, up).normalize();
            Vector3 nv;
            nv.crossVectors(lat, t).normalize();// belt normal: top is +nv from the centre
            Quaternion q;
            q.setFromUnitVectors(up, lat);// cylinder axis (+Y) -> width direction
            for (; next < acc + seg; next += spacing) {
                const float f = (next - acc) / seg;
                Vector3 p;
                p.lerpVectors(centerline[i], centerline[i + 1], f);
                Roller r;
                r.center.set(p.x - nv.x * radius, p.y - nv.y * radius, p.z - nv.z * radius);
                r.orientation.copy(q);
                out.push_back(r);
            }
            acc += seg;
        }
        return out;
    }

    void cleatPoseAt(const std::vector<Vector3>& centerline, float s, float height,
                     float foldAngle, Vector3& center, Quaternion& orientation) {

        const std::size_t n = centerline.size();
        if (n == 0) return;
        if (n == 1) {
            center = centerline[0];
            return;
        }
        float acc = 0.f;
        std::size_t seg = 0;
        float t = 0.f;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            const float len = centerline[i].distanceTo(centerline[i + 1]);
            seg = i;
            if (s <= acc + len || i + 2 == n) {
                t = len > 1e-6f ? std::clamp((s - acc) / len, 0.f, 1.f) : 0.f;
                break;
            }
            acc += len;
        }
        const Vector3& a = centerline[seg];
        const Vector3& b = centerline[seg + 1];
        Vector3 base;
        base.lerpVectors(a, b, t);
        Quaternion fold;
        fold.setFromAxisAngle(Vector3(0, 0, 1), foldAngle);// about the bar's own width axis
        orientation.multiplyQuaternions(segmentOrientation(a, b), fold);
        Vector3 up(0, 1, 0);
        up.applyQuaternion(orientation);// the folded bar's up direction (its length axis)
        center.set(base.x + up.x * height * 0.5f, base.y + up.y * height * 0.5f,
                   base.z + up.z * height * 0.5f);
    }

    float cleatFold(float s, float length, float rampLen) {

        if (rampLen <= 1e-4f) return 0.f;
        const float half = static_cast<float>(math::PI) * 0.5f;
        float f, sign;
        if (s < rampLen) {
            f = 1.f - s / rampLen;
            sign = 1.f;// start: +PI/2 (folded back) -> 0 (standing)
        } else if (s > length - rampLen) {
            f = 1.f - (length - s) / rampLen;
            sign = -1.f;// end: 0 (standing) -> -PI/2 (folded forward)
        } else {
            return 0.f;
        }
        f = std::clamp(f, 0.f, 1.f);
        f = f * f * (3.f - 2.f * f);// smoothstep ease
        return sign * half * f;
    }

    std::vector<float> cleatOffsets(float length, float spacing) {

        std::vector<float> out;
        if (length < 1e-4f || spacing < 1e-4f) return out;
        const int count = std::max(1, static_cast<int>(std::lround(length / spacing)));
        const float pitch = length / static_cast<float>(count);
        for (int i = 0; i < count; ++i) out.push_back((static_cast<float>(i) + 0.5f) * pitch);
        return out;
    }

    std::vector<Cleat> cleatTransforms(const std::vector<Vector3>& centerline,
                                       float height, float spacing) {

        std::vector<Cleat> out;
        if (centerline.size() < 2 || height <= 1e-4f || spacing <= 1e-4f) return out;
        const float total = pathLength(centerline);
        for (float s : cleatOffsets(total, spacing)) {
            Cleat c;
            cleatPoseAt(centerline, s, height, 0.f, c.center, c.orientation);
            out.push_back(c);
        }
        return out;
    }

    Arc computeArc(const Vector3& A, const Vector3& C, const Vector3& B,
                   const Vector3& incoming) {

        const float PI = static_cast<float>(math::PI);
        Arc r;
        const Vector3 rA(A.x - C.x, 0.f, A.z - C.z);
        const Vector3 rB(B.x - C.x, 0.f, B.z - C.z);
        r.radA = rA.length();
        r.radB = rB.length();
        if (r.radA < 1e-4f || r.radB < 1e-4f) return r;// degenerate (A or B at centre)
        r.a0 = std::atan2(rA.z, rA.x);
        const float a1 = std::atan2(rB.z, rB.x);
        // Shortest signed angular difference, in (-PI, PI].
        float d = a1 - r.a0;
        while (d <= -PI) d += 2.f * PI;
        while (d > PI) d -= 2.f * PI;
        if (std::abs(d) < PI - 0.05f) {
            // Minor (shorter) arc — exact for a 90 degree turn and any bend
            // under 180. Direction is fixed by geometry (which side B is on).
            r.sweep = d;
        } else {
            // ~180 degrees: both directions are equal-length, so pick the side
            // that continues the incoming flow.
            const Vector3 tCCW(-rA.z, 0.f, rA.x);// tangent at A if sweeping CCW
            const bool ccw = incoming.length() > 1e-5f
                                     ? (tCCW.x * incoming.x + tCCW.z * incoming.z) >= 0.f
                                     : (d >= 0.f);
            r.sweep = ccw ? PI : -PI;
        }
        r.valid = true;
        return r;
    }

    std::vector<Vector3> resamplePath(const std::vector<Waypoint>& wps, bool smooth,
                                      int samplesPerSegment) {

        const float PI = static_cast<float>(math::PI);
        bool hasArc = false;
        for (const auto& w : wps) {
            if (w.arcCenter) {
                hasArc = true;
                break;
            }
        }

        if (!hasArc) {
            std::vector<Vector3> ctrl;
            ctrl.reserve(wps.size());
            for (const auto& w : wps) ctrl.push_back(w.pos);
            if (!smooth || ctrl.size() < 3) return ctrl;
            CatmullRomCurve3 curve(ctrl);
            const auto divisions = static_cast<unsigned int>(
                    (ctrl.size() - 1) * static_cast<std::size_t>(std::max(2, samplesPerSegment)));
            return curve.getSpacedPoints(divisions);
        }

        std::vector<Vector3> out;
        const int n = static_cast<int>(wps.size());
        int i = 0;
        while (i < n) {
            if (wps[i].arcCenter) {
                // Need a preceding emitted point (A) and a following regular point (B).
                if (out.empty() || i + 1 >= n || wps[i + 1].arcCenter) {
                    ++i;
                    continue;
                }
                const Vector3 A = out.back();
                const Vector3 C = wps[i].pos;
                const Vector3 B = wps[i + 1].pos;
                Vector3 incoming(0.f, 0.f, 0.f);
                if (out.size() >= 2) {
                    incoming.set(out.back().x - out[out.size() - 2].x, 0.f,
                                 out.back().z - out[out.size() - 2].z);
                }
                const Arc arc = computeArc(A, C, B, incoming);
                if (!arc.valid) {
                    out.push_back(B);
                    i += 2;
                    continue;
                }
                const int steps = std::max(2, static_cast<int>(std::ceil(
                        std::abs(arc.sweep) / (PI * 0.5f) * static_cast<float>(std::max(2, samplesPerSegment)))));
                for (int k = 1; k <= steps; ++k) {
                    const float t = static_cast<float>(k) / static_cast<float>(steps);
                    const float ang = arc.a0 + arc.sweep * t;
                    const float rad = arc.radA + (arc.radB - arc.radA) * t;
                    out.emplace_back(C.x + rad * std::cos(ang), A.y + (B.y - A.y) * t,
                                     C.z + rad * std::sin(ang));
                }
                i += 2;
            } else {
                out.push_back(wps[i].pos);
                ++i;
            }
        }
        return out;
    }

    std::vector<PathRun> resamplePathByKind(const std::vector<Waypoint>& wps, bool smooth,
                                            int samplesPerSegment) {

        const float PI = static_cast<float>(math::PI);
        const int n = static_cast<int>(wps.size());
        bool hasArc = false;
        for (const auto& w : wps) {
            if (w.arcCenter) {
                hasArc = true;
                break;
            }
        }

        std::vector<Vector3> pts;
        std::vector<SegKind> edgeKind;// kind of edge pts[i] -> pts[i+1]

        if (!hasArc && (!smooth || n < 3)) {
            for (const auto& w : wps) pts.push_back(w.pos);
            for (int j = 0; j + 1 < n; ++j) edgeKind.push_back(wps[j].segKind);
        } else if (!hasArc) {
            // Smooth Catmull-Rom: sample each control-point span over its own
            // t-subrange so every output edge maps to a known waypoint segment
            // (the curve passes through each control point at t = j/(n-1)).
            std::vector<Vector3> ctrl;
            ctrl.reserve(n);
            for (const auto& w : wps) ctrl.push_back(w.pos);
            CatmullRomCurve3 curve(ctrl);
            const int steps = std::max(2, samplesPerSegment);
            Vector3 p;
            curve.getPoint(0.f, p);
            pts.push_back(p);
            for (int j = 0; j + 1 < n; ++j) {
                for (int k = 1; k <= steps; ++k) {
                    const float t = (static_cast<float>(j) + static_cast<float>(k) / static_cast<float>(steps)) /
                                    static_cast<float>(n - 1);
                    curve.getPoint(t, p);
                    pts.push_back(p);
                    edgeKind.push_back(wps[j].segKind);
                }
            }
        } else {
            // Arc walk (mirrors resamplePath). A straight edge into a regular
            // waypoint carries the previous waypoint's kind; arc spans are flat.
            int i = 0;
            while (i < n) {
                if (wps[i].arcCenter) {
                    if (pts.empty() || i + 1 >= n || wps[i + 1].arcCenter) {
                        ++i;
                        continue;
                    }
                    const Vector3 A = pts.back();
                    const Vector3 C = wps[i].pos;
                    const Vector3 B = wps[i + 1].pos;
                    Vector3 incoming(0.f, 0.f, 0.f);
                    if (pts.size() >= 2) {
                        incoming.set(pts.back().x - pts[pts.size() - 2].x, 0.f,
                                     pts.back().z - pts[pts.size() - 2].z);
                    }
                    const Arc arc = computeArc(A, C, B, incoming);
                    if (!arc.valid) {
                        pts.push_back(B);
                        edgeKind.push_back(SegKind::Flat);
                        i += 2;
                        continue;
                    }
                    const int steps = std::max(2, static_cast<int>(std::ceil(
                            std::abs(arc.sweep) / (PI * 0.5f) * static_cast<float>(std::max(2, samplesPerSegment)))));
                    for (int k = 1; k <= steps; ++k) {
                        const float t = static_cast<float>(k) / static_cast<float>(steps);
                        const float ang = arc.a0 + arc.sweep * t;
                        const float rad = arc.radA + (arc.radB - arc.radA) * t;
                        pts.emplace_back(C.x + rad * std::cos(ang), A.y + (B.y - A.y) * t,
                                         C.z + rad * std::sin(ang));
                        edgeKind.push_back(SegKind::Flat);
                    }
                    i += 2;
                } else {
                    if (!pts.empty()) edgeKind.push_back(wps[i - 1].segKind);
                    pts.push_back(wps[i].pos);
                    ++i;
                }
            }
        }

        std::vector<PathRun> runs;
        if (pts.size() < 2) return runs;
        std::size_t start = 0;
        for (std::size_t e = 0; e < edgeKind.size(); ++e) {
            if (e + 1 == edgeKind.size() || edgeKind[e + 1] != edgeKind[e]) {
                PathRun run;
                run.kind = edgeKind[e];
                run.pts.assign(pts.begin() + static_cast<long>(start),
                               pts.begin() + static_cast<long>(e) + 2);
                runs.push_back(std::move(run));
                start = e + 1;
            }
        }
        return runs;
    }

    Vector3 pointAlong(const std::vector<Vector3>& path, float dist) {

        if (path.empty()) return Vector3();
        float acc = 0.f;
        for (std::size_t i = 0; i + 1 < path.size(); ++i) {
            const float seg = path[i].distanceTo(path[i + 1]);
            if (acc + seg >= dist) {
                Vector3 r;
                r.lerpVectors(path[i], path[i + 1], (dist - acc) / std::max(seg, 1e-6f));
                return r;
            }
            acc += seg;
        }
        return path.back();
    }

    FrameProfile FrameProfile::forWidth(float width) {

        FrameProfile p;
        const float w = std::max(width, 0.1f);
        p.railHeight = std::clamp(w * 0.18f, 0.06f, 0.25f);
        p.railThickness = std::clamp(w * 0.08f, 0.03f, 0.12f);
        p.railTopOffset = p.railHeight * 0.2f;
        p.legThickness = std::clamp(w * 0.1f, 0.04f, 0.15f);
        p.legSpacing = std::clamp(w * 3.f, 1.2f, 3.f);
        p.drumRadius = std::clamp(w * 0.13f, 0.05f, 0.2f);
        return p;
    }

    std::shared_ptr<BufferGeometry> railGeometry(const std::vector<Vector3>& centerline,
                                                 float width, int side,
                                                 const FrameProfile& profile) {

        auto geom = BufferGeometry::create();
        const std::size_t n = centerline.size();
        if (n < 2) return geom;

        const float o = (side < 0 ? -1.f : 1.f) * (width * 0.5f + profile.railThickness * 0.5f);
        const float ht = profile.railThickness * 0.5f;

        // Section corners per ring, in the (lat, nv) plane about the offset
        // point: [0] = -lat bottom, [1] = +lat bottom, [2] = +lat top, [3] = -lat top.
        std::vector<Vector3> corners(n * 4);
        std::vector<Vector3> lats(n), nvs(n);
        std::vector<float> arc(n, 0.f);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0) arc[i] = arc[i - 1] + centerline[i].distanceTo(centerline[i - 1]);
            Vector3 t, lat, nv;
            frameAt(centerline, i, t, lat, nv);
            lats[i] = lat;
            nvs[i] = nv;
            Vector3 p = centerline[i];
            p.addScaledVector(lat, o);
            Vector3 top = p;
            top.addScaledVector(nv, profile.railTopOffset);
            Vector3 bottom = p;
            bottom.addScaledVector(nv, profile.railTopOffset - profile.railHeight);
            corners[i * 4 + 0] = bottom;
            corners[i * 4 + 0].addScaledVector(lat, -ht);
            corners[i * 4 + 1] = bottom;
            corners[i * 4 + 1].addScaledVector(lat, ht);
            corners[i * 4 + 2] = top;
            corners[i * 4 + 2].addScaledVector(lat, ht);
            corners[i * 4 + 3] = top;
            corners[i * 4 + 3].addScaledVector(lat, -ht);
        }

        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        pos.reserve(n * 4 * 6);
        nrm.reserve(n * 4 * 6);
        uv.reserve(n * 4 * 4);
        idx.reserve((n - 1) * 4 * 6);

        // Four longitudinal faces, each its own watertight strip. The corner
        // pair (a, b) per face is ordered so the ribbon index pattern's face
        // normal cross(b - a, tangent) points OUTWARD (see ribbonGeometry).
        struct Face {
            int a, b;    // corner indices per ring
            int normal;  // 0 = +nv (top), 1 = -nv (bottom), 2 = +lat, 3 = -lat
        };
        const Face faces[4] = {
                {3, 2, 0},// top:    b - a = +lat -> normal +nv
                {1, 0, 1},// bottom: b - a = -lat -> normal -nv
                {2, 1, 2},// +lat:   b - a = -nv  -> normal +lat
                {0, 3, 3},// -lat:   b - a = +nv  -> normal -lat
        };

        for (const auto& face : faces) {
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            for (std::size_t i = 0; i < n; ++i) {
                Vector3 fn;
                switch (face.normal) {
                    case 0: fn = nvs[i]; break;
                    case 1: fn = nvs[i]; fn.negate(); break;
                    case 2: fn = lats[i]; break;
                    default: fn = lats[i]; fn.negate(); break;
                }
                const Vector3& a = corners[i * 4 + face.a];
                const Vector3& b = corners[i * 4 + face.b];
                pos.insert(pos.end(), {a.x, a.y, a.z, b.x, b.y, b.z});
                nrm.insert(nrm.end(), {fn.x, fn.y, fn.z, fn.x, fn.y, fn.z});
                uv.insert(uv.end(), {arc[i], 0.f, arc[i], 1.f});
            }
            for (std::size_t i = 0; i + 1 < n; ++i) {
                const auto a0 = base + static_cast<unsigned int>(i * 2);
                const unsigned int b0 = a0 + 1, a1 = a0 + 2, b1 = a0 + 3;
                idx.insert(idx.end(), {a0, b0, b1, a0, b1, a1});
            }
        }

        // End caps close the section at both ends of an open rail.
        for (int end = 0; end < 2; ++end) {
            const std::size_t i = end == 0 ? 0 : n - 1;
            Vector3 t, lat, nv;
            frameAt(centerline, i, t, lat, nv);
            if (end == 0) t.negate();// outward
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            for (int c = 0; c < 4; ++c) {
                const Vector3& p = corners[i * 4 + c];
                pos.insert(pos.end(), {p.x, p.y, p.z});
                nrm.insert(nrm.end(), {t.x, t.y, t.z});
                uv.insert(uv.end(), {0.f, 0.f});
            }
            // Corners run bottom-left, bottom-right, top-right, top-left; wind
            // so the cap faces along the outward normal.
            if (end == 0) idx.insert(idx.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
            else idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }

        geom->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geom->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geom->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geom->setIndex(idx);
        geom->computeBoundingSphere();
        geom->computeBoundingBox();
        return geom;
    }

    std::vector<Leg> legTransforms(const std::vector<Vector3>& centerline,
                                   float width, float floorY,
                                   const FrameProfile& profile) {

        std::vector<Leg> out;
        const std::size_t n = centerline.size();
        if (n < 2) return out;

        const float total = pathLength(centerline);
        if (total < 1e-3f) return out;

        // Stations: one pair near each end plus evenly spread pairs between,
        // derived from a whole count so spacing stays uniform on any length.
        const float endMargin = std::min(0.3f, total * 0.25f);
        std::vector<float> stations;
        const float span = total - 2.f * endMargin;
        const int between = span > profile.legSpacing
                                    ? static_cast<int>(std::floor(span / profile.legSpacing))
                                    : 0;
        stations.push_back(endMargin);
        for (int i = 1; i <= between; ++i) {
            stations.push_back(endMargin + span * static_cast<float>(i) / static_cast<float>(between + 1));
        }
        if (span > 0.05f) stations.push_back(total - endMargin);

        const Vector3 up(0, 1, 0);
        const float o = width * 0.5f + profile.railThickness * 0.5f;

        for (float s : stations) {
            // Frame at the station: interpolate the segment it falls in.
            float acc = 0.f;
            std::size_t seg = 0;
            float f = 0.f;
            for (std::size_t i = 0; i + 1 < n; ++i) {
                const float len = centerline[i].distanceTo(centerline[i + 1]);
                seg = i;
                if (s <= acc + len || i + 2 == n) {
                    f = len > 1e-6f ? std::clamp((s - acc) / len, 0.f, 1.f) : 0.f;
                    break;
                }
                acc += len;
            }
            Vector3 c;
            c.lerpVectors(centerline[seg], centerline[seg + 1], f);
            Vector3 t(centerline[seg + 1].x - centerline[seg].x,
                      centerline[seg + 1].y - centerline[seg].y,
                      centerline[seg + 1].z - centerline[seg].z);
            if (t.length() < 1e-6f) t.set(1, 0, 0);
            t.normalize();
            Vector3 lat;
            if (std::abs(t.dot(up)) > 0.999f) lat.set(0, 0, 1);
            else lat.crossVectors(t, up).normalize();

            // Legs are vertical (like real conveyor stands), so the top anchors
            // under the rail and the yaw aligns the square section with the path.
            Quaternion yaw;
            yaw.setFromAxisAngle(up, std::atan2(lat.x, lat.z));

            for (int side = -1; side <= 1; side += 2) {
                Vector3 p = c;
                p.addScaledVector(lat, static_cast<float>(side) * o);
                const float topY = p.y + profile.railTopOffset - profile.railHeight;
                const float length = topY - floorY;
                if (length < profile.minLegLength) continue;
                Leg leg;
                leg.center.set(p.x, (topY + floorY) * 0.5f, p.z);
                leg.orientation.copy(yaw);
                leg.length = length;
                out.push_back(leg);
            }
        }
        return out;
    }

    std::vector<Roller> endDrumTransforms(const std::vector<Vector3>& centerline,
                                          const FrameProfile& profile) {

        std::vector<Roller> out;
        const std::size_t n = centerline.size();
        if (n < 2) return out;

        const Vector3 up(0, 1, 0);
        for (int end = 0; end < 2; ++end) {
            const std::size_t i = end == 0 ? 0 : n - 1;
            Vector3 t, lat, nv;
            frameAt(centerline, i, t, lat, nv);
            Quaternion q;
            q.setFromUnitVectors(up, lat);
            Roller r;
            r.center = centerline[i];
            r.center.addScaledVector(nv, -profile.drumRadius);
            r.orientation.copy(q);
            out.push_back(r);
        }
        return out;
    }

    std::shared_ptr<DataTexture> beltTexture() {

        auto tex = DataTexture::create<unsigned char>(4, 64, 64);
        auto& d = tex->image().data<unsigned char>();
        for (int y = 0; y < 64; ++y) {
            for (int x = 0; x < 64; ++x) {
                const bool groove = (y < 5) || (x < 2);
                const int i = (y * 64 + x) * 4;
                d[i + 0] = groove ? 22 : 66;
                d[i + 1] = groove ? 24 : 72;
                d[i + 2] = groove ? 30 : 82;
                d[i + 3] = 255;
            }
        }
        tex->wrapS = TextureWrapping::Repeat;
        tex->wrapT = TextureWrapping::Repeat;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        tex->generateMipmaps = false;
        tex->colorSpace = ColorSpace::sRGB;
        tex->needsUpdate();
        return tex;
    }

}// namespace threepp::conveyor
