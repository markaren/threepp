#include "threepp/utils/TriangleIntersect.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace threepp;

namespace {

    // Squared doubled area below which a triangle is treated as degenerate (no usable plane).
    constexpr float degenerateAreaSq = 1e-20f;

    float coordScale(const Triangle& t) {
        const Vector3& a = t.a();
        const Vector3& b = t.b();
        const Vector3& c = t.c();
        return std::max({std::fabs(a.x), std::fabs(a.y), std::fabs(a.z),
                         std::fabs(b.x), std::fabs(b.y), std::fabs(b.z),
                         std::fabs(c.x), std::fabs(c.y), std::fabs(c.z)});
    }

    // Unit normal + offset, so the signed distances below are true distances and eps is absolute.
    bool planeOf(const Triangle& t, Vector3& normal, float& offset) {
        Vector3 e1, e2;
        e1.subVectors(t.b(), t.a());
        e2.subVectors(t.c(), t.a());
        normal.crossVectors(e1, e2);
        const float len2 = normal.lengthSq();
        if (len2 <= degenerateAreaSq) return false;
        normal.multiplyScalar(1.f / std::sqrt(len2));
        offset = normal.dot(t.a());
        return true;
    }

    // Signed vertex distances to a plane, snapped to zero within eps so a vertex
    // on the plane counts as touching rather than as a crossing of arbitrary sign.
    void planeDistances(const Triangle& t, const Vector3& n, float offset, float eps, float d[3]) {
        d[0] = n.dot(t.a()) - offset;
        d[1] = n.dot(t.b()) - offset;
        d[2] = n.dot(t.c()) - offset;
        for (int i = 0; i < 3; ++i) {
            if (std::fabs(d[i]) < eps) d[i] = 0.f;
        }
    }

    // Points where the triangle meets the plane: vertices on it plus edge crossings.
    // At most two unless the triangle is coplanar, which the caller handles.
    int planeCrossings(const Triangle& t, const float d[3], Vector3 out[3]) {
        const Vector3* v[3]{&t.a(), &t.b(), &t.c()};
        int n = 0;
        for (int i = 0; i < 3; ++i) {
            if (d[i] == 0.f) out[n++].copy(*v[i]);
        }
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3;
            if ((d[i] < 0.f && d[j] > 0.f) || (d[i] > 0.f && d[j] < 0.f)) {
                out[n++].subVectors(*v[j], *v[i]).multiplyScalar(d[i] / (d[i] - d[j])).add(*v[i]);
            }
        }
        return n;
    }

    float cross2(float ax, float ay, float bx, float by) {
        return ax * by - ay * bx;
    }

    bool onSegment2(float px, float py, float qx, float qy, float rx, float ry) {
        return std::min(px, qx) <= rx && rx <= std::max(px, qx) &&
               std::min(py, qy) <= ry && ry <= std::max(py, qy);
    }

    bool seg2Intersect(const float p[2], const float q[2], const float r[2], const float s[2]) {
        const float d1 = cross2(s[0] - r[0], s[1] - r[1], p[0] - r[0], p[1] - r[1]);
        const float d2 = cross2(s[0] - r[0], s[1] - r[1], q[0] - r[0], q[1] - r[1]);
        const float d3 = cross2(q[0] - p[0], q[1] - p[1], r[0] - p[0], r[1] - p[1]);
        const float d4 = cross2(q[0] - p[0], q[1] - p[1], s[0] - p[0], s[1] - p[1]);

        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
            ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;

        // Collinear / touching cases.
        if (d1 == 0 && onSegment2(r[0], r[1], s[0], s[1], p[0], p[1])) return true;
        if (d2 == 0 && onSegment2(r[0], r[1], s[0], s[1], q[0], q[1])) return true;
        if (d3 == 0 && onSegment2(p[0], p[1], q[0], q[1], r[0], r[1])) return true;
        if (d4 == 0 && onSegment2(p[0], p[1], q[0], q[1], s[0], s[1])) return true;
        return false;
    }

    bool pointInTriangle2(const float p[2], const float a[2], const float b[2], const float c[2]) {
        const float d1 = cross2(b[0] - a[0], b[1] - a[1], p[0] - a[0], p[1] - a[1]);
        const float d2 = cross2(c[0] - b[0], c[1] - b[1], p[0] - b[0], p[1] - b[1]);
        const float d3 = cross2(a[0] - c[0], a[1] - c[1], p[0] - c[0], p[1] - c[1]);
        const bool neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        const bool pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(neg && pos);
    }

    // Coplanar triangles: project onto the two axes the plane faces least and
    // test in 2-D (any edge pair crossing, or one triangle inside the other).
    bool coplanarOverlap(const Triangle& t1, const Triangle& t2, const Vector3& n) {
        const float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
        int i0 = 0, i1 = 1;
        if (ax > ay) {
            if (ax > az) {
                i0 = 1;
                i1 = 2;
            }
        } else if (az > ay) {
            i0 = 0;
            i1 = 1;
        } else {
            i0 = 0;
            i1 = 2;
        }

        const auto project = [i0, i1](const Triangle& t, float out[3][2]) {
            const Vector3* v[3]{&t.a(), &t.b(), &t.c()};
            for (int i = 0; i < 3; ++i) {
                out[i][0] = (*v[i])[static_cast<size_t>(i0)];
                out[i][1] = (*v[i])[static_cast<size_t>(i1)];
            }
        };

        float a[3][2], b[3][2];
        project(t1, a);
        project(t2, b);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (seg2Intersect(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3])) return true;
            }
        }
        return pointInTriangle2(a[0], b[0], b[1], b[2]) ||
               pointInTriangle2(b[0], a[0], a[1], a[2]);
    }

    // Möller's interval test: each triangle is cut by the line where the two planes
    // meet; they overlap iff the two intervals overlap. `target`, when given,
    // receives the midpoint of the shared interval.
    bool triTriIntersect(const Triangle& t1, const Triangle& t2, Vector3* target) {
        Vector3 n1, n2;
        float o1, o2;
        if (!planeOf(t1, n1, o1) || !planeOf(t2, n2, o2)) return false;

        const float eps = 1e-6f * (1.f + std::max(coordScale(t1), coordScale(t2)));

        float d1[3];
        planeDistances(t1, n2, o2, eps, d1);
        if ((d1[0] > 0 && d1[1] > 0 && d1[2] > 0) || (d1[0] < 0 && d1[1] < 0 && d1[2] < 0)) return false;

        if (d1[0] == 0.f && d1[1] == 0.f && d1[2] == 0.f) {
            if (!coplanarOverlap(t1, t2, n1)) return false;
            if (target) {
                // Coplanar overlap is an area; report the average of the two centroids.
                Vector3 c1, c2;
                t1.getMidpoint(c1);
                t2.getMidpoint(c2);
                target->addVectors(c1, c2).multiplyScalar(0.5f);
            }
            return true;
        }

        float d2[3];
        planeDistances(t2, n1, o1, eps, d2);
        if ((d2[0] > 0 && d2[1] > 0 && d2[2] > 0) || (d2[0] < 0 && d2[1] < 0 && d2[2] < 0)) return false;

        Vector3 dir;
        dir.crossVectors(n1, n2);
        const float dirLen2 = dir.lengthSq();
        if (dirLen2 <= 1e-20f) return false;// parallel planes that are not coincident

        Vector3 p1[3], p2[3];
        const int c1 = planeCrossings(t1, d1, p1);
        const int c2 = planeCrossings(t2, d2, p2);
        if (c1 == 0 || c2 == 0) return false;

        float lo1 = std::numeric_limits<float>::infinity(), hi1 = -lo1;
        for (int i = 0; i < c1; ++i) {
            const float t = p1[i].dot(dir);
            lo1 = std::min(lo1, t);
            hi1 = std::max(hi1, t);
        }
        float lo2 = std::numeric_limits<float>::infinity(), hi2 = -lo2;
        for (int i = 0; i < c2; ++i) {
            const float t = p2[i].dot(dir);
            lo2 = std::min(lo2, t);
            hi2 = std::max(hi2, t);
        }

        const float lo = std::max(lo1, lo2);
        const float hi = std::min(hi1, hi2);
        if (lo > hi) return false;

        if (target) {
            // All crossing points lie on the intersection line: walk p1[0] along `dir` to parameter `mid`.
            const float mid = 0.5f * (lo + hi);
            const float t0 = p1[0].dot(dir);
            target->copy(dir).multiplyScalar((mid - t0) / dirLen2).add(p1[0]);
        }
        return true;
    }

    float segSegDistSq(const Vector3& p1, const Vector3& q1, const Vector3& p2, const Vector3& q2) {
        Vector3 d1, d2, r;
        d1.subVectors(q1, p1);
        d2.subVectors(q2, p2);
        r.subVectors(p1, p2);

        const float a = d1.lengthSq();
        const float e = d2.lengthSq();
        const float f = d2.dot(r);
        constexpr float tiny = 1e-20f;

        float s, t;
        if (a <= tiny && e <= tiny) return r.lengthSq();
        if (a <= tiny) {
            s = 0.f;
            t = std::clamp(f / e, 0.f, 1.f);
        } else {
            const float c = d1.dot(r);
            if (e <= tiny) {
                t = 0.f;
                s = std::clamp(-c / a, 0.f, 1.f);
            } else {
                const float b = d1.dot(d2);
                const float denom = a * e - b * b;
                s = denom != 0.f ? std::clamp((b * f - c * e) / denom, 0.f, 1.f) : 0.f;
                t = (b * s + f) / e;
                if (t < 0.f) {
                    t = 0.f;
                    s = std::clamp(-c / a, 0.f, 1.f);
                } else if (t > 1.f) {
                    t = 1.f;
                    s = std::clamp((b - c) / a, 0.f, 1.f);
                }
            }
        }

        Vector3 c1, c2;
        c1.copy(d1).multiplyScalar(s).add(p1);
        c2.copy(d2).multiplyScalar(t).add(p2);
        return c1.distanceToSquared(c2);
    }

}// namespace

namespace threepp::detail {

    float boxDistanceSq(const Box3& a, const Box3& b) {
        float sum = 0.f;
        for (size_t i = 0; i < 3; ++i) {
            const float gap = std::max(a.min()[i] - b.max()[i], b.min()[i] - a.max()[i]);
            if (gap > 0.f) sum += gap * gap;
        }
        return sum;
    }

    bool triTriOverlap(const Triangle& a, const Triangle& b) {
        return triTriIntersect(a, b, nullptr);
    }

    bool triTriIntersectionPoint(const Triangle& a, const Triangle& b, Vector3& target) {
        return triTriIntersect(a, b, &target);
    }

    float triTriDistanceSq(const Triangle& a, const Triangle& b) {
        const Vector3* va[3]{&a.a(), &a.b(), &a.c()};
        const Vector3* vb[3]{&b.a(), &b.b(), &b.c()};

        // Disjoint boxes cannot intersect, so the overlap test is skipped and the
        // boundary minimum below is the distance.
        Box3 boxA, boxB;
        for (int i = 0; i < 3; ++i) {
            boxA.expandByPoint(*va[i]);
            boxB.expandByPoint(*vb[i]);
        }
        if (boxA.intersectsBox(boxB) && triTriIntersect(a, b, nullptr)) return 0.f;

        float best = std::numeric_limits<float>::infinity();

        // Nine edge-edge pairs plus six vertex-face tests cover every disjoint configuration.
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                best = std::min(best, segSegDistSq(*va[i], *va[(i + 1) % 3], *vb[j], *vb[(j + 1) % 3]));
            }
        }

        Vector3 closest;
        for (int i = 0; i < 3; ++i) {
            b.closestPointToPoint(*va[i], closest);
            best = std::min(best, closest.distanceToSquared(*va[i]));
            a.closestPointToPoint(*vb[i], closest);
            best = std::min(best, closest.distanceToSquared(*vb[i]));
        }

        return best;
    }

}// namespace threepp::detail
