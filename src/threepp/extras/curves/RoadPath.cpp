
#include "threepp/extras/curves/RoadPath.hpp"

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

namespace {

    constexpr float kEpsilon = 1e-6f;

    // Beyond this a piece stops being an arc: the circle through its two ends
    // and its midpoint is what fixes it, and those three points crowd together
    // as the sweep approaches a full turn (at 2pi the ends COINCIDE and there
    // is no circle through them at all). A ring road becomes two arcs, which
    // costs nothing and keeps every fit well conditioned.
    constexpr float kMaxSweep = 1.9f * math::PI;

    // A bend this tight is a pivot, not a road; a circle this large is a
    // straight and fits as one.
    constexpr float kMinRadius = 1e-3f;
    constexpr float kMaxRadius = 1e5f;

    // Consecutive samples a fit may fail on before the search for a longer run
    // gives up. Small on purpose — it is there for the tangent check, which is
    // not monotone in the run's length, and not to paper over a real break.
    constexpr std::size_t kAllowedMisses = 3;

    struct Circle {
        Vector3 center;
        float radius = 0.f;
        bool valid = false;
    };

    // The circle through three XZ points, solved in a frame translated to `a`
    // so the coefficients stay the size of the road rather than the size of the
    // world it sits in.
    Circle circleThrough(const Vector3& a, const Vector3& b, const Vector3& c) {

        Circle out;
        const float bx = b.x - a.x, bz = b.z - a.z;
        const float cx = c.x - a.x, cz = c.z - a.z;
        const float d = 2.f * (bx * cz - bz * cx);
        if (std::abs(d) < 1e-12f) return out;// collinear

        const float b2 = bx * bx + bz * bz;
        const float c2 = cx * cx + cz * cz;
        const float ux = (b2 * cz - c2 * bz) / d;
        const float uz = (c2 * bx - b2 * cx) / d;
        out.center.set(a.x + ux, 0.f, a.z + uz);
        out.radius = std::sqrt(ux * ux + uz * uz);
        out.valid = std::isfinite(out.radius);
        return out;
    }

    float wrapToPi(float angle) {

        while (angle <= -math::PI) angle += math::TWO_PI;
        while (angle > math::PI) angle -= math::TWO_PI;
        return angle;
    }

    // Unit XZ tangents of the sample chain, centred where they can be. What a
    // piece's own tangent is held against at the samples it ends on.
    std::vector<Vector3> sampleTangents(const std::vector<Vector3>& samples) {

        const std::size_t n = samples.size();
        std::vector<Vector3> tangents(n);
        for (std::size_t k = 0; k < n; ++k) {
            const Vector3& before = samples[k > 0 ? k - 1 : 0];
            const Vector3& after = samples[k + 1 < n ? k + 1 : n - 1];
            Vector3 tangent{after.x - before.x, 0.f, after.z - before.z};
            const float length = tangent.length();
            if (length > kEpsilon) tangent.multiplyScalar(1.f / length);
            tangents[k] = tangent;
        }
        return tangents;
    }

    // Is `tangent` within `cosLimit` of the sampled curve's there? A sample with
    // no horizontal tangent to compare against (a stretch that only climbs)
    // abstains rather than failing every fit.
    bool tangentAgrees(const std::vector<Vector3>& tangents, std::size_t k,
                       float dx, float dz, float cosLimit) {

        const Vector3& reference = tangents[k];
        if (reference.length() < 0.5f) return true;
        const float length = std::sqrt(dx * dx + dz * dz);
        if (length < kEpsilon) return false;
        return (reference.x * dx + reference.z * dz) / length >= cosLimit;
    }

    // Does a straight through samples[i] and samples[j] carry everything
    // between them? Lateral distance in XZ, height against the linear grade,
    // no sample that runs backward past either end, and a direction that still
    // agrees with the curve at both ends.
    bool fitsStraight(const std::vector<Vector3>& samples, const std::vector<Vector3>& tangents,
                      std::size_t i, std::size_t j, float tolerance, float cosLimit) {

        const float dx = samples[j].x - samples[i].x;
        const float dz = samples[j].z - samples[i].z;
        const float length = std::sqrt(dx * dx + dz * dz);
        if (length < kEpsilon) return false;
        if (!tangentAgrees(tangents, i, dx, dz, cosLimit)) return false;
        if (!tangentAgrees(tangents, j, dx, dz, cosLimit)) return false;

        const float ux = dx / length, uz = dz / length;
        const float dy = samples[j].y - samples[i].y;
        for (std::size_t k = i + 1; k < j; ++k) {
            const float rx = samples[k].x - samples[i].x;
            const float rz = samples[k].z - samples[i].z;
            const float along = rx * ux + rz * uz;
            if (along < -tolerance || along > length + tolerance) return false;
            if (std::abs(rx * uz - rz * ux) > tolerance) return false;
            if (std::abs(samples[k].y - (samples[i].y + dy * along / length)) > tolerance) return false;
        }
        return true;
    }

    struct ArcFit {
        bool ok = false;
        Vector3 center;
        float radius = 0.f;
        float startAngle = 0.f;
        float sweep = 0.f;
    };

    // The circle through samples[i], the run's midpoint and samples[j], checked
    // against every sample in between: radially, in height against the linear
    // grade, for a turn that never reverses (a reversal is a cusp, not an arc),
    // and for tangents that still agree with the curve at both ends. `angles`
    // is scratch, reused across the search rather than reallocated per
    // candidate.
    ArcFit fitArc(const std::vector<Vector3>& samples, const std::vector<Vector3>& tangents,
                  std::size_t i, std::size_t j, float tolerance, float cosLimit,
                  std::vector<float>& angles) {

        ArcFit out;
        if (j < i + 2) return out;

        const std::size_t mid = i + (j - i) / 2;
        const Circle circle = circleThrough(samples[i], samples[mid], samples[j]);
        if (!circle.valid || circle.radius < kMinRadius || circle.radius > kMaxRadius) return out;

        const auto angleOf = [&circle](const Vector3& p) {
            return std::atan2(p.z - circle.center.z, p.x - circle.center.x);
        };

        // angles[k - i] is how far the run has turned by sample k, unwrapped —
        // the arc's own parameter, which is also what the height profile is
        // linear in.
        angles.clear();
        angles.push_back(0.f);
        float previous = angleOf(samples[i]);
        float turned = 0.f;
        float direction = 0.f;
        for (std::size_t k = i + 1; k <= j; ++k) {
            const float angle = angleOf(samples[k]);
            const float step = wrapToPi(angle - previous);
            if (direction == 0.f) {
                if (std::abs(step) > kEpsilon) direction = step > 0.f ? 1.f : -1.f;
            } else if (step * direction < -kEpsilon) {
                return out;// the turn reverses
            }
            turned += step;
            if (std::abs(turned) > kMaxSweep) return out;
            previous = angle;
            angles.push_back(turned);
        }
        if (std::abs(turned) < kEpsilon) return out;

        const float dy = samples[j].y - samples[i].y;
        for (std::size_t k = i; k <= j; ++k) {
            const float rx = samples[k].x - circle.center.x;
            const float rz = samples[k].z - circle.center.z;
            if (std::abs(std::sqrt(rx * rx + rz * rz) - circle.radius) > tolerance) return out;
            const float t = angles[k - i] / turned;
            if (std::abs(samples[k].y - (samples[i].y + dy * t)) > tolerance) return out;
        }

        // The tangent of a circle is its radius turned a quarter turn, the way
        // the run is going.
        const float sign = turned >= 0.f ? 1.f : -1.f;
        for (const std::size_t k : {i, j}) {
            const float rx = samples[k].x - circle.center.x;
            const float rz = samples[k].z - circle.center.z;
            if (!tangentAgrees(tangents, k, -rz * sign, rx * sign, cosLimit)) return out;
        }

        out.ok = true;
        out.center = circle.center;
        out.radius = circle.radius;
        out.startAngle = angleOf(samples[i]);
        out.sweep = turned;
        return out;
    }

}// namespace


float RoadPrimitive::horizontalLength() const {

    if (kind == Kind::Arc) return radius * std::abs(sweep);
    const float dx = end.x - start.x;
    const float dz = end.z - start.z;
    return std::sqrt(dx * dx + dz * dz);
}

float RoadPrimitive::length() const {

    const float horizontal = horizontalLength();
    const float dy = end.y - start.y;
    return std::sqrt(horizontal * horizontal + dy * dy);
}

Vector3 RoadPrimitive::pointAt(float t) const {

    // The ends are answered from the stored endpoints rather than evaluated:
    // consecutive pieces share those, and a joint that is only nearly shared is
    // a hairline crack in the road.
    if (t <= 0.f) return start;
    if (t >= 1.f) return end;

    const float y = start.y + (end.y - start.y) * t;
    if (kind == Kind::Straight) {
        return {start.x + (end.x - start.x) * t, y, start.z + (end.z - start.z) * t};
    }
    const float angle = startAngle + sweep * t;
    return {center.x + radius * std::cos(angle), y, center.z + radius * std::sin(angle)};
}

Vector3 RoadPrimitive::tangentAt(float t) const {

    if (kind == Kind::Straight) {
        Vector3 tangent{end.x - start.x, 0.f, end.z - start.z};
        const float length = tangent.length();
        if (length < kEpsilon) return {0.f, 0.f, 0.f};
        return tangent.multiplyScalar(1.f / length);
    }
    const float angle = startAngle + sweep * std::clamp(t, 0.f, 1.f);
    const float sign = sweep >= 0.f ? 1.f : -1.f;
    return {-std::sin(angle) * sign, 0.f, std::cos(angle) * sign};
}


RoadPath::Params::Params(float tolerance, float angleTolerance)
    : tolerance(tolerance), angleTolerance(angleTolerance) {}

RoadPath RoadPath::fromPoints(const std::vector<Vector3>& samples, bool closed, const Params& params) {

    RoadPath path;
    path.closed_ = closed;

    // Coincident samples carry no direction and would only fail every fit.
    std::vector<Vector3> points;
    points.reserve(samples.size());
    for (const auto& sample : samples) {
        if (points.empty() || points.back().distanceTo(sample) > kEpsilon) points.push_back(sample);
    }
    if (points.size() < 2) return path;

    const float tolerance = std::max(params.tolerance, 1e-5f);
    const float cosLimit = std::cos(std::clamp(params.angleTolerance, 0.f, math::PI * 0.5f));
    const std::size_t n = points.size();
    const std::vector<Vector3> tangents = sampleTangents(points);
    std::vector<float> angles;
    angles.reserve(n);

    std::size_t i = 0;
    while (i + 1 < n) {

        // The longest run a straight carries. Always at least one span, so the
        // walk cannot stall on a stretch nothing fits (a road climbing straight
        // up has no XZ projection to fit at all). The radial and height fits
        // only get worse as a run grows, but the END TANGENT can come back into
        // agreement a sample or two later, so the search rides out a few misses
        // rather than stopping at the first.
        std::size_t straightEnd = i + 1;
        for (std::size_t j = i + 2, missed = 0; j < n && missed <= kAllowedMisses; ++j) {
            if (fitsStraight(points, tangents, i, j, tolerance, cosLimit)) {
                straightEnd = j;
                missed = 0;
            } else {
                ++missed;
            }
        }

        // ...against the longest run a circle carries. A straight IS the
        // degenerate circle, so it wins ties: the piece with the simpler
        // collider and the exact swept rectangle is the better answer whenever
        // it reaches as far.
        std::size_t arcEnd = i;
        ArcFit arc;
        for (std::size_t j = i + 2, missed = 0; j < n && missed <= kAllowedMisses; ++j) {
            ArcFit candidate = fitArc(points, tangents, i, j, tolerance, cosLimit, angles);
            if (candidate.ok) {
                arc = candidate;
                arcEnd = j;
                missed = 0;
            } else {
                ++missed;
            }
        }

        RoadPrimitive primitive;
        if (arc.ok && arcEnd > straightEnd) {
            primitive.kind = RoadPrimitive::Kind::Arc;
            primitive.center = arc.center;
            primitive.radius = arc.radius;
            primitive.startAngle = arc.startAngle;
            primitive.sweep = arc.sweep;
            primitive.start = points[i];
            primitive.end = points[arcEnd];
            i = arcEnd;
        } else {
            primitive.kind = RoadPrimitive::Kind::Straight;
            primitive.start = points[i];
            primitive.end = points[straightEnd];
            i = straightEnd;
        }
        path.primitives_.push_back(primitive);
    }

    return path;
}

RoadPath RoadPath::fromCurve(const Curve3& curve, unsigned int divisions, bool closed,
                             const Params& params) {

    const unsigned int spans = std::max(divisions, 1u);
    std::vector<Vector3> samples(spans + 1);
    for (unsigned int i = 0; i <= spans; ++i) {
        curve.getPoint(static_cast<float>(i) / static_cast<float>(spans), samples[i]);
    }
    return fromPoints(samples, closed, params);
}

float RoadPath::length() const {

    float total = 0.f;
    for (const auto& primitive : primitives_) total += primitive.length();
    return total;
}
