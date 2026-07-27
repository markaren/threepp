
#include "threepp/extras/curves/RibbonGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace threepp;

namespace {

    // Below this the cross product of up and tangent says nothing about which
    // way is sideways — the curve is running straight up or straight down.
    constexpr float kDegenerate = 1e-4f;

    // Below this |cross(up, tangent)| the horizontal part of the tangent is
    // noise, not signal — sin of about six degrees. The kDegenerate this used
    // to share only tripped within 0.006 degrees of vertical, so any merely
    // steep stretch renormalized numerical noise into the side vector and its
    // azimuth spun freely ring to ring: a fan of inside-out quads on screen.
    constexpr float kLevelable = 0.1f;

    // Ceiling on the miter widening. 1/cos(theta/2) diverges as a corner
    // approaches a hairpin, and a cross-section that spikes to infinity is a
    // worse artefact than the notch it was widening to cover.
    constexpr float kMaxMiter = 2.f;

    // Radius of the circle through three consecutive centres — the local bend.
    // Collinear (or degenerate) points read as straight.
    float circumradius(const threepp::Vector3& a, const threepp::Vector3& b,
                       const threepp::Vector3& c) {

        threepp::Vector3 ab, ac, crossed;
        ab.copy(b).sub(a);
        ac.copy(c).sub(a);
        crossed.copy(ab).cross(ac);
        const float doubledArea = crossed.length();
        if (doubledArea < 1e-8f) return std::numeric_limits<float>::infinity();
        threepp::Vector3 bc;
        bc.copy(c).sub(b);
        return ab.length() * bc.length() * ac.length() / (2.f * doubledArea);
    }

}// namespace

RibbonGeometry::RibbonGeometry(const Curve3& path, const Params& params)
    : width(params.width) {

    const unsigned int spans = std::max(params.divisions, 1u);
    const unsigned int rings = spans + 1;
    const float half = params.width * 0.5f;
    const float tile = std::max(params.uvLength, 1e-4f);

    // Centres first. The frame at a ring is a function of the spans on BOTH
    // sides of it (see the miter below), which the one-sided forward pass this
    // used to be cannot see.
    std::vector<Vector3> centers(rings);
    for (unsigned int i = 0; i < rings; ++i) {
        path.getPoint(static_cast<float>(i) / static_cast<float>(spans), centers[i]);
    }

    const Vector3 up{0, 1, 0};

    // Per-span direction and the level sideways vector that goes with it.
    // Carried across spans so a stretch that runs vertically keeps the sideways
    // direction it had going in instead of flipping to an arbitrary one; also
    // the seed for a curve that starts vertical.
    std::vector<Vector3> dirs(spans);
    std::vector<Vector3> sides(spans);
    {
        Vector3 dir{0, 0, 1};
        Vector3 side{1, 0, 0};
        bool sideLocked = false;
        Vector3 delta;
        Vector3 candidate;
        for (unsigned int i = 0; i < spans; ++i) {
            delta.copy(centers[i + 1]).sub(centers[i]);
            if (delta.length() > kDegenerate) dir.copy(delta).normalize();
            candidate.copy(up).cross(dir);
            if (candidate.length() > kLevelable) {
                candidate.normalize();
                // Continuity outranks azimuth: coming out of a vertical stretch
                // or past a cusp the recomputed side can point anywhere,
                // including mirrored — and a left edge that swaps to the right
                // builds every quad between those rings inside out. The first
                // real side is exempt: the seed above is arbitrary, not a
                // preference to preserve.
                if (sideLocked && candidate.dot(side) < 0.f) candidate.negate();
                side.copy(candidate);
                sideLocked = true;
            }
            dirs[i].copy(dir);
            sides[i].copy(side);
        }
    }

    // Rings are collected before they are emitted: the pinning pass below
    // needs to walk the finished edges.
    std::vector<Vector3> lefts(rings);
    std::vector<Vector3> rights(rings);
    std::vector<Vector3> ringNormals(rings);
    std::vector<float> ringU(rings);

    Vector3 bisector;
    Vector3 side;
    Vector3 tangent;
    Vector3 normal;
    float arc = 0.f;

    for (unsigned int i = 0; i < rings; ++i) {

        const Vector3& center = centers[i];
        if (i > 0) arc += center.distanceTo(centers[i - 1]);

        // The two spans meeting at this ring. Endpoints are one-sided; a closed
        // curve wraps, which also makes the seam ring's frame bit-identical to
        // ring 0's rather than merely close to it.
        const unsigned int inSpan = (i > 0) ? i - 1 : (params.closed ? spans - 1 : 0u);
        const unsigned int outSpan = (i < spans) ? i : (params.closed ? 0u : spans - 1);

        // Miter join. The cross-section bisects the two spans and is widened by
        // 1/cos(theta/2), which is exactly where the two spans' offset edges
        // meet — so both ribbon edges stay CONTINUOUS through the corner.
        // Independent per-sample sides instead let each span's edge overshoot
        // into the next, and the ribbon folds back over itself.
        bisector.copy(sides[inSpan]).add(sides[outSpan]);
        const float cosHalf = bisector.length() * 0.5f;
        float miter = 1.f;
        if (cosHalf > kDegenerate) {
            side.copy(bisector).normalize();
            miter = std::min(1.f / cosHalf, kMaxMiter);
        } else {
            // The spans reverse: there is no bisector, and no widening that
            // would help.
            side.copy(sides[outSpan]);
        }

        tangent.copy(dirs[inSpan]).add(dirs[outSpan]);
        if (tangent.length() > kDegenerate) {
            tangent.normalize();
        } else {
            tangent.copy(dirs[outSpan]);
        }

        // The up component perpendicular to the curve. Equals `up` on the flat
        // and tips with the grade on a climb, which is what keeps a sloping
        // ribbon's faces pointing away from the surface rather than through it.
        normal.copy(tangent).cross(side);
        if (normal.length() > kDegenerate) {
            normal.normalize();
        } else {
            normal.copy(up);
        }

        const float offset = half * miter;
        float offsetLeft = offset;
        float offsetRight = offset;

        // The inner edge of a bend runs BACKWARD as soon as its offset exceeds
        // the local curvature radius (its speed is the centre's times
        // 1 - offset/radius), and everything past that radius is fold. So the
        // INNER offset is capped at the radius: the road narrows smoothly
        // through a bend it cannot carry at full width, hugging the outer side
        // the way a real road squeezes through a tight corner. At the radius
        // itself the inner edges of the two spans kiss — which is exactly the
        // point the miter above aims for — so a corner that fits is untouched.
        const bool interior = params.closed || (i > 0 && i < spans);
        if (interior) {
            const Vector3& before = centers[(i > 0) ? i - 1 : spans - 1];
            const Vector3& after = centers[(i < spans) ? i + 1 : 1];
            const float radius = circumradius(before, center, after);
            if (offset > radius) {
                // Which side is inside the bend: the one the curve caves
                // toward. `rights` sits at +side.
                Vector3 concave;
                concave.copy(before).add(after).multiplyScalar(0.5f).sub(center);
                const float caved = concave.dot(side);
                if (caved > 0.f) {
                    offsetRight = radius;
                } else if (caved < 0.f) {
                    offsetLeft = radius;
                }
            }
        }

        lefts[i].copy(center).addScaledVector(side, -offsetLeft);
        rights[i].copy(center).addScaledVector(side, offsetRight);
        ringNormals[i].copy(normal);
        ringU[i] = arc / tile;
    }

    // Inside a bend tighter than the half-width the inner offset edge runs
    // BACKWARD — its speed is the centre's times (1 - offset/radius) — so
    // every quad in the stretch crosses itself and renders inside out. There
    // is no width to recover in there: a backward step is pinned to the vertex
    // before it instead, closing the notch into a pivot the way a road pinches
    // its inner shoulder around a hairpin. The pivot's zero-area triangles
    // render as nothing, which is the point.
    {
        Vector3 forward;
        Vector3 step;
        for (auto* edge : {&lefts, &rights}) {
            auto& points = *edge;
            for (unsigned int i = 1; i < rings; ++i) {
                forward.copy(centers[i]).sub(centers[i - 1]);
                if (forward.length() <= kDegenerate) continue;
                step.copy(points[i]).sub(points[i - 1]);
                if (step.dot(forward) < 0.f) points[i].copy(points[i - 1]);
            }
        }
    }

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<unsigned int> indices;
    positions.reserve(rings * 2 * 3);
    normals.reserve(rings * 2 * 3);
    uvs.reserve(rings * 2 * 2);
    indices.reserve(spans * 6);

    for (unsigned int i = 0; i < rings; ++i) {
        positions.insert(positions.end(), {lefts[i].x, lefts[i].y, lefts[i].z});
        normals.insert(normals.end(), {ringNormals[i].x, ringNormals[i].y, ringNormals[i].z});
        uvs.insert(uvs.end(), {ringU[i], 0.f});
        positions.insert(positions.end(), {rights[i].x, rights[i].y, rights[i].z});
        normals.insert(normals.end(), {ringNormals[i].x, ringNormals[i].y, ringNormals[i].z});
        uvs.insert(uvs.end(), {ringU[i], 1.f});
    }

    // A closed curve's last ring is positionally identical to its first, so the
    // loop closes without a gap — but the two rings stay SEPARATE vertices on
    // purpose: the last needs u = totalLength/uvLength where the first needs
    // u = 0, and sharing them would wrap the texture back to the start across
    // the final span. A seam duplicate is the price of continuous UVs.
    for (unsigned int i = 0; i < spans; ++i) {
        const unsigned int a = i * 2;
        const unsigned int b = a + 1;
        const unsigned int c = a + 2;
        const unsigned int d = a + 3;
        // Wound so the face normal is `normal` (+Y on the flat): reversing this
        // leaves the ribbon back-face culled, i.e. invisible from above.
        indices.insert(indices.end(), {a, c, b});
        indices.insert(indices.end(), {b, c, d});
    }

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
    this->computeBoundingBox();
    this->computeBoundingSphere();
}

std::string RibbonGeometry::type() const {

    return "RibbonGeometry";
}

std::shared_ptr<RibbonGeometry> RibbonGeometry::create(const Curve3& path, const Params& params) {

    return std::shared_ptr<RibbonGeometry>(new RibbonGeometry(path, params));
}

std::shared_ptr<RibbonGeometry> RibbonGeometry::create(const Curve3& path, float width,
                                                       unsigned int divisions, float uvLength,
                                                       bool closed) {

    return create(path, Params(width, divisions, uvLength, closed));
}

RibbonGeometry::Params::Params(float width, unsigned int divisions, float uvLength, bool closed)
    : width(width), divisions(divisions), uvLength(uvLength), closed(closed) {}
