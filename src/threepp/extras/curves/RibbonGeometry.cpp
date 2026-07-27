
#include "threepp/extras/curves/RibbonGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    // Below this the cross product of up and tangent says nothing about which
    // way is sideways — the curve is running straight up or straight down.
    constexpr float kDegenerate = 1e-4f;

    // Ceiling on the miter widening. 1/cos(theta/2) diverges as a corner
    // approaches a hairpin, and a cross-section that spikes to infinity is a
    // worse artefact than the notch it was widening to cover.
    constexpr float kMaxMiter = 2.f;

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
        Vector3 delta;
        Vector3 candidate;
        for (unsigned int i = 0; i < spans; ++i) {
            delta.copy(centers[i + 1]).sub(centers[i]);
            if (delta.length() > kDegenerate) dir.copy(delta).normalize();
            candidate.copy(up).cross(dir);
            if (candidate.length() > kDegenerate) side.copy(candidate).normalize();
            dirs[i].copy(dir);
            sides[i].copy(side);
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
        const float u = arc / tile;
        for (int sign = -1; sign <= 1; sign += 2) {
            positions.insert(positions.end(),
                             {center.x + side.x * offset * static_cast<float>(sign),
                              center.y + side.y * offset * static_cast<float>(sign),
                              center.z + side.z * offset * static_cast<float>(sign)});
            normals.insert(normals.end(), {normal.x, normal.y, normal.z});
            uvs.insert(uvs.end(), {u, sign < 0 ? 0.f : 1.f});
        }
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
