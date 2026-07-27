
#include "threepp/extras/curves/RibbonGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    // Below this the cross product of up and tangent says nothing about which
    // way is sideways — the curve is running straight up or straight down.
    constexpr float kDegenerate = 1e-4f;

}// namespace

RibbonGeometry::RibbonGeometry(const Curve3& path, const Params& params)
    : width(params.width) {

    const unsigned int spans = std::max(params.divisions, 1u);
    const unsigned int rings = spans + 1;
    const float half = params.width * 0.5f;
    const float tile = std::max(params.uvLength, 1e-4f);

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<unsigned int> indices;
    positions.reserve(rings * 2 * 3);
    normals.reserve(rings * 2 * 3);
    uvs.reserve(rings * 2 * 2);
    indices.reserve(spans * 6);

    const Vector3 up{0, 1, 0};
    // Carried across samples so a stretch that runs vertically keeps the
    // sideways direction it had going in instead of flipping to an arbitrary
    // one; also the seed for a curve that starts vertical.
    Vector3 side{1, 0, 0};

    Vector3 center;
    Vector3 previous;
    Vector3 tangent;
    Vector3 normal;
    Vector3 firstSide;
    Vector3 firstNormal;
    float arc = 0.f;

    for (unsigned int i = 0; i < rings; ++i) {

        const float t = static_cast<float>(i) / static_cast<float>(spans);
        path.getPoint(t, center);
        if (i > 0) arc += center.distanceTo(previous);
        previous.copy(center);

        // A closed curve is welded, not merely joined: getPoint(1) already
        // lands on getPoint(0), but getTangent's one-sided difference at the
        // ends does not agree with itself to the last bit. Reusing the first
        // frame outright makes the seam exact.
        const bool seam = params.closed && i == spans;
        if (seam) {
            side.copy(firstSide);
            normal.copy(firstNormal);
        } else {
            path.getTangent(t, tangent);
            Vector3 candidate;
            candidate.copy(up).cross(tangent);
            if (candidate.length() > kDegenerate) {
                side.copy(candidate).normalize();
            }

            // The up component perpendicular to the curve. Equals `up` on the
            // flat and tips with the grade on a climb, which is what keeps a
            // sloping ribbon's faces pointing away from the surface rather
            // than through it.
            normal.copy(tangent).cross(side);
            if (normal.length() > kDegenerate) {
                normal.normalize();
            } else {
                normal.copy(up);
            }
        }
        if (i == 0) {
            firstSide.copy(side);
            firstNormal.copy(normal);
        }

        const float u = arc / tile;
        for (int sign = -1; sign <= 1; sign += 2) {
            positions.insert(positions.end(),
                             {center.x + side.x * half * static_cast<float>(sign),
                              center.y + side.y * half * static_cast<float>(sign),
                              center.z + side.z * half * static_cast<float>(sign)});
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
