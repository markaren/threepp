
#include "threepp/geometries/RoundedBoxGeometry.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    float sign(float v) {
        return static_cast<float>((v > 0.f) - (v < 0.f));
    }

    float angleBetween(const Vector3& a, const Vector3& b) {
        const float denominator = std::sqrt(a.lengthSq() * b.lengthSq());
        if (denominator == 0.f) return math::PI / 2;
        const float theta = a.dot(b) / denominator;
        return std::acos(std::clamp(theta, -1.f, 1.f));
    }

    // Where along a face a vertex lies, in uv space split between the two
    // quarter-circle edges and the flat stretch between them.
    float getUv(const Vector3& faceDirVector, const Vector3& normal, int uvAxis, int projectionAxis, float radius, float sideLength) {

        const float totArcLength = 2 * math::PI * radius / 4;

        // length of the plane between the arcs on this axis
        const float centerLength = std::max(sideLength - 2 * radius, 0.f);
        const float halfArc = math::PI / 4;

        // the normal projected onto the plane across projectionAxis
        Vector3 tempNormal = normal;
        tempNormal[projectionAxis] = 0;
        tempNormal.normalize();

        // the uv space given to a single arc
        const float arcUvRatio = 0.5f * totArcLength / (totArcLength + centerLength);

        // how far along one arc the point is
        const float arcAngleRatio = 1.f - (angleBetween(tempNormal, faceDirVector) / halfArc);

        if (sign(tempNormal[uvAxis]) == 1.f) {

            return arcAngleRatio * arcUvRatio;
        }

        // the uv space given to the plane between the arcs
        const float lenUv = centerLength / (totArcLength + centerLength);
        return lenUv + arcUvRatio + arcUvRatio * (1.f - arcAngleRatio);
    }

    constexpr int X = 0, Y = 1, Z = 2;

}// namespace

RoundedBoxGeometry::RoundedBoxGeometry(const Params& params)
    : parameters(params) {

    const float width = params.width;
    const float height = params.height;
    const float depth = params.depth;
    const float radius = params.radius;

    // an odd number of segments, so that a flat plane joins the rounded edges
    const unsigned int totalSegments = params.segments * 2 + 1;

    const auto box = BoxGeometry::create(1, 1, 1, totalSegments, totalSegments, totalSegments);

    if (totalSegments == 1) {
        // one segment is a plain box (three.js leaves it at unit size)
        setIndex(box->getIndex()->array());
        setAttribute("position", FloatBufferAttribute::create(box->getAttribute<float>("position")->array(), 3));
        setAttribute("normal", FloatBufferAttribute::create(box->getAttribute<float>("normal")->array(), 3));
        setAttribute("uv", FloatBufferAttribute::create(box->getAttribute<float>("uv")->array(), 2));
        for (const auto& group : box->groups) addGroup(group.start, group.count, group.materialIndex);
        return;
    }

    const auto flat = box->toNonIndexed();
    std::vector<float> positions = flat->getAttribute<float>("position")->array();
    std::vector<float> normals = flat->getAttribute<float>("normal")->array();
    std::vector<float> uvs = flat->getAttribute<float>("uv")->array();

    const Vector3 inner = Vector3(width, height, depth).divideScalar(2).subScalar(radius);

    const size_t faceTris = positions.size() / 6;
    const float halfSegmentSize = 0.5f / static_cast<float>(totalSegments);

    Vector3 position, normal, faceDirVector;
    for (size_t i = 0, j = 0; i < positions.size(); i += 3, j += 2) {

        position.set(positions[i], positions[i + 1], positions[i + 2]);
        normal.copy(position);
        normal.x -= sign(normal.x) * halfSegmentSize;
        normal.y -= sign(normal.y) * halfSegmentSize;
        normal.z -= sign(normal.z) * halfSegmentSize;
        normal.normalize();

        positions[i + 0] = inner.x * sign(position.x) + normal.x * radius;
        positions[i + 1] = inner.y * sign(position.y) + normal.y * radius;
        positions[i + 2] = inner.z * sign(position.z) + normal.z * radius;

        normals[i + 0] = normal.x;
        normals[i + 1] = normal.y;
        normals[i + 2] = normal.z;

        const size_t side = i / faceTris;

        switch (side) {

            case 0:// right: uvs along z, then y
                faceDirVector.set(1, 0, 0);
                uvs[j + 0] = getUv(faceDirVector, normal, Z, Y, radius, depth);
                uvs[j + 1] = 1.f - getUv(faceDirVector, normal, Y, Z, radius, height);
                break;

            case 1:// left: uvs along z, then y
                faceDirVector.set(-1, 0, 0);
                uvs[j + 0] = 1.f - getUv(faceDirVector, normal, Z, Y, radius, depth);
                uvs[j + 1] = 1.f - getUv(faceDirVector, normal, Y, Z, radius, height);
                break;

            case 2:// top: uvs along x, then z
                faceDirVector.set(0, 1, 0);
                uvs[j + 0] = 1.f - getUv(faceDirVector, normal, X, Z, radius, width);
                uvs[j + 1] = getUv(faceDirVector, normal, Z, X, radius, depth);
                break;

            case 3:// bottom: uvs along x, then z
                faceDirVector.set(0, -1, 0);
                uvs[j + 0] = 1.f - getUv(faceDirVector, normal, X, Z, radius, width);
                uvs[j + 1] = 1.f - getUv(faceDirVector, normal, Z, X, radius, depth);
                break;

            case 4:// front: uvs along x, then y
                faceDirVector.set(0, 0, 1);
                uvs[j + 0] = 1.f - getUv(faceDirVector, normal, X, Y, radius, width);
                uvs[j + 1] = 1.f - getUv(faceDirVector, normal, Y, X, radius, height);
                break;

            case 5:// back: uvs along x, then y
                faceDirVector.set(0, 0, -1);
                uvs[j + 0] = getUv(faceDirVector, normal, X, Y, radius, width);
                uvs[j + 1] = 1.f - getUv(faceDirVector, normal, Y, X, radius, height);
                break;

            default:
                break;
        }
    }

    setAttribute("position", FloatBufferAttribute::create(std::move(positions), 3));
    setAttribute("normal", FloatBufferAttribute::create(std::move(normals), 3));
    setAttribute("uv", FloatBufferAttribute::create(std::move(uvs), 2));
    for (const auto& group : box->groups) addGroup(group.start, group.count, group.materialIndex);
}

const std::string& RoundedBoxGeometry::type() const {

    static const std::string typeName = "RoundedBoxGeometry";
    return typeName;
}

std::shared_ptr<RoundedBoxGeometry> RoundedBoxGeometry::create(const Params& params) {

    Params p = params;
    // the radius is never more than half the shortest side
    p.radius = std::min({p.width / 2, p.height / 2, p.depth / 2, p.radius});
    return std::shared_ptr<RoundedBoxGeometry>(new RoundedBoxGeometry(p));
}

std::shared_ptr<RoundedBoxGeometry> RoundedBoxGeometry::create(float width, float height, float depth, unsigned int segments, float radius) {

    return create(Params(width, height, depth, segments, radius));
}

RoundedBoxGeometry::Params::Params(float width, float height, float depth, unsigned int segments, float radius)
    : width(width), height(height), depth(depth), segments(segments), radius(radius) {}
