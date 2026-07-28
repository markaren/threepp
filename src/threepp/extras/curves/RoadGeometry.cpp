
#include "threepp/extras/curves/RoadGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kEpsilon = 1e-6f;

    // One cross-section of the road: a segment through `center` along `side`,
    // reaching `leftOffset` one way and `rightOffset` the other. The two
    // offsets differ only on an arc, where the inner one is capped so the inner
    // edge stops AT the centre of the bend instead of running back through it.
    struct Ring {
        Vector3 center;
        Vector3 side;
        float leftOffset = 0.f;
        float rightOffset = 0.f;
        Vector3 normal;
        float u = 0.f;
    };

    // Level sideways vector for a horizontal tangent, or `fallback` where the
    // tangent has no horizontal part to take one from.
    Vector3 sideOf(const Vector3& tangent, const Vector3& fallback) {

        Vector3 side{tangent.z, 0.f, -tangent.x};// cross((0,1,0), tangent)
        const float length = side.length();
        if (length < kEpsilon) return fallback;
        return side.multiplyScalar(1.f / length);
    }

}// namespace


RoadGeometry::RoadGeometry(const RoadPath& path, const Params& params)
    : width(params.width) {

    const auto& primitives = path.primitives();
    if (primitives.empty()) return;

    const float half = params.width * 0.5f;
    const float tile = std::max(params.uvLength, 1e-4f);
    const float step = std::clamp(params.angularStep, 0.01f, math::PI);

    // Every piece is built in ITS OWN exact frame — perpendicular
    // cross-sections along a straight, radial ones around an arc. Nothing is
    // averaged across a joint, which is what keeps a cross-section from being
    // rotated off the annulus it belongs to: near the centre of a tight bend a
    // degree of skew is most of a radius, and the inner edge folds. The two
    // pieces meeting at a joint carry a cross-section each; the quad between
    // them covers the wedge the turn opens on the outside (see the winding).
    std::vector<Ring> rings;
    rings.reserve(primitives.size() * 8);

    const Vector3 up{0, 1, 0};
    Vector3 carried{1, 0, 0};
    float travelled = 0.f;

    for (const auto& primitive : primitives) {

        const float length = primitive.length();
        const float horizontal = primitive.horizontalLength();
        // Rise per metre of ground covered — what tips the surface normal on a
        // climb, and zero on the flat.
        const float grade = horizontal > kEpsilon ? (primitive.end.y - primitive.start.y) / horizontal : 0.f;

        const int steps = primitive.kind == RoadPrimitive::Kind::Arc
                                  ? std::max(1, static_cast<int>(std::ceil(std::abs(primitive.sweep) / step)))
                                  : 1;

        for (int k = 0; k <= steps; ++k) {

            const float t = static_cast<float>(k) / static_cast<float>(steps);
            const Vector3 tangent = primitive.tangentAt(t);

            Ring ring;
            ring.center = primitive.pointAt(t);
            ring.side = sideOf(tangent, carried);
            carried.copy(ring.side);

            ring.leftOffset = half;
            ring.rightOffset = half;
            if (primitive.kind == RoadPrimitive::Kind::Arc) {
                // `side` points outward on a left-hand sweep and inward on a
                // right-hand one, so the cap lands on whichever vertex is the
                // inner one. Capping it at the radius is what turns an annulus
                // into a pie sector when the bend is tighter than the offset:
                // the inner vertices of the whole run collapse onto the centre,
                // and the triangles between them have no area rather than
                // negative area.
                const float inner = std::min(half, primitive.radius);
                if (primitive.sweep >= 0.f) {
                    ring.leftOffset = inner;
                } else {
                    ring.rightOffset = inner;
                }
            }

            Vector3 tangent3d;
            tangent3d.copy(tangent).addScaledVector(up, grade);
            if (tangent3d.length() > kEpsilon) {
                tangent3d.normalize();
                ring.normal.copy(tangent3d).cross(ring.side);
            }
            if (ring.normal.length() < kEpsilon) {
                ring.normal.copy(up);
            } else {
                ring.normal.normalize();
            }

            ring.u = (travelled + length * t) / tile;
            rings.push_back(ring);
        }

        travelled += length;
    }

    // A closed road's seam is a joint like any other: it needs the wedge quad
    // between the last piece's cross-section and the first piece's, and the
    // duplicate is what carries u = totalLength/uvLength where the original
    // carries u = 0. Sharing them instead would wrap the texture back to the
    // start across the seam.
    if (path.closed()) {
        Ring seam = rings.front();
        seam.u = travelled / tile;
        rings.push_back(seam);
    }

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<unsigned int> indices;
    positions.reserve(rings.size() * 6);
    normals.reserve(rings.size() * 6);
    uvs.reserve(rings.size() * 4);
    indices.reserve(rings.size() * 6);

    Vector3 left, right;
    for (const auto& ring : rings) {
        left.copy(ring.center).addScaledVector(ring.side, -ring.leftOffset);
        right.copy(ring.center).addScaledVector(ring.side, ring.rightOffset);
        positions.insert(positions.end(), {left.x, left.y, left.z});
        normals.insert(normals.end(), {ring.normal.x, ring.normal.y, ring.normal.z});
        uvs.insert(uvs.end(), {ring.u, 0.f});
        positions.insert(positions.end(), {right.x, right.y, right.z});
        normals.insert(normals.end(), {ring.normal.x, ring.normal.y, ring.normal.z});
        uvs.insert(uvs.end(), {ring.u, 1.f});
    }

    // Wound so the face normal points along the cross-section's normal (+Y on
    // the flat): reversing that leaves the road back-face culled, i.e.
    // invisible from above.
    //
    // Within a piece the order below is already that winding. Across a JOINT it
    // need not be: the two cross-sections there sit on the same centreline
    // point rotated by whatever kink the fit left, so they meet in the middle
    // and the quad between them is two wedges — one covering the gap the turn
    // opens on the outside, one doubling the piece beside it on the inside. A
    // wedge that comes out facing down is turned over rather than dropped: the
    // doubled one is coplanar with the road, same normal and same material, so
    // it is invisible, where dropping it would leave the other side a hole.
    const auto vertexAt = [&positions](unsigned int i) {
        return Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    };
    const auto emit = [&](unsigned int a, unsigned int b, unsigned int c, const Vector3& reference) {
        Vector3 ab, ac;
        const Vector3 origin = vertexAt(a);
        ab.copy(vertexAt(b)).sub(origin);
        ac.copy(vertexAt(c)).sub(origin);
        if (ab.cross(ac).dot(reference) < 0.f) std::swap(b, c);
        indices.insert(indices.end(), {a, b, c});
    };

    for (std::size_t i = 0; i + 1 < rings.size(); ++i) {
        const auto a = static_cast<unsigned int>(i * 2);
        const unsigned int b = a + 1;
        const unsigned int c = a + 2;
        const unsigned int d = a + 3;
        Vector3 reference;
        reference.copy(rings[i].normal).add(rings[i + 1].normal);
        if (reference.length() < kEpsilon) reference.copy(rings[i].normal);
        emit(a, c, b, reference);
        emit(b, c, d, reference);
    }

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
    this->computeBoundingBox();
    this->computeBoundingSphere();
}

std::string RoadGeometry::type() const {

    return "RoadGeometry";
}

std::shared_ptr<RoadGeometry> RoadGeometry::create(const RoadPath& path, const Params& params) {

    return std::shared_ptr<RoadGeometry>(new RoadGeometry(path, params));
}

std::shared_ptr<RoadGeometry> RoadGeometry::create(const RoadPath& path, float width, float uvLength) {

    return create(path, Params(width, uvLength));
}

RoadGeometry::Params::Params(float width, float uvLength, float angularStep)
    : width(width), uvLength(uvLength), angularStep(angularStep) {}
