
#include "threepp/extras/curves/RoadGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    // The surface's own layout: station i owns vertices 2i (left) and 2i + 1
    // (right).
    Vector3 vertexAt(const FloatBufferAttribute& position, std::size_t index) {

        const auto i = static_cast<int>(index);
        return {position.getX(i), position.getY(i), position.getZ(i)};
    }

}// namespace


RoadGeometry::RoadGeometry(const Curve3& path, const Params& params)
    : width(params.width) {

    RoadAlignment::Params alignment;
    alignment.width = std::max(params.width, 1e-4f);
    alignment.closed = params.closed;
    alignment.seeds = std::max(params.divisions, 4u);
    alignment.tolerance = params.tolerance;
    alignment.stationAngle = params.stationAngle;
    alignment.minRadiusFactor = params.minRadiusFactor;
    alignment.profileMinRadius = params.profileMinRadius;

    alignment_ = RoadAlignment::build(path, alignment);
    const auto& stations = alignment_.stations();
    // A curve with no length to it — two control points dragged onto each other
    // — has no alignment and therefore no surface.
    if (stations.size() < 2) return;

    const float half = alignment.width * 0.5f;
    const float tile = std::max(params.uvLength, 1e-4f);

    std::vector<float> positions;
    std::vector<float> uvs;
    positions.reserve(stations.size() * 6);
    uvs.reserve(stations.size() * 4);

    for (const auto& station : stations) {
        Vector3 edge;
        edge.copy(station.point).addScaledVector(station.side, -half);
        positions.insert(positions.end(), {edge.x, edge.y, edge.z});
        edge.copy(station.point).addScaledVector(station.side, half);
        positions.insert(positions.end(), {edge.x, edge.y, edge.z});
        const float u = station.distance / tile;
        uvs.insert(uvs.end(), {u, 0.f, u, 1.f});
    }

    // The strip between consecutive cross-sections, wound so the face normal
    // points up. Reversing this leaves the road back-face culled, i.e. invisible
    // from above.
    std::vector<unsigned int> indices;
    indices.reserve((stations.size() - 1) * 6);
    for (std::size_t i = 0; i + 1 < stations.size(); ++i) {
        const auto base = static_cast<unsigned int>(i * 2);
        indices.insert(indices.end(), {base, base + 2, base + 1});
        indices.insert(indices.end(), {base + 2, base + 3, base + 1});
    }

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
    this->computeVertexNormals();
    this->computeBoundingBox();
    this->computeBoundingSphere();
}

std::vector<std::array<Vector3, 8>> RoadGeometry::hulls(const BufferGeometry& surface,
                                                        float thickness) {

    std::vector<std::array<Vector3, 8>> out;

    const auto* position = surface.getAttribute<float>("position");
    if (!position) return out;
    const auto count = static_cast<std::size_t>(position->count());
    // Cross-sections, in pairs. Anything else is not a road surface.
    if (count < 4 || count % 2 != 0) return out;
    const std::size_t rings = count / 2;

    // Each cross-section's own normal, so consecutive hulls share their joint
    // whole — top face and bottom face both — and leave no seam between them.
    std::vector<Vector3> normals(rings);
    for (std::size_t i = 0; i < rings; ++i) {
        const Vector3 left = vertexAt(*position, i * 2);
        const Vector3 right = vertexAt(*position, i * 2 + 1);
        const std::size_t before = i > 0 ? i - 1 : i;
        const std::size_t after = i + 1 < rings ? i + 1 : i;
        Vector3 back, ahead, across, along;
        back.copy(vertexAt(*position, before * 2)).add(vertexAt(*position, before * 2 + 1)).multiplyScalar(0.5f);
        ahead.copy(vertexAt(*position, after * 2)).add(vertexAt(*position, after * 2 + 1)).multiplyScalar(0.5f);
        across.copy(right).sub(left);
        along.copy(ahead).sub(back);
        normals[i].crossVectors(along, across);
        if (normals[i].length() < 1e-9f) normals[i].set(0.f, 1.f, 0.f);
        else normals[i].normalize();
    }

    out.reserve(rings - 1);
    for (std::size_t i = 0; i + 1 < rings; ++i) {
        std::array<Vector3, 8> hull;
        hull[0] = vertexAt(*position, i * 2);
        hull[1] = vertexAt(*position, i * 2 + 1);
        hull[2] = vertexAt(*position, (i + 1) * 2);
        hull[3] = vertexAt(*position, (i + 1) * 2 + 1);
        for (int corner = 0; corner < 4; ++corner) {
            const Vector3& normal = normals[i + static_cast<std::size_t>(corner / 2)];
            hull[4 + corner].copy(hull[corner]).addScaledVector(normal, -thickness);
        }
        // A strip with no length carries no volume, and PhysX will not cook one.
        if (hull[0].distanceTo(hull[2]) < 1e-5f && hull[1].distanceTo(hull[3]) < 1e-5f) continue;
        out.push_back(hull);
    }
    return out;
}

std::string RoadGeometry::type() const {

    return "RoadGeometry";
}

std::shared_ptr<RoadGeometry> RoadGeometry::create(const Curve3& path, const Params& params) {

    return std::shared_ptr<RoadGeometry>(new RoadGeometry(path, params));
}

std::shared_ptr<RoadGeometry> RoadGeometry::create(const Curve3& path, float width,
                                                   unsigned int divisions, float uvLength,
                                                   bool closed) {

    return create(path, Params(width, divisions, uvLength, closed));
}

RoadGeometry::Params::Params(float width, unsigned int divisions, float uvLength, bool closed)
    : width(width), divisions(divisions), uvLength(uvLength), closed(closed) {}
