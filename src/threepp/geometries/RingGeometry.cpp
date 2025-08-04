
#include "threepp/geometries/RingGeometry.hpp"

#include <cmath>
#include <vector>

using namespace threepp;


RingGeometry::RingGeometry(const Params& params) {

    unsigned int thetaSegments = std::max(3u, params.thetaSegments);
    unsigned int phiSegments = std::max(1u, params.phiSegments);

    std::vector<unsigned int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    // some helper variables

    float radius = params.innerRadius;
    const auto radiusStep = ((params.outerRadius - params.innerRadius) / static_cast<float>(phiSegments));
    Vector3 vertex;
    Vector2 uv;

    // generate vertices, normals and uvs

    for (unsigned j = 0; j <= phiSegments; j++) {

        for (unsigned i = 0; i <= thetaSegments; i++) {

            // values are generate from the inside of the ring to the outside

            const auto segment = params.thetaStart + static_cast<float>(i) / static_cast<float>(thetaSegments) * params.thetaLength;

            // vertex

            vertex.x = radius * std::cos(segment);
            vertex.y = radius * std::sin(segment);

            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // normal

            normals.insert(normals.end(), {0, 0, 1});

            // uv

            uv.x = (vertex.x / params.outerRadius + 1) / 2;
            uv.y = (vertex.y / params.outerRadius + 1) / 2;

            uvs.insert(uvs.end(), {uv.x, uv.y});
        }

        // increase the radius for next row of vertices

        radius += radiusStep;
    }

    // indices

    for (unsigned j = 0; j < phiSegments; j++) {

        const auto thetaSegmentLevel = j * (thetaSegments + 1);

        for (unsigned i = 0; i < thetaSegments; i++) {

            const auto segment = i + thetaSegmentLevel;

            const auto a = segment;
            const auto b = segment + thetaSegments + 1;
            const auto c = segment + thetaSegments + 2;
            const auto d = segment + 1;

            // faces

            indices.insert(indices.end(), {a, b, d});
            indices.insert(indices.end(), {b, c, d});
        }
    }

    // build geometry

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
}

std::string RingGeometry::type() const {

    return "RingGeometry";
}

std::shared_ptr<RingGeometry> RingGeometry::create(const RingGeometry::Params& params) {

    return std::shared_ptr<RingGeometry>(new RingGeometry(params));
}

std::shared_ptr<RingGeometry> RingGeometry::create(float innerRadius, float outerRadius, unsigned int thetaSegments, unsigned int phiSegments, float thetaStart, float thetaLength) {

    return create(Params(innerRadius, outerRadius, thetaSegments, phiSegments, thetaStart, thetaLength));
}

RingGeometry::Params::Params(float innerRadius, float outerRadius, unsigned int thetaSegments, unsigned int phiSegments, float thetaStart, float thetaLength)
    : innerRadius(innerRadius), outerRadius(outerRadius), thetaSegments(thetaSegments), phiSegments(phiSegments), thetaStart(thetaStart), thetaLength(thetaLength) {}
