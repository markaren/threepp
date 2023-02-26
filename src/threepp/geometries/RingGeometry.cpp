
#include "threepp/geometries/RingGeometry.hpp"

#include <cmath>
#include <list>

using namespace threepp;


RingGeometry::RingGeometry(float innerRadius, float outerRadius, unsigned int thetaSegments, unsigned int phiSegments, float thetaStart, float thetaLength) {

    thetaSegments = std::max(3u, thetaSegments);
    phiSegments = std::max(1u, phiSegments);

    std::list<unsigned int> indices;
    std::list<float> vertices;
    std::list<float> normals;
    std::list<float> uvs;

    // some helper variables

    float radius = innerRadius;
    const auto radiusStep = ((outerRadius - innerRadius) / static_cast<float>(phiSegments));
    Vector3 vertex;
    Vector2 uv;

    // generate vertices, normals and uvs

    for (unsigned j = 0; j <= phiSegments; j++) {

        for (unsigned i = 0; i <= thetaSegments; i++) {

            // values are generate from the inside of the ring to the outside

            const auto segment = thetaStart + static_cast<float>(i) / static_cast<float>(thetaSegments) * thetaLength;

            // vertex

            vertex.x = radius * std::cos(segment);
            vertex.y = radius * std::sin(segment);

            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // normal

            normals.insert(normals.end(), {0, 0, 1});

            // uv

            uv.x = (vertex.x / outerRadius + 1) / 2;
            uv.y = (vertex.y / outerRadius + 1) / 2;

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
