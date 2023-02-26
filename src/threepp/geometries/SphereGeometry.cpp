
#include "threepp/geometries/SphereGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <list>
#include <vector>

using namespace threepp;

SphereGeometry::SphereGeometry(
        float radius,
        unsigned int widthSegments,
        unsigned int heightSegments,
        float phiStart,
        float phiLength,
        float thetaStart,
        float thetaLength)
    : radius(radius) {

    std::list<unsigned int> indices;
    std::list<float> vertices;
    std::list<float> normals;
    std::list<float> uvs;

    widthSegments = std::max(3u, widthSegments);
    heightSegments = std::max(2u, heightSegments);

    const auto thetaEnd = std::min(thetaStart + thetaLength, math::PI);

    unsigned int index = 0;
    std::vector<std::vector<unsigned int>> grid;

    Vector3 vertex;
    Vector3 normal;

    // generate vertices, normals and uvs

    for (unsigned iy = 0; iy <= heightSegments; iy++) {

        std::vector<unsigned int> verticesRow;

        const float v = static_cast<float>(iy) / static_cast<float>(heightSegments);

        // special case for the poles

        float uOffset = 0;

        if (iy == 0 && thetaStart == 0) {

            uOffset = 0.5f / static_cast<float>(widthSegments);

        } else if (iy == heightSegments && thetaEnd == math::PI) {

            uOffset = -0.5f / static_cast<float>(widthSegments);
        }

        for (unsigned ix = 0; ix <= widthSegments; ix++) {

            const float u = static_cast<float>(ix) / static_cast<float>(widthSegments);

            // vertex

            vertex.x = -radius * std::cos(phiStart + u * phiLength) * std::sin(thetaStart + v * thetaLength);
            vertex.y = radius * std::cos(thetaStart + v * thetaLength);
            vertex.z = radius * std::sin(phiStart + u * phiLength) * std::sin(thetaStart + v * thetaLength);

            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // normal

            normal.copy(vertex).normalize();
            normals.insert(normals.end(), {normal.x, normal.y, normal.z});

            // uv

            uvs.insert(uvs.end(), {u + uOffset, 1 - v});

            verticesRow.emplace_back(index++);
        }

        grid.emplace_back(verticesRow);
    }
    // indices

    for (int iy = 0; iy < heightSegments; iy++) {

        for (int ix = 0; ix < widthSegments; ix++) {

            const auto a = grid[iy][ix + 1];
            const auto b = grid[iy][ix];
            const auto c = grid[iy + 1][ix];
            const auto d = grid[iy + 1][ix + 1];

            if (iy != 0 || thetaStart > 0) indices.insert(indices.end(), {a, b, d});
            if (iy != heightSegments - 1 || thetaEnd < math::PI) indices.insert(indices.end(), {b, c, d});
        }
    }

    // build geometry

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
}
