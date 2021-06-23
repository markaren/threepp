
#include "threepp/geometries/SphereGeometry.hpp"

#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;

SphereGeometry::SphereGeometry(float radius, int widthSegments, int heightSegments, float phiStart, float phiLength, float thetaStart, float thetaLength) {

    std::vector<int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    widthSegments = std::max(3, (int) std::floor(widthSegments));
    heightSegments = std::max(2, (int) std::floor(heightSegments));

    const auto thetaEnd = std::min(thetaStart + thetaLength, PI);

    int index = 0;
    std::vector<std::vector<int>> grid;

    auto vertex = Vector3();
    auto normal = Vector3();

    // generate vertices, normals and uvs

    for (int iy = 0; iy <= heightSegments; iy++) {

        std::vector<int> verticesRow;

        const auto v = iy / heightSegments;

        // special case for the poles

        int uOffset = 0;

        if (iy == 0 && thetaStart == 0) {

            uOffset = (int) 0.5 / widthSegments;

        } else if (iy == heightSegments && thetaEnd == PI) {

            uOffset = (int) -0.5 / widthSegments;
        }

        for (int ix = 0; ix <= widthSegments; ix++) {

            const auto u = ix / widthSegments;

            // vertex

            vertex.x = -radius * std::cos(phiStart + u * phiLength) * std::sin(thetaStart + v * thetaLength);
            vertex.y = radius * std::cos(thetaStart + v * thetaLength);
            vertex.z = radius * std::sin(phiStart + u * phiLength) * std::sin(thetaStart + v * thetaLength);

            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // normal

            normal.copy(vertex).normalize();
            normals.insert(normals.end(), {normal.x, normal.y, normal.z});

            // uv

            uvs.insert(uvs.end(), {(float) u + uOffset, (float) 1 - v});

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
            if (iy != heightSegments - 1 || thetaEnd < PI) indices.insert(indices.end(), {b, c, d});
        }
    }

    // build geometry

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
}
