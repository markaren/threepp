
#include "threepp/geometries/LatheGeometry.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

LatheGeometry::LatheGeometry(const std::vector<Vector2>& points, unsigned int segments, float phiStart, float phiLength) {

    // clamp phiLength so it's in range of [ 0, 2PI ]

    phiLength = std::clamp(phiLength, 0.f, math::TWO_PI);

    // buffers

    std::vector<unsigned int> indices;
    std::vector<float> vertices;
    std::vector<float> uvs;

    // helper variables

    const auto inverseSegments = 1.f / static_cast<float>(segments);
    Vector3 vertex;
    Vector2 uv;

    // generate vertices and uvs

    for (unsigned i = 0; i <= segments; i++) {

        const auto phi = phiStart + static_cast<float>(i) * inverseSegments * phiLength;

        const auto sin = std::sin(phi);
        const auto cos = std::cos(phi);

        for (unsigned j = 0; j <= (points.size() - 1); j++) {

            // vertex

            vertex.x = points[j].x * sin;
            vertex.y = points[j].y;
            vertex.z = points[j].x * cos;

            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // uv

            uv.x = static_cast<float>(i) / static_cast<float>(segments);
            uv.y = static_cast<float>(j) / static_cast<float>((points.size() - 1));

            uvs.insert(uvs.end(), {uv.x, uv.y});
        }
    }

    // indices

    for (unsigned i = 0; i < segments; i++) {

        for (unsigned j = 0; j < (points.size() - 1); j++) {

            const auto base = j + i * points.size();

            const unsigned int a = base;
            const unsigned int b = base + points.size();
            const unsigned int c = base + points.size() + 1;
            const unsigned int d = base + 1;

            // faces

            indices.insert(indices.end(), {a, b, d});
            indices.insert(indices.end(), {b, c, d});
        }
    }

    // build geometry

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));

    // generate normals

    this->computeVertexNormals();

    // if the geometry is closed, we need to average the normals along the seam.
    // because the corresponding vertices are identical (but still have different UVs).

    if (phiLength == math::TWO_PI) {// TODO approx?

        auto& normals = getAttribute<float>("normal")->array();
        Vector3 n1;
        Vector3 n2;
        Vector3 n;

        // this is the buffer offset for the last line of vertices

        const auto base = segments * points.size() * 3;

        for (unsigned i = 0, j = 0; i < points.size(); i++, j += 3) {

            // select the normal of the vertex in the first line

            n1.x = normals[j + 0];
            n1.y = normals[j + 1];
            n1.z = normals[j + 2];

            // select the normal of the vertex in the last line

            n2.x = normals[base + j + 0];
            n2.y = normals[base + j + 1];
            n2.z = normals[base + j + 2];

            // average normals

            n.addVectors(n1, n2).normalize();

            // assign the new values to both normals

            normals[j + 0] = normals[base + j + 0] = n.x;
            normals[j + 1] = normals[base + j + 1] = n.y;
            normals[j + 2] = normals[base + j + 2] = n.z;
        }
    }
}

std::string LatheGeometry::type() const {

    return "LatheGeometry";
}
