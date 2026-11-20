
#include "threepp/geometries/LatheGeometry.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

LatheGeometry::LatheGeometry(const std::vector<Vector2>& points, unsigned int segments, float phiStart, float phiLength)
    : parameters(points, segments, phiStart, phiLength) {

    // clamp phiLength so it's in range of [ 0, 2PI ]

    phiLength = std::clamp(phiLength, 0.f, math::TWO_PI);

    // buffers

    std::vector<unsigned int> indices;
    std::vector<float> vertices;
    std::vector<float> uvs;
    std::vector<float> normals;

    // helper variables

    const auto inverseSegments = 1.f / static_cast<float>(segments);
    Vector3 vertex;
    Vector2 uv;

    // The normals of the profile, in its own plane (x along the radius, y along
    // the axis). A segment's normal is its direction turned a quarter turn,
    // (dy, -dx); a point takes the sum of its two segments' normals (the longer
    // segment weighs more), an end point its one segment's. They come from the
    // profile and not from the faces (as in current three.js): the faces
    // that meet a point on the axis are without area on one side, and a normal
    // summed from them is zero on the first or the last line of vertices.

    std::vector<Vector2> profileNormals;
    profileNormals.reserve(points.size());
    Vector2 prevNormal;
    for (std::size_t j = 0; j < points.size(); j++) {

        Vector2 nextNormal;
        if (j + 1 < points.size()) {
            nextNormal.x = points[j + 1].y - points[j].y;
            nextNormal.y = -(points[j + 1].x - points[j].x);
        }

        Vector2 normal(prevNormal.x + nextNormal.x, prevNormal.y + nextNormal.y);
        profileNormals.emplace_back(normal.normalize());
        prevNormal = nextNormal;
    }

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

            // normal: the profile's, turned about the axis with the vertex

            normals.insert(normals.end(), {profileNormals[j].x * sin, profileNormals[j].y, profileNormals[j].x * cos});
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
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
}

const std::string& LatheGeometry::type() const {

    static const std::string typeName = "LatheGeometry";
    return typeName;
}
