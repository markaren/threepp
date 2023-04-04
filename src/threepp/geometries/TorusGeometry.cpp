
#include "threepp/geometries/TorusGeometry.hpp"

#include <cmath>
#include <list>

using namespace threepp;


TorusGeometry::TorusGeometry(float radius, float tube, unsigned int radialSegments, unsigned int tubularSegments, float arc) {

    // buffers

    std::list<unsigned int> indices;
    std::list<float> vertices;
    std::list<float> normals;
    std::list<float> uvs;

    // helper variables

    Vector3 center;
    Vector3 vertex;
    Vector3 normal;

    // generate vertices, normals and uvs

    for (unsigned j = 0; j <= radialSegments; j++) {

        for (unsigned i = 0; i <= tubularSegments; i++) {

            const auto u = static_cast<float>(i) / static_cast<float>(tubularSegments) * arc;
            const auto v = static_cast<float>(j) / static_cast<float>(radialSegments) * math::TWO_PI;

            // vertex

            vertex.x = (radius + tube * std::cos(v)) * std::cos(u);
            vertex.y = (radius + tube * std::cos(v)) * std::sin(u);
            vertex.z = tube * std::sin(v);

            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // normal

            center.x = radius * std::cos(u);
            center.y = radius * std::sin(u);
            normal.subVectors(vertex, center).normalize();

            normals.insert(normals.end(), {normal.x, normal.y, normal.z});

            // uv

            uvs.emplace_back(static_cast<float>(i) / static_cast<float>(tubularSegments));
            uvs.emplace_back(static_cast<float>(j) / static_cast<float>(radialSegments));
        }
    }

    // generate indices

    for (unsigned j = 1; j <= radialSegments; j++) {

        for (unsigned i = 1; i <= tubularSegments; i++) {

            // indices

            const auto a = (tubularSegments + 1) * j + i - 1;
            const auto b = (tubularSegments + 1) * (j - 1) + i - 1;
            const auto c = (tubularSegments + 1) * (j - 1) + i;
            const auto d = (tubularSegments + 1) * j + i;

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

std::string TorusGeometry::type() const {

    return "TorusGeometry";
}

std::shared_ptr<TorusGeometry> TorusGeometry::create(float radius, float tube, unsigned int radialSegments, unsigned int tubularSegments, float arc) {

    return std::shared_ptr<TorusGeometry>(new TorusGeometry(radius, tube, radialSegments, tubularSegments, arc));
}
