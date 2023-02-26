
#include "threepp/geometries/CircleGeometry.hpp"

#include <list>

using namespace threepp;


CircleGeometry::CircleGeometry(float radius, unsigned int segments, float thetaStart, float thetaLength) {

    // buffers

    std::list<unsigned int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::list<float> uvs;

    // helper variables

    Vector3 vertex;
    Vector2 uv;

    // center point

    vertices.insert(vertices.end(), {0, 0, 0});
    normals.insert(normals.end(), {0, 0, 1});
    uvs.insert(uvs.end(), {0.5f, 0.5f});

    for (unsigned s = 0, i = 3; s <= segments; s++, i += 3) {

        const auto segment = thetaStart + static_cast<float>(s) / static_cast<float>(segments) * thetaLength;

        // vertex

        vertex.x = radius * std::cos(segment);
        vertex.y = radius * std::sin(segment);

        vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

        // normal

        normals.insert(normals.end(), {0, 0, 1});

        // uvs

        uv.x = (vertices[i] / radius + 1) / 2;
        uv.y = (vertices[i + 1] / radius + 1) / 2;

        uvs.insert(uvs.end(), {uv.x, uv.y});
    }

    // indices

    for (unsigned i = 1; i <= segments; i++) {

        indices.insert(indices.end(), {i, i + 1, 0});
    }

    // build geometry

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute ::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute ::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute ::create(uvs, 2));
}
