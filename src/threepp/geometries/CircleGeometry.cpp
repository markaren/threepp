
#include "threepp/geometries/CircleGeometry.hpp"

#include <cmath>
#include <list>

using namespace threepp;


CircleGeometry::CircleGeometry(const Params& params) {

    // buffers

    std::list<unsigned int> indices;
    std::vector<float> vertices;
    std::list<float> normals;
    std::list<float> uvs;

    // helper variables

    Vector3 vertex;
    Vector2 uv;

    // center point

    vertices.insert(vertices.end(), {0, 0, 0});
    normals.insert(normals.end(), {0, 0, 1});
    uvs.insert(uvs.end(), {0.5f, 0.5f});

    for (unsigned s = 0, i = 3; s <= params.segments; s++, i += 3) {

        const auto segment = params.thetaStart + static_cast<float>(s) / static_cast<float>(params.segments) * params.thetaLength;

        // vertex

        vertex.x = params.radius * std::cos(segment);
        vertex.y = params.radius * std::sin(segment);

        vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

        // normal

        normals.insert(normals.end(), {0, 0, 1});

        // uvs

        uv.x = (vertices[i] / params.radius + 1) / 2;
        uv.y = (vertices[i + 1] / params.radius + 1) / 2;

        uvs.insert(uvs.end(), {uv.x, uv.y});
    }

    // indices

    for (unsigned i = 1; i <= params.segments; i++) {

        indices.insert(indices.end(), {i, i + 1, 0});
    }

    // build geometry

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute ::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute ::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute ::create(uvs, 2));
}

std::string CircleGeometry::type() const {

    return "CircleGeometry";
}

std::shared_ptr<CircleGeometry> CircleGeometry::create(const CircleGeometry::Params& params) {

    return std::shared_ptr<CircleGeometry>(new CircleGeometry(params));
}

std::shared_ptr<CircleGeometry> CircleGeometry::create(float radius, unsigned int segments, float thetaStart, float thetaLength) {

    return create(Params(radius, segments, thetaStart, thetaLength));
}
