
#include "threepp/geometries/PlaneGeometry.hpp"

#include <list>

using namespace threepp;


PlaneGeometry::PlaneGeometry(const Params& params)
    : width(params.width), height(params.height), widthSegments(params.widthSegments), heightSegments(params.heightSegments) {

    const auto width_half = width / 2;
    const auto height_half = height / 2;

    const auto gridX = params.widthSegments;
    const auto gridY = params.heightSegments;

    const auto gridX1 = gridX + 1;
    const auto gridY1 = gridY + 1;

    const auto segment_width = width / static_cast<float>(gridX);
    const auto segment_height = height / static_cast<float>(gridY);

    //

    std::vector<unsigned int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    vertices.reserve(gridX1 * gridY1 * 3);
    normals.reserve(gridX1 * gridY1 * 3);
    uvs.reserve(gridX1 * gridY1 * 2);
    indices.reserve(gridX * gridY * 6);

    for (unsigned iy = 0; iy < gridY1; iy++) {

        const auto y = static_cast<float>(iy) * segment_height - height_half;

        for (unsigned ix = 0; ix < gridX1; ix++) {

            const auto x = static_cast<float>(ix) * segment_width - width_half;

            vertices.push_back(x);
            vertices.push_back(-y);
            vertices.push_back(0);

            normals.push_back(0);
            normals.push_back(0);
            normals.push_back(1);

            uvs.emplace_back(static_cast<float>(ix) / static_cast<float>(gridX));
            uvs.emplace_back(1 - (static_cast<float>(iy) / static_cast<float>(gridY)));
        }
    }

    for (unsigned iy = 0; iy < gridY; iy++) {

        for (unsigned ix = 0; ix < gridX; ix++) {

            const auto a = (ix + gridX1 * iy);
            const auto b = (ix + gridX1 * (iy + 1));
            const auto c = ((ix + 1) + gridX1 * (iy + 1));
            const auto d = ((ix + 1) + gridX1 * iy);

            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(d);
            indices.push_back(b);
            indices.push_back(c);
            indices.push_back(d);
        }
    }

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
}

std::string PlaneGeometry::type() const {

    return "PlaneGeometry";
}

std::shared_ptr<PlaneGeometry> PlaneGeometry::create(const PlaneGeometry::Params& params) {

    return std::shared_ptr<PlaneGeometry>(new PlaneGeometry(params));
}

std::shared_ptr<PlaneGeometry> PlaneGeometry::create(float width, float height, unsigned int widthSegments, unsigned int heightSegments) {

    return create(Params(width, height, widthSegments, heightSegments));
}

PlaneGeometry::Params::Params(float width, float height, unsigned int widthSegments, unsigned int heightSegments)
    : width(width), height(height), widthSegments(widthSegments), heightSegments(heightSegments) {}
