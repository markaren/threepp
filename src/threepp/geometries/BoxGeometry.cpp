
#include "threepp/geometries/BoxGeometry.hpp"

#include <list>

using namespace threepp;

namespace {

    struct Helper {

        unsigned int numberOfVertices = 0;
        int groupStart = 0;

        std::list<unsigned int> indices;
        std::list<float> vertices;
        std::list<float> normals;
        std::list<float> uvs;

        explicit Helper(BoxGeometry& g, unsigned int widthSegments, unsigned int heightSegments, unsigned int depthSegments) {

            buildPlane(g, 2, 1, 0, -1, -1, g.depth, g.height, g.width, depthSegments, heightSegments, 0); // px
            buildPlane(g, 2, 1, 0, 1, -1, g.depth, g.height, -g.width, depthSegments, heightSegments, 1); // nx
            buildPlane(g, 0, 2, 1, 1, 1, g.width, g.depth, g.height, widthSegments, depthSegments, 2);    // py
            buildPlane(g, 0, 2, 1, 1, -1, g.width, g.depth, -g.height, widthSegments, depthSegments, 3);  // ny
            buildPlane(g, 0, 1, 2, 1, -1, g.width, g.height, g.depth, widthSegments, heightSegments, 4);  // pz
            buildPlane(g, 0, 1, 2, -1, -1, g.width, g.height, -g.depth, widthSegments, heightSegments, 5);// nz
        }

        void buildPlane(BoxGeometry& g, int u, int v, int w, float udir, float vdir, float width, float height, float depth, unsigned int gridX, unsigned int gridY, int materialIndex) {

            const auto segmentWidth = width / static_cast<float>(gridX);
            const auto segmentHeight = height / static_cast<float>(gridY);

            const auto widthHalf = width / 2;
            const auto heightHalf = height / 2;
            const auto depthHalf = depth / 2;

            const auto gridX1 = gridX + 1;
            const auto gridY1 = gridY + 1;

            int vertexCounter = 0;
            int groupCount = 0;

            auto vector = Vector3();

            // generate vertices, normals and uvs

            for (auto iy = 0; iy < gridY1; iy++) {

                const auto y = static_cast<float>(iy) * segmentHeight - heightHalf;

                for (auto ix = 0; ix < gridX1; ix++) {

                    const auto x = static_cast<float>(ix) * segmentWidth - widthHalf;

                    // set values to correct vector component

                    vector[u] = x * udir;
                    vector[v] = y * vdir;
                    vector[w] = depthHalf;

                    // now apply vector to vertex buffer

                    vertices.insert(vertices.end(), {vector.x, vector.y, vector.z});

                    // set values to correct vector component

                    vector[u] = 0;
                    vector[v] = 0;
                    vector[w] = depth > 0 ? 1.f : -1.f;

                    // now apply vector to normal buffer

                    normals.insert(normals.end(), {vector.x, vector.y, vector.z});

                    // uvs

                    uvs.emplace_back(static_cast<float>(ix) / static_cast<float>(gridX));
                    uvs.emplace_back(1 - (static_cast<float>(iy) / static_cast<float>(gridY)));

                    // counters

                    ++vertexCounter;
                }
            }

            // indices

            // 1. you need three indices to draw a single face
            // 2. a single segment consists of two faces
            // 3. so we need to generate six (2*3) indices per segment

            for (auto iy = 0; iy < gridY; iy++) {

                for (auto ix = 0; ix < gridX; ix++) {

                    const auto a = numberOfVertices + ix + gridX1 * iy;
                    const auto b = numberOfVertices + ix + gridX1 * (iy + 1);
                    const auto c = numberOfVertices + (ix + 1) + gridX1 * (iy + 1);
                    const auto d = numberOfVertices + (ix + 1) + gridX1 * iy;

                    // faces

                    indices.insert(indices.end(), {a, b, d});
                    indices.insert(indices.end(), {b, c, d});

                    // increase counter

                    groupCount += 6;
                }
            }

            // add a group to the geometry. this will ensure multi material support

            g.addGroup(groupStart, groupCount, materialIndex);

            // calculate new start value for groups

            groupStart += groupCount;

            // update total number of vertices

            numberOfVertices += vertexCounter;
        }
    };

}// namespace

BoxGeometry::BoxGeometry(const Params& params)
    : width(params.width), height(params.height), depth(params.depth) {

    Helper h(*this, params.widthSegments, params.heightSegments, params.depthSegments);
    this->setIndex(h.indices);
    this->setAttribute("position", FloatBufferAttribute::create(h.vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(h.normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(h.uvs, 2));
}

std::string BoxGeometry::type() const {

    return "BoxGeometry";
}
std::shared_ptr<BoxGeometry> BoxGeometry::create(const BoxGeometry::Params& params) {

    return std::shared_ptr<BoxGeometry>(new BoxGeometry(params));
}

std::shared_ptr<BoxGeometry> BoxGeometry::create(float width, float height, float depth, unsigned int widthSegments, unsigned int heightSegments, unsigned int depthSegments) {

    return create(Params(width, height, depth, widthSegments, heightSegments, depthSegments));
}

BoxGeometry::Params::Params(float width, float height, float depth, unsigned int widthSegments, unsigned int heightSegments, unsigned int depthSegments)
    : width(width), height(height), depth(depth),
      widthSegments(widthSegments), heightSegments(heightSegments), depthSegments(depthSegments) {}
