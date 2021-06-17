
#include "threepp/geometries/BoxGeometry.hpp"

using namespace threepp;

namespace {

    struct Helper {

        int numberOfVertices = 0;
        int groupStart = 0;

        std::vector<int> indices;
        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<float> uvs;

        explicit Helper(BoxGeometry &g) {

            buildPlane(g, 2, 1, 0, -1, -1, g.depth, g.height, g.width, g.depthSegments, g.heightSegments, 0); // px
            buildPlane(g,2, 1, 0, 1, -1, g.depth, g.height, -g.width, g.depthSegments, g.heightSegments, 1); // nx
            buildPlane(g,0, 2, 1, 1, 1, g.width, g.depth, g.height, g.widthSegments, g.depthSegments, 2);    // py
            buildPlane(g,0, 2, 1, 1, -1, g.width, g.depth, -g.height, g.widthSegments, g.depthSegments, 3);  // ny
            buildPlane(g,0, 1, 2, 1, -1, g.width, g.height, g.depth, g.widthSegments, g.heightSegments, 4);  // pz
            buildPlane(g,0, 1, 2, -1, -1, g.width, g.height, -g.depth, g.widthSegments, g.heightSegments, 5);// nz
        }

        void buildPlane( BoxGeometry &g, char u, char v, char w, int udir, int vdir, float width, float height, float depth, int gridX, int gridY, int materialIndex) {

            const auto segmentWidth = width / gridX;
            const auto segmentHeight = height / gridY;

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

                const auto y = iy * segmentHeight - heightHalf;

                for (auto ix = 0; ix < gridX1; ix++) {

                    const auto x = ix * segmentWidth - widthHalf;

                    // set values to correct vector component

                    vector[u] = x * udir;
                    vector[v] = y * vdir;
                    vector[w] = depthHalf;

                    // now apply vector to vertex buffer

                    vertices.insert(vertices.end(), {vector.x, vector.y, vector.z} );

                    // set values to correct vector component

                    vector[u] = 0;
                    vector[v] = 0;
                    vector[w] = depth > 0 ? 1 : -1;

                    // now apply vector to normal buffer

                    normals.insert(normals.end(), {vector.x, vector.y, vector.z} );

                    // uvs

                    uvs.emplace_back(ix / gridX);
                    uvs.emplace_back(1 - (iy / gridY));

                    // counters

                    vertexCounter += 1;
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

}

BoxGeometry::BoxGeometry(float width, float height, float depth, int widthSegments, int heightSegments, int depthSegments)
        : width(width), height(height), depth(depth), widthSegments(widthSegments), heightSegments(heightSegments), depthSegments(depthSegments) {

    Helper h(*this);
    this->setIndex(h.indices);
    this->setAttribute("position", BufferAttribute<float>(h.vertices, 3));
    this->setAttribute("normal", BufferAttribute<float>(h.normals, 3));
    this->setAttribute("uv", BufferAttribute<float>(h.uvs, 2));
}
