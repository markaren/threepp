// https://github.com/tentone/geo-three/blob/master/source/geometries/MapNodeGeometry.ts

#ifndef THREEPP_MAPNODEGEOMETRY_HPP
#define THREEPP_MAPNODEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class MapNodeGeometry: public BufferGeometry {

    public:
        static std::shared_ptr<MapNodeGeometry> create(
                int width = 1, int height = 1, int widthSegments = 1, int heightSegments = 1,
                bool skirt = true, int skirtDepth = 10) {

            return std::shared_ptr<MapNodeGeometry>(new MapNodeGeometry(width, height, widthSegments, heightSegments, skirt, skirtDepth));
        }

    private:
        explicit MapNodeGeometry(int width, int height, int widthSegments, int heightSegments, bool skirt, int skirtDepth) {

            std::vector<unsigned int> indices;
            std::vector<float> vertices;
            std::vector<float> normals;
            std::vector<float> uvs;

            buildPlane(indices, vertices, normals, uvs, width, height, widthSegments, heightSegments);

            if (skirt) {
                buildSkirt(indices, vertices, normals, uvs, width, height, widthSegments, heightSegments, skirtDepth);
            }

            setIndex(indices);
            setAttribute("position", FloatBufferAttribute::create(vertices, 3));
            setAttribute("normal", FloatBufferAttribute::create(normals, 3));
            setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        }

        void buildPlane(std::vector<unsigned int>& indices, std::vector<float>& vertices,
                        std::vector<float>& normals, std::vector<float>& uvs,
                        int width, int height, int widthSegments, int heightSegments) {
            // Half width X
            float widthHalf = static_cast<float>(width) / 2;

            // Half width Z
            float heightHalf = static_cast<float>(height) / 2;

            // Size of the grid in X
            int gridX = widthSegments + 1;

            // Size of the grid in Z
            int gridZ = heightSegments + 1;

            // Width of each segment X
            float segmentWidth = static_cast<float>(width) / static_cast<float>(widthSegments);

            // Height of each segment Z
            float segmentHeight = static_cast<float>(height) / static_cast<float>(heightSegments);

            // Generate vertices, normals and uvs
            for (int iz = 0; iz < gridZ; iz++) {
                float z = static_cast<float>(iz) * segmentHeight - heightHalf;

                for (int ix = 0; ix < gridX; ix++) {
                    float x = static_cast<float>(ix) * segmentWidth - widthHalf;

                    vertices.push_back(x);
                    vertices.push_back(0);
                    vertices.push_back(z);

                    normals.push_back(0);
                    normals.push_back(1);
                    normals.push_back(0);

                    uvs.push_back(static_cast<float>(ix) / static_cast<float>(widthSegments));
                    uvs.push_back(1 - static_cast<float>(iz) / static_cast<float>(heightSegments));
                }
            }

            // Indices
            for (int iz = 0; iz < heightSegments; iz++) {
                for (int ix = 0; ix < widthSegments; ix++) {
                    int a = ix + gridX * iz;
                    int b = ix + gridX * (iz + 1);
                    int c = ix + 1 + gridX * (iz + 1);
                    int d = ix + 1 + gridX * iz;

                    // Faces
                    indices.push_back(a);
                    indices.push_back(b);
                    indices.push_back(d);
                    indices.push_back(b);
                    indices.push_back(c);
                    indices.push_back(d);
                }
            }
        }

        void buildSkirt(std::vector<unsigned int>& indices, std::vector<float>& vertices,
                        std::vector<float>& normals, std::vector<float>& uvs,
                        float width, float height, int widthSegments, int heightSegments, float skirtDepth) {
            // Half width X
            float widthHalf = width / 2;

            // Half width Z
            float heightHalf = height / 2;

            // Size of the grid in X
            int gridX = widthSegments + 1;

            // Size of the grid in Z
            int gridZ = heightSegments + 1;

            // Width of each segment X
            float segmentWidth = width / static_cast<float>(widthSegments);

            // Height of each segment Z
            float segmentHeight = height / static_cast<float>(heightSegments);

            // Down X
            auto start = vertices.size() / 3;
            for (int ix = 0; ix < gridX; ix++) {
                float x = ix * segmentWidth - widthHalf;
                float z = -heightHalf;

                vertices.push_back(x);
                vertices.push_back(-skirtDepth);
                vertices.push_back(z);

                normals.push_back(0);
                normals.push_back(1);
                normals.push_back(0);

                uvs.push_back(static_cast<float>(ix) / widthSegments);
                uvs.push_back(1);
            }

            // Indices
            for (int ix = 0; ix < widthSegments; ix++) {
                int a = ix;
                int d = ix + 1;
                int b = ix + start;
                int c = ix + start + 1;
                indices.push_back(d);
                indices.push_back(b);
                indices.push_back(a);
                indices.push_back(d);
                indices.push_back(c);
                indices.push_back(b);
            }

            // Up X
            start = vertices.size() / 3;
            for (int ix = 0; ix < gridX; ix++) {
                float x = ix * segmentWidth - widthHalf;
                float z = heightSegments * segmentHeight - heightHalf;

                vertices.push_back(x);
                vertices.push_back(-skirtDepth);
                vertices.push_back(z);

                normals.push_back(0);
                normals.push_back(1);
                normals.push_back(0);

                uvs.push_back(static_cast<float>(ix) / widthSegments);
                uvs.push_back(0);
            }

            int offset = gridX * gridZ - widthSegments - 1;
            for (int ix = 0; ix < widthSegments; ix++) {
                int a = offset + ix;
                int d = offset + ix + 1;
                int b = ix + start;
                int c = ix + start + 1;
                indices.push_back(a);
                indices.push_back(b);
                indices.push_back(d);
                indices.push_back(b);
                indices.push_back(c);
                indices.push_back(d);
            }

            // Down Z
            start = vertices.size() / 3;
            for (int iz = 0; iz < gridZ; iz++) {
                float z = iz * segmentHeight - heightHalf;
                float x = -widthHalf;

                vertices.push_back(x);
                vertices.push_back(-skirtDepth);
                vertices.push_back(z);

                normals.push_back(0);
                normals.push_back(1);
                normals.push_back(0);

                uvs.push_back(0);
                uvs.push_back(1 - static_cast<float>(iz) / static_cast<float>(heightSegments));
            }

            for (int iz = 0; iz < heightSegments; iz++) {
                int a = iz * gridZ;
                int d = (iz + 1) * gridZ;
                int b = iz + start;
                int c = iz + start + 1;
                indices.push_back(a);
                indices.push_back(b);
                indices.push_back(d);
                indices.push_back(b);
                indices.push_back(c);
                indices.push_back(d);
            }

            // Up Z
            start = vertices.size() / 3;
            for (int iz = 0; iz < gridZ; iz++) {
                float z = iz * segmentHeight - heightHalf;
                float x = widthSegments * segmentWidth - widthHalf;

                vertices.push_back(x);
                vertices.push_back(-skirtDepth);
                vertices.push_back(z);

                normals.push_back(0);
                normals.push_back(1);
                normals.push_back(0);

                uvs.push_back(1);
                uvs.push_back(1 - static_cast<float>(iz) / static_cast<float>(heightSegments));
            }

            for (int iz = 0; iz < heightSegments; iz++) {
                int a = iz * gridZ + heightSegments;
                int d = (iz + 1) * gridZ + heightSegments;
                int b = iz + start;
                int c = iz + start + 1;
                indices.push_back(d);
                indices.push_back(b);
                indices.push_back(a);
                indices.push_back(d);
                indices.push_back(c);
                indices.push_back(b);
            }
        }
    };

}// namespace threepp

#endif//THREEPP_MAPNODEGEOMETRY_HPP
