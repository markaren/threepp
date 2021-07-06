// https://github.com/mrdoob/three.js/blob/r129/src/geometries/CircleGeometry.js

#ifndef THREEPP_CIRCLEGEOMETRY_HPP
#define THREEPP_CIRCLEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class CircleGeometry : public BufferGeometry {

    public:
        static std::shared_ptr<CircleGeometry> create(float radius = 1, int segments = 8, float thetaStart = 0, float thetaLength = math::PI * 2) {
            return std::shared_ptr<CircleGeometry>(new CircleGeometry(radius, segments, thetaStart, thetaLength));
        }

    protected:
        CircleGeometry(float radius, int segments, float thetaStart, float thetaLength) {

            // buffers

            std::vector<int> indices;
            std::vector<float> vertices;
            std::vector<float> normals;
            std::vector<float> uvs;

            // helper variables

            Vector3 vertex;
            Vector2 uv;

            // center point

            vertices.insert(vertices.end(), {0, 0, 0});
            normals.insert(normals.end(), {0, 0, 1});
            uvs.insert(uvs.end(), {0.5, 0.5});

            for (int s = 0, i = 3; s <= segments; s++, i += 3) {

                const auto segment = thetaStart + s / segments * thetaLength;

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

            for (int i = 1; i <= segments; i++) {

                indices.insert(indices.end(), {i, i + 1, 0});
            }

            // build geometry

            this->setIndex(indices);
            this->setAttribute("position", FloatBufferAttribute ::create(vertices, 3));
            this->setAttribute("normal", FloatBufferAttribute ::create(normals, 3));
            this->setAttribute("uv", FloatBufferAttribute ::create(uvs, 2));
        }
    };

}// namespace threepp

#endif//THREEPP_CIRCLEGEOMETRY_HPP
