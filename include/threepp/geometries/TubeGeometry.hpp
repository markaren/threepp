// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TubeGeometry.js

#ifndef THREEPP_TUBEGEOMETRY_HPP
#define THREEPP_TUBEGEOMETRY_HPP

#include <utility>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Curve.hpp"
#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class TubeGeometry : public BufferGeometry {

    public:
        std::shared_ptr<Curve3> path;
        int tubularSegments;
        float radius;
        int radialSegments;
        bool closed;

        static std::shared_ptr<TubeGeometry> create(const std::shared_ptr<Curve3> &path, int tubularSegments = 64, float radius = 1, int radialSegments = 8, bool closed = false) {

            return std::shared_ptr<TubeGeometry>(new TubeGeometry(path, tubularSegments, radius, radialSegments, closed));
        }

    private:
        Curve3::FrenetFrames frames;

        TubeGeometry(std::shared_ptr<Curve3> path, int tubularSegments, float radius, int radialSegments, bool closed)
            : path(std::move(path)), tubularSegments(tubularSegments), radius(radius), radialSegments(radialSegments), closed(closed) {

            this->frames = this->path->computeFrenetFrames(tubularSegments, closed);

            // helper variables

            Vector3 vertex;
            Vector3 normal;
            Vector2 uv;
            Vector3 P;

            // buffer

            std::vector<float> vertices;
            std::vector<float> normals;
            std::vector<float> uvs;
            std::vector<int> indices;

            // functions

            auto generateSegment = std::function<void(int)>([&](int i) {
                // we use getPointAt to sample evenly distributed points from the given path

                this->path->getPointAt(static_cast<float>(i) / static_cast<float>(tubularSegments), P);

                // retrieve corresponding normal and binormal

                const Vector3 &N = frames.normals[i];
                const Vector3 &B = frames.binormals[i];

                // generate normals and vertices for the current segment

                for (int j = 0; j <= radialSegments; j++) {

                    const float v = static_cast<float>(j) / static_cast<float>(radialSegments) * math::PI * 2;

                    const float sin = std::sin(v);
                    const float cos = -std::cos(v);

                    // normal

                    normal.x = (cos * N.x + sin * B.x);
                    normal.y = (cos * N.y + sin * B.y);
                    normal.z = (cos * N.z + sin * B.z);
                    normal.normalize();

                    normals.insert(normals.end(), {normal.x, normal.y, normal.z});

                    // vertex

                    vertex.x = P.x + radius * normal.x;
                    vertex.y = P.y + radius * normal.y;
                    vertex.z = P.z + radius * normal.z;

                    vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});
                }
            });

            auto generateIndices = std::function<void()>([&] {
                for (int j = 1; j <= tubularSegments; j++) {

                    for (int i = 1; i <= radialSegments; i++) {

                        const int a = (radialSegments + 1) * (j - 1) + (i - 1);
                        const int b = (radialSegments + 1) * j + (i - 1);
                        const int c = (radialSegments + 1) * j + i;
                        const int d = (radialSegments + 1) * (j - 1) + i;

                        // faces

                        indices.insert(indices.end(), {a, b, d});
                        indices.insert(indices.end(), {b, c, d});
                    }
                }
            });

            auto generateUVs = std::function<void()>([&] {
                for (int i = 0; i <= tubularSegments; i++) {

                    for (int j = 0; j <= radialSegments; j++) {

                        uv.x = static_cast<float>(i) / static_cast<float>(tubularSegments);
                        uv.y = static_cast<float>(j) / static_cast<float>(radialSegments);

                        uvs.insert(uvs.end(), {uv.x, uv.y});
                    }
                }
            });

            auto generateBufferData = std::function<void()>([&] {
                for (int i = 0; i < tubularSegments; i++) {

                    generateSegment(i);
                }

                // if the geometry is not closed, generate the last row of vertices and normals
                // at the regular position on the given path
                //
                // if the geometry is closed, duplicate the first row of vertices and normals (uvs will differ)

                generateSegment((!closed) ? tubularSegments : 0);

                // uvs are generated in a separate function.
                // this makes it easy compute correct values for closed geometries

                generateUVs();

                // finally create faces

                generateIndices();
            });

            // create buffer data

            generateBufferData();

            this->setIndex(indices);
            this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
            this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
            this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));

        }
    };

}// namespace threepp

#endif//THREEPP_TUBEGEOMETRY_HPP
