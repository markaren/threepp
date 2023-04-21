
#include "threepp/geometries/PolyhedronGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

namespace {

    struct Helper {

        std::vector<float> vertexBuffer;
        std::vector<float> uvBuffer;

        Helper(const std::vector<float>& vertices,
               const std::vector<unsigned int>& indices,
               float radius, unsigned int detail) {

            subdivide(vertices, indices, detail);

            // all vertices should lie on a conceptual sphere with a given radius

            applyRadius(radius);

            // finally, create the uv data

            generateUVs();
        }

        void subdivide(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, unsigned int detail) {

            Vector3 a;
            Vector3 b;
            Vector3 c;

            // iterate over all faces and apply a subdivision with the given detail value

            for (unsigned i = 0; i < indices.size(); i += 3) {

                // get the vertices of the face

                getVertexByIndex(vertices, indices[i + 0], a);
                getVertexByIndex(vertices, indices[i + 1], b);
                getVertexByIndex(vertices, indices[i + 2], c);

                // perform subdivision

                subdivideFace(a, b, c, detail);
            }
        }

        void subdivideFace(const Vector3& a, const Vector3& b, const Vector3& c, unsigned int detail) {

            const auto cols = detail + 1;

            // we use this multidimensional array as a data structure for creating the subdivision

            std::vector<std::vector<Vector3>> v(cols + 1);

            // construct all of the vertices for this subdivision

            for (unsigned i = 0; i <= cols; i++) {

                auto& vi = v[i];

                auto aj = a.clone().lerp(c, static_cast<float>(i) / static_cast<float>(cols));
                auto bj = b.clone().lerp(c, static_cast<float>(i) / static_cast<float>(cols));

                const auto rows = cols - i;
                vi.resize(rows + 1);

                for (unsigned j = 0; j <= rows; j++) {

                    if (j == 0 && i == cols) {

                        vi[j] = aj;

                    } else {

                        vi[j] = aj.clone().lerp(bj, static_cast<float>(j) / static_cast<float>(rows));
                    }
                }
            }

            // construct all of the faces

            for (unsigned i = 0; i < cols; i++) {

                for (unsigned j = 0; j < 2 * (cols - i) - 1; j++) {

                    const auto k = j / 2;

                    if (j % 2 == 0) {

                        pushVertex(v[i][k + 1]);
                        pushVertex(v[i + 1][k]);
                        pushVertex(v[i][k]);

                    } else {

                        pushVertex(v[i][k + 1]);
                        pushVertex(v[i + 1][k + 1]);
                        pushVertex(v[i + 1][k]);
                    }
                }
            }
        }

        void applyRadius(float radius) {

            Vector3 vertex;

            // iterate over the entire buffer and apply the radius to each vertex

            for (unsigned i = 0; i < vertexBuffer.size(); i += 3) {

                vertex.x = vertexBuffer[i + 0];
                vertex.y = vertexBuffer[i + 1];
                vertex.z = vertexBuffer[i + 2];

                vertex.normalize().multiplyScalar(radius);

                vertexBuffer[i + 0] = vertex.x;
                vertexBuffer[i + 1] = vertex.y;
                vertexBuffer[i + 2] = vertex.z;
            }
        }

        void generateUVs() {

            Vector3 vertex;

            for (unsigned i = 0; i < vertexBuffer.size(); i += 3) {

                vertex.x = vertexBuffer[i + 0];
                vertex.y = vertexBuffer[i + 1];
                vertex.z = vertexBuffer[i + 2];

                const auto u = azimuth(vertex) / 2.f / math::PI + 0.5f;
                const auto v = inclination(vertex) / math::PI + 0.5f;
                uvBuffer.insert(uvBuffer.end(), {u, 1 - v});
            }

            correctUVs();

            correctSeam();
        }

        void correctSeam() {

            // handle case when face straddles the seam, see #3269

            for (unsigned i = 0; i < uvBuffer.size(); i += 6) {

                // uv data of a single face

                const auto x0 = uvBuffer[i + 0];
                const auto x1 = uvBuffer[i + 2];
                const auto x2 = uvBuffer[i + 4];

                const auto max = std::max(std::max(x0, x1), x2);
                const auto min = std::min(std::min(x0, x1), x2);

                // 0.9 is somewhat arbitrary

                if (max > 0.9f && min < 0.1f) {

                    if (x0 < 0.2) uvBuffer[i + 0] += 1;
                    if (x1 < 0.2) uvBuffer[i + 2] += 1;
                    if (x2 < 0.2) uvBuffer[i + 4] += 1;
                }
            }
        }

        void pushVertex(const Vector3& vertex) {

            vertexBuffer.emplace_back(vertex.x);
            vertexBuffer.emplace_back(vertex.y);
            vertexBuffer.emplace_back(vertex.z);
        }

        static void getVertexByIndex(const std::vector<float>& vertices, unsigned int index, Vector3& vertex) {

            const auto stride = index * 3;

            vertex.x = vertices[stride + 0];
            vertex.y = vertices[stride + 1];
            vertex.z = vertices[stride + 2];
        }

        void correctUVs() {

            Vector3 a;
            Vector3 b;
            Vector3 c;

            Vector3 centroid;

            Vector2 uvA;
            Vector2 uvB;
            Vector2 uvC;

            for (unsigned i = 0, j = 0; i < vertexBuffer.size(); i += 9, j += 6) {

                a.set(vertexBuffer[i + 0], vertexBuffer[i + 1], vertexBuffer[i + 2]);
                b.set(vertexBuffer[i + 3], vertexBuffer[i + 4], vertexBuffer[i + 5]);
                c.set(vertexBuffer[i + 6], vertexBuffer[i + 7], vertexBuffer[i + 8]);

                uvA.set(uvBuffer[j + 0], uvBuffer[j + 1]);
                uvB.set(uvBuffer[j + 2], uvBuffer[j + 3]);
                uvC.set(uvBuffer[j + 4], uvBuffer[j + 5]);

                centroid.copy(a).add(b).add(c).divideScalar(3);

                const auto azi = azimuth(centroid);

                correctUV(uvA, j + 0, a, azi);
                correctUV(uvB, j + 2, b, azi);
                correctUV(uvC, j + 4, c, azi);
            }
        }

        void correctUV(const Vector2& uv, unsigned int stride, const Vector3& vector, float azimuth) {

            if ((azimuth < 0) && (uv.x == 1)) {

                uvBuffer[stride] = uv.x - 1;
            }

            if ((vector.x == 0) && (vector.z == 0)) {

                uvBuffer[stride] = azimuth / 2.f / math::PI + 0.5f;
            }
        }

        // Angle around the Y axis, counter-clockwise when looking from above.

        static float azimuth(const Vector3& vector) {

            return std::atan2(vector.z, -vector.x);
        }


        // Angle above the XZ plane.

        static float inclination(const Vector3& vector) {

            return std::atan2(-vector.y, std::sqrt((vector.x * vector.x) + (vector.z * vector.z)));
        }
    };

}// namespace


PolyhedronGeometry::PolyhedronGeometry(
        const std::vector<float>& vertices,
        const std::vector<unsigned int>& indices,
        float radius, unsigned int detail) {

    Helper h(vertices, indices, radius, detail);

    this->setAttribute("position", FloatBufferAttribute::create(h.vertexBuffer, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(h.vertexBuffer, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(h.uvBuffer, 2));

    if (detail == 0) {

        this->computeVertexNormals();// flat normals

    } else {

        this->normalizeNormals();// smooth normals
    }
}


std::string PolyhedronGeometry::type() const {

    return "PolyhedronGeometry";
}
