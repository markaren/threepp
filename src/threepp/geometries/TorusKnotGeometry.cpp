
#include "threepp/geometries/TorusKnotGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    void calculatePositionOnCurve(float u, unsigned int p, unsigned int q, float radius, Vector3& position) {

        const float cu = std::cos(u);
        const float su = std::sin(u);
        const float quOverP = static_cast<float>(q) / static_cast<float>(p) * u;
        const float cs = std::cos(quOverP);

        position.x = radius * (2 + cs) * 0.5f * cu;
        position.y = radius * (2 + cs) * su * 0.5f;
        position.z = radius * std::sin(quOverP) * 0.5f;
    }

}// namespace

TorusKnotGeometry::TorusKnotGeometry(float radius, float tube, unsigned int tubularSegments, unsigned int radialSegments, unsigned int p, unsigned int q) {

    // buffers

    std::vector<unsigned int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    // helper variables

    Vector3 vertex;
    Vector3 normal;

    Vector3 P1;
    Vector3 P2;

    Vector3 B;
    Vector3 T;
    Vector3 N;

    // generate vertices, normals and uvs

    for (unsigned i = 0; i <= tubularSegments; ++i) {

        // the radian "u" is used to calculate the position on the torus curve of the current tubular segement

        const auto u = static_cast<float>(i) / static_cast<float>(tubularSegments) * static_cast<float>(p) * math::TWO_PI;

        // now we calculate two points. P1 is our current position on the curve, P2 is a little farther ahead.
        // these points are used to create a special "coordinate space", which is necessary to calculate the correct vertex positions

        calculatePositionOnCurve(u, p, q, radius, P1);
        calculatePositionOnCurve(u + 0.01f, p, q, radius, P2);

        // calculate orthonormal basis

        T.subVectors(P2, P1);
        N.addVectors(P2, P1);
        B.crossVectors(T, N);
        N.crossVectors(B, T);

        // normalize B, N. T can be ignored, we don't use it

        B.normalize();
        N.normalize();

        for (unsigned j = 0; j <= radialSegments; ++j) {

            // now calculate the vertices. they are nothing more than an extrusion of the torus curve.
            // because we extrude a shape in the xy-plane, there is no need to calculate a z-value.

            const auto v = static_cast<float>(j) / static_cast<float>(radialSegments) * math::TWO_PI;
            const auto cx = -tube * std::cos(v);
            const auto cy = tube * std::sin(v);

            // now calculate the final vertex position.
            // first we orient the extrusion with our basis vectos, then we add it to the current position on the curve

            vertex.x = P1.x + (cx * N.x + cy * B.x);
            vertex.y = P1.y + (cx * N.y + cy * B.y);
            vertex.z = P1.z + (cx * N.z + cy * B.z);

            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // normal (P1 is always the center/origin of the extrusion, thus we can use it to calculate the normal)

            normal.subVectors(vertex, P1).normalize();

            normals.insert(normals.end(), {normal.x, normal.y, normal.z});

            // uv

            uvs.emplace_back(static_cast<float>(i) / static_cast<float>(tubularSegments));
            uvs.emplace_back(static_cast<float>(j) / static_cast<float>(radialSegments));
        }
    }

    // generate indices

    for (unsigned j = 1; j <= tubularSegments; j++) {

        for (unsigned i = 1; i <= radialSegments; i++) {

            // indices

            const auto a = (radialSegments + 1) * (j - 1) + (i - 1);
            const auto b = (radialSegments + 1) * j + (i - 1);
            const auto c = (radialSegments + 1) * j + i;
            const auto d = (radialSegments + 1) * (j - 1) + i;

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

std::string TorusKnotGeometry::type() const {

    return "TorusKnotGeometry";
}

std::shared_ptr<TorusKnotGeometry> TorusKnotGeometry::create(float radius, float tube, unsigned int tubularSegments, unsigned int radialSegments, unsigned int p, unsigned int q) {

    return std::shared_ptr<TorusKnotGeometry>(new TorusKnotGeometry(radius, tube, tubularSegments, radialSegments, p, q));
}
