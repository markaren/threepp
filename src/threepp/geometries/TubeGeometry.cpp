
#include "threepp/geometries/TubeGeometry.hpp"

#include <cmath>
#include <functional>
#include <list>

using namespace threepp;

TubeGeometry::TubeGeometry(std::shared_ptr<Curve3> path, const Params& params)
    : path(std::move(path)), radius(params.radius) {

    this->frames = FrenetFrames::compute(*this->path, params.tubularSegments, params.closed);

    // helper variables

    Vector3 vertex;
    Vector3 normal;
    Vector2 uv;
    Vector3 P;

    // buffer

    std::list<float> vertices;
    std::list<float> normals;
    std::list<float> uvs;
    std::list<unsigned int> indices;

    // functions

    auto generateSegment = std::function<void(unsigned int)>([&](unsigned int i) {
        // we use getPointAt to sample evenly distributed points from the given path

        this->path->getPointAt(static_cast<float>(i) / static_cast<float>(params.tubularSegments), P);

        // retrieve corresponding normal and binormal

        const Vector3& N = frames.normals[i];
        const Vector3& B = frames.binormals[i];

        // generate normals and vertices for the current segment

        for (unsigned j = 0; j <= params.radialSegments; j++) {

            const float v = static_cast<float>(j) / static_cast<float>(params.radialSegments) * math::TWO_PI;

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
        for (unsigned j = 1; j <= params.tubularSegments; j++) {

            for (unsigned i = 1; i <= params.radialSegments; i++) {

                const auto a = (params.radialSegments + 1) * (j - 1) + (i - 1);
                const auto b = (params.radialSegments + 1) * j + (i - 1);
                const auto c = (params.radialSegments + 1) * j + i;
                const auto d = (params.radialSegments + 1) * (j - 1) + i;

                // faces

                indices.insert(indices.end(), {a, b, d});
                indices.insert(indices.end(), {b, c, d});
            }
        }
    });

    auto generateUVs = std::function<void()>([&] {
        for (unsigned i = 0; i <= params.tubularSegments; i++) {

            for (unsigned j = 0; j <= params.radialSegments; j++) {

                uv.x = static_cast<float>(i) / static_cast<float>(params.tubularSegments);
                uv.y = static_cast<float>(j) / static_cast<float>(params.radialSegments);

                uvs.insert(uvs.end(), {uv.x, uv.y});
            }
        }
    });

    auto generateBufferData = std::function<void()>([&] {
        for (unsigned i = 0; i < params.tubularSegments; i++) {

            generateSegment(i);
        }

        // if the geometry is not closed, generate the last row of vertices and normals
        // at the regular position on the given path
        //
        // if the geometry is closed, duplicate the first row of vertices and normals (uvs will differ)

        generateSegment((!params.closed) ? params.tubularSegments : 0);

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

std::string TubeGeometry::type() const {

    return "TubeGeometry";
}

std::shared_ptr<TubeGeometry> TubeGeometry::create(const std::shared_ptr<Curve3>& path, const TubeGeometry::Params& params) {

    return std::shared_ptr<TubeGeometry>(new TubeGeometry(path, params));
}

std::shared_ptr<TubeGeometry> TubeGeometry::create(const std::shared_ptr<Curve3>& path, unsigned int tubularSegments, float radius, unsigned int radialSegments, bool closed) {

    return create(path, Params(tubularSegments, radius, radialSegments, closed));
}

TubeGeometry::Params::Params(unsigned int tubularSegments, float radius, unsigned int radialSegments, bool closed)
    : tubularSegments(tubularSegments), radius(radius),
      radialSegments(radialSegments), closed(closed) {}
