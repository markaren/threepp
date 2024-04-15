
#include "threepp/geometries/DecalGeometry.hpp"

#include "threepp/objects/Mesh.hpp"

using namespace threepp;

namespace {

    struct DecalVertex {

        Vector3 position;
        Vector3 normal;
    };

    DecalVertex clip(const DecalVertex& v0, const DecalVertex& v1, const Vector3& p, float s) {

        auto d0 = v0.position.dot(p) - s;
        auto d1 = v1.position.dot(p) - s;

        auto s0 = d0 / (d0 - d1);

        auto v = DecalVertex{
                Vector3(
                        v0.position.x + s0 * (v1.position.x - v0.position.x),
                        v0.position.y + s0 * (v1.position.y - v0.position.y),
                        v0.position.z + s0 * (v1.position.z - v0.position.z)),
                Vector3(
                        v0.normal.x + s0 * (v1.normal.x - v0.normal.x),
                        v0.normal.y + s0 * (v1.normal.y - v0.normal.y),
                        v0.normal.z + s0 * (v1.normal.z - v0.normal.z))};

        // need to clip more values (texture coordinates)? do it this way:
        // intersectpoint.value = a.value + s * ( b.value - a.value );

        return v;
    }

    std::vector<DecalVertex> clipGeometry(std::vector<DecalVertex>& inVertices, const Vector3& plane, const Vector3& size) {

        std::vector<DecalVertex> outVertices;

        float s = 0.5f * std::abs(size.dot(plane));

        // a single iteration clips one face,
        // which consists of three consecutive 'DecalVertex' objects

        for (unsigned i = 0; i < inVertices.size(); i += 3) {

            int total = 0;
            DecalVertex nV1;
            DecalVertex nV2;
            DecalVertex nV3;
            DecalVertex nV4;

            auto d1 = inVertices[i + 0].position.dot(plane) - s;
            auto d2 = inVertices[i + 1].position.dot(plane) - s;
            auto d3 = inVertices[i + 2].position.dot(plane) - s;

            auto v1Out = d1 > 0;
            auto v2Out = d2 > 0;
            auto v3Out = d3 > 0;

            // calculate, how many vertices of the face lie outside of the clipping plane

            total = (v1Out ? 1 : 0) + (v2Out ? 1 : 0) + (v3Out ? 1 : 0);

            switch (total) {

                case 0: {

                    // the entire face lies inside of the plane, no clipping needed

                    outVertices.emplace_back(inVertices[i]);
                    outVertices.emplace_back(inVertices[i + 1]);
                    outVertices.emplace_back(inVertices[i + 2]);
                    break;
                }

                case 1: {

                    // one vertex lies outside of the plane, perform clipping

                    if (v1Out) {

                        nV1 = inVertices[i + 1];
                        nV2 = inVertices[i + 2];
                        nV3 = clip(inVertices[i], nV1, plane, s);
                        nV4 = clip(inVertices[i], nV2, plane, s);
                    }

                    if (v2Out) {

                        nV1 = inVertices[i];
                        nV2 = inVertices[i + 2];
                        nV3 = clip(inVertices[i + 1], nV1, plane, s);
                        nV4 = clip(inVertices[i + 1], nV2, plane, s);

                        outVertices.emplace_back(nV3);
                        outVertices.emplace_back(nV2);
                        outVertices.emplace_back(nV1);

                        outVertices.emplace_back(nV2);
                        outVertices.emplace_back(nV3);
                        outVertices.emplace_back(nV4);
                        break;
                    }

                    if (v3Out) {

                        nV1 = inVertices[i];
                        nV2 = inVertices[i + 1];
                        nV3 = clip(inVertices[i + 2], nV1, plane, s);
                        nV4 = clip(inVertices[i + 2], nV2, plane, s);
                    }

                    outVertices.emplace_back(nV1);
                    outVertices.emplace_back(nV2);
                    outVertices.emplace_back(nV3);

                    outVertices.emplace_back(nV4);
                    outVertices.emplace_back(nV3);
                    outVertices.emplace_back(nV2);

                    break;
                }

                case 2: {

                    // two vertices lies outside of the plane, perform clipping

                    if (!v1Out) {

                        nV1 = inVertices[i];
                        nV2 = clip(nV1, inVertices[i + 1], plane, s);
                        nV3 = clip(nV1, inVertices[i + 2], plane, s);
                        outVertices.emplace_back(nV1);
                        outVertices.emplace_back(nV2);
                        outVertices.emplace_back(nV3);
                    }

                    if (!v2Out) {

                        nV1 = inVertices[i + 1];
                        nV2 = clip(nV1, inVertices[i + 2], plane, s);
                        nV3 = clip(nV1, inVertices[i], plane, s);
                        outVertices.emplace_back(nV1);
                        outVertices.emplace_back(nV2);
                        outVertices.emplace_back(nV3);
                    }

                    if (!v3Out) {

                        nV1 = inVertices[i + 2];
                        nV2 = clip(nV1, inVertices[i], plane, s);
                        nV3 = clip(nV1, inVertices[i + 1], plane, s);
                        outVertices.emplace_back(nV1);
                        outVertices.emplace_back(nV2);
                        outVertices.emplace_back(nV3);
                    }

                    break;
                }

                case 3: {

                    // the entire face lies outside of the plane, so let's discard the corresponding vertices

                    break;
                }
            }
        }

        return outVertices;
    }

}// namespace


DecalGeometry::DecalGeometry(
        const Mesh& mesh,
        const Vector3& position,
        const Euler& orientation,
        const Vector3& size) {

    // buffers

    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    // helpers

    Vector3 plane;

    Matrix4 projectorMatrix;
    projectorMatrix.makeRotationFromEuler(orientation);
    projectorMatrix.setPosition(position);

    Matrix4 projectorMatrixInverse;
    projectorMatrixInverse.copy(projectorMatrix).invert();

    // generate buffers

    auto pushDecalVertex = [&](std::vector<DecalVertex>& decalVertices, Vector3& vertex, Vector3& normal) {
        // transform the vertex to world space, then to projector space

        vertex.applyMatrix4(*mesh.matrixWorld);
        vertex.applyMatrix4(projectorMatrixInverse);

        normal.transformDirection(*mesh.matrixWorld);

        decalVertices.emplace_back(DecalVertex{vertex, normal});
    };

    auto generate = [&] {
        std::vector<DecalVertex> decalVertices;

        Vector3 vertex;
        Vector3 normal;

        // handle different geometry types

        const auto& geometry = mesh.geometry();

        auto positionAttribute = geometry->getAttribute<float>("position");
        auto normalAttribute = geometry->getAttribute<float>("normal");

        // first, create an array of 'DecalVertex' objects
        // three consecutive 'DecalVertex' objects represent a single face
        //
        // this data structure will be later used to perform the clipping

        if (geometry->hasIndex()) {

            // indexed BufferGeometry

            auto index = geometry->getIndex();

            for (int i = 0; i < index->count(); i++) {

                positionAttribute->setFromBufferAttribute(vertex, index->getX(i));
                normalAttribute->setFromBufferAttribute(normal, index->getX(i));

                pushDecalVertex(decalVertices, vertex, normal);
            }

        } else {

            // non-indexed BufferGeometry

            for (int i = 0; i < positionAttribute->count(); i++) {

                positionAttribute->setFromBufferAttribute(vertex, i);
                normalAttribute->setFromBufferAttribute(normal, i);

                pushDecalVertex(decalVertices, vertex, normal);
            }
        }

        // second, clip the geometry so that it doesn't extend out from the projector

        decalVertices = clipGeometry(decalVertices, plane.set(1, 0, 0), size);
        decalVertices = clipGeometry(decalVertices, plane.set(-1, 0, 0), size);
        decalVertices = clipGeometry(decalVertices, plane.set(0, 1, 0), size);
        decalVertices = clipGeometry(decalVertices, plane.set(0, -1, 0), size);
        decalVertices = clipGeometry(decalVertices, plane.set(0, 0, 1), size);
        decalVertices = clipGeometry(decalVertices, plane.set(0, 0, -1), size);

        // third, generate final vertices, normals and uvs

        for (auto& decalVertex : decalVertices) {

            // create texture coordinates (we are still in projector space)

            uvs.emplace_back(0.5f + (decalVertex.position.x / size.x));
            uvs.emplace_back(0.5f + (decalVertex.position.y / size.y));

            // transform the vertex back to world space

            decalVertex.position.applyMatrix4(projectorMatrix);

            // now create vertex and normal buffer data

            vertices.insert(vertices.end(), {decalVertex.position.x, decalVertex.position.y, decalVertex.position.z});
            normals.insert(normals.end(), {decalVertex.normal.x, decalVertex.normal.y, decalVertex.normal.z});
        }
    };

    generate();

    // build geometry

    setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
}

std::shared_ptr<DecalGeometry> DecalGeometry::create(const Mesh& mesh, const Vector3& position, const Euler& orientation, const Vector3& size) {

    return std::shared_ptr<DecalGeometry>(new DecalGeometry(mesh, position, orientation, size));
}

std::string DecalGeometry::type() const {

    return "DecalGeometry";
}
