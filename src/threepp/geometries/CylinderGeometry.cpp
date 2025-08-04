
#include "threepp/geometries/CylinderGeometry.hpp"

#include <cmath>
#include <vector>

using namespace threepp;


CylinderGeometry::CylinderGeometry(const Params& params)
    : radiusTop(params.radiusTop),
      radiusBottom(params.radiusBottom),
      height(params.height) {

    std::vector<unsigned int> indices;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;

    unsigned int index = 0;
    const auto halfHeight = height / 2;
    std::vector<std::vector<unsigned int>> indexArray;
    int groupStart = 0;

    auto generateTorso = [&] {
        Vector3 normal;
        Vector3 vertex;

        int groupCount = 0;

        // this will be used to calculate the normal
        const auto slope = (radiusBottom - radiusTop) / height;

        // generate vertices, normals and uvs

        for (unsigned y = 0; y <= params.heightSegments; y++) {

            std::vector<unsigned int> indexRow;

            const auto v = static_cast<float>(y) / static_cast<float>(params.heightSegments);

            // calculate the radius of the current row

            const auto radius = v * (radiusBottom - radiusTop) + radiusTop;

            for (unsigned x = 0; x <= params.radialSegments; x++) {

                const auto u = static_cast<float>(x) / static_cast<float>(params.radialSegments);

                const auto theta = u * params.thetaLength + params.thetaStart;

                const auto sinTheta = std::sin(theta);
                const auto cosTheta = std::cos(theta);

                // vertex

                vertex.x = radius * sinTheta;
                vertex.y = -v * height + halfHeight;
                vertex.z = radius * cosTheta;
                vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

                // normal

                normal.set(sinTheta, slope, cosTheta).normalize();
                normals.insert(normals.end(), {normal.x, normal.y, normal.z});

                // uv

                uvs.insert(uvs.end(), {u, 1 - v});

                // save index of vertex in respective row

                indexRow.emplace_back(index++);
            }

            // now save vertices of the row in our index array

            indexArray.emplace_back(indexRow);
        }

        // generate indices

        for (unsigned x = 0; x < params.radialSegments; x++) {

            for (unsigned y = 0; y < params.heightSegments; y++) {

                // we use the index array to access the correct indices

                const auto a = indexArray[y][x];
                const auto b = indexArray[y + 1][x];
                const auto c = indexArray[y + 1][x + 1];
                const auto d = indexArray[y][x + 1];

                // faces

                indices.insert(indices.end(), {a, b, d});
                indices.insert(indices.end(), {b, c, d});

                // update group counter

                groupCount += 6;
            }
        }

        // add a group to the geometry. this will ensure multi material support

        addGroup(groupStart, groupCount, 0);

        // calculate new start value for groups

        groupStart += groupCount;
    };

    generateTorso();


    auto generateCap = [&](bool top) {
        // save the index of the first center vertex
        auto centerIndexStart = index;

        Vector2 uv;
        Vector3 vertex;

        int groupCount = 0;

        float radius = (top) ? radiusTop : radiusBottom;
        float sign = (top) ? 1 : -1;

        // first we generate the center vertex data of the cap.
        // because the geometry needs one set of uvs per face,
        // we must generate a center vertex per face/segment

        for (unsigned x = 1; x <= params.radialSegments; x++) {

            // vertex

            vertices.insert(vertices.end(), {0, halfHeight * sign, 0});

            // normal

            normals.insert(normals.end(), {0, sign, 0});

            // uv

            uvs.insert(uvs.end(), {0.5, 0.5});

            // increase index

            index++;
        }

        // save the index of the last center vertex
        const auto centerIndexEnd = index;

        // now we generate the surrounding vertices, normals and uvs

        for (unsigned x = 0; x <= params.radialSegments; x++) {

            float u = static_cast<float>(x) / static_cast<float>(params.radialSegments);
            float theta = u * params.thetaLength + params.thetaStart;

            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);

            // vertex

            vertex.x = radius * sinTheta;
            vertex.y = halfHeight * sign;
            vertex.z = radius * cosTheta;
            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            // normal

            normals.insert(normals.end(), {0, sign, 0});

            // uv

            uv.x = (cosTheta * 0.5f) + 0.5f;
            uv.y = (sinTheta * 0.5f * sign) + 0.5f;
            uvs.insert(uvs.end(), {uv.x, uv.y});

            // increase index

            ++index;
        }

        // generate indices

        for (unsigned x = 0; x < params.radialSegments; x++) {

            unsigned int c = centerIndexStart + x;
            unsigned int i = centerIndexEnd + x;

            if (top) {

                // face top

                indices.insert(indices.end(), {i, i + 1, c});

            } else {

                // face bottom

                indices.insert(indices.end(), {i + 1, i, c});
            }

            groupCount += 3;
        }

        // add a group to the geometry. this will ensure multi material support

        addGroup(groupStart, groupCount, top ? 1 : 2);

        // calculate new start value for groups

        groupStart += groupCount;
    };

    if (!params.openEnded) {

        if (radiusTop > 0) generateCap(true);
        if (radiusBottom > 0) generateCap(false);
    }

    // build geometry

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
}

std::string CylinderGeometry::type() const {

    return "CylinderGeometry";
}

std::shared_ptr<CylinderGeometry> CylinderGeometry::create(const CylinderGeometry::Params& params) {

    return std::shared_ptr<CylinderGeometry>(new CylinderGeometry(params));
}

std::shared_ptr<CylinderGeometry> CylinderGeometry::create(float radiusTop, float radiusBottom, float height, unsigned int radialSegments, unsigned int heightSegments, bool openEnded, float thetaStart, float thetaLength) {

    return create(Params(radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength));
}

CylinderGeometry::Params::Params(float radiusTop, float radiusBottom, float height, unsigned int radialSegments, unsigned int heightSegments, bool openEnded, float thetaStart, float thetaLength)
    : radiusTop(radiusTop), radiusBottom(radiusBottom), height(height),
      radialSegments(radialSegments), heightSegments(heightSegments),
      openEnded(openEnded), thetaStart(thetaStart), thetaLength(thetaLength) {}
