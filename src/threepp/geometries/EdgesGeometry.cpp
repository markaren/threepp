
#include "threepp/geometries/EdgesGeometry.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Triangle.hpp"

#include <cmath>
#include <sstream>

using namespace threepp;

namespace {

    Vector3 _v0;
    Vector3 _v1;
    Vector3 _normal;
    Triangle _triangle;

    struct EdgeData {
        unsigned int index0;
        unsigned int index1;
        Vector3 normal;
    };

}// namespace


EdgesGeometry::EdgesGeometry(const BufferGeometry& geometry, float thresholdAngle) {

    const auto precisionPoints = 4;
    const auto precision = std::pow(10, precisionPoints);
    const auto thresholdDot = std::cos(math::DEG2RAD * thresholdAngle);

    const auto indexAttr = geometry.getIndex();
    const auto positionAttr = geometry.getAttribute<float>("position");
    const auto indexCount = indexAttr ? indexAttr->count() : positionAttr->count();

    std::vector<unsigned int> indexArr{0, 0, 0};
    std::vector vertKeys{'a', 'b', 'c'};
    std::vector<std::string> hashes(3);

    std::unordered_map<std::string, std::optional<EdgeData>> edgeData;
    std::vector<float> vertices;
    for (int i = 0; i < indexCount; i += 3) {

        if (indexAttr) {

            indexArr[0] = indexAttr->getX(i);
            indexArr[1] = indexAttr->getX(i + 1);
            indexArr[2] = indexAttr->getX(i + 2);

        } else {

            indexArr[0] = i;
            indexArr[1] = i + 1;
            indexArr[2] = i + 2;
        }

        Vector3 a, b, c;
        positionAttr->setFromBufferAttribute(a, indexArr[0]);
        positionAttr->setFromBufferAttribute(b, indexArr[1]);
        positionAttr->setFromBufferAttribute(c, indexArr[2]);
        _triangle.set(a, b, c);
        _triangle.getNormal(_normal);

        // create hashes for the edge from the vertices
        std::stringstream ss;
        ss << std::round(a.x * precision) << "," << std::round(a.y * precision) << "," << std::round(a.z * precision);
        hashes[0] = ss.str();
        ss.str(std::string());
        ss << std::round(b.x * precision) << "," << std::round(b.y * precision) << "," << std::round(b.z * precision);
        hashes[1] = ss.str();
        ss.str(std::string());
        ss << std::round(c.x * precision) << "," << std::round(c.y * precision) << "," << std::round(c.z * precision);
        hashes[2] = ss.str();
        ss.str(std::string());

        // skip degenerate triangles
        if (hashes[0] == hashes[1] || hashes[1] == hashes[2] || hashes[2] == hashes[0]) {

            continue;
        }

        // iterate over every edge
        for (unsigned j = 0; j < 3; j++) {

            // get the first and next vertex making up the edge
            const auto jNext = (j + 1) % 3;
            const auto vecHash0 = hashes[j];
            const auto vecHash1 = hashes[jNext];
            const auto v0 = _triangle[vertKeys[j]];
            const auto v1 = _triangle[vertKeys[jNext]];

            ss << vecHash0 << '_' << vecHash1;
            const auto hash = ss.str();
            ss.str(std::string());
            ss << vecHash1 << '_' << vecHash0;
            const auto reverseHash = ss.str();
            ss.str(std::string());

            if (edgeData.contains(reverseHash) && edgeData.at(reverseHash)) {

                // if we found a sibling edge add it into the vertex array if
                // it meets the angle threshold and delete the edge from the map.
                if (_normal.dot(edgeData[reverseHash]->normal) <= thresholdDot) {

                    vertices.insert(vertices.end(), {v0.x, v0.y, v0.z});
                    vertices.insert(vertices.end(), {v1.x, v1.y, v1.z});
                }

                edgeData[reverseHash] = std::nullopt;

            } else if (!(edgeData[hash])) {

                // if we've already got an edge here then skip adding a new one
                edgeData[hash] = {

                        indexArr[j],
                        indexArr[jNext],
                        _normal,

                };
            }
        }
    }

    // iterate over all remaining, unmatched edges and add them to the vertex array
    for (const auto& [key, data] : edgeData) {

        if (data) {

            positionAttr->setFromBufferAttribute(_v0, data->index0);
            positionAttr->setFromBufferAttribute(_v1, data->index1);

            vertices.insert(vertices.end(), {_v0.x, _v0.y, _v0.z});
            vertices.insert(vertices.end(), {_v1.x, _v1.y, _v1.z});
        }
    }

    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
}

std::string EdgesGeometry::type() const {

    return "EdgesGeometry";
}

std::shared_ptr<EdgesGeometry> EdgesGeometry::create(const BufferGeometry& geometry, float thresholdAngle) {

    return std::shared_ptr<EdgesGeometry>(new EdgesGeometry(geometry, thresholdAngle));
}
