
#include "threepp/geometries/WireframeGeometry.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace threepp;

WireframeGeometry::WireframeGeometry(const BufferGeometry& geometry) {

    // buffer

    std::vector<float> vertices;

    Vector3 vertex;

    if (geometry.hasIndex()) {

        // indexed BufferGeometry

        auto position = geometry.getAttribute<float>("position");
        auto indices = geometry.getIndex();
        auto groups = geometry.groups;

        if (groups.empty()) {
            groups = {GeometryGroup{0, indices->count(), 0}};
        }

        // every edge once, in the order the triangles first name it

        std::vector<std::pair<unsigned int, unsigned int>> edges;
        std::unordered_set<std::uint64_t> seen;

        for (const auto& group : groups) {

            const auto start = group.start;
            const auto count = group.count;

            for (int i = start, l = (start + count); i < l; i += 3) {

                for (int j = 0; j < 3; j++) {

                    const unsigned int edge1 = indices->getX(i + j);
                    const unsigned int edge2 = indices->getX(i + (j + 1) % 3);
                    const unsigned int first = std::min(edge1, edge2);// sorting prevents duplicates
                    const unsigned int second = std::max(edge1, edge2);

                    if (seen.insert((static_cast<std::uint64_t>(first) << 32u) | second).second) {

                        edges.emplace_back(first, second);
                    }
                }
            }
        }

        // generate vertices

        vertices.reserve(edges.size() * 6);

        for (const auto& e : edges) {

            position->setFromBufferAttribute(vertex, e.first);
            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

            position->setFromBufferAttribute(vertex, e.second);
            vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});
        }

    } else {

        // non-indexed BufferGeometry

        auto position = geometry.getAttribute<float>("position");

        const int triangles = position->count() / 3;
        vertices.reserve(static_cast<size_t>(triangles) * 18);

        for (int i = 0; i < triangles; i++) {

            for (int j = 0; j < 3; j++) {

                // three edges per triangle, an edge is represented as (index1, index2)
                // e.g. the first triangle has the following edges: (0,1),(1,2),(2,0)

                const auto index1 = 3 * i + j;
                position->setFromBufferAttribute(vertex, index1);
                vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});

                const auto index2 = 3 * i + ((j + 1) % 3);
                position->setFromBufferAttribute(vertex, index2);
                vertices.insert(vertices.end(), {vertex.x, vertex.y, vertex.z});
            }
        }
    }

    // build geometry

    setAttribute("position", FloatBufferAttribute::create(std::move(vertices), 3));
}

const std::string& WireframeGeometry::type() const {

    static const std::string typeName = "WireframeGeometry";
    return typeName;
}

std::shared_ptr<WireframeGeometry> WireframeGeometry::create(const BufferGeometry& geometry) {

    return std::shared_ptr<WireframeGeometry>(new WireframeGeometry(geometry));
}
