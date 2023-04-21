
#include "threepp/geometries/WireframeGeometry.hpp"

#include <list>

using namespace threepp;

WireframeGeometry::WireframeGeometry(const BufferGeometry& geometry) {

    // buffer

    std::list<float> vertices;

    // helper variables

    std::pair<unsigned int, unsigned int> edge;
    std::unordered_map<std::string, std::pair<int, int>> edges;

    Vector3 vertex;

    if (geometry.hasIndex()) {

        // indexed BufferGeometry

        auto position = geometry.getAttribute<float>("position");
        auto indices = geometry.getIndex();
        auto groups = geometry.groups;

        if (groups.empty()) {
            groups = {GeometryGroup{0, indices->count(), 0}};
        }

        // create a data structure that contains all eges without duplicates

        for (const auto& group : groups) {

            const auto start = group.start;
            const auto count = group.count;

            for (int i = start, l = (start + count); i < l; i += 3) {

                for (int j = 0; j < 3; j++) {

                    const auto edge1 = indices->getX(i + j);
                    const auto edge2 = indices->getX(i + (j + 1) % 3);
                    edge.first = std::min(edge1, edge2);// sorting prevents duplicates
                    edge.second = std::max(edge1, edge2);

                    std::string key = std::to_string(edge.first) + ',' + std::to_string(edge.second);

                    if (!edges.count(key)) {

                        edges[key] = edge;
                    }
                }
            }
        }

        // generate vertices

        for (const auto& [key, e] : edges) {

            position->setFromBufferAttribute(vertex, e.first);
            vertices.insert(vertices.begin(), {vertex.x, vertex.y, vertex.z});

            position->setFromBufferAttribute(vertex, e.second);
            vertices.insert(vertices.begin(), {vertex.x, vertex.y, vertex.z});
        }

    } else {

        // non-indexed BufferGeometry

        auto position = geometry.getAttribute<float>("position");

        for (int i = 0, l = (position->count() / 3); i < l; i++) {

            for (int j = 0; j < 3; j++) {

                // three edges per triangle, an edge is represented as (index1, index2)
                // e.g. the first triangle has the following edges: (0,1),(1,2),(2,0)

                const auto index1 = 3 * i + j;
                position->setFromBufferAttribute(vertex, index1);
                vertices.insert(vertices.begin(), {vertex.x, vertex.y, vertex.z});

                const auto index2 = 3 * i + ((j + 1) % 3);
                position->setFromBufferAttribute(vertex, index2);
                vertices.insert(vertices.begin(), {vertex.x, vertex.y, vertex.z});
            }
        }
    }

    // build geometry

    setAttribute("position", FloatBufferAttribute::create(vertices, 3));
}

std::string WireframeGeometry::type() const {

    return "WireframeGeometry";
}

std::shared_ptr<WireframeGeometry> WireframeGeometry::create(const BufferGeometry& geometry) {

    return std::shared_ptr<WireframeGeometry>(new WireframeGeometry(geometry));
}
