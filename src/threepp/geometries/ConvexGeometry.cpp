
#include "threepp/geometries/ConvexGeometry.hpp"

#include "threepp/math/ConvexHull.hpp"

using namespace threepp;


ConvexGeometry::ConvexGeometry(const std::vector<Vector3>& points) {

    // buffers

    std::vector<float> vertices;
    std::vector<float> normals;

    auto convexHull = ConvexHull();
    convexHull.setFromPoints(points);

    // generate vertices and normals

    const auto& faces = convexHull.faces;

    for (const auto& face : faces) {

        auto edge = face->edge;

        // we move along a doubly-connected edge list to access all face points (see HalfEdge docs)

        do {

            const auto point = edge->head()->point;

            vertices.insert(vertices.end(), {point.x, point.y, point.z});
            normals.insert(normals.end(), {face->normal.x, face->normal.y, face->normal.z});

            edge = edge->next;

        } while (edge != face->edge);
    }

    // build geometry

    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
}

std::shared_ptr<ConvexGeometry> ConvexGeometry::create(const std::vector<Vector3>& points) {

    return std::shared_ptr<ConvexGeometry>(new ConvexGeometry(points));
}
