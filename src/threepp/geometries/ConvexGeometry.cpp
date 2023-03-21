
#include "threepp/geometries/ConvexGeometry.hpp"

#include "threepp/extras/quickhull.hpp"

#include <array>

using namespace threepp;


ConvexGeometry::ConvexGeometry(const std::vector<Vector3>& _points) {

    using F = float;
    constexpr std::size_t dim = 3;
    using Points = std::vector<std::array<F, dim>>;
    const auto eps = std::numeric_limits<float>::epsilon();

    Points points;
    points.reserve(_points.size());
    for (const auto& p : _points) {
        points.emplace_back(std::array<F, dim>{p.x, p.y, p.z});
    }

    quick_hull<typename Points::const_iterator> qh{dim, eps};
    qh.add_points(std::cbegin(points), std::cend(points));

    auto initial_simplex = qh.get_affine_basis();
    if (initial_simplex.size() < dim + 1) {
        throw std::runtime_error("[ConvexHull] degenerated input set");
    }

    qh.create_initial_simplex(std::cbegin(initial_simplex), std::prev(std::cend(initial_simplex)));
    qh.create_convex_hull();
    if (!qh.check()) {
        throw std::runtime_error("[ConvexHull] resulted structure is not convex (generally due to precision errors)");
    }

    // buffers

    std::vector<float> vertices;
    std::vector<float> normals;

    for (const auto& f : qh.facets_) {
        auto& n = f.normal_;
        normals.emplace_back(n[0]);
        normals.emplace_back(n[1]);
        normals.emplace_back(n[2]);

        auto& verts = f.vertices_;
        for (auto& it : verts) {
            vertices.insert(vertices.end(), it->begin(), it->end());
        }
    }

    // build geometry

    this->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    this->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
}

std::shared_ptr<ConvexGeometry> ConvexGeometry::create(const std::vector<Vector3>& points) {

    return std::shared_ptr<ConvexGeometry>(new ConvexGeometry(points));
}
