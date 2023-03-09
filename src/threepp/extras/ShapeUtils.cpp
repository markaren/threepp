
#include "threepp/extras/ShapeUtils.hpp"

#include "earcut.hpp"

#include <array>

using namespace threepp;


std::vector<std::vector<unsigned int>> shapeutils::triangulateShape(std::vector<Vector2>& contour, std::vector<std::vector<Vector2>>& holes) {

    using Coord = float;
    using N = unsigned int;
    using Point = std::array<Coord, 2>;

    std::vector<Point> vertices;
    std::vector<Point> holes_;

    removeDupEndPts(contour);
    for (const auto& p : contour) {
        vertices.emplace_back(std::array<float, 2>{p.x, p.y});
    }

    for (auto& points : holes) {
        removeDupEndPts(points);
        for (auto& p : points) {
            holes_.emplace_back(std::array<float, 2>{p.x, p.y});
        }
    }

    std::vector<std::vector<Point>> polygon;
    polygon.emplace_back(vertices);
    polygon.emplace_back(holes_);
    const auto triangles = mapbox::earcut<N>(polygon);

    //

    std::vector<std::vector<unsigned int>> faces;
    for (unsigned i = 0; i < triangles.size(); i += 3) {

        faces.insert(faces.end(), {triangles.begin() + i, triangles.begin() + i + 3});
    }

    return faces;
}
