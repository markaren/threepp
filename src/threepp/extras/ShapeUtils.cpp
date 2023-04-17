
#include "threepp/extras/ShapeUtils.hpp"

#include "earcut.hpp"

#include <array>

using namespace threepp;


float shapeutils::area(const std::vector<Vector2>& contour) {

    const auto n = contour.size();
    float a = 0.0f;

    for (unsigned p = n - 1, q = 0; q < n; p = q++) {

        a += contour[p].x * contour[q].y - contour[q].x * contour[p].y;
    }

    return a * 0.5f;
}

bool shapeutils::isClockWise(const std::vector<Vector2>& pts) {

    return area(pts) < 0;
}

void shapeutils::removeDupEndPts(std::vector<Vector2>& points) {

    const auto l = points.size();

    if (l > 2 && points[l - 1].equals(points[0])) {

        points.pop_back();
    }
}

void shapeutils::addContour(std::vector<float>& vertices, const std::vector<Vector2>& contour) {

    for (const auto& i : contour) {

        vertices.emplace_back(i.x);
        vertices.emplace_back(i.y);
    }
}

std::vector<std::vector<unsigned int>> shapeutils::triangulateShape(std::vector<Vector2>& contour, std::vector<std::vector<Vector2>>& holes) {

    using Coord = float;
    using N = unsigned int;
    using Point = std::array<Coord, 2>;

    removeDupEndPts(contour);

    std::vector<Point> vertices;
    vertices.reserve(contour.size() * 2);
    std::vector<std::vector<Point>> holes_;
    holes_.reserve(holes.size());

    for (const auto& p : contour) {
        vertices.emplace_back(std::array<float, 2>{p.x, p.y});
    }

    for (auto& points : holes) {
        removeDupEndPts(points);
        holes_.emplace_back();
        for (const auto& p : points) {
            holes_.back().emplace_back(std::array<float, 2>{p.x, p.y});
        }
    }

    std::vector<std::vector<Point>> polygon;
    polygon.emplace_back(vertices);
    for (const auto& hole : holes_) {
        polygon.emplace_back(hole);
    }
    const auto triangles = mapbox::earcut<N>(polygon);

    //

    std::vector<std::vector<unsigned int>> faces(triangles.size() / 3);
    for (unsigned i = 0, j = 0; i < triangles.size(); i += 3, j++) {
        faces[j].emplace_back(triangles[i]);
        faces[j].emplace_back(triangles[i + 1]);
        faces[j].emplace_back(triangles[i + 2]);
    }

    return faces;
}
