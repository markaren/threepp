
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

    for (auto& i : contour) {

        vertices.emplace_back(i.x);
        vertices.emplace_back(i.y);
    }
}

std::vector<std::vector<unsigned int>> shapeutils::triangulateShape(std::vector<Vector2>& contour, std::vector<std::vector<Vector2>>& holes) {

    using Coord = float;
    using N = unsigned int;
    using Point = std::array<Coord, 2>;

    std::vector<Point> vertices;
    std::vector<std::vector<Point>> holes_;


    removeDupEndPts(contour);
    for (const auto& p : contour) {
        vertices.emplace_back(std::array<float, 2>{p.x, p.y});
    }

    for (auto& points : holes) {
        removeDupEndPts(points);
        holes_.emplace_back();
        for (auto& p : points) {
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

    std::vector<std::vector<unsigned int>> faces;
    for (unsigned i = 0; i < triangles.size(); i += 3) {

        faces.insert(faces.end(), {triangles.begin() + i, triangles.begin() + i + 3});
    }

    return faces;
}
