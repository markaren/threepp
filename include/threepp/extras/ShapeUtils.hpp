// https://github.com/mrdoob/three.js/blob/r129/src/extras/ShapeUtils.js

#ifndef THREEPP_SHAPEUTILS_HPP
#define THREEPP_SHAPEUTILS_HPP

#include "threepp/math/Vector2.hpp"

#include <algorithm>
#include <vector>

namespace threepp::shapeutils {

    // calculate area of the contour polygon

    float area(const std::vector<Vector2>& contour) {

        const auto n = contour.size();
        float a = 0.0f;

        for (unsigned p = n - 1, q = 0; q < n; p = q++) {

            a += contour[p].x * contour[q].y - contour[q].x * contour[p].y;
        }

        return a * 0.5f;
    }

    bool isClockWise(const std::vector<Vector2>& pts) {

        return area(pts) < 0;
    }


    void removeDupEndPts(std::vector<Vector2>& points) {

        const auto l = points.size();

        if (l > 2 && points[l - 1].equals(points[0])) {

            points.pop_back();
        }
    }

    void addContour(std::vector<float>& vertices, const std::vector<Vector2>& contour) {

        for (auto& i : contour) {

            vertices.emplace_back(i.x);
            vertices.emplace_back(i.y);
        }
    }

    //    std::vector<std::vector<float>> triangulateShape(std::vector<Vector2>& contour, std::vector<std::vector<Vector2>>& holes) {
    //
    //        std::vector<float> vertices;          // flat array of vertices like [ x0,y0, x1,y1, x2,y2, ... ]
    //        std::vector<unsigned int> holeIndices;// array of hole indices
    //        std::vector<std::vector<float>> faces;// final array of vertex indices like [ [ a,b,d ], [ b,c,d ] ]
    //
    //        removeDupEndPts(contour);
    //        addContour(vertices, contour);
    //
    //        //
    //
    //        auto holeIndex = contour.size();
    //
    //        for (auto& points : holes) {
    //            removeDupEndPts(points);
    //        }
    //
    //        for (const auto & hole : holes) {
    //
    //            holeIndices.emplace_back(holeIndex);
    //            holeIndex += hole.size();
    //            addContour(vertices, hole);
    //        }
    //
    //        //
    //
    //        const auto triangles = Earcut.triangulate(vertices, holeIndices);
    //
    //        //
    //
    //        for (unsigned i = 0; i < triangles.length; i += 3) {
    //
    //            faces.emplace_back(triangles.slice(i, i + 3));
    //        }
    //
    //        return faces;
    //    }


}// namespace threepp::shapeutils

#endif//THREEPP_SHAPEUTILS_HPP
