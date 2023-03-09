// https://github.com/mrdoob/three.js/blob/r129/src/extras/ShapeUtils.js

#ifndef THREEPP_SHAPEUTILS_HPP
#define THREEPP_SHAPEUTILS_HPP

#include "threepp/math/Vector2.hpp"

#include <algorithm>
#include <vector>

namespace threepp::shapeutils {

    // calculate area of the contour polygon

    float area(const std::vector<Vector2>& contour);

    bool isClockWise(const std::vector<Vector2>& pts);

    void removeDupEndPts(std::vector<Vector2>& points);

    void addContour(std::vector<float>& vertices, const std::vector<Vector2>& contour);

    std::vector<std::vector<unsigned int>> triangulateShape(std::vector<Vector2>& contour, std::vector<std::vector<Vector2>>& holes);


}// namespace threepp::shapeutils

#endif//THREEPP_SHAPEUTILS_HPP
