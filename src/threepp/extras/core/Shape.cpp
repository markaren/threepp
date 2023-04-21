
#include "threepp/extras/core/Shape.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

Shape::Shape(const std::optional<std::vector<Vector2>>& points)
    : Path(points), uuid{math::generateUUID()} {}

std::vector<std::vector<Vector2>> Shape::getPointsHoles(unsigned int divisions) const {

    std::vector<std::vector<Vector2>> holesPts;

    for (const auto& hole : this->holes) {

        auto points = hole->getPoints(divisions);
        holesPts.emplace_back(points);
    }

    return holesPts;
}

ShapePoints Shape::extractPoints(unsigned int divisions) const {

    return ShapePoints{
            this->getPoints(divisions),
            this->getPointsHoles(divisions)};
}
