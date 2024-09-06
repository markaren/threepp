
#include "threepp/extras/core/Shape.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

Shape::Shape(const std::vector<Vector2>& points)
    : Path(points), uuid_{math::generateUUID()} {}


const std::string& Shape::uuid() const {
    return uuid_;
}

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
