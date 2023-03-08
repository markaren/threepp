// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Shape.js

#ifndef THREEPP_SHAPE_HPP
#define THREEPP_SHAPE_HPP

#include "threepp/extras/core/Path.hpp"

namespace threepp {

    struct ShapePoints {

        std::vector<Vector2> shape;
        std::vector<std::vector<Vector2>> holes;
    };

    class Shape: public Path {

    public:
        std::string uuid{math::generateUUID()};

        explicit Shape(const std::vector<Vector2>& points): Path(points) {}

        std::vector<std::vector<Vector2>> getPointsHoles(unsigned int divisions) {

            std::vector<std::vector<Vector2>> holesPts;

            for (const auto& hole : this->holes) {

                auto points = hole->getPoints(divisions);
                holesPts.emplace_back(points);
            }

            return holesPts;
        }

        // get points of shape and holes (keypoints based on segments parameter)
        ShapePoints extractPoints(unsigned int divisions) {

            return ShapePoints{
                    this->getPoints(divisions),
                    this->getPointsHoles(divisions)};
        }

    private:
        std::vector<std::unique_ptr<Path>> holes;
    };

}// namespace threepp

#endif//THREEPP_SHAPE_HPP
