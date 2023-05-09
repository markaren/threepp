
#ifndef THREEPP_SVGTYPES_HPP
#define THREEPP_SVGTYPES_HPP

#include "threepp/math/Vector2.hpp"
#include "threepp/math/Box2.hpp"
#include "threepp/extras/core/Curve.hpp"

#include <string>
#include <vector>
#include <memory>

namespace threepp::svg {

    enum class IntersectionLocationType {
        ORIGIN,
        DESTINATION,
        BETWEEN,
        LEFT,
        RIGHT,
        BEHIND,
        BEYOND
    };

    struct ClassifyResult {
        IntersectionLocationType loc = IntersectionLocationType::ORIGIN;
        float t = 0;
    } classifyResult;

    struct EdgeIntersection {
        float x;
        float y;
        float t;
    };

    struct Intersection {
        int identifier;
        Vector2 point;
    };

    struct AHole {
        int identifier;
        bool isHole;
        std::optional<int> _for;
    };

    struct SimplePath {
        std::vector<Vector2> points;
        bool isCW;
        int identifier;
        Box2 boundingBox;
    };

}

#endif//THREEPP_SVGTYPES_HPP
