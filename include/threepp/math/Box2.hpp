// https://github.com/mrdoob/three.js/blob/r129/src/math/Box2.js

#ifndef THREEPP_BOX2_HPP
#define THREEPP_BOX2_HPP

#include "threepp/math/Vector2.hpp"

#include <ostream>
#include <vector>

namespace threepp {

    class Box2 {

    public:
        Box2();

        [[nodiscard]] const Vector2& getMin() const;

        [[nodiscard]] const Vector2& getMax() const;

        Box2(const Vector2& min, const Vector2& max);

        Box2& set(const Vector2& min, const Vector2& max);

        Box2& setFromPoints(const std::vector<Vector2>& points);

        Box2& copy(const Box2& box);

        Box2& makeEmpty();

        [[nodiscard]] bool isEmpty() const;

        void getCenter(Vector2& target);

        void getSize(Vector2& target);

        Box2& expandByPoint(const Vector2& point);

        friend std::ostream& operator<<(std::ostream& os, const Box2& v) {
            os << "Box2(max=" << v.min_ << ", max=" << v.max_ << ")";
            return os;
        }

    private:
        Vector2 min_;
        Vector2 max_;
    };

}// namespace threepp

#endif//THREEPP_BOX2_HPP
