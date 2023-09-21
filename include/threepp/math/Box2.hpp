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

        Box2& setFromCenterAndSize(const Vector2& center, const Vector2& size);

        Box2& copy(const Box2& box);

        Box2& makeEmpty();

        [[nodiscard]] bool isEmpty() const;

        void getCenter(Vector2& target) const;

        void getSize(Vector2& target) const;

        Box2& expandByPoint(const Vector2& point);

        Box2& expandByVector(const Vector2& vector);

        Box2& expandByScalar(float scalar);

        [[nodiscard]] bool containsPoint(const Vector2& point) const;

        [[nodiscard]] bool containsBox(const Box2& box) const;

        [[nodiscard]] bool intersectsBox(const Box2& box) const;

        Vector2& clampPoint(const Vector2& point, Vector2& target) const;

        [[nodiscard]] float distanceToPoint(const Vector2& point) const;

        Box2& intersect(const Box2& box);

        Box2& _union(const Box2& box);

        Box2& translate(const Vector2& offset);

        friend std::ostream& operator<<(std::ostream& os, const Box2& v) {
            os << "Box2(min=" << v.min_ << ", max=" << v.max_ << ")";
            return os;
        }

    private:
        Vector2 min_;
        Vector2 max_;
    };

}// namespace threepp

#endif//THREEPP_BOX2_HPP
