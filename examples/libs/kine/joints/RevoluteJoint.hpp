
#ifndef THREEPP_REVOLUTEJOINT_HPP
#define THREEPP_REVOLUTEJOINT_HPP

#include "KineJoint.hpp"

#include "threepp/math/MathUtils.hpp"

namespace kine {

    class RevoluteJoint: public KineJoint {

    public:
        RevoluteJoint(const threepp::Vector3& axis, KineLimit limit): KineJoint(axis, limit) {}

        [[nodiscard]] threepp::Matrix4 getTransformation(float value) const override {
            return threepp::Matrix4().makeRotationAxis(axis, value * threepp::math::DEG2RAD);
        }
    };

}// namespace kine

#endif//THREEPP_REVOLUTEJOINT_HPP
