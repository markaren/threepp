

#ifndef THREEPP_JOINT_HPP
#define THREEPP_JOINT_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include "../KineComponent.hpp"
#include "../KineLimit.hpp"

namespace kine {

    class KineJoint: public KineComponent {

    public:
        const threepp::Vector3& axis;
        const KineLimit limit;

        KineJoint(const threepp::Vector3& axis, KineLimit limit)
            : axis(axis),
              limit(limit){};

        [[nodiscard]] float getJointValue() const {
            return value_;
        }

        void setJointValue(float value) {
            value_ = value;
            limit.clampWithinLimit(value_);
        }

        [[nodiscard]] threepp::Matrix4 getTransformation() const override {
            return getTransformation(value_);
        }

        [[nodiscard]] virtual threepp::Matrix4 getTransformation(float value) const = 0;

    private:
        float value_{};
    };

}// namespace kine


#endif//THREEPP_JOINT_HPP
