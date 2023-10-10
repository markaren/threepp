

#ifndef THREEPP_PRISMATICJOINT_HPP
#define THREEPP_PRISMATICJOINT_HPP

#include "KineJoint.hpp"

namespace kine {

    class PrismaticJoint: public KineJoint {

    public:
        PrismaticJoint(const threepp::Vector3& axis, KineLimit limit): KineJoint(axis, limit) {}

        [[nodiscard]] threepp::Matrix4 getTransformation(float value) const override {
            return threepp::Matrix4().makeTranslation(tmp_.copy(axis).multiplyScalar(value));
        }


    private:
        mutable threepp::Vector3 tmp_;
    };

}// namespace kine

#endif//THREEPP_PRISMATICJOINT_HPP
