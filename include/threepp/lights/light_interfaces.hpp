
#ifndef THREEPP_LIGHT_INTERFACES_HPP
#define THREEPP_LIGHT_INTERFACES_HPP

#include "threepp/lights/LightShadow.hpp"

namespace threepp {

    class LightWithShadow {

    public:
        std::shared_ptr<LightShadow> shadow;

        virtual ~LightWithShadow() = default;

    protected:
        explicit LightWithShadow(const std::shared_ptr<LightShadow>& shadow): shadow(shadow) {}
    };

    class LightWithTarget {

    public:
        [[nodiscard]] const Object3D& target() const {

            return target_ ? *target_ : defaultTarget;
        }

        void setTarget(Object3D& target) {

            this->target_ = &target;
        }

        virtual ~LightWithTarget() = default;

    private:
        Object3D* target_ = nullptr;
        Object3D defaultTarget;
    };

}// namespace threepp

#endif//THREEPP_LIGHT_INTERFACES_HPP
