
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

    protected:
        // A copy aims at the same object as its source. The default target
        // cannot be moved through this interface, so there is nothing to copy
        // when the source has none.
        void copyTarget(const LightWithTarget& source) {

            this->target_ = source.target_;
        }

    private:
        Object3D* target_ = nullptr;
        Object3D defaultTarget;
    };

}// namespace threepp

#endif//THREEPP_LIGHT_INTERFACES_HPP
