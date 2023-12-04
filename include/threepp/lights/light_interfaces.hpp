
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
        std::shared_ptr<Object3D> target{Object3D::create()};

        virtual ~LightWithTarget() = default;
    };

}// namespace threepp

#endif//THREEPP_LIGHT_INTERFACES_HPP
