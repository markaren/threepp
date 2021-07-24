
#ifndef THREEPP_LIGHT_INTERFACES_HPP
#define THREEPP_LIGHT_INTERFACES_HPP

#include "LightShadow.hpp"

namespace threepp {

    template <class LightShadowType>
    class LightWithShadow {

    public:

        std::shared_ptr<LightShadowType> shadow;

        virtual ~LightWithShadow() = default;

    protected:
        explicit LightWithShadow(const std::shared_ptr<LightShadowType> &shadow) : shadow(shadow) {}

    };

    class LightWithTarget {

    public:

        std::shared_ptr<Object3D> target{Object3D::create()};

        virtual ~LightWithTarget() = default;

    };

}

#endif//THREEPP_LIGHT_INTERFACES_HPP
