
#ifndef THREEPP_LIGHTWITHSHADOW_HPP
#define THREEPP_LIGHTWITHSHADOW_HPP

#include "LightShadow.hpp"

namespace threepp {

    class LightWithShadow {

    public:

        virtual LightShadow *getLightShadow() = 0;

    };

}

#endif//THREEPP_LIGHTWITHSHADOW_HPP
