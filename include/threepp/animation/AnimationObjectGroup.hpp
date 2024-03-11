
#ifndef THREEPP_ANIMATIONOBJECTGROUP_HPP
#define THREEPP_ANIMATIONOBJECTGROUP_HPP

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class AnimationObjectGroup {

    public:
        const std::string uuid{math::generateUUID()};
    };

}// namespace threepp

#endif//THREEPP_ANIMATIONOBJECTGROUP_HPP
