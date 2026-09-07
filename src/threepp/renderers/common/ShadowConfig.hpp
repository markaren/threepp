// Backend-neutral shadow map configuration.
// Decouples ProgramParameters from GL-specific GLShadowMap.

#ifndef THREEPP_COMMON_SHADOWCONFIG_HPP
#define THREEPP_COMMON_SHADOWCONFIG_HPP

#include "threepp/constants.hpp"

namespace threepp {

    struct ShadowConfig {

        bool enabled = false;
        ShadowMap type = ShadowMap::PFC;
    };

}// namespace threepp

#endif//THREEPP_COMMON_SHADOWCONFIG_HPP
