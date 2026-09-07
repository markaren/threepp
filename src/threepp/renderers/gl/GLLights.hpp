// Backward-compatibility header. The canonical location is now common/Lights.hpp.

#ifndef THREEPP_GLLIGHTS_HPP
#define THREEPP_GLLIGHTS_HPP

#include "threepp/renderers/common/Lights.hpp"

namespace threepp::gl {

    // Aliases so existing GL code compiles without changes.
    using LightUniforms = threepp::LightUniforms;
    using UniformsCache = threepp::UniformsCache;
    using ShadowUniformsCache = threepp::ShadowUniformsCache;
    using GLLights = threepp::Lights;

}// namespace threepp::gl

#endif//THREEPP_GLLIGHTS_HPP
