//

#ifndef THREEPP_GLTEXTURES_HPP
#define THREEPP_GLTEXTURES_HPP

#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLProperties.hpp"
#include "threepp/renderers/gl/GLState.hpp"

#include <memory>

namespace threepp::gl {

    class GLTextures {

    public:
        GLTextures(
                std::shared_ptr<GLState> state,
                std::shared_ptr<GLProperties> properties,
                GLCapabilities capabilities,
                std::shared_ptr<GLInfo> info) {}
    };

}// namespace threepp::gl

#endif//THREEPP_GLTEXTURES_HPP
