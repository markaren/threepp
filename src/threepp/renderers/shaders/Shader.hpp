
#ifndef THREEPP_SHADER_HPP
#define THREEPP_SHADER_HPP

#include "threepp/core/Uniform.hpp"

#include <string>
#include <unordered_map>

namespace threepp::shaders {

    struct Shader {

        std::shared_ptr<UniformMap> uniforms;
        std::string vertexShader;
        std::string fragmentShader;

    };

}

#endif//THREEPP_SHADER_HPP
