
#ifndef THREEPP_SHADER_HPP
#define THREEPP_SHADER_HPP

#include "threepp/core/Uniform.hpp"

#include <string>

namespace threepp {

    struct Shader {

        UniformMap uniforms;
        std::string vertexShader;
        std::string fragmentShader;
    };

}// namespace threepp

#endif//THREEPP_SHADER_HPP
