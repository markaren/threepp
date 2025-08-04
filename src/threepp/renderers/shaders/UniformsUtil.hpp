
#ifndef THREEPP_UNIFORMSUTIL_HPP
#define THREEPP_UNIFORMSUTIL_HPP

#include "threepp/core/Uniform.hpp"

#include <unordered_map>
#include <vector>

namespace threepp::shaders {

    inline UniformMap mergeUniforms(const std::vector<UniformMap>& uniforms) {

        auto merged = UniformMap();

        for (const auto& u : uniforms) {

            for (const auto& [key, value] : u) {

                merged[key] = value;
            }
        }

        return merged;
    }

}// namespace threepp::shaders

#endif//THREEPP_UNIFORMSUTIL_HPP
