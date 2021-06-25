

#ifndef THREEPP_UNIFORMSUTIL_HPP
#define THREEPP_UNIFORMSUTIL_HPP

#include "threepp/core/Uniform.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::shaders {

    inline std::unordered_map<std::string, Uniform> mergeUniforms(const std::vector<std::unordered_map<std::string, Uniform>> &uniforms) {

        std::unordered_map<std::string, Uniform> merged;

        for (const auto &u : uniforms) {

            for (const auto &[key, value] : u) {

                merged[key] = value;
            }
        }

        return merged;
    }

}// namespace threepp::shaders

#endif//THREEPP_UNIFORMSUTIL_HPP
