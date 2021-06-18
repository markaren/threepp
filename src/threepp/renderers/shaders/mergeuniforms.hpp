

#ifndef THREEPP_MERGEUNIFORMS_HPP
#define THREEPP_MERGEUNIFORMS_HPP

#include "threepp/core/Uniform.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace threepp::shaders {

    std::unordered_map<std::string, Uniform> mergeUniforms(const std::vector<std::unordered_map<std::string, Uniform>> &uniforms) {

        std::unordered_map merged;

        for (const auto& map : uniforms) {

            for (const auto& [key, value] : map) {

                merged[key] = value;

            }
        }

        return merged;

    }

}

#endif//THREEPP_MERGEUNIFORMS_HPP
