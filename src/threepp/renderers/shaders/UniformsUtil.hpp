

#ifndef THREEPP_UNIFORMSUTIL_HPP
#define THREEPP_UNIFORMSUTIL_HPP

#include "threepp/core/Uniform.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::shaders {

    inline std::shared_ptr<UniformMap> mergeUniforms(const std::vector<UniformMap> &uniforms) {

        auto merged = std::make_shared<UniformMap>();

        for (const auto &u : uniforms) {

            for (const auto &[key, value] : u) {

                merged->operator[](key) = value;
            }
        }

        return merged;
    }

}// namespace threepp::shaders

#endif//THREEPP_UNIFORMSUTIL_HPP
