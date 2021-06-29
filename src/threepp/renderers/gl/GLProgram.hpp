// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProgram.js

#ifndef THREEPP_GLPROGRAM_HPP
#define THREEPP_GLPROGRAM_HPP

#include <unordered_map>
#include <utility>

namespace threepp::gl {

    struct GLProgram {

        int id = programIdCount++;

        int usedTimes;

        std::string cacheKey;

        GLProgram(std::string cacheKey): cacheKey(std::move(cacheKey))  {}

    private:
        inline static int programIdCount = 0;
    };

}// namespace threepp::gl

#endif//THREEPP_GLPROGRAM_HPP
