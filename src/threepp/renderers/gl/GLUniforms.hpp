// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLUniforms.js

#ifndef THREEPP_GLUNIFORMS_HPP
#define THREEPP_GLUNIFORMS_HPP

#include "threepp/core/Uniform.hpp"

#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace threepp::gl {

    struct GLTextures;

    // True if any of the given values differs from the corresponding cache slot.
    // Pure comparison, independent of any GL context, so it's directly unit-testable.
    inline bool uniformCacheDiffers(const std::vector<float>& cache, std::initializer_list<float> values) {

        std::size_t i = 0;
        for (const float v : values) {
            if (cache[i] != v) return true;
            ++i;
        }
        return false;
    }

    struct UniformObject {

        std::string id;

        explicit UniformObject(std::string id): id(std::move(id)) {}

        virtual void setValue(const UniformValue& value, GLTextures* textures = nullptr) = 0;

        virtual ~UniformObject() = default;
    };

    struct Container {

        std::vector<std::unique_ptr<UniformObject>> seq;
        std::unordered_map<std::string, UniformObject*> map;

        virtual ~Container() = default;
    };

    struct GLUniforms: Container {

        explicit GLUniforms(unsigned int program);

        void setValue(const std::string& name, const UniformValue& value, GLTextures* textures = nullptr);

        static void upload(std::vector<UniformObject*>& seq, UniformMap& values, GLTextures* textures);

        static std::vector<UniformObject*> seqWithValue(const std::vector<std::unique_ptr<UniformObject>>& seq, UniformMap& values);
    };

}// namespace threepp::gl

#endif//THREEPP_GLUNIFORMS_HPP
