// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLUniforms.js

#ifndef THREEPP_GLUNIFORMS_HPP
#define THREEPP_GLUNIFORMS_HPP

#include "threepp/core/Uniform.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace threepp::gl {

    struct GLTextures;

    struct UniformObject {

        std::string id;

        explicit UniformObject(std::string id) : id(std::move(id)) {}

        virtual void setValue(const UniformValue &value, GLTextures *textures = nullptr) = 0;

        virtual ~UniformObject() = default;
    };

    struct Container {

        std::vector<std::shared_ptr<UniformObject>> seq;
        std::unordered_map<std::string, std::shared_ptr<UniformObject>> map;

        virtual ~Container() = default;
    };

    struct GLUniforms : Container {

        explicit GLUniforms(unsigned int program);

        void setValue(const std::string &name, const UniformValue &value, GLTextures *textures = nullptr);

        static void upload(std::vector<std::shared_ptr<UniformObject>> &seq, UniformMap &values, GLTextures *textures);

        static std::vector<std::shared_ptr<UniformObject>> seqWithValue(std::vector<std::shared_ptr<UniformObject>> &seq, UniformMap &values);
    };

}// namespace threepp::gl

#endif//THREEPP_GLUNIFORMS_HPP
