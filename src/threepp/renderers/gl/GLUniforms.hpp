// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLUniforms.js

#ifndef THREEPP_GLUNIFORMS_HPP
#define THREEPP_GLUNIFORMS_HPP

#include "threepp/core/Uniform.hpp"

#include <any>
#include <unordered_map>
#include <vector>

namespace threepp::gl {

    struct GLTextures;

    struct UniformObject {

        std::string id;

        virtual void setValue(const UniformValue &value, GLTextures *textures = nullptr) = 0;
    };


    struct StructuredUniform {


    private:

        std::vector<UniformObject*> seq;
        std::unordered_map<std::string, UniformObject*> map;
    };

    struct ActiveUniformInfo {

        char name[256];
        int size;
        unsigned int type;

        ActiveUniformInfo(unsigned int program, unsigned int index );

    };

    struct GLUniforms {

        std::vector<UniformObject*> seq;
        std::unordered_map<std::string, UniformObject*> map;

        explicit GLUniforms(unsigned int program);

        void setValue(const std::string &name, const UniformValue &value, GLTextures* textures = nullptr);

    };

    inline void upload(std::vector<UniformObject*> &seq, std::unordered_map<std::string, Uniform> &values, GLTextures *textures);

    inline std::vector<UniformObject*> seqWithValue(std::vector<UniformObject*> &seq, std::unordered_map<std::string, Uniform> &values);


}// namespace threepp::gl

#endif//THREEPP_GLUNIFORMS_HPP
