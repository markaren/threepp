// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLUniforms.js

#ifndef THREEPP_GLUNIFORMS_HPP
#define THREEPP_GLUNIFORMS_HPP

#include "GLTextures.hpp"
#include "threepp/core/Uniform.hpp"

#include <glad/glad.h>

#include <any>
#include <unordered_map>
#include <vector>

namespace threepp::gl {

    struct UniformObject {

        std::string id;

        virtual void setValue(UniformValue &value, GLTextures &textures) = 0;
    };


    struct StructuredUniform {

        StructuredUniform(const GLuint id) : id(id) {}


    private:
        GLuint id;

    private:
        std::vector<UniformObject> seq;
        std::unordered_map<std::string, UniformObject> map;
    };

    struct ActiveUniformInfo {

        GLchar name[256];
        GLsizei size;
        GLenum type;

        ActiveUniformInfo(GLuint program, GLuint index ) {

            GLsizei length;
            GLint size;
            GLenum type;
            glGetActiveUniform(program, index, 256, &length, &size, &type, name);

        }

    };

    struct GLUniforms {

        std::vector<UniformObject> seq;
        std::unordered_map<std::string, UniformObject> map;

        GLUniforms(GLuint program) {

            GLint n;
            glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &n);

            for (int i = 0; i < n; i++) {

                ActiveUniformInfo info(program, i);
                glGetUniformLocation(program, info.name);

                //TODO
            }

        }

    };

    static void upload(std::vector<UniformObject> &seq, std::unordered_map<std::string, Uniform> &values, GLTextures &textures) {

        for (int i = 0, n = seq.size(); i != n; ++i) {

            auto &u = seq[i];
            Uniform &v = values[u.id];

            if (!v.needsUpdate || *v.needsUpdate) {

                // note: always updating when .needsUpdate is undefined
                u.setValue(v.value(), textures);
            }
        }
    }

    static std::vector<UniformObject> seqWithValue(std::vector<UniformObject> &seq, std::unordered_map<std::string, Uniform> &values) {

        std::vector<UniformObject> r;

        for (int i = 0, n = seq.size(); i != n; ++i) {

            auto &u = seq[i];
            if (values.count(u.id)) r.emplace_back(u);
        }

        return r;
    }


}// namespace threepp::gl

#endif//THREEPP_GLUNIFORMS_HPP
