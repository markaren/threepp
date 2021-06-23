// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLPrograms.js

#ifndef THREEPP_GLPROGRAMS_HPP
#define THREEPP_GLPROGRAMS_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/renderers/gl/GLCapabilities.hpp"

#include <glad/glad.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace threepp::gl {



    class GLPrograms {

        struct Parameters;

        bool logarithmicDepthBuffer;
        bool floatVertexTextures;
        GLint maxVertexUniforms;
        bool vertexTextures;

        GLPrograms(const GLCapabilities &capabilities)
                : logarithmicDepthBuffer(capabilities.logarithmicDepthBuffer),
                  floatVertexTextures(capabilities.floatVertexTextures),
                  maxVertexUniforms(capabilities.maxVertexUniforms),
                  vertexTextures(capabilities.vertexTextures) {}

    };

}

#endif//THREEPP_GLPROGRAMS_HPP
