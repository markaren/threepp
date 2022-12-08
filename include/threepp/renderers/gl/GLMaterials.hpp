// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLMaterials.js

#ifndef THREEPP_GLMATERIAL_HPP
#define THREEPP_GLMATERIAL_HPP

#include "GLProperties.hpp"

#include "threepp/materials/Material.hpp"

#include "threepp/core/Uniform.hpp"
#include "threepp/scenes/Scene.hpp"

namespace threepp::gl {

    struct GLMaterials {

        explicit GLMaterials(GLProperties &properties)
            : properties(properties) {}

        void refreshFogUniforms(UniformMap &uniforms, FogVariant &fog);

        void refreshMaterialUniforms(UniformMap &uniforms, Material *material, int pixelRatio, int height);

    private:
        GLProperties &properties;
    };


}// namespace threepp::gl

#endif//THREEPP_GLMATERIAL_HPP
