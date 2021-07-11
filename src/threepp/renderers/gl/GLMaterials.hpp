// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLMaterials.js

#ifndef THREEPP_GLMATERIAL_HPP
#define THREEPP_GLMATERIAL_HPP

#include "threepp/materials/materials.hpp"

#include "threepp/core/Uniform.hpp"
#include "threepp/scenes/Scene.hpp"

namespace threepp::gl {

    struct GLMaterials {

        void refreshFogUniforms(std::shared_ptr<UniformMap> &uniforms, FogVariant &fog);

        void refreshMaterialUniforms(std::shared_ptr<UniformMap> &uniforms, Material *material, int pixelRatio, float height);

    };


}// namespace threepp::gl

#endif//THREEPP_GLMATERIAL_HPP
