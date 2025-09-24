// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLMaterials.js

#ifndef THREEPP_GLMATERIAL_HPP
#define THREEPP_GLMATERIAL_HPP

#include "threepp/materials/Material.hpp"

#include "threepp/core/Uniform.hpp"
#include "threepp/scenes/Scene.hpp"

namespace threepp::gl {

    class GLProperties;

    struct GLMaterials {

        explicit GLMaterials(GLProperties& properties);

        void refreshFogUniforms(UniformMap& uniforms, FogVariant& fog);

        void refreshMaterialUniforms(UniformMap& uniforms, Material* material, float pixelRatio, int height);

        ~GLMaterials();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };


}// namespace threepp::gl

#endif//THREEPP_GLMATERIAL_HPP
