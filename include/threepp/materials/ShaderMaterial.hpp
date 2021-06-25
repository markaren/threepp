// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShaderMaterial.js

#ifndef THREEPP_SHADERMATERIAL_HPP
#define THREEPP_SHADERMATERIAL_HPP

#include "threepp/materials/Material.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

namespace threepp {

    class ShaderMaterial: public Material {

    public:

        std::string type() const override {
            return "ShaderMaterial";
        }

    protected:
        ShaderMaterial() = default;

    private:
        std::unordered_map<std::string, Uniform> uniforms_;

    };

}

#endif//THREEPP_SHADERMATERIAL_HPP
