// https://github.com/mrdoob/three.js/blob/r129/src/materials/RawShaderMaterial.js

#ifndef THREEPP_RAWSHADERMATERIAL_HPP
#define THREEPP_RAWSHADERMATERIAL_HPP

#include "threepp/materials/ShaderMaterial.hpp"

namespace threepp {

    class RawShaderMaterial: public ShaderMaterial {

    public:
        std::string type() const override {
            return "RawShaderMaterial";
        }

        static std::shared_ptr<RawShaderMaterial> create() {
            return std::shared_ptr<RawShaderMaterial>(new ShaderMaterial());
        }

    protected:
        RawShaderMaterial() = default;

    };

}

#endif//THREEPP_RAWSHADERMATERIAL_HPP
