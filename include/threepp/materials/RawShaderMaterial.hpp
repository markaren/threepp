// https://github.com/mrdoob/three.js/blob/r129/src/materials/RawShaderMaterial.js

#ifndef THREEPP_RAWSHADERMATERIAL_HPP
#define THREEPP_RAWSHADERMATERIAL_HPP

#include "threepp/materials/ShaderMaterial.hpp"

namespace threepp {

    class RawShaderMaterial: public ShaderMaterial {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<RawShaderMaterial> create();

    protected:
        RawShaderMaterial();
    };

}// namespace threepp

#endif//THREEPP_RAWSHADERMATERIAL_HPP
