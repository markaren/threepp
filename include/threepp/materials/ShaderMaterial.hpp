// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShaderMaterial.js

#ifndef THREEPP_SHADERMATERIAL_HPP
#define THREEPP_SHADERMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class ShaderMaterial: public virtual Material,
                          public MaterialWithClipping,
                          public MaterialWithLights,
                          public MaterialWithWireframe,
                          public MaterialWithLineWidth,
                          public MaterialWithDefines {

    public:
        std::string vertexShader;
        std::string fragmentShader;
        std::shared_ptr<UniformMap> uniforms;

        std::optional<std::string> index0AttributeName;
        bool uniformsNeedUpdate = false;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<ShaderMaterial> create();

    protected:
        ShaderMaterial();
    };

}// namespace threepp

#endif//THREEPP_SHADERMATERIAL_HPP
