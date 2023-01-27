// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShaderMaterial.js

#ifndef THREEPP_SHADERMATERIAL_HPP
#define THREEPP_SHADERMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include "threepp/core/Uniform.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

namespace threepp {

    class ShaderMaterial : public virtual Material,
                           public MaterialWithClipping,
                           public MaterialWithLights,
                           public MaterialWithWireframe,
                           public MaterialWithLineWidth,
                           public MaterialWithDefines {

    public:
        std::string vertexShader;
        std::string fragmentShader;
        std::shared_ptr<UniformMap> uniforms = std::make_shared<UniformMap>();

        std::optional<std::string> index0AttributeName;
        bool uniformsNeedUpdate = false;

        [[nodiscard]] std::string type() const override {

            return "ShaderMaterial";
        }

        static std::shared_ptr<ShaderMaterial> create() {

            return std::shared_ptr<ShaderMaterial>(new ShaderMaterial());
        }

    protected:
        ShaderMaterial()
            : MaterialWithLights(false),
              MaterialWithClipping(false),
              MaterialWithWireframe(false, 1),
              MaterialWithLineWidth(1),
              vertexShader(shaders::ShaderChunk::instance().default_vertex()),
              fragmentShader(shaders::ShaderChunk::instance().default_fragment()) {

            this->fog = false;
            this->lights = false;
            this->clipping = false;

            defaultAttributeValues["color"] = Color(1, 1, 1);
            defaultAttributeValues["uv"] = Vector2(0, 0);
            defaultAttributeValues["uv2"] = Vector2(0, 0);
        }
    };

}// namespace threepp

#endif//THREEPP_SHADERMATERIAL_HPP
