// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShaderMaterial.js

#ifndef THREEPP_SHADERMATERIAL_HPP
#define THREEPP_SHADERMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include "threepp/core/Uniform.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

namespace threepp {

    class ShaderMaterial : public virtual Material,
                           MaterialWithClipping,
                           MaterialWithLights,
                           MaterialWithWireframe,
                           MaterialWithLineWidth {

    public:
        std::string vertexShader;
        std::string fragmentShader;

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
        }

    private:

        std::unordered_map<std::string, Uniform> uniforms_;
    };

}// namespace threepp

#endif//THREEPP_SHADERMATERIAL_HPP
