// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShaderMaterial.js

#ifndef THREEPP_SHADERMATERIAL_HPP
#define THREEPP_SHADERMATERIAL_HPP

#include "threepp/materials/Material.hpp"

#include "threepp/core/Uniform.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

namespace threepp {

    class ShaderMaterial : public virtual MaterialWithWireframe {

    public:
        std::string vertexShader;
        std::string fragmentShader;

        bool lights = false; // set to use scene lights
        bool clipping = false; // set to use user-defined clipping planes

        [[nodiscard]] bool getWireframe() const override {
            return wireframe_;
        }

        void setWireframe(bool wireframe) override {
            wireframe_ = wireframe;
        }

        [[nodiscard]] float getWireframeLinewidth() const override {
            return wireframeLinewidth_;
        }

        void setWireframeLinewidth(float width) override {
            wireframeLinewidth_ = width;
        }

        [[nodiscard]] std::string type() const override {
            return "ShaderMaterial";
        }

        static std::shared_ptr<ShaderMaterial> create() {

            return std::shared_ptr<ShaderMaterial>(new ShaderMaterial());
        }

    protected:
        ShaderMaterial()
            : vertexShader(shaders::ShaderChunk::instance().default_vertex()),
              fragmentShader(shaders::ShaderChunk::instance().default_fragment()) {

            this->fog = false;

        }

    private:

        float linewidth_ = 1;

        bool wireframe_ = false;
        float wireframeLinewidth_ = 1;

        std::unordered_map<std::string, Uniform> uniforms_;
    };

}// namespace threepp

#endif//THREEPP_SHADERMATERIAL_HPP
