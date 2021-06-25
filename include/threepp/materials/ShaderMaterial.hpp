// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShaderMaterial.js

#ifndef THREEPP_SHADERMATERIAL_HPP
#define THREEPP_SHADERMATERIAL_HPP

#include "threepp/materials/Material.hpp"

#include "threepp/renderers/shaders/ShaderChunk.hpp"

namespace threepp {

    class ShaderMaterial : public MaterialWithWireframe {

    public:
        std::string vertexShader;
        std::string fragmentShader;

        bool fog = false; // set to use scene fog
        bool lights = false; // set to use scene lights
        bool clipping = false; // set to use user-defined clipping planes

        bool getWireframe() const override {
            return wireframe_;
        }

        void setWireframe(bool wireframe) override {
            wireframe_ = wireframe;
        }

        float getWireframeLinewidth() const override {
            return wireframeLinewidth_;
        }

        void setWireframeLinewidth(float width) override {
            wireframeLinewidth_ = width;
        }

        [[nodiscard]] std::string type() const override {
            return "ShaderMaterial";
        }

    protected:
        ShaderMaterial()
            : vertexShader(shaders::ShaderChunk::instance().default_vertex()),
              fragmentShader(shaders::ShaderChunk::instance().default_fragment()) {}

    private:

        float linewidth_ = 1;

        bool wireframe_ = false;
        float wireframeLinewidth_ = 1;


        std::unordered_map<std::string, Uniform> uniforms_;
    };

}// namespace threepp

#endif//THREEPP_SHADERMATERIAL_HPP
