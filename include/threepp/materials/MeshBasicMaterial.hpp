// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshBasicMaterial.js

#ifndef THREEPP_MESHBASICMATERIAL_HPP
#define THREEPP_MESHBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/textures/Texture.hpp"

#include <optional>

namespace threepp {

    class MeshBasicMaterial : public MaterialWithColor, MaterialWithWireframe, MaterialWithReflectivity {

    public:
        std::shared_ptr<Texture> map = nullptr;

        std::shared_ptr<Texture> lightMap = nullptr;
        float lightMapIntensity = 1.0;

        std::shared_ptr<Texture> aoMap = nullptr;
        float aoMapIntensity = 1.0;

        std::shared_ptr<Texture> specularMap = nullptr;

        std::shared_ptr<Texture> alphaMap = nullptr;

        std::shared_ptr<Texture> envMap = nullptr;
        int combine = MultiplyOperation;

        Color &getColor() override {
            return color_;
        }

        [[nodiscard]] float getReflectivity() const override {
            return reflectivity_;
        }
        [[nodiscard]] float getRefractionRatio() const override {
            return refractionRatio_;
        }

        [[nodiscard]] std::string getWireframeLinecap() const override {
            return wireframeLinecap_;
        }

        [[nodiscard]] std::string getWireframeLinejoin() const override {
            return wireframeLinejoin_;
        }

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

        static std::shared_ptr<MeshBasicMaterial> create() {
            return std::shared_ptr<MeshBasicMaterial>(new MeshBasicMaterial());
        }

    protected:
        MeshBasicMaterial() = default;

    private:
        Color color_ = Color(0xffffff);

        float reflectivity_ = 1;
        float refractionRatio_ = 0.98f;

        std::string wireframeLinecap_ = "round";
        std::string wireframeLinejoin_ = "round";

        bool wireframe_ = false;
        float wireframeLinewidth_ = 1;
    };

}// namespace threepp

#endif//THREEPP_MESHBASICMATERIAL_HPP
