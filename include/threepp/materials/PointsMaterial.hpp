// https://github.com/mrdoob/three.js/blob/r129/src/materials/PointsMaterial.js

#ifndef THREEPP_POINTSMATERIAL_HPP
#define THREEPP_POINTSMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class PointsMaterial : public MaterialWithColor, MaterialWithSize {

    public:
        [[nodiscard]] float getSize() const override {

            return size_;
        }

        [[nodiscard]] bool getSizeAttenuation() const override {

            return sizeAttenuation_;
        }

        Color &getColor() override {

            return color_;
        }

        [[nodiscard]] std::string type() const override {

            return "PointsMaterial";
        }

        static std::shared_ptr<PointsMaterial> create() {

            return std::shared_ptr<PointsMaterial>(new PointsMaterial());
        }

    protected:
        PointsMaterial() = default;

    private:
        Color color_ = Color(0xffffff);

        std::optional<Texture> map_ = std::nullopt;

        std::optional<Texture> alphaMap_ = std::nullopt;

        float size_ = 1;
        bool sizeAttenuation_ = true;
    };

}// namespace threepp

#endif//THREEPP_POINTSMATERIAL_HPP
