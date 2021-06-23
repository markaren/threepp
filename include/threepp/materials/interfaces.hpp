
#ifndef THREEPP_INTERFACES_HPP
#define THREEPP_INTERFACES_HPP

#include "threepp/materials/Material.hpp"

#include "threepp/textures/Texture.hpp"

namespace threepp {

    struct MaterialWithColor : virtual Material {

        virtual Color &getColor() = 0;

        ~MaterialWithColor() override = default;
    };

    struct MaterialWithClipping : virtual Material {

        virtual bool getClipping() = 0;

        ~MaterialWithClipping() override = default;
    };

    struct MaterialWithSize : virtual Material {

        virtual float getSize() const = 0;
        virtual bool getSizeAttenuation() const = 0;

        ~MaterialWithSize() override = default;
    };

    struct MaterialWithEmissive : virtual Material {

        virtual Color &getEmissiveColor() = 0;

        virtual float getEmissiveIntensity() const = 0;

        ~MaterialWithEmissive() override = default;
    };

    struct MaterialWithSpecular : virtual Material {

        virtual Color &getSpecularColor() = 0;

        virtual float getShininess() const = 0;

        ~MaterialWithSpecular() override = default;
    };

    struct MaterialWithReflectivity : virtual Material {

        virtual float getReflectivity() const = 0;
        virtual float getRefractionRatio() const = 0;

        ~MaterialWithReflectivity() override = default;
    };

    struct MaterialWithWireframe : virtual Material {

        virtual std::string getWireframeLinecap() const = 0;
        virtual std::string getWireframeLinejoin() const = 0;

        [[nodiscard]] virtual bool getWireframe() const = 0;
        virtual void setWireframe(bool wireframe) = 0;

        [[nodiscard]] virtual float getWireframeLinewidth() const = 0;
        virtual void setWireframeLinewidth(float width) = 0;

        ~MaterialWithWireframe() override = default;
    };

    struct MaterialWithMap : virtual Material {

        virtual std::optional<Texture> &getMap() = 0;

        ~MaterialWithMap() override = default;
    };

    struct MaterialWithAlphaMap : virtual Material {

        virtual std::optional<Texture> &getAlphaMap() = 0;

        ~MaterialWithAlphaMap() override = default;
    };

}


#endif//THREEPP_INTERFACES_HPP
