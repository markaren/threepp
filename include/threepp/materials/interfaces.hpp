
#ifndef THREEPP_INTERFACES_HPP
#define THREEPP_INTERFACES_HPP

#include "threepp/materials/Material.hpp"

#include "threepp/textures/Texture.hpp"

namespace threepp {

    struct MaterialWithColor : virtual Material {

        Color color;

        template<class T>
        explicit MaterialWithColor(T color) : color(color) {}
    };

    struct MaterialWithClipping : virtual Material {

        bool clipping;

        explicit MaterialWithClipping(bool clipping) : clipping(clipping) {}
    };

    struct MaterialWithLights : virtual Material {

        bool lights;

        explicit MaterialWithLights(bool lights) : lights(lights) {}
    };

    struct MaterialWithSize : virtual Material {

        float size;
        bool sizeAttenuation;

        MaterialWithSize(float size, bool sizeAttenuation) : size(size), sizeAttenuation(sizeAttenuation) {}
    };

    struct MaterialWithLineWidth : virtual Material {

        float lineWidth;

        explicit MaterialWithLineWidth(float lineWidth) : lineWidth(lineWidth) {}
    };

    struct MaterialWithEmissive : virtual Material {

        Color emissiveColor;
        float emissiveIntensity;

        template<class T>
        MaterialWithEmissive(T emissiveColor, float emissiveIntensity) : emissiveColor(emissiveColor), emissiveIntensity(emissiveIntensity) {}
    };

    struct MaterialWithSpecular : virtual Material {

        Color specularColor;
        float getShininess;

        template<class T>
        MaterialWithSpecular(T specularColor, float getShininess) : specularColor(specularColor), getShininess(getShininess) {}
    };

    struct MaterialWithReflectivity : virtual Material {

        float reflectivity;
        float refractionRatio;

        MaterialWithReflectivity(float reflectivity, float refractionRatio) : reflectivity(reflectivity), refractionRatio(refractionRatio) {}
    };

    struct MaterialWithWireframe : virtual Material {

        bool wireframe;
        float wireframeLinewidth;

        MaterialWithWireframe(bool wireframe, float wireframeLinewidth) : wireframe(wireframe), wireframeLinewidth(wireframeLinewidth) {}
    };

    struct MaterialWithMap : virtual Material {

        std::optional<Texture> map;
    };

    struct MaterialWithAlphaMap : virtual Material {

        std::optional<Texture> alphaMap;
    };

    struct MaterialWithSpecularMap : virtual Material {

        std::optional<Texture> specularMap;
    };

    struct MaterialWithEnvMap : virtual Material {

        std::optional<Texture> envMap;
    };

    struct MaterialWithAoMap : virtual Material {

        std::optional<Texture> aoMap;
        float aoMapIntensity;

        explicit MaterialWithAoMap(float aoMapIntensity) : aoMapIntensity(aoMapIntensity) {}
    };

    struct MaterialWithLightMap : virtual Material {

        std::optional<Texture> lightMap;
        float lightMapIntensity;

        explicit MaterialWithLightMap(float lightMapIntensity) : lightMapIntensity(lightMapIntensity) {}
    };

}// namespace threepp


#endif//THREEPP_INTERFACES_HPP
