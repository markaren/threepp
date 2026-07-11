
#ifndef THREEPP_INTERFACES_HPP
#define THREEPP_INTERFACES_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/MaterialParams.hpp"

#include "threepp/textures/Texture.hpp"

namespace threepp {

    struct MaterialWithColor: virtual Material {

        Color color;

        explicit MaterialWithColor(Color color): color(color) {}
    };

    struct MaterialWithRotation: virtual Material {

        float rotation{};
    };

    struct MaterialWithClipping: virtual Material {

        bool clipping;

        explicit MaterialWithClipping(bool clipping): clipping(clipping) {}
    };

    struct MaterialWithLights: virtual Material {

        bool lights;

        explicit MaterialWithLights(bool lights): lights(lights) {}
    };

    struct MaterialWithSize: virtual Material {

        float size;
        bool sizeAttenuation;

        MaterialWithSize(float size, bool sizeAttenuation): size(size), sizeAttenuation(sizeAttenuation) {}
    };

    struct MaterialWithLineWidth: virtual Material {

        float linewidth;

        explicit MaterialWithLineWidth(float linewidth): linewidth(linewidth) {}
    };

    struct MaterialWithEmissive: virtual Material {

        Color emissive;
        float emissiveIntensity;
        std::shared_ptr<Texture> emissiveMap;

        MaterialWithEmissive(Color emissive, float emissiveIntensity): emissive(emissive), emissiveIntensity(emissiveIntensity) {}
    };

    struct MaterialWithSpecular: virtual Material {

        Color specular;
        float shininess;

        MaterialWithSpecular(Color specular, float shininess): specular(specular), shininess(shininess) {}
    };

    struct MaterialWithRefractionRatio: virtual Material {

        float refractionRatio;

        explicit MaterialWithRefractionRatio(float refractionRatio): refractionRatio(refractionRatio) {}
    };

    struct MaterialWithReflectivity: virtual Material, MaterialWithRefractionRatio {

        float reflectivity;

        MaterialWithReflectivity(float reflectivity, float refractionRatio): MaterialWithRefractionRatio(refractionRatio), reflectivity(reflectivity) {}
    };

    struct MaterialWithWireframe: virtual Material {

        bool wireframe;
        float wireframeLinewidth;

        MaterialWithWireframe(bool wireframe, float wireframeLinewidth): wireframe(wireframe), wireframeLinewidth(wireframeLinewidth) {}
    };

    struct MaterialWithMap: virtual Material {

        std::shared_ptr<Texture> map;
    };

    struct MaterialWithAlphaMap: virtual Material {

        std::shared_ptr<Texture> alphaMap;
    };

    struct MaterialWithSpecularMap: virtual Material {

        std::shared_ptr<Texture> specularMap;
    };

    struct MaterialWithEnvMap: virtual Material {

        float envMapIntensity;// Only used by MeshStandardMaterial
        std::shared_ptr<Texture> envMap;

        explicit MaterialWithEnvMap(float envMapIntensity = 1.f): envMapIntensity(envMapIntensity) {}
    };

    struct MaterialWithGradientMap: virtual Material {

        std::shared_ptr<Texture> gradientMap;
    };

    struct MaterialWithAoMap: virtual Material {

        std::shared_ptr<Texture> aoMap;
        float aoMapIntensity;

        explicit MaterialWithAoMap(float aoMapIntensity): aoMapIntensity(aoMapIntensity) {}
    };

    struct MaterialWithBumpMap: virtual Material {

        std::shared_ptr<Texture> bumpMap;
        float bumpScale;

        explicit MaterialWithBumpMap(float bumpScale): bumpScale(bumpScale) {}
    };

    struct MaterialWithLightMap: virtual Material {

        std::shared_ptr<Texture> lightMap;
        float lightMapIntensity;

        explicit MaterialWithLightMap(float lightMapIntensity): lightMapIntensity(lightMapIntensity) {}
    };

    struct MaterialWithDisplacementMap: virtual Material {

        std::shared_ptr<Texture> displacementMap;
        float displacementScale;
        float displacementBias;

        MaterialWithDisplacementMap(float displacementScale, float displacementBias): displacementScale(displacementScale), displacementBias(displacementBias) {}
    };

    struct MaterialWithNormalMap: virtual Material {

        std::shared_ptr<Texture> normalMap;
        NormalMapType normalMapType;
        Vector2 normalScale;

        MaterialWithNormalMap(NormalMapType normalMapType, Vector2 normalScale): normalMapType(normalMapType), normalScale(normalScale) {}
    };

    // threepp extension (no three.js equivalent). Tiled detail albedo layered
    // over the base `map` at close range — the game-engine answer to large
    // surfaces (terrain) whose unique-texel macro texture is inevitably
    // coarse per meter. Sampled WORLD-anchored (worldPos.xz * detailRepeat,
    // not mesh UVs) so tiles of any size share one seamless detail field.
    // Texture convention: LINEAR color space, 0.5 = neutral; the shader
    // applies albedo *= mix(1, 2*detail, strength*fade) and fades the layer
    // out once a repeat approaches pixel scale (no distant tiling patterns).
    // Consumed by the Vulkan deferred renderer's raster G-buffer only
    // (secondary rays skip it — a primary-visibility embellishment).
    struct MaterialWithDetailMap: virtual Material {

        std::shared_ptr<Texture> detailMap;
        float detailRepeat;  // repeats per world meter (XZ-anchored)
        float detailStrength;// 0..1 modulation strength

        // Optional detail NORMAL + ROUGHNESS layer, sharing the same world-XZ
        // stochastic projection as detailMap (Vulkan deferred G-buffer only).
        // RGBA, LINEAR: RGB = tangent-space normal (0.5 = flat), A = roughness
        // modulation (0.5 = neutral, shader applies roughness *= mix(1, 2*A,
        // detailRoughStrength*fade)). Both terms fade with distance like the
        // albedo layer, so they never shimmer far away. Inert when null.
        std::shared_ptr<Texture> detailNormalMap;
        float detailNormalScale = 1.f;   // tangent-space xy perturbation scale
        float detailRoughStrength = 0.6f;// 0..1 roughness modulation strength

        MaterialWithDetailMap(float detailRepeat, float detailStrength)
            : detailRepeat(detailRepeat), detailStrength(detailStrength) {}
    };

    // Thin-leaf / foliage two-sided subsurface. Light reaching the BACK of a thin
    // card is partially transmitted to the front, so backlit canopies GLOW rather
    // than going flat-dark. Consumed by the Vulkan deferred renderer only: a
    // raster-primary sun term (wrap back-light + view-dependent forward scatter)
    // plus a small back-normal ambient term. The GL renderer and every ray-hit
    // shading path (probe GI, reflections, lidar) ignore it. translucency == 0 →
    // no effect at all (bit-exact for existing content).
    struct MaterialWithTranslucency: virtual Material {

        float translucency = 0.f;         // 0..1 transmission strength (0 = off)
        Color translucencyColor{1, 1, 1}; // tint applied to the transmitted light
    };

    struct MaterialWithMatCap: virtual Material {

        std::shared_ptr<Texture> matcap;
    };

    struct MaterialWithRoughness: virtual Material {

        float roughness;
        std::shared_ptr<Texture> roughnessMap;

        explicit MaterialWithRoughness(float roughness): roughness(roughness) {}
    };

    struct MaterialWithMetalness: virtual Material {

        float metalness;
        std::shared_ptr<Texture> metalnessMap;

        explicit MaterialWithMetalness(float metalness): metalness(metalness) {}
    };

    struct MaterialWithThickness: virtual Material {

        float thickness = 0;
        std::shared_ptr<Texture> thicknessMap;
        // Path-tracer hint: this surface is a thin shell (e.g. a single
        // FFT-displaced ocean plane, sunglasses lens, leaf), not a closed
        // volume. Default false → closed-mesh BSDF (front=enter, back=exit,
        // Beer-Lambert applied at the back-face exit using actual ray length).
        // True → both faces refract as entries and Beer-Lambert is applied
        // per-crossing using `thickness` as the in-medium proxy distance.
        // Closed glass meshes that happen to also be doubleSided should leave
        // this false; turning it on makes the back-face refraction direction
        // wrong for them.
        bool thinWalled = false;
    };

    struct MaterialWithClearcoat: virtual Material {

        float clearcoat = 0;
        std::shared_ptr<Texture> clearcoatMap;
        float clearcoatRoughness = 0;
        std::shared_ptr<Texture> clearcoatRoughnessMap;
        Vector2 clearcoatNormalScale{1, 1};
        std::shared_ptr<Texture> clearcoatNormalMap;
    };

    struct MaterialWithTransmission: virtual Material {

        float transmission = 0;
        float ior = 1.5f;
        float dispersion = 0;
        std::shared_ptr<Texture> transmissionMap;
    };

    struct MaterialWithAttenuation: virtual Material {

        float attenuationDistance = 0;
        Color attenuationColor{1, 1, 1};
    };

    struct MaterialWithSheen: virtual Material {

        Color sheenColor{0, 0, 0};
        float sheenRoughness{0.f};
        std::optional<Color> sheen;  // legacy
    };

    struct MaterialWithIridescence: virtual Material {

        float iridescence = 0.f;            // 0..1 layer intensity
        float iridescenceIOR = 1.3f;        // thin-film IOR (1.0..2.5 typical)
        float iridescenceThicknessNm = 400.f;// thin-film thickness in nanometers
    };

    struct MaterialWithPbrSpecular: virtual Material {

        float specularIntensity{1.f};
        Color specularColor{1, 1, 1};
    };

    struct MaterialWithCombine: virtual Material {

        CombineOperation combine;

        explicit MaterialWithCombine(CombineOperation combine): combine(combine) {}
    };

    struct MaterialWithDepthPacking: virtual Material {

        DepthPacking depthPacking;

        explicit MaterialWithDepthPacking(DepthPacking depthPacking): depthPacking(depthPacking) {}
    };

    struct MaterialWithFlatShading: virtual Material {

        bool flatShading;

        explicit MaterialWithFlatShading(bool flatShading): flatShading(flatShading) {}
    };

    struct MaterialWithVertexTangents: virtual Material {

        bool vertexTangents;

        explicit MaterialWithVertexTangents(bool vertexTangents): vertexTangents(vertexTangents) {}
    };

    struct MaterialWithDefines: virtual Material {

        std::unordered_map<std::string, std::string> defines;
    };

    struct MaterialWithMorphTargets: virtual Material {

        bool morphTargets = false;
        bool morphNormals = false;
    };

}// namespace threepp


#endif//THREEPP_INTERFACES_HPP
