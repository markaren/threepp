
#ifndef THREEPP_INTERFACES_HPP
#define THREEPP_INTERFACES_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/MaterialParams.hpp"

#include "threepp/textures/Texture.hpp"

#include <array>

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

    // MaterialWithRefractionRatio is a VIRTUAL base here (and in
    // MeshStandardMaterial) so that MeshPhysicalMaterial -- which reaches it
    // both through MeshStandardMaterial and through this mixin -- ends up with
    // exactly one refractionRatio. With two, `material->refractionRatio` was
    // ambiguous, dynamic_cast<MaterialWithRefractionRatio*> returned null (so
    // ObjectLoader/ObjectExporter skipped the field entirely), and the GL
    // renderer read a different copy than the one setValues() wrote.
    //
    // Consequence of the virtual base: every concrete material that inherits
    // this mixin must name MaterialWithRefractionRatio in its own ctor
    // init-list, since it has no default ctor and only the most-derived class
    // initializes a virtual base.
    struct MaterialWithReflectivity: virtual Material, virtual MaterialWithRefractionRatio {

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

    // threepp extension (no three.js equivalent). Terrain splat shading: the
    // mesh's `map` stays the coarse per-tile MACRO colour (splat bands, baked
    // AO/macro variation, painted roads), and this interface adds what resolves
    // the surface at SCREEN density instead of bake density:
    //   • terrainWeightMap  — RGBA, LINEAR, mesh UVs: coverage weight of up to
    //     four structure bands (e.g. grass/rock/scree/snow), baked with the
    //     macro colour. Selects which band texture sets shade each pixel.
    //   • terrainNormalMap  — RGB, LINEAR, mesh UVs: WORLD-space surface normal
    //     (n*0.5+0.5). Replaces the interpolated vertex normal in the G-buffer:
    //     mip selection band-limits it per screen footprint, so tiles baked at
    //     different LOD densities agree at their shared border (per-vertex
    //     normals cannot — their finite-difference epsilon tracks tile
    //     resolution and jumps across a LOD seam).
    //   • band sets — per band, a repeating 0.5-neutral albedo overlay (A =
    //     material HEIGHT for height-based band blending) and a normal +
    //     roughness map (RGB tangent normal, A = roughness modulation), sampled
    //     world-XZ anchored with the detail layer's stochastic tiling +
    //     triplanar machinery, at a per-band repeat rate. Per-band base
    //     roughness REPLACES the material roughness where bands cover.
    // Consumed by the Vulkan deferred renderer's raster G-buffer only; every
    // other renderer/path ignores it (tiles then show the macro `map`).
    struct MaterialWithTerrainMaps: virtual Material {

        static constexpr int kTerrainBands = 4;

        std::shared_ptr<Texture> terrainWeightMap; // RGBA band weights (mesh UVs)
        std::shared_ptr<Texture> terrainNormalMap; // world-space normal (mesh UVs)

        // Per-band repeating texture sets (LINEAR; null band = inert).
        std::array<std::shared_ptr<Texture>, kTerrainBands> terrainBandAlbedo{};     // 0.5-neutral, A = height
        std::array<std::shared_ptr<Texture>, kTerrainBands> terrainBandNormalRough{};// RGB normal, A = rough mod
        std::array<float, kTerrainBands> terrainBandRepeat{0.5f, 0.5f, 0.5f, 0.5f}; // repeats per world metre
        std::array<float, kTerrainBands> terrainBandRoughness{0.9f, 0.9f, 0.9f, 0.9f};// base roughness per band

        float terrainBandStrength = 1.f;    // 0..1 albedo-overlay modulation
        float terrainBandNormalScale = 1.f; // tangent xy perturbation scale
        float terrainBandRoughStrength = 0.6f;// 0..1 roughness modulation strength
        float terrainHeightBlend = 6.f;     // height-blend sharpness (0 = plain linear)

        [[nodiscard]] bool terrainMapsActive() const {
            return terrainWeightMap != nullptr &&
                   (terrainBandAlbedo[0] || terrainBandNormalRough[0]);
        }
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

        // ── Volume IN-SCATTER (the other half of the medium) ─────────────────
        // attenuationColor/attenuationDistance describe what the medium TAKES
        // OUT of a ray (Beer-Lambert absorption). These two describe what it
        // puts BACK IN: single scattering off suspended matter. Glacial and
        // silty water is bright turquoise BECAUSE of rock flour scattering the
        // skylight sideways into the eye, which absorption cannot produce at
        // any parameter value (it can only ever darken).
        //
        //   scatterColor    — single-scattering albedo weight per channel
        //                     (σ_s direction: which wavelengths scatter).
        //   scatterDistance — mean free path in METRES, i.e. 1/σ_s. 0 = OFF.
        //
        // scatterDistance == 0 is the default and every arithmetic path that
        // reads these is gated on it being > 0, so existing materials render
        // bit-identically. Consumed by the Vulkan deferred WATER body only
        // (deferred_shade_50_water_glass.glsl); the GL renderer and the path
        // tracer ignore both fields.
        float scatterDistance = 0;
        Color scatterColor{0, 0, 0};
    };

    struct MaterialWithSheen: virtual Material {

        Color sheenColor{0, 0, 0};
        float sheenRoughness{0.f};
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
