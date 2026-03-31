// Material parameter extraction and uniform packing for the Wgpu renderer.
// Extracts the material->feature-bitmask and material->uniform-data logic
// from the monolithic renderItem() into a reusable subsystem.

#ifndef THREEPP_WGPUMATERIALS_HPP
#define THREEPP_WGPUMATERIALS_HPP

#include "WgpuShaders.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector2.hpp"

#include <cstdint>

namespace threepp {
    class Material;
    class Texture;
    class BufferGeometry;
    class Object3D;
}// namespace threepp

namespace threepp::wgpu {

    // Aggregated material parameters extracted from a Material subclass.
    // Populated by extractMaterialParams() and consumed by packMaterialUniforms()
    // and the bind-group builder.
    struct MaterialParams {
        uint64_t features = ShaderFeatures::None;

        Color diffuse{1, 1, 1};
        float opacity = 1.0f;
        Color specularColor{0, 0, 0};
        float shininess = 30.0f;
        float roughness = 0.5f;
        float metalness = 0.0f;
        Color emissive{0, 0, 0};

        Texture* diffuseMap = nullptr;
        Texture* normalMap = nullptr;
        Texture* emissiveMap = nullptr;
        Texture* roughnessMap = nullptr;
        Texture* metalnessMap = nullptr;
        Texture* aoMap = nullptr;
        Texture* alphaMap = nullptr;
        Texture* specularMap = nullptr;
        Texture* lightMap = nullptr;
        Texture* bumpMap = nullptr;
        Texture* gradientMap = nullptr;
        Texture* displacementMap = nullptr;
        Texture* envMap = nullptr;

        float envMapIntensity = 1.0f;
        float displacementScale = 1.0f;
        Vector2 normalScale{1, 1};
        float aoMapIntensity = 1.0f;
        float bumpScale = 1.0f;

        float transmission = 0.0f;
        float ior = 1.5f;
        float thickness = 0.0f;
        float attenuationDistance = 0.0f;
        Color attenuationColor{1, 1, 1};

        // True if this material is a ShaderMaterial with custom WGSL shaders
        bool isCustomShader = false;
        // True if this material should be skipped entirely (e.g. ShadowMaterial)
        bool skip = false;
    };

    // Per-frame rendering context passed through the render loop.
    struct FrameContext {
        Color fogColor;
        float fogNear = 0;
        float fogFar = 0;
        float fogDensity = 0;
        uint64_t fogBits = 0;
        float toneMappingExposure = 1.0f;
        // Per-object tone mapping/sRGB bits — used when rendering to a render
        // target (where the post-process blit doesn't apply). Zero when
        // rendering to the surface (post-process handles it).
        uint64_t tonemapBits = 0;
        bool srgbOutput = false;
        bool localClippingEnabled = false;
        bool shadowActive = false;
        float transmissionTexW = 0;
        float transmissionTexH = 0;
    };

    // Extract material parameters and compute the base feature bitmask
    // from a Material subclass. Does not include object-level features
    // (instancing, skinning, topology) — those are added by the caller.
    MaterialParams extractMaterialParams(Material* rawMat, BufferGeometry* geometry);

    // Pack material parameters into the GPU uniform buffer layout.
    // The output buffer must be at least MATERIAL_UNIFORM_SIZE bytes.
    void packMaterialUniforms(float* data, const MaterialParams& params,
                              const FrameContext& ctx, Material* rawMat);

}// namespace threepp::wgpu

#endif//THREEPP_WGPUMATERIALS_HPP
