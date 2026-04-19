// WGSL shader generation and feature flags for the Wgpu renderer backend.
// Extracted from WgpuRenderer.cpp to enable independent testing and iteration.

#ifndef THREEPP_WGPUSHADERS_HPP
#define THREEPP_WGPUSHADERS_HPP

#include <cstdint>
#include <string>

namespace threepp::wgpu {

    // Feature bitmask for pipeline caching and shader variant selection.
    // Each bit or bit-group controls a specific shader feature. Using a
    // structured namespace instead of bare constexprs to prevent collisions
    // and improve discoverability.
    namespace ShaderFeatures {

        constexpr uint64_t None = 0;

        // Material lighting model (bits 0-3)
        constexpr uint64_t Texture         = 1ULL << 0;
        constexpr uint64_t Lighting        = 1ULL << 1;
        constexpr uint64_t Specular        = 1ULL << 2;
        constexpr uint64_t PBR             = 1ULL << 3;

        // Cull mode (bits 4-5)
        constexpr uint64_t CullShift       = 4;
        constexpr uint64_t CullMask        = 0x3ULL << CullShift;
        constexpr uint64_t CullNone        = 0ULL << CullShift;
        constexpr uint64_t CullFront       = 1ULL << CullShift;
        constexpr uint64_t CullBack        = 2ULL << CullShift;

        // Wireframe (bit 6)
        constexpr uint64_t Wireframe       = 1ULL << 6;

        // Blend mode (bits 7-9)
        constexpr uint64_t BlendShift      = 7;
        constexpr uint64_t BlendMask       = 0x7ULL << BlendShift;
        constexpr uint64_t BlendNormal     = 0ULL << BlendShift;
        constexpr uint64_t BlendDisabled   = 1ULL << BlendShift;
        constexpr uint64_t BlendAdditive   = 2ULL << BlendShift;
        constexpr uint64_t BlendSubtractive = 3ULL << BlendShift;
        constexpr uint64_t BlendMultiply   = 4ULL << BlendShift;

        // Additional feature bits
        constexpr uint64_t NormalMap       = 1ULL << 10;
        constexpr uint64_t DepthWriteOff   = 1ULL << 11;
        constexpr uint64_t Shadow          = 1ULL << 12;
        constexpr uint64_t FogLinear       = 1ULL << 13;
        constexpr uint64_t FogExp2         = 1ULL << 14;
        constexpr uint64_t InstanceColor   = 1ULL << 15;
        constexpr uint64_t DisplacementMap = 1ULL << 16;
        constexpr uint64_t MorphTargets    = 1ULL << 17;

        // Topology mode (bits 18-19)
        constexpr uint64_t TopoShift       = 18;
        constexpr uint64_t TopoMask        = 0x3ULL << TopoShift;
        constexpr uint64_t TopoTriangle    = 0ULL << TopoShift;
        constexpr uint64_t TopoLineList    = 1ULL << TopoShift;
        constexpr uint64_t TopoLineStrip   = 2ULL << TopoShift;
        constexpr uint64_t TopoPointList   = 3ULL << TopoShift;

        constexpr uint64_t Instanced       = 1ULL << 20;
        constexpr uint64_t VertexColors    = 1ULL << 21;
        constexpr uint64_t EmissiveMap     = 1ULL << 22;
        constexpr uint64_t RoughnessMap    = 1ULL << 23;
        constexpr uint64_t MetalnessMap    = 1ULL << 24;
        constexpr uint64_t AOMap           = 1ULL << 25;
        constexpr uint64_t AlphaMap        = 1ULL << 26;
        constexpr uint64_t SpecularMap     = 1ULL << 27;
        constexpr uint64_t LightMap        = 1ULL << 28;
        constexpr uint64_t SRGBOutput      = 1ULL << 29;
        constexpr uint64_t BumpMap         = 1ULL << 30;
        constexpr uint64_t GradientMap     = 1ULL << 31;

        // Tone mapping mode (bits 32-34)
        constexpr uint64_t TonemapShift    = 32;
        constexpr uint64_t TonemapMask     = 0x7ULL << TonemapShift;
        constexpr uint64_t TonemapNone     = 0ULL << TonemapShift;
        constexpr uint64_t TonemapLinear   = 1ULL << TonemapShift;
        constexpr uint64_t TonemapReinhard = 2ULL << TonemapShift;
        constexpr uint64_t TonemapCineon   = 3ULL << TonemapShift;
        constexpr uint64_t TonemapACES     = 4ULL << TonemapShift;

        // Environment map (bit 35)
        constexpr uint64_t EnvMap          = 1ULL << 35;

        // SkinnedMesh (bit 36)
        constexpr uint64_t Skinning        = 1ULL << 36;

        // ShadowMaterial (bit 37): renders only shadow attenuation on a transparent surface
        constexpr uint64_t ShadowMat       = 1ULL << 37;

        // LineDashed (bit 38): dashed line fragment discard pattern
        constexpr uint64_t LineDashed      = 1ULL << 38;

        // MeshNormalMaterial (bit 39): maps world-space normals to RGB color
        constexpr uint64_t NormalVis       = 1ULL << 39;

        // MeshDepthMaterial (bit 40): maps depth (near→far) to brightness
        constexpr uint64_t DepthVis        = 1ULL << 40;

        // Transmission (bit 41): screen-space refraction for transmissive materials
        constexpr uint64_t Transmission    = 1ULL << 41;

        // DepthTestOff (bit 42): material has depthTest=false → WGPUCompareFunction_Always
        constexpr uint64_t DepthTestOff    = 1ULL << 42;

        // EnvMapCube (bit 43): environment map is a cube texture (CubeReflection/CubeRefraction mapping)
        constexpr uint64_t EnvMapCube      = 1ULL << 43;

        // Convenience: test if shader needs lighting calculations.
        inline bool isLit(uint64_t features) {
            return (features & (Lighting | Specular | PBR)) != 0;
        }

        // ShadowMaterial needs shadow sampling but not full lighting.
        inline bool needsShadowSampling(uint64_t features) {
            return (features & ShadowMat) != 0 && (features & Shadow) != 0;
        }

        // Returns a human-readable string listing active features (for debugging).
        std::string describe(uint64_t features);

    }// namespace ShaderFeatures

    // Uniform buffer sizes in bytes (matching WGSL struct layouts with std140 padding).
    // Transform: model(64) + view(64) + proj(64) + normalMatrix(48) + cameraPos(12) + pad(4) = 256
    constexpr size_t TRANSFORM_UNIFORM_SIZE = 256;
    // Material: diffuse(16) + specular(16) + roughnessMetalnessOpacity(16) + emissive(16)
    //         + flags(16) + fogColor(16) + fogParams(16) + clipPlane(16)
    //         + transmissionParams(16) + attenuationParams(16) = 160
    constexpr size_t MATERIAL_UNIFORM_SIZE = 160;

    struct LightLimits;
    struct ShadowLimits;

    // Generate the main WGSL shader source for the given feature bitmask and light/shadow limits.
    std::string buildWGSL(uint64_t features, const LightLimits& limits, const ShadowLimits& shadowLimits);

    // Generate the depth-only WGSL shader for shadow passes.
    std::string buildDepthWGSL();

    // Depth shader variant for skinned (skeletal) meshes.
    std::string buildSkinnedDepthWGSL();

    // Depth shader variant for morph-target meshes.
    std::string buildMorphDepthWGSL();

}// namespace threepp::wgpu

#endif//THREEPP_WGPUSHADERS_HPP
