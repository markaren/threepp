
#include "WgpuBindGroups.hpp"

#include "WgpuShadowMap.hpp"
#include "WgpuTextures.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include <algorithm>
#include <string>

using namespace threepp;
using namespace threepp::wgpu;
namespace SF = threepp::wgpu::ShaderFeatures;

WgpuBindGroups::WgpuBindGroups() {
    entries_.reserve(36);
}

const std::vector<WGPUBindGroupEntry>& WgpuBindGroups::buildStandard(const BindGroupInputs& inputs) {
    entries_.clear();

    uint64_t features = inputs.features;

    // Binding 0: transform uniform buffer
    { WGPUBindGroupEntry e{}; e.binding = 0; e.buffer = inputs.transformBuffer; e.offset = 0; e.size = TRANSFORM_UNIFORM_SIZE; entries_.push_back(e); }
    // Binding 1: material uniform buffer
    { WGPUBindGroupEntry e{}; e.binding = 1; e.buffer = inputs.materialBuffer; e.offset = 0; e.size = MATERIAL_UNIFORM_SIZE; entries_.push_back(e); }

    // Binding 2: light uniform buffer (lit materials only)
    bool lit = SF::isLit(features);
    if (lit) {
        WGPUBindGroupEntry e{}; e.binding = 2; e.buffer = inputs.lightBuffer; e.offset = 0; e.size = inputs.lightUniformSize; entries_.push_back(e);
    }

    // Binding 3-4: diffuse texture + sampler
    bool tex = features & SF::Texture;
    auto* texEntry = &inputs.textures.getDummyTexture();
    if (tex && inputs.params.diffuseMap) {
        texEntry = &inputs.textures.getOrCreateTexture(inputs.params.diffuseMap);
    }
    if (tex) {
        { WGPUBindGroupEntry e{}; e.binding = 3; e.textureView = texEntry->view; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 4; e.sampler = texEntry->sampler; entries_.push_back(e); }
    }

    // Binding 5-6: normal map
    if (features & SF::NormalMap) {
        auto* nmEntry = inputs.params.normalMap ? &inputs.textures.getOrCreateTexture(inputs.params.normalMap) : &inputs.textures.getDummyTexture();
        { WGPUBindGroupEntry e{}; e.binding = 5; e.textureView = nmEntry->view; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 6; e.sampler = nmEntry->sampler; entries_.push_back(e); }
    }

    // Binding 7-9: shadow map (dir/spot); 36-37: point light shadow
    if (features & SF::Shadow) {
        { WGPUBindGroupEntry e{}; e.binding = 7; e.buffer = inputs.shadowMap->uniformBuffer(); e.offset = 0; e.size = inputs.shadowUniformSize; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 8; e.textureView = inputs.shadowMap->depthArrayView(); entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 9; e.sampler = inputs.shadowMap->comparisonSampler(); entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 36; e.buffer = inputs.shadowMap->ptUniformBuffer(); e.offset = 0; e.size = inputs.pointShadowUniformSize; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 37; e.textureView = inputs.shadowMap->ptDepthArrayView(); entries_.push_back(e); }
    }

    // Helper for texture map entries
    auto addTexEntries = [&](uint32_t texBinding, uint32_t sampBinding, Texture* map) {
        auto* te = map ? &inputs.textures.getOrCreateTexture(map) : &inputs.textures.getDummyTexture();
        { WGPUBindGroupEntry e{}; e.binding = texBinding; e.textureView = te->view; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = sampBinding; e.sampler = te->sampler; entries_.push_back(e); }
    };

    if (features & SF::EmissiveMap)     addTexEntries(10, 11, inputs.params.emissiveMap);
    if (features & SF::RoughnessMap)    addTexEntries(12, 13, inputs.params.roughnessMap);
    if (features & SF::MetalnessMap)    addTexEntries(14, 15, inputs.params.metalnessMap);
    if (features & SF::AOMap)           addTexEntries(16, 17, inputs.params.aoMap);
    if (features & SF::AlphaMap)        addTexEntries(18, 19, inputs.params.alphaMap);
    if (features & SF::SpecularMap)     addTexEntries(20, 21, inputs.params.specularMap);
    if (features & SF::LightMap)        addTexEntries(22, 23, inputs.params.lightMap);
    if (features & SF::BumpMap)         addTexEntries(24, 25, inputs.params.bumpMap);
    if (features & SF::GradientMap)     addTexEntries(26, 27, inputs.params.gradientMap);
    if (features & SF::DisplacementMap) addTexEntries(30, 31, inputs.params.displacementMap);

    // Binding 32-33: environment map (equirect 2D or cube)
    if (features & (SF::EnvMap | SF::EnvMapCube)) {
        const TextureEntry* te;
        if (features & SF::EnvMapCube) {
            te = inputs.params.envMap ? &inputs.textures.getOrCreateCubeTexture(inputs.params.envMap) : &inputs.textures.getDummyCubeTexture();
        } else {
            te = inputs.params.envMap ? &inputs.textures.getOrCreateEnvTexture2D(inputs.params.envMap) : &inputs.textures.getDummyTexture();
        }
        { WGPUBindGroupEntry e{}; e.binding = 32; e.textureView = te->view; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 33; e.sampler = te->sampler; entries_.push_back(e); }
    }

    // Binding 28: instance data buffer
    if (inputs.instanceBuffer) {
        WGPUBindGroupEntry e{}; e.binding = 28; e.buffer = inputs.instanceBuffer; e.offset = 0; e.size = inputs.instanceSize;
        entries_.push_back(e);
    }

    // Binding 29: morph target data buffer
    if (inputs.morphBuffer) {
        WGPUBindGroupEntry e{}; e.binding = 29; e.buffer = inputs.morphBuffer; e.offset = 0; e.size = inputs.morphSize;
        entries_.push_back(e);
    }

    // Binding 38-39: transmission sampler map
    if ((features & SF::Transmission) && inputs.transmissionTexView && inputs.transmissionSampler) {
        { WGPUBindGroupEntry e{}; e.binding = 38; e.textureView = inputs.transmissionTexView; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 39; e.sampler = inputs.transmissionSampler; entries_.push_back(e); }
    }

    // Bindings 40-42: RectAreaLight LTC LUTs (shared sampler on ltc1).
    if (features & SF::RectAreaLights) {
        const auto& l1 = inputs.textures.getOrCreateLtc1();
        const auto& l2 = inputs.textures.getOrCreateLtc2();
        { WGPUBindGroupEntry e{}; e.binding = 40; e.textureView = l1.view; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 41; e.textureView = l2.view; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 42; e.sampler = l1.sampler; entries_.push_back(e); }
    }

    // Binding 34-35: skinning data buffers
    if (inputs.skinBuffer) {
        { WGPUBindGroupEntry e{}; e.binding = 34; e.buffer = inputs.skinBuffer; e.offset = 0; e.size = inputs.skinSize;
          entries_.push_back(e); }
    }
    if (inputs.skinVertexBuffer) {
        { WGPUBindGroupEntry e{}; e.binding = 35; e.buffer = inputs.skinVertexBuffer; e.offset = 0; e.size = inputs.skinVertexSize;
          entries_.push_back(e); }
    }

    return entries_;
}

const std::vector<WGPUBindGroupEntry>& WgpuBindGroups::buildCustom(
        WGPUBuffer transformBuffer, WGPUBuffer lightBuffer,
        size_t lightUniformSize,
        WGPUBuffer customUniformBuffer, uint32_t customUniformSize,
        ShaderMaterial* sm,
        const TextureList& textures) {
    entries_.clear();

    // Binding 0: transform
    { WGPUBindGroupEntry e{}; e.binding = 0; e.buffer = transformBuffer; e.offset = 0; e.size = TRANSFORM_UNIFORM_SIZE; entries_.push_back(e); }
    // Binding 1: lights
    { WGPUBindGroupEntry e{}; e.binding = 1; e.buffer = lightBuffer; e.offset = 0; e.size = lightUniformSize; entries_.push_back(e); }

    // Binding 2: custom uniforms (if present)
    if (customUniformBuffer) {
        WGPUBindGroupEntry e{}; e.binding = 2; e.buffer = customUniformBuffer; e.offset = 0; e.size = customUniformSize;
        entries_.push_back(e);
    }

    // GPU texture bindings from unified list (sorted by name for deterministic order)
    uint32_t nextBinding = customUniformBuffer ? 3 : 2;
    for (auto& [name, views] : textures) {
        { WGPUBindGroupEntry e{}; e.binding = nextBinding++; e.textureView = views.first; entries_.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = nextBinding++; e.sampler = views.second; entries_.push_back(e); }
    }

    return entries_;
}
