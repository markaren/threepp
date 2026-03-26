#include "WgpuShadowMap.hpp"
#include "WgpuGeometries.hpp"
#include "WgpuShaders.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/lights/PointLightShadow.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

using namespace threepp;
using namespace threepp::wgpu;

namespace {

    constexpr int kMaxShadowMeshes = 256;
    constexpr uint32_t kDynStride  = 256; // minUniformBufferOffsetAlignment

    void collectShadowMeshes(Object3D& obj, std::vector<Mesh*>& out) {
        if (auto mesh = obj.as<Mesh>()) {
            auto geom = mesh->geometry();
            if (mesh->castShadow && geom && geom->hasAttribute("position"))
                out.push_back(mesh);
        }
        for (auto& child : obj.children) collectShadowMeshes(*child, out);
    }

    bool meshIsSkinned(const Mesh* mesh) {
        if (!mesh->is<SkinnedMesh>()) return false;
        auto* sm = dynamic_cast<const SkinnedMesh*>(mesh);
        auto geom = sm->geometry();
        return sm->skeleton && geom &&
               geom->hasAttribute("skinIndex") && geom->hasAttribute("skinWeight");
    }

    bool meshHasMorphTargets(Mesh* mesh) {
        auto geom = mesh->geometry();
        if (!geom || !geom->getMorphAttributes().contains("position")) return false;
        return !mesh->morphTargetInfluences().empty();
    }

    // Helper to create a depth-only render pipeline with a given shader module and BGL.
    WGPURenderPipeline makeDepthPipeline(WGPUDevice device,
                                          WGPUPipelineLayout layout,
                                          WGPUShaderModule shader,
                                          bool needVertexIndex) {
        WGPUVertexAttribute attrs[4]{};
        attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0;  attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
        attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;
        attrs[3].format = WGPUVertexFormat_Float32x3; attrs[3].offset = 32; attrs[3].shaderLocation = 3;

        WGPUVertexBufferLayout vbLayout{};
        vbLayout.arrayStride = VERTEX_STRIDE;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = needVertexIndex ? 1 : 4; // skinned/morph only need position
        vbLayout.attributes = attrs;

        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth32Float;
        depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
        depthStencil.depthCompare = WGPUCompareFunction_Less;

        WGPURenderPipelineDescriptor pd{};
        pd.layout = layout;
        pd.vertex.module     = shader;
        pd.vertex.entryPoint = WGPUStringView{"vs_main", WGPU_STRLEN};
        pd.vertex.bufferCount = 1;
        pd.vertex.buffers    = &vbLayout;
        pd.primitive.topology  = WGPUPrimitiveTopology_TriangleList;
        pd.primitive.frontFace = WGPUFrontFace_CCW;
        pd.primitive.cullMode  = WGPUCullMode_Front;
        pd.depthStencil        = &depthStencil;
        pd.multisample.count   = 1;
        pd.multisample.mask    = 0xFFFFFFFF;
        pd.fragment            = nullptr;
        return wgpuDeviceCreateRenderPipeline(device, &pd);
    }

}// namespace

WgpuShadowMap::WgpuShadowMap(WgpuState& state, WgpuGeometries& geometries)
    : state_(state), geometries_(geometries) {}

void WgpuShadowMap::init() {
    if (initialized_) return;
    initialized_ = true;

    const auto& sl = state_.shadowLimits;
    const auto mapSize = sl.mapSize;
    const auto maxLights = sl.maxShadowLights;
    const auto maxPointLights = sl.maxShadowPointLights;

    lights_.resize(maxLights);
    ptLayerViews_.resize(maxPointLights * 6, nullptr);

    // 2D array depth texture with one layer per shadow-casting light
    {
        WGPUTextureDescriptor td{};
        td.label = WGPUStringView{"shadow_depth_array", WGPU_STRLEN} ;
        td.size = {mapSize, mapSize, static_cast<uint32_t>(maxLights)};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = WGPUTextureFormat_Depth32Float;
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
        depthArrayTexture_ = wgpuDeviceCreateTexture(state_.device, &td);

        // Full array view for sampling in the fragment shader
        WGPUTextureViewDescriptor avd{};
        avd.label = WGPUStringView{"shadow_array_view", WGPU_STRLEN} ;
        avd.format = WGPUTextureFormat_Depth32Float;
        avd.dimension = WGPUTextureViewDimension_2DArray;
        avd.baseMipLevel = 0;
        avd.mipLevelCount = 1;
        avd.baseArrayLayer = 0;
        avd.arrayLayerCount = static_cast<uint32_t>(maxLights);
        avd.aspect = WGPUTextureAspect_DepthOnly;
        depthArrayView_ = wgpuTextureCreateView(depthArrayTexture_, &avd);

        // Per-layer views for rendering
        for (int i = 0; i < maxLights; i++) {
            WGPUTextureViewDescriptor lvd{};
            lvd.label = WGPUStringView{"shadow_layer_view", WGPU_STRLEN} ;
            lvd.format = WGPUTextureFormat_Depth32Float;
            lvd.dimension = WGPUTextureViewDimension_2D;
            lvd.baseMipLevel = 0;
            lvd.mipLevelCount = 1;
            lvd.baseArrayLayer = static_cast<uint32_t>(i);
            lvd.arrayLayerCount = 1;
            lvd.aspect = WGPUTextureAspect_DepthOnly;
            lights_[i].layerView = wgpuTextureCreateView(depthArrayTexture_, &lvd);
        }
    }

    // Comparison sampler
    WGPUSamplerDescriptor sd{};
    sd.label = WGPUStringView{"shadow_samp", WGPU_STRLEN} ;
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.compare = WGPUCompareFunction_Less;
    sd.maxAnisotropy = 1;
    comparisonSampler_ = wgpuDeviceCreateSampler(state_.device, &sd);

    // Dir/Spot shadow uniform buffer
    {
        WGPUBufferDescriptor bd{};
        bd.label = WGPUStringView{"shadow_ub", WGPU_STRLEN} ;
        bd.size = sl.shadowUniformSize();
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        uniformBuffer_ = wgpuDeviceCreateBuffer(state_.device, &bd);
    }

    // Point light shadow: 2D array texture (6 layers per light)
    {
        WGPUTextureDescriptor td{};
        td.label = WGPUStringView{"pt_shadow_depth_array", WGPU_STRLEN} ;
        td.size = {mapSize, mapSize, static_cast<uint32_t>(maxPointLights * 6)};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = WGPUTextureFormat_Depth32Float;
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
        ptDepthArrayTexture_ = wgpuDeviceCreateTexture(state_.device, &td);

        WGPUTextureViewDescriptor avd{};
        avd.label = WGPUStringView{"pt_shadow_array_view", WGPU_STRLEN} ;
        avd.format = WGPUTextureFormat_Depth32Float;
        avd.dimension = WGPUTextureViewDimension_2DArray;
        avd.baseMipLevel = 0; avd.mipLevelCount = 1;
        avd.baseArrayLayer = 0;
        avd.arrayLayerCount = static_cast<uint32_t>(maxPointLights * 6);
        avd.aspect = WGPUTextureAspect_DepthOnly;
        ptDepthArrayView_ = wgpuTextureCreateView(ptDepthArrayTexture_, &avd);

        for (int i = 0; i < maxPointLights * 6; i++) {
            WGPUTextureViewDescriptor lvd{};
            lvd.label = WGPUStringView{"pt_shadow_layer", WGPU_STRLEN} ;
            lvd.format = WGPUTextureFormat_Depth32Float;
            lvd.dimension = WGPUTextureViewDimension_2D;
            lvd.baseMipLevel = 0; lvd.mipLevelCount = 1;
            lvd.baseArrayLayer = static_cast<uint32_t>(i);
            lvd.arrayLayerCount = 1;
            lvd.aspect = WGPUTextureAspect_DepthOnly;
            ptLayerViews_[i] = wgpuTextureCreateView(ptDepthArrayTexture_, &lvd);
        }
    }

    // Point light shadow uniform buffer
    {
        WGPUBufferDescriptor bd{};
        bd.label = WGPUStringView{"pt_shadow_ub", WGPU_STRLEN} ;
        bd.size = sl.pointShadowUniformSize();
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        ptUniformBuffer_ = wgpuDeviceCreateBuffer(state_.device, &bd);
    }

    // Depth-only transform buffer (kMaxShadowMeshes × 256-byte aligned slots)
    {
        WGPUBufferDescriptor bd{};
        bd.label = WGPUStringView{"shadow_xform", WGPU_STRLEN} ;
        bd.size = static_cast<uint64_t>(kMaxShadowMeshes) * kDynStride;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        depthTransformBuffer_ = wgpuDeviceCreateBuffer(state_.device, &bd);
    }

    // Depth-only shader module
    std::string depthWGSL = buildDepthWGSL();

    WGPUShaderSourceWGSL wgslSource{};
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = {.data = depthWGSL.c_str(), .length = static_cast<size_t>(depthWGSL.size())};

    WGPUShaderModuleDescriptor smd{};
    smd.nextInChain = &wgslSource.chain;
    smd.label = WGPUStringView{"shadow_shader", WGPU_STRLEN} ;
    depthShader_ = wgpuDeviceCreateShaderModule(state_.device, &smd);

    // Bind group layout: one dynamic-offset uniform buffer at binding 0
    WGPUBindGroupLayoutEntry bglEntry{};
    bglEntry.binding = 0;
    bglEntry.visibility = WGPUShaderStage_Vertex;
    bglEntry.buffer.type = WGPUBufferBindingType_Uniform;
    bglEntry.buffer.hasDynamicOffset = true;
    bglEntry.buffer.minBindingSize = 64;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = WGPUStringView{"shadow_bgl", WGPU_STRLEN} ;
    bglDesc.entryCount = 1;
    bglDesc.entries = &bglEntry;
    depthBindGroupLayout_ = wgpuDeviceCreateBindGroupLayout(state_.device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.label = WGPUStringView{"shadow_pl", WGPU_STRLEN} ;
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &depthBindGroupLayout_;
    depthPipelineLayout_ = wgpuDeviceCreatePipelineLayout(state_.device, &plDesc);

    // Persistent bind group (dynamic offset per draw)
    {
        WGPUBindGroupEntry bgEntry{};
        bgEntry.binding = 0;
        bgEntry.buffer = depthTransformBuffer_;
        bgEntry.offset = 0;
        bgEntry.size = 64;

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.label = WGPUStringView{"shadow_bg", WGPU_STRLEN} ;
        bgDesc.layout = depthBindGroupLayout_;
        bgDesc.entryCount = 1;
        bgDesc.entries = &bgEntry;
        depthBindGroup_ = wgpuDeviceCreateBindGroup(state_.device, &bgDesc);
    }

    // Vertex layout (same interleaved format as the main pipeline)
    WGPUVertexAttribute attrs[4]{};
    attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0;  attrs[0].shaderLocation = 0;
    attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
    attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;
    attrs[3].format = WGPUVertexFormat_Float32x3; attrs[3].offset = 32; attrs[3].shaderLocation = 3;

    WGPUVertexBufferLayout vbLayout{};
    vbLayout.arrayStride = VERTEX_STRIDE;
    vbLayout.stepMode = WGPUVertexStepMode_Vertex;
    vbLayout.attributeCount = 4;
    vbLayout.attributes = attrs;

    WGPUDepthStencilState depthStencil{};
    depthStencil.format = WGPUTextureFormat_Depth32Float;
    depthStencil.depthWriteEnabled = WGPUOptionalBool_True;
    depthStencil.depthCompare = WGPUCompareFunction_Less;

    WGPURenderPipelineDescriptor pipeDesc{};
    pipeDesc.label = WGPUStringView{"shadow_pipe", WGPU_STRLEN} ;
    pipeDesc.layout = depthPipelineLayout_;

    auto vsEntry = WGPUStringView{"vs_main", WGPU_STRLEN} ;
    pipeDesc.vertex.module = depthShader_;
    pipeDesc.vertex.entryPoint = vsEntry;
    pipeDesc.vertex.bufferCount = 1;
    pipeDesc.vertex.buffers = &vbLayout;

    pipeDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipeDesc.primitive.frontFace = WGPUFrontFace_CCW;
    pipeDesc.primitive.cullMode = WGPUCullMode_Front; // back-face rendering reduces shadow acne
    pipeDesc.depthStencil = &depthStencil;
    pipeDesc.multisample.count = 1;
    pipeDesc.multisample.mask = 0xFFFFFFFF;
    pipeDesc.fragment = nullptr;

    depthPipeline_ = wgpuDeviceCreateRenderPipeline(state_.device, &pipeDesc);

    // --- Skinned depth pipeline ---
    {
        std::string wgsl = buildSkinnedDepthWGSL();
        WGPUShaderSourceWGSL src{};
        src.chain.sType = WGPUSType_ShaderSourceWGSL;
        src.code = {.data = wgsl.c_str(), .length = wgsl.size()};
        WGPUShaderModuleDescriptor smd{}; smd.nextInChain = &src.chain;
        skinnedDepthShader_ = wgpuDeviceCreateShaderModule(state_.device, &smd);

        WGPUBindGroupLayoutEntry entries[3]{};
        entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Vertex;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.hasDynamicOffset = true; entries[0].buffer.minBindingSize = 64;
        entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Vertex;
        entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Vertex;
        entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

        WGPUBindGroupLayoutDescriptor bgld{}; bgld.entryCount = 3; bgld.entries = entries;
        skinnedDepthBGL_ = wgpuDeviceCreateBindGroupLayout(state_.device, &bgld);

        WGPUPipelineLayoutDescriptor pld{}; pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &skinnedDepthBGL_;
        skinnedDepthPipelineLayout_ = wgpuDeviceCreatePipelineLayout(state_.device, &pld);

        skinnedDepthPipeline_ = makeDepthPipeline(state_.device, skinnedDepthPipelineLayout_, skinnedDepthShader_, true);
    }

    // --- Morph-target depth pipeline ---
    {
        std::string wgsl = buildMorphDepthWGSL();
        WGPUShaderSourceWGSL src{};
        src.chain.sType = WGPUSType_ShaderSourceWGSL;
        src.code = {.data = wgsl.c_str(), .length = wgsl.size()};
        WGPUShaderModuleDescriptor smd{}; smd.nextInChain = &src.chain;
        morphDepthShader_ = wgpuDeviceCreateShaderModule(state_.device, &smd);

        WGPUBindGroupLayoutEntry entries[2]{};
        entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Vertex;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.hasDynamicOffset = true; entries[0].buffer.minBindingSize = 64;
        entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Vertex;
        entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

        WGPUBindGroupLayoutDescriptor bgld{}; bgld.entryCount = 2; bgld.entries = entries;
        morphDepthBGL_ = wgpuDeviceCreateBindGroupLayout(state_.device, &bgld);

        WGPUPipelineLayoutDescriptor pld{}; pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &morphDepthBGL_;
        morphDepthPipelineLayout_ = wgpuDeviceCreatePipelineLayout(state_.device, &pld);

        morphDepthPipeline_ = makeDepthPipeline(state_.device, morphDepthPipelineLayout_, morphDepthShader_, true);
    }
}

void WgpuShadowMap::beginFrame(Object3D& scene) {
    if (!autoUpdate && !needsUpdate) return;

    active_ = false;
    activeLightCount_ = 0;
    numPointShadows_ = 0;

    const auto& sl = state_.shadowLimits;
    const auto mapSize = static_cast<float>(sl.mapSize);

    // Collect shadow-casting DirectionalLight and SpotLight instances
    struct ShadowCaster {
        Light* light;
        LightWithShadow* shadowInterface;
    };
    std::vector<ShadowCaster> shadowLights;

    // Collect shadow-casting PointLights
    std::vector<PointLight*> pointShadowLights;

    std::function<void(Object3D&)> findShadowLights = [&](Object3D& obj) {
        if (obj.castShadow) {
            if (auto dl = obj.as<DirectionalLight>()) {
                if (static_cast<int>(shadowLights.size()) < sl.maxShadowLights)
                    shadowLights.push_back({dl, dl});
            } else if (auto sl2 = obj.as<SpotLight>()) {
                if (static_cast<int>(shadowLights.size()) < sl.maxShadowLights)
                    shadowLights.push_back({sl2, sl2});
            } else if (auto pl = obj.as<PointLight>()) {
                if (static_cast<int>(pointShadowLights.size()) < sl.maxShadowPointLights)
                    pointShadowLights.push_back(pl);
            }
        }
        for (auto& child : obj.children) findShadowLights(*child);
    };
    findShadowLights(scene);

    if (shadowLights.empty() && pointShadowLights.empty()) return;

    init();
    active_ = true;
    activeLightCount_ = static_cast<int>(shadowLights.size());

    // Sort shadow lights: DirectionalLights first, then SpotLights
    // (matches the order in WgpuLights after stable_partition).
    std::stable_partition(shadowLights.begin(), shadowLights.end(), [](const ShadowCaster& sc) {
        return dynamic_cast<DirectionalLight*>(sc.light) != nullptr;
    });

    uint32_t numDirShadows = 0, numSpotShadows = 0;
    for (auto& sc : shadowLights) {
        if (dynamic_cast<DirectionalLight*>(sc.light)) numDirShadows++;
        else numSpotShadows++;
    }

    // Build shadow uniform buffer data:
    // [count(u32), numDirShadows(u32), numSpotShadows(u32), pad,
    //  light0{VP(64), bias(4), normalBias(4), pad(8)}, ...]
    const size_t shadowUniformSize = sl.shadowUniformSize();
    std::vector<float> shadowData(shadowUniformSize / sizeof(float), 0.0f);
    auto countBits = static_cast<uint32_t>(shadowLights.size());
    std::memcpy(&shadowData[0], &countBits, sizeof(uint32_t));
    std::memcpy(&shadowData[1], &numDirShadows, sizeof(uint32_t));
    std::memcpy(&shadowData[2], &numSpotShadows, sizeof(uint32_t));

    for (int i = 0; i < static_cast<int>(shadowLights.size()); i++) {
        auto* light = shadowLights[i].light;
        auto& shadow = shadowLights[i].shadowInterface->shadow;
        shadow->updateMatrices(*light);

        lights_[i].bias = shadow->bias;
        lights_[i].normalBias = shadow->normalBias;

        Matrix4 shadowVP;
        shadowVP.multiplyMatrices(shadow->camera->projectionMatrix,
                                  shadow->camera->matrixWorldInverse);
        {
            auto& e = shadowVP.elements;
            e[2]  = 0.5f * e[2]  + 0.5f * e[3];
            e[6]  = 0.5f * e[6]  + 0.5f * e[7];
            e[10] = 0.5f * e[10] + 0.5f * e[11];
            e[14] = 0.5f * e[14] + 0.5f * e[15];
        }
        lights_[i].lightVP = shadowVP;

        size_t offset = 4 + i * (ShadowLimits::shadowUniformPerLight / sizeof(float));
        std::memcpy(&shadowData[offset], shadowVP.elements.data(), 64);
        shadowData[offset + 16] = shadow->bias;
        shadowData[offset + 17] = shadow->normalBias;
    }
    wgpuQueueWriteBuffer(state_.queue, uniformBuffer_, 0, shadowData.data(), shadowUniformSize);

    // Render depth pass for each shadow-casting light
    for (int i = 0; i < static_cast<int>(shadowLights.size()); i++) {
        auto& shadow = shadowLights[i].shadowInterface->shadow;

        WGPUCommandEncoderDescriptor encDesc{};
        encDesc.label = WGPUStringView{"shadow_enc", WGPU_STRLEN} ;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state_.device, &encDesc);

        Matrix4 lightProj = shadow->camera->projectionMatrix;
        {
            auto& e = lightProj.elements;
            e[2]  = 0.5f * e[2]  + 0.5f * e[3];
            e[6]  = 0.5f * e[6]  + 0.5f * e[7];
            e[10] = 0.5f * e[10] + 0.5f * e[11];
            e[14] = 0.5f * e[14] + 0.5f * e[15];
        }
        Matrix4 lightVP;
        lightVP.multiplyMatrices(lightProj, shadow->camera->matrixWorldInverse);

        renderPass(encoder, scene, lightVP, i);

        WGPUCommandBufferDescriptor cmdDesc{};
        cmdDesc.label = WGPUStringView{"shadow_cmd", WGPU_STRLEN} ;
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
        wgpuQueueSubmit(state_.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }

    // Render depth pass for each point shadow light (6 faces each)
    numPointShadows_ = static_cast<int>(pointShadowLights.size());

    const size_t ptUniformSize = sl.pointShadowUniformSize();
    std::vector<float> ptData(ptUniformSize / sizeof(float), 0.0f);
    auto ptCount = static_cast<uint32_t>(pointShadowLights.size());
    std::memcpy(&ptData[0], &ptCount, sizeof(uint32_t));

    for (int pi = 0; pi < static_cast<int>(pointShadowLights.size()); pi++) {
        auto* pl = pointShadowLights[pi];
        auto* plShadow = dynamic_cast<PointLightShadow*>(pl->shadow.get());

        Vector3 lightPos;
        lightPos.setFromMatrixPosition(*pl->matrixWorld);

        float nearVal = plShadow ? plShadow->camera->nearPlane : 0.5f;
        float farVal  = pl->distance > 0.0f ? pl->distance : (plShadow ? plShadow->camera->farPlane : 500.0f);
        float biasVal = plShadow ? plShadow->bias : 0.005f;

        size_t off = 4 + static_cast<size_t>(pi) * (ShadowLimits::pointShadowPerLight / sizeof(float));
        ptData[off+0] = lightPos.x; ptData[off+1] = lightPos.y; ptData[off+2] = lightPos.z;
        ptData[off+3] = nearVal;
        ptData[off+4] = biasVal;
        ptData[off+5] = farVal;

        if (!plShadow) continue;

        std::vector<Mesh*> ptMeshes;
        collectShadowMeshes(scene, ptMeshes);
        if (static_cast<int>(ptMeshes.size()) > kMaxShadowMeshes)
            ptMeshes.resize(kMaxShadowMeshes);

        for (int face = 0; face < 6; face++) {
            plShadow->updateMatrices(*pl, static_cast<size_t>(face));

            Matrix4 lightProj = plShadow->camera->projectionMatrix;
            {
                auto& e = lightProj.elements;
                e[2]  = 0.5f * e[2]  + 0.5f * e[3];
                e[6]  = 0.5f * e[6]  + 0.5f * e[7];
                e[10] = 0.5f * e[10] + 0.5f * e[11];
                e[14] = 0.5f * e[14] + 0.5f * e[15];
            }
            Matrix4 lightVP;
            lightVP.multiplyMatrices(lightProj, plShadow->camera->matrixWorldInverse);

            // Upload all MVPs for this face before encoding draws.
            if (!ptMeshes.empty()) {
                std::vector<uint8_t> staging(ptMeshes.size() * kDynStride, 0);
                for (int mi = 0; mi < static_cast<int>(ptMeshes.size()); mi++) {
                    Matrix4 mvp;
                    mvp.multiplyMatrices(lightVP, *ptMeshes[mi]->matrixWorld);
                    std::memcpy(staging.data() + mi * kDynStride, mvp.elements.data(), 64);
                }
                wgpuQueueWriteBuffer(state_.queue, depthTransformBuffer_, 0,
                                     staging.data(), staging.size());
            }

            int layerIdx = pi * 6 + face;

            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"pt_shadow_enc", WGPU_STRLEN} ;
            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state_.device, &encDesc);

            WGPURenderPassDepthStencilAttachment depthAtt{};
            depthAtt.view = ptLayerViews_[layerIdx];
            depthAtt.depthLoadOp = WGPULoadOp_Clear;
            depthAtt.depthStoreOp = WGPUStoreOp_Store;
            depthAtt.depthClearValue = 1.0f;

            WGPURenderPassDescriptor passDesc{};
            passDesc.label = WGPUStringView{"pt_shadow_pass", WGPU_STRLEN} ;
            passDesc.colorAttachmentCount = 0;
            passDesc.colorAttachments = nullptr;
            passDesc.depthStencilAttachment = &depthAtt;

            WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
            wgpuRenderPassEncoderSetViewport(pass, 0, 0, mapSize, mapSize, 0.0f, 1.0f);

            for (int mi = 0; mi < static_cast<int>(ptMeshes.size()); mi++) {
                Mesh* mesh = ptMeshes[mi];
                uint32_t dynOffset = static_cast<uint32_t>(mi) * kDynStride;

                WGPUBindGroup bg = nullptr;
                if (meshIsSkinned(mesh)) {
                    wgpuRenderPassEncoderSetPipeline(pass, skinnedDepthPipeline_);
                    bg = buildSkinnedBindGroup(mesh);
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 1, &dynOffset);
                } else if (meshHasMorphTargets(mesh)) {
                    wgpuRenderPassEncoderSetPipeline(pass, morphDepthPipeline_);
                    bg = buildMorphBindGroup(mesh);
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 1, &dynOffset);
                } else {
                    wgpuRenderPassEncoderSetPipeline(pass, depthPipeline_);
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, depthBindGroup_, 1, &dynOffset);
                }

                auto& gb = geometries_.getOrCreateGeometryBuffers(mesh->geometry().get());
                if (gb.vertexBuffer) {
                    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                         gb.vertexCount * VERTEX_STRIDE);
                    if (gb.indexBuffer) {
                        wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                             WGPUIndexFormat_Uint32, 0,
                                                             gb.indexCount * sizeof(uint32_t));
                        wgpuRenderPassEncoderDrawIndexed(pass, gb.indexCount, 1, 0, 0, 0);
                    } else {
                        wgpuRenderPassEncoderDraw(pass, gb.vertexCount, 1, 0, 0);
                    }
                }
                if (bg) wgpuBindGroupRelease(bg);
            }

            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);

            WGPUCommandBufferDescriptor cmdDesc{};
            cmdDesc.label = WGPUStringView{"pt_shadow_cmd", WGPU_STRLEN} ;
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
            wgpuQueueSubmit(state_.queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
        }
    }
    wgpuQueueWriteBuffer(state_.queue, ptUniformBuffer_, 0, ptData.data(), ptUniformSize);

    if (needsUpdate) needsUpdate = false;
}

WGPUBindGroup WgpuShadowMap::buildSkinnedBindGroup(Mesh* mesh) {
    auto* sm = static_cast<SkinnedMesh*>(mesh);
    auto& skel = *sm->skeleton;
    skel.update();

    uint32_t boneCount = static_cast<uint32_t>(skel.bones.size());
    size_t headerFloats = 16 + 16 + 4; // bindMatrix, bindMatrixInverse, boneCount + 3 pad
    size_t totalFloats = headerFloats + boneCount * 16;
    size_t skinSz = totalFloats * sizeof(float);

    auto& entry = skinCache_[mesh];

    // Skin data buffer (bone matrices) — persistent, updated every frame
    if (!entry.skinBuf || entry.skinBufSize < skinSz) {
        if (entry.skinBuf) wgpuBufferRelease(entry.skinBuf);
        WGPUBufferDescriptor bd{}; bd.size = skinSz;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        entry.skinBuf = wgpuDeviceCreateBuffer(state_.device, &bd);
        entry.skinBufSize = skinSz;
    }
    std::vector<float> skinData(totalFloats, 0.f);
    std::memcpy(skinData.data(),      sm->bindMatrix.elements.data(),        64);
    std::memcpy(skinData.data() + 16, sm->bindMatrixInverse.elements.data(), 64);
    reinterpret_cast<uint32_t*>(skinData.data() + 32)[0] = boneCount;
    if (!skel.boneMatrices.empty())
        std::memcpy(skinData.data() + headerFloats, skel.boneMatrices.data(), boneCount * 16 * sizeof(float));
    wgpuQueueWriteBuffer(state_.queue, entry.skinBuf, 0, skinData.data(), skinSz);

    // Skin vertex buffer (index/weight per vertex) — created once, never changes
    if (!entry.vertexBuf) {
        auto* geom = mesh->geometry().get();
        auto* idxAttr = geom->getAttribute<float>("skinIndex");
        auto* wgtAttr = geom->getAttribute<float>("skinWeight");
        uint32_t vCount = idxAttr->count();
        std::vector<float> vdata(vCount * 8);
        for (uint32_t v = 0; v < vCount; v++) {
            vdata[v*8+0] = idxAttr->getX(v); vdata[v*8+1] = idxAttr->getY(v);
            vdata[v*8+2] = idxAttr->getZ(v); vdata[v*8+3] = idxAttr->getW(v);
            vdata[v*8+4] = wgtAttr->getX(v); vdata[v*8+5] = wgtAttr->getY(v);
            vdata[v*8+6] = wgtAttr->getZ(v); vdata[v*8+7] = wgtAttr->getW(v);
        }
        size_t vsz = vdata.size() * sizeof(float);
        WGPUBufferDescriptor bd{}; bd.size = vsz;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        entry.vertexBuf = wgpuDeviceCreateBuffer(state_.device, &bd);
        entry.vertexBufSize = vsz;
        wgpuQueueWriteBuffer(state_.queue, entry.vertexBuf, 0, vdata.data(), vsz);
    }

    WGPUBindGroupEntry entries[3]{};
    entries[0].binding = 0; entries[0].buffer = depthTransformBuffer_; entries[0].size = 64;
    entries[1].binding = 1; entries[1].buffer = entry.skinBuf;    entries[1].size = entry.skinBufSize;
    entries[2].binding = 2; entries[2].buffer = entry.vertexBuf;  entries[2].size = entry.vertexBufSize;
    WGPUBindGroupDescriptor bgd{}; bgd.layout = skinnedDepthBGL_; bgd.entryCount = 3; bgd.entries = entries;
    return wgpuDeviceCreateBindGroup(state_.device, &bgd);
}

WGPUBindGroup WgpuShadowMap::buildMorphBindGroup(Mesh* mesh) {
    auto* geom = mesh->geometry().get();
    auto& morphAttrs = geom->getMorphAttributes().at("position");
    uint32_t numTargets  = static_cast<uint32_t>(morphAttrs.size());
    uint32_t vertexCount = static_cast<uint32_t>(geom->getAttribute<float>("position")->count());
    auto& influences = mesh->morphTargetInfluences();

    size_t headerSize    = 4;   // numTargets + 3 pad
    size_t influenceSize = 8;   // 2 × vec4
    size_t posSize       = numTargets * vertexCount * 4;
    size_t totalFloats   = headerSize + influenceSize + posSize;
    size_t bufSz         = totalFloats * sizeof(float);

    auto& entry = morphCache_[mesh];
    if (!entry.buf || entry.bufSize < bufSz) {
        if (entry.buf) wgpuBufferRelease(entry.buf);
        WGPUBufferDescriptor bd{}; bd.size = bufSz;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        entry.buf = wgpuDeviceCreateBuffer(state_.device, &bd);
        entry.bufSize = bufSz;
    }

    std::vector<float> morphData(totalFloats, 0.f);
    reinterpret_cast<uint32_t*>(morphData.data())[0] = numTargets;
    for (uint32_t t = 0; t < numTargets && t < 8; t++) {
        if (t < influences.size()) morphData[headerSize + t] = influences[t];
    }
    size_t posOffset = headerSize + influenceSize;
    for (uint32_t t = 0; t < numTargets; t++) {
        auto* attr = morphAttrs[t] ? morphAttrs[t]->typed<float>() : nullptr;
        if (!attr) continue;
        for (uint32_t v = 0; v < vertexCount; v++) {
            size_t idx = posOffset + (t * vertexCount + v) * 4;
            morphData[idx+0] = attr->getX(v); morphData[idx+1] = attr->getY(v);
            morphData[idx+2] = attr->getZ(v);
        }
    }
    wgpuQueueWriteBuffer(state_.queue, entry.buf, 0, morphData.data(), bufSz);

    WGPUBindGroupEntry entries[2]{};
    entries[0].binding = 0; entries[0].buffer = depthTransformBuffer_; entries[0].size = 64;
    entries[1].binding = 1; entries[1].buffer = entry.buf;             entries[1].size = entry.bufSize;
    WGPUBindGroupDescriptor bgd{}; bgd.layout = morphDepthBGL_; bgd.entryCount = 2; bgd.entries = entries;
    return wgpuDeviceCreateBindGroup(state_.device, &bgd);
}

void WgpuShadowMap::renderPass(WGPUCommandEncoder encoder, Object3D& scene,
                               const Matrix4& lightVP, int lightIndex) {

    const auto mapSize = static_cast<float>(state_.shadowLimits.mapSize);

    // Collect shadow-casting meshes and upload all MVPs before encoding draws.
    std::vector<Mesh*> meshes;
    collectShadowMeshes(scene, meshes);
    if (static_cast<int>(meshes.size()) > kMaxShadowMeshes)
        meshes.resize(kMaxShadowMeshes);

    if (!meshes.empty()) {
        std::vector<uint8_t> staging(meshes.size() * kDynStride, 0);
        for (int i = 0; i < static_cast<int>(meshes.size()); i++) {
            Matrix4 mvp;
            mvp.multiplyMatrices(lightVP, *meshes[i]->matrixWorld);
            std::memcpy(staging.data() + i * kDynStride, mvp.elements.data(), 64);
        }
        wgpuQueueWriteBuffer(state_.queue, depthTransformBuffer_, 0,
                             staging.data(), staging.size());
    }

    WGPURenderPassDepthStencilAttachment depthAttachment{};
    depthAttachment.view = lights_[lightIndex].layerView;
    depthAttachment.depthLoadOp = WGPULoadOp_Clear;
    depthAttachment.depthStoreOp = WGPUStoreOp_Store;
    depthAttachment.depthClearValue = 1.0f;

    WGPURenderPassDescriptor passDesc{};
    passDesc.label = WGPUStringView{"shadow_pass", WGPU_STRLEN} ;
    passDesc.colorAttachmentCount = 0;
    passDesc.colorAttachments = nullptr;
    passDesc.depthStencilAttachment = &depthAttachment;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, mapSize, mapSize, 0.0f, 1.0f);

    for (int i = 0; i < static_cast<int>(meshes.size()); i++) {
        Mesh* mesh = meshes[i];
        uint32_t dynOffset = static_cast<uint32_t>(i) * kDynStride;

        WGPUBindGroup bg = nullptr;
        if (meshIsSkinned(mesh)) {
            wgpuRenderPassEncoderSetPipeline(pass, skinnedDepthPipeline_);
            bg = buildSkinnedBindGroup(mesh);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 1, &dynOffset);
        } else if (meshHasMorphTargets(mesh)) {
            wgpuRenderPassEncoderSetPipeline(pass, morphDepthPipeline_);
            bg = buildMorphBindGroup(mesh);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 1, &dynOffset);
        } else {
            wgpuRenderPassEncoderSetPipeline(pass, depthPipeline_);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, depthBindGroup_, 1, &dynOffset);
        }

        auto& gb = geometries_.getOrCreateGeometryBuffers(mesh->geometry().get());
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                 gb.vertexCount * VERTEX_STRIDE);
            if (gb.indexBuffer) {
                wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                     WGPUIndexFormat_Uint32, 0,
                                                     gb.indexCount * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, gb.indexCount, 1, 0, 0, 0);
            } else {
                wgpuRenderPassEncoderDraw(pass, gb.vertexCount, 1, 0, 0);
            }
        }

        if (bg) wgpuBindGroupRelease(bg);
    }

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}


void WgpuShadowMap::dispose() {
    if (!initialized_) return;

    for (auto& light : lights_) {
        if (light.layerView) wgpuTextureViewRelease(light.layerView);
    }
    if (depthArrayView_) wgpuTextureViewRelease(depthArrayView_);
    if (depthArrayTexture_) wgpuTextureRelease(depthArrayTexture_);
    if (comparisonSampler_) wgpuSamplerRelease(comparisonSampler_);
    if (uniformBuffer_) wgpuBufferRelease(uniformBuffer_);
    if (depthBindGroup_) wgpuBindGroupRelease(depthBindGroup_);
    if (depthTransformBuffer_) wgpuBufferRelease(depthTransformBuffer_);
    if (depthPipeline_) wgpuRenderPipelineRelease(depthPipeline_);
    if (depthPipelineLayout_) wgpuPipelineLayoutRelease(depthPipelineLayout_);
    if (depthBindGroupLayout_) wgpuBindGroupLayoutRelease(depthBindGroupLayout_);
    if (depthShader_) wgpuShaderModuleRelease(depthShader_);

    if (skinnedDepthPipeline_) wgpuRenderPipelineRelease(skinnedDepthPipeline_);
    if (skinnedDepthPipelineLayout_) wgpuPipelineLayoutRelease(skinnedDepthPipelineLayout_);
    if (skinnedDepthBGL_) wgpuBindGroupLayoutRelease(skinnedDepthBGL_);
    if (skinnedDepthShader_) wgpuShaderModuleRelease(skinnedDepthShader_);

    if (morphDepthPipeline_) wgpuRenderPipelineRelease(morphDepthPipeline_);
    if (morphDepthPipelineLayout_) wgpuPipelineLayoutRelease(morphDepthPipelineLayout_);
    if (morphDepthBGL_) wgpuBindGroupLayoutRelease(morphDepthBGL_);
    if (morphDepthShader_) wgpuShaderModuleRelease(morphDepthShader_);

    for (auto& [mesh, e] : skinCache_) {
        if (e.skinBuf)   wgpuBufferRelease(e.skinBuf);
        if (e.vertexBuf) wgpuBufferRelease(e.vertexBuf);
    }
    skinCache_.clear();

    for (auto& [mesh, e] : morphCache_) {
        if (e.buf) wgpuBufferRelease(e.buf);
    }
    morphCache_.clear();

    depthArrayTexture_ = nullptr;
    depthArrayView_ = nullptr;
    comparisonSampler_ = nullptr;
    uniformBuffer_ = nullptr;
    depthBindGroup_ = nullptr;
    depthTransformBuffer_ = nullptr;
    depthPipeline_ = nullptr;
    depthPipelineLayout_ = nullptr;
    depthBindGroupLayout_ = nullptr;
    depthShader_ = nullptr;
    skinnedDepthPipeline_ = nullptr; skinnedDepthPipelineLayout_ = nullptr;
    skinnedDepthBGL_ = nullptr; skinnedDepthShader_ = nullptr;
    morphDepthPipeline_ = nullptr; morphDepthPipelineLayout_ = nullptr;
    morphDepthBGL_ = nullptr; morphDepthShader_ = nullptr;

    lights_.clear();

    // Point shadow resources
    for (auto& view : ptLayerViews_) {
        if (view) { wgpuTextureViewRelease(view); view = nullptr; }
    }
    ptLayerViews_.clear();
    if (ptDepthArrayView_) { wgpuTextureViewRelease(ptDepthArrayView_); ptDepthArrayView_ = nullptr; }
    if (ptDepthArrayTexture_) { wgpuTextureRelease(ptDepthArrayTexture_); ptDepthArrayTexture_ = nullptr; }
    if (ptUniformBuffer_) { wgpuBufferRelease(ptUniformBuffer_); ptUniformBuffer_ = nullptr; }

    initialized_ = false;
    active_ = false;
    activeLightCount_ = 0;
    numPointShadows_ = 0;
}
