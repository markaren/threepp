#include "DawnShadowMap.hpp"
#include "DawnGeometries.hpp"
#include "DawnShaders.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

using namespace threepp;
using namespace threepp::dawn;

DawnShadowMap::DawnShadowMap(DawnState& state, DawnGeometries& geometries)
    : state_(state), geometries_(geometries) {}

void DawnShadowMap::init() {
    if (initialized_) return;
    initialized_ = true;

    // 2D array depth texture with one layer per shadow-casting light
    {
        WGPUTextureDescriptor td{};
        td.label = {.data = "shadow_depth_array", .length = 18};
        td.size = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, MAX_SHADOW_LIGHTS};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = WGPUTextureFormat_Depth32Float;
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
        depthArrayTexture_ = wgpuDeviceCreateTexture(state_.device, &td);

        // Full array view for sampling in the fragment shader
        WGPUTextureViewDescriptor avd{};
        avd.label = {.data = "shadow_array_view", .length = 17};
        avd.format = WGPUTextureFormat_Depth32Float;
        avd.dimension = WGPUTextureViewDimension_2DArray;
        avd.baseMipLevel = 0;
        avd.mipLevelCount = 1;
        avd.baseArrayLayer = 0;
        avd.arrayLayerCount = MAX_SHADOW_LIGHTS;
        avd.aspect = WGPUTextureAspect_DepthOnly;
        depthArrayView_ = wgpuTextureCreateView(depthArrayTexture_, &avd);

        // Per-layer views for rendering
        for (int i = 0; i < MAX_SHADOW_LIGHTS; i++) {
            WGPUTextureViewDescriptor lvd{};
            lvd.label = {.data = "shadow_layer_view", .length = 17};
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
    sd.label = {.data = "shadow_samp", .length = 11};
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.compare = WGPUCompareFunction_Less;
    sd.maxAnisotropy = 1;
    comparisonSampler_ = wgpuDeviceCreateSampler(state_.device, &sd);

    // Shadow uniform buffer
    {
        WGPUBufferDescriptor bd{};
        bd.label = {.data = "shadow_ub", .length = 9};
        bd.size = SHADOW_UNIFORM_SIZE;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        uniformBuffer_ = wgpuDeviceCreateBuffer(state_.device, &bd);
    }

    // Depth-only transform buffer (one mat4x4)
    {
        WGPUBufferDescriptor bd{};
        bd.label = {.data = "shadow_xform", .length = 12};
        bd.size = 64;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        depthTransformBuffer_ = wgpuDeviceCreateBuffer(state_.device, &bd);
    }

    // Depth-only shader module
    std::string depthWGSL = buildDepthWGSL();

    WGPUShaderSourceWGSL wgslSource{};
    wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSource.code = {.data = depthWGSL.c_str(), .length = depthWGSL.size()};

    WGPUShaderModuleDescriptor smd{};
    smd.nextInChain = &wgslSource.chain;
    smd.label = {.data = "shadow_shader", .length = 13};
    depthShader_ = wgpuDeviceCreateShaderModule(state_.device, &smd);

    // Bind group layout: one uniform buffer at binding 0
    WGPUBindGroupLayoutEntry bglEntry{};
    bglEntry.binding = 0;
    bglEntry.visibility = WGPUShaderStage_Vertex;
    bglEntry.buffer.type = WGPUBufferBindingType_Uniform;
    bglEntry.buffer.minBindingSize = 64;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.label = {.data = "shadow_bgl", .length = 10};
    bglDesc.entryCount = 1;
    bglDesc.entries = &bglEntry;
    depthBindGroupLayout_ = wgpuDeviceCreateBindGroupLayout(state_.device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.label = {.data = "shadow_pl", .length = 9};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &depthBindGroupLayout_;
    depthPipelineLayout_ = wgpuDeviceCreatePipelineLayout(state_.device, &plDesc);

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
    pipeDesc.label = {.data = "shadow_pipe", .length = 11};
    pipeDesc.layout = depthPipelineLayout_;

    WGPUStringView vsEntry = {.data = "vs_main", .length = 7};
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
}

void DawnShadowMap::beginFrame(Object3D& scene) {
    active_ = false;
    activeLightCount_ = 0;

    // Collect shadow-casting DirectionalLight and SpotLight instances
    struct ShadowCaster {
        Light* light;
        LightWithShadow* shadowInterface;
    };
    std::vector<ShadowCaster> shadowLights;

    std::function<void(Object3D&)> findShadowLights = [&](Object3D& obj) {
        if (obj.castShadow && static_cast<int>(shadowLights.size()) < MAX_SHADOW_LIGHTS) {
            if (auto dl = obj.as<DirectionalLight>()) {
                shadowLights.push_back({dl, dl});
            } else if (auto sl = obj.as<SpotLight>()) {
                shadowLights.push_back({sl, sl});
            }
        }
        for (auto& child : obj.children) findShadowLights(*child);
    };
    findShadowLights(scene);

    if (shadowLights.empty()) return;

    init();
    active_ = true;
    activeLightCount_ = static_cast<int>(shadowLights.size());

    // Sort shadow lights: DirectionalLights first, then SpotLights
    // (matches the order in DawnLights after stable_partition).
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
    std::vector<float> shadowData(SHADOW_UNIFORM_SIZE / sizeof(float), 0.0f);
    auto countBits = static_cast<uint32_t>(shadowLights.size());
    std::memcpy(&shadowData[0], &countBits, sizeof(uint32_t));
    std::memcpy(&shadowData[1], &numDirShadows, sizeof(uint32_t));
    std::memcpy(&shadowData[2], &numSpotShadows, sizeof(uint32_t));

    for (int i = 0; i < static_cast<int>(shadowLights.size()); i++) {
        auto* light = shadowLights[i].light;
        auto& shadow = shadowLights[i].shadowInterface->shadow;
        shadow->updateMatrices(*light);

        lights_[i].lightVP = shadow->matrix;
        lights_[i].bias = shadow->bias;
        lights_[i].normalBias = shadow->normalBias;

        // Z-remap the light VP from [-1,1] to [0,1] for the uniform buffer
        Matrix4 shadowVP = shadow->matrix;
        {
            auto& e = shadowVP.elements;
            e[2]  = 0.5f * e[2]  + 0.5f * e[3];
            e[6]  = 0.5f * e[6]  + 0.5f * e[7];
            e[10] = 0.5f * e[10] + 0.5f * e[11];
            e[14] = 0.5f * e[14] + 0.5f * e[15];
        }

        // Offset: 4 floats header + i * (SHADOW_UNIFORM_PER_LIGHT / 4) floats per light
        size_t offset = 4 + i * (SHADOW_UNIFORM_PER_LIGHT / sizeof(float));
        std::memcpy(&shadowData[offset], shadowVP.elements.data(), 64);
        shadowData[offset + 16] = shadow->bias;
        shadowData[offset + 17] = shadow->normalBias;
    }
    wgpuQueueWriteBuffer(state_.queue, uniformBuffer_, 0, shadowData.data(), SHADOW_UNIFORM_SIZE);

    // Render depth pass for each shadow-casting light
    for (int i = 0; i < static_cast<int>(shadowLights.size()); i++) {
        auto& shadow = shadowLights[i].shadowInterface->shadow;

        WGPUCommandEncoderDescriptor encDesc{};
        encDesc.label = {.data = "shadow_enc", .length = 10};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state_.device, &encDesc);

        // Build light view-projection with Z-remap for the depth pass
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
        cmdDesc.label = {.data = "shadow_cmd", .length = 10};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
        wgpuQueueSubmit(state_.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
    }
}

void DawnShadowMap::renderPass(WGPUCommandEncoder encoder, Object3D& scene,
                               const Matrix4& lightVP, int lightIndex) {

    WGPURenderPassDepthStencilAttachment depthAttachment{};
    depthAttachment.view = lights_[lightIndex].layerView;
    depthAttachment.depthLoadOp = WGPULoadOp_Clear;
    depthAttachment.depthStoreOp = WGPUStoreOp_Store;
    depthAttachment.depthClearValue = 1.0f;

    WGPURenderPassDescriptor passDesc{};
    passDesc.label = {.data = "shadow_pass", .length = 11};
    passDesc.colorAttachmentCount = 0;
    passDesc.colorAttachments = nullptr;
    passDesc.depthStencilAttachment = &depthAttachment;

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetPipeline(pass, depthPipeline_);

    renderObject(pass, scene, lightVP);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

void DawnShadowMap::renderObject(WGPURenderPassEncoder pass, Object3D& object,
                                 const Matrix4& lightVP) {

    if (auto mesh = object.as<Mesh>()) {
        auto geometry = mesh->geometry();
        if (mesh->castShadow && geometry && geometry->hasAttribute("position")) {
            Matrix4 mvp;
            mvp.multiplyMatrices(lightVP, *mesh->matrixWorld);

            wgpuQueueWriteBuffer(state_.queue, depthTransformBuffer_, 0, mvp.elements.data(), 64);

            WGPUBindGroupEntry entry{};
            entry.binding = 0;
            entry.buffer = depthTransformBuffer_;
            entry.offset = 0;
            entry.size = 64;

            WGPUBindGroupDescriptor bgDesc{};
            bgDesc.label = {.data = "shadow_bg", .length = 9};
            bgDesc.layout = depthBindGroupLayout_;
            bgDesc.entryCount = 1;
            bgDesc.entries = &entry;
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(state_.device, &bgDesc);

            wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

            auto& gb = geometries_.getOrCreateGeometryBuffers(geometry.get());
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
            wgpuBindGroupRelease(bg);
        }
    }

    for (auto& child : object.children) {
        renderObject(pass, *child, lightVP);
    }
}

void DawnShadowMap::dispose() {
    if (!initialized_) return;

    for (int i = 0; i < MAX_SHADOW_LIGHTS; i++) {
        if (lights_[i].layerView) wgpuTextureViewRelease(lights_[i].layerView);
    }
    if (depthArrayView_) wgpuTextureViewRelease(depthArrayView_);
    if (depthArrayTexture_) wgpuTextureRelease(depthArrayTexture_);
    if (comparisonSampler_) wgpuSamplerRelease(comparisonSampler_);
    if (uniformBuffer_) wgpuBufferRelease(uniformBuffer_);
    if (depthTransformBuffer_) wgpuBufferRelease(depthTransformBuffer_);
    if (depthPipeline_) wgpuRenderPipelineRelease(depthPipeline_);
    if (depthPipelineLayout_) wgpuPipelineLayoutRelease(depthPipelineLayout_);
    if (depthBindGroupLayout_) wgpuBindGroupLayoutRelease(depthBindGroupLayout_);
    if (depthShader_) wgpuShaderModuleRelease(depthShader_);

    depthArrayTexture_ = nullptr;
    depthArrayView_ = nullptr;
    comparisonSampler_ = nullptr;
    uniformBuffer_ = nullptr;
    depthTransformBuffer_ = nullptr;
    depthPipeline_ = nullptr;
    depthPipelineLayout_ = nullptr;
    depthBindGroupLayout_ = nullptr;
    depthShader_ = nullptr;

    for (int i = 0; i < MAX_SHADOW_LIGHTS; i++) {
        lights_[i] = {};
    }

    initialized_ = false;
    active_ = false;
    activeLightCount_ = 0;
}
