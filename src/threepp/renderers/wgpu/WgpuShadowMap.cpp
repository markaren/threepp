#include "WgpuShadowMap.hpp"
#include "WgpuGeometries.hpp"
#include "WgpuShaders.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/lights/PointLightShadow.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

using namespace threepp;
using namespace threepp::wgpu;

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

    // Depth-only transform buffer (one mat4x4)
    {
        WGPUBufferDescriptor bd{};
        bd.label = WGPUStringView{"shadow_xform", WGPU_STRLEN} ;
        bd.size = 64;
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

    // Bind group layout: one uniform buffer at binding 0
    WGPUBindGroupLayoutEntry bglEntry{};
    bglEntry.binding = 0;
    bglEntry.visibility = WGPUShaderStage_Vertex;
    bglEntry.buffer.type = WGPUBufferBindingType_Uniform;
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
            wgpuRenderPassEncoderSetPipeline(pass, depthPipeline_);
            renderObject(pass, scene, lightVP);
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

void WgpuShadowMap::renderPass(WGPUCommandEncoder encoder, Object3D& scene,
                               const Matrix4& lightVP, int lightIndex) {

    const auto mapSize = static_cast<float>(state_.shadowLimits.mapSize);

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
    wgpuRenderPassEncoderSetPipeline(pass, depthPipeline_);

    renderObject(pass, scene, lightVP);

    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

void WgpuShadowMap::renderObject(WGPURenderPassEncoder pass, Object3D& object,
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
            bgDesc.label = WGPUStringView{"shadow_bg", WGPU_STRLEN} ;
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

void WgpuShadowMap::dispose() {
    if (!initialized_) return;

    for (auto& light : lights_) {
        if (light.layerView) wgpuTextureViewRelease(light.layerView);
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
