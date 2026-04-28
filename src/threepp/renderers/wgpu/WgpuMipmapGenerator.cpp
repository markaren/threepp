
#include "WgpuMipmapGenerator.hpp"

#include <algorithm>
#include <cstring>

using namespace threepp;
using namespace threepp::wgpu;

WgpuMipmapGenerator::WgpuMipmapGenerator(WgpuState& state)
    : state_(state) {}

WgpuMipmapGenerator::~WgpuMipmapGenerator() {
    if (pipeline_)       { wgpuRenderPipelineRelease(pipeline_);       pipeline_ = nullptr; }
    if (pipelineLayout_) { wgpuPipelineLayoutRelease(pipelineLayout_); pipelineLayout_ = nullptr; }
    if (bgl_)            { wgpuBindGroupLayoutRelease(bgl_);            bgl_ = nullptr; }
    if (sampler_)        { wgpuSamplerRelease(sampler_);                sampler_ = nullptr; }
    if (shader_)         { wgpuShaderModuleRelease(shader_);            shader_ = nullptr; }
}

void WgpuMipmapGenerator::ensurePipeline(WGPUTextureFormat format) {
    if (pipeline_ && pipelineFormat_ == format) return;

    if (pipeline_)       { wgpuRenderPipelineRelease(pipeline_);       pipeline_ = nullptr; }
    if (pipelineLayout_) { wgpuPipelineLayoutRelease(pipelineLayout_); pipelineLayout_ = nullptr; }
    if (bgl_)            { wgpuBindGroupLayoutRelease(bgl_);            bgl_ = nullptr; }
    if (sampler_)        { wgpuSamplerRelease(sampler_);                sampler_ = nullptr; }
    if (shader_)         { wgpuShaderModuleRelease(shader_);            shader_ = nullptr; }

    // Fullscreen triangle blit: samples from a 2D texture view (one mip level)
    // and writes to the next level via a render attachment.
    // The vertex shader emits a canonical full-screen triangle without a vertex buffer.
    const char* wgsl = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    var p = array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    var o: V;
    o.pos = vec4<f32>(p[vi], 0., 1.);
    o.uv  = p[vi] * 0.5 + vec2<f32>(0.5, 0.5);
    o.uv.y = 1.0 - o.uv.y;
    return o;
}
@fragment fn fs(in: V) -> @location(0) vec4<f32> {
    return textureSample(src, samp, in.uv);
}
)";
    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code = {.data = wgsl, .length = static_cast<size_t>(strlen(wgsl))};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    shader_ = wgpuDeviceCreateShaderModule(state_.device, &smDesc);

    WGPUBindGroupLayoutEntry bglEntries[2]{};
    bglEntries[0].binding   = 0;
    bglEntries[0].visibility = WGPUShaderStage_Fragment;
    bglEntries[0].texture.sampleType    = WGPUTextureSampleType_Float;
    bglEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    bglEntries[1].binding   = 1;
    bglEntries[1].visibility = WGPUShaderStage_Fragment;
    bglEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;

    WGPUBindGroupLayoutDescriptor bgld{};
    bgld.entryCount = 2;
    bgld.entries    = bglEntries;
    bgl_ = wgpuDeviceCreateBindGroupLayout(state_.device, &bgld);

    WGPUPipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts     = &bgl_;
    pipelineLayout_ = wgpuDeviceCreatePipelineLayout(state_.device, &pld);

    WGPUSamplerDescriptor sd{};
    sd.magFilter    = WGPUFilterMode_Linear;
    sd.minFilter    = WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.lodMaxClamp = 32.0f;
    sd.maxAnisotropy = 1;
    sampler_ = wgpuDeviceCreateSampler(state_.device, &sd);

    WGPUVertexState vs{};
    vs.module     = shader_;
    vs.entryPoint = {.data = "vs", .length = 2};

    WGPUColorTargetState ct{};
    ct.format    = format;
    ct.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fs{};
    fs.module      = shader_;
    fs.entryPoint  = {.data = "fs", .length = 2};
    fs.targetCount = 1;
    fs.targets     = &ct;

    WGPURenderPipelineDescriptor rpd{};
    rpd.label               = WGPUStringView{"mipmap_blit_pipe", WGPU_STRLEN};
    rpd.layout              = pipelineLayout_;
    rpd.vertex              = vs;
    rpd.fragment            = &fs;
    rpd.primitive.topology  = WGPUPrimitiveTopology_TriangleList;
    rpd.multisample.count   = 1;
    rpd.multisample.mask    = 0xFFFFFFFF;
    pipeline_ = wgpuDeviceCreateRenderPipeline(state_.device, &rpd);

    pipelineFormat_ = format;
}

void WgpuMipmapGenerator::blitLayer(WGPUCommandEncoder encoder,
                                    WGPUTexture texture,
                                    uint32_t baseArrayLayer,
                                    uint32_t srcLevel,
                                    WGPUTextureViewDimension viewDim) {
    uint32_t dstLevel = srcLevel + 1;

    WGPUTextureViewDescriptor srcVd{};
    srcVd.dimension      = viewDim;
    srcVd.baseMipLevel   = srcLevel;
    srcVd.mipLevelCount  = 1;
    srcVd.baseArrayLayer = baseArrayLayer;
    srcVd.arrayLayerCount = 1;
    srcVd.aspect         = WGPUTextureAspect_All;
    WGPUTextureView srcView = wgpuTextureCreateView(texture, &srcVd);

    WGPUTextureViewDescriptor dstVd{};
    dstVd.dimension      = WGPUTextureViewDimension_2D;
    dstVd.baseMipLevel   = dstLevel;
    dstVd.mipLevelCount  = 1;
    dstVd.baseArrayLayer = baseArrayLayer;
    dstVd.arrayLayerCount = 1;
    dstVd.aspect         = WGPUTextureAspect_All;
    WGPUTextureView dstView = wgpuTextureCreateView(texture, &dstVd);

    WGPUBindGroupEntry bgEntries[2]{};
    bgEntries[0].binding     = 0;
    bgEntries[0].textureView = srcView;
    bgEntries[1].binding     = 1;
    bgEntries[1].sampler     = sampler_;

    WGPUBindGroupDescriptor bgd{};
    bgd.layout     = bgl_;
    bgd.entryCount = 2;
    bgd.entries    = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(state_.device, &bgd);

    WGPURenderPassColorAttachment ca{};
    ca.view       = dstView;
    ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    ca.loadOp     = WGPULoadOp_Clear;
    ca.storeOp    = WGPUStoreOp_Store;
    ca.clearValue = {0, 0, 0, 1};

    WGPURenderPassDescriptor rpassDesc{};
    rpassDesc.colorAttachmentCount = 1;
    rpassDesc.colorAttachments     = &ca;

    WGPURenderPassEncoder rpe = wgpuCommandEncoderBeginRenderPass(encoder, &rpassDesc);
    wgpuRenderPassEncoderSetPipeline(rpe, pipeline_);
    wgpuRenderPassEncoderSetBindGroup(rpe, 0, bg, 0, nullptr);
    wgpuRenderPassEncoderDraw(rpe, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(rpe);
    wgpuRenderPassEncoderRelease(rpe);

    wgpuBindGroupRelease(bg);
    wgpuTextureViewRelease(srcView);
    wgpuTextureViewRelease(dstView);
}

void WgpuMipmapGenerator::generate2D(WGPUTexture texture,
                                      uint32_t /*width*/, uint32_t /*height*/,
                                      uint32_t mipLevels,
                                      WGPUTextureFormat format) {
    if (mipLevels <= 1) return;
    ensurePipeline(format);

    WGPUCommandEncoderDescriptor ceDesc{};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state_.device, &ceDesc);

    for (uint32_t level = 0; level + 1 < mipLevels; ++level) {
        blitLayer(encoder, texture, 0, level, WGPUTextureViewDimension_2D);
    }

    WGPUCommandBufferDescriptor cbDesc{};
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
    wgpuQueueSubmit(state_.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(encoder);
}

void WgpuMipmapGenerator::generateCube(WGPUTexture texture,
                                        uint32_t /*faceSize*/,
                                        uint32_t mipLevels,
                                        WGPUTextureFormat format) {
    if (mipLevels <= 1) return;
    ensurePipeline(format);

    WGPUCommandEncoderDescriptor ceDesc{};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state_.device, &ceDesc);

    for (uint32_t face = 0; face < 6; ++face) {
        for (uint32_t level = 0; level + 1 < mipLevels; ++level) {
            blitLayer(encoder, texture, face, level, WGPUTextureViewDimension_2D);
        }
    }

    WGPUCommandBufferDescriptor cbDesc{};
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
    wgpuQueueSubmit(state_.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(encoder);
}
