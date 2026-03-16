
#include "threepp/renderers/dawn/ComputePipeline.hpp"
#include "threepp/renderers/dawn/GPUTexture.hpp"
#include "threepp/renderers/dawn/GPUBuffer.hpp"
#include "threepp/renderers/DawnRenderer.hpp"

#include <webgpu/webgpu.h>

#include <unordered_map>
#include <stdexcept>

using namespace threepp;

namespace {

    WGPUTextureFormat toWGPUFormat(GPUTexture::Format fmt) {
        switch (fmt) {
            case GPUTexture::Format::RGBA32Float: return WGPUTextureFormat_RGBA32Float;
            case GPUTexture::Format::RG32Float:   return WGPUTextureFormat_RG32Float;
            case GPUTexture::Format::RGBA8Unorm:  return WGPUTextureFormat_RGBA8Unorm;
        }
        return WGPUTextureFormat_RGBA8Unorm;
    }

}// namespace

struct ComputePipeline::BindingInfo {
    BindingType type;
    // For textures
    WGPUTextureView textureView = nullptr;
    WGPUTextureFormat textureFormat = WGPUTextureFormat_Undefined;
    // For buffers
    WGPUBuffer buffer = nullptr;
    size_t bufferSize = 0;
};

struct ComputePipeline::Impl {
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUShaderModule shaderModule = nullptr;
    WGPUComputePipeline pipeline = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    std::string entryPoint;

    std::unordered_map<uint32_t, BindingInfo> bindings;
    bool pipelineBuilt = false;

    ~Impl() {
        if (pipeline) wgpuComputePipelineRelease(pipeline);
        if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
        if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
        if (shaderModule) wgpuShaderModuleRelease(shaderModule);
    }

    void buildPipeline() {
        if (pipelineBuilt) {
            // Rebuild bind group layout if bindings changed
            if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
            if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
            if (pipeline) wgpuComputePipelineRelease(pipeline);
        }

        // Build bind group layout entries
        std::vector<WGPUBindGroupLayoutEntry> bglEntries;
        for (auto& [binding, info] : bindings) {
            WGPUBindGroupLayoutEntry e{};
            e.binding = binding;
            e.visibility = WGPUShaderStage_Compute;

            switch (info.type) {
                case BindingType::StorageTextureWrite:
                    e.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
                    e.storageTexture.format = info.textureFormat;
                    e.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
                    break;
                case BindingType::StorageTextureRead:
                case BindingType::Texture:
                    e.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
                    e.texture.viewDimension = WGPUTextureViewDimension_2D;
                    break;
                case BindingType::UniformBuffer:
                    e.buffer.type = WGPUBufferBindingType_Uniform;
                    e.buffer.minBindingSize = info.bufferSize;
                    break;
            }
            bglEntries.push_back(e);
        }

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = {.data = "compute_bgl", .length = 11};
        bglDesc.entryCount = bglEntries.size();
        bglDesc.entries = bglEntries.data();
        bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = {.data = "compute_pl", .length = 10};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bindGroupLayout;
        pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        WGPUStringView ep = {.data = entryPoint.c_str(), .length = entryPoint.size()};
        WGPUComputePipelineDescriptor pipeDesc{};
        pipeDesc.label = {.data = "compute_pipe", .length = 12};
        pipeDesc.layout = pipelineLayout;
        pipeDesc.compute.module = shaderModule;
        pipeDesc.compute.entryPoint = ep;
        pipeline = wgpuDeviceCreateComputePipeline(device, &pipeDesc);

        pipelineBuilt = true;
    }
};

ComputePipeline::ComputePipeline(DawnRenderer& renderer, const std::string& wgslSource,
                                 const std::string& entryPoint)
    : impl_(new Impl()) {

    impl_->device = static_cast<WGPUDevice>(renderer.nativeDevice());
    impl_->queue = static_cast<WGPUQueue>(renderer.nativeQueue());
    impl_->entryPoint = entryPoint;

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.chain.next = nullptr;
    wgslSrc.code = {.data = wgslSource.c_str(), .length = wgslSource.size()};

    WGPUShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslSrc.chain;
    shaderDesc.label = {.data = "compute_shader", .length = 14};
    impl_->shaderModule = wgpuDeviceCreateShaderModule(impl_->device, &shaderDesc);
}

ComputePipeline::~ComputePipeline() {
    delete impl_;
}

void ComputePipeline::setStorageTexture(uint32_t binding, GPUTexture& texture) {
    BindingInfo info{};
    info.type = BindingType::StorageTextureWrite;
    info.textureView = texture.view();
    info.textureFormat = toWGPUFormat(texture.format());
    impl_->bindings[binding] = info;
    impl_->pipelineBuilt = false;
}

void ComputePipeline::setTexture(uint32_t binding, GPUTexture& texture) {
    BindingInfo info{};
    info.type = BindingType::Texture;
    info.textureView = texture.view();
    info.textureFormat = toWGPUFormat(texture.format());
    impl_->bindings[binding] = info;
    impl_->pipelineBuilt = false;
}

void ComputePipeline::setUniformBuffer(uint32_t binding, GPUBuffer& buffer) {
    BindingInfo info{};
    info.type = BindingType::UniformBuffer;
    info.buffer = buffer.buffer();
    info.bufferSize = buffer.size();
    impl_->bindings[binding] = info;
    impl_->pipelineBuilt = false;
}

void ComputePipeline::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    if (!impl_->pipelineBuilt) {
        impl_->buildPipeline();
    }

    // Build bind group from current bindings
    std::vector<WGPUBindGroupEntry> entries;
    for (auto& [binding, info] : impl_->bindings) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        switch (info.type) {
            case BindingType::StorageTextureWrite:
            case BindingType::StorageTextureRead:
            case BindingType::Texture:
                e.textureView = info.textureView;
                break;
            case BindingType::UniformBuffer:
                e.buffer = info.buffer;
                e.offset = 0;
                e.size = info.bufferSize;
                break;
        }
        entries.push_back(e);
    }

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.label = {.data = "compute_bg", .length = 10};
    bgDesc.layout = impl_->bindGroupLayout;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(impl_->device, &bgDesc);

    // Create command encoder and dispatch
    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = {.data = "compute_enc", .length = 11};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(impl_->device, &encDesc);

    WGPUComputePassDescriptor passDesc{};
    passDesc.label = {.data = "compute_pass", .length = 12};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);

    wgpuComputePassEncoderSetPipeline(pass, impl_->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, z);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = {.data = "compute_cmd", .length = 11};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(impl_->queue, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bindGroup);
}
