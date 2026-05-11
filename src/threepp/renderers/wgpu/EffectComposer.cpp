
#include "threepp/renderers/wgpu/EffectComposer.hpp"
#include "ShaderPassImpl.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include "WgpuReadback.hpp"

#include "stb_image_write.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace threepp;

struct EffectComposer::Impl {
    WgpuRenderer& renderer;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUTextureFormat format;

    std::vector<std::shared_ptr<ShaderPass>> passes;

    // Shared bind group layout and pipeline layout for all passes
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUSampler sampler = nullptr;

    // Scene render target (threepp-managed)
    std::unique_ptr<RenderTarget> sceneRT;

    // Ping-pong textures for chaining passes
    WGPUTexture pingTexture = nullptr;
    WGPUTextureView pingView = nullptr;
    WGPUTexture pongTexture = nullptr;
    WGPUTextureView pongView = nullptr;

    // Tracks which texture has the final result
    WGPUTexture finalOutputTexture = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

    explicit Impl(WgpuRenderer& r)
        : renderer(r),
          device(static_cast<WGPUDevice>(r.nativeDevice())),
          queue(static_cast<WGPUQueue>(r.nativeQueue())),
          // Linear format (e.g. BGRA8Unorm). sceneRT is allocated in the linear
          // format and the tonemap pass writes already-sRGB-encoded bytes into
          // it. Using the sRGB format here would apply a second hardware sRGB
          // encode when the ShaderPass writes to ping/pong, double-encoding the
          // pixels and breaking identity composition.
          format(static_cast<WGPUTextureFormat>(r.nativeSurfaceFormatLinear())) {}

    ~Impl() {
        releaseResources();
        if (sampler) wgpuSamplerRelease(sampler);
        if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
        if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
    }

    void ensureSharedResources() {
        if (bindGroupLayout) return;

        // Bind group layout: binding 0 = texture_2d<f32>, binding 1 = sampler
        WGPUBindGroupLayoutEntry entries[2]{};

        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Fragment;
        entries[0].texture.sampleType = WGPUTextureSampleType_Float;
        entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].sampler.type = WGPUSamplerBindingType_Filtering;

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = WGPUStringView{"composer_bgl", WGPU_STRLEN} ;
        bglDesc.entryCount = 2;
        bglDesc.entries = entries;
        bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = WGPUStringView{"composer_pl", WGPU_STRLEN} ;
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bindGroupLayout;
        pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        // Sampler: linear filtering, clamp-to-edge
        WGPUSamplerDescriptor sd{};
        sd.label = WGPUStringView{"composer_sampler", WGPU_STRLEN} ;
        sd.minFilter = WGPUFilterMode_Linear;
        sd.magFilter = WGPUFilterMode_Linear;
        sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        sd.addressModeW = WGPUAddressMode_ClampToEdge;
        sd.lodMaxClamp = 32.0f;
        sd.maxAnisotropy = 1;
        sampler = wgpuDeviceCreateSampler(device, &sd);
    }

    void releaseResources() {
        if (pingView) { wgpuTextureViewRelease(pingView); pingView = nullptr; }
        if (pingTexture) { wgpuTextureRelease(pingTexture); pingTexture = nullptr; }
        if (pongView) { wgpuTextureViewRelease(pongView); pongView = nullptr; }
        if (pongTexture) { wgpuTextureRelease(pongTexture); pongTexture = nullptr; }
        sceneRT.reset();
        finalOutputTexture = nullptr;
        width = 0;
        height = 0;
    }

    WGPUTexture createTexture(uint32_t w, uint32_t h, const char* label, size_t labelLen) {
        WGPUTextureDescriptor td{};
        td.label = WGPUStringView{label, static_cast<size_t>(labelLen)};
        td.size = {w, h, 1};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = format;
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc;
        return wgpuDeviceCreateTexture(device, &td);
    }

    void ensureSize(uint32_t w, uint32_t h) {
        if (width == w && height == h && pingTexture) return;

        releaseResources();
        width = w;
        height = h;

        sceneRT = RenderTarget::create(w, h, RenderTarget::Options{});

        pingTexture = createTexture(w, h, "composer_ping", 13);
        pingView = wgpuTextureCreateView(pingTexture, nullptr);

        pongTexture = createTexture(w, h, "composer_pong", 13);
        pongView = wgpuTextureCreateView(pongTexture, nullptr);
    }

    void render(Object3D& scene, Camera& camera) {
        ensureSharedResources();

        // Determine target size from the renderer's current render target
        auto* userRT = renderer.getRenderTarget();
        uint32_t w, h;
        if (userRT) {
            w = userRT->width;
            h = userRT->height;
        } else {
            auto sz = renderer.size();
            w = static_cast<uint32_t>(sz.width());
            h = static_cast<uint32_t>(sz.height());
        }
        ensureSize(w, h);

        // 1. Scene pass: render into our internal render target
        renderer.setRenderTarget(sceneRT.get());
        renderer.render(scene, camera);

        // Submit scene commands so post passes read a finished texture.
        renderer.flush();

        // Get the scene texture from the render target
        void* nativeTex = renderer.nativeRenderTargetTexture();
        if (!nativeTex) {
            std::cerr << "EffectComposer: failed to get scene render target texture" << std::endl;
            renderer.setRenderTarget(userRT);
            return;
        }
        WGPUTexture sceneTexture = static_cast<WGPUTexture>(nativeTex);

        // Restore user's render target
        renderer.setRenderTarget(userRT);

        // 2. Post-processing passes (ping-pong)
        if (passes.empty()) {
            finalOutputTexture = sceneTexture;
            return;
        }

        for (size_t i = 0; i < passes.size(); i++) {
            auto& pass = passes[i];

            // Input: scene texture for first pass, previous output for subsequent
            WGPUTexture inputTexture = (i == 0) ? sceneTexture : ((i % 2 == 1) ? pingTexture : pongTexture);
            // Output: alternate between ping and pong
            WGPUTexture outputTexture = (i % 2 == 0) ? pingTexture : pongTexture;
            WGPUTextureView outputView = (i % 2 == 0) ? pingView : pongView;

            // Create a view for the input texture
            WGPUTextureView inputView = wgpuTextureCreateView(inputTexture, nullptr);

            // Ensure pipeline is created
            pass->pimpl_->ensurePipeline(device, pipelineLayout, format);

            // Create bind group
            WGPUBindGroupEntry bgEntries[2]{};
            bgEntries[0].binding = 0;
            bgEntries[0].textureView = inputView;
            bgEntries[1].binding = 1;
            bgEntries[1].sampler = sampler;

            WGPUBindGroupDescriptor bgDesc{};
            bgDesc.label = WGPUStringView{"composer_bg", WGPU_STRLEN} ;
            bgDesc.layout = bindGroupLayout;
            bgDesc.entryCount = 2;
            bgDesc.entries = bgEntries;
            WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

            // Create command encoder and render pass
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"composer_enc", WGPU_STRLEN} ;
            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

            WGPURenderPassColorAttachment colorAtt{};
            colorAtt.view = outputView;
            colorAtt.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            colorAtt.loadOp = WGPULoadOp_Clear;
            colorAtt.storeOp = WGPUStoreOp_Store;
            colorAtt.clearValue = {0, 0, 0, 1};

            WGPURenderPassDescriptor rpDesc{};
            rpDesc.label = WGPUStringView{"composer_pass", WGPU_STRLEN} ;
            rpDesc.colorAttachmentCount = 1;
            rpDesc.colorAttachments = &colorAtt;

            WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(encoder, &rpDesc);
            wgpuRenderPassEncoderSetPipeline(rp, pass->pimpl_->pipeline);
            wgpuRenderPassEncoderSetBindGroup(rp, 0, bindGroup, 0, nullptr);
            wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
            wgpuRenderPassEncoderEnd(rp);
            wgpuRenderPassEncoderRelease(rp);

            WGPUCommandBufferDescriptor cmdDesc{};
            cmdDesc.label = WGPUStringView{"composer_cmd", WGPU_STRLEN} ;
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
            wgpuQueueSubmit(queue, 1, &cmd);

            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
            wgpuBindGroupRelease(bindGroup);
            wgpuTextureViewRelease(inputView);

            finalOutputTexture = outputTexture;
        }
    }
};

EffectComposer::EffectComposer(WgpuRenderer& renderer)
    : pimpl_(std::make_unique<Impl>(renderer)) {}

EffectComposer::~EffectComposer() = default;

void EffectComposer::addPass(std::shared_ptr<ShaderPass> pass) {
    pimpl_->passes.push_back(std::move(pass));
}

void EffectComposer::render(Object3D& scene, Camera& camera) {
    pimpl_->render(scene, camera);
}

std::vector<unsigned char> EffectComposer::readRGBPixels() {
    if (!pimpl_->finalOutputTexture || pimpl_->width == 0 || pimpl_->height == 0) return {};
    return wgpu::readRGBPixels(pimpl_->device, pimpl_->queue, pimpl_->finalOutputTexture, pimpl_->width, pimpl_->height);
}

void EffectComposer::writeFramebuffer(const std::filesystem::path& filename) {
    auto ext = filename.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") {
        throw std::runtime_error("EffectComposer::writeFramebuffer: unsupported format " + ext);
    }

    auto pixels = readRGBPixels();
    if (pixels.empty()) return;

    const int w = static_cast<int>(pimpl_->width);
    const int h = static_cast<int>(pimpl_->height);

    if (filename.has_parent_path() && !std::filesystem::exists(filename.parent_path())) {
        std::error_code ec;
        std::filesystem::create_directories(filename.parent_path(), ec);
    }

    bool success = false;
    if (ext == ".png") {
        success = stbi_write_png(filename.string().c_str(), w, h, 3, pixels.data(), w * 3);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(filename.string().c_str(), w, h, 3, pixels.data(), 100);
    } else if (ext == ".bmp") {
        success = stbi_write_bmp(filename.string().c_str(), w, h, 3, pixels.data());
    }
    if (!success) {
        throw std::runtime_error("EffectComposer::writeFramebuffer: failed to write " + filename.string());
    }
}
