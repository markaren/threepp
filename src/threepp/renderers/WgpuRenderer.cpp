
#include "threepp/renderers/WgpuRenderer.hpp"

#include "wgpu/WgpuBindGroupCache.hpp"
#include "wgpu/WgpuBindGroups.hpp"
#include "wgpu/WgpuBufferPool.hpp"
#include "wgpu/WgpuGeometries.hpp"
#include "wgpu/WgpuLights.hpp"
#include "wgpu/WgpuMaterials.hpp"
#include "wgpu/WgpuPipelines.hpp"
#include "wgpu/WgpuReadback.hpp"
#include "wgpu/WgpuRenderTargets.hpp"
#include "wgpu/WgpuShaders.hpp"
#include "wgpu/WgpuShadowMap.hpp"
#include "wgpu/WgpuState.hpp"
#include "wgpu/WgpuTextures.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/constants.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/math/Frustum.hpp"

#include "threepp/renderers/common/Lights.hpp"
#include "threepp/renderers/common/RenderLists.hpp"

#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"

#define GLFW_INCLUDE_NONE
#ifndef __EMSCRIPTEN__
#ifdef __linux__
#define GLFW_EXPOSE_NATIVE_X11
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#ifndef __EMSCRIPTEN__
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

#include <webgpu/webgpu.h>
#ifndef __EMSCRIPTEN__
#include <webgpu/wgpu.h>
#else
#include <emscripten.h>
#endif

// X11 defines a None macro which breaks one of the enums
#undef None

// stb_image_write — implementation is already compiled in GLRenderer.cpp.
// Match the linkage used by the implementation (extern "C" in C++).
#define STBIWDEF extern "C"
#include "stb_image_write.h"
#undef STBIWDEF

#include "external/glfw/include/GLFW/glfw3.h"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace threepp;
namespace SF = threepp::wgpu::ShaderFeatures;

#ifdef __APPLE__
extern "C" void* wgpu_create_metal_layer(void* nsWindow);
#endif

namespace {

    // Maximum time to wait for async WebGPU operations before aborting.
    constexpr auto WGPU_ASYNC_TIMEOUT = std::chrono::seconds(10);

}// namespace


struct WgpuRenderer::Impl {

    // Listener for material dispose events — evicts stale pipeline cache entries.
    struct OnMaterialDispose : EventListener {
        Impl* scope_;
        explicit OnMaterialDispose(Impl* s) : scope_(s) {}
        void onEvent(Event& event) override {
            auto* mat = std::any_cast<Material*>(event.target);
            mat->removeEventListener("dispose", *this);
            scope_->trackedMaterials_.erase(mat);
            if (scope_->pipelines) {
                scope_->pipelines->onMaterialDispose(mat->id);
            }
        }
    };

    // Persistent per-material uniform buffer cache.
    struct PerMaterialEntry {
        WGPUBuffer materialBuf = nullptr;
        float materialData[wgpu::MATERIAL_UNIFORM_SIZE / sizeof(float)] = {};
    };
    std::unordered_map<int, PerMaterialEntry> perMaterialCache_;
    std::unordered_set<Material*> trackedMaterialsForUniforms_;

    struct OnMaterialDisposeForUniforms : EventListener {
        Impl* scope_;
        explicit OnMaterialDisposeForUniforms(Impl* s) : scope_(s) {}
        void onEvent(Event& event) override {
            auto* mat = std::any_cast<Material*>(event.target);
            mat->removeEventListener("dispose", *this);
            scope_->trackedMaterialsForUniforms_.erase(mat);
            auto it = scope_->perMaterialCache_.find(mat->id);
            if (it != scope_->perMaterialCache_.end()) {
                if (it->second.materialBuf) wgpuBufferRelease(it->second.materialBuf);
                scope_->perMaterialCache_.erase(it);
            }
        }
    } onMaterialDisposeForUniforms{this};

    // Persistent GPU storage buffer cache for InstancedMesh data.
    // Only re-uploaded when instanceMatrix/instanceColor version changes.
    struct InstanceEntry {
        WGPUBuffer buffer = nullptr;
        size_t bufferSize = 0;
        uint32_t matrixVersion = 0;
        uint32_t colorVersion = 0;
        bool hasColor = false;
    };
    std::unordered_map<int, InstanceEntry> instanceCache_;
    std::unordered_set<InstancedMesh*> trackedInstancedMeshes_;

    struct OnInstancedMeshDispose : EventListener {
        Impl* scope_;
        explicit OnInstancedMeshDispose(Impl* s) : scope_(s) {}
        void onEvent(Event& event) override {
            auto* mesh = std::any_cast<InstancedMesh*>(event.target);
            mesh->removeEventListener("dispose", *this);
            scope_->trackedInstancedMeshes_.erase(mesh);
            auto it = scope_->instanceCache_.find(mesh->id);
            if (it != scope_->instanceCache_.end()) {
                if (it->second.buffer) wgpuBufferRelease(it->second.buffer);
                scope_->instanceCache_.erase(it);
            }
        }
    } onInstancedMeshDispose{this};

    WgpuRenderer& scope;
    Canvas& canvas;

    // Core WebGPU objects
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUSurface surface = nullptr;

    WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;

    WindowSize size_;
    float pixelRatio_ = 1.0f;
    float uniformPad_ = 1.0f;  // written to transformData[63] (_pad), used by path tracer display shader
    Color clearColor_{0x000000};
    float clearAlpha_ = 1.0f;

    // MSAA configuration
    uint32_t sampleCount_ = 1;
    // Effective sample count for the current render pass (may be 1 when
    // rendering to the non-MSAA tone mapping intermediate RT).
    uint32_t effectiveSampleCount_ = 1;

    // Single command encoder for the entire frame.  All render passes recorded
    // within one animation frame — including nested mirror renders, the tone-map
    // blit, and the ImGui overlay — share this encoder.  wgpu-native then inserts
    // correct Vulkan image-layout transitions between passes automatically.
    // Created lazily on the first render()/executeClear() call; submitted and
    // nulled out in endFrame().
    WGPUCommandEncoder renderEncoder_ = nullptr;

    // Per-render light buffer — acquired from the pool at the start of each render()
    // call so that each render pass in the same frame gets its own light data snapshot.
    // This prevents a later render() (e.g. the HUD with no lights) from overwriting the
    // persistent lightBuffer_ before the earlier render pass's GPU work runs.
    WGPUBuffer renderLightBuffer_ = nullptr;

    // Viewport & scissor state
    struct { float x=0, y=0, w=0, h=0; } viewport_;
    struct { uint32_t x=0, y=0, w=0, h=0; } scissor_;
    bool scissorTest_ = false;
    bool viewportExplicit_ = false; // true if setViewport() called after setRenderTarget()

    // Subsystem: shared state
    wgpu::WgpuState wgpuState;

    // Subsystem: geometry buffer management (with version-based updates)
    std::unique_ptr<wgpu::WgpuGeometries> geometries;

    // Subsystem: texture upload, caching, version tracking
    std::unique_ptr<wgpu::WgpuTextures> textures;

    // Subsystem: pipeline creation and caching
    std::unique_ptr<wgpu::WgpuPipelines> pipelines;

    // Subsystem: shadow map rendering
    std::unique_ptr<wgpu::WgpuShadowMap> shadowMap;

    // Subsystem: per-frame buffer pool (avoids per-draw GPU buffer alloc/free)
    std::unique_ptr<wgpu::WgpuBufferPool> bufferPool;

    // Subsystem: light collection and GPU buffer packing
    std::unique_ptr<wgpu::WgpuLights> lights;

    // Subsystem: render target texture cache
    std::unique_ptr<wgpu::WgpuRenderTargets> renderTargets;

    // Subsystem: bind group assembly (hot path, reuses internal vector)
    std::unique_ptr<wgpu::WgpuBindGroups> bindGroups;

    // Subsystem: bind group cache — avoids per-draw wgpuDeviceCreateBindGroup()
    std::unique_ptr<wgpu::WgpuBindGroupCache> bindGroupCache;

    // Material dispose listener and set of materials with the listener registered
    OnMaterialDispose onMaterialDispose;
    std::unordered_set<Material*> trackedMaterials_;

    std::function<void(void*)> overlayCallback_;

    RenderTarget* currentRenderTarget_ = nullptr;

    // Render target state
    int activeCubeFace_ = 0;
    int activeMipmapLevel_ = 0;

    // Frustum culling
    Frustum frustum_;
    Vector3 _vector3;

    // Render list for opaque/transparent sorting
    RenderList renderList_;

    // Render info/statistics
    struct {
        size_t frame = 0;
        size_t calls = 0;
        size_t triangles = 0;
        size_t lines = 0;
        size_t points = 0;
        size_t geometries = 0;
        size_t textures = 0;
    } renderInfo;

    bool initialized = false;

    // Per-frame surface state — persists across multiple render() calls within
    // a single frame so that HUD / multi-pass rendering works correctly.
    // Acquired on the first render() of a frame, released by endFrame().
    struct FrameSurface {

        WGPUSurfaceTexture surfaceTexture{};
        WGPUTextureView colorView = nullptr;
        WGPUTextureView depthView = nullptr;    // points into cachedFrame_
        WGPUTextureView resolveView = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        bool active = false;      // true while a surface texture is acquired
        bool hasRendered = false;  // true after the first render pass of this frame
    } frame_;

    // Persistent depth + MSAA textures — recreated only on resize or sample-count change.
    // Avoids allocating/freeing large GPU textures every frame.
    struct CachedFrameTex {
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUTexture msaaColorTexture = nullptr;
        WGPUTextureView msaaColorView = nullptr;
        uint32_t width = 0, height = 0;
        uint32_t sampleCount = 0;
    } cachedFrame_;

    void ensureFrameTextures(uint32_t w, uint32_t h, uint32_t sc) {
        if (cachedFrame_.width == w && cachedFrame_.height == h && cachedFrame_.sampleCount == sc)
            return;

        // Release old resources
        if (cachedFrame_.depthView)       { wgpuTextureViewRelease(cachedFrame_.depthView);      cachedFrame_.depthView = nullptr; }
        if (cachedFrame_.depthTexture)    { wgpuTextureRelease(cachedFrame_.depthTexture);        cachedFrame_.depthTexture = nullptr; }
        if (cachedFrame_.msaaColorView)   { wgpuTextureViewRelease(cachedFrame_.msaaColorView);   cachedFrame_.msaaColorView = nullptr; }
        if (cachedFrame_.msaaColorTexture){ wgpuTextureRelease(cachedFrame_.msaaColorTexture);    cachedFrame_.msaaColorTexture = nullptr; }

        cachedFrame_.width = w;
        cachedFrame_.height = h;
        cachedFrame_.sampleCount = sc;

        WGPUTextureDescriptor dtd{};
        dtd.label = WGPUStringView{"depth_tex", sizeof("depth_tex") - 1};
        dtd.size = {w, h, 1};
        dtd.mipLevelCount = 1;
        dtd.sampleCount = sc;
        dtd.dimension = WGPUTextureDimension_2D;
        dtd.format = WGPUTextureFormat_Depth24Plus;
        dtd.usage = WGPUTextureUsage_RenderAttachment;
        cachedFrame_.depthTexture = wgpuDeviceCreateTexture(device, &dtd);
        cachedFrame_.depthView    = wgpuTextureCreateView(cachedFrame_.depthTexture, nullptr);

        if (sc > 1) {
            WGPUTextureDescriptor msaaCtd{};
            msaaCtd.label = WGPUStringView{"frame_msaa_color", sizeof("frame_msaa_color") - 1};
            msaaCtd.size = {w, h, 1};
            msaaCtd.mipLevelCount = 1;
            msaaCtd.sampleCount = sc;
            msaaCtd.dimension = WGPUTextureDimension_2D;
            msaaCtd.format = surfaceFormat;
            msaaCtd.usage = WGPUTextureUsage_RenderAttachment;
            cachedFrame_.msaaColorTexture = wgpuDeviceCreateTexture(device, &msaaCtd);
            cachedFrame_.msaaColorView    = wgpuTextureCreateView(cachedFrame_.msaaColorTexture, nullptr);
        }
    }

    void releaseCachedFrameTextures() {
        if (cachedFrame_.depthView)       { wgpuTextureViewRelease(cachedFrame_.depthView);      cachedFrame_.depthView = nullptr; }
        if (cachedFrame_.depthTexture)    { wgpuTextureRelease(cachedFrame_.depthTexture);        cachedFrame_.depthTexture = nullptr; }
        if (cachedFrame_.msaaColorView)   { wgpuTextureViewRelease(cachedFrame_.msaaColorView);   cachedFrame_.msaaColorView = nullptr; }
        if (cachedFrame_.msaaColorTexture){ wgpuTextureRelease(cachedFrame_.msaaColorTexture);    cachedFrame_.msaaColorTexture = nullptr; }
        cachedFrame_ = {};
    }

    // Pending clear flags — set by clear()/clearDepth()/clearStencil(),
    // consumed by the next render pass's loadOp.
    bool pendingClearColor_ = false;
    bool pendingClearDepth_ = false;
    bool pendingClearStencil_ = false;

    // Internal tone mapping post-process state.
    // When tone mapping or sRGB output is active, scene renders go to this
    // intermediate RT instead of the surface. endFrame() blits to the surface
    // with tone mapping + sRGB conversion applied via a fullscreen pass.
    struct ToneMapState {
        WGPUTexture colorTexture = nullptr;
        WGPUTextureView colorView = nullptr;
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthView = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;

        WGPUBindGroupLayout bindGroupLayout = nullptr;
        WGPUPipelineLayout pipelineLayout = nullptr;
        WGPUSampler sampler = nullptr;
        WGPUBuffer uniformBuf = nullptr;  // persistent 16-byte uniform buffer for tone-map params

        // Cached pipelines keyed by tone mapping mode
        // Index: 0=Linear, 1=Reinhard, 2=Cineon, 3=ACES, 4=None(sRGB only)
        WGPURenderPipeline pipelines[5]{};
        WGPUShaderModule shaderModules[5]{};

        // Per-blit cached objects — valid as long as colorTexture doesn't change (no resize)
        WGPUTextureView cachedInputView = nullptr;
        WGPUBindGroup  cachedBindGroup  = nullptr;
        float          lastExposure     = -1.f;

        bool initialized = false;
    } toneMap_;

    // Returns true if a post-process tone mapping/sRGB pass is needed.
    // Also returns true when retainFramebuffer is active: routing through the
    // intermediate color texture gives us a COPY_SRC source for the retain copy,
    // since surface textures only carry RENDER_ATTACHMENT usage.
    bool hasTransmissive_ = false;
    Texture* currentSceneEnv_ = nullptr;  // scene.environment for the current render call

    bool needsToneMapPass() const {
        return scope.toneMapping != ToneMapping::None ||
               scope.outputEncoding == Encoding::sRGB ||
               retainFramebuffer ||
               hasTransmissive_;
    }

    // Retained framebuffer for copyFramebufferToTexture support.
    // When retainFramebuffer is true, render() blits the resolved scene content
    // into retainedFB (Rgba8Unorm) so it's available for later copy operations.
    // Rgba8Unorm matches the format of DataTexture destinations.
    bool retainFramebuffer = false;
    WGPUTexture retainedFB = nullptr;
    uint32_t retainedFBWidth = 0;
    uint32_t retainedFBHeight = 0;

    // Blit pipeline used by the retain block: converts surfaceFormat (Bgra8Unorm
    // on Windows) to Rgba8Unorm via a shader pass (CopyTextureToTexture cannot
    // bridge these formats as they are not copy-compatible in WebGPU).
    struct RetainBlitState {
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout pipelineLayout = nullptr;
        WGPUBindGroupLayout bgl = nullptr;
        WGPUSampler sampler = nullptr;
        WGPUShaderModule shader = nullptr;
    } retainBlit_;

    void ensureRetainBlit() {
        if (retainBlit_.pipeline) return;

        const char* wgsl = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    var p = array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    var o: V;
    o.pos = vec4<f32>(p[vi], 0., 1.);
    o.uv = p[vi] * 0.5 + vec2<f32>(0.5);
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
        retainBlit_.shader = wgpuDeviceCreateShaderModule(device, &smDesc);

        WGPUBindGroupLayoutEntry bglEntries[2]{};
        bglEntries[0].binding = 0;
        bglEntries[0].visibility = WGPUShaderStage_Fragment;
        bglEntries[0].texture.sampleType = WGPUTextureSampleType_Float;
        bglEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
        bglEntries[1].binding = 1;
        bglEntries[1].visibility = WGPUShaderStage_Fragment;
        bglEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;
        WGPUBindGroupLayoutDescriptor bgld{};
        bgld.entryCount = 2;
        bgld.entries = bglEntries;
        retainBlit_.bgl = wgpuDeviceCreateBindGroupLayout(device, &bgld);

        WGPUPipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &retainBlit_.bgl;
        retainBlit_.pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &pld);

        WGPUSamplerDescriptor sd{};
        sd.magFilter = WGPUFilterMode_Nearest;
        sd.minFilter = WGPUFilterMode_Nearest;
        sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        sd.maxAnisotropy = 1;
        retainBlit_.sampler = wgpuDeviceCreateSampler(device, &sd);

        WGPUVertexState vs{};
        vs.module = retainBlit_.shader;
        vs.entryPoint = {.data = "vs", .length = 2};

        WGPUColorTargetState ct{};
        ct.format = WGPUTextureFormat_RGBA8Unorm;
        ct.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fs{};
        fs.module = retainBlit_.shader;
        fs.entryPoint = {.data = "fs", .length = 2};
        fs.targetCount = 1;
        fs.targets = &ct;

        WGPURenderPipelineDescriptor rpd{};
        rpd.label = WGPUStringView{"retain_blit_pipe", WGPU_STRLEN};
        rpd.layout = retainBlit_.pipelineLayout;
        rpd.vertex = vs;
        rpd.fragment = &fs;
        rpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        rpd.multisample.count = 1;
        rpd.multisample.mask = 0xFFFFFFFF;
        retainBlit_.pipeline = wgpuDeviceCreateRenderPipeline(device, &rpd);
    }

    // Scissored background clear pipeline (used when scissorTest_ is enabled and autoClear)
    WGPURenderPipeline clearPipeline_ = nullptr;
    WGPUPipelineLayout clearPipelineLayout_ = nullptr;
    WGPUBindGroupLayout clearBGL_ = nullptr;
    WGPUShaderModule clearShader_ = nullptr;
    WGPUTextureFormat clearPipelineFormat_ = WGPUTextureFormat_Undefined;
    uint32_t clearPipelineSampleCount_ = 0;

    void initClearPipeline(WGPUTextureFormat colorFormat, uint32_t sampleCount) {
        if (clearPipeline_ && clearPipelineFormat_ == colorFormat && clearPipelineSampleCount_ == sampleCount) return;
        if (clearPipeline_) { wgpuRenderPipelineRelease(clearPipeline_); clearPipeline_ = nullptr; }
        if (clearPipelineLayout_) { wgpuPipelineLayoutRelease(clearPipelineLayout_); clearPipelineLayout_ = nullptr; }
        if (clearBGL_) { wgpuBindGroupLayoutRelease(clearBGL_); clearBGL_ = nullptr; }
        if (clearShader_) { wgpuShaderModuleRelease(clearShader_); clearShader_ = nullptr; }

        const char* wgsl = R"(
struct ClearColor { color: vec4<f32> }
@group(0) @binding(0) var<uniform> u: ClearColor;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    return vec4<f32>(pos[vi], 1.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return u.color; }
)";
        WGPUShaderSourceWGSL src{};
        src.chain.sType = WGPUSType_ShaderSourceWGSL;
        src.code = {.data = wgsl, .length = static_cast<size_t>(strlen(wgsl))};
        WGPUShaderModuleDescriptor smd{};
        smd.nextInChain = &src.chain;
        clearShader_ = wgpuDeviceCreateShaderModule(device, &smd);

        WGPUBindGroupLayoutEntry bgle{};
        bgle.binding = 0;
        bgle.visibility = WGPUShaderStage_Fragment;
        bgle.buffer.type = WGPUBufferBindingType_Uniform;
        bgle.buffer.minBindingSize = 16;
        WGPUBindGroupLayoutDescriptor bgld{};
        bgld.entryCount = 1; bgld.entries = &bgle;
        clearBGL_ = wgpuDeviceCreateBindGroupLayout(device, &bgld);

        WGPUPipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &clearBGL_;
        clearPipelineLayout_ = wgpuDeviceCreatePipelineLayout(device, &pld);

        WGPUColorTargetState ct{};
        ct.format = colorFormat;
        ct.writeMask = WGPUColorWriteMask_All;
        auto fsEntry = WGPUStringView{"fs_main", sizeof("fs_main") - 1};
        WGPUFragmentState fs{};
        fs.module = clearShader_; fs.entryPoint = fsEntry; fs.targetCount = 1; fs.targets = &ct;

        WGPUDepthStencilState ds{};
        ds.format = WGPUTextureFormat_Depth24Plus;
        ds.depthWriteEnabled = WGPUOptionalBool_True;
        ds.depthCompare = WGPUCompareFunction_Always;

        WGPURenderPipelineDescriptor pd{};
        pd.label = WGPUStringView{"scissor_clear_pipe", WGPU_STRLEN};
        auto vsEntry = WGPUStringView{"vs_main", sizeof("vs_main") - 1};
        pd.layout = clearPipelineLayout_;
        pd.vertex.module = clearShader_; pd.vertex.entryPoint = vsEntry;
        pd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pd.depthStencil = &ds;
        pd.multisample.count = sampleCount; pd.multisample.mask = 0xFFFFFFFF;
        pd.fragment = &fs;
        clearPipeline_ = wgpuDeviceCreateRenderPipeline(device, &pd);
        clearPipelineFormat_ = colorFormat;
        clearPipelineSampleCount_ = sampleCount;
    }

    void drawScissoredClear(WGPURenderPassEncoder pass, Color color, float alpha,
                            WGPUTextureFormat colorFormat, uint32_t sampleCount) {
        initClearPipeline(colorFormat, sampleCount);
        if (!clearPipeline_) return;

        float rgba[4] = {color.r, color.g, color.b, alpha};
        constexpr auto kUsage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer colorBuf = bufferPool->acquire(16, kUsage, rgba);

        WGPUBindGroupEntry bge{};
        bge.binding = 0; bge.buffer = colorBuf; bge.size = 16;
        WGPUBindGroupDescriptor bgd{};
        bgd.layout = clearBGL_; bgd.entryCount = 1; bgd.entries = &bge;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgd);

        wgpuRenderPassEncoderSetPipeline(pass, clearPipeline_);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuBindGroupRelease(bg);
    }

    // ── Environment-map background pipeline ──────────────────────────
    WGPURenderPipeline envBgPipeline_ = nullptr;
    WGPUPipelineLayout envBgPipelineLayout_ = nullptr;
    WGPUBindGroupLayout envBgBGL_ = nullptr;
    WGPUShaderModule envBgShader_ = nullptr;
    WGPUTextureFormat envBgPipelineFormat_ = WGPUTextureFormat_Undefined;
    uint32_t envBgPipelineSampleCount_ = 0;

    void initEnvBgPipeline(WGPUTextureFormat colorFormat, uint32_t sampleCount) {
        if (envBgPipeline_ && envBgPipelineFormat_ == colorFormat && envBgPipelineSampleCount_ == sampleCount) return;
        if (envBgPipeline_) { wgpuRenderPipelineRelease(envBgPipeline_); envBgPipeline_ = nullptr; }
        if (envBgPipelineLayout_) { wgpuPipelineLayoutRelease(envBgPipelineLayout_); envBgPipelineLayout_ = nullptr; }
        if (envBgBGL_) { wgpuBindGroupLayoutRelease(envBgBGL_); envBgBGL_ = nullptr; }
        if (envBgShader_) { wgpuShaderModuleRelease(envBgShader_); envBgShader_ = nullptr; }

        const char* wgsl = R"(
const PI = 3.141592653589793;
struct EnvBgUniforms { invVP: mat4x4<f32> }
@group(0) @binding(0) var<uniform> u: EnvBgUniforms;
@group(0) @binding(1) var envTex: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> }

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>,3>(vec2<f32>(-1.,-1.),vec2<f32>(3.,-1.),vec2<f32>(-1.,3.));
    var o: VsOut;
    o.pos = vec4<f32>(p[vi], 1.0, 1.0);
    o.ndc = p[vi];
    return o;
}
@fragment fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    let clip = vec4<f32>(v.ndc, 1.0, 1.0);
    let world = u.invVP * clip;
    let dir = normalize(world.xyz / world.w);
    let phi = atan2(dir.z, dir.x);
    let theta = asin(clamp(dir.y, -1.0, 1.0));
    let uv = vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 - theta / PI);
    let sz = vec2<f32>(textureDimensions(envTex, 0));
    let px = vec2<i32>(i32(uv.x * sz.x) % i32(sz.x),
                       clamp(i32(uv.y * sz.y), 0, i32(sz.y) - 1));
    return vec4<f32>(textureLoad(envTex, px, 0).rgb, 1.0);
}
)";
        WGPUShaderSourceWGSL src{};
        src.chain.sType = WGPUSType_ShaderSourceWGSL;
        src.code = {.data = wgsl, .length = static_cast<size_t>(strlen(wgsl))};
        WGPUShaderModuleDescriptor smd{};
        smd.nextInChain = &src.chain;
        envBgShader_ = wgpuDeviceCreateShaderModule(device, &smd);

        // Bind group layout: uniform buffer + texture
        WGPUBindGroupLayoutEntry entries[2]{};
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = 64;
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
        WGPUBindGroupLayoutDescriptor bgld{};
        bgld.entryCount = 2; bgld.entries = entries;
        envBgBGL_ = wgpuDeviceCreateBindGroupLayout(device, &bgld);

        WGPUPipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &envBgBGL_;
        envBgPipelineLayout_ = wgpuDeviceCreatePipelineLayout(device, &pld);

        WGPUColorTargetState ct{};
        ct.format = colorFormat;
        ct.writeMask = WGPUColorWriteMask_All;
        auto fsEntry = WGPUStringView{"fs_main", sizeof("fs_main") - 1};
        WGPUFragmentState fs{};
        fs.module = envBgShader_; fs.entryPoint = fsEntry; fs.targetCount = 1; fs.targets = &ct;

        WGPUDepthStencilState ds{};
        ds.format = WGPUTextureFormat_Depth24Plus;
        ds.depthWriteEnabled = WGPUOptionalBool_True;
        ds.depthCompare = WGPUCompareFunction_Always;

        WGPURenderPipelineDescriptor pd{};
        pd.label = WGPUStringView{"env_bg_pipe", WGPU_STRLEN};
        auto vsEntry = WGPUStringView{"vs_main", sizeof("vs_main") - 1};
        pd.layout = envBgPipelineLayout_;
        pd.vertex.module = envBgShader_; pd.vertex.entryPoint = vsEntry;
        pd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pd.depthStencil = &ds;
        pd.multisample.count = sampleCount; pd.multisample.mask = 0xFFFFFFFF;
        pd.fragment = &fs;
        envBgPipeline_ = wgpuDeviceCreateRenderPipeline(device, &pd);
        envBgPipelineFormat_ = colorFormat;
        envBgPipelineSampleCount_ = sampleCount;
    }

    // Cached env background GPU texture (handles both HDR float and LDR byte data)
    WGPUTexture envBgTex_ = nullptr;
    WGPUTextureView envBgTexView_ = nullptr;
    unsigned int envBgTexId_ = 0;
    unsigned int envBgTexVer_ = 0;

    void ensureEnvBgTexture(Texture* envTex) {
        if (envBgTexView_ && envBgTexId_ == envTex->id && envBgTexVer_ == envTex->version()) return;

        // Release old
        if (envBgTexView_) { wgpuTextureViewRelease(envBgTexView_); envBgTexView_ = nullptr; }
        if (envBgTex_) { wgpuTextureRelease(envBgTex_); envBgTex_ = nullptr; }

        auto& img = envTex->image();
        uint32_t w = img.width, h = img.height;
        if (w == 0 || h == 0) return;

        const void* srcData = nullptr;
        size_t srcSize = 0;
        std::vector<float> rgbaHDR;
        std::vector<unsigned char> rgbaLDR;
        WGPUTextureFormat fmt;
        bool isHDR = false;

        // Try float data first (HDR), then unsigned char (LDR)
        try {
            auto& fdata = img.data<float>();
            if (fdata.empty()) return;
            isHDR = true;
            size_t pixels = static_cast<size_t>(w) * h;
            if (fdata.size() == pixels * 4) {
                srcData = fdata.data();
                srcSize = fdata.size() * sizeof(float);
            } else if (fdata.size() == pixels * 3) {
                rgbaHDR.resize(pixels * 4);
                for (size_t i = 0; i < pixels; i++) {
                    rgbaHDR[i*4+0] = fdata[i*3+0];
                    rgbaHDR[i*4+1] = fdata[i*3+1];
                    rgbaHDR[i*4+2] = fdata[i*3+2];
                    rgbaHDR[i*4+3] = 1.0f;
                }
                srcData = rgbaHDR.data();
                srcSize = rgbaHDR.size() * sizeof(float);
            } else return;
            fmt = WGPUTextureFormat_RGBA32Float;
        } catch (...) {
            auto& bdata = img.data<unsigned char>();
            if (bdata.empty()) return;
            size_t pixels = static_cast<size_t>(w) * h;
            int channels = (bdata.size() == pixels * 4) ? 4 : (bdata.size() == pixels * 3) ? 3 : 0;
            if (!channels) return;
            // Convert LDR to float RGBA so the pipeline format stays RGBA32Float
            rgbaHDR.resize(pixels * 4);
            for (size_t i = 0; i < pixels; i++) {
                rgbaHDR[i*4+0] = static_cast<float>(bdata[i*channels+0]) / 255.0f;
                rgbaHDR[i*4+1] = static_cast<float>(bdata[i*channels+1]) / 255.0f;
                rgbaHDR[i*4+2] = static_cast<float>(bdata[i*channels+2]) / 255.0f;
                rgbaHDR[i*4+3] = 1.0f;
            }
            srcData = rgbaHDR.data();
            srcSize = rgbaHDR.size() * sizeof(float);
            fmt = WGPUTextureFormat_RGBA32Float;
        }

        WGPUTextureDescriptor td{};
        td.label = WGPUStringView{"env_bg_tex", WGPU_STRLEN};
        td.size = {w, h, 1};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = fmt;
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        envBgTex_ = wgpuDeviceCreateTexture(device, &td);
        envBgTexView_ = wgpuTextureCreateView(envBgTex_, nullptr);

        WGPUTexelCopyTextureInfo dst{};
        dst.texture = envBgTex_;
        WGPUTexelCopyBufferLayout layout{};
        uint32_t bytesPerPixel = isHDR ? 16 : 4;
        layout.bytesPerRow = w * bytesPerPixel;
        layout.rowsPerImage = h;
        WGPUExtent3D extent = {w, h, 1};
        wgpuQueueWriteTexture(queue, &dst, srcData, srcSize, &layout, &extent);

        envBgTexId_ = envTex->id;
        envBgTexVer_ = envTex->version();
    }

    void drawEnvBackground(WGPURenderPassEncoder pass, const Matrix4& invVP,
                           Texture* envTex, WGPUTextureFormat colorFormat, uint32_t sampleCount) {
        initEnvBgPipeline(colorFormat, sampleCount);
        if (!envBgPipeline_) return;

        ensureEnvBgTexture(envTex);
        if (!envBgTexView_) return;

        constexpr auto kUsage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer ubuf = bufferPool->acquire(64, kUsage, invVP.elements.data());

        WGPUBindGroupEntry bge[2]{};
        bge[0].binding = 0; bge[0].buffer = ubuf; bge[0].size = 64;
        bge[1].binding = 1; bge[1].textureView = envBgTexView_;
        WGPUBindGroupDescriptor bgd{};
        bgd.layout = envBgBGL_; bgd.entryCount = 2; bgd.entries = bge;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgd);

        wgpuRenderPassEncoderSetPipeline(pass, envBgPipeline_);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuBindGroupRelease(bg);
    }

    // ── Transmission render target ──────────────────────────────────
    // A mipmapped copy of the framebuffer after opaques, used by transmissive shaders.
    WGPUTexture transmissionTex_ = nullptr;
    WGPUTextureView transmissionTexView_ = nullptr;
    WGPUSampler transmissionSampler_ = nullptr;
    uint32_t transmissionTexW_ = 0, transmissionTexH_ = 0;

    WGPUTextureFormat transmissionTexFmt_ = WGPUTextureFormat_Undefined;

    void ensureTransmissionRT(uint32_t w, uint32_t h, WGPUTextureFormat fmt = WGPUTextureFormat_RGBA8Unorm) {
        if (transmissionTex_ && transmissionTexW_ == w && transmissionTexH_ == h && transmissionTexFmt_ == fmt) return;
        if (transmissionTexView_) { wgpuTextureViewRelease(transmissionTexView_); transmissionTexView_ = nullptr; }
        if (transmissionTex_) { wgpuTextureRelease(transmissionTex_); transmissionTex_ = nullptr; }
        if (transmissionSampler_) { wgpuSamplerRelease(transmissionSampler_); transmissionSampler_ = nullptr; }

        uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>((std::max)(w, h))))) + 1u;

        WGPUTextureDescriptor td{};
        td.label = WGPUStringView{"transmission_rt", WGPU_STRLEN};
        td.size = {w, h, 1};
        td.mipLevelCount = mipLevels;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = fmt;
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment;
        transmissionTex_ = wgpuDeviceCreateTexture(device, &td);
        transmissionTexView_ = wgpuTextureCreateView(transmissionTex_, nullptr);

        WGPUSamplerDescriptor sd{};
        sd.label = WGPUStringView{"transmission_samp", WGPU_STRLEN};
        sd.magFilter = WGPUFilterMode_Linear;
        sd.minFilter = WGPUFilterMode_Linear;
        sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        sd.maxAnisotropy = 1;
        transmissionSampler_ = wgpuDeviceCreateSampler(device, &sd);

        transmissionTexW_ = w;
        transmissionTexH_ = h;
        transmissionTexFmt_ = fmt;
    }

    // Shared quad geometry for rendering sprites (standard attributes, not interleaved)
    std::shared_ptr<BufferGeometry> spriteGeometry_;

    BufferGeometry* getSpriteGeometry() {
        if (!spriteGeometry_) {
            spriteGeometry_ = BufferGeometry::create();
            std::vector<float> positions = {
                -0.5f, -0.5f, 0.f,
                 0.5f, -0.5f, 0.f,
                 0.5f,  0.5f, 0.f,
                -0.5f,  0.5f, 0.f};
            std::vector<float> normals = {
                0.f, 0.f, 1.f,
                0.f, 0.f, 1.f,
                0.f, 0.f, 1.f,
                0.f, 0.f, 1.f};
            std::vector<float> uvs = {
                0.f, 0.f,
                1.f, 0.f,
                1.f, 1.f,
                0.f, 1.f};
            spriteGeometry_->setAttribute("position", FloatBufferAttribute::create(positions, 3));
            spriteGeometry_->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
            spriteGeometry_->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
            spriteGeometry_->setIndex(std::vector<int>{0, 1, 2, 0, 2, 3});
        }
        return spriteGeometry_.get();
    }

    explicit Impl(WgpuRenderer& scope, Canvas& canvas)
        : scope(scope), canvas(canvas), size_(canvas.size()),
          onMaterialDispose(this) {

        viewport_.w = static_cast<float>(size_.width());
        viewport_.h = static_cast<float>(size_.height());
        scissor_.w = static_cast<uint32_t>(size_.width());
        scissor_.h = static_cast<uint32_t>(size_.height());

        initWebGPU();
    }

    void initWebGPU() {
        // Create instance. On native: use wgpu-native extras to select
        // primary backends (Vulkan/Metal/DX12) and avoid the GL backend
        // which conflicts with GLFW's GL context.
        // On Emscripten: the browser provides WebGPU natively; no extras needed.
#ifndef __EMSCRIPTEN__
        WGPUInstanceExtras instanceExtras{};
        instanceExtras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
        instanceExtras.chain.next = nullptr;
        instanceExtras.backends = WGPUInstanceBackend_Primary;

        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = &instanceExtras.chain;
#else
        WGPUInstanceDescriptor instanceDesc{};
#endif
        instance = wgpuCreateInstance(&instanceDesc);
        if (!instance) {
            std::cerr << "WgpuRenderer: Failed to create WebGPU instance" << std::endl;
            return;
        }

        createSurface();
        requestAdapter();
        if (!adapter) {
            std::cerr << "WgpuRenderer: Failed to get adapter" << std::endl;
            return;
        }

        requestDevice();
        if (!device) {
            std::cerr << "WgpuRenderer: Failed to get device" << std::endl;
            return;
        }

        queue = wgpuDeviceGetQueue(device);

        if (surface) {
            // Query preferred surface format from capabilities
            WGPUSurfaceCapabilities caps{};
            wgpuSurfaceGetCapabilities(surface, adapter, &caps);
            if (caps.formatCount > 0 && caps.formats) {
                surfaceFormat = caps.formats[0];
            }
            wgpuSurfaceCapabilitiesFreeMembers(caps);

            configureSurface();
        }

        // Populate shared state for subsystems
        wgpuState.device = device;
        wgpuState.queue = queue;
        wgpuState.surfaceFormat = surfaceFormat;

        // Inherit MSAA from Canvas parameters (WebGPU only supports 1 or 4).
        {
            const int s = canvas.samples();
            sampleCount_ = (s >= 4) ? 4u : 1u;
        }

        // Initialize subsystems
        textures = std::make_unique<wgpu::WgpuTextures>(wgpuState);
        textures->createDummyTexture();
        geometries = std::make_unique<wgpu::WgpuGeometries>(wgpuState);
        pipelines = std::make_unique<wgpu::WgpuPipelines>(wgpuState);
        shadowMap = std::make_unique<wgpu::WgpuShadowMap>(wgpuState, *geometries);
        bufferPool = std::make_unique<wgpu::WgpuBufferPool>(device, queue);
        lights = std::make_unique<wgpu::WgpuLights>(wgpuState);
        renderTargets = std::make_unique<wgpu::WgpuRenderTargets>(wgpuState);
        bindGroups = std::make_unique<wgpu::WgpuBindGroups>();
        bindGroupCache = std::make_unique<wgpu::WgpuBindGroupCache>(device);

        initialized = true;
        std::cout << "WgpuRenderer: WebGPU initialized successfully"
                  << (surface ? "" : " (headless, no surface)") << std::endl;
    }

    void createSurface() {
        WGPUSurfaceDescriptor surfDesc{};

#if defined(__EMSCRIPTEN__)
        // Emscripten: attach to the HTML canvas element.
        WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc{};
        canvasDesc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
        canvasDesc.chain.next = nullptr;
        canvasDesc.selector = WGPUStringView{"#canvas", 7};
        surfDesc.nextInChain = &canvasDesc.chain;

#elif defined(__linux__)
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        surfDesc.label = WGPUStringView{"threepp_surface", sizeof("threepp_surface") - 1};

        Display* x11Display = glfwGetX11Display();
        ::Window x11Window = glfwGetX11Window(glfwWindow);

        WGPUSurfaceSourceXlibWindow xlibSource{};
        xlibSource.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        xlibSource.chain.next = nullptr;
        xlibSource.display = x11Display;
        xlibSource.window = static_cast<uint64_t>(x11Window);
        surfDesc.nextInChain = &xlibSource.chain;

#elif defined(_WIN32)
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        surfDesc.label = WGPUStringView{"threepp_surface", sizeof("threepp_surface") - 1};

        WGPUSurfaceSourceWindowsHWND hwndSource{};
        hwndSource.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
        hwndSource.chain.next = nullptr;
        hwndSource.hinstance = GetModuleHandle(nullptr);
        hwndSource.hwnd = glfwGetWin32Window(glfwWindow);
        surfDesc.nextInChain = &hwndSource.chain;

#elif defined(__APPLE__)
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        surfDesc.label = WGPUStringView{"threepp_surface", sizeof("threepp_surface") - 1};

        void* metalLayer = wgpu_create_metal_layer(glfwGetCocoaWindow(glfwWindow));
        WGPUSurfaceSourceMetalLayer metalSource{};
        metalSource.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
        metalSource.chain.next = nullptr;
        metalSource.layer = metalLayer;
        surfDesc.nextInChain = &metalSource.chain;
#endif

        surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    }

    void requestAdapter() {
        struct UserData {
            WGPUAdapter adapter = nullptr;
            bool done = false;
        } userData;

        WGPURequestAdapterOptions options{};
        options.compatibleSurface = surface; // nullptr in headless mode
        if (const char* backendEnv = std::getenv("WGPU_BACKEND")) {
            std::string b = backendEnv;
            if      (b == "vulkan"  || b == "Vulkan")  options.backendType = WGPUBackendType_Vulkan;
            else if (b == "dx12"    || b == "D3D12")   options.backendType = WGPUBackendType_D3D12;
            else if (b == "dx11"    || b == "D3D11")   options.backendType = WGPUBackendType_D3D11;
            else if (b == "metal"   || b == "Metal")   options.backendType = WGPUBackendType_Metal;
            else if (b == "opengl"  || b == "OpenGL")  options.backendType = WGPUBackendType_OpenGL;
        }

        WGPURequestAdapterCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                ud->adapter = adapter;
            } else {
                std::cerr << "WgpuRenderer: Adapter request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuInstanceRequestAdapter(instance, &options, callbackInfo);

        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("WgpuRenderer: requestAdapter timed out");
            }
#ifdef __EMSCRIPTEN__
            emscripten_sleep(0);
#else
            wgpuInstanceProcessEvents(instance);
#endif
        }

        if (!userData.adapter) {
            throw std::runtime_error("WgpuRenderer: failed to obtain adapter");
        }
        adapter = userData.adapter;
    }

    void requestDevice() {
        struct UserData {
            WGPUDevice device = nullptr;
            bool done = false;
        } userData;

        WGPUDeviceDescriptor deviceDesc{};
        deviceDesc.label = WGPUStringView{"threepp_device", sizeof("threepp_device") - 1};

        // Request adapter's max storage buffer limit (default 128MB may be too small for large scenes)
        WGPULimits adapterLimits{};
        wgpuAdapterGetLimits(adapter, &adapterLimits);
        adapterLimits.maxStorageBufferBindingSize = (std::max)(
            adapterLimits.maxStorageBufferBindingSize, uint64_t(256u * 1024u * 1024u));
        deviceDesc.requiredLimits = &adapterLimits;

        // Install an error callback that logs instead of panicking.
        // Transient validation errors (e.g. stale scissor rect during window
        // resize) are non-fatal and should not abort the process.
        deviceDesc.uncapturedErrorCallbackInfo.callback =
            [](WGPUDevice const*, WGPUErrorType type, WGPUStringView message,
               void*, void*) {
                std::cerr << "WgpuRenderer: GPU error (type " << static_cast<int>(type) << "): "
                          << std::string_view(message.data, message.length) << std::endl;
            };

        WGPURequestDeviceCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestDeviceStatus_Success) {
                ud->device = device;
            } else {
                std::cerr << "WgpuRenderer: Device request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuAdapterRequestDevice(adapter, &deviceDesc, callbackInfo);


        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("WgpuRenderer: requestDevice timed out");
            }
#ifdef __EMSCRIPTEN__
            emscripten_sleep(0);
#else
            wgpuInstanceProcessEvents(instance);
#endif
        }

        if (!userData.device) {
            throw std::runtime_error("WgpuRenderer: failed to obtain device");
        }
        device = userData.device;

        WGPUAdapterInfo info{};
        wgpuAdapterGetInfo(adapter, &info);
        const char* backendName = "Unknown";
        switch (info.backendType) {
            case WGPUBackendType_Vulkan:    backendName = "Vulkan";  break;
            case WGPUBackendType_Metal:     backendName = "Metal";   break;
            case WGPUBackendType_D3D12:     backendName = "D3D12";   break;
            case WGPUBackendType_D3D11:     backendName = "D3D11";   break;
            case WGPUBackendType_OpenGL:    backendName = "OpenGL";  break;
            case WGPUBackendType_OpenGLES:  backendName = "OpenGLES"; break;
            default: break;
        }
        std::cout << "WgpuRenderer: backend=" << backendName
                  << " adapter=\"" << std::string_view(info.device.data, info.device.length) << "\""
                  << std::endl;
        wgpuAdapterInfoFreeMembers(info);
    }

    void configureSurface() {
        const auto w = static_cast<uint32_t>(std::floor(size_.width()  * pixelRatio_));
        const auto h = static_cast<uint32_t>(std::floor(size_.height() * pixelRatio_));
        if (w == 0 || h == 0) return; // window is minimised — wgpu forbids zero-size surfaces

        WGPUSurfaceConfiguration config{};
        config.device = device;
        config.format = surfaceFormat;
        config.usage = WGPUTextureUsage_RenderAttachment;
        config.width  = w;
        config.height = h;
#ifdef __EMSCRIPTEN__
        config.presentMode = WGPUPresentMode_Fifo;  // browser controls vsync via requestAnimationFrame
#else
        config.presentMode = canvas.vsync() ? WGPUPresentMode_Fifo : WGPUPresentMode_Immediate;
#endif
        config.alphaMode = WGPUCompositeAlphaMode_Auto;
        config.viewFormatCount = 0;
        config.viewFormats = nullptr;

        wgpuSurfaceConfigure(surface, &config);

    }

    // Acquire the surface texture for this frame (called once per frame).
    // Returns false if the surface texture could not be acquired.
    bool acquireFrame() {
        if (frame_.active) return true; // Already acquired

        if (!surface) return false;

        wgpuSurfaceGetCurrentTexture(surface, &frame_.surfaceTexture);
        if (frame_.surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Occluded) {
            // Window not visible — release the acquired texture slot back to the
            // swap chain via present, otherwise subsequent calls time out.
            if (frame_.surfaceTexture.texture) {
                wgpuTextureRelease(frame_.surfaceTexture.texture);
                frame_.surfaceTexture.texture = nullptr;
            }
            wgpuSurfacePresent(surface);
            return false;
        }
        if (frame_.surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            frame_.surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            // Outdated/Lost means the surface needs reconfiguring (e.g. after restore).
            const auto s = frame_.surfaceTexture.status;
            if (s == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
                s == WGPUSurfaceGetCurrentTextureStatus_Lost) {
                configureSurface();
            } else {
                std::cerr << "WgpuRenderer: Failed to acquire surface texture (status "
                          << static_cast<int>(s) << ")" << std::endl;
            }
            return false;
        }

        frame_.width = wgpuTextureGetWidth(frame_.surfaceTexture.texture);
        frame_.height = wgpuTextureGetHeight(frame_.surfaceTexture.texture);

        // Ensure persistent depth/MSAA textures exist at the right size.
        ensureFrameTextures(frame_.width, frame_.height, sampleCount_);

        WGPUTextureViewDescriptor vd{};
        vd.label = WGPUStringView{"surface_view", sizeof("surface_view") - 1};
        vd.format = surfaceFormat;
        vd.dimension = WGPUTextureViewDimension_2D;
        vd.baseMipLevel = 0; vd.mipLevelCount = 1;
        vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
        vd.aspect = WGPUTextureAspect_All;

        if (sampleCount_ > 1) {
            frame_.resolveView = wgpuTextureCreateView(frame_.surfaceTexture.texture, &vd);
            frame_.colorView = cachedFrame_.msaaColorView;
        } else {
            frame_.colorView = wgpuTextureCreateView(frame_.surfaceTexture.texture, &vd);
        }

        frame_.depthView = cachedFrame_.depthView;

        frame_.active = true;
        frame_.hasRendered = false;
        return true;

    }

    // Present and release the per-frame surface state.
    // Release frame resources without presenting. Used by dispose() to
    // drop the surface texture so wgpu doesn't complain about a held
    // SurfaceOutput when the surface is reconfigured or destroyed.
    void releaseFrame() {
        if (!frame_.active) return;

        // If called without a matching endFrame() (e.g. dispose), discard the encoder.
        if (renderEncoder_) {
            wgpuCommandEncoderRelease(renderEncoder_);
            renderEncoder_ = nullptr;
        }

        // depth and MSAA textures/views are owned by cachedFrame_ — do not release here.
        if (frame_.resolveView) {
            // colorView points to the cached MSAA view — do not release.
            wgpuTextureViewRelease(frame_.resolveView);
        } else {
            wgpuTextureViewRelease(frame_.colorView);
        }
        wgpuTextureRelease(frame_.surfaceTexture.texture);


        frame_ = {};
    }

    void endFrame() {
        if (!frame_.active) return;

        // If tone mapping is active, blit the intermediate RT to the surface.
        if (needsToneMapPass() && toneMap_.colorTexture) {
            toneMapBlit();
        }

        // Render overlay (e.g. ImGui) directly to the surface, after all
        // render() calls and any tone-map blit. This ensures the overlay
        // is drawn exactly once per frame and is never tone-mapped.
        if (overlayCallback_) {
            overlayOnSurface();
        }

        // Submit the single per-frame command buffer that holds all render passes
        // (mirror RT, main scene, tone-map blit, overlay).  Submitting once keeps
        // all passes in one command buffer so wgpu-native inserts correct barriers.
        if (renderEncoder_) {
            WGPUCommandBufferDescriptor cmdDesc{};
            cmdDesc.label = WGPUStringView{"frame_cmd", sizeof("frame_cmd") - 1};
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(renderEncoder_, &cmdDesc);
            wgpuQueueSubmit(queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(renderEncoder_);
            renderEncoder_ = nullptr;
        }

#ifndef __EMSCRIPTEN__
        wgpuSurfacePresent(surface);
#endif

        releaseFrame();
    }

    // Build the WGSL source for a tone mapping fullscreen pass.
    static std::string buildToneMapWGSL(ToneMapping mode, bool srgb) {
        std::ostringstream s;
        s << R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var<uniform> params: vec4<f32>; // x=exposure

struct VSOutput { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VSOutput;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv = positions[vi] * 0.5 + vec2<f32>(0.5, 0.5);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment fn fs(in: VSOutput) -> @location(0) vec4<f32> {
    var color = textureSample(inputTex, inputSampler, in.uv);
    var rgb = color.rgb;
    let exposure = params.x;
)";
        // Tone mapping
        switch (mode) {
            case ToneMapping::Linear:
                s << "    rgb = rgb * exposure;\n";
                break;
            case ToneMapping::Reinhard:
                s << "    rgb = rgb * exposure;\n";
                s << "    rgb = rgb / (vec3<f32>(1.0) + rgb);\n";
                break;
            case ToneMapping::Cineon:
                s << "    rgb = rgb * exposure;\n";
                s << "    let x = max(vec3<f32>(0.0), rgb - vec3<f32>(0.004));\n";
                s << "    rgb = (x * (6.2 * x + vec3<f32>(0.5))) / (x * (6.2 * x + vec3<f32>(1.7)) + vec3<f32>(0.06));\n";
                break;
            case ToneMapping::ACESFilmic:
                s << "    rgb = rgb * exposure;\n";
                s << "    let a = rgb * (rgb * 2.51 + vec3<f32>(0.03));\n";
                s << "    let b = rgb * (rgb * 2.43 + vec3<f32>(0.59)) + vec3<f32>(0.14);\n";
                s << "    rgb = clamp(a / b, vec3<f32>(0.0), vec3<f32>(1.0));\n";
                break;
            default:
                break;
        }
        if (srgb) {
            s << "    rgb = pow(rgb, vec3<f32>(1.0 / 2.2));\n";
        }
        s << "    return vec4<f32>(rgb, color.a);\n}\n";
        return s.str();
    }

    // Get (or lazily create) the tone mapping pipeline for the current settings.
    int toneMapPipelineIndex() {
        switch (scope.toneMapping) {
            case ToneMapping::Linear: return 0;
            case ToneMapping::Reinhard: return 1;
            case ToneMapping::Cineon: return 2;
            case ToneMapping::ACESFilmic: return 3;
            default: return 4; // sRGB only, no tone mapping
        }
    }

    void ensureToneMapResources() {
        if (!toneMap_.initialized) {
            // Bind group layout: texture, sampler, uniform buffer
            WGPUBindGroupLayoutEntry entries[3]{};
            entries[0].binding = 0;
            entries[0].visibility = WGPUShaderStage_Fragment;
            entries[0].texture.sampleType = WGPUTextureSampleType_Float;
            entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

            entries[1].binding = 1;
            entries[1].visibility = WGPUShaderStage_Fragment;
            entries[1].sampler.type = WGPUSamplerBindingType_Filtering;

            entries[2].binding = 2;
            entries[2].visibility = WGPUShaderStage_Fragment;
            entries[2].buffer.type = WGPUBufferBindingType_Uniform;
            entries[2].buffer.minBindingSize = 16;

            WGPUBindGroupLayoutDescriptor bglDesc{};
            bglDesc.label = WGPUStringView{"tonemap_bgl", sizeof("tonemap_bgl") - 1};
            bglDesc.entryCount = 3;
            bglDesc.entries = entries;
            toneMap_.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

            WGPUPipelineLayoutDescriptor plDesc{};
            plDesc.label = WGPUStringView{"tonemap_pl", sizeof("tonemap_pl") - 1};
            plDesc.bindGroupLayoutCount = 1;
            plDesc.bindGroupLayouts = &toneMap_.bindGroupLayout;
            toneMap_.pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

            WGPUSamplerDescriptor sd{};
            sd.label = WGPUStringView{"tonemap_sampler", sizeof("tonemap_sampler") - 1};
            sd.minFilter = WGPUFilterMode_Linear;
            sd.magFilter = WGPUFilterMode_Linear;
            sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
            sd.addressModeU = WGPUAddressMode_ClampToEdge;
            sd.addressModeV = WGPUAddressMode_ClampToEdge;
            sd.addressModeW = WGPUAddressMode_ClampToEdge;
            sd.maxAnisotropy = 1;
            toneMap_.sampler = wgpuDeviceCreateSampler(device, &sd);

            WGPUBufferDescriptor ubDesc{};
            ubDesc.label = WGPUStringView{"tonemap_ub", sizeof("tonemap_ub") - 1};
            ubDesc.size = 16;
            ubDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            toneMap_.uniformBuf = wgpuDeviceCreateBuffer(device, &ubDesc);

            toneMap_.initialized = true;
        }

        // Ensure the pipeline for the current tone mapping mode exists
        int idx = toneMapPipelineIndex();
        if (!toneMap_.pipelines[idx]) {
            bool srgb = (scope.outputEncoding == Encoding::sRGB);
            auto wgsl = buildToneMapWGSL(scope.toneMapping, srgb);

            WGPUShaderSourceWGSL wgslSrc{};
            wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
            wgslSrc.code = {.data = wgsl.c_str(), .length = static_cast<size_t>(wgsl.size())};

            WGPUShaderModuleDescriptor smDesc{};
            smDesc.nextInChain = &wgslSrc.chain;
            smDesc.label = WGPUStringView{"tonemap_shader", sizeof("tonemap_shader") - 1};
            toneMap_.shaderModules[idx] = wgpuDeviceCreateShaderModule(device, &smDesc);

            WGPUColorTargetState colorTarget{};
            colorTarget.format = surfaceFormat;
            colorTarget.writeMask = WGPUColorWriteMask_All;

            auto vsEntry = WGPUStringView{"vs", sizeof("vs") - 1};
            auto fsEntry = WGPUStringView{"fs", sizeof("fs") - 1};

            WGPUFragmentState fragmentState{};
            fragmentState.module = toneMap_.shaderModules[idx];
            fragmentState.entryPoint = fsEntry;
            fragmentState.targetCount = 1;
            fragmentState.targets = &colorTarget;

            WGPURenderPipelineDescriptor pipeDesc{};
            pipeDesc.label = WGPUStringView{"tonemap_pipe", sizeof("tonemap_pipe") - 1};
            pipeDesc.layout = toneMap_.pipelineLayout;
            pipeDesc.vertex.module = toneMap_.shaderModules[idx];
            pipeDesc.vertex.entryPoint = vsEntry;
            pipeDesc.vertex.bufferCount = 0;
            pipeDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
            pipeDesc.primitive.cullMode = WGPUCullMode_None;
            pipeDesc.multisample.count = 1;
            pipeDesc.multisample.mask = ~0u;
            pipeDesc.fragment = &fragmentState;

            toneMap_.pipelines[idx] = wgpuDeviceCreateRenderPipeline(device, &pipeDesc);
        }
    }

    // Ensure the intermediate RT for tone mapping is the right size.
    void ensureToneMapRT(uint32_t w, uint32_t h) {
        if (toneMap_.width == w && toneMap_.height == h && toneMap_.colorTexture) return;

        // Release old
        if (toneMap_.cachedBindGroup)  { wgpuBindGroupRelease(toneMap_.cachedBindGroup);  toneMap_.cachedBindGroup  = nullptr; }
        if (toneMap_.cachedInputView)  { wgpuTextureViewRelease(toneMap_.cachedInputView); toneMap_.cachedInputView = nullptr; }
        toneMap_.lastExposure = -1.f;
        if (toneMap_.colorView) wgpuTextureViewRelease(toneMap_.colorView);
        if (toneMap_.colorTexture) wgpuTextureRelease(toneMap_.colorTexture);
        if (toneMap_.depthView) wgpuTextureViewRelease(toneMap_.depthView);
        if (toneMap_.depthTexture) wgpuTextureRelease(toneMap_.depthTexture);

        toneMap_.width = w;
        toneMap_.height = h;

        // Color texture (non-MSAA, used as texture binding source)
        WGPUTextureDescriptor td{};
        td.label = WGPUStringView{"tonemap_color", sizeof("tonemap_color") - 1};
        td.size = {w, h, 1};
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        td.dimension = WGPUTextureDimension_2D;
        td.format = surfaceFormat;
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc;
        toneMap_.colorTexture = wgpuDeviceCreateTexture(device, &td);
        toneMap_.colorView = wgpuTextureCreateView(toneMap_.colorTexture, nullptr);

        // Depth texture
        WGPUTextureDescriptor dtd{};
        dtd.label = WGPUStringView{"tonemap_depth", sizeof("tonemap_depth") - 1};
        dtd.size = {w, h, 1};
        dtd.mipLevelCount = 1;
        dtd.sampleCount = 1;
        dtd.dimension = WGPUTextureDimension_2D;
        dtd.format = WGPUTextureFormat_Depth24Plus;
        dtd.usage = WGPUTextureUsage_RenderAttachment;
        toneMap_.depthTexture = wgpuDeviceCreateTexture(device, &dtd);
        toneMap_.depthView = wgpuTextureCreateView(toneMap_.depthTexture, nullptr);
    }

    // Blit the tone map intermediate RT to the surface with tone mapping applied.
    void toneMapBlit() {
        ensureToneMapResources();

        int idx = toneMapPipelineIndex();
        if (!toneMap_.pipelines[idx]) return;

        // Upload exposure uniform only when it changes
        const float exposure = scope.toneMappingExposure;
        if (exposure != toneMap_.lastExposure) {
            float params[4] = {exposure, 0, 0, 0};
            wgpuQueueWriteBuffer(queue, toneMap_.uniformBuf, 0, params, 16);
            toneMap_.lastExposure = exposure;
        }

        // Input texture view and bind group are stable until the RT is resized — cache them.
        if (!toneMap_.cachedInputView) {
            toneMap_.cachedInputView = wgpuTextureCreateView(toneMap_.colorTexture, nullptr);
        }
        if (!toneMap_.cachedBindGroup) {
            WGPUBindGroupEntry bgEntries[3]{};
            bgEntries[0].binding = 0;
            bgEntries[0].textureView = toneMap_.cachedInputView;
            bgEntries[1].binding = 1;
            bgEntries[1].sampler = toneMap_.sampler;
            bgEntries[2].binding = 2;
            bgEntries[2].buffer = toneMap_.uniformBuf;
            bgEntries[2].size = 16;

            WGPUBindGroupDescriptor bgDesc{};
            bgDesc.label = WGPUStringView{"tonemap_bg", sizeof("tonemap_bg") - 1};
            bgDesc.layout = toneMap_.bindGroupLayout;
            bgDesc.entryCount = 3;
            bgDesc.entries = bgEntries;
            toneMap_.cachedBindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);
        }
        WGPUBindGroup bindGroup = toneMap_.cachedBindGroup;

        // Render to the surface
        WGPUTextureView surfaceColorView;
        WGPUTextureView surfaceResolveView = nullptr;
        WGPUTexture blitMsaaTexture = nullptr;

        WGPUTextureViewDescriptor vd{};
        vd.label = WGPUStringView{"blit_view", sizeof("blit_view") - 1};
        vd.format = surfaceFormat;
        vd.dimension = WGPUTextureViewDimension_2D;
        vd.baseMipLevel = 0; vd.mipLevelCount = 1;
        vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
        vd.aspect = WGPUTextureAspect_All;

        // Tone map blit is always non-MSAA (fullscreen quad)

        surfaceColorView = wgpuTextureCreateView(frame_.surfaceTexture.texture, &vd);


        // Ensure the per-frame encoder exists (toneMapBlit is called from endFrame).
        if (!renderEncoder_) {
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"render_enc", sizeof("render_enc") - 1};
            renderEncoder_ = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        }

        WGPURenderPassColorAttachment colorAtt{};
        colorAtt.view = surfaceColorView;
        colorAtt.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        colorAtt.loadOp = WGPULoadOp_Clear;
        colorAtt.storeOp = WGPUStoreOp_Store;
        colorAtt.clearValue = {0, 0, 0, 1};

        WGPURenderPassDescriptor rpDesc{};
        rpDesc.label = WGPUStringView{"tonemap_pass", sizeof("tonemap_pass") - 1};
        rpDesc.colorAttachmentCount = 1;
        rpDesc.colorAttachments = &colorAtt;

        WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(renderEncoder_, &rpDesc);
        wgpuRenderPassEncoderSetPipeline(rp, toneMap_.pipelines[idx]);
        wgpuRenderPassEncoderSetBindGroup(rp, 0, bindGroup, 0, nullptr);
        wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(rp);
        wgpuRenderPassEncoderRelease(rp);

#ifndef __EMSCRIPTEN__
        wgpuTextureViewRelease(surfaceColorView);
#endif
        // toneMap_.uniformBuf, cachedInputView, cachedBindGroup are persistent — not released here
    }

    // Render the ImGui overlay directly to the surface (after tone map blit).
    // Uses LoadOp_Load to preserve the tone-mapped scene.
    // Must include a depth attachment because the ImGui pipeline was initialized
    // with DepthStencilFormat = Depth24Plus.
    void overlayOnSurface() {
        WGPUTextureViewDescriptor vd{};
        vd.label = WGPUStringView{"overlay_view", sizeof("overlay_view") - 1};
        vd.format = surfaceFormat;
        vd.dimension = WGPUTextureViewDimension_2D;
        vd.baseMipLevel = 0; vd.mipLevelCount = 1;
        vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
        vd.aspect = WGPUTextureAspect_All;

        WGPUTextureView surfaceView = wgpuTextureCreateView(frame_.surfaceTexture.texture, &vd);


        // Create a temporary depth texture matching the surface dimensions.
        // The ImGui pipeline expects Depth24Plus even though it doesn't use depth.
        WGPUTextureDescriptor dtd{};
        dtd.label = WGPUStringView{"overlay_depth", sizeof("overlay_depth") - 1};
        dtd.size = {frame_.width, frame_.height, 1};
        dtd.mipLevelCount = 1;
        dtd.sampleCount = 1; // surface texture is always non-MSAA
        dtd.dimension = WGPUTextureDimension_2D;
        dtd.format = WGPUTextureFormat_Depth24Plus;
        dtd.usage = WGPUTextureUsage_RenderAttachment;
        WGPUTexture overlayDepthTex = wgpuDeviceCreateTexture(device, &dtd);
        WGPUTextureView overlayDepthView = wgpuTextureCreateView(overlayDepthTex, nullptr);

        // Ensure the per-frame encoder exists (overlayOnSurface is called from endFrame).
        if (!renderEncoder_) {
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"render_enc", sizeof("render_enc") - 1};
            renderEncoder_ = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        }

        WGPURenderPassColorAttachment colorAtt{};
        colorAtt.view = surfaceView;
        colorAtt.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        colorAtt.loadOp = WGPULoadOp_Load;
        colorAtt.storeOp = WGPUStoreOp_Store;

        WGPURenderPassDepthStencilAttachment depthAtt{};
        depthAtt.view = overlayDepthView;
        depthAtt.depthLoadOp = WGPULoadOp_Clear;
        depthAtt.depthStoreOp = WGPUStoreOp_Discard;
        depthAtt.depthClearValue = 1.0f;

        WGPURenderPassDescriptor rpDesc{};
        rpDesc.label = WGPUStringView{"overlay_pass", sizeof("overlay_pass") - 1};
        rpDesc.colorAttachmentCount = 1;
        rpDesc.colorAttachments = &colorAtt;
        rpDesc.depthStencilAttachment = &depthAtt;

        WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(renderEncoder_, &rpDesc);
        overlayCallback_(static_cast<void*>(rp));
        wgpuRenderPassEncoderEnd(rp);
        wgpuRenderPassEncoderRelease(rp);

        wgpuTextureViewRelease(overlayDepthView);
        wgpuTextureRelease(overlayDepthTex);
#ifndef __EMSCRIPTEN__
        wgpuTextureViewRelease(surfaceView);
#endif
    }

    // Execute a standalone clear pass (mid-frame, outside of a scene render).
    void executeClear(bool color, bool depth, bool stencil) {
        if (!initialized) return;

        bool useSurface = (currentRenderTarget_ == nullptr && surface != nullptr);
        if (!useSurface && currentRenderTarget_ == nullptr) return;

        WGPUTextureView colorView = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUTextureView resolveView = nullptr;
        uint32_t attachW = 0, attachH = 0;

        if (useSurface) {
            if (!acquireFrame()) return;
            if (needsToneMapPass()) {
                ensureToneMapRT(frame_.width, frame_.height);
                colorView = toneMap_.colorView;
                depthView = toneMap_.depthView;
            } else {
                colorView = frame_.colorView;
                depthView = frame_.depthView;
                resolveView = frame_.resolveView;
            }
            attachW = frame_.width;
            attachH = frame_.height;
        } else {
            auto& rt = renderTargets->getOrCreate(currentRenderTarget_, sampleCount_);
            attachW = rt.width;
            attachH = rt.height;
            depthView = rt.depthView;
            if (sampleCount_ > 1 && rt.msaaColorView) {
                colorView = rt.msaaColorView;
                resolveView = rt.colorView;
            } else {
                colorView = rt.colorView;
            }
        }

        if (!renderEncoder_) {
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"render_enc", sizeof("render_enc") - 1};
            renderEncoder_ = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        }

        WGPURenderPassColorAttachment colorAttachment{};
        colorAttachment.view = colorView;
        colorAttachment.resolveTarget = resolveView;
        colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        colorAttachment.loadOp = color ? WGPULoadOp_Clear : WGPULoadOp_Load;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = {
                static_cast<double>(clearColor_.r),
                static_cast<double>(clearColor_.g),
                static_cast<double>(clearColor_.b),
                static_cast<double>(clearAlpha_)};

        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = depthView;
        depthAttachment.depthLoadOp = depth ? WGPULoadOp_Clear : WGPULoadOp_Load;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor passDesc{};
        passDesc.label = WGPUStringView{"clear_pass", sizeof("clear_pass") - 1};
        passDesc.colorAttachmentCount = 1;
        passDesc.colorAttachments = &colorAttachment;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(renderEncoder_, &passDesc);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        // After a clear, the surface has content — mark as rendered
        if (useSurface) frame_.hasRendered = true;
    }

    void render(Object3D& scene, Camera& camera) {
        if (!initialized) return;

        if (!frame_.active) {
            // New frame — recycle per-draw buffers and evict stale bind groups
            bufferPool->beginFrame();
            bindGroupCache->beginFrame();
        }

        // Reset per-frame statistics
        renderInfo.frame++;
        renderInfo.calls = 0;
        renderInfo.triangles = 0;
        renderInfo.lines = 0;
        renderInfo.points = 0;
        renderInfo.geometries = geometries->count();
        renderInfo.textures = textures->count();

        // Update window size if changed
        auto currentSize = canvas.size();
        if (currentSize.width() != size_.width() || currentSize.height() != size_.height()) {
            size_ = currentSize;
            if (surface) {
                // Size changed — must release current frame and reconfigure.
                // Skip if minimised (zero area) — wgpu forbids zero-size surfaces.
                endFrame();
                configureSurface(); // no-op when w==0 || h==0
            }
            viewport_.w = static_cast<float>(size_.width());
            viewport_.h = static_cast<float>(size_.height());
            scissor_.w = static_cast<uint32_t>(size_.width());
            scissor_.h = static_cast<uint32_t>(size_.height());
        }

        // Skip rendering entirely while the window has zero area (minimised)
        if (size_.width() == 0 || size_.height() == 0) return;

        // Update matrices
        scene.updateMatrixWorld();
        if (!camera.parent) {
            camera.updateMatrixWorld();
        }
        camera.updateWorldMatrix(false, false);

        Matrix4 projectionMatrix = camera.projectionMatrix;
        Matrix4 viewMatrix = camera.matrixWorldInverse;

        // Remap NDC z from [-1,1] (OpenGL convention used by three.js matrices)
        // to [0,1] (WebGPU/Vulkan convention). Apply: z' = 0.5*z + 0.5*w
        {
            auto& e = projectionMatrix.elements;
            e[2]  = 0.5f * e[2]  + 0.5f * e[3];
            e[6]  = 0.5f * e[6]  + 0.5f * e[7];
            e[10] = 0.5f * e[10] + 0.5f * e[11];
            e[14] = 0.5f * e[14] + 0.5f * e[15];
        }

        // Upload world-space light data, then snapshot it into a per-render pool buffer.
        // With the single deferred encoder, all render passes in a frame share one submit.
        // If lights->update() wrote directly to a persistent buffer, a later render() call
        // (e.g. the HUD scene which has no lights) would overwrite it with zeros before the
        // GPU runs the earlier passes.  Acquiring a unique pool buffer per render() call
        // ensures each render pass sees its own correct light data.
        lights->update(scene);
        {
            constexpr auto kLightUsage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            renderLightBuffer_ = bufferPool->acquire(
                lights->lightUniformSize(), kLightUsage, lights->currentData().data());
        }

        // Shadow pass (delegated to WgpuShadowMap subsystem)
        shadowMap->autoUpdate = scope.shadowMapAutoUpdate;
        shadowMap->beginFrame(scene);
        scope.shadowMapAutoUpdate = shadowMap->autoUpdate;

        // Collect renderables BEFORE the render pass so onBeforeRender callbacks can
        // trigger nested renderer->render() calls (e.g. Water mirror pass) without
        // nesting inside an active WebGPU render pass encoder.
        renderList_.init();
        {
            Matrix4 psmTemp;
            psmTemp.multiplyMatrices(projectionMatrix, viewMatrix);
            frustum_.setFromProjectionMatrix(psmTemp);
            collectRenderables(scene, psmTemp, camera, 0);
        }
        if (scope.sortObjects) renderList_.sort();
        renderList_.finish();

        // Fire onBeforeRender for each object (may trigger nested renders).
        // Nested render() calls overwrite renderList_, so we must re-collect afterward.
        bool hadBeforeRender = false;
        auto* scenePtr = &scene;
        auto* cameraPtr = &camera;
        for (auto* item : renderList_.opaque) {
            if (item->object && item->object->onBeforeRender) {
                hadBeforeRender = true;
                item->object->onBeforeRender.value()(
                        &scope, scenePtr, cameraPtr,
                        item->geometry, item->material, item->group);
            }
        }
        for (auto* item : renderList_.transparent) {
            if (item->object && item->object->onBeforeRender) {
                hadBeforeRender = true;
                item->object->onBeforeRender.value()(
                        &scope, scenePtr, cameraPtr,
                        item->geometry, item->material, item->group);
            }
        }

        // Re-collect renderables since nested renders overwrote the render list
        if (hadBeforeRender) {
            renderList_.init();
            {
                Matrix4 psmTemp;
                psmTemp.multiplyMatrices(projectionMatrix, viewMatrix);
                frustum_.setFromProjectionMatrix(psmTemp);
                collectRenderables(scene, psmTemp, camera, 0);
            }
            if (scope.sortObjects) renderList_.sort();
            renderList_.finish();
        }

        // Force intermediate RT when transmissive objects need a framebuffer copy
        hasTransmissive_ = !renderList_.transmissive.empty();

        // Determine render target views
        WGPUTextureView colorView = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUTextureView resolveView = nullptr;
        bool useSurface = (currentRenderTarget_ == nullptr && surface != nullptr);

        if (currentRenderTarget_ == nullptr && surface == nullptr) {
            return; // Headless mode with no render target set
        }

        // Actual attachment dimensions — used to clamp viewport/scissor
        uint32_t attachW = 0, attachH = 0;
        // Reset effective sample count (may be overridden when rendering to
        // the non-MSAA tone mapping intermediate RT).
        effectiveSampleCount_ = sampleCount_;

        if (useSurface) {
            // Acquire surface texture (reused across multiple render() calls per frame)
            if (!acquireFrame()) return;

            if (needsToneMapPass()) {
                // Redirect to intermediate RT; endFrame() will blit with tone mapping
                ensureToneMapRT(frame_.width, frame_.height);
                colorView = toneMap_.colorView;
                depthView = toneMap_.depthView;
                resolveView = nullptr; // No MSAA on the intermediate RT
                effectiveSampleCount_ = 1;
            } else {
                colorView = frame_.colorView;
                depthView = frame_.depthView;
                resolveView = frame_.resolveView;
            }
            attachW = frame_.width;
            attachH = frame_.height;
        } else {
            auto& rt = renderTargets->getOrCreate(currentRenderTarget_, sampleCount_);
            attachW = rt.width;
            attachH = rt.height;
            depthView = rt.depthView;
            if (sampleCount_ > 1 && rt.msaaColorView) {
                colorView = rt.msaaColorView;
                resolveView = rt.colorView;
            } else {
                colorView = rt.colorView;
                effectiveSampleCount_ = 1; // RT has no MSAA; use 1-sample pipelines
            }
        }

        // Create the per-frame encoder on the first render() call; subsequent calls
        // (e.g. mirror pass, main scene) append passes to the same encoder.
        // Submitted once in endFrame() so wgpu-native inserts correct barriers.
        if (!renderEncoder_) {
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"render_enc", sizeof("render_enc") - 1};
            renderEncoder_ = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        }
        WGPUCommandEncoder encoder = renderEncoder_;

        // Flush any pending mipmap generation before the render pass begins.
        // Mipmap command buffers must be submitted while no render pass is active
        // to avoid wgpu-native validation errors with MSAA render passes.
        textures->flushPendingMipmaps();

        // Determine clear color
        auto* sceneObj = scene.as<Scene>();
        currentSceneEnv_ = sceneObj ? sceneObj->environment.get() : nullptr;
        Color effectiveClearColor = clearColor_;
        float effectiveClearAlpha = clearAlpha_;
        if (sceneObj && sceneObj->background.isColor()) {
            effectiveClearColor = sceneObj->background.color();
            effectiveClearAlpha = 1.0f;
        }

        // Decide whether to clear on this render pass.
        // Like the GL renderer: when autoClear is on, every render() clears.
        // When autoClear is off, only explicit clear() calls (pending flags) clear.
        bool shouldClearColor = pendingClearColor_;
        bool shouldClearDepth = pendingClearDepth_;

        if (useSurface) {
            if (scope.autoClear) {
                shouldClearColor |= scope.autoClearColor;
                shouldClearDepth |= scope.autoClearDepth;
            }
        } else {
            // Render target: always clear (matches previous behavior)
            shouldClearColor = true;
            shouldClearDepth = true;
        }

        pendingClearColor_ = false;
        pendingClearDepth_ = false;
        pendingClearStencil_ = false;

        // Render pass
        WGPURenderPassColorAttachment colorAttachment{};
        colorAttachment.view = colorView;
        colorAttachment.resolveTarget = resolveView;
        colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        // When scissor test is active, loadOp_Clear would clear the entire attachment
        // ignoring the scissor rect (unlike WebGL where gl.clear() respects scissor).
        // Use loadOp_Load and draw a scissored fullscreen triangle instead.
        bool scissoredClear = scissorTest_ && shouldClearColor;
        colorAttachment.loadOp = (shouldClearColor && !scissoredClear) ? WGPULoadOp_Clear : WGPULoadOp_Load;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = {
                static_cast<double>(effectiveClearColor.r),
                static_cast<double>(effectiveClearColor.g),
                static_cast<double>(effectiveClearColor.b),
                static_cast<double>(effectiveClearAlpha)};

        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = depthView;
        // Depth clear also ignores scissor; use Load and let the clear triangle write depth=1.
        depthAttachment.depthLoadOp = (shouldClearDepth && !scissoredClear) ? WGPULoadOp_Clear : WGPULoadOp_Load;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor passDesc{};
        passDesc.label = WGPUStringView{"render_pass", sizeof("render_pass") - 1};
        passDesc.colorAttachmentCount = 1;
        passDesc.colorAttachments = &colorAttachment;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);

        // Auto-adapt viewport to render target dimensions when no explicit
        // setViewport() was called after setRenderTarget() (matches GLRenderer).
        if (currentRenderTarget_ && !viewportExplicit_) {
            auto& rtVp = currentRenderTarget_->viewport;
            wgpuRenderPassEncoderSetViewport(pass, rtVp.x, rtVp.y, rtVp.z, rtVp.w, 0.0f, 1.0f);
        } else {
            float vw = (std::min)(viewport_.w, static_cast<float>(attachW));
            float vh = (std::min)(viewport_.h, static_cast<float>(attachH));
            wgpuRenderPassEncoderSetViewport(pass, viewport_.x, viewport_.y, vw, vh, 0.0f, 1.0f);
        }
        // Always set scissor rect explicitly to the attachment dimensions.
        // wgpu-native's default scissor can derive from the configured surface
        // size rather than the actual texture size during a live window resize,
        // causing a validation error when the surface texture is smaller.
        if (scissorTest_) {
            uint32_t sw = (std::min)(scissor_.w, attachW);
            uint32_t sh = (std::min)(scissor_.h, attachH);
            wgpuRenderPassEncoderSetScissorRect(pass, scissor_.x, scissor_.y, sw, sh);
        } else {
            wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, attachW, attachH);
        }

        // When scissorTest_ is active we couldn't use loadOp_Clear (it ignores scissor).
        // Draw a fullscreen triangle at z=1 with the clear color to fill only the scissored region.
        if (scissoredClear) {
            drawScissoredClear(pass, effectiveClearColor, effectiveClearAlpha,
                               surfaceFormat, effectiveSampleCount_);
        }

        // Draw environment map background (equirectangular) when available.
        // Only on primary renders (shouldClearColor), not overlay passes.
        if (shouldClearColor && sceneObj && sceneObj->environment) {
            Matrix4 vp;
            vp.multiplyMatrices(projectionMatrix, viewMatrix);
            Matrix4 invVP;
            invVP.copy(vp).invert();
            drawEnvBackground(pass, invVP, sceneObj->environment.get(),
                              surfaceFormat, effectiveSampleCount_);
        }

        // Build per-frame rendering context
        wgpu::FrameContext frameCtx{};
        if (sceneObj && sceneObj->fog) {
            if (auto* f = std::get_if<Fog>(&*sceneObj->fog)) {
                frameCtx.fogColor = f->color;
                frameCtx.fogNear = f->nearPlane;
                frameCtx.fogFar = f->farPlane;
                frameCtx.fogBits = SF::FogLinear;
            } else if (auto* f2 = std::get_if<FogExp2>(&*sceneObj->fog)) {
                frameCtx.fogColor = f2->color;
                frameCtx.fogDensity = f2->density;
                frameCtx.fogBits = SF::FogExp2;
            }
        }
        // When rendering to a render target, apply tone mapping per-object
        // in shaders (the post-process blit only applies to surface rendering).
        // When rendering to the surface, leave bits at zero — endFrame() handles it.
        frameCtx.toneMappingExposure = scope.toneMappingExposure;
        if (currentRenderTarget_) {
            switch (scope.toneMapping) {
                case ToneMapping::Linear:    frameCtx.tonemapBits = SF::TonemapLinear; break;
                case ToneMapping::Reinhard:  frameCtx.tonemapBits = SF::TonemapReinhard; break;
                case ToneMapping::Cineon:    frameCtx.tonemapBits = SF::TonemapCineon; break;
                case ToneMapping::ACESFilmic:frameCtx.tonemapBits = SF::TonemapACES; break;
                default: break;
            }
            if (scope.outputEncoding == Encoding::sRGB) {
                frameCtx.srgbOutput = true;
            }
        }
        frameCtx.localClippingEnabled = scope.localClippingEnabled;
        frameCtx.shadowActive = shadowMap->isActive();

        // Render opaque objects (front-to-back, depth write on)
        for (auto* item : renderList_.opaque) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera, frameCtx);
        }

        // Transmissive objects: end pass, copy framebuffer, generate mipmaps, re-open pass
        if (!renderList_.transmissive.empty()) {
            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);

            // Copy from the intermediate RT (hasTransmissive_ forces needsToneMapPass,
            // so toneMap_.colorTexture is always the active color target here).
            WGPUTexture srcTex = toneMap_.colorTexture;

            ensureTransmissionRT(attachW, attachH, surfaceFormat);

            // Copy the opaque framebuffer to the transmission texture
            WGPUTexelCopyTextureInfo src{};
            src.texture = srcTex;
            src.mipLevel = 0;
            WGPUTexelCopyTextureInfo dst{};
            dst.texture = transmissionTex_;
            dst.mipLevel = 0;
            WGPUExtent3D extent = {attachW, attachH, 1};
            wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);

            // Submit the copy command
            WGPUCommandBufferDescriptor cbDesc{};
            WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
            wgpuQueueSubmit(queue, 1, &cb);
            wgpuCommandBufferRelease(cb);

            // Generate mipmaps (submits its own command buffer)
            {
                uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(static_cast<float>((std::max)(attachW, attachH))))) + 1u;
                textures->mipmapGen().generate2D(transmissionTex_, attachW, attachH, mipLevels, surfaceFormat);
            }

            // Create a fresh encoder for the rest of the frame
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"transmissive_enc", sizeof("transmissive_enc") - 1};
            encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
            renderEncoder_ = encoder;

            // Re-open render pass with Load ops to preserve existing content
            WGPURenderPassColorAttachment colorAtt2{};
            colorAtt2.view = colorView;
            colorAtt2.resolveTarget = resolveView;
            colorAtt2.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            colorAtt2.loadOp = WGPULoadOp_Load;
            colorAtt2.storeOp = WGPUStoreOp_Store;

            WGPURenderPassDepthStencilAttachment depthAtt2{};
            depthAtt2.view = depthView;
            depthAtt2.depthLoadOp = WGPULoadOp_Load;
            depthAtt2.depthStoreOp = WGPUStoreOp_Store;
            depthAtt2.depthClearValue = 1.0f;

            WGPURenderPassDescriptor passDesc2{};
            passDesc2.label = WGPUStringView{"transmissive_pass", sizeof("transmissive_pass") - 1};
            passDesc2.colorAttachmentCount = 1;
            passDesc2.colorAttachments = &colorAtt2;
            passDesc2.depthStencilAttachment = &depthAtt2;

            pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc2);

            // Restore viewport and scissor
            if (currentRenderTarget_ && !viewportExplicit_) {
                auto& rtVp = currentRenderTarget_->viewport;
                wgpuRenderPassEncoderSetViewport(pass, rtVp.x, rtVp.y, rtVp.z, rtVp.w, 0.0f, 1.0f);
            } else {
                float vw = (std::min)(viewport_.w, static_cast<float>(attachW));
                float vh = (std::min)(viewport_.h, static_cast<float>(attachH));
                wgpuRenderPassEncoderSetViewport(pass, viewport_.x, viewport_.y, vw, vh, 0.0f, 1.0f);
            }
            if (scissorTest_) {
                uint32_t sw = (std::min)(scissor_.w, attachW);
                uint32_t sh = (std::min)(scissor_.h, attachH);
                wgpuRenderPassEncoderSetScissorRect(pass, scissor_.x, scissor_.y, sw, sh);
            } else {
                wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, attachW, attachH);
            }

            // Set transmission sampler size in frame context for uniform packing
            frameCtx.transmissionTexW = static_cast<float>(attachW);
            frameCtx.transmissionTexH = static_cast<float>(attachH);

            for (auto* item : renderList_.transmissive) {
                renderItem(pass, item, projectionMatrix, viewMatrix, camera, frameCtx);
            }
        }

        // Render transparent objects (back-to-front, depth write off)
        for (auto* item : renderList_.transparent) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera, frameCtx);
        }

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

#ifndef __EMSCRIPTEN__
        // Retain framebuffer for copyFramebufferToTexture if requested
        if (retainFramebuffer && useSurface) {
            uint32_t w = static_cast<uint32_t>(size_.width());
            uint32_t h = static_cast<uint32_t>(size_.height());

            // (Re)create retained texture if size changed.
            // Format is Rgba8Unorm to match DataTexture destinations — a render
            // blit (not CopyTextureToTexture) bridges the surfaceFormat difference.
            if (!retainedFB || retainedFBWidth != w || retainedFBHeight != h) {
                if (retainedFB) wgpuTextureRelease(retainedFB);
                WGPUTextureDescriptor td{};
                td.label = WGPUStringView{"retained_fb", sizeof("retained_fb") - 1};
                td.size = {w, h, 1};
                td.mipLevelCount = 1;
                td.sampleCount = 1;
                td.dimension = WGPUTextureDimension_2D;
                td.format = WGPUTextureFormat_RGBA8Unorm;
                td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
                retainedFB = wgpuDeviceCreateTexture(device, &td);
                retainedFBWidth = w;
                retainedFBHeight = h;
            }

            // Blit toneMap_.colorTexture (surfaceFormat/Bgra8Unorm on Windows)
            // into retainedFB (Rgba8Unorm) via a shader pass. The shader stage
            // handles the format mapping transparently; CopyTextureToTexture
            // cannot be used because BGRA and RGBA are not copy-compatible.
            ensureRetainBlit();

            WGPUTextureViewDescriptor tvd{};
            tvd.dimension = WGPUTextureViewDimension_2D;
            tvd.mipLevelCount = 1;
            tvd.arrayLayerCount = 1;
            WGPUTextureView srcView = wgpuTextureCreateView(toneMap_.colorTexture, &tvd);
            WGPUTextureView dstView = wgpuTextureCreateView(retainedFB, &tvd);

            WGPUBindGroupEntry bgEntries[2]{};
            bgEntries[0].binding = 0;
            bgEntries[0].textureView = srcView;
            bgEntries[1].binding = 1;
            bgEntries[1].sampler = retainBlit_.sampler;
            WGPUBindGroupDescriptor bgd{};
            bgd.layout = retainBlit_.bgl;
            bgd.entryCount = 2;
            bgd.entries = bgEntries;
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgd);

            WGPURenderPassColorAttachment ca{};
            ca.view = dstView;
            ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            ca.loadOp = WGPULoadOp_Clear;
            ca.storeOp = WGPUStoreOp_Store;
            ca.clearValue = {0, 0, 0, 1};

            WGPURenderPassDescriptor rpassDesc{};
            rpassDesc.colorAttachmentCount = 1;
            rpassDesc.colorAttachments = &ca;
            WGPURenderPassEncoder rpe = wgpuCommandEncoderBeginRenderPass(encoder, &rpassDesc);
            wgpuRenderPassEncoderSetPipeline(rpe, retainBlit_.pipeline);
            wgpuRenderPassEncoderSetBindGroup(rpe, 0, bg, 0, nullptr);
            wgpuRenderPassEncoderDraw(rpe, 3, 1, 0, 0);
            wgpuRenderPassEncoderEnd(rpe);
            wgpuRenderPassEncoderRelease(rpe);

            wgpuBindGroupRelease(bg);
            wgpuTextureViewRelease(srcView);
            wgpuTextureViewRelease(dstView);
        }
#endif

        // Submission is deferred to endFrame() — all passes share renderEncoder_.
        if (useSurface) frame_.hasRendered = true;
    }

    void renderCustomShaderObject(WGPURenderPassEncoder pass, Mesh* mesh,
                                   ShaderMaterial* sm,
                                   const Matrix4& projectionMatrix, const Matrix4& viewMatrix,
                                   const Camera& camera) {
        auto geometry = mesh->geometry();
        if (!geometry || !geometry->hasAttribute("position")) return;

        // Register dispose listener so we evict the pipeline cache entry when the material is destroyed
        if (!sm->hasEventListener("dispose", onMaterialDispose)) {
            sm->addEventListener("dispose", onMaterialDispose);
            trackedMaterials_.insert(sm);
        }

        auto& pe = pipelines->getOrCreateCustomPipeline(sm, surfaceFormat, effectiveSampleCount_);
        if (!pe.pipeline) return;

        // Create per-draw transform buffer
        float transformData[wgpu::TRANSFORM_UNIFORM_SIZE / sizeof(float)];
        std::memset(transformData, 0, sizeof(transformData));
        std::memcpy(transformData, mesh->matrixWorld->elements.data(), 64);
        std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);
        std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64);

        Matrix3 normalMatrix;
        normalMatrix.setFromMatrix4(*mesh->matrixWorld);
        normalMatrix.invert();
        normalMatrix.transpose();
        auto& ne = normalMatrix.elements;
        transformData[48] = ne[0]; transformData[49] = ne[1]; transformData[50] = ne[2]; transformData[51] = 0;
        transformData[52] = ne[3]; transformData[53] = ne[4]; transformData[54] = ne[5]; transformData[55] = 0;
        transformData[56] = ne[6]; transformData[57] = ne[7]; transformData[58] = ne[8]; transformData[59] = 0;

        Vector3 camPos;
        camPos.setFromMatrixPosition(*camera.matrixWorld);
        transformData[60] = camPos.x;
        transformData[61] = camPos.y;
        transformData[62] = camPos.z;
        transformData[63] = uniformPad_;

        constexpr auto kUniformUsage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer perDrawTransformBuf = bufferPool->acquire(wgpu::TRANSFORM_UNIFORM_SIZE, kUniformUsage, transformData);

        // Custom uniforms at binding 2.
        // For GLSL-compat shaders, pe.customUniformNames lists exactly which uniforms appear
        // in the binding-2 UBO (directly declared in the original GLSL, not from chunk includes).
        // For WGSL shaders, pe.customUniformNames is empty and we fall back to all non-texture uniforms.
        WGPUBuffer customUniformBuf = nullptr;
        bool hasCustomUniforms = false;
        if (!pe.customUniformNames.empty()) {
            hasCustomUniforms = true;  // GLSL-compat: names already filtered
        } else {
            for (auto& [name, uniform] : sm->uniforms) {
                if (!uniform.hasValue()) continue;
                auto& val = const_cast<Uniform&>(uniform).value();
                if (!std::get_if<Texture*>(&val)) { hasCustomUniforms = true; break; }
            }
        }
        if (hasCustomUniforms) {
            const uint32_t uboBytes = pe.customUniformSize;
            const uint32_t uboFloats = uboBytes / sizeof(float);
            std::vector<float> uboData(uboFloats, 0.0f);

            // Build the sorted list of uniforms to pack, matching the GLSL UBO layout.
            std::vector<std::pair<std::string, Uniform*>> sorted;
            if (!pe.customUniformNames.empty()) {
                // GLSL-compat path: pack only the filtered uniforms in the pre-sorted order.
                // This must match the alphabetical order the translator used for the SPIR-V UBO.
                for (auto& name : pe.customUniformNames) {
                    auto it2 = sm->uniforms.find(name);
                    if (it2 != sm->uniforms.end() && it2->second.hasValue()) {
                        auto& val = const_cast<Uniform&>(it2->second).value();
                        if (!std::get_if<Texture*>(&val)) sorted.push_back({name, &it2->second});
                    }
                }
            } else {
                // WGSL path: pack all non-texture uniforms sorted alphabetically.
                sorted.reserve(sm->uniforms.size());
                for (auto& [name, uniform] : sm->uniforms) {
                    if (!uniform.hasValue()) continue;
                    auto& val = const_cast<Uniform&>(uniform).value();
                    if (!std::get_if<Texture*>(&val)) sorted.push_back({name, &uniform});
                }
                std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first < b.first; });
            }

            // Pack with 16-byte (4-float) slots per scalar/vec3, 64-byte (16-float) per mat4
            uint32_t idx = 0;
            for (auto& [name, uptr] : sorted) {
                auto& val = uptr->value();
                if (auto* f = std::get_if<float>(&val)) {
                    if (idx + 4 <= uboFloats) { uboData[idx] = *f; idx += 4; }
                } else if (auto* i = std::get_if<int>(&val)) {
                    if (idx + 4 <= uboFloats) { float fi; std::memcpy(&fi, i, 4); uboData[idx] = fi; idx += 4; }
                } else if (auto* v3 = std::get_if<Vector3>(&val)) {
                    if (idx + 4 <= uboFloats) { uboData[idx] = v3->x; uboData[idx+1] = v3->y; uboData[idx+2] = v3->z; idx += 4; }
                } else if (auto* v3p = std::get_if<Vector3*>(&val)) {
                    if (*v3p && idx + 4 <= uboFloats) { uboData[idx] = (*v3p)->x; uboData[idx+1] = (*v3p)->y; uboData[idx+2] = (*v3p)->z; idx += 4; }
                } else if (auto* m4 = std::get_if<Matrix4>(&val)) {
                    if (idx + 16 <= uboFloats) { std::memcpy(&uboData[idx], m4->elements.data(), 16 * sizeof(float)); idx += 16; }
                } else if (auto* m4p = std::get_if<Matrix4*>(&val)) {
                    if (*m4p && idx + 16 <= uboFloats) { std::memcpy(&uboData[idx], (*m4p)->elements.data(), 16 * sizeof(float)); idx += 16; }
                } else if (auto* col = std::get_if<Color>(&val)) {
                    if (idx + 4 <= uboFloats) { uboData[idx] = col->r; uboData[idx+1] = col->g; uboData[idx+2] = col->b; idx += 4; }
                }
            }
            customUniformBuf = bufferPool->acquire(uboBytes, kUniformUsage, uboData.data());
        }

        // Build unified texture list from both customTextures and Texture* uniforms
        wgpu::TextureList unifiedTextures;
        for (auto& [name, ptr] : sm->customTextures) {
            auto* gpuTex = static_cast<WgpuTexture*>(ptr);
            unifiedTextures.push_back({name, {gpuTex->view(), gpuTex->sampler()}});
        }
        for (auto& [name, uniform] : sm->uniforms) {
            if (!uniform.hasValue()) continue;
            auto& val = const_cast<Uniform&>(uniform).value();
            if (auto* texPtr = std::get_if<Texture*>(&val)) {
                if (!*texPtr) continue;
                // Check render targets first (e.g. mirror texture)
                auto* rtEntry = renderTargets->findByTextureId((*texPtr)->id);
                if (rtEntry) {
                    unifiedTextures.push_back({name, {rtEntry->colorView, rtEntry->colorSampler}});
                } else {
                    auto& entry = textures->getOrCreateTexture(*texPtr);
                    unifiedTextures.push_back({name, {entry.view, entry.sampler}});
                }
            }
        }
        // Sort and deduplicate by name
        std::sort(unifiedTextures.begin(), unifiedTextures.end(),
                  [](auto& a, auto& b) { return a.first < b.first; });
        unifiedTextures.erase(
                std::unique(unifiedTextures.begin(), unifiedTextures.end(),
                            [](auto& a, auto& b) { return a.first == b.first; }),
                unifiedTextures.end());

        // Build bind group entries via subsystem
        auto& entries = bindGroups->buildCustom(perDrawTransformBuf, renderLightBuffer_,
                                                 lights->lightUniformSize(),
                                                 customUniformBuf, pe.customUniformSize,
                                                 sm, unifiedTextures);

        WGPUBindGroup bg = bindGroupCache->get(
                mesh, sm,
                pe.bindGroupLayout, entries.data(), static_cast<uint32_t>(entries.size()),
                WGPUStringView{"custom_bg", sizeof("custom_bg") - 1});

        wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        auto& gb = geometries->getOrCreateGeometryBuffers(geometry.get());
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                  gb.vertexCount * wgpu::VERTEX_STRIDE);
            if (gb.indexBuffer) {
                wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                     WGPUIndexFormat_Uint32, 0,
                                                     gb.indexCount * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, gb.indexCount, 1, 0, 0, 0);
                renderInfo.calls++;
                renderInfo.triangles += gb.indexCount / 3;
            } else {
                wgpuRenderPassEncoderDraw(pass, gb.vertexCount, 1, 0, 0);
                renderInfo.calls++;
                renderInfo.triangles += gb.vertexCount / 3;
            }
        }

        // Bind group is owned by the cache; do not release here.
    }

    // Collect all renderable objects into the render list with z-depth for sorting.
    void collectRenderables(Object3D& object, const Matrix4& projScreenMatrix,
                            Camera& camera, unsigned int groupOrder) {
        if (!object.visible) return;

        if (object.is<Group>()) {
            groupOrder = object.renderOrder;
        } else if (auto lod = object.as<LOD>()) {
            if (lod->autoUpdate) lod->update(camera);
        } else if (auto sprite = object.as<Sprite>()) {
            if (!object.frustumCulled || frustum_.intersectsSprite(*sprite)) {
                if (scope.sortObjects) {
                    _vector3.setFromMatrixPosition(*sprite->matrixWorld)
                            .applyMatrix4(projScreenMatrix);
                }
                auto material = sprite->material().get();
                if (material && material->visible) {
                    renderList_.push(sprite, getSpriteGeometry(), material, groupOrder, _vector3.z, std::nullopt);
                }
            }
        } else if (object.is<Mesh>() || object.is<Line>() || object.is<Points>()) {
            if (!object.frustumCulled || frustum_.intersectsObject(object)) {
                if (scope.sortObjects) {
                    _vector3.setFromMatrixPosition(*object.matrixWorld)
                            .applyMatrix4(projScreenMatrix);
                }

                auto* owm = object.as<ObjectWithMaterials>();
                if (owm) {
                    auto geometry = owm->geometry();
                    if (geometry && geometry->hasAttribute("position")) {
                        const auto& materials = owm->materials();
                        if (materials.size() > 1) {
                            const auto& groups = geometry->groups;
                            for (const auto& group : groups) {
                                auto groupMat = materials.at(group.materialIndex).get();
                                if (groupMat && groupMat->visible) {
                                    renderList_.push(&object, geometry.get(), groupMat,
                                                     groupOrder, _vector3.z, group);
                                }
                            }
                        } else if (!materials.empty() && materials.front()->visible) {
                            renderList_.push(&object, geometry.get(), materials.front().get(),
                                             groupOrder, _vector3.z, std::nullopt);
                        }
                    }
                }
            }
        }

        for (auto& child : object.children) {
            collectRenderables(*child, projScreenMatrix, camera, groupOrder);
        }
    }

    // Render a single item from the render list.
    void renderItem(WGPURenderPassEncoder pass, const RenderItem* item,
                    const Matrix4& projectionMatrix, const Matrix4& viewMatrix,
                    const Camera& camera, const wgpu::FrameContext& frameCtx) {

        auto* object = item->object;
        auto* geometry = item->geometry;
        Material* rawMat = item->material;
        if (!object || !rawMat || !geometry) return;

        // Extract material parameters via WgpuMaterials subsystem
        auto params = wgpu::extractMaterialParams(rawMat, geometry, currentSceneEnv_);
        if (params.skip) return;

        // Custom WGSL shader path (ShaderMaterial)
        if (params.isCustomShader) {
            if (auto* sm = dynamic_cast<ShaderMaterial*>(rawMat)) {
                if (auto* mesh = object->as<Mesh>()) {
                    renderCustomShaderObject(pass, mesh, sm, projectionMatrix, viewMatrix, camera);
                    return;
                }
            }
        }

        uint64_t features = params.features;

        // Object type
        bool isSprite = object->is<Sprite>();
        bool isMesh = object->is<Mesh>();
        bool isLine = object->is<Line>();
        bool isPoints = object->is<Points>();
        bool isLineSegments = object->is<LineSegments>();
        auto* instancedMesh = object->as<InstancedMesh>();
        auto* skinnedMesh = object->as<SkinnedMesh>();

        // Topology
        bool isLineLoop = object->is<LineLoop>();
        if (isLine) {
            if (object->is<LineSegments>() || isLineLoop) {
                features |= SF::TopoLineList;
            } else {
                features |= SF::TopoLineStrip;
            }
        } else if (isPoints) {
            features |= SF::TopoPointList;
        }

        // Face culling
        switch (rawMat->side) {
            case Side::Front: features |= SF::CullBack; break;
            case Side::Back:  features |= SF::CullFront; break;
            case Side::Double: features |= SF::CullNone; break;
        }

        // Wireframe
        bool useWireframe = false;
        if (isMesh) {
            if (auto wf = dynamic_cast<MaterialWithWireframe*>(rawMat)) {
                if (wf->wireframe) {
                    features |= SF::Wireframe;
                    useWireframe = true;
                }
            }
        }

        // Blend mode
        auto blendVal = static_cast<int>(rawMat->blending);
        if (blendVal == 0)              features |= SF::BlendDisabled;
        else if (blendVal == 2)         features |= SF::BlendAdditive;
        else if (blendVal == 3)         features |= SF::BlendSubtractive;
        else if (blendVal == 4)         features |= SF::BlendMultiply;
        else                            features |= SF::BlendNormal;
        if (!rawMat->depthWrite) {
            features |= SF::DepthWriteOff;
        }

        // Instancing
        if (instancedMesh) {
            features |= SF::Instanced;
            if (instancedMesh->instanceColor()) {
                features |= SF::InstanceColor;
            }
        }

        // Skinning
        if (skinnedMesh && skinnedMesh->skeleton &&
            geometry->hasAttribute("skinIndex") && geometry->hasAttribute("skinWeight")) {
            features |= SF::Skinning;
        }

        // Morph targets
        auto* morphMat = dynamic_cast<MaterialWithMorphTargets*>(rawMat);
        if (morphMat && morphMat->morphTargets && geometry->getMorphAttributes().count("position") > 0) {
            auto mesh = object->as<Mesh>();
            if (mesh && !mesh->morphTargetInfluences().empty()) {
                features |= SF::MorphTargets;
            }
        }

        // Shadow
        if (isMesh && frameCtx.shadowActive && object->receiveShadow &&
            SF::isLit(features)) {
            features |= SF::Shadow;
        }

        // Fog, tone mapping, output encoding
        if (rawMat->fog) {
            features |= frameCtx.fogBits;
        }
        // Per-object tone mapping/sRGB — only set when rendering to a render target.
        features |= frameCtx.tonemapBits;
        if (frameCtx.srgbOutput) {
            features |= SF::SRGBOutput;
        }

        // Get/create pipeline
        auto& pe = pipelines->getOrCreatePipeline(features, surfaceFormat, effectiveSampleCount_);
        if (!pe.pipeline) return;

        // Upload transform uniforms
        float transformData[wgpu::TRANSFORM_UNIFORM_SIZE / sizeof(float)];
        std::memset(transformData, 0, sizeof(transformData));

        // Sprites use a billboard model matrix (always faces the camera)
        Matrix4 modelMatrix;
        if (isSprite) {
            auto* sprite = object->as<Sprite>();
            Vector3 spritePos;
            spritePos.setFromMatrixPosition(*object->matrixWorld);
            Vector3 spriteScale;
            spriteScale.setFromMatrixScale(*object->matrixWorld);
            modelMatrix.extractRotation(*camera.matrixWorld);

            // Apply center offset — matches the GL sprite shader:
            // alignedPosition = (position.xy - (center - 0.5)) * scale
            // We shift the model position so the quad's anchor matches center.
            if (sprite) {
                Vector3 right, up;
                right.setFromMatrixColumn(modelMatrix, 0);
                up.setFromMatrixColumn(modelMatrix, 1);
                float cx = sprite->center.x - 0.5f;
                float cy = sprite->center.y - 0.5f;
                spritePos.sub(right * (cx * spriteScale.x));
                spritePos.sub(up * (cy * spriteScale.y));
            }

            // Apply SpriteMaterial rotation (Z rotation in billboard local space = screen-space rotation)
            if (auto* sm = dynamic_cast<SpriteMaterial*>(rawMat)) {
                if (sm->rotation != 0.0f) {
                    Matrix4 rotZ;
                    rotZ.makeRotationZ(sm->rotation);
                    modelMatrix.multiply(rotZ);
                }
            }

            modelMatrix.scale(spriteScale);
            modelMatrix.setPosition(spritePos);
        } else {
            modelMatrix.copy(*object->matrixWorld);
        }

        std::memcpy(transformData, modelMatrix.elements.data(), 64);
        std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);
        std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64);

        Matrix3 normalMatrix;
        normalMatrix.setFromMatrix4(modelMatrix);
        normalMatrix.invert();
        normalMatrix.transpose();
        auto& ne = normalMatrix.elements;
        transformData[48] = ne[0]; transformData[49] = ne[1]; transformData[50] = ne[2]; transformData[51] = 0;
        transformData[52] = ne[3]; transformData[53] = ne[4]; transformData[54] = ne[5]; transformData[55] = 0;
        transformData[56] = ne[6]; transformData[57] = ne[7]; transformData[58] = ne[8]; transformData[59] = 0;

        Vector3 camPos;
        camPos.setFromMatrixPosition(*camera.matrixWorld);
        transformData[60] = camPos.x;
        transformData[61] = camPos.y;
        transformData[62] = camPos.z;
        transformData[63] = uniformPad_;

        // Per-draw transform buffer — must be a unique buffer per draw call because it
        // contains the view and projection matrices (camera-dependent data).  A persistent
        // per-object buffer would be overwritten by any subsequent render() call that uses
        // a different camera (e.g. water mirror pass) before the encoder is submitted,
        // causing both passes to see the last-written camera matrices.
        constexpr auto kUniformUsage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer perDrawTransform = bufferPool->acquire(wgpu::TRANSFORM_UNIFORM_SIZE, kUniformUsage, transformData);

        // Persistent per-material uniform buffer — only re-upload when data changes.
        float matData[wgpu::MATERIAL_UNIFORM_SIZE / sizeof(float)];
        wgpu::packMaterialUniforms(matData, params, frameCtx, rawMat);
        if (!rawMat->hasEventListener("dispose", onMaterialDisposeForUniforms)) {
            rawMat->addEventListener("dispose", onMaterialDisposeForUniforms);
            trackedMaterialsForUniforms_.insert(rawMat);
        }
        auto& matEntry = perMaterialCache_[rawMat->id];
        if (!matEntry.materialBuf) {
            WGPUBufferDescriptor bd{};
            bd.size = wgpu::MATERIAL_UNIFORM_SIZE;
            bd.usage = kUniformUsage;
            matEntry.materialBuf = wgpuDeviceCreateBuffer(device, &bd);
        }
        if (std::memcmp(matEntry.materialData, matData, wgpu::MATERIAL_UNIFORM_SIZE) != 0) {
            wgpuQueueWriteBuffer(queue, matEntry.materialBuf, 0, matData, wgpu::MATERIAL_UNIFORM_SIZE);
            std::memcpy(matEntry.materialData, matData, wgpu::MATERIAL_UNIFORM_SIZE);
        }
        WGPUBuffer perDrawMaterial = matEntry.materialBuf;

        // Instance data buffer — persistent per-mesh, only re-uploaded on version change.
        WGPUBuffer instanceBuffer = nullptr;
        size_t instanceBufSize = 0;
        if ((features & SF::Instanced) && instancedMesh) {
            bool hasColor = (features & SF::InstanceColor) != 0;
            size_t instanceCount = instancedMesh->count();
            size_t floatsPerInstance = hasColor ? 20 : 16;
            size_t bufSize = instanceCount * floatsPerInstance * sizeof(float);

            auto* matAttr = instancedMesh->instanceMatrix();
            auto* colAttr = instancedMesh->instanceColor();
            uint32_t matVer = matAttr ? matAttr->version : 0;
            uint32_t colVer = (hasColor && colAttr) ? colAttr->version : 0;

            // Register dispose listener on first encounter
            if (!instancedMesh->hasEventListener("dispose", onInstancedMeshDispose)) {
                instancedMesh->addEventListener("dispose", onInstancedMeshDispose);
                trackedInstancedMeshes_.insert(instancedMesh);
            }

            auto& entry = instanceCache_[instancedMesh->id];
            constexpr auto kStorageUsage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

            bool needsUpload = !entry.buffer
                               || entry.bufferSize != bufSize
                               || entry.hasColor != hasColor
                               || entry.matrixVersion != matVer
                               || entry.colorVersion != colVer;

            if (needsUpload) {
                // (Re)allocate persistent buffer if size or color-layout changed
                if (!entry.buffer || entry.bufferSize != bufSize || entry.hasColor != hasColor) {
                    if (entry.buffer) wgpuBufferRelease(entry.buffer);
                    WGPUBufferDescriptor bd{};
                    bd.size = bufSize;
                    bd.usage = kStorageUsage;
                    entry.buffer = wgpuDeviceCreateBuffer(device, &bd);
                    entry.bufferSize = bufSize;
                }

                std::vector<float> instanceData(instanceCount * floatsPerInstance, 0.0f);
                for (size_t i = 0; i < instanceCount; i++) {
                    const float* src = &matAttr->array()[i * 16];
                    float* dst = &instanceData[i * floatsPerInstance];
                    std::memcpy(dst, src, 16 * sizeof(float));
                    if (hasColor && colAttr) {
                        const float* csrc = &colAttr->array()[i * 3];
                        dst[16] = csrc[0];
                        dst[17] = csrc[1];
                        dst[18] = csrc[2];
                        dst[19] = 1.0f;
                    }
                }
                wgpuQueueWriteBuffer(queue, entry.buffer, 0, instanceData.data(), bufSize);

                entry.hasColor = hasColor;
                entry.matrixVersion = matVer;
                entry.colorVersion = colVer;
            }

            instanceBuffer = entry.buffer;
            instanceBufSize = bufSize;
        }

        // Morph target data buffer
        WGPUBuffer morphBuffer = nullptr;
        size_t morphBufSize = 0;
        if (features & SF::MorphTargets) {
            auto mesh = object->as<Mesh>();
            auto& morphAttrs = geometry->getMorphAttributes().at("position");
            uint32_t numTargets = static_cast<uint32_t>(morphAttrs.size());
            uint32_t vertexCount = static_cast<uint32_t>(geometry->getAttribute<float>("position")->count());
            auto& influences = mesh->morphTargetInfluences();

            size_t headerSize = 4;
            size_t influenceSize = 8;
            size_t posSize = numTargets * vertexCount * 4;
            size_t totalFloats = headerSize + influenceSize + posSize;
            std::vector<float> morphData(totalFloats, 0.0f);

            auto* u32Data = reinterpret_cast<uint32_t*>(morphData.data());
            u32Data[0] = numTargets;

            for (uint32_t t = 0; t < numTargets && t < 8; t++) {
                if (t < influences.size()) {
                    morphData[headerSize + t] = influences[t];
                }
            }

            size_t posOffset = headerSize + influenceSize;
            for (uint32_t t = 0; t < numTargets; t++) {
                auto* attr = dynamic_cast<TypedBufferAttribute<float>*>(morphAttrs[t].get());
                if (!attr) continue;
                for (uint32_t v = 0; v < vertexCount; v++) {
                    size_t idx = posOffset + (t * vertexCount + v) * 4;
                    morphData[idx + 0] = attr->getX(v);
                    morphData[idx + 1] = attr->getY(v);
                    morphData[idx + 2] = attr->getZ(v);
                    morphData[idx + 3] = 0.0f;
                }
            }

            size_t bufSize = totalFloats * sizeof(float);
            constexpr auto kStorageUsage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            morphBuffer = bufferPool->acquire(bufSize, kStorageUsage, morphData.data());
            morphBufSize = bufSize;
        }

        // Skinning data buffers
        WGPUBuffer skinBuffer = nullptr;
        size_t skinBufSize = 0;
        WGPUBuffer skinVertexBuffer = nullptr;
        size_t skinVertexBufSize = 0;
        if ((features & SF::Skinning) && skinnedMesh && skinnedMesh->skeleton) {
            auto& skel = *skinnedMesh->skeleton;
            skel.update();

            uint32_t boneCount = static_cast<uint32_t>(skel.bones.size());
            size_t headerFloats = 16 + 16 + 4;
            size_t totalFloats = headerFloats + boneCount * 16;
            std::vector<float> skinData(totalFloats, 0.0f);

            std::memcpy(skinData.data(), skinnedMesh->bindMatrix.elements.data(), 64);
            std::memcpy(skinData.data() + 16, skinnedMesh->bindMatrixInverse.elements.data(), 64);
            auto* u32ptr = reinterpret_cast<uint32_t*>(skinData.data() + 32);
            u32ptr[0] = boneCount;
            if (!skel.boneMatrices.empty()) {
                std::memcpy(skinData.data() + headerFloats, skel.boneMatrices.data(),
                            boneCount * 16 * sizeof(float));
            }

            size_t bufSize = totalFloats * sizeof(float);
            constexpr auto kStorageUsage2 = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            skinBuffer = bufferPool->acquire(bufSize, kStorageUsage2, skinData.data());
            skinBufSize = bufSize;

            uint32_t vertexCount = static_cast<uint32_t>(geometry->getAttribute<float>("position")->count());
            auto* skinIdxAttr = geometry->getAttribute<float>("skinIndex");
            auto* skinWgtAttr = geometry->getAttribute<float>("skinWeight");
            size_t vertFloats = vertexCount * 8;
            std::vector<float> vertData(vertFloats, 0.0f);
            for (uint32_t v = 0; v < vertexCount; v++) {
                vertData[v * 8 + 0] = skinIdxAttr->getX(v);
                vertData[v * 8 + 1] = skinIdxAttr->getY(v);
                vertData[v * 8 + 2] = skinIdxAttr->getZ(v);
                vertData[v * 8 + 3] = skinIdxAttr->getW(v);
                vertData[v * 8 + 4] = skinWgtAttr->getX(v);
                vertData[v * 8 + 5] = skinWgtAttr->getY(v);
                vertData[v * 8 + 6] = skinWgtAttr->getZ(v);
                vertData[v * 8 + 7] = skinWgtAttr->getW(v);
            }
            size_t vertBufSize = vertFloats * sizeof(float);
            skinVertexBuffer = bufferPool->acquire(vertBufSize, kStorageUsage2, vertData.data());
            skinVertexBufSize = vertBufSize;
        }

        // Build bind group entries via subsystem
        wgpu::BindGroupInputs bgInputs{
            .features = features,
            .transformBuffer = perDrawTransform,
            .materialBuffer = perDrawMaterial,
            .lightBuffer = renderLightBuffer_,
            .lightUniformSize = lights->lightUniformSize(),
            .params = params,
            .textures = *textures,
            .shadowMap = shadowMap.get(),
            .shadowUniformSize = wgpuState.shadowLimits.shadowUniformSize(),
            .pointShadowUniformSize = wgpuState.shadowLimits.pointShadowUniformSize(),
            .instanceBuffer = instanceBuffer, .instanceSize = instanceBufSize,
            .morphBuffer = morphBuffer, .morphSize = morphBufSize,
            .skinBuffer = skinBuffer, .skinSize = skinBufSize,
            .skinVertexBuffer = skinVertexBuffer, .skinVertexSize = skinVertexBufSize,
            .transmissionTexView = transmissionTexView_, .transmissionSampler = transmissionSampler_,
        };
        auto& entries = bindGroups->buildStandard(bgInputs);

        WGPUBindGroup bg = bindGroupCache->get(
                object, rawMat,
                pe.bindGroupLayout, entries.data(), static_cast<uint32_t>(entries.size()),
                WGPUStringView{"obj_bg", sizeof("obj_bg") - 1});

        wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        auto& gb = geometries->getOrCreateGeometryBuffers(geometry);
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                     gb.vertexCount * wgpu::VERTEX_STRIDE);

            uint32_t instanceCount = instancedMesh ? static_cast<uint32_t>(instancedMesh->count()) : 1;

            if (useWireframe) {
                auto& wb = geometries->getOrCreateWireframeBuffers(geometry);
                if (wb.indexBuffer) {
                    wgpuRenderPassEncoderSetIndexBuffer(pass, wb.indexBuffer,
                                                         WGPUIndexFormat_Uint32, 0,
                                                         wb.indexCount * sizeof(uint32_t));
                    wgpuRenderPassEncoderDrawIndexed(pass, wb.indexCount, instanceCount, 0, 0, 0);
                    renderInfo.calls++;
                    renderInfo.lines += wb.indexCount / 2;
                }
            } else if (isLineLoop) {
                uint32_t n = gb.vertexCount;
                std::vector<uint32_t> loopIndices;
                loopIndices.reserve(n * 2);
                for (uint32_t i = 0; i < n; i++) {
                    loopIndices.push_back(i);
                    loopIndices.push_back((i + 1) % n);
                }
                size_t loopBufSize = loopIndices.size() * sizeof(uint32_t);
                constexpr auto kIndexUsage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
                WGPUBuffer loopBuf = bufferPool->acquire(loopBufSize, kIndexUsage, loopIndices.data());
                wgpuRenderPassEncoderSetIndexBuffer(pass, loopBuf,
                                                     WGPUIndexFormat_Uint32, 0, loopBufSize);
                uint32_t drawCount = static_cast<uint32_t>(loopIndices.size());
                wgpuRenderPassEncoderDrawIndexed(pass, drawCount, instanceCount, 0, 0, 0);
                renderInfo.calls++;
                renderInfo.lines += n;
            } else if (gb.indexBuffer) {
                uint32_t drawStart = 0;
                uint32_t drawCount = gb.indexCount;
                if (item->group.has_value()) {
                    drawStart = static_cast<uint32_t>(item->group->start);
                    drawCount = static_cast<uint32_t>(item->group->count);
                }
                wgpuRenderPassEncoderSetIndexBuffer(pass, gb.indexBuffer,
                                                         WGPUIndexFormat_Uint32, 0,
                                                         gb.indexCount * sizeof(uint32_t));
                wgpuRenderPassEncoderDrawIndexed(pass, drawCount, instanceCount, drawStart, 0, 0);
                renderInfo.calls++;
                if (isLine) renderInfo.lines += isLineSegments ? drawCount / 2 : (drawCount > 0 ? drawCount - 1 : 0);
                else if (isPoints) renderInfo.points += drawCount;
                else renderInfo.triangles += drawCount / 3;
            } else {
                uint32_t drawCount = gb.vertexCount;
                if (item->group.has_value()) {
                    drawCount = static_cast<uint32_t>(item->group->count);
                }
                wgpuRenderPassEncoderDraw(pass, drawCount, instanceCount, 0, 0);
                renderInfo.calls++;
                if (isLine) renderInfo.lines += isLineSegments ? drawCount / 2 : (drawCount > 0 ? drawCount - 1 : 0);
                else if (isPoints) renderInfo.points += drawCount;
                else renderInfo.triangles += drawCount / 3;
            }
        }

        // Bind group is owned by the cache; do not release here.
    }

    void dispose() {
        if (!initialized) return;

        // Unregister all dispose listeners before destroying so objects/materials disposed
        // after the renderer don't call back into freed memory.
        for (auto* mat : trackedMaterials_) {
            mat->removeEventListener("dispose", onMaterialDispose);
        }
        trackedMaterials_.clear();

        for (auto* mat : trackedMaterialsForUniforms_) {
            mat->removeEventListener("dispose", onMaterialDisposeForUniforms);
        }
        trackedMaterialsForUniforms_.clear();

        for (auto* mesh : trackedInstancedMeshes_) {
            mesh->removeEventListener("dispose", onInstancedMeshDispose);
        }
        trackedInstancedMeshes_.clear();

        if (transmissionTexView_)  { wgpuTextureViewRelease(transmissionTexView_);     transmissionTexView_ = nullptr; }
        if (transmissionTex_)     { wgpuTextureRelease(transmissionTex_);             transmissionTex_ = nullptr; }
        if (transmissionSampler_) { wgpuSamplerRelease(transmissionSampler_);         transmissionSampler_ = nullptr; }
        if (envBgTexView_)        { wgpuTextureViewRelease(envBgTexView_);            envBgTexView_ = nullptr; }
        if (envBgTex_)            { wgpuTextureRelease(envBgTex_);                    envBgTex_ = nullptr; }
        if (envBgPipeline_)       { wgpuRenderPipelineRelease(envBgPipeline_);       envBgPipeline_ = nullptr; }
        if (envBgPipelineLayout_) { wgpuPipelineLayoutRelease(envBgPipelineLayout_); envBgPipelineLayout_ = nullptr; }
        if (envBgBGL_)            { wgpuBindGroupLayoutRelease(envBgBGL_);            envBgBGL_ = nullptr; }
        if (envBgShader_)         { wgpuShaderModuleRelease(envBgShader_);            envBgShader_ = nullptr; }

        if (retainedFB) { wgpuTextureRelease(retainedFB); retainedFB = nullptr; }
        if (retainBlit_.pipeline)       { wgpuRenderPipelineRelease(retainBlit_.pipeline);       retainBlit_.pipeline = nullptr; }
        if (retainBlit_.pipelineLayout) { wgpuPipelineLayoutRelease(retainBlit_.pipelineLayout); retainBlit_.pipelineLayout = nullptr; }
        if (retainBlit_.bgl)            { wgpuBindGroupLayoutRelease(retainBlit_.bgl);            retainBlit_.bgl = nullptr; }
        if (retainBlit_.sampler)        { wgpuSamplerRelease(retainBlit_.sampler);               retainBlit_.sampler = nullptr; }
        if (retainBlit_.shader)         { wgpuShaderModuleRelease(retainBlit_.shader);            retainBlit_.shader = nullptr; }

        for (auto& [id, entry] : instanceCache_) {
            if (entry.buffer) wgpuBufferRelease(entry.buffer);
        }
        instanceCache_.clear();

        for (auto& [id, entry] : perMaterialCache_) {
            if (entry.materialBuf) wgpuBufferRelease(entry.materialBuf);
        }
        perMaterialCache_.clear();

        // Release any active frame state without presenting (safe for dispose)
        releaseFrame();
        releaseCachedFrameTextures();
        if (toneMap_.colorView) wgpuTextureViewRelease(toneMap_.colorView);
        if (toneMap_.colorTexture) wgpuTextureRelease(toneMap_.colorTexture);
        if (toneMap_.depthView) wgpuTextureViewRelease(toneMap_.depthView);
        if (toneMap_.depthTexture) wgpuTextureRelease(toneMap_.depthTexture);
        if (toneMap_.cachedBindGroup)  { wgpuBindGroupRelease(toneMap_.cachedBindGroup);  }
        if (toneMap_.cachedInputView)  { wgpuTextureViewRelease(toneMap_.cachedInputView); }
        for (auto& p : toneMap_.pipelines) { if (p) wgpuRenderPipelineRelease(p); }
        for (auto& m : toneMap_.shaderModules) { if (m) wgpuShaderModuleRelease(m); }
        if (toneMap_.sampler) wgpuSamplerRelease(toneMap_.sampler);
        if (toneMap_.uniformBuf) wgpuBufferRelease(toneMap_.uniformBuf);
        if (toneMap_.pipelineLayout) wgpuPipelineLayoutRelease(toneMap_.pipelineLayout);
        if (toneMap_.bindGroupLayout) wgpuBindGroupLayoutRelease(toneMap_.bindGroupLayout);
        toneMap_ = {};

        if (clearPipeline_) { wgpuRenderPipelineRelease(clearPipeline_); clearPipeline_ = nullptr; }
        if (clearPipelineLayout_) { wgpuPipelineLayoutRelease(clearPipelineLayout_); clearPipelineLayout_ = nullptr; }
        if (clearBGL_) { wgpuBindGroupLayoutRelease(clearBGL_); clearBGL_ = nullptr; }
        if (clearShader_) { wgpuShaderModuleRelease(clearShader_); clearShader_ = nullptr; }

        if (bindGroupCache) bindGroupCache->dispose();
        if (bufferPool) bufferPool->dispose();
        if (geometries) geometries->dispose();
        if (textures) textures->dispose();
        if (pipelines) pipelines->dispose();
        if (shadowMap) shadowMap->dispose();
        if (lights) lights->dispose();
        if (renderTargets) renderTargets->dispose();

        if (queue) wgpuQueueRelease(queue);
        if (device) wgpuDeviceRelease(device);
        if (adapter) wgpuAdapterRelease(adapter);
        if (surface) wgpuSurfaceRelease(surface);
        if (instance) wgpuInstanceRelease(instance);

        initialized = false;
    }

    ~Impl() {
        dispose();
    }
};


// --- WgpuRenderer public API ---

WgpuRenderer::WgpuRenderer(Canvas& canvas) {
    canvas.initWindow(GraphicsAPI::WebGPU);
    pimpl_ = std::make_unique<Impl>(*this, canvas);
    canvas.setFrameEndCallback([this] { pimpl_->endFrame(); });
}

void WgpuRenderer::render(Object3D& scene, Camera& camera) {
    pimpl_->render(scene, camera);
}

void WgpuRenderer::endFrame() {
    pimpl_->endFrame();
}

WindowSize WgpuRenderer::size() const {
    return pimpl_->size_;
}

void WgpuRenderer::setSize(const std::pair<int, int>& size) {
    // pimpl_->canvas.setSize(size);
    pimpl_->size_ = {size.first, size.second};
    setViewport(0, 0, size.first, size.second);
    pimpl_->scissor_ = {0, 0,
                        static_cast<uint32_t>(size.first),
                        static_cast<uint32_t>(size.second)};
    if (pimpl_->initialized) {
        pimpl_->configureSurface();
    }
}

float WgpuRenderer::getTargetPixelRatio() const {
    return pimpl_->pixelRatio_;
}

void WgpuRenderer::setPixelRatio(float value) {
    pimpl_->pixelRatio_ = value;
    setSize({pimpl_->size_.width(), pimpl_->size_.height()});
}

void WgpuRenderer::setPixelRatioHint(float value) {
    pimpl_->uniformPad_ = value;
}

void WgpuRenderer::setViewport(const Vector4& v) {
    pimpl_->viewport_.x = v.x; pimpl_->viewport_.y = v.y;
    pimpl_->viewport_.w = v.z; pimpl_->viewport_.h = v.w;
    pimpl_->viewportExplicit_ = true;
}

void WgpuRenderer::setViewport(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->viewport_.x = std::floor(x * pr);
    pimpl_->viewport_.y = std::floor(y * pr);
    pimpl_->viewport_.w = std::floor(width * pr);
    pimpl_->viewport_.h = std::floor(height * pr);
    pimpl_->viewportExplicit_ = true;
}

void WgpuRenderer::setScissor(const Vector4& v) {
    pimpl_->scissor_.x = static_cast<uint32_t>(v.x);
    pimpl_->scissor_.y = static_cast<uint32_t>(v.y);
    pimpl_->scissor_.w = static_cast<uint32_t>(v.z);
    pimpl_->scissor_.h = static_cast<uint32_t>(v.w);
}

void WgpuRenderer::setScissor(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->scissor_.x = static_cast<uint32_t>(std::floor(x * pr));
    pimpl_->scissor_.y = static_cast<uint32_t>(std::floor(y * pr));
    pimpl_->scissor_.w = static_cast<uint32_t>(std::floor(width * pr));
    pimpl_->scissor_.h = static_cast<uint32_t>(std::floor(height * pr));
}

void WgpuRenderer::getViewport(Vector4& target) const {
    target.set(pimpl_->viewport_.x, pimpl_->viewport_.y, pimpl_->viewport_.w, pimpl_->viewport_.h);
}

void WgpuRenderer::setScissorTest(bool boolean) {
    pimpl_->scissorTest_ = boolean;
}

bool WgpuRenderer::getScissorTest() const {
    return pimpl_->scissorTest_;
}

void WgpuRenderer::getScissor(Vector4& target) const {
    target.set(static_cast<float>(pimpl_->scissor_.x), static_cast<float>(pimpl_->scissor_.y),
               static_cast<float>(pimpl_->scissor_.w), static_cast<float>(pimpl_->scissor_.h));
}

void WgpuRenderer::setClearColor(const Color& color, float alpha) {
    pimpl_->clearColor_ = color;
    pimpl_->clearAlpha_ = alpha;
}

void WgpuRenderer::getClearColor(Color& target) const {
    target = pimpl_->clearColor_;
}

float WgpuRenderer::getClearAlpha() const {
    return pimpl_->clearAlpha_;
}

void WgpuRenderer::setClearAlpha(float alpha) {
    pimpl_->clearAlpha_ = alpha;
}

void WgpuRenderer::clear(bool color, bool depth, bool stencil) {
    if (pimpl_->frame_.active) {
        // Mid-frame: execute an immediate clear pass on the current surface
        pimpl_->executeClear(color, depth, stencil);
    } else {
        // No active frame yet: set pending flags for the next render pass
        pimpl_->pendingClearColor_ |= color;
        pimpl_->pendingClearDepth_ |= depth;
        pimpl_->pendingClearStencil_ |= stencil;
    }
}

void WgpuRenderer::clearColor() {
    clear(true, false, false);
}

void WgpuRenderer::clearDepth() {
    clear(false, true, false);
}

void WgpuRenderer::clearStencil() {
    clear(false, false, true);
}

RenderTarget* WgpuRenderer::getRenderTarget() {
    return pimpl_->currentRenderTarget_;
}

void WgpuRenderer::setRenderTarget(RenderTarget* renderTarget, int activeCubeFace, int activeMipmapLevel) {
    pimpl_->currentRenderTarget_ = renderTarget;
    pimpl_->activeCubeFace_ = activeCubeFace;
    pimpl_->activeMipmapLevel_ = activeMipmapLevel;
    pimpl_->viewportExplicit_ = false;
}

int WgpuRenderer::getActiveCubeFace() const {
    return pimpl_->activeCubeFace_;
}

int WgpuRenderer::getActiveMipmapLevel() const {
    return pimpl_->activeMipmapLevel_;
}

const WgpuInfo& WgpuRenderer::info() const {
    static WgpuInfo di;
    di.render.frame = pimpl_->renderInfo.frame;
    di.render.calls = pimpl_->renderInfo.calls;
    di.render.triangles = pimpl_->renderInfo.triangles;
    di.render.lines = pimpl_->renderInfo.lines;
    di.render.points = pimpl_->renderInfo.points;
    di.memory.geometries = pimpl_->renderInfo.geometries;
    di.memory.textures = pimpl_->renderInfo.textures;
    return di;
}

void* WgpuRenderer::nativeDevice() const {
    return pimpl_->device;
}

void* WgpuRenderer::nativeQueue() const {
    return pimpl_->queue;
}

void* WgpuRenderer::nativeInstance() const {
    return pimpl_->instance;
}

void* WgpuRenderer::nativeSurface() const {
    return pimpl_->surface;
}

void WgpuRenderer::setOverlayCallback(std::function<void(void*)> cb) {
    pimpl_->overlayCallback_ = std::move(cb);
}

void* WgpuRenderer::nativeRenderTargetTexture() const {
    if (!pimpl_->currentRenderTarget_) return nullptr;
    auto& rt = pimpl_->renderTargets->getOrCreate(pimpl_->currentRenderTarget_, pimpl_->sampleCount_);
    return static_cast<void*>(rt.colorTexture);
}

uint32_t WgpuRenderer::nativeSurfaceFormat() const {
    return static_cast<uint32_t>(pimpl_->surfaceFormat);
}

void* WgpuRenderer::nativeFrameDepthView() const {
    if (!pimpl_->frame_.active) return nullptr;
    if (pimpl_->needsToneMapPass()) return static_cast<void*>(pimpl_->toneMap_.depthView);
    return static_cast<void*>(pimpl_->frame_.depthView);
}

uint32_t WgpuRenderer::nativeFrameDepthSampleCount() const {
    if (!pimpl_->frame_.active) return 1;
    return pimpl_->needsToneMapPass() ? 1u : pimpl_->sampleCount_;
}

void* WgpuRenderer::nativeRenderCommandEncoder() const {
    return static_cast<void*>(pimpl_->renderEncoder_);
}

void WgpuRenderer::setSampleCount(uint32_t count) {
    if (count != 1 && count != 4) {
        std::cerr << "WgpuRenderer::setSampleCount: unsupported count " << count
                  << " (must be 1 or 4)" << std::endl;
        return;
    }
    if (pimpl_->sampleCount_ == count) return;

    pimpl_->sampleCount_ = count;

    // Invalidate pipeline cache — pipelines encode the multisample count
    if (pimpl_->pipelines) pimpl_->pipelines->invalidateAll();

    // Invalidate render target cache — textures have wrong sample count
    if (pimpl_->renderTargets) pimpl_->renderTargets->invalidateAll();
}

uint32_t WgpuRenderer::getSampleCount() const {
    return pimpl_->sampleCount_;
}

void WgpuRenderer::setMaxLights(int maxDir, int maxPoint, int maxSpot, int maxHemi) {
    auto& lim = pimpl_->wgpuState.lightLimits;
    if (lim.maxDirLights == maxDir && lim.maxPointLights == maxPoint &&
        lim.maxSpotLights == maxSpot && lim.maxHemiLights == maxHemi)
        return;

    lim.maxDirLights = maxDir;
    lim.maxPointLights = maxPoint;
    lim.maxSpotLights = maxSpot;
    lim.maxHemiLights = maxHemi;

    // Recreate light GPU buffer with new size
    if (pimpl_->lights) pimpl_->lights->recreateBuffer();

    // Invalidate pipelines — shaders embed array sizes from LightLimits
    if (pimpl_->pipelines) pimpl_->pipelines->invalidateAll();
}

void WgpuRenderer::setShadowConfig(uint32_t mapSize, int maxShadowLights, int maxShadowPointLights) {
    auto& sl = pimpl_->wgpuState.shadowLimits;
    if (sl.mapSize == mapSize && sl.maxShadowLights == maxShadowLights &&
        sl.maxShadowPointLights == maxShadowPointLights)
        return;

    sl.mapSize = mapSize;
    sl.maxShadowLights = maxShadowLights;
    sl.maxShadowPointLights = maxShadowPointLights;

    // Dispose old shadow GPU resources — they will be re-created on next init()
    if (pimpl_->shadowMap) pimpl_->shadowMap->dispose();

    // Invalidate pipelines — shaders encode shadow array sizes
    if (pimpl_->pipelines) pimpl_->pipelines->invalidateAll();
}

void WgpuRenderer::resetState() {
    // Wgpu manages its own state; this is a no-op for API compatibility
}

std::vector<unsigned char> WgpuRenderer::readRGBPixels() {
    if (!pimpl_->initialized || !pimpl_->currentRenderTarget_) return {};

    auto& rt = pimpl_->renderTargets->getOrCreate(pimpl_->currentRenderTarget_, pimpl_->sampleCount_);
    return wgpu::readRGBPixels(pimpl_->device, pimpl_->queue, rt.colorTexture, rt.width, rt.height);
}

void WgpuRenderer::readPixels(const Vector2& position, const std::pair<int, int>& sz,
                              std::vector<unsigned char>& data) {
    auto allPixels = readRGBPixels();
    if (allPixels.empty()) return;

    auto& rt = pimpl_->renderTargets->getOrCreate(pimpl_->currentRenderTarget_, pimpl_->sampleCount_);
    int rtW = static_cast<int>(rt.width);
    int rtH = static_cast<int>(rt.height);
    int x0 = static_cast<int>(position.x);
    int y0 = static_cast<int>(position.y);
    int w = sz.first;
    int h = sz.second;

    data.resize(w * h * 3);
    for (int row = 0; row < h; row++) {
        int srcY = y0 + row;
        if (srcY < 0 || srcY >= rtH) continue;
        for (int col = 0; col < w; col++) {
            int srcX = x0 + col;
            if (srcX < 0 || srcX >= rtW) continue;
            size_t srcIdx = (srcY * rtW + srcX) * 3;
            size_t dstIdx = (row * w + col) * 3;
            data[dstIdx + 0] = allPixels[srcIdx + 0];
            data[dstIdx + 1] = allPixels[srcIdx + 1];
            data[dstIdx + 2] = allPixels[srcIdx + 2];
        }
    }
}

void WgpuRenderer::copyFramebufferToTexture(const Vector2& position, Texture& texture, int level) {
    if (!pimpl_->initialized) return;

    // Enable framebuffer retention for future frames
    pimpl_->retainFramebuffer = true;

    // Source: retained framebuffer (or current render target's color texture)
    WGPUTexture srcTexture = nullptr;
    uint32_t srcW = 0, srcH = 0;

    if (pimpl_->currentRenderTarget_) {
        auto& rt = pimpl_->renderTargets->getOrCreate(pimpl_->currentRenderTarget_, pimpl_->sampleCount_);
        srcTexture = rt.colorTexture;
        srcW = rt.width;
        srcH = rt.height;
    } else if (pimpl_->retainedFB) {
        srcTexture = pimpl_->retainedFB;
        srcW = pimpl_->retainedFBWidth;
        srcH = pimpl_->retainedFBHeight;
    }

    if (!srcTexture) return;

    // Ensure the destination texture exists on the GPU
    auto& dstEntry = pimpl_->textures->getOrCreateTexture(&texture);
    if (!dstEntry.texture) return;

    auto& img = texture.image();
    uint32_t texW = img.width;
    uint32_t texH = img.height;
    if (texW == 0 || texH == 0) return;

    // Clamp copy region to source bounds
    uint32_t sx = static_cast<uint32_t>((std::max)(0.f, position.x));
    uint32_t sy = static_cast<uint32_t>((std::max)(0.f, position.y));
    uint32_t copyW = (std::min)(texW, srcW > sx ? srcW - sx : 0u);
    uint32_t copyH = (std::min)(texH, srcH > sy ? srcH - sy : 0u);
    if (copyW == 0 || copyH == 0) return;

    // GPU copy from retained framebuffer to destination texture
    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = WGPUStringView{"copy_fb", sizeof("copy_fb") - 1};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pimpl_->device, &encDesc);

    WGPUTexelCopyTextureInfo src{};
    src.texture = srcTexture;
    src.mipLevel = 0;
    src.origin = {sx, sy, 0};
    src.aspect = WGPUTextureAspect_All;

    WGPUTexelCopyTextureInfo dst{};
    dst.texture = dstEntry.texture;
    dst.mipLevel = static_cast<uint32_t>(level);
    dst.origin = {0, 0, 0};
    dst.aspect = WGPUTextureAspect_All;

    WGPUExtent3D extent = {copyW, copyH, 1};
    wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = WGPUStringView{"copy_fb_cmd", sizeof("copy_fb_cmd") - 1};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(pimpl_->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
}

void WgpuRenderer::copyTextureToImage(Texture& texture) {
    if (!pimpl_->initialized) return;

    auto* entry = pimpl_->textures->findTexture(texture.id);
    if (!entry || !entry->texture) return;

    auto& image = texture.image();
    uint32_t w = image.width;
    uint32_t h = image.height;
    if (w == 0 || h == 0) return;

    // readRGBPixels returns RGB data from a BGRA8 GPU texture
    auto rgb = wgpu::readRGBPixels(pimpl_->device, pimpl_->queue, entry->texture, w, h);
    if (rgb.empty()) return;

    auto& data = image.data();
    // Determine the channel count the texture expects
    size_t srcChannels = 3;// readRGBPixels always returns RGB
    size_t dstChannels = data.size() / (static_cast<size_t>(w) * h);
    if (dstChannels == 0) dstChannels = srcChannels;

    data.resize(static_cast<size_t>(w) * h * dstChannels);

    size_t npixels = static_cast<size_t>(w) * h;
    if (dstChannels == 3) {
        std::memcpy(data.data(), rgb.data(), npixels * 3);
    } else if (dstChannels == 4) {
        for (size_t i = 0; i < npixels; i++) {
            data[i * 4 + 0] = rgb[i * 3 + 0];
            data[i * 4 + 1] = rgb[i * 3 + 1];
            data[i * 4 + 2] = rgb[i * 3 + 2];
            data[i * 4 + 3] = 255;
        }
    } else {
        // Fewer channels — copy first N from each RGB pixel
        for (size_t i = 0; i < npixels; i++) {
            for (size_t c = 0; c < dstChannels; c++) {
                data[i * dstChannels + c] = rgb[i * 3 + c];
            }
        }
    }
}

void WgpuRenderer::writeFramebuffer(const std::filesystem::path& filename) {
    auto ext = filename.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") {
        throw std::runtime_error("Unsupported file format: " + ext);
    }

    auto pixels = readRGBPixels();
    if (pixels.empty()) return;

    int w = pimpl_->size_.width();
    int h = pimpl_->size_.height();

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
        throw std::runtime_error("WgpuRenderer: failed to write framebuffer to " + filename.string());
    }
}

void WgpuRenderer::dispose() {
    pimpl_->dispose();
}

WgpuRenderer::~WgpuRenderer() = default;
