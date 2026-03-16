
#include "threepp/renderers/DawnRenderer.hpp"

#include "dawn/DawnBufferPool.hpp"
#include "dawn/DawnGeometries.hpp"
#include "dawn/DawnMaterials.hpp"
#include "dawn/DawnPipelines.hpp"
#include "dawn/DawnShaders.hpp"
#include "dawn/DawnShadowMap.hpp"
#include "dawn/DawnState.hpp"
#include "dawn/DawnTextures.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/constants.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/dawn/GPUTexture.hpp"
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
#ifdef __linux__
#define GLFW_EXPOSE_NATIVE_X11
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

// stb_image_write — implementation is already compiled in GLRenderer.cpp.
// Match the linkage used by the implementation (extern "C" in C++).
#define STBIWDEF extern "C"
#include "stb_image_write.h"
#undef STBIWDEF

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;
namespace SF = threepp::dawn::ShaderFeatures;

#ifdef __APPLE__
extern "C" void* dawn_create_metal_layer(void* nsWindow);
#endif

namespace {

    // Maximum time to wait for async WebGPU operations before aborting.
    constexpr auto WGPU_ASYNC_TIMEOUT = std::chrono::seconds(10);

}// namespace


struct DawnRenderer::Impl {

    // Listener for material dispose events — evicts stale pipeline cache entries.
    struct OnMaterialDispose : EventListener {
        Impl* scope_;
        explicit OnMaterialDispose(Impl* s) : scope_(s) {}
        void onEvent(Event& event) override {
            auto* mat = std::any_cast<Material*>(event.target);
            mat->removeEventListener("dispose", *this);
            if (scope_->pipelines) {
                scope_->pipelines->onMaterialDispose(mat->id);
            }
        }
    };

    DawnRenderer& scope;
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
    Color clearColor_{0x000000};
    float clearAlpha_ = 1.0f;

    // MSAA configuration
    uint32_t sampleCount_ = 1;

    // Viewport & scissor state
    struct { float x=0, y=0, w=0, h=0; } viewport_;
    struct { uint32_t x=0, y=0, w=0, h=0; } scissor_;
    bool scissorTest_ = false;

    // Uniform buffers
    WGPUBuffer lightBuffer = nullptr;       // binding 2 (per-frame, shared)

    // Subsystem: shared state
    dawn::DawnState dawnState;

    // Subsystem: geometry buffer management (with version-based updates)
    std::unique_ptr<dawn::DawnGeometries> geometries;

    // Subsystem: texture upload, caching, version tracking
    std::unique_ptr<dawn::DawnTextures> textures;

    // Subsystem: pipeline creation and caching
    std::unique_ptr<dawn::DawnPipelines> pipelines;

    // Subsystem: shadow map rendering
    std::unique_ptr<dawn::DawnShadowMap> shadowMap;

    // Subsystem: per-frame buffer pool (avoids per-draw GPU buffer alloc/free)
    std::unique_ptr<dawn::DawnBufferPool> bufferPool;

    // Material dispose listener
    OnMaterialDispose onMaterialDispose;

    // Render target cache
    struct RTEntry {
        WGPUTexture colorTexture = nullptr;       // resolve target (1x) — readback source
        WGPUTextureView colorView = nullptr;
        WGPUTexture depthTexture = nullptr;
        WGPUTextureView depthView = nullptr;
        // MSAA textures (only created when sampleCount > 1)
        WGPUTexture msaaColorTexture = nullptr;   // multi-sampled render target
        WGPUTextureView msaaColorView = nullptr;
        WGPUTexture msaaDepthTexture = nullptr;
        WGPUTextureView msaaDepthView = nullptr;
        unsigned int width = 0, height = 0;
        uint32_t sampleCount = 1;
    };
    std::unordered_map<std::string, RTEntry> rtCache;
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

    explicit Impl(DawnRenderer& scope, Canvas& canvas)
        : scope(scope), canvas(canvas), size_(canvas.size()),
          onMaterialDispose(this) {

        viewport_.w = static_cast<float>(size_.width());
        viewport_.h = static_cast<float>(size_.height());
        scissor_.w = static_cast<uint32_t>(size_.width());
        scissor_.h = static_cast<uint32_t>(size_.height());

        initWebGPU();
    }

    void initWebGPU() {
        // Create instance with primary backends (Vulkan/Metal/DX12).
        // Avoid GL backend as it conflicts with GLFW's GL context.
        WGPUInstanceExtras instanceExtras{};
        instanceExtras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
        instanceExtras.chain.next = nullptr;
        instanceExtras.backends = WGPUInstanceBackend_Primary;

        WGPUInstanceDescriptor instanceDesc{};
        instanceDesc.nextInChain = &instanceExtras.chain;
        instance = wgpuCreateInstance(&instanceDesc);
        if (!instance) {
            std::cerr << "DawnRenderer: Failed to create WebGPU instance" << std::endl;
            return;
        }

        createSurface();
        requestAdapter();
        if (!adapter) {
            std::cerr << "DawnRenderer: Failed to get adapter" << std::endl;
            return;
        }

        requestDevice();
        if (!device) {
            std::cerr << "DawnRenderer: Failed to get device" << std::endl;
            return;
        }

        queue = wgpuDeviceGetQueue(device);

        if (surface) {
            configureSurface();
        }

        // Populate shared state for subsystems
        dawnState.device = device;
        dawnState.queue = queue;
        dawnState.surfaceFormat = surfaceFormat;

        // Create uniform buffers
        createUniformBuffers();

        // Initialize subsystems
        textures = std::make_unique<dawn::DawnTextures>(dawnState);
        textures->createDummyTexture();
        geometries = std::make_unique<dawn::DawnGeometries>(dawnState);
        pipelines = std::make_unique<dawn::DawnPipelines>(dawnState);
        shadowMap = std::make_unique<dawn::DawnShadowMap>(dawnState, *geometries);
        bufferPool = std::make_unique<dawn::DawnBufferPool>(device, queue);

        initialized = true;
        std::cout << "DawnRenderer: WebGPU initialized successfully"
                  << (surface ? "" : " (headless, no surface)") << std::endl;
    }

    void createSurface() {
        auto* glfwWindow = static_cast<GLFWwindow*>(canvas.windowPtr());
        WGPUSurfaceDescriptor surfDesc{};
        WGPUStringView label = {.data = "threepp_surface", .length = 15};
        surfDesc.label = label;

#if defined(__linux__)
        Display* x11Display = glfwGetX11Display();
        ::Window x11Window = glfwGetX11Window(glfwWindow);

        WGPUSurfaceSourceXlibWindow xlibSource{};
        xlibSource.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        xlibSource.chain.next = nullptr;
        xlibSource.display = x11Display;
        xlibSource.window = static_cast<uint64_t>(x11Window);
        surfDesc.nextInChain = &xlibSource.chain;

#elif defined(_WIN32)
        WGPUSurfaceSourceWindowsHWND hwndSource{};
        hwndSource.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
        hwndSource.chain.next = nullptr;
        hwndSource.hinstance = GetModuleHandle(nullptr);
        hwndSource.hwnd = glfwGetWin32Window(glfwWindow);
        surfDesc.nextInChain = &hwndSource.chain;

#elif defined(__APPLE__)
        void* metalLayer = dawn_create_metal_layer(glfwGetCocoaWindow(glfwWindow));
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

        WGPURequestAdapterCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestAdapterStatus_Success) {
                ud->adapter = adapter;
            } else {
                std::cerr << "DawnRenderer: Adapter request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuInstanceRequestAdapter(instance, &options, callbackInfo);

        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("DawnRenderer: requestAdapter timed out");
            }
            wgpuInstanceProcessEvents(instance);
        }

        if (!userData.adapter) {
            throw std::runtime_error("DawnRenderer: failed to obtain adapter");
        }
        adapter = userData.adapter;
    }

    void requestDevice() {
        struct UserData {
            WGPUDevice device = nullptr;
            bool done = false;
        } userData;

        WGPUDeviceDescriptor deviceDesc{};
        WGPUStringView label = {.data = "threepp_device", .length = 14};
        deviceDesc.label = label;

        WGPURequestDeviceCallbackInfo callbackInfo{};
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                                    WGPUStringView message, void* userdata1, void* /*userdata2*/) {
            auto* ud = static_cast<UserData*>(userdata1);
            if (status == WGPURequestDeviceStatus_Success) {
                ud->device = device;
            } else {
                std::cerr << "DawnRenderer: Device request failed: "
                          << std::string_view(message.data, message.length) << std::endl;
            }
            ud->done = true;
        };
        callbackInfo.userdata1 = &userData;

        wgpuAdapterRequestDevice(adapter, &deviceDesc, callbackInfo);

        auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
        while (!userData.done) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("DawnRenderer: requestDevice timed out");
            }
            wgpuInstanceProcessEvents(instance);
        }

        if (!userData.device) {
            throw std::runtime_error("DawnRenderer: failed to obtain device");
        }
        device = userData.device;
    }

    void configureSurface() {
        WGPUSurfaceConfiguration config{};
        config.device = device;
        config.format = surfaceFormat;
        config.usage = WGPUTextureUsage_RenderAttachment;
        config.width = static_cast<uint32_t>(std::floor(size_.width() * pixelRatio_));
        config.height = static_cast<uint32_t>(std::floor(size_.height() * pixelRatio_));
        config.presentMode = WGPUPresentMode_Fifo;
        config.alphaMode = WGPUCompositeAlphaMode_Auto;
        config.viewFormatCount = 0;
        config.viewFormats = nullptr;

        wgpuSurfaceConfigure(surface, &config);
    }

    void createUniformBuffers() {
        WGPUBufferDescriptor d{};
        d.label = {.data = "light_buf", .length = 9};
        d.size = dawn::LIGHT_UNIFORM_SIZE;
        d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        lightBuffer = wgpuDeviceCreateBuffer(device, &d);
    }

    // Render target helpers
    void releaseRTEntry(RTEntry& e) {
        if (e.msaaColorView) wgpuTextureViewRelease(e.msaaColorView);
        if (e.msaaColorTexture) wgpuTextureRelease(e.msaaColorTexture);
        if (e.msaaDepthView) wgpuTextureViewRelease(e.msaaDepthView);
        if (e.msaaDepthTexture) wgpuTextureRelease(e.msaaDepthTexture);
        if (e.colorView) wgpuTextureViewRelease(e.colorView);
        if (e.colorTexture) wgpuTextureRelease(e.colorTexture);
        if (e.depthView) wgpuTextureViewRelease(e.depthView);
        if (e.depthTexture) wgpuTextureRelease(e.depthTexture);
    }

    RTEntry& getOrCreateRT(RenderTarget* rt) {
        auto it = rtCache.find(rt->uuid);
        if (it != rtCache.end() && it->second.width == rt->width
            && it->second.height == rt->height && it->second.sampleCount == sampleCount_) {
            return it->second;
        }
        if (it != rtCache.end()) {
            releaseRTEntry(it->second);
        }

        RTEntry entry{};
        entry.width = rt->width;
        entry.height = rt->height;
        entry.sampleCount = sampleCount_;

        WGPUTextureDescriptor ctd{};
        ctd.label = {.data = "rt_color", .length = 8};
        ctd.size = {rt->width, rt->height, 1};
        ctd.mipLevelCount = 1;
        ctd.sampleCount = 1;
        ctd.dimension = WGPUTextureDimension_2D;
        ctd.format = surfaceFormat;
        ctd.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
        entry.colorTexture = wgpuDeviceCreateTexture(device, &ctd);
        entry.colorView = wgpuTextureCreateView(entry.colorTexture, nullptr);

        WGPUTextureDescriptor dtd{};
        dtd.label = {.data = "rt_depth", .length = 8};
        dtd.size = {rt->width, rt->height, 1};
        dtd.mipLevelCount = 1;
        dtd.sampleCount = sampleCount_;
        dtd.dimension = WGPUTextureDimension_2D;
        dtd.format = WGPUTextureFormat_Depth24Plus;
        dtd.usage = WGPUTextureUsage_RenderAttachment;
        entry.depthTexture = wgpuDeviceCreateTexture(device, &dtd);
        entry.depthView = wgpuTextureCreateView(entry.depthTexture, nullptr);

        if (sampleCount_ > 1) {
            WGPUTextureDescriptor msaaCtd{};
            msaaCtd.label = {.data = "rt_msaa_color", .length = 13};
            msaaCtd.size = {rt->width, rt->height, 1};
            msaaCtd.mipLevelCount = 1;
            msaaCtd.sampleCount = sampleCount_;
            msaaCtd.dimension = WGPUTextureDimension_2D;
            msaaCtd.format = surfaceFormat;
            msaaCtd.usage = WGPUTextureUsage_RenderAttachment;
            entry.msaaColorTexture = wgpuDeviceCreateTexture(device, &msaaCtd);
            entry.msaaColorView = wgpuTextureCreateView(entry.msaaColorTexture, nullptr);
        }

        rtCache[rt->uuid] = entry;
        return rtCache[rt->uuid];
    }

    // Pack light data into GPU buffer using world-space coordinates.
    void uploadLightDataWorldSpace(Object3D& scene) {
        std::vector<float> data(dawn::LIGHT_UNIFORM_SIZE / sizeof(float), 0.0f);
        auto* u32 = reinterpret_cast<uint32_t*>(data.data());

        uint32_t nDir = 0, nPt = 0, nSp = 0, nHm = 0;
        float ambR = 0, ambG = 0, ambB = 0;

        struct DirEntry { Vector3 dir; Color col; };
        struct PtEntry  { Vector3 pos; Color col; float dist; float decay; };
        struct SpEntry  { Vector3 pos; Vector3 dir; Color col; float dist; float decay; float coneCos; float penumbraCos; };
        struct HmEntry  { Vector3 dir; Color sky; Color gnd; };
        std::vector<DirEntry> dirs;
        std::vector<PtEntry>  pts;
        std::vector<SpEntry>  sps;
        std::vector<HmEntry>  hms;

        std::function<void(Object3D&)> collect = [&](Object3D& obj) {
            if (auto al = obj.as<AmbientLight>()) {
                ambR += al->color.r * al->intensity;
                ambG += al->color.g * al->intensity;
                ambB += al->color.b * al->intensity;
            } else if (auto dl = obj.as<DirectionalLight>()) {
                if (dirs.size() < static_cast<size_t>(dawn::MAX_DIR_LIGHTS)) {
                    Vector3 lightPos, targetPos;
                    lightPos.setFromMatrixPosition(*dl->matrixWorld);
                    targetPos.setFromMatrixPosition(*dl->target().matrixWorld);
                    Vector3 direction = lightPos.clone().sub(targetPos).normalize();
                    dirs.push_back({direction, Color(dl->color).multiplyScalar(dl->intensity)});
                }
            } else if (auto pl = obj.as<PointLight>()) {
                if (pts.size() < static_cast<size_t>(dawn::MAX_POINT_LIGHTS)) {
                    Vector3 pos;
                    pos.setFromMatrixPosition(*pl->matrixWorld);
                    pts.push_back({pos, Color(pl->color).multiplyScalar(pl->intensity), pl->distance, pl->decay});
                }
            } else if (auto sl = obj.as<SpotLight>()) {
                if (sps.size() < static_cast<size_t>(dawn::MAX_SPOT_LIGHTS)) {
                    Vector3 pos, targetPos;
                    pos.setFromMatrixPosition(*sl->matrixWorld);
                    targetPos.setFromMatrixPosition(*sl->target().matrixWorld);
                    Vector3 direction = pos.clone().sub(targetPos).normalize();
                    sps.push_back({pos, direction, Color(sl->color).multiplyScalar(sl->intensity),
                                   sl->distance, sl->decay,
                                   std::cos(sl->angle), std::cos(sl->angle * (1.0f - sl->penumbra))});
                }
            } else if (auto hl = obj.as<HemisphereLight>()) {
                if (hms.size() < static_cast<size_t>(dawn::MAX_HEMI_LIGHTS)) {
                    Vector3 dir;
                    dir.setFromMatrixPosition(*hl->matrixWorld).normalize();
                    hms.push_back({dir, Color(hl->color).multiplyScalar(hl->intensity),
                                   Color(hl->groundColor).multiplyScalar(hl->intensity)});
                }
            }
            for (auto& child : obj.children) collect(*child);
        };
        collect(scene);

        nDir = dirs.size(); nPt = pts.size(); nSp = sps.size(); nHm = hms.size();
        u32[0] = nDir; u32[1] = nPt; u32[2] = nSp; u32[3] = nHm;
        data[4] = ambR; data[5] = ambG; data[6] = ambB; data[7] = 0;

        size_t off = 8;
        for (uint32_t i = 0; i < nDir; i++) {
            data[off+0] = dirs[i].dir.x; data[off+1] = dirs[i].dir.y; data[off+2] = dirs[i].dir.z; data[off+3] = 0;
            data[off+4] = dirs[i].col.r; data[off+5] = dirs[i].col.g; data[off+6] = dirs[i].col.b; data[off+7] = 0;
            off += 8;
        }

        off = 8 + dawn::MAX_DIR_LIGHTS * 8;
        for (uint32_t i = 0; i < nPt; i++) {
            data[off+0] = pts[i].pos.x; data[off+1] = pts[i].pos.y; data[off+2] = pts[i].pos.z; data[off+3] = 0;
            data[off+4] = pts[i].col.r; data[off+5] = pts[i].col.g; data[off+6] = pts[i].col.b; data[off+7] = pts[i].dist;
            data[off+8] = pts[i].decay; data[off+9] = 0; data[off+10] = 0; data[off+11] = 0;
            off += 12;
        }

        off = 8 + dawn::MAX_DIR_LIGHTS * 8 + dawn::MAX_POINT_LIGHTS * 12;
        for (uint32_t i = 0; i < nSp; i++) {
            data[off+0] = sps[i].pos.x; data[off+1] = sps[i].pos.y; data[off+2] = sps[i].pos.z; data[off+3] = 0;
            data[off+4] = sps[i].dir.x; data[off+5] = sps[i].dir.y; data[off+6] = sps[i].dir.z; data[off+7] = 0;
            data[off+8] = sps[i].col.r; data[off+9] = sps[i].col.g; data[off+10] = sps[i].col.b; data[off+11] = sps[i].dist;
            data[off+12] = sps[i].decay; data[off+13] = sps[i].coneCos; data[off+14] = sps[i].penumbraCos; data[off+15] = 0;
            off += 16;
        }

        off = 8 + dawn::MAX_DIR_LIGHTS * 8 + dawn::MAX_POINT_LIGHTS * 12 + dawn::MAX_SPOT_LIGHTS * 16;
        for (uint32_t i = 0; i < nHm; i++) {
            data[off+0] = hms[i].dir.x; data[off+1] = hms[i].dir.y; data[off+2] = hms[i].dir.z; data[off+3] = 0;
            data[off+4] = hms[i].sky.r; data[off+5] = hms[i].sky.g; data[off+6] = hms[i].sky.b; data[off+7] = 0;
            data[off+8] = hms[i].gnd.r; data[off+9] = hms[i].gnd.g; data[off+10] = hms[i].gnd.b; data[off+11] = 0;
            off += 12;
        }

        wgpuQueueWriteBuffer(queue, lightBuffer, 0, data.data(), dawn::LIGHT_UNIFORM_SIZE);
    }

    void render(Object3D& scene, Camera& camera) {
        if (!initialized) return;

        // Recycle per-draw buffers from previous frame
        bufferPool->beginFrame();

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
            if (surface) configureSurface();
            viewport_.w = static_cast<float>(size_.width());
            viewport_.h = static_cast<float>(size_.height());
            scissor_.w = static_cast<uint32_t>(size_.width());
            scissor_.h = static_cast<uint32_t>(size_.height());
        }

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

        // Upload world-space light data
        uploadLightDataWorldSpace(scene);

        // Shadow pass (delegated to DawnShadowMap subsystem)
        shadowMap->beginFrame(scene);

        // Determine render target views
        WGPUTextureView colorView = nullptr;
        WGPUTextureView depthView = nullptr;
        WGPUTextureView resolveView = nullptr;
        WGPUTexture frameDepthTexture = nullptr;
        WGPUTexture frameMsaaColorTexture = nullptr;
        WGPUSurfaceTexture surfaceTexture{};
        bool useSurface = (currentRenderTarget_ == nullptr && surface != nullptr);

        if (currentRenderTarget_ == nullptr && surface == nullptr) {
            return; // Headless mode with no render target set
        }

        if (useSurface) {
            wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
            if (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
                surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
                std::cerr << "DawnRenderer: Failed to acquire surface texture (status "
                          << static_cast<int>(surfaceTexture.status) << ")" << std::endl;
                return;
            }
            WGPUTextureViewDescriptor vd{};
            vd.label = {.data = "surface_view", .length = 12};
            vd.format = surfaceFormat;
            vd.dimension = WGPUTextureViewDimension_2D;
            vd.baseMipLevel = 0; vd.mipLevelCount = 1;
            vd.baseArrayLayer = 0; vd.arrayLayerCount = 1;
            vd.aspect = WGPUTextureAspect_All;
            uint32_t w = static_cast<uint32_t>(size_.width());
            uint32_t h = static_cast<uint32_t>(size_.height());

            if (sampleCount_ > 1) {
                resolveView = wgpuTextureCreateView(surfaceTexture.texture, &vd);
                WGPUTextureDescriptor msaaCtd{};
                msaaCtd.label = {.data = "frame_msaa_color", .length = 16};
                msaaCtd.size = {w, h, 1};
                msaaCtd.mipLevelCount = 1;
                msaaCtd.sampleCount = sampleCount_;
                msaaCtd.dimension = WGPUTextureDimension_2D;
                msaaCtd.format = surfaceFormat;
                msaaCtd.usage = WGPUTextureUsage_RenderAttachment;
                frameMsaaColorTexture = wgpuDeviceCreateTexture(device, &msaaCtd);
                colorView = wgpuTextureCreateView(frameMsaaColorTexture, nullptr);
            } else {
                colorView = wgpuTextureCreateView(surfaceTexture.texture, &vd);
            }

            WGPUTextureDescriptor dtd{};
            dtd.label = {.data = "depth_tex", .length = 9};
            dtd.size = {w, h, 1};
            dtd.mipLevelCount = 1; dtd.sampleCount = sampleCount_;
            dtd.dimension = WGPUTextureDimension_2D;
            dtd.format = WGPUTextureFormat_Depth24Plus;
            dtd.usage = WGPUTextureUsage_RenderAttachment;
            frameDepthTexture = wgpuDeviceCreateTexture(device, &dtd);
            depthView = wgpuTextureCreateView(frameDepthTexture, nullptr);
        } else {
            auto& rt = getOrCreateRT(currentRenderTarget_);
            depthView = rt.depthView;
            if (sampleCount_ > 1 && rt.msaaColorView) {
                colorView = rt.msaaColorView;
                resolveView = rt.colorView;
            } else {
                colorView = rt.colorView;
            }
        }

        // Command encoder
        WGPUCommandEncoderDescriptor encDesc{};
        encDesc.label = {.data = "cmd_enc", .length = 7};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

        // Determine clear color
        auto* sceneObj = scene.as<Scene>();
        Color effectiveClearColor = clearColor_;
        float effectiveClearAlpha = clearAlpha_;
        if (sceneObj && sceneObj->background.isColor()) {
            effectiveClearColor = sceneObj->background.color();
            effectiveClearAlpha = 1.0f;
        }

        // Render pass
        WGPURenderPassColorAttachment colorAttachment{};
        colorAttachment.view = colorView;
        colorAttachment.resolveTarget = resolveView;
        colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        colorAttachment.loadOp = WGPULoadOp_Clear;
        colorAttachment.storeOp = WGPUStoreOp_Store;
        colorAttachment.clearValue = {
                static_cast<double>(effectiveClearColor.r),
                static_cast<double>(effectiveClearColor.g),
                static_cast<double>(effectiveClearColor.b),
                static_cast<double>(effectiveClearAlpha)};

        WGPURenderPassDepthStencilAttachment depthAttachment{};
        depthAttachment.view = depthView;
        depthAttachment.depthLoadOp = WGPULoadOp_Clear;
        depthAttachment.depthStoreOp = WGPUStoreOp_Store;
        depthAttachment.depthClearValue = 1.0f;

        WGPURenderPassDescriptor passDesc{};
        passDesc.label = {.data = "render_pass", .length = 11};
        passDesc.colorAttachmentCount = 1;
        passDesc.colorAttachments = &colorAttachment;
        passDesc.depthStencilAttachment = &depthAttachment;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);

        wgpuRenderPassEncoderSetViewport(pass, viewport_.x, viewport_.y, viewport_.w, viewport_.h, 0.0f, 1.0f);
        if (scissorTest_) {
            wgpuRenderPassEncoderSetScissorRect(pass, scissor_.x, scissor_.y, scissor_.w, scissor_.h);
        }

        // Set up frustum for culling and collect renderables
        renderList_.init();
        Matrix4 projScreenMatrix;
        projScreenMatrix.multiplyMatrices(projectionMatrix, viewMatrix);
        frustum_.setFromProjectionMatrix(projScreenMatrix);
        collectRenderables(scene, projScreenMatrix, camera, 0);
        if (scope.sortObjects) {
            renderList_.sort();
        }
        renderList_.finish();

        // Build per-frame rendering context
        dawn::FrameContext frameCtx{};
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
        switch (scope.toneMapping) {
            case ToneMapping::Linear: frameCtx.tonemapBits = SF::TonemapLinear; break;
            case ToneMapping::Reinhard: frameCtx.tonemapBits = SF::TonemapReinhard; break;
            case ToneMapping::Cineon: frameCtx.tonemapBits = SF::TonemapCineon; break;
            case ToneMapping::ACESFilmic: frameCtx.tonemapBits = SF::TonemapACES; break;
            default: break;
        }
        frameCtx.toneMappingExposure = scope.toneMappingExposure;
        frameCtx.localClippingEnabled = scope.localClippingEnabled;
        frameCtx.srgbOutput = (scope.outputEncoding == Encoding::sRGB);
        frameCtx.shadowActive = shadowMap->isActive();

        // Render opaque objects (front-to-back, depth write on)
        for (auto* item : renderList_.opaque) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera, frameCtx);
        }

        // Render transparent objects (back-to-front, depth write off)
        for (auto* item : renderList_.transparent) {
            renderItem(pass, item, projectionMatrix, viewMatrix, camera, frameCtx);
        }

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        // Submit
        WGPUCommandBufferDescriptor cmdDesc{};
        cmdDesc.label = {.data = "cmd_buf", .length = 7};
        WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdDesc);
        wgpuQueueSubmit(queue, 1, &cmdBuffer);

        if (useSurface) {
            wgpuSurfacePresent(surface);
        }

        // Cleanup per-frame resources
        wgpuCommandBufferRelease(cmdBuffer);
        wgpuCommandEncoderRelease(encoder);
        if (useSurface) {
            wgpuTextureViewRelease(depthView);
            wgpuTextureRelease(frameDepthTexture);
            if (resolveView) {
                wgpuTextureViewRelease(colorView);
                wgpuTextureRelease(frameMsaaColorTexture);
                wgpuTextureViewRelease(resolveView);
            } else {
                wgpuTextureViewRelease(colorView);
            }
            wgpuTextureRelease(surfaceTexture.texture);
        }
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
        }

        auto& pe = pipelines->getOrCreateCustomPipeline(sm, surfaceFormat, sampleCount_);
        if (!pe.pipeline) return;

        // Create per-draw transform buffer
        float transformData[dawn::TRANSFORM_UNIFORM_SIZE / sizeof(float)];
        std::memset(transformData, 0, sizeof(transformData));
        std::memcpy(transformData, mesh->matrixWorld->elements.data(), 64);
        std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);
        std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64);

        Matrix4 modelView;
        modelView.multiplyMatrices(viewMatrix, *mesh->matrixWorld);
        Matrix3 normalMatrix;
        normalMatrix.setFromMatrix4(modelView);
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
        transformData[63] = 0;

        constexpr auto kUniformUsage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer perDrawTransformBuf = bufferPool->acquire(dawn::TRANSFORM_UNIFORM_SIZE, kUniformUsage, transformData);

        // Build bind group entries
        std::vector<WGPUBindGroupEntry> entries;
        { WGPUBindGroupEntry e{}; e.binding = 0; e.buffer = perDrawTransformBuf; e.offset = 0; e.size = dawn::TRANSFORM_UNIFORM_SIZE; entries.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 1; e.buffer = lightBuffer; e.offset = 0; e.size = dawn::LIGHT_UNIFORM_SIZE; entries.push_back(e); }

        // Custom uniforms at binding 2
        WGPUBuffer customUniformBuf = nullptr;
        bool hasCustomUniforms = false;
        for (auto& [name, uniform] : sm->uniforms) {
            if (uniform.hasValue()) { hasCustomUniforms = true; break; }
        }
        if (hasCustomUniforms) {
            float uboData[64];
            std::memset(uboData, 0, sizeof(uboData));
            int idx = 0;
            for (auto& [name, uniform] : sm->uniforms) {
                if (!uniform.hasValue()) continue;
                auto& val = uniform.value();
                if (auto* f = std::get_if<float>(&val)) {
                    if (idx < 64) uboData[idx++] = *f;
                } else if (auto* i = std::get_if<int>(&val)) {
                    if (idx < 64) { float fi; std::memcpy(&fi, i, 4); uboData[idx++] = fi; }
                } else if (auto* v3 = std::get_if<Vector3>(&val)) {
                    if (idx + 3 <= 64) { uboData[idx] = v3->x; uboData[idx+1] = v3->y; uboData[idx+2] = v3->z; idx += 4; }
                }
            }
            customUniformBuf = bufferPool->acquire(256, kUniformUsage, uboData);

            WGPUBindGroupEntry e{}; e.binding = 2; e.buffer = customUniformBuf; e.offset = 0; e.size = 256;
            entries.push_back(e);
        }

        // GPU texture bindings
        std::vector<std::string> texNames;
        for (auto& [name, ptr] : sm->customTextures) {
            texNames.push_back(name);
        }
        std::sort(texNames.begin(), texNames.end());

        uint32_t nextBinding = hasCustomUniforms ? 3 : 2;
        for (auto& name : texNames) {
            auto* gpuTex = static_cast<GPUTexture*>(sm->customTextures[name]);
            { WGPUBindGroupEntry e{}; e.binding = nextBinding++; e.textureView = gpuTex->view(); entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = nextBinding++; e.sampler = gpuTex->sampler(); entries.push_back(e); }
        }

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.label = {.data = "custom_bg", .length = 9};
        bgDesc.layout = pe.bindGroupLayout;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

        wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        auto& gb = geometries->getOrCreateGeometryBuffers(geometry.get());
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                  gb.vertexCount * dawn::VERTEX_STRIDE);
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

        wgpuBindGroupRelease(bg);
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
                    renderList_.push(sprite, nullptr, material, groupOrder, _vector3.z, std::nullopt);
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
                    const Camera& camera, const dawn::FrameContext& frameCtx) {

        auto* object = item->object;
        auto* geometry = item->geometry;
        Material* rawMat = item->material;
        if (!object || !rawMat || !geometry) return;

        // Extract material parameters via DawnMaterials subsystem
        auto params = dawn::extractMaterialParams(rawMat, geometry);
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
        if (rawMat->transparent) {
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
        features |= frameCtx.tonemapBits;
        if (frameCtx.srgbOutput) {
            features |= SF::SRGBOutput;
        }

        // Get/create pipeline
        auto& pe = pipelines->getOrCreatePipeline(features, surfaceFormat, sampleCount_);
        if (!pe.pipeline) return;

        // Upload transform uniforms
        float transformData[dawn::TRANSFORM_UNIFORM_SIZE / sizeof(float)];
        std::memset(transformData, 0, sizeof(transformData));
        std::memcpy(transformData, object->matrixWorld->elements.data(), 64);
        std::memcpy(transformData + 16, viewMatrix.elements.data(), 64);
        std::memcpy(transformData + 32, projectionMatrix.elements.data(), 64);

        Matrix4 modelView;
        modelView.multiplyMatrices(viewMatrix, *object->matrixWorld);
        Matrix3 normalMatrix;
        normalMatrix.setFromMatrix4(modelView);
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
        transformData[63] = 0;

        // Acquire per-draw uniform buffers from pool
        constexpr auto kUniformUsage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        WGPUBuffer perDrawTransform = bufferPool->acquire(dawn::TRANSFORM_UNIFORM_SIZE, kUniformUsage, transformData);

        // Pack material uniforms via DawnMaterials subsystem
        float matData[dawn::MATERIAL_UNIFORM_SIZE / sizeof(float)];
        dawn::packMaterialUniforms(matData, params, frameCtx, rawMat);
        WGPUBuffer perDrawMaterial = bufferPool->acquire(dawn::MATERIAL_UNIFORM_SIZE, kUniformUsage, matData);

        // Build bind group
        bool lit = SF::isLit(features);
        bool tex = features & SF::Texture;

        std::vector<WGPUBindGroupEntry> entries;
        { WGPUBindGroupEntry e{}; e.binding = 0; e.buffer = perDrawTransform; e.offset = 0; e.size = dawn::TRANSFORM_UNIFORM_SIZE; entries.push_back(e); }
        { WGPUBindGroupEntry e{}; e.binding = 1; e.buffer = perDrawMaterial; e.offset = 0; e.size = dawn::MATERIAL_UNIFORM_SIZE; entries.push_back(e); }

        if (lit) {
            WGPUBindGroupEntry e{}; e.binding = 2; e.buffer = lightBuffer; e.offset = 0; e.size = dawn::LIGHT_UNIFORM_SIZE; entries.push_back(e);
        }

        auto* texEntry = &textures->getDummyTexture();
        if (tex && params.diffuseMap) {
            texEntry = &textures->getOrCreateTexture(params.diffuseMap);
        }
        if (tex) {
            { WGPUBindGroupEntry e{}; e.binding = 3; e.textureView = texEntry->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 4; e.sampler = texEntry->sampler; entries.push_back(e); }
        }

        if (features & SF::NormalMap) {
            auto* nmEntry = params.normalMap ? &textures->getOrCreateTexture(params.normalMap) : &textures->getDummyTexture();
            { WGPUBindGroupEntry e{}; e.binding = 5; e.textureView = nmEntry->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 6; e.sampler = nmEntry->sampler; entries.push_back(e); }
        }

        if (features & SF::Shadow) {
            { WGPUBindGroupEntry e{}; e.binding = 7; e.buffer = shadowMap->uniformBuffer(); e.offset = 0; e.size = dawn::SHADOW_UNIFORM_SIZE; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 8; e.textureView = shadowMap->depthArrayView(); entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 9; e.sampler = shadowMap->comparisonSampler(); entries.push_back(e); }
        }

        auto addTexEntries = [&](uint32_t texBinding, uint32_t sampBinding, Texture* tex) {
            auto* te = tex ? &textures->getOrCreateTexture(tex) : &textures->getDummyTexture();
            { WGPUBindGroupEntry e{}; e.binding = texBinding; e.textureView = te->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = sampBinding; e.sampler = te->sampler; entries.push_back(e); }
        };

        if (features & SF::EmissiveMap)  addTexEntries(10, 11, params.emissiveMap);
        if (features & SF::RoughnessMap) addTexEntries(12, 13, params.roughnessMap);
        if (features & SF::MetalnessMap) addTexEntries(14, 15, params.metalnessMap);
        if (features & SF::AOMap)        addTexEntries(16, 17, params.aoMap);
        if (features & SF::AlphaMap)     addTexEntries(18, 19, params.alphaMap);
        if (features & SF::SpecularMap)  addTexEntries(20, 21, params.specularMap);
        if (features & SF::LightMap)     addTexEntries(22, 23, params.lightMap);
        if (features & SF::BumpMap)      addTexEntries(24, 25, params.bumpMap);
        if (features & SF::GradientMap)  addTexEntries(26, 27, params.gradientMap);
        if (features & SF::DisplacementMap) addTexEntries(30, 31, params.displacementMap);

        if (features & SF::EnvMap) {
            auto* te = params.envMap ? &textures->getOrCreateCubeTexture(params.envMap) : &textures->getDummyCubeTexture();
            { WGPUBindGroupEntry e{}; e.binding = 32; e.textureView = te->view; entries.push_back(e); }
            { WGPUBindGroupEntry e{}; e.binding = 33; e.sampler = te->sampler; entries.push_back(e); }
        }

        // Instance data buffer
        WGPUBuffer instanceBuffer = nullptr;
        if ((features & SF::Instanced) && instancedMesh) {
            bool hasColor = features & SF::InstanceColor;
            size_t instanceCount = instancedMesh->count();
            size_t floatsPerInstance = hasColor ? 20 : 16;
            size_t bufSize = instanceCount * floatsPerInstance * sizeof(float);
            std::vector<float> instanceData(instanceCount * floatsPerInstance, 0.0f);

            auto* matAttr = instancedMesh->instanceMatrix();
            auto* colAttr = instancedMesh->instanceColor();
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

            constexpr auto kStorageUsage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            instanceBuffer = bufferPool->acquire(bufSize, kStorageUsage, instanceData.data());

            WGPUBindGroupEntry e{};
            e.binding = 28;
            e.buffer = instanceBuffer;
            e.offset = 0;
            e.size = bufSize;
            entries.push_back(e);
        }

        // Morph target data buffer
        WGPUBuffer morphBuffer = nullptr;
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

            WGPUBindGroupEntry e{};
            e.binding = 29;
            e.buffer = morphBuffer;
            e.offset = 0;
            e.size = bufSize;
            entries.push_back(e);
        }

        // Skinning data buffers
        WGPUBuffer skinBuffer = nullptr;
        WGPUBuffer skinVertexBuffer = nullptr;
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

            { WGPUBindGroupEntry e{}; e.binding = 34; e.buffer = skinBuffer; e.offset = 0; e.size = bufSize;
              entries.push_back(e); }

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

            { WGPUBindGroupEntry e{}; e.binding = 35; e.buffer = skinVertexBuffer; e.offset = 0; e.size = vertBufSize;
              entries.push_back(e); }
        }

        WGPUBindGroupDescriptor bgDesc{};
        bgDesc.label = {.data = "obj_bg", .length = 6};
        bgDesc.layout = pe.bindGroupLayout;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

        wgpuRenderPassEncoderSetPipeline(pass, pe.pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);

        auto& gb = geometries->getOrCreateGeometryBuffers(geometry);
        if (gb.vertexBuffer) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, gb.vertexBuffer, 0,
                                                     gb.vertexCount * dawn::VERTEX_STRIDE);

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
                WGPUBufferDescriptor bd{};
                bd.label = {.data = "lineloop_idx", .length = 12};
                bd.size = loopIndices.size() * sizeof(uint32_t);
                bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
                WGPUBuffer loopBuf = wgpuDeviceCreateBuffer(device, &bd);
                wgpuQueueWriteBuffer(queue, loopBuf, 0, loopIndices.data(), bd.size);
                wgpuRenderPassEncoderSetIndexBuffer(pass, loopBuf,
                                                     WGPUIndexFormat_Uint32, 0, bd.size);
                uint32_t drawCount = static_cast<uint32_t>(loopIndices.size());
                wgpuRenderPassEncoderDrawIndexed(pass, drawCount, instanceCount, 0, 0, 0);
                wgpuBufferRelease(loopBuf);
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

        // Bind group is lightweight — release immediately. Buffers are managed by the pool.
        wgpuBindGroupRelease(bg);
    }

    void dispose() {
        if (!initialized) return;

        if (bufferPool) bufferPool->dispose();
        if (geometries) geometries->dispose();
        if (textures) textures->dispose();
        if (pipelines) pipelines->dispose();
        if (shadowMap) shadowMap->dispose();

        for (auto& [id, rt] : rtCache) {
            releaseRTEntry(rt);
        }
        rtCache.clear();

        if (lightBuffer) wgpuBufferRelease(lightBuffer);
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


// --- DawnRenderer public API ---

DawnRenderer::DawnRenderer(Canvas& canvas)
    : pimpl_(std::make_unique<Impl>(*this, canvas)) {}

void DawnRenderer::render(Object3D& scene, Camera& camera) {
    pimpl_->render(scene, camera);
}

WindowSize DawnRenderer::size() const {
    return pimpl_->size_;
}

void DawnRenderer::setSize(const std::pair<int, int>& size) {
    pimpl_->canvas.setSize(size);
    pimpl_->size_ = {size.first, size.second};
    setViewport(0, 0, size.first, size.second);
    if (pimpl_->initialized) {
        pimpl_->configureSurface();
    }
}

float DawnRenderer::getTargetPixelRatio() const {
    return pimpl_->pixelRatio_;
}

void DawnRenderer::setPixelRatio(float value) {
    pimpl_->pixelRatio_ = value;
    setSize({pimpl_->size_.width(), pimpl_->size_.height()});
}

void DawnRenderer::setViewport(const Vector4& v) {
    pimpl_->viewport_.x = v.x; pimpl_->viewport_.y = v.y;
    pimpl_->viewport_.w = v.z; pimpl_->viewport_.h = v.w;
}

void DawnRenderer::setViewport(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->viewport_.x = std::floor(x * pr);
    pimpl_->viewport_.y = std::floor(y * pr);
    pimpl_->viewport_.w = std::floor(width * pr);
    pimpl_->viewport_.h = std::floor(height * pr);
}

void DawnRenderer::setScissor(const Vector4& v) {
    pimpl_->scissor_.x = static_cast<uint32_t>(v.x);
    pimpl_->scissor_.y = static_cast<uint32_t>(v.y);
    pimpl_->scissor_.w = static_cast<uint32_t>(v.z);
    pimpl_->scissor_.h = static_cast<uint32_t>(v.w);
}

void DawnRenderer::setScissor(int x, int y, int width, int height) {
    float pr = pimpl_->pixelRatio_;
    pimpl_->scissor_.x = static_cast<uint32_t>(std::floor(x * pr));
    pimpl_->scissor_.y = static_cast<uint32_t>(std::floor(y * pr));
    pimpl_->scissor_.w = static_cast<uint32_t>(std::floor(width * pr));
    pimpl_->scissor_.h = static_cast<uint32_t>(std::floor(height * pr));
}

void DawnRenderer::getViewport(Vector4& target) const {
    target.set(pimpl_->viewport_.x, pimpl_->viewport_.y, pimpl_->viewport_.w, pimpl_->viewport_.h);
}

void DawnRenderer::setScissorTest(bool boolean) {
    pimpl_->scissorTest_ = boolean;
}

bool DawnRenderer::getScissorTest() const {
    return pimpl_->scissorTest_;
}

void DawnRenderer::getScissor(Vector4& target) const {
    target.set(static_cast<float>(pimpl_->scissor_.x), static_cast<float>(pimpl_->scissor_.y),
               static_cast<float>(pimpl_->scissor_.w), static_cast<float>(pimpl_->scissor_.h));
}

void DawnRenderer::setClearColor(const Color& color, float alpha) {
    pimpl_->clearColor_ = color;
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::getClearColor(Color& target) const {
    target = pimpl_->clearColor_;
}

float DawnRenderer::getClearAlpha() const {
    return pimpl_->clearAlpha_;
}

void DawnRenderer::setClearAlpha(float alpha) {
    pimpl_->clearAlpha_ = alpha;
}

void DawnRenderer::clear(bool /*color*/, bool /*depth*/, bool /*stencil*/) {
    // Clearing happens at render pass begin via loadOp = Clear
}

void DawnRenderer::clearColor() {
    clear(true, false, false);
}

void DawnRenderer::clearDepth() {
    clear(false, true, false);
}

void DawnRenderer::clearStencil() {
    clear(false, false, true);
}

RenderTarget* DawnRenderer::getRenderTarget() {
    return pimpl_->currentRenderTarget_;
}

void DawnRenderer::setRenderTarget(RenderTarget* renderTarget, int activeCubeFace, int activeMipmapLevel) {
    pimpl_->currentRenderTarget_ = renderTarget;
    pimpl_->activeCubeFace_ = activeCubeFace;
    pimpl_->activeMipmapLevel_ = activeMipmapLevel;
}

int DawnRenderer::getActiveCubeFace() const {
    return pimpl_->activeCubeFace_;
}

int DawnRenderer::getActiveMipmapLevel() const {
    return pimpl_->activeMipmapLevel_;
}

const DawnInfo& DawnRenderer::info() const {
    static DawnInfo di;
    di.render.frame = pimpl_->renderInfo.frame;
    di.render.calls = pimpl_->renderInfo.calls;
    di.render.triangles = pimpl_->renderInfo.triangles;
    di.render.lines = pimpl_->renderInfo.lines;
    di.render.points = pimpl_->renderInfo.points;
    di.memory.geometries = pimpl_->renderInfo.geometries;
    di.memory.textures = pimpl_->renderInfo.textures;
    return di;
}

void* DawnRenderer::nativeDevice() const {
    return pimpl_->device;
}

void* DawnRenderer::nativeQueue() const {
    return pimpl_->queue;
}

void* DawnRenderer::nativeInstance() const {
    return pimpl_->instance;
}

void DawnRenderer::setSampleCount(uint32_t count) {
    if (count != 1 && count != 4) {
        std::cerr << "DawnRenderer::setSampleCount: unsupported count " << count
                  << " (must be 1 or 4)" << std::endl;
        return;
    }
    if (pimpl_->sampleCount_ == count) return;

    pimpl_->sampleCount_ = count;

    // Invalidate pipeline cache — pipelines encode the multisample count
    if (pimpl_->pipelines) pimpl_->pipelines->invalidateAll();

    // Invalidate render target cache — textures have wrong sample count
    for (auto& [id, rt] : pimpl_->rtCache) {
        pimpl_->releaseRTEntry(rt);
    }
    pimpl_->rtCache.clear();
}

uint32_t DawnRenderer::getSampleCount() const {
    return pimpl_->sampleCount_;
}

void DawnRenderer::resetState() {
    // Dawn manages its own state; this is a no-op for API compatibility
}

std::vector<unsigned char> DawnRenderer::readRGBPixels() {
    if (!pimpl_->initialized || !pimpl_->currentRenderTarget_) return {};

    auto& rt = pimpl_->getOrCreateRT(pimpl_->currentRenderTarget_);
    uint32_t w = rt.width;
    uint32_t h = rt.height;

    // Row alignment: WebGPU requires bytesPerRow to be a multiple of 256
    uint32_t bytesPerPixel = 4; // BGRA8
    uint32_t unpaddedBytesPerRow = w * bytesPerPixel;
    uint32_t paddedBytesPerRow = ((unpaddedBytesPerRow + 255) / 256) * 256;
    uint32_t bufferSize = paddedBytesPerRow * h;

    WGPUBufferDescriptor bd{};
    bd.label = {.data = "readback_buf", .length = 12};
    bd.size = bufferSize;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer stagingBuf = wgpuDeviceCreateBuffer(pimpl_->device, &bd);

    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = {.data = "readback_enc", .length = 12};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pimpl_->device, &encDesc);

    WGPUTexelCopyTextureInfo src{};
    src.texture = rt.colorTexture;

    WGPUTexelCopyBufferInfo dst{};
    dst.buffer = stagingBuf;
    dst.layout.bytesPerRow = paddedBytesPerRow;
    dst.layout.rowsPerImage = h;

    WGPUExtent3D extent = {w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = {.data = "readback_cmd", .length = 12};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(pimpl_->queue, 1, &cmd);

    struct MapData { bool done = false; WGPUMapAsyncStatus status; } mapData;

    WGPUBufferMapCallbackInfo mapCb{};
    mapCb.mode = WGPUCallbackMode_AllowSpontaneous;
    mapCb.callback = [](WGPUMapAsyncStatus status, WGPUStringView /*msg*/, void* ud1, void* /*ud2*/) {
        auto* d = static_cast<MapData*>(ud1);
        d->status = status;
        d->done = true;
    };
    mapCb.userdata1 = &mapData;
    wgpuBufferMapAsync(stagingBuf, WGPUMapMode_Read, 0, bufferSize, mapCb);

    auto deadline = std::chrono::steady_clock::now() + WGPU_ASYNC_TIMEOUT;
    while (!mapData.done) {
        if (std::chrono::steady_clock::now() > deadline) {
            wgpuBufferRelease(stagingBuf);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
            throw std::runtime_error("DawnRenderer: readRGBPixels buffer map timed out");
        }
        wgpuDevicePoll(pimpl_->device, true, nullptr);
    }

    std::vector<unsigned char> result;
    if (mapData.status == WGPUMapAsyncStatus_Success) {
        auto* mapped = static_cast<const unsigned char*>(wgpuBufferGetConstMappedRange(stagingBuf, 0, bufferSize));
        result.resize(w * h * 3);
        for (uint32_t row = 0; row < h; row++) {
            for (uint32_t col = 0; col < w; col++) {
                const auto* px = mapped + row * paddedBytesPerRow + col * 4;
                size_t outIdx = (row * w + col) * 3;
                // BGRA -> RGB
                result[outIdx + 0] = px[2];
                result[outIdx + 1] = px[1];
                result[outIdx + 2] = px[0];
            }
        }
        wgpuBufferUnmap(stagingBuf);
    }

    wgpuBufferRelease(stagingBuf);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    return result;
}

void DawnRenderer::readPixels(const Vector2& position, const std::pair<int, int>& sz,
                              std::vector<unsigned char>& data) {
    auto allPixels = readRGBPixels();
    if (allPixels.empty()) return;

    auto& rt = pimpl_->getOrCreateRT(pimpl_->currentRenderTarget_);
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

void DawnRenderer::writeFramebuffer(const std::filesystem::path& filename) {
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
        throw std::runtime_error("DawnRenderer: failed to write framebuffer to " + filename.string());
    }
}

void DawnRenderer::dispose() {
    pimpl_->dispose();
}

DawnRenderer::~DawnRenderer() = default;
