
#include "threepp/canvas/EglContext.hpp"

#include "threepp/utils/LoadGlad.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__) && !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#define THREEPP_HAS_EGL 1
#else
#define THREEPP_HAS_EGL 0
#endif

using namespace threepp;

#if THREEPP_HAS_EGL

namespace {

    // EGL is reached through dlopen and a hand-rolled set of declarations
    // rather than <EGL/egl.h>. Deliberately: it keeps EGL out of the build
    // entirely — no headers to find, no library to link, no new dependency on
    // any platform — and the loader is present at runtime on exactly the
    // machines where this class is useful. The same reason vk_extensions.py
    // probes Vulkan through ctypes instead of linking it.
    using EGLDisplay = void*;
    using EGLConfig = void*;
    using EGLSurface = void*;
    using EGLContextHandle = void*;
    using EGLDeviceEXT = void*;
    using EGLAttribKHR = std::intptr_t;
    using EGLint = std::int32_t;
    using EGLenum = unsigned int;
    using EGLBoolean = unsigned int;

    constexpr EGLint EGL_NONE = 0x3038;
    constexpr EGLBoolean EGL_TRUE_ = 1;
    constexpr EGLenum EGL_OPENGL_API = 0x30A2;
    constexpr EGLint EGL_OPENGL_BIT = 0x0008;
    constexpr EGLint EGL_PBUFFER_BIT = 0x0001;
    constexpr EGLint EGL_RENDERABLE_TYPE = 0x3040;
    constexpr EGLint EGL_SURFACE_TYPE = 0x3033;
    constexpr EGLint EGL_RED_SIZE = 0x3024;
    constexpr EGLint EGL_GREEN_SIZE = 0x3023;
    constexpr EGLint EGL_BLUE_SIZE = 0x3022;
    constexpr EGLint EGL_ALPHA_SIZE = 0x3021;
    constexpr EGLint EGL_DEPTH_SIZE = 0x3025;
    constexpr EGLint EGL_STENCIL_SIZE = 0x3026;
    constexpr EGLint EGL_WIDTH = 0x3057;
    constexpr EGLint EGL_HEIGHT = 0x3056;
    constexpr EGLint EGL_CONTEXT_MAJOR_VERSION = 0x3098;
    constexpr EGLint EGL_CONTEXT_MINOR_VERSION = 0x30FB;
    constexpr EGLint EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD;
    constexpr EGLint EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001;
    constexpr EGLenum EGL_PLATFORM_DEVICE_EXT = 0x313F;
    constexpr EGLenum EGL_PLATFORM_SURFACELESS_MESA = 0x31DD;
    constexpr EGLint EGL_SUCCESS = 0x3000;
    constexpr EGLint EGL_VENDOR_STR = 0x3053;
    constexpr EGLint EGL_VERSION_STR = 0x3054;
    // NV_device_cuda: the CUDA ordinal for an EGL device. EGL device order and
    // CUDA device order are different enumerations, and on a multi-GPU node
    // that difference is how a render ends up on a different card than the
    // tensors it is supposed to match.
    constexpr EGLint EGL_CUDA_DEVICE_NV = 0x323A;

    // GL enums, so this file needs no GL header before glad is even loaded.
    constexpr unsigned int GL_VENDOR_ = 0x1F00;
    constexpr unsigned int GL_RENDERER_ = 0x1F01;
    constexpr unsigned int GL_VERSION_ = 0x1F02;

    struct Egl {

        void* lib{nullptr};

        void* (*GetProcAddress)(const char*){nullptr};
        EGLint (*GetError)(){nullptr};
        const char* (*QueryString)(EGLDisplay, EGLint){nullptr};
        EGLBoolean (*Initialize)(EGLDisplay, EGLint*, EGLint*){nullptr};
        EGLBoolean (*Terminate)(EGLDisplay){nullptr};
        EGLBoolean (*BindAPI)(EGLenum){nullptr};
        EGLBoolean (*ChooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*){nullptr};
        EGLSurface (*CreatePbufferSurface)(EGLDisplay, EGLConfig, const EGLint*){nullptr};
        EGLContextHandle (*CreateContext)(EGLDisplay, EGLConfig, EGLContextHandle, const EGLint*){nullptr};
        EGLBoolean (*MakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContextHandle){nullptr};
        EGLContextHandle (*GetCurrentContext)(){nullptr};
        EGLBoolean (*DestroyContext)(EGLDisplay, EGLContextHandle){nullptr};
        EGLBoolean (*DestroySurface)(EGLDisplay, EGLSurface){nullptr};

        // Extension entry points come from eglGetProcAddress, never dlsym — a
        // vendor loader need not export them as symbols at all.
        EGLBoolean (*QueryDevicesEXT)(EGLint, EGLDeviceEXT*, EGLint*){nullptr};
        EGLDisplay (*GetPlatformDisplayEXT)(EGLenum, void*, const EGLint*){nullptr};
        EGLBoolean (*QueryDeviceAttribEXT)(EGLDeviceEXT, EGLint, EGLAttribKHR*){nullptr};

        [[nodiscard]] bool ok() const {

            return lib && GetProcAddress && Initialize && BindAPI && ChooseConfig &&
                   CreateContext && MakeCurrent && GetPlatformDisplayEXT;
        }
    };

    template<typename Fn>
    void sym(void* lib, const char* name, Fn& out) {

        out = reinterpret_cast<Fn>(dlsym(lib, name));
    }

    const Egl& egl() {

        static Egl e = [] {
            Egl x;
            // libEGL.so.1 is the glvnd loader's soname and the only one that is
            // guaranteed installed; libEGL.so is a -devel symlink a runtime-only
            // node will not have.
            for (const char* name : {"libEGL.so.1", "libEGL.so"}) {
                x.lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
                if (x.lib) break;
            }
            if (!x.lib) return x;

            sym(x.lib, "eglGetProcAddress", x.GetProcAddress);
            sym(x.lib, "eglGetError", x.GetError);
            sym(x.lib, "eglQueryString", x.QueryString);
            sym(x.lib, "eglInitialize", x.Initialize);
            sym(x.lib, "eglTerminate", x.Terminate);
            sym(x.lib, "eglBindAPI", x.BindAPI);
            sym(x.lib, "eglChooseConfig", x.ChooseConfig);
            sym(x.lib, "eglCreatePbufferSurface", x.CreatePbufferSurface);
            sym(x.lib, "eglCreateContext", x.CreateContext);
            sym(x.lib, "eglMakeCurrent", x.MakeCurrent);
            sym(x.lib, "eglGetCurrentContext", x.GetCurrentContext);
            sym(x.lib, "eglDestroyContext", x.DestroyContext);
            sym(x.lib, "eglDestroySurface", x.DestroySurface);

            if (x.GetProcAddress) {
                x.QueryDevicesEXT = reinterpret_cast<decltype(x.QueryDevicesEXT)>(
                        x.GetProcAddress("eglQueryDevicesEXT"));
                x.GetPlatformDisplayEXT = reinterpret_cast<decltype(x.GetPlatformDisplayEXT)>(
                        x.GetProcAddress("eglGetPlatformDisplayEXT"));
                x.QueryDeviceAttribEXT = reinterpret_cast<decltype(x.QueryDeviceAttribEXT)>(
                        x.GetProcAddress("eglQueryDeviceAttribEXT"));
            }
            return x;
        }();

        return e;
    }

    std::string eglErrorName(EGLint code) {

        switch (code) {
            case 0x3000: return "EGL_SUCCESS";
            case 0x3001: return "EGL_NOT_INITIALIZED";
            case 0x3002: return "EGL_BAD_ACCESS";
            case 0x3003: return "EGL_BAD_ALLOC";
            case 0x3004: return "EGL_BAD_ATTRIBUTE";
            case 0x3005: return "EGL_BAD_CONFIG";
            case 0x3006: return "EGL_BAD_CONTEXT";
            case 0x3007: return "EGL_BAD_CURRENT_SURFACE";
            case 0x3008: return "EGL_BAD_DISPLAY";
            case 0x3009: return "EGL_BAD_MATCH";
            case 0x300A: return "EGL_BAD_NATIVE_PIXMAP";
            case 0x300B: return "EGL_BAD_NATIVE_WINDOW";
            case 0x300C: return "EGL_BAD_PARAMETER";
            case 0x300D: return "EGL_BAD_SURFACE";
            case 0x300E: return "EGL_CONTEXT_LOST";
            default: {
                char buf[24];
                std::snprintf(buf, sizeof buf, "EGL error 0x%x", static_cast<unsigned>(code));
                return {buf};
            }
        }
    }

    EGLint lastError() {

        const auto& e = egl();
        return e.GetError ? e.GetError() : EGL_SUCCESS;
    }

    bool namesSoftwareRasteriser(const std::string& renderer) {

        // lavapipe and "Mesa Offscreen" belong here as much as llvmpipe: all
        // three are contexts that come up perfectly and draw on the CPU.
        for (const char* needle : {"llvmpipe", "lavapipe", "softpipe", "swrast",
                                   "Software Rasterizer", "Mesa Offscreen"}) {
            if (renderer.find(needle) != std::string::npos) return true;
        }
        return false;
    }

    // eglGetPlatformDisplayEXT returns the SAME handle for the same device, and
    // EGL 1.5 s3.2 makes eglInitialize on an already-initialised display a
    // no-op rather than a reference count. So eglTerminate from one context
    // would mark a second context's resources for deletion. Counted here
    // instead; the scan below also relies on it to tear down rejects safely.
    std::mutex& displayMutex() {

        static std::mutex m;
        return m;
    }

    std::map<EGLDisplay, int>& displayRefs() {

        static std::map<EGLDisplay, int> refs;
        return refs;
    }

    bool acquireDisplay(EGLDisplay display, EGLint* major, EGLint* minor) {

        const auto& e = egl();
        std::lock_guard<std::mutex> lock(displayMutex());

        if (e.Initialize(display, major, minor) != EGL_TRUE_) return false;
        ++displayRefs()[display];
        return true;
    }

    void releaseDisplay(EGLDisplay display) {

        const auto& e = egl();
        std::lock_guard<std::mutex> lock(displayMutex());

        auto it = displayRefs().find(display);
        if (it == displayRefs().end()) return;
        if (--it->second > 0) return;

        displayRefs().erase(it);
        if (e.Terminate) e.Terminate(display);
    }

    std::string glStringVia(void* (*getProc)(const char*), unsigned int name) {

        using GetStringFn = const unsigned char* (*) (unsigned int);
        // Resolved through EGL rather than glad: this runs during the device
        // scan, before any candidate has been accepted, and glad may only be
        // loaded once per process against the context that wins.
        auto fn = reinterpret_cast<GetStringFn>(getProc("glGetString"));
        if (!fn) return {};
        const auto* s = fn(name);
        return s ? std::string(reinterpret_cast<const char*>(s)) : std::string{};
    }
}// namespace

struct EglContext::Impl {

    EGLDisplay display{nullptr};
    EGLSurface surface{nullptr};
    EGLContextHandle context{nullptr};

    int drawableW{0}, drawableH{0};
    int deviceIndex{-1};
    int cudaIndex{-1};

    std::string vendor, renderer, version, eglVendor, eglVersion, platform;

    explicit Impl(const Parameters& p) {

        // A constructor that throws does not run its own destructor, and the
        // scan below creates and abandons contexts by design, so teardown has
        // to be explicit on every exit.
        try {
            init(p);
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~Impl() {

        destroy();
    }

    void init(const Parameters& p) {

        const auto& e = egl();
        if (!e.ok()) {
            throw std::runtime_error(
                    "EglContext: no usable EGL loader. libEGL.so.1 must be present and must "
                    "provide eglGetPlatformDisplayEXT — on a driver installed without its EGL "
                    "half there is nothing here to use.");
        }

        std::string reasons;
        const auto note = [&reasons](const std::string& s) {
            if (!reasons.empty()) reasons += "; ";
            reasons += s;
        };

        // Devices first, in order, then the surfaceless platform. Surfaceless
        // is NOT a software fallback — on the H100 node it reached the GPU too —
        // so what keeps a CPU rasteriser out is requireHardware, checked per
        // candidate below, not the platform ordering.
        if (!p.surfaceless) {
            const int count = EglContext::deviceCount();
            if (count <= 0) {
                note("eglQueryDevicesEXT reported no devices (" + eglErrorName(lastError()) + ")");
            }
            for (int i = 0; i < count; ++i) {
                if (p.device >= 0 && i != p.device) continue;
                std::string why;
                if (tryDevice(i, p, why)) return;
                note("device " + std::to_string(i) + ": " + why);
            }
            if (p.device >= 0 && p.device >= count) {
                throw std::runtime_error(
                        "EglContext: device " + std::to_string(p.device) + " requested but EGL reports " +
                        std::to_string(count) + " device(s)");
            }
        }

        std::string why;
        if (tryCandidate(EGL_PLATFORM_SURFACELESS_MESA, nullptr, -1, "surfaceless", p, why)) return;
        note("surfaceless: " + why);

        throw std::runtime_error(
                std::string("EglContext: no usable ") + (p.requireHardware ? "hardware " : "") +
                "OpenGL context. " + reasons +
                (p.requireHardware
                         ? ". Set requireHardware=false to accept a software rasteriser."
                         : ""));
    }

    bool tryDevice(int index, const Parameters& p, std::string& why) {

        const auto& e = egl();

        EGLint count = 0;
        if (!e.QueryDevicesEXT || e.QueryDevicesEXT(0, nullptr, &count) != EGL_TRUE_ || count <= 0) {
            why = "eglQueryDevicesEXT failed (" + eglErrorName(lastError()) + ")";
            return false;
        }
        std::vector<EGLDeviceEXT> devices(static_cast<std::size_t>(count));
        if (e.QueryDevicesEXT(count, devices.data(), &count) != EGL_TRUE_ || index >= count) {
            why = "eglQueryDevicesEXT enumeration failed (" + eglErrorName(lastError()) + ")";
            return false;
        }

        auto device = devices[static_cast<std::size_t>(index)];
        if (!tryCandidate(EGL_PLATFORM_DEVICE_EXT, device, index, "device", p, why)) return false;

        // Only meaningful once the device is the chosen one.
        if (e.QueryDeviceAttribEXT) {
            EGLAttribKHR value = 0;
            if (e.QueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, &value) == EGL_TRUE_) {
                cudaIndex = static_cast<int>(value);
            }
        }
        return true;
    }

    bool tryCandidate(EGLenum platformEnum, void* native, int index,
                      const char* platformName, const Parameters& p, std::string& why) {

        const auto& e = egl();

        EGLDisplay dpy = e.GetPlatformDisplayEXT(platformEnum, native, nullptr);
        if (!dpy) {
            why = "eglGetPlatformDisplayEXT returned no display (" + eglErrorName(lastError()) + ")";
            return false;
        }

        EGLint major = 0, minor = 0;
        if (!acquireDisplay(dpy, &major, &minor)) {
            why = "eglInitialize failed (" + eglErrorName(lastError()) + ")";
            return false;
        }

        // Desktop GL, not GLES. A driver with EGL that refuses this is telling
        // you it will only do GLES here, which threepp cannot use.
        if (e.BindAPI(EGL_OPENGL_API) != EGL_TRUE_) {
            why = "eglBindAPI(EGL_OPENGL_API) refused - no desktop OpenGL through EGL here, only GLES";
            releaseDisplay(dpy);
            return false;
        }

        // Three config requests, narrowing in the same order the node probe
        // used when it measured HARDWARE here. Asking for more than the probe
        // did and then not retrying is how a working driver looks broken.
        const EGLint full[] = {
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
                EGL_NONE};
        const EGLint plain[] = {
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                EGL_NONE};
        const EGLint noPbuffer[] = {
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE};

        struct Request {
            const EGLint* attribs;
            bool pbuffer;
        };
        const Request requests[] = {{full, true}, {plain, true}, {noPbuffer, false}};

        EGLConfig config{};
        bool pbufferCapable = false;
        bool haveConfig = false;
        for (const auto& r : requests) {
            EGLint n = 0;
            if (e.ChooseConfig(dpy, r.attribs, &config, 1, &n) == EGL_TRUE_ && n > 0) {
                pbufferCapable = r.pbuffer;
                haveConfig = true;
                break;
            }
        }
        if (!haveConfig) {
            why = "eglChooseConfig found no OpenGL-capable config (" + eglErrorName(lastError()) + ")";
            releaseDisplay(dpy);
            return false;
        }

        EGLSurface surf = nullptr;
        int w = 0, h = 0;
        if (pbufferCapable && e.CreatePbufferSurface) {
            const int rw = p.width > 0 ? p.width : 1;
            const int rh = p.height > 0 ? p.height : 1;
            const EGLint pbufferAttribs[] = {EGL_WIDTH, rw, EGL_HEIGHT, rh, EGL_NONE};
            surf = e.CreatePbufferSurface(dpy, config, pbufferAttribs);
            if (surf) {
                w = rw;
                h = rh;
            }
        }
        // No surface is survivable: EGL_KHR_surfaceless_context allows a
        // current context with EGL_NO_SURFACE, and every render worth doing
        // here targets an FBO. drawableW/H stay 0 to say framebuffer 0 is not
        // a thing you may draw into.

        const EGLint ctxAttribs[] = {
                EGL_CONTEXT_MAJOR_VERSION, 3,
                EGL_CONTEXT_MINOR_VERSION, 3,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE};

        EGLContextHandle ctx = e.CreateContext(dpy, config, nullptr, ctxAttribs);
        if (!ctx) {
            // Those attributes need EGL 1.5 or EGL_KHR_create_context; on plain
            // EGL 1.4 they are EGL_BAD_ATTRIBUTE. The probe retried bare, so do
            // the same rather than reporting a driver that in fact works.
            ctx = e.CreateContext(dpy, config, nullptr, nullptr);
        }
        if (!ctx) {
            why = "eglCreateContext failed for a 3.3 core profile and for a default context (" +
                  eglErrorName(lastError()) + ")";
            if (surf && e.DestroySurface) e.DestroySurface(dpy, surf);
            releaseDisplay(dpy);
            return false;
        }

        if (e.MakeCurrent(dpy, surf, surf, ctx) != EGL_TRUE_) {
            why = surf ? "eglMakeCurrent failed (" + eglErrorName(lastError()) + ")"
                       : "eglMakeCurrent with no surface failed - the driver lacks "
                         "EGL_KHR_surfaceless_context (" +
                                 eglErrorName(lastError()) + ")";
            if (e.DestroyContext) e.DestroyContext(dpy, ctx);
            if (surf && e.DestroySurface) e.DestroySurface(dpy, surf);
            releaseDisplay(dpy);
            return false;
        }

        const std::string rendererStr = glStringVia(e.GetProcAddress, GL_RENDERER_);
        if (p.requireHardware && (rendererStr.empty() || namesSoftwareRasteriser(rendererStr))) {
            why = "GL_RENDERER = \"" + (rendererStr.empty() ? std::string("<none>") : rendererStr) +
                  "\" is a software rasteriser";
            e.MakeCurrent(dpy, nullptr, nullptr, nullptr);
            if (e.DestroyContext) e.DestroyContext(dpy, ctx);
            if (surf && e.DestroySurface) e.DestroySurface(dpy, surf);
            releaseDisplay(dpy);
            return false;
        }

        display = dpy;
        surface = surf;
        context = ctx;
        drawableW = w;
        drawableH = h;
        deviceIndex = index;
        platform = platformName;
        renderer = rendererStr;
        vendor = glStringVia(e.GetProcAddress, GL_VENDOR_);
        version = glStringVia(e.GetProcAddress, GL_VERSION_);
        if (e.QueryString) {
            const char* v = e.QueryString(dpy, EGL_VENDOR_STR);
            const char* r = e.QueryString(dpy, EGL_VERSION_STR);
            eglVendor = v ? v : "";
            eglVersion = r ? r : "";
        }

        // Only now, against the context that won. GLAD's pointers are
        // process-wide and first-load-wins, which is why the scan above read
        // GL_RENDERER through EGL instead.
        loadGlad(reinterpret_cast<GLADloadproc>(e.GetProcAddress));
        return true;
    }

    void destroy() {

        const auto& e = egl();
        if (!display) return;

        // Un-current only if this context is the current one: clearing
        // unconditionally would detach a different context bound on this thread.
        if (e.MakeCurrent && (!e.GetCurrentContext || e.GetCurrentContext() == context)) {
            e.MakeCurrent(display, nullptr, nullptr, nullptr);
        }
        if (context && e.DestroyContext) e.DestroyContext(display, context);
        if (surface && e.DestroySurface) e.DestroySurface(display, surface);
        releaseDisplay(display);

        display = nullptr;
        surface = nullptr;
        context = nullptr;
    }
};

#else// not Linux

struct EglContext::Impl {

    int drawableW{0}, drawableH{0}, deviceIndex{-1}, cudaIndex{-1};
    std::string vendor, renderer, version, eglVendor, eglVersion, platform;

    explicit Impl(const Parameters&) {

        throw std::runtime_error(
                "EglContext: EGL headless contexts are Linux-only. On Windows and macOS a "
                "hidden Canvas window already gives a hardware GL context.");
    }
};

#endif

EglContext::EglContext()
    : EglContext(Parameters{}) {}

EglContext::EglContext(const Parameters& parameters)
    : pimpl_(std::make_unique<Impl>(parameters)) {}

EglContext::~EglContext() = default;

void EglContext::makeCurrent() const {

#if THREEPP_HAS_EGL
    const auto& e = egl();
    // EGL's bound API is per-thread state and defaults to OpenGL ES, so a
    // context handed to a worker thread must re-bind desktop GL here or the
    // next eglMakeCurrent has the wrong release semantics.
    e.BindAPI(EGL_OPENGL_API);
    if (e.MakeCurrent(pimpl_->display, pimpl_->surface, pimpl_->surface, pimpl_->context) != EGL_TRUE_) {
        throw std::runtime_error("EglContext: eglMakeCurrent failed on rebind (" +
                                 eglErrorName(lastError()) + ")");
    }
#endif
}

const std::string& EglContext::renderer() const {

    return pimpl_->renderer;
}

const std::string& EglContext::vendor() const {

    return pimpl_->vendor;
}

const std::string& EglContext::version() const {

    return pimpl_->version;
}

const std::string& EglContext::eglVendor() const {

    return pimpl_->eglVendor;
}

const std::string& EglContext::eglVersion() const {

    return pimpl_->eglVersion;
}

const std::string& EglContext::platform() const {

    return pimpl_->platform;
}

int EglContext::deviceIndex() const {

    return pimpl_->deviceIndex;
}

int EglContext::cudaDeviceIndex() const {

    return pimpl_->cudaIndex;
}

int EglContext::drawableWidth() const {

    return pimpl_->drawableW;
}

int EglContext::drawableHeight() const {

    return pimpl_->drawableH;
}

bool EglContext::hardwareAccelerated() const {

#if THREEPP_HAS_EGL
    return !pimpl_->renderer.empty() && !namesSoftwareRasteriser(pimpl_->renderer);
#else
    return false;
#endif
}

int EglContext::deviceCount() {

#if THREEPP_HAS_EGL
    const auto& e = egl();
    if (!e.ok() || !e.QueryDevicesEXT) return 0;

    EGLint count = 0;
    if (e.QueryDevicesEXT(0, nullptr, &count) != EGL_TRUE_) return 0;
    return count > 0 ? count : 0;
#else
    return 0;
#endif
}

bool EglContext::available() {

#if THREEPP_HAS_EGL
    // The header promises a device, not just a loader: a launcher branching on
    // this to choose GL over a CPU path must not be told yes by a node that
    // has libEGL and no GPU.
    return egl().ok() && deviceCount() > 0;
#else
    return false;
#endif
}
