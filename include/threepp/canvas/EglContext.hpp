
#ifndef THREEPP_EGL_CONTEXT_HPP
#define THREEPP_EGL_CONTEXT_HPP

#include <memory>
#include <string>

namespace threepp {

    /// An OpenGL context taken straight from the driver, with no window system
    /// at all: no X server, no Wayland, no GLFW, no display of any kind.
    ///
    /// This exists because the usual headless story does not hold on a compute
    /// node. GLFW's Null platform can only reach OSMesa, which is a software
    /// rasteriser, and an X server (Xvfb included) serves whichever GLX its own
    /// extension advertises — Mesa's, because Xvfb never loads the vendor's
    /// server-side GLX module. So "OpenGL over a virtual display" quietly
    /// renders on the CPU. EGL is the path the vendors actually support:
    /// measured on an NVIDIA H100 compute node, both EGL_EXT_platform_device
    /// and Mesa's surfaceless platform returned
    /// GL_RENDERER = "NVIDIA H100 80GB HBM3/PCIe/SSE2" with no display server.
    ///
    /// Construct one, then build a GLRenderer with the size-only constructor.
    /// SIZE THE DRAWABLE: with no RenderTarget bound the renderer draws into,
    /// and reads back from, framebuffer 0 — which here is the pbuffer. A
    /// pbuffer smaller than the renderer clips the draw and makes readback
    /// return undefined bytes, silently and with no GL error. So either pass a
    /// pbuffer at least as large as the renderer:
    ///
    ///     EglContext ctx({1920, 1080});
    ///     GLRenderer renderer({1920, 1080});
    ///
    /// or render into a RenderTarget, where the drawable size does not matter.
    ///
    /// One context per process is the rule. GLAD's function pointers are
    /// process-wide, so a second context (or a GLFW Canvas alongside this) has
    /// them resolved against whichever context loaded them first.
    ///
    /// Linux only. Elsewhere available() is false and construction throws.
    class EglContext {

    public:
        struct Parameters {

            /// Size of the pbuffer the context is made current against, and
            /// therefore of framebuffer 0. Make it at least as large as any
            /// renderer that draws without a RenderTarget — see the note above.
            int width{1};
            int height{1};

            /// Which EGL device to bind, indexed as eglQueryDevicesEXT reports
            /// them. -1 (the default) means: try each in turn and keep the
            /// first that satisfies requireHardware.
            int device{-1};

            /// Refuse a context whose GL_RENDERER names a software rasteriser.
            /// On by default, and deliberately: a job that renders on llvmpipe
            /// succeeds, looks correct, and wastes its whole GPU allocation.
            /// Set false only when a CPU fallback is genuinely wanted.
            bool requireHardware{true};

            /// Skip the device platform and go straight to Mesa's surfaceless
            /// platform (EGL_PLATFORM_SURFACELESS_MESA). Left false, the device
            /// platform is tried first and surfaceless is the fallback — which
            /// is not a software fallback: on the H100 node measured above,
            /// surfaceless reached the GPU too. requireHardware is what keeps
            /// a software rasteriser out, not the platform choice.
            bool surfaceless{false};
        };

        /// Two overloads rather than `= {}` on one: GCC rejects a defaulted
        /// brace-initialiser for a reference to a class nested in the same
        /// class ("could not convert <brace-enclosed initializer list>"),
        /// where MSVC accepts it. Writing it this way keeps the Linux build —
        /// the one that matters for a headless context — compiling.
        EglContext();
        explicit EglContext(const Parameters& parameters);
        ~EglContext();

        EglContext(EglContext&&) = delete;
        EglContext(const EglContext&) = delete;
        EglContext& operator=(const EglContext&) = delete;
        EglContext& operator=(EglContext&&) = delete;

        /// Rebind this context to the calling thread. Construction already
        /// makes it current on the constructing thread; this is for handing the
        /// context to another thread, and it re-binds the OpenGL API too (EGL's
        /// bound API is per-thread, and defaults to OpenGL ES).
        void makeCurrent() const;

        /// GL_RENDERER. The single most useful string here: it is what tells
        /// you whether you got the GPU or fell back to a software rasteriser.
        /// "NVIDIA H100 80GB HBM3/PCIe/SSE2" is the win; "llvmpipe" or
        /// "lavapipe" means the pixels are being drawn on the CPU.
        [[nodiscard]] const std::string& renderer() const;
        [[nodiscard]] const std::string& vendor() const;
        [[nodiscard]] const std::string& version() const;

        /// The EGL implementation behind the context, as opposed to the GL one
        /// above: EGL_VENDOR and EGL_VERSION. On a glvnd system this is how you
        /// tell which vendor library actually answered.
        [[nodiscard]] const std::string& eglVendor() const;
        [[nodiscard]] const std::string& eglVersion() const;

        /// False when renderer() names a known software rasteriser. A heuristic
        /// on a driver-supplied string, not a promise — but it catches the
        /// failure that actually happens: a context that comes up perfectly and
        /// runs on the CPU.
        [[nodiscard]] bool hardwareAccelerated() const;

        /// Which platform the context came from, for logs and manifests:
        /// "device" or "surfaceless".
        [[nodiscard]] const std::string& platform() const;

        /// Index of the EGL device that won, or -1 for the surfaceless platform.
        [[nodiscard]] int deviceIndex() const;

        /// The CUDA device ordinal for the bound EGL device (EGL_CUDA_DEVICE_NV),
        /// or -1 when the driver does not report one. EGL device order and CUDA
        /// device order are NOT the same enumeration; on a multi-GPU node this
        /// is how a render is matched to the GPU a tensor lives on.
        [[nodiscard]] int cudaDeviceIndex() const;

        /// Size of framebuffer 0, or 0 when the context has no drawable at all
        /// (a surfaceless context bound with EGL_NO_SURFACE). A renderer that
        /// draws without a RenderTarget must not exceed this.
        [[nodiscard]] int drawableWidth() const;
        [[nodiscard]] int drawableHeight() const;

        /// True when libEGL loaded AND reports at least one device.
        /// Never throws — call it to decide whether to try.
        [[nodiscard]] static bool available();

        /// How many devices eglQueryDevicesEXT reports; 0 if EGL is unusable.
        [[nodiscard]] static int deviceCount();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_EGL_CONTEXT_HPP
