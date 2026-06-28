// Vulkan deferred (RasterFirst) renderer + G-buffer AOV readback.
//
// Compiled unconditionally, but the body is active only when threepp is built
// with the Vulkan backend (THREEPP_PY_HAS_VULKAN, defined by python/CMakeLists
// when THREEPP_WITH_VULKAN is ON). Otherwise init_vulkan only sets HAS_VULKAN.
//
// The deferred renderer writes a full G-buffer every frame (world normals,
// optical flow, instance-segmentation ids, albedo, depth). The renderer's
// debug-resolve compute pass (setHybridDebugView) encodes a chosen attachment
// into the swapchain, where readRGBPixels() can read it back — so the AOVs come
// out as (H, W, 3) uint8 images: normals as n*0.5+0.5, segmentation as per-id
// hashed colours, albedo passthrough. (Raw float depth has no readback path in
// the renderer yet; it is the natural next step.)
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_VULKAN

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;// the threepp_py::py alias isn't visible in the anon namespace below
using namespace threepp;

namespace {

    // AOV name -> setHybridDebugView code. 0 = the shaded RGB output (Off).
    int aov_code(const std::string& aov) {
        if (aov == "rgb" || aov == "shaded" || aov == "color") return 0;
        if (aov == "normals" || aov == "normal") return 1;
        if (aov == "motion" || aov == "flow") return 2;
        if (aov == "segmentation" || aov == "seg" || aov == "ids") return 3;
        if (aov == "albedo") return 4;
        throw std::invalid_argument(
                "unknown AOV '" + aov + "' — use one of: rgb, normals, segmentation, albedo, motion");
    }

    // Python-facing facade over VulkanRenderer that hides the deferred
    // frame-model: submit+present is deferred to the canvas frame-end callback,
    // so frames are driven through animateOnce() (not render() directly) and
    // repeated `flush_frames` times to make the MAILBOX swapchain readback
    // deterministic. See the headless-frame-model design note.
    class PyVulkanRenderer {
    public:
        explicit PyVulkanRenderer(Canvas& canvas, int flush_frames)
            : canvas_(canvas), renderer_(canvas), flush_(flush_frames < 1 ? 1 : flush_frames) {
        }

        void render(Object3D& scene, Camera& camera) {
            // Works in both usage models:
            //  - inside canvas.animate(...): the canvas loop already owns
            //    submit/present, so just record this frame (recording it via
            //    drive_frames would nest animateOnce and run 3 full frames per
            //    displayed frame — the cause of a sluggish interactive window);
            //  - standalone/headless: there is no outer loop, so we drive the
            //    deferred frame-model ourselves (animateOnce x flush).
            if (canvas_.isInsideAnimateLoop()) {
                renderer_.render(scene, camera);
            } else {
                drive_frames(scene, camera);
            }
        }

        py::array_t<uint8_t> read_pixels() { return to_numpy(renderer_.readRGBPixels()); }

        // Scene-only swapchain capture: the post-TAA / pre-overlay frame, WITHOUT
        // any sprite / ImGui overlays composited on top — what sensor pipelines
        // need to avoid feeding their own overlay back in. Reuses to_numpy →
        // (H, W, 3) uint8. Enable once, then read after a render(); the per-frame
        // copy costs only while enabled.
        void set_scene_capture(bool enabled) { renderer_.setSceneCaptureEnabled(enabled); }
        bool scene_capture() const { return renderer_.sceneCaptureEnabled(); }
        py::array_t<uint8_t> read_scene_pixels() { return to_numpy(renderer_.readSceneRGBPixels()); }

        // Viewport / scissor (forwarded to the native renderer; (x,y,w,h) overload).
        void set_viewport(int x, int y, int w, int h) { renderer_.setViewport(x, y, w, h); }
        void set_scissor(int x, int y, int w, int h) { renderer_.setScissor(x, y, w, h); }
        void set_scissor_test(bool enabled) { renderer_.setScissorTest(enabled); }

        // Metric depth as (H, W) float32 in scene units (distance from the
        // camera). The depth debug-view packs the reverse-Z depth into 24-bit
        // RGB; here we decode it and linearize with the camera near/far. Sky /
        // background pixels (cleared to far) read as `far`.
        py::array_t<float> read_depth(Object3D& scene, Camera& camera) {
            renderer_.setHybridDebugView(5);// Depth → 24-bit packed reverse-Z
            drive_frames(scene, camera);
            const std::vector<unsigned char> px = renderer_.readRGBPixels();
            renderer_.setHybridDebugView(0);

            const auto s = renderer_.framebufferSize();
            const int w = s.width(), h = s.height();
            py::array_t<float> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
            const size_t pxCount = static_cast<size_t>(w) * h;
            const size_t ch = pxCount ? px.size() / pxCount : 3;
            auto* dst = arr.mutable_data();
            const float nearP = camera.nearPlane;
            const float farP = camera.farPlane;
            if (px.size() >= pxCount * ch && (ch == 3 || ch == 4)) {
                for (size_t i = 0; i < pxCount; ++i) {
                    const uint32_t r = px[i * ch + 0], gg = px[i * ch + 1], b = px[i * ch + 2];
                    const float d = static_cast<float>((r << 16) | (gg << 8) | b) / 16777215.0f;
                    // reverse-Z [0,1] (1=near, 0=far) → distance from camera.
                    dst[i] = (d <= 0.f) ? farP : (nearP * farP) / (nearP + d * (farP - nearP));
                }
            }
            return arr;
        }

        py::array_t<uint8_t> render_aov(Object3D& scene, Camera& camera, const std::string& aov) {
            const int code = aov_code(aov);
            renderer_.setHybridDebugView(code);
            drive_frames(scene, camera);
            auto arr = to_numpy(renderer_.readRGBPixels());
            renderer_.setHybridDebugView(0);// leave the renderer in shaded mode
            return arr;
        }

        // Render once per requested AOV and return {name: (H,W,3) uint8}.
        py::dict render_aovs(Object3D& scene, Camera& camera, const std::vector<std::string>& aovs) {
            py::dict out;
            for (const auto& aov : aovs) out[py::str(aov)] = render_aov(scene, camera, aov);
            return out;
        }

        void save_frame(Object3D& scene, Camera& camera, const std::string& path) {
            canvas_.animateOnce([&] {
                renderer_.render(scene, camera);
                renderer_.writeFramebuffer(path);
            });
        }

        void set_clear_color(const Color& c, float alpha) { renderer_.setClearColor(c, alpha); }

        // Accepted for GLRenderer API parity. The Vulkan backend's analytic
        // lights cast ray-traced shadows unconditionally (there is no shadow-map
        // pass to toggle), so this flag is stored but has no rendering effect —
        // it just lets GL-targeted code run unchanged on Vulkan.
        bool shadow_map_enabled() const { return shadowMapEnabled_; }
        void set_shadow_map_enabled(bool v) { shadowMapEnabled_ = v; }

        void set_flush_frames(int n) { flush_ = n < 1 ? 1 : n; }
        void set_size(int w, int h) { renderer_.setSize({w, h}); }

        std::pair<int, int> size() const {
            const auto s = renderer_.framebufferSize();
            return {s.width(), s.height()};
        }

        // Underlying threepp renderer — used to attach the ImGui Vulkan overlay.
        VulkanRenderer& native() { return renderer_; }

    private:
        void drive_frames(Object3D& scene, Camera& camera) {
            for (int i = 0; i < flush_; ++i) {
                canvas_.animateOnce([&] { renderer_.render(scene, camera); });
            }
        }

        py::array_t<uint8_t> to_numpy(const std::vector<unsigned char>& px) {
            const auto s = renderer_.framebufferSize();
            const int w = s.width(), h = s.height();
            py::array_t<uint8_t> arr({static_cast<py::ssize_t>(h),
                                      static_cast<py::ssize_t>(w),
                                      static_cast<py::ssize_t>(3)});
            const size_t pxCount = static_cast<size_t>(w) * h;
            const size_t ch = pxCount ? px.size() / pxCount : 3;// 3 (RGB) or 4 (RGBA)
            auto* dst = arr.mutable_data();
            if (px.size() >= pxCount * ch && (ch == 3 || ch == 4)) {
                for (size_t i = 0; i < pxCount; ++i) {
                    dst[i * 3 + 0] = px[i * ch + 0];
                    dst[i * 3 + 1] = px[i * ch + 1];
                    dst[i * 3 + 2] = px[i * ch + 2];
                }
            }
            return arr;
        }

        Canvas& canvas_;
        VulkanRenderer renderer_;
        int flush_;
        bool shadowMapEnabled_ = true;// no-op parity flag (Vulkan shadows are always on)
    };

}// namespace

namespace threepp_py {

    void init_vulkan(py::module_& m) {
        py::class_<PyVulkanRenderer>(m, "VulkanRenderer")
                .def(py::init([](Canvas& c, int flush) { return std::make_unique<PyVulkanRenderer>(c, flush); }),
                     py::arg("canvas"), py::arg("flush_frames") = 3, py::keep_alive<1, 2>(),
                     "Deferred (RasterFirst) Vulkan renderer. Pass a headless Canvas "
                     "created with vsync=False.")
                .def("render", &PyVulkanRenderer::render, py::arg("scene"), py::arg("camera"))
                .def("read_pixels", &PyVulkanRenderer::read_pixels,
                     "Final shaded RGB of the last render as (H, W, 3) uint8.")
                .def_property("scene_capture",
                              [](PyVulkanRenderer& r) { return r.scene_capture(); },
                              [](PyVulkanRenderer& r, bool v) { r.set_scene_capture(v); },
                              "Toggle scene-only swapchain capture (post-TAA, pre-overlay). When on, "
                              "read it via read_scene_pixels(); off = no cost.")
                .def("read_scene_pixels", &PyVulkanRenderer::read_scene_pixels,
                     "Last captured scene-only RGB (post-TAA, pre-overlay; no sprite/ImGui) as "
                     "(H, W, 3) uint8. Requires scene_capture=True.")
                .def("render_aov", &PyVulkanRenderer::render_aov,
                     py::arg("scene"), py::arg("camera"), py::arg("aov"),
                     "Render and return a G-buffer AOV as (H, W, 3) uint8: "
                     "'rgb' | 'normals' | 'segmentation' | 'albedo' | 'motion'.")
                .def("render_aovs", &PyVulkanRenderer::render_aovs,
                     py::arg("scene"), py::arg("camera"),
                     py::arg("aovs") = std::vector<std::string>{"rgb", "normals", "segmentation"},
                     "Render the requested AOVs and return {name: (H, W, 3) uint8}.")
                .def("read_normals", [](PyVulkanRenderer& r, Object3D& s, Camera& c) { return r.render_aov(s, c, "normals"); },
                     py::arg("scene"), py::arg("camera"))
                .def("read_segmentation", [](PyVulkanRenderer& r, Object3D& s, Camera& c) { return r.render_aov(s, c, "segmentation"); },
                     py::arg("scene"), py::arg("camera"))
                .def("read_albedo", [](PyVulkanRenderer& r, Object3D& s, Camera& c) { return r.render_aov(s, c, "albedo"); },
                     py::arg("scene"), py::arg("camera"))
                .def("read_depth", &PyVulkanRenderer::read_depth, py::arg("scene"), py::arg("camera"),
                     "Metric depth as (H, W) float32 — distance from the camera in scene units. "
                     "Background reads as the camera far plane.")
                .def("set_clear_color", &PyVulkanRenderer::set_clear_color, py::arg("color"), py::arg("alpha") = 1.f)
                // No-op shadow toggle for GLRenderer API parity: Vulkan analytic
                // lights always cast ray-traced shadows, so this is stored but
                // ignored — the same script runs on GL and Vulkan unchanged.
                .def_property("shadow_map_enabled",
                              [](PyVulkanRenderer& r) { return r.shadow_map_enabled(); },
                              [](PyVulkanRenderer& r, bool v) { r.set_shadow_map_enabled(v); })
                // Tone mapping — same knobs as GLRenderer. The deferred renderer
                // syncs these (from the Renderer base) into its composite/resolve
                // pass each frame, so they can be flipped between renders. Default
                // operator is NoToneMapping (HDR clips); ACESFilmic/Neutral give a
                // filmic roll-off.
                .def_property("tone_mapping",
                              [](PyVulkanRenderer& r) { return r.native().toneMapping; },
                              [](PyVulkanRenderer& r, ToneMapping t) { r.native().toneMapping = t; })
                .def_property("tone_mapping_exposure",
                              [](PyVulkanRenderer& r) { return r.native().toneMappingExposure; },
                              [](PyVulkanRenderer& r, float e) { r.native().toneMappingExposure = e; })
                // Internal render scale: the deferred G-buffer + ray-traced channels
                // run at (extent * scale) and TAA upsamples to full resolution. 1.0 =
                // none; 0.5 quarters the shaded pixel count (quadratic perf lever).
                // Clamped to [0.25, 1.0]. Reallocates + resets accumulation, so don't
                // set it from inside a render().
                .def_property("render_scale",
                              [](PyVulkanRenderer& r) { return r.native().renderScale(); },
                              [](PyVulkanRenderer& r, float s) { r.native().setRenderScale(s); })
                // HDR bloom (added in linear HDR before the tone-map curve).
                .def_property("bloom_intensity",
                              [](PyVulkanRenderer& r) { return r.native().bloomIntensity(); },
                              [](PyVulkanRenderer& r, float v) { r.native().setBloomIntensity(v); },
                              "Bloom strength. 0 disables; typical 0.2-0.8.")
                .def_property("bloom_threshold",
                              [](PyVulkanRenderer& r) { return r.native().bloomThreshold(); },
                              [](PyVulkanRenderer& r, float v) { r.native().setBloomThreshold(v); },
                              "Bright-pass cutoff (linear-HDR luma); higher = only the brightest glow. Typical 0.8-2.0.")
                .def_property("bloom_clamp",
                              [](PyVulkanRenderer& r) { return r.native().bloomClamp(); },
                              [](PyVulkanRenderer& r, float v) { r.native().setBloomClamp(v); },
                              "Bloom input clamp to stabilise flickery ultra-bright highlights. <=0 disables (default); typical 8-32.")
                // Denoiser toggle (SVGF à-trous + temporal accumulation). ON by
                // default. With it ON the deferred GI/AO is a stochastic ~1-spp gather
                // cleaned by the denoiser; OFF switches to the deterministic 64-ray AO
                // (noise-free, no GI colour, higher per-pixel cost). Equivalent to the
                // THREEPP_DENOISE=0 startup env var, but flippable per frame.
                .def_property("denoise",
                              [](PyVulkanRenderer& r) { return r.native().denoise(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setDenoise(v); },
                              "Toggle the deferred denoiser (SVGF + temporal). Default on. "
                              "Off uses the deterministic 64-ray AO (noise-free, no GI colour).")
                // Ray-traced env ambient-occlusion / 1-bounce diffuse GI. ON by
                // default. Off drops the per-pixel occlusion rays (flat ambient IBL).
                .def_property("deferred_ao",
                              [](PyVulkanRenderer& r) { return r.native().deferredAO(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setDeferredAO(v); },
                              "Toggle ray-traced ambient occlusion / diffuse GI. Default on.")
                .def("set_flush_frames", &PyVulkanRenderer::set_flush_frames, py::arg("n"),
                     "Frames driven per render() to flush the MAILBOX swapchain (default 3; "
                     "raise to 4+ for fast-moving dynamic scenes).")
                .def("set_size", &PyVulkanRenderer::set_size, py::arg("width"), py::arg("height"),
                     "Resize the renderer's framebuffer/swapchain — call this from "
                     "canvas.on_window_resize together with updating the camera aspect.")
                .def("set_viewport", &PyVulkanRenderer::set_viewport,
                     py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"))
                .def("set_scissor", &PyVulkanRenderer::set_scissor,
                     py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"))
                .def("set_scissor_test", &PyVulkanRenderer::set_scissor_test, py::arg("enabled"))
                .def("save_frame", &PyVulkanRenderer::save_frame, py::arg("scene"), py::arg("camera"), py::arg("path"))
                .def("size", &PyVulkanRenderer::size)
                // Volumetric fog: Henyey-Greenstein phase anisotropy.
                // Clamped to [-0.95, 0.95]. Only takes effect when scene.set_fog_exp2() is active.
                // 0 = isotropic scattering, +0.9 = forward god-rays, -0.9 = back-scatter halo.
                .def_property("fog_anisotropy",
                              [](PyVulkanRenderer& r) { return r.native().getFogAnisotropy(); },
                              [](PyVulkanRenderer& r, float g) { r.native().setFogAnisotropy(g); })
                // Volumetric directional-light fog (sun shafts + aerial glow). Ray-marches
                // the air column with RT shadow rays so trees/terrain carve real light
                // shafts; the haze brightens toward the sun via fog_anisotropy. Opt-in
                // (per-step shadow-ray cost); only contributes when scene.set_fog_exp2()
                // is active. This gives an outdoor scene true volume vs flat distance haze.
                .def_property("volumetric_fog",
                              [](PyVulkanRenderer& r) { return r.native().volumetricFog(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setVolumetricFog(v); });

        m.attr("HAS_VULKAN") = true;
    }

    threepp::Renderer* py_vulkan_native_renderer(const py::handle& h) {
        if (py::isinstance<PyVulkanRenderer>(h)) {
            return &h.cast<PyVulkanRenderer&>().native();
        }
        return nullptr;
    }

}// namespace threepp_py

#else// THREEPP_PY_HAS_VULKAN not defined — GL-only build

namespace threepp_py {

    void init_vulkan(py::module_& m) {
        // Marker so Python can check availability:  threepp.HAS_VULKAN
        m.attr("HAS_VULKAN") = false;
    }

    threepp::Renderer* py_vulkan_native_renderer(const py::handle&) { return nullptr; }

}// namespace threepp_py

#endif
