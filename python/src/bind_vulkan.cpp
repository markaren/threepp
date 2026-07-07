// Vulkan deferred (RasterFirst) renderer + G-buffer AOV readback.
//
// Compiled unconditionally, but the body is active only when threepp is built
// with the Vulkan backend (THREEPP_PY_HAS_VULKAN, defined by python/CMakeLists
// when THREEPP_WITH_VULKAN is ON). Otherwise init_vulkan only sets HAS_VULKAN.
//
// The deferred renderer writes a full G-buffer every frame (world normals,
// optical flow, instance-segmentation ids, albedo, depth). Two readback paths:
//
//   * Visualisation (8-bit): render_aov / render_aovs go through the debug-
//     resolve compute pass (setHybridDebugView) → swapchain → readRGBPixels,
//     returning (H, W, 3) uint8 (normals as n*0.5+0.5, segmentation as per-id
//     hashed colour, albedo passthrough). Good for montages / eyeballing.
//
//   * Lossless (float / int): read_depth, read_normals_float, read_instance_ids,
//     read_motion and read_aovs_typed copy the native G-buffer attachment
//     straight to host memory via VulkanRendererCore::readGBufferAOV — full-
//     precision depth (f32), world normals (f32), RECOVERABLE integer instance
//     ids (u32, no hashing) and metric motion (f32). This is the material an ML
//     / sensor pipeline actually trains on.
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_VULKAN

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;// the threepp_py::py alias isn't visible in the anon namespace below
using namespace threepp;

namespace {

    // IEEE-754 half (binary16) -> float32. The G-buffer normal/motion attachments
    // are R16G16B16A16_SFLOAT; the native AOV readback returns their raw bytes, so
    // we decode the halves host-side (numpy has no lossless half decode we can rely
    // on across versions). Handles subnormals, inf and NaN.
    inline float half_to_float(uint16_t hbits) {
        const uint32_t sign = static_cast<uint32_t>(hbits & 0x8000u) << 16;
        const uint32_t exp  = (hbits & 0x7C00u) >> 10;
        const uint32_t mant = (hbits & 0x03FFu);
        uint32_t bits;
        if (exp == 0u) {
            if (mant == 0u) {
                bits = sign;// signed zero
            } else {
                // Subnormal: normalize into a float32 normal.
                int e    = 0;
                uint32_t m = mant;
                while ((m & 0x0400u) == 0u) { m <<= 1; ++e; }
                m &= 0x03FFu;
                bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
            }
        } else if (exp == 0x1Fu) {
            bits = sign | 0x7F800000u | (mant << 13);// inf / NaN
        } else {
            bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

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

        // ── Native G-buffer AOV readback (lossless float / int) ──────────
        // These decode the *last rendered frame's* G-buffer attachment straight
        // from its native GPU format via VulkanRendererCore::readGBufferAOV — no
        // 8-bit swapchain round-trip, no id hashing. The *_last() helpers assume a
        // frame is already on screen (call render()/drive first); the public
        // read_* wrappers below drive a frame themselves for one-shot use.

        // Metric depth (H, W) float32 — distance from the camera in scene units.
        // Background (cleared to the far plane) reads as `far`. Full 32-bit
        // precision from the D32 depth buffer (the old path quantized to 24 bits).
        py::array_t<float> depth_last(Camera& camera) {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!renderer_.readGBufferAOV(VulkanRenderer::GBufferAOV::Depth, raw, w, h, bpp) ||
                w <= 0 || h <= 0) {
                return py::array_t<float>({py::ssize_t(0), py::ssize_t(0)});
            }
            py::array_t<float> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
            auto* dst        = arr.mutable_data();
            const auto* d    = reinterpret_cast<const float*>(raw.data());// D32_SFLOAT
            const float nearP = camera.nearPlane, farP = camera.farPlane;
            const size_t px   = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i) {
                const float z = d[i];// reverse-Z NDC depth [0,1] (1=near, 0=far)
                dst[i] = (z <= 0.f) ? farP : (nearP * farP) / (nearP + z * (farP - nearP));
            }
            return arr;
        }

        // World-space unit normals (H, W, 3) float32, components in [-1, 1].
        py::array_t<float> normals_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!renderer_.readGBufferAOV(VulkanRenderer::GBufferAOV::Normal, raw, w, h, bpp) ||
                w <= 0 || h <= 0) {
                return py::array_t<float>({py::ssize_t(0), py::ssize_t(0), py::ssize_t(3)});
            }
            py::array_t<float> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w), py::ssize_t(3)});
            auto* dst     = arr.mutable_data();
            const auto* s = reinterpret_cast<const uint16_t*>(raw.data());// RGBA16F, xyz = n*0.5+0.5
            const size_t px = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i)
                for (int c = 0; c < 3; ++c)
                    dst[i * 3 + c] = half_to_float(s[i * 4 + c]) * 2.f - 1.f;
            return arr;
        }

        // Stable, recoverable integer instance ids (H, W) uint32. 0 = sky / no
        // hit; otherwise a per-object id that persists across frames (outIds.y).
        // No hashing, no collisions. Assign specific ids with set_instance_id.
        py::array_t<uint32_t> ids_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!renderer_.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, w, h, bpp) ||
                w <= 0 || h <= 0) {
                return py::array_t<uint32_t>({py::ssize_t(0), py::ssize_t(0)});
            }
            py::array_t<uint32_t> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
            auto* dst     = arr.mutable_data();
            const auto* s = reinterpret_cast<const uint16_t*>(raw.data());// RGBA16_UINT, .y = stable id
            const size_t px = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i) dst[i] = static_cast<uint32_t>(s[i * 4 + 1]);
            return arr;
        }

        // Semantic class ids (H, W) uint32 from outIds.z bits 8..15. 0 = unset;
        // tag objects with set_class_id. Gives semantic segmentation alongside
        // the instance ids above, from the same G-buffer read.
        py::array_t<uint32_t> class_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!renderer_.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, w, h, bpp) ||
                w <= 0 || h <= 0) {
                return py::array_t<uint32_t>({py::ssize_t(0), py::ssize_t(0)});
            }
            py::array_t<uint32_t> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
            auto* dst     = arr.mutable_data();
            const auto* s = reinterpret_cast<const uint16_t*>(raw.data());// RGBA16_UINT, .z = flags|class
            const size_t px = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i) dst[i] = static_cast<uint32_t>((s[i * 4 + 2] >> 8) & 0xFFu);
            return arr;
        }

        // Screen-space motion vectors (H, W, 2) float32, in PIXELS: where each
        // surface was last frame minus where it is now (prev - curr). +x is
        // rightward; the y sign follows the Vulkan NDC (down-positive) convention.
        py::array_t<float> motion_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!renderer_.readGBufferAOV(VulkanRenderer::GBufferAOV::Motion, raw, w, h, bpp) ||
                w <= 0 || h <= 0) {
                return py::array_t<float>({py::ssize_t(0), py::ssize_t(0), py::ssize_t(2)});
            }
            py::array_t<float> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w), py::ssize_t(2)});
            auto* dst     = arr.mutable_data();
            const auto* s = reinterpret_cast<const uint16_t*>(raw.data());// RGBA16F, .xy = NDC delta
            const float sx = 0.5f * static_cast<float>(w);
            const float sy = 0.5f * static_cast<float>(h);
            const size_t px = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i) {
                dst[i * 2 + 0] = half_to_float(s[i * 4 + 0]) * sx;// NDC delta → pixels
                dst[i * 2 + 1] = half_to_float(s[i * 4 + 1]) * sy;
            }
            return arr;
        }

        // Linear base colour + metalness (H, W, 4) uint8: rgb = linear albedo,
        // a = metalness. Native G-buffer read (no debug-blit re-render).
        py::array_t<uint8_t> albedo_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!renderer_.readGBufferAOV(VulkanRenderer::GBufferAOV::Albedo, raw, w, h, bpp) ||
                w <= 0 || h <= 0) {
                return py::array_t<uint8_t>({py::ssize_t(0), py::ssize_t(0), py::ssize_t(4)});
            }
            py::array_t<uint8_t> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w), py::ssize_t(4)});
            std::memcpy(arr.mutable_data(), raw.data(), raw.size());// RGBA8, tightly packed
            return arr;
        }

        py::array_t<float> read_depth(Object3D& scene, Camera& camera) {
            drive_frames(scene, camera);
            return depth_last(camera);
        }
        py::array_t<float> read_normals_float(Object3D& scene, Camera& camera) {
            drive_frames(scene, camera);
            return normals_last();
        }
        py::array_t<uint32_t> read_instance_ids(Object3D& scene, Camera& camera) {
            drive_frames(scene, camera);
            return ids_last();
        }
        py::array_t<uint32_t> read_class_ids(Object3D& scene, Camera& camera) {
            drive_frames(scene, camera);
            return class_last();
        }
        py::array_t<float> read_motion(Object3D& scene, Camera& camera) {
            drive_frames(scene, camera);
            return motion_last();
        }

        // Label assignment for the segmentation AOVs. instance id overrides the
        // auto-assigned stable id (outIds.y); class id (0..255) tags the object
        // for semantic segmentation (outIds.z). Take effect on the next render.
        void set_instance_id(Object3D& obj, uint32_t id) { renderer_.setObjectInstanceId(obj, id); }
        void set_class_id(Object3D& obj, uint32_t cls) { renderer_.setObjectClassId(obj, cls); }

        // Render ONCE, then read every requested AOV from that same frame, each
        // as its natural numpy dtype: depth (H,W) f32 · normals (H,W,3) f32 ·
        // instance_ids (H,W) u32 · motion (H,W,2) f32 · rgb/segmentation/albedo
        // (H,W,3) u8. The efficient multi-AOV entry point for dataset generation.
        py::dict read_aovs_typed(Object3D& scene, Camera& camera,
                                 const std::vector<std::string>& aovs) {
            drive_frames(scene, camera);
            py::dict out;
            for (const auto& name : aovs) {
                if (name == "depth") {
                    out[py::str(name)] = depth_last(camera);
                } else if (name == "normals" || name == "normal") {
                    out[py::str(name)] = normals_last();
                } else if (name == "instance_ids" || name == "ids" || name == "segmentation") {
                    out[py::str(name)] = ids_last();
                } else if (name == "class_ids" || name == "class" || name == "semantic") {
                    out[py::str(name)] = class_last();
                } else if (name == "motion" || name == "flow") {
                    out[py::str(name)] = motion_last();
                } else if (name == "rgb" || name == "shaded" || name == "color") {
                    out[py::str(name)] = to_numpy(renderer_.readRGBPixels());
                } else if (name == "albedo") {
                    out[py::str(name)] = albedo_last();
                } else {
                    throw std::invalid_argument("unknown typed AOV '" + name +
                            "' — use: depth, normals, instance_ids, class_ids, motion, rgb, albedo");
                }
            }
            return out;
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
                     "Background reads as the camera far plane. Full 32-bit precision "
                     "(native D32 read; supersedes the old 24-bit-packed path).")
                // ── Lossless float / int AOV readback (native G-buffer copy) ──
                .def("read_instance_ids", &PyVulkanRenderer::read_instance_ids,
                     py::arg("scene"), py::arg("camera"),
                     "Stable per-pixel instance ids as (H, W) uint32. 0 = sky / no hit; otherwise "
                     "a per-object id that persists across frames and visible-set changes (no "
                     "hashing, no collisions). Auto-assigned; override with set_instance_id().")
                .def("read_class_ids", &PyVulkanRenderer::read_class_ids,
                     py::arg("scene"), py::arg("camera"),
                     "Semantic class ids as (H, W) uint32 (0..255; 0 = unset). Tag objects with "
                     "set_class_id() to get semantic segmentation alongside the instance ids.")
                .def("set_instance_id", &PyVulkanRenderer::set_instance_id,
                     py::arg("object"), py::arg("instance_id"),
                     "Assign a specific stable instance id (0..65535) to an object for the ids "
                     "AOV. Overrides the auto-assigned id. Takes effect on the next render.")
                .def("set_class_id", &PyVulkanRenderer::set_class_id,
                     py::arg("object"), py::arg("class_id"),
                     "Tag an object with a semantic class id (0..255) for read_class_ids(). "
                     "Objects sharing a class id share a semantic label.")
                .def("read_normals_float", &PyVulkanRenderer::read_normals_float,
                     py::arg("scene"), py::arg("camera"),
                     "World-space unit normals as (H, W, 3) float32, components in [-1, 1] "
                     "(full precision; read_normals() stays the 8-bit visualisation).")
                .def("read_motion", &PyVulkanRenderer::read_motion,
                     py::arg("scene"), py::arg("camera"),
                     "Screen-space motion vectors as (H, W, 2) float32, in pixels "
                     "(previous - current surface position; +x rightward, y down-positive).")
                .def("read_aovs_typed", &PyVulkanRenderer::read_aovs_typed,
                     py::arg("scene"), py::arg("camera"),
                     py::arg("aovs") = std::vector<std::string>{"rgb", "depth", "normals", "instance_ids"},
                     "Render ONCE and read every requested AOV from that same frame, each as "
                     "its natural dtype: depth (H,W) f32 · normals (H,W,3) f32 · instance_ids "
                     "(H,W) u32 · motion (H,W,2) f32 · rgb (H,W,3) u8 · albedo (H,W,4) u8 "
                     "(linear rgb + metalness). The efficient multi-AOV entry point.")
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
                // Per-frame CPU/GPU pass timings (milliseconds) — see
                // VulkanRendererCore::FrameTimings. For perf triage from python.
                .def_property_readonly("frame_timings",
                                       [](PyVulkanRenderer& r) {
                                           const auto t = r.native().lastFrameTimings();
                                           py::dict d;
                                           d["photon_emit_ms"] = t.photonEmitMs;
                                           d["shade_ms"] = t.pathTraceMs;
                                           d["denoise_ms"] = t.denoiseMs;
                                           d["taa_ms"] = t.taaMs;
                                           d["raster_gbuf_ms"] = t.rasterGbufMs;
                                           d["gbuf_resolve_ms"] = t.gbufResolveMs;
                                           d["shade_b_ms"] = t.shadeBMs;
                                           d["overlay_ms"] = t.overlayMs;
                                           d["dof_ms"] = t.dofMs;
                                           d["cpu_ensure_scene_ms"] = t.cpuEnsureSceneMs;
                                           d["cpu_record_ms"] = t.cpuRecordMs;
                                           d["cpu_frame_ms"] = t.cpuFrameMs;
                                           return d;
                                       })
                // ── Physical camera + photometric light units ─────────────────
                .def_property("physical_camera",
                              [](PyVulkanRenderer& r) { return r.native().physicalCamera(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setPhysicalCamera(v); },
                              "Derive exposure from aperture/shutter/ISO (EV100) instead of "
                              "tone_mapping_exposure; the HDR target is pre-exposed so 100k-lux "
                              "daylight survives fp16. Defaults = sunny-16 (f/16, 1/125 s, ISO "
                              "100). Pair with physical_light_units. Default off.")
                .def("set_camera_exposure",
                     [](PyVulkanRenderer& r, float aperture, float shutter, float iso) {
                         r.native().setCameraExposure(aperture, shutter, iso);
                     },
                     py::arg("aperture") = 16.f, py::arg("shutter") = 1.f / 125.f,
                     py::arg("iso") = 100.f,
                     "Camera exposure triplet (used while physical_camera is on): "
                     "f-number, shutter seconds, ISO.")
                .def_property("exposure_compensation",
                              [](PyVulkanRenderer& r) { return r.native().exposureCompensation(); },
                              [](PyVulkanRenderer& r, float v) { r.native().setExposureCompensation(v); },
                              "EV compensation while physical_camera is on (+1 doubles brightness).")
                .def_property("physical_light_units",
                              [](PyVulkanRenderer& r) { return r.native().physicalLightUnits(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setPhysicalLightUnits(v); },
                              "Interpret light intensities photometrically: directional = lux "
                              "(sun ~100000), point/spot = lumens, rect/emissive = nits. Default off.")
                // ── Depth of field (thin lens, post) ──────────────────────────
                .def_property("depth_of_field",
                              [](PyVulkanRenderer& r) { return r.native().depthOfField(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setDepthOfField(v); },
                              "Thin-lens bokeh on the HDR scene (before bloom/TAA). CoC comes "
                              "from the camera: set_camera_exposure's f-number (independent of "
                              "physical_camera), FOV-derived focal length, focus_distance. "
                              "Default off (zero cost).")
                .def_property("focus_distance",
                              [](PyVulkanRenderer& r) { return r.native().focusDistance(); },
                              [](PyVulkanRenderer& r, float v) { r.native().setFocusDistance(v); },
                              "Focus plane distance in scene units/meters (default 10).")
                // ── White balance + colour grade (post composite) ─────────────
                .def("set_white_balance",
                     [](PyVulkanRenderer& r, float temperatureK, float tint) {
                         r.native().setWhiteBalance(temperatureK, tint);
                     },
                     py::arg("temperature") = 6500.f, py::arg("tint") = 0.f,
                     "Scene-illuminant white balance: Kelvin on the Planckian locus "
                     "(6500 = neutral = off), tint green(-)/magenta(+).")
                .def("set_color_grade",
                     [](PyVulkanRenderer& r, const Vector3& lift, const Vector3& gamma,
                        const Vector3& gain, float saturation, float contrast) {
                         VulkanRendererCore::ColorGrade g;
                         g.lift = lift;
                         g.gamma = gamma;
                         g.gain = gain;
                         g.saturation = saturation;
                         g.contrast = contrast;
                         r.native().setColorGrade(g);
                     },
                     py::arg("lift") = Vector3(0, 0, 0), py::arg("gamma") = Vector3(1, 1, 1),
                     py::arg("gain") = Vector3(1, 1, 1), py::arg("saturation") = 1.f,
                     py::arg("contrast") = 1.f,
                     "Lift/gamma/gain wheels + saturation + contrast, baked into a 33^3 "
                     "LUT applied after the tone map. Defaults = identity = off.")
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
                // Hybrid SSR→RT reflections: HiZ screen-space march first, RT ray
                // wherever the screen cannot answer. A perf fast path — never a
                // quality authority (any doubtful hit falls back to the RT ray).
                .def_property("ssr_reflections",
                              [](PyVulkanRenderer& r) { return r.native().ssrReflections(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setSsrReflections(v); },
                              "Toggle hybrid SSR→RT reflections (screen-space fast path with "
                              "RT fallback; quality-neutral). Default off — enable on scenes "
                              "where RT reflection rays dominate the frame.")
                // World-space irradiance probe grid (DDGI-lite): multi-bounce GI, and
                // the switch from cosmetic to MEASURED ambient — enclosed interiors
                // stop being "lit with no light" (ungated ambient/env-specular) and
                // receive only what actually bounces in through openings. Default ON;
                // converges over a few dozen frames after enable/scene load.
                .def_property("probe_gi",
                              [](PyVulkanRenderer& r) { return r.native().probeGI(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setProbeGI(v); },
                              "Toggle the world-space irradiance probe grid (multi-bounce GI + "
                              "occlusion-correct ambient). Default ON; needs deferred_ao + "
                              "denoise on. Interiors read physically dark — pair with "
                              "auto_exposure or a raised tone_mapping_exposure. False restores "
                              "the legacy cosmetic ambient.")
                // HDRI sun extraction: the env map's dominant bright disc is removed
                // from the glossy/rough PMREM mips (kills the bright "spec blob"
                // reflections) and re-injected as an analytic directional light with
                // RT shadows. Default on. NOTE: enclosed interiors lit only through
                // openings read darker (the old look leaked the sun into the ambient
                // everywhere, unshadowed); disable per scene if that ambient is wanted.
                .def_property("env_sun_extraction",
                              [](PyVulkanRenderer& r) { return r.native().envSunExtraction(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setEnvSunExtraction(v); },
                              "Extract the HDRI sun into an analytic light (default on). "
                              "Toggling rebuilds the environment on the next frame.")
                // ONE-SUN POLICY: 'auto' (default) injects the extracted sun only
                // while the scene has no visible DirectionalLight of its own (an
                // explicit scene light claims the sun role — otherwise the scene
                // gets TWO directional shadows); 'always' injects regardless;
                // 'off' disables extraction entirely (raw env, legacy spec blobs).
                .def_property("env_sun_policy",
                              [](PyVulkanRenderer& r) -> std::string {
                                  switch (r.native().envSunPolicy()) {
                                      case VulkanRenderer::EnvSunPolicy::Always: return "always";
                                      case VulkanRenderer::EnvSunPolicy::Off: return "off";
                                      default: return "auto";
                                  }
                              },
                              [](PyVulkanRenderer& r, const std::string& v) {
                                  if (v == "auto") r.native().setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Auto);
                                  else if (v == "always") r.native().setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Always);
                                  else if (v == "off") r.native().setEnvSunPolicy(VulkanRenderer::EnvSunPolicy::Off);
                                  else throw std::invalid_argument("env_sun_policy: expected 'auto', 'always' or 'off'");
                              },
                              "'auto' (a scene DirectionalLight claims the sun role), 'always', or 'off'.")
                .def_property_readonly("env_sun_found",
                                       [](PyVulkanRenderer& r) { return r.native().envSunFound(); },
                                       "True when the current environment has a detected sun disc.")
                .def_property_readonly("env_sun_direction",
                                       [](PyVulkanRenderer& r) {
                                           const auto d = r.native().envSunDirection();
                                           return std::array<float, 3>{d.x, d.y, d.z};
                                       },
                                       "Unit direction TOWARD the detected env sun (valid when env_sun_found). "
                                       "Use to align an explicit DirectionalLight with the HDRI.")
                .def_property_readonly("env_sun_color",
                                       [](PyVulkanRenderer& r) {
                                           const auto c = r.native().envSunColor();
                                           return std::array<float, 3>{c.x, c.y, c.z};
                                       },
                                       "Integrated sun-disc energy (linear RGB irradiance, valid when env_sun_found).")
                // MSAA G-buffer (1/2/4 samples, default 1 = off): kills the 1-spp
                // jittered-coverage edge flicker at the source — dominant-sample
                // resolve + coverage-blended edges (per-sample shading of the
                // geometry minority + sky blending). Rasterizes UNJITTERED when
                // > 1. VRAM cost ≈ samples× the raster attachments; 2 is the
                // recommended step for foliage/low-poly-heavy scenes.
                .def_property("gbuffer_msaa",
                              [](PyVulkanRenderer& r) { return r.native().gbufferMsaa(); },
                              [](PyVulkanRenderer& r, uint32_t s) { r.native().setGbufferMsaa(s); },
                              "G-buffer MSAA sample count (1, 2 or 4; default 1 = off). "
                              "Stabilizes silhouette/edge flicker in the deferred renderer.")
                // Sun angular radius (deg) → soft directional shadows (adaptive
                // multi-ray penumbra). 0 = hard single-ray shadow. Real sun ~0.27°.
                .def_property("sun_angular_radius",
                              [](PyVulkanRenderer& r) { return r.native().sunAngularRadius(); },
                              [](PyVulkanRenderer& r, float d) { r.native().setSunAngularRadius(d); },
                              "Directional-light angular radius in degrees for soft sun "
                              "shadows (default 0.5; 0 = hard shadow).")
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
                              [](PyVulkanRenderer& r, bool v) { r.native().setVolumetricFog(v); })
                // Automatic exposure (eye adaptation). When enabled the renderer
                // samples the scene's log-luma histogram each frame and adapts
                // toneMappingExposure automatically. toneMappingExposure is ignored
                // while auto_exposure is True.
                .def_property("auto_exposure",
                              [](PyVulkanRenderer& r) { return r.native().autoExposure(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setAutoExposure(v); },
                              "Toggle automatic exposure / eye adaptation (default off). "
                              "Drives tone-mapping exposure toward 18% gray for the scene's "
                              "weighted-average luminance. tone_mapping_exposure is ignored "
                              "while this is True.")
                .def("set_auto_exposure_speed",
                     [](PyVulkanRenderer& r, float s) { r.native().setAutoExposureSpeed(s); },
                     py::arg("ev_per_second"),
                     "Adaptation speed in EV/s (default 2.0). Darkening is applied at 0.5× speed.")
                .def("set_auto_exposure_range",
                     [](PyVulkanRenderer& r, float lo, float hi) { r.native().setAutoExposureRange(lo, hi); },
                     py::arg("min_ev"), py::arg("max_ev"),
                     "EV clamp for auto-exposure relative to linear 1.0 (default -3 to +3).");

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
