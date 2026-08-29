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
//     straight to host memory via VulkanRenderer::readGBufferAOV — full-
//     precision depth (f32), world normals (f32), RECOVERABLE integer instance
//     ids (u32, no hashing) and metric motion (f32). This is the material an ML
//     / sensor pipeline actually trains on.
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_VULKAN

// functional.h is here for enable_vertex_interop's per-frame callback — the
// first callback this file binds. It is deliberately the std::function caster
// rather than the manual py::function + py::gil_scoped_acquire idiom used in
// bind_physx.cpp: pybind11's func_handle re-acquires the GIL on BOTH invoke and
// destruction, and both matter here. PyVulkanRenderer::render releases the GIL
// for the whole frame, the callback fires inside it, and the renderer may also
// DROP the callback from inside that same frame (a topology change disables
// interop mid-ensureSceneBuilt) — which destroys the std::function with the GIL
// released.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/helpers/PathTracedLidarSensor.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace py = pybind11;// the threepp_py::py alias isn't visible in the anon namespace below
using namespace threepp;

namespace {

    // Is the Vulkan LOADER present at runtime? The two platforms differ:
    //
    //   Windows: the wheel links vulkan-1.dll with /DELAYLOAD (python/CMakeLists),
    //   so `import threepp` succeeds on machines with no Vulkan runtime — but the
    //   first delay-loaded call on such a machine raises a structured exception,
    //   not a C++ one, which no Python except can catch. This probe is what turns
    //   that crash into the clean RuntimeError below.
    //
    //   Linux: the loader is a normal link dependency; auditwheel vendors
    //   libvulkan.so.1 into the wheel, so if the module imported, the loader is
    //   loaded. Present by construction — a machine without a GPU/ICD then fails
    //   at VulkanContext creation with a catchable std::runtime_error
    //   ("no Vulkan-capable GPU found"), which pybind surfaces as RuntimeError.
    bool vulkan_loader_present() {
#ifdef _WIN32
        return ::LoadLibraryW(L"vulkan-1.dll") != nullptr;
#else
        return true;
#endif
    }

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
                // GIL released for the GPU frame. Python callbacks reachable
                // from inside (window-resize etc.) arrive through pybind11's
                // functional caster, which re-acquires per invocation.
                py::gil_scoped_release release;
                renderer_.render(scene, camera);
            } else {
                drive_frames(scene, camera);// releases the GIL itself
            }
        }

        py::array_t<uint8_t> read_pixels() { return to_numpy(read_rgb_released()); }

        // Scene-only swapchain capture: the post-TAA / pre-overlay frame, WITHOUT
        // any sprite / ImGui overlays composited on top — what sensor pipelines
        // need to avoid feeding their own overlay back in. Reuses to_numpy →
        // (H, W, 3) uint8. Enable once, then read after a render(); the per-frame
        // copy costs only while enabled.
        void set_scene_capture(bool enabled) { renderer_.setSceneCaptureEnabled(enabled); }
        bool scene_capture() const { return renderer_.sceneCaptureEnabled(); }
        py::array_t<uint8_t> read_scene_pixels() { return to_numpy(read_scene_rgb_released()); }

        // ── Multi-view: N cameras per frame ─────────────────────────────
        // Every render() produces the primary AND every added view from ONE
        // scene build, in a single submission — N viewpoints of the SAME
        // simulated instant, which N render() calls can never give. Views are
        // PERSISTENT: addView drains the device and allocates a full deferred
        // chain, so add once and render every frame, never add/remove per frame.
        uint32_t add_view(Camera& camera, int width, int height) {
            py::gil_scoped_release release;// device-idle + allocation
            return renderer_.addView(camera, width, height);
        }
        bool remove_view(uint32_t handle) {
            py::gil_scoped_release release;
            return renderer_.removeView(handle);
        }
        bool set_view_camera(uint32_t handle, Camera& camera) {
            return renderer_.setViewCamera(handle, camera);
        }
        bool set_view_display_rect(uint32_t handle, int x, int y, int w, int h) {
            return renderer_.setViewDisplayRect(handle, x, y, w, h);
        }
        bool hide_view(uint32_t handle) { return renderer_.hideView(handle); }

        // (w, h) as passed to add_view, or None for an unknown handle.
        py::object view_size(uint32_t handle) const {
            int w = 0, h = 0;
            if (!renderer_.viewSize(handle, w, h)) return py::none();
            return py::make_tuple(w, h);
        }

        py::array_t<uint8_t> read_view_rgb_pixels(uint32_t handle) {
            int w = 0, h = 0;
            if (!renderer_.viewSize(handle, w, h) || w <= 0 || h <= 0) {
                return py::array_t<uint8_t>({py::ssize_t(0), py::ssize_t(0), py::ssize_t(3)});
            }
            std::vector<unsigned char> px;
            {
                py::gil_scoped_release release;// vkDeviceWaitIdle + staging copy
                px = renderer_.readViewRGBPixels(handle);
            }
            py::array_t<uint8_t> arr({static_cast<py::ssize_t>(h),
                                      static_cast<py::ssize_t>(w),
                                      static_cast<py::ssize_t>(3)});
            const size_t want = static_cast<size_t>(w) * h * 3;
            if (px.size() >= want) {
                std::memcpy(arr.mutable_data(), px.data(), want);
            } else {
                std::memset(arr.mutable_data(), 0, want);
            }
            return arr;
        }

        // Viewport / scissor (forwarded to the native renderer; (x,y,w,h) overload).
        void set_viewport(int x, int y, int w, int h) { renderer_.setViewport(x, y, w, h); }
        void set_scissor(int x, int y, int w, int h) { renderer_.setScissor(x, y, w, h); }
        void set_scissor_test(bool enabled) { renderer_.setScissorTest(enabled); }

        // ── Native G-buffer AOV readback (lossless float / int) ──────────
        // These decode the *last rendered frame's* G-buffer attachment straight
        // from its native GPU format via VulkanRenderer::readGBufferAOV — no
        // 8-bit swapchain round-trip, no id hashing. The *_last() helpers assume a
        // frame is already on screen (call render()/drive first); the public
        // read_* wrappers below drive a frame themselves for one-shot use.

        // Each AOV is a pure decode over the fetched attachment bytes, shared
        // by the single *_last() fetches and the batched read_aovs_typed —
        // decodes run with the GIL held; only the fetch releases it.

        // Metric depth (H, W) float32 — distance from the camera in scene units.
        // Background (cleared to the far plane) reads as `far`. Full 32-bit
        // precision from the D32 depth buffer (the old path quantized to 24 bits).
        static py::array_t<float> decode_depth(const std::vector<uint8_t>& raw, int w, int h,
                                               const Camera& camera) {
            if (raw.empty() || w <= 0 || h <= 0) {
                // ShapeContainer spelled out: a braced {0, 0} is ambiguous to
                // gcc-14 (constant zeros also convert toward the buffer_info /
                // pointer overloads); non-zero shapes like {n, 7} are not.
                return py::array_t<float>(py::array::ShapeContainer{0, 0});
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

        py::array_t<float> depth_last(Camera& camera) {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!read_aov_released(VulkanRenderer::GBufferAOV::Depth, raw, w, h, bpp)) w = h = 0;
            return decode_depth(raw, w, h, camera);
        }

        // World-space unit normals (H, W, 3) float32, components in [-1, 1].
        static py::array_t<float> decode_normals(const std::vector<uint8_t>& raw, int w, int h) {
            if (raw.empty() || w <= 0 || h <= 0) {
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

        py::array_t<float> normals_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!read_aov_released(VulkanRenderer::GBufferAOV::Normal, raw, w, h, bpp)) w = h = 0;
            return decode_normals(raw, w, h);
        }

        // Stable, recoverable integer instance ids (H, W) uint32. 0 = sky / no
        // hit; otherwise a per-object id that persists across frames (outIds.y).
        // No hashing, no collisions. Assign specific ids with set_instance_id.
        static py::array_t<uint32_t> decode_ids(const std::vector<uint8_t>& raw, int w, int h) {
            if (raw.empty() || w <= 0 || h <= 0) {
                return py::array_t<uint32_t>(py::array::ShapeContainer{0, 0});// see decode_depth
            }
            py::array_t<uint32_t> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
            auto* dst     = arr.mutable_data();
            const auto* s = reinterpret_cast<const uint16_t*>(raw.data());// RGBA16_UINT, .y = stable id
            const size_t px = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i) dst[i] = static_cast<uint32_t>(s[i * 4 + 1]);
            return arr;
        }

        py::array_t<uint32_t> ids_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!read_aov_released(VulkanRenderer::GBufferAOV::Ids, raw, w, h, bpp)) w = h = 0;
            return decode_ids(raw, w, h);
        }

        // Semantic class ids (H, W) uint32 from outIds.z bits 8..15. 0 = unset;
        // tag objects with set_class_id. Gives semantic segmentation alongside
        // the instance ids above, from the same G-buffer read.
        static py::array_t<uint32_t> decode_class(const std::vector<uint8_t>& raw, int w, int h) {
            if (raw.empty() || w <= 0 || h <= 0) {
                return py::array_t<uint32_t>(py::array::ShapeContainer{0, 0});// see decode_depth
            }
            py::array_t<uint32_t> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
            auto* dst     = arr.mutable_data();
            const auto* s = reinterpret_cast<const uint16_t*>(raw.data());// RGBA16_UINT, .z = flags|class
            const size_t px = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i) dst[i] = static_cast<uint32_t>((s[i * 4 + 2] >> 8) & 0xFFu);
            return arr;
        }

        py::array_t<uint32_t> class_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!read_aov_released(VulkanRenderer::GBufferAOV::Ids, raw, w, h, bpp)) w = h = 0;
            return decode_class(raw, w, h);
        }

        // Screen-space motion vectors (H, W, 2) float32, in PIXELS: where each
        // surface was last frame minus where it is now (prev - curr). +x is
        // rightward; the y sign follows the Vulkan NDC (down-positive) convention.
        static py::array_t<float> decode_motion(const std::vector<uint8_t>& raw, int w, int h) {
            if (raw.empty() || w <= 0 || h <= 0) {
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

        py::array_t<float> motion_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!read_aov_released(VulkanRenderer::GBufferAOV::Motion, raw, w, h, bpp)) w = h = 0;
            return decode_motion(raw, w, h);
        }

        // Linear base colour + metalness (H, W, 4) uint8: rgb = linear albedo,
        // a = metalness. Native G-buffer read (no debug-blit re-render).
        static py::array_t<uint8_t> decode_albedo(const std::vector<uint8_t>& raw, int w, int h) {
            if (raw.empty() || w <= 0 || h <= 0) {
                return py::array_t<uint8_t>({py::ssize_t(0), py::ssize_t(0), py::ssize_t(4)});
            }
            py::array_t<uint8_t> arr({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w), py::ssize_t(4)});
            std::memcpy(arr.mutable_data(), raw.data(), raw.size());// RGBA8, tightly packed
            return arr;
        }

        py::array_t<uint8_t> albedo_last() {
            std::vector<uint8_t> raw;
            int w = 0, h = 0, bpp = 0;
            if (!read_aov_released(VulkanRenderer::GBufferAOV::Albedo, raw, w, h, bpp)) w = h = 0;
            return decode_albedo(raw, w, h);
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

        // ── Frame zero-copy interop (Vulkan -> CUDA), "frames out" ───────
        // Names, not the enum, at the boundary: every other AOV entry point
        // here takes a string ("depth", "albedo", ...) and the enum is bound
        // beside them for callers who prefer it. Both are accepted.
        static bool parse_frame_channel(const py::handle& h,
                                        VulkanRenderer::FrameChannel& out) {
            using FC = VulkanRenderer::FrameChannel;
            if (py::isinstance<FC>(h)) {
                out = h.cast<FC>();
                return true;
            }
            const auto n = h.cast<std::string>();
            if (n == "color") { out = FC::Color; return true; }
            if (n == "depth") { out = FC::Depth; return true; }
            if (n == "normal" || n == "normals") { out = FC::Normal; return true; }
            if (n == "motion" || n == "flow") { out = FC::Motion; return true; }
            if (n == "ids" || n == "instance_ids" || n == "segmentation") { out = FC::Ids; return true; }
            if (n == "albedo") { out = FC::Albedo; return true; }
            if (n == "splat_depth" || n == "splatdepth") { out = FC::SplatDepth; return true; }
            throw std::invalid_argument("unknown frame interop channel '" + n +
                                        "' (color/depth/normal/motion/ids/albedo/splat_depth)");
        }
        static const char* frame_channel_name(VulkanRenderer::FrameChannel c) {
            using FC = VulkanRenderer::FrameChannel;
            switch (c) {
                case FC::Color: return "color";
                case FC::Depth: return "depth";
                case FC::Normal: return "normal";
                case FC::Motion: return "motion";
                case FC::Ids: return "ids";
                case FC::Albedo: return "albedo";
                case FC::SplatDepth: return "splat_depth";
            }
            return "?";
        }

        py::list enable_frame_interop(uint32_t view, const py::iterable& channels) {
            std::vector<VulkanRenderer::FrameChannel> wanted;
            for (const auto& item : channels) {
                VulkanRenderer::FrameChannel c{};
                if (parse_frame_channel(item, c)) wanted.push_back(c);
            }
            std::vector<VulkanRenderer::FrameInteropExport> got;
            if (!wanted.empty()) {
                py::gil_scoped_release release;// allocation + a possible drain
                got = renderer_.enableFrameInterop(view, wanted);
            }
            py::list out;
            for (const auto& e : got) {
                py::dict d;
                d["channel"]         = frame_channel_name(e.channel);
                d["handle"]          = reinterpret_cast<uintptr_t>(e.osHandle);
                d["size_bytes"]      = e.sizeBytes;
                d["width"]           = e.width;
                d["height"]          = e.height;
                d["bytes_per_pixel"] = e.bytesPerPixel;
                d["bgra"]            = e.bgra;
                out.append(std::move(d));
            }
            return out;
        }
        void disable_frame_interop(uint32_t view) {
            py::gil_scoped_release release;// device drain
            renderer_.disableFrameInterop(view);
        }
        bool sync_frame_interop() {
            py::gil_scoped_release release;// one fence wait
            return renderer_.syncFrameInterop();
        }
        bool frame_interop_active(uint32_t view) const {
            return renderer_.frameInteropActive(view);
        }

        // The raw bytes of one G-buffer attachment of the last rendered frame,
        // as (H, W, bytes_per_pixel) uint8 — no decode, no lens warp applied by
        // the caller's side. This is the host readback the frame-interop path
        // is checked byte-for-byte against; the decoded read_depth /
        // read_instance_ids helpers above are what applications use.
        py::object read_gbuffer_aov_raw(const std::string& name, uint32_t view) {
            using AOV = VulkanRenderer::GBufferAOV;
            AOV aov{};
            if (name == "depth") aov = AOV::Depth;
            else if (name == "normal" || name == "normals") aov = AOV::Normal;
            else if (name == "motion" || name == "flow") aov = AOV::Motion;
            else if (name == "ids" || name == "instance_ids" || name == "segmentation") aov = AOV::Ids;
            else if (name == "albedo") aov = AOV::Albedo;
            else if (name == "splat_depth" || name == "splatdepth") aov = AOV::SplatDepth;
            else throw std::invalid_argument("unknown AOV '" + name + "'");

            std::vector<VulkanRenderer::AOVReadback> got;
            {
                py::gil_scoped_release release;// device wait + staging copy
                (void) renderer_.readViewGBufferAOVs(view, {aov}, got);
            }
            if (got.empty()) return py::none();
            const auto& r = got.front();
            py::array_t<uint8_t> arr({static_cast<py::ssize_t>(r.height),
                                      static_cast<py::ssize_t>(r.width),
                                      static_cast<py::ssize_t>(r.bytesPerPixel)});
            std::memcpy(arr.mutable_data(), r.data.data(), r.data.size());
            return std::move(arr);
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

            // Map the requested names onto the attachments they need —
            // instance_ids and class_ids decode the SAME Ids read — and fetch
            // every needed attachment from the same frame in one batched call:
            // one device wait, one submit, one staging buffer, instead of a
            // full drain per AOV.
            using AOV     = VulkanRenderer::GBufferAOV;
            auto requested = [&](std::initializer_list<const char*> names) {
                for (const auto& n : aovs)
                    for (const char* c : names)
                        if (n == c) return true;
                return false;
            };
            std::vector<AOV> wanted;
            if (requested({"depth"})) wanted.push_back(AOV::Depth);
            if (requested({"normals", "normal"})) wanted.push_back(AOV::Normal);
            if (requested({"instance_ids", "ids", "segmentation",
                           "class_ids", "class", "semantic"})) wanted.push_back(AOV::Ids);
            if (requested({"motion", "flow"})) wanted.push_back(AOV::Motion);
            if (requested({"albedo"})) wanted.push_back(AOV::Albedo);

            std::vector<VulkanRenderer::AOVReadback> got;
            if (!wanted.empty()) {
                py::gil_scoped_release release;// one wait for the whole batch
                (void) renderer_.readGBufferAOVs(wanted, got);
            }
            const auto find = [&](AOV a) -> const VulkanRenderer::AOVReadback* {
                for (const auto& r : got)
                    if (r.aov == a) return &r;
                return nullptr;
            };
            // A missing entry decodes from an empty buffer, which yields the
            // same empty-shaped arrays the *_last() fetch failures return.
            static const std::vector<uint8_t> kEmpty;

            py::dict out;
            for (const auto& name : aovs) {
                if (name == "depth") {
                    const auto* r      = find(AOV::Depth);
                    out[py::str(name)] = decode_depth(r ? r->data : kEmpty, r ? r->width : 0,
                                                      r ? r->height : 0, camera);
                } else if (name == "normals" || name == "normal") {
                    const auto* r      = find(AOV::Normal);
                    out[py::str(name)] = decode_normals(r ? r->data : kEmpty, r ? r->width : 0,
                                                        r ? r->height : 0);
                } else if (name == "instance_ids" || name == "ids" || name == "segmentation") {
                    const auto* r      = find(AOV::Ids);
                    out[py::str(name)] = decode_ids(r ? r->data : kEmpty, r ? r->width : 0,
                                                    r ? r->height : 0);
                } else if (name == "class_ids" || name == "class" || name == "semantic") {
                    const auto* r      = find(AOV::Ids);
                    out[py::str(name)] = decode_class(r ? r->data : kEmpty, r ? r->width : 0,
                                                      r ? r->height : 0);
                } else if (name == "motion" || name == "flow") {
                    const auto* r      = find(AOV::Motion);
                    out[py::str(name)] = decode_motion(r ? r->data : kEmpty, r ? r->width : 0,
                                                       r ? r->height : 0);
                } else if (name == "rgb" || name == "shaded" || name == "color") {
                    out[py::str(name)] = to_numpy(read_rgb_released());
                } else if (name == "albedo") {
                    const auto* r      = find(AOV::Albedo);
                    out[py::str(name)] = decode_albedo(r ? r->data : kEmpty, r ? r->width : 0,
                                                       r ? r->height : 0);
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
            auto arr = to_numpy(read_rgb_released());
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
            // Full GPU frames — see the GIL policy on the leaf helpers below.
            py::gil_scoped_release release;
            for (int i = 0; i < flush_; ++i) {
                canvas_.animateOnce([&] { renderer_.render(scene, camera); });
            }
        }

        // ── GIL policy ──────────────────────────────────────────────────
        // Every def that stalls on the GPU (render, device-idle readbacks)
        // releases the GIL for the stall, so torch inference, dataset writers
        // and ROS spinners on other Python threads keep running. The release
        // lives ONLY in these leaf helpers (and drive_frames/render above):
        // pybind11's gil_scoped_release must not nest, so callers hold the
        // GIL and never wrap them again. numpy arrays are always built with
        // the GIL held.
        bool read_aov_released(VulkanRenderer::GBufferAOV aov,
                               std::vector<uint8_t>& raw, int& w, int& h, int& bpp) {
            py::gil_scoped_release release;// vkDeviceWaitIdle + staging copy
            return renderer_.readGBufferAOV(aov, raw, w, h, bpp);
        }
        std::vector<unsigned char> read_rgb_released() {
            py::gil_scoped_release release;
            return renderer_.readRGBPixels();
        }
        std::vector<unsigned char> read_scene_rgb_released() {
            py::gil_scoped_release release;
            return renderer_.readSceneRGBPixels();
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
            if (px.size() >= pxCount * ch && ch == 3) {
                // readRGBPixels hands back tightly packed RGB, so the channel
                // pick below would be byte-for-byte a memcpy — do the memcpy.
                std::memcpy(dst, px.data(), pxCount * 3);
            } else if (px.size() >= pxCount * ch && ch == 4) {
                for (size_t i = 0; i < pxCount; ++i) {
                    dst[i * 3 + 0] = px[i * 4 + 0];
                    dst[i * 3 + 1] = px[i * 4 + 1];
                    dst[i * 3 + 2] = px[i * 4 + 2];
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
        // Channels of the zero-copy frames-out path. enable_frame_interop also
        // accepts the equivalent lowercase strings, which is what the helpers
        // in threepp.torch_frames pass.
        py::enum_<VulkanRenderer::FrameChannel>(m, "FrameChannel",
                                                "A per-frame image the Vulkan renderer can export "
                                                "for zero-copy CUDA/torch consumption.")
                .value("Color", VulkanRenderer::FrameChannel::Color)
                .value("Depth", VulkanRenderer::FrameChannel::Depth)
                .value("Normal", VulkanRenderer::FrameChannel::Normal)
                .value("Motion", VulkanRenderer::FrameChannel::Motion)
                .value("Ids", VulkanRenderer::FrameChannel::Ids)
                .value("Albedo", VulkanRenderer::FrameChannel::Albedo)
                .value("SplatDepth", VulkanRenderer::FrameChannel::SplatDepth);

        py::class_<PyVulkanRenderer>(m, "VulkanRenderer")
                .def(py::init([](Canvas& c, int flush) {
                         if (!vulkan_loader_present()) {
                             throw std::runtime_error(
                                     "Vulkan loader (vulkan-1.dll) not found — this machine has no "
                                     "Vulkan runtime. The GL renderer (threepp.GLRenderer) works "
                                     "everywhere; VulkanRenderer needs a Vulkan-capable GPU driver. "
                                     "Check threepp.vulkan_available() before constructing.");
                         }
                         return std::make_unique<PyVulkanRenderer>(c, flush);
                     }),
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
                // ── Multi-view: N cameras per frame ──────────────────────
                .def("add_view", &PyVulkanRenderer::add_view,
                     py::arg("camera"), py::arg("width"), py::arg("height"), py::keep_alive<1, 2>(),
                     "Attach a persistent extra view. Every render() then produces the primary "
                     "AND every added view from one scene build, in a single queue submission — "
                     "N viewpoints of the SAME simulated instant, which N render() calls cannot "
                     "give. Each view has its own G-buffer, temporal history and camera state; "
                     "acceleration structures, lights, materials and probe GI are shared.\n\n"
                     "Views are PERSISTENT: this call drains the device and allocates a full "
                     "deferred chain, while rendering an existing view every frame is cheap. Do "
                     "NOT add and remove per frame.\n\n"
                     "Secondary views are deliberately plainer than the primary — native "
                     "resolution with the built-in temporal resolve, no DLSS/FSR, no occlusion "
                     "culling, no UI overlay, no depth of field, no lens or sensor model. They "
                     "are measurement cameras, not the display.\n\n"
                     "Returns a handle (> 0), or 0 if the view could not be created — notably "
                     "when render() has not run yet, since a view shares the primary's render "
                     "pass and pipelines.")
                .def("remove_view", &PyVulkanRenderer::remove_view, py::arg("handle"),
                     "Destroy the view and free everything it owns. False for an unknown handle. "
                     "Handles are never reused, so a stale one is inert rather than dangerous.")
                .def("set_view_camera", &PyVulkanRenderer::set_view_camera,
                     py::arg("handle"), py::arg("camera"), py::keep_alive<1, 3>(),
                     "Repoint a view at a different camera. Treated as a CUT: the view's temporal "
                     "history is dropped rather than reprojected across a discontinuity that "
                     "never happened in world space.")
                .def("read_view_rgb_pixels", &PyVulkanRenderer::read_view_rgb_pixels, py::arg("handle"),
                     "This view's most recent frame as (H, W, 3) uint8, TOP-LEFT origin — the "
                     "same convention as read_pixels. Reads the view's own colour image, never "
                     "the swapchain. An unknown handle gives an empty (0, 0, 3) array.")
                .def("set_view_display_rect", &PyVulkanRenderer::set_view_display_rect,
                     py::arg("handle"), py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"),
                     "Picture-in-picture: show this view inside the primary's frame with its "
                     "top-left corner at (x, y) in window pixels. The image is already resolved, "
                     "on the device and in the swapchain's format, so this is a single image copy "
                     "in the frame's own command buffer — no readback, no upload, no texture, no "
                     "second submission.\n\n"
                     "1:1 ONLY: width/height must equal the size the view was added at, and a "
                     "mismatch draws NOTHING rather than a filtered rescale. A rect running off "
                     "the window edge is clipped. Composited after the scene capture and before "
                     "the UI overlay, so ImGui and screen-space sprites still draw on top.")
                .def("hide_view", &PyVulkanRenderer::hide_view, py::arg("handle"),
                     "Back to a measurement camera: still rendered, still readable, no longer "
                     "drawn into the frame.")
                .def("view_size", &PyVulkanRenderer::view_size, py::arg("handle"),
                     "Pixel size of a view's output as (width, height), or None if the handle is "
                     "unknown.")
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
                // AMD FidelityFX FSR 3.1 temporal upscaler. Available only in a
                // build with -DTHREEPP_WITH_FSR=ON (Windows/Vulkan) that ships
                // amd_fidelityfx_vk.dll next to the module — see fsr_available.
                // Setting it while unavailable is a no-op (the built-in TAA
                // upscaler runs); the getter reflects whether FSR is the active
                // upscaler. Frame-to-frame switchable (resets temporal history).
                .def_property("fsr",
                              [](PyVulkanRenderer& r) { return r.native().fsr(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setFsr(v); },
                              "AMD FSR 3.1 upscaler on/off (no-op / False if unavailable — see fsr_available).")
                .def_property_readonly("fsr_available",
                              [](PyVulkanRenderer& r) { return r.native().fsrAvailable(); },
                              "True when FSR was compiled in and its context created on this GPU.")
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
                // Procedural stars on SKY pixels — hash-based points in
                // direction space, so they stay pixel-crisp at any resolution
                // or FOV instead of smearing the way a baked star map does.
                .def_property("starfield",
                              [](PyVulkanRenderer& r) { return r.native().deferredStarfield(); },
                              [](PyVulkanRenderer& r, float v) { r.native().setDeferredStarfield(v); },
                              "Procedural star field drawn on sky pixels. 0 disables (default); "
                              "~1.0 is a night sky. Ramp it with the daylight rather than snapping it on.")
                // Deterministic frame clock. Without this every wall-clock read
                // in the frame path runs on glfwGetTime(), so an offline render
                // whose frames take 80 ms of wall time animates the engine-side
                // fields ~5x faster than the dt the app integrates with. None or
                // any negative value releases the pin (C++ simTime() < 0); the
                // getter hands the wall-clock state back as None rather than
                // leaking the -1 sentinel, so `r.sim_time += dt` on an unpinned
                // renderer raises instead of silently staying on the wall clock.
                .def_property("sim_time",
                              [](PyVulkanRenderer& r) -> std::optional<double> {
                                  const double t = r.native().simTime();
                                  if (t < 0.0) return std::nullopt;
                                  return t;
                              },
                              [](PyVulkanRenderer& r, const std::optional<double>& v) {
                                  r.native().setSimTime(v.has_value() ? *v : -1.0);
                              },
                              "Simulation time in seconds that pins EVERY wall-clock read the frame\n"
                              "path makes, or None while the renderer runs on the wall clock (the\n"
                              "default). Pinned, it drives every renderer-side animated field: the\n"
                              "ocean/DisplacedMesh FFT deform and its foam decay, grass wind, the\n"
                              "clouds, the shared shader timeSec (water, particle lights, splats),\n"
                              "the TAA blend dt and the DLSS/FSR frame deltas. Set it once per frame\n"
                              "BEFORE render(), monotonically non-decreasing; stepping it by a fixed\n"
                              "dt makes ocean/grass/particle animation deterministic and frame-rate\n"
                              "independent, and the output replayable bit-for-bit. Unpinned, an\n"
                              "offline render whose frames take 80 ms of wall time animates the sea\n"
                              "~5x faster than the dt the app integrates its own physics with (the\n"
                              "hull can no longer follow the waves). None or a negative value\n"
                              "returns to the wall clock, which is what a live window wants.")
                // ── GPU event camera (DVS) ───────────────────────────────
                // A per-pixel log-intensity crossing detector (the ESIM model)
                // running as a compute pass on either the renderer's own
                // deterministic shade of the raster G-buffer or the final frame
                // (event_camera_source). Enabling it marks material samplers
                // dirty and, on the 'shaded' source, gates TAA jitter off (that
                // detector must not see jitter as scene motion), so flip it at
                // a CUT and give it a few frames to settle its per-pixel
                // reference — the first frame after enabling only latches the
                // reference and emits nothing.
                .def_property("event_camera_enabled",
                              [](PyVulkanRenderer& r) { return r.native().eventCameraEnabled(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setEventCameraEnabled(v); },
                              "GPU DVS event detector on/off. Costs nothing while off. Toggling\n"
                              "does a device-idle + image resize, so do it between shots, never\n"
                              "mid-sequence.")
                .def("set_event_camera_params",
                     [](PyVulkanRenderer& r, float threshold, float decay, float min_luma,
                        uint32_t max_events_per_pixel, uint32_t frame_time_us) {
                         VulkanRenderer::EventCameraParams p{};
                         p.threshold         = threshold;
                         p.decay             = decay;
                         p.minLuma           = min_luma;
                         p.maxEventsPerPixel = max_events_per_pixel;
                         p.frameTimeUs       = frame_time_us;
                         r.native().setEventCameraParams(p);
                     },
                     py::arg("threshold") = 0.15f, py::arg("decay") = 0.85f,
                     py::arg("min_luma") = 0.005f, py::arg("max_events_per_pixel") = 5u,
                     py::arg("frame_time_us") = 0u,
                     "Contrast threshold in log-intensity units (0.15 fires on almost any\n"
                     "edge, 0.30 only on hard ones), the visualisation's per-frame decay\n"
                     "toward mid-grey, the luma floor that stops log() exploding in the\n"
                     "shadows, the per-pixel event cap, and the microsecond clock for THIS\n"
                     "frame's sample.\n"
                     "\n"
                     "Drive `frame_time_us` EVERY frame from a monotone sim clock (a\n"
                     "wall-clock stamp is not reproducible). The detector remembers the\n"
                     "previous value and interpolates each event's timestamp linearly\n"
                     "between the two — the ESIM model — so a frame's events spread across\n"
                     "(previous, this] instead of sharing one stamp. Left at its default the\n"
                     "interval is empty and every event carries 0, an obviously-unstamped\n"
                     "stream rather than a plausible wrong one. The call only stores a\n"
                     "struct, so per-frame is cheap.")
                .def_property_readonly("event_camera_params",
                                       [](PyVulkanRenderer& r) {
                                           const auto p = r.native().eventCameraParams();
                                           py::dict d;
                                           d["threshold"]            = p.threshold;
                                           d["decay"]                = p.decay;
                                           d["min_luma"]             = p.minLuma;
                                           d["max_events_per_pixel"] = p.maxEventsPerPixel;
                                           d["frame_time_us"]        = p.frameTimeUs;
                                           return d;
                                       },
                                       "The detector's current parameters as a dict.")
                .def("set_event_camera_resolution",
                     [](PyVulkanRenderer& r, uint32_t width, uint32_t height) {
                         r.native().setEventCameraResolution(width, height);
                     },
                     py::arg("width"), py::arg("height"),
                     "Pin the sensor's native resolution (0, 0 tracks the swapchain).\n"
                     "Clamped to [16, swapchain]. A real DVS is coarse — 640x480 is\n"
                     "Prophesee Gen3/4 territory — and a coarser detector is also less\n"
                     "readback and less compute.")
                .def_property_readonly("event_camera_resolution",
                                       [](PyVulkanRenderer& r) { return r.native().eventCameraResolution(); },
                                       "(width, height) the detector is actually running at.")
                .def("read_event_camera_visualisation",
                     [](PyVulkanRenderer& r) {
                         const auto res = r.native().eventCameraResolution();
                         const auto w = static_cast<py::ssize_t>(res.first);
                         const auto h = static_cast<py::ssize_t>(res.second);
                         std::vector<unsigned char> rgba;
                         {
                             py::gil_scoped_release release;
                             rgba = r.native().readEventCameraVisualisation();
                         }
                         const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h) * 4u;
                         if (w <= 0 || h <= 0 || rgba.size() < need) {
                             return py::array_t<uint8_t>({py::ssize_t(0), py::ssize_t(0)});
                         }
                         py::array_t<uint8_t> arr({h, w});
                         auto* dst = arr.mutable_data();
                         // The accumulator is RGBA8 with the mono value replicated
                         // across RGB, and it is stored BOTTOM-UP (row 0 = bottom;
                         // the C++ demo feeds it straight to a sprite, whose V=0 is
                         // the bottom). Flip it so row 0 is the top of the image,
                         // matching read_pixels() and every AOV.
                         for (py::ssize_t y = 0; y < h; ++y) {
                             const unsigned char* src = rgba.data() +
                                     static_cast<size_t>(h - 1 - y) * static_cast<size_t>(w) * 4u;
                             for (py::ssize_t x = 0; x < w; ++x)
                                 dst[y * w + x] = src[x * 4];
                         }
                         return arr;
                     },
                     "The detector's accumulator image as (H, W) uint8 at the SENSOR\n"
                     "resolution, row 0 = top: 255 = positive (brightening) event,\n"
                     "0 = negative, 128 = no event, decaying back toward 128 at the\n"
                     "`decay` rate. Empty array while the event camera is off. Two\n"
                     "renderer frames of latency (it reads the oldest ring slot, which\n"
                     "the in-flight fences guarantee complete — no device wait).")
                .def("read_event_stream",
                     [](PyVulkanRenderer& r, size_t max_events) {
                         const auto res = r.native().eventCameraResolution();
                         const auto h   = static_cast<int64_t>(res.second);
                         std::vector<VulkanRenderer::Event> buf(max_events);
                         bool overflowed = false;
                         size_t n = 0;
                         {
                             py::gil_scoped_release release;
                             n = r.native().readEventStreamInto(buf.data(), buf.size(), &overflowed);
                         }
                         py::array_t<int64_t> arr({static_cast<py::ssize_t>(n), py::ssize_t(4)});
                         auto* dst = arr.mutable_data();
                         for (size_t i = 0; i < n; ++i) {
                             const auto& e = buf[i];
                             dst[i * 4 + 0] = static_cast<int64_t>(e.x);
                             // Same bottom-up -> top-down flip as the visualisation,
                             // so a stream coordinate indexes the image the sensor
                             // hands back.
                             dst[i * 4 + 1] = h > 0 ? h - 1 - static_cast<int64_t>(e.y)
                                                    : static_cast<int64_t>(e.y);
                             dst[i * 4 + 2] = static_cast<int64_t>(e.polarity);
                             dst[i * 4 + 3] = static_cast<int64_t>(e.t_us);
                         }
                         return py::make_tuple(arr, overflowed);
                     },
                     py::arg("max_events") = size_t(1000000),
                     "The sparse event stream of the last completed detector frame as\n"
                     "((N, 4) int64, overflowed): columns are x, y (image convention,\n"
                     "row 0 = top), polarity (+1 brightening / -1 darkening) and t_us.\n"
                     "`overflowed` is True when the GPU append list saturated and events\n"
                     "were dropped — the same failure mode a real sensor's readout has.\n"
                     "\n"
                     "t_us is sub-frame interpolated: log intensity is taken to ramp\n"
                     "linearly between two consecutive detector frames and each threshold\n"
                     "crossing is stamped where it crosses, so one frame's events span\n"
                     "(previous frame_time_us, this one] rather than sharing a value.\n"
                     "Rows come back sorted ascending by (t_us, y, x, polarity) — time\n"
                     "order like a real readout, and deterministic despite the GPU's\n"
                     "atomic append order being scheduler-dependent.")
                .def_property("events_only_mode",
                              [](PyVulkanRenderer& r) { return r.native().eventsOnlyMode(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setEventsOnlyMode(v); },
                              "Present the event visualisation INSTEAD of the shaded scene.\n"
                              "Leave it off if you also want read_pixels()/AOVs from the same\n"
                              "frames — the detector runs either way. Forces the 'shaded'\n"
                              "event_camera_source (no final frame exists to read).")
                .def_property("event_camera_source",
                              [](PyVulkanRenderer& r) -> std::string {
                                  switch (r.native().eventCameraSource()) {
                                      case VulkanRenderer::EventCameraSource::Final: return "final";
                                      default: return "shaded";
                                  }
                              },
                              [](PyVulkanRenderer& r, const std::string& v) {
                                  if (v == "shaded") r.native().setEventCameraSource(VulkanRenderer::EventCameraSource::Shaded);
                                  else if (v == "final") r.native().setEventCameraSource(VulkanRenderer::EventCameraSource::Final);
                                  else throw std::invalid_argument("event_camera_source: expected 'shaded' or 'final'");
                              },
                              "What the detector looks at. 'shaded' (default): a deterministic\n"
                              "Lambert proxy of the raster G-buffer — directional lights +\n"
                              "ambient + emissive, no specular / transmission / point lights /\n"
                              "GI. Noise-free and jitter-free (a static scene emits nothing),\n"
                              "but only silhouettes and diffuse texture fire: water glitter,\n"
                              "backlit sails and light flashes are invisible to it. 'final': the\n"
                              "presented frame — the same pixels read_pixels() returns, post\n"
                              "TAA / upscale / tonemap — box-averaged to the sensor resolution\n"
                              "(a DVS pixel integrates its photodiode area). Everything the\n"
                              "picture shows fires; it inherits the picture's temporal residue\n"
                              "(denoiser, auto-exposure drift) and TAA jitter stays on. A switch\n"
                              "while enabled re-latches the per-pixel reference on the next\n"
                              "frame (no burst). Ignored under events_only_mode.")
                // Per-frame CPU/GPU pass timings (milliseconds) — see
                // VulkanRenderer::FrameTimings. For perf triage from python.
                .def_property_readonly("frame_timings",
                                       [](PyVulkanRenderer& r) {
                                           const auto t = r.native().lastFrameTimings();
                                           py::dict d;
                                           d["shade_ms"] = t.pathTraceMs;
                                           d["denoise_ms"] = t.denoiseMs;
                                           d["taa_ms"] = t.taaMs;
                                           d["raster_gbuf_ms"] = t.rasterGbufMs;
                                           d["gbuf_resolve_ms"] = t.gbufResolveMs;
                                           d["shade_b_ms"] = t.shadeBMs;
                                           d["overlay_ms"] = t.overlayMs;
                                           d["dof_ms"] = t.dofMs;
                                           d["froxel_ms"] = t.froxelMs;
                                           d["instance_expand_ms"] = t.instanceExpandMs;
                                           // Staging (or interop-export) -> vertex/normal copies +
                                           // the batched BLAS refit, for every graduated CPU
                                           // deformer and every enable_vertex_interop mesh. This is
                                           // the column the interop path lands in.
                                           d["dyn_geom_refit_ms"] = t.dynGeomRefitMs;
                                           d["gpu_total_ms"] = t.gpuTotalMs;
                                           d["gpu_pass_sum_ms"] = t.gpuPassSumMs;
                                           d["cpu_ensure_scene_ms"] = t.cpuEnsureSceneMs;
                                           d["cpu_record_ms"] = t.cpuRecordMs;
                                           d["cpu_frame_ms"] = t.cpuFrameMs;
                                           return d;
                                       })
                // ── Zero-copy mesh-vertex interop (CUDA -> Vulkan) ────────────
                // Export a mesh's position/normal buffers so an external GPU
                // producer (Warp, PhysX, torch) writes them in place, with no
                // host round trip. Pair with threepp.cuda_interop.VkInteropArray,
                // which does the CUDA-side import and hands back wp.arrays.
                .def("enable_vertex_interop",
                     [](PyVulkanRenderer& r, const Mesh& mesh, std::function<void()> on_frame,
                        bool validate, bool stable_correspondence) -> py::object {
                         const auto h = r.native().enableVertexInterop(mesh, std::move(on_frame),
                                                                       validate,
                                                                       stable_correspondence);
                         // Null handle = "not ready, or not on this device" —
                         // never an exception. Same None-when-not-ready contract
                         // as GLRenderer.gl_buffer_id, and the caller polls the
                         // same way: render once, then enable.
                         if (!h.posHandle) return py::none();
                         return py::make_tuple(
                                 py::make_tuple(reinterpret_cast<uintptr_t>(h.posHandle),
                                                h.posBytes),
                                 py::make_tuple(reinterpret_cast<uintptr_t>(h.nrmHandle),
                                                h.nrmBytes));
                     },
                     py::arg("mesh"), py::arg("on_frame"), py::arg("validate") = true,
                     py::arg("stable_correspondence") = true,
                     "Export mesh.geometry's position + normal buffers for a foreign GPU "
                     "producer and arm the per-frame device write that fills them.\n\n"
                     "Returns ((pos_handle, pos_bytes), (nrm_handle, nrm_bytes)) or None.\n\n"
                     "POLL IT: the renderer's record for a mesh is created on the frame the "
                     "mesh is first drawn, so this returns None until after the first "
                     "render() — call render() once, then enable.\n\n"
                     "FIXED-CAPACITY ALLOCATION, VARIABLE DRAW: a producer whose triangle count "
                     "varies allocates its maximum once and publishes the live count with "
                     "set_draw_range — the raster path clamps to the range, the BLAS is built "
                     "over [0, start + count), and the interop copies are trimmed to match, so "
                     "no degenerate tail is needed. Changing an attribute's "
                     "count after enabling DISABLES interop for that mesh (with a warning on "
                     "stderr) rather than tearing down memory CUDA has imported.\n\n"
                     "on_frame() runs INSIDE render(), once per frame while the mesh is "
                     "visible, post-fence and pre-record, and MUST BE SYNCHRONOUS: every "
                     "kernel writing the exported buffers has to have completed when it "
                     "returns (wp.synchronize_device(device) as the last statement). That host "
                     "ordering is the only thing sequencing the foreign write against the "
                     "frame that reads it — there is no shared semaphore.\n\n"
                     "The handles are OS handles owned by the RENDERER (Win32 NT handles on "
                     "Windows): import them, but never CloseHandle them from Python — "
                     "disable_vertex_interop / renderer teardown releases them. The layout is "
                     "tightly-packed float xyz (12-byte stride, wp.vec3), and *_bytes is the "
                     "ALLOCATION size, which may exceed count*12 — write only the real range.\n\n"
                     "validate=True (default) runs a GPU finiteness pass over the exported "
                     "positions each frame, rewriting non-finite vertices as degenerates. Leave "
                     "it on unless the producer is trusted: a NaN reaching the BLAS build is a "
                     "device-lost (GPU reset) on NVIDIA, not an error return.\n\n"
                     "stable_correspondence=True (default) means vertex i is the SAME surface "
                     "point every frame (a deforming fixed-topology mesh — cloth, a soft body); "
                     "per-vertex motion vectors then come from the previous frame's positions. "
                     "Pass False for a producer that RE-TRIANGULATES each frame (a marching-"
                     "cubes soup: one changed cell shifts every later vertex slot) — that "
                     "history is noise there, so the mesh reprojects as world-static and the "
                     "temporal passes (TAA/upscaler, reflection denoiser) stop flickering on "
                     "the regions that changed.")
                .def("disable_vertex_interop",
                     [](PyVulkanRenderer& r, const Mesh& mesh) {
                         // Drains the device — the exports may still be a
                         // transfer source for an in-flight frame — so this is a
                         // teardown call, not a per-frame one, and it releases
                         // the GIL for the stall like every other stalling def
                         // here. Dropping the Python callback happens inside:
                         // pybind11's func_handle re-acquires the GIL to destroy
                         // it, which is exactly why this file uses the
                         // std::function caster.
                         py::gil_scoped_release release;
                         r.native().disableVertexInterop(mesh);
                     },
                     py::arg("mesh"),
                     "Release the exports and return the mesh to the CPU attribute path. STOP "
                     "the foreign writes first — nothing here can wait on a CUDA stream. Close "
                     "the importing VkInteropArrays before calling this.")
                // ── Zero-copy FRAMES OUT (Vulkan -> CUDA) ─────────────────────
                // The reverse direction: the renderer publishes what it drew
                // into external-memory buffers a torch policy reads as tensors.
                // Pair with threepp.torch_frames.FrameTensors, which does the
                // CUDA import and the tensor wrapping.
                .def("enable_frame_interop", &PyVulkanRenderer::enable_frame_interop,
                     py::arg("view") = 0u,
                     py::arg("channels") = py::make_tuple("color", "depth"),
                     "Export this view's per-frame images as CUDA-importable buffers and arm "
                     "the device-to-device copies that fill them.\n\n"
                     "channels: any of 'color', 'depth', 'normal', 'motion', 'ids', 'albedo', "
                     "'splat_depth' (or FrameChannel values). view=0 is the primary; anything "
                     "else must be a live add_view handle.\n\n"
                     "Returns a list of dicts, one per EXPORTABLE channel — duplicates collapse "
                     "and unexportable ones are skipped ('splat_depth' without splat_depth_aov), "
                     "so match on the 'channel' key. Each dict carries handle, size_bytes, "
                     "width, height, bytes_per_pixel and (for 'color') bgra. An EMPTY list means "
                     "nothing could be exported: before the first render(), on a stale view "
                     "handle, or on a device with no external-memory extension (one line on "
                     "stderr) — the fallback is the read_* host readback path.\n\n"
                     "CALL IT AFTER THE FIRST render(): the exports are sized from the "
                     "G-buffer / swapchain extents, which exist only once a frame has run.\n\n"
                     "SYNC: render() -> sync_frame_interop() -> read the tensors -> next "
                     "render(). Host ordering is the only cross-API synchronization; there is "
                     "no shared semaphore.\n\n"
                     "SINGLE-BUFFERED: the tensors are live views of renderer memory that the "
                     "next render() overwrites. Clone what you need to keep.\n\n"
                     "INVALIDATION: a resize, render_scale, gbuffer MSAA or splat_depth_aov "
                     "toggle, or removing the view DISABLES interop for that view (one warning "
                     "on stderr) rather than reallocating under a live import — re-enable and "
                     "re-import. The handles are OS handles owned by the RENDERER: import them, "
                     "never CloseHandle them from Python.")
                .def("disable_frame_interop", &PyVulkanRenderer::disable_frame_interop,
                     py::arg("view") = 0u,
                     "Release this view's frame-interop exports. Close the importing tensors "
                     "FIRST (FrameTensors.close() does both in the right order) — freeing the "
                     "Vulkan allocation under a live CUDA mapping reports as nothing at all.")
                .def("sync_frame_interop", &PyVulkanRenderer::sync_frame_interop,
                     "Block until the last submitted frame's interop copies have completed. "
                     "Waits ONE frame fence, not the whole device: a single queue signals "
                     "fences in submission order, so this retires every earlier frame too. "
                     "False before the first frame.")
                .def("frame_interop_active", &PyVulkanRenderer::frame_interop_active,
                     py::arg("view") = 0u,
                     "Is this view's frame interop still armed? False after an "
                     "invalidation (a resize, render_scale, ...) tore the exports down — "
                     "the one way to notice without reading stderr. FrameTensors.stale "
                     "is this, negated.")
                .def("read_gbuffer_aov_raw", &PyVulkanRenderer::read_gbuffer_aov_raw,
                     py::arg("aov"), py::arg("view") = 0u,
                     "The raw bytes of one G-buffer attachment of the last rendered frame as "
                     "(H, W, bytes_per_pixel) uint8, or None. No decode: this is the host "
                     "readback the zero-copy frame-interop path is checked byte-for-byte "
                     "against. Applications want read_depth / read_instance_ids / "
                     "read_aovs_typed instead.")
                // ── Zero-copy GPU-particle interop (CUDA -> Vulkan) ───────────
                // The ParticleField sibling of enable_vertex_interop: export an
                // Ownership.Interop field's ONE positions allocation so a Warp /
                // PhysX / torch kernel writes ParticlePos in place. 16-byte
                // slots, i.e. wp.vec4 with no repack — simpler than the vertex
                // path's 12-byte stride. Pair with
                // threepp.cuda_interop.VkInteropArray.
                .def("enable_particle_field_interop",
                     [](PyVulkanRenderer& r, ParticleField& field,
                        std::function<void()> device_copy) -> py::object {
                         const auto h = r.native().enableParticleFieldInterop(
                                 field, std::move(device_copy));
                         // Empty handle = "not yet, or this device cannot export"
                         // — never an exception, same None-when-not-ready
                         // contract as enable_vertex_interop. On the
                         // cannot-export leg the field is left in
                         // host_fallback() and submit() becomes legal on it.
                         if (!h.osHandle) return py::none();
                         return py::make_tuple(reinterpret_cast<uintptr_t>(h.osHandle),
                                               h.sizeBytes);
                     },
                     py::arg("field"), py::arg("device_copy"),
                     "Export an Ownership.Interop ParticleField's positions allocation and arm "
                     "the per-frame device-to-device copy that fills it.\n\n"
                     "Returns (os_handle, size_bytes), or None when the device has no "
                     "external-memory extension — in which case the field is left in "
                     "host_fallback() and submit() is legal on it, so the caller drops to the "
                     "HostRing path rather than failing.\n\n"
                     "CALL IT AFTER THE FIRST render(): the field's device state and this "
                     "renderer's field pass are both created on the frame the field is first "
                     "seen, so this returns None until then — render once, then enable.\n\n"
                     "device_copy() runs INSIDE render(), once per frame, pre-record, and MUST "
                     "BE SYNCHRONOUS (wp.copy(...) then wp.synchronize_device(device) as the "
                     "last statement, or cuMemcpyDtoDAsync + cuStreamSynchronize). That host "
                     "ordering is the only thing sequencing the foreign write against the frame "
                     "that reads it — there is no shared semaphore.\n\n"
                     "The handle is an OS handle owned by the RENDERER (a Win32 NT handle on "
                     "Windows): import it, but never CloseHandle it from Python. The layout is "
                     "ParticlePos — 16-byte xyzw slots, byte-identical to wp.vec4 and to "
                     "PxVec4 — and w < 0 is the DEAD sentinel every consumer tests. "
                     "size_bytes is the ALLOCATION size and may exceed capacity*16.\n\n"
                     "Liveness is the sim's job here: set_live_count(capacity) once and let the "
                     "kernel write w < 0 for dead slots. An Interop field forfeits "
                     "reproducibility and every emitter-derived feature (age fade, size taper, "
                     "colour ramp, surface landing) — it is positions, a radius and an "
                     "orientation set, and that is the whole model.")
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
                // ── Camera intrinsics readout ─────────────────────────────────
                .def_property_readonly("camera_intrinsics",
                                       [](PyVulkanRenderer& r) {
                                           const auto k = r.native().cameraIntrinsics();
                                           py::dict d;
                                           d["fx"] = k.fx;
                                           d["fy"] = k.fy;
                                           d["cx"] = k.cx;
                                           d["cy"] = k.cy;
                                           d["width"] = k.width;
                                           d["height"] = k.height;
                                           return d;
                                       },
                                       "Pinhole intrinsics in RENDER-extent pixels (top-left "
                                       "origin, OpenCV convention) as a dict fx/fy/cx/cy/width/"
                                       "height. Derived from the camera's own film gauge and "
                                       "focal length -- set a real camera with "
                                       "`cam.film_gauge = 6.3; cam.set_focal_length(4.8)`. "
                                       "Valid after the first render.")
                // ── Lens distortion ───────────────────────────────────────────
                .def("set_lens_distortion",
                     [](PyVulkanRenderer& r, const std::string& model, float k1, float k2,
                        float k3, float k4, float p1, float p2) {
                         LensDistortion d;
                         if (model == "none") {
                             d.model = LensModel::None;
                         } else if (model == "brown_conrady") {
                             d.model = LensModel::BrownConrady;
                         } else if (model == "fisheye") {
                             d.model = LensModel::Fisheye;
                         } else {
                             throw std::invalid_argument(
                                     "set_lens_distortion: model must be 'none', "
                                     "'brown_conrady' or 'fisheye'");
                         }
                         d.k1 = k1; d.k2 = k2; d.k3 = k3; d.k4 = k4;
                         d.p1 = p1; d.p2 = p2;
                         r.native().setLensDistortion(d);
                     },
                     py::arg("model") = "none", py::arg("k1") = 0.f, py::arg("k2") = 0.f,
                     py::arg("k3") = 0.f, py::arg("k4") = 0.f, py::arg("p1") = 0.f,
                     py::arg("p2") = 0.f,
                     "OpenCV-compatible lens distortion. 'brown_conrady' takes cv2's "
                     "(k1, k2, p1, p2, k3); 'fisheye' takes cv2.fisheye's (k1..k4). "
                     "Applied to BOTH the displayed image and the AOV readback, so "
                     "segmentation/depth labels stay aligned with the distorted pixels. "
                     "Default 'none' (pinhole, zero cost).")
                .def_property_readonly("lens_distortion",
                                       [](PyVulkanRenderer& r) {
                                           const auto d = r.native().lensDistortion();
                                           py::dict o;
                                           o["model"] = d.model == LensModel::BrownConrady
                                                                ? "brown_conrady"
                                                                : (d.model == LensModel::Fisheye ? "fisheye" : "none");
                                           o["k1"] = d.k1; o["k2"] = d.k2;
                                           o["k3"] = d.k3; o["k4"] = d.k4;
                                           o["p1"] = d.p1; o["p2"] = d.p2;
                                           return o;
                                       },
                                       "Current lens distortion as a dict (see set_lens_distortion).")
                .def_property("lens_overscan",
                              [](PyVulkanRenderer& r) { return r.native().lensOverscan(); },
                              [](PyVulkanRenderer& r, float v) { r.native().setLensOverscan(v); },
                              "Render the scene with the frustum widened by this factor so the "
                              "lens warp has real geometry for the output corners instead of a "
                              "clamped, smeared border. Barrel distortion (k1 < 0) needs it; "
                              "1.15-1.3 covers typical wide lenses. Costs effective resolution "
                              "(the same pixels cover a wider field). camera_intrinsics still "
                              "reports the OUTPUT camera. Default 1 (off).")
                // ── Image-sensor noise ────────────────────────────────────────
                .def("set_sensor_noise",
                     [](PyVulkanRenderer& r, bool enabled, float full_well, float read_noise,
                        float dark_current, float prnu_percent, uint32_t seed) {
                         VulkanRenderer::SensorNoise n;
                         n.enabled = enabled;
                         n.fullWellElectrons = full_well;
                         n.readNoiseElectrons = read_noise;
                         n.darkCurrentElectronsPerSec = dark_current;
                         n.prnuPercent = prnu_percent;
                         n.seed = seed;
                         r.native().setSensorNoise(n);
                     },
                     py::arg("enabled") = true, py::arg("full_well") = 20000.f,
                     py::arg("read_noise") = 3.f, py::arg("dark_current") = 5.f,
                     py::arg("prnu_percent") = 0.5f, py::arg("seed") = 1u,
                     "Shot/read/dark-current/PRNU sensor noise, in ELECTRONS, applied after "
                     "the temporal resolve (TAA would otherwise average it away). Noise scales "
                     "with the ISO from set_camera_exposure, as on a real sensor. "
                     "Deterministic: the same seed replays the same frames. Default off.")
                .def("reset_sensor_noise",
                     [](PyVulkanRenderer& r) { r.native().resetSensorNoise(); },
                     "Restart the noise sequence -- call on episode reset so two episodes with "
                     "the same seed produce the same frames.")
                .def_property_readonly("sensor_noise",
                                       [](PyVulkanRenderer& r) {
                                           const auto n = r.native().sensorNoise();
                                           py::dict o;
                                           o["enabled"] = n.enabled;
                                           o["full_well"] = n.fullWellElectrons;
                                           o["read_noise"] = n.readNoiseElectrons;
                                           o["dark_current"] = n.darkCurrentElectronsPerSec;
                                           o["prnu_percent"] = n.prnuPercent;
                                           o["seed"] = n.seed;
                                           return o;
                                       },
                                       "Current sensor-noise settings as a dict.")
                // ── Two-phase GPU occlusion culling ───────────────────────────
                .def_property("occlusion_culling",
                              [](PyVulkanRenderer& r) { return r.native().occlusionCulling(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setOcclusionCulling(v); },
                              "Two-phase GPU occlusion culling: hidden objects stop paying "
                              "raster cost (phase-2 same-frame recovery, no popping). Works "
                              "with gbuffer_msaa. Wins scale with occlusion (interiors, city "
                              "blocks). Default off.")
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
                         VulkanRenderer::ColorGrade g;
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
                // World-space irradiance probe grid (DDGI-lite): multi-bounce GI, and
                // the switch from cosmetic to MEASURED ambient — enclosed interiors
                // stop being "lit with no light" (ungated ambient/env-specular) and
                // receive only what actually bounces in through openings. Default ON;
                // converges over a few dozen frames after enable/scene load.
                // Automatic mesh LOD: background-simplified index/BLAS chains
                // selected per entry by projected screen-space error. Default ON.
                // Exposed so a script can A/B it — chain finalization does GPU
                // work in the frame it lands, so it is the first thing to rule
                // out when hunting periodic hitches.
                .def_property("auto_lod",
                              [](PyVulkanRenderer& r) { return r.native().autoLod(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setAutoLod(v); },
                              "Toggle automatic mesh LOD (background-simplified chains chosen "
                              "by projected screen-space error). Default ON; False pins every "
                              "mesh to full detail.")
                .def_property_readonly("auto_lod_stats",
                                       [](PyVulkanRenderer& r) {
                                           const auto s = r.native().autoLodStats();
                                           py::dict d;
                                           d["index_bytes"]   = s.indexBytes;
                                           d["blas_bytes"]    = s.blasBytes;
                                           d["chains_ready"]  = s.chainsReady;
                                           d["chains_queued"] = s.chainsQueued;
                                           py::list per;
                                           for (unsigned i = 0; i < 6; ++i) per.append(s.entriesPerLevel[i]);
                                           d["entries_per_level"] = per;
                                           return d;
                                       },
                                       "Auto-LOD counters: resident index/BLAS bytes, chains "
                                       "ready/queued, and the per-level entry histogram.")
                // ReSTIR DI master toggle. ON by default because it is what makes
                // many-light and emissive-geometry scenes converge at 1 spp, and it
                // is the path the renderer is tuned and golden-tested against. A
                // scene with a handful of analytic lights and no emitters is paying
                // for convergence it does not need, and the legacy per-light NEE
                // loops are cheaper there — so this is a perf lever for simple
                // scenes as much as it is an A/B for triaging reservoir artifacts.
                .def_property("restir_di",
                              [](PyVulkanRenderer& r) { return r.native().restirDIEnabled(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setRestirDIEnabled(v); },
                              "ReSTIR DI (streaming RIS + temporal/spatial reuse at primary\n"
                              "surfaces) for the deferred shade's next-event estimation.\n"
                              "Default on. Off falls back to the legacy per-light NEE loops:\n"
                              "cheaper with a handful of lights, markedly noisier with many.")
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
                // Volumetric fog: Henyey-Greenstein phase anisotropy of the ONE air
                // medium. Clamped to [-0.95, 0.95]. Takes effect whenever a medium is
                // present — scene.set_fog_exp2(...) OR set_height_fog (the whole froxel
                // volumetric path reads it, not just the homogeneous scene.fog case).
                // 0 = isotropic scattering, +0.9 = forward god-rays, -0.9 = back-scatter halo.
                .def_property("fog_anisotropy",
                              [](PyVulkanRenderer& r) { return r.native().getFogAnisotropy(); },
                              [](PyVulkanRenderer& r, float g) { r.native().setFogAnisotropy(g); })
                // DEPRECATED (Phase 2 fog unification) — a no-op. The directional
                // sun shafts + aerial glow are now ALWAYS on when the fog medium is
                // present: set scene.set_fog_exp2(...) (or set_height_fog) and the
                // volumetrics follow automatically — the froxels own the near field
                // [0, 512 m] and the per-pixel march the far tail. Kept only so
                // existing callers keep working (the setter does nothing).
                .def_property("volumetric_fog",
                              [](PyVulkanRenderer& r) { return r.native().volumetricFog(); },
                              [](PyVulkanRenderer& r, bool v) { r.native().setVolumetricFog(v); })
                // World-Y of the water surface — the clip plane the underwater
                // murk lives below, and (since the underwater shading pass) the
                // plane the AIR medium is clipped ABOVE. Also the gate every
                // submerged path is behind: unset (1e30, the default) means the
                // renderer has no waterline and nothing underwater-specific runs.
                .def("set_fog_water_surface_y",
                     [](PyVulkanRenderer& r, float y) { r.native().setFogWaterSurfaceY(y); },
                     py::arg("y"),
                     "World-Y of the water surface: murk applies below it, the air "
                     "medium above it. 1e30 (default) = no waterline.")
                // Underwater murk — a homogeneous absorption/tint medium clipped to
                // BELOW the water surface (fog_water_surface_y). Phase 2 decouples
                // this from scene.fog: scene.fog is the AIR medium (haze / god rays,
                // unclipped) and this is the SEPARATE below-water medium, so a scene
                // can hold clear air above the waterline and murk below. density =
                // sigma_t (1/m; 0 = off); color = inscatter tint.
                .def("set_underwater_murk",
                     [](PyVulkanRenderer& r, float density, const Color& color) {
                         r.native().setUnderwaterMurk(density, color);
                     },
                     py::arg("density"), py::arg("color") = Color(1.f, 1.f, 1.f),
                     "Enable underwater murk (below fog_water_surface_y). density = "
                     "sigma_t (1/m; 0 disables); color = inscatter tint.")
                // Volumetric clouds — a far-field raymarched, wind-driven cloud
                // deck in the world-space shell [bottom_y, top_y], composited
                // over the sky and (depth-aware) in front of terrain. Lit by the
                // scene's DirectionalLight sun. Call set_clouds(...) to enable,
                // disable_clouds() to turn off (default off; off is free).
                .def("set_clouds",
                     [](PyVulkanRenderer& r, float coverage, float density,
                        float bottom_y, float top_y, const Vector3& wind, float evolve_speed) {
                         VulkanRenderer::CloudSettings s;
                         s.coverage    = coverage;
                         s.density     = density;
                         s.bottomY     = bottom_y;
                         s.topY        = top_y;
                         s.wind        = wind;
                         s.evolveSpeed = evolve_speed;
                         r.native().setClouds(s);
                     },
                     py::arg("coverage") = 0.45f, py::arg("density") = 1.0f,
                     py::arg("bottom_y") = 600.0f, py::arg("top_y") = 1400.0f,
                     py::arg("wind") = Vector3(8.f, 0.f, 2.f), py::arg("evolve_speed") = 1.0f,
                     "Enable the volumetric cloud layer. coverage 0..1 (0=clear, "
                     "1=overcast); density multiplier; bottom_y/top_y world-Y shell; "
                     "wind m/s xz drift; evolve_speed shape churn rate.")
                .def("disable_clouds", [](PyVulkanRenderer& r) { r.native().setClouds(std::nullopt); },
                     "Turn the volumetric cloud layer off (default).")
                // Fog medium PROFILE control (advanced). Phase 2: there is ONE air
                // medium. scene.set_fog_exp2(...) is the primary knob (density +
                // colour); set_height_fog is the ADVANCED control of that medium's
                // exponential height PROFILE (baseY / falloff) × wind-scrolled
                // noise, in the 0.25-512 m view froxels. PRECEDENCE: the PROFILE
                // (baseY/falloff/noise) always applies when set. DENSITY — an
                // explicit density > 0 here is the deliberate OVERRIDE and WINS
                // over scene.fog; density <= 0 is "profile-only" (scene.fog, or
                // the panel slider, supplies the density and this only shapes it).
                // The underwater murk is a SEPARATE medium (set_underwater_murk).
                .def("set_height_fog",
                     [](PyVulkanRenderer& r, float density, float base_y,
                        float falloff, float noise_amount) {
                         VulkanRenderer::HeightFogSettings s;
                         s.density     = density;
                         s.baseY       = base_y;
                         s.falloff     = falloff;
                         s.noiseAmount = noise_amount;
                         r.native().setHeightFog(s);
                     },
                     py::arg("density") = 0.02f, py::arg("base_y") = 0.0f,
                     py::arg("falloff") = 80.0f, py::arg("noise_amount") = 0.6f,
                     "Enable near-field heterogeneous height fog. density = sigma_t "
                     "at base_y; base_y world-Y; falloff exponential height scale (m); "
                     "noise_amount 0=smooth..1=fully noise-modulated.")
                .def("disable_height_fog", [](PyVulkanRenderer& r) { r.native().setHeightFog(std::nullopt); },
                     "Turn near-field height fog off (default).")
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
                     "EV clamp for auto-exposure relative to linear 1.0 (default -3 to +3).")
                // Raw beam-table lidar: ONE dispatch, ONE readback, any beam
                // set. This is what makes multi-pose scoring GPU-bound instead
                // of round-trip-bound: 12 poses as 12 PathTracedLidarSensor
                // scans is 12 tiny dispatches each followed by a synchronous
                // readback (~30% GPU); the same 48k beams as one table is one
                // trace. The sensor class remains the ergonomic path when the
                // beams follow a pose.
                .def("scan_lidar", [](PyVulkanRenderer& r,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> origins,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> directions,
                                      const LidarParams& params) {
                    if (origins.ndim() != 2 || origins.shape(1) != 3 ||
                        directions.ndim() != 2 || directions.shape(1) != 3 ||
                        origins.shape(0) != directions.shape(0)) {
                        throw std::invalid_argument("origins and directions must both be (N, 3)");
                    }
                    const auto n = static_cast<size_t>(origins.shape(0));
                    std::vector<LidarBeam> beams(n);
                    const float* po = origins.data();
                    const float* pd = directions.data();
                    for (size_t i = 0; i < n; ++i) {
                        beams[i].origin.set(po[i * 3 + 0], po[i * 3 + 1], po[i * 3 + 2]);
                        beams[i].direction.set(pd[i * 3 + 0], pd[i * 3 + 1], pd[i * 3 + 2]);
                    }
                    std::vector<LidarReturn> out;
                    std::vector<LidarReturn> clean;
                    {
                        py::gil_scoped_release release;
                        r.native().scanLidar(beams, out, params,
                                             params.pairedCleanTrace ? &clean : nullptr);
                    }
                    const auto n_ = static_cast<py::ssize_t>(out.size());
                    py::array_t<float>   pos({n_, static_cast<py::ssize_t>(3)});
                    py::array_t<float>   nrm({n_, static_cast<py::ssize_t>(3)});
                    py::array_t<float>   dist(n_);
                    py::array_t<float>   inten(n_);
                    py::array_t<int32_t> inst(n_);
                    py::array_t<int32_t> rno(n_);
                    py::array_t<int32_t> rkind(n_);
                    auto *pp = pos.mutable_data(), *pn = nrm.mutable_data();
                    auto *pdi = dist.mutable_data(), *pi = inten.mutable_data();
                    auto *pinst = inst.mutable_data(), *prno = rno.mutable_data(),
                         *pkind = rkind.mutable_data();
                    for (py::ssize_t i = 0; i < n_; ++i) {
                        const auto& q = out[static_cast<size_t>(i)];
                        pp[i * 3 + 0] = q.position.x; pp[i * 3 + 1] = q.position.y; pp[i * 3 + 2] = q.position.z;
                        pn[i * 3 + 0] = q.normal.x;   pn[i * 3 + 1] = q.normal.y;   pn[i * 3 + 2] = q.normal.z;
                        pdi[i] = q.distance; pi[i] = q.intensity;
                        pinst[i] = q.hitInstanceId; prno[i] = q.returnNo; pkind[i] = q.returnKind;
                    }
                    py::dict d;
                    d["position"] = pos; d["normal"] = nrm; d["distance"] = dist;
                    d["intensity"] = inten; d["instance_id"] = inst;
                    d["return_no"] = rno; d["return_kind"] = rkind;
                    return d;
                }, py::arg("origins"), py::arg("directions"), py::arg("params") = LidarParams{},
                   "Trace an arbitrary beam table in ONE dispatch: origins (N,3) + unit "
                   "directions (N,3) -> the same dict of numpy arrays as "
                   "PathTracedLidarSensor.scan(), row i belonging to beam i (x "
                   "samples_per_beam x max_returns when those are raised; return_no > 0 is "
                   "the real-return predicate). render() the scene once first. Use this "
                   "when the beams do not follow a single pose - e.g. scoring an object "
                   "from a ring of viewpoints in one round trip.");

        // ---- PathTracedLidarSensor (helpers/PathTracedLidarSensor.hpp) -------
        // The renderer's own lidar tracer (VulkanRenderer::scanLidar), surfaced
        // at last: LidarReturn/LidarParams/LidarModel have been bound as value
        // types since the start, but nothing on the Python side could fire
        // them — DepthSensor was the only scanner, and it cannot set fog,
        // detector threshold, laser power or reference range. Returns come
        // back as a dict of numpy arrays rather than a list of LidarReturn
        // objects: a 100k-beam scan as 100k Python objects is unusable.
        auto returns_to_dict = [](const std::vector<LidarReturn>& rs) {
            const auto n = static_cast<py::ssize_t>(rs.size());
            py::array_t<float>   pos({n, static_cast<py::ssize_t>(3)});
            py::array_t<float>   nrm({n, static_cast<py::ssize_t>(3)});
            py::array_t<float>   dist(n);
            py::array_t<float>   inten(n);
            py::array_t<int32_t> inst(n);
            py::array_t<int32_t> rno(n);
            py::array_t<int32_t> rkind(n);
            auto *pp = pos.mutable_data(), *pn = nrm.mutable_data();
            auto *pd = dist.mutable_data(), *pi = inten.mutable_data();
            auto *pinst = inst.mutable_data(), *prno = rno.mutable_data(),
                 *pkind = rkind.mutable_data();
            for (py::ssize_t i = 0; i < n; ++i) {
                const auto& r = rs[static_cast<size_t>(i)];
                pp[i * 3 + 0] = r.position.x; pp[i * 3 + 1] = r.position.y; pp[i * 3 + 2] = r.position.z;
                pn[i * 3 + 0] = r.normal.x;   pn[i * 3 + 1] = r.normal.y;   pn[i * 3 + 2] = r.normal.z;
                pd[i] = r.distance; pi[i] = r.intensity;
                pinst[i] = r.hitInstanceId; prno[i] = r.returnNo; pkind[i] = r.returnKind;
            }
            py::dict d;
            d["position"] = pos; d["normal"] = nrm; d["distance"] = dist;
            d["intensity"] = inten; d["instance_id"] = inst;
            d["return_no"] = rno; d["return_kind"] = rkind;
            return d;
        };
        py::class_<PathTracedLidarSensor, Object3D, Sensor,
                   std::shared_ptr<PathTracedLidarSensor>>(
                m, "PathTracedLidarSensor",
                "Path-traced LIDAR: fires beams through the renderer's own acceleration "
                "structure and returns full radiometric hits (position, normal, distance, "
                "intensity from the GPU back-scatter BRDF, stable instance id, return "
                "number/kind). Vulkan only; render() the scene at least once first so the "
                "TLAS exists.\n\n"
                "`params` (a threepp.LidarParams, mutated in place) exposes the whole LIDAR "
                "equation: max/min range, laser power, reference range, detector threshold, "
                "atmospheric extinction, multi-return through transmissive surfaces, beam "
                "divergence sampling, a dedicated water-column/dust medium, and the paired "
                "clean/degraded trace (see LidarParams).\n\n"
                "Beam convention matches DepthSensor: beams leave along the sensor's LOCAL "
                "-Z. The sensor is an Object3D, not a Camera, so look_at() aims it exactly "
                "backwards -- set rotation/quaternion directly, or reflect the target "
                "through the sensor position.")
                .def(py::init([](unsigned int h_res, unsigned int v_res, float max_range) {
                    return std::make_shared<PathTracedLidarSensor>(h_res, v_res, max_range);
                }), py::arg("h_res"), py::arg("v_res"), py::arg("max_range") = 100.f,
                    "Dense grid: h_res x v_res beams over the full sphere (debug / ground truth).")
                .def(py::init([](const LidarModel& model, float max_range) {
                    return std::make_shared<PathTracedLidarSensor>(model, max_range);
                }), py::arg("model"), py::arg("max_range") = 100.f,
                    "Real-sensor beam pattern, e.g. LidarModel.vlp16() / os1_64().")
                .def(py::init([](float fov_y, unsigned int width, unsigned int height, float max_range) {
                    return std::make_shared<PathTracedLidarSensor>(fov_y, width, height, max_range);
                }), py::arg("fov_y"), py::arg("width"), py::arg("height"), py::arg("max_range") = 100.f,
                    "Depth-camera mode: a pinhole grid down local -Z, same mounting as DepthSensor.")
                .def_readwrite("params", &PathTracedLidarSensor::params,
                               "LidarParams, live-tweakable between scans; mutate in place.")
                .def_readwrite("noise", &PathTracedLidarSensor::rangeNoise,
                               "Seeded RangeNoiseModel applied along each beam (default zero model: "
                               "the tracer's own range is already physical). Same replay contract as "
                               "DepthSensor.noise.")
                .def_property_readonly("beam_count", &PathTracedLidarSensor::beamCount)
                .def("scan", [returns_to_dict](PathTracedLidarSensor& self, PyVulkanRenderer& r) {
                    self.updateWorldMatrix(true, true);
                    std::vector<LidarReturn> out;
                    {
                        py::gil_scoped_release release;// blocks on the GPU trace + readback
                        self.scan(r.native(), out);
                    }
                    py::dict d = returns_to_dict(out);
                    // The clean leg of a paired trace rides along when requested —
                    // same beams, same RNG keys, particle medium off (see
                    // LidarParams.pairedCleanTrace).
                    if (!self.cleanReturns().empty()) d["clean"] = returns_to_dict(self.cleanReturns());
                    return d;
                }, py::arg("renderer"),
                   "One scan from the current pose -> dict of numpy arrays keyed position "
                   "(N,3), normal (N,3), distance, intensity, instance_id, return_no, "
                   "return_kind (all length N = beams x samples_per_beam x max_returns; "
                   "return_no > 0 is the 'real return' predicate). Adds key 'clean' when "
                   "params.paired_clean_trace is set. Call after render(); never during it.");

        m.attr("HAS_VULKAN") = true;
        // HAS_VULKAN says the backend was COMPILED IN; this says the machine can
        // actually load it. Distinct since the wheel began shipping Vulkan with a
        // delay-loaded loader: HAS_VULKAN is True everywhere the wheel installs,
        // vulkan_available() is False where no Vulkan runtime exists. (Driver /
        // GPU problems still surface as RuntimeError at construction — this only
        // answers whether the loader is present.)
        m.def("vulkan_available", [] { return vulkan_loader_present(); },
              "True when the Vulkan loader is present at runtime. HAS_VULKAN=True + "
              "vulkan_available()=False means the wheel carries the backend but this "
              "machine has no Vulkan runtime — use GLRenderer.");
        // The validation-layer tally behind the C++ CI gate
        // (threepp/renderers/vulkan/ValidationReport.hpp), so a pytest run
        // under THREEPP_VULKAN_VALIDATION=1 can assert a frame sequence
        // produced no spec violations. Counts are process-wide and never
        // reset; a per-phase delta is two readings subtracted.
        m.def("vulkan_validation_error_count",
              [] { return threepp::vulkan::validationErrorCount(); },
              "Validation-layer ERROR messages counted since process start. Always 0 "
              "unless the layer is active — check vulkan_validation_active() first, or "
              "a 'no errors' assertion passes vacuously.");
        m.def("vulkan_validation_active",
              [] { return threepp::vulkan::validationActive(); },
              "True once a renderer has actually installed the validation-layer "
              "messenger (layer requested via THREEPP_VULKAN_VALIDATION=1 or a debug "
              "build, AND found on the machine).");
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
        m.def("vulkan_available", [] { return false; },
              "Always False in a GL-only build (HAS_VULKAN is False too).");
        m.def("vulkan_validation_error_count", [] { return 0u; },
              "Always 0 in a GL-only build.");
        m.def("vulkan_validation_active", [] { return false; },
              "Always False in a GL-only build.");
    }

    threepp::Renderer* py_vulkan_native_renderer(const py::handle&) { return nullptr; }

}// namespace threepp_py

#endif
