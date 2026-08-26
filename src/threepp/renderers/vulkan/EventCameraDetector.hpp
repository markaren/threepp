// GPU-side event camera (DVS) detector. Mirrors the math of the
// renderer-agnostic helpers/EventCameraSensor.{hpp,cpp} but runs as a
// Vulkan compute pass against the renderer's existing scene-capture
// buffer, eliminating the full-frame CPU readback and per-pixel CPU
// loop that capped the helper at ~30 FPS.
//
// Pipeline:
//   1. Renderer's existing recordSceneCapture() copies the post-TAA
//      swapchain into sceneCaptureBuf_ (host-visible storage buffer).
//   2. EventCameraDetector::record() dispatches event_detect.comp,
//      which reads sceneCaptureBuf_, updates a persistent rg32f
//      log-history image (.x = emit reference, .y = last raw sample),
//      and paints an RGBA8 accumulator image.
//   3. The accumulator is copied to a 3-slot host-visible ring; the
//      host reads from the OLDEST slot (guaranteed complete) without
//      any vkDeviceWaitIdle. ~2 frames of display latency, no stall.
//
// Timestamps follow ESIM: log intensity is taken to ramp linearly
// between two consecutive samples, and each threshold crossing is
// stamped at its interpolated crossing time. A frame's events therefore
// land spread across (prevFrameTimeUs, frameTimeUs] rather than all on
// one stamp, and readEventStreamInto hands them back time-sorted. The
// detector tracks the previous stamp itself — the caller only ever
// supplies "now" via Params::frameTimeUs, but it must supply it EVERY
// frame from a monotone sim clock for the interpolation to mean
// anything. Note this raises timestamp RESOLUTION, not temporal
// bandwidth: the underlying sampling rate is still the frame rate, so
// motion faster than a frame is still missed.

#ifndef THREEPP_VULKAN_EVENT_CAMERA_DETECTOR_HPP
#define THREEPP_VULKAN_EVENT_CAMERA_DETECTOR_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class EventCameraDetector {

    public:
        struct Params {
            float    threshold        = 0.15f;
            float    decay            = 0.85f;
            float    minLuma          = 0.005f;
            uint32_t maxEventsPerPixel = 5;
            // Microsecond timestamp of THIS frame's sample. Caller-supplied
            // so it can carry a sim or wall clock; drive it every frame,
            // monotonically. The detector remembers the previous value and
            // interpolates each event's stamp along the log-intensity ramp
            // between the two, so a frame's events land spread across
            // (previous, this]. Left undriven (always 0) the interval is
            // empty and every event carries the same stamp, exactly as
            // before interpolation existed.
            uint32_t frameTimeUs       = 0u;
        };

        // 16-byte event record. Layout matches event_detect.comp's EvtRec
        // byte-for-byte; do NOT reorder or add padding without also
        // updating the shader.
        struct Event {
            uint32_t x;
            uint32_t y;
            int32_t  polarity;  // +1 or -1
            uint32_t t_us;
        };

        // Capacity = max events the stream buffer can hold per frame.
        // Beyond this the shader sets the overflow flag and drops further
        // events for that frame. Default sized for a busy 640×480 scene
        // (real DVS payloads top out around 50–100k events/frame in
        // typical motion).
        static constexpr uint32_t kEventStreamCapacity = 256u * 1024u;

        explicit EventCameraDetector(VulkanContext& ctx);
        ~EventCameraDetector();

        EventCameraDetector(const EventCameraDetector&)            = delete;
        EventCameraDetector& operator=(const EventCameraDetector&) = delete;

        // Re-allocate the per-pixel images + readback ring at the given
        // dimensions. Resets the log-history (next frame is treated as
        // first frame, no spurious burst of events).
        void resize(uint32_t width, uint32_t height);

        // Re-latch the per-pixel log-intensity reference on the next
        // record(): that frame stores the reference and emits nothing, like
        // the first frame after resize(). No GPU work, no reallocation —
        // for source switches (proxy ↔ final frame) whose luma jump is not
        // scene motion and must not read as a whole-frame burst.
        void resetReference() { firstFrame_ = true; }

        // Record the compute dispatch (sceneCaptureBuf → logHistory +
        // accumulator) plus the accumulator-to-ring copy. The caller is
        // responsible for ensuring sceneCaptureBuf was populated earlier
        // in the same command buffer (the renderer's scene-capture path
        // handles that) and that the appropriate barriers are in place.
        void record(VkCommandBuffer cb, VkBuffer sceneBuf, const Params& params);

        // Read the most recently completed visualisation. Returns RGBA8
        // bytes of size width × height × 4. Empty if not yet initialised.
        // No GPU wait — uses the oldest ring slot.
        [[nodiscard]] std::vector<unsigned char> readVisualisation() const;

        // Zero-allocation variant: write RGBA8 bytes straight into a
        // caller-provided buffer. Returns the number of bytes written
        // (always width × height × 4 on success, 0 on failure / not
        // yet initialised). The caller's buffer must have capacity ≥
        // width × height × 4. Used by the events-only fast path to
        // avoid the per-frame std::vector allocation + the redundant
        // memcpy that the std::vector return path forces (mapped → vec,
        // then vec → DataTexture inside the caller); this version goes
        // mapped → DataTexture directly.
        size_t readVisualisationInto(unsigned char* dst, size_t cap) const;

        // Sparse event stream from the OLDEST ring slot — the same
        // 2-frame-latency guarantee that visualisation readback uses,
        // so no GPU wait. Returns the event count actually written to
        // `dst` (clamped to `cap`); `overflowed` (if non-null) is set to
        // true when the shader dropped events because the GPU stream
        // buffer hit its capacity.
        //
        // Events come back SORTED ascending by (t_us, y, x, polarity).
        // Their stamps are the sub-frame interpolated crossing times
        // within (prevFrameTimeUs, frameTimeUs] of the frame that
        // produced them, so the packet is monotone in time the way a
        // real sensor's readout is; the tie-break on the remaining
        // fields makes it deterministic despite the GPU's atomicAdd
        // append order being scheduler-dependent.
        size_t readEventStreamInto(Event* dst, size_t cap, bool* overflowed) const;

        [[nodiscard]] uint32_t width() const { return width_; }
        [[nodiscard]] uint32_t height() const { return height_; }

    private:
        VulkanContext& ctx_;

        uint32_t width_  = 0;
        uint32_t height_ = 0;
        bool     firstFrame_ = true;
        // Stamp handed to the previous record(); the low end of the
        // interval this frame's events interpolate into.
        uint32_t lastFrameTimeUs_ = 0;

        Image2D  logHistoryImg_{};
        Image2D  accumulatorImg_{};

        static constexpr uint32_t kRingSize = 3;
        std::array<Buffer, kRingSize> readbackRing_{};
        // Per-ring-slot event stream buffer. 16B header + capacity·sizeof(Event)
        // bytes of payload. Host-visible so readEventStreamInto can read directly.
        std::array<Buffer, kRingSize> eventStreamRing_{};
        // Slot we'll WRITE next on record(); we then advance.
        uint32_t writeSlot_ = 0;

        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;
        // One descriptor set per ring slot. record() binds the write-slot's
        // set; each set's event-stream binding (3) points permanently at that
        // slot's eventStreamRing_ buffer, so no per-frame descriptor update is
        // needed. A single shared set rewritten every frame would be modified
        // while the previous frame's command buffer (still in flight) had it
        // bound — VUID-vkUpdateDescriptorSets-None-03047, a source of the
        // event-camera flicker. kRingSize (3) > kFramesInFlight (2), so a
        // slot's set is never touched while its own submission is pending.
        std::array<VkDescriptorSet, kRingSize> descSets_{};
        // Tracks which scene buffer the descriptors currently point at;
        // rewrite only when the buffer handle changes (resize, etc.).
        VkBuffer              currentSceneBuf_ = VK_NULL_HANDLE;

        void createPipeline();
        void allocateDescriptorPool();
        void destroyImages();
        void updateSceneBinding(VkBuffer sceneBuf);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_EVENT_CAMERA_DETECTOR_HPP
