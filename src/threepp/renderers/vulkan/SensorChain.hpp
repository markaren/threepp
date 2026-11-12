// SensorChain — a camera's image chain on a secondary view, on the GPU.
//
// A sensor view (VulkanRenderer::setViewSensorMode) gives a clean picture of
// one instant. What a camera makes of it — its lens, its exposure over time,
// its pixels' noise, or an event camera's asynchronous pixels — is per pixel,
// and doing it on the host costs a readback of every render and a CPU pass
// over every pixel. This runs it after the view's passes, in the same sensor
// frame (VulkanRenderer::setViewSensorChain):
//
//  Rgb    — accumulate the view's linear picture through the lens remap (one
//           per sub-pose of the exposure: motion blur), then finish: the mean
//           through a CMOS sensor (shot and read noise, exposure, gain, sRGB),
//           and the frame's linear sum for the host's auto-exposure.
//  Events — every render through the lens remap into events: ESIM with a
//           per-pixel threshold, photoreceptor bandwidth, readout jitter, a
//           refractory period, background activity and hot pixels.
//
// The host queues an operation before renderSensorViews and reads the result
// after it (the frame is waited on). Per-pixel state stays on the device.

#ifndef THREEPP_VULKAN_SENSOR_CHAIN_HPP
#define THREEPP_VULKAN_SENSOR_CHAIN_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <functional>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class SensorChain {

    public:
        enum class Kind { Rgb, Events };

        struct Setup {
            Kind     kind = Kind::Rgb;
            uint32_t baseW = 0, baseH = 0;// the camera's render without supersampling
            uint32_t ss = 1;              // the view is baseW*ss x baseH*ss
            uint32_t outW = 0, outH = 0;  // the sensor's image
            // per output pixel: the base pixel at the bilinear footprint's
            // top-left (-1: outside the render) and the footprint's fractions
            std::vector<int32_t> tapIndex;
            std::vector<float>   tapFx, tapFy;
            // Events: per output pixel, the contrast threshold and the hot rate (Hz, 0 = not hot)
            std::vector<float> threshold, hotRate;
        };

        struct RgbOp {
            bool     accumulate = false;// add this frame's picture
            bool     clear = false;     // ... into a cleared accumulator
            float    invGain = 1.f;
            bool     finish = false;    // then the sensor
            float    nInv = 1.f, k = 1.f, g = 1.f, fullWell = 1e4f, readNoise = 0.f;
            uint32_t seed = 0;
            bool     exportLinear = false;// also copy the linear mean to the host
        };

        struct EventOp {
            bool     first = false;// latch the references, emit nothing
            float    dt = 0.f;     // since the last render (s)
            uint32_t seed = 0;
            float    minLuma = 0.005f;
            uint32_t maxEvents = 5;
            float    refractory = 0.f;
            float    baHz = 0.f, baDark = 0.f;
            bool     background = false;
            bool     bandwidth = false;
            float    tauDay = 1e-4f, tauMax = 5e-3f, illumRef = 0.18f, illumGain = 1.f, jitter = 0.f;
            float    vizDecay = 0.8f;
            bool     writeViz = false;
        };

        // 16-byte event record, as sensor_evs.comp writes it
        struct Event {
            uint32_t xy;// x | y << 16
            float    t; // s after the render before
            int32_t  polarity;
            uint32_t signal;// bit 0: a signal event; bits 1..: the frame's operation it came from
        };

        SensorChain(VulkanContext& ctx, const Setup& setup);
        ~SensorChain();
        SensorChain(const SensorChain&)            = delete;
        SensorChain& operator=(const SensorChain&) = delete;

        [[nodiscard]] Kind kind() const { return kind_; }
        // Operations run in the order queued, each on the picture of the view
        // `srcView` names (0: the chain's own). Several in one frame are the
        // renders of several views at successive instants (a blur's sub-poses,
        // an event camera's sub-renders): one sensor frame for all of them.
        static constexpr uint32_t kMaxOps = 32;
        bool queue(const RgbOp& op, uint32_t srcView = 0);
        bool queue(const EventOp& op, uint32_t srcView = 0);
        [[nodiscard]] bool pending() const { return !ops_.empty(); }
        struct Src {
            VkImage     image;
            VkImageView view;
        };

        // Record the queued operations (`src(handle)` gives a view's colour
        // target, GENERAL, written by its resolve; handle 0 = the chain's own
        // view). The first pass also uploads the taps and the initial state.
        void addPass(rg::RenderGraph& g, uint32_t frame, const std::function<Src(uint32_t)>& src);

        // After the frame: Rgb finish — RGB8 rows top-down, the linear sum
        // (all channels) and, if asked, the linear mean (RGB floats).
        bool readRgb(std::vector<unsigned char>& rgb8, double& linearSum, std::vector<float>* linear);
        // After the frame: Events — the records (unsorted), whether the
        // buffer overflowed, the pane (RGBA8, top-down) when it was written,
        // and the photoreceptor time-constant histogram (40 bins, 10 a decade
        // from 10 us; cumulative).
        void readEvents(std::vector<Event>& out, bool& overflow);
        bool readViz(std::vector<unsigned char>& rgba8);
        void readTauHistogram(std::array<uint32_t, 40>& hist);

    private:
        VulkanContext& ctx_;
        Kind     kind_;
        uint32_t baseW_, baseH_, ss_, outW_, outH_;
        bool     uploaded_ = false, vizWritten_ = false;
        bool     consumed_ = true;  // the event list was read: the next frame starts a new one
        uint32_t opBase_   = 0;     // operations since the last read (their events' `op` counts from it)
        struct Op {
            bool     rgb;
            RgbOp    r;
            EventOp  e;
            uint32_t src;
        };
        std::vector<Op> ops_;

        VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout      layout_   = VK_NULL_HANDLE;
        VkPipeline            pipeline_ = VK_NULL_HANDLE;
        VkDescriptorPool      pool_     = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2 * kMaxOps> sets_{};

        Buffer taps_{}, staging_{};   // staging: the taps and the initial state, once
        Buffer acc_{}, out_{}, sums_{}, linHost_{};// Rgb
        Buffer state_{}, events_{}, tau_{}, viz_{}, counter_{};// Events
        VkDeviceSize tapsBytes_ = 0, stateBytes_ = 0;
        uint32_t eventCapacity_ = 0, groupsX_ = 0, groupsY_ = 0;

        void createPipeline();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SENSOR_CHAIN_HPP
