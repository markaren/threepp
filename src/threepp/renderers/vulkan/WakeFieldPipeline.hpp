// WakeFieldPipeline — the ocean's wake field (DisplacedMesh::WakeField): what
// propellers, jets, paddles and hulls leave on the water, simulated in
// world-anchored patches that do not repeat. Two compute pipelines shared by
// every ocean: wake_field.comp (the foam, the bubbles, the turbulence, the lane
// and the water's flow) and wake_ripple.comp (the waves the producers make, a
// spectrum a patch, through the ocean's own inverse FFT). Each ocean with a
// wake field owns a State: its patch images, its descriptor sets and the
// device buffer that holds the patch table and the step's sources.
//
// Where it meets the rest of the renderer:
//   * State::stateView() is what the deferred water shade samples (binding 78)
//     and State::rippleView() the ripples' slopes (binding 79); dummyView()
//     stands in for either when no ocean has one.
//   * State::tableAddress() is stable for the State's life and goes into the
//     ocean's GeometryDesc::foamAddress, which is how the shade finds the
//     patches. The table is rewritten inside the frame's command stream
//     (vkCmdUpdateBuffer), never from the host, so the frame still in flight
//     reads the origins its own images were written with.

#ifndef THREEPP_VULKAN_WAKE_FIELD_PIPELINE_HPP
#define THREEPP_VULKAN_WAKE_FIELD_PIPELINE_HPP

#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>

namespace threepp::vulkan {

    class VulkanContext;

    // Mirrors WakePatch in ocean_wake.glsl (scalar layout, 56 bytes).
    struct WakePatchGpu {
        float originX;
        float originZ;
        float size;
        float texel;
        float prevOriginX;
        float prevOriginZ;
        float live;
        float eddy;
        float ripple;
        float rippleDecay;
        float rippleOriginX;
        float rippleOriginZ;
        float rippleSize;
        float _pad;
    };
    static_assert(sizeof(WakePatchGpu) == 56, "WakePatchGpu must match ocean_wake.glsl's WakePatch");

    // Mirrors WakeTable in ocean_wake.glsl up to (not including) its sources.
    struct WakeTableGpu {
        uint32_t patchCount;
        uint32_t res;
        float    timeSec;
        float    dt;
        float    foamLife;
        float    foamThinning;
        float    aerationLife;
        float    turbulenceLife;
        float    laneLife;
        float    velocityLife;
        float    eddyTexels;
        float    spread;
        uint32_t rippleRes;
        float    rippleViscosity;
        float    rippleGain;
        float    albedo;
        WakePatchGpu patches[DisplacedMesh::kMaxWakePatches];
    };
    static_assert(sizeof(WakeTableGpu) == 64 + 56 * DisplacedMesh::kMaxWakePatches,
                  "WakeTableGpu must match ocean_wake.glsl's WakeTable");
    static_assert(sizeof(DisplacedMesh::WakeSource) == 64,
                  "DisplacedMesh::WakeSource is memcpy'd after the table; ocean_wake.glsl reads 64-byte sources");

    class WakeFieldPipeline {

    public:
        static constexpr uint32_t kMaxOceans = 16;

        // Must match the `Pc` of wake_field.comp and wake_ripple.comp (24 bytes).
        struct PushConstants {
            VkDeviceAddress tableAddr;
            uint32_t        layer;
            uint32_t        mode;
            uint32_t        sourceCount;
            uint32_t        _pad;
        };

        // One ocean's wake field.
        struct State {
            uint32_t res    = 0;
            uint32_t layers = 0;
            // 0 stateA: what the shade samples, and what a step starts from
            // 1 stateB: carried forward (step 0)     2 stateC: corrected (step 1)
            // 3 velA                                  4 velB
            std::array<VkImage, 5>       image{};
            std::array<VmaAllocation, 5> alloc{};
            std::array<VkImageView, 5>   view{};
            std::array<VkDescriptorSet, 3> ds{};
            Buffer table{};// WakeTableGpu + kMaxWakeSources sources, device-local
            double prevTimeSec = -1.0;
            std::array<float, DisplacedMesh::kMaxWakePatches> prevOriginX{};
            std::array<float, DisplacedMesh::kMaxWakePatches> prevOriginZ{};
            std::array<float, DisplacedMesh::kMaxWakePatches> prevSize{};// 0 = the layer holds nothing

            // The ripples (rippleRes 0 = this field has none).
            uint32_t rippleRes  = 0;
            uint32_t rippleMips = 1;
            VkImage       ampImage    = VK_NULL_HANDLE;// rgba32f array: a patch's spectrum, and its still dent
            VmaAllocation ampAlloc    = VK_NULL_HANDLE;
            VkImageView   ampView     = VK_NULL_HANDLE;
            VkImage       slopeImage  = VK_NULL_HANDLE;// rgba16f array, mipped: what the shade samples
            VmaAllocation slopeAlloc  = VK_NULL_HANDLE;
            VkImageView   slopeView   = VK_NULL_HANDLE;// every mip
            VkImageView   slopeTarget = VK_NULL_HANDLE;// mip 0, the compute target
            std::array<water::OceanImage, DisplacedMesh::kMaxWakePatches> work{};// rg32f: the FFT's field, a patch
            water::OceanImage scratch{};                                         // rg32f: its ping-pong, shared
            std::array<VkDescriptorSet, DisplacedMesh::kMaxWakePatches> rippleDs{};
            std::unique_ptr<water::IFFT> ifft;
            std::array<bool, DisplacedMesh::kMaxWakePatches>  rippleHeld{};  // the layer's spectrum holds waves
            std::array<float, DisplacedMesh::kMaxWakePatches> rippleQuiet{}; // s since something last pushed on it
            std::array<float, DisplacedMesh::kMaxWakePatches> rippleSpeed{}; // m/s, its fastest pusher of late
            std::array<float, DisplacedMesh::kMaxWakePatches> rippleSize{};  // m, the period its spectrum is of

            [[nodiscard]] VkImageView     stateView() const { return view[0]; }
            [[nodiscard]] VkImageView     rippleView() const { return slopeView; }// null when it has none
            [[nodiscard]] VkDeviceAddress tableAddress() const { return table.address; }
        };

        explicit WakeFieldPipeline(VulkanContext& ctx);
        ~WakeFieldPipeline();
        WakeFieldPipeline(const WakeFieldPipeline&) = delete;
        WakeFieldPipeline& operator=(const WakeFieldPipeline&) = delete;

        [[nodiscard]] VkSampler   sampler() const { return sampler_; }
        // The ripples': trilinear, anisotropic, REPEAT (a patch's ripples are a period of its own sea).
        [[nodiscard]] VkSampler   rippleSampler() const { return rippleSampler_; }
        [[nodiscard]] VkImageView dummyView() const { return dummyView_; }
        // The 1x1 stand-in's layout transition; once, in a one-shot buffer.
        void initDummy(VkCommandBuffer cb);

        // Images, views, descriptor sets and the table buffer for a wake field
        // of `res` texels a side and `layers` patches, with ripples of
        // `rippleRes` wavenumbers a side (0 = none; else rounded down to a
        // power of two in 128..1024). initState() puts the images in GENERAL
        // and zeroes them and the table (a one-shot buffer, before the State's
        // first record()).
        std::unique_ptr<State> createState(uint32_t res, uint32_t layers, uint32_t rippleRes);
        void initState(VkCommandBuffer cb, State& st);
        void destroyState(State& st);

        // One step of the field, into the frame's command stream: the table
        // and this frame's sources, then the three dispatches for every patch
        // that is on, then the ripples of every patch that has any. A frame at
        // the same time as the last (a still that renders the same instant
        // again) records nothing.
        void record(VkCommandBuffer cb, State& st, const DisplacedMesh& dm, double nowSec);

    private:
        VulkanContext&        ctx_;
        VkDescriptorSetLayout dsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipeline_       = VK_NULL_HANDLE;
        VkDescriptorSetLayout rippleDsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      ripplePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            ripplePipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_       = VK_NULL_HANDLE;
        VkSampler             sampler_        = VK_NULL_HANDLE;
        VkSampler             rippleSampler_  = VK_NULL_HANDLE;
        VkImage               dummyImage_     = VK_NULL_HANDLE;
        VmaAllocation         dummyAlloc_     = VK_NULL_HANDLE;
        VkImageView           dummyView_      = VK_NULL_HANDLE;

        void createPipeline();
        void recordRipples(VkCommandBuffer cb, State& st, uint32_t rippleMask, PushConstants pc);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_WAKE_FIELD_PIPELINE_HPP
