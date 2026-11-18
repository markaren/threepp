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
//
// THE SHORE. Every ocean also carries, after its producers' patches, two
// windows of the same field that follow the camera (kShoreTexel: a near one at
// a quarter metre a texel and a far one at two), in which the sea lays the
// foam it makes where it meets something solid: the white collar round a
// pile, a rock or a quay that comes with each wave and the lace that drifts
// off it. Each window has a SHORE MAP, a top-down slice of the still scene at
// the water's rest level (shore_capture.comp, ray queries; then the distance
// and direction to the nearest solid by jump flooding, shore_jfa.comp), taken
// again only when the window moves (the camera crosses an eighth of it) or the
// scene's structure changes. Every step, after the field's own, the source
// (shore_source.comp) reads the drawn sea at each texel near a shore and lays
// foam and air where the water drives into the solid; the field then carries,
// tears and ages it as it does a wake's. The shade draws it as the wake's foam
// (the shore windows are patches like any other). Nothing about it is a
// switch: a scene with nothing solid at its waterline has an empty map, and
// its water is shaded exactly as before. It needs ray queries; without them an
// ocean has the producers' field alone, as before.

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

    // Mirrors WakePatch in ocean_wake.glsl (scalar layout, 76 bytes).
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
        float rippleHeight;// 1 = the ripple image's alpha holds the waves' height (water_displace adds it)
        // WakeField::rippleBreaking: where its waves' slope passes breakSlope0
        // they break, fully at breakSlope1 (0 = off), laying breakFoam film
        // and breakAir bubbles in the field.
        float breakSlope0;
        float breakSlope1;
        float breakFoam;
        float breakAir;
        float shore;// 1 = one of the shore's windows (fed by shore_source.comp, nothing else)
    };
    static_assert(sizeof(WakePatchGpu) == 76, "WakePatchGpu must match ocean_wake.glsl's WakePatch");

    // The shore's windows, after the producers' patches in every ocean's field.
    constexpr uint32_t kShoreLayers = 2;
    // Their texel (m): the near window's resolves a pile 30 cm across, the far
    // one's the line along a shore seen from a mast or a drone. Each window is
    // the field's resolution of them a side (1024: 256 m and 2 km).
    constexpr float kShoreTexel[kShoreLayers] = {0.25f, 2.0f};
    // The field's resolution for an ocean whose producers asked for none.
    constexpr uint32_t kShoreRes = 1024;
    // The step from which the shore's maps are taken (see recordShore).
    constexpr uint32_t kShoreFirstStep = 4;
    // = OCEAN_WAKE_MAX_PATCHES
    constexpr uint32_t kWakeTablePatches = DisplacedMesh::kMaxWakePatches + kShoreLayers;

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
        float    lace;
        WakePatchGpu patches[kWakeTablePatches];
    };
    static_assert(sizeof(WakeTableGpu) == 68 + 76 * kWakeTablePatches,
                  "WakeTableGpu must match ocean_wake.glsl's WakeTable");
    static_assert(sizeof(DisplacedMesh::WakeSource) == 68,
                  "DisplacedMesh::WakeSource is memcpy'd after the table; ocean_wake.glsl reads 68-byte sources");

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

        // shore_capture.comp's push block (56 bytes).
        struct ShoreCapturePush {
            VkDeviceAddress geomAddr;
            VkDeviceAddress hullAddr;
            VkDeviceAddress countAddr;
            float    originX, originZ, texel, restY;
            uint32_t res, layer, hullCount, _pad;
        };
        // shore_jfa.comp's (20 bytes).
        struct ShoreJfaPush {
            uint32_t res, layer, mode;
            int32_t  step;
            float    texel;
        };
        // shore_source.comp's (96 bytes).
        struct ShoreSourcePush {
            float    tileSize0, tileSize1, tileSize2, waveScale, choppiness;
            uint32_t cascadeMask;
            float    originX, originZ, texel;
            uint32_t res, layer, shoreLayer;
            float    dt;
            uint32_t history;
            float    driftX, driftZ;
            float    keepX, keepZ, keepHalf;
            float    timeSec;
            float    gain;
            float    _pad[3];
        };

        // What the shore's step needs of the frame and of its ocean.
        struct ShoreInputs {
            VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;// the scene's, as the last frame left it
            uint32_t        tlasInstances = 0;              // its instance count: a change is a new scene
            VkDeviceAddress geomAddr      = 0;              // this frame's GeometryDesc array
            VkDeviceAddress hullAddr      = 0;              // the vessels (OceanHullGpu), or 0
            uint32_t        hullCount     = 0;
            float           restY         = 0.f;            // world y of the water's rest level
            VkDescriptorSet displaceSet   = VK_NULL_HANDLE; // the ocean's plain displace set (its cascades)
            VkDescriptorSetLayout displaceLayout = VK_NULL_HANDLE;
            float           tileSize[3]   = {};
            float           waveScale     = 1.f;
            float           choppiness    = 1.f;
            uint32_t        cascadeMask   = 0;
            float           windSpeed     = 0.f;
            float           windTheta     = 0.f;
            uint32_t        frame         = 0;              // the frame in flight
        };

        // One ocean's wake field.
        struct State {
            uint32_t res    = 0;
            uint32_t layers = 0;     // every layer: the producers', then the shore's
            uint32_t appLayers = 0;  // the producers' (DisplacedMesh::WakeField::patches; 0 = none)
            bool     shoreOn   = false;// the shore's windows follow (layers appLayers, appLayers + 1)
            // 0 stateA: what the shade samples, and what a step starts from
            // 1 stateB: carried forward (step 0)     2 stateC: corrected (step 1)
            // 3 velA                                  4 velB
            std::array<VkImage, 5>       image{};
            std::array<VmaAllocation, 5> alloc{};
            std::array<VkImageView, 5>   view{};
            std::array<VkDescriptorSet, 3> ds{};
            Buffer table{};// WakeTableGpu + kMaxWakeSources sources, device-local
            double prevTimeSec = -1.0;
            std::array<float, kWakeTablePatches> prevOriginX{};
            std::array<float, kWakeTablePatches> prevOriginZ{};
            std::array<float, kWakeTablePatches> prevSize{};// 0 = the layer holds nothing
            // The last step: whether record() stepped this frame, and its dt.
            bool  stepped  = false;
            float stepDt   = 0.f;
            double stepTime = 0.0;
            uint32_t steps  = 0;// steps taken since the State was made

            // The shore (shoreOn). Its windows' centres and sides for this
            // frame (setShoreWindows), what each map was taken of, and its
            // images: the maps (rgba16f: distance, direction, bed), the
            // flood's seeds (r32ui, two to ping-pong) and each window's
            // parcels at the last step (rgba16f: eta, D), a layer a window.
            std::array<float, kShoreLayers> shoreX{}, shoreZ{}, shoreSize{};
            std::array<float, kShoreLayers> mapX{}, mapZ{};
            std::array<bool, kShoreLayers>  mapValid{};
            std::array<float, kShoreLayers> lastX{}, lastZ{};// the window lastSurface's parcels are of
            VkAccelerationStructureKHR mapTlas = VK_NULL_HANDLE;
            uint32_t                   mapInstances = 0;
            std::array<VkImage, 4>       shoreImage{};// 0 map, 1 seedsA, 2 seedsB, 3 lastSurface
            std::array<VmaAllocation, 4> shoreAlloc{};
            std::array<VkImageView, 4>   shoreView{};
            std::array<VkDescriptorSet, 2> captureDs{};// one a frame in flight (its TLAS)
            std::array<VkAccelerationStructureKHR, 2> captureTlas{};
            std::array<VkDescriptorSet, 2> jfaDs{};    // A -> B, B -> A
            VkDescriptorSet sourceDs = VK_NULL_HANDLE;
            // Each map's seed count, for the host: a uint a window, a buffer a
            // frame in flight (read back when that frame's slot comes round,
            // its fence waited). A window whose map has no seed holds no
            // shore: it is not stepped, fed or sampled (shoreEmpty), so a
            // scene with nothing at its waterline pays for the maps alone.
            std::array<Buffer, 2>    countBuf{};
            std::array<uint32_t*, 2> countPtr{};
            std::array<int32_t, kShoreLayers>  countSlot{-1, -1};// the slot a map's count is pending in; -1 none
            std::array<uint32_t, kShoreLayers> countCall{};      // the shore call it was taken on
            std::array<bool, kShoreLayers>     shoreEmpty{};     // the last counted map had no seed
            uint32_t shoreCalls = 0;

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
        //
        // `shore`: the shore's two windows follow the producers' `layers`
        // (which may then be 0); it needs ray queries (shoreSupported()).
        std::unique_ptr<State> createState(uint32_t res, uint32_t layers, uint32_t rippleRes, bool shore = false);
        void initState(VkCommandBuffer cb, State& st);
        void destroyState(State& st);

        // Whether this device can take a shore (ray queries).
        [[nodiscard]] bool shoreSupported() const;
        // The shore's windows for this frame, round the camera at (x, z) in the
        // ocean's frame: each centred on the multiple of an eighth of its side
        // nearest the camera, so it moves in whole steps and seldom. Before
        // record().
        static void setShoreWindows(State& st, float eyeX, float eyeZ);
        // The shore's part of the step, after record(): each window's map taken
        // again if it has moved or the scene changed (capture and flood), then
        // the source. Nothing when record() did not step this frame.
        void recordShore(VkCommandBuffer cb, State& st, const ShoreInputs& in, double nowSec);

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

        // The shore's pipelines: the flood and the source (the source's on its
        // first use, with the displace set's layout), the capture where the
        // device has ray queries.
        VkDescriptorPool      shorePool_          = VK_NULL_HANDLE;
        VkDescriptorSetLayout captureDsLayout_    = VK_NULL_HANDLE;
        VkPipelineLayout      capturePipeLayout_  = VK_NULL_HANDLE;
        VkPipeline            capturePipeline_    = VK_NULL_HANDLE;
        VkDescriptorSetLayout jfaDsLayout_        = VK_NULL_HANDLE;
        VkPipelineLayout      jfaPipeLayout_      = VK_NULL_HANDLE;
        VkPipeline            jfaPipeline_        = VK_NULL_HANDLE;
        VkDescriptorSetLayout sourceDsLayout_     = VK_NULL_HANDLE;
        VkPipelineLayout      sourcePipeLayout_   = VK_NULL_HANDLE;
        VkPipeline            sourcePipeline_     = VK_NULL_HANDLE;
        VkDescriptorSetLayout sourceDisplaceLayout_ = VK_NULL_HANDLE;// the layout sourcePipeline_ was made with

        void createPipeline();
        void createShorePipelines();
        void recordRipples(VkCommandBuffer cb, State& st, uint32_t rippleMask, uint32_t heightMask, PushConstants pc);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_WAKE_FIELD_PIPELINE_HPP
