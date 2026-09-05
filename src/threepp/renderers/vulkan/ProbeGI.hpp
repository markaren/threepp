// ProbeGI — world-space irradiance probe grid (DDGI-lite) for the deferred
// renderer's multi-bounce GI.
//
// Owns a fixed 32×16×32 grid of SH-L1 irradiance probes fitted to the scene
// AABB (cell-centered) and the probe_update.comp compute pass that refreshes
// a round-robin window of probes each frame: 64 rays per probe via ray query,
// hits shaded with deferred_shade.comp's giRadiance logic (direct analytic
// lights + emissive NEE) plus a probe-grid feedback tap — the tap makes the
// cache converge to the multi-bounce fixed point over a few grid sweeps.
// deferred_shade.comp samples the same SH buffer at its GI-ray hit points
// (bindings 36/37), which is what lets enclosed interiors (the Sponza
// ground-floor corridors) receive courtyard light instead of rendering
// near-black off the 1-bounce gather. Each probe also carries an 8×8
// octahedral map of ray-hit distance moments (mean/mean²) that the sampler
// Chebyshev-tests per tap (DDGI visibility, Majercik et al. 2019) so probes
// on the far side of a wall can't leak irradiance through it.
//
// Follows the SkinningPipeline / EnvPrefilter house pattern: one class, own
// pipeline + descriptor pool, recordDispatch(cb, ...). Buffers exist from
// construction (the SH store is ~1 MB) so the deferred set can always bind
// them; the grid UBO's `enabled` flag gates all sampling when the feature is
// off (VulkanRenderer::setProbeGI, default OFF).

#ifndef THREEPP_VULKAN_PROBE_GI_HPP
#define THREEPP_VULKAN_PROBE_GI_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class ProbeGI {

    public:
        // Grid resolution (probes per axis) — 16384 probes, 64 B each = 1 MB.
        static constexpr uint32_t kDimX = 32, kDimY = 16, kDimZ = 32;
        static constexpr uint32_t kProbeCount = kDimX * kDimY * kDimZ;
        // Rays per probe — must match probe_update.comp's local_size_x.
        static constexpr uint32_t kRaysPerProbe = 64;
        // Chebyshev depth map: kDepthRes² octahedral texels per probe, each
        // packHalf2x16(mean, mean²) ray-hit distance. Must match
        // probe_common.glsl's kProbeDepthRes; texel count must equal
        // kRaysPerProbe (one thread per texel in the update's blend).
        static constexpr uint32_t kDepthRes    = 8;
        static constexpr uint32_t kDepthTexels = kDepthRes * kDepthRes;
        // Round-robin budget: the full grid refreshes every
        // kProbeCount / kProbesPerFrame = 8 frames. 2048 × 64 = 131 k rays
        // per frame — well inside the ~1 ms probe budget on an RTX 4070.
        static constexpr uint32_t kProbesPerFrame = 2048;

        ProbeGI(VulkanContext& ctx, uint32_t framesInFlight);
        ~ProbeGI();
        ProbeGI(const ProbeGI&) = delete;
        ProbeGI& operator=(const ProbeGI&) = delete;

        // Per-scene inputs — the subset of DeferredShade's set the probe rays
        // need (TLAS + lights + env + material/geometry/emissive buffers).
        // Call whenever rewriteDeferredDescriptors runs (TLAS rebuild, env
        // swap, buffer reallocation); idempotent.
        struct DescriptorWriteInputs {
            const VkBuffer* lightsUbo   = nullptr;// [framesInFlight]
            VkImageView     envView     = VK_NULL_HANDLE;// prefiltered PMREM
            VkSampler       envSampler  = VK_NULL_HANDLE;
            VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
            const VkBuffer* materialBuf = nullptr;// [framesInFlight] MaterialDesc[]
            const VkBuffer* geomDescBuf = nullptr;// [framesInFlight] GeometryDesc[] (ringed for auto-LOD level switches)
            const VkDescriptorImageInfo* materialTex = nullptr;// bindless array
            uint32_t        materialTexCount = 0;
            const VkBuffer* emissiveTriBuf = nullptr;// [framesInFlight] EmTri[]
        };
        // onlyFrame >= 0 rewrites just that frame-in-flight slot's set (kept in
        // lockstep with DeferredShade's per-FIF refresh); < 0 rewrites all slots.
        void rewriteDescriptors(const DescriptorWriteInputs& in, int onlyFrame = -1);

        // Rebind just the emissive-triangle buffer for one frame (the per-frame
        // buffer grows) — mirrors DeferredShade::rewriteEmissive.
        void rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf);

        // Fit the grid to the scene's world AABB (probes at cell centers,
        // spacing = extent/dims per axis, min-clamped so a degenerate axis
        // doesn't collapse the grid). Schedules a zero-fill of the SH store
        // on the next recordDispatch — stale radiance from the previous fit
        // must not bleed into the new layout.
        void setGridBounds(const float aabbMin[3], const float aabbMax[3]);
        // Zero the SH store, validity and history on the next dispatch, exactly
        // as a fresh grid fit does: probes bootstrap with alpha = 1 instead of
        // blending into whatever the store held. Part of the renderer's
        // resetTemporalHistory().
        void invalidateHistory() {
            needsClear_ = true;
            // The update cursor is temporal state as well: it advances by
            // kProbesPerFrame per frame, so its phase encodes how many frames
            // preceded the reset, and two captures that began after different
            // frame counts would update different probe slices on the same
            // window frame.
            probeOffset_ = 0;
            // Same argument for the update counter, and it is the stronger
            // half: it is the shader's `frame`, so it seeds every probe ray and
            // rotates the ray sphere. Left running, a reset grid rebuilt itself
            // from a different set of directions depending on how many frames
            // preceded the reset, and two runs of the same scene in one process
            // could not produce the same grid. setGridBounds already zeroed
            // both; this path zeroed only the cursor.
            updateCounter_ = 0;
            // The clear above already forces a full prev refresh; this keeps
            // the two facts from having to be reasoned about together.
            haveWindow_ = false;
        }
        [[nodiscard]] bool gridFitted() const { return gridFitted_; }

        // Upload this frame's grid UBO (origin/spacing/dims + the sampling
        // enable). Call every frame BEFORE recording — deferred_shade reads
        // the same UBO at binding 37.
        void updateGridUbo(uint32_t frame, bool enabled);

        // Record the round-robin probe update (kProbesPerFrame workgroups ×
        // kRaysPerProbe rays). The caller provides the AS-build→compute
        // barrier before and the probe-write→shade-read barrier after.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t emissiveCount, float emissiveTotalPower,
                            bool shadows, uint32_t envMipCount);

        // For DeferredShade's descriptor write (bindings 36/37/54).
        [[nodiscard]] VkBuffer shBuffer() const { return shBuf_.handle; }
        [[nodiscard]] const VkBuffer* gridUbos() const { return gridUboHandles_.data(); }
        [[nodiscard]] VkBuffer depthBuffer() const { return depthBuf_.handle; }

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        Buffer                shBuf_{};   // kProbeCount × 4 × vec4 SH-L1 store
        Buffer                depthBuf_{};// kProbeCount × kDepthTexels × uint Chebyshev depth
        // Start-of-dispatch snapshots (recordDispatch copies canonical → prev
        // before every update). The update's feedback tap / EMA history reads
        // go here, its writes to the canonical stores above — race-free, so
        // two same-seed runs produce bit-identical grids. Consumers keep
        // reading the canonical stores; no plumbing outside this class.
        Buffer                prevShBuf_{};
        Buffer                prevDepthBuf_{};
        std::vector<Buffer>   gridUbos_;        // [framesInFlight]
        std::vector<VkBuffer> gridUboHandles_;  // handles view for DeferredShade

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        float    gridOrigin_[3]  = {0.f, 0.f, 0.f};
        float    gridSpacing_[3] = {1.f, 1.f, 1.f};
        bool     gridFitted_     = false;
        bool     needsClear_     = true;// zero the SH store on first dispatch
        uint32_t probeOffset_    = 0;   // round-robin cursor
        uint32_t updateCounter_  = 0;   // rotates the per-probe ray sphere
        // The window the LAST dispatch wrote — the only place the canonical
        // stores can differ from the prev snapshots, and therefore the only
        // part the next dispatch's snapshot has to copy. haveWindow_ false
        // means "prev is not known to match": copy everything.
        uint32_t lastOffset_     = 0;
        uint32_t lastCount_      = 0;
        bool     haveWindow_     = false;

        void createPipeline();
        void createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_PROBE_GI_HPP
