// SmokeGrid — the device-side Eulerian smoke solver behind ParticleField::GridSmoke.
//
// One SmokeGrid is one field's solver state: velocity, density+temperature and
// pressure on a res^3 grid (each a ping-pong pair), one scratch volume, the
// per-frame parameter block, and eight descriptor sets — one per parity of the
// three pairs, written once at creation and never again, so the solver costs no
// descriptor traffic. SmokePipelines is the part that does not depend on a
// field: the nine compute pipelines, their layouts and the two samplers.
//
// A frame is `plan()` (host: how many fixed steps are owed, the parameter block
// and the push constants, all by value) and later `record()` (device: those
// steps, then the pass that writes the density volume in the Q20.12 fixed point
// the particle scatter uses, so the r16f convert after it is shared code).
// Nothing here touches the render graph: every image is private to this block and
// lives in VK_IMAGE_LAYOUT_GENERAL, ordered by plain memory barriers, and the
// one image the graph does know — the field's r32ui density volume — is
// declared by ParticleFieldPass exactly as it is for a scatter.

#ifndef THREEPP_VULKAN_SMOKEGRID_HPP
#define THREEPP_VULKAN_SMOKEGRID_HPP

#include "VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>

namespace threepp {
    class ParticleField;
}

namespace threepp::vulkan {

    class VulkanContext;

    // Mirrors `Pc` in shaders/smoke_common.glsl.
    struct SmokePc {
        std::int32_t res[4];
        float        invCell[4];
        float        box[4];
        float        phys0[4];
        float        phys1[4];
        float        wind[4];
    };
    static_assert(sizeof(SmokePc) == 96, "SmokePc layout drift");

    // How many sources and colliders a field may carry (the rest are ignored). A labyrinth's walls are colliders.
    inline constexpr std::uint32_t kSmokeMaxSources   = 16;
    inline constexpr std::uint32_t kSmokeMaxColliders = 64;
    inline constexpr std::uint32_t kSmokeMaxBoxes     = 128;

    // Mirrors `Params` in shaders/smoke_common.glsl (std430).
    struct SmokeGpuSource {
        float posRadius[4];
        float rates[4];
        float vel[4];
    };
    struct SmokeGpuCollider {
        float a[4];
        float b[4];
        float vel[4];
    };
    struct SmokeGpuBox {
        float lo[4];
        float hi[4];
        float vel[4];
    };
    struct SmokeGpuParams {
        float            head[4];// x sources, y capsule colliders, z time, w boxes
        SmokeGpuSource   src[kSmokeMaxSources];
        SmokeGpuCollider col[kSmokeMaxColliders];
        SmokeGpuBox      box[kSmokeMaxBoxes];
    };
    static_assert(sizeof(SmokeGpuParams) == 16 + kSmokeMaxSources * 48 + kSmokeMaxColliders * 48 + kSmokeMaxBoxes * 48, "SmokeGpuParams layout drift");

    // One frame of one field, resolved on the host so recording reads no field.
    struct SmokePlan {
        std::uint32_t  steps      = 0;
        std::uint32_t  iterations = 24;
        bool           reset      = false;
        SmokePc        pc{};
        SmokeGpuParams params{};
    };

    class SmokePipelines {

    public:
        SmokePipelines(VulkanContext& ctx, VkDescriptorSetLayout densitySetLayout);
        ~SmokePipelines();
        SmokePipelines(const SmokePipelines&)            = delete;
        SmokePipelines& operator=(const SmokePipelines&) = delete;

        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout      layout      = VK_NULL_HANDLE;// set 0
        VkPipelineLayout      writeLayout = VK_NULL_HANDLE;// set 0 + the field's density set (set 1)
        VkSampler             samplerEdge   = VK_NULL_HANDLE;
        VkSampler             samplerBorder = VK_NULL_HANDLE;
        VkPipeline            forces = VK_NULL_HANDLE, curl = VK_NULL_HANDLE, confine = VK_NULL_HANDLE,
                   advect = VK_NULL_HANDLE, advectFix = VK_NULL_HANDLE, diverge = VK_NULL_HANDLE,
                   jacobi = VK_NULL_HANDLE, project = VK_NULL_HANDLE, write = VK_NULL_HANDLE, solid = VK_NULL_HANDLE;

    private:
        VulkanContext& ctx_;
    };

    class SmokeGrid {

    public:
        using CreateImage3DFn = std::function<Image2D(std::uint32_t, std::uint32_t, std::uint32_t, VkFormat,
                                                      VkImageUsageFlags, const char*)>;

        // Null when the images or sets could not be created.
        static std::unique_ptr<SmokeGrid> create(VulkanContext& ctx, const SmokePipelines& pl,
                                                 const CreateImage3DFn& createImage3D, std::uint32_t res);
        ~SmokeGrid();
        SmokeGrid(const SmokeGrid&)            = delete;
        SmokeGrid& operator=(const SmokeGrid&) = delete;

        [[nodiscard]] std::uint32_t resolution() const { return res_; }

        // Advance the field's clock and pack this frame's parameters.
        SmokePlan plan(const ParticleField& field);

        // Run the owed steps and write the density volume (set 1 of the write pass).
        void record(VkCommandBuffer cb, const SmokePipelines& pl, const SmokePlan& plan, VkDescriptorSet densitySet);

    private:
        SmokeGrid(VulkanContext& ctx, std::uint32_t res) : ctx_(ctx), res_(res) {}
        void step(VkCommandBuffer cb, const SmokePipelines& pl, const SmokePlan& plan);

        VulkanContext& ctx_;
        std::uint32_t  res_;
        Image2D        vel_[2]{}, sca_[2]{}, p_[2]{}, aux_{}, solid_{};
        Buffer         params_{};
        VkDescriptorPool pool_ = VK_NULL_HANDLE;
        VkDescriptorSet  sets_[8]{};
        // The ping-pong parities: which half of each pair is "in".
        int v_ = 0, s_ = 0, p_in_ = 0;
        bool ready_ = false;
        // The field's clock.
        bool          clockStarted_ = false;
        double        lastTime_     = 0.0;
        std::uint64_t seenResets_   = 0;
        bool          warnedCaps_   = false;
    };

}// namespace threepp::vulkan

#endif
