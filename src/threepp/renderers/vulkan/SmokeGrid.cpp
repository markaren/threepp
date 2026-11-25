#include "SmokeGrid.hpp"

#include "VulkanContext.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/vulkan/DescriptorShadow.hpp"
#include "threepp/renderers/vulkan/VulkanLog.hpp"

#include "threepp/renderers/vulkan/shaders/smoke_advect.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_advect_fix.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_confine.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_curl.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_diverge.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_forces.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_jacobi.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_project.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_solid.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/smoke_write.comp.spv.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace threepp;
using namespace threepp::vulkan;

namespace {

    // Matches particle_density.glsl's kParticleDensityScale (Q20.12).
    constexpr float kFixedScale = 4096.f;

    // TRANSFER_SRC: the state is copied out by VulkanRenderer::readGridSmoke (tests and audits).
    constexpr VkImageUsageFlags kGridUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    void computeBarrier(VkCommandBuffer cb) {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &dep);
    }

    void memBarrier(VkCommandBuffer cb, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = srcStage;
        mb.srcAccessMask = srcAccess;
        mb.dstStageMask  = dstStage;
        mb.dstAccessMask = dstAccess;
        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &dep);
    }

}// namespace

// ── SmokePipelines ───────────────────────────────────────────────────────────

SmokePipelines::SmokePipelines(VulkanContext& ctx, VkDescriptorSetLayout densitySetLayout) : ctx_(ctx) {

    const VkDevice d = ctx.device();

    // Binding order mirrors shaders/smoke_common.glsl.
    const VkDescriptorType types[10] = {
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    VkDescriptorSetLayoutBinding b[10]{};
    for (std::uint32_t i = 0; i < 10; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = types[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 10;
    dlci.pBindings    = b;
    check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &setLayout), "vkCreateDescriptorSetLayout(smoke)");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size       = sizeof(SmokePc);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    check(vkCreatePipelineLayout(d, &plci, nullptr, &layout), "vkCreatePipelineLayout(smoke)");
    const VkDescriptorSetLayout two[2] = {setLayout, densitySetLayout};
    plci.setLayoutCount                = 2;
    plci.pSetLayouts                   = two;
    check(vkCreatePipelineLayout(d, &plci, nullptr, &writeLayout), "vkCreatePipelineLayout(smoke write)");

    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // Beyond the box there is no smoke and the air is the wind: density, pressure and the velocity (stored relative
    // to the wind) all read as zero.
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    check(vkCreateSampler(d, &sci, nullptr, &samplerBorder), "vkCreateSampler(smoke border)");

    const auto make = [&](VkPipelineLayout l, const std::uint32_t* spv, size_t bytes, const char* name) {
        return createComputePipeline(d, ctx.pipelineCache(), l, spv, bytes, name);
    };
    forces    = make(layout, kSmokeForcesCompSpv, sizeof(kSmokeForcesCompSpv), "smoke_forces");
    curl      = make(layout, kSmokeCurlCompSpv, sizeof(kSmokeCurlCompSpv), "smoke_curl");
    confine   = make(layout, kSmokeConfineCompSpv, sizeof(kSmokeConfineCompSpv), "smoke_confine");
    advect    = make(layout, kSmokeAdvectCompSpv, sizeof(kSmokeAdvectCompSpv), "smoke_advect");
    advectFix = make(layout, kSmokeAdvectFixCompSpv, sizeof(kSmokeAdvectFixCompSpv), "smoke_advect_fix");
    diverge   = make(layout, kSmokeDivergeCompSpv, sizeof(kSmokeDivergeCompSpv), "smoke_diverge");
    jacobi    = make(layout, kSmokeJacobiCompSpv, sizeof(kSmokeJacobiCompSpv), "smoke_jacobi");
    project   = make(layout, kSmokeProjectCompSpv, sizeof(kSmokeProjectCompSpv), "smoke_project");
    write     = make(writeLayout, kSmokeWriteCompSpv, sizeof(kSmokeWriteCompSpv), "smoke_write");
    solid     = make(layout, kSmokeSolidCompSpv, sizeof(kSmokeSolidCompSpv), "smoke_solid");
}

SmokePipelines::~SmokePipelines() {
    const VkDevice d = ctx_.device();
    for (VkPipeline p : {forces, curl, confine, advect, advectFix, diverge, jacobi, project, write, solid})
        if (p) vkDestroyPipeline(d, p, nullptr);
    if (layout) vkDestroyPipelineLayout(d, layout, nullptr);
    if (writeLayout) vkDestroyPipelineLayout(d, writeLayout, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(d, setLayout, nullptr);
    if (samplerBorder) vkDestroySampler(d, samplerBorder, nullptr);
}

// ── SmokeGrid ────────────────────────────────────────────────────────────────

std::unique_ptr<SmokeGrid> SmokeGrid::create(VulkanContext& ctx, const SmokePipelines& pl,
                                             const CreateImage3DFn& createImage3D, std::uint32_t res) {

    if (!createImage3D) return nullptr;
    std::unique_ptr<SmokeGrid> g(new SmokeGrid(ctx, res));

    const auto mk = [&](Image2D& img, std::uint32_t n, VkFormat fmt, const char* name) {
        img = createImage3D(n, n, n, fmt, kGridUsage, name);
        return img.image != VK_NULL_HANDLE;
    };
    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        // The velocity lives on the faces: res + 1 of them along each axis (smoke_common.glsl).
        ok = mk(g->vel_[i], res + 1, VK_FORMAT_R16G16B16A16_SFLOAT, "smokeVelocity") && ok;
        ok = mk(g->sca_[i], res, VK_FORMAT_R16G16_SFLOAT, "smokeScalars") && ok;
        ok = mk(g->p_[i], res, VK_FORMAT_R32_SFLOAT, "smokePressure") && ok;
    }
    ok = mk(g->aux_, res, VK_FORMAT_R16G16B16A16_SFLOAT, "smokeScratch") && ok;
    ok = mk(g->solid_, res, VK_FORMAT_R16G16B16A16_SFLOAT, "smokeSolid") && ok;
    if (!ok) return nullptr;// ~SmokeGrid frees what was made

    g->params_ = createBuffer(ctx.allocator(), ctx.device(), sizeof(SmokeGpuParams),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO, 0);
    if (g->params_.handle == VK_NULL_HANDLE) return nullptr;

    VkDescriptorPoolSize ps[3]{};
    ps[0] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * 8};
    ps[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 * 8};
    ps[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 8;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes    = ps;
    check(vkCreateDescriptorPool(ctx.device(), &dpci, nullptr, &g->pool_), "vkCreateDescriptorPool(smoke)");
    VkDescriptorSetLayout layouts[8];
    for (auto& l : layouts) l = pl.setLayout;
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = g->pool_;
    ai.descriptorSetCount = 8;
    ai.pSetLayouts        = layouts;
    check(vkAllocateDescriptorSets(ctx.device(), &ai, g->sets_), "vkAllocateDescriptorSets(smoke)");

    // One set per parity of (velocity, scalars, pressure): in = [parity], out = [1 - parity].
    for (int idx = 0; idx < 8; ++idx) {
        const int v = idx & 1, s = (idx >> 1) & 1, p = (idx >> 2) & 1;
        VkDescriptorImageInfo ii[8]{};
        const auto cis = [&](int i, VkSampler smp, const Image2D& img) {
            ii[i] = {smp, img.view, VK_IMAGE_LAYOUT_GENERAL};
        };
        const auto sto = [&](int i, const Image2D& img) { ii[i] = {VK_NULL_HANDLE, img.view, VK_IMAGE_LAYOUT_GENERAL}; };
        // Border: beyond the box the stored velocity (relative to the wind) is 0, the air there is the wind.
        cis(0, pl.samplerBorder, g->vel_[v]);
        sto(1, g->vel_[1 - v]);
        cis(2, pl.samplerBorder, g->sca_[s]);
        sto(3, g->sca_[1 - s]);
        cis(4, pl.samplerBorder, g->p_[p]);
        sto(5, g->p_[1 - p]);
        sto(6, g->aux_);
        cis(7, pl.samplerBorder, g->aux_);
        VkDescriptorBufferInfo bi{g->params_.handle, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[10]{};
        for (std::uint32_t i = 0; i < 10; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = g->sets_[idx];
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
        }
        const VkDescriptorType t[8] = {
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
        for (int i = 0; i < 8; ++i) {
            w[i].descriptorType = t[i];
            w[i].pImageInfo     = &ii[i];
        }
        w[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[8].pBufferInfo    = &bi;
        VkDescriptorImageInfo si{VK_NULL_HANDLE, g->solid_.view, VK_IMAGE_LAYOUT_GENERAL};
        w[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[9].pImageInfo     = &si;
        vulkan::updateDescriptorSets(ctx.device(), 10, w, 0, nullptr);
    }
    return g;
}

SmokeGrid::~SmokeGrid() {
    const VkDevice d = ctx_.device();
    if (pool_) vkDestroyDescriptorPool(d, pool_, nullptr);
    for (auto& i : vel_) destroyImage2D(ctx_.allocator(), d, i);
    for (auto& i : sca_) destroyImage2D(ctx_.allocator(), d, i);
    for (auto& i : p_) destroyImage2D(ctx_.allocator(), d, i);
    destroyImage2D(ctx_.allocator(), d, aux_);
    destroyImage2D(ctx_.allocator(), d, solid_);
    destroyBuffer(ctx_.allocator(), params_);
}

SmokePlan SmokeGrid::plan(const ParticleField& field) {

    const auto& gs = field.gridSmoke();
    const auto& dr = field.densityRepr();
    SmokePlan   out;

    // ── the clock ───────────────────────────────────────────────────────────
    const float dt = std::max(gs.stepDt, 1e-4f);
    if (!std::isfinite(gs.time)) {
        // A clock that is not a number runs no steps (the count below would be a NaN cast to an integer).
    } else if (!clockStarted_ || field.gridSmokeResets() != seenResets_) {
        clockStarted_ = true;
        seenResets_   = field.gridSmokeResets();
        lastTime_     = gs.time;
        out.reset     = true;
    } else {
        const double owed = (gs.time - lastTime_) / static_cast<double>(dt);
        if (owed < 0.0) {// the application's clock went backwards: restart from here
            lastTime_ = gs.time;
            out.reset = true;
        } else {
            const std::uint32_t whole = static_cast<std::uint32_t>(std::min(owed, 1.0e6));
            out.steps                 = std::min(whole, std::max(gs.maxStepsPerFrame, 1u));
            lastTime_ += static_cast<double>(out.steps) * dt;
            // Beyond the cap the plume runs slow: the time it could not run is dropped, not owed.
            if (whole > out.steps) lastTime_ = gs.time;
        }
    }
    out.iterations = std::clamp(gs.pressureIterations, 4u, 200u);

    // ── push constants ──────────────────────────────────────────────────────
    const float hx = std::max(dr.halfExtent.x, 1e-4f), hy = std::max(dr.halfExtent.y, 1e-4f),
                hz = std::max(dr.halfExtent.z, 1e-4f);
    const float n = static_cast<float>(res_);
    SmokePc&    pc = out.pc;
    pc.res[0] = pc.res[1] = pc.res[2] = static_cast<std::int32_t>(res_);
    pc.res[3]                         = 0;
    pc.invCell[0]                     = n / (2.f * hx);
    pc.invCell[1]                     = n / (2.f * hy);
    pc.invCell[2]                     = n / (2.f * hz);
    pc.invCell[3]                     = dt;
    pc.box[0]                         = dr.center.x - hx;
    pc.box[1]                         = dr.center.y - hy;
    pc.box[2]                         = dr.center.z - hz;
    pc.box[3]                         = gs.solidFloor ? 1.f : 0.f;
    pc.phys0[0]                       = gs.buoyancy;
    pc.phys0[1]                       = gs.weight;
    pc.phys0[2]                       = std::clamp(gs.boundaryFade, 0.f, 0.45f);
    pc.phys0[3]                       = std::max(gs.dissipation, 0.f);
    pc.phys1[0]                       = std::max(gs.cooling, 0.f);
    pc.phys1[1]                       = std::max(gs.vorticity, 0.f);
    pc.phys1[2]                       = std::max(gs.damping, 0.f);
    pc.phys1[3]                       = std::max(gs.sigma, 0.f) * kFixedScale;
    pc.wind[0]                        = gs.wind.x;
    pc.wind[1]                        = gs.wind.y;
    pc.wind[2]                        = gs.wind.z;
    pc.wind[3]                        = std::max(gs.maxSpeed, 0.1f);
    pc.turb[0]                        = std::isfinite(gs.turbulence) ? std::max(gs.turbulence, 0.f) : 0.f;
    pc.turb[1]                        = std::max(gs.turbulenceScale, 0.05f);
    pc.turb[2] = pc.turb[3] = 0.f;

    // ── the sources and colliders ───────────────────────────────────────────
    SmokeGpuParams& P  = out.params;
    const size_t    ns = std::min<size_t>(gs.sources.size(), kSmokeMaxSources), nc = std::min<size_t>(gs.colliders.size(), kSmokeMaxColliders);
    P.head[0]          = static_cast<float>(ns);
    P.head[1]          = static_cast<float>(nc);
    const size_t nb = std::min<size_t>(gs.boxes.size(), kSmokeMaxBoxes);
    P.head[3]       = static_cast<float>(nb);
    if (!warnedCaps_ && (gs.sources.size() > kSmokeMaxSources || gs.colliders.size() > kSmokeMaxColliders ||
                         gs.boxes.size() > kSmokeMaxBoxes)) {
        warnedCaps_ = true;
        THREEPP_LOGF(kLogParticles, Warn, "GridSmoke: %zu sources, %zu colliders and %zu boxes given; only the first %zu, %zu "
                     "and %zu are used",
                     gs.sources.size(), gs.colliders.size(), gs.boxes.size(), size_t(kSmokeMaxSources),
                     size_t(kSmokeMaxColliders), size_t(kSmokeMaxBoxes));
    }
    P.head[2]          = static_cast<float>(std::fmod(gs.time, 1000.0));
    for (size_t i = 0; i < ns; ++i) {
        const auto& s = gs.sources[i];
        auto&       g = P.src[i];
        g.posRadius[0] = s.position.x;
        g.posRadius[1] = s.position.y;
        g.posRadius[2] = s.position.z;
        g.posRadius[3] = std::max(s.radius, 1e-3f);
        g.rates[0]     = s.density;
        g.rates[1]     = s.temperature;
        g.rates[2]     = s.velocityBlend;
        g.vel[0]       = s.velocity.x;
        g.vel[1]       = s.velocity.y;
        g.vel[2]       = s.velocity.z;
        g.vel[3]       = std::clamp(s.noise, 0.f, 1.f);
    }
    for (size_t i = 0; i < nc; ++i) {
        const auto& c = gs.colliders[i];
        auto&       g = P.col[i];
        g.a[0]        = c.a.x;
        g.a[1]        = c.a.y;
        g.a[2]        = c.a.z;
        g.a[3]        = std::max(c.radius, 1e-3f);
        g.b[0]        = c.b.x;
        g.b[1]        = c.b.y;
        g.b[2]        = c.b.z;
        g.vel[0]      = c.velocity.x;
        g.vel[1]      = c.velocity.y;
        g.vel[2]      = c.velocity.z;
    }
    for (size_t i = 0; i < nb; ++i) {
        const auto& b = gs.boxes[i];
        auto&       g = P.box[i];
        g.lo[0]       = b.min.x;
        g.lo[1]       = b.min.y;
        g.lo[2]       = b.min.z;
        g.hi[0]       = b.max.x;
        g.hi[1]       = b.max.y;
        g.hi[2]       = b.max.z;
        g.vel[0]      = b.velocity.x;
        g.vel[1]      = b.velocity.y;
        g.vel[2]      = b.velocity.z;
    }
    return out;
}

void SmokeGrid::step(VkCommandBuffer cb, const SmokePipelines& pl, const SmokePlan& plan) {

    // Every pass covers the velocity's res + 1 faces per axis; the cell passes skip the extra layer.
    const std::uint32_t f  = res_ + 1u;
    const std::uint32_t gx = (f + 7u) / 8u, gy = (f + 3u) / 4u, gz = (f + 3u) / 4u;
    const auto run = [&](VkPipeline pipe) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        const VkDescriptorSet set = sets_[v_ | (s_ << 1) | (p_in_ << 2)];
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SmokePc), &plan.pc);
        vkCmdDispatch(cb, gx, gy, gz);
        computeBarrier(cb);
    };

    // Advection first, with the velocity the last projection left: the smoke is only ever carried by air that
    // neither gathers nor spreads, so a sealed box keeps what it holds. The forces come after it and the
    // projection cleans up what they add.
    run(pl.advect);// vel: A -> B (the new velocity), aux.xy = the scalars' predictor
    run(pl.advectFix);// scalars: A -> B, reading the OLD velocity (v_ has not flipped past it)
    v_ ^= 1;
    s_ ^= 1;
    run(pl.forces);// vel, scalars: A -> B
    v_ ^= 1;
    s_ ^= 1;
    run(pl.curl);// aux = curl(vel)
    run(pl.confine);// vel: A -> B
    v_ ^= 1;
    run(pl.diverge);// aux.x = div(vel)
    for (std::uint32_t i = 0; i < plan.iterations; ++i) {
        run(pl.jacobi);// pressure: A -> B
        p_in_ ^= 1;
    }
    run(pl.project);// vel: A -> B
    v_ ^= 1;
}

void SmokeGrid::record(VkCommandBuffer cb, const SmokePipelines& pl, const SmokePlan& plan,
                       VkDescriptorSet densitySet) {

    if (!ready_ || plan.reset) {
        const VkImage all[8] = {vel_[0].image, vel_[1].image, sca_[0].image, sca_[1].image,
                                p_[0].image,   p_[1].image,   aux_.image,    solid_.image};
        if (!ready_) {
            VkImageMemoryBarrier2 ib[8]{};
            for (int i = 0; i < 8; ++i) {
                ib[i].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                ib[i].srcStageMask     = VK_PIPELINE_STAGE_2_NONE;
                ib[i].dstStageMask     = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ib[i].dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                ib[i].oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
                ib[i].newLayout        = VK_IMAGE_LAYOUT_GENERAL;
                ib[i].srcQueueFamilyIndex = ib[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ib[i].image            = all[i];
                ib[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            }
            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 8;
            dep.pImageMemoryBarriers    = ib;
            vkCmdPipelineBarrier2(cb, &dep);
        } else {
            memBarrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }
        const VkClearColorValue zero{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (VkImage img : all) vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        v_ = s_ = p_in_ = 0;
        ready_          = true;
    }

    // This frame's sources and colliders. The previous frame's reads of the block are ordered before the
    // write, and the write before this frame's dispatches.
    memBarrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
               VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdUpdateBuffer(cb, params_.handle, 0, sizeof(SmokeGpuParams), &plan.params);
    memBarrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // The colliders, rasterised once for all of this frame's steps (they do not move between them).
    if (plan.steps > 0 && plan.params.head[1] + plan.params.head[3] > 0.5f) {
        const std::uint32_t sx = (res_ + 7u) / 8u, sy = (res_ + 3u) / 4u, sz = (res_ + 3u) / 4u;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl.solid);
        const VkDescriptorSet set = sets_[v_ | (s_ << 1) | (p_in_ << 2)];
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, pl.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SmokePc), &plan.pc);
        vkCmdDispatch(cb, sx, sy, sz);
        computeBarrier(cb);
    }

    for (std::uint32_t i = 0; i < plan.steps; ++i) step(cb, pl, plan);

    // The density grid -> the field's sigma_t volume.
    const std::uint32_t gx = (res_ + 7u) / 8u, gy = (res_ + 3u) / 4u, gz = (res_ + 3u) / 4u;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl.write);
    const VkDescriptorSet both[2] = {sets_[v_ | (s_ << 1) | (p_in_ << 2)], densitySet};
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl.writeLayout, 0, 2, both, 0, nullptr);
    vkCmdPushConstants(cb, pl.writeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SmokePc), &plan.pc);
    vkCmdDispatch(cb, gx, gy, gz);
}
