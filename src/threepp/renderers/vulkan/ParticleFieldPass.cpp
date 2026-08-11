#include "ParticleFieldPass.hpp"

#include "VulkanContext.hpp"

#include "threepp/objects/ParticleField.hpp"

#include "threepp/renderers/vulkan/shaders/particle_density_convert.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/particle_density_scatter.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/particle_emit.comp.spv.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

using namespace threepp;
using namespace threepp::vulkan;

namespace {

    // Positions: read by every future consumer as a buffer_reference (so
    // SHADER_DEVICE_ADDRESS) and, from phase 1, copied into prevPositions at the
    // head of the frame's particle block (so TRANSFER_SRC).
    constexpr VkBufferUsageFlags kPositionUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // Counts: TRANSFER_SRC because §1.3's cleanest liveCount → instanceCount
    // route is a 4-byte device copy into a VkDrawIndirectCommand; INDIRECT so a
    // future consumer can dispatch off it directly; STORAGE + device address so
    // a shader (or, under Interop, a CUDA kernel) can write it.
    constexpr VkBufferUsageFlags kCountsUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // Orientations: read as a buffer_reference by the particle vertex stage.
    constexpr VkBufferUsageFlags kOrientationUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // The field's one draw command. TRANSFER_DST because its instanceCount word
    // is filled by a device copy from the counts block, never by the host.
    constexpr VkBufferUsageFlags kIndirectUsage =
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    constexpr VmaAllocationCreateFlags kHostWrite =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    // Byte offset of VkDrawIndirectCommand::instanceCount. Spelled out because
    // the whole GPU-side-count route is "copy 4 bytes to exactly here".
    constexpr VkDeviceSize kInstanceCountOffset = sizeof(std::uint32_t);
    static_assert(offsetof(VkDrawIndirectCommand, instanceCount) == 4,
                  "VkDrawIndirectCommand::instanceCount moved");

    // Density volume. STORAGE for the scatter's imageAtomicAdd, SAMPLED for the
    // froxel passes' texelFetch, TRANSFER_DST for the per-frame zero-clear —
    // which is a vkCmdClearColorImage rather than a compute clear because it is
    // one command, needs no descriptor, and is exactly deterministic.
    // TRANSFER_SRC is for VulkanRenderer::readParticleDensityVolume, the
    // debug/test readback that turns the determinism claim into an assertion —
    // it is the only place the volume is ever a copy SOURCE.
    constexpr VkImageUsageFlags kDensityUsage =
            VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    // The r16f mirror: STORAGE for the convert's imageStore, SAMPLED for the
    // deferred shade's hardware-trilinear march. Never cleared, never read
    // back — the convert overwrites every voxel every frame.
    constexpr VkImageUsageFlags kDensityLinUsage =
            VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;

    constexpr std::uint32_t kScatterLocalSize = 64;// == particle_density_scatter.comp
    constexpr std::uint32_t kConvertLocalSize = 4; // == particle_density_convert.comp (4³)

    // MUST mirror the push block in particle_density_scatter.comp under scalar
    // layout. 120 B — inside the 128 B every Vulkan implementation guarantees.
    // The explicit tail pad is what makes "scalar layout" and "what MSVC lays
    // out for a struct containing a uint64" the same number rather than two
    // numbers that happen to agree today.
    struct DensityScatterPc {
        float           world[16];
        VkDeviceAddress posAddr;
        VkDeviceAddress countAddr;
        float           boxMin[3];
        float           boxInvSize[3];
        std::uint32_t   res;
        std::uint32_t   capacity;
        float           sigmaFixed;
        float           _pad;
    };
    static_assert(sizeof(DensityScatterPc) == 120,
                  "particle_density_scatter push-constant drift");

    constexpr std::uint32_t kEmitLocalSize = 64;// == particle_emit.comp

    // MUST mirror the push block in particle_emit.comp under scalar layout,
    // member for member and offset for offset. 128 B EXACTLY — the ceiling every
    // Vulkan implementation guarantees — which is why the trailing reserve is
    // spelled out rather than left to grow by accident: the next member added
    // here without removing one is a device that cannot create the pipeline.
    //
    // The two uint64s sit first so the block's 8-byte alignment is satisfied at
    // offset 0 and every float after them is naturally 4-aligned, making MSVC's
    // layout and GLSL's scalar layout the same bytes by construction rather than
    // by coincidence (the std140 tail-pack trap, feedback_vulkan_material_gpu_update).
    struct EmitPc {
        VkDeviceAddress posAddr;      //   0
        VkDeviceAddress prevPosAddr;  //   8
        float spawnCenter[3];         //  16
        float lifetime;               //  28
        float spawnHalf[3];           //  32
        float lifetimeJitter;         //  44
        float velocity[3];            //  48
        float speedSpread;            //  60
        float accel[3];               //  64
        float duty;                   //  76
        float driftAmp;               //  80
        float driftFreq;              //  84
        float driftGrowth;            //  88
        float size;                   //  92
        float sizeJitter;             //  96
        float time;                   // 100
        float dt;                     // 104
        std::uint32_t capacity;       // 108
        std::uint32_t seed;           // 112
        float driftScale;             // 116
        float _rsv0;                  // 120
        float _rsv1;                  // 124
    };
    static_assert(sizeof(EmitPc) == 128, "particle_emit push-constant drift");
    static_assert(offsetof(EmitPc, seed) == 112, "particle_emit push-constant layout drift");

}// namespace

ParticleFieldPass::ParticleFieldPass(VulkanContext& ctx, RetireBufferFn retireFn,
                                     RetireImageFn retireImageFn,
                                     CreateImage3DFn createImage3DFn)
    : ctx_(ctx), retireFn_(std::move(retireFn)),
      retireImageFn_(std::move(retireImageFn)),
      createImage3DFn_(std::move(createImage3DFn)) {}

ParticleFieldPass::~ParticleFieldPass() {

    // Destructor only: the renderer destroys this after vkDeviceWaitIdle, so
    // nothing can still name these. Inline destroy, not retire — the retire
    // queue is being torn down alongside us.
    for (auto& [_, st] : states_) destroyState(*st);
    states_.clear();
    for (auto& b : descBufs_) destroyBuffer(ctx_.allocator(), b);

    const VkDevice d = ctx_.device();
    if (emitPipe_)          vkDestroyPipeline(d, emitPipe_, nullptr);
    if (emitPipeLayout_)    vkDestroyPipelineLayout(d, emitPipeLayout_, nullptr);
    if (densityPipe_)       vkDestroyPipeline(d, densityPipe_, nullptr);
    if (densityPipeLayout_) vkDestroyPipelineLayout(d, densityPipeLayout_, nullptr);
    if (convertPipe_)       vkDestroyPipeline(d, convertPipe_, nullptr);
    if (convertPipeLayout_) vkDestroyPipelineLayout(d, convertPipeLayout_, nullptr);
    // The pool frees its sets; the sets are not freed individually anywhere.
    if (densityPool_)       vkDestroyDescriptorPool(d, densityPool_, nullptr);
    if (densityDsLayout_)   vkDestroyDescriptorSetLayout(d, densityDsLayout_, nullptr);
    if (convertDsLayout_)   vkDestroyDescriptorSetLayout(d, convertDsLayout_, nullptr);
}

void ParticleFieldPass::retireOrDestroy(Buffer& b) {

    if (b.handle == VK_NULL_HANDLE) return;
    if (retireFn_) retireFn_(std::move(b));
    else destroyBuffer(ctx_.allocator(), b);
    b = Buffer{};
}

void ParticleFieldPass::retireOrDestroy(Image2D& img) {

    if (img.image == VK_NULL_HANDLE) return;
    if (retireImageFn_) retireImageFn_(std::move(img));
    else destroyImage2D(ctx_.allocator(), ctx_.device(), img);
    img = Image2D{};
}

void ParticleFieldPass::destroyState(State& st) {

    for (auto& b : st.positions) destroyBuffer(ctx_.allocator(), b);
    destroyBuffer(ctx_.allocator(), st.devPositions);
    destroyBuffer(ctx_.allocator(), st.devPrevPositions);
    for (auto& b : st.counts) destroyBuffer(ctx_.allocator(), b);
    for (auto& b : st.indirect) destroyBuffer(ctx_.allocator(), b);
    destroyBuffer(ctx_.allocator(), st.orientations);
    destroyImage2D(ctx_.allocator(), ctx_.device(), st.density);
    destroyImage2D(ctx_.allocator(), ctx_.device(), st.densityLin);
}

// The emitter pipeline: ONE compute stage, ZERO descriptor sets, one 128 B push
// range. The absence of a descriptor set is the point — there is no set to
// allocate, no pool to size, no handle to go stale and nothing that could ever
// be written while a frame that names it is in flight (R6 / VUID-03047).
void ParticleFieldPass::ensureEmitPipeline() {

    if (emitPipe_ != VK_NULL_HANDLE) return;
    const VkDevice d = ctx_.device();

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(EmitPc);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 0;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    check(vkCreatePipelineLayout(d, &plci, nullptr, &emitPipeLayout_),
          "vkCreatePipelineLayout(particle emit)");

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kParticleEmitCompSpv);
    smci.pCode    = kParticleEmitCompSpv;
    VkShaderModule mod = VK_NULL_HANDLE;
    check(vkCreateShaderModule(d, &smci, nullptr, &mod),
          "vkCreateShaderModule(particle_emit)");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = emitPipeLayout_;
    const VkResult r = vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci,
                                                nullptr, &emitPipe_);
    vkDestroyShaderModule(d, mod, nullptr);
    check(r, "vkCreateComputePipelines(particle_emit)");
}

void ParticleFieldPass::ensureDensityPipeline() {

    if (densityPipe_ != VK_NULL_HANDLE) return;
    const VkDevice d = ctx_.device();

    // ONE binding: the volume. Everything else per-field rides in the push
    // block, which is what keeps the per-frame descriptor traffic at zero —
    // a set is written once, when a field's volume is created, and never again.
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 1;
    dlci.pBindings    = &b;
    check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &densityDsLayout_),
          "vkCreateDescriptorSetLayout(particle density)");

    // Sized for kMaxDensityFields fields, each holding TWO sets: the scatter's
    // (1 storage image) and the convert's (2 storage images). A field past
    // that gets no volume, which densityOverflowCount reports.
    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps.descriptorCount = kMaxDensityFields * 3u;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // FREE_DESCRIPTOR_SET: a destroyed field returns its sets to the pool. The
    // pool is exactly kMaxDensityFields fields deep, so without this a scene
    // that created and dropped four dust fields could never have a fifth.
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = kMaxDensityFields * 2u;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    check(vkCreateDescriptorPool(d, &dpci, nullptr, &densityPool_),
          "vkCreateDescriptorPool(particle density)");

    // Convert set layout: binding 0 = r32ui src, binding 1 = r16f dst.
    VkDescriptorSetLayoutBinding cb[2]{};
    for (std::uint32_t i = 0; i < 2; ++i) {
        cb[i].binding         = i;
        cb[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        cb[i].descriptorCount = 1;
        cb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo clci{};
    clci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    clci.bindingCount = 2;
    clci.pBindings    = cb;
    check(vkCreateDescriptorSetLayout(d, &clci, nullptr, &convertDsLayout_),
          "vkCreateDescriptorSetLayout(particle density convert)");

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(DensityScatterPc);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &densityDsLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    check(vkCreatePipelineLayout(d, &plci, nullptr, &densityPipeLayout_),
          "vkCreatePipelineLayout(particle density)");

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kParticleDensityScatterCompSpv);
    smci.pCode    = kParticleDensityScatterCompSpv;
    VkShaderModule mod = VK_NULL_HANDLE;
    check(vkCreateShaderModule(d, &smci, nullptr, &mod),
          "vkCreateShaderModule(particle_density_scatter)");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = densityPipeLayout_;
    const VkResult r = vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci,
                                                nullptr, &densityPipe_);
    vkDestroyShaderModule(d, mod, nullptr);
    check(r, "vkCreateComputePipelines(particle_density_scatter)");

    // The convert pipeline: no push constants, two storage images, 4³ groups.
    VkPipelineLayoutCreateInfo cplci{};
    cplci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    cplci.setLayoutCount = 1;
    cplci.pSetLayouts    = &convertDsLayout_;
    check(vkCreatePipelineLayout(d, &cplci, nullptr, &convertPipeLayout_),
          "vkCreatePipelineLayout(particle density convert)");

    VkShaderModuleCreateInfo csm{};
    csm.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    csm.codeSize = sizeof(kParticleDensityConvertCompSpv);
    csm.pCode    = kParticleDensityConvertCompSpv;
    VkShaderModule cmod = VK_NULL_HANDLE;
    check(vkCreateShaderModule(d, &csm, nullptr, &cmod),
          "vkCreateShaderModule(particle_density_convert)");
    VkPipelineShaderStageCreateInfo cstage{};
    cstage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cstage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cstage.module = cmod;
    cstage.pName  = "main";
    VkComputePipelineCreateInfo ccp{};
    ccp.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ccp.stage  = cstage;
    ccp.layout = convertPipeLayout_;
    const VkResult cr = vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &ccp,
                                                 nullptr, &convertPipe_);
    vkDestroyShaderModule(d, cmod, nullptr);
    check(cr, "vkCreateComputePipelines(particle_density_convert)");
}

bool ParticleFieldPass::ensureDensityVolume(State& st, const ParticleField& field) {

    if (st.density.image != VK_NULL_HANDLE) return true;
    if (!createImage3DFn_) return false;

    // The resolution is LATCHED here and never revisited: the image is named by
    // the deferred descriptor sets of every view, so growing it is a structural
    // change, not an allocation. DensityRepr documents this.
    const std::uint32_t res =
            std::max(8u, std::min(256u, field.densityRepr().resolution));

    ensureDensityPipeline();
    st.density = createImage3DFn_(res, res, res, VK_FORMAT_R32_UINT, kDensityUsage,
                                  "particleDensity");
    st.densityLin = createImage3DFn_(res, res, res, VK_FORMAT_R16_SFLOAT,
                                     kDensityLinUsage, "particleDensityLin");
    if (st.density.image == VK_NULL_HANDLE || st.densityLin.image == VK_NULL_HANDLE) {
        destroyImage2D(ctx_.allocator(), ctx_.device(), st.density);
        destroyImage2D(ctx_.allocator(), ctx_.device(), st.densityLin);
        st.density = st.densityLin = Image2D{};
        return false;
    }
    st.densityRes = res;

    VkDescriptorSetLayout layouts[2] = {densityDsLayout_, convertDsLayout_};
    VkDescriptorSet       sets[2]    = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = densityPool_;
    ai.descriptorSetCount = 2;
    ai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(ctx_.device(), &ai, sets) != VK_SUCCESS) {
        // The pool is exactly kMaxDensityFields deep, and prepareFrame refuses
        // to create a volume past that, so this is unreachable rather than a
        // silent cap. Undo the images so the field simply has no density.
        destroyImage2D(ctx_.allocator(), ctx_.device(), st.density);
        destroyImage2D(ctx_.allocator(), ctx_.device(), st.densityLin);
        st.density = st.densityLin = Image2D{};
        st.densityRes = 0;
        st.densitySet = st.convertSet = VK_NULL_HANDLE;
        return false;
    }
    st.densitySet = sets[0];
    st.convertSet = sets[1];

    // Written ONCE, here, outside any frame's record — the image handles never
    // change for the life of the field, so these sets are never rewritten and
    // can never be a VUID-03047 in-flight update.
    VkDescriptorImageInfo ii{};
    ii.imageView   = st.density.view;
    ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo li{};
    li.imageView   = st.densityLin.view;
    li.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w[3]{};
    for (auto& x : w) {
        x.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        x.descriptorCount = 1;
        x.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    w[0].dstSet     = st.densitySet;
    w[0].dstBinding = 0;
    w[0].pImageInfo = &ii;
    w[1].dstSet     = st.convertSet;
    w[1].dstBinding = 0;
    w[1].pImageInfo = &ii;
    w[2].dstSet     = st.convertSet;
    w[2].dstBinding = 1;
    w[2].pImageInfo = &li;
    vkUpdateDescriptorSets(ctx_.device(), 3, w, 0, nullptr);
    return true;
}

ParticleFieldPass::State& ParticleFieldPass::ensureState(const ParticleField& field) {

    auto it = states_.find(&field);
    if (it != states_.end()) return *it->second;

    auto st = std::make_unique<State>();
    st->capacity = field.capacity();
    // weak_from_this() is empty for a stack-allocated field; the sweep below
    // falls back to retire-on-absence in that case. Fields built the documented
    // way (ParticleField::create → make_shared) survive being parked with
    // visible = false, which is what the plan's park-don't-remove rule needs.
    st->owner = const_cast<ParticleField&>(field).weak_from_this();
    st->ownerTracked = !st->owner.expired();

    const VkDeviceSize bytes = VkDeviceSize(st->capacity) * sizeof(ParticlePos);
    st->rendererOwned =
            field.config().ownership == ParticleField::Ownership::Renderer;
    if (st->rendererOwned) {
        // ── NO RING (plan F-C) ──────────────────────────────────────────────
        // Two DEVICE-LOCAL buffers, single-instance, never host-mapped. The ring
        // below exists solely because a HOST memcpy for frame N would otherwise
        // land in a buffer frames N-1 / N-2 are still reading; here the writer
        // is particle_emit.comp, recorded into the same command buffer as every
        // consumer and separated from them by a barrier in recordEmit, so the
        // GPU's own dependency graph does the job three frames of latency were
        // doing. Same argument, verbatim, as 5584d2ab's single interop buffer.
        //
        // No kHostWrite: this memory is never touched by the CPU, so it goes in
        // device-local heap and a 1M-particle field costs 16 MB rather than the
        // 48 MB three host-visible copies would.
        st->devPositions = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                        kPositionUsage, VMA_MEMORY_USAGE_AUTO, 0);
        st->devPrevPositions = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                            kPositionUsage, VMA_MEMORY_USAGE_AUTO, 0);
    }
    for (std::uint32_t s = 0; s < kSlots; ++s) {
        if (!st->rendererOwned) {
            st->positions[s] = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                            kPositionUsage, VMA_MEMORY_USAGE_AUTO, kHostWrite);
        }
        // The counts block stays ringed and host-visible for BOTH modes: it is
        // 16 bytes, and under Ownership::Renderer it is written exactly once per
        // slot (liveCount == capacity, and ParticleField::dataSerial never moves
        // again) rather than once per frame.
        st->counts[s] = createBuffer(ctx_.allocator(), ctx_.device(), sizeof(FieldCountsGpu),
                                     kCountsUsage, VMA_MEMORY_USAGE_AUTO, kHostWrite);
        st->indirect[s] = createBuffer(ctx_.allocator(), ctx_.device(),
                                       sizeof(VkDrawIndirectCommand), kIndirectUsage,
                                       VMA_MEMORY_USAGE_AUTO, kHostWrite);
        st->slotSerial[s] = 0;// fresh allocation holds garbage
    }
    if (st->rendererOwned) ensureEmitPipeline();
    if (field.config().orientations) {
        st->orientations = createBuffer(ctx_.allocator(), ctx_.device(),
                                        VkDeviceSize(st->capacity) * 8u, kOrientationUsage,
                                        VMA_MEMORY_USAGE_AUTO, kHostWrite);
    }

    auto* raw = st.get();
    states_.emplace(&field, std::move(st));
    return *raw;
}

void ParticleFieldPass::ensureDescCapacity(std::uint32_t frame, std::uint32_t count) {

    const std::uint32_t want = std::max(count, 1u);
    if (want <= descCapacity_ && descBufs_[frame].handle != VK_NULL_HANDLE) return;

    const std::uint32_t newCap = std::max(want, descCapacity_ * 2u);
    // Grow EVERY slot, not just this one: the capacity is shared bookkeeping and
    // a half-grown ring would silently under-run the other frame next time.
    for (auto& b : descBufs_) retireOrDestroy(b);
    for (auto& b : descBufs_) {
        b = createBuffer(ctx_.allocator(), ctx_.device(),
                         VkDeviceSize(newCap) * sizeof(FieldDescGpu),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VMA_MEMORY_USAGE_AUTO, kHostWrite);
    }
    descCapacity_ = newCap;
}

void ParticleFieldPass::prepareFrame(std::uint64_t serial, std::uint32_t frame,
                                     const std::vector<Rec>& fields) {

    const std::uint32_t slot = static_cast<std::uint32_t>(serial % kSlots);
    // The slot the PREVIOUS frame filled, which by construction holds the
    // previous frame's positions — so it IS the prevPositions buffer plan
    // §1.2 asks for, at zero cost and with no ordering hazard.
    //
    // The plan writes that buffer as a per-frame vkCmdCopyBuffer of positions
    // "at the head of the frame's particle block, BEFORE any writer". That is
    // the right shape for Ownership::Interop, where ONE device buffer is
    // rewritten by the sim's CUDA copy inside the frame. It is the WRONG shape
    // here: under HostRing the writer is the host, in prepareFrame, which has
    // already run by the time any command is recorded — a copy at the head of
    // the command buffer would capture this frame's positions and every motion
    // vector would be exactly zero. The ring's own depth is what makes the
    // previous state still readable, and a slot is only reused three frames
    // later, i.e. one full frame after the fence that retired it.
    const std::uint32_t prevSlot = (slot + kSlots - 1u) % kSlots;

    descScratch_.clear();
    descScratch_.reserve(fields.size());
    draws_.clear();
    draws_.reserve(fields.size());
    // The bound-volume list is rebuilt from scratch every frame and compared
    // against the previous one at the end: only a genuine change bumps the
    // generation, so a steady-state dust scene never triggers a descriptor
    // rewrite.
    const std::vector<DensityVolumeDesc> prevVols = densityVols_;
    densityVols_.clear();
    densityDispatch_.clear();
    emitDispatch_.clear();
    densityOverflow_ = 0;

    // Reclaim descriptor sets whose last referencing frame has provably
    // retired — the same `R + kFramesInFlight <= S` rule the resource retire
    // queue enforces, and the reason the sweep below queues rather than frees.
    if (!densitySetRetire_.empty() && densityPool_ != VK_NULL_HANDLE) {
        auto it = densitySetRetire_.begin();
        while (it != densitySetRetire_.end()) {
            if (it->serial + impl::kFramesInFlight <= serial) {
                vkFreeDescriptorSets(ctx_.device(), densityPool_, 1, &it->set);
                it = densitySetRetire_.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (const Rec& r : fields) {
        if (!r.field) continue;
        State& st = ensureState(*r.field);
        st.lastSeenSerial = serial;

        const std::uint32_t live = std::min(r.field->liveCount(), st.capacity);

        // Positions + count into THIS frame's slot, version-gated. A static or
        // parked field re-sends nothing; a field the sim advanced re-sends the
        // live prefix only. This is the design's ONLY per-particle CPU cost —
        // and under Ownership::Renderer even that is gone: there are no host
        // positions to send, and the count is capacity, written once per slot
        // and never again (dataSerial only moves on submit/setLiveCount).
        const std::uint64_t want = r.field->dataSerial();
        if (st.slotSerial[slot] != want) {
            st.slotSerial[slot] = want;
            if (!st.rendererOwned && live > 0) {
                uploadHostVisible(ctx_.allocator(), st.positions[slot],
                                  r.field->hostPositions().data(),
                                  VkDeviceSize(live) * sizeof(ParticlePos));
            }
            FieldCountsGpu c{};
            c.liveCount = live;
            uploadHostVisible(ctx_.allocator(), st.counts[slot], &c, sizeof(c));
        }

        // Orientations, once. The serial guard is what makes this write-once in
        // practice as well as by contract.
        if (st.orientations.handle != VK_NULL_HANDLE &&
            st.oriSerial != r.field->orientationSerial() &&
            !r.field->hostOrientations().empty()) {
            st.oriSerial = r.field->orientationSerial();
            uploadHostVisible(ctx_.allocator(), st.orientations,
                              r.field->hostOrientations().data(),
                              VkDeviceSize(st.capacity) * 8u);
        }

        // The draw command. Three of its four words are CPU-known constants;
        // instanceCount is left at 0 and filled on the device by recordCounts.
        // firstInstance is 0 by design — particlefield_gbuf.vert reads
        // gl_InstanceIndex as the PARTICLE index and takes its DrawInfo index
        // from a push constant instead.
        VkDrawIndirectCommand cmd{};
        cmd.vertexCount   = r.proxyVertexCount;
        cmd.instanceCount = 0u;
        cmd.firstVertex   = 0u;
        cmd.firstInstance = 0u;
        uploadHostVisible(ctx_.allocator(), st.indirect[slot], &cmd, sizeof(cmd));

        const bool prevValid = st.slotSerial[prevSlot] != 0;

        FieldDescGpu d{};
        const auto& world = r.field->matrixWorld->elements;
        std::memcpy(d.world, world.data(), sizeof(d.world));
        if (st.rendererOwned) {
            // Single instance, both of them, and prevPositions is a REAL
            // previous state rather than an aliased ring slot: the emit
            // dispatch writes f(t) and f(t - dt) into the two buffers from the
            // same thread, earlier in this same command buffer.
            d.posAddr     = st.devPositions.address;
            d.prevPosAddr = st.devPrevPositions.address;
        } else {
            d.posAddr = st.positions[slot].address;
            // A slot that was never filled holds garbage, so the first two
            // frames of a field's life reproject onto themselves (zero motion)
            // rather than streaking in from uninitialised memory.
            d.prevPosAddr = prevValid ? st.positions[prevSlot].address : d.posAddr;
        }
        d.oriAddr     = st.orientations.address;// 0 when no orientation buffer
        d.attrAddr    = 0;                      // Config::attributes, phase 4
        d.countAddr   = st.counts[slot].address;
        d.capacity    = st.capacity;
        d.entryIndex  = r.entryIndex;
        const auto& cfg = r.field->config();
        d.uniformRadius = cfg.uniformRadius;
        d.wSemantic     = static_cast<std::uint32_t>(cfg.wSemantic);
        d.reprMask      = (r.field->meshRepr().enabled ? 1u : 0u) |
                     (r.field->billboardRepr().enabled ? 2u : 0u) |
                     (r.field->densityRepr().enabled ? 4u : 0u) |
                     (r.field->tracedRepr().enabled ? 8u : 0u);
        d.classId = r.classId;
        descScratch_.push_back(d);

        // ── The device emitter (plan F-C) ───────────────────────────────────
        // O(1) host work per field: pack a 128 B push block. Nothing is
        // version-gated because there is nothing to skip — the block is built
        // fresh every frame and costs less than the branch that would decide
        // not to. A PARKED Renderer field (setLiveCount(0)) records no dispatch
        // at all, which is the one legitimate way to stop an emitter.
        if (st.rendererOwned && live > 0 && emitPipe_ != VK_NULL_HANDLE) {
            const auto& ep = r.field->emitter();
            EmitPc pc{};
            pc.posAddr     = st.devPositions.address;
            pc.prevPosAddr = st.devPrevPositions.address;
            pc.spawnCenter[0] = ep.spawnCenter.x;
            pc.spawnCenter[1] = ep.spawnCenter.y;
            pc.spawnCenter[2] = ep.spawnCenter.z;
            pc.lifetime       = std::max(ep.lifetime, 1e-3f);
            pc.spawnHalf[0]   = ep.spawnHalfExtent.x;
            pc.spawnHalf[1]   = ep.spawnHalfExtent.y;
            pc.spawnHalf[2]   = ep.spawnHalfExtent.z;
            pc.lifetimeJitter = std::max(0.f, std::min(1.f, ep.lifetimeJitter));
            // wind is summed into velocity HERE, not in the shader: they add
            // linearly and the shader has no use for the distinction, so the
            // API keeps two authorable knobs and the GPU sees one vector.
            pc.velocity[0] = ep.velocity.x + ep.wind.x;
            pc.velocity[1] = ep.velocity.y + ep.wind.y;
            pc.velocity[2] = ep.velocity.z + ep.wind.z;
            pc.speedSpread = std::max(ep.speedSpread, 0.f);
            pc.accel[0]    = ep.accel.x;
            pc.accel[1]    = ep.accel.y;
            pc.accel[2]    = ep.accel.z;
            pc.duty        = std::max(1e-3f, std::min(1.f, ep.dutyCycle));
            pc.driftAmp    = ep.driftAmplitude;
            pc.driftFreq   = ep.driftFrequency;
            pc.driftGrowth = std::max(0.f, std::min(1.f, ep.driftGrowth));
            pc.size        = std::max(ep.size, 0.f);
            pc.sizeJitter  = std::max(0.f, std::min(1.f, ep.sizeJitter));
            pc.time        = r.field->emitterTime();
            // A negative dt would put prevPositions in the FUTURE and reverse
            // every motion vector — the defect plan F2 says numbers will not
            // catch. Floored on both sides of the API for that reason.
            pc.dt         = std::max(r.field->emitterDt(), 0.f);
            pc.capacity   = st.capacity;
            pc.seed       = ep.seed;
            pc.driftScale = std::max(ep.driftScale, 0.f);

            EmitDispatch ed{};
            ed.groups = (st.capacity + kEmitLocalSize - 1u) / kEmitLocalSize;
            std::memcpy(ed.pc, &pc, sizeof(pc));
            emitDispatch_.push_back(ed);
        }

        // ── Density volume (plan §3.3) ──────────────────────────────────────
        // A field contributes density only while it has live particles: a
        // parked field would otherwise keep a zero volume bound, keep the
        // froxel passes forced on, and keep heteroActive raised — three
        // wrong answers to "is there dust in this scene".
        const auto& dr = r.field->densityRepr();
        if (dr.enabled && std::min(r.field->liveCount(), st.capacity) > 0u) {
            if (densityVols_.size() >= kMaxDensityFields) {
                ++densityOverflow_;
            } else if (ensureDensityVolume(st, *r.field)) {
                // Half-extent is a divisor; a degenerate axis would produce an
                // infinity that poisons every sample in the volume.
                const float hx = std::max(dr.halfExtent.x, 1e-4f);
                const float hy = std::max(dr.halfExtent.y, 1e-4f);
                const float hz = std::max(dr.halfExtent.z, 1e-4f);

                DensityVolumeDesc v{};
                v.view          = st.density.view;
                v.boxMin[0]     = dr.center.x - hx;
                v.boxMin[1]     = dr.center.y - hy;
                v.boxMin[2]     = dr.center.z - hz;
                v.resolution    = float(st.densityRes);
                v.boxInvSize[0] = 1.f / (2.f * hx);
                v.boxInvSize[1] = 1.f / (2.f * hy);
                v.boxInvSize[2] = 1.f / (2.f * hz);
                v.linView       = st.densityLin.view;
                // Per-field medium + emission. Every one of these travels with
                // the volume now (plans/particle-atmosphere.md F-A); the
                // first-field-wins fill that used to sit here, and the wart
                // comment in the header that apologised for it, are both gone.
                v.albedo[0]  = dr.albedo.r;
                v.albedo[1]  = dr.albedo.g;
                v.albedo[2]  = dr.albedo.b;
                v.anisotropy = dr.anisotropy;
                v.emission[0] = std::max(dr.emissiveIntensity, 0.f);
                v.emission[1] = dr.tempBottomK;
                v.emission[2] = dr.tempTopK;
                // The shader raises the in-box height fraction to this power.
                // Clamped HERE rather than in the march: pow(0, 0) is undefined
                // in GLSL, and a floor costs nothing on the host but two
                // instructions per step per volume in the shader.
                v.emission[3] = std::max(dr.tempFalloff, 1e-3f);
                densityVols_.push_back(v);

                DensityDispatch dd{};
                dd.set        = st.densitySet;
                dd.image      = st.density.image;
                dd.linImage   = st.densityLin.image;
                dd.convertSet = st.convertSet;
                dd.res   = st.densityRes;
                dd.groups = (st.capacity + kScatterLocalSize - 1u) / kScatterLocalSize;
                std::memcpy(dd.world, d.world, sizeof(dd.world));
                dd.posAddr   = d.posAddr;
                dd.countAddr = d.countAddr;
                std::memcpy(dd.boxMin, v.boxMin, sizeof(dd.boxMin));
                std::memcpy(dd.boxInvSize, v.boxInvSize, sizeof(dd.boxInvSize));
                dd.capacity = st.capacity;
                // The scatter adds an INTEGER, so the host does the one float
                // multiply that turns σ per particle into fixed-point units.
                dd.sigmaFixed = std::max(dr.sigmaPerParticle, 0.f) * kDensityFixedScale;
                densityDispatch_.push_back(dd);
            }
        }

        DrawState ds{};
        ds.field       = r.field;
        ds.indirect    = st.indirect[slot].handle;
        ds.counts      = st.counts[slot].handle;
        ds.posAddr     = d.posAddr;
        ds.prevPosAddr = d.prevPosAddr;
        ds.oriAddr     = d.oriAddr;
        ds.vertexCount = r.proxyVertexCount;
        draws_.push_back(ds);
    }

    descCount_ = static_cast<std::uint32_t>(descScratch_.size());
    ensureDescCapacity(frame, descCount_);
    if (descCount_ > 0) {
        uploadHostVisible(ctx_.allocator(), descBufs_[frame], descScratch_.data(),
                          VkDeviceSize(descCount_) * sizeof(FieldDescGpu));
    }

    // Sweep. A field whose owning shared_ptr died is gone for good; one merely
    // absent from this frame's entry list may only be parked (visible = false
    // hides it from traverseVisible), so a tracked field is kept.
    for (auto it = states_.begin(); it != states_.end();) {
        State& st = *it->second;
        const bool absent = st.lastSeenSerial != serial;
        const bool dead = st.ownerTracked ? st.owner.expired() : absent;
        if (absent && dead) {
            for (auto& b : st.positions) retireOrDestroy(b);
            retireOrDestroy(st.devPositions);
            retireOrDestroy(st.devPrevPositions);
            for (auto& b : st.counts) retireOrDestroy(b);
            for (auto& b : st.indirect) retireOrDestroy(b);
            retireOrDestroy(st.orientations);
            // The set goes back to the pool by hand — the pool is exactly
            // kMaxDensityFields deep, so leaking a set would cap the scene at
            // four dust fields for the life of the process rather than four at
            // a time. Safe here for the same reason the buffers are: the state
            // is only swept once no in-flight frame can still name it.
            if (st.densitySet != VK_NULL_HANDLE) {
                densitySetRetire_.push_back({st.densitySet, serial});
                st.densitySet = VK_NULL_HANDLE;
            }
            if (st.convertSet != VK_NULL_HANDLE) {
                densitySetRetire_.push_back({st.convertSet, serial});
                st.convertSet = VK_NULL_HANDLE;
            }
            retireOrDestroy(st.density);
            retireOrDestroy(st.densityLin);
            it = states_.erase(it);
        } else {
            ++it;
        }
    }

    // Generation: bumped only when the BOUND LIST actually changed. The
    // renderer's descriptor sets name these views, so an unchanged list means
    // an unchanged set and no write at all.
    bool volsChanged = prevVols.size() != densityVols_.size();
    for (std::size_t i = 0; !volsChanged && i < densityVols_.size(); ++i) {
        volsChanged = prevVols[i].view != densityVols_[i].view;
    }
    if (volsChanged) ++densityGen_;
}

bool ParticleFieldPass::densityVolumeFor(const ParticleField& field,
                                         VkImage& image, std::uint32_t& res) const {

    const auto it = states_.find(&field);
    if (it == states_.end() || it->second->density.image == VK_NULL_HANDLE) return false;
    image = it->second->density.image;
    res   = it->second->densityRes;
    return true;
}

// The whole per-frame cost of Ownership::Renderer, and the whole per-frame
// TRAFFIC too: one vkCmdPushConstants + one vkCmdDispatch per field, over a
// domain the CPU has known since the field was created. Nothing is uploaded,
// nothing is copied, no descriptor is written and no particle is touched by the
// host — which is the thesis of the mode stated as a command stream.
void ParticleFieldPass::recordEmit(VkCommandBuffer cb) {

    if (emitDispatch_.empty() || emitPipe_ == VK_NULL_HANDLE) return;

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, emitPipe_);
    for (const EmitDispatch& ed : emitDispatch_) {
        vkCmdPushConstants(cb, emitPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<std::uint32_t>(sizeof(EmitPc)), ed.pc);
        vkCmdDispatch(cb, ed.groups, 1, 1);
    }

    // ONE barrier for every field, covering every consumer in this frame:
    //   COMPUTE — the density scatter, recorded next;
    //   VERTEX  — particlefield_gbuf.vert, which pulls positions AND
    //             prevPositions through buffer_reference in every view's
    //             G-buffer pass, all of which record later into this same
    //             command buffer;
    //   ACCELERATION_STRUCTURE_BUILD — nothing reads it yet (the procedural
    //             AABB BLAS is parent phase 5, deferred), but a refit that
    //             consumed these positions would sit in this same window, and
    //             the stage is free to name now rather than a hazard to
    //             rediscover later.
    // Per-field barriers would serialise dispatches the scheduler is happy to
    // overlap, and every consumer is downstream of ALL of them anyway.
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &mb;
    vkCmdPipelineBarrier2(cb, &dep);
}

void ParticleFieldPass::recordDensityScatter(VkCommandBuffer cb) {

    if (densityDispatch_.empty()) return;

    // 1. Zero every volume. UNDEFINED → GENERAL because the previous contents
    //    are about to be overwritten wholesale — this is a discard, which is
    //    exactly what an undefined old layout means, and it also covers the
    //    image's very first frame with no extra bookkeeping.
    std::vector<VkImageMemoryBarrier2> pre;
    pre.reserve(densityDispatch_.size() * 2u);
    for (const DensityDispatch& dd : densityDispatch_) {
        VkImageMemoryBarrier2 ib{};
        ib.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        ib.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        ib.srcAccessMask = 0;
        ib.dstStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        ib.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        ib.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        ib.image         = dd.image;
        ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre.push_back(ib);
        // The r16f mirror: same discard-to-GENERAL, but its writer is the
        // convert dispatch, not the clear — every voxel is overwritten, so it
        // needs no clear of its own.
        ib.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        ib.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        ib.image         = dd.linImage;
        pre.push_back(ib);
    }
    VkDependencyInfo preDep{};
    preDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    preDep.imageMemoryBarrierCount = static_cast<std::uint32_t>(pre.size());
    preDep.pImageMemoryBarriers    = pre.data();
    vkCmdPipelineBarrier2(cb, &preDep);

    VkClearColorValue zero{};
    zero.uint32[0] = 0;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (const DensityDispatch& dd : densityDispatch_) {
        vkCmdClearColorImage(cb, dd.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }

    // 2. Clear → atomic accumulate.
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &mb;
    vkCmdPipelineBarrier2(cb, &dep);

    // 3. One dispatch per field, over the CPU-CONSTANT capacity domain.
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, densityPipe_);
    for (const DensityDispatch& dd : densityDispatch_) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, densityPipeLayout_,
                                0, 1, &dd.set, 0, nullptr);
        DensityScatterPc pc{};
        std::memcpy(pc.world, dd.world, sizeof(pc.world));
        pc.posAddr   = dd.posAddr;
        pc.countAddr = dd.countAddr;
        std::memcpy(pc.boxMin, dd.boxMin, sizeof(pc.boxMin));
        std::memcpy(pc.boxInvSize, dd.boxInvSize, sizeof(pc.boxInvSize));
        pc.res        = dd.res;
        pc.capacity   = dd.capacity;
        pc.sigmaFixed = dd.sigmaFixed;
        vkCmdPushConstants(cb, densityPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(cb, dd.groups, 1, 1);
    }

    // 4. Scatter → convert. The atomics must have landed before the convert
    //    reads the fixed-point volume.
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    vkCmdPipelineBarrier2(cb, &dep);

    // 5. r32ui → r16f, one dispatch per field — hardware trilinear for the
    //    deferred shade's per-pixel dust march is the whole point of the copy.
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, convertPipe_);
    for (const DensityDispatch& dd : densityDispatch_) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, convertPipeLayout_,
                                0, 1, &dd.convertSet, 0, nullptr);
        const std::uint32_t g = (dd.res + kConvertLocalSize - 1u) / kConvertLocalSize;
        vkCmdDispatch(cb, g, g, g);
    }

    // 6. Both volumes → every read this frame: the froxel passes' manual
    //    trilinear on the uint volume (all views — they ride this one write)
    //    and the shade's hardware trilinear on the r16f mirror.
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    vkCmdPipelineBarrier2(cb, &dep);
}

void ParticleFieldPass::recordCounts(VkCommandBuffer cb) {

    if (draws_.empty()) return;

    bool any = false;
    for (const DrawState& d : draws_) {
        if (d.vertexCount == 0u || d.indirect == VK_NULL_HANDLE) continue;
        VkBufferCopy region{};
        region.srcOffset = 0;// FieldCountsGpu::liveCount
        region.dstOffset = kInstanceCountOffset;
        region.size      = sizeof(std::uint32_t);
        vkCmdCopyBuffer(cb, d.counts, d.indirect, 1, &region);
        any = true;
    }
    if (!any) return;

    // One barrier for every field: the copies are independent, the consumer is
    // the same command-processor stage, and a per-field barrier would serialise
    // what the copy engine is happy to overlap.
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &mb;
    vkCmdPipelineBarrier2(cb, &dep);
}
