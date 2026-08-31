#include "ParticleFieldPass.hpp"

#include "VulkanContext.hpp"

#include "threepp/objects/ParticleField.hpp"

#include "threepp/renderers/vulkan/shaders/particle_density_convert.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/particle_density_scatter.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/particle_emit.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/particle_height_bake.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_transmit.comp.spv.h"

#include <algorithm>
#include <cstdio>
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

    // F6: an Interop field's ring slot is the DESTINATION of the head-of-frame
    // snapshot, so it adds TRANSFER_DST — and it is device-local (no kHostWrite):
    // nothing on the host ever writes it.
    constexpr VkBufferUsageFlags kPositionUsageInterop =
            kPositionUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // F6: the exported allocation. TRANSFER_SRC is the only usage the renderer
    // itself needs — it is copied, never bound — and STORAGE is there so a
    // future phase can point a shader at it without re-exporting. No
    // SHADER_DEVICE_ADDRESS: createExternalBuffer allocates its dedicated
    // memory without VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, and a buffer that
    // asks for an address on memory that was not allocated for one is invalid.
    constexpr VkBufferUsageFlags kExternalPositionUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // Counts: TRANSFER_SRC because the liveCount → instanceCount route is a
    // 4-byte device copy into a VkDrawIndirectCommand; INDIRECT so a
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

    // Attributes (Config::attributes): one vec4 per particle, read as a
    // buffer_reference by the billboard vertex stage. The interop ring adds
    // TRANSFER_DST because the head-of-frame snapshot writes it, exactly as the
    // position ring does — the two are deliberately the same shape.
    constexpr VkBufferUsageFlags kAttributeUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    constexpr VkBufferUsageFlags kAttributeUsageInterop =
            kAttributeUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

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
    constexpr std::uint32_t kTransmitLocalSize = 64;// == particlefield_transmit.comp

    // R8: the transmittance buffer. STORAGE + an address, because the compute
    // pass writes it through a buffer_reference and the billboard VERTEX stage
    // reads it through one — no descriptor on either side, which is the
    // property the whole billboard pass is built on.
    constexpr VkBufferUsageFlags kTransmitUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // MUST mirror the push block in particlefield_transmit.comp under scalar
    // layout. 120 B — inside the 128 B every Vulkan implementation guarantees.
    // The two addresses lead so the block's 8-byte alignment is satisfied at
    // offset 0 and every float after is naturally 4-aligned, which keeps MSVC's
    // layout and GLSL's scalar layout byte-identical.
    //
    // camWorld is the ONE per-view member. recordTransmittance patches it (and
    // sunDirWorld, which is view-independent but sits beside it) into the
    // prebuilt bytes rather than rebuilding the block, so re-dispatching for a
    // second view costs two memcpys and no host arithmetic at all.
    struct TransmitPc {
        VkDeviceAddress posAddr;      //   0
        VkDeviceAddress outAddr;      //   8
        float           model[12];    //  16  ROWS of the affine field->world
        float           boxMin[3];    //  64
        float           boxInvSize[3];//  76
        float           camWorld[3];  //  88
        float           sunDirWorld[3];// 100
        std::uint32_t   capacity;     // 112
        std::uint32_t   flags;        // 116
    };                                // 120
    static_assert(sizeof(TransmitPc) == 120,
                  "TransmitPc drifted from particlefield_transmit.comp");
    // Mirrored in the shader.
    constexpr std::uint32_t kTransCamBit = 1u;
    constexpr std::uint32_t kTransSunBit = 2u;

    // MUST mirror the push block in particle_density_scatter.comp under scalar
    // layout. 120 B — inside the 128 B every Vulkan implementation guarantees.
    // The explicit tail pad keeps the scalar-layout size and MSVC's struct
    // size (8-byte-aligned, uint64 member) equal by definition rather than by
    // accident.
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
    // member for member and offset for offset. Exactly 128 B — the ceiling
    // every Vulkan implementation guarantees — so the trailing reserve is
    // spelled out: adding a member here without removing one makes a device
    // that cannot create the pipeline.
    //
    // The two uint64s sit first so the block's 8-byte alignment is satisfied at
    // offset 0 and every float after them is naturally 4-aligned, keeping
    // MSVC's layout and GLSL's scalar layout byte-identical.
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
        // Slot count in bits 0..30, follow flag in bit 31 — see the shader.
        std::uint32_t capacityAndFollow;// 108
        std::uint32_t seed;           // 112
        float driftScale;             // 116
        // ── F4/F5 ───────────────────────────────────────────────────────────
        // The follow centre, height map and lifecycle block live behind this
        // address (EmitAuxGpu) — the push block itself has no room left.
        // A field that neither follows nor rests pushes 0.
        VkDeviceAddress auxAddr;      // 120  -> EmitAuxGpu, or 0
    };
    static_assert(sizeof(EmitPc) == 128, "particle_emit push-constant drift");
    static_assert(offsetof(EmitPc, seed) == 112, "particle_emit push-constant layout drift");
    static_assert(offsetof(EmitPc, auxAddr) == 120, "EmitPc::auxAddr must land at 120");

    constexpr std::uint32_t kBakeLocalSize = 8;// == particle_height_bake.comp (8x8)

    // MUST mirror the push block in particle_height_bake.comp under scalar
    // layout. Well inside the guaranteed 128 B; the explicit tail pad keeps
    // the MSVC layout and the GLSL one byte-identical.
    struct BakePc {
        VkDeviceAddress dstAddr;//  0
        float           originX;//  8  WORLD min corner of the footprint
        float           originZ;// 12
        float           cell;   // 16  metres per texel
        std::uint32_t   res;    // 20
        float           topY;   // 24  WORLD y the rays start from
        float           depth;  // 28
        float           missY;  // 32  FIELD-LOCAL "nothing here"
        float           yOffset;// 36  the field's world Y
    };
    static_assert(sizeof(BakePc) == 40, "particle_height_bake push-constant drift");

    // Clamp bounds for EmitterParams::Surface::resolution. The floor keeps a
    // degenerate map from making surfaceAt's texel arithmetic meaningless; the
    // ceiling caps the cost at 4 MB per ring slot.
    constexpr std::uint32_t kBakeResMin = 16;
    constexpr std::uint32_t kBakeResMax = 1024;

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
    // F6: exported allocations of fields swept too recently for the
    // frame-serial rule to have freed them yet.
    for (auto& r : extRetire_) vulkan::destroyExternalBuffer(ctx_.device(), r.buf);
    extRetire_.clear();
    for (auto& b : descBufs_) destroyBuffer(ctx_.allocator(), b);
    for (auto& b : bbParamBufs_) destroyBuffer(ctx_.allocator(), b);
    for (auto& b : bbViewBufs_)  destroyBuffer(ctx_.allocator(), b);
    for (auto& b : auxBufs_)     destroyBuffer(ctx_.allocator(), b);
    for (auto& b : transBufs_)   destroyBuffer(ctx_.allocator(), b);

    const VkDevice d = ctx_.device();
    if (transPipe_)         vkDestroyPipeline(d, transPipe_, nullptr);
    if (transPipeLayout_)   vkDestroyPipelineLayout(d, transPipeLayout_, nullptr);
    if (transDsLayout_)     vkDestroyDescriptorSetLayout(d, transDsLayout_, nullptr);
    if (transSampler_)      vkDestroySampler(d, transSampler_, nullptr);
    if (emitPipe_)          vkDestroyPipeline(d, emitPipe_, nullptr);
    if (emitPipeLayout_)    vkDestroyPipelineLayout(d, emitPipeLayout_, nullptr);
    if (bakePipe_)          vkDestroyPipeline(d, bakePipe_, nullptr);
    if (bakePipeLayout_)    vkDestroyPipelineLayout(d, bakePipeLayout_, nullptr);
    if (bakePool_)          vkDestroyDescriptorPool(d, bakePool_, nullptr);
    if (bakeDsLayout_)      vkDestroyDescriptorSetLayout(d, bakeDsLayout_, nullptr);
    if (densityPipe_)       vkDestroyPipeline(d, densityPipe_, nullptr);
    if (densityPipeLayout_) vkDestroyPipelineLayout(d, densityPipeLayout_, nullptr);
    if (convertPipe_)       vkDestroyPipeline(d, convertPipe_, nullptr);
    if (convertPipeLayout_) vkDestroyPipelineLayout(d, convertPipeLayout_, nullptr);
    destroyBuffer(ctx_.allocator(), densityMajorants_);
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
    for (auto& b : st.bbIndirect) destroyBuffer(ctx_.allocator(), b);
    destroyBuffer(ctx_.allocator(), st.orientations);
    for (auto& b : st.attributes) destroyBuffer(ctx_.allocator(), b);
    // The exported allocation is outside VMA (dedicated + export info), so it
    // has its own destroy — and on Windows that call is also what closes the NT
    // handle we handed the importer, which CUDA duplicated on import.
    destroyExternalBuffer(ctx_.device(), st.posExt);
    destroyExternalBuffer(ctx_.device(), st.attrExt);
    for (auto& b : st.heights) destroyBuffer(ctx_.allocator(), b);
    destroyImage2D(ctx_.allocator(), ctx_.device(), st.density);
    destroyImage2D(ctx_.allocator(), ctx_.device(), st.densityLin);
}

// The emitter pipeline: one compute stage, zero descriptor sets, one 128 B
// push range. With no descriptor set there is no set to allocate, no pool to
// size, no handle to go stale and nothing that could ever be written while a
// frame that names it is in flight (VUID-03047).
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

    // Sized for kMaxDensityFields fields, each holding up to THREE sets: the
    // scatter's (1 storage image), the convert's (2 storage images + the shared
    // majorant SSBO) and — only for a field whose sprites march it (R8) — the
    // transmittance prepass's (1 sampled image). A field past that gets no
    // volume, which densityOverflowCount reports.
    VkDescriptorPoolSize ps[3]{};
    ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[0].descriptorCount = kMaxDensityFields * 3u;
    ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[1].descriptorCount = kMaxDensityFields;
    ps[2].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[2].descriptorCount = kMaxDensityFields;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // FREE_DESCRIPTOR_SET: a destroyed field returns its sets to the pool. The
    // pool is exactly kMaxDensityFields fields deep, so without this a scene
    // that created and dropped four dust fields could never have a fifth.
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = kMaxDensityFields * 3u;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes    = ps;
    check(vkCreateDescriptorPool(d, &dpci, nullptr, &densityPool_),
          "vkCreateDescriptorPool(particle density)");

    // The majorant buffer. Allocated here rather than per field because it is
    // ONE array indexed by the frame's volume slot, and because the convert
    // sets that name it are written once per field and never rewritten — a
    // buffer that could be reallocated would make them stale.
    densityMajorants_ = createBuffer(
            ctx_.allocator(), ctx_.device(),
            VkDeviceSize(kMaxDensityFields) * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO, 0);

    // Convert set layout: binding 0 = r32ui src, binding 1 = r16f dst,
    // binding 2 = the majorant array.
    VkDescriptorSetLayoutBinding cb[3]{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        cb[i].binding         = i;
        cb[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        cb[i].descriptorCount = 1;
        cb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    cb[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorSetLayoutCreateInfo clci{};
    clci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    clci.bindingCount = 3;
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

    // The convert pipeline: two storage images, the majorant SSBO, 4³ groups,
    // and ONE push constant — the volume's slot, which is the only thing about
    // the dispatch that is not already in its set.
    VkPushConstantRange cpcr{};
    cpcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    cpcr.offset     = 0;
    cpcr.size       = sizeof(std::uint32_t);
    VkPipelineLayoutCreateInfo cplci{};
    cplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    cplci.setLayoutCount         = 1;
    cplci.pSetLayouts            = &convertDsLayout_;
    cplci.pushConstantRangeCount = 1;
    cplci.pPushConstantRanges    = &cpcr;
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

// ── R8: the transmittance prepass's pipeline ────────────────────────────────
// Created on the first frame a field actually marches, so a scene with dust but
// no volumetric sprites — every scene in the tree before this feature — creates
// no sampler, no layout and no pipeline. ONE binding, the r16f mirror, for the
// reason the billboard pass had one until this pass took it over: an image is
// the only thing that cannot be reached by device address. Everything else, the
// positions and the output included, rides the 120 B push block.
bool ParticleFieldPass::ensureTransmittancePipeline() {

    if (transPipe_ != VK_NULL_HANDLE) return true;
    const VkDevice d = ctx_.device();

    // LINEAR + CLAMP_TO_EDGE, matching the sampler the vertex stage marched
    // with (textureSamplerIsoClamp_). The filtering is load-bearing: a nearest
    // sample of a 160³ lattice from four million positions reads as blocks, and
    // the whole thesis of the feature is that the low-frequency half survives a
    // coarse grid. CLAMP is belt-and-braces — the march skips samples outside
    // the box rather than clamping them, which is the rule the deferred march
    // states and the reason the border mode is never actually exercised.
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = 0.f;
    if (vkCreateSampler(d, &sci, nullptr, &transSampler_) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 1;
    dlci.pBindings    = &b;
    check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &transDsLayout_),
          "vkCreateDescriptorSetLayout(particlefield transmit)");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(TransmitPc);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &transDsLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    check(vkCreatePipelineLayout(d, &plci, nullptr, &transPipeLayout_),
          "vkCreatePipelineLayout(particlefield transmit)");

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kParticleFieldTransmitCompSpv);
    smci.pCode    = kParticleFieldTransmitCompSpv;
    VkShaderModule mod = VK_NULL_HANDLE;
    check(vkCreateShaderModule(d, &smci, nullptr, &mod),
          "vkCreateShaderModule(particlefield_transmit)");
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = transPipeLayout_;
    const VkResult r = vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci,
                                                nullptr, &transPipe_);
    vkDestroyShaderModule(d, mod, nullptr);
    check(r, "vkCreateComputePipelines(particlefield_transmit)");
    return transPipe_ != VK_NULL_HANDLE;
}

// R9's ring, sized in SLOTS. Same growth discipline as ensureBbParamCapacity
// and the same reason for growing EVERY frame-in-flight at once: the capacity
// is shared bookkeeping, and a half-grown ring under-runs the other frame the
// next time round. Called from prepareFrame — the post-fence, pre-record
// window — so the slot being (re)allocated is provably not one an in-flight
// frame is reading, and the old buffers go through the renderer's frame-serial
// retire queue rather than being destroyed under a frame that still names them.
void ParticleFieldPass::ensureTransCapacity(std::uint32_t count) {

    const std::uint32_t want = std::max(count, 1u);
    if (want <= transCapacity_ && transBufs_[0].handle != VK_NULL_HANDLE) return;

    const std::uint32_t newCap = std::max(want, transCapacity_ * 2u);
    for (auto& b : transBufs_) retireOrDestroy(b);
    for (auto& b : transBufs_) {
        b = createBuffer(ctx_.allocator(), ctx_.device(),
                         VkDeviceSize(newCap) * sizeof(std::uint32_t),
                         kTransmitUsage, VMA_MEMORY_USAGE_AUTO, 0);
    }
    transCapacity_ = newCap;
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
    VkDescriptorBufferInfo mi{};
    mi.buffer = densityMajorants_.handle;
    mi.offset = 0;
    mi.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w[4]{};
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
    w[3].dstSet          = st.convertSet;
    w[3].dstBinding      = 2;
    w[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[3].pBufferInfo     = &mi;
    vkUpdateDescriptorSets(ctx_.device(), 4, w, 0, nullptr);
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
        // ── No ring (plan F-C) ──────────────────────────────────────────────
        // Two device-local buffers, single-instance, never host-mapped. The
        // ring below exists solely because a host memcpy for frame N would
        // otherwise land in a buffer frames N-1 / N-2 are still reading; here
        // the writer is particle_emit.comp, recorded into the same command
        // buffer as every consumer and separated from them by a barrier in
        // recordEmit, so no ring is needed.
        //
        // No kHostWrite: this memory is never touched by the CPU, so it goes in
        // device-local heap and a 1M-particle field costs 16 MB rather than the
        // 48 MB three host-visible copies would.
        st->devPositions = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                        kPositionUsage, VMA_MEMORY_USAGE_AUTO, 0);
        st->devPrevPositions = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                            kPositionUsage, VMA_MEMORY_USAGE_AUTO, 0);
    }
    // ── F6: Ownership::Interop ──────────────────────────────────────────────
    // The exported allocation, and the gate. A device with no external-memory
    // extension cannot export or be imported by CUDA, so the field falls back
    // to a HostRing field in every respect but its declared mode, a warning is
    // printed (a field that renders nothing and says nothing would be worse),
    // and its own submit() stops throwing so the caller's pull leg can feed
    // it.
    if (field.config().ownership == ParticleField::Ownership::Interop) {
        if (ctx_.externalMemorySupported()) {
            st->interopOwned = true;
            st->posExt = vulkan::createExternalBuffer(ctx_.physicalDevice(), ctx_.device(),
                                                      bytes, kExternalPositionUsage);
            // The attribute allocation is exported HERE, beside the positions,
            // rather than at enableInterop — so the two are created together,
            // succeed together, and cannot diverge (plan R2).
            if (field.config().attributes) {
                st->attrExt = vulkan::createExternalBuffer(ctx_.physicalDevice(), ctx_.device(),
                                                           bytes, kExternalPositionUsage);
            }
        } else {
            std::fprintf(stderr,
                         "[ParticleField] Ownership::Interop asked for on a device with no "
                         "external-memory extension (VK_KHR_external_memory_win32/fd): no "
                         "buffer can be exported and no device-to-device copy is possible. "
                         "Falling back to the host ring — feed this field with submit() "
                         "(ParticleField::hostFallback() is now true).\n");
            const_cast<ParticleField&>(field).setHostFallback();
        }
    }

    for (std::uint32_t s = 0; s < kSlots; ++s) {
        if (st->interopOwned) {
            // Device-local: the snapshot's destination, never host-mapped.
            st->positions[s] = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                            kPositionUsageInterop, VMA_MEMORY_USAGE_AUTO, 0);
        } else if (!st->rendererOwned) {
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
        st->slotFill[s]   = 0;// ...so no slot is anybody's previous step yet
    }
    if (st->rendererOwned) ensureEmitPipeline();
    if (field.config().orientations) {
        st->orientations = createBuffer(ctx_.allocator(), ctx_.device(),
                                        VkDeviceSize(st->capacity) * 8u, kOrientationUsage,
                                        VMA_MEMORY_USAGE_AUTO, kHostWrite);
    }
    // Attributes. A RING only where one is needed: the interop leg receives a
    // fresh set of colours every frame, so slot N must not be the buffer frames
    // N-1/N-2 are reading. Everywhere else the contract is write-once, and
    // slot 0 is the whole buffer.
    if (field.config().attributes) {
        const std::uint32_t slots = st->interopOwned ? kSlots : 1u;
        for (std::uint32_t s = 0; s < slots; ++s) {
            st->attributes[s] = createBuffer(
                    ctx_.allocator(), ctx_.device(), bytes,
                    st->interopOwned ? kAttributeUsageInterop : kAttributeUsage,
                    VMA_MEMORY_USAGE_AUTO, st->interopOwned ? 0 : kHostWrite);
        }
    }

    auto* raw = st.get();
    states_.emplace(&field, std::move(st));
    return *raw;
}

// ── F6: export one field's positions and arm its per-frame copy ─────────────
// Unlike enableSoftBodyInterop this does not drain the device, and safely so:
// that call swaps a buffer that in-flight frames' dispatches are already
// reading and rewrites the descriptor sets naming it. Here the exported
// allocation is created with the field's state, before anything can name it,
// and is never swapped — the shaders read the snapshot ring, whose addresses
// are republished every frame anyway. Nothing in flight is invalidated, so
// there is nothing to wait for.
ParticleFieldPass::InteropExport
ParticleFieldPass::enableInterop(ParticleField& field, std::function<void()> deviceCopy) {

    State& st = ensureState(field);
    if (field.config().ownership != ParticleField::Ownership::Interop) {
        std::fprintf(stderr,
                     "[ParticleField] enableParticleFieldInterop on a field that was not "
                     "created with Ownership::Interop — ignored. Ownership is fixed at "
                     "create, like capacity.\n");
        return {};
    }
    // The fallback case already printed why, and hostFallback() is set: the
    // caller sees a null handle and keeps its host path.
    if (!st.interopOwned) return {};

    st.deviceCopy = std::move(deviceCopy);
    // takeOsHandle, not .osHandle: on POSIX the exported fd's ownership passes
    // to the importer on a successful cuImportExternalMemory, so handing the
    // stored value out a second time would give the second importer an fd the
    // first one already closed. Windows is unchanged (CUDA duplicates the NT
    // handle; we keep and close ours).
    InteropExport out{};
    out.osHandle  = vulkan::takeOsHandle(ctx_.device(), st.posExt);
    out.sizeBytes = static_cast<std::size_t>(st.posExt.size);
    if (st.attrExt.handle != VK_NULL_HANDLE) {
        out.attrHandle    = vulkan::takeOsHandle(ctx_.device(), st.attrExt);
        out.attrSizeBytes = static_cast<std::size_t>(st.attrExt.size);
    }
    return out;
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

// Same growth discipline as ensureDescCapacity, and the same reason for growing
// EVERY slot at once: the capacity is shared bookkeeping, and a half-grown ring
// under-runs the other frame-in-flight the next time round.
void ParticleFieldPass::ensureBbParamCapacity(std::uint32_t frame, std::uint32_t count) {

    const std::uint32_t want = std::max(count, 1u);
    if (want <= bbParamCapacity_ && bbParamBufs_[frame].handle != VK_NULL_HANDLE) return;

    const std::uint32_t newCap = std::max(want, bbParamCapacity_ * 2u);
    for (auto& b : bbParamBufs_) retireOrDestroy(b);
    for (auto& b : bbParamBufs_) {
        b = createBuffer(ctx_.allocator(), ctx_.device(),
                         VkDeviceSize(newCap) * sizeof(BillboardParamsGpu),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VMA_MEMORY_USAGE_AUTO, kHostWrite);
    }
    bbParamCapacity_ = newCap;
}

// F4/F5 aux records. Same growth discipline, same window, same reason.
void ParticleFieldPass::ensureAuxCapacity(std::uint32_t frame, std::uint32_t count) {

    const std::uint32_t want = std::max(count, 1u);
    if (want <= auxCapacity_ && auxBufs_[frame].handle != VK_NULL_HANDLE) return;

    const std::uint32_t newCap = std::max(want, auxCapacity_ * 2u);
    for (auto& b : auxBufs_) retireOrDestroy(b);
    for (auto& b : auxBufs_) {
        b = createBuffer(ctx_.allocator(), ctx_.device(),
                         VkDeviceSize(newCap) * sizeof(EmitAuxGpu),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VMA_MEMORY_USAGE_AUTO, kHostWrite);
    }
    auxCapacity_ = newCap;
}

// ── F5: the height-bake pipeline ────────────────────────────────────────────
// The one pipeline in this pass that owns a descriptor set, and it owns exactly
// one binding: the scene TLAS. An acceleration structure cannot be reached by
// buffer_reference — that is the constraint F4's amendment note 4 recorded
// about the density volumes, arriving here for the same reason — so the choice
// was a set or no ray tracing. Everything else the bake needs (the destination
// map, the footprint) rides in a 40 B push block, and the CONSUMER of the map
// keeps its zero-descriptor property because a float buffer does have an
// address.
void ParticleFieldPass::ensureBakePipeline() {

    if (bakePipe_ != VK_NULL_HANDLE) return;
    if (!ctx_.rayQuerySupported()) return;
    const VkDevice d = ctx_.device();

    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1;
    lci.pBindings    = &b;
    check(vkCreateDescriptorSetLayout(d, &lci, nullptr, &bakeDsLayout_),
          "vkCreateDescriptorSetLayout(particle height bake)");

    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    check(vkCreateDescriptorPool(d, &pci, nullptr, &bakePool_),
          "vkCreateDescriptorPool(particle height bake)");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = bakePool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &bakeDsLayout_;
    check(vkAllocateDescriptorSets(d, &ai, &bakeSet_),
          "vkAllocateDescriptorSets(particle height bake)");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(BakePc);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &bakeDsLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    check(vkCreatePipelineLayout(d, &plci, nullptr, &bakePipeLayout_),
          "vkCreatePipelineLayout(particle height bake)");

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kParticleHeightBakeCompSpv);
    smci.pCode    = kParticleHeightBakeCompSpv;
    VkShaderModule mod = VK_NULL_HANDLE;
    check(vkCreateShaderModule(d, &smci, nullptr, &mod),
          "vkCreateShaderModule(particle_height_bake)");

    VkComputePipelineCreateInfo cpci{};
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout       = bakePipeLayout_;
    const VkResult pr =
            vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpci, nullptr, &bakePipe_);
    vkDestroyShaderModule(d, mod, nullptr);
    check(pr, "vkCreateComputePipelines(particle_height_bake)");
}

// The TLAS the bake traces. Called every frame from prepareParticleFields, and
// the descriptor is rewritten ONLY when the handle actually moved.
//
// That conditional is the whole safety argument. A TLAS handle is recreated
// only by a structural scene rebuild, which is bracketed by vkDeviceWaitIdle,
// so a write that happens on that frame cannot land on a set an in-flight frame
// names (R6 / VUID-03047). A steady-state scene refits the SAME acceleration
// structure object in place and this function writes nothing.
void ParticleFieldPass::setTlas(VkAccelerationStructureKHR tlas) {

    if (tlas == wantTlas_) return;
    wantTlas_ = tlas;
    // Every baked map was traced against the old structure.
    ++bakeStructGen_;
}

void ParticleFieldPass::recordSurfaceBake(VkCommandBuffer cb) {

    if (bakeDispatch_.empty() || bakePipe_ == VK_NULL_HANDLE) return;

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipe_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipeLayout_, 0, 1,
                            &bakeSet_, 0, nullptr);
    for (const BakeDispatch& bd : bakeDispatch_) {
        vkCmdPushConstants(cb, bakePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<std::uint32_t>(sizeof(BakePc)), bd.pc);
        vkCmdDispatch(cb, bd.groups, bd.groups, 1);
    }

    // Two dependencies in one barrier, and the second is the one that is easy
    // to miss.
    //
    //   WRITE -> READ: the map must be complete before particle_emit.comp,
    //     recorded immediately after this, dereferences it.
    //   READ -> WRITE: this pass TRACED the acceleration structure, and
    //     recordDeformAndTlas's per-frame refit WRITES it later in this same
    //     command buffer. Without the read-before-write edge that is an
    //     unsynchronised hazard even though nothing here writes the AS.
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                       VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    VkDependencyInfo di{};
    di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers    = &mb;
    vkCmdPipelineBarrier2(cb, &di);
}

// ── F4: publish one per-view billboard record ───────────────────────────────
// Called during recording, so this block is fixed-size and never grows here:
// an address handed to a vkCmdPushConstants earlier in the same command buffer
// would be dangling the moment the buffer is reallocated. A request past the
// end therefore returns 0, and the caller must skip the draw — a null
// buffer_reference dereference is undefined behaviour, not a no-op.
//
// Writing host-visible memory at record time is safe here for exactly the
// reason prepareFrame's writes are: this frame-in-flight's fence was waited on
// before recording began, so nothing in flight can be reading this slot.
VkDeviceAddress ParticleFieldPass::pushViewRecord(std::uint32_t frame,
                                                  const BillboardViewGpu& rec) {

    if (bbViewNext_ >= kBbViewSlots) return 0;
    Buffer& buf = bbViewBufs_[frame];
    if (buf.handle == VK_NULL_HANDLE) {
        buf = createBuffer(ctx_.allocator(), ctx_.device(),
                           VkDeviceSize(kBbViewSlots) * sizeof(BillboardViewGpu),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           VMA_MEMORY_USAGE_AUTO, kHostWrite);
        if (buf.handle == VK_NULL_HANDLE) return 0;
    }
    const VkDeviceSize off = VkDeviceSize(bbViewNext_) * sizeof(BillboardViewGpu);
    uploadHostVisible(ctx_.allocator(), buf, &rec, sizeof(rec), off);
    ++bbViewNext_;
    return buf.address + off;
}

void ParticleFieldPass::prepareFrame(std::uint64_t serial, std::uint32_t frame,
                                     const std::vector<Rec>& fields) {

    const std::uint32_t slot = static_cast<std::uint32_t>(serial % kSlots);
    // The slot the previous frame filled holds the previous frame's positions,
    // so it serves directly as the prevPositions buffer — no copy needed.
    //
    // A head-of-frame vkCmdCopyBuffer snapshot (as Ownership::Interop uses,
    // where one device buffer is rewritten by the sim's CUDA copy inside the
    // frame) would be wrong here: under HostRing the writer is the host, in
    // prepareFrame, which has already run by the time any command is recorded
    // — such a copy would capture this frame's positions and every motion
    // vector would be exactly zero. The ring's own depth is what keeps the
    // previous state readable, and a slot is only reused three frames later,
    // i.e. one full frame after the fence that retired it.
    const std::uint32_t prevSlot = (slot + kSlots - 1u) % kSlots;

    descScratch_.clear();
    descScratch_.reserve(fields.size());
    draws_.clear();
    draws_.reserve(fields.size());
    bbParamScratch_.clear();
    // F4: the per-view record cursor restarts every frame, and the glow gate is
    // recomputed from scratch — a field that stopped asking for a glow must
    // stop paying for one on the very next frame, not on the next scene change.
    bbViewNext_    = 0;
    glowActive_    = false;
    glowThreshold_ = 0.f;
    // The bound-volume list is rebuilt from scratch every frame and compared
    // against the previous one at the end: only a genuine change bumps the
    // generation, so a steady-state dust scene never triggers a descriptor
    // rewrite.
    const std::vector<DensityVolumeDesc> prevVols = densityVols_;
    densityVols_.clear();
    densityDispatch_.clear();
    emitDispatch_.clear();
    interopCopies_.clear();
    auxScratch_.clear();
    // F5: the bake list is rebuilt from scratch every frame and is EMPTY on the
    // overwhelming majority of them — a map is re-traced only when the key that
    // describes it moved (BakeKey).
    bakeDispatch_.clear();
    densityOverflow_ = 0;
    // R8/R10: rebuilt from scratch every frame and EMPTY unless some visible
    // field actually marches, which is what makes "both knobs at 0 records no
    // prepass" true frame by frame rather than only at scene load.
    transDispatch_.clear();
    // Parallel to transDispatch_, and local to this frame: which
    // bbParamScratch_ record each dispatch feeds, and where in the frame's
    // transmittance buffer its slice starts. Both are patched into real
    // addresses below, once the buffer's final size is known — growing it
    // mid-loop would invalidate every address already handed out.
    std::vector<std::uint32_t> transBbIndex;
    std::vector<std::uint32_t> transElemOff;
    std::uint32_t              transElems = 0;

    // Reclaim descriptor sets whose last referencing frame has provably
    // retired — the same `R + kFramesInFlight <= S` rule the resource retire
    // queue enforces, and the reason the sweep below queues rather than frees.
    // F6: and the same rule for a swept field's exported allocation.
    if (!extRetire_.empty()) {
        auto it = extRetire_.begin();
        while (it != extRetire_.end()) {
            if (it->serial + impl::kFramesInFlight <= serial) {
                vulkan::destroyExternalBuffer(ctx_.device(), it->buf);
                it = extRetire_.erase(it);
            } else {
                ++it;
            }
        }
    }

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

        std::uint32_t live = std::min(r.field->liveCount(), st.capacity);

        // Positions + count into this frame's slot, version-gated. A static or
        // parked field re-sends nothing; a field the sim advanced re-sends the
        // live prefix only. This is the design's only per-particle CPU cost —
        // and under Ownership::Renderer even that is gone: there are no host
        // positions to send, and the count is capacity, written once per slot
        // and never again (dataSerial only moves on submit/setLiveCount).
        // True for an Interop field whose copy is not registered: the count
        // published to the device is 0 this frame, whatever the host asked for,
        // and the slot's serial is invalidated so the REAL count is re-sent on
        // the first frame after the application arms it.
        bool parkedUnarmed = false;

        // ── F6: the foreign device copy, and this frame's snapshot ──────────
        // The callback is invoked here — post-fence, pre-record — and is
        // required to be synchronous, so by the time it returns the exported
        // allocation holds this step's positions and the command buffer that
        // will snapshot them has not been submitted yet. That host ordering is
        // the entire synchronisation story: there is no external semaphore,
        // and the only overlap left is the foreign API writing while an
        // earlier frame's snapshot copy reads — GPU to GPU, and benign (worst
        // case one copy blends two sim steps of grain positions, which no eye
        // and no motion vector can resolve).
        if (st.interopOwned) {
            if (st.deviceCopy) {
                st.deviceCopy();
            } else {
                // PARK the field for this frame rather than snapshot memory
                // nothing has written. The exported allocation is uninitialised
                // until the foreign API first writes it, and uninitialised
                // device memory is not zeros — it is arbitrary bit patterns
                // that decode to NaNs and 1e38 positions, i.e. a field of
                // grains smeared across the world and a world AABB to match.
                // Zero instances is both the safe answer and the true one, so
                // the diagnostic below can promise it.
                live = 0;
                parkedUnarmed = true;
                if (!st.interopUnarmedLogged) {
                    st.interopUnarmedLogged = true;
                    std::fprintf(stderr,
                                 "[ParticleField] an Ownership::Interop field is in the scene "
                                 "but no device copy is registered — it renders nothing. Call "
                                 "VulkanRenderer::enableParticleFieldInterop(field, copy) "
                                 "once, and import the handle it returns.\n");
                }
            }
            if (live > 0) {
                // The live prefix only, so a 300k-capacity field pouring its
                // first 5k grains copies 80 KB and not 4.8 MB. The tail holds
                // whatever the previous snapshot left; nothing reads past
                // instanceCount, which is this same number.
                interopCopies_.push_back({st.posExt.handle, st.positions[slot].handle,
                                          VkDeviceSize(live) * sizeof(ParticlePos)});
                // The attributes ride the same snapshot, in the same list, in
                // the same window, closed by the same barrier — which is the
                // whole of R2's "a mode where positions are safe and attributes
                // are not cannot exist by construction".
                if (st.attrExt.handle != VK_NULL_HANDLE &&
                    st.attributes[slot].handle != VK_NULL_HANDLE) {
                    interopCopies_.push_back({st.attrExt.handle, st.attributes[slot].handle,
                                              VkDeviceSize(live) * sizeof(ParticlePos)});
                }
                st.snapped[slot] = true;
            }
        }

        const std::uint64_t want = r.field->dataSerial();
        if (st.slotSerial[slot] != want || parkedUnarmed) {
            // 0 while parked: dataSerial never moves when the only thing that
            // changed is "a copy got registered", so a slot that recorded the
            // real serial while publishing a zero count would keep publishing
            // it forever.
            st.slotSerial[slot] = parkedUnarmed ? 0u : want;
            if (!st.rendererOwned && !st.interopOwned && live > 0) {
                uploadHostVisible(ctx_.allocator(), st.positions[slot],
                                  r.field->hostPositions().data(),
                                  VkDeviceSize(live) * sizeof(ParticlePos));
                // Stamped HERE and nowhere else: what makes a slot somebody's
                // "previous step" is that host bytes landed in it, not that a
                // frame went by.
                st.slotFill[slot] = ++st.fillSeq;
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

        // Attributes, once, on the HOST leg. Same serial guard and same
        // write-once contract as the orientations above; on the interop leg
        // hostAttributes() is empty and this never fires, because the colours
        // arrive through the snapshot instead.
        if (!st.interopOwned && st.attributes[0].handle != VK_NULL_HANDLE &&
            st.attrSerial != r.field->attributeSerial() &&
            !r.field->hostAttributes().empty()) {
            st.attrSerial = r.field->attributeSerial();
            uploadHostVisible(ctx_.allocator(), st.attributes[0],
                              r.field->hostAttributes().data(),
                              VkDeviceSize(st.capacity) * 16u);
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

        // An interop field's positions change on the device every frame without
        // dataSerial moving, so its "has this slot ever been filled" answer
        // comes from the snapshot bookkeeping rather than from the host serial.
        const bool prevValid = st.interopOwned ? st.snapped[prevSlot]
                                               : st.slotSerial[prevSlot] != 0;
        // ── Config::hostStableSlots: the previous slot is prevPositions ─────
        // Only when this frame's slot and the previous one hold consecutive
        // uploads. A frame the host skipped leaves a three-frame-old submit in
        // its slot, and the ring would then hand the shader a displacement over
        // three steps — or, once the wrap puts the newer submit in the older
        // slot, a backwards one. The stretch simply switches off for that frame
        // and the sprite is round.
        const bool hostPrevIsPrevStep =
                r.field->config().hostStableSlots && !st.rendererOwned && !st.interopOwned &&
                st.slotFill[slot] != 0 && st.slotFill[slot] == st.slotFill[prevSlot] + 1u;

        FieldDescGpu d{};
        const auto& world = r.field->matrixWorld->elements;
        std::memcpy(d.world, world.data(), sizeof(d.world));
        if (st.rendererOwned) {
            // Single instance, both of them, and prevPositions is a real
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
        // Config::attributes. The interop leg reads THIS frame's ring slot for
        // exactly the reason the positions do; every other mode has one buffer
        // written once, and slot 0 is it. 0 when the field has none.
        const VkDeviceAddress attrAddr =
                st.interopOwned ? st.attributes[slot].address : st.attributes[0].address;
        d.attrAddr    = attrAddr;
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
        // not to. A parked Renderer field (setLiveCount(0)) records no dispatch
        // at all, which is the one supported way to stop an emitter.
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
            pc.capacityAndFollow = st.capacity;
            pc.seed       = ep.seed;
            pc.driftScale = std::max(ep.driftScale, 0.f);
            // ── The toroidal follow centre ──────────────────────────────────
            // Bit 31 of the count carries the mode (see EmitPc), so a field
            // that does not follow pushes the identical bytes it pushed before
            // this existed — capacity is a slot count and cannot reach 2^31.
            //
            // World → field-local here, not on the API, because this is the
            // only place the field's world matrix is guaranteed current, and
            // the shader's positions are field-local by the mode's contract.
            // Translation only: a rotated world matrix would need the inverse
            // basis applied to the wrap box as well; the rotation is ignored
            // and the box stays world-axis-aligned.
            EmitAuxGpu aux{};
            bool       needAux = false;
            if (ep.follow) {
                pc.capacityAndFollow |= 0x80000000u;
                const Vector3& fc = r.field->followCenter();
                aux.followX = fc.x - world[12];
                aux.followZ = fc.z - world[14];
                needAux     = true;
            }

            // ── F5: the surface footprint and its bake ──────────────────────
            const auto& sp = ep.surface;
            if (sp.enabled) {
                if (!ctx_.rayQuerySupported()) {
                    // A device with no ray query cannot trace a height map, and
                    // there is no second implementation to fall back to. Said
                    // once, then silent: surface interaction is simply absent.
                    if (!bakeUnsupportedLogged_) {
                        bakeUnsupportedLogged_ = true;
                        std::fprintf(stderr,
                                     "[threepp] ParticleField: EmitterParams::Surface needs "
                                     "VK_KHR_ray_query, which this device does not support "
                                     "— flakes will not rest.\n");
                    }
                } else {
                    ensureBakePipeline();
                }
            }
            if (sp.enabled && bakePipe_ != VK_NULL_HANDLE &&
                wantTlas_ != VK_NULL_HANDLE) {

                // The set holds a stale acceleration structure. Written here,
                // in prepareFrame's post-fence window, and only on the frame
                // the handle actually changed — which is a structural rebuild,
                // which is itself vkDeviceWaitIdle-guarded (see setTlas).
                if (bakeTlas_ != wantTlas_) {
                    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
                    asInfo.sType =
                            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
                    asInfo.accelerationStructureCount = 1;
                    asInfo.pAccelerationStructures    = &wantTlas_;
                    VkWriteDescriptorSet w{};
                    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.pNext           = &asInfo;
                    w.dstSet          = bakeSet_;
                    w.dstBinding      = 0;
                    w.descriptorCount = 1;
                    w.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                    vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
                    bakeTlas_ = wantTlas_;
                }

                const std::uint32_t res =
                        std::max(kBakeResMin, std::min(kBakeResMax, sp.resolution));
                // The footprint. Square, and by default the spawn slab's larger
                // lateral half-extent — which under EmitterParams::follow is
                // also exactly half the toroidal wrap period, so the map covers
                // precisely the box the flakes are folded into and no more.
                const float extent =
                        (sp.extent > 0.f)
                                ? sp.extent
                                : std::max(std::max(ep.spawnHalfExtent.x,
                                                    ep.spawnHalfExtent.z), 0.5f);
                // The centre: when the field follows, the map re-anchors on the
                // same snapped centre the wrap uses, and therefore only on
                // snaps. Anything else — the raw camera position, a second snap
                // expression — and the flakes would rest at heights sampled
                // from where the field used to be.
                float cx = ep.spawnCenter.x, cz = ep.spawnCenter.z;
                if (ep.follow) { cx = aux.followX; cz = aux.followZ; }

                // The vertical search band, field-local. Derived when the
                // author did not name one: from the spawn slab's ceiling down
                // past one lifetime of fall, which is the band the trajectory
                // can actually occupy.
                float topY = sp.searchTop, botY = sp.searchBottom;
                if (!(topY > botY)) {
                    topY = ep.spawnCenter.y + ep.spawnHalfExtent.y;
                    const float fall =
                            std::abs(ep.velocity.y + ep.wind.y) * ep.lifetime;
                    botY = topY - std::max(fall, 1.f) - 2.f;
                }

                BakeKey key{};
                key.cx     = cx;
                key.cz     = cz;
                key.extent = extent;
                key.res    = res;
                key.topY   = topY;
                key.depth  = topY - botY;
                key.worldX = world[12];
                key.worldY = world[13];
                key.worldZ = world[14];
                key.structGen = bakeStructGen_;

                // Resolution is the only thing that changes the ALLOCATION; the
                // centre and extent are a transform on the same texels.
                if (st.heightRes != res) {
                    for (auto& b : st.heights) retireOrDestroy(b);
                    for (auto& b : st.heights) {
                        b = createBuffer(ctx_.allocator(), ctx_.device(),
                                         VkDeviceSize(res) * VkDeviceSize(res) *
                                                 sizeof(float),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_AUTO, 0);
                    }
                    st.heightRes = res;
                    for (auto& k : st.bakedKey) k = BakeKey{};
                }

                Buffer& hb = st.heights[frame];
                if (hb.handle != VK_NULL_HANDLE) {
                    const float cell  = (2.f * extent) / float(res);
                    // "Nothing to land on", far below anything the trajectory
                    // reaches — so the solve reports no landing and the
                    // particle falls out of the world exactly as it did pre-F5.
                    const float missY = botY - 1.0e4f;

                    if (st.bakedKey[frame] != key) {
                        BakePc bp{};
                        bp.dstAddr = hb.address;
                        bp.originX = world[12] + cx - extent;
                        bp.originZ = world[14] + cz - extent;
                        bp.cell    = cell;
                        bp.res     = res;
                        bp.topY    = world[13] + topY;
                        bp.depth   = key.depth;
                        bp.missY   = missY;
                        bp.yOffset = world[13];

                        BakeDispatch bd{};
                        bd.groups = (res + kBakeLocalSize - 1u) / kBakeLocalSize;
                        std::memcpy(bd.pc, &bp, sizeof(bp));
                        bakeDispatch_.push_back(bd);
                        st.bakedKey[frame] = key;
                    }

                    aux.heightAddr    = hb.address;
                    aux.bakeOriginX   = cx - extent;
                    aux.bakeOriginZ   = cz - extent;
                    aux.bakeInvCell   = 1.f / cell;
                    aux.bakeRes       = res;
                    aux.bakeMiss      = missY;
                    aux.landBias      = sp.bias;
                    aux.restSeconds   = std::max(sp.restSeconds, 0.f);
                    aux.restJitter    = std::max(0.f, std::min(1.f, sp.restJitter));
                    // Floored, not clamped to zero: a zero fade would divide by
                    // zero in landedState and hand every consumer a NaN radius,
                    // which is the failure mode that defeats bounds checks
                    // rather than showing up as one bad flake.
                    aux.fadeSeconds   = std::max(sp.fadeSeconds, 1e-3f);
                    aux.splashSeconds = std::max(sp.splashSeconds, 0.f);
                    // The splash ring in absolute metres. R0 sits just past the
                    // largest radius the size jitter can produce, so a w at or
                    // above it can only be a ring — which is the whole of how
                    // the billboard recognises one without a second channel.
                    if (aux.splashSeconds > 0.f) {
                        const float rMax =
                                std::max(ep.size, 1e-5f) *
                                (1.f + std::max(0.f, std::min(1.f, ep.sizeJitter)));
                        aux.splashR0 = rMax * 1.05f;
                        aux.splashR1 = std::max(rMax * std::max(sp.splashGrow, 1.1f),
                                                aux.splashR0 * 1.01f);
                    }
                    needAux           = true;
                }
            }

            EmitDispatch ed{};
            ed.groups = (st.capacity + kEmitLocalSize - 1u) / kEmitLocalSize;
            if (needAux) {
                ed.auxIndex = static_cast<std::uint32_t>(auxScratch_.size());
                auxScratch_.push_back(aux);
            }
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
                // Per-field medium + emission, carried with the volume
                // (plans/particle-atmosphere.md F-A).
                v.albedo[0]  = dr.albedo.r;
                v.albedo[1]  = dr.albedo.g;
                v.albedo[2]  = dr.albedo.b;
                v.anisotropy = dr.anisotropy;
                v.emission[0] = std::max(dr.emissiveIntensity, 0.f);
                v.emission[1] = dr.tempBottomK;
                v.emission[2] = dr.tempTopK;
                // The shader raises the in-box height fraction to this power.
                // Clamped here rather than in the march: pow(0, 0) is undefined
                // in GLSL, and a floor costs nothing on the host but two
                // instructions per step per volume in the shader.
                v.emission[3] = std::max(dr.tempFalloff, 1e-3f);
                densityVols_.push_back(v);

                DensityDispatch dd{};
                // densityVols_ was just pushed, so this volume is its last
                // element — the same index the UBO, the descriptor array and
                // the majorant buffer all use.
                dd.volIndex   = static_cast<std::uint32_t>(densityVols_.size() - 1u);
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

        // ── Billboards (plans/particle-atmosphere.md F-D) ───────────────────
        // Same live-count gate as the density volume, for the same reason: a
        // parked field must not keep a draw in the command stream that resolves
        // to zero instances every frame, and it must not keep the overlay pass
        // switched on for content that is not there.
        const auto& bb = r.field->billboardRepr();
        if (bb.enabled && live > 0) {
            // The draw record, allocated the first frame this field asks for
            // one. Always four vertices; only instanceCount ever changes, and
            // it changes on the device.
            if (st.bbIndirect[slot].handle == VK_NULL_HANDLE) {
                st.bbIndirect[slot] = createBuffer(ctx_.allocator(), ctx_.device(),
                                                   sizeof(VkDrawIndirectCommand), kIndirectUsage,
                                                   VMA_MEMORY_USAGE_AUTO, kHostWrite);
            }
            VkDrawIndirectCommand bcmd{};
            bcmd.vertexCount   = 4u;// the quad, as a triangle strip
            bcmd.instanceCount = 0u;// filled on the device by recordCounts
            uploadHostVisible(ctx_.allocator(), st.bbIndirect[slot], &bcmd, sizeof(bcmd));

            BillboardParamsGpu bp{};
            bp.posAddr     = d.posAddr;
            bp.prevPosAddr = d.prevPosAddr;
            bp.colorHot[0] = bb.colorHot.r;
            bp.colorHot[1] = bb.colorHot.g;
            bp.colorHot[2] = bb.colorHot.b;
            bp.sizeScale   = std::max(bb.sizeScale, 0.f);
            bp.colorCool[0] = bb.colorCool.r;
            bp.colorCool[1] = bb.colorCool.g;
            bp.colorCool[2] = bb.colorCool.b;
            bp.uniformRadius = cfg.uniformRadius;
            bp.stretchMax    = std::max(bb.stretchMax, 0.f);
            bp.intensity     = std::max(bb.intensity, 0.f);
            bp.softness      = std::max(0.f, std::min(1.f, bb.softness));
            bp.coreWeight    = std::max(bb.coreWeight, 0.f);
            bp.pad0          = 0.f;
            bp.fadePower     = std::max(bb.fadePower, 0.f);
            bp.brightJitter  = std::max(0.f, std::min(1.f, bb.brightJitter));
            bp.sizeTaper     = std::max(0.f, std::min(1.f, bb.sizeTaper));
            bp.flags = (cfg.wSemantic == ParticleField::WSemantic::Radius) ? 1u : 0u;
            // ── 4c: the sprite slice ────────────────────────────────────────
            // Two bits and three floats. Both default off and everything below
            // is exactly zero then, keeping the additive path byte-identical.
            if (bb.alphaOver) bp.flags |= 2u;
            if (bb.lit) bp.flags |= 4u;
            bp.opacity    = std::max(0.f, std::min(1.f, bb.opacity));
            bp.litPhaseG  = std::max(-0.95f, std::min(0.95f, bb.litPhaseG));
            bp.litAmbient = std::max(bb.litAmbient, 0.f);
            ds.bbAlphaOver = bb.alphaOver;

            // ── Per-particle colour (R2/R3) ─────────────────────────────────
            // Present ⇒ the sprite's rgb IS attribute.rgb and colorHot /
            // colorCool are not read at all. One scheme or the other, decided
            // by this bit; no blend of the two exists to get wrong.
            bp.attrAddr = attrAddr;
            if (attrAddr != 0) bp.flags |= 8u;

            // ── The volumetric marches (R4/R5, prepassed by R8) ─────────────
            // Both knobs at 0 is the EXACT no-op: nothing below is published,
            // no dispatch is recorded, no transmittance buffer is allocated
            // (R10), the shader's uniform branch skips the fetch, and the field
            // renders the bits it did before this feature existed.
            const float volExt = std::max(bb.volumeExtinction, 0.f);
            const float volShd = std::max(0.f, std::min(1.f, bb.volumeShadow));
            if ((volExt > 0.f || volShd > 0.f) &&
                st.densityLin.view != VK_NULL_HANDLE && dr.enabled) {
                // The field's world matrix as ROWS of its affine part: the
                // marches happen in world space and the positions are
                // field-local, and this is the one basis neither the vertex
                // stage nor the prepass can otherwise reach. Written into bp
                // first because the prepass's push block copies it from there —
                // one expression, two consumers, no way for them to disagree.
                for (int rw = 0; rw < 3; ++rw) {
                    bp.model[rw * 4 + 0] = static_cast<float>(world[rw]);
                    bp.model[rw * 4 + 1] = static_cast<float>(world[rw + 4]);
                    bp.model[rw * 4 + 2] = static_cast<float>(world[rw + 8]);
                    bp.model[rw * 4 + 3] = static_cast<float>(world[rw + 12]);
                }
                // The SAME box expression the density scatter and every
                // sampler of this volume use, so the march cannot disagree
                // with the volume about where the dust is.
                const float hx = std::max(dr.halfExtent.x, 1e-4f);
                const float hy = std::max(dr.halfExtent.y, 1e-4f);
                const float hz = std::max(dr.halfExtent.z, 1e-4f);
                bp.boxMin[0] = dr.center.x - hx;
                bp.boxMin[1] = dr.center.y - hy;
                bp.boxMin[2] = dr.center.z - hz;
                bp.boxInvSize[0] = 1.f / (2.f * hx);
                bp.boxInvSize[1] = 1.f / (2.f * hy);
                bp.boxInvSize[2] = 1.f / (2.f * hz);

                // ── R8: the prepass that does the marching ──────────────────
                // Everything above describes WHERE the dust is; this hands the
                // same description to the compute pass that walks it. The
                // vertex stage no longer marches at all — it fetches by slot —
                // so the flag is only raised once the dispatch is real. A
                // device that cannot create the pipeline (or a pool with no
                // set left) therefore renders FLAT sprites rather than sprites
                // multiplied by an unwritten buffer.
                bool marching = false;
                if (ensureTransmittancePipeline()) {
                    // The set names the field's r16f mirror, whose handle never
                    // changes for the life of the field — so it is allocated
                    // and written ONCE, here, and never touched again. That is
                    // what keeps this pass out of the VUID-03047 zone despite
                    // owning a descriptor.
                    if (st.marchSet == VK_NULL_HANDLE) {
                        VkDescriptorSetAllocateInfo ai{};
                        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                        ai.descriptorPool     = densityPool_;
                        ai.descriptorSetCount = 1;
                        ai.pSetLayouts        = &transDsLayout_;
                        if (vkAllocateDescriptorSets(ctx_.device(), &ai, &st.marchSet) == VK_SUCCESS) {
                            VkDescriptorImageInfo ii{};
                            ii.imageView = st.densityLin.view;
                            // GENERAL for its whole life — the convert dispatch
                            // writes it as a storage image and nothing ever
                            // transitions it, the same contract the deferred
                            // set's binding 69 is written under.
                            ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                            ii.sampler     = transSampler_;
                            VkWriteDescriptorSet w{};
                            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                            w.dstSet          = st.marchSet;
                            w.dstBinding      = 0;
                            w.descriptorCount = 1;
                            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                            w.pImageInfo      = &ii;
                            vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
                        } else {
                            st.marchSet = VK_NULL_HANDLE;
                        }
                    }
                    if (st.marchSet != VK_NULL_HANDLE) {
                        TransmitPc tp{};
                        tp.posAddr = d.posAddr;
                        // An ELEMENT OFFSET for now; the base is not known
                        // until every field has been counted.
                        tp.outAddr = 0;
                        std::memcpy(tp.model, bp.model, sizeof(tp.model));
                        std::memcpy(tp.boxMin, bp.boxMin, sizeof(tp.boxMin));
                        std::memcpy(tp.boxInvSize, bp.boxInvSize, sizeof(tp.boxInvSize));
                        tp.capacity = st.capacity;
                        // Per KNOB, not per field: a field that asks only for
                        // self-shadowing marches one ray, not two.
                        tp.flags = (volExt > 0.f ? kTransCamBit : 0u) |
                                   (volShd > 0.f ? kTransSunBit : 0u);

                        TransDispatch td{};
                        td.set    = st.marchSet;
                        td.groups = (st.capacity + kTransmitLocalSize - 1u) / kTransmitLocalSize;
                        std::memcpy(td.pc, &tp, sizeof(tp));
                        transDispatch_.push_back(td);
                        transBbIndex.push_back(
                                static_cast<std::uint32_t>(bbParamScratch_.size()));
                        transElemOff.push_back(transElems);
                        transElems += st.capacity;
                        marching = true;
                    }
                }
                if (marching) {
                    bp.volumeExtinction = volExt;
                    bp.volumeShadow     = volShd;
                    bp.volumeAmbient    = std::max(bb.volumeAmbient, 0.f);
                    bp.volumeSunGain    = std::max(bb.volumeSunGain, 0.f);
                    bp.flags |= 16u;
                }
            }

            // ── F4 ──────────────────────────────────────────────────────────
            bp.glow             = std::max(bb.glow, 0.f);
            bp.stretchMaxScreen = std::max(bb.stretchMaxScreen, 0.f);
            bp.nearFade         = std::max(bb.nearFade, 0.f);
            bp.lodNear          = std::max(bb.lodNear, 0.f);
            bp.lodFade          = std::max(bb.lodFade, 0.f);

            // The glow chain is a scene-level decision made out of per-field
            // requests: every glow field composites additively into the one
            // offscreen target, so N fields still cost one pyramid. With no
            // glow field the renderer allocates no target and records nothing.
            if (bp.glow > 0.f) {
                ds.glow          = true;
                ds.glowThreshold = std::max(bb.glowThreshold, 0.f);
                glowActive_      = true;
                glowThreshold_   = std::max(glowThreshold_, ds.glowThreshold);
            }

            // The stretch is expressed against the frame's own motion interval:
            // the shader multiplies (pos - prevPos) by this, and the two
            // positions are dt apart, so seconds/dt is exactly "how many
            // seconds of travel to smear over". Doing the divide here keeps dt
            // out of the shader entirely and keeps the stretch stable when the
            // frame rate is not.
            //
            // A HostRing field needs two host guarantees for this:
            // Config::hostStableSlots (index i in the previous slot names the
            // same particle) and submit()'s dtSec (the interval). With both,
            // the previous slot is a real prevPositions buffer and the streak
            // is the same expression it is for a Renderer field. Without them
            // the stretch stays zero rather than silently wrong.
            const bool rendererOwned = st.rendererOwned;
            const float edt = rendererOwned ? r.field->emitterDt() : r.field->hostDt();
            bp.stretchOverDt = ((rendererOwned || hostPrevIsPrevStep) &&
                                bb.stretchSeconds > 0.f && edt > 1e-6f)
                                       ? (bb.stretchSeconds / edt)
                                       : 0.f;

            // ── The lifecycle, handed over so the shader can re-derive age ───
            // There is no age channel in the position buffer (w is the radius),
            // so the billboard vertex stage recomputes a slot's age from the
            // same closed form and the same hash particle_emit.comp used —
            // this is what makes fade-over-life, shrink-over-life and the
            // hot-to-cool colour ramp possible without a second buffer.
            // lifetime == 0 tells the shader no age is knowable, which is the
            // case for a HostRing field driven by a sim.
            if (rendererOwned) {
                const auto& ep = r.field->emitter();
                bp.lifetime       = std::max(ep.lifetime, 1e-3f);
                bp.lifetimeJitter = std::max(0.f, std::min(1.f, ep.lifetimeJitter));
                bp.duty           = std::max(1e-3f, std::min(1.f, ep.dutyCycle));
                bp.time           = r.field->emitterTime();
                bp.seed           = ep.seed;
                // ── F5: the splash decode ───────────────────────────────────
                // The same two numbers the emitter encodes with, computed by
                // the same expression a few dozen lines below — a matched pair
                // the pass owns both ends of, so neither is authored directly.
                if (ep.surface.enabled && ep.surface.splashSeconds > 0.f) {
                    const float rMax =
                            std::max(ep.size, 1e-5f) *
                            (1.f + std::max(0.f, std::min(1.f, ep.sizeJitter)));
                    bp.splashR0 = rMax * 1.05f;
                    bp.splashR1 = std::max(rMax * std::max(ep.surface.splashGrow, 1.1f),
                                           bp.splashR0 * 1.01f);
                }
            }
            bp.splashRingWidth = std::max(0.02f, std::min(1.f, bb.splashRingWidth));

            ds.bbIndirect = st.bbIndirect[slot].handle;
            ds.billboard  = true;
            // The address is patched in after the block is (re)allocated below —
            // growing it here would invalidate every address already handed out
            // this frame.
            ds.bbParamsAddr = static_cast<VkDeviceAddress>(bbParamScratch_.size());
            bbParamScratch_.push_back(bp);
        }

        draws_.push_back(ds);
    }

    descCount_ = static_cast<std::uint32_t>(descScratch_.size());
    ensureDescCapacity(frame, descCount_);
    if (descCount_ > 0) {
        uploadHostVisible(ctx_.allocator(), descBufs_[frame], descScratch_.data(),
                          VkDeviceSize(descCount_) * sizeof(FieldDescGpu));
    }

    // R8: the frame's transmittance buffer, sized to every marching field's
    // slots laid end to end, and the two addresses that name each slice — the
    // prepass's destination and the vertex stage's source. Done BEFORE the
    // billboard-params upload because it writes into bbParamScratch_, and
    // AFTER the field loop because a buffer grown mid-loop would invalidate
    // every address already handed out.
    //
    // Empty on every scene without a volumetric field, which is where R10's
    // "not one allocated byte" actually lives: transBufs_ stay VK_NULL_HANDLE.
    if (!transDispatch_.empty()) {
        ensureTransCapacity(transElems);
        const VkDeviceAddress base = transBufs_[frame].address;
        for (std::size_t i = 0; i < transDispatch_.size(); ++i) {
            const VkDeviceAddress a =
                    base + VkDeviceSize(transElemOff[i]) * sizeof(std::uint32_t);
            std::memcpy(transDispatch_[i].pc + offsetof(TransmitPc, outAddr), &a, sizeof(a));
            bbParamScratch_[transBbIndex[i]].transAddr = a;
        }
    }

    // Billboard params: one upload for every field, then turn the indices the
    // loop stashed into real device addresses.
    if (!bbParamScratch_.empty()) {
        ensureBbParamCapacity(frame, static_cast<std::uint32_t>(bbParamScratch_.size()));
        uploadHostVisible(ctx_.allocator(), bbParamBufs_[frame], bbParamScratch_.data(),
                          VkDeviceSize(bbParamScratch_.size()) * sizeof(BillboardParamsGpu));
        const VkDeviceAddress base = bbParamBufs_[frame].address;
        for (DrawState& ds : draws_) {
            if (!ds.billboard) continue;
            ds.bbParamsAddr = base + ds.bbParamsAddr * sizeof(BillboardParamsGpu);
        }
    }

    // F4/F5 aux records: one upload for the whole frame, then patch each emit
    // push block's trailing address. The patch is a memcpy into the prebuilt
    // bytes at offset 120 rather than a rebuild of the block, because the block
    // is otherwise finished and the address is the only thing that could not be
    // known while the loop was still deciding how many records there would be.
    if (!auxScratch_.empty()) {
        ensureAuxCapacity(frame, static_cast<std::uint32_t>(auxScratch_.size()));
        uploadHostVisible(ctx_.allocator(), auxBufs_[frame], auxScratch_.data(),
                          VkDeviceSize(auxScratch_.size()) * sizeof(EmitAuxGpu));
        const VkDeviceAddress base = auxBufs_[frame].address;
        for (EmitDispatch& ed : emitDispatch_) {
            if (ed.auxIndex == 0xffffffffu) continue;
            const VkDeviceAddress a = base + ed.auxIndex * sizeof(EmitAuxGpu);
            std::memcpy(ed.pc + offsetof(EmitPc, auxAddr), &a, sizeof(a));
        }
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
            for (auto& b : st.bbIndirect) retireOrDestroy(b);
            for (auto& b : st.heights) retireOrDestroy(b);
            retireOrDestroy(st.orientations);
            for (auto& b : st.attributes) retireOrDestroy(b);
            // The exported allocation cannot go through the renderer's retire
            // queue (that takes Buffers; this one is outside VMA), so it takes
            // the same rule by hand. The application's CUDA import of it is the
            // application's to release — dropping a field whose handle is still
            // imported is its bug, not one this sweep can fix.
            if (st.posExt.handle != VK_NULL_HANDLE) {
                extRetire_.push_back({st.posExt, serial});
                st.posExt = vulkan::ExternalBuffer{};
            }
            if (st.attrExt.handle != VK_NULL_HANDLE) {
                extRetire_.push_back({st.attrExt, serial});
                st.attrExt = vulkan::ExternalBuffer{};
            }
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
            // R8's set comes out of the same pool and goes back by the same
            // rule; leaking it would cap the process at four marching fields.
            if (st.marchSet != VK_NULL_HANDLE) {
                densitySetRetire_.push_back({st.marchSet, serial});
                st.marchSet = VK_NULL_HANDLE;
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

// ── R8/R9: (T_cam, T_sun) once per particle, once per view ──────────────────
// The whole optimisation, as a command stream: one bind, then per marching
// field two memcpys into a prebuilt push block, one vkCmdPushConstants and one
// vkCmdDispatch over the CPU-CONSTANT capacity domain — the same shape as the
// emit and scatter dispatches beside it. Nothing is uploaded and no descriptor
// is written; the field's set was written once when it first marched.
//
// PER VIEW BY RE-DISPATCH (R9). T_sun is view-independent and T_cam is not, and
// views are recorded sequentially into one command buffer, so the second view
// simply overwrites the first view's answers behind its own barrier rather than
// owning a second buffer. At 4M slots that trade is 16 MB per extra view saved
// for eight extra taps per particle re-run — and the sun leg has to be re-run
// anyway to keep the two halves of one packed word consistent.
//
// The BARRIER is the contract: this pass writes the buffer through a
// buffer_reference (SHADER_STORAGE_WRITE) and the billboard VERTEX stage reads
// it through one, so without the compute→vertex dependency the draws that
// follow are reading memory whose writes have not been made visible. It is one
// barrier for every field, not one per field: the dispatches are independent
// and the consumer is downstream of all of them.
//
// Recorded OUTSIDE any render-pass instance — a compute dispatch cannot go
// inside one — which is why the call site is immediately before the view's
// vkCmdBeginRendering rather than inside recordFieldBillboards.
void ParticleFieldPass::recordTransmittance(VkCommandBuffer cb, const float camWorld[3],
                                            const float sunDirWorld[3]) {

    if (transDispatch_.empty() || transPipe_ == VK_NULL_HANDLE) return;

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, transPipe_);
    for (TransDispatch& td : transDispatch_) {
        // The two per-view vectors, patched into the block prepareFrame built.
        // A rebuild would recompute a matrix and a box that did not change.
        std::memcpy(td.pc + offsetof(TransmitPc, camWorld), camWorld, 3 * sizeof(float));
        std::memcpy(td.pc + offsetof(TransmitPc, sunDirWorld), sunDirWorld, 3 * sizeof(float));
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, transPipeLayout_,
                                0, 1, &td.set, 0, nullptr);
        vkCmdPushConstants(cb, transPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<std::uint32_t>(sizeof(TransmitPc)), td.pc);
        vkCmdDispatch(cb, td.groups, 1, 1);
    }

    // COMPUTE is on the destination list as well as VERTEX: the NEXT view's
    // re-dispatch writes the same buffer this one just wrote, so without it the
    // two dispatches are an unsynchronised write-after-write over the whole
    // buffer. The vertex reads in between are what make that hazard real.
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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
    // The majorants go with them: an atomicMax accumulator that is not zeroed
    // would hold the high-water mark of the whole session, which is a valid
    // bound but a uselessly loose one the moment a plume disperses. Every slot
    // is zeroed, not just the live ones — a slot whose field went away must not
    // leave a stale bound behind for the next field that lands in it.
    vkCmdFillBuffer(cb, densityMajorants_.handle, 0, VK_WHOLE_SIZE, 0u);

    // 2. Clear → atomic accumulate. ALL_TRANSFER rather than CLEAR because the
    //    majorant zeroing above is a fill, not a clear.
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
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

    // 5. r32ui → r16f, one dispatch per field — the copy exists to give the
    //    deferred shade's per-pixel dust march hardware trilinear filtering.
    //    The same dispatch reduces the volume's maximum into the majorant
    //    buffer — it is already reading every voxel, so the bound the sensor
    //    needs costs one shared-memory reduction and one atomic per workgroup.
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, convertPipe_);
    for (const DensityDispatch& dd : densityDispatch_) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, convertPipeLayout_,
                                0, 1, &dd.convertSet, 0, nullptr);
        vkCmdPushConstants(cb, convertPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(dd.volIndex), &dd.volIndex);
        const std::uint32_t g = (dd.res + kConvertLocalSize - 1u) / kConvertLocalSize;
        vkCmdDispatch(cb, g, g, g);
    }

    // 6. Both volumes → every read this frame: the froxel passes' manual
    //    trilinear on the uint volume (all views — they ride this one write)
    //    and the shade's hardware trilinear on the r16f mirror. The majorants
    //    are read by the LIDAR pass, which is a SEPARATE submission and is
    //    ordered by submission order, not by this barrier.
    //
    //    VERTEX is on the list because the billboard stage now marches the
    //    r16f mirror for its transmittance terms (plans/particle-volumetric-
    //    sprites R4), later in this same command buffer and in a graphics pass.
    //    Without it that read is unsynchronised against the convert's writes.
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    vkCmdPipelineBarrier2(cb, &dep);
}

// ── F6: exported allocation → this frame's ring slot ────────────────────────
// One copy per interop field, then one barrier for all of them — the copies
// are independent and the copy engine can overlap them, same batching as
// recordCounts.
//
// The destination stages are everything that can read a position this frame:
// the vertex stage (the mesh proxy and the billboard quad both pull positions
// by device address), compute (the density scatter), and transfer (nothing
// today, but the AABB/BLAS phase's copy is on this list the day it lands).
void ParticleFieldPass::recordInteropSnapshot(VkCommandBuffer cb) {

    if (interopCopies_.empty()) return;

    for (const InteropCopy& c : interopCopies_) {
        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size      = c.bytes;
        vkCmdCopyBuffer(cb, c.src, c.dst, 1, &region);
    }

    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &mb;
    vkCmdPipelineBarrier2(cb, &dep);
}

void ParticleFieldPass::recordCounts(VkCommandBuffer cb) {

    if (draws_.empty()) return;

    bool any = false;
    VkBufferCopy region{};
    region.srcOffset = 0;// FieldCountsGpu::liveCount
    region.dstOffset = kInstanceCountOffset;
    region.size      = sizeof(std::uint32_t);
    for (const DrawState& d : draws_) {
        if (d.vertexCount != 0u && d.indirect != VK_NULL_HANDLE) {
            vkCmdCopyBuffer(cb, d.counts, d.indirect, 1, &region);
            any = true;
        }
        // The billboard record takes the SAME 4-byte device copy: the two
        // representations are independent draws of the same field, so each owns
        // a record and neither learns the count on the host.
        if (d.billboard && d.bbIndirect != VK_NULL_HANDLE) {
            vkCmdCopyBuffer(cb, d.counts, d.bbIndirect, 1, &region);
            any = true;
        }
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
