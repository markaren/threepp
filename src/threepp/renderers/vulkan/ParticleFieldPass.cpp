#include "ParticleFieldPass.hpp"

#include "VulkanContext.hpp"

#include "threepp/objects/ParticleField.hpp"

#include <algorithm>
#include <cstring>

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

}// namespace

ParticleFieldPass::ParticleFieldPass(VulkanContext& ctx, RetireBufferFn retireFn)
    : ctx_(ctx), retireFn_(std::move(retireFn)) {}

ParticleFieldPass::~ParticleFieldPass() {

    // Destructor only: the renderer destroys this after vkDeviceWaitIdle, so
    // nothing can still name these. Inline destroy, not retire — the retire
    // queue is being torn down alongside us.
    for (auto& [_, st] : states_) destroyState(*st);
    states_.clear();
    for (auto& b : descBufs_) destroyBuffer(ctx_.allocator(), b);
}

void ParticleFieldPass::retireOrDestroy(Buffer& b) {

    if (b.handle == VK_NULL_HANDLE) return;
    if (retireFn_) retireFn_(std::move(b));
    else destroyBuffer(ctx_.allocator(), b);
    b = Buffer{};
}

void ParticleFieldPass::destroyState(State& st) {

    for (auto& b : st.positions) destroyBuffer(ctx_.allocator(), b);
    for (auto& b : st.counts) destroyBuffer(ctx_.allocator(), b);
    for (auto& b : st.indirect) destroyBuffer(ctx_.allocator(), b);
    destroyBuffer(ctx_.allocator(), st.orientations);
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
    for (std::uint32_t s = 0; s < kSlots; ++s) {
        st->positions[s] = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                        kPositionUsage, VMA_MEMORY_USAGE_AUTO, kHostWrite);
        st->counts[s] = createBuffer(ctx_.allocator(), ctx_.device(), sizeof(FieldCountsGpu),
                                     kCountsUsage, VMA_MEMORY_USAGE_AUTO, kHostWrite);
        st->indirect[s] = createBuffer(ctx_.allocator(), ctx_.device(),
                                       sizeof(VkDrawIndirectCommand), kIndirectUsage,
                                       VMA_MEMORY_USAGE_AUTO, kHostWrite);
        st->slotSerial[s] = 0;// fresh allocation holds garbage
    }
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

    for (const Rec& r : fields) {
        if (!r.field) continue;
        State& st = ensureState(*r.field);
        st.lastSeenSerial = serial;

        // Positions + count into THIS frame's slot, version-gated. A static or
        // parked field re-sends nothing; a field the sim advanced re-sends the
        // live prefix only. This is the design's ONLY per-particle CPU cost.
        const std::uint64_t want = r.field->dataSerial();
        if (st.slotSerial[slot] != want) {
            st.slotSerial[slot] = want;
            const std::uint32_t live = std::min(r.field->liveCount(), st.capacity);
            if (live > 0) {
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
        d.posAddr = st.positions[slot].address;
        // A slot that was never filled holds garbage, so the first two frames
        // of a field's life reproject onto themselves (zero motion) rather than
        // streaking in from uninitialised memory.
        d.prevPosAddr = prevValid ? st.positions[prevSlot].address : d.posAddr;
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
            for (auto& b : st.counts) retireOrDestroy(b);
            for (auto& b : st.indirect) retireOrDestroy(b);
            retireOrDestroy(st.orientations);
            it = states_.erase(it);
        } else {
            ++it;
        }
    }
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
