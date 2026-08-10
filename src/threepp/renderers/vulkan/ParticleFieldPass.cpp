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

    // Counts: INDIRECT because §1.3's cleanest liveCount → instanceCount route
    // is a 4-byte copy into a VkDrawIndirectCommand, and a shader may also read
    // it as an SSBO.
    constexpr VkBufferUsageFlags kCountsUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    constexpr VmaAllocationCreateFlags kHostWrite =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

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
        st->slotSerial[s] = 0;// fresh allocation holds garbage
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

    descScratch_.clear();
    descScratch_.reserve(fields.size());

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

        FieldDescGpu d{};
        const auto& world = r.field->matrixWorld->elements;
        std::memcpy(d.world, world.data(), sizeof(d.world));
        d.posAddr     = st.positions[slot].address;
        d.prevPosAddr = 0;// phase 1
        d.oriAddr     = 0;// Config::orientations, phase 1
        d.attrAddr    = 0;// Config::attributes, phase 4
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
            it = states_.erase(it);
        } else {
            ++it;
        }
    }
}
