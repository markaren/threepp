// UploadState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns one-shot submits and async work: the one-shot batch, async one-shots, BLAS compaction and its retired
// storage, and the async grass prep.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_UPLOADSTATE_HPP
#define THREEPP_VULKAN_STATE_UPLOADSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct UploadState {
        // ── Batched one-shots ────────────────────────────────────────────────
        // A structural scene rebuild admits every NEW tile geometry (BLAS build
        // + prevVertex seed) and its albedo DataTexture (staging copy + mip
        // blit) through beginOneShot/endAndSubmitOneShot — one vkQueueSubmit +
        // vkQueueWaitIdle EACH. A burst of tiles then serialises a dozen queue
        // drains into one frame (measured ~37 ms on fjord flight). While a batch
        // is open, beginOneShot hands back a single shared command buffer and
        // endAndSubmitOneShot only closes the caller's recording (no submit);
        // flushOneShotBatch submits the whole batch once and waits. Transient
        // resources (BLAS scratch, image staging) that the callers would free
        // right after their submit are parked in oneShotBatchGarbage_ and
        // reclaimed after the single wait. Same one-shot semantics from the
        // caller's view (work is done + resources freed by the time the batch is
        // flushed), one drain instead of N. Callers must be pure transfer/AS
        // builds with no read-back before the flush — the tile admit path is.
        bool            oneShotBatch_ = false;
        VkCommandBuffer oneShotBatchCb_ = VK_NULL_HANDLE;
        std::vector<Buffer> oneShotBatchGarbage_;

        struct AsyncCompaction {
            VkQueryPool pool = VK_NULL_HANDLE;
            // Each record by its geometry (blasCache's key) and the structure
            // queried: a record evicted or rebuilt since is passed over.
            std::vector<std::pair<const BufferGeometry*, VkAccelerationStructureKHR>> records;
        };
        struct AsyncOneShot {
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            AsyncCompaction compaction;
        };
        std::vector<AsyncOneShot> asyncOneShots_;

        bool blasRereferenced_ = false;

        // ── Static BLAS compaction ───────────────────────────────────────────
        // A static record (built once, traced for its lifetime) is built
        // PREFER_FAST_TRACE | ALLOW_COMPACTION, its compacted size is queried
        // in the build's own command buffer, and it is then copied with
        // COPY_MODE_COMPACT into storage of exactly that size; the build-size
        // pair is freed. Deformer and interop records are refit in place and
        // keep the updatable lineage, uncompacted.
        //
        // THREEPP_VK_BLAS_COMPACT=0, read once at construction, turns the
        // compaction off (static builds then stay at their build size).
        // blasCompact_ is also false when the device does not resolve the two
        // compaction entry points.
        bool blasCompact_ = false;
        // Compacted-size queries, kBlasCompactPoolQueries per pool; pools are
        // appended as a larger batch needs them and live until teardown.
        // Slot k of one compaction round is query k % N of pool k / N.
        static constexpr uint32_t kBlasCompactPoolQueries = 256;
        std::vector<VkQueryPool> blasCompactPools_;
        // Batch mode: static records built into the open one-shot batch,
        // compacted by flushOneShotBatch after its wait. destroyBlasRecord
        // takes a record off this list.
        std::vector<BlasRecord*> blasCompactPending_;
        // A batch-built record's uncompacted address was already pushed into
        // ensureSceneBuilt's TLAS instance array when the batch flushes, so the
        // flush records old -> new here and keeps the old pair alive (retired),
        // which also keeps a later build in the same pass from being handed
        // the old address. applyBlasCompactionRemap rewrites the instances and
        // frees the retired pairs.
        std::unordered_map<VkDeviceAddress, VkDeviceAddress> blasAddressRemap_;
        struct RetiredBlas {
            VkAccelerationStructureKHR as = VK_NULL_HANDLE;
            Buffer storage;
        };
        std::vector<RetiredBlas> blasCompactRetired_;

        // ── Static geometry placement ────────────────────────────────────────
        // A static record's vertex / normal / uv / color / index / rtIndex
        // buffers (and its auto-LOD level index buffers) are device-local and
        // filled through a staging copy: allocated host-visible, they took the
        // 256 MiB BAR window first and spilled the rest to system memory on a
        // desktop without resizable BAR, and every G-buffer vertex pull, ray
        // hit fetch and BLAS input read then crossed PCIe each frame.
        // Deformer and interop records keep host-visible vertex and normal
        // buffers (rewritten per frame); their uv, color, index and rtIndex
        // are device-local too. Every host write into a geometry buffer goes
        // through stageUpload, which memcpys into a host-mappable allocation
        // as before and queues the bytes for a staging copy otherwise, so no
        // writer depends on where the buffer landed.
        struct StagedUploads {
            struct Copy {
                VkBuffer     dst       = VK_NULL_HANDLE;
                VkDeviceSize dstOffset = 0;
                VkDeviceSize srcOffset = 0;
                VkDeviceSize bytes     = 0;
            };
            std::vector<uint8_t> data;
            std::vector<Copy>    copies;
        };

        // buildBlasFor in two halves. prepareBlas validates the geometry, makes
        // the record's buffers and queues their bytes in p.up: any thread (the
        // geometry's arrays and the allocator; the caller reads interopTarget
        // from forceUnpackedGeoms_). finishBlas records the copies (from
        // p.staging when the preparing thread made it) and the acceleration
        // structure's build into the open one-shot or batch: the render thread.
        // discardBlasPrep frees a prepared record nothing has recorded yet.
        struct BlasPrep {
            std::unique_ptr<BlasRecord> rec;
            StagedUploads up;
            Buffer staging;
            VkDeviceSize vbBytes = 0;
            uint32_t vertexCount = 0, primitiveCount = 0, indexCount = 0;
            bool indexed = false, packIdx = false, rt = false, updatable = false, interopTarget = false;
            // With ray query: the structure, created, and what its build needs.
            VkAccelerationStructureGeometryKHR blasGeom{};
            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            Buffer scratch;
            uint32_t buildPrims = 0;
            bool compactThis = false;
        };

        // A hidden GrassMesh field's state prepared on a worker thread from its
        // slot edit on (beginGrassPrep, from beginHiddenAdmission): its record's
        // buffers and its rest-pose and height-fraction inputs, every byte in
        // one filled staging buffer. ensureGrassState takes it (waiting for it
        // only when the field is shown before it is ready) and records just the
        // copies and the build; the pump admits a field once it is ready.
        struct GrassPrepData {
            BlasPrep blas;
            Buffer restPos, heightFrac;
            uint32_t vertexCount = 0;
            float bladeHeight = 0.f;// estimated, field-local m (GrassMeshState::bladeHeight)
            bool ok = false;
        };
        struct GrassPrep {
            std::shared_ptr<BufferGeometry> geom;// alive while the worker reads it
            vulkan::impl::MeshOwner owner;
            std::future<std::unique_ptr<GrassPrepData>> job;
        };
        std::unordered_map<const GrassMesh*, GrassPrep> grassPreps_;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_UPLOADSTATE_HPP
