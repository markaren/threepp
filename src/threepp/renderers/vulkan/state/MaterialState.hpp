// MaterialState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns materials and textures: the material and geometry descriptor tables with their dirty ranges, the
// deduplicated material table, the bindless texture array, its samplers and the async texture prep.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct MaterialState {
        // MaterialDesc layout lives in vulkan_shared.h (the same file the GLSL
        // deferred-shade / gbuffer / LIDAR shaders pull in via #include).
        // Bringing it into Impl scope here keeps the existing
        // `MaterialDesc md{};` call sites unchanged.
        using MaterialDesc = threepp::vulkan_pt::MaterialDesc;

        // Per-frame-in-flight GeometryDesc storage — same fence-gated ring as
        // materialDescsBuffers below. Ringed for auto-LOD: a level switch
        // repoints GeometryDesc::indexAddress/indexed, and a single shared
        // buffer would need a vkDeviceWaitIdle on every switch frame to patch
        // it in place. Hot path stages in geomDescsCached_ +
        // marks geomDescsDirty_ (markGeomDescsDirty / ...Whole); renderFrame's
        // flushGeometryDescsIfDirty memcpys the marked ranges into this frame's
        // slot once its fence has signaled.
        std::array<Buffer, kFramesInFlight> geometryDescsBuffers{};
        // A forest's moved-sticky transitions and auto-LOD switches touch a few
        // thousand scattered entries of 200k+: ranges, not the 15 MB array.
        std::array<DescDirtyRanges, kFramesInFlight> geomDescsDirty_{DescDirtyRanges{.maxRanges = 1024},
                                                                     DescDirtyRanges{.maxRanges = 1024}};
        // Per-frame-in-flight material table storage; one buffer per slot so
        // the upload after a fence wait races nothing. The hot path stages new
        // records in `matTable_` and marks the RECORD ranges it touched
        // (markMatDescsDirty); renderFrame's flushMaterialDescsIfDirty memcpys
        // those ranges into `materialDescsBuffers[currentFrame]` once the
        // fence has signaled (= GPU done with this slot). Descriptor sets are
        // bound per-frame (set idx = f*imageCount_+k → buffer[f]) so the
        // binding stays valid across the swap.
        std::array<Buffer, kFramesInFlight> materialDescsBuffers{};
        std::array<DescDirtyRanges, kFramesInFlight> matDescsDirty_{};

        // ── The material table ──────────────────────────────────────────────
        // One MaterialDesc per Material the scene draws, not one per entry: an
        // entry names its record in the top bits of its GeometryDesc's flags
        // (geomMatIdx), and every shader reads it through entryMat, which
        // multiplies in the instance colour the entry's GeometryDesc carries
        // (setGeomInstanceColor). The desc is a pure function of the Material
        // (buildMaterialDesc), so the Material is the key. Per entry, a
        // streamed cell of a thousand instances of one material cost a
        // thousand 600-byte copies to pack and upload; here it costs a
        // thousand indices and at most one record.
        //
        // Record 0 is the all-zero desc, what an entry without a material (an
        // overlay, a tombstone, a span still being admitted) has always read.
        // Every other record counts the entries that name it: every write of
        // an entry's index goes through setEntryMat or assignEntryGeom. A
        // record whose count reaches zero goes dormant, key and content kept:
        // a streamed world brings the same materials back with the next
        // cell, which then finds them written and uploads nothing.
        // Only a full table reclaims a dormant record, the longest-dormant
        // first, and only when none is left do the GPU buffers grow. The
        // structural rebuild starts the table afresh in entry order.
        struct MatTableKey {
            const Material* mat = nullptr;
            unsigned int matId = 0;// Material::id: a new material at a dead one's address is a new key
            bool operator==(const MatTableKey& o) const {
                return mat == o.mat && matId == o.matId;
            }
        };
        struct MatTableKeyHash {
            size_t operator()(const MatTableKey& k) const {
                return std::hash<const void*>{}(k.mat) ^ (size_t(k.matId) * 0x9E3779B97F4A7C15ull);
            }
        };
        std::vector<MaterialDesc> matTable_;
        std::vector<MatTableKey> matTableKeys_;// per record; meaningless where the count is 0
        std::vector<uint32_t> matTableRefs_;   // entries naming each record (record 0 uncounted)
        // The pass a record was last derived in (matTableSlot): a record is
        // derived once per pass however many entries share it, and is not
        // reclaimed in that pass (its entries may not have been written yet).
        std::vector<uint32_t> matTableStamp_;
        // Records gone dormant, oldest first, each with the count of times it
        // has: an entry whose record was revived and has gone dormant again
        // since is stale and skipped.
        std::deque<std::pair<uint32_t, uint32_t>> matTableDormant_;
        std::vector<uint32_t> matTableDormantSeq_;
        std::unordered_map<MatTableKey, uint32_t, MatTableKeyHash> matTableIndex_;
        uint32_t matTableCapacity_ = 0;// records the GPU buffers hold
        uint32_t matTableEpoch_ = 1;
        // The structural rebuild fills the table before it sizes the buffers:
        // no growth while it does.
        bool matTableBuilding_ = false;
        // The last key resolved this pass: an instanced span's entries share
        // their material and skip the hash lookup.
        MatTableKey matTableMemoKey_{};
        uint32_t matTableMemoSlot_ = 0;// 0 = no memo

        // Bindless material textures (albedo only for v1). The descriptor set
        // exposes a fixed-size sampler2D[] at binding 8; closest_hit indexes
        // into it via mdesc.albedoTexIndex. Slot 0 is a 1×1 white default so
        // materials without an albedo map can still bind a valid descriptor.
        // Cache key is Texture* — the same Texture across multiple meshes only
        // uploads once. All material textures share the policy samplers (see
        // textureSampler_ below); the only per-texture sampler state honoured
        // is ClampToEdge wrap (materialTexClampUV_) — per-texture FILTER and
        // mirrored wrap remain a v2 concern.
        // kMaxMaterialTextures comes from the shared header (vulkan_shared.h,
        // included near the top of this file). Editing it there propagates to
        // every shader on the next clean rebuild.
        std::vector<Image2D> materialTextures;// owns image + view (sampler is shared)
        // Slot-parallel to materialTextures: 1 = this texture asked for
        // ClampToEdge on BOTH axes, so its binding uses the clamp variant of
        // the policy sampler. REPEAT is wrong for edge-inclusive atlases —
        // a terrain tile's splat bilinear/mip taps at uv 0/1 wrap around and
        // blend in the OPPOSITE edge, painting a seam line along every tile
        // border (metres wide at coarse mips). Maintained by
        // ensureMaterialTexture; only read for slots with a live view.
        std::vector<uint8_t> materialTexClampUV_;
        // Entry-list / cache records moved to VulkanSceneTypes.hpp (their doc
        // comments travel with them).
        using CachedTexture = vulkan::impl::CachedTexture;
        std::unordered_map<const Texture*, CachedTexture> textureCache;
        std::vector<uint32_t> freeTextureSlots;// slots reclaimed by prune
        // Slots whose cache entry was found STALE at lookup (the original
        // Texture died and a brand-new one was allocated at the recycled
        // address — the raw-pointer key collides). The image cannot be
        // destroyed at detection time: in-flight frames + bound descriptor
        // sets may still reference it, and ensureMaterialTexture's callers
        // include the lean material-patch path that deliberately avoids a
        // device drain. The structural-rebuild prune (post-drain) destroys
        // these and returns the slots to freeTextureSlots.
        std::vector<uint32_t> retiredTextureSlots_;
        // The policy samplers every material-texture binding chooses between
        // (fillMaterialTextureInfos → materialSampler()): 16× aniso is the
        // default regardless of raster jitter (dropping to isotropic under
        // TAA/DLSS/FSR does not fix distance shimmer, it only mip-blurs
        // grazing angles). The isotropic twins exist for
        // the explicit setTextureAnisotropy(1) / THREEPP_VK_ANISO=1 override.
        // Each policy exists in a REPEAT and a CLAMP_TO_EDGE flavour —
        // clamp-tagged textures (materialTexClampUV_) get the clamp twin.
        // NOTE: the per-image samplers buildSampledImage2D creates are NOT
        // bound for material textures — these are.
        VkSampler textureSampler_ = VK_NULL_HANDLE;        // 16× aniso (default)
        VkSampler textureSamplerIso_ = VK_NULL_HANDLE;     // isotropic (override=1 only)
        VkSampler textureSamplerClamp_ = VK_NULL_HANDLE;   // 16× aniso, clamp-to-edge
        VkSampler textureSamplerIsoClamp_ = VK_NULL_HANDLE;// isotropic, clamp-to-edge
        // setTextureAnisotropy override: 0 = AUTO (the policy above), 1..16
        // forces that level. Values other than 1/16 use a lazily-created
        // custom sampler; a replaced custom is PARKED (in-flight frames and
        // bound descriptor sets may still reference it; samplers are tiny)
        // and destroyed at teardown.
        float textureAnisoOverride_ = 0.f;
        VkSampler textureSamplerCustom_ = VK_NULL_HANDLE;     // repeat flavour
        VkSampler textureSamplerCustomClamp_ = VK_NULL_HANDLE;// clamp twin, same level
        float textureSamplerCustomAniso_ = 0.f;
        std::vector<VkSampler> parkedSamplers_;

        // Host mirror of the last-uploaded GeometryDesc array (entries-
        // indexed, same layout as the `geomDescs` local built in the full
        // rebuild). The lean auto-LOD path patches indexAddress/indexed in
        // place here; the per-frame ring flush (flushGeometryDescsIfDirty)
        // carries it to the GPU stall-free.
        std::vector<GeometryDesc> geomDescsCached_;

        // The same upload in its two halves. prepareMaterialImage is the CPU
        // work (the RGBA8 normalisation, the mip chain, the BC7 encode: most
        // of a new texture's cost) and touches no device state, so it can run
        // on a worker thread;
        // createPreparedImage is the device half (render thread).
        struct PreparedMaterialImage {
            bool ok = false;
            bool levelsPath = true;// every level built (BC7, odd sizes) vs RGBA8 with a blit chain
            uint32_t w = 0, h = 0;
            VkFormat format = VK_FORMAT_UNDEFINED;
            std::vector<std::vector<std::uint8_t>> levels;
            std::vector<unsigned char> rgba;
            VkSamplerAddressMode addrU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            VkSamplerAddressMode addrV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            std::string name;
        };

        struct PreparingTexture {
            std::shared_ptr<Texture> tex;// held while the job runs
            unsigned int version = 0;
            std::future<PreparedMaterialImage> job;
        };
        std::unordered_map<const Texture*, PreparingTexture> preparingTextures_;

        int bc7Supported_ = -1;
    };

}// namespace threepp::vulkan::impl
