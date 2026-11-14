// SceneState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the scene walk's snapshot and delta, and the entry list it produces: spans, slots, admission, per-entry
// moved/placed bits, motion matrices and stable ids.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct SceneState {
        // Per-entry "moved" bitmask, one bit per TLAS instance. Bit i set when
        // entry i's effective worldMatrix / pose / displacement changed since
        // last frame. Sized to ceil(meshCount/32) words at scene rebuild;
        // uploaded to meshMovedBitsBuffers[currentFrame] each frame so the
        // deferred shade can index by primaryInstanceId-1 to make the
        // FC-halving decision per-pixel instead of scene-wide.
        std::vector<uint32_t> meshMovedBits_;
        // Entries whose world matrix changed this frame through an edit that
        // said its rows were PLACED (InstancedMesh::markMatrixRows placed =
        // true: a pool's slot handed to another object), a bit each, cleared
        // with meshMovedBits_. Such a row is not in meshMovedBits_: its TLAS
        // row is marked through tlasRowDirty_, and its motion is identity with
        // the previous matrix taken as the new one (no motion vector, no
        // moving-object history reset in the shade).
        std::vector<uint32_t> entryPlacedBits_;

        std::vector<uint32_t> movedBitsShadeScratch_;// meshMovedBits_ less entryPlacedBits_ (uploadMeshMovedBits)
        std::array<Buffer, kFramesInFlight> meshMovedBitsBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> meshMovedBitsBufferCapacity{};
        // Per-entry sticky "recently moved" countdown backing GeometryDesc._pad —
        // holds the moved flag through zero-substep stall frames of a fixed-step
        // integrator under variable dt (a 1-frame flag gap disarms the _pad-gated
        // reproject guards and lets a moving object's reflection ghost into the
        // ground it vacated). Host-side only; see the stamping loop in
        // VulkanCoreFrame.cpp.
        std::vector<uint32_t> meshMovedSticky_;
        // Entries whose flag word in the draw records changed this frame, a
        // bit each like meshMovedBits_ and cleared with it: the LOD fades'
        // refresh and the sticky walk mark them (SpanDrawCache::rowWords).
        std::vector<uint32_t> drawRowFlagBits_;

        // Definition moved to VulkanSceneTypes.hpp.
        using SnapNode = vulkan::impl::SnapNode;
        static constexpr uint32_t kSnapKindMask   = 3u;
        static constexpr uint32_t kSnapKindOther  = 0u;
        static constexpr uint32_t kSnapKindMesh   = 1u;
        static constexpr uint32_t kSnapKindLine   = 2u;
        static constexpr uint32_t kSnapKindPoints = 3u;
        static constexpr uint32_t kSnapHasPos     = 4u;
        static constexpr uint32_t kSnapHasNorm    = 8u;
        static constexpr uint32_t kSnapWire       = 16u;
        static constexpr uint32_t kSnapOverlay    = 32u;
        static constexpr uint32_t kSnapTet        = 64u;
        static constexpr uint32_t kSnapParticle   = 128u;
        // material()->visible == false. The node is still recorded (so the
        // replay walk can detect a re-show) but contributes no MeshEntry /
        // LineEntry — matches three.js / GLRenderer, which drops a
        // material-hidden object from the render list. [[#mat-visible]]
        static constexpr uint32_t kSnapMatHidden  = 256u;
        // threepp::LOD node (kind Other + typed sn.lod view). Both walks run
        // the camera-driven level selection on it — three.js/GLRenderer
        // parity (projectObject calls lod.update(camera) as it projects).
        static constexpr uint32_t kSnapLod        = 512u;
        // Unlit transparent flat-color mesh (layered SVGs, in-scene UI
        // panels): folded into isOverlay so it renders through the raster
        // overlay pass, which draws in traversal order — i.e. paint order.
        // The traced blend path can neither sort overlapping transparent
        // surfaces nor tie-break exactly-coplanar layers (equal ray t), so
        // GL-parity layering is only reachable by rasterizing in order.
        static constexpr uint32_t kSnapUiBlend    = 1024u;
        // Mesh on VulkanRenderer::kSensorOnlyLayer (MeshEntry::sensorOnly).
        // A snapshot bit like every other routing decision, so enabling or
        // disabling the layer on a live mesh forces the re-expansion that
        // moves it in or out of the sensor group.
        static constexpr uint32_t kSnapSensorOnly = 2048u;
        // The node was effectively hidden when recorded (its own or an
        // ancestor's Object3D::visible was false). The walks enter a hidden
        // node's subtree unless it is, or sits under, a LOD.
        static constexpr uint32_t kSnapHidden     = 4096u;
        // The node sits under a LOD node: its visibility is the LOD's level
        // selection, so a flip takes the full re-expansion as before.
        static constexpr uint32_t kSnapUnderLod   = 8192u;
        // Object3D::resident when recorded. The expansion decides which hidden
        // meshes have entries, so a change re-expands.
        static constexpr uint32_t kSnapResident   = 16384u;
        // A node the edit-proportional path (diffSceneWalk, trySlotDelta) does
        // not add or remove: a line or points with an overlay entry, a LOD, a
        // Sky, a ParticleField, an overlay or particle-billboard mesh (none of
        // which a snapshot node's spanIndex names). A change at or under one
        // takes the full expansion.
        static constexpr uint32_t kSnapNoDelta    = 32768u;
        std::vector<SnapNode> sceneSnapshot_;
        // The full expansion's reuse of the last one (VulkanCoreScene.cpp,
        // reuseNode): the last walk's nodes, and their index by object (built
        // only when the walk meets a node out of the last walk's order).
        // Members so their storage survives between expansions.
        std::vector<SnapNode> prevSceneSnapshot_;
        std::unordered_map<const Object3D*, uint32_t> prevSnapIndex_;
        // THREEPP_VULKAN_WALK_CHECK=1: every expansion that reuses also walks
        // without reuse and compares the two, node for node; the mismatches
        // land here (and on stderr).
        uint64_t walkCheckMismatches_ = 0;
        // A mesh with entries was hidden or shown this frame (MeshEntry::hidden):
        // the TLAS fill, the frustum culls, the draw records and the emitter
        // walk re-read the flags. Set by the snapshot replay or the expansion,
        // consumed in ensureSceneBuilt.
        bool hiddenFlippedThisFrame_ = false;

        uint64_t snapshotGen_ = 1;           // bumped whenever sceneSnapshot_ is rebuilt or replaced
        uint64_t flatWatchGen_ = 0;
        std::vector<uint32_t> flatWatch_;
        // Per distinct material of the watched meshes: its routing bits and
        // version as recorded, and its meshes with spans (a version that moved
        // has their maps compared, as the replay does); per distinct geometry,
        // its attribute set's version.
        struct FlatMaterial {
            const Material* mat = nullptr;
            uint32_t first = 0;// a node with it, for its routing reads
            uint32_t bits = 0;
            unsigned int version = 0;
            std::vector<uint32_t> nodes;
        };
        std::vector<FlatMaterial> flatMats_;
        std::vector<std::pair<BufferGeometry*, unsigned int>> flatGeoms_;
        std::vector<int8_t> flatVis_;         // per node: -1 unknown, else effective visibility this frame

        // Previous-frame camera (proj_prev * view_prev) for primary-hit
        // reprojection. One UBO per frame-in-flight so updates don't race the
        // GPU. Per-instance motion matrices (prevWorld * inverse(curWorld)) live
        // in a single SSBO indexed by gl_InstanceCustomIndexEXT; the host keeps
        // an entry-aligned staging (motionScratch_) refreshed only for spans
        // that moved, with prev worlds in prevWorldByEntry_ (identity-remapped
        // by (Mesh*, instanceIndex) across full rebuilds) so each InstancedMesh
        // sub-instance has its own motion delta. First-frame / first-seen
        // entries are identity so reproject is a no-op.
        // Definitions moved to VulkanSceneTypes.hpp.
        using EntryKey     = vulkan::impl::EntryKey;
        using EntryKeyHash = vulkan::impl::EntryKeyHash;
        std::array<Buffer, kFramesInFlight> motionMatBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> motionMatBufferCapacity{};

        // Entry-index -> stable per-object instance id, built alongside
        // geomDescs and therefore in the SAME index space and with the SAME
        // lifetime: gl_InstanceCustomIndexEXT indexes both. If geomDescs are
        // valid for a frame, so is this; if entries were renumbered, both were
        // rebuilt together. Lets the lidar readback report the id the raster
        // Ids AOV reports rather than the raw TLAS instance index.
        std::vector<uint32_t> entryStableIds_;

        // Definition moved to VulkanGeometryState.hpp.
        using MeshFingerprint = vulkan::impl::MeshFingerprint;
        std::vector<MeshFingerprint> prevSceneFingerprint;
        // Per-entry record in TLAS-instance order from the last ensureSceneBuilt
        // call. renderFrame consumes this to compute per-instance motion matrices
        // after the in-flight fence has been waited (safe to write the
        // motionMatBuffers[currentFrame] HOST_VISIBLE buffer).
        //
        // Between two frames the entries still name the meshes the LAST frame
        // drew, and any of them may have been removed and freed since: the
        // caller owns its objects, and a Python app frees one the moment it
        // drops the last reference. ensureSceneBuilt re-validates the list
        // against the live scene (the snapshot replay compares address AND id
        // before it reads an object), so inside a frame every MeshEntry::mesh
        // is live. Code that runs between frames (setEmissiveCastsLight,
        // setStableCorrespondence, enableVertexInterop) must only COMPARE
        // entries, by EntrySpan::meshId or MeshEntry::lodGeomKey, and never
        // follow the pointer.
        std::vector<MeshEntry> lastVisibleEntries_;
        // Per-mesh spans over lastVisibleEntries_ (see EntrySpan). Rebuilt at
        // every full expansion; on lean frames the per-frame loops consult the
        // spans first and touch individual entries only for spans that changed.
        using EntrySpan = vulkan::impl::EntrySpan;
        std::vector<EntrySpan> entrySpans_;
        // Bumped wherever entrySpans_ is laid out anew (the rebuild, a slot
        // edit): VulkanRenderer::admitted's mesh-id index is rebuilt from it
        // only when it moved, so a streamer asking about every pending mesh
        // every frame pays a lookup each, not a scan of every span.
        uint64_t spansVersion_ = 0;
        std::unordered_map<unsigned int, uint32_t> admitIndex_;
        uint64_t admitIndexVersion_ = ~0ull;
        // Entry-aligned previous-frame world matrices for the motion-matrix
        // build (replaces the per-entry (Mesh*, instanceIndex) hash lookups —
        // ~11 ms/frame at 115k entries). Remapped by EntryKey identity across
        // full rebuilds in the same pass that remaps meshMovedSticky_.
        std::vector<std::array<float, 16>> prevWorldByEntry_;
        std::vector<uint8_t> prevWorldValidByEntry_;
        // Persistent motion-matrix staging (16 floats per entry, identity for
        // unmoved spans). motionScratchVersion_ bumps when any block changes;
        // per-FIF uploadedVersion skips the map+memcpy when the slot already
        // holds the current contents.
        std::vector<float> motionScratch_;
        uint32_t motionScratchVersion_ = 0;
        std::array<uint32_t, kFramesInFlight> motionUploadedVersion_{};
        // The entry ranges of motionScratch_ each frame-in-flight slot has not
        // received yet: one moving mesh re-sends its own blocks, not the
        // array (3.2 MB at 50k entries, every frame anything moved).
        std::array<DescDirtyRanges, kFramesInFlight> motionDirty_{};

        // Version of every input that shapes DrawInfo/indirect-cmd contents
        // EXCEPT the per-view cull bits (those live on the view as
        // cullVersion). Bumped by ensureSceneBuilt on any non-quiet frame and
        // by the moved-sticky stamp when a flag bit transitions; paired with
        // ViewContext::indirectBuiltSig to skip buildIndirectDrawData when a
        // FIF slot's device buffers already hold the exact same content.
        uint32_t drawInputsVersion_ = 1;
        // Count of entries whose moved-sticky countdown is nonzero — lets the
        // per-frame stamp skip its O(entries) walk on fully settled scenes.
        uint32_t stickyActiveCount_ = 0;
        // spansVersion_ at the last moved-sticky walk: a new layout walks every
        // entry once, after which entries with no mover and no countdown are skipped.
        uint64_t stickyWalkSpansVersion_ = ~0ull;
        // Per-entry cull mode cached from each upload of matDescs.
        //   Side::Front  → VK_CULL_MODE_BACK_BIT  (default fast path)
        //   Side::Back   → VK_CULL_MODE_FRONT_BIT
        //   Side::Double → VK_CULL_MODE_NONE
        // Stored as VkCullModeFlags directly so the gbuffer draw loop can
        // hand it straight to vkCmdSetCullMode. Indexed in lock-step with
        // lastVisibleEntries_.
        std::vector<VkCullModeFlags> lastVisibleCullMode_;
        bool sceneBuilt_ = false;

        // ── Stable entry slots (VulkanCoreSlots.cpp) ─────────────────────
        // plans/gpu-driven-instances.md, stage 2a. A mesh added or removed,
        // an InstancedMesh count changed or a mesh shown for the first time
        // edits rows of the entry index space in place (trySlotEdit) instead
        // of the structural rebuild: removed spans become FREE spans
        // (EntrySpan::isFree), new ones take free runs or the end, nothing
        // compacts, and the rows that did not change are not touched.
        // slotCapacity_: the entries every entry-indexed GPU buffer (the
        // geometry / material desc rings, the TLAS instance buffers and
        // storage, the motion and moved-bit buffers) was sized for at the
        // last structural rebuild, or grown to since (growSlotPool).
        uint32_t slotCapacity_ = 0;
        // VulkanRenderer::reserveEntrySlots: the least the pool is laid out or
        // grown to, by the next rebuild or slot edit.
        uint32_t reservedSlots_ = 0;
        // Set by trySlotEdit on the frame it edits; read by the lean tail
        // (a class change: full TLAS build, draw inputs, emitter re-walk).
        bool slotsEditedThisFrame_ = false;
        // The mesh every free slot names: one triangle with position and
        // normal, a default material, identity matrixWorld, never in a scene.
        std::shared_ptr<Mesh> slotTombstone_;

        // MaterialDesc::materialAssetIdx outlives one structural rebuild's
        // dense numbering: the rebuild leaves its map here (with each
        // material's id, the address-recycling guard) and the slot path
        // hands a material it has not seen the next index.
        std::unordered_map<const Material*, std::pair<unsigned int, uint32_t>> slotMatAssetIdx_;
        uint32_t slotNextMatAssetIdx_ = 0;

        // What the per-frame collectors ask of the scene graph, in scene order:
        // the lights and sprites of the visible part, and every SplatCloud,
        // hidden or not (see collectSplatClouds). Valid for one render():
        // render() clears root, so no list outlives the frame's graph.
        struct SceneCensus {
            const Object3D* root = nullptr;
            std::vector<Light*> lights;
            std::vector<Sprite*> sprites;
            std::vector<SplatCloud*> splats;
        };
        SceneCensus sceneCensus_;

        uint32_t admitPendingSpans_ = 0;

        bool stableIdsPending_ = false;
    };

}// namespace threepp::vulkan::impl
