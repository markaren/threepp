// VulkanGeometryState — the per-mesh GPU state the Vulkan renderer keeps
// alongside each geometry (BLAS records, the deform-on-GPU states, the
// descriptor-table mirrors) plus the automatic-LOD machinery's job/result
// types. Split out of VulkanCoreImpl.hpp; each struct is aliased back into
// VulkanRenderer::Impl at its original spot so every reference site — Impl's
// own methods and the VulkanCore*.cpp TUs — is unchanged.
//
// Declaration order matters here: SkinnedMeshState / TetMeshState /
// MorphedMeshState / DisplacedMeshState / GrassMeshState and GeomRefreshOp all
// hold a BlasRecord, so BlasRecord stays first.

#ifndef THREEPP_VULKAN_GEOMETRY_STATE_HPP
#define THREEPP_VULKAN_GEOMETRY_STATE_HPP

#include "VulkanImplCommon.hpp"
#include "VulkanResources.hpp"
#include "SkinningPipeline.hpp"
#include "TetSkinningPipeline.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/utils/GeometryLod.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace threepp::vulkan::impl {

    // Per-geometry BLAS + the buffers that back its build inputs. Vertex /
    // index / normal / uv buffers are kept alive past the build so the
    // closest-hit shader can sample them via buffer-reference pointers.
    // `uv.handle == VK_NULL_HANDLE` for geometries without a UV attribute;
    // the matching GeometryDesc.uvAddress is then 0 and closest_hit treats
    // the surface as untextured.
    struct BlasRecord {
        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        Buffer storage;
        Buffer vertex;
        Buffer index;// .handle == VK_NULL_HANDLE for non-indexed geometry
        Buffer normal;
        Buffer uv;   // .handle == VK_NULL_HANDLE if geometry has no "uv"
        // Per-vertex RGB color (BufferGeometry "color" attribute, itemSize 3).
        // .handle == VK_NULL_HANDLE when the geometry has no usable "color".
        // Surfaced to the shaders via GeometryDesc::colorAddress / DrawInfo::
        // colorAddr, gated on the material's vertexColors flag at fill time.
        Buffer color;
        // Which of the optional attribute buffers hold PACKED data instead
        // of tightly-packed float (bit 0: normal snorm16x4 — 8 B/vertex,
        // bit 1: uv unorm16x2 — 4 B/vertex, bit 2: color unorm8x4 —
        // 4 B/vertex). Set by buildBlasFor for static geometry only;
        // deforming records (skinned / tet / displaced / grass / morphed)
        // stay float because their compute passes rewrite these buffers
        // every frame. Mirrored to GeometryDesc::packedAttrs and
        // DrawInfoGpu::packedAttrs; shaders branch per fetch.
        uint32_t packedMask = 0;
        // True for FFT-displaced ocean meshes — gates world-space foam +
        // thin-shell water shading downstream (surfaced to the shaders via
        // GeometryDesc::foamAddress, now a 0/1 flag). Replaces a per-vertex
        // foam buffer that used to live on the BLAS but was never read for
        // its contents — only its non-null address served as this marker.
        bool isOceanSurface = false;
        // Previous-frame vertex positions, allocated for skinned + displaced
        // meshes only. Used by the hybrid raster prepass to compute
        // per-vertex motion vectors (skinned/displaced surfaces deform
        // each frame; the rigid-body motionMat alone produces zero motion
        // and ghosts under TAA + the deferred shade's temporal
        // accumulation). Static meshes
        // bind .vertex at the prev-pos attribute slot — inPrevPos == inPos
        // so the motion is identity-rigid as before.
        Buffer prevVertex;
        VkDeviceAddress address = 0;
        // Liveness tag: detects dangling-pointer reuse when a BufferGeometry
        // is destroyed (model unloaded) and the C++ allocator hands the
        // same address to a different geometry. Pruned in ensureSceneBuilt.
        std::weak_ptr<BufferGeometry> liveCheck;
        // Attribute-version snapshot at build time. When the user mutates
        // vertex data in-place and calls needsUpdate(), the composite
        // version changes and the BLAS is refreshed (in-place rebuild if
        // counts match, full evict+rebuild if topology changed).
        unsigned int geomVersion = 0;
        uint32_t vertexCount = 0;
        uint32_t indexCount  = 0;
        // BufferGeometry::drawRange snapshot, compared every frame by the
        // enqueue loop in ensureSceneBuilt. drawRange is not covered by any
        // BufferAttribute version, so without this the DrawInfo skip
        // signature would reuse commands built for a stale range.
        // Initialised to BufferGeometry's own default so an ordinary mesh
        // never takes the bump. (int, not uint: DrawRange members are int.)
        int lastDrawStart = 0;
        int lastDrawCount = std::numeric_limits<int>::max() / 2;
        // The primitive count the current AS was last built with. MODE_UPDATE
        // is only legal against an identical count, so a drawRange change
        // forces MODE_BUILD — the AS was sized for full capacity, which the
        // spec allows to be built with any smaller count.
        uint32_t blasBuiltPrims = 0;
        // The flags of that build. An update must carry its source build's
        // flags, and a record can change its preferred flags across paths
        // (interop wants PREFER_FAST_BUILD, the CPU routes FAST_TRACE) — a
        // mismatch forces MODE_BUILD to re-establish the lineage.
        VkBuildAccelerationStructureFlagsKHR blasBuiltFlags = 0;
        // `storage` was sized for the max of BOTH flag lineages
        // (PREFER_FAST_TRACE and PREFER_FAST_BUILD size queries), so a
        // per-frame interop rebuild may legally take FAST_BUILD. Set by
        // buildBlasFor for interop-marked geometries only — the lineages
        // are different BVH formats with different footprints, and
        // building the one the storage was not sized for overruns the
        // structure (see recordDynamicGeomRefits). enableVertexInterop
        // consults it to decide whether an already-unpacked record still
        // needs the drain + rebuild that re-sizes the storage.
        bool storageFitsFastBuild = false;
        // The refit's live FAST_BUILD fits-check failed once and said so on
        // stderr; don't repeat it every frame (recordDynamicGeomRefits).
        bool fastBuildDeniedWarned = false;
        // Persistent scratch buffer for per-frame refit. Sized to
        // buildScratchSize on first use (which is always >= updateScratchSize),
        // reused every frame to avoid the create+destroy pair that
        // dominated refreshGeomBlas's cost. Empty until the first
        // refreshGeomBlas; cleaned up in deinit / prune paths.
        Buffer blasScratch{};
        VkDeviceSize blasScratchSize = 0;
        // Per-frame BLAS-refit state — same pattern as SkinnedMeshState /
        // DisplacedMeshState. refreshGeomBlas refits via MODE_UPDATE for 63
        // of every 64 frames; the periodic full rebuild keeps the BVH
        // balanced under sustained geometry mutation (PhysX soft bodies,
        // dynamic ParticleSystems).
        uint32_t blasRefitCounter = 0;
        static constexpr uint32_t kBlasFullRebuildInterval = 64;
        // Position-buffer byte size, captured at build, for the prevVertex
        // re-sync copy below.
        VkDeviceSize vbBytes = 0;
        // Set when refreshGeomBlasBatch snapshots OLD positions into
        // prevVertex on an in-place vertex update. The change frame
        // correctly reports old→new motion; the NEXT clean frame we re-sync
        // prevVertex := vertex so a now-static re-rolled mesh returns to
        // prevVertex == vertex (zero motion), matching initial-build
        // geometry. Without this, prevVertex stays frozen at the pre-update
        // positions and the mesh emits a constant bogus per-vertex motion
        // vector every frame → persistent denoiser/TAA history rejection
        // (runtime-updated geometry stays noisy / visibly shakes).
        bool prevVertexResyncPending = false;

        // ── Per-frame dynamic residency (graduated CPU deformers) ───
        // A plain mesh whose attributes are rewritten + needsUpdate()ed
        // EVERY frame (Flock's merged bird mesh, CPU trails) used to pay
        // the occasional-edit price every frame: one device-wide drain
        // plus two submit+wait one-shots (refreshGeomBlasBatch). After
        // kDynamicGraduationStreak consecutive dirty frames the record
        // graduates to the residency the skinned/displaced/grass
        // deformers already have — staging upload + GPU copy + BLAS
        // refit recorded into the frame command buffer, zero drains
        // (recordDynamicGeomRefits). Graduation is one-way for the
        // record's lifetime; a topology change destroys the record and
        // its replacement starts cold.
        uint64_t lastDirtyFrame = 0;// frameSerial_ of the latest geom-dirty frame
        uint32_t dirtyStreak = 0;   // consecutive dirty frames ending at lastDirtyFrame
        bool perFrameDynamic = false;
        static constexpr uint32_t kDynamicGraduationStreak = 3;
        // Auto-LOD stays out of a recently-edited geometry's way: a chain
        // enqueued while the mesh deforms is guaranteed stale on arrival
        // (drainLodResults drops it, selection re-enqueues, forever).
        // Selection only considers a record this many frames after its
        // last edit — and never a graduated one.
        static constexpr uint64_t kLodDirtyQuietFrames = 8;
        // Host-visible staging for the graduated path: kFramesInFlight
        // slots of dynStagingSlotBytes (positions then normals, in the
        // buffer's own — possibly packed — format). Slot `currentFrame`
        // is written on the CPU at record time, i.e. past that slot's
        // fence wait, so the write can never race the GPU copy a still-
        // in-flight frame issued from the same slot.
        Buffer dynStaging{};
        VkDeviceSize dynStagingSlotBytes = 0;
        // The graduated path snapshots vertex→prevVertex on every dirty
        // frame; the first CLEAN frame afterwards re-syncs prevVertex to
        // the settled positions (same shake-forever failure mode as
        // prevVertexResyncPending above) — recorded into the frame cb,
        // never through the draining host-side resync pass.
        bool dynPrevResyncPending = false;

        // ── Zero-copy vertex interop (enableVertexInterop) ──────────
        // A foreign device producer (CUDA: Warp, PhysX, torch) fills these
        // EXPORTED allocations, and recordDynamicGeomRefits copies them into
        // rec.vertex / rec.normal at head of frame.
        //
        // The copy is the design — ParticleField F6's shape
        // rather than the soft-body swap's. `vertex` has seven consumers,
        // five of them by DEVICE ADDRESS (the bindless raster pull's
        // DrawInfo::posAddr, the BLAS build/refit input, GeometryDesc::
        // vertexAddress in chit/ray-query/probe/lidar, the prev-frame
        // fallbacks) — and createExternalBuffer allocates its dedicated
        // memory WITHOUT VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, so an
        // exported buffer can never carry one. Substituting the export FOR
        // rec.vertex is therefore not merely awkward, it is invalid. The
        // export stays a STORAGE|TRANSFER_SRC copy source that nothing binds,
        // exactly as ParticleFieldPass::kExternalPositionUsage argues, and
        // every downstream address keeps pointing at the same VMA allocation
        // it always did. Cost: one device→device copy per attribute per frame.
        //
        // Second, independent reason to prefer the copy here: under the swap
        // the foreign write would land in a buffer a BLAS BUILD reads, where a
        // torn or non-finite read is VK_ERROR_DEVICE_LOST rather than one
        // blended sim step.
        vulkan::ExternalBuffer posExt{};// positions  — sized to vertex.size
        vulkan::ExternalBuffer nrmExt{};// normals    — sized to normal.size
        // The producer's synchronous device write, invoked once per frame
        // post-fence / pre-record in recordDynamicGeomRefits (the same
        // contract as TetMeshState::tetPosExternalCopy and ParticleField's
        // deviceCopy: it MUST have completed when it returns).
        std::function<void()> externalCopy;
        bool interop = false;
        // W4: run the GPU sanitize dispatch over posExt before the copy into
        // rec.vertex. The CPU finiteness scan that guards every other BLAS
        // build path reads the host attribute array, which under interop is
        // stale or empty — this is its replacement, and it is ON by default
        // because a non-finite position reaching a BLAS build is a device
        // reset, not a warning. Producers that have earned the trust can turn
        // it off for the dispatch back.
        bool interopValidate = true;
        // Descriptor set naming posExt for that dispatch (one per record,
        // allocated from VertexSanitizePipeline's pool at enable time).
        VkDescriptorSet sanitizeDS = VK_NULL_HANDLE;

        // ── Automatic mesh LOD (setAutoLod) ─────────────────────────
        // One simplified INDEX buffer + its own static BLAS per chain
        // level, built beyond this record's own (LOD0) vertex/normal/uv/
        // color buffers — those stay shared across every level (meshopt
        // never moves or reorders vertices, only drops/reweaves
        // triangles), so a level swap is purely an index-buffer +
        // BLAS-reference change, consumed identically by the raster
        // vertex-pulling path (buildIndirectDrawData) and the TLAS
        // instance fill (ensureSceneBuilt).
        struct LodLevel {
            VkAccelerationStructureKHR as = VK_NULL_HANDLE;
            Buffer storage;
            Buffer index;// uint32, indexes into THIS record's own `vertex`
            VkDeviceAddress address = 0;// BLAS device address
            uint32_t indexCount = 0;
            // Absolute object-space screen-space-error bound for this
            // level (see threepp::geometrylod::Level::error). Multiplied
            // by world scale + px/unit at selection time.
            float errorBound = 0.f;
        };
        // Chain-generation state. None → not yet attempted (or the prior
        // attempt's geometry version is stale); Queued → a background
        // job is in flight; Ready → lodLevels is usable; Failed →
        // generateChain produced nothing (too small/degenerate) — never
        // retried for this geomVersion. Touched ONLY on the main thread
        // (the worker only ever sees copied job data, never the record).
        enum class LodState : uint8_t { None, Queued, Ready, Failed };
        LodState lodState = LodState::None;
        // Index i backs MeshEntry::lodLevel == i+1 (level 0 is this
        // record's own vertex/index/etc — never stored here).
        std::vector<LodLevel> lodLevels;
    };

    // Resolved index/AS data for whichever LOD level an entry currently
    // selects. Falls back to the record's own LOD0 buffers when
    // lodLevel==0 or that level isn't built yet — the ONLY two call
    // sites that pick per-entry geometry for rendering (buildIndirectDrawData's
    // raster path, and the TLAS instance / GeometryDesc fill below) go
    // through this so the rasterized and ray-traced views of an entry
    // can never disagree about which triangles exist.
    struct LodGeomSel {
        VkDeviceAddress asAddress;
        VkDeviceAddress indexAddress;
        uint32_t indexCount;
        // A selected LEVEL is always an indexed draw — even when the
        // record itself is non-indexed soup (the chain generator welds
        // soup into canonical indices whose VALUES are original vertex
        // ids, so levels index the record's unchanged soup vertex
        // buffer). Fallback = the record's own indexed-ness. Consumers
        // (DrawInfo::indexed, GeometryDesc::indexed) must read THIS,
        // never rec.index.handle directly.
        bool indexed;
    };

    // Per-SkinnedMesh deformed-geometry BLAS. Unlike static meshes, skinned
    // meshes can't share BLAS even when they share BufferGeometry — each
    // instance has its own pose. Vertex/normal buffers are host-mapped and
    // overwritten with CPU-skinned positions/normals each frame the bones
    // change; the BLAS is then rebuilt in-place against the same AS handle
    // (and storage) so its address — and the TLAS reference to it — remain
    // valid. prevBoneMats is the dirty-detection key (memcmp against the
    // current Skeleton::boneMatrices).
    struct SkinnedMeshState {
        std::unique_ptr<BlasRecord> blas;
        std::vector<float> prevBoneMats;
        std::weak_ptr<BufferGeometry> liveCheck;

        // GPU skinning input buffers — populated once at ensureSkinnedBlas
        // time and reused every frame. Output is the BLAS's own vertex /
        // normal buffer (overwritten by the compute dispatch).
        Buffer baseVertex   {};// vec3<float>, count = vertexCount
        Buffer baseNormal   {};// vec3<float>, count = vertexCount
        Buffer skinIndex    {};// vec4<float>, count = vertexCount
        Buffer skinWeight   {};// vec4<float>, count = vertexCount
        // Bone matrices buffer layout: [bindMatrix, bindMatrixInverse,
        // bones[0]...bones[N-1]] as mat4s. Host-visible so the per-frame
        // upload is a small memcpy. bindMatrix is written once at
        // allocation; bindMatrixInverse and the bones[..] portion are
        // rewritten every frame.
        //
        // A RING, not one buffer, and that is the whole fix for skinned
        // animation juddering on Vulkan while GL ran smooth at the same 60
        // FPS: the memcpy happens in ensureSceneBuilt, which runs BEFORE
        // renderFrame waits on this frame's fence, so a single buffer gets
        // overwritten while an earlier frame's skinning dispatch is still
        // reading it — that frame skins with the wrong pose and the motion
        // shows one pose twice, then jumps. refreshSkinnedBlas advances
        // boneSlot once per frame and writes only that slot;
        // recordCommandBuffer dispatches with the matching descriptor set.
        // See SkinningPipeline::kBoneSlots for why it is +1 deep.
        static constexpr uint32_t kBoneSlots = vulkan::SkinningPipeline::kBoneSlots;
        static_assert(kBoneSlots >= kFramesInFlight + 1,
                      "bone-matrix ring must cover all in-flight frames plus the one being recorded");
        std::array<Buffer, kBoneSlots> boneMatrices {};
        uint32_t boneSlot = 0;// slot written this frame (advanced by refreshSkinnedBlas)
        uint32_t vertexCount    = 0;
        uint32_t boneCount      = 0;
        uint32_t primitiveCount = 0;// for per-frame BLAS rebuild
        bool     indexed        = false;
        // Per-mesh descriptor sets wiring all of the above + the BLAS
        // output buffers into the skinning pipeline's set 0. One per ring
        // slot; identical except for binding 4, the bone-matrix buffer.
        std::array<VkDescriptorSet, kBoneSlots> skinDescSet {};
        // Persistent scratch buffer for BLAS rebuild. Sized at the
        // first ensureSkinnedBlas, reused every frame. Avoids per-frame
        // alloc/free that was the original oneshot cost.
        Buffer blasScratch {};
        VkDeviceSize blasScratchSize = 0;
        // Per-frame BLAS-refit state. Same pattern as DisplacedMeshState:
        // initial AS is BUILD (buildBlasFor) with ALLOW_UPDATE, then per-
        // frame refits via MODE_UPDATE for 63 of every 64 frames. Periodic
        // full rebuild keeps the BVH quality from drifting under large
        // articulated pose changes. Skinned-mesh vertex movement is
        // typically larger frame-to-frame than ocean displacement, so the
        // periodic rebuild matters more here.
        uint32_t blasRefitCounter = 0;
        static constexpr uint32_t kBlasFullRebuildInterval = 64;
    };

    // Per-Mesh tet-skinned (PhysX soft body) deformed BLAS. Like SkinnedMesh,
    // each body has its own pose so it can't share a BLAS. The static tet
    // bindings (tetIndex/tetWeight/restInv*) + rest normals are uploaded once;
    // the per-frame collision-tet positions are re-uploaded from the soft body's
    // tet texture each frame, then tet_skinning.comp blends the full-res visual
    // into the BLAS vertex/normal buffers and the BLAS is refit in place.
    struct TetMeshState {
        std::unique_ptr<BlasRecord> blas;
        std::weak_ptr<BufferGeometry> liveCheck;
        // Static per-vertex inputs (uploaded once at ensureTetBlas).
        Buffer tetIndex   {};// vec4<float> — 4 tet-vertex indices
        Buffer tetWeight  {};// vec4<float> — barycentric weights
        Buffer baseNormal {};// vec3<float> — rest normals
        Buffer restInv0   {};// vec3<float> — baked Dr^-1 column 0
        Buffer restInv1   {};// vec3<float> — baked Dr^-1 column 1
        Buffer restInv2   {};// vec3<float> — baked Dr^-1 column 2
        // Per-frame collision-tet world positions (vec4/texel), re-uploaded each
        // frame from the soft body's tet texture image. Ring-buffered (one
        // buffer + descriptor set per slot): the host write in refreshTetBlas
        // happens before renderFrame's fence wait, so kFramesInFlight prior
        // frames may still be reading — a single buffer would get overwritten
        // mid-flight and consecutive displayed frames would skin with the same
        // physics state (duplicate-then-skip judder). refreshTetBlas advances
        // tetPosSlot once per frame; recordCommandBuffer dispatches with that
        // slot's descriptor set.
        static constexpr uint32_t kTetPosSlots = vulkan::TetSkinningPipeline::kPosSlots;
        static_assert(kTetPosSlots >= kFramesInFlight + 1,
                      "tetPos ring must cover all in-flight frames plus the one being recorded");
        std::array<Buffer, kTetPosSlots> tetPos {};
        uint32_t tetPosSlot = 0;// slot written this frame (advanced by refreshTetBlas)
        VkDeviceSize tetPosBytes = 0;
        // Zero-copy interop (enableSoftBodyInterop): when tetPosExt holds a
        // buffer, it replaces the tetPos ring as the shader's binding-6 source
        // (rewritten in every slot's set) and tetPosExternalCopy (a CUDA
        // device→device copy registered by the PhysX glue) replaces the CPU
        // upload in refreshTetBlas.
        vulkan::ExternalBuffer tetPosExt {};
        std::function<void()>  tetPosExternalCopy;
        uint32_t vertexCount    = 0;
        uint32_t primitiveCount = 0;
        bool     indexed        = false;
        std::array<VkDescriptorSet, kTetPosSlots> tetDescSet {};
        Buffer blasScratch {};
        VkDeviceSize blasScratchSize = 0;
        uint32_t blasRefitCounter = 0;
        static constexpr uint32_t kBlasFullRebuildInterval = 64;
    };

    // Per-Mesh morphed-geometry BLAS. Two meshes sharing the same
    // BufferGeometry can have different morphTargetInfluences, so each
    // morphed mesh gets its own BLAS (same principle as SkinnedMesh).
    // prevInfluences is the dirty-detection key (memcmp).
    struct MorphedMeshState {
        std::unique_ptr<BlasRecord> blas;
        std::vector<float> prevInfluences;
        std::vector<float> blendedPositions;
        std::vector<float> blendedNormals;
        std::weak_ptr<BufferGeometry> liveCheck;
    };

    // Per-DisplacedMesh: BLAS (rebuilt-in-place each frame, same scheme as
    // SkinnedMesh) plus up to three FFT cascades (Phillips/Dynamic/IFFT
    // per cascade). The water_displace.comp pass reads the spatial-domain
    // output images of all enabled cascades and writes positions+normals
    // into the BLAS vertex/normal buffers. Cascades cover disjoint k-bands
    // (band-passed via Phillips kMin/kMax) so the surface gains real
    // multi-scale wave detail without double-counting energy.
    struct DisplacedMeshState {
        std::unique_ptr<BlasRecord> blas;
        struct Cascade {
            std::unique_ptr<water::PhillipsSpectrum> phillips;
            std::unique_ptr<water::DynamicSpectrum>  dyn;
            std::unique_ptr<water::IFFT>             ifft;
            bool  phillipsRecorded = false;
            float tileSize = 0.f;            // 0 = cascade not in use
        };
        std::array<Cascade, 3> cascades;
        uint32_t cascadeMask = 0;            // bit i set = cascade i enabled
        // Wind + fetch actually baked into the cascades' h0 spectra. When
        // dm.params drifts from these, recordDisplacedDeform rewrites the
        // Phillips params and re-dispatches the (normally one-shot) h0
        // pass — the per-cascade noise images persist, so the wave field
        // MORPHS smoothly into the new sea state instead of jumping.
        float appliedWindSpeed = 0.f;
        float appliedWindTheta = 0.f;
        float appliedFetch     = 0.f;
        water::OceanImage scratchA;          // RG32F IFFT scratch — shared across cascades (sequential dispatch)
        VkDescriptorSet displaceDS = VK_NULL_HANDLE;
        uint32_t vertexCount = 0;
        // Grid topology + rest extents (rectangular grids supported; a
        // square is just gridDimX == gridDimZ). Validated at init:
        // gridDimX · gridDimZ == vertexCount. Rest positions are
        // reconstructed from these in water_displace.comp.
        uint32_t gridDimX   = 0;
        uint32_t gridDimZ   = 0;
        float    planeSizeX = 0.f;
        float    planeSizeZ = 0.f;
        // Per-cascade height readback for CPU-side wave sampling (boat
        // hydrodynamics, pitch/roll from multi-scale wave slope, etc.).
        // Host-mapped RG32F buffers of textureSize²·8 bytes, populated
        // after every IFFT pass via vkCmdCopyImageToBuffer — a RING of
        // kFramesInFlight per cascade, indexed [cascade][currentFrame]:
        // the frame recorded in slot s copies into slot s, and the mirror
        // (mirrorDisplacedHeightfields, run right after the frame-begin
        // vkWaitForFences(inFlight[currentFrame])) reads slot currentFrame
        // — the copy the slot's previous occupant recorded kFramesInFlight
        // frames ago, which that fence proves complete. ONE buffer shared
        // by the frames in flight was memcpy'd while the GPU could still be
        // copying into it, so sampleHeight() returned a field whose age
        // (and tearing) depended on how far the GPU had got: one frame old
        // when the host was slow, two or a torn mix of both flat out.
        // Allocated lazily on the first frame that records the copies
        // (sampleHeight()'s sticky opt-in), so scenes that never query
        // wave height pay no memory for it; null handles until then.
        Buffer    heightReadback[3][kFramesInFlight] = {};
        // Per-cascade readback texture dimension. Cascades can run at
        // different FFT resolutions — each cascade's readback buffer is
        // sized to its own dim²·8 bytes (RG32F).
        uint32_t  heightReadbackDim[3] = {0, 0, 0};
        // Per-slot: true once a command buffer that copies into slot s has
        // been recorded (a frame cb, or the synchronous first-build
        // one-shot). The copies are gated on the sampleHeight() opt-in, so
        // an unwritten slot holds uninitialized VMA memory — mirroring it
        // would hand garbage heights to buoyancy for a frame; the mirror
        // skips it and keeps the last good field instead.
        bool      heightReadbackWritten[kFramesInFlight] = {};
        std::weak_ptr<BufferGeometry> liveCheck;
        // Per-frame BLAS-refit state. The initial AS is built (buildBlasFor)
        // with ALLOW_UPDATE, so subsequent per-frame refreshes can use
        // MODE_UPDATE (refit) instead of full BUILD — typically 3-10× faster
        // for the same triangle count. AS quality degrades over many refits;
        // force a full rebuild every kBlasFullRebuildInterval frames.
        uint32_t  blasRefitCounter = 0;          // frames since last full BUILD
        static constexpr uint32_t kBlasFullRebuildInterval = 64;

        // Per-mesh foam-disturbance SSBO. Host-mapped, written from
        // dm.foamDisturbances each frame before the water_displace
        // dispatch. Sized to kMaxFoamDisturbances × 16B; extra
        // entries are dropped. .address == 0 until first allocation.
        Buffer    foamDisturbBuffer{};
        static constexpr uint32_t kMaxFoamDisturbances = 64;

        // Per-mesh wake-trail SSBO. Host-mapped, written from
        // dm.wake.trail each frame. Each sample is 32 bytes (matches
        // DisplacedMesh::WakeSample). The water_displace shader
        // iterates all valid samples and sums age-decayed Kelvin
        // V-wake contributions so the wake traces the boat's past
        // path rather than snapping to the current pose.
        Buffer    wakeTrailBuffer{};
        static constexpr uint32_t kMaxWakeSamples = 64;

        // World-space foam texture — 2D R32F covering the cascade-0
        // tile (foamTileSize × foamTileSize world m), REPEAT-sampled
        // in both compute and chit so wrapping at the tile boundary
        // aligns with the FFT pattern's periodicity. Replaces the old
        // per-vertex `foam` buffer on the BLAS: with foam pinned to
        // world coords, the ocean mesh can be re-tessellated freely
        // without dragging foam along with the vertex indices.
        //
        // No ping-pong needed because foam_world.comp only reads and
        // writes its own texel each invocation — no cross-texel
        // dependency, so `imageLoad` + `imageStore` on the same image
        // is race-free.
        water::OceanImage foamImage;
        uint32_t foamRes      = 0;          // texels per side
        float    foamTileSize = 0.f;        // world extent (m) covered
        VkDescriptorSet foamWorldDS = VK_NULL_HANDLE;
        // Wall-clock timestamp of the previous foam dispatch — drives the
        // frame-rate-independent decay push constant (−1 = first frame).
        double   foamPrevTimeSec = -1.0;
        // Foam dispatch-interval state (THREEPP_OCEAN_FOAM_INTERVAL=N runs
        // foam_world every Nth frame; decay is dt-aware so skipped frames
        // just widen the decay step). Disturbance stamps supplied on skipped
        // frames are carried here and uploaded with the next dispatch — the
        // shader combines stamps with max(), so a re-stamped persistent
        // source is idempotent. Layout mirrors DisplacedMesh::FoamDisturbance
        // (static_asserted at the upload site); capped at
        // kMaxFoamDisturbances, newest kept.
        struct FoamDisturbCarry {
            float worldX, worldZ, radius, intensity;
        };
        uint32_t foamTick = 0;
        std::vector<FoamDisturbCarry> foamDisturbCarry;
    };

    // ── GrassMesh (GPU wind-deformed foliage) ────────────────────────
    // Same deform-on-GPU + BLAS-refit-in-place pattern as DisplacedMesh,
    // but FFT-free: a wind compute shader bends each blade vertex from an
    // immutable rest pose. The whole grass field is ONE mesh → one BLAS →
    // one TLAS instance, so animating it is a cheap per-frame BLAS/TLAS
    // refit instead of the O(all-instances) TLAS rebuild an animated
    // InstancedMesh would force.
    struct GrassMeshState {
        std::unique_ptr<BlasRecord> blas;
        Buffer   restPos{};   // immutable rest xyz (device addr, host-written once)
        Buffer   heightFrac{};// per-vertex height fraction (device addr)
        uint32_t vertexCount = 0;
        std::weak_ptr<BufferGeometry> liveCheck;
        uint32_t blasRefitCounter = 0;
        static constexpr uint32_t kBlasFullRebuildInterval = 64;
    };

    // Per-instance descriptor tables the closest-hit shader indexes by
    // gl_InstanceCustomIndexEXT. Layout matches the matching shader
    // structs in closest_hit.rchit.
    struct GeometryDesc {
        VkDeviceAddress vertexAddress;// positions, used for per-pixel tangent derivation
        VkDeviceAddress normalAddress;
        VkDeviceAddress indexAddress;
        VkDeviceAddress uvAddress;// 0 == no UV attribute
        VkDeviceAddress foamAddress;// 0 == no foam attribute (per-vertex float, written by water_displace.comp)
        // Previous-frame deformed vertex positions. For SkinnedMesh (or any
        // mesh that re-deforms per frame): a separate buffer holding the
        // previous frame's vertex data. The deferred shade's ray-query hit
        // handling interpolates these to give a per-vertex "previous world
        // position" so reprojection tracks the same surface point across
        // deformation. For static / rigid-only meshes: set equal to
        // vertexAddress, in which case the hit handling reads the same
        // data twice (no harm; equals current pos).
        VkDeviceAddress prevVertexAddress;
        // Per-vertex RGB color buffer (BlasRecord::color). 0 when the mesh's
        // material has vertexColors off or the geometry has no "color" — the
        // chit then skips the vertex-color multiply. Set per-instance below.
        VkDeviceAddress colorAddress;
        uint32_t indexed;
        // Bit 0: sticky "recently moved" flag, stamped per frame by the
        // loop in VulkanCoreFrame.cpp (reflection/GI history gates read it).
        // Bits 1..3: BlasRecord::packedMask << 1 — attribute buffers hold
        // packed data (bit1 nrm oct-snorm16x2 / bit2 uv unorm16x2 / bit3
        // col unorm8x4). Shaders mask accordingly; a bare `!= 0` test on
        // this word is WRONG for the moved gate.
        uint32_t flags;
    };

    struct LodJob {
        const BufferGeometry* geom = nullptr;
        unsigned int geomVersion = 0;
        std::vector<float> positions;// tightly packed xyz
        // Indexed geometry: the source index array; normals/uvs stay
        // empty (the weld step is skipped, topology already exists).
        // NON-indexed soup (FBX-style loaders never call setIndex):
        // empty — the worker first welds identical-attribute duplicates
        // into canonical indices (buildCanonicalIndices) so the
        // simplifier has shared edges to collapse. normals/uvs feed that
        // weld's equality test so genuine seams/hard edges stay split;
        // they carry copies only when the stream exists with matching
        // counts.
        std::vector<uint32_t> indices;
        std::vector<float> normals;// tightly packed xyz — simplifier attribute
                                   // metric (all paths) + soup weld equality
        std::vector<float> uvs;    // soup only; tightly packed xy
        // Normal-attribute weight for the simplifier, scaled by the
        // enqueueing entry's material glossiness: matte foliage barely
        // charges normal deviation (retiring far detail is what the LOD
        // is for), glossy paint charges it fully (shading is the mm-scale
        // normal field). See lodNormalWeightFor.
        float normalWeight = 0.5f;
    };

    struct LodResult {
        const BufferGeometry* geom = nullptr;
        unsigned int geomVersion = 0;
        std::vector<geometrylod::Level> levels;// empty ⇒ degenerate/failed
    };

    // One level's deferred BLAS build, produced by buildLodLevelFor and
    // consumed by flushLodLevelBuilds — resources (AS handle, storage,
    // index buffer, scratch) already exist; only the build execution is
    // deferred so a whole frame's worth records into one command buffer.
    struct LodPendingBuild {
        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        VkDeviceAddress vertexAddress = 0;
        VkDeviceAddress indexAddress = 0;
        uint32_t maxVertex = 0;
        uint32_t primitiveCount = 0;
        bool packedIdx = false;// index buffer is uint16 (base record's bit 3)
        Buffer scratch{};// per-build scratch: concurrent builds in one cmdbuf must not alias
    };

    // Per-entry fingerprint used to detect scene changes between frames.
    // We capture the surface state ray tracing actually consumes:
    //   - mesh / geometry / material identity (covers add/remove/swap)
    //   - effective worldMatrix 16 floats (covers transform animation
    //     AND per-instance setMatrixAt; the matrix already incorporates
    //     mesh->matrixWorld * instanceMatrix[i] for InstancedMesh)
    //   - PBR scalars (covers material slider tweaks)
    // Instance count change shows up as currFp.size() mismatch, forcing
    // structural rebuild — correct since each instance is its own TLAS slot.
    struct MeshFingerprint {
        const void* mesh;
        const void* geom;
        const void* mat;
        // Texture fingerprint slots hold the shared_ptr (not just the raw
        // pointer) so the comparison can't be fooled by an allocator
        // recycling a freed Texture's address for a brand-new Texture: a
        // held shared_ptr keeps the previous Texture alive long enough that
        // no overlap can happen, and operator!= still compares get().
        std::shared_ptr<Texture> albedoTex;            // covers map swap on the same material
        std::shared_ptr<Texture> roughnessTex;         // covers roughnessMap swap
        std::shared_ptr<Texture> metalnessTex;         // covers metalnessMap swap
        std::shared_ptr<Texture> normalTex;            // covers normalMap swap
        std::shared_ptr<Texture> transmissionTex;      // covers transmissionMap swap
        std::shared_ptr<Texture> clearcoatTex;         // covers clearcoatMap swap
        std::shared_ptr<Texture> clearcoatRoughnessTex;// covers clearcoatRoughnessMap swap
        std::shared_ptr<Texture> emissiveTex;          // covers emissiveMap swap
        std::shared_ptr<Texture> occlusionTex;         // covers aoMap swap — was MISSING: an
                                                       // occlusion-map swap neither triggered a
                                                       // structural rebuild nor got the held-
                                                       // shared_ptr recycle protection, so its
                                                       // dead predecessor's Texture* could be
                                                       // reused and hit a stale textureCache slot
        uint32_t instanceIndex;// 0 for non-instanced; distinguishes sub-instances
        // TLAS membership. An overlay flip (e.g. material.wireframe toggled)
        // keeps every pointer above identical but adds/removes this entry's
        // TLAS instance, so it must classify as STRUCTURAL: a MODE_UPDATE
        // refit with a different instance count than the TLAS's last build
        // is invalid (VUID-vkCmdBuildAccelerationStructuresKHR) and corrupts
        // traversal — observed as a ray-query hang → 2 s TDR → device lost.
        bool overlay = false;
        unsigned int matVersion = 0;// Material::version() — bumped by needsUpdate(),
                                    // KHR_animation_pointer, etc. Lets us skip the
                                    // 8 texture-of dynamic_casts + materialFromMesh
                                    // (~21 dynamic_casts/mesh) when nothing on the
                                    // material has changed since last frame.
        unsigned int geomVersion = 0;// composite BufferAttribute version (pos+norm+idx+uv+color)
        std::array<float, 16> matrix{};
        std::array<float, 15> pbr{};// + normalScale.xy + transmission/ior + clearcoat/roughness
        // TYPED views + attribute-pointer cache for the snapshot lean diff
        // (the void* fields above are compare-only). attrVersion gates the
        // cached attribute pointers: while BufferGeometry::
        // attributesVersion() is unchanged, the attribute map still holds
        // the same objects, so the raw pointers cannot dangle and the
        // composite geomVersion is 4 direct version reads instead of 4
        // string-keyed map lookups + dynamic_casts per entry per frame.
        const Material* matTyped = nullptr;
        BufferGeometry* geomTyped = nullptr;
        unsigned int attrVersion = ~0u;
        const BufferAttribute* posAttr  = nullptr;
        const BufferAttribute* normAttr = nullptr;
        const BufferAttribute* uvAttr   = nullptr;
        const BufferAttribute* idxAttr  = nullptr;
        const BufferAttribute* colAttr  = nullptr;// was MISSING: a color-only
                                                  // edit sailed through the lean
                                                  // fast path unseen even though
                                                  // geomVersionOf() counts it
    };

    // Per-frame refresh op for a plain (non-skinned/non-displaced/non-morphed)
    // dynamic geometry whose BufferAttribute versions just bumped. Batched
    // through refreshGeomBlasBatch so N soft bodies cost 2 GPU submits
    // total, not 2N.
    struct GeomRefreshOp {
        const BufferGeometry* geom;
        BlasRecord*           rec;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_GEOMETRY_STATE_HPP
