#include "VulkanCoreImpl.hpp"

#include "VulkanCpuPhaseProf.hpp"

#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kInstFlag* bit layout

// debug_resolve.comp's embedded SPIR-V array (kDebugResolveCompSpv) is only
// referenced by createDebugResolvePipeline below — moved out of the header
// with the rest of this file's methods.
#include "threepp/renderers/vulkan/shaders/debug_resolve.comp.spv.h"

namespace threepp {

    bool VulkanRenderer::Impl::ensureDrawInfoCapacity(uint32_t frame, VkDeviceSize neededBytes) {
        if (neededBytes <= view().drawInfoBufferCapacity[frame]) return false;
        const VkDeviceSize newCap = std::max<VkDeviceSize>(
                neededBytes, view().drawInfoBufferCapacity[frame] * 2u);
        destroyBuffer(ctx->allocator(), view().drawInfoBuffers[frame]);
        view().drawInfoBuffers[frame] = createBuffer(
                ctx->allocator(), ctx->device(),
                newCap,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
        view().drawInfoBufferCapacity[frame] = newCap;
        return true;
    }

    bool VulkanRenderer::Impl::ensureIndirectCmdCapacity(uint32_t frame, VkDeviceSize neededBytes) {
        if (neededBytes <= view().indirectCmdBufferCapacity[frame]) return false;
        const VkDeviceSize newCap = std::max<VkDeviceSize>(
                neededBytes, view().indirectCmdBufferCapacity[frame] * 2u);
        destroyBuffer(ctx->allocator(), view().indirectCmdBuffers[frame]);
        view().indirectCmdBuffers[frame] = createBuffer(
                ctx->allocator(), ctx->device(),
                newCap,
                // STORAGE: the occlusion-cull filter compute reads these
                // CPU-built records as an SSBO (occl_cull.comp binding 0).
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
        view().indirectCmdBufferCapacity[frame] = newCap;
        return true;
    }

    // Build the per-frame DrawInfo + indirect-command buffers for the
    // hybrid raster G-buffer pass. Called from renderFrame right after
    // cullEntriesAgainstFrustum (which sets MeshEntry::inFrustum) so we
    // can skip culled draws here. The actual GPU dispatch happens in
    // recordRasterGbufPass via 1-4 vkCmdDrawIndirect calls partitioned
    // by cull mode (+ a trailing blend-decal bucket).
    //
    // Draw partitioning: walk entries once, sort each visible draw
    // into one of four buckets (Side::Front/Back/Double, then blend
    // decals). Buckets are concatenated [Front | Back | Double | Decal]
    // in the device buffers, and each VkDrawIndirectCommand's firstInstance carries the
    // global DrawInfo index — surfaced to the VS as gl_InstanceIndex
    // so the shader fetches `draws[gl_InstanceIndex]`. This trick
    // sidesteps the gl_DrawIDARB-resets-per-call issue without
    // needing dynamic-offset descriptors or per-call push constants.
    void VulkanRenderer::Impl::buildIndirectDrawData(uint32_t frame) {
        // SPLIT, not nested: function-scope until now, so its number already
        // included the ~15 MB of memcpy at the bottom of this function and the
        // build could not be told from the copy. stop()ped where the entry walk
        // ends; the concatenate-and-copy region below carries
        // frame.F_uploadDrawInfo, as a sibling. Every early return between here
        // and the stop() still accounts (the destructor is what fires).
        THREEPP_CPUPROF_NAMED(profD, "frame.D_buildIndirect");
        for (auto& g : indirectGroups_) { g.offset = 0; g.count = 0; }
        indirectTotalDraws_ = 0;
        particleDrawSlots_.clear();

        if (lastVisibleEntries_.empty()) return;

        // Four buckets: [BACK_cull, FRONT_cull, NONE_cull, decal] order.
        // Reused member scratch (see the declarations in VulkanCoreImpl.hpp):
        // cleared, not reallocated, so capacity survives across frames.
        auto& draws = indirectDrawScratch_;
        auto& cmds  = indirectCmdScratch_;
        for (auto& v : draws) v.clear();
        for (auto& v : cmds) v.clear();
        // Occlusion-cull metadata rides the same bucketing so its final
        // concatenated order matches the command records 1:1.
        // PRIMARY ONLY, like the two-phase record itself (recordGbufferStage
        // gates the passes, but the hazard is here): occl_'s per-FIF
        // descriptor set and meta buffer are single shared instances, and a
        // secondary's build runs AFTER the primary's dispatches were recorded
        // but BEFORE the GPU executes them — prepareFrame would re-point the
        // set at the secondary's cmd/camera buffers and flushMeta would
        // replace the meta (with a different draw count), so the primary's
        // filter/cull-test would execute against the secondary's camera and
        // list. That is both "moving the second camera changes the main
        // view" and, when the counts mismatch, garbage compacted into
        // indirect draws — a GPU hang → watchdog → device lost.
        const bool wantOcclMeta = occlusionCullingEnabled_ && occl_ &&
                                  occlHiz_ && occlHiz_->valid() && !scissorTest &&
                                  !view().secondary;

        // Skip signature: the DrawInfo/cmd/occl-meta contents are a pure
        // function of (scene draw inputs, this view's cull bits, the occl
        // configuration). When this FIF slot's device buffers were last
        // filled from the exact same inputs, restore the CPU-side outputs
        // and return — a fully static frame reuses ~20 MB of records
        // verbatim instead of rebuilding and re-copying them.
        const std::array<uint64_t, 2> buildSig = {
                (uint64_t(drawInputsVersion_) << 32) | uint64_t(view().cullVersion),
                (wantOcclMeta ? (1ull << 63) : 0ull) ^
                        (wantOcclMeta ? reinterpret_cast<uint64_t>(occlHiz_->view()) : 0ull)};
        // A scene with a ParticleField never takes the skip path. The DrawInfo
        // contents would restore correctly, but particleDrawSlots_ (which is
        // per-view and holds the push-constant index the particle pipeline
        // needs) would not — it is Impl state, not view state, so a secondary
        // view's build would leave the primary reading the wrong index. Fields
        // only appear in scenes that are moving anyway, where the signature
        // never matches; this is here so it cannot be wrong in the case it does.
        if (particleFields_.empty() && view().indirectBuiltSig[frame] == buildSig) {
            indirectGroups_     = view().cachedIndirectGroups;
            indirectTotalDraws_ = view().cachedIndirectTotal;
            if (!view().secondary) occlActiveThisFrame_ = view().cachedOcclActive;
            if (view().cachedOcclActive && indirectTotalDraws_ > 0u) {
                // Descriptor upkeep only — handles and capacity are unchanged
                // by definition of the signature match.
                vulkan::OcclusionCull::FrameInputs oin{};
                oin.srcCmds    = view().indirectCmdBuffers[frame].handle;
                oin.rasterCam  = view().rasterCameraUbos[frame].handle;
                oin.hizView    = occlHiz_->view();
                oin.hizSampler = occlHiz_->sampler();
                occl_->prepareFrame(frame, indirectTotalDraws_, occlBitDomain_, oin);
            }
            return;
        }

        auto& occlMeta = indirectOcclScratch_;
        for (auto& v : occlMeta) v.clear();
        auto bucketOf = [](VkCullModeFlags cm) -> int {
            if (cm == VK_CULL_MODE_BACK_BIT)  return 0;
            if (cm == VK_CULL_MODE_FRONT_BIT) return 1;
            return 2;
        };

        // Where each visible ParticleField's DrawInfo landed, as (bucket,
        // position-in-bucket) — resolvable to a global index only after the
        // bucket offsets below are known.
        struct PfSlot { const ParticleField* field; int bucket; uint32_t pos; VkCullModeFlags cull; };
        std::vector<PfSlot> pfSlots;

        uint32_t globalIdx = 0;
        // Last-entry memos. An InstancedMesh expands to one entry per instance,
        // all sharing a single en.mesh, so these collapse a whole instanced run
        // to one lookup each; on a scene of distinct meshes they simply always
        // miss and cost one pointer compare. Keying on en.mesh is exact: the
        // BLAS resolve dispatches on en.mesh's type and looks up by en.mesh (or
        // its geometry), and the two id helpers key purely on en.mesh->id. The
        // caches they read are not mutated inside this loop. Deliberately LOCAL
        // — a BlasRecord* cached on MeshEntry would dangle after a BLAS evict,
        // since entries persist across frames via the snapshot fast path.
        // The id pair is filled LAZILY, below the skip-continues: stableIdForObject
        // MUTATES (try_emplace hands out the next auto id), so hoisting it above
        // the BLAS check would assign ids to entries that never previously got
        // one and shift every later object's stable id.
        const Object3D*   memoMesh     = nullptr;
        const BlasRecord* memoRec      = nullptr;
        bool              memoIdsValid = false;
        uint16_t          memoClassId  = 0;
        uint16_t          memoStableId = 0;
        // Material-derived per-mesh memos (one virtual material() call +
        // flag/offset derivation per RUN of instances, not per instance).
        std::shared_ptr<Material> memoMatSp;
        uint32_t          memoMatFlags = 0u;// double-sided / tex-anim bits
        bool              memoVtxColor = false;
        float             memoPolyOff  = 0.f;
        bool              memoLodStatic = false;// record has no LOD chain
        LodGeomSel        memoLodSel0{};
        // Object-space bounds for the occl-meta Arvo transform.
        bool              memoBoundsValid = false;
        float             memoCenter[3]{}, memoHalf[3]{};
        // BufferGeometry::drawRange, raw (clamped against the selected LOD's
        // element total per entry below). Changes are folded into the skip
        // signature by the drawRange compare in ensureSceneBuilt's enqueue
        // loop, which bumps drawInputsVersion_ — new DrawInfo inputs always
        // must (see the EntrySpan work).
        uint32_t          memoDrawStart = 0u;
        uint32_t          memoDrawCount = 0xFFFFFFFFu;
        for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
            const auto& en = lastVisibleEntries_[i];
            if (en.isOverlay)  continue;
            // Sensor-only surfaces rasterize into the secondary views that ASK
            // (setViewSensorSurfaces — a depth sensor's G-buffer) and never
            // into the primary, whose picture of the same scan is the splat
            // rasterizer's. Per view, because an RGB camera preview and an
            // editor viewport pane are secondary views too and must not show
            // the bake shell. Scene opt-in off ⇒ no view draws it.
            if (en.sensorOnly && !(sensorOnlySurfaces_ && view().secondary && view().sensorSurfaces)) continue;
            if (!viewCulled(i)) continue;
            if (en.mesh != memoMesh) {
                memoMesh     = en.mesh;
                memoRec      = resolveBlasForEntry(en);
                memoIdsValid = false;
                // One virtual material() call per run (it is an out-of-line
                // override returning shared_ptr by value, so the compiler
                // cannot CSE repeated calls). Held by shared_ptr so the raw
                // pointer cannot dangle on a temporary.
                memoMatSp = en.mesh->material();
                const Material* sm = memoMatSp.get();
                memoMatFlags = 0u;
                // DOUBLE_SIDED: gbuffer.frag flips N toward the viewer, so on
                // cutout foliage the jittered coverage flips a pixel's normal
                // SIGN frame to frame. The GI temporal reproject + SVGF normal
                // edge-stop must treat ±N as the SAME surface there (flag
                // consumed in deferred_shade.comp / deferred_gi_filter.comp) or
                // the GI history cold-starts every frame — measured as 8× the
                // frame-to-frame flicker on a procedural tree canopy.
                if (sm && sm->side == Side::Double) memoMatFlags |= kInstFlagDoubleSided;
                // TEXTURE-ANIMATED (scrolling UVs, video/live textures): the
                // pattern moves with NO geometric motion vectors, so the TAA
                // resolve must hold a short history (α floor) instead of
                // smearing it. TAA-only — the GI history accumulates
                // DEMODULATED irradiance, which a texture animation doesn't
                // change, so bit 6 deliberately does NOT shorten the GI cap.
                if (sm && sm->textureAnimatedHint) memoMatFlags |= kInstFlagTexAnim;
                memoVtxColor = sm && sm->vertexColors;
                // polygonOffset → per-mesh clip-z depth bias (decals). Reverse-Z:
                // a +clip-z bias pushes the surface toward NEAR so it renders on
                // top of coplanar geometry (no z-fight). threepp/GL uses NEGATIVE
                // polygonOffsetUnits for "on top", so flip the sign; units==0
                // (bool only) → a small default that clears z-fight without
                // floating. (The slope-scaled `factor` term isn't applied — flat
                // decals don't need it; bias is uniform in NDC.)
                memoPolyOff = 0.f;
                if (sm && sm->polygonOffset) {
                    const float uf = sm->polygonOffsetUnits + sm->polygonOffsetFactor;
                    memoPolyOff = (uf != 0.f) ? (-uf * 1.0e-6f) : 4.0e-6f;
                }
                memoLodStatic = memoRec && memoRec->lodLevels.empty();
                if (memoRec) memoLodSel0 = selectLodGeom(*memoRec, 0);
                memoBoundsValid = false;
                memoDrawStart = 0u;
                memoDrawCount = 0xFFFFFFFFu;
                if (auto geom = en.mesh->geometry()) {
                    if (!geom->boundingBox) geom->computeBoundingBox();
                    if (geom->boundingBox) {
                        const Vector3 c = geom->boundingBox->getCenter();
                        const Vector3 h = geom->boundingBox->getSize() * 0.5f;
                        memoCenter[0] = c.x; memoCenter[1] = c.y; memoCenter[2] = c.z;
                        memoHalf[0]   = h.x; memoHalf[1]   = h.y; memoHalf[2]   = h.z;
                        memoBoundsValid = true;
                    }
                    const auto& dr = geom->drawRange;
                    memoDrawStart = dr.start > 0 ? static_cast<uint32_t>(dr.start) : 0u;
                    memoDrawCount = dr.count > 0 ? static_cast<uint32_t>(dr.count) : 0u;
                }
            }
            const BlasRecord* rec = memoRec;
            if (!rec || rec->vertex.handle == VK_NULL_HANDLE) continue;

            // Auto-LOD: en.lodLevel==0 for every non-eligible entry (the
            // selection pass in ensureSceneBuilt only sets it >0 on plain
            // static geometry), so this passthrough is a no-op everywhere
            // else. Must resolve identically to the TLAS instance fill in
            // ensureSceneBuilt — both read en.lodLevel verbatim. Indexed-ness
            // is PER SELECTION, not per record: a level of a non-indexed soup
            // record is an indexed draw (welded canonical indices) against
            // the same soup vertex buffer. Chain-less records (the common
            // case) resolve once per run via the memo.
            // ParticleField: LOD level 0, always. en.lodLevel was selected
            // against the field's PLACEHOLDER geometry, and rec here is the
            // MeshRepr proxy — two different geometries, so the level index
            // does not carry across. prepareParticleFields resolves the proxy's
            // vertex count the same way, and the two must agree.
            const auto lodSel = (memoLodStatic || en.isParticleField)
                                        ? memoLodSel0
                                        : selectLodGeom(*rec, en.lodLevel);
            const bool indexed = lodSel.indexed;
            uint32_t vcount = indexed ? lodSel.indexCount : rec->vertexCount;
            uint32_t vfirst = 0u;
            // BufferGeometry::drawRange, base level only — a simplified LOD
            // level's index set has no correspondence with the range's units.
            // Same clamp shape as the overlay path and GLRenderer: elements are
            // indices when indexed, vertices otherwise.
            if (en.lodLevel == 0 || memoLodStatic) {
                vfirst = std::min(memoDrawStart, vcount);
                vcount = std::min(vcount - vfirst, memoDrawCount);
            }
            if (vcount == 0u) continue;

            const VkCullModeFlags wantCull =
                    (i < lastVisibleCullMode_.size())
                            ? lastVisibleCullMode_[i]
                            : VK_CULL_MODE_BACK_BIT;
            // Blend decals (alphaCutoff == -2 sentinel from materialFromMesh)
            // go to bucket [3]: drawn last, with the albedo-blend pipeline.
            // The deferred shade reads the blended albedo attachment.
            const bool isDecal = (i < matDescsCached_.size()) &&
                                 (matDescsCached_[i].alphaCutoff == -2.0f);
            const int b = isDecal ? 3 : bucketOf(wantCull);

            DrawInfoGpu di{};
            std::memcpy(di.model, en.worldMatrix.data(), 64);
            di.posAddr     = rec->vertex.address;
            di.nrmAddr     = rec->normal.address;
            di.uvAddr      = (rec->uv.handle != VK_NULL_HANDLE) ? rec->uv.address : 0ull;
            // Warp-enabled DisplacedMesh: vertices reflow every frame, so
            // prevVertex can't track a stable world point — point the
            // motion-vector source at the current buffer so the ocean
            // reprojects as world-static (see prevVertexAddress in the
            // GeometryDesc build for the full rationale). An interop record
            // whose producer declared unstable vertex correspondence
            // (interopWorldStatic — marching-cubes soups) is the same case.
            bool warpReproject = rec->interopWorldStatic;
            if (en.isDisplaced) {
                warpReproject = static_cast<DisplacedMesh*>(en.mesh)->warp.halfRange > 0.0f;
            }
            di.prevPosAddr = (rec->prevVertex.handle != VK_NULL_HANDLE && !warpReproject)
                                     ? rec->prevVertex.address
                                     : rec->vertex.address;
            di.indexAddr   = indexed ? lodSel.indexAddress : 0ull;
            // Per-vertex color: only when the material opts in (vertexColors)
            // and the geometry uploaded a color buffer — matches the
            // GeometryDesc::colorAddress gate. gbuffer.frag multiplies albedo
            // by the interpolated value.
            di.colorAddr = (rec->color.handle != VK_NULL_HANDLE && memoVtxColor)
                                   ? rec->color.address
                                   : 0ull;
            di.instanceCustomIndex = static_cast<uint32_t>(i);
            // Per-instance flag word — canonical bit layout in
            // vulkan_shared.h (kInstFlag*); shaders read it back from the
            // gbuffer IDs .z via the instance_flags.glsl accessors. The
            // material-derived bits come from the per-run memo above.
            uint32_t flags = memoMatFlags;
            if (en.isDisplaced) flags |= kInstFlagWater;
            if (en.isSkinned)   flags |= kInstFlagSkinned;
            // PERSISTENT DEFORMER: a PhysX soft body deforms EVERY frame, so
            // the GI temporal cap in deferred_shade.comp must not chase its
            // oscillating per-pixel motion magnitude (visible pumping on the
            // wobble). The shader pins a constant history cap on this bit;
            // the TAA resolve floors its blend α (shading changes per frame).
            if (en.isTet) flags |= kInstFlagDeformer;
            // MOVED-STICKY mirrored into the flag word (same countdown the
            // GeometryDesc bit-0 stamp reads; that stamp runs earlier in this
            // same frame, so the two views agree). The reproject guards ask
            // "was the surface at the PREV texel moving" — answering that via
            // geoms[prevIds.x - 1] breaks the moment the entry list renumbers
            // (tile streaming removes a parent mid-list every few frames), so
            // the prev texel carries its own moved bit instead, identity-
            // stable like the .y stable id it is compared alongside.
            if (i < meshMovedSticky_.size() && meshMovedSticky_[i] > 0u)
                flags |= kInstFlagMoving;
            // Semantic CLASS id (0..255) packed into bits 8..15 — inert to
            // every flag bit-test (they mask the low byte) and carried
            // through the MSAA resolve. Read back via outIds.z >> 8.
            if (!memoIdsValid) {
                memoClassId  = classIdForObject(*en.mesh);
                memoStableId = stableIdForObject(*en.mesh);
                memoIdsValid = true;
            }
            flags |= (static_cast<uint32_t>(memoClassId) & 0xFFu) << 8;
            di.flags   = flags;
            di.indexed = indexed ? 1u : 0u;
            // STABLE per-object instance id -> outIds.y (survives visible-set
            // reordering, unlike instanceCustomIndex/.x).
            di.stableId    = memoStableId;
            di.packedAttrs = rec->packedMask;
            di.polygonOffset = memoPolyOff;// per-mesh, derived once in the memo
            draws[b].push_back(di);

            VkDrawIndirectCommand cmd{};
            // A ParticleField's DrawInfo record is real — the particle pipeline
            // reads it through a push-constant index, and it is where the proxy
            // geometry's addresses, the field's world matrix, its material's
            // instanceCustomIndex and its flags live. Its BUCKET draw is not:
            // the field is drawn by recordParticleFieldDraws from its own
            // indirect buffer, whose instanceCount the device writes. A zero
            // vertexCount is what makes the bucket's vkCmdDrawIndirect step
            // over the record for free, WITHOUT perturbing the one-instance
            // contract every other record in this array carries.
            cmd.vertexCount   = en.isParticleField ? 0u : vcount;
            cmd.instanceCount = 1u;
            // drawRange start rides through firstVertex: the pull-fetch VS
            // consumes gl_VertexIndex, which vkCmdDrawIndirect offsets by this
            // — indexing further into the index buffer (indexed) or the vertex
            // buffer (soup), exactly GL's drawRange semantics.
            cmd.firstVertex   = vfirst;
            cmd.firstInstance = 0u;// patched below to the final-position index
            cmds[b].push_back(cmd);
            if (en.isParticleField) {
                pfSlots.push_back({static_cast<const ParticleField*>(en.mesh), b,
                                   static_cast<uint32_t>(draws[b].size()) - 1u, wantCull});
            }

            if (wantOcclMeta) {
                // Same rules as cullEntriesAgainstFrustum: deformers'
                // cached local AABB is stale per frame → always draw;
                // missing bounds → always draw. Grass is the exception —
                // its swayed extent has a tight provable bound (rest AABB
                // dilated by windStrength; grass_wind.comp keeps
                // bend ≤ 0.85·windStrength), so it does NOT get the always-
                // draw bit and can be occlusion-culled behind terrain like a
                // static mesh, using that same dilated box.
                vulkan::OcclusionCull::CullMeta cm{};
                // Per-INSTANCE visibility bit (per-mesh base + instanceIndex)
                // — NOT di.stableId, which is per-object and made phase 1
                // all-or-nothing for InstancedMesh (see occlCullBitFor).
                cm.cullBit = occlCullBitFor(en);
                // ParticleField: always draw. The occlusion path copies draw
                // records with instanceCount zeroed and never compacts them
                // ("records never move"), and a field's CPU-side world AABB is
                // not merely stale but unknowable — the positions are on the
                // device. Same escape the deformers take.
                // isVertexInterop for the ParticleField reason, one step further:
                // a foreign CUDA producer owns the positions, so the memoized
                // object-space box below describes the host array, which under
                // interop is not a conservative bound on anything. Always draw.
                bool always = en.isSkinned || en.isDisplaced ||
                              en.isMorphed || en.isTet || en.isParticleField ||
                              en.isVertexInterop;
                if (en.isGrass) {
                    Box3 worldAabb;
                    if (!grassSwayWorldAabb(en, worldAabb)) {
                        always = true;
                        cm.flags = 1u;
                    } else {
                        cm.aabbMin[0] = worldAabb.min().x;
                        cm.aabbMin[1] = worldAabb.min().y;
                        cm.aabbMin[2] = worldAabb.min().z;
                        cm.aabbMax[0] = worldAabb.max().x;
                        cm.aabbMax[1] = worldAabb.max().y;
                        cm.aabbMax[2] = worldAabb.max().z;
                    }
                } else if (!always && memoBoundsValid) {
                    // Arvo transform of the memoized object-space box: exactly
                    // the AABB of the 8 transformed corners (world matrices
                    // are affine), without the per-corner perspective divides
                    // Box3::applyMatrix4 pays.
                    const float* M = en.worldMatrix.data();
                    const float cx = M[0]*memoCenter[0] + M[4]*memoCenter[1] + M[8]*memoCenter[2] + M[12];
                    const float cy = M[1]*memoCenter[0] + M[5]*memoCenter[1] + M[9]*memoCenter[2] + M[13];
                    const float cz = M[2]*memoCenter[0] + M[6]*memoCenter[1] + M[10]*memoCenter[2] + M[14];
                    const float hx = std::abs(M[0])*memoHalf[0] + std::abs(M[4])*memoHalf[1] + std::abs(M[8])*memoHalf[2];
                    const float hy = std::abs(M[1])*memoHalf[0] + std::abs(M[5])*memoHalf[1] + std::abs(M[9])*memoHalf[2];
                    const float hz = std::abs(M[2])*memoHalf[0] + std::abs(M[6])*memoHalf[1] + std::abs(M[10])*memoHalf[2];
                    cm.aabbMin[0] = cx - hx; cm.aabbMin[1] = cy - hy; cm.aabbMin[2] = cz - hz;
                    cm.aabbMax[0] = cx + hx; cm.aabbMax[1] = cy + hy; cm.aabbMax[2] = cz + hz;
                } else {
                    always = true;
                }
                cm.flags = always ? 1u : 0u;
                occlMeta[b].push_back(cm);
            }
            ++globalIdx;
        }

        indirectTotalDraws_ = globalIdx;
        // Publish only from the primary's build: a secondary reruns this
        // function later in the same frame (wantOcclMeta forced false above)
        // and must not stomp the flag the primary's frame body branched on.
        const bool occlThisBuild = wantOcclMeta && globalIdx > 0u;
        if (!view().secondary) occlActiveThisFrame_ = occlThisBuild;
        if (globalIdx == 0u) {
            // Nothing visible: cache the (all-zero) outputs so identical
            // frames skip even the entry walk.
            view().cachedIndirectGroups = indirectGroups_;
            view().cachedIndirectTotal  = 0u;
            view().cachedOcclActive     = false;
            view().indirectBuiltSig[frame] = buildSig;
            return;
        }
        // Build done; everything below is bytes-to-the-device (plus the
        // firstInstance patch loop, which is O(draws) CPU work living inside what
        // reads as a pure memcpy region).
        profD.stop();
        THREEPP_CPUPROF("frame.F_uploadDrawInfo");

        // Concatenate buckets into the per-frame device buffers.
        const VkDeviceSize drawBytes = sizeof(DrawInfoGpu) * globalIdx;
        const VkDeviceSize cmdBytes  = sizeof(VkDrawIndirectCommand) * globalIdx;
        const bool drawGrown = ensureDrawInfoCapacity(frame, drawBytes);
        ensureIndirectCmdCapacity(frame, cmdBytes);

        // Occlusion culling: size the phase buffers + rewrite this fif's
        // sets if any input changed (AFTER the capacity calls above so
        // the src handle is final), then get the mapped meta destination.
        vulkan::OcclusionCull::CullMeta* occlMetaDst = nullptr;
        if (occlThisBuild) {
            vulkan::OcclusionCull::FrameInputs oin{};
            oin.srcCmds    = view().indirectCmdBuffers[frame].handle;
            oin.rasterCam  = view().rasterCameraUbos[frame].handle;
            oin.hizView    = occlHiz_->view();
            oin.hizSampler = occlHiz_->sampler();
            occl_->prepareFrame(frame, globalIdx, occlBitDomain_, oin);
            occlMetaDst = occl_->metaPtr(frame);
        }

        void* mappedDraws = nullptr;
        vmaMapMemory(ctx->allocator(), view().drawInfoBuffers[frame].alloc, &mappedDraws);
        void* mappedCmds = nullptr;
        vmaMapMemory(ctx->allocator(), view().indirectCmdBuffers[frame].alloc, &mappedCmds);

        uint8_t* dDst = static_cast<uint8_t*>(mappedDraws);
        uint8_t* cDst = static_cast<uint8_t*>(mappedCmds);
        uint32_t offset = 0;
        const VkCullModeFlags cullForBucket[4] = {
                VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT, VK_CULL_MODE_NONE,
                VK_CULL_MODE_NONE// decals: DecalGeometry wraps edges, don't cull
        };
        for (int b = 0; b < 4; ++b) {
            const uint32_t n = static_cast<uint32_t>(draws[b].size());
            indirectGroups_[b].cullMode = cullForBucket[b];
            indirectGroups_[b].offset   = offset;
            indirectGroups_[b].count    = n;
            if (n > 0u) {
                // Patch firstInstance now that we know each draw's final
                // position in the concatenated DrawInfo / indirect arrays.
                // Entry-order assignment up above doesn't match concat
                // order once we partition by cull mode — the VS reads
                // `draws[gl_InstanceIndex]` so the encoded index has to
                // be the FINAL position, not the bucket-order index.
                for (uint32_t k = 0; k < n; ++k) {
                    cmds[b][k].firstInstance = offset + k;
                }
                std::memcpy(dDst + offset * sizeof(DrawInfoGpu),
                            draws[b].data(), n * sizeof(DrawInfoGpu));
                std::memcpy(cDst + offset * sizeof(VkDrawIndirectCommand),
                            cmds[b].data(), n * sizeof(VkDrawIndirectCommand));
                if (occlMetaDst)
                    std::memcpy(occlMetaDst + offset, occlMeta[b].data(),
                                n * sizeof(vulkan::OcclusionCull::CullMeta));
            }
            offset += n;
        }

        // Now that the bucket offsets are final, resolve each field's DrawInfo
        // index. Same reason the firstInstance patch above exists: entry order
        // is not concatenated order.
        particleDrawSlots_.reserve(pfSlots.size());
        for (const auto& s : pfSlots) {
            particleDrawSlots_.push_back(
                    {s.field, indirectGroups_[s.bucket].offset + s.pos, s.cull});
        }

        // Bucket-offset writes above land in [0, globalIdx) of each buffer —
        // flush exactly that; the occl meta rides its own persistently-mapped
        // buffer, flushed through its owner.
        flushHostWrites(ctx->allocator(), view().drawInfoBuffers[frame].alloc, 0, drawBytes);
        flushHostWrites(ctx->allocator(), view().indirectCmdBuffers[frame].alloc, 0, cmdBytes);
        if (occlMetaDst) occl_->flushMeta(frame, globalIdx);
        vmaUnmapMemory(ctx->allocator(), view().indirectCmdBuffers[frame].alloc);
        vmaUnmapMemory(ctx->allocator(), view().drawInfoBuffers[frame].alloc);

        // Rewrite binding 4 if the DrawInfo buffer handle moved (grow).
        // The indirect cmd buffer is consumed by vkCmdDrawIndirect — no
        // descriptor binding needed for it.
        if (drawGrown) {
            VkDescriptorBufferInfo dbInfo{};
            dbInfo.buffer = view().drawInfoBuffers[frame].handle;
            dbInfo.offset = 0;
            dbInfo.range  = VK_WHOLE_SIZE;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = view().rasterDescSets[frame];
            w.dstBinding      = 4;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo     = &dbInfo;
            vkUpdateDescriptorSets(ctx->device(), 1, &w, 0, nullptr);
        }

        // Publish the skip cache for this view + FIF slot.
        view().cachedIndirectGroups = indirectGroups_;
        view().cachedIndirectTotal  = indirectTotalDraws_;
        view().cachedOcclActive     = occlThisBuild;
        view().indirectBuiltSig[frame] = buildSig;
    }

    // Begin the raster G-buffer render pass and ship the prebuilt
    // indirect-draw groups via 1-4 vkCmdDrawIndirect calls (one per
    // active cull mode, plus a trailing blend-decal group on the
    // albedo-blend pipeline). Replaces the prior per-mesh draw loop —
    // see buildIndirectDrawData above for how the GPU buffers are
    // populated.
    void VulkanRenderer::Impl::recordRasterGbufPass(VkCommandBuffer cb, uint32_t frame) {
        const auto& g = view().rasterGbufs[frame];
        // MSAA path: rasterize into the MS framebuffer/render pass/
        // pipelines (RasterGbufImages::*MS + rasterGbufRenderPassMS);
        // the single-sample framebuffer/pipelines below stay exactly
        // what they were pre-MSAA — gbuf_resolve.comp (dispatched right
        // after this, see recordCommandBuffer) fills the single-sample
        // images every downstream consumer keeps reading unchanged.
        const bool useMsaa = gbufMsaaSamples_ > 1 && g.framebufferMS != VK_NULL_HANDLE;
        recordRasterGbufPassInternal(cb, frame,
                                     useMsaa ? rasterGbufRenderPassMS : rasterGbufRenderPass,
                                     useMsaa ? g.framebufferMS : g.framebuffer,
                                     useMsaa,
                                     view().indirectCmdBuffers[frame].handle,
                                     /*clear=*/true);
    }

    // Shared body: `renderPass` must be COMPATIBLE with the pipelines'
    // creation pass (the occlusion-culling load/store variants are), and
    // `indirectBuffer` supplies the VkDrawIndirectCommand records the
    // bucket groups index into (the two-phase path swaps in the compute-
    // written phase buffers; offsets/counts are identical by design).
    void VulkanRenderer::Impl::recordRasterGbufPassInternal(VkCommandBuffer cb, uint32_t frame,
                                      VkRenderPass renderPass, VkFramebuffer fb,
                                      bool useMsaa, VkBuffer indirectBuffer,
                                      bool clear, bool particles) {
        const auto& g = view().rasterGbufs[frame];
        if (fb == VK_NULL_HANDLE) return;// not initialized

        VkClearValue clears[6]{};
        clears[0].color = {{0.f, 0.f, 0.f, 0.f}};   // normal — sky/miss as zero
        clears[1].color = {{0.f, 0.f, 0.f, 0.f}};   // motion
        clears[2].color.uint32[0] = 0u;             // instanceID — 0 reserved as sky
        clears[2].color.uint32[1] = 0u;
        clears[2].color.uint32[2] = 0u;
        clears[2].color.uint32[3] = 0u;
        clears[3].color = {{0.f, 0.f, 0.f, 0.f}};   // uv — sky has no UV
        clears[4].color = {{0.f, 0.f, 0.f, 0.f}};   // albedo+metalness — sky as zero
        clears[5].depthStencil = {0.f, 0u};// reverse-Z: clear depth to 0 (far)

        VkRenderPassBeginInfo rpbi{};
        rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass      = renderPass;
        rpbi.framebuffer     = fb;
        rpbi.renderArea.offset = {0, 0};
        rpbi.renderArea.extent = {g.width, g.height};
        // LOAD-variant passes ignore the clear values but the count must
        // still cover every CLEAR attachment — harmless to always pass.
        rpbi.clearValueCount = clear ? 6u : 0u;
        rpbi.pClearValues    = clear ? clears : nullptr;
        vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        // Split-screen: draw the pane region-sized at the origin (the render
        // pass still clears the whole gbuf, so outside the pane reads as sky).
        // regionRenderExt_ == full render extent when scissorTest is off.
        const VkExtent2D gbufReg = regionRenderExt_;
        VkViewport viewport{};
        viewport.x = 0.f;
        viewport.y = 0.f;
        viewport.width  = float(gbufReg.width);
        viewport.height = float(gbufReg.height);
        viewport.minDepth = 0.f;
        viewport.maxDepth = 1.f;
        vkCmdSetViewport(cb, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = gbufReg;
        vkCmdSetScissor(cb, 0, 1, &scissor);

        if (indirectTotalDraws_ == 0u) {
            vkCmdEndRenderPass(cb);
            return;
        }
        const bool drawParticleFields = particles && !particleDrawSlots_.empty();

        const VkPipeline indirectPipe = useMsaa ? rasterGbufIndirectPipelineMS : rasterGbufIndirectPipeline;
        const VkPipeline decalPipe    = useMsaa ? rasterGbufDecalPipelineMS : rasterGbufDecalPipeline;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, indirectPipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                rasterPipelineLayout, 0, 1,
                                &view().rasterDescSets[frame], 0, nullptr);

        // One vkCmdDrawIndirect per cull-mode group. Empty groups skip;
        // a typical static scene fires one (BACK_cull); transmissive
        // sets can fire two or three. cullMode flips between calls
        // via the dynamic cullMode state — far cheaper than re-doing
        // it per draw.
        constexpr VkDeviceSize cmdStride = sizeof(VkDrawIndirectCommand);
        for (int b = 0; b < 3; ++b) {
            const auto& g = indirectGroups_[b];
            if (g.count == 0u) continue;
            vkCmdSetCullMode(cb, g.cullMode);
            vkCmdDrawIndirect(cb,
                              indirectBuffer,
                              static_cast<VkDeviceSize>(g.offset) * cmdStride,
                              g.count,
                              static_cast<uint32_t>(cmdStride));
        }
        // Blend decals last: albedo-blend pipeline lerps over the receivers
        // rasterized above (same layout/descriptors, no set rebind needed).
        if (const auto& g = indirectGroups_[3]; g.count > 0u) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, decalPipe);
            vkCmdSetCullMode(cb, g.cullMode);
            vkCmdDrawIndirect(cb,
                              indirectBuffer,
                              static_cast<VkDeviceSize>(g.offset) * cmdStride,
                              g.count,
                              static_cast<uint32_t>(cmdStride));
        }

        // ParticleFields last, on the sibling pipeline. Same render pass, same
        // descriptor set (already bound above), same fragment stage — only the
        // vertex stage and the source of the indirect record differ.
        if (drawParticleFields) recordParticleFieldDraws(cb, useMsaa);

        vkCmdEndRenderPass(cb);
    }

    // One vkCmdDrawIndirect per visible ParticleField. Each is a SINGLE record
    // whose instanceCount was written on the device by
    // recordParticleFieldCounts, so the number of particles never crosses the
    // PCI bus as a CPU-visible value — the entire point of the entity.
    //
    // Recorded inside the caller's render pass, with the raster descriptor set
    // already bound. The per-field push constant carries the DrawInfo index
    // (firstInstance is 0 here, because gl_InstanceIndex is the PARTICLE index)
    // plus the position / prev-position / orientation buffer addresses.
    void VulkanRenderer::Impl::recordParticleFieldDraws(VkCommandBuffer cb, bool useMsaa) {
        if (!particleFieldPass_) return;
        const VkPipeline pipe = useMsaa ? rasterGbufParticlePipelineMS : rasterGbufParticlePipeline;
        if (pipe == VK_NULL_HANDLE) return;
        const auto& states = particleFieldPass_->drawStates();
        if (states.empty()) return;

        // F4 distance LOD: the gate is a camera distance, so this view's world
        // camera position has to reach the vertex stage. The inverse of the
        // unjittered view matrix is the cheapest source of it that is certainly
        // the CURRENT view's (this function is called once per view, inside
        // that view's raster pass).
        float camPos[3]{0.f, 0.f, 0.f};
        {
            Matrix4 viewInv;
            std::memcpy(viewInv.elements.data(), view().currViewUnjit_.data(), 64);
            viewInv.invert();
            camPos[0] = static_cast<float>(viewInv.elements[12]);
            camPos[1] = static_cast<float>(viewInv.elements[13]);
            camPos[2] = static_cast<float>(viewInv.elements[14]);
        }

        bool bound = false;
        VkCullModeFlags curCull = VK_CULL_MODE_FLAG_BITS_MAX_ENUM;
        for (const auto& st : states) {
            if (st.vertexCount == 0u || st.indirect == VK_NULL_HANDLE) continue;
            // Match by identity, not by position: a field whose MeshRepr is off
            // (or whose proxy geometry has not uploaded yet) produces no
            // DrawInfo, so the two lists are not index-parallel. Fields per
            // scene are counted in ones, so the scan is free.
            const ParticleDrawSlot* slot = nullptr;
            for (const auto& s : particleDrawSlots_) {
                if (s.field == st.field) { slot = &s; break; }
            }
            if (!slot) continue;

            if (!bound) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                bound = true;
            }
            if (slot->cull != curCull) {
                vkCmdSetCullMode(cb, slot->cull);
                curCull = slot->cull;
            }

            const auto& cfg = st.field->config();
            ParticleFieldPC pc{};
            pc.posAddr     = st.posAddr;
            pc.prevPosAddr = st.prevPosAddr;
            pc.oriAddr     = st.oriAddr;
            pc.drawIdx     = slot->drawIdx;
            pc.wSemantic   = static_cast<uint32_t>(cfg.wSemantic);
            pc.invUniformRadius =
                    (cfg.uniformRadius > 0.f) ? (1.f / cfg.uniformRadius) : 0.f;
            {
                const auto& mr = st.field->meshRepr();
                pc.lodFar   = std::max(mr.lodFar, 0.f);
                pc.lodFade  = std::max(mr.lodFade, 0.f);
                pc.nearCull = std::max(mr.nearCull, 0.f);
            }
            pc.camPos[0] = camPos[0];
            pc.camPos[1] = camPos[1];
            pc.camPos[2] = camPos[2];
            vkCmdPushConstants(cb, rasterPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(pc), &pc);
            vkCmdDrawIndirect(cb, st.indirect, 0, 1, sizeof(VkDrawIndirectCommand));
        }
    }

    // Lazily create the debug_resolve compute pipeline + descriptor set.
    void VulkanRenderer::Impl::createDebugResolvePipeline() {
        if (debugResolvePipeline_ != VK_NULL_HANDLE) return;

        // 6 bindings: normal/motion/ids/albedo (combined image samplers;
        // ids is a usampler) + the swapchain storage image + depth (a
        // depth-aspect combined image sampler, for the depth AOV).
        std::array<VkDescriptorSetLayoutBinding, 6> b{};
        for (uint32_t i = 0; i < 4; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        b[4].binding         = 4;
        b[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[4].descriptorCount = 1;
        b[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[5].binding         = 5;
        b[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[5].descriptorCount = 1;
        b[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &debugResolveDsLayout_),
              "vkCreateDescriptorSetLayout(debug_resolve)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(DebugResolvePC);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &debugResolveDsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &debugResolvePipelineLayout_),
              "vkCreatePipelineLayout(debug_resolve)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kDebugResolveCompSpv);
        smci.pCode    = kDebugResolveCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(debug_resolve)");

        VkPipelineShaderStageCreateInfo ssci{};
        ssci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        ssci.module = mod;
        ssci.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = ssci;
        cpci.layout = debugResolvePipelineLayout_;
        check(vkCreateComputePipelines(ctx->device(), ctx->pipelineCache(), 1, &cpci, nullptr, &debugResolvePipeline_),
              "vkCreateComputePipelines(debug_resolve)");
        vkDestroyShaderModule(ctx->device(), mod, nullptr);

        // One set per frame-in-flight (see the member comment — a single
        // shared set rewritten per frame is the in-flight-update violation).
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 5 * kFramesInFlight;// normal/motion/ids/albedo + depth
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 1 * kFramesInFlight;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kFramesInFlight;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &debugResolveDescPool_),
              "vkCreateDescriptorPool(debug_resolve)");
        std::array<VkDescriptorSetLayout, kFramesInFlight> layouts{};
        layouts.fill(debugResolveDsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = debugResolveDescPool_;
        dsai.descriptorSetCount = kFramesInFlight;
        dsai.pSetLayouts        = layouts.data();
        check(vkAllocateDescriptorSets(ctx->device(), &dsai, debugResolveDescSets_.data()),
              "vkAllocateDescriptorSets(debug_resolve)");
    }

    // Debug visualization: sample one G-buffer channel and write a
    // readable RGB image straight to the swapchain via debug_resolve.comp.
    // Replaces the old raw vkCmdBlitImage, which could only show the
    // world-normal attachment: a blit cannot bias the SIGNED motion vector
    // (it clamped to near-black) and is INVALID from the integer ids
    // attachment (R16G16B16A16_UINT) into the UNORM swapchain (the Vulkan
    // spec forbids integer<->non-integer blits). The gbuf attachments are
    // in SHADER_READ_ONLY_OPTIMAL here (the gbuffer render pass declares a
    // COMPUTE consumer dependency, same as the deferred / event-shade
    // consumers), so no gbuf barrier is needed — we only move the
    // freshly-acquired swapchain image to GENERAL for the storage write
    // and leave it there, the exact exit contract the full deferred path
    // uses so endFrame()'s shared overlay/present finalize is unchanged.
    void VulkanRenderer::Impl::recordHybridDebugResolve(VkCommandBuffer cb, uint32_t imageIndex, uint32_t frame) {
        if (hybridDebugView_ == HybridDebugView::Off) return;
        createDebugResolvePipeline();

        const auto& g = view().rasterGbufs[frame];

        // Map the public view enum to the shader's compact code. Depth is
        // not reachable from setHybridDebugView (1..4) and has no sampled
        // path here, so anything else falls through to the normal view.
        uint32_t viewCode = 1;// Normal
        switch (hybridDebugView_) {
            case HybridDebugView::Motion: viewCode = 2; break;
            case HybridDebugView::Ids:    viewCode = 3; break;
            case HybridDebugView::Albedo: viewCode = 4; break;
            case HybridDebugView::Depth:  viewCode = 5; break;
            default:                      viewCode = 1; break;
        }

        auto img = [&](VkImageView view) {
            VkDescriptorImageInfo info{};
            info.sampler     = gbufSampler_;
            info.imageView   = view;
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            return info;
        };
        VkDescriptorImageInfo normalInfo = img(g.normal.view);
        VkDescriptorImageInfo motionInfo = img(g.motion.view);
        VkDescriptorImageInfo idsInfo    = img(g.ids.view);
        VkDescriptorImageInfo albedoInfo = img(g.albedo.view);

        // Depth is a depth-aspect image; it rests in DEPTH_STENCIL_READ_ONLY
        // (the gbuffer render pass's depth finalLayout), not the colour
        // SHADER_READ_ONLY the img() helper uses.
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler     = gbufSampler_;
        depthInfo.imageView   = g.depth.view;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo outInfo{};
        outInfo.imageView   = ctx->swapchainImageViews()[imageIndex];
        outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // This frame's own set: its previous use retired with the
        // inFlight[frame] fence wait at the top of the frame, so rewriting it
        // here cannot race the OTHER in-flight frame's pending command buffer
        // (which binds the other slot's set).
        VkDescriptorSet ds = debugResolveDescSets_[frame];
        std::array<VkWriteDescriptorSet, 6> w{};
        for (auto& it : w) it.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        const VkDescriptorImageInfo* imgs[4] = {&normalInfo, &motionInfo, &idsInfo, &albedoInfo};
        for (uint32_t i = 0; i < 4; ++i) {
            w[i].dstSet          = ds;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo      = imgs[i];
        }
        w[4].dstSet          = ds;
        w[4].dstBinding      = 4;
        w[4].descriptorCount = 1;
        w[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[4].pImageInfo      = &outInfo;
        w[5].dstSet          = ds;
        w[5].dstBinding      = 5;
        w[5].descriptorCount = 1;
        w[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[5].pImageInfo      = &depthInfo;
        vkUpdateDescriptorSets(ctx->device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);

        // Freshly-acquired swapchain image (contents undefined) → GENERAL
        // for the compute storage write.
        VkImageMemoryBarrier2 toGeneral{};
        toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toGeneral.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        toGeneral.srcAccessMask = 0;
        toGeneral.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = ctx->swapchainImages()[imageIndex];
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cb, &dep);

        const VkExtent2D ext = ctx->swapchainExtent();
        DebugResolvePC pc{};
        pc.view       = viewCode;
        pc.width      = ext.width;
        pc.height     = ext.height;
        pc.gbufWidth  = g.width;
        pc.gbufHeight = g.height;
        pc.motionGain = 20.0f;// signed NDC delta is tiny; scale for visibility

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, debugResolvePipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                debugResolvePipelineLayout_, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, debugResolvePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (ext.width + 7) / 8, (ext.height + 7) / 8, 1);
        // Swapchain left in GENERAL — the shared overlay/present finalize
        // expects exactly that.
    }

}// namespace threepp
