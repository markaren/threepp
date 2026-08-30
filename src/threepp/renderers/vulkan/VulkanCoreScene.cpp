#include "VulkanCoreImpl.hpp"

#include "VulkanCpuPhaseProf.hpp"

#include <algorithm>
#include <cfloat>

namespace threepp {

uint32_t VulkanRenderer::Impl::snapMeshFlags(Mesh& m, const MaterialWithWireframe* wf) const {
            uint32_t fl = kSnapKindMesh;
            if (auto geom = m.geometry()) {
                if (geom->hasAttribute("position")) fl |= kSnapHasPos;
                if (geom->hasAttribute("normal")) fl |= kSnapHasNorm;
            }
            if (wf && wf->wireframe) fl |= kSnapWire;
            if (overlayLayer_ >= 0 &&
                m.layers.isEnabled(static_cast<unsigned>(overlayLayer_))) fl |= kSnapOverlay;
            if (m.layers.isEnabled(VulkanRenderer::kSensorOnlyLayer)) fl |= kSnapSensorOnly;
            if (auto mat = m.material(); mat && mat->tetSkinning && mat->tetTexture) fl |= kSnapTet;
            // Unlit transparent untextured mesh → raster overlay routing (see
            // kSnapUiBlend). Textured basics stay traced — the overlay fill
            // pipelines have no sampler. Vertex-colored ones route too, via
            // the overlayMeshColoredPipelines / line-geometry-cache path
            // (drawOverlayMesh); before that path existed they stayed in the
            // G-buffer and rendered opaque, ignoring transparent/opacity.
            if (!(fl & kSnapWire)) {
                if (auto mat = m.material()) {
                    if (auto* mb = dynamic_cast<MeshBasicMaterial*>(mat.get());
                        mb && mb->transparent && !mb->map) fl |= kSnapUiBlend;
                }
            }
            // ParticleSystem billboard mesh — detected by the unique material-name
            // marker (cheap: a length-mismatch reject for the empty-named common
            // case). Routed to the dedicated billboard pass and excluded from the
            // traced/rasterized scene.
            if (auto mat = m.material(); mat && mat->name == kParticleMaterialName) fl |= kSnapParticle;
            return fl;
        }

bool VulkanRenderer::Impl::sceneSnapshotMatches(Object3D& scene, Camera& camera) {
            if (!sceneBuilt_ || sceneSnapshot_.empty()) return false;
            if (prevSceneFingerprint.size() != lastVisibleEntries_.size()) return false;
            size_t cur = 0;
            bool ok = true;
            scene.traverseVisible([&](Object3D& o) {
                if (!ok) return;
                if (cur >= sceneSnapshot_.size() || sceneSnapshot_[cur].obj != &o) {
                    ok = false;
                    return;
                }
                const SnapNode& sn = sceneSnapshot_[cur++];
                const uint32_t kind = sn.flags & kSnapKindMask;
                if (kind == kSnapKindOther) {
                    // LOD parity (see the full pass): re-run level selection
                    // every frame. Pre-order walk ⇒ the mutation lands before
                    // this LOD's children are visited, so a level switch makes
                    // the node sequence diverge from the snapshot at the
                    // swapped subtree ⇒ ok=false ⇒ full re-expansion picks up
                    // the new level's meshes.
                    if ((sn.flags & kSnapLod) != 0u && sn.lod->autoUpdate) sn.lod->update(camera);
                    return;// otherwise pointer identity is all we need
                }
                // Same object at the same address ⇒ same dynamic type — the
                // typed pointers recorded by the full pass are still valid, so
                // the walk pays zero dynamic_casts. attributesVersion()
                // unchanged ⇒ the hasPos/hasNorm flag bits are still valid
                // without re-doing the string-keyed attribute lookups.
                if (kind == kSnapKindLine || kind == kSnapKindPoints) {
                    auto geom = sn.line ? sn.line->geometry() : sn.pts->geometry();
                    auto matL = sn.line ? sn.line->material() : sn.pts->material();
                    const bool matHidden = matL && !matL->visible;
                    if (geom.get() != sn.geom ||
                        (sn.geomB && sn.geomB->attributesVersion() != sn.attrVer) ||
                        matHidden != ((sn.flags & kSnapMatHidden) != 0u)) ok = false;
                    return;
                }
                Mesh* m = sn.mesh;
                if (m->geometry().get() != sn.geom || m->material().get() != sn.mat) {
                    ok = false;
                    return;
                }
                if (sn.geomB && sn.geomB->attributesVersion() != sn.attrVer) {
                    ok = false;// attribute added/replaced/removed → full pass re-derives
                    return;
                }
                // Dynamic routing bits — plain bool reads, no lookups. The
                // pointed-to material is alive (the mesh holding it just
                // compared equal) and its dynamic type is fixed.
                const bool wire = sn.wf && sn.wf->wireframe;
                const bool over = overlayLayer_ >= 0 &&
                                  o.layers.isEnabled(static_cast<unsigned>(overlayLayer_));
                const bool sensor = o.layers.isEnabled(VulkanRenderer::kSensorOnlyLayer);
                const bool tet = sn.mat && sn.mat->tetSkinning && sn.mat->tetTexture != nullptr;
                const bool particle = sn.mat && sn.mat->name == kParticleMaterialName;
                const bool uiBlend = !wire && sn.basic && sn.basic->transparent &&
                                     !sn.basic->map;
                // sn.mat compared equal above, so it's alive and dereferenceable
                // (nullptr ⇒ no material ⇒ treated as visible). [[#mat-visible]]
                const bool matHidden = sn.mat && !sn.mat->visible;
                if (wire != ((sn.flags & kSnapWire) != 0u) ||
                    over != ((sn.flags & kSnapOverlay) != 0u) ||
                    sensor != ((sn.flags & kSnapSensorOnly) != 0u) ||
                    tet != ((sn.flags & kSnapTet) != 0u) ||
                    particle != ((sn.flags & kSnapParticle) != 0u) ||
                    uiBlend != ((sn.flags & kSnapUiBlend) != 0u) ||
                    matHidden != ((sn.flags & kSnapMatHidden) != 0u)) {
                    ok = false;
                    return;
                }
                if (sn.instCount >= 0 &&
                    sn.inst->count() != static_cast<size_t>(sn.instCount)) {
                    ok = false;
                }
            });
            return ok && cur == sceneSnapshot_.size();
        }

// One mapping, then only the ranges the host actually patched. Shared by both
// desc rings because they differ in nothing but element type: entries-indexed
// arrays whose per-slot pending set is a DescDirtyRanges. The whole-array
// fallback stays one memcpy of everything — a structural rebuild has no smaller
// truth to tell.
namespace {
    // `Dirty` is always Impl::DescDirtyRanges, and it is a DEDUCED parameter
    // rather than that spelled-out type for an access reason: this is a
    // NON-MEMBER, so unlike every `T VulkanRenderer::Impl::f()` below it gets no
    // access grant to VulkanRenderer's private nested Impl. GCC defers a
    // template's access checks to instantiation and re-runs them from the
    // instantiation context — namespace scope — and rejects it there ("'struct
    // VulkanRenderer::Impl' is private within this context"); MSVC and Clang
    // check once at the declaration and let it through. Deducing the type keeps
    // the private name out of namespace scope altogether. Both call sites are
    // Impl members a few lines down, so nothing else can supply it.
    template<typename T, typename Dirty>
    void uploadDescRanges(VmaAllocator alloc, const vulkan::Buffer& buf,
                          const std::vector<T>& src,
                          const Dirty& dirty) {
        if (dirty.whole) {
            vulkan::uploadHostVisible(alloc, buf, src.data(), src.size() * sizeof(T));
            return;
        }
        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(alloc, buf.alloc, &mapped),
                      "vmaMapMemory(uploadDescRanges)");
        for (auto [first, last] : dirty.ranges) {
            // Clamp rather than assert: a shrink between mark and flush goes
            // through a structural rebuild, which sets `whole`, so this is only
            // belt and braces — but it makes a range list unable to overrun the
            // buffer no matter who marked it.
            if (last > src.size()) last = static_cast<uint32_t>(src.size());
            if (first >= last) continue;
            const VkDeviceSize off = VkDeviceSize(first) * sizeof(T);
            const VkDeviceSize n = VkDeviceSize(last - first) * sizeof(T);
            std::memcpy(static_cast<uint8_t*>(mapped) + off, src.data() + first, n);
            // Per range. VMA widens a non-coherent flush to nonCoherentAtomSize
            // itself, which can only ever publish neighbouring bytes that are
            // already coherent — a flush publishes host writes and invalidates
            // nothing, so a wider range is never wrong.
            vulkan::check(vmaFlushAllocation(alloc, buf.alloc, off, n),
                          "vmaFlushAllocation(uploadDescRanges)");
        }
        vmaUnmapMemory(alloc, buf.alloc);
    }
}// namespace

// A genuine addition, not a split: scene.8_matDescPatch times the HOST-side
// re-derivation inside ensureSceneBuilt and contains no device copy. The copy is
// here, on the other side of the fence wait, and was in no counter at all.
//
// It used to be whole-array: entries-indexed at 608 B/entry, so ONE changed
// material re-sent every entry's MaterialDesc, to every frame-in-flight slot —
// 47.7 MB and 2.8 ms per frame at 78.4k grains, for the two belt meshes whose
// scrolling texture bumps their material version. Now it sends the entry ranges
// ensureSceneBuilt actually patched (see DescDirtyRanges).
void VulkanRenderer::Impl::flushMaterialDescsIfDirty(uint32_t frame) {
            THREEPP_CPUPROF("frame.I_uploadMatDesc");
            auto& dirty = matDescsDirty_[frame];
            if (!dirty.any()) return;
            if (matDescsCached_.empty() ||
                materialDescsBuffers[frame].handle == VK_NULL_HANDLE) {
                dirty.clear();
                return;
            }
            uploadDescRanges(ctx->allocator(), materialDescsBuffers[frame],
                             matDescsCached_, dirty);
            dirty.clear();
        }

// Same shape, 64 B/entry. Dirtied per ENTRY by the frame.B_movedSticky walk on a
// 0↔1 sticky transition, and whole-array by an auto-LOD level switch — but the
// copy lands here, outside those phases, so it needs its own key.
void VulkanRenderer::Impl::flushGeometryDescsIfDirty(uint32_t frame) {
            THREEPP_CPUPROF("frame.I2_uploadGeomDesc");
            auto& dirty = geomDescsDirty_[frame];
            if (!dirty.any()) return;
            if (geomDescsCached_.empty() ||
                geometryDescsBuffers[frame].handle == VK_NULL_HANDLE) {
                dirty.clear();
                return;
            }
            uploadDescRanges(ctx->allocator(), geometryDescsBuffers[frame],
                             geomDescsCached_, dirty);
            dirty.clear();
        }

void VulkanRenderer::Impl::cullEntriesAgainstFrustum(Camera& camera) {
            THREEPP_CPUPROF("frame.C_frustumCull");
            if (lastVisibleEntries_.empty()) return;
            // Results land in THIS view's array, never on the shared entry —
            // see ViewContext::inFrustum for why that distinction is load-
            // bearing. Default-include on grow so a freshly added view draws
            // everything for the one frame before its first cull.
            auto& cull = view().inFrustum;
            // Combine projection * matrixWorldInverse to extract the world-
            // space frustum (Three.js convention; Camera::updateMatrixWorld
            // already ran in updateCameraUbo this frame).
            Matrix4 vp;
            vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
            // Same frustum + same scene inputs as the last cull ⇒ inFrustum
            // already holds the right answer — skip the walk entirely. (The
            // sticky-only drawInputsVersion_ bumps re-run it for ~30 frames
            // after motion stops; harmless.)
            if (view().cullValidVersion == drawInputsVersion_ &&
                cull.size() == lastVisibleEntries_.size() &&
                std::memcmp(view().prevCullVp, vp.elements.data(), 64) == 0) {
                return;
            }
            cull.assign(lastVisibleEntries_.size(), uint8_t{1});
            Frustum frustum;
            frustum.setFromProjectionMatrix(vp);
            const auto& planes = frustum.planes();
            auto& entries = lastVisibleEntries_;

            // Per SPAN. The old per-entry loop paid a shared_ptr geometry()
            // fetch plus an 8-corner Box3::applyMatrix4 (with per-point
            // perspective divides) per instance per frame — ~12 ms at 115k
            // static instances. Now: one span-level AABB classify answers
            // fully-outside / fully-inside for the whole run, and only spans
            // that straddle the frustum pay a per-entry test (a world-sphere
            // test — conservative vs the old box test, so it can only KEEP
            // more, never cull more).
            for (auto& sp : entrySpans_) {
                MeshEntry& e0 = entries[sp.first];
                // Default-include conservative cases — they always draw (the
                // assign(1) above already covers them). Deformers (skinned/
                // displaced/morphed/tet) because their cached local AABB
                // doesn't reflect the per-frame deformed extents; frustum-
                // culling them risks popping a still-on-screen body out of
                // the gbuffer.
                // ParticleField joins the deformers for a stronger reason than
                // theirs: its particles live on the device, so the CPU has no
                // bound to test at all — not a stale one, none. A field is never
                // frustum-culled at entry granularity; per-particle culling, if
                // it is ever wanted, belongs in the expansion shader.
                // enableVertexInterop meshes join for the ParticleField reason,
                // not the deformer one: the shape on screen is whatever a foreign
                // CUDA producer wrote into the exported buffer this frame, and the
                // host `position` array these bounds are derived from was last
                // meaningful (if ever) at build time. A producer writing an
                // in-view triangle while its host array sits at y=-50 rendered
                // NOTHING before this line existed — the span's world AABB was
                // 50 m below the camera and the whole entry was culled out of the
                // G-buffer. NB threepp's Object3D::frustumCulled does not help:
                // the Vulkan backend never reads it (it is a GL-renderer flag),
                // so this exemption list IS the only opt-out that exists here.
                if (e0.isOverlay || e0.isSkinned || e0.isDisplaced || e0.isMorphed ||
                    e0.isTet || e0.isParticleField || e0.isVertexInterop)
                    continue;
                if (e0.isGrass) {
                    // Grass CAN be frustum-culled: unlike the other deformers, its
                    // deformed extent has a tight provable bound. The CPU position
                    // attribute is the rest pose; grass_wind.comp displaces each
                    // vertex by windDir·bend with |bend| ≤ 0.85·windStrength, so the
                    // rest AABB dilated by windStrength conservatively encloses every
                    // swayed pose (windDir is ~unit). Test THAT box normally — this
                    // is what lets a large tiled meadow cull its off-screen tiles.
                    // (The tile still stays in the TLAS for shadows/reflections/GI;
                    // inFrustum only gates the raster G-buffer draw.)
                    for (uint32_t j = 0; j < sp.count; ++j) {
                        const size_t ci = size_t(sp.first) + j;
                        Box3 worldAabb;
                        if (!grassSwayWorldAabb(entries[ci], worldAabb)) continue;// keep
                        cull[ci] = frustum.intersectsBox(worldAabb) ? uint8_t{1} : uint8_t{0};
                    }
                    continue;
                }
                // Lazily (re-)derive the shared object-space bounds — cleared
                // when the geometry is edited in place.
                if (!sp.localBoundsValid) {
                    auto geom = sp.mesh->geometry();
                    if (!geom) continue;// keep(true)
                    if (!geom->boundingBox) geom->computeBoundingBox();
                    if (!geom->boundingBox) continue;// keep(true)
                    const Vector3 c = geom->boundingBox->getCenter();
                    const Vector3 h = geom->boundingBox->getSize() * 0.5f;
                    sp.localCenter[0] = c.x; sp.localCenter[1] = c.y; sp.localCenter[2] = c.z;
                    sp.localHalf[0]   = h.x; sp.localHalf[1]   = h.y; sp.localHalf[2]   = h.z;
                    sp.localRadius    = h.length();// enclosing-sphere radius
                    sp.localBoundsValid = true;
                    sp.aabbValid = false;
                }
                // Lazily recompute the span's world AABB after a matrix
                // refresh. A single-entry span gets the EXACT world box (Arvo
                // transform — identical to the old 8-corner result); an
                // instanced span gets the union of world sphere centers
                // dilated by radius x max scale (conservative).
                if (!sp.aabbValid) {
                    const float* lc = sp.localCenter;
                    if (sp.count == 1) {
                        const float* M = entries[sp.first].worldMatrix.data();
                        const float cx = M[0]*lc[0] + M[4]*lc[1] + M[8]*lc[2] + M[12];
                        const float cy = M[1]*lc[0] + M[5]*lc[1] + M[9]*lc[2] + M[13];
                        const float cz = M[2]*lc[0] + M[6]*lc[1] + M[10]*lc[2] + M[14];
                        const float hx = std::abs(M[0])*sp.localHalf[0] + std::abs(M[4])*sp.localHalf[1] + std::abs(M[8])*sp.localHalf[2];
                        const float hy = std::abs(M[1])*sp.localHalf[0] + std::abs(M[5])*sp.localHalf[1] + std::abs(M[9])*sp.localHalf[2];
                        const float hz = std::abs(M[2])*sp.localHalf[0] + std::abs(M[6])*sp.localHalf[1] + std::abs(M[10])*sp.localHalf[2];
                        sp.aabbMin[0] = cx - hx; sp.aabbMin[1] = cy - hy; sp.aabbMin[2] = cz - hz;
                        sp.aabbMax[0] = cx + hx; sp.aabbMax[1] = cy + hy; sp.aabbMax[2] = cz + hz;
                    } else {
                        // Chunked accumulation: one AABB per kCullChunk-entry
                        // run (union of world sphere centers dilated by
                        // radius x chunk max scale) + the span AABB as the
                        // union of the chunks.
                        constexpr uint32_t kCullChunk = 512;
                        const uint32_t nChunks = (sp.count + kCullChunk - 1u) / kCullChunk;
                        sp.chunkAabbs.resize(nChunks);
                        float smn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
                        float smx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                        for (uint32_t ch = 0; ch < nChunks; ++ch) {
                            const uint32_t j0 = ch * kCullChunk;
                            const uint32_t j1 = std::min(sp.count, j0 + kCullChunk);
                            float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
                            float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                            float maxScaleSq = 0.f;
                            for (uint32_t j = j0; j < j1; ++j) {
                                const float* M = entries[sp.first + j].worldMatrix.data();
                                const float cx = M[0]*lc[0] + M[4]*lc[1] + M[8]*lc[2] + M[12];
                                const float cy = M[1]*lc[0] + M[5]*lc[1] + M[9]*lc[2] + M[13];
                                const float cz = M[2]*lc[0] + M[6]*lc[1] + M[10]*lc[2] + M[14];
                                mn[0] = std::min(mn[0], cx); mx[0] = std::max(mx[0], cx);
                                mn[1] = std::min(mn[1], cy); mx[1] = std::max(mx[1], cy);
                                mn[2] = std::min(mn[2], cz); mx[2] = std::max(mx[2], cz);
                                const float s0 = M[0]*M[0] + M[1]*M[1] + M[2]*M[2];
                                const float s1 = M[4]*M[4] + M[5]*M[5] + M[6]*M[6];
                                const float s2 = M[8]*M[8] + M[9]*M[9] + M[10]*M[10];
                                maxScaleSq = std::max(maxScaleSq, std::max(s0, std::max(s1, s2)));
                            }
                            const float r = sp.localRadius * std::sqrt(maxScaleSq);
                            auto& cb = sp.chunkAabbs[ch];
                            for (int k = 0; k < 3; ++k) {
                                cb[k]     = mn[k] - r;
                                cb[k + 3] = mx[k] + r;
                                smn[k] = std::min(smn[k], cb[k]);
                                smx[k] = std::max(smx[k], cb[k + 3]);
                            }
                        }
                        for (int k = 0; k < 3; ++k) { sp.aabbMin[k] = smn[k]; sp.aabbMax[k] = smx[k]; }
                    }
                    sp.aabbValid = true;
                }
                // Span-level classify: 0 = fully outside, 2 = fully inside,
                // 1 = straddles (per-entry tests decide).
                int cls = 2;
                for (const auto& plane : planes) {
                    const auto& n = plane.normal;
                    // p-vertex (corner at max signed distance) below the plane
                    // ⇒ the whole box is outside.
                    const float px = n.x > 0 ? sp.aabbMax[0] : sp.aabbMin[0];
                    const float py = n.y > 0 ? sp.aabbMax[1] : sp.aabbMin[1];
                    const float pz = n.z > 0 ? sp.aabbMax[2] : sp.aabbMin[2];
                    if (n.x*px + n.y*py + n.z*pz + plane.constant < 0) { cls = 0; break; }
                    // n-vertex below ⇒ the box straddles this plane.
                    const float qx = n.x > 0 ? sp.aabbMin[0] : sp.aabbMax[0];
                    const float qy = n.y > 0 ? sp.aabbMin[1] : sp.aabbMax[1];
                    const float qz = n.z > 0 ? sp.aabbMin[2] : sp.aabbMax[2];
                    if (n.x*qx + n.y*qy + n.z*qz + plane.constant < 0) cls = 1;
                }
                if (cls == 0) {
                    std::fill(cull.begin() + sp.first, cull.begin() + sp.first + sp.count, uint8_t{0});
                    continue;
                }
                if (cls == 2 || sp.count == 1) continue;// keep(1) — single spans were classified exactly
                // Straddling instanced span: classify chunk AABBs first, then
                // per-entry world-sphere tests only for straddling chunks.
                const float* lc = sp.localCenter;
                constexpr uint32_t kCullChunk = 512;
                for (uint32_t ch = 0; ch < sp.chunkAabbs.size(); ++ch) {
                    const auto& cb = sp.chunkAabbs[ch];
                    int ccls = 2;
                    for (const auto& plane : planes) {
                        const auto& n = plane.normal;
                        const float px = n.x > 0 ? cb[3] : cb[0];
                        const float py = n.y > 0 ? cb[4] : cb[1];
                        const float pz = n.z > 0 ? cb[5] : cb[2];
                        if (n.x*px + n.y*py + n.z*pz + plane.constant < 0) { ccls = 0; break; }
                        const float qx = n.x > 0 ? cb[0] : cb[3];
                        const float qy = n.y > 0 ? cb[1] : cb[4];
                        const float qz = n.z > 0 ? cb[2] : cb[5];
                        if (n.x*qx + n.y*qy + n.z*qz + plane.constant < 0) ccls = 1;
                    }
                    const uint32_t j0 = ch * kCullChunk;
                    const uint32_t j1 = std::min(sp.count, j0 + kCullChunk);
                    if (ccls == 0) {
                        std::fill(cull.begin() + sp.first + j0, cull.begin() + sp.first + j1, uint8_t{0});
                        continue;
                    }
                    if (ccls == 2) continue;// whole chunk stays visible
                    for (uint32_t j = j0; j < j1; ++j) {
                        const size_t ci = size_t(sp.first) + j;
                        const float* M = entries[ci].worldMatrix.data();
                        const float cx = M[0]*lc[0] + M[4]*lc[1] + M[8]*lc[2] + M[12];
                        const float cy = M[1]*lc[0] + M[5]*lc[1] + M[9]*lc[2] + M[13];
                        const float cz = M[2]*lc[0] + M[6]*lc[1] + M[10]*lc[2] + M[14];
                        const float s0 = M[0]*M[0] + M[1]*M[1] + M[2]*M[2];
                        const float s1 = M[4]*M[4] + M[5]*M[5] + M[6]*M[6];
                        const float s2 = M[8]*M[8] + M[9]*M[9] + M[10]*M[10];
                        const float r = sp.localRadius *
                                        std::sqrt(std::max(s0, std::max(s1, s2)));
                        uint8_t vis = 1;
                        for (const auto& plane : planes) {
                            const auto& n = plane.normal;
                            if (n.x*cx + n.y*cy + n.z*cz + plane.constant < -r) { vis = 0; break; }
                        }
                        cull[ci] = vis;
                    }
                }
            }

            // Track whether the results changed — buildIndirectDrawData's
            // skip signature depends on the camera only through these bits.
            auto& prev = view().prevInFrustum;
            if (prev.size() != cull.size() ||
                std::memcmp(prev.data(), cull.data(), cull.size()) != 0) {
                ++view().cullVersion;
                prev = cull;
            }
            // Stamp the recompute gate: these results stay valid until the
            // frustum or the scene draw inputs change.
            view().cullValidVersion = drawInputsVersion_;
            std::memcpy(view().prevCullVp, vp.elements.data(), 64);
        }

void VulkanRenderer::Impl::ensureSceneBuilt(Object3D& scene, Camera& camera) {
            // force=false (matching GLRenderer): with
            // updateMatrix()'s change-detection early-out, only subtrees whose
            // transforms actually moved pay the world-matrix multiplies — a
            // forced pass re-multiplied every node every frame (several
            // ms/frame on Bistro's node count, static or not).
            {
                THREEPP_CPUPROF("scene.1_updateMatrixWorld");
                scene.updateMatrixWorld();
            }
            // A parentless camera (the common case — not add()ed to the scene)
            // is untouched by the scene's updateMatrixWorld, but the LOD level
            // selection below reads camera.matrixWorld. Same guard as
            // GLRenderer::render; change-detected early-out ⇒ ~free.
            if (!camera.parent) camera.updateMatrixWorld();

            // Honour live edits to already-uploaded material textures (e.g. a
            // DataTexture re-baked via setData + needsUpdate). Cheap version
            // scan; only re-uploads + rewrites descriptors when something changed.
            refreshDirtyMaterialTextures();

            std::fill(meshMovedBits_.begin(), meshMovedBits_.end(), 0u);

            // SNAPSHOT FAST PATH: replay the last expansion's traversal against
            // cheap invariants. On a match (the overwhelmingly common frame —
            // including matrix-only motion like a driving car or URDF joints),
            // skip the expansion AND the fingerprint re-derivation entirely:
            // reuse last frame's entries/fingerprints, refresh world matrices
            // and version numbers from the live objects, and fall into the
            // SAME diff + non-structural tail below. Mutations are still
            // caught: matrices by the memcmp diff, material/geometry edits by
            // their version reads, structure/visibility/texture-routing
            // changes by the snapshot walk itself (mismatch ⇒ full path).
            bool snapLean;
            {
                THREEPP_CPUPROF("scene.2_snapshotMatch");
                snapLean = sceneSnapshotMatches(scene, camera);
            }
            std::vector<MeshEntry>& entries = lastVisibleEntries_;// canonical list, cached across frames
            std::vector<MeshFingerprint> currFp;
            if (snapLean) {
                THREEPP_CPUPROF("scene.3_leanMatrixRefresh");
                // Per SPAN: a span's entry world matrices are a pure function
                // of (mesh->matrixWorld, instanceMatrix contents). Neither
                // moved ⇒ every entry matrix is bit-identical to last frame ⇒
                // skip the span entirely — the old per-entry recompute paid
                // getMatrixAt + a 4x4 multiply per instance per frame on
                // fully static fields. instanceMatrix edits are version-gated
                // (setMatrixAt + needsUpdate(), the same contract the GL
                // backend's attribute upload requires).
                for (auto& sp : entrySpans_) {
                    sp.movedThisFrame = false;
                    const float* mw = sp.mesh->matrixWorld->elements.data();
                    if (sp.inst) {
                        const unsigned int v = sp.inst->instanceMatrix()->version;
                        if (v == sp.instMatVersion &&
                            std::memcmp(sp.meshWorld.data(), mw, 64) == 0) continue;
                        sp.instMatVersion = v;
                        std::memcpy(sp.meshWorld.data(), mw, 64);
                        sp.movedThisFrame = true;
                        sp.aabbValid = false;
                        // Tight refresh: world = meshWorld * instanceMat[j],
                        // straight off the raw attribute floats (no Matrix4
                        // temporaries, no bounds-checked getMatrixAt).
                        const auto& arr = sp.inst->instanceMatrix()->array();
                        for (uint32_t j = 0; j < sp.count; ++j) {
                            const float* b = arr.data() + size_t(j) * 16u;
                            float* o = entries[sp.first + j].worldMatrix.data();
                            for (int c = 0; c < 4; ++c) {
                                const float b0 = b[c * 4 + 0], b1 = b[c * 4 + 1],
                                            b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
                                o[c * 4 + 0] = mw[0] * b0 + mw[4] * b1 + mw[8] * b2 + mw[12] * b3;
                                o[c * 4 + 1] = mw[1] * b0 + mw[5] * b1 + mw[9] * b2 + mw[13] * b3;
                                o[c * 4 + 2] = mw[2] * b0 + mw[6] * b1 + mw[10] * b2 + mw[14] * b3;
                                o[c * 4 + 3] = mw[3] * b0 + mw[7] * b1 + mw[11] * b2 + mw[15] * b3;
                            }
                            // Keep the fingerprint's matrix mirror fresh so a
                            // later non-lean frame's generic compare sees this
                            // frame's state, not a stale lean-era matrix.
                            prevSceneFingerprint[sp.first + j].matrix =
                                    entries[sp.first + j].worldMatrix;
                        }
                    } else {
                        if (std::memcmp(sp.meshWorld.data(), mw, 64) == 0) continue;
                        std::memcpy(sp.meshWorld.data(), mw, 64);
                        sp.movedThisFrame = true;
                        sp.aabbValid = false;
                        std::memcpy(entries[sp.first].worldMatrix.data(), mw, 64);
                        prevSceneFingerprint[sp.first].matrix = entries[sp.first].worldMatrix;
                    }
                }
                for (auto& le : lastVisibleLines_) {
                    const Object3D* src = le.line ? static_cast<const Object3D*>(le.line)
                                                  : static_cast<const Object3D*>(le.points);
                    std::memcpy(le.worldMatrix.data(), src->matrixWorld->elements.data(), 64);
                }
            } else {
            THREEPP_CPUPROF("scene.3b_fullExpansion");
            // Expand the visible scene into one MeshEntry per TLAS instance.
            // Regular meshes contribute one entry; an InstancedMesh contributes
            // count() entries each with worldMatrix = matrixWorld * instanceMat[i].
            std::vector<MeshEntry> built;
            std::vector<LineEntry> builtLines;
            std::vector<EntrySpan> builtSpans;
            // (field, its entry index). Rebuilt only here: a field appearing or
            // disappearing is a structural change, so the snapshot fast path
            // can't reach a frame where this list is stale.
            std::vector<std::pair<ParticleField*, uint32_t>> builtFields;
            sceneSnapshot_.clear();
            // Rebuilt below as the walk visits each threepp::LOD node — a
            // mesh entry whose parent chain hits one of these is exempt from
            // auto-LOD (see the selection pass just after this expansion).
            manualLodLevelRoots_.clear();
            // traverseVisible (not traverse) so an invisible parent hides its
            // whole subtree — matches three.js / GLRenderer convention. Plain
            // `traverse` walks every node regardless of visibility, leaking
            // children of hidden groups into the scene/overlay passes.
            scene.traverseVisible([&](Object3D& o) {
                SnapNode sn{};
                sn.obj = &o;
                // three.js parity — LOD level selection (GLRenderer::
                // projectObject runs lod.update(camera) as it projects).
                // traverseVisible is PRE-ORDER: this LOD's children are
                // visited after this callback, so the rest of this walk
                // already sees the level picked for this frame's camera.
                // Recorded with a typed view + kSnapLod so the replay walk
                // re-runs the selection cast-free (Object3D is a virtual
                // base — the replay cannot static_cast down).
                if (auto* lod = dynamic_cast<LOD*>(&o)) {
                    if (lod->autoUpdate) lod->update(camera);
                    sn.lod   = lod;
                    sn.flags = kSnapKindOther | kSnapLod;
                    sceneSnapshot_.push_back(sn);
                    // Auto-LOD exemption: every level object under a manual
                    // LOD node is author-selected already; don't let auto-LOD
                    // swap index buffers underneath that choice.
                    for (Object3D* child : lod->children) manualLodLevelRoots_.insert(child);
                    return;
                }
                // Line / LineSegments: never part of the traced/rasterized
                // scene, always overlay.
                // Collected before the Mesh dispatch so subclasses don't
                // accidentally route through the Mesh path.
                if (auto* line = dynamic_cast<Line*>(&o); line && line->visible) {
                    auto geom = line->geometry();
                    sn.flags = kSnapKindLine;
                    sn.line  = line;
                    sn.geom  = geom.get();
                    sn.geomB = geom.get();
                    if (geom) sn.attrVer = geom->attributesVersion();
                    const bool matHidden = line->material() && !line->material()->visible;
                    if (matHidden) sn.flags |= kSnapMatHidden;
                    if (!matHidden && geom && geom->hasAttribute("position")) {
                        sn.flags |= kSnapHasPos;
                        LineEntry le{};
                        le.line       = line;
                        le.points     = nullptr;
                        le.isSegments = (dynamic_cast<LineSegments*>(line) != nullptr);
                        le.isPoints   = false;
                        std::memcpy(le.worldMatrix.data(),
                                    line->matrixWorld->elements.data(), 64);
                        builtLines.push_back(le);
                    }
                    sceneSnapshot_.push_back(sn);
                    return;// Lines aren't Meshes; nothing more to do
                }
                // Points (point clouds) — never part of the traced/rasterized
                // scene, always rasterise as POINT_LIST in the overlay pass.
                // Shares the same geometry cache as Line via the
                // BufferGeometry* key.
                if (auto* pts = dynamic_cast<Points*>(&o); pts && pts->visible) {
                    auto geom = pts->geometry();
                    sn.flags = kSnapKindPoints;
                    sn.pts   = pts;
                    sn.geom  = geom.get();
                    sn.geomB = geom.get();
                    if (geom) sn.attrVer = geom->attributesVersion();
                    const bool matHidden = pts->material() && !pts->material()->visible;
                    if (matHidden) sn.flags |= kSnapMatHidden;
                    if (!matHidden && geom && geom->hasAttribute("position")) {
                        sn.flags |= kSnapHasPos;
                        LineEntry le{};
                        le.line       = nullptr;
                        le.points     = pts;
                        le.isSegments = false;
                        le.isPoints   = true;
                        std::memcpy(le.worldMatrix.data(),
                                    pts->matrixWorld->elements.data(), 64);
                        builtLines.push_back(le);
                    }
                    sceneSnapshot_.push_back(sn);
                    return;
                }
                auto* m = dynamic_cast<Mesh*>(&o);
                if (!m || !m->visible) {
                    sceneSnapshot_.push_back(sn);// kind Other: pointer identity only
                    return;
                }
                // Hybrid overlay (raster-only) classification. Wireframe-
                // flagged materials and overlay-layer membership both route
                // the mesh to the post-TAA raster overlay pass and exclude
                // it from the traced/rasterized scene (TLAS, raster G-buffer,
                // emissive NEE). The scene render can't see/shadow overlay
                // meshes — they're pure debug visuals.
                const MaterialWithWireframe* wf = nullptr;
                if (auto mat = m->material()) {
                    wf = dynamic_cast<MaterialWithWireframe*>(mat.get());
                }
                auto* inst = dynamic_cast<InstancedMesh*>(m);
                sn.mesh      = m;
                sn.inst      = inst;
                sn.geom      = m->geometry().get();
                sn.geomB     = m->geometry().get();
                sn.mat       = m->material().get();
                sn.wf        = wf;
                sn.basic     = dynamic_cast<const MeshBasicMaterial*>(sn.mat);
                sn.instCount = inst ? static_cast<int32_t>(inst->count()) : -1;
                sn.flags     = snapMeshFlags(*m, wf);
                // material()->visible == false: record the full mesh node (so a
                // re-show is caught by the replay walk) but build no entries, so
                // the TLAS / raster G-buffer / overlay / emissive NEE never see it.
                // [[#mat-visible]]
                const bool matHidden = m->material() && !m->material()->visible;
                if (matHidden) sn.flags |= kSnapMatHidden;
                if (sn.geomB) sn.attrVer = sn.geomB->attributesVersion();
                sceneSnapshot_.push_back(sn);
                if (matHidden) return;
                // SplatClouds are composited by the compute tile rasterizer
                // (SplatPass), not by the G-buffer. The unit quad they carry is
                // a GL-backend implementation detail — it has no normals and
                // describes no surface — so it is recorded in the snapshot (an
                // add/remove still invalidates the scene) and then dropped
                // before any entry, BLAS or TLAS instance is built from it.
                // The missing "normal" attribute would exclude it two lines
                // below anyway; saying so out loud is the difference between an
                // invariant and a coincidence.
                if (dynamic_cast<SplatCloud*>(m)) return;
                auto geom = m->geometry();
                if (!geom || !geom->hasAttribute("position")) return;
                if (!geom->hasAttribute("normal")) return;
                // Particle billboard meshes are excluded from the traced/
                // rasterized scene exactly like overlays (kSnapParticle folds
                // into isOverlay so every
                // `if (en.isOverlay) continue;` guard applies) but carry their
                // own isParticle flag so the billboard pass claims them and the
                // overlay-mesh loop skips them.
                const bool isParticle = (sn.flags & kSnapParticle) != 0u;
                const bool isOverlay = (sn.flags & (kSnapWire | kSnapOverlay | kSnapParticle | kSnapUiBlend)) != 0u;
                // Sensor-only geometry (kSensorOnlyLayer): still a full scene
                // entry — BLAS, descs, TLAS instance — because the lidar reads
                // those tables. Only the raster views it reaches and the ray
                // masks that include it differ.
                const bool sensorOnly = (sn.flags & kSnapSensorOnly) != 0u;
                // One-shot type probes: an N-instance InstancedMesh costs 3
                // dynamic_casts total, not 3·N — and on snapshot-match frames
                // none at all (the cached entry flags are reused). Consumed by
                // raster pass loops, resolveBlasForEntry, TLAS refit, and
                // dirty-detection.
                const bool isSkinned   = (dynamic_cast<SkinnedMesh*>(m)   != nullptr);
                const bool isDisplaced = (dynamic_cast<DisplacedMesh*>(m) != nullptr);
                const bool isGrass     = (dynamic_cast<GrassMesh*>(m)     != nullptr);
                // A ParticleField expands to exactly one entry whatever its
                // capacity — the property the type exists to provide. It is
                // classified here rather than dropped like SplatCloud because it
                // must reach the entry list: it is the field's one entry that
                // carries the world matrix, the entry index the FieldDesc
                // publishes as instanceCustomIndex, and (from phase 1) the one
                // indirect draw. Its Mesh geometry is a zero-area placeholder
                // (see ParticleField.cpp), so the machinery below builds it a
                // harmless one-triangle BLAS and one desc slot apiece, and the
                // field draws no pixels until a representation is switched on.
                ParticleField* particleField = dynamic_cast<ParticleField*>(m);
                const bool isParticleField = (particleField != nullptr);
                const bool isMorphed   = isMorphedMesh(*m);
                // Tet-skinned PhysX soft body — detected via the material flag set by
                // SoftBody::enableGpuSkinning() (which also carries the per-frame tet
                // texture and the static tetIndex/tetWeight/tetRestInv* attributes).
                // Mutually exclusive with the other deformers.
                const bool isTet = !isSkinned && !isDisplaced && !isGrass && !isMorphed &&
                                   m->material() && m->material()->tetSkinning &&
                                   m->material()->tetTexture != nullptr;
                // Zero-copy vertex interop (enableVertexInterop). One blasCache
                // probe per MESH, not per instance — the same budget the three
                // dynamic_casts above spend, and for the same payoff: the two
                // CPU-bounds culls read a bool instead of hashing per entry.
                bool isVertexInterop = false;
                if (!isSkinned && !isDisplaced && !isGrass && !isMorphed && !isTet &&
                    !isParticleField) {
                    if (auto ig = m->geometry()) {
                        auto ic = blasCache.find(ig.get());
                        isVertexInterop = (ic != blasCache.end() && ic->second->interop);
                    }
                }
                // Auto-LOD selection caches — once per Mesh, shared by every
                // instance entry (see the MeshEntry field doc). manualLodLevelRoots_
                // is complete for this mesh's ancestors: traverseVisible is
                // pre-order, so an enclosing LOD node registered its level
                // objects before we got here.
                bool lodExempt = !m->autoLod;// per-object opt-out (self-managed LOD, e.g. terrain tiles)
                for (Object3D* p = m->parent; !lodExempt && p; p = p->parent) {
                    if (manualLodLevelRoots_.count(p)) lodExempt = true;
                }
                const MaterialWithEmissive* lodEmissive = nullptr;
                if (auto mat = m->material()) {
                    lodEmissive = dynamic_cast<MaterialWithEmissive*>(mat.get());
                }
                if (!geom->boundingBox) geom->computeBoundingBox();
                Vector3 lodCenter;
                float lodRadius = 0.f;
                if (geom->boundingBox) {
                    lodCenter = geom->boundingBox->getCenter();
                    lodRadius = geom->boundingBox->getSize().length() * 0.5f;
                }
                // First-person viewmodel probe: is this mesh parented under a
                // Camera? Such geometry sits at the eye and must not occlude
                // (see MeshEntry::camAttached / kRayMaskNoShadow). One ancestor
                // walk per Mesh per full expansion.
                bool camAttached = false;
                for (auto* p = m->parent; p; p = p->parent) {
                    if (dynamic_cast<Camera*>(p)) {
                        camAttached = true;
                        break;
                    }
                }
                auto setLodCaches = [&](MeshEntry& e) {
                    e.lodExemptStatic = lodExempt;
                    e.lodEmissive       = lodEmissive;
                    e.lodGeomKey        = geom.get();
                    e.lodCenter[0]      = lodCenter.x;
                    e.lodCenter[1]      = lodCenter.y;
                    e.lodCenter[2]      = lodCenter.z;
                    e.lodRadius         = lodRadius;
                };
                // Per-mesh span over the entries pushed below. Caches the
                // matrix-change-detection inputs and the object-space bounds
                // so the per-frame loops can gate whole spans (see EntrySpan).
                EntrySpan sp{};
                sp.mesh  = m;
                sp.first = static_cast<uint32_t>(built.size());
                // Expansion frames hand every span fresh matrices; leaving the
                // flag set keeps the motion-matrix build computing real deltas
                // (vs the remapped prev worlds) on rebuild frames — a spawn
                // burst must not blank the movers' motion vectors for a frame.
                sp.movedThisFrame = true;
                sp.meshWorld = {};
                std::memcpy(sp.meshWorld.data(), m->matrixWorld->elements.data(), 64);
                sp.localRadius = lodRadius;
                if (geom->boundingBox) {
                    const Vector3 half = geom->boundingBox->getSize() * 0.5f;
                    sp.localCenter[0] = lodCenter.x;
                    sp.localCenter[1] = lodCenter.y;
                    sp.localCenter[2] = lodCenter.z;
                    sp.localHalf[0]   = half.x;
                    sp.localHalf[1]   = half.y;
                    sp.localHalf[2]   = half.z;
                    sp.localBoundsValid = true;
                }
                if (inst && inst->count() > 0) {
                    sp.inst           = inst;
                    sp.instMatVersion = inst->instanceMatrix()->version;
                    Matrix4 instMat;
                    Matrix4 world;
                    for (size_t j = 0; j < inst->count(); ++j) {
                        inst->getMatrixAt(j, instMat);
                        world.multiplyMatrices(*m->matrixWorld, instMat);
                        MeshEntry e{};
                        e.mesh = m;
                        e.instanceIndex = static_cast<uint32_t>(j);
                        e.isOverlay    = isOverlay;
                        e.isParticle   = isParticle;
                        e.isSkinned    = isSkinned;
                        e.isDisplaced  = isDisplaced;
                        e.isGrass      = isGrass;
                        e.isMorphed    = isMorphed;
                        e.isTet        = isTet;
                        e.isVertexInterop = isVertexInterop;
                        e.isInstanced  = true;
                        e.camAttached  = camAttached;
                        e.sensorOnly   = sensorOnly;
                        setLodCaches(e);
                        std::memcpy(e.worldMatrix.data(), world.elements.data(), 64);
                        built.push_back(e);
                    }
                } else {
                    MeshEntry e{};
                    e.mesh = m;
                    e.instanceIndex = 0u;
                    e.isOverlay    = isOverlay;
                    e.isParticle   = isParticle;
                    e.isSkinned    = isSkinned;
                    e.isDisplaced  = isDisplaced;
                    e.isGrass      = isGrass;
                    e.isMorphed    = isMorphed;
                    e.isTet        = isTet;
                    e.isVertexInterop = isVertexInterop;
                    e.isParticleField = isParticleField;
                    e.camAttached  = camAttached;
                    e.sensorOnly   = sensorOnly;
                    setLodCaches(e);
                    std::memcpy(e.worldMatrix.data(), m->matrixWorld->elements.data(), 64);
                    built.push_back(e);
                }
                sp.count = static_cast<uint32_t>(built.size()) - sp.first;
                if (isParticleField) {
                    // The placeholder geometry's bounds describe a point at the
                    // field's origin and say nothing about where the particles
                    // are, so publish "no bounds" rather than a wrong box. The
                    // frustum cull default-includes the span on the same flag.
                    sp.localBoundsValid = false;
                    sp.localRadius      = 0.f;
                    sp.aabbValid        = false;
                    builtFields.emplace_back(particleField, sp.first);
                }
                builtSpans.push_back(sp);
            });
            lastVisibleEntries_ = std::move(built);
            lastVisibleLines_   = std::move(builtLines);
            entrySpans_         = std::move(builtSpans);
            particleFields_     = std::move(builtFields);
            probeGridDirty_     = true;// scene structure changed → re-fit the probe grid
            // ...and re-bake every ParticleField surface height map (F5). The
            // set of things a flake can land on is exactly what just changed,
            // and this is the trigger the plan names ("the same trigger as
            // entry rebuild"). Cheap: it bumps a counter, and only the fields
            // that actually use a map pay for a re-trace.
            if (particleFieldPass_) particleFieldPass_->invalidateSurfaceBakes();
            }// !snapLean

            // ── Automatic mesh LOD selection (setAutoLod; ON by default) ────
            // Runs UNCONDITIONALLY every frame — camera motion alone can flip
            // a level even when nothing else about the scene changed — right
            // after entries got fresh world matrices from WHICHEVER path
            // produced them above (lean in-place diff or full re-expansion),
            // so this one spot covers both. Writes only MeshEntry::lodLevel +
            // lodChangedThisFrame_/autoLodStats_; the BLAS-address/index-
            // buffer swap itself happens later in this function (the lean
            // TLAS-refit instance loop) and in the full-rebuild instance loop
            // further down — both read en.lodLevel verbatim, never re-deriving.
            lodChangedThisFrame_ = false;
            if (autoLod_) drainLodResults();// budget: 16 geoms / 8 MiB of new levels per frame
            {
                THREEPP_CPUPROF("scene.4_lodSelect");
                VulkanRenderer::AutoLodStats stats{};
                stats.indexBytes   = lodIndexBytes_;
                stats.blasBytes    = lodBlasBytes_;
                stats.chainsReady  = lodChainsReadyCount_;
                stats.chainsQueued = lodChainsQueuedCount_;
                if (autoLod_) {
                    // Hot path by design: everything type- or structure-derived
                    // was cached on the entry at expansion (see MeshEntry doc) —
                    // this loop is float math + one blasCache hash per entry.
                    // The uncached version cost 2-4 ms/frame at 4k entries.
                    auto* persp = dynamic_cast<PerspectiveCamera*>(&camera);
                    Vector3 camPos;
                    if (persp) camPos.setFromMatrixPosition(*camera.matrixWorld);
                    // renderExtent() is the INTERNAL (render-scale-applied)
                    // resolution the G-buffer rasterizes at — τ below is in
                    // G-buffer pixels, i.e. the resolution the geometric error
                    // is actually sampled at. (Post-TAA upscale can stretch a
                    // render pixel to ~1.3 display pixels at scale 0.75; the
                    // fjord/harness visual gates validated the threshold at
                    // that operating point.)
                    const float renderHeightPx = static_cast<float>(renderExtent().height);
                    const float tanHalfFovY = persp
                            ? std::tan(math::degToRad(persp->getEffectiveFOV()) * 0.5f)
                            : 0.f;

                    // Per SPAN: exemption class, emissive values, the blasCache
                    // slot and chain eligibility are per-MESH facts — resolve
                    // them once per span. Only spans with a READY chain pay the
                    // per-entry projected-error math; everything else is a
                    // constant-time skip (plus a snap-back walk gated on
                    // lodNonZero so it runs only right after a chain retires).
                    for (auto& sp : entrySpans_) {
                        MeshEntry& e0 = entries[sp.first];
                        if (e0.isOverlay || e0.isParticle) continue;// not scene geometry — no LOD concept
                        const auto resetSpan = [&]() {
                            if (sp.lodNonZero) {
                                for (uint32_t j = 0; j < sp.count; ++j) {
                                    auto& en = entries[sp.first + j];
                                    if (en.lodLevel != 0) lodChangedThisFrame_ = true;
                                    en.lodLevel = 0;
                                }
                                sp.lodNonZero = false;
                            }
                            stats.entriesPerLevel[0] += sp.count;
                        };
                        // Deformers (stale local AABB), opted-out meshes
                        // (Object3D::autoLod == false — self-managed LOD like
                        // terrain tiles), authored manual-LOD subtrees, and
                        // non-perspective cameras (SSE needs a pinhole model):
                        // full detail.
                        if (e0.isSkinned || e0.isDisplaced || e0.isGrass || e0.isMorphed || e0.isTet ||
                            e0.lodExemptStatic || !persp) {
                            resetSpan();
                            continue;
                        }
                        // Emissive VALUES read live off the expansion-cached
                        // cast — NEE's per-tri CDF (buildAndUploadEmissiveTris)
                        // caches world-space triangles and would not notice a
                        // silent index-buffer swap underneath it. Live values
                        // (not a cached verdict) so a lantern whose intensity
                        // ramps up at dusk exempts the moment it turns on.
                        if (e0.lodEmissive &&
                            (e0.lodEmissive->emissive.r > 0.f || e0.lodEmissive->emissive.g > 0.f ||
                             e0.lodEmissive->emissive.b > 0.f || e0.lodEmissive->emissiveMap)) {
                            resetSpan();
                            continue;
                        }

                        auto blasIt = blasCache.find(e0.lodGeomKey);
                        if (blasIt == blasCache.end()) {
                            resetSpan();// brand-new geometry this frame — no LOD0 record to chain off yet
                            continue;
                        }
                        BlasRecord& rec = *blasIt->second;

                        // BufferGeometry::drawRange is honoured at the BASE level
                        // only (buildIndirectDrawData): a simplified level's index
                        // set has no correspondence with the range's units. A mesh
                        // drawing a partial range must therefore stay at LOD0 —
                        // selecting a coarser level would silently draw the WHOLE
                        // simplified mesh, the range ignored with no warning.
                        // lastDraw* is the always-runs enqueue loop's snapshot of
                        // drawRange, so this hot loop stays shared_ptr-deref-free.
                        {
                            const auto fullElems = static_cast<int64_t>(
                                    rec.indexCount != 0u ? rec.indexCount : rec.vertexCount);
                            if (rec.lastDrawStart != 0 ||
                                static_cast<int64_t>(rec.lastDrawCount) < fullElems) {
                                resetSpan();
                                continue;
                            }
                        }

                        // NON-indexed soup (FBX-style loaders never call
                        // setIndex) is eligible too: the chain generator welds
                        // it into canonical indices, and the levels drive
                        // INDEXED draws against the unchanged soup vertex
                        // buffer (selectLodGeom reports the indexed-ness).
                        const uint32_t triCount =
                                (rec.indexCount != 0u ? rec.indexCount : rec.vertexCount) / 3u;
                        // A chain enqueued while the geometry is being edited
                        // is guaranteed stale on arrival: drainLodResults drops
                        // it on the geomVersion mismatch, the dirty pass resets
                        // lodState, selection re-enqueues — a full attribute
                        // snapshot plus a wasted background simplification
                        // EVERY frame, forever (the Flock churn). Wait out a
                        // quiet window after the last edit, and never consider
                        // a graduated per-frame deformer at all.
                        const bool editQuiet =
                                !rec.perFrameDynamic &&
                                (frameSerial_ - rec.lastDirtyFrame) > BlasRecord::kLodDirtyQuietFrames;
                        const bool eligible = triCount >= 1024u && editQuiet;
                        if (eligible && rec.lodState == BlasRecord::LodState::None) {
                            if ((lodIndexBytes_ + lodBlasBytes_) <= kLodByteBudget) {
                                // The enqueue snapshots attribute data, so it needs
                                // the live geometry — the ONLY shared_ptr deref
                                // left in this loop, paid once per geometry
                                // lifetime. Failed enqueue (no position attribute)
                                // marks Failed, not Queued: no result is coming,
                                // so Queued would strand the record forever.
                                auto geomSp = e0.mesh->geometry();
                                rec.lodState = (geomSp && enqueueLodJob(e0.lodGeomKey, rec.geomVersion, *geomSp,
                                                                        lodNormalWeightFor(*e0.mesh)))
                                        ? BlasRecord::LodState::Queued
                                        : BlasRecord::LodState::Failed;
                            } else if (!lodBudgetWarned_) {
                                std::cerr << "[VulkanRenderer] auto-LOD: 256 MiB byte budget reached - "
                                             "no further chains will be generated this session\n";
                                lodBudgetWarned_ = true;
                            }
                        }

                        if (rec.lodState != BlasRecord::LodState::Ready || rec.lodLevels.empty()) {
                            resetSpan();
                            continue;
                        }

                        bool spanNonZero = false;
                        for (uint32_t j = 0; j < sp.count; ++j) {
                        MeshEntry& en = entries[sp.first + j];
                        // Re-derive the cached object-space sphere after an
                        // in-place geometry edit (flagged by the geom-dirty
                        // detection; boundingBox was invalidated there too).
                        if (en.lodSphereDirty) {
                            en.lodSphereDirty = false;
                            if (auto g = en.mesh->geometry()) {
                                if (!g->boundingBox) g->computeBoundingBox();
                                if (g->boundingBox) {
                                    const Vector3 c = g->boundingBox->getCenter();
                                    en.lodCenter[0] = c.x;
                                    en.lodCenter[1] = c.y;
                                    en.lodCenter[2] = c.z;
                                    en.lodRadius = g->boundingBox->getSize().length() * 0.5f;
                                }
                            }
                        }
                        if (en.lodRadius <= 0.f) {
                            en.lodLevel = 0;// unknown bounds — stay at full detail
                            ++stats.entriesPerLevel[0];
                            continue;
                        }

                        // World bounding sphere straight off the entry's
                        // column-major world matrix — no Matrix4 round-trip.
                        const float* M = en.worldMatrix.data();
                        const float cx = M[0] * en.lodCenter[0] + M[4] * en.lodCenter[1] + M[8] * en.lodCenter[2] + M[12];
                        const float cy = M[1] * en.lodCenter[0] + M[5] * en.lodCenter[1] + M[9] * en.lodCenter[2] + M[13];
                        const float cz = M[2] * en.lodCenter[0] + M[6] * en.lodCenter[1] + M[10] * en.lodCenter[2] + M[14];
                        const float s0 = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
                        const float s1 = M[4] * M[4] + M[5] * M[5] + M[6] * M[6];
                        const float s2 = M[8] * M[8] + M[9] * M[9] + M[10] * M[10];
                        const float worldScale = std::sqrt(std::max(s0, std::max(s1, s2)));
                        const float worldRadius = en.lodRadius * worldScale;
                        const float dx = camPos.x - cx, dy = camPos.y - cy, dz = camPos.z - cz;
                        const float distToCenter = std::sqrt(dx * dx + dy * dy + dz * dz);

                        uint8_t selected = 0;// camera-inside-sphere (or degenerate FOV) default: full detail
                        if (distToCenter > worldRadius && tanHalfFovY > 1e-6f) {
                            const float dist = distToCenter - worldRadius;
                            const float pxPerUnit = renderHeightPx / (2.f * tanHalfFovY * dist);
                            auto projErrorOf = [&](uint8_t lvl) -> float {
                                if (lvl == 0) return 0.f;
                                return rec.lodLevels[lvl - 1].errorBound * worldScale * pxPerUnit;
                            };
                            // Coarsest level whose projected error still fits under the
                            // error budget τ (setAutoLodError; default 0.75 px).
                            const float tau = lodErrorPx_;
                            for (uint8_t i = static_cast<uint8_t>(rec.lodLevels.size()); i >= 1; --i) {
                                if (projErrorOf(i) <= tau) { selected = i; break; }
                            }
                            // Hysteresis around the entry's CURRENT level (carried across
                            // lean frames via lastVisibleEntries_) — prevents a level right
                            // at the threshold from flip-flopping every frame. The raise
                            // margin scales with τ (0.8·τ keeps the validated 0.6/0.75
                            // ratio at the default).
                            const uint8_t cur = (en.lodLevel <= rec.lodLevels.size()) ? en.lodLevel : uint8_t{0};
                            uint8_t next = cur;
                            if (selected > cur && projErrorOf(selected) <= 0.8f * tau) next = selected;
                            else if (selected < cur && projErrorOf(cur) > tau) next = selected;
                            selected = next;
                        }

                        if (selected != en.lodLevel) lodChangedThisFrame_ = true;
                        en.lodLevel = selected;
                        if (selected != 0) spanNonZero = true;
                        ++stats.entriesPerLevel[std::min<uint8_t>(selected, 5)];
                        }// per-entry (Ready-chain spans only)
                        sp.lodNonZero = spanNonZero;
                    }
                } else {
                    // Feature off (or just turned off this frame): every
                    // entry snaps straight back to LOD0. Cheap even on huge
                    // scenes (one field compare/write per entry) and matters
                    // for correctness — without it, an entry's level from a
                    // prior setAutoLod(true) session would persist forever
                    // (MeshEntry::lodLevel survives across lean frames).
                    uint32_t nonOverlay = 0;
                    for (auto& sp : entrySpans_) {
                        const auto& e0 = entries[sp.first];
                        if (e0.isOverlay || e0.isParticle) continue;
                        if (sp.lodNonZero) {
                            for (uint32_t j = 0; j < sp.count; ++j) {
                                auto& en = entries[sp.first + j];
                                if (en.lodLevel != 0) lodChangedThisFrame_ = true;
                                en.lodLevel = 0;
                            }
                            sp.lodNonZero = false;
                        }
                        nonOverlay += sp.count;
                    }
                    stats.entriesPerLevel[0] = nonOverlay;
                }
                autoLodStats_ = stats;
            }

            // Change flags shared by the LEAN in-place diff and the generic
            // compare loop below — exactly one of the two fills them.
            bool structuralSame = true;
            bool matricesSame = true;
            bool materialValuesSame = true;
            bool bonesDirtyAny = false;
            bool displacedDirtyAny = false;
            bool grassDirtyAny = false;
            bool tetDirtyAny = false;
            bool geomDirtyAny = false;
            bool morphDirtyAny = false;
            std::vector<bool> entryGeomDirty(entries.size(), false);
            // Per-entry "material values bumped" bits — the ONLY entries whose
            // MaterialDesc must be re-derived in the material-values fast path
            // (materialFromMesh is a ~25-dynamic_cast + shared_ptr-churn walk).
            // Filled by both the LEAN in-place diff and the generic compare so
            // a single material toggle costs O(changed) instead of O(scene).
            std::vector<bool> entryMatDirty(entries.size(), false);

            // LEAN IN-PLACE DIFF: prevSceneFingerprint IS this frame's
            // fingerprint state — diff against the live objects and update it
            // directly instead of copying it into currFp (the copy refcounted
            // 8 shared_ptr<Texture> per entry: ~24k atomics/frame on Bistro's
            // textured materials, measured as the largest remaining cost).
            // Per entry: matrix memcmp, a material-version read through the
            // cached typed pointer, and the composite geometry version through
            // the attribute-pointer cache (attributesVersion() gates its
            // validity — zero string lookups). A bumped material version
            // re-derives that entry's textures FIRST and only commits its
            // updates when they are unchanged; a texture swap aborts to the
            // full fingerprint + compare path (which classifies it STRUCTURAL)
            // with every already-committed entry still holding verified-
            // correct values.
            bool leanOk = false;
            if (snapLean) {
                THREEPP_CPUPROF("scene.5_leanDiff");
                leanOk = true;
                // Per SPAN: material version, texture set, and geometry
                // versions are per-MESH facts — read them once per span
                // instead of once per instance (the per-entry loop paid a
                // material-version read + four attribute-version reads per
                // instance per frame). Matrix change comes straight from the
                // span refresh above (fp.matrix was updated there in the same
                // pass). Per-entry writes remain only for spans that changed.
                for (auto& sp : entrySpans_) {
                    if (!leanOk) break;
                    MeshFingerprint& fp0 = prevSceneFingerprint[sp.first];
                    const bool xfmChanged = sp.movedThisFrame;
                    bool matChanged = false;
                    const unsigned int matVer = fp0.matTyped ? fp0.matTyped->version() : 0u;
                    if (matVer != fp0.matVersion) {
                        Mesh* m = sp.mesh;
                        if (albedoTexOf(*m) != fp0.albedoTex ||
                            roughnessTexOf(*m) != fp0.roughnessTex ||
                            metalnessTexOf(*m) != fp0.metalnessTex ||
                            normalTexOf(*m) != fp0.normalTex ||
                            transmissionTexOf(*m) != fp0.transmissionTex ||
                            clearcoatTexOf(*m) != fp0.clearcoatTex ||
                            clearcoatRoughnessTexOf(*m) != fp0.clearcoatRoughnessTex ||
                            emissiveTexOf(*m) != fp0.emissiveTex ||
                            occlusionTexOf(*m) != fp0.occlusionTex) {
                            leanOk = false;// texture swap = STRUCTURAL — full path decides
                            break;
                        }
                        matChanged = true;
                        const MaterialDesc md = materialFromMesh(*m);
                        const std::array<float, 15> pbr = {
                                md.albedo[0], md.albedo[1], md.albedo[2],
                                md.roughness, md.metalness,
                                md.emissive[0], md.emissive[1], md.emissive[2],
                                md.emissiveIntensity,
                                md.normalScale[0], md.normalScale[1],
                                md.transmission, md.ior,
                                md.clearcoat, md.clearcoatRoughness};
                        for (uint32_t j = 0; j < sp.count; ++j) {
                            auto& fp = prevSceneFingerprint[sp.first + j];
                            fp.matVersion = matVer;
                            fp.pbr = pbr;
                        }
                    }
                    BufferGeometry* g = fp0.geomTyped;
                    const unsigned int av = g->attributesVersion();
                    if (av != fp0.attrVersion) {// attribute added/replaced/removed — re-cache
                        // Untyped: only `version` is polled off these, and typed
                        // lookups would come back null for narrowed attributes.
                        auto* posAttr  = g->getAttribute("position");
                        auto* normAttr = g->getAttribute("normal");
                        auto* uvAttr   = g->getAttribute("uv");
                        auto* idxAttr  = g->getIndex();
                        auto* colAttr  = g->getAttribute("color");
                        for (uint32_t j = 0; j < sp.count; ++j) {
                            auto& fp = prevSceneFingerprint[sp.first + j];
                            fp.attrVersion = av;
                            fp.posAttr  = posAttr;
                            fp.normAttr = normAttr;
                            fp.uvAttr   = uvAttr;
                            fp.idxAttr  = idxAttr;
                            fp.colAttr  = colAttr;
                        }
                    }
                    unsigned int gv = 0;// must mirror geomVersionOf()
                    if (fp0.posAttr)  gv += fp0.posAttr->version;
                    if (fp0.normAttr) gv += fp0.normAttr->version;
                    if (fp0.idxAttr)  gv += fp0.idxAttr->version;
                    if (fp0.uvAttr)   gv += fp0.uvAttr->version;
                    if (fp0.colAttr)  gv += fp0.colAttr->version;
                    const bool geomChanged = (gv != fp0.geomVersion);
                    if (geomChanged) {
                        for (uint32_t j = 0; j < sp.count; ++j)
                            prevSceneFingerprint[sp.first + j].geomVersion = gv;
                        // Particle billboard meshes mutate their attributes every
                        // frame but own no BLAS — flagging geomDirty would fire a
                        // per-frame vkDeviceWaitIdle for a refit that skips them
                        // anyway (blasCache miss). The billboard pass re-uploads
                        // their vertex cache itself, version-gated.
                        if (!entries[sp.first].isParticle) {
                            geomDirtyAny = true;
                            for (uint32_t j = 0; j < sp.count; ++j) {
                                entryGeomDirty[sp.first + j] = true;
                                entries[sp.first + j].lodSphereDirty = true;// auto-LOD's cached sphere follows
                            }
                            // boundingBox invalidation — mirrors the generic loop.
                            if (auto gg = sp.mesh->geometry()) gg->boundingBox.reset();
                            sp.localBoundsValid = false;
                            sp.aabbValid = false;
                        }
                    }
                    if (xfmChanged) matricesSame = false;// fp.matrix refreshed with the span above
                    if (matChanged) {
                        materialValuesSame = false;
                        for (uint32_t j = 0; j < sp.count; ++j)
                            entryMatDirty[sp.first + j] = true;
                    }
                    if (xfmChanged || matChanged || geomChanged) {
                        const size_t lastW = (size_t(sp.first) + sp.count - 1) >> 5;
                        if (lastW >= meshMovedBits_.size()) meshMovedBits_.resize(lastW + 1, 0u);
                        for (uint32_t i = sp.first; i < sp.first + sp.count; ++i)
                            meshMovedBits_[i >> 5] |= (1u << (i & 31u));
                    }
                }
                if (!leanOk) {
                    // Partial in-place updates are all verified-correct values;
                    // reset the flags and let the full fingerprint + compare
                    // re-derive and classify (texture swap ⇒ structural rebuild).
                    structuralSame = true;
                    matricesSame = true;
                    materialValuesSame = true;
                    geomDirtyAny = false;
                    entryGeomDirty.assign(entries.size(), false);
                }
            }

            // Per-entry fingerprint construction (full path only — the lean
            // path diffed prev's fingerprints in place above). Hot path on big
            // static scenes (Bistro): when mesh + mat + geom pointers all match
            // prev frame AND mat->version() also matches, the 8 texture-of
            // lookups and materialFromMesh (~21 dynamic_casts/mesh) all return
            // identical results — copy them straight from
            // prevSceneFingerprint[i] and only refresh the world matrix.
            if (!leanOk) {
            THREEPP_CPUPROF("scene.5b_fingerprint");
            currFp.resize(entries.size());
            const bool prevValid = sceneBuilt_ && prevSceneFingerprint.size() == entries.size();
            // Per-Material memo for the slow (non-fast-path) branch: the nine
            // texture-of lookups + materialFromMesh are a pure function of the
            // mesh's Material* and dominate a structural rebuild when instanced
            // geometry shares one material across hundreds of entries. Cache the
            // derived texture pointers + PBR floats and reuse them per material.
            struct FpMatDerived {
                std::shared_ptr<Texture> albedoTex, roughnessTex, metalnessTex, normalTex,
                        transmissionTex, clearcoatTex, clearcoatRoughnessTex, emissiveTex, occlusionTex;
                std::array<float, 15> pbr;
            };
            std::unordered_map<const Material*, FpMatDerived> fpMemo;
            for (size_t i = 0; i < entries.size(); ++i) {
                const MeshEntry& en = entries[i];
                Mesh* m = en.mesh;
                auto matSp = m->material();
                const Material* matPtr = matSp.get();
                const unsigned int matVer = matPtr ? matPtr->version() : 0u;
                const void* geomPtr = m->geometry().get();
                const unsigned int geomVer = geomVersionOf(*m->geometry());

                MeshFingerprint& fp = currFp[i];
                bool fastPath = false;
                if (prevValid) {
                    const auto& p = prevSceneFingerprint[i];
                    if (p.mesh == m && p.mat == matPtr && p.geom == geomPtr &&
                        p.instanceIndex == en.instanceIndex &&
                        p.matVersion == matVer && p.geomVersion == geomVer) {
                        // Texture pointers + pbr live on the material; matVersion
                        // unchanged means none of them moved. Copy everything from
                        // prev, then overwrite matrix (transform can change without
                        // bumping mat version — Object3D xfm is independent) and
                        // the overlay flag (entries were just re-expanded; a flag
                        // flip is exactly what the structural compare must see).
                        fp = p;
                        fp.matrix = en.worldMatrix;
                        fp.overlay = en.isOverlay;
                        fastPath = true;
                    }
                }

                if (!fastPath) {
                    fp.mesh = m;
                    fp.geom = geomPtr;
                    fp.mat  = matPtr;
                    fp.matVersion = matVer;
                    fp.geomVersion = geomVer;
                    fp.matTyped  = matPtr;
                    fp.geomTyped = m->geometry().get();
                    fp.attrVersion = fp.geomTyped->attributesVersion();
                    fp.posAttr  = fp.geomTyped->getAttribute("position");
                    fp.normAttr = fp.geomTyped->getAttribute("normal");
                    fp.uvAttr   = fp.geomTyped->getAttribute("uv");
                    fp.idxAttr  = fp.geomTyped->getIndex();
                    const Material* matKey = matPtr;
                    auto memoIt = fpMemo.find(matKey);
                    if (memoIt == fpMemo.end()) {
                        FpMatDerived d{};
                        d.albedoTex             = albedoTexOf(*m);
                        d.roughnessTex          = roughnessTexOf(*m);
                        d.metalnessTex          = metalnessTexOf(*m);
                        d.normalTex             = normalTexOf(*m);
                        d.transmissionTex       = transmissionTexOf(*m);
                        d.clearcoatTex          = clearcoatTexOf(*m);
                        d.clearcoatRoughnessTex = clearcoatRoughnessTexOf(*m);
                        d.emissiveTex           = emissiveTexOf(*m);
                        d.occlusionTex          = occlusionTexOf(*m);
                        const MaterialDesc md = materialFromMesh(*m);
                        d.pbr = {md.albedo[0], md.albedo[1], md.albedo[2],
                                 md.roughness, md.metalness,
                                 md.emissive[0], md.emissive[1], md.emissive[2],
                                 md.emissiveIntensity,
                                 md.normalScale[0], md.normalScale[1],
                                 md.transmission, md.ior,
                                 md.clearcoat, md.clearcoatRoughness};
                        memoIt = fpMemo.emplace(matKey, std::move(d)).first;
                    }
                    const FpMatDerived& d = memoIt->second;
                    fp.albedoTex             = d.albedoTex;
                    fp.roughnessTex          = d.roughnessTex;
                    fp.metalnessTex          = d.metalnessTex;
                    fp.normalTex             = d.normalTex;
                    fp.transmissionTex       = d.transmissionTex;
                    fp.clearcoatTex          = d.clearcoatTex;
                    fp.clearcoatRoughnessTex = d.clearcoatRoughnessTex;
                    fp.emissiveTex           = d.emissiveTex;
                    fp.occlusionTex          = d.occlusionTex;
                    fp.instanceIndex         = en.instanceIndex;
                    fp.overlay               = en.isOverlay;
                    fp.matrix                = en.worldMatrix;
                    fp.pbr = d.pbr;
                }
            }
            }// !leanOk (fingerprint re-derivation)

            // Per-entry bone-dirty bits. SkinnedMesh poses change without
            // touching the SkinnedMesh's worldMatrix, so the matrix-fingerprint
            // misses them. We compare current Skeleton::boneMatrices against
            // the cached prevBoneMats for each known SkinnedMesh; first-time
            // skinned meshes mark dirty so the structural-rebuild path skins
            // before its first ray trace. Gated on the cached isSkinned flag
            // so non-skinned entries skip the type probe entirely.
            std::vector<bool> entryBonesDirty(entries.size(), false);
            {
            THREEPP_CPUPROF("scene.6_deformerScans");
            // Deformer flags are per-MESH — gate on the span's first entry so
            // a non-deformer span (the 100k-instance common case) costs one
            // bool read, not count() of them.
            for (const auto& sp : entrySpans_) {
                if (!entries[sp.first].isSkinned) continue;
                for (size_t i = sp.first; i < size_t(sp.first) + sp.count; ++i) {
                auto* sm = static_cast<SkinnedMesh*>(entries[i].mesh);
                if (!sm->skeleton || sm->skeleton->bones.empty()) continue;
                auto stIt = skinnedMeshStates.find(sm);
                if (stIt == skinnedMeshStates.end()) {
                    entryBonesDirty[i] = true;
                    continue;
                }
                sm->skeleton->update();
                const auto& bm = sm->skeleton->boneMatrices;
                if (bm.size() != stIt->second->prevBoneMats.size() ||
                    std::memcmp(bm.data(), stIt->second->prevBoneMats.data(),
                                bm.size() * sizeof(float)) != 0) {
                    entryBonesDirty[i] = true;
                }
                }
            }
            }

            // DisplacedMesh — intrinsically dirty every frame (FFT spectrum
            // advances continuously). Same per-entry-bool pattern as bones,
            // routed through the cached isDisplaced flag instead of a fresh
            // dynamic_cast every frame.
            std::vector<bool> entryDisplacedDirty(entries.size(), false);
            {
                THREEPP_CPUPROF("scene.6_deformerScans");
                for (const auto& sp : entrySpans_) {
                    if (!entries[sp.first].isDisplaced) continue;
                    for (uint32_t j = 0; j < sp.count; ++j) entryDisplacedDirty[sp.first + j] = true;
                }
            }

            // GrassMesh — intrinsically dirty every frame (wind advances) UNLESS
            // the field is frozen by the distance gate (params.maxAnimDistance).
            // A frozen (far) field is NOT dirty ⇒ it is not queued for a wind
            // dispatch/BLAS refit, not folded into grassDirtyAny, and not marked
            // moved — it holds its last displaced pose. It STILL stays in the TLAS
            // (culling never touches the TLAS), so off-screen/ranged grass keeps
            // casting shadows and appearing in reflections/GI. Grass keeps no
            // prevVertex buffer, so a skipped frame reports zero per-vertex motion
            // exactly like an animated frame — no TAA ghosting from the freeze.
            std::vector<bool> entryGrassDirty(entries.size(), false);
            {
                THREEPP_CPUPROF("scene.6_deformerScans");
                // Camera world position (translation column of matrixWorld).
                const auto& cw = camera.matrixWorld->elements;
                const Vector3 camPos(cw[12], cw[13], cw[14]);
                for (const auto& sp : entrySpans_) {
                    if (!entries[sp.first].isGrass) continue;
                    for (uint32_t j = 0; j < sp.count; ++j) {
                        const size_t i = size_t(sp.first) + j;
                        entryGrassDirty[i] = grassShouldAnimate(entries[i], camPos);
                    }
                }
            }

            // Morphed meshes — dirty when morphTargetInfluences changed.
            // Skinned meshes that also carry morph targets are handled by the
            // bone path above (GPU-skinned BLAS rebuild) so we skip them
            // here. Both predicates come from the cached fingerprint flags;
            // the getMorphAttributes hash lookup + SkinnedMesh dynamic_cast
            // used to run for every entry every frame.
            std::vector<bool> entryMorphDirty(entries.size(), false);
            {
            THREEPP_CPUPROF("scene.6_deformerScans");
            for (const auto& sp : entrySpans_) {
                if (!entries[sp.first].isMorphed || entries[sp.first].isSkinned) continue;
                for (size_t i = sp.first; i < size_t(sp.first) + sp.count; ++i) {
                Mesh* m = entries[i].mesh;
                auto mIt = morphedMeshStates.find(m);
                if (mIt == morphedMeshStates.end()) {
                    entryMorphDirty[i] = true;
                    continue;
                }
                auto* morphObj = m->as<ObjectWithMorphTargetInfluences>();
                if (!morphObj) continue;
                const auto& inf = morphObj->morphTargetInfluences();
                const auto& prev = mIt->second->prevInfluences;
                if (inf.size() != prev.size() ||
                    std::memcmp(inf.data(), prev.data(), inf.size() * sizeof(float)) != 0) {
                    entryMorphDirty[i] = true;
                }
                }
            }
            }

            // LEAN path: the deformer dirty bits (bones / displaced / morph)
            // come from the scans above — fold them into the moved-bits mask
            // and the *DirtyAny flags exactly like the generic loop does.
            if (leanOk) {
                THREEPP_CPUPROF("scene.6_deformerScans");
                for (const auto& sp : entrySpans_) {
                    const auto& e0 = entries[sp.first];
                    if (!(e0.isSkinned || e0.isDisplaced || e0.isGrass || e0.isMorphed)) continue;
                    for (size_t i = sp.first; i < size_t(sp.first) + sp.count; ++i) {
                        const bool b = entryBonesDirty[i];
                        const bool d = entryDisplacedDirty[i];
                        const bool gr = entryGrassDirty[i];
                        const bool mo = entryMorphDirty[i];
                        if (!(b || d || gr || mo)) continue;
                        if (b) bonesDirtyAny = true;
                        if (d) displacedDirtyAny = true;
                        if (gr) grassDirtyAny = true;
                        if (mo) morphDirtyAny = true;
                        const size_t w = i >> 5;
                        if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                        meshMovedBits_[w] |= (1u << (i & 31u));
                    }
                }
            }
            // Continuous-motion fast path: when only the per-mesh matrices
            // changed (everything else — topology, materials, textures —
            // matches), refit the TLAS in place and let the deferred shade's
            // ray-query reproject.
            // We only have to detect the matrix-only case ahead of time;
            // motion matrices themselves are computed each frame in
            // renderFrame so we can defer the host write past the fence wait.
            // (entries IS lastVisibleEntries_ — cached in place, both paths.)
            if (sceneBuilt_ && (leanOk || currFp.size() == prevSceneFingerprint.size())) {
                // Four classes of change:
                //   structural    — pointers (mesh/geom/mat/textures): full rebuild.
                //   matrices      — per-mesh world matrices: TLAS refit + bit set.
                //   materialVals  — pbr floats (KHR_animation_pointer animates colors,
                //                   roughness, etc): re-upload matDescs in place + bit set.
                //   geomData      — same BufferGeometry pointer but attribute version
                //                   bumped (user mutated vertices in-place): re-upload
                //                   data + in-place BLAS rebuild + TLAS refit. Falls
                //                   through to structural if vertex/index count changed.
                // Splitting matters: KHR_animation_pointer changes pbr every frame
                // without changing any pointer or texture. Lumping it under
                // structural caused a full rebuild every frame, which reset
                // the gbuf+sampleIndex globally and froze accumulation
                // scene-wide.
                // The LEAN path already filled the change flags in its in-place
                // diff; only the full path runs the generic compare here.
                if (!leanOk) {
                THREEPP_CPUPROF("scene.5c_genericCompare");
                for (size_t i = 0; i < currFp.size(); ++i) {
                    const auto& a = currFp[i];
                    const auto& b = prevSceneFingerprint[i];
                    if (a.mesh != b.mesh || a.geom != b.geom || a.mat != b.mat ||
                        // TLAS membership flip (wireframe/overlay toggled) — the
                        // instance set changed, so a refit would MODE_UPDATE with
                        // a different instance count than the last build (spec
                        // violation → corrupt TLAS → traversal hang). Rebuild.
                        a.overlay != b.overlay ||
                        a.albedoTex != b.albedoTex || a.roughnessTex != b.roughnessTex ||
                        a.metalnessTex != b.metalnessTex || a.normalTex != b.normalTex ||
                        a.transmissionTex != b.transmissionTex ||
                        a.clearcoatTex != b.clearcoatTex ||
                        a.clearcoatRoughnessTex != b.clearcoatRoughnessTex ||
                        a.emissiveTex != b.emissiveTex ||
                        a.occlusionTex != b.occlusionTex) {
                        structuralSame = false;
                        break;
                    }
                    const bool xfmChanged   = std::memcmp(a.matrix.data(), b.matrix.data(), sizeof(a.matrix)) != 0;
                    // matVersion catches changes the pbr float-array doesn't —
                    // notably KHR_texture_transform animation (rotation/offset/
                    // scale of a texture). PropertyBinding bumps Material::version
                    // on every setMaterialProperty hit, so this fires whenever
                    // anything on the material has been touched even if the
                    // PBR floats themselves didn't move.
                    const bool matChanged   = std::memcmp(a.pbr.data(),    b.pbr.data(),    sizeof(a.pbr))    != 0
                                              || a.matVersion != b.matVersion;
                    const bool bonesChanged = entryBonesDirty[i];
                    const bool dispChanged  = entryDisplacedDirty[i];
                    const bool grassChanged = entryGrassDirty[i];
                    const bool geomChanged  = (a.geomVersion != b.geomVersion);
                    const bool morphChanged = entryMorphDirty[i];
                    if (xfmChanged) matricesSame = false;
                    if (matChanged) { materialValuesSame = false; entryMatDirty[i] = true; }
                    if (bonesChanged) bonesDirtyAny = true;
                    if (dispChanged)  displacedDirtyAny = true;
                    if (grassChanged) grassDirtyAny = true;
                    // Particle billboard meshes mutate attributes every frame but
                    // own no BLAS; never flag them geomDirty (would fire a per-
                    // frame vkDeviceWaitIdle for a refit that skips them). The
                    // billboard pass re-uploads their vertex cache itself.
                    if (geomChanged && !entries[i].isParticle) {
                        geomDirtyAny = true;
                        entryGeomDirty[i] = true;
                        entries[i].lodSphereDirty = true;// auto-LOD's cached sphere follows
                        // Invalidate the cached boundingBox so the next
                        // cullEntriesAgainstFrustum recomputes it from the
                        // current positions. Without this, plain meshes with
                        // in-place vertex updates (PhysX soft bodies) keep
                        // the rest-pose AABB and get culled out by the raster
                        // pass once the body settles outside that stale box —
                        // gbuffer ends up with sky at those pixels and the
                        // hybrid deferred shade renders the background through them.
                        if (auto g = entries[i].mesh->geometry()) {
                            g->boundingBox.reset();
                        }
                    }
                    if (morphChanged) morphDirtyAny = true;
                    // All flavors of change invalidate this pixel's history —
                    // share the same per-mesh bit. Reproject+halve FC for any
                    // of: matrix shift, pbr shift, pose deformation, ocean
                    // surface displacement, geometry data mutation, morph blend.
                    if (xfmChanged || matChanged || bonesChanged || dispChanged || geomChanged || morphChanged) {
                        const size_t w = i >> 5;
                        if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                        meshMovedBits_[w] |= (1u << (i & 31u));
                    }
                }
                // The lean diff invalidates span bounds itself; the generic
                // compare above only fills per-entry bits — mirror the geom-
                // dirty invalidation onto the spans (the expansion snapshotted
                // local bounds BEFORE the edit was detected).
                if (geomDirtyAny) {
                    for (auto& sp : entrySpans_) {
                        if (sp.first < entryGeomDirty.size() && entryGeomDirty[sp.first]) {
                            sp.localBoundsValid = false;
                            sp.aabbValid = false;
                        }
                    }
                }
                }
                if (structuralSame) {
                    if (bonesDirtyAny) {
                        // Re-skin every SkinnedMesh whose pose changed and
                        // rebuild its BLAS in place. The BLAS handle/address
                        // stays valid, but the BLAS's wrapping AABB grows when
                        // a pose pushes vertices outside the previous extents
                        // — the TLAS caches that AABB at build/refit time, so
                        // without a TLAS refit rays get culled before they
                        // reach the BLAS, clipping the silhouette.
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (!entryBonesDirty[i]) continue;
                            auto* sm = static_cast<SkinnedMesh*>(entries[i].mesh);
                            auto stIt = skinnedMeshStates.find(sm);
                            if (stIt == skinnedMeshStates.end()) continue;
                            refreshSkinnedBlas(*sm, *stIt->second);
                        }
                    }
                    if (displacedDirtyAny) {
                        // Queue each DisplacedMesh water update (FFT chain →
                        // water_displace.comp → BLAS rebuild in place); the
                        // chain is recorded into the frame command buffer in
                        // recordCommandBuffer — NO blocking one-shot submit
                        // (the old drain stalled the CPU on all in-flight GPU
                        // work, tens of ms on large scenes). The CPU height-
                        // field mirror is NOT taken here: this runs before the
                        // frame's fence wait, while the frames in flight may
                        // still be copying into the readback — beginDeferredFrame
                        // mirrors the queued meshes right after vkWaitForFences,
                        // from the readback slot that fence proves complete.
                        const float now = static_cast<float>(frameNowSec());
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (!entryDisplacedDirty[i]) continue;
                            auto* dm = static_cast<DisplacedMesh*>(entries[i].mesh);
                            auto stIt = displacedStates.find(dm);
                            if (stIt == displacedStates.end()) continue;
                            pendingDisplacedDeforms_.emplace_back(dm, stIt->second.get(), now);
                            ++dm->frameTick;
                        }
                    }
                    if (grassDirtyAny) {
                        // Queue each GrassMesh deform; the grass_wind dispatch +
                        // BLAS refit are recorded into the frame command buffer
                        // in recordCommandBuffer (no blocking submit), mirroring
                        // the skinned-mesh path. One mesh = one TLAS instance.
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (!entryGrassDirty[i]) continue;
                            auto* gm = static_cast<GrassMesh*>(entries[i].mesh);
                            auto stIt = grassStates.find(gm);
                            if (stIt == grassStates.end()) continue;
                            pendingGrassDeforms_.emplace_back(gm, stIt->second.get());
                            ++gm->frameTick;
                        }
                    }
                    if (morphDirtyAny) {
                        std::unordered_set<Mesh*> refreshed;
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (!entryMorphDirty[i]) continue;
                            Mesh* m = entries[i].mesh;
                            if (!refreshed.insert(m).second) continue;
                            auto mIt = morphedMeshStates.find(m);
                            if (mIt == morphedMeshStates.end()) continue;
                            refreshMorphedBlas(*m, *mIt->second);
                        }
                    }
                    // Tet-skinned soft bodies (PhysX) deform every frame — re-upload
                    // their collision-tet positions, queue the GPU skin + BLAS refit,
                    // and flag motion for reprojection/TAA like the other deformers.
                    // At most one refresh per mesh per frame: refreshTetBlas advances
                    // the tetPos write ring, whose depth only covers the in-flight
                    // frames if it moves once per submission.
                    std::unordered_set<Mesh*> tetRefreshed;
                    for (const auto& sp : entrySpans_) {
                        if (!entries[sp.first].isTet) continue;
                        for (size_t i = sp.first; i < size_t(sp.first) + sp.count; ++i) {
                            if (!tetRefreshed.insert(entries[i].mesh).second) continue;
                            auto tIt = tetMeshStates.find(entries[i].mesh);
                            if (tIt == tetMeshStates.end()) continue;
                            refreshTetBlas(*entries[i].mesh, *tIt->second);
                            tetDirtyAny = true;
                            const size_t w = i >> 5;
                            if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                            meshMovedBits_[w] |= (1u << (i & 31u));
                        }
                    }
                    // Geometries whose prevVertex was re-snapshotted (to OLD
                    // positions) this frame. The prevVertex re-sync pass below
                    // must SKIP these so their legitimate change-frame deformation
                    // motion survives — they settle on the next clean frame.
                    std::unordered_set<const BufferGeometry*> geomRefreshedThisFrame;

                    // ── Zero-copy vertex interop: the UNCONDITIONAL enqueue ──
                    // Everything else on this path is gated on the composite
                    // BufferAttribute version (geomVersionOf, entryGeomDirty). A
                    // foreign device producer bumps none of those — it writes
                    // GPU memory the host never sees — so an interop record would
                    // read "clean" forever, its BLAS would never refit, and the
                    // ray-traced view (shadows, reflections, GI, lidar) would
                    // quietly diverge from the rasterized one while the picture
                    // looked fine. Same reasoning, same answer as the tet path
                    // pushing pendingTetRebuilds_ every frame regardless.
                    bool interopDirtyAny = false;
                    if (!blasCache.empty()) {
                        std::unordered_set<const BufferGeometry*> interopEnqueued;
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (entries[i].isSkinned || entries[i].isDisplaced ||
                                entries[i].isGrass || entries[i].isTet) continue;
                            const BufferGeometry* geomKey = entries[i].mesh->geometry().get();
                            auto cIt = blasCache.find(geomKey);
                            // Refresh MeshEntry::isVertexInterop in place, every
                            // frame, for every plain entry — set and clear. This
                            // is what lets the flag be trusted without
                            // a structural rebuild behind it: arming interop
                            // changes no pointer, matrix, material or attribute
                            // version, so no fingerprint diff can see it, and the
                            // snapshot fast path reuses last frame's MeshEntry
                            // objects verbatim. Deriving it here — on the
                            // canonical entry list, before cullEntriesAgainstFrustum
                            // and buildIndirectDrawData run later this frame —
                            // makes it live in both paths and stale in neither.
                            // BufferGeometry::drawRange is not covered by any
                            // BufferAttribute version, so a set_draw_range-only
                            // frame would sail through the DrawInfo skip
                            // signature with commands built for the OLD range.
                            // Compare against the record's snapshot here — the
                            // same always-runs loop that refreshes
                            // isVertexInterop — and bump drawInputsVersion_ on change:
                            // new DrawInfo inputs must (EntrySpan contract).
                            // A non-interop record ALSO gets the geom-dirty
                            // flags: its BLAS holds the old span, and the
                            // refresh-op collection below runs the canonical
                            // fix (drained rebuild for an occasional change,
                            // graduation to the frame-cb refit for a mesh that
                            // animates its range every frame). Interop records
                            // need neither — their unconditional per-frame
                            // enqueue reads drawRange at record time.
                            if (cIt != blasCache.end()) {
                                auto& r0 = *cIt->second;
                                const auto& dr = geomKey->drawRange;
                                if (r0.lastDrawStart != dr.start ||
                                    r0.lastDrawCount != dr.count) {
                                    r0.lastDrawStart = dr.start;
                                    r0.lastDrawCount = dr.count;
                                    ++drawInputsVersion_;
                                    if (!r0.interop && r0.as != VK_NULL_HANDLE) {
                                        geomDirtyAny = true;
                                        entryGeomDirty[i] = true;
                                    }
                                }
                            }
                            const bool iop = (cIt != blasCache.end() && cIt->second->interop);
                            entries[i].isVertexInterop = iop;
                            if (!iop) continue;
                            auto& rec = *cIt->second;

                            // Interop requires FIXED-CAPACITY geometry: the
                            // fullRebuild path destroys rec.vertex on any count
                            // change, and with it the exported allocation a
                            // foreign API has already imported — an OS handle
                            // that keeps working while pointing at freed memory.
                            // Tear the interop down HERE instead, so the exports
                            // are released deliberately and the application is
                            // told, rather than losing them inside a rebuild that
                            // has no idea they existed.
                            auto* posAttr = entries[i].mesh->geometry()->getAttribute<float>("position");
                            auto* idxAttr = entries[i].mesh->geometry()->getIndex();
                            const uint32_t curVtx = posAttr ? static_cast<uint32_t>(posAttr->count()) : 0u;
                            const uint32_t curIdx = idxAttr ? static_cast<uint32_t>(idxAttr->count()) : 0u;
                            if (posAttr && (curVtx != rec.vertexCount || curIdx != rec.indexCount)) {
                                std::cerr << "[VulkanRenderer] vertex interop: geometry changed "
                                             "topology (" << rec.vertexCount << " -> " << curVtx
                                          << " verts) - interop needs fixed-capacity geometry, so "
                                             "it is being DISABLED for this mesh. Allocate for the "
                                             "maximum triangle count once and write degenerates "
                                             "for the unused tail.\n";
                                disableVertexInterop(*entries[i].mesh);
                                continue;
                            }
                            if (!interopEnqueued.insert(geomKey).second) continue;

                            pendingDynamicGeomRefits_.push_back({geomKey, &rec});
                            rec.lastDirtyFrame = frameSerial_;
                            rec.dynPrevResyncPending = true;
                            interopDirtyAny = true;
                            // The surface deforms every frame, so its pixels'
                            // temporal history is invalid every frame — same bit
                            // the tet loop above stamps for the same reason.
                            const size_t w = i >> 5;
                            if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                            meshMovedBits_[w] |= (1u << (i & 31u));
                        }
                    }

                    if (geomDirtyAny) {
                        // Re-upload vertex data for geometries whose
                        // BufferAttribute versions changed and rebuild their
                        // BLAS in-place. If any geometry changed its vertex or
                        // index count (topology change), we can't reuse the
                        // old buffers — fall through to the full structural
                        // rebuild instead.
                        //
                        // ensureSceneBuilt runs in render() BEFORE renderFrame
                        // waits on inFlight[currentFrame], so up to
                        // kFramesInFlight prior frames may still be reading
                        // rec.vertex/normal/index via closest_hit's
                        // GeometryDesc.vertexAddress fetches (and the hybrid
                        // raster gbuffer pass binds the same buffer as a
                        // vertex buffer). Memcpying into those buffers
                        // mid-flight is a device-lost on NVIDIA. Drain
                        // everything device-wide before mutating shared BLAS
                        // buffers — skinned / displaced / morphed paths above
                        // submit on the same queue, so this one wait covers
                        // them too. Graduated per-frame dynamic records
                        // (BlasRecord::perFrameDynamic) never take that drain:
                        // their upload + refit records into the frame cb via
                        // recordDynamicGeomRefits, behind the fence that
                        // guarantees their staging slot is idle. So the wait
                        // only fires when an OCCASIONAL edit needs the
                        // host-write path — build the op lists first, decide
                        // after.
                        bool topologyChanged = false;
                        std::unordered_set<const BufferGeometry*> refreshedGeoms;
                        std::vector<GeomRefreshOp> refreshOps;
                        std::vector<BlasRecord*> lodChainDoomed;
                        refreshOps.reserve(entries.size());
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (!entryGeomDirty[i]) continue;
                            if (entries[i].isSkinned)   continue;
                            if (entries[i].isDisplaced) continue;
                            if (entries[i].isGrass)     continue;

                            const BufferGeometry* geomKey = entries[i].mesh->geometry().get();
                            if (refreshedGeoms.count(geomKey)) continue;

                            auto cIt = blasCache.find(geomKey);
                            if (cIt == blasCache.end()) continue;
                            auto& rec = *cIt->second;
                            // Already enqueued unconditionally above, and its
                            // host attributes are stale by definition — letting
                            // it fall through here would double-enqueue it and,
                            // worse, take the host-pack branch that overwrites
                            // the producer's data with those stale arrays.
                            if (rec.interop) continue;

                            auto* posAttr = entries[i].mesh->geometry()->getAttribute<float>("position");
                            auto* idxAttr = entries[i].mesh->geometry()->getIndex();
                            if (!posAttr) continue;

                            const uint32_t curVtx = static_cast<uint32_t>(posAttr->count());
                            const uint32_t curIdx = idxAttr ? static_cast<uint32_t>(idxAttr->count()) : 0u;
                            if (curVtx != rec.vertexCount || curIdx != rec.indexCount) {
                                topologyChanged = true;
                                break;
                            }

                            // Streak accounting for graduation: a record dirty
                            // kDynamicGraduationStreak frames in a row is a
                            // per-frame CPU deformer (Flock's merged birds, a
                            // rewritten trail), not an occasional edit, and
                            // moves to the frame-cb path for the rest of its
                            // life.
                            rec.dirtyStreak = (frameSerial_ - rec.lastDirtyFrame <= 1)
                                    ? rec.dirtyStreak + 1u
                                    : 1u;
                            rec.lastDirtyFrame = frameSerial_;

                            auto* nrmAttr = entries[i].mesh->geometry()->getAttribute<float>("normal");
                            // The dynamic route needs normals (the same
                            // requirement refreshGeomBlasBatch enforces) and a
                            // record with no LOD chain. The quiet-window gate
                            // in the selection pass stops re-enqueues while a
                            // mesh deforms, so by graduation time the chain is
                            // gone — if one exists anyway, the drained branch
                            // below is the only place it can be destroyed
                            // safely, so the record stays there this frame.
                            const bool dynamicRoute =
                                    nrmAttr && rec.lodState == BlasRecord::LodState::None &&
                                    (rec.perFrameDynamic ||
                                     rec.dirtyStreak >= BlasRecord::kDynamicGraduationStreak);
                            if (dynamicRoute) {
                                if (!rec.perFrameDynamic) {
                                    // Graduate: allocate the staging ring, one
                                    // slot per frame in flight, each holding
                                    // positions then normals in the buffers'
                                    // own (possibly packed) formats.
                                    const VkDeviceSize nrmBytes = (rec.packedMask & 1u)
                                            ? VkDeviceSize(rec.vertexCount) * sizeof(uint32_t)
                                            : VkDeviceSize(nrmAttr->array().size()) * sizeof(float);
                                    rec.dynStagingSlotBytes = rec.vbBytes + nrmBytes;
                                    rec.dynStaging = createBuffer(
                                            ctx->allocator(), ctx->device(),
                                            rec.dynStagingSlotBytes * kFramesInFlight,
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VMA_MEMORY_USAGE_AUTO,
                                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                                    rec.perFrameDynamic = true;
                                    ++dynGeomStats_.graduated;
                                    // Settling now happens through the frame cb
                                    // — the draining host-side resync pass must
                                    // never touch this record again.
                                    rec.prevVertexResyncPending = false;
                                    std::cerr << "[VulkanRenderer] geometry ("
                                              << rec.vertexCount << " verts) dirty "
                                              << rec.dirtyStreak
                                              << " frames in a row - graduated to the "
                                                 "per-frame dynamic path (frame-cb refit, "
                                                 "no drains)\n";
                                }
                                rec.dynPrevResyncPending = true;
                                pendingDynamicGeomRefits_.push_back({geomKey, &rec});
                            } else {
                                // An in-place vertex rewrite invalidates any auto-LOD
                                // chain: the level BLASes BAKE positions (a stale level
                                // would ray-trace the pre-edit shape) and the chain's
                                // error bounds measured the old surface. Destroyed
                                // past the device-wide drain below — in-flight TLASes
                                // may still reference a level BLAS. lodState=None ⇒
                                // the selection pass re-enqueues against the new
                                // geomVersion; selectLodGeom falls back to LOD0 for
                                // every consumer meanwhile.
                                if (rec.lodState != BlasRecord::LodState::None)
                                    lodChainDoomed.push_back(&rec);
                                refreshOps.push_back({geomKey, &rec});
                            }
                            refreshedGeoms.insert(geomKey);
                        }
                        if (topologyChanged) {
                            // Vertex/index count changed — can't reuse BLAS
                            // buffers. Fall through to the full structural
                            // rebuild path below, and drop any dynamic ops
                            // queued this pass: the rebuild destroys + re-admits
                            // their records from current CPU data (their
                            // geomVersion is only stamped at record time, so
                            // the version mismatch guarantees the re-admit).
                            pendingDynamicGeomRefits_.clear();
                            goto fullRebuild;
                        }
                        if (!refreshOps.empty()) {
                            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (pre-BLAS-refresh)");
                            for (BlasRecord* doomed : lodChainDoomed) {
                                // lodChangedThisFrame_ must be forced: selection
                                // already ran this frame and may have left
                                // en.lodLevel > 0 — the EFFECTIVE level changes
                                // to 0 right here, and without the flag the
                                // geomDescs GPU patch would skip while the TLAS
                                // falls back, leaving a stale per-level index
                                // address behind.
                                destroyBlasLodLevels(*doomed);
                                lodChangedThisFrame_ = true;
                            }
                            refreshGeomBlasBatch(refreshOps);
                            for (const auto& op : refreshOps) geomRefreshedThisFrame.insert(op.geom);
                        }
                    }

                    // ── prevVertex re-sync ──────────────────────────────────
                    // A plain mesh updated in place (e.g. a re-rolled terrain)
                    // keeps prevVertex frozen at its PRE-update positions once it
                    // stops being geom-dirty: refreshGeomBlasBatch's Phase B only
                    // runs on the change frame. The G-buffer VS then reads a stale
                    // inPrevPos every subsequent frame → a constant nonzero motion
                    // vector on a static surface → the denoiser/TAA reject history
                    // forever → the mesh stays noisy and visibly shakes. One frame
                    // after the update settles (the record is no longer refreshed
                    // this frame) we copy vertex → prevVertex so motion collapses
                    // back to zero, exactly like initial-build static geometry.
                    // Skinned/displaced/tet own their own every-frame snapshot and
                    // never enter refreshGeomBlasBatch, so they are untouched.
                    {
                        std::vector<BlasRecord*> resyncRecs;
                        for (auto& [geomKey, recPtr] : blasCache) {
                            BlasRecord* rec = recPtr.get();
                            // Graduated records settle through the frame cb — a
                            // drain here would defeat the whole path. The first
                            // CLEAN frame after a dirty run records one
                            // vertex→prevVertex copy in recordDynamicGeomRefits
                            // and motion collapses to zero, same contract as
                            // the host-side pass below.
                            if (rec->perFrameDynamic) {
                                if (rec->dynPrevResyncPending &&
                                    rec->lastDirtyFrame != frameSerial_) {
                                    pendingDynamicPrevResyncs_.push_back(rec);
                                    rec->dynPrevResyncPending = false;
                                }
                                continue;
                            }
                            if (!rec->prevVertexResyncPending) continue;
                            // Re-snapshotted this frame → keep its change-frame
                            // motion; settle on the next clean frame instead.
                            if (geomRefreshedThisFrame.count(geomKey)) continue;
                            resyncRecs.push_back(rec);
                        }
                        if (!resyncRecs.empty()) {
                            // Same in-flight hazard as the refresh above: prior
                            // frames may still read prevVertex as a vertex buffer,
                            // so drain device-wide before overwriting it.
                            check(vkDeviceWaitIdle(ctx->device()),
                                  "vkDeviceWaitIdle (pre-prevVertex-resync)");
                            VkCommandBuffer cb = beginOneShot();
                            for (BlasRecord* rec : resyncRecs) {
                                if (rec->prevVertex.handle != VK_NULL_HANDLE && rec->vbBytes > 0) {
                                    VkBufferCopy region{};
                                    region.size = rec->vbBytes;
                                    vkCmdCopyBuffer(cb, rec->vertex.handle,
                                                    rec->prevVertex.handle, 1, &region);
                                }
                                rec->prevVertexResyncPending = false;
                            }
                            endAndSubmitOneShot(cb, "prevVertex resync");
                        }
                    }

                    if (!matricesSame || bonesDirtyAny || displacedDirtyAny || grassDirtyAny || tetDirtyAny || geomDirtyAny || morphDirtyAny || interopDirtyAny || lodChangedThisFrame_) {
                        THREEPP_CPUPROF("scene.7_tlasRefitFill");
                        // TLAS refit: needed when instance transforms change
                        // (matricesSame=false) AND when any skinned BLAS was
                        // just rebuilt — the TLAS's per-instance wrapped AABB
                        // is recomputed from the current BLAS extents on
                        // refit, picking up pose-deformed silhouettes that
                        // would otherwise be clipped by the stale TLAS AABB.
                        // BLAS handles + buffer addresses are unchanged so
                        // geomDescs / matDescs stay valid; we just rewrite the
                        // tlasInstancesBuffer in place and call MODE_UPDATE.
                        // EXCEPT auto-LOD: a level switch DOES change which
                        // BLAS (and which index buffer) an entry's instance
                        // references — see the plain-geometry branch below
                        // and the geomDescsCached_ patch after this loop.
                        // Persistent scratch (swapped with pendingTlasInstances_
                        // below) so both vectors keep their capacity — a fresh
                        // ~5 MB allocation per moving frame at 100k instances
                        // otherwise.
                        auto& instances = tlasInstanceScratch_;
                        instances.clear();
                        instances.reserve(entries.size());
                        // instanceCustomIndex == entry index (matches the entries-
                        // indexed geomDescs/matDescs built in the full rebuild);
                        // overlay/skipped entries push no instance, exactly as
                        // their geomDescs/matDescs slots are left default.
                        //
                        // Per SPAN: the BLAS resolve, visibility mask and LOD
                        // eligibility are per-MESH — one lookup per span, not
                        // one per instance. Per-entry work is the transform
                        // write (+ per-entry LOD selection only when the span's
                        // record actually has a chain).
                        for (const auto& sp : entrySpans_) {
                            const MeshEntry& e0 = entries[sp.first];
                            if (e0.isOverlay) continue;// raster-overlay only
                            // Same visibility-group rule as the full rebuild:
                            // blend/transmissive (non-water) → alpha mask so
                            // occlusion queries skip them, camera-parented
                            // viewmodels → no-shadow mask.
                            uint8_t spanMask = kRayMaskOpaque;
                            if (sp.first < matDescsCached_.size() && !e0.isDisplaced) {
                                const auto& cmd = matDescsCached_[sp.first];
                                if (cmd.transmission > 0.0f || cmd.alphaCutoff < 0.0f)
                                    spanMask = kRayMaskAlpha;
                            }
                            if (e0.camAttached) spanMask = kRayMaskNoShadow;
                            if (e0.sensorOnly)
                                spanMask = sensorOnlySurfaces_ ? kRayMaskSensorOnly : 0u;
                            const bool isDeformer = e0.isSkinned || e0.isDisplaced ||
                                                    e0.isGrass || e0.isTet || e0.isMorphed;
                            BlasRecord* rec = nullptr;
                            bool perEntryLod = false;
                            LodGeomSel lodSel0{};
                            if (!isDeformer) {
                                const BufferGeometry* geomKey = e0.mesh->geometry().get();
                                auto it = blasCache.find(geomKey);
                                if (it == blasCache.end()) continue;// shouldn't happen on transform-only
                                rec = it->second.get();
                                perEntryLod = !rec->lodLevels.empty();
                                lodSel0 = selectLodGeom(*rec, 0);
                            }
                            for (uint32_t j = 0; j < sp.count; ++j) {
                                const size_t i = size_t(sp.first) + j;
                                const MeshEntry& en = entries[i];
                                VkDeviceAddress blasAddr = 0;
                                if (en.isSkinned) {
                                    auto* sm = static_cast<SkinnedMesh*>(en.mesh);
                                    if (!sm->skeleton || sm->skeleton->bones.empty()) continue;
                                    auto smIt = skinnedMeshStates.find(sm);
                                    if (smIt == skinnedMeshStates.end()) continue;
                                    blasAddr = smIt->second->blas->address;
                                } else if (en.isDisplaced) {
                                    auto* dm = static_cast<DisplacedMesh*>(en.mesh);
                                    auto dmIt = displacedStates.find(dm);
                                    if (dmIt == displacedStates.end()) continue;
                                    blasAddr = dmIt->second->blas->address;
                                } else if (en.isGrass) {
                                    auto* gm = static_cast<GrassMesh*>(en.mesh);
                                    auto gIt = grassStates.find(gm);
                                    if (gIt == grassStates.end()) continue;
                                    blasAddr = gIt->second->blas->address;
                                } else if (en.isTet) {
                                    auto tIt = tetMeshStates.find(en.mesh);
                                    if (tIt == tetMeshStates.end()) continue;
                                    blasAddr = tIt->second->blas->address;
                                } else if (en.isMorphed) {
                                    auto mIt = morphedMeshStates.find(en.mesh);
                                    if (mIt == morphedMeshStates.end()) continue;
                                    blasAddr = mIt->second->blas->address;
                                } else {
                                    // RT secondary hits (reflections/GI/lidar/probe
                                    // update) read GeometryDesc::indexAddress keyed by
                                    // gl_PrimitiveID from whichever BLAS this instance
                                    // references — it must track the SAME level, or a
                                    // hit against a coarser BLAS misindexes the
                                    // still-LOD0 index buffer. `indexed` rides along: a
                                    // level of a non-indexed soup record IS an indexed
                                    // fetch. Patched into the buffer itself below, only
                                    // on a frame where a level actually changed. When
                                    // the record has NO chain, lodSel0 already equals
                                    // what the full rebuild wrote — skip the patch.
                                    const auto lodSel = perEntryLod ? selectLodGeom(*rec, en.lodLevel)
                                                                    : lodSel0;
                                    blasAddr = lodSel.asAddress;
                                    if (perEntryLod && i < geomDescsCached_.size()) {
                                        geomDescsCached_[i].indexAddress = lodSel.indexAddress;
                                        geomDescsCached_[i].indexed = lodSel.indexed ? 1u : 0u;
                                    }
                                }
                                VkAccelerationStructureInstanceKHR inst{};
                                const auto& e = en.worldMatrix;
                                for (int r = 0; r < 3; ++r) {
                                    for (int c = 0; c < 4; ++c) {
                                        inst.transform.matrix[r][c] = e[c * 4 + r];
                                    }
                                }
                                inst.instanceCustomIndex = static_cast<uint32_t>(i);
                                inst.mask = spanMask;
                                inst.instanceShaderBindingTableRecordOffset = 0;
                                inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                                inst.accelerationStructureReference = blasAddr;
                                instances.push_back(inst);
                            }
                        }
                        if (lodChangedThisFrame_ && !geomDescsCached_.empty()) {
                            // geomDescsCached_ was patched in place above; the
                            // per-frame-in-flight ring carries it to the GPU
                            // stall-free — renderFrame flushes THIS frame's
                            // slot right after its fence signals (before any
                            // recording that consumes it), the other slot when
                            // its frame comes around. Same model as matDescs;
                            // this replaced a vkDeviceWaitIdle per switch
                            // frame, which hitched exactly when the camera
                            // was moving.
                            //
                            // WHOLE-ARRAY on purpose, unlike the matDescs patch:
                            // the entries this touched are chosen inside the TLAS
                            // instance fill above, the frame's hottest loop, and
                            // a level switch is already paying for BLAS
                            // re-references. 64 B/entry with a bounded audience
                            // is not worth a mark in there.
                            markGeomDescsWhole();
                        }
                        const bool blasDeformed = bonesDirtyAny || displacedDirtyAny || grassDirtyAny || tetDirtyAny || morphDirtyAny || geomDirtyAny || lodChangedThisFrame_;
                        // Stage the refit; recordCommandBuffer records it into the
                        // frame cb after the deformable BLAS rebuilds (no drain).
                        pendingTlasInstances_.swap(instances);// scratch keeps its capacity
                        pendingTlasFullBuild_ = blasDeformed;
                        pendingTlasRefit_ = true;
                    }
                    if (!materialValuesSame) {
                        THREEPP_CPUPROF("scene.8_matDescPatch");
                        // Material-values-only update: rebuild MaterialDescs into
                        // the host-side cache, mark every per-frame slot dirty,
                        // and let renderFrame flush after the next fence wait.
                        // Pointers and textures haven't changed, so slot count
                        // and texture indices stay valid; only the pbr floats
                        // need to flow through. The old single-buffer path called
                        // vkDeviceWaitIdle here on every animated-pbr frame —
                        // stalling the whole device just to memcpy a few KB.
                        // Multi-buffered: this frame's slot is safe to write
                        // post-fence; the other slot gets flushed when its turn
                        // comes around (it's still serving the previous frame's
                        // in-flight RT trace right now).
                        // ENTRIES-indexed (one slot per entry, overlay/skipped
                        // slots left default) to match the full-rebuild layout so
                        // gl_InstanceCustomIndexEXT (== entry index) keeps indexing
                        // the right material. Dummy slots are never sampled.
                        // TARGETED PATCH: only entries whose material version
                        // actually bumped (entryMatDirty) get their MaterialDesc
                        // re-derived. materialFromMesh is a ~25-dynamic_cast walk
                        // (plus each albedoTexOf/…/occlusionTexOf is another typed
                        // probe + shared_ptr churn); re-deriving EVERY entry here
                        // turned a single brake-light / blinker emissive toggle into
                        // an O(scene) stall — tens of ms once InstancedMesh foliage
                        // expands the scene to tens of thousands of entries, firing
                        // several times a second while a blinker ticks. matDescsCached_
                        // already holds correct values for every unchanged entry (it
                        // survives matrix-only frames untouched), and a changed
                        // material keeps its pointer + textures (a texture swap aborts
                        // to the full structural rebuild), so its dedup index stays
                        // valid. Full fallback only if the cache size somehow doesn't
                        // match (never on the fast path that reaches here).
                        const bool patchMatDescs = matDescsCached_.size() == entries.size();
                        if (!patchMatDescs) {
                            matDescsCached_.assign(entries.size(), MaterialDesc{});
                            // Every entry re-derived below ⇒ every entry dirty.
                            markMatDescsWhole();
                        }
                        // Same dedup as the full-rebuild path below (consulted only on
                        // the full-rebuild fallback): entries order is identical
                        // (overlay skip matches), so identical Material* pointers
                        // produce identical materialAssetIdx values frame-to-frame.
                        std::unordered_map<const Material*, uint32_t> matAssetMap;
                        for (size_t i = 0; i < entries.size(); ++i) {
                            const MeshEntry& en = entries[i];
                            if (en.isOverlay) continue;// raster-overlay only — no MaterialDesc slot
                            if (patchMatDescs && !entryMatDirty[i]) continue;// unchanged → keep cached desc
                            Mesh* m = en.mesh;
                            MaterialDesc md = materialFromMesh(*m);
                            if (patchMatDescs) {
                                // Material* unchanged → its dedup index is stable.
                                md.materialAssetIdx = matDescsCached_[i].materialAssetIdx;
                            } else {
                                const Material* matKey = m->material().get();
                                auto [matIt, inserted] = matAssetMap.try_emplace(
                                        matKey, static_cast<uint32_t>(matAssetMap.size()));
                                md.materialAssetIdx = matIt->second;
                            }
                            if (auto tex = albedoTexOf(*m)) {
                                md.albedoTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransform, tex);
                            }
                            if (auto tex = roughnessTexOf(*m)) {
                                md.roughnessTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransformRoughMetal, tex);
                            }
                            if (auto tex = metalnessTexOf(*m)) {
                                md.metalnessTexIndex = ensureMaterialTexture(tex);
                                if (md.roughnessTexIndex < 0) copyTexUvTransform(md.uvTransformRoughMetal, tex);
                            }
                            if (auto tex = normalTexOf(*m)) {
                                md.normalTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransformNormal, tex);
                            }
                            if (auto tex = transmissionTexOf(*m)) {
                                md.transmissionTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransformTransmission, tex);
                            }
                            if (auto tex = clearcoatTexOf(*m)) {
                                md.clearcoatTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransformClearcoat, tex);
                            }
                            if (auto tex = clearcoatRoughnessTexOf(*m)) {
                                md.clearcoatRoughnessTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransformClearcoatRough, tex);
                            }
                            if (auto tex = emissiveTexOf(*m)) {
                                md.emissiveTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransformEmissive, tex);
                            }
                            if (auto tex = occlusionTexOf(*m)) {
                                md.occlusionTexIndex = ensureMaterialTexture(tex);
                                copyTexUvTransform(md.uvTransformOcclusion, tex);
                            }
                            if (auto tex = detailTexOf(*m)) {
                                md.detailTexIndex = ensureMaterialTexture(tex);
                            }
                            if (auto tex = detailNormalTexOf(*m)) {
                                md.detailNormalTexIndex = ensureMaterialTexture(tex);
                            }
                            if (auto tex = terrainWeightTexOf(*m)) {
                                md.terrainWeightTexIndex = ensureMaterialTexture(tex);
                                for (int bi = 0; bi < 4; ++bi) {
                                    if (auto bt = terrainBandAlbedoTexOf(*m, bi))
                                        md.terrainBandAlbedoTex[bi] = ensureMaterialTexture(bt);
                                    if (auto bt = terrainBandNormalTexOf(*m, bi))
                                        md.terrainBandNormalTex[bi] = ensureMaterialTexture(bt);
                                }
                            }
                            if (auto tex = terrainNormalTexOf(*m)) {
                                md.terrainNormalTexIndex = ensureMaterialTexture(tex);
                            }
                            matDescsCached_[i] = md;
                            // The dirty RANGE is recorded where the write is, so
                            // the two can never drift. entryMatDirty is set for a
                            // whole span at once (a material is a per-mesh fact),
                            // and spans are contiguous in entry order, so an
                            // instanced span of any size coalesces to one range.
                            markMatDescsDirty(static_cast<uint32_t>(i));
                        }
                        cacheCullFlags(matDescsCached_);
                    }
                    // Anything that reached one of the change branches above
                    // reshapes the DrawInfo/indirect content — invalidate the
                    // per-view indirect skip caches. A fully quiet frame (the
                    // common static case) leaves the version untouched.
                    if (!matricesSame || !materialValuesSame || geomDirtyAny ||
                        bonesDirtyAny || displacedDirtyAny || grassDirtyAny ||
                        tetDirtyAny || morphDirtyAny || lodChangedThisFrame_) {
                        ++drawInputsVersion_;
                    }
                    // Update prevSceneFingerprint so later frames compare
                    // against this frame's state, not stale. (The lean path
                    // diffed + updated prevSceneFingerprint in place.)
                    if (!leanOk) prevSceneFingerprint = std::move(currFp);
                    return;
                }
            }

            fullRebuild:
            // Structural change — invalidate the emissive-tri cache so next
            // frame's buildAndUploadEmissiveTris does a full walk regardless
            // of whether entries.size() happens to match.
            cachedEmissiveEntryCount_ = static_cast<size_t>(-1);

            // Tear down anything in-flight references the old AS / scene-desc
            // buffers via a descriptor set. vkDeviceWaitIdle is the simplest
            // safe choice here; rebuilds are rare so the stall is acceptable.
            //
            // NOTE (retire-queue migration): this drain was deliberately LEFT as
            // a full device wait. Unlike the per-resource swaps that moved to the
            // frame-serial retire queue (material/env textures, particle + sprite
            // caches), the structural teardown below frees EIGHT heterogeneous
            // resource classes as one unit — raw VkImageView + vmaDestroyImage
            // pairs (scratchA/foamImage), Win32 external buffers, descriptor-set
            // pool frees, and the bindless texture-SLOT reclamation
            // (freeTextureSlots / retiredTextureSlots_) which may REUSE a freed
            // slot within this same rebuild. The retire queue models only
            // Buffer/Image2D/AS, and the drain can't be partially lifted, so a
            // drain-free rebuild needs deeper restructuring. Kept as waitIdle.
            if (sceneBuilt_) {
                vkDeviceWaitIdle(ctx->device());
                // Device idle ⇒ reclaim anything the frame-serial queue is
                // holding (e.g. material textures retired earlier this frame)
                // rather than carry it across the rebuild. Safe: fully drained.
                flushRetireQueue();
                if (tlas) {
                    ctx->rt().destroyAccelerationStructure(ctx->device(), tlas, nullptr);
                    tlas = VK_NULL_HANDLE;
                }
                destroyBuffer(ctx->allocator(), tlasBuffer);
                for (auto& b : tlasInstancesBuffers) { destroyBuffer(ctx->allocator(), b); b = {}; }
                for (auto& b : geometryDescsBuffers) {
                    destroyBuffer(ctx->allocator(), b);
                    b = {};
                }
                for (auto& b : materialDescsBuffers) {
                    destroyBuffer(ctx->allocator(), b);
                    b = {};
                }
                tlasBuffer = {};

                // Prune stale cache entries whose underlying objects have been
                // destroyed (typical on model swap). Keeping them risks an
                // address-collision: a fresh BufferGeometry / Texture allocated
                // at the same C++ address would silently match the old cache
                // entry and reuse the wrong GPU resource.
                for (auto it = blasCache.begin(); it != blasCache.end(); ) {
                    if (it->second->liveCheck.expired()) {
                        auto& rec = it->second;
                        // destroyBlasRecord, not a hand-rolled subset — this copy
                        // had already drifted (it missed dynStaging) and would
                        // have leaked the interop exports too. The LOD chain is
                        // still freed separately: blasCache is its only owner and
                        // destroyBlasLodLevels keeps the lodBlasBytes_ accounting.
                        destroyBlasLodLevels(*rec);
                        destroyBlasRecord(*rec);
                        // Drop the force-unpacked mark with the geometry it names.
                        // Same address-collision hazard this whole prune exists
                        // for: the allocator can hand a fresh BufferGeometry the
                        // dead one's address, and a stale mark would then quietly
                        // un-pack an unrelated mesh.
                        forceUnpackedGeoms_.erase(it->first);
                        it = blasCache.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = skinnedMeshStates.begin(); it != skinnedMeshStates.end(); ) {
                    if (it->second->liveCheck.expired()) {
                        destroyBuffer(ctx->allocator(), it->second->baseVertex);
                        destroyBuffer(ctx->allocator(), it->second->baseNormal);
                        destroyBuffer(ctx->allocator(), it->second->skinIndex);
                        destroyBuffer(ctx->allocator(), it->second->skinWeight);
                        for (auto& slot : it->second->boneMatrices) {
                            destroyBuffer(ctx->allocator(), slot);
                        }
                        destroyBuffer(ctx->allocator(), it->second->blasScratch);
                        // destroyBlasRecord, not a hand-rolled subset — the
                        // manual copy here missed rec->color + rec->blasScratch.
                        if (it->second->blas) destroyBlasRecord(*it->second->blas);
                        // Return every ring slot's descriptor set to the pool,
                        // else they leak across remove/re-add cycles and the
                        // next allocateMeshDescriptorSet eventually hits
                        // VK_ERROR_OUT_OF_POOL_MEMORY.
                        for (auto& ds : it->second->skinDescSet) {
                            if (ds == VK_NULL_HANDLE) continue;
                            skinning_->freeMeshDescriptorSet(ds);
                            ds = VK_NULL_HANDLE;
                        }
                        it = skinnedMeshStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = tetMeshStates.begin(); it != tetMeshStates.end(); ) {
                    if (it->second->liveCheck.expired()) {
                        destroyBuffer(ctx->allocator(), it->second->tetIndex);
                        destroyBuffer(ctx->allocator(), it->second->tetWeight);
                        destroyBuffer(ctx->allocator(), it->second->baseNormal);
                        destroyBuffer(ctx->allocator(), it->second->restInv0);
                        destroyBuffer(ctx->allocator(), it->second->restInv1);
                        destroyBuffer(ctx->allocator(), it->second->restInv2);
                        for (auto& slot : it->second->tetPos) destroyBuffer(ctx->allocator(), slot);
                        vulkan::destroyExternalBuffer(ctx->device(), it->second->tetPosExt);
                        destroyBuffer(ctx->allocator(), it->second->blasScratch);
                        // destroyBlasRecord, not a hand-rolled subset — the
                        // manual copy here missed rec->color + rec->blasScratch.
                        if (it->second->blas) destroyBlasRecord(*it->second->blas);
                        for (auto& ds : it->second->tetDescSet) {
                            if (ds == VK_NULL_HANDLE) continue;
                            tetSkinning_->freeMeshDescriptorSet(ds);
                            ds = VK_NULL_HANDLE;
                        }
                        it = tetMeshStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = displacedStates.begin(); it != displacedStates.end(); ) {
                    if (it->second->liveCheck.expired()) {
                        auto& st = it->second;
                        // The deferred-shade inputs (bindings 13/14) cache this
                        // state's fine-cascade height + foam views, set once in
                        // ensureDisplacedState. Those images die with the state,
                        // so the cache must fall back to the dummies before the
                        // all-slots descriptor rewrite later in this build —
                        // otherwise it writes destroyed VkImageViews into live
                        // sets (VUID 02996), and the ICD dereferencing the dead
                        // handle is the load-dependent access violation.
                        for (const auto& c : st->cascades) {
                            if (c.dyn && c.dyn->ht().view == oceanFineHeightView) {
                                oceanFineHeightView = oceanFineHeightDummy.view;
                                oceanFineTileSize   = 0.0f;
                            }
                        }
                        if (st->foamImage.view == oceanFoamView) {
                            oceanFoamView     = oceanFoamDummy.view;
                            oceanFoamTileSize = 0.0f;
                        }
                        if (st->blas) destroyBlasRecord(*st->blas);
                        if (st->scratchA.view  != VK_NULL_HANDLE) vkDestroyImageView(ctx->device(), st->scratchA.view, nullptr);
                        if (st->scratchA.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->scratchA.image, st->scratchA.alloc);
                        if (st->foamImage.view  != VK_NULL_HANDLE) vkDestroyImageView(ctx->device(), st->foamImage.view, nullptr);
                        if (st->foamImage.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->foamImage.image, st->foamImage.alloc);
                        // Host-side state buffers — same set the renderer
                        // teardown destroys; missing them here leaked the
                        // readback trio + foam/wake SSBOs on every scene
                        // switch that removed an ocean (VUID 05137).
                        for (auto& ring : st->heightReadback)
                            for (auto& b : ring) destroyBuffer(ctx->allocator(), b);
                        destroyBuffer(ctx->allocator(), st->foamDisturbBuffer);
                        destroyBuffer(ctx->allocator(), st->wakeTrailBuffer);
                        // PhillipsSpectrum / DynamicSpectrum / IFFT destructors
                        // run on unique_ptr reset.
                        it = displacedStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = grassStates.begin(); it != grassStates.end(); ) {
                    if (it->second->liveCheck.expired()) {
                        auto& st = it->second;
                        if (st->blas) {
                            auto& rec = st->blas;
                            if (rec->as) ctx->rt().destroyAccelerationStructure(ctx->device(), rec->as, nullptr);
                            destroyBuffer(ctx->allocator(), rec->storage);
                            destroyBuffer(ctx->allocator(), rec->vertex);
                            destroyBuffer(ctx->allocator(), rec->index);
                            destroyBuffer(ctx->allocator(), rec->normal);
                            destroyBuffer(ctx->allocator(), rec->uv);
                            destroyBuffer(ctx->allocator(), rec->prevVertex);
                        }
                        destroyBuffer(ctx->allocator(), st->restPos);
                        destroyBuffer(ctx->allocator(), st->heightFrac);
                        it = grassStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = morphedMeshStates.begin(); it != morphedMeshStates.end(); ) {
                    if (it->second->liveCheck.expired()) {
                        auto& rec = it->second->blas;
                        if (rec) {
                            if (rec->as) ctx->rt().destroyAccelerationStructure(ctx->device(), rec->as, nullptr);
                            destroyBuffer(ctx->allocator(), rec->storage);
                            destroyBuffer(ctx->allocator(), rec->vertex);
                            destroyBuffer(ctx->allocator(), rec->index);
                            destroyBuffer(ctx->allocator(), rec->normal);
                            destroyBuffer(ctx->allocator(), rec->uv);
                            destroyBuffer(ctx->allocator(), rec->prevVertex);
                        }
                        it = morphedMeshStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = textureCache.begin(); it != textureCache.end(); ) {
                    if (it->second.ref.expired()) {
                        const uint32_t slot = it->second.slot;
                        if (slot < materialTextures.size()) {
                            destroyImage2D(ctx->allocator(), ctx->device(), materialTextures[slot]);
                            materialTextures[slot] = {};
                            freeTextureSlots.push_back(slot);
                        }
                        it = textureCache.erase(it);
                    } else {
                        ++it;
                    }
                }
                // Slots retired by ensureMaterialTexture's stale-hit guard: the
                // cache entry is already gone (so the loop above can't see
                // them); destroy the orphaned images here — same drained
                // context — and return the slots to the free list.
                for (const uint32_t slot : retiredTextureSlots_) {
                    if (slot < materialTextures.size() && materialTextures[slot].view) {
                        destroyImage2D(ctx->allocator(), ctx->device(), materialTextures[slot]);
                        materialTextures[slot] = {};
                        freeTextureSlots.push_back(slot);
                    }
                }
                retiredTextureSlots_.clear();
            }

            std::vector<VkAccelerationStructureInstanceKHR> instances;
            instances.reserve(entries.size());
            // geomDescs / matDescs are ENTRIES-indexed: one slot per entry, with
            // overlay/skipped entries left default. This makes
            // gl_InstanceCustomIndexEXT == the entry index, so the raster gbuffer
            // id, RT gbuffer id, motionMat and meshMovedBits all live in ONE index
            // space. (Previously these were filtered/contiguous, so any overlay or
            // skipped entry shifted the filtered index away from the entry index —
            // corrupting gbuffer.frag's material/normal-map lookup and the
            // deferred shade's motionMat[] / isMeshMoved() reads for every
            // mesh after the skip.)
            // Dummy slots are never referenced: skipped entries get no TLAS
            // instance and are not drawn in the gbuffer (both skip on isOverlay).
            std::vector<GeometryDesc> geomDescs(entries.size());
            std::vector<MaterialDesc> matDescs(entries.size());
            // Same index space, same lifetime — see entryStableIds_. Filled in
            // the geomDescs loop below so a skipped/overlay entry keeps the 0
            // its dummy slot already means.
            entryStableIds_.assign(entries.size(), uint16_t(0));

            // Material-asset dedup. Meshes sharing one Material C++ pointer
            // get the same matAssetIdx so the deferred shade's bilinear
            // reproject gate can accept their cross-mesh taps (tiled walls
            // etc.). The map grows by ~one entry per UNIQUE Material in the
            // scene — well below entries.size() in typical content.
            std::unordered_map<const Material*, uint32_t> matAssetMap;

            // Per-Material MaterialDesc memo. materialFromMesh + the nine
            // texture-of lookups are a ~24-dynamic_cast/entry walk that is a
            // PURE function of the mesh's Material* (nothing mesh- or instance-
            // specific), yet this loop runs once per TLAS instance. Instanced
            // vegetation multiplies one material across hundreds of entries, so
            // a structural rebuild used to re-derive ~4400 identical descs. Cache
            // the fully-resolved desc (asset idx + bindless texture indices + UV
            // transforms — all Material*-invariant within a rebuild) and reuse
            // it for every later instance of the same material. The first miss
            // still runs ensureMaterialTexture so the bindless slot is allocated;
            // later hits reuse the slot index it returned.
            std::unordered_map<const Material*, MaterialDesc> matDescMemo;

            // Admit every new static geometry (BLAS build) + albedo texture
            // upload through ONE batched submit instead of a queue drain apiece
            // — a burst of tile splits otherwise serialises a dozen
            // vkQueueWaitIdles into this frame. Batching is (re-)armed at the top
            // of every iteration and flushed just before each deformer branch
            // (whose FFT/skin/wind refit submits its own readback-bearing command
            // buffers that must run immediately) and once more before buildTlas.
            for (size_t i = 0; i < entries.size(); ++i) {
                beginOneShotBatch();// (re)arm; idempotent if a batch is already open
                const MeshEntry& en = entries[i];
                Mesh* m = en.mesh;
                // Particle billboard meshes own their vertex buffers in the
                // dedicated billboard pass — they need no BLAS and never enter
                // the TLAS. Skip before the geometry-keyed build so we don't
                // allocate (or per-frame refit) an AS for them.
                if (en.isParticle) continue;
                const BufferGeometry* geomKey = m->geometry().get();

                // The ParticleField MeshRepr proxy is nobody's Mesh geometry —
                // the field carries the zero-area placeholder as its own — so
                // nothing else in this walk would ever upload it, and the
                // particle raster pass pulls its vertices bindlessly. Cached
                // and evicted exactly like any other static geometry; the
                // field's OWN entry keeps the placeholder, which is what keeps
                // its BLAS and TLAS instance harmless (phase 1 particles do not
                // cast traced shadows — that is the §4 AABB BLAS).
                if (en.isParticleField) {
                    ensureCachedBlas(static_cast<ParticleField*>(m)->meshRepr().geometry);
                }

                // Deformer BLAS priming (skinned skin compute, ocean FFT
                // displace, grass wind refit) submits its own command buffers
                // with barriers/readbacks that must execute now — never fold
                // them into the tile-admit batch. Flush what's queued; the next
                // iteration re-arms for the following static entries.
                if (en.isSkinned || en.isDisplaced || en.isGrass || en.isTet || en.isMorphed) {
                    flushOneShotBatch();
                }

                // Skinned meshes get a per-instance deformed BLAS rather than
                // sharing the geometry-keyed cache. Two SkinnedMeshes loaded
                // from the same glTF can share BufferGeometry but never share
                // a pose, so they must not share a BLAS. DisplacedMesh follows
                // the same per-instance BLAS rule (each ocean mesh has its
                // own FFT cascade and its own continuously-rewritten vertex
                // buffer).
                BlasRecord* recPtr = nullptr;
                auto* sm = en.isSkinned ? static_cast<SkinnedMesh*>(m) : nullptr;
                if (sm && sm->skeleton && !sm->skeleton->bones.empty()) {
                    auto* st = ensureSkinnedBlas(*sm);
                    if (!st) continue;
                    recPtr = st->blas.get();
                } else if (en.isDisplaced) {
                    auto* dm = static_cast<DisplacedMesh*>(m);
                    const bool existed = displacedStates.count(dm) > 0;
                    auto* st = ensureDisplacedState(*dm);
                    if (!st) continue;
                    recPtr = st->blas.get();
                    if (existed) {
                        // Existing ocean on a structural rebuild: refresh through
                        // the FRAME command buffer (no blocking one-shot), exactly
                        // as the non-structural deformer path does — a full FFT +
                        // BLAS refit as a synchronous one-shot stalls the CPU on
                        // every tile swap. The BLAS keeps last frame's displaced
                        // content for this frame's TLAS (imperceptible for water);
                        // recordCommandBuffer refits it before the shade trace,
                        // and beginDeferredFrame mirrors the height field after
                        // the fence wait, as on the per-frame path.
                        pendingDisplacedDeforms_.emplace_back(dm, st, static_cast<float>(frameNowSec()));
                        ++dm->frameTick;
                    } else {
                        // First creation: prime synchronously so the very first
                        // TLAS build + ray-trace see the displaced surface, not
                        // the rest grid.
                        refreshDisplacedBlas(*dm, *st, static_cast<float>(frameNowSec()));
                    }
                } else if (en.isGrass) {
                    auto* gm = static_cast<GrassMesh*>(m);
                    const bool existed = grassStates.count(gm) > 0;
                    auto* st = ensureGrassState(*gm);
                    if (!st) continue;
                    recPtr = st->blas.get();
                    if (existed) {
                        // Existing grass: defer the wind refit to the frame cb
                        // (same rationale as the ocean above) — but honour the
                        // distance-gated freeze so a structural rebuild triggered
                        // by unrelated scene churn (e.g. terrain LOD tile swaps as
                        // the camera roams the valley) doesn't wake every frozen
                        // far field. recPtr already keeps the field in the TLAS.
                        const auto& cw = camera.matrixWorld->elements;
                        const Vector3 camPos(cw[12], cw[13], cw[14]);
                        if (grassShouldAnimate(en, camPos)) {
                            pendingGrassDeforms_.emplace_back(gm, st);
                            ++gm->frameTick;
                        }
                    } else {
                        // Prime the BLAS with the first wind pose before the first trace.
                        refreshGrassBlas(*gm, *st, static_cast<float>(frameNowSec()));
                    }
                } else if (en.isTet) {
                    auto* st = ensureTetBlas(*m);
                    if (!st) continue;
                    recPtr = st->blas.get();
                } else if (en.isMorphed) {
                    auto* st = ensureMorphedBlas(*m);
                    if (!st) continue;
                    recPtr = st->blas.get();
                } else {
                    auto it = blasCache.find(geomKey);
                    if (it != blasCache.end()) {
                        const unsigned int curVer = geomVersionOf(*m->geometry());
                        // A geometry marked for the unpacked rebuild whose record
                        // is still packed. Belt and braces: enableVertexInterop
                        // rebuilds inline behind a drain, so in practice a marked
                        // geometry is already unpacked by the time it gets here.
                        // This catches any future path that manages to build a
                        // packed record for a marked geometry anyway — the
                        // failure mode it prevents (a producer writing float xyz
                        // into an snorm16x4 normal buffer) is silent garbage, not
                        // a crash, so it is worth one integer test per rebuild.
                        // Self-clearing: the replacement has packedMask == 0.
                        const bool packedMismatch =
                                (it->second->packedMask != 0u) &&
                                forceUnpackedGeoms_.count(geomKey) != 0;
                        if (it->second->geomVersion != curVer || packedMismatch) {
                            auto& old = it->second;
                            if (old->interop) {
                                // A CPU edit to a SIDE attribute (color / uv /
                                // index contents — counts are fixed: the interop
                                // enqueue loop above tears interop down on any
                                // count change before this runs) bumped the
                                // composite version under an armed interop
                                // registration. Legitimate: the producer owns
                                // position/normal, the CPU still owns the rest.
                                // Evicting here would be three bugs at once —
                                // free exports a foreign API holds imports of,
                                // leave pendingDynamicGeomRefits_ (filled
                                // earlier this pass) dangling at the freed
                                // record, and rebuild from host position arrays
                                // that are stale by contract, freezing the mesh
                                // at whatever the CPU last saw. Transplant
                                // instead: build the replacement from host data
                                // (which is exactly how the fresh color/uv/index
                                // contents get uploaded), move the interop
                                // machinery across, and re-point the pending
                                // refit — this same frame's recorded refit then
                                // overwrites vertex/normal from the exports, so
                                // the stale host positions are never presented.
                                auto fresh = buildBlasFor(*m->geometry(), /*allowPacked=*/true);
                                // >=, not ==: createExternalBuffer reports the
                                // ALLOCATION size, which the allocator pads up
                                // (30752 for a 30744-byte request), so equality
                                // spuriously fails for any vertex count whose
                                // byte size is not 16-aligned. The requirement
                                // is only that the export can source the full
                                // head-of-frame copy into the new buffers.
                                const bool fits = fresh &&
                                        fresh->vertexCount == old->vertexCount &&
                                        fresh->indexCount == old->indexCount &&
                                        fresh->vertex.size <= old->posExt.size &&
                                        fresh->normal.size <= old->nrmExt.size;
                                if (fits) {
                                    fresh->liveCheck = m->geometry();
                                    fresh->posExt = old->posExt;
                                    old->posExt = {};
                                    fresh->nrmExt = old->nrmExt;
                                    old->nrmExt = {};
                                    fresh->externalCopy = std::move(old->externalCopy);
                                    old->externalCopy = nullptr;
                                    fresh->sanitizeDS = old->sanitizeDS;
                                    old->sanitizeDS = VK_NULL_HANDLE;
                                    fresh->interopValidate = old->interopValidate;
                                    fresh->interop = true;
                                    old->interop = false;
                                    fresh->perFrameDynamic = true;
                                    destroyBlasLodLevels(*old);
                                    destroyBlasRecord(*old);// interop fields cleared above
                                    BlasRecord* freshPtr = fresh.get();
                                    it->second = std::move(fresh);
                                    for (auto& op : pendingDynamicGeomRefits_) {
                                        if (op.geom == geomKey) op.rec = freshPtr;
                                    }
                                    recPtr = freshPtr;
                                } else {
                                    // Cannot transplant (an attribute appeared or
                                    // vanished, changing the buffer set). Tear the
                                    // interop down DELIBERATELY — the exports are
                                    // released behind disableVertexInterop's drain
                                    // and the application is told — then evict as
                                    // for any other geometry change.
                                    std::cerr << "[VulkanRenderer] vertex interop: geometry "
                                                 "attributes changed shape under an armed interop "
                                                 "registration - interop is being DISABLED for "
                                                 "this mesh. Edit only the CONTENTS of side "
                                                 "attributes (color/uv/index) while interop is "
                                                 "armed.\n";
                                    disableVertexInterop(*m);
                                    pendingDynamicGeomRefits_.erase(
                                            std::remove_if(pendingDynamicGeomRefits_.begin(),
                                                           pendingDynamicGeomRefits_.end(),
                                                           [&](const auto& op) { return op.geom == geomKey; }),
                                            pendingDynamicGeomRefits_.end());
                                    destroyBlasLodLevels(*old);
                                    destroyBlasRecord(*old);
                                    blasCache.erase(it);
                                    it = blasCache.end();
                                }
                            } else {
                            // Topology/positions changed under this geometry
                            // pointer — any existing LOD chain simplified the
                            // OLD data and no longer matches. Destroy it; the
                            // fresh BlasRecord created below starts at
                            // LodState::None and the selection pass re-
                            // enqueues a new chain next time it's eligible.
                            destroyBlasLodLevels(*old);
                            destroyBlasRecord(*old);
                            blasCache.erase(it);
                            // erase() returns the next bucket entry, not end().
                            // Force a fresh lookup so the "missing → build" branch
                            // below fires; otherwise recPtr binds to a sibling
                            // cache entry (different mesh's BLAS) and the TLAS
                            // instance ends up referencing the wrong AS.
                            it = blasCache.end();
                            }
                        }
                    }
                    if (it == blasCache.end()) {
                        auto rec = buildBlasFor(*m->geometry(), /*allowPacked=*/true);
                        if (!rec) continue;// degenerate / unsupported geometry
                        rec->liveCheck = m->geometry();
                        it = blasCache.emplace(geomKey, std::move(rec)).first;
                    }
                    recPtr = it->second.get();
                }

                // Overlay meshes need a BlasRecord (vertex buffer for the
                // raster overlay pass) but must not appear in the TLAS or
                // GeometryDesc/MaterialDesc arrays — the traced/rasterized
                // scene must not see them.
                if (en.isOverlay) continue;

                VkAccelerationStructureInstanceKHR inst{};
                // VkTransformMatrixKHR is row-major 3x4; threepp Matrix4 is
                // column-major 4x4 (elements[c*4 + r]). For InstancedMesh the
                // worldMatrix already incorporates the per-instance transform.
                const auto& e = en.worldMatrix;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        inst.transform.matrix[r][c] = e[c * 4 + r];
                    }
                }
                inst.instanceCustomIndex = static_cast<uint32_t>(i);
                inst.mask = kRayMaskOpaque;// placeholder; set to Opaque/Alpha once md is built below
                inst.instanceShaderBindingTableRecordOffset = 0;
                inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                // en.lodLevel==0 for every deformer/exempt entry (auto-LOD
                // selection above only ever sets it on plain cached
                // geometry), so this passthrough is a no-op for them —
                // recPtr may point at a SkinnedMeshState/DisplacedMeshState/
                // etc.'s own BlasRecord, which selectLodGeom treats
                // identically to a LOD0 blasCache record.
                const auto lodSel = selectLodGeom(*recPtr, en.lodLevel);
                inst.accelerationStructureReference = lodSel.asAddress;
                instances.push_back(inst);

                GeometryDesc gdesc{};
                gdesc.vertexAddress = recPtr->vertex.address;
                gdesc.normalAddress = recPtr->normal.address;
                // Must reference the SAME index buffer the selected BLAS
                // level was built from — RT secondary hits (reflections/GI/
                // lidar/probe update) read this keyed by gl_PrimitiveID,
                // which only lines up against that exact index array.
                gdesc.indexAddress  = lodSel.indexAddress;
                gdesc.uvAddress     = recPtr->uv.address;// 0 if no UV attribute
                gdesc.foamAddress   = recPtr->isOceanSurface ? 1ull : 0ull;// 0/1 flag (not an address); 1 == FFT-displaced ocean surface
                // prevVertexAddress: skinned + displaced meshes have a real
                // prev-vertex buffer (different from current). Static meshes
                // get vertex.address as a fallback — the chit reads the same
                // buffer for both, prevHitWorldPos ends up equal to current.
                //
                // A warp-enabled DisplacedMesh is the exception: its vertices
                // reflow every frame as the adaptive grid re-centres on the
                // vessel, so prevVertex (indexed by vertex id) doesn't track a
                // stable world point — using it corrupts motion vectors. Fall
                // back to the current buffer so the chit's prevWorldPos
                // resolves to hitWorldPos: the ocean reprojects as world-
                // static (camera-parallax only), matching the foam texture.
                // An interop record whose producer declared unstable vertex
                // correspondence (interopWorldStatic — marching-cubes soups)
                // gets the same world-static fallback.
                bool warpReproject = recPtr->interopWorldStatic;
                if (en.isDisplaced) {
                    warpReproject = static_cast<DisplacedMesh*>(m)->warp.halfRange > 0.0f;
                }
                gdesc.prevVertexAddress =
                        (recPtr->prevVertex.handle != VK_NULL_HANDLE && !warpReproject)
                                ? recPtr->prevVertex.address
                                : recPtr->vertex.address;
                // Per SELECTION, not per record: a level of a non-indexed
                // soup record IS indexed (welded canonical indices).
                gdesc.indexed = lodSel.indexed ? 1u : 0u;
                // Per-vertex color only when the material opts in (three.js
                // semantics: vertexColors == true) AND the geometry uploaded a
                // color buffer. 0 otherwise → chit skips the modulation.
                {
                    const auto mat = m->material();
                    const bool useVtxColor = recPtr->color.handle != VK_NULL_HANDLE &&
                                             mat && mat->vertexColors;
                    gdesc.colorAddress = useVtxColor ? recPtr->color.address : 0;
                }
                // Bit 0 (moved-sticky) is stamped per frame in VulkanCoreFrame;
                // seed it 0 here and carry the packed-attribute bits above it.
                gdesc.flags = recPtr->packedMask << 1;
                geomDescs[i] = gdesc;
                // Read-only: assignment of auto ids stays in the indirect draw
                // builder so this cannot renumber what the Ids AOV reports.
                entryStableIds_[i] = stableIdIfAssigned(*en.mesh);

                const Material* matKey = m->material().get();
                MaterialDesc md;
                auto memoIt = matDescMemo.find(matKey);
                if (memoIt != matDescMemo.end()) {
                    md = memoIt->second;// identical for every instance of this material
                } else {
                    md = materialFromMesh(*m);
                    {
                        auto [matIt, inserted] = matAssetMap.try_emplace(
                                matKey, static_cast<uint32_t>(matAssetMap.size()));
                        md.materialAssetIdx = matIt->second;
                    }
                    if (auto tex = albedoTexOf(*m)) {
                        md.albedoTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransform, tex);
                    }
                    if (auto tex = roughnessTexOf(*m)) {
                        md.roughnessTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransformRoughMetal, tex);
                    }
                    if (auto tex = metalnessTexOf(*m)) {
                        md.metalnessTexIndex = ensureMaterialTexture(tex);
                        if (md.roughnessTexIndex < 0) copyTexUvTransform(md.uvTransformRoughMetal, tex);
                    }
                    if (auto tex = normalTexOf(*m)) {
                        md.normalTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransformNormal, tex);
                    }
                    if (auto tex = transmissionTexOf(*m)) {
                        md.transmissionTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransformTransmission, tex);
                    }
                    if (auto tex = clearcoatTexOf(*m)) {
                        md.clearcoatTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransformClearcoat, tex);
                    }
                    if (auto tex = clearcoatRoughnessTexOf(*m)) {
                        md.clearcoatRoughnessTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransformClearcoatRough, tex);
                    }
                    if (auto tex = emissiveTexOf(*m)) {
                        md.emissiveTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransformEmissive, tex);
                    }
                    if (auto tex = occlusionTexOf(*m)) {
                        md.occlusionTexIndex = ensureMaterialTexture(tex);
                        copyTexUvTransform(md.uvTransformOcclusion, tex);
                    }
                    if (auto tex = detailTexOf(*m)) {
                        md.detailTexIndex = ensureMaterialTexture(tex);
                    }
                    if (auto tex = detailNormalTexOf(*m)) {
                        md.detailNormalTexIndex = ensureMaterialTexture(tex);
                    }
                    if (auto tex = terrainWeightTexOf(*m)) {
                        md.terrainWeightTexIndex = ensureMaterialTexture(tex);
                        // Band sets are SHARED textures (one allocation across
                        // every tile material — ensureMaterialTexture dedupes
                        // by pointer), so this costs 8 slots total, not 8/tile.
                        for (int bi = 0; bi < 4; ++bi) {
                            if (auto bt = terrainBandAlbedoTexOf(*m, bi))
                                md.terrainBandAlbedoTex[bi] = ensureMaterialTexture(bt);
                            if (auto bt = terrainBandNormalTexOf(*m, bi))
                                md.terrainBandNormalTex[bi] = ensureMaterialTexture(bt);
                        }
                    }
                    if (auto tex = terrainNormalTexOf(*m)) {
                        md.terrainNormalTexIndex = ensureMaterialTexture(tex);
                    }
                    matDescMemo.emplace(matKey, md);
                }
                matDescs[i] = md;
                // Visibility group (see vulkan_shared.h): blend/transmissive
                // surfaces move to the alpha mask so pure-visibility occlusion
                // queries (env gather, GI, emissive-NEE) cull them out — a text
                // decal's transparent quad must not block IBL light. Water
                // (DisplacedMesh) stays opaque-mask: underwater light transport
                // is handled by its volumetrics, not pass-through.
                if (!instances.empty()) {
                    instances.back().mask =
                            en.sensorOnly
                                    // Sensors only, and only once opted in: mask 0
                                    // is hit by no ray at all.
                                    ? (sensorOnlySurfaces_ ? kRayMaskSensorOnly : 0u)
                            : en.camAttached
                                    ? kRayMaskNoShadow// FP viewmodel: visible, never occludes
                                    : (!en.isDisplaced && (md.transmission > 0.0f || md.alphaCutoff < 0.0f))
                                              ? kRayMaskAlpha
                                              : kRayMaskOpaque;
                }
            }

            // Submit the whole tile-admit batch once (BLAS builds + prevVertex
            // seeds + texture staging/blits), waiting a single time. After this
            // returns every new BLAS is built and every new albedo texture is
            // resident, so buildTlas below can reference the fresh BLAS.
            flushOneShotBatch();
            buildTlas(instances);
            // Every per-frame GeometryDesc slot seeded fresh — same ring
            // model as the matDescs loop below; the lean auto-LOD patch
            // flips geomDescsDirty_ and flushGeometryDescsIfDirty carries
            // deltas per slot from then on.
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                uploadDescBuffer(geometryDescsBuffers[f], geomDescs);
            }
            // Every slot now holds the fresh array, so no slot owes anything —
            // including any range marked earlier in this same frame, which this
            // upload has just superseded.
            for (auto& d : geomDescsDirty_) d.clear();
            // Host mirror for the lean-path auto-LOD patch above. geomDescs
            // just built already reflects each entry's CURRENT en.lodLevel
            // (the selection pass runs before this heavy rebuild, so an
            // entry whose geometry/BLAS wasn't touched by this particular
            // rebuild can still carry a >0 level from before) — this is
            // just the persistent copy the lean path patches in place.
            geomDescsCached_ = geomDescs;
            // Seed every per-frame slot with the fresh matDescs so the first
            // few frames don't try to flush against a half-initialised ring.
            // matDescsCached_ stays in sync as the host-side authoritative
            // copy used by the hot-path flush.
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                uploadDescBuffer(materialDescsBuffers[f], matDescs);
            }
            matDescsCached_ = matDescs;
            for (auto& d : matDescsDirty_) d.clear();
            cacheCullFlags(matDescs);

            // Topology rebuild vs temporal history. Nothing consumed ACROSS
            // frames keys on the entry order any more: the reproject guards
            // compare the STABLE per-object id (ids .y) and read the prev
            // texel's own moved bit (ids .z, kInstFlagMoving) instead of
            // dereferencing geoms[prev ids .x], the ReSTIR reservoirs store
            // world-space light samples (position + type, never an index),
            // and the prev-world motion references are identity-remapped
            // below alongside meshMovedSticky_. The
            // per-pixel depth/normal gates then cold-start exactly the pixels
            // whose surface really changed. So neither an APPEND (spawned
            // PhysX body, grown ParticleSystem) nor a mid-list REMOVAL /
            // REORDER (a terrain tile split/merge swaps the parent for its
            // children every few frames for seconds at a stretch) needs a
            // global reset. The old positional stable-prefix check fell back
            // to clearGbufImages + sampleIndex=0 + prevWorldMats.clear() on
            // every removal — restarting the whole scene's accumulation AND
            // the RNG sample stream a few frames apart for the entire
            // streaming burst, which read as white shimmer on high-contrast
            // edges under DLSS/TAA.
            //
            // What still keys positionally is HOST state that outlives the
            // frame, so remap it by identity instead of resetting:
            //   meshMovedSticky_ — a removed entry's "recently moved"
            //     countdown must not land on an unrelated survivor (a stale
            //     moving label makes the trailing-edge guards reject history
            //     at every silhouette for 30 frames — edge shimmer).
            //   prevWorldByEntry_ — remap by the same identity so a surviving
            //     entry keeps its motion reference (a reset fakes identity
            //     motion scene-wide for a frame); removed entries' slots are
            //     simply dropped with the old index space.
            // (meshMovedBits_ is refilled in CURRENT index space every
            // ensureSceneBuilt pass, so it needs no remap.)
            //
            // A duplicate (mesh, instanceIndex) key would make the identity
            // match ambiguous — not expected, every entry is its own TLAS
            // slot — so that lone case falls back to the old global reset.
            std::vector<std::array<float, 16>> oldPrevWorld = std::move(prevWorldByEntry_);
            std::vector<uint8_t> oldPrevValid = std::move(prevWorldValidByEntry_);
            seedMotionState(entries);// identity motion + prev=current for every slot
            bool identityOk = sceneBuilt_;
            if (identityOk) {
                std::unordered_map<EntryKey, size_t, EntryKeyHash> oldIdx;
                oldIdx.reserve(prevSceneFingerprint.size());
                for (size_t i = 0; i < prevSceneFingerprint.size() && identityOk; ++i) {
                    const auto& p = prevSceneFingerprint[i];
                    identityOk = oldIdx.try_emplace(
                            EntryKey{static_cast<const Mesh*>(p.mesh), p.instanceIndex}, i).second;
                }
                if (identityOk) {
                    std::vector<uint32_t> remappedSticky(currFp.size(), 0u);
                    for (size_t i = 0; i < currFp.size(); ++i) {
                        const auto it = oldIdx.find(
                                EntryKey{static_cast<const Mesh*>(currFp[i].mesh), currFp[i].instanceIndex});
                        if (it == oldIdx.end()) continue;// new entry — no history
                        if (it->second < meshMovedSticky_.size())
                            remappedSticky[i] = meshMovedSticky_[it->second];
                        if (it->second < oldPrevWorld.size() && oldPrevValid[it->second] &&
                            i < prevWorldByEntry_.size()) {
                            prevWorldByEntry_[i] = oldPrevWorld[it->second];
                        }
                        oldIdx.erase(it);// leftovers = removed entries (slots dropped)
                    }
                    meshMovedSticky_.swap(remappedSticky);
                    stickyActiveCount_ = 0;
                    for (uint32_t v : meshMovedSticky_)
                        if (v > 0u) ++stickyActiveCount_;
                }
            }
            if (!identityOk) {
                // First build (nothing to preserve) or an ambiguous match.
                // We're already past vkDeviceWaitIdle so the clear is synchronous.
                clearGbufImages();
                sampleIndex = 0;
                meshMovedSticky_.clear();
                stickyActiveCount_ = 0;
            }

            // Grow motion-mat + mesh-moved-bits buffers if the new instance
            // count exceeds the current capacity. The descriptor write below
            // (or the initial allocate, on first build) will pick up the new
            // buffer handles.
            // motionMat[] and meshMovedBits[] are indexed by ENTRY index
            // (== instanceCustomIndex == the meshID the gbuffer writes; see the
            // geomDescs/matDescs "ONE index space" note above), NOT by the
            // compact TLAS instance count. Size them to entries.size(): when
            // leading entries are overlay (e.g. wireframe PointLightHelpers
            // added to the scene before the traced/rasterized meshes), those
            // meshes land at entry indices >= instances.size(), so a buffer
            // sized to the instance count is read out of bounds in the
            // shader — a GPU fault that surfaces at the next AS one-shot
            // fence wait (looks like a "tblas refit" crash).
            // computeAndUploadMotionMatrices uploads
            // entries.size() matrices, so the instance-count sizing overflowed
            // the host buffer too.
            const uint32_t motionSlots    = static_cast<uint32_t>(entries.size());
            const uint32_t neededBitWords = std::max<uint32_t>((motionSlots + 31u) / 32u, 1u);
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                ensureMotionMatCapacity(f, std::max<uint32_t>(motionSlots, 1u));
                ensureMeshMovedBitsCapacity(f, neededBitWords);
            }
            meshMovedBits_.resize(neededBitWords, 0u);

            // Re-point the deferred shade's descriptor set at the (re)built
            // scene resources — TLAS, geom/material descs, the bindless
            // material-texture array and the emissive-tri buffer. A rebuild
            // freed and recreated those, so the set would otherwise dangle.
            //
            // Every view: the scene is shared, so a rebuild invalidates every
            // view's set, not just the one that happens to be current. An
            // editor rebuilds on nearly every edit, which is where leaving the
            // secondaries out shows up first.
            forEachLiveView([&] { rewriteDeferredDescriptors(); });
            // The (re)build above rebound the bindless texture array. The
            // raster descriptor's binding 3 mirrors that same table, so
            // invalidate its per-slot cache — each frame slot then re-writes
            // binding 3 on its next uploadRasterCameraUbo. Per view: the flag
            // guards per-view sets, and recordSecondaryViews calls
            // uploadRasterCameraUbo once per view per frame, so each view
            // heals its own slot.
            for (auto& v : views_) v->rasterMatTexValid_.fill(0);
            prevSceneFingerprint = std::move(currFp);
            sceneBuilt_ = true;
            ++drawInputsVersion_;// structural rebuild — every draw record is new

            // Companion to the post-init dump: with the scene's BLASes +
            // geometry buffers now resident, allocationBytes − the post-init
            // baseline is the scene's true device cost (the OS-level per-
            // process counter only moves in VMA block granularity, so it
            // under-reports buffer-level savings such as packed attributes).
            if (const char* e = std::getenv("THREEPP_VK_MEMDUMP"); e && *e && *e != '0') {
                dumpMemoryStats("scene-built");
            }
        }

void VulkanRenderer::Impl::cacheCullFlags(const std::vector<MaterialDesc>& mds) {
            lastVisibleCullMode_.resize(mds.size());
            for (size_t i = 0; i < mds.size(); ++i) {
                switch (mds[i].sideMode) {
                    case 0:  lastVisibleCullMode_[i] = VK_CULL_MODE_BACK_BIT;  break;
                    case 1:  lastVisibleCullMode_[i] = VK_CULL_MODE_FRONT_BIT; break;
                    default: lastVisibleCullMode_[i] = VK_CULL_MODE_NONE;      break;
                }
            }
        }

void VulkanRenderer::Impl::collectWorldSprites(Object3D& scene) {
            lastVisibleSprites_.clear();
            scene.traverseVisible([&](Object3D& o) {
                auto* sp = dynamic_cast<Sprite*>(&o);
                if (!sp || sp->screenSpace) return;
                auto mat = sp->material();
                if (!mat || !mat->visible) return;
                auto* mm = dynamic_cast<MaterialWithMap*>(mat.get());
                if (!mm || !mm->map) return;// untextured world sprites aren't drawn
                WorldSpriteEntry e{};
                std::memcpy(e.world.data(), sp->matrixWorld->elements.data(), 64);
                e.color = {1.f, 1.f, 1.f, 1.f};
                if (auto* mc = dynamic_cast<MaterialWithColor*>(mat.get())) {
                    e.color[0] = mc->color.r;
                    e.color[1] = mc->color.g;
                    e.color[2] = mc->color.b;
                }
                e.color[3] = mat->opacity;
                if (auto* mr = dynamic_cast<MaterialWithRotation*>(mat.get())) {
                    e.rotation = mr->rotation;
                }
                e.center = sp->center;
                e.tex = mm->map.get();
                lastVisibleSprites_.push_back(e);
            });
        }

void VulkanRenderer::Impl::collectSplatClouds(Object3D& scene, Camera& camera) {
            lastVisibleSplats_.clear();
            if (!splat_) return;

            camera.updateWorldMatrix(true, false);

            // Everything the pass needs from the camera. The projection stays
            // in threepp's GL convention (NDC z in [-1,1], y up): the splat
            // pass does its own NDC -> pixel mapping, so it never wants the
            // reverse-Z form — except for linearizing the G-buffer depth,
            // which is the one thing that WAS written with it.
            std::memcpy(splatParams_.view, camera.matrixWorldInverse.elements.data(), 64);
            std::memcpy(splatParams_.proj, camera.projectionMatrix.elements.data(), 64);
            std::memcpy(splatParams_.camWorld, camera.matrixWorld->elements.data(), 64);
            const Matrix4 projRev = reverseZVk(camera.projectionMatrix);
            std::memcpy(splatProjRevZ_, projRev.elements.data(), 64);
            Matrix4 projInv;
            projInv.copy(projRev).invert();
            std::memcpy(splatParams_.projInverse, projInv.elements.data(), 64);

            const auto& cw = camera.matrixWorld->elements;
            splatParams_.camPos[0] = cw[12];
            splatParams_.camPos[1] = cw[13];
            splatParams_.camPos[2] = cw[14];
            // Camera forward is -Z of its world matrix (threepp convention).
            splatParams_.camFwd[0] = -cw[8];
            splatParams_.camFwd[1] = -cw[9];
            splatParams_.camFwd[2] = -cw[10];
            splatParams_.orthographic = camera.is<OrthographicCamera>();
            splatParams_.depthTest    = true;
            if (auto* pc = dynamic_cast<PerspectiveCamera*>(&camera)) {
                splatParams_.nearPlane = pc->nearPlane;
            } else if (auto* oc = dynamic_cast<OrthographicCamera*>(&camera)) {
                splatParams_.nearPlane = oc->nearPlane;
            }

            // Full traverse, not traverseVisible: a hidden cloud must be
            // COLLECTED so the pass can PARK it (keep its GPU buffers) rather
            // than age it out — the difference between a visibility toggle
            // costing nothing and costing a seconds-long re-upload each way.
            lastParkedSplats_.clear();
            scene.traverse([&](Object3D& o) {
                auto* sc = dynamic_cast<SplatCloud*>(&o);
                if (!sc || sc->splatCount() == 0) return;
                auto mat = sc->material();
                bool effectiveVisible = !(mat && !mat->visible);
                for (Object3D* n = &o; effectiveVisible && n; n = n->parent)
                    if (!n->visible) effectiveVisible = false;
                if (!effectiveVisible) {
                    lastParkedSplats_.push_back(sc);
                    return;
                }

                vulkan::SplatPass::CloudEntry e{};
                e.cloud = sc;
                e.debugNonFinite = sc->debugNonFinite();
                std::memcpy(e.model, sc->matrixWorld->elements.data(), 64);

                // What the app asked to draw this frame (SplatCloud::
                // setSubmitRanges); empty is the whole cloud.
                e.ranges = sc->submitRanges();

                // THREEPP_VK_SPLAT_RANGESPLIT=K submits the cloud as K
                // contiguous ranges covering ALL of it, in order. That is the
                // identity by construction — the compact index sequence is the
                // original one — so it is how the range path is proved
                // bit-for-bit against the whole-cloud path before a selection
                // policy exists to submit anything more interesting.
                if (const char* rs = std::getenv("THREEPP_VK_SPLAT_RANGESPLIT")) {

                    const int k = std::atoi(rs);
                    if (k > 1) {
                        e.ranges.clear();// the probe overrides the app's choice
                        const uint32_t total = static_cast<uint32_t>(sc->splatCount());
                        const uint32_t per   = (total + static_cast<uint32_t>(k) - 1u) /
                                               static_cast<uint32_t>(k);
                        for (uint32_t off = 0; off < total; off += per)
                            e.ranges.emplace_back(off, std::min(per, total - off));
                    }
                }

                // p1 / p99 of the view distances, from a fixed-stride sample —
                // the same estimator, the same sample size and the same
                // percentiles the GL path uses (SplatCloud.cpp's SORT_CLAMP_*),
                // because the tail-band doctrine only works if both ends of it
                // agree on where the content interval is. A stride sample is
                // exact-repeatable (no RNG, no state), and 8192 depths pin a
                // 1st percentile far more tightly than a sort key needs.
                //
                // Only the interval comes from here; the exact min/max come
                // from the projection pass's atomics on the GPU, so a splat
                // outside the sample still lands in a monotone tail bucket
                // instead of collapsing onto the end.
                Matrix4 modelView;
                modelView.multiplyMatrices(camera.matrixWorldInverse, *sc->matrixWorld);
                const auto& mv = modelView.elements;

                const auto& means = sc->data().means;
                const size_t n = means.size();
                constexpr size_t kSampleTarget = 8192;// == SORT_SAMPLE_TARGET
                const size_t stride = std::max<size_t>(1, n / kSampleTarget);
                std::vector<float> sample;
                sample.reserve(n / stride + 1);
                for (size_t i = 0; i < n; i += stride) {
                    const auto& m = means[i];
                    // View distance is positive in front of the camera; GL view
                    // space looks down -z, so it is the negated z row.
                    const float d = -(mv[2] * m.x + mv[6] * m.y + mv[10] * m.z + mv[14]);
                    if (std::isfinite(d) && d > 0.f) sample.push_back(d);
                }
                if (sample.size() >= 3) {
                    const auto last   = sample.size() - 1;
                    const auto loRank = static_cast<size_t>(0.01f * static_cast<float>(last));
                    const auto hiRank = static_cast<size_t>(0.99f * static_cast<float>(last));
                    std::nth_element(sample.begin(),
                                     sample.begin() + static_cast<std::ptrdiff_t>(loRank),
                                     sample.end());
                    e.pLo = sample[loRank];
                    std::nth_element(sample.begin() + static_cast<std::ptrdiff_t>(loRank) + 1,
                                     sample.begin() + static_cast<std::ptrdiff_t>(hiRank),
                                     sample.end());
                    e.pHi = sample[hiRank];
                    // A little air on each end so the p1/p99 splats themselves
                    // are not sitting in the tail bands (SORT_CLAMP_MARGIN).
                    const float mid  = 0.5f * (e.pLo + e.pHi);
                    const float half = 0.5f * (e.pHi - e.pLo) * 1.02f;
                    e.pLo = mid - half;
                    e.pHi = mid + half;
                } else if (!sample.empty()) {
                    e.pLo = sample.front();
                    e.pHi = sample.back();
                }

                lastVisibleSplats_.push_back(e);
            });

            // The CURRENT env view, every frame: resize() also hands it over,
            // but resize() runs on swapchain lifecycle and the environment can
            // be rebuilt between resizes — the pass would then write the DEAD
            // view into fresh descriptor sets at the next cloud upload.
            splat_->setEnvironment(envImage.view, envImage.sampler, envImage.mipLevels);
            splat_->syncClouds(lastVisibleSplats_, lastParkedSplats_);

            // ── Overlay occlusion latch ────────────────────────────────────
            // The post-resolve overlay (wireframe, lines, world sprites,
            // particle billboards) is depth-tested against a buffer the splat
            // compositor never writes, so a cloud in front of an overlay does
            // not hide it. recordSplatOverlayDepthStamp fixes that by stamping
            // the depth AOV into that buffer — which means the AOV has to
            // EXIST the moment a scene holds both, whether or not the app ever
            // asked for it.
            //
            // Here rather than in the recorder because turning it on
            // reallocates the render-extent resources (the AOV image is 1x1
            // while off), and this runs before renderFrame in the same
            // render() call — so the pending-realloc gate beginDeferredFrame
            // already owns applies it to THIS frame, at the point where the
            // device is idled. Sticky: see splatOverlayDepth_.
            if (!splatOverlayDepth_ && !lastVisibleSplats_.empty() && sceneHasOverlayContent()) {
                splatOverlayDepth_ = true;
                primaryView().rasterGbufs[0].width = 0;// force the image rebuild
                if (frameState_ != FrameState::Idle) {
                    pendingRenderScaleRealloc_ = true;
                } else {
                    vkDeviceWaitIdle(ctx->device());
                    reallocateRenderExtentResources();
                }
            }
        }


VulkanRenderer::Impl::MaterialDesc VulkanRenderer::Impl::materialFromMesh(const Mesh& m) {
            MaterialDesc d{};
            d.albedo[0] = d.albedo[1] = d.albedo[2] = 0.8f;
            d.roughness = 0.5f;
            d.metalness = 0.0f;
            d.emissive[0] = d.emissive[1] = d.emissive[2] = 0.0f;
            d.emissiveIntensity = 1.0f;
            d.albedoTexIndex = -1;
            d.roughnessTexIndex = -1;
            d.metalnessTexIndex = -1;
            d.normalTexIndex = -1;
            d.normalScale[0] = 1.0f;
            d.normalScale[1] = 1.0f;
            d.alphaCutoff = 0.0f;// disabled by default; any-hit short-circuits on alphaCutoff <= 0
            d.transmission = 0.0f;// opaque by default
            d.ior          = 1.5f;// glass-typical default; only consulted when transmission > 0
            d.transmissionTexIndex = -1;
            d.clearcoat = 0.0f;// no coat by default; lobe is skipped when clearcoat == 0
            d.clearcoatRoughness = 0.0f;
            d.clearcoatTexIndex = -1;
            d.clearcoatRoughnessTexIndex = -1;
            d.attenuationColor[0] = d.attenuationColor[1] = d.attenuationColor[2] = 1.0f;
            d.attenuationDistance = 0.0f;
            d.emissiveTexIndex = -1;
            d.specularIntensity = 1.0f;
            d.specularColor[0] = d.specularColor[1] = d.specularColor[2] = 1.0f;
            d.sheenColor[0] = d.sheenColor[1] = d.sheenColor[2] = 0.0f;
            d.sheenRoughness = 0.0f;
            d.iridescence = 0.0f;             // off by default; lobe is skipped when iridescence == 0
            d.iridescenceIOR = 1.3f;
            d.iridescenceThicknessNm = 400.0f;
            d.dispersion = 0.0f;              // off by default; lobe is skipped when dispersion == 0
            d.thickness = 0.0f;               // 0 = use back-face actual distance for Beer-Lambert (closed-mesh path)
            d.thinWalled = 0;                 // 0 = closed-mesh BSDF; 1 = thin-shell BSDF (set explicitly via MaterialWithThickness::thinWalled)
            d.occlusionTexIndex = -1;
            d.detailTexIndex = -1;
            d.detailRepeat = 0.f;
            d.detailStrength = 0.f;
            d.detailNormalTexIndex = -1;
            d.detailNormalScale = 1.f;
            d.detailRoughStrength = 0.f;
            d._padDetail = 0.f;
            d.terrainWeightTexIndex = -1;
            d.terrainNormalTexIndex = -1;
            d.terrainBandStrength = 0.f;
            d.terrainNormalScale = 1.f;
            d.terrainRoughStrength = 0.f;
            d.terrainHeightBlend = 0.f;
            d._padTerrain[0] = d._padTerrain[1] = 0.f;
            for (int bi = 0; bi < 4; ++bi) {
                d.terrainBandAlbedoTex[bi] = -1;
                d.terrainBandNormalTex[bi] = -1;
                d.terrainBandRepeat[bi] = 1.f;
                d.terrainBandRough[bi] = 0.9f;
            }
            d.translucency = 0.f;             // off by default; deferred sun/ambient translucency term is skipped when 0
            d.translucencyColor[0] = d.translucencyColor[1] = d.translucencyColor[2] = 1.0f;
            static constexpr float kIdent[9] = {1,0,0, 0,1,0, 0,0,1};
            std::copy(kIdent, kIdent+9, d.uvTransform);
            std::copy(kIdent, kIdent+9, d.uvTransformNormal);
            std::copy(kIdent, kIdent+9, d.uvTransformRoughMetal);
            std::copy(kIdent, kIdent+9, d.uvTransformEmissive);
            std::copy(kIdent, kIdent+9, d.uvTransformOcclusion);
            std::copy(kIdent, kIdent+9, d.uvTransformClearcoat);
            std::copy(kIdent, kIdent+9, d.uvTransformClearcoatRough);
            std::copy(kIdent, kIdent+9, d.uvTransformTransmission);
            auto mat = m.material();
            if (!mat) return d;
            d.alphaCutoff = mat->alphaTest;
            if (auto* col = dynamic_cast<MaterialWithColor*>(mat.get())) {
                d.albedo[0] = col->color.r;
                d.albedo[1] = col->color.g;
                d.albedo[2] = col->color.b;
            }
            if (auto* rg = dynamic_cast<MaterialWithRoughness*>(mat.get())) {
                d.roughness = rg->roughness;
            }
            if (auto* mt = dynamic_cast<MaterialWithMetalness*>(mat.get())) {
                d.metalness = mt->metalness;
            }
            if (auto* em = dynamic_cast<MaterialWithEmissive*>(mat.get())) {
                d.emissive[0] = em->emissive.r;
                d.emissive[1] = em->emissive.g;
                d.emissive[2] = em->emissive.b;
                d.emissiveIntensity = em->emissiveIntensity;
            }
            if (auto* nm = dynamic_cast<MaterialWithNormalMap*>(mat.get())) {
                d.normalScale[0] = nm->normalScale.x;
                d.normalScale[1] = nm->normalScale.y;
            }
            if (auto* tr = dynamic_cast<MaterialWithTransmission*>(mat.get())) {
                d.transmission = tr->transmission;
                d.ior          = std::max(1.0f, tr->ior);
                d.dispersion   = std::max(0.0f, tr->dispersion);
                // glTF permits transmission + alphaMode BLEND together; alpha
                // is COVERAGE, independent of the transmission tint, and the
                // spec composite is α·glassResult + (1−α)·background. Smoked
                // car windows ship exactly this pattern (BLACK baseColor as
                // the tint + low alpha): taking the tint alone renders them
                // opaque-black in both render modes, while GL — which ignores
                // the transmission extension and plain alpha-blends — shows
                // the background through. Fold the coverage into the tint,
                // tint' = mix(1, tint, α): for the dominant straight-through
                // path this reproduces the blend composite exactly, with no
                // second blend pass in either pipeline.
                if (d.transmission > 0.0f && mat->transparent && mat->opacity < 1.0f) {
                    const float a = std::clamp(mat->opacity, 0.0f, 1.0f);
                    for (float& c : d.albedo) {
                        c = (1.0f - a) + a * c;
                    }
                }
            }
            // Additive-blend effects (muzzle flashes, sparks, energy glows): the
            // surface GLOWS over the scene rather than occluding/refracting it.
            // Deferred sentinel: transmission > 1 (= 1 + strength) → "add, don't
            // mix". Ray-query-safe: transmission>1 reads as full stochastic
            // pass-through (effectively invisible to the hit test), which is
            // correct since additive blending has no physical analogue.
            if (mat->blending == Blending::Additive && d.transmission == 0.0f) {
                d.transmission = 1.0f + std::clamp(mat->opacity, 0.0f, 1.0f);
                d.ior          = 1.0f;
            }
            // Alpha-blend transparency (transparent=true, opacity<1) has no
            // physical analogue in a ray tracer, so treat it as stochastic pass-through:
            // with probability (1-opacity) the ray continues straight through
            // (ior=1 → refract returns the incident direction unchanged, F=0).
            // Deferred reads ior≈1 as the "clean alpha blend" marker (vs ior>1
            // real refractive glass).
            if (d.transmission == 0.0f && mat->transparent && mat->opacity < 1.0f) {
                d.transmission = 1.0f - mat->opacity;
                d.ior          = 1.0f;
            }
            // BLEND mode with texture alpha (alphaMode=BLEND, opacity=1.0):
            // alphaCutoff=-1.0 sentinel triggers per-texel stochastic blend in
            // the deferred shade's ray-query hit handling using the albedo
            // texture's alpha channel.
            //
            // DECAL refinement (-2.0): transparent + depthWrite=false +
            // polygonOffset is the decal authoring signature (DecalGeometry
            // scorch splats etc.). The gbuffer raster routes these to a
            // dedicated pipeline that alpha-blends ONLY the albedo attachment
            // over the receiving surface (normal/ids/motion/depth untouched) —
            // a deterministic lerp matching GL's forward blend, instead of the
            // stochastic screen-door whose per-frame id flicker defeats the
            // temporal accumulator and lets the denoiser dilate the splat.
            // Every shader-side blend test is a sign test (alphaCutoff < 0),
            // so -2 inherits all -1 semantics (no shadow cast, stochastic
            // pass-through in the ray-query hit handling) automatically.
            if (mat->transparent && d.alphaCutoff == 0.0f && d.transmission == 0.0f) {
                d.alphaCutoff = (!mat->depthWrite && mat->polygonOffset) ? -2.0f : -1.0f;
            }
            if (auto* cc = dynamic_cast<MaterialWithClearcoat*>(mat.get())) {
                d.clearcoat = cc->clearcoat;
                d.clearcoatRoughness = cc->clearcoatRoughness;
            }
            if (auto* att = dynamic_cast<MaterialWithAttenuation*>(mat.get())) {
                d.attenuationColor[0] = att->attenuationColor.r;
                d.attenuationColor[1] = att->attenuationColor.g;
                d.attenuationColor[2] = att->attenuationColor.b;
                d.attenuationDistance = att->attenuationDistance;
            }
            if (auto* th = dynamic_cast<MaterialWithThickness*>(mat.get())) {
                d.thickness  = std::max(0.0f, th->thickness);
                d.thinWalled = th->thinWalled ? 1 : 0;
            }
            if (auto* dm = dynamic_cast<MaterialWithDetailMap*>(mat.get())) {
                // texIndex stays -1 here; the texture-index fill sites bind it
                // (detailTexOf gates on detailMap && strength > 0).
                d.detailRepeat        = dm->detailRepeat;
                d.detailStrength      = std::clamp(dm->detailStrength, 0.f, 1.f);
                d.detailNormalScale   = dm->detailNormalScale;
                d.detailRoughStrength = std::clamp(dm->detailRoughStrength, 0.f, 1.f);
            }
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get())) {
                // Indices stay -1 here; the texture-index fill sites bind them
                // (terrainWeightTexOf gates on the weight map + a band set).
                d.terrainBandStrength  = std::clamp(tm->terrainBandStrength, 0.f, 1.f);
                d.terrainNormalScale   = tm->terrainBandNormalScale;
                d.terrainRoughStrength = std::clamp(tm->terrainBandRoughStrength, 0.f, 1.f);
                d.terrainHeightBlend   = std::max(tm->terrainHeightBlend, 0.f);
                for (int bi = 0; bi < 4; ++bi) {
                    d.terrainBandRepeat[bi] = tm->terrainBandRepeat[static_cast<size_t>(bi)];
                    d.terrainBandRough[bi]  = tm->terrainBandRoughness[static_cast<size_t>(bi)];
                }
            }
            if (auto* sp = dynamic_cast<MaterialWithPbrSpecular*>(mat.get())) {
                d.specularIntensity   = sp->specularIntensity;
                d.specularColor[0]    = sp->specularColor.r;
                d.specularColor[1]    = sp->specularColor.g;
                d.specularColor[2]    = sp->specularColor.b;
            }
            if (auto* tl = dynamic_cast<MaterialWithTranslucency*>(mat.get())) {
                d.translucency         = std::clamp(tl->translucency, 0.f, 1.f);
                d.translucencyColor[0] = tl->translucencyColor.r;
                d.translucencyColor[1] = tl->translucencyColor.g;
                d.translucencyColor[2] = tl->translucencyColor.b;
            }
            if (auto* sh = dynamic_cast<MaterialWithSheen*>(mat.get())) {
                d.sheenColor[0]  = sh->sheenColor.r;
                d.sheenColor[1]  = sh->sheenColor.g;
                d.sheenColor[2]  = sh->sheenColor.b;
                d.sheenRoughness = sh->sheenRoughness;
            }
            if (auto* ir = dynamic_cast<MaterialWithIridescence*>(mat.get())) {
                d.iridescence            = ir->iridescence;
                d.iridescenceIOR         = std::max(1.0f, ir->iridescenceIOR);
                d.iridescenceThicknessNm = std::max(0.0f, ir->iridescenceThicknessNm);
            }
            // sideMode mirrors threepp::Side {Front=0, Back=1, Double=2}.
            // Chit reads it for the wrong-side pass-through gate; the raster
            // gbuffer pass picks BACK / FRONT / NONE cull mode accordingly.
            d.sideMode = static_cast<int32_t>(mat->side);
            // MeshBasicMaterial is unlit: emit base color directly with no
            // PBR shading or bounce. Use roughness < 0 as the shader sentinel
            // (avoids growing the MaterialDesc layout).
            if (dynamic_cast<MeshBasicMaterial*>(mat.get())) {
                d.roughness = -1.0f;
            }
            return d;
        }
}// namespace threepp
