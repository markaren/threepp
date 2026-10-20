// Stable entry slots: a content change edits rows of the entry index space
// instead of rebuilding the scene (plans/gpu-driven-instances.md, stage 2a).
//
// The renderer's scene is a flat entry list, one MeshEntry per mesh instance,
// and the entry index is ONE index space shared by everything that names an
// instance: the TLAS instanceCustomIndex, the geometry and material desc rings,
// the DrawInfo records and the ids AOV, the motion matrices and the moved bits.
// Until 2026-10-19 any change to the list's shape (a mesh added or removed, an
// InstancedMesh count changed, a mesh shown for the first time) re-derived the
// whole list in traversal order behind a vkDeviceWaitIdle: 43-51 ms an event at
// 33k entries (VulkanEntryChurn_test), every phase of it O(scene).
//
// Here the walk's result is laid onto the slots the live entries already hold.
// A span (one mesh's run of entries) that is still in the scene keeps its
// slots. A removed span becomes a FREE span (EntrySpan::isFree: hidden, its
// mesh the tombstone, its rows zeroed) and is parked for a few frames before
// it is handed out again. A new span takes the best-fitting free run or the
// end; an instanced one asks for room to grow. A grown span takes the free run
// after it or moves. Nothing compacts, so the rows of every span that did not
// change are not touched, and no GPU buffer changes handle: they were sized
// for slotCapacity_ entries at the last structural rebuild. The frame then
// goes on as a lean one, whose TLAS fill builds the instance array from the
// spans and records a MODE_BUILD into the frame command buffer.
//
// More entries than the pool holds grow the pool in place first
// (growSlotPool: one device wait, no rebuild).
//
// What it does not take (the caller then rebuilds, exactly as before): a
// deformer (skinned, displaced, grass, tet, morph), a ParticleField or a
// vertex-interop mesh added or removed, an overlay or particle mesh added, a
// mesh that changed class (overlay, deformer, instancing), a geometry whose
// BLAS record cannot be referenced as it is. A geometry, material or texture
// swapped on a mesh that stays IS taken: the mesh is placed again, as a new
// one is, with what it holds now.

#include "VulkanCoreImpl.hpp"

#include "VulkanCpuPhaseProf.hpp"

#include "threepp/core/Assert.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

namespace threepp {

namespace {

    using vulkan::impl::MeshEntry;

    // Facts the fingerprint and the passes read off a mesh's entries that a
    // span keeping its slots must not change.
    bool sameEntryClass(const MeshEntry& a, const MeshEntry& b) {
        return a.isOverlay == b.isOverlay && a.isParticle == b.isParticle && a.isSkinned == b.isSkinned &&
               a.isDisplaced == b.isDisplaced && a.isGrass == b.isGrass && a.isMorphed == b.isMorphed &&
               a.isTet == b.isTet && a.isParticleField == b.isParticleField &&
               a.isVertexInterop == b.isVertexInterop && a.isInstanced == b.isInstanced &&
               a.camAttached == b.camAttached && a.sensorOnly == b.sensorOnly;
    }

    bool isDeformer(const MeshEntry& e) {
        return e.isSkinned || e.isDisplaced || e.isGrass || e.isMorphed || e.isTet;
    }

    constexpr float kIdentity16[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

}// namespace

Mesh& VulkanRenderer::Impl::slotTombstone() {
            if (!slotTombstone_) {
                auto g = BufferGeometry::create();
                g->setAttribute("position", FloatBufferAttribute::create(std::vector<float>{0, 0, 0, 1, 0, 0, 0, 1, 0}, 3));
                g->setAttribute("normal", FloatBufferAttribute::create(std::vector<float>{0, 0, 1, 0, 0, 1, 0, 0, 1}, 3));
                slotTombstone_ = Mesh::create(g, MeshBasicMaterial::create());
                slotTombstone_->updateMatrixWorld();
            }
            return *slotTombstone_;
        }

bool VulkanRenderer::Impl::slotsDisabled() {
            static const bool off = [] {
                const char* e = std::getenv("THREEPP_VULKAN_NO_SLOTS");
                return e && e[0] == '1';
            }();
            return off;
        }

VulkanRenderer::Impl::EntrySpan VulkanRenderer::Impl::slotFreeSpan(uint32_t first, uint32_t count,
                                                                   uint64_t parkedUntil) {
            Mesh& tomb = slotTombstone();
            EntrySpan f{};
            f.mesh = &tomb;
            f.inst = nullptr;
            f.first = first;
            f.count = count;
            f.meshId = tomb.id;
            f.hidden = true;
            f.isFree = true;
            f.parkedUntil = parkedUntil;
            std::memcpy(f.meshWorld.data(), tomb.matrixWorld->elements.data(), 64);
            return f;
        }

VulkanRenderer::Impl::MeshEntry VulkanRenderer::Impl::slotFreeEntry() {
            Mesh& tomb = slotTombstone();
            MeshEntry e{};
            e.mesh = &tomb;
            e.instanceIndex = 0u;
            e.hidden = true;
            e.lodExemptStatic = true;
            e.lodGeomKey = tomb.geometry().get();
            std::memcpy(e.worldMatrix.data(), kIdentity16, 64);
            return e;
        }

uint32_t VulkanRenderer::Impl::slotSlackFor(const EntrySpan& sp, uint32_t count) {
            if (!sp.inst) return 0u;
            const auto* im = sp.inst->instanceMatrix();
            const uint32_t cap = im ? static_cast<uint32_t>(im->count()) : count;
            if (cap <= count) return 0u;
            return std::min(cap - count, std::max(count / 2u, 64u));
        }

uint32_t VulkanRenderer::Impl::slotMaterialAssetIdx(const Material* mat) {
            const unsigned int id = mat ? mat->id : 0u;
            auto [it, inserted] = slotMatAssetIdx_.try_emplace(mat, id, slotNextMatAssetIdx_);
            if (inserted) {
                ++slotNextMatAssetIdx_;
            } else if (it->second.first != id) {
                // A new material at a dead one's address: a new index.
                it->second = {id, slotNextMatAssetIdx_++};
            }
            return it->second.second;
        }

namespace {

    VkCullModeFlags slotCullOf(const threepp::vulkan_pt::MaterialDesc& md) {
        switch (md.sideMode) {
            case 0: return VK_CULL_MODE_BACK_BIT;
            case 1: return VK_CULL_MODE_FRONT_BIT;
            default: return VK_CULL_MODE_NONE;
        }
    }

}// namespace

const VulkanRenderer::Impl::BlasRecord* VulkanRenderer::Impl::slotRecordOf(const MeshEntry& e) {
            // (A field's state only for the geometry it draws now: one left from
            // a swapped geometry waits for its admission.)
            if (e.isGrass) {
                const auto it = grassStates.find(static_cast<GrassMesh*>(e.mesh));
                if (it == grassStates.end() || it->second->liveCheck.lock() != e.mesh->geometry()) return nullptr;
                return it->second->blas.get();
            }
            const auto g = e.mesh->geometry();
            const auto it = g ? blasCache.find(g.get()) : blasCache.end();
            return it != blasCache.end() ? it->second.get() : nullptr;
        }

VulkanRenderer::Impl::MeshFingerprint VulkanRenderer::Impl::slotFingerprintOf(Mesh& m) {
            const auto geom = m.geometry();
            const auto mat = m.material();
            MeshFingerprint fp{};
            fp.mesh = &m;
            fp.geom = geom.get();
            fp.mat = mat.get();
            fp.meshId = m.id;
            fp.geomId = geom->id;
            fp.matId = mat ? mat->id : 0u;
            fp.matVersion = mat ? mat->version() : 0u;
            fp.geomVersion = geomVersionOf(*geom);
            fp.matTyped = mat.get();
            fp.geomTyped = geom.get();
            fp.attrVersion = geom->attributesVersion();
            fp.posAttr = geom->getAttribute("position");
            fp.normAttr = geom->getAttribute("normal");
            fp.uvAttr = geom->getAttribute("uv");
            fp.idxAttr = geom->getIndex();
            fp.colAttr = geom->getAttribute("color");
            fp.albedoTex = albedoTexOf(m);
            fp.roughnessTex = roughnessTexOf(m);
            fp.metalnessTex = metalnessTexOf(m);
            fp.normalTex = normalTexOf(m);
            fp.transmissionTex = transmissionTexOf(m);
            fp.clearcoatTex = clearcoatTexOf(m);
            fp.clearcoatRoughnessTex = clearcoatRoughnessTexOf(m);
            fp.emissiveTex = emissiveTexOf(m);
            fp.occlusionTex = occlusionTexOf(m);
            const MaterialDesc pd = materialFromMesh(m);
            fp.pbr = {pd.albedo[0], pd.albedo[1], pd.albedo[2],
                      pd.roughness, pd.metalness,
                      pd.emissive[0], pd.emissive[1], pd.emissive[2],
                      pd.emissiveIntensity,
                      pd.normalScale[0], pd.normalScale[1],
                      pd.transmission, pd.ior,
                      pd.clearcoat, pd.clearcoatRoughness};
            fp.overlay = false;
            return fp;
        }

void VulkanRenderer::Impl::slotTombRows(MeshEntry& e, MeshFingerprint& fp) {
            Mesh& tomb = slotTombstone();
            e = slotFreeEntry();
            fp = MeshFingerprint{};
            const auto g = tomb.geometry();
            const auto mt = tomb.material();
            fp.mesh = &tomb;
            fp.geom = g.get();
            fp.mat = mt.get();
            fp.meshId = tomb.id;
            fp.geomId = g->id;
            fp.matId = mt->id;
            fp.instanceIndex = 0u;
            fp.matVersion = mt->version();
            fp.geomVersion = geomVersionOf(*g);
            std::memcpy(fp.matrix.data(), kIdentity16, 64);
            fp.matTyped = mt.get();
            fp.geomTyped = g.get();
            fp.attrVersion = g->attributesVersion();
            fp.posAttr = g->getAttribute("position");
            fp.normAttr = g->getAttribute("normal");
        }

void VulkanRenderer::Impl::slotGrowRows(size_t oldSize, size_t newSize, const MeshEntry& tombEntry,
                                        const MeshFingerprint& tombFp) {
            std::array<float, 16> identity{};
            std::memcpy(identity.data(), kIdentity16, 64);
            auto& entries = lastVisibleEntries_;
            if (newSize > oldSize) {
                entries.resize(newSize, tombEntry);
                prevSceneFingerprint.resize(newSize, tombFp);
                geomDescsCached_.resize(newSize);
                matDescsCached_.resize(newSize);
                prevWorldByEntry_.resize(newSize, identity);
                prevWorldValidByEntry_.resize(newSize, 1u);
                motionScratch_.resize(newSize * 16u);
                for (size_t i = oldSize; i < newSize; ++i) std::memcpy(&motionScratch_[i * 16u], kIdentity16, 64);
            }
            // The arrays below are not always entries-sized (an empty sticky
            // after a reset, ids before their first assignment): grow them to
            // the list, never shrink.
            if (entryStableIds_.size() < newSize) entryStableIds_.resize(newSize, 0u);
            if (meshMovedSticky_.size() < newSize) meshMovedSticky_.resize(newSize, 0u);
            if (entryLodFade_.size() < newSize) entryLodFade_.resize(newSize, 0u);
            if (lastVisibleCullMode_.size() < newSize) lastVisibleCullMode_.resize(newSize, VK_CULL_MODE_BACK_BIT);
            if (meshMovedBits_.size() < (newSize + 31u) / 32u) meshMovedBits_.resize((newSize + 31u) / 32u, 0u);
        }

void VulkanRenderer::Impl::slotFillRow(size_t i, const MeshEntry& be, const MeshFingerprint& fp0,
                                       const BlasRecord* rec, const GeometryDesc& gd, const MaterialDesc& md0,
                                       uint32_t stableId) {
            lastVisibleEntries_[i] = be;
            MeshFingerprint& fp = prevSceneFingerprint[i];
            fp = fp0;
            fp.instanceIndex = be.instanceIndex;
            fp.matrix = be.worldMatrix;
            if (rec) {
                geomDescsCached_[i] = gd;
                MaterialDesc md = md0;
                applyInstanceColor(md, be);
                matDescsCached_[i] = md;
                lastVisibleCullMode_[i] = slotCullOf(md);
            } else {
                geomDescsCached_[i] = GeometryDesc{};
                matDescsCached_[i] = MaterialDesc{};
                lastVisibleCullMode_[i] = VK_CULL_MODE_BACK_BIT;
            }
            entryStableIds_[i] = stableId;
            // Prev = current: a new entry has no motion (seedMotionState's rule
            // for the structural rebuild's new entries).
            prevWorldByEntry_[i] = be.worldMatrix;
            prevWorldValidByEntry_[i] = 1u;
            std::memcpy(&motionScratch_[i * 16u], kIdentity16, 64);
            meshMovedSticky_[i] = 0u;
            entryLodFade_[i] = 0u;
        }

void VulkanRenderer::Impl::slotTombRow(size_t i, const MeshEntry& tombEntry, const MeshFingerprint& tombFp) {
            std::array<float, 16> identity{};
            std::memcpy(identity.data(), kIdentity16, 64);
            lastVisibleEntries_[i] = tombEntry;
            prevSceneFingerprint[i] = tombFp;
            geomDescsCached_[i] = GeometryDesc{};
            matDescsCached_[i] = MaterialDesc{};
            entryStableIds_[i] = 0u;
            prevWorldByEntry_[i] = identity;
            prevWorldValidByEntry_[i] = 1u;
            std::memcpy(&motionScratch_[i * 16u], kIdentity16, 64);
            meshMovedSticky_[i] = 0u;
            entryLodFade_[i] = 0u;
            lastVisibleCullMode_[i] = VK_CULL_MODE_BACK_BIT;
        }

bool VulkanRenderer::Impl::trySlotEdit(std::vector<MeshEntry>& built, std::vector<EntrySpan>& builtSpans,
                                       const std::vector<std::pair<ParticleField*, uint32_t>>& builtFields) {
            THREEPP_CPUPROF("scene.3c_slotEdit");
            if (slotsDisabled() || !sceneBuilt_ || slotCapacity_ == 0u) return false;
            // THREEPP_VULKAN_SLOTS_LOG=1: one line per change the slot path
            // hands to the structural rebuild, saying why.
            static const bool logDeclines = [] {
                const char* v = std::getenv("THREEPP_VULKAN_SLOTS_LOG");
                return v && v[0] == '1';
            }();
            const auto decline = [&](const char* why) {
                if (logDeclines) std::fprintf(stderr, "[VulkanRenderer] slot edit declined: %s\n", why);
                return false;
            };
            auto& entries = lastVisibleEntries_;
            const size_t oldSize = entries.size();
            if (prevSceneFingerprint.size() != oldSize || geomDescsCached_.size() != oldSize ||
                matDescsCached_.size() != oldSize || prevWorldByEntry_.size() != oldSize ||
                prevWorldValidByEntry_.size() != oldSize || motionScratch_.size() != oldSize * 16u)
                return decline("entry-indexed arrays out of step");
            // A ParticleField publishes its entry index (particleFields_). One
            // neither added nor removed (both decline below) keeps its span, and
            // the list is re-pointed at the spans' places at the end.
            if (builtFields.size() != particleFields_.size()) return decline("a ParticleField added or removed");
            // What died since the last edit goes first: a streamed world drops a
            // tile and makes the next one at once, and the allocator hands the
            // new geometry the dead one's address. Its record, still keyed there,
            // would read as "a stale record at a new geometry's address" below.
            // Only the meshes this edit removes refer to it, and the retire
            // queue keeps it for the frames in flight.
            pruneFreedAssets();
            const bool rq = ctx->rayQuerySupported();

            const auto texturesChanged = [&](const Mesh& m, const MeshFingerprint& f) {
                return albedoTexOf(m) != f.albedoTex || roughnessTexOf(m) != f.roughnessTex ||
                       metalnessTexOf(m) != f.metalnessTex || normalTexOf(m) != f.normalTex ||
                       transmissionTexOf(m) != f.transmissionTex || clearcoatTexOf(m) != f.clearcoatTex ||
                       clearcoatRoughnessTexOf(m) != f.clearcoatRoughnessTex ||
                       emissiveTexOf(m) != f.emissiveTex || occlusionTexOf(m) != f.occlusionTex;
            };

            THREEPP_CPUPROF_NAMED(seMatch, "scene.S1_slotMatch");
            // ── 1. Match the walk's spans to the live ones, and decide ───────
            // Nothing is edited before every span has been judged: a span the
            // slot path cannot take sends the whole frame to the full rebuild,
            // which reads what the walk built and the old state untouched.
            std::unordered_map<const Mesh*, uint32_t> oldByMesh;
            oldByMesh.reserve(entrySpans_.size());
            for (uint32_t s = 0; s < entrySpans_.size(); ++s) {
                const EntrySpan& sp = entrySpans_[s];
                if (sp.isFree) continue;
                if (!oldByMesh.emplace(sp.mesh, s).second) return decline("one mesh in two spans");
            }
            std::vector<int32_t> match(builtSpans.size(), -1);
            std::vector<uint8_t> kept(entrySpans_.size(), 0u);
            // Geometries a new span draws that have no record yet, with a
            // mesh to take the shared_ptr (the record's liveCheck) from.
            std::vector<std::pair<const BufferGeometry*, Mesh*>> toBuild;
            std::unordered_set<const BufferGeometry*> shownNeeds;// new geometries a shown new span draws
            // New grass fields (a field owns its record), and whether each shows.
            std::vector<std::pair<GrassMesh*, bool>> newGrass;
            for (size_t b = 0; b < builtSpans.size(); ++b) {
                const EntrySpan& bs = builtSpans[b];
                // The walk's own room to grow: the layout below leaves its own.
                if (bs.isFree) continue;
                if (bs.count == 0u) return decline("an empty span");
                const MeshEntry& be = built[bs.first];
                Mesh& m = *bs.mesh;
                const auto it = oldByMesh.find(&m);
                // A span whose hidden admission has not landed and that shows
                // now is placed again below, with the synchronous admission.
                const bool admitOnShow = it != oldByMesh.end() && entrySpans_[it->second].admitPending && !bs.hidden;
                if (it != oldByMesh.end() && entrySpans_[it->second].meshId == bs.meshId && !admitOnShow) {
                    const EntrySpan& os = entrySpans_[it->second];
                    const MeshEntry& oe = entries[os.first];
                    const MeshFingerprint& of = prevSceneFingerprint[os.first];
                    if (!sameEntryClass(be, oe) || (bs.inst == nullptr) != (os.inst == nullptr))
                        return decline("a mesh changed class (overlay, deformer, instancing, layer)");
                    const auto geom = m.geometry();
                    const auto mat = m.material();
                    // A geometry, material or texture swapped on a mesh that
                    // stays (the lean diff aborts to the full path on a texture
                    // swap under a bumped material version): the mesh is placed
                    // again below, as a new mesh is, with what it holds now; its
                    // old slots are freed like a removed mesh's. Its stable id
                    // (the Ids AOV, lidar instance ids) is the mesh's and stays.
                    const bool swapped = !geom || of.geom != geom.get() || of.geomId != geom->id ||
                                         of.mat != mat.get() || of.matId != (mat ? mat->id : 0u) ||
                                         ((mat ? mat->version() : 0u) != of.matVersion && texturesChanged(m, of));
                    if (!swapped) {
                        match[b] = static_cast<int32_t>(it->second);
                        kept[it->second] = 1u;
                        continue;
                    }
                    ++tlasStats_.swapEdits;
                }
                // A new span: a static mesh whose geometry has a record the TLAS
                // can reference as it is, or none yet; or a grass field, whose
                // record is its own (admitGrassState, step 3).
                if (be.isOverlay || be.isParticle || be.isParticleField || be.isVertexInterop ||
                    (isDeformer(be) && !be.isGrass))
                    return decline("a new overlay, particle, deformer or interop mesh");
                const auto geomSp = m.geometry();
                if (!geomSp) return decline("a new mesh without geometry");
                if (be.isGrass) {
                    if (!grassInputsValid(*geomSp)) return decline("a grass field without heightFrac");
                    newGrass.emplace_back(static_cast<GrassMesh*>(&m), !bs.hidden);
                    continue;
                }
                const BufferGeometry* gk = geomSp.get();
                if (const auto rit = blasCache.find(gk); rit != blasCache.end()) {
                    const BlasRecord& rec = *rit->second;
                    const auto live = rec.liveCheck.lock();
                    if (!live || live.get() != gk) return decline("a stale record at a new geometry's address");
                    if (rec.interop || rec.vertex.handle == VK_NULL_HANDLE) return decline("an interop or empty record");
                    if (rec.geomVersion != geomVersionOf(*gk)) return decline("a new mesh's geometry was edited");
                    if (rec.packedMask != 0u && forceUnpackedGeoms_.count(gk) != 0u)
                        return decline("a packed record marked for unpacking");
                    if (rq && rec.as == VK_NULL_HANDLE) return decline("a record built for overlays only");
                } else {
                    if (std::none_of(toBuild.begin(), toBuild.end(), [&](const auto& p) { return p.first == gk; }))
                        toBuild.emplace_back(gk, &m);
                    if (!bs.hidden) shownNeeds.insert(gk);
                }
            }
            // A geometry only hidden new spans draw is admitted over the next
            // frames (pumpHiddenAdmissions): no record now, no texture upload.
            std::unordered_set<const BufferGeometry*> deferredGeoms;
            // So is a hidden new grass field without a state (a cached field
            // coming back keeps its own).
            std::unordered_set<const Mesh*> deferredGrass;
            if (!syncAdmission()) {
                for (auto t = toBuild.begin(); t != toBuild.end();) {
                    if (shownNeeds.count(t->first) == 0u) {
                        deferredGeoms.insert(t->first);
                        t = toBuild.erase(t);
                    } else {
                        ++t;
                    }
                }
                for (const auto& [gm, shown] : newGrass) {
                    if (shown) continue;
                    const auto it = grassStates.find(gm);
                    if (it == grassStates.end() || it->second->liveCheck.lock() != gm->geometry())
                        deferredGrass.insert(gm);
                }
            }
            for (uint32_t s = 0; s < entrySpans_.size(); ++s) {
                const EntrySpan& sp = entrySpans_[s];
                if (sp.isFree || kept[s]) continue;
                // A skinned, displaced, morphed or tet mesh's state owns
                // descriptor sets and images the retire queue does not model; a
                // ParticleField its published index. A grass field's state stays
                // with its mesh (a cached field comes back as a show) and is
                // retired when the mesh or its geometry dies (pruneFreedAssets).
                const MeshEntry& e0 = entries[sp.first];
                if ((isDeformer(e0) && !e0.isGrass) || e0.isParticleField || e0.isVertexInterop)
                    return decline("a deformer, ParticleField or interop mesh removed");
            }

            seMatch.stop();
            THREEPP_CPUPROF_NAMED(sePlan, "scene.S2_slotPlan");
            // ── 2. Plan the layout on a copy of the spans ──────────────────────
            const uint64_t serial = frameSerial_;
            const uint64_t parkUntil = serial + kFramesInFlight + 1u;
            std::vector<EntrySpan> spans;
            spans.reserve(entrySpans_.size() + 2u * builtSpans.size() + 4u);
            struct RowCopy { uint32_t src, dst, count; };
            // A fill takes its entries from the walk's rows, or, for a span the
            // walk did not expand (EntrySpan::rowsDeferred), derives them from
            // instance `instanceFirst` on, as the walk would have.
            struct RowFill {
                uint32_t dst, builtFirst, count;
                size_t span;
                size_t builtSpan;
                bool derive = false;
                uint32_t instanceFirst = 0;
            };
            struct RowRange { uint32_t first, count; };
            std::vector<RowCopy> copies;
            std::vector<RowFill> fills;
            std::vector<RowRange> tombs;
            std::vector<size_t> hiddenFlips;// spans indices whose hidden flag flips
            uint32_t highWater = static_cast<uint32_t>(oldSize);
            bool edited = false;

            const auto addFree = [&](uint32_t first, uint32_t count, uint64_t until) {
                if (count == 0u) return;
                spans.push_back(slotFreeSpan(first, count, until));
            };
            // Best fit over the unparked free runs, else the end. Carved off a
            // run's BACK: its front stays where it was, right after whatever
            // span may grow into it (an instanced span's room to grow is the
            // free run after it, and new meshes must not eat it from the front).
            const auto allocate = [&](uint32_t need) -> uint32_t {
                size_t best = spans.size();
                for (size_t k = 0; k < spans.size(); ++k) {
                    const EntrySpan& f = spans[k];
                    if (!f.isFree || f.parkedUntil > serial || f.count < need) continue;
                    if (best == spans.size() || f.count < spans[best].count) best = k;
                }
                if (best == spans.size()) {
                    const uint32_t first = highWater;
                    highWater += need;
                    return first;
                }
                EntrySpan& f = spans[best];
                f.count -= need;// a run used up stays as count 0 and is dropped below
                return f.first + f.count;
            };

            std::vector<int32_t> oldToNew(entrySpans_.size(), -1);
            for (uint32_t s = 0; s < entrySpans_.size(); ++s) {
                const EntrySpan& sp = entrySpans_[s];
                if (sp.isFree) {
                    spans.push_back(sp);
                    continue;
                }
                if (!kept[s]) {
                    addFree(sp.first, sp.count, parkUntil);
                    tombs.push_back({sp.first, sp.count});
                    edited = true;
                    continue;
                }
                oldToNew[s] = static_cast<int32_t>(spans.size());
                spans.push_back(sp);
            }
            // spans index of each built span, for the snapshot remap.
            std::vector<size_t> builtToSpan(builtSpans.size(), 0u);
            for (size_t b = 0; b < builtSpans.size(); ++b) {
                if (match[b] < 0) continue;
                const EntrySpan& bs = builtSpans[b];
                const size_t idx = static_cast<size_t>(oldToNew[static_cast<size_t>(match[b])]);
                builtToSpan[b] = idx;
                const uint32_t oldCount = spans[idx].count;
                const uint32_t newCount = bs.count;
                if (newCount < oldCount) {
                    const uint32_t tail = spans[idx].first + newCount;
                    spans[idx].count = newCount;
                    addFree(tail, oldCount - newCount, parkUntil);
                    tombs.push_back({tail, oldCount - newCount});
                } else if (newCount > oldCount) {
                    const uint32_t extra = newCount - oldCount;
                    const uint32_t end = spans[idx].first + oldCount;
                    size_t after = spans.size();
                    for (size_t k = 0; k < spans.size(); ++k) {
                        if (spans[k].isFree && spans[k].first == end && spans[k].count >= extra &&
                            spans[k].parkedUntil <= serial) {
                            after = k;
                            break;
                        }
                    }
                    if (after != spans.size()) {
                        spans[after].first += extra;
                        spans[after].count -= extra;
                    } else if (end == highWater) {
                        highWater += extra;
                    } else {
                        // Moves: the rows it has go with it, prev-world and
                        // sticky included, and its old run is freed.
                        const uint32_t need = newCount + slotSlackFor(bs, newCount);
                        const uint32_t dst = allocate(need);
                        copies.push_back({spans[idx].first, dst, oldCount});
                        addFree(spans[idx].first, oldCount, parkUntil);
                        tombs.push_back({spans[idx].first, oldCount});
                        addFree(dst + newCount, need - newCount, 0u);
                        spans[idx].first = dst;
                    }
                    RowFill fl{spans[idx].first + oldCount, bs.first + oldCount, extra, idx, b};
                    if (bs.rowsDeferred) {
                        fl.builtFirst = bs.first;
                        fl.derive = true;
                        fl.instanceFirst = oldCount;
                    }
                    fills.push_back(fl);
                    spans[idx].count = newCount;
                }
                if (newCount != oldCount) {
                    spans[idx].aabbValid = false;
                    // The probe-fit chunks before the one the change touches stand.
                    spans[idx].probeChunksValid = std::min(spans[idx].probeChunksValid,
                                                           std::min(oldCount, newCount) / EntrySpan::kProbeChunk);
                    spans[idx].fadeKnown = false;
                    edited = true;
                }
                if (bs.hidden != spans[idx].hidden) hiddenFlips.push_back(idx);
            }
            std::vector<Mesh*> admitNow;// the new spans placed admitPending
            for (size_t b = 0; b < builtSpans.size(); ++b) {
                if (match[b] >= 0 || builtSpans[b].isFree) continue;
                const EntrySpan& bs = builtSpans[b];
                const uint32_t slack = slotSlackFor(bs, bs.count);
                const uint32_t dst = allocate(bs.count + slack);
                EntrySpan ns = bs;
                ns.first = dst;
                ns.isFree = false;
                ns.parkedUntil = 0u;
                ns.fadeKnown = false;
                ns.admitPending = (!deferredGeoms.empty() && deferredGeoms.count(bs.mesh->geometry().get()) != 0u) ||
                                  deferredGrass.count(bs.mesh) != 0u;
                if (ns.admitPending) admitNow.push_back(bs.mesh);
                if (ns.inst) {
                    ns.instColorAttr    = ns.inst->instanceColor();
                    ns.instColorVersion = ns.instColorAttr ? ns.instColorAttr->version : ~0u;
                }
                builtToSpan[b] = spans.size();
                spans.push_back(ns);
                addFree(dst + bs.count, slack, 0u);
                if (bs.rowsDeferred) return decline("an unexpanded span with no slots");
                fills.push_back({dst, bs.first, bs.count, builtToSpan[b], b});
                edited = true;
            }
            // Every buffer the rows land in holds the pool (and a reservation
            // made since the last build is taken here).
            if (highWater > slotCapacity_ || reservedSlots_ > slotCapacity_)
                growSlotPool(std::max<uint32_t>({1024u, highWater > slotCapacity_ ? highWater + highWater / 2u : 0u,
                                                 reservedSlots_, slotCapacity_}));
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                if (highWater > motionMatBufferCapacity[f] ||
                    (highWater + 31u) / 32u > meshMovedBitsBufferCapacity[f])
                    return decline("a motion buffer smaller than the pool");
            }

            sePlan.stop();
            THREEPP_CPUPROF_NAMED(seAssets, "scene.S3_slotAssets");
            // ── 3. New geometries' records, and the new spans' maps ───────────
            // One batch: the BLAS builds and the texture uploads the material
            // descs below admit (a queue wait for that work, not a drain, and
            // none when the new content shares its assets).
            beginOneShotBatch();
            THREEPP_CPUPROF_NAMED(seBlas, "scene.S3a_slotBlas");
            for (const auto& [gk, mesh] : toBuild) {
                const auto geomSp = mesh->geometry();
                auto rec = buildBlasFor(*geomSp, /*allowPacked=*/true, /*updatable=*/false, nullptr,
                                        VertexWriter::Host, /*traced=*/rq);
                if (!rec) continue;// degenerate: its entries get default rows and no instance, as in a rebuild
                rec->liveCheck = geomSp;
                blasCache.emplace(gk, std::move(rec));
            }
            // A field's MeshRepr proxy is nobody's Mesh geometry: the rebuild
            // admits it per field, and so does this (a lookup when it has one).
            for (const auto& fe : builtFields) ensureCachedBlas(fe.first->meshRepr().geometry);
            // New grass fields: their records and inputs in this batch, their
            // first wind pose in the frame (no blocking prime).
            for (const auto& [gm, shown] : newGrass) {
                if (deferredGrass.count(gm) == 0u) admitGrassState(*gm);
                grassPlacedThisFrame_.push_back(gm);
            }
            seBlas.stop();
            for (Mesh* m : admitNow) beginHiddenAdmission(*m);
            admitPendingSpans_ += static_cast<uint32_t>(admitNow.size());
            tlasStats_.admissionsDeferred += admitNow.size();
            // The material descs, whose first sight of a texture uploads it.
            THREEPP_CPUPROF_NAMED(seMats, "scene.S3b_slotMaterials");
            // Every new texture this edit admits is prepared (its CPU half:
            // RGBA8, mips, BC7) on worker threads at once, and the descs below
            // collect them: the frame pays the longest encode, not their sum.
            // A span's record: its geometry's in blasCache, or a grass field's own.
            // (A field's state only for the geometry it draws now: one left from
            // a swapped geometry waits for its admission.)
            const auto recordOf = [&](const MeshEntry& e) { return slotRecordOf(e); };
            if (!syncAdmission()) {
                for (const RowFill& fl : fills) {
                    const MeshEntry& e0 = built[builtSpans[fl.builtSpan].first];
                    if (recordOf(e0)) prepareTexturesAsync(*e0.mesh);
                }
            }
            const size_t texCount0 = materialTextures.size();
            const size_t texFree0 = freeTextureSlots.size();
            // Per mesh, once: the rows every entry of a fill shares.
            struct MeshRows {
                const MeshEntry* proto = nullptr;
                const BlasRecord* rec = nullptr;
                MaterialDesc md{};
                MeshFingerprint fp{};
                uint32_t stableId = 0;
            };
            std::unordered_map<const Mesh*, MeshRows> meshRows;
            for (const RowFill& fl : fills) {
                const MeshEntry& proto = built[builtSpans[fl.builtSpan].first];
                Mesh& m = *proto.mesh;
                if (meshRows.count(&m)) continue;
                MeshRows r;
                r.proto = &proto;
                const auto geom = m.geometry();
                const auto mat = m.material();
                if (recordOf(proto)) {
                    r.md = buildMaterialDesc(m);
                    r.md.materialAssetIdx = slotMaterialAssetIdx(mat.get());
                }
                // The fingerprint the full path's slow branch derives.
                r.fp = slotFingerprintOf(m);
                r.stableId = stableIdIfAssigned(m);
                meshRows.emplace(&m, std::move(r));
            }
            seMats.stop();
            THREEPP_CPUPROF_NAMED(seSubmit, "scene.S3c_slotSubmit");
            // No host wait: the frame recorded next is submitted after this
            // batch on the same queue, and its graph's entry barrier orders it.
            // Under the async-compute pilot the TLAS builds on another queue,
            // outside that order: there the batch is waited for, as before.
            if (asyncPilot_) {
                flushOneShotBatch();
                std::vector<VkAccelerationStructureInstanceKHR> none;
                applyBlasCompactionRemap(none);// compacted records were in no TLAS yet
            } else {
                flushOneShotBatchAsync();
            }
            seSubmit.stop();
            const bool texturesAdded = materialTextures.size() != texCount0 || freeTextureSlots.size() != texFree0;
            for (auto& [mesh, r] : meshRows) r.rec = recordOf(*r.proto);

            seAssets.stop();
            THREEPP_CPUPROF_NAMED(seRows, "scene.S4_slotRows");
            // ── 4. The rows ────────────────────────────────────────────────────
            const uint32_t newSize = highWater;
            MeshEntry tombEntry;
            MeshFingerprint tombFp;
            slotTombRows(tombEntry, tombFp);
            slotGrowRows(oldSize, newSize, tombEntry, tombFp);
            const auto setMoved = [&](uint32_t first, uint32_t count) {
                for (uint32_t i = first; i < first + count; ++i) meshMovedBits_[i >> 5] |= (1u << (i & 31u));
            };
            // Copies first: their sources are tombstoned below, their
            // destinations are free space no fill or tombstone touches.
            for (const RowCopy& c : copies) {
                for (uint32_t j = 0; j < c.count; ++j) {
                    const size_t s = size_t(c.src) + j, d = size_t(c.dst) + j;
                    entries[d] = entries[s];
                    prevSceneFingerprint[d] = prevSceneFingerprint[s];
                    geomDescsCached_[d] = geomDescsCached_[s];
                    matDescsCached_[d] = matDescsCached_[s];
                    entryStableIds_[d] = entryStableIds_[s];
                    prevWorldByEntry_[d] = prevWorldByEntry_[s];
                    prevWorldValidByEntry_[d] = prevWorldValidByEntry_[s];
                    std::memcpy(&motionScratch_[d * 16u], &motionScratch_[s * 16u], 64);
                    meshMovedSticky_[d] = meshMovedSticky_[s];
                    entryLodFade_[d] = entryLodFade_[s];
                    lastVisibleCullMode_[d] = lastVisibleCullMode_[s];
                }
                // No moved bits, here or for the fills below: the reproject
                // guards key on the stable id, not the entry index, and the
                // structural rebuild sets none for a renumbered or new entry
                // either. A moved bit would make it sticky-moving (a shorter
                // history) for thirty frames.
                markGeomDescsDirty(c.dst, c.count);
                markMatDescsDirty(c.dst, c.count);
            }
            for (const RowFill& fl : fills) {
                const EntrySpan& bsp = builtSpans[fl.builtSpan];
                const MeshEntry& proto = built[bsp.first];
                const MeshRows& r = meshRows.at(proto.mesh);
                const LodGeomSel lodSel0 = r.rec ? selectLodGeom(*r.rec, 0) : LodGeomSel{};
                const GeometryDesc gd = r.rec ? buildGeometryDesc(proto, *r.rec, lodSel0) : GeometryDesc{};
                // The walk's multiply, operand for operand (bit-identical rows).
                const std::vector<float>* instArr = (fl.derive && bsp.inst) ? &bsp.inst->instanceMatrix()->array() : nullptr;
                const Matrix4& meshWorld = *proto.mesh->matrixWorld;
                Matrix4 instMat;
                Matrix4 world;
                MeshEntry derived = proto;
                for (uint32_t j = 0; j < fl.count; ++j) {
                    const size_t i = size_t(fl.dst) + j;
                    if (instArr) {
                        const uint32_t k = fl.instanceFirst + j;
                        std::memcpy(instMat.elements.data(), instArr->data() + size_t(k) * 16u, 64);
                        world.multiplyMatrices(meshWorld, instMat);
                        derived.instanceIndex = k;
                        std::memcpy(derived.worldMatrix.data(), world.elements.data(), 64);
                    }
                    const MeshEntry& be = instArr ? derived : built[size_t(fl.builtFirst) + j];
                    slotFillRow(i, be, r.fp, r.rec, gd, r.md, r.stableId);
                }
                markGeomDescsDirty(fl.dst, fl.count);
                markMatDescsDirty(fl.dst, fl.count);
            }
            for (const RowRange& t : tombs) {
                for (uint32_t j = 0; j < t.count; ++j) slotTombRow(size_t(t.first) + j, tombEntry, tombFp);
                markGeomDescsDirty(t.first, t.count);
                markMatDescsDirty(t.first, t.count);
            }
            for (const size_t idx : hiddenFlips) {
                EntrySpan& sp = spans[idx];
                sp.hidden = !sp.hidden;
                for (uint32_t i = sp.first; i < sp.first + sp.count; ++i) entries[i].hidden = sp.hidden;
                setMoved(sp.first, sp.count);
                hiddenFlippedThisFrame_ = true;
                if (!sp.hidden) probeGridDirty_ = true;
            }

            seRows.stop();
            THREEPP_CPUPROF_NAMED(seSpans, "scene.S5_slotSpans");
            // ── 5. The span list: sorted by first, free runs merged ────────────
            std::vector<size_t> order(spans.size());
            for (size_t k = 0; k < order.size(); ++k) order[k] = k;
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b) { return spans[a].first < spans[b].first; });
            std::vector<EntrySpan> sorted;
            sorted.reserve(spans.size());
            std::vector<int32_t> finalIndex(spans.size(), -1);
            for (const size_t k : order) {
                EntrySpan& sp = spans[k];
                if (sp.count == 0u) continue;
                if (sp.isFree && !sorted.empty() && sorted.back().isFree &&
                    sorted.back().first + sorted.back().count == sp.first) {
                    sorted.back().count += sp.count;
                    sorted.back().parkedUntil = std::max(sorted.back().parkedUntil, sp.parkedUntil);
                    continue;
                }
                finalIndex[k] = static_cast<int32_t>(sorted.size());
                sorted.push_back(std::move(sp));
            }
#ifndef NDEBUG
            {
                uint32_t next = 0;
                for (const auto& sp : sorted) {
                    THREEPP_ASSERT(sp.first == next);
                    next += sp.count;
                }
                THREEPP_ASSERT(next == newSize);
            }
#endif
            entrySpans_ = std::move(sorted);
            ++spansVersion_;
            // The fields at their spans' places, in the walk's order (the order
            // the rebuild gives them).
            if (!builtFields.empty()) {
                std::vector<std::pair<ParticleField*, uint32_t>> fields;
                fields.reserve(builtFields.size());
                for (const auto& fe : builtFields) {
                    const Mesh* fm = fe.first;
                    for (const EntrySpan& sp : entrySpans_) {
                        if (sp.isFree || sp.mesh != fm) continue;
                        fields.emplace_back(fe.first, sp.first);
                        break;
                    }
                }
                particleFields_ = std::move(fields);
            }
            // The walk recorded indices into builtSpans; point them at the list.
            for (auto& sn : sceneSnapshot_) {
                if (sn.spanIndex < 0) continue;
                const auto b = static_cast<size_t>(sn.spanIndex);
                sn.spanIndex = b < builtToSpan.size() ? finalIndex[builtToSpan[b]] : -1;
            }

            stickyActiveCount_ = 0;
            for (const uint32_t v : meshMovedSticky_)
                if (v > 0u) ++stickyActiveCount_;
            // The motion rows this edit wrote, and the slots it appended.
            for (const RowCopy& c : copies) markMotionDirty(c.dst, c.count);
            for (const RowFill& fl : fills) markMotionDirty(fl.dst, fl.count);
            for (const RowRange& t : tombs) markMotionDirty(t.first, t.count);
            if (newSize > oldSize) markMotionDirty(static_cast<uint32_t>(oldSize), newSize - static_cast<uint32_t>(oldSize));
            ++motionScratchVersion_;
            ++lodFadeVersion_;// rows written, the high-water mark moved
            markLodFadeWhole();
            cachedEmissiveEntryCount_ = static_cast<size_t>(-1);
            ++drawInputsVersion_;
            if (texturesAdded) {
                // A map new to the bindless array: published through the same
                // per-FIF deferred rewrite an in-place texture refresh uses.
                deferredDescDirty_.fill(true);
                for (auto& v : views_) {
                    v->rasterMatTexValid_.fill(0);
                    v->shadowMatTexValid_.fill(0);
                }
            }
            // The geometries and textures the removed meshes were the last to
            // hold have died by now (or die before a later edit sees them).
            pruneFreedAssets();
            stableIdsPending_ = true;
            slotsEditedThisFrame_ = true;
            if (edited || !hiddenFlips.empty()) ++tlasStats_.slotEdits;
            return true;
        }

bool VulkanRenderer::Impl::deltaDisabled() {
            static const bool off = [] {
                const char* e = std::getenv("THREEPP_VULKAN_NO_DELTA");
                return e && e[0] == '1';
            }();
            return off;
        }

bool VulkanRenderer::Impl::trySlotDelta(SceneDelta& d) {
            THREEPP_CPUPROF("scene.3c_slotDelta");
            if (slotsDisabled() || deltaDisabled() || !sceneBuilt_ || slotCapacity_ == 0u) return false;
            static const bool logDeclines = [] {
                const char* v = std::getenv("THREEPP_VULKAN_SLOTS_LOG");
                return v && v[0] == '1';
            }();
            const auto decline = [&](const char* why) {
                if (logDeclines) std::fprintf(stderr, "[VulkanRenderer] slot delta declined: %s\n", why);
                return false;
            };
            auto& entries = lastVisibleEntries_;
            const size_t oldSize = entries.size();
            if (prevSceneFingerprint.size() != oldSize || geomDescsCached_.size() != oldSize ||
                matDescsCached_.size() != oldSize || prevWorldByEntry_.size() != oldSize ||
                prevWorldValidByEntry_.size() != oldSize || motionScratch_.size() != oldSize * 16u)
                return decline("entry-indexed arrays out of step");
            // What died since the last edit goes first (trySlotEdit's reason).
            pruneFreedAssets();
            const bool rq = ctx->rayQuerySupported();

            THREEPP_CPUPROF_NAMED(dDecide, "scene.D1_deltaDecide");
            // ── 1. Decide, before anything is edited ───────────────────────────
            std::vector<uint8_t> removedMark(entrySpans_.size(), 0u);
            for (const uint32_t r : d.removed) {
                if (r >= entrySpans_.size() || entrySpans_[r].isFree || removedMark[r]) return decline("a removed span not live");
                removedMark[r] = 1u;
                // trySlotEdit's rule: a skinned, displaced, morphed or tet mesh's
                // state owns what the retire queue does not model.
                const MeshEntry& e0 = entries[entrySpans_[r].first];
                if ((isDeformer(e0) && !e0.isGrass) || e0.isParticleField || e0.isVertexInterop)
                    return decline("a deformer, ParticleField or interop mesh removed");
            }
            std::vector<std::pair<const BufferGeometry*, Mesh*>> toBuild;
            std::unordered_set<const BufferGeometry*> shownNeeds;
            std::vector<std::pair<GrassMesh*, bool>> newGrass;
            for (const EntrySpan& bs : d.spans) {
                if (bs.isFree || bs.count == 0u || bs.rowsDeferred) return decline("an empty or unexpanded new span");
                const MeshEntry& be = d.built[bs.first];
                Mesh& m = *bs.mesh;
                if (be.isOverlay || be.isParticle || be.isParticleField || be.isVertexInterop ||
                    (isDeformer(be) && !be.isGrass))
                    return decline("a new overlay, particle, deformer or interop mesh");
                const auto geomSp = m.geometry();
                if (!geomSp) return decline("a new mesh without geometry");
                if (be.isGrass) {
                    if (!grassInputsValid(*geomSp)) return decline("a grass field without heightFrac");
                    newGrass.emplace_back(static_cast<GrassMesh*>(&m), !bs.hidden);
                    continue;
                }
                const BufferGeometry* gk = geomSp.get();
                if (const auto rit = blasCache.find(gk); rit != blasCache.end()) {
                    const BlasRecord& rec = *rit->second;
                    const auto live = rec.liveCheck.lock();
                    if (!live || live.get() != gk) return decline("a stale record at a new geometry's address");
                    if (rec.interop || rec.vertex.handle == VK_NULL_HANDLE) return decline("an interop or empty record");
                    if (rec.geomVersion != geomVersionOf(*gk)) return decline("a new mesh's geometry was edited");
                    if (rec.packedMask != 0u && forceUnpackedGeoms_.count(gk) != 0u)
                        return decline("a packed record marked for unpacking");
                    if (rq && rec.as == VK_NULL_HANDLE) return decline("a record built for overlays only");
                } else {
                    if (std::none_of(toBuild.begin(), toBuild.end(), [&](const auto& p) { return p.first == gk; }))
                        toBuild.emplace_back(gk, &m);
                    if (!bs.hidden) shownNeeds.insert(gk);
                }
            }
            // Hidden new content is admitted over the frames after (trySlotEdit).
            std::unordered_set<const BufferGeometry*> deferredGeoms;
            std::unordered_set<const Mesh*> deferredGrass;
            if (!syncAdmission()) {
                for (auto t = toBuild.begin(); t != toBuild.end();) {
                    if (shownNeeds.count(t->first) == 0u) {
                        deferredGeoms.insert(t->first);
                        t = toBuild.erase(t);
                    } else {
                        ++t;
                    }
                }
                for (const auto& [gm, shown] : newGrass) {
                    if (shown) continue;
                    const auto it = grassStates.find(gm);
                    if (it == grassStates.end() || it->second->liveCheck.lock() != gm->geometry())
                        deferredGrass.insert(gm);
                }
            }

            dDecide.stop();
            THREEPP_CPUPROF_NAMED(dPlace, "scene.D2_deltaPlace");
            // ── 2. Place the new spans ─────────────────────────────────────────
            // Into the free runs the frames in flight no longer read (best fit,
            // carved off a run's back, as trySlotEdit does), else at the end.
            // A span the edit removes is freed in place, parked for the frames
            // in flight.
            const uint64_t serial = frameSerial_;
            const uint64_t parkUntil = serial + kFramesInFlight + 1u;
            struct Run {
                uint32_t span, first, count;
            };
            std::vector<Run> runs;
            for (uint32_t s = 0; s < entrySpans_.size(); ++s) {
                const EntrySpan& sp = entrySpans_[s];
                if (sp.isFree && sp.parkedUntil <= serial && sp.count > 0u) runs.push_back({s, sp.first, sp.count});
            }
            struct Place {
                uint32_t dst, slack;
                int32_t run;// index into runs, -1: at the end
            };
            std::vector<Place> places(d.spans.size());
            uint32_t highWater = static_cast<uint32_t>(oldSize);
            for (size_t b = 0; b < d.spans.size(); ++b) {
                const EntrySpan& bs = d.spans[b];
                const uint32_t slack = slotSlackFor(bs, bs.count);
                const uint32_t need = bs.count + slack;
                int32_t best = -1;
                for (int32_t k = 0; k < static_cast<int32_t>(runs.size()); ++k)
                    if (runs[k].count >= need && (best < 0 || runs[k].count < runs[best].count)) best = k;
                if (best >= 0) {
                    runs[best].count -= need;
                    places[b] = {runs[best].first + runs[best].count, slack, best};
                } else {
                    places[b] = {highWater, slack, -1};
                    highWater += need;
                }
            }
            if (highWater > slotCapacity_ || reservedSlots_ > slotCapacity_)
                growSlotPool(std::max<uint32_t>({1024u, highWater > slotCapacity_ ? highWater + highWater / 2u : 0u,
                                                 reservedSlots_, slotCapacity_}));
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                if (highWater > motionMatBufferCapacity[f] ||
                    (highWater + 31u) / 32u > meshMovedBitsBufferCapacity[f])
                    return decline("a motion buffer smaller than the pool");
            }

            dPlace.stop();
            THREEPP_CPUPROF_NAMED(dAssets, "scene.D3_deltaAssets");
            // ── 3. The new spans' records and maps (trySlotEdit's step 3) ──────
            beginOneShotBatch();
            for (const auto& [gk, mesh] : toBuild) {
                const auto geomSp = mesh->geometry();
                auto rec = buildBlasFor(*geomSp, /*allowPacked=*/true, /*updatable=*/false, nullptr,
                                        VertexWriter::Host, /*traced=*/rq);
                if (!rec) continue;// degenerate: default rows and no instance, as in a rebuild
                rec->liveCheck = geomSp;
                blasCache.emplace(gk, std::move(rec));
            }
            for (const auto& [gm, shown] : newGrass) {
                if (deferredGrass.count(gm) == 0u) admitGrassState(*gm);
                grassPlacedThisFrame_.push_back(gm);
            }
            std::vector<uint8_t> pending(d.spans.size(), 0u);
            std::vector<Mesh*> admitNow;
            for (size_t b = 0; b < d.spans.size(); ++b) {
                Mesh* m = d.spans[b].mesh;
                pending[b] = (!deferredGeoms.empty() && deferredGeoms.count(m->geometry().get()) != 0u) ||
                             deferredGrass.count(m) != 0u;
                if (pending[b]) admitNow.push_back(m);
            }
            for (Mesh* m : admitNow) beginHiddenAdmission(*m);
            admitPendingSpans_ += static_cast<uint32_t>(admitNow.size());
            tlasStats_.admissionsDeferred += admitNow.size();
            if (!syncAdmission()) {
                for (const EntrySpan& bs : d.spans) {
                    const MeshEntry& e0 = d.built[bs.first];
                    if (slotRecordOf(e0)) prepareTexturesAsync(*e0.mesh);
                }
            }
            const size_t texCount0 = materialTextures.size();
            const size_t texFree0 = freeTextureSlots.size();
            struct MeshRows {
                MaterialDesc md{};
                MeshFingerprint fp{};
                uint32_t stableId = 0;
            };
            std::vector<MeshRows> rows(d.spans.size());
            for (size_t b = 0; b < d.spans.size(); ++b) {
                const MeshEntry& proto = d.built[d.spans[b].first];
                Mesh& m = *proto.mesh;
                if (slotRecordOf(proto)) {
                    rows[b].md = buildMaterialDesc(m);
                    rows[b].md.materialAssetIdx = slotMaterialAssetIdx(m.material().get());
                }
                rows[b].fp = slotFingerprintOf(m);
                rows[b].stableId = stableIdIfAssigned(m);
            }
            if (asyncPilot_) {
                flushOneShotBatch();
                std::vector<VkAccelerationStructureInstanceKHR> none;
                applyBlasCompactionRemap(none);
            } else {
                flushOneShotBatchAsync();
            }
            const bool texturesAdded = materialTextures.size() != texCount0 || freeTextureSlots.size() != texFree0;

            dAssets.stop();
            THREEPP_CPUPROF_NAMED(dRows, "scene.D4_deltaRows");
            // ── 4. The rows ────────────────────────────────────────────────────
            const uint32_t newSize = highWater;
            MeshEntry tombEntry;
            MeshFingerprint tombFp;
            slotTombRows(tombEntry, tombFp);
            slotGrowRows(oldSize, newSize, tombEntry, tombFp);
            for (size_t b = 0; b < d.spans.size(); ++b) {
                const EntrySpan& bs = d.spans[b];
                const MeshEntry& proto = d.built[bs.first];
                const BlasRecord* rec = slotRecordOf(proto);
                const LodGeomSel lodSel0 = rec ? selectLodGeom(*rec, 0) : LodGeomSel{};
                const GeometryDesc gd = rec ? buildGeometryDesc(proto, *rec, lodSel0) : GeometryDesc{};
                for (uint32_t j = 0; j < bs.count; ++j)
                    slotFillRow(size_t(places[b].dst) + j, d.built[size_t(bs.first) + j], rows[b].fp, rec, gd,
                                rows[b].md, rows[b].stableId);
                markGeomDescsDirty(places[b].dst, bs.count);
                markMatDescsDirty(places[b].dst, bs.count);
                markMotionDirty(places[b].dst, bs.count);
                markLodFadeDirty(places[b].dst, bs.count);
            }
            for (const uint32_t r : d.removed) {
                const EntrySpan& sp = entrySpans_[r];
                for (uint32_t j = 0; j < sp.count; ++j) {
                    const size_t i = size_t(sp.first) + j;
                    if (meshMovedSticky_[i] > 0u && stickyActiveCount_ > 0u) --stickyActiveCount_;
                    slotTombRow(i, tombEntry, tombFp);
                }
                markGeomDescsDirty(sp.first, sp.count);
                markMatDescsDirty(sp.first, sp.count);
                markMotionDirty(sp.first, sp.count);
                markLodFadeDirty(sp.first, sp.count);
            }
            if (newSize > oldSize) {
                markMotionDirty(static_cast<uint32_t>(oldSize), newSize - static_cast<uint32_t>(oldSize));
                markLodFadeDirty(static_cast<uint32_t>(oldSize), newSize - static_cast<uint32_t>(oldSize));
            }

            dRows.stop();
            THREEPP_CPUPROF_NAMED(dSpans, "scene.D5_deltaSpans");
            // ── 5. The span list, one pass ─────────────────────────────────────
            // Kept spans move over as they are; a removed one becomes a parked
            // free run; a carved run keeps its front and is followed by the spans
            // carved off its back (address order); spans placed at the end are
            // appended. Adjacent free runs merge, as trySlotEdit's sort does.
            std::vector<std::vector<uint32_t>> carvedOf(runs.size());
            std::vector<uint32_t> atEnd;
            for (uint32_t b = 0; b < places.size(); ++b) {
                if (places[b].run >= 0) carvedOf[static_cast<size_t>(places[b].run)].push_back(b);
                else atEnd.push_back(b);
            }
            std::vector<int32_t> newOfSpan(d.spans.size(), -1);
            const auto newSpanOf = [&](uint32_t b) {
                EntrySpan ns = d.spans[b];
                ns.first = places[b].dst;
                ns.isFree = false;
                ns.parkedUntil = 0u;
                ns.fadeKnown = false;
                ns.admitPending = pending[b] != 0u;
                if (ns.inst) {
                    ns.instColorAttr    = ns.inst->instanceColor();
                    ns.instColorVersion = ns.instColorAttr ? ns.instColorAttr->version : ~0u;
                }
                return ns;
            };
            // In place when nothing has to move: every carve an exact fit with
            // no room to grow after it (the run becomes the span), the rest at
            // the end, a removed span freed where it is (merged with its free
            // neighbours by the next edit that moves spans). Every span keeps its
            // index, the snapshot's included.
            bool inPlace = true;
            for (size_t k = 0; k < runs.size() && inPlace; ++k)
                if (!carvedOf[k].empty() &&
                    (carvedOf[k].size() != 1u || runs[k].count != 0u || places[carvedOf[k][0]].slack != 0u))
                    inPlace = false;
            if (inPlace) {
                for (const uint32_t r : d.removed) {
                    const uint32_t first = entrySpans_[r].first, count = entrySpans_[r].count;
                    entrySpans_[r] = slotFreeSpan(first, count, parkUntil);
                }
                for (size_t k = 0; k < runs.size(); ++k) {
                    if (carvedOf[k].empty()) continue;
                    const uint32_t b = carvedOf[k][0];
                    entrySpans_[runs[k].span] = newSpanOf(b);
                    newOfSpan[b] = static_cast<int32_t>(runs[k].span);
                }
                for (const uint32_t b : atEnd) {
                    newOfSpan[b] = static_cast<int32_t>(entrySpans_.size());
                    entrySpans_.push_back(newSpanOf(b));
                    if (places[b].slack > 0u)
                        entrySpans_.push_back(slotFreeSpan(places[b].dst + d.spans[b].count, places[b].slack, 0u));
                }
                ++tlasStats_.slotDeltasInPlace;
            } else {
                // Kept spans move over as they are; a removed one becomes a parked
                // free run; a carved run keeps its front and is followed by the
                // spans carved off its back (address order); spans placed at the
                // end are appended. Adjacent free runs merge, as trySlotEdit's
                // sort does.
                std::vector<int32_t> runOfSpan(entrySpans_.size(), -1);
                for (size_t k = 0; k < runs.size(); ++k)
                    if (!carvedOf[k].empty()) runOfSpan[runs[k].span] = static_cast<int32_t>(k);
                std::vector<EntrySpan> next;
                next.reserve(entrySpans_.size() + 2u * d.spans.size() + 2u);
                std::vector<int32_t> oldToNew(entrySpans_.size(), -1);
                const auto pushFree = [&](uint32_t first, uint32_t count, uint64_t until) {
                    if (count == 0u) return;
                    if (!next.empty() && next.back().isFree && next.back().first + next.back().count == first) {
                        next.back().count += count;
                        next.back().parkedUntil = std::max(next.back().parkedUntil, until);
                        return;
                    }
                    next.push_back(slotFreeSpan(first, count, until));
                };
                const auto pushNew = [&](uint32_t b) {
                    newOfSpan[b] = static_cast<int32_t>(next.size());
                    next.push_back(newSpanOf(b));
                    pushFree(places[b].dst + d.spans[b].count, places[b].slack, 0u);
                };
                for (uint32_t s = 0; s < entrySpans_.size(); ++s) {
                    EntrySpan& sp = entrySpans_[s];
                    if (removedMark[s]) {
                        pushFree(sp.first, sp.count, parkUntil);
                        continue;
                    }
                    if (sp.isFree) {
                        const int32_t k = runOfSpan[s];
                        if (k < 0) {
                            pushFree(sp.first, sp.count, sp.parkedUntil);
                            continue;
                        }
                        auto& carved = carvedOf[static_cast<size_t>(k)];
                        pushFree(sp.first, runs[static_cast<size_t>(k)].count, sp.parkedUntil);
                        std::sort(carved.begin(), carved.end(),
                                  [&](uint32_t x, uint32_t y) { return places[x].dst < places[y].dst; });
                        for (const uint32_t b : carved) pushNew(b);
                        continue;
                    }
                    oldToNew[s] = static_cast<int32_t>(next.size());
                    next.push_back(std::move(sp));
                }
                for (const uint32_t b : atEnd) pushNew(b);
                entrySpans_ = std::move(next);
                // The snapshot's kept nodes at their spans' new places.
                std::vector<uint8_t> isNew(d.snap.size(), 0u);
                for (const uint32_t k : d.spanSnap) isNew[k] = 1u;
                for (size_t k = 0; k < d.snap.size(); ++k) {
                    SnapNode& sn = d.snap[k];
                    if (sn.spanIndex < 0 || isNew[k]) continue;
                    const auto o = static_cast<size_t>(sn.spanIndex);
                    sn.spanIndex = o < oldToNew.size() ? oldToNew[o] : -1;
                }
            }
#ifndef NDEBUG
            {
                uint32_t at = 0;
                for (const auto& sp : entrySpans_) {
                    THREEPP_ASSERT(sp.first == at);
                    at += sp.count;
                }
                THREEPP_ASSERT(at == newSize);
            }
#endif
            ++spansVersion_;
            // The new nodes at their spans; then the in-place visibility flips.
            for (size_t b = 0; b < d.spanSnap.size(); ++b) d.snap[d.spanSnap[b]].spanIndex = newOfSpan[b];
            sceneSnapshot_.swap(d.snap);
            for (const SnapFlip& f : d.flips) {
                SnapNode& sn = sceneSnapshot_[f.snIdx];
                sn.flags = (sn.flags & ~(kSnapHidden | kSnapMatHidden)) | (f.hidden ? kSnapHidden : 0u) |
                           (f.matHidden ? kSnapMatHidden : 0u);
                if (sn.spanIndex >= 0) setSpanHidden(static_cast<size_t>(sn.spanIndex), f.hidden || f.matHidden);
            }

            dSpans.stop();
            ++motionScratchVersion_;
            ++lodFadeVersion_;
            cachedEmissiveEntryCount_ = static_cast<size_t>(-1);
            ++drawInputsVersion_;
            if (texturesAdded) {
                deferredDescDirty_.fill(true);
                for (auto& v : views_) {
                    v->rasterMatTexValid_.fill(0);
                    v->shadowMatTexValid_.fill(0);
                }
            }
            pruneFreedAssets();
            stableIdsPending_ = true;
            slotsEditedThisFrame_ = true;
            ++tlasStats_.slotEdits;
            ++tlasStats_.slotDeltas;
            tlasStats_.swapEdits += d.swaps;
            return true;
        }

void VulkanRenderer::Impl::pruneFreedAssets() {
            THREEPP_CPUPROF("scene.S6_slotPrune");
            // A record whose geometry died: no mesh draws it any more (the edit
            // that dropped its last mesh tombstoned that mesh's rows), but the
            // frames in flight may still trace its BLAS and read its buffers.
            // It leaves blasCache now (a new geometry at the same address gets
            // a record of its own) and is destroyed kFramesInFlight frames on.
            for (auto it = blasCache.begin(); it != blasCache.end();) {
                if (!it->second->liveCheck.expired()) {
                    ++it;
                    continue;
                }
                BlasRecord* rec = it->second.get();
                blasCompactPending_.erase(std::remove(blasCompactPending_.begin(), blasCompactPending_.end(), rec),
                                          blasCompactPending_.end());
                // An async batch that queried its BLAS for compaction passes it
                // over (records are matched by key and structure, and either can
                // come back on a new geometry).
                for (auto& os : asyncOneShots_)
                    for (auto& q : os.compaction.records)
                        if (q.first == it->first) q.first = nullptr;
                forceUnpackedGeoms_.erase(it->first);
                worldStaticGeoms_.erase(it->first);
                // The dynamic-geometry queues hold records by pointer for the
                // frame being recorded; a dead geometry is never queued, but a
                // frame that queued and never recorded must not hand it on.
                pendingDynamicGeomRefits_.erase(
                        std::remove_if(pendingDynamicGeomRefits_.begin(), pendingDynamicGeomRefits_.end(),
                                       [rec](const GeomRefreshOp& op) { return op.rec == rec; }),
                        pendingDynamicGeomRefits_.end());
                pendingDynamicPrevResyncs_.erase(
                        std::remove(pendingDynamicPrevResyncs_.begin(), pendingDynamicPrevResyncs_.end(), rec),
                        pendingDynamicPrevResyncs_.end());
                retiredRecords_.emplace_back(frameSerial_, std::move(it->second));
                ++tlasStats_.recordsPruned;
                it = blasCache.erase(it);
            }
            // A grass field whose mesh or geometry died: the same, with its two
            // input buffers. A new GrassMesh at the dead one's address must not
            // find the old state.
            for (auto it = grassStates.begin(); it != grassStates.end();) {
                GrassMeshState& st = *it->second;
                if (!st.liveCheck.expired() && !st.owner.gone()) {
                    ++it;
                    continue;
                }
                retireGrassState(st);
                ++tlasStats_.recordsPruned;
                it = grassStates.erase(it);
            }
            // A texture that died: no live material names it, and every row that
            // did was tombstoned. Its image is retired; its slot is free at once.
            bool freed = false;
            const auto freeSlot = [&](uint32_t slot) {
                if (slot >= materialTextures.size() || !materialTextures[slot].view) return;
                retire(std::move(materialTextures[slot]));
                materialTextures[slot] = {};
                freeTextureSlots.push_back(slot);
                ++tlasStats_.texturesPruned;
                freed = true;
            };
            for (auto it = textureCache.begin(); it != textureCache.end();) {
                if (!it->second.ref.expired()) {
                    ++it;
                    continue;
                }
                freeSlot(it->second.slot);
                it = textureCache.erase(it);
            }
            for (const uint32_t slot : retiredTextureSlots_) freeSlot(slot);
            retiredTextureSlots_.clear();
            // A texture prepared for a hidden admission whose mesh was dropped
            // before it landed: only the job holds it now.
            for (auto it = preparingTextures_.begin(); it != preparingTextures_.end();) {
                if (it->second.tex.use_count() == 1 &&
                    it->second.job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    it = preparingTextures_.erase(it);
                } else {
                    ++it;
                }
            }
            if (freed) {
                // The descriptor sets name the retired views until the per-FIF
                // rewrite (refreshDirtyMaterialTextures' path) gives the slot
                // its next image or the white default.
                deferredDescDirty_.fill(true);
                for (auto& v : views_) {
                    v->rasterMatTexValid_.fill(0);
                    v->shadowMatTexValid_.fill(0);
                }
            }
        }

std::vector<std::shared_ptr<Texture>> VulkanRenderer::Impl::texturesOf(const Mesh& m) {
            // Every map buildMaterialDesc admits (the terrain's band sets are
            // shared and resident after the first tile, but listed all the same).
            std::vector<std::shared_ptr<Texture>> out = {
                    albedoTexOf(m),     roughnessTexOf(m),   metalnessTexOf(m),     normalTexOf(m),
                    transmissionTexOf(m), clearcoatTexOf(m), clearcoatRoughnessTexOf(m), emissiveTexOf(m),
                    occlusionTexOf(m),  detailTexOf(m),      detailNormalTexOf(m),  terrainWeightTexOf(m),
                    terrainNormalTexOf(m)};
            if (terrainWeightTexOf(m)) {
                for (int bi = 0; bi < 4; ++bi) {
                    out.push_back(terrainBandAlbedoTexOf(m, bi));
                    out.push_back(terrainBandNormalTexOf(m, bi));
                }
            }
            return out;
        }

bool VulkanRenderer::Impl::grassInputsValid(const BufferGeometry& g) {
            // ensureGrassState's own test, ahead of an edit that cannot back out.
            const auto* pos = g.getAttribute<float>("position");
            const auto* hf = g.getAttribute<float>("heightFrac");
            return pos && hf && pos->count() > 0 && hf->count() == pos->count();
        }

void VulkanRenderer::Impl::retireGrassState(GrassMeshState& st) {
            // The frames in flight may still deform it, refit it and trace it.
            const GrassMeshState* p = &st;
            pendingGrassDeforms_.erase(std::remove_if(pendingGrassDeforms_.begin(), pendingGrassDeforms_.end(),
                                                      [p](const auto& d) { return d.second == p; }),
                                       pendingGrassDeforms_.end());
            if (st.blas) retiredRecords_.emplace_back(frameSerial_, std::move(st.blas));
            retire(std::move(st.restPos));
            retire(std::move(st.heightFrac));
            st.restPos = {};
            st.heightFrac = {};
        }

VulkanRenderer::Impl::GrassMeshState* VulkanRenderer::Impl::admitGrassState(GrassMesh& gm) {
            if (const auto it = grassStates.find(&gm); it != grassStates.end()) {
                // Its own state, for the geometry it draws: a cached field back.
                // Primed like a new one: it had no entries, so no scan deformed
                // it (or remembered skipping it) while it was away, and its
                // last pose would be its previous one, the wind change since
                // it left drawn as motion.
                if (it->second->liveCheck.lock() == gm.geometry() && !it->second->owner.gone()) {
                    it->second->primePending = true;
                    return it->second.get();
                }
                // A geometry swapped while the old one lives on elsewhere.
                retireGrassState(*it->second);
                grassStates.erase(it);
            }
            // Its record and inputs record into the open batch (no wait).
            auto* st = ensureGrassState(gm);
            if (st) st->primePending = true;
            return st;
        }

bool VulkanRenderer::Impl::syncAdmission() {
            static const bool sync = [] {
                const char* e = std::getenv("THREEPP_VULKAN_SYNC_ADMIT");
                return e && e[0] == '1';
            }();
            return sync;
        }

void VulkanRenderer::Impl::beginHiddenAdmission(Mesh& m) { prepareTexturesAsync(m); }

void VulkanRenderer::Impl::prepareTexturesAsync(Mesh& m) {
            const bool bc7 = bc7SampledSupported();
            for (const auto& t : texturesOf(m)) {
                if (!t) continue;
                const Texture* key = t.get();
                if (const auto it = textureCache.find(key); it != textureCache.end()) {
                    const auto live = it->second.ref.lock();
                    if (live && live.get() == key) continue;// resident already
                }
                if (preparingTextures_.count(key) != 0u) continue;
                PreparingTexture pt;
                pt.tex = t;
                pt.version = t->version();
                // The CPU half only: it reads the texture's image and nothing of
                // the renderer's state.
                pt.job = std::async(std::launch::async,
                                    [this, t, bc7] { return prepareMaterialImage(t.get(), bc7); });
                preparingTextures_.emplace(key, std::move(pt));
            }
        }

void VulkanRenderer::Impl::pumpHiddenAdmissions() {
            if (admitPendingSpans_ == 0u) return;
            THREEPP_CPUPROF("scene.3e_hiddenAdmission");
            // A few spans a frame, and no more once this frame's share of the
            // work has been spent: a landing costs the BLAS build input and the
            // upload of encoded images, not the encode.
            constexpr uint32_t kSpansPerFrame = 4u;
            constexpr double kBudgetMs = 2.0;
            const auto t0 = std::chrono::steady_clock::now();
            const auto spentMs = [&] {
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            };
            const auto ready = [&](const std::shared_ptr<Texture>& t) {
                if (!t) return true;
                const auto it = preparingTextures_.find(t.get());
                return it == preparingTextures_.end() ||
                       it->second.job.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            };
            const auto texturesReady = [&](Mesh& m) {
                for (const auto& t : texturesOf(m))
                    if (!ready(t)) return false;
                return true;
            };
            const auto cullOf = [](const MaterialDesc& md) -> VkCullModeFlags {
                switch (md.sideMode) {
                    case 0: return VK_CULL_MODE_BACK_BIT;
                    case 1: return VK_CULL_MODE_FRONT_BIT;
                    default: return VK_CULL_MODE_NONE;
                }
            };
            const size_t texCount0 = materialTextures.size();
            const size_t texFree0 = freeTextureSlots.size();
            const bool rq = ctx->rayQuerySupported();
            auto& entries = lastVisibleEntries_;
            uint32_t pending = 0u, landed = 0u;
            bool batch = false;
            // Called on a frame that ends lean (its walk matched, or its slot
            // edit took the change): every live span's mesh is alive.
            for (EntrySpan& sp : entrySpans_) {
                if (sp.isFree || !sp.admitPending) continue;
                if (landed >= kSpansPerFrame || spentMs() > kBudgetMs || !texturesReady(*sp.mesh)) {
                    ++pending;
                    continue;
                }
                Mesh& m = *sp.mesh;
                const auto geom = m.geometry();
                if (!batch) {
                    beginOneShotBatch();
                    batch = true;
                }
                const BlasRecord* recp = nullptr;
                if (entries[sp.first].isGrass) {
                    // A field's own record and inputs, primed by its first deform.
                    auto* st = admitGrassState(static_cast<GrassMesh&>(m));
                    recp = st ? st->blas.get() : nullptr;
                } else {
                    auto rit = geom ? blasCache.find(geom.get()) : blasCache.end();
                    if (geom && rit == blasCache.end()) {
                        auto rec = buildBlasFor(*geom, /*allowPacked=*/true, /*updatable=*/false, nullptr,
                                                VertexWriter::Host, /*traced=*/rq);
                        if (rec) {
                            rec->liveCheck = geom;
                            rit = blasCache.emplace(geom.get(), std::move(rec)).first;
                        }
                    }
                    recp = rit != blasCache.end() ? rit->second.get() : nullptr;
                }
                sp.admitPending = false;
                if (!recp) continue;// degenerate: default rows and no instance, as in a rebuild
                const BlasRecord& rec = *recp;
                MaterialDesc md = buildMaterialDesc(m);// takes the prepared images
                md.materialAssetIdx = slotMaterialAssetIdx(m.material().get());
                const GeometryDesc gd = buildGeometryDesc(entries[sp.first], rec, selectLodGeom(rec, 0));
                for (uint32_t i = sp.first; i < sp.first + sp.count; ++i) {
                    geomDescsCached_[i] = gd;
                    MaterialDesc mi = md;
                    applyInstanceColor(mi, entries[i]);
                    matDescsCached_[i] = mi;
                    lastVisibleCullMode_[i] = cullOf(mi);
                }
                markGeomDescsDirty(sp.first, sp.count);
                markMatDescsDirty(sp.first, sp.count);
                ++landed;
                ++tlasStats_.admissionsLanded;
            }
            if (batch) {
                // As the slot edit's batch: ahead of this frame on the queue,
                // never waited for (the async-compute pilot waits).
                if (asyncPilot_) {
                    flushOneShotBatch();
                    std::vector<VkAccelerationStructureInstanceKHR> none;
                    applyBlasCompactionRemap(none);
                } else {
                    flushOneShotBatchAsync();
                }
            }
            admitPendingSpans_ = pending;
            if (landed == 0u) return;
            // New BLASes and rows: a full TLAS build and fresh draw records, as
            // a first show (the spans stay hidden).
            hiddenFlippedThisFrame_ = true;
            ++drawInputsVersion_;
            cachedEmissiveEntryCount_ = static_cast<size_t>(-1);
            if (materialTextures.size() != texCount0 || freeTextureSlots.size() != texFree0) {
                deferredDescDirty_.fill(true);
                for (auto& v : views_) {
                    v->rasterMatTexValid_.fill(0);
                    v->shadowMatTexValid_.fill(0);
                }
            }
        }

void VulkanRenderer::Impl::refreshPendingStableIds() {
            if (!stableIdsPending_) return;
            bool still = false;
            for (const EntrySpan& sp : entrySpans_) {
                if (sp.isFree || sp.hidden || sp.count == 0u || size_t(sp.first) + sp.count > entryStableIds_.size())
                    continue;
                if (entryStableIds_[sp.first] != 0u) continue;
                const MeshEntry& e0 = lastVisibleEntries_[sp.first];
                if (e0.isOverlay || e0.isParticle || e0.isParticleField) continue;
                const uint32_t id = stableIdIfAssigned(*sp.mesh);
                if (id == 0u) {
                    still = true;// not drawn yet: its id comes with its first draw
                    continue;
                }
                for (uint32_t i = sp.first; i < sp.first + sp.count; ++i) entryStableIds_[i] = id;
            }
            stableIdsPending_ = still;
        }

void VulkanRenderer::Impl::reapRetiredRecords(bool all) {
            if (retiredRecords_.empty()) return;
            size_t keep = 0;
            for (size_t i = 0; i < retiredRecords_.size(); ++i) {
                auto& [serial, rec] = retiredRecords_[i];
                if (!all && serial + kFramesInFlight > frameSerial_) {
                    if (keep != i) retiredRecords_[keep] = std::move(retiredRecords_[i]);
                    ++keep;
                    continue;
                }
                // A chain still landing: its batch was submitted ahead of a frame
                // that has completed, so its fence has signalled; wait on it all
                // the same before freeing what it builds into.
                if (rec->lodLandingBatch != 0) {
                    for (const auto& b : lodLanding_)
                        if (b.id == rec->lodLandingBatch && b.fence != VK_NULL_HANDLE)
                            check(vkWaitForFences(ctx->device(), 1, &b.fence, VK_TRUE, UINT64_MAX),
                                  "wait auto-LOD batch of a retired record");
                }
                destroyBlasLodLevels(*rec);
                destroyBlasRecord(*rec);
            }
            retiredRecords_.resize(keep);
        }

void VulkanRenderer::Impl::growSlotPool(uint32_t capacity) {
            THREEPP_CPUPROF("scene.3d_slotPoolGrowth");
            // No device wait (until 2026-10-20 there was one): this runs ahead
            // of the frame's fence wait, so the frames in flight may still read
            // every buffer below. They keep them: each old buffer is retired
            // with this frame's serial and freed once both are done, and their
            // descriptor sets are rewritten only after their own fences (the
            // deferred-descriptor refresh, as for a texture swap). The wait was
            // never the cost (1.5 ms): the new buffers are tens of MB the
            // driver pages in on the GPU timeline, and the old ones' frees,
            // 10-20 ms either way. reserveEntrySlots avoids a growth outright.
            THREEPP_CPUPROF_NAMED(gBufs, "scene.G2_growBuffers");
            slotCapacity_ = capacity;
            // The desc rings, every slot from the host copy (which the frame-
            // in-flight flushes carry deltas from).
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                retire(std::move(geometryDescsBuffers[f]));
                uploadDescBuffer(geometryDescsBuffers[f], geomDescsCached_, slotCapacity_);
                retire(std::move(materialDescsBuffers[f]));
                uploadDescBuffer(materialDescsBuffers[f], matDescsCached_, slotCapacity_);
                ensureMotionMatCapacity(f, slotCapacity_);
                ensureMeshMovedBitsCapacity(f, (slotCapacity_ + 31u) / 32u);
            }
            for (auto& d : geomDescsDirty_) d.clear();
            for (auto& d : matDescsDirty_) d.clear();
            gBufs.stop();
            THREEPP_CPUPROF_NAMED(gTlas, "scene.G3_growTlas");
            // The TLAS: a new structure for the new pool, holding what the old
            // one held until this frame's full build (the edit's) replaces it.
            // The passes recorded ahead of that build (a ParticleField's
            // surface bake) trace it as they trace the last frame's on any
            // other frame. The old one is cloned into it in an async batch
            // ahead of the frame (no wait) and retired with this frame.
            const VkAccelerationStructureKHR oldTlas = tlas;
            Buffer oldTlasBuffer = tlasBuffer;
            const uint32_t builtBefore = tlasBuiltInstanceCount_;
            tlas = VK_NULL_HANDLE;
            tlasBuffer = {};
            buildTlas({}, /*buildNow=*/false);
            if (oldTlas != VK_NULL_HANDLE) {
                if (tlas != VK_NULL_HANDLE) {
                    VkCommandBufferAllocateInfo ai{};
                    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                    ai.commandPool = cmdPool;
                    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                    ai.commandBufferCount = 1;
                    VkCommandBuffer cb = VK_NULL_HANDLE;
                    check(vkAllocateCommandBuffers(ctx->device(), &ai, &cb), "alloc TLAS clone cb");
                    VkCommandBufferBeginInfo bi{};
                    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    check(vkBeginCommandBuffer(cb, &bi), "begin TLAS clone cb");
                    // The frame in flight may still be refitting or building the
                    // old structure: the clone reads it after those writes.
                    VkMemoryBarrier2 mb{};
                    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
                    mb.dstStageMask = ctx->asBuildStage2();
                    mb.dstAccessMask = ctx->asReadAccess2() | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                    VkDependencyInfo dep{};
                    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                    VkCopyAccelerationStructureInfoKHR copy{};
                    copy.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
                    copy.src = oldTlas;
                    copy.dst = tlas;
                    copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
                    ctx->rt().cmdCopyAccelerationStructure(cb, &copy);
                    check(vkEndCommandBuffer(cb), "end TLAS clone cb");
                    VkFenceCreateInfo fi{};
                    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                    VkFence fence = VK_NULL_HANDLE;
                    check(vkCreateFence(ctx->device(), &fi, nullptr, &fence), "create TLAS clone fence");
                    VkSubmitInfo si{};
                    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    si.commandBufferCount = 1;
                    si.pCommandBuffers = &cb;
                    check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence), "submit TLAS clone");
                    asyncOneShots_.push_back({cb, fence, {}});
                    tlasBuiltInstanceCount_ = builtBefore;
                    tlasStats_.instances = builtBefore;
                }
                retireAS(oldTlas);
                retire(std::move(oldTlasBuffer));
            }
            gTlas.stop();
            THREEPP_CPUPROF_NAMED(gSets, "scene.G4_growDescriptors");
            // Each frame-in-flight's deferred sets, after its own fence (this
            // frame's in beginDeferredFrame, the other's when its frame comes);
            // the raster and shadow sets likewise at their next use.
            deferredDescDirty_.fill(true);
            for (auto& v : views_) {
                v->rasterMatTexValid_.fill(0);
                v->shadowMatTexValid_.fill(0);
            }
            ++tlasStats_.slotPoolGrowths;
        }

}// namespace threepp
