#include "VulkanCoreImpl.hpp"

namespace threepp {

uint32_t VulkanRendererCore::CoreImpl::snapMeshFlags(Mesh& m, const MaterialWithWireframe* wf) const {
            uint32_t fl = kSnapKindMesh;
            if (auto geom = m.geometry()) {
                if (geom->hasAttribute("position")) fl |= kSnapHasPos;
                if (geom->hasAttribute("normal")) fl |= kSnapHasNorm;
            }
            if (wf && wf->wireframe) fl |= kSnapWire;
            if (overlayLayer_ >= 0 &&
                m.layers.isEnabled(static_cast<unsigned>(overlayLayer_))) fl |= kSnapOverlay;
            if (auto mat = m.material(); mat && mat->tetSkinning && mat->tetTexture) fl |= kSnapTet;
            // Unlit transparent flat-color mesh → raster overlay routing (see
            // kSnapUiBlend). Textured / vertex-colored basics stay traced —
            // the overlay fill pipeline is flat-color push-constant only.
            if (!(fl & kSnapWire)) {
                if (auto mat = m.material()) {
                    if (auto* mb = dynamic_cast<MeshBasicMaterial*>(mat.get());
                        mb && mb->transparent && !mb->map && !mb->vertexColors) fl |= kSnapUiBlend;
                }
            }
            // ParticleSystem billboard mesh — detected by the unique material-name
            // marker (cheap: a length-mismatch reject for the empty-named common
            // case). Routed to the dedicated billboard pass and excluded from the
            // traced/rasterized scene.
            if (auto mat = m.material(); mat && mat->name == kParticleMaterialName) fl |= kSnapParticle;
            return fl;
        }

bool VulkanRendererCore::CoreImpl::sceneSnapshotMatches(Object3D& scene, Camera& camera) {
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
                const bool tet = sn.mat && sn.mat->tetSkinning && sn.mat->tetTexture != nullptr;
                const bool particle = sn.mat && sn.mat->name == kParticleMaterialName;
                const bool uiBlend = !wire && sn.basic && sn.basic->transparent &&
                                     !sn.basic->map && !sn.basic->vertexColors;
                // sn.mat compared equal above, so it's alive and dereferenceable
                // (nullptr ⇒ no material ⇒ treated as visible). [[#mat-visible]]
                const bool matHidden = sn.mat && !sn.mat->visible;
                if (wire != ((sn.flags & kSnapWire) != 0u) ||
                    over != ((sn.flags & kSnapOverlay) != 0u) ||
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

void VulkanRendererCore::CoreImpl::flushMaterialDescsIfDirty(uint32_t frame) {
            if (!matDescsDirty_[frame]) return;
            matDescsDirty_[frame] = false;
            if (matDescsCached_.empty()) return;
            uploadHostVisible(ctx->allocator(), materialDescsBuffers[frame],
                              matDescsCached_.data(),
                              matDescsCached_.size() * sizeof(MaterialDesc));
        }

void VulkanRendererCore::CoreImpl::flushGeometryDescsIfDirty(uint32_t frame) {
            if (!geomDescsDirty_[frame]) return;
            geomDescsDirty_[frame] = false;
            if (geomDescsCached_.empty()) return;
            if (geometryDescsBuffers[frame].handle == VK_NULL_HANDLE) return;
            uploadHostVisible(ctx->allocator(), geometryDescsBuffers[frame],
                              geomDescsCached_.data(),
                              geomDescsCached_.size() * sizeof(GeometryDesc));
        }

void VulkanRendererCore::CoreImpl::cullEntriesAgainstFrustum(Camera& camera) {
            if (lastVisibleEntries_.empty()) return;
            // Combine projection * matrixWorldInverse to extract the world-
            // space frustum (Three.js convention; Camera::updateMatrixWorld
            // already ran in updateCameraUbo this frame).
            Matrix4 vp;
            vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
            Frustum frustum;
            frustum.setFromProjectionMatrix(vp);
            for (auto& en : lastVisibleEntries_) {
                // Default-include conservative cases — they always draw. Deformers
                // (skinned/displaced/morphed/tet) are here because their cached local
                // AABB doesn't reflect the per-frame deformed extents, so frustum-
                // culling them risks popping a still-on-screen body out of the gbuffer.
                if (en.isOverlay || en.isSkinned || en.isDisplaced || en.isMorphed || en.isTet) {
                    en.inFrustum = true;
                } else if (en.isGrass) {
                    // Grass CAN be frustum-culled: unlike the other deformers, its
                    // deformed extent has a tight provable bound. The CPU position
                    // attribute is the rest pose; grass_wind.comp displaces each
                    // vertex by windDir·bend with |bend| ≤ 0.85·windStrength, so the
                    // rest AABB dilated by windStrength conservatively encloses every
                    // swayed pose (windDir is ~unit). Test THAT box normally — this
                    // is what lets a large tiled meadow cull its off-screen tiles.
                    // (The tile still stays in the TLAS for shadows/reflections/GI;
                    // inFrustum only gates the raster G-buffer draw.)
                    Box3 worldAabb;
                    if (!grassSwayWorldAabb(en, worldAabb)) { en.inFrustum = true; continue; }
                    en.inFrustum = frustum.intersectsBox(worldAabb);
                } else {
                    auto geom = en.mesh->geometry();
                    if (!geom) { en.inFrustum = true; continue; }
                    if (!geom->boundingBox) geom->computeBoundingBox();
                    if (!geom->boundingBox) { en.inFrustum = true; continue; }
                    Box3 worldAabb = *geom->boundingBox;
                    Matrix4 w;
                    std::memcpy(w.elements.data(), en.worldMatrix.data(), 64);
                    worldAabb.applyMatrix4(w);
                    en.inFrustum = frustum.intersectsBox(worldAabb);
                }
            }
        }

void VulkanRendererCore::CoreImpl::ensureSceneBuilt(Object3D& scene, Camera& camera) {
            // force=false (matching GLRenderer): with
            // updateMatrix()'s change-detection early-out, only subtrees whose
            // transforms actually moved pay the world-matrix multiplies — a
            // forced pass re-multiplied every node every frame (several
            // ms/frame on Bistro's node count, static or not).
            scene.updateMatrixWorld();
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
            const bool snapLean = sceneSnapshotMatches(scene, camera);
            std::vector<MeshEntry>& entries = lastVisibleEntries_;// canonical list, cached across frames
            std::vector<MeshFingerprint> currFp;
            if (snapLean) {
                for (auto& e : entries) {
                    if (e.isInstanced) {
                        Matrix4 instMat;
                        Matrix4 world;
                        static_cast<InstancedMesh*>(e.mesh)->getMatrixAt(e.instanceIndex, instMat);
                        world.multiplyMatrices(*e.mesh->matrixWorld, instMat);
                        std::memcpy(e.worldMatrix.data(), world.elements.data(), 64);
                    } else {
                        std::memcpy(e.worldMatrix.data(), e.mesh->matrixWorld->elements.data(), 64);
                    }
                }
                for (auto& le : lastVisibleLines_) {
                    const Object3D* src = le.line ? static_cast<const Object3D*>(le.line)
                                                  : static_cast<const Object3D*>(le.points);
                    std::memcpy(le.worldMatrix.data(), src->matrixWorld->elements.data(), 64);
                }
            } else {
            // Expand the visible scene into one MeshEntry per TLAS instance.
            // Regular meshes contribute one entry; an InstancedMesh contributes
            // count() entries each with worldMatrix = matrixWorld * instanceMat[i].
            // Mirrors WGPU's expandMeshEntries (WgpuPathTracerAtlas.cpp:20).
            std::vector<MeshEntry> built;
            std::vector<LineEntry> builtLines;
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
                // One-shot type probes: an N-instance InstancedMesh costs 3
                // dynamic_casts total, not 3·N — and on snapshot-match frames
                // none at all (the cached entry flags are reused). Consumed by
                // raster pass loops, resolveBlasForEntry, TLAS refit, and
                // dirty-detection.
                const bool isSkinned   = (dynamic_cast<SkinnedMesh*>(m)   != nullptr);
                const bool isDisplaced = (dynamic_cast<DisplacedMesh*>(m) != nullptr);
                const bool isGrass     = (dynamic_cast<GrassMesh*>(m)     != nullptr);
                const bool isMorphed   = isMorphedMesh(*m);
                // Tet-skinned PhysX soft body — detected via the material flag set by
                // SoftBody::enableGpuSkinning() (which also carries the per-frame tet
                // texture and the static tetIndex/tetWeight/tetRestInv* attributes).
                // Mutually exclusive with the other deformers.
                const bool isTet = !isSkinned && !isDisplaced && !isGrass && !isMorphed &&
                                   m->material() && m->material()->tetSkinning &&
                                   m->material()->tetTexture != nullptr;
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
                auto setLodCaches = [&](MeshEntry& e) {
                    e.lodExemptStatic = lodExempt;
                    e.lodEmissive       = lodEmissive;
                    e.lodGeomKey        = geom.get();
                    e.lodCenter[0]      = lodCenter.x;
                    e.lodCenter[1]      = lodCenter.y;
                    e.lodCenter[2]      = lodCenter.z;
                    e.lodRadius         = lodRadius;
                };
                if (inst && inst->count() > 0) {
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
                        e.isInstanced  = true;
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
                    setLodCaches(e);
                    std::memcpy(e.worldMatrix.data(), m->matrixWorld->elements.data(), 64);
                    built.push_back(e);
                }
            });
            lastVisibleEntries_ = std::move(built);
            lastVisibleLines_   = std::move(builtLines);
            probeGridDirty_     = true;// scene structure changed → re-fit the probe grid
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
            if (autoLod_) drainLodResults();// budget: one geometry finalized per frame
            {
                VulkanRendererCore::AutoLodStats stats{};
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

                    for (auto& en : entries) {
                        if (en.isOverlay || en.isParticle) continue;// not scene geometry — no LOD concept
                        // Deformers (stale local AABB), opted-out meshes
                        // (Object3D::autoLod == false — self-managed LOD like
                        // terrain tiles), authored manual-LOD subtrees, and
                        // non-perspective cameras (SSE needs a pinhole model):
                        // full detail.
                        if (en.isSkinned || en.isDisplaced || en.isGrass || en.isMorphed || en.isTet ||
                            en.lodExemptStatic || !persp) {
                            if (en.lodLevel != 0) lodChangedThisFrame_ = true;
                            en.lodLevel = 0;
                            ++stats.entriesPerLevel[0];
                            continue;
                        }
                        // Emissive VALUES read live off the expansion-cached
                        // cast — NEE's per-tri CDF (buildAndUploadEmissiveTris)
                        // caches world-space triangles and would not notice a
                        // silent index-buffer swap underneath it. Live values
                        // (not a cached verdict) so a lantern whose intensity
                        // ramps up at dusk exempts the moment it turns on.
                        if (en.lodEmissive &&
                            (en.lodEmissive->emissive.r > 0.f || en.lodEmissive->emissive.g > 0.f ||
                             en.lodEmissive->emissive.b > 0.f || en.lodEmissive->emissiveMap)) {
                            if (en.lodLevel != 0) lodChangedThisFrame_ = true;
                            en.lodLevel = 0;
                            ++stats.entriesPerLevel[0];
                            continue;
                        }

                        auto blasIt = blasCache.find(en.lodGeomKey);
                        if (blasIt == blasCache.end()) {
                            en.lodLevel = 0;// brand-new geometry this frame — no LOD0 record to chain off yet
                            ++stats.entriesPerLevel[0];
                            continue;
                        }
                        BlasRecord& rec = *blasIt->second;

                        // NON-indexed soup (FBX-style loaders never call
                        // setIndex) is eligible too: the chain generator welds
                        // it into canonical indices, and the levels drive
                        // INDEXED draws against the unchanged soup vertex
                        // buffer (selectLodGeom reports the indexed-ness).
                        const uint32_t triCount =
                                (rec.indexCount != 0u ? rec.indexCount : rec.vertexCount) / 3u;
                        const bool eligible = triCount >= 1024u;
                        if (eligible && rec.lodState == BlasRecord::LodState::None) {
                            if ((lodIndexBytes_ + lodBlasBytes_) <= kLodByteBudget) {
                                // The enqueue snapshots attribute data, so it needs
                                // the live geometry — the ONLY shared_ptr deref
                                // left in this loop, paid once per geometry
                                // lifetime. Failed enqueue (no position attribute)
                                // marks Failed, not Queued: no result is coming,
                                // so Queued would strand the record forever.
                                auto geomSp = en.mesh->geometry();
                                rec.lodState = (geomSp && enqueueLodJob(en.lodGeomKey, rec.geomVersion, *geomSp,
                                                                        lodNormalWeightFor(*en.mesh)))
                                        ? BlasRecord::LodState::Queued
                                        : BlasRecord::LodState::Failed;
                            } else if (!lodBudgetWarned_) {
                                std::cerr << "[VulkanRenderer] auto-LOD: 256 MiB byte budget reached — "
                                             "no further chains will be generated this session\n";
                                lodBudgetWarned_ = true;
                            }
                        }

                        if (rec.lodState != BlasRecord::LodState::Ready || rec.lodLevels.empty()) {
                            en.lodLevel = 0;
                            ++stats.entriesPerLevel[0];
                            continue;
                        }

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
                        ++stats.entriesPerLevel[std::min<uint8_t>(selected, 5)];
                    }
                } else {
                    // Feature off (or just turned off this frame): every
                    // entry snaps straight back to LOD0. Cheap even on huge
                    // scenes (one field compare/write per entry) and matters
                    // for correctness — without it, an entry's level from a
                    // prior setAutoLod(true) session would persist forever
                    // (MeshEntry::lodLevel survives across lean frames).
                    uint32_t nonOverlay = 0;
                    for (auto& en : entries) {
                        if (en.isOverlay || en.isParticle) continue;
                        if (en.lodLevel != 0) lodChangedThisFrame_ = true;
                        en.lodLevel = 0;
                        ++nonOverlay;
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
                leanOk = true;
                for (size_t i = 0; i < prevSceneFingerprint.size() && leanOk; ++i) {
                    MeshFingerprint& fp = prevSceneFingerprint[i];
                    const MeshEntry& en = entries[i];
                    const bool xfmChanged =
                            std::memcmp(fp.matrix.data(), en.worldMatrix.data(), sizeof(fp.matrix)) != 0;
                    bool matChanged = false;
                    const unsigned int matVer = fp.matTyped ? fp.matTyped->version() : 0u;
                    if (matVer != fp.matVersion) {
                        Mesh* m = en.mesh;
                        if (albedoTexOf(*m) != fp.albedoTex ||
                            roughnessTexOf(*m) != fp.roughnessTex ||
                            metalnessTexOf(*m) != fp.metalnessTex ||
                            normalTexOf(*m) != fp.normalTex ||
                            transmissionTexOf(*m) != fp.transmissionTex ||
                            clearcoatTexOf(*m) != fp.clearcoatTex ||
                            clearcoatRoughnessTexOf(*m) != fp.clearcoatRoughnessTex ||
                            emissiveTexOf(*m) != fp.emissiveTex ||
                            occlusionTexOf(*m) != fp.occlusionTex) {
                            leanOk = false;// texture swap = STRUCTURAL — full path decides
                            break;
                        }
                        matChanged = true;
                        fp.matVersion = matVer;
                        const MaterialDesc md = materialFromMesh(*m);
                        fp.pbr = {md.albedo[0], md.albedo[1], md.albedo[2],
                                  md.roughness, md.metalness,
                                  md.emissive[0], md.emissive[1], md.emissive[2],
                                  md.emissiveIntensity,
                                  md.normalScale[0], md.normalScale[1],
                                  md.transmission, md.ior,
                                  md.clearcoat, md.clearcoatRoughness};
                    }
                    BufferGeometry* g = fp.geomTyped;
                    const unsigned int av = g->attributesVersion();
                    if (av != fp.attrVersion) {// attribute added/replaced/removed — re-cache
                        fp.attrVersion = av;
                        fp.posAttr  = g->getAttribute<float>("position");
                        fp.normAttr = g->getAttribute<float>("normal");
                        fp.uvAttr   = g->getAttribute<float>("uv");
                        fp.idxAttr  = g->getIndex();
                    }
                    unsigned int gv = 0;// must mirror geomVersionOf()
                    if (fp.posAttr)  gv += fp.posAttr->version;
                    if (fp.normAttr) gv += fp.normAttr->version;
                    if (fp.idxAttr)  gv += fp.idxAttr->version;
                    if (fp.uvAttr)   gv += fp.uvAttr->version;
                    const bool geomChanged = (gv != fp.geomVersion);
                    if (geomChanged) {
                        fp.geomVersion = gv;
                        // Particle billboard meshes mutate their attributes every
                        // frame but own no BLAS — flagging geomDirty would fire a
                        // per-frame vkDeviceWaitIdle for a refit that skips them
                        // anyway (blasCache miss). The billboard pass re-uploads
                        // their vertex cache itself, version-gated.
                        if (!en.isParticle) {
                            geomDirtyAny = true;
                            entryGeomDirty[i] = true;
                            // boundingBox invalidation — mirrors the generic loop.
                            if (auto gg = en.mesh->geometry()) gg->boundingBox.reset();
                            entries[i].lodSphereDirty = true;// auto-LOD's cached sphere follows
                        }
                    }
                    if (xfmChanged) {
                        matricesSame = false;
                        fp.matrix = en.worldMatrix;
                    }
                    if (matChanged) { materialValuesSame = false; entryMatDirty[i] = true; }
                    if (xfmChanged || matChanged || geomChanged) {
                        const size_t w = i >> 5;
                        if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                        meshMovedBits_[w] |= (1u << (i & 31u));
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
                        // bumping mat version — Object3D xfm is independent).
                        fp = p;
                        fp.matrix = en.worldMatrix;
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
                    fp.posAttr  = fp.geomTyped->getAttribute<float>("position");
                    fp.normAttr = fp.geomTyped->getAttribute<float>("normal");
                    fp.uvAttr   = fp.geomTyped->getAttribute<float>("uv");
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
            for (size_t i = 0; i < entries.size(); ++i) {
                if (!entries[i].isSkinned) continue;
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

            // DisplacedMesh — intrinsically dirty every frame (FFT spectrum
            // advances continuously). Same per-entry-bool pattern as bones,
            // routed through the cached isDisplaced flag instead of a fresh
            // dynamic_cast every frame.
            std::vector<bool> entryDisplacedDirty(entries.size(), false);
            for (size_t i = 0; i < entries.size(); ++i) {
                if (entries[i].isDisplaced) entryDisplacedDirty[i] = true;
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
                // Camera world position (translation column of matrixWorld).
                const auto& cw = camera.matrixWorld->elements;
                const Vector3 camPos(cw[12], cw[13], cw[14]);
                for (size_t i = 0; i < entries.size(); ++i) {
                    if (entries[i].isGrass) entryGrassDirty[i] = grassShouldAnimate(entries[i], camPos);
                }
            }

            // Morphed meshes — dirty when morphTargetInfluences changed.
            // Skinned meshes that also carry morph targets are handled by the
            // bone path above (GPU-skinned BLAS rebuild) so we skip them
            // here. Both predicates come from the cached fingerprint flags;
            // the getMorphAttributes hash lookup + SkinnedMesh dynamic_cast
            // used to run for every entry every frame.
            std::vector<bool> entryMorphDirty(entries.size(), false);
            for (size_t i = 0; i < entries.size(); ++i) {
                if (!entries[i].isMorphed || entries[i].isSkinned) continue;
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

            // LEAN path: the deformer dirty bits (bones / displaced / morph)
            // come from the scans above — fold them into the moved-bits mask
            // and the *DirtyAny flags exactly like the generic loop does.
            if (leanOk) {
                for (size_t i = 0; i < entries.size(); ++i) {
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
                if (!leanOk)
                for (size_t i = 0; i < currFp.size(); ++i) {
                    const auto& a = currFp[i];
                    const auto& b = prevSceneFingerprint[i];
                    if (a.mesh != b.mesh || a.geom != b.geom || a.mat != b.mat ||
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
                        // field mirror reads the last completed frame's
                        // readback here instead — one frame of latency.
                        const float now = static_cast<float>(glfwGetTime());
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (!entryDisplacedDirty[i]) continue;
                            auto* dm = static_cast<DisplacedMesh*>(entries[i].mesh);
                            auto stIt = displacedStates.find(dm);
                            if (stIt == displacedStates.end()) continue;
                            mirrorDisplacedHeightfields(*dm, *stIt->second);
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
                    for (size_t i = 0; i < entries.size(); ++i) {
                        if (!entries[i].isTet) continue;
                        auto tIt = tetMeshStates.find(entries[i].mesh);
                        if (tIt == tetMeshStates.end()) continue;
                        refreshTetBlas(*entries[i].mesh, *tIt->second);
                        tetDirtyAny = true;
                        const size_t w = i >> 5;
                        if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                        meshMovedBits_[w] |= (1u << (i & 31u));
                    }
                    // Geometries whose prevVertex was re-snapshotted (to OLD
                    // positions) this frame. The prevVertex re-sync pass below
                    // must SKIP these so their legitimate change-frame deformation
                    // motion survives — they settle on the next clean frame.
                    std::unordered_set<const BufferGeometry*> geomRefreshedThisFrame;
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
                        // them too.
                        check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (pre-BLAS-refresh)");
                        bool topologyChanged = false;
                        std::unordered_set<const BufferGeometry*> refreshedGeoms;
                        std::vector<GeomRefreshOp> refreshOps;
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

                            auto* posAttr = entries[i].mesh->geometry()->getAttribute<float>("position");
                            auto* idxAttr = entries[i].mesh->geometry()->getIndex();
                            if (!posAttr) continue;

                            const uint32_t curVtx = static_cast<uint32_t>(posAttr->count());
                            const uint32_t curIdx = idxAttr ? static_cast<uint32_t>(idxAttr->count()) : 0u;
                            if (curVtx != rec.vertexCount || curIdx != rec.indexCount) {
                                topologyChanged = true;
                                break;
                            }

                            // An in-place vertex rewrite invalidates any auto-LOD
                            // chain: the level BLASes BAKE positions (a stale level
                            // would ray-trace the pre-edit shape) and the chain's
                            // error bounds measured the old surface. The device-wide
                            // drain above makes the destroy safe. lodState=None ⇒
                            // the selection pass re-enqueues against the new
                            // geomVersion; selectLodGeom falls back to LOD0 for
                            // every consumer meanwhile. lodChangedThisFrame_ must be
                            // forced: selection already ran this frame and may have
                            // left en.lodLevel > 0 — the EFFECTIVE level changes to
                            // 0 right here, and without the flag the geomDescs GPU
                            // patch would skip while the TLAS falls back, leaving a
                            // stale per-level index address behind.
                            if (rec.lodState != BlasRecord::LodState::None) {
                                destroyBlasLodLevels(rec);
                                lodChangedThisFrame_ = true;
                            }
                            refreshOps.push_back({entries[i].mesh->geometry().get(), &rec});
                            refreshedGeoms.insert(geomKey);
                        }
                        if (topologyChanged) {
                            // Vertex/index count changed — can't reuse BLAS
                            // buffers. Fall through to the full structural
                            // rebuild path below.
                            goto fullRebuild;
                        }
                        refreshGeomBlasBatch(refreshOps);
                        for (const auto& op : refreshOps) geomRefreshedThisFrame.insert(op.geom);
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

                    if (!matricesSame || bonesDirtyAny || displacedDirtyAny || grassDirtyAny || tetDirtyAny || geomDirtyAny || morphDirtyAny || lodChangedThisFrame_) {
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
                        std::vector<VkAccelerationStructureInstanceKHR> instances;
                        instances.reserve(entries.size());
                        // instanceCustomIndex == entry index (matches the entries-
                        // indexed geomDescs/matDescs built in the full rebuild);
                        // overlay/skipped entries push no instance, exactly as
                        // their geomDescs/matDescs slots are left default.
                        for (size_t i = 0; i < entries.size(); ++i) {
                            const MeshEntry& en = entries[i];
                            if (en.isOverlay) continue;// raster-overlay only
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
                                const BufferGeometry* geomKey = en.mesh->geometry().get();
                                auto it = blasCache.find(geomKey);
                                if (it == blasCache.end()) continue;// shouldn't happen on transform-only
                                const auto lodSel = selectLodGeom(*it->second, en.lodLevel);
                                blasAddr = lodSel.asAddress;
                                // RT secondary hits (reflections/GI/lidar/probe update)
                                // read GeometryDesc::indexAddress keyed by gl_PrimitiveID
                                // from whichever BLAS this instance references — it must
                                // track the SAME level, or a hit against a coarser BLAS
                                // misindexes the still-LOD0 index buffer. `indexed` rides
                                // along: a level of a non-indexed soup record IS an
                                // indexed fetch. Patched into the buffer itself
                                // (vkDeviceWaitIdle-gated) below, only on a frame where a
                                // level actually changed.
                                if (i < geomDescsCached_.size()) {
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
                            // Same visibility-group rule as the full rebuild:
                            // blend/transmissive (non-water) → alpha mask so
                            // occlusion queries skip them.
                            inst.mask = kRayMaskOpaque;
                            if (i < matDescsCached_.size() && !en.isDisplaced) {
                                const auto& cmd = matDescsCached_[i];
                                if (cmd.transmission > 0.0f || cmd.alphaCutoff < 0.0f)
                                    inst.mask = kRayMaskAlpha;
                            }
                            inst.instanceShaderBindingTableRecordOffset = 0;
                            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                            inst.accelerationStructureReference = blasAddr;
                            instances.push_back(inst);
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
                            for (auto& d : geomDescsDirty_) d = true;
                        }
                        const bool blasDeformed = bonesDirtyAny || displacedDirtyAny || grassDirtyAny || tetDirtyAny || morphDirtyAny || geomDirtyAny || lodChangedThisFrame_;
                        // Stage the refit; recordCommandBuffer records it into the
                        // frame cb after the deformable BLAS rebuilds (no drain).
                        pendingTlasInstances_ = std::move(instances);
                        pendingTlasFullBuild_ = blasDeformed;
                        pendingTlasRefit_ = true;
                    }
                    if (!materialValuesSame) {
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
                        if (!patchMatDescs) matDescsCached_.assign(entries.size(), MaterialDesc{});
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
                            matDescsCached_[i] = md;
                        }
                        for (auto& d : matDescsDirty_) d = true;
                        cacheCullFlags(matDescsCached_);
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
                        destroyBlasLodLevels(*rec);
                        if (rec->as) ctx->rt().destroyAccelerationStructure(ctx->device(), rec->as, nullptr);
                        destroyBuffer(ctx->allocator(), rec->storage);
                        destroyBuffer(ctx->allocator(), rec->vertex);
                        destroyBuffer(ctx->allocator(), rec->index);
                        destroyBuffer(ctx->allocator(), rec->normal);
                        destroyBuffer(ctx->allocator(), rec->uv);
                        destroyBuffer(ctx->allocator(), rec->color);
                        destroyBuffer(ctx->allocator(), rec->prevVertex);
                        destroyBuffer(ctx->allocator(), rec->blasScratch);
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
                        destroyBuffer(ctx->allocator(), it->second->boneMatrices);
                        destroyBuffer(ctx->allocator(), it->second->blasScratch);
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
                        // Return the skinning descriptor set to the pool, else
                        // the slot leaks across remove/re-add cycles and the
                        // next allocateMeshDescriptorSet eventually hits
                        // VK_ERROR_OUT_OF_POOL_MEMORY.
                        if (it->second->skinDescSet != VK_NULL_HANDLE) {
                            skinning_->freeMeshDescriptorSet(it->second->skinDescSet);
                            it->second->skinDescSet = VK_NULL_HANDLE;
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
                        destroyBuffer(ctx->allocator(), it->second->tetPos);
                        vulkan::destroyExternalBuffer(ctx->device(), it->second->tetPosExt);
                        destroyBuffer(ctx->allocator(), it->second->blasScratch);
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
                        if (it->second->tetDescSet != VK_NULL_HANDLE) {
                            tetSkinning_->freeMeshDescriptorSet(it->second->tetDescSet);
                            it->second->tetDescSet = VK_NULL_HANDLE;
                        }
                        it = tetMeshStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = displacedStates.begin(); it != displacedStates.end(); ) {
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
                        if (st->scratchA.view  != VK_NULL_HANDLE) vkDestroyImageView(ctx->device(), st->scratchA.view, nullptr);
                        if (st->scratchA.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->scratchA.image, st->scratchA.alloc);
                        if (st->foamImage.view  != VK_NULL_HANDLE) vkDestroyImageView(ctx->device(), st->foamImage.view, nullptr);
                        if (st->foamImage.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->foamImage.image, st->foamImage.alloc);
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
                        // recordCommandBuffer refits it before the shade trace.
                        mirrorDisplacedHeightfields(*dm, *st);
                        pendingDisplacedDeforms_.emplace_back(dm, st, static_cast<float>(glfwGetTime()));
                        ++dm->frameTick;
                    } else {
                        // First creation: prime synchronously so the very first
                        // TLAS build + ray-trace see the displaced surface, not
                        // the rest grid.
                        refreshDisplacedBlas(*dm, *st, static_cast<float>(glfwGetTime()));
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
                        refreshGrassBlas(*gm, *st, static_cast<float>(glfwGetTime()));
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
                        if (it->second->geomVersion != curVer) {
                            auto& old = it->second;
                            // Topology/positions changed under this geometry
                            // pointer — any existing LOD chain simplified the
                            // OLD data and no longer matches. Destroy it; the
                            // fresh BlasRecord created below starts at
                            // LodState::None and the selection pass re-
                            // enqueues a new chain next time it's eligible.
                            destroyBlasLodLevels(*old);
                            if (old->as) ctx->rt().destroyAccelerationStructure(ctx->device(), old->as, nullptr);
                            destroyBuffer(ctx->allocator(), old->storage);
                            destroyBuffer(ctx->allocator(), old->vertex);
                            destroyBuffer(ctx->allocator(), old->index);
                            destroyBuffer(ctx->allocator(), old->normal);
                            destroyBuffer(ctx->allocator(), old->uv);
                            destroyBuffer(ctx->allocator(), old->color);
                            destroyBuffer(ctx->allocator(), old->prevVertex);
                            destroyBuffer(ctx->allocator(), old->blasScratch);
                            blasCache.erase(it);
                            // erase() returns the next bucket entry, not end().
                            // Force a fresh lookup so the "missing → build" branch
                            // below fires; otherwise recPtr binds to a sibling
                            // cache entry (different mesh's BLAS) and the TLAS
                            // instance ends up referencing the wrong AS.
                            it = blasCache.end();
                        }
                    }
                    if (it == blasCache.end()) {
                        auto rec = buildBlasFor(*m->geometry());
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
                bool warpReproject = false;
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
                gdesc._pad = 0;
                geomDescs[i] = gdesc;

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
                            (!en.isDisplaced && (md.transmission > 0.0f || md.alphaCutoff < 0.0f))
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
            for (auto& d : geomDescsDirty_) d = false;
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
            for (auto& d : matDescsDirty_) d = false;
            cacheCullFlags(matDescs);

            // Topology rebuild: the prev gbuf holds mesh IDs keyed by
            // gl_InstanceCustomIndexEXT (= entry order). If that ordering shifted,
            // those IDs no longer name the same surface, so the reproject mesh-ID
            // guard would accept stale taps — clearing the gbuf (→ guard misses
            // everywhere → histFc=0 globally, a full cold-start) is the safe path.
            //
            // But the dominant dynamic case is APPEND-ONLY: existing entries keep
            // their slots and new geometry is tacked on the end (a spawned PhysX
            // body, a grown ParticleSystem, streamed-in meshes). There, every
            // existing pixel's mesh ID is still valid and only the new entries lack
            // history — which the per-pixel mesh-ID guard already cold-starts on its
            // own. Clearing globally on every add throws away the whole scene's
            // converged accumulation (the visible reconverge flash + smeared motion
            // on every spawn). So detect the stable prefix and skip the reset.
            // prevWorldMats is keyed by (Mesh*, instanceIndex), so it stays valid
            // across an append too — new bodies are first-seen → identity motion.
            // Reorder / removal-from-the-middle shifts the prefix → falls back to
            // the clear.
            bool appendOnly = sceneBuilt_ && currFp.size() >= prevSceneFingerprint.size();
            for (size_t i = 0; appendOnly && i < prevSceneFingerprint.size(); ++i) {
                const auto& a = currFp[i];
                const auto& b = prevSceneFingerprint[i];
                if (a.mesh != b.mesh || a.geom != b.geom || a.mat != b.mat ||
                    a.instanceIndex != b.instanceIndex) {
                    appendOnly = false;
                }
            }
            if (!appendOnly) {
                // We're already past vkDeviceWaitIdle so the clear is synchronous.
                clearGbufImages();
                sampleIndex = 0;
                prevWorldMats.clear();
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
            rewriteDeferredDescriptors();
            // The (re)build above rebound the bindless texture array. The
            // raster descriptor's binding 3 mirrors that same table, so
            // invalidate its per-slot cache — each frame slot then re-writes
            // binding 3 on its next uploadRasterCameraUbo. This is the only
            // event that changes the table, so it's the only place the raster
            // mirror needs invalidating.
            rasterMatTexValid_.fill(0);
            prevSceneFingerprint = std::move(currFp);
            sceneBuilt_ = true;
        }

void VulkanRendererCore::CoreImpl::cacheCullFlags(const std::vector<MaterialDesc>& mds) {
            lastVisibleCullMode_.resize(mds.size());
            for (size_t i = 0; i < mds.size(); ++i) {
                switch (mds[i].sideMode) {
                    case 0:  lastVisibleCullMode_[i] = VK_CULL_MODE_BACK_BIT;  break;
                    case 1:  lastVisibleCullMode_[i] = VK_CULL_MODE_FRONT_BIT; break;
                    default: lastVisibleCullMode_[i] = VK_CULL_MODE_NONE;      break;
                }
            }
        }

void VulkanRendererCore::CoreImpl::collectWorldSprites(Object3D& scene) {
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

}// namespace threepp
