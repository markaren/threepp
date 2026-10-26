#include "VulkanCoreImpl.hpp"

#include "VulkanCpuPhaseProf.hpp"

#include "threepp/core/Assert.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"

#include <algorithm>
#include <cfloat>
#include <limits>
#include <typeindex>
#include <utility>
#include <unordered_map>

namespace threepp {

namespace {

    // Pre-order walk over the scene graph, as Object3D::traverseVisible, but a
    // hidden node's subtree is entered too, a LOD's hidden levels included: a
    // mesh hidden at run time keeps its entries (MeshEntry::hidden), and the
    // walk must reach it, under a hidden group or level too (a LOD's level
    // switch is then a flip). f(node, effectivelyVisible, underLod) returns
    // whether the node is a LOD.
    template<class F>
    void walkSceneGraph(Object3D& o, bool parentVisible, bool underLod, F& f) {
        const bool vis = parentVisible && o.visible;
        const bool isLod = f(o, vis, underLod);
        for (Object3D* c : o.children) walkSceneGraph(*c, vis, underLod || isLod, f);
    }

    // The same walk, calling post(node) once its subtree is done (the
    // expansion stamps SnapNode::subtreeEnd with it).
    template<class F, class P>
    void walkSceneGraph(Object3D& o, bool parentVisible, bool underLod, F& f, P& post) {
        const bool vis = parentVisible && o.visible;
        const bool isLod = f(o, vis, underLod);
        for (Object3D* c : o.children) walkSceneGraph(*c, vis, underLod || isLod, f, post);
        post(o);
    }

    // The casts the expansion walk asks of a node and of its material, answered
    // once per dynamic type: typeid is a vtable read, while a dynamic_cast
    // through Object3D's virtual base that fails walks the whole hierarchy
    // (about 0.1 us, and a plain mesh failed about fifteen of them per walk).
    // Only a cast the type is known to pass is still made, for its pointer.
    enum NodeKind : uint32_t {
        kNodeLod = 1u << 0,
        kNodeLine = 1u << 1,
        kNodeLineSegments = 1u << 2,
        kNodePoints = 1u << 3,
        kNodeMesh = 1u << 4,
        kNodeInstanced = 1u << 5,
        kNodeSplat = 1u << 6,
        kNodeSky = 1u << 7,
        kNodeSkinned = 1u << 8,
        kNodeDisplaced = 1u << 9,
        kNodeGrass = 1u << 10,
        kNodeField = 1u << 11,
        kNodeCamera = 1u << 12,
    };

    uint32_t nodeKind(Object3D& o) {
        static thread_local std::unordered_map<std::type_index, uint32_t> kinds;
        const std::type_index t(typeid(o));
        if (auto it = kinds.find(t); it != kinds.end()) return it->second;
        uint32_t k = 0u;
        if (dynamic_cast<LOD*>(&o)) k |= kNodeLod;
        if (auto* line = dynamic_cast<Line*>(&o)) {
            k |= kNodeLine;
            if (dynamic_cast<LineSegments*>(line)) k |= kNodeLineSegments;
        }
        if (dynamic_cast<Points*>(&o)) k |= kNodePoints;
        if (auto* m = dynamic_cast<Mesh*>(&o)) {
            k |= kNodeMesh;
            if (dynamic_cast<InstancedMesh*>(m)) k |= kNodeInstanced;
            if (dynamic_cast<SplatCloud*>(m)) k |= kNodeSplat;
            if (dynamic_cast<Sky*>(m)) k |= kNodeSky;
            if (dynamic_cast<SkinnedMesh*>(m)) k |= kNodeSkinned;
            if (dynamic_cast<DisplacedMesh*>(m)) k |= kNodeDisplaced;
            if (dynamic_cast<GrassMesh*>(m)) k |= kNodeGrass;
            if (dynamic_cast<ParticleField*>(m)) k |= kNodeField;
        }
        if (dynamic_cast<Camera*>(&o)) k |= kNodeCamera;
        kinds.emplace(t, k);
        return k;
    }

    enum MaterialKind : uint32_t {
        kMatWireframe = 1u << 0,
        kMatBasic = 1u << 1,
        kMatEmissive = 1u << 2,
        kMatDisplacement = 1u << 3,// MaterialWithDisplacementMap: may carry a displacementMap
    };

    uint32_t materialKind(const Material& m) {
        static thread_local std::unordered_map<std::type_index, uint32_t> kinds;
        const std::type_index t(typeid(m));
        if (auto it = kinds.find(t); it != kinds.end()) return it->second;
        uint32_t k = 0u;
        if (dynamic_cast<const MaterialWithWireframe*>(&m)) k |= kMatWireframe;
        if (dynamic_cast<const MeshBasicMaterial*>(&m)) k |= kMatBasic;
        if (dynamic_cast<const MaterialWithEmissive*>(&m)) k |= kMatEmissive;
        if (dynamic_cast<const MaterialWithDisplacementMap*>(&m)) k |= kMatDisplacement;
        kinds.emplace(t, k);
        return k;
    }

}// namespace

// The displacement a plain static mesh's material asks for, or nothing:
// a displacementMap with a CPU image, on a geometry with the normals and
// uv it moves along and reads at (see Impl::DisplacementSpec). Deformers
// own their vertex buffers and are left as they are.
bool VulkanRenderer::Impl::displacementSpecOf(Mesh& m, bool morphed, DisplacementSpec& d) const {
            if (morphed) return false;
            if (nodeKind(m) & (kNodeSkinned | kNodeDisplaced | kNodeGrass | kNodeField | kNodeSplat | kNodeSky))
                return false;
            const auto mat = m.material();
            if (!mat || !(materialKind(*mat) & kMatDisplacement) || mat->tetSkinning) return false;
            const auto* dm = dynamic_cast<const MaterialWithDisplacementMap*>(mat.get());
            if (!dm || !dm->displacementMap) return false;
            const auto geom = m.geometry();
            if (!geom || !geom->hasAttribute("normal") || !geom->hasAttribute("uv")) return false;
            Texture& tex = *dm->displacementMap;
            if (tex.matrixAutoUpdate) tex.updateMatrix();
            d = DisplacementSpec{};
            d.geom = geom;
            d.map = dm->displacementMap;
            d.scale = dm->displacementScale;
            d.bias = dm->displacementBias;
            std::copy(tex.matrix.elements.begin(), tex.matrix.elements.end(), d.uvTransform.begin());
            d.wrapS = tex.wrapS;
            d.wrapT = tex.wrapT;
            d.nearest = tex.magFilter == Filter::Nearest;
            // Refined (EdgeRefine) when the knob is on and the geometry is drawn
            // whole by one transform: an InstancedMesh has no one distance to
            // the focus, and groups / a draw range name the geometry's own
            // indices, which the refinement renumbers.
            d.refine = dispPixelEdge_ > 0.f && !(nodeKind(m) & kNodeInstanced) && geom->groups.empty() &&
                       geom->drawRange.start == 0 && geom->drawRange.count >= std::numeric_limits<int>::max() / 2;// BufferGeometry's default: all
            if (d.refine) {
                d.edge = dispCamera_;
                const auto& w = m.matrixWorld->elements;
                std::copy(w.begin(), w.end(), d.edge.world.begin());
                const float sx = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
                const float sy = std::sqrt(w[4] * w[4] + w[5] * w[5] + w[6] * w[6]);
                const float sz = std::sqrt(w[8] * w[8] + w[9] * w[9] + w[10] * w[10]);
                d.edge.reach = d.reach() * std::max({sx, sy, sz});
                d.edge.pixelEdge = dispPixelEdge_;
                d.edge.minEdge = dispMinEdge_;
                d.edge.frustum = dispFrustum_;
            }
            // FNV-1a over everything the displaced positions depend on.
            uint64_t h = 1469598103934665603ull;
            const auto mix = [&h](const void* p, size_t n) {
                const auto* c = static_cast<const unsigned char*>(p);
                for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
            };
            const Texture* tp = &tex;
            const unsigned int ver = tex.version();
            const unsigned int dims[2] = {tex.image().width(), tex.image().height()};
            const int modes[3] = {static_cast<int>(d.wrapS), static_cast<int>(d.wrapT), d.nearest ? 1 : 0};
            mix(&tp, sizeof(tp));
            mix(&ver, sizeof(ver));
            mix(dims, sizeof(dims));
            mix(&d.scale, sizeof(d.scale));
            mix(&d.bias, sizeof(d.bias));
            mix(d.uvTransform.data(), sizeof(float) * d.uvTransform.size());
            mix(modes, sizeof(modes));
            if (d.refine) {
                mix(&dispRefineEpoch_, sizeof(dispRefineEpoch_));
                mix(&d.edge.pixelEdge, sizeof(d.edge.pixelEdge));
                mix(&d.edge.minEdge, sizeof(d.edge.minEdge));
                mix(&d.edge.frustum, sizeof(d.edge.frustum));
            }
            d.signature = h | 1u;// never 0: 0 means "not displaced"
            return true;
        }

const VulkanRenderer::Impl::DisplacementSpec* VulkanRenderer::Impl::noteDisplacement(Mesh& m) {
            const BufferGeometry* g = m.geometry().get();
            DisplacementSpec spec;
            if (!displacementSpecOf(m, isMorphedMesh(m), spec)) {
                // Its material no longer displaces: the spec goes if this mesh
                // set it (another mesh sharing the geometry may have).
                if (const auto it = geomDisplacement_.find(g);
                    it != geomDisplacement_.end() && it->second.owner == &m && it->second.ownerId == m.id)
                    geomDisplacement_.erase(it);
                return nullptr;
            }
            spec.owner = &m;
            spec.ownerId = m.id;
            auto& slot = geomDisplacement_[g];
            static bool warned = false;
            if (!warned && slot.signature != 0u && slot.signature != spec.signature && slot.geom.lock().get() == g) {
                std::cerr << "[VulkanRenderer] one geometry drawn with two different displacementMaps: "
                             "the last mesh walked displaces it for both" << std::endl;
                warned = true;
            }
            slot = std::move(spec);
            return &slot;
        }

bool VulkanRenderer::Impl::displacementChanged(Mesh& m) {
            const auto mat = m.material();
            if (geomDisplacement_.empty() && (!mat || !(materialKind(*mat) & kMatDisplacement))) return false;
            const BufferGeometry* g = m.geometry().get();
            DisplacementSpec spec;
            const bool has = displacementSpecOf(m, isMorphedMesh(m), spec);
            const uint64_t now = has ? spec.signature : 0u;
            if (now == displacementSignatureOf(g)) return false;
            if (has) {
                spec.owner = &m;
                spec.ownerId = m.id;
                geomDisplacement_[g] = std::move(spec);
            }
            else geomDisplacement_.erase(g);
            return true;
        }

void VulkanRenderer::Impl::captureDisplacementFocus(Camera& camera) {
            // Where this build renders from, and what one output pixel spans
            // there: an angle under a perspective camera, a length under an
            // orthographic one.
            camera.updateMatrixWorld();
            Vector3 eye;
            eye.setFromMatrixPosition(*camera.matrixWorld);
            vulkan::ScreenEdgeTest& c = dispCamera_;
            c.eye = {eye.x, eye.y, eye.z};
            const auto& vi = camera.matrixWorldInverse.elements;
            std::copy(vi.begin(), vi.end(), c.view.begin());
            const float h = static_cast<float>(std::max(1u, ctx->swapchainExtent().height));
            c.pixelAngle = c.orthoPixel = c.halfX = c.halfY = 0.f;
            if (auto* pc = camera.as<PerspectiveCamera>()) {
                c.halfY = std::tan(0.5f * pc->getEffectiveFOV() * math::DEG2RAD);
                c.halfX = c.halfY * pc->aspect;
                c.pixelAngle = 2.f * c.halfY / h;
            } else if (auto* oc = camera.as<OrthographicCamera>()) {
                const float z = std::max(oc->zoom, 1e-6f);
                c.halfX = 0.5f * (oc->right - oc->left) / z;
                c.halfY = 0.5f * (oc->top - oc->bottom) / z;
                c.orthoPixel = 2.f * c.halfY / h;
            }
        }

uint32_t VulkanRenderer::Impl::snapMeshFlags(Mesh& m, const MaterialWithWireframe* wf) const {
            uint32_t fl = kSnapKindMesh;
            // geometry()/material() are virtual and hand back the shared_ptr by
            // value, so every call is a dispatch plus a refcount pair. This runs
            // once per mesh per expansion and nothing below it mutates the mesh,
            // so one fetch each serves the whole function.
            const auto geom = m.geometry();
            const auto mat = m.material();
            if (geom) {
                if (geom->hasAttribute("position")) fl |= kSnapHasPos;
                if (geom->hasAttribute("normal")) fl |= kSnapHasNorm;
            }
            if (wf && wf->wireframe) fl |= kSnapWire;
            if (overlayLayer_ >= 0 &&
                m.layers.isEnabled(static_cast<unsigned>(overlayLayer_))) fl |= kSnapOverlay;
            if (m.layers.isEnabled(VulkanRenderer::kSensorOnlyLayer)) fl |= kSnapSensorOnly;
            if (mat && mat->tetSkinning && mat->tetTexture) fl |= kSnapTet;
            // Unlit transparent untextured mesh → raster overlay routing (see
            // kSnapUiBlend). Textured basics stay traced — the overlay fill
            // pipelines have no sampler. Vertex-colored ones route too, via
            // the overlayMeshColoredPipelines / line-geometry-cache path
            // (drawOverlayMesh); before that path existed they stayed in the
            // G-buffer and rendered opaque, ignoring transparent/opacity.
            if (!(fl & kSnapWire) && mat && (materialKind(*mat) & kMatBasic)) {
                if (auto* mb = dynamic_cast<MeshBasicMaterial*>(mat.get());
                    mb && mb->transparent && !mb->map) fl |= kSnapUiBlend;
            }
            // ParticleSystem billboard mesh — detected by the unique material-name
            // marker (cheap: a length-mismatch reject for the empty-named common
            // case). Routed to the dedicated billboard pass and excluded from the
            // traced/rasterized scene.
            if (mat && mat->name == kParticleMaterialName) fl |= kSnapParticle;
            return fl;
        }

bool VulkanRenderer::Impl::hadEntriesBefore(SceneWalkOut& w, const Mesh& mesh) {
            // Meshes that had entries before this walk, by address and id
            // (eligible spans only, see MeshEntry::hidden): hidden now, they
            // keep their entries, hidden, so a visibility toggle is no
            // structural change. Built when the walk meets the first hidden mesh.
            if (!w.shownBeforeBuilt) {
                w.shownBeforeBuilt = true;
                for (const auto& s : entrySpans_) {
                    const MeshEntry& e0 = lastVisibleEntries_[s.first];
                    if (e0.isOverlay || e0.isParticleField) continue;
                    w.shownBefore.emplace(s.mesh, s.meshId);
                }
            }
            const auto it = w.shownBefore.find(&mesh);
            return it != w.shownBefore.end() && it->second == mesh.id;
        }

// An invisible parent hides its whole subtree: no entry under it except a
// mesh that keeps its own hidden ones (walkSceneGraph), matching three.js /
// GLRenderer. Plain `traverse` would leak children of hidden groups into the
// scene/overlay passes. Pushes exactly one SnapNode.
void VulkanRenderer::Impl::recordSceneNode(SceneWalkOut& w, Object3D& o, bool vis, bool underLod,
                                           Camera& camera) {
            // A hidden mesh has entries, hidden, when it had them before or
            // Object3D::resident asks for them up front (so its first show is
            // the same in-place flip as any later one). A LOD's level keeps the
            // entries it had (a switch back is a flip); resident does not apply
            // under a LOD.
            const auto entriesWhileHidden = [&](const Mesh& mesh, bool ul) {
                return (!ul && mesh.resident) || hadEntriesBefore(w, mesh);
            };
            SnapNode sn{};
            sn.obj = &o;
            sn.objId = o.id;
            sn.parent = o.parent;
            // three.js parity — LOD level selection (GLRenderer::
            // projectObject runs lod.update(camera) as it projects).
            // traverseVisible is PRE-ORDER: this LOD's children are
            // visited after this callback, so the rest of this walk
            // already sees the level picked for this frame's camera.
            // Recorded with a typed view + kSnapLod so the replay walk
            // re-runs the selection cast-free (Object3D is a virtual
            // base — the replay cannot static_cast down).
            const uint32_t kind = nodeKind(o);
            if (kind & kNodeLod) {
                auto* lod = dynamic_cast<LOD*>(&o);
                if (lod->autoUpdate && vis) lod->update(camera);
                sn.lod   = lod;
                sn.flags = kSnapKindOther | kSnapLod;
                w.snap.push_back(sn);
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
            if ((kind & kNodeLine) && vis) {
                auto* line = dynamic_cast<Line*>(&o);
                auto geom = line->geometry();
                sn.flags = kSnapKindLine | kSnapNoDelta;
                sn.line  = line;
                sn.geom  = geom.get();
                sn.geomB = geom.get();
                if (geom) sn.geomId = geom->id;
                if (geom) sn.attrVer = geom->attributesVersion();
                const bool matHidden = line->material() && !line->material()->visible;
                if (matHidden) sn.flags |= kSnapMatHidden;
                if (!matHidden && geom && geom->hasAttribute("position")) {
                    sn.flags |= kSnapHasPos;
                    LineEntry le{};
                    le.line       = line;
                    le.points     = nullptr;
                    le.isSegments = (kind & kNodeLineSegments) != 0u;
                    le.isPoints   = false;
                    std::memcpy(le.worldMatrix.data(),
                                line->matrixWorld->elements.data(), 64);
                    w.lines.push_back(le);
                }
                w.snap.push_back(sn);
                return;// Lines aren't Meshes; nothing more to do
            }
            // Points (point clouds) — never part of the traced/rasterized
            // scene, always rasterise as POINT_LIST in the overlay pass.
            // Shares the same geometry cache as Line via the
            // BufferGeometry* key.
            if ((kind & kNodePoints) && vis) {
                auto* pts = dynamic_cast<Points*>(&o);
                auto geom = pts->geometry();
                sn.flags = kSnapKindPoints | kSnapNoDelta;
                sn.pts   = pts;
                sn.geom  = geom.get();
                sn.geomB = geom.get();
                if (geom) sn.geomId = geom->id;
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
                    w.lines.push_back(le);
                }
                w.snap.push_back(sn);
                return;
            }
            auto* m = (kind & kNodeMesh) ? dynamic_cast<Mesh*>(&o) : nullptr;
            // Hidden: kind Other, pointer identity only, unless it has
            // entries while hidden (see entriesWhileHidden above).
            if (!m || (!vis && !entriesWhileHidden(*m, underLod))) {
                w.snap.push_back(sn);// kind Other: pointer identity only
                return;
            }
            // Hybrid overlay (raster-only) classification. Wireframe-
            // flagged materials and overlay-layer membership both route
            // the mesh to the post-TAA raster overlay pass and exclude
            // it from the traced/rasterized scene (TLAS, raster G-buffer,
            // emissive NEE). The scene render can't see/shadow overlay
            // meshes — they're pure debug visuals.
            const MaterialWithWireframe* wf = nullptr;
            if (auto mat = m->material(); mat && (materialKind(*mat) & kMatWireframe)) {
                wf = dynamic_cast<MaterialWithWireframe*>(mat.get());
            }
            auto* inst = (kind & kNodeInstanced) ? dynamic_cast<InstancedMesh*>(m) : nullptr;
            sn.mesh      = m;
            sn.inst      = inst;
            sn.geom      = m->geometry().get();
            sn.geomB     = m->geometry().get();
            sn.mat       = m->material().get();
            sn.geomId    = sn.geomB ? sn.geomB->id : 0u;
            sn.matId     = sn.mat ? sn.mat->id : 0u;
            sn.wf        = wf;
            sn.basic     = sn.mat && (materialKind(*sn.mat) & kMatBasic)
                                   ? dynamic_cast<const MeshBasicMaterial*>(sn.mat)
                                   : nullptr;
            sn.instCount = inst ? static_cast<int32_t>(inst->count()) : -1;
            sn.flags     = snapMeshFlags(*m, wf);
            // material()->visible == false: record the full mesh node (so a
            // re-show is caught by the replay walk) but build no entries, so
            // the TLAS / raster G-buffer / overlay / emissive NEE never see it.
            // Unless it has entries while hidden (entriesWhileHidden):
            // hidden either way, it has them, hidden (MeshEntry::hidden).
            // [[#mat-visible]]
            const bool matHidden = m->material() && !m->material()->visible;
            if (matHidden) sn.flags |= kSnapMatHidden;
            if (sn.geomB) sn.attrVer = sn.geomB->attributesVersion();
            w.snap.push_back(sn);
            const size_t snIdx = w.snap.size() - 1;
            const bool hidden = !vis || matHidden;
            if (hidden && !entriesWhileHidden(*m, underLod)) return;
            // SplatClouds are composited by the compute tile rasterizer
            // (SplatPass), not by the G-buffer. The unit quad they carry is
            // a GL-backend implementation detail — it has no normals and
            // describes no surface — so it is recorded in the snapshot (an
            // add/remove still invalidates the scene) and then dropped
            // before any entry, BLAS or TLAS instance is built from it.
            // The missing "normal" attribute would exclude it two lines
            // below anyway; saying so out loud is the difference between an
            // invariant and a coincidence.
            if (kind & kNodeSplat) return;
            // A threepp::Sky is the environment, not a mesh. Its dome is a
            // 1000 m box carrying a ShaderMaterial no deferred path shades;
            // as a mesh it sealed the scene in a dark room — the box's
            // inside on every pixel, every sun and GI ray stopped by it.
            // Noted for the env refresh (refreshEnvTextureFromScene bakes
            // it, vulkan/SkyBake) and dropped here, as a SplatCloud is.
            if (kind & kNodeSky) {
                w.snap[snIdx].flags |= kSnapNoDelta;
                if (!hidden) skySeen_ = dynamic_cast<Sky*>(m);// a resident hidden one gets this far
                return;
            }
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
            // One-shot type probes: an N-instance InstancedMesh pays them
            // once (nodeKind), not N times — and on snapshot-match frames
            // not at all (the cached entry flags are reused). Consumed by
            // raster pass loops, resolveBlasForEntry, TLAS refit, and
            // dirty-detection.
            const bool isSkinned   = (kind & kNodeSkinned) != 0u;
            const bool isDisplaced = (kind & kNodeDisplaced) != 0u;
            const bool isGrass     = (kind & kNodeGrass) != 0u;
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
            ParticleField* particleField = (kind & kNodeField) ? dynamic_cast<ParticleField*>(m) : nullptr;
            const bool isParticleField = (particleField != nullptr);
            // Their passes read no MeshEntry::hidden: hidden, they have no
            // entries, as before.
            if (isOverlay || isParticleField) w.snap[snIdx].flags |= kSnapNoDelta;
            if (hidden && (isOverlay || isParticleField)) return;
            const bool isMorphed   = isMorphedMesh(*m);
            // Tet-skinned PhysX soft body — detected via the material flag set by
            // SoftBody::enableGpuSkinning() (which also carries the per-frame tet
            // texture and the static tetIndex/tetWeight/tetRestInv* attributes).
            // Mutually exclusive with the other deformers.
            const bool isTet = !isSkinned && !isDisplaced && !isGrass && !isMorphed &&
                               m->material() && m->material()->tetSkinning &&
                               m->material()->tetTexture != nullptr;
            // Zero-copy vertex interop (enableVertexInterop). One blasCache
            // probe per MESH, not per instance — the same budget the type
            // probes above spend, and for the same payoff: the two
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
            // Under a level object of a manual LOD (manualLodLevelRoots_ holds
            // them; asked of the parents themselves, so a node recorded by the
            // edit-proportional walk, which keeps no such set, gets the same).
            for (Object3D* p = m->parent; !lodExempt && p; p = p->parent) {
                if (p->parent && (nodeKind(*p->parent) & kNodeLod)) lodExempt = true;
            }
            const MaterialWithEmissive* lodEmissive = nullptr;
            if (auto mat = m->material(); mat && (materialKind(*mat) & kMatEmissive)) {
                lodEmissive = dynamic_cast<MaterialWithEmissive*>(mat.get());
            }
            if (!geom->boundingBox) geom->computeBoundingBox();
            Vector3 lodCenter;
            float lodRadius = 0.f;
            if (geom->boundingBox) {
                lodCenter = geom->boundingBox->getCenter();
                lodRadius = geom->boundingBox->getSize().length() * 0.5f;
            }
            // A material displacementMap (noteDisplacement): the record's
            // positions move by up to its reach, so the bounds below grow by
            // it, and auto-LOD, which simplifies the geometry's own positions,
            // leaves the mesh alone.
            const DisplacementSpec* disp = noteDisplacement(*m);
            // The record's own box once it is built for this spec (boundsOf);
            // before that, the geometry's grown by the spec's reach.
            Box3 dispBox;
            bool dispExact = false;
            float dispReach = 0.f;
            if (disp) {
                lodExempt = true;
                if (const auto rit = blasCache.find(geom.get());
                    rit != blasCache.end() && rit->second->dispSig == disp->signature)
                    dispExact = boundsOf(*geom, dispBox);
                if (dispExact) {
                    lodCenter = dispBox.getCenter();
                    lodRadius = dispBox.getSize().length() * 0.5f;
                } else {
                    dispReach = disp->reach();
                    lodRadius += dispReach;
                }
            }
            // First-person viewmodel probe: is this mesh parented under a
            // Camera? Such geometry sits at the eye and must not occlude
            // (see MeshEntry::camAttached / kRayMaskNoShadow). One ancestor
            // walk per Mesh per full expansion.
            bool camAttached = false;
            for (auto* p = m->parent; p; p = p->parent) {
                if (nodeKind(*p) & kNodeCamera) {
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
            sp.first = static_cast<uint32_t>(w.built.size());
            sp.meshId = m->id;
            sp.hidden = hidden;
            // Expansion frames hand every span fresh matrices; leaving the
            // flag set keeps the motion-matrix build computing real deltas
            // (vs the remapped prev worlds) on rebuild frames — a spawn
            // burst must not blank the movers' motion vectors for a frame.
            sp.movedThisFrame = true;
            sp.meshWorld = {};
            std::memcpy(sp.meshWorld.data(), m->matrixWorld->elements.data(), 64);
            sp.localRadius = lodRadius;
            if (geom->boundingBox) {
                const Vector3 half = (dispExact ? dispBox.getSize() : geom->boundingBox->getSize()) * 0.5f;
                sp.localCenter[0] = lodCenter.x;
                sp.localCenter[1] = lodCenter.y;
                sp.localCenter[2] = lodCenter.z;
                sp.localHalf[0]   = half.x + dispReach;
                sp.localHalf[1]   = half.y + dispReach;
                sp.localHalf[2]   = half.z + dispReach;
                // A displacementMap record not built yet: the reach-grown box
                // stands in, and the next cull re-derives the exact one.
                sp.localBoundsValid = !disp || dispExact;
            }
            bool deferRows = false;
            if (inst && inst->count() > 0) {
                sp.inst           = inst;
                sp.instMatVersion = inst->instanceMatrix()->version;
                deferRows = w.lazyRows && hadEntriesBefore(w, *m);
                // Read the instance matrices straight off the attribute
                // rather than through getMatrixAt. The loop stops at
                // count(), and setCount() clamps count() to the buffer
                // capacity that getMatrixAt range-checks against, so that
                // check can only ever pass here; the assert keeps it in
                // debug builds in case the invariant is ever broken
                // upstream. Hoisting *m->matrixWorld likewise pulls a
                // shared_ptr dereference out of the per-instance loop.
                const auto& instArr = inst->instanceMatrix()->array();
                THREEPP_ASSERT(instArr.size() >= inst->count() * 16u);
                const Matrix4& meshWorld = *m->matrixWorld;
                Matrix4 instMat;
                Matrix4 world;
                const size_t jEnd = deferRows ? size_t(1) : inst->count();
                for (size_t j = 0; j < jEnd; ++j) {
                    std::memcpy(instMat.elements.data(), instArr.data() + j * 16u, 64);
                    world.multiplyMatrices(meshWorld, instMat);
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
                    e.isDisplacementMapped = disp != nullptr;
                    e.isInstanced  = true;
                    e.camAttached  = camAttached;
                    e.sensorOnly   = sensorOnly;
                    e.hidden       = hidden;
                    setLodCaches(e);
                    std::memcpy(e.worldMatrix.data(), world.elements.data(), 64);
                    w.built.push_back(e);
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
                e.isDisplacementMapped = disp != nullptr;
                e.isParticleField = isParticleField;
                e.camAttached  = camAttached;
                e.sensorOnly   = sensorOnly;
                e.hidden       = hidden;
                setLodCaches(e);
                std::memcpy(e.worldMatrix.data(), m->matrixWorld->elements.data(), 64);
                w.built.push_back(e);
            }
            sp.count = static_cast<uint32_t>(w.built.size()) - sp.first;
            if (deferRows) {
                sp.count = static_cast<uint32_t>(inst->count());
                sp.rowsDeferred = true;
                w.anyRowsDeferred = true;
            }
            if (isParticleField) {
                // The placeholder geometry's bounds describe a point at the
                // field's origin and say nothing about where the particles
                // are, so publish "no bounds" rather than a wrong box. The
                // frustum cull default-includes the span on the same flag.
                sp.localBoundsValid = false;
                sp.localRadius      = 0.f;
                sp.aabbValid        = false;
                w.fields.emplace_back(particleField, sp.first);
            }
            // A visibility flip of this mesh is applied in place by the
            // replay walk (setSpanHidden); overlays and fields re-expand.
            if (!isOverlay && !isParticleField) {
                w.snap[snIdx].spanIndex = static_cast<int32_t>(w.spans.size());
            }
            w.spans.push_back(sp);
            // Room to grow (VulkanCoreSlots.cpp): an InstancedMesh whose
            // count can rise is laid out with free slots after it, so a
            // setCount is a slot edit in place and not a move.
            if (!slotsDisabled() && !w.lazyRows && !isOverlay && !isParticle) {
                const uint32_t slack = slotSlackFor(sp, sp.count);
                if (slack > 0u) {
                    const uint32_t first = static_cast<uint32_t>(w.built.size());
                    w.built.resize(w.built.size() + slack, slotFreeEntry());
                    w.spans.push_back(slotFreeSpan(first, slack, 0u));
                }
            }
        }

bool VulkanRenderer::Impl::snapNodeUnchanged(const SnapNode& sn, Object3D& o, bool vis, bool underLod,
                                             Camera& camera, bool& flip, SnapFlip& f) {
            flip = false;
            const uint32_t kind = sn.flags & kSnapKindMask;
            const bool isLod = (sn.flags & kSnapLod) != 0u;
            // Recorded under a LOD or not: the walk enters the same subtrees.
            // Object3D::resident changed: the expansion decides which
            // hidden meshes have entries.
            if (((sn.flags & kSnapUnderLod) != 0u) != underLod ||
                ((sn.flags & kSnapResident) != 0u) != o.resident)
                return false;
            // Its effective visibility flipped. Only a mesh with entries
            // flips in place (below); any other node re-expands: a group's
            // own flip, a line, a mesh hidden before it was ever shown (no
            // entries yet, unless Object3D::resident), anything under a
            // LOD.
            const bool visFlip = ((sn.flags & kSnapHidden) != 0u) == vis;
            if (kind == kSnapKindOther) {
                if (visFlip) return false;
                // LOD parity (see the full pass): re-run level selection
                // every frame. Pre-order walk ⇒ the mutation lands before
                // this LOD's children are visited, so a level switch flips
                // a child's visibility under the LOD ⇒ a changed child ⇒ the
                // re-expansion picks up the new level's meshes.
                if (isLod && vis && sn.lod->autoUpdate) sn.lod->update(camera);
                return true;// otherwise pointer identity is all we need
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
                return !(visFlip || geom.get() != sn.geom || (geom && geom->id != sn.geomId) ||
                         (sn.geomB && sn.geomB->attributesVersion() != sn.attrVer) ||
                         matHidden != ((sn.flags & kSnapMatHidden) != 0u));
            }
            Mesh* m = sn.mesh;
            {
                const auto geomNow = m->geometry();
                const auto matNow = m->material();
                if (geomNow.get() != sn.geom || matNow.get() != sn.mat ||
                    (geomNow && geomNow->id != sn.geomId) ||
                    (matNow && matNow->id != sn.matId))
                    return false;
            }
            // An attribute added/replaced/removed: the full pass re-derives.
            if (sn.geomB && sn.geomB->attributesVersion() != sn.attrVer) return false;
            // A texture swapped on its material (a bumped version and a map
            // that is not the one its rows hold): the full pass, whose slot
            // edit places the mesh again. Only read on a version bump; a
            // values-only change stays on this path (the lean diff).
            if (sn.spanIndex >= 0 && sn.mat && static_cast<size_t>(sn.spanIndex) < entrySpans_.size()) {
                const MeshFingerprint& fp0 = prevSceneFingerprint[entrySpans_[sn.spanIndex].first];
                if (sn.mat->version() != fp0.matVersion && texSetOf(*m) != fp0.tex)
                    return false;
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
            if (wire != ((sn.flags & kSnapWire) != 0u) ||
                over != ((sn.flags & kSnapOverlay) != 0u) ||
                sensor != ((sn.flags & kSnapSensorOnly) != 0u) ||
                tet != ((sn.flags & kSnapTet) != 0u) ||
                particle != ((sn.flags & kSnapParticle) != 0u) ||
                uiBlend != ((sn.flags & kSnapUiBlend) != 0u))
                return false;
            if (sn.instCount >= 0 &&
                sn.inst->count() != static_cast<size_t>(sn.instCount))
                return false;
            // sn.mat compared equal above, so it's alive and dereferenceable
            // (nullptr ⇒ no material ⇒ treated as visible). [[#mat-visible]]
            const bool matHidden = sn.mat && !sn.mat->visible;
            if (visFlip || matHidden != ((sn.flags & kSnapMatHidden) != 0u)) {
                if (sn.spanIndex < 0) return false;
                // A span whose hidden admission has not landed has no record
                // and default rows: flipped visible in place it would draw
                // nothing until the pump lands it (a hole where a staged
                // terrain tile swaps in). Its first show takes the full
                // path, whose slot edit admits it as a visible mesh.
                if (vis && !matHidden && static_cast<size_t>(sn.spanIndex) < entrySpans_.size() &&
                    entrySpans_[sn.spanIndex].admitPending)
                    return false;
                flip = true;
                f.hidden = !vis;
                f.matHidden = matHidden;
            }
            return true;
        }

bool VulkanRenderer::Impl::sceneSnapshotMatches(Object3D& scene, Camera& camera) {
            if (!sceneBuilt_ || sceneSnapshot_.empty()) return false;
            if (prevSceneFingerprint.size() != lastVisibleEntries_.size()) return false;
            size_t cur = 0;
            bool ok = true;
            // Visibility flips of meshes that have entries, applied in place once
            // the WHOLE walk matched: a mismatch further on takes the full
            // re-expansion, which must see the entries as they were.
            static thread_local std::vector<SnapFlip> flips;
            flips.clear();
            auto visit = [&](Object3D& o, bool vis, bool underLod) -> bool {
                if (!ok) return false;
                // Address AND id: see SnapNode. `o` is live, so reading its id is
                // safe whatever used to be at this address.
                if (cur >= sceneSnapshot_.size() || sceneSnapshot_[cur].obj != &o ||
                    sceneSnapshot_[cur].objId != o.id) {
                    ok = false;
                    return false;
                }
                const size_t snIdx = cur;
                const SnapNode& sn = sceneSnapshot_[cur++];
                bool flip = false;
                SnapFlip f;
                if (!snapNodeUnchanged(sn, o, vis, underLod, camera, flip, f)) {
                    ok = false;
                    return false;
                }
                if (flip) {
                    f.snIdx = snIdx;
                    flips.push_back(f);
                }
                return (sn.flags & kSnapKindMask) == kSnapKindOther && (sn.flags & kSnapLod) != 0u;
            };
            walkSceneGraph(scene, true, false, visit);
            ok = ok && cur == sceneSnapshot_.size();
            if (ok) {
                for (const SnapFlip& f : flips) {
                    SnapNode& sn = sceneSnapshot_[f.snIdx];
                    sn.flags = (sn.flags & ~(kSnapHidden | kSnapMatHidden)) |
                               (f.hidden ? kSnapHidden : 0u) | (f.matHidden ? kSnapMatHidden : 0u);
                    setSpanHidden(static_cast<size_t>(sn.spanIndex), f.hidden || f.matHidden);
                }
            }
            return ok;
        }

bool VulkanRenderer::Impl::diffSceneWalk(Object3D& scene, Camera& camera, SceneDelta& d, bool useVersions) {
            THREEPP_CPUPROF("scene.3a_diffWalk");
            const auto& old = sceneSnapshot_;
            // A snapshot whose subtree ranges were not recorded, or of another
            // scene, or fingerprints not of the entries (the replay walk's own
            // precondition: snapNodeUnchanged reads them): the full expansion.
            if (prevSceneFingerprint.size() != lastVisibleEntries_.size()) return false;
            if (old.empty() || old[0].obj != &scene || old[0].objId != scene.id ||
                old[0].subtreeEnd != static_cast<uint32_t>(old.size()))
                return false;
            d.snap.reserve(old.size() + 64u);
            SceneWalkOut w{d.built, d.lines, d.spans, d.fields, d.snap};
            w.lazyRows = false;// a span recorded anew has every row
            diffSceneNode(d, w, scene, true, false, 0, -1, useVersions, camera);
            return !d.fallback && d.changed && d.lines.empty() && d.fields.empty();
        }

void VulkanRenderer::Impl::diffSceneRemoved(SceneDelta& d, size_t first, size_t end) {
            const auto& old = sceneSnapshot_;
            for (size_t k = first; k < end && !d.fallback; ++k) {
                const SnapNode& sn = old[k];
                // Under a LOD the expansion keeps no hidden level's entries: the
                // full path decides what a level switch leaves.
                if (sn.flags & kSnapNoDelta) {
                    d.fallback = true;
                    return;
                }
                if (sn.spanIndex >= 0) d.removed.push_back(static_cast<uint32_t>(sn.spanIndex));
            }
            if (end > first) d.changed = true;
        }

void VulkanRenderer::Impl::diffSceneNode(SceneDelta& d, SceneWalkOut& w, Object3D& o, bool parentVis,
                                         bool underLod, int64_t oldIdx, int32_t parentNewIdx, bool useVersions,
                                         Camera& camera) {
            if (d.fallback) return;
            const auto& old = sceneSnapshot_;
            const bool vis = parentVis && o.visible;
            const size_t newIdx = d.snap.size();
            bool isLod = false;
            bool kept = false;
            if (oldIdx >= 0) {
                const SnapNode& sn = old[static_cast<size_t>(oldIdx)];
                bool flip = false;
                SnapFlip f;
                // Under the parent it was recorded under (a moved node is a new
                // one here and a removed one there), and unchanged by the replay's
                // own test.
                if (sn.parent == o.parent && snapNodeUnchanged(sn, o, vis, underLod, camera, flip, f)) {
                    d.snap.push_back(sn);
                    if (flip) {
                        f.snIdx = newIdx;
                        d.flips.push_back(f);
                    }
                    isLod = (sn.flags & kSnapKindMask) == kSnapKindOther && (sn.flags & kSnapLod) != 0u;
                    kept = true;
                } else {
                    // Changed: recorded anew below; its span (if any) goes.
                    if (sn.flags & kSnapNoDelta) {
                        d.fallback = true;
                        return;
                    }
                    if (sn.spanIndex >= 0) d.removed.push_back(static_cast<uint32_t>(sn.spanIndex));
                }
            }
            if (!kept) {
                d.changed = true;
                const size_t spans0 = d.spans.size();
                recordSceneNode(w, o, vis, underLod, camera);
                SnapNode& rec = d.snap[newIdx];
                if (rec.flags & kSnapNoDelta) {
                    d.fallback = true;
                    return;
                }
                // The bits the expansion's visitRecord adds.
                if (!vis) rec.flags |= kSnapHidden;
                if (underLod) rec.flags |= kSnapUnderLod;
                if (o.resident) rec.flags |= kSnapResident;
                isLod = (rec.flags & kSnapLod) != 0u;// its children sit under it
                if (d.spans.size() != spans0) {
                    // One span, named by its node (an overlay's is not: kSnapNoDelta).
                    if (d.spans.size() != spans0 + 1u || rec.spanIndex != static_cast<int32_t>(spans0)) {
                        d.fallback = true;
                        return;
                    }
                    d.spanSnap.push_back(static_cast<uint32_t>(newIdx));
                }
                // A mesh with a span placed again with as many instances: what it
                // holds changed (a geometry, material or map swapped).
                if (oldIdx >= 0 && rec.spanIndex >= 0) {
                    const SnapNode& sn = old[static_cast<size_t>(oldIdx)];
                    if (sn.spanIndex >= 0 && sn.instCount == rec.instCount) ++d.swaps;
                }
            }
            d.snap[newIdx].ver = o.structureVersion();
            d.snap[newIdx].parentIdx = parentNewIdx;
            const bool entered = true;// a LOD's hidden levels too (walkSceneGraph)
            // Its old children: the nodes after it up to its subtree's end, each
            // child's subtree skipped by its own range.
            size_t oc = 0, oend = 0;
            if (oldIdx >= 0) {
                oc = static_cast<size_t>(oldIdx) + 1u;
                oend = old[static_cast<size_t>(oldIdx)].subtreeEnd;
            }
            if (entered) {
                const bool childUnderLod = underLod || isLod;
                for (Object3D* c : o.children) {
                    int64_t m = -1;
                    if (oc < oend) {
                        if (old[oc].obj == c && old[oc].objId == c->id) {
                            m = static_cast<int64_t>(oc);
                        } else {
                            // Further along: the old children before it are gone.
                            for (size_t k = old[oc].subtreeEnd; k < oend; k = old[k].subtreeEnd) {
                                if (old[k].subtreeEnd <= k) {
                                    d.fallback = true;
                                    return;
                                }
                                if (old[k].obj == c && old[k].objId == c->id) {
                                    m = static_cast<int64_t>(k);
                                    break;
                                }
                            }
                            if (m >= 0) diffSceneRemoved(d, oc, static_cast<size_t>(m));
                        }
                    }
                    if (m >= 0) oc = old[static_cast<size_t>(m)].subtreeEnd;
                    if (useVersions && m >= 0 && c->structureVersion() == old[static_cast<size_t>(m)].ver) {
                        // Nothing added, removed or swapped below it since: its
                        // nodes as they were, at their new place (indices
                        // shifted). Flags a plain field changes are settled by
                        // flatSnapshotCheck once the delta is applied.
                        const auto from = static_cast<size_t>(m), end = static_cast<size_t>(old[from].subtreeEnd);
                        const auto at = static_cast<int64_t>(d.snap.size());
                        const int64_t shift = at - static_cast<int64_t>(from);
                        d.snap.insert(d.snap.end(), old.begin() + static_cast<std::ptrdiff_t>(from),
                                      old.begin() + static_cast<std::ptrdiff_t>(end));
                        for (size_t k = static_cast<size_t>(at); k < d.snap.size(); ++k) {
                            SnapNode& cp = d.snap[k];
                            cp.subtreeEnd = static_cast<uint32_t>(static_cast<int64_t>(cp.subtreeEnd) + shift);
                            cp.parentIdx = k == static_cast<size_t>(at)
                                                   ? static_cast<int32_t>(newIdx)
                                                   : static_cast<int32_t>(cp.parentIdx + shift);
                        }
                        continue;
                    }
                    diffSceneNode(d, w, *c, vis, childUnderLod, m, static_cast<int32_t>(newIdx), useVersions, camera);
                    if (d.fallback) return;
                }
            }
            // The old children this walk did not meet (all of them when it does
            // not enter the node).
            if (oc < oend) diffSceneRemoved(d, oc, oend);
            d.snap[newIdx].subtreeEnd = static_cast<uint32_t>(d.snap.size());
        }

void VulkanRenderer::Impl::deltaWalkCheck(Object3D& scene, Camera& camera) {
            // A full recording of the scene as it is now, every node anew,
            // compared node for node with the snapshot and spans the delta left.
            std::vector<MeshEntry> built;
            std::vector<LineEntry> lines;
            std::vector<EntrySpan> spans;
            std::vector<std::pair<ParticleField*, uint32_t>> fields;
            std::vector<SnapNode> snap;
            SceneWalkOut w{built, lines, spans, fields, snap};
            w.lazyRows = false;
            std::vector<size_t> open;
            auto visit = [&](Object3D& o, bool vis, bool underLod) -> bool {
                recordSceneNode(w, o, vis, underLod, camera);
                SnapNode& rec = snap.back();
                rec.ver = o.structureVersion();
                rec.parentIdx = open.empty() ? -1 : static_cast<int32_t>(open.back());
                if (!vis) rec.flags |= kSnapHidden;
                if (underLod) rec.flags |= kSnapUnderLod;
                if (o.resident) rec.flags |= kSnapResident;
                open.push_back(snap.size() - 1);
                return (rec.flags & kSnapLod) != 0u;
            };
            auto post = [&](Object3D&) {
                snap[open.back()].subtreeEnd = static_cast<uint32_t>(snap.size());
                open.pop_back();
            };
            walkSceneGraph(scene, true, false, visit, post);
            uint64_t bad = 0;
            const auto mismatch = [&](const char* what, size_t k) {
                if (bad++ < 8) std::fprintf(stderr, "[VulkanRenderer] slot delta differs from a full walk: %s %zu\n", what, k);
            };
            if (snap.size() != sceneSnapshot_.size()) mismatch("node count", snap.size());
            for (size_t k = 0; k < std::min(snap.size(), sceneSnapshot_.size()); ++k) {
                const SnapNode& a = snap[k];
                const SnapNode& b = sceneSnapshot_[k];
                // A plain node's own hidden bit is only the replay's (its
                // descendants' flips are what draw): flatSnapshotCheck leaves it
                // for a node no mesh, line or LOD sits under.
                const uint32_t mask =
                        ((a.flags & kSnapKindMask) == kSnapKindOther && (a.flags & kSnapLod) == 0u) ? ~kSnapHidden : ~0u;
                if (a.obj != b.obj || a.objId != b.objId || a.parent != b.parent || a.subtreeEnd != b.subtreeEnd ||
                    (a.flags & mask) != (b.flags & mask) || a.geom != b.geom || a.mat != b.mat ||
                    a.instCount != b.instCount || a.ver != b.ver || a.parentIdx != b.parentIdx) {
                    if (bad < 8)
                        std::fprintf(stderr,
                                     "[VulkanRenderer]   node %zu: obj %d flags %x/%x ver %llu/%llu parent %d/%d "
                                     "end %u/%u span %d/%d\n",
                                     k, int(a.obj == b.obj), a.flags, b.flags, static_cast<unsigned long long>(a.ver),
                                     static_cast<unsigned long long>(b.ver), a.parentIdx, b.parentIdx, a.subtreeEnd,
                                     b.subtreeEnd, a.spanIndex, b.spanIndex);
                    mismatch("node", k);
                    continue;
                }
                if ((a.spanIndex < 0) != (b.spanIndex < 0)) {
                    mismatch("node's span", k);
                    continue;
                }
                if (a.spanIndex < 0) continue;
                const EntrySpan& sa = spans[static_cast<size_t>(a.spanIndex)];
                if (static_cast<size_t>(b.spanIndex) >= entrySpans_.size()) {
                    mismatch("span index", k);
                    continue;
                }
                const EntrySpan& sb = entrySpans_[static_cast<size_t>(b.spanIndex)];
                if (sb.isFree || sa.mesh != sb.mesh || sa.meshId != sb.meshId || sa.count != sb.count ||
                    sa.hidden != sb.hidden || sa.inst != sb.inst)
                    mismatch("span", k);
                else if (lastVisibleEntries_[sb.first].mesh != sb.mesh || lastVisibleEntries_[sb.first].hidden != sb.hidden)
                    mismatch("span's first row", k);
            }
            walkCheckMismatches_ += bad;
        }

bool VulkanRenderer::Impl::versionsDisabled() {
            static const bool off = [] {
                const char* e = std::getenv("THREEPP_VULKAN_NO_VERSIONS");
                return e && e[0] == '1';
            }();
            return off;
        }

bool VulkanRenderer::Impl::flatSnapshotCheck(Object3D& scene, Camera& camera) {
            if (versionsDisabled()) return false;
            auto& snap = sceneSnapshot_;
            if (!sceneBuilt_ || snap.empty() || snap[0].obj != &scene || snap[0].objId != scene.id ||
                snap[0].ver != scene.structureVersion())
                return false;
            if (prevSceneFingerprint.size() != lastVisibleEntries_.size()) return false;
            THREEPP_CPUPROF("scene.2b_flatCheck");
            constexpr uint32_t kMatBits = kSnapWire | kSnapTet | kSnapParticle | kSnapUiBlend;
            const auto routingOf = [](const SnapNode& sn) {
                const bool wire = sn.wf && sn.wf->wireframe;
                uint32_t bits = wire ? kSnapWire : 0u;
                if (sn.mat && sn.mat->tetSkinning && sn.mat->tetTexture != nullptr) bits |= kSnapTet;
                if (sn.mat && sn.mat->name == kParticleMaterialName) bits |= kSnapParticle;
                if (!wire && sn.basic && sn.basic->transparent && !sn.basic->map) bits |= kSnapUiBlend;
                return bits;
            };
            if (flatWatchGen_ != snapshotGen_) {
                // Meshes, lines, points, LOD nodes, and every node recorded
                // hidden: a mesh hidden before it was ever shown is recorded as
                // a plain node (no entries), and its show is a placement.
                flatWatch_.clear();
                flatMats_.clear();
                flatGeoms_.clear();
                std::unordered_map<const Material*, size_t> matAt;
                std::unordered_set<const BufferGeometry*> geomSeen;
                for (uint32_t k = 0; k < snap.size(); ++k) {
                    const SnapNode& sn = snap[k];
                    const uint32_t kind = sn.flags & kSnapKindMask;
                    if (kind == kSnapKindOther && (sn.flags & (kSnapLod | kSnapHidden)) == 0u) continue;
                    flatWatch_.push_back(k);
                    if (sn.geomB && geomSeen.insert(sn.geomB).second) flatGeoms_.emplace_back(sn.geomB, sn.attrVer);
                    if (kind != kSnapKindMesh || !sn.mat) continue;
                    auto [it, fresh] = matAt.emplace(sn.mat, flatMats_.size());
                    if (fresh) flatMats_.push_back({sn.mat, k, sn.flags & kMatBits, sn.mat->version(), {}});
                    if (sn.spanIndex >= 0) flatMats_[it->second].nodes.push_back(k);
                }
                flatWatchGen_ = snapshotGen_;
            }
            // Per geometry: an attribute added, replaced or removed.
            for (const auto& [g, ver] : flatGeoms_)
                if (g->attributesVersion() != ver) return false;
            // Per material: its routing, and a map swapped (a moved version,
            // then the replay's own test on each of its meshes).
            for (FlatMaterial& fm : flatMats_) {
                if (routingOf(snap[fm.first]) != fm.bits) return false;
                const unsigned int v = fm.mat->version();
                if (v == fm.version) continue;
                for (const uint32_t k : fm.nodes) {
                    const SnapNode& sn = snap[k];
                    if (static_cast<size_t>(sn.spanIndex) >= entrySpans_.size()) return false;
                    const MeshFingerprint& fp0 = prevSceneFingerprint[entrySpans_[sn.spanIndex].first];
                    const Mesh& m = *sn.mesh;
                    if (v != fp0.matVersion && texSetOf(m) != fp0.tex) return false;
                }
                fm.version = v;
            }
            flatVis_.assign(snap.size(), int8_t(-1));
            // The plain nodes whose hidden bit moved (applied with the flips).
            static thread_local std::vector<std::pair<uint32_t, bool>> groupFlips;
            static thread_local std::vector<SnapFlip> flips;
            static thread_local std::vector<uint32_t> chain;
            groupFlips.clear();
            flips.clear();
            // A node's effective visibility: its own flag and its ancestors',
            // each ancestor read once a frame.
            const auto effVis = [&](uint32_t k) -> bool {
                chain.clear();
                int32_t j = static_cast<int32_t>(k);
                while (j >= 0 && flatVis_[static_cast<size_t>(j)] < 0) {
                    chain.push_back(static_cast<uint32_t>(j));
                    j = snap[static_cast<size_t>(j)].parentIdx;
                }
                bool v = j < 0 ? true : flatVis_[static_cast<size_t>(j)] != 0;
                for (size_t c = chain.size(); c-- > 0;) {
                    const uint32_t n = chain[c];
                    v = v && snap[n].obj->visible;
                    flatVis_[n] = v ? 1 : 0;
                    const SnapNode& sn = snap[n];
                    if (n != k && (sn.flags & kSnapKindMask) == kSnapKindOther && (sn.flags & kSnapLod) == 0u &&
                        ((sn.flags & kSnapHidden) != 0u) == v)
                        groupFlips.emplace_back(n, !v);
                }
                return v;
            };
            const bool overlayOn = overlayLayer_ >= 0;
            for (const uint32_t k : flatWatch_) {
                const SnapNode& sn = snap[k];
                Object3D& o = *sn.obj;
                const bool vis = effVis(k);
                const uint32_t kind = sn.flags & kSnapKindMask;
                if (((sn.flags & kSnapResident) != 0u) != o.resident) return false;
                const bool visFlip = ((sn.flags & kSnapHidden) != 0u) == vis;
                if (kind == kSnapKindOther) {
                    // A node recorded hidden, or a LOD. A group (a LOD's level)
                    // shown or hidden is its descendants' flips, which their own
                    // nodes carry; a mesh recorded with no entries, or a LOD,
                    // takes the replay.
                    if (visFlip) {
                        if ((sn.flags & kSnapLod) != 0u || (nodeKind(o) & ~kNodeCamera) != 0u) return false;
                        groupFlips.emplace_back(k, !vis);
                    }
                    if ((sn.flags & kSnapLod) != 0u && vis && sn.lod->autoUpdate) sn.lod->update(camera);
                    continue;
                }
                if (kind != kSnapKindMesh) {
                    // A line or points (few): the replay's own test.
                    bool flip = false;
                    SnapFlip f;
                    if (!snapNodeUnchanged(sn, o, vis, (sn.flags & kSnapUnderLod) != 0u, camera, flip, f) || flip)
                        return false;
                    continue;
                }
                const bool over = overlayOn && o.layers.isEnabled(static_cast<unsigned>(overlayLayer_));
                const bool sensor = o.layers.isEnabled(VulkanRenderer::kSensorOnlyLayer);
                if (over != ((sn.flags & kSnapOverlay) != 0u) || sensor != ((sn.flags & kSnapSensorOnly) != 0u))
                    return false;
                if (sn.instCount >= 0 && sn.inst->count() != static_cast<size_t>(sn.instCount)) return false;
                const bool matHidden = sn.mat && !sn.mat->visible;
                if (!visFlip && matHidden == ((sn.flags & kSnapMatHidden) != 0u)) continue;
                // Hidden or shown: a flip of its span, as the replay's (a LOD's
                // level switch too); a span whose admission has not landed is
                // shown by a placement.
                if (sn.spanIndex < 0) return false;
                if (vis && !matHidden && static_cast<size_t>(sn.spanIndex) < entrySpans_.size() &&
                    entrySpans_[sn.spanIndex].admitPending)
                    return false;
                SnapFlip f;
                f.snIdx = k;
                f.hidden = !vis;
                f.matHidden = matHidden;
                flips.push_back(f);
            }
            for (const auto& [n, hidden] : groupFlips)
                snap[n].flags = hidden ? (snap[n].flags | kSnapHidden) : (snap[n].flags & ~kSnapHidden);
            for (const SnapFlip& f : flips) {
                SnapNode& sn = snap[f.snIdx];
                sn.flags = (sn.flags & ~(kSnapHidden | kSnapMatHidden)) | (f.hidden ? kSnapHidden : 0u) |
                           (f.matHidden ? kSnapMatHidden : 0u);
                setSpanHidden(static_cast<size_t>(sn.spanIndex), f.hidden || f.matHidden);
            }
            return true;
        }

void VulkanRenderer::Impl::setSpanHidden(size_t spanIndex, bool hidden) {
            EntrySpan& sp = entrySpans_[spanIndex];
            if (sp.hidden == hidden) return;
            sp.hidden = hidden;
            const size_t lastW = (size_t(sp.first) + sp.count - 1) >> 5;
            if (lastW >= meshMovedBits_.size()) meshMovedBits_.resize(lastW + 1, 0u);
            for (uint32_t i = sp.first; i < sp.first + sp.count; ++i) {
                lastVisibleEntries_[i].hidden = hidden;
                // Its pixels' history is the other state's: reproject + reject.
                meshMovedBits_[i >> 5] |= (1u << (i & 31u));
            }
            hiddenFlippedThisFrame_ = true;
            // Shown: the probe grid was fitted without it (the fit skips hidden
            // entries), so re-fit, as a structural show did. The grid's
            // hysteresis keeps it when the bounds barely move.
            if (!hidden) probeGridDirty_ = true;
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
// scrolling texture bumps their material version. Then it sent the entry ranges
// ensureSceneBuilt patched, which were still one desc per entry of a changed or
// streamed-in mesh. Now the array is the material table (one record per
// material), and the ranges are the records that changed.
void VulkanRenderer::Impl::flushMaterialDescsIfDirty(uint32_t frame) {
            THREEPP_CPUPROF("frame.I_uploadMatDesc");
            auto& dirty = matDescsDirty_[frame];
            if (!dirty.any()) return;
            if (matTable_.empty() ||
                materialDescsBuffers[frame].handle == VK_NULL_HANDLE) {
                dirty.clear();
                return;
            }
            uploadDescRanges(ctx->allocator(), materialDescsBuffers[frame],
                             matTable_, dirty);
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
            // Stage 4: a span draw_gen.comp writes is culled per instance there,
            // from this frame's planes; here it keeps its span-level answer.
            const bool gpuCull = drawGenCullsOnGpu();
            if (view().cullValidVersion == drawInputsVersion_ && view().cullGpuSpans == gpuCull &&
                cull.size() == lastVisibleEntries_.size() &&
                std::memcmp(view().prevCullVp, vp.elements.data(), 64) == 0) {
                return;
            }
            cull.assign(lastVisibleEntries_.size(), uint8_t{1});
            Frustum frustum;
            frustum.setFromProjectionMatrix(vp);
            const auto& planes = frustum.planes();
            auto& entries = lastVisibleEntries_;
            bool anyGpuCulled = false;

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
                // Hidden at run time (MeshEntry::hidden): drawn in no view.
                if (sp.hidden) {
                    std::fill_n(cull.begin() + sp.first, sp.count, uint8_t{0});
                    continue;
                }
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
                    Box3 box;// the record's box: a displacementMap's moved positions
                    if (!boundsOf(*geom, box)) continue;// keep(true)
                    const Vector3 c = box.getCenter();
                    const Vector3 h = box.getSize() * 0.5f;
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
                // Straddling, and culled per instance on the GPU: kept whole.
                if (gpuCull && drawGenEligibleRecord(sp)) {
                    anyGpuCulled = true;
                    continue;
                }
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
            } else if (anyGpuCulled && std::memcmp(view().prevCullVp, vp.elements.data(), 64) != 0) {
                // ...except where the GPU culls: there the planes are the input,
                // so a moved camera regenerates the records.
                ++view().cullVersion;
            }
            // Stamp the recompute gate: these results stay valid until the
            // frustum or the scene draw inputs change.
            view().cullValidVersion = drawInputsVersion_;
            view().cullGpuSpans = gpuCull;
            std::memcpy(view().prevCullVp, vp.elements.data(), 64);
        }

void VulkanRenderer::Impl::ensureSceneBuilt(Object3D& scene, Camera& camera) {
            // The whole scene build, every phase below included (the profiler
            // is flat: a nested phase adds to this key too).
            THREEPP_CPUPROF("scene.0_total");
            // displacementMap refinement: the focus, taken once (and again after
            // refineDisplacement), and whether this build re-checks every
            // displaced entry's spec (the epoch moved, see dispPixelEdge_).
            if (dispFocusPending_) {
                captureDisplacementFocus(camera);
                dispFocusPending_ = false;
            }
            const bool dispEpochNow = std::exchange(dispEpochPending_, false);
            // Deform work is queued here and recorded by the same frame. A frame
            // that never recorded (beginDeferredFrame gave up on an out-of-date
            // swapchain, a resize or a minimised window) leaves its queue behind,
            // and those Displaced/GrassMesh pointers may be freed by now and
            // their states pruned by this build: drop it. A skinned pose that
            // was never skinned is marked changed (its states are still alive,
            // only a structural rebuild below frees one), so it is redone.
            for (auto* st : pendingSkinnedRebuilds_) st->prevBoneMats.clear();
            pendingSkinnedRebuilds_.clear();
            pendingTetRebuilds_.clear();// re-queued every frame
            // The slot path's async batches that have finished.
            if (!asyncOneShots_.empty()) reapAsyncOneShots(/*deviceIdle=*/false);
            pendingDisplacedDeforms_.clear();// dirty every frame
            pendingGrassDeforms_.clear();
            grassPlacedThisFrame_.clear();
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
            hiddenFlippedThisFrame_ = false;
            slotsEditedThisFrame_ = false;

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
            // Ahead of it, when nothing in the scene was added, removed or
            // swapped (Object3D::structureVersion): no walk at all, only the
            // nodes a plain field can change (flatSnapshotCheck).
            bool snapLean = flatSnapshotCheck(scene, camera);
            if (snapLean) ++tlasStats_.flatChecks;
            static const bool flatCheck = [] {
                const char* v = std::getenv("THREEPP_VULKAN_WALK_CHECK");
                return v && v[0] == '1';
            }();
            if (snapLean && flatCheck) deltaWalkCheck(scene, camera);
            // A moved structure version: something was added, removed or
            // swapped, which the replay walk can only find to fail on.
            const bool structureMoved = !versionsDisabled() && sceneBuilt_ && !sceneSnapshot_.empty() &&
                                        sceneSnapshot_[0].obj == &scene &&
                                        sceneSnapshot_[0].ver != scene.structureVersion();
            if (!snapLean && !structureMoved) {
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
                refreshSpanMatrices();
                for (auto& le : lastVisibleLines_) {
                    const Object3D* src = le.line ? static_cast<const Object3D*>(le.line)
                                                  : static_cast<const Object3D*>(le.points);
                    std::memcpy(le.worldMatrix.data(), src->matrixWorld->elements.data(), 64);
                }
            } else {
            // The edit-proportional path first (diffSceneWalk, trySlotDelta):
            // when it takes the change, the full expansion below is skipped.
            bool deltaApplied = false;
            if (sceneBuilt_ && !slotsDisabled() && !deltaDisabled()) {
                // Only the subtrees whose structure version moved are visited
                // (when the scene's own did: otherwise the change is in a node's
                // state, which only a complete diff finds).
                const bool useVersions = !versionsDisabled() && !sceneSnapshot_.empty() &&
                                         sceneSnapshot_[0].obj == &scene &&
                                         sceneSnapshot_[0].ver != scene.structureVersion();
                SceneDelta delta;
                deltaApplied = diffSceneWalk(scene, camera, delta, useVersions) && trySlotDelta(delta);
                if (deltaApplied && useVersions) ++tlasStats_.versionedDeltas;
                if (deltaApplied && useVersions && !flatSnapshotCheck(scene, camera)) {
                    // A state change inside a subtree the versions skipped: the
                    // replay (flips) or one complete diff.
                    if (!sceneSnapshotMatches(scene, camera)) {
                        SceneDelta again;
                        deltaApplied = diffSceneWalk(scene, camera, again, false) && trySlotDelta(again);
                    }
                }
                static const bool deltaCheck = [] {
                    const char* v = std::getenv("THREEPP_VULKAN_WALK_CHECK");
                    return v && v[0] == '1';
                }();
                if (deltaApplied && deltaCheck) deltaWalkCheck(scene, camera);
            }
            if (!deltaApplied) {
            THREEPP_CPUPROF_NAMED(profExpand, "scene.3b_fullExpansion");
            // Expand the visible scene into one MeshEntry per TLAS instance.
            // Regular meshes contribute one entry; an InstancedMesh contributes
            // count() entries each with worldMatrix = matrixWorld * instanceMat[i].
            std::vector<MeshEntry> built;
            std::vector<LineEntry> builtLines;
            std::vector<EntrySpan> builtSpans;
            // A rebuild almost always lands within a few entries of the last
            // one — a rebuild fires on any add/remove/visibility toggle, which
            // in an editor or a streaming world is routine, not rare. The
            // members below are not moved-from until the very end of this
            // block, so they still hold the previous frame's counts here.
            // Without this a six-figure instanced scene grows from capacity 0
            // through ~17 reallocations, copying ~2N MeshEntry (~128 B each).
            // Capacity only — nothing is appended here, so entry ORDER, which
            // indexes the TLAS and the descriptor rings, is untouched.
            built.reserve(lastVisibleEntries_.size());
            builtLines.reserve(lastVisibleLines_.size());
            builtSpans.reserve(entrySpans_.size());
            // (field, its entry index). Rebuilt only here: a field appearing or
            // disappearing is a structural change, so the snapshot fast path
            // can't reach a frame where this list is stale.
            std::vector<std::pair<ParticleField*, uint32_t>> builtFields;
            // The last walk's nodes, kept for this one to reuse (reuseNode).
            prevSceneSnapshot_.swap(sceneSnapshot_);
            sceneSnapshot_.clear();
            sceneSnapshot_.reserve(prevSceneSnapshot_.size());
            // Re-found below if a visible Sky is still in the scene; a Sky
            // added, removed or toggled is a structural change, so the
            // snapshot fast path never serves a frame where this is stale.
            skySeen_ = nullptr;
            // Rebuilt below as the walk visits each threepp::LOD node — a
            // mesh entry whose parent chain hits one of these is exempt from
            // auto-LOD (see the selection pass just after this expansion).
            manualLodLevelRoots_.clear();
            // The walk's outputs, and its record of a node (recordSceneNode).
            SceneWalkOut w{built, builtLines, builtSpans, builtFields, sceneSnapshot_};
            // Lazy rows (VulkanCoreSlots.cpp): an InstancedMesh that already
            // has slots gets its first entry only, the one the slot path reads
            // its per-mesh facts from (the slot path derives any entry it adds
            // itself). A walk the structural rebuild consumes is never lazy: if
            // the slot path declines, the walk runs again in full below.
            w.lazyRows = sceneBuilt_ && !slotsDisabled();
            bool& lazyRows = w.lazyRows;
            bool& anyRowsDeferred = w.anyRowsDeferred;
            const auto recordNode = [&](Object3D& o, bool vis, bool underLod) {
                recordSceneNode(w, o, vis, underLod, camera);
            };
            // Reuse. A mesh node the last walk recorded, under the parent chain
            // it had, that passes the replay walk's unchanged-node test, gives
            // this walk the same snapshot node, span and first entry recordNode
            // would derive, from what the last walk derived: its class (type,
            // routing and material bits, all compared), its exemption and camera
            // attachment (the same ancestors), with the inputs that move without
            // a structural change read live (world matrices, bounds, the instance
            // count, the interop record). Everything else is recorded anew: other
            // kinds, overlays, fields, a visibility flip, a dirty LOD sphere.
            // Lazy walks only (a walk the rebuild consumes records everything).
            static const bool walkCheck = [] {
                const char* v = std::getenv("THREEPP_VULKAN_WALK_CHECK");
                return v && v[0] == '1';
            }();
            bool reuseNodes = lazyRows && !prevSceneSnapshot_.empty();
            // The last walk's node for each node met: this walk meets them in
            // the same pre-order wherever nothing changed, so a cursor into the
            // last snapshot finds each with no hashing; the index by address is
            // built only at the first node not where the cursor expects it (a
            // subtree added or removed), and the cursor resyncs on it. Was a
            // hash map of every node rebuilt and probed on every edit, plus a
            // hash set for the stable ancestors (the chain below): ~6 ms an edit
            // frame on a GeoScene pack's scene graph.
            size_t prevCursor = 0;
            bool prevIndexBuilt = false;
            // The walk's current chain of ancestors (pre-order: a node's parent
            // is on it), each with whether it is stable.
            struct WalkAncestor {
                const Object3D* obj;
                bool stable;
            };
            std::vector<WalkAncestor> walkChain;
            if (reuseNodes) walkChain.reserve(64);
            const auto prevNodeOf = [&](const Object3D& o) -> const SnapNode* {
                size_t idx = prevSceneSnapshot_.size();
                if (prevCursor < prevSceneSnapshot_.size() && prevSceneSnapshot_[prevCursor].obj == &o) {
                    idx = prevCursor;
                } else {
                    if (!prevIndexBuilt) {
                        prevIndexBuilt = true;
                        prevSnapIndex_.clear();
                        prevSnapIndex_.reserve(prevSceneSnapshot_.size());
                        for (uint32_t k = 0; k < prevSceneSnapshot_.size(); ++k)
                            prevSnapIndex_.emplace(prevSceneSnapshot_[k].obj, k);
                    }
                    if (const auto it = prevSnapIndex_.find(&o); it != prevSnapIndex_.end()) idx = it->second;
                }
                if (idx >= prevSceneSnapshot_.size()) return nullptr;
                prevCursor = idx + 1;
                const SnapNode& p = prevSceneSnapshot_[idx];
                return p.objId == o.id ? &p : nullptr;
            };
            const auto reuseNode = [&](const SnapNode& sn, Object3D& o, bool vis, bool underLod) -> bool {
                if ((sn.flags & kSnapKindMask) != kSnapKindMesh || sn.spanIndex < 0) return false;
                if (((sn.flags & kSnapUnderLod) != 0u) != underLod ||
                    ((sn.flags & kSnapResident) != 0u) != o.resident ||
                    ((sn.flags & kSnapHidden) != 0u) == vis)
                    return false;
                Mesh* m = sn.mesh;
                const auto geom = m->geometry();
                const auto mat = m->material();
                if (geom.get() != sn.geom || mat.get() != sn.mat || (geom && geom->id != sn.geomId) ||
                    (mat && mat->id != sn.matId))
                    return false;
                if (!geom || geom->attributesVersion() != sn.attrVer || !geom->boundingBox) return false;
                // The routing bits, as the replay reads them.
                const bool wire = sn.wf && sn.wf->wireframe;
                const bool over = overlayLayer_ >= 0 && o.layers.isEnabled(static_cast<unsigned>(overlayLayer_));
                const bool sensor = o.layers.isEnabled(VulkanRenderer::kSensorOnlyLayer);
                const bool tet = sn.mat && sn.mat->tetSkinning && sn.mat->tetTexture != nullptr;
                const bool particle = sn.mat && sn.mat->name == kParticleMaterialName;
                const bool uiBlend = !wire && sn.basic && sn.basic->transparent && !sn.basic->map;
                if (wire != ((sn.flags & kSnapWire) != 0u) || over != ((sn.flags & kSnapOverlay) != 0u) ||
                    sensor != ((sn.flags & kSnapSensorOnly) != 0u) || tet != ((sn.flags & kSnapTet) != 0u) ||
                    particle != ((sn.flags & kSnapParticle) != 0u) || uiBlend != ((sn.flags & kSnapUiBlend) != 0u))
                    return false;
                const bool matHidden = sn.mat && !sn.mat->visible;
                if (matHidden != ((sn.flags & kSnapMatHidden) != 0u)) return false;
                InstancedMesh* inst = sn.inst;
                if (inst && (inst->count() != static_cast<size_t>(sn.instCount) || inst->count() == 0u)) return false;
                if (static_cast<size_t>(sn.spanIndex) >= entrySpans_.size()) return false;
                const EntrySpan& os = entrySpans_[static_cast<size_t>(sn.spanIndex)];
                if (os.isFree || os.mesh != m || os.meshId != m->id || os.count == 0u) return false;
                const MeshEntry& oe = lastVisibleEntries_[os.first];
                if (oe.isOverlay || oe.isParticleField || oe.lodSphereDirty) return false;
                // A displacementMap mesh, before or now, takes the full walk,
                // which widens its bounds and exempts it from auto-LOD.
                if (oe.isDisplacementMapped || noteDisplacement(*m)) return false;
                // Exempt from auto-LOD: per object, or under a manual LOD level
                // (the same ancestors as before). The old answer settles it
                // unless it was "exempt" and the object no longer opts out.
                bool lodExempt = !m->autoLod;
                if (!lodExempt) {
                    if (oe.lodExemptStatic) return false;// an ancestor or the old opt-out: re-derive
                }
                const bool hidden = !vis || matHidden;

                SnapNode nsn = sn;
                nsn.spanIndex = static_cast<int32_t>(builtSpans.size());
                sceneSnapshot_.push_back(nsn);

                Vector3 lodCenter = geom->boundingBox->getCenter();
                const float lodRadius = geom->boundingBox->getSize().length() * 0.5f;
                bool isVertexInterop = false;
                if (!oe.isSkinned && !oe.isDisplaced && !oe.isGrass && !oe.isMorphed && !oe.isTet) {
                    auto ic = blasCache.find(geom.get());
                    isVertexInterop = (ic != blasCache.end() && ic->second->interop);
                }
                EntrySpan sp{};
                sp.mesh = m;
                sp.first = static_cast<uint32_t>(built.size());
                sp.meshId = m->id;
                sp.hidden = hidden;
                sp.movedThisFrame = true;
                sp.meshWorld = {};
                std::memcpy(sp.meshWorld.data(), m->matrixWorld->elements.data(), 64);
                sp.localRadius = lodRadius;
                {
                    const Vector3 half = geom->boundingBox->getSize() * 0.5f;
                    sp.localCenter[0] = lodCenter.x;
                    sp.localCenter[1] = lodCenter.y;
                    sp.localCenter[2] = lodCenter.z;
                    sp.localHalf[0]   = half.x;
                    sp.localHalf[1]   = half.y;
                    sp.localHalf[2]   = half.z;
                    sp.localBoundsValid = true;
                }
                MeshEntry e{};
                e.mesh = m;
                e.instanceIndex = 0u;
                e.isOverlay    = oe.isOverlay;
                e.isParticle   = oe.isParticle;
                e.isSkinned    = oe.isSkinned;
                e.isDisplaced  = oe.isDisplaced;
                e.isGrass      = oe.isGrass;
                e.isMorphed    = oe.isMorphed;
                e.isTet        = oe.isTet;
                e.isVertexInterop = isVertexInterop;
                e.isInstanced  = inst != nullptr;
                e.camAttached  = oe.camAttached;
                e.sensorOnly   = oe.sensorOnly;
                e.hidden       = hidden;
                e.lodExemptStatic = lodExempt;
                e.lodEmissive  = oe.lodEmissive;
                e.lodGeomKey   = geom.get();
                e.lodCenter[0] = lodCenter.x;
                e.lodCenter[1] = lodCenter.y;
                e.lodCenter[2] = lodCenter.z;
                e.lodRadius    = lodRadius;
                if (inst) {
                    sp.inst           = inst;
                    sp.instMatVersion = inst->instanceMatrix()->version;
                    Matrix4 instMat;
                    Matrix4 world;
                    std::memcpy(instMat.elements.data(), inst->instanceMatrix()->array().data(), 64);
                    world.multiplyMatrices(*m->matrixWorld, instMat);
                    std::memcpy(e.worldMatrix.data(), world.elements.data(), 64);
                    built.push_back(e);
                    sp.count = static_cast<uint32_t>(inst->count());
                    sp.rowsDeferred = true;
                    anyRowsDeferred = true;
                } else {
                    std::memcpy(e.worldMatrix.data(), m->matrixWorld->elements.data(), 64);
                    built.push_back(e);
                    sp.count = 1u;
                }
                builtSpans.push_back(sp);
                return true;
            };
            // Each node's subtree end (SnapNode::subtreeEnd): the walk's stack of
            // the nodes whose subtrees are open.
            std::vector<size_t> snapOpen;
            const auto postRecord = [&](Object3D&) {
                sceneSnapshot_[snapOpen.back()].subtreeEnd = static_cast<uint32_t>(sceneSnapshot_.size());
                snapOpen.pop_back();
            };
            auto visitRecord = [&](Object3D& o, bool vis, bool underLod) -> bool {
                bool reused = false;
                if (reuseNodes) {
                    const SnapNode* prev = prevNodeOf(o);
                    while (!walkChain.empty() && walkChain.back().obj != o.parent) walkChain.pop_back();
                    const bool parentStable =
                            o.parent == nullptr || (!walkChain.empty() && walkChain.back().stable);
                    // Stable: met last time under the same parent, itself stable.
                    const bool stable = prev && prev->parent == o.parent && parentStable;
                    if (stable) reused = reuseNode(*prev, o, vis, underLod);
                    walkChain.push_back({&o, stable});
                }
                if (!reused) recordNode(o, vis, underLod);// pushes exactly one SnapNode
                SnapNode& rec = sceneSnapshot_.back();
                rec.ver = o.structureVersion();
                rec.parentIdx = snapOpen.empty() ? -1 : static_cast<int32_t>(snapOpen.back());
                snapOpen.push_back(sceneSnapshot_.size() - 1);
                if (!vis) rec.flags |= kSnapHidden;
                if (underLod) rec.flags |= kSnapUnderLod;
                if (o.resident) rec.flags |= kSnapResident;
                return (rec.flags & kSnapLod) != 0u;
            };
            walkSceneGraph(scene, true, false, visitRecord, postRecord);
            profExpand.stop();
            if (reuseNodes && walkCheck) {
                // The same walk without reuse, compared node for node.
                auto snapA = std::move(sceneSnapshot_);
                auto builtA = std::move(built);
                auto spansA = std::move(builtSpans);
                auto linesA = std::move(builtLines);
                auto fieldsA = std::move(builtFields);
                auto skyA = skySeen_;
                auto rootsA = manualLodLevelRoots_;
                const bool deferredA = anyRowsDeferred;
                sceneSnapshot_.clear();
                built.clear();
                builtSpans.clear();
                builtLines.clear();
                builtFields.clear();
                skySeen_ = nullptr;
                manualLodLevelRoots_.clear();
                anyRowsDeferred = false;
                reuseNodes = false;
                walkSceneGraph(scene, true, false, visitRecord, postRecord);
                uint64_t bad = 0;
                const auto mismatch = [&](const char* what, size_t k) {
                    if (bad++ < 8) std::fprintf(stderr, "[VulkanRenderer] walk reuse differs: %s %zu\n", what, k);
                };
                // Field by field: copies leave padding undefined, and a span
                // carries vectors.
                const auto sameSnap = [](const SnapNode& a, const SnapNode& b) {
                    return a.obj == b.obj && a.parent == b.parent && a.objId == b.objId && a.geomId == b.geomId &&
                           a.matId == b.matId && a.mesh == b.mesh && a.inst == b.inst && a.line == b.line &&
                           a.pts == b.pts && a.lod == b.lod && a.geom == b.geom && a.geomB == b.geomB &&
                           a.mat == b.mat && a.wf == b.wf && a.basic == b.basic && a.instCount == b.instCount &&
                           a.spanIndex == b.spanIndex && a.flags == b.flags && a.attrVer == b.attrVer;
                };
                const auto sameRow = [](const MeshEntry& a, const MeshEntry& b) {
                    return a.mesh == b.mesh && a.worldMatrix == b.worldMatrix && a.instanceIndex == b.instanceIndex &&
                           a.isOverlay == b.isOverlay && a.isParticle == b.isParticle && a.hidden == b.hidden &&
                           a.camAttached == b.camAttached && a.sensorOnly == b.sensorOnly &&
                           a.isSkinned == b.isSkinned && a.isDisplaced == b.isDisplaced && a.isGrass == b.isGrass &&
                           a.isMorphed == b.isMorphed && a.isTet == b.isTet && a.isParticleField == b.isParticleField &&
                           a.isVertexInterop == b.isVertexInterop && a.isInstanced == b.isInstanced &&
                           a.isDisplacementMapped == b.isDisplacementMapped &&
                           a.lodLevel == b.lodLevel && a.rayFarOpaque == b.rayFarOpaque &&
                           a.lodExemptStatic == b.lodExemptStatic && a.lodSphereDirty == b.lodSphereDirty &&
                           a.lodEmissive == b.lodEmissive && a.lodGeomKey == b.lodGeomKey &&
                           std::memcmp(a.lodCenter, b.lodCenter, sizeof(a.lodCenter)) == 0 && a.lodRadius == b.lodRadius;
                };
                const auto sameSpan = [](const EntrySpan& a, const EntrySpan& b) {
                    return a.mesh == b.mesh && a.inst == b.inst && a.first == b.first && a.count == b.count &&
                           a.meshId == b.meshId && a.hidden == b.hidden && a.meshWorld == b.meshWorld &&
                           a.instMatVersion == b.instMatVersion && a.instColorVersion == b.instColorVersion &&
                           a.fadeKnown == b.fadeKnown && a.movedThisFrame == b.movedThisFrame &&
                           a.instancesMoved == b.instancesMoved &&
                           a.lodNonZero == b.lodNonZero && a.aabbValid == b.aabbValid &&
                           std::memcmp(a.localCenter, b.localCenter, sizeof(a.localCenter)) == 0 &&
                           std::memcmp(a.localHalf, b.localHalf, sizeof(a.localHalf)) == 0 &&
                           a.localRadius == b.localRadius && a.localBoundsValid == b.localBoundsValid &&
                           a.isFree == b.isFree && a.rowsDeferred == b.rowsDeferred &&
                           a.chunkAabbs.size() == b.chunkAabbs.size() && a.probeChunks.size() == b.probeChunks.size();
                };
                if (snapA.size() != sceneSnapshot_.size()) mismatch("snapshot size", snapA.size());
                for (size_t k = 0; k < std::min(snapA.size(), sceneSnapshot_.size()); ++k)
                    if (!sameSnap(snapA[k], sceneSnapshot_[k])) mismatch("snapshot node", k);
                if (builtA.size() != built.size()) mismatch("row count", builtA.size());
                for (size_t k = 0; k < std::min(builtA.size(), built.size()); ++k)
                    if (!sameRow(builtA[k], built[k])) mismatch("row", k);
                if (spansA.size() != builtSpans.size()) mismatch("span count", spansA.size());
                for (size_t k = 0; k < std::min(spansA.size(), builtSpans.size()); ++k)
                    if (!sameSpan(spansA[k], builtSpans[k])) mismatch("span", k);
                if (linesA.size() != builtLines.size()) mismatch("line count", linesA.size());
                if (fieldsA != builtFields) mismatch("fields", fieldsA.size());
                if (skyA != skySeen_) mismatch("sky", 0);
                if (rootsA != manualLodLevelRoots_) mismatch("manual LOD roots", rootsA.size());
                if (deferredA != anyRowsDeferred) mismatch("deferred rows", 0);
                walkCheckMismatches_ += bad;
                sceneSnapshot_ = std::move(snapA);
                built = std::move(builtA);
                builtSpans = std::move(spansA);
                builtLines = std::move(linesA);
                builtFields = std::move(fieldsA);
                skySeen_ = skyA;
                manualLodLevelRoots_ = std::move(rootsA);
                anyRowsDeferred = deferredA;
                reuseNodes = true;
            }
            // Stable entry slots (VulkanCoreSlots.cpp): meshes added or
            // removed, a count changed or a first show are written into the
            // entry slots in place, and the frame goes on as a lean one.
            if (trySlotEdit(built, builtSpans, builtFields)) {
                lastVisibleLines_ = std::move(builtLines);
            } else {
            if (lazyRows) {
                // The structural rebuild reads every entry: walk again, in
                // full and with the room to grow laid out.
                THREEPP_CPUPROF("scene.3b_fullExpansion");
                lazyRows = false;
                reuseNodes = false;
                anyRowsDeferred = false;
                built.clear();
                builtLines.clear();
                builtSpans.clear();
                builtFields.clear();
                sceneSnapshot_.clear();
                skySeen_ = nullptr;
                manualLodLevelRoots_.clear();
                walkSceneGraph(scene, true, false, visitRecord, postRecord);
            }
            // The same entries as before with some hidden or shown is no
            // structural change (the fingerprints match), but the TLAS masks,
            // culls, draw records and emitters must re-read the flags.
            if (built.size() == lastVisibleEntries_.size()) {
                for (size_t i = 0; i < built.size(); ++i) {
                    if (built[i].hidden == lastVisibleEntries_[i].hidden) continue;
                    const size_t w = i >> 5;
                    if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                    meshMovedBits_[w] |= (1u << (i & 31u));
                    hiddenFlippedThisFrame_ = true;
                    if (!built[i].hidden) probeGridDirty_ = true;// shown: see setSpanHidden
                }
            }
            lastVisibleEntries_ = std::move(built);
            lastVisibleLines_   = std::move(builtLines);
            entrySpans_         = std::move(builtSpans);
            ++spansVersion_;
            particleFields_     = std::move(builtFields);
            }
            }// !deltaApplied
            ++snapshotGen_;
            probeGridDirty_     = true;// scene structure changed → re-fit the probe grid
            ++lodLevelVersion_;// the entries are laid out anew
            // ...and re-bake every ParticleField surface height map (F5). The
            // set of things a flake can land on is exactly what just changed,
            // and this is the trigger the plan names ("the same trigger as
            // entry rebuild"). Cheap: it bumps a counter, and only the fields
            // that actually use a map pay for a re-trace.
            if (particleFieldPass_) particleFieldPass_->invalidateSurfaceBakes();
            cutoutRayClassified_ = false;// the fresh entries' rayFarOpaque bits are all reset
            }// !snapLean
            // A slot edit left entries, spans and fingerprints in the shape a
            // lean frame starts from: from here on this frame IS one, with the
            // spans it did not touch refreshed and diffed as on any other.
            if (slotsEditedThisFrame_) {
                snapLean = true;
                THREEPP_CPUPROF("scene.3_leanMatrixRefresh");
                refreshSpanMatrices();
            }
            // Hidden meshes' admissions that are ready land here, on a frame
            // that ends lean (never ahead of a structural rebuild, whose old
            // spans may name meshes that are gone). VulkanCoreSlots.cpp.
            if (snapLean) {
                pumpHiddenAdmissions();
                if (hiddenFlippedThisFrame_) stableIdsPending_ = true;
                refreshPendingStableIds();
            }

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
            if (autoLod_) {
                THREEPP_CPUPROF("scene.4a_lodDrain");
                drainLodResults();// budget: 2 geoms / 1 MiB of new levels per frame
            }
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
                if (lodChangedThisFrame_) ++lodLevelVersion_;
            }

            // Camera position for the cutout ray distance (setCutoutRayDistance):
            // read by the lean path's classifyCutoutFar and by the full
            // rebuild's instance loop.
            float rayCamPos[3];
            {
                Vector3 p;
                p.setFromMatrixPosition(*camera.matrixWorld);
                rayCamPos[0] = p.x;
                rayCamPos[1] = p.y;
                rayCamPos[2] = p.z;
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
                        if (texSetOf(*m) != fp0.tex) {
                            leanOk = false;// texture swap = STRUCTURAL — full path decides
                            break;
                        }
                        matChanged = true;
                        const std::array<float, 15> pbr = pbrKeyOf(materialFromMesh(*m));
                        for (uint32_t j = 0; j < sp.count; ++j) {
                            auto& fp = prevSceneFingerprint[sp.first + j];
                            fp.matVersion = matVer;
                            fp.pbr = pbr;
                        }
                    }
                    BufferGeometry* g = fp0.geomTyped;
                    const unsigned int av = g->attributesVersion();
                    if (av != fp0.attrVersion) {// attribute added/replaced/removed — re-cache
                        const GeomAttrRefs attrs = GeomAttrRefs::of(*g);
                        for (uint32_t j = 0; j < sp.count; ++j) {
                            auto& fp = prevSceneFingerprint[sp.first + j];
                            fp.attrVersion = av;
                            fp.attrs = attrs;
                        }
                    }
                    const unsigned int gv = fp0.attrs.versionSum();// = geomVersionOf(*g)
                    const bool geomChanged = (gv != fp0.geomVersion);
                    // A material edit that moved its displacementMap re-displaces
                    // the geometry (see the generic compare).
                    const bool dispMoved = (matChanged || (dispEpochNow && entries[sp.first].isDisplacementMapped)) &&
                                           !entries[sp.first].isParticle && displacementChanged(*sp.mesh);
                    if (geomChanged || dispMoved) {
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
                            sp.probeChunksValid = 0;
                        }
                    }
                    if (xfmChanged) matricesSame = false;// fp.matrix refreshed with the span above
                    if (matChanged) {
                        materialValuesSame = false;
                        for (uint32_t j = 0; j < sp.count; ++j)
                            entryMatDirty[sp.first + j] = true;
                    }
                    // An instance-matrix edit alone marked its own entries
                    // (refreshSpanMatrices, EntrySpan::instancesMoved).
                    if ((xfmChanged && !sp.instancesMoved) || matChanged || geomChanged) {
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
                MaterialTexSet tex;
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
                        p.meshId == m->id && p.geomId == m->geometry()->id &&
                        p.matId == (matPtr ? matPtr->id : 0u) &&
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
                    fp.meshId = m->id;
                    fp.geomId = m->geometry()->id;
                    fp.matId  = matPtr ? matPtr->id : 0u;
                    fp.matVersion = matVer;
                    fp.geomVersion = geomVer;
                    fp.matTyped  = matPtr;
                    fp.geomTyped = m->geometry().get();
                    fp.attrVersion = fp.geomTyped->attributesVersion();
                    fp.attrs = GeomAttrRefs::of(*fp.geomTyped);
                    const Material* matKey = matPtr;
                    auto memoIt = fpMemo.find(matKey);
                    if (memoIt == fpMemo.end()) {
                        FpMatDerived d{texSetOf(*m), pbrKeyOf(materialFromMesh(*m))};
                        memoIt = fpMemo.emplace(matKey, std::move(d)).first;
                    }
                    const FpMatDerived& d = memoIt->second;
                    fp.tex           = d.tex;
                    fp.instanceIndex = en.instanceIndex;
                    fp.overlay       = en.isOverlay;
                    fp.matrix        = en.worldMatrix;
                    fp.pbr           = d.pbr;
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
                    // A field the slot path admitted takes its first deform here.
                    // One whose state was built for a geometry it no longer draws
                    // (swapped, its new state still to be admitted) takes none.
                    // A hidden one takes none (grassWantsDeform).
                    auto* gm = static_cast<GrassMesh*>(entries[sp.first].mesh);
                    const auto stIt = grassStates.find(gm);
                    if (stIt == grassStates.end() || stIt->second->liveCheck.lock() != gm->geometry()) continue;
                    for (uint32_t j = 0; j < sp.count; ++j) {
                        const size_t i = size_t(sp.first) + j;
                        entryGrassDirty[i] = grassWantsDeform(*stIt->second, entries[i].hidden,
                                                              grassShouldAnimate(entries[i], camPos));
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
                if (entries[sp.first].isMorphed && entries[sp.first].isSkinned) {
                    // skinning.comp reads no morph deltas, so the bone path
                    // drops the targets: say so rather than render bind shape.
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        std::cerr << "[VulkanRenderer] morph targets on a skinned mesh are not applied "
                                     "by the Vulkan renderer (skinning only).\n";
                    }
                }
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
                    const bool placedNow = e0.isGrass && !grassPlacedThisFrame_.empty() &&
                                           std::find(grassPlacedThisFrame_.begin(), grassPlacedThisFrame_.end(),
                                                     sp.mesh) != grassPlacedThisFrame_.end();
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
                        if (placedNow) continue;
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
                        // Same addresses, different objects (see MeshFingerprint):
                        // STRUCTURAL, so the rebuild's liveCheck prune drops the
                        // dead objects' GPU resources before anything is looked
                        // up by those addresses again.
                        a.meshId != b.meshId || a.geomId != b.geomId || a.matId != b.matId ||
                        // TLAS membership flip (wireframe/overlay toggled) — the
                        // instance set changed, so a refit would MODE_UPDATE with
                        // a different instance count than the last build (spec
                        // violation → corrupt TLAS → traversal hang). Rebuild.
                        a.overlay != b.overlay || a.tex != b.tex) {
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
                    // A material edit that moved its displacementMap (map, scale,
                    // bias, transform; Material::needsUpdate) re-displaces the
                    // geometry: the refresh below then rebuilds its record whole.
                    const bool dispMoved = (matChanged || (dispEpochNow && entries[i].isDisplacementMapped)) &&
                                           !entries[i].isParticle && displacementChanged(*entries[i].mesh);
                    if (bonesChanged) bonesDirtyAny = true;
                    if (dispChanged)  displacedDirtyAny = true;
                    if (grassChanged) grassDirtyAny = true;
                    // Particle billboard meshes mutate attributes every frame but
                    // own no BLAS; never flag them geomDirty (would fire a per-
                    // frame vkDeviceWaitIdle for a refit that skips them). The
                    // billboard pass re-uploads their vertex cache itself.
                    if ((geomChanged || dispMoved) && !entries[i].isParticle) {
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
                            sp.probeChunksValid = 0;
                        }
                    }
                }
                }
                // Instance colors live in the entries' GeometryDescs (see
                // setGeomInstanceColor), but no fingerprint sees them: they are
                // neither a material version nor a geometry attribute. Poll
                // the attribute per span and route a change through the
                // material-values patch. Only consumed when structuralSame; a
                // structural rebuild re-derives every desc and re-records.
                for (auto& sp : entrySpans_) {
                    if (!sp.inst) continue;
                    const FloatBufferAttribute* col = sp.inst->instanceColor();
                    const unsigned int colVer = col ? col->version : ~0u;
                    if (col == sp.instColorAttr && colVer == sp.instColorVersion) continue;
                    sp.instColorAttr    = col;
                    sp.instColorVersion = colVer;
                    materialValuesSame = false;
                    const size_t lastW = (size_t(sp.first) + sp.count - 1) >> 5;
                    if (lastW >= meshMovedBits_.size()) meshMovedBits_.resize(lastW + 1, 0u);
                    for (uint32_t i = sp.first; i < sp.first + sp.count; ++i) {
                        entryMatDirty[i] = true;
                        meshMovedBits_[i >> 5] |= (1u << (i & 31u));
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
                        // One plain entry: its isVertexInterop, its record's drawRange
                        // snapshot, and an armed interop's per-frame enqueue. True when it
                        // tore its mesh's interop down (fixed capacity broken).
                        const auto plainEntry = [&](size_t i) -> bool {
                            if (entries[i].isSkinned || entries[i].isDisplaced ||
                                entries[i].isGrass || entries[i].isTet) return false;
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
                            if (!iop) return false;
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
                                return true;
                            }
                            if (!interopEnqueued.insert(geomKey).second) return false;

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
                            return false;
                        };
                        // Walked per SPAN, not per entry: a span is one mesh's entries (an
                        // InstancedMesh's instances), so the record lookup and the drawRange
                        // compare answer the same for all of them, and in an entry walk only
                        // the span's first entry can act — after it the record's drawRange
                        // snapshot is current and interopEnqueued is per geometry. The rest
                        // take the first's isVertexInterop. The entry walk did a blasCache
                        // hash lookup per instance: 2.0 to 2.4 ms a frame, still or moving,
                        // at about 100k instances in 4k meshes (pine-forest's orbit,
                        // scratch/orbitperf). A span whose first entry tore interop down walks
                        // its rest entry by entry, as before; spans that do not tile the
                        // entries (the slot editor keeps them contiguous) fall back to the walk.
                        const bool spansTile = !entrySpans_.empty() &&
                                size_t(entrySpans_.back().first) + entrySpans_.back().count == entries.size();
                        if (!spansTile) {
                            for (size_t i = 0; i < entries.size(); ++i) plainEntry(i);
                        } else {
                            for (const auto& sp : entrySpans_) {
                                if (sp.count == 0) continue;
                                const size_t i0 = sp.first, end = i0 + sp.count;
                                const auto& e0 = entries[i0];
                                // deformer flags are per mesh: the span's first entry speaks for it
                                if (e0.isSkinned || e0.isDisplaced || e0.isGrass || e0.isTet) continue;
                                if (plainEntry(i0)) {
                                    for (size_t i = i0 + 1; i < end; ++i) plainEntry(i);
                                    continue;
                                }
                                if (sp.count == 1) continue;
                                const auto cIt = blasCache.find(e0.mesh->geometry().get());
                                const bool iop = cIt != blasCache.end() && cIt->second->interop;
                                for (size_t i = i0 + 1; i < end; ++i) entries[i].isVertexInterop = iop;
                            }
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
                        // them too. Only a record with an auto-LOD chain still
                        // takes that drain (its level BLASes are freed past it).
                        // Every other edit, the first included, records its
                        // upload + refit into the frame cb via
                        // recordDynamicGeomRefits, ordered behind the in-flight
                        // frames on the queue: an edit every few frames (a rut
                        // ribbon written every third frame) used to drain on
                        // every write, 10 to 17 ms each, because it never made
                        // three frames in a row. Build the op lists first,
                        // decide after.
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
                            // A displacementMap record, or one that should now be:
                            // the in-place refits upload the geometry's own
                            // positions, so it is rebuilt whole (buildBlasFor
                            // displaces).
                            if (rec.dispSig != 0u || displacementSignatureOf(geomKey) != 0u ||
                                curVtx != rec.vertexCount || curIdx != rec.indexCount ||
                                !refreshFitsPacking(rec, *entries[i].mesh->geometry())) {
                                topologyChanged = true;
                                break;
                            }

                            // Streak accounting for graduation: a record dirty
                            // kDynamicGraduationStreak frames in a row is a
                            // per-frame CPU deformer (Flock's merged birds, a
                            // rewritten trail), not an occasional edit, and
                            // keeps a staging ring for the rest of its life
                            // instead of a staging buffer per edit.
                            rec.dirtyStreak = (frameSerial_ - rec.lastDirtyFrame <= 1)
                                    ? rec.dirtyStreak + 1u
                                    : 1u;
                            rec.lastDirtyFrame = frameSerial_;

                            auto* nrmAttr = entries[i].mesh->geometry()->getAttribute<float>("normal");
                            // The frame-cb route needs normals (the same
                            // requirement refreshGeomBlasBatch enforces) and no
                            // auto-LOD chain still landing (its build batch in
                            // flight on the GPU): the drained branch below is the
                            // only place that one can be freed safely, so that
                            // record stays there this frame. A landed chain is
                            // retired below instead. The quiet-window gate in the
                            // selection pass stops re-enqueues while a mesh
                            // deforms, but a mesh edited every so often (a rut
                            // chunk reused from a pool) gets a chain while it
                            // sits still, and its next edit used to drain.
                            const bool frameRoute = nrmAttr && rec.lodLanding.empty();
                            if (frameRoute) {
                                // An in-place vertex rewrite invalidates the chain:
                                // its level BLASes BAKE positions and its error
                                // bounds measured the old surface. Retired, not
                                // destroyed: earlier frames' TLASes may still
                                // reference a level. Selection already ran this
                                // frame and may have left en.lodLevel > 0, so the
                                // TLAS fill must re-read the effective level
                                // (LOD0 from here, see destroyBlasLodLevels' caller
                                // below).
                                if (rec.lodState != BlasRecord::LodState::None) {
                                    retireBlasLodLevels(rec);
                                    lodChangedThisFrame_ = true;
                                }
                                if (!rec.perFrameDynamic &&
                                    rec.dirtyStreak >= BlasRecord::kDynamicGraduationStreak) {
                                    // Graduate: allocate the staging ring, one
                                    // slot per frame in flight, each holding
                                    // positions then normals in the buffers'
                                    // own (possibly packed) formats.
                                    const VkDeviceSize nrmBytes = (rec.packedMask & kPackedNormal)
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
                                                 "per-frame dynamic path (staging ring, "
                                                 "frame-cb refit)\n";
                                }
                                // A record still in its static storage (built
                                // without ALLOW_UPDATE, or compacted) moves to
                                // updatable storage now, before this frame's TLAS
                                // fill reads its address. So does a record whose
                                // draw range has grown past what its structure was
                                // created for: the refit builds the live count, and
                                // a marching-cubes surface that gains triangles at
                                // every rebuild (spot_slam's SLAM surface) used to
                                // build past the end of its storage, which was
                                // VK_ERROR_DEVICE_LOST at the next submit or a
                                // neighbouring vertex buffer drawn as a fan of
                                // garbage (2026-10-18). moveToUpdatableBlasStorage
                                // is a no-op for an updatable record that still
                                // fits. The old pair is retired: earlier frames in
                                // flight still trace it.
                                // A record without a structure (a geometry only
                                // overlay meshes draw) has nothing to move.
                                if (ctx->rayQuerySupported() && rec.as != VK_NULL_HANDLE &&
                                    moveToUpdatableBlasStorage(rec, updatableBlasSize(rec, *entries[i].mesh->geometry()),
                                                               /*retireOld=*/true)) {
                                    rec.blasBuiltFlags = 0;// the next build is a MODE_BUILD
                                }
                                // Settles through the frame cb (dynPrevResyncPending);
                                // the draining host-side resync pass stays out.
                                rec.prevVertexResyncPending = false;
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
                            ++dynGeomStats_.drains;
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

                    // A rig that stopped this frame: recordSkinnedDeforms
                    // snapshots vertex -> prevVertex only for pending states,
                    // so prevVertex still holds the pose before the last step.
                    // One vertex -> prevVertex copy (recordDynamicGeomRefits,
                    // the deform.dynGeom pass, which the graph orders before
                    // deform.skin; a resyncing state is not pending this frame)
                    // collapses its motion to zero, as for a graduated record.
                    for (const auto& sp : entrySpans_) {
                        if (!entries[sp.first].isSkinned) continue;
                        for (size_t i = sp.first; i < size_t(sp.first) + sp.count; ++i) {
                            if (entryBonesDirty[i]) continue;
                            auto stIt = skinnedMeshStates.find(static_cast<SkinnedMesh*>(entries[i].mesh));
                            if (stIt == skinnedMeshStates.end()) continue;
                            SkinnedMeshState& st = *stIt->second;
                            if (!st.prevResyncPending || !st.blas) continue;
                            st.prevResyncPending = false;
                            pendingDynamicPrevResyncs_.push_back(st.blas.get());
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
                            if (rec->perFrameDynamic || rec->dynPrevResyncPending) {
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
                            ++dynGeomStats_.drains;
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

                    // Material patch FIRST, then the TLAS fill: the fill derives each
                    // instance's ray mask and FORCE_OPAQUE flag from the table, so
                    // it must read THIS frame's materials. A patched entry whose
                    // visibility class flipped (a blinker crossing the emitter threshold,
                    // opacity dropping below 1) sets instClassChanged, which forces the
                    // fill even on a frame with no motion, and a full build.
                    bool instClassChanged = false;
                    // A BLAS compacted since the last fill (compactAsync): every
                    // instance of it re-referenced, a full build.
                    if (blasRereferenced_) {
                        blasRereferenced_ = false;
                        instClassChanged = true;
                    }
                    if (hiddenFlippedThisFrame_ || slotsEditedThisFrame_) {
                        // Hidden or shown in place: new ray masks (a full TLAS
                        // build, like any class change), every view's cull and
                        // draw records, and the emitter walk. A slot edit is the
                        // same event with instances added or removed.
                        instClassChanged = true;
                        ++drawInputsVersion_;
                        cachedEmissiveEntryCount_ = static_cast<size_t>(-1);
                    }
                    // LOD fades: new words re-record the draws; a side of a
                    // crossfade taking over the shadows changes ray masks,
                    // which is a class change (a full TLAS build).
                    {
                        const LodFadeChange fc = refreshLodFades(false);
                        if (fc.words) ++drawInputsVersion_;
                        if (fc.casters) instClassChanged = true;
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
                        // The table is per material: a changed material
                        // rewrites its record once, and an entry whose instance
                        // colour changed rewrites the colour in its GeometryDesc.
                        // Overlay/skipped entries keep record 0.
                        // TARGETED PATCH: only entries whose material version
                        // actually bumped (entryMatDirty) get their MaterialDesc
                        // re-derived. materialFromMesh is a ~25-dynamic_cast walk
                        // (plus each albedoTexOf/…/occlusionTexOf is another typed
                        // probe + shared_ptr churn); re-deriving EVERY entry here
                        // turned a single brake-light / blinker emissive toggle into
                        // an O(scene) stall — tens of ms once InstancedMesh foliage
                        // expands the scene to tens of thousands of entries, firing
                        // several times a second while a blinker ticks. The table
                        // already holds correct values for every unchanged entry (it
                        // survives matrix-only frames untouched), and a changed
                        // material keeps its pointer + textures (a texture swap aborts
                        // to the full structural rebuild), so its dedup index stays
                        // valid. Full fallback only if the cache size somehow doesn't
                        // match (never on the fast path that reaches here).
                        const bool patchMatDescs = geomDescsCached_.size() == entries.size();
                        if (!patchMatDescs) {
                            // Every entry re-derived below into a fresh table;
                            // every entry's index and every record dirty.
                            geomDescsCached_.resize(entries.size());
                            for (GeometryDesc& gd : geomDescsCached_) setGeomMatIdx(gd, 0u);
                            matTableReset();
                            markGeomDescsWhole();
                        }
                        matTableBeginPass();
                        // Same dedup as the full-rebuild path below (consulted only on
                        // the full-rebuild fallback): entries order is identical
                        // (overlay skip matches), so identical Material* pointers
                        // produce identical materialAssetIdx values frame-to-frame.
                        std::unordered_map<const Material*, uint32_t> matAssetMap;
                        // One-entry memo of the material's desc. A dirty instanced span is
                        // N consecutive entries sharing one Material, and an
                        // instance-color edit dirties all of them; without the
                        // memo each re-runs materialFromMesh + the texture-of
                        // walk. Patch mode only: materialAssetIdx is taken from
                        // the entry's current record, so the memoised desc is
                        // valid for every entry of the same Material.
                        const Material* memoMat = nullptr;
                        MaterialDesc memoMd{};
                        for (size_t i = 0; i < entries.size(); ++i) {
                            const MeshEntry& en = entries[i];
                            if (en.isOverlay) continue;// raster-overlay only — no MaterialDesc
                            if (patchMatDescs && !entryMatDirty[i]) continue;// unchanged → keep its record
                            Mesh* m = en.mesh;
                            const Material* mat = m->material().get();
                            if (!(patchMatDescs && memoMat && mat == memoMat)) {
                            MaterialDesc md = materialFromMesh(*m);
                            if (patchMatDescs) {
                                // Material* unchanged → its dedup index is stable.
                                md.materialAssetIdx = entryMat(i).materialAssetIdx;
                            } else {
                                auto [matIt, inserted] = matAssetMap.try_emplace(
                                        mat, static_cast<uint32_t>(matAssetMap.size()));
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
                                md.occlusionTexIndex = ensureMaterialTexture(tex, true);
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
                            memoMat = mat;
                            memoMd  = md;
                            }
                            // Compared against the record being replaced; on the
                            // full fallback that is record 0, so assume a change.
                            const MaterialDesc& was = entryMat(i);
                            const MaterialDesc& md = memoMd;
                            if (!patchMatDescs ||
                                tlasInstanceFlags(was) != tlasInstanceFlags(md) ||
                                alphaMaskGroup(was) != alphaMaskGroup(md))
                                instClassChanged = true;
                            // The emitter walk's inputs, copied verbatim from the
                            // same MaterialWithEmissive fields it reads. A change
                            // re-walks: this is how an entry that BECOMES an
                            // emitter (and so is in no emitter mask) gets found.
                            if (!patchMatDescs || was.emissiveIntensity != md.emissiveIntensity ||
                                was.emissive[0] != md.emissive[0] || was.emissive[1] != md.emissive[1] ||
                                was.emissive[2] != md.emissive[2])
                                cachedEmissiveEntryCount_ = static_cast<size_t>(-1);
                            // The record is rewritten (and its range dirtied) once
                            // per pass however many entries share it; the entry's
                            // GeometryDesc when it lands on another record or its
                            // instance color changed.
                            setEntryMat(i, matTableSlot(mat, memoMd));
                            {
                                GeometryDesc& gd = geomDescsCached_[i];
                                float was[3];
                                std::memcpy(was, gd.instanceColor, sizeof(was));
                                setGeomInstanceColor(gd, en);
                                if (std::memcmp(was, gd.instanceColor, sizeof(was)) != 0)
                                    markGeomDescsDirty(static_cast<uint32_t>(i));
                            }
                        }
                        if (!patchMatDescs) markMatDescsWhole();
                        cacheCullFlags();
                    }
                    // Cutout ray distance: after the material patch (it reads
                    // the material table). A flipped bit changes an instance's
                    // flags, which is a class change like any other.
                    if (classifyCutoutFar(rayCamPos)) instClassChanged = true;
                    if (!matricesSame || bonesDirtyAny || displacedDirtyAny || grassDirtyAny || tetDirtyAny || geomDirtyAny || morphDirtyAny || interopDirtyAny || lodChangedThisFrame_ || instClassChanged) {
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
                        // Stage 5: instanced spans whose instances tlas_gen.comp
                        // appends after these (the expansion's world matrices).
                        const bool tlasGenOn = gpuTlasGen_ && gpuInstanceExpand_ && tlasGen_ && instExpand_ &&
                                               ctx->rayQuerySupported() && !asyncPilot_;
                        tlasGenSpans_.clear();
                        tlasGenLodRows_.clear();
                        tlasGenWork_ = 0;
                        if (gpuDrawGenCheck_) tlasGenExpected_.clear();
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
                            VkGeometryInstanceFlagsKHR spanFlags =
                                    VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                            // A cutout caster's far entries are force-opaque
                            // (setCutoutRayDistance; the bits are classifyCutoutFar's).
                            bool spanCutout = false;
                            if (sp.first < geomDescsCached_.size()) {
                                const auto& cmd = entryMat(sp.first);
                                if (!e0.isDisplaced && alphaMaskGroup(cmd)) spanMask = kRayMaskAlpha;
                                spanFlags = tlasInstanceFlags(cmd);
                                spanCutout = cutoutRayDistance_ > 0.f && cutoutCaster(cmd);
                            }
                            if (e0.camAttached) spanMask = kRayMaskNoShadow;
                            if (e0.sensorOnly)
                                spanMask = sensorOnlySurfaces_ ? kRayMaskSensorOnly : 0u;
                            if (sp.hidden) spanMask = 0u;// MeshEntry::hidden: no ray hits it
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
                            // Stage 5: one span-level descriptor instead of a
                            // per-instance fill. The GeometryDesc LOD reconcile
                            // the loop below does per entry still runs (a chain
                            // destroyed since leaves a level's index address).
                            bool sinkTlas = false;
                            if (tlasGenOn && sp.inst && !sp.isFree && !isDeformer && rec && !spanCutout &&
                                lodSel0.asAddress != 0) {
                                vulkan::TlasGen::SpanDesc d{};
                                d.blasAddress = lodSel0.asAddress;
                                d.firstEntry  = sp.first;
                                d.count       = sp.count;
                                d.workBase    = tlasGenWork_;
                                d.maskWord    = static_cast<uint32_t>(spanMask) << 24;
                                d.flagsWord   = static_cast<uint32_t>(spanFlags) << 24;
                                // An auto-LOD chain: each level's BLAS (and the
                                // fallback past it), as the loop below reads them.
                                d.lodRowBase  = static_cast<uint32_t>(tlasGenLodRows_.size());
                                if (perEntryLod) {
                                    const auto levels = static_cast<uint32_t>(rec->lodLevels.size());
                                    for (uint32_t k = 1; k <= levels + 1u; ++k)
                                        tlasGenLodRows_.push_back(selectLodGeom(*rec, static_cast<uint8_t>(k)).asAddress);
                                    d.lodRowCount = levels + 1u;
                                }
                                tlasGenSpans_.push_back(d);
                                tlasGenWork_ += sp.count;
                                if (!gpuDrawGenCheck_) {
                                    for (uint32_t j = 0; j < sp.count; ++j) {
                                        const size_t i = size_t(sp.first) + j;
                                        if (i >= geomDescsCached_.size()) break;
                                        auto& gd = geomDescsCached_[i];
                                        const auto sel = perEntryLod ? selectLodGeom(*rec, entries[i].lodLevel)
                                                                     : lodSel0;
                                        if (gd.indexAddress != sel.indexAddress ||
                                            geomFlag(gd, kGeomFlagIndexed) != sel.indexed) {
                                            gd.indexAddress = sel.indexAddress;
                                            setGeomFlag(gd, kGeomFlagIndexed, sel.indexed);
                                            markGeomDescsDirty(static_cast<uint32_t>(i));
                                        }
                                    }
                                    continue;
                                }
                                sinkTlas = true;// check mode: the loop builds the reference
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
                                    // fetch. Reconciled on EVERY fill, not only while
                                    // the record has a chain: an in-place vertex edit
                                    // destroys the chain (lodChainDoomed), and the
                                    // mirror still held the destroyed level's index
                                    // address and indexed = 1 — every secondary hit on
                                    // the mesh then fetched its triangles through a
                                    // FREED index buffer while the raster drew LOD0.
                                    // A deforming soup that had sat still long enough
                                    // to get a chain (a film's warm-up renders) was
                                    // refracted and reflected as crumpled garbage
                                    // until the next structural rebuild. Two compares
                                    // per entry; the upload is marked per entry, only
                                    // when something differs.
                                    const auto lodSel = perEntryLod ? selectLodGeom(*rec, en.lodLevel)
                                                                    : lodSel0;
                                    blasAddr = lodSel.asAddress;
                                    if (i < geomDescsCached_.size()) {
                                        auto& gd = geomDescsCached_[i];
                                        if (gd.indexAddress != lodSel.indexAddress ||
                                            geomFlag(gd, kGeomFlagIndexed) != lodSel.indexed) {
                                            gd.indexAddress = lodSel.indexAddress;
                                            setGeomFlag(gd, kGeomFlagIndexed, lodSel.indexed);
                                            markGeomDescsDirty(static_cast<uint32_t>(i));
                                        }
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
                                // The side of a LOD crossfade the shadows do not
                                // take is hit by no ray (lodFadeCasts).
                                inst.mask = (sp.fadeAny && !lodFadeCasts(entryLodFade_[i])) ? 0u : spanMask;
                                inst.instanceShaderBindingTableRecordOffset = 0;
                                inst.flags = (spanCutout && en.rayFarOpaque)
                                                     ? (spanFlags | VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR)
                                                     : spanFlags;
                                inst.accelerationStructureReference = blasAddr;
                                if (sinkTlas) {
                                    tlasGenExpected_.push_back(inst);
                                    continue;
                                }
                                instances.push_back(inst);
                            }
                        }
                        if (lodChangedThisFrame_ && !geomDescsCached_.empty()) {
                            // geomDescsCached_ was patched in place above; the
                            // per-frame-in-flight ring carries it to the GPU
                            // stall-free — renderFrame flushes THIS frame's
                            // slot right after its fence signals (before any
                            // recording that consumes it), the other slot when
                            // its frame comes around. Same model as the material table;
                            // this replaced a vkDeviceWaitIdle per switch
                            // frame, which hitched exactly when the camera
                            // was moving.
                            //
                            // WHOLE-ARRAY on purpose, unlike the material patch:
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
                        // The GPU spans go after the CPU's instances; a new split
                        // is a full build.
                        pendingTlasGpuInstances_ = tlasGenWork_;
                        tlasGenActive_ = tlasGenWork_ > 0u;
                        uint64_t partition = 0;
                        if (tlasGenActive_) {
                            partition = 1469598103934665603ull ^ instances.size();
                            for (auto& d : tlasGenSpans_) {
                                d.instBase = static_cast<uint32_t>(instances.size()) + d.workBase;
                                partition = (partition ^ ((uint64_t(d.firstEntry) << 32) | d.count)) * 1099511628211ull;
                            }
                            if (gpuDrawGenCheck_) {
                                tlasGenExpectedFrame_ = currentFrame;
                                tlasGenExpectedBase_  = static_cast<uint32_t>(instances.size());
                            }
                        }
                        const bool splitChanged = partition != tlasGenPartition_;
                        tlasGenPartition_ = partition;
                        pendingTlasInstances_.swap(instances);// scratch keeps its capacity
                        // A class flip rebuilds rather than updates: whether an
                        // update may change instance flags is not worth betting a
                        // traversal on, and it only happens on the flip frame.
                        // A deformer refit (bones, displaced, grass, tet, morph) keeps
                        // every BLAS handle, so the TLAS is UPDATED; a geometry or LOD
                        // change re-references BLASes and rebuilds.
                        pendingTlasFullBuild_ = geomDirtyAny || lodChangedThisFrame_ || instClassChanged || splitChanged;
                        (void)blasDeformed;
                        pendingTlasRefit_ = true;
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
            // The rebuild's phases, siblings under THREEPP_CPU_PHASE_PROFILE
            // (the per-event cost VulkanEntryChurn_test reports, broken down).
            THREEPP_CPUPROF_NAMED(rbDrain, "scene.R0_drainPrune");
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
                reapAsyncOneShots(/*deviceIdle=*/true);
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
                        worldStaticGeoms_.erase(it->first);// same hazard, same fix
                        it = blasCache.erase(it);
                    } else {
                        ++it;
                    }
                }
                // Displacement specs of dead geometries go too (the spec holds
                // its map alive); a live geometry's is looked up through the
                // same weak test (displacementSignatureOf, buildBlasFor).
                for (auto it = geomDisplacement_.begin(); it != geomDisplacement_.end();) {
                    if (it->second.geom.expired()) it = geomDisplacement_.erase(it);
                    else ++it;
                }
                // The per-mesh deform states go with their geometry OR their
                // mesh (MeshOwner): a freed clone's state, keyed by an address
                // the allocator may hand the next mesh, while the template
                // keeps the shared geometry alive.
                for (auto it = skinnedMeshStates.begin(); it != skinnedMeshStates.end(); ) {
                    if (it->second->liveCheck.expired() || it->second->owner.gone()) {
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
                    if (it->second->liveCheck.expired() || it->second->owner.gone()) {
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
                    if (it->second->liveCheck.expired() || it->second->owner.gone()) {
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
                        if (st->wake) {
                            if (st->wake->stateView() == oceanWakeView) oceanWakeView = wakeField_->dummyView();
                            if (st->wake->rippleView() == oceanRippleView) oceanRippleView = wakeField_->dummyView();
                            wakeField_->destroyState(*st->wake);
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
                        for (auto& b : st->vesselBuffer) destroyBuffer(ctx->allocator(), b);
                        // PhillipsSpectrum / DynamicSpectrum / IFFT destructors
                        // run on unique_ptr reset.
                        it = displacedStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                // destroyBlasRecord, not a list of its buffers: the list here
                // missed the refit scratch, the colour and the RT index, so every
                // field or morphed mesh the rebuild dropped leaked them.
                for (auto it = grassStates.begin(); it != grassStates.end(); ) {
                    if (it->second->liveCheck.expired() || it->second->owner.gone()) {
                        auto& st = it->second;
                        if (st->blas) destroyBlasRecord(*st->blas);
                        destroyBuffer(ctx->allocator(), st->restPos);
                        destroyBuffer(ctx->allocator(), st->heightFrac);
                        it = grassStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = morphedMeshStates.begin(); it != morphedMeshStates.end(); ) {
                    if (it->second->liveCheck.expired() || it->second->owner.gone()) {
                        if (auto& rec = it->second->blas) destroyBlasRecord(*rec);
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

            rbDrain.stop();
            THREEPP_CPUPROF_NAMED(rbLoop, "scene.R1_entryLoop");
            // The slot pool the edits until the next structural rebuild write
            // into (VulkanCoreSlots.cpp): every entry-indexed GPU buffer this
            // build makes is sized for it. 1.5x and not 2x: three ring slots
            // of every entry-indexed array. The material table is sized by
            // materials instead (below).
            slotCapacity_ = std::max<uint32_t>({1024u, static_cast<uint32_t>(entries.size() + entries.size() / 2u),
                                                reservedSlots_});
            std::vector<VkAccelerationStructureInstanceKHR> instances;
            instances.reserve(entries.size());
            // geomDescs are ENTRIES-indexed: one slot per entry, with
            // overlay/skipped entries left default (material record 0, the
            // all-zero desc, as their per-entry MaterialDesc used to be). This makes
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
            // The material table afresh, in entry order: a rebuild compacts it.
            matTableReset();
            matTableBuilding_ = true;
            // Same index space, same lifetime — see entryStableIds_. Filled in
            // the geomDescs loop below so a skipped/overlay entry keeps the 0
            // its dummy slot already means.
            entryStableIds_.assign(entries.size(), 0u);
            stableIdsPending_ = true;
            // LOD fade words for the new entry order, before the instance loop
            // below derives the ray masks from them.
            refreshLodFades(true);

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
            // The geometries some traced entry draws this build. A geometry
            // drawn only by overlay meshes (a wireframe, an overlay-layer
            // panel, a UI blend) gets a record without an acceleration
            // structure below: no ray meets an overlay, it is never a TLAS
            // instance, so its structure was build time at admission, a refit
            // per edit and device memory for nothing — spot_slam's wireframe
            // marching-cubes surface, 600k vertices rebuilt every 3 s, until
            // 2026-10-18. One insert per mesh, not per instance: an
            // InstancedMesh's entries are contiguous.
            std::unordered_set<const BufferGeometry*> tracedGeoms;
            for (size_t i = 0; i < entries.size(); ++i) {
                const MeshEntry& e0 = entries[i];
                if (e0.isOverlay || !e0.mesh) continue;
                if (i > 0 && entries[i - 1].mesh == e0.mesh) continue;
                tracedGeoms.insert(e0.mesh->geometry().get());
            }

            for (size_t i = 0; i < entries.size(); ++i) {
                beginOneShotBatch();// (re)arm; idempotent if a batch is already open
                const MeshEntry& en = entries[i];
                Mesh* m = en.mesh;
                // Particle billboard meshes own their vertex buffers in the
                // dedicated billboard pass — they need no BLAS and never enter
                // the TLAS. Skip before the geometry-keyed build so we don't
                // allocate (or per-frame refit) an AS for them.
                if (en.isParticle) continue;
                // A free slot (VulkanCoreSlots.cpp): no record, no instance,
                // the default descs.
                if (slotTombstone_ && m == slotTombstone_.get()) continue;
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
                    // A state built for a geometry the field no longer draws (a
                    // swap, the old geometry alive elsewhere): behind this
                    // rebuild's device wait, freed and built again.
                    if (const auto sIt = grassStates.find(gm);
                        sIt != grassStates.end() && sIt->second->liveCheck.lock() != gm->geometry()) {
                        if (sIt->second->blas) destroyBlasRecord(*sIt->second->blas);
                        destroyBuffer(ctx->allocator(), sIt->second->restPos);
                        destroyBuffer(ctx->allocator(), sIt->second->heightFrac);
                        grassStates.erase(sIt);
                    }
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
                        // A hidden field takes none (grassWantsDeform).
                        const auto& cw = camera.matrixWorld->elements;
                        const Vector3 camPos(cw[12], cw[13], cw[14]);
                        if (grassWantsDeform(*st, en.hidden, grassShouldAnimate(en, camPos))) {
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
                    const bool traced = ctx->rayQuerySupported() && tracedGeoms.count(geomKey) != 0;
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
                        // Displaced by another displacementMap spec than now (or
                        // newly, or no longer): the same evict-and-rebuild.
                        const bool dispMismatch = it->second->dispSig != displacementSignatureOf(geomKey);
                        if (it->second->geomVersion != curVer || packedMismatch || dispMismatch) {
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
                                auto fresh = buildBlasFor(*m->geometry(), /*allowPacked=*/true, /*updatable=*/false);
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
                    // A record built without a structure — its geometry was
                    // drawn only by overlay meshes then — that a traced mesh
                    // now shares (a solid mesh added beside its own wireframe):
                    // rebuilt with one, before the instance below reads its
                    // address. An interop record always has one.
                    if (it != blasCache.end() && traced && it->second->as == VK_NULL_HANDLE &&
                        !it->second->interop) {
                        destroyBlasLodLevels(*it->second);
                        destroyBlasRecord(*it->second);
                        blasCache.erase(it);
                        it = blasCache.end();
                    }
                    if (it == blasCache.end()) {
                        auto rec = buildBlasFor(*m->geometry(), /*allowPacked=*/true, /*updatable=*/false,
                                                nullptr, VertexWriter::Host, traced);
                        if (!rec) continue;// degenerate / unsupported geometry
                        rec->liveCheck = m->geometry();
                        it = blasCache.emplace(geomKey, std::move(rec)).first;
                    }
                    recPtr = it->second.get();
                }

                // Overlay meshes need a BlasRecord (the raster overlay pass
                // binds its vertex buffer) but no acceleration structure, and
                // must not appear in the TLAS or GeometryDesc/MaterialDesc
                // arrays — the traced/rasterized scene must not see them.
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
                inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;// + FORCE_OPAQUE below
                // en.lodLevel==0 for every deformer/exempt entry (auto-LOD
                // selection above only ever sets it on plain cached
                // geometry), so this passthrough is a no-op for them —
                // recPtr may point at a SkinnedMeshState/DisplacedMeshState/
                // etc.'s own BlasRecord, which selectLodGeom treats
                // identically to a LOD0 blasCache record.
                const auto lodSel = selectLodGeom(*recPtr, en.lodLevel);
                inst.accelerationStructureReference = lodSel.asAddress;
                instances.push_back(inst);

                const GeometryDesc gdesc = buildGeometryDesc(en, *recPtr, lodSel);
                geomDescs[i] = gdesc;
                // Read-only: assignment of auto ids stays in the indirect draw
                // builder so this cannot renumber what the Ids AOV reports.
                entryStableIds_[i] = stableIdIfAssigned(*en.mesh);

                const Material* matKey = m->material().get();
                auto memoIt = matDescMemo.find(matKey);
                if (memoIt == matDescMemo.end()) {
                    MaterialDesc md0 = buildMaterialDesc(*m);
                    {
                        auto [matIt, inserted] = matAssetMap.try_emplace(
                                matKey, static_cast<uint32_t>(matAssetMap.size()));
                        md0.materialAssetIdx = matIt->second;
                    }
                    memoIt = matDescMemo.emplace(matKey, md0).first;
                }
                // The memo holds the material's desc, which is its table
                // record; the entry's instance color is in its GeometryDesc.
                const uint32_t matIdx = matTableSlot(matKey, memoIt->second);
                setGeomMatIdx(geomDescs[i], matIdx);
                matTableRef(matIdx);// geomDescs becomes geomDescsCached_ below
                const MaterialDesc& md = matTable_[matIdx];
                // Visibility group (see vulkan_shared.h): blend/transmissive
                // surfaces move to the alpha mask so pure-visibility occlusion
                // queries (env gather, GI, emissive-NEE) cull them out — a text
                // decal's transparent quad must not block IBL light. Water
                // (DisplacedMesh) stays opaque-mask: underwater light transport
                // is handled by its volumetrics, not pass-through.
                if (!instances.empty()) {
                    instances.back().mask =
                            en.hidden ? 0u// MeshEntry::hidden: no ray hits it
                            : !lodFadeCasts(entryLodFade_[i]) ? 0u// the other side of a LOD crossfade
                            : en.sensorOnly
                                    // Sensors only, and only once opted in: mask 0
                                    // is hit by no ray at all.
                                    ? (sensorOnlySurfaces_ ? kRayMaskSensorOnly : 0u)
                            : en.camAttached
                                    ? kRayMaskNoShadow// FP viewmodel: visible, never occludes
                                    : (!en.isDisplaced && alphaMaskGroup(md))
                                              ? kRayMaskAlpha
                                              : kRayMaskOpaque;
                    instances.back().flags = tlasInstanceFlags(md);
                    // Cutout ray distance: the same bit classifyCutoutFar
                    // derives on lean frames, for this frame's camera.
                    const bool beyond = cutoutRayDistance_ > 0.f && cutoutCaster(md) &&
                                        entryBeyond(en, rayCamPos, cutoutRayDistance_);
                    entries[i].rayFarOpaque = beyond;
                    if (beyond) instances.back().flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
                }
            }
            cutoutRayClassified_    = true;
            cutoutRayClassifiedFor_ = cutoutRayDistance_;
            std::copy(rayCamPos, rayCamPos + 3, cutoutRayCamPos_);

            // Submit the whole tile-admit batch once (BLAS builds + prevVertex
            // seeds + texture staging/blits), waiting a single time. After this
            // returns every new BLAS is built and every new albedo texture is
            // resident, so buildTlas below can reference the fresh BLAS.
            matTableBuilding_ = false;
            rbLoop.stop();
            THREEPP_CPUPROF_NAMED(rbFlush, "scene.R2_batchFlush");
            flushOneShotBatch();
            // Records compacted by the batch flushes in this pass were pushed
            // above with their build-size addresses.
            applyBlasCompactionRemap(instances);
            rbFlush.stop();
            THREEPP_CPUPROF_NAMED(rbTlas, "scene.R3_buildTlas");
            buildTlas(instances);
            rbTlas.stop();
            THREEPP_CPUPROF_NAMED(rbDescs, "scene.R4_descUpload");
            // Every per-frame GeometryDesc slot seeded fresh — same ring
            // model as the matDescs loop below; the lean auto-LOD patch
            // flips geomDescsDirty_ and flushGeometryDescsIfDirty carries
            // deltas per slot from then on.
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                uploadDescBuffer(geometryDescsBuffers[f], geomDescs, slotCapacity_);
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
            // Seed every per-frame slot with the fresh table so the first few
            // frames don't try to flush against a half-initialised ring.
            // matTable_ stays the host-side authoritative copy the hot-path
            // flush carries deltas from. Sized by materials, with room for
            // the slot path's new ones before it has to grow.
            matTableCapacity_ = std::max<uint32_t>(256u, 2u * static_cast<uint32_t>(matTable_.size()));
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                uploadDescBuffer(materialDescsBuffers[f], matTable_, matTableCapacity_);
            }
            for (auto& d : matDescsDirty_) d.clear();
            cacheCullFlags();
            // The asset numbering the slot path continues (slotMaterialAssetIdx).
            slotMatAssetIdx_.clear();
            for (const auto& [mat, idx] : matAssetMap) slotMatAssetIdx_[mat] = {mat ? mat->id : 0u, idx};
            slotNextMatAssetIdx_ = static_cast<uint32_t>(matAssetMap.size());
            // The table just folded in every span's current instance colors.
            for (auto& sp : entrySpans_) {
                if (!sp.inst) continue;
                sp.instColorAttr    = sp.inst->instanceColor();
                sp.instColorVersion = sp.instColorAttr ? sp.instColorAttr->version : ~0u;
            }

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
            rbDescs.stop();
            THREEPP_CPUPROF_NAMED(rbRemap, "scene.R5_identityRemap");
            std::vector<std::array<float, 16>> oldPrevWorld = std::move(prevWorldByEntry_);
            std::vector<uint8_t> oldPrevValid = std::move(prevWorldValidByEntry_);
            seedMotionState(entries);// identity motion + prev=current for every slot
            bool identityOk = sceneBuilt_;
            if (identityOk) {
                std::unordered_map<EntryKey, size_t, EntryKeyHash> oldIdx;
                oldIdx.reserve(prevSceneFingerprint.size());
                for (size_t i = 0; i < prevSceneFingerprint.size() && identityOk; ++i) {
                    const auto& p = prevSceneFingerprint[i];
                    // A free slot (VulkanCoreSlots.cpp) names the tombstone,
                    // every one of them at instance 0: no history to carry,
                    // and not an ambiguous match.
                    if (slotTombstone_ && p.mesh == slotTombstone_.get()) continue;
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
            rbRemap.stop();
            THREEPP_CPUPROF_NAMED(rbViews, "scene.R6_capacityDescriptors");
            const uint32_t motionSlots    = static_cast<uint32_t>(entries.size());
            const uint32_t neededBitWords = std::max<uint32_t>((motionSlots + 31u) / 32u, 1u);
            // The buffers hold the slot pool, the host arrays the entries.
            const uint32_t poolSlots = std::max(motionSlots, slotCapacity_);
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                ensureMotionMatCapacity(f, std::max<uint32_t>(poolSlots, 1u));
                ensureMeshMovedBitsCapacity(f, std::max<uint32_t>((poolSlots + 31u) / 32u, 1u));
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
            // guards per-view sets, and addSecondaryViewPasses calls
            // uploadRasterCameraUbo once per view per frame, so each view
            // heals its own slot.
            for (auto& v : views_) { v->rasterMatTexValid_.fill(0); v->shadowMatTexValid_.fill(0); }
            prevSceneFingerprint = std::move(currFp);
            sceneBuilt_ = true;
            ++drawInputsVersion_;// structural rebuild — every draw record is new
            ++tlasStats_.structuralRebuilds;

            // Companion to the post-init dump: with the scene's BLASes +
            // geometry buffers now resident, allocationBytes − the post-init
            // baseline is the scene's true device cost (the OS-level per-
            // process counter only moves in VMA block granularity, so it
            // under-reports buffer-level savings such as packed attributes).
            if (const char* e = std::getenv("THREEPP_VK_MEMDUMP"); e && *e && *e != '0') {
                dumpMemoryStats("scene-built");
            }
            // Always on: one line when a deformer created by this build landed
            // in system memory (its deform pass then crosses PCIe every frame).
            reportDeformPlacement();
        }

VulkanRenderer::Impl::GeometryDesc VulkanRenderer::Impl::buildGeometryDesc(const MeshEntry& en, const BlasRecord& rec,
                                                                         const LodGeomSel& lodSel) const {
            GeometryDesc gdesc{};
            gdesc.vertexAddress = rec.vertex.address;
            gdesc.normalAddress = rec.normal.address;
            // Must reference the SAME index buffer the selected BLAS
            // level was built from — RT secondary hits (reflections/GI/
            // lidar/probe update) read this keyed by gl_PrimitiveID,
            // which only lines up against that exact index array. A
            // record with a ray-tracing proxy (DisplacedMesh) was built
            // over rtIndex; its `index` is the raster's.
            gdesc.indexAddress  = rec.rtIndexCount ? rec.rtIndex.address
                                                       : lodSel.indexAddress;
            gdesc.uvAddress     = rec.uv.address;// 0 if no UV attribute
            // 0 = not water; 1 = an FFT-displaced ocean surface; anything
            // else = one with a wake field, and this is its patch table.
            gdesc.foamAddress   = rec.isOceanSurface
                                          ? (rec.oceanWakeTable ? rec.oceanWakeTable : 1ull)
                                          : 0ull;
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
            bool warpReproject = rec.interopWorldStatic;
            if (en.isDisplaced) {
                warpReproject = static_cast<DisplacedMesh*>(en.mesh)->warp.halfRange > 0.0f;
            }
            gdesc.prevVertexAddress =
                    (rec.prevVertex.handle != VK_NULL_HANDLE && !warpReproject)
                            ? rec.prevVertex.address
                            : rec.vertex.address;
            // Per-vertex color only when the material opts in (three.js
            // semantics: vertexColors == true) AND the geometry uploaded a
            // color buffer (kGeomFlagVertexColor below).
            bool useVtxColor = false;
            {
                const auto mat = en.mesh->material();
                useVtxColor = rec.color.handle != VK_NULL_HANDLE && mat && mat->vertexColors;
            }
            setGeomInstanceColor(gdesc, en);
            // kGeomFlagMoved is stamped per frame in VulkanCoreFrame; it is
            // seeded 0 here, with the packed bits above it.
            // kGeomFlagRtProxy: the record traces a ray-tracing proxy
            // (rtIndex), a coarser grid than the raster drew; the water
            // shade moves its surface point onto it before it launches rays.
            // kGeomFlagGlowOnly: an emissive that glows but is not a light
            // (setEmissiveCastsLight): its ray hits keep their emission.
            // kGeomFlagIndexed per SELECTION, not per record: a level of a
            // non-indexed soup record IS indexed (welded canonical indices).
            // The material index above them is the caller's.
            gdesc.flags = (rec.packedMask << kGeomFlagPackedShift) |
                          (rec.rtIndexCount ? kGeomFlagRtProxy : 0u) |
                          (isGlowOnly(*en.mesh) ? kGeomFlagGlowOnly : 0u) |
                          (lodSel.indexed ? kGeomFlagIndexed : 0u) |
                          (useVtxColor ? kGeomFlagVertexColor : 0u);
            return gdesc;
        }

VulkanRenderer::Impl::MaterialDesc VulkanRenderer::Impl::buildMaterialDesc(Mesh& m) {
            MaterialDesc md = materialFromMesh(m);
            if (auto tex = albedoTexOf(m)) {
                md.albedoTexIndex = ensureMaterialTexture(tex);
                copyTexUvTransform(md.uvTransform, tex);
            }
            if (auto tex = roughnessTexOf(m)) {
                md.roughnessTexIndex = ensureMaterialTexture(tex);
                copyTexUvTransform(md.uvTransformRoughMetal, tex);
            }
            if (auto tex = metalnessTexOf(m)) {
                md.metalnessTexIndex = ensureMaterialTexture(tex);
                if (md.roughnessTexIndex < 0) copyTexUvTransform(md.uvTransformRoughMetal, tex);
            }
            if (auto tex = normalTexOf(m)) {
                md.normalTexIndex = ensureMaterialTexture(tex);
                copyTexUvTransform(md.uvTransformNormal, tex);
            }
            if (auto tex = transmissionTexOf(m)) {
                md.transmissionTexIndex = ensureMaterialTexture(tex);
                copyTexUvTransform(md.uvTransformTransmission, tex);
            }
            if (auto tex = clearcoatTexOf(m)) {
                md.clearcoatTexIndex = ensureMaterialTexture(tex);
                copyTexUvTransform(md.uvTransformClearcoat, tex);
            }
            if (auto tex = clearcoatRoughnessTexOf(m)) {
                md.clearcoatRoughnessTexIndex = ensureMaterialTexture(tex);
                copyTexUvTransform(md.uvTransformClearcoatRough, tex);
            }
            if (auto tex = emissiveTexOf(m)) {
                md.emissiveTexIndex = ensureMaterialTexture(tex);
                copyTexUvTransform(md.uvTransformEmissive, tex);
            }
            if (auto tex = occlusionTexOf(m)) {
                md.occlusionTexIndex = ensureMaterialTexture(tex, true);
                copyTexUvTransform(md.uvTransformOcclusion, tex);
            }
            if (auto tex = detailTexOf(m)) {
                md.detailTexIndex = ensureMaterialTexture(tex);
            }
            if (auto tex = detailNormalTexOf(m)) {
                md.detailNormalTexIndex = ensureMaterialTexture(tex);
            }
            if (auto tex = terrainWeightTexOf(m)) {
                md.terrainWeightTexIndex = ensureMaterialTexture(tex);
                // Band sets are SHARED textures (one allocation across
                // every tile material — ensureMaterialTexture dedupes
                // by pointer), so this costs 8 slots total, not 8/tile.
                for (int bi = 0; bi < 4; ++bi) {
                    if (auto bt = terrainBandAlbedoTexOf(m, bi))
                        md.terrainBandAlbedoTex[bi] = ensureMaterialTexture(bt);
                    if (auto bt = terrainBandNormalTexOf(m, bi))
                        md.terrainBandNormalTex[bi] = ensureMaterialTexture(bt);
                }
            }
            if (auto tex = terrainNormalTexOf(m)) {
                md.terrainNormalTexIndex = ensureMaterialTexture(tex);
            }
            return md;
        }

void VulkanRenderer::Impl::refreshSpanMatrices() {
            // Per SPAN: a span's entry world matrices are a pure function of
            // (mesh->matrixWorld, instanceMatrix contents); see the lean path.
            for (auto& sp : entrySpans_) {
                sp.movedThisFrame = false;
                sp.instancesMoved = false;
                const float* mw = sp.mesh->matrixWorld->elements.data();
                if (sp.inst) {
                    const unsigned int v = sp.inst->instanceMatrix()->version;
                    const bool meshMoved = std::memcmp(sp.meshWorld.data(), mw, 64) != 0;
                    if (v == sp.instMatVersion && !meshMoved) continue;
                    sp.instMatVersion = v;
                    std::memcpy(sp.meshWorld.data(), mw, 64);
                    sp.movedThisFrame = true;
                    sp.aabbValid = false;
                    sp.probeChunksValid = 0;
                    // An instance-matrix edit alone: the entries it changed
                    // are marked moved here, one by one (see instancesMoved).
                    // A moved mesh moves every instance, which the diff marks.
                    sp.instancesMoved = !meshMoved && sp.count > 0u;
                    if (sp.instancesMoved) {
                        const size_t lastW = (size_t(sp.first) + sp.count - 1) >> 5;
                        if (lastW >= meshMovedBits_.size()) meshMovedBits_.resize(lastW + 1, 0u);
                    }
                    // Tight refresh: world = meshWorld * instanceMat[j],
                    // straight off the raw attribute floats (no
                    // bounds-checked getMatrixAt). The multiply itself
                    // goes through Matrix4::multiplyMatrices rather than a
                    // hand expansion, because that is where the SSE2
                    // kernel lives — 3.9 ns against the scalar form's
                    // 16.0 ns. Its accumulation order is the scalar form's
                    // term for term, so this path stays bit-identical to
                    // the full expansion below AND to instance_expand.comp,
                    // the `precise` GPU replica that instanceExpandCheck
                    // compares against bit for bit.
                    const auto& arr = sp.inst->instanceMatrix()->array();
                    // mw is this matrix's element array; binding the
                    // Matrix4 itself keeps the mesh world out of the
                    // per-instance copy entirely.
                    const Matrix4& meshWorld = *sp.mesh->matrixWorld;
                    Matrix4 instMat;
                    Matrix4 world;
                    for (uint32_t j = 0; j < sp.count; ++j) {
                        std::memcpy(instMat.elements.data(), arr.data() + size_t(j) * 16u, 64);
                        world.multiplyMatrices(meshWorld, instMat);
                        float* wm = lastVisibleEntries_[sp.first + j].worldMatrix.data();
                        if (sp.instancesMoved && std::memcmp(wm, world.elements.data(), 64) != 0) {
                            const uint32_t i = sp.first + j;
                            meshMovedBits_[i >> 5] |= (1u << (i & 31u));
                        }
                        std::memcpy(wm, world.elements.data(), 64);
                        // Keep the fingerprint's matrix mirror fresh so a
                        // later non-lean frame's generic compare sees this
                        // frame's state, not a stale lean-era matrix.
                        prevSceneFingerprint[sp.first + j].matrix =
                                lastVisibleEntries_[sp.first + j].worldMatrix;
                    }
                } else {
                    if (std::memcmp(sp.meshWorld.data(), mw, 64) == 0) continue;
                    std::memcpy(sp.meshWorld.data(), mw, 64);
                    sp.movedThisFrame = true;
                    sp.aabbValid = false;
                    sp.probeChunksValid = 0;
                    std::memcpy(lastVisibleEntries_[sp.first].worldMatrix.data(), mw, 64);
                    prevSceneFingerprint[sp.first].matrix = lastVisibleEntries_[sp.first].worldMatrix;
                }
            }
        }

bool VulkanRenderer::Impl::classifyCutoutFar(const float camPos[3]) {
    const float dist = cutoutRayDistance_;
    // Off, and no bit left set from a time it was on.
    if (dist <= 0.f && cutoutRayClassifiedFor_ <= 0.f) return false;
    if (cutoutRayClassified_ && cutoutRayClassifiedFor_ == dist) {
        const float dx = camPos[0] - cutoutRayCamPos_[0], dy = camPos[1] - cutoutRayCamPos_[1],
                    dz = camPos[2] - cutoutRayCamPos_[2];
        const float step = 0.05f * dist;
        if (dx * dx + dy * dy + dz * dz < step * step) return false;
    }
    auto& entries = lastVisibleEntries_;
    if (geomDescsCached_.size() != entries.size()) return false;// no materials to class by yet
    THREEPP_CPUPROF("scene.8b_cutoutFar");
    bool changed = false;
    for (const auto& sp : entrySpans_) {
        const MeshEntry& e0 = entries[sp.first];
        if (e0.isOverlay || e0.isParticle) continue;
        const bool caster = dist > 0.f && cutoutCaster(entryMat(sp.first));
        for (uint32_t j = 0; j < sp.count; ++j) {
            MeshEntry& en = entries[sp.first + j];
            const bool beyond = caster && entryBeyond(en, camPos, dist);
            if (beyond != en.rayFarOpaque) {
                en.rayFarOpaque = beyond;
                changed = true;
            }
        }
    }
    cutoutRayClassified_    = true;
    cutoutRayClassifiedFor_ = dist;
    std::copy(camPos, camPos + 3, cutoutRayCamPos_);
    return changed;
}

void VulkanRenderer::Impl::cacheCullFlags() {
            lastVisibleCullMode_.resize(geomDescsCached_.size());
            for (size_t i = 0; i < geomDescsCached_.size(); ++i) {
                switch (entryMat(i).sideMode) {
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
                e.pointMix       = sc->pointMix();
                e.pointSigma     = sc->pointSigmaPixels();
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
            //
            // How to OBSERVE it, because the obvious way used to lie: reading
            // the SplatDepth AOV back. The latch leaves splatDepthMode_ at Off
            // (it wants the image, not the app's statistic), and both readback
            // paths gated on splatDepthAov() — mode != Off — rather than on
            // splatDepthAovAllocated(). So a scene where this latch HAD fired
            // and the stamp WAS occluding overlays still reported "splat_depth:
            // not allocated", which reads as "the latch never fired". Fixed at
            // both gates; SplatOverlayOcclusion_probe measures the occlusion
            // itself (pixels, not a flag) and covers the LOD / submit-ranges
            // cloud, a secondary view, and the overlay arriving frames after
            // the cloud.
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


void VulkanRenderer::Impl::setGeomInstanceColor(GeometryDesc& gd, const MeshEntry& en) {
            gd.instanceColor[0] = gd.instanceColor[1] = gd.instanceColor[2] = 1.0f;
            if (!en.isInstanced) return;
            const auto* attr = static_cast<const InstancedMesh*>(en.mesh)->instanceColor();
            if (!attr) return;
            const auto& a = attr->array();
            const size_t o = static_cast<size_t>(en.instanceIndex) * 3u;
            if (o + 3u > a.size()) return;
            gd.instanceColor[0] = a[o];
            gd.instanceColor[1] = a[o + 1];
            gd.instanceColor[2] = a[o + 2];
        }

void VulkanRenderer::Impl::matTableReset() {
            matTable_.assign(1, MaterialDesc{});
            matTableKeys_.assign(1, MatTableKey{});
            matTableRefs_.assign(1, 0u);
            matTableStamp_.assign(1, 0u);
            matTableDormant_.clear();
            matTableDormantSeq_.assign(1, 0u);
            matTableIndex_.clear();
            matTableBeginPass();
        }

uint32_t VulkanRenderer::Impl::matTableSlot(const Material* mat, const MaterialDesc& md) {
            MatTableKey key;
            key.mat = mat;
            key.matId = mat ? mat->id : 0u;
            // Derived already this pass (the memo is dropped with the pass, and
            // matTableAcquire never reclaims the record it names).
            if (matTableMemoSlot_ != 0u && key == matTableMemoKey_) return matTableMemoSlot_;
            const auto it = matTableIndex_.find(key);
            const uint32_t s = it != matTableIndex_.end() ? it->second : matTableAcquire(key);
            if (matTableStamp_[s] != matTableEpoch_) {
                matTableStamp_[s] = matTableEpoch_;
                // Bytes, not fields: the record is what the GPU reads, and a
                // record a pass leaves as it was costs no upload.
                if (std::memcmp(&matTable_[s], &md, sizeof(MaterialDesc)) != 0) {
                    matTable_[s] = md;
                    markMatDescsDirty(s);
                }
            }
            matTableMemoKey_ = key;
            matTableMemoSlot_ = s;
            return s;
        }

uint32_t VulkanRenderer::Impl::matTableAcquire(const MatTableKey& key) {
            if (!matTableBuilding_ && matTable_.size() >= matTableCapacity_) {
                while (!matTableDormant_.empty()) {
                    const auto [s, seq] = matTableDormant_.front();
                    matTableDormant_.pop_front();
                    if (matTableRefs_[s] != 0u || matTableDormantSeq_[s] != seq ||
                        matTableStamp_[s] == matTableEpoch_ || s == matTableMemoSlot_)
                        continue;
                    matTableIndex_.erase(matTableKeys_[s]);
                    matTableKeys_[s] = key;
                    matTable_[s] = MaterialDesc{};
                    matTableIndex_.emplace(key, s);
                    return s;
                }
                matTableGrow(std::max<uint32_t>(matTableCapacity_ * 2u, 256u));
            }
            const auto s = static_cast<uint32_t>(matTable_.size());
            matTable_.push_back(MaterialDesc{});
            matTableKeys_.push_back(key);
            matTableRefs_.push_back(0u);
            matTableStamp_.push_back(0u);
            matTableDormantSeq_.push_back(0u);
            matTableIndex_.emplace(key, s);
            return s;
        }

void VulkanRenderer::Impl::matTableUnref(uint32_t idx) {
            if (idx == 0u || idx >= matTableRefs_.size() || matTableRefs_[idx] == 0u) return;
            if (--matTableRefs_[idx] != 0u) return;
            // No entry names it any more: dormant, kept for its key's return.
            matTableDormant_.emplace_back(idx, ++matTableDormantSeq_[idx]);
        }

void VulkanRenderer::Impl::matTableGrow(uint32_t capacity) {
            THREEPP_CPUPROF("scene.8d_matTableGrow");
            // growSlotPool's model: the frames in flight keep the old buffers
            // (retired with this frame's serial), every slot gets a new one
            // holding the whole host table, and each frame's descriptor sets
            // are rewritten after its own fence.
            matTableCapacity_ = capacity;
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                retire(std::move(materialDescsBuffers[f]));
                uploadDescBuffer(materialDescsBuffers[f], matTable_, matTableCapacity_);
            }
            for (auto& d : matDescsDirty_) d.clear();
            deferredDescDirty_.fill(true);
            for (auto& v : views_) {
                v->rasterMatTexValid_.fill(0);
                v->shadowMatTexValid_.fill(0);
            }
        }

uint32_t VulkanRenderer::Impl::packLodFade(float fade, bool complement) {
            // The complement side is cut at 1 - fade, so the plain f and the
            // complement 1 - f of one crossfade land on the same cut and the
            // shader's below / at-or-above split partitions the pixels.
            if (!(fade < 1.f)) return 0u;// NaN too, as Mesh::setFade reads it
            const float share = complement ? 1.f - fade : fade;
            const float c = std::clamp(share, 0.f, 1.f);
            const auto cut = static_cast<uint32_t>(std::lround(c * float(kLodFadeSteps)));
            // Fully drawn reads as no fade at all: the shader's zero test.
            if (complement ? cut == 0u : cut >= kLodFadeSteps) return 0u;
            return kLodFadeActive | (complement ? kLodFadeComplement : 0u) | (cut << kLodFadeCutShift);
        }

VulkanRenderer::Impl::LodFadeChange VulkanRenderer::Impl::refreshLodFades(bool all) {
            LodFadeChange out;
            const auto& entries = lastVisibleEntries_;
            // A structural rebuild renumbers the entries: start from "no fade"
            // rather than compare against another entry's word.
            if (all || entryLodFade_.size() != entries.size()) {
                entryLodFade_.assign(entries.size(), 0u);
                all = true;
                ++lodFadeVersion_;
                markLodFadeWhole();
            }
            for (auto& sp : entrySpans_) {
                if (!sp.mesh || sp.count == 0u) continue;
                const FloatBufferAttribute* attr = sp.inst ? sp.inst->instanceFade() : nullptr;
                const unsigned int ver = attr ? attr->version : ~0u;
                const float meshFade = sp.mesh->fade();
                const bool meshComp = sp.mesh->fadeComplement();
                if (!all && sp.fadeKnown && attr == sp.fadeAttr && ver == sp.fadeVersion &&
                    meshFade == sp.meshFade && meshComp == sp.meshFadeComplement)
                    continue;
                sp.fadeKnown = true;
                sp.fadeAttr = attr;
                sp.fadeVersion = ver;
                sp.meshFade = meshFade;
                sp.meshFadeComplement = meshComp;
                // Overlays, sprites and particle fields are not G-buffer meshes:
                // no dither reaches them, so nothing may hide them from rays.
                const MeshEntry& e0 = entries[sp.first];
                const bool eligible = !(e0.isOverlay || e0.isParticle || e0.isParticleField);
                const uint32_t meshWord = eligible ? packLodFade(meshFade, meshComp) : 0u;
                const std::vector<float>* fa = (attr && eligible) ? &attr->array() : nullptr;
                bool any = false;
                bool spanChanged = false;
                for (uint32_t j = 0; j < sp.count; ++j) {
                    const size_t i = size_t(sp.first) + j;
                    uint32_t w = meshWord;
                    if (fa) {
                        const size_t o = size_t(entries[i].instanceIndex) * 2u;
                        w = o + 2u <= fa->size() ? packLodFade((*fa)[o], (*fa)[o + 1] != 0.f) : 0u;
                    }
                    const uint32_t was = entryLodFade_[i];
                    if (w != was) {
                        out.words = true;
                        spanChanged = true;
                        if (lodFadeCasts(w) != lodFadeCasts(was)) out.casters = true;
                        entryLodFade_[i] = w;
                    }
                    any = any || w != 0u;
                }
                sp.fadeAny = any;
                if (spanChanged) markLodFadeDirty(sp.first, sp.count);
            }
            if (out.casters) ++lodFadeCastEpoch_;
            if (out.words) ++lodFadeVersion_;
            return out;
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
            // ior of a surface that is NOT refractive at all (flat alpha blend,
            // additive glow). Shader twin: isFlatAlphaBlend().
            constexpr float kFlatBlendIor = 0.0f;
            d.transmissionTexIndex = -1;
            d.clearcoat = 0.0f;// no coat by default; lobe is skipped when clearcoat == 0
            d.clearcoatRoughness = 0.0f;
            d.clearcoatTexIndex = -1;
            d.clearcoatRoughnessTexIndex = -1;
            d.attenuationColor[0] = d.attenuationColor[1] = d.attenuationColor[2] = 1.0f;
            d.attenuationDistance = 0.0f;
            d.scatterColor[0] = d.scatterColor[1] = d.scatterColor[2] = 0.0f;
            d.scatterDistance = 0.0f;// 0 = in-scatter off; shader keeps the old path
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
            d.ripples = 0.f;
            d.terrainWeightTexIndex = -1;
            d.terrainNormalTexIndex = -1;
            d.terrainBandStrength = 0.f;
            d.terrainNormalScale = 1.f;
            d.terrainRoughStrength = 0.f;
            d.terrainHeightBlend = 0.f;
            d.envMapIntensity = 1.f;// unscaled IBL: what every material that never sets it carries
            d.aoMapIntensity = 1.f;
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
                // the background through. Fold the coverage into BOTH knobs
                // the deferred glass path consumes. It composites
                //   (1−t)·diffuse + t·tint·behind,
                // and the spec composite of the same glass under coverage α is
                //   α(1−t)·diffuse + (α·t·tint + (1−α))·behind,
                // so t' = 1 − α(1−t) and tint' = (α·t·tint + 1−α)/t' reproduce
                // it exactly for the dominant straight-through path, with no
                // second blend pass in either pipeline. (Folding the tint alone
                // left the opaque diffuse term at its full 1−t weight: a
                // jelly cube with t=0.59, α=0.41 shaded 41% solid green over
                // a 35% background instead of 17% over 59%, and the skeleton
                // inside all but vanished.)
                if (d.transmission > 0.0f && d.transmission <= 1.0f && mat->transparent && mat->opacity < 1.0f) {
                    const float a  = std::clamp(mat->opacity, 0.0f, 1.0f);
                    const float t  = d.transmission;
                    const float tp = 1.0f - a * (1.0f - t);
                    for (float& c : d.albedo) {
                        c = (a * t * c + (1.0f - a)) / tp;
                    }
                    d.transmission = tp;
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
                d.ior          = kFlatBlendIor;
            }
            // Alpha-blend transparency (transparent=true, opacity<1) has no
            // physical analogue in a ray tracer, so treat it as pass-through:
            // (1-opacity) of what is behind, straight on, no Fresnel split.
            // Deferred reads ior == 0 as the "flat alpha blend" marker (see
            // isFlatAlphaBlend in the shade). It was ior = 1, which is a value a
            // real material can have: KHR_materials_ior 1.0 with transmission 1
            // (the front of the glTF sample SunglassesKhronos lenses) was read as
            // "alpha blend at opacity 0" and drawn invisible, tint and all. A real
            // ior is clamped to >= 1 above, so 0 cannot collide.
            if (d.transmission == 0.0f && mat->transparent && mat->opacity < 1.0f) {
                d.transmission = 1.0f - mat->opacity;
                d.ior          = kFlatBlendIor;
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
            if (mat->transparent && d.alphaCutoff == 0.0f) {
                if (d.transmission == 0.0f) {
                    d.alphaCutoff = (!mat->depthWrite && mat->polygonOffset) ? -2.0f : -1.0f;
                } else if (auto* mm = dynamic_cast<MaterialWithMap*>(mat.get()); mm && mm->map) {
                    // BLEND with BOTH a flat opacity/transmission (handled above)
                    // AND a texture: the texel alpha is still the surface's
                    // coverage (a ground-fog card = opacity 0.87 over a soft
                    // alpha mask). Without the marker the raster wrote the
                    // whole quad opaque and only the flat term blended, so
                    // every such card rendered as a solid plate. Plain -1
                    // (never the decal -2): the decal pipeline blends albedo
                    // only and would drop the flat term.
                    d.alphaCutoff = -1.0f;
                }
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
                d.scatterColor[0] = att->scatterColor.r;
                d.scatterColor[1] = att->scatterColor.g;
                d.scatterColor[2] = att->scatterColor.b;
                // Negative would make sigma_s negative (radiance sink); clamp.
                d.scatterDistance = std::max(0.0f, att->scatterDistance);
            }
            if (auto* em = dynamic_cast<MaterialWithEnvMap*>(mat.get())) {
                // The deferred path has no per-material env MAP — one scene
                // environment lights everything — so only the INTENSITY is
                // meaningful here, and only where this material is the primary
                // surface. Negative values would flip the sign of a radiance,
                // so the knob is clamped at zero and left open above 1.
                d.envMapIntensity = std::max(0.f, em->envMapIntensity);
            }
            if (auto* st = dynamic_cast<MeshStandardMaterial*>(mat.get())) {
                d.ripples = std::max(0.f, st->ripples);
            }
            if (auto* ao = dynamic_cast<MaterialWithAoMap*>(mat.get())) {
                // The index is bound at the texture-index fill sites
                // (occlusionTexOf). Negative would brighten the indirect light
                // through the map; above 1 deepens it, as in three.js.
                d.aoMapIntensity = std::max(0.f, ao->aoMapIntensity);
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
