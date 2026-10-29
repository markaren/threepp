// VulkanSceneTypes — the records the Vulkan renderer derives from the scene
// graph once per frame: the entry list (one item per TLAS instance), the
// traversal snapshot that lets a static scene skip re-deriving it, the
// line/sprite/particle side lists, and the small cache-value types. Split out
// of VulkanCoreImpl.hpp; each struct is aliased back into VulkanRenderer::Impl
// at its original spot so every reference site — Impl's own methods and the
// VulkanCore*.cpp TUs — is unchanged.

#ifndef THREEPP_VULKAN_SCENE_TYPES_HPP
#define THREEPP_VULKAN_SCENE_TYPES_HPP

#include "VulkanResources.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/textures/Texture.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace threepp::vulkan::impl {

    // Cache value: (weak_ptr liveness tag, bindless slot, uploaded version).
    // The weak_ptr lets ensureSceneBuilt prune entries when a Texture is
    // destroyed (so a future Texture* address-collision doesn't read stale
    // GPU data). `version` is the Texture::version() at last upload; when it
    // changes (setData + needsUpdate on a DataTexture), refreshDirtyMaterialTextures
    // re-uploads the slot in place so live texture edits are honoured.
    struct CachedTexture {
        std::weak_ptr<Texture> ref;
        uint32_t slot = 0;
        unsigned int version = 0;
    };

    // Unit of work for ray tracing: a single TLAS instance. A regular Mesh
    // expands to one MeshEntry; an InstancedMesh expands to N entries (one
    // per sub-instance) all sharing the same Mesh*/BLAS but with distinct
    // worldMatrix = mesh->matrixWorld * instanceMatrix[i].
    // MeshEntry::rayShadowState bits. Far: beyond Object3D::rayShadowDistance;
    // Near: nearer than rayShadowFrom (both: no shadow or GI ray sees it).
    // Proxy: wholly beyond rayShadowFrom, so it casts on either side of a LOD
    // crossfade. CastOn / CastOff: Object3D::rayCastShadow.
    inline constexpr uint8_t kRayShadowFar   = 1u;
    inline constexpr uint8_t kRayShadowNear  = 2u;
    inline constexpr uint8_t kRayShadowProxy = 4u;
    inline constexpr uint8_t kRayCastOn      = 8u;
    inline constexpr uint8_t kRayCastOff     = 16u;

    struct MeshEntry {
        Mesh*    mesh;
        std::array<float, 16> worldMatrix;
        uint32_t instanceIndex;// 0 for non-instanced
        // Hybrid overlay flag: wireframe-flagged material OR mesh.layers
        // includes the configured overlayLayer_. Excluded from TLAS,
        // raster G-buffer, and emissive-tri NEE so the traced/rasterized
        // scene can't see/shadow them; drawn instead by the post-TAA
        // overlay pass.
        bool     isOverlay = false;
        // ParticleSystem billboard mesh (material name == kParticleMaterialName).
        // Implies isOverlay (so all the scene-exclusion guards apply), but is
        // drawn by the dedicated billboard particle pass — NOT the wireframe/
        // basic overlay-mesh loop, which would render its un-expanded quads as
        // zero-area triangles. The overlay-mesh loop skips on this flag.
        bool     isParticle = false;
        // The mesh was hidden at run time (Object3D::visible or
        // material()->visible false) after it had entries: the entry keeps
        // its slot, so the TLAS instance set, the descriptor rings and every
        // index stay put, and showing it again is just as cheap. Ray mask 0
        // (no ray hits it), culled from every raster view, no emitter. A mesh
        // hidden before it was ever shown has no entry at all, as before,
        // unless it is Object3D::resident: then it has its entries, hidden,
        // from its first expansion. Mirrors EntrySpan::hidden (a per-mesh
        // fact).
        bool     hidden = false;
        // Geometry rigidly parented UNDER a Camera — a first-person
        // viewmodel (hands, weapon, cockpit). Rasterized and shaded like
        // any other mesh, and still visible to primary/radiance traces
        // (cullMask 0xFF), but tagged kRayMaskNoShadow in the TLAS so no
        // occlusion query can see it: a viewmodel that casts sun shadows
        // paints floating hands and a gun on the floor next to the player.
        // Resolved by an ancestor walk at full expansion (a reparent is a
        // structure change ⇒ full re-expansion, so it can't go stale).
        // NOTE: castShadow can't drive this — it defaults to FALSE on
        // Object3D and no loader sets it, so honouring the flag in the RT
        // path would delete the shadows of every loaded scene.
        bool     camAttached = false;
        // Mesh on VulkanRenderer::kSensorOnlyLayer: geometry the SENSORS may
        // perceive and the camera may not (a surface baked from a splat scan —
        // threepp::splats::bakeSurface). Never rasterized into the primary
        // view and never in a radiance trace's cull mask; rasterized into
        // secondary views and hit by the lidar's 0xFF only once
        // setSensorOnlySurfaces(true) opts the scene in, before which its TLAS
        // instance carries mask 0 and nothing at all can see it.
        bool     sensorOnly = false;
        // Cached type probes. Resolved once per Mesh in ensureSceneBuilt's
        // traverseVisible callback (before the InstancedMesh fork so an
        // N-instance mesh costs 3 dynamic_casts, not 3·N). Consumers
        // (resolveBlasForEntry, recordRasterGbufPass, TLAS refit, bone /
        // displaced / morph dirty-detection) read these flags instead of
        // casting every frame.
        bool     isSkinned   = false;
        bool     isDisplaced = false;
        bool     isGrass     = false;// GPU wind-deformed grass field
        bool     isMorphed   = false;
        bool     isTet       = false;
        // threepp::ParticleField — a whole particle field (up to millions of
        // particles) as ONE entry. Its span has inst == nullptr and count == 1,
        // and its world AABB is NOT derivable on the CPU (the positions are on
        // the device), so it is never frustum-culled at entry granularity and
        // carries the occlusion "always draw" bit, exactly like the deformers.
        bool     isParticleField = false;
        // VulkanRenderer::enableVertexInterop is armed on this mesh's geometry
        // (BlasRecord::interop). It belongs beside the deformer flags for the
        // same reason isParticleField does, and for the STRONGER of the two
        // reasons: a foreign device producer owns the positions, so the host
        // `position` array — which is what every CPU bounds computation reads —
        // describes a shape that is not on screen. Not merely stale like a
        // skinned mesh's rest AABB, but arbitrary: a producer is free to leave
        // the host array at the origin, or at y=-50, or empty. Both CPU-bounds
        // culls therefore treat an interop entry the way they treat a deformer.
        //
        // Refreshed on the structural rebuild that enable/disableVertexInterop
        // force, exactly as isMorphed is (see the snapshot note below) — the
        // snapshot fast path never recomputes it.
        bool     isVertexInterop = false;
        // Its material's displacementMap displaces the record's positions
        // (Impl::noteDisplacement): the slot paths decline it, the geometry-
        // edit refresh rebuilds it whole, auto-LOD leaves it alone.
        bool     isDisplacementMapped = false;
        bool     isInstanced = false;// entry came from an InstancedMesh expansion
                                     // (the snapshot fast path recomputes its world
                                     // matrix as matrixWorld * getMatrixAt(i))
        // (The frustum-cull bit used to live here. It is now
        //  ViewContext::inFrustum, indexed in lockstep with
        //  lastVisibleEntries_, because the answer depends on WHICH CAMERA
        //  is asking and an entry is shared by all of them. Read it through
        //  viewCulled(i).)
        // Automatic mesh LOD (setAutoLod): 0 == the geometry's own
        // (finest) buffers; N>0 selects BlasRecord::lodLevels[N-1].
        // Written once per frame by the LOD selection pass in
        // ensureSceneBuilt and read verbatim by both buildIndirectDrawData
        // (raster) and the TLAS instance fill — never re-derived. Carries
        // hysteresis state across lean (snapshot-fast-path) frames since
        // entries persist in lastVisibleEntries_; resets to 0 on a full
        // re-expansion (rare, and the selection pass re-picks it that
        // same frame anyway).
        uint8_t  lodLevel    = 0;
        // Cutout ray distance (setCutoutRayDistance): this entry's cutout
        // material is forced opaque for rays, its bounds being farther from
        // the camera than the distance. Written by classifyCutoutFar (lean
        // frames) and the full rebuild's instance loop, read by both TLAS
        // instance fills. Resets to false on a full re-expansion, which
        // un-classifies (cutoutRayClassified_) so the bits are re-derived.
        bool     rayFarOpaque = false;
        // Ray shadow level of detail (Object3D::rayShadowDistance, rayShadowFrom,
        // rayCastShadow as the span resolves them): kRayShadow* bits, written by
        // classifyRayShadow, read by both TLAS fills (tlasRayMask, entryRayMask).
        uint8_t  rayShadowState = 0;
        // Auto-LOD selection caches, derived ONCE at full expansion so the
        // per-frame selection pass is pure float math — the uncached
        // version paid a dynamic_cast + ancestor hash-walk + shared_ptr
        // derefs PER ENTRY PER FRAME (~2-4 ms on 4k-entry scenes; measured
        // as the whole feature's CPU cost on Bistro/fjord).
        //   lodExemptStatic — mesh opted out (Object3D::autoLod == false,
        //     e.g. TileTerrain tiles, which are their own LOD system) or
        //     sits under a threepp::LOD subtree
        //     (structure changes force a full expansion ⇒ can't go stale).
        //   lodEmissive — cached MaterialWithEmissive cast (material
        //     POINTER swaps force a full expansion). VALUES are read live
        //     each frame, so an emissive that turns on later (dusk-driven
        //     lantern intensity) exempts immediately, cast-free.
        //   lodGeomKey — blasCache key (geometry pointer swaps force a
        //     full expansion). Also the key the between-frame calls match a
        //     geometry's entries by, without following `mesh` (see
        //     VulkanRenderer::Impl::lastVisibleEntries_).
        //   lodCenter/lodRadius — object-space bounding sphere; radius 0 ⇒
        //     unknown ⇒ entry stays LOD0. lodSphereDirty re-derives it
        //     after an in-place geometry edit (set by the geom-dirty
        //     detection, consumed lazily by the next selection pass).
        bool lodExemptStatic = false;
        bool lodSphereDirty    = false;
        const MaterialWithEmissive* lodEmissive = nullptr;
        const BufferGeometry* lodGeomKey = nullptr;
        float lodCenter[3] = {0.f, 0.f, 0.f};
        float lodRadius = 0.f;
    };

    // ── Scene-structure SNAPSHOT (ensureSceneBuilt fast path) ────────────
    // One node per traverseVisible visit from the last full expansion, in
    // visit order. ensureSceneBuilt first REPLAYS the traversal comparing
    // only cheap invariants (object/geometry/material pointers + the flags
    // that route classification); when the whole sequence matches, last
    // frame's entries + fingerprints are reused with refreshed matrices and
    // live version reads instead of being re-derived (the per-mesh
    // dynamic_cast storm + texture lookups + allocations cost ~9 ms/frame
    // on Bistro's ~1500 meshes — on a completely STATIC scene). Any
    // mismatch anywhere falls back to the full expansion, so correctness
    // never depends on the snapshot: worst case we rebuild, exactly like
    // every frame did before. three.js polling semantics are preserved —
    // transform/material/geometry mutations are still picked up by the
    // value/version diffs every frame; no user-side notification calls.
    // KNOWN EDGE (accepted as rare-as-asset-restructuring): morph
    // attributes ADDED to an already-seen geometry keep the cached
    // isMorphed=false until any structural change rebuilds the snapshot
    // (position/normal additions ARE detected via the flag bits below).
    struct SnapNode {
        Object3D* obj = nullptr;
        // obj->parent when recorded: the next expansion reuses this node's
        // record only while its whole parent chain is the one it was.
        Object3D* parent = nullptr;
        // Process-unique ids of obj, its geometry and its material, compared
        // alongside the pointers. A pointer alone cannot tell the object it was
        // recorded from apart from a NEW one the allocator placed at the dead
        // one's address, and everything below leans on "same address ⇒ same
        // object": the typed views would be the wrong type, and the recycled
        // material's parameters would never be read. Replacing a mesh (or
        // rebuilding a scene) frees and allocates same-sized objects back to
        // back, which is exactly when addresses repeat.
        // Left 0 where there is no geometry / material: the pointer compare
        // already separates null from non-null, so the id only has to decide
        // between two objects at one address.
        unsigned int objId  = 0;
        unsigned int geomId = 0;
        unsigned int matId  = 0;
        // Typed views of obj, resolved by the full pass's dynamic_casts.
        // Object3D is a VIRTUAL base, so the replay walk cannot
        // static_cast down — it reuses these instead (same object at the
        // same address ⇒ same dynamic type ⇒ pointers still valid).
        Mesh*          mesh = nullptr;
        InstancedMesh* inst = nullptr;
        Line*          line = nullptr;
        Points*        pts  = nullptr;
        LOD*           lod  = nullptr;// kSnapLod nodes: replay walk re-runs level selection
        const void* geom = nullptr;               // mesh/line/points geometry
        BufferGeometry* geomB = nullptr;          // typed view of geom (attributesVersion reads)
        const Material* mat = nullptr;            // mesh material
        const MaterialWithWireframe* wf = nullptr;// cached cast of mat (same object ⇒ still valid)
        const MeshBasicMaterial* basic = nullptr; // cached cast of mat (kSnapUiBlend replay reads)
        int32_t instCount = -1;                   // InstancedMesh count; -1 = plain Mesh
        // The EntrySpan this mesh node produced (-1: none). A visibility flip
        // of such a node toggles EntrySpan::hidden in place.
        int32_t spanIndex = -1;
        uint32_t flags = 0;                       // kind(2b) | attr/wire/overlay/tet bits
        // BufferGeometry::attributesVersion() at record time. Unchanged ⇒
        // no attribute was added/replaced/removed ⇒ the hasPos/hasNorm
        // flag bits are still valid WITHOUT re-doing the string-keyed
        // hasAttribute lookups (2/mesh/frame — ~1 ms on Bistro).
        unsigned int attrVer = 0;
        // One past the last node of this node's subtree, in the snapshot's
        // pre-order: the edit-proportional walk (diffSceneWalk) finds a child
        // among its parent's old children, and skips a removed subtree, by
        // these ranges. 0 = not recorded (a snapshot the diff walk declines).
        uint32_t subtreeEnd = 0;
        // obj->structureVersion() when recorded (diffSceneWalk copies a subtree
        // whose version has not moved), and the parent's index in the snapshot
        // (-1: the root), for flatSnapshotCheck's effective visibility.
        uint64_t ver = 0;
        int32_t parentIdx = -1;
    };

    // One contiguous run of entries sharing a single source Mesh: an
    // InstancedMesh expansion (count() entries) or a single-entry span for a
    // plain mesh. Rebuilt at every FULL expansion, in entry order; the
    // snapshot fast path guarantees the structure is unchanged on lean
    // frames, so spans stay aligned with lastVisibleEntries_.
    //
    // Purpose: every per-frame loop over the entry list used to pay per-MESH
    // costs per INSTANCE (a 100k-grain InstancedMesh re-read its material
    // version, geometry versions, blasCache slot and emissive cast 100k times
    // a frame — measured ~0.3-4 µs/instance across the loops). All per-mesh
    // facts those loops consult now live here, so they cost O(spans), and the
    // remaining per-entry work (matrix refresh, TLAS instance fill, draw
    // fill, cull tests) runs only for spans whose inputs actually changed.
    struct EntrySpan {
        Mesh*          mesh = nullptr;
        InstancedMesh* inst = nullptr;// typed view resolved at expansion; null = plain mesh
        uint32_t first = 0;           // first entry index in lastVisibleEntries_
        uint32_t count = 0;
        unsigned int meshId = 0;      // mesh->id at expansion (dangling-pointer guard)
        bool hidden = false;          // see MeshEntry::hidden
        // A hidden mesh new to the renderer (Object3D::resident) whose slot
        // edit placed it without its assets: no BLAS record yet, default rows,
        // its textures being prepared on worker threads. pumpHiddenAdmissions
        // lands them over the following frames and clears this; shown before
        // that, the span is placed again with today's synchronous admission.
        bool admitPending = false;
        // Matrix change detection. A span's entry world matrices are a pure
        // function of (mesh->matrixWorld, instanceMatrix contents); the lean
        // path refreshes them ONLY when one of these moved. instanceMatrix
        // edits are detected by BufferAttribute::version — i.e. the user must
        // call instanceMatrix()->needsUpdate() after setMatrixAt, exactly the
        // contract the GL backend's attribute upload already requires.
        std::array<float, 16> meshWorld{};// mesh->matrixWorld at last refresh
        unsigned int instMatVersion = ~0u;// instanceMatrix()->version at last refresh
        // instanceColor state last folded into this span's MaterialDescs.
        // Same contract as instMatVersion: setColorAt + instanceColor()->
        // needsUpdate(). The attribute pointer is tracked too because
        // InstancedMesh creates it lazily on the first setColorAt. Left at
        // the defaults by the expansion; the full rebuild records it, and the
        // diff records it when it patches the descs.
        const FloatBufferAttribute* instColorAttr = nullptr;
        unsigned int instColorVersion = ~0u;
        // The LOD fade source last packed into entryLodFade_ for this span
        // (refreshLodFades): InstancedMesh::instanceFade() pointer + version,
        // and the mesh-wide Mesh::fade / fadeComplement. fadeAny: some entry
        // of the span carries a fade word, so the TLAS fill must ask per entry.
        // fadeKnown is false on a span fresh from an expansion, which a lean
        // diff may still follow: its words are re-packed whatever the source.
        bool fadeKnown = false;
        const FloatBufferAttribute* fadeAttr = nullptr;
        unsigned int fadeVersion = ~0u;
        float meshFade = 1.f;
        bool meshFadeComplement = false;
        bool fadeAny = false;
        bool movedThisFrame = false;     // set by the lean refresh; consumed by
                                          // the diff, motion matrices and TLAS refit
        // With movedThisFrame: only the instance matrices moved, not the
        // mesh's own transform. The refresh then marked the entries whose
        // world matrix changed in meshMovedBits_ itself, and the diff leaves
        // the rest of the span unmarked: an InstancedMesh used as a pool of
        // slots (a forest's level meshes, re-sent whole when one slot changes
        // owner) keeps its other instances out of the moved-sticky state, and
        // so their shadow, GI and reflection histories and their static
        // shadow-cache casters.
        bool instancesMoved = false;
        // True while the motionMat scratch holds non-identity blocks for this
        // span (set when a moved span writes real deltas; the next static
        // frame resets its blocks to identity once and clears this).
        bool motionNonIdentity = false;
        // True while any entry in the span carries lodLevel > 0 — lets the
        // auto-LOD selection (and the feature-off reset) skip the per-entry
        // "snap back to 0" walk for spans that were never lifted off LOD0.
        bool lodNonZero = false;
        // Span-wide world AABB: union of the entries' world bounding spheres,
        // recomputed whenever the span's matrices are refreshed. Lets the
        // frustum cull answer fully-inside / fully-outside once per span and
        // fall back to per-entry sphere tests only on partial intersection.
        float aabbMin[3]{}, aabbMax[3]{};
        bool  aabbValid = false;
        // Chunked world AABBs (fixed-size runs of entries) for large instanced
        // spans, rebuilt together with the span AABB. The frustum cull
        // classifies a whole chunk at a time and runs per-entry tests only
        // for chunks that straddle the frustum. min/max packed as [0..2]/[3..5].
        std::vector<std::array<float, 6>> chunkAabbs;
        // Each instance's world bounding sphere (centre, radius), derived with
        // chunkAabbs: the per-instance frustum test reads these 16 bytes
        // instead of the entry's matrix.
        std::vector<std::array<float, 4>> cullSpheres;
        // Object-space bounds of the shared geometry (box + enclosing-sphere
        // radius), cached at expansion for the per-entry Arvo transform /
        // sphere tests. Invalidated (localBoundsValid=false) when the
        // geometry is edited in place, re-derived lazily.
        float localCenter[3]{}, localHalf[3]{};
        float localRadius = 0.f;
        bool  localBoundsValid = false;
        // Object3D::rayShadowDistance / rayShadowFrom as the mesh resolves them
        // (its own, else its nearest ancestor's; classifyRayShadow).
        float  rayShadowDist = 0.f;
        float  rayShadowFrom = 0.f;
        uint8_t rayCastBits  = 0;    // kRayCastOn / kRayCastOff from Object3D::rayCastShadow
        // Its rows in the TLAS instance array at the last lean fill (base ~0:
        // none) and the signature of what they were made from.
        uint32_t tlasInstBase  = ~0u;
        uint32_t tlasInstCount = 0;
        uint64_t tlasSig       = 0;
        bool     tlasVolatile  = false;// its row count changed once: placed after the stable spans
        bool   rayShadowLive = false;// a knob set, or entry states left from one
        // Stable entry slots (VulkanCoreSlots.cpp): a FREE span is a run of
        // entry slots no mesh owns. It is hidden, its entries are hidden and
        // its mesh is the renderer's tombstone, so every loop that already
        // skips a hidden span skips it; it pushes no TLAS instance. A freed run
        // is PARKED until frame serial parkedUntil: a readback of a frame still
        // in flight maps its entry indices through the current entries, and a
        // parked slot then names the tombstone, never an unrelated mesh.
        bool isFree = false;
        uint64_t parkedUntil = 0;
        // On a span the expansion walk built, never on a live one: the walk
        // pushed only this InstancedMesh's first entry (it already has slots,
        // and the slot path derives any rows it adds itself); `count` is still
        // the instance count. A walk the full rebuild consumes has none.
        bool rowsDeferred = false;
        // fitProbeGridToScene's unions of this span's entry world boxes, one
        // per kProbeChunk entries (min [0..2], max [3..5]; an empty chunk has
        // min > max), the first probeChunksValid of them current: all are
        // dropped by a matrix refresh or a geometry edit (wherever aabbValid
        // is), and a count change drops the chunks from the one it touches on.
        // min / max are exact, so the union of the chunks is the per-entry
        // walk's box bit for bit, and a re-fit after a slot edit costs
        // O(chunks), not O(entries).
        static constexpr uint32_t kProbeChunk = 512;
        std::vector<std::array<float, 6>> probeChunks;
        uint32_t probeChunksValid = 0;
    };

    struct EntryKey {
        const Mesh* mesh;
        uint32_t    instanceIndex;
        bool operator==(const EntryKey& o) const noexcept {
            return mesh == o.mesh && instanceIndex == o.instanceIndex;
        }
    };
    struct EntryKeyHash {
        size_t operator()(const EntryKey& k) const noexcept {
            const auto h1 = std::hash<const void*>{}(k.mesh);
            const auto h2 = std::hash<uint32_t>{}(k.instanceIndex);
            return h1 ^ (h2 + 0x9e3779b9u + (h1 << 6) + (h1 >> 2));
        }
    };

    // The animated attributes of one particle geometry, as one frame slot
    // binds them (see ParticleGeomRec).
    struct ParticleAnimSet {
        Buffer position;// vec3 — particle centers (all 4 quad verts equal)
        Buffer normal;  // vec3 — {size, angle, opacity}
        Buffer color;   // vec3 — RGB
        unsigned int version = ~0u;// pos+normal+color composite version held
    };

    // Per-BufferGeometry vertex/index buffers for particle billboards. Particles
    // own these buffers directly — they never build a BLAS. uv + index are
    // static (uploaded once). The animated attributes (position/normal/color)
    // follow the vulkan::LineCacheEntry scheme: one set while the geometry is
    // unchanged since its first upload; on its first change that set is
    // retired and each frame in flight gets its own, written only while that
    // frame slot is recorded, past its fence wait. One set rewritten in place
    // let the previous frame's draw read a half-written buffer (a tear between
    // old and new particles when the GPU is the bottleneck).
    struct ParticleGeomRec {
        std::vector<ParticleAnimSet> anim;// 1 while static, frames-in-flight once changed
        Buffer   uv;   // vec2 — corner offset
        Buffer   index;// uint32
        uint32_t indexCount  = 0;
        uint32_t vertexCount = 0;
        std::weak_ptr<BufferGeometry> liveCheck;

        // The set frame slot `frame` binds.
        [[nodiscard]] const ParticleAnimSet& animFor(uint32_t frame) const {
            return anim.size() == 1 ? anim[0] : anim[frame];
        }
    };

    // Per-Texture sampled image for particle textures. Keyed on the raw
    // Texture* (the ShaderMaterial uniform holds no shared_ptr) + version;
    // re-upload is vkDeviceWaitIdle-guarded. Pointer-recycle with an
    // identical version is a documented edge case.
    struct ParticleTexRec {
        Image2D      image{};
        unsigned int version = ~0u;
        uint32_t     width   = 0;
        uint32_t     height  = 0;
    };

    // Per-frame snapshot of visible world-space sprites, rebuilt each
    // perspective frame by collectWorldSprites() (sprites move/spawn/expire
    // constantly, so this is a fresh walk, not snapshot-cached).
    struct WorldSpriteEntry {
        std::array<float, 16> world;
        std::array<float, 4>  color;   // rgb + opacity
        Vector2               center;
        float                 rotation = 0.f;
        const Texture*        tex = nullptr;
    };

    // Per-Line scene snapshot, refreshed in ensureSceneBuilt alongside
    // lastVisibleEntries_. Lives only for the overlay record's draw
    // loop — neither the deferred shade nor the raster G-buffer touches this.
    struct LineEntry {
        // For Line / LineSegments the `line` pointer is the object; for
        // Points entries it is null and `points` holds the object instead.
        // Keeping a single entry struct (rather than a separate
        // PointEntry) avoids duplicating the overlay walk + geometry
        // upload paths, since both topologies share the same vertex
        // buffer layout (position + optional color).
        Line*    line;
        Points*  points;
        std::array<float, 16> worldMatrix;
        bool     isSegments; // true → LINE_LIST, false → LINE_STRIP (ignored when isPoints)
        bool     isPoints;   // true → POINT_LIST topology, overrides the line topology
    };

    struct OcclBitRange { uint32_t base; uint32_t capacity; };

    // Per-cull-mode dispatch span into indirectCmdBuffers[frame].
    struct DrawGroup {
        VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
        uint32_t        offset   = 0;// first cmd index (cmd-buffer-relative)
        uint32_t        count    = 0;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_SCENE_TYPES_HPP
