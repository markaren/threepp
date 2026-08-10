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
        //     full expansion).
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
        uint32_t flags = 0;                       // kind(2b) | attr/wire/overlay/tet bits
        // BufferGeometry::attributesVersion() at record time. Unchanged ⇒
        // no attribute was added/replaced/removed ⇒ the hasPos/hasNorm
        // flag bits are still valid WITHOUT re-doing the string-keyed
        // hasAttribute lookups (2/mesh/frame — ~1 ms on Bistro).
        unsigned int attrVer = 0;
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
        // Matrix change detection. A span's entry world matrices are a pure
        // function of (mesh->matrixWorld, instanceMatrix contents); the lean
        // path refreshes them ONLY when one of these moved. instanceMatrix
        // edits are detected by BufferAttribute::version — i.e. the user must
        // call instanceMatrix()->needsUpdate() after setMatrixAt, exactly the
        // contract the GL backend's attribute upload already requires.
        std::array<float, 16> meshWorld{};// mesh->matrixWorld at last refresh
        unsigned int instMatVersion = ~0u;// instanceMatrix()->version at last refresh
        bool movedThisFrame = false;      // set by the lean refresh; consumed by
                                          // the diff, motion matrices and TLAS refit
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
        // Object-space bounds of the shared geometry (box + enclosing-sphere
        // radius), cached at expansion for the per-entry Arvo transform /
        // sphere tests. Invalidated (localBoundsValid=false) when the
        // geometry is edited in place, re-derived lazily.
        float localCenter[3]{}, localHalf[3]{};
        float localRadius = 0.f;
        bool  localBoundsValid = false;
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

    // Per-BufferGeometry vertex/index buffers for particle billboards. The
    // animated attributes (position/normal/color) are re-uploaded every frame
    // (version-gated); uv + index are static (uploaded once). Particles own
    // these buffers directly — they never build a BLAS. Single-buffered with
    // in-place memcpy, exactly like ensureLineGeometryUploaded: a write-during-
    // read race with the other in-flight frame is benign for an overlay visual
    // (at most a sub-pixel tear on a fast-moving particle for one frame).
    struct ParticleGeomRec {
        Buffer   position;// vec3 — particle centers (all 4 quad verts equal)
        Buffer   normal;  // vec3 — {size, angle, opacity}
        Buffer   uv;      // vec2 — corner offset
        Buffer   color;   // vec3 — RGB
        Buffer   index;   // uint32
        uint32_t indexCount  = 0;
        uint32_t vertexCount = 0;
        unsigned int animVersion = ~0u;// pos+normal+color composite version
        std::weak_ptr<BufferGeometry> liveCheck;
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
