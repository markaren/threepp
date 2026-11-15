// SceneBuildFrame: the locals VulkanRenderer::Impl::ensureSceneBuilt's phases share
// (VulkanCoreScene.cpp). The driver owns one per build and passes it to every phase by
// reference; a phase opens with reference aliases to the members it reads or writes.
//
// Each member was a local of ensureSceneBuilt; they are declared in the order the
// locals were, each with the comment its local had. A local that was declared with
// a computed value is assigned at its old place instead (the members below start
// at their declared value, or unset where the local was).
#ifndef THREEPP_VULKAN_SCENEBUILDFRAME_HPP
#define THREEPP_VULKAN_SCENEBUILDFRAME_HPP

#include "VulkanCoreImpl.hpp"

namespace threepp {

    struct VulkanRenderer::Impl::SceneBuildFrame {
        // displacementMap refinement: the focus, taken once (and again after
        // refineDisplacement), and whether this build re-checks every
        // displaced entry's spec (the epoch moved, see dispPixelEdge_).
        bool dispEpochNow = false;

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
        bool snapLean = false;
        std::vector<MeshFingerprint> currFp;

        // Camera position for the cutout ray distance (setCutoutRayDistance):
        // read by the lean path's classifyCutoutFar and by the full
        // rebuild's instance loop.
        float rayCamPos[3];

        // Ray shadow LOD: the entries' states for this camera, before
        // either TLAS fill reads them (the matrices are this frame's).
        bool rayShadowChanged = false;

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
        std::vector<bool> entryGeomDirty;

        // Per-entry "material values bumped" bits — the ONLY entries whose
        // MaterialDesc must be re-derived in the material-values fast path
        // (materialFromMesh is a ~25-dynamic_cast + shared_ptr-churn walk).
        // Filled by both the LEAN in-place diff and the generic compare so
        // a single material toggle costs O(changed) instead of O(scene).
        std::vector<bool> entryMatDirty;

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

        // Per-entry bone-dirty bits. SkinnedMesh poses change without
        // touching the SkinnedMesh's worldMatrix, so the matrix-fingerprint
        // misses them. We compare current Skeleton::boneMatrices against
        // the cached prevBoneMats for each known SkinnedMesh; first-time
        // skinned meshes mark dirty so the structural-rebuild path skins
        // before its first ray trace. Gated on the cached isSkinned flag
        // so non-skinned entries skip the type probe entirely.
        std::vector<bool> entryBonesDirty;

        // DisplacedMesh — intrinsically dirty every frame (FFT spectrum
        // advances continuously). Same per-entry-bool pattern as bones,
        // routed through the cached isDisplaced flag instead of a fresh
        // dynamic_cast every frame.
        std::vector<bool> entryDisplacedDirty;

        // GrassMesh — intrinsically dirty every frame (wind advances) UNLESS
        // the field is frozen by the distance gate (params.maxAnimDistance).
        // A frozen (far) field is NOT dirty ⇒ it is not queued for a wind
        // dispatch/BLAS refit, not folded into grassDirtyAny, and not marked
        // moved — it holds its last displaced pose. It STILL stays in the TLAS
        // (culling never touches the TLAS), so off-screen/ranged grass keeps
        // casting shadows and appearing in reflections/GI. Grass keeps no
        // prevVertex buffer, so a skipped frame reports zero per-vertex motion
        // exactly like an animated frame — no TAA ghosting from the freeze.
        std::vector<bool> entryGrassDirty;

        // Morphed meshes — dirty when morphTargetInfluences changed.
        // Skinned meshes that also carry morph targets are handled by the
        // bone path above (GPU-skinned BLAS rebuild) so we skip them
        // here. Both predicates come from the cached fingerprint flags;
        // the getMorphAttributes hash lookup + SkinnedMesh dynamic_cast
        // used to run for every entry every frame.
        std::vector<bool> entryMorphDirty;

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

        // Material patch FIRST, then the TLAS fill: the fill derives each
        // instance's ray mask and FORCE_OPAQUE flag from the table, so
        // it must read THIS frame's materials. A patched entry whose
        // visibility class flipped (a blinker crossing the emitter threshold,
        // opacity dropping below 1) sets instClassChanged, which forces the
        // fill even on a frame with no motion, and a full build.
        bool instClassChanged = false;
    };

}// namespace threepp

#endif// THREEPP_VULKAN_SCENEBUILDFRAME_HPP
