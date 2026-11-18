// GeometryState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns per-geometry GPU records (the BLAS cache) and the deformable meshes' states (skinned, tet, morphed,
// displaced, grass) with the shared compute pipelines that deform them, plus the displacement-map specs.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_GEOMETRYSTATE_HPP
#define THREEPP_VULKAN_STATE_GEOMETRYSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct GeometryState {
        // Per-mesh GPU state + the auto-LOD job types moved to
        // VulkanGeometryState.hpp (their doc comments travel with them).
        using BlasRecord = vulkan::impl::BlasRecord;
        using AttrVersions = vulkan::impl::AttrVersions;
        using GeomAttrRefs = vulkan::impl::GeomAttrRefs;
        using MaterialTexSet = vulkan::impl::MaterialTexSet;
        std::unordered_map<const BufferGeometry*, std::unique_ptr<BlasRecord>> blasCache;

        // Definition moved to VulkanGeometryState.hpp.
        using SkinnedMeshState = vulkan::impl::SkinnedMeshState;
        std::unordered_map<const SkinnedMesh*, std::unique_ptr<SkinnedMeshState>> skinnedMeshStates;

        // Definition moved to VulkanGeometryState.hpp.
        using TetMeshState = vulkan::impl::TetMeshState;
        std::unordered_map<const Mesh*, std::unique_ptr<TetMeshState>> tetMeshStates;

        // Definition moved to VulkanGeometryState.hpp.
        using MorphedMeshState = vulkan::impl::MorphedMeshState;
        std::unordered_map<const Mesh*, std::unique_ptr<MorphedMeshState>> morphedMeshStates;

        // Definition moved to VulkanGeometryState.hpp.
        using DisplacedMeshState = vulkan::impl::DisplacedMeshState;
        std::unordered_map<const DisplacedMesh*, std::unique_ptr<DisplacedMeshState>> displacedStates;

        // Renderer-level water_displace pipeline. One compute pipeline shared
        // across all DisplacedMesh instances; per-instance state owns its own
        // descriptor set so binding multiple oceans in one scene is safe.
        // Water displace pipeline — see vulkan/WaterDisplacePipeline.{hpp,cpp}.
        // Owns the shared compute pipeline + descriptor pool + sampler;
        // per-mesh descriptor sets (state->displaceDS) live in DisplacedMeshState.
        std::unique_ptr<vulkan::WaterDisplacePipeline> waterDisplace_;

        // The ocean's wake field — see vulkan/WakeFieldPipeline.{hpp,cpp}. One
        // pipeline; each DisplacedMesh with a wake field owns a State
        // (DisplacedMeshState::wake).
        std::unique_ptr<vulkan::WakeFieldPipeline> wakeField_;

        // Definition moved to VulkanGeometryState.hpp.
        using GrassMeshState = vulkan::impl::GrassMeshState;
        std::unordered_map<const GrassMesh*, std::unique_ptr<GrassMeshState>> grassStates;
        // Shared grass-wind compute pipeline (no descriptor sets — all I/O by
        // device address). See vulkan/GrassWindPipeline.{hpp,cpp}.
        std::unique_ptr<vulkan::GrassWindPipeline> grassWind_;

        // Geometries whose BLAS record must be built with allowPacked=false.
        // Populated by enableVertexInterop when it finds an already-built packed
        // record: a zero-copy producer writes tightly-packed float xyz normals,
        // but a record built with allowPacked=true carries snorm16x4 ones
        // (kPackedNormal). Rebuilding inline inside enable would tear down
        // buffers in-flight frames still read, so enable marks the geometry
        // here, requests a structural rebuild, and returns a null handle; the
        // rebuild (buildBlasFor consults this set) then produces an unpacked
        // record through the ordinary machinery and the caller's next poll
        // succeeds. Matches the documented enableVertexInterop contract of
        // "call it after the first render()".
        //
        // Entries live as long as the geometry does (pruned with the blasCache
        // record whose liveCheck expired). Not cleared once consumed: a later
        // geomVersion-mismatch rebuild of the same geometry must stay unpacked,
        // or interop would silently break under the producer.
        std::unordered_set<const BufferGeometry*> forceUnpackedGeoms_;
        // Material displacementMap (three.js semantics: each vertex moves along
        // its normal by map.r * scale + bias, the normal itself unchanged). The
        // positions are displaced ONCE, on the CPU from the map's source image
        // (the GPU only ever holds its BC7 encode), when the geometry's record
        // is built, so the raster, the BLAS and every ray share one surface. A
        // spec per geometry, registered by the scene walk (noteDisplacement)
        // and read by buildBlasFor like forceUnpackedGeoms_; the record keeps
        // the signature it was built with (BlasRecord::dispSig), and a
        // different one rebuilds it. One geometry drawn by two materials with
        // different maps gets the last one walked (warned once).
        struct DisplacementSpec {
            std::weak_ptr<BufferGeometry> geom;// liveness, as BlasRecord::liveCheck
            const Mesh* owner = nullptr;// the mesh that registered it (address + id)
            unsigned int ownerId = 0;
            std::shared_ptr<Texture> map;
            float scale = 1.f, bias = 0.f;
            std::array<float, 9> uvTransform{};// the map's Matrix3, column-major
            TextureWrapping wrapS = TextureWrapping::ClampToEdge, wrapT = TextureWrapping::ClampToEdge;
            bool nearest = false;// magFilter Nearest: the texel, else bilinear (level 0)
            // Refinement around the focus (see dispPixelEdge_), copied from the
            // renderer at registration so the build reads the spec alone. The
            // world matrix places the vertices relative to the focus; it is not
            // in the signature (a moving mesh keeps the refinement it got).
            bool refine = false;
            vulkan::ScreenEdgeTest edge;
            uint64_t signature = 0;// never 0 for a spec
            // The farthest any vertex moves: |h * scale + bias| over h in [0, 1].
            [[nodiscard]] float reach() const { return std::max(std::abs(bias), std::abs(scale + bias)); }
        };
        std::unordered_map<const BufferGeometry*, DisplacementSpec> geomDisplacement_;

        // Refinement of displacementMap meshes (EdgeRefine): an edge splits
        // while it is longer than dispPixelEdge_ pixels at its distance from
        // the focus (and than dispMinEdge_ metres) and the map's relief there
        // would show (its reach over half a pixel). 0 = off: only the vertices
        // a mesh has move, as on GL. The focus is the camera of the first
        // build, or of the build after refineDisplacement(); the epoch, in
        // every spec's signature, rebuilds the displaced records when the
        // focus or a knob changes. It runs where the record is built, on the
        // render thread.
        float dispPixelEdge_ = 0.f, dispMinEdge_ = 0.01f;
        bool dispFrustum_ = true;// refine coarsely outside the focus camera's view (ScreenEdgeTest)
        uint32_t dispRefineEpoch_ = 0;
        bool dispFocusPending_ = true;// take the focus from the next build's camera
        bool dispEpochPending_ = false;// re-check every displaced entry's spec at the next build
        vulkan::ScreenEdgeTest dispCamera_;// the focus camera's part of the edge test

        // Geometries whose producer declared, through setStableCorrespondence,
        // that vertex id i is NOT the same surface point frame to frame (a
        // host-uploaded marching-cubes soup). buildBlasFor stamps the record's
        // interopWorldStatic from this set, so the declaration outlives record
        // rebuilds; pruned with the record like forceUnpackedGeoms_.
        std::unordered_set<const BufferGeometry*> worldStaticGeoms_;
        // Lazily built on the first enableVertexInterop — most scenes never own
        // one, and it costs a pipeline + descriptor pool. See
        // vulkan/VertexSanitizePipeline.{hpp,cpp}.
        std::unique_ptr<vulkan::VertexSanitizePipeline> vertexSanitize_;

        // resetTemporalHistory: every dynamic BLAS takes the BUILD path on the
        // next frame and its refit counter restarts. A refit chain and the
        // periodic full rebuild it counts toward are temporal state like any
        // history: two runs that reached the reset after different numbers of
        // refits carried different structures (and rebuilt at different
        // post-reset frames), and the ray-traced rows diverged 20 to 70 frames
        // later (fjord, settle 1800 vs 1801, 2026-09-06; exact with --static).
        bool forceBlasRebuild_          = false;
        // The request above, latched for the frame being recorded: taken at
        // frame start (beside the pending accumulation reset), so a request
        // made between frames lands in the next frame's geometry work and not
        // in the tail of the previous frame's endFrame.
        bool blasRebuildThisFrame_      = false;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_GEOMETRYSTATE_HPP
