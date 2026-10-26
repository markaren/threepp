// Private implementation header — the shared Impl for VulkanRenderer.cpp.
// Never include from public API headers.
// VMA_IMPLEMENTATION must be #defined in VulkanRenderer.cpp BEFORE including this file.
#pragma once

#include "EnvKnob.hpp"
#include "VulkanContext.hpp"
#include "VulkanResources.hpp"
#include "VulkanRetireQueue.hpp"
// Types split out of this header. Each one is aliased back into
// VulkanRenderer::Impl at the spot its definition used to occupy, so every
// reference site — Impl's own methods and the VulkanCore*.cpp TUs — is
// unchanged.
#include "VulkanImplCommon.hpp"
#include "VulkanGpuLayouts.hpp"
#include "VulkanGeometryState.hpp"
#include "VulkanSceneTypes.hpp"
#include "EdgeRefine.hpp"
#include "VulkanViewContext.hpp"
#include "EnvPrefilter.hpp"
#include "SkyBake.hpp"
#include "SceneEnvKnobs.hpp"
#include "EventCameraDetector.hpp"
#include "LidarScanner.hpp"
#include "SkinningPipeline.hpp"
#include "TetSkinningPipeline.hpp"
#include "GpuTimings.hpp"
#include "OverlayPass.hpp"
#include "VulkanFrameTypes.hpp"
#include "RenderGraph.hpp"
#include "TransientPool.hpp"
#include "TaaResolve.hpp"
#if defined(THREEPP_WITH_FSR)
#include "FsrUpscaler.hpp"
#endif
#if defined(THREEPP_WITH_DLSS)
#include "DlssUpscaler.hpp"
#endif
#include "AutoExposure.hpp"
#include "GbufResolve.hpp"
#include "BillboardGlowPass.hpp"
#include "BillboardSoftDepth.hpp"
#include "BloomPass.hpp"
#include "PostComposite.hpp"
#include "DofPass.hpp"
#include "SensorPass.hpp"
#include "SplatPass.hpp"
#include "DeferredShade.hpp"
#include "HiZPyramid.hpp"
#include "InstanceExpand.hpp"
#include "DrawGen.hpp"
#include "TlasGen.hpp"
#include "OcclusionCull.hpp"
#include "ParticleFieldPass.hpp"
#include "ProbeGI.hpp"
#include "WaterDisplacePipeline.hpp"
#include "FoamWorldPipeline.hpp"
#include "WakeFieldPipeline.hpp"
#include "GrassWindPipeline.hpp"
#include "VertexSanitizePipeline.hpp"
#include "shaders/downwash_shared.h"// DownwashGpu (uploadDownwash)
#include "shaders/vulkan_shared.h"// MaterialDesc + kMaxMaterialTextures — same source the shaders read

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/AttributeView.hpp"
#include "threepp/core/InterleavedBufferAttribute.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Frustum.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/scenes/Downwash.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

// (RT-pipeline / photon / denoise shader SPVs removed with the path tracer)
// (prefilter_env shader SPV moved into vulkan/EnvPrefilter.cpp)
// (skinning shader SPV moved into vulkan/SkinningPipeline.cpp)
// (water_displace shader SPV moved into vulkan/WaterDisplacePipeline.cpp)
// (gbuffer/gbuffer_indirect shader SPVs moved into vulkan/VulkanCorePipelines.cpp)
// (taa_resolve.comp.spv moved into vulkan/TaaResolve.cpp)
// (overlay/overlay_depth/overlay_color/overlay_point/overlay_sprite/particle/
//  sprite3d shader SPVs moved into vulkan/VulkanCorePipelines.cpp)
// (debug_resolve.comp.spv moved into vulkan/VulkanCoreIndirect.cpp)

#include "threepp/renderers/common/BC7Encode.hpp"
#include "threepp/renderers/common/BCnDecode.hpp"
#include "threepp/utils/GeometryLod.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <future>
#include <mutex>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp {

    using vulkan::VulkanContext;
    // Resource helpers moved to vulkan/VulkanResources.{hpp,cpp} to start
    // peeling apart the 11k-line VulkanRenderer.cpp monolith. Type names +
    // free functions imported into the local scope so the rest of this file
    // (which references them as unqualified Buffer / Image2D / createBuffer
    // / destroyBuffer / check / alignUp / etc.) keeps compiling unchanged.
    using vulkan::Buffer;
    using vulkan::Image2D;
    using vulkan::check;
    using vulkan::alignUp;
    using vulkan::createBuffer;
    using vulkan::destroyBuffer;
    using vulkan::createAsScratchBuffer;
    using vulkan::destroyImage2D;
    using vulkan::uploadHostVisible;
    using vulkan::flushHostWrites;
    using vulkan::invalidateHostReads;
    using vulkan::TimingPass;
    using vulkan::TP_RasterGbuf;
    using vulkan::TP_OverlayDepth;
    using vulkan::TP_DeferredShade;
    using vulkan::TP_Denoise;
    using vulkan::TP_TAA;
    using vulkan::TP_OverlayDraw;
    using vulkan::TP_GbufResolve;
    using vulkan::TP_ShadeB;
    using vulkan::TP_Dof;
    using vulkan::TP_Froxel;
    using vulkan::TP_SensorImage;
    using vulkan::TP_Splat;
    using vulkan::TP_ProbeGI;
    using vulkan::TP_ShadowMaps;
    using vulkan::TP_Rtao;

    namespace {
        // kFramesInFlight — and the long note on why it MUST stay EVEN —
        // moved to VulkanImplCommon.hpp (the extracted types size arrays with
        // it). Forwarded here so this header and the VulkanCore*.cpp TUs keep
        // referring to it unqualified.
        constexpr uint32_t kFramesInFlight = vulkan::impl::kFramesInFlight;
    }// namespace

    struct VulkanRenderer::Impl {
        Canvas& canvas;
        WindowSize size;
        // Canvas size render() last responded to. The resize predicate compares
        // against THIS, not `size`: `size` is pinned to the swapchain extent,
        // and the platform can grant a different extent than the canvas asked
        // for (Windows enforces a minimum window width on the hidden-window
        // headless fallback), so asked-for vs granted can disagree permanently.
        WindowSize lastCanvasSize;
        Color clearColor{0.f, 0.f, 0.f};
        float clearAlpha = 1.f;
        Vector4 viewport;
        Vector4 scissor;
        bool scissorTest = false;
        bool autoClear_ = true;// mirrored from Renderer::autoClear each render()

        // Split-screen (Increment 0): the primary deferred-render pane is
        // clipped to the scissor sub-rect. The deferred pipeline renders the
        // pane region-sized AT THE IMAGE ORIGIN (gbuf/shade/denoise/bloom);
        // only the final TAA write is offset to the scissor position. All
        // default to the full frame, so the single-scene path is
        // byte-identical when scissorTest is off.
        VkExtent2D regionRenderExt_{};// render-extent-space pane size
        VkExtent2D regionSwapExt_{};  // swapchain-space pane size (TAA output)
        int32_t    regionDstX_ = 0;   // swapchain write offset (image space)
        int32_t    regionDstY_ = 0;

        // Mirrored from VulkanRenderer (Renderer base) at the start of each
        // render() so the renderFrame path can read them without a pointer back
        // into the public class. Synced unconditionally — toggling these never
        // resets the accumulator (tone mapping is display-only).
        ToneMapping toneMapping_ = ToneMapping::None;
        float       toneMappingExposure_ = 1.f;

        // ── Physical camera exposure (setPhysicalCamera) ────────────────────
        // While enabled, exposure is derived from real camera parameters
        // instead of toneMappingExposure_:
        //   EV100 = log2(N²/t · 100/S) − evComp, exposure = 1/(1.2·2^EV100)
        // and the shade/resolve PRE-EXPOSES sceneHdr (multiplies the store by
        // the full exposure) so rgba16f survives 100k-lux daylight radiance.
        // Defaults are the sunny-16 daylight setting (f/16, 1/125 s, ISO 100):
        // a 100k-lux sun-lit scene lands at mid-gray. Legacy mode (default
        // off) leaves every multiply at 1.0 — numerically byte-identical.
        bool  physicalCamera_ = false;
        float camAperture_    = 16.f;       // f-number N
        float camShutter_     = 1.f / 125.f;// seconds
        float camIso_         = 100.f;
        float camEvComp_      = 0.f;// EV compensation (+1 doubles brightness)
        // Raw white-balance inputs (PostComposite only keeps the derived
        // Bradford matrix) so whiteBalance() can report them back.
        float wbTemperatureK_ = 6500.f;
        float wbTint_         = 0.f;
        [[nodiscard]] float physicalExposure() const {
            const float ev100 = std::log2(camAperture_ * camAperture_ / camShutter_ *
                                          100.f / camIso_) -
                                camEvComp_;
            return 1.f / (1.2f * std::exp2(ev100));
        }
        // The factor already baked into this frame's sceneHdr stores (1.0 in
        // legacy mode). Hoisted into preExpBits_ once per frame in
        // recordCommandBuffer; preExpHist_ remembers each frame-in-flight's
        // value so cross-frame sceneHdr consumers (the exposure meter) can
        // un-bake the right slot's exposure.
        [[nodiscard]] float preExposure() const {
            return physicalCamera_ ? currentExposure() : 1.f;
        }
        float    preExpHist_[kFramesInFlight] = {1.f, 1.f};
        uint32_t preExpBits_     = 0x3F800000u;// float bits of this frame's preExposure

        // Analytic light intensities are photometric while enabled: dir = lux
        // (as-is), point = lumens (→ candela Φ/4π at upload), spot = lumens
        // (→ Φ/π, Frostbite's cone-angle-invariant convention), rect/emissive
        // = nits (as-is). Pair with setPhysicalCamera — 100k lux needs a
        // physical exposure to land on screen.
        bool physicalLightUnits_ = false;

        // Wall-clock time of the last beginDeferredFrame() call, used to compute
        // dt for the onBeginDeferredFrame() hook (auto-exposure, etc.).
        double lastFrameTime_ = 0.0;

        std::unique_ptr<VulkanContext> ctx;
        // Device memory shared by frame-local scratch images (TransientPool):
        // declared right after ctx so it outlives every pass that binds images
        // into it and dies before the allocator. transientPool_ holds the
        // frame-local images — uv, sceneHdr, the filter / post / tail scratch,
        // the shadow maps (ShadowMapPass) —
        // each with the span of TransientPhases it is live in (images whose
        // spans do not meet alias each other, and so do frame-in-flight slots
        // and views); gbufMsPool_ holds the
        // multisampled G-buffer, whose only aliasing is across slots (it is in
        // use from the G-buffer pass to the shade, alongside shade scratch).
        std::unique_ptr<vulkan::TransientPool> transientPool_;
        std::unique_ptr<vulkan::TransientPool> gbufMsPool_;

        // Per-mesh GPU state + the auto-LOD job types moved to
        // VulkanGeometryState.hpp (their doc comments travel with them).
        using BlasRecord = vulkan::impl::BlasRecord;
        using AttrVersions = vulkan::impl::AttrVersions;
        using GeomAttrRefs = vulkan::impl::GeomAttrRefs;
        using MaterialTexSet = vulkan::impl::MaterialTexSet;
        std::unordered_map<const BufferGeometry*, std::unique_ptr<BlasRecord>> blasCache;

        // Definition moved to VulkanGeometryState.hpp.
        using LodGeomSel = vulkan::impl::LodGeomSel;
        static LodGeomSel selectLodGeom(const BlasRecord& rec, uint8_t lodLevel) {
            if (lodLevel > 0 && static_cast<size_t>(lodLevel - 1) < rec.lodLevels.size()) {
                const auto& lvl = rec.lodLevels[lodLevel - 1];
                // asAddress is 0 for a level without a structure (its record
                // has none either, so no TLAS instance reads it).
                if (lvl.index.handle != VK_NULL_HANDLE) {
                    return {lvl.address, lvl.index.address, lvl.indexCount, true};
                }
            }
            return {rec.address, rec.index.address, rec.indexCount,
                    rec.index.handle != VK_NULL_HANDLE};
        }

        // Definition moved to VulkanGeometryState.hpp.
        using SkinnedMeshState = vulkan::impl::SkinnedMeshState;
        std::unordered_map<const SkinnedMesh*, std::unique_ptr<SkinnedMeshState>> skinnedMeshStates;

        // List of SkinnedMeshState pointers whose bones changed this frame.
        // ensureSceneBuilt populates this (uploads bone matrices to the GPU
        // buffer); recordCommandBuffer consumes it by recording skinning
        // dispatch + BLAS rebuild into the main per-frame cmd buffer with
        // barriers. Cleared at the end of recordCommandBuffer.
        std::vector<SkinnedMeshState*> pendingSkinnedRebuilds_;

        // Definition moved to VulkanGeometryState.hpp.
        using TetMeshState = vulkan::impl::TetMeshState;
        std::unordered_map<const Mesh*, std::unique_ptr<TetMeshState>> tetMeshStates;
        std::vector<TetMeshState*> pendingTetRebuilds_;

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

        // World-space foam pipeline — see vulkan/FoamWorldPipeline.{hpp,cpp}.
        // Same one-pipeline-shared / per-mesh-descriptor-set pattern as
        // waterDisplace. Builds the per-DisplacedMesh foam texture each
        // frame; replaces the per-vertex foam buffer.
        std::unique_ptr<vulkan::FoamWorldPipeline> foamWorld_;

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
        // GrassMesh deforms queued in ensureSceneBuilt, recorded into the frame
        // command buffer in recordCommandBuffer (no blocking submit) — same
        // pattern as pendingSkinnedRebuilds_. Cleared at end of recordCommandBuffer.
        std::vector<std::pair<GrassMesh*, GrassMeshState*>> pendingGrassDeforms_;
        // Grass fields this frame's slot edit placed (new, or attached again):
        // their deform runs, but like every entry the structural rebuild
        // places they carry no moved bit on that frame (a moved bit is a
        // shorter history for thirty frames). Cleared with the deform queue.
        std::vector<const Mesh*> grassPlacedThisFrame_;

        // Graduated per-frame dynamic plain meshes (BlasRecord::perFrameDynamic)
        // whose attributes changed this frame — staging upload + GPU copy +
        // batched BLAS refit recorded into the frame cb by
        // recordDynamicGeomRefits, same pattern as pendingSkinnedRebuilds_.
        // Cleared there.
        std::vector<vulkan::impl::GeomRefreshOp> pendingDynamicGeomRefits_;
        // Graduated records whose first CLEAN frame follows a dirty run —
        // recordDynamicGeomRefits copies vertex→prevVertex once so their
        // motion vectors collapse back to zero (frame-cb twin of the
        // prevVertexResyncPending pass).
        std::vector<vulkan::impl::BlasRecord*> pendingDynamicPrevResyncs_;
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
        // The displacement m's material asks for (false: none, or not for a
        // mesh like this one; see VulkanCoreScene.cpp).
        bool displacementSpecOf(Mesh& m, bool morphed, DisplacementSpec& out) const;
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
        void captureDisplacementFocus(Camera& camera);
        // geom refined around d's focus (a new geometry), or nullptr when d
        // does not refine, the geometry can't be, or no edge splits.
        static std::shared_ptr<BufferGeometry> refineForDisplacement(const BufferGeometry& geom,
                                                                     const DisplacementSpec& d);
        // Registers m's displacement under its geometry; the spec, or nullptr
        // when its material displaces nothing. Render thread (the scene walks).
        const DisplacementSpec* noteDisplacement(Mesh& m);
        // After m's material changed: re-reads its displacement and returns
        // whether the geometry's spec moved (the record then needs a rebuild).
        bool displacementChanged(Mesh& m);
        [[nodiscard]] uint64_t displacementSignatureOf(const BufferGeometry* g) const {
            const auto it = geomDisplacement_.find(g);
            return (it != geomDisplacement_.end() && it->second.geom.lock().get() == g) ? it->second.signature : 0u;
        }
        // The object-space box geom's record draws: its own bounding box, or
        // the displaced positions' (BlasRecord::dispMin/Max), or, for a
        // displacementMap whose record is not built yet, its own grown by the
        // spec's reach. False when geom has no box at all.
        bool boundsOf(BufferGeometry& geom, Box3& out) const {
            // No displacementMap anywhere (the usual scene): no lookups.
            const uint64_t sig = geomDisplacement_.empty() ? 0u : displacementSignatureOf(&geom);
            if (const auto it = blasCache.find(&geom);
                sig != 0u && it != blasCache.end() && it->second->dispSig == sig) {
                const auto& r = *it->second;
                out.set(Vector3(r.dispMin[0], r.dispMin[1], r.dispMin[2]), Vector3(r.dispMax[0], r.dispMax[1], r.dispMax[2]));
                return true;
            }
            if (!geom.boundingBox) geom.computeBoundingBox();
            if (!geom.boundingBox) return false;
            out = *geom.boundingBox;
            if (sig != 0u) out.expandByScalar(geomDisplacement_.at(&geom).reach());
            return true;
        }
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

        // DisplacedMesh (FFT water) deforms queued in ensureSceneBuilt and
        // recorded into the frame command buffer by recordCommandBuffer — the
        // same no-mid-frame-submit pattern as pendingGrassDeforms_. The float
        // is the elapsed-seconds timestamp captured at stage time (drives the
        // dynamic spectrum). The old path recorded the FFT→displace→BLAS chain
        // into a one-shot and BLOCKED on its completion every frame — a full
        // CPU⇄GPU sync whose stall scaled with everything already in flight
        // (tens of ms on large scenes). CPU height-field mirrors now read the
        // readback buffers at stage time instead (one frame late).
        std::vector<std::tuple<DisplacedMesh*, DisplacedMeshState*, float>> pendingDisplacedDeforms_;

        // Single TLAS over all mesh instances in the scene.
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        Buffer tlasBuffer;
        // Per-in-flight-frame instance buffers. The per-frame TLAS refit is
        // recorded into the frame command buffer (not a one-shot drain), so the
        // host instance write for frame N must not clobber the buffer frame N-1
        // is still reading — hence one buffer per in-flight slot, indexed by
        // currentFrame. The structural full build (buildTlas) writes all slots.
        Buffer tlasInstancesBuffers[kFramesInFlight];
        // Persistent scratch for the in-frame TLAS refit (sized once; reused —
        // safe only because the frame graph's entry barrier orders each frame's
        // AS builds after the previous frame's work; submit order alone is not
        // an execution dependency).
        Buffer tlasRefitScratch_{};
        VkDeviceSize tlasRefitScratchSize_ = 0;
        // Instance count of the TLAS's last full BUILD. A MODE_UPDATE with any
        // other count is a spec violation that corrupts traversal, so
        // recordTlasRefit promotes such a refit to a full build (or skips it
        // when the existing storage can't hold the build). The structural
        // fingerprint now catches membership flips too — this is the backstop.
        uint32_t tlasBuiltInstanceCount_ = 0;
        // Per-frame TLAS refit, staged by ensureSceneBuilt and recorded into the
        // frame command buffer by recordCommandBuffer (after the deformable BLAS
        // rebuilds). Replaces the old mid-frame refitTlas one-shot drain.
        std::vector<VkAccelerationStructureInstanceKHR> pendingTlasInstances_;
        bool pendingTlasRefit_ = false;
        bool pendingTlasFullBuild_ = false;

        // Definition moved to VulkanGeometryState.hpp.
        using GeometryDesc = vulkan::impl::GeometryDesc;
        // The word view of it that the shaders reading only the material index
        // and the instance colour declare.
        static_assert(sizeof(GeometryDesc) == kGeomDescWords * 4u &&
                              offsetof(GeometryDesc, flags) == kGeomDescFlagsWord * 4u &&
                              offsetof(GeometryDesc, instanceColor) == kGeomDescColorWord * 4u,
                      "GeometryDesc changed shape: update kGeomDescWords / kGeomDescColorWord / kGeomDescFlagsWord "
                      "in vulkan_shared.h and geometry_desc.glsl.");
        // The material record an entry's GeometryDesc names (its flags' top bits).
        [[nodiscard]] static uint32_t geomMatIdx(const GeometryDesc& gd) {
            return gd.flags >> kGeomFlagMaterialShift;
        }
        static void setGeomMatIdx(GeometryDesc& gd, uint32_t idx) {
            gd.flags = (gd.flags & ~kGeomFlagMaterialMask) | (idx << kGeomFlagMaterialShift);
        }
        [[nodiscard]] static bool geomFlag(const GeometryDesc& gd, uint32_t bit) {
            return (gd.flags & bit) != 0u;
        }
        static void setGeomFlag(GeometryDesc& gd, uint32_t bit, bool on) {
            gd.flags = on ? (gd.flags | bit) : (gd.flags & ~bit);
        }
        // MaterialDesc layout lives in vulkan_shared.h (the same file the GLSL
        // deferred-shade / gbuffer / LIDAR shaders pull in via #include).
        // Bringing it into Impl scope here keeps the existing
        // `MaterialDesc md{};` call sites unchanged.
        using MaterialDesc = threepp::vulkan_pt::MaterialDesc;

        // TLAS instance flags for an entry, from the MaterialDesc its shaders
        // read (mats[instanceCustomIndex]). Every BLAS is built non-opaque so
        // alpha-cutout casters yield candidates, which made EVERY solid
        // triangle a shadow, emitter-NEE, ReSTIR or probe-shadow ray met a
        // round trip to the shader to read its material and confirm it. Those
        // five queries are all terminate-on-first-hit visibility tests, and for
        // a material that none of them can reject each one confirms
        // unconditionally — so FORCE_OPAQUE hands the RT core the same answer
        // without the round trip. "None can reject" is exactly: not an
        // emitter (the emitter loops skip emissive >= 0.05; shadowOccludes
        // skips >= 1), not transmissive, not a blend (alphaCutoff < 0), and no
        // live cutout (alphaCutoff > 0 with an albedo texture). KEEP IN SYNC
        // with shadowOccludes (deferred_shade_10_lighting_utils.glsl,
        // probe_update.comp) and the emitter-skip tests. Rays that already
        // pass gl_RayFlagsOpaqueEXT are unaffected. THREEPP_NO_FORCE_OPAQUE=1
        // turns it off (A/B and bisecting).
        [[nodiscard]] static VkGeometryInstanceFlagsKHR tlasInstanceFlags(const MaterialDesc& m) {
            static const bool disabled = vulkan::env::flag("THREEPP_NO_FORCE_OPAQUE");
            VkGeometryInstanceFlagsKHR f = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            if (disabled) return f;
            const float emMax = std::max({m.emissive[0] * m.emissiveIntensity,
                                          m.emissive[1] * m.emissiveIntensity,
                                          m.emissive[2] * m.emissiveIntensity});
            const bool alwaysOccludes = emMax < 0.05f && !(m.transmission > 0.0f) &&
                                        !(m.alphaCutoff < 0.0f) &&
                                        (m.albedoTexIndex < 0 || m.alphaCutoff == 0.0f);
            if (alwaysOccludes) f |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
            return f;
        }
        // A material whose ONLY reason not to be force-opaque is a live cutout
        // (alphaCutoff > 0 with an albedo texture): foliage cards, grids. The
        // class setCutoutRayDistance may force opaque beyond its distance.
        [[nodiscard]] static bool cutoutCaster(const MaterialDesc& m) {
            const float emMax = std::max({m.emissive[0] * m.emissiveIntensity,
                                          m.emissive[1] * m.emissiveIntensity,
                                          m.emissive[2] * m.emissiveIntensity});
            return emMax < 0.05f && !(m.transmission > 0.0f) &&
                   m.alphaCutoff > 0.0f && m.albedoTexIndex >= 0;
        }
        // Is the whole of an entry's bounding sphere farther from `cam` than
        // `dist`? Unknown bounds (lodRadius 0) are never far.
        [[nodiscard]] static bool entryBeyond(const vulkan::impl::MeshEntry& en, const float cam[3], float dist) {
            if (en.lodRadius <= 0.f) return false;
            const float* M = en.worldMatrix.data();
            const float cx = M[0] * en.lodCenter[0] + M[4] * en.lodCenter[1] + M[8] * en.lodCenter[2] + M[12];
            const float cy = M[1] * en.lodCenter[0] + M[5] * en.lodCenter[1] + M[9] * en.lodCenter[2] + M[13];
            const float cz = M[2] * en.lodCenter[0] + M[6] * en.lodCenter[1] + M[10] * en.lodCenter[2] + M[14];
            const float s0 = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
            const float s1 = M[4] * M[4] + M[5] * M[5] + M[6] * M[6];
            const float s2 = M[8] * M[8] + M[9] * M[9] + M[10] * M[10];
            const float reach = dist + en.lodRadius * std::sqrt(std::max(s0, std::max(s1, s2)));
            const float dx = cam[0] - cx, dy = cam[1] - cy, dz = cam[2] - cz;
            return dx * dx + dy * dy + dz * dz > reach * reach;
        }
        // The ray-mask half of the same per-material classification (see the
        // visibility groups in vulkan_shared.h). Water (displaced) stays in the
        // opaque group whatever its material says.
        [[nodiscard]] static bool alphaMaskGroup(const MaterialDesc& m) {
            return m.transmission > 0.0f || m.alphaCutoff < 0.0f;
        }

        // ── Per-slot dirty state for the entries-indexed desc rings ──────────
        // What one frame-in-flight slot still owes the GPU: either the whole
        // array, or a set of half-open element ranges. Ranges matter because
        // the GeometryDesc array is entries-indexed and large (until the
        // material table, the MaterialDescs were too: 47.7 MB at 78.4k
        // entries), so a whole-array flush for a small per-frame change costs
        // milliseconds of CPU. The material table's ranges are record ranges.
        //
        // Per slot, not global: a change dirtied on frame N must reach every
        // slot's buffer, but each slot may only be written after its own fence
        // has signalled, so each slot accumulates independently and is cleared
        // alone when its turn comes.
        struct DescDirtyRanges {
            // Half-open [first, last) entry ranges, kept sorted and coalesced.
            std::vector<std::pair<uint32_t, uint32_t>> ranges;
            bool whole = false;// structural rebuild / unidentifiable source

            bool any() const { return whole || !ranges.empty(); }
            void clear() {
                whole = false;
                ranges.clear();
            }
            void markWhole() {
                whole = true;
                ranges.clear();// subsumed
            }
            // The scene walk marks in increasing entry order, so coalescing
            // against the last range collapses a contiguous instanced span into
            // one range. Marks also arrive out of order: a slot not flushed last
            // frame still holds that frame's ranges when the next walk starts
            // again from entry 0. A mark that overlaps or touches the last range
            // widens it at either end; any other is a range of its own (the
            // upload copies each range, so an overlap with an older one is
            // harmless). Until 2026-10-15 a mark below the last range only
            // raised its end and was lost: an edit to an earlier entry the frame
            // after an edit to a later one never showed (VulkanInstanceColor_test,
            // phase 3).
            void mark(uint32_t first, uint32_t count = 1) {
                if (whole || count == 0) return;
                const uint32_t last = first + count;
                if (!ranges.empty() && first <= ranges.back().second && last >= ranges.back().first) {
                    ranges.back().first  = std::min(ranges.back().first, first);
                    ranges.back().second = std::max(ranges.back().second, last);
                    return;
                }
                // Many ranges suggest a scene-wide change; fall back to one
                // whole-array flush (broader than necessary, never wrong).
                if (ranges.size() >= 64) {
                    markWhole();
                    return;
                }
                ranges.emplace_back(first, last);
            }
        };

        // Per-frame-in-flight GeometryDesc storage — same fence-gated ring as
        // materialDescsBuffers below. Ringed for auto-LOD: a level switch
        // repoints GeometryDesc::indexAddress/indexed, and a single shared
        // buffer would need a vkDeviceWaitIdle on every switch frame to patch
        // it in place. Hot path stages in geomDescsCached_ +
        // marks geomDescsDirty_ (markGeomDescsDirty / ...Whole); renderFrame's
        // flushGeometryDescsIfDirty memcpys the marked ranges into this frame's
        // slot once its fence has signaled.
        std::array<Buffer, kFramesInFlight> geometryDescsBuffers{};
        std::array<DescDirtyRanges, kFramesInFlight> geomDescsDirty_{};
        // Per-frame-in-flight material table storage; one buffer per slot so
        // the upload after a fence wait races nothing. The hot path stages new
        // records in `matTable_` and marks the RECORD ranges it touched
        // (markMatDescsDirty); renderFrame's flushMaterialDescsIfDirty memcpys
        // those ranges into `materialDescsBuffers[currentFrame]` once the
        // fence has signaled (= GPU done with this slot). Descriptor sets are
        // bound per-frame (set idx = f*imageCount_+k → buffer[f]) so the
        // binding stays valid across the swap.
        std::array<Buffer, kFramesInFlight> materialDescsBuffers{};
        std::array<DescDirtyRanges, kFramesInFlight> matDescsDirty_{};

        // ── The material table ──────────────────────────────────────────────
        // One MaterialDesc per Material the scene draws, not one per entry: an
        // entry names its record in the top bits of its GeometryDesc's flags
        // (geomMatIdx), and every shader reads it through entryMat, which
        // multiplies in the instance colour the entry's GeometryDesc carries
        // (setGeomInstanceColor). The desc is a pure function of the Material
        // (buildMaterialDesc), so the Material is the key. Per entry, a
        // streamed cell of a thousand instances of one material cost a
        // thousand 600-byte copies to pack and upload; here it costs a
        // thousand indices and at most one record.
        //
        // Record 0 is the all-zero desc, what an entry without a material (an
        // overlay, a tombstone, a span still being admitted) has always read.
        // Every other record counts the entries that name it: every write of
        // an entry's index goes through setEntryMat or assignEntryGeom. A
        // record whose count reaches zero goes dormant, key and content kept:
        // a streamed world brings the same materials back with the next
        // cell, which then finds them written and uploads nothing.
        // Only a full table reclaims a dormant record, the longest-dormant
        // first, and only when none is left do the GPU buffers grow. The
        // structural rebuild starts the table afresh in entry order.
        struct MatTableKey {
            const Material* mat = nullptr;
            unsigned int matId = 0;// Material::id: a new material at a dead one's address is a new key
            bool operator==(const MatTableKey& o) const {
                return mat == o.mat && matId == o.matId;
            }
        };
        struct MatTableKeyHash {
            size_t operator()(const MatTableKey& k) const {
                return std::hash<const void*>{}(k.mat) ^ (size_t(k.matId) * 0x9E3779B97F4A7C15ull);
            }
        };
        std::vector<MaterialDesc> matTable_;
        std::vector<MatTableKey> matTableKeys_;// per record; meaningless where the count is 0
        std::vector<uint32_t> matTableRefs_;   // entries naming each record (record 0 uncounted)
        // The pass a record was last derived in (matTableSlot): a record is
        // derived once per pass however many entries share it, and is not
        // reclaimed in that pass (its entries may not have been written yet).
        std::vector<uint32_t> matTableStamp_;
        // Records gone dormant, oldest first, each with the count of times it
        // has: an entry whose record was revived and has gone dormant again
        // since is stale and skipped.
        std::deque<std::pair<uint32_t, uint32_t>> matTableDormant_;
        std::vector<uint32_t> matTableDormantSeq_;
        std::unordered_map<MatTableKey, uint32_t, MatTableKeyHash> matTableIndex_;
        uint32_t matTableCapacity_ = 0;// records the GPU buffers hold
        uint32_t matTableEpoch_ = 1;
        // The structural rebuild fills the table before it sizes the buffers:
        // no growth while it does.
        bool matTableBuilding_ = false;
        // The last key resolved this pass: an instanced span's entries share
        // their material and skip the hash lookup.
        MatTableKey matTableMemoKey_{};
        uint32_t matTableMemoSlot_ = 0;// 0 = no memo

        // Start a pass over the table: records are re-derived on their first
        // use in it.
        void matTableBeginPass() {
            ++matTableEpoch_;
            matTableMemoSlot_ = 0u;
        }
        // Empty but for record 0, for a structural rebuild to fill.
        void matTableReset();
        // The record for `mat`, created when new; `md` is its desc. Written and
        // dirtied only when its content changes. Counts no reference: the
        // caller hands the index to setEntryMat / assignEntryGeom.
        uint32_t matTableSlot(const Material* mat, const MaterialDesc& md);
        // A record for `key`: a new one while the GPU buffers have room, else
        // the longest-dormant one, else a new one after a growth.
        uint32_t matTableAcquire(const MatTableKey& key);
        void matTableGrow(uint32_t capacity);
        void matTableRef(uint32_t idx) {
            if (idx != 0u) ++matTableRefs_[idx];
        }
        void matTableUnref(uint32_t idx);
        // The record entry `i` draws with.
        [[nodiscard]] const MaterialDesc& entryMat(size_t i) const {
            return matTable_[geomMatIdx(geomDescsCached_[i])];
        }
        // Point entry `i` at record `idx`, dirtying its GeometryDesc on a change.
        void setEntryMat(size_t i, uint32_t idx) {
            const uint32_t was = geomMatIdx(geomDescsCached_[i]);
            if (was == idx) return;
            matTableRef(idx);
            setGeomMatIdx(geomDescsCached_[i], idx);
            matTableUnref(was);
            markGeomDescsDirty(static_cast<uint32_t>(i));
        }
        // Replace entry `i`'s whole GeometryDesc with `gd` naming record
        // `idx` (the caller marks the range dirty).
        void assignEntryGeom(size_t i, const GeometryDesc& gd, uint32_t idx) {
            const uint32_t was = geomMatIdx(geomDescsCached_[i]);
            matTableRef(idx);
            geomDescsCached_[i] = gd;
            setGeomMatIdx(geomDescsCached_[i], idx);
            matTableUnref(was);
        }

        // Stage a desc change for every slot at once (a host-side patch applies
        // to the scene, not to one frame). Named per array so the call sites
        // read as what they patched.
        void markMatDescsDirty(uint32_t first, uint32_t count = 1) {
            for (auto& d : matDescsDirty_) d.mark(first, count);
        }
        void markMatDescsWhole() {
            for (auto& d : matDescsDirty_) d.markWhole();
        }
        void markGeomDescsDirty(uint32_t first, uint32_t count = 1) {
            for (auto& d : geomDescsDirty_) d.mark(first, count);
        }
        void markGeomDescsWhole() {
            for (auto& d : geomDescsDirty_) d.markWhole();
        }

        // Per-FIF deferred-descriptor refresh flag. Set (all slots) when a
        // material texture is swapped in place (refreshDirtyMaterialTextures) or
        // an env texture is swapped: the new image view must be rebound in the
        // deferred/probe descriptor sets, but those sets can't be updated while
        // in flight (no descriptorBindingUpdateAfterBind on this device). Same
        // idiom as geomDescsDirty_/matDescsDirty_: the frame-begin path rewrites
        // ONLY the current slot's set once its fence has signaled, so no device
        // stall. In-flight frames sample the retired old view for ≤1 more frame
        // (it stays alive in the retire queue) — correct behavior.
        std::array<bool, kFramesInFlight> deferredDescDirty_{};

        // Scene lights mirrored to a per-frame UBO. Scalar block layout means
        // the C++ structs map directly (no std140 vec3→vec4 padding).
        // Bounds moved to VulkanImplCommon.hpp (GpuLightsUbo sizes its arrays
        // with them); forwarded so every use site stays unqualified.
        static constexpr auto kMaxDirLights   = vulkan::impl::kMaxDirLights;
        static constexpr auto kMaxPointLights = vulkan::impl::kMaxPointLights;
        static constexpr auto kMaxSpotLights  = vulkan::impl::kMaxSpotLights;
        static constexpr auto kMaxRectLights  = vulkan::impl::kMaxRectLights;

        // GPU-layout mirrors moved to VulkanGpuLayouts.hpp (their layout
        // comments and sizeof static_asserts travel with them).
        using GpuDirLight   = vulkan::impl::GpuDirLight;
        using GpuPointLight = vulkan::impl::GpuPointLight;
        using GpuSpotLight  = vulkan::impl::GpuSpotLight;
        using GpuRectLight  = vulkan::impl::GpuRectLight;
        using GpuLightsUbo  = vulkan::impl::GpuLightsUbo;
        std::array<Buffer, kFramesInFlight> lightsUbos{};

        // ── Clustered lights (deferred) ─────────────────────────────────────
        // All scene point/spot lights, power-sorted, in one unified record;
        // the UBO's 8-per-type arrays above keep the strongest 8 (same sort)
        // for the paths screen-space clusters can't serve (secondary ray
        // hits — reflection/GI, volumetric beams, probes). cluster_build.comp
        // culls the list into per-cell index rows of a 16×8×24 screen-tile ×
        // exponential-Z grid; deferred_shade's analytic split loops only its
        // own cell.
        // KEEP IN SYNC with the ClusterLight structs in cluster_build.comp +
        // deferred_shade.comp (scalar layout, 64 bytes).
        static constexpr uint32_t kMaxClusterLights   = 256;
        static constexpr uint32_t kClusterCells       = 16 * 8 * 24;
        static constexpr uint32_t kClusterMaxPerCell  = 24;
        // Layout moved to VulkanGpuLayouts.hpp.
        using GpuClusterLight = vulkan::impl::GpuClusterLight;
        using GpuShadowMapsUbo = vulkan::impl::GpuShadowMapsUbo;
        std::array<Buffer, kFramesInFlight> clusterLightsBuffers{};// host-visible mapped (CPU fills per frame)
        std::array<Buffer, kFramesInFlight> clusterGridBuffers{};  // device-local (cluster_build writes)
        uint32_t clusterLightCountThisFrame_ = 0;
        bool     fogEnabledThisFrame_ = false;// scene.fog present (froxel-volumetrics gate)

        // Layout (and its doc comment) moved to VulkanGpuLayouts.hpp.
        using GpuFogUbo = vulkan::impl::GpuFogUbo;
        std::array<Buffer, kFramesInFlight> fogUbos{};
        float    fogAnisotropy_ = 0.0f;
        float    fogWaterSurfaceY_ = 1e30f;
        float    murkDensity_ = 0.0f;               // setUnderwaterMurk (0 = off); the clearest channel of murkSigma_
        float    murkSigma_[3] = {0.0f, 0.0f, 0.0f};// setUnderwaterMurk extinction per channel (all = murkDensity_ in the scalar form)
        float    murkColor_[3] = {1.0f, 1.0f, 1.0f};// setUnderwaterMurk tint
        float    murkAmbientFalloff_ = 0.0f;        // setUnderwaterAmbientFalloff (0 = off)
        bool     murkSpriteForeshortening_ = true;  // setUnderwaterSpriteForeshortening
        // ── Resolved unified fog medium for THIS frame (computed by updateFogUbo,
        // consumed by updateCloudUbo + the froxel gate) ──────────────────────────
        // Phase 2 "one knob": scene.fog is the primary control — when present it
        // supplies the medium density (FogExp2.density / linear-Fog span) + colour
        // and the froxel volumetrics run in heterogeneous mode with a near-uniform
        // default profile (baseY 0, huge falloff). setHeightFog is the advanced
        // profile control: it sets baseY/falloff/noise; its density is used only
        // when scene.fog is absent (back-compat for existing mist users). When
        // both are set, scene.fog's density wins (heightFog.density ignored).
        bool  mediumActiveThisFrame_ = false;// air medium present → run froxels hetero
        float mediumDensityThisFrame_ = 0.0f;
        float mediumBaseYThisFrame_   = 0.0f;
        float mediumFalloffThisFrame_ = 1.0e6f;
        float mediumNoiseThisFrame_   = 0.0f;
        // Air-fog single-scattering tint (= scene.fog colour when present, else
        // white — mirrors the shade pass's medAlbedo). The overlay-fog snapshot
        // fades particles toward this (× no env, the accepted particle-domain
        // approximation). Consumed by the overlay particle draw in the record.
        float mediumTintThisFrame_[3] = {1.0f, 1.0f, 1.0f};
        // Near-uniform default profile falloff (m) when scene.fog drives the
        // medium without an explicit setHeightFog profile. exp(-y/H) ≈ 1 over any
        // real scene extent, and heightFogOpticalDepth clamps y<baseY to full
        // density → a homogeneous medium (the degenerate case of the height-fog
        // integral). The sky path detects this magnitude to avoid a divergent
        // infinite-ray optical depth (see applySkyFog).
        static constexpr float kUniformFogFalloff = 1.0e6f;

        // Layout (and its doc comment) moved to VulkanGpuLayouts.hpp.
        using GpuCloudUbo = vulkan::impl::GpuCloudUbo;
        std::array<Buffer, kFramesInFlight> cloudUbos{};
        bool  cloudsEnabled_   = false;
        // Bumped by setClouds on enable / material reconfigure (NOT on identical
        // per-frame re-sets — the fjord demo calls setClouds every frame).
        // Wrapped &1023: the aux .a channel is fp16, int-exact only to 2048.
        int   cloudEpoch_      = 1;
        float cloudCoverage_   = 0.45f;
        float cloudDensity_    = 1.0f;
        float cloudBottomY_    = 600.0f;
        float cloudTopY_       = 1400.0f;
        float cloudEvolveSpeed_= 1.0f;
        float cloudWind_[3]    = {8.0f, 0.0f, 2.0f};
        // Near-field heterogeneous height fog (VulkanRenderer::setHeightFog).
        bool  heightFogEnabled_    = false;
        float heightFogDensity_    = 0.02f;
        float heightFogBaseY_      = 0.0f;
        float heightFogFalloff_    = 80.0f;
        float heightFogNoiseAmount_= 0.6f;

        // Environment equirect (HDR float) used by the primary miss
        // for backgrounds and by closest-hit for a single mirror-reflection
        // IBL probe. Default is a 1×1 black dummy so descriptors are always
        // valid; replaced lazily when the scene's environment / background
        // texture is set or changed.
        Image2D envImage{};
        unsigned int envTextureIdUploaded = 0xFFFFFFFFu;
        bool envIsDefault  = true;
        bool envIsBgColor  = false;
        Color envBgColor{0.f, 0.f, 0.f};
        // threepp::Sky (vulkan/SkyBake): the visible Sky the last full scene
        // expansion met, if any (its box builds no entry), and whether envImage
        // is a NON-OWNING copy of the bake's persistent image — dropEnvImage
        // knows. refreshEnvTextureFromScene reads the Sky's uniforms every
        // frame; a change marks a re-bake, recorded in place at the top of the
        // frame's command buffer, with none of what a new env costs.
        Sky* skySeen_ = nullptr;
        bool envIsSky = false;
        PreethamSkyParams skyBaked_{};
        bool skyBakedValid_  = false;
        bool skyBakePending_ = false;
        std::unique_ptr<vulkan::SkyBake> skyBake_;
        // Let go of envImage: retired when this Impl owns it, forgotten when it
        // is the sky bake's, which keeps its image for the renderer's life.
        void dropEnvImage() {
            if (envIsSky) {
                envImage = {};
                envIsSky = false;
            } else {
                retire(std::move(envImage));
            }
        }
        // HDRI sun extracted at PMREM build (deferred only — see
        // envSunExtractionWanted). found=false when no sun / extraction off.
        // updateLightsUbo re-injects it as an analytic directional light.
        vulkan::EnvPrefilter::SunExtract envSun_{};
        // The scene's environment/background knobs (SceneEnvKnobs.hpp), as
        // updateLightsUbo last packed them. envSun_ above stays in ENV space;
        // the lights UBO and envSunDirection()/envSunColor() turn and scale it
        // by these. envKnobsSeen_ is the frame loop's own copy for change
        // detection (a change wipes temporal history like a new env does);
        // invalid until the first frame, so the first read is not a change.
        vulkan::SceneEnvKnobs envKnobs_{};
        vulkan::SceneEnvKnobs envKnobsSeen_{};
        bool envKnobsSeenValid_ = false;

        // Ocean fine-cascade normal-map source (deferred shade ocean binding).
        // Default 1×1 dummy R32F, replaced with the active DisplacedMesh's
        // cascade-2 height image when one is in the scene. deferred_shade.comp
        // samples this on `thinWalled` materials at world-space XZ via
        // finite differences to perturb the macro normal — adds sub-mesh
        // chop detail (FFT cells finer than the 1 m mesh resolves).
        Image2D oceanFineHeightDummy{};
        VkImageView oceanFineHeightView   = VK_NULL_HANDLE;// either dummy or cascade-2 view
        VkSampler   oceanFineHeightSampler = VK_NULL_HANDLE;
        float       oceanFineTileSize     = 0.f;          // 0 disables sampling in shader

        // World-space foam (deferred shade ocean binding). Built by
        // FoamWorldPipeline each frame from a DisplacedMesh's foamImage.
        // deferred_shade.comp samples it at world XZ on water hits — replaces
        // the per-vertex foam buffer that used to live on the BLAS. Held as a
        // dummy 1×1 R32F when no DisplacedMesh is in the scene.
        Image2D oceanFoamDummy{};
        VkImageView oceanFoamView   = VK_NULL_HANDLE;
        VkSampler   oceanFoamSampler = VK_NULL_HANDLE;
        float       oceanFoamTileSize = 0.f;              // 0 disables sampling
        // The wake field's state images (deferred shade binding 78): the live
        // ocean's, or WakeFieldPipeline's 1x1 dummy. The shade samples it only
        // through the patch table the ocean's GeometryDesc::foamAddress names.
        VkImageView oceanWakeView = VK_NULL_HANDLE;
        // Its ripples' slopes (binding 79), or the same dummy.
        VkImageView oceanRippleView = VK_NULL_HANDLE;

        // Tileable foam detail texture (deferred shade foam binding).
        // R = micro bubble brightness (three value-noise octaves matching the
        // old procedural micro look over a 4 m world mapping), G = ridged
        // lace/filament pattern (12 m mapping). Baked once at startup with a
        // full mip chain — replaces the per-pixel procedural noise octaves in
        // the foam shading: a sampled texture is band-limited at distance (no
        // shimmer under TAA) and 2 fetches replace ~20 hash evaluations.
        Image2D foamDetailImage{};

        // Env luminance CDF (Phase A: env importance sampling).
        // Conditional CDF: w×h R32F texture; row r holds the cumulative
        // distribution over columns at that latitude.
        // Marginal CDF: h×1 R32F texture; row r holds the cumulative marginal
        // 64×64 R8 blue-noise tile for sub-pixel jitter. Generated once via
        // void-and-cluster (Ulichney 1993) and uploaded at startup. Adjacent
        // pixels share correlated values (silhouette stability) but globally
        // decorrelated (no coherent shake). Animated temporally by offsetting
        // the lookup coords per frame; AA convergence via accumulator.
        Image2D blueNoiseImage{};

        // PMREM: GGX-prefiltered env mip chain. Built once per env
        // upload — see vulkan/EnvPrefilter.{hpp,cpp}. Owns the prefilter
        // compute pipeline + descriptor pool + source sampler; the host calls
        // envPrefilter_->buildPmrem(...) when scene.environment changes.
        std::unique_ptr<vulkan::EnvPrefilter> envPrefilter_;

        // Bindless material textures (albedo only for v1). The descriptor set
        // exposes a fixed-size sampler2D[] at binding 8; closest_hit indexes
        // into it via mdesc.albedoTexIndex. Slot 0 is a 1×1 white default so
        // materials without an albedo map can still bind a valid descriptor.
        // Cache key is Texture* — the same Texture across multiple meshes only
        // uploads once. All material textures share the policy samplers (see
        // textureSampler_ below); the only per-texture sampler state honoured
        // is ClampToEdge wrap (materialTexClampUV_) — per-texture FILTER and
        // mirrored wrap remain a v2 concern.
        // kMaxMaterialTextures comes from the shared header (vulkan_shared.h,
        // included near the top of this file). Editing it there propagates to
        // every shader on the next clean rebuild.
        std::vector<Image2D> materialTextures;// owns image + view (sampler is shared)
        // Slot-parallel to materialTextures: 1 = this texture asked for
        // ClampToEdge on BOTH axes, so its binding uses the clamp variant of
        // the policy sampler. REPEAT is wrong for edge-inclusive atlases —
        // a terrain tile's splat bilinear/mip taps at uv 0/1 wrap around and
        // blend in the OPPOSITE edge, painting a seam line along every tile
        // border (metres wide at coarse mips). Maintained by
        // ensureMaterialTexture; only read for slots with a live view.
        std::vector<uint8_t> materialTexClampUV_;
        // Entry-list / cache records moved to VulkanSceneTypes.hpp (their doc
        // comments travel with them).
        using CachedTexture = vulkan::impl::CachedTexture;
        std::unordered_map<const Texture*, CachedTexture> textureCache;
        std::vector<uint32_t> freeTextureSlots;// slots reclaimed by prune
        // Slots whose cache entry was found STALE at lookup (the original
        // Texture died and a brand-new one was allocated at the recycled
        // address — the raw-pointer key collides). The image cannot be
        // destroyed at detection time: in-flight frames + bound descriptor
        // sets may still reference it, and ensureMaterialTexture's callers
        // include the lean material-patch path that deliberately avoids a
        // device drain. The structural-rebuild prune (post-drain) destroys
        // these and returns the slots to freeTextureSlots.
        std::vector<uint32_t> retiredTextureSlots_;
        // The policy samplers every material-texture binding chooses between
        // (fillMaterialTextureInfos → materialSampler()): 16× aniso is the
        // default regardless of raster jitter (dropping to isotropic under
        // TAA/DLSS/FSR does not fix distance shimmer, it only mip-blurs
        // grazing angles). The isotropic twins exist for
        // the explicit setTextureAnisotropy(1) / THREEPP_VK_ANISO=1 override.
        // Each policy exists in a REPEAT and a CLAMP_TO_EDGE flavour —
        // clamp-tagged textures (materialTexClampUV_) get the clamp twin.
        // NOTE: the per-image samplers buildSampledImage2D creates are NOT
        // bound for material textures — these are.
        VkSampler textureSampler_ = VK_NULL_HANDLE;        // 16× aniso (default)
        VkSampler textureSamplerIso_ = VK_NULL_HANDLE;     // isotropic (override=1 only)
        VkSampler textureSamplerClamp_ = VK_NULL_HANDLE;   // 16× aniso, clamp-to-edge
        VkSampler textureSamplerIsoClamp_ = VK_NULL_HANDLE;// isotropic, clamp-to-edge
        // setTextureAnisotropy override: 0 = AUTO (the policy above), 1..16
        // forces that level. Values other than 1/16 use a lazily-created
        // custom sampler; a replaced custom is PARKED (in-flight frames and
        // bound descriptor sets may still reference it; samplers are tiny)
        // and destroyed at teardown.
        float textureAnisoOverride_ = 0.f;
        VkSampler textureSamplerCustom_ = VK_NULL_HANDLE;     // repeat flavour
        VkSampler textureSamplerCustomClamp_ = VK_NULL_HANDLE;// clamp twin, same level
        float textureSamplerCustomAniso_ = 0.f;
        std::vector<VkSampler> parkedSamplers_;

        // ReSTIR DI reservoir ping-pong storage, consumed by the deferred
        // shade's next-event estimation (DeferredShade reservoirPos/W). Two
        // physical images per logical buffer: reservoirPosImagesPP carries
        // lightPos.xyz + lightType.w (rgba32f), reservoirWImagesPP carries
        // W_sum + M + W + p_hat (rgba16f). At frame N the shade writes slot
        // (N & 1) and reads slot ((N+1) & 1).
        // (reservoirPosImagesPP / reservoirWImagesPP moved to ViewContext —
        //  the reservoirs are SCREEN-space, so they belong to a view.)
        // Frame counter driving Halton jitter + blue-noise offset for the
        // deferred shade's stochastic GI / soft-shadow sampling. Genuinely
        // shared: it is the sequence CLOCK, not per-view history. Every view
        // in a frame samples the same Halton/blue-noise phase.
        uint32_t sampleIndex = 0;
        // (prevCamBufData_ / prevCameraValid moved to ViewContext.)

        // Per-entry "moved" bitmask, one bit per TLAS instance. Bit i set when
        // entry i's effective worldMatrix / pose / displacement changed since
        // last frame. Sized to ceil(meshCount/32) words at scene rebuild;
        // uploaded to meshMovedBitsBuffers[currentFrame] each frame so the
        // deferred shade can index by primaryInstanceId-1 to make the
        // FC-halving decision per-pixel instead of scene-wide.
        std::vector<uint32_t> meshMovedBits_;
        std::array<Buffer, kFramesInFlight> meshMovedBitsBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> meshMovedBitsBufferCapacity{};
        // Per-entry sticky "recently moved" countdown backing GeometryDesc._pad —
        // holds the moved flag through zero-substep stall frames of a fixed-step
        // integrator under variable dt (a 1-frame flag gap disarms the _pad-gated
        // reproject guards and lets a moving object's reflection ghost into the
        // ground it vacated). Host-side only; see the stamping loop in
        // VulkanCoreFrame.cpp.
        std::vector<uint32_t> meshMovedSticky_;

        // Definition moved to VulkanSceneTypes.hpp.
        using MeshEntry = vulkan::impl::MeshEntry;

        // Definition moved to VulkanSceneTypes.hpp.
        using SnapNode = vulkan::impl::SnapNode;
        static constexpr uint32_t kSnapKindMask   = 3u;
        static constexpr uint32_t kSnapKindOther  = 0u;
        static constexpr uint32_t kSnapKindMesh   = 1u;
        static constexpr uint32_t kSnapKindLine   = 2u;
        static constexpr uint32_t kSnapKindPoints = 3u;
        static constexpr uint32_t kSnapHasPos     = 4u;
        static constexpr uint32_t kSnapHasNorm    = 8u;
        static constexpr uint32_t kSnapWire       = 16u;
        static constexpr uint32_t kSnapOverlay    = 32u;
        static constexpr uint32_t kSnapTet        = 64u;
        static constexpr uint32_t kSnapParticle   = 128u;
        // material()->visible == false. The node is still recorded (so the
        // replay walk can detect a re-show) but contributes no MeshEntry /
        // LineEntry — matches three.js / GLRenderer, which drops a
        // material-hidden object from the render list. [[#mat-visible]]
        static constexpr uint32_t kSnapMatHidden  = 256u;
        // threepp::LOD node (kind Other + typed sn.lod view). Both walks run
        // the camera-driven level selection on it — three.js/GLRenderer
        // parity (projectObject calls lod.update(camera) as it projects).
        static constexpr uint32_t kSnapLod        = 512u;
        // Unlit transparent flat-color mesh (layered SVGs, in-scene UI
        // panels): folded into isOverlay so it renders through the raster
        // overlay pass, which draws in traversal order — i.e. paint order.
        // The traced blend path can neither sort overlapping transparent
        // surfaces nor tie-break exactly-coplanar layers (equal ray t), so
        // GL-parity layering is only reachable by rasterizing in order.
        static constexpr uint32_t kSnapUiBlend    = 1024u;
        // Mesh on VulkanRenderer::kSensorOnlyLayer (MeshEntry::sensorOnly).
        // A snapshot bit like every other routing decision, so enabling or
        // disabling the layer on a live mesh forces the re-expansion that
        // moves it in or out of the sensor group.
        static constexpr uint32_t kSnapSensorOnly = 2048u;
        // The node was effectively hidden when recorded (its own or an
        // ancestor's Object3D::visible was false). The walks enter a hidden
        // node's subtree unless it is, or sits under, a LOD.
        static constexpr uint32_t kSnapHidden     = 4096u;
        // The node sits under a LOD node: its visibility is the LOD's level
        // selection, so a flip takes the full re-expansion as before.
        static constexpr uint32_t kSnapUnderLod   = 8192u;
        // Object3D::resident when recorded. The expansion decides which hidden
        // meshes have entries, so a change re-expands.
        static constexpr uint32_t kSnapResident   = 16384u;
        // A node the edit-proportional path (diffSceneWalk, trySlotDelta) does
        // not add or remove: a line or points with an overlay entry, a LOD, a
        // Sky, a ParticleField, an overlay or particle-billboard mesh (none of
        // which a snapshot node's spanIndex names). A change at or under one
        // takes the full expansion.
        static constexpr uint32_t kSnapNoDelta    = 32768u;
        std::vector<SnapNode> sceneSnapshot_;
        // The full expansion's reuse of the last one (VulkanCoreScene.cpp,
        // reuseNode): the last walk's nodes, and their index by object (built
        // only when the walk meets a node out of the last walk's order).
        // Members so their storage survives between expansions.
        std::vector<SnapNode> prevSceneSnapshot_;
        std::unordered_map<const Object3D*, uint32_t> prevSnapIndex_;
        // THREEPP_VULKAN_WALK_CHECK=1: every expansion that reuses also walks
        // without reuse and compares the two, node for node; the mismatches
        // land here (and on stderr).
        uint64_t walkCheckMismatches_ = 0;
        // A mesh with entries was hidden or shown this frame (MeshEntry::hidden):
        // the TLAS fill, the frustum culls, the draw records and the emitter
        // walk re-read the flags. Set by the snapshot replay or the expansion,
        // consumed in ensureSceneBuilt.
        bool hiddenFlippedThisFrame_ = false;

        // Classification-routing flags for a mesh — shared by the snapshot
        // build and the replay walk so the two can't drift.
        uint32_t snapMeshFlags(Mesh& m, const MaterialWithWireframe* wf) const;

        // Replay the last expansion's traversal against the snapshot. true ⇒
        // tree shape, visibility, classification routing and all mesh/geom/mat
        // pointers are unchanged since the last full expansion. Also re-runs
        // LOD level selection (camera-driven visibility mutation) on kSnapLod
        // nodes — a level switch makes the walk diverge from the snapshot ⇒
        // false ⇒ the full pass re-expands with the new level. A mesh with
        // entries that was hidden or shown (Object3D::visible, material
        // visible) is no divergence: its span flips MeshEntry::hidden in place.
        bool sceneSnapshotMatches(Object3D& scene, Camera& camera);
        // The replay's test of one node against its record: false = the node
        // changed. A mesh with entries hidden or shown is no change: `flip`
        // says so, applied once the walk is accepted.
        struct SnapFlip {
            size_t snIdx = 0;
            bool hidden = false;   // Object3D-hidden (own flag or an ancestor's)
            bool matHidden = false;// material()->visible == false
        };
        bool snapNodeUnchanged(const SnapNode& sn, Object3D& o, bool vis, bool underLod, Camera& camera,
                               bool& flip, SnapFlip& f);

        // The expansion walk's record of one node (VulkanCoreScene.cpp), shared
        // by the full expansion and the edit-proportional diff walk, so both
        // derive a node, its span and its entries the same way.
        struct SceneWalkOut {
            std::vector<MeshEntry>& built;
            std::vector<vulkan::impl::LineEntry>& lines;
            std::vector<vulkan::impl::EntrySpan>& spans;
            std::vector<std::pair<ParticleField*, uint32_t>>& fields;
            std::vector<SnapNode>& snap;
            // Lazy rows: an InstancedMesh that had entries gets its first only
            // (the slot path derives the rest).
            bool lazyRows = false;
            bool anyRowsDeferred = false;
            // The meshes with spans before this walk (by address and id),
            // built at the first question.
            std::unordered_map<const Mesh*, unsigned int> shownBefore;
            bool shownBeforeBuilt = false;
        };
        bool hadEntriesBefore(SceneWalkOut& w, const Mesh& m);
        void recordSceneNode(SceneWalkOut& w, Object3D& o, bool vis, bool underLod, Camera& camera);

        // ── The edit-proportional path ──────────────────────────────────
        // A frame whose replay walk fails used to walk the whole scene again
        // (the full expansion) and match every span of it to the live ones
        // (trySlotEdit): ~6 ms an edit frame on the pine forest's ~13k nodes,
        // whether the edit added one cell or several. diffSceneWalk walks once
        // more against the last snapshot, matching each node among its old
        // parent's old children by subtree range (SnapNode::subtreeEnd): kept
        // nodes are copied, new or changed ones recorded anew, removed
        // subtrees collected. trySlotDelta then frees the removed spans in
        // place and places the new ones, with no copy, match or sort of the
        // spans that stay. Anything else (a line, a LOD, a field, a Sky, an
        // overlay, an instanced count change) takes the full path as before.
        // THREEPP_VULKAN_NO_DELTA=1 turns it off (the A/B lane).
        struct SceneDelta {
            std::vector<SnapNode> snap;      // the new snapshot; kept nodes keep their live spanIndex
            std::vector<MeshEntry> built;    // the entries of the spans recorded anew
            std::vector<vulkan::impl::EntrySpan> spans;    // those spans (first: into built)
            std::vector<vulkan::impl::LineEntry> lines;    // stays empty (a line takes the full path)
            std::vector<std::pair<ParticleField*, uint32_t>> fields;// stays empty
            std::vector<uint32_t> spanSnap;  // per new span: its node in snap
            std::vector<uint32_t> removed;   // live spans whose nodes are gone or recorded anew
            std::vector<SnapFlip> flips;     // in-place visibility flips (snIdx into snap)
            uint32_t swaps = 0;              // meshes placed again with a geometry, material or map swapped
            bool changed = false;
            bool fallback = false;
        };
        static bool deltaDisabled();
        // useVersions: a child whose Object3D::structureVersion is the one its
        // node recorded is copied with its whole subtree, unvisited (its flags
        // are then settled by flatSnapshotCheck).
        bool diffSceneWalk(Object3D& scene, Camera& camera, SceneDelta& d, bool useVersions);
        void diffSceneNode(SceneDelta& d, SceneWalkOut& w, Object3D& o, bool parentVis, bool underLod,
                           int64_t oldIdx, int32_t parentNewIdx, bool useVersions, Camera& camera);
        // ── Walks proportional to the edit (Object3D::structureVersion) ─────
        // A frame whose scene's structure version is the snapshot's had nothing
        // added, removed or swapped anywhere in it, so every node the snapshot
        // names is alive and in place: flatSnapshotCheck tests only the nodes
        // whose state a plain field can change (meshes, lines, points and LOD
        // nodes; flatWatch_), in pre-order, each with its effective visibility
        // from its ancestors (memoised; flatVis_), by the replay's own per-node
        // test, and applies the visibility flips. Anything else returns false
        // and the replay walk decides. THREEPP_VULKAN_NO_VERSIONS=1: never.
        static bool versionsDisabled();
        bool flatSnapshotCheck(Object3D& scene, Camera& camera);
        uint64_t snapshotGen_ = 1;           // bumped whenever sceneSnapshot_ is rebuilt or replaced
        uint64_t flatWatchGen_ = 0;
        std::vector<uint32_t> flatWatch_;
        // Per distinct material of the watched meshes: its routing bits and
        // version as recorded, and its meshes with spans (a version that moved
        // has their maps compared, as the replay does); per distinct geometry,
        // its attribute set's version.
        struct FlatMaterial {
            const Material* mat = nullptr;
            uint32_t first = 0;// a node with it, for its routing reads
            uint32_t bits = 0;
            unsigned int version = 0;
            std::vector<uint32_t> nodes;
        };
        std::vector<FlatMaterial> flatMats_;
        std::vector<std::pair<BufferGeometry*, unsigned int>> flatGeoms_;
        std::vector<int8_t> flatVis_;         // per node: -1 unknown, else effective visibility this frame
        void diffSceneRemoved(SceneDelta& d, size_t first, size_t end);
        bool trySlotDelta(SceneDelta& d);
        // THREEPP_VULKAN_WALK_CHECK=1: after a delta, a full recording compared
        // node for node (mismatches to walkCheckMismatches_ and stderr).
        void deltaWalkCheck(Object3D& scene, Camera& camera);
        // Hide or show span `spanIndex`'s entries in place (MeshEntry::hidden),
        // flag their pixels' history and set hiddenFlippedThisFrame_.
        void setSpanHidden(size_t spanIndex, bool hidden);

        // Previous-frame camera (proj_prev * view_prev) for primary-hit
        // reprojection. One UBO per frame-in-flight so updates don't race the
        // GPU. Per-instance motion matrices (prevWorld * inverse(curWorld)) live
        // in a single SSBO indexed by gl_InstanceCustomIndexEXT; the host keeps
        // an entry-aligned staging (motionScratch_) refreshed only for spans
        // that moved, with prev worlds in prevWorldByEntry_ (identity-remapped
        // by (Mesh*, instanceIndex) across full rebuilds) so each InstancedMesh
        // sub-instance has its own motion delta. First-frame / first-seen
        // entries are identity so reproject is a no-op.
        // Definitions moved to VulkanSceneTypes.hpp.
        using EntryKey     = vulkan::impl::EntryKey;
        using EntryKeyHash = vulkan::impl::EntryKeyHash;
        std::array<Buffer, kFramesInFlight> motionMatBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> motionMatBufferCapacity{};
        // (The per-slot all-identity flag was replaced by the version pair
        //  motionScratchVersion_ / motionUploadedVersion_ — see the members
        //  next to prevWorldByEntry_.)
        // Per-frame emissive-triangle CDF buffer. Each entry is 4 vec4 (64 B):
        //   v0.xyz/area, v1.xyz/cumPower, v2.xyz/power, emission.rgb/_pad.
        // Built fresh each frame from the visible scene; size = numEmissiveTris.
        // Capacity is grown 2× and a descriptor rewrite triggers when the
        // VkBuffer handle changes. Uploaded by buildAndUploadEmissiveTris.
        std::array<Buffer, kFramesInFlight> emissiveTriBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> emissiveTriBufferCapacity{};
        uint32_t emissiveTriCountThisFrame_ = 0;
        float    emissiveTotalPowerThisFrame_ = 0.0f;
        // Per-NEE firefly clamp; pushed to shaders as float bits in slot [11].
        // 1e30f sentinel disables the clamp (set via setFireflyClamp(0)).
        float    fireflyClamp_ = 30.f;
        // Directional-light angular RADIUS (degrees) for the deferred renderer's
        // soft sun shadows — pushed to deferred_shade as tan(radians(deg)) in PC
        // slot [16]. 0 = exact hard 1-ray shadow (old behaviour). Default 0.0°
        // (the real sun subtends ~0.27°): thin occluders cast a stable narrow
        // penumbra instead of a per-frame lit/shadow coin flip under TAA jitter.
        float    sunAngularRadiusDeg_ = 0.0f;
        // Normal-map vMF/Toksvig specular-AA toggle (deferred G-buffer raster
        // path only — see setNormalMapToksvig). Threaded to gbuffer.frag via
        // the raster CameraUbo's prevJitter.z, which is otherwise dead (only
        // prevJitter.xy is read, by nothing in gbuffer.frag/.vert — see
        // uploadRasterCameraUbo) so no descriptor/PC layout change is needed.
        // ON by default: it is a no-op on materials without a normal map and at
        // mip 0 (nLen ~= 1), and strictly reduces normal-map minification
        // shimmer otherwise.
        bool     normalMapToksvig_ = true;
        bool     stochasticTilingIntegerHash_ = false;// setStochasticTilingIntegerHash
        bool     overlayColorGrade_ = false;          // setOverlayColorGrade
        // This view's white balance and colour grade for the overlay passes
        // (OverlayGradeGpu), from its PostComposite; zero flags when the
        // switch is off or the view grades nothing.
        vulkan::OverlayGradeGpu overlayGrade() const;

        // ── Automatic mesh LOD (setAutoLod; ON by default) ──────────────────
        // Measured: Bistro (per-pixel-bound worst case) neutral, fjord flight
        // +32% FPS, quality below animation noise, switch frames stall-free
        // (geomDescs ring). setAutoLod(false) remains as the manual override /
        // debug escape.
        bool autoLod_ = true;
        // Screen-space error budget τ for the selection pass, in RENDER-scale
        // pixels (setAutoLodError). The pass picks the coarsest level whose
        // projected simplification error stays under τ; the raise-hysteresis
        // margin scales with it (0.8·τ). 0.75 px = the validated visually-
        // lossless default; larger trades silhouette error for triangle
        // throughput (1.5-2 px is a sensible perf setting under TAA).
        float lodErrorPx_ = 0.75f;
        // Set by the selection pass in ensureSceneBuilt (VulkanCoreScene.cpp)
        // when ANY entry's chosen level changed this frame; read right after
        // to fold into the same "force a full TLAS rebuild" trigger the
        // deformer paths already use (blasDeformed) — an AS-reference change
        // per VkAccelerationStructureInstanceKHR is only unambiguously legal
        // under MODE_BUILD, not the incremental MODE_UPDATE refit. Also marks
        // geomDescsDirty_ whole-array so RT secondary rays (reflections/GI/
        // lidar/probe update) read the SAME index buffer the swapped BLAS
        // was built from — without it, gl_PrimitiveID from a hit against a
        // coarser BLAS would misindex the still-LOD0
        // GeometryDesc::indexAddress.
        bool lodChangedThisFrame_ = false;

        // ── Cutout ray distance (setCutoutRayDistance; 0 = off, the default) ─
        // Every BLAS is non-opaque so that a cutout caster's hits come back to
        // the shader as candidates to alpha-test. A shadow ray through a tree
        // crosses many leaf cards, and each is a round trip (material read, UV
        // fetch, texture sample). Beyond this distance from the camera a
        // cutout caster's instance is FORCE_OPAQUE instead: the RT core
        // commits the first card it meets. Its shadow is then the cards' own
        // outline, not the leaves'. Reflection and GI rays already trace with
        // gl_RayFlagsOpaqueEXT and are unchanged.
        // The entries are classified for a camera position and re-classified
        // once the camera has moved 5 % of the distance from it (a flag flip
        // is a full TLAS build, so not every frame of a moving camera pays
        // one); an entry that moves by itself waits for that too.
        float cutoutRayDistance_ = 0.f;
        // setCutoutRayCap: the per-ray answer to the same cost (shadowVis in
        // deferred_shade_10_lighting_utils.glsl; ShadePush flags bits 16-19).
        uint32_t cutoutRayCap_ = 0u;
        float    cutoutRayCapFrom_ = 0.f;// ...for what is shaded at least this far from the eye
        float cutoutRayClassifiedFor_ = 0.f;// the distance the entries' bits were derived for
        float cutoutRayCamPos_[3] = {0.f, 0.f, 0.f};// ...and the camera position
        bool  cutoutRayClassified_ = false;// false after a full re-expansion (the bits were reset)
        // Lean frames: re-derive MeshEntry::rayFarOpaque when due. Reads
        // the material table, so call after the material patch. True when any
        // entry's bit flipped (the TLAS must then be refilled and rebuilt).
        bool classifyCutoutFar(const float camPos[3]);
        // Host mirror of the last-uploaded GeometryDesc array (entries-
        // indexed, same layout as the `geomDescs` local built in the full
        // rebuild). The lean auto-LOD path patches indexAddress/indexed in
        // place here; the per-frame ring flush (flushGeometryDescsIfDirty)
        // carries it to the GPU stall-free.
        std::vector<GeometryDesc> geomDescsCached_;
        // Entry-index -> stable per-object instance id, built alongside
        // geomDescs and therefore in the SAME index space and with the SAME
        // lifetime: gl_InstanceCustomIndexEXT indexes both. If geomDescs are
        // valid for a frame, so is this; if entries were renumbered, both were
        // rebuilt together. Lets the lidar readback report the id the raster
        // Ids AOV reports rather than the raw TLAS instance index.
        std::vector<uint32_t> entryStableIds_;
        // Manual threepp::LOD subtrees, rebuilt every FULL scene expansion
        // (cleared at the start of the traverseVisible walk). A mesh entry
        // under one of these is exempt from auto-LOD — its levels are
        // already hand-authored; auto-LOD swapping index buffers underneath
        // an author-selected discrete level would fight that choice. Membership
        // test walks Object3D::parent by raw pointer — no dynamic_cast.
        std::unordered_set<const Object3D*> manualLodLevelRoots_;
        // Background chain-generation worker: a single lazily-started
        // std::thread (not a pool — LOD job volume is low: one job per
        // eligible unique geometry, ever, per geomVersion) draining a
        // mutex+condvar job queue. Jobs snapshot (copy) position/index data
        // at enqueue time, so the worker never touches live BufferGeometry /
        // Vulkan state — no lifetime races with a scene the main thread
        // mutates or tears down while a job runs. Joined in ~Impl before
        // anything Vulkan-related is torn down (the worker is pure CPU).
        std::thread lodWorker_;
        std::mutex lodJobMutex_;
        std::condition_variable lodJobCv_;
        bool lodWorkerStop_ = false;
        // Definition moved to VulkanGeometryState.hpp.
        using LodJob = vulkan::impl::LodJob;
        std::deque<LodJob> lodJobQueue_;
        // Definition moved to VulkanGeometryState.hpp.
        using LodResult = vulkan::impl::LodResult;
        std::mutex lodResultMutex_;
        std::deque<LodResult> lodResultQueue_;
        // Signalled once per result. Only a pinned-clock drain waits on it
        // (drainLodResults): there the frame a chain lands on must not
        // depend on how fast the worker ran.
        std::condition_variable lodResultCv_;
        // Running byte totals for every finalized LOD index buffer + BLAS
        // storage, updated as chains finalize / are evicted — cheap O(1)
        // bookkeeping instead of a full blasCache walk every frame. Gates
        // the hard cap on new chain GENERATION (already-finalized chains are
        // never evicted early to enforce it — see setAutoLod's doc comment).
        uint64_t lodIndexBytes_ = 0;
        uint64_t lodBlasBytes_  = 0;
        bool lodBudgetWarned_ = false;
        static constexpr uint64_t kLodByteBudget = 256ull * 1024ull * 1024ull;
        // Unique-geometry chain-state counters (NOT per-entry), maintained
        // at enqueue/drain time — surfaced via autoLodStats().
        uint32_t lodChainsReadyCount_  = 0;
        uint32_t lodChainsQueuedCount_ = 0;
        // setReplayableLodSchedule: with the clock pinned, drainLodResults
        // waits for the worker (on) or takes what it has (off).
        bool lodReplayableSchedule_ = true;
        // AutoLodStats::drainWaitMs: the time those waits took, cumulative.
        double lodDrainWaitMs_ = 0.0;
        VulkanRenderer::AutoLodStats autoLodStats_{};
        VulkanRenderer::DynamicGeomStats dynGeomStats_{};
        VulkanRenderer::TlasStats tlasStats_{};

        void ensureLodWorkerStarted() {
            if (lodWorker_.joinable()) return;
            lodWorker_ = std::thread([this] { lodWorkerMain(); });
        }
        void lodWorkerMain();
        // Snapshots geom's position/index (or, for non-indexed soup, its
        // position/normal/uv) arrays and hands them to the worker. Called
        // from the selection pass the first time an eligible entry
        // references a record whose lodState is None (never attempted, or
        // reset after a stale-result discard in drainLodResults). Returns
        // false when nothing was enqueued (no position attribute) so the
        // caller never strands the record in Queued with no result coming.
        // Simplifier normal-weight from the enqueueing mesh's material:
        // w = 0.5·glossiness² (glossiness = 1−roughness). Matte foliage
        // (r≈0.85 → w≈0.01) simplifies essentially position-only — the far-
        // vegetation LOD win depends on it; glossy paint (r≈0.25 → w≈0.28)
        // weights normals enough that panels never flatten visibly.
        // Unlit renders no shading at all → 0.
        // Unknown/absent material stays conservative at 0.5. dynamic_cast is
        // fine here: once per geometry lifetime, not per frame.
        static float lodNormalWeightFor(const Mesh& mesh) {
            auto mat = mesh.material();
            if (!mat) return 0.5f;
            if (dynamic_cast<MeshBasicMaterial*>(mat.get())) return 0.f;
            if (auto* r = dynamic_cast<MaterialWithRoughness*>(mat.get())) {
                const float gloss = 1.f - std::clamp(r->roughness, 0.f, 1.f);
                return 0.5f * gloss * gloss;
            }
            return 0.5f;
        }

        bool enqueueLodJob(const BufferGeometry* geomPtr, unsigned int geomVersion, BufferGeometry& geom,
                           float normalWeight);
        // Per-frame chain-finalization budget (drainLodResults, called once
        // per frame from ensureSceneBuilt): up to 2 geometries OR 1 MiB of
        // new level resources (index buffers + BLAS storage), whichever hits
        // first. On the wall clock it takes what the worker has finished; on
        // a pinned clock (setSimTime) it waits for the worker until the
        // budget is spent or no job is outstanding, so the frame each chain
        // lands on is the same in every run, unless setReplayableLodSchedule
        // turned that wait off.
        // All of a frame's level BLAS builds are recorded into ONE
        // one-shot submit (fenced and unwaited, see LodLandingBatch below) —
        // a submit per level (or even per geometry) at Bistro-scale entry
        // counts costs more in queue round-trips than the batching saves.
        // Stale results (record
        // evicted / geomVersion moved on) and failed chains are processed
        // outside the budget — they create no resources and cost only a
        // hash lookup + state write each.
        void drainLodResults();
        // Definition moved to VulkanGeometryState.hpp.
        using LodPendingBuild = vulkan::impl::LodPendingBuild;
        // Creates one level's index buffer + AS handle/storage/address
        // (PREFER_FAST_TRACE, no ALLOW_UPDATE — LOD levels are immutable
        // once built) against `rec`'s EXISTING vertex buffer, and appends
        // the deferred build to `pending`. Mirrors buildBlasFor's BLAS-build
        // shape for the subset that differs (index-only geometry input; the
        // AS device address is queried at creation — it's a property of the
        // storage binding, valid before the build executes). Returns false
        // (leaves `out` default, appends nothing) on a degenerate/zero-
        // primitive level.
        bool buildLodLevelFor(BlasRecord& rec, const geometrylod::Level& level,
                              BlasRecord::LodLevel& out, std::vector<LodPendingBuild>& pending);
        // The level builds of one drain as cmdBuildAccelerationStructures
        // into `cb`.
        void recordLodLevelBuilds(VkCommandBuffer cb, std::vector<LodPendingBuild>& pending);
        // The drain used to end in a submit + vkQueueWaitIdle, i.e. a wait
        // for up to both frames in flight inside render() on every
        // frame a chain landed (the twin: 18 ms of a 24 ms ensureSceneBuilt
        // while its chains landed). Now the builds go out in a fenced,
        // unwaited submit and the chain lands kLodLandDrains drains later,
        // when that fence has signalled (the frame fences between make the
        // wait at landing a formality); compaction is a second such submit.
        // Each landing is a fixed drain count after its submit, so the
        // schedule is the same on the wall clock and under setSimTime.
        using LodLandingBatch = vulkan::impl::LodLandingBatch;
        static constexpr uint64_t kLodLandDrains = kFramesInFlight + 1;
        uint64_t lodDrainSerial_ = 0;
        uint64_t lodBatchSerial_ = 0;
        std::vector<LodLandingBatch> lodLanding_;
        void submitLodLevelBuilds(std::vector<LodPendingBuild>& pending,
                                  const std::vector<const BufferGeometry*>& pendingGeoms,
                                  const std::vector<const BufferGeometry*>& chainGeoms);
        void landLodBatches();
        // Submits `cb` (ended here) with a fresh fence and files the batch.
        void submitLodBatch(LodLandingBatch&& b, VkCommandBuffer cb, std::vector<LodLandingBatch>& into);
        void destroyLodLandingBatches();// teardown, device idle
        // Destroys every level's AS/storage/index and decrements the running
        // byte totals — called from every blasCache erase site (destructor,
        // eviction prune, geomVersion-changed rebuild) so a record's LOD
        // resources never outlive the record itself.
        void destroyBlasLodLevels(BlasRecord& rec);
        // The same for a record with no chain still landing (lodLanding
        // empty), without a device wait: the level BLASes and index buffers
        // go to the retire queue, since only earlier frames still in flight
        // reference them. The in-place refresh's frame-cb route uses it.
        void retireBlasLodLevels(BlasRecord& rec);

        // Free every GPU resource a BlasRecord owns. One shared helper for all
        // six BlasRecord holders (blasCache, skinned / tet / displaced / grass
        // / morphed states), so adding a Buffer to BlasRecord can only ever be
        // missed in one place.
        // NB: LOD levels are NOT freed here — blasCache is the only owner of a
        // LOD chain and calls destroyBlasLodLevels() itself, which also keeps
        // the lodBlasBytes_ accounting straight.
        void destroyBlasRecord(BlasRecord& rec) {
            if (!blasCompactPending_.empty()) {
                blasCompactPending_.erase(
                        std::remove(blasCompactPending_.begin(), blasCompactPending_.end(), &rec),
                        blasCompactPending_.end());
            }
            if (rec.as) ctx->rt().destroyAccelerationStructure(ctx->device(), rec.as, nullptr);
            destroyBuffer(ctx->allocator(), rec.storage);
            destroyBuffer(ctx->allocator(), rec.vertex);
            destroyBuffer(ctx->allocator(), rec.index);
            destroyBuffer(ctx->allocator(), rec.rtIndex);
            destroyBuffer(ctx->allocator(), rec.normal);
            destroyBuffer(ctx->allocator(), rec.uv);
            destroyBuffer(ctx->allocator(), rec.uv2);
            destroyBuffer(ctx->allocator(), rec.color);
            destroyBuffer(ctx->allocator(), rec.prevVertex);
            destroyBuffer(ctx->allocator(), rec.blasScratch);
            destroyBuffer(ctx->allocator(), rec.dynStaging);
            // Zero-copy vertex interop exports, freed here and only here.
            // Note the asymmetry with the buffers above — a foreign API may
            // still hold an import of this memory. Nothing in Vulkan can wait
            // for that; the contract (documented on enableVertexInterop) is that
            // the application stops its producer before dropping the geometry,
            // as it must for the soft-body and ParticleField exports too.
            if (rec.sanitizeDS != VK_NULL_HANDLE && vertexSanitize_) {
                vertexSanitize_->freeRecordDescriptorSet(rec.sanitizeDS);
                rec.sanitizeDS = VK_NULL_HANDLE;
            }
            vulkan::destroyExternalBuffer(ctx->device(), rec.posExt);
            vulkan::destroyExternalBuffer(ctx->device(), rec.nrmExt);
            rec.externalCopy = nullptr;
            rec.interop = false;
        }

        // Cached CDF blob (16 floats per tri) reused across frames when no
        // emissive mesh moved + entries-list size unchanged. The CPU walk in
        // buildAndUploadEmissiveTris is the dominant per-frame cost on
        // texture-heavy scenes like Bistro; reusing the cache makes
        // camera-only motion a memcpy instead of a re-trace of every tri.
        std::vector<float> cachedEmissiveData_;
        uint32_t cachedEmissiveTriCount_ = 0;
        float    cachedEmissiveTotalPower_ = 0.0f;
        size_t   cachedEmissiveEntryCount_ = static_cast<size_t>(-1);
        // Entries the last walk found emissive, as meshMovedBits_ words: only a
        // moved emitter invalidates the cache (buildAndUploadEmissiveTris).
        std::vector<uint32_t> cachedEmissiveEntryBits_;
        uint32_t cachedEmissiveVersion_ = 0;
        // The range gate's exposure term the cached header holds
        // (emissiveGateK(); the emitter NEE skips a point no light reaches).
        float cachedEmissiveGateK_ = 0.0f;
        [[nodiscard]] float emissiveGateK() const;
        // Objects whose emissive glows but is not sampled as a light
        // (setEmissiveCastsLight), keyed by Object3D::id like classIds_. The
        // emitter walk skips them; GeometryDesc.flags bit 6 tells the ray hits
        // to keep their emission. glowOnlyVersion_ is bumped on every change so
        // the cached emitter list above is rebuilt.
        std::unordered_set<unsigned int> glowOnlyObjects_;
        uint32_t glowOnlyVersion_       = 0;
        uint32_t cachedGlowOnlyVersion_ = 0;
        [[nodiscard]] bool isGlowOnly(const Object3D& o) const {
            return !glowOnlyObjects_.empty() && glowOnlyObjects_.count(o.id) != 0;
        }
        std::array<uint32_t, kFramesInFlight> emissiveBufferVersion_{};
        // (prevWorldMats hash map replaced by the entry-aligned
        //  prevWorldByEntry_ / prevWorldValidByEntry_ vectors — identity-
        //  remapped across full rebuilds alongside meshMovedSticky_.)

        // Definition moved to VulkanGeometryState.hpp.
        using MeshFingerprint = vulkan::impl::MeshFingerprint;
        std::vector<MeshFingerprint> prevSceneFingerprint;
        // Per-entry record in TLAS-instance order from the last ensureSceneBuilt
        // call. renderFrame consumes this to compute per-instance motion matrices
        // after the in-flight fence has been waited (safe to write the
        // motionMatBuffers[currentFrame] HOST_VISIBLE buffer).
        //
        // Between two frames the entries still name the meshes the LAST frame
        // drew, and any of them may have been removed and freed since: the
        // caller owns its objects, and a Python app frees one the moment it
        // drops the last reference. ensureSceneBuilt re-validates the list
        // against the live scene (the snapshot replay compares address AND id
        // before it reads an object), so inside a frame every MeshEntry::mesh
        // is live. Code that runs between frames (setEmissiveCastsLight,
        // setStableCorrespondence, enableVertexInterop) must only COMPARE
        // entries, by EntrySpan::meshId or MeshEntry::lodGeomKey, and never
        // follow the pointer.
        std::vector<MeshEntry> lastVisibleEntries_;
        // Per-mesh spans over lastVisibleEntries_ (see EntrySpan). Rebuilt at
        // every full expansion; on lean frames the per-frame loops consult the
        // spans first and touch individual entries only for spans that changed.
        using EntrySpan = vulkan::impl::EntrySpan;
        std::vector<EntrySpan> entrySpans_;
        // Bumped wherever entrySpans_ is laid out anew (the rebuild, a slot
        // edit): VulkanRenderer::admitted's mesh-id index is rebuilt from it
        // only when it moved, so a streamer asking about every pending mesh
        // every frame pays a lookup each, not a scan of every span.
        uint64_t spansVersion_ = 0;
        std::unordered_map<unsigned int, uint32_t> admitIndex_;
        uint64_t admitIndexVersion_ = ~0ull;
        // Entry-aligned previous-frame world matrices for the motion-matrix
        // build (replaces the per-entry (Mesh*, instanceIndex) hash lookups —
        // ~11 ms/frame at 115k entries). Remapped by EntryKey identity across
        // full rebuilds in the same pass that remaps meshMovedSticky_.
        std::vector<std::array<float, 16>> prevWorldByEntry_;
        std::vector<uint8_t> prevWorldValidByEntry_;
        // Persistent motion-matrix staging (16 floats per entry, identity for
        // unmoved spans). motionScratchVersion_ bumps when any block changes;
        // per-FIF uploadedVersion skips the map+memcpy when the slot already
        // holds the current contents.
        std::vector<float> motionScratch_;
        uint32_t motionScratchVersion_ = 0;
        std::array<uint32_t, kFramesInFlight> motionUploadedVersion_{};
        // The entry ranges of motionScratch_ each frame-in-flight slot has not
        // received yet: one moving mesh re-sends its own blocks, not the
        // array (3.2 MB at 50k entries, every frame anything moved).
        std::array<DescDirtyRanges, kFramesInFlight> motionDirty_{};
        void markMotionDirty(uint32_t first, uint32_t count) {
            for (auto& d : motionDirty_) d.mark(first, count);
        }
        void markMotionWhole() {
            for (auto& d : motionDirty_) d.markWhole();
        }
        // TLAS refit instance staging, swapped with pendingTlasInstances_ so
        // both vectors keep their capacity across frames (a fresh 4.8 MB
        // allocation per moving frame at 100k instances otherwise).
        std::vector<VkAccelerationStructureInstanceKHR> tlasInstanceScratch_;
        // Version of every input that shapes DrawInfo/indirect-cmd contents
        // EXCEPT the per-view cull bits (those live on the view as
        // cullVersion). Bumped by ensureSceneBuilt on any non-quiet frame and
        // by the moved-sticky stamp when a flag bit transitions; paired with
        // ViewContext::indirectBuiltSig to skip buildIndirectDrawData when a
        // FIF slot's device buffers already hold the exact same content.
        uint32_t drawInputsVersion_ = 1;
        // Count of entries whose moved-sticky countdown is nonzero — lets the
        // per-frame stamp skip its O(entries) walk on fully settled scenes.
        uint32_t stickyActiveCount_ = 0;
        // Per-entry cull mode cached from each upload of matDescs.
        //   Side::Front  → VK_CULL_MODE_BACK_BIT  (default fast path)
        //   Side::Back   → VK_CULL_MODE_FRONT_BIT
        //   Side::Double → VK_CULL_MODE_NONE
        // Stored as VkCullModeFlags directly so the gbuffer draw loop can
        // hand it straight to vkCmdSetCullMode. Indexed in lock-step with
        // lastVisibleEntries_.
        std::vector<VkCullModeFlags> lastVisibleCullMode_;
        bool sceneBuilt_ = false;

        // ── Stable entry slots (VulkanCoreSlots.cpp) ─────────────────────
        // plans/gpu-driven-instances.md, stage 2a. A mesh added or removed,
        // an InstancedMesh count changed or a mesh shown for the first time
        // edits rows of the entry index space in place (trySlotEdit) instead
        // of the structural rebuild: removed spans become FREE spans
        // (EntrySpan::isFree), new ones take free runs or the end, nothing
        // compacts, and the rows that did not change are not touched.
        // slotCapacity_: the entries every entry-indexed GPU buffer (the
        // geometry / material desc rings, the TLAS instance buffers and
        // storage, the motion and moved-bit buffers) was sized for at the
        // last structural rebuild, or grown to since (growSlotPool).
        uint32_t slotCapacity_ = 0;
        // VulkanRenderer::reserveEntrySlots: the least the pool is laid out or
        // grown to, by the next rebuild or slot edit.
        uint32_t reservedSlots_ = 0;
        // Set by trySlotEdit on the frame it edits; read by the lean tail
        // (a class change: full TLAS build, draw inputs, emitter re-walk).
        bool slotsEditedThisFrame_ = false;
        // The mesh every free slot names: one triangle with position and
        // normal, a default material, identity matrixWorld, never in a scene.
        std::shared_ptr<Mesh> slotTombstone_;
        Mesh& slotTombstone();
        // A free span over [first, first + count), and the entry every slot
        // of one holds.
        EntrySpan slotFreeSpan(uint32_t first, uint32_t count, uint64_t parkedUntil);
        MeshEntry slotFreeEntry();
        // Slots an InstancedMesh span is laid out with past its count, so a
        // setCount grows in place: none for a plain mesh or a full instance
        // buffer, else min(capacity - count, max(count / 2, 64)). Both the
        // structural layout and the slot path leave it.
        static uint32_t slotSlackFor(const EntrySpan& sp, uint32_t count);
        // THREEPP_VULKAN_NO_SLOTS=1: every content change is a structural
        // rebuild in today's dense layout, the A/B lever against the slots.
        static bool slotsDisabled();
        // MaterialDesc::materialAssetIdx outlives one structural rebuild's
        // dense numbering: the rebuild leaves its map here (with each
        // material's id, the address-recycling guard) and the slot path
        // hands a material it has not seen the next index.
        std::unordered_map<const Material*, std::pair<unsigned int, uint32_t>> slotMatAssetIdx_;
        uint32_t slotNextMatAssetIdx_ = 0;
        uint32_t slotMaterialAssetIdx(const Material* mat);
        // Called at the end of the full expansion with what it built. Edits
        // lastVisibleEntries_, entrySpans_ and every entry-indexed row in
        // place and returns true, or touches nothing the full rebuild reads
        // and returns false (the change is structural: the caller rebuilds).
        bool trySlotEdit(std::vector<MeshEntry>& built, std::vector<EntrySpan>& builtSpans,
                         const std::vector<std::pair<ParticleField*, uint32_t>>& builtFields);
        // The rows both slot paths (trySlotEdit, trySlotDelta) write: a span's
        // record (its geometry's in blasCache, or a grass field's own), the
        // fingerprint the full path's slow branch derives for a mesh, the
        // tombstone's row, the row arrays grown to `newSize`, and one entry's
        // row written as new or as a tombstone.
        const BlasRecord* slotRecordOf(const MeshEntry& e);
        MeshFingerprint slotFingerprintOf(Mesh& m);
        void slotTombRows(MeshEntry& e, MeshFingerprint& fp);
        void slotGrowRows(size_t oldSize, size_t newSize, const MeshEntry& tombEntry, const MeshFingerprint& tombFp);
        void slotFillRow(size_t i, const MeshEntry& be, const MeshFingerprint& fp0, const BlasRecord* rec,
                         const GeometryDesc& gd, const MaterialDesc& md0, uint32_t stableId);
        void slotTombRow(size_t i, const MeshEntry& tombEntry, const MeshFingerprint& tombFp);
        // The lean path's per-span world-matrix refresh (scene.3), shared with
        // the frame trySlotEdit edited.
        void refreshSpanMatrices();
        // A mesh's MaterialDesc before its instance color, with every map
        // admitted to the bindless array; materialAssetIdx left for the
        // caller. One function for the structural rebuild, the material
        // patch and the slot path.
        MaterialDesc buildMaterialDesc(Mesh& m);
        // An entry's GeometryDesc for the selected BLAS level, as the
        // structural rebuild and the slot path both write it.
        GeometryDesc buildGeometryDesc(const MeshEntry& en, const BlasRecord& rec, const LodGeomSel& lodSel) const;

        // Path-traced LIDAR scanner — see vulkan/LidarScanner.{hpp,cpp}.
        // Owns its own RT pipeline + SBT + descriptor set; reuses the
        // main TLAS + geomDescs + matDescs via per-scan binding updates.
        // Synchronous scan API exposed publicly via
        // VulkanRenderer::scanLidar. Constructed lazily on first use so
        // the SPV + RT pipeline cost is paid only when a user wants it.
        std::unique_ptr<vulkan::LidarScanner> lidar_;

        // GPU event-camera detector. Lazy-constructed on first
        // setEventCameraEnabled call. Owns the persistent log-history
        // image, the accumulator image, and the per-frame readback ring.
        std::unique_ptr<vulkan::EventCameraDetector> eventCam_;
        bool eventCamEnabled_ = false;
        vulkan::EventCameraDetector::Params eventCamParams_{};
        // What event_shade feeds the detector: the G-buffer Lambert proxy or
        // the presented frame. See VulkanRenderer::EventCameraSource.
        EventCameraSource eventCamSource_ = EventCameraSource::Shaded;

        // The source the frame actually runs. Final needs a final frame
        // (eventsOnlyMode skips producing one) and a device that can imageLoad
        // the BGRA8 swapchain without a format qualifier; otherwise Shaded.
        [[nodiscard]] EventCameraSource effectiveEventCamSource() const {
            if (eventCamSource_ == EventCameraSource::Final && !eventsOnlyMode_ &&
                ctx && ctx->storageImageReadWithoutFormat()) {
                return EventCameraSource::Final;
            }
            return EventCameraSource::Shaded;
        }
        // True when the event camera reads the raw raster G-buffer — the one
        // consumer for which per-frame raster jitter is not resolved later but
        // leaks straight into a sensor as false motion (jittered silhouette
        // coverage flips → a static scene fires ~6.7k events/frame).
        // The Final source reads the post-TAA frame, where jitter has already
        // been resolved, so it keeps jitter on. Every raster-jitter gate keys
        // off this, not off eventCamEnabled_.
        [[nodiscard]] bool eventCamReadsGbuf() const {
            return eventCamEnabled_ &&
                   effectiveEventCamSource() == EventCameraSource::Shaded;
        }

        // Events-only render mode. When on, recordCommandBuffer skips
        // every pass downstream of the raster G-buffer prepass: deferred
        // shade, denoise, TAA, hybrid overlay, upscale. The swapchain is
        // cleared to black so the sprite overlay (event accumulator) has a
        // clean canvas; event_shade + event_detect run in the outer
        // beginFrameAndRecord flow exactly as in the normal render path.
        // Designed for high-rate event-camera sampling
        // (~500 Hz target) where the renderer's visual output isn't
        // displayed at all — only the event accumulator readout matters.
        // Requires hybrid OR TAA to be on so the gbuf prepass actually
        // runs (event_shade reads its normal + ids attachments).
        bool eventsOnlyMode_ = false;

        // Event-camera shade pipeline. Runs immediately after the raster
        // G-buffer pass (when event camera is on) and writes a clean
        // deterministic luma buffer that the detector consumes instead of
        // the stochastically-shaded, noisier swapchain copy. Allocated
        // lazily alongside the detector; descriptor bindings refresh
        // per-frame to track the current frame's gbuf views + material/light
        // buffers.
        vulkan::Buffer        eventLumaBuf_{};
        VkDescriptorSetLayout eventShadeDsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      eventShadePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            eventShadePipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      eventShadeDescPool_       = VK_NULL_HANDLE;
        // One descriptor set per frame-in-flight. The set is rewritten every
        // frame (gbuf views + material/light buffers are per-frame), so a
        // single shared set would be updated while the other in-flight frame
        // still has it bound in a pending command buffer —
        // VUID-vkUpdateDescriptorSets-None-03047. The GPU reading the racing
        // update (wrong-frame gbuf) was the event-camera "binary flicker".
        std::array<VkDescriptorSet, kFramesInFlight> eventShadeDescSets_{};
        uint32_t              eventLumaW_ = 0;
        uint32_t              eventLumaH_ = 0;

        // Debug-view resolve (setHybridDebugView): a compute pass that
        // visualizes one G-buffer channel to the swapchain. Replaces the old
        // raw blit, which could not correctly show the signed motion or
        // integer ids attachments. Created lazily on first debug-view frame.
        // Layout moved to VulkanGpuLayouts.hpp.
        using DebugResolvePC = vulkan::impl::DebugResolvePC;
        VkDescriptorSetLayout debugResolveDsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      debugResolvePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            debugResolvePipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      debugResolveDescPool_       = VK_NULL_HANDLE;
        // One descriptor set per frame-in-flight, rewritten every frame the
        // debug view is active (gbuf views are per-frame, the storage target
        // is the acquired swapchain image). A single shared set would be
        // updated while the other in-flight frame still had it bound in a
        // pending command buffer — VUID-vkUpdateDescriptorSets-None-03047,
        // firing per frame on the AOV readback path (render_aov routes
        // through this pass every frame).
        std::array<VkDescriptorSet, kFramesInFlight> debugResolveDescSets_{};

        // User-requested sensor resolution. 0 means "track swapchain";
        // any non-zero pair pins the detector + luma buffer at that res
        // and the event_shade pass nearest-samples the (typically larger)
        // gbuf into it. Real DVS sensors live in the 128² – 640×480 range,
        // far below the swapchain — running the detector at sensor-native
        // res cuts the per-frame work by the squared scale factor.
        uint32_t eventCamUserW_ = 0;
        uint32_t eventCamUserH_ = 0;

        // THREEPP_DENOISE=0 disables the deferred SVGF denoiser from the
        // environment — an A/B discriminator for "is this artifact shading or
        // temporal/denoise?" without plumbing a flag through every example.
        bool denoiseEnabled_ = vulkan::env::flag("THREEPP_DENOISE", true);

        // MSAA G-buffer dominant-sample resolve — see vulkan/GbufResolve.
        // Lazily constructed the first time gbufMsaaSamples_ > 1 (mirrors
        // debugResolvePipeline_'s on-demand pattern); stays null at the
        // msaa=1 default so that path allocates nothing extra.
        std::unique_ptr<vulkan::GbufResolve> gbufResolve_;

        // ── GPU skinning compute pipeline ──────────────────────────────────
        // Replaces the cpuSkin() loop. One dispatch per skinned mesh per
        // frame when bones change, recorded into the main per-frame command
        // buffer alongside the BLAS rebuild. Frees the CPU thread from
        // the synchronous per-vertex linear-blend math + blocking BLAS
        // submit (was ~10 ms / frame on stormtrooper-density meshes).
        //
        // Pipeline + descriptor pool live in vulkan/SkinningPipeline.{hpp,cpp};
        // per-mesh descriptor sets (state->skinDescSet) live in SkinnedMeshState.
        std::unique_ptr<vulkan::SkinningPipeline> skinning_;
        // GPU tetrahedral-skinning pipeline for PhysX soft bodies (mirrors skinning_).
        std::unique_ptr<vulkan::TetSkinningPipeline> tetSkinning_;

        // Per-view state moved to VulkanViewContext.hpp (the hybrid-prepass
        // note and the per-view ownership rationale travel with it).
        using RasterGbufImages = vulkan::impl::RasterGbufImages;

        // Definition moved to VulkanViewContext.hpp.
        using ViewContext = vulkan::impl::ViewContext;
        // Every view this renderer drives. views_[0] is the PRIMARY (the
        // swapchain-presenting one) and always exists — it is created in the
        // constructor and never removed, so `view()` is unconditionally valid.
        std::vector<std::unique_ptr<ViewContext>> views_;
        // The view currently being set up / recorded. Always views_[0] today;
        // the per-view render loop retargets it. A raw pointer (not an index)
        // so a view() call is one indirection in the hot record path.
        ViewContext* curView_ = nullptr;
        // Handles are dense and monotonic, never reused. 0 is reserved for the
        // primary (and doubles as "invalid" from addView), so a stale handle
        // from a removed view can never alias a later one.
        uint32_t nextViewId_ = 1;
        // Total bytes VMA currently has allocated across every heap. Sampled
        // before/after a view's allocation to report what that view cost —
        // measured rather than computed from a table of formats that would rot
        // the moment someone adds an accumulator to the G-buffer.
        [[nodiscard]] VkDeviceSize vmaAllocatedBytes() const {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice(), &mp);
            std::vector<VmaBudget> budgets(mp.memoryHeapCount);
            vmaGetHeapBudgets(ctx->allocator(), budgets.data());
            VkDeviceSize total = 0;
            for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
                total += budgets[i].statistics.allocationBytes;
            return total;
        }
        [[nodiscard]] VkDeviceSize vmaReservedBytes() const {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice(), &mp);
            std::vector<VmaBudget> budgets(mp.memoryHeapCount);
            vmaGetHeapBudgets(ctx->allocator(), budgets.data());
            VkDeviceSize total = 0;
            for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
                total += budgets[i].statistics.blockBytes;
            return total;
        }
        ViewContext&       view()       { return *curView_; }
        const ViewContext& view() const { return *curView_; }
        // The primary view, by name, for the (many) places that mean "the
        // swapchain view" rather than "whichever view is being recorded".
        ViewContext&       primaryView()       { return *views_[0]; }
        const ViewContext& primaryView() const { return *views_[0]; }
        // Did entry `i` survive THIS view's frustum cull? Out-of-range reads
        // answer "yes", which is the conservative direction: a view that has
        // not culled yet (or an entry list that grew since) draws everything
        // rather than dropping geometry.
        [[nodiscard]] bool viewCulled(size_t i) const {
            const auto& c = curView_->inFrustum;
            return i >= c.size() || c[i] != 0;
        }
        // <<< VIEWCTX_END

        VkRenderPass rasterGbufRenderPass = VK_NULL_HANDLE;
        // MSAA render pass, keyed by sample count (2 or 4). Only the pass
        // matching gbufMsaaSamples_ is ever created; the other stays
        // VK_NULL_HANDLE. Kept separate from rasterGbufRenderPass (the 1×
        // path) so the default (msaa=1) code path is 100% untouched.
        VkRenderPass rasterGbufRenderPassMS = VK_NULL_HANDLE;
        VkPipeline   rasterGbufPipelineMS         = VK_NULL_HANDLE;
        VkPipeline   rasterGbufIndirectPipelineMS = VK_NULL_HANDLE;
        VkPipeline   rasterGbufDecalPipelineMS    = VK_NULL_HANDLE;
        VkPipeline   rasterGbufParticlePipelineMS = VK_NULL_HANDLE;
        // Sample count backing the MS pipelines/render pass/images above (0
        // until first created; tracks which count they were built for, so a
        // 2→4 change knows to tear down and rebuild rather than reuse).
        VkSampleCountFlagBits gbufMsaaBuiltSamples_ = VK_SAMPLE_COUNT_1_BIT;
        // 1x1 dummy MS images (5, mirroring normal/depth/ids/uv/albedo) bound
        // to deferred_shade.comp's dispatch-B sampler2DMS bindings when
        // gbufMsaaSamples_ == 1 — sampler2DMS is a distinct SPIR-V type from
        // sampler2D, so (unlike other "1x1 dummy" bindings elsewhere) this
        // can't reuse a single-sample dummy; a real multisample image is the
        // minimum valid stand-in (2x where the device has it, else 4x — see
        // ensureGbufDummyMS). Created once, lazily, on first use.
        std::array<Image2D, 5> gbufDummyMS_{};
        bool gbufDummyMSCreated_ = false;

        VkDescriptorSetLayout rasterDsLayout       = VK_NULL_HANDLE;
        VkPipelineLayout      rasterPipelineLayout = VK_NULL_HANDLE;
        VkPipeline            rasterGbufPipeline   = VK_NULL_HANDLE;
        // Indirect-drawing variant: uses gbuffer_indirect.vert with bindless
        // vertex pulling, declares zero vertex input bindings, consumes the
        // per-frame DrawInfo SSBO at binding 4. Selected by default for the
        // gbuf pass since it collapses N draws into 1-4 vkCmdDrawIndirect
        // calls — see recordRasterGbufPass.
        VkPipeline            rasterGbufIndirectPipeline = VK_NULL_HANDLE;
        // Decal variant of the indirect pipeline (bucket [3], drawn last):
        // depth-write OFF, attachments 0-3 write-masked to zero, attachment 4
        // (albedo+metalness) alpha-blended with an RGB-only write mask — the
        // decal lerps its albedo over the receiver's, everything else (normal,
        // ids, motion, depth, metalness) stays the receiving surface's.
        VkPipeline            rasterGbufDecalPipeline = VK_NULL_HANDLE;
        // ParticleField variant: particlefield_gbuf.vert over the SAME render
        // pass, layout, fragment stage and blend/depth state as the indirect
        // pipeline. It exists as a sibling rather than a flag because it breaks
        // the one-instance-per-draw contract gbuffer_indirect.vert documents —
        // a field is one draw with instanceCount = the device-side live count,
        // so gl_InstanceIndex is the particle index and the DrawInfo index
        // arrives by push constant.
        VkPipeline            rasterGbufParticlePipeline = VK_NULL_HANDLE;
        // Host mirror of particlefield_gbuf.vert's push_constant block
        // (scalar layout; 64-bit members first so nothing needs padding).
        struct ParticleFieldPC {
            uint64_t posAddr;
            uint64_t prevPosAddr;
            uint64_t oriAddr;
            uint32_t drawIdx;
            uint32_t wSemantic;
            float    invUniformRadius;
            // F4 distance LOD. camPos is the WORLD position of the view being
            // recorded, which is why it rides the push constant rather than the
            // FieldDesc: the same field is drawn once per view and the gate has
            // to move with the camera.
            float    lodFar;
            float    camPos[3];
            float    lodFade;
            float    nearCull;
            float    _pad;
        };
        static_assert(sizeof(ParticleFieldPC) == 64,
                      "ParticleFieldPC drifted from particlefield_gbuf.vert");
        // (rasterDescPool / rasterDescSets / drawInfoBuffers /
        //  indirectCmdBuffers and their capacities moved to ViewContext. Only
        //  the LAYOUT stays shared — every view's set has the same shape, but
        //  a pool sized for one view's frames-in-flight cannot serve two.)

        // Hybrid raster overlay (wireframe + Line/LineSegments). Runs after
        // TAA resolve, draws onto the swapchain with the G-buffer's depth
        // attachment as a read-only depth source so overlays are correctly
        // occluded by the traced/rasterized scene geometry. No descriptor
        // sets — the only input is a push constant (mvp + color).
        VkPipelineLayout overlayPipelineLayout      = VK_NULL_HANDLE;
        VkPipeline       overlayWireframePipeline   = VK_NULL_HANDLE;
        // Solid-fill counterpart for MeshBasicMaterial-style overlays — flat
        // color, depth-tested, but rendered as filled triangles instead of
        // wireframe lines. Selected per-draw based on the material's
        // `wireframe` flag (true → wireframe pipeline, false → basic).
        VkPipeline       overlayBasicPipeline       = VK_NULL_HANDLE;
        // Alpha-blended counterpart to overlayBasicPipeline. Same state
        // except colorBlendAttachmentState's blendEnable=TRUE with
        // SRC_ALPHA / ONE_MINUS_SRC_ALPHA. Selected when the mesh's
        // material has `transparent == true`.
        VkPipeline       overlayBasicTransparentPipeline = VK_NULL_HANDLE;
        // Line-topology pipelines for Line / LineSegments objects. Same
        // overlay shaders, only input-assembly topology differs:
        //   LineSegments → LINE_LIST  (vertices in pairs)
        //   Line         → LINE_STRIP (connected polyline)
        // Lines are never part of the traced/rasterized scene; they're
        // collected separately from Mesh entries and walked in the overlay
        // record loop.
        VkPipeline       overlayLineListPipeline    = VK_NULL_HANDLE;
        VkPipeline       overlayLineStripPipeline   = VK_NULL_HANDLE;
        // Per-vertex color counterparts. Use overlay_color.vert/.frag
        // (location 1 = inColor) and require a 2-binding vertex input.
        // Picked when the Line's geometry has a "color" attribute AND the
        // material has vertexColors == true (matches three.js semantics).
        VkPipeline       overlayLineListColoredPipeline  = VK_NULL_HANDLE;
        VkPipeline       overlayLineStripColoredPipeline = VK_NULL_HANDLE;
        // Blended counterparts to the four line pipelines above, which all
        // share the wireframe pipeline's blend-OFF state — that silently
        // dropped Material::transparent/opacity/blending for Line overlays
        // (an additive streamline drew as an opaque near-black stroke).
        // Selection mirrors GLState::setBlending's gate: blend only when
        // NOT (Normal && !transparent). Indexed [colored][strip][mode] with
        // mode 0 = alpha (SRC_ALPHA, 1-SRC_ALPHA), 1 = additive
        // (SRC_ALPHA, ONE) — the same factors the GL backend programs.
        VkPipeline       overlayLineBlendPipelines[2][2][2] = {};
        // Vertex-coloured mesh fill (overlay_color shaders, TRIANGLE_LIST,
        // dynamic cull like overlayBasicPipeline). Lets unlit transparent
        // vertex-coloured basics take the kSnapUiBlend overlay route instead
        // of landing opaque in the G-buffer (see snapMeshFlags). Indexed
        // [0] opaque, [1] alpha-blended, [2] additive.
        VkPipeline       overlayMeshColoredPipelines[3] = {};
        // Additive counterpart to overlayBasicTransparentPipeline, so a flat
        // Blending::Additive basic mesh matches the GL backend too.
        VkPipeline       overlayBasicAdditivePipeline = VK_NULL_HANDLE;
        // Vertex-coloured wireframe (overlay_color shaders, POLYGON_MODE_LINE,
        // blend-off like the flat wireframe). HemisphereLightHelper-style
        // helpers paint a "color" attribute on a wireframe basic material;
        // the flat wireframe pipeline drew them in the material colour
        // (default white) instead.
        VkPipeline       overlayWireframeColoredPipeline = VK_NULL_HANDLE;
        // Points pipeline — POINT_LIST topology. Always vertex-coloured;
        // a Points object without a "color" attribute renders as plain
        // material colour (vertex colour defaults to white in that case
        // because we still bind the geometry's color buffer if present,
        // and skip the draw if it's missing in the dispatch path below).
        // Writes gl_PointSize from PointsMaterial::size, encoded in the
        // push constant's color.w slot for this pipeline only.
        VkPipeline       overlayPointListPipeline        = VK_NULL_HANDLE;
        // Depth prepass that fills rasterGbufs[f].unjitDepth using the
        // unjittered VP. Reuses the raster pipeline's descriptor set + push
        // constants (same camera UBO, same model matrix push). Runs after
        // recordRasterGbufPass and only renders non-overlay geometry.
        VkPipeline       overlayDepthPrepassPipeline = VK_NULL_HANDLE;
        // The prepass's cheap form (overlay_depth_import.frag): when the only
        // overlay content is ParticleField billboards, the G-buffer's depth is
        // resampled into the overlay depth attachment by one fullscreen
        // triangle instead of rasterizing the scene a second time at display
        // resolution — see overlayDepthImportsGbuffer(). One descriptor set
        // per frame in flight naming that slot's G-buffer depth, rewritten
        // only when the view changes (imageViewSerial), from the post-fence
        // window of its own slot (prepareOverlayDepthImport).
        VkPipeline            overlayDepthImportPipeline_  = VK_NULL_HANDLE;
        VkPipelineLayout      overlayDepthImportLayout_    = VK_NULL_HANDLE;
        VkDescriptorSetLayout overlayDepthImportSetLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      overlayDepthImportPool_      = VK_NULL_HANDLE;
        VkSampler             overlayDepthImportSampler_   = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> overlayDepthImportSets_{};
        std::array<uint64_t, kFramesInFlight>        overlayDepthImportSerials_{};

        // ── Hardware-MSAA overlay ───────────────────────────────────────────
        // The overlay renders post-TAA, so its vector edges (SVG fills, lines,
        // wireframes, points) get no AA from TAA (deliberately after it, to
        // avoid ghosting) and none from the swapchain (Vulkan has no
        // multisampled swapchains). So when Canvas antialiasing > 1 the whole
        // overlay pass rasterizes into a multisampled color+depth pair and
        // vkCmdEndRendering RESOLVES it onto the swapchain — the exact
        // mechanism GL gets for free from a 4x default framebuffer.
        //
        // Because the swapchain already holds the composited scene and the
        // MS target starts undefined, the first draw in the pass is a
        // fullscreen "scene inject" that seeds every sample from a 1-sample
        // copy of the swapchain (a pass may not sample its own target).
        // That makes every blend state in the pass composite against the true
        // background and makes untouched pixels resolve back unchanged.
        //
        // overlaySamples_ == 1 keeps the original direct-to-swapchain path
        // (no inject, no scratch copy, no resolve).
        uint32_t              overlaySamples_    = 0;// 0 = not yet resolved
        VkSampleCountFlagBits overlaySampleBits_ = VK_SAMPLE_COUNT_1_BIT;
        Image2D               overlayMsColor_{};  // swapchain format, N samples
        Image2D               overlayMsDepth_{};  // D32_SFLOAT, N samples
        Image2D               overlayAaScratch_{};// 1-sample swapchain copy (inject source)
        VkDescriptorSetLayout overlayInjectSetLayout_      = VK_NULL_HANDLE;
        VkDescriptorPool      overlayInjectPool_           = VK_NULL_HANDLE;
        VkDescriptorSet       overlayInjectSet_            = VK_NULL_HANDLE;
        VkPipelineLayout      overlayInjectPipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            overlayInjectPipeline_       = VK_NULL_HANDLE;

        // ── Splat depth stamp ──────────────────────────────────────────────
        // Writes the Gaussian-splat depth AOV into the overlay's depth
        // attachment between the splat composite and the overlay draw, so a
        // wireframe / line / sprite BEHIND a cloud is occluded by it. See
        // shaders/splat_overlay_depth.frag and recordSplatOverlayDepthStamp.
        // One descriptor set per frame in flight (each names that frame's AOV
        // image and is rewritten only when the view handle changes, so the
        // update never lands on a set the GPU is still reading).
        // Mirrors the push_constant block of splat_overlay_depth.frag — the
        // two move together.
        struct SplatStampPC {
            float    aovScale[2];
            float    paneOrigin[2];
            float    aovLimit[2];
            float    projA;
            float    projB;
            uint32_t ortho;
        };
        VkDescriptorSetLayout splatStampSetLayout_      = VK_NULL_HANDLE;
        VkDescriptorPool      splatStampPool_           = VK_NULL_HANDLE;
        VkPipelineLayout      splatStampPipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            splatStampPipeline_       = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> splatStampSets_{};
        std::array<uint64_t, kFramesInFlight>        splatStampSetSerials_{};// imageViewSerial
        // Sticky: set the first frame a scene holds BOTH splat clouds and
        // overlay content, and never cleared. It forces the depth AOV on
        // (splatDepthAov()) because the stamp has nothing to read otherwise.
        // Sticky rather than per-frame because turning the AOV off again means
        // reallocating the render-extent resources — a device idle — every
        // time a gizmo is hidden and shown. The cost of leaving it on is one
        // R32 image per frame in flight plus a guarded store per covered pixel.
        bool splatOverlayDepth_ = false;

        // Canvas antialiasing rounded DOWN to a power of two the device can
        // actually use for BOTH a color and a depth framebuffer attachment,
        // capped at 8. Resolved once (the answer can't change: it depends only
        // on the Canvas parameter and the physical device) and cached, because
        // the sample count must be known before createOverlayPipeline builds
        // the pass's pipelines AND before createRasterGbufImages decides
        // whether to allocate the single-sample unjitDepth at all.
        uint32_t overlaySamples() {
            if (overlaySamples_ != 0) return overlaySamples_;
            uint32_t want = canvas.samples();
            if (want < 2) {
                overlaySamples_    = 1;
                overlaySampleBits_ = VK_SAMPLE_COUNT_1_BIT;
                return overlaySamples_;
            }
            if (want > 8) want = 8;
            uint32_t p = 8;
            while (p > want) p >>= 1;// round DOWN to a power of two
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            // VK_SAMPLE_COUNT_N_BIT == N numerically, so the count doubles as
            // its own flag bit.
            const VkSampleCountFlags supported = props.limits.framebufferColorSampleCounts &
                                                 props.limits.framebufferDepthSampleCounts;
            while (p > 1 && !(supported & static_cast<VkSampleCountFlags>(p))) p >>= 1;
            overlaySamples_    = p;
            overlaySampleBits_ = static_cast<VkSampleCountFlagBits>(p);
            return overlaySamples_;
        }

        // Same answer as overlaySamples(), as the VkSampleCountFlagBits the
        // pipeline / image create infos want.
        VkSampleCountFlagBits overlaySampleBits() {
            overlaySamples();
            return overlaySampleBits_;
        }

        // Lazily (re)size the multisampled overlay targets + the 1-sample
        // inject scratch to the swapchain extent and point overlayInjectSet_
        // at the scratch. Single images (NOT per-frame-in-flight): each is
        // written and read inside one command buffer, and cross-frame WAR is
        // handled by UNDEFINED-discard barriers at the point of use. Extent
        // only changes across a swapchain recreate (device drained) and the
        // first call precedes any submit that references the set, so the
        // descriptor update never races the GPU. Old images go through the
        // frame-serial retire queue. No-op when MSAA is off.
        void ensureOverlayMsaaImages(VkExtent2D ext);

        // ── ParticleSystem billboard pass ───────────────────────────────────
        // The Vulkan backend has no generic ShaderMaterial path, so the particle
        // Mesh (a custom billboard ShaderMaterial whose quad is expanded in the
        // vertex shader) is drawn here, in the post-TAA overlay block: depth-
        // tested against unjitDepth, composited onto the swapchain. Two blend
        // variants chosen per material at draw time. See createParticlePipeline /
        // the particle draw loop in the overlay block, and particle.vert/.frag.
        VkDescriptorSetLayout particleDescSetLayout_  = VK_NULL_HANDLE;
        VkPipelineLayout      particlePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline particlePipelineNormal_   = VK_NULL_HANDLE;// alpha blend, depth-test on
        VkPipeline particlePipelineAdditive_ = VK_NULL_HANDLE;// additive blend, depth-test off

        // ── ParticleField billboards (plans/particle-atmosphere.md F-D) ─────
        // A sibling of the two above rather than a variant of them, drawn in
        // the same slot: the post-upscaler overlay render pass, right after
        // the legacy billboard loop. That position is deliberate:
        //
        //   • After the upscalers, the quads are never fed to TAA / DLSS /
        //     FSR — a 3-px spark crossing 20 px in a frame is exactly the
        //     content those filters mis-handle, and an additive glow has no
        //     depth or motion vector to give them anyway.
        //   • It is where transparents already composite, so field billboards
        //     land in the same order relative to lines, wireframe and legacy
        //     particles.
        //   • The pass already binds the unjittered depth read-only, so the
        //     quads get correct occlusion against scene geometry for free.
        //
        // Zero vertex bindings and zero uniform buffers: the quad is
        // vertex-less, the positions come by buffer_reference and the per-field
        // appearance record comes by device address. The one descriptor is the
        // optional sprite texture, which reuses the legacy path's set layout,
        // its per-frame pool and its 1x1 white default.
        // (plans/particle-volumetric-sprites briefly gave this pipeline a SET 1
        // — the field's r16f density mirror, marched by the vertex stage. R8
        // moved the marches into particlefield_transmit.comp, which is where
        // the image is bound now, so the exception closed again and the draw
        // side is back to one set.)
        VkPipelineLayout fieldBillboardPipelineLayout_ = VK_NULL_HANDLE;
        // At overlaySampleBits() — the primary's overlay pass.
        VkPipeline       fieldBillboardPipeline_       = VK_NULL_HANDLE;
        // At 1 sample — a secondary view composites into its own colour target
        // with no overlay MSAA. Aliases the above when overlaySamples() == 1,
        // and is only a second object when it has to be.
        VkPipeline       fieldBillboardPipeline1x_     = VK_NULL_HANDLE;
        bool             fieldBillboardPipeline1xOwned_ = false;
        // Alpha-over siblings: identical but for the blend state
        // (premultiplied SRC_ALPHA-over instead of ONE/ONE) — blending is
        // pipeline state, so they need their own objects. Created alongside
        // the additive pair, bound per field by BillboardRepr::alphaOver.
        // The scene's one sun, snapshotted by updateLightsUbo for the
        // billboard slice — the brightest DirectionalLight in the lights UBO
        // (already the env-sun/scene-sun decision EnvSunPolicy made), plus the
        // summed ambient. Kept as three host vectors so the pass needs no
        // lights descriptor set. Defaults: no sun, no ambient, so a `lit`
        // field in a scene with neither draws black rather than undefined.
        float            bbSunDirWorld_[3]  = {0.f, 1.f, 0.f};
        float            bbSunRadiance_[3]  = {0.f, 0.f, 0.f};
        float            bbAmbient_[3]      = {0.f, 0.f, 0.f};
        VkPipeline       fieldBillboardAlphaPipeline_   = VK_NULL_HANDLE;
        VkPipeline       fieldBillboardAlphaPipeline1x_ = VK_NULL_HANDLE;
        bool             fieldBillboardAlphaPipeline1xOwned_ = false;
        // Billboard-only bloom chain. Created lazily on the first frame a
        // field asks for a glow — a scene with no sparks allocates no target,
        // compiles no pipeline and records no pass.
        std::unique_ptr<vulkan::BillboardGlowPass> billboardGlow_;
        bool billboardGlowReadyThisFrame_ = false;
        // BillboardRepr::softDistance: the linear scene depth the soft fade
        // reads, one buffer per (view, frame in flight). Created on the first
        // frame a visible field asks for a soft edge; until then nothing of it
        // exists and no pass is recorded.
        std::unique_ptr<vulkan::BillboardSoftDepth> billboardSoftDepth_;
        // Host mirror of particlefield_billboard.{vert,frag}'s push block
        // (scalar layout). Exactly 128 B — the range every Vulkan
        // implementation guarantees — so anything per-field lives in the
        // device-address record instead and only the per-view camera is
        // here. proj is unjittered: this pass runs after the temporal resolve.
        struct FieldBillboardPC {
            float    proj[16];  //   0
            float    mv[3][4];  //  64  rows of the affine view * model
            uint64_t paramsAddr;// 112
            // Per-view record (exposure, toneMapMode, fog terms) rides behind
            // this address — the push block has no room past 128 B.
            uint64_t viewAddr;  // 120
        };
        static_assert(sizeof(FieldBillboardPC) == 128,
                      "FieldBillboardPC drifted from particlefield_billboard.vert");
        // Billboard glow pipelines.
        //   • ...GlowPipeline_ draws the same quads into BillboardGlowPass's
        //     half-extent linear-HDR target: rgba16f, one sample, no depth
        //     attachment (see that class's header for why occlusion is given
        //     up at this resolution).
        //   • ...CompositePipeline_ is the fullscreen additive draw that folds
        //     the finished pyramid into the swapchain, inside the overlay
        //     render-pass instance — the composite point does not move, so the
        //     glow stays outside TAA/upscaling like the billboards themselves.
        VkPipeline       fieldBillboardGlowPipeline_ = VK_NULL_HANDLE;
        VkPipelineLayout fieldGlowCompositeLayout_   = VK_NULL_HANDLE;
        VkPipeline       fieldGlowCompositePipeline_ = VK_NULL_HANDLE;
        struct FieldGlowPC {
            float    intensity;
            float    exposure;
            uint32_t toneMapMode;
            uint32_t _pad;
            float    invDisplay[2];
            vulkan::OverlayGradeGpu grade;// setOverlayColorGrade; zero flags when off
        };
        static_assert(sizeof(FieldGlowPC) == 120,
                      "FieldGlowPC drifted from particlefield_glow.frag");
        // Per-frame combined-image-sampler pool, reset at the top of the draw
        // loop (mirrors OverlayPass::spriteDescPools_).
        std::array<VkDescriptorPool, kFramesInFlight> particleDescPools_{};
        // 1×1 white default bound when a particle system has no texture.
        Image2D particleWhiteTex_{};
        static constexpr uint32_t kMaxParticleTexPerFrame = 64;

        // ── Overlay-pass fog (Phase 2b) ─────────────────────────────────────
        // The post-TAA overlay draws world-space ParticleSystem billboards
        // (chimney smoke etc.) that never saw the unified air-fog / murk medium.
        // particle.frag now binds this per-frame snapshot at SET 1 (the texture
        // stays at set 0) and applies the closed-form fog per fragment. The
        // world-space Sprite pipeline shares particlePipelineLayout_ but its
        // shader does not reference set 1, so it simply ignores the (still-bound)
        // fog set. std140 — mirrors particle.frag's OverlayFog block.
        // Layout moved to VulkanGpuLayouts.hpp.
        using GpuOverlayFogUbo = vulkan::impl::GpuOverlayFogUbo;
        VkDescriptorSetLayout overlayFogDescSetLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      overlayFogDescPool_      = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> overlayFogDescSets_{};
        std::array<Buffer, kFramesInFlight>          overlayFogUbos_{};

        // Definition moved to VulkanSceneTypes.hpp.
        using ParticleGeomRec = vulkan::impl::ParticleGeomRec;
        std::unordered_map<const BufferGeometry*, ParticleGeomRec> particleGeomCache_;

        // ── Lit particles (particle_light.comp) ────────────────────────────
        // Per-frame world-space particle centers (host-written, one vec4 per
        // particle across ALL visible systems, in overlay draw order) + the
        // per-particle {light,T}/{fogAdd} results the compute pass writes and
        // particle.vert reads (set 1 binding 1 of the billboard pipeline).
        // Eagerly allocated in createParticlePipeline (fixed footprint, ~1.5 MB
        // total). Systems past kMaxLitParticles fall back to the unlit path via
        // the kUnlitBase firstInstance sentinel (KEEP IN SYNC with particle.vert).
        static constexpr uint32_t kMaxLitParticles = 16384;
        static constexpr uint32_t kUnlitBase       = 0x40000000u;
        std::array<Buffer, kFramesInFlight> particleCenterBufs_{};// host-visible SSBO, kMaxLitParticles × vec4
        std::array<Buffer, kFramesInFlight> particleLightBufs_{}; // device-local SSBO, kMaxLitParticles × 2 × vec4
        VkDescriptorPool particleIoDescPool_ = VK_NULL_HANDLE;    // particle_light.comp set 1
        std::array<VkDescriptorSet, kFramesInFlight> particleIoDescSets_{};
        // Filled by prepareParticleLighting() each frame; consumed by the
        // subclass scene-dispatch hook (compute) + the overlay particle loop
        // (per-draw firstInstance base).
        uint32_t particleLightCount_ = 0;// live particles prepped this frame (0 = pass off)
        std::unordered_map<const void*, uint32_t> particleLitBase_;// mesh → base particle index

        // CPU-side staging for the centers upload (avoids per-frame realloc).
        std::vector<float> particleCenterScratch_;

        // Create particle_light.comp's IO sets (set 1: centers in, results out)
        // once both their layout owner (deferredShade_, lazily constructed on
        // the first scene build) and the particle buffers (createParticlePipeline)
        // exist. Idempotent; buffers are fixed at creation so the sets are
        // written exactly once.
        void ensureParticleIoSets();

        // Gather every visible ParticleSystem's particle centers (model → world,
        // every 4th vertex of the coincident-quad layout) into this frame's
        // centers buffer and assign each mesh its base index, in the SAME order
        // the overlay loop draws them. Runs in recordCommandBuffer BEFORE the
        // scene-dispatch hook; the hook dispatches particle_light.comp over the
        // result. No-op (count 0) when the scene has no particles, the billboard
        // pipeline isn't initialised yet, or the IO sets can't exist (no
        // deferred shade → nothing would ever light the result buffer).
        void prepareParticleLighting();

        // Definition moved to VulkanSceneTypes.hpp.
        using ParticleTexRec = vulkan::impl::ParticleTexRec;
        std::unordered_map<const Texture*, ParticleTexRec> particleTexCache_;

        // ── World-space Sprite billboards (screenSpace == false) ────────────
        // 3D-positioned camera-facing sprites (e.g. the TPS shooter's impact
        // "particles"). The Vulkan renderer only composites screen-space (HUD)
        // sprites via OverlayPass; world-space ones are drawn here in the
        // depth-tested overlay block. Reuses particlePipelineLayout_ +
        // particleDescSetLayout_ + particleDescPools_ + particleTexCache_ (the
        // push-constant size, set-0 sampler layout, and texture cache all match);
        // only the pipeline (perspective billboard vertex shader + interleaved
        // pos/uv vertex input) and the shared static quad are new.
        VkPipeline spriteWorldPipeline_ = VK_NULL_HANDLE;
        // Shared canonical sprite quad (4 interleaved pos.xyz+uv.xy verts + 6
        // indices) — identical for every Sprite, so one static copy serves all.
        Buffer spriteQuadVtx_{};
        Buffer spriteQuadIdx_{};
        // Definition moved to VulkanSceneTypes.hpp.
        using WorldSpriteEntry = vulkan::impl::WorldSpriteEntry;
        std::vector<WorldSpriteEntry> lastVisibleSprites_;

        // Definition moved to VulkanSceneTypes.hpp.
        using LineEntry = vulkan::impl::LineEntry;
        std::vector<LineEntry> lastVisibleLines_;
        // (currVPunjit_ / currViewUnjit_ / currProjUnjit_ moved to ViewContext.)

        // Definition moved to VulkanViewContext.hpp.
        using RasterCameraData = vulkan::impl::RasterCameraData;
        // (rasterCameraUbos, the prev-VP / prev-jitter / sky-reproject /
        //  depth-linearization TAA bookkeeping, and the prev camera pose all
        //  moved to ViewContext — see the RasterCameraData note there.)
        float deferredCamRotAngle_ = 0.f;

        // Nearest-filter sampler used by the deferred shade to read gbuffer
        // attachments. Nearest avoids bilinear smearing of normal/motion/ids
        // at silhouettes — primary visibility from raster is already exact-pixel.
        VkSampler gbufSampler_ = VK_NULL_HANDLE;

        // Fallback UV vertex buffer: 8 bytes of zeros, bound to vertex input
        // location 2 when a mesh has no UV attribute (rec->uv.handle is null).
        // Lets the gbuffer pipeline keep a fixed 3-binding layout regardless
        // of per-mesh UV availability.
        Buffer dummyUvBuffer_{};

        // (taa_ moved to ViewContext — TaaResolve owns the temporal HISTORY
        //  ping-pong, which is the single most view-specific thing in the
        //  renderer. See the pass-ownership note there.)
#if defined(THREEPP_WITH_FSR)
        // AMD FidelityFX FSR 3.1 upscaler. When its context creates successfully
        // (Windows only) it REPLACES the TAA temporal resolve — see the
        // fsrActive_ branch in VulkanCoreRecord.cpp. FSR writes its upscaled
        // linear-HDR output into TaaResolve's history write slot, so PostComposite
        // tonemaps it via the HDR-input path (its postFlags-bit-3 hdrMode +
        // recordPostFinalize) with no
        // descriptor rewrite. On create failure fsrActive_ stays false and the
        // TAA path runs unchanged. See FsrUpscaler.{hpp,cpp}.
        std::unique_ptr<vulkan::FsrUpscaler> fsr_;
        bool   fsrActive_    = false;// true once the FSR context created
        bool   fsrResetNext_ = true; // force FSR history reset on the next dispatch
        double fsrPrevTimeSec_ = -1.0;// for FSR frameTimeDelta (ms)
        // Camera params stashed at raster-camera-upload time (no camera at the
        // record site) — FSR needs near/far/vertical-FOV for depth reconstruction.
        float  fsrCamNear_ = 0.1f, fsrCamFar_ = 1000.f, fsrCamFovY_ = 1.0f;
        // The sub-pixel [-0.5,0.5] jitter applied to the projection this frame,
        // sourced from FSR's own sequence so the dispatch jitterOffset matches.
        float  fsrJitterX_ = 0.f, fsrJitterY_ = 0.f;
        // Runtime on/off (setFsr), distinct from fsrActive_ (compiled + context
        // created). Default on so a THREEPP_WITH_FSR build uses FSR by default.
        bool   fsrEnabled_ = true;
#endif
#if defined(THREEPP_WITH_DLSS)
        // NVIDIA DLSS Super Resolution. Same seam as FSR (replaces the TAA
        // temporal resolve, writes into TaaResolve's history write slot) and
        // OUTRANKS it: when both are available + enabled, DLSS runs — see the
        // useDlss() branch in VulkanCoreRecord.cpp. On NGX init/feature-create
        // failure (non-RTX GPU, old driver) dlssActive_ stays false and the
        // FSR/TAA fallback runs unchanged. See DlssUpscaler.{hpp,cpp}.
        std::unique_ptr<vulkan::DlssUpscaler> dlss_;
        bool   dlssActive_    = false;// true once the NGX feature created
        bool   dlssResetNext_ = true; // force DLSS history reset on the next dispatch
        double dlssPrevTimeSec_ = -1.0;// for DLSS frameTimeDelta (ms)
        // The sub-pixel [-0.5,0.5] jitter applied to the projection this frame
        // (built-in Halton — DLSS's recommended sequence/phase count matches
        // jitterPhaseCount_'s FSR2-style scaling).
        float  dlssJitterX_ = 0.f, dlssJitterY_ = 0.f;
        // Runtime on/off (setDlss), distinct from dlssActive_. Default on so a
        // THREEPP_WITH_DLSS build uses DLSS by default.
        bool   dlssEnabled_ = true;
        // Self-heal budget for sticky NGX evaluate failures (renderFrame);
        // replenished by the display-resize recreate funnel.
        uint32_t dlssHealTries_ = 0;
#endif
        // An external upscaler (FSR/DLSS) compiled in AND created (available).
        // Gates the HDR-mode plumbing both reuse — hdrOut_ stays allocated
        // regardless of the runtime toggles, so setFsr/setDlss never need a
        // descriptor rewrite. Always compiled (false without either option) to
        // keep #ifs out of the shared code.
        [[nodiscard]] bool fsrActiveForHdrPlumbing() const {
            bool active = false;
#if defined(THREEPP_WITH_FSR)
            active = active || fsrActive_;
#endif
#if defined(THREEPP_WITH_DLSS)
            active = active || dlssActive_;
#endif
            return active;
        }
        // Per-upscaler availability (compiled in + context/feature created),
        // independent of the runtime toggles — the public *Available() getters.
        [[nodiscard]] bool fsrAvailable() const {
#if defined(THREEPP_WITH_FSR)
            return fsrActive_;
#else
            return false;
#endif
        }
        [[nodiscard]] bool dlssAvailable() const {
#if defined(THREEPP_WITH_DLSS)
            return dlssActive_;
#else
            return false;
#endif
        }
        // DLSS is the ACTIVE upscaler this frame: available AND runtime-enabled.
        // Outranks FSR (see useFsr) and forces the projection jitter on, like FSR.
        [[nodiscard]] bool useDlss() const {
#if defined(THREEPP_WITH_DLSS)
            return dlssActive_ && dlssEnabled_;
#else
            return false;
#endif
        }
        // FSR is the ACTIVE upscaler this frame: available AND runtime-enabled,
        // and DLSS is not running (DLSS outranks FSR when both are on).
        // Drives the record branch and the jitter source (FSR needs jitter, so it
        // also forces the projection jitter on — see uploadRasterCameraUbo).
        [[nodiscard]] bool useFsr() const {
#if defined(THREEPP_WITH_FSR)
            return fsrActive_ && fsrEnabled_ && !useDlss();
#else
            return false;
#endif
        }
        // Runtime FSR on/off. No reallocation — the FSR and TAA paths both keep
        // their resources, so this is frame-to-frame switchable. Resets the
        // temporal history so the switched-to path doesn't inherit the other's
        // accumulation (FSR re-primes via fsrResetNext_).
        void setFsr(bool enabled) {
#if defined(THREEPP_WITH_FSR)
            if (fsrEnabled_ == enabled) return;
            fsrEnabled_ = enabled;
            if (view().taa_) view().taa_->invalidateHistory();
            markMaterialSamplerDirty();// jitter gate flips → sampler policy flips
            fsrResetNext_ = true;
#if defined(THREEPP_WITH_DLSS)
            dlssResetNext_ = true;// path hand-off re-primes whichever runs next
#endif
#else
            (void) enabled;
#endif
        }
        // Runtime DLSS on/off, same switching semantics as setFsr. Turning DLSS
        // off hands the frame to FSR (if available + enabled) or the TAA path.
        void setDlss(bool enabled) {
#if defined(THREEPP_WITH_DLSS)
            if (dlssEnabled_ == enabled) return;
            dlssEnabled_ = enabled;
            if (view().taa_) view().taa_->invalidateHistory();
            markMaterialSamplerDirty();// jitter gate flips → sampler policy flips
            dlssResetNext_ = true;
#if defined(THREEPP_WITH_FSR)
            fsrResetNext_ = true;// path hand-off re-primes whichever runs next
            // FSR was skipped at init while DLSS held the upscaler slot. If DLSS
            // is being turned off, create the FSR context now so the frame hands
            // off to FSR rather than plain TAA. dlssActive_ is still true here, so
            // fsrActiveForHdrPlumbing() was already satisfied and hdrOut_ is
            // allocated — no descriptor rewrite is needed.
            if (!enabled && fsrEnabled_ && !fsr_) {
                fsr_ = std::make_unique<vulkan::FsrUpscaler>(*ctx, kFramesInFlight);
                fsrActive_ = fsr_->create(ctx->swapchainExtent().width,
                                          ctx->swapchainExtent().height);
                fsrResetNext_ = true;
            }
#endif
#else
            (void) enabled;
#endif
        }
        // (bloom_ and post_ moved to ViewContext — bloom_ owns sceneHdr, the
        //  render-extent HDR target every view needs its own copy of, and
        //  post_ owns the display-extent hdrOut fed from it.)
        // Thin-lens depth of field on sceneHdr, recorded between the scene
        // dispatch and the bloom pyramid so bokeh still blooms + tone-maps
        // as HDR (see vulkan/DofPass.hpp). CoC is camera-derived: aperture
        // from camAperture_, focal length + sensor size from the camera's own
        // film gauge (filmHeightM_ below), focus plane at focusDistance_.
        // OFF by default (the whole pass is skipped; zero cost).
        std::unique_ptr<vulkan::DofPass> dof_;
        // Gaussian splats (SplatCloud), composited into sceneHdr between the
        // deferred shade and the DoF — linear HDR, pre-post, so splats get
        // DoF/bloom/tonemap/TAA through the same path as everything else. See
        // vulkan/SplatPass.hpp. Primary view only: the pass owns one set of
        // scratch buffers sized to the extent it was last resized to, and a
        // secondary view would fight it for them.
        // Allocates nothing until a scene actually contains a SplatCloud.
        std::unique_ptr<vulkan::SplatPass> splat_;
        // This frame's SplatClouds, gathered by the sidecar collector in
        // VulkanRenderer::render (see collectSplatClouds) — outside the
        // snapshot machinery, like collectWorldSprites, because a SplatCloud
        // is not a kind the G-buffer path knows how to draw.
        std::vector<vulkan::SplatPass::CloudEntry> lastVisibleSplats_;
        // Splat clouds in the scene but not effectively visible this frame —
        // parked, not evicted: their GPU buffers stay so a visibility toggle
        // costs nothing (see SplatPass::syncClouds).
        std::vector<const SplatCloud*> lastParkedSplats_;
        // Camera-derived splat parameters, stashed by the collector (the last
        // point in the frame that holds a Camera&). The jitter shear is added
        // at record time because the raster does not choose it until later.
        vulkan::SplatPass::RecordParams splatParams_{};
        // This frame's REVERSE-Z projection, stashed alongside — recordSplats
        // composes it with taaSkyReproj_ to get view space -> previous clip.
        float splatProjRevZ_[16]{};
        // setSplatDebugChecksum — the determinism gate's switch. Off by
        // default; the hashes cost two extra dispatches and an atomic per
        // composited pixel.
        bool splatChecksum_ = false;
        // Final image formation (lens warp + sensor noise). Runs after the
        // overlay pass so it applies to everything the camera sees, not just
        // the parts of the frame that existed before the overlay composited.
        std::unique_ptr<vulkan::SensorPass> sensorPass_;
        bool  dofEnabled_    = false;
        float focusDistance_ = 10.f;
        float tanHalfFovY_   = 0.4142f;// stashed in updateCameraUbo (45° default)
        // Sensor height in METRES, stashed in updateCameraUbo from
        // PerspectiveCamera::getFilmHeight() (filmGauge is the film WIDTH in
        // mm; the height follows from the camera's aspect). This is the one
        // sensor the whole camera model shares: FOV comes from it via
        // setFocalLength, and the DoF CoC + cameraIntrinsics() read it back.
        // 0.0197 = the default 35 mm gauge at 16:9. Non-perspective cameras
        // keep whatever the last perspective camera left here.
        float filmHeightM_ = 0.035f / 1.7777778f;
        // Unjittered projection terms stashed in updateCameraUbo, consumed by
        // cameraIntrinsics(): [0]/[5] = 2n/(r-l), 2n/(t-b); [8]/[9] = frustum
        // skew (non-zero under filmOffset / setViewOffset).
        float projP0_ = 1.f, projP5_ = 1.f, projP8_ = 0.f, projP9_ = 0.f;

        // ── Lens + sensor (the non-pinhole half of the camera) ──────────────
        // Both default to inactive, and while inactive the RCAS stage keeps
        // its original texelFetch/CAS path → byte-identical output.
        LensDistortion lens_{};
        VulkanRenderer::SensorNoise sensorNoise_{};
        // Frame counter driving the noise seed. Advances once per recorded
        // frame while noise is on; resetSensorNoise() rewinds it so the same
        // seed replays the same sequence from the top of an episode.
        uint32_t sensorNoiseFrame_ = 0u;

        // Overscan factor for the lens warp: the scene is rendered with the
        // frustum widened by this much so barrel distortion has real geometry
        // to gather into the output corners. 1 = off.
        float lensOverscan_ = 1.f;

        // Is there any image formation left to do after the overlay pass?
        [[nodiscard]] bool sensorStageActive() const {
            return lens_.active() || sensorNoise_.enabled;
        }
        // Overscan only means anything with a lens: without one the warp is
        // identity and a widened frustum would just silently crop the view.
        [[nodiscard]] float effectiveOverscan() const {
            return lens_.active() ? lensOverscan_ : 1.f;
        }

        // Pack this frame's lens + sensor state for the final SensorPass.
        // Intrinsics go over NORMALIZED (fx/W, fy/H, cx/W, cy/H) so the same
        // numbers are correct in that pass, which runs at DISPLAY extent, as
        // they are at the render extent they were measured on. Advances the
        // noise frame counter, so call exactly once per recorded frame.
        [[nodiscard]] vulkan::SensorPass::Params buildSensorParams();
        float bloomIntensity_ = 0.0f;
        // Deferred GI path: ON activates stochastic 1-bounce GI (colour bleed) +
        // temporal accumulation + à-trous. OFF falls back to the deterministic
        // AO+far≈sky approximation. setDenoise(false) → clean fallback.
        // Ray-traced env ambient occlusion / GI. ON: gives the "dirty realistic"
        // ray-traced grounding (contact darkening + 1-bounce GI). Uses the
        // deterministic Fibonacci 64-sample gather (clean + settles), no
        // per-frame flicker.
        bool  deferredAO_ = true;
        // Deferred volumetric spot-light beams (ray-marched single scattering in
        // deferred_shade.comp). σ = 0 disables (the march is skipped entirely).
        float deferredVolDensity_ = 0.f;
        float deferredVolAniso_   = 0.55f;
        // The directional sun shafts / aerial glow are ALWAYS on when the fog medium
        // is present (froxels own the near field, volumetricDirScatter the > 512 m
        // tail). There is no public toggle; this stays true and only feeds the
        // shade/froxel volFog flag.
        bool  deferredVolFog_     = true;
        // setUnlitFogScatter: the unlit branch of the shade takes the volumetric
        // in-scatter too (shade flags bit 28). Off = extinction and ambient only.
        bool  unlitFogScatter_    = false;
        // ── Rotor downwash (setDownwash; scenes/Downwash.hpp) ──────────────
        // uploadDownwash packs the enabled sources into this slot's block once
        // per frame, after the fence wait; downwashAddr_ is then its device
        // address (0 = no downwash this frame) and downwashCount_ the packed
        // count. The G-buffer (RasterCameraUbo.downwash), the grass wind push
        // block and the particle emitters' aux records read these.
        std::shared_ptr<Downwash> downwash_;
        std::array<Buffer, kFramesInFlight> downwashBufs_{};
        VkDeviceAddress downwashAddr_  = 0;
        uint32_t        downwashCount_ = 0;
        // Last frame's centres by source slot (the particles' motion vectors);
        // reset whenever the list's version changes.
        std::array<Vector3, 8> downwashPrevCenters_{};
        uint64_t downwashPrevVersion_ = ~0ull;
        const Downwash* downwashPrevOwner_ = nullptr;
        void uploadDownwash(uint32_t frame);
        // Whether a live jet of the downwash reaches a grass field whose world
        // REST box is `box` (the wind gate and the cull bound): a conservative
        // test against the jet's farthest reach over that box's heights.
        [[nodiscard]] bool downwashReachesBox(const Box3& box) const;
        bool  fogShadowOpaqueCutouts_ = true; // setFogShadowOpaqueCutouts (shade flags bit 29); off = exact
        // Procedural direction-space star field on deferred sky pixels (0 = off).
        float deferredStarIntensity_ = 0.f;

        // (deferredShade_ moved to ViewContext — its per-frame-in-flight
        //  descriptor sets bind this view's G-buffer, reservoirs and sceneHdr,
        //  so the sets themselves are per-view. The pipeline could be shared;
        //  it is not, because a set can never be rewritten while it may be in
        //  flight (VUID-03047), and a per-view pass object rules that out
        //  structurally.)
        // World-space irradiance probe grid (multi-bounce GI for the deferred
        // gather — see vulkan/ProbeGI.hpp). Created alongside deferredShade_;
        // its SH buffer + grid UBO back the deferred set's bindings 36/37, so
        // it must exist whenever deferredShade_ does. probeGIEnabled_ gates the
        // per-frame update dispatch AND the shader-side sampling (grid UBO
        // enable) — default ON (VulkanRenderer::setProbeGI(false) turns it off).
        // probeGridDirty_ re-fits the grid to the scene AABB after each
        // structural scene rebuild (new/removed meshes ⇒ new bounds).
        std::unique_ptr<vulkan::ProbeGI> probeGI_;
        bool probeGIEnabled_  = true;
        bool probeGridDirty_  = true;
        // The primary camera's world position this frame: the centre the
        // probe cascades follow (ProbeGI::recentre) when the scene is too
        // large for one scene-fit grid. Taken from the primary's camera UBO
        // upload, never a secondary view's.
        float probeCamPos_[3] = {0.f, 0.f, 0.f};
        // Lighting-step detector for the probe update. The probes blend 1/32
        // of each update, one update per probe every 8 frames, so a switched
        // light lingered in the grid for thousands of frames; a step in any
        // light's power (probeLightingStepped) runs the next
        // kProbeFastBlendFrames of probe updates at the fast blend
        // (probe_update.comp, push-constant flags bit 1). updateLightsUbo fills
        // the current signature; the dispatch compares it with the last one.
        struct ProbeLightSig {
            std::array<uint32_t, 4> counts{};// dir, point, spot, rect
            std::vector<float>      lum;     // ambient, then one entry per light
            uint32_t                emissiveCount = 0;
            float                   emissivePower = 0.f;
            bool                    valid         = false;
        };
        ProbeLightSig probeLightSig_;
        ProbeLightSig probeLightSigPrev_;
        int           probeFastBlendFrames_ = 0;
        static constexpr int kProbeFastBlendFrames = 32;// four full sweeps
        bool probeLightingStepped();
        // ── Two-phase GPU occlusion culling (setOcclusionCulling) ───────────
        // Phase 1 draws last frame's visible set; a FARTHEST-depth pyramid
        // (occlHiz_, min-reduce) is
        // built mid-frame from that depth; the cull compute tests every
        // record's world AABB against it and phase 2 draws only the newly
        // visible (render passes A/B below — same framebuffer + pipelines,
        // load/store-op variants are render-pass compatible; MSAA gets its
        // own variant pair and the pyramid reduces the raw MS attachment's
        // samples at mip 0). Deformers always draw (their CPU AABB is stale
        // — the frustum-cull rule). Default OFF: the single-pass path
        // records byte-identically and occlHiz_'s image isn't even
        // allocated. Gate: no split-screen scissor.
        std::unique_ptr<vulkan::OcclusionCull> occl_;
        std::unique_ptr<vulkan::HiZPyramid>    occlHiz_;
        bool occlusionCullingEnabled_ = false;
        VkRenderPass occlRenderPassA_   = VK_NULL_HANDLE;// CLEAR + STORE, depth → sampleable
        VkRenderPass occlRenderPassB_   = VK_NULL_HANDLE;// LOAD, final layouts as the single pass
        VkRenderPass occlRenderPassAMS_ = VK_NULL_HANDLE;// MSAA siblings (created with the MS
        VkRenderPass occlRenderPassBMS_ = VK_NULL_HANDLE;// pass on the msaa toggle)
        // True while THIS frame's raster actually ran two-phase (consumers
        // in the same recordCommandBuffer body branch on it).
        bool occlActiveThisFrame_ = false;

        // ── ParticleField (phase 0: buffers + descriptor, no consumer) ──────
        // The device-side position rings and the per-frame FieldDesc SSBO for
        // every threepp::ParticleField in the scene. ALWAYS ON and free when
        // there are no fields: the whole pass is O(fields), and a scene without
        // one does a single empty-vector test per frame.
        std::unique_ptr<vulkan::ParticleFieldPass> particleFieldPass_;
        // (field, its index in lastVisibleEntries_). Written only by a full
        // scene expansion — a field appearing or disappearing is structural, so
        // the snapshot fast path never runs with this stale.
        std::vector<std::pair<ParticleField*, uint32_t>> particleFields_;
        // Reused scratch for the per-frame Rec list handed to the pass.
        std::vector<vulkan::ParticleFieldPass::Rec> particleFieldRecs_;
        // Phase 1. Per visible field, the index of its ONE DrawInfo record in
        // this view's concatenated draw array — the push constant the particle
        // vertex stage reads. Parallel to particleFields_ /
        // particleFieldPass_->drawStates(), rebuilt by buildIndirectDrawData
        // (and restored from the view cache on its skip path, since the field's
        // DrawInfo is as static as every other one).
        struct ParticleDrawSlot {
            const ParticleField* field = nullptr;
            uint32_t             drawIdx = 0;   // index into the DrawInfo SSBO
            VkCullModeFlags      cull = VK_CULL_MODE_BACK_BIT;// from the material's Side
        };
        std::vector<ParticleDrawSlot> particleDrawSlots_;

        // ── ParticleField density volumes (phase 2, plan §3.3) ──────────────
        // Bindings 67/68 of EVERY view's deferred set. Both handles are owned
        // here, not by ParticleFieldPass, because they must be valid from the
        // renderer's first descriptor write — which happens before the pass is
        // lazily constructed, and keeps happening on scenes that never get one.
        //
        //   particleDensityUbos_  — per-FIF ParticleDensityUbo (std140, 160 B),
        //                           rewritten every frame; handle never changes.
        //   particleDensityDummy_ — 1x1x1 R32_UINT in GENERAL, bound to every
        //                           array slot no live volume occupies. Same
        //                           "always bound, harmlessly unused" idiom as
        //                           the ocean/MSAA dummies.
        Buffer  particleDensityUbos_[kFramesInFlight]{};
        Image2D particleDensityDummy_{};
        // Its r16f twin for binding 69 (the shade's linear-sampled mirrors).
        Image2D particleDensityLinDummy_{};
        // Any field contributed density this frame. Forces heteroActive, opens
        // the froxel gate with no clustered lights, and sets the shade's flags
        // bit 11 — the three gates plan §3.3 calls "real, small, easy to miss".
        bool     particleDensityActiveThisFrame_ = false;
        // ParticleFieldPass::densityGeneration() each FIF's deferred set was
        // last written with. Not equal ⇒ that set names a stale (possibly
        // retired) volume view and must be rewritten before it is recorded.
        uint64_t particleDensityDescGen_[kFramesInFlight]{};

        // ── Splat reflection volumes (plans/splat-volume-reflections.md) ────
        // Bindings 70/71 of EVERY view's deferred set. The particle-density
        // block above, mirrored member for member and for the same reasons:
        // both handles are owned HERE rather than by SplatPass because they
        // must be valid from the renderer's first descriptor write, which
        // happens before any cloud exists and keeps happening on scenes that
        // never get one.
        //
        //   splatVolumeUbos_  — per-FIF SplatVolumeUboGpu (std140, 912 B),
        //                       rewritten every frame; handle never changes.
        //   splatVolumeDummy_ — 1x1x1 RGBA16F in GENERAL, bound to every array
        //                       slot no live volume occupies.
        Buffer  splatVolumeUbos_[kFramesInFlight]{};
        Image2D splatVolumeDummy_{};
        // At least one baked volume is bound this frame. Feeds the shade's
        // flags bit 12 — and ONLY on the primary view (recordSceneDispatch),
        // because doc/vulkan_splats.md's scope wall keeps splats out of sensor
        // AOVs and a secondary's water reflection is a sensor AOV.
        bool     splatVolumeActiveThisFrame_ = false;
        // splatVolumeBindKey() each FIF's deferred set was last written with.
        // Not equal ⇒ that set names the wrong volume views and must be
        // rewritten before it is recorded.
        uint64_t splatVolumeDescKey_[kFramesInFlight]{};

        // ── GPU per-instance world matrices (stage 1: producer only) ─────────
        // instance_expand.comp recomputes, per InstancedMesh span, exactly what
        // ensureSceneBuilt's lean refresh bakes into MeshEntry::worldMatrix. No
        // consumer reads it — DrawInfo, motion, cull and the TLAS instance fill
        // all still take the CPU values — so this is pure added cost until
        // stages 2-5 move them over. What it provides now is verification that
        // the GPU producer agrees with the CPU
        // (VulkanRenderer::instanceExpandCheck).
        std::unique_ptr<vulkan::InstanceExpand> instExpand_;
        // OFF by default until a consumer reads the buffer: the pass is
        // measured slower (~+1.0 ms wall at 85k grains, 8 of 8 interleaved A/B
        // pairs) and its output is read by nobody yet.
        // setGpuInstanceExpansion(true) opts in, which is all the A/B lever and
        // VulkanInstanceExpand_test need.
        bool gpuInstanceExpand_ = false;// setGpuInstanceExpansion; the A/B lever
        // The GPU path's span list: an index into entrySpans_ plus the two
        // prefix sums the shader needs. Instanced spans only. Rebuilt (and
        // compared) every frame — it is O(spans), not O(instances), and a diff
        // is what tells the per-fif upload bookkeeping below that its
        // per-span slots no longer mean the same thing.
        struct InstExpandSpan {
            uint32_t spanIdx  = 0;// index into entrySpans_
            uint32_t count    = 0;// instances actually uploaded (clamped to the attribute)
            uint32_t matBase  = 0;// first matrix in the pool
            uint32_t workBase = 0;// exclusive prefix sum of count
            bool operator==(const InstExpandSpan& o) const noexcept {
                return spanIdx == o.spanIdx && count == o.count &&
                       matBase == o.matBase && workBase == o.workBase;
            }
        };
        std::vector<InstExpandSpan> instExpandSpans_, instExpandScratch_;

        // ── GPU draw records (stage 2; DrawGen.hpp, draw_gen.comp) ─────────
        // OFF by default, and only with the expansion on (its world matrices
        // are the input): setGpuDrawGeneration. The primary view's records of
        // eligible instanced spans (static, no occlusion culling this frame)
        // are written by compute; everything
        // else stays the CPU's.
        std::unique_ptr<vulkan::DrawGen> drawGen_;
        bool gpuDrawGen_ = false;
        // The frame's dispatch, filled by buildIndirectDrawData (primary view)
        // and uploaded by prepareDrawGen; drawGenActive_ gates the graph pass.
        std::vector<vulkan::DrawGen::SpanDesc> drawGenSpans_;
        std::vector<vulkan::DrawGen::LodRow>   drawGenLodRows_;// the spans' auto-LOD levels
        uint32_t drawGenWork_   = 0;
        bool     drawGenActive_ = false;
        float    drawGenPlanes_[24]{};
        // setGpuDrawGenerationCheck: the CPU path also builds the GPU spans'
        // records, into drawGenExpected_, for drawGenCheck to compare.
        bool gpuDrawGenCheck_ = false;
        struct DrawGenExpected {
            uint32_t entry  = 0;
            uint32_t record = 0;
            bool     culled = false;
            vulkan::impl::DrawInfoGpu draw{};
            VkDrawIndirectCommand     cmd{};
        };
        std::vector<DrawGenExpected> drawGenExpected_;
        uint32_t drawGenExpectedFrame_ = 0;
        std::array<std::vector<uint32_t>, 4> indirectCpuPosScratch_;// bucket position of each CPU record
        void prepareDrawGen(uint32_t frame);

        // ── GPU TLAS instances (stage 5; TlasGen.hpp, tlas_gen.comp) ───────
        // OFF by default, needs the expansion: setGpuTlasInstances. The lean
        // TLAS fill writes the CPU's instances to the front of the frame's
        // instance buffer and plans tlasGenSpans_; the dispatch appends theirs.
        std::unique_ptr<vulkan::TlasGen> tlasGen_;
        bool gpuTlasGen_ = false;
        std::vector<vulkan::TlasGen::SpanDesc> tlasGenSpans_;
        std::vector<uint64_t> tlasGenLodRows_;// the spans' auto-LOD level BLAS addresses
        uint32_t tlasGenWork_   = 0;
        bool     tlasGenActive_ = false;// this frame's refit carries GPU instances
        // The CPU/GPU split of the TLAS last built: a different one is a full
        // build, never an update (the instance order changes).
        uint64_t tlasGenPartition_ = 0;
        // Instances the dispatch appends after pendingTlasInstances_.
        uint32_t pendingTlasGpuInstances_ = 0;
        // setGpuDrawGenerationCheck: the CPU's instances for the GPU spans.
        std::vector<VkAccelerationStructureInstanceKHR> tlasGenExpected_;
        uint32_t tlasGenExpectedFrame_ = 0;
        uint32_t tlasGenExpectedBase_  = 0;
        void prepareTlasGen(uint32_t frame);
        // The per-entry LOD fade words (entryLodFade_) for the GPU producers:
        // the DrawInfo flag bits and the TLAS mask of a crossfading instance.
        // Per frame-in-flight, host-mapped, re-sent when lodFadeVersion_ moved:
        // the spans whose words changed since that slot's last upload
        // (lodFadeDirty_, as motionDirty_ does for the motion matrices), whole
        // after a rebuild, a slot edit or growth. Bound by the sets
        // prepareDrawGen / prepareTlasGen rewrite every frame, so growth is a
        // plain swap.
        std::array<DescDirtyRanges, kFramesInFlight> lodFadeDirty_{};
        void markLodFadeDirty(uint32_t first, uint32_t count) {
            for (auto& d : lodFadeDirty_) d.mark(first, count);
        }
        void markLodFadeWhole() {
            for (auto& d : lodFadeDirty_) d.markWhole();
        }
        std::array<Buffer, kFramesInFlight> lodFadeBuffers_{};
        std::array<uint32_t, kFramesInFlight> lodFadeUploaded_{};
        std::array<size_t, kFramesInFlight> lodFadeUploadedCount_{};
        uint32_t lodFadeVersion_ = 1;
        VkBuffer uploadLodFades(uint32_t frame);
        // MeshEntry::lodLevel for the GPU producers, a byte per entry: the
        // CPU's selection, read by the spans with an auto-LOD chain. Gathered
        // from the spans with a level above 0 (EntrySpan::lodNonZero), re-sent
        // when lodLevelVersion_ moved: a selection that changed a level, or
        // entries laid out anew.
        std::array<Buffer, kFramesInFlight> lodLevelBuffers_{};
        std::array<uint32_t, kFramesInFlight> lodLevelUploaded_{};
        std::array<size_t, kFramesInFlight> lodLevelUploadedCount_{};
        uint32_t lodLevelVersion_ = 1;
        std::vector<uint8_t> lodLevelScratch_;
        VkBuffer uploadLodLevels(uint32_t frame);
        void recordTlasGen(VkCommandBuffer cb, uint32_t frame);
        bool verifyTlasGen(VulkanRenderer::TlasGenCheck& out);
        // The span's record when draw_gen.comp may write its draws (the
        // primary view's eligibility, minus the per-frame conditions), else
        // null. One predicate for buildIndirectDrawData and the frustum cull.
        const BlasRecord* drawGenEligibleRecord(const EntrySpan& sp);
        // The primary view's frame hands draw_gen.comp spans to cull (stage 4):
        // the CPU cull keeps only its span-level answer for them.
        // Not while the overlay depth prepass DRAWS THE SCENE: it draws one
        // entry at a time off this view's CPU cull, which would then hold only
        // the span-level answer for these spans and draw every instance of a
        // span that is partly in view (the pine forest under its HUD: +2.5 ms
        // of GPU). The prepass draws the scene for overlay meshes, lines and
        // sprites only; for field billboards alone it imports the G-buffer's
        // depth and touches no entry (overlayDepthImportsGbuffer).
        [[nodiscard]] bool drawGenCullsOnGpu() const {
            return !curView_->secondary && gpuDrawGen_ && gpuInstanceExpand_ && drawGen_ && instExpand_ &&
                   !gpuDrawGenCheck_ && !occlusionCullingEnabled_ && !sceneHasOverlayMeshes();
        }
        void recordDrawGen(VkCommandBuffer cb, uint32_t frame);
        bool verifyDrawGen(VulkanRenderer::DrawGenCheck& out);
        uint32_t instExpandMatrixTotal_ = 0;// matrices in the pool
        uint32_t instExpandWorkTotal_   = 0;// dispatch domain
        // Content serial per GPU-path span, bumped whenever the CPU re-baked
        // that span's entries (EntrySpan::movedThisFrame). The per-fif stamps
        // chase it, so a span that moved on frame N is re-sent to BOTH slots
        // rather than only to the slot that happened to be current — the bug
        // a plain "did it move this frame" gate would have.
        //
        // It also closes a hole the layout diff alone cannot see: two instanced
        // meshes swapping places in entrySpans_ with identical counts produces
        // an IDENTICAL InstExpandSpan list, so the diff would keep the old
        // per-slot stamps while slot k now means a different mesh. Safe because
        // entrySpans_ is only ever rebuilt by the full expansion, and that path
        // sets movedThisFrame on EVERY span — so any rebuild bumps every serial
        // and both slots re-upload.
        std::vector<uint64_t> instExpandSerial_;
        std::array<std::vector<uint64_t>, kFramesInFlight> instExpandFifSerial_;

        float bloomThreshold_ = 1.0f;// soft-knee bright-pass cutoff (linear HDR)
        float bloomClamp_ = 0.0f;    // per-tap HDR cap before the bright pass; <= 0 = off
        float sharpenStrength_ = 0.5f;// post-TAA RCAS amount; 0 = off
        float motionBlurAmount_ = 0.f;// post-TAA motion blur: shutter open fraction
                                      // of the frame interval (0.5 = 180°); 0 = off
        float taaBlendAlpha_ = 0.16f;// 10% current, 90% history at the reference rate;
                                    // frame-rate-corrected per frame (see taaPrevTimeSec_)
        // Wall-clock anchor for the frame-rate-aware TAA blend. taaBlendAlpha_ is a
        // per-frame new-sample weight, so holding it fixed ties the history half-life
        // to frame count: the same 90 %/frame retention is an invisible ~10 ms ghost
        // at 200 fps but a long visible smear at 30 fps. Each frame the
        // weight is re-solved (in recordCommandBuffer, against kTaaRefFps) so
        // (1-alpha) is held constant in wall-clock time instead of per frame; the
        // shader's velocity/deviation gates are already per-frame-displacement based,
        // so only this base weight needs the correction.
        double taaPrevTimeSec_ = -1.0;

        // ── Deterministic frame clock ───────────────────────────────────
        // The one time source every frame-path wall-clock read goes through.
        // Negative (the default) = wall clock, behaviour unchanged. When the
        // app drives it (VulkanRenderer::setSimTime, once per frame before
        // render()), TAA/DLSS/FSR frame-dt, the shade's timeSec, foam decay,
        // deform timestamps and the cloud clock all advance on simulation
        // time, which is what makes two same-seed runs produce the same
        // pixels (wall time in the TAA blend alpha alone is enough to make
        // every run diverge from the first history frame onward).
        double simTimeSec_ = -1.0;
        // A headless canvas (ctx->presentSuppressed()) with no setSimTime has
        // no wall clock worth reading: a stepped loop renders as fast as the
        // GPU allows, so the TAA blend and the FSR/DLSS dt would think it runs
        // at hundreds of fps and ghost. advanceHeadlessClock steps this clock
        // 1/60 s per scene-building render() instead (0 on the first), and
        // says so once. Negative = not in use. setSimTime always wins.
        double headlessClockSec_ = -1.0;
        void advanceHeadlessClock();
        [[nodiscard]] double frameNowSec() const {
            if (simTimeSec_ >= 0.0) return simTimeSec_;
            if (headlessClockSec_ >= 0.0) return headlessClockSec_;
            return glfwGetTime();
        }
        // The app's or the headless clock, or negative on the wall clock.
        [[nodiscard]] double pinnedClockSec() const {
            return simTimeSec_ >= 0.0 ? simTimeSec_ : headlessClockSec_;
        }

        // (rasterMatTexValid_ — the binding-3 texture-table gate — lives on
        // ViewContext: the raster sets it guards are per-view.)

        // ── Lower-resolution render mode ────────────────────────────────
        // renderScale_ < 1 runs the deferred shade + the raster G-buffer at
        // renderExtent() instead of the swapchain extent. The TAA pass then
        // upsamples to full resolution (it dispatches at the swapchain extent
        // with a full-res history — see taa_resolve.comp); with TAA off a
        // bilinear blit upscales the low-res denoise output instead. 1.0 →
        // renderExtent() equals the swapchain extent and every pass behaves
        // exactly as before.
        float renderScale_ = 1.0f;
        // (renderScale < 1 is upsampled by TAA straight to the swapchain.)

        // Deferred render extent: swapchain extent × renderScale_, each axis
        // clamped to ≥ 1px. Exactly equal to the swapchain extent when
        // renderScale_ is 1 (the unscaled fast path).
        VkExtent2D primaryRenderExtent() const {
            const VkExtent2D s = ctx->swapchainExtent();
            if (renderScale_ >= 0.999f) return s;
            const auto px = [](uint32_t v, float k) -> uint32_t {
                const auto r = static_cast<uint32_t>(static_cast<float>(v) * k + 0.5f);
                return r < 1u ? 1u : r;
            };
            return {px(s.width, renderScale_), px(s.height, renderScale_)};
        }
        // ── The two extents every pass in the render chain is written against ──
        // renderExtent()  — where the G-buffer rasterizes and the shade
        //                   dispatches (the TAA resolve's INPUT).
        // viewOutExtent() — where the temporal history lives and the frame is
        //                   finally written (the TAA resolve's OUTPUT).
        //
        // The primary DERIVES both from the swapchain on every call, exactly as
        // before. That is deliberate: caching them would introduce a staleness
        // window on resize / renderScale change that the old code could not
        // have, and the primary path must stay provably identical. Only a
        // secondary reads its stored pair, which is fixed at addView time and
        // never tracks the window.
        //
        // Secondaries are native-res by scope: renderExt == outExt, so their
        // TAA resolve is a plain 1:1 temporal filter with no upsampling.
        VkExtent2D renderExtent() const {
            if (curView_ && curView_->secondary) return curView_->renderExt;
            return primaryRenderExtent();
        }
        VkExtent2D viewOutExtent() const {
            if (curView_ && curView_->secondary) return curView_->outExt;
            return ctx->swapchainExtent();
        }
        // ── Per-frame timing instrumentation ─────────────────────────────
        // Managed by vulkan/GpuTimings.{hpp,cpp}. Owns one VkQueryPool per
        // frame-in-flight; exposes begin/end brackets per TimingPass plus CPU
        // record/frame timing. Constructed in the Impl ctor, after ctx is
        // available.
        std::unique_ptr<vulkan::GpuTimings>  gpuTimings_;
        // Ortho/HUD overlay pass. Owns the sprite and
        // ortho-line pipelines, atlas/geometry caches, and per-frame descriptor
        // pools. Constructed after ctx is available (same constraint as gpuTimings_).
        std::unique_ptr<vulkan::OverlayPass> overlayPass_;
        // Set by VulkanRenderer::render(...) right after ensureSceneBuilt
        // and consumed by gpuTimings_->readBack() on the next frame's fence
        // wait so the public getter sees the same frame's CPU + GPU numbers.
        float    pendingCpuEnsureSceneMs_ = 0.f;
        // ReSTIR DI master toggle. When false, chit's primary RIS branch is
        // bypassed and the legacy per-light NEE classic loops run instead
        // (same pattern as bounces). Forwarded to the deferred shade via
        // pc.motionFlags bit 4 each frame.
        bool restirDIEnabled_ = true;
        // Hybrid raster overlay: layer index for opt-in overlay objects
        // (alongside auto-detected wireframe materials + Line/LineSegments).
        // -1 disables layer-based selection.
        int overlayLayer_ = -1;
        // Sensor-only surfaces: the per-scene opt-in for meshes on
        // VulkanRenderer::kSensorOnlyLayer. OFF means they are hit by nothing —
        // no raster in any view, TLAS instance mask 0 — so a scene that never
        // opts in senses and renders exactly as it did before the feature
        // existed. Toggling it clears sceneBuilt_: the masks live in the TLAS.
        // ON is necessary and not sufficient on the raster side: a secondary
        // view also has to ask (ViewContext::sensorSurfaces).
        bool sensorOnlySurfaces_ = false;
        // True when the scene has any content the post-TAA overlay pass will
        // draw this frame: an overlay-tagged mesh, or any Line/LineSegments/
        // Points entry (those always render via the overlay path). The
        // unjittered-depth prepass and the overlay draw BOTH gate on this same
        // current-frame answer, so the prepass that fills + transitions
        // unjitDepth runs in lockstep with the draw that reads it. (Previously
        // the prepass keyed off the *previous* frame's result, so the first
        // frame an overlay appeared — including frame 1 after a resize — the
        // prepass was skipped and the overlay draw read unjitDepth while it was
        // still UNDEFINED: VUID-vkCmdBeginRendering-pRenderingInfo-09588.)
        // Cheap: both lists are already populated by the visibility pass before
        // any command recording, so this is just two size/flag checks.
        // The overlay content that is hard-edged and tested against the
        // prepass's UNJITTERED depth: lines and points, world sprites, and the
        // wireframe / overlay-layer / particle-system entries. These are what
        // the prepass rasterizes the scene for.
        bool sceneHasOverlayMeshes() const {
            if (!lastVisibleLines_.empty()) return true;
            // World-space sprites are drawn in the overlay pass and depth-test
            // against unjitDepth, so they need the prepass + overlay pass too.
            if (!lastVisibleSprites_.empty()) return true;
            // Per SPAN, not per entry — isOverlay is uniform across an
            // InstancedMesh expansion, and isOverlay covers particle
            // billboards too (kSnapParticle folds into isOverlay), so a scene
            // with only particles still triggers the depth prepass + overlay
            // pass the billboard loop needs.
            for (const auto& sp : entrySpans_) {
                if (lastVisibleEntries_[sp.first].isOverlay) return true;
            }
            return false;
        }
        bool sceneHasOverlayContent() const {
            if (sceneHasOverlayMeshes()) return true;
            // ParticleField billboards draw in the overlay pass too, and they
            // are NOT entries flagged isOverlay — a field is one ordinary
            // MeshEntry whose billboard representation is a second, independent
            // draw. A scene of nothing but a campfire's embers would otherwise
            // skip the depth prepass, leaving unjitDepth UNDEFINED for a pass
            // that reads it (VUID-vkCmdBeginRendering-pRenderingInfo-09588).
            // For them alone the prepass takes its cheap form, the G-buffer
            // depth import (overlayDepthImportsGbuffer).
            return sceneHasFieldBillboards();
        }
        // The prepass fills the overlay depth from the G-buffer's depth (one
        // fullscreen triangle) rather than by drawing the scene: nothing but
        // field billboards wants the attachment, and the import pipeline
        // exists. A full-scene depth pass at display resolution cost 0.9 ms of
        // GPU and 1.2 ms of CPU on a 15k-entry forest for a few dozen dust
        // sprites — the jitter-exact depth it buys matters for hard-edged
        // wireframes and lines, not for a soft sprite drawn outside TAA.
        [[nodiscard]] bool overlayDepthImportsGbuffer() const {
            return overlayDepthImportPipeline_ != VK_NULL_HANDLE && !sceneHasOverlayMeshes();
        }
        // G-buffer debug-view mode: blit one G-buffer channel onto the
        // swapchain instead of running the deferred shade. Lets us see the
        // raster output before the shade's ray-query integration.
        enum class HybridDebugView : uint32_t {
            Off    = 0,
            Normal = 1,
            Motion = 2,
            Depth  = 3,
            Ids    = 4,
            Albedo = 5,   // raster-first material G-buffer: linear albedo in rgb
        };
        HybridDebugView hybridDebugView_ = HybridDebugView::Off;
        // VulkanRenderer::LightingView as an integer (0 = the lit frame);
        // the primary view's denoise recombine reads it.
        uint32_t lightingView_ = 0;

        // (cameraUbos moved to ViewContext.)

        // Swapchain image count, sampled once in createSwapchainDependents and
        // never refreshed: pinned for the renderer's lifetime, not merely the
        // current swapchain's. Its one consumer is the primary view's
        // TaaResolve, which sizes its descriptor pool and set vectors from it
        // at construction and indexes them `frame * imageCount + imageIndex`
        // forever after — refreshing this scalar on a swapchain recreate would
        // move the index while the arrays stayed their allocated size.
        //
        // A recreate producing a different count is legal in principle but
        // cannot happen here: every input to the negotiation (vsync_,
        // presentSuppressed_, the surface itself) is fixed at context
        // construction, leaving only caps.min/maxImageCount.
        // recreateSwapchainAndDescriptors verifies that rather than trusting
        // it. If it ever fires, the fix is to rebuild the primary view's
        // TaaResolve there (pool, set vectors and all), not to assign a new
        // value here.
        uint32_t imageCount_ = 0;

        // Per-frame command resources.
        VkCommandPool                                cmdPool = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, kFramesInFlight> cmdBuffers{};
        std::array<VkSemaphore,     kFramesInFlight> imageAvailable{};
        // The stages the submit waits imageAvailable at (endFrame) AND the
        // srcStageMask of every barrier that first touches the acquired image.
        // They must be the same set: a barrier's layout transition is ordered
        // after the semaphore wait only when its first scope includes a waited
        // stage. The UNDEFINED transitions used TOP_OF_PIPE, which is not one,
        // so the presentation engine could still be reading an image this
        // frame was already transitioning (syncval WAR vs vkAcquireNextImageKHR).
        static constexpr VkPipelineStageFlags2 kAcquireWaitStages =
                VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        // Present-wait semaphores are PER SWAPCHAIN IMAGE (indexed by the
        // acquired image index), not per frame-in-flight: a binary semaphore
        // handed to vkQueuePresentKHR stays "in use" until the presentation
        // engine releases that image (signalled by a later acquire of the
        // SAME index), so a frame-indexed semaphore can be re-signalled
        // while the swapchain still holds it (VUID-vkQueueSubmit2-semaphore-
        // 03868). Sized to the swapchain image count; recreated with the
        // swapchain (createRenderFinishedSemaphores).
        std::vector<VkSemaphore> renderFinished;
        std::array<VkFence,         kFramesInFlight> inFlight{};

        // Suppressed-present swapchain state (ctx->presentSuppressed(), i.e. a
        // headless canvas). Nothing is ever presented, so nothing is ever
        // released back to the presentation engine either: each frame-in-flight
        // slot acquires ONE image the first time it runs and keeps it for the
        // swapchain's lifetime. UINT32_MAX = this slot has not acquired yet.
        // Reset by recreateSwapchainAndDescriptors, which is the only thing that
        // invalidates the images.
        std::array<uint32_t, kFramesInFlight> pinnedSwapImage_{};
        // Whether imageAvailable[slot] still carries an unconsumed signal from
        // the acquire. True for exactly the one frame that follows an acquire;
        // false afterwards, so the submit does not wait on an already-consumed
        // binary semaphore (which would deadlock).
        std::array<bool, kFramesInFlight> acquireSemPending_{};

        uint32_t currentFrame = 0;
        bool needsResize = false;

        // ── Frame-serial deferred-deletion (retire) queue ────────────────────
        // Replaces staleness-triggered vkDeviceWaitIdle stalls (runtime texture
        // / material / particle / sprite swaps, AS rebuilds). See
        // VulkanRetireQueue.hpp for the full fence-invariant derivation.
        //
        // frameSerial_ = serial of the frame CURRENTLY being recorded. Advanced
        // once per SUBMITTED frame in endFrame() (in lockstep with currentFrame,
        // so a failed acquire that skips endFrame never desyncs serial↔slot).
        // Any retire()/retireAS() during a frame stamps frameSerial_; the drain
        // at frame start reclaims resources stamped kFramesInFlight+ frames ago.
        uint64_t frameSerial_ = 0;
        vulkan::RetireQueue retireQueue_;

        // Hand a resource to the retire queue, stamped with this frame's serial.
        void retire(Buffer&& b)   { retireQueue_.retire(std::move(b), frameSerial_); }
        void retire(Image2D&& i)  { retireQueue_.retire(std::move(i), frameSerial_); }
        void retireAS(VkAccelerationStructureKHR as) { retireQueue_.retireAS(as, frameSerial_); }
        // Frame-start reclaim: destroys everything whose referencing frame has
        // provably completed (fence waited). Call right after the
        // vkWaitForFences(inFlight[currentFrame]) at each frame-begin path.
        void drainRetireQueue() {
            retireQueue_.drain(*ctx, frameSerial_, kFramesInFlight);
            reapRetiredRecords(false);
        }
        // Post-vkDeviceWaitIdle flush: whole device idle ⇒ destroy all queued
        // resources now, or they leak at device destroy. Call after EVERY
        // remaining vkDeviceWaitIdle that precedes resource destruction.
        void flushRetireQueue() {
            retireQueue_.flushAll(*ctx);
            reapRetiredRecords(true);
        }
        // On the slot path nothing rebuilds, so nothing runs the structural
        // rebuild's prune: a streamed world (terrain tiles, cells) frees a
        // geometry and its textures with every mesh it drops, and their
        // records and bindless texture slots stayed until the next rebuild,
        // which never came (a long GeoScene flight ran out of the 2048 texture
        // slots and drew its new tiles white). pruneFreedAssets, at the end of
        // each slot edit, retires the records of geometries that died and the
        // images of textures that died, with this frame's serial; the slots
        // go back to the free list at once (each frame-in-flight descriptor
        // set is rewritten before the retired image is destroyed, as
        // refreshDirtyMaterialTextures does). VulkanCoreSlots.cpp.
        void pruneFreedAssets();
        // Records pruneFreedAssets took out of blasCache, each with the serial
        // of the frame that retired it, destroyed by the retire queue's rule.
        std::vector<std::pair<uint64_t, std::unique_ptr<BlasRecord>>> retiredRecords_;
        void reapRetiredRecords(bool all);
        // A GrassMesh on the slot path (VulkanCoreSlots.cpp). Its state is the
        // field's own record plus its rest pose and height fractions: a field
        // the slot edit admits gets one built in the edit's batch (or, added
        // hidden, over the frames after) and primed by its first deform in the
        // frame; one whose mesh or geometry died, or whose mesh took another
        // geometry, is retired: the record with retiredRecords_, the two
        // input buffers through the retire queue.
        GrassMeshState* admitGrassState(GrassMesh& gm);
        void retireGrassState(GrassMeshState& st);
        static bool grassInputsValid(const BufferGeometry& g);
        // Retire (rather than inline-destroy) a particle geometry record's
        // buffers on an in-flight topology change. Mirrors destroyParticleGeomRec.
        void retireParticleGeomRec(ParticleGeomRec& rec) {
            for (auto& a : rec.anim) retireParticleAnimSet(a);
            retire(std::move(rec.uv));
            retire(std::move(rec.index));
        }
        void retireParticleAnimSet(vulkan::impl::ParticleAnimSet& a) {
            retire(std::move(a.position));
            retire(std::move(a.normal));
            retire(std::move(a.color));
        }

        // ImGui (or any post-render overlay) hook. When set, the swapchain
        // image is transitioned GENERAL → COLOR_ATTACHMENT_OPTIMAL after the
        // deferred-shaded frame, a dynamic render pass is opened with
        // LOAD_OP_LOAD, the callback draws into it, then we transition to
        // PRESENT_SRC.
        std::function<void(void*)> overlayCallback;

        // Per-frame lifecycle state, following the multi-pass / HUD pattern:
        // the first render() of a user animate iteration runs
        // beginFrame() (acquire + per-frame UBO uploads + cmd buffer record),
        // subsequent render() calls in the same iteration append to that
        // same cmd buffer, and the Canvas frame-end callback fires endFrame()
        // (ImGui overlay, swapchain → PRESENT_SRC, submit, vkQueuePresentKHR).
        //
        // Idle              : no frame in flight; next render() runs beginFrame.
        // RecordingPostShade: deferred shade pass recorded; further
        //                     render(ortho) calls append HUD overlay draws
        //                     to the same cb.
        // RecordingOrthoOnly: render(ortho) called with no prior deferred
        //                     shade — frame was opened with an empty result
        //                     (cleared swapchain). Rare; mostly defensive.
        enum class FrameState { Idle, RecordingPostShade, RecordingOrthoOnly };
        FrameState frameState_ = FrameState::Idle;

        // A swapchain readback (readRGBPixels) finished this canvas iteration's
        // frame early, to read the pixels render() had only recorded. A pass
        // that can only EXTEND an open frame (the HUD's ortho render, a
        // split-screen pane) has nothing to draw onto after that, and from
        // Idle it would open a frame of its own. render() drops it while this
        // is set. Set only inside the canvas animate body, cleared by the
        // frame-end callback: a caller driving render() directly has no
        // iteration to speak of, and its next render() is a new frame anyway.
        bool frameEndedByReadback_ = false;

        // setOrthographicSceneRendering: an ortho camera names a 3D VIEW, not a
        // 2D overlay, so a standalone render() with one takes the deferred path.
        // Off by default — every existing 2D/HUD user of this backend renders
        // through the ortho-only overlay path and must keep doing so.
        bool orthoSceneRendering_ = false;

        // (orthoFrame_ moved to ViewContext — under multi-view the projection
        //  kind is a property of the VIEW, not of the frame: a perspective
        //  primary and an orthographic secondary coexist in one submission.
        //  Distinct from orthoSceneRendering_, which is the user's standing
        //  permission rather than any view's projection.)

        // Does THIS render() call mean "shade the scene through a parallel
        // projection"? Only a standalone one (nothing in flight) can — a second
        // render() with an ortho camera over an open frame is the HUD pattern
        // and stays overlay-only whatever the flag says.
        [[nodiscard]] bool orthoSceneRender(const Camera& camera) const {
            return orthoSceneRendering_ && frameState_ == FrameState::Idle &&
                   camera.is<OrthographicCamera>();
        }

        // Internal ortho camera used for the screen-space sprite overlay
        // auto-call from beginDeferredFrame. Lazy-created on first use, then
        // its bounds + projection are rebuilt each frame to track the
        // current swapchain extent. Lives here (vs. stack-allocated each
        // frame) so its matrixWorldInverse / projectionMatrix don't get
        // rebuilt from scratch when the size hasn't changed.
        std::shared_ptr<OrthographicCamera> screenSpaceCam_;
        uint32_t   frameImageIndex_ = 0;

        // Scene-only swapchain capture (for event-camera and similar
        // sensors that need a pre-overlay readback). When enabled, the
        // renderer copies the post-TAA swapchain image into this host-
        // visible buffer BEFORE the sprite + ImGui overlays composite,
        // so sensor pipelines don't see their own visualisation.
        bool             sceneCaptureEnabled_ = false;
        vulkan::Buffer   sceneCaptureBuf_{};
        uint32_t         sceneCaptureBufW_ = 0;
        uint32_t         sceneCaptureBufH_ = 0;

        // ── Frame zero-copy interop (VulkanRenderer::enableFrameInterop) ──
        // One exported TRANSFER_DST buffer per armed (view, channel), filled by
        // a vkCmdCopyImageToBuffer recorded at the frame's record tail. Kept
        // here rather than on ViewContext because the invalidation rule is
        // "kill the whole view's set at once" and the record walk wants the
        // armed views, not all of them — a scene with no interop iterates an
        // empty vector.
        struct FrameInteropChannel {
            VulkanRenderer::FrameChannel channel{};
            vulkan::ExternalBuffer       buf{};
            uint32_t width = 0, height = 0, bpp = 0;
            bool     bgra = false;
        };
        struct FrameInteropView {
            uint32_t viewHandle = 0;// 0 = primary, else a secondary's handle
            std::vector<FrameInteropChannel> channels;
        };
        std::vector<FrameInteropView> frameInterops_;

        // Where one channel's pixels come from THIS frame: the image, the
        // aspect and the layout it rests in at the record tail. The table is
        // readViewGBufferAOVs' verbatim, plus the Color row (the swapchain for
        // the primary, the view's colorTarget for a secondary), and it is
        // resolved twice — once at enable to size the export, once per frame to
        // record the copy — so the two can never disagree about a format.
        struct FrameInteropSource {
            VkImage            image      = VK_NULL_HANDLE;
            VkImageAspectFlags aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
            VkImageLayout      restLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            uint32_t width = 0, height = 0, bpp = 0;
            bool    bgra = false;
        };
        bool frameInteropSource(uint32_t viewHandle, VulkanRenderer::FrameChannel channel,
                                uint32_t gbufSlot, uint32_t imageIndex,
                                FrameInteropSource& out);
        std::vector<VulkanRenderer::FrameInteropExport>
        enableFrameInterop(uint32_t viewHandle,
                           const std::vector<VulkanRenderer::FrameChannel>& channels);
        void disableFrameInterop(uint32_t viewHandle);
        [[nodiscard]] bool frameInteropActive(uint32_t viewHandle) const {
            for (const auto& s : frameInterops_)
                if (s.viewHandle == viewHandle) return true;
            return false;
        }
        // Invalidation (see the header): the source images are about to be
        // reallocated or destroyed, so the exports the foreign API imported are
        // torn down with a warning rather than left pointing at freed images.
        void invalidateFrameInterop(uint32_t viewHandle, const char* why);
        bool syncFrameInterop();
        // A pass of the post-view tail graph, after the scene capture: one copy
        // per armed channel. Adds nothing when no view is armed.
        void addFrameInteropPass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        struct FrameInteropCopy {
            VkImage image;
            VkImageAspectFlags aspect;
            VkImageLayout restLayout;
            VkBuffer dst;
            uint32_t width, height;
        };
        std::vector<FrameInteropCopy> frameInteropCopies_;// this frame's, built by addFrameInteropPass
        void destroyFrameInterops();

        // Deferred-apply queue for setters that issue vkDeviceWaitIdle +
        // realloc descriptor/image resources. These collide with an open
        // cmd buffer (frameState_ != Idle), so when called mid-frame we
        // stash the request and apply it at the next beginDeferredFrame —
        // after the prior frame's fence wait, before any record. Setters
        // called when Idle apply immediately as before.
        bool pendingRenderScaleRealloc_ = false;
        bool pendingAccumulationReset_  = false;
        int  pendingProbeCascades_      = -1;// setProbeCascades asked mid-frame; -1 = nothing pending
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
        // Raster G-buffer MSAA sample count (1/2/4). 1 = today's single-sample
        // path, byte-identical output, default. See setGbufferMsaa. Reallocation
        // (render pass + pipelines + MS images) is render-extent-resource work,
        // so it reuses the exact same idle-gate as renderScale.
        uint32_t gbufMsaaSamples_        = 1;

        explicit Impl(Canvas& c);

        // Print a VMA memory-usage summary to stderr: the allocator's reserved-vs-
        // live block totals and per-heap usage/budget. Cheap; for manual/debug use
        // (there is otherwise no memory introspection in the renderer). Reflects
        // all VMA allocations — G-buffer, history, denoiser scratch, upscalers,
        // and any scene AS/geometry live at the call site.
        void dumpMemoryStats(const char* tag = "") const;
        // Prints the one "deforming geometry landed in system memory" line
        // when noteDeformPlacement counted any bytes since the last call.
        void reportDeformPlacement();

        ~Impl();

        void createCommandResources();

        // (Re)create the per-swapchain-image present-wait semaphores. Called
        // from createCommandResources and again on every swapchain recreation
        // (the image count can change and retired semaphores may be left in
        // an indeterminate signalled state). Requires an idle device.
        void createRenderFinishedSemaphores();

        // The frame's swapchain image. Normally a straight
        // vkAcquireNextImageKHR; when presents are suppressed (headless canvas)
        // the slot's image is acquired once and reused from then on, because a
        // never-presented image is never released back to the presentation
        // engine and so can never be re-acquired. Sets pinnedSwapImage_ /
        // acquireSemPending_ and returns the acquire's VkResult (VK_SUCCESS on
        // the reuse path) for the caller's OUT_OF_DATE / SUBOPTIMAL handling.
        VkResult acquireOrReuseSwapchainImage(uint32_t& imageIndex);

        // Allocate, begin, return a one-shot command buffer.
        VkCommandBuffer beginOneShot();

        // The optional `label` is folded into the error message when the
        // submit / wait throws — handy for distinguishing which one-shot site
        // hit a device-lost during runtime debugging.
        void endAndSubmitOneShot(VkCommandBuffer cb, const char* label = "one-shot");

        // ── Batched one-shots ────────────────────────────────────────────────
        // A structural scene rebuild admits every NEW tile geometry (BLAS build
        // + prevVertex seed) and its albedo DataTexture (staging copy + mip
        // blit) through beginOneShot/endAndSubmitOneShot — one vkQueueSubmit +
        // vkQueueWaitIdle EACH. A burst of tiles then serialises a dozen queue
        // drains into one frame (measured ~37 ms on fjord flight). While a batch
        // is open, beginOneShot hands back a single shared command buffer and
        // endAndSubmitOneShot only closes the caller's recording (no submit);
        // flushOneShotBatch submits the whole batch once and waits. Transient
        // resources (BLAS scratch, image staging) that the callers would free
        // right after their submit are parked in oneShotBatchGarbage_ and
        // reclaimed after the single wait. Same one-shot semantics from the
        // caller's view (work is done + resources freed by the time the batch is
        // flushed), one drain instead of N. Callers must be pure transfer/AS
        // builds with no read-back before the flush — the tile admit path is.
        bool            oneShotBatch_ = false;
        VkCommandBuffer oneShotBatchCb_ = VK_NULL_HANDLE;
        std::vector<Buffer> oneShotBatchGarbage_;

        void beginOneShotBatch() { oneShotBatch_ = true; }
        void flushOneShotBatch();
        // The batch submitted WITHOUT a host wait, for work the frame about
        // to be recorded consumes (the slot path's BLAS builds and texture
        // uploads): the frame's command buffer is submitted after it on the
        // same queue, and the render graph's entry barrier (ALL_COMMANDS,
        // every prior submission in its first scope) orders and publishes it.
        // Its transients go to the frame-serial retire queue, its command
        // buffer to asyncOneShots_ until its fence signals. Its static BLASes
        // are compacted after that (compactAsync): their compacted-size
        // queries ride the batch, in a pool of its own.
        void flushOneShotBatchAsync();
        // Free the command buffers and fences of async batches that finished
        // (all of them when the device is known idle), and compact what a
        // finished batch queried.
        void reapAsyncOneShots(bool deviceIdle);
        struct AsyncCompaction {
            VkQueryPool pool = VK_NULL_HANDLE;
            // Each record by its geometry (blasCache's key) and the structure
            // queried: a record evicted or rebuilt since is passed over.
            std::vector<std::pair<const BufferGeometry*, VkAccelerationStructureKHR>> records;
        };
        struct AsyncOneShot {
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            AsyncCompaction compaction;
        };
        std::vector<AsyncOneShot> asyncOneShots_;
        // A finished async batch's queried BLASes: each copied with
        // COPY_MODE_COMPACT into storage of its compacted size, in one more
        // async batch (ahead of the frame in queue order, as the build was),
        // the record swapped to the copy at once and the build-size pair
        // retired with the frame about to be recorded, whose TLAS is then
        // built anew (blasRereferenced_): the frames still in flight trace
        // the old pair, this one the new.
        void compactAsync(const AsyncCompaction& c);
        bool blasRereferenced_ = false;
        // A slot edit that needs more slots than the pool holds: every
        // pool-sized GPU buffer (the geometry and material desc rings, the
        // motion matrices and moved bits, the TLAS and its instance buffers)
        // is made anew for `capacity` entries and refilled from its host copy;
        // the old ones are retired with this frame (the frames in flight keep
        // reading them), and each frame-in-flight's sets that name them are
        // rewritten after its own fence. No device wait; the entries, spans
        // and every temporal history stay. VulkanCoreSlots.cpp.
        void growSlotPool(uint32_t capacity);

        // ── Static BLAS compaction ───────────────────────────────────────────
        // A static record (built once, traced for its lifetime) is built
        // PREFER_FAST_TRACE | ALLOW_COMPACTION, its compacted size is queried
        // in the build's own command buffer, and it is then copied with
        // COPY_MODE_COMPACT into storage of exactly that size; the build-size
        // pair is freed. Deformer and interop records are refit in place and
        // keep the updatable lineage, uncompacted.
        //
        // THREEPP_VK_BLAS_COMPACT=0, read once at construction, turns the
        // compaction off (static builds then stay at their build size).
        // blasCompact_ is also false when the device does not resolve the two
        // compaction entry points.
        bool blasCompact_ = false;
        // Compacted-size queries, kBlasCompactPoolQueries per pool; pools are
        // appended as a larger batch needs them and live until teardown.
        // Slot k of one compaction round is query k % N of pool k / N.
        static constexpr uint32_t kBlasCompactPoolQueries = 256;
        std::vector<VkQueryPool> blasCompactPools_;
        // Batch mode: static records built into the open one-shot batch,
        // compacted by flushOneShotBatch after its wait. destroyBlasRecord
        // takes a record off this list.
        std::vector<BlasRecord*> blasCompactPending_;
        // A batch-built record's uncompacted address was already pushed into
        // ensureSceneBuilt's TLAS instance array when the batch flushes, so the
        // flush records old -> new here and keeps the old pair alive (retired),
        // which also keeps a later build in the same pass from being handed
        // the old address. applyBlasCompactionRemap rewrites the instances and
        // frees the retired pairs.
        std::unordered_map<VkDeviceAddress, VkDeviceAddress> blasAddressRemap_;
        struct RetiredBlas {
            VkAccelerationStructureKHR as = VK_NULL_HANDLE;
            Buffer storage;
        };
        std::vector<RetiredBlas> blasCompactRetired_;

        // The flags a static record, and an auto-LOD level, is built with.
        VkBuildAccelerationStructureFlagsKHR staticBlasBuildFlags() const {
            VkBuildAccelerationStructureFlagsKHR f = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            if (blasCompact_) f |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            return f;
        }
        // Records, after builds already recorded into `cb`, one barrier and the
        // compacted-size query of ases[k] into slot k.
        void recordCompactedSizeQueries(VkCommandBuffer cb, const std::vector<VkAccelerationStructureKHR>& ases);
        // One compaction round over structures whose sizes were queried into
        // slots 0..n-1 by recordCompactedSizeQueries and whose build has
        // completed: creates each compacted pair, copies all of them in ONE
        // one-shot (one wait), then swaps the handles in. `retire` keeps the
        // replaced pairs alive and records address remaps (batch mode);
        // otherwise they are destroyed at once. A job whose compacted size is
        // not smaller than its storage is left as it is. Returns the bytes
        // freed per job (0 for a job left as it was).
        struct BlasCompactJob {
            VkAccelerationStructureKHR* as = nullptr;
            Buffer* storage = nullptr;
            VkDeviceAddress* address = nullptr;
            bool* compacted = nullptr;// may be null (auto-LOD levels)
        };
        std::vector<VkDeviceSize> compactBlases(const std::vector<BlasCompactJob>& jobs, bool retire);
        // Rewrites TLAS instance references built from batch-compacted records
        // and frees the replaced pairs. Called right before buildTlas.
        void applyBlasCompactionRemap(std::vector<VkAccelerationStructureInstanceKHR>& instances);
        void destroyBlasCompactionResources();
        // Free `buf` now, or defer it to the next flush when a batch is open.
        void destroyBufferMaybeBatched(Buffer& buf) {
            if (oneShotBatch_) {
                oneShotBatchGarbage_.push_back(buf);
                buf = {};// caller's handle retired; the garbage copy owns it now
            } else {
                destroyBuffer(ctx->allocator(), buf);
            }
        }

        // ── Static geometry placement ────────────────────────────────────────
        // A static record's vertex / normal / uv / color / index / rtIndex
        // buffers (and its auto-LOD level index buffers) are device-local and
        // filled through a staging copy: allocated host-visible, they took the
        // 256 MiB BAR window first and spilled the rest to system memory on a
        // desktop without resizable BAR, and every G-buffer vertex pull, ray
        // hit fetch and BLAS input read then crossed PCIe each frame.
        // Deformer and interop records keep host-visible vertex and normal
        // buffers (rewritten per frame); their uv, color, index and rtIndex
        // are device-local too. Every host write into a geometry buffer goes
        // through stageUpload, which memcpys into a host-mappable allocation
        // as before and queues the bytes for a staging copy otherwise, so no
        // writer depends on where the buffer landed.
        struct StagedUploads {
            struct Copy {
                VkBuffer     dst       = VK_NULL_HANDLE;
                VkDeviceSize dstOffset = 0;
                VkDeviceSize srcOffset = 0;
                VkDeviceSize bytes     = 0;
            };
            std::vector<uint8_t> data;
            std::vector<Copy>    copies;
        };
        bool hostMappable(const Buffer& b) const;
        // deviceLocal: AUTO_PREFER_DEVICE, no host access (TRANSFER_DST is
        // added for the staging copy); otherwise AUTO + sequential host writes.
        Buffer createGeomBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, bool deviceLocal);
        // Writes `bytes` from `src` at `dstOffset` of `dst`: at once when dst
        // is host-mappable, else queued in `up` for recordStagedUploads.
        void stageUpload(StagedUploads& up, const Buffer& dst, const void* src,
                         VkDeviceSize bytes, VkDeviceSize dstOffset = 0);
        // stageUpload without the host-write shortcut: always queued for the
        // staging copy. For writes recorded into a FRAME command buffer, where
        // an earlier frame still in flight may read `dst` (a host-mappable
        // device-local buffer in the BAR is no exception): the copy is ordered
        // behind those reads on the queue, a host memcpy is not.
        void stageCopy(StagedUploads& up, const Buffer& dst, const void* src,
                       VkDeviceSize bytes, VkDeviceSize dstOffset = 0);
        // Queues the uv, color and index buffers of `rec` whose attribute
        // version moved since rec.attrVersions, whole, in the record's own
        // (possibly packed) formats: through stageCopy when `viaCopy` (frame
        // command buffer), stageUpload otherwise. Returns true when the index
        // was among them, which rules out a BLAS refit. Stamps nothing.
        bool stageChangedSideAttributes(StagedUploads& up, const BlasRecord& rec,
                                        const BufferGeometry& geom, bool viaCopy);
        // Records `up`'s queued copies into `cb` from one staging buffer, then
        // a barrier that makes them visible to every later command (this cb's
        // BLAS build and prevVertex seed, and later frames). `dstUntouched`:
        // the destinations are fresh, so no barrier is needed before the
        // copies. Returns the staging buffer (empty when nothing was queued);
        // the caller frees it once cb has executed (destroyBufferMaybeBatched
        // right after endAndSubmitOneShot). Clears `up`.
        Buffer recordStagedUploads(StagedUploads& up, VkCommandBuffer cb, bool dstUntouched);
        // recordStagedUploads in two halves, the first for any thread:
        // makeStaging copies up's bytes into a new host-visible staging buffer
        // (clears up.data, keeps up.copies; the allocator and nothing of the
        // renderer's state), recordStagedCopies records the copies and their
        // barriers from it into cb (clears up.copies).
        Buffer makeStaging(StagedUploads& up);
        void recordStagedCopies(StagedUploads& up, const Buffer& staging, VkCommandBuffer cb, bool dstUntouched);
        // The one-shot form: begin, record, submit (or join the open batch),
        // release the staging buffer.
        void submitStagedUploads(StagedUploads& up, const char* label);

        // Who rewrites an updatable record's vertex and normal after its
        // build. Host: a per-frame host path writes them (morph blend, the
        // displaced ocean's existing placement), so they stay host-visible.
        // Gpu: only compute shaders write them (skinning, tet skinning, grass
        // wind), so they are device-local like a static record's: a
        // host-visible deformer buffer that spills out of the 256 MiB BAR
        // window lands in system memory and its deform pass crosses PCIe
        // every frame.
        enum class VertexWriter : uint8_t { Host, Gpu };
        // Adds `b`'s bytes to deformHostBytes_ when it landed outside device
        // memory (the dumpMemoryStats "host" bucket). Called once per deformer
        // record or input buffer at creation; ensureSceneBuilt prints one line
        // per structural build when the sum is non-zero, then clears it.
        void noteDeformPlacement(const Buffer& b);
        VkDeviceSize deformHostBytes_ = 0;

        // The composite version of the tracked attributes (GeomAttrRefs).
        static unsigned int geomVersionOf(const BufferGeometry& g) {
            return GeomAttrRefs::of(g).versionSum();
        }

        // The parts of geomVersionOf, kept apart.
        static AttrVersions attrVersionsOf(const BufferGeometry& g) {
            return GeomAttrRefs::of(g).versions();
        }

        // Stamp both version snapshots after the record's buffers were filled
        // from `g` (a build or a refresh).
        static void stampVersions(BlasRecord& rec, const BufferGeometry& g) {
            rec.geomVersion = geomVersionOf(g);
            rec.attrVersions = attrVersionsOf(g);
        }

        // An in-place refresh writes uv and color in the format the record
        // was BUILT with. A packed format was picked because every value then
        // fit it (uv in [0,1] for unorm16, LDR color for unorm8); an edit that
        // leaves that range would be clamped. False when a CHANGED uv or
        // color attribute no longer fits, so the caller re-admits the record
        // through the structural rebuild, which picks the format afresh.
        static bool refreshFitsPacking(const BlasRecord& rec, const BufferGeometry& g);

        // Build a single BLAS for the given geometry. Vertex / index buffers
        // are uploaded host-mapped, then the AS is built into freshly allocated
        // device storage. The temporary scratch buffer is destroyed on exit.
        // allowPacked: pack normal/uv/color into narrow device formats
        // (BlasRecord::packedMask). Static geometry only — the deforming
        // callers (skinned / tet / displaced / grass / morphed) keep the
        // default false because their per-frame compute rewrites assume
        // tightly-packed float layouts.
        // `updatable`: the record is refit in place later (deformers), so it is
        // built ALLOW_UPDATE and never compacted. Static records pass false and
        // are compacted (see staticBlasBuildFlags); an interop-marked geometry
        // is always updatable.
        // `rtIndices`: when given, the AS is built over these uint32 indices
        // instead of the geometry's own (BlasRecord::rtIndex); the record's
        // `index` still carries the geometry's for the raster.
        // `writer`: for an updatable record, VertexWriter::Gpu places vertex
        // and normal device-local (see VertexWriter); ignored for static and
        // interop records.
        // traced = false builds the record as it is built without ray query:
        // its buffers, no acceleration structure. For a geometry that only
        // overlay meshes draw (a wireframe, an overlay-layer panel): no ray
        // meets an overlay, it is never a TLAS instance, and its structure was
        // build time at admission, a refit per edit and device memory for
        // nothing (ensureSceneBuilt decides per geometry, see tracedGeoms).
        std::unique_ptr<BlasRecord> buildBlasFor(const BufferGeometry& geom, bool allowPacked, bool updatable,
                                                 const std::vector<uint32_t>* rtIndices = nullptr,
                                                 VertexWriter writer = VertexWriter::Host,
                                                 bool traced = true);
        // buildBlasFor in two halves. prepareBlas validates the geometry, makes
        // the record's buffers and queues their bytes in p.up: any thread (the
        // geometry's arrays and the allocator; the caller reads interopTarget
        // from forceUnpackedGeoms_). finishBlas records the copies (from
        // p.staging when the preparing thread made it) and the acceleration
        // structure's build into the open one-shot or batch: the render thread.
        // discardBlasPrep frees a prepared record nothing has recorded yet.
        struct BlasPrep {
            std::unique_ptr<BlasRecord> rec;
            StagedUploads up;
            Buffer staging;
            VkDeviceSize vbBytes = 0;
            uint32_t vertexCount = 0, primitiveCount = 0, indexCount = 0;
            bool indexed = false, packIdx = false, rt = false, updatable = false, interopTarget = false;
            // With ray query: the structure, created, and what its build needs.
            VkAccelerationStructureGeometryKHR blasGeom{};
            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            Buffer scratch;
            uint32_t buildPrims = 0;
            bool compactThis = false;
        };
        // `disp`: positions displaced by a material displacementMap (see
        // DisplacementSpec), read by the caller like interopTarget.
        bool prepareBlas(const BufferGeometry& geom, bool allowPacked, bool updatable,
                         const std::vector<uint32_t>* rtIndices, VertexWriter writer, bool traced,
                         bool interopTarget, BlasPrep& p, const DisplacementSpec* disp = nullptr);
        // geom's positions moved along its normals by d's map: false (and a
        // one-time warning) when the geometry or the image can't be read.
        static bool displacePositions(const BufferGeometry& geom, const DisplacementSpec& d,
                                      std::vector<float>& out);
        std::unique_ptr<BlasRecord> finishBlas(const BufferGeometry& geom, BlasPrep& p);
        void discardBlasPrep(BlasPrep& p);
        // A hidden GrassMesh field's state prepared on a worker thread from its
        // slot edit on (beginGrassPrep, from beginHiddenAdmission): its record's
        // buffers and its rest-pose and height-fraction inputs, every byte in
        // one filled staging buffer. ensureGrassState takes it (waiting for it
        // only when the field is shown before it is ready) and records just the
        // copies and the build; the pump admits a field once it is ready.
        struct GrassPrepData {
            BlasPrep blas;
            Buffer restPos, heightFrac;
            uint32_t vertexCount = 0;
            float bladeHeight = 0.f;// estimated, field-local m (GrassMeshState::bladeHeight)
            bool ok = false;
        };
        struct GrassPrep {
            std::shared_ptr<BufferGeometry> geom;// alive while the worker reads it
            vulkan::impl::MeshOwner owner;
            std::future<std::unique_ptr<GrassPrepData>> job;
        };
        std::unordered_map<const GrassMesh*, GrassPrep> grassPreps_;
        std::unique_ptr<GrassPrepData> prepareGrassData(const BufferGeometry& geom, bool interopTarget);
        void beginGrassPrep(GrassMesh& gm);
        [[nodiscard]] bool grassPrepPending(const GrassMesh& gm) const;
        void discardGrassPrep(GrassPrepData& d);
        // Drops preps whose field died or changed geometry (all: every one,
        // at shutdown); waits for a worker still on one.
        void sweepGrassPreps(bool all);

        // Allocate or look up the per-SkinnedMesh BLAS state. Builds the BLAS
        // once with the current pose; subsequent dirty frames go through
        // refreshSkinnedBlas which only re-skins + rebuilds in-place. Returns
        // null if the geometry is unsupported (no position/normal/skin attrs).
        SkinnedMeshState* ensureSkinnedBlas(SkinnedMesh& sm);

        // Re-skin the SkinnedMesh's vertices/normals on the CPU, copy them
        // into the host-mapped vertex/normal buffers, and rebuild the BLAS in
        // place (same AS handle and storage so the device address — and the
        // TLAS reference to it — stay valid). The TLAS doesn't need refit
        // for pose-only changes; the instance's transform is unchanged.
        void refreshSkinnedBlas(SkinnedMesh& sm, SkinnedMeshState& st);

        // Allocate or look up the per-tet-skinned-mesh BLAS state (PhysX soft body).
        // Mirrors ensureSkinnedBlas: build the BLAS once at rest pose, upload the
        // static tet bindings + rest normals + baked Dr^-1 columns, allocate the
        // per-frame tet-position buffer + descriptor set, then queue a first refresh.
        // Returns null if the geometry/material lack the tet attributes that
        // SoftBody::enableGpuSkinning() sets.
        TetMeshState* ensureTetBlas(Mesh& m);

        // Per-frame: upload the soft body's current collision-tet positions into the
        // tet-position buffer (read from the material's tet texture image, which
        // PhysxWorld::syncSoftBodies refreshes each frame), then queue the GPU skin +
        // BLAS rebuild — recorded in recordCommandBuffer next to the skinned path.
        void refreshTetBlas(Mesh& m, TetMeshState& st);

        // Rewrite binding 6 (tetPos — see tet_skinning.comp) of every ring slot's
        // descriptor set to point at `buf` (the single exported interop buffer).
        // Used by the interop enable swap; the other 8 bindings are untouched.
        // The caller vkDeviceWaitIdles first, so no set is in a pending cb.
        void rewriteTetPosBinding(TetMeshState& st, VkBuffer buf);

        // Zero-copy interop enable: swap the mesh's tet-position buffer for an
        // EXPORTED dedicated device allocation and register the CUDA copy that
        // fills it each frame. Once-per-body; drains the device (the in-flight
        // frames' dispatches read the buffer being replaced, and this is a
        // registration-time call, not a per-frame one).
        VulkanRenderer::SoftBodyInteropHandle
        enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy) {
            auto it = tetMeshStates.find(&mesh);
            if (it == tetMeshStates.end() || !ctx->externalMemorySupported()) return {};
            auto& st = *it->second;
            if (st.tetPosExt.handle != VK_NULL_HANDLE) {// already enabled — same allocation
                st.tetPosExternalCopy = std::move(deviceCopy);
                return {vulkan::takeOsHandle(ctx->device(), st.tetPosExt),
                        static_cast<size_t>(st.tetPosExt.size)};
            }
            if (st.tetPosBytes == 0 || st.tetDescSet[0] == VK_NULL_HANDLE) return {};
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (softbody interop enable)");
            st.tetPosExt = vulkan::createExternalBuffer(
                    ctx->physicalDevice(), ctx->device(), st.tetPosBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            rewriteTetPosBinding(st, st.tetPosExt.handle);
            for (auto& slot : st.tetPos)// CPU-path ring no longer read
                destroyBuffer(ctx->allocator(), slot);
            st.tetPosExternalCopy = std::move(deviceCopy);
            // takeOsHandle, not .osHandle: on POSIX the fd's ownership passes to
            // the importer, so the record must stop believing it owns it.
            return {vulkan::takeOsHandle(ctx->device(), st.tetPosExt),
                    static_cast<size_t>(st.tetPosExt.size)};
        }

        // Interop disable / CUDA-import-failure fallback: restore the host-visible
        // VMA buffer and the CPU upload path. The caller must have stopped (or
        // never started) the CUDA writes into the exported memory.
        void disableSoftBodyInterop(const Mesh& mesh);

        // F6: the same idea one abstraction up — a whole FIELD's positions,
        // exported once and filled device-to-device every frame. Thin: the pass
        // owns the state, the gate and the fallback (ParticleFieldPass::
        // enableInterop); this converts its record to the public handle type.
        // Valid BEFORE the first frame: the pass (and, inside it, the field's
        // device state and its exported allocation) is brought up on demand, so
        // an application arms interop at setup and no frame ever runs with an
        // Interop field whose copy is not registered.
        VulkanRenderer::ParticleFieldInteropHandle
        enableParticleFieldInterop(ParticleField& field, std::function<void()> deviceCopy) {
            ensureParticleFieldPass();
            const auto e = particleFieldPass_->enableInterop(field, std::move(deviceCopy));
            return {e.osHandle, e.sizeBytes, e.attrHandle, e.attrSizeBytes};
        }

        // Zero-copy MESH VERTEX interop — the same idea again, now on a plain
        // mesh's position/normal attributes. Definitions in VulkanCoreGeometry.cpp
        // beside the dynamic-refit machinery they hook into; the design notes
        // live on BlasRecord::posExt (why a copy and not a buffer swap) and on
        // VulkanRenderer::enableVertexInterop (the caller-facing contract).
        VulkanRenderer::VertexInteropHandle
        enableVertexInterop(const Mesh& mesh, std::function<void()> deviceCopy, bool validate,
                            bool stableCorrespondence);
        void disableVertexInterop(const Mesh& mesh);
        void setStableCorrespondence(const Mesh& mesh, bool stable);
        void setEmissiveCastsLight(const Object3D& obj, bool castsLight);
        // The record backing `mesh` for interop purposes, or null. Interop is a
        // plain-mesh feature: skinned / tet / displaced / grass / morphed meshes
        // already have a per-frame GPU producer of their own that would overwrite
        // whatever the foreign API wrote, so they are refused rather than
        // silently fighting over the buffer.
        BlasRecord* interopRecordFor(const Mesh& mesh);

        // Definition moved to VulkanGeometryState.hpp.
        using GeomRefreshOp = vulkan::impl::GeomRefreshOp;

        // Batched per-frame BLAS refresh. For each op:
        //   (1) snapshot rec.vertex → rec.prevVertex (chit motion-vector channel)
        //   (2) host memcpy of new positions/normals/uv/index into rec.vertex etc.
        //   (3) refit rec.as in place (MODE_UPDATE, periodic MODE_BUILD every
        //       kBlasFullRebuildInterval frames to keep the BVH balanced)
        //
        // Phase (1) and (3) record into one shared command buffer each, so the
        // whole batch costs 2 vkQueueSubmit + 2 vkQueueWaitIdle pairs regardless
        // of N. Previously, looping refreshGeomBlas paid that pair per geometry
        // — at ~30 µs queue overhead per submit, 5 soft bodies = 300 µs/frame
        // wasted on submit round-trips alone.
        //
        // Caller must (a) gate on equal vertex/index counts (topology must not
        // change — fall through to fullRebuild if it does) and (b) have already
        // drained in-flight GPU work; phase (2)'s host memcpys race in-flight
        // closest_hit reads of rec.vertex otherwise.
        void refreshGeomBlasBatch(const std::vector<GeomRefreshOp>& ops);

        // Frame-cb twin of refreshGeomBlasBatch, for every in-place edit of a
        // record with no auto-LOD chain: graduated records
        // (BlasRecord::perFrameDynamic) CPU-pack new positions/normals into
        // this frame's slot of their staging ring, the others (an occasional
        // edit) into one staging buffer retired with the frame, honouring
        // updateRange. Then records vertex→prevVertex snapshot, staging→vertex/
        // normal copies and the batched BLAS refit into `cb` with barriers:
        // zero extra submits, zero waits. Also records the one-frame
        // prevVertex re-sync for pendingDynamicPrevResyncs_.
        // Consumes (clears) both pending lists.
        void recordDynamicGeomRefits(VkCommandBuffer cb);

        // Size of `rec`'s structure in the updatable lineage (PREFER_FAST_TRACE
        // | ALLOW_UPDATE) over `geom`'s live span.
        // An occasional edit's positions and normals (the updateRange part) as
        // copies in `up`, in the record's formats; consumes both ranges.
        void stageEditedVertexRange(StagedUploads& up, const BlasRecord& rec,
                                    const FloatBufferAttribute& posAttr,
                                    const FloatBufferAttribute& nrmAttr);
        VkDeviceSize updatableBlasSize(const BlasRecord& rec, const BufferGeometry& geom) const;
        // A static record turning dynamic lives in storage sized for its static
        // build or its compacted copy: too small for the updatable lineage.
        // Moves it to new storage of `asSize` (no-op, false, when it already
        // fits) and flags the TLAS instance refill (lodChangedThisFrame_). The
        // old pair is retired (`retireOld`: an earlier frame still in flight
        // may trace it) or freed at once (the caller drained the device). The
        // new structure is empty: the caller's next build must be MODE_BUILD.
        bool moveToUpdatableBlasStorage(BlasRecord& rec, VkDeviceSize asSize, bool retireOld);

        // ── Morph-target helpers ─────────────────────────────────────────

        static bool isMorphedMesh(const Mesh& m) {
            return m.geometry()->getMorphAttributes().count("position") > 0;
        }

        static void cpuMorphBlend(Mesh& mesh,
                                  std::vector<float>& outPos,
                                  std::vector<float>& outNorm);
        // The blend itself, for any object's influences; outNorm == nullptr
        // skips the normals (Points).
        static void cpuMorphBlend(const BufferGeometry& geom,
                                  const std::vector<float>& influences,
                                  std::vector<float>& outPos,
                                  std::vector<float>* outNorm);

        MorphedMeshState* ensureMorphedBlas(Mesh& mesh) {
            auto it = morphedMeshStates.find(&mesh);
            if (it != morphedMeshStates.end()) return it->second.get();

            auto rec = buildBlasFor(*mesh.geometry(), /*allowPacked=*/false, /*updatable=*/true);
            if (!rec) return nullptr;
            rec->liveCheck = mesh.geometry();

            auto state = std::make_unique<MorphedMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = mesh.geometry();
            state->owner.bind(mesh);

            auto* raw = state.get();
            morphedMeshStates.emplace(&mesh, std::move(state));

            refreshMorphedBlas(mesh, *raw);
            return raw;
        }

        void refreshMorphedBlas(Mesh& mesh, MorphedMeshState& st);

        // ── DisplacedMesh helpers ────────────────────────────────────────
        // Lazy create + initialize the per-DisplacedMesh state. Builds the
        // BLAS from the rest geometry (will get overwritten by the displace
        // compute pass before the first ray-trace) and stands up the FFT
        // cascade. Returns nullptr if the geometry is unsupported (must be a
        // square indexed plane with N×N vertices for some power-of-two N).
        DisplacedMeshState* ensureDisplacedState(DisplacedMesh& dm);

        // Per-frame: run the FFT chain → water_displace → BLAS rebuild for
        // one DisplacedMesh. Mirrors refreshSkinnedBlas's structure.
        // Record the full per-frame water update — FFT chain → water_displace →
        // world foam → in-place BLAS rebuild — into `cb`. NO submit and NO CPU
        // wait: the caller either batches this into the frame command buffer
        // (recordCommandBuffer draining pendingDisplacedDeforms_, the per-frame
        // path) or wraps it in a one-shot for the rare structural first build
        // (refreshDisplacedBlas below).
        // timed: write the TP_Ocean* timestamp brackets. Only the per-frame
        // path may pass true, and only for the FIRST displaced mesh of the
        // frame — the query pool has one slot pair per pass, and the one-shot
        // first-build cb never reset those slots at all.
        void recordDisplacedDeform(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds, bool timed = false);
        // Its two halves, which recordDisplacedDeform runs back to back: the
        // update (FFT, displace, foam, wake, readback copies) and the BLAS
        // rebuild over the displaced vertices. The async-compute pilot records
        // them into different command buffers on different queues.
        void recordDisplacedUpdate(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds, bool timed);
        void recordDisplacedBlasRebuild(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, bool timed);

        // Mirror cascade height fields into DisplacedMesh for CPU sampling
        // (boat hydrodynamics etc.). On the batched per-frame path this reads
        // what the last COMPLETED frame's copy wrote — one/two frames of
        // latency, which wave sampling tolerates (same class as the skinned
        // path's single-buffered bone uploads).
        void mirrorDisplacedHeightfields(DisplacedMesh& dm, DisplacedMeshState& st);

        // Synchronous variant for the structural (re)build path: the TLAS /
        // geometry descriptors built right after need the displaced result in
        // place. Rare (scene restructure), so the one-shot block is fine here.
        void refreshDisplacedBlas(DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds);

        // ── GrassMesh helpers ────────────────────────────────────────────
        // Conservative world-space AABB for a wind-swayed grass field, shared by
        // the frustum cull, the occlusion-cull meta build, and the distance-gated
        // freeze so all three test the SAME box. The CPU-side "position" attribute
        // IS the rest pose (the GPU displaces a private copy into the BLAS), and
        // grass_wind.comp bends each vertex by windDir·(gust·windStrength·hf²) with
        // |gust| ≤ 0.85 and hf² ≤ 1, so the swayed vertex never leaves the rest
        // AABB dilated by windStrength (windDir is ~unit — |component| ≤ 1). That
        // makes the dilation a provably conservative bound, which is the whole
        // reason grass can be culled at all. A rotor downwash jet that reaches
        // the field adds its lean: at most the blades' height sideways and
        // never below the base (downwash.glsl, dwGrassOffset), so the box grows
        // by that height. Returns false (→ caller treats the
        // field as always-visible / always-animate) if the geometry has no bounds.
        // en.mesh is known to be a GrassMesh (cached en.isGrass flag), so the
        // static_cast avoids the per-entry dynamic_cast the hot loops must not pay.
        bool grassSwayWorldAabb(const MeshEntry& en, Box3& out) const {
            auto* gm = static_cast<GrassMesh*>(en.mesh);
            auto geom = gm->geometry();
            if (!geom) return false;
            if (!geom->boundingBox) geom->computeBoundingBox();
            if (!geom->boundingBox) return false;
            out = *geom->boundingBox;
            Matrix4 w;
            std::memcpy(w.elements.data(), en.worldMatrix.data(), 64);
            out.applyMatrix4(w);
            float grow = gm->params.windStrength;
            if (downwash_ && !downwash_->empty() && downwashReachesBox(out)) {
                float h = gm->params.bladeHeight;
                if (!(h > 0.f)) {
                    const auto it = grassStates.find(gm);
                    h = it != grassStates.end() ? it->second->bladeHeight : 0.f;
                }
                grow += h * w.getMaxScaleOnAxis();
            }
            out.expandByScalar(grow);
            return true;
        }

        // Distance-gated wind/refit freeze decision (see GrassMesh::Params::
        // maxAnimDistance). 0 ⇒ always animate. Otherwise animate only while the
        // camera is within maxAnimDistance of the field's sway-dilated AABB;
        // beyond that the field is frozen (skips the per-frame dispatch + refit,
        // holds its last pose). Distance is measured to the nearest point of the
        // box so a large field animates whenever the camera is anywhere near it.
        // Whether a field with a state deforms this frame. Hidden: no (no ray
        // hits it, no raster draws it), except an admission's prime, which
        // gives its BLAS a pose before any show; the skip is remembered, and
        // the first frame shown again is a prime whatever the distance gate
        // says (GrassMeshState::skippedHidden).
        static bool grassWantsDeform(GrassMeshState& st, bool hidden, bool animate) {
            if (hidden) {
                if (st.primePending) return true;
                st.skippedHidden = true;
                return false;
            }
            if (st.skippedHidden) {
                st.skippedHidden = false;
                st.primePending = true;
            }
            return st.primePending || animate;
        }

        bool grassShouldAnimate(const MeshEntry& en, const Vector3& camPos) const {
            auto* gm = static_cast<GrassMesh*>(en.mesh);
            const float maxD = gm->params.maxAnimDistance;
            if (maxD <= 0.f) return true;
            Box3 box;
            if (!grassSwayWorldAabb(en, box)) return true;// no bounds → animate
            if (box.distanceToPoint(camPos) <= maxD) return true;
            // A downwash jet near the field keeps it awake, and so does the
            // bend the last one left (one deform to stand back up).
            if (downwash_ && !downwash_->empty() && downwashReachesBox(box)) return true;
            const auto it = grassStates.find(gm);
            return it != grassStates.end() && it->second->downwashBent;
        }

        // Lazy create the per-GrassMesh state: a BLAS over the rest geometry
        // (built ALLOW_UPDATE, refit each frame) plus an immutable copy of the
        // rest positions and the per-vertex height fractions the wind shader
        // reads. Returns nullptr if the geometry lacks position / heightFrac.
        GrassMeshState* ensureGrassState(GrassMesh& gm);

        // Record the grass wind deform (rest → displaced positions in the BLAS
        // vertex buffer) + in-place BLAS refit into `cb`. No submit — the caller
        // batches these into the main frame command buffer (recordCommandBuffer),
        // exactly like the skinned-mesh path, so there is NO mid-frame
        // vkQueueWaitIdle. Uses the persistent per-BLAS scratch (grass geometry
        // is fixed-size, so it's allocated once on first use). Normals are left
        // static, so only the position buffer is written/barriered. Before the
        // dispatch it copies the current pose into blas->prevVertex, the motion
        // vectors' previous pose, as the skinned and displaced paths do.
        void recordGrassDeform(VkCommandBuffer cb, GrassMesh& gm, GrassMeshState& st);

        // Grass fields that swayed last frame and are not deformed this frame
        // (the distance-gated freeze): copy vertex -> prevVertex once so their
        // motion collapses to zero. grassPrevSettlePending() says whether any
        // is due, so the deform.grass pass exists on a frame with no deform.
        [[nodiscard]] bool grassPrevSettlePending() const;
        void recordGrassPrevSettles(VkCommandBuffer cb);

        // One-shot wrapper — used only for the initial prime during scene build
        // (rare). The per-frame path records into the frame cb via the pending
        // queue, so it never drains. Leaves prevVertex == vertex (no motion on
        // the field's first frame).
        void refreshGrassBlas(GrassMesh& gm, GrassMeshState& st, float /*elapsedSeconds*/);

        // Build a TLAS over the supplied instance descriptors. Empty input is
        // legal — produces an empty TLAS that always misses.
        // buildNow false: the structure and its buffers only, sized for the
        // pool; the caller's frame records its first build (recordTlasRefit
        // promotes the refit to one, nothing having been built).
        void buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances, bool buildNow = true);

        // Record a per-frame TLAS refit/rebuild into `cb` — NO blocking submit.
        // The host instance write goes to tlasInstancesBuffers[currentFrame]
        // (safe: recordCommandBuffer runs after this slot's fence wait), and the
        // build uses the persistent scratch. Leaves the TLAS handle unchanged so
        // descriptor binding 0 keeps pointing at it. Must be recorded AFTER all
        // deformable BLAS rebuilds in the same cb.
        void recordTlasRefit(VkCommandBuffer cb,
                             const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                             bool fullBuild);

        // Sized for `capacity` descs (at least descs.size()): the slot path
        // writes rows past the structural build's count into the same buffer.
        template<typename DescT>
        void uploadDescBuffer(Buffer& target, const std::vector<DescT>& descs, size_t capacity = 0) {
            const VkDeviceSize bytes = std::max<VkDeviceSize>(
                    std::max(descs.size(), capacity) * sizeof(DescT), sizeof(DescT));
            target = createBuffer(
                    ctx->allocator(), ctx->device(), bytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            if (!descs.empty()) {
                uploadHostVisible(ctx->allocator(), target, descs.data(),
                                  descs.size() * sizeof(DescT));
            }
        }

        // Flush the cached host-side MaterialDescs into this frame's slot of
        // the per-frame ring. Called from renderFrame after the fence wait —
        // the in-flight signal guarantees the previous use of this slot has
        // retired, so the memcpy races nothing. Replaces the old shared-buffer
        // path that called vkDeviceWaitIdle on every animated-pbr update.
        void flushMaterialDescsIfDirty(uint32_t frame);
        // Same fence-gated ring flush for GeometryDescs (auto-LOD level
        // switches repoint indexAddress/indexed — see geomDescsCached_).
        void flushGeometryDescsIfDirty(uint32_t frame);

        // Per-frame frustum cull: tag every entry with `inFrustum` so raster
        // passes (gbuf prepass, overlay depth prepass) can skip off-screen
        // geometry. Each draw on the raster gbuf pass costs ~15 µs of GPU
        // command-processor time regardless of whether anything actually
        // rasterizes — at 1500-mesh scenes that's ~22 ms eaten by draws
        // whose vertex shader transforms all land outside the clip cube.
        //
        // Deformable entries (skinned / displaced / morphed) keep
        // inFrustum=true unconditionally — their local AABB is the rest-
        // pose extent, not the deformed one, so a tight test would clip
        // out poses that bulge past the bind silhouette. Re-fitting a
        // tight bound every frame on the CPU isn't worth the cost; the
        // GPU pays a small constant for these. Overlay entries also stay
        // on (debug viz should always render).
        void cullEntriesAgainstFrustum(Camera& camera);

        // Pull per-mesh PBR material params off the threepp Material chain.
        // Anything not satisfying the relevant interface gets a sensible
        // default so meshes lacking a material still render. albedoTexIndex
        // stays at -1 here — caller patches it after `ensureMaterialTexture`.
        static MaterialDesc materialFromMesh(const Mesh& m);

        // three.js parity: InstancedMesh::instanceColor multiplies the
        // material's diffuse color per instance (color_vertex.glsl). Carried
        // in the entry's GeometryDesc, which every consumer (raster G-buffer,
        // traced hits, lidar) reaches per entry, and multiplied into the
        // material record's albedo there (entryMat). (1,1,1) for plain meshes
        // and for instanced meshes that never set a color: a multiply by one
        // leaves the albedo's bits as they are.
        static void setGeomInstanceColor(GeometryDesc& gd, const MeshEntry& en);

        // Dithered LOD crossfade (Mesh::setFade, InstancedMesh::setFadeAt).
        // One word per entry in the high-half layout of vulkan_shared.h
        // (kLodFade*), 0 for no fade: buildIndirectDrawData ORs it into
        // DrawInfo.flags, where gbuffer.frag dithers against it. The rest of
        // the renderer (TLAS masks, shadow-map casters, overlay occluders)
        // asks lodFadeCasts, so a crossfading pair is ONE level to them.
        // Kept beside the entries rather than in the MaterialDescs: fades
        // change every frame of a transition on scattered instances, and a
        // 4-byte word beats re-flushing 624-byte descs range by range.
        std::vector<uint32_t> entryLodFade_;
        // Bumped whenever an entry's lodFadeCasts flips, so the cached static
        // shadow cascades (shadowStaticHash) redraw with the other level.
        uint32_t lodFadeCastEpoch_ = 0;
        struct LodFadeChange {
            bool words   = false;// some entry's word changed: DrawInfo is stale
            bool casters = false;// some entry's lodFadeCasts flipped
        };
        // Polls every span's fade source (the instanceFade attribute's
        // pointer + version, else the mesh-wide fade) and re-packs the entries
        // of the spans that changed; `all` re-packs every span (structural
        // rebuild). O(spans) on a frame where nothing changed.
        LodFadeChange refreshLodFades(bool all);
        static uint32_t packLodFade(float fade, bool complement);
        // Whether shadows and rays see this entry. Exactly one side of a
        // plain-f / complement-(1-f) pair: plain while its fade is at least
        // 0.5, complement while above 0.5 (the odd step count makes the cut
        // of a pair never sit exactly on the half).
        static bool lodFadeCasts(uint32_t word) {
            if (word == 0u) return true;
            const uint32_t cut = (word & kLodFadeCutMask) >> kLodFadeCutShift;
            return (word & kLodFadeComplement) ? 2u * cut < kLodFadeSteps : 2u * cut > kLodFadeSteps;
        }

        static std::shared_ptr<Texture> emissiveTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* em = dynamic_cast<MaterialWithEmissive*>(mat.get())) {
                return em->emissiveMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> occlusionTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* ao = dynamic_cast<MaterialWithAoMap*>(mat.get())) {
                return ao->aoMap;
            }
            return nullptr;
        }

        static void copyTexUvTransform(float (&dst)[9], const std::shared_ptr<Texture>& tex) {
            if (!tex) return;
            if (tex->matrixAutoUpdate) tex->updateMatrix();
            std::copy(tex->matrix.elements.begin(), tex->matrix.elements.end(), dst);
        }

        // Tiled world-anchored detail albedo (MaterialWithDetailMap; terrain
        // etc.). Returns null when absent or strength is zero.
        static std::shared_ptr<Texture> detailTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* dm = dynamic_cast<MaterialWithDetailMap*>(mat.get())) {
                if (dm->detailMap && dm->detailStrength > 0.f) return dm->detailMap;
            }
            return nullptr;
        }

        // Detail NORMAL + ROUGHNESS map (MaterialWithDetailMap). Independent of
        // the detail albedo: a terrain can carry relief/roughness breakup with
        // no albedo layer, or vice-versa. Gated on the map plus a nonzero
        // strength on either the normal or the roughness term.
        static std::shared_ptr<Texture> detailNormalTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* dm = dynamic_cast<MaterialWithDetailMap*>(mat.get())) {
                if (dm->detailNormalMap && (dm->detailNormalScale > 0.f || dm->detailRoughStrength > 0.f))
                    return dm->detailNormalMap;
            }
            return nullptr;
        }

        // Terrain splat maps (MaterialWithTerrainMaps). The weight map is the
        // path's on-switch: bound only when a band set exists to select.
        static std::shared_ptr<Texture> terrainWeightTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get())) {
                if (tm->terrainMapsActive()) return tm->terrainWeightMap;
            }
            return nullptr;
        }

        // World-space normal map — independent of the band sets (LOD-seam fix
        // is worthwhile even for a macro-colour-only terrain).
        static std::shared_ptr<Texture> terrainNormalTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get()))
                return tm->terrainNormalMap;
            return nullptr;
        }

        static std::shared_ptr<Texture> terrainBandAlbedoTexOf(const Mesh& m, int band) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get()))
                return tm->terrainBandAlbedo[static_cast<size_t>(band)];
            return nullptr;
        }

        static std::shared_ptr<Texture> terrainBandNormalTexOf(const Mesh& m, int band) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tm = dynamic_cast<MaterialWithTerrainMaps*>(mat.get()))
                return tm->terrainBandNormalRough[static_cast<size_t>(band)];
            return nullptr;
        }

        // Walk a material's chain for the albedo (`map`) texture.
        // Returns null if the material doesn't carry one.
        static std::shared_ptr<Texture> albedoTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* mm = dynamic_cast<MaterialWithMap*>(mat.get())) {
                return mm->map;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> roughnessTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* r = dynamic_cast<MaterialWithRoughness*>(mat.get())) {
                return r->roughnessMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> metalnessTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* mt = dynamic_cast<MaterialWithMetalness*>(mat.get())) {
                return mt->metalnessMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> normalTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* nm = dynamic_cast<MaterialWithNormalMap*>(mat.get())) {
                return nm->normalMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> transmissionTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* tr = dynamic_cast<MaterialWithTransmission*>(mat.get())) {
                return tr->transmissionMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> clearcoatTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* cc = dynamic_cast<MaterialWithClearcoat*>(mat.get())) {
                return cc->clearcoatMap;
            }
            return nullptr;
        }

        static std::shared_ptr<Texture> clearcoatRoughnessTexOf(const Mesh& m) {
            auto mat = m.material();
            if (!mat) return nullptr;
            if (auto* cc = dynamic_cast<MaterialWithClearcoat*>(mat.get())) {
                return cc->clearcoatRoughnessMap;
            }
            return nullptr;
        }

        // The maps a mesh fingerprint compares (MeshFingerprint::tex).
        static MaterialTexSet texSetOf(const Mesh& m) {
            return {albedoTexOf(m), roughnessTexOf(m), metalnessTexOf(m),
                    normalTexOf(m), transmissionTexOf(m), clearcoatTexOf(m),
                    clearcoatRoughnessTexOf(m), emissiveTexOf(m), occlusionTexOf(m)};
        }

        // The material values a mesh fingerprint compares (MeshFingerprint::pbr):
        // albedo, roughness, metalness, emissive + intensity, normalScale,
        // transmission + ior, clearcoat + its roughness.
        static std::array<float, 15> pbrKeyOf(const MaterialDesc& md) {
            return {md.albedo[0], md.albedo[1], md.albedo[2],
                    md.roughness, md.metalness,
                    md.emissive[0], md.emissive[1], md.emissive[2],
                    md.emissiveIntensity,
                    md.normalScale[0], md.normalScale[1],
                    md.transmission, md.ior,
                    md.clearcoat, md.clearcoatRoughness};
        }

        // Walk the scene each frame, fingerprint the meshes, and rebuild the
        // TLAS + per-instance desc tables when anything that affects ray
        // tracing changes (mesh added/removed, transform animated, material
        // slider tweaked). BLAS records persist across rebuilds — keyed by
        // BufferGeometry pointer — so static geometry isn't re-traced.
        //
        // Per-frame cost when nothing changes is: traverse + N MeshFingerprint
        // compares. Cheap. The rare rebuild path waits the GPU idle, retires
        // the TLAS + scene desc buffers, and rewrites bindings 0/3/4 across
        // every descriptor set without re-allocating from the pool.
        void ensureSceneBuilt(Object3D& scene, Camera& camera);

        void createCameraUbos();

        // Grow emissiveTriBuffers[frame] in-place if the current frame's
        // emissive count exceeds capacity. Returns true when the buffer
        // handle changed so the caller can rewrite binding 14. 2× headroom
        // matches motionMatBuffers.
        bool ensureEmissiveTriCapacity(uint32_t frame, uint32_t needed);

        // Clears both ReSTIR DI reservoir ping-pong image pairs
        // (reservoirPosImagesPP + reservoirWImagesPP) to (0,0,0,0), so the
        // deferred shade's temporal-reuse read sees M=0 (no prior history)
        // on the next frame instead of garbage. A stale tap that *does* slip
        // past the reservoir's own validity guard (e.g. by undefined memory
        // aliasing a real mesh-ID after image creation) then returns a
        // harmless 0 mean / 0 M reservoir instead of garbage. Caller must
        // hold the GPU idle (we don't issue our own barrier-into-TRANSFER_DST
        // since the only legal layout transition path for an image being
        // cleared is GENERAL → TRANSFER_DST → GENERAL).
        void clearGbufImages();

        // Compute per-instance motion matrices = prevWorld * inverse(curWorld)
        // and upload to motionMatBuffers[frame]. Identity for first-seen
        // entries (cold-start frame after a topology rebuild) so the reproject
        // is a no-op until prevWorldMats picks up real history. Keying by
        // (Mesh*, instanceIndex) so each InstancedMesh sub-instance carries
        // its own motion delta. Caller must have already waited the
        // inFlight[frame] fence — we write a buffer the GPU may have been
        // reading on the previous use of `frame`.
        // Reset the entry-aligned motion state (scratch identity, prev =
        // current worlds). Called at every full rebuild (before the identity
        // remap restores surviving history) and by the accumulation resets.
        void seedMotionState(const std::vector<MeshEntry>& entries);

        void computeAndUploadMotionMatrices(uint32_t frame,
                                            const std::vector<MeshEntry>& entries);

        // ── GPU instance expansion (stage 1) ────────────────────────────────
        // Rebuild the GPU-path span list, grow this slot's pools, and upload
        // the SpanDescs + every span whose instance matrices the CPU re-read
        // since this slot last saw them. Caller must have already waited the
        // inFlight[frame] fence: both pools and the frame's descriptor set are
        // written here, and a descriptor written while its frame is in flight
        // is the VUID-03047 zone.
        void prepareInstanceExpansion(uint32_t frame);

        // ── ParticleField (phase 0) ─────────────────────────────────────────
        // Grow this frame's per-field position ring, copy each field's host
        // staging into THIS frame's ring slot (version-gated), and rewrite the
        // FieldDesc SSBO. Same window and same reason as
        // prepareInstanceExpansion: post-fence, pre-record.
        void prepareParticleFields(uint32_t frame);
        // Bring up the ParticleField pass if it does not exist yet. Called from
        // ensureHybridResources on the ordinary path and from
        // enableParticleFieldInterop, which must work before the first frame.
        void ensureParticleFieldPass();

        // Publish each field's device-side live count into its draw command.
        // Head of the frame command buffer, before any consumer reads it.
        void recordParticleFieldCounts(VkCommandBuffer cb);

        // ── ParticleField device emitter (F2) ───────────────────────────────
        // One dispatch per Ownership::Renderer field, at the head of the frame
        // command buffer and before the density scatter, the counts copy and
        // every view's raster pass — all of which read the positions it writes.
        // Bracketed by TP_ParticleEmit for timing. No-op without a
        // Renderer-owned field.
        void recordParticleFieldEmit(VkCommandBuffer cb, uint32_t frame);

        // ── ParticleField density (phase 2) ─────────────────────────────────
        // Clear + splat every density field into its world-anchored volume,
        // once for all views. Bracketed by TP_ParticleDensity for timing.
        // No-op without a density field.
        void recordParticleDensityScatter(VkCommandBuffer cb, uint32_t frame);
        // Per-FIF ParticleDensityUbo (bindings 68) + the 1x1x1 dummy volume.
        // Both created once and never resized.
        void ensureParticleDensityResources();
        // Rewrite the frame's ParticleDensityUbo from the pass's current volume
        // list. Post-fence, pre-record; also refreshes the deferred descriptor
        // sets when the volume LIST changed (generation bump).
        void updateParticleDensityUbo(uint32_t frame);

        // ── Splat reflection volumes (plans/splat-volume-reflections.md) ────
        // Per-FIF SplatVolumeUbo (binding 71) + the 1x1x1 dummy volume for the
        // unused slots of binding 70. Both created once and never resized —
        // the two functions above, mirrored.
        void ensureSplatVolumeResources();
        // The eight image views binding 70 names: live volumes first, the
        // 1x1x1 dummy for the rest. ONE producer, used by both the descriptor
        // write and the staleness key below, so the two cannot disagree about
        // what "the current list" is.
        void splatVolumeBindViews(std::array<VkImageView, vulkan::kMaxSplatVolumes>& out);
        // Identifies that exact list. Keyed on the view handles, not on
        // SplatPass::volumeGeneration() alone: the generation bumps when a
        // volume is baked or freed, and the free happens inside syncClouds —
        // by which time a descriptor set that still named the view has already
        // been recorded into a command buffer that may still be executing
        // (VUID-vkDestroyImageView-imageView-01026). The handle list changes
        // the moment a cloud stops being visible, which is framesInFlight+1
        // syncs before retireStale destroys anything — the margin retireStale
        // assumes every consumer has. The generation is mixed in anyway, so a
        // rebake that happens to be handed the same handle back still counts
        // as a change. The cost is a rewrite whenever a cloud is shown or
        // hidden; the density table accepts the same churn.
        [[nodiscard]] uint64_t splatVolumeBindKey();
        // Rewrite the frame's SplatVolumeUbo from SplatPass::volumeEntries():
        // world→UVW and the conservative world AABB composed on the HOST from
        // each cloud's model matrix and local bake box. Post-fence, pre-record,
        // and AFTER collectSplatClouds' syncClouds (so a cloud uploaded this
        // frame is baked before the UBO names it); also refreshes the deferred
        // descriptor sets when the volume LIST changed (generation bump).
        void updateSplatVolumeUbo(uint32_t frame);

        // One vkCmdDrawIndirect per visible ParticleField, inside the G-buffer
        // render pass and after every ordinary bucket. Separate pipeline, and
        // deliberately so: see particlefield_gbuf.vert's header.
        void recordParticleFieldDraws(VkCommandBuffer cb, bool useMsaa);

        // Upload + BLAS-build a geometry that is nobody's Mesh geometry (the
        // ParticleField MeshRepr proxy). Same blasCache, same liveCheck-based
        // eviction as every ordinary static geometry.
        const BlasRecord* ensureCachedBlas(const std::shared_ptr<BufferGeometry>& geom) {
            if (!geom) return nullptr;
            auto it = blasCache.find(geom.get());
            if (it != blasCache.end()) return it->second.get();
            auto rec = buildBlasFor(*geom, /*allowPacked=*/true, /*updatable=*/false);
            if (!rec) return nullptr;
            rec->liveCheck = geom;
            return blasCache.emplace(geom.get(), std::move(rec)).first->second.get();
        }

        // One dispatch, into the frame's already-open command buffer.
        void recordInstanceExpansion(VkCommandBuffer cb, uint32_t frame);

        // Copy the GPU world matrices back and compare them against
        // MeshEntry::worldMatrix over the instanced spans. DRAINS THE DEVICE;
        // backs VulkanRenderer::instanceExpandCheck.
        bool verifyInstanceExpansion(VulkanRenderer::InstanceExpandCheck& out);

        // Upload meshMovedBits_ to meshMovedBitsBuffers[frame]. Caller must have
        // already waited the inFlight[frame] fence.
        void uploadMeshMovedBits(uint32_t frame);

        // Walk visible entries, gather emissive triangles in world space, and
        // upload to emissiveTriBuffers[frame]. Per-tri 64-byte record:
        //   v0.xyz = world pos0,    v0.w = triangle area
        //   v1.xyz = world pos1,    v1.w = running cumPower (CDF)
        //   v2.xyz = world pos2,    v2.w = per-tri power (lum * area)
        //   emission.xyz = emissive*intensity, emission.w = unused
        // then a 64-byte header (v0.x = emissive-instance count, v0.y = the
        // range gate's exposure term) and, under kEmissiveGateMaxLights, one
        // 64-byte record per emissive instance for the shader's range gate
        // and coverage mode (emissive_lights.glsl).
        //
        // Uniform-by-area within each tri × power-weighted picking across tris
        // gives a constant area-weighted-luminance pdf for closest_hit's NEE.
        //
        // Returns true when the buffer handle changed (capacity grew); caller
        // must then rewrite descriptor binding 14 for this frame's sets.
        bool buildAndUploadEmissiveTris(uint32_t frame,
                                        const std::vector<MeshEntry>& entries);

        // Grow motionMatBuffers[frame] in-place if the current scene's
        // instance count exceeds capacity. Returns true when the buffer
        // handle changed so the caller can rewrite binding 10. We grow with
        // 2× headroom to avoid thrashing on incremental scene growth.
        bool ensureMotionMatCapacity(uint32_t frame, uint32_t needed);

        // Same dance as ensureMotionMatCapacity, but for the per-mesh
        // moved-bitmask SSBO at binding 21. `neededWords` is the number of
        // 32-bit words required to address every visible TLAS instance.
        bool ensureMeshMovedBitsCapacity(uint32_t frame, uint32_t neededWords);



        // REVERSED-Z Vulkan projection from a GL-convention (NDC z∈[-1,1]) proj.
        // threepp's projection is GL-style; fed to Vulkan as-is it (a) clips the
        // [-1,0] near-slab so HALF the depth precision is wasted on never-visible
        // geometry, and (b) bunches precision near the camera → distant z-fighting.
        // Reverse-Z maps near→1, far→0 in Vulkan [0,1] → reclaims the lost half AND
        // gives near-uniform precision (with D32F, kills distance z-fighting). The
        // new z-output row = 0.5*w_row - 0.5*z_row. Pair with depth clear 0.0 +
        // GREATER compare. Shaders are convention-agnostic (use the uploaded
        // matrices), so this is host-side only.
        static Matrix4 reverseZVk(const Matrix4& glProj) {
            Matrix4 p = glProj;
            auto& e = p.elements;// column-major: z-output = row 2, w-output = row 3
            for (int c = 0; c < 4; ++c)
                e[c * 4 + 2] = 0.5f * e[c * 4 + 3] - 0.5f * e[c * 4 + 2];
            return p;
        }

        void updateCameraUbo(uint32_t frame, Camera& camera);

        // ── Hybrid raster runtime helpers ───────────────────────────────────
        // (halton_ is also used by updateCameraUbo above to mirror the raster's
        // jitter into the deferred shade's camera UBO — keep both helpers
        // reachable.)
        // Halton(2,3) sub-pixel jitter for raster TAA. (1, base) skips the
        // zero entry so frame 0 already gets a non-zero offset.
        static float halton_(uint32_t i, uint32_t base) {
            float f = 1.f, r = 0.f;
            while (i > 0u) {
                f /= float(base);
                r += f * float(i % base);
                i /= base;
            }
            return r;
        }

        // FSR2-style jitter phase count: the Halton(2,3) sequence length must
        // scale with the upscale ratio so the sub-pixel samples fully cover the
        // higher-res OUTPUT grid — 8 at native, 8·ratio² when upsampling (≈18 at
        // 1.5×, 32 at 2×, 72 at 3×). A fixed period under-samples the grid the
        // moment renderScale < 1, leaving residual aliasing the temporal resolve
        // can't reconstruct. Mirrors ffxFsr2GetJitterPhaseCount. ratio = max over
        // both axes (they're equal under uniform renderScale); clamped to [8,128].
        static uint32_t jitterPhaseCount_(VkExtent2D render, VkExtent2D display) {
            const float rx = render.width  > 0u ? float(display.width)  / float(render.width)  : 1.f;
            const float ry = render.height > 0u ? float(display.height) / float(render.height) : 1.f;
            const float ratio = std::max(rx, ry);
            const uint32_t n = static_cast<uint32_t>(8.f * ratio * ratio + 0.5f);
            return std::clamp(n, 8u, 128u);
        }

        // Cache per-entry cull mode from each entry's material record.
        // Called wherever the table is patched so the gbuffer draw loop can
        // pick BACK-cull (Front, default fast path), FRONT-cull (Back), or
        // NONE-cull (Double).
        //
        // Side::Front + transmissive materials with proper outward winding
        // (ocean, most glass viewed from outside) render correctly under
        // BACK culling. Camera-inside-glass cases (windshield from cabin)
        // are an artist content concern: mark the glass as Side::Double if
        // you need interior viewing — chit's BVH path doesn't cull, but the
        // raster prepass does, so unmarked single-sided glass shows the
        // surface BEHIND it in the gbuffer.
        void cacheCullFlags();

        // Resolve the BlasRecord backing a given visible entry. The same
        // physical buffers feed BLAS and the raster prepass (VERTEX_BUFFER_BIT
        // was added at allocation), so this is a pure lookup, no upload.
        // Branches off the cached type flags so we don't dynamic_cast every
        // entry on every raster draw call.
        const BlasRecord* resolveBlasForEntry(const MeshEntry& en) const {
            if (en.isParticleField) {
                // A field's raster draw pulls the MeshRepr PROXY, not the
                // zero-area placeholder the field carries as its Mesh geometry
                // (the placeholder is what keeps the field's own BLAS/TLAS
                // instance harmless, and what keeps it valid on GL). Nothing
                // else names the proxy, so ensureCachedBlas uploads it during
                // scene expansion and this is the read side.
                const auto& repr = static_cast<const ParticleField*>(en.mesh)->meshRepr();
                if (!repr.enabled || !repr.geometry) return nullptr;
                auto it = blasCache.find(repr.geometry.get());
                return it != blasCache.end() ? it->second.get() : nullptr;
            }
            if (en.isSkinned) {
                auto* sm = static_cast<SkinnedMesh*>(en.mesh);
                auto it = skinnedMeshStates.find(sm);
                if (it != skinnedMeshStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isDisplaced) {
                auto* dm = static_cast<DisplacedMesh*>(en.mesh);
                auto it = displacedStates.find(dm);
                if (it != displacedStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isGrass) {
                auto* gm = static_cast<GrassMesh*>(en.mesh);
                auto it = grassStates.find(gm);
                if (it != grassStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isTet) {
                auto it = tetMeshStates.find(en.mesh);
                if (it != tetMeshStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (en.isMorphed) {
                auto it = morphedMeshStates.find(en.mesh);
                if (it != morphedMeshStates.end() && it->second->blas)
                    return it->second->blas.get();
            }
            auto* geom = en.mesh->geometry().get();
            auto it = blasCache.find(geom);
            if (it != blasCache.end()) return it->second.get();
            return nullptr;
        }

        // Per-frame raster camera UBO upload + descriptor set rewrite.
        // Must run AFTER ensureSceneBuilt (motionMatBuffers[frame] populated)
        // and AFTER the inFlight fence wait (safe to write a slot the GPU
        // was reading on the previous use of `frame`).
        void uploadRasterCameraUbo(uint32_t frame, Camera& camera);

        // Layout (and its doc comment) moved to VulkanGpuLayouts.hpp.
        using DrawInfoGpu = vulkan::impl::DrawInfoGpu;

        // ── Stable / semantic object IDs for the segmentation AOVs ───────
        // outIds.x is the per-frame visible-set index; outIds.y must be STABLE
        // across frames. We assign a dense 31-bit id per Object3D (keyed by the
        // process-stable Object3D::id) the first time it is drawn, so the label
        // survives add/remove/hide/LOD. Callers may override the instance id and
        // set an 8-bit semantic class (folded into outIds.z bits 8..15 by
        // buildIndirectDrawData). Maps are keyed by Object3D::id (never a raw
        // pointer) so a deleted object leaves an inert entry, never a dangling read.
        std::unordered_map<unsigned int, uint32_t> autoStableIds_;      // Object3D::id -> dense id
        std::unordered_map<unsigned int, uint32_t> instanceIdOverride_; // user-set instance id
        std::unordered_map<unsigned int, uint16_t> classIds_;           // user-set semantic class
        uint32_t nextAutoStableId_ = 1;// 0 reserved for sky / unassigned
        // Stable ids stop at INT32_MAX: LidarReturn::hitInstanceId reports the
        // same id as int32 with negative sentinels (-1 miss, -2 volume).
        static constexpr uint32_t kMaxStableId = 0x7FFFFFFFu;

        uint32_t stableIdForObject(const Object3D& o) {
            // Both id maps are empty unless the app opts in (setInstanceId /
            // setClassId), which is the common case — skip the probe entirely.
            if (!instanceIdOverride_.empty()) {
                if (const auto it = instanceIdOverride_.find(o.id); it != instanceIdOverride_.end()) {
                    return it->second;
                }
            }
            const auto [it, inserted] = autoStableIds_.try_emplace(o.id, 0u);
            if (inserted) {
                it->second = nextAutoStableId_;
                if (nextAutoStableId_ < kMaxStableId) ++nextAutoStableId_;// saturate
            }
            return it->second;
        }
        // Read-only sibling of stableIdForObject: same answer, but never
        // ASSIGNS an auto id. The lidar's entry->id table (entryStableIds_) is
        // filled from the TLAS build loop, which runs before the indirect draw
        // builder in a frame; calling the assigning form there would renumber
        // the auto ids into TLAS-entry order and silently change what the
        // raster Ids AOV reports for unlabelled objects. So the assignment
        // stays where it has always been and this only reads the result.
        //
        // 0 is returned for an object that has no user id AND has not been
        // auto-numbered yet — which is the documented meaning of 0 (sky /
        // unassigned), not a new sentinel. It self-heals after one frame for
        // anything the draw builder visits.
        [[nodiscard]] uint32_t stableIdIfAssigned(const Object3D& o) const {
            if (!instanceIdOverride_.empty()) {
                if (const auto it = instanceIdOverride_.find(o.id); it != instanceIdOverride_.end()) {
                    return it->second;
                }
            }
            const auto it = autoStableIds_.find(o.id);
            return it == autoStableIds_.end() ? 0u : it->second;
        }
        uint16_t classIdForObject(const Object3D& o) const {
            if (classIds_.empty()) return 0;
            const auto it = classIds_.find(o.id);
            return it == classIds_.end() ? uint16_t(0) : it->second;
        }

        // Per-INSTANCE occlusion-cull bits. The two-phase cull's persistent
        // visBits history needs one bit per drawn instance — keying it by the
        // per-OBJECT stable id above made phase 1 all-or-nothing for an
        // InstancedMesh (every instance shared one bit, and phase 2's
        // per-instance results raced on it, so a 10k-tree field either drew
        // whole or not at all and effectively never culled). Each mesh
        // reserves a contiguous bit range [base, base+capacity) the first
        // time it is drawn (capacity = its full instance count, 1 for plain
        // meshes); an entry's bit = base + instanceIndex. If count() later
        // grows past the reservation the mesh is re-based — the abandoned
        // bits go stale, which at worst mispredicts phase 1 for one frame
        // (phase 2 recovers the same frame, the cull's standing guarantee).
        // Definition moved to VulkanSceneTypes.hpp.
        using OcclBitRange = vulkan::impl::OcclBitRange;
        std::unordered_map<unsigned int, OcclBitRange> occlBitRanges_;// Object3D::id -> range
        uint32_t occlBitDomain_ = 0;// one past the highest reserved bit
        // One-entry memo: an InstancedMesh's entries are contiguous in the
        // record loop, so a 100k-instance field pays ONE map lookup per
        // frame, not 100k (the same reason the type probes are cached on
        // the entry). Refreshed on every miss; a re-based range refreshes
        // it too since the miss path rewrites it.
        unsigned int occlBitMemoMeshId_ = 0;// Object3D ids start at 1 — 0 = empty
        OcclBitRange occlBitMemoRange_{0u, 0u};

        uint32_t occlCullBitFor(const MeshEntry& en) {
            const uint32_t need = en.instanceIndex + 1u;
            if (en.mesh->id == occlBitMemoMeshId_ && need <= occlBitMemoRange_.capacity)
                return occlBitMemoRange_.base + en.instanceIndex;
            auto [it, inserted] = occlBitRanges_.try_emplace(en.mesh->id, OcclBitRange{0u, 0u});
            if (inserted || it->second.capacity < need) {
                uint32_t cap = need;
                if (en.isInstanced)// flag guarantees the static type — no dynamic_cast per record
                    cap = std::max(cap, static_cast<uint32_t>(
                                                static_cast<const InstancedMesh*>(en.mesh)->count()));
                it->second = {occlBitDomain_, cap};
                occlBitDomain_ += cap;
            }
            occlBitMemoMeshId_ = en.mesh->id;
            occlBitMemoRange_  = it->second;
            return it->second.base + en.instanceIndex;
        }

        // Definition moved to VulkanSceneTypes.hpp.
        using DrawGroup = vulkan::impl::DrawGroup;
        // [0] Front (BACK cull), [1] Back (FRONT cull), [2] Double (NONE cull),
        // [3] blend decals (NONE cull, drawn LAST with rasterGbufDecalPipeline
        // so their albedo lerps over the already-rasterized receivers).
        // The static order means recordRasterGbufPass can issue at most 4
        // vkCmdDrawIndirect calls and skip the empty ones.
        std::array<DrawGroup, 4> indirectGroups_{};
        uint32_t indirectTotalDraws_ = 0;

        // Per-frame scratch for buildIndirectDrawData, kept as members so the
        // bucket vectors retain their capacity instead of being reallocated and
        // grown by push_back every frame (a few hundred KB of fresh pages per
        // frame at high draw counts — steady-state cost plus allocator churn
        // that surfaces as frame-time variance). Cleared, not rebuilt, at the
        // top of each build; peak capacity is deliberately retained for the
        // renderer's lifetime. Only ever touched by buildIndirectDrawData,
        // which has a single call site and is neither recursive nor reentrant.
        std::array<std::vector<DrawInfoGpu>, 4>                     indirectDrawScratch_;
        std::array<std::vector<VkDrawIndirectCommand>, 4>           indirectCmdScratch_;
        std::array<std::vector<vulkan::OcclusionCull::CullMeta>, 4> indirectOcclScratch_;

        bool ensureDrawInfoCapacity(uint32_t frame, VkDeviceSize neededBytes);

        bool ensureIndirectCmdCapacity(uint32_t frame, VkDeviceSize neededBytes);

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
        void buildIndirectDrawData(uint32_t frame);

        // Begin the raster G-buffer render pass and ship the prebuilt
        // indirect-draw groups via 1-4 vkCmdDrawIndirect calls (one per
        // active cull mode, plus a trailing blend-decal group on the
        // albedo-blend pipeline). Replaces the prior per-mesh draw loop —
        // see buildIndirectDrawData above for how the GPU buffers are
        // populated.
        void recordRasterGbufPass(VkCommandBuffer cb, uint32_t frame);

        // Shared body: `renderPass` must be COMPATIBLE with the pipelines'
        // creation pass (the occlusion-culling load/store variants are), and
        // `indirectBuffer` supplies the VkDrawIndirectCommand records the
        // bucket groups index into (the two-phase path swaps in the compute-
        // written phase buffers; offsets/counts are identical by design).
        // `particles`: issue the ParticleField draws in THIS pass. The
        // two-phase occlusion path calls this twice over the same attachments,
        // and a field is always-draw, so without the flag every particle would
        // rasterize twice. Phase A gets them, so the HiZ built from its depth
        // already sees the bed.
        void recordRasterGbufPassInternal(VkCommandBuffer cb, uint32_t frame,
                                          VkRenderPass renderPass, VkFramebuffer fb,
                                          bool useMsaa, VkBuffer indirectBuffer,
                                          bool clear, bool particles = true);

        // Lazily create the debug_resolve compute pipeline + descriptor set.
        void createDebugResolvePipeline();

        // Debug visualization: sample one G-buffer channel and write a
        // readable RGB image straight to the swapchain via debug_resolve.comp.
        // Replaces the old raw vkCmdBlitImage, which could only show the
        // world-normal attachment: a blit cannot bias the SIGNED motion vector
        // (it clamped to near-black) and is INVALID from the integer ids
        // attachment (R32G32B32A32_UINT) into the UNORM swapchain (the Vulkan
        // spec forbids integer<->non-integer blits). The gbuf attachments are
        // in SHADER_READ_ONLY_OPTIMAL here (the gbuffer render pass declares a
        // COMPUTE consumer dependency, same as the deferred / event-shade
        // consumers), so no gbuf barrier is needed — we only move the
        // freshly-acquired swapchain image to GENERAL for the storage write
        // and leave it there, the exact exit contract the full deferred path
        // uses so endFrame()'s shared overlay/present finalize is unchanged.
        // A frame-graph pass: the set is written at build time and declared
        // from the reflected shader (the swapchain comes from
        // importSwapchain).
        void addHybridDebugResolvePass(vulkan::rg::RenderGraph& g, uint32_t imageIndex, uint32_t frame);


        void createLightsUbos();

        // Walk the scene each frame for AmbientLight + DirectionalLight; pack
        // into the per-frame lights UBO. Direction is computed from the
        // light's world-space position toward its (possibly defaulted) target,
        // mirroring three.js's DirectionalLight.target convention. The shader
        // expects the L vector (toward the light), so we negate.
        void updateLightsUbo(uint32_t frame, Object3D& scene, Camera& camera);

        void createFogUbos();

        // Pack scene.fog (Fog/FogExp2 variant) into the per-frame fog UBO.
        // FogExp2.density maps directly
        // to sigma_t; linear Fog reaches ~63% extinction at farPlane via
        // sigma = 1 / (far - near). Hash detect changes so the per-pixel motion
        // path halves FC and the new fog state converges quickly.
        void updateFogUbo(uint32_t frame, Object3D& scene, Camera& camera);

        void createCloudUbos();
        // Pack the setClouds state into the per-frame cloud UBO (binding 58 of
        // the deferred shade set). Disabled → enabled=0 → the march is a no-op.
        void updateCloudUbo(uint32_t frame);

        // Allocate, transition, and upload an Image2D from a tightly-
        // packed CPU buffer. Pixel layout matches `format`. Caller owns the
        // returned Image2D and must call destroyImage2D() on shutdown.
        // One-shot variant: submits + waits — the queue drain is fine at
        // load time but a measurable hitch mid-frame.
        Image2D createSampledImage2D(uint32_t w, uint32_t h, VkFormat format,
                                     const void* pixels, VkDeviceSize byteSize,
                                     VkFilter filter, VkSamplerAddressMode addrU,
                                     VkSamplerAddressMode addrV,
                                     const char* debugName = nullptr) {
            VkCommandBuffer cb = beginOneShot();
            Buffer staging{};
            Image2D out = buildSampledImage2D(cb, w, h, format, pixels, byteSize,
                                              filter, addrU, addrV, debugName, staging);
            endAndSubmitOneShot(cb);
            // Deferred to the batch flush when a batch is open (staging is still
            // referenced by the not-yet-submitted shared cb); immediate freeing
            // otherwise. The image view/sampler are host-side and valid at
            // once; only the pixel content lands when the batch submit runs,
            // which happens before any shader samples it.
            destroyBufferMaybeBatched(staging);
            return out;
        }

        // Block-compressed sampled image from PRE-ENCODED mip levels (level 0
        // first, every level present down to 1x1). BC images cannot be blit-
        // downsampled, so unlike buildSampledImage2D there is no GPU mip-gen:
        // the caller encodes each level (bcn::buildMipChainRGBA8Area +
        // bcn::bc7EncodeMode6) and this uploads the chain verbatim — one
        // staging buffer, one copy region per level. View/sampler setup
        // matches buildSampledImage2D (aniso + full LOD range). Nothing here
        // is specific to block formats: an uncompressed material texture
        // whose size is not a power of two comes this way too, with RGBA8
        // levels, because the blit chain is only exact on even levels.
        Image2D createSampledImageBC(uint32_t w, uint32_t h, VkFormat format,
                                     const std::vector<std::vector<std::uint8_t>>& levels,
                                     VkFilter filter, VkSamplerAddressMode addrU,
                                     VkSamplerAddressMode addrV,
                                     const char* debugName = nullptr);

        // In-frame upload of CPU-built levels (level 0 first; one level is an
        // unmipped image): records it into the CALLER's command buffer (must
        // be outside a render-pass instance) and retires the staging buffer
        // through the frame-serial queue — no submit, no queue drain. A
        // mid-record one-shot's vkQueueWaitIdle blocks on every in-flight
        // frame (~2 frames of GPU time); HUD TextSprites re-rasterizing their
        // atlas per ammo-counter change made that a 40-50 ms hitch on every
        // shot. Same-queue execution order + the final SHADER_READ_ONLY
        // barrier make samples in this and later frames safe without any host
        // wait. The caller builds the mips because the sprite atlases that
        // come this way are odd-sized, where buildSampledImage2D's blit chain
        // skips rows (bcn::buildMipChainRGBA8Area).
        Image2D createSampledImageLevelsInFrame(VkCommandBuffer cb,
                                                uint32_t w, uint32_t h, VkFormat format,
                                                const std::vector<std::vector<std::uint8_t>>& levels,
                                                VkFilter magFilter, VkFilter minFilter,
                                                VkSamplerMipmapMode mipmapMode,
                                                VkSamplerAddressMode addrU,
                                                VkSamplerAddressMode addrV,
                                                const char* debugName = nullptr) {
            Buffer staging{};
            Image2D out = buildSampledImageLevels(cb, w, h, format, levels,
                                                  magFilter, minFilter, mipmapMode,
                                                  addrU, addrV, debugName, staging);
            retire(std::move(staging));
            return out;
        }

        // Shared body of createSampledImageBC and the in-frame variant above:
        // image + view + sampler creation and the recorded upload of every
        // level (one staging buffer, one copy region per level, final
        // transition) into `cb`. `stagingOut` is still referenced by cb when
        // this returns — the caller covers its lifetime (drain+free for the
        // one-shot, frame-serial retire for the in-frame path).
        Image2D buildSampledImageLevels(VkCommandBuffer cb,
                                        uint32_t w, uint32_t h, VkFormat format,
                                        const std::vector<std::vector<std::uint8_t>>& levels,
                                        VkFilter magFilter, VkFilter minFilter,
                                        VkSamplerMipmapMode mipmapMode,
                                        VkSamplerAddressMode addrU,
                                        VkSamplerAddressMode addrV,
                                        const char* debugName,
                                        Buffer& stagingOut);

        // Shared body of createSampledImage2D: image + view + sampler
        // creation and the recorded upload (staging copy + mip-blit chain +
        // final transition) into `cb`. `stagingOut` is still referenced by
        // cb when this returns — the caller covers its lifetime (drain+free).
        Image2D buildSampledImage2D(VkCommandBuffer cb,
                                    uint32_t w, uint32_t h, VkFormat format,
                                    const void* pixels, VkDeviceSize byteSize,
                                    VkFilter filter, VkSamplerAddressMode addrU,
                                    VkSamplerAddressMode addrV,
                                    const char* debugName,
                                    Buffer& stagingOut);

        // ── ParticleSystem billboard resources ──────────────────────────────

        // Lazily build the 1×1 white texel bound for untextured particle
        // systems (matches the GL path, where an unset `tex` sampler reads
        // white). Created once; freed in deinit.
        void ensureParticleWhiteTexture();

        // Upload/refresh a particle texture, keyed on the raw Texture* (the
        // ShaderMaterial uniform holds no shared_ptr). Returns the cached
        // Image2D, or nullptr (caller falls back to the white default). Mirrors
        // OverlayPass::ensureSpriteAtlasTexture, minus the weak_ptr liveCheck.
        const Image2D* ensureParticleTexture(const Texture* tex);

        void destroyParticleGeomRec(ParticleGeomRec& rec);

        // Ensure the per-geometry particle vertex/index buffers exist and the
        // animated attributes (position/normal/color) are current. uv + index
        // are static (uploaded once). Returns nullptr on malformed geometry.
        // Float-typed reads on purpose: the animated attributes re-upload on
        // every version bump, so narrow (compressAttributes) storage would
        // re-widen per update — particle geometry must stay float.
        ParticleGeomRec* ensureParticleGeom(const std::shared_ptr<BufferGeometry>& geomSp);

        // Storage-image (GENERAL layout) factory used for the ReSTIR DI
        // reservoir ping-pong images (createReservoirImages). No staging
        // upload — contents are initialised the first frame after
        // sampleIndex resets to 0. The deferred shade reads/writes these
        // every frame, so we transition once at creation and keep them in
        // GENERAL forever after.
        Image2D createStorageImage2D(uint32_t w, uint32_t h, VkFormat format,
                                     VkImageUsageFlags extraUsage = 0,
                                     const char* debugName = nullptr);

        void createReservoirImages();


        // Manual accumulation reset. Mirrors the post-create reset block above:
        // wipes gbuf + accum + ReSTIR DI reservoirs, rewinds sampleIndex, and
        // invalidates reproject state so the next frame cold-starts from
        // sample 1. Issues a vkDeviceWaitIdle since clearGbufImages requires
        // the GPU idle before its TRANSFER_DST layout transition. When
        // called mid-frame (frameState_ != Idle) the GPU-side clear is
        // deferred to the next beginDeferredFrame; CPU-side counters reset
        // immediately so the user-visible behaviour matches.
        void resetAccumulation();

        // Path-traced LIDAR scan entry-point. Re-uses the main TLAS and the
        // last submitted frame's geom/mat descriptors via a private RT pipeline
        // owned by `lidar_`. Synchronous: blocks the calling thread until
        // per-beam results land in `outResults`.
        // `cleanResults`, when non-null AND params.pairedCleanTrace is set,
        // receives the second (particle-dust-free) leg at the same layout.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& outResults,
                       const LidarParams& params,
                       std::vector<LidarReturn>* cleanResults = nullptr);

        // The same scan, pipelined: fire on one frame, take delivery on a
        // later one. Blocking readback costs every frame already queued behind
        // the fence (~28 ms at two frames in flight), which is a hitch a 10 Hz
        // sensor would inflict ten times a second — see LidarScanner.
        int scanLidarBegin(const std::vector<LidarBeam>& beams, const LidarParams& params);
        [[nodiscard]] bool scanLidarReady(int handle) const;
        bool scanLidarCollect(int handle, std::vector<LidarReturn>& outResults,
                              std::vector<LidarReturn>* cleanResults = nullptr);
        // Per-slot staging for the outstanding dispatches, sized by
        // scanLidarBegin and consumed by scanLidarCollect.
        std::array<std::vector<vulkan_lidar::LidarResult>,
                   vulkan::LidarScanner::kScanSlots> lidarRaw_{};
        // Whether the dispatch in each slot was a paired clean/degraded trace,
        // i.e. whether its raw rows are two legs rather than one. Recorded at
        // dispatch because collect() cannot tell from the row count alone.
        std::array<bool, vulkan::LidarScanner::kScanSlots> lidarPaired_{};
        // Entry-index -> stable id, snapshotted per slot AT DISPATCH. A scan
        // fired with scanBegin() may be collected several frames later, by
        // which time an edit can have renumbered the entry list; translating
        // against the live table would then relabel returns that were traced
        // against the older TLAS. The snapshot is a few KB at most.
        std::array<std::vector<uint32_t>, vulkan::LidarScanner::kScanSlots> lidarStableIds_{};

        // ── Hybrid raster G-buffer prepass implementation ───────────────────
        // Lazy-initialized on first render().
        // All resources owned by Impl; cleanup in dtor + destroyRasterGbufImages
        // is also called on swapchain resize.

        Image2D createAttachmentImage2D(uint32_t w, uint32_t h, VkFormat format,
                                        VkImageUsageFlags usage,
                                        VkImageAspectFlags aspect,
                                        const char* debugName = nullptr,
                                        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
                                        vulkan::TransientPool* pool = nullptr, vulkan::TransientSpan poolSpan = 0,
                                        uint32_t poolSlots = vulkan::kTransientAllSlots);

        // 3D storage image (froxel volumetrics) — the volume sibling of
        // createAttachmentImage2D (OPTIMAL tiling, single mip, 3D view).
        Image2D createImage3D(uint32_t w, uint32_t h, uint32_t depth, VkFormat format,
                              VkImageUsageFlags usage, const char* debugName = nullptr,
                              vulkan::TransientPool* pool = nullptr, vulkan::TransientSpan poolSpan = 0,
                              uint32_t poolSlots = vulkan::kTransientAllSlots);

        void destroyRasterGbufImages();

        // Every MSAA-only render pass / pipeline the hybrid raster path owns
        // (the MS *images* live in rasterGbufs and go through
        // destroyRasterGbufImages instead). ONE owner for the whole set, called
        // both when MSAA is switched off at runtime and from ~Impl, so a future
        // teardown path cannot silently forget half of it — the destructor used
        // to miss the four rasterGbuf*MS handles entirely. Handles are nulled,
        // so calling it twice is safe. Caller must guarantee the device is idle.
        void destroyRasterGbufMsObjects();

        void createRasterGbufRenderPass();

        // Two-phase occlusion-culling variants of the G-buffer pass (1× and
        // MSAA flavours). Same attachments/formats/samples as the source
        // pass → render-pass COMPATIBLE, so the framebuffer and graphics
        // pipelines are shared; only load/store ops and layouts differ:
        //   A: CLEAR + STORE everything; colors end COLOR_ATTACHMENT (pass B
        //      loads them), depth ends DEPTH_STENCIL_READ_ONLY (the farthest
        //      HiZ + cull compute sample it between the passes — under MSAA
        //      the pyramid reduces the raw MS attachment's samples directly).
        //   B: LOAD everything; final layouts identical to the single pass,
        //      so every downstream consumer (incl. gbuf_resolve at MSAA) is
        //      untouched.
        void createOcclRenderPasses(VkSampleCountFlagBits samples,
                                    VkRenderPass& outA, VkRenderPass& outB);

        // MSAA sibling of createRasterGbufRenderPass — mirrors the 1× pass
        // exactly except attachments[*].samples. Consumed only by
        // gbuf_resolve.comp (COMPUTE reads all live samples via texelFetch);
        // nothing else ever touches the MS attachments, so dstStageMask only
        // needs COMPUTE_SHADER (no ray-query stage reads the MS attachments
        // directly — only gbuf_resolve.comp does).
        void createRasterGbufRenderPassMS(VkSampleCountFlagBits samples);

        void createRasterGbufImages(uint32_t w, uint32_t h);

        void createRasterCameraUbos();

        void createRasterDsLayoutAndPool();

        void createRasterGbufPipeline();

        // MSAA sibling of createRasterGbufPipeline — same shaders/vertex
        // layout/descriptor layout/push-constant range (reuses
        // rasterPipelineLayout + rasterDsLayout, already created by the 1×
        // path in ensureHybridResources), only ms.rasterizationSamples and
        // the target render pass differ. Kept as a full duplicate rather
        // than parametrizing createRasterGbufPipeline so the proven 1× path
        // is never touched (msaa=1 byte-identical guarantee).
        void createRasterGbufPipelineMS(VkSampleCountFlagBits samples);

        // ── Hybrid raster overlay pipeline (wireframe variant) ──────────────
        // Dynamic-rendering pipeline targeting the swapchain (B8G8R8A8_UNORM)
        // + the existing G-buffer depth (D32_SFLOAT, read-only). Pushes
        // mat4 mvp + vec4 color (80B). Triangle topology + polygon-mode
        // line draws each visible mesh as a wireframe; the host gates
        // per-draw on material.wireframe / overlayLayer membership.
        // Line/LineSegments get their own pipeline variants + cached
        // vertex buffer, also built below.
        void createOverlayPipeline();

        // ParticleSystem billboard pipelines. Two variants (alpha-blended /
        // additive) that differ only in blend + depth-test state. Both: 4 vertex
        // bindings (pos/normal/uv/color), a combined-image-sampler set 0, a 128B
        // push constant (modelView + proj), dynamic viewport/scissor, dynamic
        // rendering onto the swapchain + unjitDepth (read-only). Modeled on
        // createOverlayPipeline + OverlayPass::createSpriteOverlayPipeline.
        void createParticlePipeline();

        // World-space Sprite billboard pipeline. Perspective billboard
        // (sprite3d.vert) + the shared overlay_sprite.frag, alpha-blended and
        // depth-tested against unjitDepth (occluded by scene geometry,
        // depth-write off). Reuses particlePipelineLayout_ (128B SpritePC + set-0
        // sampler) and builds the shared static quad. Called from
        // createParticlePipeline so the shared layout already exists.
        void createSpriteWorldPipeline();

        // ParticleField billboard pipeline (F-D). Vertex-less: ZERO vertex
        // bindings, a triangle strip of 4, one indirect draw per field with the
        // instance count coming off the device. Additive, depth-TESTED against
        // the same read-only unjittered depth the legacy path uses (which the
        // legacy ADDITIVE variant deliberately does not do — a fireball is
        // meant to draw over the scene, a rain streak behind a wall is not).
        // Called from createParticlePipeline, so the shared texture set layout
        // and the 1x1 white default already exist.
        void createFieldBillboardPipeline();
        // F4: the fullscreen additive draw that folds the billboard glow
        // pyramid into the swapchain inside the overlay pass. Needs
        // billboardGlow_'s set layout, so it is created with it.
        void createFieldGlowCompositePipeline();

        // Draw every visible field's billboard quads into the render pass the
        // caller has already begun. Used by BOTH the primary's post-TAA overlay
        // pass and a secondary view's own composite — a field is scene content,
        // not a primary-view garnish, so a CameraSensor must see it.
        // The caller passes the pipeline (the MSAA overlay variant, the
        // 1-sample sibling a secondary view needs, or F4's linear-HDR glow
        // variant), and `glowPass` selects which fields are drawn and which
        // value domain they are written in.
        // `pipeAlpha` is the alpha-over sibling, bound for fields that ask for
        // it; VK_NULL_HANDLE (the glow leg) draws every field additively.
        void recordFieldBillboards(VkCommandBuffer cb, VkPipeline pipe, bool glowPass,
                                   VkPipeline pipeAlpha = VK_NULL_HANDLE);

        // F4: build this view's BillboardViewGpu record — the display transform
        // and the fog medium the quads are attenuated by — and publish it into
        // the pass's per-frame block. Returns 0 when there is no room, which
        // the caller must treat as "skip the draw": a null buffer_reference
        // dereference is undefined, not a no-op.
        [[nodiscard]] VkDeviceAddress pushBillboardViewRecord(const Matrix4& viewM,
                                                              bool linearOut);

        // F4: render the glow-enabled fields into BillboardGlowPass's offscreen
        // target and run the pyramid over it. Recorded AFTER the upscale/post
        // passes and BEFORE recordHybridOverlay, because the composite draw that
        // consumes it lives inside that overlay pass and compute cannot run
        // inside a render-pass instance. Added to the graph only when
        // fieldBillboardGlowActive().
        void recordFieldBillboardGlow(VkCommandBuffer cb);
        // R8/R9: (T_cam, T_sun) once per particle for the view whose draws come
        // next. Outside any render-pass instance; no-op when nothing marches.
        void recordFieldTransmittance(VkCommandBuffer cb);
        // BillboardRepr::softDistance: when a visible field asks for a soft
        // edge, add the pass that turns `depth` (the depth this view's sprites
        // are tested against, in DEPTH_STENCIL_READ_ONLY_OPTIMAL) into this
        // view's linear depth buffer, and return that buffer so the billboard
        // passes after it can declare their read. VK_NULL_HANDLE (and no pass)
        // when no field asks or the view cannot provide one.
        VkBuffer addFieldSoftDepthPass(vulkan::rg::RenderGraph& g, VkImage depth, VkImageView depthView,
                                       bool multisampled, VkExtent2D ext);
        // Lazily create the glow pass + its two graphics pipelines + its images.
        // Called from the PREPARE window (never mid-recording), so the pipeline
        // compile lands where every other lazy creation in this renderer does.
        void ensureFieldBillboardGlow();

        // Any visible ParticleField will draw billboards this frame. Cheap (a
        // handful of fields at most) and used by sceneHasOverlayContent(), so
        // the overlay depth prepass and the overlay pass itself run for a scene
        // whose ONLY overlay content is a field of embers.
        [[nodiscard]] bool sceneHasFieldBillboards() const;

        // Walk the scene for visible world-space Sprites (screenSpace == false)
        // with a texture map, snapshotting their world transform + material into
        // lastVisibleSprites_. Run every perspective frame (sprites move / spawn
        // / expire constantly — no snapshot caching). Mirrors OverlayPass's
        // sprite collection, minus the screen-space branch.
        void collectWorldSprites(Object3D& scene);

        // Sidecar collector for SplatClouds — same shape and the same reason as
        // collectWorldSprites: a SplatCloud is not a kind the snapshot/BLAS
        // machinery has anything to say about, so it is gathered by its own
        // traversal and nothing else has to change. Also estimates each cloud's
        // p1/p99 view distance here, where the camera is in hand, from the same
        // fixed-stride sample SplatCloud::sortByDepth uses on the GL path.
        void collectSplatClouds(Object3D& scene, Camera& camera);

        // Composite this frame's splat clouds into sceneHdr. Called between the
        // deferred shade and the depth of field. Reads splatParams_, which the
        // collector filled while it still had the camera — everything except
        // the raster's sub-pixel jitter, which is only decided later (in
        // uploadRasterCameraUbo) and is folded in here.
        void addSplatPasses(vulkan::rg::RenderGraph& g);
        void addSecondaryViewSplatPasses(vulkan::rg::RenderGraph& g);

        // Stamp the splat depth AOV into the overlay's depth attachment, so
        // the post-resolve overlay draw (wireframe, lines, world sprites,
        // particle billboards) is occluded by a cloud in front of it — split
        // in two so the stamp can be a DRAW inside the hybrid overlay pass,
        // ordered after the overlays kSplatUnoccludedOverlayLayer exempts.
        // splatStampPrepare gates (clouds AND overlay content this frame —
        // see splatOverlayDepth_ for the latch that turns the AOV on for it)
        // and points the stamp's set at this frame's AOV; true obliges the
        // overlay pass to declare and attach its depth WRITABLE and record
        // recordSplatStampDraw at the exempt/occluded boundary. Called while
        // the frame graph is built.
        bool splatStampPrepare();
        void recordSplatStampDraw(VkCommandBuffer cb);

        // Line geometry cache for the 3D hybrid overlay (recordCommandBuffer's
        // line-draw section). Keyed on raw BufferGeometry*; geomId in the
        // entry guards against recycled-pointer aliasing. (OverlayPass owns a separate
        // cache of the same shape for the ortho path — these are NOT shared.)
        //
        // The counter is advanced once per submitted frame in endFrame, which
        // also runs sweepLineGeomCache — eviction matters for apps that
        // rebuild Line/Points geometry each frame (trajectory, scan and
        // detection-box overlays), which would otherwise leak a buffer set
        // per geometry.
        std::unordered_map<const BufferGeometry*, vulkan::LineCacheEntry> lineGeomCache_;
        uint64_t overlayFrameCounter_ = 0;// lastTouch reference for lineGeomCache_ entries

        // A Points whose material sets morphTargets: GL blends the geometry's
        // morph targets in the vertex shader (morphtargets_sphere rides its
        // points on the morphing sphere that way); the overlay point pipeline
        // reads plain positions, so the blend happens here. Per OBJECT, not per
        // geometry: the geometry, and so its LineRec, is typically shared with
        // the morphing Mesh, and the two need not carry the same influences.
        // One buffer per frame in flight, written while recording slot
        // currentFrame, i.e. past that slot's fence wait, so the frame that
        // last read it has finished. Swept with lineGeomCache_.
        struct MorphedPointsRec {
            std::array<Buffer, kFramesInFlight>             vertex{};
            std::array<std::vector<float>, kFramesInFlight> influences{};// what each slot holds
            std::array<uint32_t, kFramesInFlight>           posVersion{};
            unsigned int       geomId    = 0;
            uint64_t           lastTouch = 0;
            std::vector<float> blended;
        };
        std::unordered_map<const Points*, MorphedPointsRec> morphedPointsCache_;

        // The vertex buffer holding `points`' blended positions for this
        // frame, or VK_NULL_HANDLE when it does not morph (no morphTargets on
        // its material, no position targets, or every influence zero), and
        // the caller draws the geometry's rest positions.
        VkBuffer ensureMorphedPointsUploaded(Points& points, const BufferGeometry& geom);

        // Evict entries untouched for longer than the in-flight window. The
        // margin (> kFramesInFlight) guarantees no evicted buffer is still
        // referenced by an in-flight command buffer; the retire queue then holds
        // them until their frame serial has provably passed, so this is safe to
        // call from endFrame while earlier frames are still executing.
        static constexpr uint64_t kLineGeomEvictAge = 8;
        void sweepLineGeomCache() {
            if (lineGeomCache_.empty() && morphedPointsCache_.empty()) return;
            if (overlayFrameCounter_ <= kLineGeomEvictAge) return;
            const uint64_t cutoff = overlayFrameCounter_ - kLineGeomEvictAge;
            for (auto it = lineGeomCache_.begin(); it != lineGeomCache_.end();) {
                if (it->second.lastTouch < cutoff) {
                    for (auto& rec : it->second.slots) {
                        retire(std::move(rec.vertex));
                        retire(std::move(rec.index));
                        retire(std::move(rec.color));
                    }
                    it = lineGeomCache_.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = morphedPointsCache_.begin(); it != morphedPointsCache_.end();) {
                if (it->second.lastTouch < cutoff) {
                    for (auto& b : it->second.vertex) retire(std::move(b));
                    it = morphedPointsCache_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // Lazy-upload Line / LineSegments / Points geometry into host-visible
        // vertex (+ optional index and color) buffers and return the set to
        // bind while recording frame slot currentFrame. Returns nullptr if the
        // geometry has no usable position attribute. Cached by raw pointer;
        // the BufferAttribute::version counters gate re-uploads, and a
        // geometry that changes gets one buffer set per frame in flight (see
        // vulkan::LineCacheEntry).
        const vulkan::LineRec* ensureLineGeometryUploaded(const BufferGeometry* geom);

        // ── TAA resources ───────────────────────────────────────────────────
        // Pipeline + images + descriptor sets now live in vulkan::TaaResolve.
        // This helper packs the external view sources (raster gbuffer +
        // swapchain) into the args struct TaaResolve expects, then asks
        // TaaResolve to rewrite all per-(frame, swapchain-image) descriptor
        // sets. Call after createTaaImages OR after a swapchain resize OR
        // after the raster gbuffer is reallocated.
        void rewriteTaaDescriptors();

        // The bloom pyramid's internal sets + the PostComposite (reads
        // sceneHdr + bloom level 0 + the per-frame G-buffer for the solid-bg
        // sky bypass, writes the per-frame TAA input). Call after bloom_->
        // createImages OR after the gbuffer / TAA images are reallocated.
        void rewriteBloomDescriptors();
        void ensureSplatTarget();

        // Raster-first deferred shade reads the camera + lights UBOs, the env
        // PMREM, the raster material G-buffer (normal+rough / albedo+metal /
        // depth / ids) and writes bloom_->sceneHdr. Call after the raster
        // gbuffer or sceneHdr is reallocated (resize) and after the env image
        // is rebuilt. The UBO buffers are stable; rewriting them is harmless.
        void rewriteDeferredDescriptors(int onlyFrame = -1);

        // Fit the probe grid to the scene's world AABB (called when probe GI
        // is enabled and the scene structure changed). Walks the canonical
        // entry list with the same boundingBox·worldMatrix union the frustum
        // cull uses; overlay/particle entries don't contribute geometry.
        void fitProbeGridToScene();


        // Lazy bring-up: called at the start of each render() when hybrid is on.
        // Idempotent — handles both initial creation and post-resize reallocation.
        // 1x1 multisample dummy images (2x, or 4x where the device lacks 2x)
        // for deferred_shade.comp's dispatch-B sampler2DMS/usampler2DMS
        // bindings when MSAA is off — see gbufDummyMS_'s declaration for why
        // a single-sample dummy can't stand in here. Formats mirror
        // normal/depth/ids/uv/albedo exactly
        // (SAMPLED usage only; never rasterized into, never resolved from).
        void ensureGbufDummyMS();

        void ensureHybridResources();

        // 1×1 black RGBA32F dummy so the env-map binding is always populated.
        // Replaced with the real scene environment by uploadEnvFromTexture()
        // the first time render() sees a non-empty scene.environment / .background.
        void createDefaultEnvImage();

        // Single shared sampler for every material texture in binding 8.
        // Trilinear + 16× anisotropy: minified surfaces fetch from the proper
        // mip level (set by createSampledImage2D's blit-chain) and glancing
        // angles get aniso-filtered, both of which kill the per-frame texture
        // shimmer that TAA was trying to absorb.
        void createTextureSampler();

        // Slot 0 of the bindless array is a 1×1 white texel — used to fill the
        // descriptor array padding so every slot has a valid view, and as the
        // descriptor for materials whose albedoTexIndex stays at -1 (the
        // shader still indexes safely; the multiply by 1.0 is a no-op).
        void createDefaultMaterialTexture();

        // GPU skinning pipeline moved into vulkan/SkinningPipeline.{hpp,cpp}.
        // Renderer holds a unique_ptr `skinning_` and delegates allocation +
        // dispatch through it.

        // Water displace pipeline moved into vulkan/WaterDisplacePipeline.{hpp,cpp}.
        // Renderer holds a unique_ptr `waterDisplace_` and delegates allocation
        // + dispatch through it.

        // createPmremEnvImage moved into vulkan/EnvPrefilter.{hpp,cpp}.
        // Call envPrefilter_->buildPmrem(w, h, pixels, byteSize) instead.

        static VkSamplerAddressMode wrapToVk(TextureWrapping w) {
            switch (w) {
                case TextureWrapping::ClampToEdge:    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                case TextureWrapping::MirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                default:                              return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            }
        }

        // Upload `tex` to the next bindless slot if not already cached. Returns
        // the slot index; -1 if upload fails or capacity is exhausted.
        // uv2Ok: the caller samples it with the second UV set (the aoMap), so
        // a Texture.channel of 1 is honoured and not warned about.
        int32_t ensureMaterialTexture(const std::shared_ptr<Texture>& texSp, bool uv2Ok = false);

        // Build a tightly-packed RGBA8 sampled image from a material texture's
        // CPU image (BCn decompress / channel pad / float quantize). Returns a
        // null Image2D ({}) on unsupported/degenerate input. Slot-agnostic so the
        // initial upload (ensureMaterialTexture) and the in-place re-upload
        // (refreshDirtyMaterialTextures) share one code path.
        Image2D buildMaterialImage2D(const Texture* tex);
        // The same upload in its two halves. prepareMaterialImage is the CPU
        // work (the RGBA8 normalisation, the mip chain, the BC7 encode: most
        // of a new texture's cost) and touches no device state, so it can run
        // on a worker thread;
        // createPreparedImage is the device half (render thread).
        struct PreparedMaterialImage {
            bool ok = false;
            bool levelsPath = true;// every level built (BC7, odd sizes) vs RGBA8 with a blit chain
            uint32_t w = 0, h = 0;
            VkFormat format = VK_FORMAT_UNDEFINED;
            std::vector<std::vector<std::uint8_t>> levels;
            std::vector<unsigned char> rgba;
            VkSamplerAddressMode addrU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            VkSamplerAddressMode addrV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            std::string name;
        };
        PreparedMaterialImage prepareMaterialImage(const Texture* tex, bool bc7Ok) const;
        static PreparedMaterialImage preparedLevels(uint32_t w, uint32_t h, VkFormat fmt,
                                                    std::vector<std::vector<std::uint8_t>>&& levels,
                                                    VkSamplerAddressMode u, VkSamplerAddressMode v, const char* name);
        static PreparedMaterialImage preparedRgba(uint32_t w, uint32_t h, VkFormat fmt, std::vector<unsigned char>&& rgba,
                                                  VkSamplerAddressMode u, VkSamplerAddressMode v, const char* name);
        Image2D createPreparedImage(const PreparedMaterialImage& p);
        // ── Hidden admission off the edit frame ──────────────────────────────
        // A mesh that arrives hidden (resident, visible false) is neither drawn
        // nor traced until it shows, so its admission (the BLAS build, the
        // textures' encode and upload) need not be paid on the frame it was
        // added: its slot edit places it with default rows and starts its
        // textures' CPU half (prepareMaterialImage) on worker threads, and
        // pumpHiddenAdmissions, after each frame's scene walk, lands the spans
        // whose textures are ready, a few a frame: record, images, rows, and
        // a full TLAS build as for any first show. Nothing visible changes. A
        // mesh shown before its admission landed is placed again by that
        // frame's slot edit, with the synchronous admission a mesh arriving
        // visible gets (a texture still being prepared is waited for, not
        // encoded twice). THREEPP_VULKAN_SYNC_ADMIT=1 admits hidden meshes
        // synchronously too, and encodes a slot edit's textures one after the
        // other on the render thread (the A/B lane: the admission as it was).
        static bool syncAdmission();
        struct PreparingTexture {
            std::shared_ptr<Texture> tex;// held while the job runs
            unsigned int version = 0;
            std::future<PreparedMaterialImage> job;
        };
        std::unordered_map<const Texture*, PreparingTexture> preparingTextures_;
        uint32_t admitPendingSpans_ = 0;
        void beginHiddenAdmission(Mesh& m);
        // The lidar's entry -> stable id table (entryStableIds_) is written
        // read-only (stableIdIfAssigned) where entries are made, and a mesh's
        // auto id is assigned by the indirect draw builder the first time it
        // draws: an entry made before that reads 0. The rebuild used to heal it
        // at the next rebuild; the slot path never makes one, so every mesh
        // streamed in after the first frame reported lidar id 0. On a frame
        // that ends lean after entries were made or shown, this fills the
        // visible spans still at 0 from their meshes' assigned ids, until none
        // is left.
        void refreshPendingStableIds();
        bool stableIdsPending_ = false;
        // Start the CPU half of every texture of `m` that is not resident and
        // not already being prepared (worker threads); ensureMaterialTexture
        // takes the result. The slot edit calls it for all its new meshes
        // before building their descs, so their encodes run side by side.
        void prepareTexturesAsync(Mesh& m);
        // Every map buildMaterialDesc would admit for `m` (null where unset).
        static std::vector<std::shared_ptr<Texture>> texturesOf(const Mesh& m);
        void pumpHiddenAdmissions();

        // GL compressed-format enum (Image::compressedFormat, as the DDS
        // loader stores it) → Vulkan BC format. The COLOR-SPACE TAG on the
        // Texture wins over the GL enum's own sRGB-ness, matching the old
        // decode path (which uploaded RGBA8_SRGB/UNORM purely by tag): FBX
        // materials tag albedo sRGB and data maps linear regardless of how
        // the DDS was authored. UNDEFINED = no pass-through (e.g. BC6H).
        static VkFormat glCompressedToVk(unsigned int glFmt, bool srgb);

        // BC7 sampled-image support, queried once. Universal on desktop GPUs;
        // the check exists for headless/software drivers (lavapipe et al).
        bool bc7SampledSupported() {
            if (bc7Supported_ < 0) {
                VkFormatProperties fp{};
                vkGetPhysicalDeviceFormatProperties(ctx->physicalDevice(),
                                                    VK_FORMAT_BC7_SRGB_BLOCK, &fp);
                bc7Supported_ = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? 1 : 0;
            }
            return bc7Supported_ == 1;
        }
        int bc7Supported_ = -1;

        // Re-upload any cached material texture whose Texture::version() changed
        // since last upload (e.g. a DataTexture edited via setData + needsUpdate),
        // in place at its existing bindless slot. ensureMaterialTexture caches by
        // pointer and the bindless image array is only rewritten on a full scene
        // rebuild, so without this a live texture edit would never reach the GPU.
        // Cheap version scan each frame; only drains + rewrites when dirty.
        void refreshDirtyMaterialTextures();

        // Generate a 64×64 blue-noise tile via void-and-cluster (Ulichney 1993).
        // One-time CPU work at startup; the resulting tile has:
        //   - Spatially smooth correlation between adjacent cells (silhouette
        //     stability — neighbors agree on object/background membership)
        //   - Globally well-distributed (no coherent shake possible)
        //   - Suppressed low-frequency content (variance lives in high
        //     frequencies that denoise + TAA absorb)
        //
        // ~50ms on a single core for the 64×64 case (M=4096). Acceptable as a
        // one-shot startup cost. The output is uint8 ranks normalized to
        // [0, 255]; shader maps via R8_UNORM sampler to [0, 1) jitter.
        std::vector<uint8_t> generateBlueNoiseTile_();

        // Generate the tile and upload as an R8_UNORM image. Called once at
        // ctor time (after createDescriptorPool) so the descriptor write below
        // can include it.
        void createBlueNoiseImage_();

        // Tileable foam detail tile — R = micro bubble brightness, G = ridged
        // "lace" filament pattern. All lattice frequencies are integers so
        // every octave wraps seamlessly across the tile edges. Channel content
        // matches the procedural noise it replaces in the foam shading:
        //   R ≈ vnoise·0.55 + vnoise(×2.3)·0.30 + vnoise(×5.1)·0.15 with
        //       0.154 m base features over the shader's 4 m world mapping;
        //   G ≈ 1 − 2·|nf − 0.5| (filament cores at nf = 0.5) with ~1.2 m
        //       cells over the 12 m mapping.
        std::vector<unsigned char> generateFoamDetailTile_(int res);

        void createFoamDetailImage_();

        // 1×1 R32F dummy ocean height used when no DisplacedMesh is in the
        // scene. deferred_shade.comp gates on oceanFineTileSize > 0 so the
        // sampler result is unread; the descriptor still needs a valid
        // view/sampler to keep the layout populated. Replaced (view only) with
        // the active ocean's cascade-2 height image via rewriteDeferredDescriptors().
        void createOceanFineDummy_();

        // 1×1 R32F dummy for binding 33 (world-space foam) when no
        // DisplacedMesh is in the scene. closest_hit gates on
        // pc.oceanFoamTileSize > 0 so the sampler result is unread; the
        // descriptor still needs a valid view/sampler.
        void createOceanFoamDummy_();

        // Detect the active env texture (scene.environment, falling back to
        // scene.background.texture()) and upload it if it differs from the
        // currently bound one. Returns true when descriptors must be rewritten.
        bool refreshEnvTextureFromScene(Object3D& scene);

        // Populate `infos` with the current bindless material-texture array,
        // padding any unused tail slots — and any slots reclaimed by prune —
        // with the slot-0 white default. Reclaimed slots are detected via a
        // null view (Image2D is reset to {} on destroy).
        template <std::size_t N>
        void fillMaterialTextureInfos(std::array<VkDescriptorImageInfo, N>& infos) {
            const VkImageView fallbackView = materialTextures[0].view;
            // The policy samplers (see the textureSampler_ member comment) —
            // the per-image samplers are deliberately not bound here. Slots
            // whose texture asked for ClampToEdge get the clamp flavour of
            // the SAME filter policy (materialTexClampUV_).
            const VkSampler samplerRepeat = materialSampler(false);
            const VkSampler samplerClamp  = materialSampler(true);
            for (std::size_t i = 0; i < N; ++i) {
                const bool hasSlot = i < materialTextures.size() && materialTextures[i].view;
                const bool clampUV = hasSlot && i < materialTexClampUV_.size() && materialTexClampUV_[i];
                infos[i].sampler     = clampUV ? samplerClamp : samplerRepeat;
                infos[i].imageView   = hasSlot ? materialTextures[i].view : fallbackView;
                infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }

        // Raster jitter gate — MUST mirror uploadRasterCameraUbo's
        // rasterJitterOn expression (the sampler policy keys off the same
        // condition the projection jitter does).
        [[nodiscard]] bool rasterJitterActive() const {
            return !eventCamReadsGbuf() && (useFsr() || useDlss() || gbufMsaaSamples_ <= 1);
        }
        // The sampler the material-texture bindings use this frame: AUTO
        // (16×, jittered or not) unless
        // setTextureAnisotropy forced a level, in the REPEAT (default) or
        // CLAMP_TO_EDGE wrap flavour. May lazily create the custom sampler
        // pair for forced levels other than 1/16. Implemented in
        // VulkanCoreTextures.cpp with the sampler creation code.
        [[nodiscard]] VkSampler materialSampler(bool clampUV);
        // Invalidate every descriptor write that binds material textures so
        // the next per-frame lazy rewrite picks up a sampler-policy change.
        // Same (fence-idle-safe) pattern refreshDirtyMaterialTextures uses.
        void markMaterialSamplerDirty() {
            deferredDescDirty_.fill(true);
            // Every view: the flag is per-view (it guards per-view sets).
            for (auto& v : views_) { v->rasterMatTexValid_.fill(0); v->shadowMatTexValid_.fill(0); }
        }
        void setTextureAnisotropy(float aniso);// 0 = auto; VulkanCoreTextures.cpp

        // Rebuild every resource sized to the deferred shade's render extent
        // — the gbuf / reservoir / GI ping-pongs, the denoiser + TAA images,
        // the raster G-buffer, the upscale intermediates — plus the
        // descriptor sets that bind them. Shared by swapchain recreation and
        // runtime renderScale changes. createReservoirImages resets
        // accumulation; the caller must have the GPU idle before this runs.
        void reallocateRenderExtentResources();

        void recreateSwapchainAndDescriptors();

        // Runtime render-scale change. Reallocates every render-extent
        // resource and resets accumulation. Issues a vkDeviceWaitIdle.
        // Safe to call from inside the user's animate lambda — when a
        // frame is already mid-record (frameState_ != Idle) the
        // reallocation is deferred to the next beginDeferredFrame, after
        // that frame's fence has signalled. Otherwise (Idle) applies
        // immediately. No-op when the clamped value already matches.
        void setRenderScale(float scale);

        // Raster G-buffer MSAA sample count. 1 (default) = today's single-
        // sample path, byte-identical output. 2/4 rasterize the G-buffer at
        // that sample count and resolve with a dominant-sample pick (see
        // GbufResolve) instead of a box/average resolve — averaging normals/
        // ids/depth across a silhouette produces nonsense; picking the
        // majority-covering surface is what a G-buffer needs. Reallocates
        // render-extent resources (render pass + pipelines + MS images),
        // same idle-gate as setRenderScale — safe to call from inside the
        // user's animate lambda.
        void setGbufferMsaa(uint32_t samples);

        [[nodiscard]] uint32_t gbufferMsaa() const { return gbufMsaaSamples_; }

        // VulkanRenderer::setProbeCascades. Growing the probe stores rewrites
        // the deferred sets of every view, so it shares setRenderScale's idle
        // gate: mid-frame the request waits for the next beginDeferredFrame.
        void setProbeCascades(uint32_t count);
        void applyProbeCascades(uint32_t count);// device idle

        // Whether the DEVICE can rasterize + sample the G-buffer at this
        // count. The MS G-buffer spans color, depth AND integer (ids) formats,
        // each used as both attachment and sampled image, so the usable counts
        // are the intersection of all five limit masks. The spec only
        // guarantees 1x and 4x in each of them — 2x is optional (lavapipe
        // advertises exactly 1|4), and vkCreateImage with an unsupported
        // sample count is a spec violation
        // (VUID-VkImageCreateInfo-samples-02258 family) and a crash risk, not
        // a graceful fallback. Callers that want N>1 fall back to 4, the
        // mandatory multisample count. Same clamping idea as overlaySamples()
        // above.
        [[nodiscard]] bool gbufMsaaCountSupported(uint32_t samples) const {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            const auto& lim = props.limits;
            const VkSampleCountFlags mask = lim.framebufferColorSampleCounts &
                                            lim.framebufferDepthSampleCounts &
                                            lim.sampledImageColorSampleCounts &
                                            lim.sampledImageDepthSampleCounts &
                                            lim.sampledImageIntegerSampleCounts;
            // VK_SAMPLE_COUNT_N_BIT == N numerically (see overlaySamples).
            return (mask & static_cast<VkSampleCountFlags>(samples)) != 0;
        }

        // Gaussian-splat depth AOV. OFF by default, and off means the backing
        // image is one texel: a full-res r32f per frame in flight is ~25 MB at
        // 1080p that a scene with no splats would never read.
        // Crossing Off reallocates the render-extent resources, exactly like
        // setGbufferMsaa — a setup knob, not a per-frame one. Expected <->
        // Median is a UBO flag and reallocates nothing.
        using SplatDepthMode = VulkanRenderer::SplatDepthMode;
        void setSplatDepthAov(SplatDepthMode mode);

        // What the APP asked for — the answer VulkanRenderer::splatDepthAov()
        // and the readGBufferAOV(SplatDepth) gate hand back, unchanged by the
        // renderer's own use of the same image below.
        [[nodiscard]] bool splatDepthAov() const { return splatDepthMode_ != SplatDepthMode::Off; }
        // Whether the AOV image is REAL (full-res, written) this frame.
        // splatOverlayDepth_ is the renderer's own reason to want it — the
        // overlay depth stamp reads it — and it feeds the same image, so every
        // allocate/clear/write decision has to consult both. A caller who
        // never asked still gets "you never asked" from the public getter.
        [[nodiscard]] bool splatDepthAovAllocated() const {
            return splatDepthAov() || splatOverlayDepth_;
        }
        // The statistic the raster exports. Median is the front of the cloud,
        // which is what an occlusion test wants; the expected value sits a
        // cloud-thickness behind it. So when the AOV exists ONLY for the stamp
        // the mode is Median, and when the app asked for it the app's choice
        // wins — one image, and a sensor's answer is not ours to redefine.
        [[nodiscard]] bool splatDepthMedian() const {
            return splatDepthMode_ == SplatDepthMode::Median ||
                   (splatDepthMode_ == SplatDepthMode::Off && splatOverlayDepth_);
        }
        [[nodiscard]] SplatDepthMode splatDepthAovMode() const { return splatDepthMode_; }
        SplatDepthMode splatDepthMode_ = SplatDepthMode::Off;

        // ── Auto-exposure state ───────────────────────────────────────────────
        std::unique_ptr<vulkan::AutoExposure> autoExposure_;
        bool   autoExposureEnabled_ = false;
        float  autoExpSpeed_        = 2.0f;
        float  autoExpMinEV_        = -3.0f;
        float  autoExpMaxEV_        =  3.0f;

        // MSAA dispatch B (per-sample shading at complex/edge pixels) master
        // switch. OFF by default: measured on rock_flicker (msaa=4), dispatch
        // B's fallback variant (single largest non-dominant cluster, cheap
        // env-only diffuse — no RT gather/reflections/ReSTIR, to keep cost
        // proportional to edge-pixel count) made the static-camera flicker
        // metric WORSE (mean ~4600-4800 px/frame) than Phase 1 (dominant-
        // sample resolve) alone (~3700-3900), which was itself already below
        // the msaa=1 baseline (~4000-4030). Root cause: dispatch B's cheap
        // shading model disagrees with dispatch A's full model enough that
        // the disagreement is itself a new, comparably-sized noise source at
        // every edge pixel — trading one flicker mechanism for another
        // instead of removing it. The code compiles, is wired end-to-end,
        // and is architecturally sound (see deferred_shade.comp's shadeMode
        // branches) for a future attempt with a closer-matching per-sample
        // shading model; it just isn't a net win yet, so it stays off.
        // ON by default: under the UNJITTERED msaa raster (see
        // uploadRasterCameraUbo) dispatch B no longer regresses temporal
        // stability (static 23 vs 11 px/frame on the rock harness), and with
        // the corrected coverage accounting (dispatch A blends sky-minority
        // coverage itself; B fills the geometry-minority weight it reserves)
        // it supplies the spatial edge AA the dominant-pick resolve alone
        // lacks. Only consulted when gbufMsaaSamples_ > 1.
        bool gbufShadeBEnabled_ = true;

        // Deferred scene dispatch: shades the raster material G-buffer into
        // bloom_->sceneHdr (direct analytic lights + split-sum specular IBL +
        // approximate diffuse IBL + ray-query accents). Adds its passes to
        // the frame graph between the G-buffer/AS head and the splats + post
        // tail (recordCommandBuffer); defined in VulkanRenderer.cpp.
        // The async-compute pilot builds it in two calls: AsyncOnly (probeGI,
        // clusterBuild, cloudShadow, froxels, into computeGraph_) and
        // SkipAsync (everything else, into the frame graph).
        enum class DispatchPart { All, AsyncOnly, SkipAsync };
        void addSceneDispatchPasses(vulkan::rg::RenderGraph& g, DispatchPart part = DispatchPart::All);

        // Called once after bloom_->createImages(); wires sceneHdr views.
        void onAfterBloomCreateImages() {
            // Auto-exposure meters the PRIMARY's sceneHdr — it drives the
            // display's exposure, and a secondary must not re-point it at its
            // own (differently exposed, differently sized) image.
            if (!autoExposure_ || view().secondary) return;
            VkImageView views[kFramesInFlight];
            for (uint32_t f = 0; f < kFramesInFlight; ++f)
                views[f] = view().bloom_->sceneHdrView(f);
            autoExposure_->rewriteDescriptors(views);
        }

        // CPU per-frame tick: lazy-init, then read histogram + advance EMA.
        void onBeginDeferredFrame(uint32_t frame, float dt) {
            if (!autoExposureEnabled_) return;
            if (!autoExposure_) {
                autoExposure_ = std::make_unique<vulkan::AutoExposure>(*ctx, kFramesInFlight);
                autoExposure_->adaptSpeed = autoExpSpeed_;
                autoExposure_->minEV      = autoExpMinEV_;
                autoExposure_->maxEV      = autoExpMaxEV_;
                VkImageView views[kFramesInFlight];
                for (uint32_t f = 0; f < kFramesInFlight; ++f)
                    views[f] = view().bloom_->sceneHdrView(f);
                autoExposure_->rewriteDescriptors(views);
            }
            // Physical camera: the EMA adapts an EV COMPENSATION around the
            // EV100-derived exposure instead of an absolute multiplier
            // around 1.0 (the histogram/EMA machinery is reused unchanged).
            autoExposure_->baseExposure = physicalCamera_ ? physicalExposure() : 1.f;
            autoExposure_->tick(frame, dt);
        }

        // Exposure value used for this frame's composite push constant.
        // Auto-exposure wins when active (AutoExposure::exposure() composes
        // the physical-camera base it was handed in onBeginDeferredFrame);
        // otherwise physical camera mode derives it from aperture/shutter/ISO.
        [[nodiscard]] float currentExposure() const {
            if (autoExposureEnabled_ && autoExposure_)
                return autoExposure_->exposure();
            return physicalCamera_ ? physicalExposure() : toneMappingExposure_;
        }

        // HDRI sun extraction. refreshEnvTextureFromScene detects the env's
        // dominant compact bright source, prefilters PMREM mips 1+ from a
        // sun-clamped copy (no glossy / rough env lookup ever integrates the
        // raw ~10⁴:1 disc — the "bright spec blobs in reflections" artifact),
        // and updateLightsUbo re-injects the removed energy as an analytic
        // directional light (sharp correct sun highlight, jittered soft RT
        // shadows, GI bounce, water glints).
        //
        // ONE-SUN POLICY (consulted only when extraction is wanted): Auto
        // injects the extracted sun only while the scene has NO visible
        // DirectionalLight of its own — an explicit scene light claims the sun
        // role (scenes authored for raster renderers carry a stand-in sun light
        // because raster can't shadow from an env map; injecting the env sun on
        // top lights and shadows the scene with TWO suns). Always injects
        // regardless; Off skips extraction entirely.
        VulkanRenderer::EnvSunPolicy envSunPolicy_ = VulkanRenderer::EnvSunPolicy::Auto;
        [[nodiscard]] bool envSunExtractionWanted() const {
            return envSunPolicy_ != VulkanRenderer::EnvSunPolicy::Off;
        }
        [[nodiscard]] bool envSunDefersToSceneSun() const {
            return envSunPolicy_ == VulkanRenderer::EnvSunPolicy::Auto;
        }

        // ── Shadows (setShadowMode; VulkanCoreShadows.cpp) ─────────────────
        // The request as set (THREEPP_VK_SHADOW_MODE seeds it) and the knobs of
        // the map mode; activeShadowMode() resolves Auto, and RayTraced on a
        // device without ray query, to what runs.
        VulkanRenderer::ShadowMode shadowModeReq_ = shadowModeFromEnv();
        float    shadowDistance_   = 0.f;// 0 = the camera's far plane, clamped
        // THREEPP_VK_SHADOW_CASCADES=<1..4> and THREEPP_VK_SHADOW_LAMPS=<spots>,<points>
        // seed these (ablation knobs for the map pass's cost); the setters win.
        uint32_t shadowCascades_   = shadowCascadesFromEnv();
        uint32_t shadowedSpots_    = shadowedLampsFromEnv(0);
        uint32_t shadowedPoints_   = shadowedLampsFromEnv(1);
        uint32_t shadowMapSizeCap_ = shadowMapSizeCapFromEnv();// THREEPP_VK_SHADOW_MAPSIZE, 0 = none
        static uint32_t shadowCascadesFromEnv();
        static uint32_t shadowedLampsFromEnv(int which);
        static VulkanRenderer::ShadowMode shadowModeFromEnv();
        static uint32_t shadowMapSizeCapFromEnv();
        [[nodiscard]] VulkanRenderer::ShadowMode activeShadowMode() const;
        [[nodiscard]] bool shadowMapsActive() const {
            return activeShadowMode() == VulkanRenderer::ShadowMode::ShadowMaps;
        }
        // Bindings 80-83 of every view's deferred set when the view has no
        // live maps: a 1x1 D32 2D array (cascades and spots), a 1x1 D32 cube
        // array (points), both resting in DEPTH_STENCIL_READ_ONLY, and a zeroed
        // GpuShadowMapsUbo (flags 0: every lookup answers "no map").
        vulkan::ShadowDepthArray shadowDummyArray_{};
        vulkan::ShadowDepthArray shadowDummyCube_{};
        Buffer                   shadowDummyUbo_{};
        void ensureShadowDummies();
        void destroyShadowResources();
        // What bindings 80-83 of the current view's deferred set slot `frame`
        // should name now: the view's maps while its pass is live, else the
        // dummies.
        struct ShadowBinding {
            VkImageView cascades = VK_NULL_HANDLE;
            VkImageView spots    = VK_NULL_HANDLE;
            VkImageView points   = VK_NULL_HANDLE;
            VkBuffer    ubo      = VK_NULL_HANDLE;
            uint64_t    gen      = 0;// 0 = the dummies
        };
        [[nodiscard]] ShadowBinding shadowBindingFor(uint32_t frame) const;
        // Per view, per frame, after the slot's fence: rewrite bindings 80-83
        // of the view's deferred set slot `frame` when they name something
        // else than shadowBindingFor(frame) says.
        void syncShadowDescriptors(uint32_t frame);
        // Per view, after its cull and G-buffer draw build: decide whether the
        // view draws shadow maps this frame (view().shadowLive) and, if so, fit
        // them to `camera`, build the caster list and fill the UBO; then sync
        // the deferred set's bindings 80-83.
        void prepareShadowMaps(Camera& camera);
        // The fit and fill half of prepareShadowMaps; false when the view has
        // nothing to draw a map for (no sun, no slotted lamp).
        bool fitShadowMaps(Camera& camera);
        // The caster list of the current view's maps: every entry a shadow
        // ray could hit (shadowOccludes' set: no blend, no transmission, no
        // strong emitter, no viewmodel), kept when its world sphere meets a
        // cascade's box or a slotted lamp's range. Same DrawInfo records as
        // the G-buffer's.
        // One volume per layer of ShadowMapPass::layers, in the same order:
        // a cascade's box (centre, the sun's frame, half extent; every box
        // shares the frame), a spot's cone (the lamp, its axis, tan of the
        // half angle, far) or a point lamp's cube face (the lamp, the face's
        // axis, far). A layer the view cannot see is `skip`: cleared, drawn
        // into by nothing.
        struct ShadowCullVolumes {
            struct Volume {
                enum Kind : uint32_t { Box, Cone, Face };
                Kind  kind = Box;
                float c[3]{};
                float x[3]{}, y[3]{}, l[3]{};
                float d[3]{};
                float r       = 0.f;
                float tanHalf = 0.f;
                bool  skip    = false;
                bool  cached  = false;// a cascade served from the static cache: only moving casters reach it
            };
            std::vector<Volume> layers;
        };
        // The static caches of the cascades and the lamp layers (ShadowMapPass::cache, spotCache, pointCache):
        // THREEPP_VK_SHADOW_CACHE=0 turns it off; THREEPP_VK_SHADOW_CACHE_MARGIN
        // (1..2, default 1.15) is how much wider than the slice a cascade's box
        // is frozen: 1 = the slice itself, so the image is the uncached one but
        // a camera that moves redraws every frame (the twin's drone at 30 m/s:
        // 5 ms); 1.15 = a box 15 % wider, kept while the camera stays inside
        // it, at texels 15 % coarser (0.4 ms, a 2-4 ms rebuild frame every
        // second or so). A caster is static unless it is a deformer, an interop
        // mesh, grass, or recently moved (meshMovedSticky_); the hash of that
        // set is what a frozen box is valid for.
        bool  shadowCacheOn_     = shadowCacheFromEnv();
        float shadowCacheMargin_ = shadowCacheMarginFromEnv();
        static bool  shadowCacheFromEnv();
        static float shadowCacheMarginFromEnv();
        [[nodiscard]] bool     shadowCasterDynamic(uint32_t entry) const;
        [[nodiscard]] uint64_t shadowStaticHash();
        // The hash walk's by-products: the moving casters' entry indices in
        // span order, and each entry's span. In steady state (every layer
        // skipped or served from its cache) buildShadowDrawData walks only
        // those; the caches hold the static casters.
        std::vector<uint32_t> shadowDynamicIdx_, shadowSpanOfEntry_;
        // The static set as a generation: +1 whenever shadowStaticHash changes.
        // A cached layer is valid for one generation, never for a hash value:
        // a layer unseen while a mesh moves and settles would meet its old
        // hash again with the mesh at its old place.
        uint64_t shadowStaticHashPrev_ = 0, shadowStaticGen_ = 1;
        // THREEPP_VK_SHADOW_CACHE_DEBUG=1: a line per generation step (what
        // flipped) and per frame (layers cached / rebuilt / drawn / skipped).
        bool                 shadowCacheDebug_ = vulkan::env::flag("THREEPP_VK_SHADOW_CACHE_DEBUG");
        std::vector<uint8_t> shadowDynPrev_;
        void buildShadowDrawData(uint32_t frame, const ShadowCullVolumes& vol, const Vector3& camPos);
        // setShadowCacheStaleFrames: 0 (default) redraws a cascade whenever
        // the static caster set changes; n > 0 lets it hold stale casters for
        // up to n frames (ShadowMapPass::casterSnap). updateShadowCasterSnap
        // clears the bits of the casters that left the static set, marks the
        // cascades they leave stale and lists ShadowMapPass::onTop.
        uint32_t shadowCacheStaleFrames_ = 0;
        void     updateShadowCasterSnap(vulkan::ShadowMapPass& smp, uint32_t cascades);
        // buildShadowDrawData's scratch: every caster once, then one command
        // list per (layer, group); cleared, never reallocated, between frames.
        std::vector<DrawInfoGpu>                        shadowDrawScratch_;
        std::vector<std::vector<VkDrawIndirectCommand>> shadowCmdScratch_;
        // Depth-only, reverse-Z, dynamic rendering into one D32 layer: the
        // indirect G-buffer draw reduced to position + the cutout test, with
        // depth clamp where the device has it and dynamic cull mode, front
        // face and depth bias.
        VkPipeline shadowPipeline_      = VK_NULL_HANDLE;// casters through shadow_depth.frag's alpha test
        VkPipeline shadowDepthPipeline_ = VK_NULL_HANDLE;// the rest: no fragment stage
        void createShadowPipeline();
        // Dir light 0 as updateLightsUbo packed it, with its LightShadow
        // settings (the environment's sun: the defaults, 2048 texels).
        struct ShadowSun {
            bool     valid = false;
            float    dir[3]{0.f, 1.f, 0.f};// toward the sun
            float    bias = 0.f, normalBias = 0.f, radius = 1.f, intensity = 1.f;
            uint32_t mapSize = 2048u;
        };
        ShadowSun shadowSun_;
        // The lamps holding a shadow map this frame, by slot (index = the
        // layer in the spot array / the cube in the point array), chosen from
        // the primary camera in updateLightsUbo (assignShadowSlots).
        struct ShadowLamp {
            bool         used = false;
            unsigned int id   = 0;// Object3D::id of the light
            float        pos[3]{}, dir[3]{};
            float        fov = 0.f;// spot: full cone angle of the map (rad)
            float        far = 1.f;
            float        bias = 0.f, normalBias = 0.f, radius = 1.f, intensity = 1.f;
        };
        std::vector<ShadowLamp> shadowSpotSlots_, shadowPointSlots_;
        // Writes GpuClusterLight::shadowSlot into `recs` (src: each record's
        // light) and refreshes the two slot tables; -1 everywhere when the maps
        // are not in use.
        void assignShadowSlots(std::vector<GpuClusterLight>& recs, const std::vector<Light*>& src, Camera& camera);

        // Frame recording, split along the frame's own stage seams (bodies in
        // VulkanCoreRecord.cpp). recordCommandBuffer is the narrative: it adds
        // each stage's passes to the frame graph in order, and the graph
        // derives the barriers between them from what each pass declares.
        void updatePaneRegion();
        // The frame's head, as graph passes ahead of the deformers: the GPU
        // instance expansion, the ParticleField emitter (+ surface bake), the
        // interop snapshot and indirect-count copies, and the density scatter.
        // Once for all views; every view's G-buffer and billboard draws, the
        // froxels and the shade read what they write.
        void addHeadPasses(vulkan::rg::RenderGraph& g);
        // Every ParticleField's position ring, prevPositions, counts block,
        // surface-bake map and draw-indirect records: what the emitter, the
        // interop snapshot and the counts copy write, and the density scatter,
        // the G-buffer / billboard vertex stages and the transmittance prepass
        // read through buffer device addresses and indirect draws.
        static constexpr const char* kParticleFields = "particles.fields";
        // A pass that reads kParticleFields: `draws` = the G-buffer and
        // billboard draws (indirect records + vertex pulls), otherwise the
        // transmittance prepass (compute). No-op without a ParticleField pass.
        void useParticleFields(vulkan::rg::RenderGraph& g, vulkan::rg::PassBuilder& pass, bool draws);
        // The deformers (dynamic plain meshes, skinned, tet, displaced water,
        // grass) and the per-frame TLAS refit, as graph passes. Each deformer
        // pass keeps the barriers between its own copy / dispatch / BLAS build;
        // all of them write kSceneGeometry.
        void addDeformAndTlasPasses(vulkan::rg::RenderGraph& g);
        void recordSkinnedDeforms(VkCommandBuffer cb);
        void recordTetDeforms(VkCommandBuffer cb);
        // Every deformable mesh's vertex / normal / prevVertex / index buffers,
        // its BLAS, and the ocean's FFT outputs: the graph memory resource
        // (RenderGraph::importMemory) the deformers write and the TLAS refit,
        // the raster passes and every ray query read. Ray queries declare it
        // through the TLAS's registration (registerAccelerationStructure).
        static constexpr const char* kSceneGeometry = "scene.geometry";
        // The G-buffer stage as graph passes (VulkanCoreRecord.cpp).
        void addGbufferPasses(vulkan::rg::RenderGraph& g);
        void recordGbufferRaster(VkCommandBuffer cb);
        void recordGbufferMsaaResolve(VkCommandBuffer cb);
        void recordOverlayDepthPrepass(VkCommandBuffer cb);
        [[nodiscard]] bool gbufferMsaaResolveActive();
        [[nodiscard]] bool overlayDepthPrepassActive();
        // The import form of the prepass: the descriptor refresh (graph-build
        // time, this slot's post-fence window) and the draw (inside the
        // prepass's render-pass instance).
        void prepareOverlayDepthImport();
        void recordOverlayDepthImport(VkCommandBuffer cb);
        // The overlay pass's entry walks, per span (overlaySpanEntries): the
        // indices of the entries whose span is flagged isOverlay (or, with
        // `particle`, isParticle), in entry order. One scratch, reused.
        void overlaySpanEntries(bool particle);
        std::vector<uint32_t> overlayEntryScratch_;
        // Events-only mode: the swapchain cleared to black (the canvas the
        // event accumulator's sprite overlay draws on), as a graph pass.
        void addEventsOnlyClearPass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // The acquired swapchain image, imported UNDEFINED (the entry
        // barrier's ALL_COMMANDS source chains to the acquire wait) with exit
        // layout GENERAL, where the post-graph overlay and endFrame expect it.
        // Called before any pass names it, so these layouts are the ones the
        // graph keeps.
        vulkan::rg::ImageHandle importSwapchain(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // importSwapchain plus the split-screen clear.
        void addSwapchainPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // Depth of field, bloom, the temporal resolve / upscaler and the post
        // composite, as passes of the frame graph.
        // `secondary`: a secondary view's reduced chain — no DoF, no DLSS/FSR,
        // no sharpening or motion blur, the authored TAA alpha, and its own
        // colour target in place of the swapchain.
        void addUpscaleAndPostPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex,
                                     VkExtent2D ext, VkExtent2D ptExt,
                                     uint32_t exposureBits, float preExp,
                                     bool secondary = false);
        // The frame's render graph: every view's passes and the record tail,
        // built in full and then executed once (beginDeferredFrame).
        vulkan::rg::RenderGraph frameGraph_;

        // ── The async-compute pilot (THREEPP_VK_ASYNC_PILOT=1) ─────────────
        // plans/vulkan-cpu-and-async-compute-2026-10.md, item 6. With the knob
        // and a compute-only queue family (ctx->asyncComputeQueue()), a full
        // frame goes out as four command buffers:
        //   graphics 1  frame graph: head, deformers (the ocean WITHOUT its
        //               BLAS rebuild), "async.join"; signals S1
        //   graphics 2  frame graph: the G-buffer passes (no wait)
        //   compute     computeGraph_: ocean BLAS rebuild, tlas, probeGI,
        //               clusterBuild, cloudShadow, froxels; waits S1, signals S2
        //   graphics 3  frame graph: the rest, every secondary view, the tail,
        //               the overlay; waits S2, carries the fence and present
        // so the compute work can run under the G-buffer raster. "async.join"
        // is a pass with no commands that declares everything computeGraph_
        // touches (RenderGraph::declareAsPass): its barriers bring those
        // resources into the compute graph's layouts in graphics 1, and order
        // the frame graph's later uses after it. Semaphore rules: a signal
        // covers every command earlier in its queue's submission order and a
        // wait every command later, so compute(N+1) waiting S1(N+1) is behind
        // graphics 3 of frame N (the previous frame's TLAS and probe readers),
        // and graphics 3 waiting S2 is behind compute(N). Off (the default) or
        // without such a family nothing here is created or used.
        vulkan::rg::RenderGraph computeGraph_;
        bool asyncPilot_ = false;// ctx->asyncComputeQueue(), fixed at createCommandResources
        // What moves to the compute queue (THREEPP_VK_ASYNC_PILOT: =1 both,
        // =3 the dispatches only, =4 nothing; see VulkanContext).
        bool asyncMoveAs_ = false;      // ocean BLAS rebuild + TLAS refit
        bool asyncMoveDispatch_ = false;// probeGI, clusterBuild, cloudShadow, froxels
        bool asyncFrame_ = false;// this frame took the four-command-buffer path
        uint32_t asyncSplits_[2] = {0, 0};// first frame-graph pass of graphics 2 and 3
        VkCommandPool computeCmdPool_ = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, kFramesInFlight> computeCmdBuffers_{};
        std::array<VkCommandBuffer, kFramesInFlight> asyncGbufCmdBuffers_{};
        std::array<VkCommandBuffer, kFramesInFlight> asyncTailCmdBuffers_{};
        std::array<VkSemaphore, kFramesInFlight> asyncHeadDone_{};   // S1
        std::array<VkSemaphore, kFramesInFlight> asyncComputeDone_{};// S2
        // The command buffer this frame's recording continues in after the
        // graph (overlay, HUD render() calls, the present transition).
        VkCommandBuffer frameCb() const {
            return asyncFrame_ ? asyncTailCmdBuffers_[currentFrame] : cmdBuffers[currentFrame];
        }
        void destroyAsyncPilotResources();

        // ── Several views, one graph ────────────────────────────────────────
        // Every view's passes are BUILT before any is RECORDED, so the CPU
        // state a view's pass callbacks read at record time — the current
        // view, the pane region, the draw lists buildIndirectDrawData made for
        // its camera, the camera-derived scalars updateCameraUbo and
        // uploadRasterCameraUbo leave in Impl members — would otherwise be the
        // LAST view's. Each view's passes therefore open with a "view" pass
        // (no resource uses) that applies what that view's build left behind,
        // saved into viewRecordStates_[its index in views_] once its passes
        // are built. The record tail opens with the primary's again.
        struct ViewRecordState {
            ViewContext* view = nullptr;
            VkExtent2D   regionRenderExt{}, regionSwapExt{};
            int32_t      regionDstX = 0, regionDstY = 0;
            std::array<DrawGroup, 4>      indirectGroups{};
            uint32_t                      indirectTotalDraws = 0;
            std::vector<ParticleDrawSlot> particleDrawSlots;
            bool  occlActive  = false;
            float tanHalfFovY = 0.f, filmHeightM = 0.f;
            float projP0 = 0.f, projP5 = 0.f, projP8 = 0.f, projP9 = 0.f;
#if defined(THREEPP_WITH_FSR)
            float fsrJitterX = 0.f, fsrJitterY = 0.f;
            float fsrCamNear = 0.f, fsrCamFar = 0.f, fsrCamFovY = 0.f;
#endif
#if defined(THREEPP_WITH_DLSS)
            float dlssJitterX = 0.f, dlssJitterY = 0.f;
#endif
            float deferredCamRotAngle = 0.f;
        };
        std::vector<ViewRecordState> viewRecordStates_;
        void saveViewRecordState(ViewRecordState& s);
        // Also silences the GPU timestamps for a secondary view: one query
        // pool per frame-in-flight, one slot pair per pass, and a secondary
        // re-running those passes would overwrite the primary's.
        void applyViewRecordState(const ViewRecordState& s);
        // The "view" pass that applies viewRecordStates_[index] at record time.
        void addViewEnterPass(vulkan::rg::RenderGraph& g, size_t index);
        // Execute a graph; with THREEPP_RG_DUMP=1 also print its passes and
        // barriers to stderr whenever they differ from what was last printed
        // under the same name (a pass appearing, a layout changing).
        void executeGraph(VkCommandBuffer cb, vulkan::rg::RenderGraph& graph, const char* name) {
            executeGraph(&cb, nullptr, 0, graph, name);
        }
        // The same over consecutive command buffers of one queue
        // (RenderGraph::execute with splits; the async-compute pilot).
        void executeGraph(const VkCommandBuffer* cbs, const uint32_t* splits, uint32_t splitCount,
                          vulkan::rg::RenderGraph& graph, const char* name) {
            // Which imported images share memory (TransientPool); the graph
            // orders the hand-over between them and checks their lifetimes.
            if (transientPool_) transientPool_->declare(graph);
            if (gbufMsPool_) gbufMsPool_->declare(graph);
            if (splitCount == 0) graph.execute(cbs[0]);
            else graph.execute(cbs, splits, splitCount);
            // Diagnostics (two-layout declarations, overlapping aliases): every
            // occurrence is counted, each distinct message printed and kept once.
            for (auto& e : graph.diagnostics()) {
                ++rgDiagnosticCount_;
                if (rgDiagnosticsReported_.insert(e).second) {
                    std::fprintf(stderr, "[RenderGraph] '%s': %s\n", name, e.c_str());
                    rgDiagnostics_.push_back(std::move(e));
                }
            }
            static const bool dump = vulkan::env::flag("THREEPP_RG_DUMP");
            if (!dump) return;
            std::string text = graph.dump();
            auto& last = rgDumped_[name];
            if (text == last) return;
            std::fprintf(stderr, "[RenderGraph] graph '%s' (frame serial %llu):\n%s", name,
                         static_cast<unsigned long long>(frameSerial_), text.c_str());
            last = std::move(text);
        }
        std::map<std::string, std::string> rgDumped_;
        std::set<std::string> rgDiagnosticsReported_;
        std::vector<std::string> rgDiagnostics_;// distinct, in the order first seen
        std::uint32_t rgDiagnosticCount_ = 0;   // VulkanRenderer::renderGraphDiagnosticCount()
        // The primary's tail as graph passes: the field transmittance prepass,
        // the billboard glow, the hybrid overlay and the sensor image stage.
        void addTailPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // A secondary view's share of the tail: its field billboards (and their
        // transmittance prepass, for its own eye) onto its colour target.
        void addSecondaryFieldBillboardPasses(vulkan::rg::RenderGraph& g);
        [[nodiscard]] bool hybridOverlayActive() const {
            return overlayWireframePipeline != VK_NULL_HANDLE && sceneHasOverlayContent();
        }
        [[nodiscard]] bool fieldBillboardGlowActive() const;
        // The overlay depth attachment: the MSAA overlay depth, or unjitDepth.
        [[nodiscard]] const Image2D& overlayDepthImage();
        void recordHybridOverlay(VkCommandBuffer cb, uint32_t imageIndex, bool stampActive);
        // The primary view's passes, from the frame head to the sensor stage
        // (or, in the hybrid debug view / events-only mode, the head, the
        // G-buffer and that mode's swapchain pass).
        void addPrimaryViewPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);

        // Records the ImGui (or any overlay) callback inside a dynamic render
        // pass, then transitions the swapchain image GENERAL → PRESENT_SRC.
        // When no overlay callback is set, just emits the GENERAL → PRESENT_SRC
        // barrier directly. Called from endFrame() immediately before
        // vkEndCommandBuffer + submit.
        // The frame's record tail after the secondary views, as one graph:
        // scene capture, frame interop, the event camera, the view composite.
        // The swapchain enters and leaves it in GENERAL, where the screen-space
        // sprite overlay, HUD render() calls and endFrame's ImGui + present
        // transition (all self-synchronising, outside any graph) expect it.
        void addPostViewTailPasses(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        // Snapshot the post-TAA swapchain image into the host-visible
        // sceneCaptureBuf_ before any overlay (sprites, ImGui) composites.
        // Allocates / resizes the buffer lazily on first use or swapchain
        // resize, at build time.
        void addSceneCapturePass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);

        // ── Event-camera shade compute ─────────────────────────────────
        // Creates the event_shade compute pipeline + descriptor set
        // layout. Called once, lazily, on first setEventCameraEnabled.
        void createEventShadePipeline();

        // (Re-)allocate eventLumaBuf_ at the swapchain dimensions.
        // Called from setEventCameraEnabled and on resize.
        void allocateEventLumaBuffer(uint32_t w, uint32_t h);

        // The event_shade dispatch and the detector, as two passes of the
        // post-view tail graph (after the gbuf prepass AND after the swapchain
        // holds the final frame). The event_shade set is rewritten at build
        // time, so its declaration (reflected) names this frame's gbuf images
        // (SHADER_READ_ONLY_OPTIMAL) and the acquired swapchain image
        // (GENERAL, the Final source's storage image).
        void addEventCameraPasses(vulkan::rg::RenderGraph& g, uint32_t frame, uint32_t imageIndex);

        void recordOverlayAndPresentTransition(VkCommandBuffer cb, uint32_t imageIndex);

        // Common cmd-buffer-begin epilogue used by both the deferred-render
        // and ortho-only frame starts: opens the command buffer and resets
        // the per-frame timestamp pool. Caller must have already acquired
        // the swap image, reset the fence, and reset the cmd buffer.
        void beginCommandRecording(VkCommandBuffer cb);

        // Acquire next swapchain image, run all per-frame UBO / descriptor
        // uploads, open the cmd buffer, and record the full deferred-render
        // body. On exit, the swapchain image is in GENERAL and the cmd
        // buffer is open; endFrame() finishes it (overlay + present-src +
        // submit + present). Returns false on swapchain OUT_OF_DATE —
        // caller leaves state Idle.
        bool beginDeferredFrame(Object3D& scene, Camera& camera);

        // ── Multi-view API (see ViewContext) ───────────────────────────────
        // addView stands up a COMPLETE second deferred chain — G-buffer,
        // history, camera UBOs, descriptor sets, its own TaaResolve /
        // BloomPass / PostComposite / DeferredShade, and the colour target it
        // resolves into. Eagerly, all of it: a view is a persistent object, so
        // the cost is paid once here rather than as a stall on the frame that
        // first happens to need it. Returns a handle, or 0 on failure.
        //
        // Views are NOT churned per frame. Adding one drains the device and
        // allocates; rendering an existing one costs a second pass over the
        // shared scene and nothing else.
        uint32_t addViewImpl(Camera& camera, uint32_t width, uint32_t height);
        bool     removeViewImpl(uint32_t handle);
        bool     setViewCameraImpl(uint32_t handle, Camera& camera);
        ViewContext* findView(uint32_t handle);
        // TAA switched off for the CURRENT view (setViewTaa). A secondary
        // never upscales, so its flag is final; the primary can only drop TAA
        // while no upscaler runs, because DLSS and FSR reconstruct from the
        // jitter. Read by both camera UBO uploads (the raster jitter gate)
        // and by the two resolve calls (blend alpha 1 = passthrough).
        [[nodiscard]] bool viewTaaOff() const {
            return !view().taaEnabled && (view().secondary || !(useFsr() || useDlss()));
        }
        // Allocate / free everything a SECONDARY view owns. Both assume the
        // device is idle.
        void createSecondaryViewResources(ViewContext& v);
        void destroySecondaryViewResources(ViewContext& v);
        // Drain, then service every view marked pendingCreate / pendingDestroy.
        // Called at the frame boundary; a no-op (and harmlessly retried next
        // frame) until the shared render pass exists.
        void applyPendingViewChanges();
        bool pendingViewChanges_ = false;
        // Every secondary view's chain, as passes after the primary's in the
        // frame graph. Shares the frame's TLAS/BLAS, lights, materials and
        // textures; re-runs only what is genuinely per-camera. Returns whether
        // any view added passes (the tail then re-enters the primary).
        bool addSecondaryViewPasses(vulkan::rg::RenderGraph& g);
        // Run `fn` once per LIVE view — the primary and every secondary that
        // already owns its resources — with curView_ pointed at it.
        //
        // The per-view helpers (rewriteDeferredDescriptors, clearGbufImages,
        // ...) all read view(), which made them silently primary-only at the
        // call sites that are actually SCENE-wide: a rebuilt TLAS, a swapped
        // material texture, a new environment map. A secondary left out of
        // those keeps a descriptor pointing at freed memory — it does not
        // crash, it just renders the world as it was before, or as nothing.
        //
        // Assumes the device is idle, or that the caller knows the slot it is
        // rewriting is fence-proven idle: same precondition the single-view
        // call always had.
        template<class F>
        void forEachLiveView(F&& fn) {
            ViewContext* saved = curView_;
            for (auto& v : views_) {
                if (v->pendingCreate || v->pendingDestroy) continue;
                curView_ = v.get();
                fn();
            }
            curView_ = saved;
        }
        // Copy every DISPLAYED secondary view's colour target into the primary's
        // swapchain image, at that view's rect. A pass after the secondaries
        // have resolved and after the scene capture (which must stay a clean
        // picture of the primary alone), and before the overlay, so ImGui and
        // sprites still draw on top.
        void addViewCompositePass(vulkan::rg::RenderGraph& g, uint32_t imageIndex);
        struct ViewCompositeJob {
            VkImage    src;
            VkOffset2D srcOff;
            VkOffset2D dstOff;
            VkExtent2D ext;
        };
        std::vector<ViewCompositeJob> viewCompositeJobs_;// this frame's, built by addViewCompositePass
        bool setViewDisplayRectImpl(uint32_t handle, int x, int y, int w, int h);
        // Per-view permission to rasterize sensor-only surfaces. Raster state
        // only — read while the draw list is built, so it takes effect on the
        // next frame with nothing to invalidate. The lidar's TLAS masks are a
        // separate, scene-level decision (sensorOnlySurfaces_).
        bool setViewSensorSurfacesImpl(uint32_t handle, bool enabled);
        bool setViewSplatsImpl(uint32_t handle, bool enabled);
        // Copy a secondary view's finished colour target to host memory as
        // tightly-packed RGB8, top-down (matching readRGBPixels — the Vulkan
        // readback is already top-down and must NOT be flipped).
        std::vector<unsigned char> readViewPixelsImpl(uint32_t handle);

        // Ortho-only frame start: no deferred render, no per-frame uploads.
        // Acquires the swap image, opens the cmd buffer, and clears the
        // swapchain to the configured clearColor (leaves layout = GENERAL).
        // recordOrthoOverlay() then draws the HUD scene atop this cleared image.
        bool beginFrameOrthoOnly();

        // Finish the open frame: record overlay/ImGui pass + PRESENT_SRC
        // transition, close the cmd buffer, submit and present, advance the
        // frame slot, bump sampleIndex. No-op when no frame is open.
        // Registered as Canvas frame-end callback (fires after the user's
        // animate lambda returns); also invoked from render() when the user
        // unexpectedly starts a second perspective render mid-frame.
        void endFrame();

        // State-machine dispatcher for VulkanRenderer::render(). First call
        // of a user animate iteration runs beginFrame*, subsequent calls
        // either append HUD overlay draws (ortho path) or finalize
        // the prior frame and restart (defensive — second perspective call).
        // endFrame() runs from the Canvas frame-end callback at the tail of
        // animateOnce().
        void renderFrame(Object3D& scene, Camera& camera);
    };

} // namespace threepp
