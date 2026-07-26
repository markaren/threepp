// Private implementation header — the shared CoreImpl for VulkanRenderer.cpp.
// Never include from public API headers.
// VMA_IMPLEMENTATION must be #defined in VulkanRenderer.cpp BEFORE including this file.
#pragma once

#include "VulkanContext.hpp"
#include "VulkanResources.hpp"
#include "VulkanRetireQueue.hpp"
#include "EnvPrefilter.hpp"
#include "EventCameraDetector.hpp"
#include "LidarScanner.hpp"
#include "SkinningPipeline.hpp"
#include "TetSkinningPipeline.hpp"
#include "GpuTimings.hpp"
#include "OverlayPass.hpp"
#include "VulkanFrameTypes.hpp"
#include "TaaResolve.hpp"
#if defined(THREEPP_WITH_FSR)
#include "FsrUpscaler.hpp"
#endif
#if defined(THREEPP_WITH_DLSS)
#include "DlssUpscaler.hpp"
#endif
#include "AutoExposure.hpp"
#include "GbufResolve.hpp"
#include "BloomPass.hpp"
#include "PostComposite.hpp"
#include "DofPass.hpp"
#include "DeferredShade.hpp"
#include "HiZPyramid.hpp"
#include "OcclusionCull.hpp"
#include "ProbeGI.hpp"
#include "WaterDisplacePipeline.hpp"
#include "FoamWorldPipeline.hpp"
#include "GrassWindPipeline.hpp"
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
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
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
#include <iostream>
#include <memory>
#include <mutex>
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

    namespace {
        // Frames-in-flight depth. Bumped from 2 → 3 to deepen CPU/GPU
        // pipelining: while frame N+2 is being recorded on the CPU, frame N
        // and frame N+1 can be in different stages of GPU execution. Hides
        // CPU jitter (scene-build, ImGui, frustum cull) without changing the
        // GPU schedule (queue is still serial — async compute would do that,
        // and is a much larger change).
        //
        // The 2-slot ping-pong (the ReSTIR DI reservoir images) stays at 2
        // entries — Vulkan queue execution is
        // strictly in-order within a queue, so when frame N+2 writes slot
        // (N+2)&1 the prior owner of that slot (frame N) has fully completed
        // on the GPU. Temporal reproject still reads "the previous frame"
        // because readSlot = 1 - writeSlot, which alternates correctly.
        //
        // MUST stay EVEN. The ping-pong slot is `currentFrame & 1`, and
        // currentFrame cycles mod kFramesInFlight. Only an even count makes
        // `currentFrame & 1` track the true monotonic frame parity, so the
        // slot actually alternates frame-to-frame. An ODD count (e.g. 3)
        // desyncs it: the write-slot sequence becomes 0,1,0,0,1,0,… so every
        // 3rd frame the temporal read samples a 2-frame-STALE slot while the
        // immediately-previous frame's output is overwritten unread —
        // corrupting accum/gbuf/moments/albedo/ReSTIR/TAA history on a 3-frame
        // beat (periodic ghosting + reprojection reading the wrong frame).
        // If a deeper pipeline is ever wanted, decouple the ping-pong parity
        // from this sync ring (drive the slot from a monotonic `++parity & 1`
        // and rewrite the temporal image bindings per frame) instead of
        // bumping this to an odd value.
        constexpr uint32_t kFramesInFlight = 2;
    }// namespace

    struct VulkanRendererCore::CoreImpl {
        Canvas& canvas;
        WindowSize size;
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

        // Per-geometry BLAS + the buffers that back its build inputs. Vertex /
        // index / normal / uv buffers are kept alive past the build so the
        // closest-hit shader can sample them via buffer-reference pointers.
        // `uv.handle == VK_NULL_HANDLE` for geometries without a UV attribute;
        // the matching GeometryDesc.uvAddress is then 0 and closest_hit treats
        // the surface as untextured.
        struct BlasRecord {
            VkAccelerationStructureKHR as = VK_NULL_HANDLE;
            Buffer storage;
            Buffer vertex;
            Buffer index;// .handle == VK_NULL_HANDLE for non-indexed geometry
            Buffer normal;
            Buffer uv;   // .handle == VK_NULL_HANDLE if geometry has no "uv"
            // Per-vertex RGB color (BufferGeometry "color" attribute, itemSize 3).
            // .handle == VK_NULL_HANDLE when the geometry has no usable "color".
            // Surfaced to the shaders via GeometryDesc::colorAddress / DrawInfo::
            // colorAddr, gated on the material's vertexColors flag at fill time.
            Buffer color;
            // Which of the optional attribute buffers hold PACKED data instead
            // of tightly-packed float (bit 0: normal snorm16x4 — 8 B/vertex,
            // bit 1: uv unorm16x2 — 4 B/vertex, bit 2: color unorm8x4 —
            // 4 B/vertex). Set by buildBlasFor for static geometry only;
            // deforming records (skinned / tet / displaced / grass / morphed)
            // stay float because their compute passes rewrite these buffers
            // every frame. Mirrored to GeometryDesc::packedAttrs and
            // DrawInfoGpu::packedAttrs; shaders branch per fetch.
            uint32_t packedMask = 0;
            // True for FFT-displaced ocean meshes — gates world-space foam +
            // thin-shell water shading downstream (surfaced to the shaders via
            // GeometryDesc::foamAddress, now a 0/1 flag). Replaces a per-vertex
            // foam buffer that used to live on the BLAS but was never read for
            // its contents — only its non-null address served as this marker.
            bool isOceanSurface = false;
            // Previous-frame vertex positions, allocated for skinned + displaced
            // meshes only. Used by the hybrid raster prepass to compute
            // per-vertex motion vectors (skinned/displaced surfaces deform
            // each frame; the rigid-body motionMat alone produces zero motion
            // and ghosts under TAA + the deferred shade's temporal
            // accumulation). Static meshes
            // bind .vertex at the prev-pos attribute slot — inPrevPos == inPos
            // so the motion is identity-rigid as before.
            Buffer prevVertex;
            VkDeviceAddress address = 0;
            // Liveness tag: detects dangling-pointer reuse when a BufferGeometry
            // is destroyed (model unloaded) and the C++ allocator hands the
            // same address to a different geometry. Pruned in ensureSceneBuilt.
            std::weak_ptr<BufferGeometry> liveCheck;
            // Attribute-version snapshot at build time. When the user mutates
            // vertex data in-place and calls needsUpdate(), the composite
            // version changes and the BLAS is refreshed (in-place rebuild if
            // counts match, full evict+rebuild if topology changed).
            unsigned int geomVersion = 0;
            uint32_t vertexCount = 0;
            uint32_t indexCount  = 0;
            // Persistent scratch buffer for per-frame refit. Sized to
            // buildScratchSize on first use (which is always >= updateScratchSize),
            // reused every frame to avoid the create+destroy pair that
            // dominated refreshGeomBlas's cost. Empty until the first
            // refreshGeomBlas; cleaned up in deinit / prune paths.
            Buffer blasScratch{};
            VkDeviceSize blasScratchSize = 0;
            // Per-frame BLAS-refit state — same pattern as SkinnedMeshState /
            // DisplacedMeshState. refreshGeomBlas refits via MODE_UPDATE for 63
            // of every 64 frames; the periodic full rebuild keeps the BVH
            // balanced under sustained geometry mutation (PhysX soft bodies,
            // dynamic ParticleSystems).
            uint32_t blasRefitCounter = 0;
            static constexpr uint32_t kBlasFullRebuildInterval = 64;
            // Position-buffer byte size, captured at build, for the prevVertex
            // re-sync copy below.
            VkDeviceSize vbBytes = 0;
            // Set when refreshGeomBlasBatch snapshots OLD positions into
            // prevVertex on an in-place vertex update. The change frame
            // correctly reports old→new motion; the NEXT clean frame we re-sync
            // prevVertex := vertex so a now-static re-rolled mesh returns to
            // prevVertex == vertex (zero motion), matching initial-build
            // geometry. Without this, prevVertex stays frozen at the pre-update
            // positions and the mesh emits a constant bogus per-vertex motion
            // vector every frame → persistent denoiser/TAA history rejection
            // (runtime-updated geometry stays noisy / visibly shakes).
            bool prevVertexResyncPending = false;

            // ── Automatic mesh LOD (setAutoLod) ─────────────────────────
            // One simplified INDEX buffer + its own static BLAS per chain
            // level, built beyond this record's own (LOD0) vertex/normal/uv/
            // color buffers — those stay shared across every level (meshopt
            // never moves or reorders vertices, only drops/reweaves
            // triangles), so a level swap is purely an index-buffer +
            // BLAS-reference change, consumed identically by the raster
            // vertex-pulling path (buildIndirectDrawData) and the TLAS
            // instance fill (ensureSceneBuilt).
            struct LodLevel {
                VkAccelerationStructureKHR as = VK_NULL_HANDLE;
                Buffer storage;
                Buffer index;// uint32, indexes into THIS record's own `vertex`
                VkDeviceAddress address = 0;// BLAS device address
                uint32_t indexCount = 0;
                // Absolute object-space screen-space-error bound for this
                // level (see threepp::geometrylod::Level::error). Multiplied
                // by world scale + px/unit at selection time.
                float errorBound = 0.f;
            };
            // Chain-generation state. None → not yet attempted (or the prior
            // attempt's geometry version is stale); Queued → a background
            // job is in flight; Ready → lodLevels is usable; Failed →
            // generateChain produced nothing (too small/degenerate) — never
            // retried for this geomVersion. Touched ONLY on the main thread
            // (the worker only ever sees copied job data, never the record).
            enum class LodState : uint8_t { None, Queued, Ready, Failed };
            LodState lodState = LodState::None;
            // Index i backs MeshEntry::lodLevel == i+1 (level 0 is this
            // record's own vertex/index/etc — never stored here).
            std::vector<LodLevel> lodLevels;
        };
        std::unordered_map<const BufferGeometry*, std::unique_ptr<BlasRecord>> blasCache;

        // Resolved index/AS data for whichever LOD level an entry currently
        // selects. Falls back to the record's own LOD0 buffers when
        // lodLevel==0 or that level isn't built yet — the ONLY two call
        // sites that pick per-entry geometry for rendering (buildIndirectDrawData's
        // raster path, and the TLAS instance / GeometryDesc fill below) go
        // through this so the rasterized and ray-traced views of an entry
        // can never disagree about which triangles exist.
        struct LodGeomSel {
            VkDeviceAddress asAddress;
            VkDeviceAddress indexAddress;
            uint32_t indexCount;
            // A selected LEVEL is always an indexed draw — even when the
            // record itself is non-indexed soup (the chain generator welds
            // soup into canonical indices whose VALUES are original vertex
            // ids, so levels index the record's unchanged soup vertex
            // buffer). Fallback = the record's own indexed-ness. Consumers
            // (DrawInfo::indexed, GeometryDesc::indexed) must read THIS,
            // never rec.index.handle directly.
            bool indexed;
        };
        static LodGeomSel selectLodGeom(const BlasRecord& rec, uint8_t lodLevel) {
            if (lodLevel > 0 && static_cast<size_t>(lodLevel - 1) < rec.lodLevels.size()) {
                const auto& lvl = rec.lodLevels[lodLevel - 1];
                if (lvl.as != VK_NULL_HANDLE) {
                    return {lvl.address, lvl.index.address, lvl.indexCount, true};
                }
            }
            return {rec.address, rec.index.address, rec.indexCount,
                    rec.index.handle != VK_NULL_HANDLE};
        }

        // Per-SkinnedMesh deformed-geometry BLAS. Unlike static meshes, skinned
        // meshes can't share BLAS even when they share BufferGeometry — each
        // instance has its own pose. Vertex/normal buffers are host-mapped and
        // overwritten with CPU-skinned positions/normals each frame the bones
        // change; the BLAS is then rebuilt in-place against the same AS handle
        // (and storage) so its address — and the TLAS reference to it — remain
        // valid. prevBoneMats is the dirty-detection key (memcmp against the
        // current Skeleton::boneMatrices).
        struct SkinnedMeshState {
            std::unique_ptr<BlasRecord> blas;
            std::vector<float> prevBoneMats;
            std::weak_ptr<BufferGeometry> liveCheck;

            // GPU skinning input buffers — populated once at ensureSkinnedBlas
            // time and reused every frame. Output is the BLAS's own vertex /
            // normal buffer (overwritten by the compute dispatch).
            Buffer baseVertex   {};// vec3<float>, count = vertexCount
            Buffer baseNormal   {};// vec3<float>, count = vertexCount
            Buffer skinIndex    {};// vec4<float>, count = vertexCount
            Buffer skinWeight   {};// vec4<float>, count = vertexCount
            // Bone matrices buffer layout: [bindMatrix, bindMatrixInverse,
            // bones[0]...bones[N-1]] as mat4s. Host-visible so the per-frame
            // upload is a small memcpy. bindMatrix/Inverse are written once
            // at allocation; only the bones[..] portion changes each frame.
            Buffer boneMatrices {};
            uint32_t vertexCount    = 0;
            uint32_t boneCount      = 0;
            uint32_t primitiveCount = 0;// for per-frame BLAS rebuild
            bool     indexed        = false;
            // Per-mesh descriptor set wiring all of the above + the BLAS
            // output buffers into the skinning pipeline's set 0.
            VkDescriptorSet skinDescSet = VK_NULL_HANDLE;
            // Persistent scratch buffer for BLAS rebuild. Sized at the
            // first ensureSkinnedBlas, reused every frame. Avoids per-frame
            // alloc/free that was the original oneshot cost.
            Buffer blasScratch {};
            VkDeviceSize blasScratchSize = 0;
            // Per-frame BLAS-refit state. Same pattern as DisplacedMeshState:
            // initial AS is BUILD (buildBlasFor) with ALLOW_UPDATE, then per-
            // frame refits via MODE_UPDATE for 63 of every 64 frames. Periodic
            // full rebuild keeps the BVH quality from drifting under large
            // articulated pose changes. Skinned-mesh vertex movement is
            // typically larger frame-to-frame than ocean displacement, so the
            // periodic rebuild matters more here.
            uint32_t blasRefitCounter = 0;
            static constexpr uint32_t kBlasFullRebuildInterval = 64;
        };
        std::unordered_map<const SkinnedMesh*, std::unique_ptr<SkinnedMeshState>> skinnedMeshStates;

        // List of SkinnedMeshState pointers whose bones changed this frame.
        // ensureSceneBuilt populates this (uploads bone matrices to the GPU
        // buffer); recordCommandBuffer consumes it by recording skinning
        // dispatch + BLAS rebuild into the main per-frame cmd buffer with
        // barriers. Cleared at the end of recordCommandBuffer.
        std::vector<SkinnedMeshState*> pendingSkinnedRebuilds_;

        // Per-Mesh tet-skinned (PhysX soft body) deformed BLAS. Like SkinnedMesh,
        // each body has its own pose so it can't share a BLAS. The static tet
        // bindings (tetIndex/tetWeight/restInv*) + rest normals are uploaded once;
        // the per-frame collision-tet positions are re-uploaded from the soft body's
        // tet texture each frame, then tet_skinning.comp blends the full-res visual
        // into the BLAS vertex/normal buffers and the BLAS is refit in place.
        struct TetMeshState {
            std::unique_ptr<BlasRecord> blas;
            std::weak_ptr<BufferGeometry> liveCheck;
            // Static per-vertex inputs (uploaded once at ensureTetBlas).
            Buffer tetIndex   {};// vec4<float> — 4 tet-vertex indices
            Buffer tetWeight  {};// vec4<float> — barycentric weights
            Buffer baseNormal {};// vec3<float> — rest normals
            Buffer restInv0   {};// vec3<float> — baked Dr^-1 column 0
            Buffer restInv1   {};// vec3<float> — baked Dr^-1 column 1
            Buffer restInv2   {};// vec3<float> — baked Dr^-1 column 2
            // Per-frame collision-tet world positions (vec4/texel), re-uploaded each
            // frame from the soft body's tet texture image.
            Buffer tetPos     {};
            VkDeviceSize tetPosBytes = 0;
            // Zero-copy interop (enableSoftBodyInterop): when tetPosExt holds a
            // buffer, it replaces tetPos as the shader's binding-6 source and
            // tetPosExternalCopy (a CUDA device→device copy registered by the
            // PhysX glue) replaces the CPU upload in refreshTetBlas.
            vulkan::ExternalBuffer tetPosExt {};
            std::function<void()>  tetPosExternalCopy;
            uint32_t vertexCount    = 0;
            uint32_t primitiveCount = 0;
            bool     indexed        = false;
            VkDescriptorSet tetDescSet = VK_NULL_HANDLE;
            Buffer blasScratch {};
            VkDeviceSize blasScratchSize = 0;
            uint32_t blasRefitCounter = 0;
            static constexpr uint32_t kBlasFullRebuildInterval = 64;
        };
        std::unordered_map<const Mesh*, std::unique_ptr<TetMeshState>> tetMeshStates;
        std::vector<TetMeshState*> pendingTetRebuilds_;

        // Per-Mesh morphed-geometry BLAS. Two meshes sharing the same
        // BufferGeometry can have different morphTargetInfluences, so each
        // morphed mesh gets its own BLAS (same principle as SkinnedMesh).
        // prevInfluences is the dirty-detection key (memcmp).
        struct MorphedMeshState {
            std::unique_ptr<BlasRecord> blas;
            std::vector<float> prevInfluences;
            std::vector<float> blendedPositions;
            std::vector<float> blendedNormals;
            std::weak_ptr<BufferGeometry> liveCheck;
        };
        std::unordered_map<const Mesh*, std::unique_ptr<MorphedMeshState>> morphedMeshStates;

        // Per-DisplacedMesh: BLAS (rebuilt-in-place each frame, same scheme as
        // SkinnedMesh) plus up to three FFT cascades (Phillips/Dynamic/IFFT
        // per cascade). The water_displace.comp pass reads the spatial-domain
        // output images of all enabled cascades and writes positions+normals
        // into the BLAS vertex/normal buffers. Cascades cover disjoint k-bands
        // (band-passed via Phillips kMin/kMax) so the surface gains real
        // multi-scale wave detail without double-counting energy.
        struct DisplacedMeshState {
            std::unique_ptr<BlasRecord> blas;
            struct Cascade {
                std::unique_ptr<water::PhillipsSpectrum> phillips;
                std::unique_ptr<water::DynamicSpectrum>  dyn;
                std::unique_ptr<water::IFFT>             ifft;
                bool  phillipsRecorded = false;
                float tileSize = 0.f;            // 0 = cascade not in use
            };
            std::array<Cascade, 3> cascades;
            uint32_t cascadeMask = 0;            // bit i set = cascade i enabled
            water::OceanImage scratchA;          // RG32F IFFT scratch — shared across cascades (sequential dispatch)
            VkDescriptorSet displaceDS = VK_NULL_HANDLE;
            uint32_t vertexCount = 0;
            uint32_t gridDim     = 0;            // sqrt(vertexCount); validated at init
            float    planeSize   = 0.f;
            // Per-cascade height readback for CPU-side wave sampling (boat
            // hydrodynamics, pitch/roll from multi-scale wave slope, etc.).
            // Host-mapped RG32F buffers of size textureSize²·8 bytes each;
            // populated after every IFFT pass via vkCmdCopyImageToBuffer.
            Buffer    heightReadback;
            Buffer    heightReadback1;
            Buffer    heightReadback2;
            // Per-cascade readback texture dimension. Cascades can run at
            // different FFT resolutions — each cascade's readback buffer is
            // sized to its own dim²·8 bytes (RG32F).
            uint32_t  heightReadbackDim[3] = {0, 0, 0};
            std::weak_ptr<BufferGeometry> liveCheck;
            // Per-frame BLAS-refit state. The initial AS is built (buildBlasFor)
            // with ALLOW_UPDATE, so subsequent per-frame refreshes can use
            // MODE_UPDATE (refit) instead of full BUILD — typically 3-10× faster
            // for the same triangle count. AS quality degrades over many refits;
            // force a full rebuild every kBlasFullRebuildInterval frames.
            uint32_t  blasRefitCounter = 0;          // frames since last full BUILD
            static constexpr uint32_t kBlasFullRebuildInterval = 64;

            // Per-mesh foam-disturbance SSBO. Host-mapped, written from
            // dm.foamDisturbances each frame before the water_displace
            // dispatch. Sized to kMaxFoamDisturbances × 16B; extra
            // entries are dropped. .address == 0 until first allocation.
            Buffer    foamDisturbBuffer{};
            static constexpr uint32_t kMaxFoamDisturbances = 64;

            // Per-mesh wake-trail SSBO. Host-mapped, written from
            // dm.wake.trail each frame. Each sample is 32 bytes (matches
            // DisplacedMesh::WakeSample). The water_displace shader
            // iterates all valid samples and sums age-decayed Kelvin
            // V-wake contributions so the wake traces the boat's past
            // path rather than snapping to the current pose.
            Buffer    wakeTrailBuffer{};
            static constexpr uint32_t kMaxWakeSamples = 64;

            // World-space foam texture — 2D R32F covering the cascade-0
            // tile (foamTileSize × foamTileSize world m), REPEAT-sampled
            // in both compute and chit so wrapping at the tile boundary
            // aligns with the FFT pattern's periodicity. Replaces the old
            // per-vertex `foam` buffer on the BLAS: with foam pinned to
            // world coords, the ocean mesh can be re-tessellated freely
            // without dragging foam along with the vertex indices.
            //
            // No ping-pong needed because foam_world.comp only reads and
            // writes its own texel each invocation — no cross-texel
            // dependency, so `imageLoad` + `imageStore` on the same image
            // is race-free.
            water::OceanImage foamImage;
            uint32_t foamRes      = 0;          // texels per side
            float    foamTileSize = 0.f;        // world extent (m) covered
            VkDescriptorSet foamWorldDS = VK_NULL_HANDLE;
            // Wall-clock timestamp of the previous foam dispatch — drives the
            // frame-rate-independent decay push constant (−1 = first frame).
            double   foamPrevTimeSec = -1.0;
        };
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

        // ── GrassMesh (GPU wind-deformed foliage) ────────────────────────
        // Same deform-on-GPU + BLAS-refit-in-place pattern as DisplacedMesh,
        // but FFT-free: a wind compute shader bends each blade vertex from an
        // immutable rest pose. The whole grass field is ONE mesh → one BLAS →
        // one TLAS instance, so animating it is a cheap per-frame BLAS/TLAS
        // refit instead of the O(all-instances) TLAS rebuild an animated
        // InstancedMesh would force.
        struct GrassMeshState {
            std::unique_ptr<BlasRecord> blas;
            Buffer   restPos{};   // immutable rest xyz (device addr, host-written once)
            Buffer   heightFrac{};// per-vertex height fraction (device addr)
            uint32_t vertexCount = 0;
            std::weak_ptr<BufferGeometry> liveCheck;
            uint32_t blasRefitCounter = 0;
            static constexpr uint32_t kBlasFullRebuildInterval = 64;
        };
        std::unordered_map<const GrassMesh*, std::unique_ptr<GrassMeshState>> grassStates;
        // Shared grass-wind compute pipeline (no descriptor sets — all I/O by
        // device address). See vulkan/GrassWindPipeline.{hpp,cpp}.
        std::unique_ptr<vulkan::GrassWindPipeline> grassWind_;
        // GrassMesh deforms queued in ensureSceneBuilt, recorded into the frame
        // command buffer in recordCommandBuffer (no blocking submit) — same
        // pattern as pendingSkinnedRebuilds_. Cleared at end of recordCommandBuffer.
        std::vector<std::pair<GrassMesh*, GrassMeshState*>> pendingGrassDeforms_;

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
        // the frame command buffers execute in submit order so it never races).
        Buffer tlasRefitScratch_{};
        VkDeviceSize tlasRefitScratchSize_ = 0;
        // Per-frame TLAS refit, staged by ensureSceneBuilt and recorded into the
        // frame command buffer by recordCommandBuffer (after the deformable BLAS
        // rebuilds). Replaces the old mid-frame refitTlas one-shot drain.
        std::vector<VkAccelerationStructureInstanceKHR> pendingTlasInstances_;
        bool pendingTlasRefit_ = false;
        bool pendingTlasFullBuild_ = false;

        // Per-instance descriptor tables the closest-hit shader indexes by
        // gl_InstanceCustomIndexEXT. Layout matches the matching shader
        // structs in closest_hit.rchit.
        struct GeometryDesc {
            VkDeviceAddress vertexAddress;// positions, used for per-pixel tangent derivation
            VkDeviceAddress normalAddress;
            VkDeviceAddress indexAddress;
            VkDeviceAddress uvAddress;// 0 == no UV attribute
            VkDeviceAddress foamAddress;// 0 == no foam attribute (per-vertex float, written by water_displace.comp)
            // Previous-frame deformed vertex positions. For SkinnedMesh (or any
            // mesh that re-deforms per frame): a separate buffer holding the
            // previous frame's vertex data. The deferred shade's ray-query hit
            // handling interpolates these to give a per-vertex "previous world
            // position" so reprojection tracks the same surface point across
            // deformation. For static / rigid-only meshes: set equal to
            // vertexAddress, in which case the hit handling reads the same
            // data twice (no harm; equals current pos).
            VkDeviceAddress prevVertexAddress;
            // Per-vertex RGB color buffer (BlasRecord::color). 0 when the mesh's
            // material has vertexColors off or the geometry has no "color" — the
            // chit then skips the vertex-color multiply. Set per-instance below.
            VkDeviceAddress colorAddress;
            uint32_t indexed;
            // Bit 0: sticky "recently moved" flag, stamped per frame by the
            // loop in VulkanCoreFrame.cpp (reflection/GI history gates read it).
            // Bits 1..3: BlasRecord::packedMask << 1 — attribute buffers hold
            // packed data (bit1 nrm oct-snorm16x2 / bit2 uv unorm16x2 / bit3
            // col unorm8x4). Shaders mask accordingly; a bare `!= 0` test on
            // this word is WRONG for the moved gate.
            uint32_t flags;
        };
        // MaterialDesc layout lives in vulkan_shared.h (the same file the GLSL
        // deferred-shade / gbuffer / LIDAR shaders pull in via #include).
        // Bringing it into Impl scope here keeps the existing
        // `MaterialDesc md{};` call sites unchanged.
        using MaterialDesc = threepp::vulkan_pt::MaterialDesc;

        // Per-frame-in-flight GeometryDesc storage — same fence-gated ring as
        // materialDescsBuffers below. Ringed for auto-LOD: a level switch
        // repoints GeometryDesc::indexAddress/indexed, and the single-buffer
        // version needed a vkDeviceWaitIdle on every switch frame to patch it
        // in place — a stall exactly when the camera moves (the frames where
        // responsiveness matters most). Hot path stages in geomDescsCached_ +
        // flips geomDescsDirty_[*]; renderFrame's flushGeometryDescsIfDirty
        // memcpys into this frame's slot once its fence has signaled.
        std::array<Buffer, kFramesInFlight> geometryDescsBuffers{};
        std::array<bool, kFramesInFlight> geomDescsDirty_{};
        // Per-frame-in-flight MaterialDesc storage. Was a single shared buffer
        // gated by vkDeviceWaitIdle on every animated-pbr update; now one buffer
        // per kFramesInFlight slot so the upload after a fence wait races
        // nothing. The hot path stages new descs in `matDescsCached_` + flips
        // `matDescsDirty_[*]=true`; renderFrame's flushMaterialDescsIfDirty
        // memcpys into `materialDescsBuffers[currentFrame]` once the fence has
        // signaled (= GPU done with this slot). Descriptor sets are bound
        // per-frame (set idx = f*imageCount_+k → buffer[f]) so the binding
        // stays valid across the swap.
        std::array<Buffer, kFramesInFlight> materialDescsBuffers{};
        std::vector<MaterialDesc> matDescsCached_;
        std::array<bool, kFramesInFlight> matDescsDirty_{};

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
        static constexpr uint32_t kMaxDirLights   = 8;
        static constexpr uint32_t kMaxPointLights = 8;
        static constexpr uint32_t kMaxSpotLights  = 8;
        static constexpr uint32_t kMaxRectLights  = 4;

        struct GpuDirLight {
            float direction[3];
            float color[3];
        };
        struct GpuPointLight {
            float position[3]; float range;
            float color[3];    float decay;
            float radius;      // physical source radius (world units) → RT soft shadows; 0 = hard
        };
        struct GpuSpotLight {
            float position[3];   float range;
            float color[3];      float decay;
            float direction[3];  // toward target (emission direction)
            float cosAngleOuter; // cos(angle)
            float cosAngleInner; // cos(angle * (1-penumbra))
            float radius;        // physical source radius (world units) → RT soft shadows; 0 = hard
        };
        struct GpuRectLight {
            float position[3];
            float halfU[3];  // world right  * width/2
            float halfV[3];  // world up     * height/2
            float normal[3]; // emission direction into scene
            float color[3];
        };
        struct GpuLightsUbo {
            float       ambient[3];
            uint32_t    dirCount;
            GpuDirLight dirLights[kMaxDirLights];
            uint32_t    pointCount;
            uint32_t    spotCount;
            uint32_t    rectCount;
            GpuPointLight pointLights[kMaxPointLights];
            GpuSpotLight  spotLights[kMaxSpotLights];
            GpuRectLight  rectLights[kMaxRectLights];
        };
        static_assert(sizeof(GpuDirLight)   == 24);
        static_assert(sizeof(GpuPointLight) == 36);
        static_assert(sizeof(GpuSpotLight)  == 56);
        static_assert(sizeof(GpuRectLight)  == 60);
        std::array<Buffer, kFramesInFlight> lightsUbos{};

        // ── Clustered lights (deferred) ─────────────────────────────────────
        // ALL scene point/spot lights, power-sorted, in one unified record —
        // the UBO's 8-per-type arrays above keep the STRONGEST 8 (same sort)
        // for the paths screen-space clusters can't serve (secondary ray
        // hits — reflection/GI, volumetric beams, probes). cluster_build.comp
        // culls the list
        // into per-cell index rows of a 16×8×24 screen-tile × exponential-Z
        // grid; deferred_shade's analytic split loops only its own cell.
        // KEEP IN SYNC with the ClusterLight structs in cluster_build.comp +
        // deferred_shade.comp (scalar layout, 64 bytes).
        static constexpr uint32_t kMaxClusterLights   = 256;
        static constexpr uint32_t kClusterCells       = 16 * 8 * 24;
        static constexpr uint32_t kClusterMaxPerCell  = 24;
        struct GpuClusterLight {
            float position[3];   float range;         // range 0 = infinite (three.js)
            float color[3];      float decay;         // color premultiplied by intensity
            float direction[3];  float cosAngleOuter; // spot cone; points carry -1.1/-1.05 (cone test → 1)
            float cosAngleInner;
            float radius;        // physical source radius (soft shadows)
            float cullRadius;    // conservative influence radius (range, or the atten<eps solve)
            float type;          // 0 = point, 1 = spot
        };
        static_assert(sizeof(GpuClusterLight) == 64);
        std::array<Buffer, kFramesInFlight> clusterLightsBuffers{};// host-visible mapped (CPU fills per frame)
        std::array<Buffer, kFramesInFlight> clusterGridBuffers{};  // device-local (cluster_build writes)
        uint32_t clusterLightCountThisFrame_ = 0;
        bool     fogEnabledThisFrame_ = false;// scene.fog present (froxel-volumetrics gate)

        // Homogeneous fog (participating media). FogExp2.density maps directly
        // to sigma_t; linear Fog (near/far) is converted to an equivalent
        // density. Enabled flag = 0 short-circuits all fog work in the shaders.
        // anisotropy is the Henyey-Greenstein g for single-scattering.
        struct GpuFogUbo {
            float sigmaT[3];     // per-channel extinction (1/world unit)
            float enabled;       // 1.0 = fog active, 0.0 = disabled
            float color[3];      // inscatter tint (sRGB-linear)
            float anisotropy;    // HG g, clamped [-0.95, 0.95] by setFogAnisotropy
            float waterSurfaceY; // world-Y of the water surface; 1e30 = no limit
            float worldUp[3];    // world up axis (= camera.up) for sky aerial perspective
            // Unified fog medium (setHeightFog / resolved scene.fog) params,
            // MIRRORED from GpuCloudUbo so the deferred FILTER recombines
            // (deferred_filter_common.glsl, which binds only this fog UBO — not
            // the CloudUbo) can carry the same hetero extinction the shade pass
            // applies. 0 density = no air medium. Phase 2: scene.fog now feeds
            // these (its density/profile), so the froxel hetero path is the ONE
            // air medium; the sigmaT/color/enabled fields above are the medium's
            // beam-σ / albedo / present-flag for the volumetric consumers.
            float hfDensity;     // air-medium σ_t at baseY
            float hfBaseY;       // air-medium base world Y
            float hfFalloff;     // air-medium exponential height scale (m); huge ≈ uniform
            // Underwater murk (setUnderwaterMurk) — a SEPARATE homogeneous medium
            // clipped to BELOW waterSurfaceY (the water body's own absorption),
            // decoupled from the air fog in Phase 2 so a scene can hold clear air
            // above the waterline and murk below (the fjord). 0 density = off.
            float murkDensity;   // murk σ_t (1/m); 0 = off
            float murkColor[3];  // murk inscatter tint (sRGB-linear)
        };
        static_assert(sizeof(GpuFogUbo) == 76);
        std::array<Buffer, kFramesInFlight> fogUbos{};
        float    fogAnisotropy_ = 0.0f;
        float    fogWaterSurfaceY_ = 1e30f;
        float    murkDensity_ = 0.0f;               // setUnderwaterMurk (0 = off)
        float    murkColor_[3] = {1.0f, 1.0f, 1.0f};// setUnderwaterMurk tint
        uint64_t prevFogHash_ = 0u;
        // ── Resolved unified fog medium for THIS frame (computed by updateFogUbo,
        // consumed by updateCloudUbo + the froxel gate) ──────────────────────────
        // Phase 2 "one knob": scene.fog is the primary control — when present it
        // supplies the medium DENSITY (FogExp2.density / linear-Fog span) + COLOUR
        // and the froxel volumetrics run in HETEROGENEOUS mode with a near-uniform
        // default profile (baseY 0, huge falloff). setHeightFog is the ADVANCED
        // profile control: it sets baseY/falloff/noise; its density is used ONLY
        // when scene.fog is absent (back-compat for existing mist users). When
        // BOTH are set scene.fog's density WINS (heightFog.density ignored).
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

        // Volumetric cloud layer (VulkanRenderer::setClouds) + near-field
        // heterogeneous height fog (VulkanRenderer::setHeightFog). Both ride the
        // one binding-58 scalar UBO (they share wind + timeSec). clouds.enabled
        // == 0 short-circuits the far cloud march; heteroActive == 0 keeps the
        // froxel volumetrics on today's homogeneous path (off = free /
        // image-identical). Layout matches deferred_shade.comp / cloud_march /
        // froxel_inject / froxel_integrate's scalar CloudUbo block exactly.
        struct GpuCloudUbo {
            float enabled;      // 1.0 = far cloud march active
            float coverage;     // 0 = clear .. 1 = overcast
            float density;      // density multiplier
            float bottomY;      // shell base (world Y)
            float topY;         // shell top (world Y)
            float evolveSpeed;  // shape churn rate
            float timeSec;      // wall-clock seconds (wind scroll + evolution)
            float heteroActive; // 1.0 = heterogeneous near-field froxels (height fog on)
            float wind[3];      // m/s xz drift (y ignored)
            float hfDensity;    // height-fog σ_t at baseY (0 = height fog off)
            float hfBaseY;      // height-fog base world Y
            float hfFalloff;    // height-fog exponential height scale (m)
            float hfNoiseAmount;// 0 = smooth analytic .. 1 = fully noise-modulated
            float shadowActive; // 1.0 = cloud shadow map valid this frame (clouds on)
            float epoch;        // history generation — cloud_march rejects prev-epoch
                                // history (first-enable garbage, reconfigured decks)
        };
        static_assert(sizeof(GpuCloudUbo) == 68);
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
        // HDRI sun extracted at PMREM build (deferred only — see
        // envSunExtractionWanted). found=false when no sun / extraction off.
        // updateLightsUbo re-injects it as an analytic directional light.
        vulkan::EnvPrefilter::SunExtract envSun_{};

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
        // (fillMaterialTextureInfos → materialSampler()): 16× aniso when the
        // raster is UNJITTERED (sharpness has no temporal cost), and
        // isotropic-trilinear when the raster is JITTERED (TAA/DLSS/FSR) —
        // anisotropic filtering re-sharpens grazing-angle textures back to
        // pixel frequency, which no temporal resolve can hold still; measured
        // as the dominant carrier of the "whole scene shimmers at a distance"
        // residual on terrain/Bistro-class content. Each policy exists in a
        // REPEAT and a CLAMP_TO_EDGE flavour — clamp-tagged textures
        // (materialTexClampUV_) get the clamp twin, same filter policy.
        // NOTE: the per-image samplers buildSampledImage2D creates are NOT
        // bound for material textures — these are.
        VkSampler textureSampler_ = VK_NULL_HANDLE;        // 16× aniso (unjittered raster)
        VkSampler textureSamplerIso_ = VK_NULL_HANDLE;     // isotropic (jittered raster)
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
        std::array<Image2D, 2> reservoirPosImagesPP{};
        std::array<Image2D, 2> reservoirWImagesPP{};
        // Frame counter driving Halton jitter + blue-noise offset for the
        // deferred shade's stochastic GI / soft-shadow sampling.
        uint32_t sampleIndex = 0;
        // Prev-frame camera packed as four vec4s (matches PrevCameraUbo):
        //   [0..3]  = vec4(pos.xyz,  projScaleX)   → prevCamPosX
        //   [4..7]  = vec4(fwd.xyz,  projScaleY)   → prevCamFwdY
        //   [8..11] = vec4(rgt.xyz,  0)             → prevCamRgt
        //   [12..15]= vec4(up.xyz,   0)             → prevCamUp
        std::array<float, 16> prevCamBufData_{};
        bool prevCameraValid = false;

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

        // FNV-1a 64-bit hash of the previous frame's GpuLightsUbo bytes. Used in
        // updateLightsUbo to detect changes in analytic-light state (visibility,
        // intensity, color, position, direction, range, decay, cone angles) and
        // mark the frame as moved so the per-pixel reproject path halves FC and
        // the new lighting is picked up quickly. RectAreaLight + emissive meshes
        // already trigger a reset via the per-frame emissive-triangle CDF rebuild;
        // this covers DirectionalLight / PointLight / SpotLight which only flow
        // through the lights UBO and would otherwise leave stale accumulation.
        uint64_t prevLightsHash_ = 0u;

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
            // Frustum-cull bit, populated once per frame by
            // cullEntriesAgainstFrustum() right before record. Raster passes
            // skip entries with inFrustum == false to dodge the GPU's per-
            // draw command-processor overhead on off-screen geometry. The
            // ray-query path is unaffected — TLAS culling handles it implicitly.
            // Defaults to true so passes work before the first cull pass.
            // Skinned / displaced / morphed entries always stay true; their
            // local AABB doesn't reflect deformed extents.
            bool     inFrustum   = true;
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
        std::vector<SnapNode> sceneSnapshot_;

        // Classification-routing flags for a mesh — shared by the snapshot
        // build and the replay walk so the two can't drift.
        uint32_t snapMeshFlags(Mesh& m, const MaterialWithWireframe* wf) const;

        // Replay the last expansion's traversal against the snapshot. true ⇒
        // tree shape, visibility, classification routing and all mesh/geom/mat
        // pointers are unchanged since the last full expansion. Also re-runs
        // LOD level selection (camera-driven visibility mutation) on kSnapLod
        // nodes — a level switch makes the walk diverge from the snapshot ⇒
        // false ⇒ the full pass re-expands with the new level.
        bool sceneSnapshotMatches(Object3D& scene, Camera& camera);

        // Previous-frame camera (proj_prev * view_prev) for primary-hit
        // reprojection. One UBO per frame-in-flight so updates don't race the
        // GPU. Per-instance motion matrices (prevWorld * inverse(curWorld)) live
        // in a single SSBO indexed by gl_InstanceCustomIndexEXT; the host repacks
        // the array each frame from prevWorldMats keyed by (Mesh*, instanceIndex)
        // so each InstancedMesh sub-instance has its own motion delta. First-frame
        // / first-seen entries are identity so reproject is a no-op.
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
        std::array<Buffer, kFramesInFlight> motionMatBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> motionMatBufferCapacity{};
        // Per-buffer-slot "last upload was all-identity" flag. When true and
        // the current frame also has no per-instance motion, skip the upload
        // entirely — the buffer slot is still valid identity from before.
        // Cleared on capacity-grow (new buffer is undefined).
        std::array<bool, kFramesInFlight> motionMatBufferAllIdentity_{};
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
        // shimmer otherwise — a "just right" default, not an opt-in.
        bool     normalMapToksvig_ = true;

        // ── Automatic mesh LOD (setAutoLod; ON by default) ──────────────────
        // Default-on since the full measurement pass: Bistro (per-pixel-bound
        // worst case) neutral, fjord flight +32% FPS, quality below animation
        // noise, switch frames stall-free (geomDescs ring). setAutoLod(false)
        // remains as the manual override / debug escape (Toksvig pattern).
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
        // under MODE_BUILD, not the incremental MODE_UPDATE refit. Also
        // flips geomDescsDirty_[*] so RT secondary rays (reflections/GI/
        // lidar/probe update) read the SAME index buffer the swapped BLAS
        // was built from — without it, gl_PrimitiveID from a hit against a
        // coarser BLAS would misindex the still-LOD0
        // GeometryDesc::indexAddress.
        bool lodChangedThisFrame_ = false;
        // Host mirror of the last-uploaded GeometryDesc array (entries-
        // indexed, same layout as the `geomDescs` local built in the full
        // rebuild). The lean auto-LOD path patches indexAddress/indexed in
        // place here; the per-frame ring flush (flushGeometryDescsIfDirty)
        // carries it to the GPU stall-free.
        std::vector<GeometryDesc> geomDescsCached_;
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
        // mutates or tears down while a job runs. Joined in ~CoreImpl before
        // anything Vulkan-related is torn down (the worker is pure CPU).
        std::thread lodWorker_;
        std::mutex lodJobMutex_;
        std::condition_variable lodJobCv_;
        bool lodWorkerStop_ = false;
        struct LodJob {
            const BufferGeometry* geom = nullptr;
            unsigned int geomVersion = 0;
            std::vector<float> positions;// tightly packed xyz
            // Indexed geometry: the source index array; normals/uvs stay
            // empty (the weld step is skipped, topology already exists).
            // NON-indexed soup (FBX-style loaders never call setIndex):
            // empty — the worker first welds identical-attribute duplicates
            // into canonical indices (buildCanonicalIndices) so the
            // simplifier has shared edges to collapse. normals/uvs feed that
            // weld's equality test so genuine seams/hard edges stay split;
            // they carry copies only when the stream exists with matching
            // counts.
            std::vector<uint32_t> indices;
            std::vector<float> normals;// tightly packed xyz — simplifier attribute
                                       // metric (all paths) + soup weld equality
            std::vector<float> uvs;    // soup only; tightly packed xy
            // Normal-attribute weight for the simplifier, scaled by the
            // enqueueing entry's material GLOSSINESS: matte foliage barely
            // charges normal deviation (retiring far detail is the point),
            // glossy paint charges it fully (shading IS the mm-scale normal
            // field). See lodNormalWeightFor.
            float normalWeight = 0.5f;
        };
        std::deque<LodJob> lodJobQueue_;
        struct LodResult {
            const BufferGeometry* geom = nullptr;
            unsigned int geomVersion = 0;
            std::vector<geometrylod::Level> levels;// empty ⇒ degenerate/failed
        };
        std::mutex lodResultMutex_;
        std::deque<LodResult> lodResultQueue_;
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
        VulkanRendererCore::AutoLodStats autoLodStats_{};

        void ensureLodWorkerStarted() {
            if (lodWorker_.joinable()) return;
            lodWorker_ = std::thread([this] { lodWorkerMain(); });
        }
        void lodWorkerMain() {
            for (;;) {
                LodJob job;
                {
                    std::unique_lock<std::mutex> lk(lodJobMutex_);
                    lodJobCv_.wait(lk, [this] { return lodWorkerStop_ || !lodJobQueue_.empty(); });
                    // Exit immediately on stop, even with jobs still queued —
                    // shutdown must be bounded; nothing will ever drain their
                    // results after this. A job already popped and mid-
                    // simplify (the loop body below, no lock held) still
                    // runs to completion — can't interrupt meshopt mid-call,
                    // and its result is simply never drained.
                    if (lodWorkerStop_) return;
                    job = std::move(lodJobQueue_.front());
                    lodJobQueue_.pop_front();
                }
                const size_t vertexCount = job.positions.size() / 3;
                // Soup input: weld first (identical-attribute duplicates →
                // canonical indices over the ORIGINAL vertex ids), then
                // simplify those. An empty weld result flows through as an
                // empty chain ⇒ LodState::Failed at drain, same as any other
                // degenerate geometry.
                std::vector<uint32_t> canonical;
                const uint32_t* idxData = job.indices.data();
                size_t idxCount = job.indices.size();
                if (job.indices.empty()) {
                    canonical = geometrylod::buildCanonicalIndices(
                            job.positions.data(),
                            job.normals.empty() ? nullptr : job.normals.data(),
                            job.uvs.empty() ? nullptr : job.uvs.data(),
                            vertexCount);
                    idxData = canonical.data();
                    idxCount = canonical.size();
                }
                // sparse=true for welded soup: the canonical indices reference
                // only ~1/6 of the soup vertex buffer (the representatives),
                // and meshopt needs SimplifySparse to keep the unreferenced
                // duplicates out of its wedge/seam analysis — see the
                // parameter doc in GeometryLod.hpp.
                auto levels = geometrylod::generateChain(
                        job.positions.data(), vertexCount, idxData, idxCount,
                        /*sparse=*/job.indices.empty(),
                        job.normals.empty() ? nullptr : job.normals.data(),
                        job.normalWeight);
                std::lock_guard<std::mutex> lk(lodResultMutex_);
                lodResultQueue_.push_back({job.geom, job.geomVersion, std::move(levels)});
            }
        }
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
        // keeps the normal field charged so panels never flatten visibly
        // (the CarConcept lesson). Unlit renders no shading at all → 0.
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
                           float normalWeight) {
            auto* posAttr = geom.getAttribute<float>("position");
            if (!posAttr) return false;
            LodJob job;
            job.geom = geomPtr;
            job.geomVersion = geomVersion;
            job.normalWeight = normalWeight;
            const auto& pos = posAttr->array();
            job.positions.assign(pos.begin(), pos.end());
            const auto vtxCount = posAttr->count();
            // Normals feed the simplifier's ATTRIBUTE metric on every path
            // (indexed and soup) — without them the position-only quadric
            // flattens smooth glossy surfaces' shading for free (the
            // CarConcept regression; see generateChain's header doc). On the
            // soup path they additionally drive the weld's seam preservation.
            if (FloatAttributeView nrm{geom.getAttribute("normal")};
                nrm && nrm.count() == vtxCount &&
                nrm.size() == static_cast<size_t>(vtxCount) * 3) {
                job.normals.assign(nrm.data(), nrm.data() + nrm.size());
            }
            if (const auto* idxAttr = geom.getIndex()) {
                const auto& idx = idxAttr->array();
                job.indices.assign(idx.begin(), idx.end());
            } else {
                // Soup weld only: UVs join the binary-equality test so
                // genuine UV seams stay split.
                if (FloatAttributeView uv{geom.getAttribute("uv")};
                    uv && uv.count() == vtxCount &&
                    uv.size() == static_cast<size_t>(vtxCount) * 2) {
                    job.uvs.assign(uv.data(), uv.data() + uv.size());
                }
            }
            ensureLodWorkerStarted();
            {
                std::lock_guard<std::mutex> lk(lodJobMutex_);
                lodJobQueue_.push_back(std::move(job));
            }
            lodJobCv_.notify_one();
            ++lodChainsQueuedCount_;
            return true;
        }
        // Per-frame chain-finalization budget (drainLodResults, called once
        // per frame from ensureSceneBuilt): up to 16 geometries OR 8 MiB of
        // new level resources (index buffers + BLAS storage), whichever hits
        // first. All of a frame's level BLAS builds are recorded into ONE
        // one-shot submit+wait (flushLodLevelBuilds) — a submit per level
        // (or even per geometry) at Bistro-scale entry counts costs more in
        // queue round-trips than the batching saves. Stale results (record
        // evicted / geomVersion moved on) and failed chains are processed
        // outside the budget — they create no resources and cost only a
        // hash lookup + state write each.
        void drainLodResults();
        // One level's deferred BLAS build, produced by buildLodLevelFor and
        // consumed by flushLodLevelBuilds — resources (AS handle, storage,
        // index buffer, scratch) already exist; only the build execution is
        // deferred so a whole frame's worth records into one command buffer.
        struct LodPendingBuild {
            VkAccelerationStructureKHR as = VK_NULL_HANDLE;
            VkDeviceAddress vertexAddress = 0;
            VkDeviceAddress indexAddress = 0;
            uint32_t maxVertex = 0;
            uint32_t primitiveCount = 0;
            bool packedIdx = false;// index buffer is uint16 (base record's bit 3)
            Buffer scratch{};// per-build scratch: concurrent builds in one cmdbuf must not alias
        };
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
        // Records every pending level build into one one-shot command
        // buffer, submits, waits, and frees the per-build scratches.
        void flushLodLevelBuilds(std::vector<LodPendingBuild>& pending);
        // Destroys every level's AS/storage/index and decrements the running
        // byte totals — called from every blasCache erase site (destructor,
        // eviction prune, geomVersion-changed rebuild) so a record's LOD
        // resources never outlive the record itself.
        void destroyBlasLodLevels(BlasRecord& rec) {
            for (auto& lvl : rec.lodLevels) {
                if (lvl.as) ctx->rt().destroyAccelerationStructure(ctx->device(), lvl.as, nullptr);
                lodBlasBytes_  -= std::min<uint64_t>(lodBlasBytes_,  lvl.storage.size);
                lodIndexBytes_ -= std::min<uint64_t>(lodIndexBytes_, lvl.index.size);
                destroyBuffer(ctx->allocator(), lvl.storage);
                destroyBuffer(ctx->allocator(), lvl.index);
            }
            if (rec.lodState == BlasRecord::LodState::Ready && !rec.lodLevels.empty()) {
                lodChainsReadyCount_ = lodChainsReadyCount_ > 0 ? lodChainsReadyCount_ - 1 : 0;
            }
            rec.lodLevels.clear();
            rec.lodState = BlasRecord::LodState::None;
        }

        // Free every GPU resource a BlasRecord owns. Teardown used to inline
        // this list at each of the six places that hold a BlasRecord (blasCache,
        // skinned / tet / displaced / grass / morphed states), and five of the
        // six had drifted: they omitted `color`, so the per-vertex color buffer
        // leaked for every mesh that wasn't a plain cached BLAS. One helper, so
        // adding a Buffer to BlasRecord can only ever be missed in one place.
        // NB: LOD levels are NOT freed here — blasCache is the only owner of a
        // LOD chain and calls destroyBlasLodLevels() itself, which also keeps
        // the lodBlasBytes_ accounting straight.
        void destroyBlasRecord(BlasRecord& rec) {
            if (rec.as) ctx->rt().destroyAccelerationStructure(ctx->device(), rec.as, nullptr);
            destroyBuffer(ctx->allocator(), rec.storage);
            destroyBuffer(ctx->allocator(), rec.vertex);
            destroyBuffer(ctx->allocator(), rec.index);
            destroyBuffer(ctx->allocator(), rec.normal);
            destroyBuffer(ctx->allocator(), rec.uv);
            destroyBuffer(ctx->allocator(), rec.color);
            destroyBuffer(ctx->allocator(), rec.prevVertex);
            destroyBuffer(ctx->allocator(), rec.blasScratch);
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
        uint32_t cachedEmissiveVersion_ = 0;
        std::array<uint32_t, kFramesInFlight> emissiveBufferVersion_{};
        std::unordered_map<EntryKey, std::array<float, 16>, EntryKeyHash> prevWorldMats;

        // Per-entry fingerprint used to detect scene changes between frames.
        // We capture the surface state ray tracing actually consumes:
        //   - mesh / geometry / material identity (covers add/remove/swap)
        //   - effective worldMatrix 16 floats (covers transform animation
        //     AND per-instance setMatrixAt; the matrix already incorporates
        //     mesh->matrixWorld * instanceMatrix[i] for InstancedMesh)
        //   - PBR scalars (covers material slider tweaks)
        // Instance count change shows up as currFp.size() mismatch, forcing
        // structural rebuild — correct since each instance is its own TLAS slot.
        struct MeshFingerprint {
            const void* mesh;
            const void* geom;
            const void* mat;
            // Texture fingerprint slots hold the shared_ptr (not just the raw
            // pointer) so the comparison can't be fooled by an allocator
            // recycling a freed Texture's address for a brand-new Texture: a
            // held shared_ptr keeps the previous Texture alive long enough that
            // no overlap can happen, and operator!= still compares get().
            std::shared_ptr<Texture> albedoTex;            // covers map swap on the same material
            std::shared_ptr<Texture> roughnessTex;         // covers roughnessMap swap
            std::shared_ptr<Texture> metalnessTex;         // covers metalnessMap swap
            std::shared_ptr<Texture> normalTex;            // covers normalMap swap
            std::shared_ptr<Texture> transmissionTex;      // covers transmissionMap swap
            std::shared_ptr<Texture> clearcoatTex;         // covers clearcoatMap swap
            std::shared_ptr<Texture> clearcoatRoughnessTex;// covers clearcoatRoughnessMap swap
            std::shared_ptr<Texture> emissiveTex;          // covers emissiveMap swap
            std::shared_ptr<Texture> occlusionTex;         // covers aoMap swap — was MISSING: an
                                                           // occlusion-map swap neither triggered a
                                                           // structural rebuild nor got the held-
                                                           // shared_ptr recycle protection, so its
                                                           // dead predecessor's Texture* could be
                                                           // reused and hit a stale textureCache slot
            uint32_t instanceIndex;// 0 for non-instanced; distinguishes sub-instances
            unsigned int matVersion = 0;// Material::version() — bumped by needsUpdate(),
                                        // KHR_animation_pointer, etc. Lets us skip the
                                        // 8 texture-of dynamic_casts + materialFromMesh
                                        // (~21 dynamic_casts/mesh) when nothing on the
                                        // material has changed since last frame.
            unsigned int geomVersion = 0;// composite BufferAttribute version (pos+norm+idx+uv)
            std::array<float, 16> matrix{};
            std::array<float, 15> pbr{};// + normalScale.xy + transmission/ior + clearcoat/roughness
            // TYPED views + attribute-pointer cache for the snapshot lean diff
            // (the void* fields above are compare-only). attrVersion gates the
            // cached attribute pointers: while BufferGeometry::
            // attributesVersion() is unchanged, the attribute map still holds
            // the same objects, so the raw pointers cannot dangle and the
            // composite geomVersion is 4 direct version reads instead of 4
            // string-keyed map lookups + dynamic_casts per entry per frame.
            const Material* matTyped = nullptr;
            BufferGeometry* geomTyped = nullptr;
            unsigned int attrVersion = ~0u;
            const BufferAttribute* posAttr  = nullptr;
            const BufferAttribute* normAttr = nullptr;
            const BufferAttribute* uvAttr   = nullptr;
            const BufferAttribute* idxAttr  = nullptr;
        };
        std::vector<MeshFingerprint> prevSceneFingerprint;
        // Per-entry record in TLAS-instance order from the last ensureSceneBuilt
        // call. renderFrame consumes this to compute per-instance motion matrices
        // after the in-flight fence has been waited (safe to write the
        // motionMatBuffers[currentFrame] HOST_VISIBLE buffer).
        std::vector<MeshEntry> lastVisibleEntries_;
        // Per-entry cull mode cached from each upload of matDescs.
        //   Side::Front  → VK_CULL_MODE_BACK_BIT  (default fast path)
        //   Side::Back   → VK_CULL_MODE_FRONT_BIT
        //   Side::Double → VK_CULL_MODE_NONE
        // Stored as VkCullModeFlags directly so the gbuffer draw loop can
        // hand it straight to vkCmdSetCullMode. Indexed in lock-step with
        // lastVisibleEntries_.
        std::vector<VkCullModeFlags> lastVisibleCullMode_;
        bool sceneBuilt_ = false;

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
        struct DebugResolvePC {
            uint32_t view;      // 1 = normal, 2 = motion, 3 = ids, 4 = albedo
            uint32_t width;
            uint32_t height;
            uint32_t gbufWidth;
            uint32_t gbufHeight;
            float    motionGain;
        };
        VkDescriptorSetLayout debugResolveDsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      debugResolvePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            debugResolvePipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      debugResolveDescPool_       = VK_NULL_HANDLE;
        VkDescriptorSet       debugResolveDescSet_        = VK_NULL_HANDLE;

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
        bool denoiseEnabled_ = []() {
            const char* e = std::getenv("THREEPP_DENOISE");
            return !(e && e[0] == '0');
        }();

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

        // ── Hybrid raster G-buffer prepass ──────────────────────────────────
        // Replaces the old PT primary-ray traversal: raster writes
        // depth/normal/motion/IDs into per-frame attachments; the deferred
        // shade reads them and starts its ray-query work at bounce 1.
        // Eliminates moving-object shake from stochastic primary-ray jitter
        // and makes primary visibility deterministic per pixel. AA happens
        // via TAA on top of raster, not as Monte Carlo on the primary ray.
        // Disabled by default until the integration is validated end-to-end
        // (stage 1).
        struct RasterGbufImages {
            Image2D       normal;       // rgba16f — world-space normal in xyz, .w = linear roughness
            Image2D       motion;       // rgba16f — NDC delta in .rg, .ba reserved
            Image2D       ids;          // rgba16ui — instanceCustomIndex/meshID/flags/reserved
            Image2D       uv;           // rgba16f — material UV in .rg
            Image2D       albedo;       // rgba8 unorm — linear base colour in .rgb, metalness in .a (raster-first deferred input)
            Image2D       indirect;     // rgba16f — demodulated diffuse-indirect irradiance (deferred denoiser scratch; STORAGE, not an attachment)
            Image2D       momentsSq;    // rg32f — .r = temporally-accumulated E[L²] of the indirect luminance (SVGF variance: var = E[L²] - lum(indirect)²; r32 because the square overflows fp16 at lum≈256); .g = GI content-change trend (antilag: stale moving-object contact-darkening/bounce fades at content rate); STORAGE+SAMPLED, ping-ponged like indirect
            Image2D       atrousA;      // rgba16f — SVGF multi-pass à-trous ping-pong (rgb=GI, a=variance); STORAGE scratch
            Image2D       atrousB;      // rgba16f — SVGF multi-pass à-trous ping-pong (the other half)
            Image2D       reflect;      // rgba16f — sharp 1-mirror-ray reflection radiance (.rgb), demodulated; roughness-blurred by the reflection denoise. STORAGE
            Image2D       reflAux;      // rgba16f — reflection-denoiser auxiliary (ping-pong, mirrors `reflect`: STORAGE write + SAMPLED prev-frame read)
            Image2D       shadowVis;    // rgba16f — denoised-shadow channel accumulator (.x=visibility ratio, .y=E[R²], .z=histLen, .w=trend); STORAGE + SAMPLED, ping-ponged like indirect
            Image2D       directU;      // rgba16f — unshadowed analytic direct (dir/point/spot) for the denoise recombine (U × R̃); STORAGE, current frame only
            Image2D       shadowAtrousA;// rg16f — shadow-ratio à-trous ping-pong (x=R, y=variance); STORAGE scratch
            Image2D       shadowAtrousB;// rg16f — shadow-ratio à-trous ping-pong (the other half)
            Image2D       froxelScatter;// rgba16f 3D (128×72×64, FIXED size) — froxel in-scatter accumulator (.a=histLen); STORAGE + SAMPLED, ping-ponged like indirect
            Image2D       froxelLut;    // rgba16f 3D — front-to-back-integrated volumetric LUT; STORAGE (integrate) + SAMPLED (shade, trilinear)
            Image2D       cloudColor;   // rgba16f HALF-res — cloud march result (rgb=in-scatter, a=transmittance); STORAGE (march) + SAMPLED (shade upsample + prev-fif reproject), ping-ponged
            Image2D       cloudAux;     // rg16f HALF-res — cloud mean-depth (.r) + temporal histLen (.g); STORAGE (march) + SAMPLED (prev-fif history), ping-ponged
            Image2D       cloudShadow;  // r8 512² (FIXED) — top-down cloud transmittance over an 8 km camera-centred square; STORAGE (shadow pass) + SAMPLED (surface/froxel/water sun); regenerated per frame
            Image2D       depth;        // d32_sfloat — JITTERED projection (matches color attachments above; consumed by chit + TAA)
            // Hybrid raster overlay's UNJITTERED depth attachment. Filled by
            // an extra depth-only prepass (overlay_depth.vert) right after
            // the main G-buffer pass. The wireframe overlay reads it as a
            // depth attachment so its depth test compares unjittered z
            // against unjittered z and doesn't shimmer between frames.
            Image2D       unjitDepth;   // d32_sfloat — UNJITTERED projection
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            uint32_t      width = 0;
            uint32_t      height = 0;

            // ── MSAA raster targets (only allocated when gbufMsaaSamples_ > 1) ──
            // True multisampled siblings of normal/motion/ids/uv/albedo/depth,
            // rasterized at VK_SAMPLE_COUNT_{2,4}_BIT. The single-sample images
            // above stay allocated unchanged and become the RESOLVE TARGETS: every
            // existing consumer (DeferredShade, TaaResolve, the shade's hybrid
            // set, debug blit) keeps reading them, unaware MSAA is involved. gbuf_resolve
            // (GbufResolve.{hpp,cpp}) picks the per-pixel dominant sample (majority
            // instance id, reversed-Z-nearest tie-break) and writes it into the
            // single-sample images; a tiny depth-only fullscreen pass resolves depth
            // (compute can't write a depth-aspect image). See setGbufferMsaa.
            Image2D       normalMS;
            Image2D       motionMS;
            Image2D       idsMS;
            Image2D       uvMS;
            Image2D       albedoMS;
            Image2D       depthMS;
            VkFramebuffer framebufferMS = VK_NULL_HANDLE;// MS render target (rasterGbufRenderPassMS)
        };
        std::array<RasterGbufImages, kFramesInFlight> rasterGbufs{};
        VkRenderPass rasterGbufRenderPass = VK_NULL_HANDLE;
        // MSAA render pass, keyed by sample count (2 or 4). Only the pass
        // matching gbufMsaaSamples_ is ever created; the other stays
        // VK_NULL_HANDLE. Kept separate from rasterGbufRenderPass (the 1×
        // path) so the default (msaa=1) code path is 100% untouched.
        VkRenderPass rasterGbufRenderPassMS = VK_NULL_HANDLE;
        VkPipeline   rasterGbufPipelineMS         = VK_NULL_HANDLE;
        VkPipeline   rasterGbufIndirectPipelineMS = VK_NULL_HANDLE;
        VkPipeline   rasterGbufDecalPipelineMS    = VK_NULL_HANDLE;
        // Sample count backing the MS pipelines/render pass/images above (0
        // until first created; tracks which count they were built for, so a
        // 2→4 change knows to tear down and rebuild rather than reuse).
        VkSampleCountFlagBits gbufMsaaBuiltSamples_ = VK_SAMPLE_COUNT_1_BIT;
        // 1x1 dummy MS images (5, mirroring normal/depth/ids/uv/albedo) bound
        // to deferred_shade.comp's dispatch-B sampler2DMS bindings when
        // gbufMsaaSamples_ == 1 — sampler2DMS is a distinct SPIR-V type from
        // sampler2D, so (unlike other "1x1 dummy" bindings elsewhere) this
        // can't reuse a single-sample dummy; a real 2-sample image is the
        // minimum valid stand-in. Created once, lazily, on first use.
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
        VkDescriptorPool      rasterDescPool       = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> rasterDescSets{};
        // Per-frame draw info ring. Each entry mirrors the GLSL DrawInfo
        // struct in gbuffer_indirect.vert: model matrix + buffer device
        // addresses + flags. Sized lazily; grows on demand.
        std::array<Buffer, kFramesInFlight> drawInfoBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> drawInfoBufferCapacity{};
        // Per-frame indirect command ring. Holds a contiguous array of
        // VkDrawIndirectCommand structs partitioned by cull mode (Front,
        // then Back, then Double). Counts + offsets per group recorded in
        // indirectGroupRanges_ during recordRasterGbufPass.
        std::array<Buffer, kFramesInFlight> indirectCmdBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> indirectCmdBufferCapacity{};

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

        // ── Masked overlay edge-AA ──────────────────────────────────────────
        // The overlay renders post-TAA onto the 1-sample swapchain, so its
        // vector edges (SVG fills, lines, wireframes) get no AA from either
        // MSAA (no multisampled swapchains in Vulkan) or TAA (deliberately
        // after it, to avoid ghosting). Instead of MSAA'ing the whole pass
        // (4x depth prepass + raster), every overlay pipeline writes a 1-byte
        // coverage mask as attachment 1, and a fullscreen FXAA-style pass
        // afterwards edge-blends ONLY the masked pixels (dilated 1 px). The
        // TAA-resolved scene is untouched. Gated on Canvas antialiasing > 1.
        // srcTex is a swapchain copy (a pass can't sample its own target).
        Image2D               overlayAaMask_{};   // R8 coverage, swapchain-sized
        Image2D               overlayAaScratch_{};// post-overlay swapchain copy
        VkDescriptorSetLayout overlayAaSetLayout_      = VK_NULL_HANDLE;
        VkDescriptorPool      overlayAaPool_           = VK_NULL_HANDLE;
        VkDescriptorSet       overlayAaSet_            = VK_NULL_HANDLE;
        VkPipelineLayout      overlayAaPipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            overlayAaPipeline_       = VK_NULL_HANDLE;

        // Lazily (re)size the mask + scratch to the swapchain extent and
        // point overlayAaSet_ at them. Extent only changes across a swapchain
        // recreate (device drained) and the first call precedes any submit
        // that references the set, so the descriptor update never races the
        // GPU. Old images go through the frame-serial retire queue.
        void ensureOverlayAaImages(VkExtent2D ext) {
            if (overlayAaMask_.image != VK_NULL_HANDLE &&
                overlayAaMask_.width == ext.width && overlayAaMask_.height == ext.height) return;
            if (overlayAaMask_.image != VK_NULL_HANDLE) retire(std::move(overlayAaMask_));
            if (overlayAaScratch_.image != VK_NULL_HANDLE) retire(std::move(overlayAaScratch_));
            auto make = [&](VkFormat fmt, VkImageUsageFlags usage, const char* name) {
                Image2D out{};
                out.width  = ext.width;
                out.height = ext.height;
                out.format = fmt;
                VkImageCreateInfo ici{};
                ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType     = VK_IMAGE_TYPE_2D;
                ici.format        = fmt;
                ici.extent        = {ext.width, ext.height, 1};
                ici.mipLevels     = 1;
                ici.arrayLayers   = 1;
                ici.samples       = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
                ici.usage         = usage;
                ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO;
                check(vmaCreateImage(ctx->allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
                      "vmaCreateImage(overlayAa)");
                VkImageViewCreateInfo vci{};
                vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image    = out.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format   = fmt;
                vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                vci.subresourceRange.levelCount = 1;
                vci.subresourceRange.layerCount = 1;
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                      "vkCreateImageView(overlayAa)");
                VkSamplerCreateInfo sci{};
                sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                sci.magFilter    = VK_FILTER_LINEAR;
                sci.minFilter    = VK_FILTER_LINEAR;
                sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                check(vkCreateSampler(ctx->device(), &sci, nullptr, &out.sampler),
                      "vkCreateSampler(overlayAa)");
                ctx->setObjectName(out.image, name);
                return out;
            };
            overlayAaMask_ = make(VK_FORMAT_R8_UNORM,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                  "overlayAaMask");
            // The swapchain-sized scratch copy is only needed when the AA
            // pass actually runs (Canvas antialiasing on); the mask must
            // exist regardless — the overlay pipelines declare it.
            if (canvas.samples() <= 1) return;
            overlayAaScratch_ = make(ctx->swapchainFormat(),
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                     "overlayAaScratch");
            if (overlayAaSet_ != VK_NULL_HANDLE) {
                VkDescriptorImageInfo ii[2]{};
                ii[0] = {overlayAaScratch_.sampler, overlayAaScratch_.view,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                ii[1] = {overlayAaMask_.sampler, overlayAaMask_.view,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet ws[2]{};
                for (uint32_t i = 0; i < 2; ++i) {
                    ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    ws[i].dstSet          = overlayAaSet_;
                    ws[i].dstBinding      = i;
                    ws[i].descriptorCount = 1;
                    ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    ws[i].pImageInfo      = &ii[i];
                }
                vkUpdateDescriptorSets(ctx->device(), 2, ws, 0, nullptr);
            }
        }

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
        struct GpuOverlayFogUbo {
            float fogActive;     // >0.5 = a medium is present this frame
            float hfDensity;     // air-medium σ_t at baseY (0 = no air fog)
            float hfBaseY;
            float hfFalloff;     // huge ≈ uniform
            float murkDensity;   // underwater-murk σ_t (0 = off)
            float waterSurfaceY; // world Y of the water surface (murk clip)
            float camWorldY;     // camera world Y
            float _pad0;
            float viewToWorldY[3];// world-Y row of the inverse-view
            float _pad1;
            float fogInscatter[3];// LINEAR air-fog in-scatter radiance (fade target)
            float _pad2;
            float murkInscatter[3];// LINEAR murk in-scatter radiance (fade target)
            float _pad3;
            // Lit particles (particle_light.comp): the LIT fragment path needs
            // the scene's display transform to land in the same domain as the
            // tonemapped background it alpha-blends over.
            float litActive;   // >0.5 = particle_light.comp ran this frame
            float exposure;    // FULL tone-map exposure (currentExposure())
            float toneMapMode; // threepp::ToneMapping as float (frag casts back)
            float _pad4;
        };
        static_assert(sizeof(GpuOverlayFogUbo) == 96);
        VkDescriptorSetLayout overlayFogDescSetLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      overlayFogDescPool_      = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> overlayFogDescSets_{};
        std::array<Buffer, kFramesInFlight>          overlayFogUbos_{};

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
        void ensureParticleIoSets() {
            if (particleIoDescPool_ != VK_NULL_HANDLE || !deferredShade_ ||
                particleCenterBufs_[0].handle == VK_NULL_HANDLE) return;
            VkDescriptorSetLayout ioLayout = deferredShade_->particleIoLayout();
            if (ioLayout == VK_NULL_HANDLE) return;
            VkDescriptorPoolSize ips{};
            ips.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ips.descriptorCount = 2 * kFramesInFlight;
            VkDescriptorPoolCreateInfo ipci{};
            ipci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ipci.maxSets       = kFramesInFlight;
            ipci.poolSizeCount = 1;
            ipci.pPoolSizes    = &ips;
            check(vkCreateDescriptorPool(ctx->device(), &ipci, nullptr, &particleIoDescPool_),
                  "vkCreateDescriptorPool(particleIo)");
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                VkDescriptorSetAllocateInfo asi{};
                asi.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                asi.descriptorPool     = particleIoDescPool_;
                asi.descriptorSetCount = 1;
                asi.pSetLayouts        = &ioLayout;
                check(vkAllocateDescriptorSets(ctx->device(), &asi, &particleIoDescSets_[f]),
                      "vkAllocateDescriptorSets(particleIo)");
                VkDescriptorBufferInfo cbi{};
                cbi.buffer = particleCenterBufs_[f].handle;
                cbi.offset = 0;
                cbi.range  = VK_WHOLE_SIZE;
                VkDescriptorBufferInfo obi{};
                obi.buffer = particleLightBufs_[f].handle;
                obi.offset = 0;
                obi.range  = VK_WHOLE_SIZE;
                VkWriteDescriptorSet w[2]{};
                w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[0].dstSet          = particleIoDescSets_[f];
                w[0].dstBinding      = 0;
                w[0].descriptorCount = 1;
                w[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[0].pBufferInfo     = &cbi;
                w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[1].dstSet          = particleIoDescSets_[f];
                w[1].dstBinding      = 1;
                w[1].descriptorCount = 1;
                w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[1].pBufferInfo     = &obi;
                vkUpdateDescriptorSets(ctx->device(), 2, w, 0, nullptr);
            }
        }

        // Gather every visible ParticleSystem's particle centers (model → world,
        // every 4th vertex of the coincident-quad layout) into this frame's
        // centers buffer and assign each mesh its base index, in the SAME order
        // the overlay loop draws them. Runs in recordCommandBuffer BEFORE the
        // scene-dispatch hook; the hook dispatches particle_light.comp over the
        // result. No-op (count 0) when the scene has no particles, the billboard
        // pipeline isn't initialised yet, or the IO sets can't exist (no
        // deferred shade → nothing would ever light the result buffer).
        void prepareParticleLighting() {
            particleLightCount_ = 0;
            particleLitBase_.clear();
            ensureParticleIoSets();
            if (particlePipelineNormal_ == VK_NULL_HANDLE ||
                particleIoDescSets_[currentFrame] == VK_NULL_HANDLE ||
                particleCenterBufs_[currentFrame].handle == VK_NULL_HANDLE) return;

            particleCenterScratch_.clear();
            uint32_t base = 0;
            for (const auto& en : lastVisibleEntries_) {
                if (!en.isParticle || !en.mesh) continue;
                auto geomSp = en.mesh->geometry();
                BufferGeometry* geom = geomSp.get();
                if (!geom) continue;
                auto* posAttr = geom->getAttribute<float>("position");
                if (!posAttr) continue;
                const uint32_t vtx = static_cast<uint32_t>(posAttr->count());
                const uint32_t particles = vtx / 4u;// 4 coincident verts per particle
                if (particles == 0) continue;
                if (base + particles > kMaxLitParticles) continue;// draw falls back unlit

                const auto& w = en.worldMatrix;// column-major world transform
                const auto& p = posAttr->array();
                particleCenterScratch_.resize(static_cast<size_t>(base + particles) * 4);
                float* dst = particleCenterScratch_.data() + static_cast<size_t>(base) * 4;
                for (uint32_t i = 0; i < particles; ++i) {
                    const float x = p[i * 12u + 0u];// vertex 4i of particle i
                    const float y = p[i * 12u + 1u];
                    const float z = p[i * 12u + 2u];
                    dst[i * 4u + 0u] = w[0] * x + w[4] * y + w[8] * z + w[12];
                    dst[i * 4u + 1u] = w[1] * x + w[5] * y + w[9] * z + w[13];
                    dst[i * 4u + 2u] = w[2] * x + w[6] * y + w[10] * z + w[14];
                    dst[i * 4u + 3u] = 0.f;
                }
                particleLitBase_[en.mesh] = base;
                base += particles;
            }
            if (base == 0) return;
            uploadHostVisible(ctx->allocator(), particleCenterBufs_[currentFrame],
                              particleCenterScratch_.data(),
                              particleCenterScratch_.size() * sizeof(float));
            particleLightCount_ = base;
        }

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
        std::vector<WorldSpriteEntry> lastVisibleSprites_;

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
        std::vector<LineEntry> lastVisibleLines_;
        // Cached unjittered view-projection matrix (column-major,
        // row-of-element-4 layout). Computed once per frame in
        // uploadRasterCameraUbo and read by recordOverlayPass to build
        // the per-draw mvp = vpUnjit · model push constant.
        std::array<float, 16> currVPunjit_{};
        // Cached unjittered view and reverse-Z projection matrices, mirrored
        // alongside currVPunjit_ each frame. The particle billboard pass needs
        // them SEPARATELY (not the combined VP): the distance-attenuated
        // billboard scale uses view-space depth and proj[1][1] individually, so
        // it pushes modelView = currViewUnjit_ · meshWorld and currProjUnjit_.
        std::array<float, 16> currViewUnjit_{};
        std::array<float, 16> currProjUnjit_{};

        // Per-frame raster camera data. currVPjittered drives gl_Position;
        // currVPunjittered + prevVP drive the motion-vector computation
        // (which must be jitter-free or motion vectors include the jitter
        // and pollute reproject).
        struct RasterCameraData {
            float currVPjittered[16];
            float currVPunjittered[16];
            float prevVP[16];
            float jitter[4];          // .xy = clip-space sub-texel offset, .zw = 1/resolution
            float prevJitter[4];      // .xy = previous frame's jitter. NOTE: gbuffer.frag's
                                      // motion vec is JITTER-FREE (clean prevNDC − currNDC
                                      // from the unjittered VPs — a (prev−curr) jitter delta
                                      // was tested and rejected there); .xy is kept for the
                                      // deferred shade's hybrid reproject tap correction.
                                      // .z smuggles the normal-map Toksvig toggle.
        };
        std::array<Buffer, kFramesInFlight> rasterCameraUbos{};
        bool  rasterPrevVPValid_ = false;
        float rasterPrevVP_[16]{};
        // prevVPunjit · currVPunjit⁻¹ for the TAA's sky-motion reconstruction
        // (sky rasterizes nothing → zero motion → wrong reproject under
        // camera rotation). Computed in uploadRasterCameraUbo while both
        // VPs are in hand; identity until the first real frame.
        std::array<float, 16> taaSkyReproj_{1.f, 0.f, 0.f, 0.f,
                                            0.f, 1.f, 0.f, 0.f,
                                            0.f, 0.f, 1.f, 0.f,
                                            0.f, 0.f, 0.f, 1.f};
        // Reverse-Z view-depth linearization (A,B,C,D) for the TAA depth
        // disocclusion gate: viewZ = (A·d + B)/(C·d + D). Set each frame in
        // uploadRasterCameraUbo from the inverse reverse-Z projection. Zero
        // until the first real frame ⇒ shader leaves the depth gate disabled.
        std::array<float, 4> taaDepthLin_{};
        // This frame's Halton jitter in RENDER TEXELS for the TAA resolve's
        // current-sample jitter cancellation (taa_resolve re-anchors every
        // current-frame read at the unjittered pixel center — without it the
        // composed output translates with the 8-phase jitter: the systemic
        // "everything shakes"). {0, 0} whenever the raster renders unjittered
        // (MSAA mode, event camera) — the resolve then runs bit-identical to
        // its pre-cancellation arithmetic. Set in uploadRasterCameraUbo.
        float taaJitterTexels_[2]{};
        float rasterPrevJitter_[2]{};
        bool  rasterPrevJitterValid_ = false;
        // Camera WORLD motion this frame (translation m, forward-rotation rad)
        // for the deferred reflection history policy: a chase-cam surface (car
        // sunroof with a following camera) is screen-STATIONARY — its motion
        // vectors are ~0 — while its view-dependent reflection content slides
        // with every meter the camera travels. Screen-space motion alone cannot
        // see camera+object co-motion; these can.
        float deferredCamPrevPos_[3]{};
        float deferredCamPrevFwd_[3]{};
        bool  deferredCamPrevValid_ = false;
        float deferredCamDeltaLen_ = 0.f;
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

        // ── Raster TAA resolve ─────────────────────────────────────────────
        // Encapsulated in vulkan/TaaResolve.{hpp,cpp}. Owns its pipeline,
        // descriptor pool/sets, sampler, input image + history ping-pong.
        // External deps (raster gbuffer views, swapchain views) are passed
        // in at descriptor-write time.
        std::unique_ptr<vulkan::TaaResolve> taa_;
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
            if (taa_) taa_->invalidateHistory();
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
            if (taa_) taa_->invalidateHistory();
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
        // HDR bloom pyramid — the shade/resolve writes linear HDR into
        // bloom_->sceneHdr (the shared set's binding 1); bloom_ builds the
        // Jimenez progressive pyramid from it. bloomIntensity_ == 0 skips
        // the pyramid entirely.
        std::unique_ptr<vulkan::BloomPass> bloom_;
        // Exposure / white balance / tone map (incl. AgX) / grade LUT / sRGB
        // — the camera/display end of the post stack, one dispatch into the
        // TAA input (post_composite.comp). Owns the white-balance matrix and
        // the 33³ grade LUT (setWhiteBalance / setColorGrade forward here).
        std::unique_ptr<vulkan::PostComposite> post_;
        // Thin-lens depth of field on sceneHdr, recorded between the scene
        // dispatch and the bloom pyramid so bokeh still blooms + tone-maps
        // as HDR (see vulkan/DofPass.hpp). CoC is camera-derived: aperture
        // from camAperture_, focal length from tanHalfFovY_ on a 24 mm
        // sensor, focus plane at focusDistance_. OFF by default (the whole
        // pass is skipped; zero cost).
        std::unique_ptr<vulkan::DofPass> dof_;
        bool  dofEnabled_    = false;
        float focusDistance_ = 10.f;
        float tanHalfFovY_   = 0.4142f;// stashed in updateCameraUbo (45° default)
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
        // DEPRECATED (Phase 2 fog unification): the directional sun shafts / aerial
        // glow are now ALWAYS on when the fog medium is present (froxels own the
        // near field, volumetricDirScatter the > 512 m tail — no opt-in). Kept as a
        // no-op toggle so setVolumetricFog() callers still compile; default true so
        // the deprecated getter reads "on when fog is present".
        bool  deferredVolFog_     = true;
        // Procedural direction-space star field on deferred sky pixels (0 = off).
        float deferredStarIntensity_ = 0.f;

        // Deferred shade pass. Shades the material G-buffer analytically into
        // bloom_->sceneHdr; bloom + TAA finish the frame. Owns its own focused
        // descriptor set (see DeferredShade.hpp).
        std::unique_ptr<vulkan::DeferredShade> deferredShade_;
        // World-space irradiance probe grid (multi-bounce GI for the deferred
        // gather — see vulkan/ProbeGI.hpp). Created alongside deferredShade_;
        // its SH buffer + grid UBO back the deferred set's bindings 36/37, so
        // it must exist whenever deferredShade_ does. probeGIEnabled_ gates the
        // per-frame update dispatch AND the shader-side sampling (grid UBO
        // enable) — default OFF (opt-in via VulkanRenderer::setProbeGI).
        // probeGridDirty_ re-fits the grid to the scene AABB after each
        // structural scene rebuild (new/removed meshes ⇒ new bounds).
        std::unique_ptr<vulkan::ProbeGI> probeGI_;
        bool probeGIEnabled_  = true;
        bool probeGridDirty_  = true;
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
        float bloomThreshold_ = 1.0f;// soft-knee bright-pass cutoff (linear HDR)
        float bloomClamp_ = 0.0f;    // per-tap HDR cap before the bright pass; <= 0 = off
        float sharpenStrength_ = 0.5f;// post-TAA RCAS amount; 0 = off
        float motionBlurAmount_ = 0.f;// post-TAA motion blur: shutter open fraction
                                      // of the frame interval (0.5 = 180°); 0 = off
        float taaBlendAlpha_ = 0.16f;// 10% current, 90% history at the reference rate;
                                    // frame-rate-corrected per frame (see taaPrevTimeSec_)
        // Wall-clock anchor for the frame-rate-aware TAA blend. taaBlendAlpha_ is a
        // per-FRAME new-sample weight, so holding it fixed ties the history half-life
        // to frame COUNT: the same 90 %/frame retention is an invisible ~10 ms ghost
        // at 200 fps but a long visible smear on a 30 fps (or vsync-capped + heavy)
        // frame — the "moving object leaves edge trails" report. Each frame the
        // weight is re-solved (in recordCommandBuffer, against kTaaRefFps) so
        // (1-alpha) is held constant in wall-clock time instead of per frame; the
        // shader's velocity/deviation gates are already per-frame-displacement based,
        // so only this base weight needs the correction.
        double taaPrevTimeSec_ = -1.0;

        // Per-frame-slot gate for the raster descriptor's binding 3 — the
        // 2048-entry bindless material-texture array. Its contents are
        // identical every frame and only change when the scene texture table
        // is rebuilt. Rewriting all 2048 entries every frame burned a
        // vkUpdateDescriptorSets call + a ~48 KB host array fill for nothing.
        // 1 = current, 0 = needs (re)write; value-inits to 0 so the first
        // frame writes it. Invalidated (->0) at scene (re)build; each slot
        // rewrites on its next uploadRasterCameraUbo, before
        // recordCommandBuffer binds the set. rasterDescSets live in their own
        // pool (rasterDescPool, init-only) so swapchain / main-pool rebuilds
        // don't affect them — only a texture-table change does.
        std::array<int8_t, kFramesInFlight> rasterMatTexValid_{};

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
        VkExtent2D renderExtent() const {
            const VkExtent2D s = ctx->swapchainExtent();
            if (renderScale_ >= 0.999f) return s;
            const auto px = [](uint32_t v, float k) -> uint32_t {
                const auto r = static_cast<uint32_t>(static_cast<float>(v) * k + 0.5f);
                return r < 1u ? 1u : r;
            };
            return {px(s.width, renderScale_), px(s.height, renderScale_)};
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
        bool sceneHasOverlayContent() const {
            if (!lastVisibleLines_.empty()) return true;
            // World-space sprites are drawn in the overlay pass and depth-test
            // against unjitDepth, so they need the prepass + overlay pass too.
            if (!lastVisibleSprites_.empty()) return true;
            for (const auto& en : lastVisibleEntries_) {
                // isOverlay covers particle billboards too (kSnapParticle folds
                // into isOverlay), so a scene with only particles still triggers
                // the depth prepass + overlay pass the billboard loop needs.
                if (en.isOverlay) return true;
            }
            return false;
        }
        // Free-running sub-pixel jitter sequence index for raster TAA. The
        // active Halton(2,3) period is derived per frame from the upscale ratio
        // (see jitterPhaseCount_) and applied as a modulo at the read sites, so
        // the sequence length tracks renderScale (8 at native, more when
        // upscaling) instead of a fixed 16.
        uint32_t haltonFrame_ = 0;
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

        // Per-frame-in-flight camera UBO (viewInverse + projInverse).
        // 2 mat4 packed back-to-back, std140 layout.
        std::array<Buffer, kFramesInFlight> cameraUbos{};

        uint32_t imageCount_ = 0;

        // Per-frame command resources.
        VkCommandPool                                cmdPool = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, kFramesInFlight> cmdBuffers{};
        std::array<VkSemaphore,     kFramesInFlight> imageAvailable{};
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
        void drainRetireQueue() { retireQueue_.drain(*ctx, frameSerial_, kFramesInFlight); }
        // Post-vkDeviceWaitIdle flush: whole device idle ⇒ destroy all queued
        // resources now, or they leak at device destroy. Call after EVERY
        // remaining vkDeviceWaitIdle that precedes resource destruction.
        void flushRetireQueue() { retireQueue_.flushAll(*ctx); }
        // Retire (rather than inline-destroy) a particle geometry record's
        // buffers on an in-flight topology change. Mirrors destroyParticleGeomRec.
        void retireParticleGeomRec(ParticleGeomRec& rec) {
            retire(std::move(rec.position));
            retire(std::move(rec.normal));
            retire(std::move(rec.uv));
            retire(std::move(rec.color));
            retire(std::move(rec.index));
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

        // Deferred-apply queue for setters that issue vkDeviceWaitIdle +
        // realloc descriptor/image resources. These collide with an open
        // cmd buffer (frameState_ != Idle), so when called mid-frame we
        // stash the request and apply it at the next beginDeferredFrame —
        // after the prior frame's fence wait, before any record. Setters
        // called when Idle apply immediately as before.
        bool pendingRenderScaleRealloc_ = false;
        bool pendingAccumulationReset_  = false;
        // Raster G-buffer MSAA sample count (1/2/4). 1 = today's single-sample
        // path, byte-identical output, default. See setGbufferMsaa. Reallocation
        // (render pass + pipelines + MS images) is render-extent-resource work,
        // so it reuses the exact same idle-gate as renderScale.
        uint32_t gbufMsaaSamples_        = 1;

        explicit CoreImpl(Canvas& c) : canvas(c), size(c.size()) {
            ctx = std::make_unique<VulkanContext>(
                    static_cast<GLFWwindow*>(canvas.windowPtr()),
                    /*enableRayTracing*/ true,
                    /*vsync*/ canvas.vsync());

            // The scene-dependent AS build runs lazily on the first render()
            // call. Everything below is scene-independent and safe at ctor time.
            createCommandResources();
            createCameraUbos();
            createLightsUbos();
            createFogUbos();
            createCloudUbos();
            // EnvPrefilter owns the PMREM compute pipeline + descriptor pool.
            // Construct before createDefaultEnvImage so the env upload path is
            // ready if scene.environment is set before the first render().
            envPrefilter_ = std::make_unique<vulkan::EnvPrefilter>(*ctx, cmdPool);
            createDefaultEnvImage();
            createTextureSampler();
            createDefaultMaterialTexture();
            // Allocates the ReSTIR DI reservoir ping-pong images the deferred
            // shade's NEE consumes.
            createReservoirImages();
            skinning_ = std::make_unique<vulkan::SkinningPipeline>(*ctx);
            tetSkinning_ = std::make_unique<vulkan::TetSkinningPipeline>(*ctx);
            waterDisplace_ = std::make_unique<vulkan::WaterDisplacePipeline>(*ctx);
            foamWorld_     = std::make_unique<vulkan::FoamWorldPipeline>(*ctx);
            grassWind_     = std::make_unique<vulkan::GrassWindPipeline>(*ctx);
            // Hybrid raster G-buffer infrastructure. Costs a few hundred MB
            // at 1080p for six attachments × kFramesInFlight.
            ensureHybridResources();
            // TAA pipeline + images. The deferred shade's descriptor binding 1
            // (shade output target) always points at the TAA input view.
            imageCount_ = static_cast<uint32_t>(ctx->swapchainImages().size());
            taa_ = std::make_unique<vulkan::TaaResolve>(
                    *ctx, cmdPool, imageCount_, kFramesInFlight);
            {
                // TAA input is the deferred render extent; history +
                // output are the swapchain extent. When they differ the
                // resolve pass runs as a temporal upsampler.
                const VkExtent2D inExt  = renderExtent();
                const VkExtent2D outExt = ctx->swapchainExtent();
                taa_->createImages(inExt.width, inExt.height,
                                   outExt.width, outExt.height);
            }
#if defined(THREEPP_WITH_DLSS)
            // DLSS feature at the display extent. Created BEFORE FSR because DLSS
            // OUTRANKS it (useFsr() requires !useDlss()): an FSR context built
            // alongside an active DLSS feature could never dispatch, so we skip it
            // below and save its context (~60 MB committed here). NGX feature
            // creation records GPU init work — one-shot submit+wait. On NGX
            // failure (non-RTX GPU / old driver) dlssActive_ stays false and the
            // FSR/TAA fallback runs.
            dlss_ = std::make_unique<vulkan::DlssUpscaler>(*ctx, kFramesInFlight);
            {
                VkCommandBuffer initCb = beginOneShot();
                dlssActive_ = dlss_->create(initCb,
                                            ctx->swapchainExtent().width,
                                            ctx->swapchainExtent().height,
                                            renderExtent().width,
                                            renderExtent().height);
                endAndSubmitOneShot(initCb, "DLSS feature create");
            }
            dlssResetNext_ = true;
#endif
#if defined(THREEPP_WITH_FSR)
            // FSR upscaler context at the display (swapchain) extent. FSR stores
            // the display extent at create time but takes the render extent per
            // dispatch, so a renderScale change needs no recreation — only a
            // swapchain/display resize does (handled in reallocateRenderExtentResources).
            // On success FSR replaces the TAA resolve; on failure the TAA path
            // runs unchanged. Skipped entirely when DLSS already claimed the
            // upscaler slot (it could never dispatch); if DLSS is disabled at
            // runtime, setDlss(false) creates the FSR context on demand.
            bool dlssClaimedUpscaler = false;
#if defined(THREEPP_WITH_DLSS)
            dlssClaimedUpscaler = dlssActive_;
#endif
            if (!dlssClaimedUpscaler) {
                fsr_ = std::make_unique<vulkan::FsrUpscaler>(*ctx, kFramesInFlight);
                fsrActive_ = fsr_->create(ctx->swapchainExtent().width,
                                          ctx->swapchainExtent().height);
                fsrResetNext_ = true;
            }
#endif
            // HDR bloom pyramid. sceneHdr lives at the deferred render
            // extent (it is the shared set's binding 1 target); the bloom
            // pyramid levels are half that and below.
            bloom_ = std::make_unique<vulkan::BloomPass>(*ctx, cmdPool, kFramesInFlight);
            bloom_->createImages(renderExtent().width, renderExtent().height);
            onAfterBloomCreateImages();
            // Exposure/WB/tone-map/grade/sRGB composite → TAA input.
            post_ = std::make_unique<vulkan::PostComposite>(*ctx, cmdPool, kFramesInFlight);
            // Thin-lens DoF (images/descriptors fitted in rewriteBloomDescriptors).
            dof_ = std::make_unique<vulkan::DofPass>(*ctx, cmdPool, kFramesInFlight);
            // Raster-first deferred lighting pass. Writes bloom_->sceneHdr, so
            // it must exist after bloom_; its descriptors reference the camera /
            // lights UBOs, the env image, the raster gbuffer and sceneHdr — all
            // created above by this point.
            // The deferred base traces ray-query shadow rays, so only stand it
            // up when the device supports VK_KHR_ray_query. Without it,
            // deferredShade_ stays null if ray query is unavailable.
            if (ctx->rayQuerySupported()) {
                deferredShade_ = std::make_unique<vulkan::DeferredShade>(*ctx, kFramesInFlight);
                // Probe grid backs the deferred set's bindings 36/37 (dummy-
                // free: real buffers from construction, sampling gated by the
                // grid UBO's enable flag until setProbeGI(true)).
                probeGI_ = std::make_unique<vulkan::ProbeGI>(*ctx, kFramesInFlight);
            }
            createBlueNoiseImage_();// must run before descriptor writes (binding 27)
            createOceanFineDummy_();// must run before descriptor writes (binding 32)
            createOceanFoamDummy_();// must run before descriptor writes (binding 33)
            createFoamDetailImage_();// must run before descriptor writes (binding 45 + deferred 34)
            rewriteTaaDescriptors();// after ensureHybridResources gave us raster gbuf views
            rewriteBloomDescriptors();// bloom composite reads gbuf + writes the TAA input
            rewriteDeferredDescriptors();// raster-first deferred shade inputs
            gpuTimings_ = std::make_unique<vulkan::GpuTimings>(*ctx, kFramesInFlight);
            overlayPass_ = std::make_unique<vulkan::OverlayPass>(
                    *ctx, kFramesInFlight,
                    // Atlas uploads are recorded into the frame's own cb (no
                    // one-shot submit + queue drain — that stalled on every
                    // in-flight frame each time a HUD TextSprite re-rasterized).
                    [this](VkCommandBuffer cb, uint32_t w, uint32_t h, VkFormat fmt,
                           const void* pix, VkDeviceSize sz,
                           VkFilter filter,
                           VkSamplerAddressMode addrU,
                           VkSamplerAddressMode addrV,
                           const char* name) {
                        return createSampledImage2DInFrame(cb, w, h, fmt, pix, sz,
                                                           filter, addrU, addrV, name);
                    },
                    // Retire stale sprite atlases through the frame-serial queue
                    // (no per-swap vkDeviceWaitIdle). Stamped with the frame
                    // being recorded when OverlayPass::record runs.
                    [this](Image2D&& img) { retire(std::move(img)); });

            // Optional one-shot fixed-footprint dump. Everything constructed above
            // is scene-independent, so this is the renderer's baseline cost — the
            // number to watch as new features add persistent targets. Enable with
            // THREEPP_VK_MEMDUMP=1 (or call dumpMemoryStats() from app code).
            if (const char* e = std::getenv("THREEPP_VK_MEMDUMP"); e && *e && *e != '0') {
                dumpMemoryStats("post-init");
            }
        }

        // Print a VMA memory-usage summary to stderr: the allocator's reserved-vs-
        // live block totals and per-heap usage/budget. Cheap; for manual/debug use
        // (there is otherwise no memory introspection in the renderer). Reflects
        // all VMA allocations — G-buffer, history, denoiser scratch, upscalers,
        // and any scene AS/geometry live at the call site.
        void dumpMemoryStats(const char* tag = "") const {
            if (!ctx || ctx->allocator() == VK_NULL_HANDLE) return;
            const VkPhysicalDeviceMemoryProperties* mp = nullptr;
            vmaGetMemoryProperties(ctx->allocator(), &mp);
            VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
            vmaGetHeapBudgets(ctx->allocator(), budgets);
            VmaTotalStatistics stats{};
            vmaCalculateStatistics(ctx->allocator(), &stats);
            constexpr double MB = 1024.0 * 1024.0;
            std::fprintf(stderr,
                         "[threepp][vk-mem] %s: reserved %.1f MB in %u blocks, "
                         "live %.1f MB in %u allocations\n",
                         tag,
                         stats.total.statistics.blockBytes / MB,
                         stats.total.statistics.blockCount,
                         stats.total.statistics.allocationBytes / MB,
                         stats.total.statistics.allocationCount);
            const uint32_t heaps = mp ? mp->memoryHeapCount : 0u;
            for (uint32_t i = 0; i < heaps; ++i) {
                const bool devLocal =
                        (mp->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
                std::fprintf(stderr,
                             "[threepp][vk-mem]   heap %u%s: usage %.1f MB / budget %.1f MB\n",
                             i, devLocal ? " (device-local)" : "",
                             budgets[i].usage / MB, budgets[i].budget / MB);
            }
            std::fflush(stderr);
        }

        ~CoreImpl() {
            // Stop the auto-LOD background worker first — it's pure CPU (no
            // Vulkan handles touched), so this is safe even when ctx is
            // null. Must happen before the blasCache teardown below, which
            // destroys the same records' lodLevels AS/buffers the worker
            // could otherwise still be racing to enqueue results against
            // (drainLodResults only ever runs from ensureSceneBuilt, which
            // can't run concurrently with this destructor, but the queue
            // itself is shared state the worker thread still owns until
            // joined).
            {
                std::lock_guard<std::mutex> lk(lodJobMutex_);
                lodWorkerStop_ = true;
            }
            lodJobCv_.notify_all();
            if (lodWorker_.joinable()) lodWorker_.join();

            if (!ctx) return;
            VkDevice d = ctx->device();
            // If a frame was mid-record when the renderer was destroyed (the
            // canvas frame-end callback never got to fire, or the user tore
            // down the renderer from inside the animate lambda), close the
            // open cmd buffer without submitting. The reset-but-unsignaled
            // fence is fine — no queued work depends on it — and the cmd
            // pool destroy below releases the buffer regardless.
            if (frameState_ != FrameState::Idle) {
                vkEndCommandBuffer(cmdBuffers[currentFrame]);
                frameState_ = FrameState::Idle;
            }
            vkDeviceWaitIdle(d);
            // Device idle ⇒ nothing references retired resources. Destroy them
            // now or they leak at device destroy (VUID-vkDestroyDevice-device-
            // 05137 — the class of bug that bit lineGeomCache_ below).
            flushRetireQueue();

            for (auto s : imageAvailable) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto s : renderFinished) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto f : inFlight) if (f) vkDestroyFence(d, f, nullptr);
            if (cmdPool) vkDestroyCommandPool(d, cmdPool, nullptr);
            gpuTimings_.reset();// query pool destruction while device is still valid

            if (tlas) ctx->rt().destroyAccelerationStructure(d, tlas, nullptr);
            destroyBuffer(ctx->allocator(), tlasBuffer);
            for (auto& b : tlasInstancesBuffers) destroyBuffer(ctx->allocator(), b);
            destroyBuffer(ctx->allocator(), tlasRefitScratch_);
            for (auto& b : geometryDescsBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : materialDescsBuffers) destroyBuffer(ctx->allocator(), b);
            destroyBuffer(ctx->allocator(), sceneCaptureBuf_);
            destroyBuffer(ctx->allocator(), eventLumaBuf_);
            if (eventShadePipeline_)       vkDestroyPipeline(d, eventShadePipeline_, nullptr);
            if (eventShadePipelineLayout_) vkDestroyPipelineLayout(d, eventShadePipelineLayout_, nullptr);
            if (eventShadeDescPool_)       vkDestroyDescriptorPool(d, eventShadeDescPool_, nullptr);
            if (eventShadeDsLayout_)       vkDestroyDescriptorSetLayout(d, eventShadeDsLayout_, nullptr);
            if (debugResolvePipeline_)       vkDestroyPipeline(d, debugResolvePipeline_, nullptr);
            if (debugResolvePipelineLayout_) vkDestroyPipelineLayout(d, debugResolvePipelineLayout_, nullptr);
            if (debugResolveDescPool_)       vkDestroyDescriptorPool(d, debugResolveDescPool_, nullptr);
            if (debugResolveDsLayout_)       vkDestroyDescriptorSetLayout(d, debugResolveDsLayout_, nullptr);

            for (auto& [_, rec] : blasCache) {
                destroyBlasLodLevels(*rec);
                destroyBlasRecord(*rec);
            }
            blasCache.clear();

            for (auto& [_, st] : skinnedMeshStates) {
                // Destroy the GPU-skinning input buffers + scratch first;
                // BLAS buffers below.
                destroyBuffer(ctx->allocator(), st->baseVertex);
                destroyBuffer(ctx->allocator(), st->baseNormal);
                destroyBuffer(ctx->allocator(), st->skinIndex);
                destroyBuffer(ctx->allocator(), st->skinWeight);
                destroyBuffer(ctx->allocator(), st->boneMatrices);
                destroyBuffer(ctx->allocator(), st->blasScratch);
                if (st->blas) destroyBlasRecord(*st->blas);
            }
            skinnedMeshStates.clear();

            for (auto& [_, st] : tetMeshStates) {
                destroyBuffer(ctx->allocator(), st->tetIndex);
                destroyBuffer(ctx->allocator(), st->tetWeight);
                destroyBuffer(ctx->allocator(), st->baseNormal);
                destroyBuffer(ctx->allocator(), st->restInv0);
                destroyBuffer(ctx->allocator(), st->restInv1);
                destroyBuffer(ctx->allocator(), st->restInv2);
                destroyBuffer(ctx->allocator(), st->tetPos);
                vulkan::destroyExternalBuffer(d, st->tetPosExt);
                destroyBuffer(ctx->allocator(), st->blasScratch);
                if (st->blas) destroyBlasRecord(*st->blas);
            }
            tetMeshStates.clear();

            for (auto& [_, st] : displacedStates) {
                if (st->blas) destroyBlasRecord(*st->blas);
                if (st->scratchA.view  != VK_NULL_HANDLE) vkDestroyImageView(d, st->scratchA.view, nullptr);
                if (st->scratchA.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->scratchA.image, st->scratchA.alloc);
                if (st->foamImage.view  != VK_NULL_HANDLE) vkDestroyImageView(d, st->foamImage.view, nullptr);
                if (st->foamImage.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->foamImage.image, st->foamImage.alloc);
                destroyBuffer(ctx->allocator(), st->heightReadback);
                destroyBuffer(ctx->allocator(), st->heightReadback1);
                destroyBuffer(ctx->allocator(), st->heightReadback2);
                destroyBuffer(ctx->allocator(), st->foamDisturbBuffer);
                destroyBuffer(ctx->allocator(), st->wakeTrailBuffer);
                // Per-cascade Phillips / DynamicSpectrum / IFFT are RAII; their
                // destructors handle their own VkImage / VkPipeline / DSet cleanup.
            }
            displacedStates.clear();

            for (auto& [_, st] : grassStates) {
                if (st->blas) destroyBlasRecord(*st->blas);
                destroyBuffer(ctx->allocator(), st->restPos);
                destroyBuffer(ctx->allocator(), st->heightFrac);
            }
            grassStates.clear();

            for (auto& [_, st] : morphedMeshStates) {
                if (st->blas) destroyBlasRecord(*st->blas);
            }
            morphedMeshStates.clear();

            for (auto& b : cameraUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : lightsUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : clusterLightsBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : clusterGridBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : fogUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : cloudUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : motionMatBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : meshMovedBitsBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : emissiveTriBuffers) destroyBuffer(ctx->allocator(), b);
            destroyImage2D(ctx->allocator(), d, envImage);
            destroyImage2D(ctx->allocator(), d, blueNoiseImage);
            // 1x1 stand-ins bound to the MS gbuffer descriptor slots whenever
            // MSAA is off. Created once by ensureGbufDummyMS() (guarded by
            // gbufDummyMSCreated_) and, until now, destroyed nowhere at all —
            // 5 VkImage + 5 VkImageView leaked per device. They are not
            // recreated on resize, so this destructor is their only owner.
            for (auto& img : gbufDummyMS_) destroyImage2D(ctx->allocator(), d, img);
            destroyImage2D(ctx->allocator(), d, oceanFineHeightDummy);
            destroyImage2D(ctx->allocator(), d, oceanFoamDummy);
            destroyImage2D(ctx->allocator(), d, foamDetailImage);
            for (auto& img : reservoirPosImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : reservoirWImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : materialTextures) destroyImage2D(ctx->allocator(), d, img);
            materialTextures.clear();
            if (textureSampler_) vkDestroySampler(d, textureSampler_, nullptr);
            if (textureSamplerIso_) vkDestroySampler(d, textureSamplerIso_, nullptr);
            if (textureSamplerClamp_) vkDestroySampler(d, textureSamplerClamp_, nullptr);
            if (textureSamplerIsoClamp_) vkDestroySampler(d, textureSamplerIsoClamp_, nullptr);
            if (textureSamplerCustom_) vkDestroySampler(d, textureSamplerCustom_, nullptr);
            if (textureSamplerCustomClamp_) vkDestroySampler(d, textureSamplerCustomClamp_, nullptr);
            for (VkSampler s : parkedSamplers_) vkDestroySampler(d, s, nullptr);
            parkedSamplers_.clear();

            // EnvPrefilter owns its pipeline / layout / pool / sampler.
            envPrefilter_.reset();

            // GPU skinning teardown. Per-SkinnedMeshState buffers are
            // destroyed alongside the BLAS in the skinnedMeshStates clear
            // (see below); the shared pipeline + pool live in skinning_.
            skinning_.reset();
            tetSkinning_.reset();

            // Water displace pipeline owns its handles + sampler.
            waterDisplace_.reset();
            // World-space foam pipeline; foam ping-pong images are owned by
            // each DisplacedMeshState and destroyed there.
            foamWorld_.reset();
            // Grass-wind pipeline; per-mesh buffers freed in the grassStates
            // loop above.
            grassWind_.reset();

            // Hybrid raster G-buffer cleanup. Resources are lazy-created on
            // first render(); if render() was never called, all handles stay
            // VK_NULL_HANDLE and these calls become no-ops.
            destroyRasterGbufImages();
            for (auto& b : rasterCameraUbos)    destroyBuffer(ctx->allocator(), b);
            for (auto& b : drawInfoBuffers)     destroyBuffer(ctx->allocator(), b);
            for (auto& b : indirectCmdBuffers)  destroyBuffer(ctx->allocator(), b);
            if (rasterGbufPipeline)         vkDestroyPipeline(d, rasterGbufPipeline, nullptr);
            if (rasterGbufIndirectPipeline) vkDestroyPipeline(d, rasterGbufIndirectPipeline, nullptr);
            if (rasterGbufDecalPipeline)    vkDestroyPipeline(d, rasterGbufDecalPipeline, nullptr);
            if (rasterPipelineLayout)   vkDestroyPipelineLayout(d, rasterPipelineLayout, nullptr);
            if (rasterDsLayout)         vkDestroyDescriptorSetLayout(d, rasterDsLayout, nullptr);
            if (rasterDescPool)         vkDestroyDescriptorPool(d, rasterDescPool, nullptr);
            if (rasterGbufRenderPass)   vkDestroyRenderPass(d, rasterGbufRenderPass, nullptr);
            if (occlRenderPassA_)       vkDestroyRenderPass(d, occlRenderPassA_, nullptr);
            if (occlRenderPassB_)       vkDestroyRenderPass(d, occlRenderPassB_, nullptr);
            if (occlRenderPassAMS_)     vkDestroyRenderPass(d, occlRenderPassAMS_, nullptr);
            if (occlRenderPassBMS_)     vkDestroyRenderPass(d, occlRenderPassBMS_, nullptr);
            if (overlayWireframePipeline)         vkDestroyPipeline(d, overlayWireframePipeline, nullptr);
            if (overlayBasicPipeline)             vkDestroyPipeline(d, overlayBasicPipeline, nullptr);
            if (overlayBasicTransparentPipeline)  vkDestroyPipeline(d, overlayBasicTransparentPipeline, nullptr);
            if (overlayLineListPipeline)          vkDestroyPipeline(d, overlayLineListPipeline, nullptr);
            if (overlayLineStripPipeline)         vkDestroyPipeline(d, overlayLineStripPipeline, nullptr);
            if (overlayLineListColoredPipeline)   vkDestroyPipeline(d, overlayLineListColoredPipeline, nullptr);
            if (overlayLineStripColoredPipeline)  vkDestroyPipeline(d, overlayLineStripColoredPipeline, nullptr);
            if (overlayPointListPipeline)         vkDestroyPipeline(d, overlayPointListPipeline, nullptr);
            if (overlayDepthPrepassPipeline)      vkDestroyPipeline(d, overlayDepthPrepassPipeline, nullptr);
            if (overlayPipelineLayout)      vkDestroyPipelineLayout(d, overlayPipelineLayout, nullptr);
            // Masked overlay edge-AA resources.
            if (overlayAaPipeline_)         vkDestroyPipeline(d, overlayAaPipeline_, nullptr);
            if (overlayAaPipelineLayout_)   vkDestroyPipelineLayout(d, overlayAaPipelineLayout_, nullptr);
            if (overlayAaSetLayout_)        vkDestroyDescriptorSetLayout(d, overlayAaSetLayout_, nullptr);
            if (overlayAaPool_)             vkDestroyDescriptorPool(d, overlayAaPool_, nullptr);
            if (overlayAaMask_.image != VK_NULL_HANDLE)
                destroyImage2D(ctx->allocator(), d, overlayAaMask_);
            if (overlayAaScratch_.image != VK_NULL_HANDLE)
                destroyImage2D(ctx->allocator(), d, overlayAaScratch_);
            // Particle billboard pass resources.
            if (particlePipelineNormal_)    vkDestroyPipeline(d, particlePipelineNormal_, nullptr);
            if (particlePipelineAdditive_)  vkDestroyPipeline(d, particlePipelineAdditive_, nullptr);
            if (particlePipelineLayout_)    vkDestroyPipelineLayout(d, particlePipelineLayout_, nullptr);
            if (particleDescSetLayout_)     vkDestroyDescriptorSetLayout(d, particleDescSetLayout_, nullptr);
            for (auto& pool : particleDescPools_) {
                if (pool) vkDestroyDescriptorPool(d, pool, nullptr);
            }
            // Overlay-fog UBO resources (Phase 2b).
            if (overlayFogDescPool_)        vkDestroyDescriptorPool(d, overlayFogDescPool_, nullptr);
            if (overlayFogDescSetLayout_)   vkDestroyDescriptorSetLayout(d, overlayFogDescSetLayout_, nullptr);
            for (auto& b : overlayFogUbos_) destroyBuffer(ctx->allocator(), b);
            // Lit-particle buffers + IO sets (particle_light.comp).
            if (particleIoDescPool_)        vkDestroyDescriptorPool(d, particleIoDescPool_, nullptr);
            for (auto& b : particleCenterBufs_) destroyBuffer(ctx->allocator(), b);
            for (auto& b : particleLightBufs_)  destroyBuffer(ctx->allocator(), b);
            if (spriteWorldPipeline_)       vkDestroyPipeline(d, spriteWorldPipeline_, nullptr);
            destroyBuffer(ctx->allocator(), spriteQuadVtx_);
            destroyBuffer(ctx->allocator(), spriteQuadIdx_);
            if (particleWhiteTex_.view != VK_NULL_HANDLE)
                destroyImage2D(ctx->allocator(), d, particleWhiteTex_);
            for (auto& [t, rec] : particleTexCache_) {
                destroyImage2D(ctx->allocator(), d, rec.image);
            }
            particleTexCache_.clear();
            for (auto& [g, rec] : particleGeomCache_) {
                destroyParticleGeomRec(rec);
            }
            particleGeomCache_.clear();
            // 3D hybrid-overlay line/point geometry cache (GridHelper, AxesHelper,
            // live point clouds). Pre-existing: this Impl-level cache had no
            // teardown, so its vertex/index/color buffers leaked at device
            // destroy (VUID-vkDestroyDevice-device-05137) for any scene with a
            // Line/Points overlay. Mirror OverlayPass's own lineGeomCache_ cleanup.
            for (auto& [g, rec] : lineGeomCache_) {
                destroyBuffer(ctx->allocator(), rec.vertex);
                if (rec.index.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.index);
                if (rec.color.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.color);
            }
            lineGeomCache_.clear();
            overlayPass_.reset();// destroy sprite/line pipelines + caches while device is alive
            if (gbufSampler_)           vkDestroySampler(d, gbufSampler_, nullptr);
            destroyBuffer(ctx->allocator(), dummyUvBuffer_);

#if defined(THREEPP_WITH_FSR)
            // Destroy the ffx-api context while the Vulkan device is still alive
            // (before taa_/ctx unwind). No-op if FSR never created.
            fsr_.reset();
#endif
#if defined(THREEPP_WITH_DLSS)
            // Release the NGX feature + shut NGX down while the device is alive.
            dlss_.reset();
#endif
            // TAA resolve subsystem owns its pipeline/layout/sampler/images.
            taa_.reset();
        }

        void createCommandResources();

        // (Re)create the per-swapchain-image present-wait semaphores. Called
        // from createCommandResources and again on every swapchain recreation
        // (the image count can change and retired semaphores may be left in
        // an indeterminate signalled state). Requires an idle device.
        void createRenderFinishedSemaphores();

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
        // Free `buf` now, or defer it to the next flush when a batch is open.
        void destroyBufferMaybeBatched(Buffer& buf) {
            if (oneShotBatch_) {
                oneShotBatchGarbage_.push_back(buf);
                buf = {};// caller's handle retired; the garbage copy owns it now
            } else {
                destroyBuffer(ctx->allocator(), buf);
            }
        }

        static unsigned int geomVersionOf(const BufferGeometry& g) {
            // Untyped lookups: only `version` is read, and the typed getter
            // returns null for narrowed (compressAttributes) attributes, which
            // would silently drop their edits from the composite version.
            unsigned int v = 0;
            if (auto* a = g.getAttribute("position")) v += a->version;
            if (auto* a = g.getAttribute("normal"))   v += a->version;
            if (auto* idx = g.getIndex())              v += idx->version;
            if (auto* a = g.getAttribute("uv"))        v += a->version;
            if (auto* a = g.getAttribute("color"))     v += a->version;
            return v;
        }

        // Build a single BLAS for the given geometry. Vertex / index buffers
        // are uploaded host-mapped, then the AS is built into freshly allocated
        // device storage. The temporary scratch buffer is destroyed on exit.
        // allowPacked: pack normal/uv/color into narrow device formats
        // (BlasRecord::packedMask). Static geometry only — the deforming
        // callers (skinned / tet / displaced / grass / morphed) keep the
        // default false because their per-frame compute rewrites assume
        // tightly-packed float layouts.
        std::unique_ptr<BlasRecord> buildBlasFor(const BufferGeometry& geom, bool allowPacked = false);

        // Allocate or look up the per-SkinnedMesh BLAS state. Builds the BLAS
        // once with the current pose; subsequent dirty frames go through
        // refreshSkinnedBlas which only re-skins + rebuilds in-place. Returns
        // null if the geometry is unsupported (no position/normal/skin attrs).
        SkinnedMeshState* ensureSkinnedBlas(SkinnedMesh& sm) {
            auto it = skinnedMeshStates.find(&sm);
            if (it != skinnedMeshStates.end()) return it->second.get();

            // Deforming paths (skinned / tet / morph / displaced) stay
            // float-typed by design: the skinning compute rewrites these very
            // buffers every frame, so upload-time widening would be a per-frame
            // tax and the narrow source array would be dead weight. A narrowed
            // normal here means someone ran compressAttributes() on a skinned
            // mesh — warn instead of silently dropping it from the scene.
            auto* posAttr     = sm.geometry()->getAttribute<float>("position");
            auto* nrmAttr     = sm.geometry()->getAttribute<float>("normal");
            auto* skinIdxAttr = sm.geometry()->getAttribute<float>("skinIndex");
            auto* skinWAttr   = sm.geometry()->getAttribute<float>("skinWeight");
            if (!nrmAttr && sm.geometry()->hasAttribute("normal")) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    std::cerr << "[VulkanRenderer] skinned mesh has a non-float 'normal' "
                                 "attribute — deforming geometry must keep float attributes "
                                 "(do not compressAttributes() skinned meshes). Skipping.\n";
                }
            }
            if (!posAttr || !nrmAttr || !skinIdxAttr || !skinWAttr) return nullptr;
            if (!sm.skeleton || sm.skeleton->bones.empty()) return nullptr;

            // Build BLAS with the bind-pose positions/normals first. The
            // BLAS buffers are then re-written each frame by the skinning
            // compute shader (binding 5/6) and rebuilt in-place.
            auto rec = buildBlasFor(*sm.geometry());
            if (!rec) return nullptr;
            rec->liveCheck = sm.geometry();

            // Per-vertex previous-pose buffer. Used for two purposes:
            // (1) Hybrid raster motion-vector source (existing).
            // (2) Per-vertex prev-world-position reproject (2026-05-13):
            //     the deferred shade's ray-query hit handling reads via
            //     gdesc.prevVertexAddress, interpolates, and derives the
            //     hit's prevWorldPos, which feeds the motionMat reproject.
            //     TRANSFER_DST_BIT lets the per-frame
            //     vkCmdCopyBuffer push current→prev before the skinning
            //     compute writes new positions. Device-local (no host
            //     access) — the per-frame update happens entirely on GPU.
            const VkDeviceSize vbBytes = posAttr->array().size() * sizeof(float);
            rec->prevVertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            auto state = std::make_unique<SkinnedMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = sm.geometry();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            const uint32_t boneCount   = static_cast<uint32_t>(sm.skeleton->bones.size());
            state->vertexCount = vertexCount;
            state->boneCount   = boneCount;
            state->prevBoneMats.assign(boneCount * 16, 0.f);

            auto* idxAttr = sm.geometry()->getIndex();
            state->indexed = idxAttr != nullptr;
            state->primitiveCount = state->indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;

            // ── GPU-skinning input buffers. Uploaded once, reused every frame.
            auto allocAndUpload = [&](Buffer& dst, const void* src,
                                      VkDeviceSize bytes) {
                dst = createBuffer(
                        ctx->allocator(), ctx->device(), bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), dst, src, bytes);
            };
            allocAndUpload(state->baseVertex, posAttr->array().data(),
                           vertexCount * 3 * sizeof(float));
            allocAndUpload(state->baseNormal, nrmAttr->array().data(),
                           vertexCount * 3 * sizeof(float));
            allocAndUpload(state->skinIndex, skinIdxAttr->array().data(),
                           vertexCount * 4 * sizeof(float));
            allocAndUpload(state->skinWeight, skinWAttr->array().data(),
                           vertexCount * 4 * sizeof(float));

            // Bone matrices buffer: [bindMatrix, bindMatrixInverse, bones...].
            // bindMatrix is constant. bindMatrixInverse is NOT (attached bind mode
            // ties it to the current matrixWorld) — refreshSkinnedBlas re-uploads
            // it every frame; seed both here.
            const VkDeviceSize matsBytes = (2 + boneCount) * 16 * sizeof(float);
            state->boneMatrices = createBuffer(
                    ctx->allocator(), ctx->device(), matsBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), state->boneMatrices.alloc, &mapped);
                std::memcpy(static_cast<char*>(mapped),
                            sm.bindMatrix.elements.data(),
                            16 * sizeof(float));
                std::memcpy(static_cast<char*>(mapped) + 16 * sizeof(float),
                            sm.bindMatrixInverse.elements.data(),
                            16 * sizeof(float));
                std::memset(static_cast<char*>(mapped) + 32 * sizeof(float),
                            0, boneCount * 16 * sizeof(float));
                flushHostWrites(ctx->allocator(), state->boneMatrices.alloc);
                vmaUnmapMemory(ctx->allocator(), state->boneMatrices.alloc);
            }

            // Descriptor set — wires base inputs + bone mats + BLAS outputs.
            state->skinDescSet = skinning_->allocateMeshDescriptorSet();

            std::array<VkDescriptorBufferInfo, 7> bi{};
            const Buffer* bufs[7] = {
                    &state->baseVertex, &state->baseNormal,
                    &state->skinIndex,  &state->skinWeight,
                    &state->boneMatrices,
                    &state->blas->vertex, &state->blas->normal,
            };
            std::array<VkWriteDescriptorSet, 7> wr{};
            for (uint32_t i = 0; i < 7; ++i) {
                bi[i].buffer        = bufs[i]->handle;
                bi[i].offset        = 0;
                bi[i].range         = VK_WHOLE_SIZE;
                wr[i]               = {};
                wr[i].sType         = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[i].dstSet        = state->skinDescSet;
                wr[i].dstBinding    = i;
                wr[i].descriptorCount = 1;
                wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wr[i].pBufferInfo     = &bi[i];
            }
            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(wr.size()),
                                   wr.data(), 0, nullptr);

            // BLAS rebuild scratch buffer (persistent — sized once, reused).
            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = state->blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex    = vertexCount - 1;
            if (state->indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = state->blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }
            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType  = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags         = 0;
            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries   = &blasGeom;
            blasBuild.dstAccelerationStructure = state->blas->as;
            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &state->primitiveCount, &blasSizes);
            state->blasScratchSize = blasSizes.buildScratchSize;
            state->blasScratch = createAsScratchBuffer(
                    ctx->allocator(), ctx->device(), state->blasScratchSize);

            auto* raw = state.get();
            skinnedMeshStates.emplace(&sm, std::move(state));
            // First-frame refresh: upload bones + queue a rebuild so the BLAS
            // reflects the current pose, not bind pose, when the next frame
            // records.
            refreshSkinnedBlas(sm, *raw);
            return raw;
        }

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
        TetMeshState* ensureTetBlas(Mesh& m) {
            auto found = tetMeshStates.find(&m);
            if (found != tetMeshStates.end()) return found->second.get();

            auto geom = m.geometry();
            if (!geom) return nullptr;
            auto* posAttr = geom->getAttribute<float>("position");
            auto* nrmAttr = geom->getAttribute<float>("normal");
            auto* tiAttr  = geom->getAttribute<float>("tetIndex");
            auto* twAttr  = geom->getAttribute<float>("tetWeight");
            auto* r0Attr  = geom->getAttribute<float>("tetRestInv0");
            auto* r1Attr  = geom->getAttribute<float>("tetRestInv1");
            auto* r2Attr  = geom->getAttribute<float>("tetRestInv2");
            auto mat = m.material();
            if (!posAttr || !nrmAttr || !tiAttr || !twAttr || !r0Attr || !r1Attr || !r2Attr) return nullptr;
            if (!mat || !mat->tetTexture) return nullptr;

            // BLAS built from the rest positions; the tet_skinning compute then
            // rewrites the vertex/normal buffers each frame and the BLAS is refit.
            auto rec = buildBlasFor(*geom);
            if (!rec) return nullptr;
            rec->liveCheck = geom;

            // Previous-frame vertex buffer for per-vertex motion vectors (same role
            // as the skinned path; copied current->prev before each compute).
            const VkDeviceSize vbBytes = posAttr->array().size() * sizeof(float);
            rec->prevVertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            auto state = std::make_unique<TetMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = geom;
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            state->vertexCount = vertexCount;
            auto* idxAttr = geom->getIndex();
            state->indexed = idxAttr != nullptr;
            state->primitiveCount = state->indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;

            auto allocAndUpload = [&](Buffer& dst, const void* src, VkDeviceSize bytes) {
                dst = createBuffer(
                        ctx->allocator(), ctx->device(), bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), dst, src, bytes);
            };
            allocAndUpload(state->tetIndex,   tiAttr->array().data(),  vertexCount * 4 * sizeof(float));
            allocAndUpload(state->tetWeight,  twAttr->array().data(),  vertexCount * 4 * sizeof(float));
            allocAndUpload(state->baseNormal, nrmAttr->array().data(), vertexCount * 3 * sizeof(float));
            allocAndUpload(state->restInv0,   r0Attr->array().data(),  vertexCount * 3 * sizeof(float));
            allocAndUpload(state->restInv1,   r1Attr->array().data(),  vertexCount * 3 * sizeof(float));
            allocAndUpload(state->restInv2,   r2Attr->array().data(),  vertexCount * 3 * sizeof(float));

            // Per-frame tet positions buffer, sized to the tet texture image (one
            // RGBA32F texel per collision-tet vertex). Filled by refreshTetBlas.
            const auto& tetImg = mat->tetTexture->image().data<float>();
            state->tetPosBytes = static_cast<VkDeviceSize>(tetImg.size()) * sizeof(float);
            state->tetPos = createBuffer(
                    ctx->allocator(), ctx->device(), state->tetPosBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

            // Descriptor set — 9 storage buffers (see tet_skinning.comp).
            state->tetDescSet = tetSkinning_->allocateMeshDescriptorSet();
            std::array<VkDescriptorBufferInfo, 9> bi{};
            const Buffer* bufs[9] = {
                    &state->tetIndex, &state->tetWeight, &state->baseNormal,
                    &state->restInv0, &state->restInv1, &state->restInv2,
                    &state->tetPos,
                    &state->blas->vertex, &state->blas->normal,
            };
            std::array<VkWriteDescriptorSet, 9> wr{};
            for (uint32_t i = 0; i < 9; ++i) {
                bi[i].buffer          = bufs[i]->handle;
                bi[i].offset          = 0;
                bi[i].range           = VK_WHOLE_SIZE;
                wr[i]                 = {};
                wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[i].dstSet          = state->tetDescSet;
                wr[i].dstBinding      = i;
                wr[i].descriptorCount = 1;
                wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wr[i].pBufferInfo     = &bi[i];
            }
            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(wr.size()), wr.data(), 0, nullptr);

            // Persistent BLAS-rebuild scratch (sized once, reused every frame).
            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = state->blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex    = vertexCount - 1;
            if (state->indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = state->blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }
            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType  = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags         = 0;
            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries   = &blasGeom;
            blasBuild.dstAccelerationStructure = state->blas->as;
            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &state->primitiveCount, &blasSizes);
            state->blasScratchSize = blasSizes.buildScratchSize;
            state->blasScratch = createAsScratchBuffer(
                    ctx->allocator(), ctx->device(), state->blasScratchSize);

            auto* raw = state.get();
            tetMeshStates.emplace(&m, std::move(state));
            // First-frame refresh so the BLAS reflects the current deformation.
            refreshTetBlas(m, *raw);
            return raw;
        }

        // Per-frame: upload the soft body's current collision-tet positions into the
        // tet-position buffer (read from the material's tet texture image, which
        // PhysxWorld::syncSoftBodies refreshes each frame), then queue the GPU skin +
        // BLAS rebuild — recorded in recordCommandBuffer next to the skinned path.
        void refreshTetBlas(Mesh& m, TetMeshState& st);

        // Rewrite binding 6 (tetPos — see tet_skinning.comp) of a tet mesh's
        // descriptor set to point at `buf`. Used by the interop enable/disable
        // swap; the other 8 bindings are untouched.
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
            if (st.tetPosExt.handle != VK_NULL_HANDLE) {// already enabled — same handle
                st.tetPosExternalCopy = std::move(deviceCopy);
                return {st.tetPosExt.osHandle, static_cast<size_t>(st.tetPosExt.size)};
            }
            if (st.tetPosBytes == 0 || st.tetDescSet == VK_NULL_HANDLE) return {};
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (softbody interop enable)");
            st.tetPosExt = vulkan::createExternalBuffer(
                    ctx->physicalDevice(), ctx->device(), st.tetPosBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            rewriteTetPosBinding(st, st.tetPosExt.handle);
            destroyBuffer(ctx->allocator(), st.tetPos);// CPU-path buffer no longer read
            st.tetPosExternalCopy = std::move(deviceCopy);
            return {st.tetPosExt.osHandle, static_cast<size_t>(st.tetPosExt.size)};
        }

        // Interop disable / CUDA-import-failure fallback: restore the host-visible
        // VMA buffer and the CPU upload path. The caller must have stopped (or
        // never started) the CUDA writes into the exported memory.
        void disableSoftBodyInterop(const Mesh& mesh);

        // Per-frame refresh op for a plain (non-skinned/non-displaced/non-morphed)
        // dynamic geometry whose BufferAttribute versions just bumped. Batched
        // through refreshGeomBlasBatch so N soft bodies cost 2 GPU submits
        // total, not 2N.
        struct GeomRefreshOp {
            const BufferGeometry* geom;
            BlasRecord*           rec;
        };

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

        // ── Morph-target helpers ─────────────────────────────────────────

        static bool isMorphedMesh(const Mesh& m) {
            return m.geometry()->getMorphAttributes().count("position") > 0;
        }

        static void cpuMorphBlend(Mesh& mesh,
                                  std::vector<float>& outPos,
                                  std::vector<float>& outNorm) {
            const auto& geom = *mesh.geometry();
            auto* posAttr = geom.getAttribute<float>("position");
            auto* nrmAttr = geom.getAttribute<float>("normal");
            if (!posAttr) return;

            const int vtxCount = posAttr->count();
            const auto& basePos = posAttr->array();
            outPos.assign(basePos.begin(), basePos.end());

            if (nrmAttr) {
                const auto& baseNrm = nrmAttr->array();
                outNorm.assign(baseNrm.begin(), baseNrm.end());
            } else {
                outNorm.assign(vtxCount * 3, 0.f);
            }

            const auto& morphAttrsMap = geom.getMorphAttributes();
            auto posIt = morphAttrsMap.find("position");
            if (posIt == morphAttrsMap.end()) return;
            const auto& morphPos = posIt->second;

            const std::vector<std::shared_ptr<BufferAttribute>>* morphNrm = nullptr;
            auto nrmIt = morphAttrsMap.find("normal");
            if (nrmIt != morphAttrsMap.end()) morphNrm = &nrmIt->second;

            auto* morphObj = mesh.as<ObjectWithMorphTargetInfluences>();
            if (!morphObj) return;
            const auto& influences = morphObj->morphTargetInfluences();

            const bool relative = geom.morphTargetsRelative;
            const size_t numTargets = morphPos.size();

            for (size_t t = 0; t < numTargets && t < influences.size(); ++t) {
                const float w = influences[t];
                if (w == 0.f) continue;

                auto* tAttr = dynamic_cast<TypedBufferAttribute<float>*>(morphPos[t].get());
                if (!tAttr || tAttr->count() != vtxCount) continue;
                const auto& tData = tAttr->array();

                if (relative) {
                    for (int v = 0; v < vtxCount; ++v) {
                        outPos[v * 3 + 0] += w * tData[v * 3 + 0];
                        outPos[v * 3 + 1] += w * tData[v * 3 + 1];
                        outPos[v * 3 + 2] += w * tData[v * 3 + 2];
                    }
                } else {
                    for (int v = 0; v < vtxCount; ++v) {
                        outPos[v * 3 + 0] += w * (tData[v * 3 + 0] - basePos[v * 3 + 0]);
                        outPos[v * 3 + 1] += w * (tData[v * 3 + 1] - basePos[v * 3 + 1]);
                        outPos[v * 3 + 2] += w * (tData[v * 3 + 2] - basePos[v * 3 + 2]);
                    }
                }

                if (morphNrm && t < morphNrm->size()) {
                    auto* nAttr = dynamic_cast<TypedBufferAttribute<float>*>((*morphNrm)[t].get());
                    if (nAttr && nAttr->count() == vtxCount) {
                        const auto& nData = nAttr->array();
                        if (relative) {
                            for (int v = 0; v < vtxCount; ++v) {
                                outNorm[v * 3 + 0] += w * nData[v * 3 + 0];
                                outNorm[v * 3 + 1] += w * nData[v * 3 + 1];
                                outNorm[v * 3 + 2] += w * nData[v * 3 + 2];
                            }
                        } else if (nrmAttr) {
                            const auto& baseNrm = nrmAttr->array();
                            for (int v = 0; v < vtxCount; ++v) {
                                outNorm[v * 3 + 0] += w * (nData[v * 3 + 0] - baseNrm[v * 3 + 0]);
                                outNorm[v * 3 + 1] += w * (nData[v * 3 + 1] - baseNrm[v * 3 + 1]);
                                outNorm[v * 3 + 2] += w * (nData[v * 3 + 2] - baseNrm[v * 3 + 2]);
                            }
                        }
                    }
                }
            }

            // Renormalize normals.
            for (int v = 0; v < vtxCount; ++v) {
                float& nx = outNorm[v * 3 + 0];
                float& ny = outNorm[v * 3 + 1];
                float& nz = outNorm[v * 3 + 2];
                const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 0.f) { const float inv = 1.f / len; nx *= inv; ny *= inv; nz *= inv; }
            }
        }

        MorphedMeshState* ensureMorphedBlas(Mesh& mesh) {
            auto it = morphedMeshStates.find(&mesh);
            if (it != morphedMeshStates.end()) return it->second.get();

            auto rec = buildBlasFor(*mesh.geometry());
            if (!rec) return nullptr;
            rec->liveCheck = mesh.geometry();

            auto state = std::make_unique<MorphedMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = mesh.geometry();

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
        DisplacedMeshState* ensureDisplacedState(DisplacedMesh& dm) {
            auto it = displacedStates.find(&dm);
            if (it != displacedStates.end()) return it->second.get();

            auto* posAttr = dm.geometry()->getAttribute<float>("position");
            if (!posAttr) return nullptr;
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            // Plane is gridDim × gridDim. PlaneGeometry(w, h, segX, segY)
            // produces (segX+1)·(segY+1) verts; the demo is expected to call
            // segX == segY == gridDim-1.
            const uint32_t gridDim = static_cast<uint32_t>(std::round(std::sqrt(double(vertexCount))));
            if (gridDim * gridDim != vertexCount) return nullptr;

            // Plane edge length: derive from rest-position bbox extent in X.
            float xMin = std::numeric_limits<float>::infinity();
            float xMax = -std::numeric_limits<float>::infinity();
            for (uint32_t i = 0; i < vertexCount; ++i) {
                const float x = posAttr->getX(i);
                if (x < xMin) xMin = x;
                if (x > xMax) xMax = x;
            }
            const float planeSize = xMax - xMin;
            if (!(planeSize > 0.f)) return nullptr;

            auto blas = buildBlasFor(*dm.geometry());
            if (!blas) return nullptr;
            blas->liveCheck = dm.geometry();

            // FFT-displaced ocean mesh: mark it so the chit / deferred passes
            // apply world-space foam + thin-shell water shading. Foam itself
            // lives in a world-space texture built by foam_world.comp, so the
            // only per-mesh state needed here is this "is water" marker (it
            // used to be a zero-filled per-vertex foam buffer, read solely for
            // its non-null address — the contents were dead).
            blas->isOceanSurface = true;

            // Per-vertex previous-pose buffer for hybrid raster motion vec.
            // Same size as vertex (R32G32B32 SFLOAT × vertexCount). Filled
            // GPU-side via vkCmdCopyBuffer before each water_displace dispatch.
            blas->prevVertex = createBuffer(
                    ctx->allocator(), ctx->device(),
                    VkDeviceSize(vertexCount) * 3u * sizeof(float),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            auto state = std::make_unique<DisplacedMeshState>();
            state->blas = std::move(blas);
            state->vertexCount = vertexCount;
            state->gridDim = gridDim;
            state->planeSize = planeSize;
            state->liveCheck = dm.geometry();

            // FFT cascades — one Phillips/Dynamic/IFFT chain per non-zero
            // `tileSize` in DisplacedMesh::Params. Cascades are band-passed
            // by k so each covers a disjoint wavenumber range:
            //   cascade 0 (largest tile): 0 → kNyq of cascade 1
            //   cascade 1 (middle):       kNyq of cascade 1 → kNyq of cascade 2
            //   cascade 2 (smallest):     kNyq of cascade 2 → ∞
            // where kNyq_i = π·N / tileSize_i. Cascade 0 is required;
            // tileSize1/tileSize2 == 0 disable the corresponding band.
            const float tileSizes[3] = {
                    dm.params.tileSize0,
                    dm.params.tileSize1,
                    dm.params.tileSize2,
            };
            const uint32_t textureSizes[3] = {
                    dm.params.textureSize0,
                    dm.params.textureSize1,
                    dm.params.textureSize2,
            };
            // Hand-off k between adjacent cascades = the SMALLER tile's lowest
            // natural k = 2π / tileSize_(smaller). Each cascade then covers
            // wavelengths between its own tile and the next-smaller tile:
            //   cascade 0: λ ∈ [tileSize_1, tileSize_0]
            //   cascade 1: λ ∈ [tileSize_2, tileSize_1]
            //   cascade 2: λ ∈ [<tileSize_2]  (down to its own Nyquist)
            //
            // Why not split at the larger tile's Nyquist (k = π·N / L_larger)?
            // Because that boundary is at the mesh's resolving limit — it
            // hands all the mesh-resolvable wavelengths to cascade 0 alone,
            // and cascades 1 and 2 emit only sub-mesh wavelengths that
            // alias as displacement noise. The 2π/L_smaller scheme reserves
            // a real, mesh-displayable band for each intermediate cascade.
            constexpr float kTwoPi = 6.28318530717958647692f;
            const float kHandoff01 = (tileSizes[0] > 0.f && tileSizes[1] > 0.f)
                    ? kTwoPi / tileSizes[1] : 0.f;
            const float kHandoff12 = (tileSizes[1] > 0.f && tileSizes[2] > 0.f)
                    ? kTwoPi / tileSizes[2] : 0.f;
            for (uint32_t i = 0; i < 3; ++i) {
                if (!(tileSizes[i] > 0.f)) continue;
                const uint32_t texSize = textureSizes[i];
                water::PhillipsSpectrum::Settings ps{};
                ps.textureSize = texSize;
                ps.tileSize    = tileSizes[i];
                ps.windTheta   = dm.params.windTheta;
                ps.windSpeed   = dm.params.windSpeed;
                // Suppress wavelengths shorter than ~5× the cascade's sample
                // spacing. Without this, single-cascade Phillips puts energy
                // into bands the FFT can't resolve, producing spike-crest
                // aliasing. Per-cascade `texSize` so each cascade's cutoff
                // tracks its own resolution.
                ps.smallWaveCutoff = 5.f * tileSizes[i] / float(texSize);
                if (i == 0) {
                    ps.kMin = 0.f;
                    ps.kMax = kHandoff01; // 0 if no cascade 1 → no upper bound
                } else if (i == 1) {
                    ps.kMin = kHandoff01;
                    ps.kMax = kHandoff12; // 0 if no cascade 2 → no upper bound
                } else {
                    ps.kMin = kHandoff12;
                    ps.kMax = 0.f;
                }
                auto& c = state->cascades[i];
                c.tileSize = tileSizes[i];
                c.phillips = std::make_unique<water::PhillipsSpectrum>(*ctx, ps);
                c.dyn      = std::make_unique<water::DynamicSpectrum>(
                        *ctx, *c.phillips, texSize, tileSizes[i]);
                c.ifft     = std::make_unique<water::IFFT>(*ctx, texSize);
                state->cascadeMask |= (1u << i);
            }
            if (state->cascadeMask == 0u) return nullptr; // no cascades → invalid setup

            // Per-cascade height readback buffers (host-mapped). Each cascade
            // can run at a different FFT resolution, so each readback is
            // sized to its own dim²·8 bytes (RG32F). Kept persistent so the
            // mappings survive between frames.
            auto makeReadback = [&](uint32_t dim) {
                const VkDeviceSize bytes =
                        VkDeviceSize(dim) * VkDeviceSize(dim) * 8u;
                return createBuffer(
                        ctx->allocator(), ctx->device(), bytes,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
            };
            state->heightReadback  = makeReadback(textureSizes[0]);
            state->heightReadbackDim[0] = textureSizes[0];
            if (dm.params.tileSize1 > 0.f) {
                state->heightReadback1 = makeReadback(textureSizes[1]);
                state->heightReadbackDim[1] = textureSizes[1];
            }
            if (dm.params.tileSize2 > 0.f) {
                state->heightReadback2 = makeReadback(textureSizes[2]);
                state->heightReadbackDim[2] = textureSizes[2];
            }

            // Scratch image for IFFT ping-pong (RG32F). Cascades dispatch
            // back-to-back on the same queue and share this scratch, so size
            // it to the largest enabled cascade. Smaller cascades' IFFT runs
            // only touch their own extent within the scratch — the unused
            // tail is harmless.
            uint32_t scratchDim = 0;
            for (uint32_t i = 0; i < 3; ++i) {
                if (tileSizes[i] > 0.f) scratchDim = std::max(scratchDim, textureSizes[i]);
            }
            {
                VkImageCreateInfo ici{};
                ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType     = VK_IMAGE_TYPE_2D;
                ici.format        = VK_FORMAT_R32G32_SFLOAT;
                ici.extent        = {scratchDim, scratchDim, 1};
                ici.mipLevels     = 1;
                ici.arrayLayers   = 1;
                ici.samples       = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
                ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO;
                check(vmaCreateImage(ctx->allocator(), &ici, &aci,
                                     &state->scratchA.image, &state->scratchA.alloc, nullptr),
                      "vmaCreateImage(displaceScratch)");
                state->scratchA.format = VK_FORMAT_R32G32_SFLOAT;
                state->scratchA.width  = scratchDim;
                state->scratchA.height = scratchDim;
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = state->scratchA.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = VK_FORMAT_R32G32_SFLOAT;
                vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &state->scratchA.view),
                      "vkCreateImageView(displaceScratch)");
                ctx->setObjectName(state->scratchA.image, "ocean.scratchA (IFFT ping-pong)");
                ctx->setObjectName(state->scratchA.view,  "ocean.scratchA (IFFT ping-pong)");
            }

            // World-space foam image. Coverage equals the cascade-0 tile
            // (matches the FFT periodicity, so REPEAT-sampling at any
            // world XZ folds back into the same texture cell). Resolution
            // 2048² over 1000 m → ~0.49 m per texel — fine enough to carry
            // the cascade-1 whitecap detail foam_world.comp's fine Jacobian
            // stencil extracts (16 MB R32F). R32F storage so both compute
            // imageLoad/Store and chit linear sampling work without
            // format conversions.
            {
                state->foamRes      = 2048u;
                state->foamTileSize = dm.params.tileSize0;
                VkImageCreateInfo ici{};
                ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType     = VK_IMAGE_TYPE_2D;
                ici.format        = VK_FORMAT_R32_SFLOAT;
                ici.extent        = {state->foamRes, state->foamRes, 1};
                ici.mipLevels     = 1;
                ici.arrayLayers   = 1;
                ici.samples       = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
                ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO;
                check(vmaCreateImage(ctx->allocator(), &ici, &aci,
                                     &state->foamImage.image, &state->foamImage.alloc, nullptr),
                      "vmaCreateImage(foamWorld)");
                state->foamImage.format = VK_FORMAT_R32_SFLOAT;
                state->foamImage.width  = state->foamRes;
                state->foamImage.height = state->foamRes;
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = state->foamImage.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = VK_FORMAT_R32_SFLOAT;
                vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &state->foamImage.view),
                      "vkCreateImageView(foamWorld)");
                ctx->setObjectName(state->foamImage.image, "ocean.foamWorld (2048x2048 R32F)");
                ctx->setObjectName(state->foamImage.view,  "ocean.foamWorld (2048x2048 R32F)");

                // Initial clear to zero + layout transition to GENERAL so the
                // first foam_world dispatch's imageLoad reads 0 (no foam yet)
                // and the chit's linear sampler reads from a defined image.
                VkCommandBuffer cb = beginOneShot();
                {
                    VkImageMemoryBarrier imb{};
                    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    imb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    imb.srcAccessMask = 0;
                    imb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    imb.image = state->foamImage.image;
                    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &imb);
                }
                VkClearColorValue cc{};
                cc.float32[0] = 0.0f;
                VkImageSubresourceRange sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cb, state->foamImage.image,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     &cc, 1, &sub);
                {
                    VkImageMemoryBarrier imb{};
                    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    imb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    imb.image = state->foamImage.image;
                    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                            0, 0, nullptr, 0, nullptr, 1, &imb);
                }
                endAndSubmitOneShot(cb);
                state->foamImage.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            }

            // Allocate this mesh's displace descriptor set + write bindings.
            state->displaceDS = waterDisplace_->allocateMeshDescriptorSet();

            // Bind each enabled cascade's spatial images to its (height, displace)
            // slot pair. Disabled cascades are filled with cascade 0's images
            // so the shader's combined-image-sampler bindings are always valid;
            // the shader gates which slots are actually sampled via cascadeMask.
            std::array<VkDescriptorImageInfo, 6> imageInfos{};
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t srcCascade = (state->cascadeMask & (1u << i)) ? i : 0u;
                const auto& c = state->cascades[srcCascade];
                imageInfos[i * 2 + 0].sampler     = waterDisplace_->sampler();
                imageInfos[i * 2 + 0].imageView   = c.dyn->ht().view;
                imageInfos[i * 2 + 0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                imageInfos[i * 2 + 1].sampler     = waterDisplace_->sampler();
                imageInfos[i * 2 + 1].imageView   = c.dyn->displacement().view;
                imageInfos[i * 2 + 1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            std::array<VkWriteDescriptorSet, 6> ws{};
            for (uint32_t i = 0; i < 6; ++i) {
                ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[i].dstSet = state->displaceDS;
                ws[i].dstBinding = i;
                ws[i].descriptorCount = 1;
                ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                ws[i].pImageInfo = &imageInfos[i];
            }
            vkUpdateDescriptorSets(ctx->device(), uint32_t(ws.size()), ws.data(), 0, nullptr);

            // Foam-world descriptor set — same cascade bindings 0..5 as the
            // displace set, plus binding 6 = the storage image foam target.
            state->foamWorldDS = foamWorld_->allocateMeshDescriptorSet();
            std::array<VkDescriptorImageInfo, 6> foamCascadeInfos{};
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t srcCascade = (state->cascadeMask & (1u << i)) ? i : 0u;
                const auto& c = state->cascades[srcCascade];
                foamCascadeInfos[i * 2 + 0].sampler     = foamWorld_->sampler();
                foamCascadeInfos[i * 2 + 0].imageView   = c.dyn->ht().view;
                foamCascadeInfos[i * 2 + 0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                foamCascadeInfos[i * 2 + 1].sampler     = foamWorld_->sampler();
                foamCascadeInfos[i * 2 + 1].imageView   = c.dyn->displacement().view;
                foamCascadeInfos[i * 2 + 1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            VkDescriptorImageInfo foamStorageInfo{};
            foamStorageInfo.sampler     = VK_NULL_HANDLE;
            foamStorageInfo.imageView   = state->foamImage.view;
            foamStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            std::array<VkWriteDescriptorSet, 7> fws{};
            for (uint32_t i = 0; i < 6; ++i) {
                fws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                fws[i].dstSet = state->foamWorldDS;
                fws[i].dstBinding = i;
                fws[i].descriptorCount = 1;
                fws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                fws[i].pImageInfo = &foamCascadeInfos[i];
            }
            fws[6].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            fws[6].dstSet         = state->foamWorldDS;
            fws[6].dstBinding     = 6;
            fws[6].descriptorCount= 1;
            fws[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            fws[6].pImageInfo     = &foamStorageInfo;
            vkUpdateDescriptorSets(ctx->device(), uint32_t(fws.size()), fws.data(), 0, nullptr);

            // Hand the smallest enabled cascade's height image to closest_hit
            // (binding 32) for sub-mesh-resolution normal perturbation. Picks
            // the highest enabled bit — the cascade with the smallest tileSize
            // and therefore the finest spatial resolution. The cascade VkImage
            // handle is stable until the DisplacedMesh is destroyed, so we
            // only rewrite the descriptor on first init (not per frame).
            {
                uint32_t fineIdx = 0;
                float fineTile   = 0.f;
                for (uint32_t i = 0; i < 3; ++i) {
                    if (state->cascadeMask & (1u << i)) {
                        fineIdx  = i;
                        fineTile = state->cascades[i].tileSize;
                    }
                }
                if (fineTile > 0.f) {
                    oceanFineHeightView = state->cascades[fineIdx].dyn->ht().view;
                    oceanFineTileSize   = fineTile;
                    rewriteDeferredDescriptors();
                }
            }

            // Hand this mesh's world-foam view to closest_hit (binding 33) so
            // the chit can sample foam at arbitrary world XZ during shading.
            // Like oceanFineHeight above, the foam VkImage handle is stable
            // until the DisplacedMesh is destroyed, so we only rewrite once.
            oceanFoamView     = state->foamImage.view;
            oceanFoamTileSize = state->foamTileSize;
            rewriteDeferredDescriptors();

            auto* raw = state.get();
            displacedStates.emplace(&dm, std::move(state));
            return raw;
        }

        // Per-frame: run the FFT chain → water_displace → BLAS rebuild for
        // one DisplacedMesh. Mirrors refreshSkinnedBlas's structure.
        // Record the full per-frame water update — FFT chain → water_displace →
        // world foam → in-place BLAS rebuild — into `cb`. NO submit and NO CPU
        // wait: the caller either batches this into the frame command buffer
        // (recordCommandBuffer draining pendingDisplacedDeforms_, the per-frame
        // path) or wraps it in a one-shot for the rare structural first build
        // (refreshDisplacedBlas below).
        void recordDisplacedDeform(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds);

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
        // reason grass can be culled at all. Returns false (→ caller treats the
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
            out.expandByScalar(gm->params.windStrength);
            return true;
        }

        // Distance-gated wind/refit freeze decision (see GrassMesh::Params::
        // maxAnimDistance). 0 ⇒ always animate. Otherwise animate only while the
        // camera is within maxAnimDistance of the field's sway-dilated AABB;
        // beyond that the field is frozen (skips the per-frame dispatch + refit,
        // holds its last pose). Distance is measured to the nearest point of the
        // box so a large field animates whenever the camera is anywhere near it.
        bool grassShouldAnimate(const MeshEntry& en, const Vector3& camPos) const {
            auto* gm = static_cast<GrassMesh*>(en.mesh);
            const float maxD = gm->params.maxAnimDistance;
            if (maxD <= 0.f) return true;
            Box3 box;
            if (!grassSwayWorldAabb(en, box)) return true;// no bounds → animate
            return box.distanceToPoint(camPos) <= maxD;
        }

        // Lazy create the per-GrassMesh state: a BLAS over the rest geometry
        // (built ALLOW_UPDATE, refit each frame) plus an immutable copy of the
        // rest positions and the per-vertex height fractions the wind shader
        // reads. Returns nullptr if the geometry lacks position / heightFrac.
        GrassMeshState* ensureGrassState(GrassMesh& gm) {
            auto it = grassStates.find(&gm);
            if (it != grassStates.end()) return it->second.get();

            auto* posAttr = gm.geometry()->getAttribute<float>("position");
            auto* hfAttr  = gm.geometry()->getAttribute<float>("heightFrac");
            if (!posAttr || !hfAttr) return nullptr;
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            if (vertexCount == 0 || static_cast<uint32_t>(hfAttr->count()) != vertexCount) return nullptr;

            auto blas = buildBlasFor(*gm.geometry());
            if (!blas) return nullptr;
            blas->liveCheck = gm.geometry();

            auto state = std::make_unique<GrassMeshState>();
            state->vertexCount = vertexCount;
            state->liveCheck = gm.geometry();

            // Immutable rest positions — the shader reads these and writes the
            // displaced result into the (separate) BLAS vertex buffer.
            const VkDeviceSize posBytes = VkDeviceSize(vertexCount) * 3u * sizeof(float);
            state->restPos = createBuffer(
                    ctx->allocator(), ctx->device(), posBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), state->restPos, posAttr->array().data(), posBytes);

            const VkDeviceSize hfBytes = VkDeviceSize(vertexCount) * sizeof(float);
            state->heightFrac = createBuffer(
                    ctx->allocator(), ctx->device(), hfBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), state->heightFrac, hfAttr->array().data(), hfBytes);

            state->blas = std::move(blas);
            auto* raw = state.get();
            grassStates.emplace(&gm, std::move(state));
            return raw;
        }

        // Record the grass wind deform (rest → displaced positions in the BLAS
        // vertex buffer) + in-place BLAS refit into `cb`. No submit — the caller
        // batches these into the main frame command buffer (recordCommandBuffer),
        // exactly like the skinned-mesh path, so there is NO mid-frame
        // vkQueueWaitIdle. Uses the persistent per-BLAS scratch (grass geometry
        // is fixed-size, so it's allocated once on first use). Normals are left
        // static, so only the position buffer is written/barriered.
        void recordGrassDeform(VkCommandBuffer cb, GrassMesh& gm, GrassMeshState& st);

        // One-shot wrapper — used only for the initial prime during scene build
        // (rare). The per-frame path records into the frame cb via the pending
        // queue, so it never drains.
        void refreshGrassBlas(GrassMesh& gm, GrassMeshState& st, float /*elapsedSeconds*/);

        // Build a TLAS over the supplied instance descriptors. Empty input is
        // legal — produces an empty TLAS that always misses.
        void buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances);

        // Record a per-frame TLAS refit/rebuild into `cb` — NO blocking submit.
        // The host instance write goes to tlasInstancesBuffers[currentFrame]
        // (safe: recordCommandBuffer runs after this slot's fence wait), and the
        // build uses the persistent scratch. Leaves the TLAS handle unchanged so
        // descriptor binding 0 keeps pointing at it. Must be recorded AFTER all
        // deformable BLAS rebuilds in the same cb.
        void recordTlasRefit(VkCommandBuffer cb,
                             const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                             bool fullBuild);

        template<typename DescT>
        void uploadDescBuffer(Buffer& target, const std::vector<DescT>& descs) {
            const VkDeviceSize bytes = std::max<VkDeviceSize>(
                    descs.size() * sizeof(DescT), sizeof(DescT));
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
        static MaterialDesc materialFromMesh(const Mesh& m) {
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
        void computeAndUploadMotionMatrices(uint32_t frame,
                                            const std::vector<MeshEntry>& entries);

        // Upload meshMovedBits_ to meshMovedBitsBuffers[frame]. Caller must have
        // already waited the inFlight[frame] fence.
        void uploadMeshMovedBits(uint32_t frame);

        // Walk visible entries, gather emissive triangles in world space, and
        // upload to emissiveTriBuffers[frame]. Per-tri 64-byte record:
        //   v0.xyz = world pos0,    v0.w = triangle area
        //   v1.xyz = world pos1,    v1.w = running cumPower (CDF)
        //   v2.xyz = world pos2,    v2.w = per-tri power (lum * area)
        //   emission.xyz = emissive*intensity, emission.w = unused
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

        // Cache per-entry cull mode from a freshly-built matDescs array.
        // Called wherever matDescs is uploaded so the gbuffer draw loop can
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
        void cacheCullFlags(const std::vector<MaterialDesc>& mds);

        // Resolve the BlasRecord backing a given visible entry. The same
        // physical buffers feed BLAS and the raster prepass (VERTEX_BUFFER_BIT
        // was added at allocation), so this is a pure lookup, no upload.
        // Branches off the cached type flags so we don't dynamic_cast every
        // entry on every raster draw call.
        const BlasRecord* resolveBlasForEntry(const MeshEntry& en) const {
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

        // Host mirror of gbuffer_indirect.vert's DrawInfo struct. Tight-
        // packed (120 bytes, all members naturally aligned to ≤ 8) so it
        // matches the GLSL `scalar` block layout used in the shader.
        struct DrawInfoGpu {
            float    model[16];        // 64
            uint64_t posAddr;          // 8
            uint64_t nrmAddr;          // 8
            uint64_t uvAddr;           // 8
            uint64_t prevPosAddr;      // 8
            uint64_t indexAddr;        // 8 (0 → non-indexed)
            uint64_t colorAddr;        // 8 (0 → no per-vertex color / vertexColors off)
            uint32_t instanceCustomIndex;
            uint32_t flags;            // bits 0..7 render flags | bits 8..15 semantic class id
            uint32_t indexed;
            float    polygonOffset;    // clip-z depth bias (reverse-Z: + = toward near = on top)
            uint32_t stableId;         // stable per-object instance id (-> outIds.y)
            uint32_t packedAttrs;      // BlasRecord::packedMask (also keeps 8-byte array stride)
        };
        static_assert(sizeof(DrawInfoGpu) == 136,
                      "DrawInfoGpu layout drifted from gbuffer_indirect.vert");

        // ── Stable / semantic object IDs for the segmentation AOVs ───────
        // outIds.x is the per-frame visible-set index; outIds.y must be STABLE
        // across frames. We assign a dense 16-bit id per Object3D (keyed by the
        // process-stable Object3D::id) the first time it is drawn, so the label
        // survives add/remove/hide/LOD. Callers may override the instance id and
        // set an 8-bit semantic class (folded into outIds.z bits 8..15 by
        // buildIndirectDrawData). Maps are keyed by Object3D::id (never a raw
        // pointer) so a deleted object leaves an inert entry, never a dangling read.
        std::unordered_map<unsigned int, uint16_t> autoStableIds_;      // Object3D::id -> dense id
        std::unordered_map<unsigned int, uint16_t> instanceIdOverride_; // user-set instance id
        std::unordered_map<unsigned int, uint16_t> classIds_;           // user-set semantic class
        uint32_t nextAutoStableId_ = 1;// 0 reserved for sky / unassigned

        uint16_t stableIdForObject(const Object3D& o) {
            // Both id maps are empty unless the app opts in (setInstanceId /
            // setClassId), which is the common case — skip the probe entirely.
            if (!instanceIdOverride_.empty()) {
                if (const auto it = instanceIdOverride_.find(o.id); it != instanceIdOverride_.end()) {
                    return it->second;
                }
            }
            const auto [it, inserted] = autoStableIds_.try_emplace(o.id, uint16_t(0));
            if (inserted) {
                it->second = static_cast<uint16_t>(nextAutoStableId_);
                if (nextAutoStableId_ < 0xFFFFu) ++nextAutoStableId_;// saturate at 65535
            }
            return it->second;
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
        struct OcclBitRange { uint32_t base; uint32_t capacity; };
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

        // Per-cull-mode dispatch span into indirectCmdBuffers[frame].
        struct DrawGroup {
            VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
            uint32_t        offset   = 0;// first cmd index (cmd-buffer-relative)
            uint32_t        count    = 0;
        };
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
        void recordRasterGbufPassInternal(VkCommandBuffer cb, uint32_t frame,
                                          VkRenderPass renderPass, VkFramebuffer fb,
                                          bool useMsaa, VkBuffer indirectBuffer,
                                          bool clear);

        // Lazily create the debug_resolve compute pipeline + descriptor set.
        void createDebugResolvePipeline();

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
        void recordHybridDebugResolve(VkCommandBuffer cb, uint32_t imageIndex, uint32_t frame);


        void createLightsUbos();

        // Walk the scene each frame for AmbientLight + DirectionalLight; pack
        // into the per-frame lights UBO. Direction is computed from the
        // light's world-space position toward its (possibly defaulted) target,
        // mirroring three.js's DirectionalLight.target convention. The shader
        // expects the L vector (toward the light), so we negate.
        void updateLightsUbo(uint32_t frame, Object3D& scene);

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
        // the caller encodes each level (bcn::buildMipChainRGBA8 +
        // bcn::bc7EncodeMode6) and this uploads the chain verbatim — one
        // staging buffer, one copy region per level. View/sampler setup
        // matches buildSampledImage2D (aniso + full LOD range).
        Image2D createSampledImageBC(uint32_t w, uint32_t h, VkFormat format,
                                     const std::vector<std::vector<std::uint8_t>>& levels,
                                     VkFilter filter, VkSamplerAddressMode addrU,
                                     VkSamplerAddressMode addrV,
                                     const char* debugName = nullptr) {
            Image2D out{};
            if (levels.empty()) return out;
            out.width  = w;
            out.height = h;
            out.format = format;
            const auto mipLevels = static_cast<uint32_t>(levels.size());
            out.mipLevels = mipLevels;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = format;
            ici.extent        = {w, h, 1};
            ici.mipLevels     = mipLevels;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx->allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
                  "vmaCreateImage(bc)");

            VkDeviceSize total = 0;
            for (const auto& l : levels) total += static_cast<VkDeviceSize>(l.size());
            Buffer staging = createBuffer(
                    ctx->allocator(), ctx->device(), total,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), staging.alloc, &mapped);
                auto* dst = static_cast<std::uint8_t*>(mapped);
                for (const auto& l : levels) {
                    std::memcpy(dst, l.data(), l.size());
                    dst += l.size();
                }
                flushHostWrites(ctx->allocator(), staging.alloc, 0, total);
                vmaUnmapMemory(ctx->allocator(), staging.alloc);
            }

            VkCommandBuffer cb = beginOneShot();

            VkImageMemoryBarrier toDst{};
            toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.image = out.image;
            toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toDst.subresourceRange.levelCount = mipLevels;
            toDst.subresourceRange.layerCount = 1;
            toDst.srcAccessMask = 0;
            toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &toDst);

            std::vector<VkBufferImageCopy> regions(mipLevels);
            VkDeviceSize offset = 0;
            for (uint32_t i = 0; i < mipLevels; ++i) {
                regions[i] = {};
                regions[i].bufferOffset = offset;// multiples of the 16-byte block size
                regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                regions[i].imageSubresource.mipLevel = i;
                regions[i].imageSubresource.layerCount = 1;
                regions[i].imageExtent = {std::max(1u, w >> i), std::max(1u, h >> i), 1};
                offset += static_cast<VkDeviceSize>(levels[i].size());
            }
            vkCmdCopyBufferToImage(cb, staging.handle, out.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   mipLevels, regions.data());

            VkImageMemoryBarrier toRead = toDst;
            toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 0, 0, nullptr, 0, nullptr, 1, &toRead);

            endAndSubmitOneShot(cb);
            destroyBufferMaybeBatched(staging);

            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = out.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = format;
            vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                              VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = mipLevels;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(bc)");

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            const float maxAniso = std::min(16.0f, props.limits.maxSamplerAnisotropy);

            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = filter;
            sci.minFilter = filter;
            sci.mipmapMode = (mipLevels > 1u) ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                              : VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = addrU;
            sci.addressModeV = addrV;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.anisotropyEnable = (mipLevels > 1u) ? VK_TRUE : VK_FALSE;
            sci.maxAnisotropy = (mipLevels > 1u) ? maxAniso : 1.0f;
            sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sci.unnormalizedCoordinates = VK_FALSE;
            sci.compareEnable = VK_FALSE;
            sci.minLod = 0.0f;
            sci.maxLod = (mipLevels > 1u) ? VK_LOD_CLAMP_NONE : 0.0f;
            check(vkCreateSampler(ctx->device(), &sci, nullptr, &out.sampler),
                  "vkCreateSampler(bc)");

            if (debugName) {
                ctx->setObjectName(out.image, debugName);
                ctx->setObjectName(out.view,  debugName);
            }
            return out;
        }

        // In-frame variant: records the upload into the CALLER's command
        // buffer (must be outside a render-pass instance) and retires the
        // staging buffer through the frame-serial queue — no submit, no
        // queue drain. A mid-record one-shot's vkQueueWaitIdle blocks on
        // every in-flight frame (~2 frames of GPU time); HUD TextSprites
        // re-rasterizing their atlas per ammo-counter change made that a
        // 40-50 ms hitch on every shot. Same-queue execution order + the
        // final SHADER_READ_ONLY barrier make samples in this and later
        // frames safe without any host wait.
        Image2D createSampledImage2DInFrame(VkCommandBuffer cb,
                                            uint32_t w, uint32_t h, VkFormat format,
                                            const void* pixels, VkDeviceSize byteSize,
                                            VkFilter filter, VkSamplerAddressMode addrU,
                                            VkSamplerAddressMode addrV,
                                            const char* debugName = nullptr) {
            Buffer staging{};
            Image2D out = buildSampledImage2D(cb, w, h, format, pixels, byteSize,
                                              filter, addrU, addrV, debugName, staging);
            retire(std::move(staging));
            return out;
        }

        // Shared body of the two variants above: image + view + sampler
        // creation and the recorded upload (staging copy + mip-blit chain +
        // final transition) into `cb`. `stagingOut` is still referenced by
        // cb when this returns — the caller covers its lifetime (drain+free
        // for the one-shot, frame-serial retire for the in-frame path).
        Image2D buildSampledImage2D(VkCommandBuffer cb,
                                    uint32_t w, uint32_t h, VkFormat format,
                                    const void* pixels, VkDeviceSize byteSize,
                                    VkFilter filter, VkSamplerAddressMode addrU,
                                    VkSamplerAddressMode addrV,
                                    const char* debugName,
                                    Buffer& stagingOut) {
            Image2D out{};
            out.width  = w;
            out.height = h;
            out.format = format;

            // Mip chain only for linear-filtered, multi-pixel images. The 1×1
            // env default and the 1D env CDF/marg LUTs use NEAREST + dim==1 so
            // they keep mipLevels=1 and never hit the blit path below.
            const bool wantMips = (filter == VK_FILTER_LINEAR) && (w > 1u || h > 1u);
            const uint32_t mipLevels = wantMips
                    ? (1u + static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(w, h))))))
                    : 1u;
            out.mipLevels = mipLevels;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = format;
            ici.extent        = {w, h, 1};
            ici.mipLevels     = mipLevels;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                              | (mipLevels > 1u ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0u);
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx->allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
                  "vmaCreateImage(env)");

            // Staging buffer with the source pixels.
            stagingOut = createBuffer(
                    ctx->allocator(), ctx->device(), byteSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            uploadHostVisible(ctx->allocator(), stagingOut, pixels, byteSize);

            // Transition mip 0 → TRANSFER_DST for the buffer copy.
            VkImageMemoryBarrier toDst{};
            toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.image = out.image;
            toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toDst.subresourceRange.baseMipLevel = 0;
            toDst.subresourceRange.levelCount = 1;
            toDst.subresourceRange.layerCount = 1;
            toDst.srcAccessMask = 0;
            toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &toDst);

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {w, h, 1};
            vkCmdCopyBufferToImage(cb, stagingOut.handle, out.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            if (mipLevels > 1u) {
                // Mip-chain build via vkCmdBlitImage. Each mip i transitions
                // from TRANSFER_DST → TRANSFER_SRC after being written, then
                // serves as the source for the i+1 blit. Final pass moves
                // every level to SHADER_READ_ONLY_OPTIMAL in one barrier.
                int32_t mipW = static_cast<int32_t>(w);
                int32_t mipH = static_cast<int32_t>(h);

                for (uint32_t i = 1; i < mipLevels; ++i) {
                    // Mip (i-1): TRANSFER_DST → TRANSFER_SRC.
                    VkImageMemoryBarrier b{};
                    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    b.image = out.image;
                    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    b.subresourceRange.baseMipLevel = i - 1;
                    b.subresourceRange.levelCount = 1;
                    b.subresourceRange.layerCount = 1;
                    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(cb,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 0, nullptr, 0, nullptr, 1, &b);

                    // Mip i starts UNDEFINED → TRANSFER_DST (we just allocated it).
                    VkImageMemoryBarrier bDst = b;
                    bDst.subresourceRange.baseMipLevel = i;
                    bDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    bDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    bDst.srcAccessMask = 0;
                    bDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(cb,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 0, nullptr, 0, nullptr, 1, &bDst);

                    const int32_t dstW = std::max(mipW >> 1, 1);
                    const int32_t dstH = std::max(mipH >> 1, 1);

                    VkImageBlit blit{};
                    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    blit.srcSubresource.mipLevel = i - 1;
                    blit.srcSubresource.layerCount = 1;
                    blit.srcOffsets[1] = {mipW, mipH, 1};
                    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    blit.dstSubresource.mipLevel = i;
                    blit.dstSubresource.layerCount = 1;
                    blit.dstOffsets[1] = {dstW, dstH, 1};

                    vkCmdBlitImage(cb,
                                   out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1, &blit, VK_FILTER_LINEAR);

                    mipW = dstW;
                    mipH = dstH;
                }

                // Whole chain → SHADER_READ_ONLY_OPTIMAL. Levels 0..N-2 are
                // currently TRANSFER_SRC, level N-1 is TRANSFER_DST.
                VkImageMemoryBarrier brs[2]{};
                brs[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                brs[0].image = out.image;
                brs[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                brs[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                brs[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                brs[0].subresourceRange.baseMipLevel = 0;
                brs[0].subresourceRange.levelCount = mipLevels - 1;
                brs[0].subresourceRange.layerCount = 1;
                brs[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                brs[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                brs[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                brs[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                brs[1] = brs[0];
                brs[1].subresourceRange.baseMipLevel = mipLevels - 1;
                brs[1].subresourceRange.levelCount = 1;
                brs[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                brs[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                vkCmdPipelineBarrier(cb,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                     0, 0, nullptr, 0, nullptr, 2, brs);
            } else {
                VkImageMemoryBarrier toRead = toDst;
                toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cb,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                     0, 0, nullptr, 0, nullptr, 1, &toRead);
            }

            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = out.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = format;
            vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                              VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = mipLevels;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(env)");

            // Anisotropic filtering is paired with the mip chain — they only
            // help together. Aniso without mips snaps to mip 0 (no benefit at
            // distance); mips without aniso blur at glancing angles. fillMat-
            // TextureInfos binds *this* per-image sampler into descriptor
            // binding 8, so settings here directly drive raster + RT albedo
            // sampling quality.
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            const float maxAniso = std::min(16.0f, props.limits.maxSamplerAnisotropy);

            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = filter;
            sci.minFilter = filter;
            sci.mipmapMode = (mipLevels > 1u)
                                     ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                     : VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = addrU;
            sci.addressModeV = addrV;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.anisotropyEnable = (mipLevels > 1u) ? VK_TRUE : VK_FALSE;
            sci.maxAnisotropy = (mipLevels > 1u) ? maxAniso : 1.0f;
            sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sci.unnormalizedCoordinates = VK_FALSE;
            sci.compareEnable = VK_FALSE;
            sci.minLod = 0.0f;
            sci.maxLod = (mipLevels > 1u) ? VK_LOD_CLAMP_NONE : 0.0f;
            check(vkCreateSampler(ctx->device(), &sci, nullptr, &out.sampler),
                  "vkCreateSampler(env)");

            if (debugName) {
                ctx->setObjectName(out.image, debugName);
                ctx->setObjectName(out.view,  debugName);
            }
            return out;
        }

        // ── ParticleSystem billboard resources ──────────────────────────────

        // Lazily build the 1×1 white texel bound for untextured particle
        // systems (matches the GL path, where an unset `tex` sampler reads
        // white). Created once; freed in deinit.
        void ensureParticleWhiteTexture();

        // Upload/refresh a particle texture, keyed on the raw Texture* (the
        // ShaderMaterial uniform holds no shared_ptr). Returns the cached
        // Image2D, or nullptr (caller falls back to the white default). Mirrors
        // OverlayPass::ensureSpriteAtlasTexture, minus the weak_ptr liveCheck.
        const Image2D* ensureParticleTexture(const Texture* tex) {
            if (!tex) return nullptr;
            Image& img = const_cast<Texture*>(tex)->image();
            const uint32_t w = img.width();
            const uint32_t h = img.height();
            if (w == 0 || h == 0) return nullptr;

            const unsigned int curVersion = tex->version();
            auto it = particleTexCache_.find(tex);
            if (it != particleTexCache_.end()) {
                ParticleTexRec& rec = it->second;
                const bool stale = rec.version != curVersion ||
                                   rec.width != w || rec.height != h;
                if (!stale) return &rec.image;
                // Retire the old image instead of a full device drain: in-flight
                // frames may still sample it (particle descriptors are allocated
                // per-frame from reset pools, so there's no descriptor-set hazard
                // — image lifetime is the only concern). VulkanRetireQueue.hpp.
                retire(std::move(rec.image));
                particleTexCache_.erase(it);
            }

            std::vector<unsigned char> rgba;
            const size_t pixels = static_cast<size_t>(w) * h;
            try {
                auto& src = img.data<unsigned char>();
                if (src.size() == pixels * 4) {
                    rgba.assign(src.begin(), src.end());
                } else if (src.size() == pixels * 3) {
                    rgba.resize(pixels * 4);
                    for (size_t i = 0; i < pixels; ++i) {
                        rgba[i * 4 + 0] = src[i * 3 + 0];
                        rgba[i * 4 + 1] = src[i * 3 + 1];
                        rgba[i * 4 + 2] = src[i * 3 + 2];
                        rgba[i * 4 + 3] = 255u;
                    }
                } else {
                    return nullptr;
                }
            } catch (const std::bad_variant_access&) {
                return nullptr;
            }

            // Same colorSpace→format rule as the sprite/bindless paths: only an
            // explicitly sRGB-tagged texture gets hardware sRGB decode on sample;
            // particle.frag re-encodes the linear product for the UNORM swapchain.
            const VkFormat fmt = (tex->colorSpace == ColorSpace::sRGB)
                                         ? VK_FORMAT_R8G8B8A8_SRGB
                                         : VK_FORMAT_R8G8B8A8_UNORM;
            char name[64];
            std::snprintf(name, sizeof(name), "particleTex[%p]",
                          static_cast<const void*>(tex));
            Image2D up = createSampledImage2D(
                    w, h, fmt, rgba.data(), rgba.size(),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    name);
            ParticleTexRec rec{};
            rec.image   = up;
            rec.version = curVersion;
            rec.width   = w;
            rec.height  = h;
            auto [ins, _] = particleTexCache_.emplace(tex, std::move(rec));
            return &ins->second.image;
        }

        void destroyParticleGeomRec(ParticleGeomRec& rec);

        // Ensure the per-geometry particle vertex/index buffers exist and the
        // animated attributes (position/normal/color) are current. uv + index
        // are static (uploaded once). Returns nullptr on malformed geometry.
        // Float-typed reads on purpose: the animated attributes re-upload on
        // every version bump, so narrow (compressAttributes) storage would
        // re-widen per update — particle geometry must stay float.
        ParticleGeomRec* ensureParticleGeom(const std::shared_ptr<BufferGeometry>& geomSp) {
            BufferGeometry* geom = geomSp.get();
            if (!geom) return nullptr;
            auto* posAttr  = geom->getAttribute<float>("position");
            auto* normAttr = geom->getAttribute<float>("normal");
            auto* uvAttr   = geom->getAttribute<float>("uv");
            auto* colAttr  = geom->getAttribute<float>("color");
            auto* idxAttr  = geom->getIndex();
            if (!posAttr || !normAttr || !uvAttr || !colAttr || !idxAttr) return nullptr;

            const uint32_t vtx = static_cast<uint32_t>(posAttr->count());
            const uint32_t idxCount = static_cast<uint32_t>(idxAttr->count());
            if (vtx == 0 || idxCount == 0) return nullptr;
            // Particle attributes are vec3/vec3/vec2/vec3 — bail if a custom
            // geometry doesn't match (the billboard pipeline assumes this layout).
            if (normAttr->count() != static_cast<int>(vtx) ||
                uvAttr->count()   != static_cast<int>(vtx) ||
                colAttr->count()  != static_cast<int>(vtx)) return nullptr;

            const unsigned int ver = geomVersionOf(*geom);

            auto uploadAnim = [&](ParticleGeomRec& rec) {
                uploadHostVisible(ctx->allocator(), rec.position, posAttr->array().data(), vtx * 3 * sizeof(float));
                uploadHostVisible(ctx->allocator(), rec.normal, normAttr->array().data(), vtx * 3 * sizeof(float));
                uploadHostVisible(ctx->allocator(), rec.color, colAttr->array().data(), vtx * 3 * sizeof(float));
            };

            auto it = particleGeomCache_.find(geom);
            if (it != particleGeomCache_.end()) {
                ParticleGeomRec& rec = it->second;
                const bool stale = rec.vertexCount != vtx || rec.indexCount != idxCount ||
                                   rec.liveCheck.expired() || rec.liveCheck.lock().get() != geom;
                if (!stale) {
                    if (rec.animVersion != ver) {
                        uploadAnim(rec);
                        rec.animVersion = ver;
                    }
                    return &rec;
                }
                // Topology change / recycled address — rebuild from scratch.
                // Retire the old buffers (in-flight frames may still draw from
                // them) instead of a full device drain. VulkanRetireQueue.hpp.
                retireParticleGeomRec(rec);
                particleGeomCache_.erase(it);
            }

            // Fresh build. All buffers host-visible; pos/normal/color re-uploaded
            // each frame, uv + index written once here.
            const VkBufferUsageFlags vbUsage =
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            ParticleGeomRec rec{};
            rec.vertexCount = vtx;
            rec.indexCount  = idxCount;
            rec.liveCheck   = geomSp;
            auto mkBuf = [&](VkDeviceSize bytes) {
                return createBuffer(ctx->allocator(), ctx->device(), bytes, vbUsage,
                                    VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            };
            rec.position = mkBuf(vtx * 3 * sizeof(float));
            rec.normal   = mkBuf(vtx * 3 * sizeof(float));
            rec.uv       = mkBuf(vtx * 2 * sizeof(float));
            rec.color    = mkBuf(vtx * 3 * sizeof(float));
            rec.index    = createBuffer(ctx->allocator(), ctx->device(),
                                        idxCount * sizeof(uint32_t),
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), rec.uv, uvAttr->array().data(), vtx * 2 * sizeof(float));
            uploadHostVisible(ctx->allocator(), rec.index, idxAttr->array().data(), idxCount * sizeof(uint32_t));
            auto [ins, _] = particleGeomCache_.emplace(geom, std::move(rec));
            uploadAnim(ins->second);
            ins->second.animVersion = ver;
            return &ins->second;
        }

        // Storage-image (GENERAL layout) factory used for the ReSTIR DI
        // reservoir ping-pong images (createReservoirImages). No staging
        // upload — contents are initialised the first frame after
        // sampleIndex resets to 0. The deferred shade reads/writes these
        // every frame, so we transition once at creation and keep them in
        // GENERAL forever after.
        Image2D createStorageImage2D(uint32_t w, uint32_t h, VkFormat format,
                                     VkImageUsageFlags extraUsage = 0,
                                     const char* debugName = nullptr) {
            Image2D out{};
            out.width  = w;
            out.height = h;
            out.format = format;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = format;
            ici.extent        = {w, h, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            // TRANSFER_DST so render-extent re-allocation can vkCmdClearColorImage
            // (and any other transfer-dst clears) without spec violation.
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | extraUsage;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx->allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
                  "vmaCreateImage(accum)");

            VkCommandBuffer cb = beginOneShot();
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = out.image;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
            endAndSubmitOneShot(cb);

            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = out.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = format;
            vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                              VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(accum)");

            if (debugName) {
                ctx->setObjectName(out.image, debugName);
                ctx->setObjectName(out.view,  debugName);
            }
            return out;
        }

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
        // current frame's geom/mat descriptors via a private RT pipeline owned
        // by `lidar_`. Synchronous: blocks the calling thread until per-beam
        // results land in `outResults`.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& outResults,
                       const LidarParams& params);

        // ── Hybrid raster G-buffer prepass implementation ───────────────────
        // Lazy-initialized on first render().
        // All resources owned by Impl; cleanup in dtor + destroyRasterGbufImages
        // is also called on swapchain resize.

        Image2D createAttachmentImage2D(uint32_t w, uint32_t h, VkFormat format,
                                        VkImageUsageFlags usage,
                                        VkImageAspectFlags aspect,
                                        const char* debugName = nullptr,
                                        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
            Image2D out{};
            out.width  = w;
            out.height = h;
            out.format = format;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = format;
            ici.extent        = {w, h, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = samples;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = usage;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx->allocator(), &ici, &aci,
                                 &out.image, &out.alloc, nullptr),
                  "vmaCreateImage(rasterGbuf attachment)");

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = out.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = format;
            vci.subresourceRange.aspectMask = aspect;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(rasterGbuf attachment)");
            if (debugName) {
                ctx->setObjectName(out.image, debugName);
                ctx->setObjectName(out.view,  debugName);
            }
            return out;
        }

        // 3D storage image (froxel volumetrics) — the volume sibling of
        // createAttachmentImage2D (OPTIMAL tiling, single mip, 3D view).
        Image2D createImage3D(uint32_t w, uint32_t h, uint32_t depth, VkFormat format,
                              VkImageUsageFlags usage, const char* debugName = nullptr) {
            Image2D out{};
            out.width  = w;
            out.height = h;
            out.format = format;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_3D;
            ici.format        = format;
            ici.extent        = {w, h, depth};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = usage;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx->allocator(), &ici, &aci,
                                 &out.image, &out.alloc, nullptr),
                  "vmaCreateImage(3D)");

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = out.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
            vci.format   = format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(3D)");
            if (debugName) {
                ctx->setObjectName(out.image, debugName);
                ctx->setObjectName(out.view,  debugName);
            }
            return out;
        }

        void destroyRasterGbufImages();

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

        // MSAA sibling of createRasterGbufRenderPass — same 6 attachments/
        // formats/subpass/dependency shape, only `samples` and the depth
        // finalLayout differ. The depth attachment's finalLayout stays
        // SHADER_READ_ONLY... no: depth needs its own aspect, so mirror the
        // 1× pass exactly except attachments[*].samples. Consumed only by
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

        // Walk the scene for visible world-space Sprites (screenSpace == false)
        // with a texture map, snapshotting their world transform + material into
        // lastVisibleSprites_. Run every perspective frame (sprites move / spawn
        // / expire constantly — no snapshot caching). Mirrors OverlayPass's
        // sprite collection, minus the screen-space branch.
        void collectWorldSprites(Object3D& scene);

        // Line geometry cache for the 3D hybrid overlay (recordCommandBuffer's
        // line-draw section). Keyed on raw BufferGeometry*; geomId in LineRec
        // guards against recycled-pointer aliasing. (OverlayPass owns a separate
        // cache of the same shape for the ortho path — these are NOT shared.)
        //
        // The counter is advanced once per SUBMITTED frame in endFrame, which
        // also runs sweepLineGeomCache. It used to be read but never written, so
        // every entry's lastTouch stayed 0 and nothing was ever evicted: an app
        // that rebuilds Line/Points geometry each frame (trajectory, scan and
        // detection-box overlays) leaked a buffer set per geometry, forever.
        std::unordered_map<const BufferGeometry*, vulkan::LineRec> lineGeomCache_;
        uint64_t overlayFrameCounter_ = 0;// lastTouch reference for lineGeomCache_ entries

        // Evict entries untouched for longer than the in-flight window. The
        // margin (> kFramesInFlight) guarantees no evicted buffer is still
        // referenced by an in-flight command buffer; the retire queue then holds
        // them until their frame serial has provably passed, so this is safe to
        // call from endFrame while earlier frames are still executing.
        static constexpr uint64_t kLineGeomEvictAge = 8;
        void sweepLineGeomCache() {
            if (lineGeomCache_.empty()) return;
            if (overlayFrameCounter_ <= kLineGeomEvictAge) return;
            const uint64_t cutoff = overlayFrameCounter_ - kLineGeomEvictAge;
            for (auto it = lineGeomCache_.begin(); it != lineGeomCache_.end();) {
                if (it->second.lastTouch < cutoff) {
                    retire(std::move(it->second.vertex));
                    retire(std::move(it->second.index));
                    retire(std::move(it->second.color));
                    it = lineGeomCache_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // Lazy-upload Line / LineSegments geometry into a host-visible
        // vertex (+ optional index) buffer pair. Returns nullptr if the
        // geometry has no usable position attribute. Cached by raw pointer
        // — first call allocates + writes, subsequent calls compare the
        // BufferAttribute::version counters and re-upload only when the
        // user mutated the data. In-place memcpy when the new size fits
        // the existing buffer; full recreate when it grew (matches the
        // refreshSkinnedBlas pattern — a write-during-read race for that
        // one frame is benign because the per-pixel result blends sub-
        // pixel anyway).
        const vulkan::LineRec* ensureLineGeometryUploaded(const BufferGeometry* geom) {
            if (!geom) return nullptr;
            auto posAttr = geom->getAttribute<float>("position");
            if (!posAttr || posAttr->count() == 0) return nullptr;
            auto* idxAttr = geom->getIndex();
            // Optional per-vertex color, used by AxesHelper-style overlays.
            // Untyped so a narrowed (compressAttributes) color still counts;
            // the upload below widens it through a FloatAttributeView.
            const auto* colAttr = geom->getAttribute("color");

            const uint32_t posVer = posAttr->version;
            const uint32_t idxVer = (idxAttr && idxAttr->count() > 0) ? idxAttr->version : 0u;
            const uint32_t colVer = (colAttr && colAttr->count() > 0) ? colAttr->version : 0u;

            auto it = lineGeomCache_.find(geom);
            if (it != lineGeomCache_.end() && it->second.geomId != geom->id) {
                // Recycled pointer: this address was a DIFFERENT geometry whose
                // buffers we still hold. Retire them and re-upload from scratch
                // (the version fields would otherwise alias — both at 0).
                // Through the retire queue, not destroyBuffer: this runs during
                // record, so the previous frame's line draw may still be reading
                // them. retire() no-ops on null handles.
                retire(std::move(it->second.vertex));
                retire(std::move(it->second.index));
                retire(std::move(it->second.color));
                lineGeomCache_.erase(it);
                it = lineGeomCache_.end();
            }
            if (it != lineGeomCache_.end()) {
                auto& rec = it->second;
                rec.lastTouch = overlayFrameCounter_;
                if (rec.positionVersion == posVer &&
                    rec.indexVersion    == idxVer &&
                    rec.colorVersion    == colVer) {
                    return &rec;
                }
                // Re-upload paths.
                const auto& posArr = posAttr->array();
                const VkDeviceSize vbBytes = posArr.size() * sizeof(float);
                if (vbBytes > rec.vertex.size) {
                    destroyBuffer(ctx->allocator(), rec.vertex);
                    rec.vertex = createBuffer(
                            ctx->allocator(), ctx->device(), vbBytes,
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                }
                uploadHostVisible(ctx->allocator(), rec.vertex, posArr.data(), vbBytes);
                rec.vertexCount     = static_cast<uint32_t>(posAttr->count());
                rec.positionVersion = posVer;

                if (idxAttr && idxAttr->count() > 0) {
                    const auto& indices = idxAttr->array();
                    const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                    if (rec.index.handle == VK_NULL_HANDLE || ibBytes > rec.index.size) {
                        if (rec.index.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.index);
                        rec.index = createBuffer(
                                ctx->allocator(), ctx->device(), ibBytes,
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    }
                    uploadHostVisible(ctx->allocator(), rec.index, indices.data(), ibBytes);
                    rec.indexCount   = static_cast<uint32_t>(indices.size());
                    rec.indexVersion = idxVer;
                } else if (rec.index.handle != VK_NULL_HANDLE) {
                    destroyBuffer(ctx->allocator(), rec.index);
                    rec.index        = {};
                    rec.indexCount   = 0;
                    rec.indexVersion = 0;
                }

                // Color buffer follows the same in-place / recreate logic.
                if (colAttr && colAttr->count() > 0) {
                    FloatAttributeView colView(colAttr);
                    const VkDeviceSize cbBytes = colView.size() * sizeof(float);
                    if (rec.color.handle == VK_NULL_HANDLE || cbBytes > rec.color.size) {
                        if (rec.color.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.color);
                        rec.color = createBuffer(
                                ctx->allocator(), ctx->device(), cbBytes,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    }
                    uploadHostVisible(ctx->allocator(), rec.color, colView.data(), cbBytes);
                    rec.colorVersion = colVer;
                } else if (rec.color.handle != VK_NULL_HANDLE) {
                    destroyBuffer(ctx->allocator(), rec.color);
                    rec.color        = {};
                    rec.colorVersion = 0;
                }
                return &rec;
            }

            // First-time upload.
            const auto& posArr = posAttr->array();
            vulkan::LineRec rec{};
            rec.vertexCount     = static_cast<uint32_t>(posAttr->count());
            rec.positionVersion = posVer;
            rec.geomId          = geom->id;
            rec.lastTouch       = overlayFrameCounter_;

            const VkDeviceSize vbBytes = posArr.size() * sizeof(float);
            rec.vertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), rec.vertex, posArr.data(), vbBytes);

            if (idxAttr && idxAttr->count() > 0) {
                const auto& indices = idxAttr->array();
                rec.indexCount   = static_cast<uint32_t>(indices.size());
                rec.indexVersion = idxVer;
                const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                rec.index = createBuffer(
                        ctx->allocator(), ctx->device(), ibBytes,
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), rec.index, indices.data(), ibBytes);
            }

            if (colAttr && colAttr->count() > 0) {
                FloatAttributeView colView(colAttr);
                rec.colorVersion = colVer;
                const VkDeviceSize cbBytes = colView.size() * sizeof(float);
                rec.color = createBuffer(
                        ctx->allocator(), ctx->device(), cbBytes,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), rec.color, colView.data(), cbBytes);
            }

            return &lineGeomCache_.emplace(geom, std::move(rec)).first->second;
        }

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
        // 1x1 VK_SAMPLE_COUNT_2_BIT dummy images for deferred_shade.comp's
        // dispatch-B sampler2DMS/usampler2DMS bindings when MSAA is off —
        // see gbufDummyMS_'s declaration for why a single-sample dummy can't
        // stand in here. Formats mirror normal/depth/ids/uv/albedo exactly
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
        int32_t ensureMaterialTexture(const std::shared_ptr<Texture>& texSp);

        // Build a tightly-packed RGBA8 sampled image from a material texture's
        // CPU image (BCn decompress / channel pad / float quantize). Returns a
        // null Image2D ({}) on unsupported/degenerate input. Slot-agnostic so the
        // initial upload (ensureMaterialTexture) and the in-place re-upload
        // (refreshDirtyMaterialTextures) share one code path.
        Image2D buildMaterialImage2D(const Texture* tex) {
            Image& img = const_cast<Texture*>(tex)->image();
            const uint32_t w = img.width();
            const uint32_t h = img.height();
            if (w == 0 || h == 0) return {};

            // Normalise everything to tightly-packed RGBA8. The pipeline
            // treats the bindless array as a uniform u8x4 sampler set, so
            // BCn blocks decompress, mono/dual-channel maps replicate or
            // pad, and float defaults clamp+quantize.
            const size_t pixels = static_cast<size_t>(w) * h;
            std::vector<unsigned char> rgba;
            std::vector<std::uint8_t> bcnRgba;
            const std::uint8_t* srcPtr = nullptr;
            int channels = 0;

            if (img.compressedFormat.has_value()) {
                const auto& blocks = img.data<unsigned char>();
                bcnRgba = bcn::bcnDecompress(
                        blocks.data(),
                        static_cast<int>(w),
                        static_cast<int>(h),
                        *img.compressedFormat);
                if (bcnRgba.empty()) {
                    std::cerr << "[VulkanRenderer] unsupported compressed format 0x"
                              << std::hex << *img.compressedFormat << std::dec
                              << " for material tex (" << w << "x" << h << ")\n";
                    return {};
                }
                srcPtr = bcnRgba.data();
                channels = 4;
            } else {
                bool isU8 = true;
                try {
                    auto& src = img.data<unsigned char>();
                    if (src.size() % pixels != 0) {
                        std::cerr << "[VulkanRenderer] unsupported pixel layout for material tex ("
                                  << src.size() << " bytes for " << w << "x" << h << ")\n";
                        return {};
                    }
                    channels = static_cast<int>(src.size() / pixels);
                    if (channels < 1 || channels > 4) {
                        std::cerr << "[VulkanRenderer] unsupported channel count " << channels
                                  << " for material tex (" << w << "x" << h << ")\n";
                        return {};
                    }
                    srcPtr = src.data();
                } catch (const std::bad_variant_access&) {
                    isU8 = false;
                }
                if (!isU8) {
                    // Float-pixel default (e.g. Bistro's 1×1 RGBA32F constants).
                    // Quantise to u8 with sRGB-agnostic clamp; tiny default
                    // textures only need the linear value, and HDR ranges are
                    // expressed via material scalars instead.
                    auto& srcF = img.data<float>();
                    if (srcF.size() % pixels != 0) {
                        std::cerr << "[VulkanRenderer] unsupported float-pixel layout for material tex ("
                                  << srcF.size() * sizeof(float) << " bytes for "
                                  << w << "x" << h << ")\n";
                        return {};
                    }
                    const int fch = static_cast<int>(srcF.size() / pixels);
                    if (fch < 1 || fch > 4) {
                        std::cerr << "[VulkanRenderer] unsupported float channel count " << fch
                                  << " for material tex\n";
                        return {};
                    }
                    rgba.resize(pixels * 4);
                    for (size_t i = 0; i < pixels; ++i) {
                        float r = srcF[i * fch + 0];
                        float g = (fch >= 2) ? srcF[i * fch + 1] : r;
                        float b = (fch >= 3) ? srcF[i * fch + 2] : ((fch == 1) ? r : 0.f);
                        float a = (fch >= 4) ? srcF[i * fch + 3] : 1.f;
                        auto q = [](float v) {
                            if (!(v == v)) v = 0.f;// NaN→0
                            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
                            return static_cast<unsigned char>(v * 255.f + 0.5f);
                        };
                        rgba[i * 4 + 0] = q(r);
                        rgba[i * 4 + 1] = q(g);
                        rgba[i * 4 + 2] = q(b);
                        rgba[i * 4 + 3] = q(a);
                    }
                }
            }

            // Expand srcPtr (BCn or u8) into rgba; float branch already filled it.
            if (rgba.empty()) {
                rgba.resize(pixels * 4);
                if (channels == 4) {
                    std::memcpy(rgba.data(), srcPtr, pixels * 4);
                } else {
                    for (size_t i = 0; i < pixels; ++i) {
                        const unsigned char r = srcPtr[i * channels + 0];
                        const unsigned char g = (channels >= 2) ? srcPtr[i * channels + 1] : r;
                        const unsigned char b = (channels >= 3) ? srcPtr[i * channels + 2]
                                                                : ((channels == 1) ? r : 0);
                        const unsigned char a = (channels >= 4) ? srcPtr[i * channels + 3] : 255u;
                        rgba[i * 4 + 0] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
                    }
                }
            }

            // sRGB tag → hardware decode at sample time. Loaders should mark
            // albedo maps as SRGBColorSpace; legacy (untagged) textures fall
            // through as UNORM so the shader sees raw channel values.
            const bool srgb = tex->colorSpace == ColorSpace::sRGB;
            char texName[80];
            std::snprintf(texName, sizeof(texName),
                          "materialTexture (%ux%u, tex=%p)",
                          w, h, static_cast<const void*>(tex));

            // BC7 transcode: 4x less VRAM and 4x less bandwidth per sample.
            // Everything above was normalised to RGBA8 — including BCn (DDS)
            // sources, which the old path decompressed and uploaded FAT at 4x
            // their on-disk size. Mips are built + encoded on the CPU (BC
            // images cannot blit-downsample); tiny images (LUT-like defaults,
            // 1x1 constants) stay uncompressed, as does everything when the
            // device lacks BC7 or THREEPP_NO_BC=1 (same-binary A/B hatch).
            static const bool noBc = std::getenv("THREEPP_NO_BC") != nullptr;
            if (!noBc && w >= 8u && h >= 8u && bc7SampledSupported()) {
                auto mips = bcn::buildMipChainRGBA8(rgba.data(), static_cast<int>(w),
                                                    static_cast<int>(h), srgb);
                std::vector<std::vector<std::uint8_t>> blocks;
                blocks.reserve(mips.size() + 1);
                blocks.push_back(bcn::bc7EncodeMode6(rgba.data(), static_cast<int>(w),
                                                     static_cast<int>(h)));
                int mw = static_cast<int>(w), mh = static_cast<int>(h);
                for (const auto& lvl : mips) {
                    mw = std::max(1, mw >> 1);
                    mh = std::max(1, mh >> 1);
                    blocks.push_back(bcn::bc7EncodeMode6(lvl.data(), mw, mh));
                }
                return createSampledImageBC(
                        w, h,
                        srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK,
                        blocks,
                        VK_FILTER_LINEAR,
                        wrapToVk(tex->wrapS),
                        wrapToVk(tex->wrapT),
                        texName);
            }

            const VkFormat fmt = srgb ? VK_FORMAT_R8G8B8A8_SRGB
                                      : VK_FORMAT_R8G8B8A8_UNORM;
            return createSampledImage2D(
                    w, h, fmt,
                    rgba.data(), rgba.size(),
                    VK_FILTER_LINEAR,
                    wrapToVk(tex->wrapS),
                    wrapToVk(tex->wrapT),
                    texName);
        }

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
            return !eventCamEnabled_ && (useFsr() || useDlss() || gbufMsaaSamples_ <= 1);
        }
        // The sampler the material-texture bindings use this frame: AUTO
        // policy (aniso when unjittered, isotropic when jittered) unless
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
            rasterMatTexValid_.fill(0);
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

        // Scene shade hook. Called from recordCommandBuffer between the shared
        // G-buffer/AS head and the shared bloom/TAA tail. VulkanRenderer::Impl
        // overrides it to dispatch the analytic deferred shade.
        virtual void recordSceneDispatch(VkCommandBuffer cb, uint32_t setIdx,
                                         VkExtent2D ext, VkExtent2D ptExt,
                                         uint32_t exposureBits) = 0;

        // True when decal meshes (layer-tagged as decals) should be drawn
        // into the raster G-buffer. The deferred renderer renders decals.
        virtual bool decalsEnabled() const = 0;

        // HDRI sun extraction (deferred only). When true,
        // refreshEnvTextureFromScene detects the env's dominant compact bright
        // source, prefilters PMREM mips 1+ from a sun-clamped copy (no glossy /
        // rough env lookup ever integrates the raw ~10⁴:1 disc — the "bright
        // spec blobs in reflections" artifact), and updateLightsUbo re-injects
        // the removed energy as an analytic directional light (sharp correct
        // sun highlight, jittered soft RT shadows, GI bounce, water glints).
        // (The now-removed path tracer kept the raw env instead: its env-CDF
        // NEE + MIS already handled HDRI suns without fireflies.)
        virtual bool envSunExtractionWanted() const { return false; }

        // ONE-SUN POLICY (consulted only when extraction is wanted): when true
        // (VulkanRenderer::EnvSunPolicy::Auto), the extracted env sun is
        // injected only while the scene has NO visible DirectionalLight of its
        // own — an explicit scene light claims the sun role (scenes authored
        // for raster renderers carry a stand-in sun light because raster can't
        // shadow from an env map; injecting the env sun on top lights and
        // shadows the scene with TWO suns). false = Always inject.
        virtual bool envSunDefersToSceneSun() const { return true; }

        // Called once after bloom_->createImages() (initial creation and resize).
        // Implementations that hold references to sceneHdr views (e.g. auto-exposure)
        // override to rewire their descriptor sets.
        virtual void onAfterBloomCreateImages() {}

        // Called from beginDeferredFrame() after the per-frame fence wait and all
        // CPU-side setup, immediately before recordCommandBuffer(). Implementations
        // can do per-frame CPU readbacks here (histograms, timings) since the prior
        // GPU slot for this frame index is guaranteed retired.
        // dt = seconds since the last call to this function (0 on first call).
        virtual void onBeginDeferredFrame(uint32_t /*frame*/, float /*dt*/) {}

        // Exposure value used for this frame's composite push constant.
        // Physical camera mode derives it from aperture/shutter/ISO;
        // overridden by VulkanRenderer::Impl when auto-exposure is active
        // (which composes the physical exposure as its base there).
        [[nodiscard]] virtual float currentExposure() const {
            return physicalCamera_ ? physicalExposure() : toneMappingExposure_;
        }

        void recordCommandBuffer(VkCommandBuffer cb, uint32_t imageIndex);

        // Records the ImGui (or any overlay) callback inside a dynamic render
        // pass, then transitions the swapchain image GENERAL → PRESENT_SRC.
        // When no overlay callback is set, just emits the GENERAL → PRESENT_SRC
        // barrier directly. Called from endFrame() immediately before
        // vkEndCommandBuffer + submit.
        // Snapshot the post-TAA swapchain image into the host-visible
        // sceneCaptureBuf_ before any overlay (sprites, ImGui) composites.
        // Allocates / resizes the buffer lazily on first use or swapchain
        // resize. Inserts GENERAL → TRANSFER_SRC → GENERAL barriers so the
        // downstream overlay passes see the swapchain in the layout they
        // expect.
        void recordSceneCapture(VkCommandBuffer cb, uint32_t imageIndex);

        // ── Event-camera shade compute ─────────────────────────────────
        // Creates the event_shade compute pipeline + descriptor set
        // layout. Called once, lazily, on first setEventCameraEnabled.
        void createEventShadePipeline();

        // (Re-)allocate eventLumaBuf_ at the swapchain dimensions.
        // Called from setEventCameraEnabled and on resize.
        void allocateEventLumaBuffer(uint32_t w, uint32_t h);

        // Refresh descriptor set bindings + dispatch the event_shade
        // compute. Called once per frame after the gbuf prepass, before
        // the event_detect dispatch. Gbuf images are in
        // SHADER_READ_ONLY_OPTIMAL at this point (same as for the main
        // deferred shade's gbuf consumption).
        void recordEventShade(VkCommandBuffer cb, uint32_t frame);

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
