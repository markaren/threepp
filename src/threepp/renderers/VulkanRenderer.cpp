// VulkanRenderer — hardware ray-traced renderer with dynamic scene rebuild.
//
// On every render() we walk the scene and fingerprint each visible Mesh
// (geometry/material identity + world matrix + PBR scalars). When the
// fingerprint matches the previous frame we skip the rebuild and reuse
// the existing TLAS / desc tables. When it differs we wait the GPU idle,
// retire the TLAS + scene-side desc buffers, build fresh ones, and rewrite
// bindings 0/3/4 across every descriptor set. BLAS records stay cached by
// BufferGeometry pointer so static geometry is never re-traced.
//
// Two render modes (RenderMode):
//   • ReferencePT — an iterative path tracer driven from raygen: each
//     closest_hit returns direct radiance (Lambert + VNDF-sampled GGX against
//     AmbientLight/DirectionalLights with shadow rays, plus emissive/ReSTIR
//     area lights) + a sampled bounce direction; raygen loops to kMaxBounces
//     with Russian Roulette, accumulating into a persistent rgba32f image that
//     resets on camera/env/scene change.
//   • RasterFirst — a hybrid: a raster G-buffer supplies primary visibility,
//     then deferred_shade.comp lights it analytically and adds ray-query
//     accents (shadows, reflections, AO/GI), denoised + TAA-resolved.
// Both sample `scene.environment` / `scene.background` HDR equirects (GGX-
// prefiltered into a PMREM mip chain) for the background miss and IBL.

#define VMA_IMPLEMENTATION
#include "vulkan/VulkanContext.hpp"
#include "vulkan/VulkanResources.hpp"
#include "vulkan/Denoiser.hpp"
#include "vulkan/EnvPrefilter.hpp"
#include "vulkan/EventCameraDetector.hpp"
#include "vulkan/LidarScanner.hpp"
#include "vulkan/PhotonCaustics.hpp"
#include "vulkan/SkinningPipeline.hpp"
#include "vulkan/TetSkinningPipeline.hpp"
#include "vulkan/GpuTimings.hpp"
#include "vulkan/OverlayPass.hpp"
#include "vulkan/VulkanFrameTypes.hpp"
#include "vulkan/TaaResolve.hpp"
#include "vulkan/BloomPass.hpp"
#include "vulkan/DeferredShade.hpp"
#include "vulkan/WaterDisplacePipeline.hpp"
#include "vulkan/FoamWorldPipeline.hpp"
#include "vulkan/shaders/vulkan_shared.h"// MaterialDesc + kMaxMaterialTextures + photon-grid constants — same source the shaders read
#include "wgpu/pathtracer/WgpuPathTracerEnvCdf.hpp"// reused: pure C++ template, no WGPU deps

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/canvas/Canvas.hpp"
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
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Frustum.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

// stb_image_write — implementation is already compiled in GLRenderer.cpp.
#include "stb_image_write.h"

#include "threepp/renderers/vulkan/shaders/raygen.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/raygen.rgen.ser.spv.h"
#include "threepp/renderers/vulkan/shaders/miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit.rchit.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit_alpha.rahit.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_anyhit.rahit.spv.h"
// (photon shader SPVs moved into vulkan/PhotonCaustics.cpp)
// (denoise + denoise_atrous shader SPVs moved into vulkan/Denoiser.cpp)
// (prefilter_env shader SPV moved into vulkan/EnvPrefilter.cpp)
// (skinning shader SPV moved into vulkan/SkinningPipeline.cpp)
// (water_displace shader SPV moved into vulkan/WaterDisplacePipeline.cpp)
#include "threepp/renderers/vulkan/shaders/gbuffer.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuffer.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuffer_indirect.vert.spv.h"
// (taa_resolve.comp.spv moved into vulkan/TaaResolve.cpp)
#include "threepp/renderers/vulkan/shaders/overlay.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_depth.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_depth.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_color.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_color.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_point.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_point.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/event_shade.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_sprite.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_sprite.frag.spv.h"

#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerBCn.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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
    using vulkan::TimingPass;
    using vulkan::TP_RasterGbuf;
    using vulkan::TP_OverlayDepth;
    using vulkan::TP_PhotonEmit;
    using vulkan::TP_PathTrace;
    using vulkan::TP_Denoise;
    using vulkan::TP_TAA;
    using vulkan::TP_OverlayDraw;

    namespace {
        // Frames-in-flight depth. Bumped from 2 → 3 to deepen CPU/GPU
        // pipelining: while frame N+2 is being recorded on the CPU, frame N
        // and frame N+1 can be in different stages of GPU execution. Hides
        // CPU jitter (scene-build, ImGui, frustum cull) without changing the
        // GPU schedule (queue is still serial — async compute would do that,
        // and is a much larger change).
        //
        // The 2-slot ping-pong (accumImagesPP / gbufImagesPP / denoiser_'s
        // moments / reservoirImagesPP) stays at 2 entries — Vulkan queue execution is
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

    struct VulkanRenderer::Impl {
        Canvas& canvas;
        WindowSize size;
        float pixelRatio = 1.f;
        Color clearColor{0.f, 0.f, 0.f};
        float clearAlpha = 1.f;
        Vector4 viewport;
        Vector4 scissor;
        bool scissorTest = false;

        // Mirrored from VulkanRenderer (Renderer base) at the start of each
        // render() so the renderFrame path can read them without a pointer back
        // into the public class. Synced unconditionally — toggling these never
        // resets the accumulator (tone mapping is display-only).
        ToneMapping toneMapping_ = ToneMapping::None;
        float       toneMappingExposure_ = 1.f;

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
            Buffer foam; // .handle == VK_NULL_HANDLE unless this is an FFT-displaced ocean mesh
            // Previous-frame vertex positions, allocated for skinned + displaced
            // meshes only. Used by the hybrid raster prepass to compute
            // per-vertex motion vectors (skinned/displaced surfaces deform
            // each frame; the rigid-body motionMat alone produces zero motion
            // and ghosts under TAA + PT temporal accumulation). Static meshes
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
        };
        std::unordered_map<const BufferGeometry*, std::unique_ptr<BlasRecord>> blasCache;

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
            // Legacy CPU-skin scratch — kept for potential fallback paths
            // but no longer touched on the per-frame hot path (replaced by
            // the GPU skinning compute pipeline below).
            std::vector<float> deformedPositions;
            std::vector<float> deformedNormals;
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

        // Single TLAS over all mesh instances in the scene.
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        Buffer tlasBuffer;
        Buffer tlasInstancesBuffer;

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
            // previous frame's vertex data. Chit interpolates these to give
            // raygen a per-vertex "previous world position" so reprojection
            // tracks the same surface point across deformation. For static /
            // rigid-only meshes: set equal to vertexAddress, in which case the
            // chit reads the same data twice (no harm; equals current pos).
            VkDeviceAddress prevVertexAddress;
            uint32_t indexed;
            uint32_t _pad;
        };
        // MaterialDesc layout lives in vulkan_shared.h (the same file the GLSL
        // path-tracer shaders pull in via #include). Bringing it into Impl scope
        // here keeps the existing `MaterialDesc md{};` call sites unchanged.
        using MaterialDesc = threepp::vulkan_pt::MaterialDesc;

        Buffer geometryDescsBuffer;
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
        };
        struct GpuSpotLight {
            float position[3];   float range;
            float color[3];      float decay;
            float direction[3];  // toward target (emission direction)
            float cosAngleOuter; // cos(angle)
            float cosAngleInner; // cos(angle * (1-penumbra))
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
        static_assert(sizeof(GpuPointLight) == 32);
        static_assert(sizeof(GpuSpotLight)  == 52);
        static_assert(sizeof(GpuRectLight)  == 60);
        std::array<Buffer, kFramesInFlight> lightsUbos{};

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
            float _pad[3];
        };
        static_assert(sizeof(GpuFogUbo) == 48);
        std::array<Buffer, kFramesInFlight> fogUbos{};
        float    fogAnisotropy_ = 0.0f;
        float    fogWaterSurfaceY_ = 1e30f;
        uint64_t prevFogHash_ = 0u;

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

        // Ocean fine-cascade normal-map source (binding 21 in rtDsLayout).
        // Default 1×1 dummy R32F, replaced with the active DisplacedMesh's
        // cascade-2 height image when one is in the scene. closest_hit
        // samples this on `thinWalled` materials at world-space XZ via
        // finite differences to perturb the macro normal — adds sub-mesh
        // chop detail (FFT cells finer than the 1 m mesh resolves).
        Image2D oceanFineHeightDummy{};
        VkImageView oceanFineHeightView   = VK_NULL_HANDLE;// either dummy or cascade-2 view
        VkSampler   oceanFineHeightSampler = VK_NULL_HANDLE;
        float       oceanFineTileSize     = 0.f;          // 0 disables sampling in shader

        // World-space foam (binding 33). Built by FoamWorldPipeline each
        // frame from a DisplacedMesh's foamImage. closest_hit samples it
        // at world XZ on water hits — replaces the per-vertex foam buffer
        // that used to live on the BLAS. Held as a dummy 1×1 R32F when no
        // DisplacedMesh is in the scene.
        Image2D oceanFoamDummy{};
        VkImageView oceanFoamView   = VK_NULL_HANDLE;
        VkSampler   oceanFoamSampler = VK_NULL_HANDLE;
        float       oceanFoamTileSize = 0.f;              // 0 disables sampling

        // Tileable foam detail texture (RT binding 45 / deferred binding 34).
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
        // probability of picking row r (latitudes weighted by cos(latitude)).
        // totalSum = Σ lum·cos(lat) — pdf normalisation.
        // Rebuilt whenever the env texture changes; bound 1×1 dummy (totalSum=0)
        // when env is solid color or default — chit gates env CDF on totalSum>0.
        Image2D envCdfImage{};
        Image2D envMargImage{};
        // 64×64 R8 blue-noise tile for sub-pixel jitter. Generated once via
        // void-and-cluster (Ulichney 1993) and uploaded at startup. Adjacent
        // pixels share correlated values (silhouette stability) but globally
        // decorrelated (no coherent shake). Animated temporally by offsetting
        // the lookup coords per frame; AA convergence via accumulator.
        Image2D blueNoiseImage{};
        uint32_t envCdfWidth_  = 1;
        uint32_t envCdfHeight_ = 1;
        float    envCdfTotalSum_ = 0.0f;

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
        // uploads once. All material textures share textureSampler_ (linear,
        // repeat); per-texture filter/wrap is a v2 concern.
        // kMaxMaterialTextures, kPhotonGridBits, kPhotonGridSize, kPhotonsPerCell,
        // kPhotonEmitDim, and kGatherRadius all come from the shared header
        // (vulkan_shared.h, included near the top of this file). Editing any of
        // them there propagates to every shader on the next clean rebuild.
        std::vector<Image2D> materialTextures;// owns image + view (sampler is shared)
        // Cache value pairs (weak_ptr, slot). The weak_ptr is the liveness tag —
        // when a Texture is destroyed (model unloaded), the entry is pruned in
        // ensureSceneBuilt so a future Texture* address-collision doesn't read
        // back stale GPU data.
        std::unordered_map<const Texture*, std::pair<std::weak_ptr<Texture>, uint32_t>> textureCache;
        std::vector<uint32_t> freeTextureSlots;// slots reclaimed by prune
        VkSampler textureSampler_ = VK_NULL_HANDLE;

        // Continuous-motion accumulation. Two ping-pong slots for the running
        // mean (.rgb) plus per-pixel frame count (.w packed via uintBitsToFloat),
        // and two ping-pong gbuf slots holding primary world hit (.xyz) plus
        // mesh-ID guard (.w packed). At frame N the shader writes slot
        // (N & 1) and reads slot ((N+1) & 1). sampleIndex now drives only the
        // Halton jitter; per-pixel FC packed in accumImage.w replaces it as
        // the accumulation count and survives camera / object motion.
        std::array<Image2D, 2> accumImagesPP{};
        std::array<Image2D, 2> gbufImagesPP{};
        // ReSTIR DI Stage 1b — per-pixel reservoir ping-pong storage. Two
        // physical images per logical buffer: reservoirPosImagesPP carries
        // lightPos.xyz + lightType.w (rgba32f), reservoirWImagesPP carries
        // W_sum + M + W + p_hat (rgba16f). Bindings 28/30 = write (frame N),
        // 29/31 = read (frame N-1); descriptor sets are baked once with the
        // f&1 → image-slot mapping just like accumImagesPP / gbufImagesPP.
        std::array<Image2D, 2> reservoirPosImagesPP{};
        std::array<Image2D, 2> reservoirWImagesPP{};
        // ReSTIR GI Stage 1b — per-pixel reservoir ping-pong storage. Three
        // physical images per logical buffer (all rgba32f for precision on
        // world-space xs):
        //   giResXsImagesPP : xs.xyz + W_sum   (chosen sample position + RIS sum)
        //   giResNsImagesPP : ns.xyz + M       (chosen sample normal + count)
        //   giResLoImagesPP : Lo.rgb + W       (chosen sample radiance + finalized RIS weight)
        // omegaI is re-derived per-frame as `normalize(xs - currentHitPos)` so
        // it doesn't need its own storage pair. p_hat is also not stored —
        // resampling re-evaluates target at OUR pixel via evalGiTarget, so the
        // stored sample's p_hat (computed at a possibly-different pixel) is
        // discarded each merge. Bindings 38/40/42 = write (frame N), 39/41/43
        // = read (frame N-1); descriptor sets bake the f&1 → image-slot map.
        std::array<Image2D, 2> giResXsImagesPP{};
        std::array<Image2D, 2> giResNsImagesPP{};
        std::array<Image2D, 2> giResLoImagesPP{};
        // Filtered and moments ping-pong images are owned by `denoiser_`;
        // bindings 20 / 33 / 34 are wired via accessors at descriptor-write
        // time. See vulkan/Denoiser.hpp.
        uint32_t accumWriteIdx_ = 0;
        uint32_t sampleIndex = 0;
        // Samples per pixel per frame. Each sample is an independent
        // jittered primary ray; raygen sums them and the accumulator
        // advances FC by `spp` so the running mean stays correctly weighted.
        uint32_t samplesPerPixel_ = 1;
        // # of extra primary rays fired at detected silhouette pixels
        // (gbufIds-mismatch / depth-gradient / diagonal neighbour).
        // 0 disables silhouette MSAA entirely.
        uint32_t edgeMsaaExtra_   = 1;
        // Max real scatter events per path (raygen kMaxBounces). 4 matches the
        // prior compile-time value; setMaxBounces clamps to [1, 16]. Diffuse /
        // glossy primaries further cap themselves at 3 / 4 in raygen, so this
        // mostly affects deeper metal / mirror / glass paths.
        uint32_t maxBounces_      = 4;
        // Prev-frame camera packed as four vec4s (matches PrevCameraUbo):
        //   [0..3]  = vec4(pos.xyz,  projScaleX)   → prevCamPosX
        //   [4..7]  = vec4(fwd.xyz,  projScaleY)   → prevCamFwdY
        //   [8..11] = vec4(rgt.xyz,  0)             → prevCamRgt
        //   [12..15]= vec4(up.xyz,   0)             → prevCamUp
        std::array<float, 16> prevCamBufData_{};
        bool prevCameraValid = false;

        // Set when this frame's camera viewProj or any mesh transform differs
        // from the previous frame. Forwarded to raygen via a push-constant bit
        // so the shader only does reprojection math when motion actually
        // occurred — static frames take the self-tap path and accumulate
        // bit-for-bit with no float-precision pixel-snap drift (the cause of
        // the visible static-scene wobble we observed when always reprojecting).
        bool motionThisFrame_ = false;

        // Camera viewProj changed this frame — separate bit so the raygen can
        // reproject all non-sky pixels for camera motion while leaving pixels
        // whose mesh did not move at full FC. Without this split a single
        // moving mesh anywhere in the scene would halve FC scene-wide
        // (uniformly noisy background equilibrating at FC≈2).
        bool cameraMovedThisFrame_ = false;

        // Per-entry "moved" bitmask, one bit per TLAS instance. Bit i set when
        // entry i's effective worldMatrix / pose / displacement changed since
        // last frame. Sized to ceil(meshCount/32) words at scene rebuild;
        // uploaded to meshMovedBitsBuffers[currentFrame] each frame so raygen
        // can index by primaryInstanceId-1 to make the FC-halving decision
        // per-pixel instead of scene-wide.
        std::vector<uint32_t> meshMovedBits_;
        std::array<Buffer, kFramesInFlight> meshMovedBitsBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> meshMovedBitsBufferCapacity{};

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
        // worldMatrix = mesh->matrixWorld * instanceMatrix[i] (mirrors the WGPU
        // PT's RtMeshEntry pattern in WgpuPathTracerAtlas.cpp).
        struct MeshEntry {
            Mesh*    mesh;
            std::array<float, 16> worldMatrix;
            uint32_t instanceIndex;// 0 for non-instanced
            // Hybrid overlay flag: wireframe-flagged material OR mesh.layers
            // includes the configured overlayLayer_. Excluded from TLAS,
            // raster G-buffer, and emissive-tri NEE so PT can't see/shadow
            // them; drawn instead by the post-TAA overlay pass.
            bool     isOverlay = false;
            // Cached type probes. Resolved once per Mesh in ensureSceneBuilt's
            // traverseVisible callback (before the InstancedMesh fork so an
            // N-instance mesh costs 3 dynamic_casts, not 3·N). Consumers
            // (resolveBlasForEntry, recordRasterGbufPass, TLAS refit, bone /
            // displaced / morph dirty-detection) read these flags instead of
            // casting every frame.
            bool     isSkinned   = false;
            bool     isDisplaced = false;
            bool     isMorphed   = false;
            bool     isTet       = false;
            bool     isInstanced = false;// entry came from an InstancedMesh expansion
                                         // (the snapshot fast path recomputes its world
                                         // matrix as matrixWorld * getMatrixAt(i))
            // Frustum-cull bit, populated once per frame by
            // cullEntriesAgainstFrustum() right before record. Raster passes
            // skip entries with inFrustum == false to dodge the GPU's per-
            // draw command-processor overhead on off-screen geometry. The
            // PT path is unaffected — TLAS culling handles it implicitly.
            // Defaults to true so passes work before the first cull pass.
            // Skinned / displaced / morphed entries always stay true; their
            // local AABB doesn't reflect deformed extents.
            bool     inFrustum   = true;
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
            const void* geom = nullptr;               // mesh/line/points geometry
            BufferGeometry* geomB = nullptr;          // typed view of geom (attributesVersion reads)
            const Material* mat = nullptr;            // mesh material
            const MaterialWithWireframe* wf = nullptr;// cached cast of mat (same object ⇒ still valid)
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
        std::vector<SnapNode> sceneSnapshot_;

        // Classification-routing flags for a mesh — shared by the snapshot
        // build and the replay walk so the two can't drift.
        uint32_t snapMeshFlags(Mesh& m, const MaterialWithWireframe* wf) const {
            uint32_t fl = kSnapKindMesh;
            if (auto geom = m.geometry()) {
                if (geom->hasAttribute("position")) fl |= kSnapHasPos;
                if (geom->hasAttribute("normal")) fl |= kSnapHasNorm;
            }
            if (wf && wf->wireframe) fl |= kSnapWire;
            if (overlayLayer_ >= 0 &&
                m.layers.isEnabled(static_cast<unsigned>(overlayLayer_))) fl |= kSnapOverlay;
            if (auto mat = m.material(); mat && mat->tetSkinning && mat->tetTexture) fl |= kSnapTet;
            return fl;
        }

        // Replay the last expansion's traversal against the snapshot. true ⇒
        // tree shape, visibility, classification routing and all mesh/geom/mat
        // pointers are unchanged since the last full expansion.
        bool sceneSnapshotMatches(Object3D& scene) {
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
                if (kind == kSnapKindOther) return;// pointer identity is all we need
                // Same object at the same address ⇒ same dynamic type — the
                // typed pointers recorded by the full pass are still valid, so
                // the walk pays zero dynamic_casts. attributesVersion()
                // unchanged ⇒ the hasPos/hasNorm flag bits are still valid
                // without re-doing the string-keyed attribute lookups.
                if (kind == kSnapKindLine || kind == kSnapKindPoints) {
                    auto geom = sn.line ? sn.line->geometry() : sn.pts->geometry();
                    if (geom.get() != sn.geom ||
                        (sn.geomB && sn.geomB->attributesVersion() != sn.attrVer)) ok = false;
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
                if (wire != ((sn.flags & kSnapWire) != 0u) ||
                    over != ((sn.flags & kSnapOverlay) != 0u) ||
                    tet != ((sn.flags & kSnapTet) != 0u)) {
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
        std::array<Buffer, kFramesInFlight> prevCameraUbos{};
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
        // True if any material in the current scene has transmission > 0.
        // Gates the 27-cell caustic gather in closest_hit. Photon emit also
        // requires glassVisibleThisFrame_ below — emitting 512×512 photon
        // paths per frame is wasted work when no glass is in the camera
        // frustum (no caustic gather can read fresh photons anyway). Gather
        // stays gated on sceneHasGlass_ only so caustics persist visually
        // for a few frames after the camera turns away from glass (read
        // from the world-space photon grid until it's overwritten by the
        // next emit pass).
        bool     sceneHasGlass_ = false;
        // Scene-feature flags fed into the chit's kSceneFeatures spec constant
        // at first-frame pipeline build (see currentSceneFeatures()). When a
        // feature is absent in the scene's materials, the corresponding chit
        // branch (iridescence Fresnel mix / clearcoat lobe / sheen NEE) is
        // DCEd out of the SPV entirely. Detected once per matDescs rebuild.
        bool     sceneHasClearcoat_ = false;
        bool     sceneHasIridescence_ = false;
        bool     sceneHasSheen_ = false;
        // Per-frame frustum-cull result: any glass-flagged entry visible?
        // Resolved by cullEntriesAgainstFrustum() as a side effect of the
        // pass that tags every entry's MeshEntry::inFrustum.
        bool     glassVisibleThisFrame_ = false;
        // (photon-count initialization flag is now owned by photon_;
        // access via photon_->isInitialized() / markUninitialized())
        // Ascending indices into lastVisibleEntries_ for transmissive-
        // material entries. Maintained alongside materialDescs builds
        // (full rebuild + the materialValuesSame=false hot path); a
        // matrix-only frame keeps the existing list valid since glass
        // identity is per-entry, not per-xfm. Used by
        // cullEntriesAgainstFrustum to test only the (small) subset of
        // visible-test results for glass membership when deciding whether
        // photon emit should run.
        std::vector<size_t> glassEntryIndices_;
        // Per-NEE firefly clamp; pushed to shaders as float bits in slot [11].
        // 1e30f sentinel disables the clamp (set via setFireflyClamp(0)).
        float    fireflyClamp_ = 30.f;
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

        // Ray-tracing pipeline.
        VkDescriptorSetLayout rtDsLayout = VK_NULL_HANDLE;
        VkPipelineLayout      rtPipelineLayout = VK_NULL_HANDLE;

        // RT pipeline variants. Cross product of (SER on/off) × (restirDI
        // chit spec constant). Up to 4 variants; SER ones only built on
        // supported devices. All variants are built at first-frame pipeline
        // build — vkCreateRayTracingPipelinesKHR hangs on a second call
        // against the same pipeline layout on NVIDIA drivers, so we eat the
        // per-variant one-time cost instead of rebuilding on toggle.
        // setRestirDIEnabled is zero-cost (just changes which slot
        // activeRtVariant picks).
        //
        // Index layout: variantIndex(useSer, restirDI) =
        //     (useSer ? 2 : 0) + (restirDI ? 1 : 0)
        struct RtPipelineVariant {
            VkPipeline pipeline = VK_NULL_HANDLE;
            Buffer     sbtBuffer{};
            VkStridedDeviceAddressRegionKHR rgenRegion{};
            VkStridedDeviceAddressRegionKHR missRegion{};
            VkStridedDeviceAddressRegionKHR hitRegion{};
        };
        std::array<RtPipelineVariant, 4> rtVariants_{};
        static uint32_t rtVariantIndex(bool useSer, bool restirDI) {
            return (useSer ? 2u : 0u) + (restirDI ? 1u : 0u);
        }
        // Convenience accessors — read which variant is active from
        // shouldUseSerRaygen() (which folds restirGIEnabled_ + driver
        // support into a single bool).
        const RtPipelineVariant& activeRtVariant() const {
            return rtVariants_[rtVariantIndex(shouldUseSerRaygen(), restirDIEnabled_)];
        }
        RtPipelineVariant& activeRtVariant() {
            return rtVariants_[rtVariantIndex(shouldUseSerRaygen(), restirDIEnabled_)];
        }
        VkStridedDeviceAddressRegionKHR callRegion{};// unused (no callable shaders)

        // Photon caustics subsystem (pipeline + SBT + count/data buffers).
        // Encapsulated as a separate module — see vulkan/PhotonCaustics.{hpp,cpp}.
        // Owns its state; this Impl just holds the unique_ptr and routes
        // descriptor binding + dispatch through the class.
        std::unique_ptr<vulkan::PhotonCaustics> photon_;

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
        // every pass downstream of the raster G-buffer prepass:
        // photon emit, PT raygen, denoise, TAA, hybrid overlay, upscale.
        // The swapchain is cleared to black so the sprite overlay (event
        // accumulator) has a clean canvas; event_shade + event_detect
        // run in the outer beginFrameAndRecord flow exactly as in the
        // normal PT path. Designed for high-rate event-camera sampling
        // (~500 Hz target) where the renderer's visual output isn't
        // displayed at all — only the event accumulator readout matters.
        // Requires hybrid OR TAA to be on so the gbuf prepass actually
        // runs (event_shade reads its normal + ids attachments).
        bool eventsOnlyMode_ = false;

        // Event-camera shade pipeline. Runs immediately after the raster
        // G-buffer pass (when event camera is on) and writes a clean
        // deterministic luma buffer that the detector consumes instead
        // of the PT-noisy swapchain copy. Allocated lazily alongside
        // the detector; descriptor bindings refresh per-frame to track
        // the current frame's gbuf views + material/light buffers.
        vulkan::Buffer        eventLumaBuf_{};
        VkDescriptorSetLayout eventShadeDsLayout_       = VK_NULL_HANDLE;
        VkPipelineLayout      eventShadePipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            eventShadePipeline_       = VK_NULL_HANDLE;
        VkDescriptorPool      eventShadeDescPool_       = VK_NULL_HANDLE;
        VkDescriptorSet       eventShadeDescSet_        = VK_NULL_HANDLE;
        uint32_t              eventLumaW_ = 0;
        uint32_t              eventLumaH_ = 0;

        // User-requested sensor resolution. 0 means "track swapchain";
        // any non-zero pair pins the detector + luma buffer at that res
        // and the event_shade pass nearest-samples the (typically larger)
        // gbuf into it. Real DVS sensors live in the 128² – 640×480 range,
        // far below the swapchain — running the detector at sensor-native
        // res cuts the per-frame work by the squared scale factor.
        uint32_t eventCamUserW_ = 0;
        uint32_t eventCamUserH_ = 0;

        // Spatial denoiser — see vulkan/Denoiser.{hpp,cpp}. Owns the atrous
        // + finalize compute pipelines and the filtered + moments ping-pong
        // images. Shares rtDsLayout so a single per-frame descriptor set
        // drives RT raygen, atrous, and finalize.
        std::unique_ptr<vulkan::Denoiser> denoiser_;
        // THREEPP_DENOISE=0 disables denoising from the environment — an A/B
        // discriminator for "is this artifact shading or temporal/denoise?"
        // without plumbing a flag through every example.
        bool denoiseEnabled_ = []() {
            const char* e = std::getenv("THREEPP_DENOISE");
            return !(e && e[0] == '0');
        }();

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
        // Replaces PT primary-ray traversal: raster writes depth/normal/motion/IDs
        // into per-frame attachments; raygen reads them and starts at bounce 1.
        // Eliminates moving-object shake from PT-primary jitter and makes
        // primary visibility deterministic per pixel. AA happens via TAA on
        // top of raster, not as Monte Carlo on the PT primary. Disabled by
        // default until the integration is validated end-to-end (stage 1).
        struct RasterGbufImages {
            Image2D       normal;       // rgba16f — world-space normal in xyz, .w = linear roughness
            Image2D       motion;       // rgba16f — NDC delta in .rg, .ba reserved
            Image2D       ids;          // rgba16ui — instanceCustomIndex/meshID/flags/reserved
            Image2D       uv;           // rgba16f — material UV in .rg
            Image2D       albedo;       // rgba8 unorm — linear base colour in .rgb, metalness in .a (raster-first deferred input)
            Image2D       indirect;     // rgba16f — demodulated diffuse-indirect irradiance (deferred denoiser scratch; STORAGE, not an attachment)
            Image2D       momentsSq;    // r16f — temporally-accumulated E[L²] of the indirect luminance (SVGF variance: var = E[L²] - lum(indirect)²); STORAGE+SAMPLED, ping-ponged like indirect
            Image2D       atrousA;      // rgba16f — SVGF multi-pass à-trous ping-pong (rgb=GI, a=variance); STORAGE scratch
            Image2D       atrousB;      // rgba16f — SVGF multi-pass à-trous ping-pong (the other half)
            Image2D       reflect;      // rgba16f — sharp 1-mirror-ray reflection radiance (.rgb), demodulated; roughness-blurred by the reflection denoise. STORAGE
            Image2D       reflAux;      // rgba16f — reflection-denoiser auxiliary (ping-pong, mirrors `reflect`: STORAGE write + SAMPLED prev-frame read)
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
        };
        std::array<RasterGbufImages, kFramesInFlight> rasterGbufs{};
        VkRenderPass rasterGbufRenderPass = VK_NULL_HANDLE;

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
        // occluded by path-traced geometry. No descriptor sets — the only
        // input is a push constant (mvp + color).
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
        // Lines never path-trace; they're collected separately from
        // Mesh entries and walked in the overlay record loop.
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

        // Per-Line scene snapshot, refreshed in ensureSceneBuilt alongside
        // lastVisibleEntries_. Lives only for the overlay record's draw
        // loop — neither PT nor the raster G-buffer touches this.
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

        // Per-frame raster camera data. currVPjittered drives gl_Position;
        // currVPunjittered + prevVP drive the motion-vector computation
        // (which must be jitter-free or motion vectors include the jitter
        // and pollute reproject).
        struct RasterCameraData {
            float currVPjittered[16];
            float currVPunjittered[16];
            float prevVP[16];
            float jitter[4];          // .xy = clip-space sub-texel offset, .zw = 1/resolution
            float prevJitter[4];      // .xy = previous frame's jitter; gbuffer.frag adds
                                      // (prev - curr) to the unjittered motion vec so TAA
                                      // reproject lands on the prev rasterized pixel rather
                                      // than its unjittered ideal — fixes per-frame wander
                                      // that manifests as shake on moving objects.
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

        // Nearest-filter sampler used by raygen to read gbuffer attachments.
        // Nearest avoids bilinear smearing of normal/motion/ids at silhouettes
        // — primary visibility from raster is already exact-pixel.
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
        // HDR bloom + tone-map/sRGB composite — tail of the PT post stack.
        // denoise.comp (resolve) writes linear HDR into bloom_->sceneHdr
        // (the shared set's binding 1); bloom_ down/blurs it and composites
        // bloom + tone map + sRGB into the TAA input. bloomIntensity_ == 0
        // skips the bloom passes (composite still owns the tone map).
        std::unique_ptr<vulkan::BloomPass> bloom_;
        float bloomIntensity_ = 0.0f;
        // Phase-2 GI path (RasterFirst): ON now activates REAL stochastic 1-bounce
        // GI (colour bleed, no AO/sky hacks) + temporal accumulation + à-trous.
        // OFF falls back to the clean deterministic AO+far≈sky approximation.
        // Default ON to ship the Phase-2 GI. CAVEATS (follow-ups): GI bounce uses
        // full traceRadiance (expensive in emitter-heavy scenes — cheap sun+emissive
        // bounce pending); no reproject/disocclusion yet → MOTION GHOSTS (ping-pong
        // history reproject pending). setDenoise(false) → clean fallback.
        // (One flag for BOTH paths — denoiseEnabled_ — so a single toggle
        // follows the active RenderMode; setDeferredDenoise is an alias.)
        // Ray-traced env ambient occlusion / GI (RasterFirst). ON: gives the
        // "dirty realistic" PT-like grounding (contact darkening + 1-bounce GI).
        // Uses the deterministic Fibonacci 64-sample gather (clean + settles), so
        // it's the realistic look WITHOUT the old per-frame flicker.
        bool  deferredAO_ = true;
        // Deferred volumetric spot-light beams (ray-marched single scattering in
        // deferred_shade.comp). σ = 0 disables (the march is skipped entirely).
        float deferredVolDensity_ = 0.f;
        float deferredVolAniso_   = 0.55f;
        // Procedural direction-space star field on deferred sky pixels (0 = off).
        float deferredStarIntensity_ = 0.f;

        // Raster-first deferred lighting (RenderMode::RasterFirst). Shades the
        // material G-buffer analytically into bloom_->sceneHdr, replacing the
        // raygen + denoise stages; bloom + TAA finish the frame unchanged. Owns
        // no images and does not touch rtDsLayout, so ReferencePT is unaffected.
        std::unique_ptr<vulkan::DeferredShade> deferredShade_;
        float bloomThreshold_ = 1.0f;// soft-knee bright-pass cutoff (linear HDR)
        float bloomClamp_ = 0.0f;    // per-tap HDR cap before the bright pass; <= 0 = off
        float sharpenStrength_ = 0.0f;// post-TAA RCAS amount; 0 = off
        float taaBlendAlpha_ = 0.1f;// 10% current, 90% history at the reference rate;
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

        bool perSppJitterHybrid_ = true;
        // Tracks what each per-frame slot's binding 1 (RT denoise output)
        // currently points at, so the per-frame rewrite block only fires on
        // a real state change: -1 = unknown/needs rewrite, 0 = swapchain
        // image view, 1 = taa_->inputView(frame). allocateAndUpdate-
        // Descriptors writes swapchain (sets to 0); swapchain recreation
        // reruns that path. Used to be rewritten unconditionally every frame,
        // burning imageCount_ vkUpdateDescriptorSets calls for nothing.
        std::array<int8_t, kFramesInFlight> binding1Mode_{};
        // Per-frame-slot gate for the raster descriptor's binding 3 — the
        // 2048-entry bindless material-texture array. Its contents are
        // identical every frame and only change when the scene texture table
        // is rebuilt (the same event that rebinds the RT side's binding 8 in
        // rewriteSceneDescriptors / allocateAndUpdateDescriptors). Rewriting
        // all 2048 entries every frame burned a vkUpdateDescriptorSets call +
        // a ~48 KB host array fill for nothing. 1 = current, 0 = needs
        // (re)write; value-inits to 0 so the first frame writes it. Invalidated
        // (->0) at scene (re)build; each slot rewrites on its next
        // uploadRasterCameraUbo, before recordCommandBuffer binds the set.
        // Mirrors binding1Mode_. rasterDescSets live in their own pool
        // (rasterDescPool, init-only) so swapchain / main-pool rebuilds don't
        // affect them — only a texture-table change does.
        std::array<int8_t, kFramesInFlight> rasterMatTexValid_{};

        // ── Lower-resolution path-trace mode ────────────────────────────
        // renderScale_ < 1 runs raygen, denoise, and the raster G-buffer at
        // renderExtent() instead of the swapchain extent. The TAA pass then
        // upsamples to full resolution (it dispatches at the swapchain extent
        // with a full-res history — see taa_resolve.comp); with TAA off a
        // bilinear blit upscales the low-res denoise output instead. 1.0 →
        // renderExtent() equals the swapchain extent and every pass behaves
        // exactly as before.
        float renderScale_ = 1.0f;
        // (renderScale < 1 is upsampled by TAA straight to the swapchain.)

        // PT render extent: swapchain extent × renderScale_, each axis
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
        // True when the PT pipeline renders below the swapchain extent and
        // the upscale blit is therefore required.
        bool ptScaled() const {
            const VkExtent2D r = renderExtent();
            const VkExtent2D s = ctx->swapchainExtent();
            return r.width != s.width || r.height != s.height;
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
        // (same pattern as bounces). Forwarded to chit via
        // pc.motionFlags bit 4 each frame.
        bool restirDIEnabled_ = true;
        // ReSTIR DI visibility reuse (Bitterli 2020 §5). When on, the chit
        // shadow-tests the RIS-selected candidate before temporal/spatial reuse
        // and discards occluded ones, so history converges onto visible lights.
        // One extra shadow ray per RIS pixel. Forwarded via pc.motionFlags bit 8.
        // Default on — it's the quality win that makes DI worthwhile in interiors;
        // only meaningful while restirDIEnabled_ is also set.
        bool restirDIVisibilityReuse_ = true;
        // ReSTIR GI master toggle. Stage 1a: when on, primary chit launches a
        // BSDF-sampled sub-ray to generate a single-sample reservoir for the
        // indirect contribution at primary, then hands off to raygen with the
        // sub-trace's xs as the new origin for bounce 2 (skipping classic
        // bounce 1 to avoid double-counting). At M=1 the contribution is
        // statistically equivalent to classic MC; later stages (1b temporal,
        // 1c spatial) provide the actual variance reduction. Forwarded to
        // chit via pc.motionFlags bit 6 each frame.
        bool restirGIEnabled_ = false;
        // User opt-out for SER (setSerEnabled). Default true; when false the
        // path tracer runs the plain-traceRayEXT fallback raygen even on
        // SER-capable hardware. Folded into shouldUseSerRaygen() below so the
        // flip just selects the already-built fallback variant — no rebuild.
        bool serEnabled_ = true;
        // SER helps the GI-off case but *hurts* the GI-on case because
        // ReSTIR GI Stage 2's recursive sub-trace inside the chit undoes
        // the warp-reorder coherence the raygen reorderThreadNV bought us.
        // Both pipelines are pre-built (see rtVariants_); activeRtVariant()
        // picks each frame based on this gate — which also folds in the
        // user opt-out (serEnabled_) and driver support.
        bool shouldUseSerRaygen() const {
            return serEnabled_ && ctx->rayTracingInvocationReorderSupported()
                   && !restirGIEnabled_;
        }
        // Hybrid raster overlay: layer index for opt-in overlay objects
        // (alongside auto-detected wireframe materials + Line/LineSegments).
        // -1 disables layer-based selection. Mirrors WGPU PT's overlayLayer_.
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
            for (const auto& en : lastVisibleEntries_) {
                if (en.isOverlay) return true;
            }
            return false;
        }
        // Sub-pixel jitter sequence index for raster TAA. Cycles within a
        // small period (16 frames) — long enough to look stable, short enough
        // to avoid ever-growing index drift.
        uint32_t haltonFrame_ = 0;
        // G-buffer debug-view mode: blit one G-buffer channel onto the
        // swapchain instead of running the path tracer. Lets us see the
        // raster output before raygen integration.
        enum class HybridDebugView : uint32_t {
            Off    = 0,
            Normal = 1,
            Motion = 2,
            Depth  = 3,
            Ids    = 4,
            Albedo = 5,   // raster-first material G-buffer: linear albedo in rgb
        };
        HybridDebugView hybridDebugView_ = HybridDebugView::Off;

        // Shading strategy (see VulkanRenderer::RenderMode in the public
        // header). RasterFirst aliases ReferencePT until the deferred base +
        // PT-accent passes land, so this currently only stores the user's
        // preference and does not yet branch the frame graph. Default
        // ReferencePT keeps today's behaviour byte-for-byte.
        VulkanRenderer::RenderMode renderMode_ = VulkanRenderer::RenderMode::RasterFirst;

        // Debug: gate raygen to exit immediately after step-0 primary trace
        // so pathTraceMs measures roughly the primary-trace cost. See
        // VulkanRenderer::setMeasurePrimaryTraceOnly.
        bool measurePrimaryTraceOnly_ = false;

        // Per-frame-in-flight camera UBO (viewInverse + projInverse).
        // 2 mat4 packed back-to-back, std140 layout.
        std::array<Buffer, kFramesInFlight> cameraUbos{};

        // Descriptor pool + sets indexed by [frame * imageCount + image] so
        // each set can reference both a per-frame UBO and a per-image view.
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descriptorSets;
        uint32_t imageCount_ = 0;

        // Per-frame command resources.
        VkCommandPool                                cmdPool = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, kFramesInFlight> cmdBuffers{};
        std::array<VkSemaphore,     kFramesInFlight> imageAvailable{};
        std::array<VkSemaphore,     kFramesInFlight> renderFinished{};
        std::array<VkFence,         kFramesInFlight> inFlight{};

        uint32_t currentFrame = 0;
        bool needsResize = false;

        // ImGui (or any post-PT overlay) hook. When set, the swapchain image
        // is transitioned GENERAL → COLOR_ATTACHMENT_OPTIMAL after the trace,
        // a dynamic render pass is opened with LOAD_OP_LOAD, the callback
        // draws into it, then we transition to PRESENT_SRC.
        std::function<void(void*)> overlayCallback;

        // Per-frame lifecycle state. Mirrors the WgpuRenderer multi-pass /
        // HUD pattern: the first render() of a user animate iteration runs
        // beginFrame() (acquire + per-frame UBO uploads + cmd buffer record),
        // subsequent render() calls in the same iteration append to that
        // same cmd buffer, and the Canvas frame-end callback fires endFrame()
        // (ImGui overlay, swapchain → PRESENT_SRC, submit, vkQueuePresentKHR).
        //
        // Idle              : no frame in flight; next render() runs beginFrame.
        // RecordingPostPT   : PT pass recorded; further render(ortho) calls
        //                     append HUD overlay draws to the same cb.
        // RecordingOrthoOnly: render(ortho) called with no prior PT — frame
        //                     was opened with an empty PT result (cleared
        //                     swapchain). Rare; mostly defensive.
        enum class FrameState { Idle, RecordingPostPT, RecordingOrthoOnly };
        FrameState frameState_ = FrameState::Idle;

        // Internal ortho camera used for the screen-space sprite overlay
        // auto-call from beginFrameForPT. Lazy-created on first use, then
        // its bounds + projection are rebuilt each frame to track the
        // current swapchain extent. Lives here (vs. stack-allocated each
        // frame) so its matrixWorldInverse / projectionMatrix don't get
        // rebuilt from scratch when the size hasn't changed.
        std::shared_ptr<OrthographicCamera> screenSpaceCam_;
        uint32_t   frameImageIndex_ = 0;
        std::chrono::high_resolution_clock::time_point frameStartTp_;

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
        // stash the request and apply it at the next beginFrameForPT —
        // after the prior frame's fence wait, before any record. Setters
        // called when Idle apply immediately as before.
        bool pendingRenderScaleRealloc_ = false;
        bool pendingAccumulationReset_  = false;
        // Set by setRenderMode; refreshes the deferred (RasterFirst) descriptor
        // set at the next GPU-idle point so a runtime mode toggle doesn't leave
        // it stale.
        bool pendingDeferredRewrite_    = false;

        explicit Impl(Canvas& c) : canvas(c), size(c.size()) {
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
            // EnvPrefilter owns the PMREM compute pipeline + descriptor pool.
            // Construct before createDefaultEnvImage so the env upload path is
            // ready if scene.environment is set before the first render().
            envPrefilter_ = std::make_unique<vulkan::EnvPrefilter>(*ctx, cmdPool);
            createDefaultEnvImage();
            rebuildDefaultEnvCdfImages();// 1×1 dummy so descriptors are valid before any HDR upload
            createTextureSampler();
            createDefaultMaterialTexture();
            // Builds both fallback + SER variants (when supported) and their
            // SBTs in one pass.
            createRtPipeline();
            // Denoiser reuses rtDsLayout for its descriptor set (single
            // per-frame set drives raygen + atrous + finalize) and needs
            // cmdPool for one-shot image transitions in createImages. Must
            // construct before createAccumImage since clearGbufImages now
            // includes denoiser_->momentsImage(0/1).
            denoiser_ = std::make_unique<vulkan::Denoiser>(*ctx, rtDsLayout, cmdPool);
            createAccumImage();
            skinning_ = std::make_unique<vulkan::SkinningPipeline>(*ctx);
            tetSkinning_ = std::make_unique<vulkan::TetSkinningPipeline>(*ctx);
            photon_ = std::make_unique<vulkan::PhotonCaustics>(*ctx, rtPipelineLayout);
            waterDisplace_ = std::make_unique<vulkan::WaterDisplacePipeline>(*ctx);
            foamWorld_     = std::make_unique<vulkan::FoamWorldPipeline>(*ctx);
            // Hybrid raster G-buffer infrastructure. Costs a few hundred MB
            // at 1080p for six attachments × kFramesInFlight.
            ensureHybridResources();
            // TAA pipeline + images. The RT descriptor's binding 1 (denoise
            // output target) always points at the TAA input view.
            imageCount_ = static_cast<uint32_t>(ctx->swapchainImages().size());
            taa_ = std::make_unique<vulkan::TaaResolve>(
                    *ctx, cmdPool, imageCount_, kFramesInFlight);
            {
                // TAA input is the path-trace render extent; history +
                // output are the swapchain extent. When they differ the
                // resolve pass runs as a temporal upsampler.
                const VkExtent2D inExt  = renderExtent();
                const VkExtent2D outExt = ctx->swapchainExtent();
                taa_->createImages(inExt.width, inExt.height,
                                   outExt.width, outExt.height);
            }
            // HDR bloom + tone-map/sRGB composite. sceneHdr lives at the
            // path-trace render extent (it is the shared set's binding 1
            // target); the bloom ping-pong buffers are half that.
            bloom_ = std::make_unique<vulkan::BloomPass>(*ctx, cmdPool, kFramesInFlight);
            bloom_->createImages(renderExtent().width, renderExtent().height);
            // Raster-first deferred lighting pass. Writes bloom_->sceneHdr, so
            // it must exist after bloom_; its descriptors reference the camera /
            // lights UBOs, the env image, the raster gbuffer and sceneHdr — all
            // created above by this point.
            // The deferred base traces ray-query shadow rays, so only stand it
            // up when the device supports VK_KHR_ray_query. Without it,
            // deferredShade_ stays null and RasterFirst falls back to ReferencePT.
            if (ctx->rayQuerySupported()) {
                deferredShade_ = std::make_unique<vulkan::DeferredShade>(*ctx, kFramesInFlight);
            }
            createDescriptorPool();
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
                    [this](uint32_t w, uint32_t h, VkFormat fmt,
                           const void* pix, VkDeviceSize sz,
                           VkFilter filter,
                           VkSamplerAddressMode addrU,
                           VkSamplerAddressMode addrV,
                           const char* name) {
                        return createSampledImage2D(w, h, fmt, pix, sz,
                                                   filter, addrU, addrV, name);
                    });
        }

        ~Impl() {
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

            for (auto s : imageAvailable) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto s : renderFinished) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto f : inFlight) if (f) vkDestroyFence(d, f, nullptr);
            if (cmdPool) vkDestroyCommandPool(d, cmdPool, nullptr);
            gpuTimings_.reset();// query pool destruction while device is still valid

            if (descriptorPool) vkDestroyDescriptorPool(d, descriptorPool, nullptr);

            for (auto& v : rtVariants_) {
                destroyBuffer(ctx->allocator(), v.sbtBuffer);
                if (v.pipeline) vkDestroyPipeline(d, v.pipeline, nullptr);
            }
            // Photon caustics owns its pipeline + SBT + buffers; reset
            // before destroying the shared rtPipelineLayout it references.
            photon_.reset();
            // Denoiser owns atrous + finalize pipelines + its own pipeline
            // layout (built from rtDsLayout) + filtered/moments images.
            // Reset before destroying rtDsLayout for symmetry with photon_.
            denoiser_.reset();
            if (rtPipelineLayout) vkDestroyPipelineLayout(d, rtPipelineLayout, nullptr);
            if (rtDsLayout)       vkDestroyDescriptorSetLayout(d, rtDsLayout, nullptr);

            if (tlas) ctx->rt().destroyAccelerationStructure(d, tlas, nullptr);
            destroyBuffer(ctx->allocator(), tlasBuffer);
            destroyBuffer(ctx->allocator(), tlasInstancesBuffer);
            destroyBuffer(ctx->allocator(), geometryDescsBuffer);
            for (auto& b : materialDescsBuffers) destroyBuffer(ctx->allocator(), b);
            destroyBuffer(ctx->allocator(), sceneCaptureBuf_);
            destroyBuffer(ctx->allocator(), eventLumaBuf_);
            if (eventShadePipeline_)       vkDestroyPipeline(d, eventShadePipeline_, nullptr);
            if (eventShadePipelineLayout_) vkDestroyPipelineLayout(d, eventShadePipelineLayout_, nullptr);
            if (eventShadeDescPool_)       vkDestroyDescriptorPool(d, eventShadeDescPool_, nullptr);
            if (eventShadeDsLayout_)       vkDestroyDescriptorSetLayout(d, eventShadeDsLayout_, nullptr);

            for (auto& [_, rec] : blasCache) {
                if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                destroyBuffer(ctx->allocator(), rec->storage);
                destroyBuffer(ctx->allocator(), rec->vertex);
                destroyBuffer(ctx->allocator(), rec->index);
                destroyBuffer(ctx->allocator(), rec->normal);
                destroyBuffer(ctx->allocator(), rec->uv);
                destroyBuffer(ctx->allocator(), rec->foam);
                destroyBuffer(ctx->allocator(), rec->prevVertex);
                destroyBuffer(ctx->allocator(), rec->blasScratch);
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
                auto& rec = st->blas;
                if (!rec) continue;
                if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                destroyBuffer(ctx->allocator(), rec->storage);
                destroyBuffer(ctx->allocator(), rec->vertex);
                destroyBuffer(ctx->allocator(), rec->index);
                destroyBuffer(ctx->allocator(), rec->normal);
                destroyBuffer(ctx->allocator(), rec->uv);
                destroyBuffer(ctx->allocator(), rec->foam);
                destroyBuffer(ctx->allocator(), rec->prevVertex);
                destroyBuffer(ctx->allocator(), rec->blasScratch);
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
                auto& rec = st->blas;
                if (!rec) continue;
                if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                destroyBuffer(ctx->allocator(), rec->storage);
                destroyBuffer(ctx->allocator(), rec->vertex);
                destroyBuffer(ctx->allocator(), rec->index);
                destroyBuffer(ctx->allocator(), rec->normal);
                destroyBuffer(ctx->allocator(), rec->uv);
                destroyBuffer(ctx->allocator(), rec->foam);
                destroyBuffer(ctx->allocator(), rec->prevVertex);
                destroyBuffer(ctx->allocator(), rec->blasScratch);
            }
            tetMeshStates.clear();

            for (auto& [_, st] : displacedStates) {
                if (st->blas) {
                    auto& rec = st->blas;
                    if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                    destroyBuffer(ctx->allocator(), rec->storage);
                    destroyBuffer(ctx->allocator(), rec->vertex);
                    destroyBuffer(ctx->allocator(), rec->index);
                    destroyBuffer(ctx->allocator(), rec->normal);
                    destroyBuffer(ctx->allocator(), rec->uv);
                    destroyBuffer(ctx->allocator(), rec->foam);
                    destroyBuffer(ctx->allocator(), rec->prevVertex);
                    destroyBuffer(ctx->allocator(), rec->blasScratch);
                }
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

            for (auto& [_, st] : morphedMeshStates) {
                auto& rec = st->blas;
                if (!rec) continue;
                if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                destroyBuffer(ctx->allocator(), rec->storage);
                destroyBuffer(ctx->allocator(), rec->vertex);
                destroyBuffer(ctx->allocator(), rec->index);
                destroyBuffer(ctx->allocator(), rec->normal);
                destroyBuffer(ctx->allocator(), rec->uv);
                destroyBuffer(ctx->allocator(), rec->foam);
                destroyBuffer(ctx->allocator(), rec->prevVertex);
                destroyBuffer(ctx->allocator(), rec->blasScratch);
            }
            morphedMeshStates.clear();

            for (auto& b : cameraUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : prevCameraUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : lightsUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : fogUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : motionMatBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : meshMovedBitsBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : emissiveTriBuffers) destroyBuffer(ctx->allocator(), b);
            destroyImage2D(ctx->allocator(), d, envImage);
            destroyImage2D(ctx->allocator(), d, envCdfImage);
            destroyImage2D(ctx->allocator(), d, envMargImage);
            destroyImage2D(ctx->allocator(), d, blueNoiseImage);
            destroyImage2D(ctx->allocator(), d, oceanFineHeightDummy);
            destroyImage2D(ctx->allocator(), d, oceanFoamDummy);
            destroyImage2D(ctx->allocator(), d, foamDetailImage);
            for (auto& img : accumImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : gbufImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : reservoirPosImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : reservoirWImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : giResXsImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : giResNsImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : giResLoImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : materialTextures) destroyImage2D(ctx->allocator(), d, img);
            materialTextures.clear();
            if (textureSampler_) vkDestroySampler(d, textureSampler_, nullptr);

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
            overlayPass_.reset();// destroy sprite/line pipelines + caches while device is alive
            if (gbufSampler_)           vkDestroySampler(d, gbufSampler_, nullptr);
            destroyBuffer(ctx->allocator(), dummyUvBuffer_);

            // TAA resolve subsystem owns its pipeline/layout/sampler/images.
            taa_.reset();
        }

        void createCommandResources() {
            VkCommandPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pci.queueFamilyIndex = ctx->queueFamilies().graphics;
            check(vkCreateCommandPool(ctx->device(), &pci, nullptr, &cmdPool),
                  "vkCreateCommandPool");

            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmdPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = kFramesInFlight;
            check(vkAllocateCommandBuffers(ctx->device(), &ai, cmdBuffers.data()),
                  "vkAllocateCommandBuffers");

            VkSemaphoreCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            for (uint32_t i = 0; i < kFramesInFlight; ++i) {
                check(vkCreateSemaphore(ctx->device(), &sci, nullptr, &imageAvailable[i]), "vkCreateSemaphore A");
                check(vkCreateSemaphore(ctx->device(), &sci, nullptr, &renderFinished[i]), "vkCreateSemaphore B");
                check(vkCreateFence(ctx->device(), &fci, nullptr, &inFlight[i]), "vkCreateFence");
            }
        }

        // Allocate, begin, return a one-shot command buffer.
        VkCommandBuffer beginOneShot() {
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmdPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(ctx->device(), &ai, &cb), "alloc one-shot cb");

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb");
            return cb;
        }

        // The optional `label` is folded into the error message when the
        // submit / wait throws — handy for distinguishing which one-shot site
        // hit a device-lost during runtime debugging.
        void endAndSubmitOneShot(VkCommandBuffer cb, const char* label = "one-shot") {
            check(vkEndCommandBuffer(cb), "end one-shot cb");
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            const VkResult sr = vkQueueSubmit(ctx->graphicsQueue(), 1, &si, VK_NULL_HANDLE);
            if (sr != VK_SUCCESS) {
                check(sr, (std::string("submit one-shot (") + label + ")").c_str());
            }
            const VkResult wr = vkQueueWaitIdle(ctx->graphicsQueue());
            if (wr != VK_SUCCESS) {
                check(wr, (std::string("wait one-shot (") + label + ")").c_str());
            }
            vkFreeCommandBuffers(ctx->device(), cmdPool, 1, &cb);
        }

        static unsigned int geomVersionOf(const BufferGeometry& g) {
            unsigned int v = 0;
            if (auto* a = g.getAttribute<float>("position")) v += a->version;
            if (auto* a = g.getAttribute<float>("normal"))   v += a->version;
            if (auto* idx = g.getIndex())                     v += idx->version;
            if (auto* a = g.getAttribute<float>("uv"))        v += a->version;
            return v;
        }

        // Build a single BLAS for the given geometry. Vertex / index buffers
        // are uploaded host-mapped, then the AS is built into freshly allocated
        // device storage. The temporary scratch buffer is destroyed on exit.
        std::unique_ptr<BlasRecord> buildBlasFor(const BufferGeometry& geom) {
            auto* posAttr = geom.getAttribute<float>("position");
            if (!posAttr) return nullptr;
            auto* normAttr = geom.getAttribute<float>("normal");
            if (!normAttr) return nullptr;// the RT path requires per-vertex normals
            const auto& positions = posAttr->array();
            const auto& normals = normAttr->array();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            if (vertexCount < 3) return nullptr;
            if (normAttr->count() != static_cast<int>(vertexCount)) return nullptr;

            const auto* idxAttr = geom.getIndex();
            const bool indexed = idxAttr != nullptr;
            const uint32_t primitiveCount = indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;
            if (primitiveCount == 0) return nullptr;

            // BLAS build is undefined behavior on NaN/Inf positions or
            // out-of-bounds indices — drivers respond with device-lost
            // (vkQueueWaitIdle returns VK_ERROR_DEVICE_LOST). Validate
            // before submission so a bad asset (typical of Assimp-loaded
            // URDF .dae meshes with degenerate triangles) is skipped with
            // a warning rather than killing the whole renderer.
            for (size_t i = 0; i < positions.size(); ++i) {
                if (!std::isfinite(positions[i])) {
                    std::cerr << "[VulkanRenderer] buildBlasFor: skipping geometry — "
                              << "position[" << i << "] is non-finite ("
                              << positions[i] << "), vertexCount=" << vertexCount << '\n';
                    return nullptr;
                }
            }
            if (indexed) {
                const auto& indices = idxAttr->array();
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (indices[i] >= vertexCount) {
                        std::cerr << "[VulkanRenderer] buildBlasFor: skipping geometry — "
                                  << "index[" << i << "]=" << indices[i]
                                  << " >= vertexCount=" << vertexCount << '\n';
                        return nullptr;
                    }
                }
            }

            // Hybrid: raster G-buffer pre-pass binds these same allocations
            // directly as vertex / index buffers — no duplication, no extra
            // upload, and the raster prepass + RT shadow rays warm the same
            // cache lines. TRANSFER_SRC_BIT for displaced meshes that need
            // to vkCmdCopyBuffer the current vertex into prev each frame.
            const VkBufferUsageFlags geomUsage =
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

            auto rec = std::make_unique<BlasRecord>();

            const VkDeviceSize vbBytes = positions.size() * sizeof(float);
            rec->vertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    geomUsage, VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), rec->vertex.alloc, &mapped);
            std::memcpy(mapped, positions.data(), vbBytes);
            vmaUnmapMemory(ctx->allocator(), rec->vertex.alloc);

            const VkDeviceSize nbBytes = normals.size() * sizeof(float);
            rec->normal = createBuffer(
                    ctx->allocator(), ctx->device(), nbBytes,
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            vmaMapMemory(ctx->allocator(), rec->normal.alloc, &mapped);
            std::memcpy(mapped, normals.data(), nbBytes);
            vmaUnmapMemory(ctx->allocator(), rec->normal.alloc);

            // Optional UV attribute (TEXCOORD_0). closest_hit interpolates and
            // samples albedo with these; absent → bindless texture is ignored.
            if (auto* uvAttr = geom.getAttribute<float>("uv")) {
                const auto& uvs = uvAttr->array();
                if (uvAttr->count() == static_cast<int>(vertexCount) &&
                    uvs.size() == vertexCount * 2) {
                    const VkDeviceSize uvBytes = uvs.size() * sizeof(float);
                    rec->uv = createBuffer(
                            ctx->allocator(), ctx->device(), uvBytes,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    vmaMapMemory(ctx->allocator(), rec->uv.alloc, &mapped);
                    std::memcpy(mapped, uvs.data(), uvBytes);
                    vmaUnmapMemory(ctx->allocator(), rec->uv.alloc);
                }
            }

            if (indexed) {
                const auto& indices = idxAttr->array();
                const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                rec->index = createBuffer(
                        ctx->allocator(), ctx->device(), ibBytes,
                        geomUsage, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                vmaMapMemory(ctx->allocator(), rec->index.alloc, &mapped);
                std::memcpy(mapped, indices.data(), ibBytes);
                vmaUnmapMemory(ctx->allocator(), rec->index.alloc);
            }

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = rec->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vertexCount - 1;
            if (indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = rec->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            // No OPAQUE flag — that would suppress any-hit invocation per
            // geometry regardless of the ray flags, breaking alpha-test. The
            // any-hit shader short-circuits cheaply for materials with
            // alphaCutoff <= 0, so the cost on truly opaque meshes is one
            // buffer read + compare per hit candidate.
            blasGeom.flags = 0;

            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            // ALLOW_UPDATE so refitTlas's MODE_UPDATE on the TLAS still
            // resolves to a valid build chain even if a future BLAS-level
            // refit lands. PREFER_FAST_BUILD is intentionally not set —
            // BLAS is still built once per geometry and queried-many.
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries = &blasGeom;

            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &primitiveCount, &blasSizes);

            rec->storage = createBuffer(
                    ctx->allocator(), ctx->device(), blasSizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkAccelerationStructureCreateInfoKHR blasCreate{};
            blasCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            blasCreate.buffer = rec->storage.handle;
            blasCreate.size = blasSizes.accelerationStructureSize;
            blasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            check(ctx->rt().createAccelerationStructure(ctx->device(), &blasCreate, nullptr, &rec->as),
                  "vkCreateAccelerationStructureKHR(BLAS)");

            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);

            blasBuild.dstAccelerationStructure = rec->as;
            blasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            // Per-vertex previous-pose buffer for the chit's per-vertex motion
            // vector path. Skinned / displaced / morphed meshes own their own
            // prev buffer in their state struct; plain meshes that go through
            // refreshGeomBlas (PhysX soft bodies, ParticleSystems updating
            // vertices in place) need one here too — without it, the chit reads
            // gdesc.prevVertexAddress == vertexAddress, motion is zero, and
            // freshly-spawned dynamic bodies look noisy because the temporal
            // accumulator blends mismatched history. Static meshes get a free
            // copy (prev == current forever) — chit motion vector resolves to
            // zero, matching the previous fallback behavior.
            rec->prevVertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkCommandBuffer cb = beginOneShot();
            // Seed prevVertex with the initial (current) vertex data — first
            // refresh would otherwise copy garbage into prev. Fold into the
            // same one-shot as the BLAS build to avoid an extra submit.
            VkBufferCopy seedCopy{};
            seedCopy.size = vbBytes;
            vkCmdCopyBuffer(cb, rec->vertex.handle, rec->prevVertex.handle, 1, &seedCopy);
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
            endAndSubmitOneShot(cb, "buildBlasFor");
            destroyBuffer(ctx->allocator(), scratch);

            VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addrInfo.accelerationStructure = rec->as;
            rec->address = ctx->rt().getAccelerationStructureDeviceAddress(ctx->device(), &addrInfo);

            rec->geomVersion = geomVersionOf(geom);
            rec->vertexCount = vertexCount;
            rec->indexCount  = indexed ? static_cast<uint32_t>(idxAttr->count()) : 0u;

            return rec;
        }

        // CPU linear-blend skin → outPos/outNorm in the SkinnedMesh's local
        // frame (bindMatrixInverse applied at the end). Mirrors the WGPU PT
        // path tracer's geometry-builder skinning at WgpuPathTracerGeometry.cpp.
        // Caller is responsible for sizing prevBoneMats and copying skel.boneMatrices
        // afterwards for next-frame dirty detection.
        static void cpuSkin(SkinnedMesh& sm,
                            std::vector<float>& outPos,
                            std::vector<float>& outNorm) {
            auto* posAttr = sm.geometry()->getAttribute<float>("position");
            auto* nrmAttr = sm.geometry()->getAttribute<float>("normal");
            auto* skinIdxAttr = sm.geometry()->getAttribute<float>("skinIndex");
            auto* skinWAttr   = sm.geometry()->getAttribute<float>("skinWeight");
            if (!posAttr || !nrmAttr || !skinIdxAttr || !skinWAttr || !sm.skeleton) return;

            const int vtxCount = static_cast<int>(posAttr->count());
            outPos.assign(vtxCount * 3, 0.f);
            outNorm.assign(vtxCount * 3, 0.f);

            const auto& skel = *sm.skeleton;
            const Matrix4& bindMat    = sm.bindMatrix;
            const Matrix4& bindMatInv = sm.bindMatrixInverse;
            const int boneCount = static_cast<int>(skel.bones.size());

            std::vector<Matrix4> boneMats(boneCount);
            for (int b = 0; b < boneCount; ++b) {
                if (skel.bones[b]) {
                    boneMats[b].multiplyMatrices(*skel.bones[b]->matrixWorld, skel.boneInverses[b]);
                }
            }

            Vector4 sIdx, sW;
            for (int i = 0; i < vtxCount; ++i) {
                skinIdxAttr->setFromBufferAttribute(sIdx, i);
                skinWAttr->setFromBufferAttribute(sW, i);

                Vector3 bindPos(posAttr->getX(i), posAttr->getY(i), posAttr->getZ(i));
                bindPos.applyMatrix4(bindMat);

                Vector3 bindNrm(nrmAttr->getX(i), nrmAttr->getY(i), nrmAttr->getZ(i));
                {
                    const auto& e = bindMat.elements;
                    Vector3 t;
                    t.x = e[0] * bindNrm.x + e[4] * bindNrm.y + e[8]  * bindNrm.z;
                    t.y = e[1] * bindNrm.x + e[5] * bindNrm.y + e[9]  * bindNrm.z;
                    t.z = e[2] * bindNrm.x + e[6] * bindNrm.y + e[10] * bindNrm.z;
                    bindNrm = t;
                }

                Vector3 accPos(0.f, 0.f, 0.f);
                Vector3 accNrm(0.f, 0.f, 0.f);
                for (unsigned b = 0; b < 4; ++b) {
                    const float w = sW[b];
                    if (w == 0.f) continue;
                    const int bIdx = static_cast<int>(sIdx[b]);
                    if (bIdx < 0 || bIdx >= boneCount) continue;
                    const Matrix4& m = boneMats[bIdx];

                    Vector3 bp = bindPos;
                    bp.applyMatrix4(m);
                    accPos.addScaledVector(bp, w);

                    const auto& e = m.elements;
                    Vector3 bn;
                    bn.x = e[0] * bindNrm.x + e[4] * bindNrm.y + e[8]  * bindNrm.z;
                    bn.y = e[1] * bindNrm.x + e[5] * bindNrm.y + e[9]  * bindNrm.z;
                    bn.z = e[2] * bindNrm.x + e[6] * bindNrm.y + e[10] * bindNrm.z;
                    accNrm.addScaledVector(bn, w);
                }

                accPos.applyMatrix4(bindMatInv);
                outPos[i * 3 + 0] = accPos.x;
                outPos[i * 3 + 1] = accPos.y;
                outPos[i * 3 + 2] = accPos.z;

                {
                    const auto& e = bindMatInv.elements;
                    Vector3 t;
                    t.x = e[0] * accNrm.x + e[4] * accNrm.y + e[8]  * accNrm.z;
                    t.y = e[1] * accNrm.x + e[5] * accNrm.y + e[9]  * accNrm.z;
                    t.z = e[2] * accNrm.x + e[6] * accNrm.y + e[10] * accNrm.z;
                    const float len = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
                    const float inv = (len > 0.f) ? (1.f / len) : 0.f;
                    outNorm[i * 3 + 0] = t.x * inv;
                    outNorm[i * 3 + 1] = t.y * inv;
                    outNorm[i * 3 + 2] = t.z * inv;
                }
            }
        }

        // Allocate or look up the per-SkinnedMesh BLAS state. Builds the BLAS
        // once with the current pose; subsequent dirty frames go through
        // refreshSkinnedBlas which only re-skins + rebuilds in-place. Returns
        // null if the geometry is unsupported (no position/normal/skin attrs).
        SkinnedMeshState* ensureSkinnedBlas(SkinnedMesh& sm) {
            auto it = skinnedMeshStates.find(&sm);
            if (it != skinnedMeshStates.end()) return it->second.get();

            auto* posAttr     = sm.geometry()->getAttribute<float>("position");
            auto* nrmAttr     = sm.geometry()->getAttribute<float>("normal");
            auto* skinIdxAttr = sm.geometry()->getAttribute<float>("skinIndex");
            auto* skinWAttr   = sm.geometry()->getAttribute<float>("skinWeight");
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
            //     chit reads via gdesc.prevVertexAddress, interpolates, and
            //     writes payload.prevWorldPos which raygen consumes for the
            //     motionMat reproject. TRANSFER_DST_BIT lets the per-frame
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
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), dst.alloc, &mapped);
                std::memcpy(mapped, src, bytes);
                vmaUnmapMemory(ctx->allocator(), dst.alloc);
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
        void refreshSkinnedBlas(SkinnedMesh& sm, SkinnedMeshState& st) {
            if (!st.blas || !sm.skeleton || st.boneCount == 0) return;

            // Recompute per-bone matrix = bones[b]->matrixWorld * boneInverse[b].
            // Mirrors what cpuSkin used to do on host; cheap (a few dozen
            // matrix multiplies). Upload into the bones[..] section of the
            // host-visible boneMatrices buffer (skipping the [bindMat, bindInv]
            // prefix written once in ensureSkinnedBlas).
            const auto& skel = *sm.skeleton;
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), st.boneMatrices.alloc, &mapped);
            // mats[1] = bindMatrixInverse is NOT constant in attached bind mode:
            // SkinnedMesh::updateMatrixWorld recomputes it to matrixWorld^-1 every
            // frame so that (TLAS instance transform · bindMatrixInverse) collapses
            // to a single world application. ensureSkinnedBlas wrote it once at the
            // bind pose (≈identity at the origin), so a *moving* skinned mesh gets
            // its world transform applied twice — invisible at the origin, wrong
            // once it translates/rotates. Re-upload it every frame.
            std::memcpy(static_cast<char*>(mapped) + 16 * sizeof(float),
                        sm.bindMatrixInverse.elements.data(), 16 * sizeof(float));
            char* dst = static_cast<char*>(mapped) + 32 * sizeof(float);
            for (uint32_t b = 0; b < st.boneCount; ++b) {
                Matrix4 m;
                if (b < skel.bones.size() && skel.bones[b]) {
                    m.multiplyMatrices(*skel.bones[b]->matrixWorld, skel.boneInverses[b]);
                }
                std::memcpy(dst + b * 16 * sizeof(float),
                            m.elements.data(), 16 * sizeof(float));
            }
            vmaUnmapMemory(ctx->allocator(), st.boneMatrices.alloc);

            // Cache the canonical bone matrices for next-frame dirty detection
            // (still uses Skeleton::boneMatrices since that's the host-visible
            // change signal the dirty path reads).
            const auto& bm = skel.boneMatrices;
            if (st.prevBoneMats.size() == bm.size()) {
                std::memcpy(st.prevBoneMats.data(), bm.data(), bm.size() * sizeof(float));
            } else {
                st.prevBoneMats = bm;
            }

            // Queue for per-frame skinning compute + BLAS rebuild. The actual
            // GPU work is recorded into the main command buffer at the start
            // of recordCommandBuffer — no blocking submit here.
            pendingSkinnedRebuilds_.push_back(&st);
        }

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
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), dst.alloc, &mapped);
                std::memcpy(mapped, src, bytes);
                vmaUnmapMemory(ctx->allocator(), dst.alloc);
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
        void refreshTetBlas(Mesh& m, TetMeshState& st) {
            if (!st.blas) return;
            // Zero-copy interop: the registered CUDA device→device copy writes the
            // deformed tet positions straight into the exported binding-6 buffer —
            // no host readback, no DataTexture, no map/memcpy. Runs at the SAME
            // frame point as the CPU upload below, so the (pre-existing, benign)
            // overlap with a still-in-flight prior frame's dispatch is unchanged.
            if (st.tetPosExt.handle != VK_NULL_HANDLE) {
                if (st.tetPosExternalCopy) st.tetPosExternalCopy();
                pendingTetRebuilds_.push_back(&st);
                return;
            }
            auto mat = m.material();
            if (!mat || !mat->tetTexture) return;
            const auto& tetImg = mat->tetTexture->image().data<float>();
            const VkDeviceSize bytes = std::min<VkDeviceSize>(
                    st.tetPosBytes, static_cast<VkDeviceSize>(tetImg.size()) * sizeof(float));
            if (bytes == 0) return;
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), st.tetPos.alloc, &mapped);
            std::memcpy(mapped, tetImg.data(), bytes);
            vmaUnmapMemory(ctx->allocator(), st.tetPos.alloc);
            pendingTetRebuilds_.push_back(&st);
        }

        // Rewrite binding 6 (tetPos — see tet_skinning.comp) of a tet mesh's
        // descriptor set to point at `buf`. Used by the interop enable/disable
        // swap; the other 8 bindings are untouched.
        void rewriteTetPosBinding(TetMeshState& st, VkBuffer buf) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = buf;
            bi.offset = 0;
            bi.range  = VK_WHOLE_SIZE;
            VkWriteDescriptorSet wr{};
            wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr.dstSet          = st.tetDescSet;
            wr.dstBinding      = 6;
            wr.descriptorCount = 1;
            wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr.pBufferInfo     = &bi;
            vkUpdateDescriptorSets(ctx->device(), 1, &wr, 0, nullptr);
        }

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
        void disableSoftBodyInterop(const Mesh& mesh) {
            auto it = tetMeshStates.find(&mesh);
            if (it == tetMeshStates.end()) return;
            auto& st = *it->second;
            if (st.tetPosExt.handle == VK_NULL_HANDLE) return;
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (softbody interop disable)");
            st.tetPos = createBuffer(
                    ctx->allocator(), ctx->device(), st.tetPosBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            rewriteTetPosBinding(st, st.tetPos.handle);
            vulkan::destroyExternalBuffer(ctx->device(), st.tetPosExt);
            st.tetPosExternalCopy = nullptr;
        }

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
        void refreshGeomBlasBatch(const std::vector<GeomRefreshOp>& ops) {
            if (ops.empty()) return;

            // Phase A — validate positions are finite. Non-finite values cause
            // VK_ERROR_DEVICE_LOST during BLAS build on NVIDIA. Skip individual
            // bad ops rather than throwing the whole batch out.
            std::vector<size_t> liveOps;
            liveOps.reserve(ops.size());
            for (size_t k = 0; k < ops.size(); ++k) {
                auto* posAttr = ops[k].geom->getAttribute<float>("position");
                if (!posAttr) continue;
                if (!ops[k].geom->getAttribute<float>("normal")) continue;
                bool ok = true;
                const auto& p = posAttr->array();
                for (size_t i = 0; i < p.size(); ++i) {
                    if (!std::isfinite(p[i])) {
                        std::cerr << "[VulkanRenderer] refreshGeomBlasBatch: skipping geom — "
                                  << "position[" << i << "] is non-finite (" << p[i] << ")\n";
                        ok = false;
                        break;
                    }
                }
                if (ok) liveOps.push_back(k);
            }
            if (liveOps.empty()) return;

            // Phase B — snapshot current vertex into prevVertex for the chit's
            // per-vertex motion vector. Recorded into one shared cmdbuf; submit
            // + wait once. Must run before the host memcpy of new positions
            // below or we'd snapshot the new state, not last frame's.
            {
                bool any = false;
                VkCommandBuffer cb = beginOneShot();
                for (size_t k : liveOps) {
                    auto& rec = *ops[k].rec;
                    if (rec.prevVertex.handle == VK_NULL_HANDLE) continue;
                    auto* posAttr = ops[k].geom->getAttribute<float>("position");
                    VkBufferCopy region{};
                    region.size = posAttr->array().size() * sizeof(float);
                    vkCmdCopyBuffer(cb, rec.vertex.handle, rec.prevVertex.handle, 1, &region);
                    any = true;
                }
                if (any) {
                    endAndSubmitOneShot(cb, "refreshGeomBlasBatch (prevVertex)");
                } else {
                    // No prev buffers attached — skip the empty submit so we
                    // don't round-trip the queue for zero work.
                    vkEndCommandBuffer(cb);
                    vkFreeCommandBuffers(ctx->device(), cmdPool, 1, &cb);
                }
            }

            // Phase C — host writes into the now-snapshotted vertex/normal/uv/
            // index buffers. Each is host-coherent (HOST_ACCESS_SEQUENTIAL_WRITE
            // at buildBlasFor time); the implicit submit barrier on the next
            // phase makes these visible to the BLAS build.
            for (size_t k : liveOps) {
                const auto& geom = *ops[k].geom;
                auto& rec = *ops[k].rec;
                auto* posAttr = geom.getAttribute<float>("position");
                auto* nrmAttr = geom.getAttribute<float>("normal");

                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), rec.vertex.alloc, &mapped);
                std::memcpy(mapped, posAttr->array().data(),
                            posAttr->array().size() * sizeof(float));
                vmaUnmapMemory(ctx->allocator(), rec.vertex.alloc);

                vmaMapMemory(ctx->allocator(), rec.normal.alloc, &mapped);
                std::memcpy(mapped, nrmAttr->array().data(),
                            nrmAttr->array().size() * sizeof(float));
                vmaUnmapMemory(ctx->allocator(), rec.normal.alloc);

                if (auto* uvAttr = geom.getAttribute<float>("uv");
                    uvAttr && rec.uv.handle != VK_NULL_HANDLE) {
                    vmaMapMemory(ctx->allocator(), rec.uv.alloc, &mapped);
                    std::memcpy(mapped, uvAttr->array().data(),
                                uvAttr->array().size() * sizeof(float));
                    vmaUnmapMemory(ctx->allocator(), rec.uv.alloc);
                }

                if (auto* idxAttr = geom.getIndex();
                    idxAttr && rec.index.handle != VK_NULL_HANDLE) {
                    vmaMapMemory(ctx->allocator(), rec.index.alloc, &mapped);
                    std::memcpy(mapped, idxAttr->array().data(),
                                idxAttr->array().size() * sizeof(unsigned int));
                    vmaUnmapMemory(ctx->allocator(), rec.index.alloc);
                }
            }

            // Phase D — refit each BLAS. The build-info structs need to outlive
            // the cmdBuildAccelerationStructures call until the GPU executes
            // them, so we hold them in vectors that live until after the submit-
            // wait below. Each op uses its own (persistent) scratch buffer so
            // concurrent builds within the cmdbuf don't alias.
            const uint32_t N = static_cast<uint32_t>(liveOps.size());
            std::vector<VkAccelerationStructureGeometryTrianglesDataKHR> triDatas(N);
            std::vector<VkAccelerationStructureGeometryKHR>              blasGeoms(N);
            std::vector<VkAccelerationStructureBuildGeometryInfoKHR>     blasBuilds(N);
            std::vector<VkAccelerationStructureBuildRangeInfoKHR>        ranges(N);
            std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs(N);

            for (uint32_t kk = 0; kk < N; ++kk) {
                const size_t k = liveOps[kk];
                const auto& geom = *ops[k].geom;
                auto& rec = *ops[k].rec;
                auto* posAttr = geom.getAttribute<float>("position");
                const auto* idxAttr = geom.getIndex();
                const bool indexed = idxAttr != nullptr;
                const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
                const uint32_t primitiveCount = indexed
                        ? static_cast<uint32_t>(idxAttr->count() / 3)
                        : vertexCount / 3;

                triDatas[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                triDatas[kk].vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                triDatas[kk].vertexData.deviceAddress = rec.vertex.address;
                triDatas[kk].vertexStride = 3 * sizeof(float);
                triDatas[kk].maxVertex = vertexCount - 1;
                if (indexed) {
                    triDatas[kk].indexType = VK_INDEX_TYPE_UINT32;
                    triDatas[kk].indexData.deviceAddress = rec.index.address;
                } else {
                    triDatas[kk].indexType = VK_INDEX_TYPE_NONE_KHR;
                }

                blasGeoms[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                blasGeoms[kk].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                blasGeoms[kk].geometry.triangles = triDatas[kk];
                blasGeoms[kk].flags = 0;

                // Periodic MODE_BUILD every kBlasFullRebuildInterval frames
                // keeps the BVH balanced under sustained vertex motion (soft
                // body collapse, particle swarms) where pure refits drift the
                // tree away from optimal.
                const bool fullRebuild =
                        rec.blasRefitCounter >= BlasRecord::kBlasFullRebuildInterval;
                rec.blasRefitCounter = fullRebuild ? 0u : (rec.blasRefitCounter + 1u);

                blasBuilds[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                blasBuilds[kk].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                blasBuilds[kk].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                       VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                blasBuilds[kk].mode = fullRebuild
                        ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                        : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                blasBuilds[kk].geometryCount = 1;
                blasBuilds[kk].pGeometries = &blasGeoms[kk];
                blasBuilds[kk].srcAccelerationStructure = fullRebuild ? VK_NULL_HANDLE : rec.as;
                blasBuilds[kk].dstAccelerationStructure = rec.as;

                VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
                blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
                ctx->rt().getAccelerationStructureBuildSizes(
                        ctx->device(),
                        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                        &blasBuilds[kk], &primitiveCount, &blasSizes);

                // Persistent scratch: size to max(buildScratch, updateScratch)
                // on first use (buildScratch is always >= updateScratch per
                // spec), then reuse across frames. Reallocate only if a future
                // topology change grew the required size — current refresh
                // path is fixed-topology (caller gates on counts).
                const VkDeviceSize neededScratch = blasSizes.buildScratchSize;
                if (rec.blasScratch.handle == VK_NULL_HANDLE || rec.blasScratchSize < neededScratch) {
                    destroyBuffer(ctx->allocator(), rec.blasScratch);
                    rec.blasScratch = createAsScratchBuffer(
                            ctx->allocator(), ctx->device(), neededScratch);
                    rec.blasScratchSize = neededScratch;
                }
                blasBuilds[kk].scratchData.deviceAddress = rec.blasScratch.address;

                ranges[kk].primitiveCount = primitiveCount;
                rangePtrs[kk] = &ranges[kk];

                rec.geomVersion = geomVersionOf(geom);
            }

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, N, blasBuilds.data(), rangePtrs.data());
            endAndSubmitOneShot(cb, "refreshGeomBlasBatch (BLAS)");
        }

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

        void refreshMorphedBlas(Mesh& mesh, MorphedMeshState& st) {
            cpuMorphBlend(mesh, st.blendedPositions, st.blendedNormals);
            if (st.blendedPositions.empty() || !st.blas) return;

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), st.blas->vertex.alloc, &mapped);
            std::memcpy(mapped, st.blendedPositions.data(),
                        st.blendedPositions.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), st.blas->vertex.alloc);

            vmaMapMemory(ctx->allocator(), st.blas->normal.alloc, &mapped);
            std::memcpy(mapped, st.blendedNormals.data(),
                        st.blendedNormals.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), st.blas->normal.alloc);

            auto* posAttr = mesh.geometry()->getAttribute<float>("position");
            auto* idxAttr = mesh.geometry()->getIndex();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            const bool indexed = idxAttr != nullptr;
            const uint32_t primitiveCount = indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = st.blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vertexCount - 1;
            if (indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = st.blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags = 0;

            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries = &blasGeom;
            blasBuild.dstAccelerationStructure = st.blas->as;

            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &primitiveCount, &blasSizes);

            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);
            blasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);

            auto* morphObj = mesh.as<ObjectWithMorphTargetInfluences>();
            if (morphObj) st.prevInfluences = morphObj->morphTargetInfluences();
        }

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

            // Per-vertex foam coverage (1 float per vertex, [0..1]). Written
            // by water_displace.comp via Jacobian of horizontal displacement;
            // read by closest_hit.rchit to lerp roughness/albedo toward white
            // where the surface is folding (= breaking-wave whitewater).
            blas->foam = createBuffer(
                    ctx->allocator(), ctx->device(),
                    VkDeviceSize(vertexCount) * sizeof(float),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO);
            // Zero-init: water_displace.comp's decay path reads the previous
            // frame's foam at this vertex and computes max(foamInstant, prev*
            // decay). Without explicit init the first frame reads undefined
            // GPU memory — often huge garbage values (NaN/Inf bit patterns
            // or just large floats) that decay slowly enough to be visible
            // for tens of seconds, bilinear-interpolated across the plane's
            // triangulation as a hex-tile foam pattern.
            {
                VkCommandBuffer cb = beginOneShot();
                vkCmdFillBuffer(cb, blas->foam.handle, 0, VK_WHOLE_SIZE, 0);
                endAndSubmitOneShot(cb);
            }

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
                    rewriteOceanFineDescriptors_();
                }
            }

            // Hand this mesh's world-foam view to closest_hit (binding 33) so
            // the chit can sample foam at arbitrary world XZ during shading.
            // Like oceanFineHeight above, the foam VkImage handle is stable
            // until the DisplacedMesh is destroyed, so we only rewrite once.
            oceanFoamView     = state->foamImage.view;
            oceanFoamTileSize = state->foamTileSize;
            rewriteOceanFoamDescriptors_();

            auto* raw = state.get();
            displacedStates.emplace(&dm, std::move(state));
            return raw;
        }

        // Per-frame: run the FFT chain → water_displace → BLAS rebuild for
        // one DisplacedMesh. Mirrors refreshSkinnedBlas's structure.
        void refreshDisplacedBlas(DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds) {
            VkCommandBuffer cb = beginOneShot();

            // (1)..(3) Run each enabled cascade's FFT chain in turn. Phillips
            // is one-shot per cascade. DynamicSpectrum re-runs each frame.
            // IFFT calls are sequential on the same queue so they can share
            // the single scratch image. Cascades dispatch back-to-back; the
            // Vulkan command buffer recording order plus the IFFT's internal
            // image-layout barriers serialize the work correctly.
            for (uint32_t i = 0; i < 3; ++i) {
                if (!(st.cascadeMask & (1u << i))) continue;
                auto& c = st.cascades[i];
                if (!c.phillipsRecorded) {
                    c.phillips->recordCompute(cb);
                    c.phillipsRecorded = true;
                }
                c.dyn->recordCompute(cb, elapsedSeconds);
                water::OceanImage ht  = c.dyn->ht();
                water::OceanImage dsp = c.dyn->displacement();
                ht.currentLayout  = VK_IMAGE_LAYOUT_GENERAL;
                dsp.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
                st.scratchA.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
                c.ifft->recordApply(cb, ht,  st.scratchA);
                c.ifft->recordApply(cb, dsp, st.scratchA);

                // Copy the spatial-domain height image into the host-mapped
                // readback buffer for this cascade. By the time
                // endAndSubmitOneShot returns, the buffer is filled.
                Buffer* rb = (i == 0) ? &st.heightReadback
                           : (i == 1) ? &st.heightReadback1
                                      : &st.heightReadback2;
                if (rb->handle != VK_NULL_HANDLE) {
                    VkImageMemoryBarrier imb{};
                    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    imb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    imb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    imb.image = c.dyn->ht().image;
                    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &imb);

                    VkBufferImageCopy bic{};
                    bic.bufferOffset = 0;
                    bic.bufferRowLength = 0;
                    bic.bufferImageHeight = 0;
                    bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    bic.imageOffset = {0, 0, 0};
                    bic.imageExtent = {st.heightReadbackDim[i], st.heightReadbackDim[i], 1};
                    vkCmdCopyImageToBuffer(cb, c.dyn->ht().image, VK_IMAGE_LAYOUT_GENERAL,
                                           rb->handle, 1, &bic);

                    VkBufferMemoryBarrier bmb{};
                    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    bmb.buffer = rb->handle;
                    bmb.size   = VK_WHOLE_SIZE;
                    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT,
                            0, 0, nullptr, 1, &bmb, 0, nullptr);
                }
            }

            // (3.5) Per-vertex motion: copy current vertex positions into
            // prevVertex BEFORE water_displace overwrites them. Hybrid-only
            // (raster gbuffer reads prevVertex at attribute 3).
            //
            // Skipped when the adaptive warp is active: the warp re-centres
            // the grid on the vessel every frame, so vertex i is a different
            // world point each frame and prevVertex (indexed by vertex id)
            // can't track a stable surface point. The motion-vector consumers
            // are pointed at the current vertex buffer instead (see the
            // prevVertexAddress / prevPosAddr selection sites), making the
            // ocean reproject as world-static — so the copy is dead work and
            // skipping it also saves a full vertex-buffer transfer per frame.
            const bool warpActive = dm.warp.halfRange > 0.0f;
            if (st.blas->prevVertex.handle != VK_NULL_HANDLE && !warpActive) {
                VkBufferCopy region{};
                region.size = VkDeviceSize(st.vertexCount) * 3u * sizeof(float);
                vkCmdCopyBuffer(cb, st.blas->vertex.handle,
                                st.blas->prevVertex.handle, 1, &region);
                // Barrier: copy WRITE → vertex shader READ (raster prepass)
                // and the immediate water_displace.comp WRITE on vertex.
                std::array<VkBufferMemoryBarrier, 1> bmb{};
                bmb[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                bmb[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                bmb[0].dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                       VK_ACCESS_SHADER_READ_BIT;
                bmb[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bmb[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bmb[0].buffer = st.blas->prevVertex.handle;
                bmb[0].size   = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, nullptr, uint32_t(bmb.size()), bmb.data(), 0, nullptr);
            }

            // (4) Dispatch water_displace.comp → writes positions + normals
            // into the BLAS vertex/normal buffers.

            // (4a) Foam-disturbance SSBO upload. Lazy-allocate the host-
            // mapped buffer on first use, then memcpy the (possibly empty)
            // user-supplied list each frame. Capped at kMaxFoamDisturbances;
            // extras are dropped silently. The fence wait at frame start
            // guarantees the GPU is done reading last frame's disturbances
            // before we overwrite, so no double-buffering is needed.
            const uint32_t kFoamDisturbStride = 16u;  // 4 floats per entry
            const uint32_t kFoamDisturbBytes  =
                    DisplacedMeshState::kMaxFoamDisturbances * kFoamDisturbStride;
            if (st.foamDisturbBuffer.handle == VK_NULL_HANDLE) {
                st.foamDisturbBuffer = createBuffer(
                        ctx->allocator(), ctx->device(), kFoamDisturbBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }
            const uint32_t disturbCount = static_cast<uint32_t>(std::min<size_t>(
                    dm.foamDisturbances.size(),
                    DisplacedMeshState::kMaxFoamDisturbances));
            if (disturbCount > 0u) {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), st.foamDisturbBuffer.alloc, &mapped);
                std::memcpy(mapped, dm.foamDisturbances.data(),
                            disturbCount * kFoamDisturbStride);
                vmaUnmapMemory(ctx->allocator(), st.foamDisturbBuffer.alloc);
            }

            // (4b) Wake-trail SSBO upload. Same pattern as disturbance buffer:
            // lazy-allocate, memcpy newest-first list, drop the tail beyond
            // kMaxWakeSamples. Each entry is 32 B = DisplacedMesh::WakeSample.
            const uint32_t kWakeSampleStride = 32u;
            const uint32_t kWakeTrailBytes   =
                    DisplacedMeshState::kMaxWakeSamples * kWakeSampleStride;
            if (st.wakeTrailBuffer.handle == VK_NULL_HANDLE) {
                st.wakeTrailBuffer = createBuffer(
                        ctx->allocator(), ctx->device(), kWakeTrailBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }
            const uint32_t wakeSampleCount = static_cast<uint32_t>(std::min<size_t>(
                    dm.wake.trail.size(),
                    DisplacedMeshState::kMaxWakeSamples));
            if (wakeSampleCount > 0u) {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), st.wakeTrailBuffer.alloc, &mapped);
                std::memcpy(mapped, dm.wake.trail.data(),
                            wakeSampleCount * kWakeSampleStride);
                vmaUnmapMemory(ctx->allocator(), st.wakeTrailBuffer.alloc);
            }

            vulkan::WaterDisplacePipeline::PushConstants pc{};
            pc.posOut       = st.blas->vertex.address;
            pc.normOut      = st.blas->normal.address;
            pc.foamOut      = st.blas->foam.address;
            pc.disturbAddr  = st.foamDisturbBuffer.address;
            pc.vertexCount  = st.vertexCount;
            pc.gridDim      = st.gridDim;
            pc.planeSize    = st.planeSize;
            pc.tileSize0    = dm.params.tileSize0;
            pc.tileSize1    = dm.params.tileSize1;
            pc.tileSize2    = dm.params.tileSize2;
            pc.waveScale    = dm.params.waveScale;
            pc.choppiness   = dm.params.choppiness;
            pc.cascadeMask  = st.cascadeMask;
            pc.hullCenterX    = dm.hullExclusion.centerX;
            pc.hullCenterZ    = dm.hullExclusion.centerZ;
            pc.hullHalfLength = dm.hullExclusion.halfLength;
            pc.hullHalfBeam   = dm.hullExclusion.halfBeam;
            pc.hullSinYaw     = dm.hullExclusion.sinYaw;
            pc.hullCosYaw     = dm.hullExclusion.cosYaw;
            pc.forwardSpeed   = dm.wake.enabled ? dm.wake.forwardSpeed : 0.0f;
            pc.disturbCount   = disturbCount;
            pc.warpCenterX    = dm.warp.centerX;
            pc.warpCenterZ    = dm.warp.centerZ;
            pc.warpHalfRange  = dm.warp.halfRange;
            pc.warpCoefA      = dm.warp.coefA;
            pc.wakeTrailAddr  = st.wakeTrailBuffer.address;
            pc.wakeTrailCount = wakeSampleCount;
            waterDisplace_->recordDispatch(cb, st.displaceDS, pc);

            // (4c) World-space foam pass — same inputs as water_displace
            // (cascades, disturbances, hull/wake state) but evaluated per
            // foam-texel rather than per mesh vertex. Replaces the per-
            // vertex foam buffer that water_displace used to write. Run
            // AFTER water_displace finishes so we share the descriptor
            // pool's cascade-image bindings without a layout flip; the
            // cascades stay in GENERAL throughout.
            {
                // Wall-clock foam persistence. The old fixed 0.992/frame tied
                // the foam half-life to frame rate (≈1.4 s at 60 fps, half
                // that at 120) — same bug class as the TAA temporal constants.
                // τ = 2 s reproduces the old look at 60 fps. dt clamped so an
                // alt-tab / loading stall can't wipe the accumulator in one
                // frame.
                const double nowSec = glfwGetTime();
                float foamDecay = 0.992f;// first frame: no dt reference yet
                if (st.foamPrevTimeSec >= 0.0) {
                    const float dt = std::clamp(
                            static_cast<float>(nowSec - st.foamPrevTimeSec),
                            0.0f, 0.25f);
                    foamDecay = std::exp(-dt / 2.0f);
                }
                st.foamPrevTimeSec = nowSec;

                vulkan::FoamWorldPipeline::PushConstants fpc{};
                fpc.disturbAddr    = st.foamDisturbBuffer.address;
                fpc.foamRes        = st.foamRes;
                fpc.foamTileSize   = st.foamTileSize;
                fpc.tileSize0      = dm.params.tileSize0;
                fpc.tileSize1      = dm.params.tileSize1;
                fpc.tileSize2      = dm.params.tileSize2;
                fpc.waveScale      = dm.params.waveScale;
                fpc.choppiness     = dm.params.choppiness;
                fpc.cascadeMask    = st.cascadeMask;
                fpc.hullCenterX    = dm.hullExclusion.centerX;
                fpc.hullCenterZ    = dm.hullExclusion.centerZ;
                fpc.hullHalfLength = dm.hullExclusion.halfLength;
                fpc.hullHalfBeam   = dm.hullExclusion.halfBeam;
                fpc.hullSinYaw     = dm.hullExclusion.sinYaw;
                fpc.hullCosYaw     = dm.hullExclusion.cosYaw;
                fpc.forwardSpeed   = dm.wake.enabled ? dm.wake.forwardSpeed : 0.0f;
                fpc.disturbCount   = disturbCount;
                fpc.decay          = foamDecay;
                fpc.wakeTrailAddr  = st.wakeTrailBuffer.address;
                fpc.wakeTrailCount = wakeSampleCount;
                fpc._pad           = 0;
                foamWorld_->recordDispatch(cb, st.foamWorldDS, fpc);

                // Barrier: compute WRITE → ray-trace shader READ on the foam
                // image. chit (binding 44) samples it via a combined image-
                // sampler in the same frame.
                VkImageMemoryBarrier fmb{};
                fmb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                fmb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                fmb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                fmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                fmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                fmb.image = st.foamImage.image;
                fmb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                fmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                fmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        0, 0, nullptr, 0, nullptr, 1, &fmb);
            }

            // Buffer barrier: compute write → AS-build read on the vertex/normal buffers.
            VkBufferMemoryBarrier bbs[2]{};
            bbs[0].sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bbs[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bbs[0].dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            bbs[0].buffer        = st.blas->vertex.handle;
            bbs[0].size          = VK_WHOLE_SIZE;
            bbs[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bbs[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bbs[1] = bbs[0];
            bbs[1].buffer = st.blas->normal.handle;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                0, 0, nullptr, 2, bbs, 0, nullptr);

            // (5) Rebuild the BLAS in-place (same buffers, same AS handle).
            // Mirrors refreshSkinnedBlas — extracted into a helper would be
            // nicer but the duplicated 70-line block is fine for a first cut.
            auto* posAttr = dm.geometry()->getAttribute<float>("position");
            auto* idxAttr = dm.geometry()->getIndex();
            const uint32_t vc = static_cast<uint32_t>(posAttr->count());
            const bool indexed = idxAttr != nullptr;
            const uint32_t primCount = indexed ? static_cast<uint32_t>(idxAttr->count() / 3)
                                               : vc / 3;

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = st.blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vc - 1;
            if (indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = st.blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags = 0;

            // Refit after the initial buildBlasFor; periodic full rebuild
            // keeps the BVH balanced over time. buildBlasFor itself counts as
            // the first build, so blasRefitCounter starts at 0 and the very
            // first refreshDisplacedBlas takes the UPDATE path.
            const bool fullRebuild =
                    st.blasRefitCounter >= DisplacedMeshState::kBlasFullRebuildInterval;
            st.blasRefitCounter = fullRebuild ? 0u : (st.blasRefitCounter + 1u);

            VkAccelerationStructureBuildGeometryInfoKHR build{};
            build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            build.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            build.mode  = fullRebuild
                    ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                    : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            build.geometryCount = 1;
            build.pGeometries = &blasGeom;
            build.srcAccelerationStructure = fullRebuild ? VK_NULL_HANDLE : st.blas->as;
            build.dstAccelerationStructure = st.blas->as;

            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &build, &primCount, &sizes);

            const VkDeviceSize scratchSize =
                    fullRebuild ? sizes.buildScratchSize : sizes.updateScratchSize;
            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), scratchSize);
            build.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &build, &pRange);

            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);

            // Mirror cascade height fields into DisplacedMesh for CPU sampling.
            // endAndSubmit waits for completion, so the staging buffers are
            // fully written by the time we get here.
            {
                struct { Buffer* buf; float tileSize; } cascades[] = {
                    {&st.heightReadback,  dm.params.tileSize0},
                    {&st.heightReadback1, dm.params.tileSize1},
                    {&st.heightReadback2, dm.params.tileSize2},
                };
                for (int ci = 0; ci < 3; ++ci) {
                    auto& cf = dm.heightFields[ci];
                    const uint32_t dim = st.heightReadbackDim[ci];
                    if (cascades[ci].buf->handle != VK_NULL_HANDLE && dim > 0) {
                        const size_t cells = size_t(dim) * size_t(dim);
                        const size_t bytes = cells * 2 * sizeof(float);
                        if (cf.data.size() != cells * 2)
                            cf.data.assign(cells * 2, 0.f);
                        void* mapped = nullptr;
                        vmaMapMemory(ctx->allocator(), cascades[ci].buf->alloc, &mapped);
                        std::memcpy(cf.data.data(), mapped, bytes);
                        vmaUnmapMemory(ctx->allocator(), cascades[ci].buf->alloc);
                        cf.dim      = dim;
                        cf.tileSize = cascades[ci].tileSize;
                    }
                }
            }
        }

        // Build a TLAS over the supplied instance descriptors. Empty input is
        // legal — produces an empty TLAS that always misses.
        void buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances) {
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            const VkDeviceSize instBytes = std::max<VkDeviceSize>(
                    instanceCount * sizeof(VkAccelerationStructureInstanceKHR),
                    sizeof(VkAccelerationStructureInstanceKHR));// keep buf non-empty

            tlasInstancesBuffer = createBuffer(
                    ctx->allocator(), ctx->device(), instBytes,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            if (instanceCount > 0) {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), tlasInstancesBuffer.alloc, &mapped);
                std::memcpy(mapped, instances.data(),
                            instanceCount * sizeof(VkAccelerationStructureInstanceKHR));
                vmaUnmapMemory(ctx->allocator(), tlasInstancesBuffer.alloc);
            }

            VkAccelerationStructureGeometryInstancesDataKHR instData{};
            instData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            instData.arrayOfPointers = VK_FALSE;
            instData.data.deviceAddress = tlasInstancesBuffer.address;

            VkAccelerationStructureGeometryKHR tlasGeom{};
            tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlasGeom.geometry.instances = instData;
            tlasGeom.flags = 0;// per-instance opaque is on VkAccelerationStructureInstanceKHR.flags, not here

            VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
            tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            tlasBuild.geometryCount = 1;
            tlasBuild.pGeometries = &tlasGeom;

            VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};
            tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &tlasBuild, &instanceCount, &tlasSizes);

            tlasBuffer = createBuffer(
                    ctx->allocator(), ctx->device(), tlasSizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkAccelerationStructureCreateInfoKHR tlasCreate{};
            tlasCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            tlasCreate.buffer = tlasBuffer.handle;
            tlasCreate.size = tlasSizes.accelerationStructureSize;
            tlasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            check(ctx->rt().createAccelerationStructure(ctx->device(), &tlasCreate, nullptr, &tlas),
                  "vkCreateAccelerationStructureKHR(TLAS)");

            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), tlasSizes.buildScratchSize);

            tlasBuild.dstAccelerationStructure = tlas;
            tlasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
            endAndSubmitOneShot(cb, "buildTlas");
            destroyBuffer(ctx->allocator(), scratch);
        }

        // Refit the existing TLAS in-place with new instance transforms.
        // Cheaper than a full rebuild and crucially leaves the TLAS handle
        // unchanged so descriptor binding 0 keeps pointing at it. Caller
        // must hold the same instance count as the previous build (only
        // matrices may change) — topology growth requires a full rebuild.
        void refitTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                       bool fullBuild = false) {
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            if (instanceCount == 0 || tlas == VK_NULL_HANDLE) return;

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), tlasInstancesBuffer.alloc, &mapped);
            std::memcpy(mapped, instances.data(),
                        instanceCount * sizeof(VkAccelerationStructureInstanceKHR));
            vmaUnmapMemory(ctx->allocator(), tlasInstancesBuffer.alloc);

            VkAccelerationStructureGeometryInstancesDataKHR instData{};
            instData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            instData.arrayOfPointers = VK_FALSE;
            instData.data.deviceAddress = tlasInstancesBuffer.address;

            VkAccelerationStructureGeometryKHR tlasGeom{};
            tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlasGeom.geometry.instances = instData;
            tlasGeom.flags = 0;

            const auto mode = fullBuild
                    ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                    : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;

            VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
            tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            tlasBuild.mode = mode;
            tlasBuild.srcAccelerationStructure = fullBuild ? VK_NULL_HANDLE : tlas;
            tlasBuild.dstAccelerationStructure = tlas;
            tlasBuild.geometryCount = 1;
            tlasBuild.pGeometries = &tlasGeom;

            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &tlasBuild, &instanceCount, &sizes);

            const VkDeviceSize scratchSize = fullBuild
                    ? sizes.buildScratchSize : sizes.updateScratchSize;
            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), scratchSize);
            tlasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
            endAndSubmitOneShot(cb, "refitTlas");
            destroyBuffer(ctx->allocator(), scratch);
        }

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
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), target.alloc, &mapped);
                std::memcpy(mapped, descs.data(), descs.size() * sizeof(DescT));
                vmaUnmapMemory(ctx->allocator(), target.alloc);
            }
        }

        // Flush the cached host-side MaterialDescs into this frame's slot of
        // the per-frame ring. Called from renderFrame after the fence wait —
        // the in-flight signal guarantees the previous use of this slot has
        // retired, so the memcpy races nothing. Replaces the old shared-buffer
        // path that called vkDeviceWaitIdle on every animated-pbr update.
        void flushMaterialDescsIfDirty(uint32_t frame) {
            if (!matDescsDirty_[frame]) return;
            matDescsDirty_[frame] = false;
            if (matDescsCached_.empty()) return;
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), materialDescsBuffers[frame].alloc, &mapped);
            std::memcpy(mapped, matDescsCached_.data(),
                        matDescsCached_.size() * sizeof(MaterialDesc));
            vmaUnmapMemory(ctx->allocator(), materialDescsBuffers[frame].alloc);
        }

        // Per-frame frustum cull: tag every entry with `inFrustum` so raster
        // passes (gbuf prepass, overlay depth prepass) can skip off-screen
        // geometry. Each draw on the raster gbuf pass costs ~15 µs of GPU
        // command-processor time regardless of whether anything actually
        // rasterizes — at 1500-mesh scenes that's ~22 ms eaten by draws
        // whose vertex shader transforms all land outside the clip cube.
        // The PT path is untouched: TLAS culling handles visibility
        // implicitly on the ray-traced side.
        //
        // Deformable entries (skinned / displaced / morphed) keep
        // inFrustum=true unconditionally — their local AABB is the rest-
        // pose extent, not the deformed one, so a tight test would clip
        // out poses that bulge past the bind silhouette. Re-fitting a
        // tight bound every frame on the CPU isn't worth the cost; the
        // GPU pays a small constant for these. Overlay entries also stay
        // on (debug viz should always render).
        //
        // Side effect: also resolves glassVisibleThisFrame_ from the same
        // walk so the photon emit gating doesn't need a separate frustum
        // build. The flag is true iff at least one transmissive entry
        // passed the cull.
        void cullEntriesAgainstFrustum(Camera& camera) {
            glassVisibleThisFrame_ = false;
            if (lastVisibleEntries_.empty()) return;
            // Combine projection * matrixWorldInverse to extract the world-
            // space frustum (Three.js convention; Camera::updateMatrixWorld
            // already ran in updateCameraUbo this frame).
            Matrix4 vp;
            vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
            Frustum frustum;
            frustum.setFromProjectionMatrix(vp);
            // Pre-compute a sparse "is glass" lookup. glassEntryIndices_ is
            // already sorted ascending (built by walking entries in order),
            // so we can iterate it in lockstep with i.
            size_t gi = 0;
            const size_t gN = glassEntryIndices_.size();
            for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
                auto& en = lastVisibleEntries_[i];
                // Default-include conservative cases — they always draw. Deformers
                // (skinned/displaced/morphed/tet) are here because their cached local
                // AABB doesn't reflect the per-frame deformed extents, so frustum-
                // culling them risks popping a still-on-screen body out of the gbuffer.
                if (en.isOverlay || en.isSkinned || en.isDisplaced || en.isMorphed || en.isTet) {
                    en.inFrustum = true;
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
                // Advance the glass-index cursor in lockstep; if this entry
                // is glass AND in-frustum, light the global flag.
                while (gi < gN && glassEntryIndices_[gi] < i) ++gi;
                if (gi < gN && glassEntryIndices_[gi] == i && en.inFrustum) {
                    glassVisibleThisFrame_ = true;
                }
            }
        }

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
            // mix". PT-safe — transmission>1 reads as full stochastic pass-through
            // (effectively invisible in the PT), which is correct since additive
            // blending has no physical analogue.
            if (mat->blending == Blending::Additive && d.transmission == 0.0f) {
                d.transmission = 1.0f + std::clamp(mat->opacity, 0.0f, 1.0f);
                d.ior          = 1.0f;
            }
            // Alpha-blend transparency (transparent=true, opacity<1) has no
            // physical analogue in a PT, so treat it as stochastic pass-through:
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
            // closest_hit using the albedo texture's alpha channel.
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
            // pass-through in the PT chit) automatically.
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
            if (auto* sp = dynamic_cast<MaterialWithPbrSpecular*>(mat.get())) {
                d.specularIntensity   = sp->specularIntensity;
                d.specularColor[0]    = sp->specularColor.r;
                d.specularColor[1]    = sp->specularColor.g;
                d.specularColor[2]    = sp->specularColor.b;
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
            // (avoids growing the MaterialDesc layout). Mirrors WGPU's
            // shininess = -1 convention in WgpuPathTracerAtlas.cpp.
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
        void ensureSceneBuilt(Object3D& scene) {
            // force=false (matching GLRenderer/WgpuRenderer): with
            // updateMatrix()'s change-detection early-out, only subtrees whose
            // transforms actually moved pay the world-matrix multiplies — a
            // forced pass re-multiplied every node every frame (several
            // ms/frame on Bistro's node count, static or not).
            scene.updateMatrixWorld();

            // First call each render(): start with no motion. Camera + mesh
            // motion checks below + in updateCameraUbo OR true into this flag
            // before dispatch.
            motionThisFrame_ = false;
            cameraMovedThisFrame_ = false;
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
            const bool snapLean = sceneSnapshotMatches(scene);
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
            // traverseVisible (not traverse) so an invisible parent hides its
            // whole subtree — matches three.js / GLRenderer convention. Plain
            // `traverse` walks every node regardless of visibility, leaking
            // children of hidden groups into the PT/overlay passes.
            scene.traverseVisible([&](Object3D& o) {
                SnapNode sn{};
                sn.obj = &o;
                // Line / LineSegments: never path-trace, always overlay.
                // Collected before the Mesh dispatch so subclasses don't
                // accidentally route through the Mesh path.
                if (auto* line = dynamic_cast<Line*>(&o); line && line->visible) {
                    auto geom = line->geometry();
                    sn.flags = kSnapKindLine;
                    sn.line  = line;
                    sn.geom  = geom.get();
                    sn.geomB = geom.get();
                    if (geom) sn.attrVer = geom->attributesVersion();
                    if (geom && geom->hasAttribute("position")) {
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
                // Points (point clouds) — never path-trace, always rasterise
                // as POINT_LIST in the overlay pass. Shares the same geometry
                // cache as Line via the BufferGeometry* key.
                if (auto* pts = dynamic_cast<Points*>(&o); pts && pts->visible) {
                    auto geom = pts->geometry();
                    sn.flags = kSnapKindPoints;
                    sn.pts   = pts;
                    sn.geom  = geom.get();
                    sn.geomB = geom.get();
                    if (geom) sn.attrVer = geom->attributesVersion();
                    if (geom && geom->hasAttribute("position")) {
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
                // it from PT (TLAS, raster G-buffer, emissive NEE). PT can't
                // see/shadow overlay meshes — they're pure debug visuals.
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
                sn.instCount = inst ? static_cast<int32_t>(inst->count()) : -1;
                sn.flags     = snapMeshFlags(*m, wf);
                if (sn.geomB) sn.attrVer = sn.geomB->attributesVersion();
                sceneSnapshot_.push_back(sn);
                auto geom = m->geometry();
                if (!geom || !geom->hasAttribute("position")) return;
                if (!geom->hasAttribute("normal")) return;
                const bool isOverlay = (sn.flags & (kSnapWire | kSnapOverlay)) != 0u;
                // One-shot type probes: an N-instance InstancedMesh costs 3
                // dynamic_casts total, not 3·N — and on snapshot-match frames
                // none at all (the cached entry flags are reused). Consumed by
                // raster pass loops, resolveBlasForEntry, TLAS refit, and
                // dirty-detection.
                const bool isSkinned   = (dynamic_cast<SkinnedMesh*>(m)   != nullptr);
                const bool isDisplaced = (dynamic_cast<DisplacedMesh*>(m) != nullptr);
                const bool isMorphed   = isMorphedMesh(*m);
                // Tet-skinned PhysX soft body — detected via the material flag set by
                // SoftBody::enableGpuSkinning() (which also carries the per-frame tet
                // texture and the static tetIndex/tetWeight/tetRestInv* attributes).
                // Mutually exclusive with the other deformers.
                const bool isTet = !isSkinned && !isDisplaced && !isMorphed &&
                                   m->material() && m->material()->tetSkinning &&
                                   m->material()->tetTexture != nullptr;
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
                        e.isSkinned    = isSkinned;
                        e.isDisplaced  = isDisplaced;
                        e.isMorphed    = isMorphed;
                        e.isTet        = isTet;
                        e.isInstanced  = true;
                        std::memcpy(e.worldMatrix.data(), world.elements.data(), 64);
                        built.push_back(e);
                    }
                } else {
                    MeshEntry e{};
                    e.mesh = m;
                    e.instanceIndex = 0u;
                    e.isOverlay    = isOverlay;
                    e.isSkinned    = isSkinned;
                    e.isDisplaced  = isDisplaced;
                    e.isMorphed    = isMorphed;
                    e.isTet        = isTet;
                    std::memcpy(e.worldMatrix.data(), m->matrixWorld->elements.data(), 64);
                    built.push_back(e);
                }
            });
            lastVisibleEntries_ = std::move(built);
            lastVisibleLines_   = std::move(builtLines);
            }// !snapLean

            // Change flags shared by the LEAN in-place diff and the generic
            // compare loop below — exactly one of the two fills them.
            bool structuralSame = true;
            bool matricesSame = true;
            bool materialValuesSame = true;
            bool bonesDirtyAny = false;
            bool displacedDirtyAny = false;
            bool tetDirtyAny = false;
            bool geomDirtyAny = false;
            bool morphDirtyAny = false;
            std::vector<bool> entryGeomDirty(entries.size(), false);

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
                            emissiveTexOf(*m) != fp.emissiveTex) {
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
                        geomDirtyAny = true;
                        entryGeomDirty[i] = true;
                        // boundingBox invalidation — mirrors the generic loop.
                        if (auto gg = en.mesh->geometry()) gg->boundingBox.reset();
                    }
                    if (xfmChanged) {
                        matricesSame = false;
                        fp.matrix = en.worldMatrix;
                    }
                    if (matChanged) materialValuesSame = false;
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
                    fp.albedoTex             = albedoTexOf(*m);
                    fp.roughnessTex          = roughnessTexOf(*m);
                    fp.metalnessTex          = metalnessTexOf(*m);
                    fp.normalTex             = normalTexOf(*m);
                    fp.transmissionTex       = transmissionTexOf(*m);
                    fp.clearcoatTex          = clearcoatTexOf(*m);
                    fp.clearcoatRoughnessTex = clearcoatRoughnessTexOf(*m);
                    fp.emissiveTex           = emissiveTexOf(*m);
                    fp.instanceIndex         = en.instanceIndex;
                    fp.matrix                = en.worldMatrix;
                    const MaterialDesc md = materialFromMesh(*m);
                    fp.pbr = {md.albedo[0], md.albedo[1], md.albedo[2],
                              md.roughness, md.metalness,
                              md.emissive[0], md.emissive[1], md.emissive[2],
                              md.emissiveIntensity,
                              md.normalScale[0], md.normalScale[1],
                              md.transmission, md.ior,
                              md.clearcoat, md.clearcoatRoughness};
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
                    const bool mo = entryMorphDirty[i];
                    if (!(b || d || mo)) continue;
                    if (b) bonesDirtyAny = true;
                    if (d) displacedDirtyAny = true;
                    if (mo) morphDirtyAny = true;
                    const size_t w = i >> 5;
                    if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                    meshMovedBits_[w] |= (1u << (i & 31u));
                }
            }
            // Continuous-motion fast path: when only the per-mesh matrices
            // changed (everything else — topology, materials, textures —
            // matches), refit the TLAS in place and let raygen reproject.
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
                // structural caused full rebuild every frame, which reset the
                // gbuf+sampleIndex globally and froze accumulation scene-wide.
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
                        a.emissiveTex != b.emissiveTex) {
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
                    const bool geomChanged  = (a.geomVersion != b.geomVersion);
                    const bool morphChanged = entryMorphDirty[i];
                    if (xfmChanged) matricesSame = false;
                    if (matChanged) materialValuesSame = false;
                    if (bonesChanged) bonesDirtyAny = true;
                    if (dispChanged)  displacedDirtyAny = true;
                    if (geomChanged) {
                        geomDirtyAny = true;
                        entryGeomDirty[i] = true;
                        // Invalidate the cached boundingBox so the next
                        // cullEntriesAgainstFrustum recomputes it from the
                        // current positions. Without this, plain meshes with
                        // in-place vertex updates (PhysX soft bodies) keep
                        // the rest-pose AABB and get culled out by the raster
                        // pass once the body settles outside that stale box —
                        // gbuffer ends up with sky at those pixels and the
                        // hybrid raygen renders the background through them.
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
                    if (!matricesSame || !materialValuesSame || bonesDirtyAny || displacedDirtyAny || geomDirtyAny || morphDirtyAny) {
                        motionThisFrame_ = true;
                    }
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
                        // Re-displace every DisplacedMesh: run FFT chain →
                        // water_displace.comp → BLAS rebuild in place. Same
                        // TLAS-AABB caveat as SkinnedMesh: refit happens just
                        // below.
                        const float now = static_cast<float>(glfwGetTime());
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (!entryDisplacedDirty[i]) continue;
                            auto* dm = static_cast<DisplacedMesh*>(entries[i].mesh);
                            auto stIt = displacedStates.find(dm);
                            if (stIt == displacedStates.end()) continue;
                            refreshDisplacedBlas(*dm, *stIt->second, now);
                            ++dm->frameTick;
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
                    if (tetDirtyAny) motionThisFrame_ = true;
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
                    }
                    if (!matricesSame || bonesDirtyAny || displacedDirtyAny || tetDirtyAny || geomDirtyAny || morphDirtyAny) {
                        // TLAS refit: needed when instance transforms change
                        // (matricesSame=false) AND when any skinned BLAS was
                        // just rebuilt — the TLAS's per-instance wrapped AABB
                        // is recomputed from the current BLAS extents on
                        // refit, picking up pose-deformed silhouettes that
                        // would otherwise be clipped by the stale TLAS AABB.
                        // BLAS handles + buffer addresses are unchanged so
                        // geomDescs / matDescs stay valid; we just rewrite the
                        // tlasInstancesBuffer in place and call MODE_UPDATE.
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
                                blasAddr = it->second->address;
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
                        const bool blasDeformed = bonesDirtyAny || displacedDirtyAny || tetDirtyAny || morphDirtyAny || geomDirtyAny;
                        refitTlas(instances, blasDeformed);
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
                        matDescsCached_.assign(entries.size(), MaterialDesc{});
                        // Same dedup as the full-rebuild path below: entries
                        // order is identical (overlay skip matches), so
                        // identical Material* pointers produce identical
                        // materialAssetIdx values frame-to-frame.
                        std::unordered_map<const Material*, uint32_t> matAssetMap;
                        for (size_t i = 0; i < entries.size(); ++i) {
                            const MeshEntry& en = entries[i];
                            if (en.isOverlay) continue;// raster-overlay only — no MaterialDesc slot
                            Mesh* m = en.mesh;
                            MaterialDesc md = materialFromMesh(*m);
                            {
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
                            matDescsCached_[i] = md;
                        }
                        for (auto& d : matDescsDirty_) d = true;
                        sceneHasGlass_ = false;
                        sceneHasClearcoat_ = false;
                        sceneHasIridescence_ = false;
                        sceneHasSheen_ = false;
                        glassEntryIndices_.clear();
                        // matDescsCached_ is ENTRIES-indexed (overlay slots left
                        // default), so index it directly by i. glassEntryIndices_
                        // stays entry-index based (cullEntriesAgainstFrustum walks
                        // it in lockstep with i).
                        for (size_t i = 0; i < entries.size(); ++i) {
                            if (entries[i].isOverlay) continue;
                            const auto& md = matDescsCached_[i];
                            if (md.transmission > 0.0f) {
                                sceneHasGlass_ = true;
                                glassEntryIndices_.push_back(i);
                            }
                            if (md.clearcoat > 0.0f) sceneHasClearcoat_ = true;
                            if (md.iridescence > 0.0f) sceneHasIridescence_ = true;
                            if (md.sheenRoughness > 0.0f &&
                                (md.sheenColor[0] > 0.0f || md.sheenColor[1] > 0.0f ||
                                 md.sheenColor[2] > 0.0f)) {
                                sceneHasSheen_ = true;
                            }
                        }
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
            if (sceneBuilt_) {
                vkDeviceWaitIdle(ctx->device());
                if (tlas) {
                    ctx->rt().destroyAccelerationStructure(ctx->device(), tlas, nullptr);
                    tlas = VK_NULL_HANDLE;
                }
                destroyBuffer(ctx->allocator(), tlasBuffer);
                destroyBuffer(ctx->allocator(), tlasInstancesBuffer);
                destroyBuffer(ctx->allocator(), geometryDescsBuffer);
                for (auto& b : materialDescsBuffers) {
                    destroyBuffer(ctx->allocator(), b);
                    b = {};
                }
                tlasBuffer = {};
                tlasInstancesBuffer = {};
                geometryDescsBuffer = {};

                // Prune stale cache entries whose underlying objects have been
                // destroyed (typical on model swap). Keeping them risks an
                // address-collision: a fresh BufferGeometry / Texture allocated
                // at the same C++ address would silently match the old cache
                // entry and reuse the wrong GPU resource.
                for (auto it = blasCache.begin(); it != blasCache.end(); ) {
                    if (it->second->liveCheck.expired()) {
                        auto& rec = it->second;
                        if (rec->as) ctx->rt().destroyAccelerationStructure(ctx->device(), rec->as, nullptr);
                        destroyBuffer(ctx->allocator(), rec->storage);
                        destroyBuffer(ctx->allocator(), rec->vertex);
                        destroyBuffer(ctx->allocator(), rec->index);
                        destroyBuffer(ctx->allocator(), rec->normal);
                        destroyBuffer(ctx->allocator(), rec->uv);
                        destroyBuffer(ctx->allocator(), rec->foam);
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
                            destroyBuffer(ctx->allocator(), rec->foam);
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
                            destroyBuffer(ctx->allocator(), rec->foam);
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
                            destroyBuffer(ctx->allocator(), rec->foam);
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
                            destroyBuffer(ctx->allocator(), rec->foam);
                            destroyBuffer(ctx->allocator(), rec->prevVertex);
                        }
                        it = morphedMeshStates.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = textureCache.begin(); it != textureCache.end(); ) {
                    if (it->second.first.expired()) {
                        const uint32_t slot = it->second.second;
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
            }

            std::vector<VkAccelerationStructureInstanceKHR> instances;
            instances.reserve(entries.size());
            // geomDescs / matDescs are ENTRIES-indexed: one slot per entry, with
            // overlay/skipped entries left default. This makes
            // gl_InstanceCustomIndexEXT == the entry index, so the raster gbuffer
            // id, RT gbuffer id, motionMat and meshMovedBits all live in ONE index
            // space. (Previously these were filtered/contiguous, so any overlay or
            // skipped entry shifted the filtered index away from the entry index —
            // corrupting gbuffer.frag's material/normal-map lookup and raygen's
            // motionMat[] / isMeshMoved() reads for every mesh after the skip.)
            // Dummy slots are never referenced: skipped entries get no TLAS
            // instance and are not drawn in the gbuffer (both skip on isOverlay).
            std::vector<GeometryDesc> geomDescs(entries.size());
            std::vector<MaterialDesc> matDescs(entries.size());

            // Material-asset dedup. Meshes sharing one Material C++ pointer
            // get the same matAssetIdx so raygen's bilinear reproject gate
            // can accept their cross-mesh taps (tiled walls etc.). The map
            // grows by ~one entry per UNIQUE Material in the scene — well
            // below entries.size() in typical content.
            std::unordered_map<const Material*, uint32_t> matAssetMap;

            for (size_t i = 0; i < entries.size(); ++i) {
                const MeshEntry& en = entries[i];
                Mesh* m = en.mesh;
                const BufferGeometry* geomKey = m->geometry().get();

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
                    auto* st = ensureDisplacedState(*dm);
                    if (!st) continue;
                    recPtr = st->blas.get();
                    // Trigger an initial FFT/displace dispatch so the BLAS
                    // contents (rest grid right now) become the displaced
                    // surface before the first ray-trace sees it.
                    refreshDisplacedBlas(*dm, *st, static_cast<float>(glfwGetTime()));
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
                            if (old->as) ctx->rt().destroyAccelerationStructure(ctx->device(), old->as, nullptr);
                            destroyBuffer(ctx->allocator(), old->storage);
                            destroyBuffer(ctx->allocator(), old->vertex);
                            destroyBuffer(ctx->allocator(), old->index);
                            destroyBuffer(ctx->allocator(), old->normal);
                            destroyBuffer(ctx->allocator(), old->uv);
                            destroyBuffer(ctx->allocator(), old->foam);
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
                // GeometryDesc/MaterialDesc arrays — PT must not see them.
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
                inst.accelerationStructureReference = recPtr->address;
                instances.push_back(inst);

                GeometryDesc gdesc{};
                gdesc.vertexAddress = recPtr->vertex.address;
                gdesc.normalAddress = recPtr->normal.address;
                gdesc.indexAddress  = recPtr->index.address;
                gdesc.uvAddress     = recPtr->uv.address;// 0 if no UV attribute
                gdesc.foamAddress   = recPtr->foam.address;// 0 unless this is an FFT-displaced ocean mesh
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
                gdesc.indexed = recPtr->index.handle != VK_NULL_HANDLE ? 1u : 0u;
                gdesc._pad = 0;
                geomDescs[i] = gdesc;

                MaterialDesc md = materialFromMesh(*m);
                {
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

            buildTlas(instances);
            uploadDescBuffer(geometryDescsBuffer, geomDescs);
            // Seed every per-frame slot with the fresh matDescs so the first
            // few frames don't try to flush against a half-initialised ring.
            // matDescsCached_ stays in sync as the host-side authoritative
            // copy used by the hot-path flush.
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                uploadDescBuffer(materialDescsBuffers[f], matDescs);
            }
            matDescsCached_ = matDescs;
            for (auto& d : matDescsDirty_) d = false;
            sceneHasGlass_ = false;
            sceneHasClearcoat_ = false;
            sceneHasIridescence_ = false;
            sceneHasSheen_ = false;
            glassEntryIndices_.clear();
            // matDescs is now ENTRIES-indexed, so index it directly by i (was a
            // separate filtered counter mi). glassEntryIndices_ stays entry-index
            // based — cullEntriesAgainstFrustum walks it in lockstep with i.
            for (size_t i = 0; i < entries.size(); ++i) {
                if (entries[i].isOverlay) continue;
                const auto& md = matDescs[i];
                if (md.transmission > 0.0f) {
                    sceneHasGlass_ = true;
                    glassEntryIndices_.push_back(i);
                }
                if (md.clearcoat > 0.0f) sceneHasClearcoat_ = true;
                if (md.iridescence > 0.0f) sceneHasIridescence_ = true;
                if (md.sheenRoughness > 0.0f &&
                    (md.sheenColor[0] > 0.0f || md.sheenColor[1] > 0.0f ||
                     md.sheenColor[2] > 0.0f)) {
                    sceneHasSheen_ = true;
                }
            }
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
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            const uint32_t neededBitWords = std::max<uint32_t>((instanceCount + 31u) / 32u, 1u);
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                ensureMotionMatCapacity(f, std::max<uint32_t>(instanceCount, 1u));
                ensureMeshMovedBitsCapacity(f, neededBitWords);
            }
            meshMovedBits_.resize(neededBitWords, 0u);

            if (sceneBuilt_) {
                rewriteSceneDescriptors();
            } else {
                allocateAndUpdateDescriptors();
            }
            // The (re)build above rebound the RT bindless texture array
            // (binding 8). The raster descriptor's binding 3 mirrors that same
            // table, so invalidate its per-slot cache — each frame slot then
            // re-writes binding 3 on its next uploadRasterCameraUbo. This is
            // the only event that changes the table, so it's the only place
            // the raster mirror needs invalidating.
            rasterMatTexValid_.fill(0);
            prevSceneFingerprint = std::move(currFp);
            sceneBuilt_ = true;
            // Structural change invalidates the photon grid: cells from the
            // old scene's world layout linger and would leak into the new
            // scene's gather. Force the one-shot zero-fill shim to re-run.
            if (photon_) photon_->markUninitialized();
        }

        void createCameraUbos() {
            for (auto& b : cameraUbos) {
                b = createBuffer(
                        ctx->allocator(), ctx->device(),
                        /*size*/ 2 * 16 * sizeof(float) + 4 * sizeof(float),
                        // viewInverse + projInverse + jitter (.xy = clip-space
                        // sub-pixel offset matching raster's Halton, .zw = 1/res)
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
            }
            for (auto& b : prevCameraUbos) {
                b = createBuffer(
                        ctx->allocator(), ctx->device(),
                        /*size*/ 16 * sizeof(float),// 4×vec4: posX, fwdY, rgt, up
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
            }
            // Initial cap = 1 identity matrix. ensureMotionMatCapacity grows
            // and triggers a descriptor rewrite the first time a scene is
            // built with > 1 instance.
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                motionMatBuffers[f] = createBuffer(
                        ctx->allocator(), ctx->device(),
                        /*size*/ 16 * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
                motionMatBufferCapacity[f] = 1;
                // Seed an identity so reads on the first frame (with descriptors
                // already wired) get a sane answer even before any scene build.
                static const float identity[16] = {
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1};
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), motionMatBuffers[f].alloc, &mapped);
                std::memcpy(mapped, identity, sizeof(identity));
                vmaUnmapMemory(ctx->allocator(), motionMatBuffers[f].alloc);
            }
            // Seed mesh-moved-bits buffers with capacity 1 word (32 meshes worth)
            // so descriptor writes have a valid handle before any scene build.
            // ensureMeshMovedBitsCapacity grows in place when scenes need more.
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                meshMovedBitsBuffers[f] = createBuffer(
                        ctx->allocator(), ctx->device(),
                        /*size*/ sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
                meshMovedBitsBufferCapacity[f] = 1;
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), meshMovedBitsBuffers[f].alloc, &mapped);
                const uint32_t zero = 0u;
                std::memcpy(mapped, &zero, sizeof(zero));
                vmaUnmapMemory(ctx->allocator(), meshMovedBitsBuffers[f].alloc);
            }
            // Seed emissive-tri buffers with capacity 1 so descriptor writes have
            // a valid handle even before any emissive geometry exists. The shader
            // reads it only when pc.emissiveCount > 0, so the dummy bytes are
            // never sampled — but Vulkan requires valid resource bindings.
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                emissiveTriBuffers[f] = createBuffer(
                        ctx->allocator(), ctx->device(),
                        /*size*/ 64,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
                emissiveTriBufferCapacity[f] = 1;
            }
        }

        // Grow emissiveTriBuffers[frame] in-place if the current frame's
        // emissive count exceeds capacity. Returns true when the buffer
        // handle changed so the caller can rewrite binding 14. 2× headroom
        // matches motionMatBuffers.
        bool ensureEmissiveTriCapacity(uint32_t frame, uint32_t needed) {
            if (needed <= emissiveTriBufferCapacity[frame]) return false;
            const uint32_t newCap = std::max<uint32_t>(needed, emissiveTriBufferCapacity[frame] * 2u);
            destroyBuffer(ctx->allocator(), emissiveTriBuffers[frame]);
            emissiveTriBuffers[frame] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    /*size*/ static_cast<VkDeviceSize>(newCap) * 64u,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            emissiveTriBufferCapacity[frame] = newCap;
            return true;
        }

        // Clear both ping-pong gbuf images AND both accum images to
        // (0,0,0,0). gbuf.w == 0 (mesh-ID floatBitsToUint == 0u) is sky/no-hit
        // so the raygen reproject mesh-ID guard misses on every subsequent
        // reprojection — producing fresh accumulation everywhere on the next
        // frame. We also clear accum so a stale tap that *does* slip past the
        // guard (e.g. by undefined memory aliasing a real mesh-ID after image
        // creation) returns 0 mean / 0 fc instead of garbage. Caller must
        // hold the GPU idle (we don't issue our own barrier-into-TRANSFER_DST
        // since the only legal layout transition path for an image being
        // cleared is GENERAL → TRANSFER_DST → GENERAL).
        void clearGbufImages() {
            VkCommandBuffer cb = beginOneShot();

            // Includes ReSTIR DI reservoir ping-pong (28/29 pos, 30/31 W) so
            // M=0 on frame 0's read side and the temporal-reuse path correctly
            // sees "no prior history" instead of garbage.
            // Also includes ReSTIR GI reservoir ping-pong (38/39 xs+Wsum,
            // 40/41 ns+M, 42/43 Lo+W) — the chit's validity check (ns is a
            // unit vector AND prevW>0 AND prevM>0) rejects zero-cleared slots
            // so first-frame reads see "no prior history" cleanly.
            // Also includes denoiser_'s moments images (33/34) so M2=0 on
            // first read — variance reads from a stale-undefined moments slot
            // would give huge spurious variance, blowing the σ_lum estimate.
            // Also includes the albedo ping-pong (35/36) so both slots' .a=0
            // on first read — raygen's temporal blend uses prev albedo at
            // mesh-ID validated taps, and uninitialised .a=garbage would
            // fool the demod-valid gate.
            std::array<VkImage, 19> images = {
                    gbufImagesPP[0].image, gbufImagesPP[1].image,
                    accumImagesPP[0].image, accumImagesPP[1].image,
                    reservoirPosImagesPP[0].image, reservoirPosImagesPP[1].image,
                    reservoirWImagesPP[0].image, reservoirWImagesPP[1].image,
                    giResXsImagesPP[0].image, giResXsImagesPP[1].image,
                    giResNsImagesPP[0].image, giResNsImagesPP[1].image,
                    giResLoImagesPP[0].image, giResLoImagesPP[1].image,
                    denoiser_->momentsImage(0), denoiser_->momentsImage(1),
                    denoiser_->albedoImage(0), denoiser_->albedoImage(1),
                    denoiser_->albedoSnapshotImage(),
            };

            std::array<VkImageMemoryBarrier2, 19> toTransfer{};
            for (size_t i = 0; i < images.size(); ++i) {
                toTransfer[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                toTransfer[i].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                toTransfer[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                toTransfer[i].dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                toTransfer[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toTransfer[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                toTransfer[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toTransfer[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toTransfer[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toTransfer[i].image = images[i];
                toTransfer[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toTransfer[i].subresourceRange.levelCount = 1;
                toTransfer[i].subresourceRange.layerCount = 1;
            }
            VkDependencyInfo dep1{};
            dep1.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep1.imageMemoryBarrierCount = static_cast<uint32_t>(toTransfer.size());
            dep1.pImageMemoryBarriers = toTransfer.data();
            vkCmdPipelineBarrier2(cb, &dep1);

            VkClearColorValue clear{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (VkImage img : images) {
                vkCmdClearColorImage(cb, img,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     &clear, 1, &range);
            }

            std::array<VkImageMemoryBarrier2, 19> toGeneral{};
            for (size_t i = 0; i < images.size(); ++i) {
                toGeneral[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                toGeneral[i].srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                toGeneral[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toGeneral[i].dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                toGeneral[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                toGeneral[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toGeneral[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                toGeneral[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGeneral[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGeneral[i].image = images[i];
                toGeneral[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toGeneral[i].subresourceRange.levelCount = 1;
                toGeneral[i].subresourceRange.layerCount = 1;
            }
            VkDependencyInfo dep2{};
            dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep2.imageMemoryBarrierCount = static_cast<uint32_t>(toGeneral.size());
            dep2.pImageMemoryBarriers = toGeneral.data();
            vkCmdPipelineBarrier2(cb, &dep2);

            endAndSubmitOneShot(cb);
        }

        // Compute per-instance motion matrices = prevWorld * inverse(curWorld)
        // and upload to motionMatBuffers[frame]. Identity for first-seen
        // entries (cold-start frame after a topology rebuild) so the reproject
        // is a no-op until prevWorldMats picks up real history. Keying by
        // (Mesh*, instanceIndex) so each InstancedMesh sub-instance carries
        // its own motion delta. Caller must have already waited the
        // inFlight[frame] fence — we write a buffer the GPU may have been
        // reading on the previous use of `frame`.
        void computeAndUploadMotionMatrices(uint32_t frame,
                                            const std::vector<MeshEntry>& entries) {
            const uint32_t count = static_cast<uint32_t>(entries.size());
            if (count == 0) return;

            // Per-instance "settled" threshold. Physics solvers (Bullet,
            // PhysX, etc.) often leave bodies with sub-millimeter / sub-
            // milliradian residual jitter even at rest — a parked car still
            // ticks every frame as the constraint solver re-converges. The
            // PT pipeline faithfully reflects that motion (motionMat → motion
            // vec → FC reset/reproject) and the user sees a wobble that
            // doesn't actually exist in the asset.
            //
            // Threshold the per-element matrix delta: if the largest absolute
            // change is below kSettledEps, treat motion as identity AND
            // don't update prevWorldMats. The body then locks to its prior
            // pose persistently — sub-eps accumulation can't drift past the
            // threshold because the reference doesn't update.
            //
            // 1e-4 covers ~0.1mm translation / ~0.0057° rotation. Tighter
            // than typical solver residual, looser than visible motion.
            // Tune up if your physics still wobbles; tune down if real slow
            // motion gets frozen.
            constexpr float kSettledEps = 1e-4f;

            std::vector<float> data(count * 16);
            std::vector<uint8_t> settled(count, 0);
            bool anyNonIdentity = false;
            for (uint32_t i = 0; i < count; ++i) {
                Matrix4 cur;
                std::memcpy(cur.elements.data(), entries[i].worldMatrix.data(), 64);

                Matrix4 motion;// identity by default
                EntryKey key{entries[i].mesh, entries[i].instanceIndex};
                auto it = prevWorldMats.find(key);
                if (it != prevWorldMats.end()) {
                    Matrix4 prev;
                    std::memcpy(prev.elements.data(), it->second.data(), 64);

                    // Per-element max-abs delta. Cheap, catches both
                    // translation (cols 12..14) and rotation/scale (rest).
                    float maxDelta = 0.0f;
                    for (int e = 0; e < 16; ++e) {
                        const float d = std::abs(cur.elements[e] - prev.elements[e]);
                        if (d > maxDelta) maxDelta = d;
                    }
                    if (maxDelta < kSettledEps) {
                        settled[i] = 1u;// keep motion as identity
                    } else {
                        Matrix4 curInv;
                        curInv.copy(cur).invert();
                        motion.multiplyMatrices(prev, curInv);
                        anyNonIdentity = true;
                    }
                }
                std::memcpy(&data[i * 16], motion.elements.data(), 64);
            }

            // Fast path: if every entry's motion is identity AND the buffer
            // slot was already all-identity from a previous frame, skip the
            // upload. mmap+memcpy of an all-zero scene's per-frame motionMat
            // is a real cost on heavy scenes (1500 entries × 64B = 96KB
            // mapped+written every frame for nothing). Sub-millisecond
            // individually, but it adds up on CPU-bound paths.
            if (!anyNonIdentity && motionMatBufferAllIdentity_[frame]) {
                // Buffer slot already holds identities; skip the upload.
            } else {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), motionMatBuffers[frame].alloc, &mapped);
                std::memcpy(mapped, data.data(), data.size() * sizeof(float));
                vmaUnmapMemory(ctx->allocator(), motionMatBuffers[frame].alloc);
                motionMatBufferAllIdentity_[frame] = !anyNonIdentity;
            }

            // Record this frame's matrices for next frame's motion delta —
            // BUT skip settled entries so prev stays anchored to its
            // pre-jitter pose. Re-evaluating against the same frozen prev
            // each frame keeps the body locked in render space until real
            // motion actually crosses the eps threshold.
            for (uint32_t i = 0; i < count; ++i) {
                if (settled[i]) continue;
                EntryKey key{entries[i].mesh, entries[i].instanceIndex};
                prevWorldMats[key] = entries[i].worldMatrix;
            }
        }

        // Upload meshMovedBits_ to meshMovedBitsBuffers[frame]. Caller must have
        // already waited the inFlight[frame] fence.
        void uploadMeshMovedBits(uint32_t frame) {
            if (meshMovedBits_.empty()) return;
            const VkDeviceSize bytes = meshMovedBits_.size() * sizeof(uint32_t);
            const VkDeviceSize cap   = meshMovedBitsBufferCapacity[frame] * sizeof(uint32_t);
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), meshMovedBitsBuffers[frame].alloc, &mapped);
            std::memcpy(mapped, meshMovedBits_.data(), std::min(bytes, cap));
            vmaUnmapMemory(ctx->allocator(), meshMovedBitsBuffers[frame].alloc);
        }

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
                                        const std::vector<MeshEntry>& entries) {
            emissiveTriCountThisFrame_ = 0;
            emissiveTotalPowerThisFrame_ = 0.0f;

            // Fast path: nothing that affects the world-space emissive CDF
            // has changed since the last rebuild. World-space tri positions
            // depend on mesh world matrices + emissive material values; both
            // are tracked by meshMovedBits_ (set on xfm OR mat OR bone change).
            // Camera motion does NOT invalidate. Bistro / Sponza static
            // frames hit this path and skip the per-tri walk entirely; the
            // walk is O(visible-emissive-meshes × tris) per frame and was
            // CPU-bound on Bistro before this cache.
            const bool anyMeshMoved =
                    std::any_of(meshMovedBits_.begin(), meshMovedBits_.end(),
                                [](uint32_t v) { return v != 0u; });
            const bool entriesUnchanged = (cachedEmissiveEntryCount_ == entries.size());
            if (!anyMeshMoved && entriesUnchanged) {
                emissiveTriCountThisFrame_   = cachedEmissiveTriCount_;
                emissiveTotalPowerThisFrame_ = cachedEmissiveTotalPower_;
                if (cachedEmissiveTriCount_ == 0) {
                    return false;// no emissives → nothing to upload
                }
                if (emissiveBufferVersion_[frame] == cachedEmissiveVersion_) {
                    // This frame's GPU buffer already holds the cached data.
                    return false;
                }
                const bool grew = ensureEmissiveTriCapacity(frame, cachedEmissiveTriCount_);
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), emissiveTriBuffers[frame].alloc, &mapped);
                std::memcpy(mapped, cachedEmissiveData_.data(),
                            cachedEmissiveData_.size() * sizeof(float));
                vmaUnmapMemory(ctx->allocator(), emissiveTriBuffers[frame].alloc);
                emissiveBufferVersion_[frame] = cachedEmissiveVersion_;
                return grew;
            }

            std::vector<float> data;// 16 floats per tri
            data.reserve(64 * 16);
            float cumPower = 0.0f;

            for (const auto& en : entries) {
                if (en.isOverlay) continue;// raster-overlay only — no emissive contribution to PT
                if (!en.mesh) continue;
                auto matPtr = en.mesh->material();
                if (!matPtr) continue;
                auto* em = dynamic_cast<MaterialWithEmissive*>(matPtr.get());
                if (!em) continue;
                const float emR = em->emissive.r * em->emissiveIntensity;
                const float emG = em->emissive.g * em->emissiveIntensity;
                const float emB = em->emissive.b * em->emissiveIntensity;
                const float emLum = 0.2126f * emR + 0.7152f * emG + 0.0722f * emB;
                if (emLum < 1e-6f) continue;
                // emissiveMap modulates per-texel; we don't sample textures here,
                // so use the constant tint for power. Slightly under-samples
                // bright textured emissives but keeps the build lightweight.

                auto geomPtr = en.mesh->geometry();
                if (!geomPtr) continue;
                auto* posAttr = geomPtr->getAttribute<float>("position");
                if (!posAttr) continue;
                const auto& positions = posAttr->array();
                const uint32_t vcount = static_cast<uint32_t>(posAttr->count());
                if (vcount < 3) continue;

                const auto* idxAttr = geomPtr->getIndex();
                const bool indexed = idxAttr != nullptr;
                const uint32_t triCount = indexed
                        ? static_cast<uint32_t>(idxAttr->count() / 3)
                        : vcount / 3;
                if (triCount == 0) continue;

                const float* M = en.worldMatrix.data();// column-major 4x4
                auto xform = [&](float x, float y, float z, float& wx, float& wy, float& wz) {
                    wx = M[0] * x + M[4] * y + M[8]  * z + M[12];
                    wy = M[1] * x + M[5] * y + M[9]  * z + M[13];
                    wz = M[2] * x + M[6] * y + M[10] * z + M[14];
                };

                const auto* indices = indexed ? idxAttr->array().data() : nullptr;
                for (uint32_t t = 0; t < triCount; ++t) {
                    uint32_t i0, i1, i2;
                    if (indexed) {
                        i0 = indices[t * 3 + 0];
                        i1 = indices[t * 3 + 1];
                        i2 = indices[t * 3 + 2];
                    } else {
                        i0 = t * 3 + 0;
                        i1 = t * 3 + 1;
                        i2 = t * 3 + 2;
                    }
                    if (i0 >= vcount || i1 >= vcount || i2 >= vcount) continue;
                    float w0x, w0y, w0z, w1x, w1y, w1z, w2x, w2y, w2z;
                    xform(positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2],
                          w0x, w0y, w0z);
                    xform(positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2],
                          w1x, w1y, w1z);
                    xform(positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2],
                          w2x, w2y, w2z);
                    const float ex = w1x - w0x, ey = w1y - w0y, ez = w1z - w0z;
                    const float fx = w2x - w0x, fy = w2y - w0y, fz = w2z - w0z;
                    const float cx = ey * fz - ez * fy;
                    const float cy = ez * fx - ex * fz;
                    const float cz = ex * fy - ey * fx;
                    const float area = 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);
                    if (!(area > 1e-8f)) continue;
                    const float power = emLum * area;
                    cumPower += power;

                    data.push_back(w0x); data.push_back(w0y); data.push_back(w0z);
                    data.push_back(area);
                    data.push_back(w1x); data.push_back(w1y); data.push_back(w1z);
                    data.push_back(cumPower);
                    data.push_back(w2x); data.push_back(w2y); data.push_back(w2z);
                    data.push_back(power);
                    data.push_back(emR); data.push_back(emG); data.push_back(emB);
                    data.push_back(0.0f);
                }
            }

            const uint32_t triCount = static_cast<uint32_t>(data.size() / 16);
            emissiveTriCountThisFrame_ = triCount;
            emissiveTotalPowerThisFrame_ = cumPower;

            // Update cache regardless — non-emissive scenes still want
            // entriesUnchanged + 0-tri to short-circuit out of the walk.
            cachedEmissiveData_           = std::move(data);
            cachedEmissiveTriCount_       = triCount;
            cachedEmissiveTotalPower_     = cumPower;
            cachedEmissiveEntryCount_     = entries.size();
            cachedEmissiveVersion_++;
            // Force per-frame upload below; mark this slot as up-to-date
            // after the memcpy, leaving the other slot stale until its turn.
            for (auto& v : emissiveBufferVersion_) v = 0;

            if (triCount == 0) return false;

            const bool grew = ensureEmissiveTriCapacity(frame, triCount);

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), emissiveTriBuffers[frame].alloc, &mapped);
            std::memcpy(mapped, cachedEmissiveData_.data(),
                        cachedEmissiveData_.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), emissiveTriBuffers[frame].alloc);
            emissiveBufferVersion_[frame] = cachedEmissiveVersion_;
            return grew;
        }

        // Rewrite binding 14 (emissive-tri SSBO) across all sets for the given
        // frame-in-flight. Called when buildAndUploadEmissiveTris reports the
        // backing buffer was reallocated.
        void rewriteEmissiveTriDescriptors(uint32_t frame) {
            VkDescriptorBufferInfo bufInfo{};
            bufInfo.buffer = emissiveTriBuffers[frame].handle;
            bufInfo.offset = 0;
            bufInfo.range  = VK_WHOLE_SIZE;
            std::vector<VkWriteDescriptorSet> writes(imageCount_);
            for (uint32_t k = 0; k < imageCount_; ++k) {
                const uint32_t setIdx = frame * imageCount_ + k;
                writes[k] = {};
                writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[k].dstSet = descriptorSets[setIdx];
                writes[k].dstBinding = 14;
                writes[k].descriptorCount = 1;
                writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[k].pBufferInfo = &bufInfo;
            }
            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }

        // Grow motionMatBuffers[frame] in-place if the current scene's
        // instance count exceeds capacity. Returns true when the buffer
        // handle changed so the caller can rewrite binding 10. We grow with
        // 2× headroom to avoid thrashing on incremental scene growth.
        bool ensureMotionMatCapacity(uint32_t frame, uint32_t needed) {
            if (needed <= motionMatBufferCapacity[frame]) return false;
            const uint32_t newCap = std::max<uint32_t>(needed, motionMatBufferCapacity[frame] * 2u);
            destroyBuffer(ctx->allocator(), motionMatBuffers[frame]);
            motionMatBuffers[frame] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    /*size*/ newCap * 16 * sizeof(float),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            motionMatBufferCapacity[frame] = newCap;
            // New buffer is undefined; force the next upload to actually run.
            motionMatBufferAllIdentity_[frame] = false;
            return true;
        }

        // Same dance as ensureMotionMatCapacity, but for the per-mesh
        // moved-bitmask SSBO at binding 21. `neededWords` is the number of
        // 32-bit words required to address every visible TLAS instance.
        bool ensureMeshMovedBitsCapacity(uint32_t frame, uint32_t neededWords) {
            if (neededWords <= meshMovedBitsBufferCapacity[frame]) return false;
            const uint32_t newCap = std::max<uint32_t>(
                    neededWords,
                    static_cast<uint32_t>(meshMovedBitsBufferCapacity[frame] * 2u));
            destroyBuffer(ctx->allocator(), meshMovedBitsBuffers[frame]);
            meshMovedBitsBuffers[frame] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    /*size*/ newCap * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            meshMovedBitsBufferCapacity[frame] = newCap;
            return true;
        }

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

        void updateCameraUbo(uint32_t frame, Camera& camera) {
            camera.updateMatrixWorld(true);

            float data[36];
            std::memcpy(data + 0,  camera.matrixWorld->elements.data(), 64);
            // Reverse-Z projInverse (matches the reverse-Z VP the raster uses) so
            // depth reconstruction (deferred worldFromDepth, hybrid primary) is
            // consistent with the reverse-Z depth buffer.
            Matrix4 vkProjInv = reverseZVk(camera.projectionMatrix);
            vkProjInv.invert();
            std::memcpy(data + 16, vkProjInv.elements.data(), 64);

            // Per-frame Halton(2,3) jitter — must match what uploadRasterCameraUbo
            // computes for THIS frame so raygen's hybrid primary direction lands
            // on the same surface the raster jittered to. uploadRasterCameraUbo
            // runs after this and uses the same `haltonFrame_` value before
            // incrementing, so both ubos see identical jitter. Without this,
            // raygen reconstructs V from pixel-center every frame → reflection
            // direction frozen → low-roughness metals show "lines" because TAA
            // accumulates the same env tap each frame instead of integrating.
            // Render extent: the jitter is a sub-pixel offset, so the pixel
            // size in clip space (2/width) must use the resolution raygen +
            // the raster gbuffer actually run at, not the swapchain extent.
            const VkExtent2D ext = renderExtent();
            const uint32_t hi = haltonFrame_ + 1u;
            const float jx = halton_(hi, 2) - 0.5f;
            const float jy = halton_(hi, 3) - 0.5f;
            const float jClipX = 2.f * jx / float(ext.width);
            const float jClipY = 2.f * jy / float(ext.height);
            data[32] = jClipX;
            data[33] = jClipY;
            // .zw = previous frame's jitter so raygen's hybrid reproject can
            // correct the bilinear tap by (prev - curr) pixels. The raster
            // path tracks this in rasterPrevJitter_; PT UBO upload runs before
            // the raster upload, so rasterPrevJitter_ here still holds the
            // PREVIOUS frame's value (which is exactly what we want). First
            // frame: self-seed to curr so the delta is zero.
            data[34] = rasterPrevJitterValid_ ? rasterPrevJitter_[0] : jClipX;
            data[35] = rasterPrevJitterValid_ ? rasterPrevJitter_[1] : jClipY;

            // Build camera basis-vector buffer matching PrevCameraUbo layout.
            // matrixWorld column-major: col0=right, col1=up, col2=backward(-fwd), col3=pos.
            const auto& wm = camera.matrixWorld->elements;
            const auto& pm = camera.projectionMatrix.elements;
            float curBuf[16];
            curBuf[0] = wm[12]; curBuf[1] = wm[13]; curBuf[2] = wm[14]; curBuf[3] = pm[0];
            curBuf[4] =-wm[ 8]; curBuf[5] =-wm[ 9]; curBuf[6] =-wm[10]; curBuf[7] = pm[5];
            curBuf[8] = wm[ 0]; curBuf[9] = wm[ 1]; curBuf[10]= wm[ 2]; curBuf[11]= 0.0f;
            curBuf[12]= wm[ 4]; curBuf[13]= wm[ 5]; curBuf[14]= wm[ 6]; curBuf[15]= 0.0f;

            // Camera-motion detection: position [0..2], forward [4..6], and
            // projection scale [3]=projScaleX / [7]=projScaleY. The projection
            // terms catch FOV and aspect-ratio changes that don't move the camera
            // — without them a resize or FOV tweak would leave motionThisFrame_
            // false and the shader would reuse prev-frame history with the old
            // projection basis, producing incorrect reprojection for one frame.
            if (prevCameraValid) {
                const float dx = curBuf[0] - prevCamBufData_[0];
                const float dy = curBuf[1] - prevCamBufData_[1];
                const float dz = curBuf[2] - prevCamBufData_[2];
                const float fx = curBuf[4] - prevCamBufData_[4];
                const float fy = curBuf[5] - prevCamBufData_[5];
                const float fz = curBuf[6] - prevCamBufData_[6];
                const float sx = curBuf[3] - prevCamBufData_[3];// projScaleX
                const float sy = curBuf[7] - prevCamBufData_[7];// projScaleY
                // Position threshold lowered 1e-6 → 1e-10 (|Δpos| 1e-3 → 1e-5
                // world units). The old 1e-3 was far too coarse for PURE
                // TRANSLATION: a slow pan moves the camera only ~1e-4 units/frame,
                // which fell under 1e-3 → cameraMoved stayed false → the
                // accumulator took the SELF-TAP path (reuse prev accum at the
                // SAME pixel, full FC). But the camera HAD moved, so that reused
                // image sat slightly off and dragged with the camera — the "old
                // image follows the camera" smear, textured-surface-only because
                // uniform albedo hides a misplaced reuse. Orbit never hit this:
                // rotation moves the forward vector, which trips the far more
                // sensitive 1e-8 rotation threshold, so orbit always took the
                // correct reproject path. 1e-5 units is still well above static
                // float noise (an untouched camera has an identical matrix →
                // exactly zero delta) but catches the slowest deliberate pan.
                if (dx*dx + dy*dy + dz*dz > 1e-10f ||
                    fx*fx + fy*fy + fz*fz > 1e-8f ||
                    sx*sx + sy*sy > 1e-10f) {
                    motionThisFrame_ = true;
                    cameraMovedThisFrame_ = true;
                    // sampleIndex must keep advancing — see comment in
                    // ensureSceneBuilt's matrix-changed path. Per-pixel FC
                    // halving handles convergence; freezing the seed kills
                    // Monte Carlo variance reduction.
                }
            }

            // Upload prev camera data. Self-seed on first frame so cold-start
            // is a no-op reproject (same pixel, fc grows from 0 normally).
            const float* toUpload = prevCameraValid ? prevCamBufData_.data() : curBuf;
            void* mappedPrev = nullptr;
            vmaMapMemory(ctx->allocator(), prevCameraUbos[frame].alloc, &mappedPrev);
            std::memcpy(mappedPrev, toUpload, 16 * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), prevCameraUbos[frame].alloc);

            std::memcpy(prevCamBufData_.data(), curBuf, 16 * sizeof(float));
            prevCameraValid = true;

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), cameraUbos[frame].alloc, &mapped);
            std::memcpy(mapped, data, sizeof(data));
            vmaUnmapMemory(ctx->allocator(), cameraUbos[frame].alloc);
        }

        // ── Hybrid raster runtime helpers ───────────────────────────────────
        // (halton_ is also used by updateCameraUbo above to mirror the raster's
        // jitter into the raygen camera UBO — keep both helpers reachable.)
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
        void cacheCullFlags(const std::vector<MaterialDesc>& mds) {
            lastVisibleCullMode_.resize(mds.size());
            for (size_t i = 0; i < mds.size(); ++i) {
                switch (mds[i].sideMode) {
                    case 0:  lastVisibleCullMode_[i] = VK_CULL_MODE_BACK_BIT;  break;
                    case 1:  lastVisibleCullMode_[i] = VK_CULL_MODE_FRONT_BIT; break;
                    default: lastVisibleCullMode_[i] = VK_CULL_MODE_NONE;      break;
                }
            }
        }

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
        void uploadRasterCameraUbo(uint32_t frame, Camera& camera) {
            camera.updateMatrixWorld(true);

            // VP_unjittered = projection * view, view = matrixWorldInverse.
            Matrix4 view, proj;
            std::memcpy(view.elements.data(),
                        camera.matrixWorldInverse.elements.data(), 64);
            // Reverse-Z projection (near→1, far→0) — feeds the gbuffer VP AND
            // currVPunjit_ (overlay depth prepass). Depth clear + compares flipped
            // to match (see recordRasterGbufPass clears + pipeline depthCompareOp).
            proj = reverseZVk(camera.projectionMatrix);
            Matrix4 vpUnj;
            vpUnj.multiplyMatrices(proj, view);

            // Render extent — jitter + the .zw = 1/resolution this writes
            // into the raster camera UBO must match the resolution the
            // gbuffer rasterizes at (see updateCameraUbo for the rationale).
            const VkExtent2D ext = renderExtent();
            // Sub-pixel offset in [-0.5, +0.5] per axis. Halton(2,3) is the
            // industry-standard low-discrepancy sequence for primary AA.
            const uint32_t hi = haltonFrame_ + 1u;
            const float jx = halton_(hi, 2) - 0.5f;
            const float jy = halton_(hi, 3) - 0.5f;
            // Map sub-pixel offset to clip-space: one pixel spans 2/width of
            // NDC (NDC ∈ [-1, +1]), so a 1-pixel jitter is 2/width in clip x.
            //
            // Raster jitter trade-off: ON gives sub-pixel-offset texture AA
            // on interior surfaces via PT temporal blend, but the per-frame
            // Halton coverage flip at silhouettes shows up as black-pixel
            // stippling on moving objects (1-spp env taps on the "uncovered"
            // half-frame). Without MSAA on the gbuffer pass, that aliasing
            // costs more than the interior AA gains. Disabled by default;
            // sub-pixel jitter still happens in raygen's hybrid primary via
            // the blue-noise tile (see primaryDirHybrid), so interior AA
            // doesn't disappear — only the coverage jitter is gone.
            // Raster jitter ON: sub-pixel Halton coverage +
            // temporal accumulation = proper TAA/TSR (replaces FXAA). The earlier
            // "flicker" was silhouette coverage-flip on MOVING objects with a loose
            // RGB AABB history clamp; the resolve now uses a YCoCg variance clip
            // (tighter history rejection) to suppress it. Static views converge to
            // clean AA over the 16-frame Halton cycle.
            constexpr bool kRasterJitterEnabled = true;
            const float jClipX = kRasterJitterEnabled ? 2.f * jx / float(ext.width)  : 0.f;
            const float jClipY = kRasterJitterEnabled ? 2.f * jy / float(ext.height) : 0.f;

            // Apply jitter by shifting the projection matrix's m02/m12 (the
            // entries that translate the projected NDC). For a column-major
            // 4x4 stored in elements[c*4 + r], that's elements[8] (col=2,row=0)
            // and elements[9] (col=2,row=1).
            Matrix4 projJ;
            std::memcpy(projJ.elements.data(), proj.elements.data(), 64);
            projJ.elements[8]  += jClipX;
            projJ.elements[9]  += jClipY;
            Matrix4 vpJ;
            vpJ.multiplyMatrices(projJ, view);

            RasterCameraData ubo{};
            std::memcpy(ubo.currVPjittered,   vpJ.elements.data(),  64);
            std::memcpy(ubo.currVPunjittered, vpUnj.elements.data(), 64);
            // Mirror to Impl-level cache for recordOverlayPass — the overlay
            // pass runs in recordCommandBuffer which doesn't have direct
            // access to the camera; it needs the same unjittered VP that
            // the raster prepass + TAA used so wireframes register pixel-
            // exact with the post-TAA path-traced silhouette.
            std::memcpy(currVPunjit_.data(), vpUnj.elements.data(), 64);
            // First frame: self-seed prevVP so motion vectors are zero. The
            // following frame picks up the real history.
            std::memcpy(ubo.prevVP,
                        rasterPrevVPValid_ ? rasterPrevVP_ : vpUnj.elements.data(),
                        64);
            ubo.jitter[0] = jClipX;
            ubo.jitter[1] = jClipY;
            ubo.jitter[2] = 1.f / float(ext.width);
            ubo.jitter[3] = 1.f / float(ext.height);
            // First frame: self-seed prev jitter to curr so motion vec for
            // static surfaces is exactly zero (no spurious offset on cold start).
            ubo.prevJitter[0] = rasterPrevJitterValid_ ? rasterPrevJitter_[0] : jClipX;
            ubo.prevJitter[1] = rasterPrevJitterValid_ ? rasterPrevJitter_[1] : jClipY;
            ubo.prevJitter[2] = 0.f;
            ubo.prevJitter[3] = 0.f;

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), rasterCameraUbos[frame].alloc, &mapped);
            std::memcpy(mapped, &ubo, sizeof(ubo));
            vmaUnmapMemory(ctx->allocator(), rasterCameraUbos[frame].alloc);

            // Far-plane reprojection for the TAA sky path — must be built
            // BEFORE rasterPrevVP_ rolls over to this frame's VP.
            {
                Matrix4 invCurr;
                invCurr.copy(vpUnj).invert();
                Matrix4 prevVPm;
                std::memcpy(prevVPm.elements.data(),
                            rasterPrevVPValid_ ? rasterPrevVP_ : vpUnj.elements.data(), 64);
                Matrix4 sky;
                sky.multiplyMatrices(prevVPm, invCurr);
                std::memcpy(taaSkyReproj_.data(), sky.elements.data(), 64);
            }
            std::memcpy(rasterPrevVP_, vpUnj.elements.data(), 64);
            rasterPrevVPValid_ = true;
            rasterPrevJitter_[0] = jClipX;
            rasterPrevJitter_[1] = jClipY;
            rasterPrevJitterValid_ = true;
            // Camera world motion (translation + forward rotation) for the
            // deferred reflection policy — see the member comment.
            {
                const auto& mw = camera.matrixWorld->elements;
                const float pos[3] = {mw[12], mw[13], mw[14]};
                const float fwd[3] = {-mw[8], -mw[9], -mw[10]};// -Z column = view dir
                if (deferredCamPrevValid_) {
                    const float dx = pos[0] - deferredCamPrevPos_[0];
                    const float dy = pos[1] - deferredCamPrevPos_[1];
                    const float dz = pos[2] - deferredCamPrevPos_[2];
                    deferredCamDeltaLen_ = std::sqrt(dx * dx + dy * dy + dz * dz);
                    const float c = std::clamp(fwd[0] * deferredCamPrevFwd_[0] +
                                                       fwd[1] * deferredCamPrevFwd_[1] +
                                                       fwd[2] * deferredCamPrevFwd_[2],
                                               -1.f, 1.f);
                    deferredCamRotAngle_ = std::acos(c);
                } else {
                    deferredCamDeltaLen_ = 0.f;
                    deferredCamRotAngle_ = 0.f;
                }
                std::memcpy(deferredCamPrevPos_, pos, sizeof(pos));
                std::memcpy(deferredCamPrevFwd_, fwd, sizeof(fwd));
                deferredCamPrevValid_ = true;
            }
            // 16-frame Halton cycle: longer than the eye notices, short
            // enough that any drift in the host counter never overflows.
            haltonFrame_ = (haltonFrame_ + 1u) & 15u;

            // Refresh the per-frame descriptor set. Bindings 0-2 (UBO,
            // motionMat, matDescs) are tiny, and motionMat/matDescs handles can
            // grow (handle change) when the scene instance/material count
            // increases — rewriting them every frame absorbs that automatically.
            VkDescriptorBufferInfo ubInfo{};
            ubInfo.buffer = rasterCameraUbos[frame].handle;
            ubInfo.offset = 0;
            ubInfo.range  = sizeof(RasterCameraData);

            VkDescriptorBufferInfo mmInfo{};
            mmInfo.buffer = motionMatBuffers[frame].handle;
            mmInfo.offset = 0;
            mmInfo.range  = VK_WHOLE_SIZE;

            VkDescriptorBufferInfo matsInfo{};
            matsInfo.buffer = materialDescsBuffers[frame].handle;
            matsInfo.offset = 0;
            matsInfo.range  = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[4]{};
            writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet          = rasterDescSets[frame];
            writes[0].dstBinding      = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo     = &ubInfo;
            writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet          = rasterDescSets[frame];
            writes[1].dstBinding      = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo     = &mmInfo;
            writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet          = rasterDescSets[frame];
            writes[2].dstBinding      = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2].pBufferInfo     = &matsInfo;
            uint32_t writeCount = 3;

            // Binding 3 — the bindless material-texture array (same VkImage/
            // VkSampler handles raygen sees at its own binding 8, white-default
            // padded). Its contents are identical frame-to-frame and only
            // change when the scene texture table is rebuilt, which sets
            // rasterMatTexValid_ to 0 (see the member declaration). Skip the
            // 2048-element write + the ~48 KB fillMaterialTextureInfos array
            // fill on every frame the table is unchanged for this slot (the
            // common case). matTexInfos must outlive the vkUpdateDescriptorSets
            // call below, so it is declared here in function scope.
            std::array<VkDescriptorImageInfo, kMaxMaterialTextures> matTexInfos{};
            if (rasterMatTexValid_[frame] == 0) {
                fillMaterialTextureInfos(matTexInfos);
                writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[3].dstSet          = rasterDescSets[frame];
                writes[3].dstBinding      = 3;
                writes[3].dstArrayElement = 0;
                writes[3].descriptorCount = kMaxMaterialTextures;
                writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[3].pImageInfo      = matTexInfos.data();
                writeCount = 4;
                rasterMatTexValid_[frame] = 1;
            }
            vkUpdateDescriptorSets(ctx->device(), writeCount, writes, 0, nullptr);
        }

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
            uint32_t instanceCustomIndex;
            uint32_t flags;
            uint32_t indexed;
            float    polygonOffset;    // clip-z depth bias (reverse-Z: + = toward near = on top)
        };
        static_assert(sizeof(DrawInfoGpu) == 120,
                      "DrawInfoGpu layout drifted from gbuffer_indirect.vert");

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

        bool ensureDrawInfoCapacity(uint32_t frame, VkDeviceSize neededBytes) {
            if (neededBytes <= drawInfoBufferCapacity[frame]) return false;
            const VkDeviceSize newCap = std::max<VkDeviceSize>(
                    neededBytes, drawInfoBufferCapacity[frame] * 2u);
            destroyBuffer(ctx->allocator(), drawInfoBuffers[frame]);
            drawInfoBuffers[frame] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    newCap,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            drawInfoBufferCapacity[frame] = newCap;
            return true;
        }

        bool ensureIndirectCmdCapacity(uint32_t frame, VkDeviceSize neededBytes) {
            if (neededBytes <= indirectCmdBufferCapacity[frame]) return false;
            const VkDeviceSize newCap = std::max<VkDeviceSize>(
                    neededBytes, indirectCmdBufferCapacity[frame] * 2u);
            destroyBuffer(ctx->allocator(), indirectCmdBuffers[frame]);
            indirectCmdBuffers[frame] = createBuffer(
                    ctx->allocator(), ctx->device(),
                    newCap,
                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            indirectCmdBufferCapacity[frame] = newCap;
            return true;
        }

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
        void buildIndirectDrawData(uint32_t frame) {
            for (auto& g : indirectGroups_) { g.offset = 0; g.count = 0; }
            indirectTotalDraws_ = 0;

            if (lastVisibleEntries_.empty()) return;

            // Four buckets: [BACK_cull, FRONT_cull, NONE_cull, decal] order.
            std::array<std::vector<DrawInfoGpu>, 4>            draws;
            std::array<std::vector<VkDrawIndirectCommand>, 4>  cmds;
            auto bucketOf = [](VkCullModeFlags cm) -> int {
                if (cm == VK_CULL_MODE_BACK_BIT)  return 0;
                if (cm == VK_CULL_MODE_FRONT_BIT) return 1;
                return 2;
            };

            uint32_t globalIdx = 0;
            for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
                const auto& en = lastVisibleEntries_[i];
                if (en.isOverlay)  continue;
                if (!en.inFrustum) continue;
                const BlasRecord* rec = resolveBlasForEntry(en);
                if (!rec || rec->vertex.handle == VK_NULL_HANDLE) continue;

                const bool indexed = (rec->index.handle != VK_NULL_HANDLE);
                const uint32_t vcount = indexed ? rec->indexCount : rec->vertexCount;
                if (vcount == 0u) continue;

                const VkCullModeFlags wantCull =
                        (i < lastVisibleCullMode_.size())
                                ? lastVisibleCullMode_[i]
                                : VK_CULL_MODE_BACK_BIT;
                // Blend decals (alphaCutoff == -2 sentinel from materialFromMesh)
                // go to bucket [3]: drawn last, with the albedo-blend pipeline.
                // RasterFirst only — its shade reads the blended albedo
                // attachment. ReferencePT's hybrid raygen re-samples material
                // textures via the ids/UV attachments (which the decal pipeline
                // write-masks), so there decals stay on the stochastic
                // screen-door path that the PT accumulator converges.
                const bool isDecal = renderMode_ == VulkanRenderer::RenderMode::RasterFirst &&
                                     (i < matDescsCached_.size()) &&
                                     (matDescsCached_[i].alphaCutoff == -2.0f);
                const int b = isDecal ? 3 : bucketOf(wantCull);

                DrawInfoGpu di{};
                std::memcpy(di.model, en.worldMatrix.data(), 64);
                di.posAddr     = rec->vertex.address;
                di.nrmAddr     = rec->normal.address;
                di.uvAddr      = (rec->uv.handle != VK_NULL_HANDLE) ? rec->uv.address : 0ull;
                // Warp-enabled DisplacedMesh: vertices reflow every frame, so
                // prevVertex can't track a stable world point — point the
                // motion-vector source at the current buffer so the ocean
                // reprojects as world-static (see prevVertexAddress in the
                // GeometryDesc build for the full rationale).
                bool warpReproject = false;
                if (en.isDisplaced) {
                    warpReproject = static_cast<DisplacedMesh*>(en.mesh)->warp.halfRange > 0.0f;
                }
                di.prevPosAddr = (rec->prevVertex.handle != VK_NULL_HANDLE && !warpReproject)
                                         ? rec->prevVertex.address
                                         : rec->vertex.address;
                di.indexAddr   = indexed ? rec->index.address : 0ull;
                di.instanceCustomIndex = static_cast<uint32_t>(i);
                // Flag bits match the old gbuffer.vert push-constant layout:
                //   bit 0 = is_water (DisplacedMesh), bit 3 = is_skinned.
                uint32_t flags = 0u;
                if (en.isDisplaced) flags |= 1u;
                if (en.isSkinned)   flags |= 8u;
                di.flags   = flags;
                di.indexed = indexed ? 1u : 0u;
                // polygonOffset → per-mesh clip-z depth bias (decals). Reverse-Z:
                // a +clip-z bias pushes the surface toward NEAR so it renders on
                // top of coplanar geometry (no z-fight). threepp/GL uses NEGATIVE
                // polygonOffsetUnits for "on top", so flip the sign; units==0
                // (bool only) → a small default that clears z-fight without
                // floating. (The slope-scaled `factor` term isn't applied — flat
                // decals don't need it; bias is uniform in NDC.)
                float polyOff = 0.f;
                if (auto pm = en.mesh->material(); pm && pm->polygonOffset) {
                    // Honor BOTH factor + units (combined as a constant NDC bias —
                    // the slope term can't be evaluated in the vertex shader; flat
                    // decals don't need it). 0/0 (bool only) → a small default.
                    const float uf = pm->polygonOffsetUnits + pm->polygonOffsetFactor;
                    polyOff = (uf != 0.f) ? (-uf * 1.0e-6f) : 4.0e-6f;
                }
                di.polygonOffset = polyOff;
                draws[b].push_back(di);

                VkDrawIndirectCommand cmd{};
                cmd.vertexCount   = vcount;
                cmd.instanceCount = 1u;
                cmd.firstVertex   = 0u;
                cmd.firstInstance = 0u;// patched below to the final-position index
                cmds[b].push_back(cmd);
                ++globalIdx;
            }

            indirectTotalDraws_ = globalIdx;
            if (globalIdx == 0u) return;

            // Concatenate buckets into the per-frame device buffers.
            const VkDeviceSize drawBytes = sizeof(DrawInfoGpu) * globalIdx;
            const VkDeviceSize cmdBytes  = sizeof(VkDrawIndirectCommand) * globalIdx;
            const bool drawGrown = ensureDrawInfoCapacity(frame, drawBytes);
            ensureIndirectCmdCapacity(frame, cmdBytes);

            void* mappedDraws = nullptr;
            vmaMapMemory(ctx->allocator(), drawInfoBuffers[frame].alloc, &mappedDraws);
            void* mappedCmds = nullptr;
            vmaMapMemory(ctx->allocator(), indirectCmdBuffers[frame].alloc, &mappedCmds);

            uint8_t* dDst = static_cast<uint8_t*>(mappedDraws);
            uint8_t* cDst = static_cast<uint8_t*>(mappedCmds);
            uint32_t offset = 0;
            const VkCullModeFlags cullForBucket[4] = {
                    VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT, VK_CULL_MODE_NONE,
                    VK_CULL_MODE_NONE// decals: DecalGeometry wraps edges, don't cull
            };
            for (int b = 0; b < 4; ++b) {
                const uint32_t n = static_cast<uint32_t>(draws[b].size());
                indirectGroups_[b].cullMode = cullForBucket[b];
                indirectGroups_[b].offset   = offset;
                indirectGroups_[b].count    = n;
                if (n > 0u) {
                    // Patch firstInstance now that we know each draw's final
                    // position in the concatenated DrawInfo / indirect arrays.
                    // Entry-order assignment up above doesn't match concat
                    // order once we partition by cull mode — the VS reads
                    // `draws[gl_InstanceIndex]` so the encoded index has to
                    // be the FINAL position, not the bucket-order index.
                    for (uint32_t k = 0; k < n; ++k) {
                        cmds[b][k].firstInstance = offset + k;
                    }
                    std::memcpy(dDst + offset * sizeof(DrawInfoGpu),
                                draws[b].data(), n * sizeof(DrawInfoGpu));
                    std::memcpy(cDst + offset * sizeof(VkDrawIndirectCommand),
                                cmds[b].data(), n * sizeof(VkDrawIndirectCommand));
                }
                offset += n;
            }

            vmaUnmapMemory(ctx->allocator(), indirectCmdBuffers[frame].alloc);
            vmaUnmapMemory(ctx->allocator(), drawInfoBuffers[frame].alloc);

            // Rewrite binding 4 if the DrawInfo buffer handle moved (grow).
            // The indirect cmd buffer is consumed by vkCmdDrawIndirect — no
            // descriptor binding needed for it.
            if (drawGrown) {
                VkDescriptorBufferInfo dbInfo{};
                dbInfo.buffer = drawInfoBuffers[frame].handle;
                dbInfo.offset = 0;
                dbInfo.range  = VK_WHOLE_SIZE;
                VkWriteDescriptorSet w{};
                w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet          = rasterDescSets[frame];
                w.dstBinding      = 4;
                w.descriptorCount = 1;
                w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo     = &dbInfo;
                vkUpdateDescriptorSets(ctx->device(), 1, &w, 0, nullptr);
            }
        }

        // Begin the raster G-buffer render pass and ship the prebuilt
        // indirect-draw groups via 1-4 vkCmdDrawIndirect calls (one per
        // active cull mode, plus a trailing blend-decal group on the
        // albedo-blend pipeline). Replaces the prior per-mesh draw loop —
        // see buildIndirectDrawData above for how the GPU buffers are
        // populated.
        void recordRasterGbufPass(VkCommandBuffer cb, uint32_t frame) {
            // Render extent — the gbuffer attachments are sized to it and
            // raygen launches over it; the viewport must agree.
            const VkExtent2D ext = renderExtent();
            const auto& g = rasterGbufs[frame];
            if (g.framebuffer == VK_NULL_HANDLE) return;// not initialized

            VkClearValue clears[6]{};
            clears[0].color = {{0.f, 0.f, 0.f, 0.f}};   // normal — sky/miss as zero
            clears[1].color = {{0.f, 0.f, 0.f, 0.f}};   // motion
            clears[2].color.uint32[0] = 0u;             // instanceID — 0 reserved as sky
            clears[2].color.uint32[1] = 0u;
            clears[2].color.uint32[2] = 0u;
            clears[2].color.uint32[3] = 0u;
            clears[3].color = {{0.f, 0.f, 0.f, 0.f}};   // uv — sky has no UV
            clears[4].color = {{0.f, 0.f, 0.f, 0.f}};   // albedo+metalness — sky as zero
            clears[5].depthStencil = {0.f, 0u};// reverse-Z: clear depth to 0 (far)

            VkRenderPassBeginInfo rpbi{};
            rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rpbi.renderPass      = rasterGbufRenderPass;
            rpbi.framebuffer     = g.framebuffer;
            rpbi.renderArea.offset = {0, 0};
            rpbi.renderArea.extent = {g.width, g.height};
            rpbi.clearValueCount = 6;
            rpbi.pClearValues    = clears;
            vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport{};
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width  = float(ext.width);
            viewport.height = float(ext.height);
            viewport.minDepth = 0.f;
            viewport.maxDepth = 1.f;
            vkCmdSetViewport(cb, 0, 1, &viewport);
            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = ext;
            vkCmdSetScissor(cb, 0, 1, &scissor);

            if (indirectTotalDraws_ == 0u) {
                vkCmdEndRenderPass(cb);
                return;
            }

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterGbufIndirectPipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    rasterPipelineLayout, 0, 1,
                                    &rasterDescSets[frame], 0, nullptr);

            // One vkCmdDrawIndirect per cull-mode group. Empty groups skip;
            // a typical static scene fires one (BACK_cull); transmissive
            // sets can fire two or three. cullMode flips between calls
            // via the dynamic cullMode state — far cheaper than re-doing
            // it per draw.
            constexpr VkDeviceSize cmdStride = sizeof(VkDrawIndirectCommand);
            for (int b = 0; b < 3; ++b) {
                const auto& g = indirectGroups_[b];
                if (g.count == 0u) continue;
                vkCmdSetCullMode(cb, g.cullMode);
                vkCmdDrawIndirect(cb,
                                  indirectCmdBuffers[frame].handle,
                                  static_cast<VkDeviceSize>(g.offset) * cmdStride,
                                  g.count,
                                  static_cast<uint32_t>(cmdStride));
            }
            // Blend decals last: albedo-blend pipeline lerps over the receivers
            // rasterized above (same layout/descriptors, no set rebind needed).
            if (const auto& g = indirectGroups_[3]; g.count > 0u) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterGbufDecalPipeline);
                vkCmdSetCullMode(cb, g.cullMode);
                vkCmdDrawIndirect(cb,
                                  indirectCmdBuffers[frame].handle,
                                  static_cast<VkDeviceSize>(g.offset) * cmdStride,
                                  g.count,
                                  static_cast<uint32_t>(cmdStride));
            }

            vkCmdEndRenderPass(cb);
        }

        // Debug visualization: blit one of the G-buffer color attachments to
        // the swapchain image so we can inspect raster output without wiring
        // raygen to read it. Normal channel is the most informative — solid
        // surfaces show their world-space orientation as RGB.
        void recordHybridDebugBlit(VkCommandBuffer cb, uint32_t imageIndex, uint32_t frame) {
            const auto& g = rasterGbufs[frame];
            VkImage src = VK_NULL_HANDLE;
            VkImageLayout srcLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            switch (hybridDebugView_) {
                case HybridDebugView::Normal: src = g.normal.image; break;
                case HybridDebugView::Motion: src = g.motion.image; break;
                case HybridDebugView::Depth:  src = g.depth.image;
                                              srcLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                                              break;
                case HybridDebugView::Ids:    src = g.ids.image; break;
                case HybridDebugView::Albedo: src = g.albedo.image; break;
                default: return;
            }
            if (src == VK_NULL_HANDLE) return;

            VkImage dst = ctx->swapchainImages()[imageIndex];

            // Transition src to TRANSFER_SRC_OPTIMAL, dst (already in GENERAL
            // from raygen path's preBarriers) to TRANSFER_DST_OPTIMAL.
            VkImageMemoryBarrier2 toSrc{};
            toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toSrc.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            toSrc.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            toSrc.dstStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
            toSrc.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            toSrc.oldLayout = srcLayout;
            toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.image = src;
            toSrc.subresourceRange.aspectMask =
                    (hybridDebugView_ == HybridDebugView::Depth)
                            ? VK_IMAGE_ASPECT_DEPTH_BIT
                            : VK_IMAGE_ASPECT_COLOR_BIT;
            toSrc.subresourceRange.levelCount = 1;
            toSrc.subresourceRange.layerCount = 1;

            VkImageMemoryBarrier2 toDst{};
            toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toDst.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            toDst.srcAccessMask = 0;
            toDst.dstStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
            toDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.image = dst;
            toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toDst.subresourceRange.levelCount = 1;
            toDst.subresourceRange.layerCount = 1;

            VkImageMemoryBarrier2 pre[2] = {toSrc, toDst};
            VkDependencyInfo depPre{};
            depPre.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depPre.imageMemoryBarrierCount = 2;
            depPre.pImageMemoryBarriers = pre;
            vkCmdPipelineBarrier2(cb, &depPre);

            const VkExtent2D ext = ctx->swapchainExtent();
            VkImageBlit blit{};
            blit.srcSubresource.aspectMask = toSrc.subresourceRange.aspectMask;
            blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[1] = {int32_t(g.width), int32_t(g.height), 1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.layerCount = 1;
            blit.dstOffsets[1] = {int32_t(ext.width), int32_t(ext.height), 1};
            // For depth, the swapchain doesn't accept a depth blit directly;
            // depth is informational so we'd need a tiny resolve shader. Skip
            // that for stage 1 — Normal/Motion/Ids are the visually useful views.
            if (hybridDebugView_ != HybridDebugView::Depth) {
                vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &blit, VK_FILTER_NEAREST);
            }

            // Restore src attachment to its original SHADER_READ layout (so
            // raygen can sample on the next frame). Leave dst in
            // TRANSFER_DST_OPTIMAL — the caller in recordCommandBuffer
            // handles the overlay (ImGui) draw and the final PRESENT_SRC
            // transition uniformly with the existing PT path.
            VkImageMemoryBarrier2 toShaderRead{};
            toShaderRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toShaderRead.srcStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
            toShaderRead.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            toShaderRead.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            toShaderRead.dstAccessMask = 0;
            toShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toShaderRead.newLayout = srcLayout;
            toShaderRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.image = src;
            toShaderRead.subresourceRange.aspectMask = toSrc.subresourceRange.aspectMask;
            toShaderRead.subresourceRange.levelCount = 1;
            toShaderRead.subresourceRange.layerCount = 1;

            VkDependencyInfo depPost{};
            depPost.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depPost.imageMemoryBarrierCount = 1;
            depPost.pImageMemoryBarriers = &toShaderRead;
            vkCmdPipelineBarrier2(cb, &depPost);
        }

        void createLightsUbos() {
            for (auto& b : lightsUbos) {
                b = createBuffer(
                        ctx->allocator(), ctx->device(),
                        sizeof(GpuLightsUbo),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
            }
        }

        // Walk the scene each frame for AmbientLight + DirectionalLight; pack
        // into the per-frame lights UBO. Direction is computed from the
        // light's world-space position toward its (possibly defaulted) target,
        // mirroring three.js's DirectionalLight.target convention. The shader
        // expects the L vector (toward the light), so we negate.
        void updateLightsUbo(uint32_t frame, Object3D& scene) {
            // force=false: ensureSceneBuilt already brought the graph current
            // this frame — the forced variant here was a SECOND full
            // world-multiply pass over every node, every frame.
            scene.updateMatrixWorld();

            GpuLightsUbo ubo{};

            // traverseVisible so a hidden parent prunes its child lights too.
            scene.traverseVisible([&](Object3D& o) {
                if (auto* a = dynamic_cast<AmbientLight*>(&o)) {
                    ubo.ambient[0] += a->color.r * a->intensity;
                    ubo.ambient[1] += a->color.g * a->intensity;
                    ubo.ambient[2] += a->color.b * a->intensity;
                } else if (auto* dl = dynamic_cast<DirectionalLight*>(&o)) {
                    if (ubo.dirCount >= kMaxDirLights) return;
                    Vector3 lp, tp;
                    dl->getWorldPosition(lp);
                    const_cast<Object3D&>(dl->target()).getWorldPosition(tp);
                    Vector3 toLight = lp.sub(tp);
                    if (toLight.lengthSq() < 1e-12f) toLight.set(0.f, 1.f, 0.f);
                    toLight.normalize();
                    auto& g = ubo.dirLights[ubo.dirCount++];
                    g.direction[0] = toLight.x; g.direction[1] = toLight.y; g.direction[2] = toLight.z;
                    g.color[0] = dl->color.r * dl->intensity;
                    g.color[1] = dl->color.g * dl->intensity;
                    g.color[2] = dl->color.b * dl->intensity;
                } else if (auto* pl = dynamic_cast<PointLight*>(&o)) {
                    if (ubo.pointCount >= kMaxPointLights) return;
                    Vector3 wp; pl->getWorldPosition(wp);
                    auto& g = ubo.pointLights[ubo.pointCount++];
                    g.position[0] = wp.x; g.position[1] = wp.y; g.position[2] = wp.z;
                    g.range = pl->distance;
                    g.color[0] = pl->color.r * pl->intensity;
                    g.color[1] = pl->color.g * pl->intensity;
                    g.color[2] = pl->color.b * pl->intensity;
                    g.decay = pl->decay;
                } else if (auto* sl = dynamic_cast<SpotLight*>(&o)) {
                    if (ubo.spotCount >= kMaxSpotLights) return;
                    Vector3 lp, tp;
                    sl->getWorldPosition(lp);
                    const_cast<Object3D&>(sl->target()).getWorldPosition(tp);
                    Vector3 emDir = tp - lp;
                    if (emDir.lengthSq() < 1e-12f) emDir.set(0.f, -1.f, 0.f);
                    emDir.normalize();
                    auto& g = ubo.spotLights[ubo.spotCount++];
                    g.position[0] = lp.x; g.position[1] = lp.y; g.position[2] = lp.z;
                    g.range = sl->distance;
                    g.color[0] = sl->color.r * sl->intensity;
                    g.color[1] = sl->color.g * sl->intensity;
                    g.color[2] = sl->color.b * sl->intensity;
                    g.decay = sl->decay;
                    g.direction[0] = emDir.x; g.direction[1] = emDir.y; g.direction[2] = emDir.z;
                    g.cosAngleOuter = std::cos(sl->angle);
                    g.cosAngleInner = std::cos(sl->angle * (1.0f - sl->penumbra));
                } else if (auto* rl = dynamic_cast<RectAreaLight*>(&o)) {
                    if (ubo.rectCount >= kMaxRectLights) return;
                    Vector3 wp; rl->getWorldPosition(wp);
                    const auto& el = rl->matrixWorld->elements;
                    // Column-major: col0=localX, col1=localY, col2=localZ (each with scale).
                    Vector3 worldX(el[0], el[1], el[2]); worldX.normalize();
                    Vector3 worldY(el[4], el[5], el[6]); worldY.normalize();
                    Vector3 worldZ(el[8], el[9], el[10]); worldZ.normalize();
                    auto& g = ubo.rectLights[ubo.rectCount++];
                    g.position[0] = wp.x; g.position[1] = wp.y; g.position[2] = wp.z;
                    const float hw = rl->width  * 0.5f;
                    const float hh = rl->height * 0.5f;
                    g.halfU[0] = worldX.x * hw; g.halfU[1] = worldX.y * hw; g.halfU[2] = worldX.z * hw;
                    g.halfV[0] = worldY.x * hh; g.halfV[1] = worldY.y * hh; g.halfV[2] = worldY.z * hh;
                    // RectAreaLight emits toward -Z in local space.
                    g.normal[0] = -worldZ.x; g.normal[1] = -worldZ.y; g.normal[2] = -worldZ.z;
                    g.color[0] = rl->color.r * rl->intensity;
                    g.color[1] = rl->color.g * rl->intensity;
                    g.color[2] = rl->color.b * rl->intensity;
                }
            });

            // FNV-1a 64-bit over the packed UBO. Counts + per-light state are
            // both included, so a hidden light (filtered out by `if (!o.visible)
            // return` above) zero-pads its slot and the hash flips. Any analytic-
            // light change → flag the frame moved so raygen reprojects + halves
            // FC. cameraMovedThisFrame_ is the right gate (not just
            // motionThisFrame_) because lighting affects all non-sky pixels, not
            // just pixels whose mesh moved.
            uint64_t h = 0xcbf29ce484222325ull;
            const auto* bytes = reinterpret_cast<const uint8_t*>(&ubo);
            for (size_t i = 0; i < sizeof(ubo); ++i) {
                h ^= bytes[i];
                h *= 0x100000001b3ull;
            }
            if (h != prevLightsHash_) {
                motionThisFrame_      = true;
                cameraMovedThisFrame_ = true;
                prevLightsHash_       = h;
            }

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), lightsUbos[frame].alloc, &mapped);
            std::memcpy(mapped, &ubo, sizeof(ubo));
            vmaUnmapMemory(ctx->allocator(), lightsUbos[frame].alloc);
        }

        void createFogUbos() {
            for (auto& b : fogUbos) {
                b = createBuffer(
                        ctx->allocator(), ctx->device(),
                        sizeof(GpuFogUbo),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
            }
        }

        // Pack scene.fog (Fog/FogExp2 variant) into the per-frame fog UBO.
        // Mirrors WgpuPathTracer.cpp — FogExp2.density maps directly
        // to sigma_t; linear Fog reaches ~63% extinction at farPlane via
        // sigma = 1 / (far - near). Hash detect changes so the per-pixel motion
        // path halves FC and the new fog state converges quickly.
        void updateFogUbo(uint32_t frame, Object3D& scene) {
            GpuFogUbo ubo{};

            if (auto* sc = dynamic_cast<Scene*>(&scene); sc && sc->fog.has_value()) {
                float sigma = 0.f;
                Color tint{1.f, 1.f, 1.f};
                if (std::holds_alternative<FogExp2>(*sc->fog)) {
                    const auto& f = std::get<FogExp2>(*sc->fog);
                    sigma = f.density;
                    tint  = f.color;
                } else if (std::holds_alternative<Fog>(*sc->fog)) {
                    const auto& f = std::get<Fog>(*sc->fog);
                    const float span = std::max(1e-4f, f.farPlane - f.nearPlane);
                    sigma = 1.f / span;
                    tint  = f.color;
                }
                if (sigma > 0.f) {
                    ubo.sigmaT[0] = sigma;
                    ubo.sigmaT[1] = sigma;
                    ubo.sigmaT[2] = sigma;
                    ubo.enabled   = 1.f;
                    ubo.color[0]  = tint.r;
                    ubo.color[1]  = tint.g;
                    ubo.color[2]  = tint.b;
                    ubo.anisotropy = fogAnisotropy_;
                    ubo.waterSurfaceY = fogWaterSurfaceY_;
                }
            }

            uint64_t h = 0xcbf29ce484222325ull;
            const auto* bytes = reinterpret_cast<const uint8_t*>(&ubo);
            for (size_t i = 0; i < sizeof(ubo); ++i) {
                h ^= bytes[i];
                h *= 0x100000001b3ull;
            }
            if (h != prevFogHash_) {
                motionThisFrame_      = true;
                cameraMovedThisFrame_ = true;
                prevFogHash_          = h;
            }

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), fogUbos[frame].alloc, &mapped);
            std::memcpy(mapped, &ubo, sizeof(ubo));
            vmaUnmapMemory(ctx->allocator(), fogUbos[frame].alloc);
        }

        // Allocate, transition, and upload an Image2D from a tightly-
        // packed CPU buffer. Pixel layout matches `format`. Caller owns the
        // returned Image2D and must call destroyImage2D() on shutdown.
        Image2D createSampledImage2D(uint32_t w, uint32_t h, VkFormat format,
                                     const void* pixels, VkDeviceSize byteSize,
                                     VkFilter filter, VkSamplerAddressMode addrU,
                                     VkSamplerAddressMode addrV,
                                     const char* debugName = nullptr) {
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
            Buffer staging = createBuffer(
                    ctx->allocator(), ctx->device(), byteSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), staging.alloc, &mapped);
            std::memcpy(mapped, pixels, byteSize);
            vmaUnmapMemory(ctx->allocator(), staging.alloc);

            VkCommandBuffer cb = beginOneShot();

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
            vkCmdCopyBufferToImage(cb, staging.handle, out.image,
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
                                     VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                     0, 0, nullptr, 0, nullptr, 1, &toRead);
            }

            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), staging);

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

        // Storage-image (rgba32f, GENERAL layout) used as the
        // progressive accumulation target. No staging upload — contents are
        // initialised the first frame after sampleIndex resets to 0. The
        // raygen reads/writes this every frame, so we transition once at
        // creation and keep it in GENERAL forever after.
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

        void createAccumImage() {
            // Render extent, not swapchain extent — every per-pixel PT
            // buffer (accum, gbuf, reservoirs, denoiser) is sized to the
            // resolution raygen actually launches at. Equal to the
            // swapchain extent unless renderScale_ < 1.
            const VkExtent2D ext = renderExtent();
            for (size_t i = 0; i < accumImagesPP.size(); ++i) {
                accumImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "accumImagePP[0]" : "accumImagePP[1]");
            }
            for (size_t i = 0; i < gbufImagesPP.size(); ++i) {
                gbufImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "gbufImagePP[0]" : "gbufImagePP[1]");
            }
            for (size_t i = 0; i < reservoirPosImagesPP.size(); ++i) {
                reservoirPosImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "reservoirPosImagePP[0]" : "reservoirPosImagePP[1]");
            }
            for (size_t i = 0; i < reservoirWImagesPP.size(); ++i) {
                reservoirWImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                        0, i == 0 ? "reservoirWImagePP[0]" : "reservoirWImagePP[1]");
            }
            // ReSTIR GI reservoir ping-pong — all rgba32f for world-space xs
            // precision (rgba16f would lose ~mm-level accuracy at typical
            // scene scales and break the visibility test's distance check).
            for (size_t i = 0; i < giResXsImagesPP.size(); ++i) {
                giResXsImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "giResXsImagePP[0]" : "giResXsImagePP[1]");
            }
            for (size_t i = 0; i < giResNsImagesPP.size(); ++i) {
                giResNsImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "giResNsImagePP[0]" : "giResNsImagePP[1]");
            }
            for (size_t i = 0; i < giResLoImagesPP.size(); ++i) {
                giResLoImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "giResLoImagePP[0]" : "giResLoImagePP[1]");
            }
            // Filtered + moments ping-pong are owned by Denoiser; ctor must
            // have already constructed denoiser_ before this is reached.
            denoiser_->createImages(ext.width, ext.height);
            // Storage memory contents are undefined after vmaCreateImage —
            // clear all four slots to 0 so the first frame's reproject sees
            // mesh-ID 0 (= miss) and cold-starts with histFc=0 instead of
            // reading garbage that may alias a real mesh-ID and pull in
            // arbitrary mean / fc values.
            clearGbufImages();
            sampleIndex = 0;
            accumWriteIdx_ = 0;
            prevCameraValid = false;
            prevWorldMats.clear();
        }


        // Manual accumulation reset. Mirrors the post-create reset block above:
        // wipes gbuf + accum + ReSTIR DI reservoirs, rewinds sampleIndex, and
        // invalidates reproject state so the next frame cold-starts from
        // sample 1. Issues a vkDeviceWaitIdle since clearGbufImages requires
        // the GPU idle before its TRANSFER_DST layout transition. When
        // called mid-frame (frameState_ != Idle) the GPU-side clear is
        // deferred to the next beginFrameForPT; CPU-side counters reset
        // immediately so the user-visible behaviour matches.
        void resetAccumulation() {
            sampleIndex = 0;
            accumWriteIdx_ = 0;
            prevCameraValid = false;
            prevWorldMats.clear();
            if (frameState_ != FrameState::Idle) {
                pendingAccumulationReset_ = true;
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            clearGbufImages();
        }

        // Path-traced LIDAR scan entry-point. Re-uses the main TLAS and the
        // current frame's geom/mat descriptors via a private RT pipeline owned
        // by `lidar_`. Synchronous: blocks the calling thread until per-beam
        // results land in `outResults`.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& outResults,
                       const LidarParams& params) {
            outResults.clear();
            if (beams.empty()) return;

            // Lazy-construct the LIDAR pipeline. Idempotent; if the user
            // never calls scanLidar, the pipeline + SBT cost is never paid.
            if (!lidar_) lidar_ = std::make_unique<vulkan::LidarScanner>(*ctx);

            // Drain all in-flight frames so the TLAS + geom/mat buffers
            // are stable while we read them.
            vkDeviceWaitIdle(ctx->device());

            // Force-flush MaterialDescs into buffer slot 0 — gives us a
            // known-good frame target without having to chase `currentFrame`
            // semantics across render() calls.
            matDescsDirty_[0] = true;
            flushMaterialDescsIfDirty(0);

            // Push constants encode the LIDAR equation parameters. The
            // shader multiplies the raw `laserPower · f_back · cos θ · η / r²`
            // contribution by `invReferenceIntensity` so that, AT laserPower = 1,
            // a perpendicular 1.0-albedo *Lambertian* surface at `referenceRange`
            // reads as 1.0. The reference is purely geometric (π · refRange²)
            // — it does NOT include laserPower, so raising the power slider
            // scales every return linearly (the whole point of the knob).
            // The π factor absorbs the 1/π in the Lambert BRDF — without it
            // a "100% reflective" Lambertian at the reference range would
            // read as 1/π ≈ 0.318 instead of 1.0.
            vulkan_lidar::LidarPushConstants pc{};
            pc.numBeams = static_cast<uint32_t>(beams.size());
            pc.maxRange = std::max(0.001f, params.maxRange);
            pc.laserPower = std::max(0.f, params.laserPower);
            const float refRange = std::max(0.001f, params.referenceRange);
            constexpr float kPi = 3.14159265358979323846f;
            pc.invReferenceIntensity = kPi * refRange * refRange;
            pc.atmosphericExtinction = std::max(0.f, params.atmosphericExtinction);
            pc.detectorThreshold = std::max(0.f, params.detectorThreshold);
            // Frame-varying seed so the Monte Carlo fog scatter + sub-beam
            // jitter decorrelate across successive scans; sharing the path
            // tracer's accum index gives a stable, monotonically growing
            // stream that resets on accumulation reset.
            pc.rngSeed = sampleIndex;
            pc.maxReturns = std::max(1u, params.maxReturns);
            pc.samplesPerBeam = std::max(1u, params.samplesPerBeam);
            pc.beamDivergenceTan = std::tan(0.5f * std::max(0.f, params.beamDivergenceMrad) * 0.001f);
            pc.mediumSurfaceY   = params.mediumSurfaceY;
            pc.mediumExtinction = std::max(0.f, params.mediumExtinction);
            pc.mediumAlbedo     = std::clamp(params.mediumAlbedo, 0.f, 1.f);
            pc.mediumAnisotropy = std::clamp(params.mediumAnisotropy, -0.95f, 0.95f);

            // Pack beams into the shader-side struct (vec3 + pad).
            std::vector<vulkan_lidar::LidarBeam> packed(beams.size());
            for (size_t i = 0; i < beams.size(); ++i) {
                packed[i].origin[0]    = beams[i].origin.x;
                packed[i].origin[1]    = beams[i].origin.y;
                packed[i].origin[2]    = beams[i].origin.z;
                packed[i]._pad0        = 0.f;
                packed[i].direction[0] = beams[i].direction.x;
                packed[i].direction[1] = beams[i].direction.y;
                packed[i].direction[2] = beams[i].direction.z;
                packed[i]._pad1        = 0.f;
            }

            // results[(beam * samplesPerBeam + sample) * maxReturns + slot]
            const size_t totalSlots = beams.size() * pc.samplesPerBeam * pc.maxReturns;
            std::vector<vulkan_lidar::LidarResult> raw(totalSlots);

            // Pick the most recently uploaded fog slot. render() bumps
            // currentFrame at the end, so the just-used slot is
            // (currentFrame - 1) mod N. The slot's contents reflect the
            // scene's current fog state because updateFogUbo runs every
            // frame from render().
            const uint32_t fogSlot = (currentFrame + kFramesInFlight - 1) % kFramesInFlight;

            lidar_->scan(ctx->graphicsQueue(),
                         tlas,
                         geometryDescsBuffer.handle, geometryDescsBuffer.size,
                         materialDescsBuffers[0].handle, materialDescsBuffers[0].size,
                         fogUbos[fogSlot].handle, fogUbos[fogSlot].size,
                         pc,
                         packed.data(), static_cast<uint32_t>(packed.size()),
                         raw.data());

            // Unpack into the public LidarReturn struct. We preserve the
            // fixed-stride layout (numBeams * maxReturns) — caller filters
            // entries with hitInstanceId < 0.
            outResults.resize(totalSlots);
            for (size_t i = 0; i < totalSlots; ++i) {
                const auto& r = raw[i];
                auto& o = outResults[i];
                o.position.set(r.position[0], r.position[1], r.position[2]);
                o.normal.set(r.normal[0], r.normal[1], r.normal[2]);
                o.distance      = r.distance;
                o.intensity     = r.intensity;
                o.hitInstanceId = r.instanceId;
                o.returnNo      = r.returnNo;
            }
        }

        // ── Hybrid raster G-buffer prepass implementation ───────────────────
        // Lazy-initialized on first render().
        // All resources owned by Impl; cleanup in dtor + destroyRasterGbufImages
        // is also called on swapchain resize.

        Image2D createAttachmentImage2D(uint32_t w, uint32_t h, VkFormat format,
                                        VkImageUsageFlags usage,
                                        VkImageAspectFlags aspect,
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

        void destroyRasterGbufImages() {
            if (!ctx) return;
            VkDevice d = ctx->device();
            for (auto& g : rasterGbufs) {
                if (g.framebuffer) {
                    vkDestroyFramebuffer(d, g.framebuffer, nullptr);
                    g.framebuffer = VK_NULL_HANDLE;
                }
                destroyImage2D(ctx->allocator(), d, g.normal);
                destroyImage2D(ctx->allocator(), d, g.motion);
                destroyImage2D(ctx->allocator(), d, g.ids);
                destroyImage2D(ctx->allocator(), d, g.uv);
                destroyImage2D(ctx->allocator(), d, g.albedo);
                destroyImage2D(ctx->allocator(), d, g.indirect);
                destroyImage2D(ctx->allocator(), d, g.momentsSq);
                destroyImage2D(ctx->allocator(), d, g.atrousA);
                destroyImage2D(ctx->allocator(), d, g.atrousB);
                destroyImage2D(ctx->allocator(), d, g.reflect);
                destroyImage2D(ctx->allocator(), d, g.reflAux);
                destroyImage2D(ctx->allocator(), d, g.depth);
                destroyImage2D(ctx->allocator(), d, g.unjitDepth);
                g.width  = 0;
                g.height = 0;
            }
        }

        void createRasterGbufRenderPass() {
            VkAttachmentDescription attachments[6]{};
            // 0: world-space normal (rgba16f). FragShader writes; raygen samples.
            attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
            attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // 1: motion vector (rgba16f, only rg used).
            attachments[1] = attachments[0];
            // 2: per-pixel IDs + flags (rgba16ui).
            attachments[2] = attachments[0];
            attachments[2].format = VK_FORMAT_R16G16B16A16_UINT;
            // 3: material UV (rgba16f, only rg used).
            attachments[3] = attachments[0];
            // 4: albedo + metalness (rgba8 unorm) — raster-first deferred input.
            attachments[4]        = attachments[0];
            attachments[4].format = VK_FORMAT_R8G8B8A8_UNORM;
            // 5: depth (d32_sfloat).
            attachments[5]              = attachments[0];
            attachments[5].format       = VK_FORMAT_D32_SFLOAT;
            attachments[5].finalLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRefs[5]{};
            for (uint32_t i = 0; i < 5; ++i) {
                colorRefs[i].attachment = i;
                colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            VkAttachmentReference depthRef{};
            depthRef.attachment = 5;
            depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount    = 5;
            subpass.pColorAttachments       = colorRefs;
            subpass.pDepthStencilAttachment = &depthRef;

            // Sandwich the pass between (any prior raygen reads of these
            // attachments) and (the post-pass raygen consumer). Vulkan
            // doesn't know raygen reads them — we declare the synchronization
            // explicitly via subpass dependencies.
            VkSubpassDependency deps[2]{};
            deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
            deps[0].dstSubpass    = 0;
            deps[0].srcStageMask  = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].srcSubpass    = 0;
            deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
            deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            // Consumed by raygen (ReferencePT) AND the deferred shade compute
            // pass (RasterFirst) — make the attachments visible to both stages.
            deps[1].dstStageMask  = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            VkRenderPassCreateInfo rpci{};
            rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rpci.attachmentCount = 6;
            rpci.pAttachments    = attachments;
            rpci.subpassCount    = 1;
            rpci.pSubpasses      = &subpass;
            rpci.dependencyCount = 2;
            rpci.pDependencies   = deps;
            check(vkCreateRenderPass(ctx->device(), &rpci, nullptr, &rasterGbufRenderPass),
                  "vkCreateRenderPass(rasterGbuf)");
        }

        void createRasterGbufImages(uint32_t w, uint32_t h) {
            destroyRasterGbufImages();
            const VkImageUsageFlags colorUsage =
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;
            const VkImageUsageFlags depthUsage =
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;
            for (size_t fi = 0; fi < rasterGbufs.size(); ++fi) {
                auto& g = rasterGbufs[fi];
                char nameBuf[64];
                auto N = [&](const char* tag) {
                    std::snprintf(nameBuf, sizeof(nameBuf), "rasterGbuf[%zu].%s", fi, tag);
                    return nameBuf;
                };
                g.normal = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("normal"));
                g.motion = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("motion"));
                g.ids    = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_UINT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("ids"));
                g.uv     = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("uv"));
                g.albedo = createAttachmentImage2D(w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("albedo"));
                // Deferred GI accumulator / denoiser scratch — STORAGE (compute
                // write/read) + SAMPLED so the deferred shade can sample the OTHER
                // frame-in-flight's indirect as the 1-frame GI history (reprojected
                // by the motion vector — Phase-2 GI reproject). Never a render-pass
                // attachment, so not in the framebuffer.
                g.indirect = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                     VK_IMAGE_ASPECT_COLOR_BIT, N("indirect"));
                // SVGF second-moment accumulator E[L²] (single channel). Same
                // STORAGE+SAMPLED + ping-pong as indirect: deferred_shade reproject-
                // accumulates it, deferred_denoise reads it for the per-pixel
                // temporal variance that guides the à-trous (crisp where stable,
                // blurred where noisy).
                g.momentsSq = createAttachmentImage2D(w, h, VK_FORMAT_R16_SFLOAT,
                                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                      VK_IMAGE_ASPECT_COLOR_BIT, N("momentsSq"));
                // SVGF multi-pass à-trous ping-pong scratch (rgb=GI, a=variance).
                // STORAGE only (compute imageLoad/Store, no sampling). The denoise
                // bounces the GI between these at widening step sizes.
                g.atrousA = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("atrousA"));
                g.atrousB = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("atrousB"));
                // Sharp mirror-ray reflection radiance — written by the shade, then
                // roughness-blurred + recombined by the reflection denoise pass.
                // SAMPLED too: the shade temporally accumulates it (samples the OTHER
                // frame-in-flight as 1-frame history) to anti-alias the 1-ray
                // refraction via the gbuffer jitter — sharp AND alias-free.
                g.reflect = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("reflect"));
                // Reflection-denoiser auxiliary — mirrors `reflect` exactly (STORAGE
                // write + SAMPLED prev-frame read, ping-ponged across frames-in-flight).
                g.reflAux = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("reflAux"));
                g.depth  = createAttachmentImage2D(w, h, VK_FORMAT_D32_SFLOAT,
                                                   depthUsage, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                   N("depth"));
                // Unjittered depth, written by the overlay depth prepass and
                // read by the post-TAA wireframe overlay's depth-test. Lives
                // outside the main G-buffer render pass — bound only via
                // dynamic rendering, so it's not part of the framebuffer.
                // Sized to the SWAPCHAIN extent, not the render extent: the
                // overlay composites onto the post-TAA full-resolution image
                // (TAA upscales when renderScale < 1), so its depth
                // attachment must match the swapchain, not the G-buffer.
                const VkExtent2D swapExt = ctx->swapchainExtent();
                g.unjitDepth = createAttachmentImage2D(swapExt.width, swapExt.height,
                                                       VK_FORMAT_D32_SFLOAT,
                                                       depthUsage, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                       N("unjitDepth"));

                VkImageView views[6] = {g.normal.view, g.motion.view, g.ids.view,
                                        g.uv.view, g.albedo.view, g.depth.view};
                VkFramebufferCreateInfo fci{};
                fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fci.renderPass      = rasterGbufRenderPass;
                fci.attachmentCount = 6;
                fci.pAttachments    = views;
                fci.width           = w;
                fci.height          = h;
                fci.layers          = 1;
                check(vkCreateFramebuffer(ctx->device(), &fci, nullptr, &g.framebuffer),
                      "vkCreateFramebuffer(rasterGbuf)");
                g.width  = w;
                g.height = h;
            }

            // Initialise every slot's attachments to their sampled-read layout.
            // The gbuffer render pass writes only the CURRENT frame's slot each
            // frame (its attachments use initialLayout = UNDEFINED, so it
            // ignores whatever we set here), but the temporal consumers sample a
            // slot that has not been rasterised yet on the first frame(s) it is
            // the "previous" slot — TaaResolve reads gbufIdsPrevTex for its
            // same-mesh reproject guard, and deferred/raygen read the per-frame
            // gbuffer. Without a defined starting layout that read finds the
            // image in UNDEFINED and trips VUID-vkCmdDraw-None-09600 /
            // -vkCmdDispatch-None-09600. Contents stay undefined, but the layout
            // is now valid; the temporal logic discards previous data on the
            // first frame regardless. Re-runs on resize (images are recreated).
            VkCommandBuffer initCb = beginOneShot();
            std::vector<VkImageMemoryBarrier> inits;
            inits.reserve(rasterGbufs.size() * 6);
            auto pushInit = [&](VkImage image, VkImageAspectFlags aspect, VkImageLayout layout) {
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = layout;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = image;
                b.subresourceRange.aspectMask = aspect;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                b.srcAccessMask = 0;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                inits.push_back(b);
            };
            for (auto& g : rasterGbufs) {
                pushInit(g.normal.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.ids.image,    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.uv.image,     VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.albedo.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.indirect.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.momentsSq.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.atrousA.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.atrousB.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.reflect.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.reflAux.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.depth.image,  VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
            }
            vkCmdPipelineBarrier(initCb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 0, nullptr, 0, nullptr,
                                 static_cast<uint32_t>(inits.size()), inits.data());
            endAndSubmitOneShot(initCb, "rasterGbuf init layouts");
        }

        void createRasterCameraUbos() {
            for (auto& b : rasterCameraUbos) {
                if (b.handle != VK_NULL_HANDLE) continue;
                b = createBuffer(
                        ctx->allocator(), ctx->device(),
                        sizeof(RasterCameraData),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }
        }

        void createRasterDsLayoutAndPool() {
            // binding 0: per-frame CameraUbo (vertex)
            // binding 1: motionMat[] storage (vertex; same VkBuffer as raygen's binding 10)
            // binding 2: mats[] storage (fragment; for normal-map index + uvTransformNormal)
            // binding 3: albedoMaps[] bindless sampler array (fragment;
            //            same VkImage handles as raygen's binding 8)
            // binding 4: DrawInfo[] storage (vertex; bindless vertex-pulling SSBO
            //            for the indirect-drawing gbuf pipeline). Re-pointed at
            //            this frame's drawInfoBuffers slot in
            //            recordRasterGbufPass before each indirect dispatch.
            VkDescriptorSetLayoutBinding bindings[5]{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
            bindings[2].binding         = 2;
            bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[3].binding         = 3;
            bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[3].descriptorCount = kMaxMaterialTextures;
            bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[4].binding         = 4;
            bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[4].descriptorCount = 1;
            bindings[4].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = 5;
            dlci.pBindings    = bindings;
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &rasterDsLayout),
                  "vkCreateDescriptorSetLayout(raster)");

            VkDescriptorPoolSize sizes[3]{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sizes[0].descriptorCount = kFramesInFlight;
            sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sizes[1].descriptorCount = kFramesInFlight * 3;// motionMat + mats + drawInfo
            sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[2].descriptorCount = kFramesInFlight * kMaxMaterialTextures;

            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets       = kFramesInFlight;
            dpci.poolSizeCount = 3;
            dpci.pPoolSizes    = sizes;
            check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &rasterDescPool),
                  "vkCreateDescriptorPool(raster)");

            std::array<VkDescriptorSetLayout, kFramesInFlight> layouts;
            layouts.fill(rasterDsLayout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = rasterDescPool;
            ai.descriptorSetCount = kFramesInFlight;
            ai.pSetLayouts        = layouts.data();
            check(vkAllocateDescriptorSets(ctx->device(), &ai, rasterDescSets.data()),
                  "vkAllocateDescriptorSets(raster)");
        }

        void createRasterGbufPipeline() {
            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kGbufferVertSpv);
            vsmci.pCode    = kGbufferVertSpv;
            VkShaderModule vertModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vertModule),
                  "vkCreateShaderModule(gbuffer.vert)");

            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kGbufferFragSpv);
            fsmci.pCode    = kGbufferFragSpv;
            VkShaderModule fragModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &fragModule),
                  "vkCreateShaderModule(gbuffer.frag)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertModule;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragModule;
            stages[1].pName  = "main";

            // Vertex input mirrors the BLAS allocation layout: positions /
            // normals / uvs / prev-positions packed (R32G32B32_SFLOAT for
            // pos+normal+prevPos, R32G32_SFLOAT for uv). The BLAS allocations
            // have VERTEX_BUFFER_BIT so they bind directly. Meshes without
            // a UV attribute bind dummyUvBuffer_ at binding 2; static meshes
            // bind their own vertex buffer at binding 3 (prev == curr).
            VkVertexInputBindingDescription vibs[4]{};
            vibs[0].binding   = 0;
            vibs[0].stride    = 3 * sizeof(float);
            vibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[1].binding   = 1;
            vibs[1].stride    = 3 * sizeof(float);
            vibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[2].binding   = 2;
            vibs[2].stride    = 2 * sizeof(float);
            vibs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[3].binding   = 3;
            vibs[3].stride    = 3 * sizeof(float);
            vibs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription vias[4]{};
            vias[0].location = 0;
            vias[0].binding  = 0;
            vias[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[0].offset   = 0;
            vias[1].location = 1;
            vias[1].binding  = 1;
            vias[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[1].offset   = 0;
            vias[2].location = 2;
            vias[2].binding  = 2;
            vias[2].format   = VK_FORMAT_R32G32_SFLOAT;
            vias[2].offset   = 0;
            vias[3].location = 3;
            vias[3].binding  = 3;
            vias[3].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[3].offset   = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 4;
            vi.pVertexBindingDescriptions      = vibs;
            vi.vertexAttributeDescriptionCount = 4;
            vi.pVertexAttributeDescriptions    = vias;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            // cullMode is set dynamically per-draw in recordRasterGbufPass
            // from the material's Side: BACK for Front (default fast path,
            // ~2× perf on dense meshes like ocean), FRONT for Back, NONE
            // for Double. The static value here is overridden by the dynamic
            // state, but Vulkan still wants a valid placeholder.
            rs.cullMode    = VK_CULL_MODE_BACK_BIT;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp   = VK_COMPARE_OP_GREATER;// reverse-Z (near→1, far→0)

            VkPipelineColorBlendAttachmentState cbas[5]{};
            for (auto& a : cbas) {
                a.blendEnable    = VK_FALSE;
                a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            }
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 5;
            cb.pAttachments    = cbas;

            // Dynamic viewport + scissor + cullMode (Vulkan 1.3 core via
            // extendedDynamicState). cullMode flips per-draw across BACK
            // (Side::Front, default fast path), FRONT (Side::Back), and
            // NONE (Side::Double).
            VkDynamicState dynStates[3] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR,
                                           VK_DYNAMIC_STATE_CULL_MODE};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 3;
            dyn.pDynamicStates    = dynStates;

            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            pcRange.offset     = 0;
            pcRange.size       = 80;// mat4 model + uvec4 (instId/flags/pad/pad)

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &rasterDsLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &rasterPipelineLayout),
                  "vkCreatePipelineLayout(raster)");

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = rasterPipelineLayout;
            gpci.renderPass          = rasterGbufRenderPass;
            gpci.subpass             = 0;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpci, nullptr,
                                            &rasterGbufPipeline),
                  "vkCreateGraphicsPipelines(rasterGbuf)");

            vkDestroyShaderModule(ctx->device(), vertModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragModule, nullptr);

            // Indirect-drawing variant: same fragment shader + same render
            // pass, but uses gbuffer_indirect.vert with bindless vertex
            // pulling. No vertex input bindings — the VS reads positions /
            // normals / UVs via buffer-device-address dereferences keyed
            // by gl_InstanceIndex into the DrawInfo SSBO at binding 4.
            VkShaderModuleCreateInfo vciInd{};
            vciInd.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vciInd.codeSize = sizeof(kGbufferIndirectVertSpv);
            vciInd.pCode    = kGbufferIndirectVertSpv;
            VkShaderModule vertIndirectModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vciInd, nullptr, &vertIndirectModule),
                  "vkCreateShaderModule(gbuffer_indirect.vert)");
            VkShaderModuleCreateInfo fciInd{};
            fciInd.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fciInd.codeSize = sizeof(kGbufferFragSpv);
            fciInd.pCode    = kGbufferFragSpv;
            VkShaderModule fragIndModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fciInd, nullptr, &fragIndModule),
                  "vkCreateShaderModule(gbuffer.frag for indirect)");

            VkPipelineShaderStageCreateInfo stagesInd[2]{};
            stagesInd[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stagesInd[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stagesInd[0].module = vertIndirectModule;
            stagesInd[0].pName  = "main";
            stagesInd[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stagesInd[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stagesInd[1].module = fragIndModule;
            stagesInd[1].pName  = "main";

            // Empty vertex input state — VS doesn't bind anything, it reads
            // from buffer device addresses stored in the per-draw SSBO.
            VkPipelineVertexInputStateCreateInfo viInd{};
            viInd.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            viInd.vertexBindingDescriptionCount   = 0;
            viInd.vertexAttributeDescriptionCount = 0;

            VkGraphicsPipelineCreateInfo gpciInd = gpci;// reuse most state
            gpciInd.stageCount        = 2;
            gpciInd.pStages           = stagesInd;
            gpciInd.pVertexInputState = &viInd;

            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciInd, nullptr,
                                            &rasterGbufIndirectPipeline),
                  "vkCreateGraphicsPipelines(rasterGbufIndirect)");

            // Decal variant (same shaders/layout/render pass): blend-decal
            // meshes (MaterialDesc.alphaCutoff == -2) draw AFTER the opaque
            // buckets and only tint the receiver's albedo. Depth test stays on
            // (GREATER, reverse-Z — the per-draw polygonOffset clip-z bias
            // lifts the coplanar decal above its receiver) but depth write is
            // off, and every attachment except albedo is write-masked so the
            // pixel keeps the receiving surface's normal/ids/motion — the
            // deferred shade lights the receiver, with lerped base colour,
            // exactly like GL's forward alpha blend of a coplanar decal.
            VkPipelineDepthStencilStateCreateInfo dsDecal = ds;
            dsDecal.depthWriteEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState cbasDecal[5]{};
            for (int a = 0; a < 4; ++a) {
                cbasDecal[a].blendEnable    = VK_FALSE;
                cbasDecal[a].colorWriteMask = 0;// keep receiver's normal/motion/ids/uv
            }
            cbasDecal[4].blendEnable         = VK_TRUE;
            cbasDecal[4].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbasDecal[4].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasDecal[4].colorBlendOp        = VK_BLEND_OP_ADD;
            cbasDecal[4].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasDecal[4].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cbasDecal[4].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbasDecal[4].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                               VK_COLOR_COMPONENT_B_BIT;// .a = receiver metalness, keep
            VkPipelineColorBlendStateCreateInfo cbDecal = cb;
            cbDecal.pAttachments = cbasDecal;

            // DECAL_PASS=1 specialization flips gbuffer.frag from the stochastic
            // screen-door to "emit texture alpha as the blend factor". The
            // regular pipelines keep DECAL_PASS=0, so in ReferencePT mode (where
            // decals draw with the regular pipeline) they stay on the screen-door.
            const uint32_t kDecalPassOn = 1u;
            VkSpecializationMapEntry decalSpecEntry{0, 0, sizeof(uint32_t)};
            VkSpecializationInfo decalSpecInfo{1, &decalSpecEntry, sizeof(uint32_t), &kDecalPassOn};
            VkPipelineShaderStageCreateInfo stagesDecal[2] = {stagesInd[0], stagesInd[1]};
            stagesDecal[1].pSpecializationInfo = &decalSpecInfo;

            VkGraphicsPipelineCreateInfo gpciDecal = gpciInd;
            gpciDecal.pStages            = stagesDecal;
            gpciDecal.pDepthStencilState = &dsDecal;
            gpciDecal.pColorBlendState   = &cbDecal;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciDecal, nullptr,
                                            &rasterGbufDecalPipeline),
                  "vkCreateGraphicsPipelines(rasterGbufDecal)");

            vkDestroyShaderModule(ctx->device(), vertIndirectModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragIndModule, nullptr);
        }

        // ── Hybrid raster overlay pipeline (wireframe variant) ──────────────
        // Dynamic-rendering pipeline targeting the swapchain (B8G8R8A8_UNORM)
        // + the existing G-buffer depth (D32_SFLOAT, read-only). Pushes
        // mat4 mvp + vec4 color (80B). Triangle topology + polygon-mode
        // line draws each visible mesh as a wireframe; the host gates
        // per-draw on material.wireframe / overlayLayer membership.
        // Line/LineSegments get their own pipeline variants + cached
        // vertex buffer, also built below.
        void createOverlayPipeline() {
            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kOverlayVertSpv);
            vsmci.pCode    = kOverlayVertSpv;
            VkShaderModule vertModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vertModule),
                  "vkCreateShaderModule(overlay.vert)");

            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kOverlayFragSpv);
            fsmci.pCode    = kOverlayFragSpv;
            VkShaderModule fragModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &fragModule),
                  "vkCreateShaderModule(overlay.frag)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertModule;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragModule;
            stages[1].pName  = "main";

            // Single vertex binding: position, R32G32B32_SFLOAT. Reuses the
            // BLAS's vertex buffer directly (skinned meshes get the deformed
            // current-pose positions because refreshSkinnedBlas already wrote
            // them at the start of the frame).
            VkVertexInputBindingDescription vib{};
            vib.binding   = 0;
            vib.stride    = 3 * sizeof(float);
            vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription via{};
            via.location = 0;
            via.binding  = 0;
            via.format   = VK_FORMAT_R32G32B32_SFLOAT;
            via.offset   = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 1;
            vi.pVertexBindingDescriptions      = &vib;
            vi.vertexAttributeDescriptionCount = 1;
            vi.pVertexAttributeDescriptions    = &via;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            // POLYGON_MODE_LINE renders each tri as 3 lines — that's the
            // wireframe effect. cullMode NONE so back-facing geometry's
            // edges are visible too (helps see structure on closed meshes).
            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_LINE;
            rs.cullMode    = VK_CULL_MODE_NONE;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Depth test re-enabled. Compares against rasterGbufs[f].unjitDepth
            // (filled by the overlay_depth prepass with the SAME unjittered
            // projection overlay.vert uses), so per-pixel z values match and
            // the depth test is stable across frames (no sub-pixel-jitter
            // shimmer that the jittered G-buffer depth caused in the earlier
            // depth-test-on configuration). depthWrite stays OFF — overlay
            // doesn't mutate the depth attachment so subsequent reads of it
            // (next frame's prepass clears + writes again) are well-defined.
            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;// reverse-Z (overlay vs unjitDepth)

            VkPipelineColorBlendAttachmentState cbas{};
            cbas.blendEnable    = VK_FALSE;
            cbas.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments    = &cbas;

            VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates    = dynStates;

            // mat4 mvp (64) + vec4 color (16) = 80 bytes, well under the
            // 128B push-constant guarantee. Both vertex and fragment read
            // the same block.
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pcRange.offset     = 0;
            pcRange.size       = 80;

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 0;
            plci.pSetLayouts            = nullptr;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &overlayPipelineLayout),
                  "vkCreatePipelineLayout(overlay)");

            // Dynamic rendering: declare formats up-front via
            // VkPipelineRenderingCreateInfo. Color = swapchain, depth =
            // D32_SFLOAT (matches rasterGbufs.unjitDepth which the overlay
            // pass binds as a read-only depth attachment).
            const VkFormat colorFmt = ctx->swapchainFormat();
            VkPipelineRenderingCreateInfo prci{};
            prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            prci.colorAttachmentCount    = 1;
            prci.pColorAttachmentFormats = &colorFmt;
            prci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.pNext               = &prci;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = overlayPipelineLayout;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpci, nullptr,
                                            &overlayWireframePipeline),
                  "vkCreateGraphicsPipelines(overlayWireframe)");

            // Solid-fill variant for MeshBasicMaterial-style overlays. Same
            // shaders + state otherwise; only the rasterization mode flips
            // to FILL and we cull back faces (no need to draw the inside of
            // a closed convex overlay). Reuses the just-created shader
            // modules → cheap second pipeline.
            VkPipelineRasterizationStateCreateInfo rsBasic = rs;
            rsBasic.polygonMode = VK_POLYGON_MODE_FILL;
            rsBasic.cullMode    = VK_CULL_MODE_BACK_BIT;
            VkGraphicsPipelineCreateInfo gpciBasic = gpci;
            gpciBasic.pRasterizationState = &rsBasic;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciBasic, nullptr,
                                            &overlayBasicPipeline),
                  "vkCreateGraphicsPipelines(overlayBasic)");

            // Alpha-blended fill variant. Standard "non-premultiplied" alpha:
            //   srcColor·srcAlpha + dstColor·(1-srcAlpha)
            // Depth-test stays on (occluded by PT geometry), depth-write OFF
            // so back-to-front order doesn't matter for the depth attachment
            // even if multiple transparent overlays overlap. Per-overlay
            // depth sorting is NOT performed — overlapping transparent
            // overlays may show out-of-order alpha. For typical gizmo /
            // single-transparent-mesh use this is acceptable; documented as
            // a Stage-2 limitation.
            VkPipelineColorBlendAttachmentState cbasBlend{};
            cbasBlend.blendEnable         = VK_TRUE;
            cbasBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbasBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasBlend.colorBlendOp        = VK_BLEND_OP_ADD;
            cbasBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasBlend.alphaBlendOp        = VK_BLEND_OP_ADD;
            cbasBlend.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cbBlend{};
            cbBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cbBlend.attachmentCount = 1;
            cbBlend.pAttachments    = &cbasBlend;
            VkGraphicsPipelineCreateInfo gpciBasicTr = gpciBasic;
            gpciBasicTr.pColorBlendState = &cbBlend;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciBasicTr, nullptr,
                                            &overlayBasicTransparentPipeline),
                  "vkCreateGraphicsPipelines(overlayBasicTransparent)");

            // Line / LineSegments pipelines. Same overlay shaders, same
            // depth/blend state as the basic opaque variant; only the
            // input-assembly topology and rasterization mode differ.
            // POLYGON_MODE_FILL is irrelevant for line topologies but kept
            // for pipeline validity. cullMode=NONE (lines don't have
            // facing).
            VkPipelineRasterizationStateCreateInfo rsLine = rs;
            rsLine.polygonMode = VK_POLYGON_MODE_FILL;
            rsLine.cullMode    = VK_CULL_MODE_NONE;

            VkPipelineInputAssemblyStateCreateInfo iaLineList{};
            iaLineList.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            iaLineList.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            VkGraphicsPipelineCreateInfo gpciLineList = gpci;
            gpciLineList.pInputAssemblyState = &iaLineList;
            gpciLineList.pRasterizationState = &rsLine;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciLineList, nullptr,
                                            &overlayLineListPipeline),
                  "vkCreateGraphicsPipelines(overlayLineList)");

            VkPipelineInputAssemblyStateCreateInfo iaLineStrip{};
            iaLineStrip.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            iaLineStrip.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            VkGraphicsPipelineCreateInfo gpciLineStrip = gpci;
            gpciLineStrip.pInputAssemblyState = &iaLineStrip;
            gpciLineStrip.pRasterizationState = &rsLine;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciLineStrip, nullptr,
                                            &overlayLineStripPipeline),
                  "vkCreateGraphicsPipelines(overlayLineStrip)");

            // ── Colored line pipelines ──────────────────────────────────────
            // Different shader pair (overlay_color.vert/.frag) + 2 vertex
            // bindings: position at 0, color at 1.
            VkShaderModuleCreateInfo cvsmci{};
            cvsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            cvsmci.codeSize = sizeof(kOverlayColorVertSpv);
            cvsmci.pCode    = kOverlayColorVertSpv;
            VkShaderModule cvert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &cvsmci, nullptr, &cvert),
                  "vkCreateShaderModule(overlay_color.vert)");

            VkShaderModuleCreateInfo cfsmci{};
            cfsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            cfsmci.codeSize = sizeof(kOverlayColorFragSpv);
            cfsmci.pCode    = kOverlayColorFragSpv;
            VkShaderModule cfrag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &cfsmci, nullptr, &cfrag),
                  "vkCreateShaderModule(overlay_color.frag)");

            VkPipelineShaderStageCreateInfo cStages[2]{};
            cStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            cStages[0].module = cvert;
            cStages[0].pName  = "main";
            cStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            cStages[1].module = cfrag;
            cStages[1].pName  = "main";

            VkVertexInputBindingDescription cvibs[2]{};
            cvibs[0].binding   = 0;
            cvibs[0].stride    = 3 * sizeof(float);
            cvibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            cvibs[1].binding   = 1;
            cvibs[1].stride    = 3 * sizeof(float);
            cvibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            VkVertexInputAttributeDescription cvias[2]{};
            cvias[0].location = 0;
            cvias[0].binding  = 0;
            cvias[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
            cvias[0].offset   = 0;
            cvias[1].location = 1;
            cvias[1].binding  = 1;
            cvias[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
            cvias[1].offset   = 0;
            VkPipelineVertexInputStateCreateInfo cvi{};
            cvi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            cvi.vertexBindingDescriptionCount   = 2;
            cvi.pVertexBindingDescriptions      = cvibs;
            cvi.vertexAttributeDescriptionCount = 2;
            cvi.pVertexAttributeDescriptions    = cvias;

            VkGraphicsPipelineCreateInfo gpciLineListColored = gpciLineList;
            gpciLineListColored.stageCount        = 2;
            gpciLineListColored.pStages           = cStages;
            gpciLineListColored.pVertexInputState = &cvi;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciLineListColored, nullptr,
                                            &overlayLineListColoredPipeline),
                  "vkCreateGraphicsPipelines(overlayLineListColored)");

            VkGraphicsPipelineCreateInfo gpciLineStripColored = gpciLineStrip;
            gpciLineStripColored.stageCount        = 2;
            gpciLineStripColored.pStages           = cStages;
            gpciLineStripColored.pVertexInputState = &cvi;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciLineStripColored, nullptr,
                                            &overlayLineStripColoredPipeline),
                  "vkCreateGraphicsPipelines(overlayLineStripColored)");

            // ── Point list pipeline ─────────────────────────────────────────
            // POINT_LIST topology. Uses overlay_point.vert/.frag which write
            // gl_PointSize from the push constant's color.w slot and discard
            // fragments outside a unit-radius disk (round sprite). Shares the
            // same overlayPipelineLayout + 2-binding vertex input (pos + color)
            // as the colored line variants — a Points object without a "color"
            // attribute is skipped at draw-record time.
            VkShaderModuleCreateInfo pvsmci{};
            pvsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            pvsmci.codeSize = sizeof(kOverlayPointVertSpv);
            pvsmci.pCode    = kOverlayPointVertSpv;
            VkShaderModule pvert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &pvsmci, nullptr, &pvert),
                  "vkCreateShaderModule(overlay_point.vert)");

            VkShaderModuleCreateInfo pfsmci{};
            pfsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            pfsmci.codeSize = sizeof(kOverlayPointFragSpv);
            pfsmci.pCode    = kOverlayPointFragSpv;
            VkShaderModule pfrag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &pfsmci, nullptr, &pfrag),
                  "vkCreateShaderModule(overlay_point.frag)");

            VkPipelineShaderStageCreateInfo pStages[2]{};
            pStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            pStages[0].module = pvert;
            pStages[0].pName  = "main";
            pStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            pStages[1].module = pfrag;
            pStages[1].pName  = "main";

            VkPipelineInputAssemblyStateCreateInfo iaPointList{};
            iaPointList.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            iaPointList.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

            VkGraphicsPipelineCreateInfo gpciPointList = gpciLineList;
            gpciPointList.stageCount        = 2;
            gpciPointList.pStages           = pStages;
            gpciPointList.pVertexInputState = &cvi;
            gpciPointList.pInputAssemblyState = &iaPointList;
            check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &gpciPointList, nullptr,
                                            &overlayPointListPipeline),
                  "vkCreateGraphicsPipelines(overlayPointList)");

            vkDestroyShaderModule(ctx->device(), vertModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragModule, nullptr);
            vkDestroyShaderModule(ctx->device(), cvert, nullptr);
            vkDestroyShaderModule(ctx->device(), cfrag, nullptr);
            vkDestroyShaderModule(ctx->device(), pvert, nullptr);
            vkDestroyShaderModule(ctx->device(), pfrag, nullptr);

            // ── Overlay depth prepass pipeline ──────────────────────────────
            // Renders all non-overlay scene geometry with the unjittered VP
            // into rasterGbufs[f].unjitDepth. Reuses rasterPipelineLayout
            // (same camera UBO + push constant). Position-only vertex input;
            // no color attachments. depthCompareOp = LESS so the closest
            // surface wins per pixel, the same way the main G-buffer fills
            // depth. Runs each frame after recordRasterGbufPass.
            {
                VkShaderModuleCreateInfo dvsmci{};
                dvsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                dvsmci.codeSize = sizeof(kOverlayDepthVertSpv);
                dvsmci.pCode    = kOverlayDepthVertSpv;
                VkShaderModule dvert = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &dvsmci, nullptr, &dvert),
                      "vkCreateShaderModule(overlay_depth.vert)");

                VkShaderModuleCreateInfo dfsmci{};
                dfsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                dfsmci.codeSize = sizeof(kOverlayDepthFragSpv);
                dfsmci.pCode    = kOverlayDepthFragSpv;
                VkShaderModule dfrag = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &dfsmci, nullptr, &dfrag),
                      "vkCreateShaderModule(overlay_depth.frag)");

                VkPipelineShaderStageCreateInfo dStages[2]{};
                dStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                dStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
                dStages[0].module = dvert;
                dStages[0].pName  = "main";
                dStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                dStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
                dStages[1].module = dfrag;
                dStages[1].pName  = "main";

                // Position-only vertex input. Note: the raster pipeline also
                // declares 4 vertex bindings (pos/normal/uv/prevPos) — the
                // depth prepass shares rasterPipelineLayout but its pipeline
                // declares only 1 binding here so we don't need to bind 4
                // buffers per draw. Vulkan permits more pipeline-declared
                // bindings than the shader uses; what's important is the
                // SHADER consumes only the bindings it has.
                VkVertexInputBindingDescription dvib{};
                dvib.binding   = 0;
                dvib.stride    = 3 * sizeof(float);
                dvib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                VkVertexInputAttributeDescription dvia{};
                dvia.location = 0;
                dvia.binding  = 0;
                dvia.format   = VK_FORMAT_R32G32B32_SFLOAT;
                dvia.offset   = 0;
                VkPipelineVertexInputStateCreateInfo dvi{};
                dvi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                dvi.vertexBindingDescriptionCount   = 1;
                dvi.pVertexBindingDescriptions      = &dvib;
                dvi.vertexAttributeDescriptionCount = 1;
                dvi.pVertexAttributeDescriptions    = &dvia;

                VkPipelineInputAssemblyStateCreateInfo dia{};
                dia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                dia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

                VkPipelineViewportStateCreateInfo dvp{};
                dvp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                dvp.viewportCount = 1;
                dvp.scissorCount  = 1;

                VkPipelineRasterizationStateCreateInfo drs{};
                drs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                drs.polygonMode = VK_POLYGON_MODE_FILL;
                drs.cullMode    = VK_CULL_MODE_BACK_BIT;
                drs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                drs.lineWidth   = 1.0f;

                VkPipelineMultisampleStateCreateInfo dms{};
                dms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                dms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

                VkPipelineDepthStencilStateCreateInfo dds{};
                dds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                dds.depthTestEnable  = VK_TRUE;
                dds.depthWriteEnable = VK_TRUE;
                dds.depthCompareOp   = VK_COMPARE_OP_GREATER;// reverse-Z (overlay depth prepass)

                // No color attachments — pColorBlendState attachmentCount=0.
                VkPipelineColorBlendStateCreateInfo dcb{};
                dcb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                dcb.attachmentCount = 0;

                VkDynamicState ddyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
                VkPipelineDynamicStateCreateInfo ddyn{};
                ddyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                ddyn.dynamicStateCount = 2;
                ddyn.pDynamicStates    = ddyns;

                VkPipelineRenderingCreateInfo dprci{};
                dprci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
                dprci.colorAttachmentCount  = 0;
                dprci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

                VkGraphicsPipelineCreateInfo dgpci{};
                dgpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                dgpci.pNext               = &dprci;
                dgpci.stageCount          = 2;
                dgpci.pStages             = dStages;
                dgpci.pVertexInputState   = &dvi;
                dgpci.pInputAssemblyState = &dia;
                dgpci.pViewportState      = &dvp;
                dgpci.pRasterizationState = &drs;
                dgpci.pMultisampleState   = &dms;
                dgpci.pDepthStencilState  = &dds;
                dgpci.pColorBlendState    = &dcb;
                dgpci.pDynamicState       = &ddyn;
                dgpci.layout              = rasterPipelineLayout;
                check(vkCreateGraphicsPipelines(ctx->device(), VK_NULL_HANDLE, 1, &dgpci, nullptr,
                                                &overlayDepthPrepassPipeline),
                      "vkCreateGraphicsPipelines(overlayDepthPrepass)");

                vkDestroyShaderModule(ctx->device(), dvert, nullptr);
                vkDestroyShaderModule(ctx->device(), dfrag, nullptr);
            }
        }

        // Line geometry cache shared between the ortho overlay (via
        // OverlayPass::record) and the 3D hybrid overlay
        // (recordCommandBuffer's line-draw section). Keyed on raw
        // BufferGeometry*; geomId in LineRec guards against recycled-
        // pointer aliasing. Counter bumped each frame by the 3D overlay
        // path; stale eviction intentionally omitted here (3D overlay
        // objects are persistent scene objects, not transient geometry).
        std::unordered_map<const BufferGeometry*, vulkan::LineRec> lineGeomCache_;
        uint64_t overlayFrameCounter_ = 0;// lastTouch reference for lineGeomCache_ entries

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
            const auto colAttr = geom->hasAttribute("color")
                                         ? geom->getAttribute<float>("color")
                                         : nullptr;

            const uint32_t posVer = posAttr->version;
            const uint32_t idxVer = (idxAttr && idxAttr->count() > 0) ? idxAttr->version : 0u;
            const uint32_t colVer = (colAttr && colAttr->count() > 0) ? colAttr->version : 0u;

            auto it = lineGeomCache_.find(geom);
            if (it != lineGeomCache_.end() && it->second.geomId != geom->id) {
                // Recycled pointer: this address was a DIFFERENT geometry whose
                // buffers we still hold. Retire them and re-upload from scratch
                // (the version fields would otherwise alias — both at 0).
                destroyBuffer(ctx->allocator(), it->second.vertex);
                if (it->second.index.handle) destroyBuffer(ctx->allocator(), it->second.index);
                if (it->second.color.handle) destroyBuffer(ctx->allocator(), it->second.color);
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
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), rec.vertex.alloc, &mapped);
                std::memcpy(mapped, posArr.data(), vbBytes);
                vmaUnmapMemory(ctx->allocator(), rec.vertex.alloc);
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
                    vmaMapMemory(ctx->allocator(), rec.index.alloc, &mapped);
                    std::memcpy(mapped, indices.data(), ibBytes);
                    vmaUnmapMemory(ctx->allocator(), rec.index.alloc);
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
                    const auto& colArr = colAttr->array();
                    const VkDeviceSize cbBytes = colArr.size() * sizeof(float);
                    if (rec.color.handle == VK_NULL_HANDLE || cbBytes > rec.color.size) {
                        if (rec.color.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.color);
                        rec.color = createBuffer(
                                ctx->allocator(), ctx->device(), cbBytes,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    }
                    vmaMapMemory(ctx->allocator(), rec.color.alloc, &mapped);
                    std::memcpy(mapped, colArr.data(), cbBytes);
                    vmaUnmapMemory(ctx->allocator(), rec.color.alloc);
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
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), rec.vertex.alloc, &mapped);
            std::memcpy(mapped, posArr.data(), vbBytes);
            vmaUnmapMemory(ctx->allocator(), rec.vertex.alloc);

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
                vmaMapMemory(ctx->allocator(), rec.index.alloc, &mapped);
                std::memcpy(mapped, indices.data(), ibBytes);
                vmaUnmapMemory(ctx->allocator(), rec.index.alloc);
            }

            if (colAttr && colAttr->count() > 0) {
                const auto& colArr = colAttr->array();
                rec.colorVersion = colVer;
                const VkDeviceSize cbBytes = colArr.size() * sizeof(float);
                rec.color = createBuffer(
                        ctx->allocator(), ctx->device(), cbBytes,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                vmaMapMemory(ctx->allocator(), rec.color.alloc, &mapped);
                std::memcpy(mapped, colArr.data(), cbBytes);
                vmaUnmapMemory(ctx->allocator(), rec.color.alloc);
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
        void rewriteTaaDescriptors() {
            std::array<VkImageView, kFramesInFlight> motionViews{};
            std::array<VkImageView, kFramesInFlight> idsViews{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                motionViews[f] = rasterGbufs[f].motion.view;
                idsViews[f]    = rasterGbufs[f].ids.view;
            }
            const auto& swapViews = ctx->swapchainImageViews();
            vulkan::TaaResolve::DescriptorWriteInputs in{};
            in.gbufSampler        = gbufSampler_;
            in.gbufMotionPerFrame = motionViews.data();
            in.gbufIdsPerFrame    = idsViews.data();
            in.swapchainViews     = swapViews.data();
            taa_->rewriteDescriptors(in);
        }

        // Bloom composite reads the per-frame G-buffer (for the solid-bg sky
        // bypass) and writes the per-frame TAA input. Call after bloom_->
        // createImages OR after the gbuffer / TAA images are reallocated.
        void rewriteBloomDescriptors() {
            std::array<VkImageView, kFramesInFlight> gbufViews{};
            std::array<VkImageView, kFramesInFlight> taaInViews{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                gbufViews[f]  = gbufImagesPP[f].view;
                taaInViews[f] = taa_->inputView(f);
            }
            vulkan::BloomPass::DescriptorWriteInputs in{};
            in.gbufPerFrame     = gbufViews.data();
            in.taaInputPerFrame = taaInViews.data();
            bloom_->rewriteDescriptors(in);
        }

        // Raster-first deferred shade reads the camera + lights UBOs, the env
        // PMREM, the raster material G-buffer (normal+rough / albedo+metal /
        // depth / ids) and writes bloom_->sceneHdr. Call after the raster
        // gbuffer or sceneHdr is reallocated (resize) and after the env image
        // is rebuilt. The UBO buffers are stable; rewriting them is harmless.
        void rewriteDeferredDescriptors() {
            // Needs a built TLAS (binding 8) + material buffer (binding 9). Both
            // come from the scene build; before then there's nothing to bind and
            // the deferred pass can't dispatch anyway. allocateAndUpdateDescriptors
            // (which runs right after the TLAS build) calls this, so the first
            // valid write lands before the first RasterFirst dispatch.
            if (!deferredShade_ || tlas == VK_NULL_HANDLE) return;
            std::array<VkBuffer, kFramesInFlight>    camBufs{};
            std::array<VkBuffer, kFramesInFlight>    lightBufs{};
            std::array<VkBuffer, kFramesInFlight>    matBufs{};
            std::array<VkBuffer, kFramesInFlight>    emBufs{};
            std::array<VkImageView, kFramesInFlight> normalViews{};
            std::array<VkImageView, kFramesInFlight> depthViews{};
            std::array<VkImageView, kFramesInFlight> idsViews{};
            std::array<VkImageView, kFramesInFlight> albedoViews{};
            std::array<VkImageView, kFramesInFlight> uvViews{};
            std::array<VkImageView, kFramesInFlight> motionViews{};
            std::array<VkImageView, kFramesInFlight> indirectViews{};
            std::array<VkImageView, kFramesInFlight> momentsSqViews{};
            std::array<VkImageView, kFramesInFlight> atrousAViews{};
            std::array<VkImageView, kFramesInFlight> atrousBViews{};
            std::array<VkImageView, kFramesInFlight> reflectViews{};
            std::array<VkImageView, kFramesInFlight> reflAuxViews{};
            std::array<VkImageView, kFramesInFlight> sceneHdrViews{};
            std::array<VkBuffer, kFramesInFlight> fogBufs{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                camBufs[f]       = cameraUbos[f].handle;
                lightBufs[f]     = lightsUbos[f].handle;
                fogBufs[f]       = fogUbos[f].handle;
                matBufs[f]       = materialDescsBuffers[f].handle;
                emBufs[f]        = emissiveTriBuffers[f].handle;
                normalViews[f]   = rasterGbufs[f].normal.view;
                depthViews[f]    = rasterGbufs[f].depth.view;
                idsViews[f]      = rasterGbufs[f].ids.view;
                albedoViews[f]   = rasterGbufs[f].albedo.view;
                uvViews[f]       = rasterGbufs[f].uv.view;
                motionViews[f]   = rasterGbufs[f].motion.view;
                indirectViews[f] = rasterGbufs[f].indirect.view;
                momentsSqViews[f]= rasterGbufs[f].momentsSq.view;
                atrousAViews[f]  = rasterGbufs[f].atrousA.view;
                atrousBViews[f]  = rasterGbufs[f].atrousB.view;
                reflectViews[f]  = rasterGbufs[f].reflect.view;
                reflAuxViews[f]  = rasterGbufs[f].reflAux.view;
                sceneHdrViews[f] = bloom_->sceneHdrView(f);
            }
            // Bindless material-texture array for reflected-hit texturing —
            // same source the RT set's binding 8 uses.
            std::array<VkDescriptorImageInfo, kMaxMaterialTextures> matTexInfos{};
            fillMaterialTextureInfos(matTexInfos);

            vulkan::DeferredShade::DescriptorWriteInputs in{};
            in.cameraUbo        = camBufs.data();
            in.lightsUbo        = lightBufs.data();
            in.envView          = envImage.view;
            in.envSampler       = envImage.sampler;
            in.gbufNormal       = normalViews.data();
            in.gbufDepth        = depthViews.data();
            in.gbufIds          = idsViews.data();
            in.gbufAlbedo       = albedoViews.data();
            in.gbufUv           = uvViews.data();
            in.gbufMotion       = motionViews.data();
            in.indirect         = indirectViews.data();
            in.momentsSq        = momentsSqViews.data();
            in.atrousA          = atrousAViews.data();
            in.atrousB          = atrousBViews.data();
            in.reflect          = reflectViews.data();
            in.reflAux          = reflAuxViews.data();
            in.sceneHdr         = sceneHdrViews.data();
            in.fogBuf           = fogBufs.data();
            in.fogRange         = sizeof(GpuFogUbo);
            in.tlas             = tlas;
            in.materialBuf      = matBufs.data();
            in.geomDescBuf      = geometryDescsBuffer.handle;
            in.materialTex      = matTexInfos.data();
            in.materialTexCount = kMaxMaterialTextures;
            in.emissiveTriBuf   = emBufs.data();
            // Ocean textures for the thin-shell water branch. These renderer
            // members are the live ocean FFT-fine + foam views/samplers (or 1×1
            // dummies + tile size 0 when no DisplacedMesh is present) — the same
            // handles the RT set binds at 32 + 44. ensureDisplacedState sets them
            // BEFORE the scene-build descriptor rewrite that calls us, so the
            // deferred set always picks up the current ocean state.
            in.oceanFineView    = oceanFineHeightView;
            in.oceanFineSampler = oceanFineHeightSampler;
            in.oceanFoamView    = oceanFoamView;
            in.oceanFoamSampler = oceanFoamSampler;
            in.foamDetailView    = foamDetailImage.view;
            in.foamDetailSampler = foamDetailImage.sampler;
            // ReSTIR DI reservoir ping-pong — the same 2 physical images the PT path
            // uses (created unconditionally in createAccumImage). rewriteDescriptors
            // picks write/read slots per frame. Locals must outlive the call below.
            const VkImageView resPosViews[2] = {reservoirPosImagesPP[0].view, reservoirPosImagesPP[1].view};
            const VkImageView resWViews[2]   = {reservoirWImagesPP[0].view,   reservoirWImagesPP[1].view};
            in.reservoirPos     = resPosViews;
            in.reservoirW       = resWViews;
            deferredShade_->rewriteDescriptors(in);
        }


        // Lazy bring-up: called at the start of each render() when hybrid is on.
        // Idempotent — handles both initial creation and post-resize reallocation.
        void ensureHybridResources() {
            if (dummyUvBuffer_.handle == VK_NULL_HANDLE) {
                // 1 MB of zeros = 131,072 vec2 vertices. Bound to vertex input
                // 2 when a mesh has no UV attribute. Sized generously because
                // the gbuffer pipeline's binding has fixed stride=8: the GPU
                // reads `vertexCount * 8` bytes, so the dummy must cover the
                // largest mesh's vertex count or we walk off the allocation.
                // 1 MB covers virtually any real-world mesh; if you hit the
                // limit, grow this on demand from ensureSceneBuilt.
                constexpr VkDeviceSize kDummyUvBytes = 1u << 20;
                dummyUvBuffer_ = createBuffer(
                        ctx->allocator(), ctx->device(),
                        kDummyUvBytes,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), dummyUvBuffer_.alloc, &mapped);
                std::memset(mapped, 0, kDummyUvBytes);
                vmaUnmapMemory(ctx->allocator(), dummyUvBuffer_.alloc);
            }
            if (gbufSampler_ == VK_NULL_HANDLE) {
                VkSamplerCreateInfo sci{};
                sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                sci.magFilter    = VK_FILTER_NEAREST;
                sci.minFilter    = VK_FILTER_NEAREST;
                sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.unnormalizedCoordinates = VK_FALSE;
                sci.maxLod       = 0.f;
                check(vkCreateSampler(ctx->device(), &sci, nullptr, &gbufSampler_),
                      "vkCreateSampler(gbuf)");
            }
            if (rasterGbufRenderPass == VK_NULL_HANDLE) {
                createRasterGbufRenderPass();
                createRasterCameraUbos();
                createRasterDsLayoutAndPool();
                createRasterGbufPipeline();
            }
            if (overlayWireframePipeline == VK_NULL_HANDLE) {
                createOverlayPipeline();
            }
            // Raster G-buffer matches the PT render extent — in hybrid mode
            // raygen reads it 1:1 by launch coord, so it must launch at the
            // same resolution the gbuffer rasterized at.
            const VkExtent2D ext = renderExtent();
            if (rasterGbufs[0].width != ext.width || rasterGbufs[0].height != ext.height) {
                createRasterGbufImages(ext.width, ext.height);
            }
        }

        // 1×1 black RGBA32F dummy so the env-map binding is always populated.
        // Replaced with the real scene environment by uploadEnvFromTexture()
        // the first time render() sees a non-empty scene.environment / .background.
        void createDefaultEnvImage() {
            const float pixels[4] = {0.f, 0.f, 0.f, 1.f};
            envImage = createSampledImage2D(
                    1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                    pixels, sizeof(pixels),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    "envImage(default 1x1 black)");
            envIsDefault = true;
            envIsBgColor = false;
            envTextureIdUploaded = 0xFFFFFFFFu;
        }

        // Single shared sampler for every material texture in binding 8.
        // Trilinear + 16× anisotropy: minified surfaces fetch from the proper
        // mip level (set by createSampledImage2D's blit-chain) and glancing
        // angles get aniso-filtered, both of which kill the per-frame texture
        // shimmer that TAA was trying to absorb.
        void createTextureSampler() {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx->physicalDevice(), &props);
            const float maxAniso = std::min(16.0f, props.limits.maxSamplerAnisotropy);

            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.anisotropyEnable = VK_TRUE;
            sci.maxAnisotropy = maxAniso;
            sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sci.unnormalizedCoordinates = VK_FALSE;
            sci.compareEnable = VK_FALSE;
            sci.minLod = 0.0f;
            sci.maxLod = VK_LOD_CLAMP_NONE;
            check(vkCreateSampler(ctx->device(), &sci, nullptr, &textureSampler_),
                  "vkCreateSampler(material)");
        }

        // Slot 0 of the bindless array is a 1×1 white texel — used to fill the
        // descriptor array padding so every slot has a valid view, and as the
        // descriptor for materials whose albedoTexIndex stays at -1 (the
        // shader still indexes safely; the multiply by 1.0 is a no-op).
        void createDefaultMaterialTexture() {
            const uint8_t white[4] = {255, 255, 255, 255};
            Image2D tex = createSampledImage2D(
                    1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                    white, sizeof(white),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "materialTexture[0] (default white)");
            // The Image2D's own sampler is unused (we share textureSampler_),
            // but kept alive so destroyImage2D's clean-up still works.
            materialTextures.push_back(tex);
        }

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
        int32_t ensureMaterialTexture(const std::shared_ptr<Texture>& texSp) {
            if (!texSp) return -1;
            const Texture* tex = texSp.get();
            if (auto it = textureCache.find(tex); it != textureCache.end()) {
                return static_cast<int32_t>(it->second.second);
            }
            if (freeTextureSlots.empty() && materialTextures.size() >= kMaxMaterialTextures) {
                std::cerr << "[VulkanRenderer] material texture slots exhausted ("
                          << kMaxMaterialTextures << "); using -1 fallback\n";
                return -1;
            }
            Image& img = const_cast<Texture*>(tex)->image();
            const uint32_t w = img.width();
            const uint32_t h = img.height();
            if (w == 0 || h == 0) return -1;

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
                bcnRgba = wgpu_pt::bcnDecompress(
                        blocks.data(),
                        static_cast<int>(w),
                        static_cast<int>(h),
                        *img.compressedFormat);
                if (bcnRgba.empty()) {
                    std::cerr << "[VulkanRenderer] unsupported compressed format 0x"
                              << std::hex << *img.compressedFormat << std::dec
                              << " for material tex (" << w << "x" << h << ")\n";
                    return -1;
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
                        return -1;
                    }
                    channels = static_cast<int>(src.size() / pixels);
                    if (channels < 1 || channels > 4) {
                        std::cerr << "[VulkanRenderer] unsupported channel count " << channels
                                  << " for material tex (" << w << "x" << h << ")\n";
                        return -1;
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
                        return -1;
                    }
                    const int fch = static_cast<int>(srcF.size() / pixels);
                    if (fch < 1 || fch > 4) {
                        std::cerr << "[VulkanRenderer] unsupported float channel count " << fch
                                  << " for material tex\n";
                        return -1;
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
            const VkFormat fmt = (tex->colorSpace == ColorSpace::sRGB)
                                         ? VK_FORMAT_R8G8B8A8_SRGB
                                         : VK_FORMAT_R8G8B8A8_UNORM;
            // Pre-compute the slot so the debug name can include it. The
            // allocation order below (free list pop, else push_back) is
            // mirrored in the actual assignment below.
            const uint32_t plannedSlot = freeTextureSlots.empty()
                    ? static_cast<uint32_t>(materialTextures.size())
                    : freeTextureSlots.back();
            char texName[80];
            std::snprintf(texName, sizeof(texName),
                          "materialTexture[%u] (%ux%u, tex=%p)",
                          plannedSlot, w, h, static_cast<const void*>(tex));
            Image2D out = createSampledImage2D(
                    w, h, fmt,
                    rgba.data(), rgba.size(),
                    VK_FILTER_LINEAR,
                    wrapToVk(tex->wrapS),
                    wrapToVk(tex->wrapT),
                    texName);
            uint32_t slot;
            if (!freeTextureSlots.empty()) {
                slot = freeTextureSlots.back();
                freeTextureSlots.pop_back();
                materialTextures[slot] = out;
            } else {
                slot = static_cast<uint32_t>(materialTextures.size());
                materialTextures.push_back(out);
            }
            textureCache.emplace(tex, std::make_pair(std::weak_ptr<Texture>(texSp), slot));
            return static_cast<int32_t>(slot);
        }

        // (Re)allocate envCdfImage / envMargImage from a built EnvCdfResult.
        // Caller must vkDeviceWaitIdle and destroy old images first. When called
        // with a degenerate 1×1 CDF (totalSum=0), the chit gates env importance
        // sampling off and falls back to BSDF-sampled env NEE — same behavior
        // as before this feature landed. envCdfWidth_/Height_/TotalSum_ are
        // mirrored into the push constant on the next renderFrame.
        void rebuildEnvCdfImages(const wgpu_pt::EnvCdfResult& cdf) {
            destroyImage2D(ctx->allocator(), ctx->device(), envCdfImage);
            destroyImage2D(ctx->allocator(), ctx->device(), envMargImage);
            envCdfImage = createSampledImage2D(
                    static_cast<uint32_t>(cdf.width),
                    static_cast<uint32_t>(cdf.height),
                    VK_FORMAT_R32_SFLOAT,
                    cdf.conditional.data(),
                    cdf.conditional.size() * sizeof(float),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    "envCdfImage (conditional)");
            // Marginal is uploaded as h×1 (width=h, height=1) so the shader
            // can fetch entry mid via `texelFetch(envMargTex, ivec2(mid, 0))`,
            // matching WGPU's `cdfSearch(envMargTex, 0, envH, xi)` access
            // pattern. width=1/height=h would route every binary-search lookup
            // to texel (0,0) and silently break importance sampling.
            envMargImage = createSampledImage2D(
                    static_cast<uint32_t>(cdf.height),
                    1u,
                    VK_FORMAT_R32_SFLOAT,
                    cdf.marginal.data(),
                    cdf.marginal.size() * sizeof(float),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    "envMargImage (marginal CDF)");
            envCdfWidth_    = static_cast<uint32_t>(cdf.width);
            envCdfHeight_   = static_cast<uint32_t>(cdf.height);
            envCdfTotalSum_ = cdf.totalSum;
        }

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
        std::vector<uint8_t> generateBlueNoiseTile_() {
            constexpr int N = 64;
            constexpr int M = N * N;
            constexpr float kSigma = 1.5f;
            constexpr int kRadius = 4;
            constexpr int kKernelSide = 2 * kRadius + 1;

            auto wrap = [](int v, int n) { return ((v % n) + n) % n; };

            // Precomputed Gaussian kernel for void/cluster filter.
            std::array<float, kKernelSide * kKernelSide> kernel{};
            for (int dy = -kRadius; dy <= kRadius; ++dy) {
                for (int dx = -kRadius; dx <= kRadius; ++dx) {
                    const int ki = (dy + kRadius) * kKernelSide + (dx + kRadius);
                    kernel[ki] = std::exp(-(dx * dx + dy * dy) /
                                          (2.0f * kSigma * kSigma));
                }
            }

            std::vector<uint8_t> binary(M, 0u);// 0 / 1 instead of bool for vec speed
            std::vector<float>   density(M, 0.0f);

            // Toggle a cell and incrementally update the density field over
            // the Gaussian kernel footprint (toroidal wrap for tileable noise).
            auto toggle = [&](int idx, bool toOn) {
                binary[idx] = toOn ? 1u : 0u;
                const int x = idx % N;
                const int y = idx / N;
                const float sign = toOn ? +1.0f : -1.0f;
                for (int dy = -kRadius; dy <= kRadius; ++dy) {
                    const int yy = wrap(y + dy, N);
                    for (int dx = -kRadius; dx <= kRadius; ++dx) {
                        const int xx = wrap(x + dx, N);
                        const int ki = (dy + kRadius) * kKernelSide + (dx + kRadius);
                        density[yy * N + xx] += sign * kernel[ki];
                    }
                }
            };

            // Find the tightest cluster: the "1" pixel with max density.
            auto findTightest = [&]() {
                int best = -1;
                float bestD = -1.0f;
                for (int i = 0; i < M; ++i) {
                    if (binary[i] && density[i] > bestD) {
                        bestD = density[i];
                        best = i;
                    }
                }
                return best;
            };
            // Find the largest void: the "0" pixel with min density.
            auto findLargestVoid = [&]() {
                int best = -1;
                float bestD = std::numeric_limits<float>::infinity();
                for (int i = 0; i < M; ++i) {
                    if (!binary[i] && density[i] < bestD) {
                        bestD = density[i];
                        best = i;
                    }
                }
                return best;
            };

            // Initial random pattern (~10% set). Deterministic seed for
            // reproducible tiles.
            constexpr int kInitOnes = M / 10;
            {
                std::mt19937 rng(0x12345678u);
                int placed = 0;
                while (placed < kInitOnes) {
                    const int idx = static_cast<int>(rng() % static_cast<uint32_t>(M));
                    if (!binary[idx]) {
                        toggle(idx, true);
                        ++placed;
                    }
                }
            }

            // Phase 1: stabilize via swap (tightest → void). Up to 200 iters;
            // typically converges in <100 for 64×64.
            for (int iter = 0; iter < 200; ++iter) {
                const int tight = findTightest();
                const int voidIdx = findLargestVoid();
                if (tight < 0 || voidIdx < 0 || tight == voidIdx) break;
                toggle(tight, false);
                toggle(voidIdx, true);
            }

            // Snapshot the stabilized prototype for phases 2 + 3.
            const std::vector<uint8_t> proto    = binary;
            const std::vector<float>   protoDen = density;

            std::vector<int> rank(M, 0);

            // Phase 2: rank prototype "ones" by progressive removal of the
            // tightest cluster. Lowest rank = first removed = most isolated.
            for (int r = kInitOnes - 1; r >= 0; --r) {
                const int tight = findTightest();
                if (tight < 0) break;
                rank[tight] = r;
                toggle(tight, false);
            }

            // Restore the prototype for phase 3.
            binary = proto;
            density = protoDen;

            // Phase 3: rank prototype "zeros" by progressive addition into the
            // largest void. Higher rank = added later = more clustered.
            for (int r = kInitOnes; r < M; ++r) {
                const int voidIdx = findLargestVoid();
                if (voidIdx < 0) break;
                rank[voidIdx] = r;
                toggle(voidIdx, true);
            }

            // Map rank ∈ [0, M-1] → uint8 ∈ [0, 255].
            std::vector<uint8_t> tile(static_cast<size_t>(M));
            for (int i = 0; i < M; ++i) {
                tile[i] = static_cast<uint8_t>((rank[i] * 255) / (M - 1));
            }
            return tile;
        }

        // Generate the tile and upload as an R8_UNORM image. Called once at
        // ctor time (after createDescriptorPool) so the descriptor write below
        // can include it.
        void createBlueNoiseImage_() {
            const auto tile = generateBlueNoiseTile_();
            blueNoiseImage = createSampledImage2D(
                    /*w*/ 64u, /*h*/ 64u,
                    VK_FORMAT_R8_UNORM,
                    tile.data(), tile.size(),
                    VK_FILTER_NEAREST,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "blueNoiseImage (64x64 void-and-cluster tile)");
        }

        // Tileable foam detail tile — R = micro bubble brightness, G = ridged
        // "lace" filament pattern. All lattice frequencies are integers so
        // every octave wraps seamlessly across the tile edges. Channel content
        // matches the procedural noise it replaces in the foam shading:
        //   R ≈ vnoise·0.55 + vnoise(×2.3)·0.30 + vnoise(×5.1)·0.15 with
        //       0.154 m base features over the shader's 4 m world mapping;
        //   G ≈ 1 − 2·|nf − 0.5| (filament cores at nf = 0.5) with ~1.2 m
        //       cells over the 12 m mapping.
        std::vector<unsigned char> generateFoamDetailTile_(int res) {
            auto hashf = [](int x, int y) {
                uint32_t h = static_cast<uint32_t>(x) * 374761393u +
                             static_cast<uint32_t>(y) * 668265263u;
                h = (h ^ (h >> 13)) * 1274126177u;
                h ^= h >> 16;
                return static_cast<float>(h) * (1.0f / 4294967296.0f);
            };
            // Tileable value noise: `cells` lattice points per tile edge,
            // wrapped; `seed` decorrelates octaves sharing a frequency.
            auto vnoiseT = [&](float u, float v, int cells, int seed) {
                const float x = u * static_cast<float>(cells);
                const float y = v * static_cast<float>(cells);
                const int xi = static_cast<int>(std::floor(x));
                const int yi = static_cast<int>(std::floor(y));
                float tx = x - static_cast<float>(xi);
                float ty = y - static_cast<float>(yi);
                tx = tx * tx * (3.f - 2.f * tx);
                ty = ty * ty * (3.f - 2.f * ty);
                auto at = [&](int ix, int iy) {
                    return hashf(((ix % cells) + cells) % cells + seed * 7919,
                                 ((iy % cells) + cells) % cells);
                };
                const float a = at(xi, yi), b = at(xi + 1, yi);
                const float c = at(xi, yi + 1), d = at(xi + 1, yi + 1);
                return a + (b - a) * tx + (c - a) * ty + (a - b - c + d) * tx * ty;
            };
            std::vector<unsigned char> px(static_cast<size_t>(res) * res * 2);
            for (int y = 0; y < res; ++y) {
                const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(res);
                for (int x = 0; x < res; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(res);
                    const float micro = 0.55f * vnoiseT(u, v, 26, 1) +
                                        0.30f * vnoiseT(u, v, 60, 2) +
                                        0.15f * vnoiseT(u, v, 132, 3);
                    const float nf = 0.62f * vnoiseT(u, v, 10, 4) +
                                     0.38f * vnoiseT(u, v, 23, 5);
                    const float lace = 1.f - std::fabs(nf - 0.5f) * 2.f;
                    const size_t i = (static_cast<size_t>(y) * res + x) * 2;
                    px[i + 0] = static_cast<unsigned char>(std::lround(std::clamp(micro, 0.f, 1.f) * 255.f));
                    px[i + 1] = static_cast<unsigned char>(std::lround(std::clamp(lace, 0.f, 1.f) * 255.f));
                }
            }
            return px;
        }

        void createFoamDetailImage_() {
            constexpr int kRes = 512;
            const auto tile = generateFoamDetailTile_(kRes);
            // LINEAR + size>1 → createSampledImage2D builds the full mip
            // chain (blit cascade) and a trilinear/aniso REPEAT sampler.
            foamDetailImage = createSampledImage2D(
                    /*w*/ kRes, /*h*/ kRes,
                    VK_FORMAT_R8G8_UNORM,
                    tile.data(), tile.size(),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "foamDetail (512x512 RG8 bubbles+lace, mipped)");
        }

        // 1×1 R32F dummy used at binding 32 when no DisplacedMesh is in the
        // scene. closest_hit gates on pc.oceanFineTileSize > 0 so the sampler
        // result is unread; the descriptor still needs a valid view/sampler
        // to keep the layout populated. Replaced (view only) with the active
        // ocean's cascade-2 height image via rewriteOceanFineDescriptors_().
        void createOceanFineDummy_() {
            const float zero = 0.0f;
            oceanFineHeightDummy = createSampledImage2D(
                    /*w*/ 1u, /*h*/ 1u,
                    VK_FORMAT_R32_SFLOAT,
                    &zero, sizeof(zero),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "oceanFineHeightDummy (1x1, binding 32 placeholder)");
            // Transition to GENERAL so the descriptor layout matches the
            // cascade-2 storage image's layout (also GENERAL after IFFT). One
            // declared layout simplifies rewriteOceanFineDescriptors_().
            {
                VkCommandBuffer cb = beginOneShot();
                VkImageMemoryBarrier imb{};
                imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.image = oceanFineHeightDummy.image;
                imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        0, 0, nullptr, 0, nullptr, 1, &imb);
                endAndSubmitOneShot(cb);
            }
            oceanFineHeightView    = oceanFineHeightDummy.view;
            oceanFineHeightSampler = oceanFineHeightDummy.sampler;
            oceanFineTileSize      = 0.0f;
        }

        // 1×1 R32F dummy for binding 33 (world-space foam) when no
        // DisplacedMesh is in the scene. closest_hit gates on
        // pc.oceanFoamTileSize > 0 so the sampler result is unread; the
        // descriptor still needs a valid view/sampler.
        void createOceanFoamDummy_() {
            const float zero = 0.0f;
            oceanFoamDummy = createSampledImage2D(
                    /*w*/ 1u, /*h*/ 1u,
                    VK_FORMAT_R32_SFLOAT,
                    &zero, sizeof(zero),
                    VK_FILTER_LINEAR,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    "oceanFoamDummy (1x1, binding 44 placeholder)");
            // Transition to GENERAL so the binding matches the foam image's
            // layout (compute leaves it in GENERAL; chit reads from there).
            {
                VkCommandBuffer cb = beginOneShot();
                VkImageMemoryBarrier imb{};
                imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                imb.image = oceanFoamDummy.image;
                imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        0, 0, nullptr, 0, nullptr, 1, &imb);
                endAndSubmitOneShot(cb);
            }
            oceanFoamView     = oceanFoamDummy.view;
            oceanFoamSampler  = oceanFoamDummy.sampler;
            oceanFoamTileSize = 0.0f;
        }

        // Re-write descriptor binding 32 across all sets to point at the
        // currently active view/sampler/tileSize. Called from ensureDisplacedState
        // after the FFT cascades are constructed (and on first switchover from
        // dummy → cascade-2). The cascade-2 VkImage handle is stable for the
        // lifetime of the DisplacedMesh, so we only need to rewrite once.
        // Safe no-op when descriptors haven't been allocated yet — the first
        // structural scene build calls ensureDisplacedState before
        // allocateAndUpdateDescriptors, and the latter picks up the current
        // oceanFineHeightView/Sampler/TileSize values via the standard path.
        void rewriteOceanFineDescriptors_() {
            if (descriptorSets.empty()) return;
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::vector<VkDescriptorImageInfo> infos(totalSets);
            std::vector<VkWriteDescriptorSet>  writes(totalSets);
            for (uint32_t i = 0; i < totalSets; ++i) {
                infos[i].sampler     = oceanFineHeightSampler;
                infos[i].imageView   = oceanFineHeightView;
                infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                writes[i] = VkWriteDescriptorSet{};
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptorSets[i];
                writes[i].dstBinding = 32;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo = &infos[i];
            }
            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }

        // Re-write descriptor binding 44 (world-space foam) across all chit
        // sets. Mirrors rewriteOceanFineDescriptors_; called once when a
        // DisplacedMesh appears (the foam VkImage handle stays put for the
        // lifetime of that mesh, so per-frame writes aren't needed).
        void rewriteOceanFoamDescriptors_() {
            if (descriptorSets.empty()) return;
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::vector<VkDescriptorImageInfo> infos(totalSets);
            std::vector<VkWriteDescriptorSet>  writes(totalSets);
            for (uint32_t i = 0; i < totalSets; ++i) {
                infos[i].sampler     = oceanFoamSampler;
                infos[i].imageView   = oceanFoamView;
                infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                writes[i] = VkWriteDescriptorSet{};
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptorSets[i];
                writes[i].dstBinding = 44;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo = &infos[i];
            }
            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }

        // Build a degenerate 1×1 CDF (totalSum=0) for the case where the env is
        // solid color or default. Shader uses totalSum=0 as the "no CDF" gate.
        void rebuildDefaultEnvCdfImages() {
            wgpu_pt::EnvCdfResult dummy;
            dummy.width = 1; dummy.height = 1;
            dummy.conditional = {1.0f};
            dummy.marginal    = {1.0f};
            dummy.totalSum    = 0.0f;
            rebuildEnvCdfImages(dummy);
        }

        // Detect the active env texture (scene.environment, falling back to
        // scene.background.texture()) and upload it if it differs from the
        // currently bound one. Returns true when descriptors must be rewritten.
        bool refreshEnvTextureFromScene(Object3D& scene) {
            auto* sc = dynamic_cast<Scene*>(&scene);
            std::shared_ptr<Texture> tex;
            if (sc) {
                tex = sc->environment;
                if (!tex && sc->background.isTexture()) {
                    tex = sc->background.texture();
                }
            }
            if (!tex) {
                if (sc && sc->background.isColor()) {
                    const Color& c = sc->background.color();
                    if (envIsBgColor && envBgColor.r == c.r && envBgColor.g == c.g && envBgColor.b == c.b)
                        return false;
                    vkDeviceWaitIdle(ctx->device());
                    destroyImage2D(ctx->allocator(), ctx->device(), envImage);
                    const float px[4] = {c.r, c.g, c.b, 1.f};
                    envImage = createSampledImage2D(
                            1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                            px, sizeof(px),
                            VK_FILTER_NEAREST,
                            VK_SAMPLER_ADDRESS_MODE_REPEAT,
                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                            "envImage(scene.background color)");
                    envIsDefault  = false;
                    envIsBgColor  = true;
                    envBgColor    = c;
                    envTextureIdUploaded = 0xFFFFFFFFu;
                    rebuildDefaultEnvCdfImages();// no importance sampling on solid colors
                    return true;
                }
                if (envIsDefault) return false;
                vkDeviceWaitIdle(ctx->device());
                destroyImage2D(ctx->allocator(), ctx->device(), envImage);
                createDefaultEnvImage();
                envIsBgColor = false;
                rebuildDefaultEnvCdfImages();
                return true;
            }
            envIsBgColor = false;
            if (!envIsDefault && tex->id == envTextureIdUploaded) return false;

            // Only HDR float equirects (RGBELoader output) are supported.
            // LDR backgrounds are not handled — fall through to default.
            const auto& image = tex->image();
            if (tex->type != Type::Float) {
                std::cerr << "[VulkanRenderer] env texture is not float HDR; "
                             "ignoring (HDR equirect only)" << std::endl;
                return false;
            }
            const auto& src = image.data<float>();
            const uint32_t w = image.width();
            const uint32_t h = image.height();
            if (w == 0 || h == 0 || src.size() < 4u * w * h) {
                std::cerr << "[VulkanRenderer] env texture has unexpected size; ignoring" << std::endl;
                return false;
            }

            vkDeviceWaitIdle(ctx->device());
            destroyImage2D(ctx->allocator(), ctx->device(), envImage);

            // GGX-prefilter the env into a mip chain so the closest_hit
            // shader can sample at a roughness-derived LOD instead of the cheap
            // (1-r)² fade. Mip 0 is the source mirror, mip k is convolved with
            // GGX(α=(k/(N-1))²). closest_hit fades from mirror to fully diffuse
            // by walking the chain via textureLod.
            envImage = envPrefilter_->buildPmrem(
                    w, h, src.data(), 4u * w * h * sizeof(float));
            envIsDefault = false;
            envTextureIdUploaded = tex->id;

            // Build the luminance CDF from mip 0 (source equirect, RGBA float).
            // Used by chit's env NEE to importance-sample bright HDRI features
            // (sun discs, sky bands) via the marginal/conditional CDF pair.
            const auto cdf = wgpu_pt::buildEnvCdf<float>(src, static_cast<int>(w), static_cast<int>(h));
            rebuildEnvCdfImages(cdf);
            return true;
        }

        void createRtPipeline() {
            // 0=TLAS, 1=storage image (swapchain), 2=camera UBO, 3=geometry SSBO,
            // 4=material SSBO, 5=lights UBO, 6=env equirect,
            // 7=accum write image (ping-pong), 8=material texture array,
            // 9=prev camera UBO (motion), 10=motion matrix SSBO (motion),
            // 11=prev accum read image (ping-pong), 12=gbuf write image
            // (ping-pong), 13=prev gbuf read image (ping-pong).
            // Bindings 22-26: hybrid raster G-buffer attachments — read by
            // raygen for primary visibility (depth + worldPos reconstruction,
            // motion vector for reproject, normal for shading, instance ID
            // for material lookup, UV for texture sampling on the primary).
            // Always present in the layout; raygen gates use on the hybrid
            // push-constant bit.
            std::array<VkDescriptorSetLayoutBinding, 46> bindings{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;// shadow rays
            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            // outImage is now written by denoise.comp (compute), not raygen.
            // RAYGEN flag retained so the descriptor layout stays valid for
            // both pipelines that share rtDsLayout.
            bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[2].binding = 2;
            bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[2].descriptorCount = 1;
            // Compute denoise reads cam.viewInverse for camera world pos.
            bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[3].binding = 3;
            bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[3].descriptorCount = 1;
            bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            bindings[4].binding = 4;
            bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[4].descriptorCount = 1;
            // COMPUTE added so denoise_atrous can read mats[].materialAssetIdx
            // for the cross-mesh same-material tap-acceptance gate.
            // RAYGEN added for the hybrid reproject bilinear-tap material gate
            // and primary-hit material asset capture.
            bindings[4].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_COMPUTE_BIT |
                                     VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[5].binding = 5;
            bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[5].descriptorCount = 1;
            // RAYGEN added so volumeInscatter can NEE to analytic lights from
            // primary-ray scatter points.
            bindings[5].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[6].binding = 6;
            bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[6].descriptorCount = 1;
            // RAYGEN added for volumeInscatter env-NEE (uniform-sphere fallback).
            bindings[6].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR |
                                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[7].binding = 7;
            bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[7].descriptorCount = 1;
            // Raygen writes; compute denoise reads radiance + per-pixel FC;
            // closest_hit reads .w (per-pixel FC) for the GI primary
            // saturation gate (skip sub-trace on converged pixels).
            bindings[7].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_COMPUTE_BIT |
                                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            // Bindless material albedo array. Fixed-cap kMaxMaterialTextures
            // so we can avoid VK_EXT_descriptor_indexing's variable-descriptor-
            // count plumbing for v1; closest_hit indexes via mdesc.albedoTexIndex.
            bindings[8].binding = 8;
            bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[8].descriptorCount = kMaxMaterialTextures;
            // RAYGEN added for primary-hit albedo capture from bindless array.
            bindings[8].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            // Continuous-motion bindings.
            bindings[9].binding = 9;
            bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[9].descriptorCount = 1;
            // CLOSEST_HIT added so chit can reproject for ReSTIR DI temporal reuse.
            bindings[9].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[10].binding = 10;
            bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[10].descriptorCount = 1;
            bindings[10].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[11].binding = 11;
            bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[11].descriptorCount = 1;
            bindings[11].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[12].binding = 12;
            bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[12].descriptorCount = 1;
            // Compute denoise reads gbuf for worldPos + mesh-ID edge stops.
            bindings[12].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[13].binding = 13;
            bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[13].descriptorCount = 1;
            // CLOSEST_HIT added so chit can read prev worldPos for ReSTIR DI
            // temporal-reuse depth validation.
            bindings[13].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            // Binding 14 — per-frame emissive-triangle CDF (closest_hit reads
            // it for emissive-mesh NEE). One slot per frame-in-flight; rewritten
            // by rewriteEmissiveTriDescriptors when the buffer grows.
            bindings[14].binding = 14;
            bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[14].descriptorCount = 1;
            // RAYGEN added for volumeInscatter emissive-tri NEE.
            bindings[14].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                      VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            // Binding 15 — photon count array (written by photon emit raygen,
            // read by closest_hit for caustic gather).
            bindings[15].binding = 15;
            bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[15].descriptorCount = 1;
            bindings[15].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            // Binding 16 — photon position/flux/direction data (same pipelines).
            bindings[16].binding = 16;
            bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[16].descriptorCount = 1;
            bindings[16].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            // Binding 17 — fog UBO (homogeneous participating media).
            // Read by raygen (primary-ray transmittance + volumeInscatter) and
            // closest_hit (per-light shadow-ray attenuation). Updated each frame
            // by updateFogUbo from scene.fog.
            bindings[17].binding = 17;
            bindings[17].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[17].descriptorCount = 1;
            bindings[17].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            // Binding 18 — env conditional luminance CDF (R32F, w×h). Row r holds
            // cumulative distribution over columns at latitude r. Sampled in chit
            // for env importance sampling (and in miss for the BRDF→env MIS
            // complement at scattered miss, which needs envImportancePdf).
            bindings[18].binding = 18;
            bindings[18].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[18].descriptorCount = 1;
            // RAYGEN added so the hybrid bounce-0-on-raster path can call
            // envNeeOpaque (which calls sampleEnvImportance / envImportancePdf)
            // for opaque non-silhouette primary pixels.
            bindings[18].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                      VK_SHADER_STAGE_MISS_BIT_KHR |
                                      VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            // Binding 19 — env marginal luminance CDF (R32F, 1×h). Picks a row
            // (latitude) before sampling the conditional row at that latitude.
            bindings[19].binding = 19;
            bindings[19].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[19].descriptorCount = 1;
            // RAYGEN added (paired with binding 18) for the bounce-0-on-raster
            // env CDF NEE path.
            bindings[19].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                      VK_SHADER_STAGE_MISS_BIT_KHR |
                                      VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            // Binding 20 — multi-pass à-trous ping-pong intermediates. 2-element
            // storage image array (rgba32f). Pass 0 reads accumImage and writes
            // [0]; pass 1 reads [0] and writes [1]; pass 2 reads [1] and writes
            // [0]; finalize (denoise.comp) reads [0]. COMPUTE-only.
            bindings[20].binding = 20;
            bindings[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[20].descriptorCount = 2;
            bindings[20].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            // Binding 21 — per-instance moved-bitmask SSBO (one bit per TLAS
            // instance, packed into u32 words). Sized at scene rebuild via
            // ensureMeshMovedBitsCapacity, uploaded per frame from
            // meshMovedBits_. Read by raygen.isMeshMoved() to gate FC halving
            // per-pixel under partial scene motion.
            bindings[21].binding = 21;
            bindings[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[21].descriptorCount = 1;
            bindings[21].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

            // Bindings 22-26 — hybrid raster G-buffer attachments. Sampled by
            // raygen when the hybrid push-constant bit is set; ignored
            // otherwise (descriptors stay populated so the layout is valid
            // either way). Same physical images created by ensureHybridResources;
            // descriptor writes route raygen at frame f to rasterGbufs[f].
            bindings[22].binding = 22;// world-space normal (rgba16f, decoded n*2-1)
            bindings[22].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[22].descriptorCount = 1;
            bindings[22].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[23].binding = 23;// motion vector (rgba16f, NDC delta in .rg)
            bindings[23].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[23].descriptorCount = 1;
            bindings[23].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[24].binding = 24;// depth (d32_sfloat, depth aspect view)
            bindings[24].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[24].descriptorCount = 1;
            bindings[24].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[25].binding = 25;// IDs+flags (rgba16ui, usampler2D in shader)
            bindings[25].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[25].descriptorCount = 1;
            bindings[25].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[26].binding = 26;// material UV (rgba16f, only rg used)
            bindings[26].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[26].descriptorCount = 1;
            bindings[26].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[27].binding = 27;// 64×64 R8 blue-noise tile for sub-pixel jitter
            bindings[27].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[27].descriptorCount = 1;
            bindings[27].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

            // Bindings 28-31 — ReSTIR DI Stage 1b reservoir ping-pong storage.
            // 28/29: lightPos.xyz + lightType.w (rgba32f, write/read)
            // 30/31: W_sum, M, W, p_hat (rgba16f, write/read)
            // Frame N writes 28/30, reads 29/31; descriptor-set 1 (next f) flips
            // the physical images so each frame's read sees the prior frame's
            // write. Same pattern as accumImagesPP / gbufImagesPP.
            bindings[28].binding = 28;
            bindings[28].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[28].descriptorCount = 1;
            bindings[28].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[29].binding = 29;
            bindings[29].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[29].descriptorCount = 1;
            bindings[29].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[30].binding = 30;
            bindings[30].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[30].descriptorCount = 1;
            bindings[30].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[31].binding = 31;
            bindings[31].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[31].descriptorCount = 1;
            bindings[31].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

            // Binding 32 — ocean fine-cascade height image (R32F via RG32F .r).
            // Default 1×1 dummy when no DisplacedMesh is in the scene; swapped
            // to cascade-2 height when an FFT ocean is active. closest_hit reads
            // it on thinWalled materials to perturb the shading normal at world-
            // space XZ via finite differences. Gated by pc.oceanFineTileSize > 0.
            bindings[32].binding = 32;
            bindings[32].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[32].descriptorCount = 1;
            bindings[32].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

            // Bindings 33/34 — temporal moments ping-pong (R32F).
            // 33 = write (this frame, raygen); 34 = read (prev frame, raygen).
            // Atrous reads binding 33 (the just-written current-frame moments)
            // to compute per-pixel variance = max(M2 − luminance(mean)², 0)
            // for the σ_lum edge stop. Same ping-pong pattern as
            // accumImagesPP / gbufImagesPP — descriptor sets bake the f&1 →
            // slot mapping at allocation time.
            bindings[33].binding = 33;
            bindings[33].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[33].descriptorCount = 1;
            bindings[33].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                      VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[34].binding = 34;
            bindings[34].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[34].descriptorCount = 1;
            bindings[34].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

            // Bindings 35/36 — primary-surface albedo ping-pong (rgba8).
            // 35 = write (current frame): raygen FC-blends prev albedo
            // (binding 36) with payload.primaryAlbedo and writes here.
            // Atrous reads binding 35 to demodulate radiance.
            // 36 = read (prev frame): raygen samples via bilinear reproject
            // with same mesh-ID + depth gates as accumImage reproject.
            bindings[35].binding = 35;
            bindings[35].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[35].descriptorCount = 1;
            bindings[35].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                      VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[36].binding = 36;
            bindings[36].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[36].descriptorCount = 1;
            bindings[36].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

            // Binding 37 — snapshot of chit's per-frame albedo (no temporal
            // blend). Raygen writes; atrous reads at center for re-mod.
            // Preserves texture crispness while temporal blend on 35/36
            // smooths noise on the demod division side.
            bindings[37].binding = 37;
            bindings[37].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[37].descriptorCount = 1;
            bindings[37].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                      VK_SHADER_STAGE_COMPUTE_BIT;

            // Bindings 38-43 — ReSTIR GI Stage 1b reservoir ping-pong storage.
            // 38/39: xs.xyz + W_sum   (rgba32f write/read)
            // 40/41: ns.xyz + M       (rgba32f write/read)
            // 42/43: Lo.rgb + W       (rgba32f write/read)
            // Frame N writes 38/40/42 and reads 39/41/43; descriptor sets in
            // alloc bake the f&1 → physical slot mapping (like accumImagesPP).
            // CLOSEST_HIT only — raygen doesn't touch these; chit does both
            // the temporal-merge read and the post-merge persistence write.
            bindings[38].binding = 38;
            bindings[38].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[38].descriptorCount = 1;
            bindings[38].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[39].binding = 39;
            bindings[39].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[39].descriptorCount = 1;
            bindings[39].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[40].binding = 40;
            bindings[40].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[40].descriptorCount = 1;
            bindings[40].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[41].binding = 41;
            bindings[41].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[41].descriptorCount = 1;
            bindings[41].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[42].binding = 42;
            bindings[42].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[42].descriptorCount = 1;
            bindings[42].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[43].binding = 43;
            bindings[43].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[43].descriptorCount = 1;
            bindings[43].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

            // Binding 44 — world-space foam (R32F via combined-image-sampler).
            // Built by FoamWorldPipeline each frame; closest_hit samples at
            // hit-position world XZ to apply foam coverage on water surfaces.
            // Default 1×1 dummy when no DisplacedMesh exists; swapped to the
            // active mesh's foam image once one comes online. Gated in the
            // shader by pc.oceanFoamTileSize > 0.
            bindings[44].binding = 44;
            bindings[44].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[44].descriptorCount = 1;
            bindings[44].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

            // Binding 45 — baked tileable foam detail (RG8, mipped). Static
            // for the renderer's lifetime; closest_hit samples it in the foam
            // block (R = bubbles, G = lace) instead of evaluating procedural
            // value noise per pixel. Mirrors deferred binding 34.
            bindings[45].binding = 45;
            bindings[45].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[45].descriptorCount = 1;
            bindings[45].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(bindings.size());
            dlci.pBindings = bindings.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &rtDsLayout),
                  "vkCreateDescriptorSetLayout(RT)");

            // 52-byte push constant. Layout matches the host-side `pc[13]`
            // assembled in renderFrame and the GLSL PushConstants struct in
            // raygen.rgen / closest_hit.rchit / miss.rmiss. Well under the
            // 128-byte minimum push-constant guarantee. Per-instance moved
            // bits moved out to the binding 21 SSBO (no longer 128-mesh capped).
            //
            //   [0] sampleIndex            (raygen)
            //   [1] env mip count          (closest_hit)
            //   [2] toneMapping enum
            //   [3] exposure as float bits
            //   [4] motionFlags: bit 0 = any motion this frame (mesh or
            //       camera), bit 1 = camera viewProj changed, bit 2 = scene
            //       has any glass material. Raygen takes a self-tap of
            //       accum/gbuf when bit 0 is clear, avoiding round-trip
            //       reproject precision drift on static scenes.
            //   [5]  emissiveCount       (closest_hit reads for NEE CDF)
            //   [6]  emissiveTotalPower  (float bits — CDF normalisation)
            //   [7]  samplesPerPixel     (raygen — # of jittered primary rays
            //        per frame; in-frame samples are summed with weight `spp`)
            //   [8]  envCdfWidth         (closest_hit + miss — env importance sample)
            //   [9]  envCdfHeight        (closest_hit + miss — env importance sample)
            //   [10] envCdfTotalSum bits (float — pdf normalisation; 0 disables CDF)
            //   [11] fireflyClamp bits   (float — per-NEE luminance cap; 1e30 disables)
            //   [12] oceanFineTileSize bits (float — fine-cascade tile size in m;
            //        0 disables FFT-fine-normal perturbation on thinWalled hits)
            //   [13] edgeMsaaExtra      (raygen — extra primary samples at detected
            //        silhouettes; 0 disables, N → (N+1)× total samples at edges)
            //   [14] oceanFoamTileSize bits (float — world-space foam tile size in m;
            //        0 disables foam sampling on water hits)
            //   [15] maxBounces        (raygen — host-driven scatter-event cap;
            //        replaces the prior compile-time kMaxBounces in raygen.rgen)
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                 VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                 VK_SHADER_STAGE_MISS_BIT_KHR;
            pcRange.offset = 0;
            pcRange.size   = 64;

            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &rtDsLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &rtPipelineLayout),
                  "vkCreatePipelineLayout(RT)");

            // Pipeline + SBT creation is deferred to buildAllRtPipelines(),
            // called lazily on first frame once ensureSceneBuilt has detected
            // scene features (sceneHasGlass_ etc.). That lets us spec-constant
            // those features into chit without bloating the variant matrix.
        }

        // Builds all RT pipeline variants for the current scene-feature set.
        // Called lazily on first frame after ensureSceneBuilt has detected
        // sceneHasGlass_ and other scene-fixed properties. Variant axes are
        // SER × hybrid × restirDI (runtime-toggleable); scene features go in
        // as a uint bitmask spec constant on chit and don't multiply variants.
        bool rtPipelinesBuilt_ = false;
        // Snapshot of currentSceneFeatures() used at the most recent variant
        // build. Compared every frame against the live value; if the scene
        // swapped to a model with different features (gltf_samples-style),
        // we invalidate all baked variants and rebuild the active one fresh.
        // Sentinel 0xFFFFFFFFu means "no build has happened yet" — initial
        // mismatch on first frame is fine because rtPipelinesBuilt_ gates
        // the invalidation path.
        uint32_t lastBuiltSceneFeatures_ = 0xFFFFFFFFu;

        // First-frame entry. Builds only the variant matching the user's
        // current toggle state; the other 7 (or 3, non-SER) variants are
        // built lazily by ensureCurrentRtVariantBuilt() on first toggle.
        // RT-pipeline compile is expensive (~1-2s each on NVIDIA for
        // threepp's chit), so eager-building all 8 was costing 10+s. Most
        // users never toggle, so we pay 1 compile at startup and only the
        // others if the user actually flips a switch.
        void buildAllRtPipelines() {
            const uint32_t sceneFeats = currentSceneFeatures();
            buildSingleRtVariant(shouldUseSerRaygen(), restirDIEnabled_, sceneFeats);
            lastBuiltSceneFeatures_ = sceneFeats;
            rtPipelinesBuilt_ = true;
        }

        // Throws out every built pipeline + SBT after a scene swap. Called
        // from ensureCurrentRtVariantBuilt() when currentSceneFeatures() has
        // changed since the last build (e.g. a glass model replaced a glass-
        // free one — kSceneFeatHasGlass flips and the caustic gather DCE is
        // now wrong). vkDeviceWaitIdle drains any in-flight GPU work using
        // the old pipelines before we destroy them.
        void invalidateAllRtVariants() {
            vkDeviceWaitIdle(ctx->device());
            for (auto& v : rtVariants_) {
                if (v.pipeline != VK_NULL_HANDLE) {
                    vkDestroyPipeline(ctx->device(), v.pipeline, nullptr);
                    v.pipeline = VK_NULL_HANDLE;
                }
                destroyBuffer(ctx->allocator(), v.sbtBuffer);
                v.sbtBuffer  = {};
                v.rgenRegion = {};
                v.missRegion = {};
                v.hitRegion  = {};
            }
        }

        // Lazy build of the active variant. Called every frame (cheap null
        // check); also detects scene-feature drift (gltf_samples model swap)
        // and tears down baked-stale variants. vkDeviceWaitIdle before the
        // create call drains any in-flight GPU work — NVIDIA's reported
        // hang on subsequent vkCreateRayTracingPipelinesKHR calls against
        // the same pipeline layout appears to be tied to overlapping work.
        void ensureCurrentRtVariantBuilt() {
            const uint32_t newFeats = currentSceneFeatures();
            if (rtPipelinesBuilt_ && newFeats != lastBuiltSceneFeatures_) {
                // Scene features changed since last build (model swap).
                // All baked spec-constant variants are stale — caustic
                // gather / clearcoat lobe / iridescence Fresnel / sheen NEE
                // DCE states no longer match what the chit needs. Discard
                // everything; the active variant will rebuild below.
                invalidateAllRtVariants();
                lastBuiltSceneFeatures_ = newFeats;
            }
            const bool useSer = shouldUseSerRaygen();
            const bool rdi = restirDIEnabled_;
            const uint32_t idx = rtVariantIndex(useSer, rdi);
            if (rtVariants_[idx].pipeline != VK_NULL_HANDLE) return;
            vkDeviceWaitIdle(ctx->device());
            buildSingleRtVariant(useSer, rdi, newFeats);
        }

        // Scene-feature bitmask matching chit's kSceneFeatures spec constant.
        // Bit assignments must stay in lock-step with closest_hit.rchit
        // (kSceneFeatHasGlass / Clearcoat / Iridescence / Sheen).
        //   bit 0 = sceneHasGlass        — gates caustic photon gather
        //   bit 1 = sceneHasClearcoat    — gates clearcoat lobe setup
        //   bit 2 = sceneHasIridescence  — gates evalIridescence Fresnel mix
        //   bit 3 = sceneHasSheen        — gates Charlie NDF sheen NEE terms
        uint32_t currentSceneFeatures() const {
            uint32_t f = 0;
            if (sceneHasGlass_)        f |= 1u;
            if (sceneHasClearcoat_)    f |= 2u;
            if (sceneHasIridescence_)  f |= 4u;
            if (sceneHasSheen_)        f |= 8u;
            return f;
        }

        // Single-variant pipeline build. Called once at first frame for the
        // active variant, and again by ensureCurrentRtVariantBuilt() on each
        // toggle that lands in an unbuilt slot.
        void buildSingleRtVariant(bool useSer, bool restirDISpec,
                                  uint32_t sceneFeatures) {
            auto loadModule = [this](const uint32_t* code, size_t size) {
                VkShaderModuleCreateInfo smci{};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                smci.codeSize = size;
                smci.pCode = code;
                VkShaderModule m = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &m),
                      "vkCreateShaderModule(RT)");
                return m;
            };

            VkShaderModule rgenMod = useSer
                    ? loadModule(kRaygenRgenSerSpv, sizeof(kRaygenRgenSerSpv))
                    : loadModule(kRaygenRgenSpv,    sizeof(kRaygenRgenSpv));
            VkShaderModule missMod    = loadModule(kMissRmissSpv,            sizeof(kMissRmissSpv));
            VkShaderModule sMissMod   = loadModule(kShadowMissRmissSpv,      sizeof(kShadowMissRmissSpv));
            VkShaderModule chitMod    = loadModule(kClosestHitRchitSpv,      sizeof(kClosestHitRchitSpv));
            VkShaderModule ahitMod    = loadModule(kClosestHitAlphaRahitSpv, sizeof(kClosestHitAlphaRahitSpv));
            VkShaderModule sahitMod   = loadModule(kShadowAnyhitRahitSpv,    sizeof(kShadowAnyhitRahitSpv));

            // Shader groups: same layout as other variants.
            std::array<VkRayTracingShaderGroupCreateInfoKHR, 5> groups{};
            for (auto& g : groups) {
                g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                g.generalShader = VK_SHADER_UNUSED_KHR;
                g.closestHitShader = VK_SHADER_UNUSED_KHR;
                g.anyHitShader = VK_SHADER_UNUSED_KHR;
                g.intersectionShader = VK_SHADER_UNUSED_KHR;
            }
            groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            groups[0].generalShader = 0;
            groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            groups[1].generalShader = 1;
            groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            groups[2].generalShader = 2;
            groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            groups[3].closestHitShader = 3;
            groups[3].anyHitShader = 4;
            groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            groups[4].anyHitShader = 5;

            // Raygen spec data (kSceneFeatures at constant_id 1).
            struct RgenSpecData {
                uint32_t sceneFeatures;
            } rgenSpecData{
                sceneFeatures,
            };
            VkSpecializationMapEntry rgenMapEntries[1]{};
            rgenMapEntries[0].constantID = 1;
            rgenMapEntries[0].offset = offsetof(RgenSpecData, sceneFeatures);
            rgenMapEntries[0].size = sizeof(uint32_t);
            VkSpecializationInfo rgenSpecInfo{};
            rgenSpecInfo.mapEntryCount = 1;
            rgenSpecInfo.pMapEntries = rgenMapEntries;
            rgenSpecInfo.dataSize = sizeof(RgenSpecData);
            rgenSpecInfo.pData = &rgenSpecData;

            // Chit spec data (kRestirDIEnabled + kSceneFeatures).
            struct ChitSpecData {
                VkBool32 restirDIEnabled;
                uint32_t sceneFeatures;
            } chitSpecData{
                restirDISpec ? VK_TRUE : VK_FALSE,
                sceneFeatures,
            };
            VkSpecializationMapEntry chitMapEntries[2]{};
            chitMapEntries[0].constantID = 0;
            chitMapEntries[0].offset = offsetof(ChitSpecData, restirDIEnabled);
            chitMapEntries[0].size = sizeof(VkBool32);
            chitMapEntries[1].constantID = 1;
            chitMapEntries[1].offset = offsetof(ChitSpecData, sceneFeatures);
            chitMapEntries[1].size = sizeof(uint32_t);
            VkSpecializationInfo chitSpecInfo{};
            chitSpecInfo.mapEntryCount = 2;
            chitSpecInfo.pMapEntries = chitMapEntries;
            chitSpecInfo.dataSize = sizeof(ChitSpecData);
            chitSpecInfo.pData = &chitSpecData;

            // Stages: 0=rgen, 1=primary miss, 2=shadow miss,
            //         3=path closest-hit, 4=path any-hit (alpha),
            //         5=shadow any-hit (glass+cutout).
            std::array<VkPipelineShaderStageCreateInfo, 6> stages{};
            for (auto& s : stages) {
                s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                s.pName = "main";
            }
            stages[0].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            stages[0].module = rgenMod;
            stages[0].pSpecializationInfo = &rgenSpecInfo;
            stages[1].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
            stages[1].module = missMod;
            stages[2].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
            stages[2].module = sMissMod;
            stages[3].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            stages[3].module = chitMod;
            stages[3].pSpecializationInfo = &chitSpecInfo;
            stages[4].stage  = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            stages[4].module = ahitMod;
            stages[5].stage  = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            stages[5].module = sahitMod;

            VkRayTracingPipelineCreateInfoKHR rci{};
            rci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
            rci.stageCount = static_cast<uint32_t>(stages.size());
            rci.pStages    = stages.data();
            rci.groupCount = static_cast<uint32_t>(groups.size());
            rci.pGroups    = groups.data();
            // depth 4: primary chit → GI sub-trace → sub-sub-trace → shadow.
            rci.maxPipelineRayRecursionDepth = 4;
            rci.layout = rtPipelineLayout;

            const uint32_t idx = rtVariantIndex(useSer, restirDISpec);
            check(ctx->rt().createRayTracingPipelines(
                          ctx->device(), VK_NULL_HANDLE, VK_NULL_HANDLE,
                          1, &rci, nullptr, &rtVariants_[idx].pipeline),
                  "vkCreateRayTracingPipelinesKHR (single)");
            createShaderBindingTable(useSer, restirDISpec);

            vkDestroyShaderModule(ctx->device(), rgenMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), missMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), sMissMod, nullptr);
            vkDestroyShaderModule(ctx->device(), chitMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), ahitMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), sahitMod, nullptr);
        }

        void createShaderBindingTable(bool useSer, bool restirDISpec) {
            RtPipelineVariant& v = rtVariants_[rtVariantIndex(useSer, restirDISpec)];
            const auto& props = ctx->rtPipelineProperties();
            const uint32_t handleSize = props.shaderGroupHandleSize;
            const uint32_t handleAlignment = props.shaderGroupHandleAlignment;
            const uint32_t baseAlignment = props.shaderGroupBaseAlignment;
            const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

            // 5 groups: rgen, primary miss, shadow miss, path hit, shadow hit.
            // Miss region: 2 records. Hit region: 2 records (path=sbtOffset 0, shadow=sbtOffset 1).
            constexpr uint32_t groupCount = 5;
            const uint32_t handlesDataSize = groupCount * handleSize;
            std::vector<uint8_t> handles(handlesDataSize);
            check(ctx->rt().getRayTracingShaderGroupHandles(
                          ctx->device(), v.pipeline, 0, groupCount,
                          handlesDataSize, handles.data()),
                  "vkGetRayTracingShaderGroupHandlesKHR");

            // Per-region: base aligned to shaderGroupBaseAlignment, stride = aligned
            // handle size. Region size must be a multiple of stride and >= base alignment.
            const uint32_t rgenRegionBytes = alignUp(handleSizeAligned, baseAlignment);
            const uint32_t missRegionBytes = alignUp(2 * handleSizeAligned, baseAlignment);
            const uint32_t hitRegionBytes  = alignUp(2 * handleSizeAligned, baseAlignment);
            const VkDeviceSize sbtSize =
                    static_cast<VkDeviceSize>(rgenRegionBytes) +
                    static_cast<VkDeviceSize>(missRegionBytes) +
                    static_cast<VkDeviceSize>(hitRegionBytes);

            v.sbtBuffer = createBuffer(
                    ctx->allocator(), ctx->device(), sbtSize,
                    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), v.sbtBuffer.alloc, &mapped);
            std::memset(mapped, 0, sbtSize);
            uint8_t* dst = static_cast<uint8_t*>(mapped);

            // rgen at start of buffer.
            std::memcpy(dst, handles.data() + 0 * handleSize, handleSize);
            // miss[0] = primary, miss[1] = shadow. Packed at handleSizeAligned stride.
            std::memcpy(dst + rgenRegionBytes + 0 * handleSizeAligned,
                        handles.data() + 1 * handleSize, handleSize);
            std::memcpy(dst + rgenRegionBytes + 1 * handleSizeAligned,
                        handles.data() + 2 * handleSize, handleSize);
            // hit[0] = path hit group (sbtOffset=0), hit[1] = shadow hit group (sbtOffset=1).
            std::memcpy(dst + rgenRegionBytes + missRegionBytes + 0 * handleSizeAligned,
                        handles.data() + 3 * handleSize, handleSize);
            std::memcpy(dst + rgenRegionBytes + missRegionBytes + 1 * handleSizeAligned,
                        handles.data() + 4 * handleSize, handleSize);
            vmaUnmapMemory(ctx->allocator(), v.sbtBuffer.alloc);

            const VkDeviceAddress base = v.sbtBuffer.address;
            v.rgenRegion.deviceAddress = base;
            v.rgenRegion.stride = rgenRegionBytes;
            v.rgenRegion.size   = rgenRegionBytes;
            v.missRegion.deviceAddress = base + rgenRegionBytes;
            v.missRegion.stride = handleSizeAligned;
            v.missRegion.size   = missRegionBytes;
            v.hitRegion.deviceAddress = base + rgenRegionBytes + missRegionBytes;
            v.hitRegion.stride = handleSizeAligned;
            v.hitRegion.size   = hitRegionBytes;
            callRegion = {};
        }

        void createDescriptorPool() {
            imageCount_ = static_cast<uint32_t>(ctx->swapchainImages().size());
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::array<VkDescriptorPoolSize, 5> ps{};
            ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            ps[0].descriptorCount = totalSets;
            ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ps[1].descriptorCount = totalSets * 22;// bindings 1, 7, 11, 12, 13 + 20×2 (filtered ping-pong) + 28-31 (ReSTIR DI reservoir ping-pong) + 33,34 (moments ping-pong) + 35,36,37 (albedo ping-pong + snapshot) + 38-43 (ReSTIR GI reservoir ping-pong)
            ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps[2].descriptorCount = totalSets * 4;// bindings 2 (camera), 5 (lights), 9 (prevCamera), 17 (fog)
            ps[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps[3].descriptorCount = totalSets * 7;// bindings 3,4,10,14 (existing) + 15,16 (photon) + 21 (meshMovedBits)
            ps[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[4].descriptorCount = totalSets * (3 + kMaxMaterialTextures + 5 + 1 + 1 + 1 + 1);// binding 6 (env) + binding 8 (material array) + bindings 18,19 (env CDF) + bindings 22-26 (hybrid gbuffer attachments incl. UV) + binding 27 (blue noise tile) + binding 32 (ocean fine-cascade height) + binding 44 (world-space foam) + binding 45 (foam detail tile)

            VkDescriptorPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.maxSets = totalSets;
            ci.poolSizeCount = static_cast<uint32_t>(ps.size());
            ci.pPoolSizes = ps.data();
            check(vkCreateDescriptorPool(ctx->device(), &ci, nullptr, &descriptorPool),
                  "vkCreateDescriptorPool(RT)");
        }

        void allocateAndUpdateDescriptors() {
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::vector<VkDescriptorSetLayout> layouts(totalSets, rtDsLayout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool;
            ai.descriptorSetCount = totalSets;
            ai.pSetLayouts = layouts.data();
            descriptorSets.resize(totalSets);
            check(vkAllocateDescriptorSets(ctx->device(), &ai, descriptorSets.data()),
                  "vkAllocateDescriptorSets(RT)");

            VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
            asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asWrite.accelerationStructureCount = 1;
            asWrite.pAccelerationStructures = &tlas;

            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                for (uint32_t i = 0; i < imageCount_; ++i) {
                    const uint32_t idx = f * imageCount_ + i;

                    VkWriteDescriptorSet wAS{};
                    wAS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wAS.pNext = &asWrite;
                    wAS.dstSet = descriptorSets[idx];
                    wAS.dstBinding = 0;
                    wAS.descriptorCount = 1;
                    wAS.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

                    // Binding 1 (denoise output) is initialised to the
                    // swapchain image as a placeholder; renderFrame rewrites
                    // it per-frame to taa_->inputView(f) so TAA can resolve
                    // it before swapchain present.
                    VkDescriptorImageInfo imgInfo{};
                    imgInfo.imageView = ctx->swapchainImageViews()[i];
                    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wImg{};
                    wImg.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wImg.dstSet = descriptorSets[idx];
                    wImg.dstBinding = 1;
                    wImg.descriptorCount = 1;
                    wImg.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wImg.pImageInfo = &imgInfo;

                    VkDescriptorBufferInfo bufInfo{};
                    bufInfo.buffer = cameraUbos[f].handle;
                    bufInfo.offset = 0;
                    bufInfo.range = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wUbo{};
                    wUbo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wUbo.dstSet = descriptorSets[idx];
                    wUbo.dstBinding = 2;
                    wUbo.descriptorCount = 1;
                    wUbo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    wUbo.pBufferInfo = &bufInfo;

                    VkDescriptorBufferInfo geomInfo{};
                    geomInfo.buffer = geometryDescsBuffer.handle;
                    geomInfo.offset = 0;
                    geomInfo.range = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wGeom{};
                    wGeom.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGeom.dstSet = descriptorSets[idx];
                    wGeom.dstBinding = 3;
                    wGeom.descriptorCount = 1;
                    wGeom.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wGeom.pBufferInfo = &geomInfo;

                    VkDescriptorBufferInfo matInfo{};
                    matInfo.buffer = materialDescsBuffers[f].handle;
                    matInfo.offset = 0;
                    matInfo.range = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wMat{};
                    wMat.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMat.dstSet = descriptorSets[idx];
                    wMat.dstBinding = 4;
                    wMat.descriptorCount = 1;
                    wMat.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wMat.pBufferInfo = &matInfo;

                    VkDescriptorBufferInfo lightsInfo{};
                    lightsInfo.buffer = lightsUbos[f].handle;
                    lightsInfo.offset = 0;
                    lightsInfo.range = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wLights{};
                    wLights.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wLights.dstSet = descriptorSets[idx];
                    wLights.dstBinding = 5;
                    wLights.descriptorCount = 1;
                    wLights.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    wLights.pBufferInfo = &lightsInfo;

                    VkDescriptorImageInfo envInfo{};
                    envInfo.sampler     = envImage.sampler;
                    envInfo.imageView   = envImage.view;
                    envInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wEnv{};
                    wEnv.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wEnv.dstSet = descriptorSets[idx];
                    wEnv.dstBinding = 6;
                    wEnv.descriptorCount = 1;
                    wEnv.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wEnv.pImageInfo = &envInfo;

                    // Ping-pong: frame f writes slot (f & 1), reads (1 - (f & 1)).
                    // Two frames in flight + two slots align perfectly: descriptors
                    // are baked once and never need rewrite per frame.
                    const uint32_t writeSlot = f & 1u;
                    const uint32_t readSlot  = 1u - writeSlot;

                    VkDescriptorImageInfo accumInfo{};
                    accumInfo.imageView   = accumImagesPP[writeSlot].view;
                    accumInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wAccum{};
                    wAccum.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wAccum.dstSet = descriptorSets[idx];
                    wAccum.dstBinding = 7;
                    wAccum.descriptorCount = 1;
                    wAccum.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wAccum.pImageInfo = &accumInfo;

                    // Binding 8 — fill all kMaxMaterialTextures slots; unused
                    // tail slots get the white default so the descriptor write
                    // is always valid.
                    std::array<VkDescriptorImageInfo, kMaxMaterialTextures> matTexInfos{};
                    fillMaterialTextureInfos(matTexInfos);
                    VkWriteDescriptorSet wMatTex{};
                    wMatTex.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMatTex.dstSet = descriptorSets[idx];
                    wMatTex.dstBinding = 8;
                    wMatTex.dstArrayElement = 0;
                    wMatTex.descriptorCount = kMaxMaterialTextures;
                    wMatTex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wMatTex.pImageInfo = matTexInfos.data();

                    VkDescriptorBufferInfo prevCamInfo{};
                    prevCamInfo.buffer = prevCameraUbos[f].handle;
                    prevCamInfo.offset = 0;
                    prevCamInfo.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wPrevCam{};
                    wPrevCam.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wPrevCam.dstSet = descriptorSets[idx];
                    wPrevCam.dstBinding = 9;
                    wPrevCam.descriptorCount = 1;
                    wPrevCam.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    wPrevCam.pBufferInfo = &prevCamInfo;

                    VkDescriptorBufferInfo motionInfo{};
                    motionInfo.buffer = motionMatBuffers[f].handle;
                    motionInfo.offset = 0;
                    motionInfo.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wMotion{};
                    wMotion.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMotion.dstSet = descriptorSets[idx];
                    wMotion.dstBinding = 10;
                    wMotion.descriptorCount = 1;
                    wMotion.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wMotion.pBufferInfo = &motionInfo;

                    VkDescriptorImageInfo prevAccumInfo{};
                    prevAccumInfo.imageView   = accumImagesPP[readSlot].view;
                    prevAccumInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wPrevAccum{};
                    wPrevAccum.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wPrevAccum.dstSet = descriptorSets[idx];
                    wPrevAccum.dstBinding = 11;
                    wPrevAccum.descriptorCount = 1;
                    wPrevAccum.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wPrevAccum.pImageInfo = &prevAccumInfo;

                    VkDescriptorImageInfo gbufInfo{};
                    gbufInfo.imageView   = gbufImagesPP[writeSlot].view;
                    gbufInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wGbuf{};
                    wGbuf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGbuf.dstSet = descriptorSets[idx];
                    wGbuf.dstBinding = 12;
                    wGbuf.descriptorCount = 1;
                    wGbuf.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wGbuf.pImageInfo = &gbufInfo;

                    VkDescriptorImageInfo prevGbufInfo{};
                    prevGbufInfo.imageView   = gbufImagesPP[readSlot].view;
                    prevGbufInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wPrevGbuf{};
                    wPrevGbuf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wPrevGbuf.dstSet = descriptorSets[idx];
                    wPrevGbuf.dstBinding = 13;
                    wPrevGbuf.descriptorCount = 1;
                    wPrevGbuf.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wPrevGbuf.pImageInfo = &prevGbufInfo;

                    VkDescriptorBufferInfo emTriInfo{};
                    emTriInfo.buffer = emissiveTriBuffers[f].handle;
                    emTriInfo.offset = 0;
                    emTriInfo.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wEmTri{};
                    wEmTri.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wEmTri.dstSet = descriptorSets[idx];
                    wEmTri.dstBinding = 14;
                    wEmTri.descriptorCount = 1;
                    wEmTri.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wEmTri.pBufferInfo = &emTriInfo;

                    VkDescriptorBufferInfo photonCntInfo{};
                    photonCntInfo.buffer = photon_->countBuffer();
                    photonCntInfo.offset = 0;
                    photonCntInfo.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wPhotonCnt{};
                    wPhotonCnt.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wPhotonCnt.dstSet = descriptorSets[idx];
                    wPhotonCnt.dstBinding = 15;
                    wPhotonCnt.descriptorCount = 1;
                    wPhotonCnt.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wPhotonCnt.pBufferInfo = &photonCntInfo;

                    VkDescriptorBufferInfo photonDataInfo{};
                    photonDataInfo.buffer = photon_->dataBuffer();
                    photonDataInfo.offset = 0;
                    photonDataInfo.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wPhotonData{};
                    wPhotonData.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wPhotonData.dstSet = descriptorSets[idx];
                    wPhotonData.dstBinding = 16;
                    wPhotonData.descriptorCount = 1;
                    wPhotonData.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wPhotonData.pBufferInfo = &photonDataInfo;

                    VkDescriptorBufferInfo fogInfo{};
                    fogInfo.buffer = fogUbos[f].handle;
                    fogInfo.offset = 0;
                    fogInfo.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wFog{};
                    wFog.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wFog.dstSet = descriptorSets[idx];
                    wFog.dstBinding = 17;
                    wFog.descriptorCount = 1;
                    wFog.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    wFog.pBufferInfo = &fogInfo;

                    VkDescriptorImageInfo envCdfInfo{};
                    envCdfInfo.sampler     = envCdfImage.sampler;
                    envCdfInfo.imageView   = envCdfImage.view;
                    envCdfInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wEnvCdf{};
                    wEnvCdf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wEnvCdf.dstSet = descriptorSets[idx];
                    wEnvCdf.dstBinding = 18;
                    wEnvCdf.descriptorCount = 1;
                    wEnvCdf.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wEnvCdf.pImageInfo = &envCdfInfo;

                    VkDescriptorImageInfo envMargInfo{};
                    envMargInfo.sampler     = envMargImage.sampler;
                    envMargInfo.imageView   = envMargImage.view;
                    envMargInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wEnvMarg{};
                    wEnvMarg.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wEnvMarg.dstSet = descriptorSets[idx];
                    wEnvMarg.dstBinding = 19;
                    wEnvMarg.descriptorCount = 1;
                    wEnvMarg.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wEnvMarg.pImageInfo = &envMargInfo;

                    // Binding 20 — denoise à-trous ping-pong (rgba32f, count=2).
                    // Both filtered slots are baked here; the shader uses
                    // filteredArray[N] indexed by a per-pass push constant.
                    std::array<VkDescriptorImageInfo, 2> filteredInfos{};
                    filteredInfos[0].imageView   = denoiser_->filteredView(0);
                    filteredInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    filteredInfos[1].imageView   = denoiser_->filteredView(1);
                    filteredInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wFiltered{};
                    wFiltered.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wFiltered.dstSet = descriptorSets[idx];
                    wFiltered.dstBinding = 20;
                    wFiltered.dstArrayElement = 0;
                    wFiltered.descriptorCount = 2;
                    wFiltered.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wFiltered.pImageInfo = filteredInfos.data();

                    VkDescriptorBufferInfo meshMovedInfo{};
                    meshMovedInfo.buffer = meshMovedBitsBuffers[f].handle;
                    meshMovedInfo.offset = 0;
                    meshMovedInfo.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wMeshMoved{};
                    wMeshMoved.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMeshMoved.dstSet = descriptorSets[idx];
                    wMeshMoved.dstBinding = 21;
                    wMeshMoved.descriptorCount = 1;
                    wMeshMoved.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wMeshMoved.pBufferInfo = &meshMovedInfo;

                    // Bindings 22-26: hybrid gbuffer attachments. raygen at
                    // frame f reads rasterGbufs[f].* (built one frame ahead).
                    // ensureHybridResources runs in the ctor before this, so
                    // the views are valid.
                    VkDescriptorImageInfo gbufNormalInfo{};
                    gbufNormalInfo.sampler     = gbufSampler_;
                    gbufNormalInfo.imageView   = rasterGbufs[f].normal.view;
                    gbufNormalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wGbufNormal{};
                    wGbufNormal.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGbufNormal.dstSet = descriptorSets[idx];
                    wGbufNormal.dstBinding = 22;
                    wGbufNormal.descriptorCount = 1;
                    wGbufNormal.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wGbufNormal.pImageInfo = &gbufNormalInfo;

                    VkDescriptorImageInfo gbufMotionInfo{};
                    gbufMotionInfo.sampler     = gbufSampler_;
                    gbufMotionInfo.imageView   = rasterGbufs[f].motion.view;
                    gbufMotionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wGbufMotion{};
                    wGbufMotion.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGbufMotion.dstSet = descriptorSets[idx];
                    wGbufMotion.dstBinding = 23;
                    wGbufMotion.descriptorCount = 1;
                    wGbufMotion.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wGbufMotion.pImageInfo = &gbufMotionInfo;

                    VkDescriptorImageInfo gbufDepthInfo{};
                    gbufDepthInfo.sampler     = gbufSampler_;
                    gbufDepthInfo.imageView   = rasterGbufs[f].depth.view;
                    gbufDepthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wGbufDepth{};
                    wGbufDepth.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGbufDepth.dstSet = descriptorSets[idx];
                    wGbufDepth.dstBinding = 24;
                    wGbufDepth.descriptorCount = 1;
                    wGbufDepth.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wGbufDepth.pImageInfo = &gbufDepthInfo;

                    VkDescriptorImageInfo gbufIdsInfo{};
                    gbufIdsInfo.sampler     = gbufSampler_;
                    gbufIdsInfo.imageView   = rasterGbufs[f].ids.view;
                    gbufIdsInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wGbufIds{};
                    wGbufIds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGbufIds.dstSet = descriptorSets[idx];
                    wGbufIds.dstBinding = 25;
                    wGbufIds.descriptorCount = 1;
                    wGbufIds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wGbufIds.pImageInfo = &gbufIdsInfo;

                    VkDescriptorImageInfo gbufUvInfo{};
                    gbufUvInfo.sampler     = gbufSampler_;
                    gbufUvInfo.imageView   = rasterGbufs[f].uv.view;
                    gbufUvInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wGbufUv{};
                    wGbufUv.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGbufUv.dstSet = descriptorSets[idx];
                    wGbufUv.dstBinding = 26;
                    wGbufUv.descriptorCount = 1;
                    wGbufUv.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wGbufUv.pImageInfo = &gbufUvInfo;

                    VkDescriptorImageInfo blueNoiseInfo{};
                    blueNoiseInfo.sampler     = blueNoiseImage.sampler;
                    blueNoiseInfo.imageView   = blueNoiseImage.view;
                    blueNoiseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wBlueNoise{};
                    wBlueNoise.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wBlueNoise.dstSet = descriptorSets[idx];
                    wBlueNoise.dstBinding = 27;
                    wBlueNoise.descriptorCount = 1;
                    wBlueNoise.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wBlueNoise.pImageInfo = &blueNoiseInfo;

                    // Bindings 28-31 — ReSTIR DI Stage 1b reservoir ping-pong.
                    // 28 = pos+type write (this frame), 29 = pos+type read (prev),
                    // 30 = W_sum/M/W/p_hat write, 31 = same read. f&1 picks slot.
                    VkDescriptorImageInfo resPosWriteInfo{};
                    resPosWriteInfo.imageView   = reservoirPosImagesPP[writeSlot].view;
                    resPosWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wResPosWrite{};
                    wResPosWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wResPosWrite.dstSet = descriptorSets[idx];
                    wResPosWrite.dstBinding = 28;
                    wResPosWrite.descriptorCount = 1;
                    wResPosWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wResPosWrite.pImageInfo = &resPosWriteInfo;

                    VkDescriptorImageInfo resPosReadInfo{};
                    resPosReadInfo.imageView   = reservoirPosImagesPP[readSlot].view;
                    resPosReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wResPosRead{};
                    wResPosRead.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wResPosRead.dstSet = descriptorSets[idx];
                    wResPosRead.dstBinding = 29;
                    wResPosRead.descriptorCount = 1;
                    wResPosRead.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wResPosRead.pImageInfo = &resPosReadInfo;

                    VkDescriptorImageInfo resWWriteInfo{};
                    resWWriteInfo.imageView   = reservoirWImagesPP[writeSlot].view;
                    resWWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wResWWrite{};
                    wResWWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wResWWrite.dstSet = descriptorSets[idx];
                    wResWWrite.dstBinding = 30;
                    wResWWrite.descriptorCount = 1;
                    wResWWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wResWWrite.pImageInfo = &resWWriteInfo;

                    VkDescriptorImageInfo resWReadInfo{};
                    resWReadInfo.imageView   = reservoirWImagesPP[readSlot].view;
                    resWReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wResWRead{};
                    wResWRead.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wResWRead.dstSet = descriptorSets[idx];
                    wResWRead.dstBinding = 31;
                    wResWRead.descriptorCount = 1;
                    wResWRead.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wResWRead.pImageInfo = &resWReadInfo;

                    // Binding 32 — ocean fine-cascade height (default dummy until
                    // a DisplacedMesh appears and rewrites the view via
                    // rewriteOceanFineDescriptors).
                    VkDescriptorImageInfo oceanFineInfo{};
                    oceanFineInfo.sampler     = oceanFineHeightSampler;
                    oceanFineInfo.imageView   = oceanFineHeightView;
                    oceanFineInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wOceanFine{};
                    wOceanFine.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wOceanFine.dstSet = descriptorSets[idx];
                    wOceanFine.dstBinding = 32;
                    wOceanFine.descriptorCount = 1;
                    wOceanFine.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wOceanFine.pImageInfo = &oceanFineInfo;

                    // Bindings 33/34 — temporal moments ping-pong (R32F).
                    // Same writeSlot/readSlot mapping as accumImagesPP.
                    VkDescriptorImageInfo momentsWriteInfo{};
                    momentsWriteInfo.imageView   = denoiser_->momentsView(writeSlot);
                    momentsWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wMomentsWrite{};
                    wMomentsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMomentsWrite.dstSet = descriptorSets[idx];
                    wMomentsWrite.dstBinding = 33;
                    wMomentsWrite.descriptorCount = 1;
                    wMomentsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wMomentsWrite.pImageInfo = &momentsWriteInfo;

                    VkDescriptorImageInfo momentsReadInfo{};
                    momentsReadInfo.imageView   = denoiser_->momentsView(readSlot);
                    momentsReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wMomentsRead{};
                    wMomentsRead.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMomentsRead.dstSet = descriptorSets[idx];
                    wMomentsRead.dstBinding = 34;
                    wMomentsRead.descriptorCount = 1;
                    wMomentsRead.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wMomentsRead.pImageInfo = &momentsReadInfo;

                    // Bindings 35/36 — primary-surface albedo ping-pong
                    // (RGBA8). Raygen FC-blends prev albedo (36, prev slot)
                    // with chit's payload.primaryAlbedo, writes to 35
                    // (current write slot). Atrous reads 35. Same write/
                    // read slot mapping as accumImagesPP / momentsImagesPP.
                    VkDescriptorImageInfo albedoWriteInfo{};
                    albedoWriteInfo.imageView   = denoiser_->albedoView(writeSlot);
                    albedoWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wAlbedoWrite{};
                    wAlbedoWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wAlbedoWrite.dstSet = descriptorSets[idx];
                    wAlbedoWrite.dstBinding = 35;
                    wAlbedoWrite.descriptorCount = 1;
                    wAlbedoWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wAlbedoWrite.pImageInfo = &albedoWriteInfo;

                    VkDescriptorImageInfo albedoReadInfo{};
                    albedoReadInfo.imageView   = denoiser_->albedoView(readSlot);
                    albedoReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wAlbedoRead{};
                    wAlbedoRead.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wAlbedoRead.dstSet = descriptorSets[idx];
                    wAlbedoRead.dstBinding = 36;
                    wAlbedoRead.descriptorCount = 1;
                    wAlbedoRead.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wAlbedoRead.pImageInfo = &albedoReadInfo;

                    // Binding 37 — snapshot albedo (single buffer, not
                    // ping-pong). Same view across all frames.
                    VkDescriptorImageInfo albedoSnapInfo{};
                    albedoSnapInfo.imageView   = denoiser_->albedoSnapshotView();
                    albedoSnapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wAlbedoSnap{};
                    wAlbedoSnap.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wAlbedoSnap.dstSet = descriptorSets[idx];
                    wAlbedoSnap.dstBinding = 37;
                    wAlbedoSnap.descriptorCount = 1;
                    wAlbedoSnap.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wAlbedoSnap.pImageInfo = &albedoSnapInfo;

                    // Bindings 38-43 — ReSTIR GI reservoir ping-pong storage.
                    // 38/40/42 = write (this frame), 39/41/43 = read (prev).
                    // f&1 picks slot, same as DI's reservoir bindings.
                    VkDescriptorImageInfo giXsWriteInfo{};
                    giXsWriteInfo.imageView   = giResXsImagesPP[writeSlot].view;
                    giXsWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wGiXsWrite{};
                    wGiXsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGiXsWrite.dstSet = descriptorSets[idx];
                    wGiXsWrite.dstBinding = 38;
                    wGiXsWrite.descriptorCount = 1;
                    wGiXsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wGiXsWrite.pImageInfo = &giXsWriteInfo;

                    VkDescriptorImageInfo giXsReadInfo{};
                    giXsReadInfo.imageView   = giResXsImagesPP[readSlot].view;
                    giXsReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wGiXsRead{};
                    wGiXsRead.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGiXsRead.dstSet = descriptorSets[idx];
                    wGiXsRead.dstBinding = 39;
                    wGiXsRead.descriptorCount = 1;
                    wGiXsRead.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wGiXsRead.pImageInfo = &giXsReadInfo;

                    VkDescriptorImageInfo giNsWriteInfo{};
                    giNsWriteInfo.imageView   = giResNsImagesPP[writeSlot].view;
                    giNsWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wGiNsWrite{};
                    wGiNsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGiNsWrite.dstSet = descriptorSets[idx];
                    wGiNsWrite.dstBinding = 40;
                    wGiNsWrite.descriptorCount = 1;
                    wGiNsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wGiNsWrite.pImageInfo = &giNsWriteInfo;

                    VkDescriptorImageInfo giNsReadInfo{};
                    giNsReadInfo.imageView   = giResNsImagesPP[readSlot].view;
                    giNsReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wGiNsRead{};
                    wGiNsRead.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGiNsRead.dstSet = descriptorSets[idx];
                    wGiNsRead.dstBinding = 41;
                    wGiNsRead.descriptorCount = 1;
                    wGiNsRead.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wGiNsRead.pImageInfo = &giNsReadInfo;

                    VkDescriptorImageInfo giLoWriteInfo{};
                    giLoWriteInfo.imageView   = giResLoImagesPP[writeSlot].view;
                    giLoWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wGiLoWrite{};
                    wGiLoWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGiLoWrite.dstSet = descriptorSets[idx];
                    wGiLoWrite.dstBinding = 42;
                    wGiLoWrite.descriptorCount = 1;
                    wGiLoWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wGiLoWrite.pImageInfo = &giLoWriteInfo;

                    VkDescriptorImageInfo giLoReadInfo{};
                    giLoReadInfo.imageView   = giResLoImagesPP[readSlot].view;
                    giLoReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wGiLoRead{};
                    wGiLoRead.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wGiLoRead.dstSet = descriptorSets[idx];
                    wGiLoRead.dstBinding = 43;
                    wGiLoRead.descriptorCount = 1;
                    wGiLoRead.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wGiLoRead.pImageInfo = &giLoReadInfo;

                    // Binding 44 — world-space foam (dummy by default; the
                    // active DisplacedMesh rewrites this once it comes online
                    // via rewriteOceanFoamDescriptors_).
                    VkDescriptorImageInfo oceanFoamInfo{};
                    oceanFoamInfo.sampler     = oceanFoamSampler;
                    oceanFoamInfo.imageView   = oceanFoamView;
                    oceanFoamInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wOceanFoam{};
                    wOceanFoam.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wOceanFoam.dstSet = descriptorSets[idx];
                    wOceanFoam.dstBinding = 44;
                    wOceanFoam.descriptorCount = 1;
                    wOceanFoam.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wOceanFoam.pImageInfo = &oceanFoamInfo;

                    // Binding 45 — baked tileable foam detail (static, mipped,
                    // SHADER_READ_ONLY for the renderer's lifetime).
                    VkDescriptorImageInfo foamDetailInfo{};
                    foamDetailInfo.sampler     = foamDetailImage.sampler;
                    foamDetailInfo.imageView   = foamDetailImage.view;
                    foamDetailInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    VkWriteDescriptorSet wFoamDetail{};
                    wFoamDetail.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wFoamDetail.dstSet = descriptorSets[idx];
                    wFoamDetail.dstBinding = 45;
                    wFoamDetail.descriptorCount = 1;
                    wFoamDetail.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wFoamDetail.pImageInfo = &foamDetailInfo;

                    std::array<VkWriteDescriptorSet, 46> writes{
                            wAS, wImg, wUbo, wGeom, wMat, wLights, wEnv, wAccum, wMatTex,
                            wPrevCam, wMotion, wPrevAccum, wGbuf, wPrevGbuf, wEmTri,
                            wPhotonCnt, wPhotonData, wFog, wEnvCdf, wEnvMarg, wFiltered,
                            wMeshMoved,
                            wGbufNormal, wGbufMotion, wGbufDepth, wGbufIds, wGbufUv,
                            wBlueNoise,
                            wResPosWrite, wResPosRead, wResWWrite, wResWRead,
                            wOceanFine,
                            wMomentsWrite, wMomentsRead,
                            wAlbedoWrite, wAlbedoRead, wAlbedoSnap,
                            wGiXsWrite, wGiXsRead, wGiNsWrite, wGiNsRead,
                            wGiLoWrite, wGiLoRead,
                            wOceanFoam, wFoamDetail};
                    vkUpdateDescriptorSets(ctx->device(),
                                           static_cast<uint32_t>(writes.size()),
                                           writes.data(), 0, nullptr);
                }
            }
            // Binding 1 just got initialised to swapchain views (see imgInfo
            // above) — record that so renderFrame's per-frame rewrite block
            // can detect a hybrid-on transition without firing on every frame.
            binding1Mode_.fill(0);

            // The TLAS + material buffer the raster-first deferred pass binds
            // are now valid (this runs right after the scene/TLAS build), so
            // refresh its descriptor set. No-op until ray query is supported.
            rewriteDeferredDescriptors();
        }

        // Populate `infos` with the current bindless material-texture array,
        // padding any unused tail slots — and any slots reclaimed by prune —
        // with the slot-0 white default. Reclaimed slots are detected via a
        // null view (Image2D is reset to {} on destroy).
        template <std::size_t N>
        void fillMaterialTextureInfos(std::array<VkDescriptorImageInfo, N>& infos) const {
            const VkImageView fallbackView = materialTextures[0].view;
            const VkSampler   fallbackSampler = materialTextures[0].sampler;
            for (std::size_t i = 0; i < N; ++i) {
                const bool hasSlot = i < materialTextures.size() && materialTextures[i].view;
                infos[i].sampler     = hasSlot ? materialTextures[i].sampler : fallbackSampler;
                infos[i].imageView   = hasSlot ? materialTextures[i].view    : fallbackView;
                infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }

        // Rewrite bindings 0 (TLAS), 3 (geom desc), 4 (mat desc), 8 (material
        // texture array) across every descriptor set. Used by ensureSceneBuilt's
        // rebuild path so we don't have to reset the descriptor pool when the
        // scene changes. Binding 8 is rewritten too because a rebuild may have
        // added new material textures (kept-alive tail slots still point at
        // the white default; harmless to redundantly re-bind).
        void rewriteSceneDescriptors() {
            const uint32_t totalSets = imageCount_ * kFramesInFlight;

            VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
            asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asWrite.accelerationStructureCount = 1;
            asWrite.pAccelerationStructures = &tlas;

            VkDescriptorBufferInfo geomInfo{};
            geomInfo.buffer = geometryDescsBuffer.handle;
            geomInfo.offset = 0;
            geomInfo.range = VK_WHOLE_SIZE;

            // Per-frame matDescs ring — set idx maps to frame f = idx / imageCount_,
            // and that set must bind materialDescsBuffers[f] (each frame-in-flight
            // owns its own slot so the hot path can flush without a device wait).
            std::array<VkDescriptorBufferInfo, kFramesInFlight> matInfos{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                matInfos[f].buffer = materialDescsBuffers[f].handle;
                matInfos[f].offset = 0;
                matInfos[f].range  = VK_WHOLE_SIZE;
            }

            std::array<VkDescriptorImageInfo, kMaxMaterialTextures> matTexInfos{};
            fillMaterialTextureInfos(matTexInfos);

            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(totalSets * 4);
            for (uint32_t i = 0; i < totalSets; ++i) {
                const uint32_t f = i / imageCount_;
                VkWriteDescriptorSet wAS{};
                wAS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wAS.pNext = &asWrite;
                wAS.dstSet = descriptorSets[i];
                wAS.dstBinding = 0;
                wAS.descriptorCount = 1;
                wAS.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                writes.push_back(wAS);

                VkWriteDescriptorSet wGeom{};
                wGeom.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wGeom.dstSet = descriptorSets[i];
                wGeom.dstBinding = 3;
                wGeom.descriptorCount = 1;
                wGeom.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wGeom.pBufferInfo = &geomInfo;
                writes.push_back(wGeom);

                VkWriteDescriptorSet wMat{};
                wMat.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wMat.dstSet = descriptorSets[i];
                wMat.dstBinding = 4;
                wMat.descriptorCount = 1;
                wMat.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wMat.pBufferInfo = &matInfos[f];
                writes.push_back(wMat);

                VkWriteDescriptorSet wMatTex{};
                wMatTex.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wMatTex.dstSet = descriptorSets[i];
                wMatTex.dstBinding = 8;
                wMatTex.dstArrayElement = 0;
                wMatTex.descriptorCount = kMaxMaterialTextures;
                wMatTex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wMatTex.pImageInfo = matTexInfos.data();
                writes.push_back(wMatTex);
            }

            // Binding 10 (motion matrix SSBO) may have grown during this
            // rebuild — rewrite it across all sets so descriptor 10 references
            // the (possibly new) buffer handle for each frame-in-flight.
            // Storage outside the loop so the pBufferInfo pointers stay live
            // through vkUpdateDescriptorSets.
            std::array<VkDescriptorBufferInfo, kFramesInFlight> motionInfos{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                motionInfos[f].buffer = motionMatBuffers[f].handle;
                motionInfos[f].offset = 0;
                motionInfos[f].range  = VK_WHOLE_SIZE;
            }
            // Binding 14 (emissive-tri SSBO) — same per-frame story as
            // motion. Even though buildAndUploadEmissiveTris already triggers
            // its own per-frame descriptor rewrite when it grows, refreshing
            // here is harmless and keeps the rebuild path complete.
            std::array<VkDescriptorBufferInfo, kFramesInFlight> emTriInfos{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                emTriInfos[f].buffer = emissiveTriBuffers[f].handle;
                emTriInfos[f].offset = 0;
                emTriInfos[f].range  = VK_WHOLE_SIZE;
            }
            // Binding 21 (mesh-moved bits SSBO) — handle changed when
            // ensureMeshMovedBitsCapacity grew the buffer.
            std::array<VkDescriptorBufferInfo, kFramesInFlight> meshMovedInfos{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                meshMovedInfos[f].buffer = meshMovedBitsBuffers[f].handle;
                meshMovedInfos[f].offset = 0;
                meshMovedInfos[f].range  = VK_WHOLE_SIZE;
            }
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                for (uint32_t k = 0; k < imageCount_; ++k) {
                    const uint32_t setIdx = f * imageCount_ + k;
                    VkWriteDescriptorSet wMotion{};
                    wMotion.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMotion.dstSet = descriptorSets[setIdx];
                    wMotion.dstBinding = 10;
                    wMotion.descriptorCount = 1;
                    wMotion.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wMotion.pBufferInfo = &motionInfos[f];
                    writes.push_back(wMotion);

                    VkWriteDescriptorSet wEmTri{};
                    wEmTri.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wEmTri.dstSet = descriptorSets[setIdx];
                    wEmTri.dstBinding = 14;
                    wEmTri.descriptorCount = 1;
                    wEmTri.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wEmTri.pBufferInfo = &emTriInfos[f];
                    writes.push_back(wEmTri);

                    VkWriteDescriptorSet wMeshMoved{};
                    wMeshMoved.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wMeshMoved.dstSet = descriptorSets[setIdx];
                    wMeshMoved.dstBinding = 21;
                    wMeshMoved.descriptorCount = 1;
                    wMeshMoved.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wMeshMoved.pBufferInfo = &meshMovedInfos[f];
                    writes.push_back(wMeshMoved);
                }
            }

            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);

            // RasterFirst: the deferred descriptor set holds its OWN copies of
            // the scene-dependent handles — mats (binding 9), geom descs (10),
            // the bindless material-texture array (11), the emissive-tri buffer
            // (12) and the TLAS (8). A rebuild above freed and recreated those
            // buffers, so the deferred set now dangles: its buffer-device-
            // address dereferences (fetchHit on geoms[], emissiveNEE on
            // emissiveTris[]) would read freed memory → GPU page fault → device
            // lost → crash at the next refitTlas one-shot wait (and a black
            // frame in between). rewriteSceneDescriptors only refreshes the RT
            // set, so re-point the deferred set here too. We're already past
            // vkDeviceWaitIdle (see ensureSceneBuilt's rebuild path), so the
            // sets are not in flight. No-op when ray_query is unavailable.
            rewriteDeferredDescriptors();
        }

        // Rewrite only binding 6 across every descriptor set (called
        // when refreshEnvTextureFromScene replaces the env image). Avoids
        // re-allocating the pool / sets.
        void rewriteEnvDescriptors() {
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            // Three writes per set: env image (binding 6) + env CDF (18) + env marg (19).
            std::vector<VkDescriptorImageInfo> envInfos(totalSets);
            std::vector<VkDescriptorImageInfo> cdfInfos(totalSets);
            std::vector<VkDescriptorImageInfo> margInfos(totalSets);
            std::vector<VkWriteDescriptorSet>  writes(totalSets * 3);
            for (uint32_t i = 0; i < totalSets; ++i) {
                envInfos[i].sampler     = envImage.sampler;
                envInfos[i].imageView   = envImage.view;
                envInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                cdfInfos[i].sampler     = envCdfImage.sampler;
                cdfInfos[i].imageView   = envCdfImage.view;
                cdfInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                margInfos[i].sampler     = envMargImage.sampler;
                margInfos[i].imageView   = envMargImage.view;
                margInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                auto& wEnv  = writes[i * 3 + 0];
                auto& wCdf  = writes[i * 3 + 1];
                auto& wMarg = writes[i * 3 + 2];
                wEnv.sType = wCdf.sType = wMarg.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wEnv.dstSet = wCdf.dstSet = wMarg.dstSet = descriptorSets[i];
                wEnv.dstBinding  = 6;  wEnv.descriptorCount  = 1; wEnv.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wEnv.pImageInfo  = &envInfos[i];
                wCdf.dstBinding  = 18; wCdf.descriptorCount  = 1; wCdf.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wCdf.pImageInfo  = &cdfInfos[i];
                wMarg.dstBinding = 19; wMarg.descriptorCount = 1; wMarg.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wMarg.pImageInfo = &margInfos[i];
            }
            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
            // The raster-first deferred pass also samples the env (its own
            // binding 2); refresh it with the new env image / sampler.
            rewriteDeferredDescriptors();
        }

        // Rebuild every resource sized to the PT render extent — the accum /
        // gbuf / reservoir / GI ping-pongs, the denoiser + TAA images, the
        // raster G-buffer, the upscale intermediates — plus the descriptor
        // sets that bind them. Shared by swapchain recreation and runtime
        // renderScale changes. createAccumImage resets accumulation; the
        // caller must have the GPU idle before this runs.
        void reallocateRenderExtentResources() {
            for (auto& img : accumImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : gbufImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : reservoirPosImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : reservoirWImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : giResXsImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : giResNsImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : giResLoImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            // Filtered + moments destroyed implicitly by denoiser_->createImages
            // inside createAccumImage below.
            createAccumImage();// resets sampleIndex + clears prevWorldMats
            // Resize hybrid raster attachments BEFORE descriptor rewrites —
            // bindings 22-26 point at rasterGbufs[f].*.view, so stale views
            // from the old extent need to be replaced before the new
            // descriptor sets capture them.
            ensureHybridResources();
            // TAA + upscale intermediates live at the render extent too;
            // rebuild before any descriptor write captures their views (RT
            // binding 1, all TAA descriptor sets).
            {
                // TAA input is the path-trace render extent; history +
                // output are the swapchain extent. When they differ the
                // resolve pass runs as a temporal upsampler.
                const VkExtent2D inExt  = renderExtent();
                const VkExtent2D outExt = ctx->swapchainExtent();
                taa_->createImages(inExt.width, inExt.height,
                                   outExt.width, outExt.height);
            }
            bloom_->createImages(renderExtent().width, renderExtent().height);
            vkDestroyDescriptorPool(ctx->device(), descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
            descriptorSets.clear();
            createDescriptorPool();
            // Only repopulate descriptors if scene resources exist. Pre-first-
            // render callers (e.g. setRenderScale during init) hit a fresh pool
            // with no TLAS/buffers; the first ensureSceneBuilt would then
            // double-allocate into a full pool and fail with OUT_OF_POOL_MEMORY.
            if (sceneBuilt_) {
                allocateAndUpdateDescriptors();
            }
            // TAA descriptor sets are persistent (pool lives inside TaaResolve);
            // just rewrite them to the new image / view handles.
            rewriteTaaDescriptors();
            rewriteBloomDescriptors();// gbuf + TAA-input views changed
            rewriteDeferredDescriptors();// raster gbuf + sceneHdr views changed
            // The descriptor pool was rebuilt — force the per-frame binding-1
            // target rewrite to re-run regardless of its prior cached mode.
            binding1Mode_.fill(-1);
        }

        void recreateSwapchainAndDescriptors() {
            ctx->recreateSwapchain();
            reallocateRenderExtentResources();
            size = WindowSize{static_cast<int>(ctx->swapchainExtent().width),
                              static_cast<int>(ctx->swapchainExtent().height)};
        }

        // Runtime render-scale change. Reallocates every render-extent
        // resource and resets accumulation. Issues a vkDeviceWaitIdle.
        // Safe to call from inside the user's animate lambda — when a
        // frame is already mid-record (frameState_ != Idle) the
        // reallocation is deferred to the next beginFrameForPT, after
        // that frame's fence has signalled. Otherwise (Idle) applies
        // immediately. No-op when the clamped value already matches.
        void setRenderScale(float scale) {
            const float clamped = scale < 0.25f ? 0.25f
                                : (scale > 1.0f ? 1.0f : scale);
            if (clamped == renderScale_) return;
            renderScale_ = clamped;
            if (frameState_ != FrameState::Idle) {
                pendingRenderScaleRealloc_ = true;
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            reallocateRenderExtentResources();
        }

        // Records the PT pipeline body into an already-open command buffer.
        // beginFrame() is responsible for vkBeginCommandBuffer + timing-pool
        // reset *before* this call; endFrame() handles the overlay/ImGui
        // pass, the swapchain → PRESENT_SRC transition, vkEndCommandBuffer,
        // and submission *after*. On exit, the swapchain image is in
        // VK_IMAGE_LAYOUT_GENERAL (per the post-TAA / upscale path) so the
        // endFrame transitions can rely on a known layout.
        void recordCommandBuffer(VkCommandBuffer cb, uint32_t imageIndex) {

            // ── Skinned-mesh GPU pipeline ──────────────────────────────────
            // ensureSceneBuilt populated pendingSkinnedRebuilds_ with the
            // states whose bones changed this frame and uploaded the new
            // bone matrices to each state's host-visible boneMatrices buffer.
            // Now record: one skinning dispatch per state → barrier → one
            // BLAS rebuild per state → barrier. The BLAS rebuild reads the
            // deformed vertex/normal buffers the dispatch just wrote. The
            // raygen/raster downstream reads the BLAS via TLAS.
            if (!pendingSkinnedRebuilds_.empty() && skinning_) {
                // ── Step 1: snapshot current vertex → prevVertex ──────────
                // Before the skinning compute overwrites vertex with frame
                // N's deformed positions, copy what's there (frame N-1's
                // positions) into prevVertex. The chit's per-vertex motion-
                // vector interpolation in step 1 reads prevVertex via
                // gdesc.prevVertexAddress for the reprojection.
                for (auto* st : pendingSkinnedRebuilds_) {
                    if (st->blas->prevVertex.handle == VK_NULL_HANDLE) continue;
                    VkBufferCopy region{};
                    region.size = VkDeviceSize(st->vertexCount) * 3u * sizeof(float);
                    vkCmdCopyBuffer(cb, st->blas->vertex.handle,
                                    st->blas->prevVertex.handle, 1, &region);
                }
                // Transfer write → compute storage write (for vertex, which
                // the skinning dispatch will now overwrite) and transfer
                // write → ray-tracing shader read (for prevVertex, which
                // chit reads via gdesc.prevVertexAddress later this frame).
                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
                    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo dep{};
                    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                skinning_->bindPipeline(cb);
                for (auto* st : pendingSkinnedRebuilds_) {
                    skinning_->recordDispatch(cb, st->skinDescSet, st->vertexCount);
                }

                // Compute write → AS build read + vertex attribute read +
                // shader storage read. Single global memory barrier covers
                // every pending mesh.
                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                       VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                       VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    VkDependencyInfo dep{};
                    dep.sType               = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount  = 1;
                    dep.pMemoryBarriers     = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                // BLAS rebuild per state — refit (MODE_UPDATE) by default,
                // with periodic full rebuild every kBlasFullRebuildInterval
                // frames to keep the BVH balanced under articulated motion.
                // Persistent scratch is sized to buildScratchSize, which is
                // always >= updateScratchSize, so the same buffer serves both.
                for (auto* st : pendingSkinnedRebuilds_) {
                    const bool fullRebuild =
                            st->blasRefitCounter >=
                            SkinnedMeshState::kBlasFullRebuildInterval;
                    st->blasRefitCounter = fullRebuild ? 0u
                                                       : (st->blasRefitCounter + 1u);
                    VkAccelerationStructureGeometryTrianglesDataKHR triData{};
                    triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                    triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                    triData.vertexData.deviceAddress = st->blas->vertex.address;
                    triData.vertexStride = 3 * sizeof(float);
                    triData.maxVertex    = st->vertexCount - 1;
                    if (st->indexed) {
                        triData.indexType = VK_INDEX_TYPE_UINT32;
                        triData.indexData.deviceAddress = st->blas->index.address;
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
                    blasBuild.mode  = fullRebuild
                            ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                            : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                    blasBuild.geometryCount = 1;
                    blasBuild.pGeometries   = &blasGeom;
                    blasBuild.srcAccelerationStructure =
                            fullRebuild ? VK_NULL_HANDLE : st->blas->as;
                    blasBuild.dstAccelerationStructure = st->blas->as;
                    blasBuild.scratchData.deviceAddress = st->blasScratch.address;
                    VkAccelerationStructureBuildRangeInfoKHR range{};
                    range.primitiveCount = st->primitiveCount;
                    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
                    ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
                }

                // AS build write → TLAS / RT_SHADER read.
                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                    mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                    VkDependencyInfo dep{};
                    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                pendingSkinnedRebuilds_.clear();
            }

            // ── Tet-skinning (PhysX soft bodies) — same structure as the skinned
            // block above: prevVertex snapshot → barrier → tet_skinning dispatch →
            // barrier → BLAS refit → barrier. Separate pipeline + pending list.
            if (!pendingTetRebuilds_.empty() && tetSkinning_) {
                for (auto* st : pendingTetRebuilds_) {
                    if (st->blas->prevVertex.handle == VK_NULL_HANDLE) continue;
                    VkBufferCopy region{};
                    region.size = VkDeviceSize(st->vertexCount) * 3u * sizeof(float);
                    vkCmdCopyBuffer(cb, st->blas->vertex.handle,
                                    st->blas->prevVertex.handle, 1, &region);
                }
                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
                    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo dep{};
                    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                tetSkinning_->bindPipeline(cb);
                for (auto* st : pendingTetRebuilds_) {
                    tetSkinning_->recordDispatch(cb, st->tetDescSet, st->vertexCount);
                }

                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                       VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                       VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    VkDependencyInfo dep{};
                    dep.sType               = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount  = 1;
                    dep.pMemoryBarriers     = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                for (auto* st : pendingTetRebuilds_) {
                    const bool fullRebuild =
                            st->blasRefitCounter >= TetMeshState::kBlasFullRebuildInterval;
                    st->blasRefitCounter = fullRebuild ? 0u : (st->blasRefitCounter + 1u);
                    VkAccelerationStructureGeometryTrianglesDataKHR triData{};
                    triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                    triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                    triData.vertexData.deviceAddress = st->blas->vertex.address;
                    triData.vertexStride = 3 * sizeof(float);
                    triData.maxVertex    = st->vertexCount - 1;
                    if (st->indexed) {
                        triData.indexType = VK_INDEX_TYPE_UINT32;
                        triData.indexData.deviceAddress = st->blas->index.address;
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
                    blasBuild.mode  = fullRebuild
                            ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                            : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                    blasBuild.geometryCount = 1;
                    blasBuild.pGeometries   = &blasGeom;
                    blasBuild.srcAccelerationStructure =
                            fullRebuild ? VK_NULL_HANDLE : st->blas->as;
                    blasBuild.dstAccelerationStructure = st->blas->as;
                    blasBuild.scratchData.deviceAddress = st->blasScratch.address;
                    VkAccelerationStructureBuildRangeInfoKHR range{};
                    range.primitiveCount = st->primitiveCount;
                    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
                    ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
                }

                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                    mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                    VkDependencyInfo dep{};
                    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                pendingTetRebuilds_.clear();
            }

            // ── Hybrid raster G-buffer pass ─────────────────────────────────
            // Runs ahead of any RT work so the gbuffer is ready when raygen
            // wants to read primary visibility. In G-buffer debug mode we blit
            // a chosen channel directly to the swapchain, draw the ImGui
            // overlay on top (mirrors the PT path's overlay flow), then
            // present — bypassing the entire RT pipeline.
            if (rasterGbufPipeline != VK_NULL_HANDLE) {
                gpuTimings_->begin(cb, TP_RasterGbuf, currentFrame);
                recordRasterGbufPass(cb, currentFrame);
                gpuTimings_->end(cb, TP_RasterGbuf, currentFrame);
                // ── Overlay depth prepass ──────────────────────────────────
                // Fills rasterGbufs[currentFrame].unjitDepth with the
                // unjittered VP. Consumed by the post-TAA wireframe overlay
                // pass for occlusion testing. Only runs when an overlay
                // pipeline exists AND the scene actually has overlay
                // candidates this frame (else the prepass is wasted work).
                if (overlayDepthPrepassPipeline != VK_NULL_HANDLE && sceneHasOverlayContent()) {
                    gpuTimings_->begin(cb, TP_OverlayDepth, currentFrame);
                    // Swapchain extent — unjitDepth is full-res so the
                    // post-TAA overlay can depth-test the upscaled image.
                    const VkExtent2D dext = ctx->swapchainExtent();
                    VkImage      depthImg  = rasterGbufs[currentFrame].unjitDepth.image;
                    VkImageView  depthView = rasterGbufs[currentFrame].unjitDepth.view;

                    // UNDEFINED → DEPTH_ATTACHMENT (write). We always clear
                    // each frame so the prior contents are irrelevant; the
                    // initial layout is ignored under DONT_CARE-style clear.
                    VkImageMemoryBarrier2 toDepth{};
                    toDepth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    toDepth.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                    toDepth.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                             VK_ACCESS_2_SHADER_READ_BIT;
                    toDepth.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                    toDepth.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                    toDepth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    toDepth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                    toDepth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toDepth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toDepth.image = depthImg;
                    toDepth.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                    toDepth.subresourceRange.levelCount = 1;
                    toDepth.subresourceRange.layerCount = 1;
                    VkDependencyInfo depToDepth{};
                    depToDepth.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    depToDepth.imageMemoryBarrierCount = 1;
                    depToDepth.pImageMemoryBarriers = &toDepth;
                    vkCmdPipelineBarrier2(cb, &depToDepth);

                    VkRenderingAttachmentInfo dDepthAtt{};
                    dDepthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    dDepthAtt.imageView   = depthView;
                    dDepthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                    dDepthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    dDepthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                    dDepthAtt.clearValue.depthStencil = {0.0f, 0u};// reverse-Z: clear to 0 (far)

                    VkRenderingInfo dRi{};
                    dRi.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    dRi.renderArea.offset = {0, 0};
                    dRi.renderArea.extent = dext;
                    dRi.layerCount = 1;
                    dRi.colorAttachmentCount = 0;
                    dRi.pDepthAttachment = &dDepthAtt;
                    vkCmdBeginRendering(cb, &dRi);

                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayDepthPrepassPipeline);
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            rasterPipelineLayout, 0, 1,
                                            &rasterDescSets[currentFrame], 0, nullptr);
                    VkViewport dvp{0.f, 0.f, float(dext.width), float(dext.height), 0.f, 1.f};
                    vkCmdSetViewport(cb, 0, 1, &dvp);
                    VkRect2D dsc{{0, 0}, dext};
                    vkCmdSetScissor(cb, 0, 1, &dsc);

                    for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
                        const auto& en = lastVisibleEntries_[i];
                        if (en.isOverlay) continue;// overlay meshes drawn by overlay pass instead
                        if (!en.inFrustum) continue;// frustum cull (same lever as the gbuf prepass)
                        const BlasRecord* rec = resolveBlasForEntry(en);
                        if (!rec || rec->vertex.handle == VK_NULL_HANDLE) continue;

                        struct PC {
                            float    model[16];
                            uint32_t instanceCustomIndex;
                            uint32_t flags;
                            uint32_t _pad0;
                            uint32_t _pad1;
                        } pcDepth{};
                        std::memcpy(pcDepth.model, en.worldMatrix.data(), 64);
                        pcDepth.instanceCustomIndex = static_cast<uint32_t>(i);
                        pcDepth.flags = 0u;
                        vkCmdPushConstants(cb, rasterPipelineLayout,
                                           VK_SHADER_STAGE_VERTEX_BIT,
                                           0, sizeof(pcDepth), &pcDepth);

                        VkBuffer     vbufs[1] = {rec->vertex.handle};
                        VkDeviceSize voffs[1] = {0};
                        vkCmdBindVertexBuffers(cb, 0, 1, vbufs, voffs);
                        if (rec->index.handle != VK_NULL_HANDLE) {
                            vkCmdBindIndexBuffer(cb, rec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                            auto* idxAttr = en.mesh->geometry()->getIndex();
                            if (idxAttr) {
                                vkCmdDrawIndexed(cb, static_cast<uint32_t>(idxAttr->count()),
                                                 1, 0, 0, 0);
                            }
                        } else {
                            auto* posAttr = en.mesh->geometry()->getAttribute<float>("position");
                            if (posAttr) {
                                vkCmdDraw(cb, static_cast<uint32_t>(posAttr->count()), 1, 0, 0);
                            }
                        }
                    }
                    vkCmdEndRendering(cb);

                    // Transition to DEPTH_STENCIL_READ_ONLY_OPTIMAL for the
                    // overlay pass's read-only depth attachment.
                    VkImageMemoryBarrier2 toRead{};
                    toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    toRead.srcStageMask  = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                    toRead.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                    toRead.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                    toRead.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                    toRead.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                    toRead.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toRead.image = depthImg;
                    toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                    toRead.subresourceRange.levelCount = 1;
                    toRead.subresourceRange.layerCount = 1;
                    VkDependencyInfo depToRead{};
                    depToRead.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    depToRead.imageMemoryBarrierCount = 1;
                    depToRead.pImageMemoryBarriers = &toRead;
                    vkCmdPipelineBarrier2(cb, &depToRead);
                    gpuTimings_->end(cb, TP_OverlayDepth, currentFrame);
                }
                if (hybridDebugView_ != HybridDebugView::Off) {
                    recordHybridDebugBlit(cb, imageIndex, currentFrame);
                    // Blit left swapchain in TRANSFER_DST_OPTIMAL. Same
                    // overlay + present sequence the PT path uses, just
                    // with a different srcLayout entering the transition.
                    const VkImage swap = ctx->swapchainImages()[imageIndex];
                    const VkExtent2D ext = ctx->swapchainExtent();
                    if (overlayCallback) {
                        VkImageMemoryBarrier2 toColor{};
                        toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        toColor.srcStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
                        toColor.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        toColor.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        toColor.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        toColor.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        toColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        toColor.image = swap;
                        toColor.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        toColor.subresourceRange.levelCount = 1;
                        toColor.subresourceRange.layerCount = 1;
                        VkDependencyInfo depColor{};
                        depColor.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        depColor.imageMemoryBarrierCount = 1;
                        depColor.pImageMemoryBarriers = &toColor;
                        vkCmdPipelineBarrier2(cb, &depColor);

                        VkRenderingAttachmentInfo colorAtt{};
                        colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                        colorAtt.imageView = ctx->swapchainImageViews()[imageIndex];
                        colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                        VkRenderingInfo ri{};
                        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                        ri.renderArea.offset = {0, 0};
                        ri.renderArea.extent = ext;
                        ri.layerCount = 1;
                        ri.colorAttachmentCount = 1;
                        ri.pColorAttachments = &colorAtt;
                        vkCmdBeginRendering(cb, &ri);
                        overlayCallback(static_cast<void*>(cb));
                        vkCmdEndRendering(cb);

                        VkImageMemoryBarrier2 toPresent{};
                        toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        toPresent.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        toPresent.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                        toPresent.dstAccessMask = 0;
                        toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                        toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        toPresent.image = swap;
                        toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        toPresent.subresourceRange.levelCount = 1;
                        toPresent.subresourceRange.layerCount = 1;
                        VkDependencyInfo depPresent{};
                        depPresent.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        depPresent.imageMemoryBarrierCount = 1;
                        depPresent.pImageMemoryBarriers = &toPresent;
                        vkCmdPipelineBarrier2(cb, &depPresent);
                    } else {
                        VkImageMemoryBarrier2 toPresent{};
                        toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        toPresent.srcStageMask  = VK_PIPELINE_STAGE_2_BLIT_BIT;
                        toPresent.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        toPresent.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                        toPresent.dstAccessMask = 0;
                        toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                        toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        toPresent.image = swap;
                        toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        toPresent.subresourceRange.levelCount = 1;
                        toPresent.subresourceRange.layerCount = 1;
                        VkDependencyInfo depPresent{};
                        depPresent.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        depPresent.imageMemoryBarrierCount = 1;
                        depPresent.pImageMemoryBarriers = &toPresent;
                        vkCmdPipelineBarrier2(cb, &depPresent);
                    }
                    check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
                    gpuTimings_->finishRecord();
                    return;
                }
            }

            // ── Events-only render mode early-return ───────────────────────
            // High-rate event-camera path (~500 Hz target): the gbuf prepass
            // above wrote everything event_shade needs to produce a clean
            // deterministic luma image. Skip PT/denoise/TAA/overlay/upscale
            // entirely. The swapchain is cleared to black so the sprite
            // overlay (event accumulator) has a known starting canvas;
            // event_shade + event_detect run after recordCommandBuffer
            // exactly as in the normal PT path.
            //
            // Requires the gbuf prepass to have actually run — gated above
            // on rasterGbufPipeline. If it's missing the events-only flag is
            // ignored and we fall through to the full PT path.
            if (eventsOnlyMode_ && eventCamEnabled_ &&
                rasterGbufPipeline != VK_NULL_HANDLE) {
                const VkImage swap = ctx->swapchainImages()[imageIndex];

                // UNDEFINED → GENERAL so vkCmdClearColorImage can write it,
                // and so the downstream sprite overlay + ImGui pass see the
                // layout they expect (matching the normal recordCommandBuffer
                // tail comment at line 11892).
                VkImageMemoryBarrier2 toGen{};
                toGen.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                toGen.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                toGen.srcAccessMask = 0;
                toGen.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                toGen.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toGen.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                toGen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                toGen.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGen.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGen.image = swap;
                toGen.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toGen.subresourceRange.levelCount = 1;
                toGen.subresourceRange.layerCount = 1;
                VkDependencyInfo dGen{};
                dGen.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dGen.imageMemoryBarrierCount = 1;
                dGen.pImageMemoryBarriers = &toGen;
                vkCmdPipelineBarrier2(cb, &dGen);

                VkClearColorValue cc{};
                cc.float32[0] = 0.f;
                cc.float32[1] = 0.f;
                cc.float32[2] = 0.f;
                cc.float32[3] = 1.f;
                VkImageSubresourceRange clrRange{};
                clrRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                clrRange.levelCount = 1;
                clrRange.layerCount = 1;
                vkCmdClearColorImage(cb, swap, VK_IMAGE_LAYOUT_GENERAL,
                                     &cc, 1, &clrRange);

                // Transfer-write → downstream consumers (sprite overlay
                // composites in COLOR_ATTACHMENT, ImGui in same, the
                // potential scene-capture in TRANSFER_SRC). Layout stays
                // GENERAL — caller pipelines transition out of GENERAL
                // as needed (recordOverlayAndPresentTransition handles
                // the GENERAL → PRESENT_SRC at the end).
                VkImageMemoryBarrier2 visBar{};
                visBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                visBar.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                visBar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                visBar.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                visBar.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                       VK_ACCESS_2_TRANSFER_READ_BIT;
                visBar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                visBar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                visBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                visBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                visBar.image = swap;
                visBar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                visBar.subresourceRange.levelCount = 1;
                visBar.subresourceRange.layerCount = 1;
                VkDependencyInfo dVis{};
                dVis.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dVis.imageMemoryBarrierCount = 1;
                dVis.pImageMemoryBarriers = &visBar;
                vkCmdPipelineBarrier2(cb, &dVis);

                return;
            }
            // ── End events-only render mode ─────────────────────────────────

            const VkImage img = ctx->swapchainImages()[imageIndex];

            // UNDEFINED -> GENERAL. Swapchain is now written by either the
            // TAA compute dispatch (post-denoise), the denoise compute pass
            // directly (TAA off + unscaled), or the upscale blit (TAA off +
            // scaled); raygen no longer writes the swapchain directly
            // (binding 1 was redirected away from the swapchain view).
            VkImageMemoryBarrier2 toGeneral{};
            toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            toGeneral.srcAccessMask = 0;
            toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                      VK_ACCESS_2_TRANSFER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = img;
            toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toGeneral.subresourceRange.levelCount = 1;
            toGeneral.subresourceRange.layerCount = 1;

            // Ensure prior frames' accumulator + gbuf writes are visible to
            // this frame's read-modify-write. We barrier both ping-pong slots
            // since over consecutive frames the read-from / write-to roles
            // alternate. Layout stays in GENERAL for all four storage images.
            VkImageMemoryBarrier2 accumGbufTemplate{};
            accumGbufTemplate.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            accumGbufTemplate.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            accumGbufTemplate.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            accumGbufTemplate.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            accumGbufTemplate.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            accumGbufTemplate.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            accumGbufTemplate.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            accumGbufTemplate.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumGbufTemplate.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumGbufTemplate.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            accumGbufTemplate.subresourceRange.levelCount = 1;
            accumGbufTemplate.subresourceRange.layerCount = 1;

            std::array<VkImageMemoryBarrier2, 17> preBarriers{};
            preBarriers[0] = toGeneral;
            preBarriers[1] = accumGbufTemplate; preBarriers[1].image = accumImagesPP[0].image;
            preBarriers[2] = accumGbufTemplate; preBarriers[2].image = accumImagesPP[1].image;
            preBarriers[3] = accumGbufTemplate; preBarriers[3].image = gbufImagesPP[0].image;
            preBarriers[4] = accumGbufTemplate; preBarriers[4].image = gbufImagesPP[1].image;
            // ReSTIR DI reservoir ping-pong: frame N writes one slot, reads the
            // other; the barrier ensures frame N-1's write is visible to frame
            // N's read in the same RT_SHADER stage.
            preBarriers[5] = accumGbufTemplate; preBarriers[5].image = reservoirPosImagesPP[0].image;
            preBarriers[6] = accumGbufTemplate; preBarriers[6].image = reservoirPosImagesPP[1].image;
            preBarriers[7] = accumGbufTemplate; preBarriers[7].image = reservoirWImagesPP[0].image;
            preBarriers[8] = accumGbufTemplate; preBarriers[8].image = reservoirWImagesPP[1].image;
            // ReSTIR GI reservoir ping-pong (3 image-pairs: xs, ns, Lo).
            preBarriers[9]  = accumGbufTemplate; preBarriers[9].image  = giResXsImagesPP[0].image;
            preBarriers[10] = accumGbufTemplate; preBarriers[10].image = giResXsImagesPP[1].image;
            preBarriers[11] = accumGbufTemplate; preBarriers[11].image = giResNsImagesPP[0].image;
            preBarriers[12] = accumGbufTemplate; preBarriers[12].image = giResNsImagesPP[1].image;
            preBarriers[13] = accumGbufTemplate; preBarriers[13].image = giResLoImagesPP[0].image;
            preBarriers[14] = accumGbufTemplate; preBarriers[14].image = giResLoImagesPP[1].image;
            // Temporal moments ping-pong: same RT_SHADER → RT_SHADER fence as
            // accum/gbuf. Atrous reads the just-written slot via the existing
            // memory barrier (barrierMem RT→COMPUTE in the denoise block).
            preBarriers[15] = accumGbufTemplate; preBarriers[15].image = denoiser_->momentsImage(0);
            preBarriers[16] = accumGbufTemplate; preBarriers[16].image = denoiser_->momentsImage(1);
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size());
            dep.pImageMemoryBarriers = preBarriers.data();
            vkCmdPipelineBarrier2(cb, &dep);

            // descriptorSets index is shared by photon and primary RT pipelines.
            const uint32_t setIdx = currentFrame * imageCount_ + imageIndex;

            // Extents + exposure are shared by both render modes and by the
            // bloom/TAA tail below, so hoist them out of the mode branch.
            const VkExtent2D ext   = ctx->swapchainExtent();
            // Path-trace / deferred render extent — equals `ext` unless
            // renderScale_ < 1, in which case TAA upsamples to the swapchain.
            const VkExtent2D ptExt = renderExtent();
            const float exposure   = toneMappingExposure_;
            uint32_t exposureBits;
            std::memcpy(&exposureBits, &exposure, sizeof(exposureBits));

            // RenderMode::ReferencePT — the path tracer shades everything
            // (photon caustics + raygen + à-trous denoise). RenderMode::
            // RasterFirst replaces this entire block with one analytic
            // deferred-shade compute pass; bloom + TAA below are shared.
            // RasterFirst also falls back here when ray query is unsupported
            // (deferredShade_ == null), so it never renders a black frame.
            if (renderMode_ == VulkanRenderer::RenderMode::ReferencePT || !deferredShade_) {

            // ── Photon emit pass ────────────────────────────────────────────────
            // Two gates: (1) sceneHasGlass_ — no transmissive material means
            // there's no chance of caustics anywhere, and the chit gather is
            // gated on the same flag. (2) glassVisibleThisFrame_ — even when
            // glass exists, skip the 512×512 emit pass on frames where no
            // glass AABB is in the camera frustum. The world-grid photon
            // store keeps the last emit visible to gather for free, so a few
            // frames of "stale" caustics persist after the camera pans away;
            // they're refreshed the moment glass re-enters view.
            //
            // First-frame safety: chit's gather is gated on sceneHasGlass_
            // alone (not visibility), so if a scene starts with glass but
            // glass is not in the initial frustum we'd let gather read
            // device-local photonCountBuf before anyone has written it.
            // Shim a one-shot zero-fill so the gather always sees a valid
            // (all-zero) count buffer.
            if (sceneHasGlass_ && !glassVisibleThisFrame_ && !photon_->isInitialized()) {
                photon_->recordZeroFillCounts(cb);
            }
            if (sceneHasGlass_ && glassVisibleThisFrame_) {
                gpuTimings_->begin(cb, TP_PhotonEmit, currentFrame);
                // exposureBits hoisted above the mode branch.
                const uint32_t motionFlagsPhoton =
                        (motionThisFrame_      ? 1u : 0u) |
                        (cameraMovedThisFrame_ ? 2u : 0u) |
                        (sceneHasGlass_        ? 4u : 0u);
                uint32_t emPowerBits;
                std::memcpy(&emPowerBits, &emissiveTotalPowerThisFrame_, sizeof(emPowerBits));
                uint32_t envSumBitsPhoton;
                std::memcpy(&envSumBitsPhoton, &envCdfTotalSum_, sizeof(envSumBitsPhoton));
                uint32_t fireflyBitsPhoton;
                std::memcpy(&fireflyBitsPhoton, &fireflyClamp_, sizeof(fireflyBitsPhoton));
                uint32_t oceanFineBitsPhoton;
                std::memcpy(&oceanFineBitsPhoton, &oceanFineTileSize, sizeof(oceanFineBitsPhoton));
                vulkan::PhotonCaustics::EmitPushConstants push{};
                push.v[0]  = sampleIndex;
                push.v[1]  = envImage.mipLevels;
                push.v[2]  = static_cast<uint32_t>(toneMapping_);
                push.v[3]  = exposureBits;
                push.v[4]  = motionFlagsPhoton;
                push.v[5]  = emissiveTriCountThisFrame_;
                push.v[6]  = emPowerBits;
                push.v[7]  = samplesPerPixel_;
                push.v[8]  = envCdfWidth_;
                push.v[9]  = envCdfHeight_;
                push.v[10] = envSumBitsPhoton;
                push.v[11] = fireflyBitsPhoton;
                push.v[12] = oceanFineBitsPhoton;
                photon_->recordEmitPass(cb, descriptorSets[setIdx], push);
                gpuTimings_->end(cb, TP_PhotonEmit, currentFrame);
            }
            // ── End photon emit ─────────────────────────────────────────────────

            const RtPipelineVariant& rtv = activeRtVariant();
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtv.pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    rtPipelineLayout, 0, 1,
                                    &descriptorSets[setIdx], 0, nullptr);

            // [0] sampleIndex (raygen), [1] PMREM mip count (closest_hit),
            // [2] ToneMapping enum, [3] exposure as float bits,
            // [4] motionFlags (bit 0 = any motion, bit 1 = camera moved,
            //                  bit 2 = scene has any glass material,
            //                  bit 3 = hybrid mode (gbuffer drives reproject
            //                          + primary jitter is disabled),
            //                  bit 4 = ReSTIR DI enabled (RIS at primary;
            //                          chit falls back to classic NEE when off),
            //                  bit 5 = perSppJitter (hybrid sub-pixel jitter
            //                          is taken per-sample instead of once),
            //                  bit 6 = ReSTIR GI enabled (Stage 1a single-
            //                          sample indirect reservoir at primary;
            //                          off = classic bounce-1 continuation),
            //                  bit 7 = measure primary trace only — raygen
            //                          exits the step + spp loops immediately
            //                          after the step-0 traceRayEXT; debug
            //                          knob for sizing the bounce-0-on-raster
            //                          opportunity, image goes black),
            //                  bit 8 = ReSTIR DI visibility reuse (shadow-test
            //                          the RIS pick before reuse, discard if
            //                          occluded; only set when bit 4 is too),
            // [5] emissiveCount, [6] emissiveTotalPower (float bits).
            // Per-instance moved bits live in the binding 21 SSBO.
            // (exposure / exposureBits hoisted above the mode branch.)
            const uint32_t motionFlags =
                    (motionThisFrame_           ? 1u   : 0u) |
                    (cameraMovedThisFrame_      ? 2u   : 0u) |
                    (sceneHasGlass_             ? 4u   : 0u) |
                    (restirDIEnabled_           ? 16u  : 0u) |
                    (perSppJitterHybrid_        ? 32u  : 0u) |
                    (restirGIEnabled_           ? 64u  : 0u) |
                    (measurePrimaryTraceOnly_   ? 128u : 0u) |
                    ((restirDIEnabled_ && restirDIVisibilityReuse_) ? 256u : 0u);
            uint32_t emPowerBits;
            std::memcpy(&emPowerBits, &emissiveTotalPowerThisFrame_, sizeof(emPowerBits));
            uint32_t envSumBits;
            std::memcpy(&envSumBits, &envCdfTotalSum_, sizeof(envSumBits));
            uint32_t fireflyBits;
            std::memcpy(&fireflyBits, &fireflyClamp_, sizeof(fireflyBits));
            uint32_t oceanFineBits;
            std::memcpy(&oceanFineBits, &oceanFineTileSize, sizeof(oceanFineBits));
            uint32_t oceanFoamBits;
            std::memcpy(&oceanFoamBits, &oceanFoamTileSize, sizeof(oceanFoamBits));
            const uint32_t pc[16] = {
                    sampleIndex,
                    envImage.mipLevels,
                    static_cast<uint32_t>(toneMapping_),
                    exposureBits,
                    motionFlags,
                    emissiveTriCountThisFrame_,
                    emPowerBits,
                    samplesPerPixel_,
                    envCdfWidth_,
                    envCdfHeight_,
                    envSumBits,
                    fireflyBits,
                    oceanFineBits,
                    edgeMsaaExtra_,
                    oceanFoamBits,
                    maxBounces_,
            };
            vkCmdPushConstants(cb, rtPipelineLayout,
                               VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                       VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                       VK_SHADER_STAGE_MISS_BIT_KHR,
                               0, sizeof(pc), pc);

            // (ext / ptExt hoisted above the mode branch.)
            gpuTimings_->begin(cb, TP_PathTrace, currentFrame);
            ctx->rt().cmdTraceRays(cb, &rtv.rgenRegion, &rtv.missRegion, &rtv.hitRegion,
                                   &callRegion, ptExt.width, ptExt.height, 1);
            gpuTimings_->end(cb, TP_PathTrace, currentFrame);

            // ── Spatial denoiser: 2-pass à-trous + finalize tonemap + sRGB ──────
            // RT writes accumImage + gbufImage; denoise pipeline reads them.
            // outImage is owned by the finalize pass (raygen.rgen tail was
            // stripped). All ping-pong slots stay GENERAL throughout. The
            // RT_SHADER → COMPUTE_SHADER barrier is recorded inside
            // Denoiser::recordDispatch.
            gpuTimings_->begin(cb, TP_Denoise, currentFrame);
            denoiser_->recordDispatch(cb, descriptorSets[setIdx], ptExt,
                                      denoiseEnabled_,
                                      static_cast<uint32_t>(toneMapping_),
                                      exposureBits,
                                      envIsBgColor);
            gpuTimings_->end(cb, TP_Denoise, currentFrame);
            // ── End denoise ─────────────────────────────────────────────────────

            } else {
                // ── RenderMode::RasterFirst — analytic deferred base ───────────
                // Shade the raster material G-buffer (direct analytic lights +
                // split-sum specular IBL + approximate diffuse IBL) straight
                // into bloom_->sceneHdr. No path tracing, no denoise — the base
                // is noise-free. The raster G-buffer pass already ran and its
                // render-pass dependency makes it visible to COMPUTE; bloom's
                // leading barrier makes this write visible to the composite.
                // Per-frame BLAS refits (skinned / deformable meshes) are fenced
                // only to the RT pipeline stage by the build barriers above. The
                // deferred pass traverses the same acceleration structures via
                // ray query from COMPUTE, so add an AS-build → compute fence here.
                // No-op for static scenes (no pending AS write this frame).
                // ALSO carries the GI-reproject cross-frame dependency: this
                // frame's deferred shade SAMPLES the OTHER frame-in-flight's
                // indirect image (last frame's accumulated GI history). Make the
                // prev frame's COMPUTE write to it visible to this frame's COMPUTE
                // read (the GPU executes frames sequentially per queue, so this is a
                // cache-visibility barrier, not ordering).
                {
                    VkMemoryBarrier2 asbar{};
                    asbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    asbar.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    asbar.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                                          VK_ACCESS_2_SHADER_WRITE_BIT;
                    asbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    asbar.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                          VK_ACCESS_2_SHADER_READ_BIT;
                    VkDependencyInfo asdep{};
                    asdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    asdep.memoryBarrierCount = 1;
                    asdep.pMemoryBarriers = &asbar;
                    vkCmdPipelineBarrier2(cb, &asdep);
                }
                gpuTimings_->begin(cb, TP_PathTrace, currentFrame);
                deferredShade_->recordDispatch(cb, currentFrame,
                                               ptExt.width, ptExt.height,
                                               envImage.mipLevels, /*shadows=*/true,
                                               /*ao=*/deferredAO_, sampleIndex,
                                               emissiveTriCountThisFrame_,
                                               emissiveTotalPowerThisFrame_,
                                               fireflyClamp_,
                                               oceanFineTileSize, oceanFoamTileSize,
                                               denoiseEnabled_, restirDIEnabled_,
                                               deferredVolDensity_, deferredVolAniso_,
                                               deferredStarIntensity_,
                                               deferredCamDeltaLen_, deferredCamRotAngle_,
                                               static_cast<float>(glfwGetTime()));
                gpuTimings_->end(cb, TP_PathTrace, currentFrame);// pathTraceMs = deferred SHADE only
                // Spatial denoise of the demodulated diffuse-indirect + recombine.
                // Barrier: the shade wrote sceneHdr + the indirect image (both
                // GENERAL storage); the denoise reads the indirect 5×5 neighbourhood
                // and read-modify-writes sceneHdr — compute→compute RAW/WAR.
                if (denoiseEnabled_) {
                    VkMemoryBarrier2 denoiseBar{};
                    denoiseBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    denoiseBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    denoiseBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    denoiseBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    denoiseBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo denoiseDep{};
                    denoiseDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    denoiseDep.memoryBarrierCount = 1;
                    denoiseDep.pMemoryBarriers = &denoiseBar;
                    vkCmdPipelineBarrier2(cb, &denoiseDep);
                    gpuTimings_->begin(cb, TP_Denoise, currentFrame);// denoiseMs = deferred SVGF (4 GI passes + reflection pass)
                    deferredShade_->recordDenoiseDispatch(cb, currentFrame, ptExt.width, ptExt.height);
                    gpuTimings_->end(cb, TP_Denoise, currentFrame);
                }
            }

            // ── Bloom + tone-map/sRGB composite (HDR post stack) ───────────────
            // denoise.comp (resolve) wrote linear HDR into bloom_->sceneHdr
            // (the shared set's binding 1). This pass glows the bright
            // highlights and composites bloom + tone map + sRGB into the TAA
            // input image. With bloomIntensity_ == 0 the bloom passes are
            // skipped and the composite reproduces the old finalize output
            // (tone map + sRGB + solid-bg bypass) exactly.
            bloom_->recordDispatch(cb, currentFrame, ptExt.width, ptExt.height,
                                   static_cast<uint32_t>(toneMapping_),
                                   exposureBits, envIsBgColor,
                                   bloomIntensity_, bloomThreshold_, bloomClamp_);
            // ── End bloom ──────────────────────────────────────────────────────

            // ── Raster TAA / temporal upsampler ─────────────────────────────────
            // Reads denoise output from the TAA input image (render extent),
            // blends with reprojected history (rgba16f, swapchain extent),
            // writes the result straight to the swapchain. When renderScale
            // < 1 the input is lower-res than the output, so this pass IS the
            // upscaler — jittered low-res samples accumulate into the full-
            // res history, reconstructing detail (no separate blit needed).
            // The spatial neighborhood clamp + motion-vec reproject smooths
            // per-frame Halton jitter shake on moving objects.
            // Frame-rate-aware history blend: keep the ghost-decay time constant in
            // wall-clock seconds, not frames (see taaPrevTimeSec_). taaBlendAlpha_ is
            // authored to look good at kTaaRefFps; at or above that rate this is a
            // no-op (effAlpha == taaBlendAlpha_, so the demos that already look clean
            // are byte-unchanged), and BELOW it the new-sample weight is raised so a
            // moving object's trail clears in the same real time it would at the
            // reference rate instead of lingering for a fixed frame count. `frames` is
            // (1-alpha)'s exponent = how many reference-frames this real frame spans;
            // clamped to [1, 6] so we never reduce alpha (preserving the high-fps
            // look) and a one-off hitch (loading stall, alt-tab) can't spike it toward
            // 1 and strobe the frame to raw jittered input.
            constexpr float kTaaRefFps = 90.0f;
            float effAlpha = taaBlendAlpha_;
            float taaDtFrames = 1.0f;// also pushed to the shader, which scales its
                                     // own per-frame constants (deviation-streak
                                     // ramp, soft-clip rate) by it — without that,
                                     // low fps leaves discrete stale edge bands
                                     // (one per ramp frame) behind moving objects.
            {
                const double now = glfwGetTime();
                if (taaPrevTimeSec_ >= 0.0) {
                    const double dt = now - taaPrevTimeSec_;
                    if (dt > 0.0) {
                        taaDtFrames = std::clamp(static_cast<float>(dt) * kTaaRefFps, 1.0f, 6.0f);
                        effAlpha = 1.0f - std::pow(1.0f - taaBlendAlpha_, taaDtFrames);
                    }
                }
                taaPrevTimeSec_ = now;
            }
            gpuTimings_->begin(cb, TP_TAA, currentFrame);
            taa_->recordResolve(cb, currentFrame, imageIndex,
                                ptExt.width, ptExt.height,
                                ext.width, ext.height, effAlpha, taaDtFrames,
                                sharpenStrength_ > 0.0f, sharpenStrength_,
                                taaSkyReproj_.data());
            gpuTimings_->end(cb, TP_TAA, currentFrame);
            // ── End raster TAA ─────────────────────────────────────────────────

            // ── Hybrid raster overlay pass ─────────────────────────────────────
            // Wireframe-flagged meshes (any material with wireframe == true)
            // and overlay-layer-tagged meshes drawn on top of the post-TAA
            // image. Depth-tested against the existing raster G-buffer depth
            // so overlays are correctly occluded by path-traced surfaces. No
            // depth writes — the depth attachment stays unchanged for the
            // next frame's chit + denoise consumption. Line/LineSegments
            // objects are drawn here too, via their own topology pipelines
            // and the per-Line vertex buffer cache (ensureLineGeometryUploaded).
            //
            // Skip the whole block when not in hybrid (depth attachment isn't
            // built), pipeline failed to create, or the early scan didn't
            // find any overlay candidates this frame.
            if (overlayWireframePipeline != VK_NULL_HANDLE) {
                // Gate on the same current-frame answer the unjittered-depth
                // prepass keyed off, so the prepass that filled + transitioned
                // unjitDepth to DEPTH_STENCIL_READ_ONLY_OPTIMAL ran iff we draw
                // here and read it (see sceneHasOverlayContent()).
                const bool hasOverlay = sceneHasOverlayContent();

                if (hasOverlay) {
                    gpuTimings_->begin(cb, TP_OverlayDraw, currentFrame);
                    // Swapchain GENERAL → COLOR_ATTACHMENT_OPTIMAL. The
                    // overlay always composites onto the full-resolution
                    // swapchain — TAA wrote it directly (upscaling there if
                    // renderScale < 1), so there is no render-extent target
                    // here even in scaled mode.
                    VkImageMemoryBarrier2 toColor{};
                    toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    toColor.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                            VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                    toColor.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                            VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    toColor.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    toColor.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    toColor.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    toColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toColor.image = img;
                    toColor.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    toColor.subresourceRange.levelCount = 1;
                    toColor.subresourceRange.layerCount = 1;
                    VkDependencyInfo dOv{};
                    dOv.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dOv.imageMemoryBarrierCount = 1;
                    dOv.pImageMemoryBarriers = &toColor;
                    vkCmdPipelineBarrier2(cb, &dOv);

                    VkRenderingAttachmentInfo colorAtt{};
                    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    colorAtt.imageView   = ctx->swapchainImageViews()[imageIndex];
                    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

                    // Read-only depth from the overlay depth prepass. Was
                    // transitioned to DEPTH_STENCIL_READ_ONLY_OPTIMAL at the
                    // end of the prepass; LOAD_OP_LOAD reads the prepass
                    // values, STORE_OP_NONE leaves them alone (overlay
                    // doesn't write depth).
                    VkRenderingAttachmentInfo depthAtt{};
                    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depthAtt.imageView   = rasterGbufs[currentFrame].unjitDepth.view;
                    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                    depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_NONE;

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea.offset = {0, 0};
                    ri.renderArea.extent = ext;
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &colorAtt;
                    ri.pDepthAttachment = &depthAtt;
                    vkCmdBeginRendering(cb, &ri);

                    // Pipeline is selected per-draw based on material.wireframe.
                    // Set viewport/scissor once — they're dynamic state shared
                    // across the wireframe + basic pipelines.
                    VkViewport vpDyn{0.f, 0.f, float(ext.width), float(ext.height), 0.f, 1.f};
                    vkCmdSetViewport(cb, 0, 1, &vpDyn);
                    VkRect2D scDyn{{0, 0}, ext};
                    vkCmdSetScissor(cb, 0, 1, &scDyn);

                    Matrix4 vpUnjitMat;
                    std::memcpy(vpUnjitMat.elements.data(), currVPunjit_.data(), 64);

                    // Track currently-bound pipeline so we don't redundantly
                    // re-bind on every draw when the scene's overlay objects
                    // share a mode (most do).
                    VkPipeline curPipeline = VK_NULL_HANDLE;

                    for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
                        const auto& en = lastVisibleEntries_[i];
                        if (!en.mesh || !en.isOverlay) continue;
                        Color color(1.f, 1.f, 1.f);
                        float opacity = 1.0f;
                        bool wireframe = false;
                        bool transparent = false;
                        if (auto* m = en.mesh->material().get()) {
                            if (auto* mc = dynamic_cast<MaterialWithColor*>(m)) {
                                color = mc->color;
                            }
                            if (auto* mw = dynamic_cast<MaterialWithWireframe*>(m)) {
                                wireframe = mw->wireframe;
                            }
                            opacity     = m->opacity;
                            transparent = m->transparent;
                        }
                        // Wireframe takes precedence — wireframe lines are
                        // typically opaque even when material.transparent
                        // is incidentally true.
                        VkPipeline want;
                        if (wireframe)        want = overlayWireframePipeline;
                        else if (transparent) want = overlayBasicTransparentPipeline;
                        else                  want = overlayBasicPipeline;
                        if (want != curPipeline) {
                            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                            curPipeline = want;
                        }

                        const BlasRecord* rec = resolveBlasForEntry(en);
                        if (!rec || rec->vertex.handle == VK_NULL_HANDLE) continue;

                        Matrix4 model;
                        std::memcpy(model.elements.data(), en.worldMatrix.data(), 64);
                        Matrix4 mvp;
                        mvp.multiplyMatrices(vpUnjitMat, model);

                        struct OverlayPC {
                            float mvp[16];
                            float color[4];
                        } pc{};
                        std::memcpy(pc.mvp, mvp.elements.data(), 64);
                        pc.color[0] = color.r;
                        pc.color[1] = color.g;
                        pc.color[2] = color.b;
                        pc.color[3] = opacity;
                        vkCmdPushConstants(cb, overlayPipelineLayout,
                                           VK_SHADER_STAGE_VERTEX_BIT |
                                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                                           0, sizeof(pc), &pc);

                        VkBuffer     vbufs[1] = {rec->vertex.handle};
                        VkDeviceSize voffs[1] = {0};
                        vkCmdBindVertexBuffers(cb, 0, 1, vbufs, voffs);
                        if (rec->index.handle != VK_NULL_HANDLE) {
                            vkCmdBindIndexBuffer(cb, rec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                            auto* idxAttr = en.mesh->geometry()->getIndex();
                            if (idxAttr) {
                                vkCmdDrawIndexed(cb, static_cast<uint32_t>(idxAttr->count()),
                                                 1, 0, 0, 0);
                            }
                        } else {
                            auto* posAttr = en.mesh->geometry()->getAttribute<float>("position");
                            if (posAttr) {
                                vkCmdDraw(cb, static_cast<uint32_t>(posAttr->count()), 1, 0, 0);
                            }
                        }
                    }

                    // ── Line / LineSegments / Points draws ─────────────────
                    // Always-drawn (no isOverlay flag — these are inherently
                    // overlay; they don't ray-trace). For each entry: ensure
                    // geom upload, push MVP+color, switch topology pipeline.
                    // When material.vertexColors is true AND the geometry has
                    // a color attribute → bind the colored pipeline variant +
                    // a second vertex binding at location 1.
                    //
                    // Points entries (isPoints == true) always use the
                    // POINT_LIST pipeline; the push constant's color.w slot
                    // carries PointsMaterial::size instead of opacity.
                    for (const auto& le : lastVisibleLines_) {
                        std::shared_ptr<BufferGeometry> geomPtr;
                        std::shared_ptr<Material>       matPtr;
                        if (le.isPoints) {
                            if (!le.points) continue;
                            geomPtr = le.points->geometry();
                            matPtr  = le.points->material();
                        } else {
                            if (!le.line) continue;
                            geomPtr = le.line->geometry();
                            matPtr  = le.line->material();
                        }
                        if (!geomPtr) continue;
                        const vulkan::LineRec* lrec = ensureLineGeometryUploaded(geomPtr.get());
                        if (!lrec || lrec->vertex.handle == VK_NULL_HANDLE) continue;

                        Color color(1.f, 1.f, 1.f);
                        float pcW           = 1.0f;// opacity for lines, point-size for points
                        bool useVertexColors = false;
                        if (matPtr) {
                            if (auto* mc = dynamic_cast<MaterialWithColor*>(matPtr.get())) {
                                color = mc->color;
                            }
                            if (le.isPoints) {
                                if (auto* ms = dynamic_cast<MaterialWithSize*>(matPtr.get())) {
                                    pcW = std::max(1.0f, ms->size);
                                } else {
                                    pcW = 3.0f;// sensible default for sizeless materials
                                }
                            } else {
                                pcW = matPtr->opacity;
                            }
                            useVertexColors = matPtr->vertexColors &&
                                              lrec->color.handle != VK_NULL_HANDLE;
                        }

                        VkPipeline want;
                        if (le.isPoints) {
                            // Point pipeline always reads the color binding;
                            // skip the draw if the geometry has none.
                            if (lrec->color.handle == VK_NULL_HANDLE) continue;
                            want = overlayPointListPipeline;
                        } else if (useVertexColors) {
                            want = le.isSegments ? overlayLineListColoredPipeline
                                                 : overlayLineStripColoredPipeline;
                        } else {
                            want = le.isSegments ? overlayLineListPipeline
                                                 : overlayLineStripPipeline;
                        }
                        if (want != curPipeline) {
                            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                            curPipeline = want;
                        }

                        Matrix4 model;
                        std::memcpy(model.elements.data(), le.worldMatrix.data(), 64);
                        Matrix4 mvpL;
                        mvpL.multiplyMatrices(vpUnjitMat, model);

                        struct OverlayPC {
                            float mvp[16];
                            float color[4];
                        } pcL{};
                        std::memcpy(pcL.mvp, mvpL.elements.data(), 64);
                        pcL.color[0] = color.r;
                        pcL.color[1] = color.g;
                        pcL.color[2] = color.b;
                        pcL.color[3] = pcW;
                        vkCmdPushConstants(cb, overlayPipelineLayout,
                                           VK_SHADER_STAGE_VERTEX_BIT |
                                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                                           0, sizeof(pcL), &pcL);

                        const bool twoBindings = le.isPoints || useVertexColors;
                        if (twoBindings) {
                            VkBuffer     vbufsL[2] = {lrec->vertex.handle, lrec->color.handle};
                            VkDeviceSize voffsL[2] = {0, 0};
                            vkCmdBindVertexBuffers(cb, 0, 2, vbufsL, voffsL);
                        } else {
                            VkBuffer     vbufsL[1] = {lrec->vertex.handle};
                            VkDeviceSize voffsL[1] = {0};
                            vkCmdBindVertexBuffers(cb, 0, 1, vbufsL, voffsL);
                        }
                        // Honour BufferGeometry::drawRange — the example
                        // sets it to limit how many of an over-allocated
                        // dynamic vertex buffer are actually rendered.
                        const auto& drawRange = geomPtr->drawRange;
                        if (lrec->index.handle != VK_NULL_HANDLE) {
                            vkCmdBindIndexBuffer(cb, lrec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                            const uint32_t start = static_cast<uint32_t>(std::max(0, drawRange.start));
                            const uint32_t cap   = (lrec->indexCount > start) ? (lrec->indexCount - start) : 0u;
                            const uint32_t cnt   = std::min(cap,
                                    static_cast<uint32_t>(std::max(0, drawRange.count)));
                            if (cnt > 0) vkCmdDrawIndexed(cb, cnt, 1, start, 0, 0);
                        } else {
                            const uint32_t start = static_cast<uint32_t>(std::max(0, drawRange.start));
                            const uint32_t cap   = (lrec->vertexCount > start) ? (lrec->vertexCount - start) : 0u;
                            const uint32_t cnt   = std::min(cap,
                                    static_cast<uint32_t>(std::max(0, drawRange.count)));
                            if (cnt > 0) vkCmdDraw(cb, cnt, 1, start, 0);
                        }
                    }

                    vkCmdEndRendering(cb);

                    // Swapchain back to GENERAL so the downstream blocks
                    // (ImGui overlay or the direct present-src transition)
                    // see the layout they expect.
                    VkImageMemoryBarrier2 toGeneral{};
                    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    toGeneral.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    toGeneral.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    toGeneral.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                              VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                              VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                    toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                              VK_ACCESS_2_TRANSFER_READ_BIT |
                                              VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                              VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    toGeneral.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toGeneral.image = img;
                    toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    toGeneral.subresourceRange.levelCount = 1;
                    toGeneral.subresourceRange.layerCount = 1;
                    VkDependencyInfo dBack{};
                    dBack.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dBack.imageMemoryBarrierCount = 1;
                    dBack.pImageMemoryBarriers = &toGeneral;
                    vkCmdPipelineBarrier2(cb, &dBack);
                    gpuTimings_->end(cb, TP_OverlayDraw, currentFrame);
                }
            }
            // ── End hybrid raster overlay pass ─────────────────────────────────


            // ── End of PT recording. ───────────────────────────────────────────
            // The swapchain image is left in VK_IMAGE_LAYOUT_GENERAL — endFrame
            // handles the ImGui overlay pass + GENERAL → PRESENT_SRC transition,
            // closes the command buffer, and submits.
        }

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
        void recordSceneCapture(VkCommandBuffer cb, uint32_t imageIndex) {
            const VkExtent2D ext = ctx->swapchainExtent();
            if (ext.width == 0 || ext.height == 0) return;
            if (sceneCaptureBufW_ != ext.width || sceneCaptureBufH_ != ext.height ||
                sceneCaptureBuf_.handle == VK_NULL_HANDLE) {
                destroyBuffer(ctx->allocator(), sceneCaptureBuf_);
                const VkDeviceSize bytes =
                        static_cast<VkDeviceSize>(ext.width) * ext.height * 4;
                sceneCaptureBuf_ = createBuffer(
                        ctx->allocator(), ctx->device(), bytes,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
                sceneCaptureBufW_ = ext.width;
                sceneCaptureBufH_ = ext.height;
            }

            const VkImage img = ctx->swapchainImages()[imageIndex];

            VkImageMemoryBarrier toSrc{};
            toSrc.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toSrc.oldLayout                   = VK_IMAGE_LAYOUT_GENERAL;
            toSrc.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.srcAccessMask               = VK_ACCESS_SHADER_WRITE_BIT |
                                                VK_ACCESS_TRANSFER_WRITE_BIT;
            toSrc.dstAccessMask               = VK_ACCESS_TRANSFER_READ_BIT;
            toSrc.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
            toSrc.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
            toSrc.image                       = img;
            toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toSrc.subresourceRange.levelCount = 1;
            toSrc.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &toSrc);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent                 = {ext.width, ext.height, 1};
            vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   sceneCaptureBuf_.handle, 1, &region);

            VkImageMemoryBarrier toGeneral = toSrc;
            toGeneral.oldLayout      = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toGeneral.newLayout      = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcAccessMask  = VK_ACCESS_TRANSFER_READ_BIT;
            toGeneral.dstAccessMask  = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                       VK_ACCESS_SHADER_READ_BIT |
                                       VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        }

        // ── Event-camera shade compute ─────────────────────────────────
        // Creates the event_shade compute pipeline + descriptor set
        // layout. Called once, lazily, on first setEventCameraEnabled.
        void createEventShadePipeline() {
            if (eventShadePipeline_ != VK_NULL_HANDLE) return;

            // 5 bindings: gbufNormal, gbufIds (combined image samplers),
            // matDesc (storage buffer), lightsUbo (uniform), lumaBuf (storage).
            std::array<VkDescriptorSetLayoutBinding, 5> b{};
            b[0].binding         = 0;
            b[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[0].descriptorCount = 1;
            b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            b[1].binding         = 1;
            b[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[1].descriptorCount = 1;
            b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            b[2].binding         = 2;
            b[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[2].descriptorCount = 1;
            b[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            b[3].binding         = 3;
            b[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            b[3].descriptorCount = 1;
            b[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            b[4].binding         = 4;
            b[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[4].descriptorCount = 1;
            b[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(b.size());
            dlci.pBindings    = b.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &eventShadeDsLayout_),
                  "vkCreateDescriptorSetLayout(event_shade)");

            struct ShadePC {
                uint32_t width;
                uint32_t height;
                uint32_t gbufWidth;
                uint32_t gbufHeight;
            };
            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset     = 0;
            pcr.size       = sizeof(ShadePC);

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &eventShadeDsLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcr;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &eventShadePipelineLayout_),
                  "vkCreatePipelineLayout(event_shade)");

            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = sizeof(kEventShadeCompSpv);
            smci.pCode    = kEventShadeCompSpv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &mod),
                  "vkCreateShaderModule(event_shade)");

            VkPipelineShaderStageCreateInfo ssci{};
            ssci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            ssci.module = mod;
            ssci.pName  = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = ssci;
            cpci.layout = eventShadePipelineLayout_;
            check(vkCreateComputePipelines(ctx->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &eventShadePipeline_),
                  "vkCreateComputePipelines(event_shade)");
            vkDestroyShaderModule(ctx->device(), mod, nullptr);

            // Single descriptor pool + set, reused every frame; descriptor
            // writes per-frame because gbuf views + material/light buffers
            // are per-frame and the swapchain can be resized.
            std::array<VkDescriptorPoolSize, 3> ps{};
            ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[0].descriptorCount = 2;
            ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps[1].descriptorCount = 2;
            ps[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps[2].descriptorCount = 1;
            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets       = 1;
            dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
            dpci.pPoolSizes    = ps.data();
            check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &eventShadeDescPool_),
                  "vkCreateDescriptorPool(event_shade)");
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool     = eventShadeDescPool_;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts        = &eventShadeDsLayout_;
            check(vkAllocateDescriptorSets(ctx->device(), &dsai, &eventShadeDescSet_),
                  "vkAllocateDescriptorSets(event_shade)");
        }

        // (Re-)allocate eventLumaBuf_ at the swapchain dimensions.
        // Called from setEventCameraEnabled and on resize.
        void allocateEventLumaBuffer(uint32_t w, uint32_t h) {
            if (eventLumaW_ == w && eventLumaH_ == h && eventLumaBuf_.handle != VK_NULL_HANDLE) return;
            destroyBuffer(ctx->allocator(), eventLumaBuf_);
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
            eventLumaBuf_ = createBuffer(
                    ctx->allocator(), ctx->device(), bytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    0);
            eventLumaW_ = w;
            eventLumaH_ = h;
        }

        // Refresh descriptor set bindings + dispatch the event_shade
        // compute. Called once per frame after the gbuf prepass, before
        // the event_detect dispatch. Gbuf images are in
        // SHADER_READ_ONLY_OPTIMAL at this point (same as for the main
        // RT pipeline's gbuf consumption).
        void recordEventShade(VkCommandBuffer cb, uint32_t frame) {
            if (eventShadePipeline_ == VK_NULL_HANDLE ||
                eventLumaBuf_.handle == VK_NULL_HANDLE) return;

            // Per-frame descriptor writes. Cheap; no descriptor indexing
            // shenanigans needed.
            VkDescriptorImageInfo normalInfo{};
            normalInfo.sampler     = gbufSampler_;
            normalInfo.imageView   = rasterGbufs[frame].normal.view;
            normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo idsInfo{};
            idsInfo.sampler     = gbufSampler_;
            idsInfo.imageView   = rasterGbufs[frame].ids.view;
            idsInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorBufferInfo matInfo{};
            matInfo.buffer = materialDescsBuffers[frame].handle;
            matInfo.offset = 0;
            matInfo.range  = materialDescsBuffers[frame].size
                                     ? materialDescsBuffers[frame].size
                                     : VK_WHOLE_SIZE;

            VkDescriptorBufferInfo lightsInfo{};
            lightsInfo.buffer = lightsUbos[frame].handle;
            lightsInfo.offset = 0;
            lightsInfo.range  = lightsUbos[frame].size ? lightsUbos[frame].size : VK_WHOLE_SIZE;

            VkDescriptorBufferInfo lumaInfo{};
            lumaInfo.buffer = eventLumaBuf_.handle;
            lumaInfo.offset = 0;
            lumaInfo.range  = VK_WHOLE_SIZE;

            std::array<VkWriteDescriptorSet, 5> w{};
            for (auto& it : w) it.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = eventShadeDescSet_;
            w[0].dstBinding = 0;
            w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo = &normalInfo;
            w[1].dstSet = eventShadeDescSet_;
            w[1].dstBinding = 1;
            w[1].descriptorCount = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[1].pImageInfo = &idsInfo;
            w[2].dstSet = eventShadeDescSet_;
            w[2].dstBinding = 2;
            w[2].descriptorCount = 1;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[2].pBufferInfo = &matInfo;
            w[3].dstSet = eventShadeDescSet_;
            w[3].dstBinding = 3;
            w[3].descriptorCount = 1;
            w[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[3].pBufferInfo = &lightsInfo;
            w[4].dstSet = eventShadeDescSet_;
            w[4].dstBinding = 4;
            w[4].descriptorCount = 1;
            w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[4].pBufferInfo = &lumaInfo;

            vkUpdateDescriptorSets(ctx->device(),
                                    static_cast<uint32_t>(w.size()), w.data(),
                                    0, nullptr);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, eventShadePipeline_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     eventShadePipelineLayout_, 0, 1, &eventShadeDescSet_, 0, nullptr);

            struct ShadePC {
                uint32_t width;       // sensor (output) dims
                uint32_t height;
                uint32_t gbufWidth;   // gbuf (input) dims — gbuf is sized to
                uint32_t gbufHeight;  // the render extent (≤ swapchain)
            } pc{};
            pc.width  = eventLumaW_;
            pc.height = eventLumaH_;
            const VkExtent2D rext = renderExtent();
            pc.gbufWidth  = rext.width;
            pc.gbufHeight = rext.height;
            vkCmdPushConstants(cb, eventShadePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                0, sizeof(pc), &pc);

            const uint32_t gx = (eventLumaW_ + 7u) / 8u;
            const uint32_t gy = (eventLumaH_ + 7u) / 8u;
            vkCmdDispatch(cb, gx, gy, 1);

            // Barrier: shade's storage-buffer writes → event_detect's
            // reads (also compute stage).
            VkBufferMemoryBarrier toDetect{};
            toDetect.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            toDetect.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            toDetect.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            toDetect.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDetect.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDetect.buffer = eventLumaBuf_.handle;
            toDetect.size   = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cb,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 0, nullptr, 1, &toDetect, 0, nullptr);
        }

        void recordOverlayAndPresentTransition(VkCommandBuffer cb, uint32_t imageIndex) {
            const VkImage    img = ctx->swapchainImages()[imageIndex];
            const VkExtent2D ext = ctx->swapchainExtent();
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;

            if (overlayCallback) {
                VkImageMemoryBarrier2 toColor{};
                toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                // Swapchain was last written by the TAA dispatch (compute).
                // TRANSFER bits defensively cover any transfer-stage write to
                // the image. Cover both.
                toColor.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                toColor.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                        VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toColor.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                toColor.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                toColor.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                toColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toColor.image = img;
                toColor.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toColor.subresourceRange.levelCount = 1;
                toColor.subresourceRange.layerCount = 1;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &toColor;
                vkCmdPipelineBarrier2(cb, &dep);

                VkRenderingAttachmentInfo colorAtt{};
                colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                colorAtt.imageView = ctx->swapchainImageViews()[imageIndex];
                colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

                VkRenderingInfo ri{};
                ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                ri.renderArea.offset = {0, 0};
                ri.renderArea.extent = ext;
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &colorAtt;
                vkCmdBeginRendering(cb, &ri);
                overlayCallback(static_cast<void*>(cb));
                vkCmdEndRendering(cb);

                VkImageMemoryBarrier2 toPresent{};
                toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                toPresent.dstAccessMask = 0;
                toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toPresent.image = img;
                toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toPresent.subresourceRange.levelCount = 1;
                toPresent.subresourceRange.layerCount = 1;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &toPresent;
                vkCmdPipelineBarrier2(cb, &dep);
            } else {
                VkImageMemoryBarrier2 toPresent{};
                toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                // Swapchain was last written by the TAA dispatch (compute).
                // TRANSFER bits defensively cover any transfer-stage write to
                // the image. Cover both.
                toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                         VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                toPresent.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                          VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                toPresent.dstAccessMask = 0;
                toPresent.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toPresent.image = img;
                toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toPresent.subresourceRange.levelCount = 1;
                toPresent.subresourceRange.layerCount = 1;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &toPresent;
                vkCmdPipelineBarrier2(cb, &dep);
            }
        }

        // Common cmd-buffer-begin epilogue used by both PT and ortho-only
        // frame starts: opens the command buffer and resets the per-frame
        // timestamp pool. Caller must have already acquired the swap image,
        // reset the fence, and reset the cmd buffer.
        void beginCommandRecording(VkCommandBuffer cb) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
            gpuTimings_->beginFrame(cb, currentFrame);
        }

        // Acquire next swapchain image, run all per-frame UBO / descriptor
        // uploads, open the cmd buffer, and record the full PT body. On exit,
        // the swapchain image is in GENERAL and the cmd buffer is open;
        // endFrame() finishes it (overlay + present-src + submit + present).
        // Returns false on swapchain OUT_OF_DATE — caller leaves state Idle.
        bool beginFrameForPT(Object3D& scene, Camera& camera) {
            VkDevice d = ctx->device();
            vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);
            // Fence has signaled → the previous render that wrote into this
            // frame's query pool has retired. Read it now, before we reset
            // the pool and re-record. Result is stored in gpuTimings_ for
            // the public getter to read.
            gpuTimings_->readBack(currentFrame, pendingCpuEnsureSceneMs_);

            // Apply any setter requests deferred from mid-frame. setRenderScale
            // / resetAccumulation issue vkDeviceWaitIdle + reallocate descriptor
            // pool / clear images — both unsafe with an open cmd buffer in the
            // prior frame. We're past the fence wait now so the GPU is idle for
            // *this* slot at minimum; vkDeviceWaitIdle below drains the rest.
            if (pendingRenderScaleRealloc_ || pendingAccumulationReset_ || pendingDeferredRewrite_) {
                vkDeviceWaitIdle(d);
                if (pendingRenderScaleRealloc_) {
                    reallocateRenderExtentResources();// also rewrites deferred descriptors
                    pendingRenderScaleRealloc_ = false;
                    pendingDeferredRewrite_    = false;
                }
                if (pendingAccumulationReset_) {
                    clearGbufImages();
                    pendingAccumulationReset_ = false;
                }
                if (pendingDeferredRewrite_) {
                    // Runtime render-mode toggle: re-point the deferred set at the
                    // current per-frame resources (GPU idle from the wait above).
                    rewriteDeferredDescriptors();
                    pendingDeferredRewrite_ = false;
                }
            }

            uint32_t imageIndex = 0;
            VkResult acq = vkAcquireNextImageKHR(d, ctx->swapchain(), UINT64_MAX,
                                                 imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchainAndDescriptors();
                return false;
            }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                check(acq, "vkAcquireNextImageKHR");
            }
            frameImageIndex_ = imageIndex;

            updateCameraUbo(currentFrame, camera);
            updateLightsUbo(currentFrame, scene);
            updateFogUbo(currentFrame, scene);
            // Safe to write motionMatBuffers[currentFrame] now that the
            // inFlight[currentFrame] fence has been signaled — the GPU has
            // finished its previous use of this slot.
            computeAndUploadMotionMatrices(currentFrame, lastVisibleEntries_);
            // Same fence guarantee covers materialDescsBuffers[currentFrame].
            // ensureSceneBuilt staged any material-value change in
            // matDescsCached_ + flipped matDescsDirty_[*]=true; flush this
            // slot now (the other slot flushes when its frame comes around).
            flushMaterialDescsIfDirty(currentFrame);
            // Per-frame frustum cull: tags every entry with `inFrustum`
            // for the raster passes to consume, and resolves the photon-
            // emit gating flag (glassVisibleThisFrame_) as a side effect.
            cullEntriesAgainstFrustum(camera);
            // Hybrid raster prepass: lazy-create resources on first use,
            // refresh attachments on resize, then upload the per-frame
            // camera VPs (curr jittered, curr unjittered, prev unjittered).
            // Must run after computeAndUploadMotionMatrices so the descriptor
            // rewrite picks up the populated motionMat buffer for this frame.
            ensureHybridResources();
            uploadRasterCameraUbo(currentFrame, camera);
            // Build the per-frame DrawInfo + indirect-cmd buffers used
            // by the indirect-drawing gbuf pass. Runs after the cull
            // pass + camera upload (depends on both) and before record.
            buildIndirectDrawData(currentFrame);
            // Binding 1 (RT denoise output) always targets the TAA input;
            // TAA resolves it to the swapchain (upsampling when renderScale_
            // < 1). Only rewrite when this slot's actual binding differs from the
            // desired mode — avoids firing imageCount_ vkUpdateDescriptorSets
            // calls per frame for nothing once the steady state was reached.
            //
            // TAA's spatial+temporal smoothing is what fixes the moving-object
            // shake from per-frame Halton jitter; the PT accumulator + à-trous
            // gives stills quality (chit primary is high-quality input), TAA
            // sits on top to handle motion.
            {
                const int8_t desired = 1;// binding 1 → bloom sceneHdr (HDR resolve)
                if (binding1Mode_[currentFrame] != desired) {
                    for (uint32_t i = 0; i < imageCount_; ++i) {
                        const uint32_t idx = currentFrame * imageCount_ + i;
                        VkDescriptorImageInfo info{};
                        // denoise.comp (resolve) writes linear HDR here; the
                        // bloom/composite pass then tonemaps it into the TAA
                        // input. (Was taa_->inputView before the HDR split.)
                        info.imageView   = bloom_->sceneHdrView(currentFrame);
                        info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                        VkWriteDescriptorSet w{};
                        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        w.dstSet          = descriptorSets[idx];
                        w.dstBinding      = 1;
                        w.descriptorCount = 1;
                        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                        w.pImageInfo      = &info;
                        vkUpdateDescriptorSets(ctx->device(), 1, &w, 0, nullptr);
                    }
                    binding1Mode_[currentFrame] = desired;
                }
            }
            uploadMeshMovedBits(currentFrame);
            // Same fence guarantees emissiveTriBuffers[currentFrame] is no
            // longer in use; rebuild the per-frame CDF and rewrite binding 14
            // if the buffer grew.
            if (buildAndUploadEmissiveTris(currentFrame, lastVisibleEntries_)) {
                rewriteEmissiveTriDescriptors(currentFrame);
                // Keep the raster-first deferred pass's emissive binding fresh
                // (same per-frame buffer grows for it too).
                if (deferredShade_) {
                    deferredShade_->rewriteEmissive(currentFrame,
                                                    emissiveTriBuffers[currentFrame].handle);
                }
            }
            if (refreshEnvTextureFromScene(scene)) {
                rewriteEnvDescriptors();
                // Env is a primary radiance source — can't reproject, so
                // wipe history. Clearing the gbuf alone makes the mesh-ID
                // guard miss everywhere → next frame's reproject sets
                // histFc=0 globally and we cold-start from sample 1.
                vkDeviceWaitIdle(ctx->device());
                clearGbufImages();
            }

            vkResetFences(d, 1, &inFlight[currentFrame]);
            vkResetCommandBuffer(cmdBuffers[currentFrame], 0);
            beginCommandRecording(cmdBuffers[currentFrame]);
            // Record the full PT body into the now-open cmd buffer. Leaves
            // the swapchain image in GENERAL.
            recordCommandBuffer(cmdBuffers[currentFrame], imageIndex);

            // Scene-only swapchain capture. Runs BEFORE the sprite + ImGui
            // overlays composite — sensor pipelines that consume the
            // renderer's output need a clean image without their own
            // visualisation drawn on top of it, otherwise they'd see
            // their own readout as scene motion and feedback-loop.
            if (sceneCaptureEnabled_) {
                recordSceneCapture(cmdBuffers[currentFrame], imageIndex);
            }

            // GPU event camera detection. Run event_shade first to fill
            // eventLumaBuf_ with deterministic Lambert lighting from the
            // gbuf (eliminates PT noise as a source of false events),
            // then dispatch the detector against that buffer.
            if (eventCamEnabled_ && eventCam_ &&
                eventShadePipeline_ != VK_NULL_HANDLE) {
                recordEventShade(cmdBuffers[currentFrame], currentFrame);
                eventCam_->record(cmdBuffers[currentFrame],
                                  eventLumaBuf_.handle,
                                  eventCamParams_);
            }

            // Screen-space sprite auto-overlay. Walks the main scene for
            // Sprites with screenSpace=true and composites them through
            // an internal ortho camera derived from the swapchain extent —
            // no user code beyond setting the flag on the sprite.
            // recordOrthoOverlay short-circuits when no eligible sprites
            // are found, so the typical no-sprite case pays only the walk.
            const VkExtent2D extPT = ctx->swapchainExtent();
            if (!screenSpaceCam_) {
                screenSpaceCam_ = OrthographicCamera::create(
                        0.f, float(extPT.width), float(extPT.height), 0.f,
                        0.1f, 10.f);
                screenSpaceCam_->position.z = 1.f;
            } else {
                screenSpaceCam_->left   = 0.f;
                screenSpaceCam_->right  = float(extPT.width);
                screenSpaceCam_->top    = float(extPT.height);
                screenSpaceCam_->bottom = 0.f;
                screenSpaceCam_->updateProjectionMatrix();
            }
            overlayPass_->record(cmdBuffers[currentFrame], currentFrame, frameImageIndex_,
                                 scene, *screenSpaceCam_, /*screenSpaceOnly=*/true);
            return true;
        }

        // Ortho-only frame start: no PT, no per-frame uploads. Acquires the
        // swap image, opens the cmd buffer, and clears the swapchain to the
        // configured clearColor (leaves layout = GENERAL). recordOrthoOverlay()
        // then draws the HUD scene atop this cleared image.
        bool beginFrameOrthoOnly() {
            VkDevice d = ctx->device();
            vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);
            gpuTimings_->readBack(currentFrame, pendingCpuEnsureSceneMs_);

            uint32_t imageIndex = 0;
            VkResult acq = vkAcquireNextImageKHR(d, ctx->swapchain(), UINT64_MAX,
                                                 imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchainAndDescriptors();
                return false;
            }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                check(acq, "vkAcquireNextImageKHR");
            }
            frameImageIndex_ = imageIndex;

            vkResetFences(d, 1, &inFlight[currentFrame]);
            vkResetCommandBuffer(cmdBuffers[currentFrame], 0);
            VkCommandBuffer cb = cmdBuffers[currentFrame];
            beginCommandRecording(cb);

            // UNDEFINED → TRANSFER_DST, vkCmdClearColorImage, TRANSFER_DST → GENERAL.
            const VkImage img = ctx->swapchainImages()[imageIndex];
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            {
                VkImageMemoryBarrier2 b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                b.srcAccessMask = 0;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img;
                b.subresourceRange = range;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cb, &dep);
            }
            // Encode into the swapchain's display (sRGB) space. The swapchain is a
            // plain UNORM image and this clear bypasses the shader's linearToSRGB,
            // so a color-managed (linear) clear color must be encoded here to match
            // shaded pixels. No-op when ColorManagement is disabled (legacy raw).
            Color cc;
            cc.copy(clearColor);
            ColorManagement::workingToColorSpace(cc, SRGBColorSpace);
            VkClearColorValue cv{};
            cv.float32[0] = cc.r;
            cv.float32[1] = cc.g;
            cv.float32[2] = cc.b;
            cv.float32[3] = clearAlpha;
            vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &cv, 1, &range);
            {
                VkImageMemoryBarrier2 b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                  VK_ACCESS_2_TRANSFER_READ_BIT |
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img;
                b.subresourceRange = range;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cb, &dep);
            }
            return true;
        }

        // Finish the open frame: record overlay/ImGui pass + PRESENT_SRC
        // transition, close the cmd buffer, submit and present, advance the
        // frame slot, bump sampleIndex. No-op when no frame is open.
        // Registered as Canvas frame-end callback (fires after the user's
        // animate lambda returns); also invoked from render() when the user
        // unexpectedly starts a second perspective render mid-frame.
        void endFrame() {
            if (frameState_ == FrameState::Idle) return;

            const uint32_t imageIndex = frameImageIndex_;
            VkCommandBuffer cb = cmdBuffers[currentFrame];

            recordOverlayAndPresentTransition(cb, imageIndex);
            check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            gpuTimings_->finishRecord();

            VkSemaphoreSubmitInfo waitInfo{};
            waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitInfo.semaphore = imageAvailable[currentFrame];
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

            VkSemaphoreSubmitInfo signalInfo{};
            signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalInfo.semaphore = renderFinished[currentFrame];
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            VkCommandBufferSubmitInfo cbInfo{};
            cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            cbInfo.commandBuffer = cb;

            VkSubmitInfo2 submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            submit.waitSemaphoreInfoCount = 1;
            submit.pWaitSemaphoreInfos = &waitInfo;
            submit.commandBufferInfoCount = 1;
            submit.pCommandBufferInfos = &cbInfo;
            submit.signalSemaphoreInfoCount = 1;
            submit.pSignalSemaphoreInfos = &signalInfo;
            check(vkQueueSubmit2(ctx->graphicsQueue(), 1, &submit, inFlight[currentFrame]),
                  "vkQueueSubmit2");

            VkPresentInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &renderFinished[currentFrame];
            VkSwapchainKHR sc = ctx->swapchain();
            pi.swapchainCount = 1;
            pi.pSwapchains = &sc;
            pi.pImageIndices = &imageIndex;

            VkResult pr = vkQueuePresentKHR(ctx->presentQueue(), &pi);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || needsResize) {
                needsResize = false;
                recreateSwapchainAndDescriptors();
            } else if (pr != VK_SUCCESS) {
                check(pr, "vkQueuePresentKHR");
            }

            // Cap to keep `subIdx = sampleIndex * spp + s` (raygen.rgen)
            // from overflowing uint32. With spp ≤ 256, cap at 2^24 leaves headroom
            // (16M·256 ≈ 4G, just under uint32 max). The previous 65535 cap froze
            // the blue-noise jitter and Halton sequence after ~18 min at 60 fps,
            // causing the per-pixel FC=4096 running mean to slowly absorb the
            // single deterministic sample — image visibly resets from converged
            // toward biased noise. 16M ≈ 75 hours at 60 fps, no longer reachable.
            if (sampleIndex < (1u << 24)) ++sampleIndex;

            currentFrame  = (currentFrame + 1) % kFramesInFlight;
            frameState_   = FrameState::Idle;
        }

        // State-machine dispatcher for VulkanRenderer::render(). First call
        // of a user animate iteration runs beginFrame*, subsequent calls
        // either append HUD overlay draws (ortho path) or finalize
        // the prior frame and restart (defensive — second perspective call).
        // endFrame() runs from the Canvas frame-end callback at the tail of
        // animateOnce().
        void renderFrame(Object3D& scene, Camera& camera) {
            const bool isOrtho = camera.is<OrthographicCamera>();

            if (frameState_ == FrameState::Idle) {
                if (isOrtho) {
                    if (!beginFrameOrthoOnly()) return;
                    frameState_ = FrameState::RecordingOrthoOnly;
                    // Standalone 2D render: draw the ortho overlay (Sprites +
                    // Line/LineSegments) now, so a single render(scene, orthoCam)
                    // call produces output. The HUD pattern (a perspective render
                    // first, then ortho) instead reaches recordOrthoOverlay via the
                    // frame-already-in-flight path below.
                    overlayPass_->record(cmdBuffers[currentFrame], currentFrame, frameImageIndex_,
                                         scene, camera, /*screenSpaceOnly=*/false);
                } else {
                    if (!beginFrameForPT(scene, camera)) return;
                    frameState_ = FrameState::RecordingPostPT;
                }
                return;
            }

            // Frame already in flight from a prior render() call this iteration.
            if (!isOrtho) {
                // User issued a second perspective render — finalize the
                // prior frame, then re-enter from Idle for the new one.
                endFrame();
                renderFrame(scene, camera);
                return;
            }
            // HUD-style ortho render: append Sprite draws onto the open
            // cmd buffer's swapchain image. Mesh / Line HUD overlays are
            // a follow-up — the no-op branch falls through to here so the
            // present still completes from endFrame().
            overlayPass_->record(cmdBuffers[currentFrame], currentFrame, frameImageIndex_,
                                 scene, camera, /*screenSpaceOnly=*/false);
        }
    };

    VulkanRenderer::VulkanRenderer(Canvas& canvas) {
        canvas.initWindow(GraphicsAPI::Vulkan);
        pimpl_ = std::make_unique<Impl>(canvas);
        // Mirror WgpuRenderer: the user's animate lambda may call render()
        // multiple times in one iteration (e.g. main scene + HUD overlay
        // via threepp::HUD). Each render() opens or extends the in-flight
        // frame; the present is deferred to the canvas frame-end callback
        // so all draws land on the same swapchain image. See
        // Impl::endFrame() for the submit+present body.
        canvas.setFrameEndCallback([this] {
            if (pimpl_) pimpl_->endFrame();
        });
    }

    VulkanRenderer::~VulkanRenderer() = default;

    void VulkanRenderer::render(Object3D& scene, Camera& camera) {
        const auto frameStart = std::chrono::high_resolution_clock::now();
        const auto cur = pimpl_->canvas.size();
        if (cur.width() != pimpl_->size.width() || cur.height() != pimpl_->size.height()) {
            pimpl_->needsResize = true;
        }
        // Mirror Renderer-base tone-mapping state into the Impl so renderFrame
        // can push it as a single 16-byte block. Done every render() so users
        // can flip toneMapping / toneMappingExposure freely between frames.
        pimpl_->toneMapping_         = toneMapping;
        pimpl_->toneMappingExposure_ = toneMappingExposure;
        // Only the PT-bound (perspective-camera) render() call needs the
        // scene-build pass — it populates lastVisibleEntries_, the BLAS
        // cache, motion bits and the per-mesh fingerprint state the PT
        // pipeline reads. The HUD pattern's second call (ortho camera over
        // a separate HUD scene) must not touch any of that, or it clobbers
        // motionThisFrame_ / meshMovedBits_ / lastVisibleEntries_ and the
        // next PT frame cold-starts (visibly drops to ~1-spp quality).
        // The ortho overlay record path walks the HUD scene directly instead.
        if (!camera.is<OrthographicCamera>()) {
            const auto sceneStart = std::chrono::high_resolution_clock::now();
            pimpl_->ensureSceneBuilt(scene);
            pimpl_->pendingCpuEnsureSceneMs_ =
                    std::chrono::duration<float, std::milli>(
                            std::chrono::high_resolution_clock::now() - sceneStart)
                            .count();
        }
        // Lazy RT pipeline build (first frame). Deferred from constructor so
        // scene-feature spec constants (kSceneFeatures bitmask, e.g. has-glass
        // gating the caustic gather DCE) can be baked into chit with the
        // detected values from ensureSceneBuilt above. Subsequent frames hit
        // the early-out for free.
        if (!pimpl_->rtPipelinesBuilt_) {
            pimpl_->buildAllRtPipelines();
        }
        // After init, also ensure the *current* variant is built. The first-
        // frame build only constructs the active variant; if the user has
        // since toggled into an unbuilt slot, fill it in now (pays a one-time
        // pipeline-compile cost on first toggle, then never again).
        pimpl_->ensureCurrentRtVariantBuilt();
        pimpl_->renderFrame(scene, camera);
        pimpl_->gpuTimings_->setCpuFrameMs(
                std::chrono::duration<float, std::milli>(
                        std::chrono::high_resolution_clock::now() - frameStart)
                        .count());
    }

    WindowSize VulkanRenderer::size() const { return pimpl_->size; }

    WindowSize VulkanRenderer::framebufferSize() const {
        auto* ctx = pimpl_->ctx.get();
        if (!ctx || ctx->swapchainImages().empty()) return pimpl_->size;
        const VkExtent2D ext = ctx->swapchainExtent();
        return {static_cast<int>(ext.width), static_cast<int>(ext.height)};
    }

    void VulkanRenderer::setSize(const std::pair<int, int>& s) {
        pimpl_->size = WindowSize{s.first, s.second};
        pimpl_->needsResize = true;
    }

    float VulkanRenderer::getTargetPixelRatio() const { return pimpl_->pixelRatio; }
    void VulkanRenderer::setPixelRatio(float v) { pimpl_->pixelRatio = v; }

    void VulkanRenderer::setViewport(const Vector4& v) { pimpl_->viewport = v; }
    void VulkanRenderer::setViewport(int x, int y, int w, int h) {
        pimpl_->viewport.set(static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(w), static_cast<float>(h));
    }

    void VulkanRenderer::setScissor(const Vector4& v) { pimpl_->scissor = v; }
    void VulkanRenderer::setScissor(int x, int y, int w, int h) {
        pimpl_->scissor.set(static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(w), static_cast<float>(h));
    }
    void VulkanRenderer::setScissorTest(bool b) { pimpl_->scissorTest = b; }

    void VulkanRenderer::setClearColor(const Color& c, float a) {
        pimpl_->clearColor = c;
        pimpl_->clearAlpha = a;
    }
    void VulkanRenderer::getClearColor(Color& target) const { target = pimpl_->clearColor; }
    float VulkanRenderer::getClearAlpha() const { return pimpl_->clearAlpha; }
    void VulkanRenderer::setClearAlpha(float a) { pimpl_->clearAlpha = a; }

    void VulkanRenderer::clear(bool, bool, bool) {}

    RenderTarget* VulkanRenderer::getRenderTarget() { return nullptr; }
    void VulkanRenderer::setRenderTarget(RenderTarget*, int, int) {}

    void VulkanRenderer::writeFramebuffer(const std::filesystem::path& filename) {
        auto ext = filename.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") {
            throw std::runtime_error("VulkanRenderer::writeFramebuffer: unsupported format " + ext);
        }
        const auto pixels = readRGBPixels();
        if (pixels.empty()) {
            throw std::runtime_error("VulkanRenderer::writeFramebuffer: no readable framebuffer");
        }
        const auto sz = size();
        const int  w  = sz.width();
        const int  h  = sz.height();
        if (filename.has_parent_path() && !std::filesystem::exists(filename.parent_path())) {
            std::error_code ec;
            std::filesystem::create_directories(filename.parent_path(), ec);
        }
        bool success = false;
        if (ext == ".png") {
            success = stbi_write_png(filename.string().c_str(), w, h, 3, pixels.data(), w * 3);
        } else if (ext == ".jpg" || ext == ".jpeg") {
            success = stbi_write_jpg(filename.string().c_str(), w, h, 3, pixels.data(), 100);
        } else {
            success = stbi_write_bmp(filename.string().c_str(), w, h, 3, pixels.data());
        }
        if (!success) {
            throw std::runtime_error("VulkanRenderer: failed to write framebuffer to " + filename.string());
        }
    }

    std::vector<unsigned char> VulkanRenderer::readRGBPixels() {
        auto& impl = *pimpl_;
        auto* ctx  = impl.ctx.get();
        if (!ctx || ctx->swapchainImages().empty()) return {};

        const VkExtent2D ext   = ctx->swapchainExtent();
        const auto       w     = ext.width;
        const auto       h     = ext.height;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
        if (w == 0 || h == 0) return {};

        // Wait so the previously presented swapchain image is fully written
        // and stable. Cheap unless the user is hammering render() — they
        // usually aren't between an interactive render() and a readback.
        vkDeviceWaitIdle(ctx->device());

        // Allocate a host-visible staging buffer. Reuses the same allocator
        // pattern the LIDAR scanner uses for its readback path.
        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), bytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        // One-shot transient command buffer. Doesn't share the main
        // per-frame command pool because we need to submit + wait
        // synchronously without disturbing the in-flight frame state.
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readRGBPixels)");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readRGBPixels)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(readRGBPixels)");

        const VkImage src = ctx->swapchainImages()[impl.frameImageIndex_];

        // Transition swapchain image PRESENT_SRC → TRANSFER_SRC for the copy.
        VkImageMemoryBarrier toSrc{};
        toSrc.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout                   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toSrc.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcAccessMask               = 0;
        toSrc.dstAccessMask               = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image                       = src;
        toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toSrc.subresourceRange.levelCount = 1;
        toSrc.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent                 = {w, h, 1};
        vkCmdCopyImageToBuffer(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.handle, 1, &region);

        // Restore PRESENT_SRC layout so the next frame can present this slot.
        VkImageMemoryBarrier toPresent = toSrc;
        toPresent.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toPresent.newLayout            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
        toPresent.dstAccessMask        = 0;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toPresent);

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readRGBPixels)");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readRGBPixels)");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readRGBPixels)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readRGBPixels)");

        // Map staging, convert BGRA8_UNORM → RGB8 (Vulkan picks BGRA in
        // VulkanContext::createSwapchain; the surface format is fixed).
        std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
        void* mapped = nullptr;
        vmaInvalidateAllocation(ctx->allocator(), staging.alloc, 0, bytes);
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readRGBPixels)");
        const auto* bgra = static_cast<const unsigned char*>(mapped);
        const size_t pixels = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < pixels; ++i) {
            rgb[i * 3 + 0] = bgra[i * 4 + 2];// R ← B-channel of source
            rgb[i * 3 + 1] = bgra[i * 4 + 1];// G ← G
            rgb[i * 3 + 2] = bgra[i * 4 + 0];// B ← R
        }
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);

        return rgb;
    }

    void VulkanRenderer::setSceneCaptureEnabled(bool enabled) {
        pimpl_->sceneCaptureEnabled_ = enabled;
    }

    bool VulkanRenderer::sceneCaptureEnabled() const {
        return pimpl_->sceneCaptureEnabled_;
    }

    std::vector<unsigned char> VulkanRenderer::readSceneRGBPixels() {
        auto& impl = *pimpl_;
        if (!impl.sceneCaptureEnabled_ || impl.sceneCaptureBuf_.handle == VK_NULL_HANDLE) {
            return {};
        }

        // Wait so the most recent frame's capture is flushed before we
        // memcpy. Same trade-off as readRGBPixels — cheap unless the
        // caller hammers it back-to-back.
        vkDeviceWaitIdle(impl.ctx->device());

        const uint32_t w = impl.sceneCaptureBufW_;
        const uint32_t h = impl.sceneCaptureBufH_;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
        if (bytes == 0) return {};

        vmaInvalidateAllocation(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, 0, bytes);
        void* mapped = nullptr;
        if (vmaMapMemory(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, &mapped) != VK_SUCCESS) {
            return {};
        }
        const auto* bgra = static_cast<const unsigned char*>(mapped);
        std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
        const size_t pixels = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < pixels; ++i) {
            rgb[i * 3 + 0] = bgra[i * 4 + 2];// R ← B
            rgb[i * 3 + 1] = bgra[i * 4 + 1];// G ← G
            rgb[i * 3 + 2] = bgra[i * 4 + 0];// B ← R
        }
        vmaUnmapMemory(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc);
        return rgb;
    }

    void VulkanRenderer::setEventCameraEnabled(bool enabled) {
        auto& impl = *pimpl_;
        if (enabled == impl.eventCamEnabled_) return;
        impl.eventCamEnabled_ = enabled;
        if (enabled) {
            if (!impl.eventCam_) {
                impl.eventCam_ = std::make_unique<vulkan::EventCameraDetector>(*impl.ctx);
            }
            const VkExtent2D ext = impl.ctx->swapchainExtent();
            // Honour any user-pinned sensor resolution; 0 means "track
            // swapchain". Clamp to [16, swapchain] so we never dispatch
            // an empty grid or oversample the gbuf source.
            const uint32_t w = (impl.eventCamUserW_ == 0)
                    ? ext.width
                    : std::clamp(impl.eventCamUserW_, 16u, ext.width);
            const uint32_t h = (impl.eventCamUserH_ == 0)
                    ? ext.height
                    : std::clamp(impl.eventCamUserH_, 16u, ext.height);
            // Wait for any in-flight work that might reference the old
            // images before resize() destroys them.
            vkDeviceWaitIdle(impl.ctx->device());
            impl.eventCam_->resize(w, h);
            // Set up the deterministic shade pipeline — eliminates PT
            // noise as a source of false events. The detector now reads
            // the shade output (eventLumaBuf_) instead of the scene
            // capture buffer, so we don't need to enable sceneCapture
            // for the event camera to work.
            impl.createEventShadePipeline();
            impl.allocateEventLumaBuffer(w, h);
        }
    }

    bool VulkanRenderer::eventCameraEnabled() const {
        return pimpl_->eventCamEnabled_;
    }

    void VulkanRenderer::setEventCameraParams(const EventCameraParams& p) {
        auto& impl = *pimpl_;
        impl.eventCamParams_.threshold         = p.threshold;
        impl.eventCamParams_.decay             = p.decay;
        impl.eventCamParams_.minLuma           = p.minLuma;
        impl.eventCamParams_.maxEventsPerPixel = p.maxEventsPerPixel;
        impl.eventCamParams_.frameTimeUs       = p.frameTimeUs;
    }

    VulkanRenderer::EventCameraParams VulkanRenderer::eventCameraParams() const {
        const auto& src = pimpl_->eventCamParams_;
        EventCameraParams p{};
        p.threshold         = src.threshold;
        p.decay             = src.decay;
        p.minLuma           = src.minLuma;
        p.maxEventsPerPixel = src.maxEventsPerPixel;
        p.frameTimeUs       = src.frameTimeUs;
        return p;
    }

    std::vector<unsigned char> VulkanRenderer::readEventCameraVisualisation() const {
        const auto& impl = *pimpl_;
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return {};
        return impl.eventCam_->readVisualisation();
    }

    size_t VulkanRenderer::readEventCameraVisualisationInto(unsigned char* dst, size_t cap) const {
        const auto& impl = *pimpl_;
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return 0;
        return impl.eventCam_->readVisualisationInto(dst, cap);
    }

    size_t VulkanRenderer::readEventStreamInto(Event* dst, size_t cap,
                                               bool* overflowed) const {
        const auto& impl = *pimpl_;
        if (!impl.eventCamEnabled_ || !impl.eventCam_) {
            if (overflowed) *overflowed = false;
            return 0;
        }
        // Public Event and detector Event are layout-identical (both 16B
        // {x, y, polarity, t_us}); reinterpret is safe and avoids any
        // per-event marshalling cost.
        static_assert(sizeof(Event) == sizeof(vulkan::EventCameraDetector::Event),
                      "Public Event must match detector Event byte-for-byte");
        return impl.eventCam_->readEventStreamInto(
                reinterpret_cast<vulkan::EventCameraDetector::Event*>(dst),
                cap, overflowed);
    }

    void VulkanRenderer::setEventsOnlyMode(bool enabled) {
        pimpl_->eventsOnlyMode_ = enabled;
    }

    bool VulkanRenderer::eventsOnlyMode() const {
        return pimpl_->eventsOnlyMode_;
    }

    void VulkanRenderer::setEventCameraResolution(uint32_t width, uint32_t height) {
        auto& impl = *pimpl_;
        impl.eventCamUserW_ = width;
        impl.eventCamUserH_ = height;
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return;

        // Effective sensor dims: zero means "track swapchain"; otherwise
        // clamp to [16, swapchain] so we never request 0-pixel dispatches
        // or values larger than the gbuf can possibly source.
        const VkExtent2D ext = impl.ctx->swapchainExtent();
        uint32_t w = (width  == 0) ? ext.width  : std::clamp(width,  16u, ext.width);
        uint32_t h = (height == 0) ? ext.height : std::clamp(height, 16u, ext.height);

        vkDeviceWaitIdle(impl.ctx->device());
        impl.eventCam_->resize(w, h);
        impl.allocateEventLumaBuffer(w, h);
    }

    std::pair<uint32_t, uint32_t> VulkanRenderer::eventCameraResolution() const {
        const auto& impl = *pimpl_;
        if (!impl.eventCam_) return {impl.eventCamUserW_, impl.eventCamUserH_};
        return {impl.eventCam_->width(), impl.eventCam_->height()};
    }

    void VulkanRenderer::dispose() { pimpl_.reset(); }

    void* VulkanRenderer::nativeInstance() const {
        return static_cast<void*>(pimpl_->ctx->instance());
    }
    void* VulkanRenderer::nativePhysicalDevice() const {
        return static_cast<void*>(pimpl_->ctx->physicalDevice());
    }
    void* VulkanRenderer::nativeDevice() const {
        return static_cast<void*>(pimpl_->ctx->device());
    }
    void* VulkanRenderer::nativeGraphicsQueue() const {
        return static_cast<void*>(pimpl_->ctx->graphicsQueue());
    }
    uint32_t VulkanRenderer::graphicsQueueFamily() const {
        return pimpl_->ctx->queueFamilies().graphics;
    }
    uint32_t VulkanRenderer::nativeSwapchainFormat() const {
        return static_cast<uint32_t>(pimpl_->ctx->swapchainFormat());
    }
    uint32_t VulkanRenderer::imageCount() const {
        return static_cast<uint32_t>(pimpl_->ctx->swapchainImages().size());
    }

    void VulkanRenderer::setOverlayCallback(std::function<void(void*)> callback) {
        pimpl_->overlayCallback = std::move(callback);
    }

    void VulkanRenderer::setFogAnisotropy(float g) {
        g = std::max(-0.95f, std::min(g, 0.95f));
        if (g != pimpl_->fogAnisotropy_) {
            pimpl_->fogAnisotropy_ = g;
            // Force the per-pixel motion path to halve FC so the new phase
            // function settles quickly. The fog UBO hash will catch this on
            // the next updateFogUbo call too, but flagging here covers the
            // case where setFogAnisotropy is invoked without changing density.
            pimpl_->motionThisFrame_      = true;
            pimpl_->cameraMovedThisFrame_ = true;
        }
    }

    float VulkanRenderer::getFogAnisotropy() const {
        return pimpl_->fogAnisotropy_;
    }

    void VulkanRenderer::setFogWaterSurfaceY(float y) {
        pimpl_->fogWaterSurfaceY_ = y;
    }

    void VulkanRenderer::setSamplesPerPixel(int spp) {
        const uint32_t v = static_cast<uint32_t>(std::max(1, spp));
        pimpl_->samplesPerPixel_ = v;
    }

    int VulkanRenderer::samplesPerPixel() const {
        return static_cast<int>(pimpl_->samplesPerPixel_);
    }

    void VulkanRenderer::setRenderScale(float scale) {
        pimpl_->setRenderScale(scale);
    }

    float VulkanRenderer::renderScale() const {
        return pimpl_->renderScale_;
    }

    void VulkanRenderer::setDenoise(bool enabled) {
        pimpl_->denoiseEnabled_ = enabled;
    }

    bool VulkanRenderer::denoise() const {
        return pimpl_->denoiseEnabled_;
    }

    void VulkanRenderer::setBloomIntensity(float intensity) {
        pimpl_->bloomIntensity_ = intensity < 0.f ? 0.f : intensity;
    }

    float VulkanRenderer::bloomIntensity() const {
        return pimpl_->bloomIntensity_;
    }

    void VulkanRenderer::setDeferredDenoise(bool enabled) {
        pimpl_->denoiseEnabled_ = enabled;
    }

    bool VulkanRenderer::deferredDenoise() const {
        return pimpl_->denoiseEnabled_;
    }

    void VulkanRenderer::setDeferredAO(bool enabled) {
        pimpl_->deferredAO_ = enabled;
    }

    bool VulkanRenderer::deferredAO() const {
        return pimpl_->deferredAO_;
    }

    void VulkanRenderer::setBloomThreshold(float threshold) {
        pimpl_->bloomThreshold_ = threshold < 0.f ? 0.f : threshold;
    }

    float VulkanRenderer::bloomThreshold() const {
        return pimpl_->bloomThreshold_;
    }

    void VulkanRenderer::setBloomClamp(float clampMax) {
        pimpl_->bloomClamp_ = clampMax < 0.f ? 0.f : clampMax;
    }

    float VulkanRenderer::bloomClamp() const {
        return pimpl_->bloomClamp_;
    }

    void VulkanRenderer::setSharpenStrength(float amount) {
        pimpl_->sharpenStrength_ = amount < 0.f ? 0.f : amount;
    }

    float VulkanRenderer::sharpenStrength() const {
        return pimpl_->sharpenStrength_;
    }

    void VulkanRenderer::setSilhouetteMsaaExtra(uint32_t extra) {
        // Cap at 31 so spp + extra fits comfortably in the Halton stride
        // (subIdx = sampleIndex * spp + s); also guards against pathological
        // values that would tank framerate on edge-heavy scenes.
        pimpl_->edgeMsaaExtra_ = (extra > 31u) ? 31u : extra;
    }

    uint32_t VulkanRenderer::silhouetteMsaaExtra() const {
        return pimpl_->edgeMsaaExtra_;
    }

    void VulkanRenderer::setFireflyClamp(float cap) {
        pimpl_->fireflyClamp_ = (cap <= 0.0f) ? 1e30f : cap;
    }

    float VulkanRenderer::fireflyClamp() const {
        const float v = pimpl_->fireflyClamp_;
        return (v > 1e20f) ? 0.0f : v;
    }

    void VulkanRenderer::setMaxBounces(int bounces) {
        // raygen also clamps; clamp on the host too so maxBounces() reads back
        // the actual value used and accumulation resets only on real changes.
        const int clamped = (bounces < 1) ? 1 : (bounces > 16 ? 16 : bounces);
        const uint32_t v = static_cast<uint32_t>(clamped);
        if (pimpl_->maxBounces_ == v) return;
        pimpl_->maxBounces_ = v;
        pimpl_->resetAccumulation();
    }

    int VulkanRenderer::maxBounces() const {
        return static_cast<int>(pimpl_->maxBounces_);
    }

    void VulkanRenderer::setRenderMode(RenderMode mode) {
        if (pimpl_->renderMode_ == mode) return;
        pimpl_->renderMode_ = mode;
        // The deferred (RasterFirst) descriptor set is only rewritten on scene
        // build / resize / env change — never on a mode toggle. Switching modes
        // at runtime can otherwise leave it stale (manifested as a black scene
        // until a window resize forced a rewrite). Flag a refresh, processed at
        // the next frame's GPU-idle point (beginFrameForPT).
        pimpl_->pendingDeferredRewrite_ = true;
    }

    VulkanRenderer::RenderMode VulkanRenderer::renderMode() const {
        return pimpl_->renderMode_;
    }

    VulkanRenderer::SoftBodyInteropHandle
    VulkanRenderer::enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy) {
        return pimpl_->enableSoftBodyInterop(mesh, std::move(deviceCopy));
    }

    void VulkanRenderer::setDeferredVolumetrics(float density, float anisotropy) {
        pimpl_->deferredVolDensity_ = std::max(density, 0.f);
        pimpl_->deferredVolAniso_   = std::clamp(anisotropy, -0.95f, 0.95f);
    }

    void VulkanRenderer::setDeferredStarfield(float intensity) {
        pimpl_->deferredStarIntensity_ = std::max(intensity, 0.f);
    }

    void VulkanRenderer::disableSoftBodyInterop(const Mesh& mesh) {
        pimpl_->disableSoftBodyInterop(mesh);
    }

    void VulkanRenderer::resetAccumulation() {
        pimpl_->resetAccumulation();
    }

    void VulkanRenderer::setPerSppJitterHybrid(bool enabled) {
        pimpl_->perSppJitterHybrid_ = enabled;
    }

    bool VulkanRenderer::perSppJitterHybrid() const {
        return pimpl_->perSppJitterHybrid_;
    }

    void VulkanRenderer::setRestirDIEnabled(bool enabled) {
        if (pimpl_->restirDIEnabled_ == enabled) return;
        pimpl_->restirDIEnabled_ = enabled;
        // ReSTIR is unbiased, so toggling it changes the convergence rate, not
        // the converged mean — on a settled frame the running-mean accumulator
        // would hide the switch entirely. Reset accumulation so the change is
        // actually visible. Gated on sceneBuilt_: before the first render there
        // is nothing accumulated and the gbuf/reservoir images aren't allocated
        // yet (clearGbufImages would touch null handles).
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    bool VulkanRenderer::restirDIEnabled() const {
        return pimpl_->restirDIEnabled_;
    }

    void VulkanRenderer::setRestirDIVisibilityReuse(bool enabled) {
        if (pimpl_->restirDIVisibilityReuse_ == enabled) return;
        pimpl_->restirDIVisibilityReuse_ = enabled;
        // Changes the sampling, not the converged mean — reset so it's visible
        // on a settled frame (same reasoning + pre-first-render gate as
        // setRestirDIEnabled).
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    bool VulkanRenderer::restirDIVisibilityReuse() const {
        return pimpl_->restirDIVisibilityReuse_;
    }

    void VulkanRenderer::setRestirGIEnabled(bool enabled) {
        // Both raygen variants (SER + fallback) are pre-built at init when
        // the device supports SER, so this is just a flag flip — the
        // dispatch site reads `activeRtVariant()` each frame and binds the
        // appropriate pipeline + SBT regions. No GPU work, no rebuild.
        // (We attempted a runtime rebuild here originally; NVIDIA drivers
        // hang on a second vkCreateRayTracingPipelinesKHR against the same
        // pipeline layout, hence the pre-build-both workaround.)
        if (pimpl_->restirGIEnabled_ == enabled) return;
        pimpl_->restirGIEnabled_ = enabled;
        // Reset accumulation so the toggle is visible on a settled frame — same
        // reasoning as setRestirDIEnabled. Gated on sceneBuilt_ for the same
        // pre-first-render safety.
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    bool VulkanRenderer::restirGIEnabled() const {
        return pimpl_->restirGIEnabled_;
    }

    void VulkanRenderer::setSerEnabled(bool enabled) {
        // Pure flag flip — shouldUseSerRaygen() folds this in and
        // activeRtVariant() picks the matching pre-built raygen each frame
        // (same mechanism as setRestirGIEnabled; no pipeline rebuild). SER is
        // a performance-only optimization, so the image is unchanged and no
        // accumulation reset is needed.
        pimpl_->serEnabled_ = enabled;
    }

    bool VulkanRenderer::serEnabled() const {
        return pimpl_->serEnabled_;
    }

    VulkanRenderer::FrameTimings VulkanRenderer::lastFrameTimings() const {
        return pimpl_->gpuTimings_->timings();
    }

    void VulkanRenderer::scanLidar(const std::vector<LidarBeam>& beams,
                                   std::vector<LidarReturn>& results,
                                   const LidarParams& params) {
        pimpl_->scanLidar(beams, results, params);
    }

    void VulkanRenderer::setOverlayLayer(int channel) {
        pimpl_->overlayLayer_ = (channel < 0 || channel > 31) ? -1 : channel;
    }

    int VulkanRenderer::overlayLayer() const {
        return pimpl_->overlayLayer_;
    }

    void VulkanRenderer::setHybridDebugView(int view) {
        using V = Impl::HybridDebugView;
        switch (view) {
            case 1:  pimpl_->hybridDebugView_ = V::Normal; break;
            case 2:  pimpl_->hybridDebugView_ = V::Motion; break;
            case 3:  pimpl_->hybridDebugView_ = V::Ids;    break;
            case 4:  pimpl_->hybridDebugView_ = V::Albedo; break;
            default: pimpl_->hybridDebugView_ = V::Off;    break;
        }
    }

    int VulkanRenderer::hybridDebugView() const {
        using V = Impl::HybridDebugView;
        switch (pimpl_->hybridDebugView_) {
            case V::Normal: return 1;
            case V::Motion: return 2;
            case V::Ids:    return 3;
            default:        return 0;
        }
    }

    void VulkanRenderer::setMeasurePrimaryTraceOnly(bool enabled) {
        pimpl_->measurePrimaryTraceOnly_ = enabled;
    }

    bool VulkanRenderer::measurePrimaryTraceOnly() const {
        return pimpl_->measurePrimaryTraceOnly_;
    }

}// namespace threepp
