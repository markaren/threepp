// VulkanRenderer — hardware ray-traced renderer (Phase 9 v2 + dynamic
// scene rebuild).
//
// On every render() we walk the scene and fingerprint each visible Mesh
// (geometry/material identity + world matrix + PBR scalars). When the
// fingerprint matches the previous frame we skip the rebuild and reuse
// the existing TLAS / desc tables. When it differs we wait the GPU idle,
// retire the TLAS + scene-side desc buffers, build fresh ones, and rewrite
// bindings 0/3/4 across every descriptor set. BLAS records stay cached by
// BufferGeometry pointer so static geometry is never re-traced.
//
// Shading: Phase 5/6 do Lambert + GGX direct lighting against scene
// AmbientLight + DirectionalLights with shadow rays. Phase 7 samples
// `scene.environment` / `scene.background` HDR equirects for both the
// background miss and the closest-hit env probe. Phase 8 adds progressive
// accumulation (rgba32f persistent accum image) with sub-pixel jitter,
// resetting on camera, env, or scene change. Phase 9 turns the ray pipe
// into an iterative path tracer driven from raygen — each closest_hit
// shade returns direct radiance + a sampled bounce direction; raygen
// loops up to kMaxBounces and applies Russian Roulette after the first
// indirect bounce. Phase 9 v2 swaps the cosine-only scatter for a
// probabilistic spec/diffuse split (VNDF-sampled GGX for the spec lobe)
// so polished metals reflect nearby geometry as well as the env.

#define VMA_IMPLEMENTATION
#include "vulkan/VulkanContext.hpp"
#include "vulkan/shaders/vulkan_shared.h"// MaterialDesc + kMaxMaterialTextures + photon-grid constants — same source the shaders read
#include "wgpu/pathtracer/WgpuPathTracerEnvCdf.hpp"// reused: pure C++ template, no WGPU deps

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/water/OceanFFT.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include "threepp/renderers/vulkan/shaders/raygen.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit.rchit.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit_alpha.rahit.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_anyhit.rahit.spv.h"
#include "threepp/renderers/vulkan/shaders/photon_emit.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/photon_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/photon_chit.rchit.spv.h"
#include "threepp/renderers/vulkan/shaders/prefilter_env.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/denoise.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/denoise_atrous.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/water_displace.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuffer.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuffer.frag.spv.h"

#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerBCn.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    using vulkan::VulkanContext;

    namespace {
        constexpr uint32_t kFramesInFlight = 2;

        void check(VkResult r, const char* what) {
            if (r != VK_SUCCESS) {
                throw std::runtime_error(std::string("[VulkanRenderer] ") + what + " failed: " + std::to_string(r));
            }
        }

        // Round x up to the nearest multiple of `align` (align must be POT).
        uint32_t alignUp(uint32_t x, uint32_t align) {
            return (x + align - 1) & ~(align - 1);
        }

        struct Buffer {
            VkBuffer       handle = VK_NULL_HANDLE;
            VmaAllocation  alloc  = VK_NULL_HANDLE;
            VkDeviceSize   size   = 0;
            VkDeviceAddress address = 0;
        };

        Buffer createBuffer(VmaAllocator alloc, VkDevice device,
                            VkDeviceSize size, VkBufferUsageFlags usage,
                            VmaMemoryUsage memoryUsage,
                            VmaAllocationCreateFlags flags = 0) {
            Buffer b{};
            b.size = size;

            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = size;
            bci.usage = usage;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo aci{};
            aci.usage = memoryUsage;
            aci.flags = flags;

            check(vmaCreateBuffer(alloc, &bci, &aci, &b.handle, &b.alloc, nullptr),
                  "vmaCreateBuffer");

            if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
                VkBufferDeviceAddressInfo dai{};
                dai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                dai.buffer = b.handle;
                b.address = vkGetBufferDeviceAddress(device, &dai);
            }
            return b;
        }

        void destroyBuffer(VmaAllocator alloc, Buffer& b) {
            if (b.handle) vmaDestroyBuffer(alloc, b.handle, b.alloc);
            b = {};
        }

        // Phase 7: minimal sampled-image record for the equirect environment
        // map. Held alive for the lifetime of the renderer; recreated when the
        // scene's env texture changes (compared by Texture::id).
        struct Image2D {
            VkImage       image  = VK_NULL_HANDLE;
            VmaAllocation alloc  = VK_NULL_HANDLE;
            VkImageView   view   = VK_NULL_HANDLE;
            VkSampler     sampler = VK_NULL_HANDLE;
            uint32_t      width  = 0;
            uint32_t      height = 0;
            uint32_t      mipLevels = 1;// >1 for the prefiltered env (PMREM)
            VkFormat      format = VK_FORMAT_UNDEFINED;
        };

        void destroyImage2D(VmaAllocator alloc, VkDevice device, Image2D& img) {
            if (img.sampler) vkDestroySampler(device, img.sampler, nullptr);
            if (img.view)    vkDestroyImageView(device, img.view, nullptr);
            if (img.image)   vmaDestroyImage(alloc, img.image, img.alloc);
            img = {};
        }
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
            VkDeviceAddress address = 0;
            // Liveness tag: detects dangling-pointer reuse when a BufferGeometry
            // is destroyed (model unloaded) and the C++ allocator hands the
            // same address to a different geometry. Pruned in ensureSceneBuilt.
            std::weak_ptr<BufferGeometry> liveCheck;
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
            std::vector<float> deformedPositions;
            std::vector<float> deformedNormals;
            std::weak_ptr<BufferGeometry> liveCheck;
        };
        std::unordered_map<const SkinnedMesh*, std::unique_ptr<SkinnedMeshState>> skinnedMeshStates;

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
            // Cascade-0 height readback for CPU-side wave sampling (boat
            // hydrodynamics, IK targets, sound-on-impact, etc.). Host-mapped
            // RG32F buffer of size textureSize²·8 bytes; populated after
            // every IFFT pass via vkCmdCopyImageToBuffer. Cascade 0 only is
            // sufficient for objects larger than ~10 m — the smaller cascades'
            // fine ripple displacement is below typical hull-size sampling.
            Buffer    heightReadback;
            uint32_t  heightReadbackDim = 0;     // copy of textureSize for the staging buffer
            std::weak_ptr<BufferGeometry> liveCheck;
        };
        std::unordered_map<const DisplacedMesh*, std::unique_ptr<DisplacedMeshState>> displacedStates;

        // Renderer-level water_displace pipeline. One compute pipeline shared
        // across all DisplacedMesh instances; per-instance state owns its own
        // descriptor set so binding multiple oceans in one scene is safe.
        VkDescriptorSetLayout displaceDsLayout      = VK_NULL_HANDLE;
        VkPipelineLayout      displacePipelineLayout = VK_NULL_HANDLE;
        VkPipeline            displacePipeline       = VK_NULL_HANDLE;
        VkDescriptorPool      displaceDescPool       = VK_NULL_HANDLE;
        VkSampler             displaceSampler        = VK_NULL_HANDLE;

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
            uint32_t indexed;
            uint32_t _pad;
        };
        // MaterialDesc layout lives in vulkan_shared.h (the same file the GLSL
        // path-tracer shaders pull in via #include). Bringing it into Impl scope
        // here keeps the existing `MaterialDesc md{};` call sites unchanged.
        using MaterialDesc = threepp::vulkan_pt::MaterialDesc;

        Buffer geometryDescsBuffer;
        Buffer materialDescsBuffer;

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
        };
        static_assert(sizeof(GpuFogUbo) == 32);
        std::array<Buffer, kFramesInFlight> fogUbos{};
        float    fogAnisotropy_ = 0.0f;
        uint64_t prevFogHash_ = 0u;

        // Phase 7: environment equirect (HDR float) used by the primary miss
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
        uint32_t envCdfWidth_  = 1;
        uint32_t envCdfHeight_ = 1;
        float    envCdfTotalSum_ = 0.0f;

        // PMREM (Phase 11): GGX-prefiltered env mip chain. Built once per env
        // upload by dispatching prefilter_env.comp for each mip > 0; closest_hit
        // samples textureLod(envTex, dir, roughness * (mipCount - 1)).
        VkDescriptorSetLayout prefilterDsLayout = VK_NULL_HANDLE;
        VkPipelineLayout      prefilterPipelineLayout = VK_NULL_HANDLE;
        VkPipeline            prefilterPipeline = VK_NULL_HANDLE;
        VkDescriptorPool      prefilterDescPool = VK_NULL_HANDLE;
        VkSampler             prefilterSrcSampler = VK_NULL_HANDLE;

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
        // Multi-pass à-trous ping-pong intermediates (rgba32f). filt[0] is the
        // pass 0 / pass 2 destination, filt[1] is the pass 1 destination. After
        // the final atrous pass filt[0] holds the filtered radiance that
        // denoise.comp (finalize) mixes with raw accumImage by per-pixel FC.
        std::array<Image2D, 2> filteredImagesPP{};
        uint32_t accumWriteIdx_ = 0;
        uint32_t sampleIndex = 0;
        // Samples per pixel per frame. Each sample is an independent
        // jittered primary ray; raygen sums them and the accumulator
        // advances FC by `spp` so the running mean stays correctly weighted.
        uint32_t samplesPerPixel_ = 1;
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
        };

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
        // Gates photon emit (no caustics possible without glass) and the
        // 27-cell caustic gather in closest_hit.
        bool     sceneHasGlass_ = false;
        // Per-NEE firefly clamp; pushed to shaders as float bits in slot [15].
        // 1e30f sentinel disables the clamp (set via setFireflyClamp(0)).
        float    fireflyClamp_ = 20.0f;
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
            std::array<float, 16> matrix{};
            std::array<float, 15> pbr{};// + normalScale.xy + transmission/ior + clearcoat/roughness
        };
        std::vector<MeshFingerprint> prevSceneFingerprint;
        // Per-entry record in TLAS-instance order from the last ensureSceneBuilt
        // call. renderFrame consumes this to compute per-instance motion matrices
        // after the in-flight fence has been waited (safe to write the
        // motionMatBuffers[currentFrame] HOST_VISIBLE buffer).
        std::vector<MeshEntry> lastVisibleEntries_;
        bool sceneBuilt_ = false;

        // Ray-tracing pipeline.
        VkDescriptorSetLayout rtDsLayout = VK_NULL_HANDLE;
        VkPipelineLayout      rtPipelineLayout = VK_NULL_HANDLE;
        VkPipeline            rtPipeline = VK_NULL_HANDLE;

        // Shader Binding Table (one record per group: rgen / miss / hit).
        Buffer sbtBuffer;
        VkStridedDeviceAddressRegionKHR rgenRegion{};
        VkStridedDeviceAddressRegionKHR missRegion{};
        VkStridedDeviceAddressRegionKHR hitRegion{};
        VkStridedDeviceAddressRegionKHR callRegion{};// unused

        // Photon-map emit pipeline (separate from rtPipeline; shares rtPipelineLayout).
        VkPipeline photonEmitPipeline = VK_NULL_HANDLE;
        Buffer     photonSbtBuf;
        VkStridedDeviceAddressRegionKHR photonRgenRgn{};
        VkStridedDeviceAddressRegionKHR photonMissRgn{};
        VkStridedDeviceAddressRegionKHR photonHitRgn{};
        VkStridedDeviceAddressRegionKHR photonCallRgn{};

        // Photon-map storage (binding 15 = atomic counters, binding 16 = pos/flux/dir).
        Buffer photonCountBuf;
        Buffer photonDataBuf;

        // Spatial denoiser compute pipelines. Both reuse rtDsLayout (set 0) so
        // the same per-frame descriptor set drives RT, atrous, and finalize.
        // One pipeline layout shared by both compute pipelines (16-byte
        // COMPUTE push constant). denoiseAtrousPipeline runs strided à-trous
        // ping-pong over filteredArray; denoisePipeline finalizes (FC fade +
        // tonemap + sRGB → outImage).
        VkPipelineLayout denoisePipelineLayout    = VK_NULL_HANDLE;
        VkPipeline       denoisePipeline          = VK_NULL_HANDLE;
        VkPipeline       denoiseAtrousPipeline    = VK_NULL_HANDLE;
        bool             denoiseEnabled_          = true;

        // ── Hybrid raster G-buffer prepass ──────────────────────────────────
        // Replaces PT primary-ray traversal: raster writes depth/normal/motion/IDs
        // into per-frame attachments; raygen reads them and starts at bounce 1.
        // Eliminates moving-object shake from PT-primary jitter and makes
        // primary visibility deterministic per pixel. AA happens via TAA on
        // top of raster, not as Monte Carlo on the PT primary. Disabled by
        // default until the integration is validated end-to-end (stage 1).
        struct RasterGbufImages {
            Image2D       normal;       // rgba16f — world-space normal in xyz, .w=0 (roughness in stage 2)
            Image2D       motion;       // rgba16f — NDC delta in .rg, .ba reserved
            Image2D       ids;          // rgba16ui — instanceCustomIndex/meshID/flags/reserved
            Image2D       uv;           // rgba16f — material UV in .rg (raygen samples textures with this in hybrid stage 1A)
            Image2D       depth;        // d32_sfloat
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            uint32_t      width = 0;
            uint32_t      height = 0;
        };
        std::array<RasterGbufImages, kFramesInFlight> rasterGbufs{};
        VkRenderPass rasterGbufRenderPass = VK_NULL_HANDLE;

        VkDescriptorSetLayout rasterDsLayout       = VK_NULL_HANDLE;
        VkPipelineLayout      rasterPipelineLayout = VK_NULL_HANDLE;
        VkPipeline            rasterGbufPipeline   = VK_NULL_HANDLE;
        VkDescriptorPool      rasterDescPool       = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> rasterDescSets{};

        // Per-frame raster camera data. currVPjittered drives gl_Position;
        // currVPunjittered + prevVP drive the motion-vector computation
        // (which must be jitter-free or motion vectors include the jitter
        // and pollute reproject).
        struct RasterCameraData {
            float currVPjittered[16];
            float currVPunjittered[16];
            float prevVP[16];
            float jitter[4];          // .xy = clip-space sub-texel offset, .zw = 1/resolution
        };
        std::array<Buffer, kFramesInFlight> rasterCameraUbos{};
        bool  rasterPrevVPValid_ = false;
        float rasterPrevVP_[16]{};

        // Nearest-filter sampler used by raygen to read gbuffer attachments.
        // Nearest avoids bilinear smearing of normal/motion/ids at silhouettes
        // — primary visibility from raster is already exact-pixel.
        VkSampler gbufSampler_ = VK_NULL_HANDLE;

        // Fallback UV vertex buffer: 8 bytes of zeros, bound to vertex input
        // location 2 when a mesh has no UV attribute (rec->uv.handle is null).
        // Lets the gbuffer pipeline keep a fixed 3-binding layout regardless
        // of per-mesh UV availability.
        Buffer dummyUvBuffer_{};

        // Master toggle for the hybrid path. setHybridEnabled(true) flips it
        // on; off keeps the existing full-PT primary path. Stage 1 ships
        // disabled by default so the raster prepass can land + be validated
        // before becoming the default.
        bool hybridEnabled_ = false;
        // Sub-pixel jitter sequence index for raster TAA. Cycles within a
        // small period (16 frames) — long enough to look stable, short enough
        // to avoid ever-growing index drift.
        uint32_t haltonFrame_ = 0;
        // Day-1 verification mode: blit one G-buffer channel onto the
        // swapchain instead of running the path tracer. Lets us see the
        // raster output before raygen integration.
        enum class HybridDebugView : uint32_t {
            Off    = 0,
            Normal = 1,
            Motion = 2,
            Depth  = 3,
            Ids    = 4,
        };
        HybridDebugView hybridDebugView_ = HybridDebugView::Normal;

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

        explicit Impl(Canvas& c) : canvas(c), size(c.size()) {
            ctx = std::make_unique<VulkanContext>(
                    static_cast<GLFWwindow*>(canvas.windowPtr()),
                    /*enableRayTracing*/ true);

            // The scene-dependent AS build runs lazily on the first render()
            // call. Everything below is scene-independent and safe at ctor time.
            createCommandResources();
            createCameraUbos();
            createLightsUbos();
            createFogUbos();
            createPrefilterPipeline();// must precede createDefaultEnvImage so PMREM is ready
            createDefaultEnvImage();
            rebuildDefaultEnvCdfImages();// 1×1 dummy so descriptors are valid before any HDR upload
            createAccumImage();
            createTextureSampler();
            createDefaultMaterialTexture();
            createRtPipeline();
            createShaderBindingTable();
            createDenoisePipeline();// must follow createRtPipeline (shares rtDsLayout)
            createPhotonBuffers();
            createPhotonEmitPipeline();
            createPhotonSbt();
            createWaterDisplacePipeline();
            // Hybrid raster G-buffer infrastructure is always allocated so
            // the RT descriptor sets can include valid gbuffer-attachment
            // bindings even when hybridEnabled_ stays false. Cheap: ~50MB
            // at 1080p for four attachments × two frames.
            ensureHybridResources();
            createDescriptorPool();
        }

        ~Impl() {
            if (!ctx) return;
            VkDevice d = ctx->device();
            vkDeviceWaitIdle(d);

            for (auto s : imageAvailable) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto s : renderFinished) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto f : inFlight) if (f) vkDestroyFence(d, f, nullptr);
            if (cmdPool) vkDestroyCommandPool(d, cmdPool, nullptr);

            if (descriptorPool) vkDestroyDescriptorPool(d, descriptorPool, nullptr);

            destroyBuffer(ctx->allocator(), sbtBuffer);
            if (rtPipeline)       vkDestroyPipeline(d, rtPipeline, nullptr);
            destroyBuffer(ctx->allocator(), photonSbtBuf);
            if (photonEmitPipeline) vkDestroyPipeline(d, photonEmitPipeline, nullptr);
            destroyBuffer(ctx->allocator(), photonCountBuf);
            destroyBuffer(ctx->allocator(), photonDataBuf);
            if (rtPipelineLayout) vkDestroyPipelineLayout(d, rtPipelineLayout, nullptr);
            if (rtDsLayout)       vkDestroyDescriptorSetLayout(d, rtDsLayout, nullptr);

            if (tlas) ctx->rt().destroyAccelerationStructure(d, tlas, nullptr);
            destroyBuffer(ctx->allocator(), tlasBuffer);
            destroyBuffer(ctx->allocator(), tlasInstancesBuffer);
            destroyBuffer(ctx->allocator(), geometryDescsBuffer);
            destroyBuffer(ctx->allocator(), materialDescsBuffer);

            for (auto& [_, rec] : blasCache) {
                if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                destroyBuffer(ctx->allocator(), rec->storage);
                destroyBuffer(ctx->allocator(), rec->vertex);
                destroyBuffer(ctx->allocator(), rec->index);
                destroyBuffer(ctx->allocator(), rec->normal);
                destroyBuffer(ctx->allocator(), rec->uv);
                destroyBuffer(ctx->allocator(), rec->foam);
            }
            blasCache.clear();

            for (auto& [_, st] : skinnedMeshStates) {
                auto& rec = st->blas;
                if (!rec) continue;
                if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                destroyBuffer(ctx->allocator(), rec->storage);
                destroyBuffer(ctx->allocator(), rec->vertex);
                destroyBuffer(ctx->allocator(), rec->index);
                destroyBuffer(ctx->allocator(), rec->normal);
                destroyBuffer(ctx->allocator(), rec->uv);
                destroyBuffer(ctx->allocator(), rec->foam);
            }
            skinnedMeshStates.clear();

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
                }
                if (st->scratchA.view  != VK_NULL_HANDLE) vkDestroyImageView(d, st->scratchA.view, nullptr);
                if (st->scratchA.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->scratchA.image, st->scratchA.alloc);
                destroyBuffer(ctx->allocator(), st->heightReadback);
                // Per-cascade Phillips / DynamicSpectrum / IFFT are RAII; their
                // destructors handle their own VkImage / VkPipeline / DSet cleanup.
            }
            displacedStates.clear();

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
            for (auto& img : accumImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : gbufImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : filteredImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : materialTextures) destroyImage2D(ctx->allocator(), d, img);
            materialTextures.clear();
            if (textureSampler_) vkDestroySampler(d, textureSampler_, nullptr);

            if (prefilterPipeline)        vkDestroyPipeline(d, prefilterPipeline, nullptr);
            if (prefilterPipelineLayout)  vkDestroyPipelineLayout(d, prefilterPipelineLayout, nullptr);
            if (prefilterDsLayout)        vkDestroyDescriptorSetLayout(d, prefilterDsLayout, nullptr);
            if (prefilterDescPool)        vkDestroyDescriptorPool(d, prefilterDescPool, nullptr);
            if (prefilterSrcSampler)      vkDestroySampler(d, prefilterSrcSampler, nullptr);

            if (denoisePipeline)        vkDestroyPipeline(d, denoisePipeline, nullptr);
            if (denoiseAtrousPipeline)  vkDestroyPipeline(d, denoiseAtrousPipeline, nullptr);
            if (denoisePipelineLayout)  vkDestroyPipelineLayout(d, denoisePipelineLayout, nullptr);

            if (displacePipeline)        vkDestroyPipeline(d, displacePipeline, nullptr);
            if (displacePipelineLayout)  vkDestroyPipelineLayout(d, displacePipelineLayout, nullptr);
            if (displaceDsLayout)        vkDestroyDescriptorSetLayout(d, displaceDsLayout, nullptr);
            if (displaceDescPool)        vkDestroyDescriptorPool(d, displaceDescPool, nullptr);
            if (displaceSampler)         vkDestroySampler(d, displaceSampler, nullptr);

            // Hybrid raster G-buffer cleanup. All resources are lazy-created
            // on first render() with hybridEnabled_ true; if hybrid was never
            // turned on, all handles stay VK_NULL_HANDLE and these calls
            // become no-ops.
            destroyRasterGbufImages();
            for (auto& b : rasterCameraUbos) destroyBuffer(ctx->allocator(), b);
            if (rasterGbufPipeline)     vkDestroyPipeline(d, rasterGbufPipeline, nullptr);
            if (rasterPipelineLayout)   vkDestroyPipelineLayout(d, rasterPipelineLayout, nullptr);
            if (rasterDsLayout)         vkDestroyDescriptorSetLayout(d, rasterDsLayout, nullptr);
            if (rasterDescPool)         vkDestroyDescriptorPool(d, rasterDescPool, nullptr);
            if (rasterGbufRenderPass)   vkDestroyRenderPass(d, rasterGbufRenderPass, nullptr);
            if (gbufSampler_)           vkDestroySampler(d, gbufSampler_, nullptr);
            destroyBuffer(ctx->allocator(), dummyUvBuffer_);
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

        void endAndSubmitOneShot(VkCommandBuffer cb) {
            check(vkEndCommandBuffer(cb), "end one-shot cb");
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot");
            check(vkQueueWaitIdle(ctx->graphicsQueue()), "wait one-shot");
            vkFreeCommandBuffers(ctx->device(), cmdPool, 1, &cb);
        }

        // Build a single BLAS for the given geometry. Vertex / index buffers
        // are uploaded host-mapped, then the AS is built into freshly allocated
        // device storage. The temporary scratch buffer is destroyed on exit.
        std::unique_ptr<BlasRecord> buildBlasFor(const BufferGeometry& geom) {
            auto* posAttr = geom.getAttribute<float>("position");
            if (!posAttr) return nullptr;
            auto* normAttr = geom.getAttribute<float>("normal");
            if (!normAttr) return nullptr;// Phase 5a requires per-vertex normals
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

            // Hybrid: raster G-buffer pre-pass binds these same allocations
            // directly as vertex / index buffers — no duplication, no extra
            // upload, and the raster prepass + RT shadow rays warm the same
            // cache lines.
            const VkBufferUsageFlags geomUsage =
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

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

            Buffer scratch = createBuffer(
                    ctx->allocator(), ctx->device(), blasSizes.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            blasBuild.dstAccelerationStructure = rec->as;
            blasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);

            VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addrInfo.accelerationStructure = rec->as;
            rec->address = ctx->rt().getAccelerationStructureDeviceAddress(ctx->device(), &addrInfo);

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

            auto* posAttr = sm.geometry()->getAttribute<float>("position");
            auto* nrmAttr = sm.geometry()->getAttribute<float>("normal");
            if (!posAttr || !nrmAttr) return nullptr;
            if (!sm.skeleton || sm.skeleton->bones.empty()) return nullptr;
            if (!sm.geometry()->hasAttribute("skinIndex") || !sm.geometry()->hasAttribute("skinWeight")) return nullptr;

            // Build BLAS using the bind-pose positions/normals first so the AS
            // sizes are determined and the storage is allocated. We then run a
            // skin pass (immediate refresh below) so first-frame geometry is
            // already deformed before any ray trace touches it.
            auto rec = buildBlasFor(*sm.geometry());
            if (!rec) return nullptr;
            rec->liveCheck = sm.geometry();

            auto state = std::make_unique<SkinnedMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = sm.geometry();
            state->prevBoneMats.assign(sm.skeleton->bones.size() * 16, 0.f);

            auto* raw = state.get();
            skinnedMeshStates.emplace(&sm, std::move(state));

            // Initial skin so the BLAS reflects the current pose, not bind pose.
            refreshSkinnedBlas(sm, *raw);
            return raw;
        }

        // Re-skin the SkinnedMesh's vertices/normals on the CPU, copy them
        // into the host-mapped vertex/normal buffers, and rebuild the BLAS in
        // place (same AS handle and storage so the device address — and the
        // TLAS reference to it — stay valid). The TLAS doesn't need refit
        // for pose-only changes; the instance's transform is unchanged.
        void refreshSkinnedBlas(SkinnedMesh& sm, SkinnedMeshState& st) {
            cpuSkin(sm, st.deformedPositions, st.deformedNormals);
            if (st.deformedPositions.empty() || !st.blas) return;

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), st.blas->vertex.alloc, &mapped);
            std::memcpy(mapped, st.deformedPositions.data(),
                        st.deformedPositions.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), st.blas->vertex.alloc);

            vmaMapMemory(ctx->allocator(), st.blas->normal.alloc, &mapped);
            std::memcpy(mapped, st.deformedNormals.data(),
                        st.deformedNormals.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), st.blas->normal.alloc);

            // Rebuild the BLAS contents in-place. We use BUILD (not UPDATE)
            // because large pose changes can degrade an UPDATE'd BVH, and
            // BUILD lets us treat each frame uniformly without tracking
            // build-vs-update scratch sizes separately.
            auto* posAttr = sm.geometry()->getAttribute<float>("position");
            auto* idxAttr = sm.geometry()->getIndex();
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

            Buffer scratch = createBuffer(
                    ctx->allocator(), ctx->device(), blasSizes.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);
            blasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);

            // Cache the bone matrices for next frame's dirty detection.
            const auto& bm = sm.skeleton->boneMatrices;
            if (st.prevBoneMats.size() == bm.size()) {
                std::memcpy(st.prevBoneMats.data(), bm.data(), bm.size() * sizeof(float));
            } else {
                st.prevBoneMats = bm;
            }
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
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
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
            const float texN = float(dm.params.textureSize);
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
                water::PhillipsSpectrum::Settings ps{};
                ps.textureSize = dm.params.textureSize;
                ps.tileSize    = tileSizes[i];
                ps.windTheta   = dm.params.windTheta;
                ps.windSpeed   = dm.params.windSpeed;
                // Suppress wavelengths shorter than ~5× the cascade's sample
                // spacing. Without this, single-cascade Phillips puts energy
                // into bands the FFT can't resolve, producing spike-crest
                // aliasing.
                ps.smallWaveCutoff = 5.f * tileSizes[i] / texN;
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
                        *ctx, *c.phillips, dm.params.textureSize, tileSizes[i]);
                c.ifft     = std::make_unique<water::IFFT>(*ctx, dm.params.textureSize);
                state->cascadeMask |= (1u << i);
            }
            if (state->cascadeMask == 0u) return nullptr; // no cascades → invalid setup

            // Cascade-0 height readback buffer (host-mapped). Sized for one
            // RG32F texel per FFT cell; the renderer issues a
            // vkCmdCopyImageToBuffer after each IFFT pass to fill it, then
            // memcpys into DisplacedMesh.heightField for CPU-side wave
            // sampling (boat hydrodynamics, etc.). Kept persistent so the
            // mapping survives between frames.
            const VkDeviceSize readbackBytes =
                    VkDeviceSize(dm.params.textureSize) * VkDeviceSize(dm.params.textureSize) * 8u;
            state->heightReadback = createBuffer(
                    ctx->allocator(), ctx->device(), readbackBytes,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            state->heightReadbackDim = dm.params.textureSize;

            // Scratch image for IFFT ping-pong (RG32F, same size as the dyn
            // images). One is enough for Phase 1 — both IFFT calls in a
            // frame can share since they run sequentially on the same queue.
            // Use the same helper OceanFFT.cpp uses internally — create directly.
            // (Kept duplicated rather than exposing OceanFFT's internal helpers.)
            {
                VkImageCreateInfo ici{};
                ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType     = VK_IMAGE_TYPE_2D;
                ici.format        = VK_FORMAT_R32G32_SFLOAT;
                ici.extent        = {dm.params.textureSize, dm.params.textureSize, 1};
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
                state->scratchA.width  = dm.params.textureSize;
                state->scratchA.height = dm.params.textureSize;
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = state->scratchA.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = VK_FORMAT_R32G32_SFLOAT;
                vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &state->scratchA.view),
                      "vkCreateImageView(displaceScratch)");
            }

            // Allocate this mesh's displace descriptor set + write bindings.
            VkDescriptorSetAllocateInfo dai{};
            dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dai.descriptorPool     = displaceDescPool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts        = &displaceDsLayout;
            check(vkAllocateDescriptorSets(ctx->device(), &dai, &state->displaceDS),
                  "vkAllocateDescriptorSets(displace)");

            // Bind each enabled cascade's spatial images to its (height, displace)
            // slot pair. Disabled cascades are filled with cascade 0's images
            // so the shader's combined-image-sampler bindings are always valid;
            // the shader gates which slots are actually sampled via cascadeMask.
            std::array<VkDescriptorImageInfo, 6> imageInfos{};
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t srcCascade = (state->cascadeMask & (1u << i)) ? i : 0u;
                const auto& c = state->cascades[srcCascade];
                imageInfos[i * 2 + 0].sampler     = displaceSampler;
                imageInfos[i * 2 + 0].imageView   = c.dyn->ht().view;
                imageInfos[i * 2 + 0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                imageInfos[i * 2 + 1].sampler     = displaceSampler;
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

                // Cascade 0 only: copy the spatial-domain height image into the
                // host-mapped readback buffer. Same dispatch as the IFFT — by
                // the time endAndSubmitOneShot returns, the buffer is filled.
                // Image stays in GENERAL layout; we add a memory barrier so the
                // copy reads what the IFFT wrote.
                if (i == 0 && st.heightReadback.handle != VK_NULL_HANDLE) {
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
                    bic.bufferRowLength = 0;     // tightly packed
                    bic.bufferImageHeight = 0;
                    bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    bic.imageOffset = {0, 0, 0};
                    bic.imageExtent = {st.heightReadbackDim, st.heightReadbackDim, 1};
                    vkCmdCopyImageToBuffer(cb, c.dyn->ht().image, VK_IMAGE_LAYOUT_GENERAL,
                                           st.heightReadback.handle, 1, &bic);

                    // Make the host-side memcpy after submit see the GPU writes.
                    VkBufferMemoryBarrier bmb{};
                    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    bmb.buffer = st.heightReadback.handle;
                    bmb.size   = VK_WHOLE_SIZE;
                    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT,
                            0, 0, nullptr, 1, &bmb, 0, nullptr);
                }
            }

            // (4) Dispatch water_displace.comp → writes positions + normals
            // into the BLAS vertex/normal buffers.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, displacePipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, displacePipelineLayout,
                                    0, 1, &st.displaceDS, 0, nullptr);

            struct DisplacePc {
                VkDeviceAddress posOut;
                VkDeviceAddress normOut;
                VkDeviceAddress foamOut;
                uint32_t vertexCount;
                uint32_t gridDim;
                float    planeSize;
                float    tileSize0;
                float    tileSize1;
                float    tileSize2;
                float    waveScale;
                float    choppiness;
                uint32_t cascadeMask;
            } pc{};
            pc.posOut       = st.blas->vertex.address;
            pc.normOut      = st.blas->normal.address;
            pc.foamOut      = st.blas->foam.address;// 0 if foam buffer not allocated → shader skips foam write
            pc.vertexCount  = st.vertexCount;
            pc.gridDim      = st.gridDim;
            pc.planeSize    = st.planeSize;
            pc.tileSize0    = dm.params.tileSize0;
            pc.tileSize1    = dm.params.tileSize1;
            pc.tileSize2    = dm.params.tileSize2;
            pc.waveScale    = dm.params.waveScale;
            pc.choppiness   = dm.params.choppiness;
            pc.cascadeMask  = st.cascadeMask;
            vkCmdPushConstants(cb, displacePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);
            const uint32_t groups = (st.vertexCount + 63u) / 64u;
            vkCmdDispatch(cb, groups, 1, 1);

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

            VkAccelerationStructureBuildGeometryInfoKHR build{};
            build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            build.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            build.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            build.geometryCount = 1;
            build.pGeometries = &blasGeom;
            build.dstAccelerationStructure = st.blas->as;

            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &build, &primCount, &sizes);

            Buffer scratch = createBuffer(
                    ctx->allocator(), ctx->device(), sizes.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);
            build.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &build, &pRange);

            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);

            // Mirror cascade-0 height into DisplacedMesh's public field.
            // endAndSubmit waits for completion, so the staging buffer is
            // fully written by the time we get here. RG32F packed; user
            // code samples via DisplacedMesh::sampleHeight(worldX, worldZ).
            if (st.heightReadback.handle != VK_NULL_HANDLE && st.heightReadbackDim > 0) {
                const size_t cells = size_t(st.heightReadbackDim) * size_t(st.heightReadbackDim);
                if (dm.heightField.size() != cells * 2) {
                    dm.heightField.assign(cells * 2, 0.f);
                }
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), st.heightReadback.alloc, &mapped);
                std::memcpy(dm.heightField.data(), mapped, cells * 2 * sizeof(float));
                vmaUnmapMemory(ctx->allocator(), st.heightReadback.alloc);
                dm.heightFieldDim      = st.heightReadbackDim;
                dm.heightFieldTileSize = dm.params.tileSize0;
            }
        }

        // Build a TLAS over the supplied instance descriptors. Empty input is
        // legal — produces an empty TLAS that always misses (Phase 2 fallback).
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

            Buffer scratch = createBuffer(
                    ctx->allocator(), ctx->device(), tlasSizes.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            tlasBuild.dstAccelerationStructure = tlas;
            tlasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);
        }

        // Refit the existing TLAS in-place with new instance transforms.
        // Cheaper than a full rebuild and crucially leaves the TLAS handle
        // unchanged so descriptor binding 0 keeps pointing at it. Caller
        // must hold the same instance count as the previous build (only
        // matrices may change) — topology growth requires a full rebuild.
        void refitTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances) {
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

            VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
            tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            tlasBuild.srcAccelerationStructure = tlas;
            tlasBuild.dstAccelerationStructure = tlas;
            tlasBuild.geometryCount = 1;
            tlasBuild.pGeometries = &tlasGeom;

            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &tlasBuild, &instanceCount, &sizes);

            Buffer scratch = createBuffer(
                    ctx->allocator(), ctx->device(), sizes.updateScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);
            tlasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
            endAndSubmitOneShot(cb);
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
            }
            // Alpha-blend transparency (transparent=true, opacity<1) has no
            // physical analogue in a PT, so treat it as stochastic pass-through:
            // with probability (1-opacity) the ray continues straight through
            // (ior=1 → refract returns the incident direction unchanged, F=0).
            if (d.transmission == 0.0f && mat->transparent && mat->opacity < 1.0f) {
                d.transmission = 1.0f - mat->opacity;
                d.ior          = 1.0f;
            }
            // BLEND mode with texture alpha (alphaMode=BLEND, opacity=1.0):
            // alphaCutoff=-1.0 sentinel triggers per-texel stochastic blend in
            // closest_hit using the albedo texture's alpha channel.
            if (mat->transparent && d.alphaCutoff == 0.0f && d.transmission == 0.0f) {
                d.alphaCutoff = -1.0f;
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
            d.doubleSided = (mat->side == Side::Double) ? 1 : 0;
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
            scene.updateMatrixWorld(true);

            // First call each render(): start with no motion. Camera + mesh
            // motion checks below + in updateCameraUbo OR true into this flag
            // before dispatch.
            motionThisFrame_ = false;
            cameraMovedThisFrame_ = false;
            std::fill(meshMovedBits_.begin(), meshMovedBits_.end(), 0u);

            // Expand the visible scene into one MeshEntry per TLAS instance.
            // Regular meshes contribute one entry; an InstancedMesh contributes
            // count() entries each with worldMatrix = matrixWorld * instanceMat[i].
            // Mirrors WGPU's expandMeshEntries (WgpuPathTracerAtlas.cpp:20).
            std::vector<MeshEntry> entries;
            scene.traverse([&](Object3D& o) {
                auto* m = dynamic_cast<Mesh*>(&o);
                if (!m || !m->visible) return;
                auto geom = m->geometry();
                if (!geom || !geom->hasAttribute("position")) return;
                if (!geom->hasAttribute("normal")) return;
                // Wireframe meshes can't be path-traced — a triangle hit would
                // shade the whole face. WGPU PT routes these to its raster
                // overlay; until that lands in the Vulkan path, skip them so
                // they don't show up as solid blobs. Helpers/debug visuals
                // disappear, which matches WGPU+overlay-disabled.
                if (auto mat = m->material()) {
                    if (auto* wf = dynamic_cast<MaterialWithWireframe*>(mat.get()); wf && wf->wireframe) return;
                }
                if (auto* inst = dynamic_cast<InstancedMesh*>(m); inst && inst->count() > 0) {
                    Matrix4 instMat;
                    Matrix4 world;
                    for (size_t j = 0; j < inst->count(); ++j) {
                        inst->getMatrixAt(j, instMat);
                        world.multiplyMatrices(*m->matrixWorld, instMat);
                        MeshEntry e{};
                        e.mesh = m;
                        e.instanceIndex = static_cast<uint32_t>(j);
                        std::memcpy(e.worldMatrix.data(), world.elements.data(), 64);
                        entries.push_back(e);
                    }
                } else {
                    MeshEntry e{};
                    e.mesh = m;
                    e.instanceIndex = 0u;
                    std::memcpy(e.worldMatrix.data(), m->matrixWorld->elements.data(), 64);
                    entries.push_back(e);
                }
            });

            // Per-entry fingerprint construction. Hot path on big static scenes
            // (Bistro): when mesh + mat + geom pointers all match prev frame
            // AND mat->version() also matches, the 8 texture-of lookups and
            // materialFromMesh (~21 dynamic_casts/mesh) all return identical
            // results — copy them straight from prevSceneFingerprint[i] and
            // only refresh the world matrix. Bistro has ~1500 meshes which
            // would otherwise cost ~31k dynamic_casts/frame here alone.
            std::vector<MeshFingerprint> currFp;
            currFp.resize(entries.size());
            const bool prevValid = sceneBuilt_ && prevSceneFingerprint.size() == entries.size();
            for (size_t i = 0; i < entries.size(); ++i) {
                const MeshEntry& en = entries[i];
                Mesh* m = en.mesh;
                auto matSp = m->material();
                const Material* matPtr = matSp.get();
                const unsigned int matVer = matPtr ? matPtr->version() : 0u;
                const void* geomPtr = m->geometry().get();

                MeshFingerprint& fp = currFp[i];
                bool fastPath = false;
                if (prevValid) {
                    const auto& p = prevSceneFingerprint[i];
                    if (p.mesh == m && p.mat == matPtr && p.geom == geomPtr &&
                        p.instanceIndex == en.instanceIndex &&
                        p.matVersion == matVer) {
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

            // Per-entry bone-dirty bits. SkinnedMesh poses change without
            // touching the SkinnedMesh's worldMatrix, so the matrix-fingerprint
            // misses them. We compare current Skeleton::boneMatrices against
            // the cached prevBoneMats for each known SkinnedMesh; first-time
            // skinned meshes mark dirty so the structural-rebuild path skins
            // before its first ray trace.
            std::vector<bool> entryBonesDirty(entries.size(), false);
            for (size_t i = 0; i < entries.size(); ++i) {
                auto* sm = dynamic_cast<SkinnedMesh*>(entries[i].mesh);
                if (!sm || !sm->skeleton || sm->skeleton->bones.empty()) continue;
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
            // advances continuously). Same per-entry-bool pattern as bones.
            std::vector<bool> entryDisplacedDirty(entries.size(), false);
            for (size_t i = 0; i < entries.size(); ++i) {
                if (dynamic_cast<DisplacedMesh*>(entries[i].mesh)) {
                    entryDisplacedDirty[i] = true;
                }
            }

            // Continuous-motion fast path: when only the per-mesh matrices
            // changed (everything else — topology, materials, textures —
            // matches), refit the TLAS in place and let raygen reproject.
            // We only have to detect the matrix-only case ahead of time;
            // motion matrices themselves are computed each frame in
            // renderFrame so we can defer the host write past the fence wait.
            lastVisibleEntries_ = entries;
            if (sceneBuilt_ && currFp.size() == prevSceneFingerprint.size()) {
                // Three classes of change:
                //   structural    — pointers (mesh/geom/mat/textures): full rebuild.
                //   matrices      — per-mesh world matrices: TLAS refit + bit set.
                //   materialVals  — pbr floats (KHR_animation_pointer animates colors,
                //                   roughness, etc): re-upload matDescs in place + bit set.
                // Splitting matters: KHR_animation_pointer changes pbr every frame
                // without changing any pointer or texture. Lumping it under
                // structural caused full rebuild every frame, which reset the
                // gbuf+sampleIndex globally and froze accumulation scene-wide.
                bool structuralSame = true;
                bool matricesSame = true;
                bool materialValuesSame = true;
                bool bonesDirtyAny = false;
                bool displacedDirtyAny = false;
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
                    const bool matChanged   = std::memcmp(a.pbr.data(),    b.pbr.data(),    sizeof(a.pbr))    != 0;
                    const bool bonesChanged = entryBonesDirty[i];
                    const bool dispChanged  = entryDisplacedDirty[i];
                    if (xfmChanged) matricesSame = false;
                    if (matChanged) materialValuesSame = false;
                    if (bonesChanged) bonesDirtyAny = true;
                    if (dispChanged)  displacedDirtyAny = true;
                    // All flavors of change invalidate this pixel's history —
                    // share the same per-mesh bit. Reproject+halve FC for any
                    // of: matrix shift, pbr shift, pose deformation, ocean
                    // surface displacement.
                    if (xfmChanged || matChanged || bonesChanged || dispChanged) {
                        const size_t w = i >> 5;
                        if (w >= meshMovedBits_.size()) meshMovedBits_.resize(w + 1, 0u);
                        meshMovedBits_[w] |= (1u << (i & 31u));
                    }
                }
                if (structuralSame) {
                    if (!matricesSame || !materialValuesSame || bonesDirtyAny || displacedDirtyAny) {
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
                            auto* sm = dynamic_cast<SkinnedMesh*>(entries[i].mesh);
                            if (!sm) continue;
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
                            auto* dm = dynamic_cast<DisplacedMesh*>(entries[i].mesh);
                            if (!dm) continue;
                            auto stIt = displacedStates.find(dm);
                            if (stIt == displacedStates.end()) continue;
                            refreshDisplacedBlas(*dm, *stIt->second, now);
                            ++dm->frameTick;
                        }
                    }
                    if (!matricesSame || bonesDirtyAny || displacedDirtyAny) {
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
                        for (const MeshEntry& en : entries) {
                            VkDeviceAddress blasAddr = 0;
                            if (auto* sm = dynamic_cast<SkinnedMesh*>(en.mesh); sm && sm->skeleton && !sm->skeleton->bones.empty()) {
                                auto smIt = skinnedMeshStates.find(sm);
                                if (smIt == skinnedMeshStates.end()) continue;
                                blasAddr = smIt->second->blas->address;
                            } else if (auto* dm = dynamic_cast<DisplacedMesh*>(en.mesh)) {
                                auto dmIt = displacedStates.find(dm);
                                if (dmIt == displacedStates.end()) continue;
                                blasAddr = dmIt->second->blas->address;
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
                            inst.instanceCustomIndex = static_cast<uint32_t>(instances.size());
                            inst.mask = 0xFFu;
                            inst.instanceShaderBindingTableRecordOffset = 0;
                            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                            inst.accelerationStructureReference = blasAddr;
                            instances.push_back(inst);
                        }
                        refitTlas(instances);
                    }
                    if (!materialValuesSame) {
                        // Material-values-only update: rebuild MaterialDescs and
                        // memcpy into the existing materialDescsBuffer. Pointers
                        // and textures haven't changed, so the matDescs slot count
                        // and texture indices stay valid; only the pbr floats need
                        // to flow through. Wait the device idle since the buffer is
                        // shared across frames-in-flight — without it we'd race a
                        // previous frame's RT trace still reading the old contents.
                        vkDeviceWaitIdle(ctx->device());
                        std::vector<MaterialDesc> matDescs;
                        matDescs.reserve(entries.size());
                        for (const MeshEntry& en : entries) {
                            Mesh* m = en.mesh;
                            MaterialDesc md = materialFromMesh(*m);
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
                            matDescs.push_back(md);
                        }
                        if (!matDescs.empty()) {
                            void* mapped = nullptr;
                            vmaMapMemory(ctx->allocator(), materialDescsBuffer.alloc, &mapped);
                            std::memcpy(mapped, matDescs.data(),
                                        matDescs.size() * sizeof(MaterialDesc));
                            vmaUnmapMemory(ctx->allocator(), materialDescsBuffer.alloc);
                        }
                        sceneHasGlass_ = false;
                        for (const auto& md : matDescs) {
                            if (md.transmission > 0.0f) { sceneHasGlass_ = true; break; }
                        }
                    }
                    // Update prevSceneFingerprint so later frames compare
                    // against this frame's state, not stale.
                    prevSceneFingerprint = std::move(currFp);
                    return;
                }
            }

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
                destroyBuffer(ctx->allocator(), materialDescsBuffer);
                tlasBuffer = {};
                tlasInstancesBuffer = {};
                geometryDescsBuffer = {};
                materialDescsBuffer = {};

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
                        it = blasCache.erase(it);
                    } else {
                        ++it;
                    }
                }
                for (auto it = skinnedMeshStates.begin(); it != skinnedMeshStates.end(); ) {
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
                        }
                        it = skinnedMeshStates.erase(it);
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
                        }
                        if (st->scratchA.view  != VK_NULL_HANDLE) vkDestroyImageView(ctx->device(), st->scratchA.view, nullptr);
                        if (st->scratchA.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->scratchA.image, st->scratchA.alloc);
                        // PhillipsSpectrum / DynamicSpectrum / IFFT destructors
                        // run on unique_ptr reset.
                        it = displacedStates.erase(it);
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
            std::vector<GeometryDesc> geomDescs;
            std::vector<MaterialDesc> matDescs;
            instances.reserve(entries.size());
            geomDescs.reserve(entries.size());
            matDescs.reserve(entries.size());

            for (const MeshEntry& en : entries) {
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
                if (auto* sm = dynamic_cast<SkinnedMesh*>(m); sm && sm->skeleton && !sm->skeleton->bones.empty()) {
                    auto* st = ensureSkinnedBlas(*sm);
                    if (!st) continue;
                    recPtr = st->blas.get();
                } else if (auto* dm = dynamic_cast<DisplacedMesh*>(m)) {
                    auto* st = ensureDisplacedState(*dm);
                    if (!st) continue;
                    recPtr = st->blas.get();
                    // Trigger an initial FFT/displace dispatch so the BLAS
                    // contents (rest grid right now) become the displaced
                    // surface before the first ray-trace sees it.
                    refreshDisplacedBlas(*dm, *st, static_cast<float>(glfwGetTime()));
                } else {
                    auto it = blasCache.find(geomKey);
                    if (it == blasCache.end()) {
                        auto rec = buildBlasFor(*m->geometry());
                        if (!rec) continue;// degenerate / unsupported geometry
                        rec->liveCheck = m->geometry();
                        it = blasCache.emplace(geomKey, std::move(rec)).first;
                    }
                    recPtr = it->second.get();
                }

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
                inst.instanceCustomIndex = static_cast<uint32_t>(geomDescs.size());
                inst.mask = 0xFFu;
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
                gdesc.indexed = recPtr->index.handle != VK_NULL_HANDLE ? 1u : 0u;
                gdesc._pad = 0;
                geomDescs.push_back(gdesc);

                MaterialDesc md = materialFromMesh(*m);
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
                matDescs.push_back(md);
            }

            buildTlas(instances);
            uploadDescBuffer(geometryDescsBuffer, geomDescs);
            uploadDescBuffer(materialDescsBuffer, matDescs);
            sceneHasGlass_ = false;
            for (const auto& md : matDescs) {
                if (md.transmission > 0.0f) { sceneHasGlass_ = true; break; }
            }

            // Topology rebuild: prev gbuf + accum hold mesh IDs from the old
            // scene, but the gl_InstanceCustomIndexEXT space just changed
            // meaning. Clearing the gbuf to 0 (mesh ID = 0 = sky) makes the
            // reproject mesh-ID guard miss everywhere → histFc=0 globally
            // on the very next frame, mirroring the old sampleIndex=0 reset
            // without throwing away a valid history when meshes happen to
            // re-shuffle into the same slots. We're already past
            // vkDeviceWaitIdle so the clear is safely synchronous.
            clearGbufImages();
            sampleIndex = 0;
            prevWorldMats.clear();

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
            prevSceneFingerprint = std::move(currFp);
            sceneBuilt_ = true;
        }

        void createCameraUbos() {
            for (auto& b : cameraUbos) {
                b = createBuffer(
                        ctx->allocator(), ctx->device(),
                        /*size*/ 2 * 16 * sizeof(float),// viewInverse + projInverse
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

            std::array<VkImage, 4> images = {
                    gbufImagesPP[0].image, gbufImagesPP[1].image,
                    accumImagesPP[0].image, accumImagesPP[1].image,
            };

            std::array<VkImageMemoryBarrier2, 4> toTransfer{};
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

            std::array<VkImageMemoryBarrier2, 4> toGeneral{};
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

            std::vector<float> data(count * 16);
            for (uint32_t i = 0; i < count; ++i) {
                Matrix4 cur;
                std::memcpy(cur.elements.data(), entries[i].worldMatrix.data(), 64);

                Matrix4 motion;// identity by default
                EntryKey key{entries[i].mesh, entries[i].instanceIndex};
                auto it = prevWorldMats.find(key);
                if (it != prevWorldMats.end()) {
                    Matrix4 prev;
                    std::memcpy(prev.elements.data(), it->second.data(), 64);
                    Matrix4 curInv;
                    curInv.copy(cur).invert();
                    motion.multiplyMatrices(prev, curInv);
                }
                std::memcpy(&data[i * 16], motion.elements.data(), 64);
            }

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), motionMatBuffers[frame].alloc, &mapped);
            std::memcpy(mapped, data.data(), data.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), motionMatBuffers[frame].alloc);

            // Record this frame's matrices for next frame's motion delta.
            // Only update entries we just wrote; entries removed from the
            // scene are pruned naturally (full-rebuild path clears the map).
            for (uint32_t i = 0; i < count; ++i) {
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

        void updateCameraUbo(uint32_t frame, Camera& camera) {
            camera.updateMatrixWorld(true);

            float data[32];
            std::memcpy(data + 0,  camera.matrixWorld->elements.data(),            64);
            std::memcpy(data + 16, camera.projectionMatrixInverse.elements.data(), 64);

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
                if (dx*dx + dy*dy + dz*dz > 1e-6f ||
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

        // Resolve the BlasRecord backing a given visible entry. The same
        // physical buffers feed BLAS and the raster prepass (VERTEX_BUFFER_BIT
        // was added at allocation), so this is a pure lookup, no upload.
        const BlasRecord* resolveBlasForEntry(const MeshEntry& en) const {
            if (auto* sm = dynamic_cast<SkinnedMesh*>(en.mesh)) {
                auto it = skinnedMeshStates.find(sm);
                if (it != skinnedMeshStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
            }
            if (auto* dm = dynamic_cast<DisplacedMesh*>(en.mesh)) {
                auto it = displacedStates.find(dm);
                if (it != displacedStates.end() && it->second->blas)
                    return it->second->blas.get();
                return nullptr;
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
            std::memcpy(proj.elements.data(),
                        camera.projectionMatrix.elements.data(), 64);
            Matrix4 vpUnj;
            vpUnj.multiplyMatrices(proj, view);

            const VkExtent2D ext = ctx->swapchainExtent();
            // Sub-pixel offset in [-0.5, +0.5] per axis. Halton(2,3) is the
            // industry-standard low-discrepancy sequence for primary AA.
            const uint32_t hi = haltonFrame_ + 1u;
            const float jx = halton_(hi, 2) - 0.5f;
            const float jy = halton_(hi, 3) - 0.5f;
            // Map sub-pixel offset to clip-space: one pixel spans 2/width of
            // NDC (NDC ∈ [-1, +1]), so a 1-pixel jitter is 2/width in clip x.
            // Stage 1 has no TAA resolve, so applying jitter here would just
            // shimmer the displayed image frame-to-frame. Gated off until
            // TAA history+resolve lands. Halton counter still advances so
            // we have valid sequence state when TAA wires it up.
            constexpr bool kRasterJitterEnabled = false;
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
            // First frame: self-seed prevVP so motion vectors are zero. The
            // following frame picks up the real history.
            std::memcpy(ubo.prevVP,
                        rasterPrevVPValid_ ? rasterPrevVP_ : vpUnj.elements.data(),
                        64);
            ubo.jitter[0] = jClipX;
            ubo.jitter[1] = jClipY;
            ubo.jitter[2] = 1.f / float(ext.width);
            ubo.jitter[3] = 1.f / float(ext.height);

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), rasterCameraUbos[frame].alloc, &mapped);
            std::memcpy(mapped, &ubo, sizeof(ubo));
            vmaUnmapMemory(ctx->allocator(), rasterCameraUbos[frame].alloc);

            std::memcpy(rasterPrevVP_, vpUnj.elements.data(), 64);
            rasterPrevVPValid_ = true;
            // 16-frame Halton cycle: longer than the eye notices, short
            // enough that any drift in the host counter never overflows.
            haltonFrame_ = (haltonFrame_ + 1u) & 15u;

            // Refresh the per-frame descriptor set: UBO at binding 0, the
            // current motionMat slot at binding 1. motionMatBuffers can grow
            // (handle change) when scene instance count increases — rewriting
            // every frame absorbs that automatically.
            VkDescriptorBufferInfo ubInfo{};
            ubInfo.buffer = rasterCameraUbos[frame].handle;
            ubInfo.offset = 0;
            ubInfo.range  = sizeof(RasterCameraData);

            VkDescriptorBufferInfo mmInfo{};
            mmInfo.buffer = motionMatBuffers[frame].handle;
            mmInfo.offset = 0;
            mmInfo.range  = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[2]{};
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
            vkUpdateDescriptorSets(ctx->device(), 2, writes, 0, nullptr);
        }

        // Begin the raster G-buffer render pass, iterate the visible-entry
        // list, push per-draw model matrix + instanceCustomIndex + flags,
        // bind position+normal+index buffers from the cached BlasRecord and
        // issue an indexed/non-indexed draw. Same iteration order as the
        // TLAS build so the customIndex baked into the push constant matches
        // what the chit/raygen would see.
        void recordRasterGbufPass(VkCommandBuffer cb, uint32_t frame) {
            const VkExtent2D ext = ctx->swapchainExtent();
            const auto& g = rasterGbufs[frame];
            if (g.framebuffer == VK_NULL_HANDLE) return;// not initialized

            VkClearValue clears[5]{};
            clears[0].color = {{0.f, 0.f, 0.f, 0.f}};   // normal — sky/miss as zero
            clears[1].color = {{0.f, 0.f, 0.f, 0.f}};   // motion
            clears[2].color.uint32[0] = 0u;             // instanceID — 0 reserved as sky
            clears[2].color.uint32[1] = 0u;
            clears[2].color.uint32[2] = 0u;
            clears[2].color.uint32[3] = 0u;
            clears[3].color = {{0.f, 0.f, 0.f, 0.f}};   // uv — sky has no UV
            clears[4].depthStencil = {1.f, 0u};

            VkRenderPassBeginInfo rpbi{};
            rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rpbi.renderPass      = rasterGbufRenderPass;
            rpbi.framebuffer     = g.framebuffer;
            rpbi.renderArea.offset = {0, 0};
            rpbi.renderArea.extent = {g.width, g.height};
            rpbi.clearValueCount = 5;
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

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterGbufPipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    rasterPipelineLayout, 0, 1,
                                    &rasterDescSets[frame], 0, nullptr);

            for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
                const auto& en = lastVisibleEntries_[i];
                const BlasRecord* rec = resolveBlasForEntry(en);
                if (!rec || rec->vertex.handle == VK_NULL_HANDLE) continue;

                // Push constants: 64B model matrix + 16B (instId/flags/pad/pad).
                struct PC {
                    float    model[16];
                    uint32_t instanceCustomIndex;
                    uint32_t flags;
                    uint32_t _pad0;
                    uint32_t _pad1;
                } pc{};
                std::memcpy(pc.model, en.worldMatrix.data(), 64);
                pc.instanceCustomIndex = static_cast<uint32_t>(i);
                // flag bits: 0=is_water, 1=transmissive, 2=thinWalled.
                uint32_t flags = 0u;
                if (dynamic_cast<DisplacedMesh*>(en.mesh)) flags |= 1u;
                // For transmissive/thinWalled we'd consult MaterialDesc here
                // — deferred to next pass; raygen also reads matDesc directly.
                pc.flags = flags;
                vkCmdPushConstants(cb, rasterPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT,
                                   0, sizeof(pc), &pc);

                // Vertex inputs: position (binding 0), normal (binding 1),
                // uv (binding 2). Meshes without a UV attribute fall back to
                // the 8-byte dummy buffer of zeros so the gbuffer pipeline
                // keeps a fixed 3-binding layout per draw.
                VkBuffer uvBuf = (rec->uv.handle != VK_NULL_HANDLE)
                                         ? rec->uv.handle
                                         : dummyUvBuffer_.handle;
                VkBuffer     vbufs[3] = {rec->vertex.handle, rec->normal.handle, uvBuf};
                VkDeviceSize voffs[3] = {0, 0, 0};
                vkCmdBindVertexBuffers(cb, 0, 3, vbufs, voffs);

                if (rec->index.handle != VK_NULL_HANDLE) {
                    vkCmdBindIndexBuffer(cb, rec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                    // index count = bytes / sizeof(uint32_t). Matches buildBlasFor.
                    auto* idxAttr = en.mesh->geometry()->getIndex();
                    if (idxAttr) {
                        vkCmdDrawIndexed(cb,
                                         static_cast<uint32_t>(idxAttr->count()),
                                         1, 0, 0, 0);
                    }
                } else {
                    auto* posAttr = en.mesh->geometry()->getAttribute<float>("position");
                    if (posAttr) {
                        vkCmdDraw(cb, static_cast<uint32_t>(posAttr->count()), 1, 0, 0);
                    }
                }
            }

            vkCmdEndRenderPass(cb);
        }

        // Day-1 visualization: blit one of the G-buffer color attachments to
        // the swapchain image so we can verify raster output before wiring
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
            scene.updateMatrixWorld(true);

            GpuLightsUbo ubo{};

            scene.traverse([&](Object3D& o) {
                if (!o.visible) return;
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
        // Mirrors WgpuPathTracer.cpp:3352-3379 — FogExp2.density maps directly
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

        // Phase 7: allocate, transition, and upload an Image2D from a tightly-
        // packed CPU buffer. Pixel layout matches `format`. Caller owns the
        // returned Image2D and must call destroyImage2D() on shutdown.
        Image2D createSampledImage2D(uint32_t w, uint32_t h, VkFormat format,
                                     const void* pixels, VkDeviceSize byteSize,
                                     VkFilter filter, VkSamplerAddressMode addrU,
                                     VkSamplerAddressMode addrV) {
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
            ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

            VkImageMemoryBarrier toDst{};
            toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.image = out.image;
            toDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {w, h, 1};
            vkCmdCopyBufferToImage(cb, staging.handle, out.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier toRead = toDst;
            toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 0, 0, nullptr, 0, nullptr, 1, &toRead);

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
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(env)");

            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = filter;
            sci.minFilter = filter;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = addrU;
            sci.addressModeV = addrV;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.anisotropyEnable = VK_FALSE;
            sci.maxAnisotropy = 1.0f;
            sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sci.unnormalizedCoordinates = VK_FALSE;
            sci.compareEnable = VK_FALSE;
            sci.minLod = 0.0f;
            sci.maxLod = 0.0f;
            check(vkCreateSampler(ctx->device(), &sci, nullptr, &out.sampler),
                  "vkCreateSampler(env)");

            return out;
        }

        // Phase 8: storage-image (rgba32f, GENERAL layout) used as the
        // progressive accumulation target. No staging upload — contents are
        // initialised the first frame after sampleIndex resets to 0. The
        // raygen reads/writes this every frame, so we transition once at
        // creation and keep it in GENERAL forever after.
        Image2D createStorageImage2D(uint32_t w, uint32_t h, VkFormat format) {
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
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT;
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

            return out;
        }

        void createAccumImage() {
            const VkExtent2D ext = ctx->swapchainExtent();
            for (auto& img : accumImagesPP) {
                img = createStorageImage2D(ext.width, ext.height,
                                           VK_FORMAT_R32G32B32A32_SFLOAT);
            }
            for (auto& img : gbufImagesPP) {
                img = createStorageImage2D(ext.width, ext.height,
                                           VK_FORMAT_R32G32B32A32_SFLOAT);
            }
            for (auto& img : filteredImagesPP) {
                img = createStorageImage2D(ext.width, ext.height,
                                           VK_FORMAT_R32G32B32A32_SFLOAT);
            }
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

        // ── Hybrid raster G-buffer prepass implementation ───────────────────
        // Lazy-initialized on first render() with hybridEnabled_ = true.
        // All resources owned by Impl; cleanup in dtor + destroyRasterGbufImages
        // is also called on swapchain resize.

        Image2D createAttachmentImage2D(uint32_t w, uint32_t h, VkFormat format,
                                        VkImageUsageFlags usage,
                                        VkImageAspectFlags aspect) {
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
                destroyImage2D(ctx->allocator(), d, g.depth);
                g.width  = 0;
                g.height = 0;
            }
        }

        void createRasterGbufRenderPass() {
            VkAttachmentDescription attachments[5]{};
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
            // 3: material UV (rgba16f, only rg used). Stage 1A material lookup.
            attachments[3] = attachments[0];
            // 4: depth (d32_sfloat).
            attachments[4]              = attachments[0];
            attachments[4].format       = VK_FORMAT_D32_SFLOAT;
            attachments[4].finalLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRefs[4]{};
            for (uint32_t i = 0; i < 4; ++i) {
                colorRefs[i].attachment = i;
                colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            VkAttachmentReference depthRef{};
            depthRef.attachment = 4;
            depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount    = 4;
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
            deps[1].dstStageMask  = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
            deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            VkRenderPassCreateInfo rpci{};
            rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rpci.attachmentCount = 5;
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
            for (auto& g : rasterGbufs) {
                g.normal = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT);
                g.motion = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT);
                g.ids    = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_UINT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT);
                g.uv     = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT);
                g.depth  = createAttachmentImage2D(w, h, VK_FORMAT_D32_SFLOAT,
                                                   depthUsage, VK_IMAGE_ASPECT_DEPTH_BIT);

                VkImageView views[5] = {g.normal.view, g.motion.view, g.ids.view,
                                        g.uv.view, g.depth.view};
                VkFramebufferCreateInfo fci{};
                fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fci.renderPass      = rasterGbufRenderPass;
                fci.attachmentCount = 5;
                fci.pAttachments    = views;
                fci.width           = w;
                fci.height          = h;
                fci.layers          = 1;
                check(vkCreateFramebuffer(ctx->device(), &fci, nullptr, &g.framebuffer),
                      "vkCreateFramebuffer(rasterGbuf)");
                g.width  = w;
                g.height = h;
            }
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
            // binding 0: per-frame CameraUbo. binding 1: motionMat[] (storage,
            // points at the same VkBuffer raygen has at its own binding 10).
            VkDescriptorSetLayoutBinding bindings[2]{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = 2;
            dlci.pBindings    = bindings;
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &rasterDsLayout),
                  "vkCreateDescriptorSetLayout(raster)");

            VkDescriptorPoolSize sizes[2]{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sizes[0].descriptorCount = kFramesInFlight;
            sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sizes[1].descriptorCount = kFramesInFlight;

            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets       = kFramesInFlight;
            dpci.poolSizeCount = 2;
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
            // normals / uvs are tightly packed (R32G32B32_SFLOAT for pos+normal,
            // R32G32_SFLOAT for uv) in three separate device buffers. The
            // BLAS allocations have VERTEX_BUFFER_BIT so they bind directly.
            // Meshes without a UV attribute bind dummyUvBuffer_ instead — see
            // recordRasterGbufPass.
            VkVertexInputBindingDescription vibs[3]{};
            vibs[0].binding   = 0;
            vibs[0].stride    = 3 * sizeof(float);
            vibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[1].binding   = 1;
            vibs[1].stride    = 3 * sizeof(float);
            vibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[2].binding   = 2;
            vibs[2].stride    = 2 * sizeof(float);
            vibs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription vias[3]{};
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

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 3;
            vi.pVertexBindingDescriptions      = vibs;
            vi.vertexAttributeDescriptionCount = 3;
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
            rs.cullMode    = VK_CULL_MODE_BACK_BIT;
            // threepp / GL-style winding: counter-clockwise faces are front.
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp   = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState cbas[4]{};
            for (auto& a : cbas) {
                a.blendEnable    = VK_FALSE;
                a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            }
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 4;
            cb.pAttachments    = cbas;

            // Dynamic viewport + scissor — set per-recordCommandBuffer to track resize.
            VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
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
            const VkExtent2D ext = ctx->swapchainExtent();
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
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            envIsDefault = true;
            envIsBgColor = false;
            envTextureIdUploaded = 0xFFFFFFFFu;
        }

        // Single shared sampler for every material texture in binding 8.
        // Linear filter + REPEAT addressing matches the typical raster default;
        // per-texture wrap/filter is a v2 concern.
        void createTextureSampler() {
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.anisotropyEnable = VK_FALSE;
            sci.maxAnisotropy = 1.0f;
            sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sci.unnormalizedCoordinates = VK_FALSE;
            sci.compareEnable = VK_FALSE;
            sci.minLod = 0.0f;
            sci.maxLod = 0.0f;
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
                    VK_SAMPLER_ADDRESS_MODE_REPEAT);
            // The Image2D's own sampler is unused (we share textureSampler_),
            // but kept alive so destroyImage2D's clean-up still works.
            materialTextures.push_back(tex);
        }

        // PMREM (Phase 11): set up the compute pipeline that GGX-prefilters
        // each mip level of the env equirect. One pipeline + descriptor pool
        // shared across env uploads; descriptor sets are allocated per-mip
        // each upload (pool is reset between uploads).
        static constexpr uint32_t kMaxEnvMips = 8;
        void createPrefilterPipeline() {
            // Sampler used to read mip 0 during prefilter. REPEAT in u so the
            // longitudinal seam blends; CLAMP in v to avoid pole bleed.
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.minLod = 0.0f;
            sci.maxLod = 0.0f;// only mip 0 read during prefilter
            check(vkCreateSampler(ctx->device(), &sci, nullptr, &prefilterSrcSampler),
                  "vkCreateSampler(prefilter)");

            // Descriptor layout: 0 = sampled mip-0 view, 1 = storage view of target mip.
            std::array<VkDescriptorSetLayoutBinding, 2> b{};
            b[0].binding = 0;
            b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[0].descriptorCount = 1;
            b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            b[1].binding = 1;
            b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[1].descriptorCount = 1;
            b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(b.size());
            dlci.pBindings = b.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &prefilterDsLayout),
                  "vkCreateDescriptorSetLayout(prefilter)");

            // Push constants: alpha (4) + numSamples (4) + 8 byte pad → 16.
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.offset = 0;
            pcRange.size   = 16;

            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &prefilterDsLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &prefilterPipelineLayout),
                  "vkCreatePipelineLayout(prefilter)");

            VkShaderModuleCreateInfo smci{};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = sizeof(kPrefilterEnvCompSpv);
            smci.pCode = kPrefilterEnvCompSpv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &mod),
                  "vkCreateShaderModule(prefilter)");

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage = stage;
            cpci.layout = prefilterPipelineLayout;
            check(vkCreateComputePipelines(ctx->device(), VK_NULL_HANDLE, 1, &cpci, nullptr,
                                           &prefilterPipeline),
                  "vkCreateComputePipelines(prefilter)");
            vkDestroyShaderModule(ctx->device(), mod, nullptr);

            // Descriptor pool: kMaxEnvMips sets, each with one sampled + one storage.
            std::array<VkDescriptorPoolSize, 2> ps{};
            ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[0].descriptorCount = kMaxEnvMips;
            ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ps[1].descriptorCount = kMaxEnvMips;

            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = kMaxEnvMips;
            pci.poolSizeCount = static_cast<uint32_t>(ps.size());
            pci.pPoolSizes = ps.data();
            check(vkCreateDescriptorPool(ctx->device(), &pci, nullptr, &prefilterDescPool),
                  "vkCreateDescriptorPool(prefilter)");
        }

        // Spatial denoiser. One compute pipeline that reads accumImage (binding 7)
        // + gbufImage (binding 12) + cam (binding 2) and writes outImage (binding 1)
        // — all four bindings live in rtDsLayout (with COMPUTE_BIT added to their
        // stageFlags). Push constant carries tonemap + exposure + enable flag.
        void createDenoisePipeline() {
            // Push constants: 4 × u32 = 16 bytes (toneMapping, exposureBits,
            // denoiseEnabled, _pad). COMPUTE-only, separate from rtPipelineLayout's
            // RGEN/CHIT/MISS push constant range.
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.offset = 0;
            pcRange.size   = 16;

            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &rtDsLayout;// reuse the RT descriptor set layout
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &denoisePipelineLayout),
                  "vkCreatePipelineLayout(denoise)");

            VkShaderModuleCreateInfo smci{};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = sizeof(kDenoiseCompSpv);
            smci.pCode    = kDenoiseCompSpv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &mod),
                  "vkCreateShaderModule(denoise)");

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = denoisePipelineLayout;
            check(vkCreateComputePipelines(ctx->device(), VK_NULL_HANDLE, 1, &cpci, nullptr,
                                           &denoisePipeline),
                  "vkCreateComputePipelines(denoise)");
            vkDestroyShaderModule(ctx->device(), mod, nullptr);

            // Atrous pipeline shares the same layout (same descriptor set, same
            // 16-byte COMPUTE push constant range — interpretation differs).
            VkShaderModuleCreateInfo asmci{};
            asmci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            asmci.codeSize = sizeof(kDenoiseAtrousCompSpv);
            asmci.pCode    = kDenoiseAtrousCompSpv;
            VkShaderModule amod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &asmci, nullptr, &amod),
                  "vkCreateShaderModule(denoise_atrous)");

            VkPipelineShaderStageCreateInfo astage{};
            astage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            astage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            astage.module = amod;
            astage.pName  = "main";

            VkComputePipelineCreateInfo acpci{};
            acpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            acpci.stage  = astage;
            acpci.layout = denoisePipelineLayout;
            check(vkCreateComputePipelines(ctx->device(), VK_NULL_HANDLE, 1, &acpci, nullptr,
                                           &denoiseAtrousPipeline),
                  "vkCreateComputePipelines(denoise_atrous)");
            vkDestroyShaderModule(ctx->device(), amod, nullptr);
        }

        // Renderer-level water_displace.comp pipeline. Bindings 0..5 = three
        // (height, displacement) pairs of combined-image-samplers, all RG32F
        // spatial-domain images from the IFFT chain. The push constant carries
        // the BLAS vertex/normal buffer device addresses + grid metadata.
        // One pipeline shared across all DisplacedMesh instances; per-mesh
        // descriptor sets bind the appropriate cascade images.
        void createWaterDisplacePipeline() {
            std::array<VkDescriptorSetLayoutBinding, 6> bb{};
            for (uint32_t i = 0; i < bb.size(); ++i) {
                bb[i].binding = i;
                bb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bb[i].descriptorCount = 1;
                bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = uint32_t(bb.size());
            dlci.pBindings    = bb.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &displaceDsLayout),
                  "vkCreateDescriptorSetLayout(displace)");

            // Push constants — match water_displace.comp's `Pc` struct:
            //   3 × VkDeviceAddress (24) + 9 × u32/float (36) = 60 bytes.
            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset     = 0;
            pcr.size       = 60;

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &displaceDsLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcr;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &displacePipelineLayout),
                  "vkCreatePipelineLayout(displace)");

            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = sizeof(kWaterDisplaceCompSpv);
            smci.pCode    = kWaterDisplaceCompSpv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &mod),
                  "vkCreateShaderModule(displace)");

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = displacePipelineLayout;
            check(vkCreateComputePipelines(ctx->device(), VK_NULL_HANDLE, 1, &cpci, nullptr,
                                           &displacePipeline),
                  "vkCreateComputePipelines(displace)");
            vkDestroyShaderModule(ctx->device(), mod, nullptr);

            // Pool sized for up to 16 DisplacedMesh instances at once. Each
            // takes 6 combined-image-samplers. Bump if scenes ever go beyond.
            constexpr uint32_t kMaxOceans = 16;
            VkDescriptorPoolSize ps{};
            ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps.descriptorCount = 6 * kMaxOceans;
            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets       = kMaxOceans;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes    = &ps;
            check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &displaceDescPool),
                  "vkCreateDescriptorPool(displace)");

            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = VK_FILTER_LINEAR;
            sci.minFilter    = VK_FILTER_LINEAR;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.maxLod       = VK_LOD_CLAMP_NONE;
            check(vkCreateSampler(ctx->device(), &sci, nullptr, &displaceSampler),
                  "vkCreateSampler(displace)");
        }

        // Allocate an env Image2D with a full GGX-prefiltered mip chain. Mip 0
        // holds the source equirect (uploaded from CPU); mips 1..N-1 are filled
        // by dispatching prefilter_env.comp once per mip. Roughness mapping is
        // r = mip / (numMips - 1) — linear in mip index, matching the runtime
        // sample LOD = roughness * (numMips - 1).
        Image2D createPmremEnvImage(uint32_t w, uint32_t h, const float* pixels,
                                    VkDeviceSize byteSize) {
            Image2D out{};
            out.width  = w;
            out.height = h;
            out.format = VK_FORMAT_R32G32B32A32_SFLOAT;

            const uint32_t maxDim = std::max(w, h);
            const uint32_t fullMips = static_cast<uint32_t>(
                    std::floor(std::log2(static_cast<float>(maxDim)))) + 1u;
            out.mipLevels = std::min(fullMips, kMaxEnvMips);

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = out.format;
            ici.extent        = {w, h, 1};
            ici.mipLevels     = out.mipLevels;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_STORAGE_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx->allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
                  "vmaCreateImage(envPmrem)");

            // Staging buffer for mip 0.
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

            // Mip 0: UNDEFINED → TRANSFER_DST for upload.
            {
                VkImageMemoryBarrier br{};
                br.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                br.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                br.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                br.image = out.image;
                br.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                br.subresourceRange.baseMipLevel = 0;
                br.subresourceRange.levelCount = 1;
                br.subresourceRange.layerCount = 1;
                br.srcAccessMask = 0;
                br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cb,
                                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &br);
            }
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {w, h, 1};
            vkCmdCopyBufferToImage(cb, staging.handle, out.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            // Transition: mip 0 (TRANSFER_DST → GENERAL for compute read), mips 1..N-1
            // (UNDEFINED → GENERAL for compute write). GENERAL is universal so reads and
            // writes coexist in the same dispatch.
            {
                std::array<VkImageMemoryBarrier, 2> brs{};
                brs[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                brs[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                brs[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                brs[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                brs[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                brs[0].image = out.image;
                brs[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                brs[0].subresourceRange.baseMipLevel = 0;
                brs[0].subresourceRange.levelCount = 1;
                brs[0].subresourceRange.layerCount = 1;
                brs[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                brs[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                brs[1] = brs[0];
                brs[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                brs[1].subresourceRange.baseMipLevel = 1;
                brs[1].subresourceRange.levelCount = out.mipLevels - 1;
                brs[1].srcAccessMask = 0;
                brs[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

                const uint32_t brCount = (out.mipLevels > 1) ? 2u : 1u;
                vkCmdPipelineBarrier(cb,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 0, nullptr, 0, nullptr, brCount, brs.data());
            }

            // The base view (used for closest_hit sampling) covers all mips.
            {
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = out.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = out.format;
                vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                vci.subresourceRange.baseMipLevel = 0;
                vci.subresourceRange.levelCount = out.mipLevels;
                vci.subresourceRange.layerCount = 1;
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                      "vkCreateImageView(envPmrem)");
            }

            // Sampler with LINEAR mip filtering so trilinear blends across mips.
            // maxLod = mipLevels - 1 lets textureLod sample the full chain.
            {
                VkSamplerCreateInfo sci{};
                sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                sci.magFilter = VK_FILTER_LINEAR;
                sci.minFilter = VK_FILTER_LINEAR;
                sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.minLod = 0.0f;
                sci.maxLod = static_cast<float>(out.mipLevels - 1);
                check(vkCreateSampler(ctx->device(), &sci, nullptr, &out.sampler),
                      "vkCreateSampler(envPmrem)");
            }

            // Run the prefilter compute for each mip > 0. Per-mip storage views
            // are created and destroyed inside the same command-buffer scope.
            std::vector<VkImageView> mipStorageViews;
            mipStorageViews.reserve(out.mipLevels);
            if (out.mipLevels > 1) {
                vkResetDescriptorPool(ctx->device(), prefilterDescPool, 0);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterPipeline);

                for (uint32_t mip = 1; mip < out.mipLevels; ++mip) {
                    VkImageViewCreateInfo vci{};
                    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                    vci.image = out.image;
                    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    vci.format = out.format;
                    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    vci.subresourceRange.baseMipLevel = mip;
                    vci.subresourceRange.levelCount = 1;
                    vci.subresourceRange.layerCount = 1;
                    VkImageView mipView = VK_NULL_HANDLE;
                    check(vkCreateImageView(ctx->device(), &vci, nullptr, &mipView),
                          "vkCreateImageView(prefilter mip)");
                    mipStorageViews.push_back(mipView);

                    VkDescriptorSetAllocateInfo ai{};
                    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    ai.descriptorPool = prefilterDescPool;
                    ai.descriptorSetCount = 1;
                    ai.pSetLayouts = &prefilterDsLayout;
                    VkDescriptorSet set = VK_NULL_HANDLE;
                    check(vkAllocateDescriptorSets(ctx->device(), &ai, &set),
                          "vkAllocateDescriptorSets(prefilter)");

                    VkDescriptorImageInfo srcInfo{};
                    srcInfo.sampler     = prefilterSrcSampler;
                    srcInfo.imageView   = out.view;// full-chain view; sampler reads mip 0
                    srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                    VkDescriptorImageInfo dstInfo{};
                    dstInfo.sampler     = VK_NULL_HANDLE;
                    dstInfo.imageView   = mipView;
                    dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                    std::array<VkWriteDescriptorSet, 2> ws{};
                    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    ws[0].dstSet = set;
                    ws[0].dstBinding = 0;
                    ws[0].descriptorCount = 1;
                    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    ws[0].pImageInfo = &srcInfo;
                    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    ws[1].dstSet = set;
                    ws[1].dstBinding = 1;
                    ws[1].descriptorCount = 1;
                    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    ws[1].pImageInfo = &dstInfo;
                    vkUpdateDescriptorSets(ctx->device(), 2, ws.data(), 0, nullptr);

                    struct Pc {
                        float    alpha;
                        uint32_t numSamples;
                        uint32_t _pad0;
                        uint32_t _pad1;
                    } pc{};
                    const float r = static_cast<float>(mip) /
                                    static_cast<float>(out.mipLevels - 1);
                    pc.alpha      = r * r;// GGX α = roughness²
                    pc.numSamples = 64u;
                    vkCmdPushConstants(cb, prefilterPipelineLayout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                            prefilterPipelineLayout, 0, 1, &set,
                                            0, nullptr);

                    const uint32_t mipW = std::max(1u, w >> mip);
                    const uint32_t mipH = std::max(1u, h >> mip);
                    const uint32_t gx = (mipW + 7u) / 8u;
                    const uint32_t gy = (mipH + 7u) / 8u;
                    vkCmdDispatch(cb, gx, gy, 1u);
                    // No barrier between dispatches: each writes a different mip
                    // and reads only mip 0, no hazard.
                }
            }

            // Final transition: all mips → SHADER_READ_ONLY_OPTIMAL for sampling.
            {
                VkImageMemoryBarrier br{};
                br.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                br.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                br.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                br.image = out.image;
                br.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                br.subresourceRange.baseMipLevel = 0;
                br.subresourceRange.levelCount = out.mipLevels;
                br.subresourceRange.layerCount = 1;
                br.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cb,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                     0, 0, nullptr, 0, nullptr, 1, &br);
            }

            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), staging);
            for (auto v : mipStorageViews) {
                vkDestroyImageView(ctx->device(), v, nullptr);
            }

            return out;
        }

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
            const uint32_t w = img.width;
            const uint32_t h = img.height;
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
            Image2D out = createSampledImage2D(
                    w, h, fmt,
                    rgba.data(), rgba.size(),
                    VK_FILTER_LINEAR,
                    wrapToVk(tex->wrapS),
                    wrapToVk(tex->wrapT));
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
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
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
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            envCdfWidth_    = static_cast<uint32_t>(cdf.width);
            envCdfHeight_   = static_cast<uint32_t>(cdf.height);
            envCdfTotalSum_ = cdf.totalSum;
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
                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
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

            // Phase 7 v1 supports HDR float equirects (RGBELoader output).
            // LDR backgrounds are not yet handled — fall through to default.
            const auto& image = tex->image();
            if (tex->type != Type::Float) {
                std::cerr << "[VulkanRenderer] env texture is not float HDR; "
                             "ignoring (Phase 7 v1 supports HDR equirect only)" << std::endl;
                return false;
            }
            const auto& src = const_cast<Image&>(image).data<float>();
            const uint32_t w = image.width;
            const uint32_t h = image.height;
            if (w == 0 || h == 0 || src.size() < 4u * w * h) {
                std::cerr << "[VulkanRenderer] env texture has unexpected size; ignoring" << std::endl;
                return false;
            }

            vkDeviceWaitIdle(ctx->device());
            destroyImage2D(ctx->allocator(), ctx->device(), envImage);

            // Phase 11: GGX-prefilter the env into a mip chain so the closest_hit
            // shader can sample at a roughness-derived LOD instead of the cheap
            // (1-r)² fade. Mip 0 is the source mirror, mip k is convolved with
            // GGX(α=(k/(N-1))²). closest_hit fades from mirror to fully diffuse
            // by walking the chain via textureLod.
            envImage = createPmremEnvImage(
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
            // 4=material SSBO, 5=lights UBO, 6=env equirect (Phase 7),
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
            std::array<VkDescriptorSetLayoutBinding, 27> bindings{};
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
            bindings[4].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
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
            // Compute denoise reads accumImage radiance + per-pixel FC.
            bindings[7].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_COMPUTE_BIT;
            // Bindless material albedo array. Fixed-cap kMaxMaterialTextures
            // so we can avoid VK_EXT_descriptor_indexing's variable-descriptor-
            // count plumbing for v1; closest_hit indexes via mdesc.albedoTexIndex.
            bindings[8].binding = 8;
            bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[8].descriptorCount = kMaxMaterialTextures;
            bindings[8].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            // Continuous-motion bindings.
            bindings[9].binding = 9;
            bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[9].descriptorCount = 1;
            bindings[9].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
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
            bindings[13].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
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
            bindings[18].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                      VK_SHADER_STAGE_MISS_BIT_KHR;
            // Binding 19 — env marginal luminance CDF (R32F, 1×h). Picks a row
            // (latitude) before sampling the conditional row at that latitude.
            bindings[19].binding = 19;
            bindings[19].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[19].descriptorCount = 1;
            bindings[19].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                      VK_SHADER_STAGE_MISS_BIT_KHR;
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

            // Bindings 22-25 — hybrid raster G-buffer attachments. Sampled by
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

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(bindings.size());
            dlci.pBindings = bindings.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &rtDsLayout),
                  "vkCreateDescriptorSetLayout(RT)");

            // 48-byte push constant. Layout matches the host-side `pc[12]`
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
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                 VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                 VK_SHADER_STAGE_MISS_BIT_KHR;
            pcRange.offset = 0;
            pcRange.size   = 48;

            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &rtDsLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &rtPipelineLayout),
                  "vkCreatePipelineLayout(RT)");

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

            VkShaderModule rgenMod    = loadModule(kRaygenRgenSpv,           sizeof(kRaygenRgenSpv));
            VkShaderModule missMod    = loadModule(kMissRmissSpv,            sizeof(kMissRmissSpv));
            VkShaderModule sMissMod   = loadModule(kShadowMissRmissSpv,      sizeof(kShadowMissRmissSpv));
            VkShaderModule chitMod    = loadModule(kClosestHitRchitSpv,      sizeof(kClosestHitRchitSpv));
            VkShaderModule ahitMod    = loadModule(kClosestHitAlphaRahitSpv, sizeof(kClosestHitAlphaRahitSpv));
            VkShaderModule sahitMod   = loadModule(kShadowAnyhitRahitSpv,    sizeof(kShadowAnyhitRahitSpv));

            // Stages: 0=rgen, 1=primary miss, 2=shadow miss, 3=path closest-hit,
            //         4=path any-hit (alpha), 5=shadow any-hit (glass+cutout).
            std::array<VkPipelineShaderStageCreateInfo, 6> stages{};
            for (auto& s : stages) {
                s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                s.pName = "main";
            }
            stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            stages[0].module = rgenMod;
            stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
            stages[1].module = missMod;
            stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
            stages[2].module = sMissMod;
            stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            stages[3].module = chitMod;
            stages[4].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            stages[4].module = ahitMod;
            stages[5].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            stages[5].module = sahitMod;

            // Groups:
            //   0 = rgen
            //   1 = primary miss
            //   2 = shadow miss
            //   3 = path hit  (chit=3, ahit=4) — selected by sbtOffset=0
            //   4 = shadow hit (chit=UNUSED, ahit=5) — selected by sbtOffset=1
            //       NoOpaqueEXT on shadow rays forces all geometry through ahit=5;
            //       opaque hits set shadow=0 and accept; glass multiplies and ignores.
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
            groups[4].anyHitShader = 5;// closestHitShader stays UNUSED

            VkRayTracingPipelineCreateInfoKHR rci{};
            rci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
            rci.stageCount = static_cast<uint32_t>(stages.size());
            rci.pStages = stages.data();
            rci.groupCount = static_cast<uint32_t>(groups.size());
            rci.pGroups = groups.data();
            rci.maxPipelineRayRecursionDepth = 2;// primary + 1 shadow
            rci.layout = rtPipelineLayout;

            check(ctx->rt().createRayTracingPipelines(
                          ctx->device(), VK_NULL_HANDLE, VK_NULL_HANDLE,
                          1, &rci, nullptr, &rtPipeline),
                  "vkCreateRayTracingPipelinesKHR");

            vkDestroyShaderModule(ctx->device(), rgenMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), missMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), sMissMod, nullptr);
            vkDestroyShaderModule(ctx->device(), chitMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), ahitMod,  nullptr);
            vkDestroyShaderModule(ctx->device(), sahitMod, nullptr);
        }

        void createShaderBindingTable() {
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
                          ctx->device(), rtPipeline, 0, groupCount,
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

            sbtBuffer = createBuffer(
                    ctx->allocator(), ctx->device(), sbtSize,
                    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), sbtBuffer.alloc, &mapped);
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
            vmaUnmapMemory(ctx->allocator(), sbtBuffer.alloc);

            const VkDeviceAddress base = sbtBuffer.address;
            rgenRegion.deviceAddress = base;
            rgenRegion.stride = rgenRegionBytes;
            rgenRegion.size   = rgenRegionBytes;
            missRegion.deviceAddress = base + rgenRegionBytes;
            missRegion.stride = handleSizeAligned;
            missRegion.size   = missRegionBytes;
            hitRegion.deviceAddress = base + rgenRegionBytes + missRegionBytes;
            hitRegion.stride = handleSizeAligned;
            hitRegion.size   = hitRegionBytes;
            callRegion = {};
        }

        void createPhotonBuffers() {
            photonCountBuf = createBuffer(
                    ctx->allocator(), ctx->device(),
                    static_cast<VkDeviceSize>(kPhotonGridSize) * sizeof(uint32_t),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    0);
            // 3 vec3 per slot: position, flux, direction.
            photonDataBuf = createBuffer(
                    ctx->allocator(), ctx->device(),
                    static_cast<VkDeviceSize>(kPhotonGridSize) * kPhotonsPerCell * 3 * 12,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    0);
        }

        void createPhotonEmitPipeline() {
            auto loadModule = [this](const uint32_t* code, size_t size) {
                VkShaderModuleCreateInfo smci{};
                smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                smci.codeSize = size;
                smci.pCode    = code;
                VkShaderModule m = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &m),
                      "vkCreateShaderModule(photon)");
                return m;
            };
            VkShaderModule rgenMod = loadModule(kPhotonEmitRgenSpv, sizeof(kPhotonEmitRgenSpv));
            VkShaderModule missMod = loadModule(kPhotonMissRmissSpv, sizeof(kPhotonMissRmissSpv));
            VkShaderModule chitMod = loadModule(kPhotonChitRchitSpv, sizeof(kPhotonChitRchitSpv));

            // Groups: 0=rgen, 1=miss, 2=photon closest-hit.
            std::array<VkPipelineShaderStageCreateInfo, 3> stg{};
            for (auto& s : stg) { s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; s.pName = "main"; }
            stg[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;      stg[0].module = rgenMod;
            stg[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;        stg[1].module = missMod;
            stg[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; stg[2].module = chitMod;

            std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> grp{};
            for (auto& g : grp) {
                g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                g.generalShader = g.closestHitShader = g.anyHitShader = g.intersectionShader = VK_SHADER_UNUSED_KHR;
            }
            grp[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;          grp[0].generalShader = 0;
            grp[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;          grp[1].generalShader = 1;
            grp[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; grp[2].closestHitShader = 2;

            VkRayTracingPipelineCreateInfoKHR rci{};
            rci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
            rci.stageCount = static_cast<uint32_t>(stg.size());
            rci.pStages    = stg.data();
            rci.groupCount = static_cast<uint32_t>(grp.size());
            rci.pGroups    = grp.data();
            rci.maxPipelineRayRecursionDepth = 1; // loop in raygen, no nested traces
            rci.layout = rtPipelineLayout; // same layout as path trace pipeline

            check(ctx->rt().createRayTracingPipelines(
                          ctx->device(), VK_NULL_HANDLE, VK_NULL_HANDLE,
                          1, &rci, nullptr, &photonEmitPipeline),
                  "vkCreateRayTracingPipelinesKHR(photon)");

            vkDestroyShaderModule(ctx->device(), rgenMod, nullptr);
            vkDestroyShaderModule(ctx->device(), missMod, nullptr);
            vkDestroyShaderModule(ctx->device(), chitMod, nullptr);
        }

        void createPhotonSbt() {
            const auto& props = ctx->rtPipelineProperties();
            const uint32_t handleSize        = props.shaderGroupHandleSize;
            const uint32_t handleAlignment   = props.shaderGroupHandleAlignment;
            const uint32_t baseAlignment     = props.shaderGroupBaseAlignment;
            const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

            constexpr uint32_t groupCount = 3; // rgen, miss, hit
            const uint32_t handlesSize = groupCount * handleSize;
            std::vector<uint8_t> handles(handlesSize);
            check(ctx->rt().getRayTracingShaderGroupHandles(
                          ctx->device(), photonEmitPipeline, 0, groupCount,
                          handlesSize, handles.data()),
                  "vkGetRayTracingShaderGroupHandlesKHR(photon)");

            const uint32_t rgenBytes = alignUp(handleSizeAligned, baseAlignment);
            const uint32_t missBytes = alignUp(handleSizeAligned, baseAlignment);
            const uint32_t hitBytes  = alignUp(handleSizeAligned, baseAlignment);
            const VkDeviceSize sbtSize =
                    static_cast<VkDeviceSize>(rgenBytes) +
                    static_cast<VkDeviceSize>(missBytes) +
                    static_cast<VkDeviceSize>(hitBytes);

            photonSbtBuf = createBuffer(
                    ctx->allocator(), ctx->device(), sbtSize,
                    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), photonSbtBuf.alloc, &mapped);
            std::memset(mapped, 0, sbtSize);
            uint8_t* dst = static_cast<uint8_t*>(mapped);
            std::memcpy(dst,                        handles.data() + 0 * handleSize, handleSize);
            std::memcpy(dst + rgenBytes,             handles.data() + 1 * handleSize, handleSize);
            std::memcpy(dst + rgenBytes + missBytes, handles.data() + 2 * handleSize, handleSize);
            vmaUnmapMemory(ctx->allocator(), photonSbtBuf.alloc);

            const VkDeviceAddress base = photonSbtBuf.address;
            photonRgenRgn.deviceAddress = base;
            photonRgenRgn.stride = rgenBytes;
            photonRgenRgn.size   = rgenBytes;
            photonMissRgn.deviceAddress = base + rgenBytes;
            photonMissRgn.stride = handleSizeAligned;
            photonMissRgn.size   = missBytes;
            photonHitRgn.deviceAddress = base + rgenBytes + missBytes;
            photonHitRgn.stride = handleSizeAligned;
            photonHitRgn.size   = hitBytes;
            photonCallRgn = {};
        }

        void createDescriptorPool() {
            imageCount_ = static_cast<uint32_t>(ctx->swapchainImages().size());
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::array<VkDescriptorPoolSize, 5> ps{};
            ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            ps[0].descriptorCount = totalSets;
            ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ps[1].descriptorCount = totalSets * 7;// bindings 1, 7, 11, 12, 13 + 20×2 (filtered ping-pong)
            ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps[2].descriptorCount = totalSets * 4;// bindings 2 (camera), 5 (lights), 9 (prevCamera), 17 (fog)
            ps[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps[3].descriptorCount = totalSets * 7;// bindings 3,4,10,14 (existing) + 15,16 (photon) + 21 (meshMovedBits)
            ps[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[4].descriptorCount = totalSets * (3 + kMaxMaterialTextures + 5);// binding 6 (env) + binding 8 (material array) + bindings 18,19 (env CDF) + bindings 22-26 (hybrid gbuffer attachments incl. UV)

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
                    matInfo.buffer = materialDescsBuffer.handle;
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
                    photonCntInfo.buffer = photonCountBuf.handle;
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
                    photonDataInfo.buffer = photonDataBuf.handle;
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
                    filteredInfos[0].imageView   = filteredImagesPP[0].view;
                    filteredInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    filteredInfos[1].imageView   = filteredImagesPP[1].view;
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

                    // Bindings 22-25: hybrid gbuffer attachments. raygen at
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

                    std::array<VkWriteDescriptorSet, 27> writes{
                            wAS, wImg, wUbo, wGeom, wMat, wLights, wEnv, wAccum, wMatTex,
                            wPrevCam, wMotion, wPrevAccum, wGbuf, wPrevGbuf, wEmTri,
                            wPhotonCnt, wPhotonData, wFog, wEnvCdf, wEnvMarg, wFiltered,
                            wMeshMoved,
                            wGbufNormal, wGbufMotion, wGbufDepth, wGbufIds, wGbufUv};
                    vkUpdateDescriptorSets(ctx->device(),
                                           static_cast<uint32_t>(writes.size()),
                                           writes.data(), 0, nullptr);
                }
            }
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

            VkDescriptorBufferInfo matInfo{};
            matInfo.buffer = materialDescsBuffer.handle;
            matInfo.offset = 0;
            matInfo.range = VK_WHOLE_SIZE;

            std::array<VkDescriptorImageInfo, kMaxMaterialTextures> matTexInfos{};
            fillMaterialTextureInfos(matTexInfos);

            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(totalSets * 4);
            for (uint32_t i = 0; i < totalSets; ++i) {
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
                wMat.pBufferInfo = &matInfo;
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
        }

        // Phase 7: rewrite only binding 6 across every descriptor set (called
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
        }

        void recreateSwapchainAndDescriptors() {
            ctx->recreateSwapchain();
            for (auto& img : accumImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : gbufImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : filteredImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            createAccumImage();// resets sampleIndex + clears prevWorldMats
            // Resize hybrid raster attachments BEFORE descriptor rewrites —
            // bindings 22-25 point at rasterGbufs[f].*.view, so stale views
            // from the old extent need to be replaced before the new
            // descriptor sets capture them.
            ensureHybridResources();
            vkDestroyDescriptorPool(ctx->device(), descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
            descriptorSets.clear();
            createDescriptorPool();
            allocateAndUpdateDescriptors();
            size = WindowSize{static_cast<int>(ctx->swapchainExtent().width),
                              static_cast<int>(ctx->swapchainExtent().height)};
        }

        void recordCommandBuffer(VkCommandBuffer cb, uint32_t imageIndex) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");

            // ── Hybrid raster G-buffer pass ─────────────────────────────────
            // Runs ahead of any RT work so the gbuffer is ready when raygen
            // wants to read primary visibility. In Day-1 debug mode we blit
            // a chosen channel directly to the swapchain, draw the ImGui
            // overlay on top (mirrors the PT path's overlay flow), then
            // present — bypassing the entire RT pipeline.
            if (hybridEnabled_ && rasterGbufPipeline != VK_NULL_HANDLE) {
                recordRasterGbufPass(cb, currentFrame);
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
                    return;
                }
            }

            const VkImage img = ctx->swapchainImages()[imageIndex];

            // UNDEFINED -> GENERAL for ray-gen storage write.
            VkImageMemoryBarrier2 toGeneral{};
            toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            toGeneral.srcAccessMask = 0;
            toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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

            std::array<VkImageMemoryBarrier2, 5> preBarriers{};
            preBarriers[0] = toGeneral;
            preBarriers[1] = accumGbufTemplate; preBarriers[1].image = accumImagesPP[0].image;
            preBarriers[2] = accumGbufTemplate; preBarriers[2].image = accumImagesPP[1].image;
            preBarriers[3] = accumGbufTemplate; preBarriers[3].image = gbufImagesPP[0].image;
            preBarriers[4] = accumGbufTemplate; preBarriers[4].image = gbufImagesPP[1].image;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size());
            dep.pImageMemoryBarriers = preBarriers.data();
            vkCmdPipelineBarrier2(cb, &dep);

            // descriptorSets index is shared by photon and primary RT pipelines.
            const uint32_t setIdx = currentFrame * imageCount_ + imageIndex;

            // ── Photon emit pass ────────────────────────────────────────────────
            // Skipped when no material has transmission > 0: caustics only form
            // through glass, and gatherCaustics in closest_hit is gated on the
            // same flag, so no shader will read these buffers either way.
            if (sceneHasGlass_) {
            // 1. Zero per-cell photon counters (counts only; data is overwritten
            //    in place, old overflow slots beyond the new count are never read).
            vkCmdFillBuffer(cb, photonCountBuf.handle, 0, VK_WHOLE_SIZE, 0u);

            // 2. Barrier: fill → raygen read/write in photon emit shader.
            {
                VkBufferMemoryBarrier2 fillDone{};
                fillDone.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                fillDone.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                fillDone.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                fillDone.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                fillDone.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                fillDone.srcQueueFamilyIndex = fillDone.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                fillDone.buffer = photonCountBuf.handle;
                fillDone.size   = VK_WHOLE_SIZE;

                VkDependencyInfo fillDep{};
                fillDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                fillDep.bufferMemoryBarrierCount = 1;
                fillDep.pBufferMemoryBarriers = &fillDone;
                vkCmdPipelineBarrier2(cb, &fillDep);
            }

            // 3. Photon emit dispatch.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, photonEmitPipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    rtPipelineLayout, 0, 1,
                                    &descriptorSets[setIdx], 0, nullptr);
            {
                const float exposure = toneMappingExposure_;
                uint32_t exposureBits;
                std::memcpy(&exposureBits, &exposure, sizeof(exposureBits));
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
                const uint32_t ppc[12] = {
                        sampleIndex, envImage.mipLevels,
                        static_cast<uint32_t>(toneMapping_), exposureBits,
                        motionFlagsPhoton,
                        emissiveTriCountThisFrame_, emPowerBits,
                        samplesPerPixel_,
                        envCdfWidth_, envCdfHeight_, envSumBitsPhoton,
                        fireflyBitsPhoton,
                };
                vkCmdPushConstants(cb, rtPipelineLayout,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                           VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(ppc), ppc);
            }
            ctx->rt().cmdTraceRays(cb, &photonRgenRgn, &photonMissRgn, &photonHitRgn, &photonCallRgn,
                                   kPhotonEmitDim, kPhotonEmitDim, 1);

            // 4. Barrier: photon writes → closest_hit reads (caustic gather).
            {
                std::array<VkBufferMemoryBarrier2, 2> photonDone{};
                for (auto& b : photonDone) {
                    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                    b.srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                    b.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.size = VK_WHOLE_SIZE;
                }
                photonDone[0].buffer = photonCountBuf.handle;
                photonDone[1].buffer = photonDataBuf.handle;

                VkDependencyInfo photonDep{};
                photonDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                photonDep.bufferMemoryBarrierCount = 2;
                photonDep.pBufferMemoryBarriers = photonDone.data();
                vkCmdPipelineBarrier2(cb, &photonDep);
            }
            }// if (sceneHasGlass_)
            // ── End photon emit ─────────────────────────────────────────────────

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    rtPipelineLayout, 0, 1,
                                    &descriptorSets[setIdx], 0, nullptr);

            // [0] sampleIndex (raygen), [1] PMREM mip count (closest_hit),
            // [2] ToneMapping enum, [3] exposure as float bits,
            // [4] motionFlags (bit 0 = any motion, bit 1 = camera moved,
            //                  bit 2 = scene has any glass material,
            //                  bit 3 = hybrid mode (gbuffer drives reproject
            //                          + primary jitter is disabled)),
            // [5] emissiveCount, [6] emissiveTotalPower (float bits).
            // Per-instance moved bits live in the binding 21 SSBO.
            const float exposure = toneMappingExposure_;
            uint32_t exposureBits;
            std::memcpy(&exposureBits, &exposure, sizeof(exposureBits));
            const uint32_t motionFlags =
                    (motionThisFrame_      ? 1u : 0u) |
                    (cameraMovedThisFrame_ ? 2u : 0u) |
                    (sceneHasGlass_        ? 4u : 0u) |
                    (hybridEnabled_        ? 8u : 0u);
            uint32_t emPowerBits;
            std::memcpy(&emPowerBits, &emissiveTotalPowerThisFrame_, sizeof(emPowerBits));
            uint32_t envSumBits;
            std::memcpy(&envSumBits, &envCdfTotalSum_, sizeof(envSumBits));
            uint32_t fireflyBits;
            std::memcpy(&fireflyBits, &fireflyClamp_, sizeof(fireflyBits));
            const uint32_t pc[12] = {
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
            };
            vkCmdPushConstants(cb, rtPipelineLayout,
                               VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                       VK_SHADER_STAGE_MISS_BIT_KHR,
                               0, sizeof(pc), pc);

            const VkExtent2D ext = ctx->swapchainExtent();
            ctx->rt().cmdTraceRays(cb, &rgenRegion, &missRegion, &hitRegion, &callRegion,
                                   ext.width, ext.height, 1);

            // ── Spatial denoiser: 3-pass à-trous + finalize tonemap + sRGB ──────
            // RT writes accumImage + gbufImage; denoise pipeline reads them.
            // outImage is owned by the finalize pass (raygen.rgen tail was
            // stripped). All ping-pong slots stay GENERAL throughout.
            {
                const uint32_t gx = (ext.width  + 7u) / 8u;
                const uint32_t gy = (ext.height + 7u) / 8u;

                // Bind once — both pipelines share denoisePipelineLayout.
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        denoisePipelineLayout, 0, 1,
                                        &descriptorSets[setIdx], 0, nullptr);

                // Initial barrier: RT_SHADER write → COMPUTE_SHADER read+write.
                auto barrierMem = [&](VkPipelineStageFlags2 srcStage) {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = srcStage;
                    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo bd{};
                    bd.sType                = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    bd.memoryBarrierCount   = 1;
                    bd.pMemoryBarriers      = &mb;
                    vkCmdPipelineBarrier2(cb, &bd);
                };

                barrierMem(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

                if (denoiseEnabled_) {
                    // 2 atrous passes: stride 1 → 2. Pass 0 sources from
                    // accumImage and writes filt[1]; pass 1 reads filt[1]
                    // and writes filt[0]. Finalize then reads filt[0].
                    //
                    // Initial multi-pass land used 3 passes (strides 1/2/4)
                    // with a 21×21 effective radius. On this fast-converging
                    // PT (60-200 FPS, 2 spp → settled in <1s) the third pass
                    // overblurred clean input. 2 passes give a 9×9 radius
                    // which keeps detail without leaving residual noise.
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, denoiseAtrousPipeline);

                    struct AtrousPc {
                        uint32_t stride;
                        uint32_t readFromAccum;
                        uint32_t inputIdx;
                        uint32_t outputIdx;
                    };
                    // Schedule lands the final filtered output in filt[0] so
                    // finalize.comp's hard-coded `filteredArray[0]` read is
                    // correct.
                    const AtrousPc passes[2] = {
                            {1u, 1u, 0u, 1u}, // accum → filt[1]
                            {2u, 0u, 1u, 0u}, // filt[1] → filt[0]
                    };
                    for (int i = 0; i < 2; ++i) {
                        if (i > 0) barrierMem(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                        vkCmdPushConstants(cb, denoisePipelineLayout,
                                           VK_SHADER_STAGE_COMPUTE_BIT,
                                           0, sizeof(passes[i]), &passes[i]);
                        vkCmdDispatch(cb, gx, gy, 1);
                    }

                    // Atrous → finalize barrier (filt[0] write → finalize read).
                    barrierMem(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                }

                // Finalize: tonemap + sRGB → outImage. When denoiseEnabled_ ==
                // false, this is the only compute dispatch in the block; the
                // initial RT-to-COMPUTE barrier above covers it.
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, denoisePipeline);
                const uint32_t finalizePc[4] = {
                        static_cast<uint32_t>(toneMapping_),
                        exposureBits,
                        denoiseEnabled_ ? 1u : 0u,
                        0u,
                };
                vkCmdPushConstants(cb, denoisePipelineLayout,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(finalizePc), finalizePc);
                vkCmdDispatch(cb, gx, gy, 1);
            }
            // ── End denoise ─────────────────────────────────────────────────────

            // If an overlay (ImGui) is registered, draw it on top of the
            // ray-traced image inside a dynamic render pass before presenting.
            if (overlayCallback) {
                VkImageMemoryBarrier2 toColor{};
                toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                // outImage was last written by denoise.comp, not raygen.
                toColor.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                toColor.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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
                // outImage was last written by denoise.comp, not raygen.
                toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                toPresent.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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

            check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        }

        void renderFrame(Object3D& scene, Camera& camera) {
            VkDevice d = ctx->device();
            vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);

            uint32_t imageIndex = 0;
            VkResult acq = vkAcquireNextImageKHR(d, ctx->swapchain(), UINT64_MAX,
                                                 imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchainAndDescriptors();
                return;
            }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                check(acq, "vkAcquireNextImageKHR");
            }

            updateCameraUbo(currentFrame, camera);
            updateLightsUbo(currentFrame, scene);
            updateFogUbo(currentFrame, scene);
            // Safe to write motionMatBuffers[currentFrame] now that the
            // inFlight[currentFrame] fence has been signaled — the GPU has
            // finished its previous use of this slot.
            computeAndUploadMotionMatrices(currentFrame, lastVisibleEntries_);
            // Hybrid raster prepass: lazy-create resources on first use,
            // refresh attachments on resize, then upload the per-frame
            // camera VPs (curr jittered, curr unjittered, prev unjittered).
            // Must run after computeAndUploadMotionMatrices so the descriptor
            // rewrite picks up the populated motionMat buffer for this frame.
            if (hybridEnabled_) {
                ensureHybridResources();
                uploadRasterCameraUbo(currentFrame, camera);
            }
            uploadMeshMovedBits(currentFrame);
            // Same fence guarantees emissiveTriBuffers[currentFrame] is no
            // longer in use; rebuild the per-frame CDF and rewrite binding 14
            // if the buffer grew.
            if (buildAndUploadEmissiveTris(currentFrame, lastVisibleEntries_)) {
                rewriteEmissiveTriDescriptors(currentFrame);
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
            recordCommandBuffer(cmdBuffers[currentFrame], imageIndex);

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
            cbInfo.commandBuffer = cmdBuffers[currentFrame];

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

            // Phase 8: cap to keep 1/(N+1) blend in float precision and avoid
            // overflow in the (prev*N + curr) scaling. ~65k samples is well
            // beyond visual convergence for direct lighting.
            if (sampleIndex < 65535u) ++sampleIndex;

            currentFrame = (currentFrame + 1) % kFramesInFlight;
        }
    };

    VulkanRenderer::VulkanRenderer(Canvas& canvas) {
        canvas.initWindow(GraphicsAPI::Vulkan);
        pimpl_ = std::make_unique<Impl>(canvas);
    }

    VulkanRenderer::~VulkanRenderer() = default;

    void VulkanRenderer::render(Object3D& scene, Camera& camera) {
        const auto cur = pimpl_->canvas.size();
        if (cur.width() != pimpl_->size.width() || cur.height() != pimpl_->size.height()) {
            pimpl_->needsResize = true;
        }
        // Mirror Renderer-base tone-mapping state into the Impl so renderFrame
        // can push it as a single 16-byte block. Done every render() so users
        // can flip toneMapping / toneMappingExposure freely between frames.
        pimpl_->toneMapping_         = toneMapping;
        pimpl_->toneMappingExposure_ = toneMappingExposure;
        pimpl_->ensureSceneBuilt(scene);
        pimpl_->renderFrame(scene, camera);
    }

    WindowSize VulkanRenderer::size() const { return pimpl_->size; }

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

    std::vector<unsigned char> VulkanRenderer::readRGBPixels() { return {}; }

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

    void VulkanRenderer::setSamplesPerPixel(int spp) {
        const uint32_t v = static_cast<uint32_t>(std::max(1, spp));
        pimpl_->samplesPerPixel_ = v;
    }

    int VulkanRenderer::samplesPerPixel() const {
        return static_cast<int>(pimpl_->samplesPerPixel_);
    }

    void VulkanRenderer::setDenoise(bool enabled) {
        pimpl_->denoiseEnabled_ = enabled;
    }

    bool VulkanRenderer::denoise() const {
        return pimpl_->denoiseEnabled_;
    }

    void VulkanRenderer::setFireflyClamp(float cap) {
        pimpl_->fireflyClamp_ = (cap <= 0.0f) ? 1e30f : cap;
    }

    float VulkanRenderer::fireflyClamp() const {
        const float v = pimpl_->fireflyClamp_;
        return (v > 1e20f) ? 0.0f : v;
    }

    void VulkanRenderer::setHybridEnabled(bool enabled) {
        pimpl_->hybridEnabled_ = enabled;
    }

    bool VulkanRenderer::hybridEnabled() const {
        return pimpl_->hybridEnabled_;
    }

    void VulkanRenderer::setHybridDebugView(int view) {
        using V = Impl::HybridDebugView;
        switch (view) {
            case 1:  pimpl_->hybridDebugView_ = V::Normal; break;
            case 2:  pimpl_->hybridDebugView_ = V::Motion; break;
            case 3:  pimpl_->hybridDebugView_ = V::Ids;    break;
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

}// namespace threepp
