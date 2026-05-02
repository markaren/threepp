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

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include "threepp/renderers/vulkan/shaders/raygen.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit.rchit.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit_alpha.rahit.spv.h"
#include "threepp/renderers/vulkan/shaders/prefilter_env.comp.spv.h"

#include <GLFW/glfw3.h>

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
            VkDeviceAddress address = 0;
            // Liveness tag: detects dangling-pointer reuse when a BufferGeometry
            // is destroyed (model unloaded) and the C++ allocator hands the
            // same address to a different geometry. Pruned in ensureSceneBuilt.
            std::weak_ptr<BufferGeometry> liveCheck;
        };
        std::unordered_map<const BufferGeometry*, std::unique_ptr<BlasRecord>> blasCache;

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
            uint32_t indexed;
            uint32_t _pad;
        };
        struct MaterialDesc {
            float albedo[3];
            float roughness;
            float metalness;
            float emissive[3];
            float emissiveIntensity;
            int32_t albedoTexIndex;   // -1 == no albedo texture, else slot in bindless array
            int32_t roughnessTexIndex;// -1 == no roughness texture; sampled .g (glTF convention)
            int32_t metalnessTexIndex;// -1 == no metalness texture; sampled .b (glTF convention)
            int32_t normalTexIndex;   // -1 == no normal map; tangent-space, glTF convention
            float normalScale[2];     // x/y scale applied to sampled (xy) before re-deriving z
            float alphaCutoff;        // <= 0 == disabled; otherwise any-hit ignores hits with sampled alpha < cutoff
            float transmission;       // 0..1 probability the bounce takes the transmission lobe (glass)
            float ior;                // index of refraction; 1.5 default for glass
            int32_t transmissionTexIndex;// -1 == none; sampled .r (glTF KHR_materials_transmission)
            float clearcoat;          // 0..1 layer intensity; ccF0 fixed at 0.04 (dielectric IOR ~1.5)
            float clearcoatRoughness; // 0..1; clamped to [0.04, 1] in shader to keep VNDF non-singular
            int32_t clearcoatTexIndex;          // -1 == none; sampled .r (glTF KHR_materials_clearcoat)
            int32_t clearcoatRoughnessTexIndex; // -1 == none; sampled .g
            float attenuationColor[3]; // Beer-Lambert tint (default {1,1,1} = clear)
            float attenuationDistance; // 0 = no Beer-Lambert; world-space mean-free path
            int32_t emissiveTexIndex;  // -1 == no emissive map; sampled .rgb (sRGB)
            float specularIntensity;   // KHR_materials_specular: default 1.0
            float specularColor[3];    // KHR_materials_specular: default {1,1,1}
            float sheenColor[3];       // KHR_materials_sheen: default {0,0,0}
            float sheenRoughness;      // KHR_materials_sheen: default 0.0
            int32_t doubleSided;       // 1 = shade both faces; 0 = pass through back-face hits
            float uvTransform[9];           // column-major Matrix3 for albedo channel
            int32_t occlusionTexIndex;      // -1 = none; sampled .r (glTF occlusion)
            float uvTransformNormal[9];     // for normalTexIndex
            float uvTransformRoughMetal[9]; // for roughness + metalness textures
            float uvTransformEmissive[9];   // for emissiveTexIndex
            float uvTransformOcclusion[9];  // for occlusionTexIndex
            float uvTransformClearcoat[9];  // for clearcoatTexIndex
            float uvTransformClearcoatRough[9]; // for clearcoatRoughnessTexIndex
            float uvTransformTransmission[9];   // for transmissionTexIndex
        };
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
        static constexpr uint32_t kMaxMaterialTextures = 512;
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
        uint32_t accumWriteIdx_ = 0;
        uint32_t sampleIndex = 0;
        std::array<float, 32> prevCameraData{};
        bool prevCameraValid = false;

        // Set when this frame's camera viewProj or any mesh transform differs
        // from the previous frame. Forwarded to raygen via a push-constant bit
        // so the shader only does reprojection math when motion actually
        // occurred — static frames take the self-tap path and accumulate
        // bit-for-bit with no float-precision pixel-snap drift (the cause of
        // the visible static-scene wobble we observed when always reprojecting).
        bool motionThisFrame_ = false;

        // Previous-frame camera (proj_prev * view_prev) for primary-hit
        // reprojection. One UBO per frame-in-flight so updates don't race the
        // GPU. Per-mesh motion matrices (prevWorld * inverse(curWorld)) live in
        // a single SSBO indexed by gl_InstanceCustomIndexEXT; the host repacks
        // the array each frame from prevWorldMats keyed by Mesh*. First-frame
        // / first-seen-mesh entries are identity so reproject is a no-op.
        std::array<Buffer, kFramesInFlight> prevCameraUbos{};
        std::array<Buffer, kFramesInFlight> motionMatBuffers{};
        std::array<VkDeviceSize, kFramesInFlight> motionMatBufferCapacity{};
        std::unordered_map<const Mesh*, std::array<float, 16>> prevWorldMats;

        // Per-mesh fingerprint used to detect scene changes between frames.
        // We capture the surface state ray tracing actually consumes:
        //   - mesh / geometry / material identity (covers add/remove/swap)
        //   - matrixWorld 16 floats (covers transform animation)
        //   - PBR scalars (covers material slider tweaks)
        // Anything not in this set won't trigger a rebuild; sufficient for
        // the v1 feature set (no skinning, no per-vertex anim).
        struct MeshFingerprint {
            const void* mesh;
            const void* geom;
            const void* mat;
            const void* albedoTex;            // covers map swap on the same material
            const void* roughnessTex;         // covers roughnessMap swap
            const void* metalnessTex;         // covers metalnessMap swap
            const void* normalTex;            // covers normalMap swap
            const void* transmissionTex;      // covers transmissionMap swap
            const void* clearcoatTex;         // covers clearcoatMap swap
            const void* clearcoatRoughnessTex;// covers clearcoatRoughnessMap swap
            const void* emissiveTex;          // covers emissiveMap swap
            std::array<float, 16> matrix{};
            std::array<float, 15> pbr{};// + normalScale.xy + transmission/ior + clearcoat/roughness
        };
        std::vector<MeshFingerprint> prevSceneFingerprint;
        // Mesh* in TLAS-instance order from the last ensureSceneBuilt call.
        // renderFrame consumes this to compute per-instance motion matrices
        // after the in-flight fence has been waited (safe to write the
        // motionMatBuffers[currentFrame] HOST_VISIBLE buffer).
        std::vector<Mesh*> lastVisibleMeshes_;
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
            createPrefilterPipeline();// must precede createDefaultEnvImage so PMREM is ready
            createDefaultEnvImage();
            createAccumImage();
            createTextureSampler();
            createDefaultMaterialTexture();
            createRtPipeline();
            createShaderBindingTable();
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
            }
            blasCache.clear();

            for (auto& b : cameraUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : prevCameraUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : lightsUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : motionMatBuffers) destroyBuffer(ctx->allocator(), b);
            destroyImage2D(ctx->allocator(), d, envImage);
            for (auto& img : accumImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : gbufImagesPP) destroyImage2D(ctx->allocator(), d, img);
            for (auto& img : materialTextures) destroyImage2D(ctx->allocator(), d, img);
            materialTextures.clear();
            if (textureSampler_) vkDestroySampler(d, textureSampler_, nullptr);

            if (prefilterPipeline)        vkDestroyPipeline(d, prefilterPipeline, nullptr);
            if (prefilterPipelineLayout)  vkDestroyPipelineLayout(d, prefilterPipelineLayout, nullptr);
            if (prefilterDsLayout)        vkDestroyDescriptorSetLayout(d, prefilterDsLayout, nullptr);
            if (prefilterDescPool)        vkDestroyDescriptorPool(d, prefilterDescPool, nullptr);
            if (prefilterSrcSampler)      vkDestroySampler(d, prefilterSrcSampler, nullptr);
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

            const VkBufferUsageFlags geomUsage =
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

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
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
            d.doubleSided = (mat->side == Side::Double) ? 1 : 0;
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

            std::vector<Mesh*> meshes;
            scene.traverse([&](Object3D& o) {
                auto* m = dynamic_cast<Mesh*>(&o);
                if (!m || !m->visible) return;
                auto geom = m->geometry();
                if (!geom || !geom->hasAttribute("position")) return;
                if (!geom->hasAttribute("normal")) return;
                meshes.push_back(m);
            });

            std::vector<MeshFingerprint> currFp;
            currFp.reserve(meshes.size());
            for (Mesh* m : meshes) {
                MeshFingerprint fp{};
                fp.mesh = m;
                fp.geom = m->geometry().get();
                fp.mat  = m->material().get();
                fp.albedoTex             = albedoTexOf(*m).get();
                fp.roughnessTex          = roughnessTexOf(*m).get();
                fp.metalnessTex          = metalnessTexOf(*m).get();
                fp.normalTex             = normalTexOf(*m).get();
                fp.transmissionTex       = transmissionTexOf(*m).get();
                fp.clearcoatTex          = clearcoatTexOf(*m).get();
                fp.clearcoatRoughnessTex = clearcoatRoughnessTexOf(*m).get();
                fp.emissiveTex           = emissiveTexOf(*m).get();
                const auto& e = m->matrixWorld->elements;
                for (size_t i = 0; i < 16; ++i) fp.matrix[i] = e[i];
                const MaterialDesc md = materialFromMesh(*m);
                fp.pbr = {md.albedo[0], md.albedo[1], md.albedo[2],
                          md.roughness, md.metalness,
                          md.emissive[0], md.emissive[1], md.emissive[2],
                          md.emissiveIntensity,
                          md.normalScale[0], md.normalScale[1],
                          md.transmission, md.ior,
                          md.clearcoat, md.clearcoatRoughness};
                currFp.push_back(fp);
            }

            // Continuous-motion fast path: when only the per-mesh matrices
            // changed (everything else — topology, materials, textures —
            // matches), refit the TLAS in place and let raygen reproject.
            // We only have to detect the matrix-only case ahead of time;
            // motion matrices themselves are computed each frame in
            // renderFrame so we can defer the host write past the fence wait.
            lastVisibleMeshes_ = meshes;
            if (sceneBuilt_ && currFp.size() == prevSceneFingerprint.size()) {
                bool topologySame = true;
                bool matricesSame = true;
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
                        std::memcmp(a.pbr.data(), b.pbr.data(), sizeof(a.pbr)) != 0) {
                        topologySame = false;
                        break;
                    }
                    if (std::memcmp(a.matrix.data(), b.matrix.data(), sizeof(a.matrix)) != 0) {
                        matricesSame = false;
                    }
                }
                if (topologySame) {
                    if (!matricesSame) {
                        motionThisFrame_ = true;
                        sampleIndex = 0;
                        // Transform-only update: refit TLAS with new transforms.
                        // BLAS handles + buffer addresses are unchanged so
                        // geomDescs / matDescs stay valid; we just rewrite the
                        // tlasInstancesBuffer in place and call MODE_UPDATE.
                        std::vector<VkAccelerationStructureInstanceKHR> instances;
                        instances.reserve(meshes.size());
                        for (Mesh* m : meshes) {
                            const BufferGeometry* geomKey = m->geometry().get();
                            auto it = blasCache.find(geomKey);
                            if (it == blasCache.end()) continue;// shouldn't happen on transform-only
                            VkAccelerationStructureInstanceKHR inst{};
                            const auto& e = m->matrixWorld->elements;
                            for (int r = 0; r < 3; ++r) {
                                for (int c = 0; c < 4; ++c) {
                                    inst.transform.matrix[r][c] = e[c * 4 + r];
                                }
                            }
                            inst.instanceCustomIndex = static_cast<uint32_t>(instances.size());
                            inst.mask = 0xFFu;
                            inst.instanceShaderBindingTableRecordOffset = 0;
                            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                            inst.accelerationStructureReference = it->second->address;
                            instances.push_back(inst);
                        }
                        refitTlas(instances);
                    }
                    // Update prevSceneFingerprint so later frames compare
                    // against this frame's matrices, not stale ones.
                    prevSceneFingerprint = std::move(currFp);
                    return;
                }
            }

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
                        it = blasCache.erase(it);
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
            instances.reserve(meshes.size());
            geomDescs.reserve(meshes.size());
            matDescs.reserve(meshes.size());

            for (Mesh* m : meshes) {
                const BufferGeometry* geomKey = m->geometry().get();

                auto it = blasCache.find(geomKey);
                if (it == blasCache.end()) {
                    auto rec = buildBlasFor(*m->geometry());
                    if (!rec) continue;// degenerate / unsupported geometry
                    rec->liveCheck = m->geometry();
                    it = blasCache.emplace(geomKey, std::move(rec)).first;
                }

                VkAccelerationStructureInstanceKHR inst{};
                // VkTransformMatrixKHR is row-major 3x4; threepp Matrix4 is
                // column-major 4x4 (elements[c*4 + r]).
                const auto& e = m->matrixWorld->elements;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        inst.transform.matrix[r][c] = e[c * 4 + r];
                    }
                }
                inst.instanceCustomIndex = static_cast<uint32_t>(geomDescs.size());
                inst.mask = 0xFFu;
                inst.instanceShaderBindingTableRecordOffset = 0;
                inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                inst.accelerationStructureReference = it->second->address;
                instances.push_back(inst);

                GeometryDesc gdesc{};
                gdesc.vertexAddress = it->second->vertex.address;
                gdesc.normalAddress = it->second->normal.address;
                gdesc.indexAddress  = it->second->index.address;
                gdesc.uvAddress     = it->second->uv.address;// 0 if no UV attribute
                gdesc.indexed = it->second->index.handle != VK_NULL_HANDLE ? 1u : 0u;
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

            // Grow motion-mat buffers if the new instance count exceeds the
            // current capacity. The descriptor write below (or the initial
            // allocate, on first build) will pick up the new buffer handle.
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                ensureMotionMatCapacity(f, std::max<uint32_t>(instanceCount, 1u));
            }

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
                        /*size*/ 16 * sizeof(float),// viewProjPrev (single mat4)
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
        // meshes (cold-start frame after a topology rebuild) so the reproject
        // is a no-op until prevWorldMats picks up real history. Caller must
        // have already waited the inFlight[frame] fence — we write a buffer
        // the GPU may have been reading on the previous use of `frame`.
        void computeAndUploadMotionMatrices(uint32_t frame,
                                            const std::vector<Mesh*>& meshes) {
            const uint32_t count = static_cast<uint32_t>(meshes.size());
            if (count == 0) return;

            // Worst case all motion matrices need writing — `count` mat4s.
            std::vector<float> data(count * 16);
            for (uint32_t i = 0; i < count; ++i) {
                Matrix4 cur;
                std::memcpy(cur.elements.data(),
                            meshes[i]->matrixWorld->elements.data(), 64);

                Matrix4 motion;// identity by default
                auto it = prevWorldMats.find(meshes[i]);
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
            // Only update entries we just wrote; meshes removed from the
            // scene are pruned naturally (full-rebuild path clears the map).
            for (uint32_t i = 0; i < count; ++i) {
                std::array<float, 16> m{};
                std::memcpy(m.data(), meshes[i]->matrixWorld->elements.data(), 64);
                prevWorldMats[meshes[i]] = m;
            }
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

        void updateCameraUbo(uint32_t frame, Camera& camera) {
            camera.updateMatrixWorld(true);

            float data[32];
            std::memcpy(data + 0,  camera.matrixWorld->elements.data(),            64);
            std::memcpy(data + 16, camera.projectionMatrixInverse.elements.data(), 64);

            // Camera-motion detection: epsilon on position (data[12..14]) and
            // forward column (data[8..10]) instead of memcmp so orbit-control
            // floating-point recompute doesn't fire false motion each frame.
            if (prevCameraValid) {
                const float dx = data[12] - prevCameraData[12];
                const float dy = data[13] - prevCameraData[13];
                const float dz = data[14] - prevCameraData[14];
                const float fx = data[8]  - prevCameraData[8];
                const float fy = data[9]  - prevCameraData[9];
                const float fz = data[10] - prevCameraData[10];
                if (dx*dx + dy*dy + dz*dz > 1e-6f || fx*fx + fy*fy + fz*fz > 1e-8f) {
                    motionThisFrame_ = true;
                    sampleIndex = 0;
                }
            }

            // Continuous-motion: previous-frame view-projection drives the
            // primary-hit reprojection in raygen. On the very first frame we
            // seed prev=current so the reproject is identity and history
            // starts fresh without an artificial reset. We no longer touch
            // sampleIndex on camera move — per-pixel FC packed in accum.w
            // takes over that role and survives motion.
            Matrix4 viewProj = camera.projectionMatrix;
            if (prevCameraValid) {
                Matrix4 prevView;// inverse(prevWorld) = view_prev
                std::memcpy(prevView.elements.data(), prevCameraData.data(), 64);
                prevView.invert();
                Matrix4 prevProj;
                Matrix4 prevProjInv;
                std::memcpy(prevProjInv.elements.data(), prevCameraData.data() + 16, 64);
                prevProj.copy(prevProjInv).invert();
                viewProj.copy(prevProj).multiply(prevView);
            } else {
                viewProj.copy(camera.projectionMatrix).multiply(camera.matrixWorldInverse);
            }
            void* mappedPrev = nullptr;
            vmaMapMemory(ctx->allocator(), prevCameraUbos[frame].alloc, &mappedPrev);
            std::memcpy(mappedPrev, viewProj.elements.data(), 64);
            vmaUnmapMemory(ctx->allocator(), prevCameraUbos[frame].alloc);

            std::memcpy(prevCameraData.data(), data, sizeof(data));
            prevCameraValid = true;

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), cameraUbos[frame].alloc, &mapped);
            std::memcpy(mapped, data, sizeof(data));
            vmaUnmapMemory(ctx->allocator(), cameraUbos[frame].alloc);
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

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), lightsUbos[frame].alloc, &mapped);
            std::memcpy(mapped, &ubo, sizeof(ubo));
            vmaUnmapMemory(ctx->allocator(), lightsUbos[frame].alloc);
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
            const Image& img = const_cast<Texture*>(tex)->image();
            const uint32_t w = img.width;
            const uint32_t h = img.height;
            if (w == 0 || h == 0) return -1;

            // Force RGBA8 — both UNORM and SRGB need 4 channels in this engine.
            // stb_image's default load already produces RGBA, so we expect a
            // tightly packed 4*w*h byte buffer.
            std::vector<unsigned char> rgba;
            try {
                const auto& src = const_cast<Image&>(img).data<unsigned char>();
                if (src.size() == w * h * 4) {
                    rgba.assign(src.begin(), src.end());
                } else if (src.size() == w * h * 3) {
                    rgba.resize(w * h * 4);
                    for (uint32_t i = 0; i < w * h; ++i) {
                        rgba[i * 4 + 0] = src[i * 3 + 0];
                        rgba[i * 4 + 1] = src[i * 3 + 1];
                        rgba[i * 4 + 2] = src[i * 3 + 2];
                        rgba[i * 4 + 3] = 255;
                    }
                } else {
                    std::cerr << "[VulkanRenderer] unsupported pixel layout for material tex ("
                              << src.size() << " bytes for " << w << "x" << h << ")\n";
                    return -1;
                }
            } catch (const std::bad_variant_access&) {
                std::cerr << "[VulkanRenderer] float-pixel material textures not yet supported\n";
                return -1;
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
                    return true;
                }
                if (envIsDefault) return false;
                vkDeviceWaitIdle(ctx->device());
                destroyImage2D(ctx->allocator(), ctx->device(), envImage);
                createDefaultEnvImage();
                envIsBgColor = false;
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
            return true;
        }

        void createRtPipeline() {
            // 0=TLAS, 1=storage image (swapchain), 2=camera UBO, 3=geometry SSBO,
            // 4=material SSBO, 5=lights UBO, 6=env equirect (Phase 7),
            // 7=accum write image (ping-pong), 8=material texture array,
            // 9=prev camera UBO (motion), 10=motion matrix SSBO (motion),
            // 11=prev accum read image (ping-pong), 12=gbuf write image
            // (ping-pong), 13=prev gbuf read image (ping-pong).
            std::array<VkDescriptorSetLayoutBinding, 14> bindings{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;// shadow rays
            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[2].binding = 2;
            bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
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
            bindings[5].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[6].binding = 6;
            bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[6].descriptorCount = 1;
            bindings[6].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR |
                                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bindings[7].binding = 7;
            bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[7].descriptorCount = 1;
            bindings[7].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
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
            bindings[12].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[13].binding = 13;
            bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[13].descriptorCount = 1;
            bindings[13].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(bindings.size());
            dlci.pBindings = bindings.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &rtDsLayout),
                  "vkCreateDescriptorSetLayout(RT)");

            // 20-byte push constant: .x sampleIndex, .y env mip count
            // (closest_hit), .z toneMapping enum, .w exposure as float bits,
            // .v[1] motionFlags (raygen): bit 0 = "motion happened this frame"
            // (camera viewProj or any mesh transform differs from prev frame).
            // Raygen uses this to default to a self-tap of accum/gbuf when
            // unset, avoiding round-trip reproject precision drift on static
            // scenes. Layout fits a uvec4 + uint, well under the 128-byte
            // minimum push-constant guarantee.
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            pcRange.offset = 0;
            pcRange.size   = 20;

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

            VkShaderModule rgenMod  = loadModule(kRaygenRgenSpv,            sizeof(kRaygenRgenSpv));
            VkShaderModule missMod  = loadModule(kMissRmissSpv,             sizeof(kMissRmissSpv));
            VkShaderModule sMissMod = loadModule(kShadowMissRmissSpv,       sizeof(kShadowMissRmissSpv));
            VkShaderModule chitMod  = loadModule(kClosestHitRchitSpv,       sizeof(kClosestHitRchitSpv));
            VkShaderModule ahitMod  = loadModule(kClosestHitAlphaRahitSpv,  sizeof(kClosestHitAlphaRahitSpv));

            // Stages: 0=rgen, 1=primary miss, 2=shadow miss, 3=closest hit, 4=any-hit.
            std::array<VkPipelineShaderStageCreateInfo, 5> stages{};
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

            // Groups: 0=rgen, 1=primary miss, 2=shadow miss, 3=hit (chit+ahit).
            // The any-hit attaches to the same hit group as closest_hit so the
            // SBT layout (one hit record) doesn't change. Phase 10: alpha-test.
            std::array<VkRayTracingShaderGroupCreateInfoKHR, 4> groups{};
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
        }

        void createShaderBindingTable() {
            const auto& props = ctx->rtPipelineProperties();
            const uint32_t handleSize = props.shaderGroupHandleSize;
            const uint32_t handleAlignment = props.shaderGroupHandleAlignment;
            const uint32_t baseAlignment = props.shaderGroupBaseAlignment;
            const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

            // 4 groups: rgen, primary miss, shadow miss, hit.
            // Miss region holds 2 records; rgen and hit hold 1 each.
            constexpr uint32_t groupCount = 4;
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
            const uint32_t hitRegionBytes  = alignUp(handleSizeAligned, baseAlignment);
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
            // hit follows the miss region.
            std::memcpy(dst + rgenRegionBytes + missRegionBytes,
                        handles.data() + 3 * handleSize, handleSize);
            vmaUnmapMemory(ctx->allocator(), sbtBuffer.alloc);

            const VkDeviceAddress base = sbtBuffer.address;
            rgenRegion.deviceAddress = base;
            rgenRegion.stride = rgenRegionBytes;
            rgenRegion.size   = rgenRegionBytes;
            missRegion.deviceAddress = base + rgenRegionBytes;
            missRegion.stride = handleSizeAligned;
            missRegion.size   = missRegionBytes;
            hitRegion.deviceAddress = base + rgenRegionBytes + missRegionBytes;
            hitRegion.stride = hitRegionBytes;
            hitRegion.size   = hitRegionBytes;
            callRegion = {};
        }

        void createDescriptorPool() {
            imageCount_ = static_cast<uint32_t>(ctx->swapchainImages().size());
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::array<VkDescriptorPoolSize, 5> ps{};
            ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            ps[0].descriptorCount = totalSets;
            ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ps[1].descriptorCount = totalSets * 5;// bindings 1, 7, 11, 12, 13
            ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps[2].descriptorCount = totalSets * 3;// bindings 2 (camera), 5 (lights), 9 (prevCamera)
            ps[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps[3].descriptorCount = totalSets * 3;// bindings 3 (geometry), 4 (material), 10 (motionMat)
            ps[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[4].descriptorCount = totalSets * (1 + kMaxMaterialTextures);// binding 6 (env) + binding 8 (material array)

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

                    std::array<VkWriteDescriptorSet, 14> writes{
                            wAS, wImg, wUbo, wGeom, wMat, wLights, wEnv, wAccum, wMatTex,
                            wPrevCam, wMotion, wPrevAccum, wGbuf, wPrevGbuf};
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
            std::vector<VkDescriptorImageInfo> infos(totalSets);
            std::vector<VkWriteDescriptorSet>  writes(totalSets);
            for (uint32_t i = 0; i < totalSets; ++i) {
                infos[i].sampler     = envImage.sampler;
                infos[i].imageView   = envImage.view;
                infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = descriptorSets[i];
                writes[i].dstBinding = 6;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo = &infos[i];
            }
            vkUpdateDescriptorSets(ctx->device(),
                                   static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }

        void recreateSwapchainAndDescriptors() {
            ctx->recreateSwapchain();
            for (auto& img : accumImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : gbufImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            createAccumImage();// resets sampleIndex + clears prevWorldMats
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

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);
            const uint32_t setIdx = currentFrame * imageCount_ + imageIndex;
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    rtPipelineLayout, 0, 1,
                                    &descriptorSets[setIdx], 0, nullptr);

            // .x = sampleIndex (raygen), .y = PMREM mip count (closest_hit),
            // .z = ToneMapping enum, .w = exposure as float bits, .v[1] =
            // motionFlags (bit 0 = camera or mesh motion this frame).
            const float exposure = toneMappingExposure_;
            uint32_t exposureBits;
            std::memcpy(&exposureBits, &exposure, sizeof(exposureBits));
            const uint32_t motionFlags = motionThisFrame_ ? 1u : 0u;
            const uint32_t pc[5] = {
                    sampleIndex,
                    envImage.mipLevels,
                    static_cast<uint32_t>(toneMapping_),
                    exposureBits,
                    motionFlags,
            };
            vkCmdPushConstants(cb, rtPipelineLayout,
                               VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                               0, sizeof(pc), pc);

            const VkExtent2D ext = ctx->swapchainExtent();
            ctx->rt().cmdTraceRays(cb, &rgenRegion, &missRegion, &hitRegion, &callRegion,
                                   ext.width, ext.height, 1);

            // If an overlay (ImGui) is registered, draw it on top of the
            // ray-traced image inside a dynamic render pass before presenting.
            if (overlayCallback) {
                VkImageMemoryBarrier2 toColor{};
                toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                toColor.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
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
                toPresent.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
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
            // Safe to write motionMatBuffers[currentFrame] now that the
            // inFlight[currentFrame] fence has been signaled — the GPU has
            // finished its previous use of this slot.
            computeAndUploadMotionMatrices(currentFrame, lastVisibleMeshes_);
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

}// namespace threepp
