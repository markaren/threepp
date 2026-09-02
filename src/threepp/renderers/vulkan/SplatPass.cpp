#include "threepp/renderers/vulkan/SplatPass.hpp"

#include "threepp/extras/DataUtils.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/vulkan/GpuTimings.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/splat_bake_resolve.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_bake_scatter.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_checksum.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_expand.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_indirect.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_project.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_radix_hist.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_radix_scatter.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_range.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_raster.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_scan.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_scan_add.comp.spv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace threepp::vulkan {

    namespace {

        // KEEP IN SYNC with splat_common.glsl. A mismatch here is not a
        // compile error, it is a wrong picture — the shader would index a
        // struct the host packed differently.
        constexpr uint32_t kTileW = 16, kTileH = 16;
        constexpr uint32_t kScanBlock  = 1024;// 256 threads x 4
        constexpr uint32_t kScanThreads = 256;
        constexpr uint32_t kRadixBlock = 512;// 128 threads x 4
        constexpr uint32_t kRadixBins  = 16;
        constexpr uint32_t kRadixPasses = 8; // 8 x 4 bits = the whole 32-bit key
        constexpr uint32_t kProjStride = 64; // sizeof(SplatProj)
        constexpr uint32_t kGeomStride = 40; // vec3 + float + float[6], scalar layout
        constexpr uint32_t kGlobalWords = 12;
        constexpr uint32_t kBindings   = 24;// 22 = the indirect dispatch args, 23 = the depth AOV
        constexpr uint32_t kMaxRanges  = 64;// KEEP IN SYNC with splat_common.glsl

        constexpr uint32_t kSplatFlagOrtho     = 1u;
        constexpr uint32_t kSplatFlagDebugNaN  = 2u;
        constexpr uint32_t kSplatFlagDepthTest = 4u;
        constexpr uint32_t kSplatFlagMotion    = 8u;
        constexpr uint32_t kSplatFlagFog       = 16u;
        constexpr uint32_t kSplatFlagChecksum  = 32u;
        constexpr uint32_t kSplatFlagBgSolid   = 64u;
        constexpr uint32_t kSplatFlagDepthAov  = 128u;
        constexpr uint32_t kSplatFlagDepthMed  = 256u;

        struct SplatPc {
            uint32_t count, srcOff, dstOff, sumOff;
            uint32_t arg0, arg1, arg2, arg3;
        };

        // ── Volume bake (plans/splat-volume-reflections.md, Part 1) ──────────
        // Longest axis 128 voxels, the others proportional and even, floored at
        // 16 so a thin scan still has a usable third dimension. 128^3 rgba16f
        // is 16 MB, and kMaxClouds bounds the worst case at 128 MB — against
        // clouds that themselves cost ~240 B/splat, proportionally small and
        // resident exactly as long as the cloud is.
        // The DEFAULT longest-axis budget; THREEPP_VK_SPLATVOL_RES moves it
        // (SplatPass ctor), capped at kVolResCap because the bake scratch is
        // 16 B/voxel transient — 268 MB at 256^3, which an 8 GB card absorbs
        // inside an upload and 512^3's 2.1 GB would not.
        constexpr uint32_t kVolMaxResDefault = 128;
        constexpr uint32_t kVolResCap        = 256;
        constexpr uint32_t kVolMinRes        = 16;
        // The extent estimator's sample size — the SAME fixed-stride ~8192 the
        // collector uses for the sort interval (VulkanCoreScene.cpp:2893), and
        // for the same reason: both ends have to agree on what "the subject" is
        // or the box and the depth buckets would describe different clouds.
        constexpr size_t   kVolSampleTarget = 8192;
        constexpr VkFormat kVolFormat       = VK_FORMAT_R16G16B16A16_SFLOAT;
        constexpr uint32_t kVolTexelBytes   = 8;
        constexpr uint32_t kVolCounters     = 4;// sigma, sigma*r, sigma*g, sigma*b
        constexpr uint32_t kBakeBindings    = 4;// geom, sh, scratch, out image

        // Mirrors SplatBakePc in splat_bake_common.glsl (scalar layout: vec3
        // packs to 12 bytes at 4-byte alignment, so the block is 60).
        struct SplatBakePc {
            float    boxMin[3];
            float    boxSize[3];
            float    voxelSize[3];
            uint32_t res[3];
            uint32_t splatCount;
            uint32_t shCoeffs;
            float    voxelVolume;
        };
        static_assert(sizeof(SplatBakePc) == 60,
                      "SplatBakePc must match splat_bake_common.glsl's push-constant block");

        // A 3D image, built straight off the context the way SplatPass builds
        // its sampler and every one of its buffers. ParticleFieldPass takes a
        // CreateImage3DFn from the renderer instead, because it needs the
        // renderer's retire-on-generation machinery; this pass owns its images
        // outright and frees them with the cloud, so injection would buy a
        // constructor parameter and nothing else.
        Image2D createVolumeImage(VulkanContext& ctx, uint32_t w, uint32_t h, uint32_t d,
                                  const char* name) {
            Image2D out{};
            out.width  = w;
            out.height = h;
            out.format = kVolFormat;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_3D;
            ici.format        = kVolFormat;
            ici.extent        = {w, h, d};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            // STORAGE for the resolve's imageStore, SAMPLED for the marches
            // that consume it, TRANSFER_SRC for the test-only hash readback.
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx.allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
                  "vmaCreateImage(splat volume)");

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = out.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
            vci.format   = kVolFormat;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx.device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(splat volume)");
            ctx.setObjectName(out.image, name);
            ctx.setObjectName(out.view, name);
            return out;
        }

        // Mirrors SplatUbo in splat_common.glsl (scalar layout: no vec3
        // padding surprises, but mat4 is still 64 B and vec2/vec4 keep their
        // natural alignment — the layout below is written to match exactly).
        struct SplatUboData {
            float modelView[16];
            float proj[16];
            float projInv[16];
            float model[16];
            float prevVPfromView[16];
            float camWorld[16];
            float camPosWs[4];
            float camFwdWs[4];
            float viewport[2];
            float focal[2];
            float percentile[2];
            float jitterClip[2];
            float nearPlane;
            float preExposure;
            float pointMix;
            float pointSigma;
            uint32_t splatCount;
            uint32_t shCoeffs;
            uint32_t shDegree;
            uint32_t tilesX;
            uint32_t tilesY;
            uint32_t tileBits;
            uint32_t depthBits;
            uint32_t budget;
            uint32_t flags;
            uint32_t envMipCount;
            uint32_t rangeCount;
            uint32_t ranges[kMaxRanges * 2];// (first compact index, source base)
        };

        uint32_t divUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

        // Number of bits the tile id needs at the TOP of the sort key. The
        // depth gets everything else, so the key always fills 32 bits and the
        // depth quantisation is as fine as the screen allows.
        uint32_t tileBitsFor(uint32_t tiles) {
            uint32_t b = 1;
            while ((1u << b) < tiles && b < 24) ++b;
            return b;
        }

        // Words scanScratch needs to scan an n-element array through all its
        // levels (each level's block sums live in their own region).
        uint32_t scanScratchWords(uint32_t n) {
            uint32_t total = 0, cur = std::max(n, 1u);
            while (true) {
                const uint32_t nb = divUp(cur, kScanBlock);
                total += nb;
                if (nb == 1) break;
                cur = nb;
            }
            return total;
        }

        VkPipeline makeComputePipe(VkDevice d, VkPipelineCache cache, VkPipelineLayout layout,
                                   const uint32_t* spv, size_t spvBytes, const char* label) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = spvBytes;
            smci.pCode    = spv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(d, &smci, nullptr, &mod), label);

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = layout;
            VkPipeline pipe = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(d, cache, 1, &cpci, nullptr, &pipe), label);
            vkDestroyShaderModule(d, mod, nullptr);
            return pipe;
        }

    }// namespace


    SplatPass::SplatPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {

        // Read ONCE, before anything is created: with the volume off there is
        // to be no image, no pipeline and no dispatch anywhere in the pass, so
        // a frame is byte-exact what it was before the bake existed. A knob
        // sampled per upload could not promise that across one run.
        if (const char* e = std::getenv("THREEPP_VK_SPLATVOL_OFF"); e && *e && *e != '0')
            volumeOff_ = true;

        // The resolution budget, read once for the OFF knob's reason: a knob
        // sampled per upload would let two clouds in one run bake under
        // different budgets and hash differently for reasons no test could
        // name. Clamped, not trusted — see kVolResCap for what 512 would cost.
        volMaxRes_ = kVolMaxResDefault;
        if (const char* e = std::getenv("THREEPP_VK_SPLATVOL_RES"); e && *e) {
            const long v = std::strtol(e, nullptr, 10);
            if (v > 0)
                volMaxRes_ = std::clamp(static_cast<uint32_t>(v), kVolMinRes, kVolResCap);
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
        const VkDeviceSize align = std::max<VkDeviceSize>(
                props.limits.minUniformBufferOffsetAlignment, 16);
        uboStride_ = ((sizeof(SplatUboData) + align - 1) / align) * align;

        createPipelines();
        createDescriptorPool();

        uboBuf_.resize(framesInFlight_);
        debugBuf_.resize(framesInFlight_);
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            // One region per (target, cloud): two views recording in the same
            // frame write DIFFERENT matrices for the same cloud, host-side,
            // before either dispatch executes — sharing a slot would hand both
            // of them whichever view recorded last.
            uboBuf_[f] = createBuffer(ctx_.allocator(), ctx_.device(),
                                      uboStride_ * kMaxClouds * kMaxTargets,
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                              VMA_ALLOCATION_CREATE_MAPPED_BIT);
            debugBuf_[f] = createBuffer(ctx_.allocator(), ctx_.device(),
                                        kGlobalWords * sizeof(uint32_t),
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }

        globalBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                  kGlobalWords * sizeof(uint32_t),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(globalBuf_.handle, "splat.globals");

        // Two VkDispatchIndirectCommands = 6 uints. STORAGE so the sizing
        // kernel can write it, INDIRECT so the dispatches can read it.
        indirectBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                    6 * sizeof(uint32_t),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(indirectBuf_.handle, "splat.indirectArgs");
    }

    SplatPass::~SplatPass() {
        VkDevice d = ctx_.device();
        VmaAllocator a = ctx_.allocator();
        for (auto& kv : resident_) destroyCloudResources(*kv.second);
        resident_.clear();
        for (auto* b : {&projBuf_, &countBuf_, &offsetBuf_, &keyA_, &valA_, &keyB_, &valB_,
                        &rangeBuf_, &globalBuf_, &histBuf_, &scanBuf_, &indirectBuf_})
            destroyBuffer(a, *b);
        for (auto& b : uboBuf_) destroyBuffer(a, b);
        for (auto& b : debugBuf_) destroyBuffer(a, b);

        for (VkPipeline p : {projectPipe_, scanPipe_, scanAddPipe_, expandPipe_, indirectPipe_,
                             histPipe_, scatterPipe_, rangePipe_, rasterPipe_, checksumPipe_,
                             bakeScatterPipe_, bakeResolvePipe_})
            if (p) vkDestroyPipeline(d, p, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (bakePipeLayout_) vkDestroyPipelineLayout(d, bakePipeLayout_, nullptr);
        if (bakeDsLayout_)   vkDestroyDescriptorSetLayout(d, bakeDsLayout_, nullptr);
        if (bakeDescPool_)   vkDestroyDescriptorPool(d, bakeDescPool_, nullptr);
        if (sampler_)    vkDestroySampler(d, sampler_, nullptr);
    }

    void SplatPass::destroyCloudResources(Cloud& c) {
        destroyBuffer(ctx_.allocator(), c.geom);
        destroyBuffer(ctx_.allocator(), c.sh);
        if (c.volume.image != VK_NULL_HANDLE) {
            destroyImage2D(ctx_.allocator(), ctx_.device(), c.volume);
            c.volRes[0] = c.volRes[1] = c.volRes[2] = 0;
            // The SET of baked volumes just changed, so every consumer holding
            // this view in a descriptor has to rewrite before it binds again.
            ++volumeGen_;
        }
    }

    void SplatPass::createPipelines() {
        VkDevice d = ctx_.device();

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(splat)");

        std::array<VkDescriptorSetLayoutBinding, kBindings> bnd{};
        for (uint32_t i = 0; i < kBindings; ++i) {
            bnd[i].binding         = i;
            bnd[i].descriptorCount = 1;
            bnd[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bnd[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bnd[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bnd[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bnd[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        // COMBINED, because the shader SAMPLES it (usampler2D gbufIds in
        // splat_common.glsl) and the write at w[17] says so too. This was
        // STORAGE_IMAGE from the day the V2 depth-test work added the binding —
        // a shader/layout type mismatch the validation layer flags at pipeline
        // creation (VUID 07990) and every writeSets (VUID 00319), and which the
        // NVIDIA driver TOLERATED most of the time: same-offset access
        // violations inside the ICD, on real scans, at unpredictable moments,
        // were this bug being tolerated less. Undefined behavior that mostly
        // works is worse than a crash on frame one.
        bnd[17].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[18].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;// fog
        bnd[19].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;// clouds
        bnd[20].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;// lights (ambient + suns)
        bnd[21].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;// env
        bnd[23].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;// the splat depth AOV

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = kBindings;
        dlci.pBindings    = bnd.data();
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(splat)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = sizeof(SplatPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(splat)");

        VkPipelineCache cache = ctx_.pipelineCache();
        projectPipe_  = makeComputePipe(d, cache, pipeLayout_, kSplatProjectCompSpv,
                                        sizeof(kSplatProjectCompSpv), "splat_project");
        scanPipe_     = makeComputePipe(d, cache, pipeLayout_, kSplatScanCompSpv,
                                        sizeof(kSplatScanCompSpv), "splat_scan");
        scanAddPipe_  = makeComputePipe(d, cache, pipeLayout_, kSplatScanAddCompSpv,
                                        sizeof(kSplatScanAddCompSpv), "splat_scan_add");
        expandPipe_   = makeComputePipe(d, cache, pipeLayout_, kSplatExpandCompSpv,
                                        sizeof(kSplatExpandCompSpv), "splat_expand");
        indirectPipe_ = makeComputePipe(d, cache, pipeLayout_, kSplatIndirectCompSpv,
                                        sizeof(kSplatIndirectCompSpv), "splat_indirect");
        histPipe_     = makeComputePipe(d, cache, pipeLayout_, kSplatRadixHistCompSpv,
                                        sizeof(kSplatRadixHistCompSpv), "splat_radix_hist");
        scatterPipe_  = makeComputePipe(d, cache, pipeLayout_, kSplatRadixScatterCompSpv,
                                        sizeof(kSplatRadixScatterCompSpv), "splat_radix_scatter");
        rangePipe_    = makeComputePipe(d, cache, pipeLayout_, kSplatRangeCompSpv,
                                        sizeof(kSplatRangeCompSpv), "splat_range");
        rasterPipe_   = makeComputePipe(d, cache, pipeLayout_, kSplatRasterCompSpv,
                                        sizeof(kSplatRasterCompSpv), "splat_raster");
        checksumPipe_ = makeComputePipe(d, cache, pipeLayout_, kSplatChecksumCompSpv,
                                        sizeof(kSplatChecksumCompSpv), "splat_checksum");

        // ── the volume bake's own layout ─────────────────────────────────────
        // Four bindings, none of them one the frame binds. Under
        // THREEPP_VK_SPLATVOL_OFF nothing here is created at all, which is what
        // makes "never bake" a statement about the pipeline cache as well as
        // about VRAM.
        if (volumeOff_) return;

        std::array<VkDescriptorSetLayoutBinding, kBakeBindings> bb{};
        for (uint32_t i = 0; i < kBakeBindings; ++i) {
            bb[i].binding         = i;
            bb[i].descriptorCount = 1;
            bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bb[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bb[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;// the rgba16f volume

        VkDescriptorSetLayoutCreateInfo bdlci{};
        bdlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        bdlci.bindingCount = kBakeBindings;
        bdlci.pBindings    = bb.data();
        check(vkCreateDescriptorSetLayout(d, &bdlci, nullptr, &bakeDsLayout_),
              "vkCreateDescriptorSetLayout(splat bake)");

        VkPushConstantRange bpc{};
        bpc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bpc.size       = sizeof(SplatBakePc);
        VkPipelineLayoutCreateInfo bplci{};
        bplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        bplci.setLayoutCount         = 1;
        bplci.pSetLayouts            = &bakeDsLayout_;
        bplci.pushConstantRangeCount = 1;
        bplci.pPushConstantRanges    = &bpc;
        check(vkCreatePipelineLayout(d, &bplci, nullptr, &bakePipeLayout_),
              "vkCreatePipelineLayout(splat bake)");

        bakeScatterPipe_ = makeComputePipe(d, cache, bakePipeLayout_, kSplatBakeScatterCompSpv,
                                           sizeof(kSplatBakeScatterCompSpv), "splat_bake_scatter");
        bakeResolvePipe_ = makeComputePipe(d, cache, bakePipeLayout_, kSplatBakeResolveCompSpv,
                                           sizeof(kSplatBakeResolveCompSpv), "splat_bake_resolve");
    }

    void SplatPass::createDescriptorPool() {
        const uint32_t sets = kMaxClouds * framesInFlight_ * kMaxTargets;
        VkDescriptorPoolSize sz[4]{};
        sz[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sz[0].descriptorCount = sets * 4;// splat + fog + clouds + lights
        sz[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sz[1].descriptorCount = sets * 14;// 13 sort/scratch buffers + indirect args
        sz[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sz[2].descriptorCount = sets * 3;// sceneHdr + gbuf motion + splat depth AOV
        sz[3].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sz[3].descriptorCount = sets * 3;// gbuf depth + gbuf ids + env

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = sets;
        dpci.poolSizeCount = 4;
        dpci.pPoolSizes    = sz;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(splat)");

        if (volumeOff_) return;

        // ONE bake set, for every cloud ever uploaded: uploadCloud's one-shot
        // waits on the queue before returning, so the set is provably idle when
        // the next upload rewrites it. Nothing here is per frame-in-flight
        // because nothing here happens on a frame.
        VkDescriptorPoolSize bsz[2]{};
        bsz[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bsz[0].descriptorCount = 3;// geom + sh + scratch
        bsz[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bsz[1].descriptorCount = 1;// the volume

        VkDescriptorPoolCreateInfo bdpci{};
        bdpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        bdpci.maxSets       = 1;
        bdpci.poolSizeCount = 2;
        bdpci.pPoolSizes    = bsz;
        check(vkCreateDescriptorPool(ctx_.device(), &bdpci, nullptr, &bakeDescPool_),
              "vkCreateDescriptorPool(splat bake)");

        VkDescriptorSetAllocateInfo bai{};
        bai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        bai.descriptorPool     = bakeDescPool_;
        bai.descriptorSetCount = 1;
        bai.pSetLayouts        = &bakeDsLayout_;
        check(vkAllocateDescriptorSets(ctx_.device(), &bai, &bakeSet_),
              "vkAllocateDescriptorSets(splat bake)");
    }

    void SplatPass::oneShot(const std::function<void(VkCommandBuffer)>& body) const {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(splat)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(splat)");
        body(cb);
        check(vkEndCommandBuffer(cb), "end one-shot cb(splat)");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(splat)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(splat)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void SplatPass::uploadCloud(Cloud& c, const SplatCloud& src) {
        const SplatData& data = src.data();
        const uint32_t n      = static_cast<uint32_t>(data.count());
        const uint32_t coeffs = static_cast<uint32_t>(data.coeffCount());

        c.count    = n;
        c.shCoeffs = coeffs;
        c.shDegree = static_cast<uint32_t>(data.shDegree);
        if (n == 0) return;

        const VkDeviceSize geomBytes = VkDeviceSize(n) * kGeomStride;
        const VkDeviceSize shBytes   = VkDeviceSize(n) * coeffs * 2 * sizeof(uint32_t);

        // Staging + device-local. Every geometry buffer in this renderer is
        // host-visible mapped; splat data is different in kind — uploaded once,
        // read every frame by every tile, never written again — so it belongs
        // in device-local memory even though nothing else here does.
        Buffer stageGeom = createBuffer(ctx_.allocator(), ctx_.device(), geomBytes,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
        Buffer stageSh   = createBuffer(ctx_.allocator(), ctx_.device(), shBytes,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT);

        {
            VmaAllocationInfo gi{}, si{};
            vmaGetAllocationInfo(ctx_.allocator(), stageGeom.alloc, &gi);
            vmaGetAllocationInfo(ctx_.allocator(), stageSh.alloc, &si);
            auto* gp = static_cast<float*>(gi.pMappedData);
            auto* sp = static_cast<uint32_t*>(si.pMappedData);

            for (uint32_t i = 0; i < n; ++i) {
                float* dst = gp + size_t(i) * 10;
                dst[0] = data.means[i].x;
                dst[1] = data.means[i].y;
                dst[2] = data.means[i].z;
                dst[3] = data.opacities[i];
                data.computeCovariance(i, dst + 4);

                const float* sh = data.shAt(i);
                uint32_t* so = sp + size_t(i) * coeffs * 2;
                for (uint32_t k = 0; k < coeffs; ++k) {
                    // packHalf2x16 semantics: low 16 bits = .x, high = .y.
                    // Two words per coefficient (r,g | b,0) is the same 8 bytes
                    // the GL path's RGBA16F texel spends, and the same reason:
                    // one fetch per coefficient, nothing straddling a boundary.
                    const uint16_t r = DataUtils::toHalfFloat(sh[k * 3 + 0]);
                    const uint16_t gg = DataUtils::toHalfFloat(sh[k * 3 + 1]);
                    const uint16_t b = DataUtils::toHalfFloat(sh[k * 3 + 2]);
                    so[k * 2 + 0] = uint32_t(r) | (uint32_t(gg) << 16);
                    so[k * 2 + 1] = uint32_t(b);
                }
            }
            flushHostWrites(ctx_.allocator(), stageGeom.alloc);
            flushHostWrites(ctx_.allocator(), stageSh.alloc);
        }

        c.geom = createBuffer(ctx_.allocator(), ctx_.device(), geomBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO);
        c.sh   = createBuffer(ctx_.allocator(), ctx_.device(), shBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(c.geom.handle, "splat.geom");
        ctx_.setObjectName(c.sh.handle, "splat.sh");

        // ── the reflection volume's extent, resolution and scratch ───────────
        // Everything the bake needs from the CPU is decided here; the bake
        // itself reads the DEVICE-LOCAL buffers copied above, so there is no
        // second pass over the splats and no second staging allocation.
        Buffer scratch{};
        if (!volumeOff_ && bakeScatterPipe_ != VK_NULL_HANDLE) {
            planVolume(c, data);
            if (c.volRes[0] > 0) {
                c.volume = createVolumeImage(ctx_, c.volRes[0], c.volRes[1], c.volRes[2],
                                             "splat.volume");
                const VkDeviceSize voxels = VkDeviceSize(c.volRes[0]) * c.volRes[1] * c.volRes[2];
                scratch = createBuffer(ctx_.allocator(), ctx_.device(),
                                       voxels * kVolCounters * sizeof(uint32_t),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VMA_MEMORY_USAGE_AUTO);
                ctx_.setObjectName(scratch.handle, "splat.volumeScratch");
            }
        }

        oneShot([&](VkCommandBuffer cb) {
            VkBufferCopy cg{0, 0, geomBytes};
            VkBufferCopy cs{0, 0, shBytes};
            vkCmdCopyBuffer(cb, stageGeom.handle, c.geom.handle, 1, &cg);
            vkCmdCopyBuffer(cb, stageSh.handle, c.sh.handle, 1, &cs);
            // Same command buffer as the copies, after them: a 5M-splat bake is
            // tens of milliseconds hidden inside an upload that already moves
            // ~1.2 GB, and it is off the frame path entirely.
            if (scratch.handle != VK_NULL_HANDLE) recordBake(cb, c, scratch);
        });

        destroyBuffer(ctx_.allocator(), stageGeom);
        destroyBuffer(ctx_.allocator(), stageSh);
        // Transient by contract: oneShot waits on the queue, so the scratch's
        // last reader has retired and 16 B/voxel goes straight back.
        destroyBuffer(ctx_.allocator(), scratch);

        if (c.volume.image != VK_NULL_HANDLE) ++volumeGen_;
    }

    void SplatPass::planVolume(Cloud& c, const SplatData& data) {
        c.volRes[0] = c.volRes[1] = c.volRes[2] = 0;

        const size_t n = data.count();
        if (n == 0 || data.scales.size() != n) return;

        // Cloud-local AABB over the means, ROBUST TO FLOATERS: p1/p99 per axis
        // over a fixed-stride ~8192 sample — the exact estimator, sample size
        // and percentiles the collector already uses for the sort interval
        // (VulkanCoreScene.cpp:2893), reused so both ends agree on what "the
        // subject" is. A stride sample is exact-repeatable (no RNG, no state),
        // which the determinism contract needs as much as the shader does.
        const size_t stride = std::max<size_t>(1, n / kVolSampleTarget);
        std::vector<float> ax[3], sig;
        for (size_t i = 0; i < n; i += stride) {
            const auto& m = data.means[i];
            if (!std::isfinite(m.x) || !std::isfinite(m.y) || !std::isfinite(m.z)) continue;
            ax[0].push_back(m.x);
            ax[1].push_back(m.y);
            ax[2].push_back(m.z);
            // trace(Sigma) is rotation-invariant, so the mean 1-sigma extent is
            // the scales alone — the same sigbar splat_bake_scatter.comp
            // derives from the uploaded covariance.
            const auto& s = data.scales[i];
            sig.push_back(std::sqrt((s.x * s.x + s.y * s.y + s.z * s.z) / 3.f));
        }
        if (ax[0].size() < 3) return;// nothing to bound; leave the cloud unbaked

        const auto pct = [](std::vector<float>& v, float q) {
            const auto rank = static_cast<size_t>(q * static_cast<float>(v.size() - 1));
            std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(rank), v.end());
            return v[rank];
        };

        // Pad by the p99 splat's own 3 sigma, so the splats that DEFINED the
        // box are inside it rather than clipped by it — and by a robust sigma,
        // so one background-sky Gaussian cannot inflate the box (and with it
        // the voxel size) for the whole scan.
        const float pad = 3.f * pct(sig, 0.99f);

        float size[3]{};
        float maxSize = 0.f;
        for (int a = 0; a < 3; ++a) {
            const float lo = pct(ax[a], 0.01f) - pad;
            const float hi = pct(ax[a], 0.99f) + pad;
            c.localBoxMin[a] = lo;
            // A planar cloud (or a single splat with no extent) still needs a
            // positive size for the voxel size to mean anything.
            size[a] = std::max(hi - lo, 1e-4f);
            maxSize = std::max(maxSize, size[a]);
        }
        if (!(maxSize > 0.f) || !std::isfinite(maxSize)) return;

        for (int a = 0; a < 3; ++a) {
            c.localBoxSize[a] = size[a];
            // Longest axis volMaxRes_, the others proportional, snapped even
            // and floored — total is bounded by volMaxRes_^3 by construction.
            auto r = static_cast<uint32_t>(std::lround(
                    double(volMaxRes_) * double(size[a]) / double(maxSize)));
            r = std::clamp(r, kVolMinRes, volMaxRes_) & ~1u;
            c.volRes[a] = std::max(r, kVolMinRes);
        }
    }

    void SplatPass::recordBake(VkCommandBuffer cb, Cloud& c, const Buffer& scratch) {
        SplatBakePc pc{};
        for (int a = 0; a < 3; ++a) {
            pc.boxMin[a]    = c.localBoxMin[a];
            pc.boxSize[a]   = c.localBoxSize[a];
            pc.res[a]       = c.volRes[a];
            pc.voxelSize[a] = c.localBoxSize[a] / float(c.volRes[a]);
        }
        pc.splatCount  = c.count;
        pc.shCoeffs    = c.shCoeffs;
        pc.voxelVolume = pc.voxelSize[0] * pc.voxelSize[1] * pc.voxelSize[2];

        VkDescriptorBufferInfo bi[3] = {{c.geom.handle, 0, VK_WHOLE_SIZE},
                                        {c.sh.handle, 0, VK_WHOLE_SIZE},
                                        {scratch.handle, 0, VK_WHOLE_SIZE}};
        VkDescriptorImageInfo img{VK_NULL_HANDLE, c.volume.view, VK_IMAGE_LAYOUT_GENERAL};
        std::array<VkWriteDescriptorSet, kBakeBindings> w{};
        for (uint32_t i = 0; i < kBakeBindings; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = bakeSet_;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo     = &bi[i];
        }
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[3].pBufferInfo    = nullptr;
        w[3].pImageInfo     = &img;
        vkUpdateDescriptorSets(ctx_.device(), kBakeBindings, w.data(), 0, nullptr);

        // UNDEFINED -> GENERAL, and GENERAL for the rest of the image's life:
        // the consumer table binds live volumes and dummy slots through one
        // sampler array, and the dust table's dummy is GENERAL too
        // (ParticleFieldPass.cpp:501) — one layout across every slot is one
        // less thing a descriptor write can get wrong.
        VkImageMemoryBarrier2 ib{};
        ib.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        ib.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        ib.srcAccessMask = 0;
        ib.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        ib.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        ib.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image         = c.volume.image;
        ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib.subresourceRange.levelCount = 1;
        ib.subresourceRange.layerCount = 1;
        VkDependencyInfo di{};
        di.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers    = &ib;
        vkCmdPipelineBarrier2(cb, &di);

        // The counters must start at zero, and the staging copies above must
        // have landed before the scatter reads geom/sh — barrier() covers both
        // directions (transfer -> compute, compute -> compute).
        vkCmdFillBuffer(cb, scratch.handle, 0, scratch.size, 0u);
        barrier(cb);

        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bakePipeLayout_, 0, 1,
                                &bakeSet_, 0, nullptr);
        vkCmdPushConstants(cb, bakePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bakeScatterPipe_);
        vkCmdDispatch(cb, divUp(c.count, 256), 1, 1);
        barrier(cb);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, bakeResolvePipe_);
        vkCmdDispatch(cb, divUp(c.volRes[0], 4), divUp(c.volRes[1], 4), divUp(c.volRes[2], 4));

        // The resolve's imageStore -> every later SAMPLE of this image. It is
        // the last thing the one-shot does, so this is the handover to whatever
        // frame binds the volume next.
        //
        // ALL_COMMANDS rather than an enumerated destination: the consumers are
        // a compute shade today and a .rgen sensor tomorrow, and naming
        // RAY_TRACING_SHADER here would be a VALIDATION ERROR on a device where
        // rayTracingPipeline is off (VulkanContext only enables it under
        // rayTracingEnabled_ — lavapipe, which the cross-hardware validation
        // gate runs on, is exactly that device). This barrier executes once per
        // upload at the tail of a one-shot, so the breadth costs nothing.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo fi{};
        fi.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        fi.memoryBarrierCount = 1;
        fi.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &fi);
    }

    void SplatPass::allocateScratch(uint32_t maxSplats, uint32_t entryBudget) {
        VmaAllocator a = ctx_.allocator();
        VkDevice d = ctx_.device();
        maxSplats_   = maxSplats;
        entryBudget_ = std::max(entryBudget, 1u);

        for (auto* b : {&projBuf_, &countBuf_, &offsetBuf_, &keyA_, &valA_, &keyB_, &valB_,
                        &histBuf_, &scanBuf_})
            destroyBuffer(a, *b);

        constexpr VkBufferUsageFlags kSsbo =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto mk = [&](VkDeviceSize bytes, const char* name) {
            Buffer b = createBuffer(a, d, std::max<VkDeviceSize>(bytes, 256), kSsbo,
                                    VMA_MEMORY_USAGE_AUTO);
            ctx_.setObjectName(b.handle, name);
            return b;
        };

        projBuf_   = mk(VkDeviceSize(maxSplats) * kProjStride, "splat.proj");
        countBuf_  = mk(VkDeviceSize(maxSplats) * 4, "splat.counts");
        offsetBuf_ = mk(VkDeviceSize(maxSplats) * 4, "splat.offsets");
        keyA_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.keyA");
        valA_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.valA");
        keyB_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.keyB");
        valB_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.valB");

        const uint32_t radixBlocks = divUp(entryBudget_, kRadixBlock);
        const uint32_t histWords   = kRadixBins * radixBlocks;
        histBuf_ = mk(VkDeviceSize(histWords) * 4, "splat.hist");
        scanBuf_ = mk(VkDeviceSize(std::max(scanScratchWords(histWords),
                                            scanScratchWords(maxSplats)) + 8) * 4,
                      "splat.scanScratch");
    }

    void SplatPass::writeSets(Cloud& c) {
        for (uint32_t t = 0; t < kMaxTargets; ++t)
            if (targets_[t].claimed) writeSets(c, t);
    }

    void SplatPass::writeSets(Cloud& c, uint32_t target) {
        // Every binding needs a real object: descriptorBindingPartiallyBound is
        // not enabled on this device, so a VK_NULL_HANDLE anywhere in the set
        // is a validation error, not a "don't sample it".
        if (target >= kMaxTargets) return;
        const Target& tg = targets_[target];
        if (tg.sceneHdrViews.empty() || c.sets.size() < (target + 1u) * framesInFlight_ ||
            envView_ == VK_NULL_HANDLE ||
            fogUbos_.empty() || cloudUbos_.empty() || lightsUbos_.empty() ||
            tg.splatDepthViews.empty())
            return;

        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorBufferInfo ubo{uboBuf_[f].handle,
                                       uboStride_ * (target * kMaxClouds + c.slot),
                                       sizeof(SplatUboData)};
            const VkBuffer ssbo[13] = {
                    c.geom.handle, c.sh.handle, projBuf_.handle, countBuf_.handle,
                    offsetBuf_.handle, keyA_.handle, valA_.handle, keyB_.handle,
                    valB_.handle, rangeBuf_.handle, globalBuf_.handle, histBuf_.handle,
                    scanBuf_.handle};

            VkDescriptorBufferInfo bi[13]{};
            std::array<VkWriteDescriptorSet, kBindings> w{};
            for (uint32_t i = 0; i < kBindings; ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = c.sets[target * framesInFlight_ + f];
                w[i].dstBinding      = i;
                w[i].descriptorCount = 1;
            }
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[0].pBufferInfo    = &ubo;
            for (uint32_t i = 0; i < 13; ++i) {
                bi[i] = {ssbo[i], 0, VK_WHOLE_SIZE};
                w[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i + 1].pBufferInfo    = &bi[i];
            }

            VkDescriptorImageInfo hdr{VK_NULL_HANDLE, tg.sceneHdrViews[f], VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo dep{sampler_, tg.depthViews[f],
                                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo mot{VK_NULL_HANDLE, tg.motionViews[f], VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo ids{sampler_, tg.idsViews[f],
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[14].pImageInfo     = &hdr;
            w[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[15].pImageInfo     = &dep;
            w[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[16].pImageInfo     = &mot;
            w[17].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[17].pImageInfo     = &ids;

            VkDescriptorBufferInfo ub[3]{{fogUbos_[f], 0, VK_WHOLE_SIZE},
                                         {cloudUbos_[f], 0, VK_WHOLE_SIZE},
                                         {lightsUbos_[f], 0, VK_WHOLE_SIZE}};
            for (uint32_t i = 0; i < 3; ++i) {
                w[18 + i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w[18 + i].pBufferInfo    = &ub[i];
            }
            VkDescriptorImageInfo env{envSampler_ ? envSampler_ : sampler_, envView_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[21].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[21].pImageInfo     = &env;

            VkDescriptorBufferInfo ind{indirectBuf_.handle, 0, VK_WHOLE_SIZE};
            w[22].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[22].pBufferInfo    = &ind;

            // GENERAL for the whole frame: the pass both stores to it and
            // clears it by transfer, and the AOV readback transitions it away
            // and back around its copy. Full-res when the AOV is on, 1x1 when
            // it is off — either way a real image, which is what the opening
            // comment of this function is about.
            VkDescriptorImageInfo sd{VK_NULL_HANDLE, tg.splatDepthViews[f],
                                     VK_IMAGE_LAYOUT_GENERAL};
            w[23].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[23].pImageInfo     = &sd;

            vkUpdateDescriptorSets(ctx_.device(), kBindings, w.data(), 0, nullptr);
        }
    }

    void SplatPass::setEnvironment(VkImageView view, VkSampler sampler, uint32_t mips) {
        if (view == VK_NULL_HANDLE) return;// nothing to point at yet
        if (view == envView_ && sampler == envSampler_) return;
        envView_    = view;
        envSampler_ = sampler;
        envMips_    = std::max(mips, 1u);
        envDirty_   = true;
    }

    void SplatPass::rewriteEnvironment(VkImageView view, VkSampler sampler, uint32_t mips) {
        if (view == VK_NULL_HANDLE) return;// nothing to point at yet
        envView_    = view;
        envSampler_ = sampler;
        envMips_    = std::max(mips, 1u);
        // Caller has drained the device; the old env image is already freed,
        // so every resident set holds a dead view until this lands.
        for (auto& kv : resident_) writeSets(*kv.second);
        envDirty_ = false;
    }

    void SplatPass::retireStale() {
        for (auto it = resident_.begin(); it != resident_.end();) {
            if (it->second->lastSeen + framesInFlight_ + 1 <= syncSerial_) {
                destroyCloudResources(*it->second);
                freeSlots_.emplace_back(it->second->slot, std::move(it->second->sets));
                it = resident_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void SplatPass::syncClouds(const std::vector<CloudEntry>& clouds,
                               const std::vector<const SplatCloud*>& parked) {
        frameClouds_.clear();
        ++syncSerial_;

        // HIDDEN is not DELETED. A parked cloud (in the scene, not effectively
        // visible) keeps its lastSeen fresh so retireStale leaves its geometry
        // and SH buffers resident — toggling visibility on a 5M-splat scan
        // costs nothing instead of a seconds-long re-upload each way. Only a
        // cloud that leaves the SCENE (or dies) stops appearing in either list
        // and ages out. The uuid check keeps a recycled address from parking a
        // dead cloud's buffers under a live pointer. BEFORE retireStale, so a
        // refresh on the aging boundary wins.
        for (const SplatCloud* p : parked) {
            if (!p) continue;
            if (auto it = resident_.find(p);
                it != resident_.end() && it->second->uuid == p->uuid)
                it->second->lastSeen = syncSerial_;
        }

        // BEFORE the empty early-out: deleting the scene's last splat cloud is
        // exactly when there is something to retire and nothing to submit.
        retireStale();

        // Nothing resident at all: the shared sort scratch serves nobody, and
        // by retireStale's own timing argument nothing in flight references it
        // — the last frame that recorded a splat pass drained at least
        // framesInFlight_+1 syncs ago. ~700 MB at a 5M high-water; letting it
        // linger after the scan is deleted is the absolute-caps doctrine
        // violated. (A PARKED cloud keeps the scratch alive: resident_ is not
        // empty then, and unhide should not wait on a reallocation it can see
        // coming.)
        if (resident_.empty() && maxSplats_ > 1) {
            for (auto* b : {&projBuf_, &countBuf_, &offsetBuf_, &keyA_, &valA_, &keyB_, &valB_,
                            &histBuf_, &scanBuf_})
                destroyBuffer(ctx_.allocator(), *b);
            maxSplats_    = 0;
            entryBudget_  = 0;
            budgetCapped_ = false;
        }
        if (clouds.empty()) return;

        // A cloud arriving or a cloud growing is a rare, load-time event; both
        // reallocate shared scratch and rewrite descriptor sets that other
        // frames may still be reading, so the device is drained first. This is
        // the one stall in the pass and it happens on asset load, not per frame.
        // A replaced environment means every resident set holds a dead view;
        // the rewrite needs the same waitIdle a new upload does.
        bool structural = envDirty_;
        // Live demand comes from what this frame SUBMITS. Parked clouds keep
        // their own buffers but record nothing, so the scratch never has to
        // fit them — hiding the big scan lets the scratch shrink to the scene
        // that is actually drawing.
        uint32_t liveSplats = 0;
        for (const auto& e : clouds) {
            if (!e.cloud) continue;
            // A pointer hit with a uuid MISMATCH is a recycled address wearing
            // a dead cloud's key — structural, so the branch below gets its
            // waitIdle and can destroy the corpse before uploading the heir.
            if (auto it = resident_.find(e.cloud);
                it == resident_.end() || it->second->uuid != e.cloud->uuid) {
                structural = true;
            }
            liveSplats = std::max(liveSplats, static_cast<uint32_t>(e.cloud->splatCount()));
        }
        uint32_t needSplats  = maxSplats_;
        uint32_t needEntries = entryBudget_;
        if (liveSplats > maxSplats_) {
            structural  = true;
            needSplats  = liveSplats;
            needEntries = std::max<uint32_t>(
                    needEntries,
                    static_cast<uint32_t>(std::min<uint64_t>(
                            uint64_t(liveSplats) * kEntriesPerSplat, kMaxEntries)));
        } else if (uint64_t(liveSplats) * 2 < maxSplats_) {
            // The high-water mark is at least double the live demand — V3 item
            // 5, promoted from "hygiene" the day eviction started actually
            // freeing cloud buffers while this stayed behind. Post-indirect-
            // dispatch the oversize costs VRAM only, so shrinking is a pure
            // win; the hysteresis is the factor of two itself plus the sync
            // delay below, so a hide/unhide toggle inside the window pays
            // nothing and a scene that really did get smaller reclaims within
            // a second.
            if (shrinkSince_ == 0) shrinkSince_ = syncSerial_;
            if (syncSerial_ - shrinkSince_ > 60) {
                structural    = true;
                needSplats    = std::max(liveSplats, 1u);
                needEntries   = static_cast<uint32_t>(std::min<uint64_t>(
                        uint64_t(needSplats) * kEntriesPerSplat, kMaxEntries));
                budgetCapped_ = false;
                shrinkSince_  = 0;
            }
        } else {
            shrinkSince_ = 0;
        }

        // A frame that overflowed was TRUNCATED — splats are missing from
        // whichever tiles the key list ran out in, which looks like blocky
        // holes rather than like a resource problem. The expansion reports the
        // exact shortfall, so the fix is to resize to what the frame WANTED
        // (plus a quarter of headroom for the next camera angle) rather than to
        // climb a doubling ladder and pay a wrong frame for every rung.
        bool grewOnOverflow = false;
        if (const uint32_t over = lastOverflow(); over > 0 && !budgetCapped_) {
            grewOnOverflow = true;
            const uint64_t wanted = (uint64_t(entryBudget_) + over) * 5 / 4;
            uint32_t next = static_cast<uint32_t>(std::min<uint64_t>(wanted, kMaxEntries));
            if (next > entryBudget_) {
                needEntries = std::max(needEntries, next);
                structural  = true;
            }
            if (wanted > kMaxEntries) {
                budgetCapped_ = true;
                std::cerr << "[threepp] SplatPass: this frame's tile expansion wants "
                          << wanted << " (splat, tile) pairs, past the " << kMaxEntries
                          << " ceiling (" << (kMaxEntryBytes >> 20)
                          << " MB of sort buffers). Frames will be TRUNCATED — splats "
                             "missing from some tiles. Fewer or smaller splats, or a "
                             "coarser tile grid, is the fix."
                          << std::endl;
            }
        }

        if (structural) {
            check(vkDeviceWaitIdle(ctx_.device()), "vkDeviceWaitIdle(splat sync)");

            // Device idle, so recycled-address corpses (uuid mismatch) can be
            // destroyed on the spot; their slot and sets go back to the pool
            // and the upload loop below sees a clean miss at that key.
            for (auto it = resident_.begin(); it != resident_.end();) {
                bool recycled = false;
                for (const auto& e : clouds) {
                    if (e.cloud == it->first && it->second->uuid != e.cloud->uuid) {
                        recycled = true;
                        break;
                    }
                }
                if (recycled) {
                    destroyCloudResources(*it->second);
                    freeSlots_.emplace_back(it->second->slot, std::move(it->second->sets));
                    it = resident_.erase(it);
                } else {
                    ++it;
                }
            }

            for (const auto& e : clouds) {
                if (!e.cloud || resident_.count(e.cloud)) continue;
                if (freeSlots_.empty() && slotsUsed_ >= kMaxClouds) {
                    std::cerr << "[threepp] SplatPass: more than " << kMaxClouds
                              << " splat clouds in one scene; the extra ones are not drawn"
                              << std::endl;
                    break;
                }
                auto c = std::make_unique<Cloud>();
                c->uuid     = e.cloud->uuid;
                c->lastSeen = syncSerial_;
                if (!freeSlots_.empty()) {
                    // A retired cloud's slot and sets, ready for rewriting —
                    // writeSets below refreshes every binding, so nothing of
                    // the previous owner survives into the reuse.
                    c->slot = freeSlots_.back().first;
                    c->sets = std::move(freeSlots_.back().second);
                    freeSlots_.pop_back();
                    uploadCloud(*c, *e.cloud);
                } else {
                    c->slot = slotsUsed_++;
                    uploadCloud(*c, *e.cloud);

                    // One set per (target, frame-in-flight): a target's images
                    // are its own, so the set that names them is too. Allocated
                    // for every slot up front — the pool is sized for it and an
                    // allocation is not where a view opting in later should
                    // discover it came too late.
                    const uint32_t nSets = framesInFlight_ * kMaxTargets;
                    std::vector<VkDescriptorSetLayout> layouts(nSets, dsLayout_);
                    VkDescriptorSetAllocateInfo ai{};
                    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    ai.descriptorPool     = descPool_;
                    ai.descriptorSetCount = nSets;
                    ai.pSetLayouts        = layouts.data();
                    c->sets.resize(nSets);
                    check(vkAllocateDescriptorSets(ctx_.device(), &ai, c->sets.data()),
                          "vkAllocateDescriptorSets(splat)");
                }
                resident_.emplace(e.cloud, std::move(c));
            }

            if (needSplats != maxSplats_ || needEntries != entryBudget_) {
                // Worth one line on stderr, but only when an actual frame was
                // truncated: until the resize lands those frames ARE wrong
                // (splats missing from tiles), and a silent self-correction
                // leaves whoever saw them with no explanation. Sizing a cloud
                // for the first time is not that and says nothing.
                if (grewOnOverflow)
                    std::cerr << "[threepp] SplatPass: tile expansion budget "
                              << entryBudget_ << " -> " << needEntries
                              << " (splat, tile) pairs after a truncated frame"
                              << std::endl;
                allocateScratch(needSplats, needEntries);
                // The overflow counters just consumed describe a frame that no
                // longer exists. Clearing them stops the next few syncs from
                // reading the same stale shortfall and growing again.
                for (uint32_t f = 0; f < framesInFlight_; ++f) {
                    VmaAllocationInfo ai{};
                    vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[f].alloc, &ai);
                    if (ai.pMappedData) std::memset(ai.pMappedData, 0, kGlobalWords * 4);
                }
            }
            for (auto& kv : resident_) writeSets(*kv.second);
            envDirty_ = false;// every set now points at the live environment
        }

        // Opt-in invariant report. The two numbers that matter are structural,
        // not cosmetic: a non-zero scan violation means the expansion ranges
        // overlap (splats overwriting each other), a non-zero order violation
        // means the radix sort dropped or duplicated entries. Both show up as
        // "some splats are just missing", which is indistinguishable by eye
        // from a culling bug — hence the assertion rather than the inference.
        if (const char* e = std::getenv("THREEPP_VK_SPLAT_CHECKSUM"); e && *e && *e != '0') {
            VmaAllocationInfo ai{};
            vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[lastFrame_].alloc, &ai);
            invalidateHostReads(ctx_.allocator(), debugBuf_[lastFrame_].alloc);
            if (const auto* w = static_cast<const uint32_t*>(ai.pMappedData); w && w[2]) {
                std::cerr << "[splat] entries " << w[2] << " overflow " << w[3]
                          << " visible " << w[4] << " scanBad " << w[10]
                          << " orderBad " << w[11]
                          // The three hashes are what makes this an A/B
                          // instrument rather than only an invariant report: on
                          // a real scan the final FRAME is not reproducible
                          // run to run, so a changed picture proves nothing —
                          // these hash the splat pass's own output (sorted keys,
                          // sorted payload, composited pixels) and are the level
                          // at which a sort change must be byte-identical.
                          << " hashKey " << w[7] << " hashVal " << w[8]
                          << " hashColor " << w[9] << std::endl;
            }
        }

        for (const auto& e : clouds) {
            auto it = resident_.find(e.cloud);
            if (it == resident_.end() || it->second->count == 0) continue;
            it->second->lastSeen = syncSerial_;
            FrameCloud fc{};
            fc.cloud = it->second.get();
            std::memcpy(fc.model, e.model, sizeof(fc.model));
            fc.pLo = e.pLo;
            fc.pHi = e.pHi;
            fc.debugNonFinite = e.debugNonFinite;
            fc.pointMix       = e.pointMix;
            fc.pointSigma     = e.pointSigma;
            // Validate the submission list HERE rather than trusting it in the
            // shader: a range past the end of the cloud would read another
            // cloud's memory through the same descriptor, and a caller that
            // hands over more than kMaxRanges would silently overflow the UBO.
            fc.submitCount = it->second->count;
            if (!e.ranges.empty()) {

                uint32_t total = 0;
                for (const auto& [off, n] : e.ranges) {
                    if (fc.ranges.size() >= kMaxRanges) break;
                    if (n == 0 || off >= it->second->count) continue;
                    const uint32_t clamped = std::min(n, it->second->count - off);
                    fc.ranges.emplace_back(off, clamped);
                    total += clamped;
                }
                fc.submitCount = total;
                if (fc.ranges.empty() || total == 0) continue;// nothing submitted
            }
            frameClouds_.push_back(std::move(fc));
        }
    }

    uint32_t SplatPass::acquireTarget() {
        // Slot 0 is the primary's for the pass's lifetime — never handed out.
        for (uint32_t t = 1; t < kMaxTargets; ++t) {
            if (targets_[t].claimed) continue;
            targets_[t].claimed = true;
            return t;
        }
        return kNoTarget;
    }

    void SplatPass::releaseTarget(uint32_t target) {
        if (target == 0 || target >= kMaxTargets) return;
        targets_[target] = Target{};
        // The descriptor sets stay allocated and stay pointed at a dead view
        // until the slot is claimed again, which is exactly when they are
        // rewritten — nothing records against an unclaimed slot.
        resizeTileRange();
    }

    bool SplatPass::targetValid(uint32_t target) const {
        return target < kMaxTargets && targets_[target].claimed &&
               targets_[target].width > 0 && !targets_[target].sceneHdrViews.empty();
    }

    void SplatPass::resizeTileRange() {
        uint32_t tiles = 0;
        for (const auto& t : targets_)
            if (t.claimed) tiles = std::max(tiles, t.tilesX * t.tilesY);
        if (tiles == 0) return;
        const VkDeviceSize want = VkDeviceSize(tiles) * 8;
        if (rangeBuf_.handle != VK_NULL_HANDLE && rangeBuf_.size >= want) return;
        destroyBuffer(ctx_.allocator(), rangeBuf_);
        rangeBuf_ = createBuffer(ctx_.allocator(), ctx_.device(), want,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(rangeBuf_.handle, "splat.tileRange");
    }

    void SplatPass::resize(uint32_t width, uint32_t height, const ResizeInputs& in,
                           uint32_t target) {
        if (!in.sceneHdrPerFrame || !in.depthPerFrame) return;
        if (target >= kMaxTargets) return;
        Target& tg = targets_[target];
        tg.claimed = true;

        const uint32_t tx = divUp(std::max(width, 1u), kTileW);
        const uint32_t ty = divUp(std::max(height, 1u), kTileH);

        tg.width  = width;
        tg.height = height;
        tg.tilesX = tx;
        tg.tilesY = ty;

        tg.sceneHdrViews.assign(in.sceneHdrPerFrame, in.sceneHdrPerFrame + framesInFlight_);
        tg.depthViews.assign(in.depthPerFrame, in.depthPerFrame + framesInFlight_);
        if (in.fogUbos)    fogUbos_.assign(in.fogUbos, in.fogUbos + framesInFlight_);
        if (in.cloudUbos)  cloudUbos_.assign(in.cloudUbos, in.cloudUbos + framesInFlight_);
        if (in.lightsUbos) lightsUbos_.assign(in.lightsUbos, in.lightsUbos + framesInFlight_);
        if (in.envView) {
            envView_    = in.envView;
            envSampler_ = in.envSampler;
            envMips_    = std::max(in.envMips, 1u);
        }
        if (in.motionPerFrame) tg.motionViews.assign(in.motionPerFrame, in.motionPerFrame + framesInFlight_);
        else                   tg.motionViews = tg.sceneHdrViews;
        if (in.motionImages) tg.motionImages.assign(in.motionImages, in.motionImages + framesInFlight_);
        else                 tg.motionImages.assign(framesInFlight_, VK_NULL_HANDLE);
        if (in.idsPerFrame) tg.idsViews.assign(in.idsPerFrame, in.idsPerFrame + framesInFlight_);
        else                tg.idsViews = tg.depthViews;
        // No fallback to another image here, the way motion/ids fall back
        // above: binding 23 is declared r32f and those are not, and a storage
        // image whose view format disagrees with the shader's format qualifier
        // is undefined behaviour rather than a wasted write. Without a real
        // image the sets simply are not written (writeSets returns early).
        if (in.splatDepthPerFrame) {
            tg.splatDepthViews.assign(in.splatDepthPerFrame, in.splatDepthPerFrame + framesInFlight_);
        } else {
            tg.splatDepthViews.clear();
        }
        if (in.splatDepthImages) {
            tg.splatDepthImages.assign(in.splatDepthImages, in.splatDepthImages + framesInFlight_);
        } else {
            tg.splatDepthImages.assign(framesInFlight_, VK_NULL_HANDLE);
        }

        resizeTileRange();
        if (maxSplats_ == 0) allocateScratch(1, kEntriesPerSplat);

        // This target's sets only: the others may be in flight, and every call
        // site that legitimately rewrites them all (upload, env swap) has its
        // own drained-device argument.
        for (auto& kv : resident_) writeSets(*kv.second, target);
    }

    void SplatPass::barrierIndirect(VkCommandBuffer cb) const {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void SplatPass::barrier(VkCommandBuffer cb) const {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                           VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void SplatPass::recordScan(VkCommandBuffer cb, uint32_t n, uint32_t mode0) {
        if (n == 0) return;

        struct Level { uint32_t count, srcOff, sumOff, mode; };
        std::vector<Level> levels;
        uint32_t cur = n, srcOff = 0, regionOff = 0, mode = mode0;
        while (true) {
            const uint32_t nb = divUp(cur, kScanBlock);
            levels.push_back({cur, srcOff, regionOff, mode});
            if (nb == 1) break;
            srcOff    = regionOff;
            regionOff += nb;
            cur       = nb;
            mode      = 1;// scanScratch, in place
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scanPipe_);
        for (const auto& l : levels) {
            SplatPc pc{l.count, l.srcOff, l.srcOff, l.sumOff, l.mode, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(l.count, kScanBlock), 1, 1);
            barrier(cb);
        }
        // Walk back, skipping the top level: its own block-sum array has a
        // single entry, whose exclusive scan is 0.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scanAddPipe_);
        for (int j = static_cast<int>(levels.size()) - 2; j >= 0; --j) {
            const auto& l = levels[static_cast<size_t>(j)];
            SplatPc pc{l.count, l.srcOff, l.srcOff, l.sumOff, l.mode, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(l.count, kScanThreads), 1, 1);
            barrier(cb);
        }
    }

    void SplatPass::clearDepthAov(VkCommandBuffer cb, uint32_t frame) {
        // Primary target only — the AOV is primary-only by scope, and a
        // secondary view's AOV image is 1x1 by construction.
        if (frame >= targets_[0].splatDepthImages.size()) return;
        const VkImage img = targets_[0].splatDepthImages[frame];
        if (img == VK_NULL_HANDLE) return;

        // Zero is "no cloud owns this pixel" — the sentinel the raster's
        // coverage gate produces by omission — so the AOV is only meaningful
        // if the frame starts from it. This runs from the CALLER, before the
        // hasClouds() test that skips record() entirely: a frame that draws no
        // splats has to leave an empty AOV behind, not the AOV of whichever
        // frame last used this frame-in-flight slot.
        VkClearColorValue zero{};
        VkImageSubresourceRange full{};
        full.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        full.levelCount = 1;
        full.layerCount = 1;
        vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &full);

        // transfer write -> the raster's imageLoad/imageStore.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void SplatPass::record(VkCommandBuffer cb, uint32_t frame, const RecordParams& p) {
        if (frameClouds_.empty() || !valid()) return;
        if (!targetValid(p.target)) return;
        const Target& tg = targets_[p.target];
        lastFrame_ = frame;

        // Debug hashes cost a full extra pass over the key list and an atomic
        // per composited pixel, so they are opt-in: the determinism test asks
        // through setSplatDebugChecksum, a human debugging by hand through the
        // environment variable.
        const char* cse = std::getenv("THREEPP_VK_SPLAT_CHECKSUM");
        const bool checksum = p.checksum || (cse && *cse && *cse != '0');

        // Per-stage timestamps go to the FIRST drawn cloud only — the slots are
        // one pair per stage per frame, and a second writer would both violate
        // VUID-vkCmdWriteTimestamp2-None-03864 and report the wrong interval.
        bool stagesWritten = false;// slots for this frame already used
        bool timing        = false;// this cloud's stages go to the pool
        const auto stageBegin = [&](TimingPass tp) { if (timing) p.timings->begin(cb, tp, frame); };
        const auto stageEnd   = [&](TimingPass tp) { if (timing) p.timings->end(cb, tp, frame); };

        for (const auto& fc : frameClouds_) {
            Cloud& c = *fc.cloud;
            if (c.count == 0) continue;
            // Everything per-splat runs over what this frame SUBMITS; the
            // resident count only sizes the buffers. Equal unless the frame
            // handed over a range list.
            const uint32_t nSubmit = fc.submitCount;
            if (nSubmit == 0) continue;
            timing = p.timings && !stagesWritten;

            // ── per-cloud UBO slot ───────────────────────────────────────────
            SplatUboData u{};
            // modelView = view * model, column-major throughout.
            for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 4; ++row) {
                    float s = 0.f;
                    for (int k = 0; k < 4; ++k) s += p.view[k * 4 + row] * fc.model[col * 4 + k];
                    u.modelView[col * 4 + row] = s;
                }
            std::memcpy(u.proj, p.proj, sizeof(u.proj));
            std::memcpy(u.projInv, p.projInverse, sizeof(u.projInv));
            std::memcpy(u.model, fc.model, sizeof(u.model));
            std::memcpy(u.prevVPfromView, p.prevVPfromView, sizeof(u.prevVPfromView));
            std::memcpy(u.camWorld, p.camWorld, sizeof(u.camWorld));
            u.jitterClip[0] = p.jitterClip[0];
            u.jitterClip[1] = p.jitterClip[1];
            u.envMipCount   = envMips_;
            u.camPosWs[0] = p.camPos[0];
            u.camPosWs[1] = p.camPos[1];
            u.camPosWs[2] = p.camPos[2];
            u.camFwdWs[0] = p.camFwd[0];
            u.camFwdWs[1] = p.camFwd[1];
            u.camFwdWs[2] = p.camFwd[2];
            u.viewport[0] = static_cast<float>(tg.width);
            u.viewport[1] = static_cast<float>(tg.height);
            u.focal[0]    = 0.5f * u.viewport[0] * p.proj[0];
            u.focal[1]    = 0.5f * u.viewport[1] * p.proj[5];
            u.percentile[0] = fc.pLo;
            u.percentile[1] = fc.pHi;
            u.nearPlane   = p.nearPlane;
            u.preExposure = p.preExposure;
            u.pointMix    = fc.pointMix;
            u.pointSigma  = fc.pointSigma;
            // The SUBMITTED total, not the resident total: every stage after
            // project indexes compactly, so this is the only count they need.
            u.splatCount  = fc.submitCount;
            u.rangeCount  = static_cast<uint32_t>(fc.ranges.size());
            {
                uint32_t first = 0, k = 0;
                for (const auto& [off, n] : fc.ranges) {
                    u.ranges[k * 2 + 0] = first;// compact start, ascending
                    u.ranges[k * 2 + 1] = off;  // source base
                    first += n;
                    ++k;
                }
            }
            u.shCoeffs    = c.shCoeffs;
            u.shDegree    = c.shDegree;
            u.tilesX      = tg.tilesX;
            u.tilesY      = tg.tilesY;
            u.tileBits    = tileBitsFor(tg.tilesX * tg.tilesY);
            u.depthBits   = 32u - u.tileBits;
            u.budget      = entryBudget_;
            u.flags       = (p.orthographic ? kSplatFlagOrtho : 0u) |
                      (p.depthTest ? kSplatFlagDepthTest : 0u) |
                      (fc.debugNonFinite ? kSplatFlagDebugNaN : 0u) |
                      (checksum ? kSplatFlagChecksum : 0u) |
                      (p.bgIsSolidColor ? kSplatFlagBgSolid : 0u) |
                      (p.motionVectors ? kSplatFlagMotion : 0u) |
                      (p.fog ? kSplatFlagFog : 0u) |
                      // Belt and braces: without a real image bound the sets
                      // were never written, so the flag must not be set either.
                      (p.depthAov && !tg.splatDepthViews.empty() ? kSplatFlagDepthAov : 0u) |
                      (p.depthMedian ? kSplatFlagDepthMed : 0u);

            const VkDeviceSize uboOff = uboStride_ * (p.target * kMaxClouds + c.slot);
            VmaAllocationInfo ui{};
            vmaGetAllocationInfo(ctx_.allocator(), uboBuf_[frame].alloc, &ui);
            std::memcpy(static_cast<uint8_t*>(ui.pMappedData) + uboOff, &u, sizeof(u));
            flushHostWrites(ctx_.allocator(), uboBuf_[frame].alloc, uboOff, sizeof(u));

            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1,
                                    &c.sets[p.target * framesInFlight_ + frame], 0, nullptr);

            // ── clear the frame's GPU-owned state ────────────────────────────
            // minDistBits starts at +inf's bit pattern so the atomicMin has
            // something to lose to; everything else at zero.
            uint32_t g0[kGlobalWords] = {};
            g0[0] = 0x7F7FFFFFu;// FLT_MAX
            g0[1] = 0u;
            vkCmdUpdateBuffer(cb, globalBuf_.handle, 0, sizeof(g0), g0);
            vkCmdFillBuffer(cb, rangeBuf_.handle, 0, VkDeviceSize(tg.tilesX) * tg.tilesY * 8, 0u);
            barrier(cb);

            const uint32_t radixBlocks = divUp(entryBudget_, kRadixBlock);
            SplatPc pc{};

            // ── project + cull + tile counts ─────────────────────────────────
            stageBegin(TP_SplatProject);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, projectPipe_);
            pc = {nSubmit, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
            barrier(cb);

            if (checksum) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, checksumPipe_);
                pc = {nSubmit, 0, 0, 0, 0, /*mode*/ 0, 0, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
                barrier(cb);
            }

            // ── prefix sum over the per-splat tile counts ────────────────────
            recordScan(cb, nSubmit, 0);

            if (checksum) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, checksumPipe_);
                pc = {nSubmit, 0, 0, 0, 0, /*mode*/ 2, 0, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
                barrier(cb);
            }

            // ── deterministic expansion ──────────────────────────────────────
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, expandPipe_);
            pc = {nSubmit, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
            barrier(cb);

            // ── size the sort from the data ──────────────────────────────────
            // One thread reads g.entryCount and writes the group counts the
            // radix and range dispatches below consume. Everything after the
            // expansion used to run over entryBudget_ instead, which measured
            // 7.9 ms of sorting on a frame with nothing on screen.
            // THREEPP_VK_SPLAT_NOINDIRECT restores the worst-case dispatches.
            // Same escape-hatch shape as NOMOTION/NOFOG: it is how the
            // indirect path was A/B'd for byte-identical output on a real
            // scan, and it is the first thing to try if some other driver
            // disagrees about vkCmdDispatchIndirect.
            const char* nie = std::getenv("THREEPP_VK_SPLAT_NOINDIRECT");
            const bool indirectDispatch = !(nie && *nie && *nie != '0');

            if (indirectDispatch) {

                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, indirectPipe_);
                pc = {entryBudget_, 0, 0, 0, 0, 0, 0, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, 1, 1, 1);
                barrierIndirect(cb);
            }

            // ── 8 x 4-bit LSD radix, ping-ponging A <-> B ────────────────────
            // The histogram is indexed [bin][block] with pc.arg1 = the WORST-CASE
            // block count, and the scan below still runs the worst-case extent
            // (recordScan's per-level offsets are host arithmetic). Only the
            // dispatches shrink — which is exactly why the histogram has to be
            // ZEROED first: the tail blocks used to run and write zero counts,
            // and now they do not run at all, so their bins would carry LAST
            // FRAME'S values into a scan that still reads them. Every offset
            // after the live region would be wrong, and the picture with it.
            stageEnd(TP_SplatProject);
            stageBegin(TP_SplatSort);
            for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
                const uint32_t shift = pass * 4;
                const uint32_t side  = pass & 1u;// 0: A->B, 1: B->A

                vkCmdFillBuffer(cb, histBuf_.handle, 0,
                                VkDeviceSize(kRadixBins) * radixBlocks * 4, 0u);
                barrier(cb);

                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, histPipe_);
                pc = {entryBudget_, 0, 0, 0, shift, radixBlocks, side, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                if (indirectDispatch) vkCmdDispatchIndirect(cb, indirectBuf_.handle, 0);
                else                  vkCmdDispatch(cb, radixBlocks, 1, 1);
                barrier(cb);

                recordScan(cb, kRadixBins * radixBlocks, 2);

                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scatterPipe_);
                pc = {entryBudget_, 0, 0, 0, shift, radixBlocks, side, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                if (indirectDispatch) vkCmdDispatchIndirect(cb, indirectBuf_.handle, 0);
                else                  vkCmdDispatch(cb, radixBlocks, 1, 1);
                barrier(cb);
            }

            stageEnd(TP_SplatSort);

            // ── tile ranges ──────────────────────────────────────────────────
            stageBegin(TP_SplatRaster);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rangePipe_);
            pc = {entryBudget_, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            // rangeBuf_ was zero-filled at the top of this cloud, so the tiles
            // this dispatch no longer reaches are empty rather than stale.
            if (indirectDispatch) vkCmdDispatchIndirect(cb, indirectBuf_.handle, 3 * sizeof(uint32_t));
            else                  vkCmdDispatch(cb, divUp(entryBudget_, 256), 1, 1);
            barrier(cb);

            if (checksum) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, checksumPipe_);
                for (uint32_t mode : {1u, 3u}) {
                    pc = {entryBudget_, 0, 0, 0, 0, mode, 0, 0};
                    vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cb, divUp(entryBudget_, 256), 1, 1);
                    barrier(cb);
                }
            }

            // ── composite ────────────────────────────────────────────────────
            // The motion attachment rests in SHADER_READ_ONLY_OPTIMAL (the
            // G-buffer render pass's finalLayout) and this pass writes it as a
            // STORAGE image, which only works in GENERAL. Flip it, dispatch,
            // flip it back — leaving it in GENERAL would be a silent lie to
            // every consumer downstream that samples it in the layout the
            // render pass promised.
            const bool flipMotion = p.motionVectors && tg.motionImages[frame] != VK_NULL_HANDLE;
            auto motionLayout = [&](VkImageLayout from, VkImageLayout to,
                                    VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
                VkImageMemoryBarrier2 b{};
                b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = srcAccess;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                b.dstAccessMask = dstAccess;
                b.oldLayout     = from;
                b.newLayout     = to;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image         = tg.motionImages[frame];
                b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                VkDependencyInfo di{};
                di.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                di.imageMemoryBarrierCount  = 1;
                di.pImageMemoryBarriers     = &b;
                vkCmdPipelineBarrier2(cb, &di);
            };
            if (flipMotion)
                motionLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rasterPipe_);
            pc = {0, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, tg.tilesX, tg.tilesY, 1);
            barrier(cb);
            stageEnd(TP_SplatRaster);
            stagesWritten = stagesWritten || timing;

            if (flipMotion)
                motionLayout(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            // PRIMARY TARGET ONLY. One readback buffer per frame slot, and a
            // secondary view records into the same one AFTER the primary — so
            // without this gate readDebug()/lastOverflow() would describe
            // whichever view happened to record last the moment any view opted
            // into splats, which is a silent change to what an existing test
            // measures.
            if (p.target == 0) {
                VkBufferCopy copy{0, 0, kGlobalWords * sizeof(uint32_t)};
                vkCmdCopyBuffer(cb, globalBuf_.handle, debugBuf_[frame].handle, 1, &copy);
            }
        }
    }

    void SplatPass::readDebug(uint64_t out[4]) const {
        out[0] = out[1] = out[2] = out[3] = 0;
        if (debugBuf_.empty()) return;
        // The device is drained so the last frame's copy has provably landed;
        // reading the slot that frame recorded into is then unambiguous.
        check(vkDeviceWaitIdle(ctx_.device()), "vkDeviceWaitIdle(splat readDebug)");
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[lastFrame_].alloc, &ai);
        invalidateHostReads(ctx_.allocator(), debugBuf_[lastFrame_].alloc);
        const auto* w = static_cast<const uint32_t*>(ai.pMappedData);
        if (!w) return;
        out[0] = w[7]; // hashKey
        out[1] = w[8]; // hashVal
        out[2] = w[9]; // hashColor
        out[3] = w[2]; // entryCount
    }

    std::vector<SplatPass::VolumeEntry> SplatPass::volumeEntries() const {
        std::vector<VolumeEntry> out;
        if (volumeOff_) return out;
        out.reserve(frameClouds_.size());
        for (const auto& fc : frameClouds_) {
            const Cloud& c = *fc.cloud;
            if (c.volume.view == VK_NULL_HANDLE) continue;
            VolumeEntry e{};
            e.view = c.volume.view;
            std::memcpy(e.model, fc.model, sizeof(e.model));
            std::memcpy(e.localBoxMin, c.localBoxMin, sizeof(e.localBoxMin));
            std::memcpy(e.localBoxSize, c.localBoxSize, sizeof(e.localBoxSize));
            e.voxelLocal = c.localBoxSize[0] / float(c.volRes[0]);
            for (int a = 1; a < 3; ++a)
                e.voxelLocal = std::min(e.voxelLocal, c.localBoxSize[a] / float(c.volRes[a]));
            e.count = c.count;
            out.push_back(e);
        }
        return out;
    }

    std::uint64_t SplatPass::volumeBytes() const {
        std::uint64_t bytes = 0;
        for (const auto& kv : resident_) {
            const Cloud& c = *kv.second;
            if (c.volume.image == VK_NULL_HANDLE) continue;
            bytes += std::uint64_t(c.volRes[0]) * c.volRes[1] * c.volRes[2] * kVolTexelBytes;
        }
        return bytes;
    }

    void SplatPass::readVolumeHash(std::uint64_t out[3]) const {
        out[0] = 0xcbf29ce484222325ull;// FNV-1a offset basis
        out[1] = out[2] = 0;
        if (volumeOff_) return;

        // Ascending SLOT order, not resident_ order: the map is unordered, so
        // without an imposed order the hash would not be stable even between
        // two reads of the same device state.
        std::vector<const Cloud*> ordered;
        for (const auto& kv : resident_)
            if (kv.second->volume.image != VK_NULL_HANDLE) ordered.push_back(kv.second.get());
        if (ordered.empty()) return;
        std::sort(ordered.begin(), ordered.end(),
                  [](const Cloud* a, const Cloud* b) { return a->slot < b->slot; });

        // The device is drained so every bake has provably landed; this is a
        // test accessor and never on the render path (readDebug's contract).
        check(vkDeviceWaitIdle(ctx_.device()), "vkDeviceWaitIdle(splat readVolumeHash)");

        for (const Cloud* c : ordered) {
            const VkDeviceSize voxels =
                    VkDeviceSize(c->volRes[0]) * c->volRes[1] * c->volRes[2];
            const VkDeviceSize bytes = voxels * kVolTexelBytes;
            Buffer rb = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
                                     VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                             VMA_ALLOCATION_CREATE_MAPPED_BIT);

            oneShot([&](VkCommandBuffer cb) {
                // The bake wrote this image in an EARLIER submission. Queue
                // order supplies the execution dependency; the memory one is
                // still this barrier's job, and a hash that reads a stale cache
                // line would look exactly like the non-determinism the test is
                // hunting for.
                VkImageMemoryBarrier2 ib{};
                ib.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                ib.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ib.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                ib.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
                ib.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                ib.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
                ib.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
                ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ib.image         = c->volume.image;
                ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                ib.subresourceRange.levelCount = 1;
                ib.subresourceRange.layerCount = 1;
                VkDependencyInfo di{};
                di.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                di.imageMemoryBarrierCount = 1;
                di.pImageMemoryBarriers    = &ib;
                vkCmdPipelineBarrier2(cb, &di);

                VkBufferImageCopy r{};
                r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                r.imageSubresource.layerCount = 1;
                r.imageExtent = {c->volRes[0], c->volRes[1], c->volRes[2]};
                // The volume lives in GENERAL, which is a legal source layout
                // for a copy — so nothing has to be transitioned back, and a
                // reader that stalls the device cannot leave the image in a
                // layout the next frame did not expect.
                vkCmdCopyImageToBuffer(cb, c->volume.image, VK_IMAGE_LAYOUT_GENERAL,
                                       rb.handle, 1, &r);
            });

            VmaAllocationInfo ai{};
            vmaGetAllocationInfo(ctx_.allocator(), rb.alloc, &ai);
            invalidateHostReads(ctx_.allocator(), rb.alloc);
            if (const auto* p = static_cast<const uint8_t*>(ai.pMappedData)) {
                std::uint64_t h = out[0];
                for (VkDeviceSize i = 0; i < bytes; ++i) {
                    h ^= p[i];
                    h *= 0x100000001b3ull;
                }
                out[0] = h;
                out[1] += voxels;
                // Alpha is sigma_t and sits in the last half of each rgba16f
                // texel; a zero there is a voxel no splat reached.
                for (VkDeviceSize v = 0; v < voxels; ++v)
                    if (p[v * kVolTexelBytes + 6] != 0 || p[v * kVolTexelBytes + 7] != 0) ++out[2];
            }
            destroyBuffer(ctx_.allocator(), rb);
        }
    }

    uint32_t SplatPass::lastOverflow() const {
        if (debugBuf_.empty()) return 0;
        uint32_t worst = 0;
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VmaAllocationInfo ai{};
            vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[f].alloc, &ai);
            if (const auto* w = static_cast<const uint32_t*>(ai.pMappedData)) worst = std::max(worst, w[3]);
        }
        return worst;
    }

}// namespace threepp::vulkan
