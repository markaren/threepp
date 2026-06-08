#include "threepp/renderers/vulkan/DeferredShade.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kMaxMaterialTextures

#include "threepp/renderers/vulkan/shaders/deferred_shade.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/deferred_denoise.comp.spv.h"

#include <array>
#include <cstring>

namespace threepp::vulkan {

    DeferredShade::DeferredShade(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        createPipeline();
        createDescriptorPool();
    }

    DeferredShade::~DeferredShade() {
        VkDevice d = ctx_.device();
        if (pipe_)        vkDestroyPipeline(d, pipe_, nullptr);
        if (denoisePipe_) vkDestroyPipeline(d, denoisePipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (gbufSampler_) vkDestroySampler(d, gbufSampler_, nullptr);
    }

    void DeferredShade::createPipeline() {
        VkDevice d = ctx_.device();

        // Nearest sampler for the G-buffer combined-image-sampler bindings. The
        // shader uses texelFetch (sampler ignored), but a valid sampler handle
        // is still required by the descriptor.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.f;
        check(vkCreateSampler(d, &sci, nullptr, &gbufSampler_), "vkCreateSampler(deferred)");

        VkDescriptorSetLayoutBinding b[25]{};
        auto set = [&](uint32_t i, VkDescriptorType t) {
            b[i].binding = i;
            b[i].descriptorType = t;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        set(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);          // camera
        set(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);          // lights
        set(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // env (PMREM)
        set(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf normal+rough
        set(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf depth
        set(5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf ids
        set(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // gbuf albedo+metal
        set(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);           // out sceneHdr
        set(8, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);// TLAS (shadow + reflection rays)
        set(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);          // MaterialDesc[] (emissive + reflected material)
        set(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // GeometryDesc[] (reflection-hit normals/UVs)
        set(11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // bindless material textures...
        b[11].descriptorCount = kMaxMaterialTextures;       // ...fixed-size array (reflection-hit textures)
        set(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // EmTri[] emissive triangles (area-light NEE)
        set(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // ocean FFT fine-cascade height (water chop)
        set(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // ocean world-space foam accumulator
        set(15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // gbuf uv (primary emissive-map sample)
        set(16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // demodulated diffuse-indirect (denoiser scratch)
        set(17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV indirect (other fif index) = 1-frame GI history
        set(18, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // gbuf motion (GI reproject)
        set(19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV gbuf normals (geometric GI disocclusion reset)
        set(20, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // momentsSq cur (SVGF E[L²] accumulator; denoise reads for variance)
        set(21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV momentsSq (other fif index) = 1-frame SVGF moment history
        set(22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // SVGF à-trous ping-pong A (rgb=GI, a=variance)
        set(23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // SVGF à-trous ping-pong B
        set(24, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV gbuf depth (geometric GI disocclusion: depth discontinuity)

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 25;
        dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(deferred)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = 40;// 10×u32 (…, fireflyClamp, oceanFineTileSize, oceanFoamTileSize)
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(deferred)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kDeferredShadeCompSpv);
        smci.pCode    = kDeferredShadeCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(deferred_shade)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(deferred_shade)");

        // Second pipeline — spatial denoise + recombine — shares the descriptor
        // set layout + push-constant range, just a different shader module.
        VkShaderModuleCreateInfo smciD{};
        smciD.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciD.codeSize = sizeof(kDeferredDenoiseCompSpv);
        smciD.pCode    = kDeferredDenoiseCompSpv;
        VkShaderModule modD = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciD, nullptr, &modD), "vkCreateShaderModule(deferred_denoise)");
        VkComputePipelineCreateInfo cpciD = cpci;
        cpciD.stage.module = modD;
        check(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cpciD, nullptr, &denoisePipe_),
              "vkCreateComputePipelines(deferred_denoise)");

        vkDestroyShaderModule(d, mod, nullptr);
        vkDestroyShaderModule(d, modD, nullptr);
    }

    void DeferredShade::createDescriptorPool() {
        VkDescriptorPoolSize sizes[5]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = framesInFlight_ * 2;// camera + lights
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_ * (13 + kMaxMaterialTextures);// env + 5 gbuf + 2 ocean + bindless + prevIndirect + motion + normalPrev + momentsSqPrev + depthPrev (GI reproject + SVGF + disocclusion)
        sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[2].descriptorCount = framesInFlight_ * 5;// out sceneHdr + indirect + momentsSq + atrousA + atrousB (SVGF multi-pass scratch)
        sizes[3].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        sizes[3].descriptorCount = framesInFlight_ * 1;// TLAS
        sizes[4].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[4].descriptorCount = framesInFlight_ * 3;// material + geometry + emissive-tri buffers

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 5;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(deferred)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets_.data()),
              "vkAllocateDescriptorSets(deferred)");
    }

    void DeferredShade::rewriteDescriptors(const DescriptorWriteInputs& in) {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorBufferInfo camInfo{};
            camInfo.buffer = in.cameraUbo[f];
            camInfo.offset = 0;
            camInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo lightInfo{};
            lightInfo.buffer = in.lightsUbo[f];
            lightInfo.offset = 0;
            lightInfo.range  = VK_WHOLE_SIZE;

            auto sampled = [&](VkImageView v, VkSampler s) {
                VkDescriptorImageInfo i{};
                i.sampler     = s;
                i.imageView   = v;
                i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                return i;
            };
            VkDescriptorImageInfo envInfo    = sampled(in.envView, in.envSampler);
            VkDescriptorImageInfo normalInfo = sampled(in.gbufNormal[f], gbufSampler_);
            VkDescriptorImageInfo idsInfo    = sampled(in.gbufIds[f], gbufSampler_);
            VkDescriptorImageInfo albInfo    = sampled(in.gbufAlbedo[f], gbufSampler_);
            VkDescriptorImageInfo uvInfo     = sampled(in.gbufUv[f], gbufSampler_);
            // Depth rests in DEPTH_STENCIL_READ_ONLY_OPTIMAL (the G-buffer render
            // pass's finalLayout for the depth attachment), not SHADER_READ_ONLY.
            VkDescriptorImageInfo depthInfo{};
            depthInfo.sampler     = gbufSampler_;
            depthInfo.imageView   = in.gbufDepth[f];
            depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo outInfo{};
            outInfo.imageView   = in.sceneHdr[f];
            outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo indInfo{};
            indInfo.imageView   = in.indirect[f];
            indInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo momCurInfo{};// SVGF E[L²] accumulator (this frame) — storage r/w
            momCurInfo.imageView   = in.momentsSq[f];
            momCurInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo atrAInfo{};// SVGF à-trous ping-pong A — storage
            atrAInfo.imageView   = in.atrousA[f];
            atrAInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo atrBInfo{};// SVGF à-trous ping-pong B — storage
            atrBInfo.imageView   = in.atrousB[f];
            atrBInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            // GI reproject inputs. prevIndirect = the OTHER frame-in-flight's
            // indirect image — holds last frame's accumulated GI (a 1-frame
            // history; the 2 per-frame indirect images alternate as a ping-pong).
            // Sampled in GENERAL (it's a storage image with SAMPLED usage added).
            // motion = this frame's motion vec; normalPrev = the other index's
            // world-space NORMALS (1-frame, for the GEOMETRIC disocclusion reset).
            // NOT prev-IDs: instanceCustomIndex is the per-frame draw-list index,
            // so it shifts whenever objects spawn/despawn (e.g. firing adds a
            // tracer → every ID renumbers → an ID-based disocclusion false-fires
            // globally → GI reset). World normals don't shift on re-sort and are
            // camera-independent (no false reset on camera motion either).
            const uint32_t pf = (f + 1u) % framesInFlight_;
            VkDescriptorImageInfo prevIndInfo{};
            prevIndInfo.sampler     = gbufSampler_;
            prevIndInfo.imageView   = in.indirect[pf];
            prevIndInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo prevMomInfo{};// PREV E[L²] (other fif) — sampled in GENERAL for the SVGF reproject
            prevMomInfo.sampler     = gbufSampler_;
            prevMomInfo.imageView   = in.momentsSq[pf];
            prevMomInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo depthPrevInfo{};// PREV depth (other fif) — depth-discontinuity disocclusion
            depthPrevInfo.sampler     = gbufSampler_;
            depthPrevInfo.imageView   = in.gbufDepth[pf];
            depthPrevInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            VkDescriptorImageInfo motionInfo     = sampled(in.gbufMotion[f], gbufSampler_);
            VkDescriptorImageInfo normalPrevInfo = sampled(in.gbufNormal[pf], gbufSampler_);

            // Ocean textures stay in GENERAL (written by the FFT/foam compute
            // passes, sampled here) — matching the RT set's bindings 32 + 44.
            VkDescriptorImageInfo oceanFineInfo{};
            oceanFineInfo.sampler     = in.oceanFineSampler;
            oceanFineInfo.imageView   = in.oceanFineView;
            oceanFineInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo oceanFoamInfo{};
            oceanFoamInfo.sampler     = in.oceanFoamSampler;
            oceanFoamInfo.imageView   = in.oceanFoamView;
            oceanFoamInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorBufferInfo matInfo{};
            matInfo.buffer = in.materialBuf[f];
            matInfo.offset = 0;
            matInfo.range  = VK_WHOLE_SIZE;

            VkDescriptorBufferInfo geomInfo{};
            geomInfo.buffer = in.geomDescBuf;// single buffer shared across frames
            geomInfo.offset = 0;
            geomInfo.range  = VK_WHOLE_SIZE;

            VkDescriptorBufferInfo emInfo{};
            emInfo.buffer = in.emissiveTriBuf[f];// per-frame (can grow → rewriteEmissive)
            emInfo.offset = 0;
            emInfo.range  = VK_WHOLE_SIZE;

            // TLAS for the shadow rays. The handle must outlive vkUpdateDescriptorSets,
            // so copy it locally and point the AS-write extension struct at it.
            VkAccelerationStructureKHR tlasLocal = in.tlas;
            VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
            asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asInfo.accelerationStructureCount = 1;
            asInfo.pAccelerationStructures = &tlasLocal;

            VkWriteDescriptorSet w[25]{};
            auto setw = [&](int n, uint32_t bind, VkDescriptorType t,
                            const VkDescriptorImageInfo* img, const VkDescriptorBufferInfo* buf) {
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet = sets_[f];
                w[n].dstBinding = bind;
                w[n].descriptorCount = 1;
                w[n].descriptorType = t;
                w[n].pImageInfo = img;
                w[n].pBufferInfo = buf;
            };
            setw(0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr, &camInfo);
            setw(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr, &lightInfo);
            setw(2, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo,    nullptr);
            setw(3, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalInfo, nullptr);
            setw(4, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo,  nullptr);
            setw(5, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &idsInfo,    nullptr);
            setw(6, 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albInfo,    nullptr);
            setw(7, 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &outInfo,    nullptr);
            setw(8, 8, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr);
            w[8].pNext = &asInfo;
            setw(9, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,        nullptr, &matInfo);
            setw(10, 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      nullptr, &geomInfo);
            // Bindless material-texture array — a single array write of the
            // whole array (descriptorCount = materialTexCount == kMaxMaterialTextures).
            w[11].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[11].dstSet          = sets_[f];
            w[11].dstBinding      = 11;
            w[11].dstArrayElement = 0;
            w[11].descriptorCount = in.materialTexCount;
            w[11].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[11].pImageInfo      = in.materialTex;
            setw(12, 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      nullptr, &emInfo);
            setw(13, 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &oceanFineInfo, nullptr);
            setw(14, 14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &oceanFoamInfo, nullptr);
            setw(15, 15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &uvInfo, nullptr);
            setw(16, 16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &indInfo, nullptr);
            setw(17, 17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevIndInfo, nullptr);
            setw(18, 18, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &motionInfo,  nullptr);
            setw(19, 19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalPrevInfo, nullptr);
            setw(20, 20, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &momCurInfo,    nullptr);
            setw(21, 21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevMomInfo,   nullptr);
            setw(22, 22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &atrAInfo,      nullptr);
            setw(23, 23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &atrBInfo,      nullptr);
            setw(24, 24, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthPrevInfo, nullptr);
            vkUpdateDescriptorSets(ctx_.device(), 25, w, 0, nullptr);
        }
    }

    void DeferredShade::rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf) {
        VkDescriptorBufferInfo info{};
        info.buffer = emissiveTriBuf;
        info.offset = 0;
        info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets_[frame];
        w.dstBinding = 12;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &info;
        vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
    }

    void DeferredShade::recordDispatch(VkCommandBuffer cb, uint32_t frame,
                                       uint32_t width, uint32_t height, uint32_t envMipCount,
                                       bool shadows, bool ao, uint32_t frameCounter,
                                       uint32_t emissiveCount, float emissiveTotalPower,
                                       float fireflyClamp,
                                       float oceanFineTileSize, float oceanFoamTileSize,
                                       bool denoise) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        const uint32_t flags = (shadows ? 1u : 0u) | (ao ? 2u : 0u) | (denoise ? 4u : 0u);
        uint32_t emPowerBits, fireflyBits, oceanFineBits, oceanFoamBits;
        std::memcpy(&emPowerBits,   &emissiveTotalPower, sizeof(emPowerBits));
        std::memcpy(&fireflyBits,   &fireflyClamp,       sizeof(fireflyBits));
        std::memcpy(&oceanFineBits, &oceanFineTileSize,  sizeof(oceanFineBits));
        std::memcpy(&oceanFoamBits, &oceanFoamTileSize,  sizeof(oceanFoamBits));
        const uint32_t pc[10] = {envMipCount, width, height, flags,
                                 frameCounter, emissiveCount, emPowerBits, fireflyBits,
                                 oceanFineBits, oceanFoamBits};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    }

    void DeferredShade::recordDenoiseDispatch(VkCommandBuffer cb, uint32_t frame,
                                              uint32_t width, uint32_t height) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, denoisePipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        // SVGF multi-pass à-trous wavelet: each pass is a 5×5 edge-stopping filter
        // at a WIDENING step (1,2,4,8 → reaches ±~30 px) that re-applies the edge
        // stops + re-filters variance, bouncing (rgb=GI, a=variance) between two
        // scratch images. This is what cleans large disoccluded regions (the motion
        // "cloud") + residual noise (patchy shadows) without softening real edges,
        // which a single fixed-width pass cannot. The shared 40-byte PC carries
        // [1]=width [2]=height [3]=step [4]=srcMode [5]=dstMode.
        // srcMode 0=indirect(raw), 1=atrousA, 2=atrousB.  dstMode 0=atrousA,
        // 1=atrousB, 2=recombine→sceneHdr.
        struct Pass { uint32_t step, srcMode, dstMode, feedback; };
        const Pass passes[4] = {
            {1u, 0u, 0u, 0u},// indirect → A  (step 1)
            {2u, 1u, 1u, 1u},// A → B  (step 2) + feed A (1st-pass filtered) back as temporal history
            {4u, 2u, 0u, 0u},// B → A  (step 4)
            {8u, 1u, 2u, 0u},// A → recombine into sceneHdr (step 8)
        };
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;// RAW (scratch) + WAR (history feedback writes indirect that pass 0 read)
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        for (int p = 0; p < 4; ++p) {
            const uint32_t pc[10] = {0u, width, height, passes[p].step,
                                     passes[p].srcMode, passes[p].dstMode, passes[p].feedback, 0u, 0u, 0u};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
            if (p < 3)// make this pass's scratch write visible to the next pass's read
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &mb, 0, nullptr, 0, nullptr);
        }
    }

}// namespace threepp::vulkan
