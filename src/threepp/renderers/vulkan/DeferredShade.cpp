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

        VkDescriptorSetLayoutBinding b[35]{};
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
        set(25, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reflect (sharp mirror-ray reflection radiance; shade writes, reflection denoise reads)
        set(26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV reflect (other fif index) = 1-frame reflection/glass history (temporal AA)
        // ReSTIR DI reservoir ping-pong (shared with the PT path's reservoir images).
        // 27/28 = lightPos+type write/read (rgba32f); 29/30 = W_sum/M/W/p_hat write/read
        // (rgba16f). Storage images (the PT images have no SAMPLED usage), GENERAL layout.
        set(27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir pos+type WRITE (this frame)
        set(28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir pos+type READ (prev frame, temporal reuse)
        set(29, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir W_sum/M/W/p_hat WRITE
        set(30, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir W_sum/M/W/p_hat READ
        set(31, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reflAux CUR (reflection-denoiser auxiliary; mirrors 25)
        set(32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV reflAux (other fif index) = 1-frame reflection-denoiser history (mirrors 26)
        set(33, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // scene fog UBO (shared with the PT path)
        set(34, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // tileable foam detail (R=bubbles, G=lace; mirrors RT binding 45)

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 35;
        dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(deferred)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = 64;// 16×u32 (…, starIntensity, camDelta, camRot, timeSec)
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
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &pipe_),
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
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciD, nullptr, &denoisePipe_),
              "vkCreateComputePipelines(deferred_denoise)");

        vkDestroyShaderModule(d, mod, nullptr);
        vkDestroyShaderModule(d, modD, nullptr);
    }

    void DeferredShade::createDescriptorPool() {
        VkDescriptorPoolSize sizes[5]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = framesInFlight_ * 3;// camera + lights + fog
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_ * (16 + kMaxMaterialTextures);// env + 5 gbuf + 2 ocean + foam detail + bindless + prevIndirect + motion + normalPrev + momentsSqPrev + depthPrev + reflectPrev + reflAuxPrev
        sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[2].descriptorCount = framesInFlight_ * 11;// sceneHdr + indirect + momentsSq + atrousA/B + reflect + reflAux + 4 reservoir (pos/W × write/read)
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
            VkDescriptorImageInfo reflInfo{};// sharp mirror-ray reflection radiance — storage
            reflInfo.imageView   = in.reflect[f];
            reflInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo prevReflInfo{};// PREV reflect (other fif) — sampled in GENERAL for temporal AA
            prevReflInfo.sampler     = gbufSampler_;
            prevReflInfo.imageView   = in.reflect[(f + 1u) % framesInFlight_];
            prevReflInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo reflAuxInfo{};// reflection-denoiser auxiliary — storage (mirrors reflInfo)
            reflAuxInfo.imageView   = in.reflAux[f];
            reflAuxInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo prevReflAuxInfo{};// PREV reflAux (other fif) — sampled in GENERAL (mirrors prevReflInfo)
            prevReflAuxInfo.sampler     = gbufSampler_;
            prevReflAuxInfo.imageView   = in.reflAux[(f + 1u) % framesInFlight_];
            prevReflAuxInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorBufferInfo fogInfo{};// scene fog (same UBO the PT consumes)
            fogInfo.buffer = in.fogBuf[f];
            fogInfo.offset = 0;
            fogInfo.range  = in.fogRange;

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
            // Baked foam detail tile — uploaded once at startup, stays in
            // SHADER_READ_ONLY (unlike the GENERAL-layout dynamic ocean images).
            VkDescriptorImageInfo foamDetailInfo{};
            foamDetailInfo.sampler     = in.foamDetailSampler;
            foamDetailInfo.imageView   = in.foamDetailView;
            foamDetailInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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

            // ReSTIR DI reservoir ping-pong: this frame WRITES slot (f&1) and READS the
            // other (last frame's write). Baked once at allocation — with 2 frames-in-
            // flight + 2 slots, consecutive frames alternate automatically (same trick as
            // the RT set). GENERAL layout (storage images, no SAMPLED usage).
            const uint32_t resWs = f & 1u, resRs = resWs ^ 1u;
            VkDescriptorImageInfo resPosWriteInfo{};
            resPosWriteInfo.imageView   = in.reservoirPos[resWs];
            resPosWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo resPosReadInfo{};
            resPosReadInfo.imageView   = in.reservoirPos[resRs];
            resPosReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo resWWriteInfo{};
            resWWriteInfo.imageView   = in.reservoirW[resWs];
            resWWriteInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo resWReadInfo{};
            resWReadInfo.imageView   = in.reservoirW[resRs];
            resWReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet w[35]{};
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
            setw(25, 25, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &reflInfo,      nullptr);
            setw(26, 26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevReflInfo,  nullptr);
            setw(27, 27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &resPosWriteInfo, nullptr);
            setw(28, 28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &resPosReadInfo,  nullptr);
            setw(29, 29, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &resWWriteInfo,   nullptr);
            setw(30, 30, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &resWReadInfo,    nullptr);
            setw(31, 31, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &reflAuxInfo,     nullptr);
            setw(32, 32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevReflAuxInfo, nullptr);
            setw(33, 33, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr,          &fogInfo);
            setw(34, 34, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &foamDetailInfo,  nullptr);
            vkUpdateDescriptorSets(ctx_.device(), 35, w, 0, nullptr);
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
                                       bool denoise, bool restirDI,
                                       float volDensity, float volAniso,
                                       float starIntensity,
                                       float camDeltaLen, float camRotAngle,
                                       float timeSec) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        const uint32_t flags = (shadows ? 1u : 0u) | (ao ? 2u : 0u) | (denoise ? 4u : 0u)
                             | (restirDI ? 8u : 0u);
        uint32_t emPowerBits, fireflyBits, oceanFineBits, oceanFoamBits, volDensBits, volAnisoBits, starBits,
                camDeltaBits, camRotBits, timeBits;
        std::memcpy(&emPowerBits,   &emissiveTotalPower, sizeof(emPowerBits));
        std::memcpy(&fireflyBits,   &fireflyClamp,       sizeof(fireflyBits));
        std::memcpy(&oceanFineBits, &oceanFineTileSize,  sizeof(oceanFineBits));
        std::memcpy(&oceanFoamBits, &oceanFoamTileSize,  sizeof(oceanFoamBits));
        std::memcpy(&volDensBits,   &volDensity,         sizeof(volDensBits));
        std::memcpy(&volAnisoBits,  &volAniso,           sizeof(volAnisoBits));
        std::memcpy(&starBits,      &starIntensity,      sizeof(starBits));
        std::memcpy(&camDeltaBits,  &camDeltaLen,        sizeof(camDeltaBits));
        std::memcpy(&camRotBits,    &camRotAngle,        sizeof(camRotBits));
        std::memcpy(&timeBits,      &timeSec,            sizeof(timeBits));
        const uint32_t pc[16] = {envMipCount, width, height, flags,
                                 frameCounter, emissiveCount, emPowerBits, fireflyBits,
                                 oceanFineBits, oceanFoamBits, volDensBits, volAnisoBits,
                                 starBits, camDeltaBits, camRotBits, timeBits};
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

        // ── Reflection denoise (channel = 1) ─────────────────────────────────
        // Roughness-guided edge-stopping blur of the sharp 1-mirror-ray reflection
        // (binding 25, written by the shade) + recombine × spec weight into
        // sceneHdr. ONE sharp ray (no discrete GGX samples) blurred by roughness =
        // a smooth glossy reflection with NO ghost copies. SEPARABLE: step 0
        // blurs horizontally into the atrousB scratch (free again — the GI
        // à-trous finished with it), step 1 blurs that vertically + despeckles +
        // recombines. Same Gaussian as the old dense grid at ≤2·25 instead of up
        // to 625 taps/pixel. Barrier 1: the last GI pass wrote sceneHdr (RAW for
        // the V pass's read-modify-write) and read atrousB the H pass overwrites
        // (WAR). Barrier 2: H's scratch write → V's read.
        const uint32_t rpcSep[2] = {0u /*H*/, 1u /*V*/};
        for (uint32_t s : rpcSep) {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);
            const uint32_t rpc[10] = {0u, width, height, s/*0=H,1=V*/, 0u, 0u, 0u, 1u/*channel=reflection*/, 0u, 0u};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpc), rpc);
            vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
        }
    }

}// namespace threepp::vulkan
