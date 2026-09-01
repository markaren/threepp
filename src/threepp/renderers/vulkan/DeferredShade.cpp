#include "threepp/renderers/vulkan/DeferredShade.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kMaxMaterialTextures

#include <cstdlib>// std::getenv (THREEPP_VK_SHADOW_DWELL kill switch)

#include "threepp/renderers/vulkan/shaders/deferred_shade.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/deferred_gi_filter.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/deferred_refl_filter.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/cluster_build.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/froxel_inject.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/froxel_integrate.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/cloud_march.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/cloud_shadow.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/particle_light.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/rtao.comp.spv.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace threepp::vulkan {

    // Number of descriptor bindings in the deferred set. Binding NUMBERS run
    // 0..71 with 56 retired, so the table has 71 entries and the highest
    // binding number is 71 — they are not the same quantity, which is exactly
    // how the two exact-fit stack arrays below used to drift: adding binding 67
    // to a 66-entry array wrote one element past the end (a silent stack smash
    // that no validation layer can see). Both tables are std::array of this
    // size now, filled through .at(), and the fill count is checked, so the
    // failure mode is a loud throw at init instead.
    constexpr uint32_t kDeferredBindingCount = 76;

    // ParticleField density volumes bound at once (binding 67 is an array of
    // this many). KEEP IN SYNC with kMaxDensityFields in
    // shaders/particle_density.glsl and ParticleFieldPass.hpp.
    constexpr uint32_t kMaxDensityVolumes = 4;

    // Splat reflection volumes bound at once (binding 70 is an array of this
    // many). KEEP IN SYNC with kMaxSplatVolumes in shaders/splat_volume.glsl
    // and SplatPass.hpp. Declared here rather than pulled in from SplatPass.hpp
    // for the same reason kMaxDensityVolumes is: this pass knows the size of an
    // array in its own layout, not the class that fills it.
    constexpr uint32_t kMaxSplatVolumeSlots = 8;

    DeferredShade::DeferredShade(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        createPipeline();
        createDescriptorPool();
    }

    DeferredShade::~DeferredShade() {
        VkDevice d = ctx_.device();
        if (pipe_)          vkDestroyPipeline(d, pipe_, nullptr);
        if (giFilterPipe_)   vkDestroyPipeline(d, giFilterPipe_, nullptr);
        if (reflFilterPipe_) vkDestroyPipeline(d, reflFilterPipe_, nullptr);
        if (clusterPipe_) vkDestroyPipeline(d, clusterPipe_, nullptr);
        if (froxelInjectPipe_)    vkDestroyPipeline(d, froxelInjectPipe_, nullptr);
        if (froxelIntegratePipe_) vkDestroyPipeline(d, froxelIntegratePipe_, nullptr);
        if (cloudMarchPipe_)      vkDestroyPipeline(d, cloudMarchPipe_, nullptr);
        if (cloudShadowPipe_)     vkDestroyPipeline(d, cloudShadowPipe_, nullptr);
        if (rtaoPipe_)            vkDestroyPipeline(d, rtaoPipe_, nullptr);
        if (particleLightPipe_)   vkDestroyPipeline(d, particleLightPipe_, nullptr);
        if (particlePipeLayout_)  vkDestroyPipelineLayout(d, particlePipeLayout_, nullptr);
        if (particleIoLayout_)    vkDestroyDescriptorSetLayout(d, particleIoLayout_, nullptr);
        if (lutSampler_) vkDestroySampler(d, lutSampler_, nullptr);
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

        // LINEAR clamp sampler for the froxel LUT — the trilinear filtering
        // ACROSS froxels is what turns the coarse grid into smooth beams.
        VkSamplerCreateInfo lci = sci;
        lci.magFilter  = VK_FILTER_LINEAR;
        lci.minFilter  = VK_FILTER_LINEAR;
        check(vkCreateSampler(d, &lci, nullptr, &lutSampler_), "vkCreateSampler(froxel LUT)");

        std::array<VkDescriptorSetLayoutBinding, kDeferredBindingCount> b{};
        // Dense cursor: array index and binding number diverge past the
        // retired 56 slot (55 is live — the prev-ids history; binding numbers
        // stay stable for the shaders, the array carries no zero-init gap
        // entries — those are invalid layout bindings, not padding).
        uint32_t nb = 0;
        auto set = [&](uint32_t binding, VkDescriptorType t) {
            auto& e = b.at(nb);// bound-checked: one slot per set() call, exact fit
            e.binding = binding;
            e.descriptorType = t;
            e.descriptorCount = 1;
            e.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            ++nb;
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
        b[nb - 1].descriptorCount = kMaxMaterialTextures;   // ...fixed-size array (reflection-hit textures)
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
        // ReSTIR DI reservoir ping-pong images.
        // 27/28 = lightPos+type write/read (rgba32f); 29/30 = W_sum/M/W/p_hat write/read
        // (rgba16f). Storage images (no SAMPLED usage), GENERAL layout.
        set(27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir pos+type WRITE (this frame)
        set(28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir pos+type READ (prev frame, temporal reuse)
        set(29, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir W_sum/M/W/p_hat WRITE
        set(30, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reservoir W_sum/M/W/p_hat READ
        set(31, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // reflAux CUR (reflection-denoiser auxiliary; mirrors 25)
        set(32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV reflAux (other fif index) = 1-frame reflection-denoiser history (mirrors 26)
        set(33, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // scene fog UBO
        set(34, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // tileable foam detail (R=bubbles, G=lace; mirrors RT binding 45)
        set(35, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // blue-noise tile (GI hemisphere dithering)
        set(36, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // probe SH-L1 store (ProbeGI, read)
        set(37, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // probe grid UBO (origin/spacing/dims + enable)
        // MSAA raw raster attachments (dispatch B / shadeMode==1 only —
        // always bound, harmlessly unused at msaa=1 via 1x1 dummy views,
        // same convention as the ocean/foam dummy textures above).
        set(38, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // gbuf normal MS
        set(39, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // gbuf depth MS
        set(40, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // gbuf ids MS
        set(41, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // gbuf albedo MS
        set(42, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // gbuf uv MS
        // Denoised direct-shadow channel (ratio estimator).
        set(43, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // shadow-ratio accumulator CUR (shade writes, denoise reads + feedback)
        set(44, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV shadow-ratio (other fif index) = 1-frame reproject history
        set(45, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // unshadowed analytic direct U (recombine multiplies by the filtered ratio)
        set(46, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // shadow-ratio à-trous ping-pong A (rg16f)
        set(47, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // shadow-ratio à-trous ping-pong B
        // Clustered lights (cluster_build.comp writes 48, the shade reads both).
        set(48, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // per-cell light index grid
        set(49, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // GpuClusterLight[] (power-sorted, uncapped)
        // Froxel volumetrics (inject writes 50 / reads 51; integrate 50→52; shade samples 53).
        set(50, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // froxel scatter CUR (3D)
        set(51, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV froxel scatter (other fif) — temporal EMA
        set(52, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // froxel LUT (3D, integrate writes)
        set(53, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // froxel LUT (LINEAR — shade's trilinear sample)
        set(54, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);         // probe Chebyshev depth store
        set(55, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV gbuf ids (other fif) — moving-mesh trailing-edge GI disocclusion
        // (56 retired — it carried a removed hybrid-SSR input.)
        set(57, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // raster camera UBO (cloud_march.comp's prevVP temporal reproject)
        set(58, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // volumetric cloud-layer UBO (setClouds)
        // Half-res cloud march (cloud_march.comp writes 59/62, reads 60/63 as
        // prev-fif history; the shade reads 61 = cur-fif cloud color, LINEAR).
        set(59, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // cloud color CUR (rgba16f half-res: rgb=in-scatter, a=T)
        set(60, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV cloud color (other fif) — temporal reproject
        set(61, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // cloud color CUR (LINEAR — shade's bilinear upsample)
        set(62, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // cloud aux CUR (rgba16f: r=mean depth, g=histLen, b=scene dist, a=epoch)
        set(63, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // PREV cloud aux (other fif) — temporal history
        // Cloud shadow map (cloud_shadow.comp writes 64; surface/froxel/water sun read 65).
        set(64, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // cloud shadow CUR (r8, storage write)
        set(65, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // cloud shadow (LINEAR — sun visibility sample)
        set(66, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // cloud aux CUR (shade's depth-aware upsample; texelFetch)
        // ParticleField density volumes (plan §3.3): a fixed-size array of
        // world-anchored r32ui 3D images sampled by mediumExtinction, plus the
        // std140 UBO carrying each one's world box. Integer format ⇒ NEAREST
        // only, which is what the manual trilinear in particle_density.glsl is
        // for; the sampler handle is the pass's NEAREST gbufSampler_.
        set(67, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // particle density volumes...
        b[nb - 1].descriptorCount = kMaxDensityVolumes;     // ...fixed-size array
        set(68, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // ParticleDensityUbo (boxes + medium albedo)
        // The r16f mirrors of the density volumes, LINEAR-sampled: the shade's
        // per-pixel dust march (applyParticleFog) takes 1 hardware-trilinear
        // fetch per step where the integer volume would force 8 texelFetches.
        set(69, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // particle density r16f mirrors...
        b[nb - 1].descriptorCount = kMaxDensityVolumes;     // ...same fixed-size array
        // Splat reflection volumes (plans/splat-volume-reflections.md): a
        // SECOND, parallel table — array of rgba16f 3D images baked once per
        // cloud by SplatPass, plus the std140 UBO that carries each one's
        // world->UVW matrix and conservative world AABB. rgba16f IS filterable,
        // so unlike the integer dust volumes above these are LINEAR-sampled
        // (lutSampler_) and the march is one hardware tap per step per volume.
        // Same always-bound contract: unused slots get the renderer's 1×1×1
        // dummy, so a scene with no splats is still a complete descriptor set.
        set(70, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // splat reflection volumes...
        b[nb - 1].descriptorCount = kMaxSplatVolumeSlots;   // ...fixed-size array
        set(71, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);         // SplatVolumeUbo (worldToUvw + world AABBs)

        // Half-res RT ambient occlusion + bent normals (rtao.comp). CUR is
        // written as a storage image by the RTAO pass and sampled by the shade's
        // bilateral upsample; PREV is the other FIF's result, sampled for the
        // temporal reprojection. Aux carries histLen / E[ao²] / near-field ao /
        // skyVis.
        set(72, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // rtao CUR (rtao.comp writes)
        set(73, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // rtao PREV (reproject tap)
        set(74, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // rtao CUR sampled (shade upsample)
        set(75, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // rtaoAux CUR (rtao.comp writes)
        set(76, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // rtaoAux PREV (history)

        // Exact fit is the contract: a new binding must bump
        // kDeferredBindingCount, and rewriteDescriptors must gain the matching
        // write. Under-filling would leave a zero-initialised (invalid) binding
        // in the layout, so this is checked in both directions.
        if (nb != b.size()) {
            throw std::runtime_error("[VulkanRenderer] deferred shade binding table "
                                     "does not match kDeferredBindingCount");
        }

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = nb;
        dlci.pBindings = b.data();
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(deferred)");

        // Derive the descriptor-pool sizes straight from the bindings above:
        // one entry per distinct type, summing descriptorCount × framesInFlight.
        // The bindless array at b[11] contributes kMaxMaterialTextures on its
        // own, so this reproduces the old hand-summed counts (uniform 6, sampler
        // 32+bindless, storage-image 20, AS 1, storage-buffer 7) but can never
        // fall out of step with the table as bindings are added.
        poolSizes_.clear();
        for (uint32_t i = 0; i < dlci.bindingCount; ++i) {
            const VkDescriptorType t = b[i].descriptorType;
            const uint32_t add = b[i].descriptorCount * framesInFlight_;
            auto it = std::find_if(poolSizes_.begin(), poolSizes_.end(),
                                   [t](const VkDescriptorPoolSize& s) { return s.type == t; });
            if (it == poolSizes_.end()) poolSizes_.push_back({t, add});
            else                        it->descriptorCount += add;
        }

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = 80;// 20×u32 (…, clusterLightCount, shadeMode, preExpBits)
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
        // Diagnostic mode only (THREEPP_VULKAN_PIPELINE_STATS): ask the driver
        // to keep per-executable statistics — register count, spill/scratch
        // bytes, occupancy — so a shader change can be judged on what it did to
        // the hardware rather than on wall-clock alone. The cache is bypassed
        // here because a cached pipeline has no statistics to report.
        const bool wantStats = ctx_.pipelineStatsEnabled();
        if (wantStats) cpci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        check(vkCreateComputePipelines(d, wantStats ? VK_NULL_HANDLE : ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(deferred_shade)");
        ctx_.dumpPipelineStats(pipe_, "deferred_shade");

        // Filter + composite pipelines — GI SVGF and reflection gloss
        // reconstruction (formerly the two channels of deferred_denoise). Both
        // share the descriptor set layout + push-constant range, just different
        // shader modules.
        VkShaderModuleCreateInfo smciGi{};
        smciGi.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciGi.codeSize = sizeof(kDeferredGiFilterCompSpv);
        smciGi.pCode    = kDeferredGiFilterCompSpv;
        VkShaderModule modGi = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciGi, nullptr, &modGi), "vkCreateShaderModule(deferred_gi_filter)");
        VkComputePipelineCreateInfo cpciGi = cpci;
        cpciGi.stage.module = modGi;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciGi, nullptr, &giFilterPipe_),
              "vkCreateComputePipelines(deferred_gi_filter)");

        VkShaderModuleCreateInfo smciRefl{};
        smciRefl.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciRefl.codeSize = sizeof(kDeferredReflFilterCompSpv);
        smciRefl.pCode    = kDeferredReflFilterCompSpv;
        VkShaderModule modRefl = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciRefl, nullptr, &modRefl), "vkCreateShaderModule(deferred_refl_filter)");
        VkComputePipelineCreateInfo cpciRefl = cpci;
        cpciRefl.stage.module = modRefl;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciRefl, nullptr, &reflFilterPipe_),
              "vkCreateComputePipelines(deferred_refl_filter)");

        // Third pipeline — clustered light culling — same layout again.
        VkShaderModuleCreateInfo smciC{};
        smciC.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciC.codeSize = sizeof(kClusterBuildCompSpv);
        smciC.pCode    = kClusterBuildCompSpv;
        VkShaderModule modC = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciC, nullptr, &modC), "vkCreateShaderModule(cluster_build)");
        VkComputePipelineCreateInfo cpciC = cpci;
        cpciC.stage.module = modC;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciC, nullptr, &clusterPipe_),
              "vkCreateComputePipelines(cluster_build)");

        // Froxel volumetrics — inject + integrate, same layout.
        VkShaderModuleCreateInfo smciF{};
        smciF.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciF.codeSize = sizeof(kFroxelInjectCompSpv);
        smciF.pCode    = kFroxelInjectCompSpv;
        VkShaderModule modF = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciF, nullptr, &modF), "vkCreateShaderModule(froxel_inject)");
        VkComputePipelineCreateInfo cpciF = cpci;
        cpciF.stage.module = modF;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciF, nullptr, &froxelInjectPipe_),
              "vkCreateComputePipelines(froxel_inject)");
        VkShaderModuleCreateInfo smciI{};
        smciI.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciI.codeSize = sizeof(kFroxelIntegrateCompSpv);
        smciI.pCode    = kFroxelIntegrateCompSpv;
        VkShaderModule modI = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciI, nullptr, &modI), "vkCreateShaderModule(froxel_integrate)");
        VkComputePipelineCreateInfo cpciI = cpci;
        cpciI.stage.module = modI;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciI, nullptr, &froxelIntegratePipe_),
              "vkCreateComputePipelines(froxel_integrate)");

        // Half-res cloud march — same layout again.
        VkShaderModuleCreateInfo smciM{};
        smciM.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciM.codeSize = sizeof(kCloudMarchCompSpv);
        smciM.pCode    = kCloudMarchCompSpv;
        VkShaderModule modM = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciM, nullptr, &modM), "vkCreateShaderModule(cloud_march)");
        VkComputePipelineCreateInfo cpciM = cpci;
        cpciM.stage.module = modM;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciM, nullptr, &cloudMarchPipe_),
              "vkCreateComputePipelines(cloud_march)");

        // Cloud shadow map — same layout again.
        VkShaderModuleCreateInfo smciS{};
        smciS.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smciS.codeSize = sizeof(kCloudShadowCompSpv);
        smciS.pCode    = kCloudShadowCompSpv;
        VkShaderModule modS = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciS, nullptr, &modS), "vkCreateShaderModule(cloud_shadow)");
        VkComputePipelineCreateInfo cpciS = cpci;
        cpciS.stage.module = modS;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciS, nullptr, &cloudShadowPipe_),
              "vkCreateComputePipelines(cloud_shadow)");

        // Half-res ray-traced AO + bent normals (rtao.comp) — same shared set 0
        // and 80-byte ShadePush range as the shade itself.
        VkShaderModuleCreateInfo smciAo = smciS;
        smciAo.codeSize = sizeof(kRtaoCompSpv);
        smciAo.pCode    = kRtaoCompSpv;
        VkShaderModule modAo = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smciAo, nullptr, &modAo), "vkCreateShaderModule(rtao)");
        VkComputePipelineCreateInfo cpciAo = cpci;
        cpciAo.stage.module = modAo;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciAo, nullptr, &rtaoPipe_),
              "vkCreateComputePipelines(rtao)");

        // Particle billboard lighting — set 0 is the shared deferred set; set 1
        // is the caller-owned particle IO pair (centers in, light/fog out), so
        // this one needs its own two-set pipeline layout + small push block.
        {
            VkDescriptorSetLayoutBinding iob[2]{};
            for (uint32_t i = 0; i < 2; ++i) {
                iob[i].binding         = i;
                iob[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                iob[i].descriptorCount = 1;
                iob[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo iolci{};
            iolci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            iolci.bindingCount = 2;
            iolci.pBindings    = iob;
            check(vkCreateDescriptorSetLayout(d, &iolci, nullptr, &particleIoLayout_),
                  "vkCreateDescriptorSetLayout(particleIo)");

            VkPushConstantRange ppc{};
            ppc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            ppc.offset     = 0;
            ppc.size       = 20;// count, centerBase, clusterLightCount, flags, envMipCount
            const VkDescriptorSetLayout pls[2] = {dsLayout_, particleIoLayout_};
            VkPipelineLayoutCreateInfo pplci{};
            pplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pplci.setLayoutCount         = 2;
            pplci.pSetLayouts            = pls;
            pplci.pushConstantRangeCount = 1;
            pplci.pPushConstantRanges    = &ppc;
            check(vkCreatePipelineLayout(d, &pplci, nullptr, &particlePipeLayout_),
                  "vkCreatePipelineLayout(particle_light)");

            VkShaderModuleCreateInfo smciP{};
            smciP.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smciP.codeSize = sizeof(kParticleLightCompSpv);
            smciP.pCode    = kParticleLightCompSpv;
            VkShaderModule modP = VK_NULL_HANDLE;
            check(vkCreateShaderModule(d, &smciP, nullptr, &modP), "vkCreateShaderModule(particle_light)");
            VkComputePipelineCreateInfo cpciP = cpci;
            cpciP.stage.module = modP;
            cpciP.layout       = particlePipeLayout_;
            check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpciP, nullptr, &particleLightPipe_),
                  "vkCreateComputePipelines(particle_light)");
            vkDestroyShaderModule(d, modP, nullptr);
        }

        vkDestroyShaderModule(d, mod, nullptr);
        vkDestroyShaderModule(d, modGi, nullptr);
        vkDestroyShaderModule(d, modRefl, nullptr);
        vkDestroyShaderModule(d, modC, nullptr);
        vkDestroyShaderModule(d, modF, nullptr);
        vkDestroyShaderModule(d, modI, nullptr);
        vkDestroyShaderModule(d, modM, nullptr);
        vkDestroyShaderModule(d, modS, nullptr);
        vkDestroyShaderModule(d, modAo, nullptr);
    }

    void DeferredShade::createDescriptorPool() {
        // poolSizes_ was derived from the set-layout bindings in createPipeline
        // (which runs first), so it always matches them exactly.
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes_.size());
        dpci.pPoolSizes    = poolSizes_.data();
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

    void DeferredShade::rewriteDescriptors(const DescriptorWriteInputs& in, int onlyFrame) {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            // Per-FIF refresh: skip every slot but the requested one. The "prev"
            // bindings below still read the OTHER slot's views (stable image
            // handles) — only THIS slot's descriptor SET is written, which is
            // safe because its fence has signaled.
            if (onlyFrame >= 0 && f != static_cast<uint32_t>(onlyFrame)) continue;
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
            VkDescriptorBufferInfo fogInfo{};// scene fog UBO
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
            // PREV ids (other fif) — the GI reproject's moving-mesh trailing-edge
            // guard. The ID-based disocclusion the comment above rejects stays
            // rejected: the guard compares the STABLE per-object id (.y) and the
            // prev texel's own moved-sticky bit (.z, kInstFlagMoving), both
            // identity-stable, so a draw-list renumber (topology rebuild, terrain
            // tile streaming) can't false-reset, and a static scene never fires
            // it (no moved bit).
            VkDescriptorImageInfo idsPrevInfo    = sampled(in.gbufIds[pf], gbufSampler_);

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
            geomInfo.buffer = in.geomDescBuf[f];// per-FIF ring (auto-LOD level switches flush per slot)
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

            // Blue-noise tile — NEAREST sampler (gbufSampler_): the shader computes
            // exact texel-center UVs, so no filtering is wanted. SHADER_READ_ONLY.
            VkDescriptorImageInfo bnInfo = sampled(in.blueNoise, gbufSampler_);

            // Probe GI (bindings 36/37/54) — SH store + per-frame grid UBO +
            // Chebyshev depth store.
            VkDescriptorBufferInfo probeShInfo{};
            probeShInfo.buffer = in.probeShBuf;
            probeShInfo.offset = 0;
            probeShInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo probeGridInfo{};
            probeGridInfo.buffer = in.probeGridUbo[f];
            probeGridInfo.offset = 0;
            probeGridInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo probeDepthInfo{};
            probeDepthInfo.buffer = in.probeDepthBuf;
            probeDepthInfo.offset = 0;
            probeDepthInfo.range  = VK_WHOLE_SIZE;

            // Raster camera UBO (binding 57) — cloud_march.comp reprojects
            // against last frame's view-proj (rcam.prevVP).
            VkDescriptorBufferInfo rcamInfo{};
            rcamInfo.buffer = in.rasterCamUbo[f];
            rcamInfo.offset = 0;
            rcamInfo.range  = VK_WHOLE_SIZE;

            VkDescriptorBufferInfo cloudInfo{};// volumetric cloud-layer UBO (binding 58)
            cloudInfo.buffer = in.cloudUbo[f];
            cloudInfo.offset = 0;
            cloudInfo.range  = in.cloudRange;

            // Half-res cloud march (bindings 59-63): CUR color/aux (storage
            // write), PREV color/aux (other fif, sampled LINEAR for the
            // reprojection tap), and CUR color again (sampled LINEAR — the
            // shade's bilinear upsample). All GENERAL (storage images).
            VkDescriptorImageInfo cloudColorInfo{};
            cloudColorInfo.imageView   = in.cloudColor[f];
            cloudColorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo cloudColorPrevInfo{};
            cloudColorPrevInfo.sampler     = lutSampler_;
            cloudColorPrevInfo.imageView   = in.cloudColor[pf];
            cloudColorPrevInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo cloudColorTexInfo{};
            cloudColorTexInfo.sampler     = lutSampler_;
            cloudColorTexInfo.imageView   = in.cloudColor[f];
            cloudColorTexInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo cloudAuxInfo{};
            cloudAuxInfo.imageView   = in.cloudAux[f];
            cloudAuxInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo cloudAuxPrevInfo{};
            cloudAuxPrevInfo.sampler     = lutSampler_;
            cloudAuxPrevInfo.imageView   = in.cloudAux[pf];
            cloudAuxPrevInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo cloudAuxTexInfo{};
            cloudAuxTexInfo.sampler     = lutSampler_;// texelFetch — filter irrelevant
            cloudAuxTexInfo.imageView   = in.cloudAux[f];
            cloudAuxTexInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            // Cloud shadow map (bindings 64/65): storage write (shadow pass) +
            // LINEAR sampled read (surface/froxel/water sun visibility). GENERAL.
            VkDescriptorImageInfo cloudShadowInfo{};
            cloudShadowInfo.imageView   = in.cloudShadow[f];
            cloudShadowInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo cloudShadowTexInfo{};
            cloudShadowTexInfo.sampler     = lutSampler_;
            cloudShadowTexInfo.imageView   = in.cloudShadow[f];
            cloudShadowTexInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            // MSAA raw raster attachments (dispatch B). Bound as plain
            // SHADER_READ_ONLY combined-image-samplers — texelFetch with an
            // explicit sample index needs no special layout beyond what
            // every other sampled G-buffer view already uses.
            VkDescriptorImageInfo normalMsInfo = sampled(in.gbufNormalMS[f], gbufSampler_);
            VkDescriptorImageInfo idsMsInfo    = sampled(in.gbufIdsMS[f],    gbufSampler_);
            VkDescriptorImageInfo albMsInfo    = sampled(in.gbufAlbedoMS[f], gbufSampler_);
            VkDescriptorImageInfo uvMsInfo     = sampled(in.gbufUvMS[f],     gbufSampler_);

            // Denoised direct-shadow channel. shadowVis ping-pongs across the
            // frames-in-flight exactly like indirect: CUR = storage write,
            // PREV = the other index sampled in GENERAL (reproject history).
            VkDescriptorImageInfo shadowVisInfo{};
            shadowVisInfo.imageView   = in.shadowVis[f];
            shadowVisInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo prevShadowVisInfo{};
            prevShadowVisInfo.sampler     = gbufSampler_;
            prevShadowVisInfo.imageView   = in.shadowVis[pf];
            prevShadowVisInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo directUInfo{};
            directUInfo.imageView   = in.directU[f];
            directUInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo shadowAtrAInfo{};
            shadowAtrAInfo.imageView   = in.shadowAtrousA[f];
            shadowAtrAInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo shadowAtrBInfo{};
            shadowAtrBInfo.imageView   = in.shadowAtrousB[f];
            shadowAtrBInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo depthMsInfo{};// depth MS rests DEPTH_STENCIL_READ_ONLY, like the resolved depth
            depthMsInfo.sampler     = gbufSampler_;
            depthMsInfo.imageView   = in.gbufDepthMS[f];
            depthMsInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            // Clustered lights: per-cell index grid + the uncapped light list.
            VkDescriptorBufferInfo clusterGridInfo{};
            clusterGridInfo.buffer = in.clusterGrid[f];
            clusterGridInfo.offset = 0;
            clusterGridInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo clusterLightsInfo{};
            clusterLightsInfo.buffer = in.clusterLights[f];
            clusterLightsInfo.offset = 0;
            clusterLightsInfo.range  = VK_WHOLE_SIZE;

            // Froxel volumetrics: scatter CUR (storage) + PREV (sampled, other
            // fif — temporal EMA) + LUT (storage write / LINEAR sampled read).
            VkDescriptorImageInfo froxelScatterInfo{};
            froxelScatterInfo.imageView   = in.froxelScatter[f];
            froxelScatterInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo froxelScatterPrevInfo{};
            froxelScatterPrevInfo.sampler     = gbufSampler_;
            froxelScatterPrevInfo.imageView   = in.froxelScatter[pf];
            froxelScatterPrevInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo froxelLutInfo{};
            froxelLutInfo.imageView   = in.froxelLut[f];
            froxelLutInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo froxelLutTexInfo{};
            froxelLutTexInfo.sampler     = lutSampler_;
            froxelLutTexInfo.imageView   = in.froxelLut[f];
            froxelLutTexInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            // One write per layout binding (same exact-fit contract as the
            // layout table in createPipeline); .at() turns an index that ran
            // past the end from a stack smash into a throw.
            std::array<VkWriteDescriptorSet, kDeferredBindingCount> w{};
            auto setw = [&](int n, uint32_t bind, VkDescriptorType t,
                            const VkDescriptorImageInfo* img, const VkDescriptorBufferInfo* buf) {
                auto& e = w.at(static_cast<size_t>(n));
                e.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                e.dstSet = sets_[f];
                e.dstBinding = bind;
                e.descriptorCount = 1;
                e.descriptorType = t;
                e.pImageInfo = img;
                e.pBufferInfo = buf;
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
            setw(35, 35, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &bnInfo,          nullptr);
            setw(36, 36, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr,          &probeShInfo);
            setw(37, 37, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr,          &probeGridInfo);
            setw(38, 38, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalMsInfo, nullptr);
            setw(39, 39, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthMsInfo,  nullptr);
            setw(40, 40, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &idsMsInfo,    nullptr);
            setw(41, 41, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albMsInfo,    nullptr);
            setw(42, 42, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &uvMsInfo,     nullptr);
            setw(43, 43, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &shadowVisInfo,     nullptr);
            setw(44, 44, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevShadowVisInfo, nullptr);
            setw(45, 45, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &directUInfo,       nullptr);
            setw(46, 46, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &shadowAtrAInfo,    nullptr);
            setw(47, 47, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &shadowAtrBInfo,    nullptr);
            setw(48, 48, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr, &clusterGridInfo);
            setw(49, 49, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr, &clusterLightsInfo);
            setw(50, 50, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &froxelScatterInfo,     nullptr);
            setw(51, 51, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &froxelScatterPrevInfo, nullptr);
            setw(52, 52, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &froxelLutInfo,         nullptr);
            setw(53, 53, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &froxelLutTexInfo,      nullptr);
            setw(54, 54, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr, &probeDepthInfo);
            // (binding 56 retired with the hybrid SSR — write indices stay
            // dense while binding numbers jump to 57; 55 = PREV ids, written
            // at index 65 below.)
            setw(55, 57, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr, &rcamInfo);
            setw(56, 58, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr, &cloudInfo);
            setw(57, 59, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cloudColorInfo,     nullptr);
            setw(58, 60, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudColorPrevInfo, nullptr);
            setw(59, 61, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudColorTexInfo,  nullptr);
            setw(60, 62, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cloudAuxInfo,       nullptr);
            setw(61, 63, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudAuxPrevInfo,   nullptr);
            setw(62, 64, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cloudShadowInfo,    nullptr);
            setw(63, 65, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudShadowTexInfo, nullptr);
            setw(64, 66, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudAuxTexInfo,    nullptr);
            setw(65, 55, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &idsPrevInfo,        nullptr);
            // ParticleField density volumes: the whole fixed-size array in one
            // write (every slot valid — the caller fills unused ones with its
            // 1×1×1 dummy), plus the per-FIF box UBO.
            std::array<VkDescriptorImageInfo, kMaxDensityVolumes> pdInfos{};
            for (uint32_t i = 0; i < kMaxDensityVolumes; ++i) {
                pdInfos[i].sampler     = gbufSampler_;
                pdInfos[i].imageView   = (in.particleDensity && i < in.particleDensityCount)
                                                 ? in.particleDensity[i]
                                                 : VK_NULL_HANDLE;
                pdInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            w[66].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[66].dstSet          = sets_[f];
            w[66].dstBinding      = 67;
            w[66].dstArrayElement = 0;
            w[66].descriptorCount = kMaxDensityVolumes;
            w[66].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[66].pImageInfo      = pdInfos.data();
            VkDescriptorBufferInfo pdUboInfo{};
            pdUboInfo.buffer = in.particleDensityUbo ? in.particleDensityUbo[f] : VK_NULL_HANDLE;
            pdUboInfo.offset = 0;
            pdUboInfo.range  = VK_WHOLE_SIZE;
            setw(67, 68, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &pdUboInfo);
            // The r16f mirrors — LINEAR (lutSampler_), unlike the integer
            // volumes above, which is the entire reason they exist.
            std::array<VkDescriptorImageInfo, kMaxDensityVolumes> pdLinInfos{};
            for (uint32_t i = 0; i < kMaxDensityVolumes; ++i) {
                pdLinInfos[i].sampler     = lutSampler_;
                pdLinInfos[i].imageView   = (in.particleDensityLin && i < in.particleDensityCount)
                                                    ? in.particleDensityLin[i]
                                                    : VK_NULL_HANDLE;
                pdLinInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            w[68].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[68].dstSet          = sets_[f];
            w[68].dstBinding      = 69;
            w[68].dstArrayElement = 0;
            w[68].descriptorCount = kMaxDensityVolumes;
            w[68].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[68].pImageInfo      = pdLinInfos.data();
            // Splat reflection volumes: same whole-array-in-one-write shape as
            // the density table, LINEAR (lutSampler_) because the baked volume
            // is rgba16f and svLeg wants one hardware trilinear tap per step.
            // GENERAL for their whole life (SplatPass::Cloud::volume) and the
            // dummy is created in GENERAL too, so one layout covers live and
            // unused slots alike.
            std::array<VkDescriptorImageInfo, kMaxSplatVolumeSlots> svInfos{};
            for (uint32_t i = 0; i < kMaxSplatVolumeSlots; ++i) {
                svInfos[i].sampler     = lutSampler_;
                svInfos[i].imageView   = (in.splatVolume && i < in.splatVolumeCount)
                                                 ? in.splatVolume[i]
                                                 : VK_NULL_HANDLE;
                svInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            w[69].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[69].dstSet          = sets_[f];
            w[69].dstBinding      = 70;
            w[69].dstArrayElement = 0;
            w[69].descriptorCount = kMaxSplatVolumeSlots;
            w[69].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[69].pImageInfo      = svInfos.data();
            VkDescriptorBufferInfo svUboInfo{};
            svUboInfo.buffer = in.splatVolumeUbo ? in.splatVolumeUbo[f] : VK_NULL_HANDLE;
            svUboInfo.offset = 0;
            svUboInfo.range  = VK_WHOLE_SIZE;
            setw(70, 71, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &svUboInfo);

            // Half-res RTAO (72-76). CUR pair are storage images the rtao pass
            // writes (no sampler); the PREV pair use lutSampler_ (LINEAR) for
            // the temporal reprojection tap; CUR-sampled uses gbufSampler_ —
            // the shade texelFetches it, so the filter is irrelevant and this
            // just matches the other gbuf taps. All GENERAL, like the clouds.
            VkDescriptorImageInfo rtaoCurInfo{};
            rtaoCurInfo.imageView   = in.rtao[f];
            rtaoCurInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo rtaoPrevInfo{};
            rtaoPrevInfo.sampler     = lutSampler_;
            rtaoPrevInfo.imageView   = in.rtao[pf];
            rtaoPrevInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo rtaoCurSampledInfo{};
            rtaoCurSampledInfo.sampler     = gbufSampler_;
            rtaoCurSampledInfo.imageView   = in.rtao[f];
            rtaoCurSampledInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo rtaoAuxCurInfo{};
            rtaoAuxCurInfo.imageView   = in.rtaoAux[f];
            rtaoAuxCurInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkDescriptorImageInfo rtaoAuxPrevInfo{};
            rtaoAuxPrevInfo.sampler     = lutSampler_;
            rtaoAuxPrevInfo.imageView   = in.rtaoAux[pf];
            rtaoAuxPrevInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            setw(71, 72, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &rtaoCurInfo,        nullptr);
            setw(72, 73, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rtaoPrevInfo,       nullptr);
            setw(73, 74, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rtaoCurSampledInfo, nullptr);
            setw(74, 75, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &rtaoAuxCurInfo,     nullptr);
            setw(75, 76, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rtaoAuxPrevInfo,    nullptr);
            vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
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

    void DeferredShade::recordDispatch(VkCommandBuffer cb, uint32_t frame, const DispatchParams& p) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        // gbufMsaaSamples packed into flags bits 5-6: 0=1x(default),1=2x,2=4x.
        // Bit 7 = dispatch B will run this frame: dispatch A reserves the
        // geometry-minority coverage weight for it; with bit 7 clear that
        // weight folds back into the dominant surface (no energy loss). Sky
        // minority coverage is ALWAYS blended by dispatch A itself.
        const uint32_t msaaCode = p.gbufMsaaSamples >= 4u ? 2u : (p.gbufMsaaSamples >= 2u ? 1u : 0u);
        // THREEPP_VK_SHADOW_DWELL=0: kill switch for the moving-caster shadow
        // dwell/top-up machinery (perf-triage instrument — if FPS recovers with
        // it set, the cost is the top-up RAYS; if not, it's deferred_shade
        // kernel growth/occupancy, which a runtime flag cannot undo).
        static const bool shadowDwellOff = [] {
            const char* e = std::getenv("THREEPP_VK_SHADOW_DWELL");
            return e && e[0] == '0';
        }();
        ShadePush push{};
        push.envMipCount        = p.envMipCount;
        push.width              = p.width;
        push.height             = p.height;
        push.flags = (p.shadows ? 1u : 0u) | (p.ao ? 2u : 0u) | (p.denoise ? 4u : 0u)
                   | (p.restirDI ? 8u : 0u) | (p.volFog ? 16u : 0u) | (msaaCode << 5u)
                   | (p.shadeBActive ? 128u : 0u)
                   | (p.froxelsActive ? 256u : 0u)  // froxel LUT valid this frame
                   | (shadowDwellOff ? 512u : 0u)   // bit 9: shadow dwell kill switch (was SSR)
                   | (p.bgIsSolidColor ? 1024u : 0u) // solid bg: sky store NOT pre-exposed
                   | (p.particleDensity ? 2048u : 0u) // bit 11: ParticleField dust live
                   | (p.splatVolume ? 4096u : 0u);    // bit 12: splat reflection volume live
        push.frame              = p.frameCounter;
        push.emissiveCount      = p.emissiveCount;
        push.emissiveTotalPower = p.emissiveTotalPower;
        push.fireflyClamp       = p.fireflyClamp;
        push.oceanFineTileSize  = p.oceanFineTileSize;
        push.oceanFoamTileSize  = p.oceanFoamTileSize;
        push.volDensity         = p.volDensity;
        push.volAniso           = p.volAniso;
        push.starIntensity      = p.starIntensity;
        push.camDelta           = p.camDeltaLen;
        push.camRot             = p.camRotAngle;
        push.timeSec            = p.timeSec;
        push.sunTanHalfAngle    = p.sunTanHalfAngle;
        push.clusterLightCount  = p.clusterLightCount;
        push.shadeMode          = p.shadeMode;
        push.preExpBits         = p.preExpBits;
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cb, (p.width + 7u) / 8u, (p.height + 7u) / 8u, 1);
    }

    void DeferredShade::recordClusterBuild(VkCommandBuffer cb, uint32_t frame,
                                           uint32_t lightCount,
                                           uint32_t width, uint32_t height) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, clusterPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        // Shared ShadePush block — the cull consumes width/height (tile
        // footprint) + clusterLightCount only; the rest ride as zeros.
        ShadePush push{};
        push.width             = width;
        push.height            = height;
        push.clusterLightCount = lightCount;
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        // One thread per cluster cell (16×8×24 = 3072, local_size_x = 64).
        vkCmdDispatch(cb, (16u * 8u * 24u + 63u) / 64u, 1, 1);
    }

    void DeferredShade::recordFroxels(VkCommandBuffer cb, uint32_t frame,
                                      uint32_t width, uint32_t height,
                                      bool volFog, float volDensity, float volAniso,
                                      uint32_t frameCounter,
                                      float camDeltaLen, float camRotAngle,
                                      uint32_t clusterLightCount) {
        // Shared ShadePush block — the froxel passes consume width/height
        // (cluster-cell mapping), the volFog flag, frame, the beam density/
        // anisotropy, the camera-motion history gates and the cluster count.
        ShadePush push{};
        push.width             = width;
        push.height            = height;
        push.flags             = volFog ? 16u : 0u;
        push.frame             = frameCounter;
        push.volDensity        = volDensity;
        push.volAniso          = volAniso;
        push.camDelta          = camDeltaLen;
        push.camRot            = camRotAngle;
        push.clusterLightCount = clusterLightCount;

        // Inject: one thread per froxel (128×72×64, local 4×4×4).
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, froxelInjectPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cb, 128u / 4u, 72u / 4u, 64u / 4u);

        // Inject's scatter writes → integrate's reads.
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);

        // Integrate: one thread per froxel column (128×72, local 8×8).
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, froxelIntegratePipe_);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cb, 128u / 8u, 72u / 8u, 1);
    }

    void DeferredShade::recordCloudMarch(VkCommandBuffer cb, uint32_t frame,
                                         uint32_t width, uint32_t height, uint32_t envMipCount,
                                         uint32_t frameCounter,
                                         float camDeltaLen, float camRotAngle) {
        // Shared ShadePush block — the cloud march consumes envMipCount
        // (ambient LOD), the FULL render extent (it derives half res + the
        // primary rays), frame (blue-noise jitter) and the camera-motion gates
        // (temporal history shortening). The rest ride as zeros.
        ShadePush push{};
        push.envMipCount = envMipCount;
        push.width       = width;
        push.height      = height;
        push.frame       = frameCounter;
        push.camDelta    = camDeltaLen;
        push.camRot      = camRotAngle;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cloudMarchPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        // One thread per HALF-res pixel (local 8×8).
        const uint32_t hw = (width + 1u) / 2u, hh = (height + 1u) / 2u;
        vkCmdDispatch(cb, (hw + 7u) / 8u, (hh + 7u) / 8u, 1);
    }

    void DeferredShade::recordCloudShadow(VkCommandBuffer cb, uint32_t frame, uint32_t frameCounter) {
        // Shared ShadePush block — the shadow pass consumes only frame (and
        // reads the cloud shell + sun from the UBOs). The rest ride as zeros.
        ShadePush push{};
        push.frame = frameCounter;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cloudShadowPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        // One thread per texel of the fixed 512² map (local 8×8).
        vkCmdDispatch(cb, 512u / 8u, 512u / 8u, 1);
    }

    void DeferredShade::recordRtao(VkCommandBuffer cb, uint32_t frame,
                                   uint32_t width, uint32_t height,
                                   uint32_t frameCounter) {
        // Shared ShadePush block — the RTAO pass consumes the FULL render
        // extent (it derives half res), frame (blue-noise jitter) and reads the
        // G-buffer + TLAS from the shared set. The rest ride as zeros. Same
        // field offsets the old raw uint32_t pc[19] used (width=1, height=2,
        // frame=4); ShadePush is that block, named.
        ShadePush push{};
        push.width  = width;
        push.height = height;
        push.frame  = frameCounter;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rtaoPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        // One thread per HALF-res pixel (local 8×8).
        const uint32_t hw = (width + 1u) / 2u, hh = (height + 1u) / 2u;
        vkCmdDispatch(cb, (hw + 7u) / 8u, (hh + 7u) / 8u, 1);
    }

    void DeferredShade::recordParticleLight(VkCommandBuffer cb, uint32_t frame,
                                            VkDescriptorSet ioSet,
                                            uint32_t count, uint32_t centerBase,
                                            uint32_t clusterLightCount,
                                            bool froxelsActive, uint32_t envMipCount) {
        if (count == 0 || ioSet == VK_NULL_HANDLE) return;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, particleLightPipe_);
        const VkDescriptorSet sets[2] = {sets_[frame], ioSet};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                particlePipeLayout_, 0, 2, sets, 0, nullptr);
        const uint32_t pc[5] = {count, centerBase, clusterLightCount,
                                froxelsActive ? 1u : 0u, envMipCount};
        vkCmdPushConstants(cb, particlePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), pc);
        // One thread per live particle (local_size_x = 64).
        vkCmdDispatch(cb, (count + 63u) / 64u, 1, 1);
    }

    void DeferredShade::recordFilterAndComposite(VkCommandBuffer cb, uint32_t frame,
                                                 uint32_t width, uint32_t height,
                                                 uint32_t gbufMsaaSamples, bool shadeBActive,
                                                 uint32_t preExpBits) {
        // GI SVGF filter + recombine (the à-trous passes below; count via
        // THREEPP_DENOISE_ATROUS_PASSES, default 4 — the wide passes
        // self-gate per pixel, see deferred_gi_filter.comp).
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, giFilterPipe_);
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
        // Slot [8] = msaaInfo: bits 0..2 = G-buffer MSAA sample count, bit 4 =
        // shade dispatch B ran. The RECOMBINE passes (GI dstMode 2 + reflection
        // V) weight their sceneHdr adds by the geometry coverage at complex
        // pixels — the shade pass stored only the coverage-weighted dominant
        // surface there, so a full-weight add was a bright rim on every edge.
        const uint32_t msaaInfo = (gbufMsaaSamples & 0x7u) | (shadeBActive ? 0x10u : 0u);
        // THREEPP_DENOISE_ATROUS_PASSES: 2..4, default 4 — the pass-count
        // knob. Each pass filters at a widening step (1,2,4,8), so dropping
        // tail passes halves the filter's reach each time: 4 passes cover
        // ±~30 px, 3 cover ±~14, 2 cover ±~6. The table derives from the
        // count instead of being hand-written: pass 0 always reads the raw
        // indirect (srcMode 0), pass 1 always carries the SVGF history
        // feedback — count-invariant because feedback writes the pass's
        // SOURCE (the first pass's filtered GI), not its destination — and
        // the LAST pass recombines into sceneHdr (dstMode 2; valid alongside
        // feedback, they touch different images). Floor is 2, not 1: a
        // single pass has no srcMode-1 pass to feed the filtered history
        // back, and losing that re-injection brings back the disocclusion
        // dust tail the feedback exists to remove.
        //   4 (default): {1,0,0,0} {2,1,1,1} {4,2,0,0} {8,1,2,0}
        //   3:           {1,0,0,0} {2,1,1,1} {4,2,2,0}
        //   2:           {1,0,0,0} {2,1,2,1}
        //
        // History (2026-08-19): the pass bench put this chain at 2.09 ms —
        // 37% of the ocean scene's GPU frame — and each dropped pass is
        // worth ~0.42 ms, so the default briefly went to 2. But the wide
        // steps are load-bearing exactly where lighting is a small bright
        // source in the dark (the emissive golden, the night fjord's cabin
        // lamps), so instead of a global cut the wide passes are now
        // SELF-GATING per pixel: deferred_gi_filter.comp skips the 25-tap
        // loop at step ≥ 4 wherever both channels' propagated variance says
        // the filter would be a near-identity, and passes the center
        // through. Sky-lit converged scenes pay ~2 real passes; lamp-lit
        // noise keeps the full ±30 px reach. The knob remains for A/B and
        // for forcing the old global behavior.
        static const int kAtrousPasses = [] {
            const char* e = std::getenv("THREEPP_DENOISE_ATROUS_PASSES");
            const int v = e ? std::atoi(e) : 4;
            return std::clamp(v, 2, 4);
        }();
        struct Pass { uint32_t step, srcMode, dstMode, feedback; };
        Pass passes[4]{};
        for (int p = 0; p < kAtrousPasses; ++p) {
            passes[p].step     = 1u << p;
            passes[p].srcMode  = (p == 0) ? 0u : ((p & 1) ? 1u : 2u);
            passes[p].dstMode  = (p == kAtrousPasses - 1) ? 2u : ((p & 1) ? 1u : 0u);
            passes[p].feedback = (p == 1) ? 1u : 0u;
        }
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;// RAW (scratch) + WAR (history feedback writes indirect that pass 0 read)
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        for (int p = 0; p < kAtrousPasses; ++p) {
            // Slot [0] = preExpBits: the recombine's sceneHdr adds bake the
            // same pre-exposure the shade pass stored with (1.0 legacy).
            const uint32_t pc[10] = {preExpBits, width, height, passes[p].step,
                                     passes[p].srcMode, passes[p].dstMode, passes[p].feedback, 0u, msaaInfo, 0u};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
            if (p < kAtrousPasses - 1)// make this pass's scratch write visible to the next pass's read
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &mb, 0, nullptr, 0, nullptr);
        }

        // ── Reflection gloss reconstruction + recombine ──────────────────────
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
        // Separate pipeline now (was channel 1 of the shared denoise module).
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, reflFilterPipe_);
        const uint32_t rpcSep[2] = {0u /*H*/, 1u /*V*/};
        for (uint32_t s : rpcSep) {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);
            // Slot [7] is the now-reserved channel field (the pipeline choice
            // replaced it); still pushed as 1 for continuity — the shader ignores it.
            const uint32_t rpc[10] = {preExpBits, width, height, s/*0=H,1=V*/, 0u, 0u, 0u, 1u/*reserved (ex-channel)*/, msaaInfo, 0u};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpc), rpc);
            vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
        }
    }

}// namespace threepp::vulkan
