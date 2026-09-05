// VulkanRenderer — deferred G-buffer renderer.
//
// Translation-unit layout:
//   vulkan/VulkanCoreImpl.hpp  — VulkanRenderer::Impl, the whole pImpl (TLAS/BLAS,
//                                scene fingerprint, raster G-buffer, deferred shade,
//                                TAA/bloom, fog, lights, skinning, ocean/water, LIDAR,
//                                env-PMREM, camera UBOs). Bodies live in the
//                                vulkan/VulkanCore*.cpp TUs (ctor/dtor in
//                                VulkanCoreInit.cpp); Impl's nested types live in
//                                vulkan/Vulkan{ImplCommon,GpuLayouts,GeometryState,
//                                SceneTypes,ViewContext}.hpp, aliased back into Impl.
//   VulkanRenderer.cpp (this)  — Impl::recordSceneDispatch (DeferredShade +
//                                auto-exposure) + the public method bodies.
//
// VulkanRenderer (deferred): the raster G-buffer supplies primary visibility;
//   deferred_shade.comp lights it analytically and adds ray-query accents
//   (shadows, reflections, AO/GI), denoised (SVGF) + TAA-resolved.

// VMA's implementation is compiled in vulkan/VmaImpl.cpp — defining
// VMA_IMPLEMENTATION here rebuilt the whole allocator on every edit to the
// public setters below.
#include "vulkan/VulkanCoreImpl.hpp"

#include "vulkan/VulkanCpuPhaseProf.hpp"

// stb_image_write - implementation is compiled in utils/StbImageWrite.cpp.
#include "stb_image_write.h"

namespace threepp {


    // Deferred scene dispatch. Called from recordCommandBuffer between the
    // shared G-buffer/AS head and the shared bloom/TAA tail.
    void VulkanRenderer::Impl::recordSceneDispatch(VkCommandBuffer cb, uint32_t setIdx,
                                                   VkExtent2D ext, VkExtent2D ptExt,
                                                   uint32_t exposureBits) {
        // ── VulkanRenderer deferred dispatch ───────────────────────────
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
            asbar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            asbar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            asbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            asbar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            // The AS half of this fence exists FOR ray query, so it is also
            // gated ON ray query: AS_READ paired with a COMPUTE stage is a
            // validation error when the rayQuery feature is off
            // (VUID-VkMemoryBarrier2-dstAccessMask-06256 — the validation gate
            // found this the first time the no-ray-query fallback actually
            // ran). Without ray query the deferred compute never touches an
            // acceleration structure, and the RT-pipeline consumers are fenced
            // by the AS_BUILD → RT_SHADER barriers at the build sites.
            if (ctx->rayQuerySupported()) {
                asbar.srcStageMask  |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                asbar.srcAccessMask |= VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                asbar.dstAccessMask |= VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            }
            VkDependencyInfo asdep{};
            asdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            asdep.memoryBarrierCount = 1;
            asdep.pMemoryBarriers = &asbar;
            vkCmdPipelineBarrier2(cb, &asdep);
        }
        // ── Probe-GI update (opt-in) ─────────────────────────────────────
        // Refresh a round-robin window of world-space irradiance probes
        // BEFORE the shade so this frame's gather taps a current grid.
        // Runs after the AS barrier above (the probe rays traverse the
        // same TLAS/BLAS). The grid UBO is re-uploaded every frame — its
        // `enabled` flag is what the shader-side sampling gates on.
        if (probeGI_) {
            // The UBO write is per view on purpose: it targets the
            // frame-in-flight slot every view of this frame shares, and the
            // content is identical, so it is idempotent — writing it once per
            // view costs a memcpy and keeps this independent of which view
            // records first.
            probeGI_->updateGridUbo(currentFrame, probeGIEnabled_);
            // The DISPATCH is once per FRAME, primary only. probe_update.comp
            // binds no camera anything: it is 2048 probes x 64 world-space ray
            // queries plus a snapshot copy, and running it again for a
            // secondary view did not make anything fresher — it advanced the
            // round-robin cursor and the ray seed (the shader's `frame`) once
            // per view, which made the grid at frame N a function of how many
            // cameras were attached. The gate is VulkanMultiView_test's
            // [probe] section. It stays HERE rather than moving to the frame
            // head because it must sit after the AS barrier above (the probe
            // rays traverse the TLAS the refit just built) and before the
            // primary's shade, and the primary records first.
            //
            // Safe against the early-outs: the debug-blit and events-only
            // returns in VulkanCoreRecord abort the whole record, secondaries
            // included, so there is no frame where a secondary would want a
            // grid the primary never updated.
            if (probeGIEnabled_ && !view().secondary) {
                if (probeGridDirty_) {
                    fitProbeGridToScene();
                    probeGridDirty_ = false;
                    // Grid moved → the UBO written above is stale; rewrite.
                    probeGI_->updateGridUbo(currentFrame, true);
                }
                // The bracket covers the snapshot copy recordDispatch opens
                // with as well as the probe rays themselves — they are one
                // cost and the copy is the half that was easy to forget.
                gpuTimings_->begin(cb, TP_ProbeGI, currentFrame);
                probeGI_->recordDispatch(cb, currentFrame,
                                         emissiveTriCountThisFrame_,
                                         emissiveTotalPowerThisFrame_,
                                         /*shadows=*/true, envImage.mipLevels);
                gpuTimings_->end(cb, TP_ProbeGI, currentFrame);
                // Probe SH writes → deferred shade reads (compute→compute).
                VkMemoryBarrier2 pbar{};
                pbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                pbar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                pbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                pbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                pbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo pdep{};
                pdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                pdep.memoryBarrierCount = 1;
                pdep.pMemoryBarriers = &pbar;
                vkCmdPipelineBarrier2(cb, &pdep);
            }
        }
        // The sample count THIS view's G-buffer was rastered at, not the
        // renderer-wide setting. A secondary view gets no MS framebuffer
        // (createRasterGbufImages gates it) and rasters through the 1x pass,
        // so under setGbufferMsaa(2|4) it has to shade as 1x. Handed the
        // renderer-wide count, the shade moved its reconstruction ray to an
        // MSAA sample position that the view's centre-sampled depth never
        // came from, which put the shadow-ray origin under every grazing
        // surface: a same-camera secondary's floor rendered half in shadow
        // (VulkanMultiView_test's msaa gate, 10.8 dB against the primary),
        // and the filters read coverage bits the 1x pass never packs.
        const uint32_t viewMsaaSamples =
                (view().rasterGbufs[currentFrame].framebufferMS != VK_NULL_HANDLE) ? gbufMsaaSamples_ : 1u;
        // Dispatch A always sees the TRUE sample count: even with
        // dispatch B off it must blend SKY-minority coverage itself
        // (every geometry/sky silhouette — the most visible edges).
        // shadeBActive (flags bit 7) tells it whether to additionally
        // reserve the geometry-minority weight for dispatch B or fold
        // it into the dominant surface.
        const bool shadeBActive = viewMsaaSamples > 1 && gbufShadeBEnabled_;
        // Clustered light culling: per-cell light lists for the shade's
        // analytic split (all point/spot lights, no 8-per-type cap).
        // Barrier: cull's grid writes → shade's reads (compute→compute).
        if (clusterLightCountThisFrame_ > 0) {
            view().deferredShade_->recordClusterBuild(cb, currentFrame,
                                               clusterLightCountThisFrame_,
                                               regionRenderExt_.width, regionRenderExt_.height);
            VkMemoryBarrier2 cbar{};
            cbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            cbar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            cbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            cbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            cbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo cdep{};
            cdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            cdep.memoryBarrierCount = 1;
            cdep.pMemoryBarriers = &cbar;
            vkCmdPipelineBarrier2(cb, &cdep);
        }
        // Cloud shadow map: top-down cloud transmittance regenerated each
        // frame, sampled by the surface/froxel/water sun terms below (moving
        // cloud shadows on the ground). Runs before the froxel + shade
        // passes; the barrier makes its write visible to their sampled reads.
        // Only when clouds are on (off = free / image-identical).
        if (cloudsEnabled_) {
            view().deferredShade_->recordCloudShadow(cb, currentFrame, sampleIndex);
            VkMemoryBarrier2 csBar{};
            csBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            csBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            csBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            csBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            csBar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo csDep{};
            csDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            csDep.memoryBarrierCount = 1;
            csDep.pMemoryBarriers = &csBar;
            vkCmdPipelineBarrier2(cb, &csDep);
        }
        // Froxel volumetrics: inject (RT sun shafts + clustered-light
        // beams, temporal EMA) + integrate (front-to-back LUT), whenever
        // a medium exists this frame. Runs AFTER the cluster barrier
        // (inject reads the cluster grid); the barrier below makes the
        // LUT visible to the shade's trilinear sample.
        // Phase 2: the froxels run whenever the unified AIR medium exists
        // (scene.fog OR setHeightFog — resolved into mediumActiveThisFrame_ by
        // updateFogUbo) or the explicit clear-air beam density is set.
        // …AND there is at least one clustered light: froxel_inject only ever
        // accumulates inside `if (pc.clusterLightCount > 0u)` (the sun term was
        // deliberately removed — it is owned by deferred_shade's per-pixel
        // volumetricDirScatter), so with a medium but no point/spot lights the
        // two dispatches converge the EMA onto an all-zero LUT. The same bool
        // feeds recordDispatch / recordParticleLight and clears shade flag bit
        // 256, whose froxelInscatter helpers already early-out to vec3(0) —
        // so skipping the passes is image-identical, not just cheap.
        // …and phase 2 adds the third way a medium can exist: a ParticleField
        // with a live density volume. It breaks BOTH halves of the gate above,
        // deliberately:
        //   • it is a medium that neither scene.fog nor setHeightFog declared
        //     (mediumActiveThisFrame_ knows nothing about it), and
        //   • it needs the passes even with ZERO clustered lights — the
        //     "converges onto an all-zero LUT" argument only holds for the
        //     in-scatter channel. Dust's headline product is the LUT's
        //     TRANSMITTANCE channel, which the surface path reads to attenuate
        //     everything behind the cloud, and that is non-trivial under a bare
        //     sun. Skipping the passes there would make dust invisible in
        //     exactly the scene plan §3.3 names as the trap.
        // Dust-free scenes evaluate this to the identical expression as before.
        const bool densityActive = particleDensityActiveThisFrame_;
        const bool froxelsActive = ((mediumActiveThisFrame_ || deferredVolDensity_ > 0.f) &&
                                    clusterLightCountThisFrame_ > 0) ||
                                   densityActive;
        if (froxelsActive) {
            gpuTimings_->begin(cb, TP_Froxel, currentFrame);
            view().deferredShade_->recordFroxels(cb, currentFrame,
                                          regionRenderExt_.width, regionRenderExt_.height,
                                          deferredVolFog_, deferredVolDensity_, deferredVolAniso_,
                                          sampleIndex,
                                          view().deferredCamDeltaLen_, deferredCamRotAngle_,
                                          clusterLightCountThisFrame_);
            gpuTimings_->end(cb, TP_Froxel, currentFrame);
            VkMemoryBarrier2 fbar{};
            fbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            fbar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            fbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            fbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            fbar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo fdep{};
            fdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            fdep.memoryBarrierCount = 1;
            fdep.pMemoryBarriers = &fbar;
            vkCmdPipelineBarrier2(cb, &fdep);
        }
        // Half-res volumetric cloud march (cloud_march.comp): raymarch the
        // cloud deck + temporally reproject at half res, off the per-pixel
        // shade critical path. Only when clouds are enabled (off = free /
        // image-identical). Reads the resolved G-buffer depth/ids (already
        // COMPUTE-visible via the raster pass dependency); the barrier below
        // makes its cloudColor + cloudAux writes visible to the shade's
        // depth-aware upsample.
        if (cloudsEnabled_) {
            view().deferredShade_->recordCloudMarch(cb, currentFrame,
                                             regionRenderExt_.width, regionRenderExt_.height,
                                             envImage.mipLevels, sampleIndex,
                                             view().deferredCamDeltaLen_, deferredCamRotAngle_);
            VkMemoryBarrier2 cldBar{};
            cldBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            cldBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            cldBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            cldBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            cldBar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo cldDep{};
            cldDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            cldDep.memoryBarrierCount = 1;
            cldDep.pMemoryBarriers = &cldBar;
            vkCmdPipelineBarrier2(cb, &cldDep);
        }
        // Half-res RT ambient occlusion + bent normals. Recorded AFTER the
        // probe-GI update (shares the TLAS) and BEFORE the shade, which
        // bilaterally upsamples rtao for its ambient / specular-occlusion and
        // (Phase B) its bent-normal probe+env diffuse indirect. The barrier
        // makes the pass's storage writes visible to the shade's sampled read,
        // same shape as the cloud barrier above.
        if (deferredAO_) {
            gpuTimings_->begin(cb, TP_Rtao, currentFrame);
            view().deferredShade_->recordRtao(cb, currentFrame,
                                              regionRenderExt_.width, regionRenderExt_.height,
                                              sampleIndex);
            gpuTimings_->end(cb, TP_Rtao, currentFrame);
            VkMemoryBarrier2 aoBar{};
            aoBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            aoBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            aoBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            aoBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            // BOTH read bits: the shade samples rtao (binding 74, sampler2D)
            // but reads rtaoAux (binding 75) via imageLoad on a readonly
            // image2D — a STORAGE read. A SAMPLED_READ-only mask leaves the
            // aux write unsynchronised against that load.
            aoBar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo aoDep{};
            aoDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            aoDep.memoryBarrierCount = 1;
            aoDep.pMemoryBarriers = &aoBar;
            vkCmdPipelineBarrier2(cb, &aoDep);
        }
        vulkan::DeferredShade::DispatchParams shadeParams{};
        shadeParams.width              = regionRenderExt_.width;
        shadeParams.height             = regionRenderExt_.height;
        shadeParams.envMipCount        = envImage.mipLevels;
        shadeParams.shadows            = true;
        shadeParams.ao                 = deferredAO_;
        shadeParams.frameCounter       = sampleIndex;
        shadeParams.emissiveCount      = emissiveTriCountThisFrame_;
        shadeParams.emissiveTotalPower = emissiveTotalPowerThisFrame_;
        shadeParams.fireflyClamp       = fireflyClamp_;
        shadeParams.oceanFineTileSize  = oceanFineTileSize;
        shadeParams.oceanFoamTileSize  = oceanFoamTileSize;
        shadeParams.denoise            = denoiseEnabled_;
        shadeParams.restirDI           = restirDIEnabled_;
        shadeParams.volFog             = deferredVolFog_;
        shadeParams.volDensity         = deferredVolDensity_;
        shadeParams.volAniso           = deferredVolAniso_;
        shadeParams.starIntensity      = deferredStarIntensity_;
        shadeParams.camDeltaLen        = view().deferredCamDeltaLen_;
        shadeParams.camRotAngle        = deferredCamRotAngle_;
        shadeParams.timeSec            = static_cast<float>(frameNowSec());
        shadeParams.sunTanHalfAngle    = std::tan(sunAngularRadiusDeg_ * 0.017453292519943295f);
        shadeParams.gbufMsaaSamples    = viewMsaaSamples;
        shadeParams.shadeMode          = 0u;// dispatch A
        shadeParams.shadeBActive       = shadeBActive;
        shadeParams.clusterLightCount  = clusterLightCountThisFrame_;
        shadeParams.froxelsActive      = froxelsActive;
        shadeParams.particleDensity    = densityActive;
        // ── Splats in the mirror: PRIMARY VIEW ONLY ──────────────────────────
        // The volume descriptors are world-anchored and shared by every view's
        // set (harmlessly — exactly like the particle density table). This FLAG
        // is what actually turns the svLeg marches on, and it is where the scope
        // wall is enforced: doc/vulkan_splats.md says splats "are invisible to
        // the RT sensors" and that secondary views "skip the pass entirely
        // rather than paint splats into a sensor AOV nobody asked for". A
        // secondary's water reflection is a sensor AOV, so a cloud appearing in
        // it would be a silent scope change and a golden risk. Clearing the flag
        // here leaves every secondary on the pre-change arithmetic textually.
        shadeParams.splatVolume        = splatVolumeActiveThisFrame_ && !view().secondary;
        shadeParams.preExpBits         = preExpBits_;
        shadeParams.bgIsSolidColor     = envIsBgColor;

        gpuTimings_->begin(cb, TP_DeferredShade, currentFrame);
        view().deferredShade_->recordDispatch(cb, currentFrame, shadeParams);
        gpuTimings_->end(cb, TP_DeferredShade, currentFrame);// pathTraceMs = deferred SHADE only

        // ── MSAA dispatch B: per-sample shading at complex (edge) pixels ──
        // Opt-in (gbufShadeBEnabled_, default false) and only when
        // setGbufferMsaa(2|4) is active. Reads dispatch A's outImage
        // write (imageLoad accumulate) and the raw MS G-buffer; needs a
        // compute->compute barrier on outImage between the two
        // dispatches (RAW: B reads what A just wrote) plus visibility for
        // the MS attachments (already satisfied — they've been
        // SHADER_READ_ONLY since the MSAA render pass's own subpass
        // dependency, unchanged since dispatch A started).
        if (shadeBActive) {
            VkMemoryBarrier2 shadeBar{};
            shadeBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            shadeBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            shadeBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            shadeBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            shadeBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo shadeDep{};
            shadeDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            shadeDep.memoryBarrierCount = 1;
            shadeDep.pMemoryBarriers    = &shadeBar;
            vkCmdPipelineBarrier2(cb, &shadeDep);

            // Dispatch B reuses dispatch A's params verbatim — only the mode
            // differs (and shadeBActive is trivially true when B itself runs).
            shadeParams.shadeMode    = 1u;
            shadeParams.shadeBActive = true;
            gpuTimings_->begin(cb, TP_ShadeB, currentFrame);
            view().deferredShade_->recordDispatch(cb, currentFrame, shadeParams);
            gpuTimings_->end(cb, TP_ShadeB, currentFrame);

            // Dispatch B's outImage write -> bloom/composite's read.
            VkMemoryBarrier2 postBar{};
            postBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            postBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            postBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            postBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            postBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo postDep{};
            postDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            postDep.memoryBarrierCount = 1;
            postDep.pMemoryBarriers    = &postBar;
            vkCmdPipelineBarrier2(cb, &postDep);
        }
        // Filter the demodulated lighting channels (GI SVGF + shadow ratio,
        // reflection gloss reconstruction) and composite them into sceneHdr.
        // Barrier: the shade wrote sceneHdr + the indirect image (both
        // GENERAL storage); the filter reads the indirect 5×5 neighbourhood
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
            view().deferredShade_->recordFilterAndComposite(cb, currentFrame, regionRenderExt_.width, regionRenderExt_.height,
                                                     viewMsaaSamples, shadeBActive, preExpBits_);
            gpuTimings_->end(cb, TP_Denoise, currentFrame);
        }
        // Auto-exposure: histogram over the final sceneHdr. sceneHdr writes
        // (deferred shade + optional denoise) are already visible via the
        // barriers above; bloom's leading barrier will also make them visible,
        // so this fits naturally in the gap. recordDispatch() inserts its own
        // fill→compute barrier to zero the SSBO before sampling.
        if (autoExposureEnabled_ && autoExposure_) {
            VkMemoryBarrier2 lumBar{};
            lumBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            lumBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            lumBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            lumBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            lumBar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo lumDep{};
            lumDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            lumDep.memoryBarrierCount = 1;
            lumDep.pMemoryBarriers    = &lumBar;
            vkCmdPipelineBarrier2(cb, &lumDep);
            autoExposure_->recordDispatch(cb, currentFrame,
                                         regionRenderExt_.width, regionRenderExt_.height,
                                         preExpHist_[currentFrame]);// meter un-bakes this
        }
        // ── Particle billboard lighting (particle_light.comp) ─────────
        // One thread per live overlay particle: the deferred light field
        // at the particle center + the camera→particle fog leg, written
        // to the per-FIF results SSBO the billboard pass reads. Every
        // input (TLAS, cluster grid, cloud shadow, froxel LUT, probe SH)
        // is already compute-visible via the barriers above. The leading
        // barrier orders this frame's write against the PREVIOUS frame's
        // vertex-stage reads of the same FIF slot buffer (WAR — cache-
        // visibility-free execution dependency would do, but keep the
        // access masks explicit); the trailing one hands the results to
        // the overlay pass's vertex fetches.
        if (particleLightCount_ > 0) {
            VkMemoryBarrier2 preBar{};
            preBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            preBar.srcStageMask  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            preBar.srcAccessMask = 0;// WAR: execution ordering suffices
            preBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            preBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo preDep{};
            preDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            preDep.memoryBarrierCount = 1;
            preDep.pMemoryBarriers    = &preBar;
            vkCmdPipelineBarrier2(cb, &preDep);

            view().deferredShade_->recordParticleLight(
                    cb, currentFrame, particleIoDescSets_[currentFrame],
                    particleLightCount_, /*centerBase=*/0u,
                    clusterLightCountThisFrame_, froxelsActive,
                    envImage.mipLevels);

            VkMemoryBarrier2 postBar{};
            postBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            postBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            postBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            postBar.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            postBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo postDep{};
            postDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            postDep.memoryBarrierCount = 1;
            postDep.pMemoryBarriers    = &postBar;
            vkCmdPipelineBarrier2(cb, &postDep);
        }
    }

    VulkanRenderer::VulkanRenderer(Canvas& canvas) {
        canvas.initWindow(GraphicsAPI::Vulkan);
        pimpl_ = std::make_unique<Impl>(canvas);
        // The user's animate lambda may call render()
        // multiple times in one iteration (e.g. main scene + HUD overlay
        // via threepp::HUD). Each render() opens or extends the in-flight
        // frame; the present is deferred to the canvas frame-end callback
        // so all draws land on the same swapchain image. See
        // Impl::endFrame() for the submit+present body.
        canvas.setFrameEndCallback([this] {
            if (pimpl_) pimpl_->endFrame();
        });

        // The window was created hidden (Canvas::initWindow, Vulkan branch) so
        // the seconds of device and pipeline setup do not stand in the taskbar
        // as a blank, unresponsive frame that the user clicks away from — the
        // click that used to leave the first real frame surfacing in the
        // background. Push one empty-scene frame through the normal path so the
        // window's first appearance carries real pixels (and the heaviest
        // pipelines are warm), then reveal it; GLFW_FOCUS_ON_SHOW takes the
        // foreground at that moment. A warmup failure is not fatal here: the
        // window is revealed regardless, and whatever is wrong will report from
        // the first real frame in the usual place.
        if (!canvas.headless()) {
            try {
                Scene warmup;
                PerspectiveCamera camera;
                render(warmup, camera);
                pimpl_->endFrame();
            } catch (...) {
                // fall through to the reveal
            }
            canvas.showWindow();
        }
    }

    VulkanRenderer::~VulkanRenderer() = default;

    void VulkanRenderer::render(Object3D& scene, Camera& camera) {
        const auto frameStart = std::chrono::high_resolution_clock::now();

        // ── Lens overscan ───────────────────────────────────────────────────
        // Widen the frustum for the duration of this render so the lens warp
        // has real geometry to gather into the output corners. Barrel
        // distortion maps scene points inward, so without this the corners can
        // only come from outside the rendered field and clamp to a smeared
        // edge. Scaling proj[0]/proj[5] widens about the centre RAY (which sits
        // at -proj[8]/-proj[9] in NDC regardless of proj[0]), so the principal
        // point does not move and the skew terms stay valid.
        //
        // The camera is restored before returning, so this is invisible to the
        // caller — but it does mean every consumer of the projection this frame
        // (culling, motion vectors, the G-buffer AOVs) sees the SAME widened
        // camera, which is what keeps colour and labels describing one lens.
        // updateCameraUbo divides the factor back out when it stashes the
        // intrinsics, so cameraIntrinsics() still reports the camera the user
        // configured rather than the one we rendered.
        const float overscan = core()->effectiveOverscan();
        const bool  applyOverscan = overscan > 1.0001f && !camera.is<OrthographicCamera>();
        float savedP0 = 0.f, savedP5 = 0.f;
        if (applyOverscan) {
            savedP0 = camera.projectionMatrix.elements[0];
            savedP5 = camera.projectionMatrix.elements[5];
            camera.projectionMatrix.elements[0] = savedP0 / overscan;
            camera.projectionMatrix.elements[5] = savedP5 / overscan;
        }
        struct OverscanRestore {
            Camera* cam;
            bool    active;
            float   p0, p5;
            ~OverscanRestore() {
                if (!active) return;
                cam->projectionMatrix.elements[0] = p0;
                cam->projectionMatrix.elements[5] = p5;
            }
        } overscanRestore{&camera, applyOverscan, savedP0, savedP5};

        // Resize predicate: has the CANVAS changed since we last looked, not
        // does the canvas disagree with the swapchain. The two can disagree
        // permanently — the platform grants the extent, and e.g. Windows
        // enforces a minimum window width on the hidden-window headless
        // fallback, so a 128-wide canvas gets a 232-wide swapchain no matter
        // how often it is recreated. Comparing against core()->size (pinned to
        // the swapchain extent) re-triggered a full swapchain + render-extent
        // recreate every frame in that state.
        const auto cur = core()->canvas.size();
        if (cur != core()->lastCanvasSize) {
            core()->lastCanvasSize = cur;
            core()->needsResize = true;
        }
        // Mirror Renderer-base tone-mapping state into the Impl so renderFrame
        // can push it as a single 16-byte block. Done every render() so users
        // can flip toneMapping / toneMappingExposure freely between frames.
        core()->toneMapping_         = toneMapping;
        core()->toneMappingExposure_ = toneMappingExposure;
        core()->autoClear_           = autoClear;
        // A split-screen secondary pane — a second perspective render() into a
        // scissor sub-rect while a frame is already in flight — composes
        // overlay-only (Points / Lines / Sprites) into that region and must NOT
        // run the scene-build pass. renderFrame already routes it to the
        // overlay-only path (its matching condition below), but the open frame's
        // command buffer still has the PRIMARY pane's TLAS + scene-desc buffers
        // bound: ensureSceneBuilt's structural-rebuild branch (a different scene
        // ⇒ snapshot mismatch ⇒ fullRebuild) tears those down mid-frame, which
        // invalidates the recording command buffer and surfaces as a device-lost
        // at the next buildTlas one-shot. OverlayPass::record updates the
        // scene/camera matrices itself, so skipping the build here is matrix-safe.
        const bool secondaryOverlayPane =
                core()->frameState_ != Impl::FrameState::Idle &&
                core()->scissorTest &&
                core()->scissor.z >= 1.f && core()->scissor.w >= 1.f;
        // Only the deferred-render-bound (perspective-camera) primary
        // render() call needs the scene-build pass — it populates
        // lastVisibleEntries_, the BLAS cache, motion bits and the per-mesh
        // fingerprint state the deferred shade pipeline reads. The HUD
        // pattern's second call (ortho camera over a separate HUD scene)
        // must not touch any of that, or it clobbers meshMovedBits_ /
        // lastVisibleEntries_ and the next deferred frame
        // cold-starts (visibly drops to ~1-spp quality).
        // The ortho overlay record path walks the HUD scene directly instead.
        // An ortho camera the user has declared a 3D view (see
        // setOrthographicSceneRendering) is deferred-render-bound like any
        // perspective one and DOES need the build — without it the TLAS,
        // material descs and visible-entry list this frame's shade reads would
        // be whatever the last perspective frame left behind.
        const bool orthoSceneView = core()->orthoSceneRender(camera);
        if ((!camera.is<OrthographicCamera>() || orthoSceneView) && !secondaryOverlayPane) {
            // A full-frame render while the PREVIOUS frame is still open (the
            // caller drives render() directly rather than from the canvas
            // frame-end callback, as the renderer tests do). renderFrame() below
            // finalizes it itself — "second full-frame perspective render" —
            // but that is too late to be safe: ensureSceneBuilt's
            // structural-rebuild branch destroys the TLAS/BLAS and their buffers,
            // and the open command buffer still has them BOUND. Destroying a
            // bound object invalidates the buffer, so the next vkCmd* on it and
            // the vkEndCommandBuffer inside endFrame() both fail, and the frame
            // dies as an uncaught throw far from the cause. vkDeviceWaitIdle in
            // the rebuild does not cover this: the buffer is open on the HOST,
            // not in flight, so there is nothing to wait for. Close it first,
            // and the rebuild is then free to destroy anything.
            //
            // Scoped to exactly the case that runs the scene build — the HUD
            // ortho pass and the split-screen secondary pane both skip this
            // block and must keep extending the open frame, not end it.
            if (core()->frameState_ != Impl::FrameState::Idle) core()->endFrame();
            const auto sceneStart = std::chrono::high_resolution_clock::now();
            core()->ensureSceneBuilt(scene, camera);
            // World-space Sprites (screenSpace == false) are drawn by the overlay
            // billboard pass, not the deferred/G-buffer path. Snapshot them each frame
            // with fresh world matrices (ensureSceneBuilt just ran
            // updateMatrixWorld) — independent of the snapshot/lean machinery,
            // since impact sprites move/spawn/expire every frame.
            core()->collectWorldSprites(scene);
            // Gaussian splat clouds, same sidecar treatment and the same
            // reason: a SplatCloud is not a kind the snapshot / BLAS machinery
            // has anything to say about. Also the last point in the frame that
            // still holds a Camera&, so the pass's camera parameters and the
            // per-cloud depth percentiles are stashed here.
            core()->collectSplatClouds(scene, camera);
            core()->pendingCpuEnsureSceneMs_ =
                    std::chrono::duration<float, std::milli>(
                            std::chrono::high_resolution_clock::now() - sceneStart)
                            .count();
        }
        core()->renderFrame(scene, camera);
        core()->gpuTimings_->setCpuFrameMs(
                std::chrono::duration<float, std::milli>(
                        std::chrono::high_resolution_clock::now() - frameStart)
                        .count());
        vulkan::cpuprof::Registry::get().endFrame();
        // Anything compiled this frame is in the cache object now and on disk
        // only at destruction, which a kill by PID never reaches. Cheap when
        // nothing changed; see the note on the method.
        core()->ctx->savePipelineCacheIfChanged();
    }

    WindowSize VulkanRenderer::size() const { return core()->size; }

    WindowSize VulkanRenderer::framebufferSize() const {
        auto* ctx = core()->ctx.get();
        if (!ctx || ctx->swapchainImages().empty()) return core()->size;
        const VkExtent2D ext = ctx->swapchainExtent();
        return {static_cast<int>(ext.width), static_cast<int>(ext.height)};
    }

    void VulkanRenderer::setSize(const std::pair<int, int>& s) {
        core()->size = WindowSize{s.first, s.second};
        core()->needsResize = true;
    }

    // Pixel ratio is a GL/canvas concept (logical-pixel → device-pixel scale).
    // The Vulkan swapchain is already sized in native device pixels, so there
    // is no logical domain to scale — the ratio is always 1 and the setter is
    // a warned no-op rather than misleading mutable state. Resolution scaling
    // goes through setRenderScale, the one supported lever.
    float VulkanRenderer::getTargetPixelRatio() const { return 1.f; }
    void VulkanRenderer::setPixelRatio(float) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[VulkanRenderer] setPixelRatio: unsupported (the swapchain is "
                         "native-pixel; use setRenderScale for resolution scaling) - "
                         "call ignored\n";
        }
    }

    void VulkanRenderer::setViewport(const Vector4& v) { core()->viewport = v; }
    void VulkanRenderer::setViewport(int x, int y, int w, int h) {
        core()->viewport.set(static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(w), static_cast<float>(h));
    }

    void VulkanRenderer::setScissor(const Vector4& v) { core()->scissor = v; }
    void VulkanRenderer::setScissor(int x, int y, int w, int h) {
        core()->scissor.set(static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(w), static_cast<float>(h));
    }
    void VulkanRenderer::setScissorTest(bool b) { core()->scissorTest = b; }

    void VulkanRenderer::setClearColor(const Color& c, float a) {
        core()->clearColor = c;
        core()->clearAlpha = a;
    }
    void VulkanRenderer::getClearColor(Color& target) const { target = core()->clearColor; }
    float VulkanRenderer::getClearAlpha() const { return core()->clearAlpha; }
    void VulkanRenderer::setClearAlpha(float a) { core()->clearAlpha = a; }

    void VulkanRenderer::clear(bool, bool, bool) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[VulkanRenderer] clear(): unsupported - the deferred pipeline "
                         "rewrites every attachment each render(); call ignored\n";
        }
    }

    // nullptr = "rendering to the default framebuffer" in three.js semantics —
    // accurate here, the swapchain is the only target this renderer has.
    RenderTarget* VulkanRenderer::getRenderTarget() { return nullptr; }
    void VulkanRenderer::setRenderTarget(RenderTarget* renderTarget, int, int) {
        if (!renderTarget) return;// null = default framebuffer — already the only mode
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[VulkanRenderer] setRenderTarget(): offscreen render targets are "
                         "unsupported (swapchain-only renderer); call ignored - use "
                         "readGBufferAOV/readRGBPixels for capture\n";
        }
    }

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
        // Size the write from the SWAPCHAIN extent, not size(). readRGBPixels
        // copies the presented swapchain image, so the buffer it returns is
        // framebufferSize()-shaped and nothing else. size() is the canvas's
        // idea of the window, which the platform is free to disagree with: a
        // 1920x1200 window on a 1920x1200 Windows desktop gets its client area
        // clamped to the work area (1920x1181 with a taskbar), and GLFW fires
        // no resize callback for a size it never changed. Sourcing w/h from
        // size() then handed stb a 1200-row stride over an 1181-row buffer —
        // a 19-row over-read straight off the end of the allocation.
        const auto sz = framebufferSize();
        const int  w  = sz.width();
        const int  h  = sz.height();
        // The invariant above is load-bearing (stb reads h*stride bytes with
        // no bound of its own), so check it rather than trust it.
        const auto expected = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
        if (w <= 0 || h <= 0 || pixels.size() != expected) {
            throw std::runtime_error(
                    "VulkanRenderer::writeFramebuffer: readback is " +
                    std::to_string(pixels.size()) + " bytes but " + std::to_string(w) + "x" +
                    std::to_string(h) + " RGB needs " + std::to_string(expected) +
                    "; refusing to write past the buffer");
        }
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
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx || ctx->swapchainImages().empty()) return {};

        const VkExtent2D ext   = ctx->swapchainExtent();
        const auto       w     = ext.width;
        const auto       h     = ext.height;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
        if (w == 0 || h == 0) return {};

        // The copy below is only legal when the swapchain was created with
        // TRANSFER_SRC usage (see VulkanContext::createSwapchain). Universally
        // available on desktop; fail loudly rather than issue an invalid copy.
        if (!ctx->swapchainSupportsTransferSrc()) {
            throw std::runtime_error(
                    "VulkanRenderer::readRGBPixels: the surface does not support "
                    "TRANSFER_SRC swapchain usage, so the presented image cannot be "
                    "copied out on this platform");
        }

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
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readRGBPixels)");
        // Invalidate AFTER mapping: vkInvalidateMappedMemoryRanges requires
        // the memory to be currently host mapped (ad-hoc-mapped staging — a
        // persistently-mapped buffer wouldn't care about the ordering).
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, bytes);
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

    uint32_t VulkanRenderer::addView(Camera& camera, int width, int height) {
        if (width <= 0 || height <= 0) return 0u;
        return core()->addViewImpl(camera,
                                   static_cast<uint32_t>(width),
                                   static_cast<uint32_t>(height));
    }

    bool VulkanRenderer::removeView(uint32_t handle) {
        return core()->removeViewImpl(handle);
    }

    bool VulkanRenderer::setViewCamera(uint32_t handle, Camera& camera) {
        return core()->setViewCameraImpl(handle, camera);
    }

    std::vector<unsigned char> VulkanRenderer::readViewRGBPixels(uint32_t handle) {
        return core()->readViewPixelsImpl(handle);
    }

    bool VulkanRenderer::setViewDisplayRect(uint32_t handle, int x, int y, int width, int height) {
        return core()->setViewDisplayRectImpl(handle, x, y, width, height);
    }

    bool VulkanRenderer::hideView(uint32_t handle) {
        return core()->setViewDisplayRectImpl(handle, 0, 0, 0, 0);
    }

    bool VulkanRenderer::viewSize(uint32_t handle, int& width, int& height) const {
        auto* v = const_cast<Impl*>(core())->findView(handle);
        if (!v) return false;
        width  = static_cast<int>(v->outExt.width);
        height = static_cast<int>(v->outExt.height);
        return true;
    }

    void VulkanRenderer::setSceneCaptureEnabled(bool enabled) {
        // Scene capture copies the mid-frame swapchain image into a staging
        // buffer (recordSceneCapture) — same TRANSFER_SRC precondition as
        // readRGBPixels. ctx may not exist yet (pre-first-render enable);
        // recordSceneCapture re-checks and skips with a warning in that case.
        if (enabled && core()->ctx && !core()->ctx->swapchainSupportsTransferSrc()) {
            throw std::runtime_error(
                    "VulkanRenderer::setSceneCaptureEnabled: the surface does not "
                    "support TRANSFER_SRC swapchain usage, so scene capture is "
                    "unavailable on this platform");
        }
        core()->sceneCaptureEnabled_ = enabled;
    }

    bool VulkanRenderer::sceneCaptureEnabled() const {
        return core()->sceneCaptureEnabled_;
    }

    std::vector<unsigned char> VulkanRenderer::readSceneRGBPixels() {
        auto& impl = *core();
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

        void* mapped = nullptr;
        if (vmaMapMemory(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, &mapped) != VK_SUCCESS) {
            return {};
        }
        // AFTER mapping — see readRGBPixels for the ordering rationale.
        vulkan::invalidateHostReads(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, 0, bytes);
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

    namespace {

        // Resample one G-buffer AOV through the active lens, so its pixels
        // line up with the (already warped) colour image. CPU-side because
        // readback is a synchronous, out-of-band call, not a per-frame cost —
        // and because the alternative, a warp pass per AOV format, is a lot of
        // plumbing for a path that already round-trips through host memory.
        //
        // NEAREST, always. Interpolating an instance id produces ids that
        // belong to no object; interpolating reversed-Z depth across a
        // silhouette produces surfaces that exist nowhere in the scene. Both
        // are worse than the sub-pixel error nearest sampling leaves behind,
        // and the colour path (which CAN interpolate, and does) is where
        // smoothness matters.
        //
        // Runs on a copy of the source: the warp is a gather, so writing in
        // place would feed already-warped texels back into later reads.
        // `normK` is (fx/W, fy/H, cx/W, cy/H) — the same normalized intrinsics
        // the shader gets. Normalized rather than in pixels precisely because
        // the AOV is at the render extent while the display warp runs at the
        // display extent, and both must describe the same lens.
        void warpAovForLens(const LensDistortion& lens, const float normK[4], float overscan,
                            std::vector<uint8_t>& buf,
                            uint32_t w, uint32_t h, uint32_t bpp) {
            if (w == 0u || h == 0u || bpp == 0u) return;
            const std::vector<uint8_t> src = buf;

            const float nkx = normK[0];
            const float nky = normK[1];
            const float nkz = normK[2];
            const float nkw = normK[3];
            if (std::abs(nkx) < 1e-8f || std::abs(nky) < 1e-8f) return;

            const auto fw = static_cast<float>(w);
            const auto fh = static_cast<float>(h);

            for (uint32_t y = 0; y < h; ++y) {
                for (uint32_t x = 0; x < w; ++x) {
                    const float u = (static_cast<float>(x) + 0.5f) / fw;
                    const float v = (static_cast<float>(y) + 0.5f) / fh;
                    // pixel -> normalized distorted -> normalized ideal -> pixel
                    const float xd = (u - nkz) / nkx;
                    const float yd = (v - nkw) / nky;
                    float xi = 0.f, yi = 0.f;
                    lensUndistort(lens, xd, yd, xi, yi);
                    float su = xi * nkx + nkz;
                    float sv = yi * nky + nkw;
                    // The AOV was rendered through the same overscanned
                    // frustum as the colour image, so shrink the ideal offset
                    // about the principal point exactly as sensor_image.comp
                    // does — otherwise labels and pixels describe fields of
                    // different widths.
                    if (overscan != 1.f) {
                        su = nkz + (su - nkz) / overscan;
                        sv = nkw + (sv - nkw) / overscan;
                    }

                    // Clamp to edge, matching the sampler the GPU warp uses.
                    auto sx = static_cast<int>(std::floor(su * fw));
                    auto sy = static_cast<int>(std::floor(sv * fh));
                    sx = std::clamp(sx, 0, static_cast<int>(w) - 1);
                    sy = std::clamp(sy, 0, static_cast<int>(h) - 1);

                    const size_t dstOff = (static_cast<size_t>(y) * w + x) * bpp;
                    const size_t srcOff = (static_cast<size_t>(sy) * w + sx) * bpp;
                    std::memcpy(buf.data() + dstOff, src.data() + srcOff, bpp);
                }
            }
        }

    }// namespace

    bool VulkanRenderer::readGBufferAOV(GBufferAOV aov, std::vector<uint8_t>& out,
                                        int& width, int& height, int& bytesPerPixel) {
        // Handle 0 is the primary — the whole body below is view-agnostic
        // apart from which G-buffer it reads.
        return readViewGBufferAOV(0u, aov, out, width, height, bytesPerPixel);
    }

    bool VulkanRenderer::readViewGBufferAOV(uint32_t viewHandle, GBufferAOV aov,
                                            std::vector<uint8_t>& out,
                                            int& width, int& height, int& bytesPerPixel) {
        // A batch of one — the batched form owns the machinery, so the two
        // entry points cannot drift apart.
        std::vector<AOVReadback> res;
        if (!readViewGBufferAOVs(viewHandle, {aov}, res) || res.empty()) return false;
        out           = std::move(res.front().data);
        width         = res.front().width;
        height        = res.front().height;
        bytesPerPixel = res.front().bytesPerPixel;
        return true;
    }

    bool VulkanRenderer::readGBufferAOVs(const std::vector<GBufferAOV>& aovs,
                                         std::vector<AOVReadback>& out) {
        return readViewGBufferAOVs(0u, aovs, out);
    }

    bool VulkanRenderer::readViewGBufferAOVs(uint32_t viewHandle,
                                             const std::vector<GBufferAOV>& aovs,
                                             std::vector<AOVReadback>& out) {
        out.clear();
        if (aovs.empty()) return false;
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx) return false;

        // Nothing has ever been rendered into the G-buffer until a frame has been
        // SUBMITTED: the attachments are allocated with the swapchain, so their
        // handles/extents look valid immediately and the checks below would happily
        // copy out uninitialised VRAM — plausible-looking noise labelled as depth /
        // normals / instance ids, the worst failure mode for the sensor pipelines
        // this API feeds. frameSerial_ advances once per submitted frame
        // (VulkanCoreFrame.cpp endFrame), and sceneBuilt_ says the scene walk has
        // produced entries at least once, so both together are the "a real frame's
        // contents are in there" predicate the header promises.
        if (impl.frameSerial_ == 0 || !impl.sceneBuilt_) return false;

        // The just-rendered G-buffer sits in the slot BEFORE the current one:
        // endFrame advances currentFrame after recording (VulkanCoreFrame.cpp),
        // so the freshest attachment contents are (currentFrame - 1) mod N —
        // the same slot arithmetic the fog history uses.
        // Which view's G-buffer. 0 = the primary; anything else must name a
        // live secondary, and a stale handle answers false rather than
        // silently falling back to the primary — a segmentation dataset
        // labelled with the wrong camera is worse than a missing frame.
        auto* src = viewHandle == 0u ? &impl.primaryView() : impl.findView(viewHandle);
        if (!src || src->rasterGbufs[0].width == 0) return false;

        const uint32_t n    = static_cast<uint32_t>(src->rasterGbufs.size());
        const uint32_t slot = (impl.currentFrame + n - 1u) % n;
        const auto& g       = src->rasterGbufs[slot];

        // Select each attachment, its aspect, and the layout it rests in after
        // a frame (the raster render pass' finalLayout; the MSAA resolve leaves
        // the resolved single-sample images in the same layouts). Depth carries
        // the depth aspect + DEPTH_STENCIL_READ_ONLY; every colour AOV is
        // SHADER_READ_ONLY. Unreadable requests are SKIPPED, not failed — the
        // batch delivers what it can and the caller matches entries by `aov`
        // (through the batch-of-one forwarder above, a skip surfaces as the
        // same `false` the single form always returned).
        struct Slot {
            GBufferAOV aov;
            const vulkan::Image2D* img;
            VkImageAspectFlags aspect;
            VkImageLayout restLayout;
            uint32_t bpp;
            VkDeviceSize offset;
        };
        std::vector<Slot> slots;
        slots.reserve(aovs.size());
        VkDeviceSize total = 0;
        for (const GBufferAOV aov : aovs) {
            bool dup = false;
            for (const auto& s : slots) dup = dup || s.aov == aov;
            if (dup) continue;// duplicates collapse to one read

            const vulkan::Image2D* img = nullptr;
            VkImageAspectFlags aspect  = VK_IMAGE_ASPECT_COLOR_BIT;
            VkImageLayout restLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            switch (aov) {
                case GBufferAOV::Depth:
                    img = &g.depth; aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                    restLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL; break;
                case GBufferAOV::Normal: img = &g.normal; break;
                case GBufferAOV::Motion: img = &g.motion; break;
                case GBufferAOV::Ids:    img = &g.ids;    break;
                case GBufferAOV::Albedo: img = &g.albedo; break;
                case GBufferAOV::SplatDepth:
                    // Skip rather than hand back the 1x1 placeholder: a caller
                    // who forgot setSplatDepthAov would otherwise get a
                    // successful read of a one-pixel image, which reads as
                    // "the frame had no splats in it" instead of "you never
                    // asked for this AOV".
                    //
                    // ALLOCATED, not "the app asked for it": the renderer turns
                    // the AOV on by itself for overlay occlusion (see the latch
                    // in collectSplatClouds), and that image is full-resolution
                    // and holds the real Median statistic. Gating on
                    // splatDepthAov() skipped exactly those frames, so a caller
                    // watching this read for evidence that the stamp was
                    // running got None from a scene where it WAS running — the
                    // 1x1 case is precisely what splatDepthAovAllocated()==false
                    // means, and it is still skipped.
                    if (!impl.splatDepthAovAllocated()) continue;
                    img = &g.splatDepth; restLayout = VK_IMAGE_LAYOUT_GENERAL; break;
            }
            if (!img || img->image == VK_NULL_HANDLE || img->width == 0 || img->height == 0) {
                continue;// no frame rendered yet
            }
            // Element size of the attachment format: RGBA16 (normal/motion/ids)
            // = 8, D32_SFLOAT (depth) and RGBA8_UNORM (albedo) = 4.
            uint32_t bpp = 4;
            if (img->format == VK_FORMAT_R16G16B16A16_SFLOAT ||
                img->format == VK_FORMAT_R16G16B16A16_UINT) {
                bpp = 8;
            }
            // Regions pack back to back; offsets align to 8, a multiple of both
            // texel sizes, which vkCmdCopyImageToBuffer's bufferOffset requires.
            total = (total + 7) & ~VkDeviceSize(7);
            slots.push_back({aov, img, aspect, restLayout, bpp, total});
            total += VkDeviceSize(img->width) * img->height * bpp;
        }
        if (slots.empty()) return false;

        // Wait so the last frame's writes to these attachments are complete and
        // no in-flight frame is still sampling them — same trade-off as
        // readRGBPixels, paid ONCE for the whole batch (this wait, not the
        // copies, is what made per-AOV reads expensive).
        vkDeviceWaitIdle(ctx->device());

        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), total,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readGBufferAOV)");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readGBufferAOV)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(readGBufferAOV)");

        // restLayout → TRANSFER_SRC for every image in ONE barrier call.
        // srcAccess 0 is safe: the prior vkDeviceWaitIdle already retired every
        // access, so these barriers only perform the layout transitions.
        std::vector<VkImageMemoryBarrier> toSrc(slots.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            auto& b                     = toSrc[i];
            b                           = VkImageMemoryBarrier{};
            b.sType                     = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout                 = slots[i].restLayout;
            b.newLayout                 = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcAccessMask             = 0;
            b.dstAccessMask             = VK_ACCESS_TRANSFER_READ_BIT;
            b.srcQueueFamilyIndex       = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex       = VK_QUEUE_FAMILY_IGNORED;
            b.image                     = slots[i].img->image;
            b.subresourceRange.aspectMask = slots[i].aspect;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
        }
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(toSrc.size()), toSrc.data());

        for (const auto& s : slots) {
            VkBufferImageCopy region{};
            region.bufferOffset                = s.offset;
            region.imageSubresource.aspectMask = s.aspect;
            region.imageSubresource.layerCount = 1;
            region.imageExtent                 = {s.img->width, s.img->height, 1};
            vkCmdCopyImageToBuffer(cb, s.img->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   staging.handle, 1, &region);
        }

        // Restore the resting layouts so the next frame's consumers (deferred
        // shade / TAA) find the attachments where they expect them.
        std::vector<VkImageMemoryBarrier> toRest = toSrc;
        for (size_t i = 0; i < toRest.size(); ++i) {
            toRest[i].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toRest[i].newLayout     = slots[i].restLayout;
            toRest[i].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            toRest[i].dstAccessMask = 0;
        }
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(toRest.size()), toRest.data());

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readGBufferAOV)");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readGBufferAOV)");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readGBufferAOV)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readGBufferAOV)");

        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readGBufferAOV)");
        // AFTER mapping — see readRGBPixels for the ordering rationale.
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, total);
        out.reserve(slots.size());
        for (const auto& s : slots) {
            AOVReadback r;
            r.aov              = s.aov;
            r.width            = static_cast<int>(s.img->width);
            r.height           = static_cast<int>(s.img->height);
            r.bytesPerPixel    = static_cast<int>(s.bpp);
            const size_t bytes = static_cast<size_t>(s.img->width) * s.img->height * s.bpp;
            r.data.resize(bytes);
            std::memcpy(r.data.data(), static_cast<const uint8_t*>(mapped) + s.offset, bytes);
            out.push_back(std::move(r));
        }
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);

        // The G-buffer is rendered through a pinhole; the displayed image is
        // not, once a lens is set (the RCAS stage warps it). Warp every AOV to
        // match, or the dataset ships distorted pixels with undistorted labels
        // — worse than shipping neither.
        if (impl.lens_.active()) {
            const float normK[4] = {0.5f * impl.projP0_, 0.5f * impl.projP5_,
                                    0.5f * (1.f - impl.projP8_), 0.5f * (1.f + impl.projP9_)};
            for (auto& r : out) {
                warpAovForLens(impl.lens_, normK, impl.effectiveOverscan(), r.data,
                               static_cast<uint32_t>(r.width), static_cast<uint32_t>(r.height),
                               static_cast<uint32_t>(r.bytesPerPixel));
            }
        }
        return true;
    }

    bool VulkanRenderer::readParticleDensityVolume(const ParticleField& field,
                                                   std::vector<uint32_t>& out,
                                                   uint32_t& resolution) {
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx || !impl.particleFieldPass_) return false;

        VkImage  image = VK_NULL_HANDLE;
        uint32_t res   = 0;
        if (!impl.particleFieldPass_->densityVolumeFor(field, image, res) || res == 0) return false;

        // Drain first: the volume is rewritten by the scatter dispatch of every
        // frame in flight, so a copy that raced one would read a half-splatted
        // cloud and the determinism assertion this exists for would be measuring
        // the race instead of the arithmetic.
        vkDeviceWaitIdle(ctx->device());

        const VkDeviceSize bytes = VkDeviceSize(res) * res * res * sizeof(uint32_t);
        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), bytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readParticleDensityVolume)");
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readParticleDensityVolume)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi),
                      "vkBeginCommandBuffer(readParticleDensityVolume)");

        // GENERAL is the volume's resting layout (the scatter leaves it there
        // and every froxel pass samples it there). srcAccess 0 is safe after the
        // drain above — this barrier only moves the layout.
        VkImageMemoryBarrier toSrc{};
        toSrc.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout                   = VK_IMAGE_LAYOUT_GENERAL;
        toSrc.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcAccessMask               = 0;
        toSrc.dstAccessMask               = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image                       = image;
        toSrc.subresourceRange            = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent                 = {res, res, res};
        vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.handle, 1, &region);

        VkImageMemoryBarrier toRest = toSrc;
        toRest.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toRest.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
        toRest.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
        toRest.dstAccessMask        = 0;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toRest);
        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readParticleDensityVolume)");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readParticleDensityVolume)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readParticleDensityVolume)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readParticleDensityVolume)");

        out.resize(static_cast<size_t>(res) * res * res);
        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readParticleDensityVolume)");
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, bytes);
        std::memcpy(out.data(), mapped, static_cast<size_t>(bytes));
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);
        resolution = res;
        return true;
    }

    void VulkanRenderer::setObjectInstanceId(const Object3D& obj, uint32_t instanceId) {
        core()->instanceIdOverride_[obj.id] = static_cast<uint16_t>(instanceId & 0xFFFFu);
        // stableId feeds DrawInfoGpu — invalidate the indirect-build skip
        // caches or an otherwise-static scene keeps serving the old id.
        ++core()->drawInputsVersion_;
    }

    void VulkanRenderer::setObjectClassId(const Object3D& obj, uint32_t classId) {
        core()->classIds_[obj.id] = static_cast<uint16_t>(classId > 255u ? 255u : classId);
        ++core()->drawInputsVersion_;// class id rides DrawInfoGpu::flags bits 8..15
    }

    void VulkanRenderer::setEventCameraEnabled(bool enabled) {
        auto& impl = *core();
        if (enabled == impl.eventCamEnabled_) return;
        impl.eventCamEnabled_ = enabled;
        impl.markMaterialSamplerDirty();// jitter gate flips → sampler policy flips
        if (enabled) {
            if (!impl.eventCam_) {
                impl.eventCam_ = std::make_unique<vulkan::EventCameraDetector>(*impl.ctx);
            }
            // The stream ring is sized for the per-pixel cap at resize(); a
            // cap set before enabling must reach the detector first.
            impl.eventCam_->setMaxEventsPerPixel(impl.eventCamParams_.maxEventsPerPixel);
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
            // resize() is a no-op at an unchanged size, which would carry the
            // reference + accumulator over from the previous enable (stale
            // reference → a burst on whatever changed while the camera was
            // off). Re-enabling always latches afresh: the documented "first
            // frame after enabling emits nothing".
            impl.eventCam_->resetReference();
            // Set up the source pipeline (event_shade.comp): the deterministic
            // G-buffer proxy or a box-average of the final swapchain frame,
            // per eventCamSource_. Either way the detector reads its output
            // (eventLumaBuf_), not the scene capture buffer, so sceneCapture
            // need not be enabled for the event camera to work.
            impl.createEventShadePipeline();
            impl.allocateEventLumaBuffer(w, h);
        }
    }

    bool VulkanRenderer::eventCameraEnabled() const {
        return core()->eventCamEnabled_;
    }

    void VulkanRenderer::setEventCameraSource(EventCameraSource source) {
        auto& impl = *core();
        if (source == impl.eventCamSource_) return;
        impl.eventCamSource_ = source;
        if (!impl.eventCamEnabled_) return;// takes effect when enabled
        // The jitter gate keys off the EFFECTIVE source (Shaded reads the raw
        // gbuf, Final does not) → the material sampler policy flips with it.
        impl.markMaterialSamplerDirty();
        // The proxy and the final frame have different luma; re-latch the
        // per-pixel reference on the next frame so the switch itself is not
        // a whole-frame burst of events.
        if (impl.eventCam_) impl.eventCam_->resetReference();
    }

    VulkanRenderer::EventCameraSource VulkanRenderer::eventCameraSource() const {
        return core()->eventCamSource_;
    }

    void VulkanRenderer::setEventCameraParams(const EventCameraParams& p) {
        auto& impl = *core();
        impl.eventCamParams_.threshold         = p.threshold;
        impl.eventCamParams_.decay             = p.decay;
        impl.eventCamParams_.minLuma           = p.minLuma;
        impl.eventCamParams_.maxEventsPerPixel = p.maxEventsPerPixel;
        impl.eventCamParams_.frameTimeUs       = p.frameTimeUs;
        // A larger per-pixel cap needs a larger stream ring (the ring is
        // sized so a frame can never overflow it). Reallocation destroys
        // buffers a pending submission may still bind, hence the idle wait,
        // taken only when growth is actually needed.
        if (impl.eventCam_) {
            if (impl.eventCam_->needsGrowthFor(p.maxEventsPerPixel)) {
                vkDeviceWaitIdle(impl.ctx->device());
            }
            impl.eventCam_->setMaxEventsPerPixel(p.maxEventsPerPixel);
        }
    }

    VulkanRenderer::EventCameraParams VulkanRenderer::eventCameraParams() const {
        const auto& src = core()->eventCamParams_;
        EventCameraParams p{};
        p.threshold         = src.threshold;
        p.decay             = src.decay;
        p.minLuma           = src.minLuma;
        p.maxEventsPerPixel = src.maxEventsPerPixel;
        p.frameTimeUs       = src.frameTimeUs;
        return p;
    }

    std::vector<unsigned char> VulkanRenderer::readEventCameraVisualisation() const {
        const auto& impl = *core();
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return {};
        return impl.eventCam_->readVisualisation();
    }

    size_t VulkanRenderer::readEventCameraVisualisationInto(unsigned char* dst, size_t cap) const {
        const auto& impl = *core();
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return 0;
        return impl.eventCam_->readVisualisationInto(dst, cap);
    }

    size_t VulkanRenderer::readEventStreamInto(Event* dst, size_t cap,
                                               bool* overflowed) const {
        const auto& impl = *core();
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
        auto& impl = *core();
        if (enabled == impl.eventsOnlyMode_) return;
        impl.eventsOnlyMode_ = enabled;
        // eventsOnly forces the Shaded source (no final frame exists to read),
        // which can flip the jitter gate and with it the sampler policy; also
        // re-latch the reference so the source change is not an event burst.
        if (impl.eventCamEnabled_ && impl.eventCamSource_ == EventCameraSource::Final) {
            impl.markMaterialSamplerDirty();
            if (impl.eventCam_) impl.eventCam_->resetReference();
        }
    }

    bool VulkanRenderer::eventsOnlyMode() const {
        return core()->eventsOnlyMode_;
    }

    void VulkanRenderer::setEventCameraResolution(uint32_t width, uint32_t height) {
        auto& impl = *core();
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
        const auto& impl = *core();
        if (!impl.eventCam_) return {impl.eventCamUserW_, impl.eventCamUserH_};
        return {impl.eventCam_->width(), impl.eventCam_->height()};
    }

    void VulkanRenderer::dispose() { pimpl_.reset(); }

    void* VulkanRenderer::nativeInstance() const {
        return static_cast<void*>(core()->ctx->instance());
    }
    void* VulkanRenderer::nativePhysicalDevice() const {
        return static_cast<void*>(core()->ctx->physicalDevice());
    }
    void* VulkanRenderer::nativeDevice() const {
        return static_cast<void*>(core()->ctx->device());
    }
    void* VulkanRenderer::nativeGraphicsQueue() const {
        return static_cast<void*>(core()->ctx->graphicsQueue());
    }
    uint32_t VulkanRenderer::graphicsQueueFamily() const {
        return core()->ctx->queueFamilies().graphics;
    }
    uint32_t VulkanRenderer::nativeSwapchainFormat() const {
        return static_cast<uint32_t>(core()->ctx->swapchainFormat());
    }
    uint32_t VulkanRenderer::imageCount() const {
        return static_cast<uint32_t>(core()->ctx->swapchainImages().size());
    }

    void VulkanRenderer::setOverlayCallback(std::function<void(void*)> callback) {
        core()->overlayCallback = std::move(callback);
    }

    void VulkanRenderer::setFogAnisotropy(float g) {
        g = std::max(-0.95f, std::min(g, 0.95f));
        if (g != core()->fogAnisotropy_) {
            core()->fogAnisotropy_ = g;
        }
    }

    float VulkanRenderer::getFogAnisotropy() const {
        return core()->fogAnisotropy_;
    }

    void VulkanRenderer::setFogWaterSurfaceY(float y) {
        core()->fogWaterSurfaceY_ = y;
    }

    void VulkanRenderer::setUnderwaterMurk(float density, const Color& color) {
        core()->murkDensity_  = density < 0.f ? 0.f : density;
        core()->murkColor_[0] = color.r;
        core()->murkColor_[1] = color.g;
        core()->murkColor_[2] = color.b;
    }

    std::pair<float, Color> VulkanRenderer::underwaterMurk() const {
        return {core()->murkDensity_,
                Color(core()->murkColor_[0], core()->murkColor_[1], core()->murkColor_[2])};
    }

    void VulkanRenderer::setRenderScale(float scale) {
        core()->setRenderScale(scale);
    }

    float VulkanRenderer::renderScale() const {
        return core()->renderScale_;
    }

    void VulkanRenderer::setOrthographicSceneRendering(bool enabled) {
        core()->orthoSceneRendering_ = enabled;
    }

    bool VulkanRenderer::orthographicSceneRendering() const {
        return core()->orthoSceneRendering_;
    }

    void VulkanRenderer::setFsr(bool enabled) {
        core()->setFsr(enabled);
    }

    bool VulkanRenderer::fsr() const {
        return core()->useFsr();
    }

    bool VulkanRenderer::fsrAvailable() const {
        return core()->fsrAvailable();
    }

    void VulkanRenderer::setDlss(bool enabled) {
        core()->setDlss(enabled);
    }

    bool VulkanRenderer::dlss() const {
        return core()->useDlss();
    }

    bool VulkanRenderer::dlssAvailable() const {
        return core()->dlssAvailable();
    }

    void VulkanRenderer::setTextureAnisotropy(float aniso) {
        core()->setTextureAnisotropy(aniso);
    }

    float VulkanRenderer::textureAnisotropy() const {
        return core()->textureAnisoOverride_;
    }

    bool VulkanRenderer::readTaaDebugImages(std::vector<uint8_t>& input, int& inW, int& inH,
                                            std::vector<uint8_t>& history, int& histW, int& histH) {
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx) return false;
        if (impl.frameSerial_ == 0 || !impl.sceneBuilt_) return false;
        auto* taa = impl.primaryView().taa_.get();
        if (!taa) return false;

        // Same slot arithmetic as readViewGBufferAOV: the last completed
        // frame's images live in the slot BEFORE currentFrame.
        const uint32_t slot = (impl.currentFrame + kFramesInFlight - 1u) % kFramesInFlight;
        const auto& in   = taa->inputImage2D(slot);
        const auto& hist = taa->historyImage2D(vulkan::TaaResolve::writeSlotFor(slot));
        if (in.image == VK_NULL_HANDLE || hist.image == VK_NULL_HANDLE) return false;

        vkDeviceWaitIdle(ctx->device());

        // Both images rest in GENERAL for their whole lives (TaaResolve
        // allocates and keeps them there), which is a legal transfer source —
        // no layout round-trip needed; one global barrier makes the retired
        // writes visible to the transfer stage.
        const VkDeviceSize inBytes   = static_cast<VkDeviceSize>(in.width) * in.height * 4;  // BGRA8
        const VkDeviceSize histBytes = static_cast<VkDeviceSize>(hist.width) * hist.height * 8;// RGBA16F

        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), inBytes + histBytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readTaaDebugImages)");
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readTaaDebugImages)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(readTaaDebugImages)");

        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent                 = {in.width, in.height, 1};
        vkCmdCopyImageToBuffer(cb, in.image, VK_IMAGE_LAYOUT_GENERAL, staging.handle, 1, &region);
        region.bufferOffset = inBytes;
        region.imageExtent  = {hist.width, hist.height, 1};
        vkCmdCopyImageToBuffer(cb, hist.image, VK_IMAGE_LAYOUT_GENERAL, staging.handle, 1, &region);

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readTaaDebugImages)");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readTaaDebugImages)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readTaaDebugImages)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readTaaDebugImages)");

        input.resize(static_cast<size_t>(inBytes));
        history.resize(static_cast<size_t>(histBytes));
        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readTaaDebugImages)");
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, inBytes + histBytes);
        std::memcpy(input.data(), mapped, static_cast<size_t>(inBytes));
        std::memcpy(history.data(), static_cast<const uint8_t*>(mapped) + inBytes,
                    static_cast<size_t>(histBytes));
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);

        inW   = static_cast<int>(in.width);
        inH   = static_cast<int>(in.height);
        histW = static_cast<int>(hist.width);
        histH = static_cast<int>(hist.height);
        return true;
    }

    bool VulkanRenderer::readSceneHdrDebug(std::vector<uint8_t>& hdr, int& w, int& h) {
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx) return false;
        if (impl.frameSerial_ == 0 || !impl.sceneBuilt_) return false;
        auto* bloom = impl.primaryView().bloom_.get();
        if (!bloom) return false;

        const uint32_t slot = (impl.currentFrame + kFramesInFlight - 1u) % kFramesInFlight;
        const auto& img = bloom->sceneHdrImage2D(slot);
        if (img.image == VK_NULL_HANDLE || img.width == 0) return false;

        vkDeviceWaitIdle(ctx->device());

        const VkDeviceSize bytes = static_cast<VkDeviceSize>(img.width) * img.height * 8;// RGBA16F
        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), bytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readSceneHdrDebug)");
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readSceneHdrDebug)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(readSceneHdrDebug)");

        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent                 = {img.width, img.height, 1};
        vkCmdCopyImageToBuffer(cb, img.image, VK_IMAGE_LAYOUT_GENERAL, staging.handle, 1, &region);

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readSceneHdrDebug)");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readSceneHdrDebug)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readSceneHdrDebug)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readSceneHdrDebug)");

        hdr.resize(static_cast<size_t>(bytes));
        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readSceneHdrDebug)");
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, bytes);
        std::memcpy(hdr.data(), mapped, static_cast<size_t>(bytes));
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);

        w = static_cast<int>(img.width);
        h = static_cast<int>(img.height);
        return true;
    }

    std::vector<std::pair<std::string, uint64_t>> VulkanRenderer::debugHashShadeImages() {
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx) return {};
        if (impl.frameSerial_ == 0 || !impl.sceneBuilt_) return {};

        const uint32_t n    = static_cast<uint32_t>(impl.primaryView().rasterGbufs.size());
        const uint32_t slot = (impl.currentFrame + n - 1u) % n;
        auto& g             = impl.primaryView().rasterGbufs[slot];

        // All of these are compute storage images that live in GENERAL for
        // their whole lives; every format here is 8 bytes/px (rgba16f or
        // rg32f). The second row is the state the shade carries but the six
        // classic outputs do not show: the RT AO pair, the GI filter's scratch
        // ping-pong, and the ReSTIR reservoir weights in both parity slots.
        // A divergence that surfaces in shadowVis at frame k but began in a
        // reservoir at frame k-1 is only visible with these in the split.
        auto& pv = impl.primaryView();
        const std::pair<const char*, const vulkan::Image2D*> imgs[] = {
                {"indirect", &g.indirect},   {"momentsSq", &g.momentsSq},
                {"reflect", &g.reflect},     {"reflAux", &g.reflAux},
                {"shadowVis", &g.shadowVis}, {"directU", &g.directU},
                {"rtao", &g.rtao},           {"rtaoAux", &g.rtaoAux},
                {"atrousA", &g.atrousA},     {"atrousB", &g.atrousB},
                {"reservoirW0", &pv.reservoirWImagesPP[0]},
                {"reservoirW1", &pv.reservoirWImagesPP[1]},
        };
        constexpr VkDeviceSize kBpp = 8;

        VkDeviceSize total = 0;
        for (const auto& [name, img] : imgs) {
            if (img->image == VK_NULL_HANDLE || img->width == 0) return {};
            total += static_cast<VkDeviceSize>(img->width) * img->height * kBpp;
        }
        // Probe SH store rides along (a buffer, not an image): the row that
        // says whether the probe atlas is the divergence carrier or a victim.
        const VkDeviceSize probeShBytes =
                impl.probeGI_ ? static_cast<VkDeviceSize>(vulkan::ProbeGI::kProbeCount) * 4 * 16
                              : 0;
        total += probeShBytes;
        // probeDepth = pure ray GEOMETRY (Chebyshev hit distances). probeSh
        // diverging while probeDepth stays exact means the probe rays hit the
        // same surfaces at the same distances and only the RADIANCE evaluation
        // differs; both diverging means the rays themselves differ.
        const VkDeviceSize probeDepthBytes =
                impl.probeGI_ ? static_cast<VkDeviceSize>(vulkan::ProbeGI::kProbeCount) *
                                        vulkan::ProbeGI::kDepthTexels * 4
                              : 0;
        total += probeDepthBytes;

        vkDeviceWaitIdle(ctx->device());

        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), total,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(debugHashShadeImages)");
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(debugHashShadeImages)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(debugHashShadeImages)");

        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        VkDeviceSize offset = 0;
        for (const auto& [name, img] : imgs) {
            VkBufferImageCopy region{};
            region.bufferOffset                = offset;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent                 = {img->width, img->height, 1};
            vkCmdCopyImageToBuffer(cb, img->image, VK_IMAGE_LAYOUT_GENERAL,
                                   staging.handle, 1, &region);
            offset += static_cast<VkDeviceSize>(img->width) * img->height * kBpp;
        }
        if (probeShBytes > 0) {
            VkBufferCopy bc{};
            bc.dstOffset = offset;
            bc.size      = probeShBytes;
            vkCmdCopyBuffer(cb, impl.probeGI_->shBuffer(), staging.handle, 1, &bc);
            bc.dstOffset = offset + probeShBytes;
            bc.size      = probeDepthBytes;
            vkCmdCopyBuffer(cb, impl.probeGI_->depthBuffer(), staging.handle, 1, &bc);
        }

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(debugHashShadeImages)");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(debugHashShadeImages)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(debugHashShadeImages)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(debugHashShadeImages)");

        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(debugHashShadeImages)");
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, total);

        std::vector<std::pair<std::string, uint64_t>> out;
        const auto* base = static_cast<const unsigned char*>(mapped);
        auto fnv = [&](VkDeviceSize off, VkDeviceSize sz) {
            uint64_t hsh = 0xCBF29CE484222325ULL;
            for (VkDeviceSize i = 0; i < sz; ++i) {
                hsh ^= base[off + i];
                hsh *= 0x100000001B3ULL;
            }
            return hsh;
        };
        offset = 0;
        // THREEPP_SHADE_DUMP_DIR=<dir>: also write each image's bytes for the
        // first frames as <name>_f<serial>.raw, so two runs whose hashes differ
        // can be compared pixel by pixel (WHERE a pass differs says what class
        // of input it consumed). Debug only; the hash rows are the contract.
        const char* dumpDir = std::getenv("THREEPP_SHADE_DUMP_DIR");
        for (const auto& [name, img] : imgs) {
            const VkDeviceSize sz = static_cast<VkDeviceSize>(img->width) * img->height * kBpp;
            out.emplace_back(name, fnv(offset, sz));
            if (dumpDir && impl.frameSerial_ <= 8) {
                const std::string path = std::string(dumpDir) + "/" + name + "_f" +
                                         std::to_string(impl.frameSerial_) + "_" +
                                         std::to_string(img->width) + "x" + std::to_string(img->height) + ".raw";
                if (FILE* fp = std::fopen(path.c_str(), "wb")) {
                    std::fwrite(base + offset, 1, static_cast<size_t>(sz), fp);
                    std::fclose(fp);
                }
            }
            offset += sz;
        }
        if (probeShBytes > 0) {
            out.emplace_back("probeSh", fnv(offset, probeShBytes));
            out.emplace_back("probeDepth", fnv(offset + probeShBytes, probeDepthBytes));
        }
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);
        return out;
    }

    bool VulkanRenderer::readProbeShDebug(std::vector<uint8_t>& sh) {
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx || !impl.probeGI_) return false;
        if (impl.frameSerial_ == 0) return false;

        const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(vulkan::ProbeGI::kProbeCount) * 4 * 16;
        vkDeviceWaitIdle(ctx->device());

        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), bytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readProbeShDebug)");
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readProbeShDebug)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(readProbeShDebug)");

        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VkBufferCopy bc{};
        bc.size = bytes;
        vkCmdCopyBuffer(cb, impl.probeGI_->shBuffer(), staging.handle, 1, &bc);

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readProbeShDebug)");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readProbeShDebug)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readProbeShDebug)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readProbeShDebug)");

        sh.resize(static_cast<size_t>(bytes));
        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readProbeShDebug)");
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, bytes);
        std::memcpy(sh.data(), mapped, static_cast<size_t>(bytes));
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);
        return true;
    }

    void VulkanRenderer::setSimTime(double seconds) {
        core()->simTimeSec_ = seconds;
    }

    void VulkanRenderer::resetTemporalHistory() {
        auto& impl = *core();
        // Every live view, the primary included: the camera-swap path only
        // ever reached the secondaries. The flags are the same set a camera
        // change clears; the reservoir clear is deferred to the frame path,
        // which drains the device before touching them.
        impl.forEachLiveView([&] {
            auto& v = impl.view();
            if (v.taa_) v.taa_->invalidateHistory();
            v.prevCameraValid        = false;
            v.rasterPrevVPValid_     = false;
            v.rasterPrevJitterValid_ = false;
            v.deferredCamPrevValid_  = false;
        });
        if (impl.probeGI_) impl.probeGI_->invalidateHistory();
        if (impl.occl_) impl.occl_->resetVisibility();
        // The sample index seeds every stochastic pass (ReSTIR, the gathers,
        // the soft sun, AO). A structural rebuild that cannot match entries by
        // identity zeroes it, and a scene streaming in does that at
        // run-dependent frames, so two runs reach the same frame with
        // different seeds and every ray-fed image differs from then on, with
        // no history involved. resetAccumulation() restarts the index (and the
        // motion state, and FSR's history) here, so a capture that begins after
        // this call draws the same sequence in every run.
        impl.resetAccumulation();
        // The sub-pixel jitter phase is temporal state as well: the raster
        // G-buffer of frame k depends on it, so two captures that began after
        // different numbers of frames would disagree on every AOV. Restart it.
        impl.haltonFrame_ = 0;
        impl.pendingAccumulationReset_ = true;
    }

    double VulkanRenderer::simTime() const {
        return core()->simTimeSec_;
    }

    void VulkanRenderer::setDenoise(bool enabled) {
        core()->denoiseEnabled_ = enabled;
    }

    bool VulkanRenderer::denoise() const {
        return core()->denoiseEnabled_;
    }

    void VulkanRenderer::setBloomIntensity(float intensity) {
        core()->bloomIntensity_ = intensity < 0.f ? 0.f : intensity;
    }

    float VulkanRenderer::bloomIntensity() const {
        return core()->bloomIntensity_;
    }

    void VulkanRenderer::setDeferredDenoise(bool enabled) {
        core()->denoiseEnabled_ = enabled;
    }

    bool VulkanRenderer::deferredDenoise() const {
        return core()->denoiseEnabled_;
    }

    void VulkanRenderer::setDeferredAO(bool enabled) {
        pimpl_->deferredAO_ = enabled;
    }

    bool VulkanRenderer::deferredAO() const {
        return pimpl_->deferredAO_;
    }

    void VulkanRenderer::setProbeGI(bool enabled) {
        if (pimpl_->probeGIEnabled_ == enabled) return;
        pimpl_->probeGIEnabled_ = enabled;
        // Force a re-fit + SH clear on (re-)enable so a scene swap while the
        // feature was off can't leave the grid over stale bounds.
        if (enabled) pimpl_->probeGridDirty_ = true;
        // The probe term feeds the accumulated GI channel — toggling shifts
        // its converged mean, so reset accumulation to make the change land
        // immediately (same pattern as setRestirDIEnabled).
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    bool VulkanRenderer::probeGI() const {
        return pimpl_->probeGIEnabled_;
    }

    void VulkanRenderer::setGbufferMsaa(uint32_t samples) {
        pimpl_->setGbufferMsaa(samples);
    }

    uint32_t VulkanRenderer::gbufferMsaa() const {
        return pimpl_->gbufferMsaa();
    }

    void VulkanRenderer::setSplatDepthAov(SplatDepthMode mode) {
        pimpl_->setSplatDepthAov(mode);
    }

    void VulkanRenderer::setSplatDepthAov(bool enabled) {
        pimpl_->setSplatDepthAov(enabled ? SplatDepthMode::Expected : SplatDepthMode::Off);
    }

    bool VulkanRenderer::splatDepthAov() const {
        return core()->splatDepthAov();
    }

    VulkanRenderer::SplatDepthMode VulkanRenderer::splatDepthAovMode() const {
        return core()->splatDepthAovMode();
    }

    void VulkanRenderer::setSplatDebugChecksum(bool enabled) {
        core()->splatChecksum_ = enabled;
    }

    bool VulkanRenderer::splatDebugChecksum(std::uint64_t out[4]) const {
        out[0] = out[1] = out[2] = out[3] = 0;
        if (!core()->splat_) return false;
        core()->splat_->readDebug(out);
        return out[3] != 0;
    }

    std::size_t VulkanRenderer::splatResidentClouds() const {
        return core()->splat_ ? core()->splat_->residentCount() : 0;
    }

    std::size_t VulkanRenderer::splatScratchSplats() const {
        return core()->splat_ ? core()->splat_->scratchSplats() : 0;
    }

    std::uint64_t VulkanRenderer::splatVolumeBytes() const {
        return core()->splat_ ? core()->splat_->volumeBytes() : 0;
    }

    std::uint64_t VulkanRenderer::splatVolumeGeneration() const {
        return core()->splat_ ? core()->splat_->volumeGeneration() : 0;
    }

    void VulkanRenderer::splatVolumeHash(std::uint64_t out[3]) const {
        out[0] = out[1] = out[2] = 0;
        if (core()->splat_) core()->splat_->readVolumeHash(out);
    }

    void VulkanRenderer::setGpuInstanceExpansion(bool enabled) {
        core()->gpuInstanceExpand_ = enabled;
    }

    bool VulkanRenderer::gpuInstanceExpansion() const {
        return core()->gpuInstanceExpand_;
    }

    bool VulkanRenderer::instanceExpandCheck(InstanceExpandCheck& out) {
        out = {};
        return core()->verifyInstanceExpansion(out);
    }

    void VulkanRenderer::setBloomThreshold(float threshold) {
        core()->bloomThreshold_ = threshold < 0.f ? 0.f : threshold;
    }

    float VulkanRenderer::bloomThreshold() const {
        return core()->bloomThreshold_;
    }

    void VulkanRenderer::setBloomClamp(float clampMax) {
        core()->bloomClamp_ = clampMax < 0.f ? 0.f : clampMax;
    }

    float VulkanRenderer::bloomClamp() const {
        return core()->bloomClamp_;
    }

    void VulkanRenderer::setSharpenStrength(float amount) {
        core()->sharpenStrength_ = amount < 0.f ? 0.f : amount;
    }

    float VulkanRenderer::sharpenStrength() const {
        return core()->sharpenStrength_;
    }

    void VulkanRenderer::setMotionBlur(float shutterFraction) {
        auto* c = core();
        const float clamped = std::clamp(shutterFraction, 0.f, 1.f);
        if (c->motionBlurAmount_ == clamped) return;
        c->motionBlurAmount_ = clamped;
    }

    float VulkanRenderer::motionBlur() const {
        return core()->motionBlurAmount_;
    }

    void VulkanRenderer::setPhysicalCamera(bool enabled) {
        core()->physicalCamera_ = enabled;
    }

    bool VulkanRenderer::physicalCamera() const {
        return core()->physicalCamera_;
    }

    void VulkanRenderer::setCameraExposure(float aperture, float shutterSeconds, float iso) {
        core()->camAperture_ = std::max(aperture, 0.1f);
        core()->camShutter_  = std::max(shutterSeconds, 1e-6f);
        core()->camIso_      = std::max(iso, 1.f);
    }

    VulkanRenderer::CameraExposure VulkanRenderer::cameraExposure() const {
        return {core()->camAperture_, core()->camShutter_, core()->camIso_};
    }

    void VulkanRenderer::setExposureCompensation(float ev) {
        core()->camEvComp_ = std::clamp(ev, -20.f, 20.f);
    }

    float VulkanRenderer::exposureCompensation() const {
        return core()->camEvComp_;
    }

    VulkanRenderer::CameraIntrinsics VulkanRenderer::cameraIntrinsics() const {
        auto& impl = *core();
        if (!impl.ctx) return {};
        const VkExtent2D ext = impl.renderExtent();
        if (ext.width == 0u || ext.height == 0u) return {};

        // Pixel mapping of the GL-convention projection threepp builds:
        //   x_ndc = p0·(X/−Z) − p8      u = (x_ndc·0.5 + 0.5)·W
        //   y_ndc = p5·(Y/−Z) − p9      v = (0.5 − y_ndc·0.5)·H   (top-left origin)
        // Rewriting in OpenCV's frame (+Z forward, +Y down) gives
        //   u = fx·(X/Z) + cx,  v = fy·(Y/Z) + cy
        // with the terms below. A symmetric frustum (p8 = p9 = 0) lands the
        // principal point exactly at the centre; filmOffset / setViewOffset
        // shift it, which is why the skew terms are carried through.
        const float w = static_cast<float>(ext.width);
        const float h = static_cast<float>(ext.height);

        CameraIntrinsics k;
        k.fx     = 0.5f * w * impl.projP0_;
        k.fy     = 0.5f * h * impl.projP5_;
        k.cx     = 0.5f * w * (1.f - impl.projP8_);
        k.cy     = 0.5f * h * (1.f + impl.projP9_);
        k.width  = ext.width;
        k.height = ext.height;
        return k;
    }

    void VulkanRenderer::setLensDistortion(const LensDistortion& distortion) {
        core()->lens_ = distortion;
    }

    LensDistortion VulkanRenderer::lensDistortion() const {
        return core()->lens_;
    }

    void VulkanRenderer::setLensOverscan(float factor) {
        // Below 1 would CROP the rendered field, which no lens does and which
        // would silently discard geometry the user asked to see.
        core()->lensOverscan_ = std::clamp(factor, 1.f, 4.f);
    }

    float VulkanRenderer::lensOverscan() const {
        return core()->lensOverscan_;
    }

    void VulkanRenderer::setSensorNoise(const SensorNoise& noise) {
        auto* c = core();
        // Re-seeding restarts the sequence; changing the sigmas alone must NOT
        // (an interactive slider would otherwise replay frame 0 forever). Same
        // convention as VisionSensor's RangeNoiseModel.
        if (noise.seed != c->sensorNoise_.seed) c->sensorNoiseFrame_ = 0u;
        c->sensorNoise_ = noise;
    }

    VulkanRenderer::SensorNoise VulkanRenderer::sensorNoise() const {
        return core()->sensorNoise_;
    }

    void VulkanRenderer::resetSensorNoise() {
        core()->sensorNoiseFrame_ = 0u;
    }

    void VulkanRenderer::setPhysicalLightUnits(bool enabled) {
        core()->physicalLightUnits_ = enabled;
    }

    bool VulkanRenderer::physicalLightUnits() const {
        return core()->physicalLightUnits_;
    }

    void VulkanRenderer::setDepthOfField(bool enabled) {
        core()->dofEnabled_ = enabled;
    }

    bool VulkanRenderer::depthOfField() const {
        return core()->dofEnabled_;
    }

    void VulkanRenderer::setFocusDistance(float meters) {
        core()->focusDistance_ = std::max(meters, 0.01f);
    }

    float VulkanRenderer::focusDistance() const {
        return core()->focusDistance_;
    }

    void VulkanRenderer::setOcclusionCulling(bool enabled) {
        auto* c = core();
        if (c->occlusionCullingEnabled_ == enabled) return;
        c->occlusionCullingEnabled_ = enabled;
        c->occlActiveThisFrame_ = false;// next buildIndirectDrawData re-evaluates
        // The farthest pyramid's image is allocated lazily by the next
        // frame's ensureHybridResources (which inherits the resize/MSAA
        // idle-wait guarantees), so enabling mid-run engages one frame later.
    }

    bool VulkanRenderer::occlusionCulling() const {
        return core()->occlusionCullingEnabled_;
    }

    void VulkanRenderer::setWhiteBalance(float temperatureK, float tint) {
        core()->wbTemperatureK_ = temperatureK;
        core()->wbTint_ = tint;
        if (core()->primaryView().post_) core()->primaryView().post_->setWhiteBalance(temperatureK, tint);
    }

    std::pair<float, float> VulkanRenderer::whiteBalance() const {
        return {core()->wbTemperatureK_, core()->wbTint_};
    }

    void VulkanRenderer::setColorGrade(const ColorGrade& grade) {
        if (!core()->primaryView().post_) return;
        vulkan::PostComposite::ColorGrade g;
        g.lift[0]  = grade.lift.x;  g.lift[1]  = grade.lift.y;  g.lift[2]  = grade.lift.z;
        g.gamma[0] = grade.gamma.x; g.gamma[1] = grade.gamma.y; g.gamma[2] = grade.gamma.z;
        g.gain[0]  = grade.gain.x;  g.gain[1]  = grade.gain.y;  g.gain[2]  = grade.gain.z;
        g.saturation = grade.saturation;
        g.contrast   = grade.contrast;
        core()->primaryView().post_->setColorGrade(g);
    }

    void VulkanRenderer::setFireflyClamp(float cap) {
        core()->fireflyClamp_ = (cap <= 0.0f) ? 1e30f : cap;
    }

    float VulkanRenderer::fireflyClamp() const {
        const float v = core()->fireflyClamp_;
        return (v > 1e20f) ? 0.0f : v;
    }

    void VulkanRenderer::setSunAngularRadius(float degrees) {
        core()->sunAngularRadiusDeg_ = std::max(0.f, degrees);
    }

    float VulkanRenderer::sunAngularRadius() const {
        return core()->sunAngularRadiusDeg_;
    }

    VulkanRenderer::SoftBodyInteropHandle
    VulkanRenderer::enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy) {
        return core()->enableSoftBodyInterop(mesh, std::move(deviceCopy));
    }

    VulkanRenderer::ParticleFieldInteropHandle
    VulkanRenderer::enableParticleFieldInterop(ParticleField& field,
                                               std::function<void()> deviceCopy) {
        return core()->enableParticleFieldInterop(field, std::move(deviceCopy));
    }

    VulkanRenderer::VertexInteropHandle
    VulkanRenderer::enableVertexInterop(const Mesh& mesh, std::function<void()> deviceCopy,
                                        bool validate, bool stableCorrespondence) {
        return core()->enableVertexInterop(mesh, std::move(deviceCopy), validate,
                                           stableCorrespondence);
    }

    void VulkanRenderer::disableVertexInterop(const Mesh& mesh) {
        core()->disableVertexInterop(mesh);
    }

    std::vector<VulkanRenderer::FrameInteropExport>
    VulkanRenderer::enableFrameInterop(uint32_t viewHandle,
                                       const std::vector<FrameChannel>& channels) {
        return core()->enableFrameInterop(viewHandle, channels);
    }

    void VulkanRenderer::disableFrameInterop(uint32_t viewHandle) {
        core()->disableFrameInterop(viewHandle);
    }

    bool VulkanRenderer::frameInteropActive(uint32_t viewHandle) const {
        return core()->frameInteropActive(viewHandle);
    }

    bool VulkanRenderer::syncFrameInterop() {
        return core()->syncFrameInterop();
    }

    void VulkanRenderer::setDeferredVolumetrics(float density, float anisotropy) {
        pimpl_->deferredVolDensity_ = std::max(density, 0.f);
        pimpl_->deferredVolAniso_   = std::clamp(anisotropy, -0.95f, 0.95f);
    }

    std::pair<float, float> VulkanRenderer::deferredVolumetrics() const {
        return {pimpl_->deferredVolDensity_, pimpl_->deferredVolAniso_};
    }

    // DEPRECATED (Phase 2 fog unification) — a no-op alias, kept so existing
    // callers compile (house pattern, cf. setDeferredDenoise). The directional
    // sun shafts + aerial glow are now ALWAYS on when the fog medium is present:
    // the froxels own the near field and volumetricDirScatter marches the far
    // tail. There is no separate "volumetric fog" opt-in any more — set scene.fog
    // (or setHeightFog) and the volumetrics follow automatically.
    void VulkanRenderer::setVolumetricFog(bool /*enabled*/) {}

    bool VulkanRenderer::volumetricFog() const {
        return pimpl_->deferredVolFog_;// always true — shafts follow the medium
    }

    void VulkanRenderer::setClouds(const std::optional<CloudSettings>& settings) {
        if (settings) {
            const float coverage = std::clamp(settings->coverage, 0.f, 1.f);
            const float density  = std::max(settings->density, 0.f);
            const float bottomY  = settings->bottomY;
            const float topY     = std::max(settings->topY, settings->bottomY + 1.f);
            const float evolve   = std::max(settings->evolveSpeed, 0.f);
            // History epoch: bump ONLY on enable or a material change — the
            // cloud march drops its temporal history at an epoch boundary
            // (first-enable undefined contents, reconfigured decks must not
            // ghost the old pattern for ~24 frames). Demos re-set identical
            // values every frame; an unconditional bump would permanently
            // kill the temporal accumulation.
            // Wind is NOT an epoch trigger: it only changes the drift RATE —
            // the pattern moves continuously and the motion-shortened EMA
            // tracks it (and demos tie wind to a live slider).
            const bool changed = !pimpl_->cloudsEnabled_
                              || coverage != pimpl_->cloudCoverage_
                              || density  != pimpl_->cloudDensity_
                              || bottomY  != pimpl_->cloudBottomY_
                              || topY     != pimpl_->cloudTopY_
                              || evolve   != pimpl_->cloudEvolveSpeed_;
            if (changed) pimpl_->cloudEpoch_ = (pimpl_->cloudEpoch_ + 1) & 1023;
            pimpl_->cloudsEnabled_    = true;
            pimpl_->cloudCoverage_    = coverage;
            pimpl_->cloudDensity_     = density;
            pimpl_->cloudBottomY_     = bottomY;
            pimpl_->cloudTopY_        = topY;
            pimpl_->cloudEvolveSpeed_ = evolve;
            pimpl_->cloudWind_[0]     = settings->wind.x;
            pimpl_->cloudWind_[1]     = settings->wind.y;
            pimpl_->cloudWind_[2]     = settings->wind.z;
        } else {
            pimpl_->cloudsEnabled_ = false;
        }
    }

    std::optional<VulkanRenderer::CloudSettings> VulkanRenderer::clouds() const {
        if (!pimpl_->cloudsEnabled_) return std::nullopt;
        CloudSettings s;
        s.coverage    = pimpl_->cloudCoverage_;
        s.density     = pimpl_->cloudDensity_;
        s.bottomY     = pimpl_->cloudBottomY_;
        s.topY        = pimpl_->cloudTopY_;
        s.evolveSpeed = pimpl_->cloudEvolveSpeed_;
        s.wind        = Vector3(pimpl_->cloudWind_[0], pimpl_->cloudWind_[1], pimpl_->cloudWind_[2]);
        return s;
    }

    void VulkanRenderer::setHeightFog(const std::optional<HeightFogSettings>& settings) {
        if (settings) {
            pimpl_->heightFogEnabled_     = true;
            pimpl_->heightFogDensity_     = std::max(settings->density, 0.f);
            pimpl_->heightFogBaseY_       = settings->baseY;
            pimpl_->heightFogFalloff_     = std::max(settings->falloff, 1.f);
            pimpl_->heightFogNoiseAmount_ = std::clamp(settings->noiseAmount, 0.f, 1.f);
        } else {
            pimpl_->heightFogEnabled_ = false;
        }
    }

    std::optional<VulkanRenderer::HeightFogSettings> VulkanRenderer::heightFog() const {
        if (!pimpl_->heightFogEnabled_) return std::nullopt;
        HeightFogSettings s;
        s.density     = pimpl_->heightFogDensity_;
        s.baseY       = pimpl_->heightFogBaseY_;
        s.falloff     = pimpl_->heightFogFalloff_;
        s.noiseAmount = pimpl_->heightFogNoiseAmount_;
        return s;
    }

    void VulkanRenderer::setDeferredStarfield(float intensity) {
        pimpl_->deferredStarIntensity_ = std::max(intensity, 0.f);
    }

    float VulkanRenderer::deferredStarfield() const {
        return pimpl_->deferredStarIntensity_;
    }

    void VulkanRenderer::setEnvSunPolicy(EnvSunPolicy policy) {
        if (pimpl_->envSunPolicy_ == policy) return;
        const bool wasOff = pimpl_->envSunPolicy_ == EnvSunPolicy::Off;
        const bool isOff  = policy == EnvSunPolicy::Off;
        pimpl_->envSunPolicy_ = policy;
        // Auto↔Always only changes the INJECTION, which updateLightsUbo
        // re-evaluates every frame (its UBO hash change resets accumulation).
        // Off transitions change the PMREM content, so force the env re-upload
        // path (refreshEnvTextureFromScene early-outs on a matching texture id)
        // so the mips are rebuilt with/without the sun.
        if (wasOff != isOff) {
            pimpl_->envTextureIdUploaded = 0xFFFFFFFFu;
            pimpl_->envSun_ = {};
        }
    }

    VulkanRenderer::EnvSunPolicy VulkanRenderer::envSunPolicy() const {
        return pimpl_->envSunPolicy_;
    }

    void VulkanRenderer::setEnvSunExtraction(bool enabled) {
        setEnvSunPolicy(enabled ? EnvSunPolicy::Auto : EnvSunPolicy::Off);
    }

    bool VulkanRenderer::envSunExtraction() const {
        return pimpl_->envSunPolicy_ != EnvSunPolicy::Off;
    }

    bool VulkanRenderer::envSunFound() const {
        return pimpl_->envSun_.found;
    }

    Vector3 VulkanRenderer::envSunDirection() const {
        const auto& s = pimpl_->envSun_;
        return {s.dir[0], s.dir[1], s.dir[2]};
    }

    Vector3 VulkanRenderer::envSunColor() const {
        const auto& s = pimpl_->envSun_;
        return {s.colorE[0], s.colorE[1], s.colorE[2]};
    }

    void VulkanRenderer::setAutoExposure(bool enabled) {
        pimpl_->autoExposureEnabled_ = enabled;
    }

    bool VulkanRenderer::autoExposure() const {
        return pimpl_->autoExposureEnabled_;
    }

    void VulkanRenderer::setAutoExposureSpeed(float evPerSecond) {
        pimpl_->autoExpSpeed_ = std::max(evPerSecond, 0.01f);
        if (pimpl_->autoExposure_) pimpl_->autoExposure_->adaptSpeed = pimpl_->autoExpSpeed_;
    }

    float VulkanRenderer::autoExposureSpeed() const {
        return pimpl_->autoExpSpeed_;
    }

    void VulkanRenderer::setAutoExposureRange(float minEV, float maxEV) {
        pimpl_->autoExpMinEV_ = minEV;
        pimpl_->autoExpMaxEV_ = maxEV;
        if (pimpl_->autoExposure_) {
            pimpl_->autoExposure_->minEV = minEV;
            pimpl_->autoExposure_->maxEV = maxEV;
        }
    }

    void VulkanRenderer::disableSoftBodyInterop(const Mesh& mesh) {
        core()->disableSoftBodyInterop(mesh);
    }

    void VulkanRenderer::setRestirDIEnabled(bool enabled) {
        if (core()->restirDIEnabled_ == enabled) return;
        core()->restirDIEnabled_ = enabled;
        // ReSTIR is unbiased, so toggling it changes the convergence rate, not
        // the converged mean — on a settled frame the running-mean accumulator
        // would hide the switch entirely. Reset accumulation so the change is
        // actually visible. Gated on sceneBuilt_: before the first render there
        // is nothing accumulated and the gbuf/reservoir images aren't allocated
        // yet (clearGbufImages would touch null handles).
        if (core()->sceneBuilt_) core()->resetAccumulation();
    }

    bool VulkanRenderer::restirDIEnabled() const {
        return core()->restirDIEnabled_;
    }

    void VulkanRenderer::setNormalMapToksvig(bool enabled) {
        core()->normalMapToksvig_ = enabled;
    }

    bool VulkanRenderer::normalMapToksvig() const {
        return core()->normalMapToksvig_;
    }

    void VulkanRenderer::setAutoLod(bool enabled) {
        core()->autoLod_ = enabled;
    }

    bool VulkanRenderer::autoLod() const {
        return core()->autoLod_;
    }

    void VulkanRenderer::setAutoLodError(float px) {
        core()->lodErrorPx_ = std::clamp(px, 0.1f, 8.f);
    }

    float VulkanRenderer::autoLodError() const {
        return core()->lodErrorPx_;
    }

    VulkanRenderer::AutoLodStats VulkanRenderer::autoLodStats() const {
        return core()->autoLodStats_;
    }

    VulkanRenderer::DynamicGeomStats VulkanRenderer::dynamicGeomStats() const {
        return core()->dynGeomStats_;
    }

    VulkanRenderer::TlasStats VulkanRenderer::tlasStats() const {
        return core()->tlasStats_;
    }

    VulkanRenderer::FrameTimings VulkanRenderer::lastFrameTimings() const {
        return core()->gpuTimings_->timings();
    }

    void VulkanRenderer::scanLidar(const std::vector<LidarBeam>& beams,
                                   std::vector<LidarReturn>& results,
                                   const LidarParams& params,
                                   std::vector<LidarReturn>* cleanResults) {
        core()->scanLidar(beams, results, params, cleanResults);
    }

    int VulkanRenderer::scanLidarBegin(const std::vector<LidarBeam>& beams,
                                       const LidarParams& params) {
        return core()->scanLidarBegin(beams, params);
    }

    bool VulkanRenderer::scanLidarReady(int handle) const {
        return core()->scanLidarReady(handle);
    }

    bool VulkanRenderer::scanLidarCollect(int handle, std::vector<LidarReturn>& results,
                                          std::vector<LidarReturn>* cleanResults) {
        return core()->scanLidarCollect(handle, results, cleanResults);
    }

    void VulkanRenderer::setOverlayLayer(int channel) {
        core()->overlayLayer_ = (channel < 0 || channel > 31) ? -1 : channel;
    }

    int VulkanRenderer::overlayLayer() const {
        return core()->overlayLayer_;
    }

    void VulkanRenderer::setSensorOnlySurfaces(bool enabled) {
        if (core()->sensorOnlySurfaces_ == enabled) return;
        core()->sensorOnlySurfaces_ = enabled;
        // The opt-in is carried by TLAS instance masks, which only a full
        // expansion rewrites — the snapshot fast path would keep handing the
        // old ones back.
        core()->sceneBuilt_ = false;
    }

    bool VulkanRenderer::sensorOnlySurfaces() const {
        return core()->sensorOnlySurfaces_;
    }

    bool VulkanRenderer::setViewSensorSurfaces(uint32_t handle, bool enabled) {
        return core()->setViewSensorSurfacesImpl(handle, enabled);
    }

    bool VulkanRenderer::viewSensorSurfaces(uint32_t handle) const {
        auto* v = const_cast<Impl*>(core())->findView(handle);
        return v && v->secondary && v->sensorSurfaces;
    }

    bool VulkanRenderer::setViewSplats(uint32_t handle, bool enabled) {
        return core()->setViewSplatsImpl(handle, enabled);
    }

    bool VulkanRenderer::viewSplats(uint32_t handle) const {
        auto* v = const_cast<Impl*>(core())->findView(handle);
        return v && v->secondary && v->splats;
    }

    void VulkanRenderer::setHybridDebugView(int view) {
        using V = Impl::HybridDebugView;
        switch (view) {
            case 1:  core()->hybridDebugView_ = V::Normal; break;
            case 2:  core()->hybridDebugView_ = V::Motion; break;
            case 3:  core()->hybridDebugView_ = V::Ids;    break;
            case 4:  core()->hybridDebugView_ = V::Albedo; break;
            case 5:  core()->hybridDebugView_ = V::Depth;  break;
            default: core()->hybridDebugView_ = V::Off;    break;
        }
    }

    int VulkanRenderer::hybridDebugView() const {
        using V = Impl::HybridDebugView;
        switch (core()->hybridDebugView_) {
            case V::Normal: return 1;
            case V::Motion: return 2;
            case V::Ids:    return 3;
            case V::Albedo: return 4;
            case V::Depth:  return 5;
            default:        return 0;
        }
    }

} // namespace threepp
