#include "VulkanCoreImpl.hpp"

#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kInstFlag* bit layout

// debug_resolve.comp's embedded SPIR-V array (kDebugResolveCompSpv) is only
// referenced by createDebugResolvePipeline below — moved out of the header
// with the rest of this file's methods.
#include "threepp/renderers/vulkan/shaders/debug_resolve.comp.spv.h"

namespace threepp {

    bool VulkanRendererCore::CoreImpl::ensureDrawInfoCapacity(uint32_t frame, VkDeviceSize neededBytes) {
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

    bool VulkanRendererCore::CoreImpl::ensureIndirectCmdCapacity(uint32_t frame, VkDeviceSize neededBytes) {
        if (neededBytes <= indirectCmdBufferCapacity[frame]) return false;
        const VkDeviceSize newCap = std::max<VkDeviceSize>(
                neededBytes, indirectCmdBufferCapacity[frame] * 2u);
        destroyBuffer(ctx->allocator(), indirectCmdBuffers[frame]);
        indirectCmdBuffers[frame] = createBuffer(
                ctx->allocator(), ctx->device(),
                newCap,
                // STORAGE: the occlusion-cull filter compute reads these
                // CPU-built records as an SSBO (occl_cull.comp binding 0).
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
    void VulkanRendererCore::CoreImpl::buildIndirectDrawData(uint32_t frame) {
        for (auto& g : indirectGroups_) { g.offset = 0; g.count = 0; }
        indirectTotalDraws_ = 0;

        if (lastVisibleEntries_.empty()) return;

        // Four buckets: [BACK_cull, FRONT_cull, NONE_cull, decal] order.
        std::array<std::vector<DrawInfoGpu>, 4>            draws;
        std::array<std::vector<VkDrawIndirectCommand>, 4>  cmds;
        // Occlusion-cull metadata rides the same bucketing so its final
        // concatenated order matches the command records 1:1.
        const bool wantOcclMeta = occlusionCullingEnabled_ && occl_ &&
                                  occlHiz_ && occlHiz_->valid() && !scissorTest;
        std::array<std::vector<vulkan::OcclusionCull::CullMeta>, 4> occlMeta;
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
            // Auto-LOD: en.lodLevel==0 for every non-eligible entry (the
            // selection pass in ensureSceneBuilt only sets it >0 on plain
            // indexed static geometry), so this passthrough is a no-op
            // everywhere else. Must resolve identically to the TLAS instance
            // fill in ensureSceneBuilt — both read en.lodLevel verbatim.
            const auto lodSel = selectLodGeom(*rec, en.lodLevel);
            const uint32_t vcount = indexed ? lodSel.indexCount : rec->vertexCount;
            if (vcount == 0u) continue;

            const VkCullModeFlags wantCull =
                    (i < lastVisibleCullMode_.size())
                            ? lastVisibleCullMode_[i]
                            : VK_CULL_MODE_BACK_BIT;
            // Blend decals (alphaCutoff == -2 sentinel from materialFromMesh)
            // go to bucket [3]: drawn last, with the albedo-blend pipeline.
            // The deferred shade reads the blended albedo attachment.
            const bool isDecal = decalsEnabled() &&
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
            di.indexAddr   = indexed ? lodSel.indexAddress : 0ull;
            // Per-vertex color: only when the material opts in (vertexColors)
            // and the geometry uploaded a color buffer — matches the
            // GeometryDesc::colorAddress gate. gbuffer.frag multiplies albedo
            // by the interpolated value.
            {
                const auto dmat = en.mesh->material();
                di.colorAddr = (rec->color.handle != VK_NULL_HANDLE && dmat && dmat->vertexColors)
                                       ? rec->color.address
                                       : 0ull;
            }
            di.instanceCustomIndex = static_cast<uint32_t>(i);
            // Per-instance flag word — canonical bit layout in
            // vulkan_shared.h (kInstFlag*); shaders read it back from the
            // gbuffer IDs .z via the instance_flags.glsl accessors.
            uint32_t flags = 0u;
            if (en.isDisplaced) flags |= kInstFlagWater;
            if (en.isSkinned)   flags |= kInstFlagSkinned;
            {
                const auto sm = en.mesh->material();
                // DOUBLE_SIDED: gbuffer.frag flips N toward the viewer, so on
                // cutout foliage the jittered coverage flips a pixel's normal
                // SIGN frame to frame. The GI temporal reproject + SVGF normal
                // edge-stop must treat ±N as the SAME surface there (flag
                // consumed in deferred_shade.comp / deferred_denoise.comp) or
                // the GI history cold-starts every frame — measured as 8× the
                // frame-to-frame flicker on a procedural tree canopy.
                if (sm && sm->side == Side::Double) flags |= kInstFlagDoubleSided;
                // TEXTURE-ANIMATED (scrolling UVs, video/live textures): the
                // pattern moves with NO geometric motion vectors, so the TAA
                // resolve must hold a short history (α floor) instead of
                // smearing it. TAA-only — the GI history accumulates
                // DEMODULATED irradiance, which a texture animation doesn't
                // change, so bit 6 deliberately does NOT shorten the GI cap.
                if (sm && sm->textureAnimatedHint) flags |= kInstFlagTexAnim;
            }
            // PERSISTENT DEFORMER: a PhysX soft body deforms EVERY frame, so
            // the GI temporal cap in deferred_shade.comp must not chase its
            // oscillating per-pixel motion magnitude (visible pumping on the
            // wobble). The shader pins a constant history cap on this bit;
            // the TAA resolve floors its blend α (shading changes per frame).
            if (en.isTet) flags |= kInstFlagDeformer;
            // Semantic CLASS id (0..255) packed into bits 8..15 — inert to
            // every flag bit-test (they mask the low byte) and carried
            // through the MSAA resolve. Read back via outIds.z >> 8.
            flags |= (static_cast<uint32_t>(classIdForObject(*en.mesh)) & 0xFFu) << 8;
            di.flags   = flags;
            di.indexed = indexed ? 1u : 0u;
            // STABLE per-object instance id -> outIds.y (survives visible-set
            // reordering, unlike instanceCustomIndex/.x).
            di.stableId = stableIdForObject(*en.mesh);
            di._pad     = 0u;
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

            if (wantOcclMeta) {
                // Same rules as cullEntriesAgainstFrustum: deformers'
                // cached local AABB is stale per frame → always draw;
                // missing bounds → always draw.
                vulkan::OcclusionCull::CullMeta cm{};
                cm.stableId = di.stableId;
                bool always = en.isSkinned || en.isDisplaced || en.isGrass ||
                              en.isMorphed || en.isTet;
                Box3 worldAabb;
                if (!always) {
                    auto geom = en.mesh->geometry();
                    if (geom && !geom->boundingBox) geom->computeBoundingBox();
                    if (geom && geom->boundingBox) {
                        worldAabb = *geom->boundingBox;
                        Matrix4 w;
                        std::memcpy(w.elements.data(), en.worldMatrix.data(), 64);
                        worldAabb.applyMatrix4(w);
                    } else {
                        always = true;
                    }
                }
                cm.flags = always ? 1u : 0u;
                cm.aabbMin[0] = worldAabb.min().x;
                cm.aabbMin[1] = worldAabb.min().y;
                cm.aabbMin[2] = worldAabb.min().z;
                cm.aabbMax[0] = worldAabb.max().x;
                cm.aabbMax[1] = worldAabb.max().y;
                cm.aabbMax[2] = worldAabb.max().z;
                occlMeta[b].push_back(cm);
            }
            ++globalIdx;
        }

        indirectTotalDraws_ = globalIdx;
        occlActiveThisFrame_ = wantOcclMeta && globalIdx > 0u;
        if (globalIdx == 0u) return;

        // Concatenate buckets into the per-frame device buffers.
        const VkDeviceSize drawBytes = sizeof(DrawInfoGpu) * globalIdx;
        const VkDeviceSize cmdBytes  = sizeof(VkDrawIndirectCommand) * globalIdx;
        const bool drawGrown = ensureDrawInfoCapacity(frame, drawBytes);
        ensureIndirectCmdCapacity(frame, cmdBytes);

        // Occlusion culling: size the phase buffers + rewrite this fif's
        // sets if any input changed (AFTER the capacity calls above so
        // the src handle is final), then get the mapped meta destination.
        vulkan::OcclusionCull::CullMeta* occlMetaDst = nullptr;
        if (occlActiveThisFrame_) {
            vulkan::OcclusionCull::FrameInputs oin{};
            oin.srcCmds    = indirectCmdBuffers[frame].handle;
            oin.rasterCam  = rasterCameraUbos[frame].handle;
            oin.hizView    = occlHiz_->view();
            oin.hizSampler = occlHiz_->sampler();
            occl_->prepareFrame(frame, globalIdx, oin);
            occlMetaDst = occl_->metaPtr(frame);
        }

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
                if (occlMetaDst)
                    std::memcpy(occlMetaDst + offset, occlMeta[b].data(),
                                n * sizeof(vulkan::OcclusionCull::CullMeta));
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
    void VulkanRendererCore::CoreImpl::recordRasterGbufPass(VkCommandBuffer cb, uint32_t frame) {
        const auto& g = rasterGbufs[frame];
        // MSAA path: rasterize into the MS framebuffer/render pass/
        // pipelines (RasterGbufImages::*MS + rasterGbufRenderPassMS);
        // the single-sample framebuffer/pipelines below stay exactly
        // what they were pre-MSAA — gbuf_resolve.comp (dispatched right
        // after this, see recordCommandBuffer) fills the single-sample
        // images every downstream consumer keeps reading unchanged.
        const bool useMsaa = gbufMsaaSamples_ > 1 && g.framebufferMS != VK_NULL_HANDLE;
        recordRasterGbufPassInternal(cb, frame,
                                     useMsaa ? rasterGbufRenderPassMS : rasterGbufRenderPass,
                                     useMsaa ? g.framebufferMS : g.framebuffer,
                                     useMsaa,
                                     indirectCmdBuffers[frame].handle,
                                     /*clear=*/true);
    }

    // Shared body: `renderPass` must be COMPATIBLE with the pipelines'
    // creation pass (the occlusion-culling load/store variants are), and
    // `indirectBuffer` supplies the VkDrawIndirectCommand records the
    // bucket groups index into (the two-phase path swaps in the compute-
    // written phase buffers; offsets/counts are identical by design).
    void VulkanRendererCore::CoreImpl::recordRasterGbufPassInternal(VkCommandBuffer cb, uint32_t frame,
                                      VkRenderPass renderPass, VkFramebuffer fb,
                                      bool useMsaa, VkBuffer indirectBuffer,
                                      bool clear) {
        const auto& g = rasterGbufs[frame];
        if (fb == VK_NULL_HANDLE) return;// not initialized

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
        rpbi.renderPass      = renderPass;
        rpbi.framebuffer     = fb;
        rpbi.renderArea.offset = {0, 0};
        rpbi.renderArea.extent = {g.width, g.height};
        // LOAD-variant passes ignore the clear values but the count must
        // still cover every CLEAR attachment — harmless to always pass.
        rpbi.clearValueCount = clear ? 6u : 0u;
        rpbi.pClearValues    = clear ? clears : nullptr;
        vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        // Split-screen: draw the pane region-sized at the origin (the render
        // pass still clears the whole gbuf, so outside the pane reads as sky).
        // regionRenderExt_ == full render extent when scissorTest is off.
        const VkExtent2D gbufReg = regionRenderExt_;
        VkViewport viewport{};
        viewport.x = 0.f;
        viewport.y = 0.f;
        viewport.width  = float(gbufReg.width);
        viewport.height = float(gbufReg.height);
        viewport.minDepth = 0.f;
        viewport.maxDepth = 1.f;
        vkCmdSetViewport(cb, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = gbufReg;
        vkCmdSetScissor(cb, 0, 1, &scissor);

        if (indirectTotalDraws_ == 0u) {
            vkCmdEndRenderPass(cb);
            return;
        }

        const VkPipeline indirectPipe = useMsaa ? rasterGbufIndirectPipelineMS : rasterGbufIndirectPipeline;
        const VkPipeline decalPipe    = useMsaa ? rasterGbufDecalPipelineMS : rasterGbufDecalPipeline;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, indirectPipe);
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
                              indirectBuffer,
                              static_cast<VkDeviceSize>(g.offset) * cmdStride,
                              g.count,
                              static_cast<uint32_t>(cmdStride));
        }
        // Blend decals last: albedo-blend pipeline lerps over the receivers
        // rasterized above (same layout/descriptors, no set rebind needed).
        if (const auto& g = indirectGroups_[3]; g.count > 0u) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, decalPipe);
            vkCmdSetCullMode(cb, g.cullMode);
            vkCmdDrawIndirect(cb,
                              indirectBuffer,
                              static_cast<VkDeviceSize>(g.offset) * cmdStride,
                              g.count,
                              static_cast<uint32_t>(cmdStride));
        }

        vkCmdEndRenderPass(cb);
    }

    // Lazily create the debug_resolve compute pipeline + descriptor set.
    void VulkanRendererCore::CoreImpl::createDebugResolvePipeline() {
        if (debugResolvePipeline_ != VK_NULL_HANDLE) return;

        // 6 bindings: normal/motion/ids/albedo (combined image samplers;
        // ids is a usampler) + the swapchain storage image + depth (a
        // depth-aspect combined image sampler, for the depth AOV).
        std::array<VkDescriptorSetLayoutBinding, 6> b{};
        for (uint32_t i = 0; i < 4; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        b[4].binding         = 4;
        b[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[4].descriptorCount = 1;
        b[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[5].binding         = 5;
        b[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[5].descriptorCount = 1;
        b[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &debugResolveDsLayout_),
              "vkCreateDescriptorSetLayout(debug_resolve)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(DebugResolvePC);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &debugResolveDsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &debugResolvePipelineLayout_),
              "vkCreatePipelineLayout(debug_resolve)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kDebugResolveCompSpv);
        smci.pCode    = kDebugResolveCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(debug_resolve)");

        VkPipelineShaderStageCreateInfo ssci{};
        ssci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        ssci.module = mod;
        ssci.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = ssci;
        cpci.layout = debugResolvePipelineLayout_;
        check(vkCreateComputePipelines(ctx->device(), ctx->pipelineCache(), 1, &cpci, nullptr, &debugResolvePipeline_),
              "vkCreateComputePipelines(debug_resolve)");
        vkDestroyShaderModule(ctx->device(), mod, nullptr);

        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 5;// normal/motion/ids/albedo + depth
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &debugResolveDescPool_),
              "vkCreateDescriptorPool(debug_resolve)");
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = debugResolveDescPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &debugResolveDsLayout_;
        check(vkAllocateDescriptorSets(ctx->device(), &dsai, &debugResolveDescSet_),
              "vkAllocateDescriptorSets(debug_resolve)");
    }

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
    void VulkanRendererCore::CoreImpl::recordHybridDebugResolve(VkCommandBuffer cb, uint32_t imageIndex, uint32_t frame) {
        if (hybridDebugView_ == HybridDebugView::Off) return;
        createDebugResolvePipeline();

        const auto& g = rasterGbufs[frame];

        // Map the public view enum to the shader's compact code. Depth is
        // not reachable from setHybridDebugView (1..4) and has no sampled
        // path here, so anything else falls through to the normal view.
        uint32_t viewCode = 1;// Normal
        switch (hybridDebugView_) {
            case HybridDebugView::Motion: viewCode = 2; break;
            case HybridDebugView::Ids:    viewCode = 3; break;
            case HybridDebugView::Albedo: viewCode = 4; break;
            case HybridDebugView::Depth:  viewCode = 5; break;
            default:                      viewCode = 1; break;
        }

        auto img = [&](VkImageView view) {
            VkDescriptorImageInfo info{};
            info.sampler     = gbufSampler_;
            info.imageView   = view;
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            return info;
        };
        VkDescriptorImageInfo normalInfo = img(g.normal.view);
        VkDescriptorImageInfo motionInfo = img(g.motion.view);
        VkDescriptorImageInfo idsInfo    = img(g.ids.view);
        VkDescriptorImageInfo albedoInfo = img(g.albedo.view);

        // Depth is a depth-aspect image; it rests in DEPTH_STENCIL_READ_ONLY
        // (the gbuffer render pass's depth finalLayout), not the colour
        // SHADER_READ_ONLY the img() helper uses.
        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler     = gbufSampler_;
        depthInfo.imageView   = g.depth.view;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo outInfo{};
        outInfo.imageView   = ctx->swapchainImageViews()[imageIndex];
        outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 6> w{};
        for (auto& it : w) it.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        const VkDescriptorImageInfo* imgs[4] = {&normalInfo, &motionInfo, &idsInfo, &albedoInfo};
        for (uint32_t i = 0; i < 4; ++i) {
            w[i].dstSet          = debugResolveDescSet_;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo      = imgs[i];
        }
        w[4].dstSet          = debugResolveDescSet_;
        w[4].dstBinding      = 4;
        w[4].descriptorCount = 1;
        w[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[4].pImageInfo      = &outInfo;
        w[5].dstSet          = debugResolveDescSet_;
        w[5].dstBinding      = 5;
        w[5].descriptorCount = 1;
        w[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[5].pImageInfo      = &depthInfo;
        vkUpdateDescriptorSets(ctx->device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);

        // Freshly-acquired swapchain image (contents undefined) → GENERAL
        // for the compute storage write.
        VkImageMemoryBarrier2 toGeneral{};
        toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toGeneral.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        toGeneral.srcAccessMask = 0;
        toGeneral.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = ctx->swapchainImages()[imageIndex];
        toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toGeneral.subresourceRange.levelCount = 1;
        toGeneral.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cb, &dep);

        const VkExtent2D ext = ctx->swapchainExtent();
        DebugResolvePC pc{};
        pc.view       = viewCode;
        pc.width      = ext.width;
        pc.height     = ext.height;
        pc.gbufWidth  = g.width;
        pc.gbufHeight = g.height;
        pc.motionGain = 20.0f;// signed NDC delta is tiny; scale for visibility

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, debugResolvePipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                debugResolvePipelineLayout_, 0, 1, &debugResolveDescSet_, 0, nullptr);
        vkCmdPushConstants(cb, debugResolvePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (ext.width + 7) / 8, (ext.height + 7) / 8, 1);
        // Swapchain left in GENERAL — the shared overlay/present finalize
        // expects exactly that.
    }

}// namespace threepp
