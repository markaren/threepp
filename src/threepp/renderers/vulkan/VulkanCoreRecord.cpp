#include "VulkanCoreImpl.hpp"
#include "threepp/renderers/vulkan/shaders/event_shade.comp.spv.h"

namespace threepp {

void VulkanRendererCore::CoreImpl::recordCommandBuffer(VkCommandBuffer cb, uint32_t imageIndex) {

            // ── Split-screen pane region ───────────────────────────────────
            // When scissorTest is on, clip the deferred-render pane to the
            // scissor sub-rect: render region-sized at the image origin
            // (gbuf/shade/denoise/bloom all read the members below), and
            // offset only the final TAA write to the scissor position.
            // Defaults to the full frame (byte-identical single-scene path)
            // when scissorTest is off.
            {
                const VkExtent2D fullSwap   = ctx->swapchainExtent();
                const VkExtent2D fullRender = renderExtent();
                regionSwapExt_   = fullSwap;
                regionRenderExt_ = fullRender;
                regionDstX_ = 0;
                regionDstY_ = 0;
                if (scissorTest && scissor.z >= 1.f && scissor.w >= 1.f &&
                    fullSwap.width > 0 && fullSwap.height > 0) {
                    // Cap sx at width-1 (not width) so the sw clamp below always
                    // has hi = width-sx >= 1; a degenerate scissor.x >= width
                    // would otherwise make std::clamp(.., 1, 0) — lo > hi is UB.
                    const int sx  = std::clamp(static_cast<int>(scissor.x), 0, static_cast<int>(fullSwap.width) - 1);
                    const int sw  = std::clamp(static_cast<int>(scissor.z), 1, static_cast<int>(fullSwap.width) - sx);
                    const int sh  = std::clamp(static_cast<int>(scissor.w), 1, static_cast<int>(fullSwap.height));
                    const int syB = std::clamp(static_cast<int>(scissor.y), 0, static_cast<int>(fullSwap.height) - sh);
                    regionSwapExt_ = {static_cast<uint32_t>(sw), static_cast<uint32_t>(sh)};
                    regionDstX_    = sx;
                    // GL scissor y is from the bottom; the swapchain image is
                    // top-left origin → flip. Full-height panes map to 0.
                    regionDstY_    = static_cast<int>(fullSwap.height) - (syB + sh);
                    regionRenderExt_ = {
                            std::max(1u, static_cast<uint32_t>(std::lround(
                                                 static_cast<double>(sw) * fullRender.width / fullSwap.width))),
                            std::max(1u, static_cast<uint32_t>(std::lround(
                                                 static_cast<double>(sh) * fullRender.height / fullSwap.height)))};
                }
            }

            // ── Skinned-mesh GPU pipeline ──────────────────────────────────
            // ensureSceneBuilt populated pendingSkinnedRebuilds_ with the
            // states whose bones changed this frame and uploaded the new
            // bone matrices to each state's host-visible boneMatrices buffer.
            // Now record: one skinning dispatch per state → barrier → one
            // BLAS rebuild per state → barrier. The BLAS rebuild reads the
            // deformed vertex/normal buffers the dispatch just wrote. The
            // deferred shade / raster downstream reads the BLAS via TLAS.
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

            // ── DisplacedMesh (FFT water) update + BLAS rebuild ─────────────
            // Recorded into the frame cb (no blocking submit), like the skinned
            // and tet paths above. recordDisplacedDeform issues the cascade FFT
            // chain, water_displace + world-foam dispatches, the height-field
            // readback copies (mirrored to the CPU one frame later at stage
            // time) and the in-place BLAS rebuild, with its internal barriers.
            // The publish barrier below covers AS-build→TLAS/RT reads AND the
            // displace-compute writes → raster G-buffer vertex-attribute reads
            // (the one-shot's submit+wait used to serialize those implicitly).
            if (!pendingDisplacedDeforms_.empty() && waterDisplace_) {
                for (auto& [dmPtr, stPtr, tsec] : pendingDisplacedDeforms_) {
                    recordDisplacedDeform(cb, *dmPtr, *stPtr, tsec);
                }
                VkMemoryBarrier2 mb{};
                mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mb.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                   VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
                mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
                VkDependencyInfo dep{};
                dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers    = &mb;
                vkCmdPipelineBarrier2(cb, &dep);
                pendingDisplacedDeforms_.clear();
            }

            // ── GrassMesh wind deform (GPU) + BLAS refit ────────────────────
            // Recorded into the frame cb (no blocking submit), like the skinned
            // and tet paths above. recordGrassDeform issues the grass_wind
            // dispatch + a compute→AS barrier + the in-place BLAS refit per mesh;
            // a final AS-write→RT-read barrier publishes the rebuilt geometry to
            // the trace. (The TLAS sees last frame's bounds — fine for the small
            // sway envelope, same 1-frame-late deal as skinned.)
            if (!pendingGrassDeforms_.empty() && grassWind_) {
                for (auto& [gm, st] : pendingGrassDeforms_) {
                    recordGrassDeform(cb, *gm, *st);
                }
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
                pendingGrassDeforms_.clear();
            }

            // ── Per-frame TLAS refit ────────────────────────────────────────
            // Recorded here (after every deformable BLAS rebuild above) instead
            // of a mid-frame one-shot drain in ensureSceneBuilt — this is what
            // removes the resolution-scaled stall (the old drain blocked on the
            // previous frame's path trace). One submit per frame; CPU/GPU
            // overlap restored.
            if (pendingTlasRefit_) {
                recordTlasRefit(cb, pendingTlasInstances_, pendingTlasFullBuild_);
                VkMemoryBarrier2 mb{};
                mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mb.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                mb.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                VkDependencyInfo dep{};
                dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers    = &mb;
                vkCmdPipelineBarrier2(cb, &dep);
                pendingTlasRefit_ = false;
            }

            // ── Hybrid raster G-buffer pass ─────────────────────────────────
            // Runs ahead of any ray-query work so the gbuffer is ready when
            // the deferred shade wants to read primary visibility. In
            // G-buffer debug mode we blit a chosen channel directly to the
            // swapchain, draw the ImGui overlay on top (mirrors the normal
            // overlay flow), then present — bypassing the shade dispatch
            // entirely.
            if (rasterGbufPipeline != VK_NULL_HANDLE) {
                gpuTimings_->begin(cb, TP_RasterGbuf, currentFrame);
                const bool occlMsaa = gbufMsaaSamples_ > 1 &&
                                      rasterGbufs[currentFrame].framebufferMS != VK_NULL_HANDLE;
                const VkRenderPass  occlA  = occlMsaa ? occlRenderPassAMS_ : occlRenderPassA_;
                const VkRenderPass  occlB  = occlMsaa ? occlRenderPassBMS_ : occlRenderPassB_;
                const VkFramebuffer occlFb = occlMsaa ? rasterGbufs[currentFrame].framebufferMS
                                                      : rasterGbufs[currentFrame].framebuffer;
                if (occlActiveThisFrame_ && occlA != VK_NULL_HANDLE &&
                    occlFb != VK_NULL_HANDLE) {
                    // ── Two-phase occlusion culling ────────────────────────
                    // Filter to last frame's visible set → pass A → farthest
                    // HiZ from its depth (the raw MS attachment under MSAA —
                    // its samples reduce at mip 0) → AABB test → pass B draws
                    // only the newly visible. rasterGbufMs (this timing
                    // scope) covers the WHOLE sequence, so the on/off
                    // comparison is honest.
                    occl_->recordFilter(cb, currentFrame, indirectTotalDraws_);
                    recordRasterGbufPassInternal(cb, currentFrame, occlA, occlFb,
                                                 occlMsaa,
                                                 occl_->phase1Buffer(), /*clear=*/true);
                    occlHiz_->record(cb, currentFrame);
                    occl_->recordCullTest(cb, currentFrame, indirectTotalDraws_,
                                          occlHiz_->mips(), renderExtent());
                    recordRasterGbufPassInternal(cb, currentFrame, occlB, occlFb,
                                                 occlMsaa,
                                                 occl_->phase2Buffer(), /*clear=*/false);
                } else {
                    recordRasterGbufPass(cb, currentFrame);
                }
                gpuTimings_->end(cb, TP_RasterGbuf, currentFrame);

                // ── MSAA dominant-sample resolve ────────────────────────────
                // Only when setGbufferMsaa(2|4) is active. The MSAA render
                // pass's own subpass dependency (createRasterGbufRenderPassMS
                // deps[1]) already makes the MS attachments visible to
                // COMPUTE, so gbuf_resolve.comp can read them with no extra
                // barrier. Its writes then need: compute->compute (the
                // depth-resolve fragment shader reads idsResolved) and
                // compute->{fragment,everyone-else} (the resolved colour
                // images + resolved depth feed every existing G-buffer
                // consumer downstream).
                if (gbufMsaaSamples_ > 1 && gbufResolve_ &&
                    rasterGbufs[currentFrame].framebufferMS != VK_NULL_HANDLE) {
                    const VkExtent2D resExt = {rasterGbufs[currentFrame].width, rasterGbufs[currentFrame].height};
                    gpuTimings_->begin(cb, TP_GbufResolve, currentFrame);

                    // The 5 resolved colour images rest at SHADER_READ_ONLY_
                    // OPTIMAL (every existing consumer's expected layout,
                    // same as the msaa=1 render pass's own finalLayout) —
                    // gbuf_resolve.comp's imageStore needs GENERAL. Flip to
                    // GENERAL for the duration of the compute write, then
                    // back below.
                    {
                        VkImage resolveImgs[5] = {
                                rasterGbufs[currentFrame].normal.image, rasterGbufs[currentFrame].motion.image,
                                rasterGbufs[currentFrame].ids.image, rasterGbufs[currentFrame].uv.image,
                                rasterGbufs[currentFrame].albedo.image};
                        VkImageMemoryBarrier2 toGeneral[5]{};
                        for (int i = 0; i < 5; ++i) {
                            toGeneral[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                            toGeneral[i].srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                            toGeneral[i].srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                            toGeneral[i].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                            toGeneral[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                            toGeneral[i].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                            toGeneral[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                            toGeneral[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            toGeneral[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            toGeneral[i].image = resolveImgs[i];
                            toGeneral[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            toGeneral[i].subresourceRange.levelCount = 1;
                            toGeneral[i].subresourceRange.layerCount = 1;
                        }
                        VkDependencyInfo genDep{};
                        genDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        genDep.imageMemoryBarrierCount = 5;
                        genDep.pImageMemoryBarriers    = toGeneral;
                        vkCmdPipelineBarrier2(cb, &genDep);
                    }

                    gbufResolve_->recordComputeResolve(cb, currentFrame, resExt.width, resExt.height, gbufMsaaSamples_);

                    // Compute write (resolved normal/motion/ids/uv/albedo) ->
                    // fragment read (depth-resolve pass reads idsResolved for
                    // the dominant index) + compute/fragment read (every
                    // other consumer). Also flips the layout back to
                    // SHADER_READ_ONLY_OPTIMAL — every consumer (DeferredShade,
                    // TaaResolve, the shade's hybrid set, debug blit) binds these
                    // as COMBINED_IMAGE_SAMPLER at that layout, same contract
                    // as the msaa=1 render pass's own finalLayout.
                    {
                        VkImage resolveImgs[5] = {
                                rasterGbufs[currentFrame].normal.image, rasterGbufs[currentFrame].motion.image,
                                rasterGbufs[currentFrame].ids.image, rasterGbufs[currentFrame].uv.image,
                                rasterGbufs[currentFrame].albedo.image};
                        VkImageMemoryBarrier2 toRead[5]{};
                        for (int i = 0; i < 5; ++i) {
                            toRead[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                            toRead[i].srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                            toRead[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                            toRead[i].dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                      VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                            toRead[i].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                            toRead[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                            toRead[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                            toRead[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            toRead[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            toRead[i].image = resolveImgs[i];
                            toRead[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            toRead[i].subresourceRange.levelCount = 1;
                            toRead[i].subresourceRange.layerCount = 1;
                        }
                        VkDependencyInfo resDep{};
                        resDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        resDep.imageMemoryBarrierCount = 5;
                        resDep.pImageMemoryBarriers    = toRead;
                        vkCmdPipelineBarrier2(cb, &resDep);
                    }

                    // Resolved depth image: UNDEFINED/SHADER_READ_ONLY (from
                    // last frame) -> DEPTH_ATTACHMENT_OPTIMAL for the
                    // fullscreen depth-resolve write.
                    VkImageMemoryBarrier2 toDepthAtt{};
                    toDepthAtt.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    toDepthAtt.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    toDepthAtt.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                    toDepthAtt.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                    toDepthAtt.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                    toDepthAtt.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    toDepthAtt.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                    toDepthAtt.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toDepthAtt.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toDepthAtt.image = rasterGbufs[currentFrame].depth.image;
                    toDepthAtt.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                    toDepthAtt.subresourceRange.levelCount = 1;
                    toDepthAtt.subresourceRange.layerCount = 1;
                    VkDependencyInfo depDep{};
                    depDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    depDep.imageMemoryBarrierCount = 1;
                    depDep.pImageMemoryBarriers    = &toDepthAtt;
                    vkCmdPipelineBarrier2(cb, &depDep);

                    gbufResolve_->recordDepthResolve(cb, currentFrame, resExt.width, resExt.height,
                                                     rasterGbufs[currentFrame].depthMS.view,
                                                     rasterGbufs[currentFrame].ids.view,
                                                     rasterGbufs[currentFrame].depth.view);

                    // DEPTH_ATTACHMENT_OPTIMAL -> DEPTH_STENCIL_READ_ONLY
                    // (the layout every existing consumer — DeferredShade,
                    // TaaResolve, the shade's hybrid set — expects, matching
                    // the 1x render pass's own depth finalLayout).
                    VkImageMemoryBarrier2 toRead{};
                    toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    toRead.srcStageMask  = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                    toRead.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                    toRead.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
                    toRead.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                    toRead.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                    toRead.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toRead.image = rasterGbufs[currentFrame].depth.image;
                    toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                    toRead.subresourceRange.levelCount = 1;
                    toRead.subresourceRange.layerCount = 1;
                    VkDependencyInfo readDep{};
                    readDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    readDep.imageMemoryBarrierCount = 1;
                    readDep.pImageMemoryBarriers    = &toRead;
                    vkCmdPipelineBarrier2(cb, &readDep);

                    gpuTimings_->end(cb, TP_GbufResolve, currentFrame);
                }
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
                    // Split-screen: clip the overlay depth prepass to the pane
                    // region. unjitDepth is full-res but the deferred-render
                    // pane lands at regionDst (size regionSwapExt) after the
                    // TAA write, so the overlay's occluder depth must be laid
                    // down at the same place the color pass reads it. Both
                    // default to the full frame when scissorTest is off
                    // (byte-identical single-scene).
                    VkViewport dvp{float(regionDstX_), float(regionDstY_),
                                   float(regionSwapExt_.width), float(regionSwapExt_.height), 0.f, 1.f};
                    vkCmdSetViewport(cb, 0, 1, &dvp);
                    VkRect2D dsc{{regionDstX_, regionDstY_}, regionSwapExt_};
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
                    // Blit the chosen G-buffer channel to the swapchain and
                    // leave it in GENERAL with the cmd buffer still open — the
                    // exact exit contract of the full deferred path. The shared
                    // tail (overlayPass_ screen-space sprites in beginDeferredFrame,
                    // then endFrame()'s overlay composite + single PRESENT_SRC
                    // transition + submit/present) finalizes the frame
                    // uniformly. This keeps the --shot writeFramebuffer readback
                    // (which expects PRESENT_SRC) consistent and avoids the
                    // double vkEndCommandBuffer the bespoke finalize used to do.
                    recordHybridDebugResolve(cb, imageIndex, currentFrame);
                    return;
                }
            }

            // ── Events-only render mode early-return ───────────────────────
            // High-rate event-camera path (~500 Hz target): the gbuf prepass
            // above wrote everything event_shade needs to produce a clean
            // deterministic luma image. Skip deferred-shade/denoise/TAA/
            // overlay/upscale entirely. The swapchain is cleared to black so
            // the sprite overlay (event accumulator) has a known starting
            // canvas; event_shade + event_detect run after
            // recordCommandBuffer exactly as in the normal render path.
            //
            // Requires the gbuf prepass to have actually run — gated above
            // on rasterGbufPipeline. If it's missing the events-only flag is
            // ignored and we fall through to the full deferred-shade path.
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

            // Ensure prior frames' ReSTIR DI reservoir writes are visible to
            // this frame's read-modify-write. We barrier both ping-pong slots
            // since over consecutive frames the read-from / write-to roles
            // alternate. Layout stays in GENERAL for both storage images.
            VkImageMemoryBarrier2 accumGbufTemplate{};
            accumGbufTemplate.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            accumGbufTemplate.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            accumGbufTemplate.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            accumGbufTemplate.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
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
            // ReSTIR DI reservoir ping-pong: frame N writes one slot, reads the
            // other; the barrier ensures frame N-1's write is visible to frame
            // N's read in the deferred shade's COMPUTE stage.
            preBarriers[1] = accumGbufTemplate; preBarriers[1].image = reservoirPosImagesPP[0].image;
            preBarriers[2] = accumGbufTemplate; preBarriers[2].image = reservoirPosImagesPP[1].image;
            preBarriers[3] = accumGbufTemplate; preBarriers[3].image = reservoirWImagesPP[0].image;
            preBarriers[4] = accumGbufTemplate; preBarriers[4].image = reservoirWImagesPP[1].image;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size());
            dep.pImageMemoryBarriers = preBarriers.data();
            vkCmdPipelineBarrier2(cb, &dep);

            // Split-screen: clear the whole frame to clearColor once so the
            // area outside the deferred-render pane's scissor shows the clear
            // colour. Gated on scissorTest so the default full-screen path is
            // byte-identical (the TAA write covers the whole frame there
            // anyway).
            if (autoClear_ && scissorTest) {
                Color cc;
                cc.copy(clearColor);
                ColorManagement::workingToColorSpace(cc, SRGBColorSpace);
                VkClearColorValue cv{};
                cv.float32[0] = cc.r;
                cv.float32[1] = cc.g;
                cv.float32[2] = cc.b;
                cv.float32[3] = clearAlpha;
                VkImageSubresourceRange clrRange{};
                clrRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                clrRange.levelCount = 1;
                clrRange.layerCount = 1;
                vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &clrRange);
                // Clear (transfer write) → the TAA swapchain store (compute write).
                VkImageMemoryBarrier2 clrBar{};
                clrBar.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                clrBar.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                clrBar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                clrBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                clrBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                clrBar.oldLayout            = VK_IMAGE_LAYOUT_GENERAL;
                clrBar.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
                clrBar.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
                clrBar.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
                clrBar.image                = img;
                clrBar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                clrBar.subresourceRange.levelCount = 1;
                clrBar.subresourceRange.layerCount = 1;
                VkDependencyInfo clrDep{};
                clrDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                clrDep.imageMemoryBarrierCount = 1;
                clrDep.pImageMemoryBarriers    = &clrBar;
                vkCmdPipelineBarrier2(cb, &clrDep);
            }

            // Per-frame set index (unused by the deferred leaf, which drives
            // DeferredShade from currentFrame; kept for the dispatch signature).
            const uint32_t setIdx = currentFrame * imageCount_ + imageIndex;

            // Extents + exposure are shared by both render modes and by the
            // bloom/TAA tail below, so hoist them out of the mode branch.
            const VkExtent2D ext   = ctx->swapchainExtent();
            // Deferred render extent — equals `ext` unless
            // renderScale_ < 1, in which case TAA upsamples to the swapchain.
            const VkExtent2D ptExt = renderExtent();
            const float exposure   = currentExposure();
            uint32_t exposureBits;
            std::memcpy(&exposureBits, &exposure, sizeof(exposureBits));
            // Pre-exposure (physical camera mode; 1.0 otherwise — every
            // consumer's multiply/divide is then an exact no-op). Stash this
            // frame-in-flight's value so the exposure meter can un-bake it
            // (see VulkanRenderer.cpp's readback of preExpHist_[currentFrame]).
            const float preExp     = preExposure();
            preExpHist_[currentFrame] = preExp;
            std::memcpy(&preExpBits_, &preExp, sizeof(preExpBits_));

            // Dispatch the deferred shade compute (VulkanRenderer::Impl);
            // bloom + TAA below are shared.
            recordSceneDispatch(cb, setIdx, ext, ptExt, exposureBits);

            // ── Thin-lens depth of field (opt-in) ──────────────────────────────
            // Defocus the linear-HDR scene BEFORE bloom/composite/TAA so
            // bright bokeh still blooms and tone-maps as HDR. CoC from the
            // camera: aperture = camAperture_ (setCameraExposure — exposure
            // and DoF consume the triplet independently, so this works with
            // physicalCamera off too), focal length from the camera's FOV on
            // a 24 mm full-frame sensor, focus plane at focusDistance_.
            if (dofEnabled_ && dof_ && dof_->valid()) {
                constexpr float kSensorH  = 0.024f;// full-frame sensor height (m)
                constexpr float kMaxCocPx = 32.f;  // full-res radius clamp
                const float f = (kSensorH * 0.5f) / std::max(tanHalfFovY_, 1e-3f);
                const float S = std::max(focusDistance_, f * 2.f);
                const float cocScale = f * f / (std::max(camAperture_, 0.1f) * (S - f)) *
                                       (static_cast<float>(regionRenderExt_.height) / kSensorH) * 0.5f;
                gpuTimings_->begin(cb, TP_Dof, currentFrame);
                dof_->record(cb, currentFrame,
                             regionRenderExt_.width, regionRenderExt_.height,
                             cocScale, S, kMaxCocPx);
                gpuTimings_->end(cb, TP_Dof, currentFrame);
            }

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

            // Intensity normalized by the accumulated level count so the
            // summed pyramid lands at the same overall energy a single-level
            // bloom put out for the same slider value. Computed unconditionally
            // (both orders need it; HDR mode's bloom-add moves into TaaResolve).
            const float effBloomIntensity =
                    bloomIntensity_ / static_cast<float>(std::max(bloom_->levels(), 1u));

#if defined(THREEPP_WITH_DLSS) || defined(THREEPP_WITH_FSR)
            // External upscalers (DLSS/FSR) run full-frame only — split-screen
            // falls through to the TAA path, like the HDR-input order.
            const bool upscalerFullFrame =
                    regionDstX_ == 0 && regionDstY_ == 0 &&
                    regionRenderExt_.width == ptExt.width &&
                    regionRenderExt_.height == ptExt.height &&
                    regionSwapExt_.width == ext.width &&
                    regionSwapExt_.height == ext.height;
#endif
#if defined(THREEPP_WITH_DLSS)
            // ── DLSS SUPER RESOLUTION PATH ─────────────────────────────────────
            // Outranks FSR (useFsr() is false while DLSS runs). Identical seam:
            // linear-HDR sceneHdr (render extent) + reversed-Z depth + NDC-delta
            // motion → upscaled linear HDR into TaaResolve's history WRITE slot
            // (display extent) → PostComposite adds bloom + tonemaps at display
            // res → recordPostFinalize (RCAS) → swapchain. See DlssUpscaler.{hpp,cpp}.
            if (useDlss() && dlss_ && dlss_->valid() && !dlss_->failing() && upscalerFullFrame) {
                bloom_->recordPyramid(cb, currentFrame,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      bloomIntensity_, bloomThreshold_, bloomClamp_);

                const uint32_t writeSlot = vulkan::TaaResolve::writeSlotFor(currentFrame);

                // DLSS frameTimeDelta (ms) — own clock (the TAA dt isn't
                // computed on this path).
                float dlssDtMs = 16.6f;
                {
                    const double now = glfwGetTime();
                    if (dlssPrevTimeSec_ >= 0.0) {
                        const double dt = now - dlssPrevTimeSec_;
                        if (dt > 0.0) dlssDtMs = static_cast<float>(dt * 1000.0);
                    }
                    dlssPrevTimeSec_ = now;
                }

                vulkan::DlssUpscaler::DispatchInputs din{};
                din.cmd          = cb;
                din.colorImage   = bloom_->sceneHdrImage(currentFrame);
                din.colorView    = bloom_->sceneHdrView(currentFrame);
                din.colorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
                din.colorLayout  = VK_IMAGE_LAYOUT_GENERAL;
                din.depthImage   = rasterGbufs[currentFrame].depth.image;
                din.depthView    = rasterGbufs[currentFrame].depth.view;
                din.depthFormat  = VK_FORMAT_D32_SFLOAT;
                din.depthLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                din.motionImage  = rasterGbufs[currentFrame].motion.image;
                din.motionView   = rasterGbufs[currentFrame].motion.view;
                din.motionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                din.motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                din.outputImage  = taa_->historyImage(writeSlot);
                din.outputView   = taa_->historyView(writeSlot);
                din.outputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                din.renderWidth  = regionRenderExt_.width;
                din.renderHeight = regionRenderExt_.height;
                din.displayWidth = regionSwapExt_.width;
                din.displayHeight = regionSwapExt_.height;
                din.jitterX      = dlssJitterX_;
                din.jitterY      = dlssJitterY_;
                // NDC-delta (GL Y-up) → render-pixel (texture Y-down): {0.5W, -0.5H}.
                din.motionScaleX =  0.5f * static_cast<float>(regionRenderExt_.width);
                din.motionScaleY = -0.5f * static_cast<float>(regionRenderExt_.height);
                din.frameTimeDeltaMs = dlssDtMs;
                din.preExposure  = preExp;
                din.reset        = dlssResetNext_;
                // Bias-current-color mask from the G-buffer IDs flags (deformer/
                // wind-animated surfaces — grass — favor the current frame; their
                // shader displacement isn't in the motion vectors, so history
                // ghosts at their edges without this). Mirrors the FSR reactive path.
                din.frame         = currentFrame;
                din.idsView       = rasterGbufs[currentFrame].ids.view;
                din.reactive      = true;
                din.reactiveValue = 0.6f;

                gpuTimings_->begin(cb, TP_TAA, currentFrame);
                dlss_->recordDispatch(din);
                gpuTimings_->end(cb, TP_TAA, currentFrame);
                dlssResetNext_ = false;

                // Make DLSS's UAV writes to the history slot visible to
                // PostComposite's sampled read of the same slot (both compute).
                {
                    VkImageMemoryBarrier2 b{};
                    b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
                    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.image         = taa_->historyImage(writeSlot);
                    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    b.subresourceRange.levelCount = 1;
                    b.subresourceRange.layerCount = 1;
                    VkDependencyInfo dep{};
                    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.imageMemoryBarrierCount = 1;
                    dep.pImageMemoryBarriers    = &b;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                // Tonemap at DISPLAY res, reading the DLSS output (history slot
                // via PostComposite's HDR-mode binding 6), ADDING bloom.
                post_->recordDispatch(cb, currentFrame,
                                      regionSwapExt_.width, regionSwapExt_.height,
                                      static_cast<uint32_t>(toneMapping_),
                                      exposureBits, preExpBits_, envIsBgColor,
                                      effBloomIntensity,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      /*hdrMode=*/true);

                // Finalize hdrOut_ → swapchain (display-referred RCAS or plain copy).
                taa_->recordPostFinalize(cb, currentFrame, imageIndex,
                                         regionSwapExt_.width, regionSwapExt_.height,
                                         sharpenStrength_ > 0.0f, sharpenStrength_);
            } else
#endif
#if defined(THREEPP_WITH_FSR)
            // ── FSR 3.1 UPSCALER PATH ──────────────────────────────────────────
            // Replaces the TAA temporal resolve when FSR is active. Full-frame
            // only (split-screen falls through to the TAA path, like the HDR-input
            // order). FSR takes the linear-HDR sceneHdr (render extent) + reversed-Z
            // depth + NDC-delta motion and writes the upscaled linear HDR into
            // TaaResolve's history WRITE slot (display extent) — PostComposite's
            // HDR-mode binding 6 already reads that slot. PostComposite then adds
            // bloom + tonemaps at display res (FSR, unlike the TAA HDR path, does
            // NOT fold bloom in), and recordPostFinalize sends hdrOut_ to the
            // swapchain via RCAS/copy. See FsrUpscaler.{hpp,cpp}.
            if (useFsr() && fsr_ && fsr_->valid() && upscalerFullFrame) {
                bloom_->recordPyramid(cb, currentFrame,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      bloomIntensity_, bloomThreshold_, bloomClamp_);

                const uint32_t writeSlot = vulkan::TaaResolve::writeSlotFor(currentFrame);

                // FSR frameTimeDelta (ms) — own clock (the TAA dt above isn't
                // computed on this path).
                float fsrDtMs = 16.6f;
                {
                    const double now = glfwGetTime();
                    if (fsrPrevTimeSec_ >= 0.0) {
                        const double dt = now - fsrPrevTimeSec_;
                        if (dt > 0.0) fsrDtMs = static_cast<float>(dt * 1000.0);
                    }
                    fsrPrevTimeSec_ = now;
                }

                vulkan::FsrUpscaler::DispatchInputs fin{};
                fin.cmd          = cb;
                fin.colorImage   = bloom_->sceneHdrImage(currentFrame);
                fin.colorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
                fin.colorLayout  = VK_IMAGE_LAYOUT_GENERAL;
                fin.depthImage   = rasterGbufs[currentFrame].depth.image;
                fin.depthFormat  = VK_FORMAT_D32_SFLOAT;
                fin.depthLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                fin.motionImage  = rasterGbufs[currentFrame].motion.image;
                fin.motionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                fin.motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                fin.outputImage  = taa_->historyImage(writeSlot);
                fin.outputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                fin.renderWidth  = regionRenderExt_.width;
                fin.renderHeight = regionRenderExt_.height;
                fin.displayWidth = regionSwapExt_.width;
                fin.displayHeight = regionSwapExt_.height;
                fin.jitterX      = fsrJitterX_;
                fin.jitterY      = fsrJitterY_;
                // NDC-delta (GL Y-up) → render-pixel (texture Y-down): {0.5W, -0.5H}.
                fin.motionScaleX =  0.5f * static_cast<float>(regionRenderExt_.width);
                fin.motionScaleY = -0.5f * static_cast<float>(regionRenderExt_.height);
                fin.nearPlane    = fsrCamNear_;
                fin.farPlane     = fsrCamFar_;
                fin.fovYRadians  = fsrCamFovY_;
                fin.frameTimeDeltaMs = fsrDtMs;
                fin.preExposure  = preExp;
                fin.reset        = fsrResetNext_;
                fin.sharpen      = false;// the renderer keeps its own display RCAS
                fin.sharpness    = 0.f;
                // Reactive mask: generated inside recordDispatch from the current
                // frame's G-buffer IDs flags (deformer/animated surfaces → less
                // ghosting). idsView is the same attachment PostComposite/TAA read.
                fin.frame         = currentFrame;
                fin.idsView       = rasterGbufs[currentFrame].ids.view;
                fin.reactive      = true;
                fin.reactiveValue = 0.6f;

                gpuTimings_->begin(cb, TP_TAA, currentFrame);
                fsr_->recordDispatch(fin);
                gpuTimings_->end(cb, TP_TAA, currentFrame);
                fsrResetNext_ = false;

                // Make FSR's UAV writes to the history slot visible to
                // PostComposite's sampled read of the same slot (both compute).
                {
                    VkImageMemoryBarrier2 b{};
                    b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
                    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.image         = taa_->historyImage(writeSlot);
                    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    b.subresourceRange.levelCount = 1;
                    b.subresourceRange.layerCount = 1;
                    VkDependencyInfo dep{};
                    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.imageMemoryBarrierCount = 1;
                    dep.pImageMemoryBarriers    = &b;
                    vkCmdPipelineBarrier2(cb, &dep);
                }

                // Tonemap at DISPLAY res, reading the FSR output (history slot via
                // PostComposite's HDR-mode binding 6), ADDING bloom (non-zero
                // intensity — FSR didn't fold it in). Sky mask stays render-extent.
                post_->recordDispatch(cb, currentFrame,
                                      regionSwapExt_.width, regionSwapExt_.height,
                                      static_cast<uint32_t>(toneMapping_),
                                      exposureBits, preExpBits_, envIsBgColor,
                                      effBloomIntensity,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      /*hdrMode=*/true);

                // Finalize hdrOut_ → swapchain (display-referred RCAS or plain copy).
                taa_->recordPostFinalize(cb, currentFrame, imageIndex,
                                         regionSwapExt_.width, regionSwapExt_.height,
                                         sharpenStrength_ > 0.0f, sharpenStrength_);
            } else
#endif
            {
                // ── POST STACK / TAA ────────────────────────────────────────────
                // Bloom pyramid + post composite (HDR post stack). The shade/
                // resolve wrote linear HDR into bloom_->sceneHdr (the shared
                // set's binding 1). The pyramid glows the bright highlights
                // (skipped when bloomIntensity_ == 0); the post composite then
                // closes the HDR path — exposure, white balance, tone map,
                // grade LUT, sRGB — into the TAA input image (render extent).
                bloom_->recordPyramid(cb, currentFrame,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      bloomIntensity_, bloomThreshold_, bloomClamp_);
                post_->recordDispatch(cb, currentFrame,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      static_cast<uint32_t>(toneMapping_),
                                      exposureBits, preExpBits_, envIsBgColor,
                                      effBloomIntensity);

                // Raster TAA / temporal upsampler. Reads denoise output from
                // the TAA input image (render extent), blends with reprojected
                // history (rgba16f, swapchain extent), writes the result
                // straight to the swapchain. When renderScale < 1 the input is
                // lower-res than the output, so this pass IS the upscaler —
                // jittered low-res samples accumulate into the full-res
                // history, reconstructing detail (no separate blit needed).
                gpuTimings_->begin(cb, TP_TAA, currentFrame);
                taa_->recordResolve(cb, currentFrame, imageIndex,
                                    regionRenderExt_.width, regionRenderExt_.height,
                                    regionSwapExt_.width, regionSwapExt_.height, effAlpha, taaDtFrames,
                                    sharpenStrength_ > 0.0f, sharpenStrength_,
                                    taaSkyReproj_.data(),
                                    static_cast<uint32_t>(regionDstX_), static_cast<uint32_t>(regionDstY_),
                                    ptExt.width, ptExt.height, ext.width, ext.height,
                                    taaDepthLin_.data(), motionBlurAmount_,
                                    taaJitterTexels_[0], taaJitterTexels_[1]);
                gpuTimings_->end(cb, TP_TAA, currentFrame);
            }
            // ── End post stack / TAA ────────────────────────────────────────────

            // ── Hybrid raster overlay pass ─────────────────────────────────────
            // Wireframe-flagged meshes (any material with wireframe == true)
            // and overlay-layer-tagged meshes drawn on top of the post-TAA
            // image. Depth-tested against the existing raster G-buffer depth
            // so overlays are correctly occluded by the rendered scene's
            // surfaces. No depth writes — the depth attachment stays
            // unchanged for the next frame's ray-query + denoise consumption.
            // Line/LineSegments objects are drawn here too, via their own
            // topology pipelines and the per-Line vertex buffer cache
            // (ensureLineGeometryUploaded).
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

                    // Edge-AA coverage mask (attachment 1) — cleared each frame,
                    // written by the overlay pipelines, consumed by the masked
                    // FXAA pass below when Canvas antialiasing is on. The mask
                    // must be bound regardless (the pipelines declare it).
                    ensureOverlayAaImages(ext);
                    {
                        VkImageMemoryBarrier2 maskToCa{};
                        maskToCa.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        // WAR vs the previous frame's AA fragment read; contents
                        // discarded (UNDEFINED) since loadOp clears.
                        maskToCa.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                        maskToCa.srcAccessMask = 0;
                        maskToCa.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        maskToCa.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        maskToCa.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        maskToCa.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        maskToCa.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        maskToCa.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        maskToCa.image = overlayAaMask_.image;
                        maskToCa.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        maskToCa.subresourceRange.levelCount = 1;
                        maskToCa.subresourceRange.layerCount = 1;
                        VkDependencyInfo dM{};
                        dM.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        dM.imageMemoryBarrierCount = 1;
                        dM.pImageMemoryBarriers = &maskToCa;
                        vkCmdPipelineBarrier2(cb, &dM);
                    }

                    VkRenderingAttachmentInfo colorAtts[2]{};
                    VkRenderingAttachmentInfo& colorAtt = colorAtts[0];
                    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    colorAtt.imageView   = ctx->swapchainImageViews()[imageIndex];
                    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                    colorAtts[1].sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    colorAtts[1].imageView   = overlayAaMask_.view;
                    colorAtts[1].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    colorAtts[1].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    colorAtts[1].storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                    colorAtts[1].clearValue.color = {{0.f, 0.f, 0.f, 0.f}};

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
                    ri.colorAttachmentCount = 2;
                    ri.pColorAttachments = colorAtts;
                    ri.pDepthAttachment = &depthAtt;
                    vkCmdBeginRendering(cb, &ri);

                    // Pipeline is selected per-draw based on material.wireframe.
                    // Set viewport/scissor once — they're dynamic state shared
                    // across the wireframe + basic pipelines. Split-screen: clip
                    // to the pane region (same as the depth prepass above) so the
                    // scene overlays — live point cloud, wireframes, lines — land
                    // in the deferred-render pane instead of spanning the whole
                    // window.
                    // regionSwapExt_ == full extent when scissorTest is off.
                    VkViewport vpDyn{float(regionDstX_), float(regionDstY_),
                                     float(regionSwapExt_.width), float(regionSwapExt_.height), 0.f, 1.f};
                    vkCmdSetViewport(cb, 0, 1, &vpDyn);
                    VkRect2D scDyn{{regionDstX_, regionDstY_}, regionSwapExt_};
                    vkCmdSetScissor(cb, 0, 1, &scDyn);

                    Matrix4 vpUnjitMat;
                    std::memcpy(vpUnjitMat.elements.data(), currVPunjit_.data(), 64);

                    // Track currently-bound pipeline so we don't redundantly
                    // re-bind on every draw when the scene's overlay objects
                    // share a mode (most do).
                    VkPipeline curPipeline = VK_NULL_HANDLE;

                    auto drawOverlayMesh = [&](const MeshEntry& en) {
                        Color color(1.f, 1.f, 1.f);
                        float opacity = 1.0f;
                        bool wireframe = false;
                        bool transparent = false;
                        Side side = Side::Front;
                        if (auto* m = en.mesh->material().get()) {
                            if (auto* mc = dynamic_cast<MaterialWithColor*>(m)) {
                                color = mc->color;
                            }
                            if (auto* mw = dynamic_cast<MaterialWithWireframe*>(m)) {
                                wireframe = mw->wireframe;
                            }
                            opacity     = m->opacity;
                            transparent = m->transparent;
                            side        = m->side;
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
                        if (!wireframe) {
                            // Fill pipelines carry cull mode as dynamic state so
                            // material.side is honoured — Side::Double (SVG/UI
                            // layers, often mirrored by negative scale) must not
                            // be back-face culled. Mirrors cacheCullFlags().
                            const VkCullModeFlags cull =
                                    side == Side::Front  ? VK_CULL_MODE_BACK_BIT
                                    : side == Side::Back ? VK_CULL_MODE_FRONT_BIT
                                                         : VK_CULL_MODE_NONE;
                            vkCmdSetCullMode(cb, cull);
                        }

                        const BlasRecord* rec = resolveBlasForEntry(en);
                        if (!rec || rec->vertex.handle == VK_NULL_HANDLE) return;

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
                    };

                    // ── Line / LineSegments / Points draws ─────────────────
                    // For each entry: ensure geom upload, push MVP+color,
                    // switch topology pipeline. When material.vertexColors is
                    // true AND the geometry has a color attribute → bind the
                    // colored pipeline variant + a second vertex binding at
                    // location 1.
                    //
                    // Points entries (isPoints == true) always use the
                    // POINT_LIST pipeline; the push constant's color.w slot
                    // carries PointsMaterial::size instead of opacity.
                    auto drawOverlayLine = [&](const LineEntry& le) {
                        std::shared_ptr<BufferGeometry> geomPtr;
                        std::shared_ptr<Material>       matPtr;
                        if (le.isPoints) {
                            if (!le.points) return;
                            geomPtr = le.points->geometry();
                            matPtr  = le.points->material();
                        } else {
                            if (!le.line) return;
                            geomPtr = le.line->geometry();
                            matPtr  = le.line->material();
                        }
                        if (!geomPtr) return;
                        const vulkan::LineRec* lrec = ensureLineGeometryUploaded(geomPtr.get());
                        if (!lrec || lrec->vertex.handle == VK_NULL_HANDLE) return;

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
                            if (lrec->color.handle == VK_NULL_HANDLE) return;
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
                    };

                    // ── Ordered dispatch (GL parity) ────────────────────────
                    // Nothing here writes depth, so draw order IS composite
                    // order. Mirror GLRenderer: opaque overlays first (in
                    // traversal order), then transparent ones back-to-front by
                    // view-space depth. The sort is STABLE, so exactly-coplanar
                    // transparent meshes — layered SVG fills — keep their scene
                    // traversal order, i.e. SVG paint order. (Previously lines
                    // always drew after meshes, putting grids/gizmos on top of
                    // transparent panes GL sorts behind them.)
                    struct OverlayItem {
                        bool  isLine;
                        size_t idx;
                        bool  transparent;
                        float viewZ;// camera-space z (negative in front; smaller = farther)
                    };
                    std::vector<OverlayItem> overlayItems;
                    overlayItems.reserve(lastVisibleLines_.size() + 16);
                    Matrix4 viewUnjitM;
                    std::memcpy(viewUnjitM.elements.data(), currViewUnjit_.data(), 64);
                    const auto& ve = viewUnjitM.elements;
                    auto viewZOf = [&](const std::array<float, 16>& world) {
                        return ve[2] * world[12] + ve[6] * world[13] +
                               ve[10] * world[14] + ve[14];
                    };
                    for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
                        const auto& en = lastVisibleEntries_[i];
                        if (!en.mesh || !en.isOverlay) continue;
                        // Particle billboards are isOverlay but drawn by the
                        // dedicated billboard loop below — their un-expanded
                        // quads would render as zero-area triangles here.
                        if (en.isParticle) continue;
                        bool tr = false;
                        if (auto* m = en.mesh->material().get()) {
                            // Wireframe draws opaque regardless (see drawOverlayMesh).
                            auto* mw = dynamic_cast<MaterialWithWireframe*>(m);
                            tr = m->transparent && !(mw && mw->wireframe);
                        }
                        overlayItems.push_back({false, i, tr, viewZOf(en.worldMatrix)});
                    }
                    for (size_t i = 0; i < lastVisibleLines_.size(); ++i) {
                        const auto& le = lastVisibleLines_[i];
                        const Material* m = le.isPoints
                                                    ? (le.points ? le.points->material().get() : nullptr)
                                                    : (le.line ? le.line->material().get() : nullptr);
                        overlayItems.push_back({true, i, m && m->transparent, viewZOf(le.worldMatrix)});
                    }
                    std::stable_sort(overlayItems.begin(), overlayItems.end(),
                                     [](const OverlayItem& a, const OverlayItem& b) {
                                         if (a.transparent != b.transparent) return !a.transparent;
                                         if (!a.transparent) return false;// opaque: keep traversal order
                                         return a.viewZ < b.viewZ;         // transparent: back-to-front
                                     });
                    for (const auto& it : overlayItems) {
                        if (it.isLine) drawOverlayLine(lastVisibleLines_[it.idx]);
                        else           drawOverlayMesh(lastVisibleEntries_[it.idx]);
                    }

                    // ── Particle billboards ────────────────────────────────
                    // ParticleSystem meshes (isParticle) are drawn here, after
                    // the wireframe/line/point overlays, in the same render-pass
                    // instance (reuses the viewport/scissor set above — split-
                    // screen aware). Each is a billboard quad expanded in
                    // particle.vert from per-vertex {size,angle,opacity}; depth-
                    // tested against unjitDepth (Normal) or unconditionally drawn
                    // (Additive). The vertex data lives in particleGeomCache_, the
                    // texture in particleTexCache_ (white default if untextured).
                    if (particlePipelineNormal_ != VK_NULL_HANDLE) {
                        bool anyParticle = false;
                        for (const auto& en : lastVisibleEntries_) {
                            if (en.isParticle && en.mesh) { anyParticle = true; break; }
                        }
                        const bool anySprite = !lastVisibleSprites_.empty();
                        if (anyParticle || anySprite) {
                            // Shared per-frame descriptor pool + unjittered camera
                            // for both billboard kinds (particles + world sprites).
                            vkResetDescriptorPool(ctx->device(), particleDescPools_[currentFrame], 0);
                            Matrix4 viewM, projM;
                            std::memcpy(viewM.elements.data(), currViewUnjit_.data(), 64);
                            std::memcpy(projM.elements.data(), currProjUnjit_.data(), 64);

                            // ── Overlay-fog snapshot (Phase 2b) ────────────────
                            // Fog the world-space billboards (chimney smoke) that
                            // the post-TAA overlay draws — they never saw the
                            // unified air-fog / murk medium. particle.frag reads
                            // this at set 1 and applies the closed-form fog. The
                            // inverse-view supplies the camera height + the world-Y
                            // row so the fragment can reconstruct each particle's
                            // world Y from its view-space position. fogInscatter is
                            // the LINEAR air-fog tint (no env term — the accepted
                            // particle-domain approximation); murkInscatter the
                            // linear murk tint. active==0 or a zero-length fog leg
                            // → the shader bypasses, byte-identical to pre-Phase-2b.
                            {
                                Matrix4 viewInv = viewM;
                                viewInv.invert();
                                const auto& iv = viewInv.elements;
                                GpuOverlayFogUbo of{};
                                const bool medium = mediumActiveThisFrame_ || murkDensity_ > 0.f;
                                of.fogActive     = medium ? 1.f : 0.f;
                                of.hfDensity     = mediumActiveThisFrame_ ? mediumDensityThisFrame_ : 0.f;
                                of.hfBaseY       = mediumBaseYThisFrame_;
                                of.hfFalloff     = mediumFalloffThisFrame_;
                                of.murkDensity   = murkDensity_;
                                of.waterSurfaceY = fogWaterSurfaceY_;
                                of.camWorldY     = static_cast<float>(iv[13]);
                                of.viewToWorldY[0] = static_cast<float>(iv[1]);
                                of.viewToWorldY[1] = static_cast<float>(iv[5]);
                                of.viewToWorldY[2] = static_cast<float>(iv[9]);
                                of.fogInscatter[0] = mediumTintThisFrame_[0];
                                of.fogInscatter[1] = mediumTintThisFrame_[1];
                                of.fogInscatter[2] = mediumTintThisFrame_[2];
                                of.murkInscatter[0] = murkColor_[0];
                                of.murkInscatter[1] = murkColor_[1];
                                of.murkInscatter[2] = murkColor_[2];
                                uploadHostVisible(ctx->allocator(), overlayFogUbos_[currentFrame],
                                                  &of, sizeof(of));
                                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                        particlePipelineLayout_, 1, 1,
                                                        &overlayFogDescSets_[currentFrame], 0, nullptr);
                            }
                            VkPipeline curParticlePipe = VK_NULL_HANDLE;

                            for (const auto& en : lastVisibleEntries_) {
                                if (!en.isParticle || !en.mesh) continue;
                                auto geomSp = en.mesh->geometry();
                                const ParticleGeomRec* prec = ensureParticleGeom(geomSp);
                                if (!prec) continue;

                                auto matPtr = en.mesh->material();
                                auto* sm = dynamic_cast<ShaderMaterial*>(matPtr.get());
                                if (!sm) continue;
                                const bool additive = (sm->blending != Blending::Normal);
                                const Texture* tex = nullptr;
                                if (auto uit = sm->uniforms.find("tex");
                                    uit != sm->uniforms.end() && uit->second.hasValue()) {
                                    if (auto** t = std::get_if<Texture*>(&uit->second.value())) {
                                        tex = *t;
                                    }
                                }

                                VkPipeline want = additive ? particlePipelineAdditive_
                                                           : particlePipelineNormal_;
                                if (want != curParticlePipe) {
                                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                                    curParticlePipe = want;
                                }

                                // Resolve the texture (or white default) into a
                                // per-draw descriptor set from this frame's pool.
                                const Image2D* texImg = ensureParticleTexture(tex);
                                if (!texImg) texImg = &particleWhiteTex_;
                                VkDescriptorSetAllocateInfo asi{};
                                asi.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                                asi.descriptorPool     = particleDescPools_[currentFrame];
                                asi.descriptorSetCount = 1;
                                asi.pSetLayouts        = &particleDescSetLayout_;
                                VkDescriptorSet set = VK_NULL_HANDLE;
                                if (vkAllocateDescriptorSets(ctx->device(), &asi, &set) != VK_SUCCESS) continue;
                                VkDescriptorImageInfo dii{};
                                dii.imageView   = texImg->view;
                                dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                dii.sampler     = texImg->sampler;
                                VkWriteDescriptorSet w{};
                                w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                w.dstSet          = set;
                                w.dstBinding      = 0;
                                w.descriptorCount = 1;
                                w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                w.pImageInfo      = &dii;
                                vkUpdateDescriptorSets(ctx->device(), 1, &w, 0, nullptr);
                                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                        particlePipelineLayout_, 0, 1, &set, 0, nullptr);

                                Matrix4 modelM, mvM;
                                std::memcpy(modelM.elements.data(), en.worldMatrix.data(), 64);
                                mvM.multiplyMatrices(viewM, modelM);
                                struct ParticlePC {
                                    float modelView[16];
                                    float proj[16];
                                } pc{};
                                std::memcpy(pc.modelView, mvM.elements.data(), 64);
                                std::memcpy(pc.proj, projM.elements.data(), 64);
                                vkCmdPushConstants(cb, particlePipelineLayout_,
                                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                   0, sizeof(pc), &pc);

                                VkBuffer vbufs[4] = {prec->position.handle, prec->normal.handle,
                                                     prec->uv.handle, prec->color.handle};
                                VkDeviceSize voffs[4] = {0, 0, 0, 0};
                                vkCmdBindVertexBuffers(cb, 0, 4, vbufs, voffs);
                                vkCmdBindIndexBuffer(cb, prec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                                vkCmdDrawIndexed(cb, prec->indexCount, 1, 0, 0, 0);
                            }

                            // ── World-space sprites ────────────────────────
                            // Camera-facing textured billboards positioned in 3D
                            // (TPS impact particles etc.). Shared static quad +
                            // per-sprite push constant (SpritePC). One descriptor
                            // set per unique texture this frame (deduped) since a
                            // burst shares one material/map across many sprites.
                            if (anySprite && spriteWorldPipeline_ != VK_NULL_HANDLE) {
                                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, spriteWorldPipeline_);
                                VkBuffer qv[1] = {spriteQuadVtx_.handle};
                                VkDeviceSize qo[1] = {0};
                                vkCmdBindVertexBuffers(cb, 0, 1, qv, qo);
                                vkCmdBindIndexBuffer(cb, spriteQuadIdx_.handle, 0, VK_INDEX_TYPE_UINT32);

                                std::unordered_map<const Texture*, VkDescriptorSet> setCache;
                                const Texture* curTex = nullptr;
                                bool curTexBound = false;
                                for (const auto& sd : lastVisibleSprites_) {
                                    // Resolve (and cache) the descriptor set for
                                    // this sprite's texture; white default on miss.
                                    if (!curTexBound || sd.tex != curTex) {
                                        VkDescriptorSet set = VK_NULL_HANDLE;
                                        auto cIt = setCache.find(sd.tex);
                                        if (cIt != setCache.end()) {
                                            set = cIt->second;
                                        } else {
                                            const Image2D* texImg = ensureParticleTexture(sd.tex);
                                            if (!texImg) texImg = &particleWhiteTex_;
                                            VkDescriptorSetAllocateInfo asi{};
                                            asi.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                                            asi.descriptorPool     = particleDescPools_[currentFrame];
                                            asi.descriptorSetCount = 1;
                                            asi.pSetLayouts        = &particleDescSetLayout_;
                                            if (vkAllocateDescriptorSets(ctx->device(), &asi, &set) != VK_SUCCESS) continue;
                                            VkDescriptorImageInfo dii{};
                                            dii.imageView   = texImg->view;
                                            dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                            dii.sampler     = texImg->sampler;
                                            VkWriteDescriptorSet w{};
                                            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                            w.dstSet          = set;
                                            w.dstBinding      = 0;
                                            w.descriptorCount = 1;
                                            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                            w.pImageInfo      = &dii;
                                            vkUpdateDescriptorSets(ctx->device(), 1, &w, 0, nullptr);
                                            setCache.emplace(sd.tex, set);
                                        }
                                        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                                particlePipelineLayout_, 0, 1, &set, 0, nullptr);
                                        curTex = sd.tex;
                                        curTexBound = true;
                                    }

                                    // modelView = view · spriteWorld; the billboard
                                    // is rebuilt camera-facing in sprite3d.vert from
                                    // the view-space center + world scale.
                                    Matrix4 modelM, mvM;
                                    std::memcpy(modelM.elements.data(), sd.world.data(), 64);
                                    mvM.multiplyMatrices(viewM, modelM);
                                    Vector3 worldScale;
                                    worldScale.setFromMatrixScale(modelM);
                                    Vector3 mvPos;
                                    mvPos.setFromMatrixPosition(mvM);

                                    struct SpritePC {
                                        float projection[16];
                                        float mvPos[4];
                                        float scale[2];
                                        float center[2];
                                        float color[4];
                                        float rotation;
                                        float pad[3];
                                    } pc{};
                                    std::memcpy(pc.projection, projM.elements.data(), 64);
                                    pc.mvPos[0] = static_cast<float>(mvPos.x);
                                    pc.mvPos[1] = static_cast<float>(mvPos.y);
                                    pc.mvPos[2] = static_cast<float>(mvPos.z);
                                    pc.mvPos[3] = 1.f;
                                    pc.scale[0] = static_cast<float>(worldScale.x);
                                    pc.scale[1] = static_cast<float>(worldScale.y);
                                    pc.center[0] = sd.center.x;
                                    pc.center[1] = sd.center.y;
                                    pc.color[0] = sd.color[0];
                                    pc.color[1] = sd.color[1];
                                    pc.color[2] = sd.color[2];
                                    pc.color[3] = sd.color[3];
                                    pc.rotation = sd.rotation;
                                    vkCmdPushConstants(cb, particlePipelineLayout_,
                                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                       0, sizeof(pc), &pc);
                                    vkCmdDrawIndexed(cb, 6, 1, 0, 0, 0);
                                }
                            }
                        }
                    }

                    vkCmdEndRendering(cb);

                    // ── Masked overlay edge-AA ──────────────────────────────
                    // Only when the user asked for AA (Canvas antialiasing).
                    // Copy the post-overlay swapchain (a pass can't sample its
                    // own target), then FXAA-blend the masked pixels in place
                    // via a fullscreen triangle that discards everywhere else.
                    if (canvas.samples() > 1 && overlayAaPipeline_ != VK_NULL_HANDLE) {
                        VkImageMemoryBarrier2 pre[3]{};
                        // swapchain: attachment write → transfer read
                        pre[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        pre[0].srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        pre[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        pre[0].dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        pre[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                        pre[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        pre[0].image = img;
                        // scratch: discard old contents → transfer dst (WAR vs
                        // the previous frame's fragment sample)
                        pre[1] = pre[0];
                        pre[1].srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                        pre[1].srcAccessMask = 0;
                        pre[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        pre[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        pre[1].image = overlayAaScratch_.image;
                        // mask: attachment write → fragment sample
                        pre[2] = pre[0];
                        pre[2].srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        pre[2].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        pre[2].dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                        pre[2].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                        pre[2].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        pre[2].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        pre[2].image = overlayAaMask_.image;
                        for (auto& b : pre) {
                            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            b.subresourceRange.levelCount = 1;
                            b.subresourceRange.layerCount = 1;
                        }
                        VkDependencyInfo dPre{};
                        dPre.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        dPre.imageMemoryBarrierCount = 3;
                        dPre.pImageMemoryBarriers = pre;
                        vkCmdPipelineBarrier2(cb, &dPre);

                        VkImageCopy region{};
                        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        region.srcSubresource.layerCount = 1;
                        region.dstSubresource = region.srcSubresource;
                        region.extent = {ext.width, ext.height, 1};
                        vkCmdCopyImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       overlayAaScratch_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       1, &region);

                        VkImageMemoryBarrier2 post[2]{};
                        // scratch: transfer write → fragment sample
                        post[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        post[0].srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        post[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        post[0].dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                        post[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                        post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        post[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        post[0].image = overlayAaScratch_.image;
                        // swapchain: transfer read → attachment write again
                        post[1] = post[0];
                        post[1].srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        post[1].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                        post[1].dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        post[1].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        post[1].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        post[1].image = img;
                        for (auto& b : post) {
                            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            b.subresourceRange.levelCount = 1;
                            b.subresourceRange.layerCount = 1;
                        }
                        VkDependencyInfo dPost{};
                        dPost.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        dPost.imageMemoryBarrierCount = 2;
                        dPost.pImageMemoryBarriers = post;
                        vkCmdPipelineBarrier2(cb, &dPost);

                        VkRenderingAttachmentInfo aaAtt{};
                        aaAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                        aaAtt.imageView   = ctx->swapchainImageViews()[imageIndex];
                        aaAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        aaAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                        aaAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                        VkRenderingInfo aaRi{};
                        aaRi.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                        aaRi.renderArea.offset = {0, 0};
                        aaRi.renderArea.extent = ext;
                        aaRi.layerCount = 1;
                        aaRi.colorAttachmentCount = 1;
                        aaRi.pColorAttachments = &aaAtt;
                        vkCmdBeginRendering(cb, &aaRi);
                        VkViewport vpAa{0.f, 0.f, float(ext.width), float(ext.height), 0.f, 1.f};
                        vkCmdSetViewport(cb, 0, 1, &vpAa);
                        VkRect2D scAa{{0, 0}, ext};
                        vkCmdSetScissor(cb, 0, 1, &scAa);
                        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayAaPipeline_);
                        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                overlayAaPipelineLayout_, 0, 1, &overlayAaSet_, 0, nullptr);
                        vkCmdDraw(cb, 3, 1, 0, 0);
                        vkCmdEndRendering(cb);
                    }

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


            // ── End of deferred-render recording. ──────────────────────────────
            // The swapchain image is left in VK_IMAGE_LAYOUT_GENERAL — endFrame
            // handles the ImGui overlay pass + GENERAL → PRESENT_SRC transition,
            // closes the command buffer, and submits.
        }

void VulkanRendererCore::CoreImpl::recordSceneCapture(VkCommandBuffer cb, uint32_t imageIndex) {
            const VkExtent2D ext = ctx->swapchainExtent();
            if (ext.width == 0 || ext.height == 0) return;
            // Copying out of the swapchain requires it to have been created
            // with TRANSFER_SRC usage. setSceneCaptureEnabled throws when the
            // context already exists; this covers a pre-first-render enable.
            if (!ctx->swapchainSupportsTransferSrc()) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    std::cerr << "[VulkanRenderer] scene capture skipped: the surface "
                                 "does not support TRANSFER_SRC swapchain usage\n";
                }
                return;
            }
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

void VulkanRendererCore::CoreImpl::createEventShadePipeline() {
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
            check(vkCreateComputePipelines(ctx->device(), ctx->pipelineCache(), 1, &cpci, nullptr, &eventShadePipeline_),
                  "vkCreateComputePipelines(event_shade)");
            vkDestroyShaderModule(ctx->device(), mod, nullptr);

            // One descriptor set per frame-in-flight, each rewritten every
            // frame (gbuf views + material/light buffers are per-frame). A
            // single shared set updated per-frame would be modified while the
            // OTHER in-flight frame still had it bound in a pending cmd buffer
            // (VUID-vkUpdateDescriptorSets-None-03047) — the GPU then reads the
            // racing update's wrong-frame gbuf, which was the event-camera
            // "binary flicker". Pool sizes scale with the set count.
            std::array<VkDescriptorPoolSize, 3> ps{};
            ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[0].descriptorCount = 2 * kFramesInFlight;
            ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps[1].descriptorCount = 2 * kFramesInFlight;
            ps[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps[2].descriptorCount = 1 * kFramesInFlight;
            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets       = kFramesInFlight;
            dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
            dpci.pPoolSizes    = ps.data();
            check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &eventShadeDescPool_),
                  "vkCreateDescriptorPool(event_shade)");
            std::array<VkDescriptorSetLayout, kFramesInFlight> layouts{};
            layouts.fill(eventShadeDsLayout_);
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool     = eventShadeDescPool_;
            dsai.descriptorSetCount = kFramesInFlight;
            dsai.pSetLayouts        = layouts.data();
            check(vkAllocateDescriptorSets(ctx->device(), &dsai, eventShadeDescSets_.data()),
                  "vkAllocateDescriptorSets(event_shade)");
        }

void VulkanRendererCore::CoreImpl::allocateEventLumaBuffer(uint32_t w, uint32_t h) {
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

void VulkanRendererCore::CoreImpl::recordEventShade(VkCommandBuffer cb, uint32_t frame) {
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

            // The inFlight[frame] fence was waited on at the top of
            // beginDeferredFrame, so this frame's prior use of its own
            // descriptor set has retired — updating it here can't race the
            // other in-flight frame's set.
            VkDescriptorSet ds = eventShadeDescSets_[frame];
            std::array<VkWriteDescriptorSet, 5> w{};
            for (auto& it : w) it.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = ds;
            w[0].dstBinding = 0;
            w[0].descriptorCount = 1;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo = &normalInfo;
            w[1].dstSet = ds;
            w[1].dstBinding = 1;
            w[1].descriptorCount = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[1].pImageInfo = &idsInfo;
            w[2].dstSet = ds;
            w[2].dstBinding = 2;
            w[2].descriptorCount = 1;
            w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[2].pBufferInfo = &matInfo;
            w[3].dstSet = ds;
            w[3].dstBinding = 3;
            w[3].descriptorCount = 1;
            w[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[3].pBufferInfo = &lightsInfo;
            w[4].dstSet = ds;
            w[4].dstBinding = 4;
            w[4].descriptorCount = 1;
            w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[4].pBufferInfo = &lumaInfo;

            vkUpdateDescriptorSets(ctx->device(),
                                    static_cast<uint32_t>(w.size()), w.data(),
                                    0, nullptr);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, eventShadePipeline_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     eventShadePipelineLayout_, 0, 1, &ds, 0, nullptr);

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

void VulkanRendererCore::CoreImpl::recordOverlayAndPresentTransition(VkCommandBuffer cb, uint32_t imageIndex) {
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

}// namespace threepp
