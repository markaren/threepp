#include "VulkanCoreImpl.hpp"
#include "threepp/renderers/vulkan/shaders/event_shade.comp.spv.h"

namespace threepp {


void VulkanRenderer::Impl::updatePaneRegion() {
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
}

void VulkanRenderer::Impl::recordDeformAndTlas(VkCommandBuffer cb) {
            // ── Graduated per-frame dynamic plain meshes ───────────────────
            // CPU deformers that rebake their vertices every frame (Flock's
            // merged bird mesh) — staging upload + vertex/normal copies +
            // batched BLAS refit, recorded like every other deformer here.
            // The drain-based refreshGeomBlasBatch now only serves genuinely
            // occasional edits. Internal barriers publish to the TLAS refit
            // below and the raster/RT reads after it.
            recordDynamicGeomRefits(cb);

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
                    // The set for the slot refreshSkinnedBlas wrote this frame.
                    skinning_->recordDispatch(cb, st->skinDescSet[st->boneSlot],
                                              st->vertexCount);
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
                    tetSkinning_->recordDispatch(cb, st->tetDescSet[st->tetPosSlot], st->vertexCount);
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
                bool timed = true;// TP_Ocean* slots: first displaced mesh only
                for (auto& [dmPtr, stPtr, tsec] : pendingDisplacedDeforms_) {
                    recordDisplacedDeform(cb, *dmPtr, *stPtr, tsec, timed);
                    timed = false;
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
                gpuTimings_->begin(cb, vulkan::TP_TlasRefit, currentFrame);
                recordTlasRefit(cb, pendingTlasInstances_, pendingTlasFullBuild_);
                gpuTimings_->end(cb, vulkan::TP_TlasRefit, currentFrame);
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
}

bool VulkanRenderer::Impl::recordGbufferStage(VkCommandBuffer cb, uint32_t imageIndex) {
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
                                      view().rasterGbufs[currentFrame].framebufferMS != VK_NULL_HANDLE;
                const VkRenderPass  occlA  = occlMsaa ? occlRenderPassAMS_ : occlRenderPassA_;
                const VkRenderPass  occlB  = occlMsaa ? occlRenderPassBMS_ : occlRenderPassB_;
                const VkFramebuffer occlFb = occlMsaa ? view().rasterGbufs[currentFrame].framebufferMS
                                                      : view().rasterGbufs[currentFrame].framebuffer;
                // Secondaries always take the plain pass: occlusion culling is
                // primary-only by scope, and occl_/occlHiz_ are single shared
                // instances — a secondary recording them would clobber the
                // primary's phase buffers and HiZ pyramid.
                if (!view().secondary && occlActiveThisFrame_ && occlA != VK_NULL_HANDLE &&
                    occlFb != VK_NULL_HANDLE) {
                    // ── Two-phase occlusion culling ────────────────────────
                    // Filter to last frame's visible set → pass A → farthest
                    // HiZ from its depth (the raw MS attachment under MSAA —
                    // its samples reduce at mip 0) → AABB test → pass B draws
                    // only the newly visible. rasterGbufMs (this timing
                    // scope) covers the whole sequence, so the on/off
                    // comparison measures like for like.
                    occl_->recordFilter(cb, currentFrame, indirectTotalDraws_);
                    recordRasterGbufPassInternal(cb, currentFrame, occlA, occlFb,
                                                 occlMsaa,
                                                 occl_->phase1Buffer(), /*clear=*/true);
                    occlHiz_->record(cb, currentFrame);
                    occl_->recordCullTest(cb, currentFrame, indirectTotalDraws_,
                                          occlHiz_->mips(), renderExtent());
                    recordRasterGbufPassInternal(cb, currentFrame, occlB, occlFb,
                                                 occlMsaa,
                                                 occl_->phase2Buffer(), /*clear=*/false,
                                                 /*particles=*/false);
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
                // !secondary: defense in depth — a secondary never gets a
                // framebufferMS (createRasterGbufImages gates it), and the
                // shared gbufResolve_ sets name the PRIMARY's images, so
                // resolving here for a secondary would read the wrong view.
                if (!view().secondary && gbufMsaaSamples_ > 1 && gbufResolve_ &&
                    view().rasterGbufs[currentFrame].framebufferMS != VK_NULL_HANDLE) {
                    const VkExtent2D resExt = {view().rasterGbufs[currentFrame].width, view().rasterGbufs[currentFrame].height};
                    gpuTimings_->begin(cb, TP_GbufResolve, currentFrame);

                    // The 5 resolved colour images rest at SHADER_READ_ONLY_
                    // OPTIMAL (every existing consumer's expected layout,
                    // same as the msaa=1 render pass's own finalLayout) —
                    // gbuf_resolve.comp's imageStore needs GENERAL. Flip to
                    // GENERAL for the duration of the compute write, then
                    // back below.
                    {
                        VkImage resolveImgs[5] = {
                                view().rasterGbufs[currentFrame].normal.image, view().rasterGbufs[currentFrame].motion.image,
                                view().rasterGbufs[currentFrame].ids.image, view().rasterGbufs[currentFrame].uv.image,
                                view().rasterGbufs[currentFrame].albedo.image};
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
                                view().rasterGbufs[currentFrame].normal.image, view().rasterGbufs[currentFrame].motion.image,
                                view().rasterGbufs[currentFrame].ids.image, view().rasterGbufs[currentFrame].uv.image,
                                view().rasterGbufs[currentFrame].albedo.image};
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
                    toDepthAtt.image = view().rasterGbufs[currentFrame].depth.image;
                    toDepthAtt.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                    toDepthAtt.subresourceRange.levelCount = 1;
                    toDepthAtt.subresourceRange.layerCount = 1;
                    VkDependencyInfo depDep{};
                    depDep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    depDep.imageMemoryBarrierCount = 1;
                    depDep.pImageMemoryBarriers    = &toDepthAtt;
                    vkCmdPipelineBarrier2(cb, &depDep);

                    gbufResolve_->recordDepthResolve(cb, currentFrame, resExt.width, resExt.height,
                                                     view().rasterGbufs[currentFrame].depthMS.view,
                                                     view().rasterGbufs[currentFrame].ids.view,
                                                     view().rasterGbufs[currentFrame].depth.view);

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
                    toRead.image = view().rasterGbufs[currentFrame].depth.image;
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
                // Fills the overlay pass's depth attachment with the
                // unjittered VP. Consumed by the post-TAA wireframe overlay
                // pass for occlusion testing. Only runs when an overlay
                // pipeline exists AND the scene actually has overlay
                // candidates this frame (else the prepass is wasted work).
                //
                // PRIMARY ONLY. The overlay itself is primary-only by scope,
                // and the prepass touches SHARED state: ensureOverlayMsaaImages
                // below sizes overlayMsColor_/overlayMsDepth_/overlayAaScratch_
                // to THIS view's extent and rewrites overlayInjectSet_ — a
                // single persistent set. A secondary running this re-sized
                // those to its own (smaller) extent and updated a set already
                // bound in the open command buffer, which invalidates the
                // ENTIRE buffer: every later draw silently becomes garbage
                // (corrupted gizmo/overlay, missing markers) and the submit is
                // free to end in VK_ERROR_DEVICE_LOST.
                if (!view().secondary &&
                    overlayDepthPrepassPipeline != VK_NULL_HANDLE && sceneHasOverlayContent()) {
                    gpuTimings_->begin(cb, TP_OverlayDepth, currentFrame);
                    // Swapchain extent — the depth target is full-res so the
                    // post-TAA overlay can depth-test the upscaled image.
                    const VkExtent2D dext = viewOutExtent();
                    // Hardware-MSAA overlay: rasterize the occluders into the
                    // multisampled overlayMsDepth_ so the overlay's depth test
                    // is correct PER SAMPLE (a 1-sample depth buffer would
                    // quantise every overlay edge back to whole pixels and
                    // throw the MSAA away). Allocated here because the prepass
                    // is the first consumer in the frame; idempotent.
                    ensureOverlayMsaaImages(dext);
                    const bool   overlayMsaa = overlaySamples() > 1;
                    VkImage      depthImg  = overlayMsaa ? overlayMsDepth_.image
                                                         : view().rasterGbufs[currentFrame].unjitDepth.image;
                    VkImageView  depthView = overlayMsaa ? overlayMsDepth_.view
                                                         : view().rasterGbufs[currentFrame].unjitDepth.view;

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
                                            &view().rasterDescSets[currentFrame], 0, nullptr);
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
                        if (en.sensorOnly) continue;// primary-only pass, and the primary never sees them
                        // Frustum cull, same lever as the gbuf prepass — read
                        // from THIS view's results. The overlay only ever runs
                        // for the primary, and this is the reader that a shared
                        // cull bit would have fed the last secondary's answer.
                        if (!viewCulled(i)) continue;
                        // ParticleField: its BlasRecord is the MeshRepr PROXY,
                        // and this fixed-function pass has no per-particle
                        // vertex stage — drawing it here would put ONE proxy at
                        // the field origin into the overlay depth buffer. Phase
                        // 1's answer is that particles do not occlude overlay
                        // meshes; the alternative is a second particle pipeline
                        // for a depth-only pass, which no consumer has asked for.
                        if (en.isParticleField) continue;
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
                            // Packed static records store uint16 indices (bit 3).
                            vkCmdBindIndexBuffer(cb, rec->index.handle, 0,
                                                 (rec->packedMask & 8u) ? VK_INDEX_TYPE_UINT16
                                                                        : VK_INDEX_TYPE_UINT32);
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
                    return true;
                }
            }

            return false;
}

bool VulkanRenderer::Impl::recordEventsOnlyFrame(VkCommandBuffer cb, uint32_t imageIndex) {
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

                return true;
            }
            // ── End events-only render mode ─────────────────────────────────

            return false;
}

void VulkanRenderer::Impl::recordSwapchainPrepare(VkCommandBuffer cb, uint32_t imageIndex) {
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
            preBarriers[1] = accumGbufTemplate; preBarriers[1].image = view().reservoirPosImagesPP[0].image;
            preBarriers[2] = accumGbufTemplate; preBarriers[2].image = view().reservoirPosImagesPP[1].image;
            preBarriers[3] = accumGbufTemplate; preBarriers[3].image = view().reservoirWImagesPP[0].image;
            preBarriers[4] = accumGbufTemplate; preBarriers[4].image = view().reservoirWImagesPP[1].image;
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
}

void VulkanRenderer::Impl::recordSplats(VkCommandBuffer cb) {
            // ── Gaussian splats into the linear-HDR scene ──────────────────────
            // After the shade (so the G-buffer depth it tests against, and the
            // sceneHdr it composites over, are both final) and before the DoF
            // (so bokeh, bloom, tone mapping and TAA all act on splats without
            // any of it being re-derived here). See vulkan/SplatPass.hpp.
            //
            // A secondary view renders splats only if it ASKED (setViewSplats)
            // and got a target slot. The scratch is shared and reused
            // sequentially, exactly as it already is across clouds — see
            // SplatPass::resize.
            if (!splat_) return;
            if (view().secondary) {
                recordSecondaryViewSplats(cb);
                return;
            }

            // Before the hasClouds() test, not after: the depth AOV
            // describes THIS frame, and a frame that draws no splats has to
            // leave an empty one rather than whatever the previous user of
            // this frame-in-flight slot wrote. Cheap when the AOV is off —
            // the image is one texel then.
            if (splatDepthAovAllocated()) splat_->clearDepthAov(cb, currentFrame);

            if (!splat_->hasClouds()) return;

            auto p = splatParams_;
            p.depthAov    = splatDepthAovAllocated();
            p.depthMedian = splatDepthMedian();
            // Same sub-pixel jitter the raster prepass used this frame,
            // reapplied as the same projection shear (uploadRasterCameraUbo):
            // without it the splats sit a fraction of a pixel off the geometry
            // they are depth-tested against, and TAA resolves a cloud that
            // never moved with the frame it is part of.
            const VkExtent2D ext = renderExtent();
            const float jClipX = 2.f * view().taaJitterTexels_[0] / static_cast<float>(ext.width);
            const float jClipY = 2.f * view().taaJitterTexels_[1] / static_cast<float>(ext.height);
            if (view().orthoFrame_) {
                p.proj[12] -= jClipX;
                p.proj[13] -= jClipY;
                p.jitterClip[0] = -jClipX;
                p.jitterClip[1] = -jClipY;
            } else {
                p.proj[8] += jClipX;
                p.proj[9] += jClipY;
                p.jitterClip[0] = jClipX;
                p.jitterClip[1] = jClipY;
            }

            // View space -> previous frame's unjittered clip, in one matrix.
            // taaSkyReproj_ is prevVP * inverse(currVPunjittered) and was built
            // in uploadRasterCameraUbo BEFORE rasterPrevVP_ rolled over to this
            // frame, so composing it with this frame's projection is the only
            // way to get the real previous VP here — and it guarantees the
            // splat motion vectors and the raster's come from the same
            // matrices.
            {
                Matrix4 sky, projRev, prev;
                std::memcpy(sky.elements.data(), view().taaSkyReproj_.data(), 64);
                std::memcpy(projRev.elements.data(), splatProjRevZ_, 64);
                prev.multiplyMatrices(sky, projRev);
                std::memcpy(p.prevVPfromView, prev.elements.data(), 64);
            }

            // A medium is active when there is anything to extinguish along the
            // camera->splat leg. Both terms are the ones splat_common.glsl's
            // splatFog actually evaluates; with neither, the fog branch is
            // skipped outright and the frame is bit-identical to a fogless one.
            p.fog = (heightFogEnabled_ && heightFogDensity_ > 0.f) || murkDensity_ > 0.f;
            if (const char* e = std::getenv("THREEPP_VK_SPLAT_NOMOTION"); e && *e && *e != '0')
                p.motionVectors = false;
            if (const char* e = std::getenv("THREEPP_VK_SPLAT_NOFOG"); e && *e && *e != '0')
                p.fog = false;
            // The factor the shade already baked into every sceneHdr store.
            // Getting this wrong is invisible until the physical camera is on,
            // and then the splats are wrong by the whole exposure gain.
            p.preExposure = preExposure();
            p.bgIsSolidColor = envIsBgColor;
            p.checksum       = splatChecksum_;

            p.timings        = gpuTimings_.get();// per-stage split, from inside
            gpuTimings_->begin(cb, TP_Splat, currentFrame);
            splat_->record(cb, currentFrame, p);
            gpuTimings_->end(cb, TP_Splat, currentFrame);

}

bool VulkanRenderer::Impl::splatStampPrepare(VkCommandBuffer cb) {
            // ── Splat depth -> the overlay's depth attachment ───────────────────
            // The one write that makes a cloud occlude the post-resolve overlay.
            // Everything the overlay pass draws — wireframe meshes, Lines,
            // world sprites, particle billboards — is rasterized AFTER the
            // splats are composited, and depth-tests against a prepass buffer
            // filled from scene geometry only. The compositor is a compute pass
            // with no depth attachment, so it reads that buffer and never adds
            // to it: a line behind a cloud passed the test and drew over the
            // cloud at full strength. GL has no such gap — it draws the cloud
            // LAST, in the transparent pass, over the lines. See
            // shaders/splat_overlay_depth.frag.
            //
            // This half gates and issues the pre-pass barriers, called before
            // the hybrid overlay pass begins; true means that pass must attach
            // its depth WRITABLE and record recordSplatStampDraw at the
            // exempt/occluded boundary of its draw order.
            //
            // Gated exactly like the pass it feeds. The AOV gate is the one
            // that matters: without it the image is one texel, and
            // splatOverlayDepth_ (latched in collectSplatClouds) is what turns
            // it on for scenes that need this.
            if (splatStampPipeline_ == VK_NULL_HANDLE) return false;
            if (view().secondary) return false;// overlay pass is primary-only by scope
            if (!splat_ || !splat_->hasClouds()) return false;
            if (!splatDepthAovAllocated()) return false;
            if (!sceneHasOverlayContent()) return false;
            if (overlayDepthPrepassPipeline == VK_NULL_HANDLE) return false;

            const bool  overlayMsaa = overlaySamples() > 1;
            const VkImage depthImg  = overlayMsaa ? overlayMsDepth_.image
                                                  : view().rasterGbufs[currentFrame].unjitDepth.image;
            const VkImageView depthView = overlayMsaa ? overlayMsDepth_.view
                                                      : view().rasterGbufs[currentFrame].unjitDepth.view;
            if (depthImg == VK_NULL_HANDLE || depthView == VK_NULL_HANDLE) return false;

            const auto& g = view().rasterGbufs[currentFrame];
            if (g.splatDepth.view == VK_NULL_HANDLE) return false;

            // Per-frame-in-flight set, rewritten only when the AOV view handle
            // actually changed (a resize / render-scale realloc). The set for
            // THIS frame slot cannot be in flight — its fence was waited on
            // before recording began — so the update is safe where a shared set
            // would be VUID-vkCmdBindDescriptorSets-...-03047 material.
            if (splatStampSetViews_[currentFrame] != g.splatDepth.view) {
                VkDescriptorImageInfo ii{VK_NULL_HANDLE, g.splatDepth.view, VK_IMAGE_LAYOUT_GENERAL};
                VkWriteDescriptorSet w{};
                w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet          = splatStampSets_[currentFrame];
                w.dstBinding      = 0;
                w.descriptorCount = 1;
                w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w.pImageInfo      = &ii;
                vkUpdateDescriptorSets(ctx->device(), 1, &w, 0, nullptr);
                splatStampSetViews_[currentFrame] = g.splatDepth.view;
            }

            // No timing bracket: TP_OverlayDepth's slot pair is already spent
            // by the prepass this frame, and writing it twice is
            // VUID-vkCmdWriteTimestamp2-None-03864 plus a meaningless number.
            // One fullscreen depth-only draw is not worth its own pass slot;
            // it lands inside TP_Frame like the rest of the unbracketed work.

            // The AOV's stores (compute) -> the stamp's fragment reads. The
            // image stays in GENERAL throughout the frame, so this is an
            // execution + visibility barrier only, no transition.
            VkMemoryBarrier2 aovVis{};
            aovVis.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            aovVis.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            aovVis.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            aovVis.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            aovVis.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

            // ...and the depth attachment out of the read-only layout the
            // prepass left it in: the overlay pass will attach it WRITABLE so
            // the stamp draw inside it can write. Nothing reads it between the
            // end of that pass and the next frame's prepass, and the prepass
            // transitions from UNDEFINED (it clears), so the layout is not
            // handed back.
            VkImageMemoryBarrier2 toWrite{};
            toWrite.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toWrite.srcStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            toWrite.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            toWrite.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            toWrite.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            toWrite.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            toWrite.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            toWrite.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toWrite.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toWrite.image = depthImg;
            toWrite.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            toWrite.subresourceRange.levelCount = 1;
            toWrite.subresourceRange.layerCount = 1;

            VkDependencyInfo di{};
            di.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount       = 1;
            di.pMemoryBarriers          = &aovVis;
            di.imageMemoryBarrierCount  = 1;
            di.pImageMemoryBarriers     = &toWrite;
            vkCmdPipelineBarrier2(cb, &di);
            return true;
}

void VulkanRenderer::Impl::recordSplatStampDraw(VkCommandBuffer cb) {
            // The stamp draw itself, recorded INSIDE the hybrid overlay pass —
            // after the overlays that kSplatUnoccludedOverlayLayer exempts,
            // before everything else — so "occluded by a cloud" is a position
            // in the pass's draw order rather than a property of the whole
            // pass. splatStampPrepare gated this frame and issued the
            // barriers; here is only the draw.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, splatStampPipeline_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    splatStampPipelineLayout_, 0, 1,
                                    &splatStampSets_[currentFrame], 0, nullptr);
            // Split-screen: the same pane the depth prepass laid its occluders
            // into, and the same one the TAA wrote this view's image to.
            VkViewport vp{float(regionDstX_), float(regionDstY_),
                          float(regionSwapExt_.width), float(regionSwapExt_.height), 0.f, 1.f};
            vkCmdSetViewport(cb, 0, 1, &vp);
            VkRect2D sc{{regionDstX_, regionDstY_}, regionSwapExt_};
            vkCmdSetScissor(cb, 0, 1, &sc);

            SplatStampPC pc{};
            const VkExtent2D aov = renderExtent();
            pc.aovScale[0]   = float(aov.width) / float(std::max(regionSwapExt_.width, 1u));
            pc.aovScale[1]   = float(aov.height) / float(std::max(regionSwapExt_.height, 1u));
            pc.paneOrigin[0] = float(regionDstX_);
            pc.paneOrigin[1] = float(regionDstY_);
            pc.aovLimit[0]   = float(aov.width) - 1.f;
            pc.aovLimit[1]   = float(aov.height) - 1.f;
            // The reverse-Z projection the raster depth (and therefore the
            // prepass buffer this stamp competes with) was written with. Only
            // the z row is read, and the TAA jitter lives in the x/y shear —
            // so the jittered and unjittered forms agree here, which is why
            // this can use the splat pass's own matrix.
            pc.projA = splatProjRevZ_[10];
            pc.projB = splatProjRevZ_[14];
            pc.ortho = view().orthoFrame_ ? 1u : 0u;
            vkCmdPushConstants(cb, splatStampPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);
            vkCmdDraw(cb, 3, 1, 0, 0);
}

void VulkanRenderer::Impl::recordSecondaryViewSplats(VkCommandBuffer cb) {
            // ── Splats into a secondary view's linear-HDR scene ─────────────────
            // Same slot in the frame as the primary's (after the shade, before
            // the bloom pyramid) and the same pipeline, on this view's own
            // target slot. Opt-in per view: the sort scales with SPLAT COUNT
            // rather than view size, so a second view is a second full sort and
            // that is the caller's call to make.
            //
            // Primary-only here, deliberately: the depth AOV (a secondary's AOV
            // image is 1x1 by construction) and the debug checksum (one readback
            // buffer per frame slot, which a second writer would overwrite).
            ViewContext& v = view();
            if (!v.splats || v.splatTarget == vulkan::SplatPass::kNoTarget) return;
            if (!v.camera || !splat_->hasClouds()) return;

            // The cloud list, the model matrices and the per-cloud percentile
            // interval were all built by collectSplatClouds for the PRIMARY
            // camera and are reused verbatim. Only the percentiles are camera-
            // dependent, and they are a sort-key content interval rather than a
            // clamp — the exact min/max still come from this view's own
            // projection pass, so a splat outside the interval lands in a
            // monotone tail bucket instead of collapsing onto the end.
            Camera& cam = *v.camera;
            auto p = splatParams_;
            p.target      = v.splatTarget;
            p.depthAov    = false;
            p.depthMedian = false;
            p.checksum    = false;
            p.timings     = nullptr;// suppressed for secondaries, like every other pass
            std::memcpy(p.view, cam.matrixWorldInverse.elements.data(), 64);
            std::memcpy(p.proj, cam.projectionMatrix.elements.data(), 64);
            std::memcpy(p.camWorld, cam.matrixWorld->elements.data(), 64);
            const Matrix4 projRev = reverseZVk(cam.projectionMatrix);
            Matrix4 projInv;
            projInv.copy(projRev).invert();
            std::memcpy(p.projInverse, projInv.elements.data(), 64);
            const auto& cw = cam.matrixWorld->elements;
            p.camPos[0] = cw[12];
            p.camPos[1] = cw[13];
            p.camPos[2] = cw[14];
            p.camFwd[0] = -cw[8];
            p.camFwd[1] = -cw[9];
            p.camFwd[2] = -cw[10];
            p.orthographic = v.orthoFrame_;
            if (auto* pc = dynamic_cast<PerspectiveCamera*>(&cam))       p.nearPlane = pc->nearPlane;
            else if (auto* oc = dynamic_cast<OrthographicCamera*>(&cam)) p.nearPlane = oc->nearPlane;

            // This view's own jitter, and this view's own previous VP: a
            // secondary runs the built-in TAA (VulkanCoreFrame.cpp's tail), so
            // the splats have to be jittered with the raster they are composited
            // over and reprojected with the history they are resolved against.
            const VkExtent2D ext = v.renderExt;
            const float jClipX = 2.f * v.taaJitterTexels_[0] / static_cast<float>(ext.width);
            const float jClipY = 2.f * v.taaJitterTexels_[1] / static_cast<float>(ext.height);
            if (v.orthoFrame_) {
                p.proj[12] -= jClipX;
                p.proj[13] -= jClipY;
                p.jitterClip[0] = -jClipX;
                p.jitterClip[1] = -jClipY;
            } else {
                p.proj[8] += jClipX;
                p.proj[9] += jClipY;
                p.jitterClip[0] = jClipX;
                p.jitterClip[1] = jClipY;
            }
            {
                Matrix4 sky, prev;
                std::memcpy(sky.elements.data(), v.taaSkyReproj_.data(), 64);
                prev.multiplyMatrices(sky, projRev);
                std::memcpy(p.prevVPfromView, prev.elements.data(), 64);
            }

            p.fog = (heightFogEnabled_ && heightFogDensity_ > 0.f) || murkDensity_ > 0.f;
            if (const char* e = std::getenv("THREEPP_VK_SPLAT_NOMOTION"); e && *e && *e != '0')
                p.motionVectors = false;
            if (const char* e = std::getenv("THREEPP_VK_SPLAT_NOFOG"); e && *e && *e != '0')
                p.fog = false;
            p.preExposure    = preExposure();
            p.bgIsSolidColor = envIsBgColor;

            splat_->record(cb, currentFrame, p);
}

void VulkanRenderer::Impl::recordDepthOfField(VkCommandBuffer cb) {
            // ── Thin-lens depth of field (opt-in) ──────────────────────────────
            // Defocus the linear-HDR scene BEFORE bloom/composite/TAA so
            // bright bokeh still blooms and tone-maps as HDR. CoC from the
            // camera: aperture = camAperture_ (setCameraExposure — exposure
            // and DoF consume the triplet independently, so this works with
            // physicalCamera off too), focal length from the camera's FOV on
            // the camera's OWN sensor (filmHeightM_, from PerspectiveCamera's
            // film gauge), focus plane at focusDistance_.
            // Not under a parallel projection: an orthographic camera has no
            // lens, so there is no aperture, no focal length and no circle of
            // confusion — every point projects sharp by construction. The CoC
            // derivation below would read a tan(fov/2) the projection never
            // carried and defocus the frame by an arbitrary amount.
            if (dofEnabled_ && !view().orthoFrame_ && dof_ && dof_->valid()) {
                const float     kSensorH  = filmHeightM_;// sensor height (m)
                constexpr float kMaxCocPx = 32.f;        // full-res radius clamp
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
}

void VulkanRenderer::Impl::recordUpscaleAndPost(VkCommandBuffer cb, uint32_t imageIndex,
                                                VkExtent2D ext, VkExtent2D ptExt,
                                                uint32_t exposureBits, float preExp) {
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
                // frameNowSec, not glfwGetTime: this dt sets the history blend
                // weight, so wall time here made the beauty frame irreproducible
                // — every run blended with different alphas from the first
                // history frame onward (found by the replay-audit harness).
                const double now = frameNowSec();
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
                    bloomIntensity_ / static_cast<float>(std::max(view().bloom_->levels(), 1u));

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
                view().bloom_->recordPyramid(cb, currentFrame,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      bloomIntensity_, bloomThreshold_, bloomClamp_);

                const uint32_t writeSlot = vulkan::TaaResolve::writeSlotFor(currentFrame);

                // DLSS frameTimeDelta (ms) — own clock (the TAA dt isn't
                // computed on this path).
                float dlssDtMs = 16.6f;
                {
                    const double now = frameNowSec();
                    if (dlssPrevTimeSec_ >= 0.0) {
                        const double dt = now - dlssPrevTimeSec_;
                        if (dt > 0.0) dlssDtMs = static_cast<float>(dt * 1000.0);
                    }
                    dlssPrevTimeSec_ = now;
                }

                vulkan::DlssUpscaler::DispatchInputs din{};
                din.cmd          = cb;
                din.colorImage   = view().bloom_->sceneHdrImage(currentFrame);
                din.colorView    = view().bloom_->sceneHdrView(currentFrame);
                din.colorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
                din.colorLayout  = VK_IMAGE_LAYOUT_GENERAL;
                din.depthImage   = view().rasterGbufs[currentFrame].depth.image;
                din.depthView    = view().rasterGbufs[currentFrame].depth.view;
                din.depthFormat  = VK_FORMAT_D32_SFLOAT;
                din.depthLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                din.motionImage  = view().rasterGbufs[currentFrame].motion.image;
                din.motionView   = view().rasterGbufs[currentFrame].motion.view;
                din.motionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                din.motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                din.outputImage  = view().taa_->historyImage(writeSlot);
                din.outputView   = view().taa_->historyView(writeSlot);
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
                din.idsView       = view().rasterGbufs[currentFrame].ids.view;
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
                    b.image         = view().taa_->historyImage(writeSlot);
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
                view().post_->recordDispatch(cb, currentFrame,
                                      regionSwapExt_.width, regionSwapExt_.height,
                                      static_cast<uint32_t>(toneMapping_),
                                      exposureBits, preExpBits_, envIsBgColor,
                                      effBloomIntensity,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      /*hdrMode=*/true);

                // Finalize hdrOut_ → swapchain (display-referred RCAS or plain copy).
                view().taa_->recordPostFinalize(cb, currentFrame, imageIndex,
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
                view().bloom_->recordPyramid(cb, currentFrame,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      bloomIntensity_, bloomThreshold_, bloomClamp_);

                const uint32_t writeSlot = vulkan::TaaResolve::writeSlotFor(currentFrame);

                // FSR frameTimeDelta (ms) — own clock (the TAA dt above isn't
                // computed on this path).
                float fsrDtMs = 16.6f;
                {
                    const double now = frameNowSec();
                    if (fsrPrevTimeSec_ >= 0.0) {
                        const double dt = now - fsrPrevTimeSec_;
                        if (dt > 0.0) fsrDtMs = static_cast<float>(dt * 1000.0);
                    }
                    fsrPrevTimeSec_ = now;
                }

                vulkan::FsrUpscaler::DispatchInputs fin{};
                fin.cmd          = cb;
                fin.colorImage   = view().bloom_->sceneHdrImage(currentFrame);
                fin.colorFormat  = VK_FORMAT_R16G16B16A16_SFLOAT;
                fin.colorLayout  = VK_IMAGE_LAYOUT_GENERAL;
                fin.depthImage   = view().rasterGbufs[currentFrame].depth.image;
                fin.depthFormat  = VK_FORMAT_D32_SFLOAT;
                fin.depthLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                fin.motionImage  = view().rasterGbufs[currentFrame].motion.image;
                fin.motionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                fin.motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                fin.outputImage  = view().taa_->historyImage(writeSlot);
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
                fin.idsView       = view().rasterGbufs[currentFrame].ids.view;
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
                    b.image         = view().taa_->historyImage(writeSlot);
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
                view().post_->recordDispatch(cb, currentFrame,
                                      regionSwapExt_.width, regionSwapExt_.height,
                                      static_cast<uint32_t>(toneMapping_),
                                      exposureBits, preExpBits_, envIsBgColor,
                                      effBloomIntensity,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      /*hdrMode=*/true);

                // Finalize hdrOut_ → swapchain (display-referred RCAS or plain copy).
                view().taa_->recordPostFinalize(cb, currentFrame, imageIndex,
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
                view().bloom_->recordPyramid(cb, currentFrame,
                                      regionRenderExt_.width, regionRenderExt_.height,
                                      bloomIntensity_, bloomThreshold_, bloomClamp_);
                view().post_->recordDispatch(cb, currentFrame,
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
                view().taa_->recordResolve(cb, currentFrame, imageIndex,
                                    regionRenderExt_.width, regionRenderExt_.height,
                                    regionSwapExt_.width, regionSwapExt_.height, effAlpha, taaDtFrames,
                                    sharpenStrength_ > 0.0f, sharpenStrength_,
                                    view().taaSkyReproj_.data(),
                                    static_cast<uint32_t>(regionDstX_), static_cast<uint32_t>(regionDstY_),
                                    ptExt.width, ptExt.height, ext.width, ext.height,
                                    view().taaDepthLin_.data(), motionBlurAmount_,
                                    view().taaJitterTexels_[0], view().taaJitterTexels_[1]);
                gpuTimings_->end(cb, TP_TAA, currentFrame);
            }
            // ── End post stack / TAA ────────────────────────────────────────────
}

// Any visible ParticleField draws quads this frame. Read by
// sceneHasOverlayContent() so the overlay depth prepass runs (and unjitDepth is
// therefore valid) for a scene whose only overlay content is a field.
bool VulkanRenderer::Impl::sceneHasFieldBillboards() const {
            for (const auto& [field, entryIndex] : particleFields_) {
                if (field && field->billboardRepr().enabled && field->liveCount() > 0u) return true;
            }
            return false;
        }

// ── ParticleField billboards — one indirect draw per field ──────────────────
//
// Recorded INSIDE a render-pass instance the caller has already begun, which is
// what lets the primary's post-TAA overlay pass and a secondary view's own
// composite share it verbatim. A field is scene content, not a primary-view
// garnish: a CameraSensor looking at a campfire has to see the embers, and the
// same argument the density volume makes about being world-anchored and shared
// across views applies here to the shader.
//
// Per field this issues: one push constant, one (deduplicated) texture
// descriptor bind, and one vkCmdDrawIndirect whose instanceCount was written on
// the device. Nothing is sorted — additive blending commutes, so the frame
// buffer holds the same sum whatever order the fields and their particles
// arrive in, and this phase draws additive only.
void VulkanRenderer::Impl::recordFieldBillboards(VkCommandBuffer cb, VkPipeline pipe,
                                                 bool glowPass, VkPipeline pipeAlpha) {
            if (!particleFieldPass_) return;
            if (pipe == VK_NULL_HANDLE) return;
            const auto& states = particleFieldPass_->drawStates();
            if (states.empty()) return;

            // Unjittered camera: this pass runs AFTER the temporal resolve, so
            // the jitter that the G-buffer rasterized with has already been
            // removed from the image the quads composite onto.
            Matrix4 viewM, projM;
            std::memcpy(viewM.elements.data(), view().currViewUnjit_.data(), 64);
            std::memcpy(projM.elements.data(), view().currProjUnjit_.data(), 64);

            // F4: this view's record — the display transform (which moved out of
            // the push block to make room for this address) plus the fog terms
            // the quads are now attenuated by. A zero address means the block is
            // full; the shader would dereference null, so skip the whole draw.
            const VkDeviceAddress viewAddr = pushBillboardViewRecord(viewM, glowPass);
            if (viewAddr == 0) return;

            VkPipeline boundPipe = VK_NULL_HANDLE;
            const Texture* curTex = nullptr;
            bool curTexBound = false;
            std::unordered_map<const Texture*, VkDescriptorSet> setCache;

            for (const auto& st : states) {
                if (!st.billboard || st.bbIndirect == VK_NULL_HANDLE || !st.field) continue;
                // The glow leg draws ONLY the fields that asked for one. That is
                // the per-field gate F4 item 1 requires: a 300k rain field never
                // enters this pass, so weather pays nothing at all for a feature
                // it does not use.
                if (glowPass && !st.glow) continue;

                // 4c: blending is PIPELINE state, so a field that composites
                // alpha-over needs its own object. The glow leg passes no alpha
                // pipeline — that target accumulates linear HDR for a bright
                // pass and is additive by definition — so a sprite field's halo
                // is additive there, which is what a halo is.
                VkPipeline want = (st.bbAlphaOver && pipeAlpha != VK_NULL_HANDLE)
                                          ? pipeAlpha
                                          : pipe;
                if (boundPipe != want) {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                    boundPipe = want;
                }

                // The sprite texture, through the LEGACY path's cache, layout,
                // per-frame pool and 1x1 white default — reuse rather than a
                // second copy of all four. Deduplicated per texture because a
                // scene's fields usually share one (or, by default, none).
                const Texture* tex = st.field->billboardRepr().texture.get();
                if (!curTexBound || tex != curTex) {
                    VkDescriptorSet set = VK_NULL_HANDLE;
                    if (auto it = setCache.find(tex); it != setCache.end()) {
                        set = it->second;
                    } else {
                        const Image2D* texImg = ensureParticleTexture(tex);
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
                        setCache.emplace(tex, set);
                    }
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            fieldBillboardPipelineLayout_, 0, 1, &set, 0, nullptr);
                    curTex      = tex;
                    curTexBound = true;
                }

                Matrix4 modelM, mvM;
                std::memcpy(modelM.elements.data(), st.field->matrixWorld->elements.data(), 64);
                mvM.multiplyMatrices(viewM, modelM);

                FieldBillboardPC pc{};
                std::memcpy(pc.proj, projM.elements.data(), 64);
                // ROWS of the affine view*model. Matrix4 is column-major, so
                // row i is elements[i], [i+4], [i+8], [i+12] — three dot
                // products in the shader and no layout convention to agree on.
                const auto& e = mvM.elements;
                for (int r = 0; r < 3; ++r) {
                    pc.mv[r][0] = static_cast<float>(e[r]);
                    pc.mv[r][1] = static_cast<float>(e[r + 4]);
                    pc.mv[r][2] = static_cast<float>(e[r + 8]);
                    pc.mv[r][3] = static_cast<float>(e[r + 12]);
                }
                pc.paramsAddr = st.bbParamsAddr;
                pc.viewAddr   = viewAddr;
                vkCmdPushConstants(cb, fieldBillboardPipelineLayout_,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(pc), &pc);

                vkCmdDrawIndirect(cb, st.bbIndirect, 0, 1, sizeof(VkDrawIndirectCommand));
            }
        }

// ── F4: the per-VIEW billboard record ───────────────────────────────────────
//
// Two things the quads need that belong to the camera and the frame rather than
// to the field: the display transform, and the fog medium.
//
// FOG. F3 shipped billboards with none — the legacy path's overlay-fog UBO is
// bound at set 1 for particle.frag only, and the field pipeline is deliberately
// descriptor-free — so a distant ember in the fjord's murk read as crisp as a
// near one. The terms here are the SAME snapshot the legacy path takes a few
// hundred lines below; they ride a buffer_reference instead of a set, which is
// what keeps this pass out of the VUID-03047 zone entirely (F3 note 2).
//
// The plan proposed also marching the PARTICLE-DENSITY volumes here, on the
// premise that their addresses are push-constant reachable. They are not: a
// density volume is a 3D IMAGE (r16f, binding 69, hardware trilinear), and an
// image cannot be reached by device address at all — sampling one from this
// shader would mean binding the whole density descriptor array and giving up
// the property the previous paragraph is about. Deferred, deliberately: the
// acceptance test ("distant embers in fjord murk must visibly dim") is the
// height-fog/murk term, which is exactly what this does, and an ember inside a
// smoke plume is a second-order effect against that.
VkDeviceAddress VulkanRenderer::Impl::pushBillboardViewRecord(const Matrix4& viewM,
                                                              bool linearOut) {
            if (!particleFieldPass_) return 0;

            vulkan::BillboardViewGpu rec{};
            rec.exposure    = currentExposure();
            rec.toneMapMode = static_cast<uint32_t>(toneMapping_);
            rec.flags       = linearOut ? vulkan::kBbViewLinearOut : 0u;

            const bool medium = mediumActiveThisFrame_ || murkDensity_ > 0.f;
            if (medium) {
                rec.flags |= vulkan::kBbViewFogActive;
                rec.hfDensity     = mediumActiveThisFrame_ ? mediumDensityThisFrame_ : 0.f;
                rec.hfBaseY       = mediumBaseYThisFrame_;
                rec.hfFalloff     = mediumFalloffThisFrame_;
                rec.murkDensity   = murkDensity_;
                rec.waterSurfaceY = fogWaterSurfaceY_;
                // The inverse view supplies the camera height and the world-Y
                // row, which is all the fragment needs to reconstruct a
                // particle's world Y from its view-space position — the same
                // three-float trick the legacy overlay fog uses, and cheaper
                // than carrying a full world position per vertex.
                Matrix4 viewInv = viewM;
                viewInv.invert();
                const auto& iv = viewInv.elements;
                rec.camWorldY       = static_cast<float>(iv[13]);
                rec.viewToWorldY[0] = static_cast<float>(iv[1]);
                rec.viewToWorldY[1] = static_cast<float>(iv[5]);
                rec.viewToWorldY[2] = static_cast<float>(iv[9]);
            }

            // ── 4c: the sun, rotated into VIEW space ────────────────────────
            // BillboardRepr::lit evaluates one HG lobe against it, and the
            // vertex stage has the view-space particle position and no way back
            // to world — so the basis change happens here, once per view per
            // frame, out of the snapshot updateLightsUbo already took of the
            // scene's brightest DirectionalLight.
            //
            // Rotation only: a direction has no translation, so this is the
            // upper-left 3x3 of the view matrix applied to the world vector,
            // which for the column-major Matrix4 is three dot products against
            // its columns... expressed as the rows of the 3x3, i.e. elements
            // 0/4/8 for x. Unconditional: three multiplies for a field that is
            // not lit, against a branch that would have to be taken anyway.
            {
                const auto& ve = viewM.elements;
                const float* s = bbSunDirWorld_;
                rec.sunDir[0] = static_cast<float>(ve[0] * s[0] + ve[4] * s[1] + ve[8] * s[2]);
                rec.sunDir[1] = static_cast<float>(ve[1] * s[0] + ve[5] * s[1] + ve[9] * s[2]);
                rec.sunDir[2] = static_cast<float>(ve[2] * s[0] + ve[6] * s[1] + ve[10] * s[2]);
                for (int i = 0; i < 3; ++i) {
                    rec.sunRadiance[i] = bbSunRadiance_[i];
                    rec.ambient[i]     = bbAmbient_[i];
                }
            }
            return particleFieldPass_->pushViewRecord(currentFrame, rec);
        }

// ── F4: the billboard glow source + pyramid ─────────────────────────────────
//
// Recorded between recordUpscaleAndPost and recordHybridOverlay: the composite
// that consumes it is a draw INSIDE the overlay render-pass instance, and a
// compute pyramid cannot run inside one. Nothing about the billboard composite
// point moves.
void VulkanRenderer::Impl::recordFieldBillboardGlow(VkCommandBuffer cb) {

            if (!billboardGlowReadyThisFrame_ || !billboardGlow_) return;
            if (fieldBillboardGlowPipeline_ == VK_NULL_HANDLE) return;

            const VkExtent2D gext = billboardGlow_->srcExtent();
            if (gext.width == 0 || gext.height == 0) return;

            // ── Occlusion ───────────────────────────────────────────────────
            // The overlay depth prepass has already laid this frame's occluders
            // into the overlay pass's depth attachment; reduce that buffer to
            // the glow target's half extent so the second draw of the sparks is
            // depth-tested exactly like the sharp one. The predicate below is
            // the prepass's own, verbatim: when it did not run there is no
            // occluder buffer, and recordDepthReduce clears to the far plane
            // instead — the un-occluded behaviour this pass shipped with.
            const bool overlayMsaa = overlaySamples() > 1;
            const bool depthValid  = !view().secondary &&
                                    overlayDepthPrepassPipeline != VK_NULL_HANDLE &&
                                    sceneHasOverlayContent();
            VkImageView srcDepthView = VK_NULL_HANDLE;
            VkImage     srcDepthImg  = VK_NULL_HANDLE;
            VkExtent2D  srcDepthExt{};
            if (depthValid) {
                srcDepthView = overlayMsaa ? overlayMsDepth_.view
                                           : view().rasterGbufs[currentFrame].unjitDepth.view;
                srcDepthImg  = overlayMsaa ? overlayMsDepth_.image
                                           : view().rasterGbufs[currentFrame].unjitDepth.image;
                srcDepthExt  = viewOutExtent();
            }
            if (srcDepthView != VK_NULL_HANDLE) {
                // The prepass left the image in DEPTH_STENCIL_READ_ONLY_OPTIMAL
                // and made its writes visible to the depth TEST only. This is
                // the same layout with a different access: a sampled read.
                VkImageMemoryBarrier2 toSample{};
                toSample.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                toSample.srcStageMask  = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                toSample.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                toSample.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                toSample.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                toSample.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                toSample.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                toSample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toSample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toSample.image = srcDepthImg;
                toSample.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                toSample.subresourceRange.levelCount = 1;
                toSample.subresourceRange.layerCount = 1;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &toSample;
                vkCmdPipelineBarrier2(cb, &dep);
            }
            billboardGlow_->recordDepthReduce(cb, currentFrame, srcDepthView, overlayMsaa,
                                              srcDepthExt);

            VkRenderingAttachmentInfo colorAtt{};
            colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAtt.imageView   = billboardGlow_->srcView(currentFrame);
            colorAtt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            // CLEAR, not LOAD: the target accumulates one frame's sparks and
            // nothing else, so there is no history to preserve and a clear is
            // cheaper than the read-modify-write a LOAD implies.
            colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtt.clearValue.color = {{0.f, 0.f, 0.f, 0.f}};

            VkRenderingInfo ri{};
            ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.offset    = {0, 0};
            ri.renderArea.extent    = gext;
            ri.layerCount           = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments    = &colorAtt;
            // Read-only: the sparks are additive and must not occlude each
            // other, exactly as in the sharp overlay pass.
            VkRenderingAttachmentInfo glowDepthAtt{};
            glowDepthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            glowDepthAtt.imageView   = billboardGlow_->depthView(currentFrame);
            glowDepthAtt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            glowDepthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
            glowDepthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            ri.pDepthAttachment      = &glowDepthAtt;
            vkCmdBeginRendering(cb, &ri);
            VkViewport vp{0.f, 0.f, float(gext.width), float(gext.height), 0.f, 1.f};
            vkCmdSetViewport(cb, 0, 1, &vp);
            VkRect2D sc{{0, 0}, gext};
            vkCmdSetScissor(cb, 0, 1, &sc);
            recordFieldBillboards(cb, fieldBillboardGlowPipeline_, /*glowPass=*/true);
            vkCmdEndRendering(cb);

            billboardGlow_->recordPyramid(cb, currentFrame,
                                          particleFieldPass_->glowThreshold());
        }

// Lazy creation, in the PREPARE window. A scene with no glow-enabled field
// never gets here, so it allocates no offscreen target, compiles no pipeline
// and records no pass — the "weather pays nothing" half of F4 item 1's gate.
void VulkanRenderer::Impl::ensureFieldBillboardGlow() {

            billboardGlowReadyThisFrame_ = false;
            if (!particleFieldPass_ || !particleFieldPass_->glowActive()) return;
            if (fieldBillboardGlowPipeline_ == VK_NULL_HANDLE) return;// pipelines not built yet

            if (!billboardGlow_) {
                billboardGlow_ = std::make_unique<vulkan::BillboardGlowPass>(
                        *ctx, cmdPool, kFramesInFlight);
                createFieldGlowCompositePipeline();
            }
            if (fieldGlowCompositePipeline_ == VK_NULL_HANDLE) return;

            const VkExtent2D ext = ctx->swapchainExtent();
            billboardGlowReadyThisFrame_ =
                    billboardGlow_->ensureImages(ext.width, ext.height);
        }

void VulkanRenderer::Impl::recordHybridOverlay(VkCommandBuffer cb, uint32_t imageIndex) {
            const VkImage img    = ctx->swapchainImages()[imageIndex];
            const VkExtent2D ext = ctx->swapchainExtent();

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
                    // Idempotent — the depth prepass already made these this
                    // frame; this only matters if the prepass pipeline is
                    // missing. No-op at 1 sample.
                    ensureOverlayMsaaImages(ext);
                    // Hardware MSAA: rasterize the whole overlay into
                    // overlayMsColor_/overlayMsDepth_ at N samples and let
                    // vkCmdEndRendering RESOLVE onto the swapchain — the
                    // mechanism GL gets from its 4x default framebuffer.
                    // Falls back to the original direct-to-swapchain path
                    // (no inject, no scratch copy, no resolve) at 1 sample.
                    //
                    // EXACTLY the predicate the depth prepass keyed off, and
                    // the one createRasterGbufImages used to decide whether to
                    // allocate unjitDepth at all. They must not disagree: in
                    // MSAA mode unjitDepth does not exist, and at 1 sample
                    // overlayMs* do not.
                    const bool overlayMsaa = overlaySamples() > 1;

                    // Splat depth stamp: gates and pre-pass barriers now, the
                    // draw itself inside the pass below, between the overlays
                    // kSplatUnoccludedOverlayLayer exempts and everything
                    // else. True also means the depth attachment must be
                    // WRITABLE — the stamp is the pass's one depth write.
                    const bool stampActive = splatStampPrepare(cb);

                    if (overlayMsaa) {
                        // The swapchain holds the composited scene and becomes
                        // the RESOLVE TARGET, so it can neither be loaded into
                        // the MS target nor sampled by a draw inside the pass.
                        // Copy it to a 1-sample scratch first; the inject draw
                        // then seeds every sample from that.
                        VkImageMemoryBarrier2 pre[2]{};
                        // swapchain: post-TAA compute/transfer write → transfer read
                        pre[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        pre[0].srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        pre[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                               VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        pre[0].dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        pre[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                        pre[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                        pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        pre[0].image = img;
                        // scratch: discard old contents → transfer dst (WAR vs
                        // the PREVIOUS frame's inject sample)
                        pre[1] = pre[0];
                        pre[1].srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                        pre[1].srcAccessMask = 0;
                        pre[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        pre[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        pre[1].image = overlayAaScratch_.image;
                        for (auto& b : pre) {
                            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            b.subresourceRange.levelCount = 1;
                            b.subresourceRange.layerCount = 1;
                        }
                        VkDependencyInfo dPre{};
                        dPre.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        dPre.imageMemoryBarrierCount = 2;
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

                        VkImageMemoryBarrier2 post[3]{};
                        // scratch: transfer write → inject's fragment sample
                        post[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        post[0].srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        post[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        post[0].dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                        post[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                        post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        post[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        post[0].image = overlayAaScratch_.image;
                        // swapchain: transfer read → the pass's resolve target
                        post[1] = post[0];
                        post[1].srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        post[1].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                        post[1].dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        post[1].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        post[1].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        post[1].image = img;
                        // MS color: discard (loadOp DONT_CARE, the inject
                        // covers every pixel) — WAR vs the previous frame's
                        // resolve read.
                        post[2] = post[1];
                        post[2].srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        post[2].srcAccessMask = 0;
                        post[2].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        post[2].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        post[2].image = overlayMsColor_.image;
                        for (auto& b : post) {
                            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            b.subresourceRange.levelCount = 1;
                            b.subresourceRange.layerCount = 1;
                        }
                        VkDependencyInfo dPost{};
                        dPost.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        dPost.imageMemoryBarrierCount = 3;
                        dPost.pImageMemoryBarriers = post;
                        vkCmdPipelineBarrier2(cb, &dPost);
                    } else {
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
                    }

                    VkRenderingAttachmentInfo colorAtt{};
                    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    if (overlayMsaa) {
                        colorAtt.imageView          = overlayMsColor_.view;
                        colorAtt.imageLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        // DONT_CARE both ways: the inject draw writes every
                        // pixel, and only the resolve output is kept.
                        colorAtt.loadOp             = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                        colorAtt.storeOp            = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                        colorAtt.resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
                        colorAtt.resolveImageView   = ctx->swapchainImageViews()[imageIndex];
                        colorAtt.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    } else {
                        colorAtt.imageView   = ctx->swapchainImageViews()[imageIndex];
                        colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                        colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                    }

                    // Depth from the overlay depth prepass, LOAD_OP_LOAD both
                    // ways. Read-only when nothing writes it (the overlay
                    // draws never do — STORE_OP_NONE says so). When the splat
                    // stamp draws inside this pass, splatStampPrepare already
                    // transitioned the image to DEPTH_ATTACHMENT_OPTIMAL and
                    // the stamp's write must be kept (STORE) — the next
                    // consumer is the next frame's prepass, which clears from
                    // UNDEFINED, so the writable layout is never handed back.
                    VkRenderingAttachmentInfo depthAtt{};
                    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depthAtt.imageView   = overlayMsaa ? overlayMsDepth_.view
                                                       : view().rasterGbufs[currentFrame].unjitDepth.view;
                    depthAtt.imageLayout = stampActive ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                                       : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                    depthAtt.storeOp     = stampActive ? VK_ATTACHMENT_STORE_OP_STORE
                                                       : VK_ATTACHMENT_STORE_OP_NONE;

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea.offset = {0, 0};
                    ri.renderArea.extent = ext;
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &colorAtt;
                    ri.pDepthAttachment = &depthAtt;
                    vkCmdBeginRendering(cb, &ri);

                    if (overlayMsaa) {
                        // Scene inject — FIRST draw, FULL extent (not the
                        // split-screen region): every sample of every pixel
                        // must carry the composited scene so the blends below
                        // composite against the true background and pixels no
                        // overlay touches resolve back bit-identical. Outside
                        // the region this writes the swapchain's own content
                        // straight back, so split-screen needs no special case.
                        VkViewport vpFull{0.f, 0.f, float(ext.width), float(ext.height), 0.f, 1.f};
                        vkCmdSetViewport(cb, 0, 1, &vpFull);
                        VkRect2D scFull{{0, 0}, ext};
                        vkCmdSetScissor(cb, 0, 1, &scFull);
                        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayInjectPipeline_);
                        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                overlayInjectPipelineLayout_, 0, 1,
                                                &overlayInjectSet_, 0, nullptr);
                        vkCmdDraw(cb, 3, 1, 0, 0);
                    }

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
                    std::memcpy(vpUnjitMat.elements.data(), view().currVPunjit_.data(), 64);

                    // Track currently-bound pipeline so we don't redundantly
                    // re-bind on every draw when the scene's overlay objects
                    // share a mode (most do).
                    VkPipeline curPipeline = VK_NULL_HANDLE;

                    // Blend-pipeline slot from the material, mirroring
                    // GLState::setBlending's gate: no blending only for
                    // (Normal && !transparent) or Blending::None. Additive
                    // gets its own factors; the remaining enums (Subtractive
                    // / Multiply / Custom) have no overlay variant and fall
                    // back to plain alpha when transparent, opaque otherwise.
                    // Returns 0 = opaque, 1 = alpha, 2 = additive.
                    auto overlayBlendMode = [](const Material& m) {
                        if (m.blending == Blending::Additive) return 2;
                        return (m.transparent && m.blending != Blending::None) ? 1 : 0;
                    };

                    auto drawOverlayMesh = [&](const MeshEntry& en) {
                        Color color(1.f, 1.f, 1.f);
                        float opacity = 1.0f;
                        bool wireframe = false;
                        bool depthTest = true;
                        bool vertexColors = false;
                        int blendMode = 0;
                        Side side = Side::Front;
                        if (auto* m = en.mesh->material().get()) {
                            if (auto* mc = dynamic_cast<MaterialWithColor*>(m)) {
                                color = mc->color;
                            }
                            if (auto* mw = dynamic_cast<MaterialWithWireframe*>(m)) {
                                wireframe = mw->wireframe;
                            }
                            opacity      = m->opacity;
                            blendMode    = overlayBlendMode(*m);
                            depthTest    = m->depthTest;
                            side         = m->side;
                            vertexColors = m->vertexColors;
                        }
                        // Per-vertex colours: fetch pos+color via the line-
                        // geometry cache — the BLAS vertex buffer bound below
                        // is position-only. Skinned/displaced/morphed meshes
                        // keep the flat path (the cache holds undeformed
                        // attribute-array positions), as does any geometry
                        // without a vec3 "color" attribute.
                        const vulkan::LineRec* crec = nullptr;
                        if (!wireframe && vertexColors &&
                            !en.isSkinned && !en.isDisplaced && !en.isMorphed) {
                            if (auto geom = en.mesh->geometry()) {
                                if (const auto* colAttr = geom->getAttribute("color");
                                    colAttr && colAttr->itemSize() == 3) {
                                    crec = ensureLineGeometryUploaded(geom.get());
                                    if (crec && (crec->vertex.handle == VK_NULL_HANDLE ||
                                                 crec->color.handle == VK_NULL_HANDLE))
                                        crec = nullptr;
                                }
                            }
                        }
                        // Wireframe takes precedence — wireframe lines are
                        // typically opaque even when material.transparent
                        // is incidentally true.
                        VkPipeline want;
                        if (wireframe)           want = overlayWireframePipeline;
                        else if (crec)           want = overlayMeshColoredPipelines[blendMode];
                        else if (blendMode == 2) want = overlayBasicAdditivePipeline;
                        else if (blendMode == 1) want = overlayBasicTransparentPipeline;
                        else                     want = overlayBasicPipeline;
                        if (want != curPipeline) {
                            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                            curPipeline = want;
                        }
                        // GL parity: material.depthTest == false means "draw
                        // through everything" (TransformControls gizmos and
                        // helpers rely on it). Dynamic state, so it must be set
                        // for every draw in this pass, not just when it changes.
                        vkCmdSetDepthTestEnable(cb, depthTest ? VK_TRUE : VK_FALSE);
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

                        const BlasRecord* rec = nullptr;
                        if (!crec) {
                            rec = resolveBlasForEntry(en);
                            if (!rec || rec->vertex.handle == VK_NULL_HANDLE) return;
                        }

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

                        if (crec) {
                            // overlay_color pipeline: pos at binding 0, vec3
                            // colour at binding 1, always-uint32 index. The
                            // fragment result is pc.color × vertex colour with
                            // pc.color.w opacity — three.js modulation.
                            VkBuffer     cvbufs[2] = {crec->vertex.handle, crec->color.handle};
                            VkDeviceSize cvoffs[2] = {0, 0};
                            vkCmdBindVertexBuffers(cb, 0, 2, cvbufs, cvoffs);
                            if (crec->index.handle != VK_NULL_HANDLE) {
                                vkCmdBindIndexBuffer(cb, crec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                                vkCmdDrawIndexed(cb, crec->indexCount, 1, 0, 0, 0);
                            } else {
                                vkCmdDraw(cb, crec->vertexCount, 1, 0, 0);
                            }
                            return;
                        }

                        VkBuffer     vbufs[1] = {rec->vertex.handle};
                        VkDeviceSize voffs[1] = {0};
                        vkCmdBindVertexBuffers(cb, 0, 1, vbufs, voffs);
                        if (rec->index.handle != VK_NULL_HANDLE) {
                            // Packed static records store uint16 indices (bit 3).
                            vkCmdBindIndexBuffer(cb, rec->index.handle, 0,
                                                 (rec->packedMask & 8u) ? VK_INDEX_TYPE_UINT16
                                                                        : VK_INDEX_TYPE_UINT32);
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
                    // carries PointsMaterial::size instead of opacity, and
                    // params.x the scale that turns a world-space size into
                    // pixels (0 when the size is already pixels). See
                    // overlay_point.vert for the two meanings of `size`.
                    //
                    // Perspective-ness comes off the view-projection rather
                    // than a camera pointer, which this path does not hold:
                    // an ortho VP has w row exactly (0,0,0,1), so non-zero
                    // xyz there is the answer GL's isPerspectiveMatrix gives.
                    const bool vpPerspective = vpUnjitMat.elements[3] != 0.f ||
                                               vpUnjitMat.elements[7] != 0.f ||
                                               vpUnjitMat.elements[11] != 0.f;
                    const float pointAttenScale =
                            vpPerspective ? 0.5f * float(regionSwapExt_.height) : 0.f;

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
                        float pcAtten       = 0.f; // point attenuation scale, 0 = size is px
                        bool useVertexColors = false;
                        if (matPtr) {
                            if (auto* mc = dynamic_cast<MaterialWithColor*>(matPtr.get())) {
                                color = mc->color;
                            }
                            if (le.isPoints) {
                                if (auto* ms = dynamic_cast<MaterialWithSize*>(matPtr.get())) {
                                    // No floor: an attenuated size is in METRES,
                                    // and clamping it to 1 collapsed every
                                    // world-space cloud to 1px specks. The
                                    // shader clamps the resolved PIXEL size.
                                    pcW = ms->size;
                                    if (ms->sizeAttenuation) pcAtten = pointAttenScale;
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
                        } else if (const int blendMode = matPtr ? overlayBlendMode(*matPtr) : 0;
                                   blendMode > 0) {
                            // transparent / Blending::Additive lines take the
                            // blend-enabled variants — the plain pipelines
                            // are blend-off, which rendered a dark additive
                            // streamline as an opaque near-black stroke.
                            want = overlayLineBlendPipelines[useVertexColors ? 1 : 0]
                                                            [le.isSegments ? 0 : 1]
                                                            [blendMode - 1];
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
                        vkCmdSetDepthTestEnable(cb, (matPtr && !matPtr->depthTest) ? VK_FALSE
                                                                                   : VK_TRUE);

                        Matrix4 model;
                        std::memcpy(model.elements.data(), le.worldMatrix.data(), 64);
                        Matrix4 mvpL;
                        mvpL.multiplyMatrices(vpUnjitMat, model);

                        struct OverlayPC {
                            float mvp[16];
                            float color[4];
                            float params[4];// .x = point attenuation scale
                        } pcL{};
                        std::memcpy(pcL.mvp, mvpL.elements.data(), 64);
                        pcL.color[0]  = color.r;
                        pcL.color[1]  = color.g;
                        pcL.color[2]  = color.b;
                        pcL.color[3]  = pcW;
                        pcL.params[0] = pcAtten;
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
                        bool  preStamp;// kSplatUnoccludedOverlayLayer: draws before the splat stamp
                        int   renderOrder;
                        float viewZ;// camera-space z (negative in front; smaller = farther)
                    };
                    std::vector<OverlayItem> overlayItems;
                    overlayItems.reserve(lastVisibleLines_.size() + 16);
                    Matrix4 viewUnjitM;
                    std::memcpy(viewUnjitM.elements.data(), view().currViewUnjit_.data(), 64);
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
                        overlayItems.push_back({false, i, tr,
                                                en.mesh->layers.isEnabled(kSplatUnoccludedOverlayLayer),
                                                en.mesh->renderOrder,
                                                viewZOf(en.worldMatrix)});
                    }
                    for (size_t i = 0; i < lastVisibleLines_.size(); ++i) {
                        const auto& le = lastVisibleLines_[i];
                        const Object3D* obj = le.isPoints ? static_cast<const Object3D*>(le.points)
                                                          : static_cast<const Object3D*>(le.line);
                        const Material* m = le.isPoints
                                                    ? (le.points ? le.points->material().get() : nullptr)
                                                    : (le.line ? le.line->material().get() : nullptr);
                        overlayItems.push_back({true, i, m && m->transparent,
                                                obj && obj->layers.isEnabled(kSplatUnoccludedOverlayLayer),
                                                obj ? obj->renderOrder : 0,
                                                viewZOf(le.worldMatrix)});
                    }
                    // renderOrder is the primary key inside each group, exactly
                    // as GLRenderer's painter sorts do — TransformControls pins
                    // its gizmo to renderOrder=INT_MAX so it composites over
                    // every other overlay.
                    std::stable_sort(overlayItems.begin(), overlayItems.end(),
                                     [](const OverlayItem& a, const OverlayItem& b) {
                                         if (a.transparent != b.transparent) return !a.transparent;
                                         if (a.renderOrder != b.renderOrder) return a.renderOrder < b.renderOrder;
                                         if (!a.transparent) return false;// opaque: keep traversal order
                                         return a.viewZ < b.viewZ;         // transparent: back-to-front
                                     });
                    // Exempt overlays first — a picture OF a splat surface must
                    // not be occluded BY the splat surface (see
                    // kSplatUnoccludedOverlayLayer) — then the stamp, whose
                    // depth write occludes everything after it that a cloud
                    // stands in front of. Drawn first even when no stamp runs
                    // this frame, so flagging an overlay never changes its
                    // composite position between cloudy and cloudless frames.
                    const auto drawItem = [&](const OverlayItem& it) {
                        if (it.isLine) drawOverlayLine(lastVisibleLines_[it.idx]);
                        else           drawOverlayMesh(lastVisibleEntries_[it.idx]);
                    };
                    for (const auto& it : overlayItems) {
                        if (it.preStamp) drawItem(it);
                    }
                    if (stampActive) {
                        recordSplatStampDraw(cb);
                        // The stamp bound its own pipeline; the cache must not
                        // skip the next draw's re-bind.
                        curPipeline = VK_NULL_HANDLE;
                    }
                    for (const auto& it : overlayItems) {
                        if (!it.preStamp) drawItem(it);
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
                    // (THIS FRAME-IN-FLIGHT's texture-descriptor pool is reset at
                    // the head of recordCommandBuffer — see the comment there.
                    // It used to be reset here, which covered the three
                    // consumers that existed in F3 but not F4's glow pass, which
                    // records earlier in the frame.)

                    if (particlePipelineNormal_ != VK_NULL_HANDLE) {
                        bool anyParticle = false;
                        for (const auto& en : lastVisibleEntries_) {
                            if (en.isParticle && en.mesh) { anyParticle = true; break; }
                        }
                        const bool anySprite = !lastVisibleSprites_.empty();
                        if (anyParticle || anySprite) {
                            // (The shared per-frame descriptor pool is reset
                            // above, for every consumer of it in this pass.)
                            // Unjittered camera for both billboard kinds
                            // (particles + world sprites).
                            Matrix4 viewM, projM;
                            std::memcpy(viewM.elements.data(), view().currViewUnjit_.data(), 64);
                            std::memcpy(projM.elements.data(), view().currProjUnjit_.data(), 64);

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
                                // Lit particles: hand the frag the display
                                // transform (exposure + tone-map mode) so the
                                // HDR-lit path lands in the same domain as the
                                // tonemapped background it blends over.
                                of.litActive   = particleLightCount_ > 0 ? 1.f : 0.f;
                                of.exposure    = currentExposure();
                                of.toneMapMode = static_cast<float>(toneMapping_);
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
                                // firstInstance carries this system's base into
                                // particle_light.comp's results (gl_InstanceIndex
                                // in particle.vert; instanceCount stays 1).
                                // Additive systems are emissive by design and
                                // draws the lighting pass didn't cover fall
                                // back — both via the kUnlitBase sentinel.
                                uint32_t litBase = kUnlitBase;
                                if (!additive && particleLightCount_ > 0) {
                                    if (auto bit = particleLitBase_.find(en.mesh);
                                        bit != particleLitBase_.end()) litBase = bit->second;
                                }
                                vkCmdDrawIndexed(cb, prec->indexCount, 1, 0, 0, litBase);
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

                    // ── ParticleField billboards ───────────────────────────
                    // LAST in the pass, and outside the legacy block above so a
                    // scene whose only overlay content is a field of embers
                    // still gets them (sceneHasOverlayContent() knows about
                    // fields for exactly that reason). They composite here — on
                    // the swapchain, after TAA/DLSS/FSR — because a 3-px spark
                    // crossing 20 px in a frame is precisely the content a
                    // temporal filter mis-resolves, and because this is where
                    // transparents already land.
                    recordFieldBillboards(cb,
                                          overlayMsaa ? fieldBillboardPipeline_
                                                      : fieldBillboardPipeline1x_,
                                          /*glowPass=*/false,
                                          overlayMsaa ? fieldBillboardAlphaPipeline_
                                                      : fieldBillboardAlphaPipeline1x_);

                    // ── F4: the billboard glow ─────────────────────────────
                    // The sparks' own bloom pyramid, computed just before this
                    // pass began (recordFieldBillboardGlow) and folded in here
                    // as one fullscreen additive draw. IN THIS PASS on purpose:
                    // the composite point for field billboards does not move,
                    // so they stay outside TAA/DLSS/FSR (F3 note 1) and gain a
                    // glow anyway. Additive, so its position relative to the
                    // sharp quads above cannot change the sum.
                    if (billboardGlowReadyThisFrame_ && billboardGlow_ &&
                        fieldGlowCompositePipeline_ != VK_NULL_HANDLE) {
                        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                          fieldGlowCompositePipeline_);
                        const VkDescriptorSet gset = billboardGlow_->compositeSet(currentFrame);
                        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                fieldGlowCompositeLayout_, 0, 1, &gset, 0, nullptr);
                        FieldGlowPC gpc{};
                        // Divide by the level count for the same reason
                        // PostComposite does: the upsample walk-back sums every
                        // level into level 0, so without it a deeper chain (a
                        // bigger window) would read brighter for the same knob.
                        gpc.intensity =
                                1.f / static_cast<float>(std::max(billboardGlow_->levels(), 1u));
                        gpc.exposure    = currentExposure();
                        gpc.toneMapMode = static_cast<uint32_t>(toneMapping_);
                        gpc.invDisplay[0] = ext.width  ? 1.f / float(ext.width)  : 0.f;
                        gpc.invDisplay[1] = ext.height ? 1.f / float(ext.height) : 0.f;
                        vkCmdPushConstants(cb, fieldGlowCompositeLayout_,
                                           VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(gpc), &gpc);
                        vkCmdDraw(cb, 3, 1, 0, 0);
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
}

void VulkanRenderer::Impl::recordCommandBuffer(VkCommandBuffer cb, uint32_t imageIndex) {

            // The frame, stage by stage. ORDER IS THE CONTRACT — every
            // cross-stage barrier lives inside the stage that needs it, so
            // reordering these calls reorders the frame's synchronization.
            // Each stage carries its own full commentary; this function is
            // the table of contents.
            updatePaneRegion();

            // ── The shared billboard texture pool, reset ONCE for the frame ──
            // F3 note 4 is the cautionary tale and this is its second act. That
            // pool is allocated from by FOUR consumers now — the legacy
            // ParticleSystem billboards, world sprites, ParticleField billboards
            // in the overlay, and (F4) the same fields again in the glow pass —
            // and the failure mode of running it dry is a silent `continue` on
            // vkAllocateDescriptorSets that looks exactly like the feature not
            // being implemented. F3 moved the reset to the head of the overlay
            // pass, which covered the three consumers that existed then; the
            // glow pass records BEFORE that point and quietly stopped drawing
            // after 32 frames, which is why its first captures showed a maxD of
            // 7/255 — inside the backend's own noise.
            //
            // Reset here instead: the head of the frame's recording, before any
            // consumer and after this slot's fence, which is the only place
            // that is provably correct for all of them (including the secondary
            // views, which record after the primary and would have been left
            // unreset on the debug-blit and events-only early-outs). One host
            // call per frame; every consumer sees the state it saw before.
            if (particleDescPools_[currentFrame] != VK_NULL_HANDLE)
                vkResetDescriptorPool(ctx->device(), particleDescPools_[currentFrame], 0);

            // Deformers (skinned / tet / displaced / grass) + the per-frame
            // TLAS refit, all recorded into the frame cb.
            recordDeformAndTlas(cb);

            // Raster G-buffer prepass (+ occlusion culling, MSAA resolve,
            // overlay depth prepass). True → the hybrid debug view blitted
            // straight to the swapchain and the frame is finished.
            if (recordGbufferStage(cb, imageIndex)) return;

            // Events-only mode (~500 Hz event camera): clear the swapchain,
            // skip shade/post entirely. True → the frame is finished.
            if (recordEventsOnlyFrame(cb, imageIndex)) return;

            // Swapchain → GENERAL, ReSTIR reservoir visibility, optional
            // split-screen clear.
            recordSwapchainPrepare(cb, imageIndex);

            // Per-frame set index. Was `currentFrame * imageCount_ + imageIndex`,
            // which read as live per-(frame, image) indexing but was neither: the
            // deferred leaf ignores the argument entirely and drives DeferredShade
            // from currentFrame, and the other recordSceneDispatch call site
            // (VulkanCoreFrame's secondary-view path) already passes currentFrame
            // straight through. Composing an index nothing consumed was the only
            // thing making imageCount_ look like frame-loop state that a swapchain
            // recreate had to keep current — see its declaration.
            const uint32_t setIdx = currentFrame;

            // Extents + exposure are shared by both render modes and by the
            // bloom/TAA tail below, so hoist them out of the mode branch.
            const VkExtent2D ext   = viewOutExtent();
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

            // Gather this frame's particle centers + base indices BEFORE the
            // scene dispatch — the hook tail runs particle_light.comp over them
            // (lit billboards; the overlay loop below consumes the bases).
            prepareParticleLighting();

            // Dispatch the deferred shade compute (VulkanRenderer::Impl);
            // bloom + TAA below are shared.
            recordSceneDispatch(cb, setIdx, ext, ptExt, exposureBits);

            // Gaussian splats: composite SplatClouds into sceneHdr while it
            // is still linear HDR, so everything below acts on them too.
            recordSplats(cb);

            // Thin-lens depth of field (opt-in): defocus the linear-HDR
            // scene before bloom/composite/TAA.
            recordDepthOfField(cb);

            // Bloom + tonemap + temporal resolve — exactly one of DLSS /
            // FSR / built-in TAA runs (see recordUpscaleAndPost).
            recordUpscaleAndPost(cb, imageIndex, ext, ptExt, exposureBits, preExp);

            // F4: the billboard-only bloom pyramid. AFTER the upscaler (so the
            // composite point that keeps field billboards clear of TAA/DLSS/FSR
            // is untouched) and BEFORE the overlay pass (because the fullscreen
            // draw that consumes the pyramid lives inside it, and a compute
            // chain cannot be recorded inside a render-pass instance). No-op
            // unless a visible field asked for a glow.
            recordFieldBillboardGlow(cb);

            // Post-TAA wireframe / line / particle / sprite overlays. The
            // splat depth stamp records INSIDE this pass now (splatStampPrepare
            // / recordSplatStampDraw), between the overlays that
            // kSplatUnoccludedOverlayLayer exempts and everything else.
            recordHybridOverlay(cb, imageIndex);

            // ── Camera image formation: lens distortion + sensor noise ─────────
            // Deliberately LAST. The overlay pass above composites particle
            // billboards (chimney smoke and friends), lines and wireframe
            // straight onto the swapchain, so a warp applied any earlier bent
            // the scene but not them — overlays visibly slid off the geometry
            // they belong to, worsening toward the frame edge. A real lens
            // bends everything in front of it, and a real sensor noises
            // everything the lens projects, so both belong here. Doing it here
            // also leaves the overlay's depth test in the undistorted space its
            // depth buffer is actually in. ImGui draws after (endFrame) and
            // stays clean, which is right — a HUD is not in front of the lens.
            //
            // No-op (and no allocation) unless a lens or noise is configured.
            if (sensorPass_ && sensorStageActive()) {
                gpuTimings_->begin(cb, TP_SensorImage, currentFrame);
                sensorPass_->record(cb, currentFrame,
                                    ctx->swapchainImages()[imageIndex],
                                    ctx->swapchainImageViews()[imageIndex],
                                    ext.width, ext.height, buildSensorParams());
                gpuTimings_->end(cb, TP_SensorImage, currentFrame);
            }

            // ── End of deferred-render recording. ──────────────────────────────
            // The swapchain image is left in VK_IMAGE_LAYOUT_GENERAL — endFrame
            // handles the ImGui overlay pass + GENERAL → PRESENT_SRC transition,
            // closes the command buffer, and submits.
        }

void VulkanRenderer::Impl::recordSceneCapture(VkCommandBuffer cb, uint32_t imageIndex) {
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

void VulkanRenderer::Impl::createEventShadePipeline() {
            if (eventShadePipeline_ != VK_NULL_HANDLE) return;

            // 6 bindings: gbufNormal, gbufIds (combined image samplers),
            // matDesc (storage buffer), lightsUbo (uniform), lumaBuf (storage),
            // finalImg (storage image — the acquired swapchain image, read by
            // the Final source; bound every frame, unused under Shaded).
            std::array<VkDescriptorSetLayoutBinding, 6> b{};
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
            b[5].binding         = 5;
            b[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[5].descriptorCount = 1;
            b[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(b.size());
            dlci.pBindings    = b.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &eventShadeDsLayout_),
                  "vkCreateDescriptorSetLayout(event_shade)");

            struct ShadePC {
                uint32_t width;
                uint32_t height;
                uint32_t srcWidth;
                uint32_t srcHeight;
                uint32_t source;
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
            std::array<VkDescriptorPoolSize, 4> ps{};
            ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[0].descriptorCount = 2 * kFramesInFlight;
            ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps[1].descriptorCount = 2 * kFramesInFlight;
            ps[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps[2].descriptorCount = 1 * kFramesInFlight;
            ps[3].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ps[3].descriptorCount = 1 * kFramesInFlight;
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

void VulkanRenderer::Impl::allocateEventLumaBuffer(uint32_t w, uint32_t h) {
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

void VulkanRenderer::Impl::recordEventShade(VkCommandBuffer cb, uint32_t frame, uint32_t imageIndex) {
            if (eventShadePipeline_ == VK_NULL_HANDLE ||
                eventLumaBuf_.handle == VK_NULL_HANDLE) return;

            const EventCameraSource source = effectiveEventCamSource();
            const VkImage     swapImg  = ctx->swapchainImages()[imageIndex];
            const VkImageView swapView = ctx->swapchainImageViews()[imageIndex];

            // Per-frame descriptor writes. Cheap; no descriptor indexing
            // shenanigans needed.
            VkDescriptorImageInfo normalInfo{};
            normalInfo.sampler     = gbufSampler_;
            normalInfo.imageView   = view().rasterGbufs[frame].normal.view;
            normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo idsInfo{};
            idsInfo.sampler     = gbufSampler_;
            idsInfo.imageView   = view().rasterGbufs[frame].ids.view;
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

            // The Final source's input: the acquired swapchain image, which
            // holds the presented frame by the time the record tail runs
            // (TAA/upscale/post-composite have written it) and sits in
            // GENERAL. Bound every frame regardless of source — the shader
            // statically references it, so the descriptor must be valid.
            VkDescriptorImageInfo finalInfo{};
            finalInfo.imageView   = swapView;
            finalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            // The inFlight[frame] fence was waited on at the top of
            // beginDeferredFrame, so this frame's prior use of its own
            // descriptor set has retired — updating it here can't race the
            // other in-flight frame's set.
            VkDescriptorSet ds = eventShadeDescSets_[frame];
            std::array<VkWriteDescriptorSet, 6> w{};
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
            w[5].dstSet = ds;
            w[5].dstBinding = 5;
            w[5].descriptorCount = 1;
            w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[5].pImageInfo = &finalInfo;

            vkUpdateDescriptorSets(ctx->device(),
                                    static_cast<uint32_t>(w.size()), w.data(),
                                    0, nullptr);

            if (source == EventCameraSource::Final) {
                // The swapchain was last written by the resolve/post chain
                // (compute storage writes; transfer covers the upscaler and
                // eventsOnly's clear defensively). Make those writes visible
                // to this dispatch's storage-image loads. GENERAL → GENERAL:
                // the downstream view-composite / overlay / present barriers
                // all source from COMPUTE, which orders our read before their
                // writes (WAR needs only the execution dependency).
                VkImageMemoryBarrier toRead{};
                toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                toRead.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
                toRead.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
                toRead.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT |
                                             VK_ACCESS_TRANSFER_WRITE_BIT;
                toRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
                toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toRead.image               = swapImg;
                toRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                toRead.subresourceRange.levelCount = 1;
                toRead.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(cb,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &toRead);
            }

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, eventShadePipeline_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     eventShadePipelineLayout_, 0, 1, &ds, 0, nullptr);

            struct ShadePC {
                uint32_t width;       // sensor (output) dims
                uint32_t height;
                uint32_t srcWidth;    // source dims: the gbuf (render extent,
                uint32_t srcHeight;   // ≤ swapchain) for Shaded, the swapchain
                                      // (display extent) for Final
                uint32_t source;      // 0 = Shaded proxy, 1 = Final frame
            } pc{};
            pc.width  = eventLumaW_;
            pc.height = eventLumaH_;
            if (source == EventCameraSource::Final) {
                const VkExtent2D sext = ctx->swapchainExtent();
                pc.srcWidth  = sext.width;
                pc.srcHeight = sext.height;
                pc.source    = 1u;
            } else {
                const VkExtent2D rext = renderExtent();
                pc.srcWidth  = rext.width;
                pc.srcHeight = rext.height;
                pc.source    = 0u;
            }
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

void VulkanRenderer::Impl::recordOverlayAndPresentTransition(VkCommandBuffer cb, uint32_t imageIndex) {
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


void VulkanRenderer::Impl::ensureOverlayMsaaImages(VkExtent2D ext) {
            if (overlaySamples() <= 1) return;
            if (overlayMsColor_.image != VK_NULL_HANDLE &&
                overlayMsColor_.width == ext.width && overlayMsColor_.height == ext.height) return;
            if (overlayMsColor_.image != VK_NULL_HANDLE) retire(std::move(overlayMsColor_));
            if (overlayMsDepth_.image != VK_NULL_HANDLE) retire(std::move(overlayMsDepth_));
            if (overlayAaScratch_.image != VK_NULL_HANDLE) retire(std::move(overlayAaScratch_));

            overlayMsColor_ = createAttachmentImage2D(
                    ext.width, ext.height, ctx->swapchainFormat(),
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                    "overlayMsColor", overlaySampleBits_);
            // SAMPLED as well as attached: BillboardGlowPass reduces this
            // buffer to its own half extent so the billboard glow source can
            // depth-test (billboard_glow_depth.frag, sampler2DMS variant).
            overlayMsDepth_ = createAttachmentImage2D(
                    ext.width, ext.height, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    "overlayMsDepth", overlaySampleBits_);

            // Scratch needs a sampler (the inject shader reads it), which
            // createAttachmentImage2D doesn't make.
            {
                Image2D out{};
                out.width  = ext.width;
                out.height = ext.height;
                out.format = ctx->swapchainFormat();
                VkImageCreateInfo ici{};
                ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType     = VK_IMAGE_TYPE_2D;
                ici.format        = out.format;
                ici.extent        = {ext.width, ext.height, 1};
                ici.mipLevels     = 1;
                ici.arrayLayers   = 1;
                ici.samples       = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
                ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO;
                check(vmaCreateImage(ctx->allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
                      "vmaCreateImage(overlayAaScratch)");
                VkImageViewCreateInfo vci{};
                vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image    = out.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format   = out.format;
                vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                vci.subresourceRange.levelCount = 1;
                vci.subresourceRange.layerCount = 1;
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &out.view),
                      "vkCreateImageView(overlayAaScratch)");
                VkSamplerCreateInfo sci{};
                sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                sci.magFilter    = VK_FILTER_NEAREST;// inject texelFetches 1:1
                sci.minFilter    = VK_FILTER_NEAREST;
                sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                check(vkCreateSampler(ctx->device(), &sci, nullptr, &out.sampler),
                      "vkCreateSampler(overlayAaScratch)");
                ctx->setObjectName(out.image, "overlayAaScratch");
                overlayAaScratch_ = out;
            }

            if (overlayInjectSet_ != VK_NULL_HANDLE) {
                VkDescriptorImageInfo ii{overlayAaScratch_.sampler, overlayAaScratch_.view,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet w{};
                w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet          = overlayInjectSet_;
                w.dstBinding      = 0;
                w.descriptorCount = 1;
                w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo      = &ii;
                vkUpdateDescriptorSets(ctx->device(), 1, &w, 0, nullptr);
            }
        }

void VulkanRenderer::Impl::prepareParticleLighting() {
            particleLightCount_ = 0;
            particleLitBase_.clear();
            ensureParticleIoSets();
            if (particlePipelineNormal_ == VK_NULL_HANDLE ||
                particleIoDescSets_[currentFrame] == VK_NULL_HANDLE ||
                particleCenterBufs_[currentFrame].handle == VK_NULL_HANDLE) return;

            particleCenterScratch_.clear();
            uint32_t base = 0;
            for (const auto& en : lastVisibleEntries_) {
                if (!en.isParticle || !en.mesh) continue;
                auto geomSp = en.mesh->geometry();
                BufferGeometry* geom = geomSp.get();
                if (!geom) continue;
                auto* posAttr = geom->getAttribute<float>("position");
                if (!posAttr) continue;
                const uint32_t vtx = static_cast<uint32_t>(posAttr->count());
                const uint32_t particles = vtx / 4u;// 4 coincident verts per particle
                if (particles == 0) continue;
                if (base + particles > kMaxLitParticles) continue;// draw falls back unlit

                const auto& w = en.worldMatrix;// column-major world transform
                const auto& p = posAttr->array();
                particleCenterScratch_.resize(static_cast<size_t>(base + particles) * 4);
                float* dst = particleCenterScratch_.data() + static_cast<size_t>(base) * 4;
                for (uint32_t i = 0; i < particles; ++i) {
                    const float x = p[i * 12u + 0u];// vertex 4i of particle i
                    const float y = p[i * 12u + 1u];
                    const float z = p[i * 12u + 2u];
                    dst[i * 4u + 0u] = w[0] * x + w[4] * y + w[8] * z + w[12];
                    dst[i * 4u + 1u] = w[1] * x + w[5] * y + w[9] * z + w[13];
                    dst[i * 4u + 2u] = w[2] * x + w[6] * y + w[10] * z + w[14];
                    dst[i * 4u + 3u] = 0.f;
                }
                particleLitBase_[en.mesh] = base;
                base += particles;
            }
            if (base == 0) return;
            uploadHostVisible(ctx->allocator(), particleCenterBufs_[currentFrame],
                              particleCenterScratch_.data(),
                              particleCenterScratch_.size() * sizeof(float));
            particleLightCount_ = base;
        }

vulkan::SensorPass::Params VulkanRenderer::Impl::buildSensorParams() {
            vulkan::SensorPass::Params p{};
            p.distortActive = lens_.active();
            p.noiseActive   = sensorNoise_.enabled;

            if (p.distortActive) {
                p.lensModel     = static_cast<uint32_t>(lens_.model);
                p.radial[0]     = lens_.k1;
                p.radial[1]     = lens_.k2;
                p.radial[2]     = lens_.k3;
                p.radial[3]     = lens_.k4;
                p.tangential[0] = lens_.p1;
                p.tangential[1] = lens_.p2;
                // (fx/W, fy/H) = 0.5·proj[0], 0.5·proj[5]; (cx/W, cy/H) =
                // 0.5·(1∓skew). Same derivation as cameraIntrinsics(), with
                // the extent divided out. projP0_/projP5_ are the OUTPUT
                // intrinsics — updateCameraUbo stashes them before applying
                // overscan, so the lens is described by the camera the user
                // configured, not by the widened one we rendered.
                p.normK[0] = 0.5f * projP0_;
                p.normK[1] = 0.5f * projP5_;
                p.normK[2] = 0.5f * (1.f - projP8_);
                p.normK[3] = 0.5f * (1.f + projP9_);
                p.overscan = effectiveOverscan();
            }

            if (p.noiseActive) {
                p.frameSeed      = sensorNoise_.seed * 2654435761u + sensorNoiseFrame_;
                p.fullWell       = sensorNoise_.fullWellElectrons;
                p.readNoise      = sensorNoise_.readNoiseElectrons;
                // Dark current is a rate; the exposure time turns it into a
                // count. A long exposure is noisier for the same reason a real
                // long exposure is.
                p.darkElectrons  = sensorNoise_.darkCurrentElectronsPerSec * camShutter_;
                p.prnu           = sensorNoise_.prnuPercent * 0.01f;
                p.isoGain        = camIso_ / 100.f;
                ++sensorNoiseFrame_;
            }
            return p;
        }
}// namespace threepp
