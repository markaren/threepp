#include "VulkanCoreImpl.hpp"

namespace threepp {

void VulkanRendererCore::CoreImpl::rewriteTaaDescriptors() {
            std::array<VkImageView, kFramesInFlight> motionViews{};
            std::array<VkImageView, kFramesInFlight> idsViews{};
            std::array<VkImageView, kFramesInFlight> depthViews{};
            std::array<VkImageView, kFramesInFlight> sceneHdrViews{};
            std::array<VkImageView, kFramesInFlight> bloomViews{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                motionViews[f]   = rasterGbufs[f].motion.view;
                idsViews[f]      = rasterGbufs[f].ids.view;
                depthViews[f]    = rasterGbufs[f].depth.view;
                // HDR-mode-only sources (setTaaHdrInput) — harmlessly bound +
                // unused when the toggle is off. bloom_ exists by the time
                // this runs (constructed before the first rewriteTaaDescriptors
                // call in both the ctor and resize paths).
                sceneHdrViews[f] = bloom_->sceneHdrView(f);
                bloomViews[f]    = bloom_->bloomView(f);
            }
            const auto& swapViews = ctx->swapchainImageViews();
            vulkan::TaaResolve::DescriptorWriteInputs in{};
            in.gbufSampler        = gbufSampler_;
            in.gbufMotionPerFrame = motionViews.data();
            in.gbufIdsPerFrame    = idsViews.data();
            in.gbufDepthPerFrame  = depthViews.data();
            in.swapchainViews     = swapViews.data();
            in.sceneHdrPerFrame   = sceneHdrViews.data();
            in.bloomPerFrame      = bloomViews.data();
            taa_->rewriteDescriptors(in);
        }

void VulkanRendererCore::CoreImpl::rewriteBloomDescriptors() {
            bloom_->rewriteDescriptors();
            // HDR-mode-only scratch images (TaaResolve::mblurOutHdr_ /
            // PostComposite::hdrOut_) are VRAM-costing and only needed when
            // setTaaHdrInput(true) is active — allocate them lazily, here,
            // BEFORE the view-capture loop below reads their views, so a
            // default-off run never allocates them (mirrors the occlusion-
            // culling farthest-HiZ pyramid: allocated only while its feature
            // is enabled). Once allocated they are left allocated if the
            // toggle turns off (no churn) — see ensureHdrMblurImages /
            // resizeHdrOutput's own idempotency for why repeat calls here
            // are cheap.
            if (taaHdrInput_) {
                const VkExtent2D outExt = ctx->swapchainExtent();
                taa_->ensureHdrMblurImages(outExt.width, outExt.height);
                post_->resizeHdrOutput(outExt.width, outExt.height);
            }
            std::array<VkImageView, kFramesInFlight> sceneViews{};
            std::array<VkImageView, kFramesInFlight> bloomViews{};
            std::array<VkImageView, kFramesInFlight> gbufViews{};
            std::array<VkImageView, kFramesInFlight> idsViews{};
            std::array<VkImageView, kFramesInFlight> taaInViews{};
            std::array<VkImageView, kFramesInFlight> hdrSceneViews{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                sceneViews[f] = bloom_->sceneHdrView(f);
                bloomViews[f] = bloom_->bloomView(f);
                // PostComposite's gbuf binding was the old PT sky source; the
                // deferred path's solid-bg sky test reads the raster ids
                // (skyIdsFromRasterGbuf()), so bind a valid raster view here —
                // it is unused in deferred mode but must stay non-null.
                gbufViews[f]  = rasterGbufs[f].ids.view;
                idsViews[f]   = rasterGbufs[f].ids.view;
                taaInViews[f] = taa_->inputView(f);
                // HDR-mode-only (setTaaHdrInput): PostComposite's binding 6
                // reads whichever history slot THIS frame-in-flight's
                // TaaResolve::recordResolve call just wrote — or, when
                // motion blur is active, the HDR motion-blur intermediate.
                // motionBlurAmount_ is checked here (not per-frame) because
                // this rewrite only runs on resize/toggle, same cadence as
                // every other view capture in this function. Guarded on
                // hdrMblurImagesValid() so this never reads mblurHdrView()
                // before ensureHdrMblurImages has actually run for it (e.g.
                // taaHdrInput_ is still false — the view is unused by the
                // shader in that case anyway, but must stay non-null).
                hdrSceneViews[f] = (motionBlurAmount_ > 0.f && taa_->hdrMblurImagesValid())
                        ? taa_->mblurHdrView(f)
                        : taa_->historyView(vulkan::TaaResolve::writeSlotFor(f));
            }
            vulkan::PostComposite::DescriptorWriteInputs in{};
            in.sceneHdrPerFrame  = sceneViews.data();
            in.bloomPerFrame     = bloomViews.data();
            in.gbufPerFrame      = gbufViews.data();
            in.rasterIdsPerFrame = idsViews.data();
            in.taaInputPerFrame  = taaInViews.data();
            in.hdrScenePerFrame  = hdrSceneViews.data();
            post_->rewriteDescriptors(in);

            // Finalize (HDR-mode PostComposite output → swapchain, via
            // TaaResolve's RCAS/copy — see TaaResolve::recordPostFinalize).
            {
                std::array<VkImageView, kFramesInFlight> hdrOutViews{};
                std::array<VkImage,     kFramesInFlight> hdrOutImages{};
                for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                    hdrOutViews[f]  = post_->hdrOutView(f);
                    hdrOutImages[f] = post_->hdrOutImage(f);
                }
                const auto& swapViews  = ctx->swapchainImageViews();
                const auto& swapImages = ctx->swapchainImages();
                taa_->rewritePostFinalizeDescriptors(hdrOutViews.data(), hdrOutImages.data(),
                                                     swapViews.data(), swapImages.data());
            }

            // DoF scratch + descriptors track the same lifetimes (sceneHdr /
            // raster depth / render extent), so (re)fit it here too.
            if (dof_) {
                std::array<VkBuffer, kFramesInFlight>    camBufs{};
                std::array<VkImageView, kFramesInFlight> depthViews{};
                for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                    camBufs[f]    = cameraUbos[f].handle;
                    depthViews[f] = rasterGbufs[f].depth.view;
                }
                vulkan::DofPass::ResizeInputs din{};
                din.cameraUbos       = camBufs.data();
                din.depthPerFrame    = depthViews.data();
                din.sceneHdrPerFrame = sceneViews.data();
                dof_->resize(renderExtent().width, renderExtent().height, din);
            }

        }

void VulkanRendererCore::CoreImpl::rewriteDeferredDescriptors() {
            // Needs a built TLAS (binding 8) + material buffer (binding 9). Both
            // come from the scene build; before then there's nothing to bind and
            // the deferred pass can't dispatch anyway. ensureSceneBuilt calls
            // this right after the TLAS build, so the first valid write lands
            // before the first deferred dispatch.
            if (!deferredShade_ || tlas == VK_NULL_HANDLE) return;
            std::array<VkBuffer, kFramesInFlight>    camBufs{};
            std::array<VkBuffer, kFramesInFlight>    lightBufs{};
            std::array<VkBuffer, kFramesInFlight>    matBufs{};
            std::array<VkBuffer, kFramesInFlight>    emBufs{};
            std::array<VkImageView, kFramesInFlight> normalViews{};
            std::array<VkImageView, kFramesInFlight> depthViews{};
            std::array<VkImageView, kFramesInFlight> idsViews{};
            std::array<VkImageView, kFramesInFlight> albedoViews{};
            std::array<VkImageView, kFramesInFlight> uvViews{};
            std::array<VkImageView, kFramesInFlight> motionViews{};
            std::array<VkImageView, kFramesInFlight> indirectViews{};
            std::array<VkImageView, kFramesInFlight> momentsSqViews{};
            std::array<VkImageView, kFramesInFlight> atrousAViews{};
            std::array<VkImageView, kFramesInFlight> atrousBViews{};
            std::array<VkImageView, kFramesInFlight> reflectViews{};
            std::array<VkImageView, kFramesInFlight> reflAuxViews{};
            std::array<VkImageView, kFramesInFlight> shadowVisViews{};
            std::array<VkImageView, kFramesInFlight> directUViews{};
            std::array<VkImageView, kFramesInFlight> shadowAtrousAViews{};
            std::array<VkImageView, kFramesInFlight> shadowAtrousBViews{};
            std::array<VkImageView, kFramesInFlight> froxelScatterViews{};
            std::array<VkImageView, kFramesInFlight> froxelLutViews{};
            std::array<VkImageView, kFramesInFlight> sceneHdrViews{};
            std::array<VkBuffer, kFramesInFlight> fogBufs{};
            std::array<VkBuffer, kFramesInFlight> clusterGridBufs{};
            std::array<VkBuffer, kFramesInFlight> clusterLightBufs{};
            // MSAA raw raster attachments (dispatch B). Real views when
            // gbufMsaaSamples_>1 and they exist; the 1x1 dummy MS images
            // otherwise (msaa=1's default path, or the brief window before
            // the first MSAA image realloc completes).
            std::array<VkImageView, kFramesInFlight> normalMSViews{};
            std::array<VkImageView, kFramesInFlight> depthMSViews{};
            std::array<VkImageView, kFramesInFlight> idsMSViews{};
            std::array<VkImageView, kFramesInFlight> albedoMSViews{};
            std::array<VkImageView, kFramesInFlight> uvMSViews{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                camBufs[f]       = cameraUbos[f].handle;
                lightBufs[f]     = lightsUbos[f].handle;
                fogBufs[f]       = fogUbos[f].handle;
                clusterGridBufs[f]  = clusterGridBuffers[f].handle;
                clusterLightBufs[f] = clusterLightsBuffers[f].handle;
                matBufs[f]       = materialDescsBuffers[f].handle;
                emBufs[f]        = emissiveTriBuffers[f].handle;
                normalViews[f]   = rasterGbufs[f].normal.view;
                depthViews[f]    = rasterGbufs[f].depth.view;
                idsViews[f]      = rasterGbufs[f].ids.view;
                albedoViews[f]   = rasterGbufs[f].albedo.view;
                uvViews[f]       = rasterGbufs[f].uv.view;
                motionViews[f]   = rasterGbufs[f].motion.view;
                indirectViews[f] = rasterGbufs[f].indirect.view;
                momentsSqViews[f]= rasterGbufs[f].momentsSq.view;
                atrousAViews[f]  = rasterGbufs[f].atrousA.view;
                atrousBViews[f]  = rasterGbufs[f].atrousB.view;
                reflectViews[f]  = rasterGbufs[f].reflect.view;
                reflAuxViews[f]  = rasterGbufs[f].reflAux.view;
                shadowVisViews[f]     = rasterGbufs[f].shadowVis.view;
                directUViews[f]       = rasterGbufs[f].directU.view;
                shadowAtrousAViews[f] = rasterGbufs[f].shadowAtrousA.view;
                shadowAtrousBViews[f] = rasterGbufs[f].shadowAtrousB.view;
                froxelScatterViews[f] = rasterGbufs[f].froxelScatter.view;
                froxelLutViews[f]     = rasterGbufs[f].froxelLut.view;
                sceneHdrViews[f] = bloom_->sceneHdrView(f);
                const bool haveMS = gbufMsaaSamples_ > 1 && rasterGbufs[f].normalMS.view != VK_NULL_HANDLE;
                normalMSViews[f] = haveMS ? rasterGbufs[f].normalMS.view : gbufDummyMS_[0].view;
                depthMSViews[f]  = haveMS ? rasterGbufs[f].depthMS.view  : gbufDummyMS_[1].view;
                idsMSViews[f]    = haveMS ? rasterGbufs[f].idsMS.view    : gbufDummyMS_[2].view;
                uvMSViews[f]     = haveMS ? rasterGbufs[f].uvMS.view     : gbufDummyMS_[3].view;
                albedoMSViews[f] = haveMS ? rasterGbufs[f].albedoMS.view : gbufDummyMS_[4].view;
            }
            // Bindless material-texture array for reflected-hit texturing —
            // same source the RT set's binding 8 uses.
            std::array<VkDescriptorImageInfo, kMaxMaterialTextures> matTexInfos{};
            fillMaterialTextureInfos(matTexInfos);

            vulkan::DeferredShade::DescriptorWriteInputs in{};
            in.cameraUbo        = camBufs.data();
            in.lightsUbo        = lightBufs.data();
            in.envView          = envImage.view;
            in.envSampler       = envImage.sampler;
            in.gbufNormal       = normalViews.data();
            in.gbufDepth        = depthViews.data();
            in.gbufIds          = idsViews.data();
            in.gbufAlbedo       = albedoViews.data();
            in.gbufUv           = uvViews.data();
            in.gbufMotion       = motionViews.data();
            in.indirect         = indirectViews.data();
            in.momentsSq        = momentsSqViews.data();
            in.atrousA          = atrousAViews.data();
            in.atrousB          = atrousBViews.data();
            in.reflect          = reflectViews.data();
            in.reflAux          = reflAuxViews.data();
            in.shadowVis        = shadowVisViews.data();
            in.directU          = directUViews.data();
            in.shadowAtrousA    = shadowAtrousAViews.data();
            in.shadowAtrousB    = shadowAtrousBViews.data();
            in.clusterGrid      = clusterGridBufs.data();
            in.clusterLights    = clusterLightBufs.data();
            in.froxelScatter    = froxelScatterViews.data();
            in.froxelLut        = froxelLutViews.data();
            in.sceneHdr         = sceneHdrViews.data();
            in.fogBuf           = fogBufs.data();
            in.fogRange         = sizeof(GpuFogUbo);
            in.tlas             = tlas;
            in.materialBuf      = matBufs.data();
            in.geomDescBuf      = geometryDescsBuffer.handle;
            in.materialTex      = matTexInfos.data();
            in.materialTexCount = kMaxMaterialTextures;
            in.emissiveTriBuf   = emBufs.data();
            // Ocean textures for the thin-shell water branch. These renderer
            // members are the live ocean FFT-fine + foam views/samplers (or 1×1
            // dummies + tile size 0 when no DisplacedMesh is present) — the same
            // handles the RT set binds at 32 + 44. ensureDisplacedState sets them
            // BEFORE the scene-build descriptor rewrite that calls us, so the
            // deferred set always picks up the current ocean state.
            in.oceanFineView    = oceanFineHeightView;
            in.oceanFineSampler = oceanFineHeightSampler;
            in.oceanFoamView    = oceanFoamView;
            in.oceanFoamSampler = oceanFoamSampler;
            in.foamDetailView    = foamDetailImage.view;
            in.foamDetailSampler = foamDetailImage.sampler;
            // Shared blue-noise tile — dithers the deferred GI hemisphere
            // directions so the 1-spp residual denoises cleanly instead of
            // showing as AO speckle.
            in.blueNoise         = blueNoiseImage.view;
            // ReSTIR DI reservoir ping-pong — the same 2 physical images the
            // deferred shade's NEE uses (created unconditionally in
            // createReservoirImages). rewriteDescriptors picks write/read
            // slots per frame. Locals must outlive the call below.
            const VkImageView resPosViews[2] = {reservoirPosImagesPP[0].view, reservoirPosImagesPP[1].view};
            const VkImageView resWViews[2]   = {reservoirWImagesPP[0].view,   reservoirWImagesPP[1].view};
            in.reservoirPos     = resPosViews;
            in.reservoirW       = resWViews;
            // Probe GI (bindings 36/37/54) — the SH store + per-frame grid UBO
            // + Chebyshev depth store ProbeGI owns. Real buffers even when the
            // feature is off (the grid UBO's enable flag gates sampling).
            in.probeShBuf       = probeGI_->shBuffer();
            in.probeGridUbo     = probeGI_->gridUbos();
            in.probeDepthBuf    = probeGI_->depthBuffer();
            // Hybrid SSR (bindings 55-57): (re)fit the HiZ pyramid to the
            // current render extent + depth views (idempotent — this rewrite
            // runs on exactly the resize/realloc events the pyramid must
            // track), then hand its view/sampler + the raster camera UBOs
            // (forward jittered VP) to the shade set.
            std::array<VkBuffer, kFramesInFlight> rasterCamBufs{};
            for (uint32_t f = 0; f < kFramesInFlight; ++f)
                rasterCamBufs[f] = rasterCameraUbos[f].handle;
            // msSamples stays 1: the SSR walk reads the RESOLVED depth even
            // under MSAA (binding 3 gets the MS/dummy view purely to keep the
            // descriptor valid).
            hiz_->resize(renderExtent(), depthViews.data(), depthMSViews.data(), 1);
            in.hizView          = hiz_->view();
            in.hizSampler       = hiz_->sampler();
            in.rasterCamUbo     = rasterCamBufs.data();
            in.gbufNormalMS     = normalMSViews.data();
            in.gbufDepthMS      = depthMSViews.data();
            in.gbufIdsMS        = idsMSViews.data();
            in.gbufAlbedoMS     = albedoMSViews.data();
            in.gbufUvMS         = uvMSViews.data();
            deferredShade_->rewriteDescriptors(in);
            // The probe UPDATE pass consumes the same scene inputs (TLAS,
            // lights, env, material/geometry/emissive buffers) — keep its set
            // in lockstep with the deferred one.
            {
                vulkan::ProbeGI::DescriptorWriteInputs pin{};
                pin.lightsUbo        = lightBufs.data();
                pin.envView          = envImage.view;
                pin.envSampler       = envImage.sampler;
                pin.tlas             = tlas;
                pin.materialBuf      = matBufs.data();
                pin.geomDescBuf      = geometryDescsBuffer.handle;
                pin.materialTex      = matTexInfos.data();
                pin.materialTexCount = kMaxMaterialTextures;
                pin.emissiveTriBuf   = emBufs.data();
                probeGI_->rewriteDescriptors(pin);
            }
        }

void VulkanRendererCore::CoreImpl::fitProbeGridToScene() {
            if (!probeGI_ || lastVisibleEntries_.empty()) return;
            Box3 sceneBox;
            for (const auto& en : lastVisibleEntries_) {
                if (en.isOverlay || en.isParticle) continue;
                auto geom = en.mesh->geometry();
                if (!geom) continue;
                if (!geom->boundingBox) geom->computeBoundingBox();
                if (!geom->boundingBox) continue;
                Box3 worldAabb = *geom->boundingBox;
                Matrix4 w;
                std::memcpy(w.elements.data(), en.worldMatrix.data(), 64);
                worldAabb.applyMatrix4(w);
                sceneBox.union_(worldAabb);
            }
            if (sceneBox.isEmpty()) return;
            const float mn[3] = {sceneBox.min().x, sceneBox.min().y, sceneBox.min().z};
            const float mx[3] = {sceneBox.max().x, sceneBox.max().y, sceneBox.max().z};
            probeGI_->setGridBounds(mn, mx);
        }

void VulkanRendererCore::CoreImpl::ensureGbufDummyMS() {
            if (gbufDummyMSCreated_) return;
            const VkFormat fmts[5] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT,
                                      VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                      VK_FORMAT_R8G8B8A8_UNORM};
            const VkImageAspectFlags aspects[5] = {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                   VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   VK_IMAGE_ASPECT_COLOR_BIT};
            const char* names[5] = {"gbufDummyMS.normal", "gbufDummyMS.depth", "gbufDummyMS.ids",
                                    "gbufDummyMS.uv", "gbufDummyMS.albedo"};
            for (int i = 0; i < 5; ++i) {
                const VkImageUsageFlags usage = (aspects[i] == VK_IMAGE_ASPECT_DEPTH_BIT)
                        ? (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
                        : (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
                gbufDummyMS_[i] = createAttachmentImage2D(1, 1, fmts[i], usage, aspects[i],
                                                          names[i], VK_SAMPLE_COUNT_2_BIT);
            }
            // Layout-init: SHADER_READ_ONLY (color) / DEPTH_STENCIL_READ_ONLY
            // (depth) — matches what deferred_shade.comp's descriptor binds
            // expect. Contents are never read meaningfully (dispatch B never
            // runs at msaa=1 — pc.shadeMode stays 0), only the layout must be
            // valid to satisfy descriptor/barrier validation.
            VkCommandBuffer initCb = beginOneShot();
            for (int i = 0; i < 5; ++i) {
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = (aspects[i] == VK_IMAGE_ASPECT_DEPTH_BIT)
                        ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = gbufDummyMS_[i].image;
                b.subresourceRange.aspectMask = aspects[i];
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                b.srcAccessMask = 0;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(initCb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &b);
            }
            endAndSubmitOneShot(initCb, "gbufDummyMS init layouts");
            gbufDummyMSCreated_ = true;
        }

void VulkanRendererCore::CoreImpl::ensureHybridResources() {
            ensureGbufDummyMS();
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
                // Two-phase occlusion load/store variants (compatible).
                createOcclRenderPasses(VK_SAMPLE_COUNT_1_BIT, occlRenderPassA_, occlRenderPassB_);
                createRasterCameraUbos();
                createRasterDsLayoutAndPool();
                createRasterGbufPipeline();
            }
            // Occlusion-cull compute (pipeline + visBits + phase buffers are
            // cheap; the farthest pyramid's IMAGE is only allocated when the
            // feature is enabled — see rewriteBloomDescriptors).
            if (!occl_) {
                occl_    = std::make_unique<vulkan::OcclusionCull>(*ctx, cmdPool, kFramesInFlight);
                occlHiz_ = std::make_unique<vulkan::HiZPyramid>(*ctx, kFramesInFlight);
            }
            // MSAA render pass + pipelines — only built when opted in, and
            // rebuilt when the sample count changes (2↔4) or MSAA is turned
            // off (torn down; the 1× path above is untouched either way).
            // gbufMsaaBuiltSamples_ tracks what's currently built so a no-op
            // setGbufferMsaa(N) (already N) doesn't thrash pipelines every
            // ensureHybridResources call.
            const auto wantSamples = gbufMsaaSamples_ == 4 ? VK_SAMPLE_COUNT_4_BIT
                                    : gbufMsaaSamples_ == 2 ? VK_SAMPLE_COUNT_2_BIT
                                                             : VK_SAMPLE_COUNT_1_BIT;
            if (wantSamples != gbufMsaaBuiltSamples_) {
                if (wantSamples == VK_SAMPLE_COUNT_1_BIT) {
                    // Turned MSAA off — tear down the MS render pass/pipelines.
                    // The MS *images* are freed by destroyRasterGbufImages,
                    // called unconditionally from createRasterGbufImages below.
                    if (rasterGbufPipelineMS != VK_NULL_HANDLE) {
                        vkDestroyPipeline(ctx->device(), rasterGbufPipelineMS, nullptr);
                        rasterGbufPipelineMS = VK_NULL_HANDLE;
                    }
                    if (rasterGbufIndirectPipelineMS != VK_NULL_HANDLE) {
                        vkDestroyPipeline(ctx->device(), rasterGbufIndirectPipelineMS, nullptr);
                        rasterGbufIndirectPipelineMS = VK_NULL_HANDLE;
                    }
                    if (rasterGbufDecalPipelineMS != VK_NULL_HANDLE) {
                        vkDestroyPipeline(ctx->device(), rasterGbufDecalPipelineMS, nullptr);
                        rasterGbufDecalPipelineMS = VK_NULL_HANDLE;
                    }
                    if (rasterGbufRenderPassMS != VK_NULL_HANDLE) {
                        vkDestroyRenderPass(ctx->device(), rasterGbufRenderPassMS, nullptr);
                        rasterGbufRenderPassMS = VK_NULL_HANDLE;
                    }
                    if (occlRenderPassAMS_ != VK_NULL_HANDLE) {
                        vkDestroyRenderPass(ctx->device(), occlRenderPassAMS_, nullptr);
                        occlRenderPassAMS_ = VK_NULL_HANDLE;
                    }
                    if (occlRenderPassBMS_ != VK_NULL_HANDLE) {
                        vkDestroyRenderPass(ctx->device(), occlRenderPassBMS_, nullptr);
                        occlRenderPassBMS_ = VK_NULL_HANDLE;
                    }
                } else {
                    createRasterGbufRenderPassMS(wantSamples);
                    createRasterGbufPipelineMS(wantSamples);
                    if (occlRenderPassAMS_ != VK_NULL_HANDLE)
                        vkDestroyRenderPass(ctx->device(), occlRenderPassAMS_, nullptr);
                    if (occlRenderPassBMS_ != VK_NULL_HANDLE)
                        vkDestroyRenderPass(ctx->device(), occlRenderPassBMS_, nullptr);
                    createOcclRenderPasses(wantSamples, occlRenderPassAMS_, occlRenderPassBMS_);
                }
                gbufMsaaBuiltSamples_ = wantSamples;
                // Force the image block below to re-run even if the extent
                // is unchanged — MS attachments/framebuffer must be (re)created
                // (or freed, on the ==1 branch) for the new sample count.
                rasterGbufs[0].width = 0;
            }
            if (overlayWireframePipeline == VK_NULL_HANDLE) {
                createOverlayPipeline();
            }
            if (particlePipelineNormal_ == VK_NULL_HANDLE) {
                createParticlePipeline();
            }
            // Raster G-buffer matches the deferred shade's render extent —
            // the shade compute dispatch reads it 1:1 by launch coord, so it
            // must launch at the same resolution the gbuffer rasterized at.
            const VkExtent2D ext = renderExtent();
            const bool gbufImagesRebuilt =
                    rasterGbufs[0].width != ext.width || rasterGbufs[0].height != ext.height;
            if (gbufImagesRebuilt) {
                createRasterGbufImages(ext.width, ext.height);
            }
            // GbufResolve — lazily constructed the first time MSAA turns on;
            // descriptors rewritten whenever the MS/resolved image views
            // changed (any image rebuild above) so they never point at
            // stale/destroyed views.
            if (gbufMsaaSamples_ > 1) {
                if (!gbufResolve_) {
                    gbufResolve_ = std::make_unique<vulkan::GbufResolve>(*ctx, kFramesInFlight);
                }
                if (gbufImagesRebuilt) {
                    std::array<VkImageView, kFramesInFlight> nMS{}, mMS{}, iMS{}, uMS{}, aMS{}, dMS{};
                    std::array<VkImageView, kFramesInFlight> nR{}, mR{}, iR{}, uR{}, aR{};
                    for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                        nMS[f] = rasterGbufs[f].normalMS.view;
                        mMS[f] = rasterGbufs[f].motionMS.view;
                        iMS[f] = rasterGbufs[f].idsMS.view;
                        uMS[f] = rasterGbufs[f].uvMS.view;
                        aMS[f] = rasterGbufs[f].albedoMS.view;
                        dMS[f] = rasterGbufs[f].depthMS.view;
                        nR[f]  = rasterGbufs[f].normal.view;
                        mR[f]  = rasterGbufs[f].motion.view;
                        iR[f]  = rasterGbufs[f].ids.view;
                        uR[f]  = rasterGbufs[f].uv.view;
                        aR[f]  = rasterGbufs[f].albedo.view;
                    }
                    vulkan::GbufResolve::DescriptorWriteInputs gin{};
                    gin.normalMS = nMS.data(); gin.motionMS = mMS.data(); gin.idsMS = iMS.data();
                    gin.uvMS = uMS.data(); gin.albedoMS = aMS.data(); gin.depthMS = dMS.data();
                    gin.normalResolved = nR.data(); gin.motionResolved = mR.data();
                    gin.idsResolved = iR.data(); gin.uvResolved = uR.data(); gin.albedoResolved = aR.data();
                    gbufResolve_->rewriteDescriptors(gin);
                }
            }

            // Occlusion culling's farthest pyramid: only allocated while the
            // feature is on (its image is the feature's whole VRAM cost).
            // This runs before every frame's raster and inherits the resize/
            // MSAA-toggle idle waits, so extent + sample-count changes and a
            // mid-run setOcclusionCulling(true) all land here. Under MSAA
            // the mip-0 source is the RAW MS depth attachment (the resolved
            // depth doesn't exist until gbuf_resolve, which runs AFTER the
            // two-phase raster).
            if (occlHiz_ && occlusionCullingEnabled_ &&
                rasterGbufs[0].depth.view != VK_NULL_HANDLE) {
                const bool haveMS = gbufMsaaSamples_ > 1 &&
                                    rasterGbufs[0].depthMS.view != VK_NULL_HANDLE;
                std::array<VkImageView, kFramesInFlight> dv{};
                std::array<VkImageView, kFramesInFlight> dvMS{};
                for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                    dv[f]   = rasterGbufs[f].depth.view;
                    dvMS[f] = haveMS ? rasterGbufs[f].depthMS.view : gbufDummyMS_[1].view;
                }
                occlHiz_->resize(renderExtent(), dv.data(), dvMS.data(),
                                 haveMS ? gbufMsaaSamples_ : 1u);
            }

            // HDR-mode-only scratch images (setTaaHdrInput): same "only
            // allocated while the feature is on" shape as the occlusion
            // pyramid above. taa_/bloom_/post_ don't exist yet on the very
            // first (ctor-time) call to ensureHybridResources — guard on
            // them. taaHdrPlumbingDirty_ starts true so the first real call
            // (after ctor finishes constructing them) always runs this once;
            // afterwards it only re-runs when setTaaHdrInput/setMotionBlur
            // actually changed something relevant (avoids per-frame
            // vkUpdateDescriptorSets churn in the steady state).
            if (taaHdrPlumbingDirty_ && taa_ && bloom_ && post_) {
                rewriteBloomDescriptors();// allocates lazily (gated on
                                          // taaHdrInput_), then rewrites —
                                          // see that function's ordering.
                taaHdrPlumbingDirty_ = false;
            }
        }

}// namespace threepp
