#include "VulkanCoreImpl.hpp"

#include "threepp/renderers/vulkan/shaders/gbuffer.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuffer.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/gbuffer_indirect.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_billboard.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_glow.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_billboard.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_gbuf.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_fullscreen.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_inject.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_depth.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_depth.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_overlay_depth.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_color.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_color.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_point.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_point.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_sprite.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/particle.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/particle.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/sprite3d.vert.spv.h"

namespace threepp {

void VulkanRenderer::Impl::clearGbufImages() {
            VkCommandBuffer cb = beginOneShot();

            // Clears the ReSTIR DI reservoir ping-pong (pos + W) so M=0 on
            // frame 0's read side and the deferred shade's temporal-reuse path
            // correctly sees "no prior history" instead of garbage.
            std::array<VkImage, 4> images = {
                    view().reservoirPosImagesPP[0].image, view().reservoirPosImagesPP[1].image,
                    view().reservoirWImagesPP[0].image, view().reservoirWImagesPP[1].image,
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
                toGeneral[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
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

void VulkanRenderer::Impl::destroyRasterGbufMsObjects() {
            if (!ctx) return;
            VkDevice d = ctx->device();
            if (rasterGbufPipelineMS != VK_NULL_HANDLE) {
                vkDestroyPipeline(d, rasterGbufPipelineMS, nullptr);
                rasterGbufPipelineMS = VK_NULL_HANDLE;
            }
            if (rasterGbufIndirectPipelineMS != VK_NULL_HANDLE) {
                vkDestroyPipeline(d, rasterGbufIndirectPipelineMS, nullptr);
                rasterGbufIndirectPipelineMS = VK_NULL_HANDLE;
            }
            if (rasterGbufDecalPipelineMS != VK_NULL_HANDLE) {
                vkDestroyPipeline(d, rasterGbufDecalPipelineMS, nullptr);
                rasterGbufDecalPipelineMS = VK_NULL_HANDLE;
            }
            if (rasterGbufParticlePipelineMS != VK_NULL_HANDLE) {
                vkDestroyPipeline(d, rasterGbufParticlePipelineMS, nullptr);
                rasterGbufParticlePipelineMS = VK_NULL_HANDLE;
            }
            if (rasterGbufRenderPassMS != VK_NULL_HANDLE) {
                vkDestroyRenderPass(d, rasterGbufRenderPassMS, nullptr);
                rasterGbufRenderPassMS = VK_NULL_HANDLE;
            }
            if (occlRenderPassAMS_ != VK_NULL_HANDLE) {
                vkDestroyRenderPass(d, occlRenderPassAMS_, nullptr);
                occlRenderPassAMS_ = VK_NULL_HANDLE;
            }
            if (occlRenderPassBMS_ != VK_NULL_HANDLE) {
                vkDestroyRenderPass(d, occlRenderPassBMS_, nullptr);
                occlRenderPassBMS_ = VK_NULL_HANDLE;
            }
        }

void VulkanRenderer::Impl::destroyRasterGbufImages() {
            if (!ctx) return;
            VkDevice d = ctx->device();
            for (auto& g : view().rasterGbufs) {
                if (g.framebuffer) {
                    vkDestroyFramebuffer(d, g.framebuffer, nullptr);
                    g.framebuffer = VK_NULL_HANDLE;
                }
                if (g.framebufferMS) {
                    vkDestroyFramebuffer(d, g.framebufferMS, nullptr);
                    g.framebufferMS = VK_NULL_HANDLE;
                }
                destroyImage2D(ctx->allocator(), d, g.normal);
                destroyImage2D(ctx->allocator(), d, g.motion);
                destroyImage2D(ctx->allocator(), d, g.ids);
                destroyImage2D(ctx->allocator(), d, g.uv);
                destroyImage2D(ctx->allocator(), d, g.albedo);
                destroyImage2D(ctx->allocator(), d, g.indirect);
                destroyImage2D(ctx->allocator(), d, g.momentsSq);
                destroyImage2D(ctx->allocator(), d, g.atrousA);
                destroyImage2D(ctx->allocator(), d, g.atrousB);
                destroyImage2D(ctx->allocator(), d, g.reflect);
                destroyImage2D(ctx->allocator(), d, g.reflAux);
                destroyImage2D(ctx->allocator(), d, g.shadowVis);
                destroyImage2D(ctx->allocator(), d, g.directU);
                destroyImage2D(ctx->allocator(), d, g.shadowAtrousA);
                destroyImage2D(ctx->allocator(), d, g.shadowAtrousB);
                destroyImage2D(ctx->allocator(), d, g.froxelScatter);
                destroyImage2D(ctx->allocator(), d, g.froxelLut);
                destroyImage2D(ctx->allocator(), d, g.cloudColor);
                destroyImage2D(ctx->allocator(), d, g.cloudAux);
                destroyImage2D(ctx->allocator(), d, g.cloudShadow);
                destroyImage2D(ctx->allocator(), d, g.splatDepth);
                destroyImage2D(ctx->allocator(), d, g.depth);
                destroyImage2D(ctx->allocator(), d, g.unjitDepth);
                destroyImage2D(ctx->allocator(), d, g.normalMS);
                destroyImage2D(ctx->allocator(), d, g.motionMS);
                destroyImage2D(ctx->allocator(), d, g.idsMS);
                destroyImage2D(ctx->allocator(), d, g.uvMS);
                destroyImage2D(ctx->allocator(), d, g.albedoMS);
                destroyImage2D(ctx->allocator(), d, g.depthMS);
                g.width  = 0;
                g.height = 0;
            }
        }

void VulkanRenderer::Impl::createRasterGbufRenderPass() {
            VkAttachmentDescription attachments[6]{};
            // 0: world-space normal (rgba16f). FragShader writes; deferred shade samples.
            attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
            attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // 1: motion vector (rgba16f, only rg used).
            attachments[1] = attachments[0];
            // 2: per-pixel IDs + flags (rgba16ui).
            attachments[2] = attachments[0];
            attachments[2].format = VK_FORMAT_R16G16B16A16_UINT;
            // 3: material UV (rgba16f, only rg used).
            attachments[3] = attachments[0];
            // 4: albedo + metalness (rgba8 unorm) — raster-first deferred input.
            attachments[4]        = attachments[0];
            attachments[4].format = VK_FORMAT_R8G8B8A8_UNORM;
            // 5: depth (d32_sfloat).
            attachments[5]              = attachments[0];
            attachments[5].format       = VK_FORMAT_D32_SFLOAT;
            attachments[5].finalLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRefs[5]{};
            for (uint32_t i = 0; i < 5; ++i) {
                colorRefs[i].attachment = i;
                colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            VkAttachmentReference depthRef{};
            depthRef.attachment = 5;
            depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount    = 5;
            subpass.pColorAttachments       = colorRefs;
            subpass.pDepthStencilAttachment = &depthRef;

            // Sandwich the pass between (any prior deferred-shade ray-query
            // reads of these attachments) and (the post-pass deferred-shade
            // consumer). Vulkan doesn't know the shade's ray-query reads them
            // — we declare the synchronization explicitly via subpass
            // dependencies.
            VkSubpassDependency deps[2]{};
            deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
            deps[0].dstSubpass    = 0;
            deps[0].srcStageMask  = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].srcSubpass    = 0;
            deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
            deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            // Consumed by the deferred shade compute pass — make the
            // attachments visible to that stage.
            deps[1].dstStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            VkRenderPassCreateInfo rpci{};
            rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rpci.attachmentCount = 6;
            rpci.pAttachments    = attachments;
            rpci.subpassCount    = 1;
            rpci.pSubpasses      = &subpass;
            rpci.dependencyCount = 2;
            rpci.pDependencies   = deps;
            check(vkCreateRenderPass(ctx->device(), &rpci, nullptr, &rasterGbufRenderPass),
                  "vkCreateRenderPass(rasterGbuf)");
        }

void VulkanRenderer::Impl::createOcclRenderPasses(VkSampleCountFlagBits samples,
                            VkRenderPass& outA, VkRenderPass& outB) {
            auto build = [&](bool phaseA, VkRenderPass& out) {
                VkAttachmentDescription attachments[6]{};
                attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
                attachments[0].samples        = samples;
                attachments[0].loadOp         = phaseA ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                       : VK_ATTACHMENT_LOAD_OP_LOAD;
                attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
                attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                attachments[0].initialLayout  = phaseA ? VK_IMAGE_LAYOUT_UNDEFINED
                                                       : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                attachments[0].finalLayout    = phaseA ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                                       : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                attachments[1] = attachments[0];
                attachments[2] = attachments[0];
                attachments[2].format = VK_FORMAT_R16G16B16A16_UINT;
                attachments[3] = attachments[0];
                attachments[4]        = attachments[0];
                attachments[4].format = VK_FORMAT_R8G8B8A8_UNORM;
                attachments[5]        = attachments[0];
                attachments[5].format = VK_FORMAT_D32_SFLOAT;
                attachments[5].initialLayout = phaseA ? VK_IMAGE_LAYOUT_UNDEFINED
                                                      : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                attachments[5].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

                VkAttachmentReference colorRefs[5]{};
                for (uint32_t i = 0; i < 5; ++i) {
                    colorRefs[i].attachment = i;
                    colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                }
                VkAttachmentReference depthRef{};
                depthRef.attachment = 5;
                depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

                VkSubpassDescription subpass{};
                subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
                subpass.colorAttachmentCount    = 5;
                subpass.pColorAttachments       = colorRefs;
                subpass.pDepthStencilAttachment = &depthRef;

                // A entry: prior RT/compute reads of last frame's attachments.
                // A exit / B entry: the between-pass compute (HiZ + cull) and
                // the indirect-buffer read. B exit: same consumers as the
                // single pass (the deferred shade compute dispatch).
                VkSubpassDependency deps[2]{};
                deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
                deps[0].dstSubpass    = 0;
                deps[0].srcStageMask  = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
                deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                deps[1].srcSubpass    = 0;
                deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
                deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                deps[1].dstStageMask  = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                VkRenderPassCreateInfo rpci{};
                rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
                rpci.attachmentCount = 6;
                rpci.pAttachments    = attachments;
                rpci.subpassCount    = 1;
                rpci.pSubpasses      = &subpass;
                rpci.dependencyCount = 2;
                rpci.pDependencies   = deps;
                check(vkCreateRenderPass(ctx->device(), &rpci, nullptr, &out),
                      "vkCreateRenderPass(occl phase)");
            };
            build(true, outA);
            build(false, outB);
        }

void VulkanRenderer::Impl::createRasterGbufRenderPassMS(VkSampleCountFlagBits samples) {
            if (rasterGbufRenderPassMS != VK_NULL_HANDLE) {
                vkDestroyRenderPass(ctx->device(), rasterGbufRenderPassMS, nullptr);
                rasterGbufRenderPassMS = VK_NULL_HANDLE;
            }
            VkAttachmentDescription attachments[6]{};
            attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
            attachments[0].samples        = samples;
            attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments[1] = attachments[0];
            attachments[2] = attachments[0];
            attachments[2].format = VK_FORMAT_R16G16B16A16_UINT;
            attachments[3] = attachments[0];
            attachments[4]        = attachments[0];
            attachments[4].format = VK_FORMAT_R8G8B8A8_UNORM;
            attachments[5]              = attachments[0];
            attachments[5].format       = VK_FORMAT_D32_SFLOAT;
            attachments[5].finalLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            VkAttachmentReference colorRefs[5]{};
            for (uint32_t i = 0; i < 5; ++i) {
                colorRefs[i].attachment = i;
                colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            VkAttachmentReference depthRef{};
            depthRef.attachment = 5;
            depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount    = 5;
            subpass.pColorAttachments       = colorRefs;
            subpass.pDepthStencilAttachment = &depthRef;

            VkSubpassDependency deps[2]{};
            deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
            deps[0].dstSubpass    = 0;
            deps[0].srcStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].srcSubpass    = 0;
            deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
            deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            deps[1].dstStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            VkRenderPassCreateInfo rpci{};
            rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rpci.attachmentCount = 6;
            rpci.pAttachments    = attachments;
            rpci.subpassCount    = 1;
            rpci.pSubpasses      = &subpass;
            rpci.dependencyCount = 2;
            rpci.pDependencies   = deps;
            check(vkCreateRenderPass(ctx->device(), &rpci, nullptr, &rasterGbufRenderPassMS),
                  "vkCreateRenderPass(rasterGbufMS)");
        }

void VulkanRenderer::Impl::createRasterGbufImages(uint32_t w, uint32_t h) {
            destroyRasterGbufImages();
            // STORAGE is added to the resolve-target colour images only when
            // MSAA is active — gbuf_resolve.comp (a compute pass) imageStores
            // the dominant sample's normal/motion/ids/uv/albedo into them.
            // Omitted at msaa=1 to keep that path's image usage flags exactly
            // as they were (byte-identical guarantee — no behavioural surface
            // added when the feature is off).
            // TRANSFER_SRC is added so the AOV readback path
            // (VulkanRenderer::readGBufferAOV) can vkCmdCopyImageToBuffer the
            // resolved single-sample attachments to a host staging buffer. It is
            // a pure capability flag (no render-pass / layout / perf effect); the
            // STORAGE bit below stays MSAA-gated for its byte-identical guarantee.
            const VkImageUsageFlags colorUsage =
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    (gbufMsaaSamples_ > 1 ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
            const VkImageUsageFlags depthUsage =
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            for (size_t fi = 0; fi < view().rasterGbufs.size(); ++fi) {
                auto& g = view().rasterGbufs[fi];
                char nameBuf[64];
                auto N = [&](const char* tag) {
                    std::snprintf(nameBuf, sizeof(nameBuf), "rasterGbuf[%zu].%s", fi, tag);
                    return nameBuf;
                };
                g.normal = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("normal"));
                // motion additionally carries STORAGE, unconditionally: the
                // Gaussian-splat pass writes camera-reprojection motion into it
                // for the pixels a splat cloud owns (SplatPass.hpp), and a
                // storage write needs the usage bit at CREATION — by the time a
                // SplatCloud shows up in a scene these images already exist.
                // One image of the twenty here, and only the write path pays
                // whatever compression the bit costs.
                g.motion = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage | VK_IMAGE_USAGE_STORAGE_BIT,
                                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("motion"));
                g.ids    = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_UINT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("ids"));
                g.uv     = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("uv"));
                g.albedo = createAttachmentImage2D(w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                                   colorUsage, VK_IMAGE_ASPECT_COLOR_BIT,
                                                   N("albedo"));
                // Deferred GI accumulator / denoiser scratch — STORAGE (compute
                // write/read) + SAMPLED so the deferred shade can sample the OTHER
                // frame-in-flight's indirect as the 1-frame GI history (reprojected
                // by the motion vector — Phase-2 GI reproject). Never a render-pass
                // attachment, so not in the framebuffer.
                // TRANSFER_SRC on this and the five temporal siblings below
                // (momentsSq/reflect/reflAux/shadowVis/directU): the determinism
                // audit hashes them per frame (debugHashShadeImages) to name the
                // pass a divergence enters at. Pure capability bit.
                g.indirect = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                     VK_IMAGE_ASPECT_COLOR_BIT, N("indirect"));
                // SVGF second-moment accumulator E[L²] (single channel). Same
                // STORAGE+SAMPLED + ping-pong as indirect: deferred_shade reproject-
                // accumulates it, deferred_gi_filter reads it for the per-pixel
                // temporal variance that guides the à-trous (crisp where stable,
                // blurred where noisy). R32 (not R16): E[L²] SQUARES the un-pre-
                // exposed indirect luminance, so fp16 overflows to Inf once
                // lum > ~256 — routine with physical light units (sun in klux) —
                // and Inf−E[L]² turns the variance NaN. R32F storage is a
                // mandatory format; the extra bandwidth is 2 B/px. .g carries the
                // GI content-change TREND (deferred_shade antilag: a moving
                // object's stale contact-darkening/bounce fades at content rate
                // instead of dragging as a smear) — RG32F is likewise mandatory.
                g.momentsSq = createAttachmentImage2D(w, h, VK_FORMAT_R32G32_SFLOAT,
                                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                      VK_IMAGE_ASPECT_COLOR_BIT, N("momentsSq"));
                // SVGF multi-pass à-trous ping-pong scratch (rgb=GI, a=variance).
                // STORAGE only (compute imageLoad/Store, no sampling). The denoise
                // bounces the GI between these at widening step sizes.
                g.atrousA = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("atrousA"));
                g.atrousB = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("atrousB"));
                // Sharp mirror-ray reflection radiance — written by the shade, then
                // roughness-blurred + recombined by the reflection denoise pass.
                // SAMPLED too: the shade temporally accumulates it (samples the OTHER
                // frame-in-flight as 1-frame history) to anti-alias the 1-ray
                // refraction via the gbuffer jitter — sharp AND alias-free.
                g.reflect = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("reflect"));
                // Reflection-denoiser auxiliary — mirrors `reflect` exactly (STORAGE
                // write + SAMPLED prev-frame read, ping-ponged across frames-in-flight).
                g.reflAux = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("reflAux"));
                // Denoised direct-shadow channel: the shadow-ratio accumulator
                // (STORAGE write + SAMPLED prev-fif read, same ping-pong as
                // indirect), the unshadowed analytic direct the recombine
                // multiplies by the filtered ratio, and the ratio's own rg16f
                // à-trous scratch pair.
                g.shadowVis = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                      VK_IMAGE_ASPECT_COLOR_BIT, N("shadowVis"));
                g.directU = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT, N("directU"));
                g.shadowAtrousA = createAttachmentImage2D(w, h, VK_FORMAT_R16G16_SFLOAT,
                                                          VK_IMAGE_USAGE_STORAGE_BIT,
                                                          VK_IMAGE_ASPECT_COLOR_BIT, N("shadowAtrousA"));
                g.shadowAtrousB = createAttachmentImage2D(w, h, VK_FORMAT_R16G16_SFLOAT,
                                                          VK_IMAGE_USAGE_STORAGE_BIT,
                                                          VK_IMAGE_ASPECT_COLOR_BIT, N("shadowAtrousB"));
                // Froxel volumetrics — FIXED-size 3D volumes (independent of
                // the render extent; recreated here anyway on resize, which
                // also correctly resets the temporal history). KEEP the dims
                // in sync with froxel_inject/integrate.comp.
                g.froxelScatter = createImage3D(128, 72, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                N("froxelScatter"));
                g.froxelLut = createImage3D(128, 72, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                            N("froxelLut"));
                // Half-res volumetric cloud march targets (cloud_march.comp).
                // Sized to HALF the render extent — the ~4× perf win — and
                // recreated here on resize (which also resets the temporal
                // history). KEEP the half-res derivation in sync with
                // cloud_march.comp ((w+1)/2). Always allocated (small); the
                // march is only dispatched when clouds are enabled.
                const uint32_t hw = (w + 1u) / 2u, hh = (h + 1u) / 2u;
                // TRANSFER_DST: cleared to known contents at creation (below) —
                // a layout transition alone leaves texel contents UNDEFINED, and
                // random fp16 garbage can pass the march's history-validity
                // checks on the first clouds-enabled frame.
                g.cloudColor = createAttachmentImage2D(hw, hh, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                       VK_IMAGE_ASPECT_COLOR_BIT, N("cloudColor"));
                g.cloudAux = createAttachmentImage2D(hw, hh, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                             VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                     VK_IMAGE_ASPECT_COLOR_BIT, N("cloudAux"));
                // Cloud shadow map — FIXED 512² R8 (independent of the render
                // extent; recreated here on resize). Top-down cloud
                // transmittance regenerated per frame by cloud_shadow.comp.
                g.cloudShadow = createAttachmentImage2D(512, 512, VK_FORMAT_R8_UNORM,
                                                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                        VK_IMAGE_ASPECT_COLOR_BIT, N("cloudShadow"));
                // Gaussian-splat depth AOV. Splats are PRIMARY-view
                // only (SplatPass.hpp's SCOPE note), so a secondary view gets
                // the one-texel version even with the AOV on — same reasoning
                // as the MS attachments below, and the same shape: allocate
                // 1x1 rather than nothing, because binding 23 of the splat
                // descriptor set needs a real r32f image whether or not the
                // shader writes it, and a separate dummy with its own
                // lifetime is more moving parts than one texel.
                const bool wantSplatDepth = splatDepthAovAllocated() && !view().secondary;
                g.splatDepth = createAttachmentImage2D(
                        wantSplatDepth ? w : 1u, wantSplatDepth ? h : 1u,
                        VK_FORMAT_R32_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, N("splatDepth"));
                g.depth  = createAttachmentImage2D(w, h, VK_FORMAT_D32_SFLOAT,
                                                   depthUsage, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                   N("depth"));
                // Unjittered depth, written by the overlay depth prepass and
                // read by the post-TAA wireframe overlay's depth-test. Lives
                // outside the main G-buffer render pass — bound only via
                // dynamic rendering, so it's not part of the framebuffer.
                // Sized to the SWAPCHAIN extent, not the render extent: the
                // overlay composites onto the post-TAA full-resolution image
                // (TAA upscales when renderScale < 1), so its depth
                // attachment must match the swapchain, not the G-buffer.
                // ONLY in the 1-sample overlay path: with hardware MSAA the
                // prepass fills the multisampled overlayMsDepth_ instead (one
                // shared image, not one per frame in flight), so allocating
                // these would be pure waste — a swapchain-sized D32 per FIF.
                if (overlaySamples() <= 1) {
                    const VkExtent2D swapExt = viewOutExtent();
                    g.unjitDepth = createAttachmentImage2D(swapExt.width, swapExt.height,
                                                           VK_FORMAT_D32_SFLOAT,
                                                           depthUsage, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                           N("unjitDepth"));
                }

                VkImageView views[6] = {g.normal.view, g.motion.view, g.ids.view,
                                        g.uv.view, g.albedo.view, g.depth.view};
                VkFramebufferCreateInfo fci{};
                fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fci.renderPass      = rasterGbufRenderPass;
                fci.attachmentCount = 6;
                fci.pAttachments    = views;
                fci.width           = w;
                fci.height          = h;
                fci.layers          = 1;
                check(vkCreateFramebuffer(ctx->device(), &fci, nullptr, &g.framebuffer),
                      "vkCreateFramebuffer(rasterGbuf)");
                g.width  = w;
                g.height = h;

                // MSAA raster targets — only when opted in, and PRIMARY only:
                // gbufResolve_ is one shared instance whose descriptors name the
                // primary's images (VulkanCoreDescriptors gates its writes on
                // !secondary). Giving a secondary a framebufferMS routed its
                // raster through the MS pass with a resolve that read the
                // PRIMARY's samples — the secondary's resolved gbuf was never
                // written and its depth got the primary's. No framebufferMS →
                // the secondary rasters through the 1x pass (the useMsaa check
                // in buildIndirectDrawData keys on framebufferMS).
                if (gbufMsaaSamples_ > 1 && !view().secondary) {
                    const auto samples = gbufMsaaSamples_ == 4 ? VK_SAMPLE_COUNT_4_BIT
                                                                : VK_SAMPLE_COUNT_2_BIT;
                    // MS attachments need SAMPLED only (not STORAGE — compute
                    // reads them via sampler2DMS/texelFetch(sample), never
                    // imageLoad); STORAGE_BIT on a multisampled image needs
                    // the shaderStorageImageMultisample device feature, which
                    // isn't enabled/needed here — colorUsage (which adds
                    // STORAGE for the single-sample RESOLVE targets above) must
                    // NOT be reused for the MS images themselves.
                    const VkImageUsageFlags colorUsageMS =
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                    g.normalMS = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                         colorUsageMS, VK_IMAGE_ASPECT_COLOR_BIT,
                                                         N("normalMS"), samples);
                    g.motionMS = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                         colorUsageMS, VK_IMAGE_ASPECT_COLOR_BIT,
                                                         N("motionMS"), samples);
                    g.idsMS    = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_UINT,
                                                         colorUsageMS, VK_IMAGE_ASPECT_COLOR_BIT,
                                                         N("idsMS"), samples);
                    g.uvMS     = createAttachmentImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                         colorUsageMS, VK_IMAGE_ASPECT_COLOR_BIT,
                                                         N("uvMS"), samples);
                    g.albedoMS = createAttachmentImage2D(w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                                         colorUsageMS, VK_IMAGE_ASPECT_COLOR_BIT,
                                                         N("albedoMS"), samples);
                    g.depthMS  = createAttachmentImage2D(w, h, VK_FORMAT_D32_SFLOAT,
                                                         depthUsage, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                         N("depthMS"), samples);

                    VkImageView viewsMS[6] = {g.normalMS.view, g.motionMS.view, g.idsMS.view,
                                              g.uvMS.view, g.albedoMS.view, g.depthMS.view};
                    VkFramebufferCreateInfo fciMS{};
                    fciMS.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                    fciMS.renderPass      = rasterGbufRenderPassMS;
                    fciMS.attachmentCount = 6;
                    fciMS.pAttachments    = viewsMS;
                    fciMS.width           = w;
                    fciMS.height          = h;
                    fciMS.layers          = 1;
                    check(vkCreateFramebuffer(ctx->device(), &fciMS, nullptr, &g.framebufferMS),
                          "vkCreateFramebuffer(rasterGbufMS)");
                }
            }

            // Initialise every slot's attachments to their sampled-read layout.
            // The gbuffer render pass writes only the CURRENT frame's slot each
            // frame (its attachments use initialLayout = UNDEFINED, so it
            // ignores whatever we set here), but the temporal consumers sample a
            // slot that has not been rasterised yet on the first frame(s) it is
            // the "previous" slot — TaaResolve reads gbufIdsPrevTex for its
            // same-mesh reproject guard, and the deferred shade reads the
            // per-frame gbuffer. Without a defined starting layout that read
            // finds the image in UNDEFINED and trips VUID-vkCmdDraw-None-09600 /
            // -vkCmdDispatch-None-09600. Contents stay undefined, but the layout
            // is now valid; the temporal logic discards previous data on the
            // first frame regardless. Re-runs on resize (images are recreated).
            VkCommandBuffer initCb = beginOneShot();
            std::vector<VkImageMemoryBarrier> inits;
            inits.reserve(view().rasterGbufs.size() * (gbufMsaaSamples_ > 1 ? 18 : 12));
            auto pushInit = [&](VkImage image, VkImageAspectFlags aspect, VkImageLayout layout) {
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = layout;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = image;
                b.subresourceRange.aspectMask = aspect;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                b.srcAccessMask = 0;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                inits.push_back(b);
            };
            for (auto& g : view().rasterGbufs) {
                pushInit(g.normal.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.ids.image,    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.uv.image,     VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.albedo.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                pushInit(g.indirect.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.momentsSq.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.atrousA.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.atrousB.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.reflect.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.reflAux.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.shadowVis.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.directU.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.shadowAtrousA.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.shadowAtrousB.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.froxelScatter.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.froxelLut.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.cloudColor.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.cloudAux.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.cloudShadow.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w)
                pushInit(g.splatDepth.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);// storage (compute r/w) + transfer clear
                pushInit(g.depth.image,  VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
                // null-check, not just the msaa toggle: a SECONDARY view gets no
                // MS images (see the !secondary gate at creation above) — a
                // barrier on a VK_NULL_HANDLE image is a driver crash.
                if (gbufMsaaSamples_ > 1 && g.normalMS.image != VK_NULL_HANDLE) {
                    pushInit(g.normalMS.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    pushInit(g.motionMS.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    pushInit(g.idsMS.image,    VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    pushInit(g.uvMS.image,     VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    pushInit(g.albedoMS.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                    pushInit(g.depthMS.image,  VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
                }
            }
            vkCmdPipelineBarrier(initCb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 0, nullptr, 0, nullptr,
                                 static_cast<uint32_t>(inits.size()), inits.data());
            // Cloud temporal history: clear to KNOWN contents (layout init above
            // leaves texels undefined). Color = the compositing identity
            // (0,0,0,1); aux = 0 (histLen 0 = invalid history). Runs at creation
            // AND resize, so first-enable / post-resize frames never blend
            // garbage that happens to pass the march's validity checks.
            {
                VkImageSubresourceRange full{};
                full.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                full.levelCount = 1;
                full.layerCount = 1;
                VkClearColorValue identity{};
                identity.float32[3] = 1.0f;// (0,0,0,1) — no in-scatter, full T
                VkClearColorValue zero{};
                for (auto& g : view().rasterGbufs) {
                    vkCmdClearColorImage(initCb, g.cloudColor.image,
                                         VK_IMAGE_LAYOUT_GENERAL, &identity, 1, &full);
                    vkCmdClearColorImage(initCb, g.cloudAux.image,
                                         VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &full);
                }
                VkMemoryBarrier mb{};
                mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(initCb,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     0, 1, &mb, 0, nullptr, 0, nullptr);
            }
            endAndSubmitOneShot(initCb, "rasterGbuf init layouts");
        }

void VulkanRenderer::Impl::createRasterDsLayoutAndPool() {
            // binding 0: per-frame CameraUbo (vertex + fragment — gbuffer.frag
            //            reads cam.jitter for the alpha-hash cutout discard)
            // binding 1: motionMat[] storage (vertex; same VkBuffer as the
            //            deferred shade's binding 10)
            // binding 2: mats[] storage (fragment; for normal-map index + uvTransformNormal)
            // binding 3: albedoMaps[] bindless sampler array (fragment;
            //            same VkImage handles as the deferred shade's binding 8)
            // binding 4: DrawInfo[] storage (vertex; bindless vertex-pulling SSBO
            //            for the indirect-drawing gbuf pipeline). Re-pointed at
            //            this frame's drawInfoBuffers slot in
            //            recordRasterGbufPass before each indirect dispatch.
            VkDescriptorSetLayoutBinding bindings[5]{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
            bindings[2].binding         = 2;
            bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[3].binding         = 3;
            bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[3].descriptorCount = kMaxMaterialTextures;
            bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings[4].binding         = 4;
            bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[4].descriptorCount = 1;
            bindings[4].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

            // The LAYOUT is shared by every view — one set shape, created once.
            // The POOL and SETS below are per-view (see ViewContext), so this
            // function is called again for each view added and must not
            // recreate the layout underneath the existing pipelines.
            if (rasterDsLayout == VK_NULL_HANDLE) {
                VkDescriptorSetLayoutCreateInfo dlci{};
                dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                dlci.bindingCount = 5;
                dlci.pBindings    = bindings;
                check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &rasterDsLayout),
                      "vkCreateDescriptorSetLayout(raster)");
            }

            VkDescriptorPoolSize sizes[3]{};
            sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sizes[0].descriptorCount = kFramesInFlight;
            sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            sizes[1].descriptorCount = kFramesInFlight * 3;// motionMat + mats + drawInfo
            sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sizes[2].descriptorCount = kFramesInFlight * kMaxMaterialTextures;

            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets       = kFramesInFlight;
            dpci.poolSizeCount = 3;
            dpci.pPoolSizes    = sizes;
            check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &view().rasterDescPool),
                  "vkCreateDescriptorPool(raster)");

            std::array<VkDescriptorSetLayout, kFramesInFlight> layouts;
            layouts.fill(rasterDsLayout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = view().rasterDescPool;
            ai.descriptorSetCount = kFramesInFlight;
            ai.pSetLayouts        = layouts.data();
            check(vkAllocateDescriptorSets(ctx->device(), &ai, view().rasterDescSets.data()),
                  "vkAllocateDescriptorSets(raster)");
        }

void VulkanRenderer::Impl::createRasterGbufPipeline() {
            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kGbufferVertSpv);
            vsmci.pCode    = kGbufferVertSpv;
            VkShaderModule vertModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vertModule),
                  "vkCreateShaderModule(gbuffer.vert)");

            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kGbufferFragSpv);
            fsmci.pCode    = kGbufferFragSpv;
            VkShaderModule fragModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &fragModule),
                  "vkCreateShaderModule(gbuffer.frag)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertModule;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragModule;
            stages[1].pName  = "main";

            // Vertex input mirrors the BLAS allocation layout: positions /
            // normals / uvs / prev-positions packed (R32G32B32_SFLOAT for
            // pos+normal+prevPos, R32G32_SFLOAT for uv). The BLAS allocations
            // have VERTEX_BUFFER_BIT so they bind directly. Meshes without
            // a UV attribute bind dummyUvBuffer_ at binding 2; static meshes
            // bind their own vertex buffer at binding 3 (prev == curr).
            VkVertexInputBindingDescription vibs[4]{};
            vibs[0].binding   = 0;
            vibs[0].stride    = 3 * sizeof(float);
            vibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[1].binding   = 1;
            vibs[1].stride    = 3 * sizeof(float);
            vibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[2].binding   = 2;
            vibs[2].stride    = 2 * sizeof(float);
            vibs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[3].binding   = 3;
            vibs[3].stride    = 3 * sizeof(float);
            vibs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription vias[4]{};
            vias[0].location = 0;
            vias[0].binding  = 0;
            vias[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[0].offset   = 0;
            vias[1].location = 1;
            vias[1].binding  = 1;
            vias[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[1].offset   = 0;
            vias[2].location = 2;
            vias[2].binding  = 2;
            vias[2].format   = VK_FORMAT_R32G32_SFLOAT;
            vias[2].offset   = 0;
            vias[3].location = 3;
            vias[3].binding  = 3;
            vias[3].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[3].offset   = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 4;
            vi.pVertexBindingDescriptions      = vibs;
            vi.vertexAttributeDescriptionCount = 4;
            vi.pVertexAttributeDescriptions    = vias;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            // cullMode is set dynamically per-draw in recordRasterGbufPass
            // from the material's Side: BACK for Front (default fast path,
            // ~2× perf on dense meshes like ocean), FRONT for Back, NONE
            // for Double. The static value here is overridden by the dynamic
            // state, but Vulkan still wants a valid placeholder.
            rs.cullMode    = VK_CULL_MODE_BACK_BIT;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp   = VK_COMPARE_OP_GREATER;// reverse-Z (near→1, far→0)

            VkPipelineColorBlendAttachmentState cbas[5]{};
            for (auto& a : cbas) {
                a.blendEnable    = VK_FALSE;
                a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            }
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 5;
            cb.pAttachments    = cbas;

            // Dynamic viewport + scissor + cullMode (Vulkan 1.3 core via
            // extendedDynamicState). cullMode flips per-draw across BACK
            // (Side::Front, default fast path), FRONT (Side::Back), and
            // NONE (Side::Double).
            VkDynamicState dynStates[3] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR,
                                           VK_DYNAMIC_STATE_CULL_MODE};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 3;
            dyn.pDynamicStates    = dynStates;

            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            pcRange.offset     = 0;
            pcRange.size       = 80;// mat4 model + uvec4 (instId/flags/pad/pad)

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &rasterDsLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &rasterPipelineLayout),
                  "vkCreatePipelineLayout(raster)");

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = rasterPipelineLayout;
            gpci.renderPass          = rasterGbufRenderPass;
            gpci.subpass             = 0;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci, nullptr,
                                            &rasterGbufPipeline),
                  "vkCreateGraphicsPipelines(rasterGbuf)");

            vkDestroyShaderModule(ctx->device(), vertModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragModule, nullptr);

            // Indirect-drawing variant: same fragment shader + same render
            // pass, but uses gbuffer_indirect.vert with bindless vertex
            // pulling. No vertex input bindings — the VS reads positions /
            // normals / UVs via buffer-device-address dereferences keyed
            // by gl_InstanceIndex into the DrawInfo SSBO at binding 4.
            VkShaderModuleCreateInfo vciInd{};
            vciInd.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vciInd.codeSize = sizeof(kGbufferIndirectVertSpv);
            vciInd.pCode    = kGbufferIndirectVertSpv;
            VkShaderModule vertIndirectModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vciInd, nullptr, &vertIndirectModule),
                  "vkCreateShaderModule(gbuffer_indirect.vert)");
            VkShaderModuleCreateInfo fciInd{};
            fciInd.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fciInd.codeSize = sizeof(kGbufferFragSpv);
            fciInd.pCode    = kGbufferFragSpv;
            VkShaderModule fragIndModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fciInd, nullptr, &fragIndModule),
                  "vkCreateShaderModule(gbuffer.frag for indirect)");

            VkPipelineShaderStageCreateInfo stagesInd[2]{};
            stagesInd[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stagesInd[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stagesInd[0].module = vertIndirectModule;
            stagesInd[0].pName  = "main";
            stagesInd[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stagesInd[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stagesInd[1].module = fragIndModule;
            stagesInd[1].pName  = "main";

            // Empty vertex input state — VS doesn't bind anything, it reads
            // from buffer device addresses stored in the per-draw SSBO.
            VkPipelineVertexInputStateCreateInfo viInd{};
            viInd.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            viInd.vertexBindingDescriptionCount   = 0;
            viInd.vertexAttributeDescriptionCount = 0;

            VkGraphicsPipelineCreateInfo gpciInd = gpci;// reuse most state
            gpciInd.stageCount        = 2;
            gpciInd.pStages           = stagesInd;
            gpciInd.pVertexInputState = &viInd;

            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciInd, nullptr,
                                            &rasterGbufIndirectPipeline),
                  "vkCreateGraphicsPipelines(rasterGbufIndirect)");

            // Decal variant (same shaders/layout/render pass): blend-decal
            // meshes (MaterialDesc.alphaCutoff == -2) draw AFTER the opaque
            // buckets and only tint the receiver's albedo. Depth test stays on
            // (GREATER, reverse-Z — the per-draw polygonOffset clip-z bias
            // lifts the coplanar decal above its receiver) but depth write is
            // off, and every attachment except albedo is write-masked so the
            // pixel keeps the receiving surface's normal/ids/motion — the
            // deferred shade lights the receiver, with lerped base colour,
            // exactly like GL's forward alpha blend of a coplanar decal.
            VkPipelineDepthStencilStateCreateInfo dsDecal = ds;
            dsDecal.depthWriteEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState cbasDecal[5]{};
            for (int a = 0; a < 4; ++a) {
                cbasDecal[a].blendEnable    = VK_FALSE;
                cbasDecal[a].colorWriteMask = 0;// keep receiver's normal/motion/ids/uv
            }
            cbasDecal[4].blendEnable         = VK_TRUE;
            cbasDecal[4].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbasDecal[4].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasDecal[4].colorBlendOp        = VK_BLEND_OP_ADD;
            cbasDecal[4].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasDecal[4].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cbasDecal[4].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbasDecal[4].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                               VK_COLOR_COMPONENT_B_BIT;// .a = receiver metalness, keep
            VkPipelineColorBlendStateCreateInfo cbDecal = cb;
            cbDecal.pAttachments = cbasDecal;

            // DECAL_PASS=1 specialization flips gbuffer.frag from the stochastic
            // screen-door to "emit texture alpha as the blend factor". The
            // regular pipelines keep DECAL_PASS=0.
            const uint32_t kDecalPassOn = 1u;
            VkSpecializationMapEntry decalSpecEntry{0, 0, sizeof(uint32_t)};
            VkSpecializationInfo decalSpecInfo{1, &decalSpecEntry, sizeof(uint32_t), &kDecalPassOn};
            VkPipelineShaderStageCreateInfo stagesDecal[2] = {stagesInd[0], stagesInd[1]};
            stagesDecal[1].pSpecializationInfo = &decalSpecInfo;

            VkGraphicsPipelineCreateInfo gpciDecal = gpciInd;
            gpciDecal.pStages            = stagesDecal;
            gpciDecal.pDepthStencilState = &dsDecal;
            gpciDecal.pColorBlendState   = &cbDecal;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciDecal, nullptr,
                                            &rasterGbufDecalPipeline),
                  "vkCreateGraphicsPipelines(rasterGbufDecal)");

            // ParticleField variant: same render pass, layout, fragment stage
            // and depth/blend state as the indirect pipeline — only the vertex
            // stage differs. It must be its own pipeline because
            // gbuffer_indirect.vert's contract is one instance per draw, and a
            // field is one draw with instanceCount = the device-side liveCount.
            VkShaderModuleCreateInfo vciPf{};
            vciPf.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vciPf.codeSize = sizeof(kParticleFieldGbufVertSpv);
            vciPf.pCode    = kParticleFieldGbufVertSpv;
            VkShaderModule vertPfModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vciPf, nullptr, &vertPfModule),
                  "vkCreateShaderModule(particlefield_gbuf.vert)");
            VkPipelineShaderStageCreateInfo stagesPf[2] = {stagesInd[0], stagesInd[1]};
            stagesPf[0].module = vertPfModule;
            VkGraphicsPipelineCreateInfo gpciPf = gpciInd;
            gpciPf.pStages = stagesPf;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciPf, nullptr,
                                            &rasterGbufParticlePipeline),
                  "vkCreateGraphicsPipelines(rasterGbufParticle)");
            vkDestroyShaderModule(ctx->device(), vertPfModule, nullptr);

            vkDestroyShaderModule(ctx->device(), vertIndirectModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragIndModule, nullptr);
        }

void VulkanRenderer::Impl::createRasterGbufPipelineMS(VkSampleCountFlagBits samples) {
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
            if (rasterGbufParticlePipelineMS != VK_NULL_HANDLE) {
                vkDestroyPipeline(ctx->device(), rasterGbufParticlePipelineMS, nullptr);
                rasterGbufParticlePipelineMS = VK_NULL_HANDLE;
            }

            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kGbufferVertSpv);
            vsmci.pCode    = kGbufferVertSpv;
            VkShaderModule vertModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vertModule),
                  "vkCreateShaderModule(gbuffer.vert MS)");

            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kGbufferFragSpv);
            fsmci.pCode    = kGbufferFragSpv;
            VkShaderModule fragModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &fragModule),
                  "vkCreateShaderModule(gbuffer.frag MS)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertModule;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragModule;
            stages[1].pName  = "main";

            VkVertexInputBindingDescription vibs[4]{};
            vibs[0].binding   = 0;
            vibs[0].stride    = 3 * sizeof(float);
            vibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[1].binding   = 1;
            vibs[1].stride    = 3 * sizeof(float);
            vibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[2].binding   = 2;
            vibs[2].stride    = 2 * sizeof(float);
            vibs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[3].binding   = 3;
            vibs[3].stride    = 3 * sizeof(float);
            vibs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription vias[4]{};
            vias[0].location = 0;
            vias[0].binding  = 0;
            vias[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[0].offset   = 0;
            vias[1].location = 1;
            vias[1].binding  = 1;
            vias[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[1].offset   = 0;
            vias[2].location = 2;
            vias[2].binding  = 2;
            vias[2].format   = VK_FORMAT_R32G32_SFLOAT;
            vias[2].offset   = 0;
            vias[3].location = 3;
            vias[3].binding  = 3;
            vias[3].format   = VK_FORMAT_R32G32B32_SFLOAT;
            vias[3].offset   = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 4;
            vi.pVertexBindingDescriptions      = vibs;
            vi.vertexAttributeDescriptionCount = 4;
            vi.pVertexAttributeDescriptions    = vias;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode    = VK_CULL_MODE_BACK_BIT;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = samples;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp   = VK_COMPARE_OP_GREATER;// reverse-Z (near→1, far→0)

            VkPipelineColorBlendAttachmentState cbas[5]{};
            for (auto& a : cbas) {
                a.blendEnable    = VK_FALSE;
                a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            }
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 5;
            cb.pAttachments    = cbas;

            VkDynamicState dynStates[3] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR,
                                           VK_DYNAMIC_STATE_CULL_MODE};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 3;
            dyn.pDynamicStates    = dynStates;

            // Reuses rasterPipelineLayout (same set layout + push constant
            // range as the 1× pipelines) — created unconditionally by
            // createRasterGbufPipeline() before this ever runs.
            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = rasterPipelineLayout;
            gpci.renderPass          = rasterGbufRenderPassMS;
            gpci.subpass             = 0;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci, nullptr,
                                            &rasterGbufPipelineMS),
                  "vkCreateGraphicsPipelines(rasterGbufMS)");

            vkDestroyShaderModule(ctx->device(), vertModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragModule, nullptr);

            VkShaderModuleCreateInfo vciInd{};
            vciInd.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vciInd.codeSize = sizeof(kGbufferIndirectVertSpv);
            vciInd.pCode    = kGbufferIndirectVertSpv;
            VkShaderModule vertIndirectModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vciInd, nullptr, &vertIndirectModule),
                  "vkCreateShaderModule(gbuffer_indirect.vert MS)");
            VkShaderModuleCreateInfo fciInd{};
            fciInd.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fciInd.codeSize = sizeof(kGbufferFragSpv);
            fciInd.pCode    = kGbufferFragSpv;
            VkShaderModule fragIndModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fciInd, nullptr, &fragIndModule),
                  "vkCreateShaderModule(gbuffer.frag for indirect MS)");

            VkPipelineShaderStageCreateInfo stagesInd[2]{};
            stagesInd[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stagesInd[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stagesInd[0].module = vertIndirectModule;
            stagesInd[0].pName  = "main";
            stagesInd[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stagesInd[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stagesInd[1].module = fragIndModule;
            stagesInd[1].pName  = "main";

            VkPipelineVertexInputStateCreateInfo viInd{};
            viInd.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            viInd.vertexBindingDescriptionCount   = 0;
            viInd.vertexAttributeDescriptionCount = 0;

            VkGraphicsPipelineCreateInfo gpciInd = gpci;
            gpciInd.stageCount        = 2;
            gpciInd.pStages           = stagesInd;
            gpciInd.pVertexInputState = &viInd;

            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciInd, nullptr,
                                            &rasterGbufIndirectPipelineMS),
                  "vkCreateGraphicsPipelines(rasterGbufIndirectMS)");

            VkPipelineDepthStencilStateCreateInfo dsDecal = ds;
            dsDecal.depthWriteEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState cbasDecal[5]{};
            for (int a = 0; a < 4; ++a) {
                cbasDecal[a].blendEnable    = VK_FALSE;
                cbasDecal[a].colorWriteMask = 0;
            }
            cbasDecal[4].blendEnable         = VK_TRUE;
            cbasDecal[4].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbasDecal[4].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasDecal[4].colorBlendOp        = VK_BLEND_OP_ADD;
            cbasDecal[4].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasDecal[4].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cbasDecal[4].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbasDecal[4].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                               VK_COLOR_COMPONENT_B_BIT;
            VkPipelineColorBlendStateCreateInfo cbDecal = cb;
            cbDecal.pAttachments = cbasDecal;

            const uint32_t kDecalPassOn = 1u;
            VkSpecializationMapEntry decalSpecEntry{0, 0, sizeof(uint32_t)};
            VkSpecializationInfo decalSpecInfo{1, &decalSpecEntry, sizeof(uint32_t), &kDecalPassOn};
            VkPipelineShaderStageCreateInfo stagesDecal[2] = {stagesInd[0], stagesInd[1]};
            stagesDecal[1].pSpecializationInfo = &decalSpecInfo;

            VkGraphicsPipelineCreateInfo gpciDecal = gpciInd;
            gpciDecal.pStages            = stagesDecal;
            gpciDecal.pDepthStencilState = &dsDecal;
            gpciDecal.pColorBlendState   = &cbDecal;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciDecal, nullptr,
                                            &rasterGbufDecalPipelineMS),
                  "vkCreateGraphicsPipelines(rasterGbufDecalMS)");

            // ParticleField, MSAA. Same reasoning as the 1x path above.
            VkShaderModuleCreateInfo vciPf{};
            vciPf.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vciPf.codeSize = sizeof(kParticleFieldGbufVertSpv);
            vciPf.pCode    = kParticleFieldGbufVertSpv;
            VkShaderModule vertPfModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vciPf, nullptr, &vertPfModule),
                  "vkCreateShaderModule(particlefield_gbuf.vert MS)");
            VkPipelineShaderStageCreateInfo stagesPf[2] = {stagesInd[0], stagesInd[1]};
            stagesPf[0].module = vertPfModule;
            VkGraphicsPipelineCreateInfo gpciPf = gpciInd;
            gpciPf.pStages = stagesPf;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciPf, nullptr,
                                            &rasterGbufParticlePipelineMS),
                  "vkCreateGraphicsPipelines(rasterGbufParticleMS)");
            vkDestroyShaderModule(ctx->device(), vertPfModule, nullptr);

            vkDestroyShaderModule(ctx->device(), vertIndirectModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragIndModule, nullptr);
        }

void VulkanRenderer::Impl::createOverlayPipeline() {
            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kOverlayVertSpv);
            vsmci.pCode    = kOverlayVertSpv;
            VkShaderModule vertModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vertModule),
                  "vkCreateShaderModule(overlay.vert)");

            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kOverlayFragSpv);
            fsmci.pCode    = kOverlayFragSpv;
            VkShaderModule fragModule = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &fragModule),
                  "vkCreateShaderModule(overlay.frag)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertModule;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragModule;
            stages[1].pName  = "main";

            // Single vertex binding: position, R32G32B32_SFLOAT. Reuses the
            // BLAS's vertex buffer directly (skinned meshes get the deformed
            // current-pose positions because refreshSkinnedBlas already wrote
            // them at the start of the frame).
            VkVertexInputBindingDescription vib{};
            vib.binding   = 0;
            vib.stride    = 3 * sizeof(float);
            vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription via{};
            via.location = 0;
            via.binding  = 0;
            via.format   = VK_FORMAT_R32G32B32_SFLOAT;
            via.offset   = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 1;
            vi.pVertexBindingDescriptions      = &vib;
            vi.vertexAttributeDescriptionCount = 1;
            vi.pVertexAttributeDescriptions    = &via;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            // POLYGON_MODE_LINE renders each tri as 3 lines — that's the
            // wireframe effect. cullMode NONE so back-facing geometry's
            // edges are visible too (helps see structure on closed meshes).
            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_LINE;
            rs.cullMode    = VK_CULL_MODE_NONE;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            // Hardware MSAA: the whole overlay pass rasterizes at
            // overlaySamples() and resolves onto the swapchain at
            // vkCmdEndRendering (see the overlayMs* member docs). Falls back
            // to 1 sample — the original direct-to-swapchain path — when
            // Canvas antialiasing is off or the device can't do it.
            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = overlaySampleBits();

            // Depth test re-enabled. Compares against rasterGbufs[f].unjitDepth
            // (filled by the overlay_depth prepass with the SAME unjittered
            // projection overlay.vert uses), so per-pixel z values match and
            // the depth test is stable across frames (no sub-pixel-jitter
            // shimmer that the jittered G-buffer depth caused in the earlier
            // depth-test-on configuration). depthWrite stays OFF — overlay
            // doesn't mutate the depth attachment so subsequent reads of it
            // (next frame's prepass clears + writes again) are well-defined.
            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;// reverse-Z (overlay vs unjitDepth)
            // ...but DEPTH_TEST_ENABLE is dynamic (core in 1.3, same promoted
            // set as the vkCmdSetCullMode already used here) so the draw loop
            // can honour Material::depthTest. GL disables GL_DEPTH_TEST for
            // such materials, which is what makes TransformControls gizmos and
            // similar helpers draw through solid geometry; without this the
            // overlay pass silently depth-rejected them.

            // Single color attachment: the MSAA overlay target (resolved onto
            // the swapchain), or the swapchain itself at 1 sample.
            VkPipelineColorBlendAttachmentState cbas[1]{};
            cbas[0].blendEnable    = VK_FALSE;
            cbas[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments    = cbas;

            VkDynamicState dynStates[3] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR,
                                           VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 3;
            dyn.pDynamicStates    = dynStates;

            // mat4 mvp (64) + vec4 color (16) + vec4 params (16) = 96 bytes,
            // well under the 128B push-constant guarantee. Both vertex and
            // fragment read the same block. Only overlay_point.vert declares
            // the params tail (its .x is the point-size attenuation scale);
            // the line and wireframe shaders sharing this layout stop at 80
            // and push 80, which is a legal sub-range.
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pcRange.offset     = 0;
            pcRange.size       = 96;

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 0;
            plci.pSetLayouts            = nullptr;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &overlayPipelineLayout),
                  "vkCreatePipelineLayout(overlay)");

            // Dynamic rendering: declare formats up-front via
            // VkPipelineRenderingCreateInfo. Color 0 = swapchain format (the
            // MSAA target shares it, and a resolve requires matching formats),
            // depth = D32_SFLOAT (matches overlayMsDepth_ / unjitDepth, which
            // the overlay pass binds as a read-only depth attachment).
            const VkFormat overlayFmts[1] = {ctx->swapchainFormat()};
            VkPipelineRenderingCreateInfo prci{};
            prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            prci.colorAttachmentCount    = 1;
            prci.pColorAttachmentFormats = overlayFmts;
            prci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.pNext               = &prci;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = overlayPipelineLayout;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci, nullptr,
                                            &overlayWireframePipeline),
                  "vkCreateGraphicsPipelines(overlayWireframe)");

            // Solid-fill variant for MeshBasicMaterial-style overlays. Same
            // shaders + state otherwise; only the rasterization mode flips
            // to FILL. Cull mode is DYNAMIC so the draw loop can honour
            // material.side — SVG/UI meshes are typically Side::Double and
            // often mirrored (negative scale flips winding), which a static
            // back-face cull silently deletes. Reuses the just-created
            // shader modules → cheap second pipeline.
            VkPipelineRasterizationStateCreateInfo rsBasic = rs;
            rsBasic.polygonMode = VK_POLYGON_MODE_FILL;
            rsBasic.cullMode    = VK_CULL_MODE_BACK_BIT;// overridden per draw (dynamic)
            VkDynamicState dynFillStates[4] = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR,
                                               VK_DYNAMIC_STATE_CULL_MODE,
                                               VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE};
            VkPipelineDynamicStateCreateInfo dynFill{};
            dynFill.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynFill.dynamicStateCount = 4;
            dynFill.pDynamicStates    = dynFillStates;
            VkGraphicsPipelineCreateInfo gpciBasic = gpci;
            gpciBasic.pRasterizationState = &rsBasic;
            gpciBasic.pDynamicState       = &dynFill;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciBasic, nullptr,
                                            &overlayBasicPipeline),
                  "vkCreateGraphicsPipelines(overlayBasic)");

            // Alpha-blended fill variant. Standard "non-premultiplied" alpha:
            //   srcColor·srcAlpha + dstColor·(1-srcAlpha)
            // Depth-test defaults on (occluded by scene geometry) but is
            // dynamic per material.depthTest; depth-write OFF
            // so back-to-front order doesn't matter for the depth attachment
            // even if multiple transparent overlays overlap. Per-overlay
            // depth sorting is NOT performed — overlapping transparent
            // overlays may show out-of-order alpha. For typical gizmo /
            // single-transparent-mesh use this is acceptable; documented as
            // a Stage-2 limitation.
            VkPipelineColorBlendAttachmentState cbasBlend[1]{};
            cbasBlend[0].blendEnable         = VK_TRUE;
            cbasBlend[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbasBlend[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasBlend[0].colorBlendOp        = VK_BLEND_OP_ADD;
            cbasBlend[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasBlend[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasBlend[0].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbasBlend[0].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cbBlend{};
            cbBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cbBlend.attachmentCount = 1;
            cbBlend.pAttachments    = cbasBlend;
            VkGraphicsPipelineCreateInfo gpciBasicTr = gpciBasic;
            gpciBasicTr.pColorBlendState = &cbBlend;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciBasicTr, nullptr,
                                            &overlayBasicTransparentPipeline),
                  "vkCreateGraphicsPipelines(overlayBasicTransparent)");

            // Line / LineSegments pipelines. Same overlay shaders, same
            // depth/blend state as the basic opaque variant; only the
            // input-assembly topology and rasterization mode differ.
            // POLYGON_MODE_FILL is irrelevant for line topologies but kept
            // for pipeline validity. cullMode=NONE (lines don't have
            // facing).
            VkPipelineRasterizationStateCreateInfo rsLine = rs;
            rsLine.polygonMode = VK_POLYGON_MODE_FILL;
            rsLine.cullMode    = VK_CULL_MODE_NONE;

            VkPipelineInputAssemblyStateCreateInfo iaLineList{};
            iaLineList.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            iaLineList.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            VkGraphicsPipelineCreateInfo gpciLineList = gpci;
            gpciLineList.pInputAssemblyState = &iaLineList;
            gpciLineList.pRasterizationState = &rsLine;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciLineList, nullptr,
                                            &overlayLineListPipeline),
                  "vkCreateGraphicsPipelines(overlayLineList)");

            VkPipelineInputAssemblyStateCreateInfo iaLineStrip{};
            iaLineStrip.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            iaLineStrip.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            VkGraphicsPipelineCreateInfo gpciLineStrip = gpci;
            gpciLineStrip.pInputAssemblyState = &iaLineStrip;
            gpciLineStrip.pRasterizationState = &rsLine;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciLineStrip, nullptr,
                                            &overlayLineStripPipeline),
                  "vkCreateGraphicsPipelines(overlayLineStrip)");

            // ── Colored line pipelines ──────────────────────────────────────
            // Different shader pair (overlay_color.vert/.frag) + 2 vertex
            // bindings: position at 0, color at 1.
            VkShaderModuleCreateInfo cvsmci{};
            cvsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            cvsmci.codeSize = sizeof(kOverlayColorVertSpv);
            cvsmci.pCode    = kOverlayColorVertSpv;
            VkShaderModule cvert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &cvsmci, nullptr, &cvert),
                  "vkCreateShaderModule(overlay_color.vert)");

            VkShaderModuleCreateInfo cfsmci{};
            cfsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            cfsmci.codeSize = sizeof(kOverlayColorFragSpv);
            cfsmci.pCode    = kOverlayColorFragSpv;
            VkShaderModule cfrag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &cfsmci, nullptr, &cfrag),
                  "vkCreateShaderModule(overlay_color.frag)");

            VkPipelineShaderStageCreateInfo cStages[2]{};
            cStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            cStages[0].module = cvert;
            cStages[0].pName  = "main";
            cStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            cStages[1].module = cfrag;
            cStages[1].pName  = "main";

            VkVertexInputBindingDescription cvibs[2]{};
            cvibs[0].binding   = 0;
            cvibs[0].stride    = 3 * sizeof(float);
            cvibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            cvibs[1].binding   = 1;
            cvibs[1].stride    = 3 * sizeof(float);
            cvibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            VkVertexInputAttributeDescription cvias[2]{};
            cvias[0].location = 0;
            cvias[0].binding  = 0;
            cvias[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
            cvias[0].offset   = 0;
            cvias[1].location = 1;
            cvias[1].binding  = 1;
            cvias[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
            cvias[1].offset   = 0;
            VkPipelineVertexInputStateCreateInfo cvi{};
            cvi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            cvi.vertexBindingDescriptionCount   = 2;
            cvi.pVertexBindingDescriptions      = cvibs;
            cvi.vertexAttributeDescriptionCount = 2;
            cvi.pVertexAttributeDescriptions    = cvias;

            VkGraphicsPipelineCreateInfo gpciLineListColored = gpciLineList;
            gpciLineListColored.stageCount        = 2;
            gpciLineListColored.pStages           = cStages;
            gpciLineListColored.pVertexInputState = &cvi;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciLineListColored, nullptr,
                                            &overlayLineListColoredPipeline),
                  "vkCreateGraphicsPipelines(overlayLineListColored)");

            VkGraphicsPipelineCreateInfo gpciLineStripColored = gpciLineStrip;
            gpciLineStripColored.stageCount        = 2;
            gpciLineStripColored.pStages           = cStages;
            gpciLineStripColored.pVertexInputState = &cvi;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciLineStripColored, nullptr,
                                            &overlayLineStripColoredPipeline),
                  "vkCreateGraphicsPipelines(overlayLineStripColored)");

            // ── Blended line variants ───────────────────────────────────────
            // The four line pipelines above inherit the wireframe pipeline's
            // blend-OFF attachment state, so Material::transparent / opacity /
            // blending never reached the hardware for Line / LineSegments.
            // Additive matches GLState: (SRC_ALPHA, ONE) on color and alpha.
            VkPipelineColorBlendAttachmentState cbasAdd[1] = {cbasBlend[0]};
            cbasAdd[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasAdd[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbasAdd[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            VkPipelineColorBlendStateCreateInfo cbAdd = cbBlend;
            cbAdd.pAttachments = cbasAdd;

            const VkGraphicsPipelineCreateInfo* lineBases[2][2] = {
                    {&gpciLineList, &gpciLineStrip},
                    {&gpciLineListColored, &gpciLineStripColored}};
            const VkPipelineColorBlendStateCreateInfo* lineBlends[2] = {&cbBlend, &cbAdd};
            for (int colored = 0; colored < 2; ++colored)
                for (int strip = 0; strip < 2; ++strip)
                    for (int mode = 0; mode < 2; ++mode) {
                        VkGraphicsPipelineCreateInfo g = *lineBases[colored][strip];
                        g.pColorBlendState            = lineBlends[mode];
                        check(vkCreateGraphicsPipelines(
                                      ctx->device(), ctx->pipelineCache(), 1, &g, nullptr,
                                      &overlayLineBlendPipelines[colored][strip][mode]),
                              "vkCreateGraphicsPipelines(overlayLineBlend)");
                    }

            // ── Vertex-coloured mesh fill ───────────────────────────────────
            // Same overlay_color shader pair and 2-binding input as the
            // colored lines, but triangle-list fill with the basic pipeline's
            // dynamic cull/depth state. Gives the kSnapUiBlend route a path
            // for vertexColors materials (snapMeshFlags used to keep them in
            // the G-buffer, where they rendered opaque). [0]=opaque,
            // [1]=alpha, [2]=additive.
            const VkPipelineColorBlendStateCreateInfo* meshBlends[3] = {&cb, &cbBlend, &cbAdd};
            for (int mode = 0; mode < 3; ++mode) {
                VkGraphicsPipelineCreateInfo g = gpciBasic;
                g.stageCount        = 2;
                g.pStages           = cStages;
                g.pVertexInputState = &cvi;
                g.pColorBlendState  = meshBlends[mode];
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &g, nullptr,
                                                &overlayMeshColoredPipelines[mode]),
                      "vkCreateGraphicsPipelines(overlayMeshColored)");
            }

            // Flat additive fill — Blending::Additive on an untextured basic
            // previously fell back to plain alpha blending.
            {
                VkGraphicsPipelineCreateInfo g = gpciBasic;
                g.pColorBlendState = &cbAdd;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &g, nullptr,
                                                &overlayBasicAdditivePipeline),
                      "vkCreateGraphicsPipelines(overlayBasicAdditive)");
            }

            // Vertex-coloured wireframe — the flat wireframe create-info
            // (POLYGON_MODE_LINE, static cull NONE, blend off, no dynamic
            // cull) with the overlay_color shader pair and 2-binding input.
            {
                VkGraphicsPipelineCreateInfo g = gpci;
                g.pStages           = cStages;
                g.pVertexInputState = &cvi;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &g, nullptr,
                                                &overlayWireframeColoredPipeline),
                      "vkCreateGraphicsPipelines(overlayWireframeColored)");
            }

            // ── Point list pipeline ─────────────────────────────────────────
            // POINT_LIST topology. Uses overlay_point.vert/.frag which write
            // gl_PointSize from the push constant's color.w slot and discard
            // fragments outside a unit-radius disk (round sprite). Shares the
            // same overlayPipelineLayout + 2-binding vertex input (pos + color)
            // as the colored line variants — a Points object without a "color"
            // attribute is skipped at draw-record time.
            VkShaderModuleCreateInfo pvsmci{};
            pvsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            pvsmci.codeSize = sizeof(kOverlayPointVertSpv);
            pvsmci.pCode    = kOverlayPointVertSpv;
            VkShaderModule pvert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &pvsmci, nullptr, &pvert),
                  "vkCreateShaderModule(overlay_point.vert)");

            VkShaderModuleCreateInfo pfsmci{};
            pfsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            pfsmci.codeSize = sizeof(kOverlayPointFragSpv);
            pfsmci.pCode    = kOverlayPointFragSpv;
            VkShaderModule pfrag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &pfsmci, nullptr, &pfrag),
                  "vkCreateShaderModule(overlay_point.frag)");

            VkPipelineShaderStageCreateInfo pStages[2]{};
            pStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            pStages[0].module = pvert;
            pStages[0].pName  = "main";
            pStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            pStages[1].module = pfrag;
            pStages[1].pName  = "main";

            VkPipelineInputAssemblyStateCreateInfo iaPointList{};
            iaPointList.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            iaPointList.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

            VkGraphicsPipelineCreateInfo gpciPointList = gpciLineList;
            gpciPointList.stageCount        = 2;
            gpciPointList.pStages           = pStages;
            gpciPointList.pVertexInputState = &cvi;
            gpciPointList.pInputAssemblyState = &iaPointList;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciPointList, nullptr,
                                            &overlayPointListPipeline),
                  "vkCreateGraphicsPipelines(overlayPointList)");

            vkDestroyShaderModule(ctx->device(), vertModule, nullptr);
            vkDestroyShaderModule(ctx->device(), fragModule, nullptr);
            vkDestroyShaderModule(ctx->device(), cvert, nullptr);
            vkDestroyShaderModule(ctx->device(), cfrag, nullptr);
            vkDestroyShaderModule(ctx->device(), pvert, nullptr);
            vkDestroyShaderModule(ctx->device(), pfrag, nullptr);

            // ── Overlay depth prepass pipeline ──────────────────────────────
            // Renders all non-overlay scene geometry with the unjittered VP
            // into rasterGbufs[f].unjitDepth. Reuses rasterPipelineLayout
            // (same camera UBO + push constant). Position-only vertex input;
            // no color attachments. depthCompareOp = LESS so the closest
            // surface wins per pixel, the same way the main G-buffer fills
            // depth. Runs each frame after recordRasterGbufPass.
            {
                VkShaderModuleCreateInfo dvsmci{};
                dvsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                dvsmci.codeSize = sizeof(kOverlayDepthVertSpv);
                dvsmci.pCode    = kOverlayDepthVertSpv;
                VkShaderModule dvert = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &dvsmci, nullptr, &dvert),
                      "vkCreateShaderModule(overlay_depth.vert)");

                VkShaderModuleCreateInfo dfsmci{};
                dfsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                dfsmci.codeSize = sizeof(kOverlayDepthFragSpv);
                dfsmci.pCode    = kOverlayDepthFragSpv;
                VkShaderModule dfrag = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &dfsmci, nullptr, &dfrag),
                      "vkCreateShaderModule(overlay_depth.frag)");

                VkPipelineShaderStageCreateInfo dStages[2]{};
                dStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                dStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
                dStages[0].module = dvert;
                dStages[0].pName  = "main";
                dStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                dStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
                dStages[1].module = dfrag;
                dStages[1].pName  = "main";

                // Position-only vertex input. Note: the raster pipeline also
                // declares 4 vertex bindings (pos/normal/uv/prevPos) — the
                // depth prepass shares rasterPipelineLayout but its pipeline
                // declares only 1 binding here so we don't need to bind 4
                // buffers per draw. Vulkan permits more pipeline-declared
                // bindings than the shader uses; what's important is the
                // SHADER consumes only the bindings it has.
                VkVertexInputBindingDescription dvib{};
                dvib.binding   = 0;
                dvib.stride    = 3 * sizeof(float);
                dvib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                VkVertexInputAttributeDescription dvia{};
                dvia.location = 0;
                dvia.binding  = 0;
                dvia.format   = VK_FORMAT_R32G32B32_SFLOAT;
                dvia.offset   = 0;
                VkPipelineVertexInputStateCreateInfo dvi{};
                dvi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                dvi.vertexBindingDescriptionCount   = 1;
                dvi.pVertexBindingDescriptions      = &dvib;
                dvi.vertexAttributeDescriptionCount = 1;
                dvi.pVertexAttributeDescriptions    = &dvia;

                VkPipelineInputAssemblyStateCreateInfo dia{};
                dia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                dia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

                VkPipelineViewportStateCreateInfo dvp{};
                dvp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                dvp.viewportCount = 1;
                dvp.scissorCount  = 1;

                VkPipelineRasterizationStateCreateInfo drs{};
                drs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                drs.polygonMode = VK_POLYGON_MODE_FILL;
                drs.cullMode    = VK_CULL_MODE_BACK_BIT;
                drs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                drs.lineWidth   = 1.0f;

                // Matches the overlay pass it feeds: at overlaySamples() > 1
                // the prepass fills the multisampled overlayMsDepth_ (so the
                // overlay's per-sample depth test is per-sample correct);
                // at 1 sample it fills rasterGbufs[f].unjitDepth as before.
                VkPipelineMultisampleStateCreateInfo dms{};
                dms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                dms.rasterizationSamples = overlaySampleBits();

                VkPipelineDepthStencilStateCreateInfo dds{};
                dds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                dds.depthTestEnable  = VK_TRUE;
                dds.depthWriteEnable = VK_TRUE;
                dds.depthCompareOp   = VK_COMPARE_OP_GREATER;// reverse-Z (overlay depth prepass)

                // No color attachments — pColorBlendState attachmentCount=0.
                VkPipelineColorBlendStateCreateInfo dcb{};
                dcb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                dcb.attachmentCount = 0;

                VkDynamicState ddyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
                VkPipelineDynamicStateCreateInfo ddyn{};
                ddyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                ddyn.dynamicStateCount = 2;
                ddyn.pDynamicStates    = ddyns;

                VkPipelineRenderingCreateInfo dprci{};
                dprci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
                dprci.colorAttachmentCount  = 0;
                dprci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

                VkGraphicsPipelineCreateInfo dgpci{};
                dgpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                dgpci.pNext               = &dprci;
                dgpci.stageCount          = 2;
                dgpci.pStages             = dStages;
                dgpci.pVertexInputState   = &dvi;
                dgpci.pInputAssemblyState = &dia;
                dgpci.pViewportState      = &dvp;
                dgpci.pRasterizationState = &drs;
                dgpci.pMultisampleState   = &dms;
                dgpci.pDepthStencilState  = &dds;
                dgpci.pColorBlendState    = &dcb;
                dgpci.pDynamicState       = &ddyn;
                dgpci.layout              = rasterPipelineLayout;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &dgpci, nullptr,
                                                &overlayDepthPrepassPipeline),
                      "vkCreateGraphicsPipelines(overlayDepthPrepass)");

                vkDestroyShaderModule(ctx->device(), dvert, nullptr);
                vkDestroyShaderModule(ctx->device(), dfrag, nullptr);
            }

            // ── Overlay scene-inject pipeline ───────────────────────────────
            // FIRST draw of the MSAA overlay pass: a fullscreen triangle that
            // seeds every sample of the multisampled color target with the
            // already-composited scene (a 1-sample copy of the swapchain), so
            // every subsequent overlay blend composites against the true
            // background and untouched pixels resolve back unchanged.
            // See the overlayMs*/overlayInject* member docs. Not built (and
            // never bound) when overlaySamples() == 1.
            if (overlaySamples() > 1) {
                VkDescriptorSetLayoutBinding aaBind{};
                aaBind.binding         = 0;
                aaBind.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                aaBind.descriptorCount = 1;
                aaBind.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
                VkDescriptorSetLayoutCreateInfo aaDslci{};
                aaDslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                aaDslci.bindingCount = 1;
                aaDslci.pBindings    = &aaBind;
                check(vkCreateDescriptorSetLayout(ctx->device(), &aaDslci, nullptr, &overlayInjectSetLayout_),
                      "vkCreateDescriptorSetLayout(overlayInject)");

                VkDescriptorPoolSize aaPs{};
                aaPs.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                aaPs.descriptorCount = 1;
                VkDescriptorPoolCreateInfo aaDpci{};
                aaDpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                aaDpci.maxSets       = 1;
                aaDpci.poolSizeCount = 1;
                aaDpci.pPoolSizes    = &aaPs;
                check(vkCreateDescriptorPool(ctx->device(), &aaDpci, nullptr, &overlayInjectPool_),
                      "vkCreateDescriptorPool(overlayInject)");
                VkDescriptorSetAllocateInfo aaDsai{};
                aaDsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                aaDsai.descriptorPool     = overlayInjectPool_;
                aaDsai.descriptorSetCount = 1;
                aaDsai.pSetLayouts        = &overlayInjectSetLayout_;
                check(vkAllocateDescriptorSets(ctx->device(), &aaDsai, &overlayInjectSet_),
                      "vkAllocateDescriptorSets(overlayInject)");

                VkPipelineLayoutCreateInfo aaPlci{};
                aaPlci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                aaPlci.setLayoutCount = 1;
                aaPlci.pSetLayouts    = &overlayInjectSetLayout_;
                check(vkCreatePipelineLayout(ctx->device(), &aaPlci, nullptr, &overlayInjectPipelineLayout_),
                      "vkCreatePipelineLayout(overlayInject)");

                VkShaderModuleCreateInfo avsmci{};
                avsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                avsmci.codeSize = sizeof(kOverlayFullscreenVertSpv);
                avsmci.pCode    = kOverlayFullscreenVertSpv;
                VkShaderModule avert = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &avsmci, nullptr, &avert),
                      "vkCreateShaderModule(overlay_fullscreen.vert)");
                VkShaderModuleCreateInfo afsmci{};
                afsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                afsmci.codeSize = sizeof(kOverlayInjectFragSpv);
                afsmci.pCode    = kOverlayInjectFragSpv;
                VkShaderModule afrag = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &afsmci, nullptr, &afrag),
                      "vkCreateShaderModule(overlay_inject.frag)");

                VkPipelineShaderStageCreateInfo aStages[2]{};
                aStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                aStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
                aStages[0].module = avert;
                aStages[0].pName  = "main";
                aStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                aStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
                aStages[1].module = afrag;
                aStages[1].pName  = "main";

                VkPipelineVertexInputStateCreateInfo avi{};
                avi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                VkPipelineInputAssemblyStateCreateInfo aia{};
                aia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                aia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                VkPipelineViewportStateCreateInfo avp{};
                avp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                avp.viewportCount = 1;
                avp.scissorCount  = 1;
                VkPipelineRasterizationStateCreateInfo ars{};
                ars.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                ars.polygonMode = VK_POLYGON_MODE_FILL;
                ars.cullMode    = VK_CULL_MODE_NONE;
                ars.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                ars.lineWidth   = 1.0f;
                // Same sample count as the pass it draws into.
                VkPipelineMultisampleStateCreateInfo ams{};
                ams.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                ams.rasterizationSamples = overlaySampleBits();
                // Depth test AND write off — the inject covers the whole frame
                // unconditionally. Static (not dynamic like the overlay draws)
                // so binding it doesn't disturb the dynamic depth-test-enable
                // the draw loop re-sets for every subsequent draw anyway.
                VkPipelineDepthStencilStateCreateInfo ads{};
                ads.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                ads.depthTestEnable  = VK_FALSE;
                ads.depthWriteEnable = VK_FALSE;
                VkPipelineColorBlendAttachmentState acbas{};
                acbas.blendEnable    = VK_FALSE;
                acbas.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                VkPipelineColorBlendStateCreateInfo acb{};
                acb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                acb.attachmentCount = 1;
                acb.pAttachments    = &acbas;
                VkDynamicState adyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
                VkPipelineDynamicStateCreateInfo adyn{};
                adyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                adyn.dynamicStateCount = 2;
                adyn.pDynamicStates    = adyns;

                // Must match the overlay pass instance exactly, depth
                // attachment included (the inject shares that render pass).
                const VkFormat aaColorFmt = ctx->swapchainFormat();
                VkPipelineRenderingCreateInfo aprci{};
                aprci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
                aprci.colorAttachmentCount    = 1;
                aprci.pColorAttachmentFormats = &aaColorFmt;
                aprci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

                VkGraphicsPipelineCreateInfo agpci{};
                agpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                agpci.pNext               = &aprci;
                agpci.stageCount          = 2;
                agpci.pStages             = aStages;
                agpci.pVertexInputState   = &avi;
                agpci.pInputAssemblyState = &aia;
                agpci.pViewportState      = &avp;
                agpci.pRasterizationState = &ars;
                agpci.pMultisampleState   = &ams;
                agpci.pDepthStencilState  = &ads;
                agpci.pColorBlendState    = &acb;
                agpci.pDynamicState       = &adyn;
                agpci.layout              = overlayInjectPipelineLayout_;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &agpci, nullptr,
                                                &overlayInjectPipeline_),
                      "vkCreateGraphicsPipelines(overlayInject)");

                vkDestroyShaderModule(ctx->device(), avert, nullptr);
                vkDestroyShaderModule(ctx->device(), afrag, nullptr);
            }

            // ── Splat depth stamp pipeline ──────────────────────────────────
            // A draw INSIDE the overlay pass, ordered between the overlays
            // exempted by kSplatUnoccludedOverlayLayer and everything else:
            // writes the cloud's front-surface depth into the pass's depth
            // attachment so the draws after it are occluded by clouds. Built
            // here because it must match the pass exactly — its color format
            // (attached with every write masked off: this draw touches depth
            // only) and its sample count (overlayMsColor_/overlayMsDepth_ at
            // N samples, swapchain/unjitDepth at one), fixed for the run —
            // same reasoning as the depth prepass above. See
            // shaders/splat_overlay_depth.frag.
            {
                VkDescriptorSetLayoutBinding sBind{};
                sBind.binding         = 0;
                sBind.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                sBind.descriptorCount = 1;
                sBind.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
                VkDescriptorSetLayoutCreateInfo sDslci{};
                sDslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                sDslci.bindingCount = 1;
                sDslci.pBindings    = &sBind;
                check(vkCreateDescriptorSetLayout(ctx->device(), &sDslci, nullptr, &splatStampSetLayout_),
                      "vkCreateDescriptorSetLayout(splatStamp)");

                VkDescriptorPoolSize sPs{};
                sPs.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                sPs.descriptorCount = kFramesInFlight;
                VkDescriptorPoolCreateInfo sDpci{};
                sDpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                sDpci.maxSets       = kFramesInFlight;
                sDpci.poolSizeCount = 1;
                sDpci.pPoolSizes    = &sPs;
                check(vkCreateDescriptorPool(ctx->device(), &sDpci, nullptr, &splatStampPool_),
                      "vkCreateDescriptorPool(splatStamp)");
                std::array<VkDescriptorSetLayout, kFramesInFlight> sLayouts{};
                sLayouts.fill(splatStampSetLayout_);
                VkDescriptorSetAllocateInfo sDsai{};
                sDsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                sDsai.descriptorPool     = splatStampPool_;
                sDsai.descriptorSetCount = kFramesInFlight;
                sDsai.pSetLayouts        = sLayouts.data();
                check(vkAllocateDescriptorSets(ctx->device(), &sDsai, splatStampSets_.data()),
                      "vkAllocateDescriptorSets(splatStamp)");

                VkPushConstantRange sPcr{};
                sPcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                sPcr.offset     = 0;
                sPcr.size       = sizeof(SplatStampPC);
                VkPipelineLayoutCreateInfo sPlci{};
                sPlci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                sPlci.setLayoutCount         = 1;
                sPlci.pSetLayouts            = &splatStampSetLayout_;
                sPlci.pushConstantRangeCount = 1;
                sPlci.pPushConstantRanges    = &sPcr;
                check(vkCreatePipelineLayout(ctx->device(), &sPlci, nullptr, &splatStampPipelineLayout_),
                      "vkCreatePipelineLayout(splatStamp)");

                VkShaderModuleCreateInfo svsmci{};
                svsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                svsmci.codeSize = sizeof(kOverlayFullscreenVertSpv);
                svsmci.pCode    = kOverlayFullscreenVertSpv;
                VkShaderModule svert = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &svsmci, nullptr, &svert),
                      "vkCreateShaderModule(overlay_fullscreen.vert)");
                VkShaderModuleCreateInfo sfsmci{};
                sfsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                sfsmci.codeSize = sizeof(kSplatOverlayDepthFragSpv);
                sfsmci.pCode    = kSplatOverlayDepthFragSpv;
                VkShaderModule sfrag = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &sfsmci, nullptr, &sfrag),
                      "vkCreateShaderModule(splat_overlay_depth.frag)");

                VkPipelineShaderStageCreateInfo sStages[2]{};
                sStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                sStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
                sStages[0].module = svert;
                sStages[0].pName  = "main";
                sStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                sStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
                sStages[1].module = sfrag;
                sStages[1].pName  = "main";

                VkPipelineVertexInputStateCreateInfo svi{};
                svi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                VkPipelineInputAssemblyStateCreateInfo sia{};
                sia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                sia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                VkPipelineViewportStateCreateInfo svp{};
                svp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                svp.viewportCount = 1;
                svp.scissorCount  = 1;
                VkPipelineRasterizationStateCreateInfo srs{};
                srs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                srs.polygonMode = VK_POLYGON_MODE_FILL;
                srs.cullMode    = VK_CULL_MODE_NONE;
                srs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                srs.lineWidth   = 1.0f;
                VkPipelineMultisampleStateCreateInfo sms{};
                sms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                sms.rasterizationSamples = overlaySampleBits();
                // GREATER, reverse-Z: the cloud claims a pixel only where it is
                // in front of whatever the prepass put there. A cloud behind a
                // wall must leave the wall's depth alone, or an overlay drawn
                // on the near side of that wall would vanish.
                VkPipelineDepthStencilStateCreateInfo sds{};
                sds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                sds.depthTestEnable  = VK_TRUE;
                sds.depthWriteEnable = VK_TRUE;
                sds.depthCompareOp   = VK_COMPARE_OP_GREATER;
                // The pass has a color attachment; this draw must declare it
                // and write none of it.
                VkPipelineColorBlendAttachmentState scba{};
                scba.colorWriteMask = 0;
                VkPipelineColorBlendStateCreateInfo scb{};
                scb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                scb.attachmentCount = 1;
                scb.pAttachments    = &scba;
                VkDynamicState sdyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
                VkPipelineDynamicStateCreateInfo sdyn{};
                sdyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                sdyn.dynamicStateCount = 2;
                sdyn.pDynamicStates    = sdyns;

                const VkFormat sFmts[1] = {ctx->swapchainFormat()};
                VkPipelineRenderingCreateInfo sprci{};
                sprci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
                sprci.colorAttachmentCount    = 1;
                sprci.pColorAttachmentFormats = sFmts;
                sprci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

                VkGraphicsPipelineCreateInfo sgpci{};
                sgpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                sgpci.pNext               = &sprci;
                sgpci.stageCount          = 2;
                sgpci.pStages             = sStages;
                sgpci.pVertexInputState   = &svi;
                sgpci.pInputAssemblyState = &sia;
                sgpci.pViewportState      = &svp;
                sgpci.pRasterizationState = &srs;
                sgpci.pMultisampleState   = &sms;
                sgpci.pDepthStencilState  = &sds;
                sgpci.pColorBlendState    = &scb;
                sgpci.pDynamicState       = &sdyn;
                sgpci.layout              = splatStampPipelineLayout_;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &sgpci, nullptr,
                                                &splatStampPipeline_),
                      "vkCreateGraphicsPipelines(splatStamp)");

                vkDestroyShaderModule(ctx->device(), svert, nullptr);
                vkDestroyShaderModule(ctx->device(), sfrag, nullptr);
            }
        }

void VulkanRenderer::Impl::createParticlePipeline() {
            if (particlePipelineNormal_ != VK_NULL_HANDLE) return;

            // set 0, binding 0: combined image sampler (particle texture).
            VkDescriptorSetLayoutBinding b{};
            b.binding         = 0;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo dslci{};
            dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dslci.bindingCount = 1;
            dslci.pBindings    = &b;
            check(vkCreateDescriptorSetLayout(ctx->device(), &dslci, nullptr, &particleDescSetLayout_),
                  "vkCreateDescriptorSetLayout(particle)");

            // set 1, binding 0: the unified-fog snapshot UBO (Phase 2b). Bound
            // once per frame; particle.frag fogs world-space billboards with it.
            // binding 1: the per-particle light/fog results particle_light.comp
            // wrote this frame — particle.vert fetches them by gl_InstanceIndex
            // (lit particles). The world-space Sprite pipeline shares this
            // layout but never references set 1, so it ignores the (still-
            // bound) descriptors.
            VkDescriptorSetLayoutBinding fb[2]{};
            fb[0].binding         = 0;
            fb[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            fb[0].descriptorCount = 1;
            fb[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            fb[1].binding         = 1;
            fb[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            fb[1].descriptorCount = 1;
            fb[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
            VkDescriptorSetLayoutCreateInfo fdslci{};
            fdslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            fdslci.bindingCount = 2;
            fdslci.pBindings    = fb;
            check(vkCreateDescriptorSetLayout(ctx->device(), &fdslci, nullptr, &overlayFogDescSetLayout_),
                  "vkCreateDescriptorSetLayout(overlayFog)");

            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pcRange.offset     = 0;
            pcRange.size       = 128;// mat4 modelView (64) + mat4 proj (64)
            const VkDescriptorSetLayout setLayouts[2] = {particleDescSetLayout_, overlayFogDescSetLayout_};
            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 2;
            plci.pSetLayouts            = setLayouts;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &particlePipelineLayout_),
                  "vkCreatePipelineLayout(particle)");

            // Persistent per-FIF fog UBO buffers + descriptor sets (host-visible,
            // memcpy'd each frame in the record). One pool, kFramesInFlight sets.
            for (auto& b : overlayFogUbos_) {
                b = createBuffer(ctx->allocator(), ctx->device(), sizeof(GpuOverlayFogUbo),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT);
            }
            // Lit-particle buffers (eager, fixed footprint): host-written
            // world-space centers in, particle_light.comp's light/fog results
            // out (device-local; compute writes, particle.vert reads).
            for (auto& b : particleCenterBufs_) {
                b = createBuffer(ctx->allocator(), ctx->device(),
                                 VkDeviceSize(kMaxLitParticles) * 4 * sizeof(float),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }
            for (auto& b : particleLightBufs_) {
                b = createBuffer(ctx->allocator(), ctx->device(),
                                 VkDeviceSize(kMaxLitParticles) * 8 * sizeof(float),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0);
            }
            {
                VkDescriptorPoolSize fps[2]{};
                fps[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                fps[0].descriptorCount = kFramesInFlight;
                fps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                fps[1].descriptorCount = kFramesInFlight;
                VkDescriptorPoolCreateInfo fdpci{};
                fdpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                fdpci.maxSets       = kFramesInFlight;
                fdpci.poolSizeCount = 2;
                fdpci.pPoolSizes    = fps;
                check(vkCreateDescriptorPool(ctx->device(), &fdpci, nullptr, &overlayFogDescPool_),
                      "vkCreateDescriptorPool(overlayFog)");
                for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                    VkDescriptorSetAllocateInfo asi{};
                    asi.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    asi.descriptorPool     = overlayFogDescPool_;
                    asi.descriptorSetCount = 1;
                    asi.pSetLayouts        = &overlayFogDescSetLayout_;
                    check(vkAllocateDescriptorSets(ctx->device(), &asi, &overlayFogDescSets_[f]),
                          "vkAllocateDescriptorSets(overlayFog)");
                    VkDescriptorBufferInfo dbi{};
                    dbi.buffer = overlayFogUbos_[f].handle;
                    dbi.offset = 0;
                    dbi.range  = sizeof(GpuOverlayFogUbo);
                    VkDescriptorBufferInfo lbi{};
                    lbi.buffer = particleLightBufs_[f].handle;
                    lbi.offset = 0;
                    lbi.range  = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet w[2]{};
                    w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[0].dstSet          = overlayFogDescSets_[f];
                    w[0].dstBinding      = 0;
                    w[0].descriptorCount = 1;
                    w[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    w[0].pBufferInfo     = &dbi;
                    w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[1].dstSet          = overlayFogDescSets_[f];
                    w[1].dstBinding      = 1;
                    w[1].descriptorCount = 1;
                    w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    w[1].pBufferInfo     = &lbi;
                    vkUpdateDescriptorSets(ctx->device(), 2, w, 0, nullptr);
                }
            }
            // (particle_light.comp's IO sets are created lazily by
            // ensureParticleIoSets() — deferredShade_, which owns their layout,
            // is itself lazily constructed AFTER the first particle-pipeline
            // ensure, so creating them here would race the init order.)

            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                VkDescriptorPoolSize ps{};
                ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                ps.descriptorCount = kMaxParticleTexPerFrame;
                VkDescriptorPoolCreateInfo dpci{};
                dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                dpci.maxSets       = kMaxParticleTexPerFrame;
                dpci.poolSizeCount = 1;
                dpci.pPoolSizes    = &ps;
                check(vkCreateDescriptorPool(ctx->device(), &dpci, nullptr, &particleDescPools_[f]),
                      "vkCreateDescriptorPool(particle)");
            }

            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kParticleVertSpv);
            vsmci.pCode    = kParticleVertSpv;
            VkShaderModule vert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vert),
                  "vkCreateShaderModule(particle.vert)");
            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kParticleFragSpv);
            fsmci.pCode    = kParticleFragSpv;
            VkShaderModule frag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &frag),
                  "vkCreateShaderModule(particle.frag)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName  = "main";
            // Fragment spec constant 0 (kAdditive): the fog fade injects the mist
            // colour for alpha smoke but only ATTENUATES additive glow (see
            // particle.frag). Normal=0, Additive=1.
            static const uint32_t kSpecAlpha    = 0u;
            static const uint32_t kSpecAdditive = 1u;
            VkSpecializationMapEntry particleSpecEntry{0, 0, sizeof(uint32_t)};
            VkSpecializationInfo particleSpecAlpha{1, &particleSpecEntry, sizeof(uint32_t), &kSpecAlpha};
            VkSpecializationInfo particleSpecAdd{1, &particleSpecEntry, sizeof(uint32_t), &kSpecAdditive};
            stages[1].pSpecializationInfo = &particleSpecAlpha;// Normal variant

            // 4 separate vertex bindings: pos(0), normal(1), uv(2), color(3).
            VkVertexInputBindingDescription vibs[4]{};
            vibs[0].binding = 0; vibs[0].stride = 3 * sizeof(float); vibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[1].binding = 1; vibs[1].stride = 3 * sizeof(float); vibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[2].binding = 2; vibs[2].stride = 2 * sizeof(float); vibs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vibs[3].binding = 3; vibs[3].stride = 3 * sizeof(float); vibs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            VkVertexInputAttributeDescription vias[4]{};
            vias[0].location = 0; vias[0].binding = 0; vias[0].format = VK_FORMAT_R32G32B32_SFLOAT; vias[0].offset = 0;
            vias[1].location = 1; vias[1].binding = 1; vias[1].format = VK_FORMAT_R32G32B32_SFLOAT; vias[1].offset = 0;
            vias[2].location = 2; vias[2].binding = 2; vias[2].format = VK_FORMAT_R32G32_SFLOAT;    vias[2].offset = 0;
            vias[3].location = 3; vias[3].binding = 3; vias[3].format = VK_FORMAT_R32G32B32_SFLOAT; vias[3].offset = 0;
            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 4;
            vi.pVertexBindingDescriptions      = vibs;
            vi.vertexAttributeDescriptionCount = 4;
            vi.pVertexAttributeDescriptions    = vias;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            // Side::Double — particles are billboards, draw both faces.
            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode    = VK_CULL_MODE_NONE;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            // Drawn inside the overlay render-pass instance → must match its
            // sample count (see the overlayMs* member docs).
            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = overlaySampleBits();

            VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates    = dynStates;

            const VkFormat particleFmts[1] = {ctx->swapchainFormat()};
            VkPipelineRenderingCreateInfo prci{};
            prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            prci.colorAttachmentCount    = 1;
            prci.pColorAttachmentFormats = particleFmts;
            prci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

            // ── Normal variant: non-premultiplied alpha, depth-tested ──────────
            // depthTest GREATER_OR_EQUAL (reverse-Z) so particles are occluded by
            // scene geometry; depthWrite OFF (transparent, read-only unjitDepth).
            VkPipelineDepthStencilStateCreateInfo dsNormal{};
            dsNormal.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            dsNormal.depthTestEnable  = VK_TRUE;
            dsNormal.depthWriteEnable = VK_FALSE;
            dsNormal.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;

            VkPipelineColorBlendAttachmentState cbasAlpha[1]{};
            cbasAlpha[0].blendEnable         = VK_TRUE;
            cbasAlpha[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbasAlpha[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasAlpha[0].colorBlendOp        = VK_BLEND_OP_ADD;
            cbasAlpha[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasAlpha[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbasAlpha[0].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbasAlpha[0].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cbAlpha{};
            cbAlpha.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cbAlpha.attachmentCount = 1;
            cbAlpha.pAttachments    = cbasAlpha;

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.pNext               = &prci;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &dsNormal;
            gpci.pColorBlendState    = &cbAlpha;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = particlePipelineLayout_;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci, nullptr,
                                            &particlePipelineNormal_),
                  "vkCreateGraphicsPipelines(particleNormal)");

            // ── Additive variant: src·srcAlpha + dst, depth-test OFF ───────────
            // Matches ParticleSystem's depthTest=false for non-Normal blending
            // (fireball / firework draw over the scene).
            VkPipelineDepthStencilStateCreateInfo dsAdd = dsNormal;
            dsAdd.depthTestEnable = VK_FALSE;
            VkPipelineColorBlendAttachmentState cbasAdd[1] = {cbasAlpha[0]};
            cbasAdd[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasAdd[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbasAdd[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            VkPipelineColorBlendStateCreateInfo cbAdd = cbAlpha;
            cbAdd.pAttachments = cbasAdd;
            // Additive variant reuses the same stages but with kAdditive=1 so the
            // fog fade attenuates the glow without injecting the mist colour.
            VkPipelineShaderStageCreateInfo stagesAdd[2] = {stages[0], stages[1]};
            stagesAdd[1].pSpecializationInfo = &particleSpecAdd;
            VkGraphicsPipelineCreateInfo gpciAdd = gpci;
            gpciAdd.pStages            = stagesAdd;
            gpciAdd.pDepthStencilState = &dsAdd;
            gpciAdd.pColorBlendState   = &cbAdd;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciAdd, nullptr,
                                            &particlePipelineAdditive_),
                  "vkCreateGraphicsPipelines(particleAdditive)");

            vkDestroyShaderModule(ctx->device(), vert, nullptr);
            vkDestroyShaderModule(ctx->device(), frag, nullptr);

            ensureParticleWhiteTexture();
            createSpriteWorldPipeline();
            createFieldBillboardPipeline();
        }

// ── ParticleField billboards — plans/particle-atmosphere.md F-D ─────────────
//
// Everything about this pipeline is an argument against the legacy one it sits
// beside, so the differences are worth naming:
//
//   • ZERO vertex input bindings. The quad is four vertices of nothing: the
//     corner comes from gl_VertexIndex and the particle from gl_InstanceIndex,
//     and the position/radius are pulled out of the field's device buffer by
//     address. The legacy path uploads 4 coincident vertices x 4 attributes per
//     particle per frame and is capped at 16384 particles because of it.
//   • ONE indirect draw per field, with instanceCount written on the DEVICE.
//     The CPU never learns how many particles there are.
//   • DEPTH-TESTED, unlike the legacy additive variant. That variant turns the
//     test off so a fireball draws over the scene; a rain streak or an ember
//     behind a rock must not, and the pass already binds the read-only
//     unjittered depth, so correctness here is free.
//   • ADDITIVE, ONE/ONE, with coverage folded into the radiance in the
//     fragment. Addition commutes, so there is nothing to sort — which is the
//     entire reason this slice is additive-first.
void VulkanRenderer::Impl::createFieldBillboardPipeline() {
            if (fieldBillboardPipeline_ != VK_NULL_HANDLE) return;
            if (particleDescSetLayout_ == VK_NULL_HANDLE) return;// caller ordering bug

            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pcRange.offset     = 0;
            pcRange.size       = sizeof(FieldBillboardPC);
            // Set 0 only, and it is the LEGACY path's texture layout reused
            // verbatim — so BillboardRepr::texture works through the same
            // ensureParticleTexture cache and the same 1x1 white default, and
            // an untextured field costs one extra descriptor write per draw
            // from a pool that is already reset once per frame.
            //
            // (plans/particle-volumetric-sprites added a set 1 here for the
            // density volume the vertex stage marched. R8 moved the marches to
            // particlefield_transmit.comp, which binds the volume itself, and
            // this layout went back to one set.)
            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &particleDescSetLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr,
                                         &fieldBillboardPipelineLayout_),
                  "vkCreatePipelineLayout(fieldBillboard)");

            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kParticleFieldBillboardVertSpv);
            vsmci.pCode    = kParticleFieldBillboardVertSpv;
            VkShaderModule vert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vert),
                  "vkCreateShaderModule(particlefield_billboard.vert)");
            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kParticleFieldBillboardFragSpv);
            fsmci.pCode    = kParticleFieldBillboardFragSpv;
            VkShaderModule frag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &frag),
                  "vkCreateShaderModule(particlefield_billboard.frag)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName  = "main";

            // No bindings, no attributes — the pipeline is vertex-less.
            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode    = VK_CULL_MODE_NONE;// a billboard has no back face
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates    = dynStates;

            const VkFormat fmts[1] = {ctx->swapchainFormat()};
            VkPipelineRenderingCreateInfo prci{};
            prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            prci.colorAttachmentCount    = 1;
            prci.pColorAttachmentFormats = fmts;
            prci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

            // Reverse-Z, so GREATER_OR_EQUAL. Depth-write OFF: the attachment is
            // the overlay prepass's read-only depth and a transparent must never
            // occlude the next transparent.
            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;

            // Straight ONE/ONE: the fragment already multiplied its radiance by
            // the sprite's coverage, so there is no second alpha term and
            // nothing that depends on draw order. ALPHA IS MASKED OUT — the
            // swapchain's alpha is not part of the composite and an additive
            // pass has no business accumulating into it.
            VkPipelineColorBlendAttachmentState cbas[1]{};
            cbas[0].blendEnable         = VK_TRUE;
            cbas[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas[0].colorBlendOp        = VK_BLEND_OP_ADD;
            cbas[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cbas[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas[0].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbas[0].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT;
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments    = cbas;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = overlaySampleBits();

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.pNext               = &prci;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = fieldBillboardPipelineLayout_;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci, nullptr,
                                            &fieldBillboardPipeline_),
                  "vkCreateGraphicsPipelines(fieldBillboard)");

            // The 1-sample sibling a SECONDARY view needs. Secondaries composite
            // into their own single-sample colour target and never run the
            // overlay's MSAA machinery, so when the primary's overlay is
            // multisampled the two cannot share a pipeline. When it is not, they
            // are the same object and the alias costs nothing.
            if (ms.rasterizationSamples == VK_SAMPLE_COUNT_1_BIT) {
                fieldBillboardPipeline1x_      = fieldBillboardPipeline_;
                fieldBillboardPipeline1xOwned_ = false;
            } else {
                VkPipelineMultisampleStateCreateInfo ms1 = ms;
                ms1.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                VkGraphicsPipelineCreateInfo gpci1 = gpci;
                gpci1.pMultisampleState = &ms1;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci1,
                                                nullptr, &fieldBillboardPipeline1x_),
                      "vkCreateGraphicsPipelines(fieldBillboard 1x)");
                fieldBillboardPipeline1xOwned_ = true;
            }

            // ── 4c: the ALPHA-OVER variant ──────────────────────────────────
            // One state differs: premultiplied SRC_ALPHA-over (ONE,
            // ONE_MINUS_SRC_ALPHA) instead of ONE/ONE, so a sprite OCCLUDES
            // what is behind it. Everything else — depth test, write mask,
            // samples, layout, both stages — is the same object's, because
            // everything else about the two modes is the same.
            //
            // The colour write mask still excludes alpha, and that is not in
            // tension with reading SRC_ALPHA: the write mask governs what
            // reaches the attachment, never what the blend equation may use as
            // a factor. So the swapchain's alpha stays exactly as untouched as
            // it is on the additive path.
            //
            // NOTHING IS SORTED here either. Draws are issued in field order
            // and, within a field, instances in slot order, which Vulkan's
            // primitive-order guarantee makes the BLEND order as well — so a
            // host with a few hundred sprites gets correct back-to-front by
            // submitting them that way, and a device radix sort stays deferred
            // to the count that needs one.
            {
                VkPipelineColorBlendAttachmentState cbasA[1] = {cbas[0]};
                cbasA[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                cbasA[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                VkPipelineColorBlendStateCreateInfo cbA = cb;
                cbA.pAttachments = cbasA;

                VkGraphicsPipelineCreateInfo gpciA = gpci;
                gpciA.pColorBlendState = &cbA;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciA,
                                                nullptr, &fieldBillboardAlphaPipeline_),
                      "vkCreateGraphicsPipelines(fieldBillboard alpha)");
                if (ms.rasterizationSamples == VK_SAMPLE_COUNT_1_BIT) {
                    fieldBillboardAlphaPipeline1x_      = fieldBillboardAlphaPipeline_;
                    fieldBillboardAlphaPipeline1xOwned_ = false;
                } else {
                    VkPipelineMultisampleStateCreateInfo ms1 = ms;
                    ms1.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                    VkGraphicsPipelineCreateInfo gpciA1 = gpciA;
                    gpciA1.pMultisampleState = &ms1;
                    check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciA1,
                                                    nullptr, &fieldBillboardAlphaPipeline1x_),
                          "vkCreateGraphicsPipelines(fieldBillboard alpha 1x)");
                    fieldBillboardAlphaPipeline1xOwned_ = true;
                }
            }

            // ── F4: the GLOW variant ────────────────────────────────────────
            // Same two stages, same layout, same additive blend — three
            // differences and each is forced by where it draws:
            //   • rgba16f, because BillboardGlowPass's target holds LINEAR HDR
            //     (the fragment stage skips its display transform there) and
            //     the bright pass downstream is defined on linear radiance;
            //   • ONE sample, because that target never participates in the
            //     overlay's MSAA machinery;
            //   • its own HALF-EXTENT depth attachment, reduced from the
            //     overlay's full-extent one by BillboardGlowPass (a render area
            //     must fit its attachments, so the full-extent buffer cannot be
            //     attached here directly). Same reverse-Z test as the sharp
            //     quads, so a spark behind a wall contributes no halo.
            {
                const VkFormat glowFmt[1] = {VK_FORMAT_R16G16B16A16_SFLOAT};
                VkPipelineRenderingCreateInfo prciGlow = prci;
                prciGlow.pColorAttachmentFormats = glowFmt;
                prciGlow.depthAttachmentFormat   = vulkan::BillboardGlowPass::kDepthFormat;

                VkPipelineDepthStencilStateCreateInfo dsGlow{};
                dsGlow.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                dsGlow.depthTestEnable  = VK_TRUE;
                dsGlow.depthWriteEnable = VK_FALSE;
                dsGlow.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;

                VkPipelineMultisampleStateCreateInfo msGlow = ms;
                msGlow.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

                VkGraphicsPipelineCreateInfo gpciGlow = gpci;
                gpciGlow.pNext              = &prciGlow;
                gpciGlow.pDepthStencilState = &dsGlow;
                gpciGlow.pMultisampleState  = &msGlow;
                check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpciGlow,
                                                nullptr, &fieldBillboardGlowPipeline_),
                      "vkCreateGraphicsPipelines(fieldBillboard glow)");
            }

            vkDestroyShaderModule(ctx->device(), vert, nullptr);
            vkDestroyShaderModule(ctx->device(), frag, nullptr);
        }

// ── F4: the billboard-glow COMPOSITE pipeline ───────────────────────────────
//
// A fullscreen additive draw that folds BillboardGlowPass's finished pyramid
// into the swapchain from INSIDE the overlay render-pass instance — which is
// the whole trick of F4 item 1. The bloom the sparks get therefore arrives
// without the billboard composite point moving one instruction earlier, so
// F3 note 1's "field billboards are outside TAA/DLSS/FSR entirely" survives by
// construction rather than by re-argument.
//
// Depth TEST off (the halo is a lens effect and has no depth of its own) while
// the pass's depth attachment stays declared, because a pipeline used inside a
// render-pass instance must agree with its attachment formats.
//
// Created lazily alongside the glow pass, so it needs that pass's set layout.
void VulkanRenderer::Impl::createFieldGlowCompositePipeline() {
            if (fieldGlowCompositePipeline_ != VK_NULL_HANDLE) return;
            if (!billboardGlow_) return;

            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            pcRange.size       = sizeof(FieldGlowPC);
            const VkDescriptorSetLayout setLayout = billboardGlow_->compositeSetLayout();
            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &setLayout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcRange;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &fieldGlowCompositeLayout_),
                  "vkCreatePipelineLayout(fieldGlowComposite)");

            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kOverlayFullscreenVertSpv);
            vsmci.pCode    = kOverlayFullscreenVertSpv;
            VkShaderModule vert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vert),
                  "vkCreateShaderModule(overlay_fullscreen.vert glow)");
            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kParticleFieldGlowFragSpv);
            fsmci.pCode    = kParticleFieldGlowFragSpv;
            VkShaderModule frag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &frag),
                  "vkCreateShaderModule(particlefield_glow.frag)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName  = "main";

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;
            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode    = VK_CULL_MODE_NONE;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;
            VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates    = dynStates;

            const VkFormat fmts[1] = {ctx->swapchainFormat()};
            VkPipelineRenderingCreateInfo prci{};
            prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            prci.colorAttachmentCount    = 1;
            prci.pColorAttachmentFormats = fmts;
            prci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_FALSE;
            ds.depthWriteEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState cbas[1]{};
            cbas[0].blendEnable         = VK_TRUE;
            cbas[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas[0].colorBlendOp        = VK_BLEND_OP_ADD;
            cbas[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cbas[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas[0].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbas[0].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT;
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments    = cbas;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = overlaySampleBits();

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.pNext               = &prci;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = fieldGlowCompositeLayout_;
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci, nullptr,
                                            &fieldGlowCompositePipeline_),
                  "vkCreateGraphicsPipelines(fieldGlowComposite)");

            vkDestroyShaderModule(ctx->device(), vert, nullptr);
            vkDestroyShaderModule(ctx->device(), frag, nullptr);
        }

void VulkanRenderer::Impl::createSpriteWorldPipeline() {
            if (spriteWorldPipeline_ != VK_NULL_HANDLE) return;

            // Shared canonical quad: 4 interleaved (pos.xyz, uv.xy) verts +
            // 6 indices — matches Sprite's geometry (src/objects/Sprite.cpp).
            {
                static const float quad[] = {
                        -0.5f, -0.5f, 0.f, 0.f, 0.f,
                         0.5f, -0.5f, 0.f, 1.f, 0.f,
                         0.5f,  0.5f, 0.f, 1.f, 1.f,
                        -0.5f,  0.5f, 0.f, 0.f, 1.f};
                static const uint32_t idx[] = {0, 1, 2, 0, 2, 3};
                spriteQuadVtx_ = createBuffer(
                        ctx->allocator(), ctx->device(), sizeof(quad),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                spriteQuadIdx_ = createBuffer(
                        ctx->allocator(), ctx->device(), sizeof(idx),
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), spriteQuadVtx_, quad, sizeof(quad));
                uploadHostVisible(ctx->allocator(), spriteQuadIdx_, idx, sizeof(idx));
            }

            VkShaderModuleCreateInfo vsmci{};
            vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vsmci.codeSize = sizeof(kSprite3dVertSpv);
            vsmci.pCode    = kSprite3dVertSpv;
            VkShaderModule vert = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &vsmci, nullptr, &vert),
                  "vkCreateShaderModule(sprite3d.vert)");
            VkShaderModuleCreateInfo fsmci{};
            fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fsmci.codeSize = sizeof(kOverlaySpriteFragSpv);
            fsmci.pCode    = kOverlaySpriteFragSpv;
            VkShaderModule frag = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx->device(), &fsmci, nullptr, &frag),
                  "vkCreateShaderModule(overlay_sprite.frag for sprite3d)");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName  = "main";

            // One interleaved binding: pos.xyz at 0, uv.xy at offset 12.
            VkVertexInputBindingDescription vib{};
            vib.binding   = 0;
            vib.stride    = 5 * sizeof(float);
            vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            VkVertexInputAttributeDescription vias[2]{};
            vias[0].location = 0; vias[0].binding = 0; vias[0].format = VK_FORMAT_R32G32B32_SFLOAT; vias[0].offset = 0;
            vias[1].location = 1; vias[1].binding = 0; vias[1].format = VK_FORMAT_R32G32_SFLOAT;    vias[1].offset = 3 * sizeof(float);
            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount   = 1;
            vi.pVertexBindingDescriptions      = &vib;
            vi.vertexAttributeDescriptionCount = 2;
            vi.pVertexAttributeDescriptions    = vias;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount  = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode    = VK_CULL_MODE_NONE;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            // Same render-pass instance as the overlay → same sample count.
            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = overlaySampleBits();

            // Depth-tested (occluded by scene), depth-write off (transparent).
            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_FALSE;
            ds.depthCompareOp   = VK_COMPARE_OP_GREATER_OR_EQUAL;// reverse-Z

            VkPipelineColorBlendAttachmentState cbas[1]{};
            cbas[0].blendEnable         = VK_TRUE;
            cbas[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbas[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbas[0].colorBlendOp        = VK_BLEND_OP_ADD;
            cbas[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbas[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbas[0].alphaBlendOp        = VK_BLEND_OP_ADD;
            cbas[0].colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments    = cbas;

            VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn{};
            dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dyn.dynamicStateCount = 2;
            dyn.pDynamicStates    = dynStates;

            const VkFormat spriteFmts[1] = {ctx->swapchainFormat()};
            VkPipelineRenderingCreateInfo prci{};
            prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            prci.colorAttachmentCount    = 1;
            prci.pColorAttachmentFormats = spriteFmts;
            prci.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;

            VkGraphicsPipelineCreateInfo gpci{};
            gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gpci.pNext               = &prci;
            gpci.stageCount          = 2;
            gpci.pStages             = stages;
            gpci.pVertexInputState   = &vi;
            gpci.pInputAssemblyState = &ia;
            gpci.pViewportState      = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pMultisampleState   = &ms;
            gpci.pDepthStencilState  = &ds;
            gpci.pColorBlendState    = &cb;
            gpci.pDynamicState       = &dyn;
            gpci.layout              = particlePipelineLayout_;// 128B SpritePC + set-0 sampler
            check(vkCreateGraphicsPipelines(ctx->device(), ctx->pipelineCache(), 1, &gpci, nullptr,
                                            &spriteWorldPipeline_),
                  "vkCreateGraphicsPipelines(spriteWorld)");

            vkDestroyShaderModule(ctx->device(), vert, nullptr);
            vkDestroyShaderModule(ctx->device(), frag, nullptr);
        }

}// namespace threepp
