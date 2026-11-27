#include "threepp/renderers/vulkan/BloomPass.hpp"
#include "threepp/renderers/vulkan/TransientPool.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

namespace threepp::vulkan {

    BloomPass::BloomPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight),
          pyramid_(ctx, framesInFlight, kMaxLevels, "bloom") {
        sceneHdr_.resize(framesInFlight_);
        pyr_.resize(framesInFlight_ * kMaxLevels);
    }

    BloomPass::~BloomPass() {
        destroyImages();
    }

    void BloomPass::destroyImages() {
        VkDevice d = ctx_.device();
        for (auto& img : sceneHdr_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : pyr_)      destroyImage2D(ctx_.allocator(), d, img);
    }

    Image2D BloomPass::createStorageSampledImage(uint32_t w, uint32_t h, const char* label, uint32_t poolSlots,
                                                 TransientSpan span) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = VK_FORMAT_R16G16B16A16_SFLOAT;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = out.format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_SRC for the determinism audit's sceneHdr readback
        // (VulkanRenderer::readSceneHdrDebug) — a usage bit is free.
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        TransientPool* pool = poolSlots ? pool_ : nullptr;
        check(createImageMaybePooled(ctx_.allocator(), pool,
                                     span ? span : transientGroup(TransientPhase::Post, poolView_),
                                     poolSlots, ici, &out.image, &out.alloc),
              label);

        // A pooled image is discarded at its first use in every frame.
        if (out.alloc != VK_NULL_HANDLE) transitionFreshImage(out.image);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = out.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vulkan::createImageView(bloom)");
        ctx_.setObjectName(out.image, label);
        ctx_.setObjectName(out.view,  label);
        return out;
    }

    void BloomPass::transitionFreshImage(VkImage img) {
        submitOneShot(ctx_.device(), ctx_.graphicsQueue(), cmdPool_, "bloom", [&](VkCommandBuffer cb) {
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        });
    }

    void BloomPass::createImages(uint32_t width, uint32_t height) {
        destroyImages();
        width_  = width;
        height_ = height;
        // Pyramid levels halve from half res down to ~1/64 of the frame.
        levels_ = BloomPyramid::levelCount(width_, height_, kMaxLevels);
        // sceneHdr is live from the shade (its first write) through the post
        // composite — no later frame reads it — so it shares memory with
        // images live before the shade (uv) and after the post chain (the
        // tail's). Pooled, it holds nothing after the frame's graph:
        // readSceneHdrDebug reads it back whole only under THREEPP_VK_NO_ALIAS.
        for (uint32_t f = 0; f < framesInFlight_; ++f)
            sceneHdr_[f] = createStorageSampledImage(width_, height_, "vmaCreateImage(bloom.sceneHdr)",
                                                     transientSlot(f),
                                                     transientSpan(TransientPhase::Light, TransientPhase::Post,
                                                                   poolView_));
        for (uint32_t f = 0; f < framesInFlight_; ++f)
            for (uint32_t l = 0; l < levels_; ++l) {
                const VkExtent2D e = BloomPyramid::levelExtent(width_, height_, l);
                pyr_[f * kMaxLevels + l] =
                        createStorageSampledImage(e.width, e.height, "vmaCreateImage(bloom.pyr)", transientSlot(f));
            }
    }

    void BloomPass::rewriteDescriptors() {
        for (uint32_t f = 0; f < framesInFlight_; ++f)
            pyramid_.writeSets(f, sceneHdr_[f].view, &pyr_[f * kMaxLevels], levels_);
    }

    void BloomPass::declare(rg::RenderGraph& graph, rg::PassBuilder& pass, uint32_t frame,
                            rg::ImageHandle sceneHdr) const {
        pass.use(sceneHdr, rg::sampled(VK_IMAGE_LAYOUT_GENERAL));
        for (uint32_t l = 0; l < levels_; ++l) {
            const auto h = graph.importImage("bloom.pyr", pyr_[frame * kMaxLevels + l].image,
                                             VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_GENERAL);
            pass.use(h, rg::storageReadWrite());
            pass.use(h, rg::sampled(VK_IMAGE_LAYOUT_GENERAL));
        }
    }

    void BloomPass::recordPyramid(VkCommandBuffer cb, uint32_t frame,
                                  uint32_t width, uint32_t height,
                                  float bloomIntensity, float bloomThreshold,
                                  float bloomClamp) {
        if (bloomIntensity <= 0.0f) return;
        // sceneHdr -> level 0 runs the Karis average, the soft-knee bright pass
        // and the per-tap clamp; the deeper levels the plain 13-tap. The upsample
        // accumulates the whole chain into level 0, and PostComposite divides
        // bloomIntensity by the level count, so total energy matches a
        // single-level bloom.
        BloomPyramid::Chain chain;
        chain.input      = {width, height};
        chain.base       = {width_, height_};
        chain.levels     = levels_;
        chain.threshold  = bloomThreshold;
        chain.clampMax   = bloomClamp;
        chain.firstLevel = 1u;
        pyramid_.record(cb, frame, chain);
    }

}// namespace threepp::vulkan
