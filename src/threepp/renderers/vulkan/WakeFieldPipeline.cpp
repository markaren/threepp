#include "threepp/renderers/vulkan/WakeFieldPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/wake_field.comp.spv.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace threepp::vulkan {

    static_assert(sizeof(WakeFieldPipeline::PushConstants) == 24,
                  "WakeFieldPipeline::PushConstants must match wake_field.comp's Pc layout (24 bytes)");

    namespace {

        constexpr VkFormat kStateFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        constexpr VkFormat kVelFormat   = VK_FORMAT_R16G16_SFLOAT;
        constexpr VkDeviceSize kTableBytes =
                sizeof(WakeTableGpu) + DisplacedMesh::kMaxWakeSources * sizeof(DisplacedMesh::WakeSource);

        void arrayImage(VulkanContext& ctx, VkFormat format, uint32_t res, uint32_t layers, const char* name,
                        VkImage& image, VmaAllocation& alloc, VkImageView& view) {
            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = format;
            ici.extent        = {res, res, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = layers;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx.allocator(), &ici, &aci, &image, &alloc, nullptr), "vmaCreateImage(wakeField)");
            VkImageViewCreateInfo vci{};
            vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image            = image;
            vci.viewType         = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            vci.format           = format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
            check(createImageView(ctx.device(), &vci, nullptr, &view), "createImageView(wakeField)");
            ctx.setObjectName(image, name);
            ctx.setObjectName(view, name);
        }

        // UNDEFINED -> GENERAL, zeroed.
        void clearToGeneral(VkCommandBuffer cb, VkImage image, uint32_t layers) {
            VkImageMemoryBarrier imb{};
            imb.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imb.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            imb.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            imb.srcAccessMask       = 0;
            imb.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            imb.image               = image;
            imb.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
            imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &imb);
            const VkClearColorValue cc{};
            vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &cc, 1, &imb.subresourceRange);
            imb.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            imb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &imb);
        }

        // Compute writes of one dispatch -> compute reads and writes of the next.
        void computeBarrier(VkCommandBuffer cb) {
            VkMemoryBarrier mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        }

    }// namespace

    WakeFieldPipeline::WakeFieldPipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    WakeFieldPipeline::~WakeFieldPipeline() {
        VkDevice d = ctx_.device();
        if (dummyView_)      vkDestroyImageView(d, dummyView_, nullptr);
        if (dummyImage_)     vmaDestroyImage(ctx_.allocator(), dummyImage_, dummyAlloc_);
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
    }

    void WakeFieldPipeline::createPipeline() {
        // Bindings: 0, 1 = state sources, 2 = velocity source (combined image
        // samplers over the array images); 3 = state target, 4 = velocity
        // target (storage images).
        std::array<VkDescriptorSetLayoutBinding, 5> bb{};
        for (uint32_t i = 0; i < 5; ++i) {
            bb[i].binding         = i;
            bb[i].descriptorType  = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bb[i].descriptorCount = 1;
            bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bb.size());
        dlci.pBindings    = bb.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(wakeField)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(wakeField)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kWakeFieldCompSpv);
        smci.pCode    = kWakeFieldCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod), "vkCreateShaderModule(wakeField)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(wakeField)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // Three sets an ocean (one per dispatch of a step), freed with its State.
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 3 * 3 * kMaxOceans;
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 2 * 3 * kMaxOceans;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = 3 * kMaxOceans;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(wakeField)");

        // LINEAR, and a zero border: water outside a patch is undisturbed, and
        // that is also what a step reads where the patch has just arrived.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.borderColor  = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler(wakeField)");

        arrayImage(ctx_, kStateFormat, 1, 1, "oceanWakeDummy (1x1, binding 78 placeholder)",
                   dummyImage_, dummyAlloc_, dummyView_);
    }

    void WakeFieldPipeline::initDummy(VkCommandBuffer cb) {
        clearToGeneral(cb, dummyImage_, 1);
    }

    std::unique_ptr<WakeFieldPipeline::State> WakeFieldPipeline::createState(uint32_t res, uint32_t layers) {
        auto st = std::make_unique<State>();
        st->res    = std::clamp(res, 64u, 2048u);
        st->layers = std::clamp(layers, 1u, DisplacedMesh::kMaxWakePatches);
        static const char* const kNames[5] = {"ocean.wake.stateA", "ocean.wake.stateB", "ocean.wake.stateC",
                                              "ocean.wake.velA", "ocean.wake.velB"};
        for (uint32_t i = 0; i < 5; ++i) {
            arrayImage(ctx_, i < 3 ? kStateFormat : kVelFormat, st->res, st->layers, kNames[i],
                       st->image[i], st->alloc[i], st->view[i]);
        }
        st->table = createBuffer(ctx_.allocator(), ctx_.device(), kTableBytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(st->table.handle, "ocean.wake.table");

        std::array<VkDescriptorSetLayout, 3> layouts{dsLayout_, dsLayout_, dsLayout_};
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = descPool_;
        dai.descriptorSetCount = 3;
        dai.pSetLayouts        = layouts.data();
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, st->ds.data()), "vkAllocateDescriptorSets(wakeField)");

        // image index bound at (state0, state1, vel, stateOut, velOut), per dispatch
        static constexpr uint32_t kBind[3][5] = {
                {0, 0, 3, 1, 4},// advect:  A, velA        -> B, velB
                {0, 1, 3, 2, 4},// correct: A, B, velA     -> C   (velB bound, not written)
                {2, 2, 4, 0, 3},// settle:  C, velB        -> A, velA
        };
        for (uint32_t m = 0; m < 3; ++m) {
            std::array<VkDescriptorImageInfo, 5> ii{};
            std::array<VkWriteDescriptorSet, 5>  ws{};
            for (uint32_t b = 0; b < 5; ++b) {
                ii[b].sampler     = b < 3 ? sampler_ : VK_NULL_HANDLE;
                ii[b].imageView   = st->view[kBind[m][b]];
                ii[b].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                ws[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[b].dstSet          = st->ds[m];
                ws[b].dstBinding      = b;
                ws[b].descriptorCount = 1;
                ws[b].descriptorType  = b < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                ws[b].pImageInfo      = &ii[b];
            }
            updateDescriptorSets(ctx_.device(), uint32_t(ws.size()), ws.data(), 0, nullptr);
        }
        return st;
    }

    void WakeFieldPipeline::initState(VkCommandBuffer cb, State& st) {
        for (uint32_t i = 0; i < 5; ++i) clearToGeneral(cb, st.image[i], st.layers);
        vkCmdFillBuffer(cb, st.table.handle, 0, VK_WHOLE_SIZE, 0u);
    }

    void WakeFieldPipeline::destroyState(State& st) {
        VkDevice d = ctx_.device();
        for (uint32_t i = 0; i < 5; ++i) {
            if (st.view[i])  vkDestroyImageView(d, st.view[i], nullptr);
            if (st.image[i]) vmaDestroyImage(ctx_.allocator(), st.image[i], st.alloc[i]);
            st.view[i]  = VK_NULL_HANDLE;
            st.image[i] = VK_NULL_HANDLE;
        }
        if (st.ds[0]) vkFreeDescriptorSets(d, descPool_, uint32_t(st.ds.size()), st.ds.data());
        st.ds = {};
        destroyBuffer(ctx_.allocator(), st.table);
    }

    void WakeFieldPipeline::record(VkCommandBuffer cb, State& st, const DisplacedMesh& dm, double nowSec) {
        // The first step, and one after the clock went back (a run started
        // over), take no time: they deposit and decay nothing away. Otherwise
        // dt is clamped so a stall cannot throw the field a quarter of a patch
        // away in one step. A frame at the same instant as the last one
        // changes nothing: the table and the images stay as that step left them.
        const double since = nowSec - st.prevTimeSec;
        const bool restart = st.prevTimeSec < 0.0 || since < 0.0;
        const float dt = restart ? 0.0f : std::min(static_cast<float>(since), 0.25f);
        if (!restart && dt <= 0.0f) return;
        st.prevTimeSec = nowSec;

        const DisplacedMesh::WakeField& wf = dm.wakeField;
        struct Upload {
            WakeTableGpu              table;
            DisplacedMesh::WakeSource sources[DisplacedMesh::kMaxWakeSources];
        };
        static_assert(sizeof(Upload) == kTableBytes, "the table and its sources go up in one piece");
        Upload up{};
        WakeTableGpu& tb = up.table;
        tb.patchCount     = st.layers;
        tb.res            = st.res;
        tb.timeSec        = static_cast<float>(std::fmod(nowSec, 4096.0));
        tb.dt             = dt;
        tb.foamLife       = std::max(wf.foamLife, 1e-3f);
        tb.foamThinning   = std::max(wf.foamThinning, 1.0f);
        tb.aerationLife   = std::max(wf.aerationLife, 1e-3f);
        tb.turbulenceLife = std::max(wf.turbulenceLife, 1e-3f);
        tb.laneLife       = std::max(wf.laneLife, 1e-3f);
        tb.velocityLife   = std::max(wf.velocityLife, 1e-3f);
        tb.eddyTexels     = std::max(wf.eddyTexels, 3.0f);// (a patch's own eddy size is in its record)
        tb.spread         = std::max(wf.spread, 0.0f);

        uint32_t liveMask = 0;
        for (uint32_t i = 0; i < st.layers; ++i) {
            const DisplacedMesh::WakePatch& p = dm.wakePatches[i];
            WakePatchGpu& g = tb.patches[i];
            if (!(p.size > 0.0f)) {
                st.prevSize[i] = 0.0f;// off: whatever the layer held is forgotten
                continue;
            }
            const float texel = p.size / static_cast<float>(st.res);
            g.originX = std::floor((p.centerX - 0.5f * p.size) / texel) * texel;
            g.originZ = std::floor((p.centerZ - 0.5f * p.size) / texel) * texel;
            g.size    = p.size;
            g.texel   = texel;
            // A patch that was off, or has another size, starts from still water.
            const bool history = st.prevSize[i] == p.size;
            g.prevOriginX = history ? st.prevOriginX[i] : g.originX;
            g.prevOriginZ = history ? st.prevOriginZ[i] : g.originZ;
            g.live        = history ? 2.0f : 1.0f;
            g.eddy        = std::max(p.eddy > 0.0f ? p.eddy : wf.eddyTexels * texel, 3.0f * texel);
            st.prevOriginX[i] = g.originX;
            st.prevOriginZ[i] = g.originZ;
            st.prevSize[i]    = p.size;
            liveMask |= 1u << i;
        }
        const uint32_t sourceCount = static_cast<uint32_t>(
                std::min<size_t>(dm.wakeSources.size(), DisplacedMesh::kMaxWakeSources));
        if (sourceCount > 0u) {
            std::memcpy(up.sources, dm.wakeSources.data(), sourceCount * sizeof(DisplacedMesh::WakeSource));
        }

        // The table, in the command stream: last frame's water shade may still
        // be reading the old one, and this frame's must read the new.
        VkBufferMemoryBarrier bmb{};
        bmb.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bmb.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        bmb.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb.buffer              = st.table.handle;
        bmb.size                = VK_WHOLE_SIZE;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
        vkCmdUpdateBuffer(cb, st.table.handle, 0,
                          sizeof(WakeTableGpu) + sourceCount * sizeof(DisplacedMesh::WakeSource), &up);
        bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
        if (liveMask == 0u) return;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        const uint32_t groups = (st.res + 7u) / 8u;
        PushConstants pc{};
        pc.tableAddr   = st.table.address;
        pc.sourceCount = sourceCount;
        for (uint32_t mode = 0; mode < 3; ++mode) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &st.ds[mode], 0, nullptr);
            pc.mode = mode;
            for (uint32_t i = 0; i < st.layers; ++i) {
                if (!(liveMask & (1u << i))) continue;
                pc.layer = i;
                vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, groups, groups, 1);
            }
            computeBarrier(cb);
        }
    }

}// namespace threepp::vulkan
