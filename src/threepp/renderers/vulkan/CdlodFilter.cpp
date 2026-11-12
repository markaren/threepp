#include "threepp/renderers/vulkan/CdlodFilter.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/renderers/vulkan/shaders/cdlod_filter.comp.spv.h"
#include <algorithm>
#include <stdexcept>

namespace threepp::vulkan {
CdlodFilter::CdlodFilter(VulkanContext& ctx,const water::OceanImage& h,const water::OceanImage& d):ctx_(ctx),dim_(h.width) {
    for(uint32_t n=dim_;n>1;n>>=1)++levels_;
    VkFormatProperties props{};vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice(),VK_FORMAT_R32G32B32A32_SFLOAT,&props);
    const VkFormatFeatureFlags required=VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT|VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if((props.optimalTilingFeatures&required)!=required)throw std::runtime_error("CDLOD RGBA32F linear filter unavailable");
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType=VK_IMAGE_TYPE_2D;ici.format=VK_FORMAT_R32G32B32A32_SFLOAT;ici.extent={dim_,dim_,1};
    ici.mipLevels=levels_;ici.arrayLayers=1;ici.samples=VK_SAMPLE_COUNT_1_BIT;ici.tiling=VK_IMAGE_TILING_OPTIMAL;
    ici.usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};aci.usage=VMA_MEMORY_USAGE_AUTO;
    check(vmaCreateImage(ctx.allocator(),&ici,&aci,&image_,&alloc_,nullptr),"create CDLOD pyramid");
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=image_;vi.viewType=VK_IMAGE_VIEW_TYPE_2D;vi.format=ici.format;
    vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,levels_,0,1};
    check(createImageView(ctx.device(),&vi,nullptr,&view_),"CDLOD pyramid view");
    vi.subresourceRange.levelCount=1;check(createImageView(ctx.device(),&vi,nullptr,&target_),"CDLOD target view");
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.magFilter=si.minFilter=VK_FILTER_LINEAR;
    si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT;si.maxLod=VK_LOD_CLAMP_NONE;
    check(vkCreateSampler(ctx.device(),&si,nullptr,&sampler_),"CDLOD sampler");
    VkDescriptorSetLayoutBinding b[3]{};
    for(unsigned i=0;i<3;++i){b[i].binding=i;b[i].descriptorCount=1;b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;b[i].descriptorType=i==2?VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;}
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=3;dl.pBindings=b;
    check(vkCreateDescriptorSetLayout(ctx.device(),&dl,nullptr,&dsl_),"CDLOD descriptor layout");
    VkDescriptorPoolSize sizes[2]={{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1}};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=2;dp.pPoolSizes=sizes;
    check(vkCreateDescriptorPool(ctx.device(),&dp,nullptr,&pool_),"CDLOD pool");
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=pool_;da.descriptorSetCount=1;da.pSetLayouts=&dsl_;
    check(vkAllocateDescriptorSets(ctx.device(),&da,&ds_),"CDLOD descriptor set");
    VkDescriptorImageInfo ii[3]={{sampler_,h.view,VK_IMAGE_LAYOUT_GENERAL},{sampler_,d.view,VK_IMAGE_LAYOUT_GENERAL},{VK_NULL_HANDLE,target_,VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[3]{};for(unsigned i=0;i<3;++i){w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[i].dstSet=ds_;w[i].dstBinding=i;w[i].descriptorCount=1;w[i].descriptorType=b[i].descriptorType;w[i].pImageInfo=&ii[i];}
    updateDescriptorSets(ctx.device(),3,w,0,nullptr);
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT,0,12};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&dsl_;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&pr;
    check(vkCreatePipelineLayout(ctx.device(),&pl,nullptr,&layout_),"CDLOD pipeline layout");
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=sizeof(kCdlodFilterCompSpv);sm.pCode=kCdlodFilterCompSpv;VkShaderModule mod;
    check(vkCreateShaderModule(ctx.device(),&sm,nullptr,&mod),"CDLOD filter shader");
    VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pi.layout=layout_;pi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pi.stage.module=mod;pi.stage.pName="main";
    check(vkCreateComputePipelines(ctx.device(),ctx.pipelineCache(),1,&pi,nullptr,&pipeline_),"CDLOD filter pipeline");vkDestroyShaderModule(ctx.device(),mod,nullptr);
}
CdlodFilter::~CdlodFilter(){auto d=ctx_.device();vkDestroyPipeline(d,pipeline_,nullptr);vkDestroyPipelineLayout(d,layout_,nullptr);vkDestroyDescriptorPool(d,pool_,nullptr);vkDestroyDescriptorSetLayout(d,dsl_,nullptr);vkDestroySampler(d,sampler_,nullptr);vkDestroyImageView(d,target_,nullptr);vkDestroyImageView(d,view_,nullptr);vmaDestroyImage(ctx_.allocator(),image_,alloc_);}
void CdlodFilter::record(VkCommandBuffer cb,float tileSize,float waveScale) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.image=image_;b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,levels_,0,1};
    b.oldLayout=initialized_?VK_IMAGE_LAYOUT_GENERAL:VK_IMAGE_LAYOUT_UNDEFINED;b.newLayout=VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask=initialized_?VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT:0;b.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
    VkMemoryBarrier source{VK_STRUCTURE_TYPE_MEMORY_BARRIER};source.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;source.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&source,0,nullptr,1,&b);
    struct {uint32_t dim;float tile,scale;} pc{dim_,tileSize,waveScale};
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_);vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,layout_,0,1,&ds_,0,nullptr);vkCmdPushConstants(cb,layout_,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),&pc);vkCmdDispatch(cb,(dim_+7)/8,(dim_+7)/8,1);
    b.oldLayout=VK_IMAGE_LAYOUT_GENERAL;b.subresourceRange.levelCount=1;
    for(uint32_t l=1;l<levels_;++l){
        b.subresourceRange.baseMipLevel=l-1;b.srcAccessMask=l==1?VK_ACCESS_SHADER_WRITE_BIT:VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb,l==1?VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT:VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
        VkImageBlit bl{};bl.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,l-1,0,1};bl.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,l,0,1};
        int n=int(std::max(1u,dim_>>(l-1))),m=int(std::max(1u,dim_>>l));bl.srcOffsets[1]={n,n,1};bl.dstOffsets[1]={m,m,1};
        vkCmdBlitImage(cb,image_,VK_IMAGE_LAYOUT_GENERAL,image_,VK_IMAGE_LAYOUT_GENERAL,1,&bl,VK_FILTER_LINEAR);
    }
    b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,levels_,0,1};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);initialized_=true;
}
}
