#include "threepp/renderers/vulkan/OverlayPass.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/InterleavedBufferAttribute.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Sprite.hpp"

// SPIR-V blobs shared with createOverlayPipeline (3D hybrid overlay) and
// createOrthoLinePipelines. const arrays have internal linkage in C++ so
// including in two TUs is safe.
#include "threepp/renderers/vulkan/shaders/overlay.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_point.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_point.vert.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_sprite.frag.spv.h"
#include "threepp/renderers/vulkan/shaders/overlay_sprite.vert.spv.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace threepp::vulkan {

// ─────────────────────────────────────────────────────────────────────────────
// Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

OverlayPass::OverlayPass(VulkanContext& ctx, uint32_t framesInFlight,
                         SampledImageCreator uploadFn)
    : ctx_(ctx), framesInFlight_(framesInFlight), uploadFn_(std::move(uploadFn)) {
    spriteDescPools_.resize(framesInFlight_, VK_NULL_HANDLE);
}

OverlayPass::~OverlayPass() {
    VkDevice d = ctx_.device();
    if (overlaySpritePipeline_)     vkDestroyPipeline(d, overlaySpritePipeline_, nullptr);
    if (spritePipelineLayout_)      vkDestroyPipelineLayout(d, spritePipelineLayout_, nullptr);
    if (orthoLineListPipeline_)     vkDestroyPipeline(d, orthoLineListPipeline_, nullptr);
    if (orthoLineStripPipeline_)    vkDestroyPipeline(d, orthoLineStripPipeline_, nullptr);
    if (orthoMeshPipeline_)         vkDestroyPipeline(d, orthoMeshPipeline_, nullptr);
    if (orthoMeshTransparentPipeline_) vkDestroyPipeline(d, orthoMeshTransparentPipeline_, nullptr);
    if (orthoPointListPipeline_)    vkDestroyPipeline(d, orthoPointListPipeline_, nullptr);
    if (orthoLinePipelineLayout_)   vkDestroyPipelineLayout(d, orthoLinePipelineLayout_, nullptr);
    if (spriteDescSetLayout_)       vkDestroyDescriptorSetLayout(d, spriteDescSetLayout_, nullptr);
    for (auto& pool : spriteDescPools_) {
        if (pool) vkDestroyDescriptorPool(d, pool, nullptr);
    }
    for (auto& [t, rec] : spriteAtlasCache_) {
        destroyImage2D(ctx_.allocator(), d, rec.image);
    }
    spriteAtlasCache_.clear();
    for (auto& [g, rec] : spriteGeomCache_) {
        destroyBuffer(ctx_.allocator(), rec.vertex);
        if (rec.index.handle != VK_NULL_HANDLE) destroyBuffer(ctx_.allocator(), rec.index);
    }
    spriteGeomCache_.clear();
    for (auto& [g, rec] : lineGeomCache_) {
        destroyBuffer(ctx_.allocator(), rec.vertex);
        if (rec.index.handle != VK_NULL_HANDLE) destroyBuffer(ctx_.allocator(), rec.index);
        if (rec.color.handle != VK_NULL_HANDLE) destroyBuffer(ctx_.allocator(), rec.color);
    }
    lineGeomCache_.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy pipeline setup
// ─────────────────────────────────────────────────────────────────────────────

void OverlayPass::createOrthoLinePipelines() {
    VkShaderModuleCreateInfo vsmci{};
    vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vsmci.codeSize = sizeof(kOverlayVertSpv);
    vsmci.pCode    = kOverlayVertSpv;
    VkShaderModule vertModule = VK_NULL_HANDLE;
    check(vkCreateShaderModule(ctx_.device(), &vsmci, nullptr, &vertModule),
          "vkCreateShaderModule(ortho overlay.vert)");
    VkShaderModuleCreateInfo fsmci{};
    fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fsmci.codeSize = sizeof(kOverlayFragSpv);
    fsmci.pCode    = kOverlayFragSpv;
    VkShaderModule fragModule = VK_NULL_HANDLE;
    check(vkCreateShaderModule(ctx_.device(), &fsmci, nullptr, &fragModule),
          "vkCreateShaderModule(ortho overlay.frag)");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName  = "main";

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

    VkPipelineInputAssemblyStateCreateInfo iaList{};
    iaList.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaList.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    VkPipelineInputAssemblyStateCreateInfo iaStrip = iaList;
    iaStrip.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;

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

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // No depth attachment in the ortho overlay pass → depth test off.
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState cbas{};
    cbas.blendEnable    = VK_FALSE;
    cbas.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &cbas;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    if (orthoLinePipelineLayout_ == VK_NULL_HANDLE) {
        VkPushConstantRange pcRange{};// mat4 mvp (64) + vec4 color (16) = 80
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 80;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &orthoLinePipelineLayout_),
              "vkCreatePipelineLayout(orthoLine)");
    }

    const VkFormat colorFmt = ctx_.swapchainFormat();
    VkPipelineRenderingCreateInfo prci{};
    prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &colorFmt;
    prci.depthAttachmentFormat   = VK_FORMAT_UNDEFINED;// color-only pass

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.pNext               = &prci;
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &iaList;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pDepthStencilState  = &ds;
    gpci.pColorBlendState    = &cb;
    gpci.pDynamicState       = &dyn;
    gpci.layout              = orthoLinePipelineLayout_;
    check(vkCreateGraphicsPipelines(ctx_.device(), VK_NULL_HANDLE, 1, &gpci, nullptr,
                                    &orthoLineListPipeline_),
          "vkCreateGraphicsPipelines(orthoLineList)");

    VkGraphicsPipelineCreateInfo gpciStrip = gpci;
    gpciStrip.pInputAssemblyState = &iaStrip;
    check(vkCreateGraphicsPipelines(ctx_.device(), VK_NULL_HANDLE, 1, &gpciStrip, nullptr,
                                    &orthoLineStripPipeline_),
          "vkCreateGraphicsPipelines(orthoLineStrip)");

    // Identical state to the line pipelines (position-only input,
    // depth off, CULL_NONE, same layout) but TRIANGLE_LIST topology;
    // the transparent variant adds standard src-alpha blending.
    VkPipelineInputAssemblyStateCreateInfo iaTri = iaList;
    iaTri.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkGraphicsPipelineCreateInfo gpciMesh = gpci;
    gpciMesh.pInputAssemblyState = &iaTri;
    check(vkCreateGraphicsPipelines(ctx_.device(), VK_NULL_HANDLE, 1, &gpciMesh, nullptr,
                                    &orthoMeshPipeline_),
          "vkCreateGraphicsPipelines(orthoMesh)");

    VkPipelineColorBlendAttachmentState cbasT = cbas;
    cbasT.blendEnable         = VK_TRUE;
    cbasT.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cbasT.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbasT.colorBlendOp        = VK_BLEND_OP_ADD;
    cbasT.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cbasT.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbasT.alphaBlendOp        = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo cbT = cb;
    cbT.pAttachments = &cbasT;
    VkGraphicsPipelineCreateInfo gpciMeshT = gpciMesh;
    gpciMeshT.pColorBlendState = &cbT;
    check(vkCreateGraphicsPipelines(ctx_.device(), VK_NULL_HANDLE, 1, &gpciMeshT, nullptr,
                                    &orthoMeshTransparentPipeline_),
          "vkCreateGraphicsPipelines(orthoMeshTransparent)");

    vkDestroyShaderModule(ctx_.device(), vertModule, nullptr);
    vkDestroyShaderModule(ctx_.device(), fragModule, nullptr);
}

void OverlayPass::createOrthoPointPipeline() {
    if (orthoPointListPipeline_ != VK_NULL_HANDLE) return;
    // Needs the shared layout (mat4 mvp + vec4 color) created by the line
    // pipelines; build those first if this is the first overlay primitive.
    if (orthoLinePipelineLayout_ == VK_NULL_HANDLE) createOrthoLinePipelines();

    VkShaderModuleCreateInfo vsmci{};
    vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vsmci.codeSize = sizeof(kOverlayPointVertSpv);
    vsmci.pCode    = kOverlayPointVertSpv;
    VkShaderModule vertModule = VK_NULL_HANDLE;
    check(vkCreateShaderModule(ctx_.device(), &vsmci, nullptr, &vertModule),
          "vkCreateShaderModule(overlay_point.vert)");
    VkShaderModuleCreateInfo fsmci{};
    fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fsmci.codeSize = sizeof(kOverlayPointFragSpv);
    fsmci.pCode    = kOverlayPointFragSpv;
    VkShaderModule fragModule = VK_NULL_HANDLE;
    check(vkCreateShaderModule(ctx_.device(), &fsmci, nullptr, &fragModule),
          "vkCreateShaderModule(overlay_point.frag)");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName  = "main";

    // Two bindings: position (loc 0) + per-vertex color (loc 1) — the point
    // shader requires both, mirroring the colored line variants.
    VkVertexInputBindingDescription vibs[2]{};
    vibs[0].binding = 0; vibs[0].stride = 3 * sizeof(float); vibs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    vibs[1].binding = 1; vibs[1].stride = 3 * sizeof(float); vibs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription vias[2]{};
    vias[0].location = 0; vias[0].binding = 0; vias[0].format = VK_FORMAT_R32G32B32_SFLOAT; vias[0].offset = 0;
    vias[1].location = 1; vias[1].binding = 1; vias[1].format = VK_FORMAT_R32G32B32_SFLOAT; vias[1].offset = 0;
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 2;
    vi.pVertexBindingDescriptions      = vibs;
    vi.vertexAttributeDescriptionCount = 2;
    vi.pVertexAttributeDescriptions    = vias;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

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

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState cbas{};
    cbas.blendEnable    = VK_FALSE;
    cbas.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &cbas;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    const VkFormat colorFmt = ctx_.swapchainFormat();
    VkPipelineRenderingCreateInfo prci{};
    prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &colorFmt;
    prci.depthAttachmentFormat   = VK_FORMAT_UNDEFINED;

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
    gpci.layout              = orthoLinePipelineLayout_;
    check(vkCreateGraphicsPipelines(ctx_.device(), VK_NULL_HANDLE, 1, &gpci, nullptr,
                                    &orthoPointListPipeline_),
          "vkCreateGraphicsPipelines(orthoPointList)");

    vkDestroyShaderModule(ctx_.device(), vertModule, nullptr);
    vkDestroyShaderModule(ctx_.device(), fragModule, nullptr);
}

void OverlayPass::createSpriteOverlayPipeline() {
    if (overlaySpritePipeline_ != VK_NULL_HANDLE) return;

    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 1;
    dslci.pBindings    = &b;
    check(vkCreateDescriptorSetLayout(ctx_.device(), &dslci, nullptr, &spriteDescSetLayout_),
          "vkCreateDescriptorSetLayout(spriteOverlay)");

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = 128;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &spriteDescSetLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &spritePipelineLayout_),
          "vkCreatePipelineLayout(spriteOverlay)");

    // Per-frame-in-flight descriptor pool. Reset before each ortho
    // overlay record so the prior frame's allocations return to the
    // pool without per-set vkFreeDescriptorSets calls.
    for (uint32_t f = 0; f < framesInFlight_; ++f) {
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = kMaxSpritesPerFrame;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kMaxSpritesPerFrame;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &spriteDescPools_[f]),
              "vkCreateDescriptorPool(spriteOverlay)");
    }

    VkShaderModuleCreateInfo vsmci{};
    vsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vsmci.codeSize = sizeof(kOverlaySpriteVertSpv);
    vsmci.pCode    = kOverlaySpriteVertSpv;
    VkShaderModule vert = VK_NULL_HANDLE;
    check(vkCreateShaderModule(ctx_.device(), &vsmci, nullptr, &vert),
          "vkCreateShaderModule(overlay_sprite.vert)");
    VkShaderModuleCreateInfo fsmci{};
    fsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fsmci.codeSize = sizeof(kOverlaySpriteFragSpv);
    fsmci.pCode    = kOverlaySpriteFragSpv;
    VkShaderModule frag = VK_NULL_HANDLE;
    check(vkCreateShaderModule(ctx_.device(), &fsmci, nullptr, &frag),
          "vkCreateShaderModule(overlay_sprite.frag)");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    // Sprite's InterleavedBuffer: 5 floats per vertex (pos.xyz at 0..2,
    // uv.xy at 3..4). One binding, two attributes.
    VkVertexInputBindingDescription vib{};
    vib.binding   = 0;
    vib.stride    = 5 * sizeof(float);
    vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription vias[2]{};
    vias[0].location = 0; vias[0].binding = 0;
    vias[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    vias[0].offset   = 0;
    vias[1].location = 1; vias[1].binding = 0;
    vias[1].format   = VK_FORMAT_R32G32_SFLOAT;
    vias[1].offset   = 3 * sizeof(float);
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

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    // Standard non-premultiplied alpha.
    VkPipelineColorBlendAttachmentState cbas{};
    cbas.blendEnable         = VK_TRUE;
    cbas.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cbas.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbas.colorBlendOp        = VK_BLEND_OP_ADD;
    cbas.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cbas.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbas.alphaBlendOp        = VK_BLEND_OP_ADD;
    cbas.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &cbas;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                   VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    // Dynamic rendering — color attachment only, no depth attachment.
    // HUD draws after the existing overlay pass so the swapchain has
    // already been transitioned through COLOR_ATTACHMENT_OPTIMAL.
    const VkFormat colorFmt = ctx_.swapchainFormat();
    VkPipelineRenderingCreateInfo prci{};
    prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &colorFmt;

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
    gpci.layout              = spritePipelineLayout_;
    check(vkCreateGraphicsPipelines(ctx_.device(), VK_NULL_HANDLE, 1, &gpci, nullptr,
                                    &overlaySpritePipeline_),
          "vkCreateGraphicsPipelines(overlaySprite)");

    vkDestroyShaderModule(ctx_.device(), vert, nullptr);
    vkDestroyShaderModule(ctx_.device(), frag, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Atlas / geometry cache helpers
// ─────────────────────────────────────────────────────────────────────────────

const OverlayPass::SpriteAtlasRec*
OverlayPass::ensureSpriteAtlasTexture(const std::shared_ptr<Texture>& texSp) {
    if (!texSp) return nullptr;
    const Texture* tex = texSp.get();
    Image& img = const_cast<Texture*>(tex)->image();
    const uint32_t w = img.width();
    const uint32_t h = img.height();
    if (w == 0 || h == 0) return nullptr;

    const unsigned int curVersion = tex->version();
    auto it = spriteAtlasCache_.find(tex);
    if (it != spriteAtlasCache_.end()) {
        SpriteAtlasRec& rec = it->second;
        const bool stale = rec.liveCheck.expired() ||
                           rec.liveCheck.lock().get() != tex ||
                           rec.textureVersion != curVersion ||
                           rec.width != w || rec.height != h;
        if (!stale) return &rec;
        // Stale — destroy and re-upload.
        vkDeviceWaitIdle(ctx_.device());
        destroyImage2D(ctx_.allocator(), ctx_.device(), rec.image);
        spriteAtlasCache_.erase(it);
    }

    // TextSprite / Font::rasterize emits RGBA8. Other sprite atlases
    // (image-file loaded textures) usually do too. Anything else
    // we don't bother supporting here — the user can manually
    // upload via the bindless texture path before render().
    std::vector<unsigned char> rgba;
    const size_t pixels = static_cast<size_t>(w) * h;
    try {
        auto& src = img.data<unsigned char>();
        if (src.size() == pixels * 4) {
            rgba.assign(src.begin(), src.end());
        } else if (src.size() == pixels * 3) {
            rgba.resize(pixels * 4);
            for (size_t i = 0; i < pixels; ++i) {
                rgba[i*4+0] = src[i*3+0];
                rgba[i*4+1] = src[i*3+1];
                rgba[i*4+2] = src[i*3+2];
                rgba[i*4+3] = 255u;
            }
        } else {
            return nullptr;
        }
    } catch (const std::bad_variant_access&) {
        return nullptr;
    }

    // Match GL/WGPU's colorSpace→format rule (`isSrgb = colorSpace
    // == sRGB`): ONLY an explicitly sRGB-tagged texture gets hardware
    // sRGB decode on sample. Linear AND NoColorSpace are sampled raw.
    // The TextSprite glyph atlas is NoColorSpace but Font::rasterize
    // bakes LINEAR bytes (color.r*255) into it — so it must be sampled
    // raw (UNORM), and overlay_sprite.frag applies the linear→sRGB
    // output encode for the UNORM swapchain. (Was: NoColorSpace fell
    // to SRGB here → double sRGB decode → dark non-white text.)
    const VkFormat fmt = (tex->colorSpace == ColorSpace::sRGB)
                                 ? VK_FORMAT_R8G8B8A8_SRGB
                                 : VK_FORMAT_R8G8B8A8_UNORM;
    char spriteName[64];
    std::snprintf(spriteName, sizeof(spriteName),
                  "spriteAtlas[%p]", static_cast<const void*>(tex));
    Image2D up = uploadFn_(
            w, h, fmt,
            rgba.data(), rgba.size(),
            VK_FILTER_LINEAR,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            spriteName);

    SpriteAtlasRec rec{};
    rec.image          = up;
    rec.textureVersion = curVersion;
    rec.width          = w;
    rec.height         = h;
    rec.liveCheck      = std::weak_ptr<Texture>(texSp);
    auto [ins, _] = spriteAtlasCache_.emplace(tex, std::move(rec));
    return &ins->second;
}

const OverlayPass::SpriteGeomRec*
OverlayPass::ensureSpriteGeometryUploaded(const BufferGeometry* geom) {
    if (!geom) return nullptr;
    auto it = spriteGeomCache_.find(geom);
    if (it != spriteGeomCache_.end()) return &it->second;

    // Sprite's geometry uses one InterleavedBuffer (5 floats /
    // vertex: pos.xyz at 0..2, uv.xy at 3..4). Pull the underlying
    // shared array via the interleaved attribute, not the position
    // attribute's stride-aware accessor, so we upload the full
    // interleaved 5-float layout the pipeline's vertex input
    // expects.
    auto* posAttr = geom->getAttribute<float>("position");
    const auto* idxAttr = geom->getIndex();
    if (!posAttr || !idxAttr) return nullptr;
    const auto* posIB = dynamic_cast<const InterleavedBufferAttribute*>(posAttr);
    if (!posIB) return nullptr;
    const auto& packed = posIB->data->array();
    if (packed.empty()) return nullptr;

    SpriteGeomRec rec{};
    const VkDeviceSize vbSize = packed.size() * sizeof(float);
    rec.vertex = createBuffer(
            ctx_.allocator(), ctx_.device(), vbSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    {
        void* mapped = nullptr;
        vmaMapMemory(ctx_.allocator(), rec.vertex.alloc, &mapped);
        std::memcpy(mapped, packed.data(), vbSize);
        vmaUnmapMemory(ctx_.allocator(), rec.vertex.alloc);
    }

    const auto& indices = idxAttr->array();
    std::vector<uint32_t> idx32(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        idx32[i] = static_cast<uint32_t>(indices[i]);
    }
    const VkDeviceSize ibSize = idx32.size() * sizeof(uint32_t);
    rec.index = createBuffer(
            ctx_.allocator(), ctx_.device(), ibSize,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    {
        void* mapped = nullptr;
        vmaMapMemory(ctx_.allocator(), rec.index.alloc, &mapped);
        std::memcpy(mapped, idx32.data(), ibSize);
        vmaUnmapMemory(ctx_.allocator(), rec.index.alloc);
    }
    rec.indexCount = static_cast<uint32_t>(idx32.size());

    auto [ins, _] = spriteGeomCache_.emplace(geom, std::move(rec));
    return &ins->second;
}

const LineRec*
OverlayPass::ensureLineGeometryUploaded(const BufferGeometry* geom) {
    if (!geom) return nullptr;
    auto posAttr = geom->getAttribute<float>("position");
    if (!posAttr || posAttr->count() == 0) return nullptr;
    auto* idxAttr = geom->getIndex();
    // Optional per-vertex color, used by AxesHelper-style overlays.
    const auto colAttr = geom->hasAttribute("color")
                                 ? geom->getAttribute<float>("color")
                                 : nullptr;

    const uint32_t posVer = posAttr->version;
    const uint32_t idxVer = (idxAttr && idxAttr->count() > 0) ? idxAttr->version : 0u;
    const uint32_t colVer = (colAttr && colAttr->count() > 0) ? colAttr->version : 0u;

    auto it = lineGeomCache_.find(geom);
    if (it != lineGeomCache_.end() && it->second.geomId != geom->id) {
        // Recycled pointer: this address was a DIFFERENT geometry whose
        // buffers we still hold. Retire them and re-upload from scratch
        // (the version fields would otherwise alias — both at 0).
        destroyBuffer(ctx_.allocator(), it->second.vertex);
        if (it->second.index.handle) destroyBuffer(ctx_.allocator(), it->second.index);
        if (it->second.color.handle) destroyBuffer(ctx_.allocator(), it->second.color);
        lineGeomCache_.erase(it);
        it = lineGeomCache_.end();
    }
    if (it != lineGeomCache_.end()) {
        auto& rec = it->second;
        rec.lastTouch = overlayFrameCounter_;
        if (rec.positionVersion == posVer &&
            rec.indexVersion    == idxVer &&
            rec.colorVersion    == colVer) {
            return &rec;
        }
        // Re-upload paths.
        const auto& posArr = posAttr->array();
        const VkDeviceSize vbBytes = posArr.size() * sizeof(float);
        if (vbBytes > rec.vertex.size) {
            destroyBuffer(ctx_.allocator(), rec.vertex);
            rec.vertex = createBuffer(
                    ctx_.allocator(), ctx_.device(), vbBytes,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        }
        void* mapped = nullptr;
        vmaMapMemory(ctx_.allocator(), rec.vertex.alloc, &mapped);
        std::memcpy(mapped, posArr.data(), vbBytes);
        vmaUnmapMemory(ctx_.allocator(), rec.vertex.alloc);
        rec.vertexCount     = static_cast<uint32_t>(posAttr->count());
        rec.positionVersion = posVer;

        if (idxAttr && idxAttr->count() > 0) {
            const auto& indices = idxAttr->array();
            const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
            if (rec.index.handle == VK_NULL_HANDLE || ibBytes > rec.index.size) {
                if (rec.index.handle != VK_NULL_HANDLE) destroyBuffer(ctx_.allocator(), rec.index);
                rec.index = createBuffer(
                        ctx_.allocator(), ctx_.device(), ibBytes,
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }
            vmaMapMemory(ctx_.allocator(), rec.index.alloc, &mapped);
            std::memcpy(mapped, indices.data(), ibBytes);
            vmaUnmapMemory(ctx_.allocator(), rec.index.alloc);
            rec.indexCount   = static_cast<uint32_t>(indices.size());
            rec.indexVersion = idxVer;
        } else if (rec.index.handle != VK_NULL_HANDLE) {
            destroyBuffer(ctx_.allocator(), rec.index);
            rec.index        = {};
            rec.indexCount   = 0;
            rec.indexVersion = 0;
        }

        // Color buffer follows the same in-place / recreate logic.
        if (colAttr && colAttr->count() > 0) {
            const auto& colArr = colAttr->array();
            const VkDeviceSize cbBytes = colArr.size() * sizeof(float);
            if (rec.color.handle == VK_NULL_HANDLE || cbBytes > rec.color.size) {
                if (rec.color.handle != VK_NULL_HANDLE) destroyBuffer(ctx_.allocator(), rec.color);
                rec.color = createBuffer(
                        ctx_.allocator(), ctx_.device(), cbBytes,
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }
            vmaMapMemory(ctx_.allocator(), rec.color.alloc, &mapped);
            std::memcpy(mapped, colArr.data(), cbBytes);
            vmaUnmapMemory(ctx_.allocator(), rec.color.alloc);
            rec.colorVersion = colVer;
        } else if (rec.color.handle != VK_NULL_HANDLE) {
            destroyBuffer(ctx_.allocator(), rec.color);
            rec.color        = {};
            rec.colorVersion = 0;
        }
        return &rec;
    }

    // First-time upload.
    const auto& posArr = posAttr->array();
    LineRec rec{};
    rec.vertexCount     = static_cast<uint32_t>(posAttr->count());
    rec.positionVersion = posVer;
    rec.geomId          = geom->id;
    rec.lastTouch       = overlayFrameCounter_;

    const VkDeviceSize vbBytes = posArr.size() * sizeof(float);
    rec.vertex = createBuffer(
            ctx_.allocator(), ctx_.device(), vbBytes,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    void* mapped = nullptr;
    vmaMapMemory(ctx_.allocator(), rec.vertex.alloc, &mapped);
    std::memcpy(mapped, posArr.data(), vbBytes);
    vmaUnmapMemory(ctx_.allocator(), rec.vertex.alloc);

    if (idxAttr && idxAttr->count() > 0) {
        const auto& indices = idxAttr->array();
        rec.indexCount   = static_cast<uint32_t>(indices.size());
        rec.indexVersion = idxVer;
        const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
        rec.index = createBuffer(
                ctx_.allocator(), ctx_.device(), ibBytes,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        vmaMapMemory(ctx_.allocator(), rec.index.alloc, &mapped);
        std::memcpy(mapped, indices.data(), ibBytes);
        vmaUnmapMemory(ctx_.allocator(), rec.index.alloc);
    }

    if (colAttr && colAttr->count() > 0) {
        const auto& colArr = colAttr->array();
        rec.colorVersion = colVer;
        const VkDeviceSize cbBytes = colArr.size() * sizeof(float);
        rec.color = createBuffer(
                ctx_.allocator(), ctx_.device(), cbBytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        vmaMapMemory(ctx_.allocator(), rec.color.alloc, &mapped);
        std::memcpy(mapped, colArr.data(), cbBytes);
        vmaUnmapMemory(ctx_.allocator(), rec.color.alloc);
    }

    return &lineGeomCache_.emplace(geom, std::move(rec)).first->second;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main record entry point
// ─────────────────────────────────────────────────────────────────────────────

void OverlayPass::record(VkCommandBuffer cb, uint32_t frame, uint32_t imageIndex,
                          Object3D& scene, Camera& camera, bool screenSpaceOnly,
                          uint32_t regionX, uint32_t regionY,
                          uint32_t regionW, uint32_t regionH) {
    // Lazy pipeline setup on first HUD overlay of the program.
    if (overlaySpritePipeline_ == VK_NULL_HANDLE) {
        createSpriteOverlayPipeline();
    }
    // Ensure scene + camera matrices are current — the user's HUD
    // class typically only updates the camera's projection on
    // resize and the sprite positions per-frame; matrixWorld
    // computation here keeps us self-contained (no implicit
    // requirement that the user call updateMatrixWorld).
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);

    // Advance the overlay-frame clock and evict line/mesh geometry
    // buffers untouched for longer than the in-flight window. Overlays
    // that rebuild transient geometry every frame (e.g. a detection
    // box overlay calling makeBoxLines per detection) would otherwise
    // leave a dead cache entry per geometry forever. The margin (>
    // frames-in-flight) guarantees no evicted buffer is still
    // referenced by an in-flight command buffer.
    ++overlayFrameCounter_;
    if (overlayFrameCounter_ > 8) {
        const uint64_t cutoff = overlayFrameCounter_ - 8;
        for (auto it = lineGeomCache_.begin(); it != lineGeomCache_.end();) {
            if (it->second.lastTouch < cutoff) {
                destroyBuffer(ctx_.allocator(), it->second.vertex);
                if (it->second.index.handle) destroyBuffer(ctx_.allocator(), it->second.index);
                if (it->second.color.handle) destroyBuffer(ctx_.allocator(), it->second.color);
                it = lineGeomCache_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Reset the per-frame descriptor pool before allocating this
    // frame's sprite sets. Safe because the inFlight fence wait
    // at frame start already guarantees the prior frame's sets
    // aren't bound on any in-flight cmd buffer.
    check(vkResetDescriptorPool(ctx_.device(),
                                spriteDescPools_[frame], 0),
          "vkResetDescriptorPool(sprite)");

    // Collect visible sprites. traverseVisible skips invisible
    // subtrees the same way the PT walk does. effWorld captures the
    // world matrix used for the draw — for screenSpace sprites we
    // synthesise this from screenAnchor + viewport + position
    // instead of using sp->matrixWorld (which carries the user's
    // 3D placement, irrelevant in screen-space mode).
    const VkExtent2D extEarly = ctx_.swapchainExtent();
    struct SpriteDraw {
        Sprite* sprite = nullptr;
        std::shared_ptr<Texture> atlas;
        float color[4]{1.f,1.f,1.f,1.f};
        float rotation = 0.f;
        Matrix4 effWorld;
    };
    std::vector<SpriteDraw> draws;
    scene.traverseVisible([&](Object3D& o) {
        auto* sp = dynamic_cast<Sprite*>(&o);
        if (!sp) return;
        if (screenSpaceOnly && !sp->screenSpace) return;
        auto mat = sp->material();
        if (!mat) return;
        auto* mm = dynamic_cast<MaterialWithMap*>(mat.get());
        if (!mm || !mm->map) return;
        SpriteDraw d;
        d.sprite = sp;
        d.atlas  = mm->map;
        if (auto* mc = dynamic_cast<MaterialWithColor*>(mat.get())) {
            d.color[0] = mc->color.r;
            d.color[1] = mc->color.g;
            d.color[2] = mc->color.b;
        }
        d.color[3] = mat->opacity;
        if (auto* mr = dynamic_cast<MaterialWithRotation*>(mat.get())) {
            d.rotation = mr->rotation;
        }
        if (sp->screenSpace) {
            // Pixel position = anchor·viewport + position.xy.
            // Negative offsets read as "from the opposite edge".
            const float pxX = sp->screenAnchor.x * float(extEarly.width)
                            + static_cast<float>(sp->position.x);
            const float pxY = sp->screenAnchor.y * float(extEarly.height)
                            + static_cast<float>(sp->position.y);
            // Compose: T(pxX,pxY,0) * R(quaternion) * S(scale). Reuse
            // the sprite's own quaternion + scale (TextSprite::applyScale
            // packs glyph-atlas pixel dimensions into scale.xy).
            d.effWorld.compose(Vector3(pxX, pxY, 0.f),
                               sp->quaternion,
                               sp->scale);
        } else {
            std::memcpy(d.effWorld.elements.data(),
                        sp->matrixWorld->elements.data(), 64);
        }
        draws.push_back(std::move(d));
    });

    // Collect Line / LineSegments for the ortho overlay (the deferred
    // "Mesh / Line HUD overlay" follow-up). ensureSceneBuilt is skipped
    // for ortho cameras, so lastVisibleLines_ isn't populated — gather
    // straight from the scene here.
    //
    // ONLY in the true ortho/HUD path (screenSpaceOnly == false). The
    // screenSpaceOnly == true call comes from the perspective path
    // (beginFrameForPT, compositing screen-space Sprites over the PT
    // image): a perspective scene's Lines are already drawn correctly by
    // the 3D hybrid overlay, so collecting them here would double-draw
    // them through the internal screen-space camera at wrong positions.
    struct OrthoLineDraw {
        Line* line = nullptr;
        bool  isSegments = false;
        Matrix4 world;
    };
    std::vector<OrthoLineDraw> lineDraws;
    if (!screenSpaceOnly) {
        scene.traverseVisible([&](Object3D& o) {
            auto* ln = dynamic_cast<Line*>(&o);
            if (!ln) return;
            auto g = ln->geometry();
            if (!g || !g->hasAttribute("position")) return;
            OrthoLineDraw ld;
            ld.line = ln;
            ld.isSegments = (dynamic_cast<LineSegments*>(ln) != nullptr);
            std::memcpy(ld.world.elements.data(), ln->matrixWorld->elements.data(), 64);
            lineDraws.push_back(ld);
        });
    }

    // Collect filled Mesh overlays (flat MeshBasicMaterial-style fills,
    // e.g. SVG ShapeGeometry HUD art). Same gating as lines: only the
    // explicit ortho/HUD render, never the PT screen-space auto-pass
    // (which would wrongly flatten the path-traced 3D scene's meshes).
    struct OrthoMeshDraw {
        Mesh*   mesh = nullptr;
        Matrix4 world;
        Color   color{1.f, 1.f, 1.f};
        float   opacity = 1.f;
        bool    transparent = false;
    };
    std::vector<OrthoMeshDraw> meshDraws;
    if (!screenSpaceOnly) {
        scene.traverseVisible([&](Object3D& o) {
            auto* m = dynamic_cast<Mesh*>(&o);
            if (!m) return;// Sprites/Lines aren't Meshes — handled above
            auto g = m->geometry();
            if (!g || !g->hasAttribute("position")) return;
            auto mat = m->material();
            if (!mat) return;
            OrthoMeshDraw md;
            md.mesh = m;
            std::memcpy(md.world.elements.data(), m->matrixWorld->elements.data(), 64);
            if (auto* mc = dynamic_cast<MaterialWithColor*>(mat.get())) md.color = mc->color;
            md.opacity     = mat->opacity;
            md.transparent = mat->transparent;
            meshDraws.push_back(md);
        });
    }

    // Collect Points objects (point clouds). POINT_LIST topology; the push
    // constant's color.w carries PointsMaterial::size (pixels). Requires a
    // per-vertex "color" attribute — the point pipeline always reads binding 1.
    // Same ortho/HUD gating as lines/meshes (never the PT screen-space pass).
    struct OrthoPointDraw {
        Points* points = nullptr;
        Matrix4 world;
        Color   color{1.f, 1.f, 1.f};
        float   size = 3.f;
    };
    std::vector<OrthoPointDraw> pointDraws;
    if (!screenSpaceOnly) {
        scene.traverseVisible([&](Object3D& o) {
            auto* p = dynamic_cast<Points*>(&o);
            if (!p) return;
            auto g = p->geometry();
            if (!g || !g->hasAttribute("position") || !g->hasAttribute("color")) return;
            OrthoPointDraw pd;
            pd.points = p;
            std::memcpy(pd.world.elements.data(), p->matrixWorld->elements.data(), 64);
            if (auto mat = p->material()) {
                if (auto* mc = dynamic_cast<MaterialWithColor*>(mat.get())) pd.color = mc->color;
                if (auto* ms = dynamic_cast<MaterialWithSize*>(mat.get())) pd.size = std::max(1.f, ms->size);
            }
            pointDraws.push_back(pd);
        });
    }

    if (draws.empty() && lineDraws.empty() && meshDraws.empty() && pointDraws.empty()) return;
    if (draws.size() > kMaxSpritesPerFrame) {
        std::cerr << "[VulkanRenderer] HUD sprite count " << draws.size()
                  << " exceeds kMaxSpritesPerFrame=" << kMaxSpritesPerFrame
                  << "; extras dropped\n";
        draws.resize(kMaxSpritesPerFrame);
    }

    const VkExtent2D ext = ctx_.swapchainExtent();

    // Open a fresh dynamic render pass on the swapchain so we can
    // emit draw commands. The hybrid 3D overlay pass earlier in
    // recordCommandBuffer leaves the swapchain in GENERAL; we
    // transition GENERAL → COLOR_ATTACHMENT_OPTIMAL, draw, then
    // transition back to GENERAL so endFrame's overlay/present
    // logic sees the layout it expects.
    const VkImage img = ctx_.swapchainImages()[imageIndex];
    {
        VkImageMemoryBarrier2 toColor{};
        toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toColor.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toColor.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
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
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &toColor;
        vkCmdPipelineBarrier2(cb, &dep);
    }

    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView   = ctx_.swapchainImageViews()[imageIndex];
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.offset = {0, 0};
    ri.renderArea.extent = ext;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    vkCmdBeginRendering(cb, &ri);

    // Split-screen: clip to the region sub-rect (regionW == 0 → full frame).
    const bool   regionActive = regionW > 0u && regionH > 0u;
    const float  rgx = regionActive ? float(regionX) : 0.f;
    const float  rgy = regionActive ? float(regionY) : 0.f;
    const float  rgw = regionActive ? float(regionW) : float(ext.width);
    const float  rgh = regionActive ? float(regionH) : float(ext.height);
    VkViewport vp{rgx, rgy, rgw, rgh, 0.f, 1.f};
    vkCmdSetViewport(cb, 0, 1, &vp);
    VkRect2D sc{{int32_t(rgx), int32_t(rgy)}, {uint32_t(rgw), uint32_t(rgh)}};
    vkCmdSetScissor(cb, 0, 1, &sc);

    // ── Filled Mesh overlays ────────────────────────────────────────
    // Drawn FIRST so Sprites/TextSprites composite on top of the vector
    // art (panels behind labels). Uses the same flat-color overlay
    // shader + ortho MVP (with GL→Vulkan clip-z remap) as the lines.
    if (!meshDraws.empty()) {
        if (orthoMeshPipeline_ == VK_NULL_HANDLE) createOrthoLinePipelines();
        Matrix4 zfix;
        zfix.set(1.f, 0.f, 0.f, 0.f,
                 0.f, 1.f, 0.f, 0.f,
                 0.f, 0.f, 0.5f, 0.5f,
                 0.f, 0.f, 0.f, 1.f);
        Matrix4 vpMat;
        vpMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        Matrix4 cvp;
        cvp.multiplyMatrices(zfix, vpMat);

        struct MeshPC {
            float mvp[16];
            float color[4];
        };
        VkPipeline curMesh = VK_NULL_HANDLE;
        for (const auto& md : meshDraws) {
            const LineRec* rec = ensureLineGeometryUploaded(md.mesh->geometry().get());
            if (!rec || rec->vertex.handle == VK_NULL_HANDLE) continue;

            VkPipeline want = md.transparent ? orthoMeshTransparentPipeline_ : orthoMeshPipeline_;
            if (want != curMesh) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                curMesh = want;
            }

            Matrix4 mvp;
            mvp.multiplyMatrices(cvp, md.world);
            MeshPC pc{};
            std::memcpy(pc.mvp, mvp.elements.data(), 64);
            pc.color[0] = md.color.r;
            pc.color[1] = md.color.g;
            pc.color[2] = md.color.b;
            pc.color[3] = md.opacity;
            vkCmdPushConstants(cb, orthoLinePipelineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            VkBuffer     vb[1] = {rec->vertex.handle};
            VkDeviceSize vo[1] = {0};
            vkCmdBindVertexBuffers(cb, 0, 1, vb, vo);
            if (rec->index.handle != VK_NULL_HANDLE) {
                vkCmdBindIndexBuffer(cb, rec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cb, rec->indexCount, 1, 0, 0, 0);
            } else {
                vkCmdDraw(cb, rec->vertexCount, 1, 0, 0);
            }
        }
    }

    // ── Sprite overlay ──────────────────────────────────────────────
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, overlaySpritePipeline_);

    // Per-frame projection (ortho) extracted from the camera. matrixWorldInverse
    // is the view matrix; we precompute modelView for each sprite below.
    Matrix4 projMat;
    std::memcpy(projMat.elements.data(),
                camera.projectionMatrix.elements.data(), 64);

    // 128-byte push constant layout, matches overlay_sprite.vert/frag.
    struct SpritePC {
        float projection[16];   // 64
        float mvPos[4];         // 16
        float scale[2];         // 8
        float center[2];        // 8
        float color[4];         // 16
        float rotation;         // 4
        float _pad[3];          // 12
    };
    static_assert(sizeof(SpritePC) == 128,
                  "SpritePC must match push-constant layout");

    for (const auto& d : draws) {
        const auto* atlas = ensureSpriteAtlasTexture(d.atlas);
        if (!atlas) continue;

        // Per-sprite descriptor set bound to the atlas texture.
        VkDescriptorSetAllocateInfo asi{};
        asi.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        asi.descriptorPool     = spriteDescPools_[frame];
        asi.descriptorSetCount = 1;
        asi.pSetLayouts        = &spriteDescSetLayout_;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(ctx_.device(), &asi, &set) != VK_SUCCESS) continue;
        VkDescriptorImageInfo dii{};
        dii.imageView   = atlas->image.view;
        dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        dii.sampler     = atlas->image.sampler;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = set;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &dii;
        vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                spritePipelineLayout_, 0, 1, &set, 0, nullptr);

        Sprite* sp = d.sprite;
        // modelView = viewInverse * effWorld. effWorld is either
        // sp->matrixWorld (3D) or a screen-space composite built
        // from screenAnchor + viewport + position (screen-space).
        Matrix4 modelView;
        modelView.multiplyMatrices(camera.matrixWorldInverse, d.effWorld);
        Vector3 worldScale;
        worldScale.setFromMatrixScale(d.effWorld);
        Vector3 mvPos;
        mvPos.setFromMatrixPosition(modelView);

        SpritePC pc{};
        std::memcpy(pc.projection, projMat.elements.data(), 64);
        pc.mvPos[0] = static_cast<float>(mvPos.x);
        pc.mvPos[1] = static_cast<float>(mvPos.y);
        pc.mvPos[2] = static_cast<float>(mvPos.z);
        pc.mvPos[3] = 1.f;
        pc.scale[0] = static_cast<float>(worldScale.x);
        pc.scale[1] = static_cast<float>(worldScale.y);
        pc.center[0] = sp->center.x;
        pc.center[1] = sp->center.y;
        pc.color[0] = d.color[0];
        pc.color[1] = d.color[1];
        pc.color[2] = d.color[2];
        pc.color[3] = d.color[3];
        pc.rotation = d.rotation;

        vkCmdPushConstants(cb, spritePipelineLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(pc), &pc);

        // Sprite geometry: shared 4-vertex interleaved quad + 6-index
        // buffer. Upload through the existing line-geometry helper —
        // it doesn't know about Lines specifically, it just stages
        // a position+optional-color vertex buffer. We need a slightly
        // different uploader for the 5-float interleaved layout.
        const BufferGeometry* geom = sp->geometry().get();
        const SpriteGeomRec* gr = ensureSpriteGeometryUploaded(geom);
        if (!gr || gr->vertex.handle == VK_NULL_HANDLE) continue;

        VkBuffer     vbufs[1] = {gr->vertex.handle};
        VkDeviceSize voffs[1] = {0};
        vkCmdBindVertexBuffers(cb, 0, 1, vbufs, voffs);
        vkCmdBindIndexBuffer(cb, gr->index.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, gr->indexCount, 1, 0, 0, 0);
    }

    // ── Line / LineSegments overlay ─────────────────────────────────
    // Drawn after the sprites in the same color-only pass, so boxes /
    // gizmos land on top of any image sprite.
    if (!lineDraws.empty()) {
        if (orthoLineListPipeline_ == VK_NULL_HANDLE) createOrthoLinePipelines();

        // overlay.vert flips Y but does NOT remap GL clip-z to Vulkan's
        // [0,w] range; bake that remap (z' = 0.5z + 0.5w) into the MVP so
        // the shader's standard MVP multiply is correct.
        Matrix4 zfix;
        zfix.set(1.f, 0.f, 0.f, 0.f,
                 0.f, 1.f, 0.f, 0.f,
                 0.f, 0.f, 0.5f, 0.5f,
                 0.f, 0.f, 0.f, 1.f);
        Matrix4 vpMat;
        vpMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        Matrix4 cvp;
        cvp.multiplyMatrices(zfix, vpMat);

        struct LinePC {
            float mvp[16];
            float color[4];
        };
        VkPipeline curLine = VK_NULL_HANDLE;
        for (const auto& ld : lineDraws) {
            auto g = ld.line->geometry();
            const LineRec* lrec = ensureLineGeometryUploaded(g.get());
            if (!lrec || lrec->vertex.handle == VK_NULL_HANDLE) continue;

            Color color(1.f, 1.f, 1.f);
            float opacity = 1.f;
            if (auto m = ld.line->material()) {
                if (auto* mc = dynamic_cast<MaterialWithColor*>(m.get())) color = mc->color;
                opacity = m->opacity;
            }

            VkPipeline want = ld.isSegments ? orthoLineListPipeline_ : orthoLineStripPipeline_;
            if (want != curLine) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
                curLine = want;
            }

            Matrix4 mvp;
            mvp.multiplyMatrices(cvp, ld.world);
            LinePC pc{};
            std::memcpy(pc.mvp, mvp.elements.data(), 64);
            pc.color[0] = color.r;
            pc.color[1] = color.g;
            pc.color[2] = color.b;
            pc.color[3] = opacity;
            vkCmdPushConstants(cb, orthoLinePipelineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            VkBuffer     vb[1] = {lrec->vertex.handle};
            VkDeviceSize vo[1] = {0};
            vkCmdBindVertexBuffers(cb, 0, 1, vb, vo);

            const auto& dr = g->drawRange;
            if (lrec->index.handle != VK_NULL_HANDLE) {
                vkCmdBindIndexBuffer(cb, lrec->index.handle, 0, VK_INDEX_TYPE_UINT32);
                const uint32_t start = static_cast<uint32_t>(std::max(0, dr.start));
                const uint32_t cap   = (lrec->indexCount > start) ? (lrec->indexCount - start) : 0u;
                const uint32_t cnt   = std::min(cap, static_cast<uint32_t>(std::max(0, dr.count)));
                if (cnt > 0) vkCmdDrawIndexed(cb, cnt, 1, start, 0, 0);
            } else {
                const uint32_t start = static_cast<uint32_t>(std::max(0, dr.start));
                const uint32_t cap   = (lrec->vertexCount > start) ? (lrec->vertexCount - start) : 0u;
                const uint32_t cnt   = std::min(cap, static_cast<uint32_t>(std::max(0, dr.count)));
                if (cnt > 0) vkCmdDraw(cb, cnt, 1, start, 0);
            }
        }
    }

    // ── Points overlay (point clouds) ───────────────────────────────
    // Drawn last so dense scan/map clouds composite over grid + gizmos.
    // Mirrors the inline primary-scene point path: POINT_LIST pipeline,
    // pos (binding 0) + color (binding 1), color.w = point size in pixels.
    if (!pointDraws.empty()) {
        if (orthoPointListPipeline_ == VK_NULL_HANDLE) createOrthoPointPipeline();

        Matrix4 zfix;
        zfix.set(1.f, 0.f, 0.f, 0.f,
                 0.f, 1.f, 0.f, 0.f,
                 0.f, 0.f, 0.5f, 0.5f,
                 0.f, 0.f, 0.f, 1.f);
        Matrix4 vpMat;
        vpMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        Matrix4 cvp;
        cvp.multiplyMatrices(zfix, vpMat);

        struct PointPC {
            float mvp[16];
            float color[4];// .rgb = tint, .w = point size (pixels)
        };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, orthoPointListPipeline_);
        for (const auto& pd : pointDraws) {
            auto g = pd.points->geometry();
            const LineRec* prec = ensureLineGeometryUploaded(g.get());
            if (!prec || prec->vertex.handle == VK_NULL_HANDLE) continue;
            if (prec->color.handle == VK_NULL_HANDLE) continue;// pipeline reads binding 1

            Matrix4 mvp;
            mvp.multiplyMatrices(cvp, pd.world);
            PointPC pc{};
            std::memcpy(pc.mvp, mvp.elements.data(), 64);
            pc.color[0] = pd.color.r;
            pc.color[1] = pd.color.g;
            pc.color[2] = pd.color.b;
            pc.color[3] = pd.size;
            vkCmdPushConstants(cb, orthoLinePipelineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            VkBuffer     vb[2] = {prec->vertex.handle, prec->color.handle};
            VkDeviceSize vo[2] = {0, 0};
            vkCmdBindVertexBuffers(cb, 0, 2, vb, vo);

            const auto& dr = g->drawRange;
            const uint32_t start = static_cast<uint32_t>(std::max(0, dr.start));
            const uint32_t cap   = (prec->vertexCount > start) ? (prec->vertexCount - start) : 0u;
            const uint32_t cnt   = std::min(cap, static_cast<uint32_t>(std::max(0, dr.count)));
            if (cnt > 0) vkCmdDraw(cb, cnt, 1, start, 0);
        }
    }

    vkCmdEndRendering(cb);

    // Back to GENERAL for endFrame's overlay/present transition.
    {
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
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cb, &dep);
    }
}

}// namespace threepp::vulkan
