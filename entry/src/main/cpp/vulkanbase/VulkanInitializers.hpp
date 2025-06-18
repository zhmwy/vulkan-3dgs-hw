#pragma once

#include <vector>
#include "vulkan/vulkan.h"

namespace vkbase
{
    inline VkMemoryAllocateInfo vkMemoryAllocateInfo()
    {
        VkMemoryAllocateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        return info;
    }

    inline VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo(
        VkCommandPool commandPool, 
        VkCommandBufferLevel level, 
        uint32_t bufferCount)
    {
        VkCommandBufferAllocateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = commandPool;
        info.level = level;
        info.commandBufferCount = bufferCount;
        return info;
    }

    inline VkCommandBufferBeginInfo vkCommandBufferBeginInfo()
    {
        VkCommandBufferBeginInfo info {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        return info;
    }

    inline VkImageMemoryBarrier vkImageMemoryBarrier()
    {
        VkImageMemoryBarrier barrier {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        return barrier;
    }

    inline VkImageCreateInfo vkImageCreateInfo()
    {
        VkImageCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        return info;
    }

    inline VkSemaphoreCreateInfo vkSemaphoreCreateInfo()
    {
        VkSemaphoreCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        return info;
    }

    inline VkFenceCreateInfo vkFenceCreateInfo(VkFenceCreateFlags flags = 0)
    {
        VkFenceCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        info.flags = flags;
        return info;
    }

    inline VkSubmitInfo vkSubmitInfo()
    {
        VkSubmitInfo info {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        return info;
    }

    inline VkBufferCreateInfo vkBufferCreateInfo()
    {
        VkBufferCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        return info;
    }

    inline VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo(
        const std::vector<VkDescriptorPoolSize>& poolSizes,
        uint32_t maxSets)
    {
        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        info.pPoolSizes = poolSizes.data();
        info.maxSets = maxSets;
        return info;
    }

    inline VkDescriptorPoolSize vkDescriptorPoolSize(
        VkDescriptorType type,
        uint32_t descriptorCount)
    {
        VkDescriptorPoolSize size {};
        size.type = type;
        size.descriptorCount = descriptorCount;
        return size;
    }

    inline VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding(
        VkDescriptorType type,
        VkShaderStageFlags stageFlags,
        uint32_t binding,
        uint32_t descriptorCount = 1)
    {
        VkDescriptorSetLayoutBinding layoutBinding {};
        layoutBinding.descriptorType = type;
        layoutBinding.stageFlags = stageFlags;
        layoutBinding.binding = binding;
        layoutBinding.descriptorCount = descriptorCount;
        return layoutBinding;
    }

    inline VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo(
        const std::vector<VkDescriptorSetLayoutBinding>& bindings)
    {
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.pBindings = bindings.data();
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        return info;
    }

    inline VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo(
        const VkDescriptorSetLayout* pSetLayouts,
        uint32_t setLayoutCount = 1)
    {
        VkPipelineLayoutCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = setLayoutCount;
        info.pSetLayouts = pSetLayouts;
        return info;
    }

    inline VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo(
        VkDescriptorPool descriptorPool,
        const VkDescriptorSetLayout* pSetLayouts,
        uint32_t descriptorSetCount)
    {
        VkDescriptorSetAllocateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = descriptorPool;
        info.pSetLayouts = pSetLayouts;
        info.descriptorSetCount = descriptorSetCount;
        return info;
    }

    inline VkWriteDescriptorSet vkWriteDescriptorSet(
        VkDescriptorSet dstSet,
        VkDescriptorType type,
        uint32_t binding,
        VkDescriptorImageInfo *imageInfo,
        uint32_t descriptorCount = 1)
    {
        VkWriteDescriptorSet descriptorSet {};
        descriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorSet.dstSet = dstSet;
        descriptorSet.descriptorType = type;
        descriptorSet.dstBinding = binding;
        descriptorSet.pImageInfo = imageInfo;
        descriptorSet.descriptorCount = descriptorCount;
        return descriptorSet;
    }

    inline VkVertexInputBindingDescription vkVertexInputBindingDescription(
        uint32_t binding,
        uint32_t stride,
        VkVertexInputRate inputRate)
    {
        VkVertexInputBindingDescription bindDescription {};
        bindDescription.binding = binding;
        bindDescription.stride = stride;
        bindDescription.inputRate = inputRate;
        return bindDescription;
    }

    inline VkVertexInputAttributeDescription vkVertexInputAttributeDescription(
        uint32_t binding,
        uint32_t location,
        VkFormat format,
        uint32_t offset)
    {
        VkVertexInputAttributeDescription attribDescription {};
        attribDescription.location = location;
        attribDescription.binding = binding;
        attribDescription.format = format;
        attribDescription.offset = offset;
        return attribDescription;
    }

    inline VkPipelineVertexInputStateCreateInfo vkPipelineVertexInputStateCreateInfo()
    {
        VkPipelineVertexInputStateCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        return info;
    }

    inline VkPipelineInputAssemblyStateCreateInfo vkPipelineInputAssemblyStateCreateInfo(
        VkPrimitiveTopology topology,
        VkPipelineInputAssemblyStateCreateFlags flags,
        VkBool32 primitiveRestartEnable)
    {
        VkPipelineInputAssemblyStateCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        info.topology = topology;
        info.flags = flags;
        info.primitiveRestartEnable = primitiveRestartEnable;
        return info;
    }

    inline VkPipelineRasterizationStateCreateInfo vkPipelineRasterizationStateCreateInfo(
        VkPolygonMode polygonMode,
        VkCullModeFlags cullMode,
        VkFrontFace frontFace,
        VkPipelineRasterizationStateCreateFlags flags = 0)
    {
        VkPipelineRasterizationStateCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        info.polygonMode = polygonMode;
        info.cullMode = cullMode;
        info.frontFace = frontFace;
        info.flags = flags;
        info.depthClampEnable = VK_FALSE;
        info.lineWidth = 1.0f;
        return info;
    }

    inline VkPipelineColorBlendAttachmentState vkPipelineColorBlendAttachmentState(
        VkColorComponentFlags colorWriteMask,
        VkBool32 blendEnable)
    {
        VkPipelineColorBlendAttachmentState state {};
        state.colorWriteMask = colorWriteMask;
        state.blendEnable = blendEnable;
        return state;
    }

    inline VkPipelineColorBlendStateCreateInfo vkPipelineColorBlendStateCreateInfo(
        uint32_t attachmentCount,
        const VkPipelineColorBlendAttachmentState * pAttachments)
    {
        VkPipelineColorBlendStateCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        info.attachmentCount = attachmentCount;
        info.pAttachments = pAttachments;
        return info;
    }

    inline VkPipelineDepthStencilStateCreateInfo vkPipelineDepthStencilStateCreateInfo(
        VkBool32 depthTestEnable,
        VkBool32 depthWriteEnable,
        VkCompareOp depthCompareOp)
    {
        VkPipelineDepthStencilStateCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        info.depthTestEnable = depthTestEnable;
        info.depthWriteEnable = depthWriteEnable;
        info.depthCompareOp = depthCompareOp;
        info.back.compareOp = VK_COMPARE_OP_ALWAYS;
        return info;
    }

    inline VkPipelineViewportStateCreateInfo vkPipelineViewportStateCreateInfo(
        uint32_t viewportCount,
        uint32_t scissorCount,
        VkPipelineViewportStateCreateFlags flags = 0)
    {
        VkPipelineViewportStateCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        info.viewportCount = viewportCount;
        info.scissorCount = scissorCount;
        info.flags = flags;
        return info;
    }

    inline VkPipelineMultisampleStateCreateInfo vkPipelineMultisampleStateCreateInfo(
        VkSampleCountFlagBits rasterizationSamples,
        VkPipelineMultisampleStateCreateFlags flags = 0)
    {
        VkPipelineMultisampleStateCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        info.rasterizationSamples = rasterizationSamples;
        info.flags = flags;
        return info;
    }

    inline VkPipelineDynamicStateCreateInfo vkPipelineDynamicStateCreateInfo(
        const std::vector<VkDynamicState>& pDynamicStates,
        VkPipelineDynamicStateCreateFlags flags = 0)
    {
        VkPipelineDynamicStateCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        info.pDynamicStates = pDynamicStates.data();
        info.dynamicStateCount = static_cast<uint32_t>(pDynamicStates.size());
        info.flags = flags;
        return info;
    }

    inline VkGraphicsPipelineCreateInfo vkPipelineCreateInfo(
        VkPipelineLayout layout,
        VkRenderPass renderPass,
        VkPipelineCreateFlags flags = 0)
    {
        VkGraphicsPipelineCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.layout = layout;
        info.renderPass = renderPass;
        info.flags = flags;
        info.basePipelineIndex = -1;
        info.basePipelineHandle = VK_NULL_HANDLE;
        return info;
    }

    inline VkComputePipelineCreateInfo vkComputePipelineCreateInfo(
        VkPipelineLayout layout, 
        VkPipelineCreateFlags flags = 0)
    {
        VkComputePipelineCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.layout = layout;
        info.flags = flags;
        return info;
    }
}