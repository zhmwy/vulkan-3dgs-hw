
#include "VulkanTools.h"

namespace vkbase
{
    bool errorModeSilent = false;

    std::string getErrorString(VkResult errorCode)
    {
        switch (errorCode)
        {
#define ERSTR(r) case VK_ ##r: return #r
            ERSTR(NOT_READY);
            ERSTR(TIMEOUT);
            ERSTR(EVENT_SET);
            ERSTR(EVENT_RESET);
            ERSTR(INCOMPLETE);
            ERSTR(ERROR_OUT_OF_HOST_MEMORY);
            ERSTR(ERROR_OUT_OF_DEVICE_MEMORY);
            ERSTR(ERROR_INITIALIZATION_FAILED);
            ERSTR(ERROR_DEVICE_LOST);
            ERSTR(ERROR_MEMORY_MAP_FAILED);
            ERSTR(ERROR_LAYER_NOT_PRESENT);
            ERSTR(ERROR_EXTENSION_NOT_PRESENT);
            ERSTR(ERROR_FEATURE_NOT_PRESENT);
            ERSTR(ERROR_INCOMPATIBLE_DRIVER);
            ERSTR(ERROR_TOO_MANY_OBJECTS);
            ERSTR(ERROR_FORMAT_NOT_SUPPORTED);
            ERSTR(ERROR_SURFACE_LOST_KHR);
            ERSTR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
            ERSTR(SUBOPTIMAL_KHR);
            ERSTR(ERROR_OUT_OF_DATE_KHR);
            ERSTR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
            ERSTR(ERROR_VALIDATION_FAILED_EXT);
            ERSTR(ERROR_INVALID_SHADER_NV);
#undef ERSTR
        default:
            return "UNKNOWN_ERROR";
        }
    }

    void setImageLayout(
        VkCommandBuffer cmdbuffer,
        VkImage image,
        VkImageLayout oldImageLayout,
        VkImageLayout newImageLayout,
        VkImageSubresourceRange subresourceRange,
        VkPipelineStageFlags srcStageMask,
        VkPipelineStageFlags dstStageMask)
    {
        // Create an image barrier object
        VkImageMemoryBarrier imageMemoryBarrier = vkbase::vkImageMemoryBarrier();
        imageMemoryBarrier.oldLayout = oldImageLayout;
        imageMemoryBarrier.newLayout = newImageLayout;
        imageMemoryBarrier.image = image;
        imageMemoryBarrier.subresourceRange = subresourceRange;

        switch (oldImageLayout)
        {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            imageMemoryBarrier.srcAccessMask = 0;
            break;

        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            break;
        }

        switch (newImageLayout)
        {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            if (imageMemoryBarrier.srcAccessMask == 0)
            {
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            break;
        }

        vkCmdPipelineBarrierGS(
            cmdbuffer,
            srcStageMask,
            dstStageMask,
            0,
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier);
    }

    VkShaderModule loadSpvShader(const char *fileName, VkDevice device)
    {
        std::ifstream spvifs(fileName, std::ios::binary | std::ios::in | std::ios::ate);

        if (spvifs.is_open())
        {
            size_t size = spvifs.tellg();
            spvifs.seekg(0, std::ios::beg);
            char* shaderCode = new char[size];
            spvifs.read(shaderCode, size);
            spvifs.close();

            assert(size > 0);

            VkShaderModule shaderModule;
            VkShaderModuleCreateInfo mCreateInfo{};
            mCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            mCreateInfo.codeSize = size;
            mCreateInfo.pCode = (uint32_t*)shaderCode;

            VK_CHECK_RESULT(vkCreateShaderModuleGS(device, &mCreateInfo, NULL, &shaderModule));

            delete[] shaderCode;

            return shaderModule;
        }
        else
        {
            std::cerr << "Error: Could not open shader file \"" << fileName << "\"" << "\n";
            return VK_NULL_HANDLE;
        }
    }
}
