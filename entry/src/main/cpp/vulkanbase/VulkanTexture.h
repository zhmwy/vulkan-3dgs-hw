#pragma once

#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "vulkan/vulkan.h"


#include "VulkanBuffer.h"
#include "VulkanDevice.h"
#include "VulkanTools.h"

#if defined(__ANDROID__)
#	include <android/asset_manager.h>
#endif

namespace vks
{
class TextureGS
{
  public:
	VulkanDevice *        vkDevice;
	VkImage               vkImage;
	VkImageLayout         vkImageLayout;
	VkDeviceMemory        vkDeviceMemory;
	VkImageView           vkImageView;
	uint32_t              width, height;
	uint32_t              texMipLevels;
	uint32_t              texLayerCount;
	VkDescriptorImageInfo vkDescriptor;
	VkSampler             vkSampler;

	void      updateDescriptorGS();
	void      destroyGS();

};

class Texture2DGS : public TextureGS
{
  public:
	void fromImageBuffer(
	    void *             texBuffer,
	    VkDeviceSize       texBufferSize,
	    VkFormat           texFormat,
	    uint32_t           texWidth,
	    uint32_t           texHeight,
	    VulkanDevice      *texDevice,
	    VkQueue            texCopyQueue,
	    VkFilter           texFilter          = VK_FILTER_LINEAR,
	    VkImageUsageFlags  texImageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
	    VkImageLayout      texImageLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
};


} 
