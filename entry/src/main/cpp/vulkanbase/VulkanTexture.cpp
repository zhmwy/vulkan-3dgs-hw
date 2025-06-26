
#include <VulkanTexture.h>

namespace vkbase
{
	void TextureGS::updateDescriptorGS()
	{
		vkDescriptor.sampler = vkSampler;
		vkDescriptor.imageView = vkImageView;
		vkDescriptor.imageLayout = vkImageLayout;
	}

	void TextureGS::destroyGS()
	{
		vkDestroyImageViewGS(vkDevice->vdLogicalDevice, vkImageView, nullptr);
		vkDestroyImageGS(vkDevice->vdLogicalDevice, vkImage, nullptr);
		if (vkSampler)
		{
			vkDestroySamplerGS(vkDevice->vdLogicalDevice, vkSampler, nullptr);
		}
		vkFreeMemoryGS(vkDevice->vdLogicalDevice, vkDeviceMemory, nullptr);
	}

	void Texture2DGS::fromImageBuffer(void* buffer, VkDeviceSize bufferSize, VkFormat format, uint32_t texWidth, uint32_t texHeight, std::shared_ptr<VulkanDevice> device, VkQueue copyQueue, VkFilter filter, VkImageUsageFlags imageUsageFlags, VkImageLayout imageLayout)
	{
		assert(buffer);

		this->vkDevice = device;
		width = texWidth;
		height = texHeight;
		texMipLevels = 1;

		VkMemoryAllocateInfo memAllocInfo = vkbase::vkMemoryAllocateInfo();
		VkMemoryRequirements memReqs;

		// Use a separate command buffer for texture loading
		VkCommandBuffer copyCmd = device->vdCreateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		// Create a host-visible staging buffer that contains the raw image data
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;

		VkBufferCreateInfo bufferCreateInfo = vkbase::vkBufferCreateInfo();
		bufferCreateInfo.size = bufferSize;
		// This buffer is used as a transfer source for the buffer copy
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VK_CHECK_RESULT(vkCreateBufferGS(device->vdLogicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));

		// Get memory requirements for the staging buffer (alignment, memory type bits)
		vkGetBufferMemoryRequirementsGS(device->vdLogicalDevice, stagingBuffer, &memReqs);

		memAllocInfo.allocationSize = memReqs.size;
		// Get memory type index for a host visible buffer
		memAllocInfo.memoryTypeIndex = device->vdGetMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VK_CHECK_RESULT(vkAllocateMemoryGS(device->vdLogicalDevice, &memAllocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemoryGS(device->vdLogicalDevice, stagingBuffer, stagingMemory, 0));

		// Copy texture data into staging buffer
		uint8_t *data;
		VK_CHECK_RESULT(vkMapMemoryGS(device->vdLogicalDevice, stagingMemory, 0, memReqs.size, 0, (void **)&data));
		memcpy(data, buffer, bufferSize);
		vkUnmapMemoryGS(device->vdLogicalDevice, stagingMemory);

		VkBufferImageCopy copyRegion = {};
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent.width = width;
		copyRegion.imageExtent.height = height;
		copyRegion.imageExtent.depth = 1;
		copyRegion.bufferOffset = 0;

		// Create optimal tiled target image
		VkImageCreateInfo iCreateInfo = vkbase::vkImageCreateInfo();
		iCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		iCreateInfo.format = format;
		iCreateInfo.mipLevels = texMipLevels;
		iCreateInfo.arrayLayers = 1;
		iCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		iCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		iCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		iCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		iCreateInfo.extent = { width, height, 1 };
		iCreateInfo.usage = imageUsageFlags;
		// Ensure that the TRANSFER_DST bit is set for staging
		if (!(iCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
		{
			iCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		}
		VK_CHECK_RESULT(vkCreateImageGS(device->vdLogicalDevice, &iCreateInfo, nullptr, &vkImage));

		vkGetImageMemoryRequirementsGS(device->vdLogicalDevice, vkImage, &memReqs);

		memAllocInfo.allocationSize = memReqs.size;

		memAllocInfo.memoryTypeIndex = device->vdGetMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemoryGS(device->vdLogicalDevice, &memAllocInfo, nullptr, &vkDeviceMemory));
		VK_CHECK_RESULT(vkBindImageMemoryGS(device->vdLogicalDevice, vkImage, vkDeviceMemory, 0));

		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = texMipLevels;
		subresourceRange.layerCount = 1;

		// Image barrier for optimal image (target)
		// Optimal image will be used as destination for the copy
		vkbase::setImageLayout(
			copyCmd,
			vkImage,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			subresourceRange);

		// Copy mip levels from staging buffer
		vkCmdCopyBufferToImageGS(
			copyCmd,
			stagingBuffer,
			vkImage,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&copyRegion
		);

		// Change texture image layout to shader read after all mip levels have been copied
		this->vkImageLayout = imageLayout;
		vkbase::setImageLayout(
			copyCmd,
			vkImage,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			imageLayout,
			subresourceRange);

		device->vdFlushCommandBuffer(copyCmd, copyQueue);

		// Clean up staging resources
		vkFreeMemoryGS(device->vdLogicalDevice, stagingMemory, nullptr);
		vkDestroyBufferGS(device->vdLogicalDevice, stagingBuffer, nullptr);

		// Create sampler
		VkSamplerCreateInfo sCreateInfo = {};
		sCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		sCreateInfo.magFilter = filter;
		sCreateInfo.minFilter = filter;
		sCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sCreateInfo.mipLodBias = 0.0f;
		sCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
		sCreateInfo.minLod = 0.0f;
		sCreateInfo.maxLod = 0.0f;
		sCreateInfo.maxAnisotropy = 1.0f;
		VK_CHECK_RESULT(vkCreateSamplerGS(device->vdLogicalDevice, &sCreateInfo, nullptr, &vkSampler));

		// Create image view
		VkImageViewCreateInfo viewCreateInfo = {};
		viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewCreateInfo.pNext = NULL;
		viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewCreateInfo.format = format;
		viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		viewCreateInfo.subresourceRange.levelCount = 1;
		viewCreateInfo.image = vkImage;
		VK_CHECK_RESULT(vkCreateImageViewGS(device->vdLogicalDevice, &viewCreateInfo, nullptr, &vkImageView));

		// Update descriptor image info member that can be used for setting up descriptor sets
		updateDescriptorGS();
	}


}
