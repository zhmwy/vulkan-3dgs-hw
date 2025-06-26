
#pragma once

#include "vulkan/vulkan.h"
#include <algorithm>
#include <assert.h>
#include <exception>
#include <vector>

namespace vkbase
{
struct VulkanDevice
{
	VkPhysicalDevice vdPhysicalDevice;
	VkDevice vdLogicalDevice;
	VkPhysicalDeviceProperties vdProperties;
	VkPhysicalDeviceFeatures vdFeatures;
	VkPhysicalDeviceFeatures vdEnabledFeatures;
	VkPhysicalDeviceMemoryProperties vdMemoryProperties;
	std::vector<VkQueueFamilyProperties> vdQueueFamilyProperties;
	std::vector<std::string> vdSupportedExtensions;
	VkCommandPool cdCommandPool = VK_NULL_HANDLE;
    
	struct
	{
		uint32_t graphics;
		uint32_t compute;
		uint32_t transfer;
	} qFamilyIndices;
    
	explicit VulkanDevice(VkPhysicalDevice physicalDevice);
	~VulkanDevice();
	uint32_t        vdGetMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound = nullptr) const;
	uint32_t        vdGetQueueFamilyIndex(VkQueueFlagBits queueFlags) const;
	VkResult        vdCreateLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, std::vector<const char *> enabledExtensions, void *pNextChain, bool useSwapChain = true, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);

	VkCommandPool   vdCreateCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VkCommandBuffer vdCreateCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin = false);
	VkCommandBuffer vdCreateCommandBuffer(VkCommandBufferLevel level, bool begin = false);
	void            vdFlushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free = true);
	void            vdFlushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);
	bool            vdExtensionSupported(std::string extension);
};
}        // namespace vkbase
