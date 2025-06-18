#include <VulkanDevice.h>
#include "VulkanTools.h"
#include <unordered_set>

namespace vkbase
{	
	VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice)
	{
		assert(physicalDevice);
		this->vdPhysicalDevice = physicalDevice;

		// Store Properties features, limits and properties of the physical device for later use
		// Device properties also contain limits and sparse properties
		vkGetPhysicalDevicePropertiesGS(physicalDevice, &vdProperties);
		// Features should be checked by the examples before using them
		vkGetPhysicalDeviceFeaturesGS(physicalDevice, &vdFeatures);
		// Memory properties are used regularly for creating all kinds of buffers
		vkGetPhysicalDeviceMemoryPropertiesGS(physicalDevice, &vdMemoryProperties);
		// Queue family properties, used for setting up requested queues upon device creation
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyPropertiesGS(physicalDevice, &queueFamilyCount, nullptr);
		assert(queueFamilyCount > 0);
		vdQueueFamilyProperties.resize(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyPropertiesGS(physicalDevice, &queueFamilyCount, vdQueueFamilyProperties.data());

		// Get list of supported extensions
		uint32_t extCount = 0;
		vkEnumerateDeviceExtensionPropertiesGS(physicalDevice, nullptr, &extCount, nullptr);
		if (extCount > 0)
		{
			std::vector<VkExtensionProperties> extensions(extCount);
			if (vkEnumerateDeviceExtensionPropertiesGS(physicalDevice, nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
			{
				for (auto ext : extensions)
				{
					vdSupportedExtensions.push_back(ext.extensionName);
				}
			}
		}
	}

	VulkanDevice::~VulkanDevice()
	{
		if (cdCommandPool)
		{
			vkDestroyCommandPoolGS(vdLogicalDevice, cdCommandPool, nullptr);
		}
		if (vdLogicalDevice)
		{
			vkDestroyDeviceGS(vdLogicalDevice, nullptr);
		}
	}

	uint32_t VulkanDevice::vdGetMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound) const
	{
		for (uint32_t i = 0; i < vdMemoryProperties.memoryTypeCount; i++)
		{
			if ((typeBits & 1) == 1)
			{
				if ((vdMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					if (memTypeFound)
					{
						*memTypeFound = true;
					}
					return i;
				}
			}
			typeBits >>= 1;
		}

		if (memTypeFound)
		{
			*memTypeFound = false;
			return 0;
		}
		else
		{
            LOGI("Could not find a matching memory type");
			return -1;
		}
	}

	uint32_t VulkanDevice::vdGetQueueFamilyIndex(VkQueueFlagBits queueFlags) const
	{
    LOGI("VulkanDevice::getQueueFamilyIndex");
		// Dedicated queue for compute
		// Try to find a queue family index that supports compute but not graphics
		if (queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			for (uint32_t i = 0; i < static_cast<uint32_t>(vdQueueFamilyProperties.size()); i++)
			{
				if ((vdQueueFamilyProperties[i].queueFlags & queueFlags) && ((vdQueueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
				{
					return i;
				}
			}
		}

		// Dedicated queue for transfer
		// Try to find a queue family index that supports transfer but not graphics and compute
		if (queueFlags & VK_QUEUE_TRANSFER_BIT)
		{
			for (uint32_t i = 0; i < static_cast<uint32_t>(vdQueueFamilyProperties.size()); i++)
			{
				if ((vdQueueFamilyProperties[i].queueFlags & queueFlags) && ((vdQueueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((vdQueueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
				{
					return i;
				}
			}
		}

		// For other queue types or if no separate compute queue is present, return the first one to support the requested flags
		for (uint32_t i = 0; i < static_cast<uint32_t>(vdQueueFamilyProperties.size()); i++)
		{
			if (vdQueueFamilyProperties[i].queueFlags & queueFlags)
			{
				return i;
			}
		}

		LOGE("getQueueFamilyIndex: Could not find a matching queue family index");
		return -1;
	}

	VkResult VulkanDevice::vdCreateLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, std::vector<const char*> enabledExtensions, void* pNextChain, bool useSwapChain, VkQueueFlags requestedQueueTypes)
	{			

		// Desired queues need to be requested upon logical device creation
		// Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
		// requests different queue types
        LOGI("VulkanDevice::createLogicalDevice");
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};

		// Get queue family indices for the requested queue family types
		// Note that the indices may overlap depending on the implementation

		const float defaultQueuePriority(0.0f);

		// Graphics queue
		if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT)
		{
			queueFamilyIndices.graphics = vdGetQueueFamilyIndex(VK_QUEUE_GRAPHICS_BIT);
			VkDeviceQueueCreateInfo queueInfo{};
			queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueInfo.queueFamilyIndex = queueFamilyIndices.graphics;
			queueInfo.queueCount = 1;
			queueInfo.pQueuePriorities = &defaultQueuePriority;
			queueCreateInfos.push_back(queueInfo);
		}
		else
		{
			queueFamilyIndices.graphics = 0;
		}

		// Dedicated compute queue
		if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT)
		{
			queueFamilyIndices.compute = vdGetQueueFamilyIndex(VK_QUEUE_COMPUTE_BIT);
			if (queueFamilyIndices.compute != queueFamilyIndices.graphics)
			{
				// If compute family index differs, we need an additional queue create info for the compute queue
				VkDeviceQueueCreateInfo queueInfo{};
				queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueInfo.queueFamilyIndex = queueFamilyIndices.compute;
				queueInfo.queueCount = 1;
				queueInfo.pQueuePriorities = &defaultQueuePriority;
				queueCreateInfos.push_back(queueInfo);
			}
		}
		else
		{
			// Else we use the same queue
			queueFamilyIndices.compute = queueFamilyIndices.graphics;
		}

		// Dedicated transfer queue
		if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT)
		{
			queueFamilyIndices.transfer = vdGetQueueFamilyIndex(VK_QUEUE_TRANSFER_BIT);
			if ((queueFamilyIndices.transfer != queueFamilyIndices.graphics) && (queueFamilyIndices.transfer != queueFamilyIndices.compute))
			{
				// If compute family index differs, we need an additional queue create info for the compute queue
				VkDeviceQueueCreateInfo queueInfo{};
				queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueInfo.queueFamilyIndex = queueFamilyIndices.transfer;
				queueInfo.queueCount = 1;
				queueInfo.pQueuePriorities = &defaultQueuePriority;
				queueCreateInfos.push_back(queueInfo);
			}
		}
		else
		{
			// Else we use the same queue
			queueFamilyIndices.transfer = queueFamilyIndices.graphics;
		}
		LOGD("VulkanDevice: queueFamilyIndices.compute = %{public}d", queueFamilyIndices.compute);
		LOGD("VulkanDevice: queueFamilyIndices.graphics = %{public}d", queueFamilyIndices.graphics);
		LOGD("VulkanDevice: queueFamilyIndices.transfer = %{public}d", queueFamilyIndices.transfer);

		// Create the logical device representation
		std::vector<const char*> deviceExtensions(enabledExtensions);
		if (useSwapChain)
		{
			// If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
			deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		}
		deviceExtensions.push_back("VK_OHOS_native_buffer"); //todo

		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
		deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
		deviceCreateInfo.pEnabledFeatures = &enabledFeatures;
		
		// If a pNext(Chain) has been passed, we need to add it to the device creation info
		VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
		if (pNextChain) {
			physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
			physicalDeviceFeatures2.features = enabledFeatures;
			physicalDeviceFeatures2.pNext = pNextChain;
			deviceCreateInfo.pEnabledFeatures = nullptr;
			deviceCreateInfo.pNext = &physicalDeviceFeatures2;
		}

		if (deviceExtensions.size() > 0)
		{
			for (const char* enabledExtension : deviceExtensions)
			{
				LOGD("VulkanDevice: enabledExtension %{public}s", enabledExtension);
				if (!vdExtensionSupported(std::string(enabledExtension))) {
					LOGE("createLogicalDevice: vkCreateDevice Enabled device extension \"%{public}s\" is not present at device level\n", enabledExtension);
				}
			}

			deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
			deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		}

		this->vdEnabledFeatures = enabledFeatures;

        LOGI("VulkanDevice: vkCreateDevice device: &deviceCreateInfo :p %{public}p", &deviceCreateInfo);
        VkResult result = vkCreateDeviceGS(vdPhysicalDevice, &deviceCreateInfo, nullptr, &vdLogicalDevice);
        LOGI("VulkanDevice: vkCreateDevice device: &logicalDevice :p %{public}p", &vdLogicalDevice);
        LOGI("VulkanDevice: vkCreateDevice device: logicalDevice :p %{public}p", vdLogicalDevice);
        if (result != VK_SUCCESS) 
		{
			LOGE("VulkanDevice: vkCreateDevice failed, error code = %{public}d", result);
			return result;
		}

		// Create a default command pool for graphics command buffers
		cdCommandPool = vdCreateCommandPool(queueFamilyIndices.graphics);

		return result;
	}

	VkCommandPool VulkanDevice::vdCreateCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags)
	{
    LOGI("VulkanDevice::createCommandPool 2");
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = createFlags;
		VkCommandPool cmdPool;
		VK_CHECK_RESULT(vkCreateCommandPoolGS(vdLogicalDevice, &cmdPoolInfo, nullptr, &cmdPool));
		return cmdPool;
	}

	VkCommandBuffer VulkanDevice::vdCreateCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin)
	{
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = vkbase::vkCommandBufferAllocateInfo(pool, level, 1);
		VkCommandBuffer cmdBuffer;
		VK_CHECK_RESULT(vkAllocateCommandBuffersGS(vdLogicalDevice, &cmdBufAllocateInfo, &cmdBuffer));
		// If requested, also start recording for the new command buffer
		if (begin)
		{
			VkCommandBufferBeginInfo cmdBufInfo = vkbase::vkCommandBufferBeginInfo();
			VK_CHECK_RESULT(vkBeginCommandBufferGS(cmdBuffer, &cmdBufInfo));
		}
		return cmdBuffer;
	}
			
	VkCommandBuffer VulkanDevice::vdCreateCommandBuffer(VkCommandBufferLevel level, bool begin)
	{
		return vdCreateCommandBuffer(level, cdCommandPool, begin);
	}

	void VulkanDevice::vdFlushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free)
	{
		if (commandBuffer == VK_NULL_HANDLE)
		{
			return;
		}

		VK_CHECK_RESULT(vkEndCommandBufferGS(commandBuffer));

		VkSubmitInfo submitInfo = vkbase::vkSubmitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		// Create fence to ensure that the command buffer has finished executing
		VkFenceCreateInfo fenceInfo = vkbase::vkFenceCreateInfo(VK_FLAGS_NONE);
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFenceGS(vdLogicalDevice, &fenceInfo, nullptr, &fence));
		// Submit to the queue
		VK_CHECK_RESULT(vkQueueSubmitGS(queue, 1, &submitInfo, fence));
		// Wait for the fence to signal that command buffer has finished executing
		VK_CHECK_RESULT(vkWaitForFencesGS(vdLogicalDevice, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
		vkDestroyFenceGS(vdLogicalDevice, fence, nullptr);
		if (free)
		{
			vkFreeCommandBuffersGS(vdLogicalDevice, pool, 1, &commandBuffer);
		}
	}

	void VulkanDevice::vdFlushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free)
	{
		return vdFlushCommandBuffer(commandBuffer, queue, cdCommandPool, free);
	}

	bool VulkanDevice::vdExtensionSupported(std::string extension)
	{
        LOGI("VulkanDevice::extensionSupported ");
		return (std::find(vdSupportedExtensions.begin(), vdSupportedExtensions.end(), extension) != vdSupportedExtensions.end());
	}

};
