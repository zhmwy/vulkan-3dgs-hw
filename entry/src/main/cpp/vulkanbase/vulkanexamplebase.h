/*
* Vulkan Example base class
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <array>
#include <unordered_map>
#include <numeric>
#include <ctime>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <sys/stat.h>

#include <string>
#include <numeric>
#include <array>
#include "VulkanOhos.h"
#include "vulkan/vulkan.h"

#include "VulkanBuffer.h"
#include "VulkanDevice.h"
#include "VulkanTexture.h"
#include "VulkanInitializers.hpp"
#include "VulkanTools.h"

#define DEFAULT_FENCE_TIMEOUT 100000000000
class VulkanExampleBase
{
public:

	void createPipelineCache();
	void createCommandPool();

	bool libLoaded = false;
protected:
    
	// Vulkan instance, stores all per-application states
	VkInstance instance;
	std::vector<std::string> supportedInstanceExtensions;
	// Physical device (GPU) that Vulkan will use
	VkPhysicalDevice physicalDevice;
	// Stores physical device properties (for e.g. checking device limits)
	VkPhysicalDeviceProperties deviceProperties;
	// Stores the features available on the selected physical device (for e.g. checking if a feature is available)
	VkPhysicalDeviceFeatures deviceFeatures;
	// Stores all available memory (type) properties for the physical device
	VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
	/** @brief Set of physical device features to be enabled for this example (must be set in the derived constructor) */
	VkPhysicalDeviceFeatures enabledFeatures{};
	/** @brief Set of device extensions to be enabled for this example (must be set in the derived constructor) */
	std::vector<const char*> enabledDeviceExtensions;
	std::vector<const char*> enabledInstanceExtensions;
	/** @brief Optional pNext structure for passing extension structures to device creation */
	void* deviceCreatepNextChain = nullptr;
	/** @brief Logical device, application's view of the physical device (GPU) */
	VkDevice device;
	// Handle to the device graphics queue that command buffers are submitted to
	VkQueue queue;
//     VkQueue compute_queue;
	// Depth buffer format (selected during Vulkan initialization)
	VkFormat depthFormat;
	// Command buffer pool
	VkCommandPool cmdPool;
	/** @brief Pipeline stages used to wait at for graphics queue submissions */
	VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	// Contains command buffers and semaphores to be presented to the queue
	VkSubmitInfo submitInfo;
    
	// Global render pass for frame buffer writes
	VkRenderPass renderPass = VK_NULL_HANDLE;
    
	// Descriptor set pool
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    
	std::vector<VkShaderModule> shaderModules;
    
	VkPipelineCache pipelineCache;
    
	struct {
		// Swap chain image presentation
		VkSemaphore presentComplete;
		// Command buffer submission and execution
		VkSemaphore renderComplete;
	} semaphores;

public:
	uint32_t highResWidth;
	uint32_t highResHeight;

	uint32_t lowResWidth;
	uint32_t lowResHeight;

	bool prepared = false;

	vks::VulkanDevice *vulkanDevice;

	std::string name = "XEngineExample";
	uint32_t apiVersion = VK_API_VERSION_1_3;

	VulkanExampleBase();
	virtual ~VulkanExampleBase();
    
	bool initVulkan();

	virtual VkResult createInstance();
    
	virtual void render() = 0;
    
	virtual void setupRenderPass();
    virtual void getEnabledFeatures();
	virtual void getEnabledExtensionsGS();
	virtual bool prepare();
	VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage);
	std::string getPath(std::string path);
};