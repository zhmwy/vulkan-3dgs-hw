#pragma once

#include <vector>

#include <string>
#include "VulkanOhos.h"
#include "vulkan/vulkan.h"

#include "VulkanDevice.h"
#include "VulkanTexture.h"
#include "VulkanInitializers.hpp"
#include "VulkanTools.h"

class VulkanComputeBase
{
public:
	VulkanComputeBase();
	virtual ~VulkanComputeBase();
    
	bool initVulkan();

	virtual VkResult createInstance();
    
	virtual void render() = 0;
    
    virtual void getEnabledFeatures();
	virtual void getEnabledExtensionsGS();
    
	virtual bool prepare();
	VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage);
	std::string getPath(std::string path);
    
	void createPipelineCache();
	void createCommandPool();
    
protected:

	bool prepared = false;

	vks::VulkanDevice *vulkanDevice;
    
	VkInstance instance;
	std::vector<std::string> supportedInstanceExtensions;
	VkPhysicalDevice physicalDevice;
	VkPhysicalDeviceProperties deviceProperties;
	VkPhysicalDeviceFeatures deviceFeatures;
	VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    
	VkPhysicalDeviceFeatures enabledFeatures{};
    
	std::vector<const char*> enabledDeviceExtensions;
	std::vector<const char*> enabledInstanceExtensions;
    
	void* deviceCreatepNextChain = nullptr;
	VkDevice device;
    
	VkQueue queue;
	VkCommandPool cmdPool;
    
	VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submitInfo;
    
	VkRenderPass renderPass = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    
	std::vector<VkShaderModule> shaderModules;    
	VkPipelineCache pipelineCache;

	bool libLoaded = false;

};