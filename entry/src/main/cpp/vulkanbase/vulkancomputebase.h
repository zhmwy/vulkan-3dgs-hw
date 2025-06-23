#pragma once

#include <map>
#include <vector>

#include <string>
#include "VulkanOhos.h"
#include "vk_mem_alloc.h"
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
    
	virtual void compute() = 0;
    
    virtual void getEnabledFeatures();
	virtual void getEnabledExtensionsGS();
    
	virtual bool prepare();
	VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage);
	std::string getPath(std::string path);
    
	void createPipelineCache();
	void createCommandPool();
    
protected:
    void prepreAllocator();
    
    void setupDescriptorPool();
    
    void prepareCompute();
    
    void prepareSyncSemaphore();

    void queryEventTime(int training_step);
    void queryCmdTimeStart(std::string enventName);
    void queryCmdTimeEnd(std::string enventName);

    struct Compute {
        VkQueue queue;                                // Separate queue for compute commands
        VkCommandPool commandPool;                    // Use a separate command pool
        VkCommandBuffer commandBuffer;                // Command buffer storing the dispatch commands and barriers
        VkSemaphore semaphore;                        // Execution dependency between compute & graphic submission
        VkDescriptorSetLayout descriptorSetLayout;    // Compute shader binding layout
        VkDescriptorSet descriptorSet;                // Compute shader bindings
        VkPipelineLayout pipelineLayout;              // Layout of the compute pipeline
        
        VkQueryPool queryPool;
    } computeRes;
    
    VkSemaphore syncSemaphore;

	bool prepared = false;

    VmaAllocator allocator;

	std::shared_ptr<vkbase::VulkanDevice> vulkanDevice;
    
	VkInstance instance;
	std::vector<std::string> supportedInstanceExtensions;
	VkPhysicalDevice physicalDevice;
	VkPhysicalDeviceProperties deviceProperties;
	VkPhysicalDeviceFeatures deviceFeatures;
	VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
    
	VkPhysicalDeviceFeatures enabledFeatures{};
    
	std::vector<const char*> enabledDeviceExtensions;
	std::vector<const char*> enabledInstanceExtensions;
    
    VkPhysicalDeviceFeatures2 features_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    
	VkDevice device;
    
	//VkQueue queue;
	VkCommandPool cmdPool;
    
	VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submitInfo;
    
	VkRenderPass renderPass = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    
	std::vector<VkShaderModule> shaderModules;    
	VkPipelineCache pipelineCache;

	bool libLoaded = false;
    
    static const int MAX_TIME_QUERY_NUM = 100;
    static const int QUERY_TIME_RANGE = 500;

    struct QueryTime {
        uint32_t startIndex; // Start index in the query pool
        uint32_t endIndex;   // End index in the query pool
    };
    std::map<std::string, QueryTime> queryTime;
    std::map<std::string, float> elapsedTimeSum;

};