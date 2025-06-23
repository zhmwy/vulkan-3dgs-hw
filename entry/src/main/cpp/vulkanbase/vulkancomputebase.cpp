#include "vulkancomputebase.h"
#include "common/common.h"
#include "file/file_operator.h"

VkResult VulkanComputeBase::createInstance()
{
	std::string name = "VulkanComputeGS";
	uint32_t apiVersion = VK_API_VERSION_1_3;
    
    LOGI("VulkanComputeBase::createInstance is comming");
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = name.c_str();
	appInfo.pEngineName = name.c_str();
	appInfo.apiVersion = apiVersion;

	std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME, VK_OHOS_SURFACE_EXTENSION_NAME };
    
	// Get extensions supported by the instance and store for later use
	uint32_t extCount = 0;
	vkEnumerateInstanceExtensionPropertiesGS(nullptr, &extCount, nullptr);
	if (extCount > 0)
	{
		std::vector<VkExtensionProperties> extensions(extCount);
		if (vkEnumerateInstanceExtensionPropertiesGS(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
		{
			for (VkExtensionProperties extension : extensions)
			{
				supportedInstanceExtensions.push_back(extension.extensionName);
			}
		}
	}

	// Enabled requested instance extensions
	if (enabledInstanceExtensions.size() > 0) 
	{
		for (const char * enabledExtension : enabledInstanceExtensions) 
		{
			// Output message if requested extension is not available
			if (std::find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
			{
				std::cerr << "Enabled instance extension \"" << enabledExtension << "\" is not present at instance level\n";
			}
			instanceExtensions.push_back(enabledExtension);
		}
	}

	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = NULL;
	instanceCreateInfo.pApplicationInfo = &appInfo;

	if (instanceExtensions.size() > 0)
	{
		instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
	}
    if (vkCreateInstanceGS == nullptr) {
        LOGI("VulkanComputeBase::create Instance vkCreateInstance is null");
    }
    return vkCreateInstanceGS(&instanceCreateInfo, nullptr, &instance);
}

void VulkanComputeBase::createPipelineCache()
{
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VK_CHECK_RESULT(vkCreatePipelineCacheGS(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}

bool VulkanComputeBase::prepare()
{
	createCommandPool();
	createPipelineCache();
    prepreAllocator();
    setupDescriptorPool();
    prepareCompute();
    prepareSyncSemaphore();
    
    LOGI("VulkanComputeBase prepare end");
    return true;
}

VkPipelineShaderStageCreateInfo VulkanComputeBase::loadShader(std::string fileName, VkShaderStageFlagBits stage)
{
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = stage;
	shaderStage.module = vkbase::loadSpvShader(fileName.c_str(), device);
	shaderStage.pName = "main";
	assert(shaderStage.module != VK_NULL_HANDLE);
	shaderModules.push_back(shaderStage.module);

	return shaderStage;
}

VulkanComputeBase::VulkanComputeBase()
{
	libLoaded = vkbase::loadVulkanLibrary();
	if (libLoaded) {
        LOGI("VulkanComputeBase dlopen libvulkan.so success ");
	} else {
        LOGE("VulkanComputeBase dlopen libvulkan.so failed ");
	}
}

VulkanComputeBase::~VulkanComputeBase()
{
	// Clean up Vulkan resources
	if (descriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPoolGS(device, descriptorPool, nullptr);
	}
	if (renderPass != VK_NULL_HANDLE)
	{
		vkDestroyRenderPassGS(device, renderPass, nullptr);
	}

	for (auto& shaderModule : shaderModules)
	{
		vkDestroyShaderModuleGS(device, shaderModule, nullptr);
	}

	vkDestroyPipelineCacheGS(device, pipelineCache, nullptr);

	vkDestroyCommandPoolGS(device, cmdPool, nullptr);

    vulkanDevice = nullptr;
    
	vkDestroyInstanceGS(instance, nullptr);
}

bool VulkanComputeBase::initVulkan()
{
	VkResult err;
    assert(libLoaded);
    if (!libLoaded) {
        return false;
    }

	// Vulkan instance
	err = createInstance();
	if (err) {
		LOGE("initVulkan: Could not create Vulkan instance : %{public}s", err);
		return false;
	}

	vkbase::loadVulkanFunctions(instance);

	// Physical device
	uint32_t gpuCount = 0;
	// Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevicesGS(instance, &gpuCount, nullptr));
    if (gpuCount == 0) {
		LOGE("initVulkan: No device with Vulkan support found");
		return false;
	}
	// Enumerate devices
	std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
	err = vkEnumeratePhysicalDevicesGS(instance, &gpuCount, physicalDevices.data());
	if (err) {
		LOGE("initVulkan: Could not enumerate physical devices");
		return false;
	}

	// GPU selection

	// Select physical device to be used for the Vulkan example
	// Defaults to the first device unless specified by command line
	uint32_t selectedDevice = 0;

	physicalDevice = physicalDevices[selectedDevice];

	// Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
	vkGetPhysicalDevicePropertiesGS(physicalDevice, &deviceProperties);
	vkGetPhysicalDeviceFeaturesGS(physicalDevice, &deviceFeatures);
	vkGetPhysicalDeviceMemoryPropertiesGS(physicalDevice, &deviceMemoryProperties);
    
    getEnabledFeatures();

	// Vulkan device creation
	// This is handled by a separate class that gets a logical device representation
	// and encapsulates functions related to a device
	vulkanDevice = std::make_shared<vkbase::VulkanDevice>(physicalDevice);

	// Derived examples can enable extensions based on the list of supported extensions read from the physical device
	getEnabledExtensionsGS();

	VkResult res = vulkanDevice->vdCreateLogicalDevice(enabledFeatures, enabledDeviceExtensions, &features_);
	if (res != VK_SUCCESS) {
		LOGE("initVulkan: create logic device failed");
		return false;
	}
	device = vulkanDevice->vdLogicalDevice;

    // Get a graphics queue from the device
    //vkGetDeviceQueueGS(device, vulkanDevice->queueFamilyIndices.graphics, 0, &queue);
	return true;
}

void VulkanComputeBase::createCommandPool()
{
    LOGI("VulkanComputeBase::createCommandPool");
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
//	cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
	cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK_RESULT(vkCreateCommandPoolGS(device, &cmdPoolInfo, nullptr, &cmdPool));
}

void VulkanComputeBase::getEnabledFeatures() {}
void VulkanComputeBase::getEnabledExtensionsGS() {}

std::string VulkanComputeBase::getPath(std::string path) 
{
	std::string fileName = FileOperator::GetInstance()->GetFileAbsolutePath(path);
	return fileName;
}

void VulkanComputeBase::prepreAllocator()
{
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddrGS;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddrGS;
    
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.instance = instance;
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
    
    vmaCreateAllocator(&allocatorCreateInfo, &allocator);
}

void VulkanComputeBase::setupDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vkbase::vkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
        vkbase::vkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2),
        vkbase::vkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2),
        vkbase::vkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048),
        vkbase::vkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 64),
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vkbase::vkDescriptorPoolCreateInfo(poolSizes, 2048);
    VK_CHECK_RESULT(vkCreateDescriptorPoolGS(device, &descriptorPoolInfo, nullptr, &descriptorPool));
}


void VulkanComputeBase::prepareCompute()
{
    vkGetDeviceQueueGS(device, vulkanDevice->queueFamilyIndices.compute, 0, &computeRes.queue);

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vkbase::vkDescriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0),
        vkbase::vkDescriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout = vkbase::vkDescriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayoutGS(device, &descriptorLayout, nullptr, &computeRes.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vkbase::vkPipelineLayoutCreateInfo(&computeRes.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayoutGS(device, &pipelineLayoutCreateInfo, nullptr, &computeRes.pipelineLayout));

    VkDescriptorSetAllocateInfo allocInfo = vkbase::vkDescriptorSetAllocateInfo(
        descriptorPool, &computeRes.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSetsGS(device, &allocInfo, &computeRes.descriptorSet));

    std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
        //vkbase::vkWriteDescriptorSet(computeRes.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &textureColorMap.vkDescriptor),
        //vkbase::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImage.vkDescriptor)
    };
    vkUpdateDescriptorSetsGS(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);

    VkComputePipelineCreateInfo computePipelineCreateInfo = vkbase::vkComputePipelineCreateInfo(
        computeRes.pipelineLayout, 0);

    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPoolGS(device, &cmdPoolInfo, nullptr, &computeRes.commandPool));

    VkCommandBufferAllocateInfo cmdBufAllocateInfo = vkbase::vkCommandBufferAllocateInfo(
        computeRes.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
    VK_CHECK_RESULT(vkAllocateCommandBuffersGS(device, &cmdBufAllocateInfo, &computeRes.commandBuffer));

    VkSemaphoreCreateInfo semaphoreCreateInfo = vkbase::vkSemaphoreCreateInfo();
    VK_CHECK_RESULT(vkCreateSemaphoreGS(device, &semaphoreCreateInfo, nullptr, &computeRes.semaphore));

    VkQueryPoolCreateInfo queryPoolInfo = {};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = MAX_TIME_QUERY_NUM * 2; // Two timestamps: start and end
    VK_CHECK_RESULT(vkCreateQueryPoolGS(device, &queryPoolInfo, nullptr, &computeRes.queryPool));
    // buildComputeCommandBuffer();
}

void VulkanComputeBase::prepareSyncSemaphore()
{
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkbase::vkSemaphoreCreateInfo();
    VK_CHECK_RESULT(vkCreateSemaphoreGS(device, &semaphoreCreateInfo, nullptr, &syncSemaphore));
    
    VkSubmitInfo submitInfo = vkbase::vkSubmitInfo();
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &syncSemaphore;
    VK_CHECK_RESULT(vkQueueSubmitGS(computeRes.queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK_RESULT(vkQueueWaitIdleGS(computeRes.queue));
}

void VulkanComputeBase::queryCmdTimeStart(std::string eventName)
{
    if (queryTime.find(eventName) != queryTime.end())
    {
        throw std::runtime_error("Time Query Event existed!");
    }
    unsigned int queryIndex = queryTime.size();
    queryTime[eventName] = { queryIndex * 2, queryIndex * 2 + 1 };
    vkCmdWriteTimestampGS(computeRes.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, computeRes.queryPool, queryTime[eventName].startIndex);
}

void VulkanComputeBase::queryCmdTimeEnd(std::string eventName)
{
    if (queryTime.find(eventName) == queryTime.end()) {
        throw std::runtime_error("Can Not Find Time Query Event!");
    }
    vkCmdWriteTimestampGS(computeRes.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, computeRes.queryPool, queryTime[eventName].endIndex);
}

void VulkanComputeBase::queryEventTime(int training_step)
{
    uint64_t timestamps[MAX_TIME_QUERY_NUM * 2];
    vkGetQueryPoolResultsGS(device, computeRes.queryPool, 0, 2 * queryTime.size(), sizeof(timestamps), timestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);

    double timestampPeriod = deviceProperties.limits.timestampPeriod; 
    
    for (const auto & query: queryTime)
    {
        elapsedTimeSum[query.first] += (timestamps[query.second.endIndex] - timestamps[query.second.startIndex]) * timestampPeriod / 1e6;
    }
    queryTime.clear();

    if ((training_step + 1) % QUERY_TIME_RANGE == 0)
    {
        for (const auto& query : elapsedTimeSum)
        {
            LOGI("[%{public}s] execution time : %{public}f ms.", query.first.c_str(), query.second / QUERY_TIME_RANGE);
        }
        elapsedTimeSum.clear();
    }
}
