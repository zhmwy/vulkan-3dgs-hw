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
    if (device) {
        vkDestroySemaphoreGS(device, syncSemaphore, nullptr);
        
        vkDestroyPipelineLayoutGS(device, computeRes.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayoutGS(device, computeRes.descriptorSetLayout, nullptr);
        vkDestroySemaphoreGS(device, computeRes.semaphore, nullptr);
        vkDestroyCommandPoolGS(device, computeRes.commandPool, nullptr);
    }
    
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
        vkbase::vkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024),
        vkbase::vkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024),
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

void VulkanComputeBase::createPipelineLayout(    
    STComputePipeLine& pipe_3dgs,
    uint32_t numImages,
    uint32_t numArrays, 
    bool usePushConstant,
    uint32_t pushConstantSize
) {
    
    // Define descriptor set bindings
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings(1+numArrays);

    setLayoutBindings[0] = vkbase::vkDescriptorSetLayoutBinding(
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,              // descriptor type
        VK_SHADER_STAGE_COMPUTE_BIT,                       // shader stage
        0,                                                   // binding number
        numImages
    );

    
    for (uint32_t i = 0; i < numArrays; ++i) {
        setLayoutBindings[i+1] = vkbase::vkDescriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,              // descriptor type
            VK_SHADER_STAGE_COMPUTE_BIT,                    // shader stage
            i + 1                                     // binding number
        );
    }

    // Create descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = vkbase::vkDescriptorSetLayoutCreateInfo(setLayoutBindings);

    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayoutGS(device, &layoutInfo, nullptr, &pipe_3dgs.descriptorSetLayout)
    );

    // Define push constant range (optional)
    VkPushConstantRange pushConstantRange = {};
    if (usePushConstant) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = pushConstantSize;  // Example push constant size
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipe_3dgs.descriptorSetLayout;

    if (usePushConstant) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    } else {
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
    }

    VK_CHECK_RESULT(
        vkCreatePipelineLayoutGS(device, &pipelineLayoutInfo, nullptr, &pipe_3dgs.pipelineLayout)
    );

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = vkbase::vkDescriptorSetAllocateInfo(
        descriptorPool, 
        &pipe_3dgs.descriptorSetLayout, 1
    );

    VK_CHECK_RESULT(
        vkAllocateDescriptorSetsGS(device, &allocInfo, &pipe_3dgs.descriptorSet)
    );
}

void VulkanComputeBase::createPipelineLayout(
    STComputePipeLine& pipe_3dgs,
    uint32_t numBindings, 
    bool usePushConstant,
    uint32_t pushConstantSize
) {
    // Define descriptor set bindings
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings(numBindings);
    
    for (uint32_t i = 0; i < numBindings; ++i) {
        setLayoutBindings[i] = vkbase::vkDescriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,              // descriptor type
            VK_SHADER_STAGE_COMPUTE_BIT,                    // shader stage
            i                                               // binding number
        );
    }

    // Create descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = vkbase::vkDescriptorSetLayoutCreateInfo(setLayoutBindings);

    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayoutGS(device, &layoutInfo, nullptr, &pipe_3dgs.descriptorSetLayout)
    );

    // Define push constant range (optional)
    VkPushConstantRange pushConstantRange = {};
    if (usePushConstant) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = pushConstantSize;  // Example push constant size
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipe_3dgs.descriptorSetLayout;

    if (usePushConstant) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    } else {
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
    }

    VK_CHECK_RESULT(
        vkCreatePipelineLayoutGS(device, &pipelineLayoutInfo, nullptr, &pipe_3dgs.pipelineLayout)
    );

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = vkbase::vkDescriptorSetAllocateInfo(
        descriptorPool, 
        &pipe_3dgs.descriptorSetLayout, 1
    );

    VK_CHECK_RESULT(
        vkAllocateDescriptorSetsGS(device, &allocInfo, &pipe_3dgs.descriptorSet)
    );
}

void VulkanComputeBase::buildComputeCommandBuffer(std::function<void()> func)
{
    vkQueueWaitIdleGS(computeRes.queue);

    VkCommandBufferBeginInfo cmdBufInfo = vkbase::vkCommandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBufferGS(computeRes.commandBuffer, &cmdBufInfo));
    
    func();

    VK_CHECK_RESULT(vkEndCommandBufferGS(computeRes.commandBuffer));
}

void VulkanComputeBase::executeComputeCommand()
{
    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    VkSubmitInfo computeSubmitInfo = vkbase::vkSubmitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &computeRes.commandBuffer;
    computeSubmitInfo.waitSemaphoreCount = 1;
    computeSubmitInfo.pWaitSemaphores = &syncSemaphore;
    computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores = &computeRes.semaphore;
    VkResult res = vkQueueSubmitGS(computeRes.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE);

    if (res != VK_SUCCESS) {
#ifdef VULKAN_HW
        LOGE("VulkanExample Fatal : VkResult is %s", vkbase::getErrorString(res).c_str());
#elif defined(VULKAN_PC)
        std::cerr << "VulkanExample Fatal : VkResult is " << vkbase::errorString(res) << std::endl;
#endif
    }
    
    vkQueueWaitIdleGS(computeRes.queue);
}

void VulkanComputeBase::executeTransCommand()
{
    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkSubmitInfo computeSubmitInfo = vkbase::vkSubmitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &computeRes.commandBuffer;
    computeSubmitInfo.waitSemaphoreCount = 1;
    computeSubmitInfo.pWaitSemaphores = &syncSemaphore;
    computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores = &computeRes.semaphore;
    VkResult res = vkQueueSubmitGS(computeRes.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE);

    if (res != VK_SUCCESS) {
#ifdef VULKAN_HW
        LOGE("VulkanExample Fatal : VkResult is %s", vkbase::getErrorString(res).c_str());
#elif defined(VULKAN_PC)
        std::cerr << "VulkanExample Fatal : VkResult is " << vkbase::errorString(res) << std::endl;
#endif
    }
    
    vkQueueWaitIdleGS(computeRes.queue);
}

void VulkanComputeBase::ComputeMemoryBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrierGS(
        computeRes.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanComputeBase::TransComputeBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrierGS(
        computeRes.commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanComputeBase::ComputeTransBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrierGS(
        computeRes.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanComputeBase::ComputeShaderMemoryBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrierGS(
        computeRes.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanComputeBase::createGpu2CpuCopy(vkgs::Buffer &buffer)
{
    vkgs::CpuBuffer cpu_buffer = buffer.cpu_buffer;
    VkBufferCopy region = {};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = buffer.size();
    vkCmdCopyBufferGS(
        computeRes.commandBuffer,
        buffer,
        cpu_buffer,
        1, &region
    );
    TransComputeBarrierBlock();
}

void VulkanComputeBase::updateDescriptorSet(VkDescriptorSet& descriptorSet, 
    std::vector<vkbase::Texture2DGS>& images,
    std::vector<vkgs::Buffer*>& buffers)
{
    // Write descriptor sets
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size() + 1);
    
    std::unique_ptr<VkDescriptorImageInfo[]> imageInfos = std::make_unique<VkDescriptorImageInfo[]>(images.size());
    for (int i = 0; i < images.size(); ++i) {
        imageInfos[i] = { images[i].vkSampler, images[i].vkImageView, images[i].vkImageLayout };
    }
    
    descriptorWrites[0] = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorCount = images.size();
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[0].pImageInfo = imageInfos.get();
    
    // Prepare buffer info
    std::unique_ptr<VkDescriptorBufferInfo[]> bufferInfos = std::make_unique<VkDescriptorBufferInfo[]>(buffers.size());
    for (int i = 0; i < buffers.size(); ++i) {
        bufferInfos[i] = { *buffers[i], 0, buffers[i]->size() };
    }

    for (int i = 0; i < buffers.size(); ++i) {
        descriptorWrites[i+1] = {};
        descriptorWrites[i+1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i+1].dstSet = descriptorSet;
        descriptorWrites[i+1].dstBinding = i+1;
        descriptorWrites[i+1].descriptorCount = 1;
        descriptorWrites[i+1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i+1].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSetsGS(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void VulkanComputeBase::updateDescriptorSet(
    VkDescriptorSet& descriptorSet,
    std::vector<vkgs::Buffer*>& buffers
) {

    // Prepare buffer info
    std::unique_ptr<VkDescriptorBufferInfo[]> bufferInfos = std::make_unique<VkDescriptorBufferInfo[]>(buffers.size());
    for (int i = 0; i < buffers.size(); ++i) {
        bufferInfos[i] = { *buffers[i], 0, buffers[i]->size() };
    }

    // Write descriptor sets
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());

    for (int i = 0; i < buffers.size(); ++i) {
        descriptorWrites[i] = {};
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSetsGS(device, buffers.size(), descriptorWrites.data(), 0, nullptr);
}

void VulkanComputeBase::bindPipeline2CommandBufferPushConstants(
    const STComputePipeLine& pipe, 
    const int PIPELINE_INDEX,
    uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
    std::vector<uint32_t>& pushConstants
) {
    vkCmdBindPipelineGS(computeRes.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipelines[PIPELINE_INDEX]);
    vkCmdBindDescriptorSetsGS(computeRes.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipelineLayout, 0, 1, &pipe.descriptorSet, 0, nullptr);

    // Push constants
    vkCmdPushConstantsGS(
        computeRes.commandBuffer, 
        pipe.pipelineLayout, 
        VK_SHADER_STAGE_COMPUTE_BIT, 
        0, sizeof(uint32_t) * pushConstants.size(), 
        pushConstants.data()
    );

    vkCmdDispatchGS(computeRes.commandBuffer, dispatchX, dispatchY, dispatchZ);

    ComputeShaderMemoryBarrierBlock();
}