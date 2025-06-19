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
// 	createCommandBuffers();
//	createSynchronizationPrimitives();
//	setupRenderPass();
	createPipelineCache();
// 	setupFrameBuffer();
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

	delete vulkanDevice;

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
	vulkanDevice = new vkbase::VulkanDevice(physicalDevice);

	// Derived examples can enable extensions based on the list of supported extensions read from the physical device
	getEnabledExtensionsGS();

	VkResult res = vulkanDevice->vdCreateLogicalDevice(enabledFeatures, enabledDeviceExtensions, deviceCreatepNextChain);
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