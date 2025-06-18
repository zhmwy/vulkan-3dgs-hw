/*
* Vulkan Example base class
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "common/common.h"
#include "file/file_operator.h"

VkResult VulkanExampleBase::createInstance()
{
    LOGI("VulkanExampleBase::createInstance is comming");
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
        LOGI("VulkanExampleBase::create Instance vkCreateInstance is null");
    }
    return vkCreateInstanceGS(&instanceCreateInfo, nullptr, &instance);
}

void VulkanExampleBase::createPipelineCache()
{
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VK_CHECK_RESULT(vkCreatePipelineCacheGS(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}

bool VulkanExampleBase::prepare()
{
	createCommandPool();
// 	createCommandBuffers();
//	createSynchronizationPrimitives();
	setupRenderPass();
	createPipelineCache();
// 	setupFrameBuffer();
    LOGI("VulkanExampleBase prepare end");
    return true;
}

VkPipelineShaderStageCreateInfo VulkanExampleBase::loadShader(std::string fileName, VkShaderStageFlagBits stage, bool isBase)
{
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = stage;
	shaderStage.module = vks::tools::loadSpvShader(fileName.c_str(), device);
	shaderStage.pName = "main";
	assert(shaderStage.module != VK_NULL_HANDLE);
	if (isBase) {
		shaderModules.push_back(shaderStage.module);
	} else {
		exampleModules.push_back(shaderStage.module);
	}

	return shaderStage;
}

void VulkanExampleBase::prepareFrame()
{
}

void VulkanExampleBase::submitFrame()
{
}

VulkanExampleBase::VulkanExampleBase()
{
	libLoaded = vks::ohos::loadVulkanLibrary();
	if (libLoaded) {
        LOGI("VulkanExampleBase dlopen libvulkan.so success ");
	} else {
        LOGE("VulkanExampleBase dlopen libvulkan.so failed ");
	}
}

VulkanExampleBase::~VulkanExampleBase()
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
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		vkDestroyFramebufferGS(device, frameBuffers[i], nullptr);
	}

	for (auto& shaderModule : shaderModules)
	{
		vkDestroyShaderModuleGS(device, shaderModule, nullptr);
	}
	for (auto& shaderModule : exampleModules)
	{
		vkDestroyShaderModuleGS(device, shaderModule, nullptr);
	}

	vkDestroyPipelineCacheGS(device, pipelineCache, nullptr);

	vkDestroyCommandPoolGS(device, cmdPool, nullptr);

	vkDestroySemaphoreGS(device, semaphores.presentComplete, nullptr);
	vkDestroySemaphoreGS(device, semaphores.renderComplete, nullptr);

	delete vulkanDevice;

	vkDestroyInstanceGS(instance, nullptr);
}

bool VulkanExampleBase::initVulkan()
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

	vks::ohos::loadVulkanFunctions(instance);

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
	vulkanDevice = new vks::VulkanDevice(physicalDevice);

	// Derived examples can enable extensions based on the list of supported extensions read from the physical device
	getEnabledExtensionsGS();

	VkResult res = vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, deviceCreatepNextChain);
	if (res != VK_SUCCESS) {
		LOGE("initVulkan: create logic device failed");
		return false;
	}
	device = vulkanDevice->logicalDevice;

    // Get a graphics queue from the device
    vkGetDeviceQueueGS(device, vulkanDevice->queueFamilyIndices.graphics, 0, &queue);
//     // Get a compute queue from the device
//     vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute_queue);
    // Find a suitable depth format
	VkBool32 validDepthFormat = vks::tools::supportedDepthFormat(physicalDevice, &depthFormat);
	// assert(validDepthFormat);

	// Create synchronization objects
	VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
	// Create a semaphore used to synchronize image presentation
	// Ensures that the image is displayed before we start submitting new commands to the queue
	VK_CHECK_RESULT(vkCreateSemaphoreGS(device, &semaphoreCreateInfo, nullptr, &semaphores.presentComplete));
	// Create a semaphore used to synchronize command submission
	// Ensures that the image is not presented until all commands have been submitted and executed
	VK_CHECK_RESULT(vkCreateSemaphoreGS(device, &semaphoreCreateInfo, nullptr, &semaphores.renderComplete));

	// Set up submit info structure
	// Semaphores will stay the same during application lifetime
	// Command buffer submission info is set by each example
	submitInfo = vks::initializers::submitInfo();
	submitInfo.pWaitDstStageMask = &submitPipelineStages;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &semaphores.presentComplete;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &semaphores.renderComplete;
	LOGI("initVulkan successfully");
	return true;
}

void VulkanExampleBase::createSynchronizationPrimitives()
{
}

void VulkanExampleBase::createCommandPool()
{
               LOGI("VulkanExampleBase::createCommandPool");
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
//	cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
	cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK_RESULT(vkCreateCommandPoolGS(device, &cmdPoolInfo, nullptr, &cmdPool));
}

void VulkanExampleBase::getEnabledFeatures() {}
void VulkanExampleBase::getEnabledExtensionsGS() {}

void VulkanExampleBase::setupRenderPass()
{
   LOGI("VulkanExampleBase::setupRenderPass");
	std::array<VkAttachmentDescription, 2> attachments = {};
	// Color attachment
//	attachments[0].format = swapChain.colorFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	// Depth attachment
	attachments[1].format = depthFormat;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference = {};
	depthReference.attachment = 1;
	depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription = {};
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pColorAttachments = &colorReference;
	subpassDescription.pDepthStencilAttachment = &depthReference;
	subpassDescription.inputAttachmentCount = 0;
	subpassDescription.pInputAttachments = nullptr;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = nullptr;
	subpassDescription.pResolveAttachments = nullptr;

	// Subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
	dependencies[0].dependencyFlags = 0;

	dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].dstSubpass = 0;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].srcAccessMask = 0;
	dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
	dependencies[1].dependencyFlags = 0;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpassDescription;
	renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
	renderPassInfo.pDependencies = dependencies.data();

	VK_CHECK_RESULT(vkCreateRenderPassGS(device, &renderPassInfo, nullptr, &renderPass));
}



std::string VulkanExampleBase::getPath(std::string path) 
{
	std::string fileName = FileOperator::GetInstance()->GetFileAbsolutePath(path);
	return fileName;
}