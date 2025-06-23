#include "patch_compute.h"

VulkanComputePatch::VulkanComputePatch()
{
    
#ifdef VULKAN_HW
    enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    enabledDeviceExtensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    enabledDeviceExtensions.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
#endif
    
}

VulkanComputePatch::~VulkanComputePatch()
{
    ;
}

void VulkanComputePatch::loadPatchData(std::string& dataFolder)
{
    if (dataFolder.empty()){
        throw std::runtime_error("The input data folder is empty.");
    }
    
    if (dataFolder.back() == '/') {
        dataFolder.pop_back();
    }
    
    inputGaussian.basepath = dataFolder.substr(0, dataFolder.find_last_of("/"));
    FileOperator::GetInstance()->set_defaultDir(inputGaussian.basepath);
    
    DataLoader::loadPatchInput(dataFolder, inputGaussian, params.height, params.width);
}

bool VulkanComputePatch::prepare()
{
    if (prepared) {
        return true;
    }
    
	VulkanComputeBase::prepare();
}

void VulkanComputePatch::compute()
{
    ;
}

void VulkanComputePatch::initBuffersValues()
{
    ;
}