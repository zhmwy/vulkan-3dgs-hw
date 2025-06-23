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
    
    loadAssets(image_width,image_height);
    preparePatchBuffers();
    
    setupRandomInitializationPipeline();
}

void VulkanComputePatch::compute()
{
    ;
}

void VulkanComputePatch::initBuffersValues()
{
    ;
}

void VulkanComputePatch::loadAssets(int origin_width, int origin_height) 
{
    LOGI("Load Assets: loading images.");
    for (const std::string& filename : inputGaussian.img_list) {
        LOGI("Load Assets: loading image %{public}s", filename.c_str());
        cv::Mat img = cv::imread(filename);
        if (img.empty()){
            std::cerr << "Failed to load image file: " << filename << std::endl;
            return;
        }
        cv::resize(img, img, cv::Size(origin_width, origin_height));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);
        
        vkbase::Texture2DGS texture_original;
        texture_original.fromImageBuffer(
            img.data,
            origin_width * origin_height * 4,
            VK_FORMAT_R8G8B8A8_UNORM,
            origin_width, origin_height, 
            vulkanDevice, computeRes.queue,
            VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL
        );
        
        inputGaussian.images.push_back(texture_original);
    }
}

void VulkanComputePatch::preparePatchBuffers()
{
    patchBuffers.intrParams = vkgs::Buffer(
        allocator, inputGaussian.images.size() * 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.cameraR = vkgs::Buffer(
        allocator, inputGaussian.images.size() * 9 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.cameraT = vkgs::Buffer(
        allocator, inputGaussian.images.size() * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.depthRange = vkgs::Buffer(
        allocator, inputGaussian.images.size() * 2 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.planeHypotheses = vkgs::Buffer(
        allocator, image_height * image_width * 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.costs = vkgs::Buffer(
        allocator, image_height * image_width * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.selectedViews = vkgs::Buffer(
        allocator, image_height * image_width * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.srcPairs = vkgs::Buffer(
        allocator, inputGaussian.images.size() * 32 * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
}

void VulkanComputePatch::setupRandomInitializationPipeline()
{
    ;
}