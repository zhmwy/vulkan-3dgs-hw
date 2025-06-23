
#ifndef VULKAN_3DGS_HW_PATCH_COMPUTE_H
#define VULKAN_3DGS_HW_PATCH_COMPUTE_H

#include "vulkancomputebase.h"
#include "cores/dataloader.h"
#include "cores/buffer.h"

#define VULKAN_HW

class VulkanComputePatch : public VulkanComputeBase
{
public:
    VulkanComputePatch();
    ~VulkanComputePatch();
    
    struct PatchParams
    {
        uint32_t width  = 480;
        uint32_t height = 640;
    }params;
    
    void loadPatchData(std::string& dataFolder);
    
	virtual void compute();
	virtual bool prepare();
    virtual void initBuffersValues();
    
protected:
    
    struct PatchBuffers {
        vkgs::Buffer intrParams;
        vkgs::Buffer cameraR;
        vkgs::Buffer cameraT;
        vkgs::Buffer depthRange;
        
        vkgs::Buffer costs;
        vkgs::Buffer planeHypotheses;
        vkgs::Buffer selectedViews;
        
        vkgs::Buffer srcPairs;
    
        std::vector<vkbase::Texture2DGS> depths; 
    } patchBuffers;
    
    
    DataLoader::InputInitStruct inputGaussian;
    
    int image_width  = 480;
    int image_height = 640;
    
    STComputePipeLine pipeRandomInitialization;
    
    void setupRandomInitializationPipeline();
    
    void loadAssets(int origin_width, int origin_height);
    void preparePatchBuffers();
};

#endif //VULKAN_3DGS_HW_PATCH_COMPUTE_H
