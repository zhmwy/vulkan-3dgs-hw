
#ifndef VULKAN_3DGS_HW_PATCH_COMPUTE_H
#define VULKAN_3DGS_HW_PATCH_COMPUTE_H

#include "vulkancomputebase.h"
#include "cores/dataloader.h"

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
    
    DataLoader::InputInitStruct inputGaussian;
};

#endif //VULKAN_3DGS_HW_PATCH_COMPUTE_H
