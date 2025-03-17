//
// Created on 2025/3/5.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_VULKANRENDER_H
#define VULKAN_3DGS_HW_VULKANRENDER_H
#include "model_compute.h"

class VulkanRender : public VulkanComputeExample
{
public:
    virtual bool prepare() override;
    virtual void draw() override;
};
#endif //VULKAN_3DGS_HW_VULKANRENDER_H
