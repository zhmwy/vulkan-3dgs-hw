
#pragma once
#ifndef VULKAN_3DGS_HW_MAIN_H
#define VULKAN_3DGS_HW_MAIN_H

#include <string>
#include <napi/native_api.h>

class InterfaceVulkan{
public:
    InterfaceVulkan(){};
    virtual void InitVulkan(std::string datapath, std::string modelpath, napi_threadsafe_function g_progress_cb_train)=0; //datapath是输入路径，modelpath是输出路径，优化后的txt覆盖原路径即可
    virtual void TrainVulkan()=0;
    virtual float GetProcessVulkan()=0;
    virtual ~InterfaceVulkan()=default;
};


extern "C" InterfaceVulkan* createInstanceVulkan();
extern "C" void destoryInstanceVulkan(InterfaceVulkan* instance);




#endif //VULKAN_3DGS_HW_MAIN_H
