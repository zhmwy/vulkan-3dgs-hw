//
// Created on 2025/3/6.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#pragma once
#ifndef VULKAN_3DGS_HW_MAIN_H
#define VULKAN_3DGS_HW_MAIN_H

#include <string>
#include <napi/native_api.h>

// napi_value getSyncPixelMap(napi_env env, napi_callback_info info);
// napi_value getSandBoxPixelMap(napi_env env, napi_callback_info info);


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


class InterfaceImplVulkan : public InterfaceVulkan {
private:
    std::string _dataPath;
    std::string _outputPath;
    float _progress;
    bool _isInitialized;
    bool _isTraining;
    napi_threadsafe_function _g_progress_cb_train;
    void updateProgress(float value);
    

public:
    InterfaceImplVulkan();
    ~InterfaceImplVulkan();
    void InitVulkan(std::string datapath, std::string outputpath, napi_threadsafe_function g_progress_cb_train);
    void TrainVulkan();
    float GetProcessVulkan();
};


#endif //VULKAN_3DGS_HW_MAIN_H
