//
// Created on 2025/3/6.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#pragma once
#ifndef VULKAN_3DGS_HW_MAIN_H
#define VULKAN_3DGS_HW_MAIN_H

#include <cstdint>
#include <string>
#include <unistd.h>
#include "common/common.h"
#include "file/file_operator.h"
#include "render/plugin_render.h"
#include "render/vulkanRender.h"
#include "queue"
#include "human/head_mask/manageHeadMasks.h"
#include <sys/time.h>

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

napi_value Train3DGS(napi_env env, napi_callback_info info);

class Interface{
public:
    Interface(){};
    virtual void Init(std::string datapath, std::string modelpath)=0; //datapath是输入路径，modelpath是输出路径，优化后的txt覆盖原路径即可
    virtual void Train()=0;
    virtual float GetProcess()=0;
    virtual ~Interface()=default;
};

extern "C" Interface* createInstance();
extern "C" void destoryInstance(Interface* instance);

#endif //VULKAN_3DGS_HW_MAIN_H
