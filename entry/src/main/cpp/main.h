//
// Created on 2025/3/6.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

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
#include <sys/time.h>

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

napi_value Train3DGS(napi_env env, napi_callback_info info);

#endif //VULKAN_3DGS_HW_MAIN_H
