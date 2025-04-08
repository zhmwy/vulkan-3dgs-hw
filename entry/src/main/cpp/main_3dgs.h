//
// Created on 2025/4/8.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#pragma once
#ifndef VULKAN_3DGS_HW_MAIN_H
#define VULKAN_3DGS_HW_MAIN_H

#endif //VULKAN_3DGS_HW_MAIN_H

#include <rawfile/raw_file_manager.h>
#include <multimedia/image_framework/image_source_mdk.h>
#include <multimedia/image_framework/image_pixel_map_mdk.h>
#include "file/file_operator.h"
// #include "napi/native_api.h"
// #include "interface.h"

napi_value getSyncPixelMap(napi_env env, napi_callback_info info);
napi_value getSandBoxPixelMap(napi_env env, napi_callback_info info);
napi_value InitProgress(napi_env env, napi_callback_info info);
napi_value Train3DGS(napi_env env, napi_callback_info info);

