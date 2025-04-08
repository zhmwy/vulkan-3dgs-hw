//
// Created on 2025/3/3.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_RENDER_TO_CV_MAT_H
#define VULKAN_3DGS_HW_RENDER_TO_CV_MAT_H
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <multimedia/image_framework/image_source_mdk.h>
#include <multimedia/image_framework/image_pixel_map_mdk.h>
#include <rawfile/raw_file.h>
#include <rawfile/raw_file_manager.h>
#include <hilog/log.h>
#include "stdio.h"
#include "napi/native_api.h"

napi_value getSyncPixelMap(napi_env env, napi_callback_info info);
napi_value getSandBoxPixelMap(napi_env env, napi_callback_info info);

#endif //VULKAN_3DGS_HW_RENDER_TO_CV_MAT_H
