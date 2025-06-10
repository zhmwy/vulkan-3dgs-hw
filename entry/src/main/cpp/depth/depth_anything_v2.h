//
// Created on 2025/6/9.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_DEPTH_ANYTHING_V2_H
#define VULKAN_3DGS_HW_DEPTH_ANYTHING_V2_H
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <dirent.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include "hilog/log.h"
#include "HIAIModelManager.h"

int ProcessDepthEstimation(std::string inputPath);

#endif //VULKAN_3DGS_HW_DEPTH_ANYTHING_V2_H
