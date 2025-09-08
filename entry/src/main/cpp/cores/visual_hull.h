//
// Created on 2025/9/8.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_VISUAL_HULL_H
#define VULKAN_3DGS_HW_VISUAL_HULL_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "dataloader.h"

DataLoader::InputInitStruct performVisualHullSegmentation(
    const DataLoader::InputInitStruct& target,
    float threshold_scale = 1.0f,
    bool use_median = false
);

#endif //VULKAN_3DGS_HW_VISUAL_HULL_H
