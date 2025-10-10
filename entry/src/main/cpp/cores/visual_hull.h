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

// DataLoader::InputInitStruct performVisualHullSegmentation(
//     const DataLoader::InputInitStruct& target,
//     int num_cameras = 4,
//     int max_outside_count = 0,
//     float threshold_scale = 1.0f, 
//     bool use_median = false
// );

DataLoader::InputInitStruct performVisualHullSegmentation(
    const DataLoader::InputInitStruct& target, 
    float max_outside_percentage = 0.125,
    float threshold_scale = 1.0f, 
    bool use_median = 1.0f);

#endif //VULKAN_3DGS_HW_VISUAL_HULL_H
