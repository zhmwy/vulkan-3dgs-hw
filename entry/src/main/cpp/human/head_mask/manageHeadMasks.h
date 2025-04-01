//
// Created on 2025/3/11.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_MANAGEHEADMASKS_H
#define VULKAN_3DGS_HW_MANAGEHEADMASKS_H
#include "hilog/log.h"

#include "opencv2/opencv.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include "common/common.h"

void manageHeadMasks(std::string dataPath, int& width, int& height, std::vector<float>& intr);
cv::Mat cropImage(const cv::Mat& image, const cv::Point& center, const cv::Size& size, cv::Point& actualTopLeft);
std::vector<float> calculateNewIntrinsics(
    float fx, float fy, 
    float cx, float cy,
    const cv::Point& crop_origin, 
    const cv::Size& highres_crop_size,
    float scale_factor);
#endif //VULKAN_3DGS_HW_MANAGEHEADMASKS_H
