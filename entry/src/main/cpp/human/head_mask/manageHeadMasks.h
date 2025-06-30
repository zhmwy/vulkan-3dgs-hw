//
// Created on 2025/3/11.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_MANAGEHEADMASKS_H
#define VULKAN_3DGS_HW_MANAGEHEADMASKS_H
#include "hilog/log.h"

#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include "common/common.h"

// Helper struct to store keypoint data
struct KeypointData {
    std::string filename;
    cv::Rect2f bbox;  // human bounding box
    cv::Point2f nose, leftEye, rightEye, leftEar, rightEar;
    cv::Point2f leftShoulder, rightShoulder;
    float leftEarConf, rightEarConf, leftShoulderConf, rightShoulderConf;
    // We can add other keypoints and confidences if needed
};

void manageHeadMasks(std::string dataPath, int& width, int& height, std::vector<float>& intr);
cv::Mat cropImage(const cv::Mat& image, const cv::Point& center, const cv::Size& size, cv::Point& actualTopLeft);
void cropAndSaveFace(const cv::Mat& orig_img, const cv::Point& faceCenter, cv::Size& cropSize, 
                    const std::string& outputPath);
cv::Size calculateOptimalFaceDimensions(const std::vector<KeypointData>& keypoints) ;
std::vector<KeypointData> parseKeypointsFile(const std::string& filepath);
cv::Mat applyBeautyFilters(const cv::Mat& image, float smoothing_level = 0.5f, float whitening_level = 0.3f);
std::vector<float> calculateNewIntrinsics(
    float fx, float fy, 
    float cx, float cy,
    const cv::Point& crop_origin, 
    const cv::Size& highres_crop_size,
    float scale_factor);
#endif //VULKAN_3DGS_HW_MANAGEHEADMASKS_H
