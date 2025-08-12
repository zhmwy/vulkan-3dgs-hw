//
// Created on 2025/7/30.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_LANDMARK_H
#define VULKAN_3DGS_HW_LANDMARK_H
#include "napi/native_api.h"
#include "neural_network_runtime/neural_network_runtime.h"
#include "neural_network_runtime/neural_network_core.h"
#include "hiai_foundation/hiai_options.h"

#include <iostream>
#include <string>
#include "hilog/log.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include <sys/time.h>

#include <rawfile/raw_file_manager.h>

#include "HIAIModelManager.h"
#include <fstream>
#include <sys/stat.h> // For mkdir
#include <errno.h>    // For errno
#include <chrono>
#include "dirent.h"

#include "file/file_operator.h"
#include "common/common.h"
#include "../../../base/math/vector.h"
#include <Eigen/Dense>
#include <human/info.h>

struct FaceProcessingInfo {
    std::string imageName;
    cv::Point faceCenter;
    cv::Rect2d roi_bbox;
    double angle;
    bool isWithinBounds;
};

struct FaceResult {
    cv::Rect box;
    float confidence;
};

std::vector<Base::Math::Vec3> Get2dLandmarks(std::vector<float>& param, cv::Rect2d& face_box);
int ProcessKeypoints(const std::string& inputPath);
int ProcessLandmarks(const std::string& scenePath, std::pair<int, int>& selectedIndices, std::vector<FaceProcessingInfo>& faceInfos);
void detectFace(const std::string& inputPath, std::vector<FaceProcessingInfo>& faceInfos, std::pair<int, int> selectedIndices);

#endif //VULKAN_3DGS_HW_LANDMARK_H
