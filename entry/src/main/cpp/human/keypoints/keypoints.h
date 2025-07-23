//
// Created on 2024/10/31.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef POINTRIX_HUMAN_KEYPOINTS_H
#define POINTRIX_HUMAN_KEYPOINTS_H

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
#include "utils.h"
#include <fstream>
#include <sys/stat.h> // For mkdir
#include <errno.h>    // For errno
#include <chrono>
#include "dirent.h"

#include "file/file_operator.h"
#include "common/common.h"
#include "../../../base/math/vector.h"
#include <Eigen/Dense>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200  // 全局domain宏，标识业务领域
#define LOG_TAG "MY_TAG"   // 全局tag宏，标识模块日志tag

using namespace cv;
using namespace std;
using namespace cv::dnn;
using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

struct FaceProcessingInfo {
    std::string imageName;
    cv::Point faceCenter;
    cv::Rect2d roi_bbox;
    double angle;
    bool isWithinBounds;
};

OH_NN_ReturnCode RunModel();
OH_NN_ReturnCode LoadModel(const std::string& modelPath);
OH_NN_ReturnCode LoadModel(NativeResourceManager* resourceMgr);
OH_NN_ReturnCode LoadImage(NativeResourceManager* resourceMgr, const std::string& imagePath, const std::string& sandboxPath, vector<Vec4d>& params);
// napi_value GetKeypoints(napi_env env, napi_callback_info info);
std::vector<Base::Math::Vec3> Get2dLandmarks(std::vector<float>& param, cv::Rect2d& face_box, std::string& savePath);
int ProcessKeypoints(const std::string& inputPath);
int ProcessLandmarks(const std::string& scenePath, std::pair<int, int>& selectedIndices, std::vector<FaceProcessingInfo>& faceInfos);
napi_value InitProgress(napi_env env, napi_callback_info info);
#endif //POINTRIX_HUMAN_KEYPOINTS_H