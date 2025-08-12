//
// Created on 2024/10/28.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef POINTRIX_HUMAN_TRIANGULATION_H
#define POINTRIX_HUMAN_TRIANGULATION_H

#endif //POINTRIX_HUMAN_TRIANGULATION_H
#include "napi/native_api.h"
#include "hilog/log.h"
#include "stdio.h"
#include "stdlib.h"
#include <filemanagement/fileio/oh_fileio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <rawfile/raw_file.h>
#include <rawfile/raw_dir.h>
#include <rawfile/raw_file_manager.h>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <sys/time.h>
#include "common/common.h"
#include"human/info.h"

// napi_value ProcessTriangulation(napi_env env, napi_callback_info info);
std::map<std::string, double> ProcessTriangulation(std::string inputPath);

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200  // 全局domain宏，标识业务领域
#define LOG_TAG "MY_TAG"   // 全局tag宏，标识模块日志tag

// Structure to store camera extrinsics (rotation and translation)
struct CameraExtrinsics {
    cv::Mat R; // Rotation matrix
    cv::Mat t; // Translation vector
};

// Structure to store 2D keypoints with confidence
struct KeyPoint2D {
    double x;
    double y;
    double confidence;
};

// Structure to store keypoints for each frame
struct FrameData {
    int frame_id;
    std::string imageName;
    std::vector<KeyPoint2D> keypoints;
};

// Structure to store camera intrinsics
struct CameraIntrinsics {
    double fx;
    double fy;
    double cx;
    double cy;
    int width;
    int height;
};