//
// Created on 2024/10/31.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef POINTRIX_HUMAN_SEGMENT_H
#define POINTRIX_HUMAN_SEGMENT_H

#endif //POINTRIX_HUMAN_KEYPOINTS_H
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
#include "../keypoints/HIAIModelManager.h"
#include "../keypoints/keypoints.h"
#include "../keypoints/utils.h"
#include <fstream>
#include <sys/stat.h>
#include <errno.h>
#include <chrono>
#include <dirent.h>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200  // 全局domain宏，标识业务领域
#define LOG_TAG "MY_TAG"   // 全局tag宏，标识模块日志tag

bool PostProcessingSeg(std::vector<std::vector<float>> result, std::vector<cv::Vec4d> &params, 
                        std::vector<OutputParams> &output, int inputSize, cv::Size imgSize);
int GetSegments(std::string inputPath);
int GetSegments_Batched(std::string inputPath);