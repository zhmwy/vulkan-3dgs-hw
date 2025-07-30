//
// Created on 2025/7/30.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_UTILS_H
#define VULKAN_3DGS_HW_UTILS_H
#include <iostream>
#include <string>
#include "hilog/log.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>
#include "common/common.h"

using namespace cv;
using namespace std;
using namespace cv::dnn;

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200  // 全局domain宏，标识业务领域
#define LOG_TAG "MY_TAG"   // 全局tag宏，标识模块日志tag

void LetterBox(const cv::Mat& image, cv::Mat& outImage,
               cv::Vec4d& params,
               const cv::Size& newShape = cv::Size(640, 640),
               bool autoShape = false,
               bool scaleFill=false,
               bool scaleUp=true,
               int stride= 32,
               const cv::Scalar& color = cv::Scalar(114,114,114));

int Preprocessing(const std::vector<cv::Mat> &SrcImgs,
                              std::vector<cv::Mat> &OutSrcImgs,
                              std::vector<cv::Vec4d> &params, int imgSize=640);

#endif //VULKAN_3DGS_HW_UTILS_H
