//
// Created on 2025/7/30.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_FACE_H
#define VULKAN_3DGS_HW_FACE_H
#include "cores/dataloader.h"
#include <opencv2/opencv.hpp>
#include "hilog/log.h"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include "common/common.h"
#include "landmark/landmark.h"
#include "human/info.h"

// Helper struct to store keypoint data
struct KeypointData {
    std::string filename;
    cv::Rect2f bbox;  // human bounding box
    cv::Point2f nose, leftEye, rightEye, leftEar, rightEar;
    cv::Point2f leftShoulder, rightShoulder;
    float leftEarConf, rightEarConf, leftShoulderConf, rightShoulderConf;
    // We can add other keypoints and confidences if needed
};


std::string getBaseName(const std::string& path);
std::vector<FaceProcessingInfo> analyzeFaces(const std::map<std::string, double>& faceAngles,
                                            const std::vector<cv::Point>& faceCenters,
                                            const std::vector<std::string>& imgList);
std::pair<int, int> selectTwoImageIndices(const std::vector<FaceProcessingInfo>& faceInfos);

void manageHeadMasks(std::string dataPath, int& width, int& height, std::vector<float>& intr);
cv::Mat cropImage(const cv::Mat& image, const cv::Point& center, const cv::Size& size, cv::Point& actualTopLeft);
void cropAndSaveFace(const cv::Mat& orig_img, const cv::Point& faceCenter, cv::Size& cropSize, 
                    const std::string& scenePath, const std::string& basename);
cv::Size calculateOptimalFaceDimensions(const std::vector<KeypointData>& keypoints) ;
std::vector<KeypointData> parseKeypointsFile(const std::string& filepath);
cv::Mat applyBeautyFilters(const cv::Mat& image, float smoothing_level = 0.5f, float whitening_level = 0.3f);
std::vector<float> calculateNewIntrinsics(
    float fx, float fy, 
    float cx, float cy,
    const cv::Point& crop_origin, 
    const cv::Size& highres_crop_size,
    float scale_factor);

std::vector<cv::Point> projectHeadCenter(DataLoader::InputInitStruct& data);
void createEnlargedFace(const std::string scenepath, DataLoader::InputInitStruct& data, 
                                   std::vector<FaceProcessingInfo>& faceInfos,
                                   const std::pair<int, int>& selectedIndices, const bool& masked=false);
void FaceTune(DataLoader::InputInitStruct& inputGaussian, std::string scenePath);
#endif //VULKAN_3DGS_HW_FACE_H
