//
// Created on 2024/10/28.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef POINTRIX_HUMAN_HUMAN_H
#define POINTRIX_HUMAN_HUMAN_H

#endif //POINTRIX_HUMAN_HUMAN_H
#include "triangulation/triangulation.h"
#include "keypoints/keypoints.h"
#include "SMPL/include/fit_smpl.h"
#include "segmentation/segment.h"
#include "head_mask/manageHeadMasks.h"

struct FaceProcessingInfo {
    std::string imageName;
    cv::Point faceCenter;
    double angle;
    bool isWithinBounds;
};

std::string getBaseName(const std::string& path);
std::vector<FaceProcessingInfo> analyzeFaces(const std::map<std::string, double>& faceAngles,
                                            const std::vector<cv::Point>& faceCenters,
                                            const std::vector<std::string>& imgList);
std::pair<int, int> selectTwoImageIndices(const std::vector<FaceProcessingInfo>& faceInfos);
napi_value HumanPreprocess_napi(napi_env env, napi_callback_info info);
std::vector<std::string> selectTwoImages(const std::map<std::string, double>& faceAngles, const std::string& folderPath);
bool HumanPreprocess(std::string inputPath, std::vector<float>& head_bbox, 
                    std::vector<float>& body_bbox, std::map<std::string, double>& faceAngles);