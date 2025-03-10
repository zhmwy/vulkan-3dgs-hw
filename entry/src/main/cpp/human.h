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

napi_value HumanPreprocess_napi(napi_env env, napi_callback_info info);
bool HumanPreprocess(std::string inputPath, std::vector<float>& head_bbox);