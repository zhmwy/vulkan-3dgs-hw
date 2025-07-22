//
// Created on 2025/7/15.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_BFM_DATA_H
#define VULKAN_3DGS_HW_BFM_DATA_H

#endif //VULKAN_3DGS_HW_BFM_DATA_H
#include <vector>

extern const std::vector<float> u_base;  // 204维
extern const std::vector<float> w_shp_base;  //204 x 40 = 8160维
extern const std::vector<float> w_exp_base;  //204 x 10 = 2040维
extern const std::vector<float> param_mean;
extern const std::vector<float> param_std;
