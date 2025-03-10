//
// Created on 2024/11/18.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef POINTRIX_HUMAN_FIT_SMPL_H
#define POINTRIX_HUMAN_FIT_SMPL_H

#endif //POINTRIX_HUMAN_FIT_SMPL_H
#include <rawfile/raw_file_manager.h>
#include "hilog/log.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "napi/native_api.h"
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200  // 全局domain宏，标识业务领域
#define LOG_TAG "MY_TAG"   // 全局tag宏，标识模块日志tag

#include <chrono>
#include "napi/native_api.h"
#include "definition/def.h"
#include "toolbox/Singleton.hpp"

#include "smpl/SMPL.h"
#include <Eigen/Dense>
#include <random>

// napi_value FitSMPL(napi_env env, napi_callback_info info);
void FitSMPL(std::string inputPath);
int get_theta_from_skeleton(std::string path, std::vector<Eigen::Vector3d>& thetas);