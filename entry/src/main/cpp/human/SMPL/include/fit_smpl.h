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
#include "common/common.h"
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

// napi_value FitSMPL(napi_env env, napi_callback_info info);
void FitSMPL(std::string inputPath, std::vector<std::vector<float>>& head_bboxes, std::vector<std::vector<float>>& body_bboxes);
int get_theta_from_skeleton(std::string path, std::vector<Eigen::Vector3d>& thetas);


// Function declarations - keeping the same as original
Matrix3d get_joint_local_coordinate_system(const map<string, Vector3d>& joint_coords);
Matrix3d get_subjoint_local_coordinate_system(const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const string& parent_joint_name, const string& joint_name);
Matrix3d get_knee_local_coordinate_system(const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const string& parent_joint_name, const string& joint_name);
Vector3d calculate_axis_angle(const Vector3d& local_vector, const string& side, const Vector3d& default_axis, int ignore_axis = -1);
pair<Vector3d, Matrix3d> compute_joint_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const map<string, string>& joint_names, const vector<string>& parent_joints, const Vector3d& default_axis, int ignore_axis);
Vector3d compute_subjoint_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const map<string, string>& joint_names, const Vector3d& default_axis, int ignore_axis);
pair<Vector3d, Matrix3d> compute_shoulder_axis_angle(const string& side, const map<string, Vector3d>& joint_coords);
Vector3d compute_elbow_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R);
pair<Vector3d, Matrix3d> compute_hip_axis_angle(const string& side, const map<string, Vector3d>& joint_coords);
Vector3d compute_knee_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R);