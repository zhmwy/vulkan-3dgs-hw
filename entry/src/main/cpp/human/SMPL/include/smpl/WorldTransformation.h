#ifndef WORLD_TRANSFORMATION_H
#define WORLD_TRANSFORMATION_H

#include <Eigen/Dense>
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "toolbox/TorchEx.hpp"
#include "iostream"

namespace smpl {

class WorldTransformation final
{
private: // PRIVATE ATTRIBUTES
    Eigen::MatrixXd m__joints;        // 存储关节信息
    Eigen::MatrixXd m__poseRot;       // 存储姿态旋转信息
    Eigen::MatrixXi m__kineTree;      // 存储运动学树
    Eigen::MatrixXd m__transformations; // 存储变换矩阵

protected: // PROTECTED ATTRIBUTES
    // (没有改变，保留原样)

public: // PUBLIC ATTRIBUTES
    // (没有改变，保留原样)

private: // PRIVATE METHODS
    // %% Transformations %%
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> localTransform(const std::vector<Eigen::Matrix<double, 4, 3>> &poseRotHomo);    Eigen::MatrixXd globalTransform(Eigen::MatrixXd &localTransformations) noexcept(false);  // 全局变换
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> globalTransform(const std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> &localTransformations)  noexcept(false);  // 全局变换
    void relativeTransform(const std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> &globalTransformations) noexcept(false);  // 相对变换

protected: // PROTECTED METHODS
    // (没有改变，保留原样)

public: // PUBLIC METHODS
    // %% Constructor and Destructor %%
    WorldTransformation() noexcept(true);  // 构造函数
    WorldTransformation(const Eigen::MatrixXi &kineTree) noexcept(false);  // 构造函数，接受运动学树
    WorldTransformation(const WorldTransformation &worldTransformation) noexcept(false);  // 拷贝构造函数
    ~WorldTransformation() noexcept(true);  // 析构函数

    // %% Operators %%
    WorldTransformation &operator=(const WorldTransformation &worldTransformation) noexcept(false);  // 赋值操作符

    // %% Setter and Getter %%
    void setJoint(const Eigen::MatrixXd &joints) noexcept(false);  // 设置关节信息
    void setPoseRotation(const Eigen::MatrixXd &poseRot) noexcept(false);  // 设置姿态旋转信息
    void setKinematicTree(const Eigen::MatrixXi &kineTree) noexcept(false);  // 设置运动学树

    Eigen::MatrixXd getTransformation() noexcept(false);  // 获取变换矩阵

    // %% Transformation wrapper %%
    void transform();  // 执行变换
};

}

#endif