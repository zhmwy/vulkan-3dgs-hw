#ifndef SMPL_H
#define SMPL_H

#include <string>
#include <unordered_set>

#include "smpl/BlendShape.h"
#include "smpl/JointRegression.h"
#include "smpl/WorldTransformation.h"
#include "smpl/LinearBlendSkinning.h"
#include <string>
#include <Eigen/Dense>
#include "Eigen/src/Core/Matrix.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include "definition/def.h"
#include "toolbox/TorchEx.hpp"
#include <vector>

#include "hilog/log.h"
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200  // 全局domain宏，标识业务领域
#define LOG_TAG "MY_TAG"   // 全局tag宏，标识模块日志tag

namespace smpl {

class SMPL final
{

private: // PRIVATE ATTRIBUTES


protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

protected: // PROTECTED METHODS

public: // PUBLIC METHODS
    
    std::string m__modelPath;
    std::string m__vertPath;

    Eigen::MatrixXi m__faceIndices;
    Eigen::MatrixXd m__shapeBlendBasis;
    
    Eigen::MatrixXd m__poseBlendBasis;
    Eigen::MatrixXd m__templateRestShape;
    Eigen::MatrixXd m__jointRegressor;
    Eigen::MatrixXi m__kinematicTree;

    BlendShape m__blender;
    JointRegression m__regressor;
    WorldTransformation m__transformer;
    LinearBlendSkinning m__skinner;
    Eigen::MatrixXd m__weights;

    // %% Constructor and Destructor %%
    SMPL() noexcept(true);
    SMPL(std::string &modelPath, 
        std::string &vertPath) noexcept(false);
    SMPL(const SMPL& smpl) noexcept(false);
    ~SMPL() noexcept(true);

    // %% Operators %%
    SMPL &operator=(const SMPL& smpl) noexcept(false);

    // %% Setter and Getter %%
    void setModelPath(const std::string &modelPath) noexcept(false);
    void setVertPath(const std::string &vertexPath) noexcept(false);

    Eigen::MatrixXd getRestShape() noexcept(false);
    Eigen::MatrixXi getFaceIndex() noexcept(false);
    Eigen::MatrixXd getRestJoint() noexcept(false);
    Eigen::MatrixXd getPosedJoint() noexcept(false);
    Eigen::MatrixXd getVertex() noexcept(false);

    // %% Modeling %%
    void init() noexcept(false);
    void launch(
        Eigen::MatrixXd &beta,
        Eigen::MatrixXd &theta) noexcept(false);
    void out(int64_t index) noexcept(false);
    void out_new() noexcept(false);
};
//=============================================================================
} // namespace smpl
//=============================================================================
#endif // SMPL_H
//=============================================================================
