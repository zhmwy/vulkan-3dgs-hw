#ifndef BLEND_SHAPE_H
#define BLEND_SHAPE_H
#include <Eigen/Dense>
#include "toolbox/Exception.h"
#include "definition/def.h"
#include "toolbox/TorchEx.hpp"
#include "toolbox/Exception.h"
#include <Eigen/Dense>
#include <iostream>

namespace smpl {

#include <Eigen/Dense> // 确保包含 Eigen 库

class BlendShape final
{

private: // PRIVATE ATTRIBUTES

    Eigen::MatrixXd m__beta;
    Eigen::MatrixXd m__shapeBlendBasis;
    Eigen::MatrixXd m__shapeBlendShape;

    Eigen::MatrixXd m__theta;
    Eigen::MatrixXd m__restTheta;
    Eigen::MatrixXd m__poseRot;
    Eigen::MatrixXd m__restPoseRot;
    Eigen::MatrixXd m__poseBlendBasis;
    Eigen::MatrixXd m__poseBlendShape;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    // %% Blend Shape Generation %%
    void shapeBlend() noexcept(false);
    void poseBlend() noexcept(false);
    Eigen::MatrixXd rodrigues(Eigen::MatrixXd &theta) noexcept(false);
    Eigen::Matrix<double, 1, 24 * 9, Eigen::RowMajor> unroll(Eigen::MatrixXd &rotation) noexcept(false);
    Eigen::MatrixXd linRotMin() noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    BlendShape() noexcept(true);
    BlendShape(Eigen::MatrixXd &shapeBlendBasis,
        Eigen::MatrixXd &poseBlendBasis) noexcept(false);
    BlendShape(const BlendShape &blendShape) noexcept(false);
    ~BlendShape() noexcept(true);

    // %% Operators %%
    BlendShape &operator=(const BlendShape &blendShape) noexcept(false);

    // %% Blend Shape Wrapper %%
    void blend() noexcept(false);

    // %% Setter and Getter %%
    void setBeta(const Eigen::MatrixXd &beta) noexcept(false);
    void setShapeBlendBasis(const Eigen::MatrixXd &shapeBlendBasis)
        noexcept(false);

    void setTheta(const Eigen::MatrixXd &theta) noexcept(false);
    void setRestTheta(const Eigen::MatrixXd &restTheta) noexcept(false);    
    void setPoseBlendBasis(const Eigen::MatrixXd &poseBlendBasis)
        noexcept(false);

    Eigen::MatrixXd getShapeBlendShape() noexcept(false);
    Eigen::MatrixXd getPoseRotation() noexcept(false);
    Eigen::MatrixXd getRestPoseRotation() noexcept(false);
    Eigen::MatrixXd getPoseBlendShape() noexcept(false);

};

} // namespace SMPL
#endif // BLEND_SHAPE_H

