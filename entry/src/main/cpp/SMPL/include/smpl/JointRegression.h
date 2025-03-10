#ifndef JOINT_REGRESSION_H
#define JOINT_REGRESSION_H
#include <Eigen/Dense>
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "iostream"

namespace smpl {

class JointRegression final
{
private: // PRIVATE ATTRIBUTES

    Eigen::MatrixXd m__restShape;
    Eigen::MatrixXd m__shapeBlendShape;
    Eigen::MatrixXd m__poseBlendShape;
    Eigen::MatrixXd m__templateRestShape;

    Eigen::MatrixXd m__joints;
    Eigen::MatrixXd m__jointRegressor;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

private: // PRIVATE METHODS

    void linearCombine() noexcept(false);
    void jointRegress() noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    JointRegression() noexcept(true);
    // 修改声明，使用常量引用
    JointRegression(const Eigen::MatrixXd &jointRegressor, 
        const Eigen::MatrixXd &templateRestShape) noexcept(false);

    JointRegression(const JointRegression &jointRegression) noexcept(false);
    ~JointRegression() noexcept(true);

    // %% Operators %%
    JointRegression &operator=(const JointRegression &jointRegression)
        noexcept(false);
    
    // %% Getter and Setter %%
    void setShapeBlendShape(const Eigen::MatrixXd &shapeBlendShape) 
        noexcept(false);
    void setPoseBlendShape(const Eigen::MatrixXd &poseBlendShape)
        noexcept(false);
    void setTemplateRestShape(const Eigen::MatrixXd &templateRestShape)
        noexcept(false);
    void setJointRegressor(const Eigen::MatrixXd &jointRegressor)
        noexcept(false);
    
    Eigen::MatrixXd getRestShape() noexcept(false);
    Eigen::MatrixXd getJoint() noexcept(false);

    // %% Joint Regression Wrapper %%
    void regress() noexcept(false);
};


//=============================================================================
} // namespace smpl
//=============================================================================
#endif // JOINT_REGRESSION_H
//=============================================================================
