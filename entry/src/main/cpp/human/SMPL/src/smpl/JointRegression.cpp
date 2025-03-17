#include "smpl/JointRegression.h"

namespace smpl {

JointRegression::JointRegression() noexcept(true) :
    m__restShape(),
    m__shapeBlendShape(),
    m__poseBlendShape(),
    m__templateRestShape(),
    m__joints(),
    m__jointRegressor()
{
}

JointRegression::JointRegression(const Eigen::MatrixXd &jointRegressor,
    const Eigen::MatrixXd &templateRestShape) noexcept(false) :

    m__restShape(),
    m__shapeBlendShape(),
    m__poseBlendShape(),
    m__joints()
{
    if (jointRegressor.rows() == JOINT_NUM && jointRegressor.cols() == VERTEX_NUM) {
        m__jointRegressor = jointRegressor;  // 直接赋值
    }
    else {
        throw smpl_error("JointRegression", 
            "Failed to initialize joint regressor!");
    }

    if (templateRestShape.rows() == VERTEX_NUM && templateRestShape.cols() == 3) {
        m__templateRestShape = templateRestShape;  // 直接赋值
    }
    else {
        throw smpl_error("JointRegression", 
            "Failed to initialize template shape!");
    }
}

JointRegression::JointRegression(const JointRegression &jointRegression)
    noexcept(false) :
    m__restShape(),
    m__joints()
{
    try {
        *this = jointRegression;
    }
    catch(std::exception &e) {
        throw;
    }
}

JointRegression::~JointRegression() noexcept(true)
{
}

JointRegression &JointRegression::operator=(
    const JointRegression &jointRegression) noexcept(false)
{
    // hard copy
    if (jointRegression.m__jointRegressor.rows() == JOINT_NUM && jointRegression.m__jointRegressor.cols() == VERTEX_NUM) {
        m__jointRegressor = jointRegression.m__jointRegressor;
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy joint regressor!");
    }

    if (jointRegression.m__templateRestShape.rows() == VERTEX_NUM && jointRegression.m__templateRestShape.cols() == 3) {
        m__templateRestShape = jointRegression.m__templateRestShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to copy template shape!");
    }

    // soft copy
    m__restShape = jointRegression.m__restShape;
    m__joints = jointRegression.m__joints;

    return *this;
}

void JointRegression::setShapeBlendShape(const Eigen::MatrixXd &shapeBlendShape) noexcept(false)
{
    if (shapeBlendShape.rows() == BATCH_SIZE && shapeBlendShape.cols() == VERTEX_NUM * 3) {
        m__shapeBlendShape = shapeBlendShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set shape blend shape!");
    }
}

void JointRegression::setPoseBlendShape(const Eigen::MatrixXd &poseBlendShape) noexcept(false)
{
    if (poseBlendShape.rows() == BATCH_SIZE && poseBlendShape.cols() == VERTEX_NUM * 3) {
        m__poseBlendShape = poseBlendShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set pose blend shape!");        
    }
}

void JointRegression::setTemplateRestShape(
    const Eigen::MatrixXd &templateRestShape) noexcept(false)
{
    if (templateRestShape.rows() == VERTEX_NUM && templateRestShape.cols() == 3) {
        m__templateRestShape = templateRestShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set template shape!");
    }
}

void JointRegression::setJointRegressor(const Eigen::MatrixXd &jointRegressor) noexcept(false)
{
    if (jointRegressor.rows() == JOINT_NUM && jointRegressor.cols() == VERTEX_NUM) {
        m__jointRegressor = jointRegressor;
    }
    else {
        throw smpl_error("JointRegression", "Failed to set joint regressor!");
    }
}

Eigen::MatrixXd JointRegression::getRestShape() noexcept(false)
{
    Eigen::MatrixXd restShape;

    if (m__restShape.rows() == VERTEX_NUM && m__restShape.cols() == 3) {
        restShape = m__restShape;
    }
    else {
        throw smpl_error("JointRegression", "Failed to get rest shape!");
    }

    return restShape;
}

Eigen::MatrixXd JointRegression::getJoint() noexcept(false)
{
    Eigen::MatrixXd joints;

    if (m__joints.rows() == JOINT_NUM && m__joints.cols() ==  3) {
        joints = m__joints;
    }
    else {
        throw smpl_error("JointRegression", "Failed to get joints!");
    }

    return joints;
}

void JointRegression::regress() noexcept(false)
{
    // linearly combine shapes
    try {

        linearCombine();

    }
    catch(std::exception &e) {
        throw;
    }

    // joint regression
    try {
        jointRegress();

    }
    catch(std::exception &e) {
        throw;
    }
}

void JointRegression::linearCombine() noexcept(false)
{
    if (m__shapeBlendShape.rows() == BATCH_SIZE && m__shapeBlendShape.cols() == VERTEX_NUM * 3
        && m__poseBlendShape.rows() == BATCH_SIZE && m__poseBlendShape.cols() == VERTEX_NUM * 3
        && m__templateRestShape.rows() == VERTEX_NUM && m__templateRestShape.cols() == 3) {
        
        m__restShape.resize(3, VERTEX_NUM);  
        Eigen::MatrixXd xxx = m__restShape.transpose();  
        m__restShape = xxx;//VERTEX_NUM , 3

        m__shapeBlendShape.resize(3,VERTEX_NUM);
        xxx = m__shapeBlendShape.transpose();  
        m__shapeBlendShape = xxx;//VERTEX_NUM , 3

        m__poseBlendShape.resize(3,VERTEX_NUM);
        xxx = m__poseBlendShape.transpose();  
        m__poseBlendShape = xxx;//VERTEX_NUM , 3

        m__restShape = m__templateRestShape 
            + m__shapeBlendShape
            + m__poseBlendShape;

    }
    else {
        throw smpl_error("JointRegression", "Cannot linearly combine shapes!");
    }
}

void JointRegression::jointRegress() noexcept(false)
{
    if (m__shapeBlendShape.rows() == VERTEX_NUM && m__shapeBlendShape.cols() ==  3
        && m__templateRestShape.rows() == VERTEX_NUM && m__templateRestShape.cols() == 3) {
        
        Eigen::MatrixXd blendShape = m__templateRestShape 
            + m__shapeBlendShape;  // (6890, 3)

        m__joints.resize(3,24);

        m__joints = (blendShape.transpose() * m__jointRegressor.transpose()).transpose(); 

        
    }
    else {
        throw smpl_error("JointRegression", 
            "Cannot regress vertices to joints!");
    }
}

}


