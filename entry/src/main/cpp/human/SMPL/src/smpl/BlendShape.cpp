#include "smpl/BlendShape.h"

namespace smpl {

BlendShape::BlendShape() noexcept(true) :

    m__beta(),
    m__shapeBlendBasis(),
    m__shapeBlendShape(),
    m__theta(),
    m__restTheta(),
    m__poseRot(),
    m__restPoseRot(),
    m__poseBlendBasis(),
    m__poseBlendShape()
{
}

BlendShape::BlendShape(Eigen::MatrixXd &shapeBlendBasis,
        Eigen::MatrixXd &poseBlendBasis) noexcept(false) :
    m__beta(),
    m__shapeBlendBasis(),
    m__shapeBlendShape(),
    m__theta(),
    m__restTheta(),
    m__poseRot(),
    m__restPoseRot(),
    m__poseBlendBasis(),
    m__poseBlendShape()
{
    // 验证输入的矩阵尺寸
    if (shapeBlendBasis.rows() == VERTEX_NUM && shapeBlendBasis.cols() == 3 * SHAPE_BASIS_DIM) {
        m__shapeBlendBasis = shapeBlendBasis; 


    }
    else {
        throw smpl_error("BlendShape", "Failed to initialize shape blend basis!");
    }

    if (poseBlendBasis.rows() == VERTEX_NUM && poseBlendBasis.cols() == 3 * POSE_BASIS_DIM) {
        m__poseBlendBasis = poseBlendBasis; // 直接赋值，无需转换
    }
    else {
        throw smpl_error("BlendShape", "Failed to initialize pose blend basis!");
    }
}

BlendShape::BlendShape(const BlendShape &blendShape) noexcept(false) :
    m__beta(blendShape.m__beta),
    m__shapeBlendBasis(blendShape.m__shapeBlendBasis),
    m__shapeBlendShape(blendShape.m__shapeBlendShape),
    m__theta(blendShape.m__theta),
    m__restTheta(blendShape.m__restTheta),
    m__poseRot(blendShape.m__poseRot),
    m__restPoseRot(blendShape.m__restPoseRot),
    m__poseBlendBasis(blendShape.m__poseBlendBasis),
    m__poseBlendShape(blendShape.m__poseBlendShape)
{
    // 拷贝构造函数将成员逐一赋值
    try {
        // 如果有额外的成员需要初始化，可以在此处添加
    }
    catch (std::exception &e) {
        throw;
    }
}

BlendShape::~BlendShape() noexcept(true)
{
}

BlendShape &BlendShape::operator=(const BlendShape &blendShape) noexcept(false)
{
    //
    //  hard copy
    //

    // 直接拷贝 Eigen::MatrixXd 类型的成员
    m__shapeBlendBasis = blendShape.m__shapeBlendBasis;
    m__poseBlendBasis = blendShape.m__poseBlendBasis;
    m__beta = blendShape.m__beta;
    m__theta = blendShape.m__theta;
    m__restTheta = blendShape.m__restTheta;

    //
    //  soft copy
    //
    m__shapeBlendShape = blendShape.m__shapeBlendShape;
    m__poseBlendShape = blendShape.m__poseBlendShape;
    m__poseRot = blendShape.m__poseRot;
    m__restPoseRot = blendShape.m__restPoseRot;

    return *this;
}

void BlendShape::setBeta(const Eigen::MatrixXd &beta) noexcept(false)
{
    // 检查 beta 的维度是否正确
    if (beta.rows() == BATCH_SIZE && beta.cols() == SHAPE_BASIS_DIM) {
        // 直接赋值，因为 Eigen::MatrixXd 自带深拷贝
        m__beta = beta;
    }
    else {
        throw smpl_error("BlendShape", "Failed to set beta!");
    }

    return;
}

void BlendShape::setShapeBlendBasis(const Eigen::MatrixXd &shapeBlendBasis) noexcept(false)
{

    if (shapeBlendBasis.rows() == 3 * VERTEX_NUM && shapeBlendBasis.cols() ==  SHAPE_BASIS_DIM) {

        m__shapeBlendBasis = shapeBlendBasis;
    }
    else {
           
        throw smpl_error("BlendShape", "Failed to set shape blend basis!");
    }

    return;
}

void BlendShape::setTheta(const Eigen::MatrixXd &theta) noexcept(false)
{
    // 检查 theta 的维度是否正确

    
    if (theta.rows() == JOINT_NUM && theta.cols() ==  3) {

        m__theta = theta;
    }
    else {
        throw smpl_error("BlendShape", "Failed to set theta!");
    }

    return;
}

void BlendShape::setRestTheta(const Eigen::MatrixXd &restTheta) noexcept(false)
{
    // 检查输入矩阵的维度是否符合预期
    if (restTheta.rows() == BATCH_SIZE && restTheta.cols() == JOINT_NUM * 3) {
        m__restTheta = restTheta; 
    } else {
        throw smpl_error("BlendShape", "Failed to set rest theta, invalid dimensions!");
    }
    
    return;
}

void BlendShape::setPoseBlendBasis(const Eigen::MatrixXd &poseBlendBasis) noexcept(false)
{
    // 检查 poseBlendBasis 的维度是否正确
    if (poseBlendBasis.rows() == 3 * VERTEX_NUM && poseBlendBasis.cols() ==  POSE_BASIS_DIM) {
        // 直接赋值，Eigen::MatrixXd 支持深拷贝
        m__poseBlendBasis = poseBlendBasis;
    }
    else {
        throw smpl_error("BlendShape", "Failed to set pose blend basis!");
    }

    return;
}

Eigen::MatrixXd BlendShape::getShapeBlendShape() noexcept(false)
{
    Eigen::MatrixXd shapeBlendShape;

    // 假设 m__shapeBlendShape 是 Eigen::MatrixXd 类型
    if (m__shapeBlendShape.rows() == BATCH_SIZE && m__shapeBlendShape.cols() == VERTEX_NUM * 3) {
        shapeBlendShape = m__shapeBlendShape;  
    } else {
        throw smpl_error("BlendShape", "Failed to get shape blend shape!");
    }

    return shapeBlendShape;
}

Eigen::MatrixXd BlendShape::getPoseRotation() noexcept(false)
{
    Eigen::MatrixXd poseRotation;

    if (m__poseRot.rows() ==  JOINT_NUM && m__poseRot.cols() == 9) {
        poseRotation = m__poseRot;  // 直接赋值
    }
    else {
        throw smpl_error("BlendShape", "Failed to get pose rotation!");
    }

    return poseRotation;
}

Eigen::MatrixXd BlendShape::getRestPoseRotation() noexcept(false)
{
    Eigen::MatrixXd restPoseRotation;

    // 检查 m__restPoseRot 是否具有正确的维度
    if (m__restPoseRot.rows() == JOINT_NUM && m__restPoseRot.cols() ==  3 * 3) {
        restPoseRotation = m__restPoseRot;  
    }
    else {
        throw smpl_error("BlendShape", "Failed to get rest pose rotation!");
    }

    return restPoseRotation;
}

Eigen::MatrixXd BlendShape::getPoseBlendShape() noexcept(false)
{
    Eigen::MatrixXd poseBlendShape;

    // 假设 m__poseBlendShape 是 Eigen::MatrixXd 类型，检查它的尺寸
    if (m__poseBlendShape.rows() == BATCH_SIZE && 
        m__poseBlendShape.cols() == VERTEX_NUM * 3) {
        poseBlendShape = m__poseBlendShape;  // 直接赋值
    }
    else {
        throw smpl_error("BlendShape", "Failed to get pose blend shape!");
    }

    return poseBlendShape;
}

void BlendShape::blend() noexcept(false)
{
    //
    // pose blend
    //
    try {
        poseBlend();// save result in <m__poseBlendShape> (N, 6890, 3).
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // shape blend
    //
    try {
        shapeBlend();// save result in <m__shapeBlendShape> (N, 6890, 3).
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}

void BlendShape::shapeBlend() noexcept(false)
{
    if (m__shapeBlendBasis.rows() == 3 *VERTEX_NUM && m__shapeBlendBasis.cols() ==  SHAPE_BASIS_DIM
        && m__beta.rows() == BATCH_SIZE && m__beta.cols() == SHAPE_BASIS_DIM) {
        m__shapeBlendShape = m__beta * m__shapeBlendBasis.transpose();  // (6890, 3)
    }
    else {
        throw smpl_error("BlendShape", "Cannot blend shape-dependent shape!");
    }

    return;



}

void BlendShape::poseBlend() noexcept(false)
{
    //
    // pose rotation and rest pose rotation
    //
    
    if (m__theta.rows() == JOINT_NUM  && m__theta.cols() ==  3) {
        try {
            m__poseRot = rodrigues(m__theta); // (24, 9) 行存储
        }
        catch(std::exception &e) {
            throw;
        }
    }
    else {
        throw smpl_error("BlendShape", "Cannot blend pose-dependent shape!");
    }
    
    if (m__restTheta.size() != 0 && m__restTheta.rows() == BATCH_SIZE && 
        m__restTheta.cols() == JOINT_NUM * 3) {
        try {
            m__restPoseRot = rodrigues(m__restTheta); // (24, 9)
        }
        catch(std::exception &e) {
            throw;
        }
    }
    else {//----------------if
        Eigen::Matrix3d identityMatrix = Eigen::Matrix3d::Identity();
    
        Eigen::MatrixXd m__restPoseRot(JOINT_NUM, 9);
        for (int i = 0; i < JOINT_NUM; ++i) {
            m__restPoseRot.row(i) << identityMatrix(0, 0), identityMatrix(0, 1), identityMatrix(0, 2),
                                    identityMatrix(1, 0), identityMatrix(1, 1), identityMatrix(1, 2),
                                    identityMatrix(2, 0), identityMatrix(2, 1), identityMatrix(2, 2);
        }
    }

    //
    // pose blend coefficients
    //
    Eigen::MatrixXd poseBlendCoeffs;
    try {
        poseBlendCoeffs = linRotMin();//1,207
        
    }
    catch(std::exception &e) {
        throw;
    }

    if (m__poseBlendBasis.rows() ==3 *  VERTEX_NUM && 
        m__poseBlendBasis.cols() == POSE_BASIS_DIM) {
        m__poseBlendShape = poseBlendCoeffs * m__poseBlendBasis.transpose(); // (1, 6890* 3)
//         OH_LOG_INFO(LOG_APP, "m__poseBlendBasis %{public}f %{public}f %{public}f", m__poseBlendBasis(6890, 0), m__poseBlendBasis(6891, 1), m__poseBlendBasis(6892, 2));
//         OH_LOG_INFO(LOG_APP, "m__poseBlendShape %{public}f %{public}f %{public}f", m__poseBlendShape(0, 256), m__poseBlendShape(0, 257), m__poseBlendShape(0, 258));
    }
    else {
        throw smpl_error("BlendShape", "Cannot blend pose-dependented shape!");
    }

    return;
}

Eigen::MatrixXd BlendShape::rodrigues(Eigen::MatrixXd &theta) noexcept(false)
{
    
    if (theta.rows() != JOINT_NUM || theta.cols() !=   3) {
        throw smpl_error("BlendShape", "Cannot do arbitrary rotation!");
    }

    // 旋转角度和轴
    const double eps = 1e-8;
    
    Eigen::MatrixXd angles = theta.rowwise().norm().replicate(1, 3); // (JOINT_NUM, 3)
    Eigen::MatrixXd axes = theta.array() / (angles.array() + eps);   // (JOINT_NUM, 3)



    std::vector<Eigen::Matrix3d> skew_matrices(JOINT_NUM);  // (JOINT_NUM, 3*3)

    for (int j = 0; j < JOINT_NUM; ++j) {
        double x = axes(j, 0), y = axes(j, 1), z = axes(j, 2);
        skew_matrices[j] << 0, -z, y,
                            z, 0, -x,
                            -y, x, 0;  // (3*3)
    }


    std::vector<Eigen::Matrix3d> rotations(JOINT_NUM);  // (JOINT_NUM, 3*3)
    for (int j = 0; j < JOINT_NUM; ++j) {
        double angle = angles(j, 0);
        Eigen::Matrix3d skew_matrix = skew_matrices[j];

        rotations[j] = Eigen::Matrix3d::Identity() + sin(angle) * skew_matrix + (1 - cos(angle)) * skew_matrix * skew_matrix;  // (3*3)
    }

    Eigen::MatrixXd rotations_matrix(JOINT_NUM, 9);  // (JOINT_NUM, 9)按行存储
    for (int j = 0; j < JOINT_NUM; ++j) {
        rotations_matrix.row(j) << rotations[j](0, 0), rotations[j](0, 1), rotations[j](0, 2),
                                rotations[j](1, 0), rotations[j](1, 1), rotations[j](1, 2),
                                rotations[j](2, 0), rotations[j](2, 1), rotations[j](2, 2);
    }

    return rotations_matrix; 
}

Eigen::MatrixXd BlendShape::linRotMin() noexcept(false)
{
    //
    // unroll rotations
    //
    Eigen::MatrixXd unPoseRot, unRestPoseRot;

    try {
        unPoseRot = unroll(m__poseRot); // (N, 216)
        for(int i=0; i<24;++i){
            int j = 9*i;
//             OH_LOG_INFO(LOG_APP, "unPoseRot %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f",
//             unPoseRot(0, j), unPoseRot(0, j+1), unPoseRot(0, j+2),
//             unPoseRot(0, j+3), unPoseRot(0, j+4), unPoseRot(0, j+5), 
//             unPoseRot(0, j+6), unPoseRot(0, j+7), unPoseRot(0, j+8));
        }
        Eigen::Matrix3d identityMatrix = Eigen::Matrix3d::Identity();
        Eigen::MatrixXd m__restPoseRot(JOINT_NUM, 9);


        for (int i = 0; i < JOINT_NUM; ++i) {
            m__restPoseRot.row(i) << identityMatrix(0, 0), identityMatrix(0, 1), identityMatrix(0, 2),
                                    identityMatrix(1, 0), identityMatrix(1, 1), identityMatrix(1, 2),
                                    identityMatrix(2, 0), identityMatrix(2, 1), identityMatrix(2, 2);
        }
        unRestPoseRot = unroll(m__restPoseRot); // (N, 216)
        
    }
    
    catch (std::exception &e) {
        throw;
    }

    //
    // truncate rotations
    //

    unPoseRot = unPoseRot.block(0, 9, unPoseRot.rows(), unPoseRot.cols() - 9); // (N, 207)，丢掉第一个矩阵
    unRestPoseRot = unRestPoseRot.block(0, 9, unRestPoseRot.rows(), unRestPoseRot.cols() - 9); // (N, 207)
    for(int i=0; i<23;++i){
            int j = 9*i;
//             OH_LOG_INFO(LOG_APP, "unRestPoseRot %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f",
//             unRestPoseRot(0, j), unRestPoseRot(0, j+1), unRestPoseRot(0, j+2),
//             unRestPoseRot(0, j+3), unRestPoseRot(0, j+4), unRestPoseRot(0, j+5), 
//             unRestPoseRot(0, j+6), unRestPoseRot(0, j+7), unRestPoseRot(0, j+8));
    }
    
    //
    // pose blend coefficients
    //
    Eigen::MatrixXd poseBlendCoeffs = unPoseRot - unRestPoseRot; // (N, 207)

    return poseBlendCoeffs;
}

Eigen::Matrix<double, 1, 24 * 9, Eigen::RowMajor> BlendShape::unroll(Eigen::MatrixXd &rotation) noexcept(false)
{
    if (rotation.rows() != JOINT_NUM || rotation.cols() !=  9) {
        throw smpl_error("BlendShape", "Cannot unroll a rotation!");
    }

    Eigen::Matrix<double, 1 , 24 * 9 , Eigen::RowMajor> unRotation;
    
    int index = 0;
    for (int i = 0; i < rotation.rows(); ++i) {
        for (int j = 0; j < rotation.cols(); ++j) {
            unRotation(0, index) = rotation(i, j);
            ++index;
        }
    }

    return unRotation;
}

//=============================================================================
} // namespace smpl
//=============================================================================
