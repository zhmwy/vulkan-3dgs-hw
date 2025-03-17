#include "smpl/WorldTransformation.h"

namespace smpl {

WorldTransformation::WorldTransformation() noexcept(true) :
    m__joints(),
    m__poseRot(),
    m__kineTree(),
    m__transformations()
{
}

WorldTransformation::WorldTransformation(
    const Eigen::MatrixXi &kineTree) noexcept(false) :
    m__joints(),
    m__poseRot(),
    m__kineTree(),
    m__transformations()
{
    if (kineTree.rows() == 2 && kineTree.cols() == JOINT_NUM) {
        m__kineTree = kineTree;
    }
    else {
        throw smpl_error("WorldTransformation", 
            "Failed to initialize kinematic tree!");
    }
}

WorldTransformation::WorldTransformation(const WorldTransformation &worldTransformation) noexcept(false) :
    m__transformations()
{
    try {
        *this = worldTransformation;
    }
    catch(std::exception& e) {
        throw;
    }
}

WorldTransformation::~WorldTransformation() noexcept(true)
{
}

WorldTransformation &WorldTransformation::operator=(
        const WorldTransformation &worldTransformation) noexcept(false)
{
    //
    // hard copy
    //
    if (worldTransformation.m__joints.rows() == BATCH_SIZE && 
        worldTransformation.m__joints.cols() == JOINT_NUM * 3) {
        m__joints = worldTransformation.m__joints;
    } else {
        throw smpl_error("WorldTransformation", "Failed to copy joints");
    }

    if (worldTransformation.m__poseRot.rows() == BATCH_SIZE && 
        worldTransformation.m__poseRot.cols() == JOINT_NUM * 9) {
        m__poseRot = worldTransformation.m__poseRot;
    } else {
        throw smpl_error("WorldTransformation", 
            "Failed to copy pose rotations");
    }

    if (worldTransformation.m__kineTree.rows() == 2 && 
        worldTransformation.m__kineTree.cols() == JOINT_NUM) {
        m__kineTree = worldTransformation.m__kineTree;
    } else {
        throw smpl_error("WorldTransformation",
            "Failed to copy kinematic tree!");
    }

    //
    // soft copy
    //
    if (worldTransformation.m__transformations.rows() == BATCH_SIZE && 
        worldTransformation.m__transformations.cols() == JOINT_NUM * 16) {
        m__transformations = worldTransformation.m__transformations;
    }

    return *this;
}

void WorldTransformation::setJoint(const Eigen::MatrixXd &joints) noexcept(false)
{
    if (joints.rows() == JOINT_NUM && joints.cols() == 3) {
        m__joints = joints;
    }
    else {
        throw smpl_error("WorldTransformation", "Failed to set joints");
    }

    return;
}

void WorldTransformation::setPoseRotation(const Eigen::MatrixXd &poseRot) noexcept(false)
{
    if (poseRot.rows() == JOINT_NUM && poseRot.cols() ==  9) {

        
        m__poseRot = poseRot;

    }
    else {
        throw smpl_error("WorldTransformation", 
            "Failed to set pose rotations");
    }

    return;
}

void WorldTransformation::setKinematicTree(const Eigen::MatrixXi &kineTree) noexcept(false)
{
    if (kineTree.rows() == 2 && kineTree.cols() == JOINT_NUM) {
        m__kineTree = kineTree;
    }
    else {
        throw smpl_error("WorldTransformation", 
            "Failed to set kinematic tree!");
    }

    return;
}

Eigen::MatrixXd WorldTransformation::getTransformation() noexcept(false)
{
    Eigen::MatrixXd transformation;

    if (m__transformations.rows() == JOINT_NUM && 
        m__transformations.cols() ==   16) {
        transformation = m__transformations;
    }

    return transformation;
}








void WorldTransformation::transform() {
    //
    // rotations in homogeneous coordinates
    //
    std::vector<Eigen::Matrix<double, 4, 3>> poseRotHomo;

    for (int i = 0; i < JOINT_NUM; ++i) {
        Eigen::Matrix3d rot;
        rot << m__poseRot(i, 0), m__poseRot(i, 1), m__poseRot(i, 2),
               m__poseRot(i, 3), m__poseRot(i, 4), m__poseRot(i, 5),
               m__poseRot(i, 6), m__poseRot(i, 7), m__poseRot(i, 8);

        // Now create a 4x3 matrix by stacking rot and a zero row
        Eigen::Matrix<double, 4, 3> poseRotHomo_i;
        poseRotHomo_i.block<3, 3>(0, 0) = rot;

        poseRotHomo_i.row(3).setZero();

        poseRotHomo.push_back(poseRotHomo_i);



    }

    //
    // local transformation
    //
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> localTransformations;
    try {
        localTransformations = localTransform(poseRotHomo);
    } catch (std::exception &e) {
        throw;
    }

    //
    // global transformation
    //
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> globalTransformations;
    try {
        globalTransformations = globalTransform(localTransformations);
    } catch (std::exception &e) {
        throw;
    }

    //
    // relative transformation
    //
    try {
        relativeTransform(globalTransformations);
    } catch (std::exception &e) {
        throw;
    }

    return;
}

std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> WorldTransformation::localTransform(const std::vector<Eigen::Matrix<double, 4, 3>> &poseRotHomo) {
    if (poseRotHomo.size() != JOINT_NUM) {
        throw smpl_error("WorldTransformation",
                         "Cannot transform bones locally!");
    }

    std::vector<Eigen::Vector3d> translations(JOINT_NUM);

    // For joint 0:
    translations[0] = m__joints.row(0).transpose();





    for (int i = 1; i < JOINT_NUM; ++i) {
        int ancestor = static_cast<int>(m__kineTree(0, i));
        translations[i] = (m__joints.row(i) - m__joints.row(ancestor)).transpose();
    }

    // Now, construct local transformations
    std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> localTransformations;

    for (int i = 0; i < JOINT_NUM; ++i) {
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> localTransform_i = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Zero();
        localTransform_i.block<4, 3>(0, 0) = poseRotHomo[i];
        localTransform_i.col(3).head(3) = translations[i];
        localTransform_i(3, 3) = 1.0;
        localTransformations.push_back(localTransform_i);
        
    }

    return localTransformations;
}

std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> WorldTransformation::globalTransform(
    const std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> &localTransformations) {
    if (localTransformations.size() != JOINT_NUM) {
        throw smpl_error("WorldTransformation",
                         "Cannot transform bones globally!");
    }

    std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> globalTransformations(JOINT_NUM);

    // For joint 0:
    globalTransformations[0] = localTransformations[0];

    // For joints 1 to JOINT_NUM-1:
    for (int i = 1; i < JOINT_NUM; ++i) {
        int ancestor = static_cast<int>(m__kineTree(0, i));
        globalTransformations[i] = globalTransformations[ancestor] * localTransformations[i];
        
    }

    return globalTransformations;
}

void WorldTransformation::relativeTransform(
    const std::vector<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> &globalTransformations) {
    if (globalTransformations.size() != JOINT_NUM) {
        throw smpl_error("WorldTransformation",
                         "Cannot transform bones relatively!");
    }

    // Prepare m__transformations as Eigen::MatrixXd of size (JOINT_NUM, 16)
    m__transformations.resize(JOINT_NUM, 16);

    for (int i = 0; i < JOINT_NUM; ++i) {
        // Extract rotation part from globalTransformations[i]
        Eigen::Matrix3d rotation = globalTransformations[i].block<3, 3>(0, 0);

        // Compute eliminated_i
        Eigen::Vector3d eliminated_i = rotation * m__joints.row(i).transpose();

        // Build eliminatedHomo_i
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> eliminatedHomo_i = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Zero();
        eliminatedHomo_i.block<3, 1>(0, 3) = eliminated_i;

        // Compute relativeTransformation_i
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relativeTransformation_i = globalTransformations[i] - eliminatedHomo_i;

        // Flatten relativeTransformation_i into a row vector of size 16
        Eigen::Map<Eigen::Matrix<double, 1, 16, Eigen::RowMajor>> row(relativeTransformation_i.data());

        // Store in m__transformations
        m__transformations.row(i) = row;
    }
    
    return;
}



#if 0


void WorldTransformation::transform() noexcept(false)
{
    //
    // rotations in homogeneous coordinates
    //

    Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(JOINT_NUM,  3); 
    Eigen::MatrixXd poseRotHomo(BATCH_SIZE, JOINT_NUM * 3 + 3); 
    poseRotHomo << m__poseRot, zeros;

    //
    // local transformation
    //
    Eigen::MatrixXd localTransformations;
    try {
          std::cout <<"sdf"<< std::endl;
        localTransformations = localTransform(poseRotHomo);
          std::cout <<"sdf"<< std::endl;
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // global transformation
    //
    Eigen::MatrixXd globalTransformations;
    try {
        globalTransformations = globalTransform(localTransformations);
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // relative transformation
    //
    try {
        relativeTransform(globalTransformations);
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}


Eigen::MatrixXd WorldTransformation::localTransform(Eigen::MatrixXd &poseRotHomo) noexcept(false)
{
    if (poseRotHomo.rows() != BATCH_SIZE || poseRotHomo.cols() != JOINT_NUM * 3 + 3) {
        throw smpl_error("WorldTransformation", 
            "Cannot transform bones locally!");
    }

    std::vector<Eigen::MatrixXd> translations;
    translations.push_back(m__joints.row(0)); 

    Eigen::MatrixXd ancestor;
    Eigen::MatrixXd translation;
    for (int64_t i = 1; i < JOINT_NUM; i++) {
        ancestor = m__kineTree.row(i);  // 直接使用 double 类型
        translation = m__joints.row(i) - m__joints.row(ancestor(0)); 
        translations.push_back(translation); 
    }

    Eigen::MatrixXd localTranslations(BATCH_SIZE, JOINT_NUM * 3);
    for (int64_t i = 0; i < JOINT_NUM; i++) {
        localTranslations.col(i) = translations[i]; 
    }

    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(BATCH_SIZE, 1); 
    Eigen::MatrixXd localTransformationsHomo(BATCH_SIZE, JOINT_NUM * 3 + 1); 
    localTransformationsHomo << localTranslations, ones;

    Eigen::MatrixXd localTransformations(BATCH_SIZE, JOINT_NUM * 3 + 4);
    localTransformations << poseRotHomo, localTransformationsHomo;

    return localTransformations;
}

Eigen::MatrixXd WorldTransformation::globalTransform(Eigen::MatrixXd &localTransformations) noexcept(false)
{
    if (localTransformations.rows() != BATCH_SIZE || 
        localTransformations.cols() != JOINT_NUM * 3 + 4) {
        throw smpl_error("WorldTransformations", 
            "Cannot transform bones globally!");
    }

    std::vector<Eigen::MatrixXd> transformations;
    transformations.push_back(localTransformations.row(0)); 

    Eigen::MatrixXd ancestor;
    Eigen::MatrixXd transformation;
    for (int64_t i = 1; i < JOINT_NUM; i++) {
        ancestor = m__kineTree.row(i);
        transformation = transformations[ancestor(0)] * localTransformations.row(i);
        transformations.push_back(transformation); 
    }

    Eigen::MatrixXd globalTransformations(BATCH_SIZE, JOINT_NUM * 3 + 4);
    for (int64_t i = 0; i < JOINT_NUM; i++) {
        globalTransformations.row(i) = transformations[i];
    }

    return globalTransformations;
}

void WorldTransformation::relativeTransform(Eigen::MatrixXd &globalTransformations) noexcept(false)
{
    if (globalTransformations.rows() != BATCH_SIZE || 
        globalTransformations.cols() != JOINT_NUM * 3 + 4) {
        throw smpl_error("WorldTransformation", "Cannot transform bones relatively!");
    }

    // 从 globalTransformations 中提取出关节部分并进行元素级的乘法操作
    Eigen::MatrixXd eliminated = globalTransformations.block(0, 0, BATCH_SIZE, 3).array() * m__joints.array();

    // 初始化零矩阵，大小为 BATCH_SIZE x JOINT_NUM
    Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(BATCH_SIZE, JOINT_NUM);

    // 创建 homogeneous 变换矩阵，原来是 3x4 的矩阵，调整为 4x4
    Eigen::MatrixXd eliminatedHomo(BATCH_SIZE, JOINT_NUM * 4);
    eliminatedHomo.block(0, 0, BATCH_SIZE, 3) = eliminated; 
    eliminatedHomo.block(0, 3, BATCH_SIZE, JOINT_NUM).setZero();  // 填充 0 到剩余位置

    // 计算相对变换矩阵
    Eigen::MatrixXd relativeTransformations = globalTransformations - eliminatedHomo;

    // 存储最终的相对变换结果
    m__transformations = relativeTransformations;

    return;
}

#endif




} // namespace smpl

