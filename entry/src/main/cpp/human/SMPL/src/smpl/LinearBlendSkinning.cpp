#include "smpl/LinearBlendSkinning.h"

//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {


LinearBlendSkinning::LinearBlendSkinning() noexcept(true) :

    m__restShape(),
    m__transformation(),
    m__weights(),
    m__posedVert()
{
}

LinearBlendSkinning::LinearBlendSkinning(Eigen::MatrixXd &weights) noexcept(false) {
    if (weights.rows() == VERTEX_NUM && weights.cols() == JOINT_NUM) {
        m__weights = weights;
    } else {
        throw smpl_error("LinearBlendSkinning", 
            "Failed to initialize linear blend weights!");
    }
}

LinearBlendSkinning::LinearBlendSkinning(const LinearBlendSkinning &linearBlendSkinning) noexcept(false) 
    : m__posedVert(linearBlendSkinning.m__posedVert) 
{
    try {
        *this = linearBlendSkinning;
    } catch (std::exception &e) {
        throw;
    }
}


LinearBlendSkinning::~LinearBlendSkinning() noexcept(true)
{
}


LinearBlendSkinning &LinearBlendSkinning::operator=(
    const LinearBlendSkinning &linearBlendSkinning) noexcept(false)
{


    if (linearBlendSkinning.m__restShape.rows() == BATCH_SIZE * VERTEX_NUM &&
        linearBlendSkinning.m__restShape.cols() == 3) {
        m__restShape = linearBlendSkinning.m__restShape;
    } else {
        throw smpl_error("LinearBlendSkinning", "Failed to copy deformed shape in rest pose!");
    }

    // 复制 transformation
    if (linearBlendSkinning.m__transformation.rows() == BATCH_SIZE * JOINT_NUM &&
        linearBlendSkinning.m__transformation.cols() == 16) {
        m__transformation = linearBlendSkinning.m__transformation;
    } else {
        throw smpl_error("LinearBlendSkinning", "Failed to copy world transformation!");
    }

    // 复制 weights
    if (linearBlendSkinning.m__weights.rows() == VERTEX_NUM &&
        linearBlendSkinning.m__weights.cols() == JOINT_NUM) {
        m__weights = linearBlendSkinning.m__weights;
    } else {
        throw smpl_error("LinearBlendSkinning", "Failed to copy linear blend weights!");
    }

    // 复制 posedVert
    if (linearBlendSkinning.m__posedVert.rows() == BATCH_SIZE * VERTEX_NUM &&
        linearBlendSkinning.m__posedVert.cols() == 3) {
        m__posedVert = linearBlendSkinning.m__posedVert;
    }

    return *this;
}

void LinearBlendSkinning::setRestShape(const Eigen::MatrixXd &restShape) noexcept(false)
{
    if (restShape.rows() == BATCH_SIZE * VERTEX_NUM && restShape.cols() == 3) {
        m__restShape = restShape;
    }
    else {
        throw smpl_error("LinearBlendSkinning", "Failed to set deformed shape in rest pose!");
    }

    return;
}

void LinearBlendSkinning::setWeight(const Eigen::MatrixXd &weights) noexcept(false)
{
    if (weights.rows() == VERTEX_NUM && weights.cols() == JOINT_NUM) {

        m__weights = weights;
    }
    else {
        throw smpl_error("LinearBlendSkinning", "Failed to set linear blend weights!");
    }

    return;
}


void LinearBlendSkinning::setTransformation(const Eigen::MatrixXd &transformation) noexcept(false)
{
    // 
    
    if (transformation.size() == 16 * JOINT_NUM) {
        m__transformation = transformation;
    }
    else {
        throw smpl_error("LinearBlendSkinning", "Failed to set world transformation!");
    }

    return;
}


Eigen::MatrixXd LinearBlendSkinning::getVertex() noexcept(false)
{
    Eigen::MatrixXd vertices;
    
    if (m__posedVert.rows() == VERTEX_NUM &&
        m__posedVert.cols() == 3) {
        vertices = m__posedVert;
    }
    else {
        throw smpl_error("LinearBlendSkinning", 
            "Failed to get vertices of new pose!");
    }

    return vertices;
}


/**skinning
 * 
 * Brief
 * ----------
 * 
 *      Do all the skinning stuffs.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void LinearBlendSkinning::skinning() noexcept(false)
{
    //
    // Cartesian coordinates to homogeneous coordinates
    //

    Eigen::MatrixXd restShapeHomo;
    try {
        restShapeHomo = cart2homo(m__restShape); // (6890, 4)
        // std::cout<<restShapeHomo <<std::endl;
    }
    catch (std::exception &e) {
        throw;
    }

    //
    // linear blend skinning
    //
    // 计算 coefficients，不需要 transpose，直接重新调整顺序即可
    Eigen::MatrixXd coefficients = m__weights * m__transformation;//6890 * 16


    // 计算 verticesHomo
    Eigen::Matrix4d xxx;  // 定义为 4x4 矩阵而不是动态大小
    Eigen::MatrixXd verticesHomo(VERTEX_NUM, 4);  // 假设 verticesHomo 行数和 restShapeHomo 匹配


    for (int i = 0; i < VERTEX_NUM; ++i) {
        // 将 coefficients 的第 i 行映射为 4x4 矩阵
        Eigen::MatrixXd xxx = coefficients.row(i);
        xxx.resize(4,4);
        verticesHomo.row(i) = xxx.transpose() * (restShapeHomo.row(i)).transpose();  // 6890*4
    }


    //
    // homogeneous coordinates to Cartesian coordinates
    //
    try {
        m__posedVert = homo2cart(verticesHomo);
        
    }
    catch (std::exception &e) {
        throw;
    }

    return;
}


/**cart2homo
 * 
 * Brief
 * ----------
 * 
 *      Convert Cartesian coordinates to homogeneous coordinates.
 * 
 * Argument
 * ----------
 * 
 *      @cart: - Tensor -
 *          Vectors in Cartesian coordinates, (N, 6890, 3).
 * 
 * Return
 * ----------
 * 
 *      @homo: - Tensor -
 *          Vectors in homogeneous coordinates, (N, 6890, 4).
 * 
 */
Eigen::MatrixXd LinearBlendSkinning::cart2homo(Eigen::MatrixXd &cart) 
    noexcept(false)
{
    if (cart.rows() != VERTEX_NUM || cart.cols() !=   3) {
        throw smpl_error("LinearBlendSkinning", "Cannot convert Cartesian coordinates to homogeneous one!");
    }

    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(VERTEX_NUM, 1); 
    Eigen::MatrixXd homo(cart.rows(), cart.cols() + 1); 
    
    homo << cart, ones; 

    return homo;
}


/**homo2cart
 * 
 * Brief
 * ----------
 * 
 *      Convert Cartesian coordinates to homogeneous coordinates.
 * 
 * Argument
 * ----------
 * 
 *      @homo: - Tensor -
 *          Vectors in homogeneous coordinates, (N, 6890, 4).
 * 
 * Return
 * ----------
 * 
 *      @cart: - Tensor -
 *          Vectors in Cartesian coordinates, (N, 6890, 3).
 * 
 */
Eigen::MatrixXd LinearBlendSkinning::homo2cart(Eigen::MatrixXd &homo) 
    noexcept(false)
{
    if (homo.rows() != VERTEX_NUM || homo.cols() !=  4) {
        throw smpl_error("LinearBlendSkinning", 
            "Cannot convert homogeneous coordinates to Cartesian ones!");
    }

    // Extract the homogeneous weight (the last column)
    Eigen::MatrixXd homoW = homo.col(3); // (N, 1)
    Eigen::MatrixXd cartesianCoords = homo.leftCols(3); 
    cartesianCoords.array() /= homoW.replicate(1, 3).array();


    return cartesianCoords;
}


//=============================================================================
} // namespace smpl
//=============================================================================
