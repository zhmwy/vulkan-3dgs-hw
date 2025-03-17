#include "smpl/SMPL.h"

namespace smpl {

SMPL::SMPL() noexcept(true) :
    m__modelPath(),
    m__vertPath(),
    m__faceIndices(),
    m__shapeBlendBasis(),
    m__poseBlendBasis(),
    m__templateRestShape(),
    m__jointRegressor(),
    m__kinematicTree(),
    m__weights(),
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{
}

SMPL::SMPL(std::string &modelPath, 
    std::string &vertPath) noexcept(false) :
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{


    std::filesystem::path path(modelPath);
    if (std::filesystem::exists(path)) {
        m__modelPath = modelPath;
        m__vertPath = vertPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to initialize model path!");
    }
}


SMPL::SMPL(const SMPL& smpl) noexcept(false)
{
    try {
        *this = smpl;
    }
    catch(std::exception &e) {
        throw;
    }
}

SMPL::~SMPL() noexcept(true)
{
}

SMPL &SMPL::operator=(const SMPL& smpl) noexcept(false)
{
    std::filesystem::path path(smpl.m__modelPath);
    if (std::filesystem::exists(path)) {
        m__modelPath = smpl.m__modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to copy model path!");
    }

    try {
        m__vertPath = smpl.m__vertPath;
        m__blender = smpl.m__blender;
        m__regressor = smpl.m__regressor;
        m__transformer = smpl.m__transformer;
        m__skinner = smpl.m__skinner;
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // soft copy
    //
    if (smpl.m__faceIndices.rows() == FACE_INDEX_NUM &&
        smpl.m__faceIndices.cols() == 3) {
        m__faceIndices = smpl.m__faceIndices;
    }

    if (smpl.m__shapeBlendBasis.rows() == VERTEX_NUM &&
        smpl.m__shapeBlendBasis.cols() == SHAPE_BASIS_DIM * 3) {
        m__shapeBlendBasis = smpl.m__shapeBlendBasis;
    }

    if (smpl.m__poseBlendBasis.rows() == VERTEX_NUM &&
        smpl.m__poseBlendBasis.cols() == POSE_BASIS_DIM * 3) {
        m__poseBlendBasis = smpl.m__poseBlendBasis;
    }

    if (smpl.m__jointRegressor.rows() == JOINT_NUM &&
        smpl.m__jointRegressor.cols() == VERTEX_NUM) {
        m__jointRegressor = smpl.m__jointRegressor;
    }

    if (smpl.m__templateRestShape.rows() == VERTEX_NUM &&
        smpl.m__templateRestShape.cols() == 3) {
        m__templateRestShape = smpl.m__templateRestShape;
    }

    if (smpl.m__kinematicTree.rows() == 2 &&
        smpl.m__kinematicTree.cols() == JOINT_NUM) {
        m__kinematicTree = smpl.m__kinematicTree;
    }

    if (smpl.m__weights.rows() == VERTEX_NUM &&
        smpl.m__weights.cols() == JOINT_NUM) {
        m__weights = smpl.m__weights;
    }

    return *this;
}

void SMPL::setModelPath(const std::string &modelPath) noexcept(false)
{
    std::filesystem::path path(modelPath);
    if (std::filesystem::exists(path)) {
        m__modelPath = modelPath;
    }
    else {
        throw smpl_error("SMPL", "Failed to initialize model path!");
    }

    return;
}

void SMPL::setVertPath(const std::string &vertexPath) noexcept(false)
{
    m__vertPath = vertexPath;

    return;
}

Eigen::MatrixXd SMPL::getRestShape() noexcept(false)
{
    Eigen::MatrixXd restShape;

    try {
        restShape = m__regressor.getRestShape();
    }
    catch(std::exception &e) {
        throw;
    }
    return restShape;
}

Eigen::MatrixXi  SMPL::getFaceIndex() noexcept(false)
{
    if (m__faceIndices.rows() == FACE_INDEX_NUM && m__faceIndices.cols() == 3) {
        return m__faceIndices;
    } else {
        throw smpl_error("SMPL", "Failed to get face indices!");
    }
}

Eigen::MatrixXd SMPL::getRestJoint() noexcept(false)
{
    Eigen::MatrixXd joints;

    try {
        joints = m__regressor.getJoint();
    }
    catch (std::exception &e) {
        throw;
    }

    return joints;
}

Eigen::MatrixXd SMPL::getPosedJoint() noexcept(false)
{
    Eigen::MatrixXd joints;
    Eigen::MatrixXd vertices;

    try {
        vertices = m__skinner.getVertex(); // (VERTEX_NUM, 3)
        //m__jointRegressor: (JOINT_NUM, VERTEX_NUM)
        joints = (vertices.transpose() * m__jointRegressor.transpose()).transpose(); //(JOINT_NUM, 3)
    }
    catch (std::exception &e) {
        throw;
    }

    return joints;
}

Eigen::MatrixXd SMPL::getVertex() noexcept(false)
{
    Eigen::MatrixXd vertices;

    try {
        vertices = m__skinner.getVertex();
    }
    catch(std::exception &e) {
        throw;
    }

    return vertices;
}

void SMPL::launch(
    Eigen::MatrixXd &beta, 
    Eigen::MatrixXd &theta) noexcept(false)
{
    if (beta.rows() != 1 || beta.cols() != SHAPE_BASIS_DIM ||
        theta.rows() != JOINT_NUM || theta.cols() !=  3) 
        {
        OH_LOG_ERROR(LOG_APP, "Beta or Theta not in correct shape");
        
        throw smpl_error("SMPL", "Cannot launch a SMPL model!");
    }

    try {
        //
        // blend shapes
        //
        OH_LOG_INFO(LOG_APP, "*******************blend shapes");
        
        m__blender.setBeta(beta);
        m__blender.setTheta(theta);
        for(int i=0;i<24;++i){
            OH_LOG_INFO(LOG_APP, "thetas are: %{public}f, %{public}f, %{public}f", theta(i, 0), theta(i, 1), theta(i, 2));
        }
        
        m__blender.setShapeBlendBasis(m__shapeBlendBasis);
        m__blender.setPoseBlendBasis(m__poseBlendBasis);
        m__blender.blend();

        Eigen::MatrixXd shapeBlendShape = m__blender.getShapeBlendShape();
        Eigen::MatrixXd poseBlendShape = m__blender.getPoseBlendShape();
        Eigen::MatrixXd poseRotation = m__blender.getPoseRotation();

        //
        // regress joints
        //
        OH_LOG_INFO(LOG_APP, "******************regress joints");
        
        m__regressor.setTemplateRestShape(m__templateRestShape);
        m__regressor.setJointRegressor(m__jointRegressor);
        m__regressor.setShapeBlendShape(shapeBlendShape);
        m__regressor.setPoseBlendShape(poseBlendShape);
        m__regressor.regress();


        Eigen::MatrixXd restShape = m__regressor.getRestShape();
        OH_LOG_INFO(LOG_APP, "restShape: %{public}f",restShape(0, 1));
        
        Eigen::MatrixXd joints = m__regressor.getJoint();
        Eigen::MatrixXd restShapeData = restShape;
        
        //
        // transform
        //
        OH_LOG_INFO(LOG_APP, "*******************transform");
        
        m__transformer.setKinematicTree(m__kinematicTree);
        m__transformer.setJoint(joints);
        m__transformer.setPoseRotation(poseRotation);
        m__transformer.transform();
        Eigen::MatrixXd transformation = m__transformer.getTransformation();//24,16
        
        //
        // skinning
        //
        OH_LOG_INFO(LOG_APP, "*******************skinning");
        
        m__skinner.setWeight(m__weights);
        m__skinner.setRestShape(restShape);//6890,3
        m__skinner.setTransformation(transformation);
        m__skinner.skinning();
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}


void SMPL::out_new() noexcept(false)
{
    Eigen::MatrixXd vertices = m__skinner.getVertex();
    OH_LOG_INFO(LOG_APP, "vertices: %{public}ld, %{public}ld", vertices.rows(), vertices.cols());

    if (vertices.rows() == VERTEX_NUM && vertices.cols() == 3
        && m__faceIndices.rows() == FACE_INDEX_NUM && m__faceIndices.cols() == 3) {
        std::ofstream file(m__vertPath);
        OH_LOG_INFO(LOG_APP, "ofstream OK");
        
        // 提取指定索引的顶点
        Eigen::MatrixXd slice = vertices; 

        // 导出顶点数据
        for (int64_t i = 0; i < VERTEX_NUM; i++) {
//             OH_LOG_INFO(LOG_APP, "%{public}f %{public}f %{public}f", slice(i, 0), slice(i, 1), slice(i, 2));
            
            file << 'v' << ' '
                 << slice(i, 0) << ' '
                 << slice(i, 1) << ' ' 
                 << slice(i, 2) << '\n';
        }

        // 导出面片索引
        for (int64_t i = 0; i < FACE_INDEX_NUM; i++) {
//             OH_LOG_INFO(LOG_APP, "%{public}d %{public}d %{public}d", m__faceIndices(i, 0), m__faceIndices(i, 1), m__faceIndices(i, 2));
            
            file << 'f' << ' '
                 << m__faceIndices(i, 0) << ' '
                 << m__faceIndices(i, 1) << ' '
                 << m__faceIndices(i, 2) << '\n';
        }
    }
    else {
        throw smpl_error("SMPL", "Cannot export the deformed mesh!");
    }

    return;
}


void SMPL::out(int64_t index) noexcept(false)
{
    Eigen::MatrixXd vertices = m__skinner.getVertex(); // (N, 6890, 3)

    if (vertices.rows() == VERTEX_NUM && vertices.cols() == 3
        && m__faceIndices.rows() == FACE_INDEX_NUM && m__faceIndices.cols() == 3) {
        std::ofstream file(m__vertPath);

        // 提取指定索引的顶点
        Eigen::RowVectorXd slice = vertices.row(index); // (6890, 3) -> (3)

        // 导出顶点数据
        for (int64_t i = 0; i < VERTEX_NUM; i++) {
            file << 'v' << ' '
                 << slice(i, 0) << ' '
                 << slice(i, 1) << ' ' 
                 << slice(i, 2) << '\n';
        }

        // 导出面片索引
        for (int64_t i = 0; i < FACE_INDEX_NUM; i++) {
            file << 'f' << ' '
                 << m__faceIndices(i, 0) << ' '
                 << m__faceIndices(i, 1) << ' '
                 << m__faceIndices(i, 2) << '\n';
        }
    }
    else {
        throw smpl_error("SMPL", "Cannot export the deformed mesh!");
    }

    return;
}

}