#include "fit_smpl.h"

#define SINGLE_SMPL smpl::Singleton<smpl::SMPL>

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

// Function to read keypoints from a PLY file
Eigen::MatrixXd read_keypoints_from_ply(const std::string& filename) {
    // Open the PLY file
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        return Eigen::MatrixXd();
    }

    std::string line;
    int num_vertices = 0;
    bool header_ended = false;

    // Read the header to determine the number of vertices
    while (std::getline(infile, line)) {
        // Remove any leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line == "end_header") {
            header_ended = true;
            break;
        }

        // Parse the header lines
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element") {
            std::string element_type;
            iss >> element_type;
            if (element_type == "vertex") {
                iss >> num_vertices;
            }
            // We can add parsing for edges here if needed
        }
        // Skip other header lines (properties, comments, etc.)
    }

    if (!header_ended) {
        std::cerr << "Invalid PLY file: header not properly terminated with 'end_header'." << std::endl;
        return Eigen::MatrixXd();
    }

    if (num_vertices <= 0) {
        std::cerr << "No vertices found in the PLY file." << std::endl;
        return Eigen::MatrixXd();
    }

    // Prepare to read the vertex data
    Eigen::MatrixXd keypoints(num_vertices, 3); // To store x, y, z positions
    for (int i = 0; i < num_vertices; ++i) {
        if (!std::getline(infile, line)) {
            std::cerr << "Unexpected end of file when reading vertex data." << std::endl;
            return Eigen::MatrixXd();
        }

        std::istringstream iss(line);
        double x, y, z;
        int r, g, b;
        if (!(iss >> x >> y >> z >> r >> g >> b)) {
            std::cerr << "Error parsing vertex data at vertex index " << i << std::endl;
            return Eigen::MatrixXd();
        }

        keypoints(i, 0) = x;
        keypoints(i, 1) = y;
        keypoints(i, 2) = z;
    }

    // Skip the edge data or process it if needed
    // For now, we ignore the edges

    // Close the file
    infile.close();

    // Return the matrix of keypoints
    return keypoints;
}

// Define the matching indices between detected keypoints and SMPL keypoints
const std::vector<std::pair<int, int>> matching_indices = {
    {5, 16},  // left shoulder
    {6, 17},  // right shoulder
    {7, 18},  // left elbow
    {8, 19},  // right elbow
    {9, 20},  // left wrist
    {10, 21}, // right wrist
    {11, 1},  // left hip
    {12, 2},  // right hip
    {13, 4},  // left knee
    {14, 5},  // right knee
    {15, 7},  // left ankle
    {16, 8}   // right ankle
};

int count_lines(const std::string& filename) {
    std::ifstream file(filename);
    int line_count = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        line_count++;
    }
    
    file.close();
    return line_count;
}

void append_points3D(std::string& outputPath, Eigen::MatrixXd& vertices){
    // Write vertices to points3D.txt
    int num_lines = count_lines(outputPath);
    OH_LOG_INFO(LOG_APP, "Original number of lines is: %{public}d", num_lines);
    std::ofstream outFile(outputPath, std::ios::app);
    num_lines = count_lines(outputPath);
    OH_LOG_INFO(LOG_APP, "Number of lines after opening it: %{public}d", num_lines);
    if (!outFile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Failed to open file %{public}s for writing", outputPath.c_str());
    } else {
        outFile << std::fixed << std::setprecision(6);
        for (int64_t i = 0; i < VERTEX_NUM; ++i) {
            outFile << i << " "
                    << vertices(i, 0) << " "
                    << vertices(i, 1) << " "
                    << vertices(i, 2) << " "
                    << "125 125 125 1.000000\n";
        }
        num_lines = count_lines(outputPath);
        OH_LOG_INFO(LOG_APP, "Number of lines after writing: %{public}d", num_lines);
        outFile.close();
        OH_LOG_INFO(LOG_APP, "Successfully wrote vertices to %{public}s", outputPath.c_str());
    }
}

void write_points3D(std::string& outputPath, Eigen::MatrixXd& vertices, std::vector<float>& head_bbox){
    double x_min = 0, x_max = 0, y_min = 0, y_max = 0, z_min = 0, z_max = 0;
    bool first = true;

    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Failed to open file %{public}s for writing", outputPath.c_str());
    }

    outFile << std::fixed << std::setprecision(6);
    for (int64_t i = 0; i < VERTEX_NUM; ++i) {
        // Write vertex data to the file
        outFile << i << " "
                << vertices(i, 0) << " "
                << vertices(i, 1) << " "
                << vertices(i, 2) << " "
                << "125 125 125 1.000000\n";

        // Check if current index is in the specified ranges
        if ((i >= 0 && i < 549) || (i >= 3549 && i < 3999)) {
            double x = vertices(i, 0);
            double y = vertices(i, 1);
            double z = vertices(i, 2);

            if (first) {
                x_min = x_max = x;
                y_min = y_max = y;
                z_min = z_max = z;
                first = false;
            } else {
                x_min = std::min(x_min, x);
                x_max = std::max(x_max, x);
                y_min = std::min(y_min, y);
                y_max = std::max(y_max, y);
                z_min = std::min(z_min, z);
                z_max = std::max(z_max, z);
            }
        }
    }
    outFile.close();
    OH_LOG_INFO(LOG_APP, "Successfully wrote vertices to %{public}s", outputPath.c_str());

    if (!first) {
        head_bbox.push_back(static_cast<float>(x_min));
        head_bbox.push_back(static_cast<float>(x_max));
        head_bbox.push_back(static_cast<float>(y_min));
        head_bbox.push_back(static_cast<float>(y_max));
        head_bbox.push_back(static_cast<float>(z_min));
        head_bbox.push_back(static_cast<float>(z_max));
    } else {
        OH_LOG_ERROR(LOG_APP, "No points in the specified ranges to compute min-max");
    }
}

void FitSMPL(std::string inputPath, std::vector<float>& head_bbox) {
    auto begin = clk::now();
    
	std::string outputPath = inputPath + "/slam/smpl.txt";
	// 使用 Eigen 创建矩阵
	Eigen::MatrixXd beta(1, SHAPE_BASIS_DIM);
	Eigen::MatrixXd theta(JOINT_NUM , 3);  // 3 为每个关节的坐标维度
	Eigen::MatrixXd vertices(1, SHAPE_BASIS_DIM);  // 用于保存顶点信息
    
    beta.setZero();
    theta.setZero();
    
    // Get thetas from skeleton
    std::vector<Eigen::Vector3d> thetas;
    get_theta_from_skeleton(inputPath, thetas);
    
    // Map the thetas to correct rows in theta matrix
    if (!thetas.empty()) {
        // Left shoulder (16th row)
        theta.row(16) = thetas[0];
        // Right shoulder (17th row)
        theta.row(17) = thetas[1];
        // Left elbow (18th row)
        theta.row(18) = thetas[2];
        // Right elbow (19th row)
        theta.row(19) = thetas[3];
        // Left hip (1st row)
        theta.row(1) = thetas[4];
        // Right hip (2nd row)
        theta.row(2) = thetas[5];
        // Left knee (4th row)
        theta.row(4) = thetas[6];
        // Right knee (5th row)
        theta.row(5) = thetas[7];
    }
    
    SINGLE_SMPL::get()->init();
    SINGLE_SMPL::get()->launch(beta, theta);
    
    Eigen::MatrixXd smpl_joints = SINGLE_SMPL::get()->getPosedJoint();
    Eigen::MatrixXd detected_keypoints_all = read_keypoints_from_ply(inputPath + "/skeleton.ply");

    size_t num_matching = matching_indices.size();
    Eigen::MatrixXd smpl_keypoints(num_matching, 3);
    Eigen::MatrixXd detected_keypoints(num_matching, 3);

    for (size_t i = 0; i < num_matching; ++i) {
        int detected_idx = matching_indices[i].first;
        int smpl_idx = matching_indices[i].second;
        smpl_keypoints.row(i) = smpl_joints.row(smpl_idx);
        detected_keypoints.row(i) = detected_keypoints_all.row(detected_idx);
    }

    // Center the data
    Eigen::RowVector3d smpl_center = smpl_keypoints.colwise().mean();
    Eigen::RowVector3d detected_center = detected_keypoints.colwise().mean();

    Eigen::MatrixXd smpl_centered = smpl_keypoints.rowwise() - smpl_center;
    Eigen::MatrixXd detected_centered = detected_keypoints.rowwise() - detected_center;

    // Compute covariance matrix
    Eigen::Matrix3d covariance = smpl_centered.transpose() * detected_centered;

    // Perform SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d S = svd.singularValues();

    // Compute rotation matrix R
    Eigen::Matrix3d R = V * U.transpose();

    // Correct reflection issue if necessary
    if (R.determinant() < 0) {
        // Reflection detected, correcting for it
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // Compute scale
    double var_smple = smpl_centered.squaredNorm();
    double scale_factor = S.sum() / var_smple;

    // Compute translation
    Eigen::RowVector3d t = detected_center - scale_factor * smpl_center * R.transpose();
        
    vertices = SINGLE_SMPL::get()->getVertex();
    vertices = (vertices * R.transpose()) * scale_factor;
    vertices.rowwise() += t;
    write_points3D(outputPath, vertices, head_bbox);
    
    OH_LOG_INFO(LOG_APP, "Printing SMPL parameters");
    for (int64_t i = 0; i < VERTEX_NUM; i++) {
        OH_LOG_INFO(LOG_APP, "%{public}f %{public}f %{public}f", vertices(i, 0), vertices(i, 1), vertices(i, 2));
    }
    OH_LOG_INFO(LOG_APP, "Printing Finished");
    
	SINGLE_SMPL::destroy();
    auto end = clk::now();
    auto duration = std::chrono::duration_cast<ms>(end - begin);
    OH_LOG_INFO(LOG_APP, "Time duration to estimate SMPL: %{public}f s", (double)duration.count() / 1000);
}