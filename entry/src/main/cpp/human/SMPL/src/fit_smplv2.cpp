#include "fit_smpl.h"

#define SINGLE_SMPL smpl::Singleton<smpl::SMPL>

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

// Function to check if a point is valid (non-zero coordinates)
bool isValidPoint(const Eigen::Vector3d& point) {
    return !(point[0] == 0.0 && point[1] == 0.0 && point[2] == 0.0);
}

// Function to read keypoints from a PLY file with validity check
std::pair<Eigen::MatrixXd, std::vector<bool>> read_keypoints_from_ply_with_validity(const std::string& filename) {
    // Open the PLY file
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        return {Eigen::MatrixXd(), {}};
    }

    std::string line;
    int num_vertices = 0;
    bool header_ended = false;

    // Read the header to determine the number of vertices
    while (std::getline(infile, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line == "end_header") {
            header_ended = true;
            break;
        }

        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element") {
            std::string element_type;
            iss >> element_type;
            if (element_type == "vertex") {
                iss >> num_vertices;
            }
        }
    }

    if (!header_ended || num_vertices <= 0) {
        std::cerr << "Invalid PLY file format." << std::endl;
        return {Eigen::MatrixXd(), {}};
    }

    Eigen::MatrixXd keypoints(num_vertices, 3);
    std::vector<bool> validity(num_vertices, false);

    for (int i = 0; i < num_vertices; ++i) {
        if (!std::getline(infile, line)) {
            std::cerr << "Unexpected end of file when reading vertex data." << std::endl;
            return {Eigen::MatrixXd(), {}};
        }

        std::istringstream iss(line);
        double x, y, z;
        int r, g, b;
        if (!(iss >> x >> y >> z >> r >> g >> b)) {
            std::cerr << "Error parsing vertex data at vertex index " << i << std::endl;
            return {Eigen::MatrixXd(), {}};
        }

        keypoints(i, 0) = x;
        keypoints(i, 1) = y;
        keypoints(i, 2) = z;
        
        // Point is valid if not all coordinates are 0
        validity[i] = isValidPoint(Eigen::Vector3d(x, y, z));
    }

    infile.close();
    return {keypoints, validity};
}

// Function to calculate head bounding box
bool calculate_head_bbox(const Eigen::MatrixXd& keypoints, const std::vector<bool>& validity, std::vector<float>& head_bbox) {
    // Check if first 5 keypoints are available (nose, eyes, ears)
    if (keypoints.rows() < 5) {
        std::cerr << "Not enough keypoints for head bounding box calculation" << std::endl;
        return false;
    }
    
    // Check if both ears are valid
    if (!validity[3] || !validity[4]) {
        std::cerr << "At least one ear keypoint is missing" << std::endl;
        return false;
    }
    
    // Calculate the center point as the mean of the two ears
    Eigen::Vector3d left_ear(keypoints(3, 0), keypoints(3, 1), keypoints(3, 2));
    Eigen::Vector3d right_ear(keypoints(4, 0), keypoints(4, 1), keypoints(4, 2));
    Eigen::Vector3d center_ears = (left_ear + right_ear) / 2.0;
    Eigen::Vector3d left_shoulder(keypoints(5, 0), keypoints(5, 1), keypoints(5, 2));
    Eigen::Vector3d right_shoulder(keypoints(6, 0), keypoints(6, 1), keypoints(6, 2));
    Eigen::Vector3d center_shoulders = (left_shoulder + right_shoulder) / 2.0;
    
    // Calculate the distance between the ears
    double distance = (center_ears - center_shoulders).norm();
    
    // The bounding box size is 1.1 times the ear distance
    double bbox_size = distance;
    double half_size = bbox_size / 2.0;
    
    // Calculate the bounding box
    head_bbox = {
        static_cast<float>(center_ears[0] - half_size), // x_min
        static_cast<float>(center_ears[0] + half_size), // x_max
        static_cast<float>(center_ears[1] - half_size), // y_min
        static_cast<float>(center_ears[1] + half_size), // y_max
        static_cast<float>(center_ears[2] - half_size), // z_min
        static_cast<float>(center_ears[2] + half_size)  // z_max
    };
    
    return true;
}

// Function to calculate body bounding box from valid keypoints
std::vector<float> calculate_body_bbox(const Eigen::MatrixXd& keypoints, const std::vector<bool>& validity) {
    double x_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();
    double y_min = std::numeric_limits<double>::max();
    double y_max = std::numeric_limits<double>::lowest();
    double z_min = std::numeric_limits<double>::max();
    double z_max = std::numeric_limits<double>::lowest();
    
    bool found_valid = false;
    
    // Find min/max values from valid keypoints
    for (int i = 0; i < keypoints.rows() && i < validity.size(); i++) {
        if (validity[i]) {
            x_min = std::min(x_min, keypoints(i, 0));
            x_max = std::max(x_max, keypoints(i, 0));
            y_min = std::min(y_min, keypoints(i, 1));
            y_max = std::max(y_max, keypoints(i, 1));
            z_min = std::min(z_min, keypoints(i, 2));
            z_max = std::max(z_max, keypoints(i, 2));
            found_valid = true;
        }
    }
    
    if (!found_valid) {
        return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    // Calculate center and extent
    Eigen::Vector3d center((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0);
    Eigen::Vector3d extent((x_max - x_min) / 2.0, (y_max - y_min) / 2.0, (z_max - z_min) / 2.0);
    
    // Enlarge by 1.25 times
    extent *= 1.25;
    
    // Return enlarged bounding box
    return {
        static_cast<float>(center[0] - extent[0]),
        static_cast<float>(center[0] + extent[0]),
        static_cast<float>(center[1] - extent[1]),
        static_cast<float>(center[1] + extent[1]),
        static_cast<float>(center[2] - extent[2]),
        static_cast<float>(center[2] + extent[2])
    };
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

const std::vector<std::pair<int, int>> procrustes_indices = {
    {5, 16},  // left shoulder
    {6, 17},  // right shoulder
    {7, 18},  // left elbow
    {8, 19},  // right elbow
    {11, 1},  // left hip
    {12, 2},  // right hip
    {13, 4},  // left knee
    {14, 5},  // right knee
    {0, 332},  // nose (SMPL vertex 332)
    {1, 2800}, // left eye (SMPL vertex 2800)
    {2, 6260}  // right eye (SMPL vertex 6260)
};

// Modified function to perform Procrustes analysis with validity check
std::pair<Eigen::Matrix3d, Eigen::RowVector3d> perform_procrustes_analysis(
    const Eigen::MatrixXd& smpl_joints,
    const Eigen::MatrixXd& smpl_vertices,  // New parameter
    const Eigen::MatrixXd& detected_keypoints, 
    const std::vector<bool>& validity) {
    
    // Count valid matching pairs
    int valid_count = 0;
    for (const auto& pair : procrustes_indices) {
        int detected_idx = pair.first;
        if (detected_idx < validity.size() && validity[detected_idx]) {
            valid_count++;
        }
    }
    
    if (valid_count < 4) {
        std::cerr << "Not enough valid keypoints for Procrustes analysis" << std::endl;
        return {Eigen::Matrix3d::Identity(), Eigen::RowVector3d::Zero()};
    }
    
    // Create matrices for valid keypoints
    Eigen::MatrixXd smpl_keypoints(valid_count, 3);
    Eigen::MatrixXd detected_keypoints_valid(valid_count, 3);
    
    int valid_idx = 0;
    for (const auto& pair : procrustes_indices) {
        int detected_idx = pair.first;
        int smpl_idx = pair.second;
        
        if (detected_idx < validity.size() && validity[detected_idx]) {
            // Get SMPL point from either joints or vertices
            if (smpl_idx < smpl_joints.rows()) {
                smpl_keypoints.row(valid_idx) = smpl_joints.row(smpl_idx);
            } else {
                smpl_keypoints.row(valid_idx) = smpl_vertices.row(smpl_idx);
            }
            
            detected_keypoints_valid.row(valid_idx) = detected_keypoints.row(detected_idx);
            valid_idx++;
        }
    }
    
    // Center the data
    Eigen::RowVector3d smpl_center = smpl_keypoints.colwise().mean();
    Eigen::RowVector3d detected_center = detected_keypoints_valid.colwise().mean();
    
    Eigen::MatrixXd smpl_centered = smpl_keypoints.rowwise() - smpl_center;
    Eigen::MatrixXd detected_centered = detected_keypoints_valid.rowwise() - detected_center;
    
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
        V.col(2) *= -1;
        R = V * U.transpose();
    }
    
    // Compute scale
    double var_smpl = smpl_centered.squaredNorm();
    double scale_factor = S.sum() / var_smpl;
    
    // Compute translation
    Eigen::RowVector3d t = detected_center - scale_factor * smpl_center * R.transpose();
    
    // Scale the rotation matrix
    R *= scale_factor;
    
    return {R, t};
}

// Modified function to write vertices with filtering
void write_points3D(std::string& outputPath, 
                   Eigen::MatrixXd& vertices,
                   const std::vector<float>& body_bbox) {
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Failed to open file %{public}s for writing", outputPath.c_str());
        return;
    }

    outFile << std::fixed << std::setprecision(6);
    
    // Extract body bbox
    float body_x_min = body_bbox[0];
    float body_x_max = body_bbox[1];
    float body_y_min = body_bbox[2];
    float body_y_max = body_bbox[3];
    float body_z_min = body_bbox[4];
    float body_z_max = body_bbox[5];
    
    for (int64_t i = 0; i < VERTEX_NUM; ++i) {
        double x = vertices(i, 0);
        double y = vertices(i, 1);
        double z = vertices(i, 2);
        
        // Check if vertex is within the body bounding box
        bool within_bbox = (x >= body_x_min && x <= body_x_max && 
                           y >= body_y_min && y <= body_y_max && 
                           z >= body_z_min && z <= body_z_max);
        
        if (within_bbox) {
            // Write vertex to file
            outFile << i << " " << x << " " << y << " " << z << " 125 125 125 1.000000\n";
            LOGI("%{public}f %{public}f %{public}f", x, y, z);
        }
    }
    
    outFile.close();
    OH_LOG_INFO(LOG_APP, "Successfully wrote vertices to %{public}s", outputPath.c_str());
}

// Enhanced main function
void FitSMPL(std::string inputPath, std::vector<float>& head_bbox, std::vector<float>& body_bbox) {
    auto begin = clk::now();
    
    std::string outputPath = inputPath + "/smpl.txt";
    
    // Read keypoints with validity check
    auto [detected_keypoints_all, validity] = read_keypoints_from_ply_with_validity(inputPath + "/skeleton.ply");
    
    if (detected_keypoints_all.rows() == 0) {
        OH_LOG_ERROR(LOG_APP, "Failed to read keypoints from skeleton file");
        return;
    }
    
    // Step 1: Calculate head bounding box from first 5 keypoints
    if (!calculate_head_bbox(detected_keypoints_all, validity, head_bbox)) {
        OH_LOG_ERROR(LOG_APP, "Failed to calculate head bounding box, terminating");
        return;
    }
    
    // Step 2: Check if there are at least 4 valid points in the last 12 keypoints
    int valid_limb_count = 0;
    for (int i = 5; i < 17 && i < detected_keypoints_all.rows(); i++) {
        if (validity[i]) valid_limb_count++;
    }
    
    if (valid_limb_count < 4) {
        OH_LOG_ERROR(LOG_APP, "Not enough valid limb keypoints (found %d, need at least 4)", valid_limb_count);
        return;
    }
    
    // Initialize SMPL matrices
    Eigen::MatrixXd beta(1, SHAPE_BASIS_DIM);
    Eigen::MatrixXd theta(JOINT_NUM, 3);
    beta.setZero();
    theta.setZero();
    
    // Step 3: Get thetas considering invalid points
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

    // Initialize SMPL model
    SINGLE_SMPL::get()->init();
    SINGLE_SMPL::get()->launch(beta, theta);
    
    Eigen::MatrixXd smpl_joints = SINGLE_SMPL::get()->getPosedJoint(); 
    Eigen::MatrixXd vertices = SINGLE_SMPL::get()->getVertex();

    // Step 4: Perform Procrustes analysis using valid keypoints
    auto [R, t] = perform_procrustes_analysis(smpl_joints, vertices, detected_keypoints_all, validity);

    vertices = vertices * R.transpose();
    vertices.rowwise() += t;
    
    // Step 4: Calculate body bounding box
    body_bbox = calculate_body_bbox(detected_keypoints_all, validity);
    
    // Step 5: Write filtered vertices to output file
    write_points3D(outputPath, vertices, body_bbox);
    
    SINGLE_SMPL::destroy();
    auto end = clk::now();
    auto duration = std::chrono::duration_cast<ms>(end - begin);
    OH_LOG_INFO(LOG_APP, "Time duration to estimate SMPL: %{public}f s", (double)duration.count() / 1000);
}