#include "human/info.h"
#include "triangulation.h"

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;
// Enum to identify the input type
enum class InputType {
    KEYPOINTS,
    LANDMARKS
};

void readKeypoints(const std::string& filename, std::map<std::string, FrameData>& frames) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Cannot open keypoints file: %{public}s", filename.c_str());
        return;
    }
    
    std::string line;
    std::string current_image;
    
    while (std::getline(infile, line)) {
        if (line.empty())
            continue;
            
        if (line.find(".jpg") != std::string::npos) {
            current_image = line;
            FrameData frame;
            frame.imageName = current_image;
            frames[current_image] = frame;
            
            // Skip the next line (bounding box line)
            if (std::getline(infile, line)) {
                // Line is read and discarded
            }
        } else {
            if (current_image.empty())
                continue;
                
            std::stringstream ss(line);
            KeyPoint2D kp;
            ss >> kp.x >> kp.y >> kp.confidence;
            frames[current_image].keypoints.push_back(kp);
        }
    }
    
    infile.close();
}

bool readCameraIntrinsics(const std::string& filename, CameraIntrinsics& intrinsics) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Cannot open camera intrinsics file: %{public}s", filename.c_str());
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        break;
    }
    while (line[0] == '#') {
        if (!std::getline(infile, line)) break;
    }
    std::stringstream ss(line);
    int camera_id;
    std::string model;
    // We require that width is the number of columns and height is the number of rows.
    ss >> camera_id >> model >> intrinsics.width >> intrinsics.height;
    ss >> intrinsics.fx >> intrinsics.fy >> intrinsics.cx >> intrinsics.cy;
    if(intrinsics.height < intrinsics.width){
        int temp = intrinsics.height;
        intrinsics.height = intrinsics.width;
        intrinsics.width = temp;
        float temp2 = intrinsics.cy;
        intrinsics.cy = intrinsics.cx;
        intrinsics.cx = temp2;
        temp2 = intrinsics.fy;
        intrinsics.fy = intrinsics.fx;
        intrinsics.fx = temp2;
    }
    if(intrinsics.width != 480){
        float ratio = intrinsics.width / 480.;
        assert(abs(ratio - intrinsics.height / 640.) < 0.1);
        intrinsics.width /= ratio;
        intrinsics.height /= ratio;
        intrinsics.cx /= ratio;
        intrinsics.cy /= ratio;
        intrinsics.fx /= ratio;
        intrinsics.fy /= ratio;
    }
    LOGI("Processed intr: %{public}f %{public}f %{public}f %{public}f",intrinsics.cx,intrinsics.cy,intrinsics.fx,intrinsics.fy);
    infile.close();
    return true;
}

void readCameraExtrinsics(const std::string& filename, std::map<std::string, CameraExtrinsics>& extrinsics_map) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Cannot open extrinsics file: %{public}s", filename.c_str());
        return;
    }
    std::string line;
//     while (std::getline(infile, line)) {
//         if (line.empty() || line[0] == '#') continue;
//         break;
//     }
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        int image_id, camera_id;
        double qw, qx, qy, qz, tx, ty, tz;
        std::string image_name;
        if(!( ss >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> image_name)){
            continue;
        }
            
        cv::Mat R = cv::Mat::zeros(3, 3, CV_64F);
        double q0 = qw, q1 = qx, q2 = qy, q3 = qz;
        R.at<double>(0,0) = 1 - 2*(q2*q2 + q3*q3);
        R.at<double>(0,1) = 2*(q1*q2 - q0*q3);
        R.at<double>(0,2) = 2*(q1*q3 + q0*q2);
        R.at<double>(1,0) = 2*(q1*q2 + q0*q3);
        R.at<double>(1,1) = 1 - 2*(q1*q1 + q3*q3);
        R.at<double>(1,2) = 2*(q2*q3 - q0*q1);
        R.at<double>(2,0) = 2*(q1*q3 - q0*q2);
        R.at<double>(2,1) = 2*(q2*q3 + q0*q1);
        R.at<double>(2,2) = 1 - 2*(q1*q1 + q2*q2);

        cv::Mat t = (cv::Mat_<double>(3,1) << tx, ty, tz);
//         LOGI("Extrinsics for %{public}s is: %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f %{public}f",
//                                                 image_name.c_str(), qw, qx, qy, qz, tx, ty, tz);
        extrinsics_map[image_name] = {R, t};
    }
//     LOGI("Extrinsics size: %{public}zu", extrinsics_map.size());
    infile.close();
}

void multiViewTriangulation(
    const CameraIntrinsics& intrinsics,
    const std::map<std::string, CameraExtrinsics>& extrinsics_map,
    const std::map<std::string, FrameData>& frames,
    std::vector<cv::Point3d>& keypoints3d,
    InputType inputType)
{
    int num_keypoints = (inputType == InputType::LANDMARKS) ? 16 : 17;
    keypoints3d.resize(num_keypoints);
    
    // Determine if we should use confidence filtering
    bool useConfidenceFiltering = (inputType == InputType::KEYPOINTS);

    // For each keypoint index:
    for (int k = 0; k < num_keypoints; ++k) {
        if (useConfidenceFiltering) {
            // First pass: collect all confidence values for this keypoint.
            std::vector<double> all_confidences;
            for (const auto& frame_pair : frames) {
                const FrameData& frame = frame_pair.second;
                if (frame.keypoints.size() > k) {
                    if(frame.keypoints[k].confidence > 0.7){
                        all_confidences.push_back(frame.keypoints[k].confidence);
                    }
                }
            }

            double confidence_threshold = 0.8;  // default fallback
            if (all_confidences.size() >= 2) {
                // Sort confidences in descending order.
                std::sort(all_confidences.begin(), all_confidences.end(), std::greater<double>());
                // For example, choose the value at the 25th percentile index.
                size_t idx = all_confidences.size() / 4;
                if (idx >= all_confidences.size()) idx = all_confidences.size() - 1;
                confidence_threshold = all_confidences[idx];
            }
            else {
                OH_LOG_WARN(LOG_APP, "Not enough observations for keypoint %{public}d", k);
                keypoints3d[k] = cv::Point3d(0, 0, 0); // Default value
                continue;
            }
                
            LOGI("Conf thresh for kpt %{public}d is %{public}f", k, confidence_threshold);
            std::vector<cv::Mat> As;
            std::vector<double> confidences;

            // Second pass: process only high-confidence observations.
            for (const auto& frame_pair : frames) {
                const std::string& image_name = frame_pair.first;
                const FrameData& frame = frame_pair.second;

                if (frame.keypoints.size() > k) {
                    const KeyPoint2D& kp = frame.keypoints[k];

                    if (kp.confidence < confidence_threshold)
                        continue;

                    auto extr_it = extrinsics_map.find(image_name);
                    if (extr_it == extrinsics_map.end())
                        continue;

                    const CameraExtrinsics& extr = extr_it->second;

                    cv::Mat K = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.cx,
                                                            0, intrinsics.fy, intrinsics.cy,
                                                            0, 0, 1);
                    cv::Mat Rt;
                    cv::hconcat(extr.R, extr.t, Rt);
                    cv::Mat P = K * Rt;

                    cv::Mat A(2, 4, CV_64F);
                    double u = kp.x;
                    double v = kp.y;
                    A.row(0) = u * P.row(2) - P.row(0);
                    A.row(1) = v * P.row(2) - P.row(1);

                    As.push_back(A);
                    confidences.push_back(kp.confidence);
                }
            }
            LOGI("Using %{public}zu points for the triangulation of point %{public}d", As.size(), k);
            if (As.size() >= 2) {
                cv::Mat A_total(2 * As.size(), 4, CV_64F);
                for (size_t i = 0; i < As.size(); ++i) {
                    As[i].copyTo(A_total.rowRange(i * 2, i * 2 + 2));
                }

                cv::Mat u, s, vt;
                cv::SVD::compute(A_total, s, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
                cv::Mat X_homogeneous = vt.row(3).t();
                X_homogeneous = X_homogeneous / X_homogeneous.at<double>(3, 0);
                cv::Point3d point3d(
                    X_homogeneous.at<double>(0, 0),
                    X_homogeneous.at<double>(1, 0),
                    X_homogeneous.at<double>(2, 0)
                );
                keypoints3d[k] = point3d;
            }
            else {
                OH_LOG_WARN(LOG_APP, "Not enough observations for keypoint %{public}d", k);
                keypoints3d[k] = cv::Point3d(0, 0, 0); // Default value
            }
        } else {
            // For landmarks, no confidence filtering
            std::vector<cv::Mat> As;

            for (const auto& frame_pair : frames) {
                const std::string& image_name = frame_pair.first;
                const FrameData& frame = frame_pair.second;

                if (frame.keypoints.size() > k) {
                    const KeyPoint2D& kp = frame.keypoints[k];

                    auto extr_it = extrinsics_map.find(image_name);
                    if (extr_it == extrinsics_map.end())
                        continue;

                    const CameraExtrinsics& extr = extr_it->second;

                    cv::Mat K = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.cx,
                                                            0, intrinsics.fy, intrinsics.cy,
                                                            0, 0, 1);
                    cv::Mat Rt;
                    cv::hconcat(extr.R, extr.t, Rt);
                    cv::Mat P = K * Rt;

                    cv::Mat A(2, 4, CV_64F);
                    double u = kp.x;
                    double v = kp.y;
                    A.row(0) = u * P.row(2) - P.row(0);
                    A.row(1) = v * P.row(2) - P.row(1);

                    As.push_back(A);
                }
            }
            
            LOGI("Using %{public}zu points for the triangulation of landmark %{public}d", As.size(), k);
            if (As.size() >= 2) {
                cv::Mat A_total(2 * As.size(), 4, CV_64F);
                for (size_t i = 0; i < As.size(); ++i) {
                    As[i].copyTo(A_total.rowRange(i * 2, i * 2 + 2));
                }

                cv::Mat u, s, vt;
                cv::SVD::compute(A_total, s, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
                cv::Mat X_homogeneous = vt.row(3).t();
                X_homogeneous = X_homogeneous / X_homogeneous.at<double>(3, 0);
                cv::Point3d point3d(
                    X_homogeneous.at<double>(0, 0),
                    X_homogeneous.at<double>(1, 0),
                    X_homogeneous.at<double>(2, 0)
                );
                keypoints3d[k] = point3d;
            }
            else {
                OH_LOG_WARN(LOG_APP, "Not enough observations for landmark %{public}d", k);
                keypoints3d[k] = cv::Point3d(0, 0, 0); // Default value
            }
        }
    }
}

// Function to save 3D points and lines to a .ply file
void savePointsAndLinesToPLY(const std::vector<cv::Point3d>& points,
                             const std::string& filename) {
    // Define the edges (lines) connecting points as per the given indices
    std::vector<std::pair<int, int>> edges = {
        {0,1}, {0,2}, {1,2}, {1,3}, {2,4}, {3,5}, {4,6},
        {5,6}, {5,7}, {6,8}, {7,9}, {8,10}, {5,11}, {6,12},
        {11,12}, {11,13}, {12,14}, {13,15}, {14,16}
    };

    // Colors for points and lines can be defined here
    // For points, you can set a default color or generate colors
    // For this example, assign all points a red color
    const uint8_t point_red = 255;
    const uint8_t point_green = 0;
    const uint8_t point_blue = 0;

    // For lines, colors are controllable, initially set to blue
    const uint8_t line_red = 0;
    const uint8_t line_green = 0;
    const uint8_t line_blue = 255;

    // Open the file for writing
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Cannot open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write PLY header
    ofs << "ply\n";
    ofs << "format ascii 1.0\n";

    // Number of vertices
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";

    ofs << "end_header\n";

    // Write vertex data with colors
    for (const auto& pt : points) {
        OH_LOG_INFO(LOG_APP, "skeleton is: %{public}f %{public}f %{public}f", pt.x, pt.y, pt.z);
        
        ofs << pt.x << " " << pt.y << " " << pt.z << " "
            << static_cast<int>(point_red) << " "
            << static_cast<int>(point_green) << " "
            << static_cast<int>(point_blue) << "\n";
    }

    ofs.close();
        OH_LOG_INFO(LOG_APP, "Writing skeleton to %{public}s", filename.c_str());
}

bool upward_vec(const std::vector<cv::Point3d>& keypoints3d, cv::Vec3d& upVec){
    cv::Point3d left_hip = keypoints3d[11];
    cv::Point3d right_hip = keypoints3d[12];
    cv::Point3d left_shoulder = keypoints3d[5];
    cv::Point3d right_shoulder = keypoints3d[6];
    cv::Vec3d leftUpVec(left_shoulder.x - left_hip.x,
                        left_shoulder.y - left_hip.y,
                        left_shoulder.z - left_hip.z);
    cv::Vec3d rightUpVec(right_shoulder.x - right_hip.x,
                        right_shoulder.y - right_hip.y,
                        right_shoulder.z - right_hip.z);
    upVec[0] = (leftUpVec[0] + rightUpVec[0]) / 2.0;
    upVec[1] = (leftUpVec[1] + rightUpVec[1]) / 2.0,
    upVec[2] = (leftUpVec[2] + rightUpVec[2]) / 2.0;
    double upVec_norm = cv::norm(upVec);
    
    if (upVec_norm > 0) {
        upVec /= upVec_norm;
    } else {
        OH_LOG_WARN(LOG_APP, "Face direction vector has zero norm");
        return false;
    }
    double dot_product = upVec.dot(cv::Vec3d(0, 1, 0));
    return dot_product > 0;
}

// New function to estimate face direction angles for each camera
void estimateFaceDirections(
    const std::vector<cv::Point3d>& keypoints3d,
    const CameraIntrinsics& intrinsics,
    const std::map<std::string, CameraExtrinsics>& extrinsics_map,
    std::map<std::string, double>& faceAngles)
{
    // Check if we have the required keypoints (nose=0, left_ear=3, right_ear=4)
    if (keypoints3d.size() < 5) {
        OH_LOG_ERROR(LOG_APP, "Not enough keypoints for face direction estimation");
        return;
    }
    
    // Get the 3D positions of nose and ears
    cv::Point3d nose = keypoints3d[0];
    cv::Point3d left_ear = keypoints3d[3];
    cv::Point3d right_ear = keypoints3d[4];
    cv::Vec3d upVec;
    bool flip_axis = upward_vec(keypoints3d, upVec);
    
    // Calculate the middle point between the two ears
    cv::Point3d mid_ears((left_ear.x + right_ear.x) / 2.0,
                         (left_ear.y + right_ear.y) / 2.0,
                         (left_ear.z + right_ear.z) / 2.0);
    
    // Calculate the face forward direction vector (from mid_ears to nose)
    cv::Vec3d face_direction(nose.x - mid_ears.x,
                            nose.y - mid_ears.y,
                            nose.z - mid_ears.z);
    
    // Normalize the face direction vector
    double face_dir_norm = cv::norm(face_direction);
    if (face_dir_norm > 0) {
        face_direction /= face_dir_norm;
    } else {
        OH_LOG_WARN(LOG_APP, "Face direction vector has zero norm");
        return;
    }
    
    // For each camera, calculate the angle
    for (const auto& extr_pair : extrinsics_map) {
        const std::string& camera_name = extr_pair.first;
        const CameraExtrinsics& extr = extr_pair.second;
        
        // Get camera position in world coordinates
        // Camera position = -R^T * t
        cv::Mat R_t = extr.R.t();
        cv::Mat camera_pos = -R_t * extr.t;
        
        // Calculate the camera's forward direction (viewing direction)
        // This is the third row of R (or third column of R^T)
        cv::Vec3d camera_forward(extr.R.at<double>(0, 2),
                                extr.R.at<double>(1, 2),
                                extr.R.at<double>(2, 2));
        
        // Calculate the vector from camera origin to center of imaging plane
        // For a simple approximation, we can use the camera's forward direction
        // normalized and scaled by focal length, but since we want a unit vector
        // for angle calculation, we just use the normalized forward direction
        double cam_dir_norm = cv::norm(camera_forward);
        if (cam_dir_norm > 0) {
            camera_forward /= cam_dir_norm;
        }
        
        // Calculate the angle between face direction and camera direction
        double dot_product = face_direction.dot(camera_forward);
        
        // Clamp dot product to [-1, 1] to avoid numerical errors
        dot_product = std::max(-1.0, std::min(1.0, dot_product));
        
        // Calculate angle in radians and convert to degrees
        double angle_rad = std::acos(dot_product);
        double angle_deg = angle_rad * 180.0 / CV_PI;
        if(flip_axis){
            angle_deg = 180 - angle_deg;
        }
        faceAngles[camera_name] = angle_deg;

        OH_LOG_INFO(LOG_APP, "Face angle for camera %{public}s: %{public}f degrees", 
                    camera_name.c_str(), angle_deg);
    }
}

std::map<std::string, double> ProcessTriangulation(std::string inputPath) {
    auto begin = clk::now();
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    std::map<std::string, FrameData> frames;
    CameraIntrinsics intrinsics;
    std::map<std::string, CameraExtrinsics> extrinsics_map;
    InputType inputType;

    // Check which input file exists to determine the input type
    std::string keypointsFile = inputPath + "/keypoints.txt";
    std::string landmarksFile = inputPath + "/landmarks.txt";
    
    if (std::ifstream(landmarksFile).good()) {
        // Use landmarks file
        inputType = InputType::LANDMARKS;
        readKeypoints(landmarksFile, frames);
        OH_LOG_INFO(LOG_APP, "Using landmarks.txt as input");
    } else {
        // Use keypoints file (default)
        inputType = InputType::KEYPOINTS;
        readKeypoints(keypointsFile, frames);
        OH_LOG_INFO(LOG_APP, "Using keypoints.txt as input");
    }

    // Read camera intrinsics
    std::string camerasFile = inputPath + "/cameras.txt";
    if (!readCameraIntrinsics(camerasFile, intrinsics)) {
        OH_LOG_ERROR(LOG_APP, "Failed to read camera intrinsics");
    }

    // Read camera extrinsics
    std::string imagesFile = inputPath + "/images.txt";
    readCameraExtrinsics(imagesFile, extrinsics_map);

    // Perform triangulation
    std::vector<cv::Point3d> keypoints3d;
    multiViewTriangulation(intrinsics, extrinsics_map, frames, keypoints3d, inputType);

    // Estimate face directions only for keypoints
    std::map<std::string, double> faceAngles;
    if (inputType == InputType::KEYPOINTS) {
        estimateFaceDirections(keypoints3d, intrinsics, extrinsics_map, faceAngles);
        
        // Save face angles to a file
        std::string anglesPath = inputPath + "/face_angles.txt";
        std::ofstream anglesFile(anglesPath);
        if (anglesFile.is_open()) {
            for (const auto& angle_pair : faceAngles) {
                anglesFile << angle_pair.first << " " << angle_pair.second << std::endl;
            }
            anglesFile.close();
            OH_LOG_INFO(LOG_APP, "Face angles saved to %{public}s", anglesPath.c_str());
        }
    }

    // Save results
    std::string plyPath = (inputType == InputType::LANDMARKS) ? 
                          inputPath + "/landmarks.ply" : 
                          inputPath + "/skeleton.ply";
    savePointsAndLinesToPLY(keypoints3d, plyPath);
    
    gettimeofday(&t2, NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
    OH_LOG_INFO(LOG_APP, "Triangulation completed in %{public}f seconds", timeuse);
    
    return faceAngles;
}
