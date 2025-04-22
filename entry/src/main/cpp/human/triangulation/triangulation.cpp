#include "triangulation.h"

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

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
    }
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
        ss >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> image_name;
        
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
    std::vector<cv::Point3d>& keypoints3d)
{
    int num_keypoints = 17;
    keypoints3d.resize(num_keypoints);

    // For each keypoint index:
    for (int k = 0; k < num_keypoints; ++k) {
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
    }
}

double calculateReprojectionError(
    const CameraIntrinsics& intrinsics,
    const std::map<std::string, CameraExtrinsics>& extrinsics_map,
    const std::map<std::string, FrameData>& frames,
    const std::vector<cv::Point3d>& keypoints3d)
{
    double total_error = 0.0;
    int count = 0;

    // For each frame
    for (const auto& frame_pair : frames) {
        const std::string& image_name = frame_pair.first;
        const FrameData& frame = frame_pair.second;

        auto extr_it = extrinsics_map.find(image_name);
        if (extr_it == extrinsics_map.end())
            continue;

        const CameraExtrinsics& extr = extr_it->second;

        // Compute projection matrix
        cv::Mat K = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.cx,
                                                0, intrinsics.fy, intrinsics.cy,
                                                0, 0, 1);
        cv::Mat Rt;
        cv::hconcat(extr.R, extr.t, Rt);
        cv::Mat P = K * Rt; // 3x4

        // For each keypoint
        for (int k = 0; k < static_cast<int>(keypoints3d.size()); ++k) {
            cv::Point3d X = keypoints3d[k];

            // Check if X is valid
            if (X == cv::Point3d(0, 0, 0))
                continue; // Invalid point

            // Get the observed 2D keypoint
            if (frame.keypoints.size() > k) {
                const KeyPoint2D& kp = frame.keypoints[k];
                if (kp.confidence < 0.9)
                    continue;

                // Project 3D point to 2D   
                cv::Mat X_homogeneous = (cv::Mat_<double>(4, 1) << X.x, X.y, X.z, 1.0);
                cv::Mat x_homogeneous = P * X_homogeneous; // 3x1
                double x_proj = x_homogeneous.at<double>(0, 0) / x_homogeneous.at<double>(2, 0);
                double y_proj = x_homogeneous.at<double>(1, 0) / x_homogeneous.at<double>(2, 0);
                // Compute the error
                double err = std::sqrt((x_proj - kp.x) * (x_proj - kp.x) +
                                       (y_proj - kp.y) * (y_proj - kp.y));
                total_error += err;
                count++;
            }
        }
    }
    
    if (count > 0) {
        double avg_error = total_error / count;
        OH_LOG_Print(LOG_APP, LOG_INFO,  LOG_DOMAIN, LOG_TAG, "Reprojection Error is %{public}f.", avg_error);
        
        return avg_error;
    } else {
        OH_LOG_Print(LOG_APP, LOG_ERROR,  LOG_DOMAIN, LOG_TAG, "No valid reprojections.");
        return -1;
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

double ProcessTriangulation(std::string inputPath) {
    auto begin = clk::now();
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    std::map<std::string, FrameData> frames;
    CameraIntrinsics intrinsics;
    std::map<std::string, CameraExtrinsics> extrinsics_map;

    // Read keypoints
    std::string keypointsFile = inputPath + "/keypoints.txt";
    readKeypoints(keypointsFile, frames);

    // Read camera intrinsics
    std::string camerasFile = inputPath + "/cameras.txt";
    if (!readCameraIntrinsics(camerasFile, intrinsics)) {
        OH_LOG_ERROR(LOG_APP, "Failed to read camera intrinsics");
        return -1.0; // Indicate error
    }

    // Read camera extrinsics
    std::string imagesFile = inputPath + "/images.txt";
    readCameraExtrinsics(imagesFile, extrinsics_map);

    // Perform triangulation
    std::vector<cv::Point3d> keypoints3d;
    multiViewTriangulation(intrinsics, extrinsics_map, frames, keypoints3d);

    // Save results
    std::string plyPath = inputPath + "/skeleton.ply";
    savePointsAndLinesToPLY(keypoints3d, plyPath);

    // Calculate reprojection error
    double reprojectionError = calculateReprojectionError(intrinsics, extrinsics_map, frames, keypoints3d);

    // Log timing
    gettimeofday(&t2, NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000;
    timeuse *= 1000;
    OH_LOG_INFO(LOG_APP, "Running time: %{public}f ms", timeuse);

    auto end = clk::now();
    auto duration = std::chrono::duration_cast<ms>(end - begin);
    OH_LOG_INFO(LOG_APP, "Total triangulation duration: %{public}f s", duration.count() / 1000.0);

    return reprojectionError;
}