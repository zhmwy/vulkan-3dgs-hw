#include "face.h"

using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

std::vector<cv::Point> projectHeadCenter(DataLoader::InputInitStruct& data)
{
    // Calculate the center of the 3D bounding box
    float center_x = (data.head_bbox[0] + data.head_bbox[1]) / 2.0f;
    float center_y = (data.head_bbox[2] + data.head_bbox[3]) / 2.0f;
    float center_z = (data.head_bbox[4] + data.head_bbox[5]) / 2.0f;
    
    // Create a 3D point for the center in homogeneous coordinates
    Eigen::Vector4f point_3d(center_x, center_y, center_z, 1.0f);
    
    // Vector to store results
    std::vector<cv::Point> projected_points;
    projected_points.reserve(data.extr_tmp.size());
    
    // For each camera pose
    for (size_t i = 0; i < data.extr_tmp.size(); ++i) {
        // Get the current extrinsics
        const DataLoader::InputInitStruct::DataExtr extr = data.extr_tmp[i];
        
        // Get the corresponding intrinsics (either shared or per-camera)
        const std::vector<float> intr = (data.intr_tmp.size() == 1) ? data.intr_tmp[0] : data.intr_tmp[i];
        if (intr.size() != 4) {
            throw std::invalid_argument("Intrinsics must have 4 elements (fx, fy, cx, cy)");
        }
        
        // Extract quaternion components and translation vector
        float qw = extr.values[0];
        float qx = extr.values[1];
        float qy = extr.values[2];
        float qz = extr.values[3];
        float tx = extr.values[4];
        float ty = extr.values[5];
        float tz = extr.values[6];
        
        // Convert quaternion to rotation matrix
        Eigen::Quaternionf q(qw, qx, qy, qz);
        Eigen::Matrix3f R = q.normalized().toRotationMatrix();
        
        // Create transformation matrix (extrinsic matrix)
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = Eigen::Vector3f(tx, ty, tz);
        
        // Transform 3D point from world to camera coordinates
        Eigen::Vector4f point_camera = T * point_3d;
        
        // Check if the point is in front of the camera
        if (point_camera[2] <= 0) {
            // Point is behind the camera, store an invalid point
            projected_points.push_back(cv::Point(-1, -1));
            continue;
        }
        
        // Project 3D point to 2D image coordinates
        float fx = intr[0];
        float fy = intr[1];
        float cx = intr[2];
        float cy = intr[3];
        
        float x_2d = (fx * point_camera[0] / point_camera[2]) + cx;
        float y_2d = (fy * point_camera[1] / point_camera[2]) + cy; 
        LOGI("x_2d: %{public}f, y_2d: %{public}f", x_2d, y_2d);
        // Store the projected 2D point
        projected_points.push_back(cv::Point(x_2d, y_2d));
    }
    
    return projected_points;
}

// Modified createEnlargedFace function
void createEnlargedFace(const std::string scenepath, DataLoader::InputInitStruct& data, 
                                   std::vector<FaceProcessingInfo>& faceInfos,
                                   const std::pair<int, int>& selectedIndices) {
    // Create directories
    std::filesystem::create_directories(scenepath + "/human");
    std::filesystem::create_directories(scenepath + "/human/images");
    std::filesystem::create_directories(scenepath + "/human/faces");
    
    // Copy intr_tmp multiple times to support adding face regions
    if (data.intr_tmp.size() == 1 && data.extr_tmp.size() > 1) {
        size_t N = data.extr_tmp.size();
        std::vector<float> first_intrinsic = data.intr_tmp[0];
        data.intr_tmp.resize(N, first_intrinsic);
    }

    // Parse keypoints file to calculate optimal face dimensions
    std::string keypointsPath = scenepath + "/keypoints.txt";
    std::vector<KeypointData> keypoints = parseKeypointsFile(keypointsPath);
    cv::Size optimalFaceSize = calculateOptimalFaceDimensions(keypoints);
    optimalFaceSize = cv::Size(optimalFaceSize.width * RATIO, optimalFaceSize.height * RATIO);
    
    LOGI("Optimal face dimensions: width=%{public}d, height=%{public}d", 
         optimalFaceSize.width, optimalFaceSize.height);
    
    // Process each face
    for (int i = 0; i < faceInfos.size(); ++i) {
        if (!faceInfos[i].isWithinBounds) {
            continue;
        }
        
        std::string basename = faceInfos[i].imageName;
        int imgIndex = -1;
        
        // Find the corresponding image index
        for (int j = 0; j < data.img_list.size(); ++j) {
            if (getBaseName(data.img_list[j]) == basename) {
                imgIndex = j;
                break;
            }
        }
        
        if (imgIndex == -1) {
            continue;
        }
        
        // Read original image
        cv::Mat orig_img = cv::imread(data.img_list[imgIndex]);
        cv::Point scaledFaceCenter = faceInfos[i].faceCenter;
        
        // Check if this is one of the selected images and calculate roi_bbox
        if (i == selectedIndices.first || i == selectedIndices.second) {
            cv::Point2d center(scaledFaceCenter.x, scaledFaceCenter.y);
            cv::Point2d half(optimalFaceSize.width * 0.5, optimalFaceSize.height * 0.5);
            cv::Point2d pt1 = center - half;
            cv::Point2d pt2 = center + half;
//            faceInfos[i].roi_bbox = cv::Rect2d(pt1, pt2);
        }
        
        // Schedule 2: Optimal face crops (no intrinsics calculation)
        auto begin = clk::now();
        cropAndSaveFace(orig_img, scaledFaceCenter, optimalFaceSize, scenepath, basename);
        LOGW("Beauty filter takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        // Schedule 1: Original 480x640 crops with intrinsics calculation
        cv::Point topLeft;
        orig_img = cv::imread(data.img_list[imgIndex]); // Reload beautifed face
        cv::Mat cropped_img_480x640 = cropImage(orig_img, scaledFaceCenter, 
                                                cv::Size(480, 640), topLeft);
        cv::imwrite(scenepath + "/human/images/" + basename, cropped_img_480x640);
        data.img_list.push_back(scenepath + "/human/images/" + basename);
        
        // Calculate new intrinsics for the 480x640 crop
        std::vector<float> intr = data.intr_tmp[0];
        std::vector<float> new_intr = calculateNewIntrinsics(intr[0], intr[1], intr[2], intr[3], 
                                                            topLeft, cv::Size(480, 640), RATIO);
        LOGI("fx:%{public}f fy:%{public}f cx:%{public}f cy:%{public}f", 
             new_intr[0], new_intr[1], new_intr[2], new_intr[3]);
        data.intr_tmp.push_back(new_intr);
        data.extr_tmp.push_back(data.extr_tmp[imgIndex]);
        data.time_stamps.push_back(data.time_stamps[imgIndex]);
    }
    
    data.extr_tmp_test = data.extr_tmp;
}