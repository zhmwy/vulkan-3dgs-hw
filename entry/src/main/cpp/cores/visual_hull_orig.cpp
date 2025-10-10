#include "visual_hull.h"

namespace VisualHullSegmentation {

using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3>;
using MatX2f = Eigen::Matrix<float, Eigen::Dynamic, 2>;
using VecXf = Eigen::VectorXf;

// Convert quaternion to rotation matrix
Mat3 quaternionToRotationMatrix(float qw, float qx, float qy, float qz) {
    Eigen::Quaternionf q(qw, qx, qy, qz);
    q.normalize();
    return q.toRotationMatrix();
}

// Vectorized projection of 3D points to 2D image coordinates
void projectPointsBatch(
    const MatX3f& points_world,
    const Mat3& R,
    const Vec3& T,
    float fx, float fy, float cx, float cy,
    MatX2f& projected_points,
    std::vector<bool>& valid_depth
) {
    // Transform to camera coordinates: points_cam = R * points_world^T + T
    MatX3f points_cam = (R * points_world.transpose()).transpose();
    points_cam.rowwise() += T.transpose();
    
    // Check for valid depth (z > 0)
    valid_depth.resize(points_cam.rows());
    for (int i = 0; i < points_cam.rows(); ++i) {
        valid_depth[i] = points_cam(i, 2) > 0;
    }
    
    // Project to image plane (vectorized)
    projected_points.resize(points_cam.rows(), 2);
    VecXf z_inv = points_cam.col(2).cwiseInverse();
    projected_points.col(0) = (points_cam.col(0).array() * z_inv.array()) * fx + cx;
    projected_points.col(1) = (points_cam.col(1).array() * z_inv.array()) * fy + cy;
}

// Vectorized check for points within mask
void checkPointsInMaskBatch(
    const MatX2f& projected_points,
    const std::vector<bool>& valid_depth,
    const cv::Mat& mask,
    int width, int height,
    std::vector<bool>& in_mask
) {
    in_mask.resize(projected_points.rows(), false);
    
    #pragma omp parallel for
    for (int i = 0; i < projected_points.rows(); ++i) {
        if (!valid_depth[i]) {
            in_mask[i] = false;
            continue;
        }
        
        float x = projected_points(i, 0);
        float y = projected_points(i, 1);
        
        // Check if point is within image bounds
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int ix = static_cast<int>(x);
            int iy = static_cast<int>(y);
            in_mask[i] = (mask.at<uint8_t>(iy, ix) > 0);
        } else {
            in_mask[i] = false;
        }
    }
}

// Select cameras that are maximally spread apart
std::vector<int> selectWellDistributedCameras(
    const std::vector<DataLoader::InputInitStruct::DataExtr>& extr_tmp,
    int sequence_length,
    int num_cameras_to_select
) {
    // Ensure we don't select more cameras than available
    num_cameras_to_select = std::min(num_cameras_to_select, sequence_length);
    
    // Calculate forward vectors and positions for each camera
    std::vector<Vec3> forward_vectors;
    std::vector<Vec3> camera_positions;
    
    for (int i = 0; i < sequence_length; ++i) {
        const auto& extr = extr_tmp[i];
        Mat3 R = quaternionToRotationMatrix(
            extr.values[0], extr.values[1], 
            extr.values[2], extr.values[3]
        );
        Vec3 T(extr.values[4], extr.values[5], extr.values[6]);
        
        // Camera forward is -Z in camera space, transformed to world
        Vec3 forward = R.transpose() * Vec3(0, 0, -1);
        forward.normalize();
        forward_vectors.push_back(forward);
        
        // Camera position in world space (inverse of T in world-to-camera)
        Mat4 Rt = Mat4::Identity();
        Rt.block<3, 3>(0, 0) = R;
        Rt.block<3, 1>(0, 3) = T;
        Mat4 c2w = Rt.inverse();
        Vec3 cam_pos = c2w.block<3, 1>(0, 3);
        camera_positions.push_back(cam_pos);
    }
    
    // Use greedy algorithm to select maximally spread cameras
    std::vector<int> selected_indices;
    std::vector<bool> is_selected(sequence_length, false);
    
    // Start with the first camera (or could be random)
    selected_indices.push_back(0);
    is_selected[0] = true;
    
    // Iteratively select cameras that maximize minimum angle to already selected cameras
    while (selected_indices.size() < static_cast<size_t>(num_cameras_to_select)) {
        float max_min_angle = -1.0f;
        int best_candidate = -1;
        
        for (int i = 0; i < sequence_length; ++i) {
            if (is_selected[i]) continue;
            
            // Calculate minimum angle to all selected cameras
            float min_angle = std::numeric_limits<float>::max();
            for (int selected_idx : selected_indices) {
                // Use both viewing direction and position difference for diversity
                Vec3 dir_diff = forward_vectors[i] - forward_vectors[selected_idx];
                float dir_angle = dir_diff.norm();
                
                // Also consider spatial distance (normalized)
                Vec3 pos_diff = camera_positions[i] - camera_positions[selected_idx];
                float spatial_dist = pos_diff.norm();
                
                // Combined metric (weighted sum of angular and spatial distance)
                float combined_metric = dir_angle + 0.3f * spatial_dist;
                min_angle = std::min(min_angle, combined_metric);
            }
            
            // Track the candidate with maximum minimum angle
            if (min_angle > max_min_angle) {
                max_min_angle = min_angle;
                best_candidate = i;
            }
        }
        
        if (best_candidate >= 0) {
            selected_indices.push_back(best_candidate);
            is_selected[best_candidate] = true;
        } else {
            break; // No more candidates
        }
    }
    
    return selected_indices;
}

// Add this new function for visualization
void visualizeProjection(
    const cv::Mat& background_image,
    const MatX2f& projected_points,
    const std::vector<bool>& valid_depth,
    const std::vector<bool>& in_mask,
    const std::string& output_path,
    int point_size = 2
) {
    cv::Mat vis_image = background_image.clone();
    
    // Make sure we have a color image
    if (vis_image.channels() == 1) {
        cv::cvtColor(vis_image, vis_image, cv::COLOR_GRAY2BGR);
    }
    
    // Draw points
    for (size_t i = 0; i < projected_points.rows(); ++i) {
        if (valid_depth[i]) {
            int x = static_cast<int>(projected_points(i, 0));
            int y = static_cast<int>(projected_points(i, 1));
            
            // Check if point is within image bounds
            if (x >= 0 && x < vis_image.cols && y >= 0 && y < vis_image.rows) {
                // Blue (BGR: 255,0,0) for points inside mask, Red (BGR: 0,0,255) for outside
                cv::Scalar color = in_mask[i] ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
                cv::circle(vis_image, cv::Point(x, y), point_size, color, -1);
            }
        }
    }
    
    // Save the visualization
    cv::imwrite(output_path, vis_image);
}

// Main entrance function for segmentation with flexible parameters
DataLoader::InputInitStruct segmentGaussiansFromMask(
    const DataLoader::InputInitStruct& target,
    int num_cameras = 4,           // Flexible number of cameras (4-8)
    int max_outside_count = 2,     // Flexible threshold for outside count
    float threshold_scale = 1.0f,
    bool use_median = false
) {
    // Validate parameters
//     num_cameras = std::max(4, std::min(8, num_cameras));
    num_cameras = std::min(num_cameras, target.sequence_length);
    max_outside_count = std::min(max_outside_count, num_cameras - 1);
    
    std::cout << "Starting visual hull segmentation (projection-based)..." << std::endl;
    std::cout << "Number of Gaussians: " << target.positions.size() << std::endl;
    std::cout << "Sequence length: " << target.sequence_length << std::endl;
    std::cout << "Number of cameras to use: " << num_cameras << std::endl;
    std::cout << "Maximum outside count threshold: " << max_outside_count << std::endl;
    
    // Create visualization directory
    std::string vis_dir = target.basepath + "/visual_hull/proj";
    std::filesystem::create_directories(vis_dir);
    std::cout << "Visualization directory: " << vis_dir << std::endl;
    
    // Select well-distributed cameras
    std::vector<int> selected_indices = selectWellDistributedCameras(
        target.extr_tmp, target.sequence_length, num_cameras
    );
    
    std::cout << "Selected camera indices: ";
    for (int idx : selected_indices) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    
    // Convert all gaussian positions to Eigen matrix for vectorized operations
    MatX3f all_positions(target.positions.size(), 3);
    for (size_t i = 0; i < target.positions.size(); ++i) {
        all_positions(i, 0) = target.positions[i].x;
        all_positions(i, 1) = target.positions[i].y;
        all_positions(i, 2) = target.positions[i].z;
    }
    
    // Track visibility statistics
    std::vector<int> visible_count(target.positions.size(), 0);
    std::vector<int> outside_count(target.positions.size(), 0);
    std::vector<bool> is_active(target.positions.size(), true);
    
    // Process each selected camera
    for (size_t cam_i = 0; cam_i < selected_indices.size(); ++cam_i) {
        int cam_idx = selected_indices[cam_i];
        
        std::cout << "Processing camera " << cam_idx << " (" << (cam_i + 1) 
                  << "/" << selected_indices.size() << ")" << std::endl;
        
        // Read and prepare mask
        cv::Mat mask_img = cv::imread(target.mask_list[cam_idx], cv::IMREAD_GRAYSCALE);
        if (mask_img.empty()) {
            std::cerr << "Failed to read mask: " << target.mask_list[cam_idx] << std::endl;
            continue;
        }
        
        cv::Mat resized_mask;
        cv::resize(mask_img, resized_mask, cv::Size(target.width, target.height));
        
        cv::Mat binary_mask;
        cv::threshold(resized_mask, binary_mask, 0, 255, cv::THRESH_BINARY);
        
        // Get camera parameters
        const auto& extr = target.extr_tmp[cam_idx];
        const auto& intr = target.intr_tmp[0];
        
        float fx = intr[0];
        float fy = intr[1];
        float cx = intr[2];
        float cy = intr[3];
        
        // Build world-to-camera transformation
        Mat3 R = quaternionToRotationMatrix(
            extr.values[0], extr.values[1], 
            extr.values[2], extr.values[3]
        );
        Vec3 T(extr.values[4], extr.values[5], extr.values[6]);
        
        // Collect indices of active gaussians for efficient processing
        std::vector<int> active_indices;
        for (size_t i = 0; i < target.positions.size(); ++i) {
            if (is_active[i]) {
                active_indices.push_back(i);
            }
        }
        
        if (active_indices.empty()) {
            std::cout << "No active gaussians remaining, early termination." << std::endl;
            break;
        }
        
        // Create matrix of active positions
        MatX3f active_positions(active_indices.size(), 3);
        for (size_t j = 0; j < active_indices.size(); ++j) {
            active_positions.row(j) = all_positions.row(active_indices[j]);
        }
        
        // Project all active points to this camera (vectorized)
        MatX2f projected_points;
        std::vector<bool> valid_depth;
        projectPointsBatch(active_positions, R, T, fx, fy, cx, cy, 
                          projected_points, valid_depth);
        
        // Check which points fall within the mask (vectorized)
        std::vector<bool> in_mask;
        checkPointsInMaskBatch(projected_points, valid_depth, binary_mask, 
                              target.width, target.height, in_mask);
        
        cv::Mat vis_background;
        vis_background = binary_mask.clone();
            
        // Generate output filename
        std::stringstream ss;
        ss << vis_dir << "/view_" << std::setfill('0') << std::setw(3) << cam_idx 
           << "_cam_" << std::setfill('0') << std::setw(2) << cam_i << ".png";
        std::string vis_output_path = ss.str();
        
        // Create visualization
        visualizeProjection(vis_background, projected_points, valid_depth, 
                          in_mask, vis_output_path, 2);
        std::cout << "  Saved visualization to: " << vis_output_path << std::endl;
        
        // Update statistics for each gaussian
        for (size_t j = 0; j < active_indices.size(); ++j) {
            int idx = active_indices[j];
            if (in_mask[j]) {
                visible_count[idx]++;
            } else {
                outside_count[idx]++;
                // Early termination: if outside more than threshold, mark as inactive
                if (outside_count[idx] > max_outside_count) {
                    is_active[idx] = false;
                }
            }
        }
        
        int remaining_active = std::count(is_active.begin(), is_active.end(), true);
        std::cout << "Active gaussians after view " << cam_idx << ": " << remaining_active 
                  << " (visible in this view: " 
                  << std::count(in_mask.begin(), in_mask.end(), true) << ")" << std::endl;
    }
    
    // Create final mask: gaussians with outside_count <= max_outside_count are foreground
    std::vector<bool> final_mask(target.positions.size());
    int foreground_count = 0;
    for (size_t i = 0; i < target.positions.size(); ++i) {
        final_mask[i] = (outside_count[i] <= max_outside_count);
        if (final_mask[i]) {
            foreground_count++;
        }
    }
    
    std::cout << "\nSegmentation Statistics:" << std::endl;
    std::cout << "  Total gaussians: " << target.positions.size() << std::endl;
    std::cout << "  Foreground gaussians: " << foreground_count << std::endl;
    std::cout << "  Background gaussians: " << (target.positions.size() - foreground_count) << std::endl;
    std::cout << "  Used " << selected_indices.size() << " cameras with threshold " << max_outside_count << std::endl;
    
    // Create output with only foreground Gaussians
    DataLoader::InputInitStruct output;
    output.width = target.width;
    output.height = target.height;
    output.sequence_length = target.sequence_length;
    output.img_list = target.img_list;
    output.mask_list = target.mask_list;
    output.extr_tmp = target.extr_tmp;
    output.extr_tmp_test = target.extr_tmp_test;
    output.intr_tmp = target.intr_tmp;
    output.images = target.images;
    output.images_low_res = target.images_low_res;
    output.masks = target.masks;
    output.masks_low_res = target.masks_low_res;
    
    // Filter Gaussians based on final mask
    for (size_t i = 0; i < target.positions.size(); ++i) {
        if (final_mask[i]) {
            output.positions.push_back(target.positions[i]);
            output.scales.push_back(target.scales[i]);
            output.rotations.push_back(target.rotations[i]);
            output.opacities.push_back(target.opacities[i]);
            output.colors.push_back(target.colors[i]);
        }
    }
    
    std::cout << "Segmentation complete. Kept " << output.positions.size() 
              << " out of " << target.positions.size() << " Gaussians." << std::endl;
    
    return output;
}

} // namespace VisualHullSegmentation

// Main entrance function with flexible parameters
DataLoader::InputInitStruct performVisualHullSegmentation(
    const DataLoader::InputInitStruct& target, 
    int num_cameras,
    int max_outside_count,
    float threshold_scale, 
    bool use_median) {
    return VisualHullSegmentation::segmentGaussiansFromMask(
        target, num_cameras, max_outside_count, threshold_scale, use_median
    );
}