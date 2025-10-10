#include "visual_hull.h"
#include "hilog/log.h"

namespace VisualHullSegmentation {

using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3>;
using MatX2f = Eigen::Matrix<float, Eigen::Dynamic, 2>;
using VecXf = Eigen::VectorXf;

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0000
#define LOG_TAG "VisualHull"

// Helper function to find corresponding image index by matching filenames
int findImageIndexByMaskName(
    const std::string& mask_path,
    const std::vector<std::string>& img_list
) {
    // Extract filename from mask path
    std::filesystem::path mask_file(mask_path);
    std::string mask_filename = mask_file.filename().string();
    
    // Remove extension to get base name
    std::string mask_base_name = mask_filename.substr(0, mask_filename.find_last_of("."));
    
    // Search through img_list to find matching base name
    for (size_t i = 0; i < img_list.size(); ++i) {
        std::filesystem::path img_file(img_list[i]);
        std::string img_filename = img_file.filename().string();
        std::string img_base_name = img_filename.substr(0, img_filename.find_last_of("."));
        
        // Match base names (ignoring extensions)
        if (mask_base_name == img_base_name) {
            return static_cast<int>(i);
        }
    }
    
    return -1; // Not found
}

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

// Visualization function
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
    float max_outside_percentage = 0.3f,  // e.g., 0.3 means 30% of views
    float threshold_scale = 1.0f,
    bool use_median = false
) {
    LOGI("Starting visual hull segmentation (projection-based)...");
    LOGI("Number of Gaussians: %{public}zu", target.positions.size());
    LOGI("Sequence length: %{public}d", target.sequence_length);
    LOGI("Number of available masks: %{public}zu", target.mask_list.size());
    LOGI("Max outside percentage: %{public}f", max_outside_percentage);
    
    // Calculate max outside count based on percentage
    int max_outside_count = static_cast<int>(max_outside_percentage * target.mask_list.size());
    max_outside_count = std::max(0, std::min(max_outside_count, static_cast<int>(target.mask_list.size()) - 1));
    
    LOGI("Maximum outside count threshold: %{public}d", max_outside_count);
    
    // Create visualization directory
    std::string vis_dir = target.basepath + "/visual_hull/proj";
    std::filesystem::create_directories(vis_dir);
    LOGI("Visualization directory: %{public}s", vis_dir.c_str());
    
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
    
    // Process each available mask
    for (size_t mask_i = 0; mask_i < target.mask_list.size(); ++mask_i) {
        // Find the corresponding camera index by matching mask name with img_list
        int cam_idx = findImageIndexByMaskName(target.mask_list[mask_i], target.img_list);
        
        if (cam_idx < 0) {
            LOGE("Failed to find corresponding image for mask: %{public}s", target.mask_list[mask_i].c_str());
            continue;
        }
        
        LOGI("Processing mask %{public}zu/%{public}zu (camera index: %{public}d)", 
             mask_i + 1, target.mask_list.size(), cam_idx);
        
        // Read and prepare mask
        cv::Mat mask_img = cv::imread(target.mask_list[mask_i], cv::IMREAD_GRAYSCALE);
        if (mask_img.empty()) {
            LOGE("Failed to read mask: %{public}s", target.mask_list[mask_i].c_str());
            continue;
        }
        
        cv::Mat resized_mask;
        cv::resize(mask_img, resized_mask, cv::Size(target.width, target.height));
        
        cv::Mat binary_mask;
        cv::threshold(resized_mask, binary_mask, 0, 255, cv::THRESH_BINARY);
        
        // Get camera parameters using the matched index
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
            LOGI("No active gaussians remaining, early termination.");
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
        ss << vis_dir << "/mask_" << std::setfill('0') << std::setw(3) << mask_i 
           << "_cam_" << std::setfill('0') << std::setw(3) << cam_idx << ".png";
        std::string vis_output_path = ss.str();
        
        // Create visualization
        visualizeProjection(vis_background, projected_points, valid_depth, 
                          in_mask, vis_output_path, 2);
        LOGI("  Saved visualization to: %{public}s", vis_output_path.c_str());
        
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
        int visible_in_view = std::count(in_mask.begin(), in_mask.end(), true);
        LOGI("Active gaussians after mask %{public}zu: %{public}d (visible in this view: %{public}d)", 
             mask_i, remaining_active, visible_in_view);
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
    
    LOGI("\nSegmentation Statistics:");
    LOGI("  Total gaussians: %{public}zu", target.positions.size());
    LOGI("  Foreground gaussians: %{public}d", foreground_count);
    LOGI("  Background gaussians: %{public}zu", target.positions.size() - foreground_count);
    LOGI("  Used %{public}zu masks with threshold %{public}d", target.mask_list.size(), max_outside_count);
    
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
    output.basepath = target.basepath;
    
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
    
    LOGI("Segmentation complete. Kept %{public}zu out of %{public}zu Gaussians.", 
         output.positions.size(), target.positions.size());
    
    return output;
}

} // namespace VisualHullSegmentation

// Main entrance function with updated parameters
DataLoader::InputInitStruct performVisualHullSegmentation(
    const DataLoader::InputInitStruct& target, 
    float max_outside_percentage,
    float threshold_scale, 
    bool use_median) {
    return VisualHullSegmentation::segmentGaussiansFromMask(
        target, max_outside_percentage, threshold_scale, use_median);
}