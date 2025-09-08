#include "visual_hull.h"

namespace VisualHullSegmentation {

using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;

// Convert quaternion to rotation matrix
Mat3 quaternionToRotationMatrix(float qw, float qx, float qy, float qz) {
    Eigen::Quaternionf q(qw, qx, qy, qz);
    q.normalize();
    return q.toRotationMatrix();
}

// Get ray directions for all pixels in camera coordinate
Eigen::MatrixXf getRayDirections(int H, int W, float fx, float fy, float cx, float cy) {
    Eigen::MatrixXf directions(H * W, 3);
    
    int idx = 0;
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            float pixel_center = 0.5f;
            directions(idx, 0) = (i + pixel_center - cx) / fx;
            directions(idx, 1) = (j + pixel_center - cy) / fy;
            directions(idx, 2) = 1.0f;
            idx++;
        }
    }
    
    return directions;
}

// Calculate distance from ray to point
float rayPointDistance(const Vec3& ray_origin, const Vec3& ray_dir, const Vec3& point) {
    Vec3 diff = point - ray_origin;
    Vec3 cross_product = diff.cross(ray_dir);
    return cross_product.norm();
}

// Trace Gaussians with rays from segmentation mask
std::vector<bool> traceGaussians(
    const std::vector<Base::Math::Vec3>& positions,
    const std::vector<Base::Math::Vec3>& scales,
    const cv::Mat& segmap,
    const Mat4& c2w,
    float fx, float fy, float cx, float cy,
    float threshold_scale = 1.0f
) {
    int H = segmap.rows;
    int W = segmap.cols;
    
    // Get all ray directions
    Eigen::MatrixXf all_rays_d = getRayDirections(H, W, fx, fy, cx, cy);
    
    // Collect rays for pixels in the mask
    std::vector<Vec3> selected_rays_d_cam;
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            if (segmap.at<uint8_t>(j, i) > 0) {
                int idx = j * W + i;
                selected_rays_d_cam.push_back(Vec3(
                    all_rays_d(idx, 0),
                    all_rays_d(idx, 1),
                    all_rays_d(idx, 2)
                ));
            }
        }
    }
    
    if (selected_rays_d_cam.empty()) {
        return std::vector<bool>(positions.size(), false);
    }
    
    // Transform rays to world coordinates
    Mat3 R = c2w.block<3, 3>(0, 0);
    Vec3 ray_origin = c2w.block<3, 1>(0, 3);
    
    std::vector<Vec3> rays_d_world;
    for (const auto& ray_cam : selected_rays_d_cam) {
        Vec3 ray_world = R * ray_cam;
        ray_world.normalize();
        rays_d_world.push_back(ray_world);
    }
    
    // Calculate global threshold based on Gaussian scales
    float sum_max_scales = 0.0f;
    for (const auto& scale : scales) {
        float max_scale = std::max({scale.x, scale.y, scale.z});
        sum_max_scales += max_scale;
    }
    float avg_threshold = (sum_max_scales / scales.size()) * threshold_scale;
    
    // Check each Gaussian
    std::vector<bool> masks(positions.size());
    
    for (size_t g_idx = 0; g_idx < positions.size(); ++g_idx) {
        Vec3 gaussian_pos(positions[g_idx].x, positions[g_idx].y, positions[g_idx].z);
        
        // Find minimum distance to any ray
        float min_dist = std::numeric_limits<float>::max();
        for (const auto& ray_dir : rays_d_world) {
            LOGI("%{public}d", g_idx);
            float dist = rayPointDistance(ray_origin, ray_dir, gaussian_pos);
            min_dist = std::min(min_dist, dist);
        }
        
        // Check if Gaussian is close enough to any ray
        masks[g_idx] = (min_dist < avg_threshold);
    }
    
    return masks;
}

// Select cameras based on viewing directions
std::vector<int> selectCamerasByDirection(
    const std::vector<DataLoader::InputInitStruct::DataExtr>& extr_tmp,
    int sequence_length
) {
    // Calculate forward vectors for each camera
    std::vector<Vec3> forward_vectors;
    for (int i = 0; i < sequence_length; ++i) {
        const auto& extr = extr_tmp[i];
        Mat3 R = quaternionToRotationMatrix(
            extr.values[0], extr.values[1], 
            extr.values[2], extr.values[3]
        );
        
        // Camera forward is -Z in camera space, transformed to world
        Vec3 forward = R.transpose() * Vec3(0, 0, -1);
        forward.normalize();
        forward_vectors.push_back(forward);
    }
    
    // Define 4 target directions: front, back, left, right
    std::vector<Vec3> target_directions = {
        Vec3(0, 0, 1),   // front (z+)
        Vec3(0, 0, -1),  // back (z-)
        Vec3(-1, 0, 0),  // left (x-)
        Vec3(1, 0, 0)    // right (x+)
    };
    
    // Find best matching camera for each target direction
    std::vector<int> selected_indices;
    for (const auto& target : target_directions) {
        float max_dot = -2.0f;
        int best_idx = -1;
        
        for (int i = 0; i < sequence_length; ++i) {
            float dot = forward_vectors[i].dot(target);
            if (dot > max_dot) {
                max_dot = dot;
                best_idx = i;
            }
        }
        
        if (best_idx >= 0) {
            selected_indices.push_back(best_idx);
        }
    }
    
    return selected_indices;
}

// Main entrance function for segmentation
DataLoader::InputInitStruct segmentGaussiansFromMask(
    const DataLoader::InputInitStruct& target,
    float threshold_scale = 1.0f,
    bool use_median = false
) {
    std::cout << "Starting visual hull segmentation..." << std::endl;
    std::cout << "Number of Gaussians: " << target.positions.size() << std::endl;
    std::cout << "Sequence length: " << target.sequence_length << std::endl;
    
    // Select 4 cameras based on viewing directions
    std::vector<int> selected_indices = selectCamerasByDirection(
        target.extr_tmp, target.sequence_length
    );
    
    std::cout << "Selected camera indices: ";
    for (int idx : selected_indices) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    
    // Initialize mask for all Gaussians (start with all true, then AND with each view)
    std::vector<bool> global_mask(target.positions.size(), true);
    
    // Process each selected camera
    for (size_t cam_i = 0; cam_i < selected_indices.size(); ++cam_i) {
        int cam_idx = selected_indices[cam_i];
        
        std::cout << "Processing camera " << cam_idx << " (" << (cam_i + 1) 
                  << "/" << selected_indices.size() << ")" << std::endl;
        
        // Read mask image
        cv::Mat mask_img = cv::imread(target.mask_list[cam_idx], cv::IMREAD_GRAYSCALE);
        if (mask_img.empty()) {
            LOGE("Failed to read mask: %{public}s", target.mask_list[cam_idx].c_str());
            continue;
        }
        
        // Resize mask to match camera resolution
        cv::Mat resized_mask;
        cv::resize(mask_img, resized_mask, cv::Size(target.width, target.height));
        
        // Convert mask to binary
        cv::Mat binary_mask;
        cv::threshold(resized_mask, binary_mask, 0, 255, cv::THRESH_BINARY);
        
        // Get camera parameters
        const auto& extr = target.extr_tmp[cam_idx];
        const auto& intr = target.intr_tmp[cam_idx];
        
        float fx = intr[0];
        float fy = intr[1];
        float cx = intr[2];
        float cy = intr[3];
        
        // Build camera-to-world matrix
        Mat3 R = quaternionToRotationMatrix(
            extr.values[0], extr.values[1], 
            extr.values[2], extr.values[3]
        );
        Vec3 T(extr.values[4], extr.values[5], extr.values[6]);
        
        // Build world-to-camera matrix (Rt)
        Mat4 Rt = Mat4::Identity();
        Rt.block<3, 3>(0, 0) = R;
        Rt.block<3, 1>(0, 3) = T;
        
        // Invert to get camera-to-world matrix
        Mat4 c2w = Rt.inverse();
        
        // Trace Gaussians for this view
        std::vector<bool> view_mask = traceGaussians(
            target.positions,
            target.scales,
            binary_mask,
            c2w,
            fx, fy, cx, cy,
            threshold_scale
        );
        
        // AND with global mask (visual hull intersection)
        for (size_t i = 0; i < global_mask.size(); ++i) {
            global_mask[i] = global_mask[i] && view_mask[i];
        }
        
        int remaining = std::count(global_mask.begin(), global_mask.end(), true);
        LOGI("Remaining %{public}d Gaussians after view %{public}d", cam_idx, remaining);
    }
    
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
    
    // Filter Gaussians based on mask
    for (size_t i = 0; i < target.positions.size(); ++i) {
        if (global_mask[i]) {
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

// Main entrance function
DataLoader::InputInitStruct performVisualHullSegmentation(
    const DataLoader::InputInitStruct& target, float threshold_scale, bool use_median) {
    return VisualHullSegmentation::segmentGaussiansFromMask(
        target, threshold_scale, use_median
    );
}