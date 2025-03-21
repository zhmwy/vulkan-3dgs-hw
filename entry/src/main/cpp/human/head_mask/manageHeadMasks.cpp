#include "manageHeadMasks.h"

namespace fs = std::filesystem;

// Structure to hold bounding box information
struct BoundingBox {
    int x, y;         // top-left corner
    int width, height; // dimensions
    cv::Point center;  // center point
};

// Function to calculate bounding box for a mask image
BoundingBox calculateBoundingBox(const cv::Mat& mask) {
    // Find non-zero (non-black) pixels
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(mask, nonZeroPoints);
    
    if (nonZeroPoints.empty()) {
        // If no non-black pixels, use the center of the image
        BoundingBox result;
        result.x = 0;
        result.y = 0;
        result.width = 0;
        result.height = 0;
        result.center = cv::Point(mask.cols / 2, mask.rows / 2);
        return result;
    }
    
    // Get bounding box
    cv::Rect bbox = cv::boundingRect(nonZeroPoints);
    
    BoundingBox result;
    result.x = bbox.x;
    result.y = bbox.y;
    result.width = bbox.width;
    result.height = bbox.height;
    result.center = cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    
    return result;
}

// Function to compute the final bounding box size from a list of bounding boxes
cv::Size calculateFinalBoundingBoxSize(const std::vector<BoundingBox>& bboxes) {
    int maxWidth = 0;
    int maxHeight = 0;
    
    for (const auto& bbox : bboxes) {
        maxWidth = std::max(maxWidth, bbox.width);
        maxHeight = std::max(maxHeight, bbox.height);
    }
    
    return cv::Size(maxWidth, maxHeight);
}

// Function to crop image based on center point and size
cv::Mat cropImage(const cv::Mat& image, const cv::Point& center, const cv::Size& size, cv::Point& actualTopLeft) {
    // Calculate top-left corner
    int x = center.x - size.width / 2;
    int y = center.y - size.height / 2;
    
    // Ensure the crop region is within image bounds
    x = std::max(0, std::min(x, image.cols - size.width));
    y = std::max(0, std::min(y, image.rows - size.height));
    
    // Save the actual top-left corner used
    actualTopLeft = cv::Point(x, y);
    
    // Create the ROI
    cv::Rect roi(x, y, size.width, size.height);
    
    // Return the cropped image
    return image(roi).clone();
}

// Function to calculate new camera intrinsics
std::string calculateNewIntrinsics(
    const std::string& camera_name,
    double fx, double fy, 
    double cx, double cy,
    const cv::Point& crop_origin, 
    const cv::Size& highres_crop_size,
    double scale_factor) {
    
    // Calculate new principal point after cropping
    // Adjust for the crop offset and scale to high-res
    double new_cx = (cx - crop_origin.x) * scale_factor;
    double new_cy = (cy - crop_origin.y) * scale_factor;
    std::cout << crop_origin.x << " " << crop_origin.y << std::endl;
    // Scale focal lengths
    double new_fx = fx * scale_factor;
    double new_fy = fy * scale_factor;
    
    // Format the new intrinsics
    std::stringstream ss;
    ss << 0 << " PINHOLE " 
       << highres_crop_size.width << " " 
       << highres_crop_size.height << " "
       << new_fx << " "
       << new_fy << " "
       << new_cx << " "
       << new_cy;
    
    return ss.str();
}

// Process all images
cv::Size processImages(
    const std::string& mask_folder, 
    const std::string& highres_folder,
    const std::string& images_folder,
    const std::string& slam_folder,
    std::vector<float>& intr) {
    
    std::string intrinsics_output_file = slam_folder + "/cameras.txt";
    
    // Get all mask files
    std::vector<fs::path> mask_files;
    for (const auto& entry : fs::directory_iterator(mask_folder)) {
        if (entry.is_regular_file()) {
            mask_files.push_back(entry.path());
        }
    }
    
    // Sort mask files by name
    std::sort(mask_files.begin(), mask_files.end());
    
    // Calculate the scale factor between high-res and mask images
    const double scale_factor = 2.25; // 1440/480 = 1920/640 = 3
    
    // Original camera intrinsics
    double fx = intr[0];
    double fy = intr[1];
    double cx = intr[2];
    double cy = intr[3];

    // Gather all bounding boxes first to determine final size
    std::vector<BoundingBox> all_bboxes;
    
    for (const auto& mask_file : mask_files) {
        // Load mask image
        cv::Mat mask = cv::imread(mask_file.string(), cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            LOGE("Failed to load mask from %{public}s", mask_file.string().c_str());
            continue;
        }
        
        // Threshold to ensure binary mask
        cv::threshold(mask, mask, 125, 255, cv::THRESH_BINARY);
        
        // Calculate bounding box
        BoundingBox bbox = calculateBoundingBox(mask);
        all_bboxes.push_back(bbox);
    }
    
    if (all_bboxes.empty()) {
        LOGE("No valid bounding boxes found!");
    }
    
    // Calculate final bounding box size
    cv::Size final_bbox_size = calculateFinalBoundingBoxSize(all_bboxes);
    LOGI("Final bounding box size: %{public}d x %{public}d", final_bbox_size.width, final_bbox_size.height);
    
    // Scale the final bounding box size for high-res images
    cv::Size highres_bbox_size(final_bbox_size.width * scale_factor, final_bbox_size.height * scale_factor);
    
    // Open file for writing new intrinsics
    std::ofstream intrinsics_file(intrinsics_output_file);
    if (!intrinsics_file.is_open()) {
        LOGE("Failed to open intrinsics output file from %{public}s",intrinsics_output_file.c_str());
    }
    
    // Process each image pair
    for (size_t i = 0; i < mask_files.size(); ++i) {
        std::string mask_filename = mask_files[i].filename().string();
        
        // Extract camera name from filename (without extension)
        std::string camera_name = mask_filename.substr(0, mask_filename.find_last_of('.'));
        
        // Construct high-res filename (assuming same name)
        std::string highres_path = (fs::path(highres_folder) / mask_filename).string();
        
        // Load mask image
        cv::Mat mask = cv::imread(mask_files[i].string(), cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            LOGE("Failed to load mask from %{public}s", mask_files[i].string().c_str());
            continue;
        }
        
        // Threshold to ensure binary mask
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
        
        // Load high-res image
        cv::Mat highres = cv::imread(highres_path);
        if (highres.empty()) {
            LOGE("Failed to load high-res image from %{public}s", highres_path.c_str());
            continue;
        }
        
        // Calculate bounding box for mask
        BoundingBox bbox = all_bboxes[i];
        
        // Crop mask using the final bbox size
        cv::Point mask_crop_origin;
        cv::Mat cropped_mask = cropImage(mask, bbox.center, final_bbox_size, mask_crop_origin);
        
        // Scale center for high-res image
        cv::Point highres_center(bbox.center.x * scale_factor, bbox.center.y * scale_factor);
        cv::Point highres_crop_origin;
        
        // Crop high-res image
        cv::Mat cropped_highres = cropImage(highres, highres_center, highres_bbox_size, highres_crop_origin);
        
        // Resize the cropped mask to match the high-res crop
        cv::Mat resized_mask;
        cv::resize(cropped_mask, resized_mask, highres_bbox_size, 0, 0, cv::INTER_NEAREST);
        
        // Apply mask to high-res image
        cv::Mat result;
        cv::bitwise_and(cropped_highres, cropped_highres, result, resized_mask);
        
        // Save result
        std::string output_path = (fs::path(images_folder) / mask_filename).string();
        cv::imwrite(output_path, result);
        cv::imwrite("/data/storage/el2/base/haps/entry/files/temp/" + std::to_string(i) + ".jpg", result);

        LOGI("Saving new images to %{public}s.", output_path.c_str());
        
        // Calculate new intrinsics
        std::string new_intrinsics = calculateNewIntrinsics(
            camera_name,
            fx, fy, cx, cy,
            mask_crop_origin, 
            highres_bbox_size,
            scale_factor
        );
        
        // Write to file
        intrinsics_file << new_intrinsics << std::endl;
    }
    
    intrinsics_file.close();
    LOGI("Processing completed. Processed %{public}zu images.", mask_files.size());
    LOGI("Saving intrinsics to %{public}s.", intrinsics_output_file.c_str());
    
    return highres_bbox_size;
}

void manageHeadMasks(std::string dataPath, int& width, int& height, std::vector<float>& intr) {
    // Default paths
    std::string mask_folder = dataPath + "/head/head_masks";
    std::string highres_folder = dataPath + "/images_orig";
    std::string output_img_folder = dataPath + "/head/images";
    std::string output_slam_folder = dataPath + "/head/slam";
    
    // Create output folder if it doesn't exist
    if (!fs::exists(output_img_folder)) {
        LOGI("Creating %{public}s", output_img_folder.c_str());
        fs::create_directories(output_img_folder);
    }
    if (!fs::exists(output_slam_folder)) {
        LOGI("Creating %{public}s", output_slam_folder.c_str());
        fs::create_directories(output_slam_folder);
    }
    std::filesystem::copy_file(dataPath + "/slam/images.txt", output_slam_folder + "/images.txt", std::filesystem::copy_options::overwrite_existing);
//     std::filesystem::copy_file(slam_folder + "/points3D.txt", output_slam_folder + "/points3D.txt", std::filesystem::copy_options::overwrite_existing);

    cv::Size head_size = processImages(mask_folder, highres_folder, 
    output_img_folder, output_slam_folder, intr);
    width = head_size.width;
    height = head_size.height;
}