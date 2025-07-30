#include "face.h"

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
std::vector<float> calculateNewIntrinsics(
    float fx, float fy, 
    float cx, float cy,
    const cv::Point& crop_origin, 
    const cv::Size& highres_crop_size,
    float scale_factor) {

    float new_cx = cx * scale_factor - crop_origin.x;
    float new_cy = cy * scale_factor - crop_origin.y;
    
    // Scale focal lengths
    float new_fx = fx * scale_factor;
    float new_fy = fy * scale_factor;
    
    std::vector<float> new_intr = {new_fx, new_fy, new_cx, new_cy};
    
    return new_intr;
}

// Function to parse keypoints.txt file
std::vector<KeypointData> parseKeypointsFile(const std::string& filepath) {
    std::vector<KeypointData> keypoints;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        LOGI("Failed to open keypoints.txt file at %{public}s", filepath.c_str());
        return keypoints;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;
        
        // First line should be filename
        KeypointData kp;
        kp.filename = line;
        
        // Read bounding box
        if (!std::getline(file, line)) break;
        std::istringstream bbox_stream(line);
        bbox_stream >> kp.bbox.x >> kp.bbox.y >> kp.bbox.width >> kp.bbox.height;
        
        // Read 17 keypoints
        std::vector<cv::Point2f> points(17);
        std::vector<float> confidences(17);
        
        for (int i = 0; i < 17; ++i) {
            if (!std::getline(file, line)) break;
            std::istringstream kp_stream(line);
            kp_stream >> points[i].x >> points[i].y >> confidences[i];
        }
        
        // Assign specific keypoints we need
        kp.nose = points[0];
        kp.leftEye = points[1];
        kp.rightEye = points[2];
        kp.leftEar = points[3];
        kp.rightEar = points[4];
        kp.leftShoulder = points[5];
        kp.rightShoulder = points[6];
        
        // Store confidence values for keypoints we use in dimension calculation
        kp.leftEarConf = confidences[3];
        kp.rightEarConf = confidences[4];
        kp.leftShoulderConf = confidences[5];
        kp.rightShoulderConf = confidences[6];
        
        keypoints.push_back(kp);
    }
    
    file.close();
    return keypoints;
}

// Calculate optimal face bounding box dimensions based on keypoints
cv::Size calculateOptimalFaceDimensions(const std::vector<KeypointData>& keypoints) {
    float maxEarDistance = 0.0f;
    float maxShoulderToTopDistance = 0.0f;
    const float confidenceThreshold = 0.7f;
    
    for (const auto& kp : keypoints) {
        // Calculate ear-to-ear distance only if both ears have high confidence
        if (kp.leftEarConf >= confidenceThreshold && kp.rightEarConf >= confidenceThreshold) {
            float earDistance = std::abs(kp.leftEar.x - kp.rightEar.x);
            maxEarDistance = std::max(maxEarDistance, earDistance);
        }
        
        // Calculate distance from bbox top to shoulders
        // Use the shoulder with higher confidence, or both if both are confident
        float shoulderY = 0.0f;
        bool validShoulder = false;
        
        if (kp.leftShoulderConf >= confidenceThreshold && kp.rightShoulderConf >= confidenceThreshold) {
            shoulderY = std::max(kp.leftShoulder.y, kp.rightShoulder.y);
            validShoulder = true;
        } else if (kp.leftShoulderConf >= confidenceThreshold) {
            shoulderY = kp.leftShoulder.y;
            validShoulder = true;
        } else if (kp.rightShoulderConf >= confidenceThreshold) {
            shoulderY = kp.rightShoulder.y;
            validShoulder = true;
        }
        
        if (validShoulder) {
            float distanceToTop = shoulderY - kp.bbox.y;
            maxShoulderToTopDistance = std::max(maxShoulderToTopDistance, distanceToTop);
        }
    }
    // Add some padding (optional)
    int width = static_cast<int>(maxEarDistance * 1.2);
    int height = static_cast<int>(maxShoulderToTopDistance * 0.9);
    
    return cv::Size(width, height);
}

void cropAndSaveFace(const cv::Mat& orig_img, const cv::Point& faceCenter, 
                     cv::Size& cropSize, const std::string& scenePath, const std::string& basename) {
    // Crop the face region
    cv::Point topLeft;
    cv::Mat cropped_img = cropImage(orig_img, faceCenter, cropSize, topLeft);
    
    // Apply beauty filters to the cropped face
    float smoothing_level = 0.3f;  // Adjustable parameter (0.0 to 1.0)
    float whitening_level = 0.1f;  // Adjustable parameter (0.0 to 1.0)
    cv::Mat beautified_face = applyBeautyFilters(cropped_img, smoothing_level, whitening_level);
    cv::imwrite(scenePath + "/human/faces/" + basename, beautified_face);
    
    // Create a copy of the original image to paste the beautified face onto
    cv::Mat result_img = orig_img.clone();
    
    // Calculate the region where to paste the beautified face
    cv::Rect face_roi(topLeft.x, topLeft.y, beautified_face.cols, beautified_face.rows);
    
    // Ensure the ROI is within image bounds
    face_roi &= cv::Rect(0, 0, orig_img.cols, orig_img.rows);
    
    // If the face ROI is valid, paste the beautified face back
    if (face_roi.width > 0 && face_roi.height > 0) {
        // Resize beautified face if needed due to boundary clipping
        if (face_roi.width != beautified_face.cols || face_roi.height != beautified_face.rows) {
            cv::resize(beautified_face, beautified_face, cv::Size(face_roi.width, face_roi.height));
        }
        // Simply copy the beautified face to the result image
        beautified_face.copyTo(result_img(face_roi));
    }
    
    // Save the final result
    cv::imwrite(scenePath + "/images/" + basename, result_img);
}