#include "human/HiAI/HIAIModelManager.h"
#include "human/HiAI/utils.h"
#include "human/keypoints/keypoints.h"
#include "landmark/landmark.h"
#include "base/math/vector.h"
#include <filesystem>
#include "face.h"

// Enhanced function to process all images in a folder with fixed 480x640 crops
void detectAndProcessAllFaces(DataLoader::InputInitStruct& target, const std::string& scenePath, bool masked)
{
    // Construct model path
    size_t lastSlash = scenePath.find_last_of('/');
    std::string modelPath = scenePath.substr(0, lastSlash) + "/weights/yolov11-face.om";
    
    if (LoadModel(modelPath) != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "Model loading failed from path: %{public}s", modelPath.c_str());
        return;
    }
    
    OH_LOG_INFO(LOG_APP, "Successfully loaded face detection model");
    
    // Create output directories
    std::string facesDir = scenePath + "/faces";
    std::filesystem::create_directories(facesDir);
    
    std::string segmentedDir;
    if (masked) {
        segmentedDir = scenePath + "/segmented/face";
        std::filesystem::create_directories(segmentedDir);
        OH_LOG_INFO(LOG_APP, "Created segmented directory: %{public}s", segmentedDir.c_str());
    }
    
    OH_LOG_INFO(LOG_APP, "Created output directory: %{public}s", facesDir.c_str());
    
    // Store original sizes for tracking
    size_t original_img_list_size = target.img_list.size();
    size_t original_mask_list_size = target.mask_list.size();
    size_t original_intr_tmp_size = target.intr_tmp.size();
    size_t original_extr_tmp_size = target.extr_tmp.size();
    size_t original_time_stamps_size = target.time_stamps.size();
    
    // Copy intr_tmp multiple times if needed (following createEnlargedFace pattern)
    if (target.intr_tmp.size() == 1 && target.extr_tmp.size() > 1) {
        size_t N = target.extr_tmp.size();
        std::vector<float> first_intrinsic = target.intr_tmp[0];
        target.intr_tmp.resize(N, first_intrinsic);
    }
    
    // Statistics
    int totalImages = original_img_list_size;
    int processedImages = 0;
    int totalFacesDetected = 0;
    int validFacesSaved = 0;
    
    OH_LOG_INFO(LOG_APP, "Starting to process %{public}d images (masked=%{public}s)", 
               totalImages, masked ? "true" : "false");
    
    // Fixed crop size following createEnlargedFace
    cv::Size cropSize(480, 640);
    
    // Process each image in the original list
    for (size_t imgIdx = 0; imgIdx < original_img_list_size; ++imgIdx) {
        const std::string& imgPath = target.img_list[imgIdx];
        
        // Extract image name
        size_t lastSlash = imgPath.find_last_of('/');
        std::string imgName = (lastSlash != std::string::npos) ? 
                              imgPath.substr(lastSlash + 1) : imgPath;
        
        OH_LOG_INFO(LOG_APP, "Processing image [%{public}zu/%{public}d]: %{public}s", 
                   imgIdx + 1, totalImages, imgName.c_str());
        
        // Detect faces in current image
        std::vector<FaceResult> faceResults;
        inferenceFaceModel(imgPath, faceResults);
        
        // Load image
        cv::Mat frame = cv::imread(imgPath);
        if (frame.empty()) {
            OH_LOG_ERROR(LOG_APP, "Failed to read image: %{public}s", imgPath.c_str());
            continue;
        }
        
        processedImages++;
        totalFacesDetected += faceResults.size();
        
        if (faceResults.empty()) {
            OH_LOG_WARN(LOG_APP, "No faces detected in image: %{public}s", imgName.c_str());
            continue;
        }
        
        OH_LOG_INFO(LOG_APP, "Detected %{public}zu faces in %{public}s", 
                   faceResults.size(), imgName.c_str());
        
        // Load mask if needed
        cv::Mat maskImage;
        if (masked && imgIdx < original_mask_list_size) {
            maskImage = cv::imread(target.mask_list[imgIdx]);
            if (maskImage.empty()) {
                OH_LOG_WARN(LOG_APP, "Could not read mask: %{public}s", target.mask_list[imgIdx].c_str());
            }
        }
        
        // Process each detected face
        for (size_t faceIdx = 0; faceIdx < faceResults.size(); ++faceIdx) {
            const FaceResult& face = faceResults[faceIdx];
            cv::Rect box = face.box;
            
            // Ensure box is within image bounds
            cv::Rect safeBox(
                std::max(0, box.x),
                std::max(0, box.y),
                std::min(box.width, frame.cols - box.x),
                std::min(box.height, frame.rows - box.y)
            );
            
            // Filter out small faces (height < 320)
            if (safeBox.height < 100) {
                OH_LOG_INFO(LOG_APP, "Skipping small face in %{public}s: height=%{public}d (threshold=320)", 
                           imgName.c_str(), safeBox.height);
                continue;
            }
            
            // Calculate face center
            cv::Point faceCenter(
                safeBox.x + safeBox.width / 2,
                safeBox.y + safeBox.height / 2
            );
            
            // Crop image to fixed 480x640 size centered on face
            cv::Point topLeft;
            cv::Mat cropped_img_480x640 = cropImage(frame, faceCenter, cropSize, topLeft);
            
            // Check if crop is valid
            if (cropped_img_480x640.empty()) {
                OH_LOG_WARN(LOG_APP, "Failed to crop face %{public}zu in image %{public}s", 
                           faceIdx, imgName.c_str());
                continue;
            }
            
            // Generate output filename
            std::string nameWithoutExt = imgName.substr(0, imgName.find_last_of('.'));
            std::string extension = imgName.substr(imgName.find_last_of('.'));
            std::string outputName = nameWithoutExt + "_face_" + std::to_string(faceIdx) + extension;
            std::string outputPath = facesDir + "/" + outputName;
            
            // Save cropped face
            bool success = cv::imwrite(outputPath, cropped_img_480x640);
            if (!success) {
                OH_LOG_ERROR(LOG_APP, "Failed to save cropped face: %{public}s", outputPath.c_str());
                continue;
            }
            
            OH_LOG_INFO(LOG_APP, "Saved face crop %{public}zu to: %{public}s (size: 480x640, confidence: %{public}f)", 
                       faceIdx, outputPath.c_str(), face.confidence);
            
            // Append cropped image path to img_list
            target.img_list.push_back(outputPath);
            
            // Process mask if needed
            if (masked) {
                if (!maskImage.empty()) {
                    // Crop mask with same parameters
                    cv::Mat cropped_mask_480x640 = cropImage(maskImage, faceCenter, cropSize, topLeft);
                    
                    if (!cropped_mask_480x640.empty()) {
                        std::string maskOutputPath = segmentedDir + "/" + outputName;
                        bool maskSuccess = cv::imwrite(maskOutputPath, cropped_mask_480x640);
                        
                        if (maskSuccess) {
                            target.mask_list.push_back(maskOutputPath);
                            OH_LOG_INFO(LOG_APP, "Saved cropped mask to: %{public}s", maskOutputPath.c_str());
                        } else {
                            OH_LOG_WARN(LOG_APP, "Failed to save cropped mask: %{public}s", maskOutputPath.c_str());
                            target.mask_list.push_back(""); // Maintain alignment
                        }
                    } else {
                        OH_LOG_WARN(LOG_APP, "Failed to crop mask for face %{public}zu", faceIdx);
                        target.mask_list.push_back(""); // Maintain alignment
                    }
                } else {
                    // No mask available for this image
                    target.mask_list.push_back(""); // Maintain alignment
                }
            }
            
            // Calculate new intrinsics for the 480x640 crop
            std::vector<float> origIntr;
            if (original_intr_tmp_size == 1) {
                // Shared intrinsics for all images
                origIntr = target.intr_tmp[0];
            } else if (imgIdx < original_intr_tmp_size) {
                // Per-image intrinsics
                origIntr = target.intr_tmp[imgIdx];
            } else {
                OH_LOG_ERROR(LOG_APP, "No intrinsics available for image index %{public}zu", imgIdx);
                // Remove appended data to maintain consistency
                target.img_list.pop_back();
                if (masked && !target.mask_list.empty()) {
                    target.mask_list.pop_back();
                }
                continue;
            }
            
            // Calculate new intrinsics for the fixed 480x640 crop
            std::vector<float> newIntr = calculateNewIntrinsics(
                origIntr[0], origIntr[1], origIntr[2], origIntr[3],
                topLeft, cropSize, RATIO
            );
            
            OH_LOG_INFO(LOG_APP, "Face %{public}zu in %{public}s - New intrinsics: fx=%{public}f, fy=%{public}f, cx=%{public}f, cy=%{public}f",
                       faceIdx, imgName.c_str(), newIntr[0], newIntr[1], newIntr[2], newIntr[3]);
            
            // Append new intrinsics
            target.intr_tmp.push_back(newIntr);
            
            // Append corresponding extrinsics from original image
            if (imgIdx < original_extr_tmp_size) {
                target.extr_tmp.push_back(target.extr_tmp[imgIdx]);
            } else {
                OH_LOG_ERROR(LOG_APP, "No extrinsics available for image index %{public}zu", imgIdx);
                // Remove appended data to maintain consistency
                target.img_list.pop_back();
                if (masked && !target.mask_list.empty()) {
                    target.mask_list.pop_back();
                }
                target.intr_tmp.pop_back();
                continue;
            }
            
            // Append corresponding timestamp from original image
            if (imgIdx < original_time_stamps_size) {
                target.time_stamps.push_back(target.time_stamps[imgIdx]);
            } else {
                OH_LOG_WARN(LOG_APP, "No timestamp available for image index %{public}zu, using default", imgIdx);
                target.time_stamps.push_back(0.0); // Default timestamp
            }
            
            validFacesSaved++;
        }
    }
    
    // Update extr_tmp_test following createEnlargedFace pattern
    target.extr_tmp_test = target.extr_tmp;
    
    // Unload model
    HIAIModelManager::GetInstance().UnloadModel();
    
    // Log summary statistics
    OH_LOG_INFO(LOG_APP, "========================================");
    OH_LOG_INFO(LOG_APP, "Face Detection Summary:");
    OH_LOG_INFO(LOG_APP, "Total images processed: %{public}d", processedImages);
    OH_LOG_INFO(LOG_APP, "Total faces detected: %{public}d", totalFacesDetected);
    OH_LOG_INFO(LOG_APP, "Valid faces saved (height >= 320): %{public}d", validFacesSaved);
    OH_LOG_INFO(LOG_APP, "New total images in list: %{public}zu (added %{public}d face crops)", 
               target.img_list.size(), validFacesSaved);
    if (masked) {
        OH_LOG_INFO(LOG_APP, "New total masks in list: %{public}zu", target.mask_list.size());
    }
    OH_LOG_INFO(LOG_APP, "Crop size: 480x640 (fixed)");
    OH_LOG_INFO(LOG_APP, "Output directories: %{public}s", facesDir.c_str());
    if (masked) {
        OH_LOG_INFO(LOG_APP, "                    %{public}s", segmentedDir.c_str());
    }
    OH_LOG_INFO(LOG_APP, "========================================");
}