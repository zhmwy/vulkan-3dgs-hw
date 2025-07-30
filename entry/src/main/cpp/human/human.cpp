#include "human.h"



std::vector<std::string> selectTwoImages(const std::map<std::string, double>& faceAngles, 
                                         const std::string& folderPath) {
    std::vector<std::string> result;
    
    // Check if the map is empty
    if (faceAngles.empty()) {
        return result;
    }
    
    // Create a vector of pairs for easier sorting and filtering
    std::vector<std::pair<std::string, double>> validImages;
    
    // Filter images that exist in the folder and have angle < 90
    for (const auto& [imageName, angle] : faceAngles) {
        if (angle < 90.0) {
            // Construct the full path to check if file exists
            std::filesystem::path imagePath = std::filesystem::path(folderPath) / imageName;
            
            if (std::filesystem::exists(imagePath)) {
                validImages.push_back({imageName, angle});
            }
        }
    }
    
    // If no valid images found, return empty vector
    if (validImages.empty()) {
        return result;
    }
    
    // Sort by angle to find the smallest
    std::sort(validImages.begin(), validImages.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // First image is the one with smallest angle
    std::string firstImage = validImages[0].first;
    double smallestAngle = validImages[0].second;
    
    // Target angle for the second image
    double targetAngle = smallestAngle + 30.0;
    
    // Find the second image with angle closest to targetAngle
    std::string secondImage;
    double minDifference = std::numeric_limits<double>::max();
    bool foundSecond = false;
    
    for (const auto& [imageName, angle] : validImages) {
        // Skip the first image
        if (imageName == firstImage) {
            continue;
        }
        
        double difference = std::abs(angle - targetAngle);
        if (difference < minDifference) {
            minDifference = difference;
            secondImage = imageName;
            foundSecond = true;
        }
    }
    
    // If we couldn't find a second image, return empty vector
    if (!foundSecond) {
        return result;
    }
    
    // Construct full paths for the selected images
    std::filesystem::path firstImagePath = std::filesystem::path(folderPath) / firstImage;
    std::filesystem::path secondImagePath = std::filesystem::path(folderPath) / secondImage;
    
    // Add the paths to the result vector
    result.push_back(firstImagePath.string());
    result.push_back(secondImagePath.string());
    
    return result;
}

bool HumanPreprocess(std::string scenepath, DataLoader::InputInitStruct& data) {
    // Execute core logic
    auto begin = clk::now();
    
    int isHumanCentric = ProcessKeypoints(scenepath);
    if(isHumanCentric){
        OH_LOG_WARN(LOG_APP, "Dataset is human centric. Default to 3DGS on human.");
        std::map<std::string, double> faceAngles = ProcessTriangulation(scenepath);
        
        FitSMPL(scenepath, data.head_bbox, data.body_bbox);
        OH_LOG_INFO(LOG_APP, "SMPL estimation finished.");
                    
        // Calculate face centers
        std::vector<cv::Point> faceCenters = projectHeadCenter(data);
            
        // Analyze faces without requiring saved images
        std::vector<FaceProcessingInfo> faceInfos = analyzeFaces(faceAngles, faceCenters, data.img_list);
            
        // Select two images based on analysis
        std::pair<int, int> selectedIndices = selectTwoImageIndices(faceInfos);
        
        // Detect face region
        detectFace(scenepath, faceInfos, selectedIndices);
        if (selectedIndices.first != -1 && selectedIndices.second != -1) {
            // Process faces and create enlarged faces
            data.max_movement = (data.body_bbox[3] - data.body_bbox[2]) * 0.01;
            createEnlargedFace(scenepath, data, faceInfos, selectedIndices);
            ProcessLandmarks(scenepath, selectedIndices, faceInfos);
        }
        
        LOGW("Human preprocessing takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        return true;
    }
    else{
        OH_LOG_WARN(LOG_APP, "Dataset is not human centric. Default to 3DGS on scene.");
        return false;
    }
}