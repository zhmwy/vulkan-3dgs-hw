#include "human.h"

std::string getBaseName(const std::string& path){
    size_t pos = path.find_last_of('/');
    std::string basename = path.substr(pos + 1);
    return basename;
}

// Modified function to check face validity without requiring saved images
std::vector<FaceProcessingInfo> analyzeFaces(const std::map<std::string, double>& faceAngles,
                                            const std::vector<cv::Point>& faceCenters,
                                            const std::vector<std::string>& imgList) {
    std::vector<FaceProcessingInfo> faceInfos;
    
    // Build face processing info for each image
    for (int i = 0; i < imgList.size() && i < faceCenters.size(); ++i) {
        std::string basename = getBaseName(imgList[i]);
        
        // Check if this image has angle information
        auto angleIt = faceAngles.find(basename);
        if (angleIt == faceAngles.end()) {
            continue;
        }
        
        // Read image to get dimensions
        cv::Mat img = cv::imread(imgList[i]);
        if (img.empty()) {
            continue;
        }
        
        float scale_factor = img.size[0] / 640;
        cv::Point scaledFaceCenter = faceCenters[i] * scale_factor;
        
        // Check if face center is within bounds
        bool isWithinBounds = (scaledFaceCenter.x > 0 && 
                              scaledFaceCenter.y > 0 && 
                              scaledFaceCenter.x < img.cols && 
                              scaledFaceCenter.y < img.rows);
        
        FaceProcessingInfo info;
        info.imageName = basename;
        info.faceCenter = scaledFaceCenter;
        info.angle = angleIt->second;
        info.isWithinBounds = isWithinBounds;
        
        faceInfos.push_back(info);
    }
    
    return faceInfos;
}

// New function to select two images based on face info (without requiring saved files)
std::pair<int, int> selectTwoImageIndices(const std::vector<FaceProcessingInfo>& faceInfos) {
    std::pair<int, int> result(-1, -1);
    
    // Filter valid faces (within bounds and angle < 90)
    std::vector<std::pair<int, double>> validFaces;
    for (int i = 0; i < faceInfos.size(); ++i) {
        if (faceInfos[i].isWithinBounds && faceInfos[i].angle < 90.0) {
            validFaces.push_back({i, faceInfos[i].angle});
        }
    }
    
    if (validFaces.empty()) {
        return result;
    }
    
    // Sort by angle
    std::sort(validFaces.begin(), validFaces.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // First image is the one with smallest angle
    result.first = validFaces[0].first;
    double smallestAngle = validFaces[0].second;
    
    // Find second image closest to target angle
    double targetAngle = smallestAngle + 30.0;
    double minDifference = std::numeric_limits<double>::max();
    
    for (int i = 1; i < validFaces.size(); ++i) {
        double difference = std::abs(validFaces[i].second - targetAngle);
        if (difference < minDifference) {
            minDifference = difference;
            result.second = validFaces[i].first;
        }
    }
    
    return result;
}

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

bool HumanPreprocess(std::string inputPath, std::vector<float>& head_bbox, 
                    std::vector<float>& body_bbox, std::map<std::string, double>& faceAngles) {
    // Execute core logic
    auto begin = clk::now();
    
    int isHumanCentric = ProcessKeypoints(inputPath);
    if(isHumanCentric){
        OH_LOG_WARN(LOG_APP, "Dataset is human centric. Default to 3DGS on human.");
        faceAngles = ProcessTriangulation(inputPath);
        FitSMPL(inputPath, head_bbox, body_bbox);
        OH_LOG_INFO(LOG_APP, "SMPL estimation finished.");
                
        LOGW("Human preprocessing takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        return true;
    }
    else{
        OH_LOG_WARN(LOG_APP, "Dataset is not human centric. Default to 3DGS on scene.");
        return false;
    }
}