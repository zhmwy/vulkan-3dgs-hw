//
// Created on 2025/7/30.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#include "landmark.h"
#include <map>

std::string getBaseName(const std::string& path){
    size_t pos = path.find_last_of('/');
    std::string basename = path.substr(pos + 1);
    return basename;
}

// Modified function to check face validity without requiring saved images
std::vector<FaceProcessingInfo> analyzeFaces(const std::map<std::string, std::vector<double>>& faceAngles,
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
        
        cv::Point scaledFaceCenter = faceCenters[i] * RATIO;
        
        // Check if face center is within bounds
        bool isWithinBounds = (scaledFaceCenter.x > 0 && 
                              scaledFaceCenter.y > 0 && 
                              scaledFaceCenter.x < ORIG_WIDTH && 
                              scaledFaceCenter.y < ORIG_HEIGHT);
        
        FaceProcessingInfo info;
        info.imageName = basename;
        info.faceCenter = scaledFaceCenter;
        info.angle = angleIt->second[0];
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