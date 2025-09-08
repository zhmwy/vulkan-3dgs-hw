#include "keypoints.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;

// Structure to hold person tracking information
struct PersonTracker {
    int id;
    Ptr<TrackerMIL> tracker;  // Using MIL tracker
    Rect2d lastBox;
    float avgConfidence;
    int framesSinceLastDetection;
    vector<Point2f> recentPositions; // Store recent positions for trajectory
    bool isActive;
};

// Validate and adjust bounding box to be within image boundaries
Rect2d validateBoundingBox(const Rect& box, int imgWidth, int imgHeight) {
    Rect2d validBox;
    validBox.x = max(0.0, (double)box.x);
    validBox.y = max(0.0, (double)box.y);
    validBox.width = min((double)box.width, imgWidth - validBox.x - 1);
    validBox.height = min((double)box.height, imgHeight - validBox.y - 1);
    
    // Ensure minimum size (MIL tracker needs sufficient pixels)
    const double MIN_SIZE = 20.0;
    if (validBox.width < MIN_SIZE) validBox.width = MIN_SIZE;
    if (validBox.height < MIN_SIZE) validBox.height = MIN_SIZE;
    
    // Final boundary check
    if (validBox.x + validBox.width > imgWidth) {
        validBox.x = imgWidth - validBox.width;
    }
    if (validBox.y + validBox.height > imgHeight) {
        validBox.y = imgHeight - validBox.height;
    }
    
    return validBox;
}

// Calculate IoU (Intersection over Union) for matching detections with trackers
float calculateIoU(const Rect& box1, const Rect2d& box2) {
    float x1 = max(box1.x, (int)box2.x);
    float y1 = max(box1.y, (int)box2.y);
    float x2 = min(box1.x + box1.width, (int)(box2.x + box2.width));
    float y2 = min(box1.y + box1.height, (int)(box2.y + box2.height));
    
    float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float unionArea = area1 + area2 - intersection;
    
    return (unionArea > 0) ? (intersection / unionArea) : 0;
}

// Calculate distance between two points
float calculateDistance(const Point2f& p1, const Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Match detections to existing trackers using IoU and distance metrics
vector<int> matchDetectionsToTrackers(const vector<OutputPose>& detections, 
                                      vector<PersonTracker>& trackers,
                                      float iouThreshold = 0.3,
                                      float distanceThreshold = 100.0) {
    vector<int> assignments(detections.size(), -1);
    vector<bool> trackerUsed(trackers.size(), false);
    
    // Create cost matrix based on IoU and distance
    vector<vector<float>> costMatrix(detections.size(), vector<float>(trackers.size(), 0));
    
    for (size_t i = 0; i < detections.size(); i++) {
        Point2f detCenter(detections[i].box.x + detections[i].box.width / 2.0,
                         detections[i].box.y + detections[i].box.height / 2.0);
        
        for (size_t j = 0; j < trackers.size(); j++) {
            if (!trackers[j].isActive) continue;
            
            // Calculate IoU
            float iou = calculateIoU(detections[i].box, trackers[j].lastBox);
            
            // Calculate center distance
            Point2f trackCenter(trackers[j].lastBox.x + trackers[j].lastBox.width / 2.0,
                               trackers[j].lastBox.y + trackers[j].lastBox.height / 2.0);
            float distance = calculateDistance(detCenter, trackCenter);
            
            // Combine metrics (prioritize IoU, use distance as secondary)
            if (iou > 0) {
                costMatrix[i][j] = iou;
            } else if (distance < distanceThreshold) {
                // Use normalized distance as cost when no IoU overlap
                costMatrix[i][j] = max(0.0f, 1.0f - distance / distanceThreshold) * 0.5f;
            }
        }
    }
    
    // Greedy assignment based on highest score
    for (size_t i = 0; i < detections.size(); i++) {
        float maxScore = iouThreshold;
        int bestTracker = -1;
        
        for (size_t j = 0; j < trackers.size(); j++) {
            if (!trackerUsed[j] && trackers[j].isActive && costMatrix[i][j] > maxScore) {
                maxScore = costMatrix[i][j];
                bestTracker = j;
            }
        }
        
        if (bestTracker != -1) {
            assignments[i] = trackers[bestTracker].id;
            trackerUsed[bestTracker] = true;
            
            // Update tracker's recent positions for trajectory analysis
            Point2f center(detections[i].box.x + detections[i].box.width / 2.0,
                          detections[i].box.y + detections[i].box.height / 2.0);
            trackers[bestTracker].recentPositions.push_back(center);
            if (trackers[bestTracker].recentPositions.size() > 10) {
                trackers[bestTracker].recentPositions.erase(trackers[bestTracker].recentPositions.begin());
            }
            
            trackers[bestTracker].framesSinceLastDetection = 0;
        }
    }
    
    // Mark unused trackers
    for (size_t j = 0; j < trackers.size(); j++) {
        if (!trackerUsed[j] && trackers[j].isActive) {
            trackers[j].framesSinceLastDetection++;
        }
    }
    
    return assignments;
}

// 判断人体是否居中 - Modified to check based on mean of all persons
bool isHumanCentered(const vector<Rect>& boxes, int imgWidth=480, int imgHeight=640) {
    if (boxes.empty()) {
        return false;
    }
    
    // Calculate mean center point and mean dimensions
    float meanCenterX = 0, meanCenterY = 0;
    float meanWidth = 0, meanHeight = 0;
    
    for (const auto& box : boxes) {
        meanCenterX += (box.x + box.width / 2.0);
        meanCenterY += (box.y + box.height / 2.0);
        meanWidth += box.width;
        meanHeight += box.height;
    }
    
    meanCenterX /= boxes.size();
    meanCenterY /= boxes.size();
    meanWidth /= boxes.size();
    meanHeight /= boxes.size();
    
    // Calculate the center region (50% of the image dimensions)
    int centerX = imgWidth / 2;
    int centerY = imgHeight / 2;
    int centerWidth = imgWidth / 2;
    int centerHeight = imgHeight / 2;

    // Check if the mean bounding box is within the center region
    bool isCentered = (meanCenterX >= centerX - centerWidth / 2) &&
                      (meanCenterX <= centerX + centerWidth / 2) &&
                      (meanCenterY >= centerY - centerHeight / 2) &&
                      (meanCenterY <= centerY + centerHeight / 2);

    // Check if the mean height is larger than 30% of the image height
    bool isHeightValid = meanHeight > 0.3 * imgHeight;

    return isCentered && isHeightValid;
}

void inferenceModel(std::string& imgPath, std::vector<OutputPose>& result)
{
    vector<Vec4d> params;
    LoadImage(imgPath, params);
    RunModel();
    PostProcessing(HIAIModelManager::GetInstance().GetResult(), params, result);
}

int ProcessKeypoints(const std::string& inputPath) {
    // Construct model path
    auto begin = clk::now();
    int totalProcessedImages = 0;
    
    size_t lastSlash = inputPath.find_last_of('/');
    std::string modelPath = inputPath.substr(0, lastSlash) + "/weights/yolov8-pose.om";
    
    // Multi-person tracking data structures
    vector<PersonTracker> trackers;
    int nextPersonID = 0;
    const int MAX_FRAMES_TO_SKIP = 5; // Maximum frames a tracker can miss before removal
    
    // Data storage per person ID
    map<int, vector<string>> keypoints_outputs_map;
    map<int, map<int, vector<float>>> keypoints_dict_per_person_map;
    map<int, vector<int>> keypoint_count_per_person_map;
    int human_centered_count = 0;
    
    if (LoadModel(modelPath) != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "Model loading failed");
        return 0;
    }

    // Process images
    std::string imgDir = inputPath + "/images/";
    vector<string> image_paths;
    // 收集所有 .jpg 和 .png 图片
    glob(imgDir + "/*.jpg", image_paths);
    vector<string> png_paths;
    glob(imgDir + "/*.png", png_paths);
    image_paths.insert(image_paths.end(), png_paths.begin(), png_paths.end());
    
    // Extract filenames from paths
    for(auto& path : image_paths) {
        size_t last_slash = path.find_last_of("/\\");
        if(last_slash != string::npos) {
            path = path.substr(last_slash + 1);
        }
    }

    srand(time(NULL));//随机
    
    vector<int> selectedIndices;
    vector<string> sampled_image_paths;

    if (image_paths.size() > 30) {
        double interval = static_cast<double>(image_paths.size()) / 30.0;
        for (int i = 0; i < 30; ++i) {
            int index = static_cast<int>(i * interval);
            selectedIndices.push_back(index);
        }
        for (int idx : selectedIndices) {
            sampled_image_paths.push_back(image_paths[idx]);
        }
    } else {
        sampled_image_paths = image_paths;  // 少于 30 张时，直接使用全部图片
        for (int i = 0; i < image_paths.size(); ++i) {
            selectedIndices.push_back(i);
        }
    }
    
    int frame_all = sampled_image_paths.size();
    int id = 0;
    
    // Process each frame
    for (const auto& imgName : sampled_image_paths) {
        int frameIdx = selectedIndices[id];
        id++;
        
        std::vector<OutputPose> result;
        std::string imgPath = imgDir + imgName;
        
        // Load image for tracking
        Mat frame = imread(imgPath);
        if (frame.empty()) {
            LOGE("Failed to load image: %{public}s", imgPath.c_str());
            continue;
        }
        
        int imgWidth = frame.cols;
        int imgHeight = frame.rows;
        
        // Run inference
        inferenceModel(imgPath, result);
        ++totalProcessedImages;
        
        // Update existing trackers
        for (auto& tracker : trackers) {
            if (tracker.isActive && tracker.tracker) {
                Rect2i box = tracker.lastBox;
                
                // Validate box before update
                box = validateBoundingBox(Rect(box.x, box.y, box.width, box.height), imgWidth, imgHeight);
                
                try {
                    bool success = tracker.tracker->update(frame, box);
                    if (success) {
                        tracker.lastBox = box;
                    } else {
                        // Tracker failed, increase missed frames count
                        tracker.framesSinceLastDetection++;
                    }
                } catch (...) {
                    // If tracking fails, mark tracker as having missed a frame
                    tracker.framesSinceLastDetection++;
                    LOGW("Tracker update failed for person ID %{public}d", tracker.id);
                }
            }
        }
        
        if (result.size() > 0) {
            // Collect all boxes for centering check
            vector<Rect> allBoxes;
            for (const auto& pose : result) {
                allBoxes.push_back(pose.box);
            }
            
            // Check if persons (as a group) are centered
            if (isHumanCentered(allBoxes, ORIG_WIDTH, ORIG_HEIGHT)) {
                human_centered_count++;
            }
            
            // Match detections to existing trackers
            vector<int> assignments = matchDetectionsToTrackers(result, trackers);
            
            // Process each detected person
            for (size_t i = 0; i < result.size(); i++) {
                const auto& pose = result[i];
                int personID = assignments[i];
                
                // Validate bounding box
                Rect2d validBox = validateBoundingBox(pose.box, imgWidth, imgHeight);
                
                if (personID == -1) {
                    // New person detected - create new tracker
                    PersonTracker newTracker;
                    newTracker.id = nextPersonID++;
                    newTracker.tracker = TrackerMIL::create();  // Create MIL tracker
                    newTracker.lastBox = validBox;
                    newTracker.avgConfidence = 0.0;
                    newTracker.framesSinceLastDetection = 0;
                    newTracker.isActive = true;
                    
                    // Initialize tracker with error handling
                    try {
                        newTracker.tracker->init(frame, validBox);
                    } catch (const cv::Exception& e) {
                        LOGW("Exception initializing tracker for person ID %{public}d: %{public}s", 
                             newTracker.id, e.what());
                        newTracker.isActive = false;
                    }
                    
                    Point2f center(validBox.x + validBox.width / 2.0, 
                                  validBox.y + validBox.height / 2.0);
                    newTracker.recentPositions.push_back(center);
                    
                    trackers.push_back(newTracker);
                    personID = newTracker.id;
                    
                    // Initialize data structures for new person
                    keypoints_outputs_map[personID] = vector<string>();
                    keypoints_dict_per_person_map[personID] = map<int, vector<float>>();
                    keypoint_count_per_person_map[personID] = vector<int>(17, 0);
                    
                    LOGI("New person detected with ID: %{public}d", personID);
                } else {
                    // Update existing tracker with new detection
                    for (auto& tracker : trackers) {
                        if (tracker.id == personID) {
                            // Re-initialize tracker with new detection for better accuracy
                            tracker.lastBox = validBox;
                            tracker.tracker = TrackerMIL::create();  // Create new MIL tracker
                            
                            try {
                                tracker.tracker->init(frame, validBox);
                                tracker.isActive = true;
                            } catch (const cv::Exception& e) {
                                LOGW("Exception re-initializing tracker for person ID %{public}d: %{public}s", 
                                     tracker.id, e.what());
                                tracker.isActive = false;
                            }
                            
                            tracker.framesSinceLastDetection = 0;
                            break;
                        }
                    }
                }
                
                // Save detection results for this person (regardless of tracker status)
                keypoints_dict_per_person_map[personID][frameIdx] = pose.kps;
                
                // Generate keypoints data string
                string data = imgName + "\n";
                data += to_string(pose.box.x / RATIO) + " " + to_string(pose.box.y / RATIO) + " " + 
                        to_string(pose.box.width / RATIO) + " " + to_string(pose.box.height / RATIO) + "\n";
                        
                for (size_t k = 0; k < pose.kps.size(); k += 3) {
                    float x = pose.kps[k] / RATIO;        
                    float y = pose.kps[k + 1] / RATIO;
                    float conf = pose.kps[k + 2];
                    if (conf >= 0.75) {
                        keypoint_count_per_person_map[personID][k / 3]++;
                    }
                    data += to_string(x) + " " + to_string(y) + " " + to_string(conf) + "\n";
                }
                keypoints_outputs_map[personID].push_back(data);
            }
            
            // Deactivate trackers that haven't been detected for too long
            for (auto& tracker : trackers) {
                if (tracker.framesSinceLastDetection > MAX_FRAMES_TO_SKIP) {
                    tracker.isActive = false;
                }
            }
        }
        else {
            LOGE("No persons detected in %{public}s", imgName.c_str());
            
            // Update all trackers even when no detection
            for (auto& tracker : trackers) {
                if (tracker.isActive) {
                    tracker.framesSinceLastDetection++;
                    if (tracker.framesSinceLastDetection > MAX_FRAMES_TO_SKIP) {
                        tracker.isActive = false;
                    }
                }
            }
        }
    }
    
    LOGW("Processed %{public}d images in total. It takes %{public}f seconds", totalProcessedImages, 
                                            std::chrono::duration<double>(clk::now() - begin).count());
    
    // Check overall centering percentage
    double percentage = (double)human_centered_count / frame_all;
    if (percentage < 0.7) {
        LOGI("Persons are not centered enough (%.2f%%), returning", percentage * 100);
        return 0;
    }
    
    // Process each tracked person
    int numValidPersons = 0;
    for (const auto& kv : keypoints_outputs_map) {
        if (!kv.second.empty()) {
            numValidPersons++;
        }
    }
    
    LOGI("Tracked %{public}d persons in total", numValidPersons);
    
    int outputPersonIndex = 1;
    for (const auto& personData : keypoints_outputs_map) {
        int personID = personData.first;
        
        // Skip persons with no data
        if (personData.second.empty()) {
            continue;
        }
        
        LOGI("Processing person ID %{public}d (output as person %{public}d)", personID, outputPersonIndex);
        
        // Create output file for this person
        std::string outputPath = inputPath + "/keypoints" + to_string(outputPersonIndex) + ".txt";
        
        // Check if any keypoint has less than 2 frames for this person
        bool hasLessThanTwo = false;
        for (int i = 0; i < 17; ++i) {
            if (keypoint_count_per_person_map[personID][i] < 2) {
                hasLessThanTwo = true;
                break;
            }
        }
        
        if (hasLessThanTwo) {
            LOGI("Person %{public}d: Some keypoints have less than 2 frames. Writing initial data.", outputPersonIndex);
            ofstream ofs(outputPath, ios::out);
            for (const auto& line : keypoints_outputs_map[personID]) {
                ofs << line;
            }
            ofs.close();
            outputPersonIndex++;
            continue;
        }
        
        // Find keypoints with insufficient detections for this person
        vector<int> few_keypt;
        for (int i = 0; i < 17; ++i) {
            if (keypoint_count_per_person_map[personID][i] < 2) {
                few_keypt.push_back(i);
            }
        }
        
        if (!few_keypt.empty()) {
            LOGI("Person %{public}d 缺", outputPersonIndex);
            for (int elem : few_keypt) {
                LOGI("%{public}d", elem);
            }
            LOGI("关键点");
            
            // Find candidate frames for missing keypoints
            vector<vector<int>> houxuan_id(17);
            
            for (int pt : few_keypt) {
                for (const auto& kv : keypoints_dict_per_person_map[personID]) {
                    int frameId = kv.first;
                    const vector<float>& kps = kv.second;
                    int idx = pt * 3 + 2;
                    if (idx >= kps.size())
                        continue;
                    float conf = kps[idx];
                    if (conf > 0.4f) {
                        if (find(houxuan_id[pt].begin(), houxuan_id[pt].end(), frameId) == houxuan_id[pt].end())
                            houxuan_id[pt].push_back(frameId);
                    }
                }
            }
            
            set<int> processedFrames;
            for (int frameId : selectedIndices) {
                processedFrames.insert(frameId);
            }
            
            int window_range = 3;
            
            for (int elem : few_keypt) {
                LOGI("Person %{public}d: 关键点 %{public}d 的候选帧:", outputPersonIndex, elem);
                for (size_t j = 0; j < houxuan_id[elem].size(); j++) {
                    LOGI("%{public}d ", houxuan_id[elem][j]);
                }
            }
            
            // Process missing keypoints for this person
            for (int pt : few_keypt) {
                if (keypoint_count_per_person_map[personID][pt] >= 2)
                    continue;
                    
                LOGI("Person %{public}d: 开始补充关键点 %{public}d 的候选帧，当前检测次数为 %{public}d", 
                     outputPersonIndex, pt, keypoint_count_per_person_map[personID][pt]);
                
                while (keypoint_count_per_person_map[personID][pt] < 2) {
                    if (houxuan_id[pt].empty()) {
                        int randomFrame;
                        do {
                            randomFrame = rand() % image_paths.size();
                        } while (processedFrames.find(randomFrame) != processedFrames.end());
                        
                        processedFrames.insert(randomFrame);
                        std::vector<OutputPose> result;
                        std::string fullPath = imgDir + image_paths[randomFrame];
                        
                        // Load frame for tracking
                        Mat frame = imread(fullPath);
                        if (!frame.empty()) {
                            inferenceModel(fullPath, result);
                            ++totalProcessedImages;
                            
                            // Try to find a person in this frame
                            if (!result.empty()) {
                                // Use the first detection as a fallback
                                const auto& pose = result[0];
                                keypoints_dict_per_person_map[personID][randomFrame] = pose.kps;
                                selectedIndices.push_back(randomFrame);
                                
                                int idx = pt * 3 + 2;
                                if (idx < pose.kps.size()) {
                                    float newConf = pose.kps[idx];
                                    LOGI("Person %{public}d: 随机帧 %{public}d 关键点 %{public}d 置信度: %{public}f", 
                                         outputPersonIndex, randomFrame, pt, newConf);
                                    if (newConf > 0.4f) {
                                        houxuan_id[pt].push_back(randomFrame);
                                        LOGI("Person %{public}d: 随机帧 %{public}d 置信度合格，加入候选列表", 
                                             outputPersonIndex, randomFrame);
                                    }
                                }
                            }
                        }
                    }
                    else {
                        for (int candFrame : houxuan_id[pt]) {
                            int startFrame = max(0, candFrame - window_range);
                            int endFrame = min((int)image_paths.size() - 1, candFrame + window_range);
                            for (int neighbor = startFrame; neighbor <= endFrame && keypoint_count_per_person_map[personID][pt] < 2; neighbor++) {
                                if (processedFrames.find(neighbor) != processedFrames.end())
                                    continue;
                                    
                                std::vector<OutputPose> result;
                                std::string fullPath = imgDir + image_paths[neighbor];
                                
                                Mat frame = imread(fullPath);
                                if (!frame.empty()) {
                                    inferenceModel(fullPath, result);
                                    ++totalProcessedImages;
                                    
                                    processedFrames.insert(neighbor);
                                    selectedIndices.push_back(neighbor);
                                    
                                    // Find best matching detection for this person
                                    if (!result.empty()) {
                                        // Use the first detection as a simple fallback
                                        const auto& pose = result[0];
                                        keypoints_dict_per_person_map[personID][neighbor] = pose.kps;
                                        
                                        string file_name = image_paths[neighbor];
                                        string data = file_name + "\n";
                                        data += to_string(pose.box.x / RATIO) + " " + to_string(pose.box.y / RATIO) + " " + 
                                                to_string(pose.box.width / RATIO) + " " + to_string(pose.box.height / RATIO) + "\n";
                                        for (size_t k = 0; k < pose.kps.size(); k += 3) {
                                            float x = pose.kps[k] / RATIO;
                                            float y = pose.kps[k+1] / RATIO;
                                            float conf = pose.kps[k+2];
                                            data += to_string(x) + " " + to_string(y) + " " + to_string(conf) + "\n";
                                        }
                                        keypoints_outputs_map[personID].push_back(data);
                                        
                                        int idx = pt * 3 + 2;
                                        if (idx < pose.kps.size()) {
                                            float newConf = pose.kps[idx];
                                            if (newConf > 0.75f) {
                                                keypoint_count_per_person_map[personID][pt]++;
                                                LOGI("Person %{public}d: 候选帧%{public}d成功补充关键点%{public}d，新置信度为%{public}f，当前计数： %{public}d", 
                                                     outputPersonIndex, neighbor, pt, newConf, keypoint_count_per_person_map[personID][pt]);
                                                if (keypoint_count_per_person_map[personID][pt] >= 2) {
                                                    LOGI("Person %{public}d: 关键点%{public}d已达到2次有效检测，停止补充。", outputPersonIndex, pt);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (keypoint_count_per_person_map[personID][pt] >= 2)
                                break;
                        }
                        houxuan_id[pt].clear();
                    }
                }
            }
            LOGW("Processed %{public}d images in total. It takes %{public}f seconds", totalProcessedImages, 
                                                    std::chrono::duration<double>(clk::now() - begin).count());
        }
        
        // Save keypoints for this person
        ofstream ofs(outputPath, ios::out);
        for (const auto& line : keypoints_outputs_map[personID]) {
            LOGI("Person %{public}d keypoints written: %{public}s", outputPersonIndex, line.c_str());
            ofs << line;
        }
        ofs.close();
        LOGI("Person %{public}d keypoints saved to %{public}s", outputPersonIndex, outputPath.c_str());
        
        outputPersonIndex++;
    }
    
    HIAIModelManager::GetInstance().UnloadModel();

    return numValidPersons > 0 ? 1 : 0;
}