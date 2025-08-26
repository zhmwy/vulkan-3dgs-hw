#include "keypoints.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;


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

// Helper function to find which person ID a detection corresponds to based on x-coordinate
int findPersonID(float x, const vector<float>& personXPositions, float threshold = 50.0) {
    for (size_t i = 0; i < personXPositions.size(); i++) {
        if (abs(x - personXPositions[i]) < threshold) {
            return i;
        }
    }
    return -1; // New person
}

int ProcessKeypoints(const std::string& inputPath) {
    // Construct model path
    auto begin = clk::now();
    int totalProcessedImages = 0;
    
    size_t lastSlash = inputPath.find_last_of('/');
    std::string modelPath = inputPath.substr(0, lastSlash) + "/weights/yolov8-pose.om";
    
    // Multi-person data structures
    vector<float> personXPositions; // Store average x positions for each person
    vector<vector<string>> keypoints_outputs; // One vector per person
    vector<map<int, vector<float>>> keypoints_dict_per_person; // One map per person
    vector<vector<int>> keypoint_count_per_person; // Keypoint counts per person
    int human_centered_count = 0; // Single counter for all persons together
    
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
    
    // First pass: identify persons and their positions
    for (const auto& imgName : sampled_image_paths) {
        int idx = selectedIndices[id];
        id++;
        std::vector<OutputPose> result;
        std::string imgPath = imgDir + imgName;
        inferenceModel(imgPath, result);
        ++totalProcessedImages;
        
        if (result.size() > 0) {
            // Sort by x-coordinate for consistent person tracking
            sort(result.begin(), result.end(), [](const OutputPose &a, const OutputPose &b) {
                return a.box.x < b.box.x;
            });
            
            // Collect all boxes for centering check
            vector<Rect> allBoxes;
            for (const auto& pose : result) {
                allBoxes.push_back(pose.box);
            }
            
            // Check if persons (as a group) are centered
            if (isHumanCentered(allBoxes, ORIG_WIDTH, ORIG_HEIGHT)) {
                human_centered_count++;
            }
            
            // Process each detected person
            for (const auto& pose : result) {
                float centerX = pose.box.x + pose.box.width / 2.0;
                int personID = findPersonID(centerX, personXPositions);
                
                if (personID == -1) {
                    // New person detected
                    personXPositions.push_back(centerX);
                    personID = personXPositions.size() - 1;
                    
                    // Initialize data structures for new person
                    keypoints_outputs.push_back(vector<string>());
                    keypoints_dict_per_person.push_back(map<int, vector<float>>());
                    keypoint_count_per_person.push_back(vector<int>(17, 0));
                } else {
                    // Update average position (running average)
                    personXPositions[personID] = (personXPositions[personID] + centerX) / 2.0;
                }
                
                // Save detection results for this person
                keypoints_dict_per_person[personID][idx] = pose.kps;
                
                // Generate keypoints data string
                string data = imgName + "\n";
                data += to_string(pose.box.x / RATIO) + " " + to_string(pose.box.y / RATIO) + " " + 
                        to_string(pose.box.width / RATIO) + " " + to_string(pose.box.height / RATIO) + "\n";
                        
                for (size_t k = 0; k < pose.kps.size(); k += 3) {
                    float x = pose.kps[k] / RATIO;        
                    float y = pose.kps[k + 1] / RATIO;
                    float conf = pose.kps[k + 2];
                    if (conf >= 0.75) {
                        keypoint_count_per_person[personID][k / 3]++;
                    }
                    data += to_string(x) + " " + to_string(y) + " " + to_string(conf) + "\n";
                }
                keypoints_outputs[personID].push_back(data);
            }
        }
        else {
            LOGE("No persons detected in %{public}s", imgName.c_str());
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
    
    // Process each person separately
    int numPersons = personXPositions.size();
    LOGI("Detected %{public}d persons in total", numPersons);
    
    for (int personID = 0; personID < numPersons; personID++) {
        LOGI("Processing person %{public}d", personID + 1);
        
        // Create output file for this person
        std::string outputPath = inputPath + "/keypoints" + to_string(personID + 1) + ".txt";
        
        // Check if any keypoint has less than 2 frames for this person
        bool hasLessThanTwo = false;
        for (int i = 0; i < 17; ++i) {
            if (keypoint_count_per_person[personID][i] < 2) {
                hasLessThanTwo = true;
                break;
            }
        }
        
        if (hasLessThanTwo) {
            LOGI("Person %{public}d: Some keypoints have less than 2 frames. Writing initial data.", personID + 1);
            ofstream ofs(outputPath, ios::out);
            for (const auto& line : keypoints_outputs[personID]) {
                ofs << line;
            }
            ofs.close();
            continue; // Move to next person
        }
        
        // Find keypoints with insufficient detections for this person
        vector<int> few_keypt;
        for (int i = 0; i < 17; ++i) {
            if (keypoint_count_per_person[personID][i] < 2) {
                few_keypt.push_back(i);
            }
        }
        
        if (!few_keypt.empty()) {
            LOGI("Person %{public}d 缺", personID + 1);
            for (int elem : few_keypt) {
                LOGI("%{public}d", elem);
            }
            LOGI("关键点");
            
            // Find candidate frames for missing keypoints
            vector<vector<int>> houxuan_id(17);
            
            for (int pt : few_keypt) {
                for (const auto& kv : keypoints_dict_per_person[personID]) {
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
                LOGI("Person %{public}d: 关键点 %{public}d 的候选帧:", personID + 1, elem);
                for (size_t j = 0; j < houxuan_id[elem].size(); j++) {
                    LOGI("%{public}d ", houxuan_id[elem][j]);
                }
            }
            
            // Process missing keypoints for this person
            for (int pt : few_keypt) {
                if (keypoint_count_per_person[personID][pt] >= 2)
                    continue;
                    
                LOGI("Person %{public}d: 开始补充关键点 %{public}d 的候选帧，当前检测次数为 %{public}d", 
                     personID + 1, pt, keypoint_count_per_person[personID][pt]);
                
                while (keypoint_count_per_person[personID][pt] < 2) {
                    if (houxuan_id[pt].empty()) {
                        int randomFrame;
                        do {
                            randomFrame = rand() % image_paths.size();
                        } while (processedFrames.find(randomFrame) != processedFrames.end());
                        
                        processedFrames.insert(randomFrame);
                        std::vector<OutputPose> result;
                        std::string fullPath = imgDir + image_paths[randomFrame];
                        inferenceModel(fullPath, result);
                        ++totalProcessedImages;
                        
                        // Find the person in this frame
                        sort(result.begin(), result.end(), [](const OutputPose &a, const OutputPose &b) {
                            return a.box.x < b.box.x;
                        });
                        
                        for (const auto& pose : result) {
                            float centerX = pose.box.x + pose.box.width / 2.0;
                            if (findPersonID(centerX, vector<float>{personXPositions[personID]}) == 0) {
                                keypoints_dict_per_person[personID][randomFrame] = pose.kps;
                                selectedIndices.push_back(randomFrame);
                                
                                int idx = pt * 3 + 2;
                                if (idx >= pose.kps.size())
                                    continue;
                                float newConf = pose.kps[idx];
                                LOGI("Person %{public}d: 随机帧 %{public}d 关键点 %{public}d 置信度: %{public}f", 
                                     personID + 1, randomFrame, pt, newConf);
                                if (newConf > 0.4f) {
                                    houxuan_id[pt].push_back(randomFrame);
                                    LOGI("Person %{public}d: 随机帧 %{public}d 置信度合格，加入候选列表", 
                                         personID + 1, randomFrame);
                                }
                                break;
                            }
                        }
                    }
                    else {
                        for (int candFrame : houxuan_id[pt]) {
                            int startFrame = max(0, candFrame - window_range);
                            int endFrame = min((int)image_paths.size() - 1, candFrame + window_range);
                            for (int neighbor = startFrame; neighbor <= endFrame && keypoint_count_per_person[personID][pt] < 2; neighbor++) {
                                if (processedFrames.find(neighbor) != processedFrames.end())
                                    continue;
                                    
                                std::vector<OutputPose> result;
                                std::string fullPath = imgDir + image_paths[neighbor];
                                inferenceModel(fullPath, result);
                                ++totalProcessedImages;
                                
                                processedFrames.insert(neighbor);
                                selectedIndices.push_back(neighbor);
                                
                                // Find the person in this frame
                                sort(result.begin(), result.end(), [](const OutputPose &a, const OutputPose &b) {
                                    return a.box.x < b.box.x;
                                });
                                
                                for (const auto& pose : result) {
                                    float centerX = pose.box.x + pose.box.width / 2.0;
                                    if (findPersonID(centerX, vector<float>{personXPositions[personID]}) == 0) {
                                        keypoints_dict_per_person[personID][neighbor] = pose.kps;
                                        
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
                                        keypoints_outputs[personID].push_back(data);
                                        
                                        int idx = pt * 3 + 2;
                                        if (idx >= pose.kps.size())
                                            continue;
                                        float newConf = pose.kps[idx];
                                        if (newConf > 0.75f) {
                                            keypoint_count_per_person[personID][pt]++;
                                            LOGI("Person %{public}d: 候选帧%{public}d成功补充关键点%{public}d，新置信度为%{public}f，当前计数： %{public}d", 
                                                 personID + 1, neighbor, pt, newConf, keypoint_count_per_person[personID][pt]);
                                            if (keypoint_count_per_person[personID][pt] >= 2) {
                                                LOGI("Person %{public}d: 关键点%{public}d已达到2次有效检测，停止补充。", personID + 1, pt);
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                            if (keypoint_count_per_person[personID][pt] >= 2)
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
        for (const auto& line : keypoints_outputs[personID]) {
            LOGI("Person %{public}d keypoints written: %{public}s", personID + 1, line.c_str());
            ofs << line;
        }
        ofs.close();
        LOGI("Person %{public}d keypoints saved to %{public}s", personID + 1, outputPath.c_str());
    }
    
    HIAIModelManager::GetInstance().UnloadModel();

    return numPersons > 0 ? 1 : 0;
}