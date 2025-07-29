#include "keypoints.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

// Modified LoadModel to use direct file access
OH_NN_ReturnCode LoadModel(const std::string& modelPath) {
    FILE* modelFile = fopen(modelPath.c_str(), "rb");
    if (!modelFile) {
        OH_LOG_ERROR(LOG_APP, "Failed to open model file at %{public}s", modelPath.c_str());
        return OH_NN_FAILED;
    }

    fseek(modelFile, 0, SEEK_END);
    long modelSize = ftell(modelFile);
    fseek(modelFile, 0, SEEK_SET);

    std::unique_ptr<uint8_t[]> modelData = std::make_unique<uint8_t[]>(modelSize);
    size_t bytesRead = fread(modelData.get(), 1, modelSize, modelFile);
    fclose(modelFile);

    if (bytesRead != static_cast<size_t>(modelSize)) {
        OH_LOG_ERROR(LOG_APP, "Incomplete model read");
        return OH_NN_FAILED;
    }

    return HIAIModelManager::GetInstance().LoadModelFromBuffer(modelData.get(), modelSize);
}

// Function to run the model
OH_NN_ReturnCode RunModel() {
    OH_NN_ReturnCode ret = HIAIModelManager::GetInstance().RunModel();
    if (ret != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "RunModel failed");
        return ret;
    }

    OH_LOG_INFO(LOG_APP, "RunModel success");
    return ret;
}

// Modified LoadImage to use direct OpenCV loading
OH_NN_ReturnCode LoadImage(const std::string& imagePath, vector<Vec4d>& params) {
    cv::Mat frame = cv::imread(imagePath);
    
    if (frame.empty()) {
        OH_LOG_ERROR(LOG_APP, "Failed to read image: %s", imagePath.c_str());
        return OH_NN_FAILED;
    }

    vector<cv::Mat> srcs{frame};
    vector<cv::Mat> input_images;
    Preprocessing(srcs, input_images, params);

    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1/255.0, cv::Size(640, 640));
    std::vector<std::pair<float*, size_t>> inputDataList{
        {reinterpret_cast<float*>(blob.data), blob.total()}
    };

    return HIAIModelManager::GetInstance().InitIOTensors(inputDataList);
}

// 判断人体是否居中
bool isHumanCentered(const Rect& box, int imgWidth=480, int imgHeight=640) {
    // Calculate the center region (50% of the image dimensions)
    int centerX = imgWidth / 2;
    int centerY = imgHeight / 2;
    int centerWidth = imgWidth / 2;
    int centerHeight = imgHeight / 2;

    // Check if the bounding box is within the center region
    bool isCentered = (box.x + box.width / 2 >= centerX - centerWidth / 2) &&
                      (box.x + box.width / 2 <= centerX + centerWidth / 2) &&
                      (box.y + box.height / 2 >= centerY - centerHeight / 2) &&
                      (box.y + box.height / 2 <= centerY + centerHeight / 2);

    // Check if the height of the bounding box is larger than 30% of the image height
    bool isHeightValid = box.height > 0.3 * imgHeight;

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
//     size_t secondLastSlash = inputPath.substr(0, lastSlash).find_last_of('/');
    std::string modelPath = inputPath.substr(0, lastSlash) + "/weights/yolov8-pose.om";
    vector<vector<string>> keypoints_outputs;
    
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

    // Create output file
    std::string outputPath = inputPath + "/keypoints.txt";
    std::ofstream outfile(outputPath);
    if (!outfile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Failed to create output file: %s", outputPath.c_str());
        return 0;
    }

    srand(time(NULL));//随机
    
    vector<int> keypoint_count(17, 0); // 统计每个关键点的检测次数
    map<int, vector<float>> keypoints_dict; //keypoints_dict[selectedIndex] = 17组x,y,conf
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
    }
    
    int frame_all = sampled_image_paths.size();
    int human_centered_count = 0;

    int id = 0;
    cv::Mat imgSample = cv::imread(imgDir + sampled_image_paths[0]);
    int width = imgSample.cols;
    int height = imgSample.rows;
    float ratio = width / 480.0;
    assert(ratio == height/640.0);
    
    for (const auto& imgName : sampled_image_paths) {
        int idx = selectedIndices[id];
        id++;
        std::vector<OutputPose> result;
        std::string imgPath = imgDir + imgName;
        inferenceModel(imgPath, result);
        ++totalProcessedImages;
        if (result.size() > 0) {
            // 按检测框的 x 坐标排序
            sort(result.begin(), result.end(), [](const OutputPose &a, const OutputPose &b) {
                return a.box.width > b.box.width;
            });
    
            // 检查是否有检测结果在图片中心
            for (const auto& pose : result) {
                if (isHumanCentered(pose.box, width, height)) {
                    human_centered_count++;
                    break; // 每张图片只计一次
                }
            }

            // 确保 keypoints_outputs 数组大小足够
            if (keypoints_outputs.size() < result.size()) {
                keypoints_outputs.resize(result.size());
            }
            // 保存第一人的检测结果到 keypoints_dict
            if (!result.empty()) {
                keypoints_dict[idx] = result[0].kps;
            }
            // 累积关键点数据到 keypoints_outputs
            for (size_t i = 0; i < 1; ++i) {
                string data = imgName + "\n";
                data += to_string(result[0].box.x / ratio) + " " + to_string(result[0].box.y / ratio) + " " + 
                        to_string(result[0].box.width / ratio) + " " + to_string(result[0].box.height / ratio) + "\n";
                for (size_t k = 0; k < result[i].kps.size(); k += 3) {
                    float x = result[0].kps[k] / ratio;        
                    float y = result[0].kps[k + 1] / ratio;
                    float conf = result[0].kps[k + 2];
                    if (conf >= 0.75) {
                        keypoint_count[k / 3]++;
                    }
                    data += to_string(x) + " " + to_string(y) + " " + to_string(conf) + "\n";
                }
                keypoints_outputs[0].push_back(data);
            }
        }
        else {
            LOGE("No persons detected in %{public}s", imgName.c_str());
        }
    }
    LOGW("Processed %{public}d images in total. It takes %{public}f seconds", totalProcessedImages, 
                                            std::chrono::duration<double>(clk::now() - begin).count());
    
    double percentage = (double)human_centered_count / frame_all;
    if (percentage < 0.7) {
        return 0;
    }
    
    // Check if any keypoint has less than 2 frames
    bool hasLessThanTwo = false;
    for (int i = 0; i < 17; ++i) {
        if (keypoint_count[i] < 2) {
            hasLessThanTwo = true;
            break;
        }
    }
    if (hasLessThanTwo) {
        LOGI("Some keypoints have less than 2 frames. Writing initial data.");
        for (size_t i = 0; i < 1; ++i) {
            ofstream ofs(outputPath, ios::out);
            for (const auto& line : keypoints_outputs[i]) {
                LOGI("Keypoints written: %{public}s", line.c_str());
                ofs << line;
            }
            ofs.close();
        }
        return 1;
    }

    // 根据统计结果确定哪些关键点整体检测不足（例如检测次数 < 2）
    vector<int> few_keypt;
    for (int i = 0; i < 17; ++i) {
        if (keypoint_count[i] < 2) {
            few_keypt.push_back(i);
        }
    }
    LOGI("缺");
    for (int elem : few_keypt) {
        LOGI("%{public}d", elem);
    }
    LOGI("关键点");

    // 新的候选帧搜集逻辑
    vector<vector<int>> houxuan_id(17);  // 每个关键点对应的候选帧列表
    
    for (int pt : few_keypt) {
        for (int frameId : selectedIndices) {
            const vector<float>& kps = keypoints_dict[frameId];
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
        LOGI("关键点 %{public}d 的候选帧:", elem);
        for (size_t j = 0; j < houxuan_id[elem].size(); j++) {
            LOGI("%{public}d ", houxuan_id[elem][j]);
        }
    }
    
    for (int pt : few_keypt) {
        if (keypoint_count[pt] >= 2)
            continue;
        LOGI("开始补充关键点 %{public}d 的候选帧，当前检测次数为 %{public}d", pt, keypoint_count[pt]);
        
        while (keypoint_count[pt] < 2) {
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
                keypoints_dict[randomFrame] = result[0].kps;
                selectedIndices.push_back(randomFrame);
                
                int idx = pt * 3 + 2;
                if (idx >= result[0].kps.size())
                    continue;
                float newConf = result[0].kps[idx];
                LOGI("随机帧 %{public}d 关键点 %{public}d 置信度: %{public}f", randomFrame, pt, newConf);
                if (newConf > 0.4f ) {
                    houxuan_id[pt].push_back(randomFrame);
                    LOGI("随机帧 %{public}d 置信度合格，加入候选列表", randomFrame);
                }
            }
            else {
                for (int candFrame : houxuan_id[pt]) {
                    int startFrame = max(0, candFrame - window_range);
                    int endFrame = min((int)image_paths.size() - 1, candFrame + window_range);
                    for (int neighbor = startFrame; neighbor <= endFrame && keypoint_count[pt] < 2; neighbor++) {
                        if (processedFrames.find(neighbor) != processedFrames.end())
                            continue;
                            
                        std::vector<OutputPose> result;
                        std::string fullPath = imgDir + image_paths[neighbor];
                        inferenceModel(fullPath, result);
                        ++totalProcessedImages;
                        
                        keypoints_dict[neighbor] = result[0].kps;
                        processedFrames.insert(neighbor);  
                        selectedIndices.push_back(neighbor);
                        
                        if (keypoints_outputs.size() < 1)
                            keypoints_outputs.resize(1);
                        string file_name = image_paths[neighbor];
                        string data = file_name + "\n";
                        data += to_string(result[0].box.x / ratio) + " " + to_string(result[0].box.y / ratio) + " " + 
                                to_string(result[0].box.width / ratio) + " " + to_string(result[0].box.height / ratio) + "\n";
                        for (size_t k = 0; k < result[0].kps.size(); k += 3) {
                            float x = result[0].kps[k] / ratio;
                            float y = result[0].kps[k+1] / ratio;
                            float conf = result[0].kps[k+2];
                            data += to_string(x) + " " + to_string(y) + " " + to_string(conf) + "\n";
                        }
                        keypoints_outputs[0].push_back(data);
                        
                        int idx = pt * 3 + 2;
                        if (idx >= result[0].kps.size())
                            continue;
                        float newConf = result[0].kps[idx];
                        if (newConf > 0.75f) {
                            keypoint_count[pt]++;
                            LOGI("候选帧%{public}d成功补充关键点%{public}d，新置信度为%{public}f，当前计数： %{public}d", 
                                                neighbor, pt, newConf, keypoint_count[pt]);
                            if (keypoint_count[pt] >= 2) {
                                LOGI("关键点%{public}d已达到2次有效检测，停止补充。",pt);
                                break;
                            }
                        }
                    }
                    if (keypoint_count[pt] >= 2)
                        break;
                }
                houxuan_id[pt].clear();
            }
        }
        LOGW("Processed %{public}d images in total. It takes %{public}f seconds", totalProcessedImages, 
                                            std::chrono::duration<double>(clk::now() - begin).count());
    }

    // 保存所有关键点结果到文件
    for (size_t i = 0; i < 1; ++i) {
        ofstream ofs(outputPath, ios::out);
        for (const auto& line : keypoints_outputs[i]) {
            LOGI("Keypoints written: %{public}s", line.c_str());
            ofs << line;
        }
        ofs.close();
    }
    HIAIModelManager::GetInstance().UnloadModel();

    return 1;
}


void inferenceFaceModel(const std::string& imgPath, std::vector<FaceResult>& result)
{
    vector<Vec4d> params;
    LoadImage(imgPath, params);
    RunModel();
    auto model_output = HIAIModelManager::GetInstance().GetResult();
    
    std::vector<float> subvec(model_output[0].end() - 8420, model_output[0].end() - 8380);
    PostProcessingFace(HIAIModelManager::GetInstance().GetResult(), params, result);
}

void processSingleImage(const std::string& imgPath, const std::string& outputDir, FaceProcessingInfo& faceInfo)
{
    // Detect face
    std::vector<FaceResult> result;
    inferenceFaceModel(imgPath, result);
    
    // Load image
    cv::Mat frame = cv::imread(imgPath);
    if (frame.empty()) {
        OH_LOG_ERROR(LOG_APP, "Failed to read image: %{public}s", imgPath.c_str());
        return;
    }
    // Max score bbox
    if (!result.empty()) {
        const FaceResult& face = result[0];
        cv::Rect box = face.box;
        cv::Rect safeBox(
            std::max(0, box.x),
            std::max(0, box.y),
            std::min(box.width, frame.cols - box.x),
            std::min(box.height, frame.rows - box.y)
        );
        // Update roi bbox in faceInfo
        faceInfo.roi_bbox = safeBox;
        
        // Save cropped face image
        size_t lastSlash = imgPath.find_last_of('/');
        std::string imgName = imgPath.substr(lastSlash + 1);
        std::string outputPath = outputDir + "face_" + imgName;
        if (safeBox.width > 0 && safeBox.height > 0) {
            cv::Mat cropped = frame(safeBox);
            bool success = cv::imwrite(outputPath, cropped);
            if (success) {
                OH_LOG_INFO(LOG_APP, "Saved cropped face to: %{public}s", outputPath.c_str());
            } else {
                OH_LOG_ERROR(LOG_APP, "Failed to save image: %{public}s", outputPath.c_str());
            }
        } else {
            OH_LOG_WARN(LOG_APP, "Invalid crop box for image %{public}s", imgPath.c_str());
        }
    } else {
        OH_LOG_WARN(LOG_APP, "No face detected in image: %{public}s", imgPath.c_str());
    }
}

void detectFace(const std::string& inputPath, std::vector<FaceProcessingInfo>& faceInfos, std::pair<int, int> selectedIndices)
{
    // Construct model path
    size_t lastSlash = inputPath.find_last_of('/');
    std::string modelPath = inputPath.substr(0, lastSlash) + "/weights/yolov11-face.om";
    
    if (LoadModel(modelPath) != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "Model loading failed");
        return;
    }
    
    // Construct output path
    std::string outputDir = inputPath + "/human/selectedFace/";
    std::filesystem::create_directories(inputPath + "/human");
    std::filesystem::create_directories(outputDir);
    
    // Process images
//    std::string testImageName = "1181002925921078.jpg";
    
    // Save box
    processSingleImage(inputPath + "/images/" + faceInfos[selectedIndices.first].imageName, outputDir, faceInfos[selectedIndices.first]);
    processSingleImage(inputPath + "/images/" + faceInfos[selectedIndices.second].imageName, outputDir, faceInfos[selectedIndices.second]);
//    
//    processSingleImage(inputPath + "/images/" + testImageName, outputDir, faceInfos[selectedIndices.second]);
    HIAIModelManager::GetInstance().UnloadModel();
}