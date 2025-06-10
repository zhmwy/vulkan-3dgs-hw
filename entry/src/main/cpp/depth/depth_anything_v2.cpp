#include "depth_anything_v2.h"
#include "common/common.h"

using namespace cv;
using namespace std;
using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

class DepthAnythingHiAI {
private:
    // Model parameters
    const int INPUT_WIDTH = 210;
    const int INPUT_HEIGHT = 280;
    const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
    const std::vector<float> STD = {0.229f, 0.224f, 0.225f};
    
public:
    cv::Mat preprocess(const cv::Mat& image) {
        cv::Mat processed;
        
        // Convert BGR to RGB and normalize to [0, 1]
        cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);
        processed.convertTo(processed, CV_32F, 1.0 / 255.0);
        
        // Resize to model input size
        cv::resize(processed, processed, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, cv::INTER_CUBIC);
        
        // Normalize with mean and std
        std::vector<cv::Mat> channels(3);
        cv::split(processed, channels);
        
        for (int i = 0; i < 3; i++) {
            channels[i] = (channels[i] - MEAN[i]) / STD[i];
        }
        
        cv::merge(channels, processed);
        
        return processed;
    }
    
    std::vector<float> matToVector(const cv::Mat& mat) {
        // Convert HWC to CHW format
        std::vector<float> vec;
        vec.reserve(mat.channels() * mat.rows * mat.cols);
        
        std::vector<cv::Mat> channels;
        cv::split(mat, channels);
        
        for (const auto& channel : channels) {
            vec.insert(vec.end(), (float*)channel.data, (float*)channel.data + channel.total());
        }
        
        return vec;
    }
    
    cv::Mat postprocess(const cv::Mat& depth_map, int original_width, int original_height) {
        cv::Mat processed_depth;
        
        // Normalize to [0, 255]
        double min_val, max_val;
        cv::minMaxLoc(depth_map, &min_val, &max_val);
        
        depth_map.convertTo(processed_depth, CV_8U, 255.0 / (max_val - min_val), 
                           -min_val * 255.0 / (max_val - min_val));
        
        // Resize to original dimensions
        cv::resize(processed_depth, processed_depth, 
                  cv::Size(original_width, original_height), 0, 0, cv::INTER_CUBIC);
        
        // Apply color map
//         cv::Mat depth_color;
//         cv::applyColorMap(processed_depth, depth_color, cv::COLORMAP_INFERNO);
        
        return processed_depth;
    }
};

OH_NN_ReturnCode LoadDepthModel(const std::string& modelPath) {
    FILE* modelFile = fopen(modelPath.c_str(), "rb");
    OH_LOG_INFO(LOG_APP, "Opening model file at %s", modelPath.c_str());
    
    if (!modelFile) {
        OH_LOG_ERROR(LOG_APP, "Failed to open model file at %s", modelPath.c_str());
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

    OH_NN_ReturnCode ret = HIAIModelManager::GetInstance().LoadModelFromBuffer(modelData.get(), modelSize);
    if (ret != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "LoadModelFromBuffer failed");
        return ret;
    }
    OH_LOG_INFO(LOG_APP, "Model loaded successfully");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LoadDepthImage(cv::Mat& frame, DepthAnythingHiAI& depthEstimator) {
    // Preprocess
    cv::Mat preprocessed = depthEstimator.preprocess(frame);
    
    // Convert to blob format (CHW)
    std::vector<float> input_data = depthEstimator.matToVector(preprocessed);
    
    OH_LOG_INFO(LOG_APP, "Input data size: %{public}zu", input_data.size());
    
    std::vector<std::pair<float*, size_t>> inputDataList{
        {input_data.data(), input_data.size()}
    };

    return HIAIModelManager::GetInstance().InitIOTensors(inputDataList);
}

int ProcessDepthEstimation(std::string inputPath) {
    DepthAnythingHiAI depthEstimator;
    const int INPUT_WIDTH = 480;
    const int INPUT_HEIGHT = 640;
    
    // Load model
    size_t lastSlash = inputPath.find_last_of('/');
    std::string modelPath = inputPath.substr(0, lastSlash) + "/weights/depth_anything_280.om";
    if (LoadDepthModel(modelPath) != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "Model loading failed");
        return -1;
    }

    // Setup directories
    std::string imgDir = inputPath + "/images/";
    std::string depthDir = inputPath + "/depth_maps/";
    std::filesystem::create_directory(depthDir);
    
    DIR* dir = opendir(imgDir.c_str());
    if (!dir) {
        OH_LOG_ERROR(LOG_APP, "Failed to open directory: %s", imgDir.c_str());
        return -1;
    }

    std::vector<std::string> imgList;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename != "." && filename != ".." && 
            (filename.find(".jpg") != std::string::npos || 
             filename.find(".png") != std::string::npos ||
             filename.find(".jpeg") != std::string::npos)) {
            imgList.push_back(filename);
        }
    }
    closedir(dir);
    
    OH_LOG_INFO(LOG_APP, "Found %{public}zu images to process", imgList.size());
    
    auto begin = clk::now();
    
    for (size_t i = 0; i < imgList.size(); ++i) {
        std::string imgName = imgList[i];
        std::string fullPath = imgDir + imgName;
        
        cv::Mat img = cv::imread(fullPath);
        if (img.empty()) {
            OH_LOG_ERROR(LOG_APP, "Failed to load image: %s", fullPath.c_str());
            continue;
        }
        
        // Store original dimensions
        int original_width = img.cols;
        int original_height = img.rows;
        
        OH_LOG_INFO(LOG_APP, "Processing image: %{public}s (%{public}dx%{public}d)", imgName.c_str(), original_width, original_height);
        
        // Load and run model
        if (LoadDepthImage(img, depthEstimator) != OH_NN_SUCCESS) {
            OH_LOG_ERROR(LOG_APP, "Failed to load image data for inference");
            continue;
        }
        
        if (HIAIModelManager::GetInstance().RunModel() != OH_NN_SUCCESS) {
            OH_LOG_ERROR(LOG_APP, "Failed to run model");
            continue;
        }
        
        // Get results
        std::vector<std::vector<float>> outputs = HIAIModelManager::GetInstance().GetResult();

        if (outputs.empty()) {
            OH_LOG_ERROR(LOG_APP, "No output from model");
            continue;
        }
        
        // Convert output to depth map
        std::vector<float>& depth_data = outputs[0];
//         LOGI("%{public}zu", depth_data.size());
        for(int i=0; i<10; ++i){
            LOGI("%{public}f", depth_data[i]);
        }
        cv::Mat depth_map(280, 210, CV_32FC1, depth_data.data());
        
        // Postprocess
        cv::Mat depth = depthEstimator.postprocess(depth_map, INPUT_WIDTH, INPUT_HEIGHT);
        
        // Optional: Save grayscale depth map
//         cv::Mat depth_gray;
//         depth_map.convertTo(depth_gray, CV_8U, 255.0);
        std::string depthPath = depthDir + imgName;
        cv::imwrite(depthPath, depth);
        
        auto elapsed = std::chrono::duration<double>(clk::now() - begin).count();
        OH_LOG_INFO(LOG_APP, "Processed %{public}zu/%{public}zu images in %{public}.3f seconds", 
                    i + 1, imgList.size(), elapsed);
    }
    
    auto total_time = std::chrono::duration<double>(clk::now() - begin).count();
    OH_LOG_INFO(LOG_APP, "Total processing time: %{public}.3f seconds for %{public}zu images", 
                total_time, imgList.size());
    
    return 0;
}