#include "keypoints.h"
#include <numeric>
#include <cmath>
using namespace std;
using namespace cv;
using namespace cv::dnn;

int PreprocessingLandmark(const std::vector<cv::Mat> &SrcImgs, std::vector<cv::Mat> &OutSrcImgs,
                          std::vector<cv::Vec4d> &params, int imgSize = 120) {
    OutSrcImgs.clear();
    Size input_size = Size(imgSize, imgSize);

    // 信封处理
    for (size_t i = 0; i < SrcImgs.size(); ++i) {
        Mat temp_img = SrcImgs[i];
        Vec4d temp_param = {1, 1, 0, 0};
        if (temp_img.size() != input_size) {
            Mat borderImg;
            LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
            OutSrcImgs.push_back(borderImg);
            params.push_back(temp_param);
        } else {
            OutSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }

    int lack_num = 1 - SrcImgs.size();
    if (lack_num > 0) {
        Mat temp_img = Mat::zeros(input_size, CV_8UC3);
        Vec4d temp_param = {1, 1, 0, 0};
        OutSrcImgs.push_back(temp_img);
        params.push_back(temp_param);
    }
    return 0;
}

// Structure to hold landmark results
struct OutputLandmark {
    std::vector<cv::Point2f> landmarks;
    std::vector<float> scores;
};

// Convert heatmaps to landmark coordinates
void GetPredsFromHeatmap(const cv::Mat& heatmaps, std::vector<cv::Point2f>& landmarks, std::vector<float>& scores) {
    int num_landmarks = heatmaps.size[1];  // 68
    int hm_height = heatmaps.size[2];      // 64
    int hm_width = heatmaps.size[3];       // 64
    
    landmarks.clear();
    scores.clear();
    landmarks.resize(num_landmarks);
    scores.resize(num_landmarks);
    
    float* hm_data = (float*)heatmaps.data;
    
    for (int i = 0; i < num_landmarks; ++i) {
        // Find max value and its location for each heatmap
        float max_val = -1e10;
        int max_idx = 0;
        
        // Offset to current heatmap
        int offset = i * hm_height * hm_width;
        
        for (int j = 0; j < hm_height * hm_width; ++j) {
            if (hm_data[offset + j] > max_val) {
                max_val = hm_data[offset + j];
                max_idx = j;
            }
        }
        
        scores[i] = max_val;
        
        // Convert index to (x, y) coordinates
        int pX = max_idx % hm_width;
        int pY = max_idx / hm_width;
        
        // Sub-pixel refinement
        float x = pX;
        float y = pY;
        
        if (pX > 0 && pX < hm_width - 1 && pY > 0 && pY < hm_height - 1) {
            float diff_x = hm_data[offset + pY * hm_width + pX + 1] - 
                          hm_data[offset + pY * hm_width + pX - 1];
            float diff_y = hm_data[offset + (pY + 1) * hm_width + pX] - 
                          hm_data[offset + (pY - 1) * hm_width + pX];
            
            x += (diff_x > 0 ? 0.25 : -0.25);
            y += (diff_y > 0 ? 0.25 : -0.25);
        }
        
        // Scale from heatmap coordinates to image coordinates (256x256)
        x = (x - 0.5) * 120.0 / hm_width;
        y = (y - 0.5) * 120.0 / hm_height;
        
        landmarks[i] = cv::Point2f(x, y);
    }
}

int PostProcessingLandmark(std::vector<std::vector<float>> result, std::vector<cv::Vec4d> &params, 
                          std::vector<OutputLandmark> &output) {
    float *all_data = result[0].data();
    
    // Create Mat from output data [1, 68, 64, 64]
    std::vector<int> sizes = {1, 68, 64, 64};
    cv::Mat heatmaps(4, sizes.data(), CV_32F, all_data);
    
    OutputLandmark landmark_result;
    GetPredsFromHeatmap(heatmaps, landmark_result.landmarks, landmark_result.scores);
    
    // Transform landmarks back to original image coordinates
    for (size_t i = 0; i < landmark_result.landmarks.size(); ++i) {
        landmark_result.landmarks[i].x = (landmark_result.landmarks[i].x - params[0][2]) / params[0][0];
        landmark_result.landmarks[i].y = (landmark_result.landmarks[i].y - params[0][3]) / params[0][1];
    }
    
    output.push_back(landmark_result);
    return 0;
}

// Modified LoadImage for landmark detection
OH_NN_ReturnCode LoadImageLandmark(const std::string& imagePath, vector<Vec4d>& params) {
    cv::Mat frame = cv::imread(imagePath);
    
    if (frame.empty()) {
        OH_LOG_ERROR(LOG_APP, "Failed to read image: %s", imagePath.c_str());
        return OH_NN_FAILED;
    }

    vector<cv::Mat> srcs{frame};
    vector<cv::Mat> input_images;
    PreprocessingLandmark(srcs, input_images, params, 120);

    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1/255.0, cv::Size(120, 120));
    std::vector<std::pair<float*, size_t>> inputDataList{
        {reinterpret_cast<float*>(blob.data), blob.total()}
    };

    return HIAIModelManager::GetInstance().InitIOTensors(inputDataList);
}

void inferenceLandmarkModel(const std::string& imgPath, std::vector<OutputLandmark>& result, vector<Vec4d>& params) {
    LoadImageLandmark(imgPath, params);
    RunModel();
    PostProcessingLandmark(HIAIModelManager::GetInstance().GetResult(), params, result);
}

int ProcessLandmarks(const std::vector<std::string>& imagePaths, const std::string& modelDir, 
                     const std::string& outputPath) {
    auto begin = clk::now();
    int totalProcessedImages = 0;
    
    // Construct model path
    std::string modelPath = modelDir + "/debug.om";
    
    if (LoadModel(modelPath) != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "Model loading failed");
        return 0;
    }

    // Create output file
    std::ofstream outfile(outputPath);
    if (!outfile.is_open()) {
        OH_LOG_ERROR(LOG_APP, "Failed to create output file: %s", outputPath.c_str());
        return 0;
    }

    // Process each image
    for (const auto& imgPath : imagePaths) {
        std::vector<OutputLandmark> result;
        vector<Vec4d> params;
        
        // Extract filename
        size_t last_slash = imgPath.find_last_of("/\\");
        std::string imgName = (last_slash != string::npos) ? 
                              imgPath.substr(last_slash + 1) : imgPath;
        
        inferenceLandmarkModel(imgPath, result, params);
        ++totalProcessedImages;
        
        if (!result.empty() && !result[0].landmarks.empty()) {
            // Write filename
            outfile << imgName << "\n";
            
            // Skip bounding box line (write empty line or placeholder)
            outfile << "0 0 0 0\n";
            
            // Write first 17 landmarks
            int num_landmarks_to_save = std::min(17, (int)result[0].landmarks.size());
            for (int i = 0; i < num_landmarks_to_save; ++i) {
                outfile << result[0].landmarks[i].x << " " 
                       << result[0].landmarks[i].y << " " 
                       << result[0].scores[i] << "\n";
            }
            
            LOGI("Processed %{public}s with %{public}d landmarks", imgName.c_str(), num_landmarks_to_save);
        } else {
            LOGE("No landmarks detected in %{public}s", imgName.c_str());
        }
    }
    
    outfile.close();
    HIAIModelManager::GetInstance().UnloadModel();
    
    LOGW("Processed %{public}d images in total. It takes %{public}f seconds", totalProcessedImages, 
         std::chrono::duration<double>(clk::now() - begin).count());
    
    return 1;
}

// Wrapper function to match the expected interface
int ProcessLandmarks(const std::string& scenePath, const std::vector<std::string>& imagePaths) {
    // Extract model directory from first image path
    if (imagePaths.empty()) {
        OH_LOG_ERROR(LOG_APP, "No image paths provided");
        return 0;
    }
    
    size_t lastSlash = scenePath.find_last_of('/');
    std::string baseDir = (lastSlash != std::string::npos) ? 
                          scenePath.substr(0, lastSlash) : ".";
    std::string modelDir = baseDir + "/weights";
    
    std::string outputPath = scenePath + "/landmarks.txt";
    
    return ProcessLandmarks(imagePaths, modelDir, outputPath);
}