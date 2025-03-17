#include "segment.h"

using namespace cv;
using namespace std;
using namespace cv::dnn;
using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

OH_NN_ReturnCode LoadSegModel(const std::string& modelPath) {
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

void SaveSegmentation(cv::Mat& img, const std::vector<OutputParams>& result) {
    cv::Mat output(img.size(), img.type(), cv::Scalar(255, 255, 255));
    for (const auto& res : result) {
        if (res.id == 0 && !res.boxMask.empty()) {
            cv::Mat fullMask = cv::Mat::zeros(img.size(), CV_8UC1);
            res.boxMask.copyTo(fullMask(res.box));
            img.copyTo(output, fullMask);
        }
    }
    img = output;
}

OH_NN_ReturnCode LoadImages(vector<Vec4d>& params, vector<cv::Mat>& srcs, int imgSize) {
    vector<cv::Mat> input_images;
    Preprocessing(srcs, input_images, params, imgSize);

    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1/255.0, cv::Size(imgSize, imgSize));
    std::vector<std::pair<float*, size_t>> inputDataList{
        {reinterpret_cast<float*>(blob.data), blob.total()}
    };

    return HIAIModelManager::GetInstance().InitIOTensors(inputDataList);
}

OH_NN_ReturnCode LoadImage(vector<Vec4d>& params, cv::Mat& frame, int imgSize) {
    vector<cv::Mat> srcs{frame};
    vector<cv::Mat> input_images;
    Preprocessing(srcs, input_images, params, imgSize);
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1/255.0, cv::Size(imgSize, imgSize));
    OH_LOG_INFO(LOG_APP, "Input images size: %{public}zu", blob.total());
    
    std::vector<std::pair<float*, size_t>> inputDataList{
        {reinterpret_cast<float*>(blob.data), blob.total()}
    };

    return HIAIModelManager::GetInstance().InitIOTensors(inputDataList);
}

int GetSegments_Batched(std::string inputPath) {
    const int imgSize = 320;
    int model_output_size = 116 * 2100 * (imgSize / 320) ^2;
    int model_mask_size = 32 * (imgSize/4)^2;
    // Load model
    size_t lastSlash = inputPath.find_last_of('/');
    std::string modelPath = inputPath.substr(0, lastSlash) + "/weights/yolov8n-seg-320-batch4.om";
    if (LoadSegModel(modelPath) != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "Model loading failed");
        return -1;
    }

    // Process images
    std::string imgDir = inputPath + "/images/";
    std::string segmentDir = inputPath + "/segments/";
    std::filesystem::create_directory(segmentDir);
    DIR* dir = opendir(imgDir.c_str());
    if (!dir) {
        OH_LOG_ERROR(LOG_APP, "Failed to open directory: %s", imgDir.c_str());
        return -1;
    }

    std::vector<std::string> imgList;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename != "." && filename != "..") {
            imgList.push_back(filename);
        }
    }
    closedir(dir);
    
    vector<cv::Mat> input_imgs;
    vector<string> input_names;
    const int batch_size = 4;
    auto begin = clk::now();
    
    for (int i=0; i<imgList.size(); ++i) {
        vector<Vec4d> params;
        
        std::string imgName = imgList[i];
        std::string fullPath = imgDir + imgName;
        input_names.push_back(imgName);
        cv::Mat img = cv::imread(fullPath);
        if (img.empty()) {
            OH_LOG_ERROR(LOG_APP, "Failed to reload image: %s", fullPath.c_str());
            continue;
        }
        input_imgs.push_back(img);
        
        if(input_names.size()==batch_size){
            OH_LOG_INFO(LOG_APP, "Running with batch size: %{public}zu", input_imgs.size());
            
            if (LoadImages(params, input_imgs, imgSize) != OH_NN_SUCCESS) continue;
            if (RunModel() != OH_NN_SUCCESS) continue;
            vector<vector<float>> outputs = HIAIModelManager::GetInstance().GetResult();  // size仍然为2，只不过长度变为了3897600和3276800
            
            for(int j=0; j<batch_size; ++j){
                vector<vector<float>> single_output;
                
                vector<float> single_bbox(outputs[0].begin() + model_output_size*j, outputs[0].begin() + model_output_size*(j+1));
                vector<float> single_mask(outputs[0].begin() + model_mask_size*j, outputs[0].begin() + model_mask_size*(j+1));
                single_output.push_back(single_bbox);
                single_output.push_back(single_mask);
                OH_LOG_INFO(LOG_APP, "Single output size: %{public}zu", single_output.size());
                
                std::vector<OutputParams> output;
                PostProcessingSeg(single_output, params, output, imgSize);
                cv::Mat currentImg = input_imgs[j];
                SaveSegmentation(currentImg, output);
                std::string segmentPath = segmentDir + input_names[j];
        
                if (!cv::imwrite(segmentPath, currentImg)) {
                    OH_LOG_ERROR(LOG_APP, "Failed to save result: %{public}s", segmentPath.c_str());
                }
            }
            OH_LOG_INFO(LOG_APP, "Body segmentation completed in %{public}.3f seconds for image %{public}d", 
            std::chrono::duration<double>(clk::now() - begin).count(), i);
            input_imgs.clear();
            input_names.clear();
        }
    }


    return 0;
}

int GetSegments(std::string inputPath) {
    const int imgSize = 320;
    int model_output_size = 116 * 2100 * (imgSize / 320) ^2;
    int model_mask_size = 32 * (imgSize/4)^2;
    
    // Load model
    size_t lastSlash = inputPath.find_last_of('/');
    std::string modelPath = inputPath.substr(0, lastSlash) + "/weights/yolov8n-seg-320-new.om";
    if (LoadSegModel(modelPath) != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "Model loading failed");
        return -1;
    }

    // Process images
    std::string imgDir = inputPath + "/images/";
    std::string segmentDir = inputPath + "/segments/";
    std::filesystem::create_directory(segmentDir);
    DIR* dir = opendir(imgDir.c_str());
    if (!dir) {
        OH_LOG_ERROR(LOG_APP, "Failed to open directory: %s", imgDir.c_str());
        return -1;
    }

    std::vector<std::string> imgList;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename != "." && filename != "..") {
            imgList.push_back(filename);
        }
    }
    closedir(dir);
    auto begin = clk::now();

    for (int i=0; i<imgList.size(); i+=3) {
        std::string imgName = imgList[i];
        std::string fullPath = imgDir + imgName;
        vector<Vec4d> params;
        
        cv::Mat img = cv::imread(fullPath);
        if (LoadImage(params, img, imgSize) != OH_NN_SUCCESS) continue;
        if (RunModel() != OH_NN_SUCCESS) continue;

        if (img.empty()) {
            OH_LOG_ERROR(LOG_APP, "Failed to reload image: %s", fullPath.c_str());
            continue;
        }

        std::vector<OutputParams> output;
        PostProcessingSeg(HIAIModelManager::GetInstance().GetResult(), params, output, imgSize);
        SaveSegmentation(img, output);
        std::string segmentPath = segmentDir + imgName;
        
        if (!cv::imwrite(segmentPath, img)) {
            OH_LOG_ERROR(LOG_APP, "Failed to save result: %s", segmentPath.c_str());
        }
        OH_LOG_INFO(LOG_APP, "Body segmentation completed in %{public}.3f seconds for image %{public}d", 
            std::chrono::duration<double>(clk::now() - begin).count(), i);
    }
    
    return 0;
}