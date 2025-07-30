//
// Created on 2025/7/30.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#include "human/HiAI/HIAIModelManager.h"
#include "human/HiAI/utils.h"
#include "human/keypoints/keypoints.h"
#include "landmark.h"
#include "base/math/vector.h"


int PostProcessingFace(std::vector<std::vector<float>> result, std::vector<cv::Vec4d> &params, std::vector<FaceResult> &output) {
    const float *all_data = result[0].data();
    Mat output0 = Mat(Size(8400, 5), CV_32F, (float *)all_data).t(); // [5 ,8400] -> [8400, 5]

    // 一张图片的预测框
    vector<float> confidences;
    vector<Rect> boxes;
    for (int r = 0; r < output0.rows; ++r) {
        // 预测框坐标映射到原图上
        float score = output0.at<float>(r, 4);
        if (score > 0.25) {
            // rect [x,y,w,h]
            float x = (output0.at<float>(r, 0) - params[0][2]) / params[0][0]; // x
            float y = (output0.at<float>(r, 1) - params[0][3]) / params[0][1]; // y
            float w = output0.at<float>(r, 2) / params[0][0];                          // w
            float h = output0.at<float>(r, 3) / params[0][1];                          // h

            int left = MAX(int(x - 0.5 * w + 0.5), 0);
            int top = MAX(int(y - 0.5 * h + 0.5), 0);

            confidences.push_back(score);
            boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }
    }

    // 对一张图的预测框执行NMS处理
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, nms_result); // 还需要classThreshold？

    // 对一张图片：依据NMS处理得到的索引，得到类别id、confidence、box，并置于结构体OutputDet的容器中
    for (size_t i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        FaceResult result;

        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    return 0;
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

cv::Rect2d parse_roi_box_from_bbox(const cv::Rect2d& bbox) {
    float left   = static_cast<float>(bbox.x);
    float top    = static_cast<float>(bbox.y);
    float right  = static_cast<float>(bbox.x + bbox.width);
    float bottom = static_cast<float>(bbox.y + bbox.height);

    float old_size = (right - left + bottom - top) / 2.0f;
    float center_x = right - (right - left) / 2.0f;
    float center_y = bottom - (bottom - top) / 2.0f + old_size * 0.14f;
    float size = old_size * 1.58f;

    cv::Rect2d roi_box;
    roi_box.x = center_x - size / 2.0f; // x1
    roi_box.y = center_y - size / 2.0f; // y1
    roi_box.width = size;
    roi_box.height = size;
    return roi_box;
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
        faceInfo.roi_bbox = parse_roi_box_from_bbox(safeBox);
        
        // Save cropped face image
        size_t lastSlash = imgPath.find_last_of('/');
        std::string imgName = imgPath.substr(lastSlash + 1);
        std::string outputPath = outputDir + imgName;
        if (safeBox.width > 0 && safeBox.height > 0) {
            cv::Mat cropped = frame(faceInfo.roi_bbox);
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