#include "utils.h"
#include "HIAIModelManager.h"

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

void LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape, bool autoShape,
               bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color) {
    // 取较小的缩放比例
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);
    LOGI("原图尺寸：w:%{public}d * h:%{public}d, 要求尺寸：w:%{public}d * h:%{public}d, 即将采用的缩放比：%%{public}f\n", shape.width, shape.height,
           newShape.width, newShape.height, r);

    // 依据前面的缩放比例后，原图的尺寸
    float ratio[2]{r, r};
    int new_un_pad[2] = {(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};
    LOGI("等比例缩放后的尺寸该为：w:%{public}d * h:%{public}d\n", new_un_pad[0], new_un_pad[1]);

    // 计算距离目标尺寸的padding像素数
    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);
    if (autoShape) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;
    printf("填充padding: dw=%f , dh=%f\n", dw, dh);

    // 等比例缩放
    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1]) {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    } else {
        outImage = image.clone();
    }

    // 图像四周padding填充，至此原图与目标尺寸一致
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0]; // width的缩放比例
    params[1] = ratio[1]; // height的缩放比例
    params[2] = left;     // 水平方向两边的padding像素数
    params[3] = top;      // 垂直方向两边的padding像素数
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


int Preprocessing(const std::vector<cv::Mat> &SrcImgs, std::vector<cv::Mat> &OutSrcImgs,
                  std::vector<cv::Vec4d> &params, int imgSize) {
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