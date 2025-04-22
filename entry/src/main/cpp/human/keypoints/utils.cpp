//
// Created on 2024/11/5.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "utils.h"
using namespace cv;
using namespace std;

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

int PostProcessing(std::vector<std::vector<float>> result, std::vector<cv::Vec4d> &params, std::vector<OutputPose> &output) {
    float *all_data = result[0].data();
    int64_t one_output_length = 8400 * 56;
    Mat output0 = Mat(Size(8400, 56), CV_32F, all_data).t(); // [1, 56 ,8400] -> [1, 8400, 56]

    float *pdata = (float *)output0.data; // [classid,x,y,w,h,x,y,...21个点]
    int rows = output0.rows;              // 预测框的数量 8400

    // 一张图片的预测框

    vector<float> confidences;
    vector<Rect> boxes;
    vector<int> labels;
    vector<vector<float>> kpss;
    for (int r = 0; r < rows; ++r) {
        // 得到人类别概率
        auto kps_ptr = pdata + 5;
        // 预测框坐标映射到原图上
        float score = pdata[4];
        if (score > 0.25) {
            // rect [x,y,w,h]
            float x = (pdata[0] - params[0][2]) / params[0][0]; // x
            float y = (pdata[1] - params[0][3]) / params[0][1]; // y
            float w = pdata[2] / params[0][0];                          // w
            float h = pdata[3] / params[0][1];                          // h
    
            int left = MAX(int(x - 0.5 * w + 0.5), 0);
            int top = MAX(int(y - 0.5 * h + 0.5), 0);
    
            std::vector<float> kps;
            for (int k = 0; k < 17; k++) {
                float kps_x = (*(kps_ptr + 3 * k) - params[0][2]) / params[0][0];
                float kps_y = (*(kps_ptr + 3 * k + 1) - params[0][3]) / params[0][1];
                float kps_s = *(kps_ptr + 3 * k + 2);
    
    //          cout << *(kps_ptr + 3*k) << endl;
    
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }
    
            confidences.push_back(score);
            labels.push_back(0);
            kpss.push_back(kps);
            boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }
        pdata += 56; // 下一个预测框
    }

    // 对一张图的预测框执行NMS处理
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, nms_result); // 还需要classThreshold？

    // 对一张图片：依据NMS处理得到的索引，得到类别id、confidence、box，并置于结构体OutputDet的容器中
    for (size_t i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        OutputPose result;

        result.confidence = confidences[idx];
        result.box = boxes[idx];
        result.label = labels[idx];
        result.kps = kpss[idx];
        output.push_back(result);
    }
    return 0;
}