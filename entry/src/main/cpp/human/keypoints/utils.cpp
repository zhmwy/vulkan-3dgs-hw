//
// Created on 2024/11/5.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "utils.h"
#include "human/keypoints/keypoints.h"
using namespace cv;
using namespace std;

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

