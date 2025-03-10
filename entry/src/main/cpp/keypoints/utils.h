//
// Created on 2024/11/5.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef POINTRIX_HUMAN_UITLS_H
#define POINTRIX_HUMAN_UITLS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>

struct  OutputPose {
    cv::Rect_<float> box;
    int label =0;
    float confidence =0.0;
    std::vector<float> kps;

};

struct PoseKeyPoint {
	float x = 0;
	float y = 0;
	float confidence = 0;
};

struct OutputParams {
	int id;             //������id
	float confidence;   //������Ŷ�
	cv::Rect box;       //���ο�
	cv::RotatedRect rotatedBox;  //obb������ο�
	cv::Mat boxMask;       //���ο���mask����ʡ�ڴ�ռ�ͼӿ��ٶ�
	std::vector<PoseKeyPoint> keyPoints; //pose key points

};

struct MaskParams {
	//int segChannels = 32;
	//int segWidth = 160;
	//int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};

void LetterBox(const cv::Mat& image, cv::Mat& outImage,
               cv::Vec4d& params,
               const cv::Size& newShape = cv::Size(640, 640),
               bool autoShape = false,
               bool scaleFill=false,
               bool scaleUp=true,
               int stride= 32,
               const cv::Scalar& color = cv::Scalar(114,114,114));

int Preprocessing(const std::vector<cv::Mat> &SrcImgs,
                              std::vector<cv::Mat> &OutSrcImgs,
                              std::vector<cv::Vec4d> &params);

int PostProcessing(std::vector<std::vector<float>> result, 
                   std::vector<cv::Vec4d> &params,
                   std::vector<OutputPose> &output);

bool PostProcessingSeg(std::vector<std::vector<float>> result, 
                       std::vector<cv::Vec4d> &params, 
                       std::vector<OutputParams> &output);

#endif //POINTRIX_HUMAN_UITLS_H
