//
// Created on 2024/11/5.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "../keypoints/utils.h"
#include "segment.h"
using namespace cv;
using namespace std;

template <typename T>
T VectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
};

void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputParams& output, const MaskParams& maskParams) {
	int net_width = maskParams.netWidth;
	int net_height = maskParams.netHeight;
	int seg_channels = maskProtos.size[1];
	int seg_height = maskProtos.size[2];
	int seg_width = maskProtos.size[3];
	float mask_threshold = maskParams.maskThreshold;
	cv::Vec4f params = maskParams.params;
	cv::Size src_img_shape = maskParams.srcImgShape;

	cv::Rect temp_rect = output.box;
	//crop from mask_protos
	int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
	int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

	//�������� mask_protos(roi_rangs).clone()λ�ñ�����˵�����output.box���ݲ��ԣ����߾��ο��1�����صģ����������ע�Ͳ��ַ�ֹ������
	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width) {
		if (seg_width - rang_x > 0)
			rang_w = seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > seg_height) {
		if (seg_height - rang_y > 0)
			rang_h = seg_height - rang_y;
		else
			rang_y -= 1;
	}

	std::vector<cv::Range> roi_rangs;
	roi_rangs.push_back(cv::Range(0, 1));
	roi_rangs.push_back(cv::Range::all());
	roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

	//crop
	cv::Mat temp_mask_protos = maskProtos(roi_rangs).clone();
	cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
	cv::Mat matmul_res = (maskProposals * protos).t();
	cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	cv::Mat dest, mask;

	//sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
	int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
	int width = ceil(net_width / seg_width * rang_w / params[0]);
	int height = ceil(net_height / seg_height * rang_h / params[1]);

	resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
	cv::Rect mask_rect = temp_rect - cv::Point(left, top);
	mask_rect &= cv::Rect(0, 0, width, height);
	mask = mask(mask_rect) > mask_threshold;
	if (mask.rows != temp_rect.height || mask.cols != temp_rect.width) { //https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp/pull/30
		resize(mask, mask, temp_rect.size(), cv::INTER_NEAREST);
	}
	output.boxMask = mask;

}

bool PostProcessingSeg(std::vector<std::vector<float>> result, std::vector<cv::Vec4d> &params, std::vector<OutputParams> &output, int imgSize) {
	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;

	float* all_data = result[0].data();
    float* mask_data = result[1].data();

	std::vector<int> mask_protos_shape = {1, 32, imgSize/4, imgSize/4}; // 这些大小注意一下
	int mask_protos_length = VectorProduct(mask_protos_shape);
	int64_t one_output_length = 116 * 2100;
	int net_width = 116;
	int socre_array_length = net_width - 4 - 32;

    cv::Mat output0 = cv::Mat(cv::Size(2100, 116), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
	all_data += one_output_length;
	float* pdata = (float*)output0.data;
    
	int rows = output0.rows;
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
	for (int r = 0; r < rows; ++r) {    //stride
		cv::Mat scores(1, socre_array_length, CV_32F, pdata + 4);
		cv::Point classIdPoint;
		double max_class_socre;
		minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
		max_class_socre = (float)max_class_socre;
		if (max_class_socre >= _classThreshold) {
			std::vector<float> temp_proto(pdata + 4 + socre_array_length, pdata + net_width);
			picked_proposals.push_back(temp_proto);
			//rect [x,y,w,h]
			float x = (pdata[0] - params[0][2]) / params[0][0];  //x
			float y = (pdata[1] - params[0][3]) / params[0][1];  //y
			float w = pdata[2] / params[0][0];  //w
			float h = pdata[3] / params[0][1];  //h
            LOGI("%{public}f %{public}f %{public}f %{public}f", x, y, w, h);
			int left = MAX(int(x - 0.5 * w + 0.5), 0);
			int top = MAX(int(y - 0.5 * h + 0.5), 0);
			class_ids.push_back(classIdPoint.x);
			confidences.push_back(max_class_socre);
			boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
		}
		pdata += net_width;//��һ��
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
	std::vector<std::vector<float>> temp_mask_proposals;
	cv::Rect holeImgRect(0, 0, 480, 640); // 确保是不是640， 640
	std::vector<OutputParams> temp_output;
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputParams result;
		result.id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		if (result.box.area() < 1)
			continue;
		temp_mask_proposals.push_back(picked_proposals[idx]);
		temp_output.push_back(result);
	}

	MaskParams mask_params;
	mask_params.params = params[0];
	mask_params.srcImgShape = cv::Size(1080, 1920);  // 是图像原始大小
	mask_params.netHeight = imgSize;
	mask_params.netWidth = imgSize;
	mask_params.maskThreshold = _maskThreshold;
	cv::Mat mask_protos = cv::Mat(mask_protos_shape, CV_32F, mask_data);
        
	for (int i = 0; i < temp_mask_proposals.size(); ++i) {
		GetMask2(cv::Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
	}

	output = temp_output;

	if (output.size())
		return true;
	else
		return false;
}