#include "manageHeadMasks.h"

// Beauty filter functions converted from Python
cv::Mat createSkinMaskColor(const cv::Mat& image) {
    // Convert to different color spaces
    cv::Mat hsv, ycrcb;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // Define skin color range in HSV
    cv::Scalar lower_hsv(0, 20, 70);
    cv::Scalar upper_hsv(20, 255, 255);
    cv::Mat mask_hsv;
    cv::inRange(hsv, lower_hsv, upper_hsv, mask_hsv);
    
    // Define skin color range in YCrCb
    cv::Scalar lower_ycrcb(0, 133, 77);
    cv::Scalar upper_ycrcb(255, 173, 127);
    cv::Mat mask_ycrcb;
    cv::inRange(ycrcb, lower_ycrcb, upper_ycrcb, mask_ycrcb);
    
    // Combine masks
    cv::Mat mask;
    cv::bitwise_and(mask_hsv, mask_ycrcb, mask);
    
    // Clean up the mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::GaussianBlur(mask, mask, cv::Size(9, 9), 3);
    
    return mask;
}

cv::Mat skinSmoothing(const cv::Mat& image, float smoothing_level = 0.5f) {
    // Create skin mask
    cv::Mat mask = createSkinMaskColor(image);
    
    // Normalize mask to 0-1 range
    cv::Mat mask_norm;
    mask.convertTo(mask_norm, CV_32F, 1.0/255.0);
    
    // Apply bilateral filter for edge-preserving smoothing
    int d = static_cast<int>(9 + smoothing_level * 10);
    double sigma_color = 20 + smoothing_level * 60;
    double sigma_space = 20 + smoothing_level * 60;
    
    cv::Mat smoothed;
    cv::bilateralFilter(image, smoothed, d, sigma_color, sigma_space);
    
    // Apply multiple passes of smaller bilateral filters
    cv::Mat temp;
      for (int i = 0; i < 2; ++i) {
        cv::bilateralFilter(smoothed, temp, 5, 30, 30);
        smoothed = temp.clone();
    }
    
    // Blend original and smoothed based on mask
    cv::Mat result;
    image.convertTo(result, CV_32F);
    cv::Mat smoothed_float;
    smoothed.convertTo(smoothed_float, CV_32F);
    
    std::vector<cv::Mat> channels_orig(3);
    std::vector<cv::Mat> channels_smooth(3);
    std::vector<cv::Mat> channels_result(3);
    
    cv::split(result, channels_orig);
    cv::split(smoothed_float, channels_smooth);
    
    for (int c = 0; c < 3; ++c) {
        channels_result[c] = channels_orig[c].mul(1.0f - mask_norm) + 
                            channels_smooth[c].mul(mask_norm);
    }
    
    cv::merge(channels_result, result);
    cv::Mat result_uint8;
    result.convertTo(result_uint8, CV_8U);
    
    return result_uint8;
}

cv::Mat skinWhitening(const cv::Mat& image, float whitening_level = 0.3f) {
    // Create skin mask
    cv::Mat mask = createSkinMaskColor(image);
    
    // Normalize mask to 0-1 range
    cv::Mat mask_norm;
    mask.convertTo(mask_norm, CV_32F, 1.0/255.0);
    
    // Convert to LAB color space
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F);
    
    // Split channels
    std::vector<cv::Mat> lab_channels(3);
    cv::split(lab, lab_channels);
    
    // Increase L channel (lightness)
    float brightening_factor = 1.0f + whitening_level * 0.3f;
    lab_channels[0] *= brightening_factor;
    cv::threshold(lab_channels[0], lab_channels[0], 255, 255, cv::THRESH_TRUNC);
    
    // Slightly reduce a and b channels to reduce redness and yellowness
    float color_reduction = 1.0f - whitening_level * 0.2f;
    lab_channels[1] = 128 + (lab_channels[1] - 128) * color_reduction;
    lab_channels[2] = 128 + (lab_channels[2] - 128) * color_reduction;
    
    // Merge channels
    cv::Mat lab_new;
    cv::merge(lab_channels, lab_new);
    lab_new.convertTo(lab_new, CV_8U);
    
    // Convert back to BGR
    cv::Mat brightened;
    cv::cvtColor(lab_new, brightened, cv::COLOR_Lab2BGR);
    
    // Blend based on mask
    cv::Mat result;
    image.convertTo(result, CV_32F);
    cv::Mat brightened_float;
    brightened.convertTo(brightened_float, CV_32F);
    
    std::vector<cv::Mat> channels_orig(3);
    std::vector<cv::Mat> channels_bright(3);
    std::vector<cv::Mat> channels_result(3);
    
    cv::split(result, channels_orig);
    cv::split(brightened_float, channels_bright);
    
    for (int c = 0; c < 3; ++c) {
        channels_result[c] = channels_orig[c].mul(1.0f - mask_norm) + 
                            channels_bright[c].mul(mask_norm);
    }
    
    cv::merge(channels_result, result);
    cv::Mat result_uint8;
    result.convertTo(result_uint8, CV_8U);
    
    return result_uint8;
}

cv::Mat applyBeautyFilters(const cv::Mat& image, float smoothing_level, float whitening_level) {
    // Apply skin smoothing first
    cv::Mat result = skinSmoothing(image, smoothing_level);
    
    // Apply whitening
    result = skinWhitening(result, whitening_level);
    
    return result;
}
