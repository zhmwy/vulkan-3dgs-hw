//
// Created on 2025/3/28.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#ifndef VULKAN_3DGS_HW_VIDEO2IMAGES_H
#define VULKAN_3DGS_HW_VIDEO2IMAGES_H

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    #include <libavutil/display.h>
}
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include "common/common.h"
#include <filesystem>
bool processVideoWithFFmpeg(
    const std::string& videoPath, 
    const std::vector<std::string>& timestamps,
    const std::unordered_set<std::string>& uniqueImages,
    const std::string& scenepath,
    const std::string& ext,
    int width, int height,
    std::vector<std::string>& img_list);
#endif //VULKAN_3DGS_HW_VIDEO2IMAGES_H
