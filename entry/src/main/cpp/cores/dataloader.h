#pragma once
#include <iostream>
#include <vector>
#include "tinyply.h"
#define GLM_FORCE_PRECISION_HIGH     // 高精度
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES  // 确保默认对齐方式
#include <glm/glm.hpp>
#include "vulkanexamplebase.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>

#include "file/file_operator.h"
#include "video2images.h"

using namespace tinyply;

// DataLoader Tool Class
class DataLoader
{
public:
    // PlyDataLoader Test
    struct PlyDataStruct {
        std::vector<glm::vec3> xyz;
        std::vector<glm::vec3> scale;
        std::vector<glm::vec4> rotate;
    };
    
    struct InputInitStruct {
        std::vector<glm::vec3> positions;      // 3D positions (x, y, z)
        std::vector<glm::vec3> scales;         // Scaling factors (scale_x, scale_y, scale_z)
        std::vector<glm::vec4> rotations;      // Rotation (rotate_0, rotate_1, rotate_2, rotate_3)
        std::vector<float> opacities;          // Opacity

        struct Data {
            float values[48];  // 48维浮点数组
        };
        struct DataExtr{
            float values[7];  
        };

        
        std::vector<Data> colors;         // RGB colors (r, g, b)
        std::vector<vks::Texture2D> images;   // Images
        std::vector<vks::Texture2D> images_low_res;

        std::vector<std::string> img_list;
        std::vector<DataExtr> extr_tmp;  
        std::vector<DataExtr> extr_tmp_test; 
        std::vector<std::vector<float>> intr_tmp;

        std::vector<float> head_bbox;

        std::vector<glm::vec3> cam_centers;
        glm::vec3 translate;
        float radius;
        glm::vec3 mean;
        float variance = 0;

        std::string basepath = "";
        
        void normalizeData();
        void computePositionStats();
    };

    static PlyDataStruct load_input_data(const std::string& filepath);
    
    static void loadInputLegoInitDatav2(std::string dataFolder, DataLoader::InputInitStruct& data, bool& resumed);
    static InputInitStruct loadHeadGaussians(std::string dataFolder, DataLoader::InputInitStruct& gaussian);
    static void loadInputGaussianData(std::string& gaussianPath, DataLoader::InputInitStruct& data);
    static void loadExtraGaussianData(std::string& gaussianPath, DataLoader::InputInitStruct& data);
    
    static void loadExtrinsics(std::string dataPath, DataLoader::InputInitStruct& data);
    static void loadIntrinsics(std::string dataPath, DataLoader::InputInitStruct& data);
    
    static InputInitStruct loadInputTo(const std::string _scenepath);
    static void loadInput12(const std::string _scenepath, DataLoader::InputInitStruct& data, bool& resumed);
    static void createEnlargedFace(const std::string scenepath, DataLoader::InputInitStruct& data, std::vector<cv::Point>& faceCenters);
    
    static std::vector<cv::Point> projectHeadCenter(DataLoader::InputInitStruct& data);
    
    static void saveToPLY_tinyply(const std::string& filename, const DataLoader::InputInitStruct& data, bool save_head);
    static void printPlyHeader(const std::string& filepath);
    
};
