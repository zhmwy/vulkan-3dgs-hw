#pragma once
#include <iostream>
#include <vector>
#include "tinyply.h"
#include "vulkancomputebase.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>

#include "file/file_operator.h"
#include "human/human.h"
#include "base/math/vector.h"


using namespace tinyply;

// DataLoader Tool Class
class DataLoader
{
public:
    // PlyDataLoader Test
    struct PlyDataStruct {
        std::vector<Base::Math::Vec3> xyz;
        std::vector<Base::Math::Vec3> scale;
        std::vector<Base::Math::Vec4> rotate;
    };
    
    struct InputInitStruct {
        std::vector<Base::Math::Vec3> positions;      // 3D positions (x, y, z)
        std::vector<Base::Math::Vec3> scales;         // Scaling factors (scale_x, scale_y, scale_z)
        std::vector<Base::Math::Vec4> rotations;      // Rotation (rotate_0, rotate_1, rotate_2, rotate_3)
        std::vector<float> opacities;          // Opacity

        struct Data {
            float values[48];  // 48维浮点数组
        };
        struct DataExtr{
            float values[7];  
        };

        
        std::vector<Data> colors;         // RGB colors (r, g, b)
        std::vector<vkbase::Texture2DGS> images;   // Images
        std::vector<vkbase::Texture2DGS> images_low_res;

        std::vector<std::string> img_list;
        std::vector<DataExtr> extr_tmp;  
        std::vector<DataExtr> extr_tmp_test; 
        std::vector<std::vector<float>> intr_tmp;

        std::vector<float> head_bbox;
        std::vector<float> body_bbox;
        float max_movement;
        
        std::vector<Base::Math::Vec3> cam_centers;
        Base::Math::Vec3 translate;
        float radius = 1.0;
        Base::Math::Vec3 mean = Base::Math::Vec3(0.0, 0.0, 0.0);
        float variance = 1.0;

        std::string basepath = "/data/storage/el1/base/files";
        
        std::vector<float> time_stamps; // Time for Gaussian-Flow

        
        int width, height;
        
        //void normalizeData();
        //void computePositionStats();
        void getNerfppNorm();
        
        void denormalizeExtr();
        void denormalizePoints();
    };

    static PlyDataStruct load_input_data(const std::string& filepath);
    
    static void loadInputLegoInitDatav2(std::string dataFolder, DataLoader::InputInitStruct& data, bool& resumed);
    static InputInitStruct loadHeadGaussians(std::string dataFolder, DataLoader::InputInitStruct& gaussian);
    static void loadInputGaussianData(std::string& gaussianPath, DataLoader::InputInitStruct& data);
    static void loadExtraGaussianData(std::string& gaussianPath, DataLoader::InputInitStruct& data);
    
    static void loadExtrinsics(std::string dataPath, DataLoader::InputInitStruct& data);
    static void loadIntrinsics(std::string dataPath, DataLoader::InputInitStruct& data);
    
    static void loadInputTo(const std::string _scenepath, DataLoader::InputInitStruct& data);
    static void loadInput12(const std::string _scenepath, DataLoader::InputInitStruct& data, bool& resumed, int height = 640, int width = 480);
    static void createEnlargedFace(const std::string scenepath, DataLoader::InputInitStruct& data, 
                                   const std::vector<FaceProcessingInfo>& faceInfos,
                                   const std::pair<int, int>& selectedIndices);
    
    static std::vector<cv::Point> projectHeadCenter(DataLoader::InputInitStruct& data);
    
    static void saveToPLY_tinyply(const std::string& filename, const DataLoader::InputInitStruct& data, bool save_head = false);
    static void printPlyHeader(const std::string& filepath);
    
    static void saveCameras(const std::string fullDataPath, const DataLoader::InputInitStruct& data);
    
};
