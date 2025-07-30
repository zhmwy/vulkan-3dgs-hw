#include "human.h"
#include "info.h"

int RATIO;
int WIDTH;
int HEIGHT;

void GetImageInfo(std::string imgPath){
    cv::Mat origImg = cv::imread(imgPath);
    WIDTH = origImg.cols;
    HEIGHT = origImg.rows;
    RATIO = HEIGHT / 640.0;
    assert(RATIO == WIDTH / 480.0);
}

bool HumanPreprocess(std::string scenepath, DataLoader::InputInitStruct& data) {
    // Execute core logic
    auto begin = clk::now();
    GetImageInfo(data.img_list[0]);
    
    int isHumanCentric = ProcessKeypoints(scenepath);
    if(isHumanCentric){
        OH_LOG_WARN(LOG_APP, "Dataset is human centric. Default to 3DGS on human.");
        std::map<std::string, double> faceAngles = ProcessTriangulation(scenepath);
        
        FitSMPL(scenepath, data.head_bbox, data.body_bbox);
        OH_LOG_INFO(LOG_APP, "SMPL estimation finished.");
                    
        // Calculate face centers
        std::vector<cv::Point> faceCenters = projectHeadCenter(data);
            
        // Analyze faces without requiring saved images
        std::vector<FaceProcessingInfo> faceInfos = analyzeFaces(faceAngles, faceCenters, data.img_list);
            
        // Select two images based on analysis
        std::pair<int, int> selectedIndices = selectTwoImageIndices(faceInfos);
        
        // Detect face region
        detectFace(scenepath, faceInfos, selectedIndices);
        if (selectedIndices.first != -1 && selectedIndices.second != -1) {
            // Process faces and create enlarged faces
            data.max_movement = (data.body_bbox[3] - data.body_bbox[2]) * 0.01;
            createEnlargedFace(scenepath, data, faceInfos, selectedIndices);
            ProcessLandmarks(scenepath, selectedIndices, faceInfos);
        }
        
        LOGW("Human preprocessing takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        return true;
    }
    else{
        OH_LOG_WARN(LOG_APP, "Dataset is not human centric. Default to 3DGS on scene.");
        return false;
    }
}