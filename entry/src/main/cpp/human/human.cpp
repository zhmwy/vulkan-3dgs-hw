#include "human.h"

bool HumanPreprocess(std::string inputPath, std::vector<float>& head_bbox, std::vector<float>& body_bbox) {
    // Execute core logic
    auto begin = clk::now();
    
    int isHumanCentric = ProcessKeypoints(inputPath);
    if(isHumanCentric){
        OH_LOG_WARN(LOG_APP, "Dataset is human centric. Default to 3DGS on human.");
        double reprojError = ProcessTriangulation(inputPath);
        FitSMPL(inputPath, head_bbox, body_bbox);
        OH_LOG_INFO(LOG_APP, "SMPL estimation finished.");
                
        LOGW("Human preprocessing takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        return true;
    }
    else{
        OH_LOG_WARN(LOG_APP, "Dataset is not human centric. Default to 3DGS on scene.");
        return false;
    }
}