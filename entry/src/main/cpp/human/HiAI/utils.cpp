#include "utils.h"
#include "HIAIModelManager.h"

// Modified LoadModel to use direct file access
OH_NN_ReturnCode LoadModel(const std::string& modelPath) {
    FILE* modelFile = fopen(modelPath.c_str(), "rb");
    if (!modelFile) {
        OH_LOG_ERROR(LOG_APP, "Failed to open model file at %{public}s", modelPath.c_str());
        return OH_NN_FAILED;
    }

    fseek(modelFile, 0, SEEK_END);
    long modelSize = ftell(modelFile);
    fseek(modelFile, 0, SEEK_SET);

    std::unique_ptr<uint8_t[]> modelData = std::make_unique<uint8_t[]>(modelSize);
    size_t bytesRead = fread(modelData.get(), 1, modelSize, modelFile);
    fclose(modelFile);

    if (bytesRead != static_cast<size_t>(modelSize)) {
        OH_LOG_ERROR(LOG_APP, "Incomplete model read");
        return OH_NN_FAILED;
    }

    return HIAIModelManager::GetInstance().LoadModelFromBuffer(modelData.get(), modelSize);
}

// Function to run the model
OH_NN_ReturnCode RunModel() {
    OH_NN_ReturnCode ret = HIAIModelManager::GetInstance().RunModel();
    if (ret != OH_NN_SUCCESS) {
        OH_LOG_ERROR(LOG_APP, "RunModel failed");
        return ret;
    }

    OH_LOG_INFO(LOG_APP, "RunModel success");
    return ret;
}
