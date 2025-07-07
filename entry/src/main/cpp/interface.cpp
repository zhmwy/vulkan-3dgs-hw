#include "interface.h"
#include "common/common.h"
#include "file/file_operator.h"
#include "render/model_compute.h"

#include <multimedia/image_framework/image_source_mdk.h>
#include <multimedia/image_framework/image_pixel_map_mdk.h>
 


class InterfaceImplVulkan : public InterfaceVulkan {
private:
    std::string _dataPath;
    std::string _outputPath;
    float _progress;
    bool _isInitialized;
    bool _isTraining;
    napi_threadsafe_function _g_progress_cb_train;
    void updateProgress(float value);
    
    std::unique_ptr<VulkanComputeGS> _renderer = NULL;

public:
    InterfaceImplVulkan();
    ~InterfaceImplVulkan();
    void InitVulkan(std::string datapath, std::string outputpath, napi_threadsafe_function g_progress_cb_train);
    void TrainVulkan();
    float GetProcessVulkan();
};

InterfaceImplVulkan::InterfaceImplVulkan() : _progress(0.0f), _isInitialized(false), _isTraining(false), _g_progress_cb_train(nullptr) {}

InterfaceImplVulkan::~InterfaceImplVulkan() {
    if (_isTraining) {
        _isTraining = false;
    }
    
}

void InterfaceImplVulkan::InitVulkan(std::string datapath, std::string outputpath, napi_threadsafe_function g_progress_cb_train) {
    _dataPath = datapath;
    _outputPath = outputpath;
    _g_progress_cb_train = g_progress_cb_train;
    // Reset progress
    updateProgress(0.0f);

    _renderer = std::make_unique<VulkanComputeGS>();
    _renderer->loadTestData(_dataPath);

    _renderer->initVulkan();
    _renderer->prepare();
    _renderer->initBuffersValues();
    _isInitialized = true;
}

void InterfaceImplVulkan::TrainVulkan() {
    if (!_isInitialized || _isTraining) return;
        
    _isTraining = true;
    GetProcessVulkan();

    try {
        std::vector<int> max_steps = {1499, 500, 500};
        
        // Main training loop
        for (int step = 0; step < max_steps[0]; ++step) {
            if (!_isTraining) break;
            _renderer->render();
            updateProgress(static_cast<float>(step * 100) / max_steps[0]);
            GetProcessVulkan();
        }
        
        _renderer->saveCameras(_dataPath);
        _renderer->saveGaussians(_dataPath);
        
        if (_isTraining) {
            // Post-training operations
            _renderer->total_psnr = 0;
            
            for (size_t i = 0; i < _renderer->inputGaussian.img_list.size(); i++) {
                _renderer->renderImage(i);
                _renderer->renderRGB2Cpu(i, _dataPath);
            }
            
            LOGI("PSNR: %{public}f", _renderer->total_psnr / _renderer->inputGaussian.img_list.size());
            LOGI("PSNR_mask: %{public}f", _renderer->total_psnr_mask / _renderer->total_psnr_mask_num);
            
            updateProgress(100.0f);
            GetProcessVulkan();
        }
    } catch (...) {
        _isTraining = false;
    }
    _isTraining = false;
}

void InterfaceImplVulkan::updateProgress(float value){
    _progress = value;
}

float InterfaceImplVulkan::GetProcessVulkan() {
    if (_g_progress_cb_train) {
        //int* data = new int(_progress);
        napi_status status = napi_acquire_threadsafe_function(_g_progress_cb_train);
        napi_call_threadsafe_function(_g_progress_cb_train, &_progress, napi_tsfn_blocking);
        status = napi_release_threadsafe_function(_g_progress_cb_train, napi_tsfn_release);
    }
    return _progress;
}
// 训练次数 / 总次数 * 100%

extern "C" InterfaceVulkan* createInstanceVulkan() {
    return new InterfaceImplVulkan();
}

extern "C" void destoryInstanceVulkan(InterfaceVulkan* instance) {
    if (instance) {
        delete instance;
    }
}