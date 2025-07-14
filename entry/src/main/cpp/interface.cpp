#include "interface.h"
#include "common/common.h"
#include "file/file_operator.h"

#include "human/head_mask/manageHeadMasks.h"
#include "render/model_compute.h"

#include <multimedia/image_framework/image_source_mdk.h>
#include <multimedia/image_framework/image_pixel_map_mdk.h>
#include <chrono>

VulkanComputeGS* _renderer = NULL;

InterfaceImplVulkan::InterfaceImplVulkan() : _progress(0.0f), _isInitialized(false), _isTraining(false), _g_progress_cb_train(nullptr) {}

InterfaceImplVulkan::~InterfaceImplVulkan() {
    if (_isTraining) {
        _isTraining = false;
    }
    
    if (_renderer) {
        delete _renderer;
        _renderer = nullptr;
    }
}

void InterfaceImplVulkan::InitVulkan(std::string datapath, std::string outputpath, napi_threadsafe_function g_progress_cb_train) {
    _dataPath = datapath;
    _outputPath = outputpath;
    _g_progress_cb_train = g_progress_cb_train;
    // Reset progress
    updateProgress(0.0f);

    _renderer = new VulkanComputeGS();
    _renderer->loadTestData(_dataPath);
    std::string sandbox_basepath = _renderer->basepath;

    _renderer->initVulkan();
    _renderer->prepare();
    _renderer->initBuffersValues();
    _isInitialized = true;
}

void InterfaceImplVulkan::TrainVulkan() {
    if (!_isInitialized || _isTraining) return;
        
    _isTraining = true;
//     GetProcessVulkan();

    try {
        std::vector<int> max_steps = {1500, 500, 500};
        
        double total_time = 120.0;
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        int step = 0;
        
        while(duration.count() < total_time) {
            if (!_isTraining) break;
            _renderer->render();
            updateProgress(static_cast<float>(duration.count() * 100) / total_time);
            GetProcessVulkan();
            step++;
            
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            
            // do not densify in the last 5 seconds
            if (duration.count() > total_time - 5.0){
                _renderer->densify_start_step = 100000;
            }
        }
        
//         // Main training loop
//         for (int step = 0; step < max_steps[0]; ++step) {
//             if (!_isTraining) break;
//             _renderer->render();
//             updateProgress(static_cast<float>(step * 100) / max_steps[0]);
//             GetProcessVulkan();
//         }
        
//         _renderer->saveCameras(_dataPath);
        _renderer->saveGaussians(_dataPath);
        
        if (_isTraining) {
            // Post-training operations
            _renderer->total_psnr = 0;
            
            for (size_t i = 0; i < _renderer->inputGaussian.img_list.size(); i++) {
                _renderer->renderImage(i);
                _renderer->renderRGB2Cpu(i, _dataPath, WIDTH, HEIGHT);
            }
            
            LOGE("PSNR: %{public}f ; steps: %{public}d", _renderer->total_psnr / _renderer->inputGaussian.img_list.size(), step);
//             LOGI("PSNR_mask: %{public}f", _renderer->total_psnr_mask / _renderer->total_psnr_mask_num);
            
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
        int* data = new int(_progress);
        napi_status status = napi_acquire_threadsafe_function(_g_progress_cb_train);
        napi_call_threadsafe_function(_g_progress_cb_train, data, napi_tsfn_blocking);
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