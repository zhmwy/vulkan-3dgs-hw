#include "interface.h"
#include "common/common.h"
#include "file/file_operator.h"

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

// A. Efficient function to check if segmentation is complete
bool InterfaceImplVulkan::isSegmentationComplete() {
    std::string segmentedPath = _dataPath + "/segmented";
    
    // Check if directory exists
    if (!std::filesystem::exists(segmentedPath)) {
        return false;
    }
    
    // Efficiently count files in the directory
    auto dirIter = std::filesystem::directory_iterator(segmentedPath);
    int fileCount = std::distance(std::filesystem::directory_iterator(segmentedPath), 
                                  std::filesystem::directory_iterator{});
    
    return fileCount >= _renderer->inputGaussian.sequence_length;
}

void InterfaceImplVulkan::InitVulkan(std::string datapath, std::string outputpath, 
                                        int trainingMode, bool isObject, napi_threadsafe_function g_progress_cb_train) {
    _dataPath = datapath;
    _outputPath = outputpath;
    _g_progress_cb_train = g_progress_cb_train;
    // Reset progress
    updateProgress(0.0f);

    _renderer = new VulkanComputeGS();
    _renderer->loadTestData(_dataPath);
    std::string sandbox_basepath = _renderer->basepath;
    
    _renderer->trainingMode = trainingMode;
    _renderer->isObject = isObject;
    
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
        std::vector<int> max_steps = {7000, 500, 500};
        
        double total_time = 60.0;
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        int step = 0;
        
        while(duration.count() < total_time) {
            if (!_isTraining) break;

#if 1
            // Currently performance is not good.
            // Check for segmentation completion and apply if ready (only once)
            if (!_renderer->segmentationFinished && isSegmentationComplete() && _renderer->trainingMode != SCENE) {
                LOGI("Segmentation masks finished!");
//                 _renderer->performSegmentationUpdate(_dataPath);
                _renderer->segmentationFinished = true;
//                 LOGI("Foreground segmentation applied, continuing with foreground-only training.");
            }
#endif       
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

            if (duration.count() > total_time - 15.0){
                _renderer->low_resolution_steps = -1;
            }
        }
        
//         // Main training loop
//         for (step = 0; step < max_steps[0]; ++step) {
//             if (!_isTraining) break;
//                
//             if (!_renderer->segmentationFinished && isSegmentationComplete() && _renderer->trainingMode != SCENE) {
//                 LOGI("Segmentation masks finished!");
// //                 _renderer->performSegmentationUpdate(_dataPath);
//                 _renderer->segmentationFinished = true;
// //                 LOGI("Foreground segmentation applied, continuing with foreground-only training.");
//             }
//                
//             _renderer->render();
//             updateProgress(static_cast<float>(step * 100) / max_steps[0]);
//             GetProcessVulkan();
//         }
//         end = std::chrono::high_resolution_clock::now();
//         duration = end - start;

//         _renderer->saveCameras(_dataPath);
        _renderer->saveGaussians(_dataPath);
        
        if (_isTraining) {
            // Post-training operations
            _renderer->total_psnr = 0;
//             _renderer->inputGaussian.processAndCropForeground(_dataPath);
            
            for (size_t i = 0; i < _renderer->inputGaussian.img_list.size(); i++) {
                _renderer->renderImage(i);
                _renderer->renderRGB2Cpu(i, _dataPath, WIDTH, HEIGHT);
            }
            
            LOGE("PSNR: %{public}f ; steps: %{public}d; times: %{public}f", _renderer->total_psnr / _renderer->inputGaussian.img_list.size(), step, float(duration.count()));
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