//
// Created on 2025/3/6.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "main.h"

std::mutex PluginRender::m_mutex;
std::condition_variable PluginRender::m_con;
bool PluginRender::stop = false;
std::unordered_map<std::string, PluginRender *> PluginRender::m_instance;
OH_NativeXComponent_Callback PluginRender::m_callback;

napi_threadsafe_function g_progress_cb_train = nullptr;

void CallJsProgressTrain(napi_env env, napi_value js_cb, void* context, void* data) {
    int progress = *static_cast<int*>(data);
    napi_value undefined, argv[1];
    napi_get_undefined(env, &undefined);
    napi_create_int32(env, progress, &argv[0]);
    napi_call_function(env, undefined, js_cb, 1, argv, nullptr);
    delete static_cast<int*>(data);
}

void ReportProgressTrain(int progress) {
    if (g_progress_cb_train) {
        int* data = new int(progress);
        napi_status status = napi_acquire_threadsafe_function(g_progress_cb_train);
        napi_call_threadsafe_function(g_progress_cb_train, data, napi_tsfn_blocking);
        status = napi_release_threadsafe_function(g_progress_cb_train, napi_tsfn_release);
    }
}

struct TrainingConfig {
    napi_async_work work;
    napi_deferred deferred;
    NativeResourceManager* resourceMgr;
    std::string dataFolder;
    int max_steps=2000;
    int result=0;
};
#if 0
void ExecuteTrainWork(napi_env env, void* data) {
    TrainingConfig* config = static_cast<TrainingConfig*>(data);
    VulkanComputeExample* vulkanExample = new VulkanComputeExample();
    vulkanExample->loadTestData(config->dataFolder);
    vulkanExample->initVulkan();

    const int max_steps = 2000;
    const int max_views = 157;
    
    for (int step = 0; step < max_steps; ++step) {
        if (!vulkanExample->prepare()) {
            LOGE("Preparation failed at step %d", step);
            break;
        }
        vulkanExample->SetMethod(0);
        vulkanExample->renderLoop();
        int progress = (step * 100) / max_steps;
        ReportProgressTrain(progress);
    }
#if 0
    vulkanExample->saveGaussians();
    vulkanExample->trainign_step = 0;

    std::string resumePath = "/assets/case2";
    vulkanExample->loadTestData(resumePath, true);
    vulkanExample->encodeCpu2Gpu(true);
    vulkanExample->ComputeMemoryBarrierBlock();
    vulkanExample->draw();
#endif
    for (int view = 0; view < max_views; ++view) {
        vulkanExample->renderRGB2Cpu(view);
    }
    LOGI("Training finished!");
    
    config->result = 0;
}
#endif 

void Render(napi_env env, void* cfg){
    TrainingConfig* config = static_cast<TrainingConfig*>(cfg);
    
    VulkanRender* vulkanExample = new VulkanRender();
//     std::string ply_path = config->dataFolder + "/point_cloud.ply";
    vulkanExample->loadTestData(config->dataFolder );
    vulkanExample->hasKNN = true;
    vulkanExample->initVulkan();
    vulkanExample->prepare();
    vulkanExample->initBuffersValues();
//     vulkanExample->render();
    
    for (int i = 0; i < 100; i++){
        vulkanExample->renderImage(i);
        vulkanExample->renderRGB2Cpu(i);
    }
}

#if 1
void ExecuteTrainWork(napi_env env, void* cfg) {
    TrainingConfig* config = static_cast<TrainingConfig*>(cfg);
    VulkanRender* vulkanExample = new VulkanRender();
    vulkanExample->loadTestData(config->dataFolder);
//     vulkanExample->loadTestData();
    
    vulkanExample->initVulkan();
    vulkanExample->prepare();
    vulkanExample->initBuffersValues();
    
    for (int step = 0; step < config->max_steps; ++step) {
        vulkanExample->render();
//         vulkanExample->renderImage(step);
//         vulkanExample->renderRGB2Cpu(step);
        int progress = (step * 100) / config->max_steps;
        ReportProgressTrain(progress);
    }
#if 0
    //人脸重点优化部分
    vulkanExample->saveGaussians();
    vulkanExample->training_step = 0;

    std::string resumePath = "/assets/case2";
    vulkanExample->loadTestData(resumePath, true);
    vulkanExample->encodeCpu2Gpu(true);
    vulkanExample->ComputeMemoryBarrierBlock();
    vulkanExample->executeComputeCommand();
#endif
    for (int i = 0; i < 100; i++){
        vulkanExample->renderImage(i);
        vulkanExample->renderRGB2Cpu(i);
    }
    config->result = 0;
}
#endif
void CompleteTrainWork(napi_env env, napi_status status, void* cfg) {
    TrainingConfig* config = static_cast<TrainingConfig*>(cfg);
    napi_value result;
    napi_create_int32(env, config->result, &result);
    napi_resolve_deferred(env, config->deferred, result);

    napi_delete_async_work(env, config->work);
    OH_ResourceManager_ReleaseNativeResourceManager(config->resourceMgr);
    
    delete config;
}

napi_value Train3DGS(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    TrainingConfig* config = new TrainingConfig();
    config->resourceMgr = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);

    char datafolderBuf[256];
    size_t datafolderSize;
    napi_get_value_string_utf8(env, argv[1], datafolderBuf, sizeof(datafolderBuf), &datafolderSize);
    config->dataFolder = std::string(datafolderBuf, datafolderSize);
    config->max_steps = 2000;
    
    FileOperator::GetInstance()->InitEnv(env);
    FileOperator::GetInstance()->mgr_ = config->resourceMgr;
    FileOperator::GetInstance()->CopyRawDir("");

    napi_value promise;
    napi_create_promise(env, &config->deferred, &promise);
    // ... (Create and queue async work)
    napi_value work_name;
    napi_create_string_utf8(env, "Train3DGSWork", NAPI_AUTO_LENGTH, &work_name);
    napi_create_async_work(env, nullptr, work_name,
                          ExecuteTrainWork,
                          CompleteTrainWork,
                          config,
                          &config->work);
    napi_queue_async_work(env, config->work);
    return promise;
}

napi_threadsafe_function g_progress_cb = nullptr;
std::queue<int> progress_queue;
std::mutex queue_mutex;

napi_value InitProgress(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value resource_name;
    napi_status status;

    // Create resource name string first
    status = napi_create_string_utf8(env, "Progress", NAPI_AUTO_LENGTH, &resource_name);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create resource name");
        return nullptr;
    }

    // Get the callback function
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    // Create threadsafe function with proper parameters
    status = napi_create_threadsafe_function(env,
        args[0],        // JavaScript callback
        nullptr,        // async resource
        resource_name,  // async resource name
        0,              // max queue size (0 = unlimited)
        1,              // initial thread count
        nullptr,        // thread finalize data
        nullptr,        // thread finalize callback
        nullptr,        // context
        CallJsProgressTrain, // call_js_cb
        &g_progress_cb_train  // result
    );

    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create threadsafe function");
        return nullptr;
    }

    return nullptr;
}
