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

    std::vector<int> max_steps={1, 500, 500};

    int result=0;
    bool focus_on_face = false;
};


void Render(napi_env env, void* cfg){
    TrainingConfig* config = static_cast<TrainingConfig*>(cfg);
    
    VulkanRender* vulkanExample = new VulkanRender();
//     std::string ply_path = config->dataFolder + "/point_cloud.ply";
    vulkanExample->loadTestData(config->dataFolder);
    vulkanExample->hasKNN = true;
    vulkanExample->initVulkan();
    vulkanExample->prepare();
    vulkanExample->initBuffersValues();
//     vulkanExample->render();
    
    for (int i = 0; i < 100; i++){
        vulkanExample->renderImage(i);
        vulkanExample->renderRGB2Cpu(i, config->dataFolder);
    }
}

void ExecuteTrainWork(napi_env env, void* cfg) {
    TrainingConfig* config = static_cast<TrainingConfig*>(cfg);
    VulkanRender* InitlTrainer = new VulkanRender();
    InitlTrainer->loadTestData(config->dataFolder);
    std::string sandbox_basepath = InitlTrainer->basepath;
    
    InitlTrainer->initVulkan();
    InitlTrainer->prepare();
    InitlTrainer->initBuffersValues();
    auto begin = clk::now();
    
    for (int step = 0; step < config->max_steps[0]; ++step) {
        InitlTrainer->render();

        int progress = (step * 100) / config->max_steps[0];
        ReportProgressTrain(progress);
    }

//     InitlTrainer->saveGaussians("/123/");
    InitlTrainer->saveCameras("/123/");
    
    
    LOGW("Step 1 takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
    for (int i = 0; i < InitlTrainer->inputGaussian.img_list.size(); i++){

        InitlTrainer->renderImage(i);
        InitlTrainer->renderRGB2Cpu(i, config->dataFolder);
    }
    if(config->focus_on_face){
        begin = clk::now();
        InitlTrainer->saveGaussians(config->dataFolder, true);
        InitlTrainer->training_step = 0;
    
        InitlTrainer->loadTestData(config->dataFolder, LOAD_HEAD);
        InitlTrainer->encodeCpu2Gpu();
        InitlTrainer->ComputeMemoryBarrierBlock();
        InitlTrainer->executeComputeCommand();
        LOGW("Saving and loading takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        begin = clk::now();
        for (int i = 0; i < 114; i++){
            InitlTrainer->renderImage(i);
            InitlTrainer->renderRGB2Cpu(i, config->dataFolder);
        }
        LOGW("Rendering takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        begin = clk::now();
        int width, height;
        manageHeadMasks(sandbox_basepath + config->dataFolder, width, height, InitlTrainer->inputGaussian.intr_tmp[0]);
        LOGW("Processing head masks takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        std::string head_folder = config->dataFolder + "/head";
        LOGI("Loading new intrinsics from head folder: %{public}s", head_folder.c_str());
//         delete vulkanExample;
        
        // Step 2: Train head gaussians on higher resolution.
        VulkanRender* HeadTrainer = new VulkanRender();
        HeadTrainer->low_resolution_steps = 0;
        HeadTrainer->loadTestData(head_folder);
        HeadTrainer->initVulkan();
        HeadTrainer->prepare();
        HeadTrainer->initBuffersValues();
        begin = clk::now();
        
        for (int step = 0; step < config->max_steps[1]; ++step) {
            HeadTrainer->render();
//             HeadTrainer->renderImage(step);
//             HeadTrainer->renderRGB2Cpu(step, config->dataFolder, width, height);
            int progress = (step * 100) / config->max_steps[1];
            ReportProgressTrain(progress);
        }
        LOGW("Step 2 takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        for (int i = 0; i < 114; i++){
            HeadTrainer->renderImage(i);
            HeadTrainer->renderRGB2Cpu(i, config->dataFolder,width, height);
        }
        HeadTrainer->saveGaussians(config->dataFolder, false);
        
        // Step 3: Put head gaussians back and train the full scene again.
        VulkanRender* FullTrainer = new VulkanRender();
        FullTrainer->low_resolution_steps = 0;
        FullTrainer->loadTestData(config->dataFolder, LOAD_EXTRA);
        FullTrainer->initVulkan();
        FullTrainer->prepare();
        FullTrainer->initBuffersValues();
        
        begin = clk::now();
        
        for (int step = 0; step < config->max_steps[2]; ++step) {
            FullTrainer->render();
//             FullTrainer->renderImage(step);
//             FullTrainer->renderRGB2Cpu(step);
            int progress = (step * 100) / config->max_steps[2];
            ReportProgressTrain(progress);
        }
        LOGW("Step 3 takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        for (int i = 0; i < 100; i++){
            FullTrainer->renderImage(i);
            FullTrainer->renderRGB2Cpu(i, config->dataFolder);
        }
    }

    config->result = 0;
}

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
