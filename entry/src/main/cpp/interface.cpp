#include "interface.h"
#include "common/common.h"
#include "file/file_operator.h"
#include "render/plugin_render.h"
#include "render/vulkanRender.h"
#include "human/head_mask/manageHeadMasks.h"

std::mutex PluginRender::m_mutex;
std::condition_variable PluginRender::m_con;
bool PluginRender::stop = false;
std::unordered_map<std::string, PluginRender*> PluginRender::m_instance;
OH_NativeXComponent_Callback PluginRender::m_callback;

napi_threadsafe_function g_progress_cb_train = nullptr;

VulkanRender* _renderer = NULL;

// Modified InterfaceImpl with progress integration

InterfaceImpl::InterfaceImpl() : _progress(0.0f), _isInitialized(false), _isTraining(false) {}

InterfaceImpl::~InterfaceImpl() {
    if (_isTraining) {
        _isTraining = false;
    }
    
    if (_renderer) {
        delete _renderer;
        _renderer = nullptr;
    }
}

void InterfaceImpl::Init(std::string datapath, std::string outputpath) {
    _dataPath = datapath;
    _outputPath = outputpath;
    
    // Reset progress
    updateProgress(0.0f);

    _renderer = new VulkanRender();
    _renderer->loadTestData(_dataPath);
    std::string sandbox_basepath = _renderer->basepath;

    _renderer->initVulkan();
    _renderer->prepare();
    _renderer->initBuffersValues();
    _isInitialized = true;
}

void InterfaceImpl::Train() {
    if (!_isInitialized || _isTraining) return;
        
    _isTraining = true;
    GetProcess();

    try {
        std::vector<int> max_steps = {1500, 500, 500};
        
        // Main training loop
        for (int step = 0; step < max_steps[0]; ++step) {
            if (!_isTraining) break;
            _renderer->render();
            updateProgress(static_cast<float>(step * 100) / max_steps[0]);
            GetProcess();
        }
//         _renderer->saveCameras(_dataPath);
//         _renderer->saveGaussians(_dataPath);
        
        if (_isTraining) {
            // Post-training operations
//                 _renderer->saveCameras("/123/");
            for (size_t i = 0; i < _renderer->inputGaussian.img_list.size(); i++) {
                if(i >= 101){
                    printf("");
                }
                _renderer->renderImage(i);
                _renderer->renderRGB2Cpu(i, _outputPath);
            }
            updateProgress(100.0f);
            GetProcess();
        }
    } catch (...) {
        _isTraining = false;
    }
    _isTraining = false;
}

void InterfaceImpl::updateProgress(float value){
    _progress = value;
}

float InterfaceImpl::GetProcess() {
    if (g_progress_cb_train) {
        int* data = new int(_progress);
        napi_status status = napi_acquire_threadsafe_function(g_progress_cb_train);
        napi_call_threadsafe_function(g_progress_cb_train, data, napi_tsfn_blocking);
        status = napi_release_threadsafe_function(g_progress_cb_train, napi_tsfn_release);
    }
    return _progress;
}

// Progress reporting functions remain unchanged
void CallJsProgressTrain(napi_env env, napi_value js_cb, void* context, void* data) {
    int progress = *static_cast<int*>(data);
    napi_value undefined, argv[1];
    napi_get_undefined(env, &undefined);
    napi_create_int32(env, progress, &argv[0]);
    napi_call_function(env, undefined, js_cb, 1, argv, nullptr);
    delete static_cast<int*>(data);
}

// void ReportProgressTrain(int progress) {
//     if (g_progress_cb_train) {
//         int* data = new int(progress);
//         napi_status status = napi_acquire_threadsafe_function(g_progress_cb_train);
//         napi_call_threadsafe_function(g_progress_cb_train, data, napi_tsfn_blocking);
//         status = napi_release_threadsafe_function(g_progress_cb_train, napi_tsfn_release);
//     }
// }

struct AsyncContext {
    napi_async_work work;
    napi_deferred deferred;
    NativeResourceManager* resourceMgr;
    InterfaceImpl* impl;
    std::string datapath;
    std::string outputpath;
};

void ExecuteTraining(napi_env env, void* data) {
    AsyncContext* context = static_cast<AsyncContext*>(data);
    context->impl->Init(context->datapath, context->outputpath);
    context->impl->Train();
}

void CompleteTraining(napi_env env, napi_status status, void* data) {
    AsyncContext* context = static_cast<AsyncContext*>(data);
    
    napi_value result;
    napi_create_int32(env, 0, &result);
    napi_resolve_deferred(env, context->deferred, result);
    
    napi_delete_async_work(env, context->work);
    OH_ResourceManager_ReleaseNativeResourceManager(context->resourceMgr);
    destoryInstance(context->impl);
    delete context;
}

napi_value Train3DGS(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    // Initialize resource manager
    NativeResourceManager* resourceMgr = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);

    // Get data path
    char dataPath[256];
    size_t pathLength;
    napi_get_value_string_utf8(env, argv[1], dataPath, sizeof(dataPath), &pathLength);
    std::string dataFolder(dataPath, pathLength);

    // Initialize file operations
    FileOperator::GetInstance()->InitEnv(env);
    FileOperator::GetInstance()->mgr_ = resourceMgr;
    FileOperator::GetInstance()->CopyRawDir("");

    // Create interface instance
    InterfaceImpl* impl = static_cast<InterfaceImpl*>(createInstance());
//     impl->Init(dataFolder, dataFolder);

    // Create async context
    AsyncContext* context = new AsyncContext();
    context->resourceMgr = resourceMgr;
    context->impl = impl;
    context->datapath = dataFolder;
    context->outputpath = dataFolder;

    // Setup promise
    napi_value promise;
    napi_create_promise(env, &context->deferred, &promise);

    // Create async work
    napi_value work_name;
    napi_create_string_utf8(env, "TrainingWork", NAPI_AUTO_LENGTH, &work_name);
    napi_create_async_work(env, nullptr, work_name,
        ExecuteTraining,
        CompleteTraining,
        context,
        &context->work
    );
    napi_queue_async_work(env, context->work);

    return promise;
}

// Threadsafe function initialization remains unchanged
napi_value InitProgress(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value resource_name;
    napi_status status = napi_create_string_utf8(env, "Progress", NAPI_AUTO_LENGTH, &resource_name);
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    status = napi_create_threadsafe_function(env, args[0], nullptr, resource_name,
        0, 1, nullptr, nullptr, nullptr, CallJsProgressTrain, &g_progress_cb_train);

    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create threadsafe function");
    }
    return nullptr;
}

extern "C" Interface* createInstance() {
    return new InterfaceImpl();
}

extern "C" void destoryInstance(Interface* instance) {
    if (instance) {
        delete instance;
    }
}