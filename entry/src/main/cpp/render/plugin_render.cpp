/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <string>
#include <unistd.h>
#include "common/common.h"
#include "file/file_operator.h"
#include "plugin_render.h"
#include "queue"

#if 0
struct ThreadData {
    napi_env env;
    napi_threadsafe_function tsfn;
    std::string dataFolder;
    NativeResourceManager* resourceMgr;
};

void TrainingThread(ThreadData* data) {
    FileOperator::GetInstance()->mgr_ = data->resourceMgr;
    FileOperator::GetInstance()->CopyRawDir("");

    VulkanComputeExample* vulkanExample = new VulkanComputeExample();
    vulkanExample->loadTestData(data->dataFolder);
    vulkanExample->initVulkan();

    for (int step = 0; step < 1000; step++) {
        if (!vulkanExample->prepare()) {
            LOGE("Preparation failed at step %d", step);
            break;
        }
        LOGI("Starting the %{public}d step.", step);
        vulkanExample->SetMethod(0);
        vulkanExample->renderLoop();
        LOGI("Render Loop exited");
        // Report progress
        napi_value progress;
        napi_create_int32(data->env, step + 1, &progress);
        napi_call_threadsafe_function(data->tsfn, progress, napi_tsfn_blocking);
    }

    delete vulkanExample;
    napi_release_threadsafe_function(data->tsfn, napi_tsfn_release);
    delete data;
}

napi_value Train3DGS(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    // Get ResourceManager and data folder
    NativeResourceManager* resourceMgr = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);
    char datafolderBuf[256];
    size_t datafolderSize;
    napi_get_value_string_utf8(env, argv[1], datafolderBuf, sizeof(datafolderBuf), &datafolderSize);
    std::string dataFolder(datafolderBuf, datafolderSize);

    // Create threadsafe function
    napi_threadsafe_function tsfn;
    napi_create_threadsafe_function(
        env,
        argv[2],    // JavaScript callback
        nullptr,    // Optional async name
        nullptr,    // Resource name (deprecated)
        0,          // Unlimited queue size
        1,          // Initial thread count
        nullptr,    // Finalize data
        nullptr,    // Finalize callback
        nullptr,    // Context
        // Correctly typed callback wrapper
        [](napi_env env, napi_value js_cb, void* context, void* data) {
            napi_handle_scope scope;
            napi_open_handle_scope(env, &scope);
        
            // Validate callback type
            napi_valuetype cb_type;
            napi_typeof(env, js_cb, &cb_type);
            if (cb_type != napi_function) {
                OH_LOG_ERROR(LOG_APP, "Invalid callback type: %{public}d", cb_type);
                napi_close_handle_scope(env, scope);
                return;
            }
        
            int* progress = static_cast<int*>(data);
            LOGI("Dispatching progress: %{public}d", *progress);
        
            // Create progress value
            napi_value progress_val;
            napi_status create_status = napi_create_int32(env, *progress, &progress_val);
            delete progress;
            if (create_status != napi_ok) {
                OH_LOG_ERROR(LOG_APP, "Create int failed: %{public}d", create_status);
                napi_close_handle_scope(env, scope);
                return;
            }
        
            // Call JS function
            napi_value undefined;
            napi_get_undefined(env, &undefined);
            
            napi_value result;
            napi_status call_status = napi_call_function(
                env,
                undefined,       // recv
                js_cb,           // function
                1,               // argc
                &progress_val,   // argv
                &result          // result
            );
            
            if (call_status != napi_ok) {
                OH_LOG_ERROR(LOG_APP, "Call failed: %{public}d", call_status);
            }
        
            napi_close_handle_scope(env, scope);
        },
        &tsfn
    );

    // Start training thread
    ThreadData* threadData = new ThreadData{env, tsfn, dataFolder, resourceMgr};
    std::thread thr(TrainingThread, threadData);
    thr.detach();

    napi_value result;
    napi_create_int32(env, 0, &result);
    return result;
}

napi_value Train3DGS(napi_env env, napi_callback_info info){
    size_t argc = 2;
    napi_value argv[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    
    NativeResourceManager* resourceMgr = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);
    if (resourceMgr == nullptr) {
        OH_LOG_ERROR(LOG_APP, "OH_ResourceManager_InitNativeResourceManager failed");
        napi_value errorCode;
        napi_create_int32(env, -1, &errorCode);
        return errorCode;
    }
    
    char datafolderBuf[256];
    size_t datafolderSize;
    napi_get_value_string_utf8(env, argv[1], datafolderBuf, sizeof(datafolderBuf), &datafolderSize);
    std::string dataFolder(datafolderBuf, datafolderSize);
    
    FileOperator::GetInstance()->InitEnv(env);
    FileOperator::GetInstance()->mgr_ = resourceMgr;
    FileOperator::GetInstance()->CopyRawDir("");
    
    VulkanComputeExample* vulkanExample= new VulkanComputeExample();
    vulkanExample->loadTestData(dataFolder);
    vulkanExample->initVulkan();
    //vulkanExample->setupWindow(hInstance, WndProc)
    while (true) {
        bool prepared = vulkanExample->prepare();
        if (!prepared) {
            LOGE("vulkan example is not prepared");
            break;
        }
        vulkanExample->SetMethod(0);
        vulkanExample->renderLoop();
    }
    delete(vulkanExample);
}

void OnSurfaceCreatedCB(OH_NativeXComponent *component, void *window)
{
    LOGI("PluginRender OnSurfaceCrateCB");
    if ((nullptr == component) || (nullptr == window)) {
        LOGE("PluginRender OnSurfaceCreatedCB failed: component or window is null");
        return;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = { '\0' };
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NATIVEXCOMPONENT_RESULT_SUCCESS != OH_NativeXComponent_GetXComponentId(component, idStr, &idSize)) {
        LOGE("PluginRender OnSurfaceCreatedCB failed: Unable to get XComponent id");
        return;
    }

    std::string id(idStr);
    auto render = PluginRender::GetInstance(id);
    render->RenderCreated(component, window);
}

void PluginRender::RenderCreated(OH_NativeXComponent *component, void *window)
{
    uint64_t width;
    uint64_t height;
    int32_t ret = OH_NativeXComponent_GetXComponentSize(component, window, &width, &height);
    LOGD("PluginRender OnSurfaceCreated ret is %{public}d, w:%{public}lu, d:%{public}lu", ret, width, height);
    m_vulkanexample->loadTestData();
    if (!m_vulkanexample->initVulkan()) {
        LOGE("PluginRender OnSurfaceCreated vulkanExample initVulkan FALSE");
        return;
    }
    m_vulkanexample->setupWindow(static_cast<OHNativeWindow *>(window));

    m_renderThread = std::thread(std::bind(&PluginRender::RenderThread, this));
}

void PluginRender::RenderThread()
{
    LOGI("RenderThread started");
    while (true) {
        std::unique_lock<std::mutex> locker(m_mutex);
        if (!stop) {
            bool prepared = m_vulkanexample->prepare();
            if (!prepared) {
                LOGE("vulkan example is not prepared");
                break;
            }
            m_vulkanexample->SetMethod(m_mode);
            m_vulkanexample->renderLoop();
        } else {
            LOGI("PluginRender Render Thread stop ");
            break;
        }
    }
}

void OnSurfaceChangedCB(OH_NativeXComponent *component, void *window) {}

void OnSurfaceDestroyedCB(OH_NativeXComponent *component, void *window)
{
    LOGI("PluginRender OnSurfaceDestroyedCB");
    if ((nullptr == component) || (nullptr == window)) {
        LOGE("PluginRender OnSurfaceDestroyedCB : component or window is null");
        return;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = { '\0' };
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NATIVEXCOMPONENT_RESULT_SUCCESS != OH_NativeXComponent_GetXComponentId(component, idStr, &idSize)) {
        LOGE("PluginRender OnSurfaceDestroyedCB : Unable to get XComponent id");
        return;
    }

    std::string id(idStr);
    PluginRender::Release(id);
}

void DispatchTouchEventCB(OH_NativeXComponent *component, void *window) {}

void PluginRender::SetMode(int mode)
{
    m_mode = mode;
}

napi_value PluginRender::SetUpscaleMethod(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    int upscaleMode;
    napi_get_value_int32(env, args[0], &upscaleMode);
    LOGI("PluginRender::Add get params is %{public}d", upscaleMode);

    if ((nullptr == env) || (nullptr == info)) {
        LOGE("PluginRender SetUpscaleMethod : env or info is null");
        return nullptr;
    }

    napi_value thisArg;
    if (napi_ok != napi_get_cb_info(env, info, nullptr, nullptr, &thisArg, nullptr)) {
        LOGE("PluginRender SetUpscaleMethod : napi_get_cb_info fail");
        return nullptr;
    }

    napi_value exportInstance;
    if (napi_ok != napi_get_named_property(env, thisArg, OH_NATIVE_XCOMPONENT_OBJ, &exportInstance)) {
        LOGE("PluginRender SetUpscaleMethod : napi_get_named_property fail");
        return nullptr;
    }

    OH_NativeXComponent *nativeXComponent = nullptr;
    if (napi_ok != napi_unwrap(env, exportInstance, reinterpret_cast<void **>(&nativeXComponent))) {
        LOGE("PluginRender SetUpscaleMethod : napi_unwrap fail");
        return nullptr;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NATIVEXCOMPONENT_RESULT_SUCCESS != OH_NativeXComponent_GetXComponentId(nativeXComponent, idStr, &idSize)) {
        LOGE("PluginRender SetUpscaleMethod : Unable to get XComponent id");
        return nullptr;
    }
    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
    if (render) {
        render->SetMode(upscaleMode);
    }
    return nullptr;
}

napi_value PluginRender::SetVRSUsed(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    bool useVRS;
    napi_get_value_bool(env, args[0], &useVRS);
    LOGI("PluginRender::SetVRSUsed get params is %{public}d", useVRS);

    if ((nullptr == env) || (nullptr == info)) {
        LOGE("PluginRender SetVRSUsed : env or info is null");
        return nullptr;
    }

    napi_value thisArg;
    if (napi_ok != napi_get_cb_info(env, info, nullptr, nullptr, &thisArg, nullptr)) {
        LOGE("PluginRender SetVRSUsed : napi_get_cb_info fail");
        return nullptr;
    }

    napi_value exportInstance;
    if (napi_ok != napi_get_named_property(env, thisArg, OH_NATIVE_XCOMPONENT_OBJ, &exportInstance)) {
        LOGE("PluginRender SetVRSUsed : napi_get_named_property fail");
        return nullptr;
    }

    OH_NativeXComponent *nativeXComponent = nullptr;
    if (napi_ok != napi_unwrap(env, exportInstance, reinterpret_cast<void **>(&nativeXComponent))) {
        LOGE("PluginRender SetVRSUsed : napi_unwrap fail");
        return nullptr;
    }

    char idStr[OH_XCOMPONENT_ID_LEN_MAX + 1] = {'\0'};
    uint64_t idSize = OH_XCOMPONENT_ID_LEN_MAX + 1;
    if (OH_NATIVEXCOMPONENT_RESULT_SUCCESS != OH_NativeXComponent_GetXComponentId(nativeXComponent, idStr, &idSize)) {
        LOGE("PluginRender SetVRSUsed : Unable to get XComponent id");
        return nullptr;
    }
    std::string id(idStr);
    PluginRender *render = PluginRender::GetInstance(id);
//     if (render) {
//         render->m_vulkanexample->UseVRS(useVRS);
//     }
    return nullptr;
}
#endif
PluginRender::PluginRender(std::string &id)
{
    this->m_id = id;
//     this->m_vulkanexample = new VulkanExample();
    this->m_vulkanexample = new VulkanComputeExample();
    OH_NativeXComponent_Callback *renderCallback = &PluginRender::m_callback;
//     renderCallback->OnSurfaceCreated = OnSurfaceCreatedCB;
//     renderCallback->OnSurfaceChanged = OnSurfaceChangedCB;
//     renderCallback->OnSurfaceDestroyed = OnSurfaceDestroyedCB;
//     renderCallback->DispatchTouchEvent = DispatchTouchEventCB;
    LOGI("PluginRender Initialized");
}

PluginRender *PluginRender::GetInstance(std::string &id)
{
    if (m_instance.find(id) == m_instance.end()) {
        PluginRender *instance = new PluginRender(id);
        m_instance[id] = instance;
        return instance;
    } else {
        return m_instance[id];
    }
}
#if 1
void PluginRender::Export(napi_env env, napi_value exports)
{
    LOGI("PluginRender export for %{public}s", m_id.c_str());
    if ((nullptr == env) || (nullptr == exports)) {
        LOGE("PluginRender Export: env or exports is null");
        return;
    }

//     napi_property_descriptor desc[] = {
//         {"setUpscaleMethod", nullptr, PluginRender::SetUpscaleMethod, nullptr, nullptr, nullptr, napi_default, nullptr},
//         {"setVRSUsed", nullptr, PluginRender::SetVRSUsed, nullptr, nullptr, nullptr, napi_default, nullptr}
//     };
//
//     if (napi_ok != napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc)) {
//         LOGE("PluginRender Export: napi_define_properties failed");
//     }
}

void PluginRender::Release(std::string &id)
{
    LOGE("PluginRender start release");
    PluginRender *render = PluginRender::GetInstance(id);
    if (render != nullptr) {
        std::unique_lock<std::mutex> locker(m_mutex);
        stop = true;
        m_con.notify_all();
        m_instance.erase(m_instance.find(id));
        delete render->m_vulkanexample;
    }
}
#endif