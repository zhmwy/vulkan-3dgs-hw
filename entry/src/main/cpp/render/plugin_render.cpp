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