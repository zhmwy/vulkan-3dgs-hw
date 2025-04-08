#include "interface.h"
#include "common/common.h"
#include "file/file_operator.h"
#include "render/plugin_render.h"
#include "render/vulkanRender.h"
#include "human/head_mask/manageHeadMasks.h"

#include <multimedia/image_framework/image_source_mdk.h>
#include <multimedia/image_framework/image_pixel_map_mdk.h>

std::mutex PluginRender::m_mutex;
std::condition_variable PluginRender::m_con;
bool PluginRender::stop = false;
std::unordered_map<std::string, PluginRender*> PluginRender::m_instance;
OH_NativeXComponent_Callback PluginRender::m_callback;
// napi_threadsafe_function g_progress_cb_train = nullptr;
VulkanRender* _renderer = NULL;

/*
napi_value getSyncPixelMap(napi_env env, napi_callback_info info)
{
      size_t argc = 2;
      napi_value args[2] = {nullptr};

      napi_get_cb_info(env, info, &argc, args , nullptr, nullptr);
      
      napi_valuetype srcType;
      napi_typeof(env, args[0], &srcType);

      // 入参args[0]是资源管理器，用来初始化native层的资源管理器
      NativeResourceManager *mNativeResMgr = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
      
      size_t strSize;
      char srcBuf[2048];
      // 入参args[1]是文件名称
      napi_get_value_string_utf8(env, args[1], srcBuf, sizeof(srcBuf), &strSize);
      // 用资源管理器打开Raw文件
      RawFile * rawFile = OH_ResourceManager_OpenRawFile(mNativeResMgr, srcBuf);
      if (rawFile != NULL) {
         // 获取文件大小，并读取数据
         long len = OH_ResourceManager_GetRawFileSize(rawFile);

         uint8_t * data = static_cast<uint8_t *>(malloc(len));
         int res = OH_ResourceManager_ReadRawFile(rawFile, data, len);

         OhosImageSource imageSource_c;
         imageSource_c.buffer = data;
         imageSource_c.bufferSize = len;

         OhosImageSourceOps ops{};
         napi_value imageSource;
         napi_value pixelMap;

         // 用读取到的Raw数据创建ImageSource
         int32_t ret = OH_ImageSource_Create(env, &imageSource_c, &ops, &imageSource);

         // 初始化native层的ImageSource
         ImageSourceNative * imageSourceNative_c = OH_ImageSource_InitNative(env, imageSource);
         OhosImageDecodingOps decodingOps{};
         // 创建pixelMap
         OH_ImageSource_CreatePixelMap(imageSourceNative_c, &decodingOps, &pixelMap);

         // 下列方法,为gif等动图格式提供。
         // napi_value pixelMapList;
         // OH_ImageSource_CreatePixelMapList(imageSourceNative_c, &decodingOps, &pixelMapList);
         // OhosImageSourceDelayTimeList list{};
         // OH_ImageSource_GetDelayTime(imageSourceNative_c, &list);
         // uint32_t count;
         // OH_ImageSource_GetFrameCount(imageSourceNative_c, &count);

         OhosImageSourceInfo info{};
         // 读取图片宽高
         OH_ImageSource_GetImageInfo(imageSourceNative_c, 0, &info);
         OH_LOG_Print(LOG_APP, LOG_INFO, 0xFF00, "[decode]", "imageInfo width:%{public}d , height:%{public}d", info.size.width, info.size.height);
         
         // 读取图片源的ImageWidth配置参数并打印日志
         OhosImageSourceProperty target;
         char exifKey_c[] = "ImageWidth";
         target.size = strlen(exifKey_c);
         target.value = exifKey_c;

         OhosImageSourceProperty response{};
         response.size = 20;
         response.value = static_cast<char *>(malloc(20));
         OH_ImageSource_GetImageProperty(imageSourceNative_c, &target, &response);
         OH_LOG_Print(LOG_APP, LOG_INFO, 0xFF00, "[decode]", "ImageProperty width after modify:%{public}s", response.value);

         // 处理完毕，释放native层资源
         OH_ImageSource_Release(imageSourceNative_c);
         OH_ResourceManager_CloseRawFile(rawFile);
         return pixelMap;
      }
    else{
//         LOGE("Rawfile open failed");
    }
      OH_ResourceManager_ReleaseNativeResourceManager(mNativeResMgr);
      return nullptr;
}

napi_value getSandBoxPixelMap(napi_env env, napi_callback_info info)
{
      size_t argc = 2;
      napi_value args[2] = {nullptr};

      napi_get_cb_info(env, info, &argc, args , nullptr, nullptr);
      
      size_t sandboxPathSize, strSize;
      char sandboxPathBuf[256], srcBuf[256];
      // 入参args[1]是文件名称
      napi_get_value_string_utf8(env, args[0], sandboxPathBuf, sizeof(sandboxPathBuf), &sandboxPathSize);
      napi_get_value_string_utf8(env, args[1], srcBuf, sizeof(srcBuf), &strSize);
      std::string sandboxPath(sandboxPathBuf, sandboxPathSize);
      std::string str(srcBuf, strSize);
      std::string imagePath = sandboxPath + str;

    
      FILE *fp;
      fp = fopen(imagePath.c_str(), "rb");
      fseek(fp, 0, SEEK_END);
      long len = ftell(fp);
      fseek(fp, 0, SEEK_SET);

      uint8_t * data = static_cast<uint8_t *>(malloc(len));
      size_t bytesRead = fread(data, 1, len, fp);
      fclose(fp);
    
      if (bytesRead==len) {
         OhosImageSource imageSource_c;
         imageSource_c.buffer = data;
         imageSource_c.bufferSize = len;

         OhosImageSourceOps ops{};
         napi_value imageSource;
         napi_value pixelMap;

         // 用读取到的Raw数据创建ImageSource
         int32_t ret = OH_ImageSource_Create(env, &imageSource_c, &ops, &imageSource);

         // 初始化native层的ImageSource
         ImageSourceNative * imageSourceNative_c = OH_ImageSource_InitNative(env, imageSource);
         OhosImageDecodingOps decodingOps{};
         // 创建pixelMap
         OH_ImageSource_CreatePixelMap(imageSourceNative_c, &decodingOps, &pixelMap);

         // 下列方法,为gif等动图格式提供。
         // napi_value pixelMapList;
         // OH_ImageSource_CreatePixelMapList(imageSourceNative_c, &decodingOps, &pixelMapList);
         // OhosImageSourceDelayTimeList list{};
         // OH_ImageSource_GetDelayTime(imageSourceNative_c, &list);
         // uint32_t count;
         // OH_ImageSource_GetFrameCount(imageSourceNative_c, &count);

         OhosImageSourceInfo info{};
         // 读取图片宽高
         OH_ImageSource_GetImageInfo(imageSourceNative_c, 0, &info);
         OH_LOG_Print(LOG_APP, LOG_INFO, 0xFF00, "[decode]", "imageInfo width:%{public}d , height:%{public}d", info.size.width, info.size.height);
         
         // 读取图片源的ImageWidth配置参数并打印日志
         OhosImageSourceProperty target;
         char exifKey_c[] = "ImageWidth";
         target.size = strlen(exifKey_c);
         target.value = exifKey_c;

         OhosImageSourceProperty response{};
         response.size = 20;
         response.value = static_cast<char *>(malloc(20));
         OH_ImageSource_GetImageProperty(imageSourceNative_c, &target, &response);
         OH_LOG_Print(LOG_APP, LOG_INFO, 0xFF00, "[decode]", "ImageProperty width after modify:%{public}s", response.value);

         // 处理完毕，释放native层资源
         OH_ImageSource_Release(imageSourceNative_c);
         return pixelMap;
      }
    else{
//         LOGE("Image file open failed");
    }
      return nullptr;
}
*/


InterfaceImpl::InterfaceImpl() : _progress(0.0f), _isInitialized(false), _isTraining(false), _g_progress_cb_train(nullptr) {}

InterfaceImpl::~InterfaceImpl() {
    if (_isTraining) {
        _isTraining = false;
    }
    
    if (_renderer) {
        delete _renderer;
        _renderer = nullptr;
    }
}

void InterfaceImpl::Init(std::string datapath, std::string outputpath, napi_threadsafe_function g_progress_cb_train) {
    _dataPath = datapath;
    _outputPath = outputpath;
    _g_progress_cb_train = g_progress_cb_train;
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
    if (_g_progress_cb_train) {
        int* data = new int(_progress);
        napi_status status = napi_acquire_threadsafe_function(_g_progress_cb_train);
        napi_call_threadsafe_function(_g_progress_cb_train, data, napi_tsfn_blocking);
        status = napi_release_threadsafe_function(_g_progress_cb_train, napi_tsfn_release);
    }
    return _progress;
}


extern "C" Interface* createInstance() {
    return new InterfaceImpl();
}

extern "C" void destoryInstance(Interface* instance) {
    if (instance) {
        delete instance;
    }
}