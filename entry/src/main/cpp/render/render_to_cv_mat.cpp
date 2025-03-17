#include "render_to_cv_mat.h"
   
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
        LOGI("File len is: %{public}zu", len);
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
        LOGE("Rawfile open failed");
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
      LOGI("Reading image from %{public}s", imagePath.c_str());
    
      FILE *fp;
      fp = fopen(imagePath.c_str(), "rb");
      fseek(fp, 0, SEEK_END);
      long len = ftell(fp);
      fseek(fp, 0, SEEK_SET);
      LOGI("File len is: %{public}zu", len);
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
        LOGE("Image file open failed");
    }
      return nullptr;
}