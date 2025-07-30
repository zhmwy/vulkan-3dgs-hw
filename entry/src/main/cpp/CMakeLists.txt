# the minimum version of CMake.
cmake_minimum_required(VERSION 3.4.1)
project(XComponent)

set(CMAKE_CXX_STANDARD 17)

set(KOMPUTE_OPT_REPO_SUBMODULE_BUILD 1)
set(DKOMPUTE_OPT_ENABLE_SPDLOG 1)
set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

ADD_DEFINITIONS(-DVK_USE_PLATFORM_OHOS=1)
ADD_DEFINITIONS(-DVK_NO_PROTOTYPES=1)
add_definitions(-DOHOS_PLATFORM)
add_definitions(-DSTB_IMAGE_IMPLEMENTATION)

include_directories(
    ${NATIVERENDER_ROOT_PATH}
    ${NATIVERENDER_ROOT_PATH}/include
)

add_library(3dgs SHARED
    interface.cpp
#    render/plugin_render.cpp
#    render/algorithm/fsr.cpp
#    render/render_to_cv_mat.cpp
#    napi_init.cpp
#    render/vulkanRender.cpp
#    manager/plugin_manager.cpp
    vulkanbase/VulkanOhos.cpp
#    vulkanbase/VulkanBuffer.cpp
    vulkanbase/VulkanDevice.cpp
    vulkanbase/vulkancomputebase.cpp
    vulkanbase/VulkanInitializers.hpp
#    vulkanbase/VulkanSwapChain.cpp
    vulkanbase/VulkanTools.cpp
    vulkanbase/VulkanTexture.cpp
    file/file_operator.cpp
    file/file.cpp
    # render/model_3d_sponza.cpp
#    render/vulkan_obj_model.cpp
#    render/vulkan_obj_mesh.cpp
    render/model_compute.cpp
    cores/buffer.cpp
    cores/cpu_buffer.cpp
    cores/dataloader.cpp
    cores/tinyply.cpp
#    cores/video2images.cpp
    human/face/manageHeadMasks.cpp human/face/beauty_filters.cpp human/face/createEnlargedFace.cpp human/human.cpp
    human/keypoints/keypointsv2.cpp human/keypoints/HIAIModelManager.cpp human/keypoints/utils.cpp
    human/triangulation/triangulationv2.cpp
    human/SMPL/src/fit_smplv2.cpp human/SMPL/src/get_theta.cpp human/SMPL/src/get_theta_from_skeleton.cpp human/SMPL/src/smpl/SMPL.cpp
    human/SMPL/src/smpl/BlendShape.cpp human/SMPL/src/smpl/JointRegression.cpp human/SMPL/src/smpl/WorldTransformation.cpp
    human/SMPL/src/smpl/LinearBlendSkinning.cpp human/SMPL/src/toolbox/Exception.cpp human/SMPL/src/definition/def.cpp
    human/SMPL/src/smpl/smpl_male.cpp
    human/segmentation/segment.cpp human/segmentation/utils.cpp
    human/face/landmark/landmark2d.cpp human/face/landmark/bfm_data.cpp human/face/landmark/landmark3d.cpp human/face/landmark/faceDetect.cpp human/face/landmark/utils.cpp
)


target_include_directories(3dgs PUBLIC
        3rdParty
        3rdParty/glm
        3rdParty/
#        3rdParty/vulkan_radix_sort
        vulkanbase
)

find_library(
    # Sets the name of the path variable.
    hilog-lib
    # Specifies the name of the NDK library that
    # you want CMake to locate.
    hilog_ndk.z
)

find_library(
    # Sets the name of the path variable.
    libace-lib
    # Specifies the name of the NDK library that
    # you want CMake to locate.
    ace_ndk.z
)

find_library(
    # Sets the name of the path variable.
    libnapi-lib
    # Specifies the name of the NDK library that
    # you want CMake to locate.
    ace_napi.z
)

find_library(
    # Sets the name of the path variable.
    libuv-lib
    # Specifies the name of the NDK library that
    # you want CMake to locate.
    uv
)


#add_subdirectory(${NATIVERENDER_ROOT_PATH}/3rdParty/kompute/ ${CMAKE_CURRENT_BINARY_DIR}/kompute)
#add_subdirectory(3rdParty/vulkan_radix_sort)
add_library(libassimp SHARED IMPORTED)
set_target_properties(
    libassimp
    PROPERTIES
    IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/arm64-v8a/libassimp.so
)

find_library(
    # Sets the name of the path variable.
    xengine-lib
    # Specifies the name of the NDK library that
    # you want CMake to locate.
    xengine
)

target_link_libraries(3dgs PUBLIC
 ${hilog-lib} ${libace-lib} ${libnapi-lib} ${libuv-lib} libnative_window.so libc++.a librawfile.z.so libassimp ${xengine-lib})


# Human Part

#target_link_libraries(3dgs PUBLIC ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH}/libsmpl.so)
target_include_directories(3dgs PUBLIC human/SMPL/include)
target_include_directories(3dgs PUBLIC human/segmentation)
FIND_LIBRARY(hiai_foundation-lib hiai_foundation)

target_link_libraries(3dgs PUBLIC libace_napi.z.so
                                   libhilog_ndk.z.so
                                   libohfileio.so
                                   librawfile.z.so
                                   libneural_network_core.so
                                   libneural_network_runtime.so
#                                   mindspore_lite_ndk
                                   ${hiai_foundation-lib}
                                   libpixelmap_ndk.z.so
                                   libimage_source_ndk.z.so)
# OpenCV相关
target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/${OHOS_ARCH}/include/opencv4)
target_link_libraries(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_calib3d.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_core.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_dnn.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_features2d.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_flann.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_gapi.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_highgui.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_imgcodecs.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_imgproc.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_ml.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_objdetect.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_photo.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_stitching.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_video.so.408
                                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_videoio.so.408)

#Eigen和Ceres-solver相关
target_link_libraries(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/${OHOS_ARCH}/lib/libceres.a)
target_link_libraries(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/${OHOS_ARCH}/lib/libgflags_nothreads.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/${OHOS_ARCH}/lib/libgflags.a)
target_link_libraries(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/GKlib/${OHOS_ARCH}/lib/libGKlib.a)

target_link_libraries(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgmock_main.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgmock.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgtest_main.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgtest.a)

target_link_libraries(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/METIS/${OHOS_ARCH}/lib/libmetis.a)
#target_link_libraries(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/glog/libglog.so.1)

target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/${OHOS_ARCH}/include)
target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen/${OHOS_ARCH}/include/eigen3)
target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/${OHOS_ARCH}/include)
target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/GKlib/${OHOS_ARCH}/include)
#target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glog/${OHOS_ARCH}/include)

target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gmock
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gmock/internal
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gmock/internal/custom
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gtest
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gtest/internal
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gtest/internal/custom)
target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/METIS/${OHOS_ARCH}/include)
target_include_directories(3dgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/${OHOS_ARCH}/include)


add_library(
    input SHARED
#    render/render_to_cv_mat.cpp
    napi_init.cpp
    main.cpp
)

target_link_libraries(input PRIVATE 3dgs)

