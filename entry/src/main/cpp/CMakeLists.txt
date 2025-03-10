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

add_library(nativerender SHARED napi_init.cpp main.cpp )
target_link_libraries(nativerender PUBLIC ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH}/lib3dgs.so)


# ktx
set(KTX_SOURCES
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/lib/texture.c
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/lib/hashlist.c
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/lib/checkheader.c
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/lib/swap.c
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/lib/memstream.c
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/lib/filestream.c
)
set(KTX_INCLUDE
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/include
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/lib
	${NATIVERENDER_ROOT_PATH}/3rdParty/ktx/other_include
)

add_library(libktx ${KTX_SOURCES})

target_include_directories(libktx PUBLIC ${KTX_INCLUDE})
target_include_directories(nativerender PRIVATE
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
add_subdirectory(3rdParty/vulkan_radix_sort)
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

target_link_libraries(nativerender PUBLIC
 ${hilog-lib} ${libace-lib} ${libnapi-lib} ${libuv-lib} libnative_window.so libc++.a libktx librawfile.z.so libassimp ${xengine-lib} vk_radix_sort)

# Human Part
target_link_libraries(nativerender PUBLIC ${NATIVERENDER_ROOT_PATH}/../../../libs/${OHOS_ARCH}/libsmpl.so)
target_include_directories(nativerender PRIVATE SMPL/include)
target_include_directories(nativerender PRIVATE segmentation)

target_link_libraries(nativerender PUBLIC libace_napi.z.so
                                   libhilog_ndk.z.so
                                   libohfileio.so
                                   librawfile.z.so
                                   libneural_network_core.so
                                   libneural_network_runtime.so
                                   mindspore_lite_ndk
                                   ${hiai_foundation-lib}
                                   libpixelmap_ndk.z.so
                                   libimage_source_ndk.z.so)
# OpenCV相关
target_include_directories(nativerender PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/opencv/${OHOS_ARCH}/include/opencv4)
target_link_libraries(nativerender PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/opencv/libopencv_calib3d.so.408
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
target_link_libraries(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/${OHOS_ARCH}/lib/libceres.a)
target_link_libraries(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/${OHOS_ARCH}/lib/libgflags_nothreads.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/${OHOS_ARCH}/lib/libgflags.a)
target_link_libraries(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/GKlib/${OHOS_ARCH}/lib/libGKlib.a)

target_link_libraries(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgmock_main.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgmock.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgtest_main.a
                                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/lib/libgtest.a)

target_link_libraries(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/METIS/${OHOS_ARCH}/lib/libmetis.a)
target_link_libraries(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}/glog/libglog.so.1)

target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/${OHOS_ARCH}/include)
target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen/${OHOS_ARCH}/include/eigen3)
target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags/${OHOS_ARCH}/include)
target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/GKlib/${OHOS_ARCH}/include)
target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glog/${OHOS_ARCH}/include)

target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gmock
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gmock/internal
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gmock/internal/custom
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gtest
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gtest/internal
                                        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/${OHOS_ARCH}/include/gtest/internal/custom)
target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/METIS/${OHOS_ARCH}/include)
target_include_directories(nativerender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json/${OHOS_ARCH}/include)