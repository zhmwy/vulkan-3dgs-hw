/*
* Vulkan Example - Compute shader image processing
*
* This sample uses a compute shader to apply different filters to an image
*
* Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/
#pragma once

#ifndef VULKANEXAMPLE_H
#define VULKANEXAMPLE_H

#include "vulkanexamplebase.h"
#include "vk_mem_alloc.h"
#include "cores/buffer.h"
#include "cores/cpu_buffer.h"
#include <cstdint>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "file/file_operator.h"
// #include "vulkan_obj_model.h"
// #include "common/common.h"
#include <chrono>
#include <thread>

// #include <vk_radix_sort.h>

#include "cores/dataloader.h"
#include "cores/random.h"
#include "cores/timer.h"
// #include "vulkanbase\VulkanOhos.h"
#include "cores/dataloader.h"
#include <assert.h>
#include <cmath> // For log function
#include <random>
#include <chrono>
#include <cstdint>

#include "vk_mem_alloc.h"
#include "cores/buffer.h"
#include "cores/cpu_buffer.h"

#include <vulkan/vulkan.h>

#define ENABLE_VALIDATION false

#define FORWARD_PIPE 0
#define BACKWARD_PIPE 1

#define SORT_SIZE 100    // must be equal to TILE_NUM_AVG_MAX!!!!!!!!

#define LOAD_HEAD 0
#define LOAD_EXTRA 1


#define VULKAN_HW
// #define VULKAN_PC

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
};

class VulkanComputeExample : public VulkanExampleBase
{
public:
    void loadTestData(std::string& dataFolder, int flag=-1);
//     void loadIntrinsics(std::string& dataFolder);
    
    void encodeCpu2Gpu();
    void createGpu2CpuCopys();
    VulkanComputeExample();
    ~VulkanComputeExample();
    void DestoryBuffer(vkgs::Buffer buffer);
    void DestoryBuffers();

    #ifdef VULKAN_PC
    uint32_t screenWidth;
    uint32_t screenHeight;
    #endif

//     int render_count = 0;
    Timer timer;

    VmaAllocator allocator;

    // DataLoader::InputGaussianStruct inputGaussian;

    std::string basepath = "";
    DataLoader::InputInitStruct inputGaussian;
    std::vector<float> extr_tmp;
    std::vector<float> intr_tmp; 
    std::vector<float> intr_render_tmp;

    // Predefine 3DGS hyperparameters
    int MAX_SPLAT_COUNT = 200000;
    static const int MAX_TIME_QUERY_NUM = 100;
    static const int QUERY_TIME_RANGE = 500;
    int splat_count = 10000;
    int training_step = 0;
    int max_training_step = 10000;
    int last_step_tag = 0;
    int render_index = 0;
    bool upd_cmd_buffer = true;
    float low_resolotion_scale = 0.25;

    int low_resolution_steps = 1000;
    int sh_degree = 1;
    
    RandomGenerator random_generator;
    int current_img_idx = 0;
    
    struct sortBuffers {
        vkgs::Buffer key;
        vkgs::Buffer index;
        vkgs::Buffer unsort_index;
        vkgs::Buffer sorted_idx_array;
        vkgs::Buffer sort_storage;
        vkgs::Buffer densify_idx;
        vkgs::Buffer splat_visible_point_count;
        vkgs::CpuBuffer splat_visible_point_count_cpu;
        std::vector<uint32_t> sorted_key;
        std::vector<uint32_t> sorted_index;
        
        // 3DGS.cpp sort buffer
        vkgs::Buffer sortKBufferEven;
        vkgs::Buffer sortKBufferOdd;
        vkgs::Buffer sortVBufferEven;
        vkgs::Buffer sortVBufferOdd;
        vkgs::Buffer sortHistBuffer;
        vkgs::Buffer sortHistOffsets;
        
        VkQueryPool queryPool = VK_NULL_HANDLE;
//         VrdxSorter sorter = VK_NULL_HANDLE;
    } sort_buffers;

    struct SplatBuffers {
        // project buffers
        vkgs::Buffer xyz_array;     // [P, 3]
        vkgs::Buffer scale_array;   // [P, 3]
        vkgs::Buffer rotate_array;
        vkgs::Buffer intr_params;   // [4]
        vkgs::Buffer intr_params_render;
        vkgs::Buffer extr_params;           // [1, 7]
        vkgs::Buffer extr_params_render;    // [1, 7]
        vkgs::Buffer visible;
        vkgs::Buffer uvd;     // [P, 3]
        vkgs::Buffer cov2d;
        vkgs::Buffer cov3d;
        vkgs::Buffer radius;
        vkgs::Buffer tiles;
        vkgs::Buffer tiles_rect;
        vkgs::Buffer extr_mat;          // 12 parameters
        vkgs::Buffer extr_act;          // 5  normalized, [w, x, y, z, norm, norm^3]
        vkgs::Buffer camera_center;     // 3  parameters
        vkgs::Buffer camera_extrs;
        
        
        // alpha blending buffers
        vkgs::Buffer conic;
        vkgs::Buffer opacity;
        vkgs::Buffer sh_coeff;
        vkgs::Buffer feature;
        // vkgs::Buffer sort_idx;
        vkgs::Buffer tile_range;
        vkgs::Buffer background;
        vkgs::Buffer rfeat;
        vkgs::Buffer rfeat_alpha;
        vkgs::Buffer final_T;
        vkgs::Buffer ncontrib;

        // Derivatives
        vkgs::Buffer dL_duvd;
        vkgs::Buffer dL_dcov2d;
        vkgs::Buffer dL_dxyz;
        vkgs::Buffer dL_dscale;
        vkgs::Buffer dL_drotate;
        vkgs::Buffer dL_dintr_params;
        vkgs::Buffer dL_dextr_params;

        // vkgs::Buffer dL_drfeat_array;
        // vkgs::Buffer dL_dconic_array;
        // vkgs::Buffer dL_dopacity_array;
        // vkgs::Buffer dL_dfeature_array;
        // vkgs::Buffer dL_dndc_array;
        // vkgs::Buffer dL_dbackground_array;

        // for Adam
        vkgs::Buffer xyz_array_avg;
        vkgs::Buffer scale_array_avg;
        vkgs::Buffer rotate_array_avg;
        vkgs::Buffer feature_array_avg;
        vkgs::Buffer opacity_array_avg;

        vkgs::Buffer xyz_array_exp_avg;
        vkgs::Buffer scale_array_exp_avg;
        vkgs::Buffer rotate_array_exp_avg;
        vkgs::Buffer feature_array_exp_avg;
        vkgs::Buffer opacity_array_exp_avg;

        vkgs::Buffer camera_extrs_avg;
        vkgs::Buffer camera_extrs_exp_avg;
        
        // for groundtruth
        vkgs::Buffer input_image_array;
        vkgs::Buffer input_image_alpha_array;

        vkgs::Buffer dL_drfeat;
        vkgs::Buffer dL_dconic;
        vkgs::Buffer dL_dopacity;
        vkgs::Buffer dL_dfeature;
        vkgs::Buffer dL_dndc;
        vkgs::Buffer dL_dbackground;
        vkgs::Buffer dL_dshs_array;

        // activation patch buffers
        vkgs::Buffer opacity_act_array;
        vkgs::Buffer dL_dopacity_act_array;
        
        // for densification
        vkgs::Buffer max_radii2D;
        vkgs::Buffer xyz_gradient_accum;
        vkgs::Buffer denom;
        vkgs::Buffer curPointsNum;
        vkgs::Buffer xyz_gradient_mean;


        vkgs::Buffer dL_duvd_int;
        vkgs::Buffer dL_dconic_int;
        vkgs::Buffer dL_dopacity_act_array_int;
        vkgs::Buffer dL_dfeature_int;
        vkgs::Buffer dL_dndc_int;

        vkgs::Buffer psnr_sum;
        
    } splat_buffers;

    bool hasKNN = false;
    
    // Input image
    vks::Texture2D textureColorMap;
    // Storage image that the compute shader uses to apply the filter effect to
    vks::Texture2D storageImage;
    VmaAllocation storageImageAllocation;

    // Resources for the graphics part of the example
    struct Graphics {
        VkDescriptorSetLayout descriptorSetLayout;    // Image display shader binding layout
        VkDescriptorSet descriptorSetPreCompute;      // Image display shader bindings before compute shader image manipulation
        VkDescriptorSet descriptorSetPostCompute;     // Image display shader bindings after compute shader image manipulation
        VkPipeline pipeline;                          // Image display pipeline
        VkPipelineLayout pipelineLayout;              // Layout of the graphics pipeline
        VkSemaphore semaphore;                        // Execution dependency between compute & graphic submission
        // Used to pass data to the graphics shaders
        struct UniformData {
            glm::mat4 projection;
            glm::mat4 modelView;
        } uniformData;
        vks::Buffer uniformBuffer;
    } graphics;

    // Resources for the compute part of the example
    struct Compute {
        VkQueue queue;                                // Separate queue for compute commands
        VkCommandPool commandPool;                    // Use a separate command pool
        VkCommandBuffer commandBuffer;                // Command buffer storing the dispatch commands and barriers
        VkSemaphore semaphore;                        // Execution dependency between compute & graphic submission
        VkDescriptorSetLayout descriptorSetLayout;    // Compute shader binding layout
        VkDescriptorSet descriptorSet;                // Compute shader bindings
        VkPipelineLayout pipelineLayout;              // Layout of the compute pipeline
        std::vector<VkPipeline> pipelines;            // Compute pipelines for image filters
        int32_t pipelineIndex;                        // Current image filtering compute pipeline index
        
        VkQueryPool queryPool;
    } compute;

    struct QueryTime {
        uint32_t startIndex; // Start index in the query pool
        uint32_t endIndex;   // End index in the query pool
    };
    std::map<std::string, QueryTime> queryTime;
    std::map<std::string, float> elapsedTimeSum;
    
    struct Compute3DGS {
        VkDescriptorSetLayout descriptorSetLayout;    // Compute shader binding layout
        VkDescriptorSet descriptorSet;                // Compute shader bindings
        VkPipelineLayout pipelineLayout;              // Layout of the compute pipeline
        std::vector<VkPipeline> pipelines;
    };
    Compute3DGS pipe_project, pipe_project_render;
    Compute3DGS pipe_compute_key;
    Compute3DGS pipe_compute_tile;
    Compute3DGS pipe_inverse;
    Compute3DGS pipe_blending;
    
    Compute3DGS pipe_blending_int;
    Compute3DGS pipe_int2float;
    
    Compute3DGS pipe_optimize;
    Compute3DGS pipe_optimize_extr;
    
    Compute3DGS pipe_loss;
    Compute3DGS pipe_copy_img2buffer;
    Compute3DGS pipe_copy_buffer2img; 
    Compute3DGS pipe_buffer_fill;
    Compute3DGS pipe_reset_psnr;
    Compute3DGS pipe_sh;
    Compute3DGS pipe_sh_render;
    Compute3DGS pipe_densification_stats;
    Compute3DGS pipe_mean_grad;
    Compute3DGS pipe_clone;
    Compute3DGS pipe_split;
    Compute3DGS pipe_reset_opacity;
    Compute3DGS pipe_extr_process;
    Compute3DGS pipe_extr_process_render;
    
    
    Compute3DGS pipe_knn;
    Compute3DGS pipe_prune;
    
    Compute3DGS pipe_sort_even;
    Compute3DGS pipe_sort_odd;
    Compute3DGS pipe_hist_even;
    Compute3DGS pipe_hist_odd;
    Compute3DGS pipe_hist_offset;
    
    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount;
    uint32_t vertexBufferSize;

    std::vector<std::string> filterNames;
    
    float total_psnr = 0;
    long long total_sort_point_count = 0;
    double max_avg_splat_tile_num = 0;
    float current_scale=1.0;

    int image_height;
    int image_width;
    
    
    int use_method = 0;
    int cur_method = 0;
    
    virtual void viewChanged()
    {
        updateUniformBuffers();
    }
    
    void SetMethod(int method)
    {
        use_method = method;
        compute.pipelineIndex = use_method;
        #ifdef VULKAN_HW
//         LOGI("VulkanExample curr set method: %{public}d", use_method);
        #endif
    }

    // Redundant vulkan create methos
    void createPipelineLayout(    
        Compute3DGS& pipe_3dgs,
        uint32_t numBindings, 
        bool usePushConstant,
        uint32_t pushConstantSize=0
    );

    void updateDescriptorSet(
        VkDescriptorSet& descriptorSet,
        std::vector<vkgs::Buffer*>& buffers
    );

    void bindPipeline2CommandBuffer(
        const VulkanComputeExample::Compute3DGS& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        bool float_constant = false
    );
    void bindPipeline2CommandBuffer2(
        const VulkanComputeExample::Compute3DGS& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        bool float_constant = false
    );
    void bindPipeline2CommandBufferPushConstants(
        const VulkanComputeExample::Compute3DGS& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        std::vector<uint32_t>& pushConstants
    );

    void ComputeMemoryBarrierBlock();
    void ComputeShaderMemoryBarrierBlock();
    void ComputeTransBarrierBlock();
    void TransComputeBarrierBlock();

    void createGpu2CpuCopy(vkgs::Buffer &buffer);

    // 3dgs compute pipelines
    void setBufferFillPipeline();
    void setResetPSNRPipeline();
    void setImage2BufferPipeline();
    void setBuffer2ImagePipeline();
    void prepare3dgsBuffers();
    void setProjectionPipeline();  
    void setKNNPipeline();
    void setProjectionRenderPipeline();
    void setSortPipeline(); 
    void setComputeKeyPipeline();
    void setComputeTilePipeline();
    void setDensificationStatsPipeline();
    void setMeanGradPipeline();
    void setClonePipeline();
    void setPrunePipeline();
    void setSplitPipeline();
    void setResetOpacityPipeline();
    void setInverseIndexPipeline();
    void setAlphaBlendingPipeline();
    void setAlphaBlendingIntPipeline();
    void setIntToFloatGradientPipeline();
    void setOptimizationPipeline();
    void setExtrOptimizationPipeline();
    
    void setLossPipeline();
    void setComputeSHPipeline();
    void setComputeSHRenderPipeline();
    
    void setHist2Pipeline();
    void setHistEvenPipeline();
    void setHistOddPipeline();
    void setSortEvenPipeline();
    void setSortOddPipeline();
    
    void setExtrProcessPipeline();
    void setExtrProcessRenderPipeline();
    
    
    // 3dgs command buffer
    void forwardCmd();
    void forwardCmdTime();
    
    void backwardCmd();
    void zeroGradCmd();
    void imageToBufferCmd();
    void bufferToImageCmd();
    void optimizeStepCmd();
    void densifyCmd();
    void updRenderCmd();
    void updRenderCmd(int index);
    virtual void encodeInitBuffersValues2CommandBuffer();
    virtual void encoderenderSelectedImage();
    void encodeGetImageBuffer()
    {
        createGpu2CpuCopy(splat_buffers.rfeat);
    }
    void radixSortCmd(uint32_t numInstances);
    void copyExtrCmd();
    
    // Methods
    void prepreAllocator();
    void prepareStorageImage();
    void resizeAndUploadTexture(vks::Texture2D& texture, const unsigned char* original_image_data, int width, int height, float scale, vks::VulkanDevice* vulkanDevice, VkQueue queue);
    void resizeAndUploadTexture(vks::Texture2D& texture, const unsigned char* original_image_data, int width, int height, int new_width, int new_height, vks::VulkanDevice* vulkanDevice, VkQueue queue);
    
    vks::Texture2D loadAsset(int imgidx, int origin_width, int origin_height);
    void loadAssets(int origin_width, int origin_height, float scale);
    
    void buildCommandBuffers();
    
    virtual void encodePipeline2CommandBuffer();
    void buildComputeCommandBuffer(std::function<void()> func);
    
    void generateQuad();
    void setupDescriptorPool();
    void prepareGraphics();
    void prepareCompute();
    void prepareUniformBuffers();
    void updateUniformBuffers();

    virtual bool prepare();
    void calculatePSNR(float* mse, int width, int height);
    virtual void draw(){};
    virtual void printLog(bool train = true);
    void executeComputeCommand();
    void executeTransCommand();

    // void drawUI(const VkCommandBuffer commandBuffer);

    void queryEventTime();
    void queryCmdTimeStart(std::string enventName);
    void queryCmdTimeEnd(std::string enventName);
    
    void printGpuBuffers();
    void saveGaussians(std::string dataPath, bool save_head = false);
    void saveCameras(std::string dataPath);
    
    virtual void getEnabledFeatures() override;
    virtual void getEnabledExtensionsGS() override;
    
    virtual void render()
    {
        if (!prepared) {
            return;
        }
        buildComputeCommandBuffer([this]() { 
            encodePipeline2CommandBuffer(); 
        });
        
        executeComputeCommand();
        printLog();

        if (training_step < max_training_step){
            training_step++;
        }
    }
    int selected_image_index = 0;
    virtual void initBuffersValues()
    {
        if (!prepared) { return; }

        buildComputeCommandBuffer([this]() { 
            encodeInitBuffersValues2CommandBuffer(); 
         });

        executeComputeCommand();
        //draw();
        queryEventTime();

//         if (training_step < max_training_step) {
//             training_step++;
//         }
    }
    virtual void renderImage(int index)
    {
        if (!prepared) {
            return;
        }

        selected_image_index = index;

        buildComputeCommandBuffer([this]() {
            encoderenderSelectedImage();
            });

        executeComputeCommand();
        printLog(false);
    }
    
    const void* getArrayFromGpuBuffer(const vkgs::Buffer& buffer);
    
    void renderRGB2Cpu(int i, std::string& dataFolder, int width=480, int height=640)
    {
        if (!prepared) {
            return;
        }

        buildComputeCommandBuffer([this]() {
            encodeGetImageBuffer();
            });

        executeComputeCommand();
        
        const float* num= reinterpret_cast<const float*> (getArrayFromGpuBuffer(splat_buffers.rfeat));
        int point_num = splat_buffers.rfeat.size() / 3 / sizeof(float);
        
        // Create OpenCV Mat for the image (8-bit, 3 channels)
        cv::Mat image(height, width, CV_8UC3);
    
        // Convert from planar format (RRRGGGBBB) to interleaved format (BGR for OpenCV)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixelIndex = y * width + x;
                
                // Make sure we don't go out of bounds
                if (pixelIndex < point_num) {
                    // Get RGB values from planar format
                    float r = num[pixelIndex];                // R value
                    float g = num[point_num + pixelIndex];    // G value
                    float b = num[2 * point_num + pixelIndex]; // B value
                    // Convert float values [0,1] to uchar [0,255]
                    uchar r_uchar = static_cast<uchar>(std::min(std::max(r * 255.0f, 0.0f), 255.0f));
                    uchar g_uchar = static_cast<uchar>(std::min(std::max(g * 255.0f, 0.0f), 255.0f));
                    uchar b_uchar = static_cast<uchar>(std::min(std::max(b * 255.0f, 0.0f), 255.0f));
                    
                    // OpenCV uses BGR format
                    image.at<cv::Vec3b>(y, x) = cv::Vec3b(b_uchar, g_uchar, r_uchar);
                }
            }
        }
        
        std::string save_dir_debug = dataFolder + "/temp";
        std::filesystem::create_directories(save_dir_debug);
        std::string filename = save_dir_debug + "/" + std::to_string(i) + ".jpg";
        LOGI("Saving rendered image to: %{public}s", filename.c_str());
        cv::imwrite(filename, image);
    }
    #ifdef VULKAN_PC
    void OnUpdateUIOverlay(vks::UIOverlay* overlay)
    {
        if (overlay->header("Settings")) {
            if (overlay->comboBox("Shader", &compute.pipelineIndex, filterNames)) {
                buildComputeCommandBuffer();
            }
        }
    }
    #endif
    template<typename T>
    void printGpuBuffer(const std::string& varName, const vkgs::Buffer& buffer, int pointNum = 50, int length = 3);
private:
    VkPhysicalDeviceFeatures2 features_{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceImagelessFramebufferFeatures imageless_framebuffer_feature_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES};
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDevice16BitStorageFeatures k16bit_storage_features_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features_{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
};

#endif // VULKANEXAMPLE_H