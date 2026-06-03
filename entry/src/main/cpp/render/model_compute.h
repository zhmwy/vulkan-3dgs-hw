#pragma once

#ifndef VULKANEXAMPLE_H
#define VULKANEXAMPLE_H

#include "vulkancomputebase.h"
#include "vk_mem_alloc.h"
#include "cores/buffer.h"
#include "cores/cpu_buffer.h"
#include <cstdint>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "file/file_operator.h"
#include <chrono>
#include <thread>

#include "base/math/matrix.h"
#include "base/math/matrix_util.h"


#include "cores/dataloader.h"
#include "cores/visual_hull.h"

#include "cores/random.h"
#include "cores/timer.h"
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

#define SORT_SIZE 10    // must be equal to TILE_NUM_AVG_MAX!!!!!!!!

#define LOAD_HEAD 0
#define LOAD_EXTRA 1

#define HEIGHT 640
#define WIDTH  480

#define SCENE 0
#define OBJECT 1
#define HUMAN 2
#define MULTI_HUMAN 3

#define VULKAN_HW
// #define VULKAN_PC

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

class VulkanComputeGS : public VulkanComputeBase
{
public:
    void loadTestData(std::string& dataFolder, int flag=-1);
    void encodeCpu2Gpu();
    void createGpu2CpuCopys();
    VulkanComputeGS();
    ~VulkanComputeGS();
    void DestoryBuffers();  

    VmaAllocator allocator;

    std::string basepath = "";
    DataLoader::InputInitStruct inputGaussian;

    // Predefine 3DGS hyperparameters
    int MAX_SPLAT_COUNT = 300000;
    static const int MAX_TIME_QUERY_NUM = 100;
    static const int QUERY_TIME_RANGE = 500;
    int splat_count = 10000;
    int training_step = 0;
    int max_training_step = 10000;
    
    int densify_start_step = 500;
    int densify_end_step = 6500;
    int densify_interval_step = 100;
    
    float low_resolotion_scale = 0.25;
    
    int low_resolution_steps = -1;
    int sh_degree = 1;
    
    int trainingMode;  // 0 for scene, 1 for object, 2 for single human, 3 for multi-persons
    bool isObject;  // decide whether to train object only
    bool segmentationFinished = false; // Do not change this unless you are confident that segmentation has been done

    RandomGenerator random_generator;
    int current_img_idx = 0;
    
    struct sortBuffers {
        vkgs::Buffer unsort_index;
        vkgs::Buffer sorted_idx_array;
        vkgs::Buffer densify_idx;
        vkgs::Buffer splat_visible_point_count;
        vkgs::CpuBuffer splat_visible_point_count_cpu;
        
        // 3DGS.cpp sort buffer
        vkgs::Buffer sortKBufferEven;
        vkgs::Buffer sortKBufferOdd;
        vkgs::Buffer sortVBufferEven;
        vkgs::Buffer sortVBufferOdd;
        vkgs::Buffer sortHistBuffer;
        vkgs::Buffer sortHistOffsets;
        
    } sort_buffers;

    struct SplatBuffers {
        // project buffers
        vkgs::Buffer xyz_array;     // [P, 3]
        vkgs::Buffer scale_array;   // [P, 3]
        vkgs::Buffer rotate_array;
        vkgs::Buffer intr_params;   // [4]
        vkgs::Buffer intr_avg;
        vkgs::Buffer intr_exp_avg;
        vkgs::Buffer normal_array;
        
        
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
        
        vkgs::Buffer body_bbox;
        vkgs::Buffer is_body_point;
        vkgs::Buffer max_movement;
        
        // alpha blending buffers
        vkgs::Buffer conic;
        vkgs::Buffer opacity;
        vkgs::Buffer sh_coeff;
        vkgs::Buffer feature;
        // vkgs::Buffer sort_idx;
        vkgs::Buffer tile_range;
        vkgs::Buffer background;
        vkgs::Buffer rfeat;
        vkgs::Buffer alpha_array;
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
        vkgs::Buffer camera_training_steps;
        
        // for groundtruth
        vkgs::Buffer input_image_array;
        vkgs::Buffer input_image_alpha_array;
        vkgs::Buffer input_mask_array;

        vkgs::Buffer dL_drfeat;
        vkgs::Buffer dL_dalpha_array;
        
        vkgs::Buffer dL_dconic;
        vkgs::Buffer dL_dopacity;
        vkgs::Buffer dL_dfeature;
        vkgs::Buffer dL_dndc;
        vkgs::Buffer dL_dbackground;
        vkgs::Buffer dL_dshs_array;
        

        vkgs::Buffer dL_dmap;
        vkgs::Buffer dm_dmu1;
        vkgs::Buffer dm_dsigma1_sq;
        vkgs::Buffer dm_dsigma12;
        vkgs::Buffer ssim_map;
        
        // activation patch buffers
        vkgs::Buffer opacity_act_array;
        vkgs::Buffer dL_dopacity_act_array;
        
        // for densification
        vkgs::Buffer max_radii2D;
        vkgs::Buffer xyz_gradient_accum;
        vkgs::Buffer denom;
        vkgs::Buffer curPointsNum;
        vkgs::Buffer xyz_gradient_mean;
        vkgs::Buffer xyz_gradient_mean_int;
        vkgs::Buffer prev_selected_mask;
        vkgs::Buffer prev_lff_xyz_grad;
        vkgs::Buffer interval_coef;

        vkgs::Buffer dL_duvd_int;
        vkgs::Buffer dL_dconic_int;
        vkgs::Buffer dL_dopacity_act_array_int;
        vkgs::Buffer dL_dfeature_int;
        vkgs::Buffer dL_dndc_int;

        vkgs::Buffer psnr_sum;
        
    } splat_buffers;

    bool hasKNN = false;
    
    // Input image
    vkbase::Texture2DGS textureColorMap;
    VkSemaphore syncSemaphore;

    // Resources for the compute part of the example
    struct Compute {
        VkQueue queue;                                // Separate queue for compute commands
        VkCommandPool commandPool;                    // Use a separate command pool
        VkCommandBuffer commandBuffer;                // Command buffer storing the dispatch commands and barriers
        VkSemaphore semaphore;                        // Execution dependency between compute & graphic submission
        VkDescriptorSetLayout descriptorSetLayout;    // Compute shader binding layout
        VkDescriptorSet descriptorSet;                // Compute shader bindings
        VkPipelineLayout pipelineLayout;              // Layout of the compute pipeline
        
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
    Compute3DGS pipe_flow, pipe_mask_body;
    Compute3DGS pipe_compute_key;
    Compute3DGS pipe_compute_tile;
    Compute3DGS pipe_inverse;
    
    Compute3DGS pipe_blending;
    Compute3DGS pipe_int2float;
    Compute3DGS pipe_blending_object;
    
    Compute3DGS pipe_optimize;
    Compute3DGS pipe_post_flow_optimize;
    Compute3DGS pipe_optimize_flow;
    Compute3DGS pipe_optimize_extr;
    
    Compute3DGS pipe_loss;
    Compute3DGS pipe_ssim;
    Compute3DGS pipe_loss_mask;
    
    Compute3DGS pipe_copy_img2buffer;
    Compute3DGS pipe_copy_buffer2img; 
    Compute3DGS pipe_buffer_fill;
    Compute3DGS pipe_sh;
    Compute3DGS pipe_sh_render;
    
    Compute3DGS pipe_densification_stats;
    Compute3DGS pipe_mean_grad;
    Compute3DGS pipe_clone;
    Compute3DGS pipe_split;
    Compute3DGS pipe_clone_3dgs;
    Compute3DGS pipe_split_3dgs;
    
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
    
    float total_psnr = 0;
    float total_psnr_mask = 0;
    int total_psnr_mask_num = 0;
    int total_sort_point_count = 0;
    double max_avg_splat_tile_num = 0;
    float current_scale=1.0;

    int image_height;
    int image_width;

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
        const VulkanComputeGS::Compute3DGS& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        bool float_constant = false
    );
    void bindPipeline2CommandBufferPushConstants(
        const VulkanComputeGS::Compute3DGS& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        std::vector<uint32_t>& pushConstants
    );
    void bindPipeline2CommandBufferPushConstantsFloat(
        const VulkanComputeGS::Compute3DGS& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        std::vector<float>& pushConstants
    );
    
    void ComputeMemoryBarrierBlock();
    void ComputeShaderMemoryBarrierBlock();
    void ComputeTransBarrierBlock();
    void TransComputeBarrierBlock();

    void createGpu2CpuCopy(vkgs::Buffer &buffer);

    // 3dgs compute pipelines
    void setBufferFillPipeline();
    void setImage2BufferPipeline();

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
    void setSplitPipeline();

    
    void setResetOpacityPipeline();
    void setInverseIndexPipeline();

    void setAlphaBlendingPipeline();
    
    void setIntToFloatGradientPipeline();
    void setOptimizationPipeline();
    void setExtrOptimizationPipeline();
    
    void setLossPipeline();
    void setSSIMPipeline();
    void setLossMaskPipeline();
    
    void setComputeSHPipeline();
    void setComputeSHRenderPipeline();
    
    void setHist2Pipeline();
    void setHistEvenPipeline();
    void setHistOddPipeline();
    void setSortEvenPipeline();
    void setSortOddPipeline();
    
    void setExtrProcessPipeline();
    void setExtrProcessRenderPipeline();
    void setMaskBodyPipeline();

    // 3dgs command buffer
    void forwardCmd();
    void forwardCmdTime();
    
    void backwardCmd();
    void zeroGradCmd();
    void imageToBufferCmd();
    void optimizeStepCmd();
    void densifyCmd();
    
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

    void loadAssets(int origin_width, int origin_height, float scale, bool reload_with_masks=false);
    
    virtual void encodePipeline2CommandBuffer();
    void buildComputeCommandBuffer(std::function<void()> func);
    
    void setupDescriptorPool();
    void prepareSyncSemaphore();
    void prepareCompute();

    virtual bool prepare();
    float calculatePSNR(float* mse, int width, int height, float& psnr_total);
    virtual void draw(){};
    virtual void printLog(bool train = true);
    void executeComputeCommand();
    void executeTransCommand();

    void queryEventTime();
    void queryCmdTimeStart(std::string enventName);
    void queryCmdTimeEnd(std::string enventName);
    
    void printGpuBuffers();
    void saveGaussians(std::string dataPath, bool save_head = false);
    
    virtual void getEnabledFeatures() override;
    virtual void getEnabledExtensionsGS() override;

    virtual void render();
    int selected_image_index = 0;
    virtual void initBuffersValues();
    virtual void renderImage(int index);

    const void* getArrayFromGpuBuffer(const vkgs::Buffer& buffer);

    void renderRGB2Cpu(int i, std::string &dataFolder, int width = 480, int height = 640);

    template<typename T>
    void printGpuBuffer(const std::string& varName, const vkgs::Buffer& buffer, int pointNum = 50, int length = 3, int offset = 0);
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