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
// #include "vulkan_obj_model.h"
// #include "common/common.h"
#include <chrono>
#include <thread>

#include "base/math/matrix.h"
#include "base/math/matrix_util.h"

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

#define SORT_SIZE 10    // must be equal to TILE_NUM_AVG_MAX!!!!!!!!

#define LOAD_HEAD 0
#define LOAD_EXTRA 1

#define HEIGHT 640
#define WIDTH  480

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

    std::string basepath = "";
    DataLoader::InputInitStruct inputGaussian;

    // Predefine 3DGS hyperparameters
    int MAX_SPLAT_COUNT = 300000;
    
    int splat_count = 10000;
    int training_step = 0;
    int max_training_step = 10000;
    
    float low_resolotion_scale = 0.25;
    int low_resolution_steps = 500;
    int sh_degree = 1;
    
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


        vkgs::Buffer dL_duvd_int;
        vkgs::Buffer dL_dconic_int;
        vkgs::Buffer dL_dopacity_act_array_int;
        vkgs::Buffer dL_dfeature_int;
        vkgs::Buffer dL_dndc_int;

        vkgs::Buffer psnr_sum;
        
    } splat_buffers;

    bool hasKNN = false;
    
    STComputePipeLine pipe_project, pipe_project_render;
    STComputePipeLine pipe_compute_key;
    STComputePipeLine pipe_compute_tile;
    STComputePipeLine pipe_inverse;
    STComputePipeLine pipe_blending;
    
    STComputePipeLine pipe_blending_int;
    STComputePipeLine pipe_int2float;
    
    STComputePipeLine pipe_optimize;
    STComputePipeLine pipe_optimize_extr;
    
    STComputePipeLine pipe_loss;
    STComputePipeLine pipe_ssim;
    STComputePipeLine pipe_copy_img2buffer;
    STComputePipeLine pipe_copy_buffer2img; 
    STComputePipeLine pipe_buffer_fill;
    STComputePipeLine pipe_sh;
    STComputePipeLine pipe_sh_render;
    
    STComputePipeLine pipe_densification_stats;
    STComputePipeLine pipe_mean_grad;
    STComputePipeLine pipe_clone;
    STComputePipeLine pipe_split;
    STComputePipeLine pipe_clone_3dgs;
    STComputePipeLine pipe_split_3dgs;
    
    STComputePipeLine pipe_reset_opacity;
    STComputePipeLine pipe_extr_process;
    STComputePipeLine pipe_extr_process_render;
    
    STComputePipeLine pipe_knn;
    STComputePipeLine pipe_prune;
    
    STComputePipeLine pipe_sort_even;
    STComputePipeLine pipe_sort_odd;
    STComputePipeLine pipe_hist_even;
    STComputePipeLine pipe_hist_odd;
    STComputePipeLine pipe_hist_offset;
    
    float total_psnr = 0;
    float total_psnr_mask = 0;
    int total_psnr_mask_num = 0;
    int total_sort_point_count = 0;
    double max_avg_splat_tile_num = 0;
    float current_scale=1.0;

    int image_height;
    int image_width;

    void updateDescriptorSet(
        VkDescriptorSet& descriptorSet,
        std::vector<vkgs::Buffer*>& buffers
    );

    void bindPipeline2CommandBuffer(
        const VulkanComputeGS::STComputePipeLine& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        bool float_constant = false
    );
    void bindPipeline2CommandBufferPushConstants(
        const VulkanComputeGS::STComputePipeLine& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        std::vector<uint32_t>& pushConstants
    );

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
    void setClone3DGSPipeline();
    void setSplit3DGSPipeline();
    
    void setResetOpacityPipeline();
    void setInverseIndexPipeline();
    void setAlphaBlendingPipeline();
    void setAlphaBlendingIntPipeline();
    void setIntToFloatGradientPipeline();
    void setOptimizationPipeline();
    void setExtrOptimizationPipeline();
    
    void setLossPipeline();
    void setSSIMPipeline();
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

    void loadAssets(int origin_width, int origin_height, float scale);
    
    virtual void encodePipeline2CommandBuffer();
    
    void prepareSyncSemaphore();

    virtual bool prepare();
    float calculatePSNR(float* mse, int width, int height, float& psnr_total);
    virtual void draw(){};
    virtual void printLog(bool train = true);
    
    void printGpuBuffers();
    void saveGaussians(std::string dataPath, bool save_head = false);
    void saveCameras(std::string dataPath);
    
    virtual void getEnabledFeatures() override;
    virtual void getEnabledExtensionsGS() override;

    virtual void compute();
    int selected_image_index = 0;
    virtual void initBuffersValues();
    virtual void renderImage(int index);

    const void* getArrayFromGpuBuffer(const vkgs::Buffer& buffer);

    void renderRGB2Cpu(int i, std::string &dataFolder, int width = 480, int height = 640);

    template<typename T>
    void printGpuBuffer(const std::string& varName, const vkgs::Buffer& buffer, int pointNum = 50, int length = 3, int offset = 0);
private:
    
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