/*
* Vulkan Example - Compute shader image processing
*
* This sample uses a compute shader to apply different filters to an image
*
* Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#ifndef VULKANEXAMPLE_H
#define VULKANEXAMPLE_H

#include "vulkanexamplebase.h"
#include "vk_mem_alloc.h"
#include "cores/buffer.h"
#include "cores/cpu_buffer.h"
#include <vector>
#include <string>
// #include "file/file_operator.h"
// #include "vulkan_obj_model.h"
// #include "common/common.h"
#include <chrono>
#include <thread>

#include <vk_radix_sort.h>

#include "cores/dataloader.h"
#include "cores/random.h"
#include "cores/timer.h"

#define ENABLE_VALIDATION false

#define FORWARD_PIPE 0
#define BACKWARD_PIPE 1


// #define VULKAN_PC
#define VULKAN_HW

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
    void loadTestData();
    void encodeCpu2Gpu();
    void createGpu2CpuCopys();
    VulkanComputeExample();
    ~VulkanComputeExample();

    #ifdef VULKAN_PC
    uint32_t screenWidth;
    uint32_t screenHeight;
    #endif

    //int render_count = 0;
    Timer timer;

    VmaAllocator allocator;

    DataLoader::InputInitStruct inputGaussian;
    std::vector<float> extr_tmp;
    std::vector<float> intr_tmp; 
    std::vector<float> intr_render_tmp;

    // Predefine 3DGS hyperparameters
    int MAX_SPLAT_COUNT = 500000;
    static const int MAX_TIME_QUERY_NUM = 100;
    static const int QUERY_TIME_RANGE = 1000;
    int splat_count = 10000;
    int training_step = 0;
    int max_training_step = 10000;
    int last_step_tag = 0;
    int render_index = 0;
    bool upd_cmd_buffer = true;
    float low_resolotion_scale = 1;
    int sh_degree = 16;
    int save_step = 1000;
    bool isInputScale = false;
    int high_resolution_step = 1000;

    int current_img_idx = -1;

    RandomGenerator random_generator;
    struct sortBuffers {
        vkgs::Buffer key;
        vkgs::Buffer index;
        vkgs::Buffer unsort_index;
        vkgs::Buffer sorted_idx_array;
        vkgs::Buffer sort_storage;
        vkgs::Buffer densify_idx;
        vkgs::Buffer splat_visible_point_count;
        vkgs::CpuBuffer splat_visible_point_count_cpu;

        // 3DGS.cpp sort buffer
        vkgs::Buffer sortKBufferEven;
        vkgs::Buffer sortKBufferOdd;
        vkgs::Buffer sortVBufferEven;
        vkgs::Buffer sortVBufferOdd;
        vkgs::Buffer sortHistBuffer;

        VkQueryPool queryPool = VK_NULL_HANDLE;
        VrdxSorter sorter = VK_NULL_HANDLE;
    } sort_buffers;

    struct SplatBuffers {
        // project buffers
        vkgs::Buffer xyz_array;     // [P, 3]
        vkgs::Buffer scale_array;   // [P, 3]
        vkgs::Buffer rotate_array;
        vkgs::Buffer intr_params;   // [4]
        vkgs::Buffer intr_params_render;
        vkgs::Buffer extr_params;   // [1, 7]
        vkgs::Buffer extr_params_render;   // [1, 7]
        vkgs::Buffer visible;
        vkgs::Buffer uvd;     // [P, 3]
        vkgs::Buffer cov2d;
        vkgs::Buffer cov3d;
        vkgs::Buffer radius;
        vkgs::Buffer tiles;
        vkgs::Buffer tiles_rect;
        vkgs::Buffer extr_mat;      // 12 parameters
        vkgs::Buffer extr_act;      // 5  normalized, [w, x, y, z, norm, norm^3]
        vkgs::Buffer camera_center; // 3  parameters
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

        vkgs::Buffer psnr_sum;

    } splat_buffers;

    //bool hasKNN = false;

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

    Compute3DGS pipe_project;
    Compute3DGS pipe_project_render;
    Compute3DGS pipe_compute_key;
    Compute3DGS pipe_compute_tile; 
    Compute3DGS pipe_inverse;
    Compute3DGS pipe_blending;
    Compute3DGS pipe_optimize;
    Compute3DGS pipe_loss; 
    Compute3DGS pipe_copy_img2buffer; 
    Compute3DGS pipe_copy_buffer2img; 
    Compute3DGS pipe_buffer_fill; 
    Compute3DGS pipe_sh;
    Compute3DGS pipe_sh_render;
    Compute3DGS pipe_densification_stats;
    Compute3DGS pipe_mean_grad;
    Compute3DGS pipe_clone;
    Compute3DGS pipe_split;
    Compute3DGS pipe_reset_opacity;
    Compute3DGS pipe_knn;
    Compute3DGS pipe_prune;
    Compute3DGS pipe_sort_even;
    Compute3DGS pipe_sort_odd;
    Compute3DGS pipe_hist_even;
    Compute3DGS pipe_hist_odd;

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount;
    uint32_t vertexBufferSize;

    std::vector<std::string> filterNames;

    double total_psnr = 0;
    long long total_sort_point_count = 0;
    double max_avg_splat_tile_num = 0;
    float current_scale;
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
        LOGI("VulkanExample curr set method: %{public}d", use_method);
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

    void bindPipeline2CommandBufferPushConstants(
        const VulkanComputeExample::Compute3DGS& pipe, 
        const int PIPELINE_INDEX,
        uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
        std::vector<uint32_t>& pushConstants
    );

    void ComputeMemoryBarrierBlock();
    void ComputeShaderMemoryBarrierBlock();
    void TransComputeBarrierBlock();
    void ComputeTransBarrierBlock();

    void createGpu2CpuCopy(vkgs::Buffer &buffer);

    // 3dgs compute pipelines
    void setBufferFillPipeline();
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
    void setOptimizitionPipeline();
    void setLossPipeline();
    void setComputeSHPipeline();
    void setComputeSHRenderPipeline();
    void setHistEvenPipeline();
    void setHistOddPipeline();
    void setSortEvenPipeline();
    void setSortOddPipeline();
    // 3dgs command buffer
    void forwardCmd();
    void backwardCmd();
    void zeroGardCmd();
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
    void loadAssets(float scale);
    void buildCommandBuffers();
    virtual void encodePipeline2CommandBuffer();
    //void buildComputeCommandBuffer();
    void buildComputeCommandBuffer(std::function<void()> func);
    void generateQuad();
    void setupDescriptorPool();
    void prepareGraphics();
    const void* getArrayFromGpuBuffer(const vkgs::Buffer& buffer);
    void prepareCompute();
    void prepareUniformBuffers();
    void updateUniformBuffers();
    virtual bool prepare();
    void calculatePSNR(float mse, int width, int height);
    virtual void draw(){}
    virtual void printLog();
    void executeComputeCommand();

    void queryEventTime();
    void queryCmdTimeStart(std::string enventName);
    void queryCmdTimeEnd(std::string enventName);

    void printGpuBuffers();

    virtual void getEnabledFeatures() override;
    virtual void getEnabledExtensions() override;

    void renderRGB2Cpu()
    {
        if (!prepared) {
            return;
        }

        buildComputeCommandBuffer([this]() {
            encodeGetImageBuffer();
            });

        executeComputeCommand();

        const float* num = reinterpret_cast<const float*> (getArrayFromGpuBuffer(splat_buffers.rfeat));
        int point_num = splat_buffers.rfeat.size() / 3 / sizeof(float);

        for (int i = 0; i < std::min(100, point_num); i++) {
            std::cout << i << ": ";
            for (int j = 0; j < 3; j++) {
                std::cout << num[j * point_num + i] << "  ";
            }
            std::cout << std::endl;
        }
    }

    virtual void render()
    {
        if (!prepared) {
            return;
        }
        buildComputeCommandBuffer([this]() { 
            encodePipeline2CommandBuffer(); 
        });
        /*buildComputeCommandBuffer(&VulkanComputeExample::encodePipeline2CommandBuffer);*/
        
        // std::this_thread::sleep_for(std::chrono::milliseconds(500));

        executeComputeCommand();
        draw();
        printLog();

        //render_count++;
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

        if (training_step < max_training_step) {
            training_step++;
        }
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
        draw();
    }

    template<typename T>
    void printGpuBuffer(const std::string& varName, const vkgs::Buffer& buffer, int pointNum, int length);
};

#endif // VULKANEXAMPLE_H