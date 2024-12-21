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
#include "file/file_operator.h"
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

#define SORT_SIZE 100

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
    void loadTestData();
    void encodeCpu2Gpu();
    void createGpu2CpuCopys();
    VulkanComputeExample();
    ~VulkanComputeExample();

    #ifdef VULKAN_PC
    uint32_t screenWidth;
    uint32_t screenHeight;
    #endif

    int render_count = 0;
    Timer timer;

    VmaAllocator allocator;

    // DataLoader::InputGaussianStruct inputeGaussian;
    DataLoader::InputLegoInitStruct inputeGaussian;
    std::vector<float> extr_tmp;
    std::vector<float> intr_tmp; 

    // Predefine 3DGS hyperparameters
    int MAX_SPLAT_COUNT = 10000;
    int trainign_step = 0;
    int max_trainign_step = 10000;
    int last_step_tag = 0;
    int render_index = 0;
    bool upd_cmd_buffer = true;

    RandomGenerator random_generator;
    int current_img_idx = 0;

    struct sortBuffers {
        vkgs::Buffer key;
        vkgs::Buffer index;
        vkgs::Buffer unsort_index;
        vkgs::Buffer sorted_idx_array;
        vkgs::Buffer sort_storage;
        vkgs::Buffer splat_visible_point_count;
        vkgs::CpuBuffer splat_visible_point_count_cpu;
        VkQueryPool queryPool = VK_NULL_HANDLE;
        VrdxSorter sorter = VK_NULL_HANDLE;
    } sort_buffers;

    struct SplatBuffers {
        // project buffers
        vkgs::Buffer xyz_array;     // [P, 3]
        vkgs::Buffer scale_array;   // [P, 3]
        vkgs::Buffer rotate_array;
        vkgs::Buffer intr_params;   // [4]
        vkgs::Buffer extr_params;   // [3, 4]
        vkgs::Buffer extr_params_render;   // [3, 4]
        vkgs::Buffer visible;
        vkgs::Buffer uvd;     // [P, 3]
        vkgs::Buffer cov2d;
        vkgs::Buffer cov3d;
        vkgs::Buffer radius;
        vkgs::Buffer tiles;
        vkgs::Buffer tiles_rect;

        // alpha blending buffers
        vkgs::Buffer conic;
        vkgs::Buffer opacity;
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

        // for groundtruth
        vkgs::Buffer input_image_array;
        vkgs::Buffer input_image_alpha_array;

        vkgs::Buffer dL_drfeat;
        vkgs::Buffer dL_dconic;
        vkgs::Buffer dL_dopacity;
        vkgs::Buffer dL_dfeature;
        vkgs::Buffer dL_dndc;
        vkgs::Buffer dL_dbackground;

        // activation patch buffers
        vkgs::Buffer opacity_act_array;
        vkgs::Buffer dL_dopacity_act_array;
    } splat_buffers;

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
    } compute;

    struct Compute3DGS {
        VkDescriptorSetLayout descriptorSetLayout;    // Compute shader binding layout
        VkDescriptorSet descriptorSet;                // Compute shader bindings
        VkPipelineLayout pipelineLayout;              // Layout of the compute pipeline
        std::vector<VkPipeline> pipelines;
    } pipe_project, pipe_project_render, pipe_compute_key, pipe_compute_tile, pipe_inverse, pipe_blending, pipe_optimize, pipe_loss, pipe_copy_img2buffer, pipe_copy_buffer2img, pipe_buffer_fill;

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount;
    uint32_t vertexBufferSize;

    std::vector<std::string> filterNames;
    
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

    void ComputeMemoryBarrierBlock();

    void createGpu2CpuCopy(vkgs::Buffer &buffer);

    // 3dgs compute pipelines
    void setBufferFillPipeline();
    void setImage2BufferPipeline();
    void setBuffer2ImagePipeline();
    void prepare3dgsBuffers();
    void setProjectionPipeline();  
    void setProjectionRenderPipeline();
    void setSortPipeline(); 
    void setComputeKeyPipeline();
    void setComputeTilePipeline();
    void setInverseIndexPipeline();
    void setAlphaBlendingPipeline();
    void setOptimizitionPipeline();
    void setLossPipeline();
    // 3dgs command buffer
    void forwardCmd();
    void backwardCmd();
    void zeroGardCmd();
    void imageToBufferCmd();
    void bufferToImageCmd();
    void optimizeStepCmd();
    void updRenderCmd();

    // Methods
    void prepreAllocator();
    void prepareStorageImage();
    void loadAssets();
    void buildCommandBuffers();
    void encodePipeline2CommandBuffer();
    void buildComputeCommandBuffer();
    void generateQuad();
    void setupDescriptorPool();
    void prepareGraphics();
    void prepareCompute();
    void prepareUniformBuffers();
    void updateUniformBuffers();
    bool prepare();
    void draw();
    // void drawUI(const VkCommandBuffer commandBuffer);

    void printGpuBuffers();

    virtual void getEnabledFeatures() override;
    virtual void getEnabledExtensions() override;

    virtual void render()
    {
        if (!prepared) {
            return;
        }
        if (render_count < 2)
        {
            // buildComputeCommandBuffer();
            buildCommandBuffers();
        }
        if (cur_method != use_method || upd_cmd_buffer) {
            upd_cmd_buffer = false;
//             buildComputeCommandBuffer();
            buildCommandBuffers();
            #ifdef VULKAN_HW
//             LOGI("VulkanExample rebuild command buffers");
            #elif defined(VULKAN_PC)
            // std::cout << "VulkanExample rebuild command buffers" << std::endl;
            #endif
            cur_method = use_method;
        }
        buildComputeCommandBuffer();
        if (camera.updated) {
            updateUniformBuffers();
        }
//         std::this_thread::sleep_for(std::chrono::milliseconds(500));

        draw();
        render_count++;
        if (trainign_step < max_trainign_step){
            trainign_step++;
        }
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
    void printGpuBuffer(const std::string& varName, const vkgs::Buffer& buffer, int pointNum = 3, int length = 3);
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