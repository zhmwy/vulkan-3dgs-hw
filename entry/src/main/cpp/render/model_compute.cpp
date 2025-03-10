#define VMA_IMPLEMENTATION

#include "vk_mem_alloc.h"
#include "cores/buffer.h"
#include "cores/cpu_buffer.h"

#include <vulkan/vulkan.h>

#include "model_compute.h"
// #include "vulkanbase\VulkanOhos.h"
#include "cores/dataloader.h"
#include <assert.h>
#include <cmath> // For log function
#include <random>
#include <chrono>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include <stb_image.h>  // 引入stb_image.h库
#include <stb_image_resize2.h>



#define TILE_W (((int)(current_scale * image_width) + 15) / 16)
#define TILE_H (((int)(current_scale * image_height) + 15) / 16)
//#define TILE_NUM TILE_W * TILE_H
#define TILE_NUM_AVG 100

#define TILE_W_MAX (((int)(image_width) + 15) / 16)
#define TILE_H_MAX (((int)(image_height) + 15) / 16)
//#define TILE_NUM_AVG_MAX TILE_W_MAX * TILE_H_MAX
#define TILE_NUM_AVG_MAX 100


/********************* For Unit Test ****************************/
void VulkanComputeExample::loadTestData()
{
//     inputGaussian = DataLoader::loadInputLegoInitData();
    inputGaussian = DataLoader::loadInputInitDataPM();
    
//     inputGaussian = DataLoader::loadInputLegoInitDatav2();
    inputGaussian.computeNormalization();
    inputGaussian.computePositionStats();
    inputGaussian.normalizeData();
    splat_count = inputGaussian.positions.size();
    this->basepath = inputGaussian.basepath;
    random_generator.initialize(inputGaussian.img_list.size());
    
    intr_render_tmp = inputGaussian.intr_tmp[0];
    intr_tmp = std::vector<float>(4);
    for (int i = 0; i < 4; i++)
        intr_tmp[i] = intr_render_tmp[i] * low_resolotion_scale;
}

void VulkanComputeExample::encodeCpu2Gpu()
{
    inputGaussian.rotations.resize(MAX_SPLAT_COUNT);
    inputGaussian.positions.resize(MAX_SPLAT_COUNT);
    inputGaussian.scales.resize(MAX_SPLAT_COUNT);
    inputGaussian.opacities.resize(MAX_SPLAT_COUNT);
    inputGaussian.colors.resize(MAX_SPLAT_COUNT);

    splat_buffers.xyz_array.FromCpu(
        compute.commandBuffer,
        inputGaussian.positions.data()
    );
    /*splat_buffers.scale_array.FromCpu(
        compute.commandBuffer,
        inputGaussian.scales.data()
    );*/
    splat_buffers.rotate_array.FromCpu(
        compute.commandBuffer,
        inputGaussian.rotations.data()
    );
     splat_buffers.opacity.FromCpu(
         compute.commandBuffer,
         inputGaussian.opacities.data()
     );
    splat_buffers.sh_coeff.FromCpu(
        compute.commandBuffer,
        inputGaussian.colors.data()
    );
    
    splat_buffers.extr_params.FromCpu(
        compute.commandBuffer,
        inputGaussian.extr_tmp[0].data()
    );

    splat_buffers.intr_params.FromCpu(
        compute.commandBuffer,
        intr_tmp.data()
    );

    splat_buffers.intr_params_render.FromCpu(
        compute.commandBuffer,
        intr_render_tmp.data()
    );

    std::vector<int> curPointsNum(1);
    curPointsNum[0] = splat_count;
    splat_buffers.curPointsNum.FromCpu(
        compute.commandBuffer,
        curPointsNum.data()
    );
}

void VulkanComputeExample::createGpu2CpuCopys()
{
    createGpu2CpuCopy(splat_buffers.xyz_array);
    createGpu2CpuCopy(splat_buffers.scale_array);
    createGpu2CpuCopy(splat_buffers.rotate_array);
    createGpu2CpuCopy(splat_buffers.feature);
    createGpu2CpuCopy(splat_buffers.opacity);
    createGpu2CpuCopy(splat_buffers.extr_params);
    createGpu2CpuCopy(splat_buffers.intr_params);

    createGpu2CpuCopy(splat_buffers.visible);
    createGpu2CpuCopy(splat_buffers.radius);
    createGpu2CpuCopy(splat_buffers.tiles);
    createGpu2CpuCopy(splat_buffers.tiles_rect);
    createGpu2CpuCopy(splat_buffers.uvd);
    createGpu2CpuCopy(splat_buffers.cov2d);
    createGpu2CpuCopy(splat_buffers.cov3d);

    createGpu2CpuCopy(splat_buffers.tile_range);
    createGpu2CpuCopy(sort_buffers.sorted_idx_array);

    createGpu2CpuCopy(splat_buffers.conic);
    createGpu2CpuCopy(splat_buffers.rfeat);
    createGpu2CpuCopy(splat_buffers.rfeat_alpha);
    createGpu2CpuCopy(splat_buffers.opacity_act_array);
    createGpu2CpuCopy(splat_buffers.dL_drfeat);
    createGpu2CpuCopy(splat_buffers.input_image_array); 

    createGpu2CpuCopy(splat_buffers.dL_duvd);             // Copy dL_duvd
    createGpu2CpuCopy(splat_buffers.dL_dconic);           // Copy dL_dconic
    createGpu2CpuCopy(splat_buffers.dL_dopacity_act_array); // Copy dL_dopacity_act_array
    createGpu2CpuCopy(splat_buffers.dL_dfeature);         // Copy dL_dfeature
    createGpu2CpuCopy(splat_buffers.dL_dndc);             // Copy dL_dndc
    createGpu2CpuCopy(splat_buffers.dL_dbackground);      // Copy dL_dbackground
    createGpu2CpuCopy(splat_buffers.sh_coeff);
    createGpu2CpuCopy(splat_buffers.curPointsNum);
    createGpu2CpuCopy(sort_buffers.densify_idx);
    createGpu2CpuCopy(splat_buffers.xyz_gradient_mean);
    
    
    createGpu2CpuCopy(splat_buffers.extr_params_render);
    createGpu2CpuCopy(splat_buffers.intr_params);
    createGpu2CpuCopy(splat_buffers.uvd);
//     createGpu2CpuCopy(sort_buffers.splat_visible_point_count);
    createGpu2CpuCopy(splat_buffers.rfeat);
    createGpu2CpuCopy(splat_buffers.rfeat_alpha);
    
    createGpu2CpuCopy(splat_buffers.input_image_array); 
    createGpu2CpuCopy(splat_buffers.input_image_alpha_array);
    
    createGpu2CpuCopy(splat_buffers.dL_duvd);             // Copy dL_duvd
    createGpu2CpuCopy(splat_buffers.dL_dxyz);
    
    createGpu2CpuCopy(splat_buffers.dL_duvd_int);
    createGpu2CpuCopy(splat_buffers.dL_dconic_int);
    createGpu2CpuCopy(splat_buffers.dL_dfeature_int);
    createGpu2CpuCopy(splat_buffers.dL_dopacity_act_array_int);
    
}

int cumsum_cpu(const vkgs::Buffer& buffer){
    const vkgs::CpuBuffer &cpu_buffer = buffer.cpu_buffer;
    if (cpu_buffer.isInit == false) return -1;
    int size = cpu_buffer.size() / sizeof(int32_t);  // 根据元素大小计算数量
    const int32_t* num = reinterpret_cast<const int32_t*>(cpu_buffer.data());
    int sum = 0;
    for(int i = 0 ; i < size;i++){
        sum += num[i];
    }
    return sum;
}

void VulkanComputeExample::printGpuBuffers()
{
    LOGW("Current image index: %{public}d, training step: %{public}d", current_img_idx, training_step);
    printGpuBuffer<float>("dL_duvd", splat_buffers.dL_duvd, 4, 3);
    printGpuBuffer<float>("dL_dconic", splat_buffers.dL_dconic, 4, 3);
    printGpuBuffer<float>("dL_dfeature", splat_buffers.dL_dfeature, 4, 3);
    printGpuBuffer<float>("dL_dopacity_act", splat_buffers.dL_dopacity_act_array, 4, 1);
    
    printGpuBuffer<int>("dL_duvd_int", splat_buffers.dL_duvd_int, 4, 3);
    printGpuBuffer<int>("dL_dconic_int", splat_buffers.dL_dconic_int, 4, 3);
    printGpuBuffer<int>("dL_dfeature_int", splat_buffers.dL_dfeature_int, 4, 3);
    printGpuBuffer<int>("dL_dopacity_act_int", splat_buffers.dL_dopacity_act_array_int, 4, 1);
}

void VulkanComputeExample::radixSortCmd(uint32_t numInstances)
{
    uint32_t launch_size = splat_count * SORT_SIZE;
    uint32_t numRadixSortBlocksPerWorkgroup = 256;
    uint32_t invocationSize = (launch_size + numRadixSortBlocksPerWorkgroup - 1) / numRadixSortBlocksPerWorkgroup;
    invocationSize = (invocationSize + 255) / 256;

    for (uint32_t i = 0; i < 4; i++) {

        std::vector<uint32_t> pushConstants;
        pushConstants.push_back(numInstances);
        pushConstants.push_back(i * 8);
        pushConstants.push_back(invocationSize);
        pushConstants.push_back(numRadixSortBlocksPerWorkgroup);
        bindPipeline2CommandBufferPushConstants(
            i % 2 == 0 ? pipe_hist_even : pipe_hist_odd, 0, 
            invocationSize, 1, 1, 
            pushConstants
        );

        bindPipeline2CommandBufferPushConstants(
            i % 2 == 0 ? pipe_sort_even : pipe_sort_odd, 0, 
            invocationSize, 1, 1, 
            pushConstants
        );
    }
}

void VulkanComputeExample::imageToBufferCmd()
{
    VkBufferImageCopy copyRegionI = {};
    copyRegionI.bufferOffset = 0;
    copyRegionI.bufferRowLength = 0;         // Tightly packed
    copyRegionI.bufferImageHeight = 0;       // Tightly packed
    copyRegionI.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionI.imageSubresource.mipLevel = 0;
    copyRegionI.imageSubresource.baseArrayLayer = 0;
    copyRegionI.imageSubresource.layerCount = 1;
    copyRegionI.imageOffset = { 0, 0, 0 };
    copyRegionI.imageExtent = {(unsigned int)(image_width * current_scale), (unsigned int)(image_height * current_scale), 1 };

    // int current_img_idx = training_step % inputGaussian.img_list.size();
    current_img_idx = random_generator.generate();
    
//     current_img_idx = 0;
    
    if (training_step >= 1000)
    {
        vkCmdCopyImageToBuffer(
            compute.commandBuffer,
            inputGaussian.images[current_img_idx].image,
            inputGaussian.images[current_img_idx].imageLayout,
            splat_buffers.input_image_alpha_array,
            1,
            &copyRegionI
        );
        splat_buffers.intr_params.FromCpu(
            compute.commandBuffer,
            intr_render_tmp.data()
        );
        if (inputGaussian.intr_tmp.size() > 1)
        {
            auto intr_tmp_low_resolution = inputGaussian.intr_tmp[current_img_idx];
            for (int i = 0; i < 4; i++)
                intr_tmp_low_resolution[i] = intr_tmp_low_resolution[i] * low_resolotion_scale;
            splat_buffers.intr_params.FromCpu(
                compute.commandBuffer,
                intr_tmp_low_resolution.data()
            );
        }
    }
    else
    {
        vkCmdCopyImageToBuffer(
                compute.commandBuffer,
                inputGaussian.images_low_res[current_img_idx].image,
                inputGaussian.images_low_res[current_img_idx].imageLayout,
                splat_buffers.input_image_alpha_array,
                1,
                &copyRegionI
            );
    }
    
    splat_buffers.extr_params.FromCpu(
        compute.commandBuffer,
        inputGaussian.extr_tmp[current_img_idx].data()
    );

    TransComputeBarrierBlock();
    
    uint32_t X = 0;
    uint32_t Y = 0;
    X = ((unsigned int)(image_width * current_scale) + 16 - 1) / 16; // Ceiling division
    Y = ((unsigned int)(image_height * current_scale) + 16 - 1) / 16; // Ceiling division
    bindPipeline2CommandBuffer(pipe_copy_img2buffer, 0, X, Y, 1);
}

void VulkanComputeExample::zeroGradCmd()
{
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dbackground, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dconic, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dcov2d, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dextr_params, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dfeature, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dintr_params, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dndc, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dopacity, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dopacity_act_array, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_drfeat, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_drotate, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dscale, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_duvd, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dxyz, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dshs_array, 0, VK_WHOLE_SIZE, 0);
    
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dconic_int, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dfeature_int, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dndc_int, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_dopacity_act_array_int, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.dL_duvd_int, 0, VK_WHOLE_SIZE, 0);
}

void VulkanComputeExample::forwardCmd()
{

    uint32_t X = 0;
    uint32_t Y = 0;
    
    /********************* Inverse index gaussian point ***********************/
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_inverse, FORWARD_PIPE, X, 1, 1);
    /********************* Inverse index gaussian point ***********************/
    
    uint32_t grid_width = (int)(current_scale * image_width) / 16;
    grid_width = ((int)(current_scale * image_width) + 15) / 16;
    uint32_t grid_height = (int)(current_scale * image_height) / 16;
    grid_height = ((int)(current_scale * image_height) + 15) / 16;
    vkCmdFillBuffer(
        compute.commandBuffer,
        splat_buffers.tile_range, 0,
        grid_width * grid_height * 2 * sizeof(int32_t), 0
    );
    TransComputeBarrierBlock();

    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_compute_key, FORWARD_PIPE, X, 1, 1);
    ComputeTransBarrierBlock();

    VkBufferCopy copyRegionBB = {};
    copyRegionBB.srcOffset = 0; // Start at the beginning of the source buffer
    copyRegionBB.dstOffset = 0; // Start at the beginning of the destination buffer
    copyRegionBB.size = splat_count * sizeof(uint32_t) * SORT_SIZE; // Size of data to copy
    vkCmdCopyBuffer(
        compute.commandBuffer,
        sort_buffers.unsort_index,
        sort_buffers.sortVBufferEven,
        1,            // Number of regions to copy
        &copyRegionBB   // Pointer to the copy region
    );
    TransComputeBarrierBlock();

    radixSortCmd(0);
    ComputeShaderMemoryBarrierBlock();

    X = (splat_count * SORT_SIZE + 256 - 1) / 256;
    bindPipeline2CommandBuffer(pipe_compute_tile, FORWARD_PIPE, X, 1, 1);
    

    /*********************  Alpha blending ***********************/
    X = ((int)(current_scale * image_width) + 16 - 1) / 16; // Ceiling division
    Y = ((int)(current_scale * image_height) + 16 - 1) / 16; // Ceiling division
    bindPipeline2CommandBuffer(pipe_blending, FORWARD_PIPE, X, Y, 1);
    /*********************  Alpha blending ***********************/
    
    X = ((int)(current_scale * image_width) + 16 - 1) / 16; // Ceiling division
    Y = ((int)(current_scale * image_height) + 16 - 1) / 16; // Ceiling division
    bindPipeline2CommandBuffer(pipe_copy_buffer2img, 0, X, Y, 1);
}

void VulkanComputeExample::forwardCmdTime()
{

    uint32_t X = 0;
    uint32_t Y = 0;
    
    /********************* Inverse index gaussian point ***********************/
    queryCmdTimeStart("07. forward inverse");
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_inverse, FORWARD_PIPE, X, 1, 1);
    queryCmdTimeEnd("07. forward inverse");
    /********************* Inverse index gaussian point ***********************/

    
    queryCmdTimeStart("08. forward fill tile_range");
    uint32_t grid_width = (int)(current_scale * image_width) / 16;
    grid_width = ((int)(current_scale * image_width) + 15) / 16;
    uint32_t grid_height = (int)(current_scale * image_height) / 16;
    grid_height = ((int)(current_scale * image_height) + 15) / 16;
    vkCmdFillBuffer(
        compute.commandBuffer,
        splat_buffers.tile_range, 0,
        grid_width * grid_height * 2 * sizeof(int32_t), 0
    );
    TransComputeBarrierBlock();
    queryCmdTimeEnd("08. forward fill tile_range");
        
    queryCmdTimeStart("09. forward compute key");
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_compute_key, FORWARD_PIPE, X, 1, 1);
    ComputeTransBarrierBlock();
    queryCmdTimeEnd("09. forward compute key");
    
    queryCmdTimeStart("10. forward copy index");
    VkBufferCopy copyRegionBB = {};
    copyRegionBB.srcOffset = 0; // Start at the beginning of the source buffer
    copyRegionBB.dstOffset = 0; // Start at the beginning of the destination buffer
    copyRegionBB.size = splat_count * sizeof(uint32_t) * SORT_SIZE; // Size of data to copy
    vkCmdCopyBuffer(
        compute.commandBuffer,
        sort_buffers.unsort_index,
        sort_buffers.sortVBufferEven,
        1,            // Number of regions to copy
        &copyRegionBB   // Pointer to the copy region
    );
    TransComputeBarrierBlock();
    queryCmdTimeEnd("10. forward copy index");
    
//     vrdxCmdSortKeyValueIndirect(
//         compute.commandBuffer, 
//         sort_buffers.sorter, 
//         splat_count * SORT_SIZE, 
//         // required buffers
//         sort_buffers.splat_visible_point_count, 0,
//         sort_buffers.key, 0, 
//         sort_buffers.index, 0, 
//         sort_buffers.sort_storage, 0,
//         NULL, 0
//     );
    
    queryCmdTimeStart("11. forward sort");
    radixSortCmd(0);
    ComputeShaderMemoryBarrierBlock();
    queryCmdTimeEnd("11. forward sort");
    
    queryCmdTimeStart("12. forward compute tile");
    X = (splat_count * SORT_SIZE + 256 - 1) / 256;
    bindPipeline2CommandBuffer(pipe_compute_tile, FORWARD_PIPE, X, 1, 1);
    queryCmdTimeEnd("12. forward compute tile");
    
    /*********************  Alpha blending ***********************/
    queryCmdTimeStart("13. forward blend");
    X = ((int)(current_scale * image_width) + 16 - 1) / 16; // Ceiling division
    Y = ((int)(current_scale * image_height) + 16 - 1) / 16; // Ceiling division
    bindPipeline2CommandBuffer(pipe_blending, FORWARD_PIPE, X, Y, 1);
    queryCmdTimeEnd("13. forward blend");
    /*********************  Alpha blending ***********************/

    queryCmdTimeStart("14. forward copy buffer2img");
    X = ((int)(current_scale * image_width) + 16 - 1) / 16; // Ceiling division
    Y = ((int)(current_scale * image_height) + 16 - 1) / 16; // Ceiling division
    bindPipeline2CommandBuffer(pipe_copy_buffer2img, 0, X, Y, 1);
    queryCmdTimeEnd("14. forward copy buffer2img");
}


void VulkanComputeExample::bufferToImageCmd()
{
    VkBufferImageCopy copyRegionI = {};
    copyRegionI.bufferOffset = 0;
    copyRegionI.bufferRowLength = 0;         // Tightly packed
    copyRegionI.bufferImageHeight = 0;       // Tightly packed
    copyRegionI.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionI.imageSubresource.mipLevel = 0;
    copyRegionI.imageSubresource.baseArrayLayer = 0;
    copyRegionI.imageSubresource.layerCount = 1;
    copyRegionI.imageOffset = { 0, 0, 0 };
    copyRegionI.imageExtent = {(unsigned int)(image_width * current_scale), (unsigned int)(image_height * current_scale), 1 };

    vkCmdCopyBufferToImage(
        compute.commandBuffer,
        splat_buffers.rfeat_alpha,
        storageImage.image,
        storageImage.imageLayout,
        1,
        &copyRegionI
    );
    TransComputeBarrierBlock();
}

void VulkanComputeExample::backwardCmd()
{
    uint32_t X = 0;
    uint32_t Y = 0;
    
    /*********************  l2 loss ***********************/
    queryCmdTimeStart("15. L2 loss");
    X = ((int)(current_scale * image_width) * (int)(current_scale * image_height) + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_loss, FORWARD_PIPE, X, 1, 1);
    queryCmdTimeEnd("15. L2 loss");
    /*********************  l2 loss ***********************/

    // Calling the backward pipeline

    /*********************  Alpha blending backward ***********************/
    queryCmdTimeStart("16. Backward blending");
    X = ((int)(current_scale * image_width) + 16 - 1) / 16; // Ceiling division
    Y = ((int)(current_scale * image_height) + 16 - 1) / 16; // Ceiling division
    bindPipeline2CommandBuffer(pipe_blending_int, BACKWARD_PIPE, X, Y, 1);
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_int2float, BACKWARD_PIPE, X, 1, 1);
    queryCmdTimeEnd("16. Backward blending");
    /*********************  Alpha blending backward ***********************/
    
    /*********************  Compute SH backward ***********************/
    queryCmdTimeStart("17. Backward SH");
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_sh, BACKWARD_PIPE, X, 1, 1);
    queryCmdTimeEnd("17. Backward SH");
    /*********************  Compute SH backward ***********************/
    
    /********************* Inverse index gaussian point backward ***********************/
    queryCmdTimeStart("18. Backward inverse");
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_inverse, BACKWARD_PIPE, X, 1, 1);
    queryCmdTimeEnd("18. Backward inverse");
    /********************* Inverse index gaussian point backward ***********************/

    /*********************  Project EWA backward ***********************/
    queryCmdTimeStart("19. Backward EWA");
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_project, BACKWARD_PIPE, X, 1, 1);
    queryCmdTimeEnd("19. Backward EWA");
    /*********************  Project EWA backward ***********************/
}
 
void VulkanComputeExample::optimizeStepCmd()
{
    uint32_t X = 0;
    /*********************  Adam optimizer ***********************/
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_optimize, FORWARD_PIPE, X, 1, 1, true);
    /*********************  Adam optimizer ***********************/
}

void VulkanComputeExample::densifyCmd()
{
    uint32_t X = 0;
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    bindPipeline2CommandBuffer(pipe_mean_grad, FORWARD_PIPE, X, 1, 1);
    ComputeTransBarrierBlock();
    
    VkBufferCopy copyRegionBB = {};
    copyRegionBB.srcOffset = 0; // Start at the beginning of the source buffer
    copyRegionBB.dstOffset = 0; // Start at the beginning of the destination buffer
    copyRegionBB.size = splat_count * sizeof(uint32_t); // Size of data to copy
    
    vkCmdCopyBuffer(
        compute.commandBuffer,
        splat_buffers.xyz_gradient_mean,
        sort_buffers.sortKBufferEven,
        1,            // Number of regions to copy
        &copyRegionBB   // Pointer to the copy region
    );
    vkCmdCopyBuffer(
        compute.commandBuffer,
        sort_buffers.densify_idx,
        sort_buffers.sortVBufferEven,
        1,            // Number of regions to copy
        &copyRegionBB   // Pointer to the copy region
    );
    TransComputeBarrierBlock();

//     vrdxCmdSortKeyValueIndirect(
//         compute.commandBuffer,
//         sort_buffers.sorter,
//         MAX_SPLAT_COUNT * SORT_SIZE,
//         // required buffers
//         splat_buffers.curPointsNum, 0,
//         splat_buffers.xyz_gradient_mean, 0,
//         sort_buffers.densify_idx, 0,
//         sort_buffers.sort_storage, 0,
//         NULL, 0
//     );
    
    radixSortCmd(splat_count);
    ComputeShaderMemoryBarrierBlock();
    
    vkCmdCopyBuffer(
        compute.commandBuffer,
        sort_buffers.sortKBufferEven,
        splat_buffers.xyz_gradient_mean,
        1,            // Number of regions to copy
        &copyRegionBB   // Pointer to the copy region
    );
    vkCmdCopyBuffer(
        compute.commandBuffer,
        sort_buffers.sortVBufferEven,
        sort_buffers.densify_idx,
        1,            // Number of regions to copy
        &copyRegionBB   // Pointer to the copy region
    );
    TransComputeBarrierBlock();
    
//     return;
    
//     bindPipeline2CommandBuffer(pipe_prune, FORWARD_PIPE, X, 1, 1, true);
    bindPipeline2CommandBuffer(pipe_clone, FORWARD_PIPE, X, 1, 1, true);
    bindPipeline2CommandBuffer(pipe_split, FORWARD_PIPE, X, 1, 1, true);
    
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.xyz_gradient_accum, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.denom, 0, VK_WHOLE_SIZE, 0);
    //vkCmdFillBuffer(compute.commandBuffer, splat_buffers.curPointsNum, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(compute.commandBuffer, splat_buffers.max_radii2D, 0, VK_WHOLE_SIZE, 0);
}

void VulkanComputeExample::updRenderCmd( /**/ )
{
    VkImageCopy copyRegionI2I = {};
    copyRegionI2I.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionI2I.srcSubresource.mipLevel = 0;
    copyRegionI2I.srcSubresource.baseArrayLayer = 0;
    copyRegionI2I.srcSubresource.layerCount = 1;
    copyRegionI2I.srcOffset = {0, 0, 0};

    copyRegionI2I.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegionI2I.dstSubresource.mipLevel = 0;
    copyRegionI2I.dstSubresource.baseArrayLayer = 0;
    copyRegionI2I.dstSubresource.layerCount = 1;
    copyRegionI2I.dstOffset = {static_cast<int32_t>((unsigned int)(image_width * current_scale)), static_cast<int32_t>((unsigned int)(image_height * current_scale)), 1};
    copyRegionI2I.extent = {(unsigned int)(image_width * current_scale), (unsigned int)(image_height * current_scale), 1};

    // render fix view
    if (timer.elapsed() - timer.last_time > 16) {
        timer.upd_last_time(timer.elapsed());
        render_index += 1;
        // render_index = 6;
        last_step_tag = training_step;
        upd_cmd_buffer = true;

        vkCmdCopyImage(
            compute.commandBuffer,
            inputGaussian.images[6].image,
            inputGaussian.images[6].imageLayout,
            textureColorMap.image,
            textureColorMap.imageLayout,
            1,
            &copyRegionI2I
        );
    }
    splat_buffers.extr_params_render.FromCpu(
        compute.commandBuffer,
        // inputGaussian.extr_tmp[0].data()
        inputGaussian.extr_tmp_test[render_index % inputGaussian.extr_tmp_test.size()].data()
    );
    
    if (inputGaussian.intr_tmp.size() > 1)
    {
        splat_buffers.intr_params_render.FromCpu(
            compute.commandBuffer,
            inputGaussian.intr_tmp[render_index % inputGaussian.intr_tmp.size()].data()
        );
    }
    TransComputeBarrierBlock();
}

void VulkanComputeExample::encodePipeline2CommandBuffer()
{
    queryTime.clear();
    
    uint32_t X = 0;
    X = (splat_count + 256 - 1) / 256; // Ceiling division
    VkDeviceSize size = sizeof(uint32_t);
    if (training_step >= 1000)
        current_scale = 1.0;
    else
        current_scale = low_resolotion_scale;
    if (hasKNN == false)
    {
        queryCmdTimeStart("KNN");
        bindPipeline2CommandBuffer(pipe_knn, FORWARD_PIPE, X, 1, 1);
        queryCmdTimeEnd("KNN");
        hasKNN = true;
        //render_count++;
        training_step--;
        render_count--;
        return;
    }
    
    queryCmdTimeStart("00. compute overall");
    if (training_step < max_training_step){
        queryCmdTimeStart("01. image to buffer");
        imageToBufferCmd();
        queryCmdTimeEnd("01. image to buffer");

        queryCmdTimeStart("02. reset grad");
        zeroGradCmd();
        queryCmdTimeEnd("02. reset grad");

        queryCmdTimeStart("03. forward fill visible point");
        vkCmdFillBuffer(compute.commandBuffer, sort_buffers.splat_visible_point_count, 0, size, 0);
        queryCmdTimeEnd("03. forward fill visible point");
        
        queryCmdTimeStart("04. forward fill buffer");
        bindPipeline2CommandBuffer(pipe_buffer_fill, 0, X, 1, 1);
        queryCmdTimeEnd("04. forward fill buffer");
        
        queryCmdTimeStart("05. forward EWA");
        bindPipeline2CommandBuffer(pipe_project, FORWARD_PIPE, X, 1, 1);
        queryCmdTimeEnd("05. forward EWA");
        
        queryCmdTimeStart("06. forward SH");
        bindPipeline2CommandBuffer(pipe_sh, FORWARD_PIPE, X, 1, 1, true);
        queryCmdTimeEnd("06. forward SH");
        
        forwardCmdTime();
        backwardCmd();

        queryCmdTimeStart("20. training optimize");
        optimizeStepCmd();
        queryCmdTimeEnd("20. training optimize");
        
        bindPipeline2CommandBuffer(pipe_densification_stats, FORWARD_PIPE, X, 1, 1);

        if (training_step >= 500 && training_step % 100 == 0)
        {
            densifyCmd();
        }
        if (training_step % 3000 == 0 && training_step > 1000)
        {
            bindPipeline2CommandBuffer(pipe_reset_opacity, FORWARD_PIPE, X, 1, 1);
        }
    }
    queryCmdTimeEnd("00. compute overall");
    
    current_scale = 1.0;
    
    ComputeTransBarrierBlock();
    updRenderCmd();
    vkCmdFillBuffer(compute.commandBuffer, sort_buffers.splat_visible_point_count, 0, size, 0);
    TransComputeBarrierBlock();
    bindPipeline2CommandBuffer(pipe_buffer_fill, 0, X, 1, 1);
    bindPipeline2CommandBuffer(pipe_project_render, FORWARD_PIPE, X, 1, 1);
    bindPipeline2CommandBuffer(pipe_sh_render, FORWARD_PIPE, X, 1, 1, true);
    forwardCmd();
    ComputeTransBarrierBlock();
    bufferToImageCmd();
    

//     createGpu2CpuCopys();
    createGpu2CpuCopy(splat_buffers.curPointsNum);
    createGpu2CpuCopy(splat_buffers.psnr_sum);
    createGpu2CpuCopy(sort_buffers.splat_visible_point_count);
    if (training_step == save_step)
    {
        createGpu2CpuCopy(splat_buffers.xyz_array);
        createGpu2CpuCopy(splat_buffers.scale_array);
        createGpu2CpuCopy(splat_buffers.rotate_array);
        createGpu2CpuCopy(splat_buffers.sh_coeff);
        createGpu2CpuCopy(splat_buffers.opacity);
    }
}


/********************* End ****************************/

VulkanComputeExample::VulkanComputeExample() : VulkanExampleBase()
{
#ifdef WIN32
    AllocConsole();
    // Redirect stdout and stdin to console
    FILE* stream;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONIN$", "r", stdin);
#endif
#ifdef VULKAN_HW
    enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    enabledDeviceExtensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    enabledDeviceExtensions.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
#endif

    std::string title = "Compute shader image load/store";
    camera.type = Camera::CameraType::lookat;
    camera.setPosition(glm::vec3(0.0f, 0.0f, -2.0f));
    camera.setRotation(glm::vec3(0.0f));
    // camera.setPerspective(60.0f, (float)width * 0.5f / (float)height, 1.0f, 256.0f);

# ifdef VULKAN_PC
    screenWidth = width;
    screenHeight = height;
# endif
    camera.setPerspective(60.0f, (float)screenWidth / (float)screenHeight, 1.0f, 256.0f);

    // Initialize Vulkan handles to VK_NULL_HANDLE
    graphics.descriptorSetLayout = VK_NULL_HANDLE;
    graphics.descriptorSetPreCompute = VK_NULL_HANDLE;
    graphics.descriptorSetPostCompute = VK_NULL_HANDLE;
    graphics.pipeline = VK_NULL_HANDLE;
    graphics.pipelineLayout = VK_NULL_HANDLE;
    graphics.semaphore = VK_NULL_HANDLE;

    compute.queue = VK_NULL_HANDLE;
    compute.commandPool = VK_NULL_HANDLE;
    compute.commandBuffer = VK_NULL_HANDLE;
    compute.semaphore = VK_NULL_HANDLE;
    compute.descriptorSetLayout = VK_NULL_HANDLE;
    compute.descriptorSet = VK_NULL_HANDLE;
    compute.pipelineLayout = VK_NULL_HANDLE;
    compute.pipelineIndex = 0;
}

VulkanComputeExample::~VulkanComputeExample()
{
    if (device) {
        // Graphics
        vkDestroyPipeline(device, graphics.pipeline, nullptr);
        vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);
        vkDestroySemaphore(device, graphics.semaphore, nullptr);
        graphics.uniformBuffer.destroy();

        // Compute
        for (auto& pipeline : compute.pipelines)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
        vkDestroySemaphore(device, compute.semaphore, nullptr);
        vkDestroyCommandPool(device, compute.commandPool, nullptr);

        vertexBuffer.destroy();
        indexBuffer.destroy();

        textureColorMap.destroy();
        storageImage.destroy();

        vmaDestroyImage(allocator, storageImage.image, storageImageAllocation);
        vmaDestroyAllocator(allocator);
        //FreeConsole();
    }
}

void VulkanComputeExample::getEnabledFeatures()
{
    LOGI("getEnabledFeatures");
    deviceFeatures.shaderInt64 = VK_TRUE;
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
    enabledFeatures = deviceFeatures;

    // features
    k16bit_storage_features_.pNext = &buffer_device_address_features_;

    timeline_semaphore_features_.pNext = &k16bit_storage_features_;

    imageless_framebuffer_feature_.pNext = &timeline_semaphore_features_;

    features_.pNext = &imageless_framebuffer_feature_;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features_);

    deviceCreatepNextChain = &features_; 
}

void VulkanComputeExample::getEnabledExtensions()
{
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());
    // VK_EXT_shader_atomic_float
    bool hasAtomicFloatExtension = false;
    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0) {
            hasAtomicFloatExtension = true;
            break;
        }
    }
    if (!hasAtomicFloatExtension) {
        throw std::runtime_error("VK_EXT_shader_atomic_float extension is not supported");
    }
    enabledDeviceExtensions = {
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    };
    
    bool isBufferDeviceAddressSupported = false;
    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, "VK_KHR_buffer_device_address") == 0) {
            isBufferDeviceAddressSupported = true;
            break;
        }
    }
    
    if (isBufferDeviceAddressSupported) {
        printf("VK_KHR_buffer_device_address is supported\n");
    } else {
        throw std::runtime_error("VK_KHR_buffer_device_address extension is not supported");
    }
    enabledFeatures.shaderInt64 = true;
}

void VulkanComputeExample::prepare3dgsBuffers()
{
    splat_buffers.xyz_array = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.scale_array = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.rotate_array = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.sh_coeff = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 48 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dshs_array = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 48 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.intr_params = vkgs::Buffer(
        allocator, 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.intr_params_render = vkgs::Buffer(
        allocator, 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.extr_params = vkgs::Buffer(
        allocator, 12 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    ); 
    splat_buffers.extr_params_render = vkgs::Buffer(
        allocator, 12 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.visible = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.uvd = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.cov2d = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.cov3d = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 6 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.radius = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(int32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.tiles = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(int32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.tiles_rect = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 4 * sizeof(int32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.conic = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.opacity = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.feature = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    uint32_t grid_width = image_width / 16;
    grid_width = (image_width + 15) / 16;
    uint32_t grid_height = image_height / 16;
    grid_height = (image_height + 15) / 16;
    splat_buffers.tile_range = vkgs::Buffer(
        allocator, grid_width * grid_height * 2 * sizeof(int32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.background = vkgs::Buffer(
        allocator, image_width * image_height * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.rfeat = vkgs::Buffer(
        allocator, image_width * image_height * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.final_T = vkgs::Buffer(
        allocator, image_width * image_height * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.ncontrib = vkgs::Buffer(
        allocator, image_width * image_height * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.dL_duvd = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dcov2d = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dxyz = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dscale = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_drotate = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dintr_params = vkgs::Buffer(
        allocator, 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dextr_params = vkgs::Buffer(
        allocator, 12 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.dL_drfeat = vkgs::Buffer(
        allocator, image_width * image_height * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dconic = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dopacity = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 1 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dfeature = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dndc = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 2 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dbackground = vkgs::Buffer(
        allocator, image_width * image_height * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.xyz_array_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.scale_array_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.rotate_array_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 4 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.feature_array_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 48 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.opacity_array_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    // Exponential moving averages (exp_avg) for Adam optimizer
    splat_buffers.xyz_array_exp_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.scale_array_exp_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.rotate_array_exp_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 4 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.feature_array_exp_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 48 * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.opacity_array_exp_avg = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(float), 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.input_image_array = vkgs::Buffer(
        allocator, image_width * image_height * 3 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    // For image transfer
    splat_buffers.input_image_alpha_array = vkgs::Buffer(
        allocator, image_width * image_height * 4 * sizeof(uint8_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.rfeat_alpha = vkgs::Buffer(
        allocator, image_width * image_height * 4 * sizeof(uint8_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    // Activation patch buffers
    splat_buffers.opacity_act_array = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dopacity_act_array = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    ); 

    splat_buffers.max_radii2D = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.xyz_gradient_accum = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.denom = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.curPointsNum = vkgs::Buffer(
        allocator, 1 * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.xyz_gradient_mean = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    sort_buffers.densify_idx = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.psnr_sum = vkgs::Buffer(
        allocator, sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );

    splat_buffers.dL_duvd_int = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(int),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dconic_int = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(int),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dopacity_act_array_int = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * sizeof(int),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dfeature_int = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 3 * sizeof(int),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    splat_buffers.dL_dndc_int = vkgs::Buffer(
        allocator, MAX_SPLAT_COUNT * 2 * sizeof(int),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
}

void VulkanComputeExample::createPipelineLayout(
    Compute3DGS& pipe_3dgs,
    uint32_t numBindings, 
    bool usePushConstant,
    uint32_t pushConstantSize
) {
    // Define descriptor set bindings
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings(numBindings);
    
    for (uint32_t i = 0; i < numBindings; ++i) {
        setLayoutBindings[i] = vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,              // descriptor type
            VK_SHADER_STAGE_COMPUTE_BIT,                    // shader stage
            i                                               // binding number
        );
    }

    // Create descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &pipe_3dgs.descriptorSetLayout)
    );

    // Define push constant range (optional)
    VkPushConstantRange pushConstantRange = {};
    if (usePushConstant) {
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = pushConstantSize;  // Example push constant size
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &pipe_3dgs.descriptorSetLayout;

    if (usePushConstant) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    } else {
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
    }

    VK_CHECK_RESULT(
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipe_3dgs.pipelineLayout)
    );

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(
        descriptorPool, 
        &pipe_3dgs.descriptorSetLayout, 1
    );

    VK_CHECK_RESULT(
        vkAllocateDescriptorSets(device, &allocInfo, &pipe_3dgs.descriptorSet)
    );
}

void VulkanComputeExample::updateDescriptorSet(
    VkDescriptorSet& descriptorSet,
    std::vector<vkgs::Buffer*>& buffers
) {

    // Prepare buffer info
    std::unique_ptr<VkDescriptorBufferInfo[]> bufferInfos = std::make_unique<VkDescriptorBufferInfo[]>(buffers.size());
    for (int i = 0; i < buffers.size(); ++i) {
        bufferInfos[i] = { *buffers[i], 0, buffers[i]->size() };
    }

    // Write descriptor sets
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());

    for (int i = 0; i < buffers.size(); ++i) {
        descriptorWrites[i] = {};
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSets(device, buffers.size(), descriptorWrites.data(), 0, nullptr);
}


void VulkanComputeExample::setProjectionRenderPipeline()
{
    const uint32_t bindingCount = 24;
    createPipelineLayout(
        pipe_project_render,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.xyz_array,       //0
        &splat_buffers.scale_array,     //1
        &splat_buffers.rotate_array,    //2
        &splat_buffers.intr_params_render,     //3
        &splat_buffers.extr_params_render,     //4
        &splat_buffers.visible,         //5
        &splat_buffers.uvd,             //6
        &splat_buffers.cov2d,           //7
        &splat_buffers.cov3d,           //8
        &splat_buffers.radius,          //9
        &splat_buffers.tiles,           //10
        &splat_buffers.tiles_rect,      //11
        &splat_buffers.dL_duvd,         //12
        &splat_buffers.dL_dcov2d,       //13
        &splat_buffers.dL_dxyz,         //14
        &splat_buffers.dL_dscale,       //15
        &splat_buffers.dL_drotate,      //16
        &splat_buffers.dL_dintr_params, //17
        &splat_buffers.dL_dextr_params, //18
        // activation patch buffers
        &splat_buffers.opacity,         //19
        &splat_buffers.opacity_act_array,     //20
        &splat_buffers.dL_dopacity_act_array,     //21
        &splat_buffers.dL_dopacity,     //22
        & splat_buffers.curPointsNum //23
    };

    updateDescriptorSet(pipe_project_render.descriptorSet, splatBuffers);

    pipe_project_render.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/project_ewa_forward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/project_ewa_forward.comp.spv";
    std::cout << "[setProjectionPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_project_render.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_project_render.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setProjectionPipeline] Projection pipeline set" << std::endl;
}

void VulkanComputeExample::setProjectionPipeline()
{
    const uint32_t bindingCount = 24;
    createPipelineLayout(
        pipe_project,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.xyz_array,       //0
        &splat_buffers.scale_array,     //1
        &splat_buffers.rotate_array,    //2
        &splat_buffers.intr_params,     //3
        &splat_buffers.extr_params,     //4
        &splat_buffers.visible,         //5
        &splat_buffers.uvd,             //6
        &splat_buffers.cov2d,           //7
        &splat_buffers.cov3d,           //8
        &splat_buffers.radius,          //9
        &splat_buffers.tiles,           //10
        &splat_buffers.tiles_rect,      //11
        &splat_buffers.dL_duvd,         //12
        &splat_buffers.dL_dcov2d,       //13
        &splat_buffers.dL_dxyz,         //14
        &splat_buffers.dL_dscale,       //15
        &splat_buffers.dL_drotate,      //16
        &splat_buffers.dL_dintr_params, //17
        &splat_buffers.dL_dextr_params, //18
        // activation patch buffers
        &splat_buffers.opacity,         //19
        &splat_buffers.opacity_act_array,     //20
        &splat_buffers.dL_dopacity_act_array,     //21
        &splat_buffers.dL_dopacity,     //22
        &splat_buffers.curPointsNum //23
    };

    updateDescriptorSet(pipe_project.descriptorSet, splatBuffers);

    pipe_project.pipelines.resize(2);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/project_ewa_forward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/project_ewa_forward.comp.spv";
    std::cout << "[setProjectionPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_project.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_project.pipelines[FORWARD_PIPE]
        )
    );

    shaderCode = getPath("shader/computeshader/project_ewa_backward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/project_ewa_backward.comp.spv";
    std::cout << "[setProjectionPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_project.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_project.pipelines[BACKWARD_PIPE]
        )
    );

    std::cout << "[setProjectionPipeline] Projection pipeline set" << std::endl;
}

void VulkanComputeExample::setKNNPipeline()
{
    const uint32_t bindingCount = 24;
    createPipelineLayout(
        pipe_knn,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.xyz_array,       //0
        & splat_buffers.scale_array,     //1
    };

    updateDescriptorSet(pipe_knn.descriptorSet, splatBuffers);

    pipe_knn.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/simpleSimpleKnn.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/simpleSimpleKnn.comp.spv";
    std::cout << "[setKNNPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_knn.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_knn.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setKNNPipeline] knn pipeline set" << std::endl;
}

void VulkanComputeExample::setInverseIndexPipeline()
{
    const uint32_t bindingCount = 6;
    createPipelineLayout(
        pipe_inverse,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.cov2d,       //0
        &splat_buffers.visible,     //1
        &splat_buffers.conic,       //2
        &splat_buffers.dL_dconic,       //3
        &splat_buffers.dL_dcov2d,       //4
        & splat_buffers.curPointsNum //5
    };

    updateDescriptorSet(pipe_inverse.descriptorSet, splatBuffers);

    pipe_inverse.pipelines.resize(2);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/inverse_cov2_forward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/inverse_cov2_forward.comp.spv";
    std::cout << "[setInverseIndexPipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_inverse.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_inverse.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setInverseIndexPipeline] Inverse index pipeline set" << std::endl;

    shaderCode = getPath("shader/computeshader/inverse_cov2_backward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/inverse_cov2_backward.comp.spv";
    std::cout << "[setInverseIndexPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_inverse.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_inverse.pipelines[BACKWARD_PIPE]
        )
    );

    std::cout << "[setInverseIndexPipeline] Inverse index pipeline set" << std::endl;
}

void VulkanComputeExample::setAlphaBlendingPipeline()
{
    const uint32_t bindingCount = 17;
    createPipelineLayout(
        pipe_blending,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.uvd,       //0
        &splat_buffers.conic,     //1
        &splat_buffers.opacity_act_array,   //2
        &splat_buffers.feature,   //3
        &sort_buffers.sorted_idx_array,            //4
        &splat_buffers.tile_range,      //5
        &splat_buffers.background,//6
        &splat_buffers.rfeat,     //7
        &splat_buffers.final_T,   //8
        &splat_buffers.ncontrib,  //9
        &splat_buffers.dL_drfeat,  //10
        &splat_buffers.dL_duvd,  //11
        &splat_buffers.dL_dconic,  //12
        &splat_buffers.dL_dopacity_act_array,  //13
        &splat_buffers.dL_dfeature,  //14
        &splat_buffers.dL_dndc,  //15
        &splat_buffers.dL_dbackground,  //16
    };

    updateDescriptorSet(pipe_blending.descriptorSet, splatBuffers);

    pipe_blending.pipelines.resize(2);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/alpha_blending_forward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/alpha_blending_forward.comp.spv";
    std::cout << "[setAlphaBlendingPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_blending.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_blending.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setAlphaBlendingPipeline] Alpha blending pipeline set" << std::endl;

    shaderCode = getPath("shader/computeshader/alpha_blending_backward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/alpha_blending_backward.comp.spv";
    std::cout << "[setAlphaBlendingPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_blending.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_blending.pipelines[BACKWARD_PIPE]
        )
    );

    std::cout << "[setAlphaBlendingPipeline] Alpha blending pipeline set" << std::endl;
}


void VulkanComputeExample::setAlphaBlendingIntPipeline()
{
    const uint32_t bindingCount = 17;
    createPipelineLayout(
        pipe_blending_int,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.uvd,       //0
        &splat_buffers.conic,     //1
        &splat_buffers.opacity_act_array,   //2
        &splat_buffers.feature,   //3
        &sort_buffers.sorted_idx_array,            //4
        &splat_buffers.tile_range,      //5
        &splat_buffers.background,//6
        &splat_buffers.rfeat,     //7
        &splat_buffers.final_T,   //8
        &splat_buffers.ncontrib,  //9
        &splat_buffers.dL_drfeat,  //10
        &splat_buffers.dL_duvd_int,  //11
        &splat_buffers.dL_dconic_int,  //12
        &splat_buffers.dL_dopacity_act_array_int,  //13
        &splat_buffers.dL_dfeature_int,  //14
        &splat_buffers.dL_dndc_int,  //15
        &splat_buffers.dL_dbackground,  //16
    };

    updateDescriptorSet(pipe_blending_int.descriptorSet, splatBuffers);

    pipe_blending_int.pipelines.resize(2);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/alpha_blending_forward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/alpha_blending_forward.comp.spv";
    std::cout << "[setAlphaBlendingPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_blending_int.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_blending_int.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setAlphaBlendingPipeline] Alpha blending pipeline set" << std::endl;

    shaderCode = getPath("shader/computeshader/alpha_blending_backward_int.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/alpha_blending_backward.comp.spv";
    std::cout << "[setAlphaBlendingPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_blending_int.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_blending_int.pipelines[BACKWARD_PIPE]
        )
    );

    std::cout << "[setAlphaBlendingPipeline] Alpha blending pipeline set" << std::endl;
}

void VulkanComputeExample::setIntToFloatGradientPipeline()
{
    const uint32_t bindingCount = 10;
    createPipelineLayout(
        pipe_int2float,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.dL_duvd_int,  //0
        &splat_buffers.dL_dconic_int,  //1
        &splat_buffers.dL_dopacity_act_array_int,  //2
        &splat_buffers.dL_dfeature_int,  //3
        &splat_buffers.dL_dndc_int,  //4
        &splat_buffers.dL_duvd,  //5
        &splat_buffers.dL_dconic,  //6
        &splat_buffers.dL_dopacity_act_array,  //7
        &splat_buffers.dL_dfeature,  //8
        &splat_buffers.dL_dndc,  //9
    };
    
    updateDescriptorSet(pipe_int2float.descriptorSet, splatBuffers);

    pipe_int2float.pipelines.resize(2);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/int_to_float_gradient.comp.spv");

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_int2float.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_int2float.pipelines[BACKWARD_PIPE]
        )
    );
    std::cout << "[setIntToFloatGradientPipeline] Int 2 Float pipeline set" << std::endl;
}


void VulkanComputeExample::setComputeSHPipeline()
{
    const uint32_t bindingCount = 10;
    createPipelineLayout(
        pipe_sh,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 8 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.sh_coeff,       //0
        &splat_buffers.xyz_array,     //1
        &splat_buffers.extr_params,   //2
        &splat_buffers.visible,   //3
        &splat_buffers.feature,            //4
        &splat_buffers.dL_dfeature,      //5
        &splat_buffers.dL_dshs_array,//6
        &splat_buffers.dL_dxyz,     //7
        &splat_buffers.dL_dextr_params,   //8
        & splat_buffers.curPointsNum //9
    };

    updateDescriptorSet(pipe_sh.descriptorSet, splatBuffers);

    pipe_sh.pipelines.resize(2);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/sh_forward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/sh_forward.comp.spv";
    std::cout << "[setComputeSHPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_sh.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_sh.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setComputeSHPipeline] compute SH pipeline set" << std::endl;

    shaderCode = getPath("shader/computeshader/sh_backward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/sh_backward.comp.spv";
    std::cout << "[setComputeSHPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_sh.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_sh.pipelines[BACKWARD_PIPE]
        )
    );

    std::cout << "[setComputeSHPipeline] compute SH pipeline set" << std::endl;
}

void VulkanComputeExample::setComputeSHRenderPipeline()
{
    const uint32_t bindingCount = 10;
    createPipelineLayout(
        pipe_sh_render,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 8 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.sh_coeff,       //0
        &splat_buffers.xyz_array,     //1
        &splat_buffers.extr_params_render,   //2
        &splat_buffers.visible,   //3
        &splat_buffers.feature,            //4
        &splat_buffers.dL_dfeature,      //5
        &splat_buffers.dL_dshs_array,//6
        &splat_buffers.dL_dxyz,     //7
        &splat_buffers.dL_dextr_params,   //8
        & splat_buffers.curPointsNum //9
    };

    updateDescriptorSet(pipe_sh_render.descriptorSet, splatBuffers);

    pipe_sh_render.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/sh_forward.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/sh_forward.comp.spv";
    std::cout << "[setComputeSHPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_sh_render.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_sh_render.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setComputeSHRenderPipeline] compute SH pipeline render set" << std::endl;

}

void VulkanComputeExample::setOptimizitionPipeline()
{
    const uint32_t bindingCount = 22;
    createPipelineLayout(
        pipe_optimize,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 8 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.xyz_array,                // 0
        &splat_buffers.scale_array,              // 1
        &splat_buffers.rotate_array,             // 2
        &splat_buffers.sh_coeff,                  // 3
        &splat_buffers.opacity,                  // 4
        &splat_buffers.dL_dxyz,                  // 5
        &splat_buffers.dL_dscale,                // 6
        &splat_buffers.dL_drotate,               // 7
        &splat_buffers.dL_dshs_array,              // 8
        &splat_buffers.dL_dopacity,              // 9
        &splat_buffers.xyz_array_exp_avg,        // 10
        &splat_buffers.scale_array_exp_avg,      // 11
        &splat_buffers.rotate_array_exp_avg,     // 12
        &splat_buffers.feature_array_exp_avg,    // 13
        &splat_buffers.opacity_array_exp_avg,    // 14
        &splat_buffers.xyz_array_avg,            // 15
        &splat_buffers.scale_array_avg,          // 16
        &splat_buffers.rotate_array_avg,         // 17
        &splat_buffers.feature_array_avg,        // 18
        &splat_buffers.opacity_array_avg,         // 19
        &splat_buffers.curPointsNum, // 20
        //& splat_buffers.visible,
        & splat_buffers.radius

    };

    updateDescriptorSet(pipe_optimize.descriptorSet, splatBuffers);

    pipe_optimize.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/sparse_adam.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/sparse_adam.comp.spv";
    std::cout << "[setComputeKeyPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_optimize.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_optimize.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setOptimizitionPipeline] Optimizition Pipeline set" << std::endl;
}

void VulkanComputeExample::setLossPipeline()
{
    const uint32_t bindingCount = 4;
    createPipelineLayout(
        pipe_loss,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.input_image_array,                // 0
        &splat_buffers.rfeat,              // 1
        &splat_buffers.dL_drfeat,             // 2
        &splat_buffers.psnr_sum
    };

    updateDescriptorSet(pipe_loss.descriptorSet, splatBuffers);

    pipe_loss.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/loss.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/loss.comp.spv";
    std::cout << "[setComputeKeyPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_loss.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_loss.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setLossPipeline] Loss Pipeline set" << std::endl;
}

void VulkanComputeExample::setComputeKeyPipeline()
{
    const uint32_t bindingCount = 10;
    createPipelineLayout(
        pipe_compute_key,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.uvd,                    //0
        &splat_buffers.tiles,                //1
        &splat_buffers.tiles_rect,                  //2
        &splat_buffers.visible,                //3
        &sort_buffers.sortKBufferEven,                           //4
        &sort_buffers.unsort_index,                  //5
        &sort_buffers.splat_visible_point_count,     //6
        &splat_buffers.curPointsNum, //7
        &splat_buffers.conic,
        &splat_buffers.opacity_act_array
    };

    updateDescriptorSet(pipe_compute_key.descriptorSet, splatBuffers);

    pipe_compute_key.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/compute_keys.comp.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/compute_keys.comp.spv";
    std::cout << "[setComputeKeyPipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_compute_key.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_compute_key.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setComputeKeyPipeline] Compute key pipeline set" << std::endl;
}

void VulkanComputeExample::setComputeTilePipeline()
{
    const uint32_t bindingCount = 6;
    createPipelineLayout(
        pipe_compute_tile,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &sort_buffers.sortKBufferEven,                           //0
        &sort_buffers.unsort_index,                  //1
        &sort_buffers.sortVBufferEven,                         //2
        &sort_buffers.splat_visible_point_count,     //3
        &splat_buffers.tile_range,                   //4
        &sort_buffers.sorted_idx_array,              //5
    };

    updateDescriptorSet(pipe_compute_tile.descriptorSet, splatBuffers);

    pipe_compute_tile.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/compute_tile_range.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/compute_tile_range.comp.spv";
    std::cout << "[setComputeTilePipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_compute_tile.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_compute_tile.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setComputeTilePipeline] Compute tile pipeline set" << std::endl;
}

void VulkanComputeExample::setDensificationStatsPipeline()
{
    const uint32_t bindingCount = 6;
    createPipelineLayout(
        pipe_densification_stats,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.radius,                           //0
        //& splat_buffers.dL_dndc,                  //1
        & splat_buffers.dL_duvd,                  //1
        & splat_buffers.curPointsNum,                         //2
        & splat_buffers.max_radii2D,     //3
        & splat_buffers.xyz_gradient_accum,                   //4
        & splat_buffers.denom,              //5
    };

    updateDescriptorSet(pipe_densification_stats.descriptorSet, splatBuffers);

    pipe_densification_stats.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/add_densification_stats.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/add_densification_stats.comp.spv";
    std::cout << "[setDensificationStatsPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_densification_stats.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_densification_stats.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setDensificationStatsPipeline] Compute tile pipeline set" << std::endl;
}

void VulkanComputeExample::setMeanGradPipeline()
{
    const uint32_t bindingCount = 5;
    createPipelineLayout(
        pipe_mean_grad,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.xyz_gradient_accum,                           //0
        & splat_buffers.denom,                  //1
        & splat_buffers.curPointsNum,                         //2
        & splat_buffers.xyz_gradient_mean,     //3
        & sort_buffers.densify_idx //4
    };

    updateDescriptorSet(pipe_mean_grad.descriptorSet, splatBuffers);

    pipe_mean_grad.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/cal_mean_grads.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/cal_mean_grads.comp.spv";
    std::cout << "[setMeanGradPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_mean_grad.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_mean_grad.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setMeanGradPipeline] Compute tile pipeline set" << std::endl;
}

void VulkanComputeExample::setClonePipeline()
{
    const uint32_t bindingCount = 7;
    createPipelineLayout(
        pipe_clone,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 8 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
         &sort_buffers.densify_idx,                           //0
         &splat_buffers.xyz_array,                  //1
         &splat_buffers.sh_coeff,                         //2
         &splat_buffers.opacity,     //3
         &splat_buffers.scale_array,                   //4
         &splat_buffers.rotate_array,              //5
         &splat_buffers.curPointsNum   //6
    };

    updateDescriptorSet(pipe_clone.descriptorSet, splatBuffers);

    pipe_clone.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/densify_and_clone.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/densify_and_clone.comp.spv";
    std::cout << "[setClonePipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_clone.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_clone.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setClonePipeline] Compute tile pipeline set" << std::endl;
}

void VulkanComputeExample::setPrunePipeline()
{
    const uint32_t bindingCount = 17;
    createPipelineLayout(
        pipe_prune,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 8 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
         &sort_buffers.densify_idx,                           //0
         &splat_buffers.xyz_array,                  //1
         &splat_buffers.sh_coeff,                         //2
         &splat_buffers.opacity,     //3
         &splat_buffers.scale_array,                   //4
         &splat_buffers.rotate_array,              //5
         & splat_buffers.xyz_array_exp_avg,        // 6
        & splat_buffers.scale_array_exp_avg,      // 7
        & splat_buffers.rotate_array_exp_avg,     // 8
        & splat_buffers.feature_array_exp_avg,    // 9
        & splat_buffers.opacity_array_exp_avg,    // 10
        & splat_buffers.xyz_array_avg,            // 11
        & splat_buffers.scale_array_avg,          // 12
        & splat_buffers.rotate_array_avg,         // 13
        & splat_buffers.feature_array_avg,        // 14
        & splat_buffers.opacity_array_avg,         // 15
         &splat_buffers.curPointsNum   //16
    };

    updateDescriptorSet(pipe_prune.descriptorSet, splatBuffers);

    pipe_prune.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/prune_points.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/prune_points.comp.spv";
    std::cout << "[setPrunePipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_prune.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_prune.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setPrunePipeline] prune pipeline set" << std::endl;
}

void VulkanComputeExample::setSplitPipeline()
{
    const uint32_t bindingCount = 7;
    createPipelineLayout(
        pipe_split,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 8 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &sort_buffers.densify_idx,                           //0
        & splat_buffers.xyz_array,                  //1
        & splat_buffers.sh_coeff,                         //2
        & splat_buffers.opacity,     //3
        &splat_buffers.scale_array,                   //4
        & splat_buffers.rotate_array,              //5
        & splat_buffers.curPointsNum   //6
    };

    updateDescriptorSet(pipe_split.descriptorSet, splatBuffers);

    pipe_split.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/densify_and_split.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/densify_and_split.comp.spv";
    std::cout << "[setSplitPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_split.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_split.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setSplitPipeline] Compute tile pipeline set" << std::endl;
}

void VulkanComputeExample::setResetOpacityPipeline()
{
    const uint32_t bindingCount = 2;
    createPipelineLayout(
        pipe_reset_opacity,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.opacity,                           //0
        &splat_buffers.curPointsNum,                  //1
    };

    updateDescriptorSet(pipe_reset_opacity.descriptorSet, splatBuffers);

    pipe_reset_opacity.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/reset_opacity.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/reset_opacity.comp.spv";
    std::cout << "[setResetOpacityPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_reset_opacity.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_reset_opacity.pipelines[FORWARD_PIPE]
        )
    );

    std::cout << "[setResetOpacityPipeline] Reset opacity pipeline set" << std::endl;
}

void VulkanComputeExample::setSortPipeline()
{
//     // create sorter
//     VrdxSorterCreateInfo sorter_info = {};
//     sorter_info.physicalDevice = physicalDevice;
//     sorter_info.device = device;
// //   sorter_info.pipelineCache = context_.pipeline_cache();
//     vrdxCreateSorter(&sorter_info, &sort_buffers.sorter);
//
//     // preallocate sorter storage
//     VrdxSorterStorageRequirements requirements;
//     vrdxGetSorterKeyValueStorageRequirements(sort_buffers.sorter, MAX_SPLAT_COUNT * SORT_SIZE, &requirements);
//     sort_buffers.sort_storage = vkgs::Buffer(allocator, requirements.size, requirements.usage);

    sort_buffers.splat_visible_point_count = vkgs::Buffer(
        allocator, 
        sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    );

    sort_buffers.splat_visible_point_count_cpu = vkgs::CpuBuffer(
        allocator, sizeof(uint32_t)
    );

//     sort_buffers.key = vkgs::Buffer(
//         allocator, 
//         MAX_SPLAT_COUNT * sizeof(uint32_t) * SORT_SIZE,
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
//         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
//         VK_BUFFER_USAGE_TRANSFER_SRC_BIT
//     );
//
//     sort_buffers.index = vkgs::Buffer(
//         allocator, 
//         MAX_SPLAT_COUNT * sizeof(uint32_t) * SORT_SIZE,
//         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
//         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
//         VK_BUFFER_USAGE_TRANSFER_SRC_BIT
//     );

    sort_buffers.unsort_index = vkgs::Buffer(
        allocator, 
        MAX_SPLAT_COUNT * sizeof(uint32_t) * SORT_SIZE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    );

    sort_buffers.sorted_idx_array = vkgs::Buffer(
        allocator, 
        MAX_SPLAT_COUNT * sizeof(uint32_t) * SORT_SIZE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    );
    
    uint32_t numWorkgroups = (splat_count * SORT_SIZE / 256 + 256 - 1) / 256;
    uint32_t radixBin = 256; //2^8
    sort_buffers.sortHistBuffer = vkgs::Buffer(
        allocator, 
        numWorkgroups * radixBin * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    );

    sort_buffers.sortKBufferEven = vkgs::Buffer(
        allocator, 
        MAX_SPLAT_COUNT * SORT_SIZE * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    );

    sort_buffers.sortKBufferOdd = vkgs::Buffer(
        allocator, 
        MAX_SPLAT_COUNT * SORT_SIZE * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    );

    sort_buffers.sortVBufferEven = vkgs::Buffer(
        allocator, 
        MAX_SPLAT_COUNT * SORT_SIZE * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    );

    sort_buffers.sortVBufferOdd = vkgs::Buffer(
        allocator, 
        MAX_SPLAT_COUNT * SORT_SIZE * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    );

    setHistEvenPipeline();
    setHistOddPipeline();
    setSortEvenPipeline();
    setSortOddPipeline();
}

void VulkanComputeExample::setHistEvenPipeline()
{
    uint32_t bindingCount = 3;
    createPipelineLayout(
        pipe_hist_even,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &sort_buffers.sortKBufferEven,
        &sort_buffers.sortHistBuffer,
        &sort_buffers.splat_visible_point_count
    };

    updateDescriptorSet(pipe_hist_even.descriptorSet, splatBuffers);

    pipe_hist_even.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/sort/hist.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/sort/hist.spv";
    std::cout << "[setInverseIndexPipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_hist_even.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_hist_even.pipelines[0]
        )
    );
}

void VulkanComputeExample::setHistOddPipeline()
{
    uint32_t bindingCount = 3;
    createPipelineLayout(
        pipe_hist_odd,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &sort_buffers.sortKBufferOdd,
        &sort_buffers.sortHistBuffer,
        &sort_buffers.splat_visible_point_count
    };

    updateDescriptorSet(pipe_hist_odd.descriptorSet, splatBuffers);

    pipe_hist_odd.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/sort/hist.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/sort/hist.spv";
    std::cout << "[setInverseIndexPipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_hist_odd.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_hist_odd.pipelines[0]
        )
    ); 
}

void VulkanComputeExample::setSortEvenPipeline()
{
    uint32_t bindingCount = 6;
    createPipelineLayout(
        pipe_sort_even,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &sort_buffers.sortKBufferEven,
        &sort_buffers.sortKBufferOdd,
        &sort_buffers.sortVBufferEven,
        &sort_buffers.sortVBufferOdd,
        &sort_buffers.sortHistBuffer,
        &sort_buffers.splat_visible_point_count
    };

    updateDescriptorSet(pipe_sort_even.descriptorSet, splatBuffers);

    pipe_sort_even.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/sort/sort.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/sort/sort.spv";
    std::cout << "[setInverseIndexPipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_sort_even.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_sort_even.pipelines[0]
        )
    ); 
}

void VulkanComputeExample::setSortOddPipeline()
{
    uint32_t bindingCount = 6;
    createPipelineLayout(
        pipe_sort_odd,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &sort_buffers.sortKBufferOdd,
        &sort_buffers.sortKBufferEven,
        &sort_buffers.sortVBufferOdd,
        &sort_buffers.sortVBufferEven,
        &sort_buffers.sortHistBuffer,
        &sort_buffers.splat_visible_point_count
    };

    updateDescriptorSet(pipe_sort_odd.descriptorSet, splatBuffers);

    pipe_sort_odd.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/sort/sort.spv");
#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/sort/sort.spv";
    std::cout << "[setInverseIndexPipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_sort_odd.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_sort_odd.pipelines[0]
        )
    ); 
}

void VulkanComputeExample::setImage2BufferPipeline()
{
    const uint32_t bindingCount = 2;
    createPipelineLayout(
        pipe_copy_img2buffer,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.input_image_alpha_array,                  //0
        &splat_buffers.input_image_array,                //1
    };

    updateDescriptorSet(pipe_copy_img2buffer.descriptorSet, splatBuffers);

    pipe_copy_img2buffer.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/rgba_to_buffer.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/rgba_to_buffer.comp.spv";
    std::cout << "[setImage2BufferPipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_copy_img2buffer.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_copy_img2buffer.pipelines[0]
        )
    );

    std::cout << "[setImage2BufferPipeline]  Compute Image2BufferPipeline set" << std::endl;
}

void VulkanComputeExample::setBuffer2ImagePipeline()
{
    const uint32_t bindingCount = 2;
    createPipelineLayout(
        pipe_copy_buffer2img,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.rfeat,                  //0
        &splat_buffers.rfeat_alpha,                //1
    };

    updateDescriptorSet(pipe_copy_buffer2img.descriptorSet, splatBuffers);

    pipe_copy_buffer2img.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/buffer_to_rgba.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/buffer_to_rgba.comp.spv";
    std::cout << "[setBuffer2ImagePipeline] shaderCode: " << shaderCode << std::endl;
#endif
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_copy_buffer2img.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_copy_buffer2img.pipelines[0]
        )
    );

    std::cout << "[setBuffer2ImagePipeline]  Compute Buffer2ImagePipeline set" << std::endl;
}

void VulkanComputeExample::setBufferFillPipeline()
{
    const uint32_t bindingCount = 2;
    createPipelineLayout(
        pipe_buffer_fill,
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 4 // Push constant size (xyz_size; width; height;)
    );

    std::vector<vkgs::Buffer*> splatBuffers = {
        &splat_buffers.visible,                  //0
        &splat_buffers.curPointsNum,                //1
    };

    updateDescriptorSet(pipe_buffer_fill.descriptorSet, splatBuffers);

    pipe_buffer_fill.pipelines.resize(1);

    // Load SPIR-V shader code
    std::string shaderCode = getPath("shader/computeshader/buffer_fill.comp.spv");

#ifdef VULKAN_PC
    shaderCode = "../../shaders/glsl/computeshader/buffer_fill.comp.spv";
    std::cout << "[setBufferFillPipeline] shaderCode: " << shaderCode << std::endl;
#endif

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);;
    pipelineInfo.layout = pipe_buffer_fill.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe_buffer_fill.pipelines[0]
        )
    );

    std::cout << "[setBufferFillPipeline] Buffer Fill Pipeline set" << std::endl;
}
void VulkanComputeExample::prepareStorageImage()
{
    
    // Create a random initial image
    std::vector<float> currentImageData(textureColorMap.width * textureColorMap.height);
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& pixel : currentImageData) {
        pixel = dist(rng);
    }
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    VkFormatProperties formatProperties;
    // Get device properties for the requested texture format
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
    // Check if requested image format supports image storage operations required for storing pixel from the compute shader
    assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

    // Prepare blit target texture
    storageImage.width = textureColorMap.width;
    storageImage.height = textureColorMap.height;

    VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.extent = { storageImage.width, storageImage.height, 1 };
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // Image will be sampled in the fragment shader and used as storage target in the compute shader
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.flags = 0;
    // If compute and graphics queue family indices differ, we create an image that can be shared between them
    // This can result in worse performance than exclusive sharing mode, but save some synchronization to keep the sample simple
    std::vector<uint32_t> queueFamilyIndices;
    if (vulkanDevice->queueFamilyIndices.graphics != vulkanDevice->queueFamilyIndices.compute) {
        queueFamilyIndices = {
            vulkanDevice->queueFamilyIndices.graphics,
            vulkanDevice->queueFamilyIndices.compute
        };
        imageCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        imageCreateInfo.queueFamilyIndexCount = 2;
        imageCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    }
    // VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &storageImage.image));

    // VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    // VkMemoryRequirements memReqs;
    // vkGetImageMemoryRequirements(device, storageImage.image, &memReqs);
    // memAllocInfo.allocationSize = memReqs.size;
    // memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    // VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &storageImage.deviceMemory));
    // VK_CHECK_RESULT(vkBindImageMemory(device, storageImage.image, storageImage.deviceMemory, 0));

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    
    VK_CHECK_RESULT(vmaCreateImage(allocator, &imageCreateInfo, &allocInfo, &storageImage.image, &storageImageAllocation, nullptr));

    // Transfer the image data to the buffers
    // void* data;
    size_t imageSize = textureColorMap.width * textureColorMap.height * sizeof(float);
    // vkMapMemory(device, storageImage.deviceMemory, 0, imageSize, 0, &data);
    // memcpy(data, currentImageData.data(), imageSize);
    // vkUnmapMemory(device, storageImage.deviceMemory);
    
    void* mappedData;
    vmaMapMemory(allocator, storageImageAllocation, &mappedData);
    memcpy(mappedData, currentImageData.data(), imageSize);
    vmaUnmapMemory(allocator, storageImageAllocation);
    
    // Transition image to the general layout, so we can use it as a storage image in the compute shader
    VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    storageImage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    vks::tools::setImageLayout(layoutCmd, storageImage.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, storageImage.imageLayout);
    vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);

    // Create sampler
    VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeV = sampler.addressModeU;
    sampler.addressModeW = sampler.addressModeU;
    sampler.mipLodBias = 0.0f;
    sampler.maxAnisotropy = 1.0f;
    sampler.compareOp = VK_COMPARE_OP_NEVER;
    sampler.minLod = 0.0f;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &storageImage.sampler));

    // Create image view
    VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
    view.image = VK_NULL_HANDLE;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    view.image = storageImage.image;
    VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &storageImage.view));

    // Initialize a descriptor for later use
    storageImage.descriptor.imageLayout = storageImage.imageLayout;
    storageImage.descriptor.imageView = storageImage.view;
    storageImage.descriptor.sampler = storageImage.sampler;
    storageImage.device = vulkanDevice;
}

void VulkanComputeExample::resizeAndUploadTexture(
    vks::Texture2D& texture,
    const unsigned char* original_image_data,
    int width, int height,
    float scale,
    vks::VulkanDevice* vulkanDevice,
    VkQueue queue
) {
    // 计算新尺寸
    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);

    if (new_width <= 0 || new_height <= 0) {
        std::cerr << "Invalid scaled dimensions: " << new_width << "x" << new_height << std::endl;
        return;
    }

    // 分配缩小后的图像缓冲区
    unsigned char* resizedImage = new unsigned char[new_width * new_height * 4];

    // 计算 stride（每行字节数）
    int input_stride = width * 4;
    int output_stride = new_width * 4;

    // 使用 stb_image_resize 进行缩放
    bool success = stbir_resize_uint8_linear(
        original_image_data, width, height, input_stride, // 输入图像
        resizedImage, new_width, new_height, output_stride, // 输出图像
        STBIR_RGBA // 颜色格式（RGBA）
    );

    if (!success) {
        std::cerr << "Failed to resize image!" << std::endl;
        delete[] resizedImage;
        return;
    }

    // 上传到 Vulkan 纹理
    texture.fromBuffer(
        resizedImage,               // 图像数据
        new_width * new_height * 4,  // 数据大小
        VK_FORMAT_R8G8B8A8_UNORM,    // 图像格式
        new_width,                   // 宽度
        new_height,                   // 高度
        vulkanDevice,
        queue,
        VK_FILTER_LINEAR,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // 释放缩小后的图像数据
    delete[] resizedImage;
}

void VulkanComputeExample::loadAssets(float scale) {
    std::cout << "Loading assets..." << std::endl;
    
    std::string filename = inputGaussian.img_list[0];
    std::cout << "[loadAssets] filename: " << filename << std::endl;
    
    // 使用stb_image加载图像文件（无论是PNG还是JPEG）
    int width, height, channels;
    unsigned char* image_data = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!image_data) {
        std::cerr << "Failed to load image file: " << filename << std::endl;
        return;
    }

    // 使用 Vulkan 的 fromBuffer 方法将图像数据上传
    textureColorMap.fromBuffer(
        image_data,                // 图像数据
        width * height * 4,         // 数据大小，RGBA 格式需要每个像素 4 字节
        VK_FORMAT_R8G8B8A8_UNORM,   // 图像格式
        width,                      // 图像宽度
        height,                     // 图像高度
        vulkanDevice,               // Vulkan 设备
        queue,                      // Vulkan 队列
        VK_FILTER_LINEAR,           // 滤波器
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,  // 用途
        VK_IMAGE_LAYOUT_GENERAL     // 图像布局
    );
    // 释放加载的图像数据
    stbi_image_free(image_data);

    image_width = textureColorMap.width;
    image_height = textureColorMap.height;

    // 遍历 inputGaussian.img_list 一次性加载图像
    for (const std::string& filename : inputGaussian.img_list) {
        // 加载图像
        int width, height, channels;
        unsigned char* image_data = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!image_data) {
            std::cerr << "Failed to load image file: " << filename << std::endl;
            continue;
        }

        // 创建 Vulkan 纹理对象（原始分辨率）
        vks::Texture2D texture_original;
        texture_original.fromBuffer(
            image_data,              // 原始图像数据
            width * height * 4,      // 数据大小
            VK_FORMAT_R8G8B8A8_UNORM, // 图像格式
            width,                    // 宽度
            height,                   // 高度
            vulkanDevice,
            queue,
            VK_FILTER_LINEAR,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_LAYOUT_GENERAL
        );

        // 创建 Vulkan 纹理对象（缩小分辨率）
        vks::Texture2D texture_resized;
        resizeAndUploadTexture(texture_resized, image_data, width, height, scale, vulkanDevice, queue);

        // 存入对应的 vector
        inputGaussian.images.push_back(texture_original);
        inputGaussian.images_low_res.push_back(texture_resized);

        // 释放 STB 读取的图像数据
        stbi_image_free(image_data);
    }

    std::cout << "Total images loaded: " << inputGaussian.images.size() << std::endl;
}




void VulkanComputeExample::buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    clearValues[0].color = defaultClearColor;
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = screenWidth;
    renderPassBeginInfo.renderArea.extent.height = screenHeight;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        renderPassBeginInfo.framebuffer = frameBuffers[i];

        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        VkImageMemoryBarrier imageMemoryBarrier = {};
        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.image = storageImage.image;
        imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(
            drawCmdBuffers[i],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_FLAGS_NONE,
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier);

        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = vks::initializers::viewport((float)screenWidth * 0.5f, (float)screenHeight, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        VkRect2D scissor = vks::initializers::rect2D(screenWidth, screenHeight, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertexBuffer.buffer, offsets);
        vkCmdBindIndexBuffer(drawCmdBuffers[i], indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Left (pre compute)
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSetPreCompute, 0, NULL);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
        vkCmdDrawIndexed(drawCmdBuffers[i], indexCount, 1, 0, 0, 0);

        // Right (post compute)
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSetPostCompute, 0, NULL);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);

        viewport.x = (float)screenWidth / 2.0f;
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdDrawIndexed(drawCmdBuffers[i], indexCount, 1, 0, 0, 0);

        vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VulkanComputeExample::bindPipeline2CommandBuffer(
    const VulkanComputeExample::Compute3DGS& pipe, 
    const int PIPELINE_INDEX,
    uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
    bool float_constant
) {
    vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipelines[PIPELINE_INDEX]);
    vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipelineLayout, 0, 1, &pipe.descriptorSet, 0, nullptr);

    // Push constants
    uint32_t xyz_size = splat_count;
    if (float_constant) {
        float start_lr = 0.00016;
        float end_lr = 0.0000016;
        // Linear learning rate decay
//         float pos_lr = (start_lr - ((start_lr - end_lr) * (training_step / max_training_step)));
        float pos_lr = (start_lr - ((start_lr - end_lr) * (training_step / 30000)));
        
        std::vector<float> pushConstants = { static_cast<float>(xyz_size), static_cast<float>((int)(current_scale * image_width)), static_cast<float>((int)(current_scale * image_height)), pos_lr, static_cast<float>(sh_degree), 0, 0, 0};
        vkCmdPushConstants(compute.commandBuffer, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 8, pushConstants.data());
    }else{
        std::vector<uint32_t> pushConstants = { xyz_size, (unsigned int)(current_scale * image_width), (unsigned int)(current_scale * image_height), 0};
        vkCmdPushConstants(compute.commandBuffer, pipe.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4, pushConstants.data());
    }

    // Dispatch compute shader
    // uint32_t groupCountX = (MAX_SPLAT_COUNT + 256 - 1) / 256; // Ceiling division
    vkCmdDispatch(compute.commandBuffer, dispatchX, dispatchY, dispatchZ);

    ComputeShaderMemoryBarrierBlock();
}

void VulkanComputeExample::bindPipeline2CommandBufferPushConstants(
    const VulkanComputeExample::Compute3DGS& pipe, 
    const int PIPELINE_INDEX,
    uint32_t dispatchX, uint32_t dispatchY, uint32_t dispatchZ,
    std::vector<uint32_t>& pushConstants
) {
    vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipelines[PIPELINE_INDEX]);
    vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipelineLayout, 0, 1, &pipe.descriptorSet, 0, nullptr);

    // print out pushConstants
    // for (size_t i = 0; i < pushConstants.size(); ++i) {
    //     std::cout << "pushConstants[" << i << "] = " << pushConstants[i] << std::endl;
    // }

    // Push constants
    vkCmdPushConstants(
        compute.commandBuffer, 
        pipe.pipelineLayout, 
        VK_SHADER_STAGE_COMPUTE_BIT, 
        0, sizeof(uint32_t) * 4, 
        pushConstants.data()
    );

    // Dispatch compute shader
    // uint32_t groupCountX = (MAX_SPLAT_COUNT + 256 - 1) / 256; // Ceiling division
    vkCmdDispatch(compute.commandBuffer, dispatchX, dispatchY, dispatchZ);

    ComputeShaderMemoryBarrierBlock();
}

void VulkanComputeExample::ComputeMemoryBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        compute.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanComputeExample::TransComputeBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        compute.commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanComputeExample::ComputeTransBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        compute.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}

void VulkanComputeExample::ComputeShaderMemoryBarrierBlock()
{
    // Add memory barrier to ensure writes are visible
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        compute.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );
}
void VulkanComputeExample::createGpu2CpuCopy(vkgs::Buffer &buffer)
{
    vkgs::CpuBuffer cpu_buffer = buffer.cpu_buffer;
    VkBufferCopy region = {};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = buffer.size();
    vkCmdCopyBuffer(
        compute.commandBuffer,
        buffer,
        cpu_buffer,
        1, &region
    );
}

void VulkanComputeExample::buildComputeCommandBuffer()
{
    vkQueueWaitIdle(compute.queue);

    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
    VkCommandBufferBeginInfo cmdBufInfo2 = vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));
    
//     vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[compute.pipelineIndex]);
//     vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
//     vkCmdDispatch(compute.commandBuffer, storageImage.width / 16, storageImage.height / 16, 1);
    
    uint32_t X = 0;
    uint32_t Y = 0;

    if (render_count == 0) {
        encodeCpu2Gpu();
        ComputeMemoryBarrierBlock();
    }
    encodePipeline2CommandBuffer();

    VK_CHECK_RESULT(vkEndCommandBuffer(compute.commandBuffer));
}

void VulkanComputeExample::generateQuad()
{
    std::vector<Vertex> vertices = {
        { {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f } },
        { { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f } },
        { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
        { {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } }
    };

    std::vector<uint32_t> indices = { 0,1,2, 2,3,0 };
    indexCount = static_cast<uint32_t>(indices.size());

    struct StagingBuffers {
        vks::Buffer vertices;
        vks::Buffer indices;
    } stagingBuffers;

    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
        &stagingBuffers.vertices, vertices.size() * sizeof(Vertex), vertices.data()));
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
        &stagingBuffers.indices, indices.size() * sizeof(uint32_t), indices.data()));

    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer, vertices.size() * sizeof(Vertex)));
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indexBuffer, indices.size() * sizeof(uint32_t)));

    vulkanDevice->copyBuffer(&stagingBuffers.vertices, &vertexBuffer, queue);
    vulkanDevice->copyBuffer(&stagingBuffers.indices, &indexBuffer, queue);

    stagingBuffers.vertices.destroy();
    stagingBuffers.indices.destroy();
}

void VulkanComputeExample::setupDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 64),
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2048);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
}

void VulkanComputeExample::prepareGraphics()
{
    VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
    VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &graphics.semaphore));

    VkSubmitInfo submitInfo = vks::initializers::submitInfo();
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &graphics.semaphore;
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK_RESULT(vkQueueWaitIdle(queue));

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1)
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

    VkDescriptorSetAllocateInfo allocInfo =
        vks::initializers::descriptorSetAllocateInfo(descriptorPool, &graphics.descriptorSetLayout, 1);

    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSetPreCompute));
    std::vector<VkWriteDescriptorSet> baseImageWriteDescriptorSets = {
        vks::initializers::writeDescriptorSet(graphics.descriptorSetPreCompute, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &graphics.uniformBuffer.descriptor),
        vks::initializers::writeDescriptorSet(graphics.descriptorSetPreCompute, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textureColorMap.descriptor)
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(baseImageWriteDescriptorSets.size()), baseImageWriteDescriptorSets.data(), 0, nullptr);

    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSetPostCompute));
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
        vks::initializers::writeDescriptorSet(graphics.descriptorSetPostCompute, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &graphics.uniformBuffer.descriptor),
        vks::initializers::writeDescriptorSet(graphics.descriptorSetPostCompute, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &storageImage.descriptor)
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&graphics.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(
        0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(
        1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(
        VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(
        1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(
        VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(
        dynamicStateEnables);
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    // std::string vshader_file = FileOperator::GetInstance()->GetFileAbsolutePath("shader/computeshader/texture.vert.spv");
    // std::string fshader_file = FileOperator::GetInstance()->GetFileAbsolutePath("shader/computeshader/texture.frag.spv");
    std::string vshader_file  = getPath("shader/computeshader/texture.vert.spv");
    std::string fshader_file  = getPath("shader/computeshader/texture.frag.spv");
#ifdef VULKAN_PC
    vshader_file = "../../shaders/glsl/computeshader/texture.vert.spv";
    fshader_file = "../../shaders/glsl/computeshader/texture.frag.spv";
    std::cout << "[prepareGraphics 385] vshader_file: " << vshader_file << std::endl;
    std::cout << "[prepareGraphics 386] fshader_file: " << fshader_file << std::endl;
#endif
    shaderStages[0] = loadShader(vshader_file, VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = loadShader(fshader_file, VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
        vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
    };
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
        vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
        vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),
    };
    VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
    vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(
        graphics.pipelineLayout, renderPass, 0);
    pipelineCreateInfo.pVertexInputState = &vertexInputState;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));
}

template <typename T>
void VulkanComputeExample::printGpuBuffer(const std::string& varName, const vkgs::Buffer& buffer, int pointNum, int length)
{
    
    const vkgs::CpuBuffer &cpu_buffer = buffer.cpu_buffer;
    if (cpu_buffer.isInit == false) return;
    int size = cpu_buffer.size() / sizeof(T);  // 根据元素大小计算数量
    size = std::min(size, pointNum * length);
    const T* num = reinterpret_cast<const T*>(cpu_buffer.data());
    
    
    std::cout << "Variable: " << varName << std::endl;
#ifdef VULKAN_HW
    LOGW("Variable:  %{public}s", varName.c_str());
#endif
//     if (varName == "final_T" || varName == "ncontrib" || varName == "rfeat" || varName == "dL_drfeat") {
//         std::vector<int> Is = { 101,101,101,785,785,785 };
//         std::vector<int> Js = { 438,439,440,263,264,265 };
//        
//         if (varName == "rfeat" || varName == "dL_drfeat") {
//             for (int k = 0; k < Is.size(); k++) {
//                 int i = Is[k];
//                 int j = Js[k];
//                 #ifdef VULKAN_HW
//                 LOGW("%{public}d, %{public}d: %{public}.12f %{public}.12f %{public}.12f", i, j, num[i * 800 + j], num[i * 800 + j + 800 * 800], num[i * 800 + j + 800 * 800 * 2]);
//                 #endif
//             }
//         }
//         else if (varName == "final_T") {
//             for (int k = 0; k < Is.size(); k++) {
//                 int i = Is[k];
//                 int j = Js[k];
//                 #ifdef VULKAN_HW
//                 LOGW("%{public}d, %{public}d: %{public}f", i, j, num[i * 800 + j]);
//                 #endif
//             }
//         }
//         else{
//             for (int k = 0; k < Is.size(); k++) {
//                 int i = Is[k];
//                 int j = Js[k];
//                 #ifdef VULKAN_HW
//                 LOGW("%{public}d, %{public}d: %{public}d", i, j, num[i * 800 + j]);
//                 #endif
//             }
//         }
//         return;
//     }
    
    if (varName == "tile_range"){
        
        for(int i = 0; i < pointNum;i++){
            LOGW("%{public}d: %{public}d %{public}d", i, num[i * 2 + 0], num[i * 2 + 1]);
        }
        return;
    }
    
    #ifdef VULKAN_HW
    for (int i = 0; i < pointNum; i++) {
        LOGW("%{public}d: ", i);
        for (int j = 0; j < length; j++) {
            if constexpr (std::is_same<T, int32_t>::value)
                LOGW("%{public}d ", num[i * length + j]);
            else if constexpr (std::is_same<T, uint32_t>::value)
                LOGW("%{public}d ", num[i * length + j]);
            else if constexpr (std::is_same<T, uint8_t>::value)
                LOGW("%{public}d ", unsigned(num[i * length + j]));
            
            else {
                if (varName.substr(0, 2) == "dL" || varName == "xyz_gradient_accum" || varName == "xyz_gradient_mean") {
                    LOGW("%{public}.16f ", num[i * length + j]);
                }
                else
                    LOGW("%{public}f ", num[i * length + j]);
            }
        }
    }
    #endif

}

void VulkanComputeExample::prepareCompute()
{
    vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

    VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(
        descriptorPool, &compute.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));

    std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
        vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &textureColorMap.descriptor),
        vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImage.descriptor)
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);

    VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(
        compute.pipelineLayout, 0);

    filterNames = { "emboss", "edgedetect", "sgd" };
    for (auto& shaderName : filterNames) {
        // std::string file = "shader/computeshader/" + shaderName + ".comp.spv";
        // std::string fileName = FileOperator::GetInstance()->GetFileAbsolutePath(file);
        std::string fileName = getPath("shader/computeshader/" + shaderName + ".comp.spv");
#ifdef VULKAN_PC
        fileName = "../../shaders/glsl/computeshader/" + shaderName + ".comp.spv";
        std::cout << "[prepareCompute 454] fileName: " << fileName << std::endl;
#endif
//         std::string fileName = getShadersPath() + "computeshader/" + shaderName + ".comp.spv";
        computePipelineCreateInfo.stage = loadShader(fileName, VK_SHADER_STAGE_COMPUTE_BIT);
        VkPipeline pipeline;
//         VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));
        VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &pipeline));
        compute.pipelines.push_back(pipeline);
    }

    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

    VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(
        compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer));

    VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
    VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));

    VkQueryPoolCreateInfo queryPoolInfo = {};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = MAX_TIME_QUERY_NUM * 2; // Two timestamps: start and end
    VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolInfo, nullptr, &compute.queryPool));
    // buildComputeCommandBuffer();
}

void VulkanComputeExample::prepreAllocator()
{
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.instance = instance;
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
    
    vmaCreateAllocator(&allocatorCreateInfo, &allocator);
}

void VulkanComputeExample::prepareUniformBuffers()
{
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
        &graphics.uniformBuffer, sizeof(Graphics::UniformData)));
    VK_CHECK_RESULT(graphics.uniformBuffer.map());
}

void VulkanComputeExample::updateUniformBuffers()
{
    camera.setPerspective(60.0f, (float)screenWidth * 0.5f / (float)screenHeight, 1.0f, 256.0f);
    graphics.uniformData.projection = camera.matrices.perspective;
    graphics.uniformData.modelView = camera.matrices.view;
    memcpy(graphics.uniformBuffer.mapped, &graphics.uniformData, sizeof(Graphics::UniformData));
}

bool VulkanComputeExample::prepare()
{
    if (prepared) {
        return true;
    }
    VulkanExampleBase::prepare();

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    uint32_t maxAllocations = deviceProperties.limits.maxMemoryAllocationCount;
    std::cout << "maxAllocations = " << maxAllocations << std::endl;

    prepreAllocator();
    loadAssets(low_resolotion_scale);
    generateQuad();
    prepareUniformBuffers();
    prepareStorageImage();
    setupDescriptorPool();
    prepareGraphics();
    prepareCompute();
    buildCommandBuffers();

    // 3dgs specific
    prepare3dgsBuffers();
    setProjectionPipeline();
    setProjectionRenderPipeline();
    setSortPipeline();
    setComputeKeyPipeline();
    setComputeTilePipeline();
    setInverseIndexPipeline();
    setAlphaBlendingPipeline();

    setIntToFloatGradientPipeline();
    setAlphaBlendingIntPipeline();
    
    setOptimizitionPipeline();
    setLossPipeline();
    setImage2BufferPipeline();
    setBuffer2ImagePipeline();
    setBufferFillPipeline();
    setComputeSHPipeline();
    setComputeSHRenderPipeline();
    setClonePipeline();
    setSplitPipeline();
    setDensificationStatsPipeline();
    setMeanGradPipeline();
    setResetOpacityPipeline();
    
    setKNNPipeline();
    setPrunePipeline();
    
//     buildComputeCommandBuffer();

    prepared = true;
    return prepared;
}

void VulkanComputeExample::calculatePSNR(float mse, int width, int height) {
    const int MAX_PIXEL_VALUE = 255; // 像素值的最大值
    const int CHANNELS = 3;          // RGB 三个通道
    const int64_t totalPixels = static_cast<int64_t>(width) * height * CHANNELS;

    // 防止 MSE 为零，避免除以零错误
    if (mse == 0) {
        total_psnr = INFINITY; // PSNR 理论上为无穷大
    }
    else
    {
        // 单通道的 MSE
        double meanSquaredError = static_cast<double>(mse) / totalPixels;

        // PSNR 公式
        double psnr = 10.0 * std::log10((MAX_PIXEL_VALUE * MAX_PIXEL_VALUE) / meanSquaredError);

        total_psnr += psnr;
    }
}

template<typename T>
void copyGpuBufferToVector(const vkgs::Buffer& buffer, std::vector<T>& destVector, int pointNum, int length) {
    const vkgs::CpuBuffer& cpu_buffer = buffer.cpu_buffer;
    if (!cpu_buffer.isInit) return;  // 确保 CPU Buffer 已经同步

    int size = cpu_buffer.size() / sizeof(T);  // 计算 buffer 内的数据数量
    size = std::min(size, pointNum * length);  // 限制拷贝的数据量

    const T* dataPtr = reinterpret_cast<const T*>(cpu_buffer.data());
    destVector.assign(dataPtr, dataPtr + size);  // **复制到目标 vector**
}

void VulkanComputeExample::draw()
{
    
    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.commandBufferCount = 1;
    computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
    computeSubmitInfo.waitSemaphoreCount = 1;
    computeSubmitInfo.pWaitSemaphores = &graphics.semaphore;
    computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
    VkResult res = vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE);
    
    if (res != VK_SUCCESS) {
        #ifdef VULKAN_HW
        LOGE("VulkanExample Fatal : VkResult is %s", vks::tools::errorString(res).c_str());
        #elif defined(VULKAN_PC)
        std::cerr << "VulkanExample Fatal : VkResult is " << vks::tools::errorString(res) << std::endl;
        #endif
    }

    VulkanExampleBase::prepareFrame();

    VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore graphicsWaitSemaphores[] = { compute.semaphore, semaphores.presentComplete };
    VkSemaphore graphicsSignalSemaphores[] = { graphics.semaphore, semaphores.renderComplete };

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
    submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
    res = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        #ifdef VULKAN_HW
        LOGE("VulkanExample Fatal : VkResult is %s", vks::tools::errorString(res).c_str());
        #elif defined(VULKAN_PC)
        std::cerr << "VulkanExample Fatal : VkResult is " << vks::tools::errorString(res) << std::endl;
        #endif
    }
    VulkanExampleBase::submitFrame();
    
    if (training_step >= 0)
    {
        const vkgs::CpuBuffer& cpu_buffer = splat_buffers.curPointsNum.cpu_buffer;
        if (cpu_buffer.isInit && training_step > 0)
        {
            const int* num = reinterpret_cast<const int*>(cpu_buffer.data());
            splat_count = num[0];
            //std::cout << splat_count << std::endl;
        }

        const vkgs::CpuBuffer& cpu_buffer1 = splat_buffers.psnr_sum.cpu_buffer;
        float psnr_sum = -1;
        if (cpu_buffer1.isInit)
        {
            const float* num = reinterpret_cast<const float*>(cpu_buffer1.data());
            psnr_sum = num[0];
            //std::cout << splat_count << std::endl;
        }
        
        const vkgs::CpuBuffer& cpu_buffer2 = sort_buffers.splat_visible_point_count.cpu_buffer;
        int sort_point_count = -1;
        if (cpu_buffer2.isInit)
        {
            const int* num = reinterpret_cast<const int*>(cpu_buffer2.data());
            sort_point_count = num[0];
            //std::cout << splat_count << std::endl;
        }
        total_sort_point_count += sort_point_count;
        max_avg_splat_tile_num = std::max(max_avg_splat_tile_num, 1.0 * sort_point_count / splat_count);
        calculatePSNR(psnr_sum, (int)(current_scale * image_width), (int)(current_scale * image_height));
        if ((training_step + 1) % 100 == 0 && training_step != max_training_step)
        {
            std::cout << "Traning Step[" << training_step << "] Point Num: " << splat_count << " Average psnr: " << total_psnr / 100 << "; Mean Sort Point: " << total_sort_point_count / 100 << "; Max Avg Splat Tile: " << max_avg_splat_tile_num << std::endl;
            LOGI("Training Step[%{public}d]: Point Num: [%{public}d], Average PSNR： [%{public}d], Mean Sort Point: [%{public}f], Max Avg Splat Tile: [%{public}d]", training_step, total_psnr / 100, total_sort_point_count / 100.0, max_avg_splat_tile_num);
            total_psnr = 0;
            total_sort_point_count = 0;
            max_avg_splat_tile_num = 0;
        }
    } 
    
    
    if (training_step == save_step)
    {

        std::vector<float> positionData;
        copyGpuBufferToVector(splat_buffers.xyz_array, positionData, splat_count, 3);
        inputGaussian.positions.clear();
        for (int i = 0; i < splat_count; i++) {
            inputGaussian.positions.emplace_back(
                positionData[i * 3], 
                positionData[i * 3 + 1], 
                positionData[i * 3 + 2]
            );
        }
        
        std::vector<float> scaleData;
        copyGpuBufferToVector(splat_buffers.scale_array, scaleData, splat_count, 3);
        inputGaussian.scales.clear();
        for (int i = 0; i < splat_count; i++) {
            inputGaussian.scales.emplace_back(
                scaleData[i * 3], 
                scaleData[i * 3 + 1], 
                scaleData[i * 3 + 2]
            );
        }
        
        std::vector<float> rotationData;
        copyGpuBufferToVector(splat_buffers.rotate_array, rotationData, splat_count, 4);
        inputGaussian.rotations.clear();
        for (int i = 0; i < splat_count; i++) {
            inputGaussian.rotations.emplace_back(
                rotationData[i * 4], 
                rotationData[i * 4 + 1], 
                rotationData[i * 4 + 2], 
                rotationData[i * 4 + 3]
            );
        }
        
        std::vector<float> featureData;
        copyGpuBufferToVector(splat_buffers.sh_coeff, featureData, splat_count, 48);
        inputGaussian.colors.clear();
        for (int i = 0; i < splat_count; i++) {
            DataLoader::InputInitStruct::Data colorData;
            for (int j = 0; j < 48; j++) {
                colorData.values[j] = featureData[i * 48 + j];
            }
            inputGaussian.colors.push_back(colorData);
        }
        
        copyGpuBufferToVector(splat_buffers.opacity, inputGaussian.opacities, splat_count, 1);

        std::cout << "Successfully copied GPU buffers to inputGaussian!" << std::endl;
        DataLoader::saveToPLY_tinyply(this->basepath + "/" + "assets/AREngine/2024_09_27_11_20_29/outputGS.ply", inputGaussian);
    }
    
    queryEventTime();
    if(training_step >= 0){
//         printGpuBuffers();
    }
    
}

void VulkanComputeExample::queryCmdTimeStart(std::string eventName)
{
    if (queryTime.find(eventName) != queryTime.end())
    {
        throw std::runtime_error("Time Query Event existed!");
    }
    unsigned int queryIndex = queryTime.size();
    queryTime[eventName] = { queryIndex * 2, queryIndex * 2 + 1 };
    vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, compute.queryPool, queryTime[eventName].startIndex);
}

void VulkanComputeExample::queryCmdTimeEnd(std::string eventName)
{
    if (queryTime.find(eventName) == queryTime.end()) {
        throw std::runtime_error("Can Not Find Time Query Event!");
    }
    vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, compute.queryPool, queryTime[eventName].endIndex);
}

void VulkanComputeExample::queryEventTime()
{
    uint64_t timestamps[MAX_TIME_QUERY_NUM * 2];
    vkGetQueryPoolResults(device, compute.queryPool, 0, 2 * queryTime.size(), sizeof(timestamps), timestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);

    double timestampPeriod = deviceProperties.limits.timestampPeriod; 
    
    for (const auto & query: queryTime)
    {
        elapsedTimeSum[query.first] += (timestamps[query.second.endIndex] - timestamps[query.second.startIndex]) * timestampPeriod / 1e6;
    }
    queryTime.clear();
    if ((training_step + 1) % QUERY_TIME_RANGE == 0)
    {
        for (const auto& query : elapsedTimeSum)
        {
            LOGI("[%{public}s] execution time : %{public}f ms.", query.first.c_str(), query.second / QUERY_TIME_RANGE);
//             std::cout << "[" << query.first << "] execution time : " << query.second / QUERY_TIME_RANGE << " ms" << std::endl;
        }
        elapsedTimeSum.clear();
    }
    
    
}

#ifdef VULKAN_PC
VULKAN_EXAMPLE_MAIN()
#endif


