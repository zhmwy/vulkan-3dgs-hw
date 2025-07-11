#include "patch_compute.h"
#include "base/math/matrix_util.h"
#include "opencv2/imgproc.hpp"

#include <opencv2/opencv.hpp>
#include <numeric>
#include <unistd.h>

VulkanComputePatch::VulkanComputePatch()
{
#ifdef VULKAN_HW
    enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    enabledDeviceExtensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    enabledDeviceExtensions.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
#endif
    
}

VulkanComputePatch::~VulkanComputePatch()
{
    if (device) {
        DestoryPatchBuffers();
    }
}

void VulkanComputePatch::loadPatchData(std::string& dataFolder)
{
    if (dataFolder.empty()){
        throw std::runtime_error("The input data folder is empty.");
    }
    
    if (dataFolder.back() == '/') {
        dataFolder.pop_back();
    }
    
    inputGaussian.basepath = dataFolder.substr(0, dataFolder.find_last_of("/"));
    FileOperator::GetInstance()->set_defaultDir(inputGaussian.basepath);
    
    DataLoader::loadPatchInput(dataFolder, inputGaussian, params.height, params.width);
}

bool VulkanComputePatch::prepare()
{
    if (prepared) {
        return true;
    }
    
	VulkanComputeBase::prepare();
    
    loadAssets(image_width,image_height);
    std::vector<int> pairs = selectViewPair();
    std::vector<float> zRange = calcDepthRange();
    
    preparePatchBuffers();
    
    setupRandomInitializationPipeline();
    setupPropagationPipeline();
    setupDepthAndNormalPipeline();
    
    prepared = true;
    
    return true;
}

void VulkanComputePatch::debugOutputs(std::string dataPath)
{
    buildComputeCommandBuffer([this](){
        ComputeTransBarrierBlock();
        createGpu2CpuCopy(patchBuffers.planeHypotheses);
        createGpu2CpuCopy(patchBuffers.costs);
    });
    executeComputeCommand();
    
    VK_CHECK_RESULT(vkQueueWaitIdleGS(computeRes.queue));
    {
        cv::Mat floatMat(image_height, image_width, CV_32FC4, 
                    (float *)patchBuffers.planeHypotheses.cpu_buffer.data());
        
        std::stringstream ss;
        ss  << "Matrix values:" << floatMat << std::endl;
        
        cv::Mat ucharMat;
        floatMat.convertTo(ucharMat, CV_8UC4, 255.0);
        
        cv::cvtColor(ucharMat, ucharMat, cv::COLOR_RGBA2BGR);
        cv::imwrite(dataPath + "/test1.png", ucharMat);
    }
    {
        cv::Mat src_32fc4(image_height, image_width, CV_32FC4, 
                    (float *)patchBuffers.planeHypotheses.cpu_buffer.data());
        
        cv::Mat channel_32fc1;
        cv::extractChannel(src_32fc4, channel_32fc1, 3);
        
        cv::Mat ucharMat;
        channel_32fc1.convertTo(ucharMat, CV_8UC1, 255.0/5.0);
        cv::imwrite(dataPath + "/test2.png", ucharMat);
    }
}

void VulkanComputePatch::compute()
{
    buildComputeCommandBuffer([this](){
        std::vector<uint32_t> pushConstants;
        pushConstants.push_back(image_width);
        pushConstants.push_back(image_height);
        pushConstants.push_back(params.srcCount);
        pushConstants.push_back(0);
        
        pushConstants.push_back(params.patch_size);
        pushConstants.push_back(params.radius_increment);
        pushConstants.push_back(params.sigma_spatial);
        pushConstants.push_back(params.sigma_color);
        pushConstants.push_back(params.top_k);
        pushConstants.push_back(0);
        
        int X = ((int)(image_width) + 16 - 1) / 16; // Ceiling division
        int Y = ((int)(image_height) + 16 - 1) / 16; // Ceiling division
        bindPipeline2CommandBufferPushConstants(pipeRandomInitialization, 0, X, Y, 1, pushConstants);
        
        
        for (int i=0; i<10; ++i) {
            pushConstants.back() = 0;
            X = ((int)(image_width) + 32 - 1) / 32; // Ceiling division
            Y = ((int)(image_height) + 16 - 1) / 16; // Ceiling division
            bindPipeline2CommandBufferPushConstants(pipePropagation, 0, X, Y, 1, pushConstants);
        
            pushConstants.back() = 1;
            X = ((int)(image_width) + 32 - 1) / 32; // Ceiling division
            Y = ((int)(image_height) + 16 - 1) / 16; // Ceiling division
            bindPipeline2CommandBufferPushConstants(pipePropagation, 0, X, Y, 1, pushConstants);
        }
        
        X = ((int)(image_width) + 16 - 1) / 16; // Ceiling division
        Y = ((int)(image_height) + 16 - 1) / 16; // Ceiling division
        bindPipeline2CommandBufferPushConstants(pipeDepthAndNormal, 0, X, Y, 1, pushConstants);
        
    });
    executeComputeCommand();
}

void VulkanComputePatch::initBuffersValues()
{
    if (!prepared) {
        return;
    }
    
    buildComputeCommandBuffer([this]() {
        {
            std::vector<float> depthRanges = calcDepthRange();
            std::vector<STCamera> vCameras(inputGaussian.extr_tmp.size());
            for (int i=0; i<inputGaussian.extr_tmp.size(); i++) {
                Base::Math::Mat3X3 rmat = inputGaussian.quat2matrix3(inputGaussian.extr_tmp[i]);
                memcpy(vCameras[i].R, rmat.data, sizeof(float)*9);
                memcpy(vCameras[i].K, inputGaussian.intr_tmp[0].data(), sizeof(float)*4);
                memcpy(vCameras[i].t, inputGaussian.extr_tmp[i].values+4, sizeof(float)*3);
                vCameras[i].minDepth = depthRanges[2*i+0];
                vCameras[i].maxDepth = depthRanges[2*i+1];
            }
            
            patchBuffers.cameras.FromCpu(computeRes.commandBuffer, vCameras.data());
        }
        patchBuffers.srcPairs.FromCpu(computeRes.commandBuffer, selectViewPair());
        
        vkCmdFillBufferGS(computeRes.commandBuffer, patchBuffers.planeHypotheses, 0, VK_WHOLE_SIZE, 0);
    });
    executeComputeCommand();
}

void VulkanComputePatch::loadAssets(int origin_width, int origin_height) 
{
    LOGI("Load Assets: loading images.");
    for (const std::string& filename : inputGaussian.img_list) {
        LOGI("Load Assets: loading image %{public}s", filename.c_str());
        cv::Mat img = cv::imread(filename);
        if (img.empty()){
            std::cerr << "Failed to load image file: " << filename << std::endl;
            return;
        }
        cv::resize(img, img, cv::Size(origin_width, origin_height));
        
        cv::Mat bgrFloat;
        img.convertTo(bgrFloat, CV_32FC3, 1.0/255.0);
        cv::cvtColor(bgrFloat, bgrFloat, cv::COLOR_BGR2GRAY);
        
        vkbase::Texture2DGS texture_original;
        texture_original.fromImageBuffer(
            bgrFloat.data,
            origin_width * origin_height * 4,
            VK_FORMAT_R32_SFLOAT,
            origin_width, origin_height, 
            vulkanDevice, computeRes.queue,
            VK_FILTER_LINEAR, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        
        inputGaussian.images.push_back(texture_original);
    }
}

void VulkanComputePatch::DestoryPatchBuffers()
{
    vkDestroyBufferGS(device, patchBuffers.cameras, nullptr);
    vkDestroyBufferGS(device, patchBuffers.planeHypotheses, nullptr);
    vkDestroyBufferGS(device, patchBuffers.costs, nullptr);
    vkDestroyBufferGS(device, patchBuffers.selectedViews, nullptr);
    vkDestroyBufferGS(device, patchBuffers.srcPairs, nullptr);
}

void VulkanComputePatch::preparePatchBuffers()
{
    patchBuffers.cameras = vkgs::Buffer(
        allocator, inputGaussian.images.size() * sizeof(STCamera),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.planeHypotheses = vkgs::Buffer(
        allocator, image_height * image_width * 4 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.costs = vkgs::Buffer(
        allocator, image_height * image_width * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.selectedViews = vkgs::Buffer(
        allocator, image_height * image_width * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
    
    patchBuffers.srcPairs = vkgs::Buffer(
        allocator, inputGaussian.images.size() * params.srcCount * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    );
}

void VulkanComputePatch::setupDepthAndNormalPipeline()
{
    const uint32_t bindingCount = 5;
    createPipelineLayout(
        pipeDepthAndNormal,
        inputGaussian.images.size(), 
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 10
    );

    std::vector<vkgs::Buffer*> pBuffers = {
        &patchBuffers.srcPairs,
        &patchBuffers.cameras,
        &patchBuffers.planeHypotheses,
        &patchBuffers.costs,
        &patchBuffers.selectedViews
    };

    updateDescriptorSet(pipeDepthAndNormal.descriptorSet, inputGaussian.images, pBuffers);

    pipeDepthAndNormal.pipelines.resize(1);

    std::string shaderCode = getPath("shader/patch/depthNormal.comp.spv");

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);
    pipelineInfo.layout = pipePropagation.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelinesGS(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeDepthAndNormal.pipelines[0]
        )
    );

    LOGI("[setupDepthAndNormalPipeline] DepthAndNormal pipeline set");
}

void VulkanComputePatch::setupPropagationPipeline()
{
    const uint32_t bindingCount = 5;
    createPipelineLayout(
        pipePropagation,
        inputGaussian.images.size(), 
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 10
    );

    std::vector<vkgs::Buffer*> pBuffers = {
        &patchBuffers.srcPairs,
        &patchBuffers.cameras,
        &patchBuffers.planeHypotheses,
        &patchBuffers.costs,
        &patchBuffers.selectedViews
    };

    updateDescriptorSet(pipePropagation.descriptorSet, inputGaussian.images, pBuffers);

    pipePropagation.pipelines.resize(1);

    std::string shaderCode = getPath("shader/patch/propagation.comp.spv");

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);
    pipelineInfo.layout = pipePropagation.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelinesGS(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipePropagation.pipelines[0]
        )
    );

    LOGI("[setupPropagationPipeline] Propagation pipeline set");
}

void VulkanComputePatch::setupRandomInitializationPipeline()
{
    const uint32_t bindingCount = 5;
    createPipelineLayout(
        pipeRandomInitialization,
        inputGaussian.images.size(), 
        bindingCount,  // Number of descriptor set bindings
        true,
        sizeof(uint32_t) * 10
    );

    std::vector<vkgs::Buffer*> pBuffers = {
        &patchBuffers.srcPairs,
        &patchBuffers.cameras,
        &patchBuffers.planeHypotheses,
        &patchBuffers.costs,
        &patchBuffers.selectedViews
    };

    //updateDescriptorSet(pipeRandomInitialization.descriptorSet, inputGaussian.images, pBuffers);
    updateDescriptorSet(pipeRandomInitialization.descriptorSet, inputGaussian.images, pBuffers);

    pipeRandomInitialization.pipelines.resize(1);

    std::string shaderCode = getPath("shader/patch/initialization.comp.spv");

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = loadShader(shaderCode, VK_SHADER_STAGE_COMPUTE_BIT);
    pipelineInfo.layout = pipeRandomInitialization.pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelinesGS(
            device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeRandomInitialization.pipelines[0]
        )
    );

    LOGI("[setupRandomInitializationPipeline] Random Initialization pipeline set");
}

std::vector<float> VulkanComputePatch::calcDepthRange()
{
    std::vector<float> result(inputGaussian.images.size()*2, -1);
    
    for (int image=0; image<inputGaussian.images.size(); ++image) {
        Base::Math::Mat4X4 W2C = DataLoader::InputInitStruct::quat2matrix4(inputGaussian.extr_tmp[image]);
        Base::Math::Mat4X4 C2W = Base::Math::Inverse(W2C);
        
        float zMax = -FLT_MAX;
        float zMin = FLT_MAX;
        
        std::vector<int> pIds = inputGaussian.pointsOnImage[image];
        std::vector<float> zs;
        for (int pid: pIds) {
            if (pid < 0) {
                continue;
            }
            Base::Math::Vec3 vPoint = inputGaussian.positions[inputGaussian.point3DMap[pid]];
            
            Base::Math::Vec3 pc = Base::Math::MultiplyPoint3X4(Base::Math::Transpose(W2C), vPoint);
            zs.push_back(pc.z);
        }
        
        std::sort(zs.begin(), zs.end());
        result[2*image+0] = zs[int(zs.size() * 0.01)] * 0.75;
        result[2*image+1] = zs[int(zs.size() * 0.99)] * 1.25;
    }
    
    return result;
}

int VulkanComputePatch::calcScore(int image1, int image2)
{
    if (image1 == image2) {
        LOGE("Calc Same Image Score");
        return 0;
    }
    Base::Math::Mat4X4 W2C1 = DataLoader::InputInitStruct::quat2matrix4(inputGaussian.extr_tmp[image1]);
    Base::Math::Mat4X4 C2W1 = Base::Math::Inverse(W2C1);
    Base::Math::Vec3 cam_center1(C2W1[0][3], C2W1[1][3], C2W1[2][3]);
    
    Base::Math::Mat4X4 W2C2 = DataLoader::InputInitStruct::quat2matrix4(inputGaussian.extr_tmp[image2]);
    Base::Math::Mat4X4 C2W2 = Base::Math::Inverse(W2C2);
    Base::Math::Vec3 cam_center2(C2W2[0][3], C2W2[1][3], C2W2[2][3]);
    
    std::vector<int> sorted1 = inputGaussian.pointsOnImage[image1];
    std::vector<int> sorted2 = inputGaussian.pointsOnImage[image2];
    
    std::sort(sorted1.begin(), sorted1.end());
    std::sort(sorted2.begin(), sorted2.end());
    
    std::vector<int> interPoints;
    std::set_intersection(
        sorted1.begin(), sorted1.end(),
        sorted2.begin(), sorted2.end(),
        std::back_inserter(interPoints)
    );
    
    std::vector<float> angles;
    int score = 0;
    for (size_t i=0; i<interPoints.size(); ++i) {
        int pid = interPoints[i];
        if (pid == -1) {
            continue;
        }
        
        Base::Math::Vec3 vPoint = inputGaussian.positions[pid];
        
        float theta = (180.0 / M_PI) * 
                        std::acos(Base::Math::Dot(cam_center1-vPoint, cam_center2-vPoint) / 
                                  Base::Math::Magnitude(cam_center1-vPoint) / 
                                  Base::Math::Magnitude(cam_center2-vPoint));
        
        angles.push_back(theta);
        score += 1;        
    }
    
    if (!angles.empty()) {
        const double percentile = 0.75;
        const size_t n = angles.size();
        size_t pos = static_cast<size_t>(percentile * n);
        
        std::nth_element(angles.begin(), angles.begin() + pos, angles.end());
        if (angles[pos] < 1) {
            score = 0;
        }
    }
    
    return score;
}

std::vector<int> VulkanComputePatch::selectViewPair()
{
    std::vector<std::vector<int>> vScore(inputGaussian.images.size());    
    for(size_t i=0; i<vScore.size(); ++i)
    {
        vScore[i].resize(vScore.size(), 0);
    }
    
    for(size_t i=0; i<vScore.size(); ++i)
    {
        for (size_t j=i+1; j<vScore.size(); ++j)
        {
            int score = calcScore(i, j);
            vScore[i][j] = score;
            vScore[j][i] = score;
        }
    }
    
    std::vector<int> vPairs(vScore.size()*params.srcCount, -1);
    for(size_t i=0; i<vScore.size(); ++i)
    {
        std::vector<size_t> indices(vScore.size());
        std::iota(indices.begin(), indices.end(), 0); 
        std::sort(indices.begin(), indices.end(),
                  [&i, &vScore](size_t i1, size_t i2) { return vScore[i][i1] > vScore[i][i2]; });
        
        size_t minK = std::min((size_t)params.srcCount, vScore.size());
        for (int k=0; k<minK; ++k) {
            if (indices[i] < 0) {
                break;
            }
            vPairs[i*params.srcCount + k] = indices[k];
        }
    }
    
    return vPairs;
}
