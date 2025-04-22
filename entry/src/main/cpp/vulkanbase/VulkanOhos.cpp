//
// Created on 2024/2/4.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".

#include "VulkanOhos.h"
#include <iostream>
#include <dlfcn.h>
#include "common/common.h"


PFN_vkCreateInstance vkCreateInstanceGS;
PFN_vkGetDeviceProcAddr vkGetDeviceProcAddrGS;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddrGS;
PFN_vkCreateDevice vkCreateDeviceGS;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevicesGS;
PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDevicePropertiesGS;
PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2GS;
PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionPropertiesGS;
PFN_vkEnumerateDeviceLayerProperties vkEnumerateDeviceLayerPropertiesGS;
PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatPropertiesGS;
PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeaturesGS;
PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2GS;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyPropertiesGS;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryPropertiesGS;
PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionPropertiesGS;
PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerPropertiesGS;
PFN_vkCmdPipelineBarrier vkCmdPipelineBarrierGS;
PFN_vkCreateShaderModule vkCreateShaderModuleGS;
PFN_vkCreateBuffer vkCreateBufferGS;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirementsGS;
PFN_vkMapMemory vkMapMemoryGS;
PFN_vkUnmapMemory vkUnmapMemoryGS;
PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRangesGS;
PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRangesGS;
PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBufferGS;
PFN_vkBindBufferMemory vkBindBufferMemoryGS;
PFN_vkDestroyBuffer vkDestroyBufferGS;
PFN_vkAllocateMemory vkAllocateMemoryGS;
PFN_vkBindImageMemory vkBindImageMemoryGS;
PFN_vkGetImageSubresourceLayout vkGetImageSubresourceLayoutGS;
PFN_vkCmdCopyBuffer vkCmdCopyBufferGS;
PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImageGS;
PFN_vkCmdCopyImage vkCmdCopyImageGS;
PFN_vkCmdBlitImage vkCmdBlitImageGS;
PFN_vkCmdClearAttachments vkCmdClearAttachmentsGS;
PFN_vkCreateSampler vkCreateSamplerGS;
PFN_vkDestroySampler vkDestroySamplerGS;
PFN_vkDestroyImage vkDestroyImageGS;
PFN_vkFreeMemory vkFreeMemoryGS;
PFN_vkCreateRenderPass vkCreateRenderPassGS;
PFN_vkCmdBeginRenderPass vkCmdBeginRenderPassGS;
PFN_vkCmdEndRenderPass vkCmdEndRenderPassGS;
PFN_vkCmdNextSubpass vkCmdNextSubpassGS;
PFN_vkCmdExecuteCommands vkCmdExecuteCommandsGS;
PFN_vkCmdClearColorImage vkCmdClearColorImageGS;
PFN_vkCreateImage vkCreateImageGS;
PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirementsGS;
PFN_vkCreateImageView vkCreateImageViewGS;
PFN_vkDestroyImageView vkDestroyImageViewGS;
PFN_vkCreateSemaphore vkCreateSemaphoreGS;
PFN_vkDestroySemaphore vkDestroySemaphoreGS;
PFN_vkCreateFence vkCreateFenceGS;
PFN_vkDestroyFence vkDestroyFenceGS;
PFN_vkWaitForFences vkWaitForFencesGS;
PFN_vkResetFences vkResetFencesGS;
PFN_vkResetDescriptorPool vkResetDescriptorPoolGS;
PFN_vkCreateCommandPool vkCreateCommandPoolGS;
PFN_vkDestroyCommandPool vkDestroyCommandPoolGS;
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffersGS;
PFN_vkBeginCommandBuffer vkBeginCommandBufferGS;
PFN_vkEndCommandBuffer vkEndCommandBufferGS;
PFN_vkGetDeviceQueue vkGetDeviceQueueGS;
PFN_vkQueueSubmit vkQueueSubmitGS;
PFN_vkQueueWaitIdle vkQueueWaitIdleGS;
PFN_vkDeviceWaitIdle vkDeviceWaitIdleGS;
PFN_vkCreateFramebuffer vkCreateFramebufferGS;
PFN_vkCreatePipelineCache vkCreatePipelineCacheGS;
PFN_vkCreatePipelineLayout vkCreatePipelineLayoutGS;
PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelinesGS;
PFN_vkCreateComputePipelines vkCreateComputePipelinesGS;
PFN_vkCreateDescriptorPool vkCreateDescriptorPoolGS;
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayoutGS;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSetsGS;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSetsGS;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSetsGS;
PFN_vkCmdBindPipeline vkCmdBindPipelineGS;
PFN_vkCmdBindVertexBuffers vkCmdBindVertexBuffersGS;
PFN_vkCmdBindIndexBuffer vkCmdBindIndexBufferGS;
PFN_vkCmdSetViewport vkCmdSetViewportGS;
PFN_vkCmdSetScissor vkCmdSetScissorGS;
PFN_vkCmdSetLineWidth vkCmdSetLineWidthGS;
PFN_vkCmdSetDepthBias vkCmdSetDepthBiasGS;
PFN_vkCmdPushConstants vkCmdPushConstantsGS;
PFN_vkCmdDrawIndexed vkCmdDrawIndexedGS;
PFN_vkCmdDraw vkCmdDrawGS;
PFN_vkCmdDrawIndexedIndirect vkCmdDrawIndexedIndirectGS;
PFN_vkCmdDrawIndirect vkCmdDrawIndirectGS;
PFN_vkCmdDispatch vkCmdDispatchGS;
PFN_vkDestroyPipeline vkDestroyPipelineGS;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayoutGS;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayoutGS;
PFN_vkDestroyDevice vkDestroyDeviceGS;
PFN_vkDestroyInstance vkDestroyInstanceGS;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPoolGS;
PFN_vkFreeCommandBuffers vkFreeCommandBuffersGS;
PFN_vkDestroyRenderPass vkDestroyRenderPassGS;
PFN_vkDestroyFramebuffer vkDestroyFramebufferGS;
PFN_vkDestroyShaderModule vkDestroyShaderModuleGS;
PFN_vkDestroyPipelineCache vkDestroyPipelineCacheGS;
PFN_vkCreateQueryPool vkCreateQueryPoolGS;
PFN_vkDestroyQueryPool vkDestroyQueryPoolGS;
PFN_vkGetQueryPoolResults vkGetQueryPoolResultsGS;
PFN_vkCmdBeginQuery vkCmdBeginQueryGS;
PFN_vkCmdEndQuery vkCmdEndQueryGS;
PFN_vkCmdResetQueryPool vkCmdResetQueryPoolGS;
PFN_vkCmdCopyQueryPoolResults vkCmdCopyQueryPoolResultsGS;
PFN_vkCmdWriteTimestamp vkCmdWriteTimestampGS;
PFN_vkCmdUpdateBuffer vkCmdUpdateBufferGS;
PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddressGS;

PFN_vkCreateSurfaceOHOS vkCreateSurfaceOHOSGS;
PFN_vkDestroySurfaceKHR vkDestroySurfaceKHRGS;
PFN_vkCmdFillBuffer vkCmdFillBufferGS;

PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR vkGetPhysicalDeviceFragmentShadingRatesKHRGS;
PFN_vkCmdSetFragmentShadingRateKHR vkCmdSetFragmentShadingRateKHRGS;
PFN_vkCreateRenderPass2KHR vkCreateRenderPass2KHRGS;

void *libVulkan;

namespace vks {
    namespace ohos {
        bool loadVulkanLibrary() {
            LOGI("vk::ohos::loadVulkanLibrary:Loading libvulkan.so");

            const char *path_ = "libvulkan.so";
            libVulkan = dlopen(path_, RTLD_NOW | RTLD_LOCAL);
            if (!libVulkan) {
                LOGI("vk::ohos::loadVulkanLibrary: Could not load vulkan library %{public}s", dlerror());
                return false;
            }
            // // Load base function pointers
            vkEnumerateInstanceExtensionPropertiesGS = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
                dlsym(libVulkan, "vkEnumerateInstanceExtensionProperties"));

            vkEnumerateInstanceLayerPropertiesGS = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
                dlsym(libVulkan, "vkEnumerateInstanceLayerProperties"));
    
            vkCreateInstanceGS = reinterpret_cast<PFN_vkCreateInstance>(dlsym(libVulkan, "vkCreateInstance"));
            vkGetInstanceProcAddrGS =
                reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(libVulkan, "vkGetInstanceProcAddr"));
            vkGetDeviceProcAddrGS = reinterpret_cast<PFN_vkGetDeviceProcAddr>(dlsym(libVulkan, "vkGetDeviceProcAddr"));
            return true;
        }

        void loadVulkanFunctions(VkInstance instance) {
            vkEnumeratePhysicalDevicesGS = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
                vkGetInstanceProcAddrGS(instance, "vkEnumeratePhysicalDevices"));
            vkGetPhysicalDevicePropertiesGS = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceProperties"));
            vkGetPhysicalDeviceProperties2GS = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceProperties2"));
            vkEnumerateDeviceLayerPropertiesGS = reinterpret_cast<PFN_vkEnumerateDeviceLayerProperties>(
                vkGetInstanceProcAddrGS(instance, "vkEnumerateDeviceLayerProperties"));
            vkEnumerateDeviceExtensionPropertiesGS = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
                vkGetInstanceProcAddrGS(instance, "vkEnumerateDeviceExtensionProperties"));
            vkGetPhysicalDeviceQueueFamilyPropertiesGS = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
            vkGetPhysicalDeviceFeaturesGS = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
                vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceFeatures"));
            vkGetPhysicalDeviceFeatures2GS = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceFeatures2"));
            vkCreateDeviceGS = reinterpret_cast<PFN_vkCreateDevice>(vkGetInstanceProcAddrGS(instance, "vkCreateDevice"));
            vkGetPhysicalDeviceFormatPropertiesGS = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
                vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceFormatProperties"));
            vkGetPhysicalDeviceMemoryPropertiesGS = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
                vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceMemoryProperties"));
            vkCmdPipelineBarrierGS =
                reinterpret_cast<PFN_vkCmdPipelineBarrier>(vkGetInstanceProcAddrGS(instance, "vkCmdPipelineBarrier"));
            vkCreateShaderModuleGS =
                reinterpret_cast<PFN_vkCreateShaderModule>(vkGetInstanceProcAddrGS(instance, "vkCreateShaderModule"));
            vkCreateBufferGS = reinterpret_cast<PFN_vkCreateBuffer>(vkGetInstanceProcAddrGS(instance, "vkCreateBuffer"));
            vkGetBufferMemoryRequirementsGS = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(
                vkGetInstanceProcAddrGS(instance, "vkGetBufferMemoryRequirements"));
            vkMapMemoryGS = reinterpret_cast<PFN_vkMapMemory>(vkGetInstanceProcAddrGS(instance, "vkMapMemory"));
            vkUnmapMemoryGS = reinterpret_cast<PFN_vkUnmapMemory>(vkGetInstanceProcAddrGS(instance, "vkUnmapMemory"));
            vkFlushMappedMemoryRangesGS = reinterpret_cast<PFN_vkFlushMappedMemoryRanges>(
                vkGetInstanceProcAddrGS(instance, "vkFlushMappedMemoryRanges"));
            vkInvalidateMappedMemoryRangesGS = reinterpret_cast<PFN_vkInvalidateMappedMemoryRanges>(
                vkGetInstanceProcAddrGS(instance, "vkInvalidateMappedMemoryRanges"));
            vkCmdCopyBufferGS = reinterpret_cast<PFN_vkCmdCopyBuffer>(vkGetInstanceProcAddrGS(instance, "vkCmdCopyBuffer"));
            vkCmdCopyImageToBufferGS = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(vkGetInstanceProcAddrGS(instance, "vkCmdCopyImageToBuffer"));
            vkBindBufferMemoryGS =
                reinterpret_cast<PFN_vkBindBufferMemory>(vkGetInstanceProcAddrGS(instance, "vkBindBufferMemory"));
            vkDestroyBufferGS = reinterpret_cast<PFN_vkDestroyBuffer>(vkGetInstanceProcAddrGS(instance, "vkDestroyBuffer"));

            vkAllocateMemoryGS =
                reinterpret_cast<PFN_vkAllocateMemory>(vkGetInstanceProcAddrGS(instance, "vkAllocateMemory"));
            vkFreeMemoryGS = reinterpret_cast<PFN_vkFreeMemory>(vkGetInstanceProcAddrGS(instance, "vkFreeMemory"));
            vkCreateRenderPassGS =
                reinterpret_cast<PFN_vkCreateRenderPass>(vkGetInstanceProcAddrGS(instance, "vkCreateRenderPass"));
            vkCmdBeginRenderPassGS =
                reinterpret_cast<PFN_vkCmdBeginRenderPass>(vkGetInstanceProcAddrGS(instance, "vkCmdBeginRenderPass"));
            vkCmdEndRenderPassGS =
                reinterpret_cast<PFN_vkCmdEndRenderPass>(vkGetInstanceProcAddrGS(instance, "vkCmdEndRenderPass"));
            vkCmdNextSubpassGS =
                reinterpret_cast<PFN_vkCmdNextSubpass>(vkGetInstanceProcAddrGS(instance, "vkCmdNextSubpass"));
            vkCmdExecuteCommandsGS =
                reinterpret_cast<PFN_vkCmdExecuteCommands>(vkGetInstanceProcAddrGS(instance, "vkCmdExecuteCommands"));
            vkCmdClearColorImageGS =
                reinterpret_cast<PFN_vkCmdClearColorImage>(vkGetInstanceProcAddrGS(instance, "vkCmdClearColorImage"));

            vkCreateImageGS = reinterpret_cast<PFN_vkCreateImage>(vkGetInstanceProcAddrGS(instance, "vkCreateImage"));
            vkGetImageMemoryRequirementsGS = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(
                vkGetInstanceProcAddrGS(instance, "vkGetImageMemoryRequirements"));
            vkCreateImageViewGS =
                reinterpret_cast<PFN_vkCreateImageView>(vkGetInstanceProcAddrGS(instance, "vkCreateImageView"));
            vkDestroyImageViewGS =
                reinterpret_cast<PFN_vkDestroyImageView>(vkGetInstanceProcAddrGS(instance, "vkDestroyImageView"));
            vkBindImageMemoryGS =
                reinterpret_cast<PFN_vkBindImageMemory>(vkGetInstanceProcAddrGS(instance, "vkBindImageMemory"));
            vkGetImageSubresourceLayoutGS = reinterpret_cast<PFN_vkGetImageSubresourceLayout>(
                vkGetInstanceProcAddrGS(instance, "vkGetImageSubresourceLayout"));
            vkCmdCopyImageGS = reinterpret_cast<PFN_vkCmdCopyImage>(vkGetInstanceProcAddrGS(instance, "vkCmdCopyImage"));
            vkCmdBlitImageGS = reinterpret_cast<PFN_vkCmdBlitImage>(vkGetInstanceProcAddrGS(instance, "vkCmdBlitImage"));
            vkDestroyImageGS = reinterpret_cast<PFN_vkDestroyImage>(vkGetInstanceProcAddrGS(instance, "vkDestroyImage"));

            vkCmdClearAttachmentsGS =
                reinterpret_cast<PFN_vkCmdClearAttachments>(vkGetInstanceProcAddrGS(instance, "vkCmdClearAttachments"));

            vkCmdCopyBufferGS = reinterpret_cast<PFN_vkCmdCopyBuffer>(vkGetInstanceProcAddrGS(instance, "vkCmdCopyBuffer"));
            vkCmdCopyBufferToImageGS =
                reinterpret_cast<PFN_vkCmdCopyBufferToImage>(vkGetInstanceProcAddrGS(instance, "vkCmdCopyBufferToImage"));

            vkCreateSamplerGS = reinterpret_cast<PFN_vkCreateSampler>(vkGetInstanceProcAddrGS(instance, "vkCreateSampler"));
            vkDestroySamplerGS =
                reinterpret_cast<PFN_vkDestroySampler>(vkGetInstanceProcAddrGS(instance, "vkDestroySampler"));
            ;

            vkCreateSemaphoreGS =
                reinterpret_cast<PFN_vkCreateSemaphore>(vkGetInstanceProcAddrGS(instance, "vkCreateSemaphore"));
            vkDestroySemaphoreGS =
                reinterpret_cast<PFN_vkDestroySemaphore>(vkGetInstanceProcAddrGS(instance, "vkDestroySemaphore"));

            vkCreateFenceGS = reinterpret_cast<PFN_vkCreateFence>(vkGetInstanceProcAddrGS(instance, "vkCreateFence"));
            vkDestroyFenceGS = reinterpret_cast<PFN_vkDestroyFence>(vkGetInstanceProcAddrGS(instance, "vkDestroyFence"));
            vkWaitForFencesGS = reinterpret_cast<PFN_vkWaitForFences>(vkGetInstanceProcAddrGS(instance, "vkWaitForFences"));
            vkResetFencesGS = reinterpret_cast<PFN_vkResetFences>(vkGetInstanceProcAddrGS(instance, "vkResetFences"));
            ;
            vkResetDescriptorPoolGS =
                reinterpret_cast<PFN_vkResetDescriptorPool>(vkGetInstanceProcAddrGS(instance, "vkResetDescriptorPool"));

            vkCreateCommandPoolGS =
                reinterpret_cast<PFN_vkCreateCommandPool>(vkGetInstanceProcAddrGS(instance, "vkCreateCommandPool"));
            vkDestroyCommandPoolGS =
                reinterpret_cast<PFN_vkDestroyCommandPool>(vkGetInstanceProcAddrGS(instance, "vkDestroyCommandPool"));
            ;

            vkAllocateCommandBuffersGS = reinterpret_cast<PFN_vkAllocateCommandBuffers>(
                vkGetInstanceProcAddrGS(instance, "vkAllocateCommandBuffers"));
            vkBeginCommandBufferGS =
                reinterpret_cast<PFN_vkBeginCommandBuffer>(vkGetInstanceProcAddrGS(instance, "vkBeginCommandBuffer"));
            vkEndCommandBufferGS =
                reinterpret_cast<PFN_vkEndCommandBuffer>(vkGetInstanceProcAddrGS(instance, "vkEndCommandBuffer"));

            vkGetDeviceQueueGS =
                reinterpret_cast<PFN_vkGetDeviceQueue>(vkGetInstanceProcAddrGS(instance, "vkGetDeviceQueue"));
            vkQueueSubmitGS = reinterpret_cast<PFN_vkQueueSubmit>(vkGetInstanceProcAddrGS(instance, "vkQueueSubmit"));
            vkQueueWaitIdleGS = reinterpret_cast<PFN_vkQueueWaitIdle>(vkGetInstanceProcAddrGS(instance, "vkQueueWaitIdle"));

            vkDeviceWaitIdleGS =
                reinterpret_cast<PFN_vkDeviceWaitIdle>(vkGetInstanceProcAddrGS(instance, "vkDeviceWaitIdle"));

            vkCreateFramebufferGS =
                reinterpret_cast<PFN_vkCreateFramebuffer>(vkGetInstanceProcAddrGS(instance, "vkCreateFramebuffer"));

            vkCreatePipelineCacheGS =
                reinterpret_cast<PFN_vkCreatePipelineCache>(vkGetInstanceProcAddrGS(instance, "vkCreatePipelineCache"));
            vkCreatePipelineLayoutGS =
                reinterpret_cast<PFN_vkCreatePipelineLayout>(vkGetInstanceProcAddrGS(instance, "vkCreatePipelineLayout"));
            vkCreateGraphicsPipelinesGS = reinterpret_cast<PFN_vkCreateGraphicsPipelines>(
                vkGetInstanceProcAddrGS(instance, "vkCreateGraphicsPipelines"));
            vkCreateComputePipelinesGS = reinterpret_cast<PFN_vkCreateComputePipelines>(
                vkGetInstanceProcAddrGS(instance, "vkCreateComputePipelines"));

            vkCreateDescriptorPoolGS =
                reinterpret_cast<PFN_vkCreateDescriptorPool>(vkGetInstanceProcAddrGS(instance, "vkCreateDescriptorPool"));
            vkCreateDescriptorSetLayoutGS = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(
                vkGetInstanceProcAddrGS(instance, "vkCreateDescriptorSetLayout"));

            vkAllocateDescriptorSetsGS = reinterpret_cast<PFN_vkAllocateDescriptorSets>(
                vkGetInstanceProcAddrGS(instance, "vkAllocateDescriptorSets"));
            vkUpdateDescriptorSetsGS =
                reinterpret_cast<PFN_vkUpdateDescriptorSets>(vkGetInstanceProcAddrGS(instance, "vkUpdateDescriptorSets"));

            vkCmdBindDescriptorSetsGS = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(
                vkGetInstanceProcAddrGS(instance, "vkCmdBindDescriptorSets"));
            vkCmdBindPipelineGS =
                reinterpret_cast<PFN_vkCmdBindPipeline>(vkGetInstanceProcAddrGS(instance, "vkCmdBindPipeline"));
            vkCmdBindVertexBuffersGS =
                reinterpret_cast<PFN_vkCmdBindVertexBuffers>(vkGetInstanceProcAddrGS(instance, "vkCmdBindVertexBuffers"));
            vkCmdBindIndexBufferGS =
                reinterpret_cast<PFN_vkCmdBindIndexBuffer>(vkGetInstanceProcAddrGS(instance, "vkCmdBindIndexBuffer"));

            vkCmdSetViewportGS =
                reinterpret_cast<PFN_vkCmdSetViewport>(vkGetInstanceProcAddrGS(instance, "vkCmdSetViewport"));
            vkCmdSetScissorGS = reinterpret_cast<PFN_vkCmdSetScissor>(vkGetInstanceProcAddrGS(instance, "vkCmdSetScissor"));
            vkCmdSetLineWidthGS =
                reinterpret_cast<PFN_vkCmdSetLineWidth>(vkGetInstanceProcAddrGS(instance, "vkCmdSetLineWidth"));
            vkCmdSetDepthBiasGS =
                reinterpret_cast<PFN_vkCmdSetDepthBias>(vkGetInstanceProcAddrGS(instance, "vkCmdSetDepthBias"));
            vkCmdPushConstantsGS =
                reinterpret_cast<PFN_vkCmdPushConstants>(vkGetInstanceProcAddrGS(instance, "vkCmdPushConstants"));
            ;

            vkCmdDrawIndexedGS =
                reinterpret_cast<PFN_vkCmdDrawIndexed>(vkGetInstanceProcAddrGS(instance, "vkCmdDrawIndexed"));
            vkCmdDrawGS = reinterpret_cast<PFN_vkCmdDraw>(vkGetInstanceProcAddrGS(instance, "vkCmdDraw"));
            vkCmdDrawIndexedIndirectGS = reinterpret_cast<PFN_vkCmdDrawIndexedIndirect>(
                vkGetInstanceProcAddrGS(instance, "vkCmdDrawIndexedIndirect"));
            vkCmdDrawIndirectGS =
                reinterpret_cast<PFN_vkCmdDrawIndirect>(vkGetInstanceProcAddrGS(instance, "vkCmdDrawIndirect"));
            vkCmdDispatchGS = reinterpret_cast<PFN_vkCmdDispatch>(vkGetInstanceProcAddrGS(instance, "vkCmdDispatch"));

            vkDestroyPipelineGS =
                reinterpret_cast<PFN_vkDestroyPipeline>(vkGetInstanceProcAddrGS(instance, "vkDestroyPipeline"));
            vkDestroyPipelineLayoutGS = reinterpret_cast<PFN_vkDestroyPipelineLayout>(
                vkGetInstanceProcAddrGS(instance, "vkDestroyPipelineLayout"));
            ;
            vkDestroyDescriptorSetLayoutGS = reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(
                vkGetInstanceProcAddrGS(instance, "vkDestroyDescriptorSetLayout"));
            vkDestroyDeviceGS = reinterpret_cast<PFN_vkDestroyDevice>(vkGetInstanceProcAddrGS(instance, "vkDestroyDevice"));
            vkDestroyInstanceGS =
                reinterpret_cast<PFN_vkDestroyInstance>(vkGetInstanceProcAddrGS(instance, "vkDestroyInstance"));
            vkDestroyDescriptorPoolGS = reinterpret_cast<PFN_vkDestroyDescriptorPool>(
                vkGetInstanceProcAddrGS(instance, "vkDestroyDescriptorPool"));
            vkFreeCommandBuffersGS =
                reinterpret_cast<PFN_vkFreeCommandBuffers>(vkGetInstanceProcAddrGS(instance, "vkFreeCommandBuffers"));
            vkDestroyRenderPassGS =
                reinterpret_cast<PFN_vkDestroyRenderPass>(vkGetInstanceProcAddrGS(instance, "vkDestroyRenderPass"));
            vkDestroyFramebufferGS =
                reinterpret_cast<PFN_vkDestroyFramebuffer>(vkGetInstanceProcAddrGS(instance, "vkDestroyFramebuffer"));
            vkDestroyShaderModuleGS =
                reinterpret_cast<PFN_vkDestroyShaderModule>(vkGetInstanceProcAddrGS(instance, "vkDestroyShaderModule"));
            vkDestroyPipelineCacheGS =
                reinterpret_cast<PFN_vkDestroyPipelineCache>(vkGetInstanceProcAddrGS(instance, "vkDestroyPipelineCache"));

            vkCreateQueryPoolGS =
                reinterpret_cast<PFN_vkCreateQueryPool>(vkGetInstanceProcAddrGS(instance, "vkCreateQueryPool"));
            vkDestroyQueryPoolGS =
                reinterpret_cast<PFN_vkDestroyQueryPool>(vkGetInstanceProcAddrGS(instance, "vkDestroyQueryPool"));
            vkGetQueryPoolResultsGS =
                reinterpret_cast<PFN_vkGetQueryPoolResults>(vkGetInstanceProcAddrGS(instance, "vkGetQueryPoolResults"));

            vkCmdBeginQueryGS = reinterpret_cast<PFN_vkCmdBeginQuery>(vkGetInstanceProcAddrGS(instance, "vkCmdBeginQuery"));
            vkCmdEndQueryGS = reinterpret_cast<PFN_vkCmdEndQuery>(vkGetInstanceProcAddrGS(instance, "vkCmdEndQuery"));
            vkCmdResetQueryPoolGS =
                reinterpret_cast<PFN_vkCmdResetQueryPool>(vkGetInstanceProcAddrGS(instance, "vkCmdResetQueryPool"));
            vkCmdCopyQueryPoolResultsGS = reinterpret_cast<PFN_vkCmdCopyQueryPoolResults>(
                vkGetInstanceProcAddrGS(instance, "vkCmdCopyQueryPoolResults"));
            vkCmdWriteTimestampGS =
                reinterpret_cast<PFN_vkCmdWriteTimestamp>(vkGetInstanceProcAddrGS(instance, "vkCmdWriteTimestamp"));
            vkCmdUpdateBufferGS =
                reinterpret_cast<PFN_vkCmdUpdateBuffer>(vkGetInstanceProcAddrGS(instance, "vkCmdUpdateBuffer"));
            vkGetBufferDeviceAddressGS =
                reinterpret_cast<PFN_vkGetBufferDeviceAddress>(vkGetInstanceProcAddrGS(instance, "vkGetBufferDeviceAddress"));

            vkCreateSurfaceOHOSGS =
                reinterpret_cast<PFN_vkCreateSurfaceOHOS>(vkGetInstanceProcAddrGS(instance, "vkCreateSurfaceOHOS"));
            vkDestroySurfaceKHRGS =
                reinterpret_cast<PFN_vkDestroySurfaceKHR>(vkGetInstanceProcAddrGS(instance, "vkDestroySurfaceKHR"));

            vkCmdFillBufferGS = reinterpret_cast<PFN_vkCmdFillBuffer>(vkGetInstanceProcAddrGS(instance, "vkCmdFillBuffer"));

            if (!vkCreateRenderPass2KHRGS) {
                vkCreateRenderPass2KHRGS = reinterpret_cast<PFN_vkCreateRenderPass2KHR>(
                    vkGetInstanceProcAddrGS(instance, "vkCreateRenderPass2KHR"));
            }

            if (!vkGetPhysicalDeviceFragmentShadingRatesKHRGS) {
                vkGetPhysicalDeviceFragmentShadingRatesKHRGS =
                    reinterpret_cast<PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR>(
                        vkGetInstanceProcAddrGS(instance, "vkGetPhysicalDeviceFragmentShadingRatesKHR"));
            }
            LOGI("Loaded vulkan functions");
        }
        void freeVulkanLibrary() { dlclose(libVulkan); }

        void loadDeviceFunc(VkDevice device)
        {
            if (!vkCmdSetFragmentShadingRateKHRGS) {
                vkCmdSetFragmentShadingRateKHRGS = reinterpret_cast<PFN_vkCmdSetFragmentShadingRateKHR>(
                    vkGetDeviceProcAddrGS(device, "vkCmdSetFragmentShadingRateKHR"));
            }
        }
    }
}