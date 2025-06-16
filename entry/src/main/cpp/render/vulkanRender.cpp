#include "vulkanRender.h"

bool VulkanRender::prepare()
{
	VulkanExampleBase::prepare();
	VulkanComputeExample::prepare();
    buildCommandBuffers();
	prepared = true;
	return true;
}

void VulkanRender::draw()
{
    //if (render_count < 2)
    //{
    //    // buildComputeCommandBuffer();
    //    buildCommandBuffers();
    //}
    if (cur_method != use_method || upd_cmd_buffer) {
        upd_cmd_buffer = false;
        // buildComputeCommandBuffer();
        buildCommandBuffers();
#ifdef VULKAN_HW
        LOGI("VulkanExample rebuild command buffers");
#elif defined(VULKAN_PC)
        // std::cout << "VulkanExample rebuild command buffers" << std::endl;
#endif
        cur_method = use_method;
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
    VkResult res = vkQueueSubmitGS(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
#ifdef VULKAN_HW
        LOGE("VulkanExample Fatal : VkResult is %s", vks::tools::errorString(res).c_str());
#elif defined(VULKAN_PC)
        std::cerr << "VulkanExample Fatal : VkResult is " << vks::tools::errorString(res) << std::endl;
#endif
    }
    VulkanExampleBase::submitFrame();
}