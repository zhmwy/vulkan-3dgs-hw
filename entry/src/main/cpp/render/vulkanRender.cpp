#include "vulkanRender.h"

bool VulkanRender::prepare()
{
	VulkanExampleBase::prepare();
	VulkanComputeExample::prepare();
	prepared = true;
	return true;
}