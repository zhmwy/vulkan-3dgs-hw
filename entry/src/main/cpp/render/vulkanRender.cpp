#include "vulkanRender.h"

bool VulkanRender::prepare()
{
	VulkanComputeBase::prepare();
	VulkanComputeExample::prepare();
	prepared = true;
	return true;
}