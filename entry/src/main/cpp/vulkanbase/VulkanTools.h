
#pragma once

#include "vulkan/vulkan.h"
#include "VulkanInitializers.hpp"

#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include "VulkanOhos.h"
#include "common/common.h"

#define VK_FLAGS_NONE 0
#define DEFAULT_FENCE_TIMEOUT 100000000000

#define VK_CHECK_RESULT(f)																				\
{																										\
	VkResult res = (f);																					\
	if (res != VK_SUCCESS)																				\
	{																									\
		LOGE("VkResult is %{public}s in %{public}s, at line %{public}u", vks::tools::getErrorString(res).c_str(), __FILE__, __LINE__); \
	}																									\
}
	//assert(res == VK_SUCCESS);																		\

namespace vks
{
	namespace tools
	{
		std::string getErrorString(VkResult errorCode);

		VkBool32 supportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat *depthFormat);

		void setImageLayout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkImageSubresourceRange subresourceRange,
			VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        VkShaderModule loadSpvShader(const char *fileName, VkDevice device);
	}
}
