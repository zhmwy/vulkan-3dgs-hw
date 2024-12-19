#ifndef VKGS_ENGINE_VULKAN_CPU_BUFFER_H
#define VKGS_ENGINE_VULKAN_CPU_BUFFER_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

namespace vkgs {

class CpuBuffer {
 public:
  CpuBuffer();

  CpuBuffer(VmaAllocator allocator, VkDeviceSize size);

  ~CpuBuffer();

  operator VkBuffer() const;

  const void* data() const;
  VkDeviceSize size() const;
  bool isInit = false;

 private:
	 
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_CPU_BUFFER_H
