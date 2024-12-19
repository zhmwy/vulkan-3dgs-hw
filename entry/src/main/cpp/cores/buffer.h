#ifndef VKGS_ENGINE_VULKAN_BUFFER_H
#define VKGS_ENGINE_VULKAN_BUFFER_H

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "cpu_buffer.h"

namespace vkgs {

class Buffer {
 public:
  Buffer();

  Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage);

  ~Buffer();

  operator bool() const noexcept { return impl_ != nullptr; }

  operator VkBuffer() const;

  VkDeviceSize size() const;

  void FromCpu(VkCommandBuffer command_buffer, const void* src,
               VkDeviceSize size);
  void FromCpu(VkCommandBuffer command_buffer, const void* src);

  template <typename T>
  void FromCpu(VkCommandBuffer command_buffer, const std::vector<T>& v) {
    FromCpu(command_buffer, v.data(), v.size() * sizeof(T));
  }
public:
    CpuBuffer cpu_buffer;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk

#endif  // VKGS_ENGINE_VULKAN_BUFFER_H
