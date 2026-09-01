#ifndef PXVK_VULKAN_BACKEND_H
#define PXVK_VULKAN_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace pxvk {

struct DispatchInfo {
    std::string shader_path;
    const void *parameters = nullptr;
    std::size_t parameter_bytes = 0;
    const void *constants = nullptr;
    std::size_t constant_bytes = 0;
    const void *texture_metadata = nullptr;
    std::size_t texture_metadata_bytes = 0;
    std::uint32_t grid_x = 1;
    std::uint32_t grid_y = 1;
    std::uint32_t grid_z = 1;
    std::uint32_t shared_bytes_per_group = 0;
    bool storage_backed_shared = false;
};

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();
    VulkanBackend(const VulkanBackend &) = delete;
    VulkanBackend &operator=(const VulkanBackend &) = delete;

    bool initialize(std::string &error);
    void shutdown();
    bool initialized() const;

    const std::string &device_name() const;
    std::uint64_t total_device_memory() const;
    std::uint32_t max_workgroup_invocations() const;
    std::uint32_t max_workgroup_size_x() const;
    std::uint32_t max_grid_size_x() const;
    std::uint32_t max_shared_memory() const;
    std::uint32_t compute_unit_count() const;

    bool ensure_heap(std::size_t bytes, std::string &error);
    std::size_t heap_size() const;
    bool write_heap(std::uint32_t address, const void *source, std::size_t bytes,
                    std::string &error);
    bool read_heap(void *destination, std::uint32_t address, std::size_t bytes,
                   std::string &error);
    bool move_heap(std::uint32_t destination, std::uint32_t source,
                   std::size_t bytes, std::string &error);
    bool fill_heap(std::uint32_t destination, std::uint8_t value,
                   std::size_t bytes, std::string &error);
    bool dispatch(const DispatchInfo &info, std::string &error);
    bool synchronize(std::string &error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pxvk

#endif
