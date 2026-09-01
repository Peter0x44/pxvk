#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "vulkan_backend.h"

#include <volk.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pxvk {
namespace {

bool checked_range(std::size_t offset, std::size_t bytes, std::size_t limit)
{
    return offset <= limit && bytes <= limit - offset;
}

std::vector<std::uint32_t> read_spirv(const std::string &path, std::string &error)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "cannot open shader '" + path + "'";
        return {};
    }
    const std::streamoff end = input.tellg();
    if (end <= 0 || (end % 4) != 0 ||
        static_cast<std::uint64_t>(end) > std::numeric_limits<std::uint32_t>::max()) {
        error = "invalid SPIR-V size for '" + path + "'";
        return {};
    }
    std::vector<std::uint32_t> words(static_cast<std::size_t>(end) / 4u);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char *>(words.data()), end);
    if (!input || words.empty() || words.front() != 0x07230203u) {
        error = "invalid SPIR-V module '" + path + "'";
        return {};
    }
    return words;
}

} // namespace

struct VulkanBackend::Impl {
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void *mapped = nullptr;
        VkDeviceSize size = 0;
        bool coherent = false;
    };

    struct Pipeline {
        VkShaderModule shader = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        std::vector<std::uint32_t> bindings;
        Buffer parameters;
        Buffer constants;
        Buffer textures;
        Buffer scratch;
    };

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    std::string name;
    std::uint64_t total_memory = 0;
    Buffer heap;
    std::unordered_map<std::string, Pipeline> pipelines;

    void destroy_buffer(Buffer &value)
    {
        if (device == VK_NULL_HANDLE) return;
        if (value.mapped != nullptr) vkUnmapMemory(device, value.memory);
        if (value.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, value.buffer, nullptr);
        if (value.memory != VK_NULL_HANDLE) vkFreeMemory(device, value.memory, nullptr);
        value = {};
    }

    void destroy_pipeline(Pipeline &value)
    {
        destroy_buffer(value.scratch);
        destroy_buffer(value.textures);
        destroy_buffer(value.constants);
        destroy_buffer(value.parameters);
        if (device != VK_NULL_HANDLE) {
            if (value.pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(device, value.pipeline, nullptr);
            if (value.descriptor_pool != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device, value.descriptor_pool, nullptr);
            if (value.layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(device, value.layout, nullptr);
            if (value.descriptor_layout != VK_NULL_HANDLE)
                vkDestroyDescriptorSetLayout(device, value.descriptor_layout, nullptr);
            if (value.shader != VK_NULL_HANDLE)
                vkDestroyShaderModule(device, value.shader, nullptr);
        }
        value = {};
    }

    bool memory_type(std::uint32_t bits, std::uint32_t &index, bool &coherent) const
    {
        int fallback = -1;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) == 0u) continue;
            const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0u) continue;
            if (fallback < 0) fallback = static_cast<int>(i);
            if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u) {
                index = i;
                coherent = true;
                return true;
            }
        }
        if (fallback >= 0) {
            index = static_cast<std::uint32_t>(fallback);
            coherent = false;
            return true;
        }
        return false;
    }

    bool create_buffer(VkDeviceSize requested, Buffer &output, std::string &error)
    {
        const VkDeviceSize size = std::max<VkDeviceSize>(requested, 4u);
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &buffer_info, nullptr, &output.buffer) != VK_SUCCESS) {
            error = "vkCreateBuffer failed";
            return false;
        }

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, output.buffer, &requirements);
        std::uint32_t type_index = 0;
        bool coherent = false;
        if (!memory_type(requirements.memoryTypeBits, type_index, coherent)) {
            error = "no host-visible Vulkan memory type for storage buffer";
            destroy_buffer(output);
            return false;
        }

        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type_index;
        if (vkAllocateMemory(device, &allocation, nullptr, &output.memory) != VK_SUCCESS) {
            error = "vkAllocateMemory failed";
            destroy_buffer(output);
            return false;
        }
        if (vkBindBufferMemory(device, output.buffer, output.memory, 0) != VK_SUCCESS) {
            error = "vkBindBufferMemory failed";
            destroy_buffer(output);
            return false;
        }
        if (vkMapMemory(device, output.memory, 0, size, 0, &output.mapped) != VK_SUCCESS) {
            error = "vkMapMemory failed";
            destroy_buffer(output);
            return false;
        }
        output.size = size;
        output.coherent = coherent;
        std::memset(output.mapped, 0, static_cast<std::size_t>(size));
        return flush(output, 0, size, error);
    }

    bool flush(Buffer &buffer, VkDeviceSize offset, VkDeviceSize size, std::string &error)
    {
        if (buffer.coherent || size == 0) return true;
        const VkDeviceSize atom = properties.limits.nonCoherentAtomSize;
        const VkDeviceSize start = (offset / atom) * atom;
        const VkDeviceSize end = std::min(
            buffer.size,
            ((offset + size + atom - 1u) / atom) * atom);
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = buffer.memory;
        range.offset = start;
        range.size = end - start;
        if (vkFlushMappedMemoryRanges(device, 1, &range) != VK_SUCCESS) {
            error = "vkFlushMappedMemoryRanges failed";
            return false;
        }
        return true;
    }

    bool invalidate(Buffer &buffer, VkDeviceSize offset, VkDeviceSize size, std::string &error)
    {
        if (buffer.coherent || size == 0) return true;
        const VkDeviceSize atom = properties.limits.nonCoherentAtomSize;
        const VkDeviceSize start = (offset / atom) * atom;
        const VkDeviceSize end = std::min(
            buffer.size,
            ((offset + size + atom - 1u) / atom) * atom);
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = buffer.memory;
        range.offset = start;
        range.size = end - start;
        if (vkInvalidateMappedMemoryRanges(device, 1, &range) != VK_SUCCESS) {
            error = "vkInvalidateMappedMemoryRanges failed";
            return false;
        }
        return true;
    }

    bool resize_buffer(Buffer &buffer, std::size_t bytes, std::string &error)
    {
        if (buffer.size >= bytes) return true;
        Buffer replacement;
        if (!create_buffer(bytes, replacement, error)) return false;
        if (buffer.mapped != nullptr) {
            if (!invalidate(buffer, 0, buffer.size, error)) {
                destroy_buffer(replacement);
                return false;
            }
            std::memcpy(replacement.mapped, buffer.mapped,
                        static_cast<std::size_t>(buffer.size));
            if (!flush(replacement, 0, buffer.size, error)) {
                destroy_buffer(replacement);
                return false;
            }
        }
        destroy_buffer(buffer);
        buffer = replacement;
        return true;
    }

    bool create_pipeline(const DispatchInfo &info, Pipeline &output, std::string &error)
    {
        const std::vector<std::uint32_t> words = read_spirv(info.shader_path, error);
        if (words.empty()) return false;

        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = words.size() * sizeof(std::uint32_t);
        shader_info.pCode = words.data();
        if (vkCreateShaderModule(device, &shader_info, nullptr, &output.shader) != VK_SUCCESS) {
            error = "vkCreateShaderModule failed for '" + info.shader_path + "'";
            return false;
        }

        output.bindings = {0u, 1u};
        if (info.constant_bytes != 0u) output.bindings.push_back(2u);
        if (info.texture_metadata_bytes != 0u) output.bindings.push_back(3u);
        if (info.storage_backed_shared) output.bindings.push_back(4u);

        std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
        layout_bindings.reserve(output.bindings.size());
        for (std::uint32_t binding : output.bindings) {
            VkDescriptorSetLayoutBinding item{};
            item.binding = binding;
            item.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            item.descriptorCount = 1;
            item.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            layout_bindings.push_back(item);
        }
        VkDescriptorSetLayoutCreateInfo descriptor_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptor_info.bindingCount = static_cast<std::uint32_t>(layout_bindings.size());
        descriptor_info.pBindings = layout_bindings.data();
        if (vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr,
                                        &output.descriptor_layout) != VK_SUCCESS) {
            error = "vkCreateDescriptorSetLayout failed";
            destroy_pipeline(output);
            return false;
        }

        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &output.descriptor_layout;
        if (vkCreatePipelineLayout(device, &layout_info, nullptr, &output.layout) != VK_SUCCESS) {
            error = "vkCreatePipelineLayout failed";
            destroy_pipeline(output);
            return false;
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = static_cast<std::uint32_t>(output.bindings.size());
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        if (vkCreateDescriptorPool(device, &pool_info, nullptr,
                                   &output.descriptor_pool) != VK_SUCCESS) {
            error = "vkCreateDescriptorPool failed";
            destroy_pipeline(output);
            return false;
        }
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = output.descriptor_pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &output.descriptor_layout;
        if (vkAllocateDescriptorSets(device, &set_info, &output.descriptor_set) != VK_SUCCESS) {
            error = "vkAllocateDescriptorSets failed";
            destroy_pipeline(output);
            return false;
        }

        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = output.shader;
        pipeline_info.stage.pName = "main";
        const std::uint32_t shared_words =
            std::max(1u, (info.shared_bytes_per_group + 3u) / 4u);
        VkSpecializationMapEntry shared_stride_entry{};
        shared_stride_entry.constantID = 0u;
        shared_stride_entry.offset = 0u;
        shared_stride_entry.size = sizeof(shared_words);
        VkSpecializationInfo shared_stride_info{};
        shared_stride_info.mapEntryCount = 1u;
        shared_stride_info.pMapEntries = &shared_stride_entry;
        shared_stride_info.dataSize = sizeof(shared_words);
        shared_stride_info.pData = &shared_words;
        if (info.dynamic_shared_stride)
            pipeline_info.stage.pSpecializationInfo = &shared_stride_info;
        pipeline_info.layout = output.layout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                     &output.pipeline) != VK_SUCCESS) {
            error = "vkCreateComputePipelines failed for '" + info.shader_path + "'";
            destroy_pipeline(output);
            return false;
        }
        return true;
    }

    bool prepare_aux(Buffer &buffer, const void *data, std::size_t bytes,
                     std::string &error)
    {
        const std::size_t needed = std::max<std::size_t>(bytes, 4u);
        if (!resize_buffer(buffer, needed, error)) return false;
        if (data != nullptr && bytes != 0u) std::memcpy(buffer.mapped, data, bytes);
        if (bytes < static_cast<std::size_t>(buffer.size)) {
            std::memset(static_cast<unsigned char *>(buffer.mapped) + bytes, 0,
                        static_cast<std::size_t>(buffer.size) - bytes);
        }
        return flush(buffer, 0, buffer.size, error);
    }
};

VulkanBackend::VulkanBackend() : impl_(new (std::nothrow) Impl) {}
VulkanBackend::~VulkanBackend() { shutdown(); }

bool VulkanBackend::initialize(std::string &error)
{
    if (!impl_) {
        error = "out of memory creating Vulkan backend";
        return false;
    }
    if (impl_->device != VK_NULL_HANDLE) return true;
    if (volkInitialize() != VK_SUCCESS) {
        error = "32-bit Vulkan loader is unavailable";
        return false;
    }

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "pxvk";
    application.applicationVersion = 1;
    application.pEngineName = "pxvk";
    application.engineVersion = 1;
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    if (vkCreateInstance(&instance_info, nullptr, &impl_->instance) != VK_SUCCESS) {
        error = "vkCreateInstance(Vulkan 1.2) failed";
        shutdown();
        return false;
    }
    volkLoadInstance(impl_->instance);

    std::uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(impl_->instance, &device_count, nullptr) != VK_SUCCESS ||
        device_count == 0u) {
        error = "no Vulkan physical device";
        shutdown();
        return false;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(impl_->instance, &device_count, devices.data()) != VK_SUCCESS) {
        error = "vkEnumeratePhysicalDevices failed";
        shutdown();
        return false;
    }

    int forced_index = -1;
    if (const char *value = std::getenv("PXVK_VULKAN_DEVICE")) {
        char *end = nullptr;
        errno = 0;
        const long parsed = std::strtol(value, &end, 10);
        if (errno == 0 && end != value && *end == '\0' && parsed >= 0 &&
            parsed < static_cast<long>(device_count)) {
            forced_index = static_cast<int>(parsed);
        } else {
            error = "PXVK_VULKAN_DEVICE is not a valid device index";
            shutdown();
            return false;
        }
    }

    auto try_device = [&](std::uint32_t index) -> bool {
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceProperties(devices[index], &properties);
        vkGetPhysicalDeviceFeatures(devices[index], &features);
        if (properties.apiVersion < VK_API_VERSION_1_2 || !features.shaderInt64 ||
            properties.limits.maxComputeWorkGroupInvocations < 512u ||
            properties.limits.maxStorageBufferRange < (128u * 1024u * 1024u)) {
            return false;
        }
        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &family_count, families.data());
        for (std::uint32_t family = 0; family < family_count; ++family) {
            if (families[family].queueCount != 0u &&
                (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
                impl_->physical_device = devices[index];
                impl_->queue_family = family;
                impl_->properties = properties;
                return true;
            }
        }
        return false;
    };

    if (forced_index >= 0) {
        if (!try_device(static_cast<std::uint32_t>(forced_index))) {
            error = "selected Vulkan device does not satisfy the pxvk gate";
            shutdown();
            return false;
        }
    } else {
        for (std::uint32_t i = 0; i < device_count; ++i) {
            VkPhysicalDeviceProperties candidate{};
            vkGetPhysicalDeviceProperties(devices[i], &candidate);
            if (candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
            if (try_device(i)) break;
        }
        if (impl_->physical_device == VK_NULL_HANDLE) {
            for (std::uint32_t i = 0; i < device_count; ++i) {
                if (try_device(i)) break;
            }
        }
        if (impl_->physical_device == VK_NULL_HANDLE) {
            error = "no Vulkan device satisfies the pxvk gate";
            shutdown();
            return false;
        }
    }

    vkGetPhysicalDeviceMemoryProperties(impl_->physical_device, &impl_->memory_properties);
    impl_->name = impl_->properties.deviceName;
    for (std::uint32_t i = 0; i < impl_->memory_properties.memoryHeapCount; ++i) {
        if ((impl_->memory_properties.memoryHeaps[i].flags &
             VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
            impl_->total_memory += impl_->memory_properties.memoryHeaps[i].size;
        }
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = impl_->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enabled;
    if (vkCreateDevice(impl_->physical_device, &device_info, nullptr, &impl_->device) !=
        VK_SUCCESS) {
        error = "vkCreateDevice failed";
        shutdown();
        return false;
    }
    volkLoadDevice(impl_->device);
    vkGetDeviceQueue(impl_->device, impl_->queue_family, 0, &impl_->queue);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = impl_->queue_family;
    if (vkCreateCommandPool(impl_->device, &pool_info, nullptr, &impl_->command_pool) !=
        VK_SUCCESS) {
        error = "vkCreateCommandPool failed";
        shutdown();
        return false;
    }
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = impl_->command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(impl_->device, &command_info, &impl_->command_buffer) !=
        VK_SUCCESS) {
        error = "vkAllocateCommandBuffers failed";
        shutdown();
        return false;
    }
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(impl_->device, &fence_info, nullptr, &impl_->fence) != VK_SUCCESS) {
        error = "vkCreateFence failed";
        shutdown();
        return false;
    }
    return true;
}

void VulkanBackend::shutdown()
{
    if (!impl_) return;
    if (impl_->device != VK_NULL_HANDLE) vkDeviceWaitIdle(impl_->device);
    for (auto &item : impl_->pipelines) impl_->destroy_pipeline(item.second);
    impl_->pipelines.clear();
    impl_->destroy_buffer(impl_->heap);
    if (impl_->device != VK_NULL_HANDLE) {
        if (impl_->fence != VK_NULL_HANDLE) vkDestroyFence(impl_->device, impl_->fence, nullptr);
        if (impl_->command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(impl_->device, impl_->command_pool, nullptr);
        vkDestroyDevice(impl_->device, nullptr);
    }
    if (impl_->instance != VK_NULL_HANDLE) vkDestroyInstance(impl_->instance, nullptr);
    impl_->instance = VK_NULL_HANDLE;
    impl_->physical_device = VK_NULL_HANDLE;
    impl_->device = VK_NULL_HANDLE;
    impl_->queue = VK_NULL_HANDLE;
    impl_->command_pool = VK_NULL_HANDLE;
    impl_->command_buffer = VK_NULL_HANDLE;
    impl_->fence = VK_NULL_HANDLE;
    impl_->name.clear();
    impl_->total_memory = 0;
}

bool VulkanBackend::initialized() const
{
    return impl_ && impl_->device != VK_NULL_HANDLE;
}

const std::string &VulkanBackend::device_name() const
{
    static const std::string empty;
    return impl_ ? impl_->name : empty;
}

std::uint64_t VulkanBackend::total_device_memory() const
{
    return impl_ ? impl_->total_memory : 0u;
}

std::uint32_t VulkanBackend::max_workgroup_invocations() const
{
    return impl_ ? impl_->properties.limits.maxComputeWorkGroupInvocations : 0u;
}

std::uint32_t VulkanBackend::max_workgroup_size_x() const
{
    return impl_ ? impl_->properties.limits.maxComputeWorkGroupSize[0] : 0u;
}

std::uint32_t VulkanBackend::max_grid_size_x() const
{
    return impl_ ? impl_->properties.limits.maxComputeWorkGroupCount[0] : 0u;
}

std::uint32_t VulkanBackend::max_shared_memory() const
{
    return impl_ ? impl_->properties.limits.maxComputeSharedMemorySize : 0u;
}

std::uint32_t VulkanBackend::compute_unit_count() const
{
    return 1u;
}

bool VulkanBackend::ensure_heap(std::size_t bytes, std::string &error)
{
    if (!initialized()) {
        error = "Vulkan backend is not initialized";
        return false;
    }
    if (bytes > impl_->properties.limits.maxStorageBufferRange) {
        error = "CUDA virtual heap exceeds maxStorageBufferRange";
        return false;
    }
    return impl_->resize_buffer(impl_->heap, bytes, error);
}

std::size_t VulkanBackend::heap_size() const
{
    return impl_ ? static_cast<std::size_t>(impl_->heap.size) : 0u;
}

bool VulkanBackend::write_heap(std::uint32_t address, const void *source,
                               std::size_t bytes, std::string &error)
{
    if ((source == nullptr && bytes != 0u) ||
        !checked_range(address, bytes, heap_size())) {
        error = "invalid host-to-device range";
        return false;
    }
    if (bytes != 0u) {
        std::memcpy(static_cast<unsigned char *>(impl_->heap.mapped) + address,
                    source, bytes);
    }
    return impl_->flush(impl_->heap, address, bytes, error);
}

bool VulkanBackend::read_heap(void *destination, std::uint32_t address,
                              std::size_t bytes, std::string &error)
{
    if ((destination == nullptr && bytes != 0u) ||
        !checked_range(address, bytes, heap_size())) {
        error = "invalid device-to-host range";
        return false;
    }
    if (!impl_->invalidate(impl_->heap, address, bytes, error)) return false;
    if (bytes != 0u) {
        std::memcpy(destination,
                    static_cast<const unsigned char *>(impl_->heap.mapped) + address,
                    bytes);
    }
    return true;
}

bool VulkanBackend::move_heap(std::uint32_t destination, std::uint32_t source,
                              std::size_t bytes, std::string &error)
{
    if (!checked_range(destination, bytes, heap_size()) ||
        !checked_range(source, bytes, heap_size())) {
        error = "invalid device-to-device range";
        return false;
    }
    if (!impl_->invalidate(impl_->heap, source, bytes, error)) return false;
    std::memmove(static_cast<unsigned char *>(impl_->heap.mapped) + destination,
                 static_cast<unsigned char *>(impl_->heap.mapped) + source, bytes);
    return impl_->flush(impl_->heap, destination, bytes, error);
}

bool VulkanBackend::fill_heap(std::uint32_t destination, std::uint8_t value,
                              std::size_t bytes, std::string &error)
{
    if (!checked_range(destination, bytes, heap_size())) {
        error = "invalid device memset range";
        return false;
    }
    std::memset(static_cast<unsigned char *>(impl_->heap.mapped) + destination,
                value, bytes);
    return impl_->flush(impl_->heap, destination, bytes, error);
}

bool VulkanBackend::dispatch(const DispatchInfo &info, std::string &error)
{
    if (!initialized() || impl_->heap.buffer == VK_NULL_HANDLE) {
        error = "dispatch requires an initialized CUDA heap";
        return false;
    }
    try {
        std::string pipeline_key = info.shader_path;
        if (info.dynamic_shared_stride) {
            const std::uint32_t shared_words =
                std::max(1u, (info.shared_bytes_per_group + 3u) / 4u);
            pipeline_key += "#" + std::to_string(shared_words);
        }
        auto found = impl_->pipelines.find(pipeline_key);
        if (found == impl_->pipelines.end()) {
            Impl::Pipeline pipeline;
            if (!impl_->create_pipeline(info, pipeline, error)) return false;
            found = impl_->pipelines.emplace(
                std::move(pipeline_key), std::move(pipeline)).first;
        }
        Impl::Pipeline &pipeline = found->second;
        std::vector<std::uint32_t> expected = {0u, 1u};
        if (info.constant_bytes != 0u) expected.push_back(2u);
        if (info.texture_metadata_bytes != 0u) expected.push_back(3u);
        if (info.storage_backed_shared) expected.push_back(4u);
        if (pipeline.bindings != expected) {
            error = "shader resource profile changed for '" + info.shader_path + "'";
            return false;
        }

        if (!impl_->prepare_aux(pipeline.parameters, info.parameters,
                                info.parameter_bytes, error)) return false;
        if (info.constant_bytes != 0u &&
            !impl_->prepare_aux(pipeline.constants, info.constants,
                                info.constant_bytes, error)) return false;
        if (info.texture_metadata_bytes != 0u &&
            !impl_->prepare_aux(pipeline.textures, info.texture_metadata,
                                info.texture_metadata_bytes, error)) return false;
        if (info.storage_backed_shared) {
            const std::uint64_t groups = static_cast<std::uint64_t>(info.grid_x) *
                                         info.grid_y * info.grid_z;
            const std::uint64_t words =
                std::max<std::uint64_t>(1u, (info.shared_bytes_per_group + 3u) / 4u);
            const std::uint64_t bytes = groups * words * 4u;
            if (bytes > std::numeric_limits<std::size_t>::max() ||
                !impl_->prepare_aux(pipeline.scratch, nullptr,
                                    static_cast<std::size_t>(bytes), error)) return false;
        }

        std::vector<VkDescriptorBufferInfo> buffer_infos;
        std::vector<VkWriteDescriptorSet> writes;
        buffer_infos.reserve(pipeline.bindings.size());
        writes.reserve(pipeline.bindings.size());
        auto add_binding = [&](std::uint32_t binding, const Impl::Buffer &buffer) {
            VkDescriptorBufferInfo buffer_info{};
            buffer_info.buffer = buffer.buffer;
            buffer_info.offset = 0;
            buffer_info.range = buffer.size;
            buffer_infos.push_back(buffer_info);
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = pipeline.descriptor_set;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes.push_back(write);
        };
        add_binding(0u, impl_->heap);
        add_binding(1u, pipeline.parameters);
        if (info.constant_bytes != 0u) add_binding(2u, pipeline.constants);
        if (info.texture_metadata_bytes != 0u) add_binding(3u, pipeline.textures);
        if (info.storage_backed_shared) add_binding(4u, pipeline.scratch);
        for (std::size_t i = 0; i < writes.size(); ++i)
            writes[i].pBufferInfo = &buffer_infos[i];
        vkUpdateDescriptorSets(impl_->device, static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);

        if (vkResetCommandBuffer(impl_->command_buffer, 0) != VK_SUCCESS) {
            error = "vkResetCommandBuffer failed";
            return false;
        }
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(impl_->command_buffer, &begin) != VK_SUCCESS) {
            error = "vkBeginCommandBuffer failed";
            return false;
        }
        VkMemoryBarrier host_to_compute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        host_to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(impl_->command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &host_to_compute, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.pipeline);
        vkCmdBindDescriptorSets(impl_->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline.layout, 0, 1, &pipeline.descriptor_set,
                                0, nullptr);
        vkCmdDispatch(impl_->command_buffer, info.grid_x, info.grid_y, info.grid_z);
        VkMemoryBarrier compute_to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        compute_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        compute_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(impl_->command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                             &compute_to_host, 0, nullptr, 0, nullptr);
        if (vkEndCommandBuffer(impl_->command_buffer) != VK_SUCCESS) {
            error = "vkEndCommandBuffer failed";
            return false;
        }
        if (vkResetFences(impl_->device, 1, &impl_->fence) != VK_SUCCESS) {
            error = "vkResetFences failed";
            return false;
        }
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &impl_->command_buffer;
        if (vkQueueSubmit(impl_->queue, 1, &submit, impl_->fence) != VK_SUCCESS) {
            error = "vkQueueSubmit failed";
            return false;
        }
        if (vkWaitForFences(impl_->device, 1, &impl_->fence, VK_TRUE,
                            std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) {
            error = "vkWaitForFences failed";
            return false;
        }
        return impl_->invalidate(impl_->heap, 0, impl_->heap.size, error);
    } catch (const std::bad_alloc &) {
        error = "out of memory preparing Vulkan dispatch";
        return false;
    }
}

bool VulkanBackend::synchronize(std::string &error)
{
    if (!initialized()) {
        error = "Vulkan backend is not initialized";
        return false;
    }
    if (vkQueueWaitIdle(impl_->queue) != VK_SUCCESS) {
        error = "vkQueueWaitIdle failed";
        return false;
    }
    if (impl_->heap.buffer != VK_NULL_HANDLE)
        return impl_->invalidate(impl_->heap, 0, impl_->heap.size, error);
    return true;
}

} // namespace pxvk
