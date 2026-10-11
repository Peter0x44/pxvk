#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "device.h"

#include <volk.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>

namespace pxvk::vk {

namespace {

constexpr VkDeviceSize kLaunchRange = 4096;        // bound UBO range per dispatch
constexpr VkDeviceSize kLaunchRingBytes = 1u << 20;
constexpr VkDeviceSize kStagingInitialBytes = 16u << 20;
constexpr std::uint32_t kMaxTimedDispatches = 4096;
constexpr std::uint32_t kSpecHeapBase = 4;

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *mapped = nullptr;
};

// A switch is on when set to anything but empty, 0, off, false or no
// (docs/environment.md).
bool env_switch(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0 && _stricmp(value, "off") != 0 &&
           _stricmp(value, "false") != 0 && _stricmp(value, "no") != 0;
}

std::string vk_error(const char *what, VkResult result)
{
    return std::string(what) + " failed (VkResult " + std::to_string(static_cast<int>(result)) + ")";
}

} // namespace

struct HostBuffer {
    Buffer buffer;
};

struct Shader {
    VkShaderModule module = VK_NULL_HANDLE;
    std::uint32_t launch_block_bytes = 0;
    bool warp_synchronous = false;
    bool shared_in_storage = false;
    std::map<std::vector<std::uint32_t>, VkPipeline> pipelines;
};

struct Device::Impl {
    Capabilities caps;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkPhysicalDeviceLimits limits{};
    std::uint32_t timestamp_valid_bits = 0;
    float timestamp_period = 1.0f;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool recording = false;
    std::uint32_t commands = 0;

    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;

    Buffer heap, staging, readback, launch, scratch;
    Buffer dummy; // split heap: target of windows past the heap end (see update_descriptors)
    VkDeviceSize staging_head = 0, launch_head = 0;
    std::uint32_t heap_base = 0;

    std::vector<std::unique_ptr<Shader>> shaders;
    Stats stats;

    bool timing = false;
    bool pipeline_stats = false; // PXVK_PIPELINE_STATS: print driver statistics per pipeline
    VkQueryPool queries = VK_NULL_HANDLE;
    std::uint32_t timed = 0;
    std::vector<double> times;
    std::vector<const char *> timed_labels;
    std::map<std::string, KernelTime> profile;

    // ---- buffers ------------------------------------------------------------
    int find_memory(std::uint32_t bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) const
    {
        int fallback = -1;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if (!(bits & (1u << i))) continue;
            const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
            if ((flags & required) != required) continue;
            if ((flags & preferred) == preferred) return static_cast<int>(i);
            if (fallback < 0) fallback = static_cast<int>(i);
        }
        return fallback;
    }

    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                       VkMemoryPropertyFlags preferred, Buffer &out, std::string &error)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r = vkCreateBuffer(device, &info, nullptr, &out.buffer);
        if (r != VK_SUCCESS) {
            error = vk_error("vkCreateBuffer", r);
            return false;
        }
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
        const int type = find_memory(requirements.memoryTypeBits, required, preferred);
        if (type < 0) {
            error = "no suitable Vulkan memory type";
            destroy(out);
            return false;
        }
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = static_cast<std::uint32_t>(type);
        r = vkAllocateMemory(device, &alloc, nullptr, &out.memory);
        if (r != VK_SUCCESS) {
            error = vk_error("vkAllocateMemory", r);
            destroy(out);
            return false;
        }
        vkBindBufferMemory(device, out.buffer, out.memory, 0);
        if (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            r = vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped);
            if (r != VK_SUCCESS) {
                error = vk_error("vkMapMemory", r);
                destroy(out);
                return false;
            }
        }
        out.size = size;
        return true;
    }

    void destroy(Buffer &b)
    {
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }

    void update_descriptors()
    {
        VkDescriptorBufferInfo infos[5]{};
        VkWriteDescriptorSet writes[5]{};
        // Split heap: binding 0 holds one window per 2^shift bytes, plus one
        // more that shaders clamp out-of-range indices to. That one and the
        // windows past the end of the current heap see a small dummy buffer,
        // so stray accesses never touch live data (robustBufferAccess keeps
        // them inside it). Bindings 1-2, unused, see the first window.
        const VkDeviceSize window = caps.heap_windows ? VkDeviceSize{1} << caps.heap_window_shift : 0;
        std::vector<VkDescriptorBufferInfo> windows(caps.heap_windows ? caps.heap_windows + 1 : 0);
        for (std::uint32_t w = 0; w < windows.size(); ++w) {
            const VkDeviceSize offset = VkDeviceSize{w} * window;
            windows[w] = w < caps.heap_windows && offset < heap.size
                             ? VkDescriptorBufferInfo{heap.buffer, offset, std::min(window, heap.size - offset)}
                             : VkDescriptorBufferInfo{dummy.buffer, 0, VK_WHOLE_SIZE};
        }
        for (std::uint32_t i = 0; i < 5; ++i) {
            infos[i].buffer = i < 3 ? heap.buffer : i == 3 ? launch.buffer : scratch.buffer;
            infos[i].offset = 0;
            infos[i].range = i == 3 ? kLaunchRange : i < 3 && window ? std::min(window, heap.size) : VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType =
                i == 3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        if (caps.heap_windows) {
            writes[0].descriptorCount = caps.heap_windows + 1;
            writes[0].pBufferInfo = windows.data();
        }
        if (heap.buffer) vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    }

    // ---- command batching ---------------------------------------------------
    void begin()
    {
        if (recording) return;
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &info);
        if (timing && queries) vkCmdResetQueryPool(cmd, queries, 0, kMaxTimedDispatches * 2);
        recording = true;
        commands = 0;
        timed = 0;
        timed_labels.clear();
    }

    // Serializes every command against all previous ones (CUDA stream order).
    void barrier()
    {
        begin();
        if (commands++ == 0) return;
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                          VK_ACCESS_UNIFORM_READ_BIT;
        const VkPipelineStageFlags stages =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // After a failed submission the batch is lost: start the next one from
    // empty rings (otherwise the next staging allocation can run past the
    // end of the mapped staging buffer). The command pool is reset when the
    // batch never reached the GPU, so begin() can record into it again.
    void abandon_batch(bool reset_pool)
    {
        if (reset_pool) vkResetCommandPool(device, pool, 0);
        staging_head = 0;
        launch_head = 0;
        timed = 0;
        timed_labels.clear();
    }

    bool submit_and_wait(std::string &error)
    {
        if (!recording) return true;
        // Make every write of the batch visible to the host.
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(cmd);
        recording = false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        VkResult r = vkQueueSubmit(queue, 1, &submit, fence);
        if (r != VK_SUCCESS) {
            error = vk_error("vkQueueSubmit", r);
            abandon_batch(true);
            return false;
        }
        ++stats.submits;
        LARGE_INTEGER wait_start, wait_end, frequency;
        QueryPerformanceCounter(&wait_start);
        r = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        QueryPerformanceCounter(&wait_end);
        QueryPerformanceFrequency(&frequency);
        stats.wait_ms += double(wait_end.QuadPart - wait_start.QuadPart) * 1000.0 / double(frequency.QuadPart);
        if (r != VK_SUCCESS) {
            // The device is lost; the work may still be pending, so the pool
            // is not reset, but nothing will write the rings any more.
            error = vk_error("vkWaitForFences", r);
            abandon_batch(false);
            return false;
        }
        vkResetFences(device, 1, &fence);
        vkResetCommandPool(device, pool, 0);
        staging_head = 0;
        launch_head = 0;
        if (timing && queries && timed) {
            std::vector<std::uint64_t> values(timed * 2);
            vkGetQueryPoolResults(device, queries, 0, timed * 2, values.size() * 8, values.data(), 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            times.clear();
            for (std::uint32_t i = 0; i < timed; ++i) {
                times.push_back(static_cast<double>(values[2 * i + 1] - values[2 * i]) *
                                timestamp_period * 1e-6);
                KernelTime &k = profile[timed_labels[i] ? timed_labels[i] : "?"];
                ++k.count;
                k.total_ms += times.back();
            }
        }
        return true;
    }

    // Reserves host-visible staging space, flushing the batch if it is full.
    bool staging_alloc(VkDeviceSize bytes, VkDeviceSize &offset, std::string &error)
    {
        bytes = (bytes + 15) & ~VkDeviceSize(15);
        if (staging_head + bytes > staging.size) {
            if (!submit_and_wait(error)) return false;
            if (bytes > staging.size) {
                destroy(staging);
                VkDeviceSize size = kStagingInitialBytes;
                while (size < bytes) size *= 2;
                if (!create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   0, staging, error))
                    return false;
            }
        }
        offset = staging_head;
        staging_head += bytes;
        return true;
    }

    // Shared memory is a specialization constant, so every distinct size is
    // a separate pipeline. Applications such as PhysX cloth vary the dynamic
    // size from frame to frame; rounding up to power-of-two buckets bounds
    // the number of pipelines per kernel (extra shared memory is harmless).
    std::uint32_t shared_words(const Shader &shader, std::uint32_t bytes) const
    {
        bytes = std::max<std::uint32_t>(bytes, 4);
        std::uint32_t bucket = 1024;
        while (bucket < bytes) bucket *= 2;
        if (!shader.shared_in_storage && bucket > caps.max_shared_bytes)
            bucket = std::max<std::uint32_t>(bytes, std::min<std::uint32_t>(caps.max_shared_bytes, (bytes + 4095) & ~4095u));
        return bucket / 4;
    }

    void print_pipeline_stats(VkPipeline pipeline, const DispatchShape &shape)
    {
        VkPipelineInfoKHR pipeline_info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
        pipeline_info.pipeline = pipeline;
        std::uint32_t count = 0;
        vkGetPipelineExecutablePropertiesKHR(device, &pipeline_info, &count, nullptr);
        std::vector<VkPipelineExecutablePropertiesKHR> executables(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
        vkGetPipelineExecutablePropertiesKHR(device, &pipeline_info, &count, executables.data());
        for (std::uint32_t i = 0; i < count; ++i) {
            std::fprintf(stderr, "pipeline stats (block %ux%ux%u) %s:", shape.block[0], shape.block[1], shape.block[2],
                         executables[i].name);
            VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
            executable.pipeline = pipeline;
            executable.executableIndex = i;
            std::uint32_t n = 0;
            vkGetPipelineExecutableStatisticsKHR(device, &executable, &n, nullptr);
            std::vector<VkPipelineExecutableStatisticKHR> list(n, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
            vkGetPipelineExecutableStatisticsKHR(device, &executable, &n, list.data());
            for (const auto &stat : list) {
                switch (stat.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: std::fprintf(stderr, " %s=%u", stat.name, stat.value.b32); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: std::fprintf(stderr, " %s=%lld", stat.name, static_cast<long long>(stat.value.i64)); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: std::fprintf(stderr, " %s=%llu", stat.name, static_cast<unsigned long long>(stat.value.u64)); break;
                default: std::fprintf(stderr, " %s=%g", stat.name, stat.value.f64); break;
                }
            }
            std::fprintf(stderr, "\n");
        }
    }

    VkPipeline pipeline(Shader &shader, const DispatchShape &shape, std::string &error)
    {
        const std::uint32_t words = shared_words(shader, shape.shared_bytes);
        std::vector<std::uint32_t> key{shape.block[0], shape.block[1], shape.block[2], words, heap_base};
        auto found = shader.pipelines.find(key);
        if (found != shader.pipelines.end()) return found->second;

        std::vector<VkSpecializationMapEntry> entries;
        std::vector<std::uint32_t> data;
        auto add = [&](std::uint32_t id, std::uint32_t value) {
            entries.push_back({id, static_cast<std::uint32_t>(data.size() * 4), 4});
            data.push_back(value);
        };
        add(0, shape.block[0]);
        add(1, shape.block[1]);
        add(2, shape.block[2]);
        add(3, words);
        add(kSpecHeapBase, heap_base);
        VkSpecializationInfo spec{};
        spec.mapEntryCount = static_cast<std::uint32_t>(entries.size());
        spec.pMapEntries = entries.data();
        spec.dataSize = data.size() * 4;
        spec.pData = data.data();

        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = shader.module;
        info.stage.pName = "main";
        info.stage.pSpecializationInfo = &spec;
        info.layout = pipeline_layout;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
        if (shader.warp_synchronous && caps.subgroup_control) {
            // Warp-synchronous PTX needs 32-lane subgroups that are fully
            // populated, exactly like a CUDA warp.
            const std::uint64_t threads = std::uint64_t{shape.block[0]} * shape.block[1] * shape.block[2];
            if (threads > 32ull * caps.max_workgroup_subgroups) {
                error = "block of " + std::to_string(threads) + " threads exceeds the device's " +
                        std::to_string(caps.max_workgroup_subgroups) + " subgroups per workgroup";
                return VK_NULL_HANDLE;
            }
            subgroup.requiredSubgroupSize = 32;
            info.stage.pNext = &subgroup;
            if (shape.block[0] % 32 == 0)
                info.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
        }
        if (pipeline_stats) info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        VkPipeline result = VK_NULL_HANDLE;
        LARGE_INTEGER create_start, create_end, frequency;
        QueryPerformanceCounter(&create_start);
        const VkResult r = vkCreateComputePipelines(device, pipeline_cache, 1, &info, nullptr, &result);
        QueryPerformanceCounter(&create_end);
        QueryPerformanceFrequency(&frequency);
        stats.pipeline_ms += double(create_end.QuadPart - create_start.QuadPart) * 1000.0 / double(frequency.QuadPart);
        if (r != VK_SUCCESS) {
            error = vk_error("vkCreateComputePipelines", r);
            return VK_NULL_HANDLE;
        }
        ++stats.pipelines;
        if (pipeline_stats) print_pipeline_stats(result, shape);
        shader.pipelines.emplace(std::move(key), result);
        return result;
    }
};

Device::Device() : impl_(new Impl) {}
Device::~Device() { shutdown(); }

std::vector<std::string> Device::enumerate(std::string &error)
{
    std::vector<std::string> out;
    if (volkInitialize() != VK_SUCCESS) {
        error = "Vulkan loader is unavailable";
        return out;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {
        error = "vkCreateInstance failed";
        return out;
    }
    volkLoadInstanceOnly(instance);
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        out.push_back(std::to_string(i) + ": " + p.deviceName);
    }
    vkDestroyInstance(instance, nullptr);
    return out;
}

bool Device::initialize(const std::string &selector, std::string &error)
{
    if (selector.empty()) return initialize(-1, error);
    std::string enumerate_error;
    const std::vector<std::string> names = enumerate(enumerate_error);
    // A selector that matches nothing is reported with the devices that exist.
    auto fail = [&](const std::string &message) {
        error = message + "; available devices:";
        for (const std::string &name : names) error += " \"" + name + "\""; // "index: name"
        if (names.empty()) error += " none";
        return false;
    };
    bool numeric = true;
    for (char c : selector) numeric = numeric && c >= '0' && c <= '9';
    if (numeric) {
        if (std::strtoul(selector.c_str(), nullptr, 10) >= names.size())
            return fail("PXVK_DEVICE index " + selector + " is out of range");
        return initialize(std::atoi(selector.c_str()), error);
    }
    auto lower = [](std::string text) {
        for (char &c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return text;
    };
    for (std::size_t i = 0; i < names.size(); ++i)
        if (lower(names[i]).find(lower(selector)) != std::string::npos) return initialize(static_cast<int>(i), error);
    return fail("no Vulkan device matches PXVK_DEVICE=\"" + selector + "\"");
}

bool Device::initialize(int index, std::string &error)
{
    Impl &d = *impl_;
    if (d.device) return true;
    if (volkInitialize() != VK_SUCCESS) {
        error = "Vulkan loader is unavailable";
        return false;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "pxvk";
    app.pEngineName = "pxvk";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VkResult r = vkCreateInstance(&instance_info, nullptr, &d.instance);
    if (r != VK_SUCCESS) {
        error = vk_error("vkCreateInstance", r);
        return false;
    }
    volkLoadInstance(d.instance);

    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(d.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(d.instance, &count, devices.data());
    if (count == 0) {
        error = "no Vulkan devices";
        return false;
    }
    if (index >= static_cast<int>(count)) {
        error = "Vulkan device index out of range";
        return false;
    }
    if (index < 0) {
        index = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devices[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                index = static_cast<int>(i);
                break;
            }
        }
    }
    d.physical = devices[static_cast<std::size_t>(index)];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d.physical, &props);
    if (props.apiVersion < VK_API_VERSION_1_1) {
        error = "pxvk requires Vulkan 1.1";
        return false;
    }
    d.limits = props.limits;
    d.timestamp_period = props.limits.timestampPeriod;
    Capabilities &c = d.caps;
    c.name = props.deviceName;
    c.vendor_id = props.vendorID;
    c.api_version = props.apiVersion;
    c.discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    c.max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    for (int i = 0; i < 3; ++i) {
        c.max_workgroup_size[i] = props.limits.maxComputeWorkGroupSize[i];
        c.max_workgroup_count[i] = props.limits.maxComputeWorkGroupCount[i];
    }
    c.max_shared_bytes = props.limits.maxComputeSharedMemorySize;
    c.max_heap_bytes = props.limits.maxStorageBufferRange;
    {
        static const char digits[] = "0123456789abcdef";
        for (unsigned char byte : props.pipelineCacheUUID) {
            c.pipeline_cache_uuid.push_back(digits[byte >> 4]);
            c.pipeline_cache_uuid.push_back(digits[byte & 15]);
        }
    }

    std::uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &extension_count, extensions.data());
    auto has_extension = [&](const char *name) {
        for (const auto &e : extensions)
            if (std::strcmp(e.extensionName, name) == 0) return true;
        return false;
    };
    const bool api12 = props.apiVersion >= VK_API_VERSION_1_2;
    const bool api13 = props.apiVersion >= VK_API_VERSION_1_3;
    const bool float_controls = api12 || has_extension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
    const bool subgroup_ext = api13 || has_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    VkPhysicalDeviceFloatControlsProperties float_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup_basic{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceMaintenance3Properties maintenance3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &subgroup_basic;
    subgroup_basic.pNext = &maintenance3;
    void **tail = &maintenance3.pNext;
    if (float_controls) {
        *tail = &float_props;
        tail = &float_props.pNext;
    }
    if (subgroup_ext) {
        *tail = &subgroup_props;
        tail = &subgroup_props.pNext;
    }
    vkGetPhysicalDeviceProperties2(d.physical, &props2);
    c.subgroup_size = subgroup_basic.subgroupSize;
    c.min_subgroup_size = subgroup_ext ? subgroup_props.minSubgroupSize : subgroup_basic.subgroupSize;
    c.native_ftz = float_controls && float_props.shaderDenormFlushToZeroFloat32;
    c.preserve_nan = float_controls && float_props.shaderSignedZeroInfNanPreserveFloat32;

    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures indexing_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    const bool indexing_ext = api12 || has_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void **feature_tail = &features2.pNext;
    if (subgroup_ext) {
        *feature_tail = &subgroup_features;
        feature_tail = &subgroup_features.pNext;
    }
    if (indexing_ext) {
        *feature_tail = &indexing_features;
        feature_tail = &indexing_features.pNext;
    }
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    const bool atomic_float_ext = has_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    if (atomic_float_ext) {
        *feature_tail = &atomic_float_features;
        feature_tail = &atomic_float_features.pNext;
    }
    VkPhysicalDeviceShaderClockFeaturesKHR clock_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
    const bool clock_ext = has_extension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    if (clock_ext) *feature_tail = &clock_features;
    vkGetPhysicalDeviceFeatures2(d.physical, &features2);
    c.float_atomic_add = atomic_float_ext && atomic_float_features.shaderBufferFloat32AtomicAdd &&
                         !env_switch("PXVK_NO_FLOAT_ATOMICS");
    c.shader_clock = clock_ext && clock_features.shaderSubgroupClock;
    c.int64 = features2.features.shaderInt64;
    c.max_workgroup_subgroups = subgroup_ext ? subgroup_props.maxComputeWorkgroupSubgroups : 0;
    c.subgroup_control = subgroup_ext && subgroup_features.subgroupSizeControl &&
                         subgroup_features.computeFullSubgroups &&
                         (subgroup_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                         subgroup_props.minSubgroupSize <= 32 && subgroup_props.maxSubgroupSize >= 32;

    vkGetPhysicalDeviceMemoryProperties(d.physical, &d.memory_properties);
    for (std::uint32_t i = 0; i < d.memory_properties.memoryHeapCount; ++i)
        if (d.memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            c.device_local_bytes = std::max<std::uint64_t>(c.device_local_bytes, d.memory_properties.memoryHeaps[i].size);

    // Split heap only where a single storage-buffer descriptor cannot cover
    // what PhysX applications allocate (FluidMark alone asks for a 256 MB
    // GPU heap): a window per 2^shift bytes, selected per access.
    constexpr std::uint64_t kSplitBelow = 512ull << 20;
    const bool can_split = indexing_ext && indexing_features.shaderStorageBufferArrayNonUniformIndexing;
    if (props.limits.maxStorageBufferRange < kSplitBelow && props.limits.maxStorageBufferRange >= (1u << 20) &&
        can_split) {
        std::uint32_t shift = 20; // windows of at least 1 MiB
        while (shift < 30 && (2ull << shift) <= props.limits.maxStorageBufferRange) ++shift;
        std::uint64_t wanted = std::min<std::uint64_t>(c.device_local_bytes * 3 / 4, 0xF0000000u);
        wanted = std::min<std::uint64_t>(wanted, maintenance3.maxMemoryAllocationSize);
        // Bindings 1, 2 and 4, plus the out-of-range window, also count.
        const std::uint64_t storage_slots = props.limits.maxPerStageDescriptorStorageBuffers > 4
                                                ? props.limits.maxPerStageDescriptorStorageBuffers - 4
                                                : 1;
        const std::uint64_t windows =
            std::min<std::uint64_t>({(wanted + (1ull << shift) - 1) >> shift, storage_slots, std::uint64_t{32}});
        if (windows >= 2) {
            c.heap_window_shift = shift;
            c.heap_windows = static_cast<std::uint32_t>(windows);
            c.max_heap_bytes = windows << shift;
        }
    }
    // One allocation, leaving headroom for the application's own rendering;
    // 32-bit device pointers end at 4 GiB, minus space for the guard region.
    c.max_heap_bytes = std::min<std::uint64_t>({c.max_heap_bytes, c.device_local_bytes * 3 / 4, 0xF0000000u});
    if (maintenance3.maxMemoryAllocationSize)
        c.max_heap_bytes = std::min<std::uint64_t>(c.max_heap_bytes, maintenance3.maxMemoryAllocationSize);
    c.max_heap_bytes &= ~std::uint64_t{3}; // vkCmdFillBuffer sizes are multiples of 4

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &family_count, families.data());
    bool found_family = false;
    for (std::uint32_t i = 0; i < family_count && !found_family; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            d.family = i;
            d.timestamp_valid_bits = families[i].timestampValidBits;
            found_family = true;
        }
    }
    if (!found_family) {
        error = "no compute queue";
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = d.family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    enable.features.robustBufferAccess = VK_TRUE;
    enable.features.shaderInt64 = c.int64 ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceSubgroupSizeControlFeatures enable_subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    std::vector<const char *> enabled_extensions;
    if (c.subgroup_control) {
        enable_subgroup.subgroupSizeControl = VK_TRUE;
        enable_subgroup.computeFullSubgroups = VK_TRUE;
        enable.pNext = &enable_subgroup;
        if (!api13) enabled_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
    if (float_controls && !api12) enabled_extensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
    VkPhysicalDeviceDescriptorIndexingFeatures enable_indexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    if (c.heap_windows != 0) {
        enable_indexing.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
        enable_indexing.pNext = enable.pNext;
        enable.pNext = &enable_indexing;
        if (!api12) enabled_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    }
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enable_atomic_float{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    if (c.float_atomic_add) {
        enable_atomic_float.shaderBufferFloat32AtomicAdd = VK_TRUE;
        enable_atomic_float.pNext = enable.pNext;
        enable.pNext = &enable_atomic_float;
        enabled_extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    VkPhysicalDeviceShaderClockFeaturesKHR enable_clock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
    if (c.shader_clock) {
        enable_clock.shaderSubgroupClock = VK_TRUE;
        enable_clock.pNext = enable.pNext;
        enable.pNext = &enable_clock;
        enabled_extensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    }
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR enable_executable_info{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    const bool want_stats = env_switch("PXVK_PIPELINE_STATS");
    if (want_stats && has_extension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
        d.pipeline_stats = true;
        enable_executable_info.pipelineExecutableInfo = VK_TRUE;
        enable_executable_info.pNext = enable.pNext;
        enable.pNext = &enable_executable_info;
        enabled_extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    }
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &enable;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<std::uint32_t>(enabled_extensions.size());
    device_info.ppEnabledExtensionNames = enabled_extensions.data();
    r = vkCreateDevice(d.physical, &device_info, nullptr, &d.device);
    if (r != VK_SUCCESS) {
        error = vk_error("vkCreateDevice", r);
        return false;
    }
    volkLoadDevice(d.device);
    vkGetDeviceQueue(d.device, d.family, 0, &d.queue);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = d.family;
    vkCreateCommandPool(d.device, &pool_info, nullptr, &d.pool);
    VkCommandBufferAllocateInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = d.pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    vkAllocateCommandBuffers(d.device, &cmd_info, &d.cmd);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(d.device, &fence_info, nullptr, &d.fence);

    VkDescriptorSetLayoutBinding bindings[5]{};
    for (std::uint32_t i = 0; i < 5; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType =
            i == 3 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = i == 0 && c.heap_windows != 0 ? c.heap_windows + 1 : 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 5;
    layout_info.pBindings = bindings;
    vkCreateDescriptorSetLayout(d.device, &layout_info, nullptr, &d.set_layout);
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &d.set_layout;
    vkCreatePipelineLayout(d.device, &pipeline_layout_info, nullptr, &d.pipeline_layout);
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 + c.heap_windows},
                                     {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1}};
    VkDescriptorPoolCreateInfo descriptor_pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 2;
    descriptor_pool_info.pPoolSizes = sizes;
    vkCreateDescriptorPool(d.device, &descriptor_pool_info, nullptr, &d.descriptor_pool);
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = d.descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &d.set_layout;
    vkAllocateDescriptorSets(d.device, &set_info, &d.set);
    VkPipelineCacheCreateInfo cache_info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    vkCreatePipelineCache(d.device, &cache_info, nullptr, &d.pipeline_cache);

    if (!d.create_buffer(kStagingInitialBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
                         d.staging, error))
        return false;
    // Launch blocks prefer device-local host-visible (BAR) memory.
    if (!d.create_buffer(kLaunchRingBytes + kLaunchRange, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d.launch, error))
        return false;
    if (!d.create_buffer(64u << 10, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                         d.scratch, error))
        return false;
    if (c.heap_windows != 0 &&
        !d.create_buffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, d.dummy, error))
        return false;
    return true;
}

void Device::shutdown()
{
    Impl &d = *impl_;
    if (!d.device) {
        if (d.instance) vkDestroyInstance(d.instance, nullptr);
        d.instance = VK_NULL_HANDLE;
        return;
    }
    std::string ignored;
    d.submit_and_wait(ignored);
    vkDeviceWaitIdle(d.device);
    for (auto &shader : d.shaders) {
        for (auto &p : shader->pipelines) vkDestroyPipeline(d.device, p.second, nullptr);
        vkDestroyShaderModule(d.device, shader->module, nullptr);
    }
    d.shaders.clear();
    d.destroy(d.heap);
    d.destroy(d.staging);
    d.destroy(d.readback);
    d.destroy(d.launch);
    d.destroy(d.scratch);
    d.destroy(d.dummy);
    if (d.queries) vkDestroyQueryPool(d.device, d.queries, nullptr);
    vkDestroyPipelineCache(d.device, d.pipeline_cache, nullptr);
    vkDestroyDescriptorPool(d.device, d.descriptor_pool, nullptr);
    vkDestroyPipelineLayout(d.device, d.pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(d.device, d.set_layout, nullptr);
    vkDestroyFence(d.device, d.fence, nullptr);
    vkDestroyCommandPool(d.device, d.pool, nullptr);
    vkDestroyDevice(d.device, nullptr);
    vkDestroyInstance(d.instance, nullptr);
    d.device = VK_NULL_HANDLE;
    d.instance = VK_NULL_HANDLE;
}

const Capabilities &Device::caps() const { return impl_->caps; }
std::uint64_t Device::heap_size() const { return impl_->heap.size; }
void Device::set_heap_base(std::uint32_t base) { impl_->heap_base = base; }
std::uint32_t Device::heap_base() const { return impl_->heap_base; }
const Stats &Device::stats() const { return impl_->stats; }

bool Device::ensure_heap(std::uint64_t bytes, std::string &error)
{
    Impl &d = *impl_;
    if (bytes <= d.heap.size) return true;
    if (bytes > d.caps.max_heap_bytes) {
        error = "CUDA heap exceeds the device's heap limit";
        return false;
    }
    // A Vulkan buffer cannot be resized: growing means a new buffer and a
    // copy. PhysX allocates one pool per context, so the heap grows once at
    // startup; size it to what is needed (in 64 KiB driver pages) rather
    // than doubling, which left half of FluidMark's 512 MiB heap unused.
    // Growth by small steps still adds at least a quarter, so an
    // application allocating piecemeal does not copy on every allocation.
    constexpr VkDeviceSize kPage = 64u << 10;
    const auto round = [&](VkDeviceSize n) {
        return std::min<VkDeviceSize>((n + kPage - 1) & ~(kPage - 1), d.caps.max_heap_bytes);
    };
    const VkDeviceSize needed = round(bytes);
    VkDeviceSize size = std::max(needed, round(d.heap.size + d.heap.size / 4));
    if (!d.submit_and_wait(error)) return false;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    Buffer grown;
    if (!d.create_buffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, grown, error)) {
        // The old heap stays allocated during the copy; if the extra quarter
        // does not fit beside it, try exactly what is needed.
        if (needed >= size) return false;
        size = needed;
        if (!d.create_buffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, grown, error)) return false;
    }
    d.begin();
    if (d.heap.buffer) {
        VkBufferCopy region{0, 0, d.heap.size};
        vkCmdCopyBuffer(d.cmd, d.heap.buffer, grown.buffer, 1, &region);
        vkCmdFillBuffer(d.cmd, grown.buffer, d.heap.size, VK_WHOLE_SIZE, 0);
    } else {
        vkCmdFillBuffer(d.cmd, grown.buffer, 0, VK_WHOLE_SIZE, 0);
    }
    d.commands = 1;
    if (!d.submit_and_wait(error)) {
        d.destroy(grown);
        return false;
    }
    d.destroy(d.heap);
    d.heap = grown;
    d.update_descriptors();
    return true;
}

bool Device::upload(std::uint64_t offset, const void *data, std::size_t bytes, std::string &error)
{
    Impl &d = *impl_;
    if (offset + bytes > d.heap.size) {
        error = "upload outside the heap";
        return false;
    }
    const auto *source = static_cast<const unsigned char *>(data);
    while (bytes != 0) {
        const std::size_t chunk = std::min<std::size_t>(bytes, d.staging.size / 2);
        VkDeviceSize staged = 0;
        if (!d.staging_alloc(chunk, staged, error)) return false;
        std::memcpy(static_cast<unsigned char *>(d.staging.mapped) + staged, source, chunk);
        d.barrier();
        VkBufferCopy region{staged, offset, chunk};
        vkCmdCopyBuffer(d.cmd, d.staging.buffer, d.heap.buffer, 1, &region);
        ++d.stats.uploads;
        d.stats.upload_bytes += chunk;
        source += chunk;
        offset += chunk;
        bytes -= chunk;
    }
    return true;
}

bool Device::fill(std::uint64_t offset, std::uint8_t value, std::size_t bytes, std::string &error)
{
    Impl &d = *impl_;
    if (offset + bytes > d.heap.size) {
        error = "fill outside the heap";
        return false;
    }
    // vkCmdFillBuffer needs 4-byte alignment; unaligned edges go through
    // staging.
    const std::uint64_t aligned_start = (offset + 3) & ~std::uint64_t(3);
    const std::uint64_t end = offset + bytes;
    const std::uint64_t aligned_end = end & ~std::uint64_t(3);
    if (aligned_start >= aligned_end) {
        std::vector<std::uint8_t> pattern(bytes, value);
        return upload(offset, pattern.data(), bytes, error);
    }
    if (aligned_start > offset) {
        std::uint8_t head[4] = {value, value, value, value};
        if (!upload(offset, head, static_cast<std::size_t>(aligned_start - offset), error)) return false;
    }
    d.barrier();
    const std::uint32_t word = value * 0x01010101u;
    vkCmdFillBuffer(d.cmd, d.heap.buffer, aligned_start, aligned_end - aligned_start, word);
    if (end > aligned_end) {
        std::uint8_t tail[4] = {value, value, value, value};
        if (!upload(aligned_end, tail, static_cast<std::size_t>(end - aligned_end), error)) return false;
    }
    return true;
}

bool Device::copy(std::uint64_t dst, std::uint64_t src, std::size_t bytes, std::string &error)
{
    Impl &d = *impl_;
    if (dst + bytes > d.heap.size || src + bytes > d.heap.size) {
        error = "copy outside the heap";
        return false;
    }
    if (bytes == 0) return true;
    const bool overlap = dst < src + bytes && src < dst + bytes;
    if (overlap) {
        // Overlapping copies bounce through host memory to keep memmove semantics.
        std::vector<std::uint8_t> temp(bytes);
        return download(temp.data(), src, bytes, error) && upload(dst, temp.data(), bytes, error);
    }
    d.barrier();
    VkBufferCopy region{src, dst, bytes};
    vkCmdCopyBuffer(d.cmd, d.heap.buffer, d.heap.buffer, 1, &region);
    return true;
}

bool Device::download(void *data, std::uint64_t offset, std::size_t bytes, std::string &error)
{
    Impl &d = *impl_;
    if (offset + bytes > d.heap.size) {
        error = "download outside the heap";
        return false;
    }
    if (bytes == 0) return true;
    if (d.readback.size < bytes) {
        if (!d.submit_and_wait(error)) return false;
        d.destroy(d.readback);
        VkDeviceSize size = 1u << 20;
        while (size < bytes) size *= 2;
        if (!d.create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             VK_MEMORY_PROPERTY_HOST_CACHED_BIT, d.readback, error))
            return false;
    }
    d.barrier();
    VkBufferCopy region{offset, 0, bytes};
    vkCmdCopyBuffer(d.cmd, d.heap.buffer, d.readback.buffer, 1, &region);
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
    if (!d.submit_and_wait(error)) return false;
    std::memcpy(data, d.readback.mapped, bytes);
    ++d.stats.downloads;
    d.stats.download_bytes += bytes;
    return true;
}

HostBuffer *Device::allocate_host(std::size_t bytes, void **mapped, std::string &error)
{
    Impl &d = *impl_;
    auto host = std::make_unique<HostBuffer>();
    if (!d.create_buffer(std::max<std::size_t>(bytes, 16),
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT, host->buffer, error))
        return nullptr;
    *mapped = host->buffer.mapped;
    return host.release();
}

void Device::free_host(HostBuffer *buffer)
{
    if (buffer == nullptr) return;
    std::string ignored;
    impl_->submit_and_wait(ignored);
    impl_->destroy(buffer->buffer);
    delete buffer;
}

bool Device::copy_from_host(HostBuffer *buffer, std::uint64_t source, std::uint64_t heap_offset, std::size_t bytes,
                            std::string &error)
{
    Impl &d = *impl_;
    if (source + bytes > buffer->buffer.size || heap_offset + bytes > d.heap.size) {
        error = "pinned copy outside its buffers";
        return false;
    }
    if (bytes == 0) return true;
    d.barrier();
    VkBufferCopy region{source, heap_offset, bytes};
    vkCmdCopyBuffer(d.cmd, buffer->buffer.buffer, d.heap.buffer, 1, &region);
    ++d.stats.uploads;
    d.stats.upload_bytes += bytes;
    return true;
}

bool Device::copy_to_host(HostBuffer *buffer, std::uint64_t destination, std::uint64_t heap_offset, std::size_t bytes,
                          std::string &error)
{
    Impl &d = *impl_;
    if (destination + bytes > buffer->buffer.size || heap_offset + bytes > d.heap.size) {
        error = "pinned copy outside its buffers";
        return false;
    }
    if (bytes == 0) return true;
    d.barrier();
    VkBufferCopy region{heap_offset, destination, bytes};
    vkCmdCopyBuffer(d.cmd, d.heap.buffer, buffer->buffer.buffer, 1, &region);
    ++d.stats.downloads;
    d.stats.download_bytes += bytes;
    return true;
}

bool Device::busy() const { return impl_->recording; }

Shader *Device::create_shader(const std::vector<std::uint32_t> &spirv, std::uint32_t launch_block_bytes,
                              bool warp_synchronous, bool shared_in_storage, std::string &error)
{
    Impl &d = *impl_;
    if (launch_block_bytes > kLaunchRange) {
        error = "kernel launch block exceeds 4096 bytes";
        return nullptr;
    }
    // Warp-synchronous PTX relies on 32 lanes running in lockstep. Wider
    // subgroups (64/128 lanes) contain whole warps; narrower ones that cannot
    // be forced to 32 would silently compute wrong results.
    // (Kernels compiled with CompileOptions::emulate_warps do not report
    // warp_synchronous.)
    if (warp_synchronous && d.caps.narrow_subgroups()) {
        error = "warp-synchronous kernel needs 32-lane subgroups, device may use " +
                std::to_string(d.caps.min_subgroup_size);
        return nullptr;
    }
    auto shader = std::make_unique<Shader>();
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = spirv.size() * 4;
    info.pCode = spirv.data();
    const VkResult r = vkCreateShaderModule(d.device, &info, nullptr, &shader->module);
    if (r != VK_SUCCESS) {
        error = vk_error("vkCreateShaderModule", r);
        return nullptr;
    }
    shader->launch_block_bytes = launch_block_bytes;
    shader->warp_synchronous = warp_synchronous;
    shader->shared_in_storage = shared_in_storage;
    d.shaders.push_back(std::move(shader));
    return d.shaders.back().get();
}

bool Device::dispatch(Shader *shader, const DispatchShape &shape, const void *launch_block,
                      std::size_t launch_block_bytes, std::string &error, const char *label)
{
    Impl &d = *impl_;
    if (!d.heap.buffer) {
        error = "dispatch before any device memory exists";
        return false;
    }
    const std::uint64_t threads = std::uint64_t(shape.block[0]) * shape.block[1] * shape.block[2];
    for (int i = 0; i < 3; ++i) {
        if (shape.block[i] == 0 || shape.block[i] > d.caps.max_workgroup_size[i] || shape.grid[i] == 0 ||
            shape.grid[i] > d.caps.max_workgroup_count[i]) {
            error = "launch dimensions exceed device limits";
            return false;
        }
    }
    if (threads > d.caps.max_workgroup_invocations) {
        error = "block size exceeds maxComputeWorkGroupInvocations";
        return false;
    }
    if (shader->shared_in_storage) {
        const std::uint64_t groups = std::uint64_t(shape.grid[0]) * shape.grid[1] * shape.grid[2];
        const std::uint64_t needed = groups * d.shared_words(*shader, shape.shared_bytes) * 4;
        if (needed > d.scratch.size) {
            VkDeviceSize size = std::max<VkDeviceSize>(d.scratch.size, 64u << 10);
            while (size < needed) size *= 2;
            if (size > d.limits.maxStorageBufferRange) {
                error = "storage-backed shared memory exceeds maxStorageBufferRange";
                return false;
            }
            if (!d.submit_and_wait(error)) return false;
            // Create the new buffer first: on failure the old one stays valid.
            Buffer grown;
            if (!d.create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                 grown, error))
                return false;
            d.destroy(d.scratch);
            d.scratch = grown;
            d.update_descriptors();
        }
    } else if (shape.shared_bytes > d.caps.max_shared_bytes) {
        error = "shared memory exceeds maxComputeSharedMemorySize";
        return false;
    }
    if (launch_block_bytes > kLaunchRange) {
        error = "launch block too large";
        return false;
    }
    const VkPipeline pipeline = d.pipeline(*shader, shape, error);
    if (!pipeline) return false;

    const VkDeviceSize align = std::max<VkDeviceSize>(d.limits.minUniformBufferOffsetAlignment, 16);
    VkDeviceSize offset = (d.launch_head + align - 1) / align * align;
    if (offset + kLaunchRange > d.launch.size) {
        if (!d.submit_and_wait(error)) return false;
        offset = 0;
    }
    d.launch_head = offset + std::max<std::size_t>(launch_block_bytes, 16);
    std::memcpy(static_cast<unsigned char *>(d.launch.mapped) + offset, launch_block, launch_block_bytes);

    d.barrier();
    if (d.timing && !d.queries && d.timestamp_valid_bits) {
        VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        q.queryType = VK_QUERY_TYPE_TIMESTAMP;
        q.queryCount = kMaxTimedDispatches * 2;
        vkCreateQueryPool(d.device, &q, nullptr, &d.queries);
        vkCmdResetQueryPool(d.cmd, d.queries, 0, kMaxTimedDispatches * 2);
    }
    const bool time_this = d.timing && d.queries && d.timed < kMaxTimedDispatches;
    if (time_this)
        vkCmdWriteTimestamp(d.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, d.queries, d.timed * 2);
    vkCmdBindPipeline(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    const std::uint32_t dynamic_offset = static_cast<std::uint32_t>(offset);
    vkCmdBindDescriptorSets(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline_layout, 0, 1, &d.set, 1,
                            &dynamic_offset);
    vkCmdDispatch(d.cmd, shape.grid[0], shape.grid[1], shape.grid[2]);
    if (time_this) {
        vkCmdWriteTimestamp(d.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, d.queries, d.timed * 2 + 1);
        d.timed_labels.push_back(label);
        ++d.timed;
    }
    ++d.stats.dispatches;
    return true;
}

bool Device::synchronize(std::string &error) { return impl_->submit_and_wait(error); }

void Device::load_pipeline_cache(const std::string &path)
{
    Impl &d = *impl_;
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) return;
    std::vector<char> data;
    char chunk[65536];
    for (std::size_t n; (n = std::fread(chunk, 1, sizeof(chunk), file)) > 0;) data.insert(data.end(), chunk, chunk + n);
    std::fclose(file);
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = data.size();
    info.pInitialData = data.data();
    VkPipelineCache cache = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(d.device, &info, nullptr, &cache) == VK_SUCCESS) {
        vkDestroyPipelineCache(d.device, d.pipeline_cache, nullptr);
        d.pipeline_cache = cache;
    }
}

bool Device::save_pipeline_cache(const std::string &path)
{
    Impl &d = *impl_;
    std::size_t size = 0;
    if (vkGetPipelineCacheData(d.device, d.pipeline_cache, &size, nullptr) != VK_SUCCESS || size == 0) return false;
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(d.device, d.pipeline_cache, &size, data.data()) != VK_SUCCESS) return false;
    const std::string temporary = path + ".tmp" + std::to_string(GetCurrentProcessId());
    FILE *file = std::fopen(temporary.c_str(), "wb");
    if (!file) return false;
    const bool written = std::fwrite(data.data(), 1, size, file) == size;
    std::fclose(file);
    if (!written || !MoveFileExA(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileA(temporary.c_str());
        return false;
    }
    return true;
}

void Device::set_timing(bool enabled) { impl_->timing = enabled; }

const std::vector<double> &Device::last_dispatch_times() const { return impl_->times; }

std::vector<std::pair<std::string, KernelTime>> Device::take_profile()
{
    std::vector<std::pair<std::string, KernelTime>> out(impl_->profile.begin(), impl_->profile.end());
    impl_->profile.clear();
    return out;
}

} // namespace pxvk::vk
