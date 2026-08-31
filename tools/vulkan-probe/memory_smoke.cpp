#include <volk.h>
#include <spirv/unified1/spirv.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {

constexpr VkDeviceSize kBufferBytes = 32ull * 1024ull * 1024ull;
constexpr VkDeviceSize kMemoryCopyBytes = 2ull * sizeof(std::uint32_t);
constexpr VkDeviceSize kOracleCopyBytes = 304ull * sizeof(std::uint32_t);
constexpr std::uint32_t kInput = 0x12345678u;
constexpr std::uint32_t kMask = 0xa5a55a5au;
constexpr std::uint32_t kAddend = 0x10203040u;
constexpr std::uint32_t kExpected = (kInput ^ kMask) + kAddend;
constexpr std::size_t kMathOffset = 16;
constexpr std::size_t kBarrierOffset = 64;
constexpr std::size_t kTextureInputOffset = 160;
constexpr std::size_t kTextureOutputOffset = 256;
constexpr std::array<std::uint32_t, 14> kExpectedMath = {
    0x3eaaaaabu,
    0x00000000u,
    0x3eaaaaabu,
    0x3fb504f3u,
    0x00000000u,
    0x3f3504f3u,
    0xa8800000u,
    0x00000000u,
    0xfffffffdu,
    0x4b800000u,
    0x00000000u,
    0x00000000u,
    0x3f800000u,
    0x3f800000u,
};
constexpr std::array<const char *, 14> kMathNames = {
    "div_one_third",
    "div_min_normal_half_ftz",
    "rcp_three_approx",
    "sqrt_two_approx",
    "sqrt_min_subnormal_ftz",
    "rsqrt_two_approx",
    "fma_fused",
    "mul_add_separate",
    "cvt_rzi_neg3_75",
    "cvt_rn_u32_16777217",
    "add_min_subnormal_zero_ftz",
    "mul_min_subnormal_one_ftz",
    "min_nan_one",
    "max_nan_one",
};

void check(VkResult result, const char *operation)
{
    if (result != VK_SUCCESS) {
        std::ostringstream message;
        message << operation << " failed with VkResult " << static_cast<int>(result);
        throw std::runtime_error(message.str());
    }
}

std::string hex32(std::uint32_t value)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setfill('0') << std::setw(8) << value;
    return output.str();
}

std::string json_string(const char *value)
{
    std::ostringstream output;
    output << '"';
    for (const unsigned char c : std::string(value)) {
        switch (c) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (c < 0x20) {
                output << "\\u" << std::hex << std::setfill('0') << std::setw(4)
                       << static_cast<unsigned int>(c) << std::dec;
            } else {
                output << static_cast<char>(c);
            }
            break;
        }
    }
    output << '"';
    return output.str();
}

class SpirvBuilder {
public:
    SpirvBuilder()
    {
        words_ = {
            SpvMagicNumber,
            0x00010500u,
            0u,
            20u,
            0u,
        };
    }

    void emit(SpvOp operation, std::initializer_list<std::uint32_t> operands)
    {
        words_.push_back(
            (static_cast<std::uint32_t>(operands.size() + 1u) << SpvWordCountShift) |
            static_cast<std::uint32_t>(operation));
        words_.insert(words_.end(), operands.begin(), operands.end());
    }

    void emit_entry_point(std::uint32_t function, std::uint32_t interface_id)
    {
        std::vector<std::uint32_t> operands = {
            static_cast<std::uint32_t>(SpvExecutionModelGLCompute),
            function,
        };
        append_string(operands, "main");
        operands.push_back(interface_id);
        words_.push_back(
            (static_cast<std::uint32_t>(operands.size() + 1u) << SpvWordCountShift) |
            static_cast<std::uint32_t>(SpvOpEntryPoint));
        words_.insert(words_.end(), operands.begin(), operands.end());
    }

    const std::vector<std::uint32_t> &words() const
    {
        return words_;
    }

private:
    static void append_string(std::vector<std::uint32_t> &output, const char *text)
    {
        const std::size_t bytes = std::strlen(text) + 1u;
        const std::size_t word_count = (bytes + 3u) / 4u;
        const std::size_t first_word = output.size();
        output.resize(first_word + word_count, 0u);
        std::memcpy(output.data() + first_word, text, bytes);
    }

    std::vector<std::uint32_t> words_;
};

std::vector<std::uint32_t> make_memory_shader()
{
    constexpr std::uint32_t id_void = 1;
    constexpr std::uint32_t id_u32 = 2;
    constexpr std::uint32_t id_runtime_array = 3;
    constexpr std::uint32_t id_buffer_struct = 4;
    constexpr std::uint32_t id_ptr_buffer_struct = 5;
    constexpr std::uint32_t id_buffer = 6;
    constexpr std::uint32_t id_ptr_u32 = 7;
    constexpr std::uint32_t id_function_type = 8;
    constexpr std::uint32_t id_zero = 9;
    constexpr std::uint32_t id_one = 10;
    constexpr std::uint32_t id_mask = 11;
    constexpr std::uint32_t id_addend = 12;
    constexpr std::uint32_t id_main = 13;
    constexpr std::uint32_t id_label = 14;
    constexpr std::uint32_t id_input_pointer = 15;
    constexpr std::uint32_t id_input = 16;
    constexpr std::uint32_t id_xor = 17;
    constexpr std::uint32_t id_result = 18;
    constexpr std::uint32_t id_output_pointer = 19;

    SpirvBuilder builder;
    builder.emit(SpvOpCapability, {SpvCapabilityShader});
    builder.emit(SpvOpMemoryModel, {SpvAddressingModelLogical, SpvMemoryModelGLSL450});
    builder.emit_entry_point(id_main, id_buffer);
    builder.emit(
        SpvOpExecutionMode,
        {id_main, SpvExecutionModeLocalSize, 1u, 1u, 1u});

    builder.emit(SpvOpDecorate, {id_runtime_array, SpvDecorationArrayStride, 4u});
    builder.emit(SpvOpMemberDecorate, {id_buffer_struct, 0u, SpvDecorationOffset, 0u});
    builder.emit(SpvOpDecorate, {id_buffer_struct, SpvDecorationBlock});
    builder.emit(SpvOpDecorate, {id_buffer, SpvDecorationDescriptorSet, 0u});
    builder.emit(SpvOpDecorate, {id_buffer, SpvDecorationBinding, 0u});

    builder.emit(SpvOpTypeVoid, {id_void});
    builder.emit(SpvOpTypeInt, {id_u32, 32u, 0u});
    builder.emit(SpvOpTypeRuntimeArray, {id_runtime_array, id_u32});
    builder.emit(SpvOpTypeStruct, {id_buffer_struct, id_runtime_array});
    builder.emit(
        SpvOpTypePointer,
        {id_ptr_buffer_struct, SpvStorageClassStorageBuffer, id_buffer_struct});
    builder.emit(
        SpvOpVariable,
        {id_ptr_buffer_struct, id_buffer, SpvStorageClassStorageBuffer});
    builder.emit(SpvOpTypePointer, {id_ptr_u32, SpvStorageClassStorageBuffer, id_u32});
    builder.emit(SpvOpTypeFunction, {id_function_type, id_void});
    builder.emit(SpvOpConstant, {id_u32, id_zero, 0u});
    builder.emit(SpvOpConstant, {id_u32, id_one, 1u});
    builder.emit(SpvOpConstant, {id_u32, id_mask, kMask});
    builder.emit(SpvOpConstant, {id_u32, id_addend, kAddend});

    builder.emit(SpvOpFunction, {id_void, id_main, 0u, id_function_type});
    builder.emit(SpvOpLabel, {id_label});
    builder.emit(
        SpvOpAccessChain,
        {id_ptr_u32, id_input_pointer, id_buffer, id_zero, id_zero});
    builder.emit(SpvOpLoad, {id_u32, id_input, id_input_pointer});
    builder.emit(SpvOpBitwiseXor, {id_u32, id_xor, id_input, id_mask});
    builder.emit(SpvOpIAdd, {id_u32, id_result, id_xor, id_addend});
    builder.emit(
        SpvOpAccessChain,
        {id_ptr_u32, id_output_pointer, id_buffer, id_zero, id_one});
    builder.emit(SpvOpStore, {id_output_pointer, id_result});
    builder.emit(SpvOpReturn, {});
    builder.emit(SpvOpFunctionEnd, {});
    return builder.words();
}

struct BufferAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocation_size = 0;
    bool host_coherent = false;
    bool device_local = false;
};

std::uint32_t select_memory_type(
    const VkPhysicalDeviceMemoryProperties &properties,
    std::uint32_t memory_type_bits,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred)
{
    for (int pass = 0; pass < 2; ++pass) {
        for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
            if ((memory_type_bits & (1u << index)) == 0u) {
                continue;
            }
            const VkMemoryPropertyFlags flags = properties.memoryTypes[index].propertyFlags;
            if ((flags & required) != required) {
                continue;
            }
            if (pass == 0 && (flags & preferred) != preferred) {
                continue;
            }
            return index;
        }
    }
    throw std::runtime_error("no compatible Vulkan memory type");
}

BufferAllocation create_buffer(
    VkPhysicalDevice physical_device,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred)
{
    BufferAllocation allocation;
    try {
        VkBufferCreateInfo buffer_info = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            nullptr,
            0,
            size,
            usage,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr,
        };
        check(vkCreateBuffer(device, &buffer_info, nullptr, &allocation.buffer), "vkCreateBuffer");

        VkMemoryRequirements memory_requirements = {};
        vkGetBufferMemoryRequirements(device, allocation.buffer, &memory_requirements);
        VkPhysicalDeviceMemoryProperties memory_properties = {};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
        const std::uint32_t memory_type = select_memory_type(
            memory_properties,
            memory_requirements.memoryTypeBits,
            required,
            preferred);
        const VkMemoryPropertyFlags flags =
            memory_properties.memoryTypes[memory_type].propertyFlags;

        VkMemoryAllocateInfo allocation_info = {
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            memory_requirements.size,
            memory_type,
        };
        check(
            vkAllocateMemory(device, &allocation_info, nullptr, &allocation.memory),
            "vkAllocateMemory");
        check(
            vkBindBufferMemory(device, allocation.buffer, allocation.memory, 0),
            "vkBindBufferMemory");
        allocation.allocation_size = memory_requirements.size;
        allocation.host_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        allocation.device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        return allocation;
    } catch (...) {
        if (allocation.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, allocation.buffer, nullptr);
        }
        if (allocation.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, allocation.memory, nullptr);
        }
        throw;
    }
}

struct DeviceResult {
    std::uint32_t index = 0;
    VkPhysicalDeviceProperties properties = {};
    bool pass = false;
    bool device_memory_local = false;
    bool staging_coherent = false;
    std::uint32_t input_after = 0;
    std::uint32_t result = 0;
    std::array<std::uint32_t, 14> math = {};
    std::array<std::uint32_t, 64> barrier = {};
    std::array<std::array<std::uint32_t, 4>, 12> texture = {};
    std::string error;
};

DeviceResult run_device(
    std::uint32_t device_index,
    VkPhysicalDevice physical_device,
    const std::vector<std::uint32_t> &shader,
    bool oracle_mode,
    bool require_memory_result)
{
    DeviceResult result;
    result.index = device_index;
    vkGetPhysicalDeviceProperties(physical_device, &result.properties);

    VkDevice device = VK_NULL_HANDLE;
    BufferAllocation device_buffer;
    BufferAllocation staging_buffer;
    void *mapped = nullptr;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    try {
        std::uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device,
            &queue_family_count,
            queue_families.data());
        std::uint32_t queue_family = UINT32_MAX;
        for (std::uint32_t index = 0; index < queue_family_count; ++index) {
            if ((queue_families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
                queue_family = index;
                break;
            }
        }
        if (queue_family == UINT32_MAX) {
            throw std::runtime_error("device has no compute queue");
        }

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info = {
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            nullptr,
            0,
            queue_family,
            1,
            &priority,
        };
        VkDeviceCreateInfo device_info = {
            VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            nullptr,
            0,
            1,
            &queue_info,
            0,
            nullptr,
            0,
            nullptr,
            nullptr,
        };
        check(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");
        volkLoadDevice(device);
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        device_buffer = create_buffer(
            physical_device,
            device,
            kBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            0,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        staging_buffer = create_buffer(
            physical_device,
            device,
            kBufferBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        result.device_memory_local = device_buffer.device_local;
        result.staging_coherent = staging_buffer.host_coherent;

        const VkDeviceSize copy_bytes = oracle_mode ? kOracleCopyBytes : kMemoryCopyBytes;
        check(
            vkMapMemory(device, staging_buffer.memory, 0, staging_buffer.allocation_size, 0, &mapped),
            "vkMapMemory");
        auto *words = static_cast<std::uint32_t *>(mapped);
        std::memset(words, 0, static_cast<std::size_t>(copy_bytes));
        words[0] = kInput;
        words[1] = 0xdeadbeefu;
        if (oracle_mode) {
            for (std::uint32_t row = 0; row < 8; ++row) {
                for (std::uint32_t component = 0; component < 4; ++component) {
                    words[kTextureInputOffset + row * 4u + component] =
                        0x10000000u * (component + 1u) + row * 0x100u + component;
                }
            }
        }
        if (!staging_buffer.host_coherent) {
            VkMappedMemoryRange range = {
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                nullptr,
                staging_buffer.memory,
                0,
                VK_WHOLE_SIZE,
            };
            check(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
        }

        VkDescriptorSetLayoutBinding binding = {
            0,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        };
        VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &binding,
        };
        check(
            vkCreateDescriptorSetLayout(
                device,
                &descriptor_layout_info,
                nullptr,
                &descriptor_layout),
            "vkCreateDescriptorSetLayout");

        VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            1,
            1,
            &pool_size,
        };
        check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo set_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptor_pool,
            1,
            &descriptor_layout,
        };
        check(vkAllocateDescriptorSets(device, &set_info, &descriptor_set), "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo descriptor_buffer_info = {
            device_buffer.buffer,
            0,
            kBufferBytes,
        };
        VkWriteDescriptorSet descriptor_write = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            &descriptor_buffer_info,
            nullptr,
        };
        vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, nullptr);

        VkPipelineLayoutCreateInfo pipeline_layout_info = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptor_layout,
            0,
            nullptr,
        };
        check(
            vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout),
            "vkCreatePipelineLayout");
        VkShaderModuleCreateInfo shader_info = {
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            nullptr,
            0,
            shader.size() * sizeof(std::uint32_t),
            shader.data(),
        };
        check(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");
        VkPipelineShaderStageCreateInfo stage_info = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            shader_module,
            "main",
            nullptr,
        };
        VkComputePipelineCreateInfo pipeline_info = {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            stage_info,
            pipeline_layout,
            VK_NULL_HANDLE,
            -1,
        };
        check(
            vkCreateComputePipelines(
                device,
                VK_NULL_HANDLE,
                1,
                &pipeline_info,
                nullptr,
                &pipeline),
            "vkCreateComputePipelines");

        VkCommandPoolCreateInfo command_pool_info = {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            0,
            queue_family,
        };
        check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo command_buffer_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            command_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1,
        };
        check(
            vkAllocateCommandBuffers(device, &command_buffer_info, &command_buffer),
            "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr,
        };
        check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

        VkBufferCopy copy = {0, 0, copy_bytes};
        vkCmdCopyBuffer(command_buffer, staging_buffer.buffer, device_buffer.buffer, 1, &copy);
        VkBufferMemoryBarrier upload_barrier = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            device_buffer.buffer,
            0,
            copy_bytes,
        };
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            1,
            &upload_barrier,
            0,
            nullptr);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            &descriptor_set,
            0,
            nullptr);
        vkCmdDispatch(command_buffer, 1, 1, 1);
        VkBufferMemoryBarrier download_barrier = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            device_buffer.buffer,
            0,
            copy_bytes,
        };
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            1,
            &download_barrier,
            0,
            nullptr);
        vkCmdCopyBuffer(command_buffer, device_buffer.buffer, staging_buffer.buffer, 1, &copy);
        check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
        check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");
        VkSubmitInfo submit_info = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,
            nullptr,
            0,
            nullptr,
            nullptr,
            1,
            &command_buffer,
            0,
            nullptr,
        };
        check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        if (!staging_buffer.host_coherent) {
            VkMappedMemoryRange range = {
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                nullptr,
                staging_buffer.memory,
                0,
                VK_WHOLE_SIZE,
            };
            check(vkInvalidateMappedMemoryRanges(device, 1, &range), "vkInvalidateMappedMemoryRanges");
        }
        result.input_after = words[0];
        result.result = words[1];
        result.pass = !require_memory_result ||
            (result.input_after == kInput && result.result == kExpected);
        if (oracle_mode) {
            std::copy_n(words + kMathOffset, result.math.size(), result.math.begin());
            std::copy_n(words + kBarrierOffset, result.barrier.size(), result.barrier.begin());
            for (std::size_t row = 0; row < result.texture.size(); ++row) {
                std::copy_n(
                    words + kTextureOutputOffset + row * 4u,
                    4u,
                    result.texture[row].begin());
            }
            result.pass = result.pass &&
                std::equal(result.math.begin(), result.math.end(), kExpectedMath.begin());
            for (std::size_t index = 0; index < result.barrier.size(); ++index) {
                const std::uint32_t expected =
                    index + 1u == result.barrier.size() ? 1u :
                    static_cast<std::uint32_t>(index + 2u);
                result.pass = result.pass && result.barrier[index] == expected;
            }
            for (std::size_t row = 0; row < result.texture.size(); ++row) {
                const int coordinate = static_cast<int>(row) - 2;
                for (std::size_t component = 0; component < 4; ++component) {
                    const std::uint32_t expected = coordinate < 0 || coordinate >= 8
                        ? 0u
                        : 0x10000000u * static_cast<std::uint32_t>(component + 1u) +
                            static_cast<std::uint32_t>(coordinate) * 0x100u +
                            static_cast<std::uint32_t>(component);
                    result.pass = result.pass && result.texture[row][component] == expected;
                }
            }
        }
        if (!result.pass) {
            result.error = oracle_mode
                ? "standalone oracle result mismatch"
                : "integer/global-memory result mismatch";
        }
    } catch (const std::exception &error) {
        result.error = error.what();
        result.pass = false;
    }

    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        if (mapped != nullptr) {
            vkUnmapMemory(device, staging_buffer.memory);
        }
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        if (shader_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shader_module, nullptr);
        }
        if (pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        }
        if (descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        }
        if (descriptor_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
        }
        if (staging_buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, staging_buffer.buffer, nullptr);
        }
        if (staging_buffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, staging_buffer.memory, nullptr);
        }
        if (device_buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, device_buffer.buffer, nullptr);
        }
        if (device_buffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, device_buffer.memory, nullptr);
        }
        vkDestroyDevice(device, nullptr);
    }
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    const bool ptx_oracle_mode =
        argc == 4 && std::string(argv[2]) == "--ptx-oracle";
    const bool incomplete_ptx_mode =
        argc == 3 && std::string(argv[2]) == "--ptx-oracle";
    if ((argc != 2 && argc != 3 && !ptx_oracle_mode) || incomplete_ptx_mode) {
        std::cerr << "usage: vulkan-memory-smoke <output.json> [oracle.spv]\n"
                  << "       vulkan-memory-smoke <output.json> --ptx-oracle <generated.spv>\n";
        return 2;
    }

    check(volkInitialize(), "volkInitialize");
    std::uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        check(vkEnumerateInstanceVersion(&loader_version), "vkEnumerateInstanceVersion");
    }
    VkApplicationInfo application_info = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "pxvk-vulkan-memory-smoke",
        1,
        "none",
        0,
        loader_version < VK_API_VERSION_1_4 ? loader_version : VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo instance_info = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application_info,
        0,
        nullptr,
        0,
        nullptr,
    };
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");
    volkLoadInstance(instance);

    std::uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr), "vkEnumeratePhysicalDevices");
    if (device_count == 0) {
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("no Vulkan physical devices");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()), "vkEnumeratePhysicalDevices");

    const bool oracle_mode = argc >= 3;
    const bool require_memory_result = !ptx_oracle_mode;
    std::vector<std::uint32_t> shader;
    if (oracle_mode) {
        const char *shader_path = ptx_oracle_mode ? argv[3] : argv[2];
        std::ifstream shader_input(shader_path, std::ios::binary | std::ios::ate);
        if (!shader_input) {
            vkDestroyInstance(instance, nullptr);
            std::cerr << "failed to open SPIR-V input file\n";
            return 3;
        }
        const std::streamsize byte_count = shader_input.tellg();
        if (byte_count <= 0 ||
            byte_count % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
            vkDestroyInstance(instance, nullptr);
            std::cerr << "invalid SPIR-V byte length\n";
            return 3;
        }
        shader.resize(static_cast<std::size_t>(byte_count) / sizeof(std::uint32_t));
        shader_input.seekg(0);
        shader_input.read(reinterpret_cast<char *>(shader.data()), byte_count);
        if (!shader_input || shader.front() != SpvMagicNumber) {
            vkDestroyInstance(instance, nullptr);
            std::cerr << "invalid SPIR-V module\n";
            return 3;
        }
    } else {
        shader = make_memory_shader();
    }
    std::vector<DeviceResult> results;
    results.reserve(device_count);
    bool all_passed = true;
    for (std::uint32_t index = 0; index < device_count; ++index) {
        DeviceResult result = run_device(
            index, devices[index], shader, oracle_mode, require_memory_result);
        all_passed = all_passed && result.pass;
        results.push_back(std::move(result));
    }

    std::ofstream output(argv[1], std::ios::binary);
    if (!output) {
        vkDestroyInstance(instance, nullptr);
        std::cerr << "failed to open output file\n";
        return 3;
    }
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"pointer_bits\": " << sizeof(void *) * 8u << ",\n"
           << "  \"test\": \""
           << (ptx_oracle_mode ? "ptx_generated_oracle" :
               oracle_mode ? "cuda_semantic_oracle" : "integer_global_memory")
           << "\",\n"
           << "  \"buffer_bytes\": " << kBufferBytes << ",\n"
           << "  \"input\": \"" << hex32(kInput) << "\",\n"
           << "  \"mask\": \"" << hex32(kMask) << "\",\n"
           << "  \"addend\": \"" << hex32(kAddend) << "\",\n"
           << "  \"expected\": \"" << hex32(kExpected) << "\",\n"
           << "  \"spirv_words\": " << shader.size() << ",\n"
           << "  \"devices\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const DeviceResult &result = results[index];
        output << "    {\n"
               << "      \"index\": " << result.index << ",\n"
               << "      \"name\": " << json_string(result.properties.deviceName) << ",\n"
               << "      \"vendor_id\": \"" << hex32(result.properties.vendorID) << "\",\n"
               << "      \"device_id\": \"" << hex32(result.properties.deviceID) << "\",\n"
               << "      \"api_version\": \""
               << VK_API_VERSION_MAJOR(result.properties.apiVersion) << '.'
               << VK_API_VERSION_MINOR(result.properties.apiVersion) << '.'
               << VK_API_VERSION_PATCH(result.properties.apiVersion) << "\",\n"
               << "      \"driver_version_raw\": " << result.properties.driverVersion << ",\n"
               << "      \"device_memory_local\": "
               << (result.device_memory_local ? "true" : "false") << ",\n"
               << "      \"staging_host_coherent\": "
               << (result.staging_coherent ? "true" : "false") << ",\n"
               << "      \"input_after\": \"" << hex32(result.input_after) << "\",\n"
               << "      \"result\": \"" << hex32(result.result) << "\",\n";
        if (oracle_mode) {
            output << "      \"math\": {\n";
            for (std::size_t math_index = 0; math_index < result.math.size(); ++math_index) {
                output << "        \"" << kMathNames[math_index] << "\": \""
                       << hex32(result.math[math_index]) << "\""
                       << (math_index + 1 == result.math.size() ? "\n" : ",\n");
            }
            output << "      },\n"
                   << "      \"barrier\": {\"values\": [";
            for (std::size_t barrier_index = 0; barrier_index < result.barrier.size(); ++barrier_index) {
                output << (barrier_index == 0 ? "" : ", ") << result.barrier[barrier_index];
            }
            output << "]},\n"
                   << "      \"texture\": {\"rows\": [\n";
            for (std::size_t row = 0; row < result.texture.size(); ++row) {
                output << "        {\"coord\": " << static_cast<int>(row) - 2 << ", \"value\": [";
                for (std::size_t component = 0; component < 4; ++component) {
                    output << (component == 0 ? "" : ", ")
                           << "\"" << hex32(result.texture[row][component]) << "\"";
                }
                output << "]}" << (row + 1 == result.texture.size() ? "\n" : ",\n");
            }
            output << "      ]},\n";
        }
        output << "      \"pass\": " << (result.pass ? "true" : "false") << ",\n"
               << "      \"error\": " << json_string(result.error.c_str()) << "\n"
               << "    }" << (index + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    vkDestroyInstance(instance, nullptr);

    for (const DeviceResult &result : results) {
        std::cout << (result.pass ? "PASS  " : "FAIL  ") << result.properties.deviceName;
        if (!result.error.empty()) {
            std::cout << ": " << result.error;
        }
        std::cout << '\n';
    }
    return all_passed ? 0 : 1;
}
