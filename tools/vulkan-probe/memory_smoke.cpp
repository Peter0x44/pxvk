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
constexpr VkDeviceSize kOracleCopyBytes = 514ull * sizeof(std::uint32_t);
constexpr VkDeviceSize kKernelFixtureCopyBytes = 512ull * sizeof(std::uint32_t);
constexpr VkDeviceSize kConstantFixtureCopyBytes = 514ull * sizeof(std::uint32_t);
constexpr VkDeviceSize kTextureMetadataBytes = 6ull * sizeof(std::uint32_t);
constexpr VkDeviceSize kReorderFixtureCopyBytes =
    32768ull * sizeof(std::uint32_t);
constexpr std::size_t kReorderedPositionOffset = 512;
constexpr std::size_t kReorderedVelocityOffset = 528;
constexpr std::size_t kReorderCellStartBase = 1024;
constexpr std::size_t kReorderCellEndBase = 16384;
constexpr std::size_t kKernelHashOutputOffset = 64;
constexpr std::size_t kKernelIndexOutputOffset = 128;
constexpr std::uint32_t kExpectedKernelHash = 1u + (2u << 6u) + (3u << 12u);
constexpr std::size_t kDensityOutputOffset = 64;
constexpr std::size_t kDensityRemappedOutputOffset = 128;
constexpr std::size_t kDensitySortedPositionsOffset = 192;
constexpr std::size_t kDensityParticleHashOffset = 512;
constexpr std::size_t kDensityMappingOffset = 768;
constexpr std::size_t kDensityCellStartBase = 1024;
constexpr std::size_t kDensityCellCount = 524288;
constexpr std::size_t kDensityCellEndBase =
    kDensityCellStartBase + kDensityCellCount;
constexpr VkDeviceSize kDensityPayloadBytes =
    (kDensityCellEndBase + kDensityCellCount) * sizeof(std::uint32_t);
constexpr VkDeviceSize kDensityConstantBytes = 10240u;
constexpr VkDeviceSize kDensityTextureMetadataBytes =
    4u * 3u * sizeof(std::uint32_t);
constexpr std::uint32_t kExpectedDensityBits = 0x40000000u;
constexpr std::size_t kForceOutputOffset = 64;
constexpr std::size_t kForceSortedPositionsOffset = 192;
constexpr std::size_t kForceSortedVelocitiesOffset = 448;
constexpr std::size_t kForceSortedDensitiesOffset = 704;
constexpr std::size_t kForceParticleHashOffset = 768;
constexpr std::size_t kForceMappingOffset = 832;
constexpr std::size_t kForceCellStartBase = 1024;
constexpr std::size_t kForceCellCount = 524288;
constexpr std::size_t kForceCellEndBase =
    kForceCellStartBase + kForceCellCount;
constexpr VkDeviceSize kForcePayloadBytes =
    (kForceCellEndBase + kForceCellCount) * sizeof(std::uint32_t);
constexpr VkDeviceSize kForceConstantBytes = 10240u;
constexpr VkDeviceSize kForceTextureMetadataBytes =
    6u * 3u * sizeof(std::uint32_t);
constexpr std::array<std::uint32_t, 4> kExpectedForce = {
    0x40800000u, 0x40a00000u, 0x40c00000u, 0u,
};
constexpr std::size_t kDeformablePackedBaseOffset = 64;
constexpr std::size_t kDeformablePositionOffset = 128;
constexpr std::size_t kDeformableMotionOffset = 160;
constexpr std::size_t kDeformableFlagsOffset = 193;
constexpr std::size_t kDeformableIgnoredOffset = 225;
constexpr VkDeviceSize kDeformablePayloadBytes = 256u * sizeof(std::uint32_t);
constexpr VkDeviceSize kDeformableConstantBytes = 7168u;
constexpr std::uint32_t kGuardBefore = 0x13579bdfu;
constexpr std::uint32_t kGuardAfter = 0x2468ace0u;
constexpr std::array<std::uint32_t, 4> kExpectedDeformablePosition = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0u,
};
constexpr std::array<std::uint32_t, 4> kExpectedDeformableMotion = {
    0x40440000u, 0x408c0000u, 0x40c80000u, 0x40e00000u,
};
constexpr std::size_t kSolverAOffset = 32;
constexpr std::size_t kSolverBOffset = 48;
constexpr std::size_t kSolverCOffset = 64;
constexpr std::size_t kSolverPositionOffset = 80;
constexpr std::size_t kSolverFlagsOffset = 96;
constexpr std::size_t kSolverPackedOffset = 112;
constexpr std::size_t kSolverOutputOffset = 128;
constexpr VkDeviceSize kSolverPayloadBytes = 256u * sizeof(std::uint32_t);
constexpr VkDeviceSize kSolverConstantBytes = 5888u;
constexpr std::array<std::uint32_t, 4> kExpectedSolverA = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0u,
};
constexpr std::array<std::uint32_t, 4> kExpectedSolverPosition = {
    0xc1100000u, 0xc1100000u, 0xc1100000u, 0u,
};
constexpr std::array<std::uint32_t, 16> kExpectedSolverOutput = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0x00000400u,
    0x41f00000u, 0x41f80000u, 0x42000000u, 0xabcdef01u,
    0xc1100000u, 0xc1100000u, 0xc1100000u, 0x00000000u,
    0x41a00000u, 0x41a80000u, 0x41b00000u, 0x00000000u,
};
constexpr std::size_t kCollisionBatchOffset = 8;
constexpr std::size_t kCollisionInputPositionOffset = 20;
constexpr std::size_t kCollisionInputMotionOffset = 28;
constexpr std::size_t kCollisionOutput8Offset = 36;
constexpr std::size_t kCollisionOutput12Offset = 44;
constexpr std::size_t kCollisionOutput20Offset = 52;
constexpr std::size_t kCollisionFlagsOffset = 56;
constexpr std::size_t kCollisionOutputPositionOffset = 60;
constexpr std::size_t kCollisionOutputMotionOffset = 68;
constexpr std::size_t kCollisionVector40Offset = 76;
constexpr std::size_t kCollisionVector44Offset = 84;
constexpr std::size_t kCollisionScalar48Offset = 92;
constexpr std::size_t kCollisionScalar52Offset = 96;
constexpr std::size_t kCollisionScalar56Offset = 100;
constexpr std::size_t kCollisionScalar60Offset = 104;
constexpr std::size_t kCollisionVector64Offset = 108;
constexpr std::size_t kCollisionActiveOffset = 116;
constexpr VkDeviceSize kCollisionPayloadBytes =
    192u * sizeof(std::uint32_t);
constexpr VkDeviceSize kCollisionConstantBytes = 17992u;
constexpr VkDeviceSize kCollisionTextureMetadataBytes =
    17u * 3u * sizeof(std::uint32_t);
constexpr std::array<std::uint32_t, 4> kExpectedCollisionPosition = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
};
constexpr std::array<std::uint32_t, 4> kExpectedCollisionMotion = {
    0x40a00000u, 0x40c00000u, 0x40e00000u, 0x41000000u,
};
constexpr std::array<std::uint32_t, 4> kExpectedCollisionScalars = {
    0xbf800000u, 0x00000000u, 0x0000ffffu, 0x00000000u,
};

constexpr std::size_t kSequenceUpdatePackedOffset = 8;
constexpr std::size_t kSequenceUpdatePositionOffset = 16;
constexpr std::size_t kSequenceUpdateMotionOffset = 24;
constexpr std::size_t kSequenceUpdateFlagsOffset = 32;
constexpr std::size_t kSequenceUpdateIgnoredOffset = 36;
constexpr std::size_t kSequenceCollisionBatchOffset = 40;
constexpr std::size_t kSequenceCollisionOutput8Offset = 48;
constexpr std::size_t kSequenceCollisionOutput12Offset = 56;
constexpr std::size_t kSequenceCollisionOutput20Offset = 64;
constexpr std::size_t kSequenceCollisionOutputPositionOffset = 72;
constexpr std::size_t kSequenceCollisionOutputMotionOffset = 80;
constexpr std::size_t kSequenceCollisionVector40Offset = 88;
constexpr std::size_t kSequenceCollisionVector44Offset = 96;
constexpr std::size_t kSequenceCollisionScalar48Offset = 104;
constexpr std::size_t kSequenceCollisionScalar52Offset = 108;
constexpr std::size_t kSequenceCollisionScalar56Offset = 112;
constexpr std::size_t kSequenceCollisionScalar60Offset = 116;
constexpr std::size_t kSequenceCollisionVector64Offset = 120;
constexpr std::size_t kSequenceCollisionActiveOffset = 128;
constexpr std::size_t kSequenceSolverAOffset = 144;
constexpr std::size_t kSequenceSolverBOffset = 152;
constexpr std::size_t kSequenceSolverCOffset = 160;
constexpr std::size_t kSequenceSolverFlagsOffset = 168;
constexpr std::size_t kSequenceSolverPackedOffset = 172;
constexpr std::size_t kSequenceSolverOutputOffset = 180;
constexpr VkDeviceSize kSequencePayloadBytes =
    320u * sizeof(std::uint32_t);
constexpr std::size_t kConstraintsWorkUnitOffset = 8;
constexpr std::size_t kConstraintsStartAddrsOffset = 16;
constexpr std::size_t kConstraintsParametersOffset = 52;
constexpr std::size_t kConstraintsKeyOffset = 96;
constexpr std::size_t kConstraintsInputPositionOffset = 104;
constexpr std::size_t kConstraintsInputMotionOffset = 112;
constexpr std::size_t kConstraintsInputOtherOffset = 120;
constexpr std::array<std::size_t, 8> kConstraintsVectorOutputOffsets = {
    128, 136, 144, 152, 160, 168, 216, 224,
};
constexpr std::array<std::size_t, 4> kConstraintsScalarOutputOffsets = {
    176, 180, 184, 188,
};
constexpr std::size_t kConstraintsWeightOffset = 192;
constexpr std::size_t kConstraintsOutputPositionOffset = 200;
constexpr std::size_t kConstraintsOutputVelocityOffset = 208;
constexpr std::size_t kConstraintsErrorOffset = 232;
constexpr VkDeviceSize kConstraintsPayloadBytes =
    320u * sizeof(std::uint32_t);
constexpr std::array<std::uint32_t, 4> kExpectedConstraintsPosition = {
    0x40400000u, 0x40900000u, 0x40c00000u, 0u,
};
constexpr std::array<std::uint32_t, 4> kExpectedConstraintsVelocity = {
    0x40800000u, 0x40a00000u, 0x40c00000u, 0u,
};
constexpr std::size_t kDynamicResponseWorkUnitOffset = 8;
constexpr std::size_t kDynamicResponseStartAddrsOffset = 16;
constexpr std::size_t kDynamicResponseParametersOffset = 52;
constexpr std::size_t kDynamicResponseNormalOffset = 96;
constexpr std::size_t kDynamicResponseAccumulatedPositionOffset = 104;
constexpr std::size_t kDynamicResponseBodyVelocityOffset = 112;
constexpr std::size_t kDynamicResponseCountOffset = 120;
constexpr std::size_t kDynamicResponseWeightOffset = 124;
constexpr std::size_t kDynamicResponseAveragedPositionOffset = 128;
constexpr std::size_t kDynamicResponseParticleVelocityOffset = 136;
constexpr std::size_t kDynamicResponseOutputOffset = 144;
constexpr std::size_t kDynamicResponseDestIdOffset = 152;
constexpr std::size_t kDynamicResponseSourceIdOffset = 156;
constexpr std::size_t kDynamicResponseFlagsOffset = 160;
constexpr std::size_t kDynamicResponseErrorOffset = 164;
constexpr VkDeviceSize kDynamicResponsePayloadBytes =
    192u * sizeof(std::uint32_t);
constexpr std::array<std::uint32_t, 4> kExpectedDynamicResponsePosition = {
    0x40e00000u, 0x41000000u, 0x41100000u, 0u,
};
constexpr std::array<std::uint32_t, 4> kExpectedDynamicResponseVelocity = {
    0x3f800000u, 0x40400000u, 0u, 0u,
};
constexpr std::array<std::uint32_t, 4> kExpectedDynamicResponseOutput = {
    0xc0400000u, 0u, 0u, 0u,
};
constexpr std::array<std::uint32_t, 4> kExpectedReorderedPosition = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0u,
};
constexpr std::array<std::uint32_t, 4> kExpectedReorderedVelocity = {
    0x40800000u, 0x40a00000u, 0x40c00000u, 0x40e00000u,
};
constexpr std::uint32_t kInput = 0x12345678u;
constexpr std::uint32_t kMask = 0xa5a55a5au;
constexpr std::uint32_t kAddend = 0x10203040u;
constexpr std::uint32_t kExpected = (kInput ^ kMask) + kAddend;
constexpr std::size_t kMathOffset = 16;
constexpr std::size_t kBarrierOffset = 64;
constexpr std::size_t kTextureInputOffset = 160;
constexpr std::size_t kTextureOutputOffset = 256;
constexpr std::size_t kTextureFloatInputOffset = 320;
constexpr std::size_t kTextureFloatOutputOffset = 384;
constexpr std::size_t kLocalOffset = 448;
constexpr std::size_t kConstantOutputOffset = 512;
constexpr std::array<std::uint32_t, 2> kExpectedConstant = {
    0x1234abcdu, 0x7fc12345u,
};
constexpr std::array<std::array<std::uint32_t, 4>, 8>
    kExpectedFloatTextureInput = {{
        {0x3f800000u, 0x80000000u, 0x00000001u, 0x7fc12345u},
        {0x40000000u, 0xbf800000u, 0x00800000u, 0x7f800000u},
        {0x40400000u, 0xc0000000u, 0x3eaaaaabu, 0xff800000u},
        {0x40800000u, 0x3f000000u, 0x3f3504f3u, 0x7fc00000u},
        {0x40a00000u, 0x3fc00000u, 0x3dcccccdu, 0x00000000u},
        {0x40c00000u, 0x40200000u, 0x3f7fffffu, 0x80000001u},
        {0x40e00000u, 0x40600000u, 0x3f800001u, 0x7f7fffffu},
        {0x41000000u, 0x40900000u, 0xbf000000u, 0xff7fffffu},
    }};
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

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment)
{
    return ((value + alignment - 1u) / alignment) * alignment;
}

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
    std::array<std::uint32_t, 64> local = {};
    std::array<std::array<std::uint32_t, 4>, 12> texture = {};
    std::array<std::array<std::uint32_t, 4>, 12> texture_float = {};
    std::array<std::uint32_t, 2> constant = {};
    std::uint32_t kernel_hash = 0;
    std::uint32_t kernel_index = 0;
    std::array<std::uint32_t, 4> reordered_position = {};
    std::array<std::uint32_t, 4> reordered_velocity = {};
    std::uint32_t cell_start = 0;
    std::uint32_t cell_end = 0;
    std::uint32_t density = 0;
    std::uint32_t remapped_density = 0;
    std::array<std::uint32_t, 4> force = {};
    std::array<std::uint32_t, 4> deformable_position = {};
    std::array<std::uint32_t, 4> deformable_motion = {};
    std::uint32_t deformable_flags = 0;
    bool deformable_guards_intact = false;
    std::array<std::uint32_t, 4> solver_a = {};
    std::array<std::uint32_t, 4> solver_position = {};
    std::array<std::uint32_t, 16> solver_output = {};
    bool solver_guards_intact = false;
    std::array<std::uint32_t, 4> collision_position = {};
    std::array<std::uint32_t, 4> collision_motion = {};
    std::array<std::uint32_t, 4> collision_scalars = {};
    bool collision_guards_intact = false;
    std::array<std::uint32_t, 4> sequence_update_motion = {};
    std::array<std::uint32_t, 4> sequence_collision_motion = {};
    std::array<std::uint32_t, 4> sequence_collision_scalars = {};
    std::array<std::uint32_t, 4> sequence_solver_a = {};
    std::array<std::uint32_t, 4> sequence_solver_position = {};
    std::array<std::uint32_t, 16> sequence_solver_output = {};
    bool sequence_guards_intact = false;
    std::array<std::uint32_t, 4> constraints_position = {};
    std::array<std::uint32_t, 4> constraints_velocity = {};
    std::uint32_t constraints_weight = 0;
    bool constraints_guards_intact = false;
    std::array<std::uint32_t, 4> dynamic_response_normal = {};
    std::array<std::uint32_t, 4> dynamic_response_position = {};
    std::array<std::uint32_t, 4> dynamic_response_velocity = {};
    std::array<std::uint32_t, 4> dynamic_response_output = {};
    std::uint32_t dynamic_response_weight = 0;
    std::uint32_t dynamic_response_dest_id = 0;
    std::uint32_t dynamic_response_flags = 0;
    bool dynamic_response_guards_intact = false;
    std::string error;
};

DeviceResult run_device(
    std::uint32_t device_index,
    VkPhysicalDevice physical_device,
    const std::vector<std::uint32_t> &shader,
    bool oracle_mode,
    bool ptx_oracle_mode,
    bool require_memory_result,
    bool kernel_fixture_mode,
    bool reorder_fixture_mode,
    bool constant_fixture_mode,
    bool texture_fixture_mode,
    bool texture_float_fixture_mode,
    bool density_fixture_mode,
    bool force_fixture_mode,
    bool deformable_fixture_mode,
    bool solver_fixture_mode,
    bool collision_fixture_mode,
    bool constraints_fixture_mode,
    bool dynamic_response_fixture_mode)
{
    DeviceResult result;
    result.index = device_index;
    vkGetPhysicalDeviceProperties(physical_device, &result.properties);
    const bool selected_texture_mode =
        texture_fixture_mode || texture_float_fixture_mode;
    const VkDeviceSize texture_metadata_alignment =
        std::max<VkDeviceSize>(
            4u, result.properties.limits.minStorageBufferOffsetAlignment);
    const VkDeviceSize texture_metadata_offset =
        align_up(4096u, texture_metadata_alignment);
    const VkDeviceSize density_constant_offset =
        align_up(kDensityPayloadBytes, texture_metadata_alignment);
    const VkDeviceSize density_texture_metadata_offset =
        align_up(
            density_constant_offset + kDensityConstantBytes,
            texture_metadata_alignment);
    const VkDeviceSize deformable_constant_offset =
        align_up(kDeformablePayloadBytes, texture_metadata_alignment);
    const VkDeviceSize solver_constant_offset =
        align_up(kSolverPayloadBytes, texture_metadata_alignment);
    const VkDeviceSize collision_constant_offset =
        align_up(kCollisionPayloadBytes, texture_metadata_alignment);
    const VkDeviceSize collision_texture_metadata_offset =
        align_up(
            collision_constant_offset + kCollisionConstantBytes,
            texture_metadata_alignment);
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

        const VkDeviceSize copy_bytes = oracle_mode
            ? kOracleCopyBytes
            : reorder_fixture_mode
                ? kReorderFixtureCopyBytes
                : kernel_fixture_mode
                    ? kKernelFixtureCopyBytes
                    : constant_fixture_mode
                        ? kConstantFixtureCopyBytes
                        : dynamic_response_fixture_mode
                            ? kDynamicResponsePayloadBytes
                            : constraints_fixture_mode
                                ? kConstraintsPayloadBytes
                                : collision_fixture_mode
                                ? collision_texture_metadata_offset +
                                      kCollisionTextureMetadataBytes
                                : solver_fixture_mode
                                    ? solver_constant_offset + kSolverConstantBytes
                                    : deformable_fixture_mode
                                        ? deformable_constant_offset +
                                              kDeformableConstantBytes
                                        : force_fixture_mode
                                            ? density_texture_metadata_offset +
                                                  kForceTextureMetadataBytes
                                            : density_fixture_mode
                                                ? density_texture_metadata_offset +
                                                      kDensityTextureMetadataBytes
                                                : selected_texture_mode
                                                    ? texture_metadata_offset +
                                                          kTextureMetadataBytes
                                                    : kMemoryCopyBytes;
        check(
            vkMapMemory(device, staging_buffer.memory, 0, staging_buffer.allocation_size, 0, &mapped),
            "vkMapMemory");
        auto *words = static_cast<std::uint32_t *>(mapped);
        std::memset(words, 0, static_cast<std::size_t>(copy_bytes));
        if (dynamic_response_fixture_mode) {
            const auto byte_offset = [](std::size_t offset) {
                return static_cast<std::uint32_t>(
                    offset * sizeof(std::uint32_t));
            };
            const auto initialize = [&](std::size_t offset,
                                        std::initializer_list<std::uint32_t> values) {
                words[offset - 1u] = kGuardBefore;
                std::copy(values.begin(), values.end(), words + offset);
                words[offset + values.size()] = kGuardAfter;
            };
            words[0] = byte_offset(kDynamicResponseStartAddrsOffset);
            words[1] = 1u;
            words[2] = byte_offset(kDynamicResponseWorkUnitOffset);
            words[3] = 1u;
            words[4] = byte_offset(kDynamicResponseErrorOffset);
            initialize(kDynamicResponseWorkUnitOffset, {0u, 1u, 0u, 0u});
            words[kDynamicResponseStartAddrsOffset - 1u] = kGuardBefore;
            std::fill_n(words + kDynamicResponseStartAddrsOffset, 32u, 0u);
            words[kDynamicResponseStartAddrsOffset + 32u] = kGuardAfter;
            auto *start_addrs = words + kDynamicResponseStartAddrsOffset;
            start_addrs[8] = byte_offset(kDynamicResponseNormalOffset);
            start_addrs[9] =
                byte_offset(kDynamicResponseAccumulatedPositionOffset);
            start_addrs[10] = byte_offset(kDynamicResponseBodyVelocityOffset);
            start_addrs[11] = byte_offset(kDynamicResponseCountOffset);
            start_addrs[12] = byte_offset(kDynamicResponseWeightOffset);
            start_addrs[13] =
                byte_offset(kDynamicResponseAveragedPositionOffset);
            start_addrs[14] =
                byte_offset(kDynamicResponseParticleVelocityOffset);
            start_addrs[15] = byte_offset(kDynamicResponseOutputOffset);
            start_addrs[16] = byte_offset(kDynamicResponseDestIdOffset);
            start_addrs[17] = byte_offset(kDynamicResponseSourceIdOffset);
            start_addrs[18] = byte_offset(kDynamicResponseFlagsOffset);
            start_addrs[27] = byte_offset(kDynamicResponseParametersOffset);
            words[kDynamicResponseParametersOffset - 1u] = kGuardBefore;
            std::fill_n(words + kDynamicResponseParametersOffset, 40u, 0u);
            words[kDynamicResponseParametersOffset + 29u] = 0x3f000000u;
            words[kDynamicResponseParametersOffset + 34u] = 1u;
            words[kDynamicResponseParametersOffset + 40u] = kGuardAfter;
            initialize(
                kDynamicResponseNormalOffset,
                {0x3f800000u, 0u, 0u, 0xdeadbeefu});
            initialize(
                kDynamicResponseAccumulatedPositionOffset,
                {0x40e00000u, 0x41000000u, 0x41100000u, 0xcafebabeu});
            initialize(
                kDynamicResponseBodyVelocityOffset,
                {0u, 0u, 0u, 0xabcdef01u});
            initialize(kDynamicResponseCountOffset, {1u});
            initialize(kDynamicResponseWeightOffset, {0xfeedfaceu});
            initialize(
                kDynamicResponseAveragedPositionOffset,
                {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu});
            initialize(
                kDynamicResponseParticleVelocityOffset,
                {0xc0000000u, 0x40400000u, 0u, 0xbaadf00du});
            initialize(
                kDynamicResponseOutputOffset,
                {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu});
            initialize(kDynamicResponseDestIdOffset, {0xfeedfaceu});
            initialize(kDynamicResponseSourceIdOffset, {0x12345678u});
            initialize(kDynamicResponseFlagsOffset, {1u});
            initialize(kDynamicResponseErrorOffset, {0xfeedfaceu});
        } else if (constraints_fixture_mode) {
            const auto byte_offset = [](std::size_t offset) {
                return static_cast<std::uint32_t>(
                    offset * sizeof(std::uint32_t));
            };
            const auto initialize = [&](std::size_t offset,
                                        std::initializer_list<std::uint32_t> values) {
                words[offset - 1u] = kGuardBefore;
                std::copy(values.begin(), values.end(), words + offset);
                words[offset + values.size()] = kGuardAfter;
            };
            words[0] = byte_offset(kConstraintsStartAddrsOffset);
            words[1] = 1u;
            words[2] = byte_offset(kConstraintsWorkUnitOffset);
            words[3] = 1u;
            words[4] = byte_offset(kConstraintsErrorOffset);
            initialize(kConstraintsWorkUnitOffset, {0u, 1u, 0u, 0u});
            words[kConstraintsStartAddrsOffset - 1u] = kGuardBefore;
            std::fill_n(words + kConstraintsStartAddrsOffset, 32u, 0u);
            words[kConstraintsStartAddrsOffset + 32u] = kGuardAfter;
            auto *start_addrs = words + kConstraintsStartAddrsOffset;
            start_addrs[3] = byte_offset(kConstraintsVectorOutputOffsets[7]);
            start_addrs[4] = byte_offset(kConstraintsVectorOutputOffsets[0]);
            start_addrs[5] = byte_offset(kConstraintsVectorOutputOffsets[1]);
            start_addrs[6] = byte_offset(kConstraintsVectorOutputOffsets[2]);
            start_addrs[7] = byte_offset(kConstraintsKeyOffset);
            start_addrs[8] = byte_offset(kConstraintsVectorOutputOffsets[3]);
            start_addrs[9] = byte_offset(kConstraintsVectorOutputOffsets[4]);
            start_addrs[10] = byte_offset(kConstraintsVectorOutputOffsets[5]);
            start_addrs[11] = byte_offset(kConstraintsScalarOutputOffsets[0]);
            start_addrs[12] = byte_offset(kConstraintsWeightOffset);
            start_addrs[13] = byte_offset(kConstraintsOutputPositionOffset);
            start_addrs[14] = byte_offset(kConstraintsOutputVelocityOffset);
            start_addrs[15] = byte_offset(kConstraintsVectorOutputOffsets[6]);
            start_addrs[16] = byte_offset(kConstraintsScalarOutputOffsets[1]);
            start_addrs[17] = byte_offset(kConstraintsScalarOutputOffsets[2]);
            start_addrs[18] = byte_offset(kConstraintsScalarOutputOffsets[3]);
            start_addrs[20] = byte_offset(kConstraintsInputPositionOffset);
            start_addrs[21] = byte_offset(kConstraintsInputMotionOffset);
            start_addrs[22] = byte_offset(kConstraintsInputOtherOffset);
            start_addrs[27] = byte_offset(kConstraintsParametersOffset);
            words[kConstraintsParametersOffset - 1u] = kGuardBefore;
            std::fill_n(words + kConstraintsParametersOffset, 40u, 0u);
            words[kConstraintsParametersOffset] = 0x3f000000u;
            words[kConstraintsParametersOffset + 40u] = kGuardAfter;
            initialize(kConstraintsKeyOffset, {0x0000ffffu, 0x12345678u});
            initialize(
                kConstraintsInputPositionOffset,
                {0x3f800000u, 0x40000000u, 0x40400000u, 0xdeadbeefu});
            initialize(
                kConstraintsInputMotionOffset,
                {0x40800000u, 0x40a00000u, 0x40c00000u, 0xcafebabeu});
            initialize(
                kConstraintsInputOtherOffset,
                {0x3dcccccdu, 0x3e4ccccdu, 0x3e99999au, 0xabcdef01u});
            for (const std::size_t offset : kConstraintsVectorOutputOffsets) {
                initialize(
                    offset,
                    {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu});
            }
            for (const std::size_t offset : kConstraintsScalarOutputOffsets) {
                initialize(offset, {0xfeedfaceu});
            }
            initialize(kConstraintsWeightOffset, {0xfeedfaceu});
            initialize(
                kConstraintsOutputPositionOffset,
                {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu});
            initialize(
                kConstraintsOutputVelocityOffset,
                {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu});
            initialize(kConstraintsErrorOffset, {0xfeedfaceu});
        } else if (collision_fixture_mode) {
            words[0] = 0u;
            const auto initialize = [&](std::size_t offset,
                                        std::initializer_list<std::uint32_t> values) {
                words[offset - 1u] = kGuardBefore;
                std::copy(values.begin(), values.end(), words + offset);
                words[offset + values.size()] = kGuardAfter;
            };
            initialize(kCollisionBatchOffset, {0x00010000u, 0u});
            initialize(
                kCollisionInputPositionOffset,
                {0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u});
            initialize(
                kCollisionInputMotionOffset,
                {0x40a00000u, 0x40c00000u, 0x40e00000u, 0x41000000u});
            for (const std::size_t offset : {
                     kCollisionOutput8Offset,
                     kCollisionOutput12Offset,
                     kCollisionOutputPositionOffset,
                     kCollisionOutputMotionOffset,
                     kCollisionVector40Offset,
                     kCollisionVector44Offset,
                     kCollisionVector64Offset,
                 }) {
                initialize(
                    offset,
                    {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu});
            }
            for (const std::size_t offset : {
                     kCollisionOutput20Offset,
                     kCollisionScalar48Offset,
                     kCollisionScalar52Offset,
                     kCollisionScalar56Offset,
                     kCollisionScalar60Offset,
                 }) {
                initialize(offset, {0xfeedfaceu});
            }
            initialize(kCollisionFlagsOffset, {0u});
            initialize(kCollisionActiveOffset, {1u});

            auto *constants = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) +
                collision_constant_offset);
            const auto byte_offset = [](std::size_t offset) {
                return static_cast<std::uint32_t>(
                    offset * sizeof(std::uint32_t));
            };
            constants[0] = byte_offset(kCollisionInputPositionOffset);
            constants[1] = byte_offset(kCollisionInputMotionOffset);
            constants[2] = byte_offset(kCollisionOutput8Offset);
            constants[3] = byte_offset(kCollisionOutput12Offset);
            constants[5] = byte_offset(kCollisionOutput20Offset);
            constants[6] = byte_offset(kCollisionFlagsOffset);
            constants[8] = byte_offset(kCollisionOutputPositionOffset);
            constants[9] = byte_offset(kCollisionOutputMotionOffset);
            constants[10] = byte_offset(kCollisionVector40Offset);
            constants[11] = byte_offset(kCollisionVector44Offset);
            constants[12] = byte_offset(kCollisionScalar48Offset);
            constants[13] = byte_offset(kCollisionScalar52Offset);
            constants[14] = byte_offset(kCollisionScalar56Offset);
            constants[15] = byte_offset(kCollisionScalar60Offset);
            constants[16] = byte_offset(kCollisionVector64Offset);
            constants[17] = byte_offset(kCollisionActiveOffset);
            constants[4480u + 17u] = byte_offset(kCollisionBatchOffset);
        } else if (solver_fixture_mode) {
            words[0] = 0u;
            words[1] = 0u;
            words[2] = static_cast<std::uint32_t>(
                kSolverOutputOffset * sizeof(std::uint32_t));
            words[3] = 0u;

            const auto initialize = [&](std::size_t offset,
                                        std::initializer_list<std::uint32_t> values) {
                words[offset - 1u] = kGuardBefore;
                std::copy(values.begin(), values.end(), words + offset);
                words[offset + values.size()] = kGuardAfter;
            };
            initialize(
                kSolverAOffset,
                {0x41200000u, 0x41300000u, 0x41400000u, 0u});
            initialize(
                kSolverBOffset,
                {0x41a00000u, 0x41a80000u, 0x41b00000u, 0u});
            initialize(
                kSolverCOffset,
                {0x41f00000u, 0x41f80000u, 0x42000000u, 0u});
            initialize(
                kSolverPositionOffset,
                {0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u});
            initialize(kSolverFlagsOffset, {0x00000400u});
            initialize(kSolverPackedOffset, {0xabcdef01u});
            words[kSolverOutputOffset - 1u] = kGuardBefore;
            std::fill_n(words + kSolverOutputOffset, 16u, 0u);
            words[kSolverOutputOffset + 16u] = kGuardAfter;

            auto *constants = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) +
                solver_constant_offset);
            constants[1] = 1u;
            constants[8] = 0x3f800000u;
            constants[9] = 0u;
            constants[14] = static_cast<std::uint32_t>(
                kSolverAOffset * sizeof(std::uint32_t));
            constants[15] = static_cast<std::uint32_t>(
                kSolverPositionOffset * sizeof(std::uint32_t));
            constants[16] = static_cast<std::uint32_t>(
                kSolverBOffset * sizeof(std::uint32_t));
            constants[17] = static_cast<std::uint32_t>(
                kSolverCOffset * sizeof(std::uint32_t));
            constants[18] = static_cast<std::uint32_t>(
                kSolverCOffset * sizeof(std::uint32_t));
            constants[21] = static_cast<std::uint32_t>(
                kSolverFlagsOffset * sizeof(std::uint32_t));
            constants[22] = static_cast<std::uint32_t>(
                kSolverPackedOffset * sizeof(std::uint32_t));
        } else if (deformable_fixture_mode) {
            words[0] = static_cast<std::uint32_t>(
                (kDeformablePackedBaseOffset + 1u) * sizeof(std::uint32_t));
            words[1] = 1u;
            words[2] = static_cast<std::uint32_t>(
                kDeformableIgnoredOffset * sizeof(std::uint32_t));
            words[kDeformablePackedBaseOffset] = kGuardBefore;
            words[kDeformablePackedBaseOffset + 1u] = 0xfeedfaceu;
            words[kDeformablePackedBaseOffset + 2u] = kGuardAfter;
            words[kDeformablePositionOffset - 1u] = kGuardBefore;
            words[kDeformablePositionOffset] = 0x3f800000u;
            words[kDeformablePositionOffset + 1u] = 0x40000000u;
            words[kDeformablePositionOffset + 2u] = 0x40400000u;
            words[kDeformablePositionOffset + 3u] = 0u;
            words[kDeformablePositionOffset + 4u] = kGuardAfter;
            words[kDeformableMotionOffset - 1u] = kGuardBefore;
            words[kDeformableMotionOffset] = 0x40800000u;
            words[kDeformableMotionOffset + 1u] = 0x40a00000u;
            words[kDeformableMotionOffset + 2u] = 0x40c00000u;
            words[kDeformableMotionOffset + 3u] = 0x40e00000u;
            words[kDeformableMotionOffset + 4u] = kGuardAfter;
            words[kDeformableFlagsOffset - 1u] = kGuardBefore;
            words[kDeformableFlagsOffset] = 0x00000014u;
            words[kDeformableFlagsOffset + 1u] = kGuardAfter;
            words[kDeformableIgnoredOffset - 1u] = kGuardBefore;
            words[kDeformableIgnoredOffset] = 0xcafebabeu;
            words[kDeformableIgnoredOffset + 1u] = kGuardAfter;

            auto *constants = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) +
                deformable_constant_offset);
            constants[0] = 0x3e800000u;
            constants[1] = 0xbf000000u;
            constants[2] = 0x3f800000u;
            constants[7] = 1u;
            constants[15] = 0x3f000000u;
            constants[16] = static_cast<std::uint32_t>(
                kDeformablePositionOffset * sizeof(std::uint32_t));
            constants[17] = static_cast<std::uint32_t>(
                kDeformableMotionOffset * sizeof(std::uint32_t));
            constants[22] = static_cast<std::uint32_t>(
                kDeformableFlagsOffset * sizeof(std::uint32_t));
        } else if (force_fixture_mode) {
            words[0] = static_cast<std::uint32_t>(
                kForceOutputOffset * sizeof(std::uint32_t));
            words[1] = 1u;
            words[11] = static_cast<std::uint32_t>(
                kForceSortedPositionsOffset * sizeof(std::uint32_t));
            words[12] = 1u;
            words[13] = static_cast<std::uint32_t>(
                kForceSortedVelocitiesOffset * sizeof(std::uint32_t));
            words[14] = 1u;
            words[15] = static_cast<std::uint32_t>(
                kForceSortedDensitiesOffset * sizeof(std::uint32_t));
            words[16] = 1u;
            words[17] = static_cast<std::uint32_t>(
                kForceParticleHashOffset * sizeof(std::uint32_t));
            words[18] = 1u;
            words[19] = static_cast<std::uint32_t>(
                kForceMappingOffset * sizeof(std::uint32_t));
            words[20] = 1u;
            words[21] = static_cast<std::uint32_t>(
                kForceCellStartBase * sizeof(std::uint32_t));
            words[22] = static_cast<std::uint32_t>(kForceCellCount);
            words[23] = static_cast<std::uint32_t>(
                kForceCellEndBase * sizeof(std::uint32_t));
            words[24] = static_cast<std::uint32_t>(kForceCellCount);
            std::fill_n(
                words + kForceOutputOffset,
                kExpectedForce.size(),
                0xdeadbeefu);
            words[kForceSortedPositionsOffset] = 0x3f800000u;
            words[kForceSortedPositionsOffset + 1u] = 0x40000000u;
            words[kForceSortedPositionsOffset + 2u] = 0x40400000u;
            words[kForceSortedPositionsOffset + 3u] = 0u;
            words[kForceSortedVelocitiesOffset] = 0x40800000u;
            words[kForceSortedVelocitiesOffset + 1u] = 0x40a00000u;
            words[kForceSortedVelocitiesOffset + 2u] = 0x40c00000u;
            words[kForceSortedVelocitiesOffset + 3u] = 0x40e00000u;
            words[kForceSortedDensitiesOffset] = kExpectedDensityBits;
            std::fill_n(
                words + kForceParticleHashOffset,
                64u,
                0xffffffffu);
            words[kForceParticleHashOffset] = kExpectedKernelHash;
            words[kForceMappingOffset] = 0u;
            std::fill_n(
                words + kForceCellStartBase,
                kForceCellCount,
                0xffffffffu);
            std::fill_n(
                words + kForceCellEndBase,
                kForceCellCount,
                0xffffffffu);
            words[kForceCellStartBase + kExpectedKernelHash] = 0u;
            words[kForceCellEndBase + kExpectedKernelHash] = 1u;

            auto *constants = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) + density_constant_offset);
            constants[0] = 0x3f800000u;
            constants[10] = 0x3f800000u;
            constants[12] = 0x3f800000u;
            constants[16] = 0x42c80000u;
            constants[18] = 0x3f800000u;
            constants[21] = 0x3f800000u;
            auto *metadata = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) +
                density_texture_metadata_offset);
            const std::array<std::array<std::uint32_t, 3>, 6> records = {{
                {static_cast<std::uint32_t>(kForceSortedPositionsOffset), 64u, 4u},
                {static_cast<std::uint32_t>(kForceSortedVelocitiesOffset), 64u, 4u},
                {static_cast<std::uint32_t>(kForceSortedDensitiesOffset), 64u, 1u},
                {static_cast<std::uint32_t>(kForceParticleHashOffset), 64u, 1u},
                {static_cast<std::uint32_t>(kForceCellStartBase),
                 static_cast<std::uint32_t>(kForceCellCount), 1u},
                {static_cast<std::uint32_t>(kForceCellEndBase),
                 static_cast<std::uint32_t>(kForceCellCount), 1u},
            }};
            for (std::size_t record = 0u; record < records.size(); ++record) {
                std::copy(
                    records[record].begin(),
                    records[record].end(),
                    metadata + record * 3u);
            }
        } else if (density_fixture_mode) {
            words[0] = static_cast<std::uint32_t>(
                kDensityOutputOffset * sizeof(std::uint32_t));
            words[1] = 1u;
            words[2] = static_cast<std::uint32_t>(
                kDensityRemappedOutputOffset * sizeof(std::uint32_t));
            words[3] = 1u;
            words[11] = static_cast<std::uint32_t>(
                kDensitySortedPositionsOffset * sizeof(std::uint32_t));
            words[12] = 1u;
            words[13] = static_cast<std::uint32_t>(
                kDensityParticleHashOffset * sizeof(std::uint32_t));
            words[14] = 1u;
            words[15] = static_cast<std::uint32_t>(
                kDensityMappingOffset * sizeof(std::uint32_t));
            words[16] = 1u;
            words[17] = static_cast<std::uint32_t>(
                kDensityCellStartBase * sizeof(std::uint32_t));
            words[18] = static_cast<std::uint32_t>(kDensityCellCount);
            words[19] = static_cast<std::uint32_t>(
                kDensityCellEndBase * sizeof(std::uint32_t));
            words[20] = static_cast<std::uint32_t>(kDensityCellCount);
            words[kDensityOutputOffset] = 0xdeadbeefu;
            words[kDensityRemappedOutputOffset] = 0xdeadbeefu;
            words[kDensitySortedPositionsOffset] = 0x3f800000u;
            words[kDensitySortedPositionsOffset + 1u] = 0x40000000u;
            words[kDensitySortedPositionsOffset + 2u] = 0x40400000u;
            words[kDensitySortedPositionsOffset + 3u] = 0u;
            std::fill_n(
                words + kDensityParticleHashOffset,
                64u,
                0xffffffffu);
            words[kDensityParticleHashOffset] = kExpectedKernelHash;
            words[kDensityMappingOffset] = 0u;
            std::fill_n(
                words + kDensityCellStartBase,
                kDensityCellCount,
                0xffffffffu);
            std::fill_n(
                words + kDensityCellEndBase,
                kDensityCellCount,
                0xffffffffu);
            words[kDensityCellStartBase + kExpectedKernelHash] = 0u;
            words[kDensityCellEndBase + kExpectedKernelHash] = 1u;

            auto *constants = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) + density_constant_offset);
            constants[13] = 0x40000000u;
            constants[18] = 0x3f800000u;
            constants[21] = 0x3f800000u;
            auto *metadata = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) +
                density_texture_metadata_offset);
            metadata[0] =
                static_cast<std::uint32_t>(kDensitySortedPositionsOffset);
            metadata[1] = 64u;
            metadata[2] = 4u;
            metadata[3] =
                static_cast<std::uint32_t>(kDensityParticleHashOffset);
            metadata[4] = 64u;
            metadata[5] = 1u;
            metadata[6] = static_cast<std::uint32_t>(kDensityCellStartBase);
            metadata[7] = static_cast<std::uint32_t>(kDensityCellCount);
            metadata[8] = 1u;
            metadata[9] = static_cast<std::uint32_t>(kDensityCellEndBase);
            metadata[10] = static_cast<std::uint32_t>(kDensityCellCount);
            metadata[11] = 1u;
        } else if (reorder_fixture_mode) {
            constexpr std::array<std::uint32_t, 9> pointers = {
                2048u, 2112u, 4096u, 65536u, 768u,
                2176u, 256u, 512u, 2240u,
            };
            for (std::size_t parameter = 0u; parameter < pointers.size(); ++parameter) {
                words[parameter * 2u] = pointers[parameter];
                words[parameter * 2u + 1u] = 1u;
            }
            words[18] = 1u;
            words[192] = 0x3f800000u;
            words[193] = 0x40000000u;
            words[194] = 0x40400000u;
            words[195] = 0u;
            words[544] = 0x40800000u;
            words[545] = 0x40a00000u;
            words[546] = 0x40c00000u;
            words[547] = 0x40e00000u;
            words[kKernelHashOutputOffset] = kExpectedKernelHash;
            words[kKernelIndexOutputOffset] = 0u;
            words[560] = 0u;
            std::fill_n(words + kReorderedPositionOffset, 4u, 0xdeadbeefu);
            std::fill_n(words + kReorderedVelocityOffset, 4u, 0xdeadbeefu);
            words[kReorderCellStartBase + kExpectedKernelHash] = 0xdeadbeefu;
            words[kReorderCellEndBase + kExpectedKernelHash] = 0xdeadbeefu;
        } else if (kernel_fixture_mode) {
            words[0] = 256u;
            words[1] = 1u;
            words[2] = 512u;
            words[3] = 1u;
            words[4] = 768u;
            words[5] = 1u;
            words[6] = 1024u;
            words[7] = 1u;
            words[8] = 1280u;
            words[9] = 1u;
            words[10] = 1536u;
            words[11] = 1u;
            words[12] = 1u;
            words[kKernelHashOutputOffset] = 0xdeadbeefu;
            words[kKernelIndexOutputOffset] = 0xdeadbeefu;
            words[192] = 0x3f800000u;
            words[193] = 0x40000000u;
            words[194] = 0x40400000u;
            words[256] = 0u;
            words[257] = 1u;
            words[320] = 1u;
            words[385] = 0x3f800000u;
        } else if (constant_fixture_mode) {
            words[0] = static_cast<std::uint32_t>(
                kConstantOutputOffset * sizeof(std::uint32_t));
            words[4] = kExpectedConstant[0];
            words[5] = kExpectedConstant[1];
            words[kConstantOutputOffset] = 0xdeadbeefu;
            words[kConstantOutputOffset + 1u] = 0xdeadbeefu;
        } else if (selected_texture_mode) {
            const std::size_t output_offset = texture_fixture_mode
                ? kTextureOutputOffset
                : kTextureFloatOutputOffset;
            words[0] = static_cast<std::uint32_t>(
                output_offset * sizeof(std::uint32_t));
            for (std::uint32_t row = 0u; row < 8u; ++row) {
                words[kTextureInputOffset + row] =
                    0x10000000u + row * 0x100u;
                for (std::uint32_t component = 0u; component < 4u; ++component) {
                    words[kTextureFloatInputOffset + row * 4u + component] =
                        kExpectedFloatTextureInput[row][component];
                }
            }
            std::fill_n(words + output_offset, 12u * 4u, 0xdeadbeefu);
            auto *metadata = reinterpret_cast<std::uint32_t *>(
                static_cast<std::uint8_t *>(mapped) + texture_metadata_offset);
            metadata[0] = static_cast<std::uint32_t>(kTextureInputOffset);
            metadata[1] = 8u;
            metadata[2] = 1u;
            metadata[3] = static_cast<std::uint32_t>(kTextureFloatInputOffset);
            metadata[4] = 8u;
            metadata[5] = 4u;
        } else {
            words[0] = kInput;
            words[1] = 0xdeadbeefu;
        }
        if (oracle_mode) {
            for (std::uint32_t row = 0; row < 8; ++row) {
                for (std::uint32_t component = 0; component < 4; ++component) {
                    words[kTextureInputOffset + row * 4u + component] =
                        0x10000000u * (component + 1u) + row * 0x100u + component;
                }
            }
            for (std::uint32_t row = 0; row < kExpectedFloatTextureInput.size(); ++row) {
                for (std::uint32_t component = 0; component < 4u; ++component) {
                    words[kTextureFloatInputOffset + row * 4u + component] =
                        kExpectedFloatTextureInput[row][component];
                }
            }
            words[4] = kExpectedConstant[0];
            words[5] = kExpectedConstant[1];
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

        const std::uint32_t binding_count =
            selected_texture_mode || density_fixture_mode ||
                    force_fixture_mode || collision_fixture_mode
                ? 4u
                : ptx_oracle_mode || constant_fixture_mode ||
                          deformable_fixture_mode || solver_fixture_mode
                    ? 3u
                    : kernel_fixture_mode || reorder_fixture_mode ||
                              constraints_fixture_mode ||
                              dynamic_response_fixture_mode
                        ? 2u
                        : 1u;
        std::array<VkDescriptorSetLayoutBinding, 4> bindings = {};
        for (std::uint32_t binding_index = 0;
             binding_index < binding_count;
             ++binding_index) {
            bindings[binding_index] = {
                binding_index,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                1,
                VK_SHADER_STAGE_COMPUTE_BIT,
                nullptr,
            };
        }
        VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            binding_count,
            bindings.data(),
        };
        check(
            vkCreateDescriptorSetLayout(
                device,
                &descriptor_layout_info,
                nullptr,
                &descriptor_layout),
            "vkCreateDescriptorSetLayout");

        VkDescriptorPoolSize pool_size = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding_count,
        };
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
        std::array<VkDescriptorBufferInfo, 4> descriptor_buffer_infos = {};
        std::array<VkWriteDescriptorSet, 4> descriptor_writes = {};
        for (std::uint32_t binding_index = 0;
             binding_index < binding_count;
             ++binding_index) {
            descriptor_buffer_infos[binding_index] = {
                device_buffer.buffer,
                0,
                kBufferBytes,
            };
            if (collision_fixture_mode && binding_index == 2u) {
                descriptor_buffer_infos[binding_index].offset =
                    collision_constant_offset;
                descriptor_buffer_infos[binding_index].range =
                    kCollisionConstantBytes;
            } else if (collision_fixture_mode && binding_index == 3u) {
                descriptor_buffer_infos[binding_index].offset =
                    collision_texture_metadata_offset;
                descriptor_buffer_infos[binding_index].range =
                    kCollisionTextureMetadataBytes;
            } else if (solver_fixture_mode && binding_index == 2u) {
                descriptor_buffer_infos[binding_index].offset =
                    solver_constant_offset;
                descriptor_buffer_infos[binding_index].range =
                    kSolverConstantBytes;
            } else if (deformable_fixture_mode && binding_index == 2u) {
                descriptor_buffer_infos[binding_index].offset =
                    deformable_constant_offset;
                descriptor_buffer_infos[binding_index].range =
                    kDeformableConstantBytes;
            } else if ((density_fixture_mode || force_fixture_mode) &&
                binding_index == 2u) {
                descriptor_buffer_infos[binding_index].offset =
                    density_constant_offset;
                descriptor_buffer_infos[binding_index].range =
                    force_fixture_mode
                        ? kForceConstantBytes
                        : kDensityConstantBytes;
            } else if ((density_fixture_mode || force_fixture_mode) &&
                       binding_index == 3u) {
                descriptor_buffer_infos[binding_index].offset =
                    density_texture_metadata_offset;
                descriptor_buffer_infos[binding_index].range =
                    force_fixture_mode
                        ? kForceTextureMetadataBytes
                        : kDensityTextureMetadataBytes;
            } else if (selected_texture_mode && binding_index == 3u) {
                descriptor_buffer_infos[binding_index].offset =
                    texture_metadata_offset;
                descriptor_buffer_infos[binding_index].range =
                    kTextureMetadataBytes;
            }
            descriptor_writes[binding_index] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                binding_index,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &descriptor_buffer_infos[binding_index],
                nullptr,
            };
        }
        vkUpdateDescriptorSets(
            device, binding_count, descriptor_writes.data(), 0, nullptr);

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
        if (dynamic_response_fixture_mode) {
            std::copy_n(
                words + kDynamicResponseNormalOffset,
                result.dynamic_response_normal.size(),
                result.dynamic_response_normal.begin());
            std::copy_n(
                words + kDynamicResponseAveragedPositionOffset,
                result.dynamic_response_position.size(),
                result.dynamic_response_position.begin());
            std::copy_n(
                words + kDynamicResponseParticleVelocityOffset,
                result.dynamic_response_velocity.size(),
                result.dynamic_response_velocity.begin());
            std::copy_n(
                words + kDynamicResponseOutputOffset,
                result.dynamic_response_output.size(),
                result.dynamic_response_output.begin());
            result.dynamic_response_weight =
                words[kDynamicResponseWeightOffset];
            result.dynamic_response_dest_id =
                words[kDynamicResponseDestIdOffset];
            result.dynamic_response_flags =
                words[kDynamicResponseFlagsOffset];
            const auto guards_match =
                [&](std::size_t offset, std::size_t count) {
                    return words[offset - 1u] == kGuardBefore &&
                        words[offset + count] == kGuardAfter;
                };
            result.dynamic_response_guards_intact =
                guards_match(kDynamicResponseWorkUnitOffset, 4u) &&
                guards_match(kDynamicResponseStartAddrsOffset, 32u) &&
                guards_match(kDynamicResponseParametersOffset, 40u);
            for (const std::size_t offset : {
                     kDynamicResponseNormalOffset,
                     kDynamicResponseAccumulatedPositionOffset,
                     kDynamicResponseBodyVelocityOffset,
                     kDynamicResponseAveragedPositionOffset,
                     kDynamicResponseParticleVelocityOffset,
                     kDynamicResponseOutputOffset,
                 }) {
                result.dynamic_response_guards_intact =
                    result.dynamic_response_guards_intact &&
                    guards_match(offset, 4u);
            }
            for (const std::size_t offset : {
                     kDynamicResponseCountOffset,
                     kDynamicResponseWeightOffset,
                     kDynamicResponseDestIdOffset,
                     kDynamicResponseSourceIdOffset,
                     kDynamicResponseFlagsOffset,
                     kDynamicResponseErrorOffset,
                 }) {
                result.dynamic_response_guards_intact =
                    result.dynamic_response_guards_intact &&
                    guards_match(offset, 1u);
            }
            const std::array<std::uint32_t, 4> zero_vector = {};
            result.pass =
                result.dynamic_response_normal == zero_vector &&
                result.dynamic_response_position ==
                    kExpectedDynamicResponsePosition &&
                result.dynamic_response_velocity ==
                    kExpectedDynamicResponseVelocity &&
                result.dynamic_response_output ==
                    kExpectedDynamicResponseOutput &&
                result.dynamic_response_weight == 0x3f800000u &&
                result.dynamic_response_dest_id == 0x12345678u &&
                result.dynamic_response_flags == 0u &&
                std::equal(
                    words + kDynamicResponseAccumulatedPositionOffset,
                    words + kDynamicResponseAccumulatedPositionOffset + 4u,
                    zero_vector.begin()) &&
                std::equal(
                    words + kDynamicResponseBodyVelocityOffset,
                    words + kDynamicResponseBodyVelocityOffset + 4u,
                    zero_vector.begin()) &&
                words[kDynamicResponseCountOffset] == 0u &&
                words[kDynamicResponseSourceIdOffset] == 0x12345678u &&
                words[kDynamicResponseParametersOffset + 29u] ==
                    0x3f000000u &&
                words[kDynamicResponseParametersOffset + 34u] == 1u &&
                words[kDynamicResponseErrorOffset] == 0xfeedfaceu &&
                result.dynamic_response_guards_intact;
        } else if (constraints_fixture_mode) {
            std::copy_n(
                words + kConstraintsOutputPositionOffset,
                result.constraints_position.size(),
                result.constraints_position.begin());
            std::copy_n(
                words + kConstraintsOutputVelocityOffset,
                result.constraints_velocity.size(),
                result.constraints_velocity.begin());
            result.constraints_weight = words[kConstraintsWeightOffset];
            const auto guards_match =
                [&](std::size_t offset, std::size_t count) {
                    return words[offset - 1u] == kGuardBefore &&
                        words[offset + count] == kGuardAfter;
                };
            result.constraints_guards_intact =
                guards_match(kConstraintsWorkUnitOffset, 4u) &&
                guards_match(kConstraintsStartAddrsOffset, 32u) &&
                guards_match(kConstraintsParametersOffset, 40u) &&
                guards_match(kConstraintsKeyOffset, 2u) &&
                guards_match(kConstraintsInputPositionOffset, 4u) &&
                guards_match(kConstraintsInputMotionOffset, 4u) &&
                guards_match(kConstraintsInputOtherOffset, 4u) &&
                guards_match(kConstraintsWeightOffset, 1u) &&
                guards_match(kConstraintsOutputPositionOffset, 4u) &&
                guards_match(kConstraintsOutputVelocityOffset, 4u) &&
                guards_match(kConstraintsErrorOffset, 1u);
            for (const std::size_t offset : kConstraintsVectorOutputOffsets) {
                result.constraints_guards_intact =
                    result.constraints_guards_intact &&
                    guards_match(offset, 4u);
            }
            for (const std::size_t offset : kConstraintsScalarOutputOffsets) {
                result.constraints_guards_intact =
                    result.constraints_guards_intact &&
                    guards_match(offset, 1u);
            }
            const std::array<std::uint32_t, 4> zero_vector = {};
            result.pass =
                result.constraints_position == kExpectedConstraintsPosition &&
                result.constraints_velocity == kExpectedConstraintsVelocity &&
                result.constraints_weight == 0x3f800000u &&
                words[kConstraintsParametersOffset] == 0x3f000000u &&
                words[kConstraintsKeyOffset] == 0x0000ffffu &&
                words[kConstraintsKeyOffset + 1u] == 0x12345678u &&
                words[kConstraintsErrorOffset] == 0xfeedfaceu &&
                result.constraints_guards_intact;
            for (const std::size_t offset : kConstraintsVectorOutputOffsets) {
                result.pass =
                    result.pass &&
                    std::equal(
                        words + offset,
                        words + offset + zero_vector.size(),
                        zero_vector.begin());
            }
            for (const std::size_t offset : kConstraintsScalarOutputOffsets) {
                result.pass = result.pass && words[offset] == 0u;
            }
        } else if (collision_fixture_mode) {
            std::copy_n(
                words + kCollisionOutputPositionOffset,
                result.collision_position.size(),
                result.collision_position.begin());
            std::copy_n(
                words + kCollisionOutputMotionOffset,
                result.collision_motion.size(),
                result.collision_motion.begin());
            const std::array<std::size_t, 4> scalar_offsets = {
                kCollisionScalar48Offset,
                kCollisionScalar52Offset,
                kCollisionScalar56Offset,
                kCollisionScalar60Offset,
            };
            for (std::size_t index = 0u;
                 index < scalar_offsets.size();
                 ++index) {
                result.collision_scalars[index] = words[scalar_offsets[index]];
            }
            const auto guards_match =
                [&](std::size_t offset, std::size_t count) {
                    return words[offset - 1u] == kGuardBefore &&
                        words[offset + count] == kGuardAfter;
                };
            result.collision_guards_intact =
                guards_match(kCollisionBatchOffset, 2u) &&
                guards_match(kCollisionInputPositionOffset, 4u) &&
                guards_match(kCollisionInputMotionOffset, 4u) &&
                guards_match(kCollisionOutput8Offset, 4u) &&
                guards_match(kCollisionOutput12Offset, 4u) &&
                guards_match(kCollisionOutput20Offset, 1u) &&
                guards_match(kCollisionFlagsOffset, 1u) &&
                guards_match(kCollisionOutputPositionOffset, 4u) &&
                guards_match(kCollisionOutputMotionOffset, 4u) &&
                guards_match(kCollisionVector40Offset, 4u) &&
                guards_match(kCollisionVector44Offset, 4u) &&
                guards_match(kCollisionScalar48Offset, 1u) &&
                guards_match(kCollisionScalar52Offset, 1u) &&
                guards_match(kCollisionScalar56Offset, 1u) &&
                guards_match(kCollisionScalar60Offset, 1u) &&
                guards_match(kCollisionVector64Offset, 4u) &&
                guards_match(kCollisionActiveOffset, 1u);
            const std::array<std::uint32_t, 4> zero_vector = {};
            result.pass =
                result.collision_position == kExpectedCollisionPosition &&
                result.collision_motion == kExpectedCollisionMotion &&
                result.collision_scalars == kExpectedCollisionScalars &&
                std::equal(
                    words + kCollisionOutput8Offset,
                    words + kCollisionOutput8Offset + 4u,
                    zero_vector.begin()) &&
                std::equal(
                    words + kCollisionOutput12Offset,
                    words + kCollisionOutput12Offset + 4u,
                    zero_vector.begin()) &&
                words[kCollisionOutput20Offset] == 0xbf800000u &&
                words[kCollisionFlagsOffset] == 0u &&
                std::equal(
                    words + kCollisionVector40Offset,
                    words + kCollisionVector40Offset + 4u,
                    zero_vector.begin()) &&
                std::equal(
                    words + kCollisionVector44Offset,
                    words + kCollisionVector44Offset + 4u,
                    zero_vector.begin()) &&
                std::equal(
                    words + kCollisionVector64Offset,
                    words + kCollisionVector64Offset + 4u,
                    zero_vector.begin()) &&
                words[kCollisionActiveOffset] == 0u &&
                result.collision_guards_intact;
        } else if (solver_fixture_mode) {
            std::copy_n(
                words + kSolverAOffset,
                result.solver_a.size(),
                result.solver_a.begin());
            std::copy_n(
                words + kSolverPositionOffset,
                result.solver_position.size(),
                result.solver_position.begin());
            std::copy_n(
                words + kSolverOutputOffset,
                result.solver_output.size(),
                result.solver_output.begin());
            const auto guards_match =
                [&](std::size_t offset, std::size_t count) {
                    return words[offset - 1u] == kGuardBefore &&
                        words[offset + count] == kGuardAfter;
                };
            result.solver_guards_intact =
                guards_match(kSolverAOffset, 4u) &&
                guards_match(kSolverBOffset, 4u) &&
                guards_match(kSolverCOffset, 4u) &&
                guards_match(kSolverPositionOffset, 4u) &&
                guards_match(kSolverFlagsOffset, 1u) &&
                guards_match(kSolverPackedOffset, 1u) &&
                guards_match(kSolverOutputOffset, 16u);
            result.pass =
                result.solver_a == kExpectedSolverA &&
                result.solver_position == kExpectedSolverPosition &&
                result.solver_output == kExpectedSolverOutput &&
                words[kSolverFlagsOffset] == 0x00000400u &&
                words[kSolverPackedOffset] == 0xabcdef01u &&
                result.solver_guards_intact;
        } else if (deformable_fixture_mode) {
            std::copy_n(
                words + kDeformablePositionOffset,
                result.deformable_position.size(),
                result.deformable_position.begin());
            std::copy_n(
                words + kDeformableMotionOffset,
                result.deformable_motion.size(),
                result.deformable_motion.begin());
            result.deformable_flags = words[kDeformableFlagsOffset];
            result.deformable_guards_intact =
                words[kDeformablePackedBaseOffset] == kGuardBefore &&
                words[kDeformablePackedBaseOffset + 1u] == 0xfeedfaceu &&
                words[kDeformablePackedBaseOffset + 2u] == kGuardAfter &&
                words[kDeformablePositionOffset - 1u] == kGuardBefore &&
                words[kDeformablePositionOffset + 4u] == kGuardAfter &&
                words[kDeformableMotionOffset - 1u] == kGuardBefore &&
                words[kDeformableMotionOffset + 4u] == kGuardAfter &&
                words[kDeformableFlagsOffset - 1u] == kGuardBefore &&
                words[kDeformableFlagsOffset + 1u] == kGuardAfter &&
                words[kDeformableIgnoredOffset - 1u] == kGuardBefore &&
                words[kDeformableIgnoredOffset] == 0xcafebabeu &&
                words[kDeformableIgnoredOffset + 1u] == kGuardAfter;
            result.pass =
                result.deformable_position == kExpectedDeformablePosition &&
                result.deformable_motion == kExpectedDeformableMotion &&
                result.deformable_flags == 0u &&
                result.deformable_guards_intact;
        } else if (force_fixture_mode) {
            std::copy_n(
                words + kForceOutputOffset,
                result.force.size(),
                result.force.begin());
            result.pass = result.force == kExpectedForce;
        } else if (density_fixture_mode) {
            result.density = words[kDensityOutputOffset];
            result.remapped_density = words[kDensityRemappedOutputOffset];
            result.pass =
                result.density == kExpectedDensityBits &&
                result.remapped_density == kExpectedDensityBits;
        } else if (reorder_fixture_mode) {
            std::copy_n(
                words + kReorderedPositionOffset,
                result.reordered_position.size(),
                result.reordered_position.begin());
            std::copy_n(
                words + kReorderedVelocityOffset,
                result.reordered_velocity.size(),
                result.reordered_velocity.begin());
            result.cell_start =
                words[kReorderCellStartBase + kExpectedKernelHash];
            result.cell_end =
                words[kReorderCellEndBase + kExpectedKernelHash];
            result.pass =
                result.reordered_position == kExpectedReorderedPosition &&
                result.reordered_velocity == kExpectedReorderedVelocity &&
                result.cell_start == 0u &&
                result.cell_end == 1u;
        } else if (kernel_fixture_mode) {
            result.kernel_hash = words[kKernelHashOutputOffset];
            result.kernel_index = words[kKernelIndexOutputOffset];
            result.pass =
                result.kernel_hash == kExpectedKernelHash &&
                result.kernel_index == 0u;
        } else if (constant_fixture_mode) {
            std::copy_n(
                words + kConstantOutputOffset,
                result.constant.size(),
                result.constant.begin());
            result.pass = result.constant == kExpectedConstant;
        } else if (selected_texture_mode) {
            auto &texture_result = texture_fixture_mode
                ? result.texture
                : result.texture_float;
            const std::size_t output_offset = texture_fixture_mode
                ? kTextureOutputOffset
                : kTextureFloatOutputOffset;
            result.pass = true;
            for (std::size_t row = 0u; row < texture_result.size(); ++row) {
                std::copy_n(
                    words + output_offset + row * 4u,
                    4u,
                    texture_result[row].begin());
                const int coordinate = static_cast<int>(row) - 2;
                for (std::size_t component = 0u; component < 4u; ++component) {
                    const std::uint32_t expected =
                        coordinate < 0 || coordinate >= 8
                            ? 0u
                            : texture_fixture_mode
                                ? component == 0u
                                    ? 0x10000000u +
                                          static_cast<std::uint32_t>(coordinate) *
                                              0x100u
                                    : 0u
                                : kExpectedFloatTextureInput[
                                      static_cast<std::size_t>(coordinate)][component];
                    result.pass =
                        result.pass && texture_result[row][component] == expected;
                }
            }
        } else {
            result.pass = !require_memory_result ||
                (result.input_after == kInput && result.result == kExpected);
        }
        if (oracle_mode) {
            std::copy_n(words + kMathOffset, result.math.size(), result.math.begin());
            std::copy_n(words + kBarrierOffset, result.barrier.size(), result.barrier.begin());
            std::copy_n(words + kLocalOffset, result.local.size(), result.local.begin());
            for (std::size_t row = 0; row < result.texture.size(); ++row) {
                std::copy_n(
                    words + kTextureOutputOffset + row * 4u,
                    4u,
                    result.texture[row].begin());
            }
            for (std::size_t row = 0; row < result.texture_float.size(); ++row) {
                std::copy_n(
                    words + kTextureFloatOutputOffset + row * 4u,
                    4u,
                    result.texture_float[row].begin());
            }
            if (ptx_oracle_mode) {
                std::copy_n(
                    words + kConstantOutputOffset,
                    result.constant.size(),
                    result.constant.begin());
                result.pass = result.pass && result.constant == kExpectedConstant;
            }
            result.pass = result.pass &&
                std::equal(result.math.begin(), result.math.end(), kExpectedMath.begin());
            for (std::size_t index = 0; index < result.barrier.size(); ++index) {
                const std::uint32_t expected =
                    index + 1u == result.barrier.size() ? 1u :
                    static_cast<std::uint32_t>(index + 2u);
                result.pass = result.pass && result.barrier[index] == expected;
            }
            for (std::size_t index = 0; index < result.local.size(); ++index) {
                result.pass =
                    result.pass &&
                    result.local[index] ==
                        0x5a000000u + static_cast<std::uint32_t>(index);
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
            for (std::size_t row = 0; row < result.texture_float.size(); ++row) {
                const int coordinate = static_cast<int>(row) - 2;
                for (std::size_t component = 0; component < 4u; ++component) {
                    const std::uint32_t expected =
                        coordinate < 0 || coordinate >= 8
                            ? 0u
                            : kExpectedFloatTextureInput[
                                  static_cast<std::size_t>(coordinate)][component];
                    result.pass =
                        result.pass &&
                        result.texture_float[row][component] == expected;
                }
            }
        }
        if (!result.pass) {
            result.error = dynamic_response_fixture_mode
                ? "fluid dynamic response fixture result mismatch"
                : constraints_fixture_mode
                    ? "fluid collision constraints fixture result mismatch"
                    : collision_fixture_mode
                    ? "deformable collision fixture result mismatch"
                    : solver_fixture_mode
                        ? "deformable solver fixture result mismatch"
                        : deformable_fixture_mode
                            ? "deformable update fixture result mismatch"
                            : force_fixture_mode
                                ? "force kernel fixture result mismatch"
                                : density_fixture_mode
                                    ? "density kernel fixture result mismatch"
                                    : reorder_fixture_mode
                                        ? "reorder kernel fixture result mismatch"
                                        : kernel_fixture_mode
                                            ? "selected kernel fixture result mismatch"
                                            : constant_fixture_mode
                                                ? "constant kernel fixture result mismatch"
                                                : selected_texture_mode
                                                    ? "selected texture kernel fixture result mismatch"
                                                    : oracle_mode
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

DeviceResult run_deformable_sequence_device(
    std::uint32_t device_index,
    VkPhysicalDevice physical_device,
    const std::array<std::vector<std::uint32_t>, 3> &shaders)
{
    DeviceResult result;
    result.index = device_index;
    vkGetPhysicalDeviceProperties(physical_device, &result.properties);

    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        4u, result.properties.limits.minStorageBufferOffsetAlignment);
    const VkDeviceSize update_parameter_offset =
        align_up(kSequencePayloadBytes, alignment);
    const VkDeviceSize collision_parameter_offset =
        align_up(update_parameter_offset + 3u * sizeof(std::uint32_t), alignment);
    const VkDeviceSize solver_parameter_offset =
        align_up(collision_parameter_offset + sizeof(std::uint32_t), alignment);
    const VkDeviceSize update_constant_offset =
        align_up(solver_parameter_offset + 4u * sizeof(std::uint32_t), alignment);
    const VkDeviceSize collision_constant_offset =
        align_up(update_constant_offset + kDeformableConstantBytes, alignment);
    const VkDeviceSize solver_constant_offset =
        align_up(collision_constant_offset + kCollisionConstantBytes, alignment);
    const VkDeviceSize texture_metadata_offset =
        align_up(solver_constant_offset + kSolverConstantBytes, alignment);
    const VkDeviceSize copy_bytes =
        texture_metadata_offset + kCollisionTextureMetadataBytes;

    VkDevice device = VK_NULL_HANDLE;
    BufferAllocation device_buffer;
    BufferAllocation staging_buffer;
    void *mapped = nullptr;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 3> descriptor_sets = {};
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    std::array<VkShaderModule, 3> shader_modules = {};
    std::array<VkPipeline, 3> pipelines = {};
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    try {
        std::uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device,
            &queue_family_count,
            queue_families.data());
        std::uint32_t queue_family = UINT32_MAX;
        for (std::uint32_t index = 0u; index < queue_family_count; ++index) {
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
        check(
            vkCreateDevice(physical_device, &device_info, nullptr, &device),
            "vkCreateDevice");
        volkLoadDevice(device);
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        device_buffer = create_buffer(
            physical_device,
            device,
            kBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            0,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        staging_buffer = create_buffer(
            physical_device,
            device,
            kBufferBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        result.device_memory_local = device_buffer.device_local;
        result.staging_coherent = staging_buffer.host_coherent;

        check(
            vkMapMemory(
                device,
                staging_buffer.memory,
                0,
                staging_buffer.allocation_size,
                0,
                &mapped),
            "vkMapMemory");
        auto *bytes = static_cast<std::uint8_t *>(mapped);
        auto *words = reinterpret_cast<std::uint32_t *>(bytes);
        std::memset(mapped, 0, static_cast<std::size_t>(copy_bytes));
        const auto initialize = [&](std::size_t offset,
                                    std::initializer_list<std::uint32_t> values) {
            words[offset - 1u] = kGuardBefore;
            std::copy(values.begin(), values.end(), words + offset);
            words[offset + values.size()] = kGuardAfter;
        };
        const auto sentinel_vector = [&initialize](std::size_t offset) {
            initialize(
                offset,
                {0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu});
        };
        const auto sentinel_scalar = [&initialize](std::size_t offset) {
            initialize(offset, {0xfeedfaceu});
        };
        const auto byte_offset = [](std::size_t offset) {
            return static_cast<std::uint32_t>(
                offset * sizeof(std::uint32_t));
        };

        initialize(kSequenceUpdatePackedOffset, {0xfeedfaceu});
        initialize(
            kSequenceUpdatePositionOffset,
            {0x3f800000u, 0x40000000u, 0x40400000u, 0u});
        initialize(
            kSequenceUpdateMotionOffset,
            {0x40800000u, 0x40a00000u, 0x40c00000u, 0x40e00000u});
        initialize(kSequenceUpdateFlagsOffset, {0x00000014u});
        initialize(kSequenceUpdateIgnoredOffset, {0xcafebabeu});
        initialize(kSequenceCollisionBatchOffset, {0x00010000u, 0u});
        for (const std::size_t offset : {
                 kSequenceCollisionOutput8Offset,
                 kSequenceCollisionOutput12Offset,
                 kSequenceCollisionOutputPositionOffset,
                 kSequenceCollisionOutputMotionOffset,
                 kSequenceCollisionVector40Offset,
                 kSequenceCollisionVector44Offset,
                 kSequenceCollisionVector64Offset,
             }) {
            sentinel_vector(offset);
        }
        for (const std::size_t offset : {
                 kSequenceCollisionOutput20Offset,
                 kSequenceCollisionScalar48Offset,
                 kSequenceCollisionScalar52Offset,
                 kSequenceCollisionScalar56Offset,
                 kSequenceCollisionScalar60Offset,
             }) {
            sentinel_scalar(offset);
        }
        initialize(kSequenceCollisionActiveOffset, {1u});
        initialize(
            kSequenceSolverAOffset,
            {0x41200000u, 0x41300000u, 0x41400000u, 0u});
        initialize(
            kSequenceSolverBOffset,
            {0x41a00000u, 0x41a80000u, 0x41b00000u, 0u});
        initialize(
            kSequenceSolverCOffset,
            {0x41f00000u, 0x41f80000u, 0x42000000u, 0u});
        initialize(kSequenceSolverFlagsOffset, {0x00000400u});
        initialize(kSequenceSolverPackedOffset, {0xabcdef01u});
        words[kSequenceSolverOutputOffset - 1u] = kGuardBefore;
        std::fill_n(words + kSequenceSolverOutputOffset, 16u, 0u);
        words[kSequenceSolverOutputOffset + 16u] = kGuardAfter;

        auto *update_parameters =
            reinterpret_cast<std::uint32_t *>(bytes + update_parameter_offset);
        update_parameters[0] = byte_offset(kSequenceUpdatePackedOffset);
        update_parameters[1] = 1u;
        update_parameters[2] = byte_offset(kSequenceUpdateIgnoredOffset);
        auto *collision_parameters =
            reinterpret_cast<std::uint32_t *>(bytes + collision_parameter_offset);
        collision_parameters[0] = 0u;
        auto *solver_parameters =
            reinterpret_cast<std::uint32_t *>(bytes + solver_parameter_offset);
        solver_parameters[0] = 0u;
        solver_parameters[1] = 0u;
        solver_parameters[2] = byte_offset(kSequenceSolverOutputOffset);
        solver_parameters[3] = 0u;

        auto *update_constants =
            reinterpret_cast<std::uint32_t *>(bytes + update_constant_offset);
        update_constants[0] = 0x3e800000u;
        update_constants[1] = 0xbf000000u;
        update_constants[2] = 0x3f800000u;
        update_constants[7] = 1u;
        update_constants[15] = 0x3f000000u;
        update_constants[16] = byte_offset(kSequenceUpdatePositionOffset);
        update_constants[17] = byte_offset(kSequenceUpdateMotionOffset);
        update_constants[22] = byte_offset(kSequenceUpdateFlagsOffset);

        auto *collision_constants =
            reinterpret_cast<std::uint32_t *>(bytes + collision_constant_offset);
        collision_constants[0] = byte_offset(kSequenceUpdatePositionOffset);
        collision_constants[1] = byte_offset(kSequenceUpdateMotionOffset);
        collision_constants[2] = byte_offset(kSequenceCollisionOutput8Offset);
        collision_constants[3] = byte_offset(kSequenceCollisionOutput12Offset);
        collision_constants[5] = byte_offset(kSequenceCollisionOutput20Offset);
        collision_constants[6] = byte_offset(kSequenceUpdateFlagsOffset);
        collision_constants[8] =
            byte_offset(kSequenceCollisionOutputPositionOffset);
        collision_constants[9] =
            byte_offset(kSequenceCollisionOutputMotionOffset);
        collision_constants[10] =
            byte_offset(kSequenceCollisionVector40Offset);
        collision_constants[11] =
            byte_offset(kSequenceCollisionVector44Offset);
        collision_constants[12] =
            byte_offset(kSequenceCollisionScalar48Offset);
        collision_constants[13] =
            byte_offset(kSequenceCollisionScalar52Offset);
        collision_constants[14] =
            byte_offset(kSequenceCollisionScalar56Offset);
        collision_constants[15] =
            byte_offset(kSequenceCollisionScalar60Offset);
        collision_constants[16] =
            byte_offset(kSequenceCollisionVector64Offset);
        collision_constants[17] =
            byte_offset(kSequenceCollisionActiveOffset);
        collision_constants[4480u + 17u] =
            byte_offset(kSequenceCollisionBatchOffset);

        auto *solver_constants =
            reinterpret_cast<std::uint32_t *>(bytes + solver_constant_offset);
        solver_constants[1] = 1u;
        solver_constants[8] = 0x3f800000u;
        solver_constants[14] = byte_offset(kSequenceSolverAOffset);
        solver_constants[15] =
            byte_offset(kSequenceCollisionOutputPositionOffset);
        solver_constants[16] = byte_offset(kSequenceSolverBOffset);
        solver_constants[17] = byte_offset(kSequenceSolverCOffset);
        solver_constants[18] = byte_offset(kSequenceSolverCOffset);
        solver_constants[21] = byte_offset(kSequenceSolverFlagsOffset);
        solver_constants[22] = byte_offset(kSequenceSolverPackedOffset);

        if (!staging_buffer.host_coherent) {
            VkMappedMemoryRange range = {
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                nullptr,
                staging_buffer.memory,
                0,
                VK_WHOLE_SIZE,
            };
            check(
                vkFlushMappedMemoryRanges(device, 1, &range),
                "vkFlushMappedMemoryRanges");
        }

        std::array<VkDescriptorSetLayoutBinding, 4> bindings = {};
        for (std::uint32_t binding_index = 0u;
             binding_index < bindings.size();
             ++binding_index) {
            bindings[binding_index] = {
                binding_index,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                1,
                VK_SHADER_STAGE_COMPUTE_BIT,
                nullptr,
            };
        }
        VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            static_cast<std::uint32_t>(bindings.size()),
            bindings.data(),
        };
        check(
            vkCreateDescriptorSetLayout(
                device,
                &descriptor_layout_info,
                nullptr,
                &descriptor_layout),
            "vkCreateDescriptorSetLayout");

        VkDescriptorPoolSize pool_size = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            static_cast<std::uint32_t>(
                descriptor_sets.size() * bindings.size()),
        };
        VkDescriptorPoolCreateInfo pool_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr,
            0,
            static_cast<std::uint32_t>(descriptor_sets.size()),
            1,
            &pool_size,
        };
        check(
            vkCreateDescriptorPool(
                device, &pool_info, nullptr, &descriptor_pool),
            "vkCreateDescriptorPool");
        const std::array<VkDescriptorSetLayout, 3> set_layouts = {
            descriptor_layout, descriptor_layout, descriptor_layout,
        };
        VkDescriptorSetAllocateInfo set_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptor_pool,
            static_cast<std::uint32_t>(set_layouts.size()),
            set_layouts.data(),
        };
        check(
            vkAllocateDescriptorSets(
                device, &set_info, descriptor_sets.data()),
            "vkAllocateDescriptorSets");

        const std::array<VkDeviceSize, 3> parameter_offsets = {
            update_parameter_offset,
            collision_parameter_offset,
            solver_parameter_offset,
        };
        const std::array<VkDeviceSize, 3> parameter_ranges = {
            3u * sizeof(std::uint32_t),
            sizeof(std::uint32_t),
            4u * sizeof(std::uint32_t),
        };
        const std::array<VkDeviceSize, 3> constant_offsets = {
            update_constant_offset,
            collision_constant_offset,
            solver_constant_offset,
        };
        const std::array<VkDeviceSize, 3> constant_ranges = {
            kDeformableConstantBytes,
            kCollisionConstantBytes,
            kSolverConstantBytes,
        };
        std::array<VkDescriptorBufferInfo, 12> descriptor_infos = {};
        std::array<VkWriteDescriptorSet, 12> descriptor_writes = {};
        for (std::size_t set_index = 0u;
             set_index < descriptor_sets.size();
             ++set_index) {
            const std::array<VkDescriptorBufferInfo, 4> infos = {{
                {device_buffer.buffer, 0, kBufferBytes},
                {device_buffer.buffer,
                 parameter_offsets[set_index],
                 parameter_ranges[set_index]},
                {device_buffer.buffer,
                 constant_offsets[set_index],
                 constant_ranges[set_index]},
                {device_buffer.buffer,
                 texture_metadata_offset,
                 kCollisionTextureMetadataBytes},
            }};
            for (std::size_t binding_index = 0u;
                 binding_index < infos.size();
                 ++binding_index) {
                const std::size_t descriptor_index =
                    set_index * bindings.size() + binding_index;
                descriptor_infos[descriptor_index] = infos[binding_index];
                descriptor_writes[descriptor_index] = {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    descriptor_sets[set_index],
                    static_cast<std::uint32_t>(binding_index),
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    &descriptor_infos[descriptor_index],
                    nullptr,
                };
            }
        }
        vkUpdateDescriptorSets(
            device,
            static_cast<std::uint32_t>(descriptor_writes.size()),
            descriptor_writes.data(),
            0,
            nullptr);

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
            vkCreatePipelineLayout(
                device,
                &pipeline_layout_info,
                nullptr,
                &pipeline_layout),
            "vkCreatePipelineLayout");
        for (std::size_t index = 0u; index < shaders.size(); ++index) {
            VkShaderModuleCreateInfo shader_info = {
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                nullptr,
                0,
                shaders[index].size() * sizeof(std::uint32_t),
                shaders[index].data(),
            };
            check(
                vkCreateShaderModule(
                    device,
                    &shader_info,
                    nullptr,
                    &shader_modules[index]),
                "vkCreateShaderModule");
            VkPipelineShaderStageCreateInfo stage_info = {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr,
                0,
                VK_SHADER_STAGE_COMPUTE_BIT,
                shader_modules[index],
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
                    &pipelines[index]),
                "vkCreateComputePipelines");
        }

        VkCommandPoolCreateInfo command_pool_info = {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            0,
            queue_family,
        };
        check(
            vkCreateCommandPool(
                device, &command_pool_info, nullptr, &command_pool),
            "vkCreateCommandPool");
        VkCommandBufferAllocateInfo command_buffer_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            command_pool,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1,
        };
        check(
            vkAllocateCommandBuffers(
                device, &command_buffer_info, &command_buffer),
            "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr,
        };
        check(
            vkBeginCommandBuffer(command_buffer, &begin_info),
            "vkBeginCommandBuffer");
        VkBufferCopy copy = {0, 0, copy_bytes};
        vkCmdCopyBuffer(
            command_buffer,
            staging_buffer.buffer,
            device_buffer.buffer,
            1,
            &copy);
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
        VkBufferMemoryBarrier dispatch_barrier = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            device_buffer.buffer,
            0,
            copy_bytes,
        };
        for (std::size_t index = 0u; index < pipelines.size(); ++index) {
            vkCmdBindPipeline(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelines[index]);
            vkCmdBindDescriptorSets(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_layout,
                0,
                1,
                &descriptor_sets[index],
                0,
                nullptr);
            vkCmdDispatch(command_buffer, 1, 1, 1);
            if (index + 1u < pipelines.size()) {
                vkCmdPipelineBarrier(
                    command_buffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0,
                    nullptr,
                    1,
                    &dispatch_barrier,
                    0,
                    nullptr);
            }
        }
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
        vkCmdCopyBuffer(
            command_buffer,
            device_buffer.buffer,
            staging_buffer.buffer,
            1,
            &copy);
        check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

        VkFenceCreateInfo fence_info = {
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0,
        };
        check(
            vkCreateFence(device, &fence_info, nullptr, &fence),
            "vkCreateFence");
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
        check(
            vkQueueSubmit(queue, 1, &submit_info, fence),
            "vkQueueSubmit");
        check(
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");
        if (!staging_buffer.host_coherent) {
            VkMappedMemoryRange range = {
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                nullptr,
                staging_buffer.memory,
                0,
                VK_WHOLE_SIZE,
            };
            check(
                vkInvalidateMappedMemoryRanges(device, 1, &range),
                "vkInvalidateMappedMemoryRanges");
        }

        std::copy_n(
            words + kSequenceUpdateMotionOffset,
            result.sequence_update_motion.size(),
            result.sequence_update_motion.begin());
        std::copy_n(
            words + kSequenceCollisionOutputMotionOffset,
            result.sequence_collision_motion.size(),
            result.sequence_collision_motion.begin());
        const std::array<std::size_t, 4> scalar_offsets = {
            kSequenceCollisionScalar48Offset,
            kSequenceCollisionScalar52Offset,
            kSequenceCollisionScalar56Offset,
            kSequenceCollisionScalar60Offset,
        };
        for (std::size_t index = 0u; index < scalar_offsets.size(); ++index) {
            result.sequence_collision_scalars[index] =
                words[scalar_offsets[index]];
        }
        std::copy_n(
            words + kSequenceSolverAOffset,
            result.sequence_solver_a.size(),
            result.sequence_solver_a.begin());
        std::copy_n(
            words + kSequenceCollisionOutputPositionOffset,
            result.sequence_solver_position.size(),
            result.sequence_solver_position.begin());
        std::copy_n(
            words + kSequenceSolverOutputOffset,
            result.sequence_solver_output.size(),
            result.sequence_solver_output.begin());

        const auto guards_match =
            [words](std::size_t offset, std::size_t count) {
                return words[offset - 1u] == kGuardBefore &&
                    words[offset + count] == kGuardAfter;
            };
        const std::array<std::pair<std::size_t, std::size_t>, 25> ranges = {{
            {kSequenceUpdatePackedOffset, 1u},
            {kSequenceUpdatePositionOffset, 4u},
            {kSequenceUpdateMotionOffset, 4u},
            {kSequenceUpdateFlagsOffset, 1u},
            {kSequenceUpdateIgnoredOffset, 1u},
            {kSequenceCollisionBatchOffset, 2u},
            {kSequenceCollisionOutput8Offset, 4u},
            {kSequenceCollisionOutput12Offset, 4u},
            {kSequenceCollisionOutput20Offset, 1u},
            {kSequenceCollisionOutputPositionOffset, 4u},
            {kSequenceCollisionOutputMotionOffset, 4u},
            {kSequenceCollisionVector40Offset, 4u},
            {kSequenceCollisionVector44Offset, 4u},
            {kSequenceCollisionScalar48Offset, 1u},
            {kSequenceCollisionScalar52Offset, 1u},
            {kSequenceCollisionScalar56Offset, 1u},
            {kSequenceCollisionScalar60Offset, 1u},
            {kSequenceCollisionVector64Offset, 4u},
            {kSequenceCollisionActiveOffset, 1u},
            {kSequenceSolverAOffset, 4u},
            {kSequenceSolverBOffset, 4u},
            {kSequenceSolverCOffset, 4u},
            {kSequenceSolverFlagsOffset, 1u},
            {kSequenceSolverPackedOffset, 1u},
            {kSequenceSolverOutputOffset, 16u},
        }};
        result.sequence_guards_intact = std::all_of(
            ranges.begin(),
            ranges.end(),
            [&guards_match](const auto &range) {
                return guards_match(range.first, range.second);
            });
        const std::array<std::uint32_t, 4> zero_vector = {};
        result.pass =
            result.sequence_update_motion == kExpectedDeformableMotion &&
            result.sequence_collision_motion == kExpectedDeformableMotion &&
            result.sequence_collision_scalars == kExpectedCollisionScalars &&
            result.sequence_solver_a == kExpectedSolverA &&
            result.sequence_solver_position == kExpectedSolverPosition &&
            result.sequence_solver_output == kExpectedSolverOutput &&
            std::equal(
                words + kSequenceUpdatePositionOffset,
                words + kSequenceUpdatePositionOffset + 4u,
                kExpectedDeformablePosition.begin()) &&
            words[kSequenceUpdateFlagsOffset] == 0u &&
            std::equal(
                words + kSequenceCollisionOutput8Offset,
                words + kSequenceCollisionOutput8Offset + 4u,
                zero_vector.begin()) &&
            std::equal(
                words + kSequenceCollisionOutput12Offset,
                words + kSequenceCollisionOutput12Offset + 4u,
                zero_vector.begin()) &&
            words[kSequenceCollisionOutput20Offset] == 0xbf800000u &&
            std::equal(
                words + kSequenceCollisionVector40Offset,
                words + kSequenceCollisionVector40Offset + 4u,
                zero_vector.begin()) &&
            std::equal(
                words + kSequenceCollisionVector44Offset,
                words + kSequenceCollisionVector44Offset + 4u,
                zero_vector.begin()) &&
            std::equal(
                words + kSequenceCollisionVector64Offset,
                words + kSequenceCollisionVector64Offset + 4u,
                zero_vector.begin()) &&
            words[kSequenceCollisionActiveOffset] == 0u &&
            words[kSequenceSolverFlagsOffset] == 0x00000400u &&
            words[kSequenceSolverPackedOffset] == 0xabcdef01u &&
            result.sequence_guards_intact;
        if (!result.pass) {
            result.error = "ordered deformable sequence fixture result mismatch";
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
        for (VkPipeline pipeline : pipelines) {
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
            }
        }
        for (VkShaderModule shader_module : shader_modules) {
            if (shader_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shader_module, nullptr);
            }
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
    const bool kernel_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--kernel-fixture";
    const bool reorder_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--reorder-fixture";
    const bool constant_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--constant-fixture";
    const bool texture_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--texture-fixture";
    const bool texture_float_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--texture-float-fixture";
    const bool density_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--density-fixture";
    const bool force_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--force-fixture";
    const bool deformable_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--deformable-fixture";
    const bool solver_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--deformable-solver-fixture";
    const bool collision_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--deformable-collision-fixture";
    const bool constraints_fixture_mode =
        argc == 4 && std::string(argv[2]) == "--fluid-constraints-fixture";
    const bool dynamic_response_fixture_mode =
        argc == 4 &&
        std::string(argv[2]) == "--fluid-dynamic-response-fixture";
    const bool sequence_fixture_mode =
        argc == 6 && std::string(argv[2]) == "--deformable-sequence-fixture";
    const bool incomplete_named_mode =
        argc == 3 &&
        (std::string(argv[2]) == "--ptx-oracle" ||
         std::string(argv[2]) == "--kernel-fixture" ||
         std::string(argv[2]) == "--reorder-fixture" ||
         std::string(argv[2]) == "--constant-fixture" ||
         std::string(argv[2]) == "--texture-fixture" ||
         std::string(argv[2]) == "--texture-float-fixture" ||
         std::string(argv[2]) == "--density-fixture" ||
         std::string(argv[2]) == "--force-fixture" ||
         std::string(argv[2]) == "--deformable-fixture" ||
         std::string(argv[2]) == "--deformable-solver-fixture" ||
         std::string(argv[2]) == "--deformable-sequence-fixture" ||
         std::string(argv[2]) == "--fluid-constraints-fixture" ||
         std::string(argv[2]) == "--fluid-dynamic-response-fixture" ||
         std::string(argv[2]) == "--deformable-collision-fixture");
    if ((argc != 2 && argc != 3 &&
         !ptx_oracle_mode && !kernel_fixture_mode && !reorder_fixture_mode &&
         !constant_fixture_mode && !texture_fixture_mode &&
         !texture_float_fixture_mode && !density_fixture_mode &&
         !force_fixture_mode && !deformable_fixture_mode &&
         !solver_fixture_mode && !collision_fixture_mode &&
         !constraints_fixture_mode && !dynamic_response_fixture_mode &&
         !sequence_fixture_mode) ||
        incomplete_named_mode) {
        std::cerr << "usage: vulkan-memory-smoke <output.json> [oracle.spv]\n"
                  << "       vulkan-memory-smoke <output.json> --ptx-oracle <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --kernel-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --reorder-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --constant-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --texture-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --texture-float-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --density-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --force-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --deformable-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --deformable-solver-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --deformable-collision-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --fluid-constraints-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --fluid-dynamic-response-fixture <generated.spv>\n"
                  << "       vulkan-memory-smoke <output.json> --deformable-sequence-fixture <update.spv> <collision.spv> <solver.spv>\n";
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

    const bool oracle_mode =
        (argc == 3 && !incomplete_named_mode) || ptx_oracle_mode;
    const bool shader_input_mode =
        oracle_mode || kernel_fixture_mode || reorder_fixture_mode ||
        constant_fixture_mode || texture_fixture_mode ||
        texture_float_fixture_mode || density_fixture_mode ||
        force_fixture_mode || deformable_fixture_mode || solver_fixture_mode ||
        collision_fixture_mode || constraints_fixture_mode ||
        dynamic_response_fixture_mode;
    const bool require_memory_result =
        !ptx_oracle_mode && !kernel_fixture_mode && !reorder_fixture_mode &&
        !constant_fixture_mode && !texture_fixture_mode &&
        !texture_float_fixture_mode && !density_fixture_mode &&
        !force_fixture_mode && !deformable_fixture_mode &&
        !solver_fixture_mode && !collision_fixture_mode &&
        !constraints_fixture_mode && !dynamic_response_fixture_mode;
    std::vector<std::uint32_t> shader;
    std::array<std::vector<std::uint32_t>, 3> sequence_shaders;
    const auto load_shader = [](const char *path,
                                std::vector<std::uint32_t> &target) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            std::cerr << "failed to open SPIR-V input file: " << path << '\n';
            return false;
        }
        const std::streamsize byte_count = input.tellg();
        if (byte_count <= 0 ||
            byte_count % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
            std::cerr << "invalid SPIR-V byte length: " << path << '\n';
            return false;
        }
        target.resize(
            static_cast<std::size_t>(byte_count) / sizeof(std::uint32_t));
        input.seekg(0);
        input.read(reinterpret_cast<char *>(target.data()), byte_count);
        if (!input || target.front() != SpvMagicNumber) {
            std::cerr << "invalid SPIR-V module: " << path << '\n';
            return false;
        }
        return true;
    };
    bool shaders_loaded = true;
    if (sequence_fixture_mode) {
        for (std::size_t index = 0u;
             index < sequence_shaders.size();
             ++index) {
            shaders_loaded =
                load_shader(argv[3 + index], sequence_shaders[index]) &&
                shaders_loaded;
        }
    } else if (shader_input_mode) {
        const char *shader_path =
            ptx_oracle_mode || kernel_fixture_mode || reorder_fixture_mode ||
                    constant_fixture_mode || texture_fixture_mode ||
                    texture_float_fixture_mode || density_fixture_mode ||
                    force_fixture_mode || deformable_fixture_mode ||
                    solver_fixture_mode || collision_fixture_mode ||
                    constraints_fixture_mode || dynamic_response_fixture_mode
                ? argv[3]
                : argv[2];
        shaders_loaded = load_shader(shader_path, shader);
    } else {
        shader = make_memory_shader();
    }
    if (!shaders_loaded) {
        vkDestroyInstance(instance, nullptr);
        return 3;
    }
    std::vector<DeviceResult> results;
    results.reserve(device_count);
    bool all_passed = true;
    for (std::uint32_t index = 0; index < device_count; ++index) {
        DeviceResult result = sequence_fixture_mode
            ? run_deformable_sequence_device(
                  index, devices[index], sequence_shaders)
            : run_device(
                  index,
                  devices[index],
                  shader,
                  oracle_mode,
                  ptx_oracle_mode,
                  require_memory_result,
                  kernel_fixture_mode,
                  reorder_fixture_mode,
                  constant_fixture_mode,
                  texture_fixture_mode,
                  texture_float_fixture_mode,
                  density_fixture_mode,
                  force_fixture_mode,
                  deformable_fixture_mode,
                  solver_fixture_mode,
                  collision_fixture_mode,
                  constraints_fixture_mode,
                  dynamic_response_fixture_mode);
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
           << (sequence_fixture_mode ? "deformable_sequence_fixture" :
               dynamic_response_fixture_mode
                   ? "fluid_dynamic_response_fixture" :
               constraints_fixture_mode ? "fluid_constraints_fixture" :
               collision_fixture_mode ? "deformable_collision_fixture" :
               solver_fixture_mode ? "deformable_solver_fixture" :
               deformable_fixture_mode ? "deformable_update_fixture" :
               force_fixture_mode ? "force_kernel_fixture" :
               density_fixture_mode ? "density_kernel_fixture" :
               reorder_fixture_mode ? "reorder_kernel_fixture" :
               kernel_fixture_mode ? "selected_kernel_fixture" :
               constant_fixture_mode ? "constant_kernel_fixture" :
               texture_fixture_mode ? "texture_kernel_fixture" :
               texture_float_fixture_mode ? "texture_float_kernel_fixture" :
               ptx_oracle_mode ? "ptx_generated_oracle" :
               oracle_mode ? "cuda_semantic_oracle" : "integer_global_memory")
           << "\",\n"
           << "  \"buffer_bytes\": " << kBufferBytes << ",\n";
    if (sequence_fixture_mode) {
        output << "  \"kernel_order\": [\"PxgCudaDeformableUpdate\", "
                  "\"KernelDeformableUberCollision\", "
                  "\"KernelDeformableUberSolver<512>\"],\n"
               << "  \"particle_count\": 1,\n"
               << "  \"shape_count\": 0,\n"
               << "  \"iteration_count\": 0,\n"
               << "  \"collision_static_shared_bytes\": 4097,\n"
               << "  \"solver_dynamic_shared_bytes\": 7072,\n";
    } else if (dynamic_response_fixture_mode) {
        output << "  \"kernel\": "
                  "\"PxgCudaFluidCollisionDynamicResponseBatch\",\n"
               << "  \"particle_count\": 1,\n"
               << "  \"input_constraint_flags\": \"0x00000001\",\n"
               << "  \"impulse_scale_bits\": \"0x3f000000\",\n"
               << "  \"static_shared_bytes\": 304,\n"
               << "  \"expected_position_bits\": [\"0x40e00000\", "
                  "\"0x41000000\", \"0x41100000\", \"0x00000000\"],\n"
               << "  \"expected_velocity_bits\": [\"0x3f800000\", "
                  "\"0x40400000\", \"0x00000000\", \"0x00000000\"],\n"
               << "  \"expected_response_bits\": [\"0xc0400000\", "
                  "\"0x00000000\", \"0x00000000\", \"0x00000000\"],\n";
    } else if (constraints_fixture_mode) {
        output << "  \"kernel\": \"PxgCudaFluidCollisionConstraintsBatch\",\n"
               << "  \"particle_count\": 1,\n"
               << "  \"constraint_key\": \"0x0000ffff\",\n"
               << "  \"time_step_bits\": \"0x3f000000\",\n"
               << "  \"static_shared_bytes\": 1840,\n"
               << "  \"expected_position_bits\": [\"0x40400000\", "
                  "\"0x40900000\", \"0x40c00000\", \"0x00000000\"],\n"
               << "  \"expected_velocity_bits\": [\"0x40800000\", "
                  "\"0x40a00000\", \"0x40c00000\", \"0x00000000\"],\n"
               << "  \"expected_weight_bits\": \"0x3f800000\",\n";
    } else if (collision_fixture_mode) {
        output << "  \"kernel\": \"KernelDeformableUberCollision\",\n"
               << "  \"particle_count\": 1,\n"
               << "  \"shape_count\": 0,\n"
               << "  \"static_shared_bytes\": 4097,\n"
               << "  \"expected_collision_scalar_bits\": ["
                  "\"0xbf800000\", \"0x00000000\", "
                  "\"0x0000ffff\", \"0x00000000\"],\n";
    } else if (solver_fixture_mode) {
        output << "  \"kernel\": \"KernelDeformableUberSolver<512>\",\n"
               << "  \"particle_count\": 1,\n"
               << "  \"iteration_count\": 0,\n"
               << "  \"dynamic_shared_bytes\": 7072,\n";
    } else if (deformable_fixture_mode) {
        output << "  \"kernel\": \"PxgCudaDeformableUpdate\",\n"
               << "  \"instance_count\": 1,\n"
               << "  \"acceleration_bits\": [\"0x3e800000\", "
                  "\"0xbf000000\", \"0x3f800000\"],\n"
               << "  \"time_step_bits\": \"0x3f000000\",\n"
               << "  \"input_position_bits\": [\"0x3f800000\", "
                  "\"0x40000000\", \"0x40400000\", \"0x00000000\"],\n"
               << "  \"input_motion_bits\": [\"0x40800000\", "
                  "\"0x40a00000\", \"0x40c00000\", \"0x40e00000\"],\n"
               << "  \"input_flags\": \"0x00000014\",\n"
               << "  \"expected_motion_bits\": [\"0x40440000\", "
                  "\"0x408c0000\", \"0x40c80000\", \"0x40e00000\"],\n"
               << "  \"expected_flags\": \"0x00000000\",\n";
    } else if (force_fixture_mode) {
        output << "  \"kernel\": \"KernelFluidForce\",\n"
               << "  \"position_bits\": [\"0x3f800000\", \"0x40000000\", "
                  "\"0x40400000\", \"0x00000000\"],\n"
               << "  \"velocity_bits\": [\"0x40800000\", \"0x40a00000\", "
                  "\"0x40c00000\", \"0x40e00000\"],\n"
               << "  \"density_bits\": \""
               << hex32(kExpectedDensityBits) << "\",\n"
               << "  \"hash\": \"" << hex32(kExpectedKernelHash) << "\",\n"
               << "  \"force_constants\": {\"time_step\": \"0x3f800000\", "
                  "\"damping\": \"0x00000000\", "
                  "\"maximum_velocity\": \"0x42c80000\"},\n"
               << "  \"expected_force_bits\": [\"0x40800000\", "
                  "\"0x40a00000\", \"0x40c00000\", \"0x00000000\"],\n";
    } else if (density_fixture_mode) {
        output << "  \"kernel\": \"KernelFluidDensity\",\n"
               << "  \"position_bits\": [\"0x3f800000\", \"0x40000000\", "
                  "\"0x40400000\", \"0x00000000\"],\n"
               << "  \"hash\": \"" << hex32(kExpectedKernelHash) << "\",\n"
               << "  \"density_constants\": {\"radius_term\": \"0x3f800000\", "
                  "\"distance_scale\": \"0x3f800000\", "
                  "\"normalization\": \"0x40000000\"},\n"
               << "  \"expected_density_bits\": \""
               << hex32(kExpectedDensityBits) << "\",\n";
    } else if (reorder_fixture_mode) {
        output << "  \"kernel\": \"KernelReorderPosVelAndFindCellStartPacked\",\n"
               << "  \"position_bits\": [\"0x3f800000\", \"0x40000000\", "
                  "\"0x40400000\", \"0x00000000\"],\n"
               << "  \"velocity_bits\": [\"0x40800000\", \"0x40a00000\", "
                  "\"0x40c00000\", \"0x40e00000\"],\n"
               << "  \"hash\": \"" << hex32(kExpectedKernelHash) << "\",\n";
    } else if (kernel_fixture_mode) {
        output << "  \"kernel\": \"KernelFluidCalcHashPacked\",\n"
               << "  \"position_bits\": [\"0x3f800000\", \"0x40000000\", "
                  "\"0x40400000\"],\n"
               << "  \"cell_divisor_bits\": \"0x3f800000\",\n"
               << "  \"expected_hash\": \"" << hex32(kExpectedKernelHash) << "\",\n";
    } else if (constant_fixture_mode) {
        output << "  \"kernel\": \"constant_probe\",\n"
               << "  \"expected\": [\"" << hex32(kExpectedConstant[0])
               << "\", \"" << hex32(kExpectedConstant[1]) << "\"],\n";
    } else if (texture_fixture_mode || texture_float_fixture_mode) {
        output << "  \"kernel\": \""
               << (texture_fixture_mode ? "texture_probe" : "texture_float_probe")
               << "\",\n"
               << "  \"coordinates\": [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],\n";
    } else {
        output << "  \"input\": \"" << hex32(kInput) << "\",\n"
               << "  \"mask\": \"" << hex32(kMask) << "\",\n"
               << "  \"addend\": \"" << hex32(kAddend) << "\",\n"
               << "  \"expected\": \"" << hex32(kExpected) << "\",\n";
    }
    if (sequence_fixture_mode) {
        output << "  \"spirv_words\": ["
               << sequence_shaders[0].size() << ", "
               << sequence_shaders[1].size() << ", "
               << sequence_shaders[2].size() << "],\n";
    } else {
        output << "  \"spirv_words\": " << shader.size() << ",\n";
    }
    output << "  \"devices\": [\n";
    const auto write_bits = [&output](const auto &values) {
        output << '[';
        for (std::size_t index = 0u; index < values.size(); ++index) {
            output << (index == 0u ? "" : ", ")
                   << "\"" << hex32(values[index]) << "\"";
        }
        output << ']';
    };
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
               << (result.staging_coherent ? "true" : "false") << ",\n";
        if (sequence_fixture_mode) {
            output << "      \"update_motion_bits\": ";
            write_bits(result.sequence_update_motion);
            output << ",\n      \"collision_motion_bits\": ";
            write_bits(result.sequence_collision_motion);
            output << ",\n      \"collision_scalar_bits\": ";
            write_bits(result.sequence_collision_scalars);
            output << ",\n      \"solver_a_bits\": ";
            write_bits(result.sequence_solver_a);
            output << ",\n      \"solver_position_bits\": ";
            write_bits(result.sequence_solver_position);
            output << ",\n      \"solver_output_bits\": ";
            write_bits(result.sequence_solver_output);
            output << ",\n"
                   << "      \"guards_intact\": "
                   << (result.sequence_guards_intact ? "true" : "false")
                   << ",\n";
        } else if (dynamic_response_fixture_mode) {
            output << "      \"output_normal_bits\": ";
            write_bits(result.dynamic_response_normal);
            output << ",\n      \"output_position_bits\": ";
            write_bits(result.dynamic_response_position);
            output << ",\n      \"output_velocity_bits\": ";
            write_bits(result.dynamic_response_velocity);
            output << ",\n      \"output_response_bits\": ";
            write_bits(result.dynamic_response_output);
            output << ",\n"
                   << "      \"output_weight_bits\": \""
                   << hex32(result.dynamic_response_weight) << "\",\n"
                   << "      \"output_destination_id\": \""
                   << hex32(result.dynamic_response_dest_id) << "\",\n"
                   << "      \"output_constraint_flags\": \""
                   << hex32(result.dynamic_response_flags) << "\",\n"
                   << "      \"guards_intact\": "
                   << (result.dynamic_response_guards_intact ? "true" : "false")
                   << ",\n";
        } else if (constraints_fixture_mode) {
            output << "      \"output_position_bits\": ";
            write_bits(result.constraints_position);
            output << ",\n      \"output_velocity_bits\": ";
            write_bits(result.constraints_velocity);
            output << ",\n"
                   << "      \"output_weight_bits\": \""
                   << hex32(result.constraints_weight) << "\",\n"
                   << "      \"guards_intact\": "
                   << (result.constraints_guards_intact ? "true" : "false")
                   << ",\n";
        } else if (collision_fixture_mode) {
            output << "      \"output_position_bits\": [";
            for (std::size_t component = 0u;
                 component < result.collision_position.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.collision_position[component])
                       << "\"";
            }
            output << "],\n"
                   << "      \"output_motion_bits\": [";
            for (std::size_t component = 0u;
                 component < result.collision_motion.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.collision_motion[component])
                       << "\"";
            }
            output << "],\n"
                   << "      \"collision_scalar_bits\": [";
            for (std::size_t component = 0u;
                 component < result.collision_scalars.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.collision_scalars[component])
                       << "\"";
            }
            output << "],\n"
                   << "      \"guards_intact\": "
                   << (result.collision_guards_intact ? "true" : "false")
                   << ",\n";
        } else if (solver_fixture_mode) {
            output << "      \"output_bits\": [";
            for (std::size_t component = 0u;
                 component < result.solver_output.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.solver_output[component])
                       << "\"";
            }
            output << "],\n"
                   << "      \"position_bits\": [";
            for (std::size_t component = 0u;
                 component < result.solver_position.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.solver_position[component])
                       << "\"";
            }
            output << "],\n"
                   << "      \"guards_intact\": "
                   << (result.solver_guards_intact ? "true" : "false")
                   << ",\n";
        } else if (deformable_fixture_mode) {
            output << "      \"output_position_bits\": [";
            for (std::size_t component = 0u;
                 component < result.deformable_position.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.deformable_position[component])
                       << "\"";
            }
            output << "],\n"
                   << "      \"output_motion_bits\": [";
            for (std::size_t component = 0u;
                 component < result.deformable_motion.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.deformable_motion[component])
                       << "\"";
            }
            output << "],\n"
                   << "      \"output_flags\": \""
                   << hex32(result.deformable_flags) << "\",\n"
                   << "      \"guards_intact\": "
                   << (result.deformable_guards_intact ? "true" : "false")
                   << ",\n";
        } else if (force_fixture_mode) {
            output << "      \"force_bits\": [";
            for (std::size_t component = 0u;
                 component < result.force.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.force[component]) << "\"";
            }
            output << "],\n";
        } else if (density_fixture_mode) {
            output << "      \"density_bits\": \""
                   << hex32(result.density) << "\",\n"
                   << "      \"remapped_density_bits\": \""
                   << hex32(result.remapped_density) << "\",\n";
        } else if (reorder_fixture_mode) {
            output << "      \"reordered_position_bits\": [";
            for (std::size_t component = 0u;
                 component < result.reordered_position.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.reordered_position[component]) << "\"";
            }
            output << "],\n"
                   << "      \"reordered_velocity_bits\": [";
            for (std::size_t component = 0u;
                 component < result.reordered_velocity.size();
                 ++component) {
                output << (component == 0u ? "" : ", ")
                       << "\"" << hex32(result.reordered_velocity[component]) << "\"";
            }
            output << "],\n"
                   << "      \"cell_start\": " << result.cell_start << ",\n"
                   << "      \"cell_end\": " << result.cell_end << ",\n";
        } else if (kernel_fixture_mode) {
            output << "      \"kernel_hash\": \"" << hex32(result.kernel_hash)
                   << "\",\n"
                   << "      \"kernel_index\": " << result.kernel_index << ",\n";
        } else if (constant_fixture_mode) {
            output << "      \"constant\": {\"values\": [\""
                   << hex32(result.constant[0]) << "\", \""
                   << hex32(result.constant[1]) << "\"]},\n";
        } else if (texture_fixture_mode || texture_float_fixture_mode) {
            const auto &texture_result = texture_fixture_mode
                ? result.texture
                : result.texture_float;
            output << "      \"texture\": {\"rows\": [\n";
            for (std::size_t row = 0u; row < texture_result.size(); ++row) {
                output << "        {\"coord\": " << static_cast<int>(row) - 2
                       << ", \"value\": [";
                for (std::size_t component = 0u; component < 4u; ++component) {
                    output << (component == 0u ? "" : ", ")
                           << "\"" << hex32(texture_result[row][component]) << "\"";
                }
                output << "]}"
                       << (row + 1u == texture_result.size() ? "\n" : ",\n");
            }
            output << "      ]},\n";
        } else {
            output << "      \"input_after\": \"" << hex32(result.input_after) << "\",\n"
                   << "      \"result\": \"" << hex32(result.result) << "\",\n";
        }
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
                   << "      \"local\": {\"values\": [";
            for (std::size_t local_index = 0;
                 local_index < result.local.size();
                 ++local_index) {
                output << (local_index == 0 ? "" : ", ")
                       << "\"" << hex32(result.local[local_index]) << "\"";
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
            output << "      ]},\n"
                   << "      \"texture_float\": {\"rows\": [\n";
            for (std::size_t row = 0; row < result.texture_float.size(); ++row) {
                output << "        {\"coord\": " << static_cast<int>(row) - 2 << ", \"value\": [";
                for (std::size_t component = 0; component < 4; ++component) {
                    output << (component == 0 ? "" : ", ")
                           << "\"" << hex32(result.texture_float[row][component]) << "\"";
                }
                output << "]}" << (row + 1 == result.texture_float.size() ? "\n" : ",\n");
            }
            output << "      ]},\n";
            if (ptx_oracle_mode) {
                output << "      \"constant\": {\"values\": [";
                for (std::size_t constant_index = 0;
                     constant_index < result.constant.size();
                     ++constant_index) {
                    output << (constant_index == 0 ? "" : ", ")
                           << "\"" << hex32(result.constant[constant_index]) << "\"";
                }
                output << "]},\n";
            }
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
