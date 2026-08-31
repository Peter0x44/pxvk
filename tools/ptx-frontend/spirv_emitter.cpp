#include <nlohmann/json.hpp>
#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

enum class ValueType {
    boolean,
    u32,
    s32,
    f32,
};

struct Value {
    ValueType type;
    std::uint32_t id;
};

struct OracleBinding {
    std::uint32_t output_word;
};

constexpr std::uint32_t kTextureMetadataWords = 3u;
constexpr std::uint32_t kTextureBaseWord = 0u;
constexpr std::uint32_t kTextureElementCount = 1u;
constexpr std::uint32_t kTextureComponentCount = 2u;

class SpirvModule {
public:
    SpirvModule(
        std::uint32_t shared_bytes,
        std::uint32_t constant_bytes,
        bool texture_metadata)
    {
        emit(capabilities_, SpvOpCapability, {SpvCapabilityShader});
        emit(capabilities_, SpvOpCapability, {SpvCapabilityRoundingModeRTE});
        emit(capabilities_, SpvOpCapability, {SpvCapabilitySignedZeroInfNanPreserve});
        glsl_ = allocate_id();
        emit_string(imports_, SpvOpExtInstImport, {glsl_}, "GLSL.std.450");
        emit(memory_model_, SpvOpMemoryModel, {SpvAddressingModelLogical, SpvMemoryModelGLSL450});

        type_void_ = allocate_id();
        emit(declarations_, SpvOpTypeVoid, {type_void_});
        type_bool_ = allocate_id();
        emit(declarations_, SpvOpTypeBool, {type_bool_});
        type_u32_ = allocate_id();
        emit(declarations_, SpvOpTypeInt, {type_u32_, 32u, 0u});
        type_s32_ = allocate_id();
        emit(declarations_, SpvOpTypeInt, {type_s32_, 32u, 1u});
        type_f32_ = allocate_id();
        emit(declarations_, SpvOpTypeFloat, {type_f32_, 32u});
        type_v3u32_ = allocate_id();
        emit(declarations_, SpvOpTypeVector, {type_v3u32_, type_u32_, 3u});

        type_function_void_ = allocate_id();
        emit(declarations_, SpvOpTypeFunction, {type_function_void_, type_void_});
        type_runtime_u32_ = allocate_id();
        emit(declarations_, SpvOpTypeRuntimeArray, {type_runtime_u32_, type_u32_});
        emit(annotations_, SpvOpDecorate, {type_runtime_u32_, SpvDecorationArrayStride, 4u});
        type_buffer_ = allocate_id();
        emit(declarations_, SpvOpTypeStruct, {type_buffer_, type_runtime_u32_});
        emit(annotations_, SpvOpMemberDecorate, {type_buffer_, 0u, SpvDecorationOffset, 0u});
        emit(annotations_, SpvOpDecorate, {type_buffer_, SpvDecorationBlock});

        type_ptr_storage_buffer_ = allocate_id();
        emit(declarations_, SpvOpTypePointer,
             {type_ptr_storage_buffer_, SpvStorageClassStorageBuffer, type_buffer_});
        type_ptr_storage_u32_ = allocate_id();
        emit(declarations_, SpvOpTypePointer,
             {type_ptr_storage_u32_, SpvStorageClassStorageBuffer, type_u32_});
        type_ptr_input_v3u32_ = allocate_id();
        emit(declarations_, SpvOpTypePointer,
             {type_ptr_input_v3u32_, SpvStorageClassInput, type_v3u32_});
        type_ptr_input_u32_ = allocate_id();
        emit(declarations_, SpvOpTypePointer,
             {type_ptr_input_u32_, SpvStorageClassInput, type_u32_});
        const std::uint32_t shared_words =
            shared_bytes / 4u + (shared_bytes % 4u == 0u ? 0u : 1u);
        const std::uint32_t shared_length =
            constant(ValueType::u32, std::max(1u, shared_words)).id;
        type_shared_array_ = allocate_id();
        emit(declarations_, SpvOpTypeArray, {type_shared_array_, type_u32_, shared_length});
        type_ptr_workgroup_array_ = allocate_id();
        emit(declarations_, SpvOpTypePointer,
             {type_ptr_workgroup_array_, SpvStorageClassWorkgroup, type_shared_array_});
        type_ptr_workgroup_u32_ = allocate_id();
        emit(declarations_, SpvOpTypePointer,
             {type_ptr_workgroup_u32_, SpvStorageClassWorkgroup, type_u32_});

        buffer_ = allocate_id();
        emit(declarations_, SpvOpVariable,
             {type_ptr_storage_buffer_, buffer_, SpvStorageClassStorageBuffer});
        emit(annotations_, SpvOpDecorate, {buffer_, SpvDecorationDescriptorSet, 0u});
        emit(annotations_, SpvOpDecorate, {buffer_, SpvDecorationBinding, 0u});

        if (constant_bytes != 0u) {
            constant_buffer_ = allocate_id();
            emit(declarations_, SpvOpVariable,
                 {type_ptr_storage_buffer_, constant_buffer_,
                  SpvStorageClassStorageBuffer});
            emit(annotations_, SpvOpDecorate,
                 {constant_buffer_, SpvDecorationDescriptorSet, 0u});
            emit(annotations_, SpvOpDecorate,
                 {constant_buffer_, SpvDecorationBinding, 2u});
            emit(annotations_, SpvOpDecorate,
                 {constant_buffer_, SpvDecorationNonWritable});
        }

        if (texture_metadata) {
            texture_metadata_buffer_ = allocate_id();
            emit(declarations_, SpvOpVariable,
                 {type_ptr_storage_buffer_, texture_metadata_buffer_,
                  SpvStorageClassStorageBuffer});
            emit(annotations_, SpvOpDecorate,
                 {texture_metadata_buffer_, SpvDecorationDescriptorSet, 0u});
            emit(annotations_, SpvOpDecorate,
                 {texture_metadata_buffer_, SpvDecorationBinding, 3u});
            emit(annotations_, SpvOpDecorate,
                 {texture_metadata_buffer_, SpvDecorationNonWritable});
        }

        local_id_ = allocate_id();
        emit(declarations_, SpvOpVariable,
             {type_ptr_input_v3u32_, local_id_, SpvStorageClassInput});
        emit(annotations_, SpvOpDecorate,
             {local_id_, SpvDecorationBuiltIn, SpvBuiltInLocalInvocationId});

        shared_ = allocate_id();
        emit(declarations_, SpvOpVariable,
             {type_ptr_workgroup_array_, shared_, SpvStorageClassWorkgroup});
    }

    std::uint32_t allocate_id()
    {
        return next_id_++;
    }

    std::uint32_t type_id(ValueType type) const
    {
        switch (type) {
        case ValueType::boolean: return type_bool_;
        case ValueType::u32: return type_u32_;
        case ValueType::s32: return type_s32_;
        case ValueType::f32: return type_f32_;
        }
        throw std::runtime_error("invalid SPIR-V value type");
    }

    Value constant(ValueType type, std::uint32_t bits)
    {
        const auto key = std::make_pair(type, bits);
        const auto found = constants_.find(key);
        if (found != constants_.end()) {
            return {type, found->second};
        }
        const std::uint32_t result = allocate_id();
        emit(declarations_, SpvOpConstant, {type_id(type), result, bits});
        constants_.emplace(key, result);
        return {type, result};
    }

    Value instruction(SpvOp operation, ValueType type, std::initializer_list<std::uint32_t> operands)
    {
        const std::uint32_t result = allocate_id();
        std::vector<std::uint32_t> full = {type_id(type), result};
        full.insert(full.end(), operands.begin(), operands.end());
        emit(functions_, operation, full);
        return {type, result};
    }

    Value ext_instruction(ValueType type, GLSLstd450 operation, std::initializer_list<std::uint32_t> operands)
    {
        const std::uint32_t result = allocate_id();
        std::vector<std::uint32_t> full = {
            type_id(type), result, glsl_, static_cast<std::uint32_t>(operation),
        };
        full.insert(full.end(), operands.begin(), operands.end());
        emit(functions_, SpvOpExtInst, full);
        return {type, result};
    }

    void function_instruction(SpvOp operation, std::initializer_list<std::uint32_t> operands)
    {
        emit(functions_, operation, operands);
    }

    void function_instruction(SpvOp operation, const std::vector<std::uint32_t> &operands)
    {
        emit(functions_, operation, operands);
    }

    void decorate(std::uint32_t id, SpvDecoration decoration)
    {
        emit(annotations_, SpvOpDecorate, {id, static_cast<std::uint32_t>(decoration)});
    }


    void begin_function(std::uint32_t function)
    {
        emit(functions_, SpvOpFunction,
             {type_void_, function, SpvFunctionControlMaskNone, type_function_void_});
        emit(functions_, SpvOpLabel, {allocate_id()});
    }

    void end_function()
    {
        emit(functions_, SpvOpReturn, {});
        emit(functions_, SpvOpFunctionEnd, {});
    }

    void define_main(std::uint32_t main_function,
                     std::uint32_t math_function,
                     std::uint32_t barrier_function,
                     std::uint32_t local_function,
                     std::uint32_t texture_function,
                     std::uint32_t texture_float_function,
                     std::uint32_t constant_function)
    {
        if (constant_buffer_ != 0u) {
            emit_entry_point(
                main_function,
                {local_id_, buffer_, shared_, constant_buffer_});
        } else {
            emit_entry_point(main_function, {local_id_, buffer_, shared_});
        }
        emit(execution_modes_, SpvOpExecutionMode,
             {main_function, SpvExecutionModeLocalSize, 64u, 1u, 1u});
        emit(execution_modes_, SpvOpExecutionMode,
             {main_function, SpvExecutionModeRoundingModeRTE, 32u});
        emit(execution_modes_, SpvOpExecutionMode,
             {main_function, SpvExecutionModeSignedZeroInfNanPreserve, 32u});

        begin_function(main_function);
        const Value thread = local_invocation_x();
        const Value zero = constant(ValueType::u32, 0u);
        const Value twelve = constant(ValueType::u32, 12u);

        const Value is_math_thread = instruction(
            SpvOpIEqual, ValueType::boolean, {thread.id, zero.id});
        const std::uint32_t math_label = allocate_id();
        const std::uint32_t math_merge = allocate_id();
        function_instruction(SpvOpSelectionMerge, {math_merge, SpvSelectionControlMaskNone});
        function_instruction(SpvOpBranchConditional,
                             {is_math_thread.id, math_label, math_merge});
        function_instruction(SpvOpLabel, {math_label});
        emit_void_call(math_function);
        emit_void_call(constant_function);
        function_instruction(SpvOpBranch, {math_merge});
        function_instruction(SpvOpLabel, {math_merge});

        emit_void_call(barrier_function);
        emit_void_call(local_function);

        const Value is_texture_thread = instruction(
            SpvOpULessThan, ValueType::boolean, {thread.id, twelve.id});
        const std::uint32_t texture_label = allocate_id();
        const std::uint32_t texture_merge = allocate_id();
        function_instruction(SpvOpSelectionMerge, {texture_merge, SpvSelectionControlMaskNone});
        function_instruction(SpvOpBranchConditional,
                             {is_texture_thread.id, texture_label, texture_merge});
        function_instruction(SpvOpLabel, {texture_label});
        emit_void_call(texture_function);
        emit_void_call(texture_float_function);
        function_instruction(SpvOpBranch, {texture_merge});
        function_instruction(SpvOpLabel, {texture_merge});
        end_function();
    }

    void define_kernel_entry_point(std::uint32_t function, std::uint32_t local_size)
    {
        ensure_kernel_globals();
        std::vector<std::uint32_t> interfaces = {
            local_id_, workgroup_id_, buffer_, parameter_buffer_, shared_,
        };
        if (constant_buffer_ != 0u) {
            interfaces.push_back(constant_buffer_);
        }
        if (texture_metadata_buffer_ != 0u) {
            interfaces.push_back(texture_metadata_buffer_);
        }
        emit_entry_point(function, interfaces);
        emit(execution_modes_, SpvOpExecutionMode,
             {function, SpvExecutionModeLocalSize, local_size, 1u, 1u});
        emit(execution_modes_, SpvOpExecutionMode,
             {function, SpvExecutionModeRoundingModeRTE, 32u});
        emit(execution_modes_, SpvOpExecutionMode,
             {function, SpvExecutionModeSignedZeroInfNanPreserve, 32u});
    }

    std::uint32_t function_variable(ValueType type)
    {
        std::uint32_t *pointer_type = nullptr;
        switch (type) {
        case ValueType::boolean: pointer_type = &type_ptr_function_bool_; break;
        case ValueType::u32: pointer_type = &type_ptr_function_u32_; break;
        case ValueType::s32: pointer_type = &type_ptr_function_s32_; break;
        case ValueType::f32: pointer_type = &type_ptr_function_f32_; break;
        }
        if (*pointer_type == 0u) {
            *pointer_type = allocate_id();
            emit(declarations_, SpvOpTypePointer,
                 {*pointer_type, SpvStorageClassFunction, type_id(type)});
        }
        const std::uint32_t variable = allocate_id();
        emit(functions_, SpvOpVariable,
             {*pointer_type, variable, SpvStorageClassFunction});
        return variable;
    }
    std::uint32_t function_u32_array(std::uint32_t byte_size)
    {
        const std::uint32_t length =
            constant(ValueType::u32, std::max(1u, (byte_size + 3u) / 4u)).id;
        const std::uint32_t array_type = allocate_id();
        emit(declarations_, SpvOpTypeArray, {array_type, type_u32_, length});
        const std::uint32_t pointer_type = allocate_id();
        emit(declarations_, SpvOpTypePointer,
             {pointer_type, SpvStorageClassFunction, array_type});
        if (type_ptr_function_u32_ == 0u) {
            type_ptr_function_u32_ = allocate_id();
            emit(declarations_, SpvOpTypePointer,
                 {type_ptr_function_u32_, SpvStorageClassFunction, type_u32_});
        }
        const std::uint32_t variable = allocate_id();
        emit(functions_, SpvOpVariable,
             {pointer_type, variable, SpvStorageClassFunction});
        return variable;
    }


    Value load_variable(std::uint32_t variable, ValueType type)
    {
        return instruction(SpvOpLoad, type, {variable});
    }

    void store_variable(std::uint32_t variable, Value value, ValueType type)
    {
        const Value stored = coerce(value, type);
        function_instruction(SpvOpStore, {variable, stored.id});
    }

    Value workgroup_invocation_x()
    {
        ensure_kernel_globals();
        const Value zero = constant(ValueType::u32, 0u);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_input_u32_,
            {workgroup_id_, zero.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }

    Value load_parameter_word(std::uint32_t word_index)
    {
        ensure_kernel_globals();
        const Value zero = constant(ValueType::u32, 0u);
        const Value index = constant(ValueType::u32, word_index);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_storage_u32_,
            {parameter_buffer_, zero.id, index.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }

    Value local_invocation_x()
    {
        const Value zero = constant(ValueType::u32, 0u);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_input_u32_,
            {local_id_, zero.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }

    void store_buffer(Value word_index, Value value)
    {
        const Value zero = constant(ValueType::u32, 0u);
        const Value index = coerce(word_index, ValueType::u32);
        const Value stored = coerce(value, ValueType::u32);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_storage_u32_,
            {buffer_, zero.id, index.id});
        function_instruction(SpvOpStore, {pointer.id, stored.id});
    }

    Value load_buffer(Value word_index)
    {
        const Value zero = constant(ValueType::u32, 0u);
        const Value index = coerce(word_index, ValueType::u32);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_storage_u32_,
            {buffer_, zero.id, index.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }

    Value load_constant(Value word_index)
    {
        if (constant_buffer_ == 0u) {
            throw std::runtime_error("constant load without module constant storage");
        }
        const Value zero = constant(ValueType::u32, 0u);
        const Value index = coerce(word_index, ValueType::u32);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_storage_u32_,
            {constant_buffer_, zero.id, index.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }

    Value load_texture_metadata(std::uint32_t reference, std::uint32_t field)
    {
        if (texture_metadata_buffer_ == 0u) {
            throw std::runtime_error("texture fetch without texture metadata storage");
        }
        const Value zero = constant(ValueType::u32, 0u);
        const Value index =
            constant(
                ValueType::u32,
                reference * kTextureMetadataWords + field);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_storage_u32_,
            {texture_metadata_buffer_, zero.id, index.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }

    void store_shared(Value word_index, Value value)
    {
        const Value index = coerce(word_index, ValueType::u32);
        const Value stored = coerce(value, ValueType::u32);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_workgroup_u32_,
            {shared_, index.id});
        function_instruction(SpvOpStore, {pointer.id, stored.id});
    }

    Value load_shared(Value word_index)
    {
        const Value index = coerce(word_index, ValueType::u32);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_workgroup_u32_,
            {shared_, index.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }
    void store_local(std::uint32_t local, Value word_index, Value value)
    {
        const Value index = coerce(word_index, ValueType::u32);
        const Value stored = coerce(value, ValueType::u32);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_function_u32_,
            {local, index.id});
        function_instruction(SpvOpStore, {pointer.id, stored.id});
    }

    Value load_local(std::uint32_t local, Value word_index)
    {
        const Value index = coerce(word_index, ValueType::u32);
        const Value pointer = instruction_with_type(
            SpvOpAccessChain,
            type_ptr_function_u32_,
            {local, index.id});
        return instruction(SpvOpLoad, ValueType::u32, {pointer.id});
    }


    Value coerce(Value value, ValueType type)
    {
        if (value.type == type) {
            return value;
        }
        if (value.type == ValueType::boolean || type == ValueType::boolean) {
            throw std::runtime_error("cannot bitcast predicate and scalar values");
        }
        return instruction(SpvOpBitcast, type, {value.id});
    }

    Value flush_ftz(Value input)
    {
        const Value value = coerce(input, ValueType::f32);
        const Value bits = instruction(SpvOpBitcast, ValueType::u32, {value.id});
        const Value magnitude = instruction(
            SpvOpBitwiseAnd,
            ValueType::u32,
            {bits.id, constant(ValueType::u32, 0x7fffffffu).id});
        const Value nonzero = instruction(
            SpvOpINotEqual,
            ValueType::boolean,
            {magnitude.id, constant(ValueType::u32, 0u).id});
        const Value below_normal = instruction(
            SpvOpULessThan,
            ValueType::boolean,
            {magnitude.id, constant(ValueType::u32, 0x00800000u).id});
        const Value subnormal = instruction(
            SpvOpLogicalAnd, ValueType::boolean, {nonzero.id, below_normal.id});
        const Value sign = instruction(
            SpvOpBitwiseAnd,
            ValueType::u32,
            {bits.id, constant(ValueType::u32, 0x80000000u).id});
        const Value flushed = instruction(
            SpvOpSelect, ValueType::u32, {subnormal.id, sign.id, bits.id});
        return instruction(SpvOpBitcast, ValueType::f32, {flushed.id});
    }


    std::vector<std::uint32_t> finish()
    {
        std::vector<std::uint32_t> output = {
            SpvMagicNumber,
            0x00010500u,
            0u,
            next_id_,
            0u,
        };
        append(output, capabilities_);
        append(output, imports_);
        append(output, memory_model_);
        append(output, entry_points_);
        append(output, execution_modes_);
        append(output, annotations_);
        append(output, declarations_);
        append(output, functions_);
        return output;
    }

private:
    static void append(std::vector<std::uint32_t> &destination,
                       const std::vector<std::uint32_t> &source)
    {
        destination.insert(destination.end(), source.begin(), source.end());
    }

    static void append_string(std::vector<std::uint32_t> &output, const std::string &text)
    {
        const std::size_t byte_count = text.size() + 1u;
        const std::size_t word_count = (byte_count + 3u) / 4u;
        const std::size_t first = output.size();
        output.resize(first + word_count, 0u);
        std::memcpy(output.data() + first, text.c_str(), byte_count);
    }

    static void emit(std::vector<std::uint32_t> &section,
                     SpvOp operation,
                     std::initializer_list<std::uint32_t> operands)
    {
        emit(section, operation, std::vector<std::uint32_t>(operands));
    }

    static void emit(std::vector<std::uint32_t> &section,
                     SpvOp operation,
                     const std::vector<std::uint32_t> &operands)
    {
        section.push_back(
            (static_cast<std::uint32_t>(operands.size() + 1u) << SpvWordCountShift) |
            static_cast<std::uint32_t>(operation));
        section.insert(section.end(), operands.begin(), operands.end());
    }

    static void emit_string(std::vector<std::uint32_t> &section,
                            SpvOp operation,
                            std::initializer_list<std::uint32_t> prefix,
                            const std::string &text)
    {
        std::vector<std::uint32_t> operands(prefix);
        append_string(operands, text);
        emit(section, operation, operands);
    }

    void emit_entry_point(
        std::uint32_t function,
        const std::vector<std::uint32_t> &interfaces)
    {
        std::vector<std::uint32_t> operands = {
            static_cast<std::uint32_t>(SpvExecutionModelGLCompute), function,
        };
        append_string(operands, "main");
        operands.insert(operands.end(), interfaces.begin(), interfaces.end());
        emit(entry_points_, SpvOpEntryPoint, operands);
    }

    Value instruction_with_type(SpvOp operation,
                                std::uint32_t result_type,
                                std::initializer_list<std::uint32_t> operands)
    {
        const std::uint32_t result = allocate_id();
        std::vector<std::uint32_t> full = {result_type, result};
        full.insert(full.end(), operands.begin(), operands.end());
        emit(functions_, operation, full);
        return {ValueType::u32, result};
    }

    void ensure_kernel_globals()
    {
        if (parameter_buffer_ != 0u) {
            return;
        }
        parameter_buffer_ = allocate_id();
        emit(declarations_, SpvOpVariable,
             {type_ptr_storage_buffer_, parameter_buffer_, SpvStorageClassStorageBuffer});
        emit(annotations_, SpvOpDecorate,
             {parameter_buffer_, SpvDecorationDescriptorSet, 0u});
        emit(annotations_, SpvOpDecorate,
             {parameter_buffer_, SpvDecorationBinding, 1u});

        workgroup_id_ = allocate_id();
        emit(declarations_, SpvOpVariable,
             {type_ptr_input_v3u32_, workgroup_id_, SpvStorageClassInput});
        emit(annotations_, SpvOpDecorate,
             {workgroup_id_, SpvDecorationBuiltIn, SpvBuiltInWorkgroupId});
    }

    void emit_void_call(std::uint32_t function)
    {
        const std::uint32_t result = allocate_id();
        emit(functions_, SpvOpFunctionCall, {type_void_, result, function});
    }

    std::uint32_t next_id_ = 1u;
    std::uint32_t glsl_ = 0u;
    std::uint32_t type_void_ = 0u;
    std::uint32_t type_bool_ = 0u;
    std::uint32_t type_u32_ = 0u;
    std::uint32_t type_s32_ = 0u;
    std::uint32_t type_f32_ = 0u;
    std::uint32_t type_v3u32_ = 0u;
    std::uint32_t type_function_void_ = 0u;
    std::uint32_t type_runtime_u32_ = 0u;
    std::uint32_t type_buffer_ = 0u;
    std::uint32_t type_ptr_storage_buffer_ = 0u;
    std::uint32_t type_ptr_storage_u32_ = 0u;
    std::uint32_t type_ptr_input_v3u32_ = 0u;
    std::uint32_t type_ptr_input_u32_ = 0u;
    std::uint32_t type_ptr_function_bool_ = 0u;
    std::uint32_t type_ptr_function_u32_ = 0u;
    std::uint32_t type_ptr_function_s32_ = 0u;
    std::uint32_t type_ptr_function_f32_ = 0u;
    std::uint32_t type_shared_array_ = 0u;
    std::uint32_t type_ptr_workgroup_array_ = 0u;
    std::uint32_t type_ptr_workgroup_u32_ = 0u;
    std::uint32_t buffer_ = 0u;
    std::uint32_t constant_buffer_ = 0u;
    std::uint32_t texture_metadata_buffer_ = 0u;
    std::uint32_t local_id_ = 0u;
    std::uint32_t shared_ = 0u;
    std::uint32_t parameter_buffer_ = 0u;
    std::uint32_t workgroup_id_ = 0u;
    std::map<std::pair<ValueType, std::uint32_t>, std::uint32_t> constants_;
    std::vector<std::uint32_t> capabilities_;
    std::vector<std::uint32_t> imports_;
    std::vector<std::uint32_t> memory_model_;
    std::vector<std::uint32_t> entry_points_;
    std::vector<std::uint32_t> execution_modes_;
    std::vector<std::uint32_t> annotations_;
    std::vector<std::uint32_t> declarations_;
    std::vector<std::uint32_t> functions_;
};

ValueType parse_type(const std::string &type)
{
    if (type == ".u32" || type == ".b32") return ValueType::u32;
    if (type == ".s32") return ValueType::s32;
    if (type == ".f32") return ValueType::f32;
    if (type == ".pred") return ValueType::boolean;
    throw std::runtime_error("oracle emitter does not support register type '" + type + "'");
}

std::string register_operand_name(const Json &operand)
{
    if (operand.at("kind").get<std::string>() != "register") {
        throw std::runtime_error("oracle IR operand is not a register");
    }
    return operand.at("name").get<std::string>();
}

std::vector<std::string> register_operand_list(const Json &operand)
{
    if (operand.at("kind").get<std::string>() == "register") {
        return {register_operand_name(operand)};
    }
    if (operand.at("kind").get<std::string>() != "list") {
        throw std::runtime_error("oracle IR operand is not a register list");
    }
    std::vector<std::string> output;
    for (const Json &value : operand.at("values")) {
        output.push_back(register_operand_name(value));
    }
    return output;
}

std::uint32_t module_static_shared_bytes(const Json &manifest)
{
    std::uint32_t bytes = 0u;
    for (const Json &variable : manifest.at("variables")) {
        if (variable.at("space").get<std::string>() != "shared" ||
            variable.at("external").get<bool>()) {
            continue;
        }
        bytes = std::max(
            bytes,
            variable.at("offset").get<std::uint32_t>() +
                variable.at("size").get<std::uint32_t>());
    }
    return bytes;
}

std::uint32_t module_constant_bytes(const Json &manifest)
{
    std::uint32_t bytes = 0u;
    for (const Json &variable : manifest.at("variables")) {
        if (variable.at("space").get<std::string>() != "const" ||
            variable.at("external").get<bool>()) {
            continue;
        }
        bytes = std::max(
            bytes,
            variable.at("offset").get<std::uint32_t>() +
                variable.at("size").get<std::uint32_t>());
    }
    return bytes;
}

bool entry_uses_textures(const Json &entry)
{
    for (const Json &instruction : entry.at("instructions")) {
        if (instruction.at("base").get<std::string>() == "tex") {
            return true;
        }
    }
    return false;
}

std::uint32_t required_shared_bytes(
    const Json &manifest,
    const Json &entry,
    std::uint32_t dynamic_shared_bytes)
{
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(module_static_shared_bytes(manifest)) +
        entry.at("static_shared_bytes").get<std::uint32_t>() +
        dynamic_shared_bytes;
    if (bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("selected-kernel shared storage exceeds 32-bit size");
    }
    return static_cast<std::uint32_t>(bytes);
}

class EntryLowerer {
public:
    EntryLowerer(SpirvModule &module,
                 const Json &manifest,
                 const Json &entry,
                 OracleBinding binding,
                 std::uint32_t function)
        : module_(module), entry_(entry), binding_(binding), function_(function)
    {
        for (const Json &bank : entry_.at("register_banks")) {
            banks_.push_back({
                bank.at("prefix").get<std::string>(),
                parse_type(bank.at("type").get<std::string>()),
                bank.at("count").get<std::size_t>(),
            });
        }
        for (const Json &variable : manifest.at("variables")) {
            if (variable.at("space").get<std::string>() == "const") {
                storage_offsets_.emplace(
                    variable.at("name").get<std::string>(),
                    variable.at("offset").get<std::uint32_t>());
            }
        }
        for (const Json &variable : entry_.at("variables")) {
            storage_offsets_.emplace(
                variable.at("name").get<std::string>(),
                variable.at("offset").get<std::uint32_t>());
        }
    }

    void lower()
    {
        module_.begin_function(function_);
        if (entry_.at("static_local_bytes").get<std::uint32_t>() != 0u) {
            local_ = module_.function_u32_array(
                entry_.at("static_local_bytes").get<std::uint32_t>());
        }
        for (const Json &instruction : entry_.at("instructions")) {
            lower_instruction(instruction);
        }
        module_.end_function();
    }

private:
    struct RegisterBank {
        std::string prefix;
        ValueType type;
        std::size_t count;
    };

    [[noreturn]] void unsupported(const Json &instruction, const std::string &detail) const
    {
        throw std::runtime_error(
            entry_.at("name").get<std::string>() + ": " + detail + " for '" +
            instruction.at("variant").get<std::string>() + "' at PTX line " +
            std::to_string(instruction.at("line").get<std::size_t>()));
    }

    ValueType register_type(const std::string &name) const
    {
        if (name == "%tid.x") return ValueType::u32;
        for (const RegisterBank &bank : banks_) {
            if (bank.count == 1u && name == bank.prefix) return bank.type;
            if (bank.count > 1u && name.rfind(bank.prefix, 0u) == 0u) {
                const std::string suffix = name.substr(bank.prefix.size());
                if (!suffix.empty() &&
                    std::all_of(suffix.begin(), suffix.end(), [](char value) {
                        return value >= '0' && value <= '9';
                    }) &&
                    std::stoul(suffix) < bank.count) {
                    return bank.type;
                }
            }
        }
        throw std::runtime_error("undeclared register in oracle IR: " + name);
    }

    Value register_value(const std::string &name)
    {
        if (name == "%tid.x") return module_.local_invocation_x();
        const auto found = registers_.find(name);
        if (found == registers_.end()) {
            throw std::runtime_error("read before definition in oracle IR: " + name);
        }
        return found->second;
    }

    Value scalar_operand(const Json &operand, ValueType desired)
    {
        const std::string kind = operand.at("kind").get<std::string>();
        if (kind == "register") {
            return module_.coerce(
                register_value(register_operand_name(operand)), desired);
        }
        if (kind == "symbol") {
            const auto found =
                storage_offsets_.find(operand.at("name").get<std::string>());
            if (found != storage_offsets_.end()) {
                return module_.constant(desired, found->second);
            }
        }
        if (kind == "float_bits") {
            return module_.coerce(
                module_.constant(
                    ValueType::f32, operand.at("bits").get<std::uint32_t>()),
                desired);
        }
        if (kind == "integer") {
            return module_.constant(
                desired,
                static_cast<std::uint32_t>(operand.at("value").get<std::int64_t>()));
        }
        throw std::runtime_error("unsupported structured scalar operand kind '" + kind + "'");
    }

    Value address_expression(const Json &expression)
    {
        if (expression.at("kind").get<std::string>() != "add") {
            return scalar_operand(expression, ValueType::u32);
        }
        const Value left = address_expression(expression.at("left"));
        const Value right = address_expression(expression.at("right"));
        return module_.instruction(SpvOpIAdd, ValueType::u32, {left.id, right.id});
    }

    Value address_operand(const Json &operand)
    {
        if (operand.at("kind").get<std::string>() != "memory" ||
            operand.at("values").size() != 1u) {
            throw std::runtime_error("invalid structured address operand");
        }
        return address_expression(operand.at("values").at(0));
    }

    Value word_index(Value byte_address)
    {
        const Value address = module_.coerce(byte_address, ValueType::u32);
        return module_.instruction(
            SpvOpShiftRightLogical,
            ValueType::u32,
            {address.id, module_.constant(ValueType::u32, 2u).id});
    }

    void define(const std::string &name, Value value)
    {
        registers_[name] = module_.coerce(value, register_type(name));
    }

    bool has_modifier(const Json &instruction, const std::string &modifier) const
    {
        const auto &modifiers = instruction.at("modifiers");
        return std::find(modifiers.begin(), modifiers.end(), modifier) != modifiers.end();
    }

    Value float_input(const Json &operand, bool ftz)
    {
        Value value = scalar_operand(operand, ValueType::f32);
        return ftz ? module_.flush_ftz(value) : value;
    }

    Value float_output(Value value, bool ftz)
    {
        return ftz ? module_.flush_ftz(value) : value;
    }

    Value float_binary(SpvOp operation, const Json &instruction)
    {
        const Json &operands = instruction.at("structured_operands");
        const bool ftz = has_modifier(instruction, "ftz");
        const Value left = float_input(operands.at(1), ftz);
        const Value right = float_input(operands.at(2), ftz);
        Value result = module_.instruction(operation, ValueType::f32, {left.id, right.id});
        module_.decorate(result.id, SpvDecorationNoContraction);
        return float_output(result, ftz);
    }

    void lower_texture(const Json &instruction)
    {
        const Json &operands = instruction.at("structured_operands");
        const std::vector<std::string> destinations =
            register_operand_list(operands.at(0));
        const Json &texture = operands.at(1);
        if (destinations.size() != 4u ||
            texture.at("kind").get<std::string>() != "memory" ||
            texture.at("values").size() != 2u ||
            texture.at("values").at(0).at("kind").get<std::string>() != "symbol") {
            unsupported(instruction, "unsupported texture shape");
        }
        const std::string texture_name =
            texture.at("values").at(0).at("name").get<std::string>();
        if (texture_name != "probe_tex" && texture_name != "probe_tex_float") {
            unsupported(instruction, "unsupported texture reference");
        }
        const Json &coordinates = texture.at("values").at(1);
        if (coordinates.at("kind").get<std::string>() != "list" ||
            coordinates.at("values").size() != 1u) {
            unsupported(instruction, "unsupported texture coordinate shape");
        }
        const Value coordinate =
            scalar_operand(coordinates.at("values").at(0), ValueType::s32);
        const Value zero_s = module_.constant(ValueType::s32, 0u);
        const Value eight_s = module_.constant(ValueType::s32, 8u);
        const Value lower_ok = module_.instruction(
            SpvOpSGreaterThanEqual, ValueType::boolean, {coordinate.id, zero_s.id});
        const Value upper_ok = module_.instruction(
            SpvOpSLessThan, ValueType::boolean, {coordinate.id, eight_s.id});
        const Value in_bounds = module_.instruction(
            SpvOpLogicalAnd, ValueType::boolean, {lower_ok.id, upper_ok.id});
        const Value safe_signed = module_.instruction(
            SpvOpSelect, ValueType::s32, {in_bounds.id, coordinate.id, zero_s.id});
        const Value safe = module_.coerce(safe_signed, ValueType::u32);
        const Value row = module_.instruction(
            SpvOpIMul,
            ValueType::u32,
            {safe.id, module_.constant(ValueType::u32, 4u).id});
        const std::uint32_t input_base =
            texture_name == "probe_tex" ? 160u : 320u;
        const Value base = module_.instruction(
            SpvOpIAdd,
            ValueType::u32,
            {row.id, module_.constant(ValueType::u32, input_base).id});
        for (std::uint32_t component = 0u; component < 4u; ++component) {
            const Value index = module_.instruction(
                SpvOpIAdd,
                ValueType::u32,
                {base.id, module_.constant(ValueType::u32, component).id});
            const Value loaded = module_.load_buffer(index);
            const Value selected = module_.instruction(
                SpvOpSelect,
                ValueType::u32,
                {in_bounds.id, loaded.id, module_.constant(ValueType::u32, 0u).id});
            define(destinations[component], selected);
        }
    }

    void lower_instruction(const Json &instruction)
    {
        const std::string variant = instruction.at("variant").get<std::string>();
        const std::string base = instruction.at("base").get<std::string>();
        const Json &operands = instruction.at("structured_operands");
        const auto destination = [&]() {
            return register_operand_name(operands.at(0));
        };
        if (!instruction.at("predicate").get<std::string>().empty()) {
            unsupported(instruction, "predication is outside the oracle slice");
        }

        if (base == "ret") return;
        if (base == "ld" && has_modifier(instruction, "param")) {
            define(destination(),
                   module_.constant(ValueType::u32, binding_.output_word * 4u));
            return;
        }
        if (base == "mov") {
            const std::string name = destination();
            define(name, scalar_operand(operands.at(1), register_type(name)));
            return;
        }
        if (base == "div") {
            define(destination(), float_binary(SpvOpFDiv, instruction));
            return;
        }
        if (base == "rcp") {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value input = float_input(operands.at(1), ftz);
            Value result = module_.instruction(
                SpvOpFDiv,
                ValueType::f32,
                {module_.constant(ValueType::f32, 0x3f800000u).id, input.id});
            module_.decorate(result.id, SpvDecorationNoContraction);
            define(destination(), float_output(result, ftz));
            return;
        }
        if (base == "sqrt" || base == "rsqrt") {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value input = float_input(operands.at(1), ftz);
            const GLSLstd450 operation =
                base == "sqrt" ? GLSLstd450Sqrt : GLSLstd450InverseSqrt;
            define(destination(),
                   float_output(module_.ext_instruction(
                       ValueType::f32, operation, {input.id}), ftz));
            return;
        }
        if (base == "fma") {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value a = float_input(operands.at(1), ftz);
            const Value b = float_input(operands.at(2), ftz);
            const Value c = float_input(operands.at(3), ftz);
            Value result = module_.ext_instruction(
                ValueType::f32, GLSLstd450Fma, {a.id, b.id, c.id});
            module_.decorate(result.id, SpvDecorationNoContraction);
            define(destination(), float_output(result, ftz));
            return;
        }
        if (base == "mul" && has_modifier(instruction, "f32")) {
            define(destination(), float_binary(SpvOpFMul, instruction));
            return;
        }
        if (base == "add" && has_modifier(instruction, "f32")) {
            define(destination(), float_binary(SpvOpFAdd, instruction));
            return;
        }
        if (base == "add" || base == "sub" || base == "and" || base == "shl") {
            const std::string name = destination();
            const ValueType type = register_type(name);
            const Value left = scalar_operand(operands.at(1), type);
            const Value right = scalar_operand(
                operands.at(2), base == "shl" ? ValueType::u32 : type);
            const SpvOp operation = base == "add" ? SpvOpIAdd :
                base == "sub" ? SpvOpISub :
                base == "and" ? SpvOpBitwiseAnd : SpvOpShiftLeftLogical;
            define(name, module_.instruction(operation, type, {left.id, right.id}));
            return;
        }
        if (base == "mad") {
            const std::string name = destination();
            const ValueType type = register_type(name);
            const Value a = scalar_operand(operands.at(1), type);
            const Value b = scalar_operand(operands.at(2), type);
            const Value c = scalar_operand(operands.at(3), type);
            const Value product = module_.instruction(SpvOpIMul, type, {a.id, b.id});
            define(name, module_.instruction(SpvOpIAdd, type, {product.id, c.id}));
            return;
        }
        if (base == "cvt") {
            if (variant == "cvt.rzi.ftz.s32.f32") {
                const Value source = module_.flush_ftz(
                    scalar_operand(operands.at(1), ValueType::f32));
                define(destination(),
                       module_.instruction(SpvOpConvertFToS, ValueType::s32, {source.id}));
                return;
            }
            if (variant == "cvt.rn.f32.u32") {
                const Value source = scalar_operand(operands.at(1), ValueType::u32);
                define(destination(), module_.instruction(
                    SpvOpConvertUToF, ValueType::f32, {source.id}));
                return;
            }
            unsupported(instruction, "unsupported conversion");
        }
        if (base == "min" || base == "max") {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value left = float_input(operands.at(1), ftz);
            const Value right = float_input(operands.at(2), ftz);
            const GLSLstd450 operation =
                base == "min" ? GLSLstd450FMin : GLSLstd450FMax;
            const Value arithmetic = module_.ext_instruction(
                ValueType::f32, operation, {left.id, right.id});
            const Value left_nan = module_.instruction(
                SpvOpIsNan, ValueType::boolean, {left.id});
            const Value right_nan = module_.instruction(
                SpvOpIsNan, ValueType::boolean, {right.id});
            const Value right_checked = module_.instruction(
                SpvOpSelect, ValueType::f32, {right_nan.id, left.id, arithmetic.id});
            const Value selected = module_.instruction(
                SpvOpSelect, ValueType::f32, {left_nan.id, right.id, right_checked.id});
            define(destination(), float_output(selected, ftz));
            return;
        }
        if (base == "st" && has_modifier(instruction, "global")) {
            const Value address = address_operand(operands.at(0));
            const Value base_index = word_index(address);
            const std::vector<std::string> sources =
                register_operand_list(operands.at(1));
            for (std::size_t component = 0u; component < sources.size(); ++component) {
                Value index = base_index;
                if (component != 0u) {
                    index = module_.instruction(
                        SpvOpIAdd,
                        ValueType::u32,
                        {base_index.id,
                         module_.constant(
                             ValueType::u32,
                             static_cast<std::uint32_t>(component)).id});
                }
                module_.store_buffer(index, register_value(sources[component]));
            }
            return;
        }
        if (base == "st" && has_modifier(instruction, "shared")) {
            module_.store_shared(
                word_index(address_operand(operands.at(0))),
                register_value(register_operand_name(operands.at(1))));
            return;
        }
        if (base == "ld" && has_modifier(instruction, "shared")) {
            define(destination(),
                   module_.load_shared(word_index(address_operand(operands.at(1)))));
            return;
        }
        if (base == "ld" && has_modifier(instruction, "const")) {
            const Value loaded =
                module_.load_constant(word_index(address_operand(operands.at(1))));
            define(
                destination(),
                has_modifier(instruction, "f32")
                    ? module_.coerce(loaded, ValueType::f32)
                    : loaded);
            return;
        }
        if (base == "st" && has_modifier(instruction, "local")) {
            if (local_ == 0u) {
                unsupported(instruction, "local store without local storage");
            }
            module_.store_local(
                local_,
                word_index(address_operand(operands.at(0))),
                register_value(register_operand_name(operands.at(1))));
            return;
        }
        if (base == "ld" && has_modifier(instruction, "local")) {
            if (local_ == 0u) {
                unsupported(instruction, "local load without local storage");
            }
            define(
                destination(),
                module_.load_local(
                    local_, word_index(address_operand(operands.at(1)))));
            return;
        }
        if (base == "bar" && variant == "bar.sync") {
            const Value workgroup = module_.constant(ValueType::u32, 2u);
            const Value semantics = module_.constant(ValueType::u32, 0x108u);
            module_.function_instruction(
                SpvOpControlBarrier, {workgroup.id, workgroup.id, semantics.id});
            return;
        }
        if (base == "tex" &&
            (variant == "tex.1d.v4.u32.s32" ||
             variant == "tex.1d.v4.f32.s32")) {
            lower_texture(instruction);
            return;
        }
        unsupported(instruction, "unsupported instruction");
    }

    SpirvModule &module_;
    const Json &entry_;
    OracleBinding binding_;
    std::uint32_t function_;
    std::vector<RegisterBank> banks_;
    std::map<std::string, Value> registers_;
    std::map<std::string, std::uint32_t> storage_offsets_;
    std::uint32_t local_ = 0u;
};


class KernelEntryLowerer {
public:
    KernelEntryLowerer(
        SpirvModule &module,
        const Json &manifest,
        const Json &entry,
        std::uint32_t function,
        std::uint32_t local_size,
        std::uint32_t dynamic_shared_bytes)
        : module_(module),
          entry_(entry),
          function_(function),
          local_size_(local_size),
          dynamic_shared_bytes_(dynamic_shared_bytes)
    {
        const std::uint32_t module_shared =
            module_static_shared_bytes(manifest);
        const std::uint32_t total_static =
            required_shared_bytes(manifest, entry, 0u);
        for (const Json &variable : manifest.at("variables")) {
            if (variable.at("space").get<std::string>() != "shared") {
                continue;
            }
            const bool external = variable.at("external").get<bool>();
            add_storage_symbol(
                variable.at("name").get<std::string>(),
                external
                    ? total_static
                    : variable.at("offset").get<std::uint32_t>(),
                external);
        }
        for (const Json &variable : manifest.at("variables")) {
            if (variable.at("space").get<std::string>() == "const") {
                add_storage_symbol(
                    variable.at("name").get<std::string>(),
                    variable.at("offset").get<std::uint32_t>(),
                    false);
            }
        }
        std::uint32_t texture_index = 0u;
        for (const Json &name : manifest.at("texture_references")) {
            texture_references_.emplace(
                name.get<std::string>(), texture_index++);
        }
        for (const Json &variable : entry_.at("variables")) {
            const std::string space =
                variable.at("space").get<std::string>();
            const bool external = variable.at("external").get<bool>();
            std::uint32_t offset =
                variable.at("offset").get<std::uint32_t>();
            if (space == "shared") {
                offset = external ? total_static : module_shared + offset;
            } else if (space != "local") {
                continue;
            }
            add_storage_symbol(
                variable.at("name").get<std::string>(), offset, external);
        }
    }
    void lower()
    {
        module_.define_kernel_entry_point(function_, local_size_);
        module_.begin_function(function_);
        declare_registers();
        if (entry_.at("static_local_bytes").get<std::uint32_t>() != 0u) {
            local_ = module_.function_u32_array(
                entry_.at("static_local_bytes").get<std::uint32_t>());
        }
        const std::uint32_t pc = module_.function_variable(ValueType::u32);
        const Json &blocks = entry_.at("cfg").at("blocks");
        const std::uint32_t exit_pc = static_cast<std::uint32_t>(blocks.size());
        barrier_prelude_ = has_barrier();
        const Value initial_pc = barrier_prelude_
            ? lower_uniform_barrier_prelude(exit_pc)
            : module_.constant(ValueType::u32, 0u);
        module_.store_variable(pc, initial_pc, ValueType::u32);

        const std::uint32_t loop_header = module_.allocate_id();
        const std::uint32_t dispatch = module_.allocate_id();
        const std::uint32_t switch_merge = module_.allocate_id();
        const std::uint32_t loop_continue = module_.allocate_id();
        const std::uint32_t loop_merge = module_.allocate_id();
        const std::uint32_t invalid_case = module_.allocate_id();
        std::vector<std::uint32_t> case_labels(blocks.size());
        for (std::uint32_t &label : case_labels) {
            label = module_.allocate_id();
        }

        module_.function_instruction(SpvOpBranch, {loop_header});
        module_.function_instruction(SpvOpLabel, {loop_header});
        module_.function_instruction(
            SpvOpLoopMerge,
            {loop_merge, loop_continue, SpvLoopControlMaskNone});
        module_.function_instruction(SpvOpBranch, {dispatch});
        module_.function_instruction(SpvOpLabel, {dispatch});
        const Value selector = module_.load_variable(pc, ValueType::u32);
        module_.function_instruction(
            SpvOpSelectionMerge,
            {switch_merge, SpvSelectionControlMaskNone});
        std::vector<std::uint32_t> switch_operands = {
            selector.id, invalid_case,
        };
        for (std::uint32_t block = 0u; block < case_labels.size(); ++block) {
            switch_operands.push_back(block);
            switch_operands.push_back(case_labels[block]);
        }
        module_.function_instruction(SpvOpSwitch, switch_operands);

        const Json &instructions = entry_.at("instructions");
        for (std::uint32_t block = 0u; block < case_labels.size(); ++block) {
            module_.function_instruction(SpvOpLabel, {case_labels[block]});
            const Json &description = blocks.at(block);
            const std::size_t begin =
                description.at("instruction_begin").get<std::size_t>();
            const std::size_t end =
                description.at("instruction_end").get<std::size_t>();
            for (std::size_t index = begin; index < end; ++index) {
                const Json &instruction = instructions.at(index);
                const std::string base =
                    instruction.at("base").get<std::string>();
                if (base != "bra" && base != "ret") {
                    lower_instruction(instruction);
                }
            }
            const Value next = next_block(
                block, instructions.at(end - 1u), exit_pc);
            module_.store_variable(pc, next, ValueType::u32);
            module_.function_instruction(SpvOpBranch, {switch_merge});
        }

        module_.function_instruction(SpvOpLabel, {invalid_case});
        module_.store_variable(
            pc, module_.constant(ValueType::u32, exit_pc), ValueType::u32);
        module_.function_instruction(SpvOpBranch, {switch_merge});

        module_.function_instruction(SpvOpLabel, {switch_merge});
        const Value current = module_.load_variable(pc, ValueType::u32);
        const Value finished = module_.instruction(
            SpvOpIEqual,
            ValueType::boolean,
            {current.id, module_.constant(ValueType::u32, exit_pc).id});
        module_.function_instruction(
            SpvOpBranchConditional,
            {finished.id, loop_merge, loop_continue});
        module_.function_instruction(SpvOpLabel, {loop_continue});
        module_.function_instruction(SpvOpBranch, {loop_header});
        module_.function_instruction(SpvOpLabel, {loop_merge});
        module_.end_function();
    }

private:
    struct RegisterVariable {
        ValueType type;
        std::uint32_t id;
    };
    struct StorageSymbol {
        std::uint32_t offset;
        bool external_shared;
    };

    void add_storage_symbol(
        const std::string &name,
        std::uint32_t offset,
        bool external_shared)
    {
        if (!storage_symbols_.emplace(
                name, StorageSymbol{offset, external_shared}).second) {
            throw std::runtime_error(
                entry_.at("name").get<std::string>() +
                ": duplicate storage symbol '" + name + "'");
        }
    }


    [[noreturn]] void unsupported(
        const Json &instruction, const std::string &detail) const
    {
        throw std::runtime_error(
            entry_.at("name").get<std::string>() + ": " + detail + " for '" +
            instruction.at("variant").get<std::string>() + "' at PTX line " +
            std::to_string(instruction.at("line").get<std::size_t>()));
    }

    bool has_barrier() const
    {
        for (const Json &instruction : entry_.at("instructions")) {
            if (instruction.at("variant").get<std::string>() == "bar.sync") {
                return true;
            }
        }
        return false;
    }

    void lower_block_instructions(std::uint32_t block, bool skip_terminal)
    {
        const Json &description = entry_.at("cfg").at("blocks").at(block);
        const std::size_t begin =
            description.at("instruction_begin").get<std::size_t>();
        std::size_t end =
            description.at("instruction_end").get<std::size_t>();
        if (skip_terminal) {
            --end;
        }
        for (std::size_t index = begin; index < end; ++index) {
            lower_instruction(entry_.at("instructions").at(index));
        }
    }

    Value lower_uniform_barrier_prelude(std::uint32_t exit_pc)
    {
        const Json &blocks = entry_.at("cfg").at("blocks");
        const Json &instructions = entry_.at("instructions");
        if (blocks.size() < 5u ||
            instructions.at(
                blocks.at(4).at("instruction_begin").get<std::size_t>())
                    .at("variant").get<std::string>() != "bar.sync") {
            throw std::runtime_error(
                "selected-kernel barrier CFG is outside the finite prelude shape");
        }

        lower_block_instructions(0u, true);
        const std::size_t block0_end =
            blocks.at(0).at("instruction_end").get<std::size_t>();
        const Value inactive =
            predicate_value(
                instructions.at(block0_end - 1u)
                    .at("predicate").get<std::string>());
        const std::uint32_t active_label = module_.allocate_id();
        const std::uint32_t active_merge = module_.allocate_id();
        module_.function_instruction(
            SpvOpSelectionMerge,
            {active_merge, SpvSelectionControlMaskNone});
        module_.function_instruction(
            SpvOpBranchConditional,
            {inactive.id, active_merge, active_label});
        module_.function_instruction(SpvOpLabel, {active_label});

        lower_block_instructions(1u, true);
        const std::size_t block1_end =
            blocks.at(1).at("instruction_end").get<std::size_t>();
        const Value skip_previous =
            predicate_value(
                instructions.at(block1_end - 1u)
                    .at("predicate").get<std::string>());
        const std::uint32_t previous_label = module_.allocate_id();
        const std::uint32_t previous_merge = module_.allocate_id();
        module_.function_instruction(
            SpvOpSelectionMerge,
            {previous_merge, SpvSelectionControlMaskNone});
        module_.function_instruction(
            SpvOpBranchConditional,
            {skip_previous.id, previous_merge, previous_label});
        module_.function_instruction(SpvOpLabel, {previous_label});
        lower_block_instructions(2u, true);
        lower_block_instructions(3u, false);
        module_.function_instruction(SpvOpBranch, {previous_merge});
        module_.function_instruction(SpvOpLabel, {previous_merge});
        module_.function_instruction(SpvOpBranch, {active_merge});
        module_.function_instruction(SpvOpLabel, {active_merge});

        const Value workgroup = module_.constant(ValueType::u32, 2u);
        const Value semantics = module_.constant(ValueType::u32, 0x108u);
        module_.function_instruction(
            SpvOpControlBarrier,
            {workgroup.id, workgroup.id, semantics.id});

        const std::size_t barrier_end =
            blocks.at(4).at("instruction_end").get<std::size_t>();
        return next_block(
            4u, instructions.at(barrier_end - 1u), exit_pc);
    }

    void declare_registers()
    {
        std::map<std::string, bool> referenced;
        for (const Json &instruction : entry_.at("instructions")) {
            for (const char *field : {"definitions", "uses"}) {
                for (const Json &name : instruction.at(field)) {
                    referenced.emplace(name.get<std::string>(), true);
                }
            }
        }
        for (const Json &bank : entry_.at("register_banks")) {
            const std::string prefix = bank.at("prefix").get<std::string>();
            const std::string type_name = bank.at("type").get<std::string>();
            const std::size_t count = bank.at("count").get<std::size_t>();
            const bool supported =
                type_name == ".pred" || type_name == ".u32" ||
                type_name == ".s32" || type_name == ".b32" ||
                type_name == ".f32";
            bool bank_referenced = false;
            for (std::size_t index = 0u; index < count && !bank_referenced; ++index) {
                const std::string name =
                    count == 1u ? prefix : prefix + std::to_string(index);
                bank_referenced = referenced.find(name) != referenced.end();
            }
            if (!supported && !bank_referenced) {
                continue;
            }
            const ValueType type = parse_type(type_name);
            if (count == 1u) {
                variables_.emplace(
                    prefix, RegisterVariable{type, module_.function_variable(type)});
                continue;
            }
            for (std::size_t index = 0u; index < count; ++index) {
                variables_.emplace(
                    prefix + std::to_string(index),
                    RegisterVariable{type, module_.function_variable(type)});
            }
        }
    }

    RegisterVariable variable(const std::string &name) const
    {
        const auto found = variables_.find(name);
        if (found == variables_.end()) {
            throw std::runtime_error(
                entry_.at("name").get<std::string>() +
                ": undeclared register '" + name + "'");
        }
        return found->second;
    }

    Value read_register(const std::string &name)
    {
        if (name == "%tid.x") {
            return module_.local_invocation_x();
        }
        if (name == "%ntid.x") {
            return module_.constant(ValueType::u32, local_size_);
        }
        if (name == "%ctaid.x") {
            return module_.workgroup_invocation_x();
        }
        const RegisterVariable source = variable(name);
        return module_.load_variable(source.id, source.type);
    }

    void define(const Json &operand, Value value)
    {
        const RegisterVariable destination =
            variable(register_operand_name(operand));
        module_.store_variable(destination.id, value, destination.type);
    }

    Value scalar_operand(const Json &operand, ValueType desired)
    {
        const std::string kind = operand.at("kind").get<std::string>();
        if (kind == "register") {
            return module_.coerce(
                read_register(register_operand_name(operand)), desired);
        }
        if (kind == "float_bits") {
            return module_.coerce(
                module_.constant(
                    ValueType::f32, operand.at("bits").get<std::uint32_t>()),
                desired);
        }
        if (kind == "integer") {
            return module_.constant(
                desired,
                static_cast<std::uint32_t>(
                    operand.at("value").get<std::int64_t>()));
        }
        if (kind == "symbol") {
            const std::string name =
                operand.at("name").get<std::string>();
            const auto found = storage_symbols_.find(name);
            if (found != storage_symbols_.end()) {
                if (found->second.external_shared &&
                    dynamic_shared_bytes_ == 0u) {
                    throw std::runtime_error(
                        entry_.at("name").get<std::string>() +
                        ": external shared symbol '" + name +
                        "' requires nonzero dynamic shared bytes");
                }
                return module_.constant(desired, found->second.offset);
            }
        }
        throw std::runtime_error(
            "unsupported selected-kernel scalar operand kind '" + kind + "'");
    }

    Value address_expression(const Json &expression)
    {
        if (expression.at("kind").get<std::string>() != "add") {
            return scalar_operand(expression, ValueType::u32);
        }
        const Value left = address_expression(expression.at("left"));
        const Value right = address_expression(expression.at("right"));
        return module_.instruction(
            SpvOpIAdd, ValueType::u32, {left.id, right.id});
    }

    Value address_operand(const Json &operand)
    {
        if (operand.at("kind").get<std::string>() != "memory" ||
            operand.at("values").size() != 1u) {
            throw std::runtime_error(
                "invalid selected-kernel structured address");
        }
        return address_expression(operand.at("values").at(0));
    }

    void parameter_address(
        const Json &expression,
        std::string &name,
        std::uint32_t &offset) const
    {
        const std::string kind = expression.at("kind").get<std::string>();
        if (kind == "symbol") {
            name = expression.at("name").get<std::string>();
            return;
        }
        if (kind != "add") {
            throw std::runtime_error(
                "unsupported selected-kernel parameter address");
        }
        parameter_address(expression.at("left"), name, offset);
        const Json &right = expression.at("right");
        if (right.at("kind").get<std::string>() != "integer") {
            throw std::runtime_error(
                "non-constant selected-kernel parameter offset");
        }
        offset += static_cast<std::uint32_t>(
            right.at("value").get<std::int64_t>());
    }

    Value load_parameter(const Json &operand)
    {
        if (operand.at("kind").get<std::string>() != "memory" ||
            operand.at("values").size() != 1u) {
            throw std::runtime_error(
                "invalid selected-kernel parameter operand");
        }
        std::string name;
        std::uint32_t inner_offset = 0u;
        parameter_address(
            operand.at("values").at(0), name, inner_offset);
        for (const Json &parameter : entry_.at("parameters")) {
            if (parameter.at("name").get<std::string>() == name) {
                const std::uint32_t byte_offset =
                    parameter.at("offset").get<std::uint32_t>() + inner_offset;
                if ((byte_offset & 3u) != 0u) {
                    throw std::runtime_error(
                        "unaligned selected-kernel u32 parameter load");
                }
                return module_.load_parameter_word(byte_offset / 4u);
            }
        }
        throw std::runtime_error(
            "unknown selected-kernel parameter '" + name + "'");
    }

    Value word_index(Value byte_address)
    {
        const Value address =
            module_.coerce(byte_address, ValueType::u32);
        return module_.instruction(
            SpvOpShiftRightLogical,
            ValueType::u32,
            {address.id, module_.constant(ValueType::u32, 2u).id});
    }

    bool has_modifier(
        const Json &instruction, const std::string &modifier) const
    {
        const auto &modifiers = instruction.at("modifiers");
        return std::find(
            modifiers.begin(), modifiers.end(), modifier) != modifiers.end();
    }

    Value float_input(const Json &operand, bool ftz)
    {
        Value value = scalar_operand(operand, ValueType::f32);
        return ftz ? module_.flush_ftz(value) : value;
    }

    Value float_output(Value value, bool ftz)
    {
        return ftz ? module_.flush_ftz(value) : value;
    }

    Value predicate_value(const std::string &predicate)
    {
        const std::string normalized =
            !predicate.empty() && predicate.front() == '@'
                ? predicate.substr(1u)
                : predicate;
        const bool inverted =
            !normalized.empty() && normalized.front() == '!';
        const std::string name =
            inverted ? normalized.substr(1u) : normalized;
        Value value = read_register(name);
        if (value.type != ValueType::boolean) {
            throw std::runtime_error(
                "selected-kernel branch predicate is not boolean");
        }
        if (inverted) {
            value = module_.instruction(
                SpvOpLogicalNot, ValueType::boolean, {value.id});
        }
        return value;
    }

    std::uint32_t block_number(const std::string &id) const
    {
        if (id.size() < 2u || id.front() != 'b') {
            throw std::runtime_error("invalid CFG block id '" + id + "'");
        }
        return static_cast<std::uint32_t>(std::stoul(id.substr(1u)));
    }

    Value next_block(
        std::uint32_t block,
        const Json &terminal,
        std::uint32_t exit_pc)
    {
        const std::string base = terminal.at("base").get<std::string>();
        const Json &successors =
            entry_.at("cfg").at("blocks").at(block).at("successors");
        if (base == "ret" || successors.empty()) {
            return module_.constant(ValueType::u32, exit_pc);
        }
        const Value target = module_.constant(
            ValueType::u32,
            block_number(successors.at(0).get<std::string>()));
        const std::string predicate =
            terminal.at("predicate").get<std::string>();
        if (predicate.empty()) {
            return target;
        }
        if (successors.size() != 2u) {
            throw std::runtime_error(
                "predicated branch does not have two CFG successors");
        }
        const Value fallthrough = module_.constant(
            ValueType::u32,
            block_number(successors.at(1).get<std::string>()));
        const Value condition = predicate_value(predicate);
        return module_.instruction(
            SpvOpSelect,
            ValueType::u32,
            {condition.id, target.id, fallthrough.id});
    }

    void lower_texture(const Json &instruction)
    {
        const Json &operands = instruction.at("structured_operands");
        const Json &destinations = operands.at(0);
        const Json &texture = operands.at(1);
        if (destinations.at("kind").get<std::string>() != "list" ||
            destinations.at("values").size() != 4u ||
            texture.at("kind").get<std::string>() != "memory" ||
            texture.at("values").size() != 2u ||
            texture.at("values").at(0).at("kind").get<std::string>() != "symbol") {
            unsupported(instruction, "unsupported texture shape");
        }
        const std::string texture_name =
            texture.at("values").at(0).at("name").get<std::string>();
        const auto reference = texture_references_.find(texture_name);
        if (reference == texture_references_.end()) {
            unsupported(instruction, "unknown texture reference");
        }
        const Json &coordinates = texture.at("values").at(1);
        if (coordinates.at("kind").get<std::string>() != "list" ||
            coordinates.at("values").size() != 1u) {
            unsupported(instruction, "unsupported texture coordinate shape");
        }

        const Value coordinate =
            scalar_operand(coordinates.at("values").at(0), ValueType::s32);
        const Value coordinate_u = module_.coerce(coordinate, ValueType::u32);
        const Value base_word =
            module_.load_texture_metadata(reference->second, kTextureBaseWord);
        const Value element_count =
            module_.load_texture_metadata(reference->second, kTextureElementCount);
        const Value component_count =
            module_.load_texture_metadata(reference->second, kTextureComponentCount);
        const Value zero_s = module_.constant(ValueType::s32, 0u);
        const Value zero_u = module_.constant(ValueType::u32, 0u);
        const Value lower_ok = module_.instruction(
            SpvOpSGreaterThanEqual,
            ValueType::boolean,
            {coordinate.id, zero_s.id});
        const Value upper_ok = module_.instruction(
            SpvOpULessThan,
            ValueType::boolean,
            {coordinate_u.id, element_count.id});
        const Value in_bounds = module_.instruction(
            SpvOpLogicalAnd,
            ValueType::boolean,
            {lower_ok.id, upper_ok.id});
        const Value safe_coordinate = module_.instruction(
            SpvOpSelect,
            ValueType::u32,
            {in_bounds.id, coordinate_u.id, zero_u.id});
        const Value row_offset = module_.instruction(
            SpvOpIMul,
            ValueType::u32,
            {safe_coordinate.id, component_count.id});

        const bool float_result =
            instruction.at("variant").get<std::string>() ==
            "tex.1d.v4.f32.s32";
        for (std::uint32_t component = 0u; component < 4u; ++component) {
            const Value component_value =
                module_.constant(ValueType::u32, component);
            const Value component_present = module_.instruction(
                SpvOpULessThan,
                ValueType::boolean,
                {component_value.id, component_count.id});
            const Value safe_component = module_.instruction(
                SpvOpSelect,
                ValueType::u32,
                {component_present.id, component_value.id, zero_u.id});
            const Value component_offset = module_.instruction(
                SpvOpIAdd,
                ValueType::u32,
                {row_offset.id, safe_component.id});
            const Value word = module_.instruction(
                SpvOpIAdd,
                ValueType::u32,
                {base_word.id, component_offset.id});
            const Value loaded = module_.load_buffer(word);
            const Value defined = module_.instruction(
                SpvOpLogicalAnd,
                ValueType::boolean,
                {in_bounds.id, component_present.id});
            const Value selected = module_.instruction(
                SpvOpSelect,
                ValueType::u32,
                {defined.id, loaded.id, zero_u.id});
            define(
                destinations.at("values").at(component),
                float_result
                    ? module_.coerce(selected, ValueType::f32)
                    : selected);
        }
    }

    void lower_instruction(const Json &instruction)
    {
        const std::string base =
            instruction.at("base").get<std::string>();
        const std::string variant =
            instruction.at("variant").get<std::string>();
        const Json &operands = instruction.at("structured_operands");
        if (!instruction.at("predicate").get<std::string>().empty()) {
            unsupported(
                instruction, "non-branch predication is outside this slice");
        }
        if (base == "tex" &&
            (variant == "tex.1d.v4.u32.s32" ||
             variant == "tex.1d.v4.f32.s32")) {
            lower_texture(instruction);
            return;
        }

        if (base == "ld" && has_modifier(instruction, "param")) {
            define(operands.at(0), load_parameter(operands.at(1)));
            return;
        }
        if (base == "ld" && has_modifier(instruction, "global")) {
            const Value loaded =
                module_.load_buffer(word_index(address_operand(operands.at(1))));
            define(
                operands.at(0),
                has_modifier(instruction, "f32")
                    ? module_.coerce(loaded, ValueType::f32)
                    : loaded);
            return;
        }
        if (base == "ld" && has_modifier(instruction, "shared")) {
            const Value loaded =
                module_.load_shared(word_index(address_operand(operands.at(1))));
            define(operands.at(0), loaded);
            return;
        }
        if (base == "ld" && has_modifier(instruction, "const")) {
            const Value loaded =
                module_.load_constant(word_index(address_operand(operands.at(1))));
            define(
                operands.at(0),
                has_modifier(instruction, "f32")
                    ? module_.coerce(loaded, ValueType::f32)
                    : loaded);
            return;
        }
        if (base == "ld" && has_modifier(instruction, "local")) {
            if (local_ == 0u) {
                unsupported(instruction, "local load without local storage");
            }
            const Value loaded = module_.load_local(
                local_, word_index(address_operand(operands.at(1))));
            define(
                operands.at(0),
                has_modifier(instruction, "f32")
                    ? module_.coerce(loaded, ValueType::f32)
                    : loaded);
            return;
        }
        if (base == "st" && has_modifier(instruction, "global")) {
            const Value base_index = word_index(address_operand(operands.at(0)));
            const std::vector<std::string> sources =
                register_operand_list(operands.at(1));
            for (std::size_t component = 0u; component < sources.size(); ++component) {
                Value index = base_index;
                if (component != 0u) {
                    index = module_.instruction(
                        SpvOpIAdd,
                        ValueType::u32,
                        {base_index.id,
                         module_.constant(
                             ValueType::u32,
                             static_cast<std::uint32_t>(component)).id});
                }
                module_.store_buffer(index, read_register(sources[component]));
            }
            return;
        }
        if (base == "st" && has_modifier(instruction, "shared")) {
            module_.store_shared(
                word_index(address_operand(operands.at(0))),
                scalar_operand(operands.at(1), ValueType::u32));
            return;
        }
        if (base == "st" && has_modifier(instruction, "local")) {
            if (local_ == 0u) {
                unsupported(instruction, "local store without local storage");
            }
            module_.store_local(
                local_,
                word_index(address_operand(operands.at(0))),
                scalar_operand(operands.at(1), ValueType::u32));
            return;
        }
        if (base == "mov" || base == "cvta") {
            const ValueType type =
                has_modifier(instruction, "f32")
                    ? ValueType::f32
                    : ValueType::u32;
            define(operands.at(0), scalar_operand(operands.at(1), type));
            return;
        }
        if ((base == "add" || base == "sub" || base == "mul") &&
            has_modifier(instruction, "f32")) {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value a = float_input(operands.at(1), ftz);
            const Value b = float_input(operands.at(2), ftz);
            const SpvOp operation =
                base == "add" ? SpvOpFAdd :
                base == "sub" ? SpvOpFSub :
                SpvOpFMul;
            Value result = module_.instruction(
                operation, ValueType::f32, {a.id, b.id});
            module_.decorate(result.id, SpvDecorationNoContraction);
            define(operands.at(0), float_output(result, ftz));
            return;
        }
        if (base == "fma" && has_modifier(instruction, "f32")) {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value a = float_input(operands.at(1), ftz);
            const Value b = float_input(operands.at(2), ftz);
            const Value c = float_input(operands.at(3), ftz);
            Value result = module_.ext_instruction(
                ValueType::f32, GLSLstd450Fma, {a.id, b.id, c.id});
            module_.decorate(result.id, SpvDecorationNoContraction);
            define(operands.at(0), float_output(result, ftz));
            return;
        }
        if (base == "add" || base == "sub" || base == "mad") {
            const Value a =
                scalar_operand(operands.at(1), ValueType::s32);
            const Value b =
                scalar_operand(operands.at(2), ValueType::s32);
            if (base == "add" || base == "sub") {
                define(
                    operands.at(0),
                    module_.instruction(
                        base == "add" ? SpvOpIAdd : SpvOpISub,
                        ValueType::s32,
                        {a.id, b.id}));
                return;
            }
            const Value c =
                scalar_operand(operands.at(3), ValueType::s32);
            const Value product = module_.instruction(
                SpvOpIMul, ValueType::s32, {a.id, b.id});
            define(
                operands.at(0),
                module_.instruction(
                    SpvOpIAdd, ValueType::s32, {product.id, c.id}));
            return;
        }
        if (variant == "and.pred") {
            const Value a =
                scalar_operand(operands.at(1), ValueType::boolean);
            const Value b =
                scalar_operand(operands.at(2), ValueType::boolean);
            define(
                operands.at(0),
                module_.instruction(
                    SpvOpLogicalAnd,
                    ValueType::boolean,
                    {a.id, b.id}));
            return;
        }
        if (base == "and" || base == "or" ||
            base == "shl" || base == "shr") {
            const Value a =
                scalar_operand(operands.at(1), ValueType::u32);
            const Value b =
                scalar_operand(operands.at(2), ValueType::u32);
            const SpvOp operation =
                base == "and" ? SpvOpBitwiseAnd :
                base == "or" ? SpvOpBitwiseOr :
                base == "shl" ? SpvOpShiftLeftLogical :
                SpvOpShiftRightLogical;
            define(
                operands.at(0),
                module_.instruction(
                    operation, ValueType::u32, {a.id, b.id}));
            return;
        }
        if (base == "setp" && has_modifier(instruction, "f32")) {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value a = float_input(operands.at(1), ftz);
            const Value b = float_input(operands.at(2), ftz);
            SpvOp operation = SpvOpNop;
            if (variant == "setp.gt.ftz.f32") {
                operation = SpvOpFOrdGreaterThan;
            }
            if (variant == "setp.lt.ftz.f32") {
                operation = SpvOpFOrdLessThan;
            }
            if (variant == "setp.leu.ftz.f32") {
                operation = SpvOpFUnordLessThanEqual;
            }
            if (variant == "setp.neu.ftz.f32") {
                operation = SpvOpFUnordNotEqual;
            }
            if (operation == SpvOpNop) {
                unsupported(instruction, "unsupported float comparison");
            }
            define(
                operands.at(0),
                module_.instruction(
                    operation, ValueType::boolean, {a.id, b.id}));
            return;
        }
        if (base == "setp") {
            const bool unsigned_compare = has_modifier(instruction, "u32");
            const ValueType type =
                unsigned_compare ? ValueType::u32 : ValueType::s32;
            const Value a = scalar_operand(operands.at(1), type);
            const Value b = scalar_operand(operands.at(2), type);
            SpvOp operation = SpvOpNop;
            if (variant == "setp.ge.u32") operation = SpvOpUGreaterThanEqual;
            if (variant == "setp.gt.u32") operation = SpvOpUGreaterThan;
            if (variant == "setp.le.u32") operation = SpvOpULessThanEqual;
            if (variant == "setp.eq.s32") operation = SpvOpIEqual;
            if (variant == "setp.ne.s32") operation = SpvOpINotEqual;
            if (variant == "setp.gt.s32") operation = SpvOpSGreaterThan;
            if (operation == SpvOpNop) {
                unsupported(instruction, "unsupported comparison");
            }
            define(
                operands.at(0),
                module_.instruction(
                    operation, ValueType::boolean, {a.id, b.id}));
            return;
        }
        if (variant == "selp.f32") {
            const Value when_true =
                scalar_operand(operands.at(1), ValueType::f32);
            const Value when_false =
                scalar_operand(operands.at(2), ValueType::f32);
            const Value condition =
                scalar_operand(operands.at(3), ValueType::boolean);
            define(
                operands.at(0),
                module_.instruction(
                    SpvOpSelect,
                    ValueType::f32,
                    {condition.id, when_true.id, when_false.id}));
            return;
        }
        if (variant == "bar.sync") {
            if (!barrier_prelude_) {
                const Value workgroup = module_.constant(ValueType::u32, 2u);
                const Value semantics = module_.constant(ValueType::u32, 0x108u);
                module_.function_instruction(
                    SpvOpControlBarrier,
                    {workgroup.id, workgroup.id, semantics.id});
            }
            return;
        }
        if (base == "div") {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value a = float_input(operands.at(1), ftz);
            const Value b = float_input(operands.at(2), ftz);
            Value result = module_.instruction(
                SpvOpFDiv, ValueType::f32, {a.id, b.id});
            module_.decorate(result.id, SpvDecorationNoContraction);
            define(operands.at(0), float_output(result, ftz));
            return;
        }
        if (base == "rcp") {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value input = float_input(operands.at(1), ftz);
            Value result = module_.instruction(
                SpvOpFDiv,
                ValueType::f32,
                {module_.constant(ValueType::f32, 0x3f800000u).id, input.id});
            module_.decorate(result.id, SpvDecorationNoContraction);
            define(operands.at(0), float_output(result, ftz));
            return;
        }
        if (base == "sqrt") {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value input = float_input(operands.at(1), ftz);
            define(
                operands.at(0),
                float_output(
                    module_.ext_instruction(
                        ValueType::f32, GLSLstd450Sqrt, {input.id}),
                    ftz));
            return;
        }
        if (base == "max" && has_modifier(instruction, "f32")) {
            const bool ftz = has_modifier(instruction, "ftz");
            const Value left = float_input(operands.at(1), ftz);
            const Value right = float_input(operands.at(2), ftz);
            const Value arithmetic = module_.ext_instruction(
                ValueType::f32, GLSLstd450FMax, {left.id, right.id});
            const Value left_nan = module_.instruction(
                SpvOpIsNan, ValueType::boolean, {left.id});
            const Value right_nan = module_.instruction(
                SpvOpIsNan, ValueType::boolean, {right.id});
            const Value right_checked = module_.instruction(
                SpvOpSelect,
                ValueType::f32,
                {right_nan.id, left.id, arithmetic.id});
            const Value selected = module_.instruction(
                SpvOpSelect,
                ValueType::f32,
                {left_nan.id, right.id, right_checked.id});
            define(operands.at(0), float_output(selected, ftz));
            return;
        }
        if (variant == "cvt.rmi.ftz.f32.f32") {
            const Value input = float_input(operands.at(1), true);
            define(
                operands.at(0),
                float_output(
                    module_.ext_instruction(
                        ValueType::f32, GLSLstd450Floor, {input.id}),
                    true));
            return;
        }
        if (variant == "cvt.rzi.ftz.s32.f32") {
            const Value input = float_input(operands.at(1), true);
            define(
                operands.at(0),
                module_.instruction(
                    SpvOpConvertFToS,
                    ValueType::s32,
                    {input.id}));
            return;
        }
        if (base == "min" || base == "max") {
            const Value a =
                scalar_operand(operands.at(1), ValueType::s32);
            const Value b =
                scalar_operand(operands.at(2), ValueType::s32);
            define(
                operands.at(0),
                module_.ext_instruction(
                    ValueType::s32,
                    base == "min" ? GLSLstd450SMin : GLSLstd450SMax,
                    {a.id, b.id}));
            return;
        }
        unsupported(instruction, "unsupported selected-kernel instruction");
    }

    SpirvModule &module_;
    const Json &entry_;
    std::uint32_t function_;
    std::uint32_t local_size_;
    bool barrier_prelude_ = false;
    std::map<std::string, RegisterVariable> variables_;
    std::uint32_t dynamic_shared_bytes_;
    std::uint32_t local_ = 0u;
    std::map<std::string, StorageSymbol> storage_symbols_;
    std::map<std::string, std::uint32_t> texture_references_;
};
const Json &find_entry(const Json &manifest, const std::string &name)
{
    for (const Json &entry : manifest.at("entries")) {
        if (entry.at("name").get<std::string>() == name) return entry;
    }
    throw std::runtime_error("oracle IR is missing entry '" + name + "'");
}

std::vector<std::uint32_t> emit_oracle(const Json &manifest)
{
    if (manifest.at("schema_version").get<std::uint32_t>() != 5u) {
        throw std::runtime_error("oracle emitter requires typed manifest schema 5");
    }
    if (manifest.at("address_size").get<std::uint32_t>() != 32u) {
        throw std::runtime_error("oracle emitter requires 32-bit PTX addresses");
    }
    std::uint32_t shared_bytes = 0u;
    for (const Json &entry : manifest.at("entries")) {
        shared_bytes = std::max(
            shared_bytes, required_shared_bytes(manifest, entry, 0u));
    }
    SpirvModule module(
        shared_bytes, module_constant_bytes(manifest), false);
    const std::uint32_t math = module.allocate_id();
    const std::uint32_t barrier = module.allocate_id();
    const std::uint32_t local = module.allocate_id();
    const std::uint32_t texture = module.allocate_id();
    const std::uint32_t texture_float = module.allocate_id();
    const std::uint32_t constant = module.allocate_id();
    const std::uint32_t main_function = module.allocate_id();

    EntryLowerer(
        module, manifest, find_entry(manifest, "math_probe"), {16u}, math).lower();
    EntryLowerer(
        module, manifest, find_entry(manifest, "barrier_probe"), {64u}, barrier).lower();
    EntryLowerer(
        module, manifest, find_entry(manifest, "local_probe"), {448u}, local).lower();
    EntryLowerer(
        module, manifest, find_entry(manifest, "texture_probe"), {256u}, texture).lower();
    EntryLowerer(
        module,
        manifest,
        find_entry(manifest, "texture_float_probe"),
        {384u},
        texture_float).lower();
    EntryLowerer(
        module,
        manifest,
        find_entry(manifest, "constant_probe"),
        {512u},
        constant).lower();
    module.define_main(
        main_function, math, barrier, local, texture, texture_float, constant);
    return module.finish();
}

std::vector<std::uint32_t> emit_kernel_entry(
    const Json &manifest,
    const std::string &entry_name,
    std::uint32_t local_size,
    std::uint32_t dynamic_shared_bytes)
{
    if (manifest.at("schema_version").get<std::uint32_t>() != 5u) {
        throw std::runtime_error("kernel emitter requires typed manifest schema 5");
    }
    if (manifest.at("address_size").get<std::uint32_t>() != 32u) {
        throw std::runtime_error("kernel emitter requires 32-bit PTX addresses");
    }
    if (local_size == 0u) {
        throw std::runtime_error("kernel emitter local size must be nonzero");
    }
    const Json &entry = find_entry(manifest, entry_name);
    SpirvModule module(
        required_shared_bytes(manifest, entry, dynamic_shared_bytes),
        module_constant_bytes(manifest),
        entry_uses_textures(entry));
    const std::uint32_t function = module.allocate_id();
    KernelEntryLowerer(
        module,
        manifest,
        entry,
        function,
        local_size,
        dynamic_shared_bytes).lower();
    return module.finish();
}

Json read_json(const char *path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open input manifest");
    Json result;
    input >> result;
    return result;
}

void write_spirv(const char *path, const std::vector<std::uint32_t> &words)
{
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("failed to open SPIR-V output");
    output.write(reinterpret_cast<const char *>(words.data()),
                 static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
    if (!output) throw std::runtime_error("failed to write SPIR-V output");
}

} // namespace

int main(int argc, char **argv)
{
    const bool kernel_mode = argc == 7 && std::string(argv[1]) == "--entry";
    if (argc != 3 && !kernel_mode) {
        std::cerr << "usage: ptx-spirv-emitter <typed-manifest.json> <output.spv>\n"
                  << "       ptx-spirv-emitter --entry <name> <local-size> "
                     "<dynamic-shared-bytes> <typed-manifest.json> <output.spv>\n";
        return 2;
    }
    try {
        const char *input_path = kernel_mode ? argv[5] : argv[1];
        const char *output_path = kernel_mode ? argv[6] : argv[2];
        const std::vector<std::uint32_t> words = kernel_mode
            ? emit_kernel_entry(
                  read_json(input_path),
                  argv[2],
                  static_cast<std::uint32_t>(std::stoul(argv[3])),
                  static_cast<std::uint32_t>(std::stoul(argv[4])))
            : emit_oracle(read_json(input_path));
        write_spirv(output_path, words);
        std::cout << "Emitted " << words.size() << " SPIR-V words from "
                  << input_path << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ptx-spirv-emitter: " << error.what() << '\n';
        return 1;
    }
}
