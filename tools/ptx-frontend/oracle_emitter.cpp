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

class SpirvModule {
public:
    SpirvModule()
    {
        emit(capabilities_, SpvOpCapability, {SpvCapabilityShader});
        emit(capabilities_, SpvOpCapability, {SpvCapabilityRoundingModeRTE});
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

        const std::uint32_t shared_length = constant(ValueType::u32, 64u).id;
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
                     std::uint32_t texture_function)
    {
        emit_entry_point(main_function, {local_id_, buffer_, shared_});
        emit(execution_modes_, SpvOpExecutionMode,
             {main_function, SpvExecutionModeLocalSize, 64u, 1u, 1u});
        emit(execution_modes_, SpvOpExecutionMode,
             {main_function, SpvExecutionModeRoundingModeRTE, 32u});

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
        function_instruction(SpvOpBranch, {math_merge});
        function_instruction(SpvOpLabel, {math_merge});

        emit_void_call(barrier_function);

        const Value is_texture_thread = instruction(
            SpvOpULessThan, ValueType::boolean, {thread.id, twelve.id});
        const std::uint32_t texture_label = allocate_id();
        const std::uint32_t texture_merge = allocate_id();
        function_instruction(SpvOpSelectionMerge, {texture_merge, SpvSelectionControlMaskNone});
        function_instruction(SpvOpBranchConditional,
                             {is_texture_thread.id, texture_label, texture_merge});
        function_instruction(SpvOpLabel, {texture_label});
        emit_void_call(texture_function);
        function_instruction(SpvOpBranch, {texture_merge});
        function_instruction(SpvOpLabel, {texture_merge});
        end_function();
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

    void emit_entry_point(std::uint32_t function,
                          std::initializer_list<std::uint32_t> interfaces)
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
    std::uint32_t type_shared_array_ = 0u;
    std::uint32_t type_ptr_workgroup_array_ = 0u;
    std::uint32_t type_ptr_workgroup_u32_ = 0u;
    std::uint32_t buffer_ = 0u;
    std::uint32_t local_id_ = 0u;
    std::uint32_t shared_ = 0u;
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

class EntryLowerer {
public:
    EntryLowerer(SpirvModule &module,
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
    }

    void lower()
    {
        module_.begin_function(function_);
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
            return module_.coerce(register_value(register_operand_name(operand)), desired);
        }
        if (kind == "symbol" && operand.at("name").get<std::string>() == "values") {
            return module_.constant(desired, 0u);
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
            texture.at("values").at(0).at("kind").get<std::string>() != "symbol" ||
            texture.at("values").at(0).at("name").get<std::string>() != "probe_tex") {
            unsupported(instruction, "unsupported texture shape");
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
        const Value base = module_.instruction(
            SpvOpIAdd,
            ValueType::u32,
            {row.id, module_.constant(ValueType::u32, 160u).id});
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
        if (base == "bar" && variant == "bar.sync") {
            const Value workgroup = module_.constant(ValueType::u32, 2u);
            const Value semantics = module_.constant(ValueType::u32, 0x108u);
            module_.function_instruction(
                SpvOpControlBarrier, {workgroup.id, workgroup.id, semantics.id});
            return;
        }
        if (base == "tex" && variant == "tex.1d.v4.u32.s32") {
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
    if (manifest.at("schema_version").get<std::uint32_t>() != 4u) {
        throw std::runtime_error("oracle emitter requires typed manifest schema 4");
    }
    if (manifest.at("address_size").get<std::uint32_t>() != 32u) {
        throw std::runtime_error("oracle emitter requires 32-bit PTX addresses");
    }
    SpirvModule module;
    const std::uint32_t math = module.allocate_id();
    const std::uint32_t barrier = module.allocate_id();
    const std::uint32_t texture = module.allocate_id();
    const std::uint32_t main_function = module.allocate_id();

    EntryLowerer(module, find_entry(manifest, "math_probe"), {16u}, math).lower();
    EntryLowerer(module, find_entry(manifest, "barrier_probe"), {64u}, barrier).lower();
    EntryLowerer(module, find_entry(manifest, "texture_probe"), {256u}, texture).lower();
    module.define_main(main_function, math, barrier, texture);
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
    if (argc != 3) {
        std::cerr << "usage: ptx-oracle-emitter "
                     "<typed-manifest.json> <output.spv>\n";
        return 2;
    }
    try {
        const std::vector<std::uint32_t> words = emit_oracle(read_json(argv[1]));
        write_spirv(argv[2], words);
        std::cout << "Emitted " << words.size() << " SPIR-V words from "
                  << argv[1] << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ptx-oracle-emitter: " << error.what() << '\n';
        return 1;
    }
}
