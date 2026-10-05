// Minimal SPIR-V module builder: sections, deduplicated types/constants and
// a linear function body. Only what the PTX code generator needs.
#pragma once

#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace pxvk::spirv {

class Builder {
public:
    std::uint32_t id() { return next_id_++; }

    // ---- module-level sections --------------------------------------------
    void capability(SpvCapability cap)
    {
        for (std::uint32_t c : capabilities_list_)
            if (c == static_cast<std::uint32_t>(cap)) return;
        capabilities_list_.push_back(cap);
        emit(capabilities_, SpvOpCapability, {static_cast<std::uint32_t>(cap)});
    }

    void extension(const char *name)
    {
        for (const auto &e : extension_names_)
            if (e == name) return;
        extension_names_.push_back(name);
        std::vector<std::uint32_t> words;
        append_string(words, name);
        emit_words(extensions_, SpvOpExtension, words);
    }

    std::uint32_t glsl()
    {
        if (glsl_ == 0) {
            glsl_ = id();
            std::vector<std::uint32_t> words{glsl_};
            append_string(words, "GLSL.std.450");
            emit_words(imports_, SpvOpExtInstImport, words);
        }
        return glsl_;
    }

    void decorate(std::uint32_t target, SpvDecoration decoration,
                  std::initializer_list<std::uint32_t> literals = {})
    {
        std::vector<std::uint32_t> words{target, static_cast<std::uint32_t>(decoration)};
        words.insert(words.end(), literals.begin(), literals.end());
        emit_words(annotations_, SpvOpDecorate, words);
    }

    void member_decorate(std::uint32_t type, std::uint32_t member, SpvDecoration decoration,
                         std::initializer_list<std::uint32_t> literals = {})
    {
        std::vector<std::uint32_t> words{type, member, static_cast<std::uint32_t>(decoration)};
        words.insert(words.end(), literals.begin(), literals.end());
        emit_words(annotations_, SpvOpMemberDecorate, words);
    }

    void name(std::uint32_t target, const std::string &text)
    {
        std::vector<std::uint32_t> words{target};
        append_string(words, text);
        emit_words(debug_, SpvOpName, words);
    }

    // ---- types ---------------------------------------------------------------
    std::uint32_t type(SpvOp op, std::initializer_list<std::uint32_t> operands)
    {
        std::vector<std::uint32_t> key{static_cast<std::uint32_t>(op)};
        key.insert(key.end(), operands.begin(), operands.end());
        auto found = types_.find(key);
        if (found != types_.end()) return found->second;
        const std::uint32_t result = id();
        std::vector<std::uint32_t> words{result};
        words.insert(words.end(), operands.begin(), operands.end());
        emit_words(globals_, op, words);
        types_.emplace(std::move(key), result);
        return result;
    }
    std::uint32_t t_void() { return type(SpvOpTypeVoid, {}); }
    std::uint32_t t_bool() { return type(SpvOpTypeBool, {}); }
    std::uint32_t t_u32() { return type(SpvOpTypeInt, {32, 0}); }
    std::uint32_t t_u64()
    {
        capability(SpvCapabilityInt64);
        return type(SpvOpTypeInt, {64, 0});
    }
    std::uint32_t t_f32() { return type(SpvOpTypeFloat, {32}); }
    std::uint32_t t_vec(std::uint32_t component, std::uint32_t count)
    {
        return type(SpvOpTypeVector, {component, count});
    }
    std::uint32_t t_ptr(SpvStorageClass storage, std::uint32_t pointee)
    {
        return type(SpvOpTypePointer, {static_cast<std::uint32_t>(storage), pointee});
    }
    std::uint32_t t_fn_void() { return type(SpvOpTypeFunction, {t_void()}); }

    // Types that must stay distinct (decorated blocks, arrays) bypass the cache.
    std::uint32_t unique_type(SpvOp op, std::initializer_list<std::uint32_t> operands)
    {
        const std::uint32_t result = id();
        std::vector<std::uint32_t> words{result};
        words.insert(words.end(), operands.begin(), operands.end());
        emit_words(globals_, op, words);
        return result;
    }

    // ---- constants -----------------------------------------------------------
    std::uint32_t constant(std::uint32_t type_id, std::uint32_t value)
    {
        const auto key = std::make_pair(type_id, static_cast<std::uint64_t>(value));
        auto found = constants_.find(key);
        if (found != constants_.end()) return found->second;
        const std::uint32_t result = id();
        if (type_id == t_bool()) {
            emit(globals_, value ? SpvOpConstantTrue : SpvOpConstantFalse, {type_id, result});
        } else {
            emit(globals_, SpvOpConstant, {type_id, result, value});
        }
        constants_.emplace(key, result);
        return result;
    }
    std::uint32_t constant64(std::uint64_t value)
    {
        const std::uint32_t type_id = t_u64();
        const auto key = std::make_pair(type_id, value);
        auto found = constants_.find(key);
        if (found != constants_.end()) return found->second;
        const std::uint32_t result = id();
        emit(globals_, SpvOpConstant,
             {type_id, result, static_cast<std::uint32_t>(value),
              static_cast<std::uint32_t>(value >> 32)});
        constants_.emplace(key, result);
        return result;
    }
    std::uint32_t u32(std::uint32_t value) { return constant(t_u32(), value); }
    std::uint32_t f32_bits(std::uint32_t bits) { return constant(t_f32(), bits); }
    std::uint32_t boolean(bool value) { return constant(t_bool(), value ? 1u : 0u); }

    std::uint32_t spec_u32(std::uint32_t spec_id, std::uint32_t default_value)
    {
        const std::uint32_t result = id();
        emit(globals_, SpvOpSpecConstant, {t_u32(), result, default_value});
        decorate(result, SpvDecorationSpecId, {spec_id});
        return result;
    }

    std::uint32_t global_variable(std::uint32_t pointer_type, SpvStorageClass storage)
    {
        const std::uint32_t result = id();
        emit(globals_, SpvOpVariable, {pointer_type, result, static_cast<std::uint32_t>(storage)});
        interface_.push_back(result);
        return result;
    }

    void raw_global(SpvOp op, std::initializer_list<std::uint32_t> words)
    {
        emit(globals_, op, words);
    }

    // ---- function body -------------------------------------------------------
    std::uint32_t local_variable(std::uint32_t pointer_type, std::uint32_t initializer = 0)
    {
        const std::uint32_t result = id();
        if (initializer != 0)
            emit(locals_, SpvOpVariable,
                 {pointer_type, result, SpvStorageClassFunction, initializer});
        else
            emit(locals_, SpvOpVariable, {pointer_type, result, SpvStorageClassFunction});
        return result;
    }

    void op(SpvOp opcode, std::initializer_list<std::uint32_t> operands)
    {
        emit(body_, opcode, operands);
    }
    void op(SpvOp opcode, const std::vector<std::uint32_t> &operands)
    {
        emit_words(body_, opcode, operands);
    }
    std::uint32_t value(SpvOp opcode, std::uint32_t result_type,
                        std::initializer_list<std::uint32_t> operands)
    {
        const std::uint32_t result = id();
        std::vector<std::uint32_t> words{result_type, result};
        words.insert(words.end(), operands.begin(), operands.end());
        emit_words(body_, opcode, words);
        return result;
    }
    std::uint32_t ext(std::uint32_t result_type, GLSLstd450 instruction,
                      std::initializer_list<std::uint32_t> operands)
    {
        const std::uint32_t result = id();
        std::vector<std::uint32_t> words{result_type, result, glsl(),
                                         static_cast<std::uint32_t>(instruction)};
        words.insert(words.end(), operands.begin(), operands.end());
        emit_words(body_, SpvOpExtInst, words);
        return result;
    }

    void label(std::uint32_t id_value)
    {
        emit(body_, SpvOpLabel, {id_value});
        open_ = true;
    }
    // Terminators close the current block.
    void branch(std::uint32_t target)
    {
        emit(body_, SpvOpBranch, {target});
        open_ = false;
    }
    void branch_conditional(std::uint32_t condition, std::uint32_t t, std::uint32_t f)
    {
        emit(body_, SpvOpBranchConditional, {condition, t, f});
        open_ = false;
    }
    void ret()
    {
        emit(body_, SpvOpReturn, {});
        open_ = false;
    }
    void unreachable()
    {
        emit(body_, SpvOpUnreachable, {});
        open_ = false;
    }
    bool open() const { return open_; }

    // ---- final assembly ------------------------------------------------------
    void execution_mode(std::uint32_t entry, SpvExecutionMode mode,
                        std::initializer_list<std::uint32_t> literals = {})
    {
        std::vector<std::uint32_t> words{entry, static_cast<std::uint32_t>(mode)};
        words.insert(words.end(), literals.begin(), literals.end());
        emit_words(execution_modes_, SpvOpExecutionMode, words);
    }

    // Removes unreachable blocks from the function body. Unreachable blocks
    // that are still named as the merge or continue target of a reachable
    // structured header are kept in the canonical form the SPIR-V rules
    // require: OpUnreachable for merges, a branch to the header for
    // continue targets.
    void prune_unreachable()
    {
        struct BlockInfo {
            std::size_t begin, end; // word range including the OpLabel
            std::uint32_t label;
            std::vector<std::uint32_t> successors;
            std::uint32_t merge = 0, cont = 0;
        };
        std::vector<BlockInfo> blocks;
        std::map<std::uint32_t, std::size_t> by_label;
        for (std::size_t i = 0; i < body_.size();) {
            const std::uint32_t count = body_[i] >> 16;
            const SpvOp opcode = static_cast<SpvOp>(body_[i] & 0xffffu);
            if (opcode == SpvOpLabel) {
                if (!blocks.empty()) blocks.back().end = i;
                by_label[body_[i + 1]] = blocks.size();
                blocks.push_back({i, 0, body_[i + 1], {}, 0, 0});
            } else if (opcode == SpvOpBranch) {
                blocks.back().successors.push_back(body_[i + 1]);
            } else if (opcode == SpvOpBranchConditional) {
                blocks.back().successors.push_back(body_[i + 2]);
                blocks.back().successors.push_back(body_[i + 3]);
            } else if (opcode == SpvOpLoopMerge) {
                blocks.back().merge = body_[i + 1];
                blocks.back().cont = body_[i + 2];
            } else if (opcode == SpvOpSelectionMerge) {
                blocks.back().merge = body_[i + 1];
            }
            i += count;
        }
        if (blocks.empty()) return;
        blocks.back().end = body_.size();
        std::vector<bool> reachable(blocks.size(), false);
        std::vector<std::size_t> work{0};
        reachable[0] = true;
        while (!work.empty()) {
            const std::size_t b = work.back();
            work.pop_back();
            for (std::uint32_t s : blocks[b].successors) {
                const std::size_t t = by_label.at(s);
                if (!reachable[t]) {
                    reachable[t] = true;
                    work.push_back(t);
                }
            }
        }
        std::map<std::uint32_t, std::uint32_t> merge_of, continue_header;
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            if (!reachable[b]) continue;
            if (blocks[b].merge) merge_of[blocks[b].merge] = blocks[b].label;
            if (blocks[b].cont) continue_header[blocks[b].cont] = blocks[b].label;
        }
        std::vector<std::uint32_t> out;
        out.reserve(body_.size());
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            if (reachable[b]) {
                out.insert(out.end(), body_.begin() + blocks[b].begin, body_.begin() + blocks[b].end);
            } else if (continue_header.count(blocks[b].label)) {
                emit(out, SpvOpLabel, {blocks[b].label});
                emit(out, SpvOpBranch, {continue_header[blocks[b].label]});
            } else if (merge_of.count(blocks[b].label)) {
                emit(out, SpvOpLabel, {blocks[b].label});
                emit(out, SpvOpUnreachable, {});
            }
        }
        body_.swap(out);
    }

    std::vector<std::uint32_t> finish(std::uint32_t function, std::uint32_t version)
    {
        prune_unreachable();
        std::vector<std::uint32_t> fn;
        emit(fn, SpvOpFunction, {t_void(), function, SpvFunctionControlMaskNone, t_fn_void()});
        // The first body word is the entry label; variables must follow it.
        const std::size_t entry_label_words = 2;
        fn.insert(fn.end(), body_.begin(), body_.begin() + entry_label_words);
        fn.insert(fn.end(), locals_.begin(), locals_.end());
        fn.insert(fn.end(), body_.begin() + entry_label_words, body_.end());
        emit(fn, SpvOpFunctionEnd, {});

        std::vector<std::uint32_t> memory_model;
        emit(memory_model, SpvOpMemoryModel, {SpvAddressingModelLogical, SpvMemoryModelGLSL450});
        std::vector<std::uint32_t> entry{SpvExecutionModelGLCompute, function};
        append_string(entry, "main");
        if (version >= 0x10400) {
            entry.insert(entry.end(), interface_.begin(), interface_.end());
        } else {
            for (std::uint32_t v : interface_)
                if (input_interface_.count(v)) entry.push_back(v);
        }
        std::vector<std::uint32_t> entry_point;
        emit_words(entry_point, SpvOpEntryPoint, entry);

        std::vector<std::uint32_t> out{SpvMagicNumber, version, 0x00500000u /* generator */,
                                       next_id_, 0};
        for (const auto *section : {&capabilities_, &extensions_, &imports_, &memory_model,
                                    &entry_point, &execution_modes_, &debug_, &annotations_,
                                    &globals_, &fn})
            out.insert(out.end(), section->begin(), section->end());
        return out;
    }

    void mark_input(std::uint32_t variable) { input_interface_[variable] = true; }

private:
    static void append_string(std::vector<std::uint32_t> &words, const std::string &text)
    {
        const std::size_t count = text.size() / 4 + 1;
        const std::size_t start = words.size();
        words.resize(start + count, 0);
        std::memcpy(words.data() + start, text.data(), text.size());
    }
    static void emit(std::vector<std::uint32_t> &out, SpvOp opcode,
                     std::initializer_list<std::uint32_t> operands)
    {
        out.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16) |
                      static_cast<std::uint32_t>(opcode));
        out.insert(out.end(), operands.begin(), operands.end());
    }
    static void emit_words(std::vector<std::uint32_t> &out, SpvOp opcode,
                           const std::vector<std::uint32_t> &operands)
    {
        out.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16) |
                      static_cast<std::uint32_t>(opcode));
        out.insert(out.end(), operands.begin(), operands.end());
    }

    std::uint32_t next_id_ = 1;
    std::uint32_t glsl_ = 0;
    bool open_ = false;
    std::vector<std::uint32_t> capabilities_list_;
    std::vector<std::string> extension_names_;
    std::vector<std::uint32_t> capabilities_, extensions_, imports_, execution_modes_, debug_,
        annotations_, globals_, locals_, body_;
    std::vector<std::uint32_t> interface_;
    std::map<std::uint32_t, bool> input_interface_;
    std::map<std::vector<std::uint32_t>, std::uint32_t> types_;
    std::map<std::pair<std::uint32_t, std::uint64_t>, std::uint32_t> constants_;
};

} // namespace pxvk::spirv
