// PTX abstract syntax tree and parser for the classic PhysX PTX dialect
// (PTX ISA 1.x-4.x, 32-bit addressing). The parser is deliberately
// permissive about which instructions appear: semantic checking happens when
// an entry is compiled, so unsupported instructions in unlaunched kernels do
// not prevent a module from loading. Unknown syntax (calls, body-level
// .param, dual-destination setp) still fails the whole module.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pxvk::ptx {

struct Error : std::runtime_error {
    Error(const std::string &message, int line)
        : std::runtime_error("PTX line " + std::to_string(line) + ": " + message), line(line) {}
    int line;
};

enum class Space { none, reg, global, shared, local, constant, param, tex };

// Scalar PTX type such as .u32 or .f32. `kind` is one of b/u/s/f/pred/texref.
struct Type {
    char kind = 'b';
    int bits = 32;
    bool operator==(const Type &o) const { return kind == o.kind && bits == o.bits; }
    bool operator!=(const Type &o) const { return !(*this == o); }
    bool is_float() const { return kind == 'f'; }
    bool is_signed() const { return kind == 's'; }
    bool is_pred() const { return kind == 'p'; }
};

bool parse_type_name(std::string_view name, Type &out);

struct Operand {
    enum class Kind { reg, imm, fimm, symbol, address, vector, special };
    Kind kind = Kind::imm;
    std::string name;        // register, symbol, special register ("%tid.x"), or address base
    bool negated = false;    // "!%p" predicate operand
    std::int64_t imm = 0;    // integer immediate or address offset
    std::uint64_t fbits = 0; // float immediate bits
    int fwidth = 32;         // 32 for 0f, 64 for 0d
    enum class Base { none, reg, symbol } base = Base::none; // address base kind
    std::vector<Operand> elems; // vector operand, or tex [ref, {coords}]
};

struct Instruction {
    std::string op;                // base opcode, e.g. "ld"
    std::vector<std::string> mods; // modifiers without dots, e.g. {"global","v4","f32"}
    std::string guard;             // guard predicate register, empty if none
    bool guard_negated = false;
    std::vector<Operand> operands;
    int line = 0;

    bool has(std::string_view mod) const
    {
        for (const auto &m : mods)
            if (m == mod) return true;
        return false;
    }
    std::string text() const; // opcode with modifiers, for diagnostics
};

struct Statement {
    bool is_label = false;
    std::string label;
    Instruction inst;
};

struct Variable {
    Space space = Space::global;
    std::string name;
    Type type;
    std::uint32_t align = 0;
    // Array element counts, outermost first. Empty for scalars; a single 0
    // denotes an unsized extern array ("[]").
    std::vector<std::uint64_t> dims;
    bool external = false;
    std::vector<std::uint8_t> init; // little-endian initializer bytes, if any
    bool has_init = false;
    int line = 0;

    std::uint64_t element_bytes() const { return type.kind == 'p' ? 1 : type.bits / 8; }
    std::uint64_t size_bytes() const;
};

struct RegDecl {
    Type type;
    std::string name; // base name; with count>0 registers are name0..name(count-1)
    std::uint32_t count = 0;
};

struct Param {
    Type type;
    std::string name;
    std::uint32_t align = 0;
    std::uint64_t array = 0; // element count for ".param .b8 x[N]", 0 if scalar
    std::uint32_t size_bytes() const
    {
        return static_cast<std::uint32_t>((type.bits / 8) * (array ? array : 1));
    }
};

struct Function {
    bool is_entry = false;
    std::string name;
    std::vector<Param> returns;
    std::vector<Param> params;
    std::vector<RegDecl> regs;
    std::vector<Variable> vars; // function-scope .local / .shared declarations
    std::vector<Statement> body;
    std::uint32_t maxntid[3] = {0, 0, 0};
    std::uint32_t reqntid[3] = {0, 0, 0};
    bool has_body = false;
    int line = 0;
};

struct Module {
    int version_major = 0, version_minor = 0;
    std::string target;
    int address_size = 32;
    std::vector<Variable> vars; // module-scope variables incl. texrefs
    std::vector<Function> functions;

    const Function *find(std::string_view name) const;
};

Module parse(std::string_view text);

} // namespace pxvk::ptx
