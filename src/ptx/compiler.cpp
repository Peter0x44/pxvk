#include "compiler.h"

#include "cfg.h"
#include "spirv_builder.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace pxvk::ptx {
namespace {

using spirv::Builder;

// Address-space provenance of a register value, used to resolve generic
// (state-space-less) loads and stores.
enum Provenance : unsigned {
    prov_none = 0,
    prov_global = 1,
    prov_shared = 2,
    prov_local = 4,
};

bool is_type_name(const std::string &s)
{
    Type t;
    return parse_type_name(s, t);
}

// Last type modifier of an instruction, e.g. "f32" of add.ftz.f32.
Type last_type(const Instruction &in)
{
    for (auto it = in.mods.rbegin(); it != in.mods.rend(); ++it) {
        Type t;
        if (parse_type_name(*it, t)) return t;
    }
    throw Error("instruction " + in.text() + " has no type", in.line);
}

// For cvt-like instructions: {destination, source} types.
std::pair<Type, Type> two_types(const Instruction &in)
{
    std::vector<Type> types;
    for (const auto &m : in.mods) {
        Type t;
        if (parse_type_name(m, t)) types.push_back(t);
    }
    if (types.size() != 2) throw Error(in.text() + " expects two types", in.line);
    return {types[0], types[1]};
}

int vector_width(const Instruction &in)
{
    if (in.has("v2")) return 2;
    if (in.has("v4")) return 4;
    return 1;
}

Space memory_space(const Instruction &in)
{
    if (in.has("global")) return Space::global;
    if (in.has("shared")) return Space::shared;
    if (in.has("local")) return Space::local;
    if (in.has("const")) return Space::constant;
    if (in.has("param")) return Space::param;
    return Space::none;
}

class KernelCompiler {
public:
    KernelCompiler(const Module &module, const Function &function, const CompileOptions &options)
        : m_(module), f_(function), options_(options)
    {
    }

    CompiledKernel run()
    {
        info_.name = f_.name;
        std::copy(std::begin(f_.maxntid), std::end(f_.maxntid), info_.maxntid);
        std::copy(std::begin(f_.reqntid), std::end(f_.reqntid), info_.reqntid);
        layout_parameters();
        collect_symbols();
        layout_shared_and_local();
        decide_float_mode();
        if (!native_ftz_) analyze_flushed();
        analyze_provenance();
        if (!options_.no_contraction) plan_contractions();

        b_.capability(SpvCapabilityShader);
        main_ = b_.id();
        declare_registers();

        const Cfg cfg = build_cfg(f_);
        const Structured tree = structurize(cfg);
        is_loop_ = tree.construct_is_loop;
        constructs_.assign(is_loop_.size(), {});
        for (auto &c : constructs_) {
            c.header = b_.id();
            c.body = b_.id();
            c.cont = b_.id();
            c.merge = b_.id();
        }
        cfg_ = &cfg;
        find_transparent_blocks(tree);
        if (options_.emulate_warps && contains_warp_sync(tree.root)) {
            // Emulated warps add workgroup barriers that every invocation
            // must reach: reject kernels where that is not guaranteed.
            check_warp_uniformity(tree.root, true);
            info_.warp_emulated = true;
        }

        b_.label(b_.id());
        // Module-global addresses come from the launch block, so pipelines do
        // not depend on where a module instance's globals were allocated.
        for (const auto &entry : symbol_spec_)
            symbol_constants_[entry.first] = launch_word(info_.symbol_table_offset + 4u * entry.second);
        if (options_.shared_in_storage && (info_.static_shared_bytes != 0 || info_.uses_dynamic_shared)) {
            // Linear workgroup index times the per-group stride (spec 3).
            const std::uint32_t u32 = b_.t_u32();
            const std::uint32_t v3 = b_.t_vec(u32, 3);
            const std::uint32_t id = b_.value(SpvOpLoad, v3, {builtin_vector(SpvBuiltInWorkgroupId)});
            const std::uint32_t count = b_.value(SpvOpLoad, v3, {builtin_vector(SpvBuiltInNumWorkgroups)});
            auto part = [&](std::uint32_t v, std::uint32_t c) { return b_.value(SpvOpCompositeExtract, u32, {v, c}); };
            std::uint32_t linear = b_.value(SpvOpIMul, u32, {part(id, 2), part(count, 1)});
            linear = b_.value(SpvOpIAdd, u32, {linear, part(id, 1)});
            linear = b_.value(SpvOpIMul, u32, {linear, part(count, 0)});
            linear = b_.value(SpvOpIAdd, u32, {linear, part(id, 0)});
            const std::uint32_t stride = b_.spec_u32(kSpecSharedWords, std::max<std::uint32_t>(1, info_.static_shared_bytes / 4));
            scratch_base_ = b_.value(SpvOpIMul, u32, {linear, stride});
        }
        emit(tree.root);
        if (b_.open()) b_.ret();

        finish_interface();
        CompiledKernel out;
        out.info = info_;
        out.spirv = b_.finish(main_, 0x10300);
        return out;
    }

private:
    // ---- layout ----------------------------------------------------------
    struct ParamSlot {
        std::uint32_t offset;
        Param param;
    };

    void layout_parameters()
    {
        std::uint32_t offset = 0;
        for (const Param &p : f_.params) {
            const std::uint32_t align = std::max<std::uint32_t>(p.align ? p.align : p.type.bits / 8, 1);
            offset = (offset + align - 1) / align * align;
            params_[p.name] = {offset, p};
            info_.params.push_back({offset, p.size_bytes()});
            offset += p.size_bytes();
        }
        info_.param_bytes = offset;
    }

    // Module-scope symbols referenced by the function body.
    void collect_symbols()
    {
        std::set<std::string> referenced;
        for (const Statement &s : f_.body) {
            if (s.is_label) continue;
            for (const Operand &o : s.inst.operands) collect_operand_symbols(o, referenced);
        }
        for (const Variable &v : m_.vars) {
            if (!referenced.count(v.name)) continue;
            if (v.space == Space::global || v.space == Space::constant) {
                symbol_spec_[v.name] = static_cast<std::uint32_t>(info_.symbols.size());
                info_.symbols.push_back(v.name);
            } else if (v.space == Space::tex) {
                texture_index_[v.name] = static_cast<std::uint32_t>(info_.textures.size());
                info_.textures.push_back(v.name);
            }
        }
        info_.texture_table_offset = (info_.param_bytes + 15u) / 16u * 16u;
        info_.symbol_table_offset = info_.texture_table_offset +
                                    static_cast<std::uint32_t>(info_.textures.size()) * kTextureEntryBytes;
        info_.launch_block_bytes = info_.symbol_table_offset +
                                   static_cast<std::uint32_t>(info_.symbols.size()) * 4u;
        if (info_.launch_block_bytes == 0) info_.launch_block_bytes = 16;
    }

    static void collect_operand_symbols(const Operand &o, std::set<std::string> &out)
    {
        if (o.kind == Operand::Kind::symbol) out.insert(o.name);
        if (o.kind == Operand::Kind::address && o.base == Operand::Base::symbol) out.insert(o.name);
        for (const Operand &e : o.elems) collect_operand_symbols(e, out);
    }

    void layout_shared_and_local()
    {
        std::set<std::string> referenced;
        for (const Statement &s : f_.body)
            if (!s.is_label)
                for (const Operand &o : s.inst.operands) collect_operand_symbols(o, referenced);
        std::uint32_t shared = 0;
        std::uint32_t local = 0;
        std::vector<const Variable *> externs;
        auto place = [&](const Variable &v) {
            if (v.space == Space::shared) {
                if (v.external || (v.dims.size() == 1 && v.dims[0] == 0)) {
                    externs.push_back(&v);
                    return;
                }
                const std::uint32_t align = std::max<std::uint32_t>(v.align, 4);
                shared = (shared + align - 1) / align * align;
                shared_offset_[v.name] = shared;
                shared += static_cast<std::uint32_t>(v.size_bytes());
            } else if (v.space == Space::local) {
                const std::uint32_t align = std::max<std::uint32_t>(v.align, 4);
                local = (local + align - 1) / align * align;
                local_offset_[v.name] = local;
                local += static_cast<std::uint32_t>(v.size_bytes());
            }
        };
        for (const Variable &v : m_.vars)
            if (referenced.count(v.name)) place(v);
        for (const Variable &v : f_.vars) place(v);
        std::uint32_t extern_align = 4;
        for (const Variable *v : externs) extern_align = std::max<std::uint32_t>(extern_align, v->align);
        shared = (shared + 3u) & ~3u;
        if (!externs.empty()) {
            const std::uint32_t base = (shared + extern_align - 1) / extern_align * extern_align;
            for (const Variable *v : externs)
                if (referenced.count(v->name) || v->space == Space::shared) shared_offset_[v->name] = base;
            info_.uses_dynamic_shared = true;
            shared = base;
        }
        info_.static_shared_bytes = shared;
        info_.local_bytes = (local + 3u) & ~3u;
    }

    void decide_float_mode()
    {
        bool all_ftz = true;
        bool any_float = false;
        static const std::set<std::string> arithmetic = {
            "add", "sub", "mul", "mad", "fma", "div", "rcp", "sqrt", "rsqrt", "min", "max",
            "abs", "neg", "setp", "sin", "cos", "lg2", "ex2", "cvt", "set", "testp"};
        for (const Statement &s : f_.body) {
            if (s.is_label || !arithmetic.count(s.inst.op)) continue;
            bool is_float = false;
            for (const auto &m : s.inst.mods)
                if (m == "f32") is_float = true;
            if (!is_float) continue;
            any_float = true;
            if (!s.inst.has("ftz")) all_ftz = false;
        }
        native_ftz_ = options_.native_ftz && any_float && all_ftz;
    }

    // Emulated .ftz flushes every input and output. Most inputs are outputs
    // of other .ftz instructions, so find registers whose every definition
    // already holds a non-denormal value and skip their input flushes. This
    // is a greatest fixpoint so values carried around loops by mov qualify.
    static bool denormal_bits(std::uint32_t bits) { return (bits & 0x7f800000u) == 0 && (bits & 0x7fffffffu) != 0; }

    bool flush_safe(const Operand &o) const
    {
        if (o.kind == Operand::Kind::reg) return flushed_.count(o.name) != 0;
        if (o.kind == Operand::Kind::imm) return !denormal_bits(static_cast<std::uint32_t>(o.imm));
        if (o.kind == Operand::Kind::fimm) return o.fwidth == 32 && !denormal_bits(static_cast<std::uint32_t>(o.fbits));
        return false;
    }

    void analyze_flushed()
    {
        static const std::set<std::string> flushing = {"add", "sub", "mul", "mad", "fma", "div", "rcp", "sqrt",
                                                       "rsqrt", "sin", "cos", "lg2", "ex2", "abs", "neg", "min",
                                                       "max"};
        for (const Statement &s : f_.body)
            if (!s.is_label && !s.inst.operands.empty() && s.inst.operands[0].kind == Operand::Kind::reg)
                flushed_.insert(s.inst.operands[0].name);
        bool changed = true;
        while (changed) {
            changed = false;
            for (const Statement &s : f_.body) {
                if (s.is_label) continue;
                const Instruction &in = s.inst;
                if (in.operands.empty() || in.op == "st" || in.op == "bra" || in.op == "red" || in.op == "bar" ||
                    in.op == "ret" || in.op == "membar")
                    continue;
                const Operand &dst = in.operands[0];
                bool safe = false;
                if (dst.kind == Operand::Kind::reg) {
                    if (flushing.count(in.op)) {
                        safe = in.has("f32") && in.has("ftz");
                    } else if (in.op == "cvt" && in.mods.size() >= 2) {
                        // float destination from an integer source, or .ftz
                        const std::string &d = in.mods[in.mods.size() - 2], &src = in.mods.back();
                        safe = d == "f32" && (src[0] == 's' || src[0] == 'u' || in.has("ftz"));
                    } else if (in.op == "mov" && in.operands.size() == 2) {
                        safe = flush_safe(in.operands[1]);
                    } else if (in.op == "selp" && in.operands.size() == 4) {
                        safe = flush_safe(in.operands[1]) && flush_safe(in.operands[2]);
                    }
                }
                if (!safe) {
                    if (dst.kind == Operand::Kind::reg) changed |= flushed_.erase(dst.name) != 0;
                    for (const Operand &e : dst.elems)
                        if (e.kind == Operand::Kind::reg) changed |= flushed_.erase(e.name) != 0;
                }
            }
        }
    }

    // ---- contraction ---------------------------------------------------------
    struct Fusion {
        const Instruction *mul;
        int index;    // add/sub source operand holding the product
        bool negated; // product reached the add through a neg
    };

    // ptxas fuses an f32 mul without rounding modifier into fma when every
    // use of the product is an add/sub without rounding modifier and the
    // product has a single definition (verified against native execution
    // with tools/opcheck). Vulkan drivers do not contract on their own, so
    // matching native results requires doing it here.
    static bool contractible(const Instruction &in)
    {
        if (!in.has("f32") || in.has("sat")) return false;
        return !(in.has("rn") || in.has("rz") || in.has("rm") || in.has("rp"));
    }

    // Basic-block index of every instruction as ptxas sees it when it
    // contracts: dead instructions are removed first, so a branch that only
    // skipped dead code disappears and does not split the block.
    std::map<const Instruction *, int> contraction_blocks(
        const std::map<std::string, std::vector<std::pair<const Instruction *, int>>> &uses) const
    {
        auto dead = [&](const Instruction &in) {
            static const std::set<std::string> effects = {"st", "red", "atom", "bar", "membar", "bra",
                                                          "call", "ret", "exit", "trap", "brkpt"};
            if (effects.count(in.op) || in.operands.empty() || in.operands[0].kind != Operand::Kind::reg) return false;
            const auto found = uses.find(in.operands[0].name);
            return found == uses.end() || found->second.empty();
        };
        std::vector<const Statement *> live;
        for (const Statement &s : f_.body)
            if (s.is_label || !dead(s.inst)) live.push_back(&s);
        std::set<const Statement *> joined; // empty branches and their labels
        for (std::size_t i = 0; i + 1 < live.size(); ++i) {
            const Statement &s = *live[i], &next = *live[i + 1];
            if (!s.is_label && s.inst.op == "bra" && !s.inst.operands.empty() && next.is_label &&
                s.inst.operands[0].name == next.label) {
                joined.insert(&s);
                joined.insert(&next);
            }
        }
        std::map<const Instruction *, int> block_of;
        int block = 0;
        for (const Statement &s : f_.body) {
            if (s.is_label) {
                if (!joined.count(&s)) ++block;
                continue;
            }
            block_of[&s.inst] = block;
            if (!joined.count(&s) && (s.inst.op == "bra" || s.inst.op == "ret" || s.inst.op == "exit")) ++block;
        }
        return block_of;
    }

    void plan_contractions()
    {
        std::map<std::string, int> defs;
        std::map<std::string, std::vector<std::pair<const Instruction *, int>>> uses;
        for (const Statement &s : f_.body) {
            if (s.is_label) continue;
            const Instruction &in = s.inst;
            const bool has_destination = in.op != "st" && in.op != "bra" && in.op != "red" &&
                                         in.op != "bar" && in.op != "ret" && in.op != "membar";
            for (std::size_t i = 0; i < in.operands.size(); ++i) {
                const Operand &o = in.operands[i];
                if (i == 0 && has_destination) {
                    if (o.kind == Operand::Kind::reg) ++defs[o.name];
                    for (const Operand &e : o.elems)
                        if (e.kind == Operand::Kind::reg) ++defs[e.name];
                    continue;
                }
                if (o.kind == Operand::Kind::reg) uses[o.name].push_back({&in, static_cast<int>(i)});
                if (o.kind == Operand::Kind::address && o.base == Operand::Base::reg)
                    uses[o.name].push_back({&in, -1});
                for (const Operand &e : o.elems)
                    if (e.kind == Operand::Kind::reg) uses[e.name].push_back({&in, -1});
            }
            if (!in.guard.empty()) uses[in.guard].push_back({&in, -1});
        }
        const std::map<const Instruction *, int> block_of = contraction_blocks(uses);
        // Product registers that can be folded into adds: the mul result
        // itself, or the result of a neg applied to it (ptxas folds that too).
        std::map<std::string, Fusion> candidates;
        auto add_like_uses_only = [&](const std::string &reg, const Instruction &mul) {
            const auto &list = uses[reg];
            if (list.empty() || defs[reg] != 1) return false;
            for (const auto &[user, index] : list) {
                if ((user->op != "add" && user->op != "sub") || !contractible(*user) || index < 1 ||
                    block_of.at(user) != block_of.at(&mul) || user->has("ftz") != mul.has("ftz") ||
                    (user->operands[1].kind == Operand::Kind::reg && user->operands[2].kind == Operand::Kind::reg &&
                     user->operands[1].name == user->operands[2].name))
                    return false;
            }
            return true;
        };
        for (const Statement &s : f_.body) {
            if (s.is_label) continue;
            const Instruction &mul = s.inst;
            if (mul.op != "mul" || !contractible(mul) || !mul.guard.empty() || mul.operands.size() != 3 ||
                mul.operands[0].kind != Operand::Kind::reg)
                continue;
            const std::string &product = mul.operands[0].name;
            if (defs[product] != 1 || mul.operands[1].negated || mul.operands[2].negated) continue;
            const auto &list = uses[product];
            bool ok = !list.empty();
            std::vector<std::string> negations;
            for (const auto &[user, index] : list) {
                if (user->op == "neg" && user->has("f32") && user->guard.empty() &&
                    user->has("ftz") == mul.has("ftz") && user->operands[0].kind == Operand::Kind::reg &&
                    add_like_uses_only(user->operands[0].name, mul)) {
                    negations.push_back(user->operands[0].name);
                    continue;
                }
                if ((user->op != "add" && user->op != "sub") || !contractible(*user) || index < 1 ||
                    user->has("ftz") != mul.has("ftz") ||
                    (user->operands[1].kind == Operand::Kind::reg && user->operands[2].kind == Operand::Kind::reg &&
                     user->operands[1].name == user->operands[2].name))
                    ok = false;
            }
            // ptxas contracts within a basic block only: a product with a
            // use in another block stays a separate multiply for every use.
            for (const auto &[user, index] : list)
                if (block_of.at(user) != block_of.at(&mul)) ok = false;
            if (!ok) continue;
            candidates[product] = {&mul, 0, false};
            for (const std::string &n : negations) candidates[n] = {&mul, 0, true};
        }
        // Each add/sub fuses at most one product; like ptxas, prefer the
        // first source operand.
        for (const Statement &s : f_.body) {
            if (s.is_label) continue;
            const Instruction &in = s.inst;
            if ((in.op != "add" && in.op != "sub") || !contractible(in) || in.operands.size() != 3) continue;
            for (int index = 1; index <= 2; ++index) {
                const Operand &o = in.operands[static_cast<std::size_t>(index)];
                if (o.kind != Operand::Kind::reg) continue;
                const auto found = candidates.find(o.name);
                if (found == candidates.end()) continue;
                Fusion fusion = found->second;
                fusion.index = index;
                fused_[&in] = fusion;
                if (!stash_.count(fusion.mul)) {
                    const std::uint32_t ptr = b_.t_ptr(SpvStorageClassFunction, b_.t_f32());
                    stash_[fusion.mul] = {b_.local_variable(ptr), b_.local_variable(ptr)};
                }
                break;
            }
        }
    }

    // Vulkan only guarantees "multiply then add" precision for Fma unless it
    // is decorated NoContraction, which requires a single rounding.
    std::uint32_t fused_multiply_add(std::uint32_t a, std::uint32_t b, std::uint32_t c)
    {
        const std::uint32_t r = b_.ext(b_.t_f32(), GLSLstd450Fma, {a, b, c});
        b_.decorate(r, SpvDecorationNoContraction);
        return r;
    }

    // fma for an add/sub whose operand `fusion.index` is a fused product.
    std::uint32_t fused_add(const Instruction &in, const Fusion &fusion)
    {
        const std::uint32_t f32 = b_.t_f32();
        const auto &stash = stash_.at(fusion.mul);
        std::uint32_t a = b_.value(SpvOpLoad, f32, {stash.first});
        std::uint32_t b = b_.value(SpvOpLoad, f32, {stash.second});
        std::uint32_t c = fin(in, fusion.index == 1 ? 2 : 1);
        bool negate_product = fusion.negated;
        if (in.op == "sub") {
            if (fusion.index == 1) c = b_.value(SpvOpFNegate, f32, {c}); // a*b - c
            else negate_product = !negate_product;                       // c - a*b
        }
        if (negate_product) a = b_.value(SpvOpFNegate, f32, {a});
        return fused_multiply_add(a, b, c);
    }

    // ---- provenance --------------------------------------------------------
    unsigned symbol_provenance(const std::string &name) const
    {
        if (shared_offset_.count(name)) return prov_shared;
        if (local_offset_.count(name)) return prov_local;
        if (symbol_spec_.count(name)) return prov_global;
        return prov_none;
    }

    void analyze_provenance()
    {
        static const std::set<std::string> arithmetic = {"add", "sub", "mad", "mov", "cvta",
                                                         "selp", "and", "or", "shl", "mul"};
        bool changed = true;
        while (changed) {
            changed = false;
            for (const Statement &s : f_.body) {
                if (s.is_label) continue;
                const Instruction &in = s.inst;
                if (in.operands.empty() || in.operands[0].kind != Operand::Kind::reg) continue;
                if (in.op == "st" || in.op == "bra" || in.op == "setp") continue;
                unsigned p = 0;
                if (in.op == "ld") {
                    p = memory_space(in) == Space::param ? prov_global : prov_none;
                } else if (in.op == "cvta") {
                    if (in.has("shared")) p = prov_shared;
                    else if (in.has("local")) p = prov_local;
                    else p = prov_global;
                } else if (arithmetic.count(in.op)) {
                    for (std::size_t i = 1; i < in.operands.size(); ++i) {
                        const Operand &o = in.operands[i];
                        if (o.kind == Operand::Kind::reg) p |= provenance_[o.name];
                        else if (o.kind == Operand::Kind::symbol) p |= symbol_provenance(o.name);
                    }
                }
                unsigned &slot = provenance_[in.operands[0].name];
                if ((slot | p) != slot) {
                    slot |= p;
                    changed = true;
                }
            }
        }
    }

    Space resolve_generic(const Operand &address, int line)
    {
        unsigned p = prov_none;
        if (address.base == Operand::Base::reg) p = provenance_[address.name];
        else if (address.base == Operand::Base::symbol) p = symbol_provenance(address.name);
        if (p == prov_shared) return Space::shared;
        if (p == prov_local) return Space::local;
        if ((p & (prov_shared | prov_local)) != 0)
            throw Error("generic address with ambiguous state space", line);
        return Space::global;
    }

    // ---- types and registers ----------------------------------------------
    std::uint32_t storage_type(Type t)
    {
        if (t.kind == 'p') return b_.t_bool();
        if (t.kind == 'f') {
            if (t.bits != 32) throw Error("only .f32 floating point is supported", f_.line);
            return b_.t_f32();
        }
        if (t.bits == 64) {
            info_.uses_int64 = true;
            return b_.t_u64();
        }
        return b_.t_u32();
    }

    struct Register {
        std::uint32_t variable;
        std::uint32_t type_id;
        Type type;
    };

    void declare_registers()
    {
        std::set<std::string> used;
        for (const Statement &s : f_.body) {
            if (s.is_label) continue;
            if (!s.inst.guard.empty()) used.insert(s.inst.guard);
            std::function<void(const Operand &)> visit = [&](const Operand &o) {
                if (o.kind == Operand::Kind::reg ||
                    (o.kind == Operand::Kind::address && o.base == Operand::Base::reg))
                    used.insert(o.name);
                for (const Operand &e : o.elems) visit(e);
            };
            for (const Operand &o : s.inst.operands) visit(o);
        }
        for (const RegDecl &decl : f_.regs) {
            auto add = [&](const std::string &name) {
                if (!used.count(name)) return;
                const std::uint32_t type_id = storage_type(decl.type);
                const std::uint32_t var = b_.local_variable(b_.t_ptr(SpvStorageClassFunction, type_id));
                registers_[name] = {var, type_id, decl.type};
            };
            if (decl.count == 0) add(decl.name);
            else
                for (std::uint32_t i = 0; i < decl.count; ++i) add(decl.name + std::to_string(i));
        }
    }

    const Register &reg(const std::string &name, int line)
    {
        const auto found = registers_.find(name);
        if (found == registers_.end()) throw Error("undeclared register " + name, line);
        return found->second;
    }

    // Canonical SPIR-V representation of a PTX type: bool, f32, u64 or u32.
    std::uint32_t canonical(Type t)
    {
        if (t.kind == 'p') return b_.t_bool();
        if (t.kind == 'f') return b_.t_f32();
        if (t.bits == 64) {
            info_.uses_int64 = true;
            return b_.t_u64();
        }
        return b_.t_u32();
    }

    std::uint32_t convert_bits(std::uint32_t value, std::uint32_t from, std::uint32_t to, int line)
    {
        if (from == to) return value;
        const bool from32 = from == b_.t_u32() || from == b_.t_f32();
        const bool to32 = to == b_.t_u32() || to == b_.t_f32();
        if (from32 && to32) return b_.value(SpvOpBitcast, to, {value});
        if (from == b_.t_u64() && to32) {
            const std::uint32_t low = b_.value(SpvOpUConvert, b_.t_u32(), {value});
            return to == b_.t_u32() ? low : b_.value(SpvOpBitcast, to, {low});
        }
        if (from32 && to == b_.t_u64()) {
            const std::uint32_t bits = from == b_.t_u32() ? value : b_.value(SpvOpBitcast, b_.t_u32(), {value});
            return b_.value(SpvOpUConvert, b_.t_u64(), {bits});
        }
        throw Error("unsupported register type conversion", line);
    }

    std::uint32_t mask_bits(std::uint32_t value, int bits)
    {
        if (bits >= 32) return value;
        return b_.value(SpvOpBitwiseAnd, b_.t_u32(), {value, b_.u32((1u << bits) - 1u)});
    }

    std::uint32_t sign_extend(std::uint32_t value, int bits)
    {
        if (bits >= 32) return value;
        const std::uint32_t shift = b_.u32(32u - static_cast<std::uint32_t>(bits));
        const std::uint32_t up = b_.value(SpvOpShiftLeftLogical, b_.t_u32(), {value, shift});
        return b_.value(SpvOpShiftRightArithmetic, b_.t_u32(), {up, shift});
    }

    std::uint32_t read_reg(const std::string &name, Type want, int line)
    {
        const Register &r = reg(name, line);
        std::uint32_t v = b_.value(SpvOpLoad, r.type_id, {r.variable});
        const std::uint32_t target = canonical(want);
        if (r.type.kind == 'p' || want.kind == 'p') {
            if (r.type.kind != want.kind) throw Error("predicate type mismatch for " + name, line);
            return v;
        }
        v = convert_bits(v, r.type_id, target, line);
        if (target == b_.t_u32() && want.bits < 32 && r.type.bits > want.bits) v = mask_bits(v, want.bits);
        return v;
    }

    void write_reg(const std::string &name, std::uint32_t value, std::uint32_t value_type, int line)
    {
        const Register &r = reg(name, line);
        if (r.type.kind == 'p') {
            if (value_type != b_.t_bool()) throw Error("non-predicate write to " + name, line);
        } else {
            value = convert_bits(value, value_type, r.type_id, line);
            if (r.type_id == b_.t_u32() && r.type.bits < 32) value = mask_bits(value, r.type.bits);
        }
        b_.op(SpvOpStore, {r.variable, value});
    }

    void write(const Operand &o, std::uint32_t value, Type type, int line)
    {
        if (o.kind == Operand::Kind::symbol && o.name == "_") return; // sink
        if (o.kind != Operand::Kind::reg) throw Error("destination must be a register", line);
        // PTX sign-extends signed values into a wider destination register.
        if (type.kind == 's' && type.bits <= 32 && reg(o.name, line).type_id == b_.t_u64()) {
            if (type.bits < 32) value = sign_extend(value, type.bits);
            write_reg(o.name, b_.value(SpvOpSConvert, b_.t_u64(), {value}), b_.t_u64(), line);
            return;
        }
        write_reg(o.name, value, canonical(type), line);
    }

    // ---- built-ins -----------------------------------------------------------
    std::uint32_t builtin_vector(SpvBuiltIn builtin)
    {
        auto found = builtins_.find(builtin);
        if (found != builtins_.end()) return found->second;
        const std::uint32_t v3 = b_.t_vec(b_.t_u32(), 3);
        const std::uint32_t var = b_.global_variable(b_.t_ptr(SpvStorageClassInput, v3), SpvStorageClassInput);
        b_.decorate(var, SpvDecorationBuiltIn, {static_cast<std::uint32_t>(builtin)});
        b_.mark_input(var);
        builtins_[builtin] = var;
        return var;
    }

    std::uint32_t workgroup_size_component(int c)
    {
        if (workgroup_size_[0] == 0) {
            for (int i = 0; i < 3; ++i)
                workgroup_size_[i] = b_.spec_u32(kSpecWorkgroupX + static_cast<std::uint32_t>(i), 1);
            const std::uint32_t composite = b_.id();
            b_.raw_global(SpvOpSpecConstantComposite,
                          {b_.t_vec(b_.t_u32(), 3), composite, workgroup_size_[0],
                           workgroup_size_[1], workgroup_size_[2]});
            b_.decorate(composite, SpvDecorationBuiltIn, {SpvBuiltInWorkgroupSize});
        }
        return workgroup_size_[c];
    }

    std::uint32_t special(const std::string &name, int line)
    {
        const auto dot = name.find('.');
        const std::string base = name.substr(0, dot);
        const int c = dot == std::string::npos ? 0 : name[dot + 1] - 'x';
        auto component = [&](SpvBuiltIn builtin) {
            const std::uint32_t v3 = b_.t_vec(b_.t_u32(), 3);
            const std::uint32_t vec = b_.value(SpvOpLoad, v3, {builtin_vector(builtin)});
            return b_.value(SpvOpCompositeExtract, b_.t_u32(), {vec, static_cast<std::uint32_t>(c)});
        };
        if (base == "%tid") return component(SpvBuiltInLocalInvocationId);
        if (base == "%ctaid") return component(SpvBuiltInWorkgroupId);
        if (base == "%nctaid") return component(SpvBuiltInNumWorkgroups);
        if (base == "%ntid") return workgroup_size_component(c);
        if (base == "%warpsize") return b_.u32(32);
        throw Error("unsupported special register " + name, line);
    }

    std::uint32_t symbol_address(const std::string &name, int line)
    {
        if (auto it = shared_offset_.find(name); it != shared_offset_.end()) return b_.u32(it->second);
        if (auto it = local_offset_.find(name); it != local_offset_.end()) return b_.u32(it->second);
        if (auto it = symbol_constants_.find(name); it != symbol_constants_.end()) return it->second;
        throw Error("unknown symbol " + name, line);
    }

    // ---- operands --------------------------------------------------------------
    std::uint32_t operand(const Operand &o, Type want, int line)
    {
        std::uint32_t v = 0;
        switch (o.kind) {
        case Operand::Kind::reg:
            v = read_reg(o.name, want, line);
            break;
        case Operand::Kind::special: {
            v = special(o.name, line);
            if (want.bits == 64) v = b_.value(SpvOpUConvert, b_.t_u64(), {v});
            else if (want.kind == 'f') throw Error("special register read as float", line);
            else if (want.bits < 32) v = mask_bits(v, want.bits);
            break;
        }
        case Operand::Kind::imm:
            if (want.kind == 'p') v = b_.boolean(o.imm != 0);
            else if (want.kind == 'f') throw Error("integer literal used as float operand", line);
            else if (want.bits == 64) v = b_.constant64(static_cast<std::uint64_t>(o.imm));
            else {
                std::uint32_t bits = static_cast<std::uint32_t>(o.imm);
                if (want.bits < 32) bits &= (1u << want.bits) - 1u;
                v = b_.u32(bits);
            }
            break;
        case Operand::Kind::fimm:
            if (o.fwidth != 32) throw Error("f64 literals are unsupported", line);
            if (want.kind == 'f') v = b_.f32_bits(static_cast<std::uint32_t>(o.fbits));
            else if (want.bits == 32) v = b_.u32(static_cast<std::uint32_t>(o.fbits));
            else throw Error("float literal width mismatch", line);
            break;
        case Operand::Kind::symbol: {
            if (want.kind == 'f' || want.kind == 'p') throw Error("symbol used as non-integer", line);
            v = symbol_address(o.name, line);
            if (want.bits == 64) v = b_.value(SpvOpUConvert, b_.t_u64(), {v});
            break;
        }
        case Operand::Kind::address: // "symbol+offset" immediate
            if (o.base != Operand::Base::symbol) throw Error("unexpected address operand", line);
            v = b_.value(SpvOpIAdd, b_.t_u32(),
                         {symbol_address(o.name, line), b_.u32(static_cast<std::uint32_t>(o.imm))});
            break;
        default:
            throw Error("unsupported operand kind", line);
        }
        if (o.negated) {
            if (want.kind != 'p') throw Error("'!' on a non-predicate operand", line);
            v = b_.value(SpvOpLogicalNot, b_.t_bool(), {v});
        }
        return v;
    }

    // A byte address within a state space, kept as a u32 value plus a
    // constant offset. PTX accesses are naturally aligned, so an aligned
    // offset means the base is aligned too and the element index can be
    // (base >> k) + offset / size: drivers then share the base computation
    // between neighbouring accesses and fold the offset into the access.
    // The split is exact as long as base itself does not wrap below the
    // start of its space, so it is used only for the heap, where that holds
    // for |offset| < 64 KiB because the runtime never allocates its first
    // 64 KiB (kHeapReserve).
    struct Address {
        std::uint32_t base = 0;
        std::uint32_t offset = 0;
    };

    static Address offset_by(Address a, std::uint32_t bytes)
    {
        a.offset += bytes;
        return a;
    }

    std::uint32_t byte_address(const Address &a) { return add_offset(a.base, a.offset); }

    std::uint32_t element_index(const Address &a, std::uint32_t shift)
    {
        const std::int32_t offset = static_cast<std::int32_t>(a.offset);
        if ((a.offset & ((1u << shift) - 1u)) != 0 || offset <= -65536 || offset >= 65536)
            return b_.value(SpvOpShiftRightLogical, b_.t_u32(), {byte_address(a), b_.u32(shift)});
        const std::uint32_t index = b_.value(SpvOpShiftRightLogical, b_.t_u32(), {a.base, b_.u32(shift)});
        return add_offset(index, static_cast<std::uint32_t>(offset / (1 << shift)));
    }

    Address address(const Operand &o, int line)
    {
        if (o.kind != Operand::Kind::address) throw Error("expected a memory operand", line);
        const std::uint32_t offset = static_cast<std::uint32_t>(o.imm);
        switch (o.base) {
        case Operand::Base::none: return {b_.u32(offset), 0};
        case Operand::Base::reg: return {read_reg(o.name, Type{'u', 32}, line), offset};
        case Operand::Base::symbol: return {symbol_address(o.name, line), offset};
        }
        throw Error("bad memory operand", line);
    }

    // ---- interface variables ----------------------------------------------------
    std::uint32_t heap_view(int words)
    {
        int index = words == 1 ? 0 : words == 2 ? 1 : 2;
        if (heap_[index] != 0) return heap_[index];
        const std::uint32_t element = words == 1 ? b_.t_u32() : b_.t_vec(b_.t_u32(), static_cast<std::uint32_t>(words));
        const std::uint32_t rta = b_.unique_type(SpvOpTypeRuntimeArray, {element});
        b_.decorate(rta, SpvDecorationArrayStride, {static_cast<std::uint32_t>(4 * words)});
        const std::uint32_t block = b_.unique_type(SpvOpTypeStruct, {rta});
        b_.decorate(block, SpvDecorationBlock);
        b_.member_decorate(block, 0, SpvDecorationOffset, {0});
        const std::uint32_t var = b_.global_variable(b_.t_ptr(SpvStorageClassStorageBuffer, block),
                                                     SpvStorageClassStorageBuffer);
        b_.decorate(var, SpvDecorationDescriptorSet, {0});
        b_.decorate(var, SpvDecorationBinding, {kBindingHeap + static_cast<std::uint32_t>(index)});
        b_.decorate(var, SpvDecorationAliased);
        heap_element_[index] = element;
        heap_[index] = var;
        return var;
    }

    std::uint32_t heap_base()
    {
        if (heap_base_ == 0) heap_base_ = b_.spec_u32(kSpecHeapBase, 0);
        return heap_base_;
    }

    bool split_heap() const { return options_.heap_window_shift != 0; }

    // Split heap: binding 0 as an array of windows (see CompileOptions).
    std::uint32_t heap_windows_view()
    {
        if (heap_[0] != 0) return heap_[0];
        const std::uint32_t element = b_.t_u32();
        const std::uint32_t rta = b_.unique_type(SpvOpTypeRuntimeArray, {element});
        b_.decorate(rta, SpvDecorationArrayStride, {4});
        const std::uint32_t block = b_.unique_type(SpvOpTypeStruct, {rta});
        b_.decorate(block, SpvDecorationBlock);
        b_.member_decorate(block, 0, SpvDecorationOffset, {0});
        // One more window than the heap needs: out-of-range indices are
        // clamped to it, and the runtime binds a dummy buffer there.
        const std::uint32_t array = b_.unique_type(SpvOpTypeArray, {block, b_.u32(options_.heap_windows + 1)});
        const std::uint32_t var = b_.global_variable(b_.t_ptr(SpvStorageClassStorageBuffer, array),
                                                     SpvStorageClassStorageBuffer);
        b_.decorate(var, SpvDecorationDescriptorSet, {0});
        b_.decorate(var, SpvDecorationBinding, {kBindingHeap});
        b_.decorate(var, SpvDecorationAliased);
        b_.extension("SPV_EXT_descriptor_indexing");
        b_.capability(SpvCapabilityShaderNonUniform);
        b_.capability(SpvCapabilityStorageBufferArrayNonUniformIndexing);
        heap_element_[0] = element;
        heap_[0] = var;
        return var;
    }

    std::uint32_t heap_pointer(int words, const Address &address)
    {
        if (split_heap()) {
            if (words != 1) throw Error("internal: vector heap access with a split heap", 0);
            const std::uint32_t u32 = b_.t_u32();
            const std::uint32_t var = heap_windows_view();
            const std::uint32_t relative = b_.value(SpvOpISub, u32, {byte_address(address), heap_base()});
            // Clamped: an address outside the heap (garbage data) must not
            // index past the descriptor array, which is undefined behaviour.
            // The extra last window is a dummy buffer, so such accesses never
            // reach live data; robustBufferAccess keeps them inside it.
            const std::uint32_t window = b_.ext(u32, GLSLstd450UMin,
                {b_.value(SpvOpShiftRightLogical, u32, {relative, b_.u32(options_.heap_window_shift)}),
                 b_.u32(options_.heap_windows)});
            const std::uint32_t within = b_.value(SpvOpBitwiseAnd, u32,
                                                  {relative, b_.u32((1u << options_.heap_window_shift) - 1u)});
            const std::uint32_t element = b_.value(SpvOpShiftRightLogical, u32, {within, b_.u32(2)});
            // Invocations may access different windows.
            b_.decorate(window, SpvDecorationNonUniform);
            const std::uint32_t pointer = b_.value(SpvOpAccessChain, b_.t_ptr(SpvStorageClassStorageBuffer, u32),
                                                   {var, window, b_.u32(0), element});
            b_.decorate(pointer, SpvDecorationNonUniform);
            return pointer;
        }
        const std::uint32_t var = heap_view(words);
        const int index = words == 1 ? 0 : words == 2 ? 1 : 2;
        // The heap base is aligned far beyond any access size.
        const Address relative{b_.value(SpvOpISub, b_.t_u32(), {address.base, heap_base()}), address.offset};
        const std::uint32_t element = element_index(relative, words == 1 ? 2u : words == 2 ? 3u : 4u);
        return b_.value(SpvOpAccessChain,
                        b_.t_ptr(SpvStorageClassStorageBuffer, heap_element_[index]),
                        {var, b_.u32(0), element});
    }

    std::uint32_t launch_block()
    {
        if (launch_ != 0) return launch_;
        const std::uint32_t v4 = b_.t_vec(b_.t_u32(), 4);
        const std::uint32_t count = std::max<std::uint32_t>(1, (info_.launch_block_bytes + 15) / 16);
        const std::uint32_t array = b_.unique_type(SpvOpTypeArray, {v4, b_.u32(count)});
        b_.decorate(array, SpvDecorationArrayStride, {16});
        const std::uint32_t block = b_.unique_type(SpvOpTypeStruct, {array});
        b_.decorate(block, SpvDecorationBlock);
        b_.member_decorate(block, 0, SpvDecorationOffset, {0});
        launch_ = b_.global_variable(b_.t_ptr(SpvStorageClassUniform, block), SpvStorageClassUniform);
        b_.decorate(launch_, SpvDecorationDescriptorSet, {0});
        b_.decorate(launch_, SpvDecorationBinding, {kBindingLaunch});
        b_.decorate(launch_, SpvDecorationNonWritable);
        return launch_;
    }

    std::uint32_t launch_word(std::uint32_t byte_offset)
    {
        if (byte_offset % 4 != 0) throw Error("unaligned launch-block read", f_.line);
        const std::uint32_t ptr = b_.value(SpvOpAccessChain, b_.t_ptr(SpvStorageClassUniform, b_.t_u32()),
                                           {launch_block(), b_.u32(0), b_.u32(byte_offset / 16),
                                            b_.u32((byte_offset / 4) % 4)});
        return b_.value(SpvOpLoad, b_.t_u32(), {ptr});
    }

    std::uint32_t launch_word_dynamic(std::uint32_t word_index)
    {
        const std::uint32_t vec = b_.value(SpvOpShiftRightLogical, b_.t_u32(), {word_index, b_.u32(2)});
        const std::uint32_t lane = b_.value(SpvOpBitwiseAnd, b_.t_u32(), {word_index, b_.u32(3)});
        const std::uint32_t ptr = b_.value(SpvOpAccessChain, b_.t_ptr(SpvStorageClassUniform, b_.t_u32()),
                                           {launch_block(), b_.u32(0), vec, lane});
        return b_.value(SpvOpLoad, b_.t_u32(), {ptr});
    }

    std::uint32_t shared_array()
    {
        if (shared_ != 0) return shared_;
        if (options_.shared_in_storage) {
            const std::uint32_t rta = b_.unique_type(SpvOpTypeRuntimeArray, {b_.t_u32()});
            b_.decorate(rta, SpvDecorationArrayStride, {4});
            const std::uint32_t block = b_.unique_type(SpvOpTypeStruct, {rta});
            b_.decorate(block, SpvDecorationBlock);
            b_.member_decorate(block, 0, SpvDecorationOffset, {0});
            shared_ = b_.global_variable(b_.t_ptr(SpvStorageClassStorageBuffer, block), SpvStorageClassStorageBuffer);
            b_.decorate(shared_, SpvDecorationDescriptorSet, {0});
            b_.decorate(shared_, SpvDecorationBinding, {kBindingScratch});
            return shared_;
        }
        const std::uint32_t words = b_.spec_u32(kSpecSharedWords, std::max<std::uint32_t>(1, info_.static_shared_bytes / 4));
        const std::uint32_t array = b_.unique_type(SpvOpTypeArray, {b_.t_u32(), words});
        shared_ = b_.global_variable(b_.t_ptr(SpvStorageClassWorkgroup, array), SpvStorageClassWorkgroup);
        return shared_;
    }

    std::uint32_t local_array()
    {
        if (local_ != 0) return local_;
        if (info_.local_bytes == 0) throw Error("local memory access without .local storage", f_.line);
        const std::uint32_t array = b_.unique_type(SpvOpTypeArray, {b_.t_u32(), b_.u32(info_.local_bytes / 4)});
        local_ = b_.local_variable(b_.t_ptr(SpvStorageClassFunction, array));
        return local_;
    }

    // ---- word-level memory primitives ----------------------------------------
    std::uint32_t word_pointer(Space space, const Address &address)
    {
        if (space != Space::shared && space != Space::local) return heap_pointer(1, address);
        // Not element_index: shared and local memory start at address 0, so
        // a valid access such as st[top + 1] with top == -1 has a base below
        // the start of the space and the base/offset split would be wrong.
        const std::uint32_t index = b_.value(SpvOpShiftRightLogical, b_.t_u32(), {byte_address(address), b_.u32(2)});
        switch (space) {
        case Space::shared:
            if (options_.shared_in_storage) {
                const std::uint32_t slot = b_.value(SpvOpIAdd, b_.t_u32(), {scratch_base_, index});
                return b_.value(SpvOpAccessChain, b_.t_ptr(SpvStorageClassStorageBuffer, b_.t_u32()),
                                {shared_array(), b_.u32(0), slot});
            }
            return b_.value(SpvOpAccessChain, b_.t_ptr(SpvStorageClassWorkgroup, b_.t_u32()),
                            {shared_array(), index});
        case Space::local:
            return b_.value(SpvOpAccessChain, b_.t_ptr(SpvStorageClassFunction, b_.t_u32()),
                            {local_array(), index});
        default:
            return heap_pointer(1, address);
        }
    }

    std::uint32_t load_pointer(std::uint32_t type, std::uint32_t pointer, bool is_volatile)
    {
        if (is_volatile)
            return b_.value(SpvOpLoad, type, {pointer, SpvMemoryAccessVolatileMask});
        return b_.value(SpvOpLoad, type, {pointer});
    }

    void store_pointer(std::uint32_t pointer, std::uint32_t value, bool is_volatile)
    {
        if (is_volatile) b_.op(SpvOpStore, {pointer, value, SpvMemoryAccessVolatileMask});
        else b_.op(SpvOpStore, {pointer, value});
    }

    std::uint32_t add_offset(std::uint32_t address, std::uint32_t offset)
    {
        if (offset == 0) return address;
        return b_.value(SpvOpIAdd, b_.t_u32(), {address, b_.u32(offset)});
    }

    // Loads `count` consecutive 32-bit words starting at an aligned address.
    std::vector<std::uint32_t> load_words(Space space, const Address &address, int count, bool is_volatile)
    {
        std::vector<std::uint32_t> words;
        if (space == Space::global && (count == 2 || count == 4) && !is_volatile && !split_heap()) {
            const std::uint32_t vec = load_pointer(b_.t_vec(b_.t_u32(), static_cast<std::uint32_t>(count)),
                                                   heap_pointer(count, address), false);
            for (int i = 0; i < count; ++i)
                words.push_back(b_.value(SpvOpCompositeExtract, b_.t_u32(), {vec, static_cast<std::uint32_t>(i)}));
            return words;
        }
        for (int i = 0; i < count; ++i)
            words.push_back(load_pointer(b_.t_u32(), word_pointer(space, offset_by(address, 4u * i)), is_volatile));
        return words;
    }

    void store_words(Space space, const Address &address, const std::vector<std::uint32_t> &words, bool is_volatile)
    {
        const int count = static_cast<int>(words.size());
        if (space == Space::global && (count == 2 || count == 4) && !is_volatile && !split_heap()) {
            std::vector<std::uint32_t> operands{b_.t_vec(b_.t_u32(), static_cast<std::uint32_t>(count))};
            const std::uint32_t composite = b_.id();
            operands.push_back(composite);
            operands.insert(operands.end(), words.begin(), words.end());
            b_.op(SpvOpCompositeConstruct, operands);
            store_pointer(heap_pointer(count, address), composite, false);
            return;
        }
        for (int i = 0; i < count; ++i)
            store_pointer(word_pointer(space, offset_by(address, 4u * i)), words[i], is_volatile);
    }

    // Read-modify-write of a sub-word value. Global and shared memory use
    // atomics so neighbouring bytes written by other invocations survive.
    void store_subword(Space space, const Address &address, std::uint32_t value, int bits)
    {
        const std::uint32_t u32 = b_.t_u32();
        const std::uint32_t shift = b_.value(SpvOpShiftLeftLogical, u32,
            {b_.value(SpvOpBitwiseAnd, u32, {byte_address(address), b_.u32(3)}), b_.u32(3)});
        const std::uint32_t mask = b_.value(SpvOpShiftLeftLogical, u32, {b_.u32((1u << bits) - 1u), shift});
        const std::uint32_t bits_in = b_.value(SpvOpShiftLeftLogical, u32, {mask_bits(value, bits), shift});
        const std::uint32_t pointer = word_pointer(space, address);
        if (space == Space::local) {
            const std::uint32_t old = b_.value(SpvOpLoad, u32, {pointer});
            const std::uint32_t kept = b_.value(SpvOpBitwiseAnd, u32, {old, b_.value(SpvOpNot, u32, {mask})});
            b_.op(SpvOpStore, {pointer, b_.value(SpvOpBitwiseOr, u32, {kept, bits_in})});
            return;
        }
        // Replace the bytes with a single compare-and-swap loop, so other
        // invocations never see a half-written word (an AtomicAnd followed
        // by an AtomicOr briefly exposed the bytes as zero).
        const std::uint32_t scope = b_.u32(space == Space::shared ? SpvScopeWorkgroup : SpvScopeDevice);
        const std::uint32_t semantics = b_.u32(SpvMemorySemanticsMaskNone);
        const std::uint32_t keep = b_.value(SpvOpNot, u32, {mask});
        const std::uint32_t header = b_.id(), body = b_.id(), cont = b_.id(), merge = b_.id();
        b_.branch(header);
        b_.label(header);
        b_.op(SpvOpLoopMerge, {merge, cont, SpvLoopControlMaskNone});
        b_.branch(body);
        b_.label(body);
        const std::uint32_t old = b_.value(SpvOpAtomicLoad, u32, {pointer, scope, semantics});
        const std::uint32_t replaced = b_.value(SpvOpBitwiseOr, u32, {b_.value(SpvOpBitwiseAnd, u32, {old, keep}), bits_in});
        const std::uint32_t seen =
            b_.value(SpvOpAtomicCompareExchange, u32, {pointer, scope, semantics, semantics, replaced, old});
        b_.branch_conditional(b_.value(SpvOpIEqual, b_.t_bool(), {seen, old}), merge, cont);
        b_.label(cont);
        b_.branch(header);
        b_.label(merge);
    }

    std::uint32_t extract_subword(std::uint32_t word, std::uint32_t address, int bits, bool is_signed)
    {
        const std::uint32_t u32 = b_.t_u32();
        const std::uint32_t shift = b_.value(SpvOpShiftLeftLogical, u32,
            {b_.value(SpvOpBitwiseAnd, u32, {address, b_.u32(3)}), b_.u32(3)});
        const std::uint32_t shifted = b_.value(SpvOpShiftRightLogical, u32, {word, shift});
        return is_signed ? sign_extend(shifted, bits) : mask_bits(shifted, bits);
    }

    // Converts loaded 32-bit words into element values of `type`.
    std::vector<std::uint32_t> words_to_elements(const std::vector<std::uint32_t> &words, Type type, int count)
    {
        std::vector<std::uint32_t> out;
        if (type.bits == 32) {
            for (std::uint32_t w : words)
                out.push_back(type.kind == 'f' ? b_.value(SpvOpBitcast, b_.t_f32(), {w}) : w);
        } else if (type.bits == 64) {
            for (int i = 0; i < count; ++i) {
                const std::uint32_t lo = b_.value(SpvOpUConvert, b_.t_u64(), {words[2 * i]});
                const std::uint32_t hi = b_.value(SpvOpUConvert, b_.t_u64(), {words[2 * i + 1]});
                const std::uint32_t shifted = b_.value(SpvOpShiftLeftLogical, b_.t_u64(), {hi, b_.u32(32)});
                out.push_back(b_.value(SpvOpBitwiseOr, b_.t_u64(), {lo, shifted}));
            }
        } else {
            const int per_word = 32 / type.bits;
            for (int i = 0; i < count; ++i) {
                const std::uint32_t w = words[i / per_word];
                const std::uint32_t shift = static_cast<std::uint32_t>((i % per_word) * type.bits);
                std::uint32_t v = shift ? b_.value(SpvOpShiftRightLogical, b_.t_u32(), {w, b_.u32(shift)}) : w;
                v = type.kind == 's' ? sign_extend(v, type.bits) : mask_bits(v, type.bits);
                out.push_back(v);
            }
        }
        return out;
    }

    std::vector<std::uint32_t> elements_to_words(const std::vector<std::uint32_t> &elems, Type type)
    {
        std::vector<std::uint32_t> out;
        if (type.bits == 32) {
            for (std::uint32_t e : elems)
                out.push_back(type.kind == 'f' ? b_.value(SpvOpBitcast, b_.t_u32(), {e}) : e);
        } else if (type.bits == 64) {
            for (std::uint32_t e : elems) {
                out.push_back(b_.value(SpvOpUConvert, b_.t_u32(), {e}));
                const std::uint32_t hi = b_.value(SpvOpShiftRightLogical, b_.t_u64(), {e, b_.u32(32)});
                out.push_back(b_.value(SpvOpUConvert, b_.t_u32(), {hi}));
            }
        } else {
            const int per_word = 32 / type.bits;
            for (std::size_t i = 0; i < elems.size(); i += per_word) {
                std::uint32_t w = 0;
                for (int j = 0; j < per_word && i + j < elems.size(); ++j) {
                    std::uint32_t v = mask_bits(elems[i + j], type.bits);
                    if (j) v = b_.value(SpvOpShiftLeftLogical, b_.t_u32(), {v, b_.u32(static_cast<std::uint32_t>(j * type.bits))});
                    w = w ? b_.value(SpvOpBitwiseOr, b_.t_u32(), {w, v}) : v;
                }
                out.push_back(w);
            }
        }
        return out;
    }

    // ---- loads and stores -------------------------------------------------------
    void lower_param_load(const Instruction &in)
    {
        const Operand &a = in.operands[1];
        if (a.kind != Operand::Kind::address || a.base != Operand::Base::symbol)
            throw Error("unsupported parameter address", in.line);
        const auto found = params_.find(a.name);
        if (found == params_.end()) throw Error("unknown parameter " + a.name, in.line);
        const Type type = last_type(in);
        const int count = vector_width(in);
        const std::uint32_t byte = found->second.offset + static_cast<std::uint32_t>(a.imm);
        const std::uint32_t total = static_cast<std::uint32_t>(type.bits / 8 * count);
        std::vector<std::uint32_t> words;
        for (std::uint32_t w = byte / 4; w < (byte + total + 3) / 4; ++w) words.push_back(launch_word(w * 4));
        std::vector<std::uint32_t> elems;
        if (type.bits < 32) {
            for (int i = 0; i < count; ++i) {
                const std::uint32_t at = byte + static_cast<std::uint32_t>(i * type.bits / 8);
                const std::uint32_t w = words[at / 4 - byte / 4];
                std::uint32_t v = (at % 4) ? b_.value(SpvOpShiftRightLogical, b_.t_u32(), {w, b_.u32((at % 4) * 8)}) : w;
                elems.push_back(type.kind == 's' ? sign_extend(v, type.bits) : mask_bits(v, type.bits));
            }
        } else {
            if (byte % 4) throw Error("unaligned parameter load", in.line);
            elems = words_to_elements(words, type, count);
        }
        write_elements(in.operands[0], elems, type, in.line);
    }

    void write_elements(const Operand &dst, const std::vector<std::uint32_t> &elems, Type type, int line)
    {
        if (dst.kind == Operand::Kind::vector) {
            for (std::size_t i = 0; i < dst.elems.size(); ++i) write(dst.elems[i], elems[i], type, line);
        } else {
            write(dst, elems[0], type, line);
        }
    }

    std::vector<std::uint32_t> read_elements(const Operand &src, Type type, int line)
    {
        std::vector<std::uint32_t> out;
        if (src.kind == Operand::Kind::vector)
            for (const Operand &e : src.elems) out.push_back(operand(e, type, line));
        else
            out.push_back(operand(src, type, line));
        return out;
    }

    Space access_space(const Instruction &in, const Operand &address_operand)
    {
        Space space = memory_space(in);
        if (space == Space::none) space = resolve_generic(address_operand, in.line);
        if (space == Space::constant) space = Space::global;
        return space;
    }

    void note_space(Space space, bool is_volatile)
    {
        // Emulated warps need no particular subgroup size.
        if (space == Space::shared && is_volatile && !options_.emulate_warps) info_.warp_synchronous = true;
    }

    void lower_load(const Instruction &in)
    {
        if (memory_space(in) == Space::param) {
            lower_param_load(in);
            return;
        }
        const Type type = last_type(in);
        const int count = vector_width(in);
        const bool is_volatile = in.has("volatile");
        const Space space = access_space(in, in.operands[1]);
        note_space(space, is_volatile);
        const Address addr = address(in.operands[1], in.line);
        const int total = type.bits / 8 * count;
        std::vector<std::uint32_t> elems;
        if (total < 4) {
            const std::uint32_t word = load_pointer(b_.t_u32(), word_pointer(space, addr), is_volatile);
            for (int i = 0; i < count; ++i)
                elems.push_back(extract_subword(word, byte_address(offset_by(addr, static_cast<std::uint32_t>(i * type.bits / 8))),
                                                type.bits, type.kind == 's'));
        } else {
            elems = words_to_elements(load_words(space, addr, total / 4, is_volatile), type, count);
        }
        write_elements(in.operands[0], elems, type, in.line);
    }

    void lower_store(const Instruction &in)
    {
        const Type type = last_type(in);
        const int count = vector_width(in);
        const bool is_volatile = in.has("volatile");
        const Space space = access_space(in, in.operands[0]);
        if (space == Space::param) throw Error("st.param is only valid in device functions", in.line);
        note_space(space, is_volatile);
        const Address addr = address(in.operands[0], in.line);
        const std::vector<std::uint32_t> elems = read_elements(in.operands[1], type, in.line);
        const int total = type.bits / 8 * count;
        // With emulate_warps the barriers are emitted by lower_instruction,
        // outside any guard, so that they stay in uniform control flow.
        const bool warp_sync = space == Space::shared && is_volatile && !options_.emulate_warps;
        if (warp_sync) warp_barrier();
        if (total < 4) {
            for (int i = 0; i < count; ++i)
                store_subword(space, offset_by(addr, static_cast<std::uint32_t>(i * type.bits / 8)), elems[i], type.bits);
        } else {
            store_words(space, addr, elements_to_words(elems, type), is_volatile);
        }
        if (warp_sync) warp_barrier();
    }

    // Warp-synchronous PTX assumes the 32 lanes of a warp execute in
    // lockstep, so a volatile shared store is visible to every lane's next
    // instruction. Vulkan guarantees neither, so surround such stores with
    // execution+memory barriers: subgroup-wide on devices with 32-lane (or
    // wider) subgroups, workgroup-wide with emulate_warps. A workgroup
    // barrier makes every warp advance store by store together, and since
    // all loads of a phase complete before any store of the next, this is
    // exactly lockstep execution.
    void warp_barrier()
    {
        const std::uint32_t scope = options_.emulate_warps ? SpvScopeWorkgroup : SpvScopeSubgroup;
        b_.op(SpvOpControlBarrier, {b_.u32(scope), b_.u32(scope),
                                    b_.u32(SpvMemorySemanticsAcquireReleaseMask |
                                           (options_.shared_in_storage ? SpvMemorySemanticsUniformMemoryMask
                                                                       : SpvMemorySemanticsWorkgroupMemoryMask))});
    }

    bool is_warp_sync_store(const Instruction &in)
    {
        return in.op == "st" && in.has("volatile") && access_space(in, in.operands[0]) == Space::shared;
    }

    void lower_texture(const Instruction &in)
    {
        // tex.1d.v4.dtype.s32 {d0,d1,d2,d3}, [texref, {coord}]
        if (!in.has("1d") || !in.has("v4") || in.operands.size() != 2)
            throw Error("unsupported texture instruction " + in.text(), in.line);
        const auto [dtype, ctype] = two_types(in);
        if (ctype.kind != 's' || ctype.bits != 32 || dtype.bits != 32)
            throw Error("unsupported texture types " + in.text(), in.line);
        const Operand &src = in.operands[1];
        if (src.elems.size() != 2 || src.elems[1].kind != Operand::Kind::vector || src.elems[1].elems.size() != 1)
            throw Error("unsupported texture operand", in.line);
        const auto found = texture_index_.find(src.elems[0].name);
        if (found == texture_index_.end()) throw Error("unknown texture " + src.elems[0].name, in.line);
        const std::uint32_t entry = info_.texture_table_offset + found->second * kTextureEntryBytes;
        const std::uint32_t u32 = b_.t_u32();
        const std::uint32_t base = launch_word(entry);
        const std::uint32_t count = launch_word(entry + 4);
        const std::uint32_t comps = launch_word(entry + 8);
        const std::uint32_t coord = operand(src.elems[1].elems[0], Type{'s', 32}, in.line);
        // tex1Dfetch returns zero outside [0, count): an unsigned compare
        // rejects negative coordinates as well.
        const std::uint32_t in_range = b_.value(SpvOpULessThan, b_.t_bool(), {coord, count});
        const std::uint32_t index = b_.value(SpvOpSelect, u32, {in_range, coord, b_.u32(0)});
        const std::uint32_t element = b_.value(SpvOpIAdd, u32,
            {base, b_.value(SpvOpShiftLeftLogical, u32, {b_.value(SpvOpIMul, u32, {index, comps}), b_.u32(2)})});
        std::uint32_t values[4];
        for (int c = 0; c < 4; ++c) {
            const std::uint32_t present = b_.value(SpvOpLogicalAnd, b_.t_bool(),
                {in_range, b_.value(SpvOpULessThan, b_.t_bool(), {b_.u32(static_cast<std::uint32_t>(c)), comps})});
            // Clamp the component address so absent components never read
            // outside the bound texture memory.
            const std::uint32_t at = b_.value(SpvOpSelect, u32,
                {present, add_offset(element, 4u * c), element});
            const std::uint32_t word = load_pointer(u32, heap_pointer(1, Address{at, 0}), false);
            values[c] = b_.value(SpvOpSelect, u32, {present, word, b_.u32(0)});
        }
        const Operand &dst = in.operands[0];
        for (int c = 0; c < 4 && c < static_cast<int>(dst.elems.size()); ++c) {
            const std::uint32_t v = dtype.kind == 'f' ? b_.value(SpvOpBitcast, b_.t_f32(), {values[c]}) : values[c];
            write(dst.elems[c], v, dtype, in.line);
        }
    }

    // ---- floating point -------------------------------------------------------
    std::uint32_t flush(std::uint32_t value)
    {
        // |x| < 2^-126 (false for NaN) selects the signed zero.
        const std::uint32_t u32 = b_.t_u32();
        const std::uint32_t magnitude = b_.ext(b_.t_f32(), GLSLstd450FAbs, {value});
        const std::uint32_t denormal = b_.value(SpvOpFOrdLessThan, b_.t_bool(), {magnitude, b_.f32_bits(0x00800000u)});
        const std::uint32_t bits = b_.value(SpvOpBitcast, u32, {value});
        const std::uint32_t sign = b_.value(SpvOpBitwiseAnd, u32, {bits, b_.u32(0x80000000u)});
        const std::uint32_t result = b_.value(SpvOpSelect, u32, {denormal, sign, bits});
        return b_.value(SpvOpBitcast, b_.t_f32(), {result});
    }

    std::uint32_t fin(const Instruction &in, std::size_t index)
    {
        const std::uint32_t v = operand(in.operands[index], Type{'f', 32}, in.line);
        return in.has("ftz") && !native_ftz_ && !flush_safe(in.operands[index]) ? flush(v) : v;
    }

    void fout(const Instruction &in, std::uint32_t value)
    {
        if (in.has("ftz") && !native_ftz_) value = flush(value);
        write(in.operands[0], value, Type{'f', 32}, in.line);
    }

    bool lower_float(const Instruction &in, Type type)
    {
        if (type.kind != 'f') return false;
        if (type.bits != 32) throw Error("only f32 arithmetic is supported", in.line);
        const std::uint32_t f32 = b_.t_f32();
        const std::string &op = in.op;
        // Only round-to-nearest is implemented; directed rounding would be
        // silently wrong.
        if (in.has("rz") || in.has("rm") || in.has("rp"))
            throw Error("directed rounding is unsupported: " + in.text(), in.line);
        if ((op == "add" || op == "sub") && fused_.count(&in)) {
            fout(in, fused_add(in, fused_.at(&in)));
        } else if (op == "add" || op == "sub" || op == "mul") {
            const SpvOp code = op == "add" ? SpvOpFAdd : op == "sub" ? SpvOpFSub : SpvOpFMul;
            const std::uint32_t x = fin(in, 1), y = fin(in, 2);
            if (const auto stash = stash_.find(&in); stash != stash_.end()) {
                // Inputs of a product fused into later adds (see plan_contractions).
                b_.op(SpvOpStore, {stash->second.first, x});
                b_.op(SpvOpStore, {stash->second.second, y});
            }
            const std::uint32_t r = b_.value(code, f32, {x, y});
            // Contraction is decided by plan_contractions (mirroring ptxas);
            // everything else must stay unfused, which some Vulkan drivers
            // (Intel) would otherwise not guarantee.
            b_.decorate(r, SpvDecorationNoContraction);
            if (in.has("sat")) throw Error(".sat is unsupported", in.line);
            fout(in, r);
        } else if (op == "fma" || op == "mad") {
            if (in.has("sat")) throw Error(".sat is unsupported", in.line);
            fout(in, fused_multiply_add(fin(in, 1), fin(in, 2), fin(in, 3)));
        } else if (op == "div") {
            if (in.has("approx")) {
                // a * rcp(b), without .full's operand scaling: for
                // 2^126 < |b| < 2^128 the reciprocal flushes to zero.
                const std::uint32_t a = fin(in, 1), b = fin(in, 2);
                const std::uint32_t r = b_.value(SpvOpFDiv, f32, {b_.f32_bits(0x3f800000u), b});
                fout(in, b_.value(SpvOpFMul, f32, {a, r}));
            } else if (in.has("full")) {
                // NVIDIA hardware evaluates div.full as a * rcp(b) with
                // operand scaling for |b| > 2^126, mirrored here.
                const std::uint32_t a = fin(in, 1), b = fin(in, 2);
                const std::uint32_t huge = b_.value(SpvOpFOrdGreaterThan, b_.t_bool(),
                    {b_.ext(f32, GLSLstd450FAbs, {b}), b_.f32_bits(0x7e800000u)});
                const std::uint32_t scale = b_.value(SpvOpSelect, f32, {huge, b_.f32_bits(0x3e800000u), b_.f32_bits(0x3f800000u)});
                const std::uint32_t sa = b_.value(SpvOpFMul, f32, {a, scale});
                const std::uint32_t sb = b_.value(SpvOpFMul, f32, {b, scale});
                const std::uint32_t r = b_.value(SpvOpFDiv, f32, {b_.f32_bits(0x3f800000u), sb});
                fout(in, b_.value(SpvOpFMul, f32, {sa, r}));
            } else {
                fout(in, b_.value(SpvOpFDiv, f32, {fin(in, 1), fin(in, 2)}));
            }
        } else if (op == "rcp") {
            fout(in, b_.value(SpvOpFDiv, f32, {b_.f32_bits(0x3f800000u), fin(in, 1)}));
        } else if (op == "sqrt") {
            fout(in, b_.ext(f32, GLSLstd450Sqrt, {fin(in, 1)}));
        } else if (op == "rsqrt") {
            fout(in, b_.ext(f32, GLSLstd450InverseSqrt, {fin(in, 1)}));
        } else if (op == "sin" || op == "cos" || op == "lg2" || op == "ex2") {
            const GLSLstd450 code = op == "sin" ? GLSLstd450Sin : op == "cos" ? GLSLstd450Cos
                                  : op == "lg2" ? GLSLstd450Log2 : GLSLstd450Exp2;
            fout(in, b_.ext(f32, code, {fin(in, 1)}));
        } else if (op == "abs") {
            fout(in, b_.ext(f32, GLSLstd450FAbs, {fin(in, 1)}));
        } else if (op == "neg") {
            fout(in, b_.value(SpvOpFNegate, f32, {fin(in, 1)}));
        } else if (op == "min" || op == "max") {
            // PTX min/max return the non-NaN operand, exactly NMin/NMax.
            fout(in, b_.ext(f32, op == "min" ? GLSLstd450NMin : GLSLstd450NMax, {fin(in, 1), fin(in, 2)}));
        } else {
            return false;
        }
        return true;
    }

    std::uint32_t compare(const std::string &cmp, Type type, std::uint32_t a, std::uint32_t b, int line)
    {
        static const std::map<std::string, SpvOp> fcmp = {
            {"eq", SpvOpFOrdEqual}, {"ne", SpvOpFOrdNotEqual}, {"lt", SpvOpFOrdLessThan},
            {"le", SpvOpFOrdLessThanEqual}, {"gt", SpvOpFOrdGreaterThan},
            {"ge", SpvOpFOrdGreaterThanEqual}, {"equ", SpvOpFUnordEqual},
            {"neu", SpvOpFUnordNotEqual}, {"ltu", SpvOpFUnordLessThan},
            {"leu", SpvOpFUnordLessThanEqual}, {"gtu", SpvOpFUnordGreaterThan},
            {"geu", SpvOpFUnordGreaterThanEqual}};
        const std::uint32_t boolean = b_.t_bool();
        if (type.kind == 'f') {
            if (cmp == "num" || cmp == "nan") {
                const std::uint32_t na = b_.value(SpvOpIsNan, boolean, {a});
                const std::uint32_t nb = b_.value(SpvOpIsNan, boolean, {b});
                const std::uint32_t any = b_.value(SpvOpLogicalOr, boolean, {na, nb});
                return cmp == "nan" ? any : b_.value(SpvOpLogicalNot, boolean, {any});
            }
            const auto found = fcmp.find(cmp);
            if (found == fcmp.end()) throw Error("unsupported float comparison " + cmp, line);
            return b_.value(found->second, boolean, {a, b});
        }
        const bool is_signed = type.kind == 's';
        static const std::map<std::string, std::pair<SpvOp, SpvOp>> icmp = {
            {"eq", {SpvOpIEqual, SpvOpIEqual}}, {"ne", {SpvOpINotEqual, SpvOpINotEqual}},
            {"lt", {SpvOpULessThan, SpvOpSLessThan}}, {"le", {SpvOpULessThanEqual, SpvOpSLessThanEqual}},
            {"gt", {SpvOpUGreaterThan, SpvOpSGreaterThan}}, {"ge", {SpvOpUGreaterThanEqual, SpvOpSGreaterThanEqual}},
            {"lo", {SpvOpULessThan, SpvOpULessThan}}, {"ls", {SpvOpULessThanEqual, SpvOpULessThanEqual}},
            {"hi", {SpvOpUGreaterThan, SpvOpUGreaterThan}}, {"hs", {SpvOpUGreaterThanEqual, SpvOpUGreaterThanEqual}}};
        const auto found = icmp.find(cmp);
        if (found == icmp.end()) throw Error("unsupported integer comparison " + cmp, line);
        if (type.kind == 'p') {
            if (cmp != "eq" && cmp != "ne") throw Error("unsupported predicate comparison", line);
            return b_.value(cmp == "eq" ? SpvOpLogicalEqual : SpvOpLogicalNotEqual, boolean, {a, b});
        }
        if (is_signed && type.bits < 32) {
            a = sign_extend(a, type.bits);
            b = sign_extend(b, type.bits);
        }
        return b_.value(is_signed ? found->second.second : found->second.first, boolean, {a, b});
    }

    // ---- integer helpers ------------------------------------------------------
    std::uint32_t int_type(Type t) { return t.bits == 64 ? b_.t_u64() : b_.t_u32(); }

    std::uint32_t shift(const Instruction &in, Type type, bool left)
    {
        const std::uint32_t ty = int_type(type);
        const std::uint32_t value = operand(in.operands[1], type, in.line);
        const Operand &amount_op = in.operands[2];
        const std::uint32_t width = static_cast<std::uint32_t>(type.bits);
        const bool arithmetic = !left && type.kind == 's';
        std::uint32_t v = value;
        if (arithmetic && type.bits < 32) v = sign_extend(v, type.bits);
        const SpvOp code = left ? SpvOpShiftLeftLogical : arithmetic ? SpvOpShiftRightArithmetic : SpvOpShiftRightLogical;
        if (amount_op.kind == Operand::Kind::imm) {
            const std::uint64_t amount = static_cast<std::uint64_t>(amount_op.imm);
            if (amount < width) return b_.value(code, ty, {v, b_.u32(static_cast<std::uint32_t>(amount))});
            if (arithmetic) return b_.value(code, ty, {v, b_.u32(width - 1)});
            return ty == b_.t_u64() ? b_.constant64(0) : b_.u32(0);
        }
        // PTX clamps shift amounts to the register width.
        const std::uint32_t amount = operand(amount_op, Type{'u', 32}, in.line);
        const std::uint32_t clamped = b_.ext(b_.t_u32(), GLSLstd450UMin, {amount, b_.u32(width - 1)});
        const std::uint32_t shifted = b_.value(code, ty, {v, clamped});
        if (arithmetic) return shifted;
        const std::uint32_t too_far = b_.value(SpvOpUGreaterThanEqual, b_.t_bool(), {amount, b_.u32(width)});
        const std::uint32_t zero = ty == b_.t_u64() ? b_.constant64(0) : b_.u32(0);
        return b_.value(SpvOpSelect, ty, {too_far, zero, shifted});
    }

    // ---- instruction dispatch -------------------------------------------------
    void lower_instruction(const Instruction &in)
    {
        const bool emulated_warp_store = options_.emulate_warps && is_warp_sync_store(in);
        if (predicate_ != 0 && in.op == "bar")
            throw Error("bar.sync inside warp-synchronous code under a branch", in.line);
        if (emulated_warp_store) warp_barrier();
        // Condition: the instruction's guard, and the predicate of an arm
        // emitted by emit_predicated.
        std::uint32_t condition = predicate_;
        if (!in.guard.empty()) {
            std::uint32_t p = read_reg(in.guard, Type{'p', 1}, in.line);
            if (in.guard_negated) p = b_.value(SpvOpLogicalNot, b_.t_bool(), {p});
            condition = condition ? b_.value(SpvOpLogicalAnd, b_.t_bool(), {condition, p}) : p;
        }
        if (condition != 0) {
            // Predicated instruction: if (condition) { in }
            const std::uint32_t then_label = b_.id(), merge = b_.id();
            b_.op(SpvOpSelectionMerge, {merge, SpvSelectionControlMaskNone});
            b_.branch_conditional(condition, then_label, merge);
            b_.label(then_label);
            lower_unguarded(in);
            b_.branch(merge);
            b_.label(merge);
        } else {
            lower_unguarded(in);
        }
        if (emulated_warp_store) warp_barrier();
    }

    void lower_unguarded(const Instruction &in)
    {
        const std::string &op = in.op;
        const int line = in.line;
        if (op == "ld") return lower_load(in);
        if (op == "st") return lower_store(in);
        if (op == "tex") return lower_texture(in);
        if (op == "bar") {
            if (!(in.has("sync") || in.mods.empty()) || in.operands.size() != 1 ||
                in.operands[0].kind != Operand::Kind::imm || in.operands[0].imm != 0)
                throw Error("only bar.sync 0 is supported", line);
            info_.uses_barrier = true;
            b_.op(SpvOpControlBarrier, {b_.u32(SpvScopeWorkgroup), b_.u32(SpvScopeWorkgroup),
                                        b_.u32(SpvMemorySemanticsAcquireReleaseMask |
                                               SpvMemorySemanticsWorkgroupMemoryMask |
                                               SpvMemorySemanticsUniformMemoryMask)});
            return;
        }
        if (op == "membar") {
            const std::uint32_t scope = in.has("cta") ? SpvScopeWorkgroup : SpvScopeDevice;
            b_.op(SpvOpMemoryBarrier, {b_.u32(scope), b_.u32(SpvMemorySemanticsAcquireReleaseMask |
                                                             SpvMemorySemanticsWorkgroupMemoryMask |
                                                             SpvMemorySemanticsUniformMemoryMask)});
            return;
        }
        if (op == "cvta") {
            const Type t = last_type(in);
            write(in.operands[0], operand(in.operands[1], t, line), t, line);
            return;
        }
        if (op == "cvt") return lower_cvt(in);
        if (op == "atom" || op == "red") return lower_atomic(in);
        if (op == "setp") return lower_setp(in);

        const Type type = last_type(in);
        if (lower_float(in, type)) return;
        const std::uint32_t ty = canonical(type);
        auto src = [&](std::size_t i) { return operand(in.operands[i], type, line); };
        auto result = [&](std::uint32_t v) { write(in.operands[0], v, type, line); };

        if (op == "mov") {
            if (in.operands[1].kind == Operand::Kind::vector) {
                // mov.b64 d, {lo, hi} packs 32-bit halves
                if (type.bits != 64 || in.operands[1].elems.size() != 2) throw Error("unsupported vector mov", line);
                const std::uint32_t lo = b_.value(SpvOpUConvert, b_.t_u64(), {operand(in.operands[1].elems[0], Type{'b', 32}, line)});
                const std::uint32_t hi = b_.value(SpvOpUConvert, b_.t_u64(), {operand(in.operands[1].elems[1], Type{'b', 32}, line)});
                result(b_.value(SpvOpBitwiseOr, b_.t_u64(), {lo, b_.value(SpvOpShiftLeftLogical, b_.t_u64(), {hi, b_.u32(32)})}));
                return;
            }
            if (in.operands[0].kind == Operand::Kind::vector) {
                if (type.bits != 64 || in.operands[0].elems.size() != 2) throw Error("unsupported vector mov", line);
                const std::uint32_t v = src(1);
                write(in.operands[0].elems[0], b_.value(SpvOpUConvert, b_.t_u32(), {v}), Type{'b', 32}, line);
                const std::uint32_t hi = b_.value(SpvOpShiftRightLogical, b_.t_u64(), {v, b_.u32(32)});
                write(in.operands[0].elems[1], b_.value(SpvOpUConvert, b_.t_u32(), {hi}), Type{'b', 32}, line);
                return;
            }
            return result(src(1));
        }
        if (op == "selp") {
            const std::uint32_t c = operand(in.operands[3], Type{'p', 1}, line);
            return result(b_.value(SpvOpSelect, ty, {c, src(1), src(2)}));
        }
        if (type.kind == 'p') {
            if (op == "and" || op == "or" || op == "xor") {
                const SpvOp code = op == "and" ? SpvOpLogicalAnd : op == "or" ? SpvOpLogicalOr : SpvOpLogicalNotEqual;
                return result(b_.value(code, ty, {src(1), src(2)}));
            }
            if (op == "not") return result(b_.value(SpvOpLogicalNot, ty, {src(1)}));
            throw Error("unsupported predicate instruction " + in.text(), line);
        }
        const bool is_signed = type.kind == 's';
        if (op == "add" || op == "sub") {
            if (in.has("sat") || in.has("cc")) throw Error("unsupported " + in.text(), line);
            return result(b_.value(op == "add" ? SpvOpIAdd : SpvOpISub, ty, {src(1), src(2)}));
        }
        if (op == "mul" || op == "mad") {
            const bool wide = in.has("wide");
            const bool hi = in.has("hi");
            std::uint32_t product = 0;
            if (wide) {
                if (type.bits == 16) {
                    const std::uint32_t a = is_signed ? sign_extend(src(1), 16) : src(1);
                    const std::uint32_t c = is_signed ? sign_extend(src(2), 16) : src(2);
                    product = b_.value(SpvOpIMul, b_.t_u32(), {a, c});
                    if (op == "mad") product = b_.value(SpvOpIAdd, b_.t_u32(), {product, operand(in.operands[3], Type{type.kind, 32}, line)});
                    write(in.operands[0], product, Type{type.kind, 32}, line);
                    return;
                }
                if (type.bits != 32) throw Error("unsupported wide multiply", line);
                const std::uint32_t u64 = b_.t_u64();
                const SpvOp ext_op = is_signed ? SpvOpSConvert : SpvOpUConvert;
                product = b_.value(SpvOpIMul, u64, {b_.value(ext_op, u64, {src(1)}), b_.value(ext_op, u64, {src(2)})});
                if (op == "mad") product = b_.value(SpvOpIAdd, u64, {product, operand(in.operands[3], Type{type.kind, 64}, line)});
                write(in.operands[0], product, Type{type.kind, 64}, line);
                return;
            }
            if (hi) {
                if (type.bits != 32) throw Error("unsupported mul.hi width", line);
                const std::uint32_t pair = b_.unique_type(SpvOpTypeStruct, {ty, ty});
                const std::uint32_t r = b_.value(is_signed ? SpvOpSMulExtended : SpvOpUMulExtended, pair, {src(1), src(2)});
                product = b_.value(SpvOpCompositeExtract, ty, {r, 1});
            } else {
                product = b_.value(SpvOpIMul, ty, {src(1), src(2)});
            }
            if (op == "mad") product = b_.value(SpvOpIAdd, ty, {product, src(3)});
            return result(product);
        }
        if (op == "div" || op == "rem") {
            const std::uint32_t a = is_signed ? sign_extend(src(1), type.bits) : src(1);
            const std::uint32_t c = is_signed ? sign_extend(src(2), type.bits) : src(2);
            const SpvOp code = op == "div" ? (is_signed ? SpvOpSDiv : SpvOpUDiv) : (is_signed ? SpvOpSRem : SpvOpUMod);
            return result(b_.value(code, ty, {a, c}));
        }
        if (op == "min" || op == "max") {
            const std::uint32_t a = is_signed ? sign_extend(src(1), type.bits) : src(1);
            const std::uint32_t c = is_signed ? sign_extend(src(2), type.bits) : src(2);
            const GLSLstd450 code = op == "min" ? (is_signed ? GLSLstd450SMin : GLSLstd450UMin)
                                                : (is_signed ? GLSLstd450SMax : GLSLstd450UMax);
            return result(b_.ext(ty, code, {a, c}));
        }
        if (op == "neg") return result(b_.value(SpvOpSNegate, ty, {src(1)}));
        if (op == "abs") return result(b_.ext(ty, GLSLstd450SAbs, {sign_extend(src(1), type.bits)}));
        if (op == "and" || op == "or" || op == "xor") {
            const SpvOp code = op == "and" ? SpvOpBitwiseAnd : op == "or" ? SpvOpBitwiseOr : SpvOpBitwiseXor;
            return result(b_.value(code, ty, {src(1), src(2)}));
        }
        if (op == "not") return result(b_.value(SpvOpNot, ty, {src(1)}));
        if (op == "shl") return result(shift(in, type, true));
        if (op == "shr") return result(shift(in, type, false));
        if (op == "bfe") return result(lower_bfe(in, type));
        if (op == "clz") {
            const std::uint32_t v = src(1);
            const std::uint32_t u32 = b_.t_u32();
            if (type.bits == 64) {
                const std::uint32_t lo = b_.value(SpvOpUConvert, u32, {v});
                const std::uint32_t hi = b_.value(SpvOpUConvert, u32, {b_.value(SpvOpShiftRightLogical, b_.t_u64(), {v, b_.u32(32)})});
                const std::uint32_t hi_nonzero = b_.value(SpvOpINotEqual, b_.t_bool(), {hi, b_.u32(0)});
                // FindUMsb(0) is -1, so 31 - msb and 63 - msb give 32 and 64 for zero.
                const std::uint32_t hc = b_.value(SpvOpISub, u32, {b_.u32(31), b_.ext(u32, GLSLstd450FindUMsb, {hi})});
                const std::uint32_t lc = b_.value(SpvOpISub, u32, {b_.u32(63), b_.ext(u32, GLSLstd450FindUMsb, {lo})});
                write(in.operands[0], b_.value(SpvOpSelect, u32, {hi_nonzero, hc, lc}), Type{'u', 32}, line);
                return;
            }
            write(in.operands[0], b_.value(SpvOpISub, u32, {b_.u32(31), b_.ext(u32, GLSLstd450FindUMsb, {v})}), Type{'u', 32}, line);
            return;
        }
        if (op == "popc" || op == "brev") {
            // Vulkan only allows 32-bit operands: split 64-bit values.
            const std::uint32_t u32 = b_.t_u32(), u64 = b_.t_u64();
            const std::uint32_t v = src(1);
            if (type.bits == 64) {
                const std::uint32_t lo = b_.value(SpvOpUConvert, u32, {v});
                const std::uint32_t hi = b_.value(SpvOpUConvert, u32, {b_.value(SpvOpShiftRightLogical, u64, {v, b_.u32(32)})});
                if (op == "popc") {
                    const std::uint32_t count = b_.value(SpvOpIAdd, u32, {b_.value(SpvOpBitCount, u32, {lo}),
                                                                           b_.value(SpvOpBitCount, u32, {hi})});
                    write(in.operands[0], count, Type{'u', 32}, line);
                    return;
                }
                const std::uint32_t new_hi = b_.value(SpvOpUConvert, u64, {b_.value(SpvOpBitReverse, u32, {lo})});
                const std::uint32_t new_lo = b_.value(SpvOpUConvert, u64, {b_.value(SpvOpBitReverse, u32, {hi})});
                return result(b_.value(SpvOpBitwiseOr, u64,
                                       {b_.value(SpvOpShiftLeftLogical, u64, {new_hi, b_.u32(32)}), new_lo}));
            }
            if (op == "popc") {
                write(in.operands[0], b_.value(SpvOpBitCount, u32, {v}), Type{'u', 32}, line);
                return;
            }
            return result(b_.value(SpvOpBitReverse, ty, {v}));
        }
        throw Error("unsupported instruction " + in.text(), line);
    }

    // atom/red.add.f32: a compare-and-swap loop on the word's bits. It rounds
    // to nearest; on global memory it also flushes subnormal inputs and
    // results to sign-preserving zero, while on shared memory NVIDIA keeps
    // them (both verified with pxvk-opcheck). Vulkan's float atomics promise
    // neither, so the add is done here. PhysX 2.8.4 accumulates cloth normals
    // this way; the result depends on the order threads arrive, as natively.
    // On devices with native flush-to-zero the shared form flushes too.
    void lower_float_atomic_add(const Instruction &in, Space space, std::uint32_t pointer, std::size_t value_index)
    {
        const std::uint32_t u32 = b_.t_u32(), f32 = b_.t_f32();
        const std::uint32_t scope = b_.u32(space == Space::shared ? SpvScopeWorkgroup : SpvScopeDevice);
        const std::uint32_t semantics = b_.u32(SpvMemorySemanticsMaskNone);
        const bool ftz = space == Space::global;
        const auto flush = [&](std::uint32_t v) { return ftz ? this->flush(v) : v; };
        const std::uint32_t addend = flush(operand(in.operands[value_index], Type{'f', 32}, in.line));
        const std::uint32_t header = b_.id(), body = b_.id(), cont = b_.id(), merge = b_.id();
        b_.branch(header);
        b_.label(header);
        b_.op(SpvOpLoopMerge, {merge, cont, SpvLoopControlMaskNone});
        b_.branch(body);
        b_.label(body);
        const std::uint32_t old = b_.value(SpvOpAtomicLoad, u32, {pointer, scope, semantics});
        const std::uint32_t old_f = flush(b_.value(SpvOpBitcast, f32, {old}));
        const std::uint32_t sum = b_.value(SpvOpFAdd, f32, {old_f, addend});
        b_.decorate(sum, SpvDecorationNoContraction);
        const std::uint32_t replaced = b_.value(SpvOpBitcast, u32, {flush(sum)});
        const std::uint32_t seen =
            b_.value(SpvOpAtomicCompareExchange, u32, {pointer, scope, semantics, semantics, replaced, old});
        b_.branch_conditional(b_.value(SpvOpIEqual, b_.t_bool(), {seen, old}), merge, cont);
        b_.label(cont);
        b_.branch(header);
        b_.label(merge);
        if (in.op != "red") write(in.operands[0], b_.value(SpvOpBitcast, f32, {old}), Type{'f', 32}, in.line);
    }

    // atom/red on 32-bit integers in global or shared memory, and add.f32.
    void lower_atomic(const Instruction &in)
    {
        const Type type = last_type(in);
        if (type.kind == 'f' && type.bits == 32 && in.has("add")) {
            const bool is_red = in.op == "red";
            const std::size_t addr_index = is_red ? 0 : 1;
            Space space = memory_space(in);
            if (space == Space::none) space = resolve_generic(in.operands[addr_index], in.line);
            if (space != Space::global && space != Space::shared)
                throw Error("atomics are only supported on global and shared memory", in.line);
            const std::uint32_t pointer = word_pointer(space, address(in.operands[addr_index], in.line));
            lower_float_atomic_add(in, space, pointer, addr_index + 1);
            return;
        }
        if (type.bits != 32 || type.kind == 'f')
            throw Error("only 32-bit integer atomics and add.f32 are supported", in.line);
        const bool is_red = in.op == "red";
        const std::size_t addr_index = is_red ? 0 : 1;
        Space space = memory_space(in);
        if (space == Space::none) space = resolve_generic(in.operands[addr_index], in.line);
        if (space != Space::global && space != Space::shared)
            throw Error("atomics are only supported on global and shared memory", in.line);
        const std::uint32_t pointer = word_pointer(space, address(in.operands[addr_index], in.line));
        const std::uint32_t u32 = b_.t_u32();
        const std::uint32_t scope = b_.u32(space == Space::shared ? SpvScopeWorkgroup : SpvScopeDevice);
        const std::uint32_t semantics = b_.u32(SpvMemorySemanticsMaskNone);
        const std::uint32_t a = operand(in.operands[addr_index + 1], type, in.line);
        const bool is_signed = type.kind == 's';
        std::uint32_t result = 0;
        if (in.has("cas")) {
            const std::uint32_t c = operand(in.operands[addr_index + 2], type, in.line);
            result = b_.value(SpvOpAtomicCompareExchange, u32, {pointer, scope, semantics, semantics, c, a});
        } else {
            SpvOp code = SpvOpNop;
            if (in.has("add")) code = SpvOpAtomicIAdd;
            else if (in.has("exch")) code = SpvOpAtomicExchange;
            else if (in.has("and")) code = SpvOpAtomicAnd;
            else if (in.has("or")) code = SpvOpAtomicOr;
            else if (in.has("xor")) code = SpvOpAtomicXor;
            else if (in.has("min")) code = is_signed ? SpvOpAtomicSMin : SpvOpAtomicUMin;
            else if (in.has("max")) code = is_signed ? SpvOpAtomicSMax : SpvOpAtomicUMax;
            else throw Error("unsupported atomic operation " + in.text(), in.line);
            result = b_.value(code, u32, {pointer, scope, semantics, a});
        }
        if (!is_red) write(in.operands[0], result, type, in.line);
    }

    std::uint32_t lower_bfe(const Instruction &in, Type type)
    {
        if (type.bits != 32) throw Error("only 32-bit bfe is supported", in.line);
        const std::uint32_t u32 = b_.t_u32();
        const std::uint32_t a = operand(in.operands[1], type, in.line);
        const std::uint32_t pos = b_.value(SpvOpBitwiseAnd, u32, {operand(in.operands[2], Type{'u', 32}, in.line), b_.u32(0xff)});
        const std::uint32_t len = b_.value(SpvOpBitwiseAnd, u32, {operand(in.operands[3], Type{'u', 32}, in.line), b_.u32(0xff)});
        const std::uint32_t boolean = b_.t_bool();
        // PTX: d = (a >> min(pos,31)) sign/zero-extended from bit len-1;
        // len == 0 gives 0 (unsigned) / 0 (signed); pos >= 32 gives 0 or the
        // sign fill.
        const std::uint32_t pos_c = b_.ext(u32, GLSLstd450UMin, {pos, b_.u32(31)});
        const std::uint32_t shifted = b_.value(type.kind == 's' ? SpvOpShiftRightArithmetic : SpvOpShiftRightLogical, u32, {a, pos_c});
        const std::uint32_t len_c = b_.ext(u32, GLSLstd450UMin, {len, b_.u32(32)});
        const std::uint32_t full = b_.value(SpvOpUGreaterThanEqual, boolean, {len_c, b_.u32(32)});
        const std::uint32_t mask = b_.value(SpvOpSelect, u32, {full, b_.u32(0xffffffffu),
            b_.value(SpvOpISub, u32, {b_.value(SpvOpShiftLeftLogical, u32, {b_.u32(1), b_.ext(u32, GLSLstd450UMin, {len_c, b_.u32(31)})}), b_.u32(1)})});
        std::uint32_t r = b_.value(SpvOpBitwiseAnd, u32, {shifted, mask});
        if (type.kind == 's') {
            // sign-extend from bit (len-1) when 0 < len < 32
            const std::uint32_t sh = b_.value(SpvOpISub, u32, {b_.u32(32), b_.ext(u32, GLSLstd450UMax, {len_c, b_.u32(1)})});
            const std::uint32_t ext = b_.value(SpvOpShiftRightArithmetic, u32, {b_.value(SpvOpShiftLeftLogical, u32, {r, sh}), sh});
            r = b_.value(SpvOpSelect, u32, {full, r, ext});
        } else {
            const std::uint32_t pos_out = b_.value(SpvOpUGreaterThanEqual, boolean, {pos, b_.u32(32)});
            r = b_.value(SpvOpSelect, u32, {pos_out, b_.u32(0), r});
        }
        const std::uint32_t zero_len = b_.value(SpvOpIEqual, boolean, {len_c, b_.u32(0)});
        return b_.value(SpvOpSelect, u32, {zero_len, b_.u32(0), r});
    }

    void lower_setp(const Instruction &in)
    {
        // setp.cmp[.bool].type p, a, b[, {!}c]
        const Type type = last_type(in);
        std::string cmp;
        std::string combine;
        for (const auto &m : in.mods) {
            if (m == "ftz" || is_type_name(m)) continue;
            if (m == "and" || m == "or" || m == "xor") combine = m;
            else if (cmp.empty()) cmp = m;
        }
        std::uint32_t a = type.kind == 'f' ? fin(in, 1) : operand(in.operands[1], type, in.line);
        std::uint32_t c = type.kind == 'f' ? fin(in, 2) : operand(in.operands[2], type, in.line);
        std::uint32_t p = compare(cmp, type, a, c, in.line);
        if (!combine.empty()) {
            const std::uint32_t q = operand(in.operands[3], Type{'p', 1}, in.line);
            const SpvOp code = combine == "and" ? SpvOpLogicalAnd : combine == "or" ? SpvOpLogicalOr : SpvOpLogicalNotEqual;
            p = b_.value(code, b_.t_bool(), {p, q});
        }
        write(in.operands[0], p, Type{'p', 1}, in.line);
    }

    void lower_cvt(const Instruction &in)
    {
        const auto [dst, src] = two_types(in);
        const int line = in.line;
        std::string rounding;
        for (const auto &m : in.mods)
            if (m == "rn" || m == "rz" || m == "rm" || m == "rp" || m == "rni" || m == "rzi" || m == "rmi" || m == "rpi")
                rounding = m;
        if (in.has("sat")) throw Error("cvt.sat is unsupported", line);
        const bool ftz = in.has("ftz") && !native_ftz_;
        if (dst.kind == 'f' && src.kind == 'f') {
            if (dst.bits != 32 || src.bits != 32) throw Error("only f32 cvt is supported", line);
            std::uint32_t v = operand(in.operands[1], src, line);
            if (ftz) v = flush(v);
            const GLSLstd450 code = rounding == "rni" ? GLSLstd450RoundEven : rounding == "rzi" ? GLSLstd450Trunc
                                  : rounding == "rmi" ? GLSLstd450Floor : rounding == "rpi" ? GLSLstd450Ceil
                                  : GLSLstd450FAbs;
            if (code == GLSLstd450FAbs) throw Error("unsupported f32->f32 cvt", line);
            v = b_.ext(b_.t_f32(), code, {v});
            if (ftz) v = flush(v);
            write(in.operands[0], v, dst, line);
            return;
        }
        if (dst.kind == 'f') {
            // integer -> f32 (correctly rounded to nearest in Vulkan)
            if (rounding != "rn" && !rounding.empty()) throw Error("unsupported int->float rounding", line);
            std::uint32_t v = operand(in.operands[1], src, line);
            if (src.kind == 's' && src.bits < 32) v = sign_extend(v, src.bits);
            v = b_.value(src.kind == 's' ? SpvOpConvertSToF : SpvOpConvertUToF, b_.t_f32(), {v});
            write(in.operands[0], v, dst, line);
            return;
        }
        if (src.kind == 'f') {
            // f32 -> integer with PTX saturation: NaN -> 0, clamp to range.
            if (dst.bits != 32) throw Error("only f32->32-bit int cvt is supported", line);
            std::uint32_t v = operand(in.operands[1], src, line);
            if (ftz) v = flush(v);
            const GLSLstd450 code = rounding == "rni" ? GLSLstd450RoundEven : rounding == "rmi" ? GLSLstd450Floor
                                  : rounding == "rpi" ? GLSLstd450Ceil : GLSLstd450Trunc;
            v = b_.ext(b_.t_f32(), code, {v});
            const std::uint32_t f32 = b_.t_f32();
            const std::uint32_t boolean = b_.t_bool();
            std::uint32_t r;
            if (dst.kind == 's') {
                const std::uint32_t lo = b_.f32_bits(0xcf000000u); // -2^31
                const std::uint32_t hi = b_.f32_bits(0x4effffffu); // largest float < 2^31
                const std::uint32_t clamped = b_.ext(f32, GLSLstd450FClamp, {v, lo, hi});
                r = b_.value(SpvOpConvertFToS, b_.t_u32(), {clamped});
                const std::uint32_t over = b_.value(SpvOpFOrdGreaterThanEqual, boolean, {v, b_.f32_bits(0x4f000000u)});
                r = b_.value(SpvOpSelect, b_.t_u32(), {over, b_.u32(0x7fffffffu), r});
            } else {
                const std::uint32_t clamped = b_.ext(f32, GLSLstd450FClamp, {v, b_.f32_bits(0), b_.f32_bits(0x4f7fffffu)});
                r = b_.value(SpvOpConvertFToU, b_.t_u32(), {clamped});
                const std::uint32_t over = b_.value(SpvOpFOrdGreaterThanEqual, boolean, {v, b_.f32_bits(0x4f800000u)});
                r = b_.value(SpvOpSelect, b_.t_u32(), {over, b_.u32(0xffffffffu), r});
            }
            const std::uint32_t nan = b_.value(SpvOpIsNan, boolean, {v});
            r = b_.value(SpvOpSelect, b_.t_u32(), {nan, b_.u32(0), r});
            write(in.operands[0], r, dst, line);
            return;
        }
        // integer -> integer: sign/zero extend from the source width, then
        // truncate to the destination width (write_reg masks narrow regs).
        std::uint32_t v = operand(in.operands[1], src, line);
        if (src.bits < 64 && dst.bits == 64) {
            if (src.kind == 's') v = sign_extend(v, src.bits);
            v = b_.value(src.kind == 's' ? SpvOpSConvert : SpvOpUConvert, b_.t_u64(), {v});
        } else if (src.bits == 64 && dst.bits < 64) {
            v = b_.value(SpvOpUConvert, b_.t_u32(), {v});
            if (dst.bits < 32) v = mask_bits(v, dst.bits);
        } else if (src.bits < 32) {
            v = src.kind == 's' ? sign_extend(v, src.bits) : mask_bits(v, src.bits);
            if (dst.bits < 32) v = mask_bits(v, dst.bits);
        } else if (dst.bits < 32) {
            v = mask_bits(v, dst.bits);
        }
        write(in.operands[0], v, dst, line);
    }

    // ---- structured emission ---------------------------------------------------
    struct Construct {
        std::uint32_t header = 0, body = 0, cont = 0, merge = 0;
        bool needs_ladder = false;
        bool transparent = false; // block emitted inline, without a loop wrapper
    };

    // A block construct only needs a breakable (loop) wrapper when some
    // branch to it is not in tail position, i.e. is not the last action on
    // its path through if-arms to the end of the block's body. Branches in
    // tail position reach the block's end by simply falling through, so such
    // blocks are emitted inline. Most PTX if/else joins are like this, and
    // drivers compile plain selections much better than one-shot loops.
    void count_branches(const StructNode &node, int tail, std::vector<int> &total, std::vector<int> &tails)
    {
        switch (node.kind) {
        case StructNode::Kind::seq:
            for (std::size_t i = 0; i < node.children.size(); ++i)
                count_branches(node.children[i], i + 1 == node.children.size() ? tail : -1, total, tails);
            break;
        case StructNode::Kind::branch_if:
            for (const StructNode &arm : node.children) count_branches(arm, tail, total, tails);
            break;
        case StructNode::Kind::block:
            count_branches(node.children[0], node.construct, total, tails);
            break;
        case StructNode::Kind::loop:
            count_branches(node.children[0], -1, total, tails);
            break;
        case StructNode::Kind::br:
            ++total[node.construct];
            if (node.construct == tail) ++tails[node.construct];
            break;
        default:
            break;
        }
    }

    void find_transparent_blocks(const Structured &tree)
    {
        std::vector<int> total(constructs_.size(), 0), tails(constructs_.size(), 0);
        count_branches(tree.root, -1, total, tails);
        for (std::size_t i = 0; i < constructs_.size(); ++i)
            constructs_[i].transparent = !is_loop_[i] && total[i] == tails[i];
    }

    std::uint32_t pending()
    {
        if (pending_ == 0)
            pending_ = b_.local_variable(b_.t_ptr(SpvStorageClassFunction, b_.t_u32()), b_.u32(0));
        return pending_;
    }

    void emit(const StructNode &node)
    {
        if (!b_.open()) {
            // Code after an unconditional exit is unreachable; give it a
            // block of its own so the module stays well formed.
            if (node.kind == StructNode::Kind::seq && node.children.empty()) return;
            b_.label(b_.id());
        }
        switch (node.kind) {
        case StructNode::Kind::seq:
            for (const StructNode &child : node.children) emit(child);
            break;
        case StructNode::Kind::code:
            for (const Instruction *in : cfg_->blocks[node.cfg_block].code) lower_instruction(*in);
            break;
        case StructNode::Kind::ret:
            b_.ret();
            break;
        case StructNode::Kind::branch_if: {
            const CfgBlock &block = cfg_->blocks[node.cfg_block];
            std::uint32_t p = read_reg(block.predicate, Type{'p', 1}, block.line);
            if (block.negated) p = b_.value(SpvOpLogicalNot, b_.t_bool(), {p});
            if (options_.emulate_warps && (contains_warp_sync(node.children[0]) || contains_warp_sync(node.children[1]))) {
                emit_predicated(node, p, block.line);
                break;
            }
            const std::uint32_t then_label = b_.id(), else_label = b_.id(), merge = b_.id();
            b_.op(SpvOpSelectionMerge, {merge, SpvSelectionControlMaskNone});
            b_.branch_conditional(p, then_label, else_label);
            b_.label(then_label);
            emit(node.children[0]);
            if (b_.open()) b_.branch(merge);
            b_.label(else_label);
            emit(node.children[1]);
            if (b_.open()) b_.branch(merge);
            b_.label(merge);
            // If both arms exited, the merge block is unreachable. Following
            // code (if any) will still be emitted here; an OpUnreachable is
            // added at the end of the function when nothing follows.
            break;
        }
        case StructNode::Kind::block:
            if (constructs_[node.construct].transparent) {
                emit(node.children[0]);
                break;
            }
            [[fallthrough]];
        case StructNode::Kind::loop: {
            Construct &c = constructs_[node.construct];
            stack_.push_back(node.construct);
            b_.branch(c.header);
            b_.label(c.header);
            b_.op(SpvOpLoopMerge, {c.merge, c.cont, SpvLoopControlMaskNone});
            b_.branch(c.body);
            b_.label(c.body);
            emit(node.children[0]);
            if (b_.open()) b_.branch(c.merge); // falling off a construct exits it
            b_.label(c.cont);
            b_.branch(c.header);
            stack_.pop_back();
            b_.label(c.merge);
            if (c.needs_ladder) emit_ladder();
            break;
        }
        case StructNode::Kind::br:
            emit_br(node.construct);
            break;
        }
    }

    // ---- warp emulation uniformity --------------------------------------------
    // Emulation brackets each warp-synchronous store with workgroup barriers,
    // so every such store must be reached by all invocations of the
    // workgroup, or sit in a predicated if-arm (emit_predicated) whose `if`
    // is itself reached by all. An invocation stops being guaranteed to reach
    // the rest of a sequence once an earlier element can leave it early: a
    // return, or a branch to a construct enclosing that element (break,
    // continue, multi-level exit). A loop body counts as uniform only if the
    // loop contains bar.sync, because CUDA already requires uniform control
    // flow around bar.sync; other loops may have per-invocation trip counts.
    void collect_constructs(const StructNode &node, std::set<int> &inside)
    {
        if (node.kind == StructNode::Kind::block || node.kind == StructNode::Kind::loop) inside.insert(node.construct);
        for (const StructNode &child : node.children) collect_constructs(child, inside);
    }

    bool leaves_early(const StructNode &node, const std::set<int> &inside)
    {
        if (node.kind == StructNode::Kind::ret) return true;
        // A branch to a transparent block is in tail position (see
        // find_transparent_blocks): it skips nothing.
        if (node.kind == StructNode::Kind::br && !inside.count(node.construct) && !constructs_[node.construct].transparent)
            return true;
        for (const StructNode &child : node.children)
            if (leaves_early(child, inside)) return true;
        return false;
    }

    bool leaves_early(const StructNode &node)
    {
        std::set<int> inside;
        collect_constructs(node, inside);
        return leaves_early(node, inside);
    }

    bool contains_barrier(const StructNode &node)
    {
        if (node.kind == StructNode::Kind::code) {
            for (const Instruction *in : cfg_->blocks[node.cfg_block].code)
                if (in->op == "bar") return true;
            return false;
        }
        for (const StructNode &child : node.children)
            if (contains_barrier(child)) return true;
        return false;
    }

    void check_warp_uniformity(const StructNode &node, bool uniform)
    {
        const auto reject = [] {
            throw Error("warp-synchronous code that not every invocation reaches cannot be emulated", 0);
        };
        switch (node.kind) {
        case StructNode::Kind::seq:
            for (const StructNode &child : node.children) {
                check_warp_uniformity(child, uniform);
                if (leaves_early(child)) uniform = false;
            }
            return;
        case StructNode::Kind::code:
            if (!uniform && contains_warp_sync(node)) reject();
            return;
        case StructNode::Kind::branch_if:
            if (contains_warp_sync(node)) {
                // Emitted predicated; its arms must be straight-line code
                // (checked by emit_straight) and the `if` itself uniform.
                if (!uniform) reject();
                return;
            }
            for (const StructNode &child : node.children) check_warp_uniformity(child, false);
            return;
        case StructNode::Kind::block:
            check_warp_uniformity(node.children[0], uniform);
            return;
        case StructNode::Kind::loop:
            check_warp_uniformity(node.children[0], uniform && contains_barrier(node.children[0]));
            return;
        default:
            return;
        }
    }

    bool contains_warp_sync(const StructNode &node)
    {
        if (node.kind == StructNode::Kind::code) {
            for (const Instruction *in : cfg_->blocks[node.cfg_block].code)
                if (is_warp_sync_store(*in)) return true;
            return false;
        }
        for (const StructNode &child : node.children)
            if (contains_warp_sync(child)) return true;
        return false;
    }

    // emulate_warps: an `if` whose arms contain warp-synchronous code (e.g.
    // `if (tid < 32) { warp scan }`) cannot branch, because its workgroup
    // barriers must be reached by every invocation. Both arms are emitted
    // for all invocations instead, each instruction predicated on its arm's
    // condition (lower_instruction), with the barriers unconditional. This
    // needs straight-line arms; anything else is rejected rather than
    // miscompiled.
    void emit_predicated(const StructNode &node, std::uint32_t p, int line)
    {
        const std::uint32_t outer = predicate_;
        const std::uint32_t not_p = b_.value(SpvOpLogicalNot, b_.t_bool(), {p});
        for (int arm = 0; arm < 2; ++arm) {
            const std::uint32_t own = arm == 0 ? p : not_p;
            predicate_ = outer ? b_.value(SpvOpLogicalAnd, b_.t_bool(), {outer, own}) : own;
            emit_straight(node.children[static_cast<std::size_t>(arm)], line);
        }
        predicate_ = outer;
    }

    void emit_straight(const StructNode &node, int line)
    {
        switch (node.kind) {
        case StructNode::Kind::seq:
            for (const StructNode &child : node.children) emit_straight(child, line);
            return;
        case StructNode::Kind::code:
            for (const Instruction *in : cfg_->blocks[node.cfg_block].code) lower_instruction(*in);
            return;
        case StructNode::Kind::br:
            // A tail branch to a transparent block falls through anyway.
            if (constructs_[node.construct].transparent) return;
            break;
        default:
            break;
        }
        throw Error("warp-synchronous code under a branch with nested control flow cannot be emulated", line);
    }

    std::uint32_t construct_target(int id) const
    {
        return is_loop_[id] ? constructs_[id].cont : constructs_[id].merge;
    }

    void emit_br(int target)
    {
        // A tail branch to a transparent block: falling through reaches it.
        if (constructs_[target].transparent) return;
        const int innermost = stack_.back();
        if (target == innermost) {
            b_.branch(construct_target(target));
            return;
        }
        // Multi-level exit: record the destination and break out; every
        // construct passed on the way re-dispatches at its merge block.
        for (auto it = stack_.rbegin(); it != stack_.rend() && *it != target; ++it)
            constructs_[*it].needs_ladder = true;
        b_.op(SpvOpStore, {pending(), b_.u32(static_cast<std::uint32_t>(target) + 1u)});
        b_.branch(constructs_[innermost].merge);
    }

    // Emitted at the merge block of a construct that a multi-level branch
    // passed through; we are now directly inside the enclosing construct.
    void emit_ladder()
    {
        const int enclosing = stack_.back();
        const std::uint32_t u32 = b_.t_u32(), boolean = b_.t_bool();
        const std::uint32_t value = b_.value(SpvOpLoad, u32, {pending()});
        const std::uint32_t active = b_.value(SpvOpINotEqual, boolean, {value, b_.u32(0)});
        const std::uint32_t here = b_.value(SpvOpIEqual, boolean,
            {value, b_.u32(static_cast<std::uint32_t>(enclosing) + 1u)});
        const std::uint32_t go = b_.id(), rest = b_.id();
        b_.op(SpvOpSelectionMerge, {rest, SpvSelectionControlMaskNone});
        b_.branch_conditional(active, go, rest);
        b_.label(go);
        b_.op(SpvOpStore, {pending(), b_.value(SpvOpSelect, u32, {here, b_.u32(0), value})});
        Construct &e = constructs_[enclosing];
        if (is_loop_[enclosing]) {
            b_.branch_conditional(here, e.cont, e.merge);
        } else {
            b_.branch(e.merge);
        }
        // Targets further out than `enclosing` were marked by emit_br, so the
        // ladder at its merge block continues the dispatch.
        b_.label(rest);
    }

    void finish_interface()
    {
        if (b_.open()) b_.ret();
        // The specialized WorkgroupSize built-in (spec IDs 0-2) overrides
        // LocalSize; it must exist even when the kernel never reads %ntid.
        workgroup_size_component(0);
        b_.execution_mode(main_, SpvExecutionModeLocalSize, {1, 1, 1});
        if (native_ftz_) {
            b_.extension("SPV_KHR_float_controls");
            b_.capability(SpvCapabilityDenormFlushToZero);
            b_.execution_mode(main_, SpvExecutionModeDenormFlushToZero, {32});
        }
        if (options_.preserve_nan) {
            b_.extension("SPV_KHR_float_controls");
            b_.capability(SpvCapabilitySignedZeroInfNanPreserve);
            b_.execution_mode(main_, SpvExecutionModeSignedZeroInfNanPreserve, {32});
        }
        if (info_.uses_int64) {
            if (!options_.int64) throw Error("kernel uses 64-bit integers, which the device does not support", 0);
            b_.capability(SpvCapabilityInt64);
        }
        if (info_.static_shared_bytes != 0 || info_.uses_dynamic_shared) shared_array();
    }

    const Module &m_;
    const Function &f_;
    CompileOptions options_;
    Builder b_;
    KernelInfo info_;
    std::uint32_t main_ = 0;
    bool native_ftz_ = false;
    std::uint32_t predicate_ = 0; // emulate_warps: condition of the arm being emitted predicated
    std::set<std::string> flushed_; // registers that never hold a denormal (emulated ftz)
    std::map<std::string, ParamSlot> params_;
    std::map<std::string, std::uint32_t> symbol_spec_, texture_index_, shared_offset_, local_offset_;
    std::map<std::string, std::uint32_t> symbol_constants_;
    std::map<std::string, unsigned> provenance_;
    std::map<const Instruction *, Fusion> fused_;
    std::map<const Instruction *, std::pair<std::uint32_t, std::uint32_t>> stash_;
    std::map<std::string, Register> registers_;
    std::map<std::uint32_t, std::uint32_t> builtins_;
    std::uint32_t workgroup_size_[3] = {0, 0, 0};
    std::uint32_t heap_[3] = {0, 0, 0};
    std::uint32_t heap_element_[3] = {0, 0, 0};
    std::uint32_t heap_base_ = 0, launch_ = 0, shared_ = 0, local_ = 0, pending_ = 0, scratch_base_ = 0;
    const Cfg *cfg_ = nullptr;
    std::vector<bool> is_loop_;
    std::vector<Construct> constructs_;
    std::vector<int> stack_;
};

} // namespace

CompiledKernel compile_kernel(const Module &module, const std::string &entry,
                              const CompileOptions &options)
{
    if (module.address_size != 32) throw Error("only .address_size 32 is supported", 1);
    const Function *function = module.find(entry);
    if (function == nullptr || !function->is_entry || !function->has_body)
        throw Error("no kernel entry named " + entry, 1);
    return KernelCompiler(module, *function, options).run();
}

} // namespace pxvk::ptx
