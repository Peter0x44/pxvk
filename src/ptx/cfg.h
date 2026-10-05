// Control-flow graph construction and structurization for PTX functions.
//
// Structurization follows Ramsey, "Beyond Relooper" (ICFP 2022): walk the
// dominator tree; each loop header becomes a Loop construct and each merge
// node (two or more forward predecessors) becomes the continuation of a Block
// construct that encloses its dominator's code. Every branch is then either a
// structured exit of an enclosing construct or an inlined subtree. The result
// maps directly onto SPIR-V structured control flow.
#pragma once

#include "ptx.h"

#include <memory>
#include <string>
#include <vector>

namespace pxvk::ptx {

struct CfgBlock {
    std::vector<const Instruction *> code;
    enum class Term { jump, branch, ret } term = Term::ret;
    int target = -1;    // jump/branch taken target
    int fallthrough = -1; // branch not-taken target
    std::string predicate;
    bool negated = false;
    int line = 0;
};

struct Cfg {
    std::vector<CfgBlock> blocks; // block 0 is the entry
};

Cfg build_cfg(const Function &function);

struct StructNode {
    enum class Kind { seq, code, block, loop, branch_if, br, ret } kind = Kind::seq;
    std::vector<StructNode> children; // seq items; block/loop: [body]; if: [then, else]
    int cfg_block = -1;  // code: block to emit; if: block whose predicate decides
    int construct = -1;  // block/loop: construct id; br: target construct
};

struct Structured {
    StructNode root;
    std::vector<bool> construct_is_loop; // indexed by construct id
};

Structured structurize(const Cfg &cfg);

} // namespace pxvk::ptx
