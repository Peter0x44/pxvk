// Control-flow graph construction and structurization for PTX functions.
//
// Structurization follows Ramsey, "Beyond Relooper" (ICFP 2022): walk the
// dominator tree; each loop header becomes a Loop construct and each merge
// node (two or more forward predecessors) becomes the continuation of a Block
// construct that encloses its dominator's code. Every branch is then either a
// structured exit of an enclosing construct or an inlined subtree. The result
// maps directly onto SPIR-V structured control flow.
//
// One departure from the paper: code a loop header dominates but that lies
// outside the loop (where the loop's exits lead) is placed after the Loop
// construct, not inside its body. Both run the same, but SPIR-V reconverges
// invocations at a construct's merge, so code nested in the body runs once
// per exit iteration with only the invocations that left in that iteration;
// after the loop it runs with everyone, as warp-synchronous PTX expects
// (VK_KHR_shader_maximal_reconvergence makes this guaranteed). A branch to the
// code directly after the loop is a loop exit (StructNode::Kind::exit).
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
    // br: to the end of a block construct, or to a loop's header (continue);
    // exit: out of a loop construct, to the code that follows it.
    enum class Kind { seq, code, block, loop, branch_if, br, exit, ret } kind = Kind::seq;
    std::vector<StructNode> children; // seq items; block/loop: [body]; if: [then, else]
    int cfg_block = -1;  // code: block to emit; if: block whose predicate decides
    int construct = -1;  // block/loop: construct id; br, exit: target construct
};

struct Structured {
    StructNode root;
    std::vector<bool> construct_is_loop; // indexed by construct id
};

Structured structurize(const Cfg &cfg);

} // namespace pxvk::ptx
