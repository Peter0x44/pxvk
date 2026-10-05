#include "cfg.h"

#include <algorithm>
#include <map>

namespace pxvk::ptx {

Cfg build_cfg(const Function &function)
{
    Cfg cfg;
    std::map<std::string, int> labels;
    std::vector<std::string> jump_labels; // per block, unresolved target label
    cfg.blocks.emplace_back();
    jump_labels.emplace_back();
    bool current_has_code = false;
    // A block is "closed" once it received a terminator; the next statement
    // opens a new block (possibly unreachable).
    bool closed = false;

    auto open_block = [&](bool fall_from_previous) {
        const int previous = static_cast<int>(cfg.blocks.size()) - 1;
        cfg.blocks.emplace_back();
        jump_labels.emplace_back();
        if (fall_from_previous) {
            cfg.blocks[previous].term = CfgBlock::Term::jump;
            cfg.blocks[previous].target = previous + 1;
        }
        current_has_code = false;
        closed = false;
    };

    for (const Statement &s : function.body) {
        if (s.is_label) {
            if (closed) open_block(false);
            else if (current_has_code) open_block(true);
            labels[s.label] = static_cast<int>(cfg.blocks.size()) - 1;
            continue;
        }
        const Instruction &in = s.inst;
        if (closed) open_block(false);
        CfgBlock &block = cfg.blocks.back();
        const bool guarded = !in.guard.empty();
        if (in.op == "bra" || in.op == "ret" || in.op == "exit") {
            block.line = in.line;
            if (in.op == "bra") {
                if (in.operands.size() != 1 || in.operands[0].kind != Operand::Kind::symbol)
                    throw Error("unsupported branch form", in.line);
                jump_labels.back() = in.operands[0].name;
            }
            if (guarded) {
                block.term = CfgBlock::Term::branch;
                block.predicate = in.guard;
                block.negated = in.guard_negated;
                // Not-taken edge falls through to the next block.
                const int self = static_cast<int>(cfg.blocks.size()) - 1;
                block.fallthrough = self + 1;
                if (in.op != "bra") block.target = -2; // resolved to a return block
                open_block(false);
            } else {
                block.term = in.op == "bra" ? CfgBlock::Term::jump : CfgBlock::Term::ret;
                closed = true;
            }
            continue;
        }
        block.code.push_back(&in);
        current_has_code = true;
    }
    // The last block falls off the end of the function: return.
    if (!closed) cfg.blocks.back().term = CfgBlock::Term::ret;

    int return_block = -1;
    for (std::size_t i = 0; i < cfg.blocks.size(); ++i) {
        CfgBlock &b = cfg.blocks[i];
        if (!jump_labels[i].empty()) {
            const auto found = labels.find(jump_labels[i]);
            if (found == labels.end())
                throw Error("branch to unknown label " + jump_labels[i], b.line);
            b.target = found->second;
        }
        if (b.target == -2) {
            if (return_block < 0) {
                return_block = static_cast<int>(cfg.blocks.size());
                cfg.blocks.emplace_back(); // term = ret by default
            }
            cfg.blocks[i].target = return_block;
        }
    }
    return cfg;
}

namespace {

std::vector<int> successors(const CfgBlock &b)
{
    switch (b.term) {
    case CfgBlock::Term::jump: return {b.target};
    case CfgBlock::Term::branch: return {b.target, b.fallthrough};
    default: return {};
    }
}

class Structurizer {
public:
    explicit Structurizer(const Cfg &cfg) : cfg_(cfg) {}

    Structured run()
    {
        const int n = static_cast<int>(cfg_.blocks.size());
        rpo_index_.assign(n, -1);
        order_.clear();
        // Iterative DFS for postorder.
        std::vector<bool> seen(n, false);
        std::vector<std::pair<int, std::size_t>> stack{{0, 0}};
        seen[0] = true;
        std::vector<int> post;
        while (!stack.empty()) {
            auto &[node, child] = stack.back();
            const auto succ = successors(cfg_.blocks[node]);
            if (child < succ.size()) {
                const int s = succ[child++];
                if (s >= 0 && s < n && !seen[s]) {
                    seen[s] = true;
                    stack.emplace_back(s, 0);
                }
            } else {
                post.push_back(node);
                stack.pop_back();
            }
        }
        order_.assign(post.rbegin(), post.rend());
        for (std::size_t i = 0; i < order_.size(); ++i) rpo_index_[order_[i]] = static_cast<int>(i);

        preds_.assign(n, {});
        for (int b : order_)
            for (int s : successors(cfg_.blocks[b])) preds_[s].push_back(b);

        compute_dominators();

        loop_header_.assign(n, false);
        forward_preds_.assign(n, 0);
        for (int b : order_) {
            for (int s : successors(cfg_.blocks[b])) {
                if (rpo_index_[s] <= rpo_index_[b]) {
                    if (!dominates(s, b))
                        throw Error("irreducible control flow is unsupported",
                                    cfg_.blocks[b].line);
                    loop_header_[s] = true;
                } else {
                    ++forward_preds_[s];
                }
            }
        }
        children_.assign(n, {});
        for (int b : order_)
            if (b != 0) children_[idom_[b]].push_back(b);

        Structured out;
        out.root = do_tree(0, {});
        out.construct_is_loop = construct_is_loop_;
        return out;
    }

private:
    struct Context {
        enum class Kind { block_followed_by, loop_headed_by, branch_if } kind;
        int label;
        int construct;
    };

    bool dominates(int a, int b) const
    {
        while (b != a && b != 0) b = idom_[b];
        return b == a;
    }

    int intersect(int a, int b) const
    {
        while (a != b) {
            while (rpo_index_[a] > rpo_index_[b]) a = idom_[a];
            while (rpo_index_[b] > rpo_index_[a]) b = idom_[b];
        }
        return a;
    }

    void compute_dominators()
    {
        const int n = static_cast<int>(cfg_.blocks.size());
        idom_.assign(n, -1);
        idom_[0] = 0;
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t i = 1; i < order_.size(); ++i) {
                const int b = order_[i];
                int new_idom = -1;
                for (int p : preds_[b]) {
                    if (idom_[p] < 0) continue;
                    new_idom = new_idom < 0 ? p : intersect(p, new_idom);
                }
                if (new_idom != idom_[b]) {
                    idom_[b] = new_idom;
                    changed = true;
                }
            }
        }
    }

    bool is_merge(int b) const { return forward_preds_[b] >= 2; }

    StructNode do_tree(int x, std::vector<Context> context)
    {
        std::vector<int> merges;
        for (int c : children_[x])
            if (is_merge(c)) merges.push_back(c);
        std::sort(merges.begin(), merges.end(),
                  [&](int a, int b) { return rpo_index_[a] > rpo_index_[b]; });
        if (loop_header_[x]) {
            const int construct = new_construct(true);
            context.push_back({Context::Kind::loop_headed_by, x, construct});
            StructNode loop;
            loop.kind = StructNode::Kind::loop;
            loop.construct = construct;
            loop.children.push_back(node_within(x, merges, 0, context));
            return loop;
        }
        return node_within(x, merges, 0, context);
    }

    StructNode node_within(int x, const std::vector<int> &merges, std::size_t i,
                           std::vector<Context> context)
    {
        if (i == merges.size()) {
            StructNode seq;
            StructNode code;
            code.kind = StructNode::Kind::code;
            code.cfg_block = x;
            seq.children.push_back(code);
            const CfgBlock &b = cfg_.blocks[x];
            switch (b.term) {
            case CfgBlock::Term::ret: {
                StructNode r;
                r.kind = StructNode::Kind::ret;
                seq.children.push_back(r);
                break;
            }
            case CfgBlock::Term::jump:
                seq.children.push_back(do_branch(x, b.target, context));
                break;
            case CfgBlock::Term::branch: {
                StructNode node;
                node.kind = StructNode::Kind::branch_if;
                node.cfg_block = x;
                auto inner = context;
                inner.push_back({Context::Kind::branch_if, -1, -1});
                node.children.push_back(do_branch(x, b.target, inner));
                node.children.push_back(do_branch(x, b.fallthrough, inner));
                seq.children.push_back(std::move(node));
                break;
            }
            }
            return seq;
        }
        const int y = merges[i];
        const int construct = new_construct(false);
        auto inner = context;
        inner.push_back({Context::Kind::block_followed_by, y, construct});
        StructNode block;
        block.kind = StructNode::Kind::block;
        block.construct = construct;
        block.children.push_back(node_within(x, merges, i + 1, inner));
        StructNode seq;
        seq.children.push_back(std::move(block));
        seq.children.push_back(do_tree(y, context));
        return seq;
    }

    StructNode do_branch(int source, int target, const std::vector<Context> &context)
    {
        const bool backward = rpo_index_[target] <= rpo_index_[source];
        if (backward || is_merge(target)) {
            const auto kind =
                backward ? Context::Kind::loop_headed_by : Context::Kind::block_followed_by;
            for (auto it = context.rbegin(); it != context.rend(); ++it) {
                if (it->kind == kind && it->label == target) {
                    StructNode br;
                    br.kind = StructNode::Kind::br;
                    br.construct = it->construct;
                    return br;
                }
            }
            throw Error("structurizer could not resolve a branch target", cfg_.blocks[source].line);
        }
        return do_tree(target, context);
    }

    int new_construct(bool loop)
    {
        construct_is_loop_.push_back(loop);
        return static_cast<int>(construct_is_loop_.size()) - 1;
    }

    const Cfg &cfg_;
    std::vector<int> order_, rpo_index_, idom_, forward_preds_;
    std::vector<std::vector<int>> preds_, children_;
    std::vector<bool> loop_header_;
    std::vector<bool> construct_is_loop_;
};

} // namespace

Structured structurize(const Cfg &cfg)
{
    return Structurizer(cfg).run();
}

} // namespace pxvk::ptx
