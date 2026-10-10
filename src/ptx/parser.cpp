#include "ptx.h"

#include <cctype>
#include <cstring>

namespace pxvk::ptx {

bool parse_type_name(std::string_view name, Type &out)
{
    if (name == "pred") {
        out = {'p', 1};
        return true;
    }
    if (name == "texref") {
        out = {'t', 64};
        return true;
    }
    if (name.size() < 2) return false;
    const char kind = name[0];
    if (kind != 'b' && kind != 'u' && kind != 's' && kind != 'f') return false;
    int bits = 0;
    for (char c : name.substr(1)) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
        bits = bits * 10 + (c - '0');
    }
    if (bits != 8 && bits != 16 && bits != 32 && bits != 64) return false;
    if (kind == 'f' && bits < 16) return false;
    out = {kind, bits};
    return true;
}

std::uint64_t Variable::size_bytes() const
{
    std::uint64_t count = 1;
    for (std::uint64_t d : dims) count *= d;
    return count * element_bytes();
}

std::string Instruction::text() const
{
    std::string out = op;
    for (const auto &m : mods) out += "." + m;
    return out;
}

const Function *Module::find(std::string_view name) const
{
    for (const auto &f : functions)
        if (f.name == name) return &f;
    return nullptr;
}

namespace {

struct Token {
    enum class Kind { ident, directive, number, string, punct, end };
    Kind kind = Kind::end;
    std::string text;
    int line = 0;
};

class Lexer {
public:
    explicit Lexer(std::string_view src) : src_(src) {}

    std::vector<Token> run()
    {
        std::vector<Token> out;
        for (;;) {
            Token t = next();
            const bool end = t.kind == Token::Kind::end;
            out.push_back(std::move(t));
            if (end) break;
        }
        return out;
    }

private:
    static bool ident_start(char c)
    {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$' || c == '%';
    }
    static bool ident_char(char c)
    {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
    }

    Token next()
    {
        skip_space();
        Token t;
        t.line = line_;
        if (pos_ >= src_.size()) return t;
        const char c = src_[pos_];
        if (c == '"') {
            const std::size_t start = ++pos_;
            while (pos_ < src_.size() && src_[pos_] != '"') {
                if (src_[pos_] == '\n') ++line_;
                ++pos_;
            }
            t.kind = Token::Kind::string;
            t.text = std::string(src_.substr(start, pos_ - start));
            if (pos_ < src_.size()) ++pos_;
            return t;
        }
        if (c == '.' && pos_ + 1 < src_.size() &&
            (std::isalnum(static_cast<unsigned char>(src_[pos_ + 1])) || src_[pos_ + 1] == '_')) {
            const std::size_t start = ++pos_;
            while (pos_ < src_.size() && ident_char(src_[pos_])) ++pos_;
            t.kind = Token::Kind::directive;
            t.text = std::string(src_.substr(start, pos_ - start));
            return t;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            const std::size_t start = pos_;
            if (c == '0' && pos_ + 1 < src_.size() &&
                std::strchr("xXfFdD", src_[pos_ + 1]) != nullptr) {
                pos_ += 2;
                while (pos_ < src_.size() && std::isxdigit(static_cast<unsigned char>(src_[pos_])))
                    ++pos_;
            } else {
                while (pos_ < src_.size() &&
                       (std::isdigit(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '.'))
                    ++pos_;
                // decimal exponent: 1.5e3, 2E-4
                std::size_t e = pos_;
                if (e < src_.size() && (src_[e] == 'e' || src_[e] == 'E')) {
                    ++e;
                    if (e < src_.size() && (src_[e] == '+' || src_[e] == '-')) ++e;
                    if (e < src_.size() && std::isdigit(static_cast<unsigned char>(src_[e]))) {
                        while (e < src_.size() && std::isdigit(static_cast<unsigned char>(src_[e]))) ++e;
                        pos_ = e;
                    }
                }
            }
            if (pos_ < src_.size() && (src_[pos_] == 'U' || src_[pos_] == 'u')) ++pos_;
            t.kind = Token::Kind::number;
            t.text = std::string(src_.substr(start, pos_ - start));
            return t;
        }
        if (ident_start(c)) {
            const std::size_t start = pos_++;
            while (pos_ < src_.size() && ident_char(src_[pos_])) ++pos_;
            t.kind = Token::Kind::ident;
            t.text = std::string(src_.substr(start, pos_ - start));
            return t;
        }
        ++pos_;
        t.kind = Token::Kind::punct;
        t.text = std::string(1, c);
        return t;
    }

    void skip_space()
    {
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == '\n') {
                ++line_;
                ++pos_;
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                ++pos_;
            } else if (c == '/' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '/') {
                while (pos_ < src_.size() && src_[pos_] != '\n') ++pos_;
            } else if (c == '/' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '*') {
                pos_ += 2;
                while (pos_ + 1 < src_.size() && !(src_[pos_] == '*' && src_[pos_ + 1] == '/')) {
                    if (src_[pos_] == '\n') ++line_;
                    ++pos_;
                }
                pos_ += 2;
            } else {
                break;
            }
        }
    }

    std::string_view src_;
    std::size_t pos_ = 0;
    int line_ = 1;
};

bool is_special_register(std::string_view name)
{
    static const char *names[] = {"%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid",
                                  "%nwarpid", "%smid", "%nsmid", "%gridid", "%clock",
                                  "%clock64", "%lanemask_eq", "%lanemask_lt", "%warpsize"};
    for (const char *n : names)
        if (name == n) return true;
    return false;
}

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : t_(std::move(tokens)) {}

    Module module()
    {
        Module m;
        while (!at_end()) {
            const Token &tok = peek();
            if (tok.kind != Token::Kind::directive) fail("expected a directive");
            const std::string d = tok.text;
            if (d == "version") {
                next();
                const std::string v = expect_kind(Token::Kind::number).text;
                const auto dot = v.find('.');
                m.version_major = std::stoi(v.substr(0, dot));
                m.version_minor = dot == std::string::npos ? 0 : std::stoi(v.substr(dot + 1));
            } else if (d == "target") {
                next();
                m.target = expect_kind(Token::Kind::ident).text;
                while (accept(",")) m.target += "," + expect_kind(Token::Kind::ident).text;
            } else if (d == "address_size") {
                next();
                m.address_size = std::stoi(expect_kind(Token::Kind::number).text);
            } else if (d == "section") {
                skip_section();
            } else if (d == "file" || d == "loc" || d == "pragma") {
                skip_line();
            } else {
                module_declaration(m);
            }
        }
        return m;
    }

private:
    // ---- token helpers ---------------------------------------------------
    const Token &peek(std::size_t ahead = 0) const
    {
        const std::size_t i = std::min(pos_ + ahead, t_.size() - 1);
        return t_[i];
    }
    bool at_end() const { return peek().kind == Token::Kind::end; }
    const Token &next()
    {
        const Token &tok = t_[pos_];
        if (pos_ + 1 < t_.size()) ++pos_;
        return tok;
    }
    bool is_punct(const char *p, std::size_t ahead = 0) const
    {
        return peek(ahead).kind == Token::Kind::punct && peek(ahead).text == p;
    }
    bool is_directive(const char *d, std::size_t ahead = 0) const
    {
        return peek(ahead).kind == Token::Kind::directive && peek(ahead).text == d;
    }
    bool accept(const char *p)
    {
        if (!is_punct(p)) return false;
        next();
        return true;
    }
    bool accept_directive(const char *d)
    {
        if (!is_directive(d)) return false;
        next();
        return true;
    }
    void expect(const char *p)
    {
        if (!accept(p)) fail(std::string("expected '") + p + "'");
    }
    const Token &expect_kind(Token::Kind kind)
    {
        if (peek().kind != kind) fail("unexpected token '" + peek().text + "'");
        return next();
    }
    [[noreturn]] void fail(const std::string &message) const
    {
        throw Error(message + " (near '" + peek().text + "')", peek().line);
    }
    void skip_line()
    {
        const int line = peek().line;
        while (!at_end() && peek().line == line) next();
    }

    // Skips a .section directive, including a multi-line { ... } body such as
    // nvcc's DWARF sections.
    void skip_section()
    {
        const int line = peek().line;
        int depth = 0;
        while (!at_end() && (depth > 0 || peek().line == line)) {
            if (is_punct("{")) ++depth;
            else if (is_punct("}") && --depth == 0) {
                next();
                break;
            }
            next();
        }
    }

    static std::uint64_t parse_uint(const std::string &text)
    {
        std::string s = text;
        if (!s.empty() && (s.back() == 'U' || s.back() == 'u')) s.pop_back();
        return std::stoull(s, nullptr, (s.size() > 2 && (s[1] == 'x' || s[1] == 'X')) ? 16 : 10);
    }
    std::uint64_t number_value()
    {
        return parse_uint(expect_kind(Token::Kind::number).text);
    }
    std::int64_t signed_number()
    {
        const bool negative = accept("-");
        const std::uint64_t value = number_value();
        return negative ? -static_cast<std::int64_t>(value) : static_cast<std::int64_t>(value);
    }

    Type type_directive()
    {
        const Token &tok = expect_kind(Token::Kind::directive);
        Type type;
        if (!parse_type_name(tok.text, type)) fail("unknown type ." + tok.text);
        return type;
    }

    // ---- declarations ------------------------------------------------------
    void module_declaration(Module &m)
    {
        bool external = false;
        for (;;) {
            if (accept_directive("visible") || accept_directive("weak") ||
                accept_directive("common"))
                continue;
            if (accept_directive("extern")) {
                external = true;
                continue;
            }
            break;
        }
        if (is_directive("entry") || is_directive("func")) {
            Function f = function();
            if (f.has_body || !m.find(f.name)) {
                for (auto it = m.functions.begin(); it != m.functions.end(); ++it)
                    if (it->name == f.name) {
                        m.functions.erase(it);
                        break;
                    }
                m.functions.push_back(std::move(f));
            }
            return;
        }
        Variable v = variable(external);
        m.vars.push_back(std::move(v));
    }

    Variable variable(bool external)
    {
        Variable v;
        v.external = external;
        v.line = peek().line;
        const std::string space = expect_kind(Token::Kind::directive).text;
        if (space == "global") v.space = Space::global;
        else if (space == "const") v.space = Space::constant;
        else if (space == "shared") v.space = Space::shared;
        else if (space == "local") v.space = Space::local;
        else if (space == "tex") v.space = Space::tex;
        else fail("unsupported state space ." + space);
        int vector = 1;
        for (;;) {
            if (accept_directive("align")) {
                v.align = static_cast<std::uint32_t>(number_value());
            } else if (accept_directive("v2")) {
                vector = 2;
            } else if (accept_directive("v4")) {
                vector = 4;
            } else if (is_directive("ptr") || is_directive("attribute")) {
                fail("unsupported variable attribute");
            } else {
                break;
            }
        }
        v.type = type_directive();
        if (v.type.kind == 't') v.space = Space::tex;
        v.name = expect_kind(Token::Kind::ident).text;
        if (vector != 1) v.dims.push_back(static_cast<std::uint64_t>(vector));
        while (accept("[")) {
            if (accept("]")) {
                v.dims.push_back(0);
                continue;
            }
            v.dims.push_back(number_value());
            expect("]");
        }
        if (accept("=")) {
            v.has_init = true;
            initializer(v);
            // name[] = {...}: the omitted dimension comes from the initializer.
            std::uint64_t known = v.element_bytes();
            std::uint64_t *unsized = nullptr;
            for (std::uint64_t &d : v.dims) {
                if (d == 0 && !unsized) unsized = &d;
                else known *= d;
            }
            if (unsized && known) *unsized = (v.init.size() + known - 1) / known;
        }
        expect(";");
        return v;
    }

    void initializer(Variable &v)
    {
        const std::size_t width = static_cast<std::size_t>(v.element_bytes());
        auto append = [&](std::uint64_t bits) {
            for (std::size_t i = 0; i < width; ++i)
                v.init.push_back(static_cast<std::uint8_t>(bits >> (8 * i)));
        };
        auto value = [&]() {
            const bool negative = accept("-");
            const std::string text = expect_kind(Token::Kind::number).text;
            const bool is_float = v.type.kind == 'f';
            const std::uint64_t sign = width == 8 ? 1ull << 63 : width == 4 ? 1ull << 31 : 1ull << 15;
            std::uint64_t bits = 0;
            if (text.size() > 2 && (text[1] == 'f' || text[1] == 'F' || text[1] == 'd' || text[1] == 'D')) {
                // Hex float: exact bit pattern; negation flips the sign bit.
                bits = std::stoull(text.substr(2), nullptr, 16);
                if (negative) bits ^= sign;
            } else if (is_float && text.find_first_of(".eE") != std::string::npos &&
                       text.compare(0, 2, "0x") != 0 && text.compare(0, 2, "0X") != 0) {
                // Decimal float literal, converted to the variable's width.
                double d = std::stod(text);
                if (negative) d = -d;
                if (width == 8) {
                    std::memcpy(&bits, &d, 8);
                } else if (width == 4) {
                    const float f = static_cast<float>(d);
                    std::uint32_t u;
                    std::memcpy(&u, &f, 4);
                    bits = u;
                } else {
                    fail("unsupported .f16 initializer");
                }
            } else {
                bits = parse_uint(text);
                if (negative) bits = static_cast<std::uint64_t>(-static_cast<std::int64_t>(bits));
            }
            append(bits);
        };
        int depth = 0;
        if (!is_punct("{")) {
            value();
            return;
        }
        do {
            if (accept("{")) {
                ++depth;
                continue;
            }
            if (accept("}")) {
                --depth;
                continue;
            }
            if (accept(",")) continue;
            value();
        } while (depth > 0);
    }

    Param param()
    {
        if (!accept_directive("param") && !accept_directive("reg")) fail("expected .param");
        Param p;
        for (;;) {
            if (accept_directive("align")) p.align = static_cast<std::uint32_t>(number_value());
            else if (accept_directive("ptr")) {
                // .ptr.space.align N qualifiers carry no layout information
                while (peek().kind == Token::Kind::directive && peek().text != "b8" &&
                       peek().text != "align") {
                    Type ignored;
                    if (parse_type_name(peek().text, ignored)) break;
                    next();
                }
            } else break;
        }
        p.type = type_directive();
        if (accept_directive("ptr")) {
            // .param .u32 .ptr.global.align 16 name: no layout information
            while (peek().kind == Token::Kind::directive) {
                if (accept_directive("align")) number_value();
                else next();
            }
        }
        p.name = expect_kind(Token::Kind::ident).text;
        if (accept("[")) {
            p.array = number_value();
            expect("]");
        }
        return p;
    }

    std::vector<Param> param_list()
    {
        std::vector<Param> out;
        expect("(");
        if (accept(")")) return out;
        do {
            out.push_back(param());
        } while (accept(","));
        expect(")");
        return out;
    }

    Function function()
    {
        Function f;
        f.line = peek().line;
        f.is_entry = next().text == "entry";
        if (!f.is_entry && is_punct("(")) f.returns = param_list();
        f.name = expect_kind(Token::Kind::ident).text;
        if (is_punct("(")) f.params = param_list();
        for (;;) {
            if (accept_directive("maxntid")) {
                read_dims(f.maxntid);
            } else if (accept_directive("reqntid")) {
                read_dims(f.reqntid);
            } else if (accept_directive("minnctapersm") || accept_directive("maxnreg") ||
                       accept_directive("maxnctapersm")) {
                number_value();
            } else if (accept_directive("noreturn")) {
            } else {
                break;
            }
        }
        if (accept(";")) return f;
        f.has_body = true;
        expect("{");
        body(f, 1);
        return f;
    }

    void read_dims(std::uint32_t (&dims)[3])
    {
        dims[0] = static_cast<std::uint32_t>(number_value());
        dims[1] = dims[2] = 1;
        if (accept(",")) dims[1] = static_cast<std::uint32_t>(number_value());
        if (accept(",")) dims[2] = static_cast<std::uint32_t>(number_value());
    }

    void body(Function &f, int depth)
    {
        while (depth > 0) {
            if (at_end()) fail("unterminated function body");
            if (accept("{")) {
                ++depth;
                continue;
            }
            if (accept("}")) {
                --depth;
                continue;
            }
            const Token &tok = peek();
            if (tok.kind == Token::Kind::directive) {
                if (tok.text == "reg") {
                    register_declaration(f);
                } else if (tok.text == "local" || tok.text == "shared") {
                    f.vars.push_back(variable(false));
                } else if (tok.text == "pragma" || tok.text == "loc" || tok.text == "file") {
                    skip_line();
                    accept(";");
                } else {
                    fail("unsupported directive ." + tok.text + " in function body");
                }
                continue;
            }
            if (tok.kind == Token::Kind::ident && is_punct(":", 1) && tok.text[0] != '%') {
                Statement s;
                s.is_label = true;
                s.label = next().text;
                next();
                f.body.push_back(std::move(s));
                continue;
            }
            Statement s;
            s.inst = instruction();
            f.body.push_back(std::move(s));
        }
    }

    void register_declaration(Function &f)
    {
        next(); // .reg
        while (accept_directive("v2") || accept_directive("v4")) fail("vector registers are unsupported");
        const Type type = type_directive();
        do {
            RegDecl r;
            r.type = type;
            r.name = expect_kind(Token::Kind::ident).text;
            if (accept("<")) {
                r.count = static_cast<std::uint32_t>(number_value());
                expect(">");
            }
            f.regs.push_back(std::move(r));
        } while (accept(","));
        expect(";");
    }

    Instruction instruction()
    {
        Instruction in;
        in.line = peek().line;
        if (accept("@")) {
            in.guard_negated = accept("!");
            in.guard = expect_kind(Token::Kind::ident).text;
        }
        in.op = expect_kind(Token::Kind::ident).text;
        while (peek().kind == Token::Kind::directive && peek().line == in.line) {
            in.mods.push_back(next().text);
        }
        if (!accept(";")) {
            do {
                in.operands.push_back(operand());
            } while (accept(","));
            expect(";");
        }
        return in;
    }

    Operand operand()
    {
        Operand o;
        if (accept("[")) {
            o.kind = Operand::Kind::address;
            address(o);
            expect("]");
            return o;
        }
        if (accept("{")) {
            o.kind = Operand::Kind::vector;
            do {
                o.elems.push_back(operand());
            } while (accept(","));
            expect("}");
            return o;
        }
        if (accept("!")) {
            o = operand();
            o.negated = true;
            return o;
        }
        if (is_punct("-") || peek().kind == Token::Kind::number) {
            const bool negative = accept("-");
            const std::string text = expect_kind(Token::Kind::number).text;
            if (text.size() > 2 && (text[1] == 'f' || text[1] == 'F')) {
                o.kind = Operand::Kind::fimm;
                o.fbits = std::stoull(text.substr(2), nullptr, 16);
                if (negative) o.fbits ^= 0x80000000u;
                return o;
            }
            if (text.size() > 2 && (text[1] == 'd' || text[1] == 'D')) {
                o.kind = Operand::Kind::fimm;
                o.fwidth = 64;
                o.fbits = std::stoull(text.substr(2), nullptr, 16);
                if (negative) o.fbits ^= 0x8000000000000000ull;
                return o;
            }
            if (text.find('.') != std::string::npos) fail("decimal float literals are unsupported");
            o.kind = Operand::Kind::imm;
            const std::uint64_t value = parse_uint(text);
            o.imm = negative ? -static_cast<std::int64_t>(value) : static_cast<std::int64_t>(value);
            return o;
        }
        const std::string name = expect_kind(Token::Kind::ident).text;
        if (is_special_register(name)) {
            o.kind = Operand::Kind::special;
            o.name = name;
            if (peek().kind == Token::Kind::directive &&
                (peek().text == "x" || peek().text == "y" || peek().text == "z"))
                o.name += "." + next().text;
            return o;
        }
        o.name = name;
        o.kind = name[0] == '%' ? Operand::Kind::reg : Operand::Kind::symbol;
        if (o.kind == Operand::Kind::symbol && (is_punct("+") && peek(1).kind == Token::Kind::number)) {
            // "symbol+offset" used as an immediate address operand
            next();
            o.kind = Operand::Kind::address;
            o.base = Operand::Base::symbol;
            o.imm = static_cast<std::int64_t>(number_value());
        }
        return o;
    }

    void address(Operand &o)
    {
        if (peek().kind == Token::Kind::number || is_punct("-")) {
            o.base = Operand::Base::none;
            o.imm = signed_number();
            return;
        }
        const std::string name = expect_kind(Token::Kind::ident).text;
        if (accept(",")) {
            // texture fetch: [texref, {coordinates}]
            Operand ref;
            ref.kind = Operand::Kind::symbol;
            ref.name = name;
            o.elems.push_back(ref);
            o.elems.push_back(operand());
            o.base = Operand::Base::symbol;
            o.name = name;
            return;
        }
        o.name = name;
        o.base = name[0] == '%' ? Operand::Base::reg : Operand::Base::symbol;
        if (accept("+")) {
            o.imm = signed_number();
        } else if (accept("-")) {
            o.imm = -static_cast<std::int64_t>(number_value());
        }
    }

    std::vector<Token> t_;
    std::size_t pos_ = 0;
};

} // namespace

Module parse(std::string_view text)
{
    Lexer lexer(text);
    Parser parser(lexer.run());
    Module m = parser.module();
    if (m.version_major == 0) throw Error("missing .version directive", 1);
    return m;
}

} // namespace pxvk::ptx
