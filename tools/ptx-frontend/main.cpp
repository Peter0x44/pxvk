#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;

enum class TokenKind {
    word,
    punctuation,
    newline,
    end,
};

struct Token {
    TokenKind kind = TokenKind::end;
    std::string text;
    std::size_t line = 1;
    std::size_t column = 1;
};

class ParseError final : public std::runtime_error {
public:
    ParseError(std::size_t line, std::size_t column, const std::string &message)
        : std::runtime_error(message), line_(line), column_(column)
    {
    }

    std::size_t line() const { return line_; }
    std::size_t column() const { return column_; }

private:
    std::size_t line_;
    std::size_t column_;
};

bool is_punctuation(char value)
{
    switch (value) {
    case '{':
    case '}':
    case '[':
    case ']':
    case '(':
    case ')':
    case ',':
    case ';':
    case ':':
    case '+':
    case '-':
    case '<':
    case '>':
    case '@':
    case '!':
    case '=':
        return true;
    default:
        return false;
    }
}

std::vector<Token> tokenize(std::string_view source)
{
    std::vector<Token> tokens;
    std::size_t index = 0;
    std::size_t line = 1;
    std::size_t column = 1;

    const auto advance = [&](char value, std::size_t &current_line, std::size_t &current_column) {
        if (value == '\n') {
            ++current_line;
            current_column = 1;
        } else {
            ++current_column;
        }
    };

    while (index < source.size()) {
        const char value = source[index];
        if (value == '\r') {
            ++index;
            continue;
        }
        if (value == '\n') {
            tokens.push_back({TokenKind::newline, "\n", line, column});
            advance(value, line, column);
            ++index;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(value)) != 0) {
            advance(value, line, column);
            ++index;
            continue;
        }
        if (value == '/' && index + 1 < source.size() && source[index + 1] == '/') {
            while (index < source.size() && source[index] != '\n') {
                advance(source[index], line, column);
                ++index;
            }
            continue;
        }
        if (value == '/' && index + 1 < source.size() && source[index + 1] == '*') {
            const std::size_t start_line = line;
            const std::size_t start_column = column;
            advance(source[index], line, column);
            advance(source[index + 1], line, column);
            index += 2;
            bool closed = false;
            while (index < source.size()) {
                if (source[index] == '*' && index + 1 < source.size() && source[index + 1] == '/') {
                    advance(source[index], line, column);
                    advance(source[index + 1], line, column);
                    index += 2;
                    closed = true;
                    break;
                }
                advance(source[index], line, column);
                ++index;
            }
            if (!closed) {
                throw ParseError(start_line, start_column, "unterminated block comment");
            }
            continue;
        }
        if (is_punctuation(value)) {
            tokens.push_back({TokenKind::punctuation, std::string(1, value), line, column});
            advance(value, line, column);
            ++index;
            continue;
        }

        const std::size_t start = index;
        const std::size_t start_line = line;
        const std::size_t start_column = column;
        while (index < source.size()) {
            const char current = source[index];
            if (current == '\r' || current == '\n' ||
                std::isspace(static_cast<unsigned char>(current)) != 0 ||
                is_punctuation(current) ||
                (current == '/' && index + 1 < source.size() &&
                 (source[index + 1] == '/' || source[index + 1] == '*'))) {
                break;
            }
            advance(current, line, column);
            ++index;
        }
        if (index == start) {
            throw ParseError(line, column, "unrecognized character");
        }
        tokens.push_back({
            TokenKind::word,
            std::string(source.substr(start, index - start)),
            start_line,
            start_column,
        });
    }
    tokens.push_back({TokenKind::end, "", line, column});
    return tokens;
}

struct Parameter {
    std::string name;
    std::string type;
    std::size_t offset = 0;
    std::size_t size = 0;
    std::size_t alignment = 0;
};

struct Instruction {
    std::string variant;
    std::string base;
    std::vector<std::string> modifiers;
    std::vector<std::string> operands;
    std::string predicate;
    std::size_t line = 0;
};

struct Entry {
    std::string name;
    std::string kind = "entry";
    std::string linkage = "visible";
    std::vector<Parameter> return_parameters;
    std::vector<Parameter> parameters;
    std::vector<Instruction> instructions;
    std::vector<std::uint32_t> max_threads;
    std::uint32_t minimum_ctas_per_sm = 0;
    std::size_t packed_parameter_bytes = 0;
    std::size_t return_parameter_bytes = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t static_local_bytes = 0;
};

struct ModuleVariable {
    std::string name;
    std::string space;
    std::string type;
    std::size_t size = 0;
    std::size_t alignment = 0;
    bool external = false;
};

struct Module {
    std::string version;
    std::string target;
    std::uint32_t address_size = 0;
    std::vector<std::string> texture_references;
    std::vector<ModuleVariable> variables;
    std::vector<Entry> functions;
    std::vector<Entry> entries;
};

std::size_t align_up(std::size_t value, std::size_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

std::size_t scalar_size(const Token &token)
{
    static const std::map<std::string, std::size_t> sizes = {
        {".b8", 1}, {".u8", 1}, {".s8", 1},
        {".b16", 2}, {".u16", 2}, {".s16", 2},
        {".b32", 4}, {".u32", 4}, {".s32", 4}, {".f32", 4},
        {".b64", 8}, {".u64", 8}, {".s64", 8}, {".f64", 8},
        {".pred", 1},
    };
    const auto found = sizes.find(token.text);
    if (found == sizes.end()) {
        throw ParseError(token.line, token.column, "unsupported scalar type '" + token.text + "'");
    }
    return found->second;
}

std::vector<std::string> split_opcode(const std::string &opcode)
{
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start < opcode.size()) {
        const std::size_t dot = opcode.find('.', start);
        parts.push_back(opcode.substr(start, dot == std::string::npos ? std::string::npos : dot - start));
        if (dot == std::string::npos) {
            break;
        }
        start = dot + 1;
    }
    return parts;
}

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    Module parse_module()
    {
        Module module;
        bool have_version = false;
        bool have_target = false;
        bool have_address_size = false;
        skip_newlines();
        while (!at_end()) {
            if (match_word(".version")) {
                const Token value = expect_word("PTX version");
                module.version = value.text;
                finish_header_line();
                have_version = true;
            } else if (match_word(".target")) {
                const Token value = expect_word("PTX target");
                module.target = value.text;
                finish_header_line();
                have_target = true;
            } else if (match_word(".address_size")) {
                const Token value = expect_word("address size");
                module.address_size = parse_unsigned(value);
                finish_header_line();
                have_address_size = true;
            } else if (match_word(".global")) {
                if (match_word(".texref")) {
                    const Token name = expect_word("texture reference name");
                    expect_text(";");
                    module.texture_references.push_back(name.text);
                    skip_to_line_end();
                } else {
                    module.variables.push_back(parse_module_variable({".global"}));
                }
            } else if (match_word(".const")) {
                module.variables.push_back(parse_module_variable({".const"}));
            } else if (match_word(".shared")) {
                module.variables.push_back(parse_module_variable({".shared"}));
            } else if (match_word(".extern")) {
                module.variables.push_back(parse_module_variable({".extern"}));
            } else if (match_word(".weak")) {
                expect_text(".func");
                module.functions.push_back(parse_routine("function", "weak"));
            } else if (match_word(".visible")) {
                if (match_word(".entry")) {
                    module.entries.push_back(parse_routine("entry", "visible"));
                } else if (match_word(".func")) {
                    module.functions.push_back(parse_routine("function", "visible"));
                } else {
                    fail(peek(), "expected '.entry' or '.func' after '.visible'");
                }
            } else if (match_word(".entry")) {
                module.entries.push_back(parse_routine("entry", "default"));
            } else if (match_word(".func")) {
                module.functions.push_back(parse_routine("function", "default"));
            } else {
                fail(peek(), "unsupported module declaration '" + peek().text + "'");
            }
            skip_newlines();
        }
        if (!have_version || !have_target || !have_address_size) {
            fail(peek(), "module is missing version, target, or address size");
        }
        if (module.entries.empty() && module.functions.empty() && module.variables.empty()) {
            fail(peek(), "module has no declarations");
        }
        return module;
    }

private:
    static const std::set<std::string> &supported_bases()
    {
        static const std::set<std::string> bases = {
            "abs", "add", "and", "bar", "bfe", "bra", "clz", "cvt", "cvta",
            "div", "fma", "ld", "mad", "max", "min", "mov", "mul", "neg", "not",
            "or", "rcp", "ret", "rsqrt", "selp", "setp", "shl", "shr", "sqrt",
            "st", "sub", "tex", "xor",
        };
        return bases;
    }

    std::size_t parse_parameter_list(std::vector<Parameter> &parameters)
    {
        expect_text("(");
        skip_newlines();
        std::size_t parameter_offset = 0;
        while (!match_text(")")) {
            expect_text(".param");
            std::size_t explicit_alignment = 0;
            if (match_word(".align")) {
                explicit_alignment = parse_unsigned(expect_word("parameter alignment"));
            }
            const Token type = expect_word("parameter type");
            const std::size_t element_size = scalar_size(type);
            const Token name = expect_word("parameter name");
            std::size_t element_count = 1;
            if (match_text("[")) {
                element_count = parse_unsigned(expect_word("array length"));
                expect_text("]");
            }
            const std::size_t size = element_size * element_count;
            const std::size_t alignment =
                explicit_alignment == 0 ? element_size : explicit_alignment;
            parameter_offset = align_up(parameter_offset, alignment);
            parameters.push_back({name.text, type.text, parameter_offset, size, alignment});
            parameter_offset += size;
            skip_newlines();
            if (match_text(")")) {
                break;
            }
            expect_text(",");
            skip_newlines();
        }
        return parameter_offset;
    }

    Entry parse_routine(const std::string &kind, const std::string &linkage)
    {
        Entry routine;
        routine.kind = kind;
        routine.linkage = linkage;
        if (kind == "function" && peek().text == "(") {
            routine.return_parameter_bytes = parse_parameter_list(routine.return_parameters);
            skip_newlines();
        }
        routine.name = expect_word(kind + " name").text;
        skip_newlines();
        routine.packed_parameter_bytes = parse_parameter_list(routine.parameters);
        skip_newlines();
        while (peek().text == ".maxntid" || peek().text == ".minnctapersm") {
            if (match_word(".maxntid")) {
                routine.max_threads.clear();
                while (peek().kind != TokenKind::newline && !at_end()) {
                    if (match_text(",")) {
                        continue;
                    }
                    routine.max_threads.push_back(parse_unsigned(expect_word("thread dimension")));
                }
            } else {
                expect_text(".minnctapersm");
                routine.minimum_ctas_per_sm = parse_unsigned(expect_word("minimum CTA count"));
            }
            skip_newlines();
        }
        expect_text("{");
        skip_newlines();
        while (!match_text("}")) {
            if (at_end()) {
                fail(peek(), "unterminated " + kind + " body");
            }
            if (peek().text == ".reg" || peek().text == ".shared" ||
                peek().text == ".local" || peek().text == ".extern") {
                parse_declaration(routine);
            } else if (peek().kind == TokenKind::word && peek(1).text == ":") {
                consume();
                consume();
                skip_to_line_end();
            } else {
                routine.instructions.push_back(parse_instruction());
            }
            skip_newlines();
        }
        return routine;
    }

    ModuleVariable parse_module_variable(std::vector<std::string> prefix)
    {
        std::vector<Token> declaration;
        for (const std::string &text : prefix) {
            declaration.push_back({TokenKind::word, text, peek().line, peek().column});
        }
        while (!at_end() && peek().text != ";") {
            if (peek().kind == TokenKind::newline) {
                fail(peek(), "module variable declaration is missing ';'");
            }
            declaration.push_back(consume());
        }
        expect_text(";");
        ModuleVariable variable;
        variable.external = std::any_of(
            declaration.begin(), declaration.end(),
            [](const Token &token) { return token.text == ".extern"; });
        for (const Token &token : declaration) {
            if (token.text == ".const" || token.text == ".shared" || token.text == ".global") {
                variable.space = token.text.substr(1);
            }
        }
        for (std::size_t index = 0; index + 1 < declaration.size(); ++index) {
            if (declaration[index].text == ".align") {
                variable.alignment = parse_unsigned(declaration[index + 1]);
            }
        }
        for (std::size_t index = 0; index < declaration.size(); ++index) {
            try {
                const std::size_t element_size = scalar_size(declaration[index]);
                variable.type = declaration[index].text;
                if (index + 1 >= declaration.size()) {
                    fail(declaration[index], "module variable has no name");
                }
                variable.name = declaration[index + 1].text;
                std::size_t count = 1;
                for (std::size_t cursor = index + 2; cursor < declaration.size(); ++cursor) {
                    if (declaration[cursor].text == "[") {
                        if (cursor + 1 < declaration.size() && declaration[cursor + 1].text != "]") {
                            count = parse_unsigned(declaration[cursor + 1]);
                        } else {
                            count = 0;
                        }
                        break;
                    }
                }
                variable.size = element_size * count;
                if (variable.alignment == 0) {
                    variable.alignment = element_size;
                }
                return variable;
            } catch (const ParseError &) {
                if (!declaration[index].text.empty() && declaration[index].text[0] == '.' &&
                    declaration[index].text != ".align" && declaration[index].text != ".extern" &&
                    declaration[index].text != ".const" && declaration[index].text != ".shared" &&
                    declaration[index].text != ".global") {
                    throw;
                }
            }
        }
        fail(declaration.front(), "module variable has no supported type");
    }

    void parse_declaration(Entry &entry)
    {
        std::vector<Token> declaration;
        while (!at_end() && peek().text != ";") {
            if (peek().kind == TokenKind::newline) {
                fail(peek(), "declaration is missing ';'");
            }
            declaration.push_back(consume());
        }
        expect_text(";");
        if (declaration.empty()) {
            fail(peek(), "empty declaration");
        }
        const bool shared = std::any_of(
            declaration.begin(), declaration.end(),
            [](const Token &token) { return token.text == ".shared"; });
        const bool local = std::any_of(
            declaration.begin(), declaration.end(),
            [](const Token &token) { return token.text == ".local"; });
        if (!shared && !local) {
            return;
        }
        const bool external = std::any_of(
            declaration.begin(), declaration.end(),
            [](const Token &token) { return token.text == ".extern"; });
        if (external) {
            return;
        }
        std::size_t explicit_alignment = 0;
        for (std::size_t index = 0; index + 1 < declaration.size(); ++index) {
            if (declaration[index].text == ".align") {
                explicit_alignment = parse_unsigned(declaration[index + 1]);
                break;
            }
        }
        for (std::size_t index = 0; index < declaration.size(); ++index) {
            if (declaration[index].text == ".u8" || declaration[index].text == ".u16" ||
                declaration[index].text == ".u32" || declaration[index].text == ".u64" ||
                declaration[index].text == ".s8" || declaration[index].text == ".s16" ||
                declaration[index].text == ".s32" || declaration[index].text == ".s64" ||
                declaration[index].text == ".b8" || declaration[index].text == ".b16" ||
                declaration[index].text == ".b32" || declaration[index].text == ".b64" ||
                declaration[index].text == ".f32") {
                const std::size_t element_size = scalar_size(declaration[index]);
                std::size_t count = 1;
                for (std::size_t cursor = index + 1; cursor + 2 < declaration.size(); ++cursor) {
                    if (declaration[cursor].text == "[") {
                        count = parse_unsigned(declaration[cursor + 1]);
                        if (declaration[cursor + 2].text != "]") {
                            fail(declaration[cursor + 2], "malformed storage array");
                        }
                        break;
                    }
                }
                const std::size_t alignment =
                    explicit_alignment == 0 ? element_size : explicit_alignment;
                std::size_t &storage_bytes =
                    shared ? entry.static_shared_bytes : entry.static_local_bytes;
                storage_bytes = align_up(storage_bytes, alignment);
                storage_bytes += element_size * count;
                return;
            }
        }
        fail(declaration.front(), "storage declaration has no supported type");
    }

    Instruction parse_instruction()
    {
        Instruction instruction;
        if (match_text("@")) {
            std::string predicate = "@";
            if (match_text("!")) {
                predicate += "!";
            }
            predicate += expect_word("predicate register").text;
            instruction.predicate = predicate;
        }
        const Token opcode = expect_word("instruction opcode");
        instruction.variant = opcode.text;
        instruction.line = opcode.line;
        const std::vector<std::string> parts = split_opcode(opcode.text);
        instruction.base = parts.front();
        instruction.modifiers.assign(parts.begin() + 1, parts.end());
        if (supported_bases().count(instruction.base) == 0) {
            fail(opcode, "unsupported instruction base '" + instruction.base + "'");
        }

        std::vector<Token> operand_tokens;
        while (!at_end() && peek().text != ";") {
            if (peek().kind == TokenKind::newline) {
                fail(peek(), "instruction is missing ';'");
            }
            operand_tokens.push_back(consume());
        }
        expect_text(";");
        instruction.operands = split_operands(operand_tokens);
        return instruction;
    }

    static std::vector<std::string> split_operands(const std::vector<Token> &tokens)
    {
        std::vector<std::string> operands;
        std::string current;
        int round_depth = 0;
        int square_depth = 0;
        int brace_depth = 0;
        for (const Token &token : tokens) {
            if (token.text == "(" ) ++round_depth;
            if (token.text == "[") ++square_depth;
            if (token.text == "{") ++brace_depth;
            if (token.text == "," && round_depth == 0 && square_depth == 0 && brace_depth == 0) {
                if (!current.empty()) {
                    operands.push_back(current);
                    current.clear();
                }
                continue;
            }
            current += token.text;
            if (token.text == ")") --round_depth;
            if (token.text == "]") --square_depth;
            if (token.text == "}") --brace_depth;
            if (round_depth < 0 || square_depth < 0 || brace_depth < 0) {
                throw ParseError(token.line, token.column, "unbalanced operand delimiter");
            }
        }
        if (round_depth != 0 || square_depth != 0 || brace_depth != 0) {
            const Token &token = tokens.empty() ? Token{} : tokens.back();
            throw ParseError(token.line, token.column, "unbalanced operand delimiter");
        }
        if (!current.empty()) {
            operands.push_back(current);
        }
        return operands;
    }

    static std::uint32_t parse_unsigned(const Token &token)
    {
        try {
            std::size_t consumed = 0;
            const unsigned long value = std::stoul(token.text, &consumed, 10);
            if (consumed != token.text.size() || value > UINT32_MAX) {
                throw std::out_of_range("invalid");
            }
            return static_cast<std::uint32_t>(value);
        } catch (const std::exception &) {
            throw ParseError(token.line, token.column, "invalid unsigned integer '" + token.text + "'");
        }
    }

    void finish_header_line()
    {
        if (match_text(";")) {
            skip_to_line_end();
            return;
        }
        if (peek().kind != TokenKind::newline && !at_end()) {
            fail(peek(), "unexpected token after header directive");
        }
        skip_newlines();
    }

    void skip_to_line_end()
    {
        while (!at_end() && peek().kind != TokenKind::newline) {
            consume();
        }
        skip_newlines();
    }

    void skip_newlines()
    {
        while (peek().kind == TokenKind::newline) {
            consume();
        }
    }

    bool at_end() const { return peek().kind == TokenKind::end; }

    const Token &peek(std::size_t lookahead = 0) const
    {
        const std::size_t position = std::min(index_ + lookahead, tokens_.size() - 1);
        return tokens_[position];
    }

    Token consume()
    {
        const Token token = peek();
        if (!at_end()) {
            ++index_;
        }
        return token;
    }

    bool match_text(const std::string &text)
    {
        if (peek().text == text) {
            consume();
            return true;
        }
        return false;
    }

    bool match_word(const std::string &text)
    {
        if (peek().kind == TokenKind::word && peek().text == text) {
            consume();
            return true;
        }
        return false;
    }

    Token expect_word(const std::string &description)
    {
        if (peek().kind != TokenKind::word) {
            fail(peek(), "expected " + description);
        }
        return consume();
    }

    void expect_text(const std::string &text)
    {
        if (!match_text(text)) {
            fail(peek(), "expected '" + text + "'");
        }
    }

    [[noreturn]] static void fail(const Token &token, const std::string &message)
    {
        throw ParseError(token.line, token.column, message);
    }

    std::vector<Token> tokens_;
    std::size_t index_ = 0;
};

Json parameters_to_json(const std::vector<Parameter> &parameters)
{
    Json output = Json::array();
    for (const Parameter &parameter : parameters) {
        output.push_back({
            {"name", parameter.name},
            {"type", parameter.type},
            {"offset", parameter.offset},
            {"size", parameter.size},
            {"alignment", parameter.alignment},
        });
    }
    return output;
}

Json routine_to_json(const Entry &routine)
{
    Json output;
    output["name"] = routine.name;
    output["kind"] = routine.kind;
    output["linkage"] = routine.linkage;
    output["packed_parameter_bytes"] = routine.packed_parameter_bytes;
    output["return_parameter_bytes"] = routine.return_parameter_bytes;
    output["static_shared_bytes"] = routine.static_shared_bytes;
    output["static_local_bytes"] = routine.static_local_bytes;
    output["max_threads"] = routine.max_threads;
    output["minimum_ctas_per_sm"] = routine.minimum_ctas_per_sm;
    output["return_parameters"] = parameters_to_json(routine.return_parameters);
    output["parameters"] = parameters_to_json(routine.parameters);

    std::set<std::string> base_families;
    std::set<std::string> variants;
    output["instructions"] = Json::array();
    for (const Instruction &instruction : routine.instructions) {
        base_families.insert(instruction.base);
        variants.insert(instruction.variant);
        output["instructions"].push_back({
            {"line", instruction.line},
            {"variant", instruction.variant},
            {"base", instruction.base},
            {"modifiers", instruction.modifiers},
            {"predicate", instruction.predicate},
            {"operands", instruction.operands},
        });
    }
    output["instruction_count"] = routine.instructions.size();
    output["base_families"] =
        std::vector<std::string>(base_families.begin(), base_families.end());
    output["variants"] = std::vector<std::string>(variants.begin(), variants.end());
    return output;
}

Json module_to_json(const Module &module, std::size_t source_bytes)
{
    Json output;
    output["schema_version"] = 2;
    output["source_bytes"] = source_bytes;
    output["ptx_version"] = module.version;
    output["target"] = module.target;
    output["address_size"] = module.address_size;
    output["texture_references"] = module.texture_references;
    output["variables"] = Json::array();
    for (const ModuleVariable &variable : module.variables) {
        output["variables"].push_back({
            {"name", variable.name},
            {"space", variable.space},
            {"type", variable.type},
            {"size", variable.size},
            {"alignment", variable.alignment},
            {"external", variable.external},
        });
    }
    output["functions"] = Json::array();
    for (const Entry &function : module.functions) {
        output["functions"].push_back(routine_to_json(function));
    }
    output["entries"] = Json::array();
    for (const Entry &entry : module.entries) {
        output["entries"].push_back(routine_to_json(entry));
    }
    return output;
}

Json routine_summary_to_json(const Entry &routine)
{
    std::set<std::string> base_families;
    std::set<std::string> variants;
    for (const Instruction &instruction : routine.instructions) {
        base_families.insert(instruction.base);
        variants.insert(instruction.variant);
    }
    return {
        {"name", routine.name},
        {"kind", routine.kind},
        {"linkage", routine.linkage},
        {"packed_parameter_bytes", routine.packed_parameter_bytes},
        {"return_parameter_bytes", routine.return_parameter_bytes},
        {"static_shared_bytes", routine.static_shared_bytes},
        {"static_local_bytes", routine.static_local_bytes},
        {"parameters", parameters_to_json(routine.parameters)},
        {"max_threads", routine.max_threads},
        {"minimum_ctas_per_sm", routine.minimum_ctas_per_sm},
        {"instruction_count", routine.instructions.size()},
        {"base_families", std::vector<std::string>(base_families.begin(), base_families.end())},
        {"variants", std::vector<std::string>(variants.begin(), variants.end())},
    };
}

Json corpus_to_json(
    const std::vector<std::pair<std::filesystem::path, std::pair<Module, std::size_t>>> &modules)
{
    Json output;
    output["schema_version"] = 1;
    output["modules"] = Json::array();
    std::size_t entry_count = 0;
    std::size_t function_count = 0;
    std::size_t texture_count = 0;
    std::size_t constant_count = 0;
    std::size_t static_shared_count = 0;
    std::size_t external_shared_count = 0;
    std::set<std::string> base_families;
    std::set<std::string> variants;

    for (const auto &item : modules) {
        const Module &module = item.second.first;
        Json module_json;
        module_json["file"] = item.first.filename().string();
        module_json["source_bytes"] = item.second.second;
        module_json["ptx_version"] = module.version;
        module_json["target"] = module.target;
        module_json["address_size"] = module.address_size;
        module_json["texture_references"] = module.texture_references;
        module_json["variables"] = Json::array();
        std::size_t module_static_shared_bytes = 0;
        for (const ModuleVariable &variable : module.variables) {
            module_json["variables"].push_back({
                {"name", variable.name},
                {"space", variable.space},
                {"type", variable.type},
                {"size", variable.size},
                {"alignment", variable.alignment},
                {"external", variable.external},
            });
            if (variable.space == "const") {
                ++constant_count;
            } else if (variable.space == "shared" && variable.external) {
                ++external_shared_count;
            } else if (variable.space == "shared") {
                ++static_shared_count;
                module_static_shared_bytes =
                    align_up(module_static_shared_bytes, variable.alignment);
                module_static_shared_bytes += variable.size;
            }
        }
        module_json["module_static_shared_bytes"] = module_static_shared_bytes;
        module_json["function_count"] = module.functions.size();
        for (const Entry &function : module.functions) {
            for (const Instruction &instruction : function.instructions) {
                base_families.insert(instruction.base);
                variants.insert(instruction.variant);
            }
        }
        module_json["entries"] = Json::array();
        for (const Entry &entry : module.entries) {
            Json summary = routine_summary_to_json(entry);
            summary["effective_static_shared_bytes"] =
                entry.static_shared_bytes + module_static_shared_bytes;
            module_json["entries"].push_back(std::move(summary));
            for (const Instruction &instruction : entry.instructions) {
                base_families.insert(instruction.base);
                variants.insert(instruction.variant);
            }
        }
        entry_count += module.entries.size();
        function_count += module.functions.size();
        texture_count += module.texture_references.size();
        output["modules"].push_back(std::move(module_json));
    }
    output["totals"] = {
        {"module_count", modules.size()},
        {"entry_count", entry_count},
        {"function_count", function_count},
        {"texture_reference_count", texture_count},
        {"constant_count", constant_count},
        {"module_static_shared_count", static_shared_count},
        {"external_shared_count", external_shared_count},
        {"base_families", std::vector<std::string>(base_families.begin(), base_families.end())},
        {"variants", std::vector<std::string>(variants.begin(), variants.end())},
    };
    return output;
}

Module parse_source(const std::string &source)
{
    return Parser(tokenize(source)).parse_module();
}

bool expect_parse_failure(const std::string &source)
{
    try {
        (void)parse_source(source);
        return false;
    } catch (const ParseError &) {
        return true;
    }
}

int self_test()
{
    const std::string valid =
        ".version 4.1\n"
        ".target sm_52\n"
        ".address_size 32\n"
        ".visible .entry sample(\n"
        "  .param .u32 p0,\n"
        "  .param .align 4 .b8 p1[8]\n"
        ") {\n"
        "  .shared .align 4 .u32 values[3];\n"
        "  .shared .align 8 .u64 next;\n"
        "  ld.param.u32 %r1, [p0];\n"
        "  ret;\n"
        "}\n";
    const Module module = parse_source(valid);
    if (module.entries.size() != 1 || module.entries[0].packed_parameter_bytes != 12 ||
        module.entries[0].static_shared_bytes != 24 || module.entries[0].instructions.size() != 2) {
        std::cerr << "valid parser self-test produced wrong metadata\n";
        return 1;
    }
    const std::string missing_semicolon =
        ".version 4.1\n.target sm_52\n.address_size 32\n"
        ".visible .entry bad() { mov.u32 %r1, 1\n }\n";
    const std::string unsupported_opcode =
        ".version 4.1\n.target sm_52\n.address_size 32\n"
        ".visible .entry bad() { atom.global.add.u32 %r1, [%r2], %r3; }\n";
    if (!expect_parse_failure(missing_semicolon) || !expect_parse_failure(unsupported_opcode)) {
        std::cerr << "invalid parser self-test unexpectedly succeeded\n";
        return 1;
    }
    std::cout << "PTX frontend self-tests passed\n";
    return 0;
}

std::string read_file(const std::string &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file");
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        return self_test();
    }
    const bool corpus_requested = argc >= 2 && std::string(argv[1]) == "--corpus";
    const bool corpus_mode = corpus_requested && argc == 4;
    if ((corpus_requested && !corpus_mode) || (!corpus_requested && argc != 3)) {
        std::cerr << "usage: ptx-frontend <input.ptx> <output.json>\n"
                  << "       ptx-frontend --corpus <directory> <output.json>\n"
                  << "       ptx-frontend --self-test\n";
        return 2;
    }

    try {
        if (corpus_mode) {
            std::vector<std::filesystem::path> paths;
            for (const std::filesystem::directory_entry &item :
                 std::filesystem::directory_iterator(argv[2])) {
                if (item.is_regular_file() && item.path().extension() == ".ptx") {
                    paths.push_back(item.path());
                }
            }
            std::sort(paths.begin(), paths.end());
            if (paths.empty()) {
                throw std::runtime_error("corpus directory contains no PTX files");
            }
            std::vector<std::pair<std::filesystem::path, std::pair<Module, std::size_t>>> modules;
            for (const std::filesystem::path &path : paths) {
                const std::string source = read_file(path.string());
                try {
                    modules.push_back({path, {parse_source(source), source.size()}});
                } catch (const ParseError &error) {
                    std::cerr << path.string() << ':' << error.line() << ':' << error.column()
                              << ": " << error.what() << '\n';
                    return 1;
                }
            }
            std::ofstream output(argv[3], std::ios::binary);
            if (!output) {
                throw std::runtime_error("failed to open output file");
            }
            output << corpus_to_json(modules).dump(2) << '\n';
            std::cout << "Parsed " << modules.size() << " PTX modules from " << argv[2] << '\n';
            return 0;
        }

        const std::string source = read_file(argv[1]);
        const Module module = parse_source(source);
        std::ofstream output(argv[2], std::ios::binary);
        if (!output) {
            throw std::runtime_error("failed to open output file");
        }
        output << module_to_json(module, source.size()).dump(2) << '\n';
        std::cout << "Parsed " << module.entries.size() << " entries from " << argv[1] << '\n';
        return 0;
    } catch (const ParseError &error) {
        std::cerr << argv[1] << ':' << error.line() << ':' << error.column()
                  << ": " << error.what() << '\n';
        return 1;
    } catch (const std::exception &error) {
        std::cerr << argv[1] << ": " << error.what() << '\n';
        return 1;
    }
}
