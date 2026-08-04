
#include "threepp/loaders/xacro/Expr.hpp"

#include "threepp/loaders/xacro/YamlLite.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kE = 2.71828182845904523536;

    const std::set<std::string>& functionNames() {

        static const std::set<std::string> names{
                "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt", "exp",
                "log", "log10", "fabs", "abs", "floor", "ceil", "round", "min", "max",
                "pow", "radians", "degrees", "str", "float", "int", "bool", "len",
                "dict", "list", "load_yaml"};
        return names;
    }

    struct Token {
        enum class Type { End,
                          Number,
                          String,
                          Name,
                          Op };

        Type type{Type::End};
        std::string text;
        Value value;
    };

    class Lexer {

    public:
        explicit Lexer(std::string_view s): s_(s) {}

        std::vector<Token> run() {

            std::vector<Token> out;
            while (true) {
                skipSpace();
                if (p_ >= s_.size()) break;

                const char c = s_[p_];
                if (std::isdigit(static_cast<unsigned char>(c)) ||
                    (c == '.' && p_ + 1 < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_ + 1])))) {
                    out.push_back(number());
                } else if (c == '_' || std::isalpha(static_cast<unsigned char>(c))) {
                    out.push_back(name());
                } else if (c == '\'' || c == '"') {
                    out.push_back(string());
                } else {
                    out.push_back(op());
                }
            }
            out.push_back(Token{});
            return out;
        }

    private:
        void skipSpace() {

            while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
        }

        Token number() {

            const std::size_t start = p_;
            bool real = false;
            while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
            if (p_ < s_.size() && s_[p_] == '.') {
                real = true;
                ++p_;
                while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
            }
            if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
                const std::size_t save = p_;
                ++p_;
                if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
                if (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) {
                    real = true;
                    while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
                } else {
                    p_ = save;
                }
            }

            const std::string text(s_.substr(start, p_ - start));
            Token t;
            t.type = Token::Type::Number;
            t.text = text;
            if (real) {
                // The lexer already vetted every character, so the only job left is the
                // conversion itself — which libc++ without floating-point from_chars
                // (see ColladaLoader.cpp) hands to strtod on the null-terminated copy.
                double d{};
#if defined(__cpp_lib_to_chars)
                std::from_chars(text.data(), text.data() + text.size(), d);
#else
                d = std::strtod(text.c_str(), nullptr);
#endif
                t.value = Value(d);
            } else {
                long long i{};
                auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), i);
                if (ec != std::errc{}) throw XacroError("integer literal out of range: " + text);
                t.value = Value(i);
            }
            return t;
        }

        Token name() {

            const std::size_t start = p_;
            while (p_ < s_.size() &&
                   (s_[p_] == '_' || std::isalnum(static_cast<unsigned char>(s_[p_])))) ++p_;

            Token t;
            t.type = Token::Type::Name;
            t.text = std::string(s_.substr(start, p_ - start));
            return t;
        }

        Token string() {

            const char quote = s_[p_++];
            std::string out;
            while (true) {
                if (p_ >= s_.size()) throw XacroError("unterminated string literal");
                const char c = s_[p_++];
                if (c == quote) break;
                if (c != '\\') {
                    out += c;
                    continue;
                }
                if (p_ >= s_.size()) throw XacroError("unterminated string literal");
                const char e = s_[p_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '0': out += '\0'; break;
                    case '\\': out += '\\'; break;
                    case '\'': out += '\''; break;
                    case '"': out += '"'; break;
                    default: out += e; break;
                }
            }

            Token t;
            t.type = Token::Type::String;
            t.value = Value(out);
            return t;
        }

        Token op() {

            static const char* twoChar[] = {"**", "//", "==", "!=", "<=", ">="};
            for (const char* o : twoChar) {
                if (s_.compare(p_, 2, o) == 0) {
                    p_ += 2;
                    Token t;
                    t.type = Token::Type::Op;
                    t.text = o;
                    return t;
                }
            }

            static const std::string oneChar = "+-*/%<>()[],.=";
            const char c = s_[p_];
            if (oneChar.find(c) == std::string::npos) {
                throw XacroError(std::string("unexpected character '") + c + "'");
            }
            ++p_;

            Token t;
            t.type = Token::Type::Op;
            t.text = std::string(1, c);
            return t;
        }

        std::string_view s_;
        std::size_t p_{};
    };

    struct Node;
    using NodePtr = std::unique_ptr<Node>;

    struct Node {
        enum class Kind { Literal,
                          Name,
                          Attr,
                          Index,
                          Call,
                          Unary,
                          Binary,
                          BoolOp,
                          Not,
                          Compare,
                          Ternary,
                          ListLit };

        Kind kind{Kind::Literal};
        Value literal;
        std::string text;
        std::vector<NodePtr> kids;
        std::vector<std::string> ops;    // Compare
        std::vector<std::string> kwNames;// Call
        std::size_t positional{};        // Call
    };

    NodePtr make(Node::Kind kind) {

        auto n = std::make_unique<Node>();
        n->kind = kind;
        return n;
    }

    class Parser {

    public:
        explicit Parser(std::vector<Token> tokens): t_(std::move(tokens)) {}

        NodePtr run() {

            NodePtr n = ternary();
            if (peek().type != Token::Type::End) {
                throw XacroError("unexpected token '" + describe(peek()) + "'");
            }
            return n;
        }

    private:
        static std::string describe(const Token& t) {

            switch (t.type) {
                case Token::Type::End: return "<end>";
                case Token::Type::Number: return t.text;
                case Token::Type::String: return t.value.repr();
                default: return t.text;
            }
        }

        const Token& peek(std::size_t ahead = 0) const {

            const std::size_t i = p_ + ahead;
            return i < t_.size() ? t_[i] : t_.back();
        }

        bool isOp(const std::string& op, std::size_t ahead = 0) const {

            const Token& t = peek(ahead);
            return t.type == Token::Type::Op && t.text == op;
        }

        bool isKeyword(const std::string& kw, std::size_t ahead = 0) const {

            const Token& t = peek(ahead);
            return t.type == Token::Type::Name && t.text == kw;
        }

        void expectOp(const std::string& op) {

            if (!isOp(op)) throw XacroError("expected '" + op + "' but found '" + describe(peek()) + "'");
            ++p_;
        }

        NodePtr ternary() {

            NodePtr value = orExpr();
            if (!isKeyword("if")) return value;

            ++p_;
            NodePtr cond = orExpr();
            if (!isKeyword("else")) throw XacroError("expected 'else' in a conditional expression");
            ++p_;
            NodePtr other = ternary();

            auto n = make(Node::Kind::Ternary);
            n->kids.push_back(std::move(cond));
            n->kids.push_back(std::move(value));
            n->kids.push_back(std::move(other));
            return n;
        }

        NodePtr orExpr() {

            NodePtr lhs = andExpr();
            while (isKeyword("or")) {
                ++p_;
                auto n = make(Node::Kind::BoolOp);
                n->text = "or";
                n->kids.push_back(std::move(lhs));
                n->kids.push_back(andExpr());
                lhs = std::move(n);
            }
            return lhs;
        }

        NodePtr andExpr() {

            NodePtr lhs = notExpr();
            while (isKeyword("and")) {
                ++p_;
                auto n = make(Node::Kind::BoolOp);
                n->text = "and";
                n->kids.push_back(std::move(lhs));
                n->kids.push_back(notExpr());
                lhs = std::move(n);
            }
            return lhs;
        }

        NodePtr notExpr() {

            if (isKeyword("not")) {
                ++p_;
                auto n = make(Node::Kind::Not);
                n->kids.push_back(notExpr());
                return n;
            }
            return comparison();
        }

        NodePtr comparison() {

            NodePtr first = arith();

            std::vector<std::string> ops;
            std::vector<NodePtr> operands;

            while (true) {
                std::string op;
                if (isOp("==") || isOp("!=") || isOp("<") || isOp("<=") || isOp(">") || isOp(">=")) {
                    op = peek().text;
                    ++p_;
                } else if (isKeyword("in")) {
                    op = "in";
                    ++p_;
                } else if (isKeyword("not") && isKeyword("in", 1)) {
                    op = "not in";
                    p_ += 2;
                } else {
                    break;
                }
                ops.push_back(op);
                operands.push_back(arith());
            }

            if (ops.empty()) return first;

            auto n = make(Node::Kind::Compare);
            n->ops = std::move(ops);
            n->kids.push_back(std::move(first));
            for (auto& o : operands) n->kids.push_back(std::move(o));
            return n;
        }

        NodePtr arith() {

            NodePtr lhs = term();
            while (isOp("+") || isOp("-")) {
                const std::string op = peek().text;
                ++p_;
                auto n = make(Node::Kind::Binary);
                n->text = op;
                n->kids.push_back(std::move(lhs));
                n->kids.push_back(term());
                lhs = std::move(n);
            }
            return lhs;
        }

        NodePtr term() {

            NodePtr lhs = factor();
            while (isOp("*") || isOp("/") || isOp("//") || isOp("%")) {
                const std::string op = peek().text;
                ++p_;
                auto n = make(Node::Kind::Binary);
                n->text = op;
                n->kids.push_back(std::move(lhs));
                n->kids.push_back(factor());
                lhs = std::move(n);
            }
            return lhs;
        }

        NodePtr factor() {

            if (isOp("+") || isOp("-")) {
                const std::string op = peek().text;
                ++p_;
                auto n = make(Node::Kind::Unary);
                n->text = op;
                n->kids.push_back(factor());
                return n;
            }
            return power();
        }

        NodePtr power() {

            NodePtr base = postfix();
            if (isOp("**")) {
                ++p_;
                auto n = make(Node::Kind::Binary);
                n->text = "**";
                n->kids.push_back(std::move(base));
                n->kids.push_back(factor());
                return n;
            }
            return base;
        }

        NodePtr postfix() {

            NodePtr base = primary();
            while (true) {
                if (isOp("[")) {
                    ++p_;
                    auto n = make(Node::Kind::Index);
                    n->kids.push_back(std::move(base));
                    n->kids.push_back(ternary());
                    expectOp("]");
                    base = std::move(n);
                } else if (isOp(".")) {
                    ++p_;
                    if (peek().type != Token::Type::Name) throw XacroError("expected an attribute name after '.'");
                    auto n = make(Node::Kind::Attr);
                    n->text = peek().text;
                    ++p_;
                    n->kids.push_back(std::move(base));
                    base = std::move(n);
                } else if (isOp("(")) {
                    ++p_;
                    base = call(std::move(base));
                } else {
                    return base;
                }
            }
        }

        NodePtr call(NodePtr callee) {

            auto n = make(Node::Kind::Call);
            n->kids.push_back(std::move(callee));

            std::vector<NodePtr> positional;
            std::vector<NodePtr> keyword;

            while (!isOp(")")) {
                if (peek().type == Token::Type::Name && isOp("=", 1)) {
                    n->kwNames.push_back(peek().text);
                    p_ += 2;
                    keyword.push_back(ternary());
                } else {
                    if (!n->kwNames.empty()) {
                        throw XacroError("positional argument follows a keyword argument");
                    }
                    positional.push_back(ternary());
                }
                if (isOp(",")) {
                    ++p_;
                    continue;
                }
                break;
            }
            expectOp(")");

            n->positional = positional.size();
            for (auto& a : positional) n->kids.push_back(std::move(a));
            for (auto& a : keyword) n->kids.push_back(std::move(a));
            return n;
        }

        NodePtr primary() {

            const Token& t = peek();

            if (t.type == Token::Type::Number || t.type == Token::Type::String) {
                auto n = make(Node::Kind::Literal);
                n->literal = t.value;
                ++p_;
                return n;
            }

            if (t.type == Token::Type::Name) {
                auto n = make(Node::Kind::Name);
                n->text = t.text;
                ++p_;
                return n;
            }

            if (isOp("(")) {
                ++p_;
                NodePtr inner = ternary();
                expectOp(")");
                return inner;
            }

            if (isOp("[")) {
                ++p_;
                auto n = make(Node::Kind::ListLit);
                while (!isOp("]")) {
                    n->kids.push_back(ternary());
                    if (isOp(",")) {
                        ++p_;
                        continue;
                    }
                    break;
                }
                expectOp("]");
                return n;
            }

            throw XacroError("unexpected token '" + describe(t) + "'");
        }

        std::vector<Token> t_;
        std::size_t p_{};
    };

    struct Operand {
        enum class Kind { Val,
                          Func,
                          Module };

        Kind kind{Kind::Val};
        Value value;
        std::string name;
    };

    bool isIntegral(const Value& v) { return v.isInt() || v.isBool(); }

    long long floorDiv(long long a, long long b) {

        long long q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
        return q;
    }

    long long intPow(long long base, long long exp) {

        long long r = 1;
        while (exp > 0) {
            if (exp & 1) r *= base;
            base *= base;
            exp >>= 1;
        }
        return r;
    }

    Value add(const Value& a, const Value& b) {

        if (a.isNumber() && b.isNumber()) {
            if (isIntegral(a) && isIntegral(b)) return Value(a.asInt() + b.asInt());
            return Value(a.asNumber() + b.asNumber());
        }
        if (a.isString() && b.isString()) return Value(a.asString() + b.asString());
        if (a.isList() && b.isList()) {
            List out = a.asList();
            const auto& r = b.asList();
            out.insert(out.end(), r.begin(), r.end());
            return Value(std::move(out));
        }
        throw XacroError("cannot add " + a.typeName() + " and " + b.typeName());
    }

    Value arithmetic(const std::string& op, const Value& a, const Value& b) {

        if (op == "+") return add(a, b);

        if (!a.isNumber() || !b.isNumber()) {
            throw XacroError("cannot apply '" + op + "' to " + a.typeName() + " and " + b.typeName());
        }

        const bool ints = isIntegral(a) && isIntegral(b);

        if (op == "-") return ints ? Value(a.asInt() - b.asInt()) : Value(a.asNumber() - b.asNumber());
        if (op == "*") return ints ? Value(a.asInt() * b.asInt()) : Value(a.asNumber() * b.asNumber());

        if (op == "/") {
            if (b.asNumber() == 0.0) throw XacroError("division by zero");
            return Value(a.asNumber() / b.asNumber());
        }

        if (op == "//") {
            if (b.asNumber() == 0.0) throw XacroError("division by zero");
            if (ints) return Value(floorDiv(a.asInt(), b.asInt()));
            return Value(std::floor(a.asNumber() / b.asNumber()));
        }

        if (op == "%") {
            if (b.asNumber() == 0.0) throw XacroError("modulo by zero");
            if (ints) {
                const long long l = a.asInt();
                const long long r = b.asInt();
                long long m = l % r;
                if (m != 0 && ((m < 0) != (r < 0))) m += r;
                return Value(m);
            }
            const double l = a.asNumber();
            const double r = b.asNumber();
            double m = std::fmod(l, r);
            if (m != 0.0 && ((m < 0) != (r < 0))) m += r;
            return Value(m);
        }

        if (op == "**") {
            if (ints && b.asInt() >= 0) return Value(intPow(a.asInt(), b.asInt()));
            return Value(std::pow(a.asNumber(), b.asNumber()));
        }

        throw XacroError("unknown operator '" + op + "'");
    }

    bool contains(const Value& needle, const Value& haystack) {

        if (haystack.isDict()) {
            const std::string key = needle.isString() ? needle.asString() : needle.toString();
            return haystack.asDict().count(key) != 0;
        }
        if (haystack.isString()) {
            if (!needle.isString()) throw XacroError("'in' on a string needs a string on the left");
            return haystack.asString().find(needle.asString()) != std::string::npos;
        }
        if (haystack.isList()) {
            for (const auto& e : haystack.asList()) {
                if (e == needle) return true;
            }
            return false;
        }
        throw XacroError("'in' is not supported for " + haystack.typeName());
    }

    bool compareOp(const std::string& op, const Value& a, const Value& b) {

        if (op == "==") return a == b;
        if (op == "!=") return !(a == b);
        if (op == "in") return contains(a, b);
        if (op == "not in") return !contains(a, b);

        const int c = Value::compare(a, b);
        if (op == "<") return c < 0;
        if (op == "<=") return c <= 0;
        if (op == ">") return c > 0;
        return c >= 0;
    }

    Value subscript(const Value& base, const Value& key) {

        if (base.isDict()) {
            const std::string k = key.isString() ? key.asString() : key.toString();
            const auto& d = base.asDict();
            const auto it = d.find(k);
            if (it == d.end()) throw XacroError("no key '" + k + "' in " + base.toString());
            return it->second;
        }

        if (base.isList() || base.isString()) {
            const long long size = base.isList()
                                           ? static_cast<long long>(base.asList().size())
                                           : static_cast<long long>(base.asString().size());
            long long i = key.asInt();
            if (i < 0) i += size;
            if (i < 0 || i >= size) {
                throw XacroError(base.typeName() + " index " + key.toString() + " out of range");
            }
            if (base.isList()) return base.asList()[static_cast<std::size_t>(i)];
            return Value(std::string(1, base.asString()[static_cast<std::size_t>(i)]));
        }

        throw XacroError("cannot subscript " + base.typeName());
    }

    Operand evalNode(const Node& n, const EvalContext& ctx);

    Value evalValue(const Node& n, const EvalContext& ctx) {

        Operand o = evalNode(n, ctx);
        if (o.kind != Operand::Kind::Val) {
            throw XacroError("'" + o.name + "' cannot be used as a value");
        }
        return std::move(o.value);
    }

    double numeric(const std::vector<Value>& args, std::size_t i) { return args[i].asNumber(); }

    void expectArgs(const std::string& name, const std::vector<Value>& args, std::size_t least, std::size_t most) {

        if (args.size() < least || args.size() > most) {
            throw XacroError(name + "() takes " + std::to_string(least) + ".." + std::to_string(most) +
                             " arguments, got " + std::to_string(args.size()));
        }
    }

    Value callFunction(const std::string& name, std::vector<Value> args,
                       const std::vector<std::pair<std::string, Value>>& kwargs,
                       const EvalContext& ctx) {

        if (!kwargs.empty() && name != "dict") {
            throw XacroError(name + "() does not take keyword arguments");
        }

        if (name == "dict") {
            Dict d;
            if (args.size() == 1 && args[0].isDict()) d = args[0].asDict();
            else if (!args.empty()) throw XacroError("dict() takes keyword arguments or a single dict");
            for (const auto& [k, v] : kwargs) d[k] = v;
            return Value(std::move(d));
        }

        if (name == "list") {
            expectArgs(name, args, 0, 1);
            List out;
            if (!args.empty()) {
                const Value& v = args[0];
                if (v.isList()) out = v.asList();
                else if (v.isDict()) {
                    for (const auto& entry : v.asDict()) out.emplace_back(entry.first);
                } else if (v.isString()) {
                    for (char c : v.asString()) out.emplace_back(std::string(1, c));
                } else {
                    throw XacroError("list() cannot iterate " + v.typeName());
                }
            }
            return Value(std::move(out));
        }

        if (name == "len") {
            expectArgs(name, args, 1, 1);
            const Value& v = args[0];
            if (v.isList()) return Value(static_cast<long long>(v.asList().size()));
            if (v.isDict()) return Value(static_cast<long long>(v.asDict().size()));
            if (v.isString()) return Value(static_cast<long long>(v.asString().size()));
            throw XacroError("len() is not defined for " + v.typeName());
        }

        if (name == "str") {
            expectArgs(name, args, 1, 1);
            return Value(args[0].toString());
        }

        if (name == "bool") {
            expectArgs(name, args, 1, 1);
            return Value(args[0].truthy());
        }

        if (name == "int") {
            expectArgs(name, args, 1, 1);
            const Value& v = args[0];
            if (v.isString()) {
                const Value c = classify(v.asString());
                if (!c.isNumber()) throw XacroError("int() cannot convert '" + v.asString() + "'");
                return Value(static_cast<long long>(c.asNumber()));
            }
            return Value(static_cast<long long>(v.asNumber()));
        }

        if (name == "float") {
            expectArgs(name, args, 1, 1);
            const Value& v = args[0];
            if (v.isString()) {
                const Value c = classify(v.asString());
                if (!c.isNumber()) throw XacroError("float() cannot convert '" + v.asString() + "'");
                return Value(c.asNumber());
            }
            return Value(v.asNumber());
        }

        if (name == "load_yaml") {
            expectArgs(name, args, 1, 1);
            std::filesystem::path path(args[0].isString() ? args[0].asString() : args[0].toString());
            if (path.is_relative() && !ctx.document.empty()) {
                path = ctx.document.parent_path() / path;
            }
            if (!std::filesystem::exists(path)) {
                throw XacroError("load_yaml: no such file '" + path.string() + "'");
            }
            return loadYamlFile(path);
        }

        if (name == "min" || name == "max") {
            if (args.size() == 1 && args[0].isList()) {
                const List l = args[0].asList();
                if (l.empty()) throw XacroError(name + "() of an empty list");
                args.assign(l.begin(), l.end());
            }
            if (args.empty()) throw XacroError(name + "() needs at least one argument");
            Value best = args[0];
            for (std::size_t i = 1; i < args.size(); ++i) {
                const int c = Value::compare(args[i], best);
                if ((name == "min" && c < 0) || (name == "max" && c > 0)) best = args[i];
            }
            return best;
        }

        if (name == "abs" || name == "fabs") {
            expectArgs(name, args, 1, 1);
            if (name == "abs" && isIntegral(args[0])) {
                const long long v = args[0].asInt();
                return Value(v < 0 ? -v : v);
            }
            return Value(std::fabs(numeric(args, 0)));
        }

        if (name == "floor" || name == "ceil") {
            expectArgs(name, args, 1, 1);
            const double d = name == "floor" ? std::floor(numeric(args, 0)) : std::ceil(numeric(args, 0));
            return Value(static_cast<long long>(d));
        }

        if (name == "round") {
            expectArgs(name, args, 1, 2);
            const double d = numeric(args, 0);
            if (args.size() == 1) return Value(static_cast<long long>(std::round(d)));
            const double scale = std::pow(10.0, static_cast<double>(args[1].asInt()));
            return Value(std::round(d * scale) / scale);
        }

        if (name == "atan2") {
            expectArgs(name, args, 2, 2);
            return Value(std::atan2(numeric(args, 0), numeric(args, 1)));
        }

        if (name == "pow") {
            expectArgs(name, args, 2, 2);
            return arithmetic("**", args[0], args[1]);
        }

        if (name == "log") {
            expectArgs(name, args, 1, 2);
            if (args.size() == 2) return Value(std::log(numeric(args, 0)) / std::log(numeric(args, 1)));
            return Value(std::log(numeric(args, 0)));
        }

        expectArgs(name, args, 1, 1);
        const double x = numeric(args, 0);

        if (name == "sin") return Value(std::sin(x));
        if (name == "cos") return Value(std::cos(x));
        if (name == "tan") return Value(std::tan(x));
        if (name == "asin") return Value(std::asin(x));
        if (name == "acos") return Value(std::acos(x));
        if (name == "atan") return Value(std::atan(x));
        if (name == "sqrt") return Value(std::sqrt(x));
        if (name == "exp") return Value(std::exp(x));
        if (name == "log10") return Value(std::log10(x));
        if (name == "radians") return Value(x * kPi / 180.0);
        if (name == "degrees") return Value(x * 180.0 / kPi);

        throw XacroError("unknown function '" + name + "'");
    }

    Operand resolveName(const std::string& name, const EvalContext& ctx) {

        if (name == "True") return {Operand::Kind::Val, Value(true), {}};
        if (name == "False") return {Operand::Kind::Val, Value(false), {}};
        if (name == "None") return {Operand::Kind::Val, Value{}, {}};

        if (ctx.scope) {
            if (const Value* v = ctx.scope->find(name)) return {Operand::Kind::Val, *v, {}};
        }

        if (name == "true") return {Operand::Kind::Val, Value(true), {}};
        if (name == "false") return {Operand::Kind::Val, Value(false), {}};

        if (name == "pi") return {Operand::Kind::Val, Value(kPi), {}};
        if (name == "tau") return {Operand::Kind::Val, Value(2.0 * kPi), {}};
        if (name == "e") return {Operand::Kind::Val, Value(kE), {}};

        if (name == "math" || name == "xacro") return {Operand::Kind::Module, Value{}, name};

        if (functionNames().count(name)) return {Operand::Kind::Func, Value{}, name};

        if (ctx.argsAsProperties && ctx.args) {
            if (const auto it = ctx.args->find(name); it != ctx.args->end()) {
                if (ctx.diags) {
                    ctx.diags->warn("'" + name + "' is not a property; using the argument of that name",
                                    ctx.document, ctx.locator ? ctx.locator->currentLine() : 0);
                }
                // Arguments are strings; as a property the text gets the type it looks like,
                // so `${count + 1}` keeps working for a numeric argument.
                const Value& v = it->second;
                return {Operand::Kind::Val, v.isString() ? classify(v.asString()) : v, {}};
            }
        }

        throw XacroError("undefined name '" + name + "'");
    }

    Operand resolveAttr(const Operand& base, const std::string& attr) {

        // What load_yaml returns answers to both `d['k']` and `d.k`, the way the dict
        // wrapper python xacro installs on its loader does. Descriptions lean on it:
        // franka reads `link_inertials.origin.rpy` straight out of its inertia file.
        if (base.kind == Operand::Kind::Val && base.value.isDict()) {
            const auto& dict = base.value.asDict();
            const auto found = dict.find(attr);
            if (found == dict.end()) throw XacroError("no member '" + attr + "' in the mapping");
            return {Operand::Kind::Val, found->second, {}};
        }

        if (base.kind != Operand::Kind::Module) {
            throw XacroError("attribute access is only supported on 'math', 'xacro' and mappings");
        }

        if (base.name == "math") {
            if (attr == "pi") return {Operand::Kind::Val, Value(kPi), {}};
            if (attr == "tau") return {Operand::Kind::Val, Value(2.0 * kPi), {}};
            if (attr == "e") return {Operand::Kind::Val, Value(kE), {}};
            if (functionNames().count(attr)) return {Operand::Kind::Func, Value{}, attr};
            throw XacroError("math has no member '" + attr + "'");
        }

        if (attr == "load_yaml") return {Operand::Kind::Func, Value{}, "load_yaml"};
        throw XacroError("xacro has no member '" + attr + "'");
    }

    Operand evalNode(const Node& n, const EvalContext& ctx) {

        switch (n.kind) {

            case Node::Kind::Literal:
                return {Operand::Kind::Val, n.literal, {}};

            case Node::Kind::Name:
                return resolveName(n.text, ctx);

            case Node::Kind::Attr:
                return resolveAttr(evalNode(*n.kids[0], ctx), n.text);

            case Node::Kind::Index:
                return {Operand::Kind::Val,
                        subscript(evalValue(*n.kids[0], ctx), evalValue(*n.kids[1], ctx)), {}};

            case Node::Kind::ListLit: {
                List out;
                out.reserve(n.kids.size());
                for (const auto& k : n.kids) out.push_back(evalValue(*k, ctx));
                return {Operand::Kind::Val, Value(std::move(out)), {}};
            }

            case Node::Kind::Call: {
                const Operand callee = evalNode(*n.kids[0], ctx);
                if (callee.kind != Operand::Kind::Func) {
                    throw XacroError("this expression is not callable");
                }
                std::vector<Value> args;
                for (std::size_t i = 0; i < n.positional; ++i) {
                    args.push_back(evalValue(*n.kids[1 + i], ctx));
                }
                std::vector<std::pair<std::string, Value>> kwargs;
                for (std::size_t i = 0; i < n.kwNames.size(); ++i) {
                    kwargs.emplace_back(n.kwNames[i], evalValue(*n.kids[1 + n.positional + i], ctx));
                }
                return {Operand::Kind::Val, callFunction(callee.name, std::move(args), kwargs, ctx), {}};
            }

            case Node::Kind::Unary: {
                const Value v = evalValue(*n.kids[0], ctx);
                if (!v.isNumber()) throw XacroError("cannot negate " + v.typeName());
                if (n.text == "+") return {Operand::Kind::Val, isIntegral(v) ? Value(v.asInt()) : v, {}};
                return {Operand::Kind::Val,
                        isIntegral(v) ? Value(-v.asInt()) : Value(-v.asNumber()), {}};
            }

            case Node::Kind::Binary:
                return {Operand::Kind::Val,
                        arithmetic(n.text, evalValue(*n.kids[0], ctx), evalValue(*n.kids[1], ctx)), {}};

            case Node::Kind::Not:
                return {Operand::Kind::Val, Value(!evalValue(*n.kids[0], ctx).truthy()), {}};

            case Node::Kind::BoolOp: {
                Value lhs = evalValue(*n.kids[0], ctx);
                const bool t = lhs.truthy();
                if (n.text == "or") {
                    if (t) return {Operand::Kind::Val, std::move(lhs), {}};
                } else if (!t) {
                    return {Operand::Kind::Val, std::move(lhs), {}};
                }
                return {Operand::Kind::Val, evalValue(*n.kids[1], ctx), {}};
            }

            case Node::Kind::Compare: {
                Value lhs = evalValue(*n.kids[0], ctx);
                for (std::size_t i = 0; i < n.ops.size(); ++i) {
                    Value rhs = evalValue(*n.kids[i + 1], ctx);
                    if (!compareOp(n.ops[i], lhs, rhs)) return {Operand::Kind::Val, Value(false), {}};
                    lhs = std::move(rhs);
                }
                return {Operand::Kind::Val, Value(true), {}};
            }

            default: {
                const bool cond = evalValue(*n.kids[0], ctx).truthy();
                return {Operand::Kind::Val, evalValue(*n.kids[cond ? 1 : 2], ctx), {}};
            }
        }
    }

}// namespace

Value threepp::xacro::evaluate(std::string_view expression, const EvalContext& ctx) {

    try {
        Lexer lexer(expression);
        Parser parser(lexer.run());
        const NodePtr ast = parser.run();
        return evalValue(*ast, ctx);

    } catch (const XacroError& e) {
        throw XacroError(e.message() + " (in expression \"" + std::string(expression) + "\")",
                         e.document().empty() ? ctx.document : e.document());
    }
}
