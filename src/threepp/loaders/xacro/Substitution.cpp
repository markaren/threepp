
#include "threepp/loaders/xacro/Substitution.hpp"

#include "threepp/loaders/xacro/Expr.hpp"

#include <cstdlib>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    bool isSpace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

    std::string trim(std::string_view s) {

        std::size_t b = 0;
        std::size_t e = s.size();
        while (b < e && isSpace(s[b])) ++b;
        while (e > b && isSpace(s[e - 1])) --e;
        return std::string(s.substr(b, e - b));
    }

    // index of the closing delimiter for the opener at `open`, skipping quoted text
    std::size_t matchDelimiter(std::string_view s, std::size_t open, char opener, char closer,
                               const std::filesystem::path& document) {

        int depth = 0;
        char quote = 0;
        for (std::size_t i = open; i < s.size(); ++i) {
            const char c = s[i];
            if (quote) {
                if (c == '\\' && i + 1 < s.size()) ++i;
                else if (c == quote) quote = 0;
                continue;
            }
            if (c == '\'' || c == '"') {
                quote = c;
            } else if (c == opener) {
                ++depth;
            } else if (c == closer) {
                if (--depth == 0) return i;
            }
        }
        throw XacroError(std::string("unterminated '$") + opener + "' in \"" + std::string(s) + "\"", document);
    }

    EvalContext evalContext(const SubstCtx& ctx) {

        EvalContext e;
        e.scope = ctx.scope;
        e.args = ctx.args;
        e.argsAsProperties = ctx.argsAsProperties;
        e.document = ctx.document;
        e.diags = ctx.diags;
        e.locator = ctx.locator;
        return e;
    }

    // A value landing inside a string literal has to survive the expression lexer, which
    // reads backslashes as escapes. On Windows every resolved path is full of them.
    std::string escapedFor(std::string_view text, char quote) {

        std::string out;
        for (const char c : text) {
            if (c == '\\' || c == quote) out += '\\';
            out += c;
        }
        return out;
    }

    // $(find pkg) and $(arg id) may sit inside an expression, including inside a string
    // literal there: `${xacro.load_yaml('$(find pkg)/config/x.yaml')}` is how a ROS
    // description names a file it wants to read. Expand those before the expression is
    // parsed, the way xacro's own text evaluation recurses into what it finds.
    std::string expandInExpression(std::string_view raw, const SubstCtx& ctx) {

        std::string out;
        char quote = 0;

        for (std::size_t i = 0; i < raw.size();) {

            const char c = raw[i];

            if (c == '$' && i + 1 < raw.size() && (raw[i + 1] == '{' || raw[i + 1] == '(')) {
                const bool braced = raw[i + 1] == '{';

                // An opener with no closer is not a substitution. xacro's scanner only
                // matches complete patterns, so `${'cost is $(unknown'}` is a string with
                // a dollar in it, not an error about it.
                std::size_t end;
                try {
                    end = matchDelimiter(raw, i + 1, braced ? '{' : '(',
                                         braced ? '}' : ')', ctx.document);
                } catch (const XacroError&) {
                    out += c;
                    ++i;
                    continue;
                }

                const std::string text = substitute(raw.substr(i, end - i + 1), ctx).text;
                out += quote ? escapedFor(text, quote) : text;
                i = end + 1;
                continue;
            }

            if (quote) {
                if (c == '\\' && i + 1 < raw.size()) {
                    out += c;
                    out += raw[i + 1];
                    i += 2;
                    continue;
                }
                if (c == quote) quote = 0;
            } else if (c == '\'' || c == '"') {
                quote = c;
            }

            out += c;
            ++i;
        }

        return out;
    }

    // Only the ones that carry a substitution pay for the pass.
    Value evaluateText(std::string_view raw, const SubstCtx& ctx) {

        if (raw.find('$') == std::string_view::npos) return evaluate(raw, evalContext(ctx));
        return evaluate(expandInExpression(raw, ctx), evalContext(ctx));
    }

}// namespace

SubstResult threepp::xacro::substitute(std::string_view raw, const SubstCtx& ctx) {

    SubstResult result;
    std::string& out = result.text;

    std::size_t i = 0;
    while (i < raw.size()) {

        const char c = raw[i];
        if (c != '$' || i + 1 >= raw.size()) {
            out += c;
            ++i;
            continue;
        }

        const char next = raw[i + 1];

        if (next == '$') {
            out += '$';
            i += 2;
            continue;
        }

        if (next != '{' && next != '(') {
            out += c;
            ++i;
            continue;
        }

        const bool braced = next == '{';
        const std::size_t end = matchDelimiter(raw, i + 1, braced ? '{' : '(', braced ? '}' : ')', ctx.document);
        const std::string_view inner = raw.substr(i + 2, end - i - 2);
        const bool whole = i == 0 && end + 1 == raw.size();

        if (braced) {
            const Value v = evaluateText(inner, ctx);
            if (whole) result.whole = v;
            out += v.toString();
        } else {
            std::size_t k = 0;
            while (k < inner.size() && !isSpace(inner[k])) ++k;
            const std::string command(inner.substr(0, k));
            const std::string_view tail = inner.substr(k);

            if (command == "eval") {
                if (!whole) {
                    throw XacroError("$(eval ...) must be the whole value, but found it inside \"" +
                                             std::string(raw) + "\"",
                                     ctx.document);
                }
                const Value v = evaluateText(tail, ctx);
                result.whole = v;
                out += v.toString();

            } else {
                // nested substitutions run first, so $(find ${pkg}) resolves
                const std::string rest = substitute(tail, ctx).text;

                if (command == "arg") {
                    const std::string name = trim(rest);
                    if (!ctx.args) throw XacroError("$(arg " + name + "): no arguments available", ctx.document);
                    const auto it = ctx.args->find(name);
                    if (it == ctx.args->end()) throw XacroError("undefined arg '" + name + "'", ctx.document);
                    out += it->second.toString();

                } else if (command == "find") {
                    const std::string package = trim(rest);
                    if (!ctx.packages) throw XacroError("$(find " + package + "): no package resolver", ctx.document);
                    std::vector<std::string> tried;
                    const auto dir = ctx.packages->resolve(package, ctx.document, &tried);
                    if (!dir) {
                        std::string message = "cannot find package '" + package + "'; tried:";
                        for (const auto& t : tried) message += "\n  " + t;
                        throw XacroError(message, ctx.document);
                    }
                    out += dir->string();

                } else if (command == "env" || command == "optenv") {
                    std::string name = trim(rest);
                    std::string fallback;
                    if (command == "optenv") {
                        const std::size_t sp = name.find_first_of(" \t");
                        if (sp != std::string::npos) {
                            fallback = trim(std::string_view(name).substr(sp));
                            name = name.substr(0, sp);
                        }
                    }
                    const char* value = std::getenv(name.c_str());
                    if (value) {
                        out += value;
                    } else if (command == "optenv") {
                        out += fallback;
                    } else {
                        throw XacroError("environment variable '" + name + "' is not set", ctx.document);
                    }

                } else if (command == "dirname") {
                    if (ctx.document.empty()) {
                        throw XacroError("$(dirname) needs a document path", ctx.document);
                    }
                    out += ctx.document.parent_path().string();

                } else {
                    throw XacroError("unknown substitution command '" + command + "'", ctx.document);
                }
            }
        }

        i = end + 1;
    }

    return result;
}

std::string threepp::xacro::substituteText(std::string_view raw, const SubstCtx& ctx) {

    return substitute(raw, ctx).text;
}
