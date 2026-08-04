
#include "threepp/loaders/xacro/YamlLite.hpp"

#include "threepp/loaders/xacro/Diagnostics.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    struct Line {
        int indent{};
        int number{};
        std::string text;
    };

    std::string trim(const std::string& s) {

        const auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\r'; };
        std::size_t b = 0;
        std::size_t e = s.size();
        while (b < e && isSpace(s[b])) ++b;
        while (e > b && isSpace(s[e - 1])) --e;
        return s.substr(b, e - b);
    }

    bool parseFullInt(const std::string& s, long long& out) {

        std::string_view v(s);
        if (v.empty()) return false;
        if (v.front() == '+') v.remove_prefix(1);
        if (v.empty()) return false;
        auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
        return ec == std::errc{} && ptr == v.data() + v.size();
    }

    bool parseFullDouble(const std::string& s, double& out) {

        std::string_view v(s);
        if (v.empty()) return false;
        if (v.front() == '+') v.remove_prefix(1);
        if (v.empty()) return false;
#if defined(__cpp_lib_to_chars)
        auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
        return ec == std::errc{} && ptr == v.data() + v.size();
#else
        // libc++ without floating-point from_chars (see ColladaLoader.cpp).
        const std::string copy(v);
        char* next = nullptr;
        out = std::strtod(copy.c_str(), &next);
        return next == copy.c_str() + copy.size();
#endif
    }

    class Parser {

    public:
        Parser(std::string_view text, std::filesystem::path document)
            : document_(std::move(document)) {

            scan(text);
        }

        Value parse() {

            if (lines_.empty()) return Value{};

            std::size_t i = 0;
            Value v = parseBlock(i, lines_[0].indent);
            if (i < lines_.size()) {
                fail("unexpected indentation", lines_[i].number);
            }
            return v;
        }

    private:
        [[noreturn]] void fail(const std::string& what, int line) const {

            throw XacroError("YAML line " + std::to_string(line) + ": " + what, document_);
        }

        void scan(std::string_view text) {

            int number = 0;
            bool sawDocumentStart = false;
            std::size_t pos = 0;

            while (pos <= text.size()) {

                const std::size_t nl = text.find('\n', pos);
                std::string_view raw = text.substr(pos, nl == std::string_view::npos ? std::string_view::npos : nl - pos);
                pos = nl == std::string_view::npos ? text.size() + 1 : nl + 1;
                ++number;

                if (!raw.empty() && raw.back() == '\r') raw.remove_suffix(1);

                int indent = 0;
                std::size_t k = 0;
                while (k < raw.size() && (raw[k] == ' ' || raw[k] == '\t')) {
                    if (raw[k] == '\t') fail("tabs are not allowed in indentation", number);
                    ++indent;
                    ++k;
                }

                std::string content = stripComment(std::string(raw.substr(k)));
                content = trim(content);
                if (content.empty()) continue;

                if (content == "---") {
                    if (sawDocumentStart || !lines_.empty()) {
                        fail("multi-document YAML is not supported", number);
                    }
                    sawDocumentStart = true;
                    continue;
                }
                if (content == "...") break;
                if (content[0] == '%') fail("YAML directives are not supported", number);

                lines_.push_back(Line{indent, number, content});
            }
        }

        static std::string stripComment(const std::string& s) {

            char quote = 0;
            for (std::size_t i = 0; i < s.size(); ++i) {
                const char c = s[i];
                if (quote) {
                    if (c == '\\' && quote == '"' && i + 1 < s.size()) {
                        ++i;
                    } else if (c == quote) {
                        quote = 0;
                    }
                } else if (c == '\'' || c == '"') {
                    quote = c;
                } else if (c == '#' && (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\t')) {
                    return s.substr(0, i);
                }
            }
            return s;
        }

        static bool isSequenceEntry(const std::string& s) {

            return !s.empty() && s[0] == '-' && (s.size() == 1 || s[1] == ' ');
        }

        Value parseBlock(std::size_t& i, int indent) {

            const std::string& text = lines_[i].text;
            if (isSequenceEntry(text)) return parseSequence(i, indent);

            if (findKeySeparator(text) == std::string::npos) {
                const Value v = parseScalarOrFlow(text, lines_[i].number);
                ++i;
                return v;
            }
            return parseMapping(i, indent);
        }

        Value parseSequence(std::size_t& i, int indent) {

            List out;
            while (i < lines_.size() && lines_[i].indent == indent && isSequenceEntry(lines_[i].text)) {

                const std::string body = lines_[i].text.substr(1);
                std::size_t k = 0;
                while (k < body.size() && body[k] == ' ') ++k;

                if (k >= body.size()) {
                    ++i;
                    if (i < lines_.size() && lines_[i].indent > indent) {
                        out.push_back(parseBlock(i, lines_[i].indent));
                    } else {
                        out.emplace_back();
                    }
                } else {
                    const int inner = indent + 1 + static_cast<int>(k);
                    lines_[i].indent = inner;
                    lines_[i].text = body.substr(k);
                    out.push_back(parseBlock(i, inner));
                }
            }
            return Value(std::move(out));
        }

        Value parseMapping(std::size_t& i, int indent) {

            Dict out;
            while (i < lines_.size() && lines_[i].indent == indent) {

                const Line& line = lines_[i];
                if (isSequenceEntry(line.text)) fail("unexpected sequence entry in a mapping", line.number);

                const std::size_t sep = findKeySeparator(line.text);
                if (sep == std::string::npos) fail("expected 'key: value'", line.number);

                const std::string key = unquote(trim(line.text.substr(0, sep)), line.number);
                const std::string rest = trim(line.text.substr(sep + 1));
                const int number = line.number;

                // `key: &name` with nothing after it names the block below, the way a bare
                // `key:` introduces one — the anchor belongs to the whole collection.
                const std::string anchor = rest.size() > 1 && rest[0] == '&' &&
                                                           rest.find_first_of(" \t") == std::string::npos
                                                   ? rest.substr(1)
                                                   : std::string{};

                Value value;

                if (rest.empty() || !anchor.empty()) {
                    ++i;
                    if (i < lines_.size() &&
                        (lines_[i].indent > indent ||
                         (lines_[i].indent == indent && isSequenceEntry(lines_[i].text)))) {
                        value = parseBlock(i, lines_[i].indent);
                    }
                    if (!anchor.empty()) anchors_[anchor] = value;
                } else {
                    value = parseScalarOrFlow(rest, number);
                    ++i;
                }

                // `<<: *base` (or a block) merges the mapping in. Left as a key called "<<"
                // it would read as a member of the robot, which is worse than an error. What
                // the document says itself outranks what it merged in, whichever came first
                // in the file - emplace declines to overwrite, a later key overwrites below.
                if (key == "<<") {
                    if (!value.isDict()) fail("a merge key needs a mapping", number);
                    for (const auto& [name, member] : value.asDict()) out.emplace(name, member);
                    continue;
                }
                out[key] = std::move(value);
            }
            return Value(std::move(out));
        }

        static std::size_t findKeySeparator(const std::string& s) {

            char quote = 0;
            int depth = 0;
            for (std::size_t i = 0; i < s.size(); ++i) {
                const char c = s[i];
                if (quote) {
                    if (c == '\\' && quote == '"' && i + 1 < s.size()) {
                        ++i;
                    } else if (c == quote) {
                        quote = 0;
                    }
                } else if (c == '\'' || c == '"') {
                    quote = c;
                } else if (c == '[' || c == '{') {
                    ++depth;
                } else if (c == ']' || c == '}') {
                    --depth;
                } else if (c == ':' && depth == 0 && (i + 1 == s.size() || s[i + 1] == ' ')) {
                    return i;
                }
            }
            return std::string::npos;
        }

        std::string unquote(const std::string& s, int line) const {

            if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
                std::string out;
                for (std::size_t i = 1; i + 1 < s.size(); ++i) {
                    if (s[i] == '\'' && i + 2 < s.size() && s[i + 1] == '\'') ++i;
                    out += s[i];
                }
                return out;
            }
            if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
                std::string out;
                for (std::size_t i = 1; i + 1 < s.size(); ++i) {
                    if (s[i] == '\\' && i + 2 < s.size()) {
                        const char e = s[++i];
                        switch (e) {
                            case 'n': out += '\n'; break;
                            case 't': out += '\t'; break;
                            case 'r': out += '\r'; break;
                            case '0': out += '\0'; break;
                            case '\\': out += '\\'; break;
                            case '"': out += '"'; break;
                            default: out += e; break;
                        }
                    } else {
                        out += s[i];
                    }
                }
                return out;
            }
            if (!s.empty() && (s[0] == '&' || s[0] == '*')) {
                fail("anchors and aliases are not supported", line);
            }
            if (!s.empty() && s[0] == '!') {
                fail("tags are not supported", line);
            }
            return s;
        }

        Value parseScalarOrFlow(const std::string& text, int line) {

            if (text == "|" || text == ">" || text == "|-" || text == ">-" ||
                text == "|+" || text == ">+") {
                fail("block scalars are not supported", line);
            }

            // `*name` stands for whatever `&name` introduced. Anchors are resolved as the
            // document is read, so an alias can only name something already defined -
            // which is all YAML allows anyway.
            if (text[0] == '*') {
                const std::string name = trim(text.substr(1));
                const auto found = anchors_.find(name);
                if (found == anchors_.end()) fail("no anchor named '" + name + "'", line);
                return found->second;
            }
            if (text[0] == '&') {
                const std::size_t k = text.find_first_of(" \t");
                const std::string name = text.substr(1, k == std::string::npos ? k : k - 1);
                if (name.empty()) fail("an anchor needs a name", line);

                Value value;
                if (k != std::string::npos) value = parseScalarOrFlow(trim(text.substr(k)), line);
                anchors_[name] = value;
                return value;
            }

            if (text[0] == '[' || text[0] == '{') {
                std::size_t p = 0;
                Value v = parseFlow(text, p, line);
                while (p < text.size() && text[p] == ' ') ++p;
                if (p != text.size()) fail("trailing characters after a flow collection", line);
                return v;
            }
            return scalar(text, line);
        }

        Value parseFlow(const std::string& s, std::size_t& p, int line) const {

            const auto skip = [&] { while (p < s.size() && s[p] == ' ') ++p; };

            skip();
            if (p >= s.size()) fail("unexpected end of a flow collection", line);

            if (s[p] == '[') {
                ++p;
                List out;
                skip();
                if (p < s.size() && s[p] == ']') {
                    ++p;
                    return Value(std::move(out));
                }
                while (true) {
                    out.push_back(parseFlow(s, p, line));
                    skip();
                    if (p < s.size() && s[p] == ',') {
                        ++p;
                        continue;
                    }
                    if (p < s.size() && s[p] == ']') {
                        ++p;
                        return Value(std::move(out));
                    }
                    fail("expected ',' or ']' in a flow sequence", line);
                }
            }

            if (s[p] == '{') {
                ++p;
                Dict out;
                skip();
                if (p < s.size() && s[p] == '}') {
                    ++p;
                    return Value(std::move(out));
                }
                while (true) {
                    skip();
                    const std::string key = unquote(trim(readFlowToken(s, p, ":,}", line)), line);
                    skip();
                    if (p >= s.size() || s[p] != ':') fail("expected ':' in a flow mapping", line);
                    ++p;
                    out[key] = parseFlow(s, p, line);
                    skip();
                    if (p < s.size() && s[p] == ',') {
                        ++p;
                        continue;
                    }
                    if (p < s.size() && s[p] == '}') {
                        ++p;
                        return Value(std::move(out));
                    }
                    fail("expected ',' or '}' in a flow mapping", line);
                }
            }

            return scalar(trim(readFlowToken(s, p, ",]}", line)), line);
        }

        static std::string readFlowToken(const std::string& s, std::size_t& p, const char* stops, int) {

            std::string out;
            char quote = 0;
            while (p < s.size()) {
                const char c = s[p];
                if (quote) {
                    out += c;
                    if (c == '\\' && quote == '"' && p + 1 < s.size()) {
                        out += s[++p];
                    } else if (c == quote) {
                        quote = 0;
                    }
                    ++p;
                    continue;
                }
                if (c == '\'' || c == '"') {
                    quote = c;
                    out += c;
                    ++p;
                    continue;
                }
                bool stop = false;
                for (const char* q = stops; *q; ++q) {
                    if (c == *q) stop = true;
                }
                if (stop) break;
                out += c;
                ++p;
            }
            return out;
        }

        // The two tags python xacro registers on its yaml loader. Both yield a float in
        // radians; anything else is a tag we would silently get wrong, so it stays an error.
        Value tagged(const std::string& raw, int line) const {

            std::size_t k = 0;
            while (k < raw.size() && raw[k] != ' ' && raw[k] != '\t') ++k;

            const std::string tag = raw.substr(0, k);
            const std::string rest = trim(raw.substr(k));

            if (tag != "!degrees" && tag != "!radians") {
                fail("the '" + tag + "' tag is not supported", line);
            }
            if (rest.empty() || rest[0] == '!') {
                fail("the '" + tag + "' tag needs a number", line);
            }

            const Value inner = scalar(rest, line);
            if (!inner.isNumber() || inner.isBool()) {
                fail("the '" + tag + "' tag needs a number, got '" + rest + "'", line);
            }

            constexpr double pi = 3.14159265358979323846;
            constexpr double degToRad = pi / 180.0;

            return Value(tag == "!degrees" ? inner.asNumber() * degToRad : inner.asNumber());
        }

        Value scalar(const std::string& raw, int line) const {

            if (raw.size() >= 2 && (raw.front() == '\'' || raw.front() == '"') && raw.back() == raw.front()) {
                return Value(unquote(raw, line));
            }
            if (!raw.empty() && (raw[0] == '&' || raw[0] == '*')) {
                fail("anchors and aliases are not supported", line);
            }
            if (!raw.empty() && raw[0] == '!') return tagged(raw, line);

            if (raw.empty() || raw == "~" || raw == "null" || raw == "Null" || raw == "NULL") return Value{};
            if (raw == "true" || raw == "True" || raw == "TRUE") return Value(true);
            if (raw == "false" || raw == "False" || raw == "FALSE") return Value(false);

            if (raw == ".inf" || raw == ".Inf" || raw == ".INF" || raw == "+.inf") {
                return Value(std::numeric_limits<double>::infinity());
            }
            if (raw == "-.inf" || raw == "-.Inf" || raw == "-.INF") {
                return Value(-std::numeric_limits<double>::infinity());
            }
            if (raw == ".nan" || raw == ".NaN" || raw == ".NAN") {
                return Value(std::numeric_limits<double>::quiet_NaN());
            }

            long long i{};
            if (parseFullInt(raw, i)) return Value(i);

            double d{};
            if (parseFullDouble(raw, d)) return Value(d);

            return Value(raw);
        }

        std::filesystem::path document_;
        std::vector<Line> lines_;
        std::map<std::string, Value> anchors_;
    };

}// namespace

Value threepp::xacro::parseYaml(std::string_view text, const std::filesystem::path& document) {

    Parser parser(text, document);
    return parser.parse();
}

Value threepp::xacro::loadYamlFile(const std::filesystem::path& path) {

    std::ifstream in(path, std::ios::binary);
    if (!in) throw XacroError("cannot open YAML file '" + path.string() + "'");

    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    return parseYaml(text, path);
}
