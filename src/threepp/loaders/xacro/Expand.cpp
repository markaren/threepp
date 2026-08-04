
#include "threepp/loaders/xacro/Expand.hpp"

#include "threepp/loaders/xacro/Expr.hpp"
#include "threepp/loaders/xacro/Scope.hpp"
#include "threepp/loaders/xacro/Substitution.hpp"

#include "pugixml.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <list>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

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

    std::string lower(std::string s) {

        for (auto& c : s) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        }
        return s;
    }

    bool startsWith(std::string_view s, std::string_view prefix) {

        return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
    }

    bool isXacroUri(std::string_view uri) {

        return uri == "http://ros.org/wiki/xacro" ||
               uri == "http://wiki.ros.org/xacro" ||
               uri == "http://www.ros.org/wiki/xacro";
    }

    std::string resolvePrefix(const pugi::xml_node& root) {

        for (const auto& attr : root.attributes()) {
            const std::string name = attr.name();
            if (startsWith(name, "xmlns:") && isXacroUri(attr.value())) return name.substr(6);
        }
        return "xacro";
    }

    // Which file a node physically sits in, and the prefix that file binds the xacro
    // namespace to. Both change across an include, so they travel with the node.
    struct DocCtx {
        std::filesystem::path path;
        std::string prefix{"xacro"};
        // Index of `path` in the expander's document list. A DocCtx dies with the frame that
        // made it, so an error being located after the stack has unwound asks for the file
        // by index instead of holding on to this one.
        std::size_t id = 0;
    };

    struct MacroParam {
        enum class Kind { Required,
                          Default,
                          Inherit,
                          InheritDefault,
                          Block,
                          BlockChildren };

        std::string name;
        Kind kind = Kind::Required;
        std::string def;
    };

    struct MacroDef {
        pugi::xml_node body;
        DocCtx doc;
        std::vector<MacroParam> params;
    };

    struct BlockArg {
        pugi::xml_node node;
        bool children = false;
        DocCtx doc;
    };

    // Splits a params attribute on whitespace, but keeps quoted text and ${...}/$(...)
    // spans together so `a:='two words'` and `b:=${1 + 2}` survive as one token.
    std::vector<std::string> splitParams(std::string_view raw) {

        std::vector<std::string> out;
        std::string token;
        int depth = 0;
        char quote = 0;

        for (std::size_t i = 0; i < raw.size(); ++i) {
            const char c = raw[i];

            if (quote) {
                token += c;
                if (c == '\\' && i + 1 < raw.size()) token += raw[++i];
                else if (c == quote) quote = 0;
                continue;
            }

            if (c == '\'' || c == '"') {
                quote = c;
                token += c;
                continue;
            }

            if (c == '$' && i + 1 < raw.size() && (raw[i + 1] == '{' || raw[i + 1] == '(')) {
                ++depth;
                token += c;
                token += raw[++i];
                continue;
            }

            if (depth > 0 && (c == '}' || c == ')')) {
                --depth;
                token += c;
                continue;
            }

            if (depth == 0 && isSpace(c)) {
                if (!token.empty()) out.push_back(std::exchange(token, {}));
                continue;
            }

            token += c;
        }

        if (!token.empty()) out.push_back(std::move(token));
        return out;
    }

    // The quotes in `rpy:='0 0 0'` group the default, they are not part of it. Leaving them
    // on writes `rpy="'0 0 0'"` into the URDF, which reads as three numbers to nobody.
    std::string unquoted(const std::string& s) {

        if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"') && s.back() == s.front()) {
            return s.substr(1, s.size() - 2);
        }
        return s;
    }

    // Where a parameter's default begins, and how long the separator is. ':=' is the form
    // xacro documents, but the older bare '=' is still out there — franka_description writes
    // `params="name prefix=${ee_prefix} rpy:='0 0 0'"` and means the same thing by both.
    // ':=' is looked for first so that its own '=' is not mistaken for the separator.
    std::pair<std::size_t, std::size_t> defaultSeparator(const std::string& token) {

        if (const auto walrus = token.find(":="); walrus != std::string::npos) return {walrus, 2};
        if (const auto plain = token.find('='); plain != std::string::npos) return {plain, 1};

        return {std::string::npos, 0};
    }

    std::vector<MacroParam> parseParams(std::string_view raw, const std::string& macro,
                                        const std::filesystem::path& document) {

        std::vector<MacroParam> params;

        for (const auto& token : splitParams(raw)) {

            MacroParam p;
            if (startsWith(token, "**")) {
                p.kind = MacroParam::Kind::BlockChildren;
                p.name = token.substr(2);
            } else if (startsWith(token, "*")) {
                p.kind = MacroParam::Kind::Block;
                p.name = token.substr(1);
            } else if (const auto sep = defaultSeparator(token); sep.first != std::string::npos) {
                p.name = token.substr(0, sep.first);
                const std::string rest = token.substr(sep.first + sep.second);
                if (rest == "^") {
                    p.kind = MacroParam::Kind::Inherit;
                } else if (startsWith(rest, "^|")) {
                    p.kind = MacroParam::Kind::InheritDefault;
                    p.def = unquoted(rest.substr(2));
                } else {
                    p.kind = MacroParam::Kind::Default;
                    p.def = unquoted(rest);
                }
            } else {
                p.name = token;
            }

            if (p.name.empty()) {
                throw XacroError("macro '" + macro + "': malformed parameter '" + token + "'", document);
            }
            params.push_back(std::move(p));
        }

        return params;
    }

    bool isBlockParam(const MacroParam& p) {

        return p.kind == MacroParam::Kind::Block || p.kind == MacroParam::Kind::BlockChildren;
    }

    // Where the lines of one document start, so pugixml's byte offset for a node can be
    // reported as the line an author would look for. Built once per file that an error
    // touches - a run that goes well never counts a newline.
    class LineIndex {

    public:
        explicit LineIndex(std::string_view text) {

            starts_.push_back(0);
            for (std::size_t i = 0; i < text.size(); ++i) {
                if (text[i] == '\n') starts_.push_back(i + 1);
            }
        }

        [[nodiscard]] std::size_t lineOf(std::size_t offset) const {

            const auto after = std::upper_bound(starts_.begin(), starts_.end(), offset);
            return static_cast<std::size_t>(after - starts_.begin());
        }

    private:
        std::vector<std::size_t> starts_;
    };

    // pugixml reports offsets into the buffer it parsed. That is the file byte for byte
    // for UTF-8, but a UTF-16 or UTF-32 file is transcoded first and the offsets no longer
    // point into anything we can read back, so those documents go without lines.
    bool transcoded(std::string_view text) {

        const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(text[i]); };

        if (text.size() >= 2 && ((byte(0) == 0xFF && byte(1) == 0xFE) || (byte(0) == 0xFE && byte(1) == 0xFF))) {
            return true;
        }
        return text.size() >= 4 && byte(0) == 0 && byte(1) == 0;
    }

    std::optional<std::string> readText(const std::filesystem::path& path) {

        std::ifstream in(path, std::ios::binary);
        if (!in) return std::nullopt;

        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    std::string joinNames(const std::map<std::string, MacroDef>& macros) {

        if (macros.empty()) return "none are defined";

        std::string out;
        for (const auto& entry : macros) {
            if (!out.empty()) out += ", ";
            out += entry.first;
        }
        return out;
    }


    class Expander: public Locator {

    public:
        Expander(const ExpandInputs& inputs, Diagnostics& diags)
            : inputs_(inputs), diags_(diags), args_(inputs.args) {}

        void run(const pugi::xml_node& root, pugi::xml_document& out);

        // Pin an error to the last node the run had entered - the innermost one, since a
        // node is marked on the way in and nothing clears it on the way out.
        void locate(XacroError& e);

        [[nodiscard]] std::size_t currentLine() const override;

    private:
        void warn(const std::string& message, const DocCtx& doc) {
            // The line is counted in the file the run is standing in; anything else gets none.
            diags_.warn(message, doc.path, doc.id == hereDoc_ ? currentLine() : 0);
        }

        [[nodiscard]] SubstResult subst(std::string_view raw, const DocCtx& doc) const;

        [[nodiscard]] Value typed(std::string_view raw, const DocCtx& doc) const;

        void processChildren(const pugi::xml_node& src, pugi::xml_node dst, const DocCtx& doc);

        void processNode(const pugi::xml_node& node, pugi::xml_node dst, const DocCtx& doc);

        void copyAttributes(const pugi::xml_node& src, pugi::xml_node dst, const DocCtx& doc);

        void handleXacro(const std::string& local, const pugi::xml_node& node,
                         pugi::xml_node dst, const DocCtx& doc);

        void doProperty(const pugi::xml_node& node, const DocCtx& doc);
        void doArg(const pugi::xml_node& node, const DocCtx& doc);
        void doMacro(const pugi::xml_node& node, const DocCtx& doc);
        void doInclude(const pugi::xml_node& node, pugi::xml_node dst, const DocCtx& doc);
        void doInsertBlock(const pugi::xml_node& node, pugi::xml_node dst, const DocCtx& doc);
        void instantiate(const std::string& name, const pugi::xml_node& call,
                         pugi::xml_node dst, const DocCtx& doc);

        [[nodiscard]] bool condition(const pugi::xml_node& node, const DocCtx& doc);

        std::size_t noteDocument(const std::filesystem::path& path);

        void mark(const pugi::xml_node& node, const DocCtx& doc) {
            here_ = node;
            hereDoc_ = doc.id;
        }

        const LineIndex* linesOf(const std::filesystem::path& path) const;

        const ExpandInputs& inputs_;
        Diagnostics& diags_;

        Scope scope_;
        std::map<std::string, Value> args_;
        std::map<std::string, MacroDef> macros_;
        std::map<std::string, BlockArg> blocks_;
        std::vector<std::filesystem::path> includeStack_;
        std::list<pugi::xml_document> owned_;

        // Every document the run has opened, and where it had got to. The nodes stay valid
        // because the documents they belong to are owned here or by the caller.
        std::vector<std::filesystem::path> documents_;
        mutable std::map<std::filesystem::path, std::optional<LineIndex>> lines_;
        pugi::xml_node here_;
        std::size_t hereDoc_ = 0;

        std::size_t outputElements_ = 0;
        std::size_t instantiations_ = 0;
        std::size_t macroDepth_ = 0;
    };

    SubstResult Expander::subst(std::string_view raw, const DocCtx& doc) const {

        SubstCtx ctx;
        ctx.scope = &scope_;
        ctx.args = &args_;
        ctx.packages = inputs_.packages;
        ctx.document = doc.path;
        ctx.diags = &diags_;
        ctx.locator = this;
        ctx.argsAsProperties = inputs_.argsAsProperties;
        return substitute(raw, ctx);
    }

    Value Expander::typed(std::string_view raw, const DocCtx& doc) const {

        const auto result = subst(raw, doc);
        return result.whole ? *result.whole : classify(result.text);
    }

    void Expander::run(const pugi::xml_node& root, pugi::xml_document& out) {

        DocCtx doc{inputs_.document, resolvePrefix(root), noteDocument(inputs_.document)};

        std::error_code ec;
        auto canonical = std::filesystem::weakly_canonical(inputs_.document, ec);
        includeStack_.push_back(ec ? inputs_.document : canonical);

        auto outRoot = out.append_child(root.name());

        mark(root, doc);
        processChildren(root, outRoot, doc);

        // The root's own attributes go last: `<robot name="$(arg name)">` is the standard
        // idiom and the <xacro:arg> that gives `name` a default is one of its children.
        mark(root, doc);
        copyAttributes(root, outRoot, doc);
    }

    void Expander::processChildren(const pugi::xml_node& src, pugi::xml_node dst, const DocCtx& doc) {

        for (const auto& child : src.children()) {
            processNode(child, dst, doc);
        }
    }

    std::size_t Expander::noteDocument(const std::filesystem::path& path) {

        documents_.push_back(path);
        return documents_.size() - 1;
    }

    const LineIndex* Expander::linesOf(const std::filesystem::path& path) const {

        if (const auto it = lines_.find(path); it != lines_.end()) {
            return it->second ? &*it->second : nullptr;
        }

        auto& slot = lines_[path];

        // The root document may have been parsed from a string the caller still holds;
        // everything else was read from disk and can be read again.
        const std::string_view given = path == inputs_.document ? inputs_.source : std::string_view{};
        const auto text = given.empty() ? readText(path) : std::nullopt;
        const std::string_view source = given.empty() ? (text ? std::string_view(*text) : std::string_view{}) : given;

        if (!source.empty() && !transcoded(source)) slot.emplace(source);

        return slot ? &*slot : nullptr;
    }

    std::size_t Expander::currentLine() const {

        if (!here_ || hereDoc_ >= documents_.size()) return 0;

        const auto offset = here_.offset_debug();
        if (offset < 0) return 0;

        const auto* lines = linesOf(documents_[hereDoc_]);
        return lines ? lines->lineOf(static_cast<std::size_t>(offset)) : 0;
    }

    void Expander::locate(XacroError& e) {

        if (e.line() != 0 || hereDoc_ >= documents_.size()) return;

        const auto& path = documents_[hereDoc_];
        if (!e.document().empty() && e.document() != path) return;

        if (const auto line = currentLine()) e.locate(path, line);
    }

    void Expander::processNode(const pugi::xml_node& node, pugi::xml_node dst, const DocCtx& doc) {

        // Marking is all the bookkeeping an error needs, and it is two stores. Catching per
        // node instead would rethrow once per frame, and a rethrow re-enters the unwinder:
        // a macro nested to its budget then dies on the stack rather than on the budget.
        mark(node, doc);

        switch (node.type()) {
            case pugi::node_pcdata:
            case pugi::node_cdata:
                dst.append_child(node.type()).set_value(subst(node.value(), doc).text.c_str());
                return;
            case pugi::node_element:
                break;
            default:
                return;
        }

        const std::string name = node.name();
        const std::string tag = doc.prefix + ":";

        if (startsWith(name, tag)) {
            handleXacro(name.substr(tag.size()), node, dst, doc);
            return;
        }

        if (++outputElements_ > inputs_.budget.maxOutputElements) {
            throw XacroError("expansion produced more than " +
                                     std::to_string(inputs_.budget.maxOutputElements) + " elements",
                             doc.path);
        }

        auto copy = dst.append_child(node.name());
        copyAttributes(node, copy, doc);
        processChildren(node, copy, doc);
    }

    void Expander::copyAttributes(const pugi::xml_node& src, pugi::xml_node dst, const DocCtx& doc) {

        const std::string reserved = doc.prefix + ":";

        for (const auto& attr : src.attributes()) {
            const std::string name = attr.name();
            if (startsWith(name, "xmlns:") && isXacroUri(attr.value())) continue;
            if (startsWith(name, reserved)) continue;
            dst.append_attribute(attr.name()) = subst(attr.value(), doc).text.c_str();
        }
    }

    void Expander::handleXacro(const std::string& local, const pugi::xml_node& node,
                               pugi::xml_node dst, const DocCtx& doc) {

        if (local == "property") {
            doProperty(node, doc);

        } else if (local == "arg") {
            doArg(node, doc);

        } else if (local == "macro") {
            doMacro(node, doc);

        } else if (local == "include") {
            doInclude(node, dst, doc);

        } else if (local == "insert_block") {
            doInsertBlock(node, dst, doc);

        } else if (local == "if") {
            if (condition(node, doc)) processChildren(node, dst, doc);

        } else if (local == "unless") {
            if (!condition(node, doc)) processChildren(node, dst, doc);

        } else {
            instantiate(local, node, dst, doc);
        }
    }

    void Expander::doProperty(const pugi::xml_node& node, const DocCtx& doc) {

        const auto nameAttr = node.attribute("name");
        if (!nameAttr) throw XacroError("xacro:property without a name", doc.path);

        const std::string name = trim(subst(nameAttr.value(), doc).text);
        if (name.empty()) throw XacroError("xacro:property with an empty name", doc.path);

        for (const auto& child : node.children()) {
            if (child.type() == pugi::node_element) {
                throw XacroError("xacro:property '" + name +
                                         "': a block body is not supported, use value=\"...\"",
                                 doc.path);
            }
        }

        const auto valueAttr = node.attribute("value");
        const auto defaultAttr = node.attribute("default");

        if (valueAttr && defaultAttr) {
            throw XacroError("xacro:property '" + name + "': value and default are mutually exclusive",
                             doc.path);
        }
        if (!valueAttr && defaultAttr && scope_.has(name)) return;

        const auto source = valueAttr ? valueAttr : defaultAttr;
        const Value value = source ? typed(source.value(), doc) : Value(std::string{});

        const std::string where = node.attribute("scope").as_string("local");

        if (where == "local") {
            scope_.set(name, value);
        } else if (where == "parent") {
            scope_.setParent(name, value);
        } else if (where == "global") {
            scope_.setGlobal(name, value);
        } else {
            throw XacroError("xacro:property '" + name + "': unknown scope '" + where +
                                     "' (expected local, parent or global)",
                             doc.path);
        }
    }

    void Expander::doArg(const pugi::xml_node& node, const DocCtx& doc) {

        const auto nameAttr = node.attribute("name");
        if (!nameAttr) throw XacroError("xacro:arg without a name", doc.path);

        const std::string name = trim(subst(nameAttr.value(), doc).text);
        if (name.empty()) throw XacroError("xacro:arg with an empty name", doc.path);

        if (args_.count(name)) return;

        // Without a default the argument stays undeclared, so $(arg name) reports it as
        // missing rather than quietly handing out an empty string.
        if (const auto defaultAttr = node.attribute("default")) {
            args_[name] = Value(subst(defaultAttr.value(), doc).text);
        }
    }

    void Expander::doMacro(const pugi::xml_node& node, const DocCtx& doc) {

        const auto nameAttr = node.attribute("name");
        if (!nameAttr) throw XacroError("xacro:macro without a name", doc.path);

        const std::string name = trim(subst(nameAttr.value(), doc).text);
        if (name.empty()) throw XacroError("xacro:macro with an empty name", doc.path);

        MacroDef def;
        def.body = node;
        def.doc = doc;
        def.params = parseParams(node.attribute("params").value(), name, doc.path);

        if (macros_.count(name)) {
            warn("macro '" + name + "' redefined", doc);
        }
        macros_[name] = std::move(def);
    }

    void Expander::doInclude(const pugi::xml_node& node, pugi::xml_node dst, const DocCtx& doc) {

        if (node.attribute("ns")) {
            throw XacroError("xacro:include: the 'ns' attribute is not supported", doc.path);
        }

        const auto fileAttr = node.attribute("filename");
        if (!fileAttr) throw XacroError("xacro:include without a filename", doc.path);

        std::string filename = trim(subst(fileAttr.value(), doc).text);
        if (filename.empty()) throw XacroError("xacro:include with an empty filename", doc.path);

        if (startsWith(filename, "package://")) {
            const std::string rest = filename.substr(10);
            const auto slash = rest.find('/');
            const std::string package = rest.substr(0, slash);
            if (!inputs_.packages) {
                throw XacroError("cannot resolve '" + filename + "': no package resolver", doc.path);
            }
            std::vector<std::string> tried;
            const auto dir = inputs_.packages->resolve(package, doc.path, &tried);
            if (!dir) {
                std::string message = "cannot find package '" + package + "'; tried:";
                for (const auto& t : tried) message += "\n  " + t;
                throw XacroError(message, doc.path);
            }
            filename = (slash == std::string::npos ? *dir : *dir / rest.substr(slash + 1)).string();
        }

        std::filesystem::path file(filename);
        if (file.is_relative()) file = (doc.path.parent_path() / file).lexically_normal();

        // The canonical form is for recognising a file we are already inside — it resolves
        // symlinks and, on Windows, 8.3 short names. It is not the document's identity:
        // $(dirname) and error messages keep the spelling the include actually used.
        std::error_code ec;
        auto canonical = std::filesystem::weakly_canonical(file, ec);
        if (ec) canonical = file;

        if (std::find(includeStack_.begin(), includeStack_.end(), canonical) != includeStack_.end()) {
            std::string chain;
            for (const auto& p : includeStack_) chain += "\n  " + p.string();
            chain += "\n  " + canonical.string();
            throw XacroError("include cycle:" + chain, doc.path);
        }

        if (includeStack_.size() >= inputs_.budget.maxIncludeDepth) {
            throw XacroError("includes nested more than " +
                                     std::to_string(inputs_.budget.maxIncludeDepth) + " deep",
                             doc.path);
        }

        if (!std::filesystem::is_regular_file(file, ec)) {
            throw XacroError("cannot include '" + file.string() + "': no such file", doc.path);
        }

        auto& included = owned_.emplace_back();
        const auto parsed = included.load_file(file.string().c_str());
        if (!parsed) {
            throw XacroError("cannot parse '" + file.string() + "': " + parsed.description(), doc.path);
        }

        const auto root = included.document_element();
        if (!root) throw XacroError("'" + file.string() + "' has no root element", doc.path);

        const DocCtx sub{file, resolvePrefix(root), noteDocument(file)};

        includeStack_.push_back(canonical);
        processChildren(root, dst, sub);
        includeStack_.pop_back();
    }

    void Expander::doInsertBlock(const pugi::xml_node& node, pugi::xml_node dst, const DocCtx& doc) {

        const auto nameAttr = node.attribute("name");
        if (!nameAttr) throw XacroError("xacro:insert_block without a name", doc.path);

        const std::string name = trim(subst(nameAttr.value(), doc).text);

        const auto it = blocks_.find(name);
        if (it == blocks_.end()) {
            std::string known;
            for (const auto& bound : blocks_) {
                if (!known.empty()) known += ", ";
                known += bound.first;
            }
            throw XacroError("xacro:insert_block '" + name + "': no such block (bound: " +
                                     (known.empty() ? "none" : known) + ")",
                             doc.path);
        }

        // The block's own file decides $(dirname) and relative paths; the scope is the
        // one in force where the block is inserted.
        if (it->second.children) {
            processChildren(it->second.node, dst, it->second.doc);
        } else {
            processNode(it->second.node, dst, it->second.doc);
        }
    }

    bool Expander::condition(const pugi::xml_node& node, const DocCtx& doc) {

        const auto valueAttr = node.attribute("value");
        if (!valueAttr) throw XacroError(std::string("xacro:") + node.name() + " without a value", doc.path);

        const auto reject = [&](const Value& v) -> bool {
            throw XacroError(std::string("conditional \"") + valueAttr.value() + "\" evaluated to " +
                                     v.toString() + ", which is not a boolean",
                             doc.path);
        };

        const auto coerce = [&](const Value& v) {
            if (v.isBool()) return v.truthy();
            if (v.isInt() || v.isDouble()) {
                const double d = v.asNumber();
                if (d == 0.0) return false;
                if (d == 1.0) return true;
            }
            return reject(v);
        };

        const auto literal = [](const std::string& text) -> std::optional<bool> {
            const std::string l = lower(trim(text));
            if (l == "true" || l == "1") return true;
            if (l == "false" || l == "0") return false;
            return std::nullopt;
        };

        const auto result = subst(valueAttr.value(), doc);

        if (result.whole) {
            if (result.whole->isString()) {
                if (const auto b = literal(result.whole->asString())) return *b;
            }
            return coerce(*result.whole);
        }

        if (const auto b = literal(result.text)) return *b;

        EvalContext ctx;
        ctx.scope = &scope_;
        ctx.args = &args_;
        ctx.argsAsProperties = inputs_.argsAsProperties;
        ctx.document = doc.path;
        ctx.diags = &diags_;
        ctx.locator = this;
        return coerce(evaluate(trim(result.text), ctx));
    }

    void Expander::instantiate(const std::string& name, const pugi::xml_node& call,
                               pugi::xml_node dst, const DocCtx& doc) {

        const auto found = macros_.find(name);
        if (found == macros_.end()) {
            throw XacroError("unknown element '" + doc.prefix + ":" + name + "'; defined macros: " +
                                     joinNames(macros_),
                             doc.path);
        }

        // A macro body may define further macros, so take a copy rather than hold a
        // reference into a map that is about to be written to.
        const MacroDef def = found->second;

        if (++instantiations_ > inputs_.budget.maxInstantiations) {
            throw XacroError("more than " + std::to_string(inputs_.budget.maxInstantiations) +
                                     " macro instantiations",
                             doc.path);
        }
        if (macroDepth_ >= inputs_.budget.maxMacroDepth) {
            throw XacroError("macro '" + name + "' nested more than " +
                                     std::to_string(inputs_.budget.maxMacroDepth) + " deep",
                             doc.path);
        }

        std::map<std::string, Value> supplied;
        for (const auto& attr : call.attributes()) {
            supplied[attr.name()] = typed(attr.value(), doc);
        }

        std::vector<pugi::xml_node> callBlocks;
        for (const auto& child : call.children()) {
            if (child.type() == pugi::node_element) callBlocks.push_back(child);
        }

        auto savedBlocks = std::move(blocks_);
        blocks_.clear();
        scope_.pushFrame();
        ++macroDepth_;

        std::size_t nextBlock = 0;

        for (const auto& param : def.params) {

            if (isBlockParam(param)) {
                if (nextBlock >= callBlocks.size()) {
                    throw XacroError("macro '" + name + "': missing block parameter '" + param.name + "'",
                                     doc.path);
                }
                blocks_[param.name] = BlockArg{callBlocks[nextBlock++],
                                               param.kind == MacroParam::Kind::BlockChildren, doc};
                continue;
            }

            if (const auto it = supplied.find(param.name); it != supplied.end()) {
                scope_.set(param.name, it->second);
                continue;
            }

            switch (param.kind) {
                case MacroParam::Kind::Default:
                    scope_.set(param.name, typed(param.def, def.doc));
                    break;
                case MacroParam::Kind::Inherit: {
                    const Value* inherited = scope_.find(param.name);
                    if (!inherited) {
                        throw XacroError("macro '" + name + "': parameter '" + param.name +
                                                 ":=^' has nothing to inherit",
                                         doc.path);
                    }
                    scope_.set(param.name, *inherited);
                    break;
                }
                case MacroParam::Kind::InheritDefault: {
                    if (const Value* inherited = scope_.find(param.name)) {
                        scope_.set(param.name, *inherited);
                    } else {
                        scope_.set(param.name, typed(param.def, def.doc));
                    }
                    break;
                }
                default:
                    throw XacroError("macro '" + name + "': missing parameter '" + param.name + "'",
                                     doc.path);
            }
        }

        for (const auto& entry : supplied) {
            const std::string& attribute = entry.first;
            const bool declared = std::any_of(def.params.begin(), def.params.end(),
                                              [&](const MacroParam& p) { return !isBlockParam(p) && p.name == attribute; });
            if (!declared) {
                warn("macro '" + name + "': ignoring undeclared attribute '" + attribute + "'", doc);
            }
        }
        if (nextBlock < callBlocks.size()) {
            warn("macro '" + name + "': ignoring " + std::to_string(callBlocks.size() - nextBlock) +
                         " unbound block argument(s)",
                 doc);
        }

        processChildren(def.body, dst, def.doc);

        --macroDepth_;
        scope_.popFrame();
        blocks_ = std::move(savedBlocks);
    }

}// namespace

bool threepp::xacro::needsProcessing(const pugi::xml_document& doc) {

    const auto root = doc.document_element();
    if (!root) return false;

    for (const auto& attr : root.attributes()) {
        const std::string name = attr.name();
        if (name == "xmlns:xacro") return true;
        if (startsWith(name, "xmlns:") && isXacroUri(attr.value())) return true;
    }
    return false;
}

std::string threepp::xacro::documentPrefix(const pugi::xml_document& doc) {

    const auto root = doc.document_element();
    return root ? resolvePrefix(root) : "xacro";
}

bool threepp::xacro::expand(const pugi::xml_document& in, pugi::xml_document& out,
                            const ExpandInputs& inputs, Diagnostics& diags) {

    const auto root = in.document_element();
    if (!root) {
        diags.error("document has no root element", inputs.document);
        return false;
    }

    // Outlives the try, so an error can still ask it where the run had got to.
    Expander expander(inputs, diags);

    try {
        expander.run(root, out);
    } catch (XacroError& e) {
        expander.locate(e);
        diags.error(e);
        return false;
    } catch (const std::exception& e) {
        diags.error(e.what(), inputs.document);
        return false;
    }

    return diags.ok();
}
