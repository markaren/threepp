
#include "threepp/loaders/xacro/Expand.hpp"

#include "threepp/loaders/xacro/Expr.hpp"
#include "threepp/loaders/xacro/Scope.hpp"
#include "threepp/loaders/xacro/Substitution.hpp"

#include "pugixml.hpp"

#include <algorithm>
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
            } else if (const auto sep = token.find(":="); sep != std::string::npos) {
                p.name = token.substr(0, sep);
                const std::string rest = token.substr(sep + 2);
                if (rest == "^") {
                    p.kind = MacroParam::Kind::Inherit;
                } else if (startsWith(rest, "^|")) {
                    p.kind = MacroParam::Kind::InheritDefault;
                    p.def = rest.substr(2);
                } else {
                    p.kind = MacroParam::Kind::Default;
                    p.def = rest;
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

    std::string joinNames(const std::map<std::string, MacroDef>& macros) {

        if (macros.empty()) return "none are defined";

        std::string out;
        for (const auto& entry : macros) {
            if (!out.empty()) out += ", ";
            out += entry.first;
        }
        return out;
    }


    class Expander {

    public:
        Expander(const ExpandInputs& inputs, Diagnostics& diags)
            : inputs_(inputs), diags_(diags), args_(inputs.args) {}

        void run(const pugi::xml_node& root, pugi::xml_document& out);

    private:
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

        const ExpandInputs& inputs_;
        Diagnostics& diags_;

        Scope scope_;
        std::map<std::string, Value> args_;
        std::map<std::string, MacroDef> macros_;
        std::map<std::string, BlockArg> blocks_;
        std::vector<std::filesystem::path> includeStack_;
        std::list<pugi::xml_document> owned_;

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
        ctx.argsAsProperties = inputs_.argsAsProperties;
        return substitute(raw, ctx);
    }

    Value Expander::typed(std::string_view raw, const DocCtx& doc) const {

        const auto result = subst(raw, doc);
        return result.whole ? *result.whole : classify(result.text);
    }

    void Expander::run(const pugi::xml_node& root, pugi::xml_document& out) {

        DocCtx doc{inputs_.document, resolvePrefix(root)};

        std::error_code ec;
        auto canonical = std::filesystem::weakly_canonical(inputs_.document, ec);
        includeStack_.push_back(ec ? inputs_.document : canonical);

        auto outRoot = out.append_child(root.name());
        processChildren(root, outRoot, doc);

        // The root's own attributes go last: `<robot name="$(arg name)">` is the standard
        // idiom and the <xacro:arg> that gives `name` a default is one of its children.
        copyAttributes(root, outRoot, doc);
    }

    void Expander::processChildren(const pugi::xml_node& src, pugi::xml_node dst, const DocCtx& doc) {

        for (const auto& child : src.children()) {
            processNode(child, dst, doc);
        }
    }

    void Expander::processNode(const pugi::xml_node& node, pugi::xml_node dst, const DocCtx& doc) {

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
            diags_.warn("macro '" + name + "' redefined", doc.path);
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
        if (file.is_relative()) file = doc.path.parent_path() / file;

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

        const DocCtx sub{canonical, resolvePrefix(root)};

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
                diags_.warn("macro '" + name + "': ignoring undeclared attribute '" + attribute + "'", doc.path);
            }
        }
        if (nextBlock < callBlocks.size()) {
            diags_.warn("macro '" + name + "': ignoring " + std::to_string(callBlocks.size() - nextBlock) +
                                " unbound block argument(s)",
                        doc.path);
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

bool threepp::xacro::expand(const pugi::xml_document& in, pugi::xml_document& out,
                            const ExpandInputs& inputs, Diagnostics& diags) {

    const auto root = in.document_element();
    if (!root) {
        diags.error("document has no root element", inputs.document);
        return false;
    }

    try {
        Expander expander(inputs, diags);
        expander.run(root, out);
    } catch (const XacroError& e) {
        diags.error(e);
        return false;
    } catch (const std::exception& e) {
        diags.error(e.what(), inputs.document);
        return false;
    }

    return diags.ok();
}
