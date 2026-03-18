
#ifndef THREEPP_XACROPROCESSOR_HPP
#define THREEPP_XACROPROCESSOR_HPP

#include "pugixml.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <list>
#include <map>
#include <string>

namespace threepp::xacro {

    // Recursive-descent evaluator for xacro ${...} expressions.
    // Supports: arithmetic (+,-,*,/,**), unary minus, parentheses,
    //           comparison (==,!=,<,>,<=,>=), boolean (and,or,not),
    //           math functions (sin,cos,tan,asin,acos,atan,atan2,sqrt,
    //           exp,log,abs,floor,ceil,round,radians,degrees,min,max,pow),
    //           constants (pi, true, false), and variable lookup.
    class ExprParser {
        const std::string& s_;
        size_t pos_;
        const std::map<std::string, std::string>& vars_;

    public:
        ExprParser(const std::string& s, const std::map<std::string, std::string>& vars)
            : s_(s), pos_(0), vars_(vars) {}

        double evaluate() { return parseOr(); }

    private:
        void skipWs() {
            while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_])))
                ++pos_;
        }

        bool tryOp(const std::string& op) {
            skipWs();
            if (pos_ + op.size() > s_.size()) return false;
            if (s_.substr(pos_, op.size()) == op) {
                pos_ += op.size();
                return true;
            }
            return false;
        }

        bool tryKeyword(const std::string& kw) {
            skipWs();
            if (pos_ + kw.size() > s_.size()) return false;
            if (s_.substr(pos_, kw.size()) == kw) {
                char next = pos_ + kw.size() < s_.size()
                                ? s_[pos_ + kw.size()]
                                : '\0';
                if (!std::isalnum(static_cast<unsigned char>(next)) && next != '_') {
                    pos_ += kw.size();
                    return true;
                }
            }
            return false;
        }

        std::string parseName() {
            skipWs();
            std::string name;
            while (pos_ < s_.size() &&
                   (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_'))
                name += s_[pos_++];
            return name;
        }

        double parseNumber() {
            skipWs();
            size_t start = pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
            if (pos_ < s_.size() && s_[pos_] == '.') {
                ++pos_;
                while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
            }
            if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
                ++pos_;
                if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
                while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
            }
            try { return std::stod(s_.substr(start, pos_ - start)); } catch (...) { return 0.0; }
        }

        double parseOr() {
            double left = parseAnd();
            while (tryKeyword("or")) left = (left != 0.0 || parseAnd() != 0.0) ? 1.0 : 0.0;
            return left;
        }

        double parseAnd() {
            double left = parseNot();
            while (tryKeyword("and")) left = (left != 0.0 && parseNot() != 0.0) ? 1.0 : 0.0;
            return left;
        }

        double parseNot() {
            if (tryKeyword("not")) return parseNot() == 0.0 ? 1.0 : 0.0;
            return parseComparison();
        }

        double parseComparison() {
            double left = parseAddSub();
            skipWs();
            if (tryOp("==")) return (left == parseAddSub()) ? 1.0 : 0.0;
            if (tryOp("!=")) return (left != parseAddSub()) ? 1.0 : 0.0;
            if (tryOp("<=")) return (left <= parseAddSub()) ? 1.0 : 0.0;
            if (tryOp(">=")) return (left >= parseAddSub()) ? 1.0 : 0.0;
            if (tryOp("<"))  return (left <  parseAddSub()) ? 1.0 : 0.0;
            if (tryOp(">"))  return (left >  parseAddSub()) ? 1.0 : 0.0;
            return left;
        }

        double parseAddSub() {
            double left = parseMulDiv();
            while (true) {
                skipWs();
                if (pos_ < s_.size() && s_[pos_] == '+') { ++pos_; left += parseMulDiv(); }
                else if (pos_ < s_.size() && s_[pos_] == '-') { ++pos_; left -= parseMulDiv(); }
                else break;
            }
            return left;
        }

        double parseMulDiv() {
            double left = parseUnary();
            while (true) {
                skipWs();
                if (pos_ < s_.size() && s_[pos_] == '*') {
                    ++pos_;
                    if (pos_ < s_.size() && s_[pos_] == '*') { ++pos_; left = std::pow(left, parseUnary()); }
                    else left *= parseUnary();
                } else if (pos_ < s_.size() && s_[pos_] == '/') {
                    ++pos_; left /= parseUnary();
                } else break;
            }
            return left;
        }

        double parseUnary() {
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == '-') { ++pos_; return -parseAtom(); }
            if (pos_ < s_.size() && s_[pos_] == '+') { ++pos_; }
            return parseAtom();
        }

        double callFunc(const std::string& name) {
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == '(') ++pos_;
            double a1 = parseOr();
            double a2 = 0.0;
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; a2 = parseOr(); }
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == ')') ++pos_;

            if (name == "sin")     return std::sin(a1);
            if (name == "cos")     return std::cos(a1);
            if (name == "tan")     return std::tan(a1);
            if (name == "asin")    return std::asin(a1);
            if (name == "acos")    return std::acos(a1);
            if (name == "atan")    return std::atan(a1);
            if (name == "atan2")   return std::atan2(a1, a2);
            if (name == "sqrt")    return std::sqrt(a1);
            if (name == "exp")     return std::exp(a1);
            if (name == "log")     return std::log(a1);
            if (name == "abs")     return std::abs(a1);
            if (name == "floor")   return std::floor(a1);
            if (name == "ceil")    return std::ceil(a1);
            if (name == "round")   return std::round(a1);
            if (name == "radians") return a1 * math::PI / 180.0;
            if (name == "degrees") return a1 * 180.0 / math::PI;
            if (name == "min")     return std::min(a1, a2);
            if (name == "max")     return std::max(a1, a2);
            if (name == "pow")     return std::pow(a1, a2);
            std::cerr << "[xacro] Unknown function: " << name << "\n";
            return 0.0;
        }

        double parseAtom() {
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == '(') {
                ++pos_;
                double val = parseOr();
                skipWs();
                if (pos_ < s_.size() && s_[pos_] == ')') ++pos_;
                return val;
            }
            if (pos_ < s_.size() &&
                (std::isdigit(static_cast<unsigned char>(s_[pos_])) ||
                 (s_[pos_] == '.' && pos_ + 1 < s_.size() &&
                  std::isdigit(static_cast<unsigned char>(s_[pos_ + 1])))))
                return parseNumber();

            if (pos_ < s_.size() &&
                (std::isalpha(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) {
                std::string name = parseName();
                skipWs();
                if (pos_ < s_.size() && s_[pos_] == '(') return callFunc(name);
                if (name == "pi" || name == "PI") return math::PI;
                if (name == "true")  return 1.0;
                if (name == "false") return 0.0;
                auto it = vars_.find(name);
                if (it != vars_.end()) {
                    try { return std::stod(it->second); } catch (...) {
                        if (it->second == "true")  return 1.0;
                        if (it->second == "false") return 0.0;
                    }
                }
                std::cerr << "[xacro] Undefined variable: " << name << "\n";
                return 0.0;
            }
            return 0.0;
        }
    };

    static std::string formatDouble(double v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.10g", v);
        return buf;
    }

    // Preprocesses a xacro XML document into a plain URDF document.
    class Processor {
    public:
        using Props  = std::map<std::string, std::string>;
        using Macros = std::map<std::string, pugi::xml_node>;

        explicit Processor(std::filesystem::path basePath,
                           Props initialArgs = {})
            : basePath_(std::move(basePath)), initialArgs_(std::move(initialArgs)) {}

        static bool needsProcessing(const pugi::xml_document& doc) {
            const auto root = doc.document_element();
            return root && root.attribute("xmlns:xacro");
        }

        void process(const pugi::xml_document& src, pugi::xml_document& out) {
            Props props;
            props["pi"] = formatDouble(math::PI);
            for (const auto& [k, v] : initialArgs_) props[k] = v;
            Macros macros;

            const auto srcRoot = src.document_element();
            if (!srcRoot) return;

            auto outRoot = out.append_child(srcRoot.name());
            for (const auto& attr : srcRoot.attributes()) {
                std::string attrName = attr.name();
                if (attrName.find("xmlns:xacro") == std::string::npos)
                    outRoot.append_attribute(attr.name()) =
                            substituteStr(attr.value(), props).c_str();
            }
            processChildren(srcRoot, outRoot, out, props, macros);
        }

    private:
        std::filesystem::path basePath_;
        Props initialArgs_;
        std::list<pugi::xml_document> docStorage_;

        std::string substituteStr(const std::string& str, const Props& props) {
            if (str.find("${") == std::string::npos &&
                str.find("$(") == std::string::npos) return str;

            std::string result;
            result.reserve(str.size());
            size_t i = 0;
            while (i < str.size()) {
                if (i + 1 < str.size() && str[i] == '$' && str[i + 1] == '{') {
                    size_t start = i + 2;
                    int depth = 1;
                    size_t j = start;
                    while (j < str.size() && depth > 0) {
                        if (str[j] == '{') ++depth;
                        else if (str[j] == '}') --depth;
                        ++j;
                    }
                    std::string expr = str.substr(start, j - start - 1);
                    size_t es = expr.find_first_not_of(' ');
                    size_t ee = expr.find_last_not_of(' ');
                    if (es != std::string::npos) expr = expr.substr(es, ee - es + 1);

                    bool isIdent = !expr.empty();
                    for (char c : expr)
                        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') { isIdent = false; break; }

                    if (isIdent) {
                        auto it = props.find(expr);
                        if (it != props.end()) { result += it->second; i = j; continue; }
                    }

                    try {
                        ExprParser ep(expr, props);
                        result += formatDouble(ep.evaluate());
                    } catch (...) {
                        result += "${" + expr + "}";
                    }
                    i = j;
                } else if (i + 1 < str.size() && str[i] == '$' && str[i + 1] == '(') {
                    size_t start = i + 2;
                    size_t j = str.find(')', start);
                    if (j == std::string::npos) { result += str[i++]; continue; }
                    std::string inner = str.substr(start, j - start);
                    if (inner.size() > 4 && inner.substr(0, 4) == "arg ") {
                        std::string argName = inner.substr(4);
                        auto it = props.find(argName);
                        result += (it != props.end()) ? it->second : "";
                    } else if (inner.size() > 5 && inner.substr(0, 5) == "find ") {
                        auto it = props.find(inner);
                        result += (it != props.end()) ? it->second : "$(" + inner + ")";
                    } else {
                        result += "$(" + inner + ")";
                    }
                    i = j + 1;
                } else {
                    result += str[i++];
                }
            }
            return result;
        }

        bool evalCondition(const std::string& expr, const Props& props) {
            std::string val = substituteStr(expr, props);
            if (val == "true"  || val == "1") return true;
            if (val == "false" || val == "0") return false;
            try {
                ExprParser ep(val, props);
                return ep.evaluate() != 0.0;
            } catch (...) { return false; }
        }

        static std::string xacroLocal(const char* name) {
            std::string s = name;
            if (s.size() > 6 && s.substr(0, 6) == "xacro:") return s.substr(6);
            return {};
        }

        void processChildren(const pugi::xml_node& src, pugi::xml_node& dst,
                             pugi::xml_document& outDoc, Props& props, Macros& macros) {
            for (const auto& child : src.children())
                processNode(child, dst, outDoc, props, macros);
        }

        void processNode(const pugi::xml_node& node, pugi::xml_node& dst,
                         pugi::xml_document& outDoc, Props& props, Macros& macros) {
            switch (node.type()) {
                case pugi::node_pcdata:
                case pugi::node_cdata: {
                    std::string text = substituteStr(node.value(), props);
                    dst.append_child(node.type()).set_value(text.c_str());
                    return;
                }
                case pugi::node_comment: return;
                case pugi::node_element: break;
                default: dst.append_copy(node); return;
            }

            std::string local = xacroLocal(node.name());
            if (!local.empty()) {
                handleXacro(local, node, dst, outDoc, props, macros);
            } else {
                auto newNode = dst.append_child(node.name());
                for (const auto& attr : node.attributes())
                    newNode.append_attribute(attr.name()) =
                            substituteStr(attr.value(), props).c_str();
                processChildren(node, newNode, outDoc, props, macros);
            }
        }

        void handleXacro(const std::string& local, const pugi::xml_node& node,
                         pugi::xml_node& dst, pugi::xml_document& outDoc,
                         Props& props, Macros& macros) {
            if (local == "property") {
                std::string name  = node.attribute("name").value();
                std::string value = substituteStr(node.attribute("value").value(), props);
                if (!name.empty()) props[name] = value;

            } else if (local == "macro") {
                std::string name = node.attribute("name").value();
                if (!name.empty()) macros[name] = node;

            } else if (local == "include") {
                std::string filename = substituteStr(node.attribute("filename").value(), props);
                includeFile(resolveFilename(filename), dst, outDoc, props, macros);

            } else if (local == "if") {
                if (evalCondition(node.attribute("value").value(), props))
                    processChildren(node, dst, outDoc, props, macros);

            } else if (local == "unless") {
                if (!evalCondition(node.attribute("value").value(), props))
                    processChildren(node, dst, outDoc, props, macros);

            } else if (local == "arg") {
                std::string name = node.attribute("name").value();
                if (!name.empty() && props.find(name) == props.end())
                    props[name] = substituteStr(node.attribute("default").value(), props);

            } else {
                expandMacro(local, node, dst, outDoc, props, macros);
            }
        }

        void expandMacro(const std::string& name, const pugi::xml_node& call,
                         pugi::xml_node& dst, pugi::xml_document& outDoc,
                         const Props& outerProps, Macros& macros) {
            auto it = macros.find(name);
            if (it == macros.end()) {
                std::cerr << "[xacro] Unknown macro: " << name << "\n";
                return;
            }
            const pugi::xml_node& def = it->second;
            std::string paramsStr = def.attribute("params").value();

            Props localProps = outerProps;
            for (const auto& param : utils::split(paramsStr, ' ')) {
                if (param.empty() || param[0] == '*') continue;
                std::string paramName, paramDefault;
                bool hasDefault = false;
                auto sepPos = param.find(":=");
                if (sepPos != std::string::npos) {
                    paramName    = param.substr(0, sepPos);
                    paramDefault = param.substr(sepPos + 2);
                    hasDefault   = true;
                } else {
                    paramName = param;
                }
                if (const auto attr = call.attribute(paramName.c_str()))
                    localProps[paramName] = substituteStr(attr.value(), outerProps);
                else if (hasDefault)
                    localProps[paramName] = substituteStr(paramDefault, outerProps);
            }
            processChildren(def, dst, outDoc, localProps, macros);
        }

        void includeFile(const std::string& filename, pugi::xml_node& dst,
                         pugi::xml_document& outDoc, Props& props, Macros& macros) {
            if (filename.empty() || !std::filesystem::exists(filename)) {
                std::cerr << "[xacro] Cannot include: " << filename << "\n";
                return;
            }
            auto& incDoc = docStorage_.emplace_back();
            if (!incDoc.load_file(filename.c_str())) {
                std::cerr << "[xacro] Failed to parse include: " << filename << "\n";
                docStorage_.pop_back();
                return;
            }
            const auto incRoot = incDoc.document_element();
            if (!incRoot) return;
            auto savedBase = basePath_;
            basePath_ = std::filesystem::path(filename).parent_path();
            processChildren(incRoot, dst, outDoc, props, macros);
            basePath_ = savedBase;
        }

        std::string resolveFilename(const std::string& filename) {
            std::string s = filename;
            // $(find pkg)/rest  — strip the ROS package-find prefix and resolve from package root
            size_t findPos = s.find("$(find ");
            if (findPos != std::string::npos) {
                size_t closePos = s.find(')', findPos);
                if (closePos != std::string::npos) {
                    std::string rest = s.substr(closePos + 1);
                    if (!rest.empty() && (rest[0] == '/' || rest[0] == '\\'))
                        rest = rest.substr(1);
                    // Walk up from basePath_ to find the package root (contains package.xml)
                    auto pkgRoot = basePath_;
                    for (int i = 0; i < 10; ++i) {
                        if (exists(pkgRoot / "package.xml")) {
                            return (pkgRoot / rest).string();
                        }
                        pkgRoot = pkgRoot.parent_path();
                    }
                    // Fallback: resolve relative to basePath_
                    s = rest;
                }
            }
            // package:// URI
            if (s.find("package://") == 0) s = s.substr(10);
            // Make absolute relative to current base path
            if (!std::filesystem::path(s).is_absolute())
                s = (basePath_ / s).string();
            return s;
        }
    };

}// namespace threepp::xacro

#endif//THREEPP_XACROPROCESSOR_HPP
