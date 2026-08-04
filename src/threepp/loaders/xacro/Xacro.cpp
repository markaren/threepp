
#include "threepp/loaders/Xacro.hpp"

#include "threepp/loaders/xacro/Expand.hpp"

#include "pugixml.hpp"

#include <set>
#include <sstream>
#include <string_view>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    std::string serialize(const pugi::xml_document& doc) {

        std::ostringstream out;
        doc.save(out, "  ", pugi::format_indent);
        return out.str();
    }

    std::string trim(std::string_view s) {

        std::size_t b = 0;
        std::size_t e = s.size();
        const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
        while (b < e && space(s[b])) ++b;
        while (e > b && space(s[e - 1])) --e;
        return std::string(s.substr(b, e - b));
    }

    std::vector<ArgDecl> scan(const pugi::xml_document& doc) {

        const auto root = doc.document_element();
        if (!root) return {};

        const std::string tag = documentPrefix(doc) + ":arg";

        std::vector<ArgDecl> args;
        std::set<std::string> seen;

        for (const auto& child : root.children(tag.c_str())) {

            ArgDecl decl;
            decl.name = trim(child.attribute("name").value());
            if (decl.name.empty()) continue;
            if (!seen.insert(decl.name).second) continue;

            if (const auto def = child.attribute("default")) {
                decl.hasDefault = true;
                decl.defaultValue = def.value();
            }
            args.push_back(std::move(decl));
        }

        return args;
    }

}// namespace

struct Processor::Impl {

    std::map<std::string, std::string> args;
    std::map<std::string, std::filesystem::path> packages;

    [[nodiscard]] ProcessResult run(const pugi::xml_document& doc,
                                    const std::filesystem::path& document) const {

        PackageResolver resolver;
        for (const auto& [package, dir] : packages) resolver.addPackagePath(package, dir);

        ExpandInputs inputs;
        for (const auto& [name, value] : args) inputs.args[name] = Value(value);
        inputs.packages = &resolver;
        inputs.document = document;

        pugi::xml_document out;
        Diagnostics diags;

        ProcessResult result;
        result.ok = expand(doc, out, inputs, diags);
        result.errors = diags.errors();
        result.warnings = diags.warnings();
        if (result.ok) result.xml = serialize(out);

        return result;
    }
};

Processor::Processor()
    : pimpl_(std::make_unique<Impl>()) {}

Processor::~Processor() = default;

Processor& Processor::setArg(const std::string& name, const std::string& value) {

    pimpl_->args[name] = value;
    return *this;
}

Processor& Processor::setArgs(const std::map<std::string, std::string>& args) {

    for (const auto& [name, value] : args) pimpl_->args[name] = value;
    return *this;
}

Processor& Processor::addPackagePath(const std::string& package, const std::filesystem::path& dir) {

    pimpl_->packages[package] = dir;
    return *this;
}

ProcessResult Processor::processFile(const std::filesystem::path& path) const {

    pugi::xml_document doc;
    const auto parsed = doc.load_file(path.string().c_str());
    if (!parsed) {
        ProcessResult result;
        result.errors.emplace_back(path.string() + ": " + parsed.description());
        return result;
    }

    return pimpl_->run(doc, path);
}

ProcessResult Processor::processString(const std::string& xml,
                                       const std::filesystem::path& baseDir) const {

    pugi::xml_document doc;
    const auto parsed = doc.load_string(xml.c_str());
    if (!parsed) {
        ProcessResult result;
        result.errors.emplace_back(std::string("<string>: ") + parsed.description());
        return result;
    }

    // A stand-in document name, so the directory-relative machinery has a parent path.
    return pimpl_->run(doc, baseDir / "(string)");
}

std::vector<ArgDecl> threepp::xacro::scanArgs(const std::filesystem::path& file) {

    pugi::xml_document doc;
    if (!doc.load_file(file.string().c_str())) return {};

    return scan(doc);
}

std::vector<ArgDecl> threepp::xacro::scanArgsFromString(const std::string& xml) {

    pugi::xml_document doc;
    if (!doc.load_string(xml.c_str())) return {};

    return scan(doc);
}
