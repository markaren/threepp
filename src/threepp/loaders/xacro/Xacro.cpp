
#include "threepp/loaders/Xacro.hpp"

#include "threepp/loaders/xacro/Expand.hpp"

#include "pugixml.hpp"

#include <sstream>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    std::string serialize(const pugi::xml_document& doc) {

        std::ostringstream out;
        doc.save(out, "  ", pugi::format_indent);
        return out.str();
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
