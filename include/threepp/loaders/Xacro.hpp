
#ifndef THREEPP_XACRO_HPP
#define THREEPP_XACRO_HPP

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace threepp::xacro {

    struct ProcessResult {
        bool ok = false;
        std::string xml;                 // the flattened document, when ok
        std::vector<std::string> errors;  // "file: message", in the order they were found
        std::vector<std::string> warnings;
    };

    // Turns a xacro document into a plain one: properties, arguments, macros, includes and
    // conditionals are executed; everything else is copied with its ${...} and $(...) spans
    // resolved. Nothing is silently skipped — the first thing that cannot be evaluated ends
    // the run and is reported in `errors`.
    class Processor {

    public:
        Processor();

        // An argument as if given on the xacro command line: it wins over the default of the
        // matching <xacro:arg>.
        Processor& setArg(const std::string& name, const std::string& value);

        Processor& setArgs(const std::map<std::string, std::string>& args);

        // Where $(find package) should look, ahead of the manifest walk and the environment.
        Processor& addPackagePath(const std::string& package, const std::filesystem::path& dir);

        [[nodiscard]] ProcessResult processFile(const std::filesystem::path& path) const;

        // baseDir stands in for the document's directory: relative includes, $(dirname) and
        // load_yaml resolve against it.
        [[nodiscard]] ProcessResult processString(const std::string& xml,
                                                  const std::filesystem::path& baseDir) const;

        ~Processor();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

    struct ArgDecl {
        std::string name;
        // The declared text, exactly as written — neither substituted nor evaluated. UR's
        // joint_limit_params defaults to
        // "$(find ur_description)/config/$(arg ur_type)/joint_limits.yaml", and the point of
        // showing it is that it is still a recipe at this stage, not a path.
        std::string defaultValue;
        bool hasDefault = false;
    };

    // The <xacro:arg> declarations directly under the document root, in document order and
    // deduplicated by name (first wins, as in expansion). Deliberately shallow: it follows
    // no includes, expands no macros and evaluates nothing, because the whole point is to
    // learn what a file wants to be told BEFORE anything that needs telling can run. A file
    // that declares no arguments — any plain URDF — yields an empty vector, and so does one
    // that cannot be read or parsed, rather than throwing.
    [[nodiscard]] std::vector<ArgDecl> scanArgs(const std::filesystem::path& file);

    [[nodiscard]] std::vector<ArgDecl> scanArgsFromString(const std::string& xml);

    // Where the editor keeps the arguments a robot was imported with, so a rebuild uses the
    // same ones. The names live under one key, comma-joined (a xacro argument name is an
    // identifier, so that needs no escaping); each value lives under its own key, verbatim,
    // because values are very often paths. Declared here, in the layer that owns the
    // concept, so the editor and ObjectLoader agree on the spelling without the loaders
    // having to know the editor exists.
    inline constexpr const char* argsUserDataKey = "xacroArgs";
    inline constexpr const char* argValueUserDataPrefix = "xacroArg:";

}// namespace threepp::xacro

#endif//THREEPP_XACRO_HPP
