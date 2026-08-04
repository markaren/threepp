
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

}// namespace threepp::xacro

#endif//THREEPP_XACRO_HPP
