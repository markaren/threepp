// Flatten a xacro document into a plain URDF, on the command line.
//
// threepp parses xacro in-process wherever a URDF is accepted, so nothing in
// the library needs this. What needs it is everything OUTSIDE the library: a
// robot description vendored into an asset repo should be a plain .urdf, so it
// loads with no package paths to configure and no expansion to go wrong, and so
// that what ships is exactly what was reviewed. This is the tool that produces
// such a file, and checking it in is what makes the vendored asset
// reproducible rather than a thing someone once generated.
//
//   xacro_to_urdf <in.xacro> <out.urdf> [--package NAME=DIR]... [name:=value]...
//
//   --package NAME=DIR   where $(find NAME) and package://NAME/ should look
//   name:=value          a xacro argument, exactly as on the xacro CLI
//
// Writing to "-" prints to stdout instead.

#include "threepp/loaders/Xacro.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

using namespace threepp;

namespace {

    void usage() {
        std::cerr << "usage: xacro_to_urdf <in.xacro> <out.urdf|-> [--package NAME=DIR]... [name:=value]...\n";
    }

}// namespace

int main(int argc, char** argv) {

    std::filesystem::path input;
    std::filesystem::path output;
    std::map<std::string, std::string> args;
    std::map<std::string, std::filesystem::path> packages;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        }

        if (arg == "--package") {
            if (i + 1 >= argc) {
                std::cerr << "--package needs NAME=DIR\n";
                return 1;
            }
            const std::string spec = argv[++i];
            const auto eq = spec.find('=');
            if (eq == std::string::npos) {
                std::cerr << "--package needs NAME=DIR, got '" << spec << "'\n";
                return 1;
            }
            packages.emplace(spec.substr(0, eq), spec.substr(eq + 1));
            continue;
        }

        // name:=value, the xacro CLI spelling. Checked before the positional
        // cases so a path can never be mistaken for an argument.
        if (const auto assign = arg.find(":="); assign != std::string::npos) {
            args.emplace(arg.substr(0, assign), arg.substr(assign + 2));
            continue;
        }

        if (input.empty()) {
            input = arg;
        } else if (output.empty()) {
            output = arg;
        } else {
            std::cerr << "unexpected argument '" << arg << "'\n";
            usage();
            return 1;
        }
    }

    if (input.empty() || output.empty()) {
        usage();
        return 1;
    }

    if (!std::filesystem::exists(input)) {
        std::cerr << "no such file: " << input.string() << "\n";
        return 1;
    }

    xacro::Processor processor;
    processor.setArgs(args);
    for (const auto& [name, dir] : packages) {
        processor.addPackagePath(name, dir);
    }

    const xacro::ProcessResult result = processor.processFile(input);

    for (const auto& warning : result.warnings) {
        std::cerr << "warning: " << warning << "\n";
    }

    if (!result.ok) {
        for (const auto& error : result.errors) {
            std::cerr << "error: " << error << "\n";
        }
        return 1;
    }

    if (output == "-") {
        std::cout << result.xml;
        return 0;
    }

    std::ofstream out(output, std::ios::binary);
    if (!out) {
        std::cerr << "cannot write " << output.string() << "\n";
        return 1;
    }
    out << result.xml;
    out.close();

    std::cout << "wrote " << output.string() << " (" << result.xml.size() << " bytes)\n";
    return 0;
}
