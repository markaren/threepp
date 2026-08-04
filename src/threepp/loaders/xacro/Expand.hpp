// Structural expansion: a xacro document in, a plain document out.
// Properties, args, macros, includes and conditionals are executed in document order,
// exactly as python xacro does, and the first thing that cannot be made sense of stops
// the run rather than being papered over.

#ifndef THREEPP_XACRO_EXPAND_HPP
#define THREEPP_XACRO_EXPAND_HPP

#include "threepp/loaders/xacro/Diagnostics.hpp"
#include "threepp/loaders/xacro/PackageResolver.hpp"
#include "threepp/loaders/xacro/Value.hpp"

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>

namespace pugi {
    class xml_document;
    class xml_node;
}// namespace pugi

namespace threepp::xacro {

    // Ceilings on what one document may unfold into, so a malformed or hostile file
    // fails with a message instead of exhausting memory.
    struct Budget {
        std::size_t maxOutputElements = 1000000;
        std::size_t maxIncludeDepth = 64;
        std::size_t maxMacroDepth = 128;
        std::size_t maxInstantiations = 100000;
    };

    struct ExpandInputs {
        std::map<std::string, Value> args;// the $(arg) table, as if passed on the xacro command line
        PackageResolver* packages = nullptr;
        std::filesystem::path document;// the file `in` was read from; relative includes,
                                       // $(dirname) and load_yaml resolve against its directory
        Budget budget;
        // URDFLoader compatibility: a name that is not a property may resolve from the
        // argument table (with a warning). Off for the standalone Processor.
        bool argsAsProperties = false;
    };

    bool expand(const pugi::xml_document& in, pugi::xml_document& out,
                const ExpandInputs& inputs, Diagnostics& diags);

    // True when the document declares the xacro namespace on its root element.
    [[nodiscard]] bool needsProcessing(const pugi::xml_document& doc);

    // The prefix this document binds the xacro namespace to, "xacro" when it binds none —
    // the same answer expansion works from, so anything that has to recognise a xacro
    // element before expanding asks here rather than assuming.
    [[nodiscard]] std::string documentPrefix(const pugi::xml_document& doc);

}// namespace threepp::xacro

#endif
