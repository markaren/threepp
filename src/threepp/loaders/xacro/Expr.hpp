// The python subset xacro expressions are written in, evaluated over Value.
// Anything the evaluator cannot make sense of throws; it never falls back to a default.

#ifndef THREEPP_XACRO_EXPR_HPP
#define THREEPP_XACRO_EXPR_HPP

#include "threepp/loaders/xacro/Diagnostics.hpp"
#include "threepp/loaders/xacro/Scope.hpp"
#include "threepp/loaders/xacro/Value.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <string_view>

namespace threepp::xacro {

    struct EvalContext {
        const Scope* scope = nullptr;
        // Consulted for a name the scope does not know, and only when argsAsProperties is
        // set: URDFLoader has always exposed its arguments as ${name} as well.
        const std::map<std::string, Value>* args = nullptr;
        bool argsAsProperties = false;
        std::filesystem::path document;// the file the expression sits in; load_yaml resolves against it
        Diagnostics* diags = nullptr;
    };

    [[nodiscard]] Value evaluate(std::string_view expression, const EvalContext& ctx);

}// namespace threepp::xacro

#endif
