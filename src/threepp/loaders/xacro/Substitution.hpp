// ${expression} and $(command ...) substitution over attribute values and text nodes.

#ifndef THREEPP_XACRO_SUBSTITUTION_HPP
#define THREEPP_XACRO_SUBSTITUTION_HPP

#include "threepp/loaders/xacro/Diagnostics.hpp"
#include "threepp/loaders/xacro/PackageResolver.hpp"
#include "threepp/loaders/xacro/Scope.hpp"
#include "threepp/loaders/xacro/Value.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace threepp::xacro {

    struct SubstCtx {
        const Scope* scope = nullptr;
        const std::map<std::string, Value>* args = nullptr;
        PackageResolver* packages = nullptr;
        std::filesystem::path document;
        Diagnostics* diags = nullptr;
        bool argsAsProperties = false;// see EvalContext
    };

    struct SubstResult {
        std::string text;
        // set when the input was a single ${...} or $(eval ...) span, so the caller can
        // keep the type instead of the rendering
        std::optional<Value> whole;

        [[nodiscard]] Value value() const { return whole ? *whole : Value(text); }
    };

    [[nodiscard]] SubstResult substitute(std::string_view raw, const SubstCtx& ctx);

    [[nodiscard]] std::string substituteText(std::string_view raw, const SubstCtx& ctx);

}// namespace threepp::xacro

#endif
