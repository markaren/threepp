
#include "threepp/loaders/xacro/Diagnostics.hpp"

using namespace threepp;
using namespace threepp::xacro;

namespace {

    std::string decorate(const std::string& message, const std::filesystem::path& document) {

        if (document.empty()) return message;
        return document.string() + ": " + message;
    }

}// namespace

XacroError::XacroError(const std::string& message, std::filesystem::path document)
    : std::runtime_error(decorate(message, document)),
      message_(message),
      document_(std::move(document)) {}

void Diagnostics::error(const std::string& message, const std::filesystem::path& document) {

    errors_.emplace_back(decorate(message, document));
}

void Diagnostics::error(const XacroError& e) {

    errors_.emplace_back(decorate(e.message(), e.document()));
}

void Diagnostics::warn(const std::string& message, const std::filesystem::path& document) {

    warnings_.emplace_back(decorate(message, document));
}

void Diagnostics::clear() {

    errors_.clear();
    warnings_.clear();
}
