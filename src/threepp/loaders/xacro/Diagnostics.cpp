
#include "threepp/loaders/xacro/Diagnostics.hpp"

using namespace threepp;
using namespace threepp::xacro;

std::string threepp::xacro::decorate(const std::string& message,
                                     const std::filesystem::path& document, std::size_t line) {

    if (document.empty()) return message;
    if (line == 0) return document.string() + ": " + message;

    return document.string() + ":" + std::to_string(line) + ": " + message;
}

XacroError::XacroError(const std::string& message, std::filesystem::path document, std::size_t line)
    : std::runtime_error(decorate(message, document, line)),
      message_(message),
      document_(std::move(document)),
      line_(line),
      what_(decorate(message_, document_, line_)) {}

void XacroError::locate(const std::filesystem::path& document, std::size_t line) {

    // A line only means anything against the file it was counted in.
    if (document_.empty()) document_ = document;
    if (line_ == 0 && document_ == document) line_ = line;

    what_ = decorate(message_, document_, line_);
}

void Diagnostics::error(const std::string& message, const std::filesystem::path& document) {

    errors_.emplace_back(decorate(message, document));
}

void Diagnostics::error(const XacroError& e) {

    errors_.emplace_back(decorate(e.message(), e.document(), e.line()));
}

void Diagnostics::warn(const std::string& message, const std::filesystem::path& document,
                       std::size_t line) {

    warnings_.emplace_back(decorate(message, document, line));
}

void Diagnostics::clear() {

    errors_.clear();
    warnings_.clear();
}
