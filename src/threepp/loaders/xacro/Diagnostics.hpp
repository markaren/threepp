// Error type and diagnostic collector shared by the xacro front end.

#ifndef THREEPP_XACRO_DIAGNOSTICS_HPP
#define THREEPP_XACRO_DIAGNOSTICS_HPP

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace threepp::xacro {

    class XacroError: public std::runtime_error {

    public:
        explicit XacroError(const std::string& message, std::filesystem::path document = {});

        [[nodiscard]] const std::string& message() const { return message_; }

        [[nodiscard]] const std::filesystem::path& document() const { return document_; }

    private:
        std::string message_;
        std::filesystem::path document_;
    };

    class Diagnostics {

    public:
        void error(const std::string& message, const std::filesystem::path& document = {});

        void error(const XacroError& e);

        void warn(const std::string& message, const std::filesystem::path& document = {});

        [[nodiscard]] bool ok() const { return errors_.empty(); }

        [[nodiscard]] const std::vector<std::string>& errors() const { return errors_; }

        [[nodiscard]] const std::vector<std::string>& warnings() const { return warnings_; }

        void clear();

    private:
        std::vector<std::string> errors_;
        std::vector<std::string> warnings_;
    };

}// namespace threepp::xacro

#endif
