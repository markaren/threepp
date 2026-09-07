// Error type and diagnostic collector shared by the xacro front end.

#ifndef THREEPP_XACRO_DIAGNOSTICS_HPP
#define THREEPP_XACRO_DIAGNOSTICS_HPP

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace threepp::xacro {

    class XacroError: public std::runtime_error {

    public:
        explicit XacroError(const std::string& message, std::filesystem::path document = {},
                            std::size_t line = 0);

        [[nodiscard]] const std::string& message() const { return message_; }

        [[nodiscard]] const std::filesystem::path& document() const { return document_; }

        // 1-based, 0 when the error could not be pinned to a line.
        [[nodiscard]] std::size_t line() const { return line_; }

        // Errors are thrown where the rule is known and located on the way out, by whoever
        // still holds the node. The first frame to place one wins: it is the innermost, and
        // so the closest to what the author actually wrote.
        void locate(const std::filesystem::path& document, std::size_t line);

        [[nodiscard]] const char* what() const noexcept override { return what_.c_str(); }

    private:
        std::string message_;
        std::filesystem::path document_;
        std::size_t line_ = 0;
        std::string what_;
    };

    // "file:line: message", dropping either half that is not known.
    [[nodiscard]] std::string decorate(const std::string& message,
                                       const std::filesystem::path& document, std::size_t line = 0);

    // Where a run has got to. Counting lines costs a read of the file, so the answer is
    // asked for only by whoever is about to report something.
    struct Locator {

        virtual ~Locator() = default;

        [[nodiscard]] virtual std::size_t currentLine() const = 0;
    };

    class Diagnostics {

    public:
        void error(const std::string& message, const std::filesystem::path& document = {});

        void error(const XacroError& e);

        void warn(const std::string& message, const std::filesystem::path& document = {},
                  std::size_t line = 0);

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
