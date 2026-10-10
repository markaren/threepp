// The whitespace rule of the xacro front end, in one place.

#ifndef THREEPP_XACRO_STRINGS_HPP
#define THREEPP_XACRO_STRINGS_HPP

#include <cstddef>
#include <string>
#include <string_view>

namespace threepp::xacro {

    // Deliberately not std::isspace, and so not utils::trim: what counts as
    // blank in a xacro document must not depend on the process locale.
    // (YamlLite and Value keep their own rules: a YAML line never spans a
    // newline, and a literal also sheds form feeds and vertical tabs.)
    inline bool isSpace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

    inline std::string trim(std::string_view s) {

        std::size_t b = 0;
        std::size_t e = s.size();
        while (b < e && isSpace(s[b])) ++b;
        while (e > b && isSpace(s[e - 1])) --e;
        return std::string(s.substr(b, e - b));
    }

}// namespace threepp::xacro

#endif//THREEPP_XACRO_STRINGS_HPP
