
#ifndef THREEPP_STRINGUTILS_HPP
#define THREEPP_STRINGUTILS_HPP

#include <sstream>
#include <string>
#include <vector>
#include <regex>

namespace threepp::utils {

    inline std::vector<std::string> split(const std::string &s, char delimiter) {

        std::string token;
        std::vector<std::string> tokens;
        std::istringstream tokenStream(s);

        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }

    std::vector<std::string> regexSplit(const std::string &s, const std::regex &sep_regex) {
        std::sregex_token_iterator iter(s.begin(), s.end(), sep_regex, -1);
        std::sregex_token_iterator end;
        return {iter, end};
    }

    inline std::string join(const std::vector<std::string> &v, char c = '\n') {

        std::stringstream ss;

        for (auto p = v.begin(); p != v.end(); ++p) {
            ss << *p;
            if (p != v.end() - 1) {
                ss << c;
            }
        }

        return ss.str();
    }

    inline std::string addLineNumbers(const std::string &str) {

        auto lines = split(str, '\n');

        for (int i = 0; i < lines.size(); i++) {

            lines[i] = std::to_string(i + 1) + ": " + lines[i];
        }

        return join(lines, '\n');
    }

    inline std::string replaceAll(const std::string& text, const std::string& replaceFrom, const std::string& replaceTo) {
        std::string result = text;
        size_t start_pos = 0;
        while (((start_pos = text.find(replaceFrom, start_pos)) != std::string::npos)) {
            result.replace(start_pos, replaceFrom.length(), replaceTo);
            start_pos += replaceTo.length();
        }

        return result;
    }

    // trim from start (in place)
    inline void trimStart(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
    }

    // trim from end (in place)
    inline void trimEnd(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(), s.end());
    }

    // trim from both sides
    inline void trim(std::string &s) {
        trimStart(s);
        trimEnd(s);
    }


}// namespace threepp::utils

#endif//THREEPP_STRINGUTILS_HPP
