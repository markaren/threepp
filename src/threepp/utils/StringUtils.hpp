
#ifndef THREEPP_STRINGUTILS_HPP
#define THREEPP_STRINGUTILS_HPP

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

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

    // trim from start (in place)
    inline std::string trimStart(std::string s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
        return s;
    }

    inline std::string trimStartInplace(std::string& s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
        return s;
    }

    // trim from end (in place)
    inline std::string trimEnd(std::string s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(),
                s.end());
        return s;
    }

    inline std::string trimEndInplace(std::string& s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(),
                s.end());
        return s;
    }

    // trim from both sides
    inline std::string trim(std::string s) {
        s = trimStart(s);
        s = trimEnd(s);
        return s;
    }

    // trim from both sides
    inline std::string trimInplace(std::string& s) {
        trimStartInplace(s);
        trimEndInplace(s);
        return s;
    }

    inline std::string toLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); }// correct
        );
        return s;
    }

    inline std::string toLowerInplace(std::string& s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); }// correct
        );
        return s;
    }


}// namespace threepp::utils

#endif//THREEPP_STRINGUTILS_HPP
