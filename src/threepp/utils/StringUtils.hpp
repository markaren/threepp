
#ifndef THREEPP_STRINGUTILS_HPP
#define THREEPP_STRINGUTILS_HPP

#include <algorithm>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace threepp::utils {

    inline std::vector<std::string> split(const std::string& s, char delimiter) {

        std::string token;
        std::vector<std::string> tokens;
        std::istringstream tokenStream(s);

        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }

    template<class ArrayLike>
    inline std::string join(const ArrayLike& v, char c = '\n') {

        auto p = v.cbegin();
        std::stringstream ss;
        for (unsigned i = 0; i < v.size(); ++i) {
            ss << *p;
            if (i != v.size() - 1) ss << c;
            ++p;
        }
        return ss.str();
    }

    inline void replaceAll(std::string& text, const std::string& replaceFrom, const std::string& replaceTo) {
        std::string& result = text;
        size_t start_pos = 0;
        while (((start_pos = text.find(replaceFrom, start_pos)) != std::string::npos)) {
            result.replace(start_pos, replaceFrom.length(), replaceTo);
            start_pos += replaceTo.length();
        }
    }

    inline std::string trimStart(std::string s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
        return s;
    }

    inline void trimStartInplace(std::string& s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
    }

    inline std::string trimEnd(std::string s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(),
                s.end());
        return s;
    }

    inline void trimEndInplace(std::string& s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(),
                s.end());
    }

    inline std::string trim(std::string s) {
        s = trimStart(s);
        s = trimEnd(s);
        return s;
    }

    inline void trimInplace(std::string& s) {
        trimStartInplace(s);
        trimEndInplace(s);
    }

    inline std::string toLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return s;
    }

    inline void toLowerInplace(std::string& s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
    }


}// namespace threepp::utils

#endif//THREEPP_STRINGUTILS_HPP
