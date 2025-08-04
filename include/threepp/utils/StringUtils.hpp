
#ifndef THREEPP_STRINGUTILS_HPP
#define THREEPP_STRINGUTILS_HPP

#include <sstream>
#include <string>
#include <vector>

namespace threepp::utils {

    std::vector<std::string> split(const std::string& s, char delimiter);

    template<class ArrayLike>
    std::string join(const ArrayLike& v, char c = '\n') {

        if (v.empty()) return "";

        auto p = v.cbegin();
        std::stringstream ss;
        for (unsigned i = 0; i < v.size(); ++i) {
            ss << *p;
            if (i != v.size() - 1) ss << c;
            ++p;
        }
        return ss.str();
    }

    void replaceAll(std::string& text, const std::string& replaceFrom, const std::string& replaceTo);

    std::string trimStart(std::string s);

    void trimStartInplace(std::string& s);

    std::string trimEnd(std::string s);

    void trimEndInplace(std::string& s);

    std::string trim(std::string s);

    void trimInplace(std::string& s);

    std::string toLower(std::string s);

    void toLowerInplace(std::string& s);

    bool isNumber(const std::string& s);

    bool endsWith(std::string const& value, std::string const& ending);

    int parseInt(const std::string& str);

    float parseFloat(const std::string& str);

}// namespace threepp::utils

#endif//THREEPP_STRINGUTILS_HPP
