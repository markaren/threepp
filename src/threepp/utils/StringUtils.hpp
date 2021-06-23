
#ifndef THREEPP_STRINGUTILS_HPP
#define THREEPP_STRINGUTILS_HPP

#include <string>
#include <vector>

namespace threepp::utils {

    std::vector<std::string> split(const std::string &s, char delimiter);

    void join(const std::vector<std::string> &v, char c, std::string &s);

    std::string addLineNumbers(const std::string &str);

}

#endif//THREEPP_STRINGUTILS_HPP
