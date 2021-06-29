
#include "StringUtils.hpp"

#include "sstream"


std::vector<std::string> split(const std::string &s, char delimiter) {

    std::string token;
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string join(const std::vector<std::string> &v, char c) {

    std::string s;

    for (auto p = v.begin(); p != v.end(); ++p) {
        s += *p;
        if (p != v.end() - 1) {
            s += c;
        }
    }

    return s;
}

std::string addLineNumbers(const std::string &str) {

    auto lines = split(str, '\n');

    for (int i = 0; i < lines.size(); i++) {

        lines[i] = std::to_string(i + 1) + ": " + lines[i];
    }

    return join(lines, '\n');
}
