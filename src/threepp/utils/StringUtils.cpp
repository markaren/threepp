
#include "threepp/utils/StringUtils.hpp"

#include <algorithm>
#include <cctype>

#ifdef _MSC_VER
#include <charconv>
#endif
#include <iostream>

using namespace threepp;

std::vector<std::string> utils::split(const std::string& s, char delimiter) {

    std::string token;
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

void utils::replaceAll(std::string& text, const std::string& replaceFrom, const std::string& replaceTo) {
    std::string& result = text;
    size_t start_pos = 0;
    while ((start_pos = text.find(replaceFrom, start_pos)) != std::string::npos) {
        result.replace(start_pos, replaceFrom.length(), replaceTo);
        start_pos += replaceTo.length();
    }
}

std::string utils::trimStart(std::string s) {
    s.erase(s.begin(), std::ranges::find_if(s, [](unsigned char ch) {
                return !std::isspace(ch);
            }));
    return s;
}

void utils::trimStartInplace(std::string& s) {
    s.erase(s.begin(), std::ranges::find_if(s, [](unsigned char ch) {
                return !std::isspace(ch);
            }));
}

std::string utils::trimEnd(std::string s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(),
            s.end());
    return s;
}

void utils::trimEndInplace(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(),
            s.end());
}

std::string utils::trim(std::string s) {
    s = trimStart(s);
    s = trimEnd(s);
    return s;
}

void utils::trimInplace(std::string& s) {
    trimStartInplace(s);
    trimEndInplace(s);
}

std::string utils::toLower(std::string s) {
    std::ranges::transform(s, s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
    return s;
}

void utils::toLowerInplace(std::string& s) {
    std::ranges::transform(s, s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
}

// https://stackoverflow.com/questions/4654636/how-to-determine-if-a-string-is-a-number-with-c
bool utils::isNumber(const std::string& s) {

    char* p;
    strtod(s.c_str(), &p);
    return !*p;
}

// https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
bool utils::endsWith(std::string const& value, std::string const& ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int utils::parseInt(const std::string& str) {
    int value{};
#ifdef _MSC_VER
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
    if (ec != std::errc()) {
        throw std::runtime_error("Threepp: Error during int conversion!");
    }
#else
    try {
        value = std::stoi(str);
    } catch (const std::exception& e) {
        std::cerr << "Threepp: Error during int conversion: " << e.what() << '\n';
    }
#endif

    return value;
}

float utils::parseFloat(const std::string& str) {
    float value{};
#ifdef _MSC_VER
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
    if (ec != std::errc()) {
        throw std::runtime_error("Threepp: Error during float conversion!");
    }
#else
    try {
        value = std::stof(str);
    } catch (const std::exception& e) {
        std::cerr << "Threepp: Error during float conversion: " << e.what() << '\n';
    }
#endif

    return value;
}
