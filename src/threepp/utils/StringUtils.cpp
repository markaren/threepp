
#include "threepp/utils/StringUtils.hpp"

#ifdef _MSC_VER
#include <charconv>
#endif
#include <iostream>

using namespace threepp;

int utils::parseInt(const std::string& str) {
    int value{};
#ifdef _MSC_VER
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
    if (ec != std::errc()) {
        throw std::runtime_error("Error during conversion!");
    }
#else
    try {
        value = std::stoi(str);
    } catch (const std::exception& e) {
        std::cerr << "Error during conversion: " << e.what() << '\n';
    }
#endif

    return value;
}

float utils::parseFloat(const std::string& str) {
    float value{};
#ifdef _MSC_VER
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
    if (ec != std::errc()) {
        throw std::runtime_error("Error during conversion!");
    }
#else
    try {
        value = std::stof(str);
    } catch (const std::exception& e) {
        std::cerr << "Error during conversion: " << e.what() << '\n';
    }
#endif

    return value;
}
