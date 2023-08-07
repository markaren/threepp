
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/utils/StringUtils.hpp"

#include <array>
#include <list>
#include <vector>

using namespace threepp;

TEST_CASE("join") {

    {
        std::string join = utils::join(std::vector<std::string>{"1", "2", "3", "4"}, ' ');
        REQUIRE(join == "1 2 3 4");
    }

    {
        std::string join = utils::join(std::list<std::string>{"1", "2", "3", "4"}, ' ');
        REQUIRE(join == "1 2 3 4");
    }

    {
        std::string join = utils::join(std::array<std::string, 4>{"1", "2", "3", "4"}, ' ');
        REQUIRE(join == "1 2 3 4");
    }
}

TEST_CASE("split") {

    std::vector<std::string> answer{"1", "2", "3", "4"};

    {
        auto split = utils::split("1 2 3 4", ' ');
        REQUIRE(split == answer);
    }

    {
        auto split = utils::split("1\n2\n3\n4", '\n');
        REQUIRE(split == answer);
    }
}


TEST_CASE("trim") {

    {
        std::string str{"hello"};
        REQUIRE(utils::trim(str) == str);
        REQUIRE(utils::trimStart(str) == str);
        REQUIRE(utils::trimEnd(str) == str);
    }

    {
        std::string str{" hello "};
        auto trim = utils::trimStart(str);
        REQUIRE(trim == str.substr(1));

        str = "    hello ";
        trim = utils::trimStart(str);
        REQUIRE(trim == str.substr(4));
    }

    {
        std::string str{" hello "};
        auto trim = utils::trimEnd(str);
        REQUIRE(trim == std::string{str.begin(), str.end() - 1});

        str = "    hello    ";
        trim = utils::trimEnd(str);
        REQUIRE(trim == std::string{str.begin(), str.end() - 4});
    }

    {
        std::string str{" hello "};
        auto trim = utils::trim(str);
        REQUIRE(trim == std::string{str.begin() + 1, str.end() - 1});

        str = "    hello    ";
        trim = utils::trim(str);
        REQUIRE(trim == std::string{str.begin() + 4, str.end() - 4});
    }
}

TEST_CASE("parseNumber successfully parses numbers from strings", "[parseNumber]") {
    SECTION("Integer parsing") {
        std::string strInt = "123";
        auto intResult = utils::parseInt(strInt);
        REQUIRE(intResult == 123);
    }

    SECTION("Float parsing") {
        std::string strFloat = "456.789";
        auto floatResult = utils::parseFloat(strFloat);
        REQUIRE_THAT(floatResult, Catch::Matchers::WithinRel(456.789f));
    }
}
