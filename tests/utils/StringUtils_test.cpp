
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/utils/StringUtils.hpp"

#include <vector>
#include <list>
#include <array>

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
        REQUIRE(trim == std::string{str.begin(), str.end()-1});

        str = "    hello    ";
        trim = utils::trimEnd(str);
        REQUIRE(trim == std::string{str.begin(), str.end()-4});
    }

    {
        std::string str{" hello "};
        auto trim = utils::trim(str);
        REQUIRE(trim == std::string{str.begin()+1, str.end()-1});

        str = "    hello    ";
        trim = utils::trim(str);
        REQUIRE(trim == std::string{str.begin()+4, str.end()-4});
    }

}
