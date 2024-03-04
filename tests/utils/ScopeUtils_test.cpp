#include "threepp/utils/Scope.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("scope exit")
{
	bool v = true;
	{
		auto tmp = threepp::utils::at_scope_exit([&] {v = false; });
	}

	REQUIRE(v == false);

}

TEST_CASE("set at scope exit")
{
	bool v = true;
	{
		auto tmp = threepp::utils::set_at_scope_exit(v, false);
	}
	REQUIRE(v == false);
}

TEST_CASE("reset at scope exit")
{
	int i = 0;
	{
		auto tmp = threepp::utils::reset_at_scope_exit(i, 10);
		REQUIRE(i == 10);
	}
	REQUIRE(i == 0);
}
