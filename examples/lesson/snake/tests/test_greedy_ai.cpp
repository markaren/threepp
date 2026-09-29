// test_greedy_ai.cpp: a player is testable too. No window, no keyboard, just a Game.

#include "control/GreedyAI.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace snake;

TEST_CASE("the AI heads for the apple") {
    GreedyAI ai;

    SECTION("straight on") {
        Game game(10, 10, Snake({{2, 5}, {1, 5}}, Direction::Right), {7, 5});
        CHECK(ai.next(game) == Direction::Right);
    }

    SECTION("a turn") {
        Game game(10, 10, Snake({{2, 5}, {1, 5}}, Direction::Right), {2, 8});
        CHECK(ai.next(game) == Direction::Up);
    }
}

TEST_CASE("the AI doesn't drive into a wall") {
    GreedyAI ai;
    // at the right wall, heading right, the apple straight behind
    Game game(10, 10, Snake({{9, 5}, {8, 5}}, Direction::Right), {0, 5});

    const Direction d = ai.next(game);

    CHECK((d == Direction::Up || d == Direction::Down));
}

TEST_CASE("the AI plays a whole game in milliseconds") {
    GreedyAI ai;
    Game game(10, 10, 3);

    while (game.alive() && !game.won() && game.ticks() < 10000) {
        game.steer(ai.next(game));
        game.tick();
    }

    CHECK(game.score() >= 20);
}
