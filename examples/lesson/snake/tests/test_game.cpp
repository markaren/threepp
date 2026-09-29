// test_game.cpp: the rules of Snake, checked without a window.
// These tests link snake_model only; threepp isn't part of this program.

#include "model/Game.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace snake;

TEST_CASE("a new game starts alive, with the apple on a free cell") {
    for (unsigned seed = 1; seed <= 50; ++seed) {
        Game game(20, 20, seed);

        CHECK(game.alive());
        CHECK(game.snake().length() == 3);
        CHECK(game.isFree(game.apple()));
    }
}

TEST_CASE("the snake moves one cell per tick") {
    Game game(20, 20, 1);
    const Vec2i head = game.snake().head();

    game.tick();

    CHECK(game.snake().head() == head + Vec2i{1, 0});
    CHECK(game.snake().length() == 3);
    CHECK(game.ticks() == 1);
}

TEST_CASE("the snake can't turn back onto itself") {
    Game game(20, 20, 1);// heading right

    SECTION("straight back") {
        game.steer(Direction::Left);
        game.tick();
        CHECK(game.snake().heading() == Direction::Right);
    }

    SECTION("two quick turns between ticks") {
        game.steer(Direction::Up);
        game.steer(Direction::Left);
        game.tick();
        CHECK(game.snake().heading() == Direction::Right);
    }

    CHECK(game.alive());
}

TEST_CASE("eating grows the snake and scores") {
    Snake snake({{5, 5}, {4, 5}, {3, 5}}, Direction::Right);
    Game game(10, 10, snake, {6, 5});

    int eaten = 0;
    game.onEat([&](const Game&) { ++eaten; });

    game.tick();

    CHECK(game.snake().head() == Vec2i{6, 5});
    CHECK(game.snake().length() == 4);
    CHECK(game.score() == 1);
    CHECK(eaten == 1);
    CHECK(game.isFree(game.apple()));
}

TEST_CASE("hitting the wall ends the game") {
    Snake snake({{9, 5}, {8, 5}}, Direction::Right);
    Game game(10, 10, snake, {0, 0});

    int deaths = 0;
    game.onDie([&](const Game&) { ++deaths; });

    game.tick();
    game.tick();// a dead snake stays dead

    CHECK_FALSE(game.alive());
    CHECK(game.snake().head() == Vec2i{9, 5});
    CHECK(deaths == 1);
    CHECK(game.ticks() == 1);
}

TEST_CASE("biting itself ends the game") {
    // curled up, heading down into its own body
    Snake snake({{5, 5}, {6, 5}, {6, 4}, {5, 4}, {4, 4}}, Direction::Down);
    Game game(10, 10, snake, {0, 0});

    game.tick();

    CHECK_FALSE(game.alive());
}

TEST_CASE("the snake may chase its own tail") {
    // a closed ring: the head moves into the cell the tail is leaving
    Snake snake({{5, 5}, {6, 5}, {6, 4}, {5, 4}}, Direction::Down);
    Game game(10, 10, snake, {0, 0});

    game.tick();

    CHECK(game.alive());
    CHECK(game.snake().head() == Vec2i{5, 4});
}

TEST_CASE("filling the board wins") {
    Snake snake({{0, 0}}, Direction::Right);
    Game game(2, 1, snake, {1, 0});

    game.tick();

    CHECK(game.won());
    CHECK(game.alive());
}

TEST_CASE("the same seed plays the same game") {
    Game a(20, 20, 7);
    Game b(20, 20, 7);
    const Direction moves[] = {Direction::Up, Direction::Left, Direction::Down, Direction::Right};

    for (int i = 0; i < 40; ++i) {
        a.steer(moves[i / 5 % 4]);
        b.steer(moves[i / 5 % 4]);
        a.tick();
        b.tick();
        REQUIRE(a.snake().body() == b.snake().body());
        REQUIRE(a.apple() == b.apple());
    }
}
