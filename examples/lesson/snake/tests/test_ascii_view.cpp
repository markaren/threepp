// test_ascii_view.cpp: a view is testable too, when it's a function of the game.

#include "view/AsciiView.hpp"

#include <catch2/catch_test_macros.hpp>

#include <sstream>

using namespace snake;

TEST_CASE("the board as text, top row first") {
    Snake snake({{1, 1}, {0, 1}}, Direction::Right);
    Game game(4, 3, snake, {3, 2});

    CHECK(AsciiView::draw(game) ==
          "######\n"
          "#   @#\n"
          "#oO  #\n"
          "#    #\n"
          "######  score 0\n");
}

TEST_CASE("a dead snake's head is an X") {
    Snake snake({{3, 1}, {2, 1}}, Direction::Right);
    Game game(4, 3, snake, {0, 0});

    game.tick();// into the wall

    CHECK(AsciiView::draw(game).find('X') != std::string::npos);
    CHECK(AsciiView::draw(game).find('O') == std::string::npos);
}

TEST_CASE("the view prints once per tick, not once per frame") {
    Game game(10, 10, 1);
    std::ostringstream out;
    AsciiView view(out);

    view.update(game);
    view.update(game);
    const auto once = out.str().size();

    game.tick();
    view.update(game);

    CHECK(once == AsciiView::draw(game).size());
    CHECK(out.str().size() == 2 * once);
}
