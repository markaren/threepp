// Canvas manual placement — the pre-window half. A requested position surfaces
// through position(), setPosition() before the window exists records the
// request rather than touching GLFW, and the default is {0, 0}. The canvas is
// headless and never animated, so no window is ever created and no GPU or
// display is needed (headless picks the GLFW Null platform on a display-less
// machine). The windowed half — hints vs move-then-reveal — depends on the
// platform's window manager and is not assertable cross-platform.

#include <catch2/catch_test_macros.hpp>

#include "threepp/canvas/Canvas.hpp"

using namespace threepp;

TEST_CASE("requested position is visible before the window exists") {
    Canvas canvas(Canvas::Parameters().title("CanvasPosition").size(64, 64).headless(true).position(120, 240));
    CHECK(canvas.position() == std::pair<int, int>{120, 240});
}

TEST_CASE("position defaults to {0, 0} when none was requested") {
    Canvas canvas(Canvas::Parameters().title("CanvasPosition").size(64, 64).headless(true));
    CHECK(canvas.position() == std::pair<int, int>{0, 0});
}

TEST_CASE("setPosition before the window exists updates the request") {
    Canvas canvas(Canvas::Parameters().title("CanvasPosition").size(64, 64).headless(true));
    canvas.setPosition({30, 40});
    CHECK(canvas.position() == std::pair<int, int>{30, 40});
}
