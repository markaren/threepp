// Canvas manual placement — the pre-window half. A requested position surfaces
// through position(), setPosition() before the window exists records the
// request rather than touching GLFW, and the default is {0, 0}. The canvas is
// headless and never animated, so no window is ever created and no GPU or
// display is needed (headless picks the GLFW Null platform on a display-less
// machine, which needs GLFW 3.4 — see canvasCanExist below). The windowed half — hints vs move-then-reveal — depends on the
// platform's window manager and is not assertable cross-platform.

#include <catch2/catch_test_macros.hpp>

#include "threepp/canvas/Canvas.hpp"

#include <cstdlib>

using namespace threepp;

#ifndef THREEPP_GLFW_HAS_NULL_PLATFORM
#define THREEPP_GLFW_HAS_NULL_PLATFORM 1
#endif

namespace {

    // Whether a canvas — any canvas — can exist here at all. Windows and macOS
    // always have a window system. On Linux a headless canvas falls back to the
    // GLFW Null platform when there is no display server, but only from GLFW
    // 3.4: link an older one (Ubuntu 22.04's libglfw3-dev is 3.3.x, which is
    // what CI's ASan job builds against) and glfwInit fails on X11 with
    // "The DISPLAY environment variable is missing". Skipping there reports the
    // truth — the placement code below was never reached — instead of a defect.
    bool canvasCanExist() {
#if defined(_WIN32) || defined(__APPLE__) || THREEPP_GLFW_HAS_NULL_PLATFORM
        return true;
#else
        const char* x11 = std::getenv("DISPLAY");
        const char* wl = std::getenv("WAYLAND_DISPLAY");
        return (x11 && *x11) || (wl && *wl);
#endif
    }

    constexpr auto kNoPlatform =
            "no display server, and this GLFW predates the Null platform (needs 3.4)";

}// namespace

TEST_CASE("requested position is visible before the window exists") {
    if (!canvasCanExist()) SKIP(kNoPlatform);
    Canvas canvas(Canvas::Parameters().title("CanvasPosition").size(64, 64).headless(true).position(120, 240));
    CHECK(canvas.position() == std::pair<int, int>{120, 240});
}

TEST_CASE("position defaults to {0, 0} when none was requested") {
    if (!canvasCanExist()) SKIP(kNoPlatform);
    Canvas canvas(Canvas::Parameters().title("CanvasPosition").size(64, 64).headless(true));
    CHECK(canvas.position() == std::pair<int, int>{0, 0});
}

TEST_CASE("setPosition before the window exists updates the request") {
    if (!canvasCanExist()) SKIP(kNoPlatform);
    Canvas canvas(Canvas::Parameters().title("CanvasPosition").size(64, 64).headless(true));
    canvas.setPosition({30, 40});
    CHECK(canvas.position() == std::pair<int, int>{30, 40});
}
