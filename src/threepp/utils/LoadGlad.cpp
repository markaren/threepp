

#include "threepp/utils/LoadGlad.hpp"

#include <iostream>
#include <stdexcept>


namespace {

    // Which loader filled in GLAD's pointers. They are process-wide and
    // first-load-wins, so a second call is normally a harmless no-op — the
    // GLRenderer constructor calls loadGlad() right after Canvas::initWindow
    // has already loaded through the context's own getter. It is NOT harmless
    // when the second call would have used a different loader: an EGL context
    // whose entry points were resolved through GLX (or vice versa) gives
    // pointers into the wrong dispatch, which crashes or silently misbehaves
    // rather than failing. That case is worth a word on stderr.
    enum class Loader { none,
                        builtin,
                        supplied };

    Loader& gladLoadedBy() {

        static Loader by = Loader::none;
        return by;
    }

    // Throw rather than exit(): this runs inside a library that is embedded in
    // other hosts (the Python module), where a bare exit kills the interpreter
    // and takes the traceback with it.
    [[noreturn]] void throwLoadFailure(const char* how) {

        throw std::runtime_error(
                std::string("Failed to initialize GLAD via ") + how +
                " - no OpenGL context is current, or the loader cannot reach it. "
                "Create the GLRenderer's Canvas before switching the process to another graphics API.");
    }
}// namespace

bool threepp::gladLoaded() {

    return gladLoadedBy() != Loader::none;
}

void threepp::loadGlad() {

    // Silent when something already loaded them: GLRenderer's size-only
    // constructor guards with gladLoaded(), and a caller that does not is
    // simply asking for the pointers to exist, which they do.
    if (gladLoadedBy() != Loader::none) return;

    if (!gladLoadGL()) {
        throwLoadFailure("gladLoadGL");
    }
    gladLoadedBy() = Loader::builtin;
}

void threepp::loadGlad(GLADloadproc procAddress) {

    if (gladLoadedBy() == Loader::builtin) {
        std::cerr << "threepp: GLAD was already loaded through gladLoadGL() (the GLX dispatch); "
                     "this context's own loader is being ignored, and its entry points may "
                     "belong to a different context. Use one GL context per process."
                  << std::endl;
        return;
    }
    if (gladLoadedBy() != Loader::none) return;

    if (!procAddress) {
        throwLoadFailure("a null loader");
    }
    if (!gladLoadGLLoader(procAddress)) {
        throwLoadFailure("the supplied loader");
    }
    gladLoadedBy() = Loader::supplied;
}
