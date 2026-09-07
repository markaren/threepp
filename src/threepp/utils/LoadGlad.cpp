

#include "threepp/utils/LoadGlad.hpp"

#include <stdexcept>


void threepp::loadGlad() {

    static bool gladInitialized = false;

    if (!gladInitialized) {
        // Throw rather than exit(): this runs inside a library that is embedded
        // in other hosts (the Python module), where a bare exit kills the
        // interpreter and takes the traceback with it.
        if (!gladLoadGL()) {
            throw std::runtime_error(
                    "Failed to initialize GLAD - no OpenGL context is current. "
                    "Create the GLRenderer's Canvas before switching the process to another graphics API.");
        }
        gladInitialized = true;
    }
}
